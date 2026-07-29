# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Shell wrapper generation for agent CLI commands.

Produces per-provider bash functions (``claude()``, ``codex()``, ``vibe()``,
etc.) that set git identity, handle session resume, and support both
interactive and headless (``--terok-timeout``) modes.

The shell itself lives in the Jinja template
``resources/templates/agent-wrappers.sh.j2``; this module only prepares the
per-provider data the template renders.  Keeping the shell in a template —
rather than assembling it from Python string fragments — lets the wrapper
logic be read as shell, with the vendor-specific blocks visible inline.
"""

from __future__ import annotations

import shlex
from functools import lru_cache
from importlib import resources
from typing import Any

from jinja2 import BaseLoader, Environment

from .providers import (
    AGENTS,
    LAUNCHER_ALWAYS,
    LAUNCHER_ON_PROVIDER_SELECT,
    OPENCODE_PROVIDERS,
    Agent,
)

INITIAL_PROMPT_PATH = "/home/dev/.terok/initial-prompt.txt"
"""Container path of the per-task initial prompt the TUI/CLI writes at launch."""

INITIAL_PROMPT_CONSUMED_PATH = "/home/dev/.terok/initial-prompt.consumed.txt"
"""Where the prompt file is moved after an agent picks it up (one-shot semantics)."""

INSTRUCTIONS_PATH = "/home/dev/.terok/instructions.md"
"""Container path of the resolved terok per-task system instructions."""

CONTAINER_WORKSPACE = "/workspace"
"""Container path the host-side repo is bind-mounted at (see container/env.py)."""

OPENCODE_HARNESS = "opencode"
"""Agent whose wrapper the curated OpenCode providers delegate to.

Every provider in
[`OPENCODE_PROVIDERS`][terok_executor.provider.providers.OPENCODE_PROVIDERS] is
driven by this harness: their one-word commands select a provider on it rather
than naming an agent of their own.
"""

IDENTITY_NAME_ENV = "TEROK_AGENT_IDENTITY_NAME"
"""Env var a provider alias sets to override the harness wrapper's git author name."""

IDENTITY_EMAIL_ENV = "TEROK_AGENT_IDENTITY_EMAIL"
"""Env var a provider alias sets to override the harness wrapper's git author email."""

_TEMPLATE_NAME = "agent-wrappers.sh.j2"


# ── Public API ──────────────────────────────────────────────────────────────


def generate_all_wrappers() -> str:
    """Render ``terok-executor.sh``: a shell wrapper function for every agent.

    The output file contains a shell function per agent (``claude()``,
    ``codex()``, ``vibe()``, …), each with correct git env vars, timeout
    support, and session-resume logic, plus the two shared helper functions
    they call.  This lets interactive CLI users invoke any agent regardless of
    which agent was configured as default.

    Each curated OpenCode provider additionally gets a one-word alias
    (``blablador()``, ``kisski()``, …) that delegates to the harness wrapper, so
    those commands inherit its feature set instead of reaching the launcher
    symlink directly and losing it.
    """
    agents = [_wrapper_context(a) for a in AGENTS.values()]
    shortcuts = [_shortcut_context(name) for name in OPENCODE_PROVIDERS]
    return _env().from_string(_template_source()).render(agents=agents, shortcuts=shortcuts)


def generate_agent_wrapper(agent: Agent) -> str:
    """Render a single agent's wrapper function, without the shared helpers.

    Used to inspect one agent's wrapper in isolation; the full file (with
    the ``_terok_resume_or_fresh`` / ``_terok_trust_workspace_for_vibe``
    helpers) is produced by
    [`generate_all_wrappers`][terok_executor.provider.wrappers.generate_all_wrappers].
    """
    ctx = _wrapper_context(agent)
    # Jinja resolves macros dynamically, so the template module's macro
    # attributes (claude_wrapper / generic_wrapper) are not statically known.
    macros: Any = _env().from_string(_template_source()).module
    if ctx["is_claude"]:
        return str(macros.claude_wrapper(ctx))
    return str(macros.generic_wrapper(ctx))


def generate_provider_shortcut(name: str) -> str:
    """Render one curated provider's one-word alias, without the shared helpers.

    Companion to
    [`generate_agent_wrapper`][terok_executor.provider.wrappers.generate_agent_wrapper]
    for inspecting a single alias; the full file is produced by
    [`generate_all_wrappers`][terok_executor.provider.wrappers.generate_all_wrappers].
    """
    macros: Any = _env().from_string(_template_source()).module
    return str(macros.provider_shortcut(_shortcut_context(name)))


# ── Per-agent data preparation ───────────────────────────────────────────────


def _wrapper_context(agent: Agent) -> dict[str, object]:
    """Prepare the data the template renders into one agent's shell wrapper.

    Every shell-significant value is resolved here so the template stays a pure
    layout concern: identities are shell-quoted, the session path is resolved,
    and the headless/interactive command strings are assembled (including the
    stale-session resume guard and the extra-args expansion).
    """
    session_path = f"/home/dev/.terok/{agent.session_file}" if agent.session_file else ""
    binary = agent.binary
    extra = _extra_args_expansion(agent, session_path)
    wrap = (
        f"_terok_resume_or_fresh {session_path} {agent.resume_flag} "
        if session_path and agent.resume_flag
        else ""
    )
    # The agent's launcher (declared in its YAML ``wrapper.launcher``) decides
    # how the command is built.  ``on_provider_select`` agents run through a
    # ``_runner`` array so a runtime selection can swap the bare binary for
    # ``<launcher> --provider NAME`` (which rewrites their config); the
    # ``provider_launcher`` value drives that array in the template.  ``always``
    # agents run their launcher on every invocation (it does per-run prep, e.g.
    # picker scoping + instruction injection).  No launcher → the bare binary.
    launcher = agent.launcher
    if launcher is not None and launcher.mode == LAUNCHER_ALWAYS:
        provider_launcher = ""
        cmd = launcher.script
    elif launcher is not None and launcher.mode == LAUNCHER_ON_PROVIDER_SELECT:
        provider_launcher = launcher.script
        cmd = '"${_runner[@]}"'
    else:
        provider_launcher = ""
        cmd = binary
    return {
        "name": agent.name,
        "binary": binary,
        "is_claude": agent.name == "claude",
        "is_vibe": agent.name == "vibe",
        "is_codex": agent.name == "codex",
        "is_opencode": agent.name == "opencode",
        "provider_launcher": provider_launcher,
        **_claude_provider_context(agent),
        **_readiness_guard_context(agent, provider_launcher),
        "author_name": shlex.quote(agent.git_author_name),
        "author_email": shlex.quote(agent.git_author_email),
        "refuse_pattern": "|".join(agent.refuse_subcommands),
        "auto_approve_flags": list(agent.auto_approve_flags),
        "opencode_plugin_dir": _opencode_plugin_dir(agent),
        "session_path": session_path,
        "resume_flag": agent.resume_flag or "",
        "seed_prefix": _seed_prefix(agent),
        "headless_cmd": f'{wrap}timeout "$_timeout" {cmd}{extra} "$@"',
        "interactive_cmd": f'{wrap}command {cmd}{extra} "$@"',
    }


def _shortcut_context(name: str) -> dict[str, str]:
    """Prepare the data the template renders into one curated provider's alias.

    The alias delegates to the harness wrapper rather than the launcher symlink
    it shadows, so it inherits session resume, initial-prompt seeding,
    ``--terok-timeout`` and auto-approve — and keeps inheriting whatever that
    wrapper grows later, with nothing to duplicate per provider.

    Git authorship is handed over explicitly rather than derived inside the
    harness wrapper: a bare ``opencode --provider blablador`` stays attributed to
    the harness, while the pinned alias keeps the per-model attribution the ACP
    wrapper already uses.
    """
    return {
        "name": name,
        "target": OPENCODE_HARNESS,
        "author_name": shlex.quote(_display_name(name)),
        "author_email": shlex.quote(f"noreply@{name}.localhost"),
    }


def _display_name(provider: str) -> str:
    """Capitalise a provider name for git authorship (``blablador`` → ``Blablador``).

    Mirrors ``${PROVIDER^}`` in ``opencode-provider-acp`` so a commit made from
    the CLI alias and one made through ACP carry the same author.
    """
    return provider[:1].upper() + provider[1:]


def _claude_provider_context(agent: Agent) -> dict[str, str]:
    """Resolve the env-var names Claude's wrapper uses to honor a selected provider.

    Claude takes a runtime provider override as plain environment: the wrapper
    points its base-URL and bearer vars at the selected provider's materialized
    ``TEROK_PROVIDER_<NAME>_*`` handles.  The var names come straight from
    Claude's roster entry so they cannot drift from its default routing.  Empty
    strings for agents without a binding, which leave the block unrendered.
    """
    binding = agent.provider_binding
    protocol = agent.protocol or ""
    bearer_env = ""
    base_url_env = ""
    if binding is not None:
        bearer_env = binding.token_env.get("oauth") or binding.token_env.get("_default") or ""
        base_url_env = binding.base_url_env
    return {
        "provider_protocol": protocol,
        "provider_protocol_var": protocol.upper().replace("-", "_"),
        "provider_base_url_env": base_url_env,
        "provider_bearer_env": bearer_env,
    }


def _readiness_guard_context(agent: Agent, provider_launcher: str) -> dict[str, object]:
    """Resolve whether this wrapper pre-flights provider readiness, and against whom.

    A native, protocol-routed agent (codex, vibe) launches its bare binary on
    the default path, trusting a vault-materialized credential to be there. When
    no provider serving its wire protocol was authenticated at container-creation
    time, that credential is absent and the agent dies with an opaque parse error
    (codex: ``EOF while parsing a value at line 1 column 0``). The guard checks
    the effective provider's ``TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL>`` handle —
    the same startup snapshot ``hilfe`` and ``providers`` read — and refuses with
    an actionable message instead.

    Truthy only for agents that both route through a launcher and carry a fixed
    protocol + default provider to check; harnesses (opencode, pi, no default)
    and bindingless CLIs (copilot) opt out and render no guard.
    """
    binding = agent.provider_binding
    default = binding.default if binding else None
    if not (provider_launcher and default and agent.protocol):
        return {"readiness_guard": False, "default_provider": ""}
    return {"readiness_guard": True, "default_provider": default}


def _extra_args_expansion(agent: Agent, session_path: str) -> str:
    """Build the extra-args shell expansions placed between the binary and ``"$@"``."""
    parts: list[str] = []
    if agent.auto_approve_flags:
        parts.append('"${_approve_args[@]}"')
    if session_path and agent.resume_flag:
        parts.append('"${_resume_args[@]}"')
    if agent.name == "codex":
        parts.append('"${_instr_args[@]}"')
    return (" " + " ".join(parts)) if parts else ""


def _opencode_plugin_dir(agent: Agent) -> str:
    """Return the OpenCode session-plugin directory, or ``""`` when not applicable.

    Only the ``opencode`` harness itself carries this — the curated
    OpenCode-driven providers run through its wrapper, not their own.
    """
    if not (agent.session_file and agent.uses_opencode_instructions):
        return ""
    return "$HOME/.config/opencode/plugins"


def _seed_prefix(agent: Agent) -> str:
    """Return the shell-quoted argv prefix used when seeding the initial prompt.

    Most CLIs accept a bare positional string as the first message; OpenCode
    routes through its ``run`` subcommand and Copilot through ``-p`` so the
    text is interpreted as a prompt rather than a path or unrelated argument.
    """
    if agent.uses_opencode_instructions:
        tokens = ["run"]
    elif agent.name == "copilot":
        tokens = ["-p"]
    else:
        tokens = []
    return "".join(f"{shlex.quote(t)} " for t in tokens)


# ── Jinja plumbing ──────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _env() -> Environment:
    """Shared Jinja environment with a shell-quote filter (shell output, not HTML)."""
    env = Environment(  # nosec B701 — shell output, not HTML
        loader=BaseLoader(), keep_trailing_newline=True, autoescape=False
    )
    env.filters["shquote"] = shlex.quote
    return env


@lru_cache(maxsize=1)
def _template_source() -> str:
    """Read the wrapper template from package resources."""
    return (
        resources.files("terok_executor") / "resources" / "templates" / _TEMPLATE_NAME
    ).read_text(encoding="utf-8")
