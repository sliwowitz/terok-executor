# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Shell wrapper generation for agent CLI commands.

Produces per-provider bash functions (``claude()``, ``codex()``, ``vibe()``,
etc.) that record task-local agent history, set git identity, handle session
resume, and support both interactive and headless (``--terok-timeout``) modes.

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

CLAUDE_OAUTH_UNROUTED_ENV = "TEROK_CLAUDE_OAUTH_UNROUTED"
"""Env var that marks Claude's default endpoint as having no route for the stored Claude login.

terok sets it, to
[`CLAUDE_OAUTH_UNROUTED_ON`][terok_executor.provider.wrappers.CLAUDE_OAUTH_UNROUTED_ON],
in one case only: the vault holds a Claude OAuth login that terok's
configuration neither routes to Claude (``allow_oauth``) nor exposes to it
(``expose_oauth_token``), terok's "skipped" mode.  Never in the routed or the
exposed mode: there Claude's default endpoint reaches the provider, through the
vault or directly with the real token.

In the skipped mode terok removes Claude's own route to the provider but keeps
the login's ``TEROK_PROVIDER_<NAME>_*`` handle, on purpose: OpenCode, Pi and
Toad reach the provider through the vault with it.  The handle alone would
therefore pass Claude's readiness check, and Claude would start on its default
endpoint, go straight to the provider's hosts and meet the egress firewall's
bare refusal.  With the variable on, the ``claude`` wrapper and the
``terok-claude-acp`` launcher refuse a launch on that endpoint up front
instead, naming the cause and the remedies.  A provider selected in the shell
(``--provider`` or ``TEROK_PROVIDER``) that serves Claude's protocol still
routes the launch, Claude's default provider included: the wrapper sends
Claude through that provider's vault handle, not straight to its hosts.  The
ACP launcher honours no selection, so it refuses whenever the variable is on.
Only Claude's entry points read it; any other value, or none, leaves them as
they were."""

CLAUDE_OAUTH_UNROUTED_ON = "1"
"""The value that turns Claude's routing refusal on.

The other half of the
[`CLAUDE_OAUTH_UNROUTED_ENV`][terok_executor.provider.wrappers.CLAUDE_OAUTH_UNROUTED_ENV]
contract: terok sets the variable to this value rather than spelling it, and
the rendered check compares against it, so the two cannot drift apart.  The
check compares strings, so any other spelling, such as ``true``, leaves
Claude's entry points as they were."""

_TEMPLATE_NAME = "agent-wrappers.sh.j2"


# ── Public API ──────────────────────────────────────────────────────────────


def generate_all_wrappers() -> str:
    """Render ``terok-executor.sh``: a shell wrapper function for every agent.

    The output file contains a shell function per agent (``claude()``,
    ``codex()``, ``vibe()``, …), each with correct git env vars, timeout
    support, and session-resume logic, plus the shared helper functions they
    call.  This lets interactive CLI users invoke any agent regardless of
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
    the ``_terok_record_agent`` / ``_terok_wrapper_help`` / ``_terok_resume_hint`` /
    ``_terok_trust_workspace_for_vibe`` helpers) is produced by
    [`generate_all_wrappers`][terok_executor.provider.wrappers.generate_all_wrappers].
    Claude's render also carries ``_terok_claude_ready`` and
    ``_terok_claude_routed``, the default-provider pre-flight that belongs to
    it alone (its ACP launcher calls it too).
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
    the headless/interactive command strings are assembled (including the
    extra-args expansion), and the ``--help`` header lists exactly the terok
    flags this agent's wrapper acts on.
    """
    session_path = f"/home/dev/.terok/{agent.session_file}" if agent.session_file else ""
    binary = agent.binary
    is_claude = agent.name == "claude"
    extra = _extra_args_expansion(agent, session_path)
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
    # Claude's macro carries its own session file and provider routing rather
    # than the roster-driven ones, so it is flagged explicitly here.
    resumes = bool(agent.resume_flag or agent.resume_subcommand)
    has_session = is_claude or bool(session_path and resumes)
    # A wrapper routes a runtime-selected provider either through a launcher or,
    # for Claude, by exporting the provider's endpoint as plain environment.
    routes_provider = bool(provider_launcher) or is_claude
    return {
        "name": agent.name,
        "binary": binary,
        "is_claude": is_claude,
        "help_flags": _help_flags(routes_provider, has_session),
        "is_vibe": agent.name == "vibe",
        "is_codex": agent.name == "codex",
        "is_opencode": agent.name == "opencode",
        "provider_launcher": provider_launcher,
        **_claude_provider_context(agent),
        **_readiness_guard_context(agent, routes_provider),
        "author_name": shlex.quote(agent.git_author_name),
        "author_email": shlex.quote(agent.git_author_email),
        "refuse_pattern": "|".join(agent.refuse_subcommands),
        "auto_approve_flags": list(agent.auto_approve_flags),
        "opencode_plugin_dir": _opencode_plugin_dir(agent),
        "session_path": session_path,
        "resume_flag": agent.resume_flag or "",
        "resume_subcommand": agent.resume_subcommand or "",
        "headless_subcommand": agent.headless_subcommand or "",
        "seed_prefix": _seed_prefix(agent),
        "headless_cmd": f'timeout "$_timeout" {cmd}{extra} "$@"',
        "interactive_cmd": f'command {cmd}{extra} "$@"',
    }


def _help_flags(has_provider_flag: bool, has_session: bool) -> str:
    """Render the terok-flag lines the ``-h``/``--help`` header shows for one agent.

    Only flags this wrapper acts on are listed: ``--provider`` needs a
    provider launcher (or Claude's env routing), ``--terok-new-session`` a
    recorded session to skip (by flag or by subcommand); ``--terok-timeout``
    is universal.
    """
    rows = [("--terok-timeout SECS", "run without a terminal; stop the agent after SECS seconds")]
    if has_provider_flag:
        rows.append(
            (
                "--provider NAME",
                "route the agent through the authenticated provider NAME (run: providers)",
            )
        )
    if has_session:
        rows.append(("--terok-new-session", "start a new session; do not resume the recorded one"))
    return "\n".join(f"    {flag:<22} {text}" for flag, text in rows)


def _shortcut_context(name: str) -> dict[str, str]:
    """Prepare the data the template renders into one curated provider's alias.

    The alias delegates to the harness wrapper rather than the launcher symlink
    it shadows, so it inherits session resume, initial-prompt seeding,
    ``--terok-timeout`` and auto-approve — and keeps inheriting whatever that
    wrapper grows later, with nothing to duplicate per provider.

    Git authorship is handed over explicitly rather than derived inside the
    harness wrapper: a bare ``opencode --provider blablador`` stays attributed to
    the harness, while the pinned alias keeps the per-model attribution the ACP
    wrapper already uses.  The alias also owns a session file of its own
    (``<name>-session.txt``), so two aliases used in one task never resume each
    other's conversation.
    """
    return {
        "name": name,
        "target": OPENCODE_HARNESS,
        "author_name": shlex.quote(_display_name(name)),
        "author_email": shlex.quote(f"noreply@{name}.localhost"),
        "session_path": f"/home/dev/.terok/{name}-session.txt",
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


def _readiness_guard_context(agent: Agent, routes_provider: bool) -> dict[str, object]:
    """Resolve whether this wrapper pre-flights provider readiness, and against whom.

    A native, protocol-routed agent (claude, codex, vibe) launches on its
    default provider's path, trusting a vault-materialized credential to be
    there. When no provider serving its wire protocol was authenticated at
    container-creation time, that credential is absent and the agent fails
    opaquely. Codex dies with a parse error (``EOF while parsing a value at
    line 1 column 0``). Claude falls into its first-run login, and the egress
    firewall refuses that login: it reserves the provider's hosts for the vault.
    The user then sees a bare ``ConnectionRefused``
    ([terok-ai/terok#1273](https://github.com/terok-ai/terok/issues/1273)).
    The guard checks the effective provider's
    ``TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL>`` handle — the same startup snapshot
    ``hilfe`` and ``providers`` read — and refuses with an actionable message
    instead.

    Claude's guard also refuses a *selected* provider that holds no credential
    in the task, rather than let Claude's fallback swap in the default — that
    would send the session to a provider nobody picked.  Its refusal names the
    ``terok auth`` entry for that provider, hence *auth_entries* (see
    ``_provider_auth_entries`` below).

    A handle proves a stored login, not a route from Claude's default
    endpoint: terok keeps the handle of a Claude OAuth login it neither routes
    to Claude nor exposes to it, for the task's other agents, and flags the
    case in
    [`CLAUDE_OAUTH_UNROUTED_ENV`][terok_executor.provider.wrappers.CLAUDE_OAUTH_UNROUTED_ENV].
    Claude's default-endpoint check reads that flag too, by the name
    *oauth_unrouted_env* carries and against the value *oauth_unrouted_on*
    carries, so the shell spells neither half of the contract.

    Truthy only for agents whose wrapper routes a provider (*routes_provider*:
    through a launcher, or Claude's environment routing) and that carry a fixed
    protocol + default provider to check; harnesses (opencode, pi, no default)
    and bindingless CLIs (copilot) opt out and render no guard.
    """
    binding = agent.provider_binding
    default = binding.default if binding else None
    if not (routes_provider and default and agent.protocol):
        return {
            "readiness_guard": False,
            "default_provider": "",
            "auth_entries": [],
            "oauth_unrouted_env": "",
            "oauth_unrouted_on": "",
        }
    return {
        "readiness_guard": True,
        "default_provider": default,
        "auth_entries": _provider_auth_entries(),
        "oauth_unrouted_env": CLAUDE_OAUTH_UNROUTED_ENV,
        "oauth_unrouted_on": CLAUDE_OAUTH_UNROUTED_ON,
    }


def _provider_auth_entries() -> list[tuple[str, str]]:
    """Pair each agent-bound provider with the ``terok auth`` entry that authenticates it.

    The roster stores a bound provider's credential under the binding agent's
    own auth entry (``terok auth claude`` authenticates ``anthropic``); every
    other provider is authenticated under its own name.  A refusal naming
    ``terok auth anthropic`` would send the user to a command that does not
    exist, so the bound pairs are spelled out for the wrapper, which falls back
    to the provider's own name for the rest.  Read from the live registry, so a
    user override that re-binds an agent's default is followed.
    """
    pairs: list[tuple[str, str]] = []
    for agent in AGENTS.values():
        binding = agent.provider_binding
        if binding is not None and binding.default and binding.default != agent.name:
            pairs.append((binding.default, agent.name))
    return sorted(pairs)


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
