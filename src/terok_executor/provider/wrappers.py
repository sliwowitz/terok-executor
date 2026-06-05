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

from .providers import AGENT_PROVIDERS, AgentProvider

INITIAL_PROMPT_PATH = "/home/dev/.terok/initial-prompt.txt"
"""Container path of the per-task initial prompt the TUI/CLI writes at launch."""

INITIAL_PROMPT_CONSUMED_PATH = "/home/dev/.terok/initial-prompt.consumed.txt"
"""Where the prompt file is moved after an agent picks it up (one-shot semantics)."""

INSTRUCTIONS_PATH = "/home/dev/.terok/instructions.md"
"""Container path of the resolved terok per-task system instructions."""

CONTAINER_WORKSPACE = "/workspace"
"""Container path the host-side repo is bind-mounted at (see container/env.py)."""

_TEMPLATE_NAME = "agent-wrappers.sh.j2"


# ── Public API ──────────────────────────────────────────────────────────────


def generate_all_wrappers(has_agents: bool) -> str:
    """Render ``terok-executor.sh``: a shell wrapper function for every provider.

    The output file contains a shell function per provider (``claude()``,
    ``codex()``, ``vibe()``, …), each with correct git env vars, timeout
    support, and session-resume logic, plus the two shared helper functions
    they call.  This lets interactive CLI users invoke any agent regardless of
    which provider was configured as default.

    Args:
        has_agents: Whether an ``agents.json`` was written — adds Claude's
            ``--agents`` flag to the ``claude()`` wrapper.
    """
    providers = [_wrapper_context(p) for p in AGENT_PROVIDERS.values()]
    return _env().from_string(_template_source()).render(providers=providers, has_agents=has_agents)


def generate_agent_wrapper(provider: AgentProvider, has_agents: bool = False) -> str:
    """Render a single provider's wrapper function, without the shared helpers.

    Used to inspect one provider's wrapper in isolation; the full file (with
    the ``_terok_resume_or_fresh`` / ``_terok_trust_workspace_for_vibe``
    helpers) is produced by
    [`generate_all_wrappers`][terok_executor.provider.wrappers.generate_all_wrappers].
    """
    ctx = _wrapper_context(provider)
    # Jinja resolves macros dynamically, so the template module's macro
    # attributes (claude_wrapper / generic_wrapper) are not statically known.
    macros: Any = _env().from_string(_template_source()).module
    if ctx["is_claude"]:
        return str(macros.claude_wrapper(ctx, has_agents))
    return str(macros.generic_wrapper(ctx))


# ── Per-provider data preparation ───────────────────────────────────────────


def _wrapper_context(provider: AgentProvider) -> dict[str, object]:
    """Prepare the data the template renders into one provider's shell wrapper.

    Every shell-significant value is resolved here so the template stays a pure
    layout concern: identities are shell-quoted, the session path is resolved,
    and the headless/interactive command strings are assembled (including the
    stale-session resume guard and the extra-args expansion).
    """
    session_path = f"/home/dev/.terok/{provider.session_file}" if provider.session_file else ""
    binary = provider.binary
    extra = _extra_args_expansion(provider, session_path)
    wrap = (
        f"_terok_resume_or_fresh {session_path} {provider.resume_flag} "
        if session_path and provider.resume_flag
        else ""
    )
    return {
        "name": provider.name,
        "binary": binary,
        "is_claude": provider.name == "claude",
        "is_vibe": provider.name == "vibe",
        "is_codex": provider.name == "codex",
        "author_name": shlex.quote(provider.git_author_name),
        "author_email": shlex.quote(provider.git_author_email),
        "refuse_pattern": "|".join(provider.refuse_subcommands),
        "auto_approve_flags": list(provider.auto_approve_flags),
        "opencode_plugin_dir": _opencode_plugin_dir(provider),
        "session_path": session_path,
        "resume_flag": provider.resume_flag or "",
        "seed_prefix": _seed_prefix(provider),
        "headless_cmd": f'{wrap}timeout "$_timeout" {binary}{extra} "$@"',
        "interactive_cmd": f'{wrap}command {binary}{extra} "$@"',
    }


def _extra_args_expansion(provider: AgentProvider, session_path: str) -> str:
    """Build the extra-args shell expansions placed between the binary and ``"$@"``."""
    parts: list[str] = []
    if provider.auto_approve_flags:
        parts.append('"${_approve_args[@]}"')
    if session_path and provider.resume_flag:
        parts.append('"${_resume_args[@]}"')
    if provider.name == "codex":
        parts.append('"${_instr_args[@]}"')
    return (" " + " ".join(parts)) if parts else ""


def _opencode_plugin_dir(provider: AgentProvider) -> str:
    """Return the OpenCode session-plugin directory, or ``""`` when not applicable."""
    if not (provider.session_file and provider.uses_opencode_instructions):
        return ""
    if provider.opencode_config is not None:
        return f"$HOME/{provider.opencode_config.config_dir}/opencode/plugins"
    return "$HOME/.config/opencode/plugins"


def _seed_prefix(provider: AgentProvider) -> str:
    """Return the shell-quoted argv prefix used when seeding the initial prompt.

    Most CLIs accept a bare positional string as the first message; OpenCode
    routes through its ``run`` subcommand and Copilot through ``-p`` so the
    text is interpreted as a prompt rather than a path or unrelated argument.
    """
    if provider.uses_opencode_instructions:
        tokens = ["run"]
    elif provider.name == "copilot":
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
