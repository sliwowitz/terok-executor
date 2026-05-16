# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Shell wrapper generation for agent CLI commands.

Produces per-provider bash functions (``claude()``, ``codex()``, ``vibe()``,
etc.) that set git identity, handle session resume, and support both
interactive and headless (``--terok-timeout``) modes.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .providers import AGENT_PROVIDERS, AgentProvider

if TYPE_CHECKING:
    from collections.abc import Callable


INITIAL_PROMPT_PATH = "/home/dev/.terok/initial-prompt.txt"
"""Container path of the per-task initial prompt the TUI/CLI writes at launch."""

INITIAL_PROMPT_CONSUMED_PATH = "/home/dev/.terok/initial-prompt.consumed.txt"
"""Where the prompt file is moved after an agent picks it up (one-shot semantics)."""

INSTRUCTIONS_PATH = "/home/dev/.terok/instructions.md"
"""Container path of the resolved terok per-task system instructions."""

CONTAINER_WORKSPACE = "/workspace"
"""Container path the host-side repo is bind-mounted at (see container/env.py)."""


@dataclass(frozen=True)
class WrapperConfig:
    """Groups parameters for generating the Claude shell wrapper."""

    has_agents: bool
    has_instructions: bool = False


def generate_agent_wrapper(
    provider: AgentProvider,
    has_agents: bool,
    *,
    claude_wrapper_fn: Callable[[WrapperConfig], str] | None = None,
) -> str:
    """Generate the shell wrapper function content for a single provider.

    For Claude, uses *claude_wrapper_fn* (which should be
    ``agents._generate_claude_wrapper``) to produce the full wrapper with
    ``--add-dir /``, ``--agents``, and session resume support.  The function
    is passed in by the caller to avoid a circular import between this module
    and ``agents``.

    For other providers, produces a simpler wrapper that sets git env vars
    and delegates to the binary.  Instructions are delivered via
    ``opencode.json`` (OpenCode/Blablador), ``model_instructions_file``
    (Codex), or ``--append-system-prompt`` (Claude) -- not via the wrapper.

    Args:
        claude_wrapper_fn: ``(cfg: WrapperConfig) -> str``.
            Required when ``provider.name == "claude"``.

    See also [`generate_all_wrappers`][terok_executor.provider.wrappers.generate_all_wrappers] which produces wrappers for every
    registered provider in one file.
    """
    if provider.name == "claude":
        if claude_wrapper_fn is None:
            raise ValueError("claude_wrapper_fn is required for Claude provider")
        return claude_wrapper_fn(WrapperConfig(has_agents=has_agents))

    return _generate_generic_wrapper(provider)


def generate_all_wrappers(
    has_agents: bool,
    *,
    claude_wrapper_fn: Callable[[WrapperConfig], str] | None = None,
) -> str:
    """Generate shell wrappers for **all** registered providers in one file.

    The output file contains a shell function per provider (``claude()``,
    ``codex()``, ``vibe()``, etc.), each with correct git env vars, timeout
    support, and session resume logic.  This allows interactive CLI users to
    invoke any agent regardless of which provider was configured as default.

    A shared ``_terok_resume_or_fresh`` helper is emitted at the top of the
    file for stale-session fallback (see `_RESUME_FALLBACK_FN`).

    Args:
        claude_wrapper_fn: Required -- produces the Claude wrapper.
    """
    sections: list[str] = [_RESUME_FALLBACK_FN, _TRUST_WORKSPACE_FN]
    for provider in AGENT_PROVIDERS.values():
        section = generate_agent_wrapper(
            provider,
            has_agents,
            claude_wrapper_fn=claude_wrapper_fn,
        )
        sections.append(section)
    return "\n".join(sections)


# -- Private helpers ----------------------------------------------------------


def refuse_subcommands_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines that refuse credential-handling subcommands.

    For agents whose YAML declares ``wrapper.refuse_subcommands``, the
    in-container wrapper short-circuits ``login``/``logout``/``setup-token``
    with a friendly error pointing at ``terok auth``.  Best-effort UX only:
    the egress firewall (terok-ai/terok#873) is the actual enforcement —
    a user can still call the binary directly to bypass this guard.

    Empty list when the provider has no refuse list — keeps the rendered
    wrapper a no-op for non-credentialed agents.
    """
    if not provider.refuse_subcommands:
        return []
    pattern = "|".join(provider.refuse_subcommands)
    return [
        '    case "${1:-}" in',
        f"        {pattern})",
        '            echo "Login is unavailable inside a terok task." >&2',
        f"            echo \"Run 'terok auth {provider.name}' on the host instead.\" >&2",
        "            return 1 ;;",
        "    esac",
    ]


def _auto_approve_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for auto-approve flag injection (Codex only today)."""
    if not provider.auto_approve_flags:
        return []
    lines = ["    local _approve_args=()"]
    lines.append('    if [[ "${TEROK_UNRESTRICTED:-}" == "1" ]]; then')
    for flag in provider.auto_approve_flags:
        lines.append(f"        _approve_args+=({shlex.quote(flag)})")
    lines.append("    fi")
    return lines


def _opencode_plugin_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for OpenCode session plugin symlink setup."""
    if not (provider.session_file and provider.uses_opencode_instructions):
        return []
    if provider.opencode_config is not None:
        plugin_dir = f"$HOME/{provider.opencode_config.config_dir}/opencode/plugins"
    else:
        plugin_dir = "$HOME/.config/opencode/plugins"
    return [
        "    # Ensure OpenCode session plugin is installed",
        "    local _plugin_src=/usr/local/share/terok/opencode-session-plugin.mjs",
        f"    local _plugin_dir={plugin_dir}",
        '    if [ -f "$_plugin_src" ]; then',
        '        mkdir -p "$_plugin_dir"',
        '        ln -sf "$_plugin_src" "$_plugin_dir/terok-session.mjs"',
        "    fi",
    ]


def initial_prompt_block(session_path: str | None) -> list[str]:
    """Emit bash lines that pick up the task's initial prompt as the first message.

    Fires only on a bare interactive launch (no user args, no headless timeout)
    when no session is being resumed.  The file is renamed after consumption so
    a subsequent agent invocation falls through to ``--resume`` (or starts
    fresh) instead of replaying the same prompt.

    The renamed copy is preserved as a paper trail; the user can ``mv`` it
    back to recover from a launch that crashed before the session was saved.
    """
    guards = ['[ -z "$_timeout" ]', "[ $# -eq 0 ]"]
    if session_path:
        guards.append(f"[ ! -s {session_path} ]")
    guards.append(f"[ -s {INITIAL_PROMPT_PATH} ]")
    return [
        "    # Pick up the task's initial prompt as the first message (one-shot).",
        f"    if {' && '.join(guards)}; then",
        f'        set -- "$(cat {INITIAL_PROMPT_PATH})"',
        f"        mv {INITIAL_PROMPT_PATH} {INITIAL_PROMPT_CONSUMED_PATH}",
        "    fi",
    ]


def _session_resume_block(provider: AgentProvider, session_path: str | None) -> list[str]:
    """Emit bash lines for session resume arg injection."""
    if not (session_path and provider.resume_flag):
        return []
    return [
        "    local _resume_args=()",
        f"    if [ -s {session_path} ] && \\",
        '       { [ -n "$_timeout" ] || [ $# -eq 0 ]; }; then',
        f'        _resume_args+=({provider.resume_flag} "$(cat {session_path})")',
        "    fi",
    ]


def _codex_instr_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for Codex model_instructions_file injection."""
    if provider.name != "codex":
        return []
    return [
        "    local _instr_args=()",
        "    [ -f /home/dev/.terok/instructions.md ] && \\",
        "        _instr_args+=(-c 'model_instructions_file=\"/home/dev/.terok/instructions.md\"')",
    ]


def _vibe_model_sync_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines for lazy Mistral model sync before vibe runs.

    Uses a pure-bash mtime check to avoid Python startup overhead when
    the cache is fresh (<24h).  Only invoked from the vibe() wrapper,
    keeping login shells fast.
    """
    if provider.name != "vibe":
        return []
    return [
        "    # Lazy Mistral model sync (pure-bash mtime check, avoids Python startup)",
        "    if command -v vibe-model-sync >/dev/null 2>&1; then",
        "        local _mc=$HOME/.vibe/mistral-models.txt",
        '        if [ ! -f "$_mc" ] || [ -n "$(find "$_mc" -mmin +1440 2>/dev/null)" ]; then',
        "            vibe-model-sync >/dev/null 2>&1 || true",
        "        fi",
        "    fi",
    ]


def _vibe_subshell_setup_block(provider: AgentProvider) -> list[str]:
    """Emit bash lines that prep Vibe's per-task settings inside each subshell.

    Mistral Vibe exposes no theme/instructions override on the CLI itself —
    every knob terok cares about flows through env vars consumed by
    ``vibe.core.config._settings.VibeConfig`` (pydantic-settings,
    ``env_prefix="VIBE_"``, ``case_sensitive=False``).  This block:

    1. **Forces yolo mode** by exporting ``VIBE_BYPASS_TOOL_PERMISSIONS=true``.
       Maps to the model's ``bypass_tool_permissions`` field at
       ``vibe/core/config/_settings.py:510``; both the CLI and the ACP
       loop short-circuit the approval callback when it's True.
       Unconditional here (the ``unrestricted`` mode in
       ``container/env.py:339-341`` only fires for autopilot tasks, so
       interactive ``vibe`` invocations would otherwise nag for every
       tool use).

    2. **Injects per-task instructions** by writing
       ``/home/dev/.terok/instructions.md`` to
       ``$HOME/.vibe/prompts/terok-task-<TASK_ID>.md`` and exporting
       ``VIBE_SYSTEM_PROMPT_ID=terok-task-<TASK_ID>``.  Resolution path:
       ``VibeConfig.system_prompt`` property
       (``vibe/core/config/_settings.py:703``) walks the trust-free
       ``user_prompts_dirs`` (``$VIBE_HOME/prompts``,
       ``vibe/core/config/harness_files/_harness_manager.py:120-125``).
       Trap on subshell ``EXIT`` removes the per-task file so the
       shared ``~/.vibe`` mount doesn't accumulate cruft from finished
       tasks.

    3. **Trusts ``/workspace``** by appending it to
       ``$HOME/.vibe/trusted_folders.toml`` (idempotent, ``flock``-guarded).
       Without this, ``HarnessFilesManager.load_project_docs`` returns
       early (workdir untrusted) and the project's ``AGENTS.md`` chain
       never composes into the prompt
       (``vibe/core/config/harness_files/_harness_manager.py:196``).

    Bash function variables are declared ``local`` so they don't leak
    into the user's shell when the function returns.

    Empty list for non-vibe providers — keeps the rendered wrapper a
    no-op everywhere else.
    """
    if provider.name != "vibe":
        return []
    return [
        "    # ── Vibe: yolo + per-task instructions + workspace trust ──",
        "    export VIBE_BYPASS_TOOL_PERMISSIONS=true",
        '    if [ -f "' + INSTRUCTIONS_PATH + '" ] && [ -n "${TASK_ID:-}" ]; then',
        '        local _vibe_prompt_id="terok-task-${TASK_ID}"',
        '        local _vibe_prompts_dir="${HOME}/.vibe/prompts"',
        '        local _vibe_prompt_file="${_vibe_prompts_dir}/${_vibe_prompt_id}.md"',
        '        mkdir -p "${_vibe_prompts_dir}"',
        '        cp "' + INSTRUCTIONS_PATH + '" "${_vibe_prompt_file}"',
        '        export VIBE_SYSTEM_PROMPT_ID="${_vibe_prompt_id}"',
        "        trap 'rm -f \"${_vibe_prompt_file}\"' EXIT",
        "    fi",
        '    _terok_trust_workspace_for_vibe "' + CONTAINER_WORKSPACE + '"',
    ]


_TRUST_WORKSPACE_FN = """\
# Idempotently mark a container path as trusted in Vibe.
#
# Vibe's trust system lives at $VIBE_HOME/trusted_folders.toml as
# ``trusted = [...]`` + ``untrusted = [...]``.  The CLI has a --trust
# flag; the ACP entrypoint has no equivalent.  Without trust, the
# project's AGENTS.md chain isn't composed into the system prompt
# (vibe/core/config/harness_files/_harness_manager.py:196 returns early
# when ``trusted_workdir`` is None).
#
# ~/.vibe is shared across every task container, so the write is
# guarded with ``flock`` to avoid TOML corruption when two tasks land
# at the same moment.  ``/workspace`` is a container-only path that
# never resolves on the host, so persisting it in shared state is
# benign — it only affects future containers that mount the same
# /workspace, which is exactly what we want.
#
# The TOML merge itself lives in
# ``/usr/local/share/terok/terok-trust-workspace.py`` (installed by the
# L1 Dockerfile); the same script is invoked by the ACP wrapper
# (``terok-vibe-acp``) so future Vibe schema changes land in one place.
_terok_trust_workspace_for_vibe() {
    local _path="$1"
    local _tf="${HOME}/.vibe/trusted_folders.toml"
    local _merge=/usr/local/share/terok/terok-trust-workspace.py
    [ -x "${_merge}" ] || return 0
    mkdir -p "$(dirname "${_tf}")"
    (
        flock -x 200
        python3 "${_merge}" "${_path}" "${_tf}"
    ) 200>"${_tf}.lock"
}
"""
"""Top-of-file helper Vibe's per-subshell setup calls into.  Lives at module
scope (not inside the wrapper function) so it's defined exactly once even
though the wrapper is per-mode (headless / interactive).  The TOML merge
itself lives in ``resources/scripts/terok-trust-workspace.py`` and is
deployed to ``/usr/local/share/terok/`` by the L1 Dockerfile template — the
ACP wrapper invokes the same script, so future Vibe schema changes live in
one place."""


def _vibe_capture_fn(provider: AgentProvider, session_path: str | None) -> list[str]:
    """Emit bash lines for Vibe post-run session capture helper."""
    if not (provider.name == "vibe" and session_path):
        return []
    return [
        "    _terok_capture_vibe_session() {",
        '        python3 -c "',
        "import json, os, glob",
        "files = sorted(glob.glob(os.path.expanduser('~/.vibe/logs/session/session_*/meta.json')),",
        "               key=os.path.getmtime, reverse=True)",
        "if files:",
        "    with open(files[0]) as f:",
        "        sid = json.load(f).get('session_id', '')",
        "    if sid:",
        "        print(sid)",
        f'" > {session_path} 2>/dev/null || true',
        "    }",
    ]


def _extra_args_expansion(provider: AgentProvider, session_path: str | None) -> str:
    """Build the extra-args shell expansions between the binary and ``"$@"``."""
    parts: list[str] = []
    if provider.auto_approve_flags:
        parts.append('"${_approve_args[@]}"')
    if session_path and provider.resume_flag:
        parts.append('"${_resume_args[@]}"')
    if provider.name == "codex":
        parts.append('"${_instr_args[@]}"')
    return (" " + " ".join(parts)) if parts else ""


def _wrap_invocation(cmd: str, provider: AgentProvider, session_path: str | None) -> str:
    """Wrap a shell invocation with the stale-session fallback when resume is active."""
    if session_path and provider.resume_flag:
        return f"_terok_resume_or_fresh {session_path} {provider.resume_flag} {cmd}"
    return cmd


_RESUME_FALLBACK_FN = """\
# WORKAROUND: stale-session guard (timing-based heuristic).
#
# When a user starts an agent, exits immediately (no real interaction),
# and re-runs, the captured session ID points to a conversation that was
# never persisted.  The agent then fails with "No conversation found".
#
# This is a best-effort mitigation, not a proper fix: we assume that
# any non-zero exit within 2 seconds of launch is a stale-session error
# and retry without --resume.  This heuristic can misfire (e.g. a fast
# config error would also trigger a retry), but the retry is harmless —
# it just runs without resume, which is the correct fallback anyway.
#
# A proper fix would validate the session ID against the agent's storage
# before injecting --resume, but that requires agent-specific probes
# that don't exist yet.
_terok_resume_or_fresh() {
    local _session_file="$1" _resume_flag="$2"; shift 2
    local _start; _start=$(date +%s)
    "$@"; local _rc=$?
    local _elapsed=$(( $(date +%s) - _start ))
    if [ $_rc -ne 0 ] && [ $_elapsed -lt 2 ] && [ -s "$_session_file" ]; then
        echo "terok: session not found (stale?), retrying without resume" >&2
        rm -f "$_session_file"
        local _retry=() _skip=false
        for _a in "$@"; do
            if $_skip; then _skip=false; continue; fi
            if [ "$_a" = "$_resume_flag" ]; then _skip=true; continue; fi
            _retry+=("$_a")
        done
        "${_retry[@]}"; _rc=$?
    fi
    return $_rc
}
"""


def _generate_generic_wrapper(provider: AgentProvider) -> str:
    """Generate a shell wrapper for non-Claude providers.

    Sets git identity env vars and wraps the binary with optional timeout
    support (``--terok-timeout``), matching the Claude wrapper's interface.

    Session resume logic (for providers with ``session_file``):

    - An OpenCode plugin (or post-run parse for Vibe) captures the session
      ID to ``/home/dev/.terok/<session_file>``.
    - Resume args (``--session <id>`` or ``--resume <id>``) are injected
      only in headless mode (``--terok-timeout`` present) or on bare
      interactive launch (no user args).
    - When the user passes their own arguments, passthrough is transparent
      -- no resume args are injected.
    """
    author_name = shlex.quote(provider.git_author_name)
    author_email = shlex.quote(provider.git_author_email)
    binary = provider.binary
    session_path = f"/home/dev/.terok/{provider.session_file}" if provider.session_file else None
    extra = _extra_args_expansion(provider, session_path)

    lines = [
        "# Generated by terok",
        f"{binary}() {{",
    ]
    lines.extend(refuse_subcommands_block(provider))
    lines.extend(
        [
            '    local _timeout=""',
            "    # Extract terok-specific flags (must come before agent flags)",
            "    while [[ $# -gt 0 ]]; do",
            '        case "$1" in',
            '            --terok-timeout) _timeout="$2"; shift 2 ;;',
            "            *) break ;;",
            "        esac",
            "    done",
            "    [ -r /usr/local/share/terok/terok-env-git-identity.sh ] && \\",
            "        . /usr/local/share/terok/terok-env-git-identity.sh",
        ]
    )

    lines.extend(_vibe_model_sync_block(provider))
    lines.extend(_auto_approve_block(provider))
    lines.extend(_opencode_plugin_block(provider))
    lines.extend(_session_resume_block(provider, session_path))
    lines.extend(_codex_instr_block(provider))
    lines.extend(initial_prompt_block(session_path))
    lines.extend(_vibe_capture_fn(provider, session_path))

    headless_cmd = _wrap_invocation(
        f'timeout "$_timeout" {binary}{extra} "$@"', provider, session_path
    )
    interactive_cmd = _wrap_invocation(f'command {binary}{extra} "$@"', provider, session_path)

    # Vibe-only subshell setup: forced yolo env, per-task system-prompt
    # injection (with EXIT trap cleanup), and workspace trust.  Empty
    # list for every other provider, so the rendering is identical to
    # the pre-#47 generator for non-vibe wrappers.  Re-indent from the
    # block's native 4-space depth to the subshell-body 12-space depth.
    vibe_setup = ["            " + line[4:] for line in _vibe_subshell_setup_block(provider)]

    # Headless mode (with timeout)
    lines.append('    if [ -n "$_timeout" ]; then')
    lines.append("        (")
    lines.append(f"            _terok_apply_git_identity {author_name} {author_email}")
    lines.extend(vibe_setup)
    if session_path:
        lines.append(f"            export TEROK_SESSION_FILE={session_path}")
    lines.append(f"        {headless_cmd}")
    if provider.name == "vibe" and session_path:
        lines.append("        local _rc=$?; _terok_capture_vibe_session; return $_rc")
    lines.append("        )")

    # Interactive mode (no timeout)
    lines.append("    else")
    lines.append("        (")
    lines.append(f"            _terok_apply_git_identity {author_name} {author_email}")
    lines.extend(vibe_setup)
    if session_path:
        lines.append(f"            export TEROK_SESSION_FILE={session_path}")
    lines.append(f"        {interactive_cmd}")
    if provider.name == "vibe" and session_path:
        lines.append("        local _rc=$?; _terok_capture_vibe_session; return $_rc")
    lines.append("        )")
    lines.append("    fi")
    lines.append("}")

    return "\n".join(lines) + "\n"
