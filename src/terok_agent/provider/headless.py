# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Headless (autopilot) command construction and config resolution.

Provider definitions live in :mod:`providers`, shell wrapper generation
lives in :mod:`wrappers`.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from .providers import AgentProvider

# ── Dataclasses / types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved per-run config for a headless provider.

    Produced by :func:`apply_provider_config` after best-effort feature mapping.
    """

    model: str | None
    """Model override for providers that support it, else ``None``."""

    max_turns: int | None
    """Max turns for providers that support it, else ``None``."""

    timeout: int
    """Effective timeout in seconds."""

    prompt_extra: str
    """Extra text to append to the prompt (best-effort feature analogues)."""

    warnings: tuple[str, ...]
    """Warnings about unsupported features (for user display)."""


@dataclass(frozen=True)
class CLIOverrides:
    """CLI flag overrides for a headless agent run."""

    model: str | None = None
    """Explicit ``--model`` from CLI (takes precedence over config)."""

    max_turns: int | None = None
    """Explicit ``--max-turns`` from CLI."""

    timeout: int | None = None
    """Explicit ``--timeout`` from CLI."""

    instructions: str | None = None
    """Resolved instructions text. Delivery is provider-aware."""


# ── Public API ───────────────────────────────────────────────────────────────


def apply_provider_config(
    provider: AgentProvider,
    config: dict,
    overrides: CLIOverrides | None = None,
) -> ProviderConfig:
    """Resolve config values for a provider with best-effort feature mapping.

    CLI flag overrides take precedence over config values.  When the provider
    lacks a feature, an analogue is used where possible (e.g. injecting
    max-turns guidance into the prompt), and a warning is emitted for
    features that have no analogue.

    Args:
        config: Merged agent config dict (from :func:`resolve_agent_config`).
        overrides: CLI flag overrides (model, max_turns, timeout, instructions).
    """
    if overrides is None:
        overrides = CLIOverrides()
    from .config import resolve_provider_value

    warnings: list[str] = []
    prompt_parts: list[str] = []

    # --- Model ---
    cfg_model = resolve_provider_value("model", config, provider.name)
    model = overrides.model or (str(cfg_model) if cfg_model is not None else None)
    if model and not provider.model_flag:
        warnings.append(
            f"{provider.label} does not support model selection; ignoring model={model!r}"
        )
        model = None

    # --- Max turns ---
    cfg_turns = resolve_provider_value("max_turns", config, provider.name)
    max_turns_raw = overrides.max_turns if overrides.max_turns is not None else cfg_turns
    max_turns: int | None = int(max_turns_raw) if max_turns_raw is not None else None
    if max_turns is not None and not provider.max_turns_flag:
        # Best-effort: inject into prompt as guidance
        prompt_parts.append(f"Important: complete this task in no more than {max_turns} steps.")
        warnings.append(
            f"{provider.label} does not support --max-turns; "
            f"added guidance to prompt instead ({max_turns} steps)"
        )
        max_turns = None

    # --- Timeout ---
    cfg_timeout = resolve_provider_value("timeout", config, provider.name)
    timeout = (
        overrides.timeout
        if overrides.timeout is not None
        else (int(cfg_timeout) if cfg_timeout is not None else 1800)
    )

    # --- Subagents (warning only — filtering is handled elsewhere) ---
    subagents = config.get("subagents")
    if subagents and not provider.supports_agents_json:
        warnings.append(
            f"{provider.label} does not support sub-agents (--agents); "
            f"sub-agent definitions will be ignored"
        )

    # --- Instructions ---
    # Claude receives instructions via --append-system-prompt in the wrapper.
    # Codex receives instructions via -c model_instructions_file=... in the wrapper.
    # OpenCode-based providers receive instructions via opencode.json `instructions`
    # array (injected by prepare_agent_config_dir).
    # Remaining providers get best-effort prompt prepending.
    instructions = overrides.instructions
    if (
        instructions
        and provider.name not in {"claude", "codex"}
        and not provider.uses_opencode_instructions
    ):
        prompt_parts.insert(0, instructions)

    return ProviderConfig(
        model=model,
        max_turns=max_turns,
        timeout=timeout,
        prompt_extra="\n".join(prompt_parts),
        warnings=tuple(warnings),
    )


def build_headless_command(
    provider: AgentProvider,
    *,
    timeout: int,
    model: str | None = None,
    max_turns: int | None = None,
) -> str:
    """Assemble the bash command string for a headless agent run.

    The command assumes:
    - ``init-ssh-and-repo.sh`` has already set up the workspace
    - The prompt is in ``/home/dev/.terok/prompt.txt``
    - For Claude, the ``claude()`` wrapper function is sourced via bash -l

    Returns a bash command string suitable for ``["bash", "-lc", cmd]``.
    """
    if provider.name == "claude":
        return _build_claude_command(provider, timeout=timeout, model=model, max_turns=max_turns)
    return _build_generic_command(provider, timeout=timeout, model=model, max_turns=max_turns)


# ── Private helpers ──────────────────────────────────────────────────────────


def _build_claude_command(
    provider: AgentProvider,
    *,
    timeout: int,
    model: str | None,
    max_turns: int | None,
) -> str:
    """Build the headless command for Claude using the wrapper function."""
    # Claude uses the claude() wrapper from terok-agent.sh which handles
    # --add-dir, --agents, git env, and timeout
    flags = ""
    if model:
        flags += f" --model {shlex.quote(model)}"
    if max_turns:
        flags += f" --max-turns {int(max_turns)}"

    return (
        f"init-ssh-and-repo.sh &&"
        f" claude --terok-timeout {timeout}"
        f" -p "
        '"$(cat /home/dev/.terok/prompt.txt)"'
        f"{flags} --output-format stream-json --verbose"
    )


def _build_generic_command(
    provider: AgentProvider,
    *,
    timeout: int,
    model: str | None,
    max_turns: int | None,
) -> str:
    """Build the headless command for non-Claude providers.

    Uses the shell wrapper function (e.g. ``codex()``) instead of invoking the
    binary directly, so that git env vars and session resume logic from
    ``terok-agent.sh`` are applied.  The wrapper parses ``--terok-timeout``
    to wrap the actual invocation with ``timeout``.
    """
    parts = ["init-ssh-and-repo.sh &&"]

    # Call the wrapper function (sourced via bash -l from profile.d);
    # it handles git identity env vars and session resume args.
    parts.append(provider.binary)
    parts.append("--terok-timeout")
    parts.append(str(int(timeout)))

    # Subcommand (e.g. "exec" for codex, "run" for opencode)
    if provider.headless_subcommand:
        parts.append(provider.headless_subcommand)

    # Auto-approve flags are injected by the shell wrapper (wrappers.py)
    # based on TEROK_UNRESTRICTED env var — not here.

    # Model
    if model and provider.model_flag:
        parts.append(provider.model_flag)
        parts.append(shlex.quote(model))

    # Max turns
    if max_turns and provider.max_turns_flag:
        parts.append(provider.max_turns_flag)
        parts.append(str(int(max_turns)))

    # Output format
    for flag in provider.output_format_flags:
        parts.append(flag)

    # Verbose
    if provider.verbose_flag:
        parts.append(provider.verbose_flag)

    # Prompt — flag-based or positional
    if provider.prompt_flag:
        parts.append(provider.prompt_flag)
    parts.append('"$(cat /home/dev/.terok/prompt.txt)"')

    return " ".join(parts)
