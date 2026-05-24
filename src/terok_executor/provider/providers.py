# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Agent provider registry + per-provider behaviour.

Each supported AI coding agent is described by an
[`AgentProvider`][terok_executor.provider.providers.AgentProvider]
dataclass that owns both its capability shape (flags, environment,
session handling) and the behaviour bound to that shape — config
resolution ([`AgentProvider.apply_config`][terok_executor.provider.providers.AgentProvider.apply_config])
and headless-command assembly ([`AgentProvider.build_headless_command`][terok_executor.provider.providers.AgentProvider.build_headless_command]).

The ``AGENT_PROVIDERS`` dict maps short names to descriptors and is
populated at package load time from the YAML roster.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenCodeProviderConfig:
    """Immutable descriptor for an OpenCode-based provider wrapper."""

    display_name: str
    """Human-readable display name (e.g., 'Helmholtz Blablador')."""

    base_url: str
    """Base URL for the OpenAI-compatible API (e.g., 'https://api.helmholtz-blablador.fz-juelich.de/v1')."""

    preferred_model: str
    """Preferred model ID (e.g., 'alias-huge')."""

    fallback_model: str
    """Fallback model ID if preferred is unavailable (e.g., 'alias-code')."""

    env_var_prefix: str
    """Environment variable prefix for API key (e.g., 'BLABLADOR' → BLABLADOR_API_KEY)."""

    config_dir: str
    """Configuration directory name (e.g., '.blablador')."""

    auth_key_url: str
    """URL where users can obtain API keys for documentation."""

    def to_env(self, name: str) -> dict[str, str]:
        """Return env vars for container injection, keyed by TEROK_OC_{NAME}_*."""
        prefix = f"TEROK_OC_{name.upper()}_"
        return {
            f"{prefix}BASE_URL": self.base_url,
            f"{prefix}PREFERRED_MODEL": self.preferred_model,
            f"{prefix}FALLBACK_MODEL": self.fallback_model,
            f"{prefix}DISPLAY_NAME": self.display_name,
            f"{prefix}ENV_VAR_PREFIX": self.env_var_prefix,
            f"{prefix}CONFIG_DIR": self.config_dir,
        }


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved per-run config for a headless provider.

    Produced by [`AgentProvider.apply_config`][terok_executor.provider.providers.AgentProvider.apply_config]
    after best-effort feature mapping.
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


def resolve_provider_value(
    key: str,
    config: dict[str, Any],
    provider_name: str,
) -> Any | None:
    """Extract a provider-aware config value.

    Supports two forms:

    * **Flat value** — ``model: opus`` → same for all providers.
    * **Per-provider dict** — ``model: {claude: opus, codex: o3, _default: fast}``
      → looks up *provider_name*, falls back to ``_default``, then ``None``.

    Returns ``None`` when the key is absent or has no match for the provider.

    **Null override behaviour**: when a per-provider dict maps a provider to
    ``null`` (Python ``None``), that ``None`` is treated as "no value" and the
    resolver falls back to ``_default``.  This is intentional — it allows a
    lower-priority config layer to set a provider-specific value that a
    higher-priority layer can effectively *unset* by mapping it to ``null``,
    letting the ``_default`` (or ``None``) bubble up instead.

    Internal to provider config resolution — full config-stack composition
    (``build_agent_config_stack``, ``resolve_agent_config``) lives in terok,
    which owns the global/project/preset layer semantics.
    """
    val = config.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        provider_val = val.get(provider_name)
        if provider_val is not None:
            return provider_val
        return val.get("_default")
    return val


@dataclass(frozen=True)
class AgentProvider:
    """Describes how to run one AI coding agent (all modes: interactive + headless)."""

    name: str
    """Short key used in CLI dispatch (e.g. ``"claude"``, ``"codex"``)."""

    label: str
    """Human-readable display name (e.g. ``"Claude"``, ``"Codex"``)."""

    binary: str
    """CLI binary name (e.g. ``"claude"``, ``"codex"``, ``"opencode"``)."""

    git_author_name: str
    """AI identity name for Git author/committer policy application."""

    git_author_email: str
    """AI identity email for Git author/committer policy application."""

    # -- Headless command construction --

    headless_subcommand: str | None
    """Subcommand for headless mode (e.g. ``"exec"`` for codex, ``"run"`` for opencode).

    ``None`` means the binary uses flags only (e.g. ``claude -p``).
    """

    prompt_flag: str
    """Flag for passing the prompt.

    ``"-p"`` for flag-based, ``""`` for positional (after subcommand).
    """

    auto_approve_env: dict[str, str]
    """Environment variables for fully autonomous execution.

    Injected into the container env by ``_apply_unrestricted_env()`` when
    ``TEROK_UNRESTRICTED=1``.  Read by agents regardless of launch path.
    Claude uses ``/etc/claude-code/managed-settings.json`` instead.
    """

    auto_approve_flags: tuple[str, ...]
    """CLI flags injected by the shell wrapper when ``TEROK_UNRESTRICTED=1``.

    Only for agents that lack an env var or managed config mechanism
    (currently Codex only).  Empty for all other agents — their env vars
    and ``/etc/`` config files handle permissions across all launch paths.
    """

    output_format_flags: tuple[str, ...]
    """Flags for structured output (e.g. ``("--output-format", "stream-json")``)."""

    model_flag: str | None
    """Flag for model override (``"--model"``, ``"--agent"``, or ``None``)."""

    max_turns_flag: str | None
    """Flag for maximum turns (``"--max-turns"`` or ``None``)."""

    verbose_flag: str | None
    """Flag for verbose output (``"--verbose"`` or ``None``)."""

    # -- Session support --

    supports_session_resume: bool
    """Whether the provider supports resuming a previous session."""

    resume_flag: str | None
    """Flag to resume a session (e.g. ``"--resume"``, ``"--session"``)."""

    continue_flag: str | None
    """Flag to continue a session (e.g. ``"--continue"``)."""

    session_file: str | None
    """Filename in ``/home/dev/.terok/`` for stored session ID.

    Providers that capture session IDs via plugin or post-run parsing set this
    to a filename (e.g. ``"opencode-session.txt"``).  Providers with their own
    hook mechanism (Claude) or no session support set this to ``None``.
    """

    # -- Claude-specific capabilities --

    supports_agents_json: bool
    """Whether the provider supports ``--agents`` JSON (Claude only)."""

    supports_session_hook: bool
    """Whether the provider supports SessionStart hooks (Claude only)."""

    supports_add_dir: bool
    """Whether the provider supports ``--add-dir "/"`` (Claude only)."""

    # -- Log formatting --

    log_format: str
    """Log format identifier: ``"claude-stream-json"`` or ``"plain"``."""

    opencode_config: OpenCodeProviderConfig | None = None
    """Configuration for OpenCode-based providers (Blablador, KISSKI, etc.).

    When set, this provider uses OpenCode with a custom OpenAI-compatible API.
    The configuration includes API endpoints, model preferences, and provider-specific
    settings that are injected into the container environment.
    """

    refuse_subcommands: tuple[str, ...] = ()
    """Subcommands the in-container wrapper refuses with a friendly error.

    Used to block credential-handling flows (``login``, ``logout``,
    ``setup-token``) that would otherwise pollute the host-shared mount —
    operators authenticate on the host via ``terok auth`` instead.  Best
    effort only; the firewall is the actual enforcement
    ([terok-ai/terok#873](https://github.com/terok-ai/terok/issues/873)).
    """

    @property
    def uses_opencode_instructions(self) -> bool:
        """Whether the provider uses OpenCode's instruction system."""
        return self.opencode_config is not None or self.name == "opencode"

    # ── Headless behaviour ───────────────────────────────────────────

    def apply_config(
        self,
        config: dict[str, Any],
        overrides: CLIOverrides | None = None,
    ) -> ProviderConfig:
        """Resolve config values for this provider with best-effort feature mapping.

        CLI flag *overrides* take precedence over *config* values.  When this
        provider lacks a feature, an analogue is used where possible (e.g.
        injecting max-turns guidance into the prompt), and a warning is
        emitted for features that have no analogue.
        """
        if overrides is None:
            overrides = CLIOverrides()

        warnings: list[str] = []
        prompt_parts: list[str] = []

        # --- Model ---
        cfg_model = resolve_provider_value("model", config, self.name)
        model = overrides.model or (str(cfg_model) if cfg_model is not None else None)
        if model and not self.model_flag:
            warnings.append(
                f"{self.label} does not support model selection; ignoring model={model!r}"
            )
            model = None

        # --- Max turns ---
        cfg_turns = resolve_provider_value("max_turns", config, self.name)
        max_turns_raw = overrides.max_turns if overrides.max_turns is not None else cfg_turns
        max_turns: int | None = int(max_turns_raw) if max_turns_raw is not None else None
        if max_turns is not None and not self.max_turns_flag:
            # Best-effort: inject into prompt as guidance
            prompt_parts.append(f"Important: complete this task in no more than {max_turns} steps.")
            warnings.append(
                f"{self.label} does not support --max-turns; "
                f"added guidance to prompt instead ({max_turns} steps)"
            )
            max_turns = None

        # --- Timeout ---
        cfg_timeout = resolve_provider_value("timeout", config, self.name)
        timeout = (
            overrides.timeout
            if overrides.timeout is not None
            else (int(cfg_timeout) if cfg_timeout is not None else 1800)
        )

        # --- Subagents (warning only — filtering is handled elsewhere) ---
        subagents = config.get("subagents")
        if subagents and not self.supports_agents_json:
            warnings.append(
                f"{self.label} does not support sub-agents (--agents); "
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
            and self.name not in {"claude", "codex"}
            and not self.uses_opencode_instructions
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
        self,
        *,
        timeout: int,
        model: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Assemble the bash command string for a headless agent run.

        The command assumes:

        - ``init-ssh-and-repo.sh`` has already set up the workspace
        - The prompt is in ``/home/dev/.terok/prompt.txt``
        - For Claude, the ``claude()`` wrapper function is sourced via ``bash -l``

        Returns a bash command string suitable for ``["bash", "-lc", cmd]``.
        Dispatches to provider-specific assembly: Claude routes through the
        shell wrapper (which adds ``--add-dir``, ``--agents``, git env);
        everything else uses the generic shape with subcommand + flags.
        """
        if self.name == "claude":
            return self._build_claude_command(timeout=timeout, model=model, max_turns=max_turns)
        return self._build_generic_command(timeout=timeout, model=model, max_turns=max_turns)

    def _build_claude_command(
        self,
        *,
        timeout: int,
        model: str | None,
        max_turns: int | None,
    ) -> str:
        """Build the headless command for Claude using the wrapper function.

        Claude uses the ``claude()`` wrapper from ``terok-executor.sh`` which
        handles ``--add-dir``, ``--agents``, git env, and timeout.
        """
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
        self,
        *,
        timeout: int,
        model: str | None,
        max_turns: int | None,
    ) -> str:
        """Build the headless command for non-Claude providers.

        Uses the shell wrapper function (e.g. ``codex()``) instead of invoking the
        binary directly, so that git env vars and session resume logic from
        ``terok-executor.sh`` are applied.  The wrapper parses ``--terok-timeout``
        to wrap the actual invocation with ``timeout``.
        """
        parts = ["init-ssh-and-repo.sh &&"]

        # Call the wrapper function (sourced via bash -l from profile.d);
        # it handles git identity env vars and session resume args.
        parts.append(self.binary)
        parts.append("--terok-timeout")
        parts.append(str(int(timeout)))

        # Subcommand (e.g. "exec" for codex, "run" for opencode)
        if self.headless_subcommand:
            parts.append(self.headless_subcommand)

        # Auto-approve flags are injected by the shell wrapper (wrappers.py)
        # based on TEROK_UNRESTRICTED env var — not here.

        # Model
        if model and self.model_flag:
            parts.append(self.model_flag)
            parts.append(shlex.quote(model))

        # Max turns
        if max_turns and self.max_turns_flag:
            parts.append(self.max_turns_flag)
            parts.append(str(int(max_turns)))

        # Output format
        for flag in self.output_format_flags:
            parts.append(flag)

        # Verbose
        if self.verbose_flag:
            parts.append(self.verbose_flag)

        # Prompt — flag-based or positional
        if self.prompt_flag:
            parts.append(self.prompt_flag)
        parts.append('"$(cat /home/dev/.terok/prompt.txt)"')

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Provider registry — populated from YAML by __init__.py at package load time
# ---------------------------------------------------------------------------

AGENT_PROVIDERS: dict[str, AgentProvider] = {}
"""All agent providers, keyed by name.  Loaded from ``resources/agents/*.yaml``."""

PROVIDER_NAMES: tuple[str, ...] = ()


# ── Public API ──────────────────────────────────────────────────────────────


def resolve_provider(
    providers: dict[str, AgentProvider],
    name: str | None,
    *,
    default_agent: str | None = None,
) -> AgentProvider:
    """Look up a provider by name from *providers*, with fallback chain.

    Resolution order: explicit *name* → *default_agent* → ``"claude"``.
    Raises ``SystemExit`` if the resolved name is not found.
    """
    resolved = name or default_agent or "claude"
    provider = providers.get(resolved)
    if provider is None:
        valid = ", ".join(sorted(providers))
        raise SystemExit(f"Unknown provider {resolved!r}. Valid providers: {valid}")
    return provider


def get_provider(name: str | None, *, default_agent: str | None = None) -> AgentProvider:
    """Resolve a provider name against the global [`AGENT_PROVIDERS`][terok_executor.provider.providers.AGENT_PROVIDERS] registry.

    Convenience wrapper around [`resolve_provider`][terok_executor.provider.providers.resolve_provider].
    """
    return resolve_provider(AGENT_PROVIDERS, name, default_agent=default_agent)


def collect_opencode_provider_env() -> dict[str, str]:
    """Collect environment variables for all OpenCode-based providers.

    Returns a dictionary of environment variables that will be injected into containers
    to configure OpenCode-based providers. Each provider with opencode_config set
    contributes variables prefixed with TEROK_OC_{PROVIDER_NAME}_*.
    """
    env: dict[str, str] = {}
    for provider in AGENT_PROVIDERS.values():
        if provider.opencode_config is not None:
            env.update(provider.opencode_config.to_env(provider.name))
    return env
