# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-executor: single-agent task runner for hardened Podman containers.

Builds agent images, launches instrumented containers, and manages the
lifecycle of one AI coding agent at a time.  Designed for standalone use
(``terok-executor run claude .``) and as a library for terok orchestration.

The public surface is ``__all__`` below.  Key entry points:

- [`AgentRunner`][terok_executor.AgentRunner] — launch agents in containers
- [`Authenticator`][terok_executor.credentials.auth.Authenticator] — credential flow
- [`ImageBuilder`][terok_executor.ImageBuilder] — image construction
- [`AgentRoster.shared`][terok_executor.roster.loader.AgentRoster.shared] — YAML agent registry (process-wide cache)

Every public name is served lazily (PEP 562 ``__getattr__``): the
submodule that defines a symbol is imported only when that symbol is
first accessed, so a bare ``import terok_executor`` pays for neither the
``acp`` protocol stack nor ``terok_sandbox`` until something actually
reaches for a symbol that needs them.  The three ACP names
([`ACPEndpointStatus`][terok_executor.ACPEndpointStatus],
[`acp_socket_is_live`][terok_executor.acp_socket_is_live],
[`list_authenticated_agents`][terok_executor.list_authenticated_agents])
are deliberately kept off the roster-bootstrap path so the host-side
``acp list`` probe stays cheap.

Implementation-detail types (raw config schema fragments, ACP error
classes, internal result types, sidecar image / inject helpers) stay
in their submodules; reach into ``terok_executor.<sub>`` when you
need them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-executor")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

if TYPE_CHECKING:
    # Names for type-checkers and IDEs only — never imported at runtime,
    # where ``__getattr__`` below resolves each on first use.  The
    # ``X as X`` redundant-alias form marks each as a deliberate
    # re-export (the ``py.typed`` contract for downstream consumers).
    from ._tree import COMMANDS as COMMANDS
    from .acp import (
        ACPEndpointStatus as ACPEndpointStatus,
        acp_socket_is_live as acp_socket_is_live,
        list_authenticated_agents as list_authenticated_agents,
    )
    from .commands import COMMANDS as AGENT_COMMANDS  # noqa: F401 — renamed re-export
    from .config_schema import (
        ExecutorConfigView as ExecutorConfigView,
        RawImageSection as RawImageSection,
    )
    from .container.build import (
        AGENTS_LABEL as AGENTS_LABEL,
        DEFAULT_BASE_IMAGE as DEFAULT_BASE_IMAGE,
        BuildError as BuildError,
        ImageBuilder as ImageBuilder,
        ImageSet as ImageSet,
        build_project_image as build_project_image,
        known_family as known_family,
    )
    from .container.cache import seed_workspace_from_clone_cache as seed_workspace_from_clone_cache
    from .container.env import (
        ContainerEnvSpec as ContainerEnvSpec,
        assemble_container_env as assemble_container_env,
    )
    from .container.inject import inject_prompt as inject_prompt
    from .container.runner import AgentRunner as AgentRunner
    from .credentials.auth import (
        AUTH_PROVIDERS as AUTH_PROVIDERS,
        Authenticator as Authenticator,
        AuthSession as AuthSession,
        credential_provider as credential_provider,
        prepare_oauth_session as prepare_oauth_session,
        store_api_key as store_api_key,
    )
    from .credentials.vault_commands import (
        VAULT_COMMANDS as VAULT_COMMANDS,
        scan_leaked_credentials as scan_leaked_credentials,
    )
    from .krun import (
        KrunHost as KrunHost,
        KrunHostKeypair as KrunHostKeypair,
        ensure_krun_host_keypair as ensure_krun_host_keypair,
    )
    from .provider.agents import (
        AgentConfigSpec as AgentConfigSpec,
        prepare_agent_config_dir as prepare_agent_config_dir,
    )
    from .provider.instructions import (
        bundled_default_instructions as bundled_default_instructions,
        resolve_instructions as resolve_instructions,
    )
    from .provider.providers import (
        AGENT_NAMES as AGENT_NAMES,
        AGENTS as AGENTS,
        Agent as Agent,
        CLIOverrides as CLIOverrides,
        get_agent as get_agent,
        resolve_agent_value as resolve_agent_value,
    )
    from .roster import AgentRoster as AgentRoster
    from .sandbox import ensure_sandbox_ready as ensure_sandbox_ready
    from .storage import (
        SharedMountStorageInfo as SharedMountStorageInfo,
        TaskStorageInfo as TaskStorageInfo,
    )


#: Public symbol → defining submodule (relative dotted path).  The single
#: source of truth for lazy resolution *and* ``__all__``.
_LAZY: dict[str, str] = {
    # ACP host-proxy (per-task multi-agent aggregator)
    "ACPEndpointStatus": ".acp",
    "acp_socket_is_live": ".acp",
    "list_authenticated_agents": ".acp",
    # Provider registry + behaviour
    "AGENTS": ".provider.providers",
    "Agent": ".provider.providers",
    "AGENT_NAMES": ".provider.providers",
    "CLIOverrides": ".provider.providers",
    "get_agent": ".provider.providers",
    "resolve_agent_value": ".provider.providers",
    # Agent config preparation
    "AgentConfigSpec": ".provider.agents",
    "prepare_agent_config_dir": ".provider.agents",
    # Instructions
    "bundled_default_instructions": ".provider.instructions",
    "resolve_instructions": ".provider.instructions",
    # Auth
    "AUTH_PROVIDERS": ".credentials.auth",
    "Authenticator": ".credentials.auth",
    "AuthSession": ".credentials.auth",
    "credential_provider": ".credentials.auth",
    "prepare_oauth_session": ".credentials.auth",
    "store_api_key": ".credentials.auth",
    # Vault credential scanning
    "scan_leaked_credentials": ".credentials.vault_commands",
    "VAULT_COMMANDS": ".credentials.vault_commands",
    # Config schema (executor-owned slice of the shared config.yml)
    "ExecutorConfigView": ".config_schema",
    "RawImageSection": ".config_schema",
    # Build: image construction + resource staging
    "AGENTS_LABEL": ".container.build",
    "DEFAULT_BASE_IMAGE": ".container.build",
    "BuildError": ".container.build",
    "ImageBuilder": ".container.build",
    "ImageSet": ".container.build",
    "build_project_image": ".container.build",
    "known_family": ".container.build",
    # Container environment assembly
    "ContainerEnvSpec": ".container.env",
    "assemble_container_env": ".container.env",
    # Clone cache + injection helpers
    "inject_prompt": ".container.inject",
    "seed_workspace_from_clone_cache": ".container.cache",
    # Runner facade
    "AgentRunner": ".container.runner",
    # Roster (agent catalog + config resolution)
    "AgentRoster": ".roster",
    # Command registries
    "COMMANDS": "._tree",
    "AGENT_COMMANDS": ".commands:COMMANDS",
    # Storage queries (filesystem footprint measurement)
    "SharedMountStorageInfo": ".storage",
    "TaskStorageInfo": ".storage",
    # Sandbox bootstrap composition
    "ensure_sandbox_ready": ".sandbox",
    # Krun (KVM-microVM) provisioning + runtime factory
    "KrunHost": ".krun",
    "KrunHostKeypair": ".krun",
    "ensure_krun_host_keypair": ".krun",
}

#: The three ACP names that must resolve *without* triggering the roster
#: bootstrap: they read a Unix socket / the credential DB, never the
#: agent registry, so keeping them off the bootstrap path preserves a
#: cheap host-side ``acp list`` and an ``acp``-free import.
_BOOTSTRAP_FREE = frozenset(
    {"ACPEndpointStatus", "acp_socket_is_live", "list_authenticated_agents"}
)

__all__ = ["__version__", *_LAZY]

_bootstrapped = False


def _bootstrap_roster() -> None:
    """Populate the module-level agent dicts from the YAML roster.

    ``AGENTS`` / ``AUTH_PROVIDERS`` / ``OPENCODE_PROVIDERS`` are declared
    empty in their defining modules and filled here, once, from the
    shared roster.  The empty-then-fill dance is what breaks the
    ``roster → auth/providers → roster`` import cycle: the leaf modules
    stay dependency-free and the root package (the one layer allowed to
    depend on the roster) does the wiring.
    """
    global AGENT_NAMES  # noqa: PLW0603 — tuple requires rebind

    import terok_executor.provider.providers as _reg
    from terok_executor.credentials.auth import AUTH_PROVIDERS
    from terok_executor.roster import AgentRoster

    roster = AgentRoster.shared()
    _reg.AGENTS.update(roster.agents)
    AUTH_PROVIDERS.update(roster.auth_providers)
    AGENT_NAMES = _reg.AGENT_NAMES = roster.agent_names
    _reg.OPENCODE_PROVIDERS.update(
        {
            p.name: p.opencode_config.config_dir
            for p in roster.providers.values()
            if p.opencode_config
        }
    )


def _ensure_bootstrapped() -> None:
    """Run [`_bootstrap_roster`][terok_executor._bootstrap_roster] exactly once.

    Deferred out of module import so a bare ``import terok_executor``
    doesn't pay for the roster YAML load (and its ``terok_sandbox``
    pull).  Triggered on the first access of any registry-backed public
    name and by the CLI entry point before it dispatches a command.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    _bootstrap_roster()


def __getattr__(name: str) -> object:
    """Resolve a public name to its defining submodule on first access (PEP 562).

    A ``"module:attr"`` target renames on the way through — the sole
    aliased export is ``AGENT_COMMANDS``, which is ``COMMANDS`` in
    [`.commands`][terok_executor.commands] (``COMMANDS`` at the package
    root is the *composed* tree from [`._tree`][terok_executor._tree]).
    """
    try:
        target = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    if name not in _BOOTSTRAP_FREE:
        _ensure_bootstrapped()
    module_path, _, source_name = target.partition(":")
    value = getattr(importlib.import_module(module_path, __name__), source_name or name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    """Expose the lazy names to ``dir()`` / autocompletion."""
    return sorted({*globals(), *_LAZY})
