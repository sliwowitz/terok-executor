# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-executor: single-agent task runner for hardened Podman containers.

Builds agent images, launches instrumented containers, and manages the
lifecycle of one AI coding agent at a time.  Designed for standalone use
(``terok-executor run claude .``) and as a library for terok orchestration.

The public surface is ``__all__`` below.  Key entry points:

- [`AgentRunner`][terok_executor.AgentRunner] — launch agents in containers
- [`Authenticator`][terok_executor.Authenticator] — credential flow
- [`ImageBuilder`][terok_executor.ImageBuilder] — image construction
- [`AgentRoster.shared`][terok_executor.AgentRoster.shared] — YAML agent registry (process-wide cache)

Implementation-detail types (raw config schema fragments, ACP error
classes, internal result types, sidecar image / inject helpers) stay
in their submodules; reach into ``terok_executor.<sub>`` when you
need them.
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-executor")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

# -- terok-util shared types (re-exported for convenience) --------------------
from terok_util import ConfigStack

# -- terok-sandbox protocol types (re-exported for convenience) ----------------
from terok_executor.integrations.sandbox import ConfigScope

# -- Commands + CLI surface ----------------------------------------------------
from ._tree import COMMANDS

# -- ACP host-proxy (per-task multi-agent aggregator) -------------------------
from .acp import ACPEndpointStatus, acp_socket_is_live, list_authenticated_agents
from .commands import COMMANDS as AGENT_COMMANDS

# -- Config schema + read/write accessors for the executor-owned image: section --
from .config_schema import ExecutorConfigView, RawImageSection

# -- Container (build, env assembly, runner) -----------------------------------
from .container.build import (
    AGENTS_LABEL,
    DEFAULT_BASE_IMAGE,
    BuildError,
    ImageBuilder,
    ImageSet,
    build_project_image,
)
from .container.cache import seed_workspace_from_clone_cache
from .container.env import ContainerEnvSpec, assemble_container_env
from .container.inject import inject_prompt
from .container.runner import AgentRunner

# -- Credentials (auth flows, extractors, vault commands) ----------------------
from .credentials.auth import AUTH_PROVIDERS, Authenticator
from .credentials.vault_commands import VAULT_COMMANDS, scan_leaked_credentials

# -- Krun (KVM-microVM) provisioning + runtime factory -----------------------
from .krun import KrunHost, KrunHostKeypair, ensure_krun_host_keypair

# -- Provider (descriptor + headless behaviour, instructions, agent config) ----
from .provider.agents import AgentConfigSpec, parse_md_agent, prepare_agent_config_dir
from .provider.instructions import bundled_default_instructions, resolve_instructions
from .provider.providers import (
    AGENT_PROVIDERS,
    PROVIDER_NAMES,
    AgentProvider,
    CLIOverrides,
    get_provider,
    resolve_provider_value,
)

# -- Roster (agent catalog + config resolution) --------------------------------
from .roster import AgentRoster

# -- Sandbox bootstrap composition ---------------------------------------------
from .sandbox import ensure_sandbox_ready

# -- Storage queries (filesystem footprint measurement) -------------------------
from .storage import SharedMountStorageInfo, TaskStorageInfo

# -- Bootstrap YAML roster into module-level dicts ---------------------------
# AGENT_PROVIDERS and AUTH_PROVIDERS are empty dicts populated here to avoid
# circular imports (roster → auth/providers → roster).


def _bootstrap_roster() -> None:
    """Populate module-level provider dicts from the YAML roster."""
    global PROVIDER_NAMES  # noqa: PLW0603 — tuple requires rebind

    import terok_executor.provider.providers as _reg

    roster = AgentRoster.shared()
    AGENT_PROVIDERS.update(roster.providers)
    AUTH_PROVIDERS.update(roster.auth_providers)
    PROVIDER_NAMES = _reg.PROVIDER_NAMES = roster.agent_names


_bootstrap_roster()

__all__ = [
    "__version__",
    # ACP host-proxy
    "ACPEndpointStatus",
    "acp_socket_is_live",
    "list_authenticated_agents",
    # Provider registry + behaviour
    "AGENT_PROVIDERS",
    "AgentProvider",
    "CLIOverrides",
    "PROVIDER_NAMES",
    "get_provider",
    "resolve_provider_value",
    # Agent config preparation
    "AgentConfigSpec",
    "parse_md_agent",
    "prepare_agent_config_dir",
    # Auth
    "AUTH_PROVIDERS",
    "Authenticator",
    # Instructions
    "bundled_default_instructions",
    "resolve_instructions",
    # Config stack
    "ConfigScope",
    "ConfigStack",
    # Config schema (executor-owned slice of the shared config.yml)
    "ExecutorConfigView",
    "RawImageSection",
    # Build: image construction + resource staging
    "AGENTS_LABEL",
    "DEFAULT_BASE_IMAGE",
    "BuildError",
    "ImageBuilder",
    "ImageSet",
    "build_project_image",
    # Vault credential scanning
    "scan_leaked_credentials",
    # Roster
    "AgentRoster",
    # Command registry
    "AGENT_COMMANDS",
    "COMMANDS",
    "VAULT_COMMANDS",
    # Storage queries
    "SharedMountStorageInfo",
    "TaskStorageInfo",
    # Runner facade
    "AgentRunner",
    # Container environment assembly
    "ContainerEnvSpec",
    "assemble_container_env",
    # Clone cache + injection helpers
    "inject_prompt",
    "seed_workspace_from_clone_cache",
    # Sandbox bootstrap composition
    "ensure_sandbox_ready",
    # Krun (KVM-microVM) provisioning + runtime factory
    "KrunHost",
    "KrunHostKeypair",
    "ensure_krun_host_keypair",
]
