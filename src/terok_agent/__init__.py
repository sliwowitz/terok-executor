# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-agent: single-agent task runner for hardened Podman containers.

Builds agent images, launches instrumented containers, and manages the
lifecycle of one AI coding agent at a time.  Designed for standalone use
(``terok-agent run claude .``) and as a library for terok orchestration.

Public API::

    # Provider registry
    from terok_agent import HEADLESS_PROVIDERS, HeadlessProvider, get_provider
    from terok_agent import PROVIDER_NAMES, CLIOverrides
    from terok_agent import apply_provider_config, build_headless_command
    from terok_agent import collect_opencode_provider_env, collect_all_auto_approve_env

    # Agent config preparation
    from terok_agent import AgentConfigSpec, prepare_agent_config_dir, parse_md_agent

    # Auth
    from terok_agent import AUTH_PROVIDERS, AuthProvider, authenticate

    # Instructions
    from terok_agent import resolve_instructions, bundled_default_instructions

    # Credential proxy
    from terok_agent import ensure_proxy_routes

    # Config stack
    from terok_agent import ConfigStack, ConfigScope, resolve_provider_value

Internal symbols (available via submodule import for white-box tests)::

    from terok_agent.provider.headless import generate_agent_wrapper, generate_all_wrappers
    from terok_agent.provider.headless import OpenCodeProviderConfig, ProviderConfig, WrapperConfig
    from terok_agent.roster.config_stack import deep_merge, load_yaml_scope, load_json_scope
    from terok_agent.provider.instructions import has_custom_instructions
    from terok_agent._util import podman_userns_args
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-agent")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

# -- Container environment assembly --------------------------------------------
# -- Config resolution ---------------------------------------------------------
# Re-export protocol types from terok-sandbox for convenience
from terok_sandbox.doctor import CheckVerdict, DoctorCheck

# -- Command registry ----------------------------------------------------------
from .commands import COMMANDS as AGENT_COMMANDS, CommandDef

# -- Build: image construction + resource staging ------------------------------
from .container.build import (
    DEFAULT_BASE_IMAGE,
    BuildError,
    ImageSet,
    build_base_images,
    build_sidecar_image,
    l0_image_tag,
    l1_image_tag,
    l1_sidecar_image_tag,
    render_l1_sidecar,
    stage_scripts,
    stage_tmux_config,
    stage_toad_agents,
)
from .container.env import ContainerEnvResult, ContainerEnvSpec, assemble_container_env

# -- Runner facade -------------------------------------------------------------
from .container.runner import AgentRunner

# -- Auth ----------------------------------------------------------------------
from .credentials.auth import (
    AUTH_PROVIDERS,
    PHANTOM_CREDENTIALS_MARKER,
    AuthProvider,
    authenticate,
    store_api_key,
)

# -- Credential proxy ----------------------------------------------------------
from .credentials.extractors import extract_credential
from .credentials.proxy_commands import PROXY_COMMANDS, scan_leaked_credentials
from .doctor import agent_doctor_checks
from .paths import mounts_dir

# -- Agent config preparation --------------------------------------------------
from .provider.agents import AgentConfigSpec, parse_md_agent, prepare_agent_config_dir
from .provider.config import resolve_provider_value

# -- Provider registry ---------------------------------------------------------
from .provider.headless import (
    HEADLESS_PROVIDERS,
    PROVIDER_NAMES,
    CLIOverrides,
    HeadlessProvider,
    apply_provider_config,
    build_headless_command,
    collect_all_auto_approve_env,
    collect_opencode_provider_env,
    get_provider,
)

# -- Instructions --------------------------------------------------------------
from .provider.instructions import bundled_default_instructions, resolve_instructions
from .roster import CredentialProxyRoute, SidecarSpec, ensure_proxy_routes, get_roster

# -- Config stack --------------------------------------------------------------
from .roster.config_stack import ConfigScope, ConfigStack

# -- Bootstrap YAML roster into module-level dicts ---------------------------
# HEADLESS_PROVIDERS and AUTH_PROVIDERS are empty dicts populated here to avoid
# circular imports (roster → auth/headless_providers → roster).


def _bootstrap_roster() -> None:
    """Populate module-level provider dicts from the YAML roster."""
    global PROVIDER_NAMES  # noqa: PLW0603 — tuple requires rebind

    import terok_agent.provider.headless as _hp

    from .roster import get_roster

    roster = get_roster()
    HEADLESS_PROVIDERS.update(roster.providers)
    AUTH_PROVIDERS.update(roster.auth_providers)
    PROVIDER_NAMES = _hp.PROVIDER_NAMES = roster.agent_names


_bootstrap_roster()

__all__ = [
    "__version__",
    # Provider registry
    "HEADLESS_PROVIDERS",
    "PROVIDER_NAMES",
    "HeadlessProvider",
    "get_provider",
    "CLIOverrides",
    "apply_provider_config",
    "build_headless_command",
    "collect_opencode_provider_env",
    "collect_all_auto_approve_env",
    # Agent config preparation
    "AgentConfigSpec",
    "prepare_agent_config_dir",
    "parse_md_agent",
    # Auth
    "AUTH_PROVIDERS",
    "AuthProvider",
    "PHANTOM_CREDENTIALS_MARKER",
    "authenticate",
    "store_api_key",
    # Instructions
    "bundled_default_instructions",
    "resolve_instructions",
    # Config stack
    "ConfigScope",
    "ConfigStack",
    "resolve_provider_value",
    # Build: image construction + resource staging
    "DEFAULT_BASE_IMAGE",
    "BuildError",
    "ImageSet",
    "build_base_images",
    "build_sidecar_image",
    "l0_image_tag",
    "l1_image_tag",
    "l1_sidecar_image_tag",
    "render_l1_sidecar",
    "stage_scripts",
    "stage_toad_agents",
    "stage_tmux_config",
    # Credential proxy
    "CredentialProxyRoute",
    "ensure_proxy_routes",
    "extract_credential",
    # Roster
    "SidecarSpec",
    "get_roster",
    # Command registry
    "AGENT_COMMANDS",
    "PROXY_COMMANDS",
    "CommandDef",
    "mounts_dir",
    "scan_leaked_credentials",
    # Doctor (container health checks)
    "CheckVerdict",
    "DoctorCheck",
    "agent_doctor_checks",
    # Runner facade
    "AgentRunner",
    # Container environment assembly
    "ContainerEnvSpec",
    "ContainerEnvResult",
    "assemble_container_env",
]
