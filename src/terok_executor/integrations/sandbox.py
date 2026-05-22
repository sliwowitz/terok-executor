# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the ``terok_sandbox`` wheel.

Re-export catalog: every ``from terok_sandbox …`` import in
``terok_executor`` lives here.  The contract is enforced by
``.importlinter`` (``terok_sandbox`` is a protected module with
``terok_executor.integrations.sandbox`` as the sole allowed importer).

When a sibling release renames, splits, or relocates a symbol, only
this file needs to change — the rest of terok-executor keeps reading
the same ``terok_executor.integrations.sandbox.X`` name.  Convention
shared with terok-sandbox (which adapts terok-shield and
terok-clearance the same way) and terok-main (where the same pattern
lives at ``terok.lib.integrations.*``).
"""

from terok_sandbox import (  # noqa: F401 — re-exported public API
    CODEX_SHARED_OAUTH_MARKER,
    CONTAINER_RUNTIME_DIR,
    PHANTOM_CREDENTIALS_MARKER,
    READY_MARKER,
    CommandDef,
    ConfigScope,
    ConfigStack,
    ContainerRuntime,
    CredentialDB,
    GitGate,
    GpuConfigError,
    KrunRuntime,
    LifecycleHooks,
    PodmanRuntime,
    RunSpec,
    Sandbox,
    SandboxConfig,
    SandboxConfigView,
    SetupVerdict,
    Sharing,
    SSHManager,
    TcpSSHTransport,
    VaultStatus,
    VolumeSpec,
    check_environment,
    ensure_infra_keypair,
    get_server_status,
    get_ssh_signer_port,
    get_token_broker_port,
    get_vault_status,
    install_vault_systemd,
    installed_versions,
    is_systemd_available,
    is_vault_running,
    is_vault_socket_active,
    is_vault_systemd_available,
    namespace_runtime_dir,
    needs_setup,
    podman_port_resolver,
    read_stamp,
    sanitize_tty,
    stage_line,
    stamp_path,
    start_vault,
    stop_vault,
    systemd_creds_has_tpm2,
    uninstall_vault_systemd,
    yaml_update_section,
)
from terok_sandbox.commands import (  # noqa: F401 — re-exported public API
    COMMANDS,
    ArgDef,
    CommandTree,
    _handle_sandbox_setup,
    _handle_sandbox_uninstall,
)
from terok_sandbox.config_stack import deep_merge  # noqa: F401 — re-exported public API
from terok_sandbox.doctor import CheckVerdict, DoctorCheck  # noqa: F401 — re-exported public API
from terok_sandbox.paths import (  # noqa: F401 — re-exported public API
    namespace_config_dir,
    namespace_state_dir,
    read_config_section,
)

#: Public surface of the adapter.  Underscore-prefixed handler symbols
#: (``_handle_sandbox_setup`` / ``_handle_sandbox_uninstall``) stay
#: importable for the executor's CLI registry splice and the tests
#: that mock them, but their underscore-prefix keeps them out of
#: ``__all__`` — they're internal adapter wiring, not part of the
#: stable cross-package contract.
__all__ = [
    "CODEX_SHARED_OAUTH_MARKER",
    "COMMANDS",
    "CONTAINER_RUNTIME_DIR",
    "PHANTOM_CREDENTIALS_MARKER",
    "READY_MARKER",
    "ArgDef",
    "CheckVerdict",
    "CommandDef",
    "CommandTree",
    "ConfigScope",
    "ConfigStack",
    "ContainerRuntime",
    "CredentialDB",
    "DoctorCheck",
    "GitGate",
    "GpuConfigError",
    "KrunRuntime",
    "LifecycleHooks",
    "PodmanRuntime",
    "RunSpec",
    "SSHManager",
    "Sandbox",
    "SandboxConfig",
    "SandboxConfigView",
    "SetupVerdict",
    "Sharing",
    "TcpSSHTransport",
    "VaultStatus",
    "VolumeSpec",
    "check_environment",
    "deep_merge",
    "ensure_infra_keypair",
    "get_server_status",
    "get_ssh_signer_port",
    "get_token_broker_port",
    "get_vault_status",
    "install_vault_systemd",
    "installed_versions",
    "is_systemd_available",
    "is_vault_running",
    "is_vault_socket_active",
    "is_vault_systemd_available",
    "namespace_config_dir",
    "namespace_runtime_dir",
    "namespace_state_dir",
    "needs_setup",
    "podman_port_resolver",
    "read_config_section",
    "read_stamp",
    "sanitize_tty",
    "stage_line",
    "stamp_path",
    "start_vault",
    "stop_vault",
    "systemd_creds_has_tpm2",
    "uninstall_vault_systemd",
    "yaml_update_section",
]
