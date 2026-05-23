# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the ``terok_sandbox`` wheel.

Re-export catalog: every ``from terok_sandbox …`` import in
``terok_executor`` lives here.  The contract is enforced by
``.importlinter`` (``terok_sandbox`` is a protected module with
``terok_executor.integrations.sandbox`` as the sole allowed importer).

Cross-cutting helpers that originate in `terok_util` (the
[`CommandDef`][terok_util.cli_types.CommandDef] /
[`ArgDef`][terok_util.cli_types.ArgDef] /
[`CommandTree`][terok_util.cli_types.CommandTree] family,
[`namespace_state_dir`][terok_util.paths.namespace_state_dir] /
[`namespace_config_dir`][terok_util.paths.namespace_config_dir] /
[`namespace_runtime_dir`][terok_util.paths.namespace_runtime_dir],
[`ensure_dir`][terok_util.fs.ensure_dir] /
[`ensure_dir_writable`][terok_util.fs.ensure_dir_writable] /
[`write_sensitive_file`][terok_util.fs.write_sensitive_file],
[`ConfigStack`][terok_util.config_stack.ConfigStack] /
[`deep_merge`][terok_util.config_stack.deep_merge],
[`sanitize_tty`][terok_util.security.sanitize_tty],
[`podman_userns_args`][terok_util.podman.podman_userns_args]) are
imported directly from `terok_util` at every call site — they don't
flow through this adapter even when the same symbol also happens to
exist on ``terok_sandbox``.  This adapter owns the sandbox-specific
surface only.

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
    ConfigScope,
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
    needs_setup,
    podman_port_resolver,
    read_stamp,
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
    _handle_sandbox_setup,
    _handle_sandbox_uninstall,
)
from terok_sandbox.doctor import CheckVerdict, DoctorCheck  # noqa: F401 — re-exported public API

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
    "CheckVerdict",
    "ConfigScope",
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
    "needs_setup",
    "podman_port_resolver",
    "read_stamp",
    "stage_line",
    "stamp_path",
    "start_vault",
    "stop_vault",
    "systemd_creds_has_tpm2",
    "uninstall_vault_systemd",
    "yaml_update_section",
]
