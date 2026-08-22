# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Agent-level container health checks (implementation).

Contributes domain-specific checks to the layered doctor protocol
(``terok_sandbox.doctor``): socat bridge liveness, credential file
integrity in shared mounts, and phantom token / base URL verification
for the vault.

The checks are returned as `DoctorCheck` specs — probe commands
+ evaluate callables — that the top-level orchestrator (``terok sickbay``)
executes inside containers via ``podman exec``.

Public entry point: [`AgentRoster.doctor_checks`][terok_executor.roster.loader.AgentRoster.doctor_checks].
This module is the implementation home; consumers call it through the
roster method so the dependency direction stays roster → checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from terok_executor.integrations.sandbox import CheckVerdict, DoctorCheck

from .vault_addr import (
    CONTAINER_VAULT_SOCKET,
    LOOPBACK_BRIDGE_SOCKET,
    LOOPBACK_VAULT_PORT,
)

if TYPE_CHECKING:
    from .roster import AgentRoster

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BRIDGE_PIDDIR = "/tmp/.terok"  # nosec B108 — matches ensure-bridges.sh in-container paths
_SSH_AGENT_PIDFILE = f"{_BRIDGE_PIDDIR}/ssh-agent.pid"
_SSH_AGENT_SOCKET = "/tmp/ssh-agent.sock"  # nosec B108
_VAULT_LOOPBACK_PIDFILE = f"{_BRIDGE_PIDDIR}/vault-loopback.pid"
_VAULT_SOCKET_PIDFILE = f"{_BRIDGE_PIDDIR}/vault-socket.pid"

# ``ensure-bridges.sh`` writes a bridge's PID file only inside the branch that
# actually starts it, so a missing PID file means the bridge is *intentionally*
# absent (its start condition was not met), not dead.  The guarded probe prints
# this marker on that path; ``_bridge_eval`` reads it back and reports the
# bridge as an expected absence instead of a failure.
_BRIDGE_ABSENT_MARKER = "terok-bridge-not-expected"

# Matches phantom tokens: "terok-p-" prefix + 32 hex chars
_PHANTOM_TOKEN_RE = re.compile(r"^terok-p-[0-9a-fA-F]{32}$")

# Known real API key prefixes (obvious non-phantom patterns)
_REAL_KEY_PREFIXES = ("sk-ant-", "sk-", "gho_", "ghp_", "ghs_", "glpat-")


# ── Public API ───────────────────────────────────────────────────────────


def _build_agent_doctor_checks(
    roster: AgentRoster,
    *,
    token_broker_port: int | None = None,
) -> list[DoctorCheck]:
    """Assemble agent-level health checks for in-container diagnostics.

    Private to the doctor module — call through
    [`AgentRoster.doctor_checks`][terok_executor.roster.loader.AgentRoster.doctor_checks].

    Args:
        roster: The loaded agent roster.
        token_broker_port: Host-side vault broker TCP port.  ``None``
            selects socket mode; any integer selects TCP mode.  Base URL
            checks use the port (or the in-container loopback port) to
            derive the expected host.
    """
    socket_mode = token_broker_port is None
    checks: list[DoctorCheck] = [
        _make_ssh_bridge_check(),
        _make_vault_bridge_check(socket_mode=socket_mode),
    ]
    checks.extend(_make_credential_file_checks(roster))
    checks.extend(_make_phantom_token_checks(roster))
    checks.extend(_make_base_url_checks(roster, token_broker_port))
    return checks


# ── Check factories (in assembly order) ─────────────────────────────────


def _guarded_probe(pidfile: str, liveness: str) -> list[str]:
    """Wrap a *liveness* test so a missing *pidfile* reads as an expected absence.

    ``ensure-bridges.sh`` writes a bridge's PID file only inside the branch
    that starts it, so the PID file is the shell's own record of "I started
    this bridge".  A task with no vault-routed provider never starts the
    vault bridge, and a task with no signer token never starts the SSH
    bridge — neither writes its PID file.  A bare liveness probe would read
    that legitimate absence as a dead bridge.  When *pidfile* is missing the
    guard prints [`_BRIDGE_ABSENT_MARKER`][terok_executor.doctor._BRIDGE_ABSENT_MARKER]
    and exits clean; otherwise it runs *liveness*.
    """
    return [
        "bash",
        "-c",
        f'[ -f "{pidfile}" ] || {{ echo {_BRIDGE_ABSENT_MARKER}; exit 0; }}; {liveness}',
    ]


def _bridge_eval(
    *, alive_detail: str, dead_detail: str, absent_detail: str
) -> Callable[[int, str, str], CheckVerdict]:
    """Build the evaluator shared by the guarded bridge probes.

    The absence marker wins over the return code: a guard that short-
    circuited exits ``0`` too, so the marker is the only way to tell an
    expected absence (*absent_detail*, ``ok``) from a live bridge
    (*alive_detail*, ``ok``) or a dead one (*dead_detail*, ``error``).
    """

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Evaluate a guarded bridge liveness probe."""
        if _BRIDGE_ABSENT_MARKER in stdout:
            return CheckVerdict("ok", absent_detail)
        if rc == 0:
            return CheckVerdict("ok", alive_detail)
        return CheckVerdict("error", dead_detail, fixable=True)

    return _eval


def _make_ssh_bridge_check() -> DoctorCheck:
    """Check that the SSH signer socat bridge is alive — when the task has a signer."""
    return DoctorCheck(
        category="bridge",
        label="SSH agent bridge (socat)",
        probe_cmd=_guarded_probe(
            _SSH_AGENT_PIDFILE,
            f"kill -0 $(cat {_SSH_AGENT_PIDFILE} 2>/dev/null) 2>/dev/null"
            f" && test -S {_SSH_AGENT_SOCKET}",
        ),
        evaluate=_bridge_eval(
            alive_detail="SSH agent bridge alive (PID + socket)",
            dead_detail="SSH agent bridge dead — socat process or socket missing",
            absent_detail="SSH agent bridge not started — no signer token for this task",
        ),
        fix_cmd=["bash", "-lc", "source ensure-bridges.sh"],
        fix_description="Re-source ensure-bridges.sh to restart the socat bridge.",
    )


def _make_vault_bridge_check(*, socket_mode: bool) -> DoctorCheck:
    """Check the vault-side socat bridge for the active transport.

    In socket mode the bridge exposes the mounted host socket as a TCP
    loopback so HTTP-only clients can reach it.  In TCP mode the bridge
    exposes a local Unix socket that tunnels to the host broker over TCP
    (for socket-only clients).
    """
    if socket_mode:
        label = f"Vault loopback bridge (TCP → {CONTAINER_VAULT_SOCKET})"
        pidfile = _VAULT_LOOPBACK_PIDFILE
        liveness = (
            f"test -S {CONTAINER_VAULT_SOCKET} && kill -0 $(cat {pidfile} 2>/dev/null) 2>/dev/null"
        )
        dead_detail = "Vault loopback bridge dead — HTTP clients cannot reach the mounted socket"
    else:
        label = "Vault socket bridge (/tmp/terok-vault.sock → broker TCP)"
        pidfile = _VAULT_SOCKET_PIDFILE
        liveness = (
            f"kill -0 $(cat {pidfile} 2>/dev/null) 2>/dev/null && test -S {LOOPBACK_BRIDGE_SOCKET}"
        )
        dead_detail = "Vault socket bridge dead — socat process or socket missing"

    return DoctorCheck(
        category="bridge",
        label=label,
        probe_cmd=_guarded_probe(pidfile, liveness),
        evaluate=_bridge_eval(
            alive_detail=f"{label} alive",
            dead_detail=dead_detail,
            absent_detail=f"{label} not started — no vault-routed provider for this task",
        ),
        fix_cmd=["bash", "-lc", "source ensure-bridges.sh"],
        fix_description="Re-source ensure-bridges.sh to restart the socat bridge.",
    )


# ---------------------------------------------------------------------------
# Credential file check (shared mount integrity)
# ---------------------------------------------------------------------------


def _make_credential_file_checks(roster: AgentRoster) -> list[DoctorCheck]:
    """Check known credential files in shared mounts for leaked real secrets."""
    checks: list[DoctorCheck] = []
    for name, route in roster.vault_routes.items():
        if not route.credential_file:
            continue
        auth = roster.auth_providers.get(name)
        if not auth:
            continue

        container_path = f"{auth.container_mount}/{route.credential_file}"
        provider_name = name

        def _make_eval(pname: str, cpath: str) -> Callable[[int, str, str], CheckVerdict]:
            """Create an evaluate closure for a specific provider."""

            def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
                """Check if file contains phantom tokens or real secrets."""
                if rc != 0:
                    if re.search(r"no such file", stderr, re.IGNORECASE):
                        return CheckVerdict("ok", f"{pname}: no credential file (clean)")
                    return CheckVerdict(
                        "warn",
                        f"{pname}: cannot read {cpath} — {stderr.strip() or 'unknown error'}",
                    )
                content = stdout.strip()
                if not content:
                    return CheckVerdict("ok", f"{pname}: credential file empty")
                # Check for real API key patterns in content
                for prefix in _REAL_KEY_PREFIXES:
                    if prefix in content:
                        return CheckVerdict(
                            "error",
                            f"{pname}: real API key detected in {cpath}",
                            fixable=True,
                        )
                return CheckVerdict("ok", f"{pname}: credential file looks clean")

            return _eval

        checks.append(
            DoctorCheck(
                category="mount",
                label=f"Credential file ({name})",
                probe_cmd=["cat", container_path],
                evaluate=_make_eval(provider_name, container_path),
                fix_cmd=["rm", "-f", container_path],
                fix_description=f"Remove leaked credential file {container_path}.",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Phantom token integrity
# ---------------------------------------------------------------------------


def _make_phantom_token_checks(roster: AgentRoster) -> list[DoctorCheck]:
    """Verify that API key env vars contain phantom tokens, not real keys."""
    checks: list[DoctorCheck] = []
    seen_vars: set[str] = set()

    for name, route in roster.vault_routes.items():
        # Collect all phantom-token env var names (deduped downstream)
        env_vars = list(route.token_env.values())
        for var in env_vars:
            if var in seen_vars:
                continue
            seen_vars.add(var)

            def _make_eval(env_var: str, pname: str) -> Callable[[int, str, str], CheckVerdict]:
                """Create evaluate closure for a specific env var."""

                def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
                    """Check if env var looks like a phantom token."""
                    val = stdout.strip()
                    if rc != 0 or not val:
                        hint = "not set" if rc != 0 else "empty"
                        return CheckVerdict("warn", f"{env_var}: {hint}")
                    if _PHANTOM_TOKEN_RE.match(val):
                        return CheckVerdict("ok", f"{env_var}: phantom token ({pname})")
                    for prefix in _REAL_KEY_PREFIXES:
                        if val.startswith(prefix):
                            return CheckVerdict(
                                "error",
                                f"{env_var}: real API key detected — restart task",
                            )
                    # Unknown format — not a recognised phantom token
                    return CheckVerdict("warn", f"{env_var}: unrecognised token format ({pname})")

                return _eval

            checks.append(
                DoctorCheck(
                    category="env",
                    label=f"Phantom token ({var})",
                    probe_cmd=["printenv", var],
                    evaluate=_make_eval(var, name),
                )
            )
    return checks


# ---------------------------------------------------------------------------
# Base URL override checks
# ---------------------------------------------------------------------------


def _make_base_url_checks(roster: AgentRoster, token_broker_port: int | None) -> list[DoctorCheck]:
    """Verify base URL env vars point to the vault, not upstream.

    Under socket transport the base URL points at the in-container
    loopback bridge (``localhost:<LOOPBACK_VAULT_PORT>``); under TCP
    transport it points at ``host.containers.internal:<broker_port>``.
    """
    checks: list[DoctorCheck] = []
    seen_vars: set[str] = set()
    if token_broker_port is None:
        expected_host = f"localhost:{LOOPBACK_VAULT_PORT}"
        mode_label = "vault loopback"
    else:
        expected_host = f"host.containers.internal:{token_broker_port}"
        mode_label = "vault broker"

    for name, route in roster.vault_routes.items():
        if not route.base_url_env:
            continue
        var = route.base_url_env
        if var in seen_vars:
            continue
        seen_vars.add(var)

        def _make_eval(
            env_var: str, pname: str, host: str, mode: str
        ) -> Callable[[int, str, str], CheckVerdict]:
            """Create evaluate closure for a base URL check."""

            def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
                """Check if base URL points to the active vault endpoint."""
                val = stdout.strip()
                if not val:
                    # Just "not set" — the original "vault bypass possible"
                    # was misleading: an agent can override the env var
                    # any time, so this isn't a bypass *signal*; it just
                    # means the agent doesn't know where the vault lives.
                    return CheckVerdict("warn", f"{env_var}: not set")
                if urlparse(val).netloc == host:
                    return CheckVerdict("ok", f"{env_var}: routed through {mode} ({pname})")
                return CheckVerdict(
                    "error",
                    f"{env_var}: points to {val!r}, not {mode} — restart task",
                )

            return _eval

        checks.append(
            DoctorCheck(
                category="env",
                label=f"Base URL ({var})",
                probe_cmd=["printenv", var],
                evaluate=_make_eval(var, name, expected_host, mode_label),
            )
        )
    return checks
