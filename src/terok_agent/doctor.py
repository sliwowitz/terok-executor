# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Agent-level container health checks.

Contributes domain-specific checks to the layered doctor protocol
(``terok_sandbox.doctor``): socat bridge liveness, credential file
integrity in shared mounts, and phantom token / base URL verification.

The checks are returned as :class:`DoctorCheck` specs — probe commands
+ evaluate callables — that the top-level orchestrator (``terok sickbay``)
executes inside containers via ``podman exec``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from terok_sandbox.doctor import CheckVerdict, DoctorCheck

if TYPE_CHECKING:
    from .roster import AgentRoster

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BRIDGE_PIDDIR = "/tmp/.terok"  # nosec B108 — matches ensure-bridges.sh in-container paths
_SSH_AGENT_PIDFILE = f"{_BRIDGE_PIDDIR}/ssh-agent.pid"
_SSH_AGENT_SOCKET = "/tmp/ssh-agent.sock"  # nosec B108
_GH_PROXY_PIDFILE = f"{_BRIDGE_PIDDIR}/gh-proxy.pid"
_GH_PROXY_SOCKET = "/tmp/terok-gh-proxy.sock"  # nosec B108

# Matches 32-character hexadecimal phantom tokens
_PHANTOM_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{32}$")

# Known real API key prefixes (obvious non-phantom patterns)
_REAL_KEY_PREFIXES = ("sk-ant-", "sk-", "gho_", "ghp_", "ghs_", "glpat-")


# ---------------------------------------------------------------------------
# Bridge checks
# ---------------------------------------------------------------------------


def _make_ssh_bridge_check() -> DoctorCheck:
    """Check that the SSH agent socat bridge is alive inside the container."""

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Evaluate bridge liveness probe."""
        if rc == 0:
            return CheckVerdict("ok", "SSH agent bridge alive (PID + socket)")
        return CheckVerdict(
            "error",
            "SSH agent bridge dead — socat process or socket missing",
            fixable=True,
        )

    return DoctorCheck(
        category="bridge",
        label="SSH agent bridge (socat)",
        probe_cmd=[
            "bash",
            "-c",
            f"kill -0 $(cat {_SSH_AGENT_PIDFILE} 2>/dev/null) 2>/dev/null"
            f" && test -S {_SSH_AGENT_SOCKET}",
        ],
        evaluate=_eval,
        fix_cmd=["bash", "-lc", "source ensure-bridges.sh"],
        fix_description="Re-source ensure-bridges.sh to restart the socat bridge.",
    )


def _make_gh_proxy_bridge_check() -> DoctorCheck:
    """Check that the gh credential proxy socat bridge is alive."""

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Evaluate bridge liveness probe."""
        if rc == 0:
            return CheckVerdict("ok", "gh proxy bridge alive (PID + socket)")
        return CheckVerdict(
            "error",
            "gh proxy bridge dead — socat process or socket missing",
            fixable=True,
        )

    return DoctorCheck(
        category="bridge",
        label="gh proxy bridge (socat)",
        probe_cmd=[
            "bash",
            "-c",
            f"kill -0 $(cat {_GH_PROXY_PIDFILE} 2>/dev/null) 2>/dev/null"
            f" && test -S {_GH_PROXY_SOCKET}",
        ],
        evaluate=_eval,
        fix_cmd=["bash", "-lc", "source ensure-bridges.sh"],
        fix_description="Re-source ensure-bridges.sh to restart the socat bridge.",
    )


# ---------------------------------------------------------------------------
# Credential file check (shared mount integrity)
# ---------------------------------------------------------------------------


def _make_credential_file_checks(roster: AgentRoster) -> list[DoctorCheck]:
    """Check known credential files in shared mounts for leaked real secrets."""
    checks: list[DoctorCheck] = []
    for name, route in roster.proxy_routes.items():
        if not route.credential_file:
            continue
        auth = roster.auth_providers.get(name)
        if not auth:
            continue

        container_path = f"{auth.container_mount}/{route.credential_file}"
        provider_name = name

        def _make_eval(pname: str, cpath: str):  # noqa: ANN202 — closure factory
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

    for name, route in roster.proxy_routes.items():
        # Collect all phantom env vars (api_key + oauth types)
        env_vars = list(route.phantom_env.keys()) + list(route.oauth_phantom_env.keys())
        for var in env_vars:
            if var in seen_vars:
                continue
            seen_vars.add(var)

            def _make_eval(env_var: str, pname: str):  # noqa: ANN202
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
                    # Unknown format — could be a new phantom token format or custom key
                    return CheckVerdict("ok", f"{env_var}: non-standard format ({pname})")

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


def _make_base_url_checks(roster: AgentRoster, proxy_port: int) -> list[DoctorCheck]:
    """Verify base URL env vars point to the credential proxy, not upstream."""
    checks: list[DoctorCheck] = []
    seen_vars: set[str] = set()
    expected_host = f"host.containers.internal:{proxy_port}"

    for name, route in roster.proxy_routes.items():
        if not route.base_url_env:
            continue
        var = route.base_url_env
        if var in seen_vars:
            continue
        seen_vars.add(var)

        def _make_eval(env_var: str, pname: str, host: str):  # noqa: ANN202
            """Create evaluate closure for a base URL check."""

            def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
                """Check if base URL points to proxy."""
                val = stdout.strip()
                if not val:
                    return CheckVerdict("warn", f"{env_var}: not set — proxy bypass possible")
                if urlparse(val).netloc == host:
                    return CheckVerdict("ok", f"{env_var}: routed through proxy ({pname})")
                return CheckVerdict(
                    "error",
                    f"{env_var}: points to {val!r}, not proxy — restart task",
                )

            return _eval

        checks.append(
            DoctorCheck(
                category="env",
                label=f"Base URL ({var})",
                probe_cmd=["printenv", var],
                evaluate=_make_eval(var, name, expected_host),
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def agent_doctor_checks(
    roster: AgentRoster,
    *,
    proxy_port: int | None = None,
) -> list[DoctorCheck]:
    """Return agent-level health checks for in-container diagnostics.

    Args:
        roster: The loaded agent roster.
        proxy_port: Credential proxy TCP port. Required for base URL checks;
            if ``None``, base URL checks are skipped.

    Returns:
        List of :class:`DoctorCheck` instances ready for orchestration.
    """
    checks: list[DoctorCheck] = [
        _make_ssh_bridge_check(),
        _make_gh_proxy_bridge_check(),
    ]
    checks.extend(_make_credential_file_checks(roster))
    checks.extend(_make_phantom_token_checks(roster))
    if proxy_port is not None:
        checks.extend(_make_base_url_checks(roster, proxy_port))
    return checks
