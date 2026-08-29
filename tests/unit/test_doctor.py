# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent-level container health checks."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from terok_sandbox.doctor import DoctorCheck

from terok_executor.doctor import (
    _BRIDGE_ABSENT_MARKER,
    _PHANTOM_TOKEN_RE,
    _make_base_url_checks,
    _make_credential_file_checks,
    _make_gate_bridge_check,
    _make_phantom_token_checks,
    _make_ssh_bridge_check,
    _make_vault_bridge_check,
    _socat_alive,
)
from terok_executor.integrations.sandbox import CONTAINER_VAULT_SOCKET
from terok_executor.roster import AgentRoster

TOKEN_BROKER_PORT = 18731
#: The vault loopback bridge's listen address, as ``ensure-bridges.sh`` binds it.
VAULT_LISTEN_ADDRESS = "TCP-LISTEN:9419"
VAULT_LISTEN_SPEC = f"{VAULT_LISTEN_ADDRESS},bind=127.0.0.1,fork,reuseaddr"

#: The git gate bridge's listen address, as ``ensure-bridges.sh`` binds it.
GATE_LISTEN_SPEC = "TCP-LISTEN:9418,fork,reuseaddr"


@contextmanager
def _spawned(*argv: str) -> Iterator[int]:
    """Yield the PID of a live sleeper carrying *argv* on its command line."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", *argv])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait()


def _gate_pidfile() -> str:
    """The container-side gate PID file the probe is generated against."""
    from terok_executor.doctor import _GATE_PIDFILE

    return _GATE_PIDFILE


def _run(fragment: str) -> bool:
    """Return whether the generated shell *fragment* succeeds."""
    return subprocess.run(["bash", "-c", fragment], check=False).returncode == 0


class TestSocatLiveness:
    """``_socat_alive`` must identify the bridge, not merely a live PID."""

    def test_recycled_pid_is_not_the_bridge(self, tmp_path: Path) -> None:
        """A restarted container's PID collision must not report a healthy bridge.

        Container PIDs restart from 1 and ``/tmp/.terok`` survives the restart,
        so a stale PID file names whatever inherited its number.  A signal probe
        called that alive and the doctor reported a dead vault loopback bridge
        as green while every URL-transport client saw ECONNREFUSED.
        """
        pidfile = tmp_path / "vault-loopback.pid"
        with _spawned() as pid:
            pidfile.write_text(str(pid))
            assert not _run(_socat_alive(str(pidfile), VAULT_LISTEN_ADDRESS))

    def test_matching_command_line_is_the_bridge(self, tmp_path: Path) -> None:
        """A process whose command line carries the listen spec is the bridge."""
        pidfile = tmp_path / "vault-loopback.pid"
        with _spawned(VAULT_LISTEN_SPEC) as pid:
            pidfile.write_text(str(pid))
            assert _run(_socat_alive(str(pidfile), VAULT_LISTEN_ADDRESS))

    def test_empty_pidfile_never_reads_the_kernel_command_line(self, tmp_path: Path) -> None:
        """An empty PID file is dead, not a match against ``/proc//cmdline``."""
        pidfile = tmp_path / "vault-loopback.pid"
        pidfile.write_text("")
        assert not _run(_socat_alive(str(pidfile), VAULT_LISTEN_ADDRESS))

    def test_exited_pid_is_dead(self, tmp_path: Path) -> None:
        """A PID with no ``/proc`` entry left is not a bridge."""
        proc = subprocess.Popen([sys.executable, "-c", ""])
        proc.wait()
        pidfile = tmp_path / "vault-loopback.pid"
        pidfile.write_text(str(proc.pid))
        assert not _run(_socat_alive(str(pidfile), VAULT_LISTEN_ADDRESS))

    def test_probes_name_the_listen_spec_the_script_uses(self) -> None:
        """The doctor's needle must track ``ensure-bridges.sh``'s socat listeners."""
        socket_probe = " ".join(_make_vault_bridge_check(socket_mode=True).probe_cmd)
        tcp_probe = " ".join(_make_vault_bridge_check(socket_mode=False).probe_cmd)
        ssh_probe = " ".join(_make_ssh_bridge_check().probe_cmd)
        assert f"{VAULT_LISTEN_ADDRESS}," in socket_probe
        assert "UNIX-LISTEN:/tmp/terok-vault.sock," in tcp_probe
        assert "UNIX-LISTEN:/tmp/ssh-agent.sock," in ssh_probe


class TestBridgeTargets:
    """A bridge is only healthy when the far end it dials exists."""

    def _probe(self, check: DoctorCheck, pidfile: Path, env: dict[str, str]) -> bool:
        """Run *check*'s probe with its pidfile redirected into a temp path."""
        script = check.probe_cmd[-1].replace(_gate_pidfile(), str(pidfile))
        completed = subprocess.run(
            ["bash", "-c", script], env={**os.environ, **env}, check=False, capture_output=True
        )
        return completed.returncode == 0 and _BRIDGE_ABSENT_MARKER not in completed.stdout.decode()

    def test_gate_bridge_aimed_at_an_absent_socket_is_not_healthy(self, tmp_path: Path) -> None:
        """The failure that reads as a cold-start race, caught for what it is.

        A container older than the current ``/run/terok`` layout advertises a
        gate socket nothing binds.  The bridge starts, listens and accepts, so
        a liveness-only probe passes; git then hangs for socat's retry budget
        and reports an empty reply.
        """
        pidfile = tmp_path / "gate.pid"
        with _spawned(GATE_LISTEN_SPEC) as pid:
            pidfile.write_text(str(pid))
            stale = {"TEROK_GATE_SOCKET": "/run/terok/gate-server.sock"}
            assert not self._probe(_make_gate_bridge_check(), pidfile, stale)

    def test_gate_bridge_with_a_bound_socket_is_healthy(self, tmp_path: Path) -> None:
        """Listener plus a real target is the only passing shape in socket mode."""
        target = tmp_path / "gate.sock"
        with socket.socket(socket.AF_UNIX) as srv:
            srv.bind(str(target))
            pidfile = tmp_path / "gate.pid"
            with _spawned(GATE_LISTEN_SPEC) as pid:
                pidfile.write_text(str(pid))
                env = {"TEROK_GATE_SOCKET": str(target)}
                assert self._probe(_make_gate_bridge_check(), pidfile, env)

    def test_gate_bridge_in_tcp_mode_needs_no_socket(self, tmp_path: Path) -> None:
        """TCP mode dials a host port; there is no path to test."""
        pidfile = tmp_path / "gate.pid"
        with _spawned(GATE_LISTEN_SPEC) as pid:
            pidfile.write_text(str(pid))
            assert self._probe(_make_gate_bridge_check(), pidfile, {"TEROK_GATE_SOCKET": ""})

    def test_vault_probe_tests_the_advertised_socket(self) -> None:
        """The vault probe must dial ``$TEROK_VAULT_SOCKET``, not this host's constant.

        Testing the constant would report a stale container's dead bridge as
        healthy — the container dials what it was told, not what we would tell
        it today.
        """
        probe = " ".join(_make_vault_bridge_check(socket_mode=True).probe_cmd)
        assert f'"${{TEROK_VAULT_SOCKET:-{CONTAINER_VAULT_SOCKET}}}"' in probe

    def test_gate_check_is_part_of_the_battery(self) -> None:
        """A dead gate bridge was invisible to the doctor until now."""
        checks = AgentRoster.load().doctor_checks()
        assert any("Git gate bridge" in c.label for c in checks)


class TestSSHBridgeCheck:
    """SSH signer socat bridge liveness check."""

    def test_ok_when_alive(self) -> None:
        check = _make_ssh_bridge_check()
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"

    def test_error_when_dead(self) -> None:
        check = _make_ssh_bridge_check()
        verdict = check.evaluate(1, "", "")
        assert verdict.severity == "error"
        assert verdict.fixable is True

    def test_has_fix_cmd(self) -> None:
        check = _make_ssh_bridge_check()
        assert check.fix_cmd is not None
        assert "ensure-bridges.sh" in " ".join(check.fix_cmd)

    def test_category_is_bridge(self) -> None:
        check = _make_ssh_bridge_check()
        assert check.category == "bridge"

    def test_ok_when_bridge_intentionally_absent(self) -> None:
        # No signer token → bridge never starts; the guard emits the marker.
        check = _make_ssh_bridge_check()
        verdict = check.evaluate(0, f"{_BRIDGE_ABSENT_MARKER}\n", "")
        assert verdict.severity == "ok"
        assert "no signer token" in verdict.detail

    def test_probe_guards_on_pidfile(self) -> None:
        check = _make_ssh_bridge_check()
        probe = " ".join(check.probe_cmd)
        assert "[ -f " in probe and "ssh-agent.pid" in probe


class TestVaultBridgeCheck:
    """Vault socat bridge liveness check — two probe shapes, one semantic role."""

    def test_socket_mode_ok_when_alive(self) -> None:
        check = _make_vault_bridge_check(socket_mode=True)
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"

    def test_socket_mode_error_when_dead(self) -> None:
        check = _make_vault_bridge_check(socket_mode=True)
        verdict = check.evaluate(1, "", "")
        assert verdict.severity == "error"
        assert verdict.fixable is True

    def test_tcp_mode_ok_when_alive(self) -> None:
        check = _make_vault_bridge_check(socket_mode=False)
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"

    def test_tcp_mode_error_when_dead(self) -> None:
        check = _make_vault_bridge_check(socket_mode=False)
        verdict = check.evaluate(1, "", "")
        assert verdict.severity == "error"
        assert verdict.fixable is True

    def test_socket_mode_probes_mounted_vault_socket(self) -> None:
        check = _make_vault_bridge_check(socket_mode=True)
        probe = " ".join(check.probe_cmd)
        assert CONTAINER_VAULT_SOCKET in probe
        assert "vault-loopback.pid" in probe

    def test_tcp_mode_probes_local_bridge_socket(self) -> None:
        check = _make_vault_bridge_check(socket_mode=False)
        probe = " ".join(check.probe_cmd)
        assert "/tmp/terok-vault.sock" in probe
        assert "vault-socket.pid" in probe

    def test_has_fix_cmd(self) -> None:
        check = _make_vault_bridge_check(socket_mode=True)
        assert check.fix_cmd is not None

    def test_category_is_bridge(self) -> None:
        check = _make_vault_bridge_check(socket_mode=True)
        assert check.category == "bridge"

    def test_socket_mode_ok_when_no_routed_provider(self) -> None:
        # A task with nothing vault-routed never starts the loopback bridge;
        # the guard reports an expected absence, not a dead bridge.
        check = _make_vault_bridge_check(socket_mode=True)
        verdict = check.evaluate(0, f"{_BRIDGE_ABSENT_MARKER}\n", "")
        assert verdict.severity == "ok"
        assert "no vault-routed provider" in verdict.detail

    def test_tcp_mode_ok_when_no_routed_provider(self) -> None:
        check = _make_vault_bridge_check(socket_mode=False)
        verdict = check.evaluate(0, f"{_BRIDGE_ABSENT_MARKER}\n", "")
        assert verdict.severity == "ok"

    def test_both_modes_guard_on_their_pidfile(self) -> None:
        for socket_mode, pidfile in ((True, "vault-loopback.pid"), (False, "vault-socket.pid")):
            check = _make_vault_bridge_check(socket_mode=socket_mode)
            probe = " ".join(check.probe_cmd)
            assert "[ -f " in probe and pidfile in probe


class TestCredentialFileChecks:
    """Known credential file leak detection."""

    def test_generates_checks_for_routed_providers(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_credential_file_checks(roster)
        # Should have at least one check for providers with credential_file
        providers_with_cred = [
            n
            for n, r in roster.vault_routes.items()
            if r.credential_file and n in roster.auth_providers
        ]
        assert len(checks) == len(providers_with_cred)

    def test_clean_when_file_missing(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_credential_file_checks(roster)
        if checks:
            # rc != 0 with "No such file" stderr means file doesn't exist
            verdict = checks[0].evaluate(1, "", "cat: /path: No such file or directory\n")
            assert verdict.severity == "ok"

    def test_warn_on_permission_denied(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_credential_file_checks(roster)
        if checks:
            verdict = checks[0].evaluate(1, "", "cat: /path: Permission denied\n")
            assert verdict.severity == "warn"
            assert "Permission denied" in verdict.detail

    def test_error_on_real_key(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_credential_file_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, '{"api_key": "sk-ant-real-key"}', "")
            assert verdict.severity == "error"
            assert verdict.fixable is True

    def test_clean_on_empty_file(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_credential_file_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, "", "")
            assert verdict.severity == "ok"


class TestPhantomTokenChecks:
    """Phantom token integrity verification."""

    def test_phantom_token_regex(self) -> None:
        assert _PHANTOM_TOKEN_RE.match("terok-p-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
        assert not _PHANTOM_TOKEN_RE.match("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
        assert not _PHANTOM_TOKEN_RE.match("sk-ant-something")
        assert not _PHANTOM_TOKEN_RE.match("too-short")

    def test_generates_checks_for_env_vars(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_phantom_token_checks(roster)
        # Should have at least some checks
        assert len(checks) > 0

    def test_ok_for_phantom_token(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_phantom_token_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, "terok-p-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4\n", "")
            assert verdict.severity == "ok"

    def test_warn_for_unrecognised_format(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_phantom_token_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4\n", "")
            assert verdict.severity == "warn"
            assert "unrecognised" in verdict.detail

    def test_error_for_real_key(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_phantom_token_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, "sk-ant-api03-real-key-here\n", "")
            assert verdict.severity == "error"

    def test_warn_when_unset(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_phantom_token_checks(roster)
        if checks:
            # rc=1 from printenv means var is unset
            verdict = checks[0].evaluate(1, "", "")
            assert verdict.severity == "warn"

    def test_no_duplicate_env_vars(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_phantom_token_checks(roster)
        env_vars = [" ".join(c.probe_cmd) for c in checks]
        assert len(env_vars) == len(set(env_vars)), "duplicate env var checks"


class TestBaseUrlChecks:
    """Base URL override verification."""

    def test_generates_checks_tcp(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_base_url_checks(roster, TOKEN_BROKER_PORT)
        assert len(checks) > 0

    def test_generates_checks_socket(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_base_url_checks(roster, None)
        assert len(checks) > 0

    def test_tcp_ok_when_routed(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_base_url_checks(roster, TOKEN_BROKER_PORT)
        if checks:
            verdict = checks[0].evaluate(
                0, f"http://host.containers.internal:{TOKEN_BROKER_PORT}\n", ""
            )
            assert verdict.severity == "ok"

    def test_socket_ok_when_routed(self) -> None:
        from terok_executor.vault_addr import LOOPBACK_VAULT_PORT

        roster = AgentRoster.shared()
        checks = _make_base_url_checks(roster, None)
        if checks:
            verdict = checks[0].evaluate(0, f"http://localhost:{LOOPBACK_VAULT_PORT}\n", "")
            assert verdict.severity == "ok"

    def test_error_when_bypassed(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_base_url_checks(roster, TOKEN_BROKER_PORT)
        if checks:
            verdict = checks[0].evaluate(0, "https://api.anthropic.com\n", "")
            assert verdict.severity == "error"

    def test_warn_when_unset(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_base_url_checks(roster, TOKEN_BROKER_PORT)
        if checks:
            verdict = checks[0].evaluate(0, "", "")
            assert verdict.severity == "warn"

    def test_no_duplicate_vars(self) -> None:
        roster = AgentRoster.shared()
        checks = _make_base_url_checks(roster, TOKEN_BROKER_PORT)
        vars_checked = [" ".join(c.probe_cmd) for c in checks]
        assert len(vars_checked) == len(set(vars_checked))


class TestAgentDoctorChecks:
    """Integration: AgentRoster.doctor_checks() assembly."""

    def test_includes_bridge_checks(self) -> None:
        roster = AgentRoster.shared()
        checks = roster.doctor_checks(token_broker_port=TOKEN_BROKER_PORT)
        categories = {c.category for c in checks}
        assert "bridge" in categories

    def test_includes_mount_checks(self) -> None:
        roster = AgentRoster.shared()
        checks = roster.doctor_checks(token_broker_port=TOKEN_BROKER_PORT)
        categories = {c.category for c in checks}
        assert "mount" in categories

    def test_includes_env_checks(self) -> None:
        roster = AgentRoster.shared()
        checks = roster.doctor_checks(token_broker_port=TOKEN_BROKER_PORT)
        categories = {c.category for c in checks}
        assert "env" in categories

    def test_base_url_checks_emitted_in_both_modes(self) -> None:
        """Socket and TCP mode both need base-URL checks — the probe host differs."""
        roster = AgentRoster.shared()
        checks_tcp = roster.doctor_checks(token_broker_port=TOKEN_BROKER_PORT)
        checks_socket = roster.doctor_checks(token_broker_port=None)
        base_url_tcp = [c for c in checks_tcp if "Base URL" in c.label]
        base_url_socket = [c for c in checks_socket if "Base URL" in c.label]
        assert len(base_url_tcp) > 0
        assert len(base_url_socket) > 0

    def test_all_are_doctor_check_instances(self) -> None:
        roster = AgentRoster.shared()
        checks = roster.doctor_checks(token_broker_port=TOKEN_BROKER_PORT)
        for check in checks:
            assert isinstance(check, DoctorCheck)
