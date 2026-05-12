# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""First-run preflight contract: what's mandatory, what's optional, what's silent.

Mandatory checks (podman, sandbox services, images) block the launch
when still unmet after the interactive offer; optional checks
(SSH key, credentials) never block — they print a consequence and the
launch proceeds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from terok_executor.preflight import (
    CheckResult,
    check_credentials,
    check_images,
    check_podman,
    check_sandbox_services,
    check_shield,
    check_ssh_key,
    run_preflight,
)

# ── Individual checks ────────────────────────────────────────────────


@patch("terok_executor.preflight.subprocess.run")
@patch("terok_executor.preflight.shutil.which", return_value="/usr/bin/podman")
def test_podman_ok(_which: MagicMock, _run: MagicMock) -> None:
    """Podman found and responds → ok."""
    assert check_podman().ok is True


@patch("terok_executor.preflight.shutil.which", return_value=None)
def test_podman_missing(_which: MagicMock) -> None:
    """Podman not on PATH → fail."""
    r = check_podman()
    assert r.ok is False
    assert "not found" in r.message


# ── Sandbox services aggregate ───────────────────────────────────────


@patch("terok_sandbox.get_server_status")
@patch("terok_sandbox.check_environment")
@patch("terok_sandbox.is_vault_running", return_value=True)
@patch("terok_sandbox.is_vault_socket_active", return_value=False)
def test_sandbox_services_ok(
    _sock: MagicMock,
    _run: MagicMock,
    mock_env: MagicMock,
    mock_status: MagicMock,
) -> None:
    """All three (shield, vault, gate) ready → ok."""
    mock_env.return_value = MagicMock(health="ok")
    mock_status.return_value = MagicMock(mode="systemd")
    assert check_sandbox_services().ok is True


@patch("terok_sandbox.get_server_status")
@patch("terok_sandbox.check_environment")
@patch("terok_sandbox.is_vault_running", return_value=False)
@patch("terok_sandbox.is_vault_socket_active", return_value=False)
def test_sandbox_services_lists_missing(
    _sock: MagicMock,
    _run: MagicMock,
    mock_env: MagicMock,
    mock_status: MagicMock,
) -> None:
    """Missing items are all named in the same check's message."""
    mock_env.return_value = MagicMock(health="setup-needed")
    mock_status.return_value = MagicMock(mode=None)
    r = check_sandbox_services()
    assert r.ok is False
    for expected in ("vault", "shield", "gate"):
        assert expected in r.message


# ── check_credentials ────────────────────────────────────────────────


@patch("terok_sandbox.config.SandboxConfig.open_credential_db")
def test_credentials_found(mock_db_cls: MagicMock) -> None:
    """Credentials stored → ok."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = {"type": "api_key", "value": "sk-test"}
    assert check_credentials("claude").ok is True
    db.close.assert_called_once()


@patch("terok_sandbox.config.SandboxConfig.open_credential_db")
def test_credentials_missing(mock_db_cls: MagicMock) -> None:
    """No credentials → fail."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = None
    r = check_credentials("claude")
    assert r.ok is False
    assert "not found" in r.message
    db.close.assert_called_once()


@patch("terok_sandbox.config.SandboxConfig.open_credential_db", side_effect=Exception("db error"))
def test_credentials_db_unavailable(_cls: MagicMock) -> None:
    """DB open fails → fail with message."""
    r = check_credentials("claude")
    assert r.ok is False
    assert "unavailable" in r.message


# ── check_ssh_key ────────────────────────────────────────────────────


@patch("terok_sandbox.config.SandboxConfig.open_credential_db")
def test_ssh_key_present(mock_db_cls: MagicMock) -> None:
    """Existing key in scope → ok."""
    db = mock_db_cls.return_value
    db.list_ssh_keys_for_scope.return_value = [MagicMock()]
    r = check_ssh_key()
    assert r.ok is True
    db.close.assert_called_once()


@patch("terok_sandbox.config.SandboxConfig.open_credential_db")
def test_ssh_key_absent(mock_db_cls: MagicMock) -> None:
    """Empty scope → fail."""
    db = mock_db_cls.return_value
    db.list_ssh_keys_for_scope.return_value = []
    r = check_ssh_key()
    assert r.ok is False


# ── check_images ─────────────────────────────────────────────────────


@patch("terok_executor.preflight.subprocess.run")
def test_images_exist(mock_run: MagicMock) -> None:
    """Image exists → ok."""
    mock_run.return_value = MagicMock(returncode=0)
    assert check_images("ubuntu:24.04").ok is True


@patch("terok_executor.preflight.subprocess.run")
def test_images_missing(mock_run: MagicMock) -> None:
    """Image doesn't exist → fail."""
    mock_run.return_value = MagicMock(returncode=1)
    assert check_images("ubuntu:24.04").ok is False


# ── check_shield ─────────────────────────────────────────────────────


@patch("terok_sandbox.check_environment")
def test_shield_ok(mock_env: MagicMock) -> None:
    """Shield active → ok."""
    mock_env.return_value = MagicMock(health="ok")
    assert check_shield().ok is True


@patch("terok_sandbox.check_environment")
def test_shield_missing(mock_env: MagicMock) -> None:
    """Shield not installed → fail (informational)."""
    mock_env.return_value = MagicMock(health="setup-needed")
    r = check_shield()
    assert r.ok is False
    assert "unrestricted" in r.message


# ── run_preflight orchestration ──────────────────────────────────────


def _patch_all_ok():
    """Patch every individual check to report ok."""
    return [
        patch(
            "terok_executor.preflight.check_podman",
            return_value=CheckResult("podman", True, "ok"),
        ),
        patch(
            "terok_executor.preflight.check_sandbox_services",
            return_value=CheckResult("sandbox services", True, "ready"),
        ),
        patch(
            "terok_executor.preflight.check_images",
            return_value=CheckResult("container images", True, "ready"),
        ),
        patch(
            "terok_executor.preflight.check_ssh_key",
            return_value=CheckResult("ssh key", True, "present"),
        ),
        patch(
            "terok_executor.preflight.check_credentials",
            return_value=CheckResult("claude credentials", True, "stored"),
        ),
        patch(
            "terok_executor.preflight.check_shield",
            return_value=CheckResult("shield", True, "ok"),
        ),
    ]


class TestRunPreflight:
    """Orchestration contract — mandatory vs optional, TTY vs --yes."""

    def test_all_ok(self) -> None:
        """Every check passes → returns True."""
        with (
            _patch_all_ok()[0],
            _patch_all_ok()[1],
            _patch_all_ok()[2],
            _patch_all_ok()[3],
            _patch_all_ok()[4],
            _patch_all_ok()[5],
        ):
            assert run_preflight("claude", interactive=False) is True

    @patch(
        "terok_executor.preflight.check_podman",
        return_value=CheckResult("podman", False, "nope"),
    )
    def test_podman_missing_bails_immediately(self, _pod: MagicMock) -> None:
        """Podman is a hard stop — no further checks run."""
        assert run_preflight("claude", interactive=False) is False

    def test_missing_credentials_do_not_block(self) -> None:
        """Missing credentials are optional — launch still reported as ready."""
        with (
            patch(
                "terok_executor.preflight.check_podman",
                return_value=CheckResult("podman", True, "ok"),
            ),
            patch(
                "terok_executor.preflight.check_sandbox_services",
                return_value=CheckResult("sandbox services", True, "ready"),
            ),
            patch(
                "terok_executor.preflight.check_images",
                return_value=CheckResult("container images", True, "ready"),
            ),
            patch(
                "terok_executor.preflight.check_ssh_key",
                return_value=CheckResult("ssh key", True, "present"),
            ),
            patch(
                "terok_executor.preflight.check_credentials",
                return_value=CheckResult("claude credentials", False, "not found"),
            ),
            patch(
                "terok_executor.preflight.check_shield",
                return_value=CheckResult("shield", True, "ok"),
            ),
        ):
            assert run_preflight("claude", interactive=False) is True

    def test_missing_sandbox_services_blocks(self) -> None:
        """Sandbox services are mandatory — missing → False in non-interactive."""
        with (
            patch(
                "terok_executor.preflight.check_podman",
                return_value=CheckResult("podman", True, "ok"),
            ),
            patch(
                "terok_executor.preflight.check_sandbox_services",
                return_value=CheckResult("sandbox services", False, "missing"),
            ),
            patch(
                "terok_executor.preflight.check_images",
                return_value=CheckResult("container images", True, "ready"),
            ),
            patch(
                "terok_executor.preflight.check_ssh_key",
                return_value=CheckResult("ssh key", True, "ok"),
            ),
            patch(
                "terok_executor.preflight.check_credentials",
                return_value=CheckResult("claude credentials", True, "ok"),
            ),
            patch(
                "terok_executor.preflight.check_shield",
                return_value=CheckResult("shield", True, "ok"),
            ),
        ):
            assert run_preflight("claude", interactive=False) is False

    def test_assume_yes_accepts_fixes_without_input(self) -> None:
        """``--yes`` drives interactive remediation without calling input()."""
        check_results_seq = [
            CheckResult("sandbox services", False, "missing"),
            CheckResult("sandbox services", True, "ready"),
        ]
        with (
            patch(
                "terok_executor.preflight.check_podman",
                return_value=CheckResult("podman", True, "ok"),
            ),
            patch(
                "terok_executor.preflight.check_sandbox_services",
                side_effect=check_results_seq,
            ),
            patch(
                "terok_executor.preflight._fix_sandbox_services", return_value=True
            ) as fix_services,
            patch(
                "terok_executor.preflight.check_images",
                return_value=CheckResult("container images", True, "ready"),
            ),
            patch(
                "terok_executor.preflight.check_ssh_key",
                return_value=CheckResult("ssh key", True, "ok"),
            ),
            patch(
                "terok_executor.preflight.check_credentials",
                return_value=CheckResult("claude credentials", True, "ok"),
            ),
            patch(
                "terok_executor.preflight.check_shield",
                return_value=CheckResult("shield", True, "ok"),
            ),
            patch("terok_executor.preflight.input") as mock_input,
        ):
            result = run_preflight("claude", interactive=True, assume_yes=True)

        assert result is True
        fix_services.assert_called_once()
        mock_input.assert_not_called()
