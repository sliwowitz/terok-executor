# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the preflight prerequisite checker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from terok_executor.preflight import (
    CheckResult,
    check_credentials,
    check_images,
    check_podman,
    check_proxy,
    check_shield,
    run_preflight,
)

# ── check_podman ─────────────────────────────────────────────────────


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


# ── check_proxy ──────────────────────────────────────────────────────


@patch("terok_sandbox.is_proxy_running", return_value=True)
@patch("terok_sandbox.is_proxy_socket_active", return_value=False)
def test_proxy_running(_sock: MagicMock, _run: MagicMock) -> None:
    """Proxy daemon running → ok."""
    assert check_proxy().ok is True


@patch("terok_sandbox.is_proxy_running", return_value=False)
@patch("terok_sandbox.is_proxy_socket_active", return_value=True)
def test_proxy_socket_active(_sock: MagicMock, _run: MagicMock) -> None:
    """Proxy systemd socket active → ok."""
    assert check_proxy().ok is True


@patch("terok_sandbox.is_proxy_running", return_value=False)
@patch("terok_sandbox.is_proxy_socket_active", return_value=False)
def test_proxy_not_running(_sock: MagicMock, _run: MagicMock) -> None:
    """Neither running nor socket active → fail."""
    assert check_proxy().ok is False


# ── check_credentials ────────────────────────────────────────────────


@patch("terok_sandbox.CredentialDB")
def test_credentials_found(mock_db_cls: MagicMock) -> None:
    """Credentials stored → ok."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = {"type": "api_key", "value": "sk-test"}
    assert check_credentials("claude").ok is True
    db.close.assert_called_once()


@patch("terok_sandbox.CredentialDB")
def test_credentials_missing(mock_db_cls: MagicMock) -> None:
    """No credentials → fail."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = None
    r = check_credentials("claude")
    assert r.ok is False
    assert "not found" in r.message
    db.close.assert_called_once()


@patch("terok_sandbox.CredentialDB", side_effect=Exception("db error"))
def test_credentials_db_unavailable(_cls: MagicMock) -> None:
    """DB open fails → fail with message."""
    r = check_credentials("claude")
    assert r.ok is False
    assert "unavailable" in r.message


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


# ── run_preflight ────────────────────────────────────────────────────


@patch("terok_executor.preflight.check_shield", return_value=CheckResult("shield", True, "ok"))
@patch("terok_executor.preflight.check_images", return_value=CheckResult("images", True, "ready"))
@patch(
    "terok_executor.preflight.check_credentials",
    return_value=CheckResult("claude creds", True, "stored"),
)
@patch("terok_executor.preflight.check_proxy", return_value=CheckResult("proxy", True, "running"))
@patch("terok_executor.preflight.check_podman", return_value=CheckResult("podman", True, "ok"))
def test_preflight_all_ok(
    _pod: MagicMock,
    _proxy: MagicMock,
    _creds: MagicMock,
    _imgs: MagicMock,
    _shield: MagicMock,
) -> None:
    """All checks pass → returns True."""
    assert run_preflight("claude", interactive=False) is True


@patch("terok_executor.preflight.check_podman", return_value=CheckResult("podman", False, "nope"))
def test_preflight_no_podman(_pod: MagicMock) -> None:
    """Podman missing → returns False immediately."""
    assert run_preflight("claude", interactive=False) is False


@patch("terok_executor.preflight.check_shield", return_value=CheckResult("shield", False, "nope"))
@patch("terok_executor.preflight.check_images", return_value=CheckResult("images", True, "ready"))
@patch(
    "terok_executor.preflight.check_credentials",
    return_value=CheckResult("claude creds", True, "stored"),
)
@patch("terok_executor.preflight.check_proxy", return_value=CheckResult("proxy", True, "running"))
@patch("terok_executor.preflight.check_podman", return_value=CheckResult("podman", True, "ok"))
def test_preflight_shield_missing_still_ok(
    _pod: MagicMock,
    _proxy: MagicMock,
    _creds: MagicMock,
    _imgs: MagicMock,
    _shield: MagicMock,
) -> None:
    """Shield missing is informational — doesn't block the run."""
    assert run_preflight("claude", interactive=False) is True


@patch("terok_executor.preflight.check_shield", return_value=CheckResult("shield", True, "ok"))
@patch("terok_executor.preflight.check_images", return_value=CheckResult("images", True, "ready"))
@patch(
    "terok_executor.preflight.check_credentials",
    return_value=CheckResult("claude creds", False, "not found"),
)
@patch("terok_executor.preflight.check_proxy", return_value=CheckResult("proxy", True, "running"))
@patch("terok_executor.preflight.check_podman", return_value=CheckResult("podman", True, "ok"))
def test_preflight_creds_missing_non_interactive(
    _pod: MagicMock,
    _proxy: MagicMock,
    _creds: MagicMock,
    _imgs: MagicMock,
    _shield: MagicMock,
) -> None:
    """Missing credentials in non-interactive mode → returns False."""
    assert run_preflight("claude", interactive=False) is False


@patch("terok_executor.preflight._provider_hints")
@patch("terok_executor.preflight._fix_credentials", return_value=True)
@patch("terok_executor.preflight._confirm", return_value=True)
@patch("terok_executor.preflight.check_shield", return_value=CheckResult("shield", True, "ok"))
@patch("terok_executor.preflight.check_images", return_value=CheckResult("images", True, "ready"))
@patch(
    "terok_executor.preflight.check_credentials",
    return_value=CheckResult("claude creds", False, "not found"),
)
@patch("terok_executor.preflight.check_proxy", return_value=CheckResult("proxy", True, "running"))
@patch("terok_executor.preflight.check_podman", return_value=CheckResult("podman", True, "ok"))
def test_preflight_creds_fixed_interactively(
    _pod: MagicMock,
    _proxy: MagicMock,
    _creds: MagicMock,
    _imgs: MagicMock,
    _shield: MagicMock,
    _confirm: MagicMock,
    _fix: MagicMock,
    _hints: MagicMock,
) -> None:
    """Missing credentials + interactive fix → returns True."""
    assert run_preflight("claude", interactive=True) is True
    _fix.assert_called_once_with("claude")
