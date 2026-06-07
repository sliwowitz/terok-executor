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

from terok_executor.preflight import CheckResult, Preflight


def _pf(**overrides) -> Preflight:
    """``Preflight`` with default ``provider="claude"`` plus any test overrides."""
    return Preflight(provider="claude", **overrides)


# ── Individual checks ────────────────────────────────────────────────


@patch("terok_executor.preflight.subprocess.run")
@patch("terok_executor.preflight.shutil.which", return_value="/usr/bin/podman")
def test_podman_ok(_which: MagicMock, mock_run: MagicMock) -> None:
    """Podman found and responds → ok."""
    mock_run.return_value = MagicMock(returncode=0, stdout=b"5.0.0\n", stderr=b"")
    assert _pf().check_podman().ok is True


@patch("terok_executor.preflight.shutil.which", return_value=None)
def test_podman_missing(_which: MagicMock) -> None:
    """Podman not on PATH → fail."""
    r = _pf().check_podman()
    assert r.ok is False
    assert "not found" in r.message


@patch("terok_executor.preflight.subprocess.run")
@patch("terok_executor.preflight.shutil.which", return_value="/usr/bin/podman")
def test_podman_present_but_nonzero_exit(_which: MagicMock, mock_run: MagicMock) -> None:
    """``podman version`` exits non-zero → fail with stderr detail.

    Guards the CodeRabbit finding on PR #365: a binary that exists but
    isn't responding (broken install, missing dependencies, half-set-up
    rootless config) was previously misreported as healthy.
    """
    mock_run.return_value = MagicMock(returncode=125, stdout=b"", stderr=b"socket missing\n")
    r = _pf().check_podman()
    assert r.ok is False
    assert "socket missing" in r.message


# ── Sandbox services aggregate ───────────────────────────────────────
#
# Post-supervisor-refactor: neither the vault nor the git gate is a
# host service any more — the per-container supervisor (spawned by the
# terok-sandbox OCI hook) embeds the vault proxy and serves the gate
# in-process.  Preflight therefore only checks the shield OCI hooks.


@patch("terok_executor.integrations.sandbox.check_environment")
def test_sandbox_services_ok(mock_env: MagicMock) -> None:
    """Shield hooks installed → ok."""
    mock_env.return_value = MagicMock(health="ok")
    r = _pf().check_sandbox_services()
    assert r.ok is True
    assert "shield" in r.message


@patch("terok_executor.integrations.sandbox.check_environment")
def test_sandbox_services_lists_missing(mock_env: MagicMock) -> None:
    """Missing shield hooks → fail, named in the message."""
    mock_env.return_value = MagicMock(health="setup-needed")
    r = _pf().check_sandbox_services()
    assert r.ok is False
    assert "shield" in r.message


@patch("terok_executor.integrations.sandbox.check_environment")
def test_sandbox_services_bypass_is_ready(mock_env: MagicMock) -> None:
    """``bypass`` health is a *ready* environment — the hooks are installed,
    the operator merely opted out of egress filtering.  The bypass itself is
    surfaced elsewhere as a warning, so the readiness verdict must pass rather
    than report a missing service."""
    mock_env.return_value = MagicMock(health="bypass")
    r = _pf().check_sandbox_services()
    assert r.ok is True
    assert "shield" in r.message


# ── check_git ────────────────────────────────────────────────────────


@patch("terok_executor.preflight.shutil.which", return_value="/usr/bin/git")
def test_git_present(_which: MagicMock) -> None:
    """git on PATH → ok."""
    assert _pf().check_git().ok is True


@patch("terok_executor.preflight.shutil.which", return_value=None)
def test_git_missing_returns_consequence(_which: MagicMock) -> None:
    """git missing → fail, message names the consequence."""
    r = _pf().check_git()
    assert r.ok is False
    assert "gate disabled" in r.message


# ── check_credentials ────────────────────────────────────────────────


@patch("terok_executor.integrations.sandbox.SandboxConfig.open_credential_db")
def test_credentials_found(mock_db_cls: MagicMock) -> None:
    """Credentials stored → ok."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = {"type": "api_key", "value": "sk-test"}
    assert _pf().check_credentials().ok is True
    db.close.assert_called_once()


@patch("terok_executor.integrations.sandbox.SandboxConfig.open_credential_db")
def test_credentials_missing(mock_db_cls: MagicMock) -> None:
    """No credentials → fail."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = None
    r = _pf().check_credentials()
    assert r.ok is False
    assert "not found" in r.message
    db.close.assert_called_once()


@patch(
    "terok_executor.integrations.sandbox.SandboxConfig.open_credential_db",
    side_effect=Exception("db error"),
)
def test_credentials_db_unavailable(_cls: MagicMock) -> None:
    """DB open fails → fail with message."""
    r = _pf().check_credentials()
    assert r.ok is False
    assert "unavailable" in r.message


@patch("terok_executor.integrations.sandbox.SandboxConfig.open_credential_db")
def test_credentials_default_set(mock_db_cls: MagicMock) -> None:
    """No ``credential_set`` override → DB is queried with ``"default"``."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = {"type": "api_key"}
    _pf().check_credentials()
    # claude resolves to its default provider (anthropic) for the DB lookup.
    db.load_credential.assert_called_once_with("default", "anthropic")


@patch("terok_executor.integrations.sandbox.SandboxConfig.open_credential_db")
def test_credentials_custom_set(mock_db_cls: MagicMock) -> None:
    """``credential_set=<id>`` → DB is queried with that set, not ``"default"``."""
    db = mock_db_cls.return_value
    db.load_credential.return_value = {"type": "api_key"}
    _pf(credential_set="my-proj").check_credentials()
    db.load_credential.assert_called_once_with("my-proj", "anthropic")


def test_fix_credentials_threads_mounts_dir_to_authenticator() -> None:
    """When ``mounts_dir`` is supplied, ``_fix_credentials`` routes it to ``Authenticator``.

    Pairs with the new ``credential_set`` parameter — both must be honoured
    so the captured token's vault row and the post-capture phantom marker
    land in the same per-project tree.
    """
    from pathlib import Path

    from terok_executor.preflight import _fix_credentials

    with (
        patch("terok_executor.credentials.auth.Authenticator") as auth_cls,
        patch("terok_executor.credentials.vault_config.write_vault_config"),
    ):
        ok = _fix_credentials(
            "claude",
            base_image="ubuntu:24.04",
            credential_set="my-proj",
            mounts_dir=Path("/proj/root/mounts"),
        )
    assert ok is True
    auth_cls.return_value.run.assert_called_once()
    kwargs = auth_cls.return_value.run.call_args.kwargs
    assert kwargs["mounts_dir"] == Path("/proj/root/mounts")
    assert kwargs["credential_set"] == "my-proj"


def test_fix_credentials_default_mounts_dir_falls_back_to_global() -> None:
    """No ``mounts_dir`` override → ``Authenticator`` sees the host-wide default."""
    from pathlib import Path

    from terok_executor.preflight import _fix_credentials

    with (
        patch("terok_executor.credentials.auth.Authenticator") as auth_cls,
        patch("terok_executor.credentials.vault_config.write_vault_config"),
        patch("terok_executor.paths.mounts_dir", return_value=Path("/global/mounts")),
    ):
        _fix_credentials("claude", base_image="ubuntu:24.04")
    kwargs = auth_cls.return_value.run.call_args.kwargs
    assert kwargs["mounts_dir"] == Path("/global/mounts")
    assert kwargs["credential_set"] == "default"


# ── check_ssh_key ────────────────────────────────────────────────────


@patch("terok_executor.integrations.sandbox.SandboxConfig.open_credential_db")
def test_ssh_key_present(mock_db_cls: MagicMock) -> None:
    """Existing key in scope → ok."""
    db = mock_db_cls.return_value
    db.list_ssh_keys_for_scope.return_value = [MagicMock()]
    r = _pf().check_ssh_key()
    assert r.ok is True
    db.close.assert_called_once()


@patch("terok_executor.integrations.sandbox.SandboxConfig.open_credential_db")
def test_ssh_key_absent(mock_db_cls: MagicMock) -> None:
    """Empty scope → fail."""
    db = mock_db_cls.return_value
    db.list_ssh_keys_for_scope.return_value = []
    r = _pf().check_ssh_key()
    assert r.ok is False


# ── check_images ─────────────────────────────────────────────────────


@patch("terok_executor.preflight.subprocess.run")
def test_images_exist(mock_run: MagicMock) -> None:
    """Image exists → ok."""
    mock_run.return_value = MagicMock(returncode=0)
    assert _pf(base_image="ubuntu:24.04").check_images().ok is True


@patch("terok_executor.preflight.subprocess.run")
def test_images_missing(mock_run: MagicMock) -> None:
    """Image doesn't exist → fail."""
    mock_run.return_value = MagicMock(returncode=1)
    assert _pf(base_image="ubuntu:24.04").check_images().ok is False


# ── check_shield ─────────────────────────────────────────────────────


@patch("terok_executor.integrations.sandbox.check_environment")
def test_shield_ok(mock_env: MagicMock) -> None:
    """Shield active → ok."""
    mock_env.return_value = MagicMock(health="ok")
    assert _pf().check_shield().ok is True


@patch("terok_executor.integrations.sandbox.check_environment")
def test_shield_missing(mock_env: MagicMock) -> None:
    """Shield not installed → fail (informational)."""
    mock_env.return_value = MagicMock(health="setup-needed")
    r = _pf().check_shield()
    assert r.ok is False
    assert "unrestricted" in r.message


# ── Preflight.run orchestration ──────────────────────────────────────


def _patch_all_ok():
    """Patch every individual check to report ok."""
    return [
        patch.object(Preflight, "check_podman", return_value=CheckResult("podman", True, "ok")),
        patch.object(
            Preflight,
            "check_sandbox_services",
            return_value=CheckResult("sandbox services", True, "ready"),
        ),
        patch.object(
            Preflight, "check_images", return_value=CheckResult("container images", True, "ready")
        ),
        patch.object(
            Preflight, "check_ssh_key", return_value=CheckResult("ssh key", True, "present")
        ),
        patch.object(
            Preflight,
            "check_credentials",
            return_value=CheckResult("claude credentials", True, "stored"),
        ),
        patch.object(Preflight, "check_shield", return_value=CheckResult("shield", True, "ok")),
    ]


class TestPreflightRun:
    """Orchestration contract — mandatory vs optional, TTY vs --yes."""

    def test_all_ok(self) -> None:
        """Every check passes → returns True."""
        patches = _patch_all_ok()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            assert _pf(interactive=False).run() is True

    @patch.object(Preflight, "check_podman", return_value=CheckResult("podman", False, "nope"))
    def test_podman_missing_bails_immediately(self, _pod: MagicMock) -> None:
        """Podman is a hard stop — no further checks run."""
        assert _pf(interactive=False).run() is False

    def test_missing_credentials_do_not_block(self) -> None:
        """Missing credentials are optional — launch still reported as ready."""
        with (
            patch.object(Preflight, "check_podman", return_value=CheckResult("podman", True, "ok")),
            patch.object(
                Preflight,
                "check_sandbox_services",
                return_value=CheckResult("sandbox services", True, "ready"),
            ),
            patch.object(
                Preflight,
                "check_images",
                return_value=CheckResult("container images", True, "ready"),
            ),
            patch.object(
                Preflight, "check_ssh_key", return_value=CheckResult("ssh key", True, "present")
            ),
            patch.object(
                Preflight,
                "check_credentials",
                return_value=CheckResult("claude credentials", False, "not found"),
            ),
            patch.object(Preflight, "check_shield", return_value=CheckResult("shield", True, "ok")),
        ):
            assert _pf(interactive=False).run() is True

    def test_missing_sandbox_services_blocks(self) -> None:
        """Sandbox services are mandatory — missing → False in non-interactive."""
        with (
            patch.object(Preflight, "check_podman", return_value=CheckResult("podman", True, "ok")),
            patch.object(
                Preflight,
                "check_sandbox_services",
                return_value=CheckResult("sandbox services", False, "missing"),
            ),
            patch.object(
                Preflight,
                "check_images",
                return_value=CheckResult("container images", True, "ready"),
            ),
            patch.object(
                Preflight, "check_ssh_key", return_value=CheckResult("ssh key", True, "ok")
            ),
            patch.object(
                Preflight,
                "check_credentials",
                return_value=CheckResult("claude credentials", True, "ok"),
            ),
            patch.object(Preflight, "check_shield", return_value=CheckResult("shield", True, "ok")),
        ):
            assert _pf(interactive=False).run() is False

    def test_assume_yes_accepts_fixes_without_input(self) -> None:
        """``--yes`` drives interactive remediation without calling input()."""
        check_results_seq = [
            CheckResult("sandbox services", False, "missing"),
            CheckResult("sandbox services", True, "ready"),
        ]
        with (
            patch.object(Preflight, "check_podman", return_value=CheckResult("podman", True, "ok")),
            patch.object(Preflight, "check_sandbox_services", side_effect=check_results_seq),
            patch(
                "terok_executor.preflight._fix_sandbox_services", return_value=True
            ) as fix_services,
            patch.object(
                Preflight,
                "check_images",
                return_value=CheckResult("container images", True, "ready"),
            ),
            patch.object(
                Preflight, "check_ssh_key", return_value=CheckResult("ssh key", True, "ok")
            ),
            patch.object(
                Preflight,
                "check_credentials",
                return_value=CheckResult("claude credentials", True, "ok"),
            ),
            patch.object(Preflight, "check_shield", return_value=CheckResult("shield", True, "ok")),
            patch("terok_executor.preflight.input") as mock_input,
        ):
            result = _pf(interactive=True, assume_yes=True).run()

        assert result is True
        fix_services.assert_called_once()
        mock_input.assert_not_called()


class TestFixSshKey:
    """``_fix_ssh_key`` provisions the gate-signing key via the new cfg seam."""

    def test_success_returns_true_and_reports(self, capsys) -> None:
        """Happy path: SSHManager opens, init succeeds, we print key + return True."""
        from terok_executor.preflight import _fix_ssh_key

        fake_mgr = MagicMock()
        fake_mgr.init.return_value = {
            "key_type": "ed25519",
            "fingerprint": "abcdef0123456789" * 4,
            "public_line": "ssh-ed25519 AAAA… proj-key",
        }
        # The context-manager protocol surface ``open_for_config`` uses.
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_mgr
        fake_ctx.__exit__.return_value = False
        with patch(
            "terok_executor.integrations.sandbox.SSHManager.open_for_config", return_value=fake_ctx
        ) as m_open:
            assert _fix_ssh_key("proj") is True
        # Sandbox seam called via the new ``open_for_config(cfg=)`` shape — not the
        # removed ``open(db_path=…)``, which was the leaky tier-knob variant.
        m_open.assert_called_once()
        kwargs = m_open.call_args.kwargs
        assert kwargs["scope"] == "proj"
        # ``cfg`` is a SandboxConfig instance (the seam takes a config, not knobs).
        from terok_sandbox import SandboxConfig

        assert isinstance(kwargs["cfg"], SandboxConfig)
        out = capsys.readouterr().out
        assert "ed25519" in out
        assert "ssh-ed25519" in out

    def test_failure_prints_error_and_returns_false(self, capsys) -> None:
        """If ``mgr.init`` raises, swallow + report on stderr + return False."""
        from terok_executor.preflight import _fix_ssh_key

        fake_mgr = MagicMock()
        fake_mgr.init.side_effect = RuntimeError("keygen failed")
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_mgr
        fake_ctx.__exit__.return_value = False
        with patch(
            "terok_executor.integrations.sandbox.SSHManager.open_for_config", return_value=fake_ctx
        ):
            assert _fix_ssh_key("proj") is False
        # Operator-actionable diagnostic lands on stderr, not stdout.
        captured = capsys.readouterr()
        assert "keygen failed" in captured.err
        assert "ed25519" not in captured.out
