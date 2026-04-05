# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent-level container health checks."""

from __future__ import annotations

from terok_sandbox.doctor import DoctorCheck

from terok_agent.doctor import (
    _PHANTOM_TOKEN_RE,
    _make_base_url_checks,
    _make_credential_file_checks,
    _make_gh_proxy_bridge_check,
    _make_phantom_token_checks,
    _make_ssh_bridge_check,
    agent_doctor_checks,
)
from terok_agent.roster import get_roster

PROXY_PORT = 18731


class TestSSHBridgeCheck:
    """SSH agent socat bridge liveness check."""

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


class TestGhProxyBridgeCheck:
    """gh credential proxy socat bridge liveness check."""

    def test_ok_when_alive(self) -> None:
        check = _make_gh_proxy_bridge_check()
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"

    def test_error_when_dead(self) -> None:
        check = _make_gh_proxy_bridge_check()
        verdict = check.evaluate(1, "", "")
        assert verdict.severity == "error"
        assert verdict.fixable is True

    def test_has_fix_cmd(self) -> None:
        check = _make_gh_proxy_bridge_check()
        assert check.fix_cmd is not None

    def test_category_is_bridge(self) -> None:
        check = _make_gh_proxy_bridge_check()
        assert check.category == "bridge"


class TestCredentialFileChecks:
    """Known credential file leak detection."""

    def test_generates_checks_for_routed_providers(self) -> None:
        roster = get_roster()
        checks = _make_credential_file_checks(roster)
        # Should have at least one check for providers with credential_file
        providers_with_cred = [
            n
            for n, r in roster.proxy_routes.items()
            if r.credential_file and n in roster.auth_providers
        ]
        assert len(checks) == len(providers_with_cred)

    def test_clean_when_file_missing(self) -> None:
        roster = get_roster()
        checks = _make_credential_file_checks(roster)
        if checks:
            # rc != 0 with "No such file" stderr means file doesn't exist
            verdict = checks[0].evaluate(1, "", "cat: /path: No such file or directory\n")
            assert verdict.severity == "ok"

    def test_warn_on_permission_denied(self) -> None:
        roster = get_roster()
        checks = _make_credential_file_checks(roster)
        if checks:
            verdict = checks[0].evaluate(1, "", "cat: /path: Permission denied\n")
            assert verdict.severity == "warn"
            assert "Permission denied" in verdict.detail

    def test_error_on_real_key(self) -> None:
        roster = get_roster()
        checks = _make_credential_file_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, '{"api_key": "sk-ant-real-key"}', "")
            assert verdict.severity == "error"
            assert verdict.fixable is True

    def test_clean_on_empty_file(self) -> None:
        roster = get_roster()
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
        roster = get_roster()
        checks = _make_phantom_token_checks(roster)
        # Should have at least some checks
        assert len(checks) > 0

    def test_ok_for_phantom_token(self) -> None:
        roster = get_roster()
        checks = _make_phantom_token_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, "terok-p-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4\n", "")
            assert verdict.severity == "ok"

    def test_warn_for_unrecognised_format(self) -> None:
        roster = get_roster()
        checks = _make_phantom_token_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4\n", "")
            assert verdict.severity == "warn"
            assert "unrecognised" in verdict.detail

    def test_error_for_real_key(self) -> None:
        roster = get_roster()
        checks = _make_phantom_token_checks(roster)
        if checks:
            verdict = checks[0].evaluate(0, "sk-ant-api03-real-key-here\n", "")
            assert verdict.severity == "error"

    def test_warn_when_unset(self) -> None:
        roster = get_roster()
        checks = _make_phantom_token_checks(roster)
        if checks:
            # rc=1 from printenv means var is unset
            verdict = checks[0].evaluate(1, "", "")
            assert verdict.severity == "warn"

    def test_no_duplicate_env_vars(self) -> None:
        roster = get_roster()
        checks = _make_phantom_token_checks(roster)
        env_vars = [" ".join(c.probe_cmd) for c in checks]
        assert len(env_vars) == len(set(env_vars)), "duplicate env var checks"


class TestBaseUrlChecks:
    """Base URL override verification."""

    def test_generates_checks(self) -> None:
        roster = get_roster()
        checks = _make_base_url_checks(roster, PROXY_PORT)
        assert len(checks) > 0

    def test_ok_when_routed(self) -> None:
        roster = get_roster()
        checks = _make_base_url_checks(roster, PROXY_PORT)
        if checks:
            verdict = checks[0].evaluate(0, f"http://host.containers.internal:{PROXY_PORT}\n", "")
            assert verdict.severity == "ok"

    def test_error_when_bypassed(self) -> None:
        roster = get_roster()
        checks = _make_base_url_checks(roster, PROXY_PORT)
        if checks:
            verdict = checks[0].evaluate(0, "https://api.anthropic.com\n", "")
            assert verdict.severity == "error"

    def test_warn_when_unset(self) -> None:
        roster = get_roster()
        checks = _make_base_url_checks(roster, PROXY_PORT)
        if checks:
            verdict = checks[0].evaluate(0, "", "")
            assert verdict.severity == "warn"

    def test_no_duplicate_vars(self) -> None:
        roster = get_roster()
        checks = _make_base_url_checks(roster, PROXY_PORT)
        vars_checked = [" ".join(c.probe_cmd) for c in checks]
        assert len(vars_checked) == len(set(vars_checked))


class TestAgentDoctorChecks:
    """Integration: agent_doctor_checks() assembly."""

    def test_includes_bridge_checks(self) -> None:
        roster = get_roster()
        checks = agent_doctor_checks(roster, proxy_port=PROXY_PORT)
        categories = {c.category for c in checks}
        assert "bridge" in categories

    def test_includes_mount_checks(self) -> None:
        roster = get_roster()
        checks = agent_doctor_checks(roster, proxy_port=PROXY_PORT)
        categories = {c.category for c in checks}
        assert "mount" in categories

    def test_includes_env_checks(self) -> None:
        roster = get_roster()
        checks = agent_doctor_checks(roster, proxy_port=PROXY_PORT)
        categories = {c.category for c in checks}
        assert "env" in categories

    def test_skips_base_url_without_port(self) -> None:
        roster = get_roster()
        checks_with = agent_doctor_checks(roster, proxy_port=PROXY_PORT)
        checks_without = agent_doctor_checks(roster, proxy_port=None)
        base_url_with = [c for c in checks_with if "Base URL" in c.label]
        base_url_without = [c for c in checks_without if "Base URL" in c.label]
        assert len(base_url_with) > 0
        assert len(base_url_without) == 0

    def test_all_are_doctor_check_instances(self) -> None:
        roster = get_roster()
        checks = agent_doctor_checks(roster, proxy_port=PROXY_PORT)
        for check in checks:
            assert isinstance(check, DoctorCheck)
