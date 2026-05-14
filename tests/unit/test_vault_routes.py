# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for vault route parsing, routes.json, and CLI handlers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from terok_executor.roster import VaultRoute, get_roster
from terok_executor.roster.schema import RawAgentYaml
from tests.unit.conftest import TEST_VAULT_PASSPHRASE


def _vault_route(name: str, data: dict) -> VaultRoute | None:
    """Validate *data* and project the ``vault:`` section to a [`VaultRoute`][]."""
    spec = RawAgentYaml.model_validate(data)
    if spec.vault is None:
        return None
    return spec.vault.to_dataclass(provider=name)


class TestVaultRoutesParsed:
    """Verify vault YAML sections are parsed into the roster."""

    def test_claude_route_exists(self) -> None:
        """Claude has a vault route with Anthropic upstream and OAuth support."""
        reg = get_roster()
        route = reg.vault_routes.get("claude")
        assert route is not None
        assert route.route_prefix == "claude"
        assert route.upstream == "https://api.anthropic.com"
        assert route.auth_header == "dynamic"
        assert route.oauth_extra_headers == {"anthropic-beta": "oauth-2025-04-20"}
        assert "ANTHROPIC_API_KEY" in route.phantom_env
        assert "CLAUDE_CODE_OAUTH_TOKEN" in route.oauth_phantom_env
        assert route.base_url_env == "ANTHROPIC_BASE_URL"
        assert route.socket_env == "ANTHROPIC_UNIX_SOCKET"

    def test_codex_route_exists(self) -> None:
        """Codex has a vault route with OpenAI + ChatGPT upstreams."""
        route = get_roster().vault_routes.get("codex")
        assert route is not None
        assert route.upstream == "https://api.openai.com"
        assert route.path_upstreams == {"/backend-api/": "https://chatgpt.com"}
        assert route.oauth_extra_headers == {}
        assert route.shared_config_patch is not None
        assert route.shared_config_patch["file"] == "config.toml"

    def test_gh_route_exists(self) -> None:
        """GitHub CLI has a vault route with token-style auth."""
        route = get_roster().vault_routes.get("gh")
        assert route is not None
        assert route.auth_prefix == "token "
        assert route.upstream == "https://api.github.com"

    def test_glab_route_exists(self) -> None:
        """GitLab CLI has a vault route with PRIVATE-TOKEN header."""
        route = get_roster().vault_routes.get("glab")
        assert route is not None
        assert route.auth_header == "PRIVATE-TOKEN"
        assert route.auth_prefix == ""
        assert route.route_prefix == "gl"

    def test_api_key_only_providers_have_no_oauth_phantom_env(self) -> None:
        """Providers without OAuth support have empty oauth_phantom_env."""
        for name in ("vibe", "blablador", "kisski"):
            route = get_roster().vault_routes.get(name)
            assert route is not None, f"{name} missing vault route"
            assert route.oauth_phantom_env == {}, f"{name} should have no oauth_phantom_env"
            assert route.socket_env == "", f"{name} should have no socket_env"

    def test_rejects_legacy_socket_path_field(self) -> None:
        """socket_path was removed — declaring it must fail loudly so stale
        agent manifests don't silently drift out of sync."""
        base = {"route_prefix": "test", "upstream": "https://example.com"}
        with pytest.raises(ValidationError, match="socket_path.*no longer"):
            _vault_route("test", {"vault": {**base, "socket_path": "/tmp/s.sock"}})

    def test_opencode_agents_have_routes(self) -> None:
        """Blablador and KISSKI have vault routes."""
        reg = get_roster()
        for name in ("blablador", "kisski"):
            route = reg.vault_routes.get(name)
            assert route is not None, f"{name} missing vault route"
            assert route.credential_type == "api_key"

    def test_copilot_has_no_route(self) -> None:
        """Copilot has no vault section (tier-3, no base URL support)."""
        assert get_roster().vault_routes.get("copilot") is None

    def test_claude_has_oauth_refresh(self) -> None:
        """Claude has oauth_refresh config for proactive token refresh."""
        route = get_roster().vault_routes.get("claude")
        assert route is not None
        assert route.oauth_refresh is not None
        assert "token_url" in route.oauth_refresh
        assert "client_id" in route.oauth_refresh

    def test_codex_has_oauth_refresh(self) -> None:
        """Codex has an oauth_refresh block so vault can rotate tokens in the background."""
        route = get_roster().vault_routes.get("codex")
        assert route is not None
        assert route.oauth_refresh is not None
        assert route.oauth_refresh["token_url"] == "https://auth.openai.com/oauth/token"
        assert route.oauth_refresh["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"


class TestSharedDomain:
    """Verify the ``vault.shared_domain`` flag is parsed and surfaced."""

    def test_default_is_false(self) -> None:
        """API-only upstreams (claude, codex, gh, …) leave the flag unset."""
        roster = get_roster()
        for name in ("claude", "codex", "gh", "vibe", "blablador", "kisski", "openrouter"):
            route = roster.vault_routes[name]
            assert route.shared_domain is False, f"{name} should not be shared_domain"

    def test_glab_is_shared_domain(self) -> None:
        """gitlab.com hosts both API and ``git push`` traffic."""
        assert get_roster().vault_routes["glab"].shared_domain is True

    def test_sonar_is_shared_domain(self) -> None:
        """sonarcloud.io hosts API + project pages + docs + badges."""
        assert get_roster().vault_routes["sonar"].shared_domain is True

    def test_unknown_provider_defaults_to_false(self) -> None:
        """Hand-rolled vault sections without the field default to False."""
        route = _vault_route(
            "test",
            {"vault": {"route_prefix": "t", "upstream": "https://api.example.com"}},
        )
        assert route is not None
        assert route.shared_domain is False

    def test_explicit_true_round_trips(self) -> None:
        """``shared_domain: true`` is preserved through schema → dataclass."""
        route = _vault_route(
            "test",
            {
                "vault": {
                    "route_prefix": "t",
                    "upstream": "https://example.com",
                    "shared_domain": True,
                }
            },
        )
        assert route is not None
        assert route.shared_domain is True


class TestGenerateRoutesJson:
    """Verify routes.json generation."""

    def test_generates_valid_json(self) -> None:
        """generate_routes_json() produces parseable JSON with expected keys."""
        routes_json = get_roster().generate_routes_json()
        routes = json.loads(routes_json)
        assert "claude" in routes
        assert routes["claude"]["upstream"] == "https://api.anthropic.com"
        assert routes["claude"]["auth_header"] == "dynamic"
        assert routes["claude"]["oauth_extra_headers"] == {"anthropic-beta": "oauth-2025-04-20"}
        assert routes["codex"]["path_upstreams"] == {"/backend-api/": "https://chatgpt.com"}
        assert "oauth_extra_headers" not in routes["codex"]

    def test_all_routes_have_upstream(self) -> None:
        """Every route in the JSON has an upstream field."""
        routes = json.loads(get_roster().generate_routes_json())
        for prefix, cfg in routes.items():
            assert "upstream" in cfg, f"Route '{prefix}' missing upstream"

    def test_glab_keyed_by_provider_name(self) -> None:
        """GitLab route is keyed by provider name 'glab'."""
        routes = json.loads(get_roster().generate_routes_json())
        assert "glab" in routes

    def test_claude_routes_json_includes_oauth_refresh(self) -> None:
        """Claude's routes.json entry includes oauth_refresh config."""
        routes = json.loads(get_roster().generate_routes_json())
        assert "oauth_refresh" in routes["claude"]
        assert routes["claude"]["oauth_refresh"]["client_id"]

    def test_gh_routes_json_omits_oauth_refresh(self) -> None:
        """Providers without oauth_refresh omit it from routes.json."""
        routes = json.loads(get_roster().generate_routes_json())
        assert "oauth_refresh" not in routes["gh"]


class TestScanLeakedCredentials:
    """Verify scan_leaked_credentials detects real secrets in shared mounts."""

    def test_empty_when_no_files(self, tmp_path) -> None:
        """Returns empty list when no credential files exist."""
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        assert scan_leaked_credentials(tmp_path) == []

    def test_detects_nonempty_credential_file(self, tmp_path) -> None:
        """Returns (provider, path) when a credential file is present and non-empty."""
        from terok_executor import get_roster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers.get("claude")
        route = roster.vault_routes.get("claude")
        assert auth is not None and route is not None

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred_file = cred_dir / route.credential_file
        cred_file.write_text('{"claudeAiOauth": {"accessToken": "sk-leaked"}}')

        leaked = scan_leaked_credentials(tmp_path)
        providers = [p for p, _ in leaked]
        assert "claude" in providers

    def test_skips_empty_files(self, tmp_path) -> None:
        """Empty credential files are not flagged."""
        from terok_executor import get_roster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        (cred_dir / route.credential_file).write_text("")

        assert scan_leaked_credentials(tmp_path) == []

    def test_skips_providers_without_credential_file(self, tmp_path, monkeypatch) -> None:
        """Providers whose vault route has no credential_file are skipped."""
        from unittest.mock import MagicMock

        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        # Mock a roster with a provider that has a vault route but no credential_file
        mock_roster = MagicMock()
        mock_route = MagicMock()
        mock_route.credential_file = ""
        mock_roster.vault_routes = {"fake-provider": mock_route}
        mock_roster.auth_providers = {"fake-provider": MagicMock(host_dir_name="_fake")}
        monkeypatch.setattr("terok_executor.roster.loader.get_roster", lambda: mock_roster)

        assert scan_leaked_credentials(tmp_path) == []

    def test_clean_removes_leaked_files(self, tmp_path) -> None:
        """The clean handler removes detected credential files."""
        from unittest.mock import patch

        from terok_executor import get_roster
        from terok_executor.credentials.vault_commands import _handle_clean

        roster = get_roster()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred_file = cred_dir / route.credential_file
        cred_file.write_text('{"secret": true}')

        with patch("terok_executor.paths.mounts_dir", return_value=tmp_path):
            _handle_clean()

        assert not cred_file.exists()

    def test_clean_no_files(self, capsys) -> None:
        """Clean reports nothing when no leaked files found."""
        from pathlib import Path
        from unittest.mock import patch

        from terok_executor.credentials.vault_commands import _handle_clean

        with patch("terok_executor.paths.mounts_dir", return_value=Path("/nonexistent")):
            _handle_clean()

        assert "No leaked" in capsys.readouterr().out


class TestVaultCommandHandlers:
    """Verify vault CLI command handlers."""

    @patch("terok_sandbox.start_vault")
    @patch("terok_executor.credentials.vault_commands._ensure_routes")
    @patch("terok_sandbox.is_vault_running", return_value=False)
    def test_start_generates_routes_and_starts(self, _running, _routes, _start, capsys) -> None:
        """start generates routes then starts the daemon."""
        from terok_executor.credentials.vault_commands import _handle_start

        _handle_start()
        _routes.assert_called_once()
        _start.assert_called_once()
        assert "started" in capsys.readouterr().out

    @patch("terok_sandbox.is_vault_running", return_value=True)
    def test_start_already_running_exits(self, _running) -> None:
        """start exits if vault is already running."""
        from terok_executor.credentials.vault_commands import _handle_start

        with pytest.raises(SystemExit):
            _handle_start()

    @patch("terok_sandbox.stop_vault")
    @patch("terok_sandbox.is_vault_running", return_value=True)
    def test_stop_stops_daemon(self, _running, _stop, capsys) -> None:
        """stop calls stop_vault when running."""
        from terok_executor.credentials.vault_commands import _handle_stop

        _handle_stop()
        _stop.assert_called_once()
        assert "stopped" in capsys.readouterr().out

    @patch("terok_sandbox.is_vault_running", return_value=False)
    def test_stop_not_running(self, _running, capsys) -> None:
        """stop prints info when not running."""
        from terok_executor.credentials.vault_commands import _handle_stop

        _handle_stop()
        assert "not running" in capsys.readouterr().out

    @patch("terok_executor.credentials.vault_commands.scan_leaked_credentials", return_value=[])
    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    @patch("terok_sandbox.get_vault_status")
    def test_status_prints_info(self, mock_status, _sd, _scan, capsys) -> None:
        """status prints formatted vault info."""
        mock_status.return_value = MagicMock(
            mode="daemon",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=("claude", "gh"),
            ssh_keys_stored=0,
            passphrase_source="keyring",
            locked=False,
        )
        from terok_executor.credentials.vault_commands import _handle_status

        _handle_status()
        out = capsys.readouterr().out
        assert "running" in out
        assert "claude" in out

    @patch("terok_executor.credentials.vault_commands.scan_leaked_credentials", return_value=[])
    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    @patch("terok_sandbox.get_vault_status")
    def test_status_surfaces_passphrase_source_and_ssh_key_count(
        self, mock_status, _sd, _scan, capsys
    ) -> None:
        """The fields enriched by sandbox#276 round-trip into ``vault status`` output."""
        mock_status.return_value = MagicMock(
            mode="daemon",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=(),
            ssh_keys_stored=7,
            passphrase_source="systemd-creds",
            locked=False,
        )
        from terok_executor.credentials.vault_commands import _handle_status

        _handle_status()
        out = capsys.readouterr().out
        assert "SSH keys:    7" in out
        assert "resolved via systemd-creds" in out

    @patch("terok_executor.credentials.vault_commands.scan_leaked_credentials", return_value=[])
    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    @patch("terok_sandbox.get_vault_status")
    def test_status_announces_locked_vault(self, mock_status, _sd, _scan, capsys) -> None:
        """A locked vault prints the explicit ``Locked: yes`` line and the unlock hint."""
        mock_status.return_value = MagicMock(
            mode="daemon",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=(),
            ssh_keys_stored=0,
            passphrase_source=None,
            locked=True,
        )
        from terok_executor.credentials.vault_commands import _handle_status

        _handle_status()
        out = capsys.readouterr().out
        assert "Locked:      yes" in out
        assert "vault unlock" in out

    @patch("terok_executor.credentials.vault_commands.scan_leaked_credentials", return_value=[])
    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    @patch("terok_sandbox.get_vault_status")
    def test_status_marks_unlocked_explicitly(self, mock_status, _sd, _scan, capsys) -> None:
        """A resolved vault prints ``Locked: no`` alongside the chain-tier source."""
        mock_status.return_value = MagicMock(
            mode="systemd",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=(),
            ssh_keys_stored=0,
            passphrase_source="systemd-creds",
            locked=False,
        )
        from terok_executor.credentials.vault_commands import _handle_status

        _handle_status()
        out = capsys.readouterr().out
        assert "Locked:      no" in out
        assert "resolved via systemd-creds" in out

    @patch("terok_executor.credentials.vault_commands.scan_leaked_credentials", return_value=[])
    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    @patch("terok_sandbox.get_vault_status")
    def test_status_surfaces_plaintext_passphrase_warning(
        self, mock_status, _sd, _scan, capsys
    ) -> None:
        """``plaintext_passphrase_path`` lights up a stderr WARNING (sandbox#282)."""
        from pathlib import Path

        plaintext_path = Path("/etc/terok/config.yml")
        mock_status.return_value = MagicMock(
            mode="systemd",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=(),
            ssh_keys_stored=0,
            passphrase_source="systemd-creds",
            locked=False,
            plaintext_passphrase_path=plaintext_path,
        )
        from terok_executor.credentials.vault_commands import _handle_status

        _handle_status()
        captured = capsys.readouterr()
        # Warning lives on stderr so structured stdout stays greppable.
        assert "WARNING" in captured.err
        assert "plaintext" in captured.err
        assert str(plaintext_path) in captured.err
        assert "WARNING" not in captured.out

    @patch("terok_executor.credentials.vault_commands.scan_leaked_credentials", return_value=[])
    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    @patch("terok_sandbox.get_vault_status")
    def test_status_silent_when_no_plaintext_warning(self, mock_status, _sd, _scan, capsys) -> None:
        """Default-None case is silent — no plaintext line at all."""
        mock_status.return_value = MagicMock(
            mode="systemd",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=(),
            ssh_keys_stored=0,
            passphrase_source="systemd-creds",
            locked=False,
            plaintext_passphrase_path=None,
        )
        from terok_executor.credentials.vault_commands import _handle_status

        _handle_status()
        captured = capsys.readouterr()
        assert "plaintext" not in captured.err
        assert "plaintext" not in captured.out

    @patch("terok_sandbox.install_vault_systemd")
    @patch("terok_executor.credentials.vault_commands._ensure_routes")
    @patch("terok_sandbox.is_vault_systemd_available", return_value=True)
    def test_install_generates_routes_and_installs(self, _sd, _routes, _install, capsys) -> None:
        """install generates routes then installs systemd units."""
        from terok_executor.credentials.vault_commands import _handle_install

        _handle_install()
        _routes.assert_called_once()
        _install.assert_called_once()
        assert "installed" in capsys.readouterr().out

    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    def test_install_no_systemd_exits(self, _sd) -> None:
        """install exits when systemd unavailable."""
        from terok_executor.credentials.vault_commands import _handle_install

        with pytest.raises(SystemExit):
            _handle_install()

    @patch("terok_sandbox.uninstall_vault_systemd")
    @patch("terok_sandbox.is_vault_systemd_available", return_value=True)
    def test_uninstall_removes_units(self, _sd, _uninstall, capsys) -> None:
        """uninstall removes systemd units."""
        from terok_executor.credentials.vault_commands import _handle_uninstall

        _handle_uninstall()
        _uninstall.assert_called_once()
        assert "removed" in capsys.readouterr().out

    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    def test_uninstall_no_systemd_exits(self, _sd) -> None:
        """uninstall exits when systemd unavailable."""
        from terok_executor.credentials.vault_commands import _handle_uninstall

        with pytest.raises(SystemExit):
            _handle_uninstall()

    @patch(
        "terok_executor.credentials.vault_commands._ensure_routes",
        return_value=Path("/tmp/routes.json"),
    )
    def test_routes_prints_path(self, _routes, capsys) -> None:
        """routes prints the written path."""
        from terok_executor.credentials.vault_commands import _handle_routes

        _handle_routes()
        assert "routes.json" in capsys.readouterr().out

    @patch(
        "terok_executor.credentials.vault_commands.scan_leaked_credentials",
        return_value=[("claude", Path("/envs/_claude-config/.credentials.json"))],
    )
    @patch("terok_sandbox.is_vault_systemd_available", return_value=False)
    @patch("terok_sandbox.get_vault_status")
    def test_status_shows_leak_warning(self, mock_status, _sd, _scan, capsys) -> None:
        """status shows WARNING when leaked credentials detected."""
        mock_status.return_value = MagicMock(
            mode="daemon",
            running=True,
            socket_path="/run/proxy.sock",
            db_path="/data/creds.db",
            routes_path="/data/routes.json",
            routes_configured=3,
            credentials_stored=("claude",),
            ssh_keys_stored=0,
            passphrase_source="keyring",
            locked=False,
        )
        from terok_executor.credentials.vault_commands import _handle_status

        _handle_status()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "claude" in out
        assert "clean" in out


class TestInjectedCredentialsFile:
    """Verify _is_injected_credentials_file detects phantom vs real credentials."""

    def test_recognises_injected_file(self, tmp_path: Path) -> None:
        """Correctly identifies a terok-injected .credentials.json."""
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "",
                "scopes": "user:inference user:profile",
                "subscriptionType": "max",
            }
        }
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps(cred))
        assert _is_injected_credentials_file(cred_file) is True

    def test_rejects_real_credentials(self, tmp_path: Path) -> None:
        """Real OAuth tokens are NOT identified as injected."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred = {"claudeAiOauth": {"accessToken": "sk-ant-real-token", "refreshToken": "rt-real"}}
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps(cred))
        assert _is_injected_credentials_file(cred_file) is False

    def test_rejects_phantom_token_with_refresh(self, tmp_path: Path) -> None:
        """Phantom accessToken with a non-empty refreshToken is suspicious -- flag it."""
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "rt-leaked-somehow",
            }
        }
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps(cred))
        assert _is_injected_credentials_file(cred_file) is False

    def test_handles_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON falls through to False (treat as potential leak)."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text("{not valid json")
        assert _is_injected_credentials_file(cred_file) is False

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns False."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        assert _is_injected_credentials_file(tmp_path / "nonexistent.json") is False

    def test_handles_non_dict_oauth_section(self, tmp_path: Path) -> None:
        """Non-dict claudeAiOauth returns False."""
        from terok_executor.credentials.vault_commands import _is_injected_credentials_file

        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps({"claudeAiOauth": "not a dict"}))
        assert _is_injected_credentials_file(cred_file) is False


class TestScanSkipsInjectedFile:
    """Verify scan_leaked_credentials skips terok-injected .credentials.json."""

    def test_skips_injected_credentials(self, tmp_path: Path) -> None:
        """Injected phantom credentials are NOT flagged as leaked."""
        from terok_executor import get_roster
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "",
                "subscriptionType": "max",
            }
        }
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        assert scan_leaked_credentials(tmp_path) == []

    def test_still_detects_real_credentials(self, tmp_path: Path) -> None:
        """Real OAuth tokens are still flagged even when file structure matches."""
        from terok_executor import get_roster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {"claudeAiOauth": {"accessToken": "sk-ant-real", "refreshToken": "rt-real"}}
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        leaked = scan_leaked_credentials(tmp_path)
        assert len(leaked) == 1
        assert leaked[0][0] == "claude"

    def test_skips_injected_codex_auth_json(self, tmp_path: Path) -> None:
        """Injected shared Codex auth.json is NOT flagged as leaked."""
        from terok_sandbox import CODEX_SHARED_OAUTH_MARKER

        from terok_executor import get_roster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers["codex"]
        route = roster.vault_routes["codex"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "tokens": {
                "access_token": CODEX_SHARED_OAUTH_MARKER,
                "refresh_token": CODEX_SHARED_OAUTH_MARKER,
                "id_token": "dummy.dummy.dummy",
            }
        }
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        assert scan_leaked_credentials(tmp_path) == []

    def test_codex_auth_json_with_live_api_key_is_still_leaked(self, tmp_path: Path) -> None:
        """Marker tokens do not hide a live top-level OPENAI_API_KEY."""
        from terok_sandbox import CODEX_SHARED_OAUTH_MARKER

        from terok_executor import get_roster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers["codex"]
        route = roster.vault_routes["codex"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "OPENAI_API_KEY": "sk-real-leak",
            "tokens": {
                "access_token": CODEX_SHARED_OAUTH_MARKER,
                "refresh_token": CODEX_SHARED_OAUTH_MARKER,
                "id_token": "dummy.dummy.dummy",
            },
        }
        (cred_dir / route.credential_file).write_text(json.dumps(cred))

        assert scan_leaked_credentials(tmp_path) == [("codex", cred_dir / route.credential_file)]

    def test_malformed_codex_auth_json_is_suspicious_not_crashing(self, tmp_path: Path) -> None:
        """Non-object auth.json roots are treated as leaks, not parser crashes."""
        from terok_executor import get_roster
        from terok_executor.credentials.vault_commands import scan_leaked_credentials

        roster = get_roster()
        auth = roster.auth_providers["codex"]
        route = roster.vault_routes["codex"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        (cred_dir / route.credential_file).write_text(json.dumps(["not", "an", "object"]))

        assert scan_leaked_credentials(tmp_path) == [("codex", cred_dir / route.credential_file)]


class TestCleanSkipsInjectedFile:
    """Verify the clean handler preserves injected .credentials.json."""

    def test_clean_preserves_injected_file(self, tmp_path: Path) -> None:
        """Clean removes real leaks but preserves injected phantom credentials."""
        from unittest.mock import patch

        from terok_executor import get_roster
        from terok_executor.credentials.auth import PHANTOM_CREDENTIALS_MARKER
        from terok_executor.credentials.vault_commands import _handle_clean

        roster = get_roster()
        auth = roster.auth_providers["claude"]
        route = roster.vault_routes["claude"]

        cred_dir = tmp_path / auth.host_dir_name
        cred_dir.mkdir()
        cred = {
            "claudeAiOauth": {
                "accessToken": PHANTOM_CREDENTIALS_MARKER,
                "refreshToken": "",
            }
        }
        cred_file = cred_dir / route.credential_file
        cred_file.write_text(json.dumps(cred))

        with patch("terok_executor.paths.mounts_dir", return_value=tmp_path):
            _handle_clean()

        # Injected file should still exist
        assert cred_file.is_file()


class TestFormatCredentials:
    """Verify _format_credentials() type-annotated status display."""

    def test_shows_credential_types(self, tmp_path: Path) -> None:
        """Formats credentials as 'name (type)' from the DB."""
        from terok_sandbox import CredentialDB

        from terok_executor.credentials.vault_commands import _format_credentials

        db_path = tmp_path / "creds.db"
        db = CredentialDB(db_path, passphrase=TEST_VAULT_PASSPHRASE)
        db.store_credential("default", "claude", {"type": "oauth", "access_token": "t"})
        db.store_credential("default", "vibe", {"type": "api_key", "key": "k"})
        db.close()

        status = MagicMock(
            credentials_stored=("claude", "vibe"),
            db_path=db_path,
        )
        result = _format_credentials(status)
        assert result == "claude (oauth), vibe (api_key)"

    def test_unknown_type_when_missing(self, tmp_path: Path) -> None:
        """Credentials without a type field show 'unknown'."""
        from terok_sandbox import CredentialDB

        from terok_executor.credentials.vault_commands import _format_credentials

        db_path = tmp_path / "creds.db"
        db = CredentialDB(db_path, passphrase=TEST_VAULT_PASSPHRASE)
        db.store_credential("default", "legacy", {"key": "k"})
        db.close()

        status = MagicMock(credentials_stored=("legacy",), db_path=db_path)
        assert "unknown" in _format_credentials(status)

    def test_empty_credentials(self) -> None:
        """Returns 'none stored' when no credentials exist."""
        from terok_executor.credentials.vault_commands import _format_credentials

        status = MagicMock(credentials_stored=())
        assert _format_credentials(status) == "none stored"

    def test_status_display_degrades_gracefully_on_db_error(self) -> None:
        """Status display shows plain names when its read-only DB connection fails."""
        from terok_executor.credentials.vault_commands import _format_credentials

        status = MagicMock(
            credentials_stored=("claude", "gh"),
            db_path=Path("/nonexistent/creds.db"),
        )
        assert _format_credentials(status) == "claude, gh"


class TestToVaultRoute:
    """Verify ``vault:`` schema parsing edge cases."""

    def test_socket_env_alone_accepted(self) -> None:
        """socket_env (without socket_path) is the new valid form."""
        route = _vault_route(
            "test",
            {
                "vault": {
                    "route_prefix": "test",
                    "upstream": "https://example.com",
                    "socket_env": "TEST_SOCKET",
                }
            },
        )
        assert route is not None
        assert route.socket_env == "TEST_SOCKET"

    def test_neither_socket_field_accepted(self) -> None:
        """Omitting socket_env is valid (agent has no socket transport)."""
        route = _vault_route(
            "test",
            {
                "vault": {
                    "route_prefix": "test",
                    "upstream": "https://example.com",
                }
            },
        )
        assert route is not None
        assert route.socket_env == ""

    def test_oauth_phantom_env_parsed(self) -> None:
        """oauth_phantom_env is parsed from YAML data."""
        route = _vault_route(
            "test",
            {
                "vault": {
                    "route_prefix": "test",
                    "upstream": "https://example.com",
                    "oauth_phantom_env": {"MY_OAUTH_TOKEN": True},
                }
            },
        )
        assert route is not None
        assert route.oauth_phantom_env == {"MY_OAUTH_TOKEN": True}

    def test_missing_required_field_raises(self) -> None:
        """Missing route_prefix or upstream raises ValidationError."""
        with pytest.raises(ValidationError, match="route_prefix"):
            _vault_route("test", {"vault": {"upstream": "https://x.com"}})
        with pytest.raises(ValidationError, match="upstream"):
            _vault_route("test", {"vault": {"route_prefix": "t"}})

    def test_no_vault_returns_none(self) -> None:
        """Agent without vault section returns None.

        An empty ``vault: {}`` block is invalid (route_prefix and upstream
        are required), so it raises rather than returning None.
        """
        assert _vault_route("test", {}) is None
        with pytest.raises(ValidationError):
            _vault_route("test", {"vault": {}})

    @pytest.mark.parametrize("field", ["path_upstreams", "oauth_extra_headers"])
    def test_optional_vault_maps_reject_falsy_non_mappings(self, field: str) -> None:
        """Falsy lists/strings must not be silently treated as absent maps."""
        with pytest.raises(ValidationError, match=field):
            _vault_route(
                "test",
                {
                    "vault": {
                        "route_prefix": "test",
                        "upstream": "https://example.com",
                        field: [],
                    }
                },
            )


class TestEnsureVaultRoutes:
    """Verify ensure_vault_routes writes routes.json to disk."""

    def test_writes_routes_json(self, tmp_path):
        """ensure_vault_routes() creates a valid routes.json file."""
        mock_cfg = MagicMock()
        mock_cfg.routes_path = tmp_path / "proxy" / "routes.json"

        from terok_executor.roster import ensure_vault_routes

        path = ensure_vault_routes(cfg=mock_cfg)

        assert path == mock_cfg.routes_path
        assert path.is_file()
        routes = json.loads(path.read_text())
        # Should have at least claude route from the YAML roster
        assert "claude" in routes
        assert "upstream" in routes["claude"]

    def test_falls_back_to_default_config(self, tmp_path, monkeypatch):
        """ensure_vault_routes(cfg=None) creates a SandboxConfig with standalone defaults."""
        import terok_sandbox

        mock_cfg = MagicMock()
        mock_cfg.routes_path = tmp_path / "proxy" / "routes.json"
        monkeypatch.setattr(terok_sandbox, "SandboxConfig", lambda: mock_cfg)

        from terok_executor.roster import ensure_vault_routes

        path = ensure_vault_routes()
        assert path.is_file()


class TestVaultHandlerCfgSignatures:
    """All vault command handlers accept a ``cfg`` keyword argument."""

    def test_all_leaf_handlers_accept_cfg(self) -> None:
        import inspect

        from terok_executor.credentials.vault_commands import VAULT_COMMANDS

        vault_group = VAULT_COMMANDS[0]
        for cmd in vault_group.children:
            # Skip the nested ``passphrase`` group; its leaves don't take cfg.
            if cmd.children:
                continue
            sig = inspect.signature(cmd.handler)
            assert "cfg" in sig.parameters, f"{cmd.handler.__name__} missing cfg param"


class TestVaultCommandsOverlay:
    """Executor's ``VAULT_COMMANDS`` overlays sandbox's vault subtree.

    Sandbox owns the verb registry and argparse schema (one source of
    truth for ``--key=``, structural nesting of ``vault passphrase``).
    Executor overrides handlers at the five shared paths via
    ``CommandTree.overlay`` and extends the vault subtree with two
    executor-only verbs (``routes`` / ``clean``).  Sandbox-only verbs
    (``unlock`` / ``lock`` / ``passphrase {seal,to-keyring,destroy}``)
    flow through unchanged — that's the property new sandbox commands
    rely on to reach ``terok-executor vault …`` zero-edit.
    """

    def test_sandbox_only_verbs_pass_through(self) -> None:
        """``unlock`` / ``lock`` (and the ``passphrase`` subgroup) pass through unchanged."""
        from terok_sandbox.commands import COMMANDS as SANDBOX_COMMANDS

        from terok_executor.credentials.vault_commands import VAULT_COMMANDS

        sandbox_vault = SANDBOX_COMMANDS.find_at(("vault",))
        executor_vault = VAULT_COMMANDS[0]
        sandbox_by_name = {c.name: c for c in sandbox_vault.children}
        executor_by_name = {c.name: c for c in executor_vault.children}
        for verb in ("unlock", "lock"):
            assert executor_by_name[verb].handler is sandbox_by_name[verb].handler
            assert executor_by_name[verb].args == sandbox_by_name[verb].args
        # Nested ``passphrase`` subgroup survives identically — every leaf
        # routes to the sandbox handler.
        sandbox_passphrase = {c.name: c for c in sandbox_by_name["passphrase"].children}
        executor_passphrase = {c.name: c for c in executor_by_name["passphrase"].children}
        for verb in ("seal", "to-keyring", "destroy"):
            assert executor_passphrase[verb].handler is sandbox_passphrase[verb].handler

    def test_shared_verbs_use_executor_handlers(self) -> None:
        """``start`` / ``stop`` / ``status`` / ``install`` / ``uninstall`` route to executor."""
        from terok_executor.credentials.vault_commands import (
            VAULT_COMMANDS,
            _handle_install,
            _handle_start,
            _handle_status,
            _handle_stop,
            _handle_uninstall,
        )

        expected = {
            "start": _handle_start,
            "stop": _handle_stop,
            "status": _handle_status,
            "install": _handle_install,
            "uninstall": _handle_uninstall,
        }
        by_name = {cmd.name: cmd for cmd in VAULT_COMMANDS[0].children}
        for verb, handler in expected.items():
            assert by_name[verb].handler is handler, f"{verb} should use executor handler"

    def test_executor_only_verbs_appended(self) -> None:
        """``routes`` and ``clean`` exist in executor's vault group but not sandbox's."""
        from terok_sandbox.commands import COMMANDS as SANDBOX_COMMANDS

        from terok_executor.credentials.vault_commands import VAULT_COMMANDS

        sandbox_names = {c.name for c in SANDBOX_COMMANDS.find_at(("vault",)).children}
        executor_names = {c.name for c in VAULT_COMMANDS[0].children}
        executor_only = executor_names - sandbox_names
        assert executor_only == {"routes", "clean"}

    def test_deep_path_shares_identity_with_shortcut(self) -> None:
        """``terok-executor sandbox vault X`` and ``terok-executor vault X`` resolve to
        the same ``CommandDef`` — the load-bearing property for wraps to apply uniformly."""
        from terok_executor.cli import COMMANDS

        deep = COMMANDS.find_at(("sandbox", "vault"))
        shortcut = COMMANDS.find_at(("vault",))
        assert deep is shortcut
