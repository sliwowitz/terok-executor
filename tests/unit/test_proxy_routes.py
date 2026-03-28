# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for credential proxy route parsing and routes.json generation."""

from __future__ import annotations

import json

from terok_agent.registry import get_registry


class TestProxyRoutesParsed:
    """Verify credential_proxy YAML sections are parsed into the registry."""

    def test_claude_route_exists(self) -> None:
        """Claude has a proxy route with Anthropic upstream."""
        reg = get_registry()
        route = reg.proxy_routes.get("claude")
        assert route is not None
        assert route.route_prefix == "claude"
        assert route.upstream == "https://api.anthropic.com"
        assert route.auth_header == "dynamic"
        assert "ANTHROPIC_API_KEY" in route.phantom_env
        assert route.base_url_env == "ANTHROPIC_BASE_URL"

    def test_codex_route_exists(self) -> None:
        """Codex has a proxy route with OpenAI upstream."""
        route = get_registry().proxy_routes.get("codex")
        assert route is not None
        assert route.upstream == "https://api.openai.com"
        assert "OPENAI_API_KEY" in route.phantom_env

    def test_gh_route_exists(self) -> None:
        """GitHub CLI has a proxy route with token-style auth."""
        route = get_registry().proxy_routes.get("gh")
        assert route is not None
        assert route.auth_prefix == "token "
        assert route.upstream == "https://api.github.com"

    def test_glab_route_exists(self) -> None:
        """GitLab CLI has a proxy route with PRIVATE-TOKEN header."""
        route = get_registry().proxy_routes.get("glab")
        assert route is not None
        assert route.auth_header == "PRIVATE-TOKEN"
        assert route.auth_prefix == ""
        assert route.route_prefix == "gl"

    def test_opencode_agents_have_routes(self) -> None:
        """Blablador and KISSKI have proxy routes."""
        reg = get_registry()
        for name in ("blablador", "kisski"):
            route = reg.proxy_routes.get(name)
            assert route is not None, f"{name} missing proxy route"
            assert route.credential_type == "api_key"

    def test_copilot_has_no_route(self) -> None:
        """Copilot has no credential_proxy section (tier-3, no base URL support)."""
        assert get_registry().proxy_routes.get("copilot") is None

    def test_claude_has_oauth_refresh(self) -> None:
        """Claude has oauth_refresh config for proactive token refresh."""
        route = get_registry().proxy_routes.get("claude")
        assert route is not None
        assert route.oauth_refresh is not None
        assert "token_url" in route.oauth_refresh
        assert "client_id" in route.oauth_refresh

    def test_codex_has_no_oauth_refresh(self) -> None:
        """Codex does not yet have oauth_refresh configured."""
        route = get_registry().proxy_routes.get("codex")
        assert route is not None
        assert route.oauth_refresh is None


class TestGenerateRoutesJson:
    """Verify routes.json generation."""

    def test_generates_valid_json(self) -> None:
        """generate_routes_json() produces parseable JSON with expected keys."""
        routes_json = get_registry().generate_routes_json()
        routes = json.loads(routes_json)
        assert "claude" in routes
        assert routes["claude"]["upstream"] == "https://api.anthropic.com"
        assert routes["claude"]["auth_header"] == "dynamic"

    def test_all_routes_have_upstream(self) -> None:
        """Every route in the JSON has an upstream field."""
        routes = json.loads(get_registry().generate_routes_json())
        for prefix, cfg in routes.items():
            assert "upstream" in cfg, f"Route '{prefix}' missing upstream"

    def test_glab_keyed_by_provider_name(self) -> None:
        """GitLab route is keyed by provider name 'glab'."""
        routes = json.loads(get_registry().generate_routes_json())
        assert "glab" in routes

    def test_claude_routes_json_includes_oauth_refresh(self) -> None:
        """Claude's routes.json entry includes oauth_refresh config."""
        routes = json.loads(get_registry().generate_routes_json())
        assert "oauth_refresh" in routes["claude"]
        assert routes["claude"]["oauth_refresh"]["client_id"]

    def test_gh_routes_json_omits_oauth_refresh(self) -> None:
        """Providers without oauth_refresh omit it from routes.json."""
        routes = json.loads(get_registry().generate_routes_json())
        assert "oauth_refresh" not in routes["gh"]


class TestEnsureProxyRoutes:
    """Verify ensure_proxy_routes writes routes.json to disk."""

    def test_writes_routes_json(self, tmp_path, monkeypatch):
        """ensure_proxy_routes() creates a valid routes.json file."""
        from unittest.mock import MagicMock

        import terok_sandbox

        mock_cfg = MagicMock()
        mock_cfg.proxy_routes_path = tmp_path / "proxy" / "routes.json"
        monkeypatch.setattr(terok_sandbox, "SandboxConfig", lambda: mock_cfg)

        from terok_agent.registry import ensure_proxy_routes

        path = ensure_proxy_routes()

        assert path == mock_cfg.proxy_routes_path
        assert path.is_file()
        routes = json.loads(path.read_text())
        # Should have at least claude route from the YAML registry
        assert "claude" in routes
        assert "upstream" in routes["claude"]
