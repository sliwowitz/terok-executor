# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the provider layer (``resources/providers/*.yaml``).

The keystone invariant: each [`Provider`][terok_executor.roster.types.Provider]'s
``routes.json`` projection reproduces the historical agent-``vault`` route
entry, under the clean provider name.  Locking this *before* the loader is
switched to source routes from providers keeps the byte-identical-except-keys
contract honest — and keeps the sandbox vault untouched.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from terok_executor.credentials.auth import credential_provider
from terok_executor.roster.loader import _provider_route_entry, load_roster
from terok_executor.roster.schema import RawProvider
from terok_executor.roster.types import Provider, ProviderAuth

# Agent name → the clean provider name its vault block maps to (identity for
# names that don't rename: coderabbit, openrouter, blablador, kisski).
_AGENT_TO_PROVIDER = {
    "claude": "anthropic",
    "codex": "openai",
    "vibe": "mistral",
    "gh": "github",
    "glab": "gitlab",
    "sonar": "sonarcloud",
}

_EXPECTED_PROVIDERS = {
    "anthropic",
    "openai",
    "mistral",
    "github",
    "gitlab",
    "sonarcloud",
    "coderabbit",
    "openrouter",
    "blablador",
    "kisski",
}


class TestProviderRegistry:
    """The bundled provider set and its relationship to agent vault routes."""

    def test_all_expected_providers_load(self) -> None:
        assert set(load_roster().providers) == _EXPECTED_PROVIDERS

    def test_every_vault_route_has_a_matching_provider(self) -> None:
        roster = load_roster()
        for agent_name in roster.vault_routes:
            provider_name = _AGENT_TO_PROVIDER.get(agent_name, agent_name)
            assert provider_name in roster.providers, agent_name


class TestRouteEntryEquivalence:
    """The byte-identical-except-keys invariant, checked route by route."""

    def test_projection_reproduces_current_route(self) -> None:
        roster = load_roster()
        for agent_name, route in roster.vault_routes.items():
            provider = roster.providers[_AGENT_TO_PROVIDER.get(agent_name, agent_name)]
            expected = {
                "upstream": route.upstream,
                "auth_header": route.auth_header,
                "auth_prefix": route.auth_prefix,
                "path_upstreams": route.path_upstreams or None,
                "oauth_extra_headers": route.oauth_extra_headers or None,
                "oauth_refresh": route.oauth_refresh or None,
            }
            assert _provider_route_entry(provider).model_dump() == expected, agent_name


class TestWireAuth:
    """The OAuth-or-API-key header derivation that replaces ``auth_header: dynamic``."""

    def test_dual_header_modes_are_dynamic(self) -> None:
        # Anthropic: OAuth on Authorization vs API key on x-api-key → sentinel.
        header, prefix, extra = load_roster().providers["anthropic"].wire_auth()
        assert header == "dynamic"
        assert prefix == ""
        assert extra == {"anthropic-beta": "oauth-2025-04-20"}

    def test_single_mode_is_verbatim(self) -> None:
        assert load_roster().providers["mistral"].wire_auth() == ("Authorization", "Bearer ", {})

    def test_same_header_dual_mode_is_not_dynamic(self) -> None:
        provider = Provider(
            name="x",
            upstream="https://example.test",
            oauth_auth=ProviderAuth(header="Authorization", prefix="Bearer "),
            api_key_auth=ProviderAuth(header="Authorization", prefix="Bearer "),
        )
        assert provider.wire_auth() == ("Authorization", "Bearer ", {})

    def test_no_auth_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="no auth mode"):
            Provider(name="x", upstream="https://example.test").wire_auth()

    def test_same_header_differing_prefix_raises(self) -> None:
        # Same header but disagreeing prefixes can't both be serialised into the
        # single routes.json auth_prefix — must fail loud, not pick one silently.
        provider = Provider(
            name="x",
            upstream="https://example.test",
            oauth_auth=ProviderAuth(header="Authorization", prefix="Bearer "),
            api_key_auth=ProviderAuth(header="Authorization", prefix="token "),
        )
        with pytest.raises(ValueError, match="different prefix"):
            provider.wire_auth()


class TestSchemaStrictness:
    """``RawProvider`` rejects typos and credential-less routes."""

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawProvider.model_validate(
                {
                    "upstream": "https://example.test",
                    "auth": {"api_key": {"header": "Authorization"}},
                    "oops": 1,
                }
            )

    def test_auth_requires_a_mode(self) -> None:
        with pytest.raises(ValidationError):
            RawProvider.model_validate({"upstream": "https://example.test", "auth": {}})

    def test_missing_upstream_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawProvider.model_validate({"auth": {"api_key": {"header": "Authorization"}}})


class TestCredentialProviderResolution:
    """``credential_provider`` maps an auth target (agent) to its DB credential key.

    This is what keeps the auth *write* side aligned with the provider-keyed
    routes: ``terok-executor auth claude`` must store under ``anthropic`` so
    ``routed = stored & routes`` intersects at runtime.
    """

    def test_native_resolves_to_default_provider(self) -> None:
        assert credential_provider("claude") == "anthropic"
        assert credential_provider("codex") == "openai"
        assert credential_provider("vibe") == "mistral"

    def test_tool_resolves_to_its_provider(self) -> None:
        assert credential_provider("gh") == "github"
        assert credential_provider("glab") == "gitlab"

    def test_opencode_shim_is_identity(self) -> None:
        # openrouter/blablador/kisski provider names equal their agent names.
        assert credential_provider("blablador") == "blablador"

    def test_unbound_name_passes_through(self) -> None:
        # Harnesses (no binding) and unknown names resolve to themselves.
        assert credential_provider("opencode") == "opencode"
        assert credential_provider("not-an-agent") == "not-an-agent"
