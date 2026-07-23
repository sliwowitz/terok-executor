# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Roster → shield egress projection (``deny_to_vault_hosts`` / ``compose_egress``).

The projection is generic: no provider is special-cased in code.  Behaviour is
driven entirely by declarative roster fields — ``shared_domain`` and the
per-task ``exposed_providers`` set decide what is *not* denied; ``egress.allow``
decides what is additionally allowed.  These tests pin each rule branch on
synthetic providers and then confirm the real bundled roster projects the way
the design intends.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from terok_executor.roster.loader import AgentRoster, load_roster
from terok_executor.roster.schema import RawProvider
from terok_executor.roster.types import EgressProjection, Provider

# Real-roster hosts asserted below — the expected output of the projection over
# the bundled provider YAMLs (dedicated API endpoints, refresh/backend hosts,
# and the two shared-domain apexes).
ANTHROPIC_API = "api.anthropic.com"
ANTHROPIC_REFRESH = "platform.claude.com"  # anthropic oauth_refresh.token_url
OPENAI_API = "api.openai.com"
OPENAI_REFRESH = "auth.openai.com"  # openai oauth_refresh.token_url
OPENAI_BACKEND = "chatgpt.com"  # openai path_upstreams
GITHUB_API = "api.github.com"
GITLAB_APEX = "gitlab.com"  # shared_domain: true
SONARCLOUD_APEX = "sonarcloud.io"  # shared_domain: true


class TestRelayedHosts:
    """``Provider.relayed_hosts`` — every host the vault relays to for a provider."""

    def test_upstream_host(self) -> None:
        """The ``upstream`` host is always included."""
        p = Provider(name="x", upstream="https://api.example.com")
        assert p.relayed_hosts() == frozenset({"api.example.com"})

    def test_includes_path_upstreams(self) -> None:
        """Every ``path_upstreams`` override host is a relayed host too."""
        p = Provider(
            name="x",
            upstream="https://api.example.com",
            path_upstreams={"/backend/": "https://backend.example.net"},
        )
        assert p.relayed_hosts() == frozenset({"api.example.com", "backend.example.net"})

    def test_includes_oauth_refresh_token_url(self) -> None:
        """The OAuth-refresh ``token_url`` host is relayed (the vault refreshes, not the agent)."""
        p = Provider(
            name="x",
            upstream="https://api.example.com",
            oauth_refresh={"token_url": "https://auth.example.org/oauth/token"},
        )
        assert p.relayed_hosts() == frozenset({"api.example.com", "auth.example.org"})

    def test_unparseable_url_dropped(self) -> None:
        """A value with no parseable host contributes nothing (never a bogus deny)."""
        p = Provider(name="x", upstream="not-a-url")
        assert p.relayed_hosts() == frozenset()


class TestDenyToVaultHosts:
    """``AgentRoster.deny_to_vault_hosts`` — the t20 security-deny projection."""

    def test_dedicated_relayed_host_denied(self) -> None:
        """A plain relayed provider contributes its host to the deny set."""
        roster = AgentRoster(_providers={"x": Provider(name="x", upstream="https://api.x.com")})
        assert roster.deny_to_vault_hosts() == frozenset({"api.x.com"})

    def test_shared_domain_skipped(self) -> None:
        """``shared_domain`` providers are never host-denied (apex serves git/docs too)."""
        roster = AgentRoster(
            _providers={
                "g": Provider(name="g", upstream="https://gitlab.example", shared_domain=True)
            }
        )
        assert roster.deny_to_vault_hosts() == frozenset()

    def test_bare_provider_name_does_not_skip(self) -> None:
        """The exposed set is keyed by roster-entry (agent) name, not provider name.

        A provider name with no matching agent maps to nothing, so its host stays
        denied — only an exposed *agent* frees the provider it binds.
        """
        roster = AgentRoster(_providers={"a": Provider(name="a", upstream="https://api.a.com")})
        assert roster.deny_to_vault_hosts(
            exposed_credential_providers=frozenset({"a"})
        ) == frozenset({"api.a.com"})

    def test_multiple_providers_union_all_relayed_hosts(self) -> None:
        """The deny set is the union of every non-skipped provider's relayed hosts."""
        roster = AgentRoster(
            _providers={
                "a": Provider(name="a", upstream="https://api.a.com"),
                "b": Provider(
                    name="b",
                    upstream="https://api.b.com",
                    oauth_refresh={"token_url": "https://auth.b.com/t"},
                ),
            }
        )
        assert roster.deny_to_vault_hosts() == frozenset({"api.a.com", "api.b.com", "auth.b.com"})


class TestComposeEgress:
    """``AgentRoster.compose_egress`` — the bundled, deterministic projection."""

    def test_returns_sorted_deduped_projection(self) -> None:
        """``deny_to_vault`` is a sorted tuple (deterministic bundle)."""
        roster = AgentRoster(
            _providers={
                "b": Provider(name="b", upstream="https://api.b.com"),
                "a": Provider(name="a", upstream="https://api.a.com"),
            }
        )
        proj = roster.compose_egress()
        assert isinstance(proj, EgressProjection)
        assert proj.deny_to_vault == ("api.a.com", "api.b.com")

    def test_provider_allow_gathers_egress_allow_sorted_deduped(self) -> None:
        """``provider_allow`` is the sorted, de-duplicated union of every ``egress_allow``."""
        roster = AgentRoster(
            _providers={
                "a": Provider(
                    name="a", upstream="https://api.a.com", egress_allow=("z.example", "a.example")
                ),
                "b": Provider(name="b", upstream="https://api.b.com", egress_allow=("a.example",)),
            }
        )
        assert roster.compose_egress().provider_allow == ("a.example", "z.example")

    def test_empty_roster_empty_projection(self) -> None:
        """An empty roster yields an empty projection, not an error."""
        assert AgentRoster().compose_egress() == EgressProjection(
            deny_to_vault=(), provider_allow=()
        )


class TestRealRosterProjection:
    """The projection over the real bundled provider YAMLs matches the design."""

    def test_dedicated_api_hosts_denied(self) -> None:
        """Dedicated provider API endpoints are denied directly (defense-in-depth)."""
        deny = load_roster().deny_to_vault_hosts()
        assert {ANTHROPIC_API, OPENAI_API, GITHUB_API} <= deny

    def test_refresh_and_backend_hosts_denied(self) -> None:
        """OAuth-refresh + path-backend hosts are relayed too, so they are denied."""
        deny = load_roster().deny_to_vault_hosts()
        assert {ANTHROPIC_REFRESH, OPENAI_REFRESH, OPENAI_BACKEND} <= deny

    def test_shared_domain_apexes_not_denied(self) -> None:
        """gitlab.com / sonarcloud.io ride mixed apexes — never host-denied."""
        deny = load_roster().deny_to_vault_hosts()
        assert GITLAB_APEX not in deny
        assert SONARCLOUD_APEX not in deny

    def test_exposing_claude_frees_anthropic_hosts(self) -> None:
        """Exposing the ``claude`` agent (subscription mode) frees the provider it
        binds (anthropic), mapped via provider_binding — not other providers."""
        deny = load_roster().deny_to_vault_hosts(exposed_credential_providers=frozenset({"claude"}))
        assert ANTHROPIC_API not in deny
        assert ANTHROPIC_REFRESH not in deny
        assert OPENAI_API in deny

    def test_exposing_codex_frees_openai_hosts(self) -> None:
        """The mapping works for codex → openai too (a second real binding)."""
        deny = load_roster().deny_to_vault_hosts(exposed_credential_providers=frozenset({"codex"}))
        assert OPENAI_API not in deny
        assert OPENAI_REFRESH not in deny
        assert ANTHROPIC_API in deny


class TestRawProviderEgress:
    """``egress.allow`` in a provider YAML projects onto ``Provider.egress_allow``."""

    def _raw(self, extra: dict) -> dict:
        """A minimal valid provider dict merged with *extra*."""
        return {
            "upstream": "https://api.x.com",
            "auth": {"api_key": {"header": "x-api-key"}},
            **extra,
        }

    def test_egress_allow_projects_to_provider(self) -> None:
        """A declared ``egress.allow`` list becomes the provider's ``egress_allow`` tuple."""
        raw = RawProvider.model_validate(self._raw({"egress": {"allow": ["telemetry.x.com"]}}))
        assert raw.to_dataclass(name="x").egress_allow == ("telemetry.x.com",)

    def test_egress_absent_defaults_empty(self) -> None:
        """A provider with no ``egress`` block projects to an empty tuple."""
        raw = RawProvider.model_validate(self._raw({}))
        assert raw.to_dataclass(name="x").egress_allow == ()

    def test_egress_strict_keys_reject_typo(self) -> None:
        """A typo inside ``egress`` fails fast (strict keys), not silently ignored."""
        with pytest.raises(ValidationError):
            RawProvider.model_validate(self._raw({"egress": {"allwo": ["telemetry.x.com"]}}))
