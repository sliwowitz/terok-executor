# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-container agent/provider readiness manifest generator.

Exercises the ``terok-agents`` script that runs inside task containers.  The
script is staged into the image (not importable by module name), so it is
loaded directly from its source path and its pure derivation function is
driven with synthetic copies of the three runtime inputs.
"""

from __future__ import annotations

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

import terok_executor.resources.scripts as _scripts_pkg
from tests.constants import CONTAINER_AGENTS_MANIFEST_PATH

# Container-side base URLs the env builder would emit; only the protocol
# suffix of the variable name matters for compatibility, but realistic values
# keep the fixtures honest.
_LOOPBACK = "http://localhost:9419"


def _load_generator() -> ModuleType:
    """Load the staged ``terok-agents`` script as an importable module.

    Loading by path (rather than by package name) is required because the
    file is shipped into the container image, not exposed as a Python module.
    """
    script_path = Path(_scripts_pkg.__file__).parent / "terok-agents.py"
    loader = SourceFileLoader("terok_agents_generator", str(script_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    """The loaded ``terok-agents`` generator module."""
    return _load_generator()


# Roster wire protocols, mirrored into the baked map the script reads.
AGENT_PROTOCOLS = {
    "claude": "anthropic-messages",
    "codex": "openai-responses",
    "vibe": "openai-chat",
}


def _facts(protocols: dict[str, str]) -> dict[str, dict[str, str | None]]:
    """Wrap a ``name → protocol`` map into the baked per-agent facts shape.

    Each entry gains a synthetic banner ``label`` so the rollup and banner can
    be exercised without the real roster.
    """
    return {
        name: {"protocol": protocol, "label": f"  {name} - {name.title()}"}
        for name, protocol in protocols.items()
    }


# The baked facts the generator reads: the three protocol-speaking natives.
AGENT_FACTS = _facts(AGENT_PROTOCOLS)

# The baked protocol → candidate-providers universe (mirrors the real roster).
AGENT_UNIVERSE = {
    "anthropic-messages": ["anthropic", "openrouter"],
    "openai-responses": ["openai"],
    "openai-chat": ["blablador", "kisski", "mistral", "openai", "openrouter"],
}

# A realistic in-container environment: anthropic (anthropic-messages),
# openai (openai-responses), and openrouter (both openai-chat and
# anthropic-messages) are authenticated; unrelated vars are noise.
AUTHED_ENV = {
    "TEROK_PROVIDER_ANTHROPIC_TOKEN": "terok-p-aaa",
    "TEROK_PROVIDER_ANTHROPIC_BASE_ANTHROPIC_MESSAGES": f"{_LOOPBACK}/v1",
    "TEROK_PROVIDER_OPENAI_TOKEN": "terok-p-bbb",
    "TEROK_PROVIDER_OPENAI_BASE_OPENAI_RESPONSES": f"{_LOOPBACK}/v1",
    "TEROK_PROVIDER_OPENROUTER_TOKEN": "terok-p-ccc",
    "TEROK_PROVIDER_OPENROUTER_BASE_OPENAI_CHAT": f"{_LOOPBACK}/api/v1",
    "TEROK_PROVIDER_OPENROUTER_BASE_ANTHROPIC_MESSAGES": f"{_LOOPBACK}/api",
    "PATH": "/usr/local/bin",
    "TEROK_UNRESTRICTED": "1",
    "TEROK_OC_OPENROUTER_BASE_URL": f"{_LOOPBACK}/api/v1",
}


def _pair(manifest: dict, agent: str, provider: str) -> dict:
    """Return the single manifest pair for *agent* × *provider*."""
    matches = [p for p in manifest["pairs"] if p["agent"] == agent and p["provider"] == provider]
    assert len(matches) == 1, f"expected one {agent}/{provider} pair, got {matches}"
    return matches[0]


class TestBuildManifest:
    """The pure derivation: three runtime inputs → the readiness matrix."""

    def test_compatible_pairs_are_ready(self, generator: ModuleType) -> None:
        """An installed agent whose protocol the authed provider serves is ready."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        # claude speaks anthropic-messages → anthropic and openrouter serve it.
        assert _pair(manifest, "claude", "anthropic")["ready"] is True
        assert _pair(manifest, "claude", "openrouter")["ready"] is True
        # codex speaks openai-responses → only openai serves it.
        assert _pair(manifest, "codex", "openai")["ready"] is True
        # vibe speaks openai-chat → only openrouter serves it here.
        assert _pair(manifest, "vibe", "openrouter")["ready"] is True

    def test_harness_agents_surface_paired_with_openai_chat_providers(
        self, generator: ModuleType
    ) -> None:
        """Harnesses declare openai-chat, so they appear paired with every authed
        openai-chat provider — the universal cross-provider path (must not be
        omitted from the manifest, as they were when harnesses had no protocol)."""
        facts = _facts({**AGENT_PROTOCOLS, "opencode": "openai-chat", "pi": "openai-chat"})
        manifest = generator.build_manifest(set(facts), facts, AGENT_UNIVERSE, AUTHED_ENV)
        assert _pair(manifest, "opencode", "openrouter")["ready"] is True
        assert _pair(manifest, "pi", "openrouter")["ready"] is True
        # openai serves openai-responses (not -chat) here, so opencode isn't ready on it.
        assert _pair(manifest, "opencode", "openai")["ready"] is False

    def test_incompatible_pairs_are_not_ready(self, generator: ModuleType) -> None:
        """A provider that does not serve the agent's protocol is not ready."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        assert _pair(manifest, "codex", "anthropic")["ready"] is False
        # openrouter serves openai-chat + anthropic-messages, not openai-responses.
        assert _pair(manifest, "codex", "openrouter")["ready"] is False
        # claude/vibe cannot use the openai (openai-responses-only) endpoint.
        assert _pair(manifest, "claude", "openai")["ready"] is False
        assert _pair(manifest, "vibe", "openai")["ready"] is False

    def test_pair_shape(self, generator: ModuleType) -> None:
        """Each pair carries the agent/provider context and the lone ``ready`` flag."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        assert _pair(manifest, "claude", "anthropic") == {
            "agent": "claude",
            "provider": "anthropic",
            "protocol": "anthropic-messages",
            "ready": True,
        }

    def test_uninstalled_agent_is_omitted(self, generator: ModuleType) -> None:
        """A protocol-known agent absent from the image is not enumerated at all."""
        # claude has a baked protocol but is absent from this image's selection.
        manifest = generator.build_manifest(
            {"codex", "vibe"}, AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        assert not [p for p in manifest["pairs"] if p["agent"] == "claude"]

    def test_full_matrix_is_emitted(self, generator: ModuleType) -> None:
        """Every installed protocol-speaking agent is paired with every authed provider."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        # 3 agents × 3 authenticated providers.
        assert len(manifest["pairs"]) == 9
        assert manifest["version"] == generator.MANIFEST_VERSION

    def test_pairs_are_deterministically_ordered(self, generator: ModuleType) -> None:
        """Pairs sort by (agent, provider) so the file is stable across runs."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        keys = [(p["agent"], p["provider"]) for p in manifest["pairs"]]
        assert keys == sorted(keys)

    def test_no_authenticated_providers_yields_no_pairs(self, generator: ModuleType) -> None:
        """With nothing authenticated there is no provider to pair an agent with."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, {"PATH": "/bin"}
        )
        assert manifest["pairs"] == []

    def test_agents_absent_from_facts_are_not_paired(self, generator: ModuleType) -> None:
        """An agent missing from the baked facts map is never paired.

        Tools (gh) are baked into the dev-tool section, not the facts map, so
        they never contribute a pair even when installed.
        """
        installed = set(AGENT_FACTS) | {"gh"}
        manifest = generator.build_manifest(installed, AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV)
        named_agents = {p["agent"] for p in manifest["pairs"]}
        assert named_agents == set(AGENT_FACTS)


class TestAgentRollup:
    """The ``agents`` view: usable/reason per installed agent, for the banner."""

    def _rollup(self, manifest: dict, name: str) -> dict:
        matches = [a for a in manifest["agents"] if a["name"] == name]
        assert len(matches) == 1, f"expected one rollup for {name}, got {matches}"
        return matches[0]

    def test_usable_when_a_provider_serves_the_protocol(self, generator: ModuleType) -> None:
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        claude = self._rollup(manifest, "claude")
        assert claude["usable"] is True
        assert claude["reason"] == ""

    def test_unusable_reason_names_the_protocol_not_a_provider(self, generator: ModuleType) -> None:
        """Reason names the wire protocol — any compatible provider satisfies it,
        so it must not name a single 'default' provider."""
        # Only anthropic is authenticated → codex (openai-responses) is out.
        env = {
            "TEROK_PROVIDER_ANTHROPIC_TOKEN": "terok-p-aaa",
            "TEROK_PROVIDER_ANTHROPIC_BASE_ANTHROPIC_MESSAGES": f"{_LOOPBACK}/v1",
        }
        manifest = generator.build_manifest(set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, env)
        codex = self._rollup(manifest, "codex")
        assert codex["usable"] is False
        assert codex["reason"] == "needs an openai-responses provider"
        # claude can reach anthropic → still usable.
        assert self._rollup(manifest, "claude")["usable"] is True

    def test_usable_via_any_compatible_provider_not_the_default(
        self, generator: ModuleType
    ) -> None:
        """vibe (openai-chat) is usable because openrouter serves openai-chat —
        even though 'mistral' (its roster default) is not authenticated."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        vibe = self._rollup(manifest, "vibe")
        assert vibe["usable"] is True
        assert vibe["reason"] == ""

    def test_agent_without_protocol_is_always_usable(self, generator: ModuleType) -> None:
        """A pure frontend (toad: no protocol) has nothing to assess → bright."""
        facts = {"toad": {"protocol": None, "label": "  toad - Toad"}}
        manifest = generator.build_manifest({"toad"}, facts, {}, {"PATH": "/bin"})
        toad = self._rollup(manifest, "toad")
        assert toad["usable"] is True
        assert toad["reason"] == ""

    def test_banner_keeps_usable_labels_and_dims_the_rest(self, generator: ModuleType) -> None:
        """Usable agents print verbatim; unusable ones are stripped, dimmed, annotated."""
        env = {
            "TEROK_PROVIDER_ANTHROPIC_TOKEN": "terok-p-aaa",
            "TEROK_PROVIDER_ANTHROPIC_BASE_ANTHROPIC_MESSAGES": f"{_LOOPBACK}/v1",
        }
        facts = {
            "claude": {
                "protocol": "anthropic-messages",
                "label": "  \033[36mclaude\033[0m - Claude Code",
            },
            "codex": {
                "protocol": "openai-responses",
                "label": "  \033[36mcodex\033[0m - OpenAI Codex",
            },
        }
        manifest = generator.build_manifest(set(facts), facts, AGENT_UNIVERSE, env)
        banner = generator.render_agent_banner(manifest["agents"])
        lines = banner.splitlines()
        # claude usable → its cyan label is printed as-is.
        assert "  \033[36mclaude\033[0m - Claude Code" in lines[0]
        # codex unusable → dimmed, embedded ANSI stripped, reason appended.
        assert lines[1].startswith("\033[2m")
        assert "\033[36m" not in lines[1]
        assert "(needs an openai-responses provider)" in lines[1]

    def test_banner_tolerates_non_list_input(self, generator: ModuleType) -> None:
        assert generator.render_agent_banner(None) == ""


class TestProtocolRollup:
    """The ``protocols`` view: candidate providers per protocol, for the section."""

    def _row(self, manifest: dict, protocol: str) -> dict:
        matches = [r for r in manifest["protocols"] if r["protocol"] == protocol]
        assert len(matches) == 1, f"expected one row for {protocol}, got {matches}"
        return matches[0]

    def test_lists_candidates_and_marks_authenticated(self, generator: ModuleType) -> None:
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        chat = self._row(manifest, "openai-chat")
        assert chat["candidates"] == ["blablador", "kisski", "mistral", "openai", "openrouter"]
        # Of those, only openai + openrouter are authenticated in AUTHED_ENV.
        assert chat["authenticated"] == ["openai", "openrouter"]

    def test_candidates_listed_even_when_none_authenticated(self, generator: ModuleType) -> None:
        """The point of the universe: surface providers to authenticate when zero are."""
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, {"PATH": "/x"}
        )
        chat = self._row(manifest, "openai-chat")
        assert chat["candidates"]  # non-empty
        assert chat["authenticated"] == []

    def test_render_marks_authed_magenta_and_rest_dim(self, generator: ModuleType) -> None:
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        out = generator.render_provider_protocols(manifest["protocols"])
        assert "openai-chat" in out
        # authenticated provider → magenta; unauthenticated candidate → dim.
        assert "\033[35mopenrouter\033[0m" in out
        assert "\033[2mblablador\033[0m" in out

    def test_render_tolerates_non_list_input(self, generator: ModuleType) -> None:
        assert generator.render_provider_protocols(None) == ""


class TestEnvironmentScanning:
    """Parsing the two families of ``TEROK_PROVIDER_*`` variables."""

    def test_authenticated_providers_from_token_vars(self, generator: ModuleType) -> None:
        """Provider names come from ``*_TOKEN`` vars, lower-cased, base vars ignored."""
        providers = generator._authenticated_providers(AUTHED_ENV)
        assert providers == {"anthropic", "openai", "openrouter"}

    def test_served_protocols_from_base_vars(self, generator: ModuleType) -> None:
        """Served protocols come from ``*_BASE_<PROTOCOL>`` vars in env-var token form."""
        served = generator._served_protocols(AUTHED_ENV)
        assert served["anthropic"] == {"ANTHROPIC_MESSAGES"}
        assert served["openai"] == {"OPENAI_RESPONSES"}
        assert served["openrouter"] == {"OPENAI_CHAT", "ANTHROPIC_MESSAGES"}

    def test_protocol_token_normalisation(self, generator: ModuleType) -> None:
        """A roster protocol maps to the base-var token form."""
        assert generator._protocol_token("anthropic-messages") == "ANTHROPIC_MESSAGES"
        assert generator._protocol_token("openai-chat") == "OPENAI_CHAT"


class TestSourcesAndSink:
    """Reading the on-disk inputs and writing the manifest."""

    def test_read_installed_agents(self, generator: ModuleType, tmp_path: Path) -> None:
        """The installed set is parsed from ``TEROK_INSTALLED_AGENTS``."""
        env_file = tmp_path / "installed.env"
        env_file.write_text("TEROK_INSTALLED_AGENTS=claude,codex, vibe ,\n")
        assert generator._read_installed_agents(env_file) == {"claude", "codex", "vibe"}

    def test_read_installed_agents_missing_file(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        """A missing manifest yields an empty set rather than raising."""
        assert generator._read_installed_agents(tmp_path / "absent.env") == set()

    def test_read_agent_facts(self, generator: ModuleType, tmp_path: Path) -> None:
        """The baked facts round-trip, normalising a missing protocol to ``None``."""
        baked = tmp_path / "agent-protocols.json"
        baked.write_text(
            json.dumps(
                {
                    "claude": {
                        "protocol": "anthropic-messages",
                        "label": "  claude - Claude Code",
                    },
                    "toad": {"label": "  toad - Toad"},
                }
            )
        )
        facts = generator._read_agent_facts(baked)
        assert facts["claude"]["protocol"] == "anthropic-messages"
        # Missing protocol key defaults to None, not KeyError.
        assert facts["toad"]["protocol"] is None
        assert facts["toad"]["label"] == "  toad - Toad"

    def test_read_agent_facts_malformed(self, generator: ModuleType, tmp_path: Path) -> None:
        """Malformed JSON yields an empty map rather than raising."""
        baked = tmp_path / "agent-protocols.json"
        baked.write_text("{not json")
        assert generator._read_agent_facts(baked) == {}

    def test_read_provider_protocols(self, generator: ModuleType, tmp_path: Path) -> None:
        """The baked protocol→providers universe round-trips; non-list values drop."""
        baked = tmp_path / "provider-protocols.json"
        baked.write_text(
            json.dumps({"anthropic-messages": ["anthropic", "openrouter"], "bogus": "nope"})
        )
        universe = generator._read_provider_protocols(baked)
        assert universe == {"anthropic-messages": ["anthropic", "openrouter"]}

    def test_read_provider_protocols_malformed(self, generator: ModuleType, tmp_path: Path) -> None:
        """Malformed JSON yields an empty map rather than raising."""
        baked = tmp_path / "provider-protocols.json"
        baked.write_text("{not json")
        assert generator._read_provider_protocols(baked) == {}

    def test_write_manifest_round_trips(self, generator: ModuleType, tmp_path: Path) -> None:
        """The manifest is written as JSON under a freshly created parent dir."""
        out = tmp_path / "nested" / "agents.json"
        manifest = generator.build_manifest(
            set(AGENT_FACTS), AGENT_FACTS, AGENT_UNIVERSE, AUTHED_ENV
        )
        generator._write_manifest(out, manifest)
        assert json.loads(out.read_text()) == manifest

    def test_manifest_path_matches_contract(self, generator: ModuleType) -> None:
        """The generator writes to the agreed container manifest location."""
        assert generator.MANIFEST_PATH == CONTAINER_AGENTS_MANIFEST_PATH
