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
        manifest = generator.build_manifest(set(AGENT_PROTOCOLS), AGENT_PROTOCOLS, AUTHED_ENV)
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
        protocols = {**AGENT_PROTOCOLS, "opencode": "openai-chat", "pi": "openai-chat"}
        manifest = generator.build_manifest(set(protocols), protocols, AUTHED_ENV)
        assert _pair(manifest, "opencode", "openrouter")["ready"] is True
        assert _pair(manifest, "pi", "openrouter")["ready"] is True
        # openai serves openai-responses (not -chat) here, so opencode isn't ready on it.
        assert _pair(manifest, "opencode", "openai")["ready"] is False

    def test_incompatible_pairs_are_not_ready(self, generator: ModuleType) -> None:
        """A provider that does not serve the agent's protocol is not ready."""
        manifest = generator.build_manifest(set(AGENT_PROTOCOLS), AGENT_PROTOCOLS, AUTHED_ENV)
        assert _pair(manifest, "codex", "anthropic")["ready"] is False
        # openrouter serves openai-chat + anthropic-messages, not openai-responses.
        assert _pair(manifest, "codex", "openrouter")["ready"] is False
        # claude/vibe cannot use the openai (openai-responses-only) endpoint.
        assert _pair(manifest, "claude", "openai")["ready"] is False
        assert _pair(manifest, "vibe", "openai")["ready"] is False

    def test_pair_shape(self, generator: ModuleType) -> None:
        """Each pair carries the agent/provider context and the lone ``ready`` flag."""
        manifest = generator.build_manifest(set(AGENT_PROTOCOLS), AGENT_PROTOCOLS, AUTHED_ENV)
        assert _pair(manifest, "claude", "anthropic") == {
            "agent": "claude",
            "provider": "anthropic",
            "protocol": "anthropic-messages",
            "ready": True,
        }

    def test_uninstalled_agent_is_omitted(self, generator: ModuleType) -> None:
        """A protocol-known agent absent from the image is not enumerated at all."""
        # claude has a baked protocol but is absent from this image's selection.
        manifest = generator.build_manifest({"codex", "vibe"}, AGENT_PROTOCOLS, AUTHED_ENV)
        assert not [p for p in manifest["pairs"] if p["agent"] == "claude"]

    def test_full_matrix_is_emitted(self, generator: ModuleType) -> None:
        """Every installed protocol-speaking agent is paired with every authed provider."""
        manifest = generator.build_manifest(set(AGENT_PROTOCOLS), AGENT_PROTOCOLS, AUTHED_ENV)
        # 3 agents × 3 authenticated providers.
        assert len(manifest["pairs"]) == 9
        assert manifest["version"] == generator.MANIFEST_VERSION

    def test_pairs_are_deterministically_ordered(self, generator: ModuleType) -> None:
        """Pairs sort by (agent, provider) so the file is stable across runs."""
        manifest = generator.build_manifest(set(AGENT_PROTOCOLS), AGENT_PROTOCOLS, AUTHED_ENV)
        keys = [(p["agent"], p["provider"]) for p in manifest["pairs"]]
        assert keys == sorted(keys)

    def test_no_authenticated_providers_yields_no_pairs(self, generator: ModuleType) -> None:
        """With nothing authenticated there is no provider to pair an agent with."""
        manifest = generator.build_manifest(set(AGENT_PROTOCOLS), AGENT_PROTOCOLS, {"PATH": "/bin"})
        assert manifest["pairs"] == []

    def test_agents_without_protocol_are_absent(self, generator: ModuleType) -> None:
        """An agent missing from the baked protocol map is never enumerated.

        Tools and harnesses speak no fixed protocol, so they never appear even
        when installed — compatibility is undefined for them.
        """
        installed = set(AGENT_PROTOCOLS) | {"gh", "opencode"}
        manifest = generator.build_manifest(installed, AGENT_PROTOCOLS, AUTHED_ENV)
        named_agents = {p["agent"] for p in manifest["pairs"]}
        assert named_agents == set(AGENT_PROTOCOLS)


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

    def test_read_agent_protocols(self, generator: ModuleType, tmp_path: Path) -> None:
        """The baked map round-trips, dropping blank protocols."""
        baked = tmp_path / "agent-protocols.json"
        baked.write_text(json.dumps({"claude": "anthropic-messages", "copilot": ""}))
        assert generator._read_agent_protocols(baked) == {"claude": "anthropic-messages"}

    def test_read_agent_protocols_malformed(self, generator: ModuleType, tmp_path: Path) -> None:
        """Malformed JSON yields an empty map rather than raising."""
        baked = tmp_path / "agent-protocols.json"
        baked.write_text("{not json")
        assert generator._read_agent_protocols(baked) == {}

    def test_write_manifest_round_trips(self, generator: ModuleType, tmp_path: Path) -> None:
        """The manifest is written as JSON under a freshly created parent dir."""
        out = tmp_path / "nested" / "agents.json"
        manifest = generator.build_manifest(set(AGENT_PROTOCOLS), AGENT_PROTOCOLS, AUTHED_ENV)
        generator._write_manifest(out, manifest)
        assert json.loads(out.read_text()) == manifest

    def test_manifest_path_matches_contract(self, generator: ModuleType) -> None:
        """The generator writes to the agreed container manifest location."""
        assert generator.MANIFEST_PATH == CONTAINER_AGENTS_MANIFEST_PATH
