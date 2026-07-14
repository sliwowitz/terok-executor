# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-container ``providers`` command.

Exercises the ``providers`` script that renders the readiness manifest inside
task containers.  Like ``terok-agents`` it is staged into the image (not
importable by module name), so it is loaded from its source path and driven
against synthetic manifests, with output captured via ``capsys``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

import terok_executor.resources.scripts as _scripts_pkg

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip SGR escapes so assertions read the text a user actually sees."""
    return _ANSI_RE.sub("", text)


def _load_command() -> ModuleType:
    """Load the staged ``providers`` script as an importable module.

    Loading by path (rather than by package name) is required because the
    file is shipped into the container image, not exposed as a Python module.
    """
    script_path = Path(_scripts_pkg.__file__).parent / "providers"
    loader = SourceFileLoader("providers_command", str(script_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def command() -> ModuleType:
    """The loaded ``providers`` command module."""
    return _load_command()


def _pair(agent: str, provider: str, protocol: str, *, ready: bool) -> dict:
    """One manifest pair in the shape ``terok-agents`` writes."""
    return {"agent": agent, "provider": provider, "protocol": protocol, "ready": ready}


# A container where only anthropic is authenticated: claude is ready, the
# openai-protocol agents are blocked, and every other candidate provider in
# the baked universe is waiting to be authenticated on the host.
ANTHROPIC_ONLY_MANIFEST = {
    "version": 2,
    "pairs": [
        _pair("claude", "anthropic", "anthropic-messages", ready=True),
        _pair("codex", "anthropic", "openai-responses", ready=False),
        _pair("vibe", "anthropic", "openai-chat", ready=False),
    ],
    "protocols": [
        {
            "protocol": "anthropic-messages",
            "candidates": ["anthropic", "openrouter"],
            "authenticated": ["anthropic"],
        },
        {
            "protocol": "openai-chat",
            "candidates": ["blablador", "kisski", "mistral", "openai", "openrouter"],
            "authenticated": [],
        },
        {"protocol": "openai-responses", "candidates": ["openai"], "authenticated": []},
    ],
}


def _run(
    command: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    manifest: dict,
    *argv: str,
) -> tuple[str, str]:
    """Run the command against *manifest* and return ``(stdout, stderr)``."""
    manifest_path = tmp_path / "agents.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(command, "MANIFEST", manifest_path)
    assert command.main(["providers", *argv]) == 0
    captured = capsys.readouterr()
    return captured.out, captured.err


class TestLockedProviders:
    """Unauthenticated candidates surface with the host-side unlock command."""

    def test_all_lists_locked_candidates_per_protocol(
        self,
        command: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--all`` renders one locked row per protocol, headed by the command."""
        out, _ = _run(command, monkeypatch, tmp_path, capsys, ANTHROPIC_ONLY_MANIFEST, "--all")
        assert "Locked providers" in out
        assert "unlock on the host:" in out
        assert "terok auth" in out
        assert "openai-chat" in out
        assert "blablador, kisski, mistral, openai, openrouter" in out
        # The authenticated candidate is not offered for unlocking again.
        assert "anthropic-messages" in out
        assert "anthropic, openrouter" not in out

    def test_default_view_hints_with_a_count(
        self,
        command: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The default view compresses the locked list into a one-line pointer."""
        out, _ = _run(command, monkeypatch, tmp_path, capsys, ANTHROPIC_ONLY_MANIFEST)
        plain = _plain(out)
        # openrouter, blablador, kisski, mistral, openai — distinct across protocols.
        assert "5 locked providers - unlock on the host: terok auth <provider>" in plain
        assert "Also list non-ready providers: providers --all" in plain
        assert "Use: <agent> --provider <name>" in plain
        # The full per-protocol breakdown stays behind --all.
        assert "Locked providers" not in out

    def test_nothing_authenticated_still_lists_locked(
        self,
        command: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With zero pairs the locked section is the actionable part of the output."""
        manifest = {
            "version": 2,
            "pairs": [],
            "protocols": [
                {"protocol": "openai-chat", "candidates": ["openrouter"], "authenticated": []}
            ],
        }
        out, err = _run(command, monkeypatch, tmp_path, capsys, manifest)
        assert "authenticate a provider on the host" in err
        assert "Locked providers" in out
        assert "openrouter" in out

    def test_no_hint_when_everything_is_authenticated(
        self,
        command: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Fully-authenticated protocols leave nothing to unlock — no hint, no section."""
        manifest = {
            "version": 2,
            "pairs": [_pair("claude", "anthropic", "anthropic-messages", ready=True)],
            "protocols": [
                {
                    "protocol": "anthropic-messages",
                    "candidates": ["anthropic"],
                    "authenticated": ["anthropic"],
                }
            ],
        }
        for argv in ([], ["--all"]):
            out, _ = _run(command, monkeypatch, tmp_path, capsys, manifest, *argv)
            assert "locked" not in out.lower()

    def test_manifest_without_protocols_view_stays_quiet(
        self,
        command: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A manifest lacking the ``protocols`` view renders as before, hint-free."""
        manifest = {
            "version": 2,
            "pairs": [_pair("claude", "anthropic", "anthropic-messages", ready=True)],
        }
        out, _ = _run(command, monkeypatch, tmp_path, capsys, manifest, "--all")
        assert "Ready agent" in out
        assert "locked" not in out.lower()
