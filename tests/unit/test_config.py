# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the executor's global-config write helpers.

[`set_global_image_agents`][terok_executor.set_global_image_agents] is
the surface every "set the default agents" entry point routes through —
``terok-executor agents set``, the terok wrapper's mirror, and the TUI
modal.  Round-trip preservation and parent-dir auto-creation are the
contracts the callers rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_executor.config import (
    get_global_image_agents,
    get_global_image_base_image,
    set_global_image_agents,
    writable_config_path,
)


@pytest.fixture
def override_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `TEROK_CONFIG_FILE` at a per-test path and clear sandbox's reader cache.

    Sandbox's [`read_config_section`][terok_sandbox.paths.read_config_section]
    caches results across calls, so a stale cache from an earlier test
    would mask the freshly-written file.  Reset before the test so the
    reader sees an empty starting state.
    """
    cfg = tmp_path / "config.yml"
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(cfg))

    from terok_sandbox import paths as sandbox_paths

    sandbox_paths._config_section_cache.clear()
    return cfg


class TestWritableConfigPath:
    """The path picker honours TEROK_CONFIG_FILE and falls back to namespace dir."""

    def test_honours_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``TEROK_CONFIG_FILE`` wins over the namespace lookup."""
        target = tmp_path / "custom.yml"
        monkeypatch.setenv("TEROK_CONFIG_FILE", str(target))
        assert writable_config_path() == target

    def test_falls_back_to_namespace_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No override → ``namespace_config_dir() / 'config.yml'``."""
        monkeypatch.delenv("TEROK_CONFIG_FILE", raising=False)
        from terok_sandbox.paths import namespace_config_dir

        assert writable_config_path() == namespace_config_dir() / "config.yml"


class TestSetGlobalImageAgents:
    """End-to-end write semantics: creates parent, preserves comments, round-trips."""

    def test_creates_file_when_missing(self, override_config: Path) -> None:
        """Missing target file is created, parent dirs included."""
        assert not override_config.exists()
        path = set_global_image_agents("claude,vibe")
        assert path == override_config
        assert override_config.read_text(encoding="utf-8").strip().startswith("image:")

    def test_creates_parent_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deeply-nested parent is created."""
        target = tmp_path / "deep" / "nested" / "config.yml"
        monkeypatch.setenv("TEROK_CONFIG_FILE", str(target))
        set_global_image_agents("all")
        assert target.is_file()

    def test_updates_existing_image_section(self, override_config: Path) -> None:
        """Updating an existing image.agents key replaces the value, keeps siblings."""
        override_config.write_text(
            "image:\n  base_image: ubuntu:24.04\n  agents: claude\n", encoding="utf-8"
        )
        set_global_image_agents("all,-vibe")
        content = override_config.read_text(encoding="utf-8")
        assert "agents: all,-vibe" in content
        assert "base_image: ubuntu:24.04" in content

    def test_preserves_other_top_level_sections(self, override_config: Path) -> None:
        """A pre-existing unrelated top-level section survives the write."""
        override_config.write_text(
            "tui:\n  default_tmux: true\nimage:\n  agents: claude\n", encoding="utf-8"
        )
        set_global_image_agents("vibe")
        content = override_config.read_text(encoding="utf-8")
        assert "default_tmux: true" in content
        assert "agents: vibe" in content

    def test_preserves_comments(self, override_config: Path) -> None:
        """``ruamel.yaml`` round-trip keeps top-level comments and ordering."""
        override_config.write_text(
            "# Top comment\nimage:\n  # inline comment\n  agents: claude\n",
            encoding="utf-8",
        )
        set_global_image_agents("vibe")
        content = override_config.read_text(encoding="utf-8")
        assert "# Top comment" in content
        assert "# inline comment" in content
        assert "agents: vibe" in content


class TestGetGlobalImageAgents:
    """The merged-stack reader resolves ``None`` for absent, value for present."""

    def test_returns_none_when_unset(self, override_config: Path) -> None:  # noqa: ARG002
        """No file → ``None``; distinguishes from explicit ``"all"``."""
        assert get_global_image_agents() is None

    def test_round_trip_after_set(self, override_config: Path) -> None:  # noqa: ARG002
        """``set`` then ``get`` returns the value that was written."""
        set_global_image_agents("claude,vibe")
        # Reset cache so the freshly-written file is re-read.
        from terok_sandbox import paths as sandbox_paths

        sandbox_paths._config_section_cache.clear()
        assert get_global_image_agents() == "claude,vibe"


class TestGetGlobalImageBaseImage:
    """The merged-stack reader resolves ``None`` for absent, value for present."""

    def test_returns_none_when_unset(self, override_config: Path) -> None:  # noqa: ARG002
        """No file → ``None``; callers apply the schema fallback themselves."""
        assert get_global_image_base_image() is None

    def test_returns_explicit_value(self, override_config: Path) -> None:
        """An explicit ``image.base_image`` is returned verbatim."""
        override_config.write_text("image:\n  base_image: ubuntu:24.04\n", encoding="utf-8")
        assert get_global_image_base_image() == "ubuntu:24.04"
