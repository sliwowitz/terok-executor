# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent instruction resolution module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from terok_agent.provider.instructions import (
    bundled_default_instructions,
    has_custom_instructions,
    resolve_instructions,
)
from tests.constants import NONEXISTENT_PROJECT_ROOT, WORKSPACE_ROOT

DEFAULT_INSTRUCTIONS = bundled_default_instructions()


def resolve_with_project_file(
    config: dict[str, object],
    *,
    file_text: str | None = None,
) -> str:
    """Resolve instructions with an optional ``instructions.md`` project file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        if file_text is not None:
            (root / "instructions.md").write_text(file_text, encoding="utf-8")
        return resolve_instructions(config, "claude", project_root=root)


class TestBundledDefault:
    """Tests for bundled default instructions."""

    def test_bundled_default_exists(self) -> None:
        """Bundled default instructions exist and are non-trivial."""
        assert isinstance(DEFAULT_INSTRUCTIONS, str)
        assert len(DEFAULT_INSTRUCTIONS) > 100
        assert "terok" in DEFAULT_INSTRUCTIONS

    def test_bundled_default_contains_key_sections(self) -> None:
        """Bundled default references workspace root and key tools."""
        assert f"{WORKSPACE_ROOT}/" in DEFAULT_INSTRUCTIONS
        assert "sudo" in DEFAULT_INSTRUCTIONS
        assert "git" in DEFAULT_INSTRUCTIONS.lower()


class TestResolveInstructions:
    """Tests for resolve_instructions()."""

    @pytest.mark.parametrize(
        ("config", "provider", "expected"),
        [
            ({"instructions": "Do the thing."}, "claude", "Do the thing."),
            (
                {"instructions": {"claude": "Claude instructions", "_default": "Default"}},
                "claude",
                "Claude instructions",
            ),
            ({}, "claude", DEFAULT_INSTRUCTIONS),
            ({"instructions": None}, "claude", DEFAULT_INSTRUCTIONS),
            (
                {"instructions": ["_inherit", "Extra text."]},
                "claude",
                f"{DEFAULT_INSTRUCTIONS}\n\nExtra text.",
            ),
            ({"instructions": "_inherit"}, "claude", DEFAULT_INSTRUCTIONS),
        ],
        ids=[
            "flat-string",
            "provider-claude",
            "missing-key",
            "null",
            "list-inherit-prefix",
            "bare-inherit-string",
        ],
    )
    def test_instruction_resolution(
        self,
        config: dict[str, object],
        provider: str,
        expected: str,
    ) -> None:
        """Instructions resolve correctly for various config shapes."""
        assert resolve_instructions(config, provider) == expected


class TestFileAppend:
    """Tests for standalone instructions.md file append behavior."""

    def test_default_plus_file(self) -> None:
        """File content is appended to bundled default."""
        result = resolve_with_project_file({}, file_text="Project notes.")
        assert result == f"{DEFAULT_INSTRUCTIONS}\n\nProject notes."

    def test_no_file(self) -> None:
        """Without a file, only YAML-resolved text is returned."""
        assert resolve_with_project_file({}) == DEFAULT_INSTRUCTIONS

    def test_empty_file_ignored(self) -> None:
        """Whitespace-only file is ignored."""
        assert resolve_with_project_file({"instructions": "base"}, file_text="  \n  ") == "base"


class TestHasCustomInstructions:
    """Tests for has_custom_instructions()."""

    def test_has_from_config(self) -> None:
        """Config with instructions key returns True."""
        assert has_custom_instructions({"instructions": "Custom"})

    def test_absent_returns_false(self) -> None:
        """Empty config returns False."""
        assert not has_custom_instructions({})

    def test_true_with_instructions_file(self) -> None:
        """True when instructions.md exists under project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "instructions.md").write_text("Content", encoding="utf-8")
            assert has_custom_instructions({}, project_root=root)

    def test_false_without_file(self) -> None:
        """False when project root has no instructions.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not has_custom_instructions({}, project_root=Path(tmpdir))

    def test_false_nonexistent_root(self) -> None:
        """False when project root does not exist."""
        assert not has_custom_instructions({}, project_root=NONEXISTENT_PROJECT_ROOT)
