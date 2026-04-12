# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for terok_executor.paths — namespace directory resolution."""

from __future__ import annotations

import os
import unittest.mock
from pathlib import Path

import pytest

from terok_executor import paths


class TestStateRoot:
    """Verify ``state_root()`` resolution via sandbox namespace resolver."""

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """TEROK_EXECUTOR_STATE_DIR takes first priority."""
        monkeypatch.setenv("TEROK_EXECUTOR_STATE_DIR", str(tmp_path))
        assert paths.state_root() == tmp_path

    def test_env_tilde_expansion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tilde in TEROK_EXECUTOR_STATE_DIR is expanded."""
        monkeypatch.setenv("TEROK_EXECUTOR_STATE_DIR", "~/agent-state")
        result = paths.state_root()
        assert "~" not in str(result)
        assert result == Path.home() / "agent-state"

    def test_root_fallback(self) -> None:
        """Root user gets /var/lib/terok/executor."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert paths.state_root() == Path("/var/lib/terok/executor")

    def test_default_ends_with_executor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default path ends with terok/executor."""
        monkeypatch.delenv("TEROK_EXECUTOR_STATE_DIR", raising=False)
        result = paths.state_root()
        assert result.name == "executor"
        assert result.parent.name == "terok"


class TestMountsDir:
    """Verify ``mounts_dir()`` lives under sandbox-live/."""

    def test_default_under_sandbox_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """mounts_dir() returns sandbox-live/mounts by default."""
        monkeypatch.delenv("TEROK_SANDBOX_LIVE_DIR", raising=False)
        result = paths.mounts_dir()
        assert result.name == "mounts"
        assert result.parent.name == "sandbox-live"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """TEROK_SANDBOX_LIVE_DIR overrides the sandbox-live root."""
        monkeypatch.setenv("TEROK_SANDBOX_LIVE_DIR", str(tmp_path))
        assert paths.mounts_dir() == tmp_path / "mounts"


class TestNamespaceConstants:
    """Verify namespace constants."""

    def test_subdir_is_executor(self) -> None:
        """_SUBDIR is 'executor'."""
        assert paths._SUBDIR == "executor"
