# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Exercise the deployed Vibe hook protocol and non-destructive installation."""

from __future__ import annotations

import io
import json
import runpy
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

_SCRIPT = resources.files("terok_executor") / "resources/scripts/terok-vibe-session.py"


@pytest.fixture
def hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Load the production script with an isolated Vibe home and task pointer."""
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
    monkeypatch.setenv("TEROK_SESSION_FILE", str(tmp_path / "session.txt"))
    return runpy.run_path(str(_SCRIPT))


def test_install_preserves_user_hooks_and_comments(hook: dict, tmp_path: Path) -> None:
    """Installation appends only the two missing hooks; a second run changes nothing."""
    home = tmp_path / "vibe"
    home.mkdir()
    path = home / "hooks.toml"
    original = '# keep this comment\n[[hooks]]\nname="user"\ntype="pre_tool"\ncommand="true"\n'
    path.write_text(original)
    hook["install"]()
    installed = path.read_text()
    assert installed.startswith(original)
    hooks = tomllib.loads(installed)["hooks"]
    assert [item["name"] for item in hooks] == [
        "user",
        "terok-session-pre_tool",
        "terok-session-post_agent",
    ]
    assert all(item["command"].endswith(" capture") for item in hooks[1:])
    hook["install"]()
    assert path.read_text() == installed


@pytest.mark.parametrize(
    "content",
    [
        "[broken",
        'hooks = "invalid"',
        "hooks = []",
        '[[hooks]]\nname="terok-session-pre_tool"\ntype="pre_tool"\ncommand="custom"\n',
    ],
)
def test_invalid_or_conflicting_hooks_are_not_overwritten(
    hook: dict, tmp_path: Path, content: str
) -> None:
    """An unreadable or conflicting config is an explicit failure, not a destructive repair."""
    home = tmp_path / "vibe"
    home.mkdir()
    path = home / "hooks.toml"
    path.write_text(content)
    with pytest.raises(ValueError):
        hook["install"]()
    assert path.read_text() == content


@pytest.mark.parametrize("kind", ["pre_tool", "post_agent"])
def test_capture_records_root_but_not_subagents(
    hook: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Both native hook types carry the root ID; child sessions leave it untouched."""
    pointer = tmp_path / "session.txt"
    for session_id, parent in (("root", None), ("child", "root")):
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                json.dumps(
                    {
                        "hook_event_name": kind,
                        "session_id": session_id,
                        "parent_session_id": parent,
                    }
                )
            ),
        )
        hook["capture"]()
        assert pointer.read_text() == "root\n"


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "{}",
        '{"session_id":"unverified"}',
        '{"parent_session_id":null,"session_id":42}',
        '{"parent_session_id":null,"session_id":""}',
        '{"parent_session_id":null,"session_id":"bad\\nid"}',
    ],
)
def test_incompatible_payload_keeps_pointer_unchanged(
    hook: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    """Missing context must not be treated as a root or trigger history discovery."""
    pointer = tmp_path / "session.txt"
    pointer.write_text("known\n")
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    with pytest.raises(ValueError):
        hook["capture"]()
    assert pointer.read_text() == "known\n"


def test_shared_hook_outside_wrapper_is_noop(hook: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-terok Vibe process sharing this home has no task pointer to mutate."""
    monkeypatch.delenv("TEROK_SESSION_FILE")
    monkeypatch.setattr(sys, "stdin", io.StringIO("not even read"))
    hook["capture"]()


def test_install_preserves_linked_hook_config(hook: dict, tmp_path: Path) -> None:
    """An operator's hooks.toml symlink still points to the updated config."""
    home = tmp_path / "vibe"
    home.mkdir()
    target = tmp_path / "linked-hooks.toml"
    target.write_text("# linked configuration\n")
    path = home / "hooks.toml"
    path.symlink_to(target)
    hook["install"]()
    assert path.is_symlink()
    assert len(tomllib.loads(target.read_text())["hooks"]) == 2
