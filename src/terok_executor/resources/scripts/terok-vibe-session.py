# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
# terok:container — this file is deployed into task containers, not used on the host.

"""Capture Vibe's root session via native hooks, without relocating transcripts.

``install`` appends our hooks to Vibe's user hooks, preserving existing content.
``capture`` receives Vibe's JSON invocation on stdin and records only its root
session in the task-local ``TEROK_SESSION_FILE``. No shared-history discovery.
"""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import sys
import tempfile
import tomllib
from pathlib import Path

_HOOK_TYPES = ("pre_tool", "post_agent")


def install() -> None:
    """Add idempotent hooks under a lock, without replacing the user's hooks."""
    home = Path(os.environ.get("VIBE_HOME", str(Path.home() / ".vibe"))).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    path = (home / "hooks.toml").resolve()
    command = f"python3 {shlex.quote(str(Path(__file__).resolve()))} capture"
    with (home / "hooks.toml.terok.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = path.read_text() if path.exists() else ""
        hooks = tomllib.loads(original).get("hooks", [])
        if not isinstance(hooks, list) or not all(isinstance(hook, dict) for hook in hooks):
            raise ValueError("hooks must be a TOML array of tables")
        additions = []
        for kind in _HOOK_TYPES:
            expected = {"name": f"terok-session-{kind}", "type": kind, "command": command}
            existing = [hook for hook in hooks if hook.get("name") == expected["name"]]
            if existing:
                if existing != [expected]:
                    raise ValueError(f"conflicting hook: {expected['name']}")
                continue
            additions.append(
                "\n[[hooks]]\n"
                + "".join(f"{key} = {json.dumps(value)}\n" for key, value in expected.items())
            )
        if additions:
            updated = original + "\n" + "".join(additions)
            # Inline hook arrays cannot be extended with array-of-table syntax.
            # Reject rather than corrupting a valid but unsupported user layout.
            tomllib.loads(updated)
            _write_atomic(path, updated)


def capture() -> None:
    """Save the invoking task's root session; subagents never replace it."""
    target = os.environ.get("TEROK_SESSION_FILE")
    if not target:
        return  # Shared hooks may also run outside a terok wrapper.
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict) or "parent_session_id" not in payload:
        raise ValueError("Vibe hook payload lacks root-session context")
    if payload["parent_session_id"] is not None:
        return
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id or any(c.isspace() for c in session_id):
        raise ValueError("Vibe hook payload lacks a valid session ID")
    _write_atomic(Path(target), session_id + "\n")


def _write_atomic(path: Path, content: str) -> None:
    """Publish a complete file so simultaneous readers cannot see partial data."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        {"install": install, "capture": capture}[sys.argv[1]]()
    except (OSError, ValueError) as error:
        print(f"terok: Vibe session capture unavailable: {error}", file=sys.stderr)
        sys.exit(1)
