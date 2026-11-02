#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
# terok:container - shared one-shot prompt protocol for wrappers and the banner.

"""Claim an immutable initial prompt or display it while it remains pending.

``claim`` emits the prompt once and creates its shared consumed marker.
``show`` displays a pending prompt without claiming it. Exit 1 means no
pending prompt; exit 2 reports an unsafe entry or an I/O failure.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

INPUTS = Path("/home/dev/.terok/input")
STATE = Path("/home/dev/.terok/events")
PROMPT = "initial-prompt.txt"
CONSUMED = "initial-prompt.consumed"
MAX_BYTES = 1024 * 1024


def pending_prompt(inputs: Path, state: Path, *, claim: bool) -> str | None:
    """Return a pending prompt, claiming it exclusively when requested."""
    try:
        directory = os.open(state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        try:
            info = os.stat(CONSUMED, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _require_marker(info)
            return None
        prompt = _read_prompt(inputs)
        if not prompt:
            return None
        if claim:
            try:
                fd = os.open(
                    CONSUMED,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                _require_marker(os.stat(CONSUMED, dir_fd=directory, follow_symlinks=False))
                return None
            os.close(fd)
        return prompt
    finally:
        os.close(directory)


def _require_marker(info: os.stat_result) -> None:
    """Accept only an empty, single-link regular consumption marker."""
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size:
        raise ValueError("initial-prompt.consumed must be an empty regular file, not a link")


def _read_prompt(inputs: Path) -> str | None:
    """Read the bounded immutable input without following a final symlink."""
    try:
        directory = os.open(inputs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        try:
            fd = os.open(PROMPT, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("initial-prompt.txt must be a single regular file")
            content = stream.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise ValueError("initial-prompt.txt exceeds the one-megabyte limit")
        return content.decode("utf-8")
    finally:
        os.close(directory)


def main() -> int:
    """Serve both wrappers and the login banner with the same prompt protocol."""
    if sys.argv[1:] not in (["claim"], ["show"]):
        print("Usage: terok-initial-prompt.py claim|show", file=sys.stderr)
        return 2
    try:
        prompt = pending_prompt(INPUTS, STATE, claim=sys.argv[1] == "claim")
    except (OSError, ValueError) as exc:
        print(f"terok: cannot read or claim initial prompt: {exc}", file=sys.stderr)
        return 2
    if prompt is None:
        return 1
    if sys.argv[1] == "show":
        print(f"📝 Initial prompt ({INPUTS / PROMPT})")
    print(prompt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
