# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Publish follow-ups through the task's host-owned input directory."""

from __future__ import annotations

from pathlib import Path

from terok_util import write_untrusted_text

from terok_executor.integrations.sandbox import Sandbox
from terok_executor.task_files import (
    CONTAINER_AGENT_INPUTS,
    MAX_PROMPT_BYTES,
    require_task_input_protocol,
)


def inject_prompt(container_name: str, prompt_text: str) -> None:
    """Publish a follow-up through the task's read-only input mount.

    Sealed tasks use this same narrow live input channel. Legacy containers
    must be recreated instead of copying through a task-controlled pathname.
    """
    if len(prompt_text.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_BYTES // 1024} KiB limit")
    container = Sandbox().runtime.container(container_name)
    require_task_input_protocol(container)
    for source, destination in container.mounts:
        if destination == CONTAINER_AGENT_INPUTS:
            write_untrusted_text(Path(source) / "prompt.txt", prompt_text)
            return
    raise RuntimeError("Task has no read-only input mount; rebuild images and recreate the task")
