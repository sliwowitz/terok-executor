# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Separate host instructions from writable task state.

The input directory is mounted read-only, while agent configuration and
session pointers remain writable. Initial prompts are published once; the
task acknowledges delivery with a separate consumed marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from terok_util import (
    open_untrusted_directory,
    read_untrusted_text,
    untrusted_file_lock,
    write_untrusted_text,
)

if TYPE_CHECKING:
    from .integrations.sandbox import (
        Container,
        ContainerInspection,
        Image,
        ImageInspection,
        VolumeSpec,
    )

CONTAINER_AGENT_INPUTS = "/home/dev/.terok/input"
CONTAINER_AGENT_CONFIG = "/home/dev/.terok"
CONTAINER_AGENT_EVENTS = "/home/dev/.terok/events"
INITIAL_PROMPT_FILENAME = "initial-prompt.txt"
INITIAL_PROMPT_CONSUMED_FILENAME = "initial-prompt.consumed"
INPUTS_PROTOCOL_LABEL = "ai.terok.inputs-protocol"
INPUTS_PROTOCOL = "1"
TASK_INPUT_PROTOCOL = "4"
# Prompts travel as one argv element; leave headroom below Linux's 128 KiB cap.
MAX_PROMPT_BYTES = 120 * 1024


class InputProtocolMismatch(RuntimeError):
    """A successful runtime inspection proves that task inputs use an obsolete protocol."""


def agent_input_dir(task_dir: Path) -> Path:
    """Return the host-only input directory beside the task's writable configuration."""
    return task_dir / "agent-inputs"


def agent_event_dir(task_dir: Path) -> Path:
    """Return the narrow live task-acknowledgement directory, including sealed tasks."""
    return task_dir / "agent-events"


def require_task_input_protocol(
    container: Container, *, inspection: ContainerInspection | None = None
) -> None:
    """Refuse obsolete task inputs without treating inspection failures as upgrade requests.

    Reuse *inspection* when the caller has already observed the retained
    container. Only [`InputProtocolMismatch`][terok_executor.InputProtocolMismatch]
    establishes a reason to recreate; unavailable runtime data must abort
    without discarding retained state.
    """
    from .integrations.sandbox import RuntimeInspectionError

    observed = inspection if inspection is not None else container.inspect()
    if observed is None:
        raise RuntimeInspectionError(f"Cannot verify task {container.name!r}: container is absent")
    if observed.image is None:
        raise RuntimeInspectionError(
            f"Cannot verify task {container.name!r}: originating image is unavailable"
        )
    image = _inspect_protocol_image(observed.image)
    if (
        observed.env.get("TEROK_CONTAINER_PROTOCOL") != TASK_INPUT_PROTOCOL
        or image.labels.get(INPUTS_PROTOCOL_LABEL) != INPUTS_PROTOCOL
    ):
        raise InputProtocolMismatch(
            f"Task container {container.name!r} uses an obsolete input protocol. "
            "Rebuild the agent and project images, then recreate the task. "
            "Its shared vendor configuration and session files can be retained."
        )


def require_image_input_protocol(image: Image) -> None:
    """Reject obsolete images before an orchestrator removes a retained task."""
    if _inspect_protocol_image(image).labels.get(INPUTS_PROTOCOL_LABEL) != INPUTS_PROTOCOL:
        raise InputProtocolMismatch(
            f"Image {image.ref!r} lacks the read-only task-input protocol. "
            "Rebuild the agent and project images, then recreate the task."
        )


def agent_config_volumes(agent_config_dir: Path) -> tuple[VolumeSpec, ...]:
    """Bind task state read-write and host instructions read-only.

    The directory mount exposes prompts published after background startup.
    A file mount would keep the old inode after atomic publication.
    """
    from .integrations.sandbox import Sharing, VolumeSpec

    inputs = agent_input_dir(agent_config_dir.parent)
    events = agent_event_dir(agent_config_dir.parent)
    for directory in (agent_config_dir, inputs, events):
        with open_untrusted_directory(directory, create=True):
            # Acquisition itself prepares each mount root without following task-created links.
            pass
    return (
        VolumeSpec(agent_config_dir, CONTAINER_AGENT_CONFIG, sharing=Sharing.PRIVATE),
        VolumeSpec(
            inputs, CONTAINER_AGENT_INPUTS, sharing=Sharing.PRIVATE, read_only=True, live=True
        ),
        VolumeSpec(events, CONTAINER_AGENT_EVENTS, sharing=Sharing.PRIVATE, live=True),
    )


def write_initial_prompt(task_dir: Path, prompt: str) -> None:
    """Publish the initial prompt once, allowing an identical repeated submission.

    A different prompt requires a new task. No existing file is opened for
    truncation, including legacy task-created links.
    """
    content = prompt + "\n"
    data = content.encode("utf-8")
    if len(data) > MAX_PROMPT_BYTES:
        raise ValueError(f"Initial prompt exceeds the {MAX_PROMPT_BYTES // 1024} KiB limit")
    directory = agent_input_dir(task_dir)
    with untrusted_file_lock(directory / ".initial-prompt.lock"):
        target = directory / INITIAL_PROMPT_FILENAME
        try:
            existing = read_untrusted_text(target, max_bytes=MAX_PROMPT_BYTES)
        except FileNotFoundError:
            write_untrusted_text(target, content)
            return
        if existing != content:
            raise ValueError("Initial prompt is immutable; create a new task")


def _inspect_protocol_image(image: Image) -> ImageInspection:
    """Require a successful image observation before deciding protocol compatibility."""
    from .integrations.sandbox import RuntimeInspectionError

    inspected = image.inspect()
    if inspected is None:
        raise RuntimeInspectionError(f"Cannot verify image {image.ref!r}: image is absent")
    return inspected


__all__ = [
    "CONTAINER_AGENT_INPUTS",
    "INITIAL_PROMPT_CONSUMED_FILENAME",
    "INITIAL_PROMPT_FILENAME",
    "MAX_PROMPT_BYTES",
    "InputProtocolMismatch",
    "agent_config_volumes",
    "agent_input_dir",
    "agent_event_dir",
    "require_task_input_protocol",
    "require_image_input_protocol",
    "write_initial_prompt",
]
