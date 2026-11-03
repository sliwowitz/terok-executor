# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Follow-ups use the same live host-owned channel in shared and sealed tasks."""

from unittest.mock import MagicMock, patch

import pytest
from terok_util import UnsafePathError

from terok_executor.container.inject import inject_prompt
from terok_executor.integrations.sandbox import ContainerInspection, ImageInspection
from terok_executor.task_files import (
    CONTAINER_AGENT_INPUTS,
    INPUTS_PROTOCOL,
    INPUTS_PROTOCOL_LABEL,
    MAX_PROMPT_BYTES,
    TASK_INPUT_PROTOCOL,
)


@pytest.fixture
def sandbox(tmp_path):
    """Provide inspected task identity and mounts without launching a container."""
    sandbox = MagicMock()
    container = sandbox.runtime.container.return_value
    container.image.inspect.return_value = ImageInspection(
        id="image-id", labels={INPUTS_PROTOCOL_LABEL: INPUTS_PROTOCOL}
    )
    container.mounts = [(str(tmp_path), CONTAINER_AGENT_INPUTS)]
    container.inspect.return_value = ContainerInspection(
        id="container-id",
        state="exited",
        image=container.image,
        env={"TEROK_CONTAINER_PROTOCOL": TASK_INPUT_PROTOCOL},
        mounts=tuple(container.mounts),
    )
    with patch("terok_executor.container.inject.Sandbox", return_value=sandbox):
        yield sandbox


@pytest.mark.parametrize(
    "prompt", ["Fix the test", "Behebe den Fehler — Ünïcödé", "One\nTwo\nThree"]
)
def test_followup_is_published_verbatim(tmp_path, sandbox, prompt):
    """Atomic host publication replaces unsafe container-side copies."""
    inject_prompt("task", prompt)
    assert (tmp_path / "prompt.txt").read_text() == prompt
    sandbox.copy_to.assert_not_called()


@pytest.mark.parametrize("character", ["x", "é"])
def test_followup_accepts_exact_byte_limit(tmp_path, sandbox, character):
    """The maximum UTF-8 byte count works without counting code points instead."""
    prompt = character * (MAX_PROMPT_BYTES // len(character.encode("utf-8")))
    inject_prompt("task", prompt)
    assert (tmp_path / "prompt.txt").read_text() == prompt


@pytest.mark.parametrize("character", ["x", "é"])
def test_followup_size_is_validated_before_runtime_access(tmp_path, sandbox, character):
    """Reject oversized text before touching the container or its published inputs."""
    prompt = character * (MAX_PROMPT_BYTES // len(character.encode("utf-8")) + 1)
    with pytest.raises(ValueError, match="120 KiB"):
        inject_prompt("task", prompt)
    sandbox.runtime.container.assert_not_called()
    assert not (tmp_path / "prompt.txt").exists()


def test_missing_input_mount_requires_recreation(sandbox):
    """Legacy containers cannot fall back to writing a writable task path."""
    sandbox.runtime.container.return_value.mounts = []
    with pytest.raises(RuntimeError, match="recreate"):
        inject_prompt("task", "Instruction")
    sandbox.copy_to.assert_not_called()


def test_followup_rejects_unsafe_host_destination(tmp_path, sandbox):
    """Even a poisoned host input entry cannot redirect follow-up publication."""
    outside = tmp_path / "outside"
    outside.write_text("Unchanged")
    (tmp_path / "prompt.txt").symlink_to(outside)
    with pytest.raises(UnsafePathError):
        inject_prompt("task", "Instruction")
    assert outside.read_text() == "Unchanged"
