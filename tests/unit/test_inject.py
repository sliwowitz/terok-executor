# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Follow-ups use the same live host-owned channel in shared and sealed tasks."""

from unittest.mock import MagicMock, patch

import pytest
from terok_util import UnsafePathError

from terok_executor.container.inject import inject_prompt
from terok_executor.task_files import (
    CONTAINER_AGENT_INPUTS,
    INPUTS_PROTOCOL,
    INPUTS_PROTOCOL_LABEL,
    TASK_INPUT_PROTOCOL,
)


@pytest.fixture
def sandbox(tmp_path):
    """Provide inspected task identity and mounts without launching a container."""
    sandbox = MagicMock()
    container = sandbox.runtime.container.return_value
    container.env = {"TEROK_CONTAINER_PROTOCOL": TASK_INPUT_PROTOCOL}
    container.image.labels.return_value = {INPUTS_PROTOCOL_LABEL: INPUTS_PROTOCOL}
    container.mounts = [(str(tmp_path), CONTAINER_AGENT_INPUTS)]
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
