# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Executor can run a container, exec into it, and stop and remove it.

The assertions are deliberately unremarkable — a container that starts,
answers an exec, stops on request, and leaves nothing behind.  Their value
is not in what they claim but in *where they run*: the matrix replays this
across the podman version spread (4.3 → 5.8) and the distro spread, which
is where the launch path actually breaks — a userns flag podman 4.x spells
differently, a stop that leaves the container in ``stopping``, a runtime
default that changed under us.
"""

from __future__ import annotations

import pytest

from .conftest import ExecutorEnv, Launcher, hooks_missing, podman_missing
from .helpers import container_exists, container_state, exec_in

pytestmark = [pytest.mark.needs_podman, podman_missing, hooks_missing]

MARKER = "executor-lifecycle-ok"
"""Payload echoed from inside the container — proof the exec really ran there."""


def test_run_exec_stop_removes_every_trace(
    executor_env: ExecutorEnv,
    launch: Launcher,
) -> None:
    """A launched container is exec-able, stoppable, and removable."""
    sandbox = executor_env.sandbox()

    name = launch("lifecycle")
    assert container_state(name) == "running"

    assert exec_in(name, "echo", MARKER).strip() == MARKER

    sandbox.stop([name])
    assert container_state(name) == "exited"

    sandbox.rm([name])
    assert not container_exists(name)
