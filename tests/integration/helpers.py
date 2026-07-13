# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Podman helpers shared by the integration suite.

Every function here shells out to the real ``podman`` binary.  Two rules
hold throughout, and callers must not break them:

- **Scoped mutation.**  The only operator state a test may touch is one
  container named ``terok-executor-itest-<slug>-<token>``, created and
  removed by the test itself.  Nothing else on the host is written.
- **No registry traffic at launch time.**  The base image is pulled once
  per session by the ``podman_image`` fixture; container launches find it
  in the local store and never reach out.
"""

from __future__ import annotations

import subprocess  # nosec B404 — driving the real podman CLI is the point
import uuid

from tests.constants import PODMAN_COMMAND_TIMEOUT, PODMAN_CONTAINER_PREFIX


def unique_container_name(slug: str) -> str:
    """Return a collision-free container name for the test called *slug*."""
    return f"{PODMAN_CONTAINER_PREFIX}-{slug}-{uuid.uuid4().hex[:8]}"


def podman(*args: str, timeout: int = PODMAN_COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Run ``podman *args`` and return the completed process (never raises on exit code)."""
    return subprocess.run(  # nosec B603 B607 — fixed binary, test-controlled args
        ["podman", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def podman_checked(*args: str, timeout: int = PODMAN_COMMAND_TIMEOUT) -> str:
    """Run ``podman *args``, assert it succeeded, and return its stdout.

    Failures surface podman's own stderr — a bare ``CalledProcessError`` in a
    matrix log tells you nothing about *which* podman disagreed.
    """
    proc = podman(*args, timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(f"podman {' '.join(args)} failed ({proc.returncode}): {proc.stderr}")
    return proc.stdout


def podman_rm(name: str) -> None:
    """Force-remove container *name*, tolerating its absence.

    Called from the ``finally`` of every launching test, so it must never
    raise: a cleanup failure would mask the real assertion error.
    """
    try:
        podman("rm", "-f", "-t", "0", name)
    except (subprocess.TimeoutExpired, OSError):  # pragma: no cover — cleanup fallback
        pass


def container_exists(name: str) -> bool:
    """Return whether podman still knows a container called *name*."""
    return podman("container", "exists", name).returncode == 0


def container_state(name: str) -> str:
    """Return the podman lifecycle state of *name* (``running``, ``exited``, …)."""
    return podman_checked("inspect", "-f", "{{.State.Status}}", name).strip()


def exec_in(name: str, *command: str) -> str:
    """Exec *command* inside running container *name* and return its stdout."""
    return podman_checked("exec", name, *command)


def file_mode_in(name: str, container_path: str) -> int:
    """Return the permission bits of *container_path* as seen from inside *name*.

    Reading the mode through the container — rather than off the host — is
    the whole point: it is the view the agent's tooling gets after podman's
    uid mapping and mount handling have had their say.
    """
    return int(exec_in(name, "stat", "-c", "%a", container_path).strip(), 8)
