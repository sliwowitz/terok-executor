# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Every credential file executor places must reach the agent as 0600.

The claim under test is a *filesystem* claim, and only a real launch can
falsify it.  Two failure modes have bitten this code:

- a ``touch()`` without an explicit mode inherits the process umask, so on
  a permissive umask the file lands 0644 — glab then refuses to start,
  because it aborts on a ``config.yml`` looser than owner-only;
- files written by an *older* executor persist in the shared mounts across
  upgrades, so assembly must re-clamp what it finds, not just what it
  creates.  The glab mount is writable (its config doubles as a settings
  file), which means the clamp is the *only* thing standing between a
  legacy 0644 file and the agent.

A unit test can assert what ``os.stat`` says on the host.  It cannot say
what the file looks like from inside the container, after podman's uid
mapping and mount handling — which is the view glab actually gets.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from terok_executor.container.env import ContainerEnvSpec, assemble_container_env
from terok_executor.roster import AgentRoster
from tests.constants import CREDENTIAL_FILE_MODE, LOOSE_FILE_MODE, PERMISSIVE_UMASK

from .conftest import ExecutorEnv, Launcher, hooks_missing, podman_missing
from .helpers import file_mode_in

pytestmark = [pytest.mark.needs_podman, podman_missing, hooks_missing]


def _credential_mounts(roster: AgentRoster) -> list:
    """Return every roster mount that carries a credential file."""
    return [m for m in roster.mounts if m.credential_file]


def _seed_legacy_loose_files(roster: AgentRoster, mounts_dir: Path) -> list[Path]:
    """Pre-create every credential file world-readable, as a pre-0.3 launch left it.

    This is the clamp's input.  It also puts a file where the writable
    mounts (glab) would otherwise have none — assembly never creates those,
    so without seeding, the one mount the 0644 bug was reported against
    would not be under test at all.
    """
    seeded = []
    for mount in _credential_mounts(roster):
        host_file = mounts_dir / mount.host_dir / mount.credential_file
        host_file.parent.mkdir(parents=True, exist_ok=True)
        host_file.touch()
        host_file.chmod(LOOSE_FILE_MODE)
        seeded.append(host_file)
    return seeded


def test_credential_files_are_owner_only_in_the_container(
    executor_env: ExecutorEnv,
    roster: AgentRoster,
    launch: Launcher,
) -> None:
    """Assembled credential files land 0600 on the host and inside the container."""
    seeded = _seed_legacy_loose_files(roster, executor_env.mounts_dir)
    assert seeded, "roster declares no credential files — the test would assert nothing"

    spec = ContainerEnvSpec(
        task_id="cred-modes",
        agent_name="claude",
        envs_dir=executor_env.mounts_dir,
    )

    # Assemble under an all-permissive umask: a mode that survives *this*
    # came from executor, not from the environment it happened to run in.
    previous_umask = os.umask(PERMISSIVE_UMASK)
    try:
        result = assemble_container_env(spec, roster)
    finally:
        os.umask(previous_umask)

    for host_file in seeded:
        mode = host_file.stat().st_mode & 0o777
        assert mode == CREDENTIAL_FILE_MODE, f"{host_file} is {mode:o} on the host"

    name = launch("cred-modes", env=result.env, volumes=result.volumes)

    for mount in _credential_mounts(roster):
        container_file = f"{mount.container_path}/{mount.credential_file}"
        mode = file_mode_in(name, container_file)
        assert mode == CREDENTIAL_FILE_MODE, (
            f"{mount.provider}: {container_file} is {mode:o} inside the container"
        )
