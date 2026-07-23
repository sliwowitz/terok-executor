# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Fixtures and skip guards for the podman-backed integration suite.

These tests exist for the claims a mock cannot falsify: the file modes a
real kernel and a real podman uid mapping produce, the env a real
``podman run`` actually delivers, and the run/exec/stop lifecycle across
the podman version spread (4.3 → 5.8) the matrix covers.  Everything that
can be asserted without a container already lives in ``tests/unit``, and
nothing here mirrors it.

Environment requirements are expressed as markers, not directory layout:

- ``needs_podman``: podman on the host, plus shield's OCI hooks — every
  container executor launches goes through
  [`Sandbox.run`][terok_sandbox.Sandbox], which refuses to start a
  container it cannot shield.

**Isolation.**  ``HOME`` is deliberately *not* redirected (unlike the unit
suite): podman's rootless image store lives under the real ``HOME``, and a
tmp one would hide the image the session pulled.  Isolation is achieved the
other way round — every path executor would resolve from the environment is
passed in explicitly (``SandboxConfig(state_dir=…, vault_dir=…)``,
``ContainerEnvSpec(envs_dir=…)``), all rooted in ``tmp_path``.  The single
patched symbol is the ``SandboxConfig`` *class* seen by the env assembler,
which constructs its own config from nothing; the object it hands back is a
real config over real tmp directories, not a mock.
"""

from __future__ import annotations

import shutil
import socket
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from terok_util.matrix import binary_on_path, check_capability_contract

from terok_executor.integrations.sandbox import RunSpec, Sandbox, SandboxConfig, VolumeSpec
from terok_executor.roster import AgentRoster
from tests.constants import (
    CONTAINER_KEEPALIVE_COMMAND,
    CONTAINER_WORKSPACE_DIR,
    INTEGRATION_VAULT_PASSPHRASE,
    PODMAN_BASE_IMAGE,
    PODMAN_PULL_TIMEOUT,
)

from .helpers import podman, podman_rm, unique_container_name

_INTERNET_PROBE_HOST = ("one.one.one.one", 443)
"""Address the internet probe dials — the base-image pull needs egress."""


# ── Host capability probes ────────────────────────────────────────────


def _has(binary: str) -> bool:
    """Return whether *binary* is on ``PATH``."""
    return shutil.which(binary) is not None


def _hooks_available() -> bool:
    """Return whether shield's global OCI hooks are installed.

    Imported lazily and defensively: shield is a transitive dependency, and
    a host without it should skip rather than error at collection.
    """
    try:
        from terok_shield.podman_info import has_global_hooks
    except ImportError:  # pragma: no cover — shield always ships with sandbox
        return False
    try:
        return bool(has_global_hooks())
    except Exception:  # pragma: no cover — defensive; a probe never fails a run
        return False


def _internet_reachable() -> bool:
    """Return whether the host can open an outbound TCP connection."""
    try:
        with socket.create_connection(_INTERNET_PROBE_HOST, timeout=5):
            return True
    except OSError:
        return False


# ── Skip guards (dev machines) ────────────────────────────────────────
# On a developer laptop a missing binary is a host limitation, so the
# suite skips.  Inside the matrix the image was built to a contract, and
# the same absence is a failure — see the contract check below.

podman_missing = pytest.mark.skipif(not _has("podman"), reason="podman not installed")
hooks_missing = pytest.mark.skipif(
    not _hooks_available(), reason="shield OCI hooks not installed (run `terok-shield setup`)"
)


# ── Matrix capability contract ────────────────────────────────────────


_CAPABILITY_PROBES = {
    "podman": lambda: _has("podman"),
    "nft": lambda: binary_on_path("nft"),
    "dnsmasq": lambda: binary_on_path("dnsmasq"),
    "hooks": _hooks_available,
    "internet": _internet_reachable,
}


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail the session up front when the matrix capability contract is broken.

    Without this, a slot whose image lost ``nft`` would dissolve into a page
    of skips and read as green.
    """
    if broken := check_capability_contract(_CAPABILITY_PROBES):
        pytest.exit(broken, returncode=3)


# ── The container base image ──────────────────────────────────────────


@pytest.fixture(scope="session")
def podman_image() -> str:
    """Ensure the base image is in the local store; return its reference.

    Pulled at most once per session.  Container launches afterwards resolve
    it locally, so no test is at the mercy of a registry.
    """
    if not _has("podman"):
        pytest.skip("podman not installed")
    if podman("image", "exists", PODMAN_BASE_IMAGE).returncode == 0:
        return PODMAN_BASE_IMAGE
    proc = podman("pull", PODMAN_BASE_IMAGE, timeout=PODMAN_PULL_TIMEOUT)
    if proc.returncode != 0:
        pytest.skip(f"cannot pull {PODMAN_BASE_IMAGE}: {proc.stderr.strip()}")
    return PODMAN_BASE_IMAGE


# ── Isolated executor environment ─────────────────────────────────────


@dataclass(frozen=True)
class ExecutorEnv:
    """The tmp-rooted state an executor launch reads and writes."""

    cfg: SandboxConfig
    """Real sandbox config — real vault DB, real ports, tmp directories."""

    mounts_dir: Path
    """Base for the shared credential mounts (``ContainerEnvSpec.envs_dir``)."""

    workspace_dir: Path
    """Host side of the container's ``/workspace`` — the sandbox always runs
    with that as the working directory, so every launch must mount it."""

    task_dir: Path
    """Per-task state dir; shield writes its dossier here on ``pre_start``."""

    def sandbox(self) -> Sandbox:
        """Return a Sandbox bound to this config — the launcher executor uses."""
        return Sandbox(config=self.cfg)


@pytest.fixture
def executor_env(tmp_path: Path) -> Iterator[ExecutorEnv]:
    """Yield a fully tmp-rooted [`ExecutorEnv`][tests.integration.conftest.ExecutorEnv].

    The vault DB is a real SQLCipher file opened with a throwaway passphrase
    (the operator's own passphrase chain is never consulted, and their vault
    is never opened).  ``SandboxConfig`` is patched only where the env
    assembler constructs one implicitly — see the module docstring.
    """
    env = ExecutorEnv(
        cfg=SandboxConfig(
            state_dir=tmp_path / "state",
            vault_dir=tmp_path / "vault",
            config_dir=tmp_path / "config",
            credentials_passphrase_command=f"echo {INTEGRATION_VAULT_PASSPHRASE}",
        ),
        mounts_dir=tmp_path / "mounts",
        task_dir=tmp_path / "task",
        workspace_dir=tmp_path / "workspace",
    )
    env.mounts_dir.mkdir(parents=True, exist_ok=True)
    env.task_dir.mkdir(parents=True, exist_ok=True)
    env.workspace_dir.mkdir(parents=True, exist_ok=True)
    env.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

    with patch("terok_executor.integrations.sandbox.SandboxConfig", return_value=env.cfg):
        yield env


@pytest.fixture
def roster() -> AgentRoster:
    """Return the live agent roster loaded from the bundled YAML."""
    return AgentRoster.shared()


# ── Launching containers ──────────────────────────────────────────────


type Launcher = Callable[..., str]
"""``launch(slug, env=…, volumes=…) -> container_name``."""


@pytest.fixture
def launch(executor_env: ExecutorEnv, podman_image: str) -> Iterator[Launcher]:
    """Yield a launcher that starts containers exactly as executor does.

    Launches go through [`Sandbox.run`][terok_sandbox.Sandbox] — the same
    ``podman run`` assembly ``AgentRunner.launch_prepared`` performs, shield
    args and all — with the keepalive command in place of an agent, so what
    the matrix exercises is executor's launch path rather than a hand-rolled
    podman command line.

    Every container is named before it is created and force-removed in the
    ``finally``, so an assertion failure (or a podman that half-created it)
    still leaves the host clean.
    """
    started: list[str] = []

    def _launch(
        slug: str,
        *,
        env: dict[str, str] | None = None,
        volumes: tuple[VolumeSpec, ...] = (),
    ) -> str:
        name = unique_container_name(slug)
        started.append(name)
        # The workspace mount is not optional: the sandbox runs every
        # container with ``-w /workspace`` (as production does, where the
        # task's workspace lives there), so podman refuses to start a
        # container that lacks it.
        workspace = VolumeSpec(
            host_path=executor_env.workspace_dir,
            container_path=CONTAINER_WORKSPACE_DIR,
        )
        executor_env.sandbox().run(
            RunSpec(
                container_name=name,
                image=podman_image,
                env=dict(env or {}),
                volumes=(workspace, *volumes),
                command=CONTAINER_KEEPALIVE_COMMAND,
                task_dir=executor_env.task_dir,
            )
        )
        return name

    try:
        yield _launch
    finally:
        for name in started:
            podman_rm(name)
