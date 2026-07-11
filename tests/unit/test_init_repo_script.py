# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for ``init-ssh-and-repo.sh``'s repo-sync section.

The script runs for real in a scratch ``HOME`` with ``file://`` fixture
repos and a no-op ``ensure-bridges.sh`` stub on ``PATH`` (``source``
resolves it via ``PATH`` lookup).  No podman involved — this exercises
exactly the git logic a task container runs at start.

The regression pinned here: terok-managed remotes must be re-asserted
from the launch env *before* the first fetch.  Gate URLs embed a
per-task auth token, so a workspace surviving a container recreate
holds a dead origin URL — historically only fixed up for ``file://``
gates, so the HTTP gate 403'd forever.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "src"
    / "terok_executor"
    / "resources"
    / "scripts"
    / "init-ssh-and-repo.sh"
)


def _git(*args: str | Path, cwd: Path | None = None) -> str:
    """Run a git command for fixture setup, returning stripped stdout."""
    result = subprocess.run(
        ["git", *map(str, args)],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_source_repo(path: Path) -> None:
    """Create a commit-bearing repo at *path* to serve as clone source."""
    _git("init", "-b", "main", path)
    (path / "file.txt").write_text("v1\n")
    _git("add", ".", cwd=path)
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-m",
        "v1",
        cwd=path,
    )


@pytest.fixture()
def script_env(tmp_path: Path) -> dict[str, str]:
    """Minimal env the script needs: scratch HOME + stubbed ensure-bridges.sh."""
    home = tmp_path / "home"
    home.mkdir()
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    (stub_bin / "ensure-bridges.sh").write_text("true\n")
    return {
        "HOME": str(home),
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
    }


def _run_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_restarted_task_reasserts_origin_before_fetch(
    tmp_path: Path, script_env: dict[str, str]
) -> None:
    """A stale origin (dead per-task gate URL) is replaced by CODE_REPO pre-fetch.

    The old origin points at a repo that no longer answers — like a gate
    URL embedding a revoked token.  Under ``set -e`` the script only
    survives if the re-assert happens before the fetch.
    """
    gate = tmp_path / "gate.git"
    _make_source_repo(gate)
    workspace = tmp_path / "workspace"
    _git("clone", gate.as_uri(), workspace)
    dead_url = (tmp_path / "gone.git").as_uri()
    _git("remote", "set-url", "origin", dead_url, cwd=workspace)

    result = _run_script({**script_env, "REPO_ROOT": str(workspace), "CODE_REPO": gate.as_uri()})

    assert result.returncode == 0, result.stderr
    assert ">> fixing origin remote" in result.stdout
    assert _git("remote", "get-url", "origin", cwd=workspace) == gate.as_uri()
    assert _git("remote", "get-url", "--push", "origin", cwd=workspace) == gate.as_uri()


def test_restarted_task_fetch_skips_secondary_remotes(
    tmp_path: Path, script_env: dict[str, str]
) -> None:
    """The init fetch targets origin only — external/gate remotes may be unreachable."""
    gate = tmp_path / "gate.git"
    _make_source_repo(gate)
    workspace = tmp_path / "workspace"
    _git("clone", gate.as_uri(), workspace)
    _git("remote", "add", "external", (tmp_path / "unreachable.git").as_uri(), cwd=workspace)

    result = _run_script({**script_env, "REPO_ROOT": str(workspace), "CODE_REPO": gate.as_uri()})

    assert result.returncode == 0, result.stderr


def test_new_task_cache_mismatch_still_wipes(tmp_path: Path, script_env: dict[str, str]) -> None:
    """The stale-cache wipe wins over the origin re-assert on new tasks.

    A cached workspace whose origin matches neither CODE_REPO nor
    CLONE_FROM is wrong content, not just a wrong URL — it must be wiped
    and re-cloned, not repointed and kept.
    """
    gate = tmp_path / "gate.git"
    _make_source_repo(gate)
    stale_source = tmp_path / "stale.git"
    _make_source_repo(stale_source)
    (stale_source / "file.txt").write_text("stale\n")
    _git("add", ".", cwd=stale_source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "stale", cwd=stale_source)
    workspace = tmp_path / "workspace"
    _git("clone", stale_source.as_uri(), workspace)
    (workspace / ".new-task-marker").touch()

    result = _run_script({**script_env, "REPO_ROOT": str(workspace), "CODE_REPO": gate.as_uri()})

    assert result.returncode == 0, result.stderr
    assert "wiping for fresh clone" in result.stdout
    assert _git("remote", "get-url", "origin", cwd=workspace) == gate.as_uri()
    assert (workspace / "file.txt").read_text() == "v1\n"
    assert not (workspace / ".new-task-marker").exists()
