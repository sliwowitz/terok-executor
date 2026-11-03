# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Clone-cache workspace seeding for faster task startup.

When a gate mirror has been synced (``terok gate-sync`` or
``AgentRunner._setup_gate``), a non-bare clone cache may exist on the
host.  Copying that cache into the task workspace before container
launch replaces the slow in-container ``git clone`` with a fast file
copy followed by a lightweight ``git fetch + reset``.

The public entry point is [`seed_workspace_from_clone_cache`][terok_executor.container.cache.seed_workspace_from_clone_cache].
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from terok_util import UnsafePathError, open_untrusted_directory, require_host_tool

if TYPE_CHECKING:
    from terok_executor.integrations.sandbox import SandboxConfig

_logger = logging.getLogger(__name__)


def seed_workspace_from_clone_cache(
    workspace_path: Path,
    scope: str,
    *,
    origin_url: str | None = None,
    cfg: SandboxConfig | None = None,
) -> bool:
    """Pre-populate *workspace_path* from the clone cache for *scope*.

    Returns ``True`` if the workspace was successfully seeded.

    After copying, rewrites the git origin remote to *origin_url* so that
    the in-container init script's sanity check (which compares origin
    against ``CODE_REPO``) passes — the cache's origin points to a local
    ``file://`` URL that won't match.

    Skips seeding when the cache doesn't exist, the workspace already
    contains a ``.git`` directory, or the copy fails.  Failures are
    logged and swallowed — the container falls back to a full clone.
    """
    if (workspace_path / ".git").is_dir():
        return False

    cache_dir = _resolve_cache_dir(scope, cfg)
    if cache_dir is None or not (cache_dir / ".git").is_dir():
        return False

    try:
        _logger.info("Seeding workspace from clone cache: %s → %s", cache_dir, workspace_path)
        _copy_tree(cache_dir, workspace_path)
    except UnsafePathError:
        raise
    except (OSError, shutil.Error, subprocess.CalledProcessError) as exc:
        _logger.warning("Clone cache seed failed (non-fatal): %s", exc)
        _wipe_workspace_contents(workspace_path)
        return False

    if not (workspace_path / ".git").is_dir():
        _logger.warning("Cache copy did not produce .git; falling back to container clone")
        _wipe_workspace_contents(workspace_path)
        return False

    if origin_url:
        _rewrite_origin(workspace_path, origin_url)

    return True


# ── Private helpers ──────────────────────────────────────────────────────


def _resolve_cache_dir(scope: str, cfg: SandboxConfig | None) -> Path | None:
    """Derive the clone cache directory from config and scope."""
    if cfg is None:
        from terok_executor.integrations.sandbox import SandboxConfig as _SC

        cfg = _SC()
    cache_dir = cfg.clone_cache_base_path / scope
    return cache_dir if cache_dir.is_dir() else None


def _copy_tree(src: Path, dst: Path) -> None:
    """Stage the cache privately, then publish entries without traversing destination links."""
    with open_untrusted_directory(dst, create=True) as destination:
        if set(os.listdir(destination)) - {".new-task-marker"}:
            raise UnsafePathError(f"Refusing to seed a nonempty task workspace: {dst}")
        with tempfile.TemporaryDirectory(prefix=".terok-seed-", dir=dst.parent) as temporary:
            staged = Path(temporary)
            try:
                subprocess.run(
                    [require_host_tool("cp"), "--reflink=auto", "-a", f"{src}/.", str(staged)],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
            except FileNotFoundError:
                shutil.copytree(src, staged, symlinks=True, dirs_exist_ok=True)
            with open_untrusted_directory(staged) as source:
                for name in os.listdir(source):
                    os.rename(name, name, src_dir_fd=source, dst_dir_fd=destination)


def _rewrite_origin(workspace_path: Path, url: str) -> None:
    """Rewrite the git origin remote to *url* (best-effort)."""
    try:
        subprocess.run(
            [
                require_host_tool("git"),
                "-C",
                str(workspace_path),
                "remote",
                "set-url",
                "origin",
                url,
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        _logger.warning("Failed to rewrite origin after cache seed: %s", exc)


def _wipe_workspace_contents(workspace_path: Path) -> None:
    """Remove all contents of *workspace_path* so the fallback clone finds it empty.

    The init script requires an empty workspace for ``git clone``.
    """
    with open_untrusted_directory(workspace_path) as directory:
        for name in os.listdir(directory):
            try:
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(name, dir_fd=directory)
                else:
                    os.unlink(name, dir_fd=directory)
            except OSError:
                pass
