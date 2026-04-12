# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Resolves filesystem paths for executor state and bind-mount directories.

Delegates to :func:`terok_sandbox.paths.namespace_state_dir` for the
shared XDG/FHS resolution logic — no vendored copy of the platform
detection code.
"""

from pathlib import Path

from terok_sandbox.paths import namespace_state_dir

_SUBDIR = "executor"


def state_root() -> Path:
    """Writable state root for executor-owned data.

    Priority: ``TEROK_EXECUTOR_STATE_DIR`` → ``/var/lib/terok/executor`` (root)
    → ``platformdirs`` → ``$XDG_DATA_HOME/terok/executor``
    → ``~/.local/share/terok/executor``.
    """
    return namespace_state_dir(_SUBDIR, "TEROK_EXECUTOR_STATE_DIR")


def mounts_dir() -> Path:
    """Base directory for agent config bind-mounts (container-writable).

    Lives under the ``sandbox-live/`` tree alongside task workspaces,
    grouping all container-writable content for security-aware
    partitioning (``noexec,nosuid,nodev``).

    Each agent/tool gets a subdirectory (e.g. ``_claude-config/``) that is
    bind-mounted read-write into task containers.  These directories are
    intentionally separated from the credentials store since they are
    container-exposed and subject to potential poisoning.
    """
    return namespace_state_dir("sandbox-live", "TEROK_SANDBOX_LIVE_DIR") / "mounts"
