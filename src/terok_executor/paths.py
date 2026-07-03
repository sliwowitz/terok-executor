# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Resolves filesystem paths for executor state and bind-mount directories.

Delegates to [`terok_util.namespace_state_dir`][terok_util.paths.namespace_state_dir]
for the shared XDG/FHS resolution logic — no vendored copy of the
platform detection code.
"""

from pathlib import Path

from terok_util import namespace_state_dir

_SUBDIR = "executor"


def state_root() -> Path:
    """Writable state root for executor-owned data.

    Priority: ``TEROK_EXECUTOR_STATE_DIR`` → ``/var/lib/terok/executor`` (root)
    → ``platformdirs`` → ``$XDG_DATA_HOME/terok/executor``
    → ``~/.local/share/terok/executor``.
    """
    return namespace_state_dir(_SUBDIR, env_var="TEROK_EXECUTOR_STATE_DIR")


def container_state_root() -> Path:
    """Parent of every per-container state directory (``state_root()/run``).

    Listing it names every container a ``run`` created on this host —
    including ``--name`` overrides the default-name prefix can't find.
    """
    return state_root() / "run"


def container_state_dir(container_name: str) -> Path:
    """Host-side state directory for one container, derived from its name.

    ``state_root()/run/<name>`` carries what a run keeps on the host even
    when the workspace lives in-container: shield state and the staged
    agent config.  ``run`` creates it, ``rm`` removes it.  Deriving the
    path from the container name keeps podman the only container
    registry — no executor-side index is needed to find a container's
    state.

    Raises:
        ValueError: If *container_name* is empty or carries a path
            separator / traversal segment — such a name would redirect
            the directory outside executor-owned state.
    """
    if (
        not container_name
        or container_name in (".", "..")
        or "/" in container_name
        or "\\" in container_name
    ):
        raise ValueError(f"invalid container name: {container_name!r}")
    return container_state_root() / container_name


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
    return namespace_state_dir("sandbox-live", env_var="TEROK_SANDBOX_LIVE_DIR") / "mounts"
