# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container supervisor sidecar JSON writer.

The terok-sandbox OCI hook (installed by ``terok-sandbox setup``)
spawns one supervisor process per container at start.  The hook is
triggered by — and reads from — the ``terok.sandbox.sidecar`` OCI
annotation; the annotation's value is the absolute path to the JSON
written here.

Schema mirrors the sandbox writer (`terok_sandbox.launch._write_sidecar`):
keys ``container_name``, ``ipc_mode`` (``"socket"`` or ``"tcp"``),
``db_path``, ``scope_id``, ``project_id``, ``task_id``, ``runtime_dir``,
plus ``tcp_port`` / ``ssh_signer_port`` in TCP mode and an optional
``dossier_path``.  When the git gate is wired the payload also carries
``gate_base_path`` / ``gate_token`` (and ``gate_port`` in TCP mode) so
the per-container supervisor can serve the gate in-process.  Socket
paths are deliberately absent — in socket mode the supervisor derives
them from the container name and runtime dir, so only the
freshly-allocated TCP ports need carrying.
The caller (``AgentRunner.launch_prepared``) emits the returned
path as the OCI annotation so the hook can find this file.

Path: ``<cfg.state_dir>/sidecar/<container-name>.json``.  The
single ``sidecar/`` segment is the canonical location — no XDG
guessing, no nested ``terok/`` infix — and matches what the
``terok-sandbox`` writer also emits.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from terok_executor.integrations.sandbox import PerContainerResources, SandboxConfig

_logger = logging.getLogger(__name__)


def write_supervisor_sidecar(
    container_name: str,
    *,
    cfg: SandboxConfig,
    per_container: PerContainerResources,
    scope_id: str = "",
    project_id: str = "",
    task_id: str = "",
    dossier_path: Path | str | None = None,
    gate_base_path: str | None = None,
    gate_token: str | None = None,
    gate_port: int | None = None,
) -> Path | None:
    """Persist the per-container supervisor sidecar JSON.

    Best-effort: a write failure is logged to stderr and returns
    ``None`` — the supervisor will refuse to spawn (no sidecar = no
    terok-managed container) but the launch itself isn't blocked.
    Mirrors the soft-fail policy of
    `terok_sandbox.launch._write_sidecar`.

    Args:
        container_name: The ``--name`` passed to ``podman run`` (and
            therefore the sidecar filename key).
        cfg: Sandbox config — sources ``state_dir``, ``services_mode``,
            ``db_path``, plus the active transport's socket path or
            broker port.
        scope_id: Credential scope; empty for non-scoped runs.
        project_id: Terok project ID; empty when not under a project.
        task_id: Terok task ID; empty for standalone executor runs.
        dossier_path: Optional path to the per-task dossier file the
            shield reads; ``None`` for executor runs that don't carry
            a dossier.
        gate_base_path: Absolute path to the dir holding the per-project
            bare mirrors the supervisor's gate serves; ``None`` when the
            gate is not wired for this container.
        gate_token: Per-container gate access token the supervisor
            validates in-process; ``None`` when the gate is not wired.
        gate_port: TCP port the gate listens on (TCP mode only);
            ``None`` in socket mode or when the gate is not wired.

    Returns:
        The written sidecar path, or ``None`` if the write failed.

    Raises:
        ValueError: If ``container_name`` is empty or carries a path
            separator / traversal segment — such a name would let the
            sidecar write escape ``state_dir/sidecar``.
    """
    # ``container_name`` becomes a path segment below, so a value like
    # ``../../.ssh/config`` would let a caller write outside the sidecar
    # dir *before* podman ever rejects the invalid ``--name``.  Reject
    # any name carrying a path separator or traversal segment up front —
    # mirrors the ``--name`` guard terok-sandbox applies on its own
    # sidecar writer.
    if (
        not container_name
        or container_name in (".", "..")
        or "/" in container_name
        or "\\" in container_name
    ):
        raise ValueError(f"invalid container name: {container_name!r}")

    sidecar_dir = cfg.state_dir / "sidecar"
    try:
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"warning: sidecar dir setup failed: {exc}", file=sys.stderr)
        return None

    payload: dict[str, object] = {
        "container_name": container_name,
        "ipc_mode": cfg.services_mode,
        "db_path": str(cfg.db_path),
        "scope_id": scope_id or "",
        "project_id": project_id or "",
        "task_id": task_id or "",
        # The supervisor runs in crun's rootless userns where geteuid==0;
        # ``namespace_runtime_dir`` would misroute to ``/run/terok``.
        "runtime_dir": str(cfg.runtime_dir),
    }
    if cfg.services_mode == "tcp":
        payload["tcp_port"] = per_container.token_broker_port
        payload["ssh_signer_port"] = per_container.ssh_signer_port
    if dossier_path is not None:
        payload["dossier_path"] = str(dossier_path)
    # Gate config travels here only when the gate is wired; the
    # supervisor composes the gate iff both gate_base_path and
    # gate_token are present.
    if gate_base_path is not None:
        payload["gate_base_path"] = gate_base_path
    if gate_token is not None:
        payload["gate_token"] = gate_token
    if gate_port is not None:
        payload["gate_port"] = gate_port

    target = sidecar_dir / f"{container_name}.json"
    # The payload can carry a live gate_token, so the file must not be
    # world-readable.  Create it 0600 directly (the process umask would
    # otherwise leave it 0644) and fchmod to also cover the re-launch case
    # where the file already exists with looser permissions.
    #
    # ``O_NOFOLLOW`` refuses to open the final path component if it is a
    # symlink — a pre-planted symlink at ``target`` (e.g. pointing at
    # ``~/.ssh/authorized_keys``) would otherwise let this write clobber
    # an arbitrary file with the token payload.  The container_name
    # traversal guard above blocks one vector; this closes the symlink
    # vector at the open(2) level.  The ``ELOOP`` it raises is an OSError,
    # so it falls into the existing soft-fail branch and returns ``None``.
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), 0o600)
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"warning: sidecar write failed: {exc}", file=sys.stderr)
        return None

    _logger.debug("Wrote supervisor sidecar for %s → %s", container_name, target)
    return target
