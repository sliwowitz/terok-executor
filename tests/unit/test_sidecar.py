# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container supervisor sidecar JSON contract.

The terok-sandbox OCI hook reads
``$XDG_STATE_HOME/terok/sidecar/<container-name>.json`` on container
start and spawns one supervisor process per container.  This module
verifies that
[`write_supervisor_sidecar`][terok_executor.container.sidecar.write_supervisor_sidecar]
emits exactly the keys the supervisor's loader expects.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from terok_sandbox import (
    PerContainerResources,
    SandboxConfig,
    allocate_per_container_resources,
)

from terok_executor.container.sidecar import write_supervisor_sidecar


def _cfg(tmp_path: Path) -> SandboxConfig:
    """A SandboxConfig rooted at *tmp_path* so the sidecar lands in a controllable spot."""
    return SandboxConfig(state_dir=tmp_path, vault_dir=tmp_path / "credentials")


def _socket_per_container(cfg: SandboxConfig, name: str) -> PerContainerResources:
    """Allocate per-container resources for *name* (socket mode)."""
    return allocate_per_container_resources(cfg, name)


class TestWriteSupervisorSidecar:
    """Sidecar JSON contract for the per-container supervisor."""

    def test_writes_socket_mode_payload(self, tmp_path: Path) -> None:
        """Socket-mode sidecar — vault/ssh paths are derived by the supervisor."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-001")
        target = write_supervisor_sidecar("agent-task-001", cfg=cfg, per_container=per_container)

        assert target is not None
        assert target == cfg.state_dir / "sidecar" / "agent-task-001.json"
        payload = json.loads(target.read_text())
        assert payload["container_name"] == "agent-task-001"
        assert payload["ipc_mode"] in ("socket", "tcp")
        assert payload["db_path"] == str(cfg.db_path)
        assert payload["scope_id"] == ""
        assert payload["project_id"] == ""
        assert payload["task_id"] == ""

    def test_propagates_scope_and_task_ids(self, tmp_path: Path) -> None:
        """Caller-provided scope / project / task IDs land in the payload."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-002")
        target = write_supervisor_sidecar(
            "agent-task-002",
            cfg=cfg,
            per_container=per_container,
            scope_id="myproj",
            project_id="proj-abc",
            task_id="task-42",
        )

        assert target is not None
        payload = json.loads(target.read_text())
        assert payload["scope_id"] == "myproj"
        assert payload["project_id"] == "proj-abc"
        assert payload["task_id"] == "task-42"

    def test_dossier_path_optional(self, tmp_path: Path) -> None:
        """``dossier_path`` is only emitted when the caller supplies one."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-003")

        target = write_supervisor_sidecar("agent-task-003", cfg=cfg, per_container=per_container)
        assert target is not None
        assert "dossier_path" not in json.loads(target.read_text())

        dossier = tmp_path / "dossier.toml"
        dossier.write_text("")
        target = write_supervisor_sidecar(
            "agent-task-004",
            cfg=cfg,
            per_container=_socket_per_container(cfg, "agent-task-004"),
            dossier_path=dossier,
        )
        assert target is not None
        assert json.loads(target.read_text())["dossier_path"] == str(dossier)

    def test_socket_mode_omits_socket_path(self, tmp_path: Path) -> None:
        """Socket mode: the supervisor derives sockets — they're NOT in the sidecar."""
        cfg = SandboxConfig(
            state_dir=tmp_path,
            vault_dir=tmp_path / "credentials",
            services_mode="socket",
        )
        per_container = _socket_per_container(cfg, "agent-task-socket")
        target = write_supervisor_sidecar("agent-task-socket", cfg=cfg, per_container=per_container)

        assert target is not None
        payload = json.loads(target.read_text())
        assert payload["ipc_mode"] == "socket"
        assert "socket_path" not in payload
        assert "ssh_signer_socket" not in payload
        assert "tcp_port" not in payload

    def test_tcp_mode_emits_per_container_ports(self, tmp_path: Path) -> None:
        """TCP mode: ``tcp_port`` + ``ssh_signer_port`` from the per-container
        allocation (NOT from cfg's singleton)."""
        cfg = SandboxConfig(
            state_dir=tmp_path,
            vault_dir=tmp_path / "credentials",
            services_mode="tcp",
        )
        per_container = allocate_per_container_resources(cfg, "agent-task-tcp")
        target = write_supervisor_sidecar("agent-task-tcp", cfg=cfg, per_container=per_container)

        assert target is not None
        payload = json.loads(target.read_text())
        assert payload["ipc_mode"] == "tcp"
        assert payload["tcp_port"] == per_container.token_broker_port
        assert payload["ssh_signer_port"] == per_container.ssh_signer_port
        assert isinstance(payload["tcp_port"], int)
        assert "socket_path" not in payload

    def test_gate_fields_omitted_when_not_given(self, tmp_path: Path) -> None:
        """No gate kwargs → the supervisor sees no gate config and won't serve one."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-nogate")
        target = write_supervisor_sidecar("agent-task-nogate", cfg=cfg, per_container=per_container)

        assert target is not None
        payload = json.loads(target.read_text())
        assert "gate_base_path" not in payload
        assert "gate_token" not in payload
        assert "gate_port" not in payload

    def test_gate_fields_emitted_when_given(self, tmp_path: Path) -> None:
        """Gate kwargs land in the payload so the supervisor serves the gate."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-gate")
        target = write_supervisor_sidecar(
            "agent-task-gate",
            cfg=cfg,
            per_container=per_container,
            gate_base_path="/var/lib/terok/gate",
            gate_token="terok-g-deadbeef",
            gate_port=9418,
        )

        assert target is not None
        payload = json.loads(target.read_text())
        assert payload["gate_base_path"] == "/var/lib/terok/gate"
        assert payload["gate_token"] == "terok-g-deadbeef"
        assert payload["gate_port"] == 9418

    def test_sidecar_file_is_0600(self, tmp_path: Path) -> None:
        """The sidecar can carry a live gate token, so it must be 0600 — never
        world/group readable regardless of the process umask."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-perm")
        target = write_supervisor_sidecar(
            "agent-task-perm",
            cfg=cfg,
            per_container=per_container,
            gate_token="terok-g-secret",
        )

        assert target is not None
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"

    def test_existing_loose_file_is_tightened_to_0600(self, tmp_path: Path) -> None:
        """A re-launch overwriting a pre-existing 0644 sidecar still ends 0600.

        ``os.open`` only applies its mode on *creation*; the explicit
        ``fchmod`` is what covers the re-launch case where the file
        already exists with looser permissions.
        """
        cfg = _cfg(tmp_path)
        sidecar_dir = cfg.state_dir / "sidecar"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        stale = sidecar_dir / "agent-task-relaunch.json"
        stale.write_text("{}")
        stale.chmod(0o644)

        target = write_supervisor_sidecar(
            "agent-task-relaunch",
            cfg=cfg,
            per_container=_socket_per_container(cfg, "agent-task-relaunch"),
        )

        assert target == stale
        assert stat.S_IMODE(stale.stat().st_mode) == 0o600

    def test_mkdir_failure_soft_fails_to_none(self, tmp_path: Path) -> None:
        """An OSError creating the sidecar dir returns ``None`` (best-effort) —
        the launch path turns that into a fail-closed refusal, but the writer
        itself must not raise."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-mkdirfail")

        with patch(
            "pathlib.Path.mkdir",
            side_effect=OSError("read-only filesystem"),
        ):
            target = write_supervisor_sidecar(
                "agent-task-mkdirfail", cfg=cfg, per_container=per_container
            )

        assert target is None

    @pytest.mark.parametrize(
        "bad_name",
        ["", ".", "..", "../../.ssh/config", "a/b", "sub/agent", "evil\\name"],
    )
    def test_unsafe_container_name_rejected(self, tmp_path: Path, bad_name: str) -> None:
        """A name with a path separator / traversal segment is rejected before
        any path is built — otherwise it would be an arbitrary-file-overwrite
        primitive (the write lands outside ``state_dir/sidecar``)."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-safe")

        with pytest.raises(ValueError, match="invalid container name"):
            write_supervisor_sidecar(bad_name, cfg=cfg, per_container=per_container)

    def test_write_failure_soft_fails_to_none(self, tmp_path: Path) -> None:
        """An OSError opening/writing the file returns ``None`` rather than raising."""
        cfg = _cfg(tmp_path)
        per_container = _socket_per_container(cfg, "agent-task-writefail")

        with patch(
            "terok_executor.container.sidecar.os.open",
            side_effect=OSError("disk full"),
        ):
            target = write_supervisor_sidecar(
                "agent-task-writefail", cfg=cfg, per_container=per_container
            )

        assert target is None
        # The dir was still created — only the file write failed.
        assert (cfg.state_dir / "sidecar").is_dir()

    def test_symlink_target_soft_fails_to_none(self, tmp_path: Path) -> None:
        """A pre-planted symlink at the sidecar target is refused (O_NOFOLLOW) and
        soft-fails to ``None`` instead of clobbering the symlink destination.

        Without ``O_NOFOLLOW`` an attacker who can plant a symlink at
        ``<state_dir>/sidecar/<name>.json`` could redirect the token-bearing
        write to an arbitrary file (CWE-22).  The open raises ``ELOOP`` (an
        OSError), which the existing soft-fail branch turns into ``None``."""
        cfg = _cfg(tmp_path)
        name = "agent-task-symlink"
        per_container = _socket_per_container(cfg, name)

        # Plant the symlink at the exact target the writer will use.
        sidecar_dir = cfg.state_dir / "sidecar"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        victim = tmp_path / "victim.txt"
        victim.write_text("untouched")
        (sidecar_dir / f"{name}.json").symlink_to(victim)

        target = write_supervisor_sidecar(name, cfg=cfg, per_container=per_container)

        assert target is None
        # The symlink destination was NOT written through.
        assert victim.read_text() == "untouched"
