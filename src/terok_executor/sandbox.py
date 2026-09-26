# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Own executor setup and compose sandbox provisioning below the agent roster."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from typing import Any

from terok_util import SetupCheck, SetupReceipt, SetupStatus, require_no_downgrade, require_setup

from terok_executor.integrations.sandbox import SandboxConfig, check_sandbox_setup
from terok_executor.paths import state_root
from terok_executor.roster import AgentRoster

_OWNER = "terok-executor"
_RECEIPT_NAME = "setup.json"


def check_setup(cfg: SandboxConfig | None = None, *, live: bool = False) -> tuple[SetupCheck, ...]:
    """Check owned routes and compose public sandbox checks without changing state."""
    cfg = cfg or SandboxConfig()
    try:
        expected = AgentRoster.load().generate_routes_json()
        routes = _check_routes(cfg, expected)
    except (OSError, ValueError) as exc:
        expected = ""
        routes = SetupCheck(_OWNER, "roster", SetupStatus.INVALID, str(exc))
    return (
        _receipt(cfg, expected).check(),
        routes,
        *check_sandbox_setup(cfg, live=live),
    )


def ensure_sandbox_ready(
    *,
    cfg: SandboxConfig | None = None,
    no_vault: bool = False,
    **aggregator_kwargs: Any,
) -> None:
    """Preflight the full closure, generate routes, then provision sandbox services.

    Routes must be current before the supervisor starts. Skipping route generation
    does not certify absent routes or incomplete lower setup. Successful child
    receipts survive failure in executor's final verification or receipt write.
    """
    from terok_executor.integrations.sandbox import _handle_sandbox_setup, stage_line

    cfg = cfg or SandboxConfig()
    require_no_downgrade(check_setup(cfg))
    roster = AgentRoster.load()
    expected = roster.generate_routes_json()
    receipt = _receipt(cfg, expected)
    if not no_vault:
        receipt.clear()
        with stage_line("Vault routes") as stage:
            roster.ensure_vault_routes(cfg=cfg)
            stage.ok("regenerated")
    _handle_sandbox_setup(cfg=cfg, no_vault=no_vault, **aggregator_kwargs)
    require_setup((_check_routes(cfg, expected), *check_sandbox_setup(cfg)))
    receipt.write()


def _receipt(cfg: SandboxConfig, expected: str | None = None) -> SetupReceipt:
    """Describe only executor's version, route location, and current roster projection."""
    if expected is None:
        expected = AgentRoster.load().generate_routes_json()
    return SetupReceipt(
        state_root() / _RECEIPT_NAME,
        _OWNER,
        version(_OWNER),
        {
            "routes_path": str(cfg.routes_path),
            "routes_hash": hashlib.sha256(expected.encode()).hexdigest(),
        },
    )


def _check_routes(cfg: SandboxConfig, expected: str) -> SetupCheck:
    """Verify the actual route artifact rather than trusting its receipt alone."""
    try:
        actual = cfg.routes_path.read_text(encoding="utf-8").rstrip("\n")
    except FileNotFoundError:
        return SetupCheck(_OWNER, "routes", SetupStatus.MISSING, "Vault routes are missing")
    except (OSError, UnicodeError):
        return SetupCheck(_OWNER, "routes", SetupStatus.INVALID, "Vault routes cannot be read")
    if actual != expected:
        return SetupCheck(
            _OWNER, "routes", SetupStatus.STALE, "Vault routes differ from the roster"
        )
    return SetupCheck(_OWNER, "routes", SetupStatus.READY)
