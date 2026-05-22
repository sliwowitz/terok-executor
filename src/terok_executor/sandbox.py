# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Compose terok-sandbox install with executor-side route generation.

Routes come from the YAML agent roster — sandbox (correctly) doesn't
know about rosters, so the pair that makes a runnable sandbox lives
here.  One entry point means every frontend that wants a functional
runtime reaches for the same composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terok_executor.integrations.sandbox import SandboxConfig


def ensure_sandbox_ready(
    *,
    cfg: SandboxConfig | None = None,
    no_vault: bool = False,
    **aggregator_kwargs: Any,
) -> None:
    """Generate vault routes, then run the sandbox install aggregator.

    Order matters: the aggregator's vault phase restarts the vault
    systemd unit, which reads ``routes.json`` at startup.  A bare
    aggregator call leaves the vault empty of routing config and
    credential fetch breaks on the next ``terok-executor run`` until
    the operator remembers to run ``vault routes``.

    ``no_vault`` gates the routes pre-step (if vault isn't being
    touched, don't regenerate); everything else (``root``, other
    ``no_*`` flags) flows through to the aggregator.

    Routes regeneration renders a ``Vault routes`` stage line so it
    sits in the same column as the aggregator's own output rather
    than failing silently above it — a corrupt YAML roster is the
    most plausible reason for setup to fail before the aggregator
    even starts, and a stage-shaped failure beats an unframed
    traceback.
    """
    from terok_executor.integrations.sandbox import _handle_sandbox_setup, stage_line
    from terok_executor.roster.loader import ensure_vault_routes

    if not no_vault:
        with stage_line("Vault routes") as s:
            ensure_vault_routes(cfg=cfg)
            s.ok("regenerated")
    _handle_sandbox_setup(cfg=cfg, no_vault=no_vault, **aggregator_kwargs)
