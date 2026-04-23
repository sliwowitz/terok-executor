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
    from terok_sandbox import SandboxConfig


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
    """
    from terok_sandbox.commands import _handle_sandbox_setup

    from terok_executor.roster.loader import ensure_vault_routes

    if not no_vault:
        ensure_vault_routes(cfg=cfg)
    _handle_sandbox_setup(cfg=cfg, no_vault=no_vault, **aggregator_kwargs)
