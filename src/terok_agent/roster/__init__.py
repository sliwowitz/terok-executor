# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Agent catalog — YAML-driven agent and tool roster with layered config.

Delegates to :mod:`.loader` for YAML deserialization and roster construction,
and to :mod:`.config_stack` for generic layered config resolution.
"""

from .loader import (
    AgentRoster,
    CredentialProxyRoute,
    MountDef,
    SidecarSpec,
    ensure_proxy_routes,
    get_roster,
    load_roster,
)

__all__ = [
    "AgentRoster",
    "CredentialProxyRoute",
    "MountDef",
    "SidecarSpec",
    "ensure_proxy_routes",
    "get_roster",
    "load_roster",
]
