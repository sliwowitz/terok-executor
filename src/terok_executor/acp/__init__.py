# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-task host-side ACP (Agent Client Protocol) aggregator.

Bridges a single ACP client (Zed, Toad, …) to one of several in-container
agents (claude, codex, copilot, …) by namespacing models as
``agent:model`` (e.g. ``claude:opus-4.6``) under ACP's standard
``category: "model"`` configOption.

Bind-trigger surfaces: explicit ``session/set_model`` /
``session/set_config_option(configId="model")``, or — for clients that
trust the advertised ``currentModelId`` — lazily on the first
backend-needing method (e.g. ``session/prompt``).  Cross-agent switching
mid-session is out of scope for v1; subsequent picks against a different
agent are rejected at the protocol level.

Public surface: [`ACPRoster`][terok_executor.acp.roster.ACPRoster], [`AgentRosterCache`][terok_executor.acp.cache.AgentRosterCache],
[`list_authenticated_agents`][terok_executor.acp.roster.list_authenticated_agents], [`acp_socket_is_live`][terok_executor.acp.daemon.acp_socket_is_live],
plus the daemon entry point [`serve_acp`][terok_executor.acp.daemon.serve_acp].  Re-exported from
``terok_executor`` for the host-side caller (terok) so it doesn't
have to reach into ``terok_executor.acp.daemon`` directly.
"""

from __future__ import annotations

from .cache import AgentRosterCache, CacheKey
from .daemon import acp_socket_is_live, serve_acp
from .endpoint import ACPEndpointStatus
from .probe import ProbeError, probe_agent_models
from .proxy import AgentBindError
from .roster import ACPRoster, list_authenticated_agents

__all__ = [
    "ACPEndpointStatus",
    "ACPRoster",
    "AgentBindError",
    "AgentRosterCache",
    "CacheKey",
    "ProbeError",
    "acp_socket_is_live",
    "list_authenticated_agents",
    "probe_agent_models",
    "serve_acp",
]
