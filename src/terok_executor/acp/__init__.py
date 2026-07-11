# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-task host-side ACP (Agent Client Protocol) aggregator.

Bridges a single ACP client (Zed, Toad, …) to one of several
in-container agents (claude, codex, copilot, …) by namespacing models
as ``agent:model`` (e.g. ``claude:opus-4.6``) under ACP's standard
``category: "model"`` configOption.

Module map:

- [`daemon`][terok_executor.acp.daemon] — Unix-socket server, container
  lifecycle supervision, and the standalone ``terok-executor acp``
  entry point.  Owns [`serve_acp`][terok_executor.acp.daemon.serve_acp]
  and the [`acp_socket_is_live`][terok_executor.acp.daemon.acp_socket_is_live]
  probe used to distinguish live daemons from stale socket files.
- [`roster`][terok_executor.acp.roster] — per-task aggregation: walks
  the image's ``ai.terok.agents`` label, probes each agent, and answers
  "what models does this container offer?"  Owns
  [`ACPRoster`][terok_executor.acp.roster.ACPRoster] and the
  vault-side [`list_authenticated_agents`][terok_executor.acp.roster.list_authenticated_agents].
- [`proxy`][terok_executor.acp.proxy] — the typed bidirectional ACP
  mediator: implements both `acp.Agent` (toward the connected
  client) and `acp.Client` (toward the bound backend wrapper)
  on one object.  Drives the bind handshake on first model pick.
- [`probe`][terok_executor.acp.probe] — the minimal ``initialize +
  session/new`` handshake that extracts an agent's model roster.
- [`cache`][terok_executor.acp.cache] — thread-safe per-agent model
  cache; survives reconnects, invalidated on credential rotation.
- [`endpoint`][terok_executor.acp.endpoint] — the
  [`ACPEndpointStatus`][terok_executor.acp.endpoint.ACPEndpointStatus]
  enum the host CLI uses to classify endpoints in ``terok acp list``.
- [`model_options`][terok_executor.acp.model_options] — the
  ``agent:model`` namespace vocabulary and the typed builders +
  rewriter that keep the proxy's frames schema-valid.

Bind-trigger surfaces: explicit ``session/set_model`` /
``session/set_config_option(configId="model")``, or — for clients
that trust the advertised ``currentModelId`` — lazily on the first
backend-needing method (e.g. ``session/prompt``).  Cross-agent
switching mid-session is out of scope for v1; subsequent picks against
a different agent are rejected at the protocol level.

The exports below are re-exported from ``terok_executor`` so the
host-side caller (terok) doesn't have to reach into the submodules.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .daemon import acp_socket_is_live as acp_socket_is_live
    from .endpoint import ACPEndpointStatus as ACPEndpointStatus
    from .roster import list_authenticated_agents as list_authenticated_agents

#: Public symbol → defining submodule.  Only the three names the host
#: CLI re-exports through ``terok_executor`` live here; the protocol
#: machinery ([`ACPRoster`][terok_executor.acp.roster.ACPRoster],
#: [`ACPProxy`][terok_executor.acp.proxy.ACPProxy], the probe/cache
#: types) is internal — import it from its submodule when you need it.
#: Resolving lazily keeps [`acp_socket_is_live`][terok_executor.acp.daemon.acp_socket_is_live]
#: (and this package's import) free of the ``acp`` pydantic schema build,
#: which only [`.probe`][terok_executor.acp.probe] / [`.proxy`][terok_executor.acp.proxy]
#: pull in when a session is actually served.
_LAZY: dict[str, str] = {
    "ACPEndpointStatus": ".endpoint",
    "acp_socket_is_live": ".daemon",
    "list_authenticated_agents": ".roster",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> object:
    """Resolve a re-exported name to its defining submodule on first access (PEP 562)."""
    try:
        module_path = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the lazy names to ``dir()`` / autocompletion."""
    return sorted({*globals(), *_LAZY})
