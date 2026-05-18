# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Discover the models an in-container ACP agent advertises.

Each in-container agent ships an ACP wrapper script (``terok-{agent}-acp``)
that exposes the agent over JSON-RPC on stdio.  To learn which models
the wrapper currently advertises, drive a minimal handshake:

1. ``initialize`` — version negotiation
2. ``session/new`` — receive the ``models`` block
3. close stdin — agent exits cleanly

The handshake is cheap but non-trivial to repeat; the result is cached
by [`AgentRosterCache`][terok_executor.acp.cache.AgentRosterCache] and
reused for the lifetime of the authenticated session.

The probe spawns the wrapper directly via the ACP SDK's
`acp.spawn_agent_process`.  Argv is supplied by
the caller (the roster), so the probe itself doesn't need to know about
``podman``, ``krun``, or the sandbox runtime.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, NoReturn

from acp import CLIENT_METHODS, PROTOCOL_VERSION, RequestError, spawn_agent_process
from acp.schema import ClientCapabilities

_logger = logging.getLogger(__name__)

DEFAULT_PROBE_TIMEOUT_SEC = 3.0
"""Per-call timeout for the full probe handshake.

Bounds first-call ``session/new`` latency: the daemon probes every
configured agent in parallel, so the slowest probe dominates.  Three
seconds is enough for a healthy wrapper to handshake and short enough
that a fully-unauthed image doesn't make the user wait ten seconds for
a model picker.  Override per call with the ``timeout`` parameter."""


async def probe_agent_models(
    *,
    agent_id: str,
    wrapper_argv: list[str],
    timeout: float = DEFAULT_PROBE_TIMEOUT_SEC,
    cwd: str = "/workspace",
) -> tuple[str, ...]:
    """Drive the minimal ACP handshake against ``terok-{agent_id}-acp``.

    Spawns the wrapper via `acp.spawn_agent_process`
    (which owns the asyncio stdio bridging and the graceful subprocess
    shutdown dance), sends ``initialize`` and ``session/new``, reads
    the ``models`` block, and returns the bare model ids.

    Raises [`ProbeError`][terok_executor.acp.probe.ProbeError] on
    timeout, transport failure, or any handshake error.  Callers (the
    roster cache) typically catch it, cache an empty roster, and skip
    the agent until the container is restarted.
    """
    command, *args = wrapper_argv
    try:
        async with asyncio.timeout(timeout):
            async with spawn_agent_process(_ProbeClient(), command, *args) as (client, _proc):
                await client.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                )
                resp = await client.new_session(cwd=cwd, mcp_servers=[])
    except TimeoutError as exc:
        _logger.warning("ACP probe for agent %r timed out after %.1fs", agent_id, timeout)
        raise ProbeError(f"probe timed out for agent {agent_id!r}") from exc
    except Exception as exc:
        raise ProbeError(f"probe failed for agent {agent_id!r}: {exc}") from exc

    if resp.models is None:
        return ()
    return tuple(m.model_id for m in resp.models.available_models)


class _ProbeClient:
    """Minimal `acp.Client` impl for the probe handshake.

    The probe never triggers backend → client traffic in a healthy
    wrapper: ``initialize`` + ``session/new`` complete before any tool
    call could ask for a permission or a file read.  Stray
    ``session_update`` notifications a chatty wrapper might emit are
    swallowed (they're informational); requests are fast-failed with
    ``method_not_found`` so a misbehaving wrapper doesn't hang the
    probe waiting for a response the proxy can't provide.
    """

    def on_connect(self, _conn: Any) -> None:
        """Required by the `acp.Client` protocol; nothing to do here."""

    async def session_update(self, *_: object, **__: object) -> None:
        """Swallow progress notifications a chatty wrapper might emit."""
        return None

    async def request_permission(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe can't relay permission prompts."""
        _not_supported_during_probe(CLIENT_METHODS["session_request_permission"])

    async def read_text_file(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe can't relay fs reads."""
        _not_supported_during_probe(CLIENT_METHODS["fs_read_text_file"])

    async def write_text_file(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe can't relay fs writes."""
        _not_supported_during_probe(CLIENT_METHODS["fs_write_text_file"])

    async def create_terminal(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe can't open terminals."""
        _not_supported_during_probe(CLIENT_METHODS["terminal_create"])

    async def terminal_output(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe owns no terminals."""
        _not_supported_during_probe(CLIENT_METHODS["terminal_output"])

    async def release_terminal(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe owns no terminals."""
        _not_supported_during_probe(CLIENT_METHODS["terminal_release"])

    async def wait_for_terminal_exit(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe owns no terminals."""
        _not_supported_during_probe(CLIENT_METHODS["terminal_wait_for_exit"])

    async def kill_terminal(self, *_: object, **__: object) -> NoReturn:
        """Fast-fail — probe owns no terminals."""
        _not_supported_during_probe(CLIENT_METHODS["terminal_kill"])

    async def ext_method(self, _name: str, _payload: dict[str, Any]) -> NoReturn:
        """Fast-fail — probe doesn't carry extension surfaces."""
        _not_supported_during_probe("ext")

    async def ext_notification(self, _name: str, _payload: dict[str, Any]) -> None:
        """Swallow extension notifications during the probe."""
        return None


def _not_supported_during_probe(method: str) -> NoReturn:
    """Fail-fast helper for [`_ProbeClient`][terok_executor.acp.probe._ProbeClient]."""
    raise RequestError.method_not_found(method)


class ProbeError(RuntimeError):
    """Raised when an agent fails to respond to the probe handshake.

    The cache stores empty rosters for failed probes (so we don't hammer
    a misconfigured agent on every session) — callers should treat
    ``ProbeError`` as "this agent is currently unusable" rather than
    bubble it to the user.
    """
