# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""ACP proxy — one connection's worth of typed JSON-RPC mediation.

[`ACPProxy`][terok_executor.acp.proxy.ACPProxy] is the bridge behind
[`ACPRoster.attach`][terok_executor.acp.roster.ACPRoster.attach].  It
implements **both** sides of the ACP protocol on the same object:

- `acp.Agent` — facing the connected ACP client (Zed, Toad, …).
  An `acp.agent.connection.AgentSideConnection`
  reads the client's frames, deserialises them into typed pydantic
  models, and dispatches to ``self.initialize`` / ``self.new_session``
  / ``self.prompt`` / etc.
- `acp.Client` — facing the bound in-container backend wrapper.
  Once a model has been picked, a
  `acp.client.connection.ClientSideConnection`
  to ``terok-{agent}-acp`` reads the wrapper's frames and dispatches
  backend → client traffic (``session/update``, ``request_permission``,
  ``fs/*``, ``terminal/*``) onto the same proxy object so it can forward
  to the connected client.

Two phases drive the lifecycle:

- **Pre-bind**: ``initialize`` and ``session/new`` answer locally,
  advertising the aggregated ``agent:model`` list in
  `acp.schema.SessionModelState` plus a mirroring
  ``configOptions[category=model]``.  No backend process exists yet.
- **Bound**: on the first model-picking client request — modern ACP's
  ``session/set_model`` or older Zed's ``session/set_config_option(category=model)``,
  or lazily on the first backend-needing method like ``session/prompt``
  — the proxy spawns the in-container wrapper through
  `acp.spawn_agent_process`, replays
  ``initialize`` + ``session/new`` + ``session/set_model`` against it,
  and from then on forwards typed calls in both directions.  Backend
  responses and notifications carrying model ids are re-namespaced on
  the way out so the client always sees ``agent:model`` ids.

V1 takes shortcuts where the design is still settling: one session per
connection (Zed reconnects on every chat — fix on the roadmap), one
bound agent per session (no cross-agent switches without reconnect),
no push notifications when the authed-agent set changes mid-connection.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process
from acp.agent.connection import AgentSideConnection
from acp.client.connection import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    ClientCapabilities,
    ConfigOptionUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EnvVariable,
    ForkSessionResponse,
    HttpMcpServer,
    Implementation,
    InitializeResponse,
    KillTerminalResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResumeSessionResponse,
    SessionInfoUpdate,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TerminalOutputResponse,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from .model_options import (
    MODEL_OPTION_CATEGORY,
    build_aggregated_session_new,
    build_model_option,
    namespace_model_options_in_place,
    split_namespaced,
)

if TYPE_CHECKING:
    from .roster import ACPRoster

_logger = logging.getLogger(__name__)

CLIENT_SESSION_ID = "proxy-1"
"""Synthetic session id the proxy advertises to the connected client.

Backend session ids never reach the client — every backend-originated
frame has its ``sessionId`` rewritten to this constant on the way out.
"""

CONTAINER_WORKSPACE = "/workspace"
"""Path the backend ``session/new`` runs in.

ACP clients send their host filesystem path in ``new_session.cwd``
(Zed: ``/var/home/user/prog/X``) which doesn't exist inside the
container.  ``claude-agent-acp`` chdirs into ``cwd`` before exec; an
ENOENT there surfaces as the famously misleading "Claude Code native
binary not found …".  Pinning to the container's workspace mount is a
stopgap until the host↔sandbox path strategy lands.
"""

PROXY_AGENT_NAME = "terok-acp"
PROXY_AGENT_TITLE = "Terok ACP host-proxy"
PROXY_AGENT_VERSION = "1"

SessionUpdatePayload = (
    UserMessageChunk
    | AgentMessageChunk
    | AgentThoughtChunk
    | ToolCallStart
    | ToolCallProgress
    | AgentPlanUpdate
    | AvailableCommandsUpdate
    | CurrentModeUpdate
    | ConfigOptionUpdate
    | SessionInfoUpdate
    | UsageUpdate
)


class ACPProxy:
    """One client connection's worth of proxy state.

    Constructed by [`attach`][terok_executor.acp.roster.ACPRoster.attach];
    lives for the duration of a single client connection.  Not reusable
    — discard after [`run`][terok_executor.acp.proxy.ACPProxy.run] returns.
    """

    def __init__(self, *, roster: ACPRoster) -> None:
        self._roster = roster
        self._client: AgentSideConnection | None = None
        self._backend: ClientSideConnection | None = None
        self._backend_stack = AsyncExitStack()
        self._bind_lock = asyncio.Lock()
        self._bound_agent: str | None = None
        self._backend_session_id: str | None = None
        self._client_mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None
        self._client_additional_directories: list[str] | None = None
        self._client_capabilities: ClientCapabilities | None = None
        self._aggregated_models: list[str] = []
        # Namespaced ``agent:model`` advertised as ``currentModelId`` in
        # ``session/new``.  Lazy-bind target for clients that go straight
        # from ``session/new`` to ``session/prompt`` without an explicit
        # model selection — they trust that value and we follow through.
        self._default_namespaced: str | None = None
        self._session_announced = False

    async def run(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Run the typed proxy loop until the client disconnects.

        Hands the client side to
        `acp.agent.connection.AgentSideConnection`
        which dispatches typed methods on this object.  Always tears the
        bound backend down on exit, even on cancellation.
        """
        self._client = AgentSideConnection(self, writer, reader, listening=False)
        try:
            await self._client.listen()
        finally:
            await self._teardown_backend()

    def on_connect(self, _conn: Any) -> None:
        """Required by the `acp.Agent` / `acp.Client` protocols."""

    # ── Agent protocol (connected client → proxy) ─────────────────────

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **_kw: Any,
    ) -> InitializeResponse:
        """Answer ``initialize`` locally and capture the client's caps.

        ``client_capabilities`` is replayed verbatim on the backend's
        ``initialize`` during bind — whatever the client said it can do,
        the backend believes.  ``client_info`` is accepted to satisfy
        the protocol but discarded (the proxy doesn't relay it).
        """
        del protocol_version, client_info
        self._client_capabilities = client_capabilities or ClientCapabilities()
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_info=Implementation(
                name=PROXY_AGENT_NAME, title=PROXY_AGENT_TITLE, version=PROXY_AGENT_VERSION
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_kw: Any,
    ) -> NewSessionResponse:
        """Answer ``session/new`` with the aggregated model list.

        Synthesises [`CLIENT_SESSION_ID`][terok_executor.acp.proxy.CLIENT_SESSION_ID]
        so the client can proceed to model selection before any backend
        exists.  The real backend session id is captured on bind and
        translated on every forwarded frame.
        """
        del cwd
        if self._session_announced:
            raise RequestError.invalid_request(
                {"details": "proxy supports one session per connection (v1)"}
            )
        self._session_announced = True
        self._client_mcp_servers = mcp_servers
        self._client_additional_directories = additional_directories
        self._aggregated_models = await self._roster.list_available_agents()
        self._default_namespaced = self._aggregated_models[0] if self._aggregated_models else None
        return build_aggregated_session_new(CLIENT_SESSION_ID, self._aggregated_models)

    async def set_session_model(
        self, model_id: str, session_id: str, **_kw: Any
    ) -> SetSessionModelResponse | None:
        """Bind on first call; same-agent re-pick forwards through."""
        self._require_client_session(session_id)
        await self._select_model(model_id)
        return SetSessionModelResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **_kw: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Bind on ``category=model``; otherwise forward to the bound backend.

        Older ACP clients (Zed v1.0.x at the time of writing) pick the
        model through ``session/set_config_option(category="model")``;
        modern clients use the dedicated ``session/set_model``.  Accept
        both.  Non-model categories pass through to the bound backend;
        a non-model option pre-bind is rejected.
        """
        self._require_client_session(session_id)
        if config_id == MODEL_OPTION_CATEGORY and isinstance(value, str):
            await self._select_model(value)
            opt = build_model_option(self._aggregated_models, current=value)
            return SetSessionConfigOptionResponse(config_options=[opt])
        backend, backend_session = self._require_bound()
        resp = await backend.set_config_option(
            config_id=config_id, session_id=backend_session, value=value
        )
        if resp is not None and self._bound_agent is not None:
            namespace_model_options_in_place(resp.config_options, self._bound_agent)
        return resp

    async def prompt(
        self,
        prompt: list,
        session_id: str,
        message_id: str | None = None,
        **_kw: Any,
    ) -> PromptResponse:
        """Lazy-bind to the default model if needed, then forward."""
        self._require_client_session(session_id)
        await self._ensure_bound_for_default()
        backend, backend_session = self._require_bound()
        return await backend.prompt(
            prompt=prompt, session_id=backend_session, message_id=message_id
        )

    async def cancel(self, session_id: str, **_kw: Any) -> None:
        """Fire-and-forget cancel to the backend (no-op if not bound)."""
        del session_id
        if self._backend is not None and self._backend_session_id is not None:
            await self._backend.cancel(session_id=self._backend_session_id)

    async def authenticate(self, method_id: str, **_kw: Any) -> AuthenticateResponse | None:
        """Forward to bound backend; pre-bind authenticate is rejected."""
        backend, _ = self._require_bound()
        return await backend.authenticate(method_id=method_id)

    async def set_session_mode(
        self, mode_id: str, session_id: str, **_kw: Any
    ) -> SetSessionModeResponse | None:
        """Forward to bound backend."""
        self._require_client_session(session_id)
        backend, backend_session = self._require_bound()
        return await backend.set_session_mode(mode_id=mode_id, session_id=backend_session)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_kw: Any,
    ) -> LoadSessionResponse | None:
        """Forward to bound backend (v1 advertises no session-load capability)."""
        self._require_client_session(session_id)
        backend, backend_session = self._require_bound()
        return await backend.load_session(
            cwd=cwd,
            session_id=backend_session,
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
        )

    async def list_sessions(
        self,
        additional_directories: list[str] | None = None,
        cursor: str | None = None,
        cwd: str | None = None,
        **_kw: Any,
    ) -> ListSessionsResponse:
        """Forward to bound backend."""
        backend, _ = self._require_bound()
        return await backend.list_sessions(
            additional_directories=additional_directories, cursor=cursor, cwd=cwd
        )

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_kw: Any,
    ) -> ForkSessionResponse:
        """Forward to bound backend."""
        self._require_client_session(session_id)
        backend, backend_session = self._require_bound()
        return await backend.fork_session(
            cwd=cwd,
            session_id=backend_session,
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_kw: Any,
    ) -> ResumeSessionResponse:
        """Forward to bound backend."""
        self._require_client_session(session_id)
        backend, backend_session = self._require_bound()
        return await backend.resume_session(
            cwd=cwd,
            session_id=backend_session,
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
        )

    async def close_session(self, session_id: str, **_kw: Any) -> Any:
        """Forward to bound backend and tear down the wrapper.

        v1 keeps one backend per connection; after a successful close the
        wrapper has nothing more to do, so reap it eagerly instead of
        leaving it around to be killed by ``_teardown_backend`` on
        disconnect.  Returns ``None`` when no backend was ever bound.
        """
        self._require_client_session(session_id)
        if self._backend is None or self._backend_session_id is None:
            return None
        try:
            return await self._backend.close_session(session_id=self._backend_session_id)
        finally:
            await self._teardown_backend()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Forward extension methods to the bound backend."""
        backend, _ = self._require_bound()
        return await backend.ext_method(method, params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Forward extension notifications to the bound backend (silent if not bound)."""
        if self._backend is not None:
            await self._backend.ext_notification(method, params)

    # ── Client protocol (backend → proxy → connected client) ──────────

    async def session_update(
        self,
        session_id: str,
        update: SessionUpdatePayload,
        **_kw: Any,
    ) -> None:
        """Rewrite session id and any model ids, then forward to the client."""
        del session_id
        if isinstance(update, ConfigOptionUpdate) and self._bound_agent is not None:
            namespace_model_options_in_place(update.config_options, self._bound_agent)
        await self._require_client().session_update(session_id=CLIENT_SESSION_ID, update=update)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **_kw: Any,
    ) -> RequestPermissionResponse:
        """Forward permission request to the connected client."""
        del session_id
        return await self._require_client().request_permission(
            options=options, session_id=CLIENT_SESSION_ID, tool_call=tool_call
        )

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **_kw: Any,
    ) -> ReadTextFileResponse:
        """Forward fs read to the connected client."""
        del session_id
        return await self._require_client().read_text_file(
            path=path, session_id=CLIENT_SESSION_ID, limit=limit, line=line
        )

    async def write_text_file(
        self, content: str, path: str, session_id: str, **_kw: Any
    ) -> WriteTextFileResponse | None:
        """Forward fs write to the connected client."""
        del session_id
        return await self._require_client().write_text_file(
            content=content, path=path, session_id=CLIENT_SESSION_ID
        )

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **_kw: Any,
    ) -> CreateTerminalResponse:
        """Forward terminal create to the connected client."""
        del session_id
        return await self._require_client().create_terminal(
            command=command,
            session_id=CLIENT_SESSION_ID,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
        )

    async def terminal_output(
        self, session_id: str, terminal_id: str, **_kw: Any
    ) -> TerminalOutputResponse:
        """Forward terminal output read to the connected client."""
        del session_id
        return await self._require_client().terminal_output(
            session_id=CLIENT_SESSION_ID, terminal_id=terminal_id
        )

    async def release_terminal(
        self, session_id: str, terminal_id: str, **_kw: Any
    ) -> ReleaseTerminalResponse | None:
        """Forward terminal release to the connected client."""
        del session_id
        return await self._require_client().release_terminal(
            session_id=CLIENT_SESSION_ID, terminal_id=terminal_id
        )

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **_kw: Any
    ) -> WaitForTerminalExitResponse:
        """Forward wait-for-exit to the connected client."""
        del session_id
        return await self._require_client().wait_for_terminal_exit(
            session_id=CLIENT_SESSION_ID, terminal_id=terminal_id
        )

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **_kw: Any
    ) -> KillTerminalResponse | None:
        """Forward kill to the connected client."""
        del session_id
        return await self._require_client().kill_terminal(
            session_id=CLIENT_SESSION_ID, terminal_id=terminal_id
        )

    # ── Bind ──────────────────────────────────────────────────────────

    async def _select_model(self, namespaced: str) -> None:
        """Drive the bind/re-pick decision for a namespaced ``agent:model`` id.

        - First call binds the backend.
        - Subsequent calls against the same agent forward to it.
        - Cross-agent switches are rejected at the protocol level — v1
          doesn't carry the multi-backend session bookkeeping that would
          allow them.
        """
        agent_id, model_id = split_namespaced(namespaced)
        if not agent_id or not model_id:
            raise RequestError.invalid_params(
                {"details": f"model id must be 'agent:model', got {namespaced!r}"}
            )
        async with self._bind_lock:
            if self._bound_agent is None:
                await self._bind(agent_id, model_id)
                return
            if agent_id != self._bound_agent:
                raise RequestError.invalid_params(
                    {
                        "details": (
                            f"session is already bound to agent {self._bound_agent!r}; "
                            f"v1 does not support cross-agent switches"
                        )
                    }
                )
        backend, backend_session = self._require_bound()
        await backend.set_session_model(model_id=model_id, session_id=backend_session)

    async def _ensure_bound_for_default(self) -> None:
        """Bind to the advertised default model if no backend is bound yet."""
        async with self._bind_lock:
            if self._backend is not None:
                return
            if self._default_namespaced is None:
                raise RequestError.invalid_request(
                    {
                        "details": "no agent available — none of the configured wrappers probed successfully"
                    }
                )
            agent_id, model_id = split_namespaced(self._default_namespaced)
            await self._bind(agent_id, model_id)

    async def _bind(self, agent_id: str, model_id: str) -> None:
        """Spawn the backend wrapper and replay the handshake.

        Callers hold [`_bind_lock`][terok_executor.acp.proxy.ACPProxy._bind_lock]
        — concurrent bind attempts from a racing ``prompt`` and
        ``set_session_model`` collapse to one.
        """
        command, *args = self._roster.wrapper_argv(agent_id)
        try:
            backend, _proc = await self._backend_stack.enter_async_context(
                spawn_agent_process(self, command, *args)
            )
            self._backend = backend
            await backend.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=self._client_capabilities or ClientCapabilities(),
            )
            new_resp = await backend.new_session(
                cwd=CONTAINER_WORKSPACE,
                mcp_servers=self._client_mcp_servers or [],
                additional_directories=self._client_additional_directories,
            )
            self._backend_session_id = new_resp.session_id
            await backend.set_session_model(model_id=model_id, session_id=self._backend_session_id)
        except Exception as exc:
            await self._teardown_backend()
            raise AgentBindError(f"bind {agent_id!r}: {exc}") from exc
        self._bound_agent = agent_id
        _logger.debug("ACP proxy: bound agent=%r model=%r", agent_id, model_id)

    async def _teardown_backend(self) -> None:
        """Close the backend exit stack and reset state for a possible rebind."""
        try:
            await self._backend_stack.aclose()
        except Exception:  # noqa: BLE001
            _logger.debug("ACP proxy: backend teardown error", exc_info=True)
        self._backend_stack = AsyncExitStack()
        self._backend = None
        self._backend_session_id = None
        # Keep _bound_agent set if we've already declared one — clearing
        # it would let a misbehaving client switch agents mid-session
        # past the v1 guard.  On clean disconnect the proxy is discarded
        # so the state doesn't outlive the connection.

    def _require_bound(self) -> tuple[ClientSideConnection, str]:
        """Return ``(backend, backend_session_id)`` or raise ``invalid_request``.

        Single helper for the forward-only handlers: callers get a
        statically-narrowed ``ClientSideConnection`` plus the session id
        the backend gave us at bind time, so mypy never has to second-
        guess the optional fields.
        """
        if self._backend is None or self._backend_session_id is None:
            raise RequestError.invalid_request({"details": "no agent bound — pick a model first"})
        return self._backend, self._backend_session_id

    def _require_client(self) -> AgentSideConnection:
        """Return the connected-client wrapper or raise (unreachable in practice).

        [`run`][terok_executor.acp.proxy.ACPProxy.run] sets ``self._client``
        before `acp.agent.connection.AgentSideConnection.listen`
        starts dispatching frames, so Client-side forwarders only run after
        ``_client`` is set — but mypy doesn't know that.  The runtime check
        costs a comparison; the alternative is one ``assert`` per forwarder.
        """
        if self._client is None:
            raise RuntimeError("proxy used before run() — client side not wired")
        return self._client

    def _require_client_session(self, session_id: str) -> None:
        """Reject session-scoped calls that target an unknown session id.

        The proxy advertises exactly one session per connection
        (`CLIENT_SESSION_ID`) and only after ``session/new`` has been
        called.  Anything else is either a client bug (stale id) or a
        protocol-violating request that would otherwise mutate backend
        state from an invalid client view.
        """
        if not self._session_announced or session_id != CLIENT_SESSION_ID:
            raise RequestError.invalid_request({"details": f"unknown session id: {session_id!r}"})


class AgentBindError(RuntimeError):
    """Surface error raised when the proxy fails to bind a backend agent.

    Always converted to a JSON-RPC error response on the wire — never
    bubbles to the caller of [`run`][terok_executor.acp.proxy.ACPProxy.run].
    """
