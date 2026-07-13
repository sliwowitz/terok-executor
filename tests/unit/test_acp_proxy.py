# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed ACP proxy.

The proxy implements both sides of the ACP protocol on one object — it
acts as an [`Agent`][acp.Agent] toward the connected client and as a
[`Client`][acp.Client] toward the in-container backend wrapper.  These
unit tests drive the typed methods directly (no JSON-RPC framing) and
patch [`spawn_agent_process`][acp.spawn_agent_process] when bind
behaviour is exercised, so no real subprocess ever starts.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from acp import RequestError
from acp.schema import (
    ConfigOptionUpdate,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
)

from terok_executor.acp import proxy as proxy_module
from terok_executor.acp.model_options import (
    MODEL_OPTION_CATEGORY,
    build_aggregated_session_new,
    build_model_option,
    humanise_model_id,
    namespace_model_options_in_place,
)
from terok_executor.acp.proxy import (
    CLIENT_SESSION_ID,
    ACPProxy,
    AgentBindError,
)

# JSON-RPC error codes the proxy maps onto via `RequestError.invalid_*`.
_JSONRPC_INVALID_REQUEST = -32600
_JSONRPC_INVALID_PARAMS = -32602

# The id a backend gives its *own* model selector.  Deliberately not the
# proxy's ``"model"``: the proxy must address the backend by the id that
# backend advertised, not by assuming its own convention holds downstream.
_BACKEND_MODEL_CONFIG_ID = "llm"


class _StubRoster:
    """Minimal stand-in for :class:`ACPRoster`.

    The proxy only reads ``list_available_agents`` and ``wrapper_argv``;
    a thin stub keeps tests fast and isolated from sandbox plumbing.
    """

    def __init__(self, available: list[str]) -> None:
        self._available = available

    async def list_available_agents(self) -> list[str]:
        """Return the canned ``agent:model`` list."""
        return list(self._available)

    def wrapper_argv(self, agent_id: str) -> list[str]:
        """Return a sentinel argv — never exec'd in unit tests."""
        return ["echo", f"terok-{agent_id}-acp"]


class _FakeBackend:
    """Records typed calls and returns canned responses for the proxy's bind path."""

    def __init__(self, *, session_id: str = "be-1") -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.session_id = session_id

    async def initialize(self, **kw: Any) -> InitializeResponse:
        self.calls.append(("initialize", kw))
        return InitializeResponse(protocol_version=kw["protocol_version"])

    async def new_session(self, **kw: Any) -> NewSessionResponse:
        # A real 0.11 wrapper advertises its models as a select among its
        # config options — the dedicated ``models`` block is gone.  The
        # proxy reads this to learn which option a model pick addresses.
        self.calls.append(("new_session", kw))
        return NewSessionResponse(
            session_id=self.session_id,
            config_options=[
                SessionConfigOptionSelect(
                    id=_BACKEND_MODEL_CONFIG_ID,
                    name="Model",
                    type="select",
                    category="model",
                    current_value="opus-4.6",
                    options=[
                        SessionConfigSelectOption(value="opus-4.6", name="Opus"),
                        SessionConfigSelectOption(value="haiku-4.5", name="Haiku"),
                    ],
                )
            ],
        )

    async def prompt(self, **kw: Any) -> PromptResponse:
        self.calls.append(("prompt", kw))
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, **kw: Any) -> None:
        self.calls.append(("cancel", kw))

    async def close_session(self, **kw: Any) -> None:
        self.calls.append(("close_session", kw))
        return None

    async def set_config_option(self, **kw: Any) -> SetSessionConfigOptionResponse:
        # Echoes a bare-id model option so the proxy's post-bind
        # ``namespace_model_options_in_place`` rewrite is observable
        # end-to-end.
        self.calls.append(("set_config_option", kw))
        return SetSessionConfigOptionResponse(
            config_options=[
                SessionConfigOptionSelect(
                    id="model",
                    name="Model",
                    type="select",
                    category="model",
                    current_value="opus-4.6",
                    options=[SessionConfigSelectOption(value="opus-4.6", name="Opus")],
                )
            ]
        )

    # ── Forward-only Agent methods (used by post-bind forwarding tests) ──

    async def authenticate(self, **kw: Any) -> None:
        self.calls.append(("authenticate", kw))
        return None

    async def set_session_mode(self, **kw: Any) -> None:
        self.calls.append(("set_session_mode", kw))
        return None

    async def load_session(self, **kw: Any) -> None:
        self.calls.append(("load_session", kw))
        return None

    async def list_sessions(self, **kw: Any) -> Any:
        self.calls.append(("list_sessions", kw))
        return None

    async def fork_session(self, **kw: Any) -> Any:
        self.calls.append(("fork_session", kw))
        return None

    async def resume_session(self, **kw: Any) -> Any:
        self.calls.append(("resume_session", kw))
        return None

    async def ext_method(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("ext_method", {"method": name, "params": params}))
        return {"ok": True}

    async def ext_notification(self, name: str, params: dict[str, Any]) -> None:
        self.calls.append(("ext_notification", {"method": name, "params": params}))


def _patch_spawn(monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend) -> None:
    """Install *backend* as the next ``spawn_agent_process`` result."""

    @asynccontextmanager
    async def _fake_spawn(_client, _command, *_args, **_kw):
        yield backend, None

    monkeypatch.setattr(proxy_module, "spawn_agent_process", _fake_spawn)


def _new_proxy(available: list[str]) -> ACPProxy:
    """Build a proxy backed by ``_StubRoster(available)``."""
    return ACPProxy(roster=_StubRoster(available))  # type: ignore[arg-type]


async def _pick_model(
    proxy: ACPProxy, namespaced: str, *, session_id: str = CLIENT_SESSION_ID
) -> SetSessionConfigOptionResponse | None:
    """Pick *namespaced* the way a client does.

    Since ACP 1.16 the config-option channel is the *only* way to select a
    model — ``session/set_model`` no longer exists — so this is what every
    bind-driving test goes through.
    """
    return await proxy.set_config_option(
        config_id=MODEL_OPTION_CATEGORY, session_id=session_id, value=namespaced
    )


def _backend_model_picks(backend: _FakeBackend) -> list[str]:
    """Model values the proxy pushed onto *backend*'s own model selector."""
    return [
        kw["value"]
        for name, kw in backend.calls
        if name == "set_config_option" and kw["config_id"] == _BACKEND_MODEL_CONFIG_ID
    ]


class TestInitialize:
    """Pre-bind ``initialize`` is answered locally."""

    def test_returns_implementation_metadata(self) -> None:
        """Response carries protocol version and proxy identification."""
        proxy = _new_proxy([])
        resp = asyncio.run(proxy.initialize(protocol_version=1))
        assert resp.agent_info is not None
        assert resp.agent_info.name == "terok-acp"

    def test_captures_client_capabilities_for_bind_replay(self) -> None:
        """Caps from the client land on the proxy for verbatim replay on bind."""
        from acp.schema import ClientCapabilities

        proxy = _new_proxy([])
        caps = ClientCapabilities()
        asyncio.run(proxy.initialize(protocol_version=1, client_capabilities=caps))
        assert proxy._client_capabilities is caps


class TestSessionNew:
    """Pre-bind ``session/new`` aggregates the model list locally."""

    def test_returns_synthetic_session_id(self) -> None:
        """Synthetic ``proxy-1`` is returned — no backend exists yet."""
        proxy = _new_proxy(["claude:opus-4.6"])
        resp = asyncio.run(proxy.new_session(cwd="/host/proj", mcp_servers=[]))
        assert resp.session_id == CLIENT_SESSION_ID

    def test_aggregates_namespaced_model_options(self) -> None:
        """Every available ``agent:model`` appears in the ``configOptions`` selector."""
        proxy = _new_proxy(["claude:opus-4.6", "codex:gpt-5.5"])
        resp = asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        assert resp.config_options is not None
        model_opt = next(opt for opt in resp.config_options if opt.category == "model")
        assert isinstance(model_opt, SessionConfigOptionSelect)
        assert [e.value for e in model_opt.options] == [
            "claude:opus-4.6",
            "codex:gpt-5.5",
        ]

    def test_rejects_second_session_new(self) -> None:
        """v1 supports one session per connection — second call errors."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        assert exc.value.code == _JSONRPC_INVALID_REQUEST

    def test_remembers_default_for_lazy_bind(self) -> None:
        """The first listed model is the lazy-bind default for prompts."""
        proxy = _new_proxy(["claude:opus-4.6", "codex:gpt-5.5"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        assert proxy._default_namespaced == "claude:opus-4.6"


class TestSetConfigOptionPreBind:
    """Clients send model selection via ``set_config_option``."""

    def test_model_with_bad_namespace_raises(self) -> None:
        """Malformed ``agent:model`` short-circuits before any spawn."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(
                proxy.set_config_option(
                    config_id="model",
                    session_id=CLIENT_SESSION_ID,
                    value="no-namespace",
                )
            )
        assert exc.value.code == _JSONRPC_INVALID_PARAMS

    def test_non_model_category_pre_bind_errors(self) -> None:
        """Pre-bind ``set_config_option`` for a non-model knob has no backend."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(
                proxy.set_config_option(
                    config_id="behavior",
                    session_id=CLIENT_SESSION_ID,
                    value="strict",
                )
            )
        assert exc.value.code == _JSONRPC_INVALID_REQUEST


class TestPromptLazyBindGate:
    """``prompt`` lazy-binds when a default exists, otherwise errors."""

    def test_prompt_without_any_available_agent_raises(self) -> None:
        """No probed agents → no default → prompt has nothing to bind to."""
        proxy = _new_proxy([])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(proxy.prompt(prompt=[], session_id=CLIENT_SESSION_ID))
        assert exc.value.code == _JSONRPC_INVALID_REQUEST


class TestSessionIdValidation:
    """Session-scoped handlers reject stale or pre-``session/new`` ids."""

    def test_model_pick_pre_new_session_rejected(self) -> None:
        """Picking a model before ``session/new`` errors."""
        proxy = _new_proxy(["claude:opus-4.6"])
        with pytest.raises(RequestError) as exc:
            asyncio.run(_pick_model(proxy, "claude:opus-4.6"))
        assert exc.value.code == _JSONRPC_INVALID_REQUEST

    def test_prompt_with_unknown_session_id_rejected(self) -> None:
        """Arbitrary client-supplied session ids are rejected post-new."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        with pytest.raises(RequestError) as exc:
            asyncio.run(proxy.prompt(prompt=[], session_id="someone-elses-session"))
        assert exc.value.code == _JSONRPC_INVALID_REQUEST


class TestBind:
    """End-to-end bind flow with a patched ``spawn_agent_process``."""

    def test_model_pick_drives_backend_handshake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The first model pick spawns the backend + replays the three frames.

        The patched backend captures the exact arguments so we can pin
        the handshake shape: namespace stripped on the way down, client
        ``cwd`` overridden to the container workspace, ``mcp_servers``
        defaulted to the empty list when the client didn't supply any,
        and the pick addressed to the backend's *own* selector id.
        """
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/host/proj", mcp_servers=None)
            resp = await _pick_model(proxy, "claude:opus-4.6")
            assert isinstance(resp, SetSessionConfigOptionResponse)

        asyncio.run(_drive())

        method_order = [name for name, _ in backend.calls]
        assert method_order == ["initialize", "new_session", "set_config_option"]
        new_session_call = backend.calls[1][1]
        assert new_session_call["cwd"] == proxy_module.CONTAINER_WORKSPACE
        assert new_session_call["mcp_servers"] == []
        set_model_call = backend.calls[2][1]
        assert set_model_call["config_id"] == _BACKEND_MODEL_CONFIG_ID
        assert set_model_call["value"] == "opus-4.6"  # namespace stripped
        assert set_model_call["session_id"] == backend.session_id

    def test_backend_without_model_selector_binds_anyway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-model wrapper advertises no selector — the pick is a no-op, not an error."""

        class _NoModelsBackend(_FakeBackend):
            async def new_session(self, **kw: Any) -> NewSessionResponse:
                self.calls.append(("new_session", kw))
                return NewSessionResponse(session_id=self.session_id)

        backend = _NoModelsBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await _pick_model(proxy, "claude:opus-4.6")

        asyncio.run(_drive())
        assert [name for name, _ in backend.calls] == ["initialize", "new_session"]
        assert proxy._bound_agent == "claude"

    def test_additional_directories_forwarded_on_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client-supplied ``additional_directories`` reach the backend's ``session/new``."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(
                cwd="/host/proj", mcp_servers=[], additional_directories=["/extra"]
            )
            await _pick_model(proxy, "claude:opus-4.6")

        asyncio.run(_drive())
        new_session_call = backend.calls[1][1]
        assert new_session_call["additional_directories"] == ["/extra"]

    def test_cross_agent_pick_after_bind_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """v1 forbids cross-agent switches; the second pick errors out."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6", "codex:gpt-5.5"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await _pick_model(proxy, "claude:opus-4.6")
            with pytest.raises(RequestError) as exc:
                await _pick_model(proxy, "codex:gpt-5.5")
            assert exc.value.code == _JSONRPC_INVALID_PARAMS

        asyncio.run(_drive())

    def test_same_agent_repick_forwards_to_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-picking a model on the same agent forwards through stripped."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6", "claude:haiku-4.5"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await _pick_model(proxy, "claude:opus-4.6")
            await _pick_model(proxy, "claude:haiku-4.5")

        asyncio.run(_drive())

        # First push is part of the bind handshake, second is the re-pick.
        assert _backend_model_picks(backend) == ["opus-4.6", "haiku-4.5"]

    def test_bind_failure_propagates_as_agent_bind_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backend initialize failure tears down and bubbles ``AgentBindError``."""

        class _FailingBackend(_FakeBackend):
            async def initialize(self, **_kw: Any) -> InitializeResponse:
                raise RuntimeError("simulated wrapper crash")

        _patch_spawn(monkeypatch, _FailingBackend())
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            with pytest.raises(AgentBindError):
                await _pick_model(proxy, "claude:opus-4.6")
            # State reset — a retry should be possible.
            assert proxy._backend is None
            assert proxy._bound_agent is None

        asyncio.run(_drive())


class TestBackendForwarding:
    """Post-bind responses get model ids re-namespaced on the way out."""

    def test_set_config_option_response_namespaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backend echoes bare ``opus-4.6``; client sees ``claude:opus-4.6``."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> SetSessionConfigOptionResponse | None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await _pick_model(proxy, "claude:opus-4.6")
            return await proxy.set_config_option(
                config_id="theme",
                session_id=CLIENT_SESSION_ID,
                value="dark",
            )

        resp = asyncio.run(_drive())
        assert resp is not None
        model_opt = next(o for o in resp.config_options if o.category == "model")
        assert isinstance(model_opt, SessionConfigOptionSelect)
        assert model_opt.current_value == "claude:opus-4.6"
        assert [e.value for e in model_opt.options] == ["claude:opus-4.6"]

    def test_close_session_with_no_backend_is_noop(self) -> None:
        """``close_session`` before any bind returns ``None`` without erroring."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.new_session(cwd="/x", mcp_servers=[]))
        result = asyncio.run(proxy.close_session(session_id=CLIENT_SESSION_ID))
        assert result is None

    def test_set_config_option_model_binds_and_returns_aggregated_option(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Older Zed's model selection (``configId="model"``) binds and returns the aggregate."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6", "claude:haiku-4.5"])

        async def _drive() -> SetSessionConfigOptionResponse | None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            return await proxy.set_config_option(
                config_id="model",
                session_id=CLIENT_SESSION_ID,
                value="claude:haiku-4.5",
            )

        resp = asyncio.run(_drive())
        assert resp is not None
        opt = resp.config_options[0]
        assert isinstance(opt, SessionConfigOptionSelect)
        assert opt.current_value == "claude:haiku-4.5"
        assert [e.value for e in opt.options] == ["claude:opus-4.6", "claude:haiku-4.5"]

    def test_prompt_lazy_binds_to_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A client that never picks a model binds lazily on first ``prompt``."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await proxy.prompt(prompt=[], session_id=CLIENT_SESSION_ID)

        asyncio.run(_drive())
        method_order = [name for name, _ in backend.calls]
        assert method_order == ["initialize", "new_session", "set_config_option", "prompt"]
        assert _backend_model_picks(backend) == ["opus-4.6"]

    def test_close_session_tears_backend_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``close_session`` reaps the wrapper; the proxy can rebind afterwards."""
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await _pick_model(proxy, "claude:opus-4.6")
            await proxy.close_session(session_id=CLIENT_SESSION_ID)

        asyncio.run(_drive())
        assert any(name == "close_session" for name, _ in backend.calls)
        assert proxy._backend is None
        assert proxy._backend_session_id is None


class TestNamespaceModelOptionsInPlace:
    """Typed in-place rewriter — used on every backend → client config option."""

    def test_namespaces_select_current_and_values(self) -> None:
        """Bare ``opus-4.6`` becomes ``claude:opus-4.6`` for both fields."""
        opt = SessionConfigOptionSelect(
            id="model",
            name="Model",
            type="select",
            category="model",
            current_value="opus-4.6",
            options=[
                SessionConfigSelectOption(value="opus-4.6", name="Opus"),
                SessionConfigSelectOption(value="haiku-4.5", name="Haiku"),
            ],
        )
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "claude:opus-4.6"
        assert [e.value for e in opt.options] == ["claude:opus-4.6", "claude:haiku-4.5"]

    def test_already_namespaced_left_untouched(self) -> None:
        """Idempotent — round-tripping a proxy-built option doesn't double-prefix."""
        opt = build_model_option(["claude:opus-4.6", "claude:haiku-4.5"], current="claude:opus-4.6")
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "claude:opus-4.6"
        assert [e.value for e in opt.options] == ["claude:opus-4.6", "claude:haiku-4.5"]

    def test_colon_bearing_backend_value_still_prefixed(self) -> None:
        """A bare model id that contains a colon (``azure:gpt-4.1``) still gets prefixed.

        Idempotency keys on the agent prefix, not on "contains a colon" —
        otherwise multi-vendor backends that already namespace their own
        ids would leak through as bare values the client can't address.
        """
        opt = SessionConfigOptionSelect(
            id="model",
            name="Model",
            type="select",
            category="model",
            current_value="azure:gpt-4.1",
            options=[SessionConfigSelectOption(value="azure:gpt-4.1", name="GPT-4.1")],
        )
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "claude:azure:gpt-4.1"
        assert [e.value for e in opt.options] == ["claude:azure:gpt-4.1"]

    def test_non_model_category_untouched(self) -> None:
        """Other categories pass through unchanged."""
        opt = SessionConfigOptionSelect(
            id="mode",
            name="Mode",
            type="select",
            category="mode",
            current_value="ask",
            options=[SessionConfigSelectOption(value="ask", name="Ask")],
        )
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "ask"

    def test_empty_or_none_input_is_noop(self) -> None:
        """Both ``None`` and ``[]`` inputs are accepted without error."""
        namespace_model_options_in_place(None, "claude")
        namespace_model_options_in_place([], "claude")

    def test_select_group_options_are_namespaced(self) -> None:
        """Grouped options (``SessionConfigSelectGroup``) get the same rewrite as flat ones.

        Some backends present model variants grouped by family; the
        rewriter must descend into each group's ``options`` list.
        """
        from acp.schema import SessionConfigSelectGroup

        opt = SessionConfigOptionSelect(
            id="model",
            name="Model",
            type="select",
            category="model",
            current_value="opus-4.6",
            options=[
                SessionConfigSelectGroup(
                    group="claude-3",
                    name="Claude 3",
                    options=[
                        SessionConfigSelectOption(value="opus-4.6", name="Opus"),
                        SessionConfigSelectOption(value="haiku-4.5", name="Haiku"),
                    ],
                )
            ],
        )
        namespace_model_options_in_place([opt], "claude")
        assert opt.current_value == "claude:opus-4.6"
        group = opt.options[0]
        assert isinstance(group, SessionConfigSelectGroup)
        assert [e.value for e in group.options] == ["claude:opus-4.6", "claude:haiku-4.5"]


class TestSessionUpdateForwarding:
    """Backend → proxy → client session updates rewrite session id and model ids."""

    def test_config_option_update_namespaces_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``ConfigOptionUpdate`` carrying the model option gets namespaced.

        The proxy's ``Client.session_update`` is wired to the connected
        client side; replace the latter with a recorder to capture the
        rewritten typed update.
        """
        backend = _FakeBackend()
        _patch_spawn(monkeypatch, backend)
        proxy = _new_proxy(["claude:opus-4.6"])

        captured: list[Any] = []

        class _RecordingClient:
            async def session_update(self, *, session_id: str, update: Any) -> None:
                captured.append((session_id, update))

        async def _drive() -> None:
            await proxy.initialize(protocol_version=1)
            await proxy.new_session(cwd="/x", mcp_servers=[])
            await _pick_model(proxy, "claude:opus-4.6")
            # Inject the recorder *after* bind so the proxy's normal
            # initialise path isn't disturbed.
            proxy._client = _RecordingClient()  # type: ignore[assignment]
            await proxy.session_update(
                session_id=backend.session_id,
                update=ConfigOptionUpdate(
                    session_update="config_option_update",
                    config_options=[
                        SessionConfigOptionSelect(
                            id="model",
                            name="Model",
                            type="select",
                            category="model",
                            current_value="opus-4.6",
                            options=[SessionConfigSelectOption(value="opus-4.6", name="Opus")],
                        )
                    ],
                ),
            )

        asyncio.run(_drive())

        assert len(captured) == 1
        session_id, update = captured[0]
        assert session_id == CLIENT_SESSION_ID
        model_opt = update.config_options[0]
        assert isinstance(model_opt, SessionConfigOptionSelect)
        assert model_opt.current_value == "claude:opus-4.6"
        assert model_opt.options[0].value == "claude:opus-4.6"


class TestBuildHelpers:
    """The pre-bind aggregate builders."""

    def test_build_aggregated_session_new_empty_models(self) -> None:
        """Empty list yields a schema-valid response with no selector at all."""
        resp = build_aggregated_session_new("sess-x", [])
        assert resp.session_id == "sess-x"
        assert resp.config_options is None

    def test_humanise_model_id_round_trip(self) -> None:
        """The label format is ``Agent: model``."""
        assert humanise_model_id("claude:opus-4.6") == "Claude: opus-4.6"

    def test_humanise_model_id_preserves_slashes_in_model(self) -> None:
        """OpenRouter-style slash-bearing model ids survive humanisation."""
        assert humanise_model_id("opencode:opencode/big-pickle") == "Opencode: opencode/big-pickle"

    def test_humanise_unnamespaced_passes_through(self) -> None:
        """Unrecognised ids are returned verbatim — no crash."""
        assert humanise_model_id("plain") == "plain"


class _RecordingClient:
    """Stand-in for the proxy's [`AgentSideConnection`][acp.agent.connection.AgentSideConnection].

    Captures every Client-side typed call the proxy forwards on behalf
    of the backend.  All methods record + return a sensible default so
    a test can drive any Client-side path without configuring per-method
    canned responses up front.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def session_update(self, *, session_id: str, update: Any, **_kw: Any) -> None:
        self.calls.append(("session_update", {"session_id": session_id, "update": update}))

    async def request_permission(self, **kw: Any) -> Any:
        self.calls.append(("request_permission", kw))
        return None

    async def read_text_file(self, **kw: Any) -> Any:
        self.calls.append(("read_text_file", kw))
        return None

    async def write_text_file(self, **kw: Any) -> Any:
        self.calls.append(("write_text_file", kw))
        return None

    async def create_terminal(self, **kw: Any) -> Any:
        self.calls.append(("create_terminal", kw))
        return None

    async def terminal_output(self, **kw: Any) -> Any:
        self.calls.append(("terminal_output", kw))
        return None

    async def release_terminal(self, **kw: Any) -> Any:
        self.calls.append(("release_terminal", kw))
        return None

    async def wait_for_terminal_exit(self, **kw: Any) -> Any:
        self.calls.append(("wait_for_terminal_exit", kw))
        return None

    async def kill_terminal(self, **kw: Any) -> Any:
        self.calls.append(("kill_terminal", kw))
        return None


def _bound_proxy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: tuple[str, ...] = ("claude:opus-4.6",),
) -> tuple[ACPProxy, _FakeBackend, _RecordingClient]:
    """Build a proxy already past the bind handshake, wired to recorders.

    Returns ``(proxy, backend, client)`` so the test can assert against
    either side of the typed bridge after exercising a forwarder.
    """
    backend = _FakeBackend()
    _patch_spawn(monkeypatch, backend)
    proxy = _new_proxy(list(available))

    async def _bind() -> None:
        await proxy.initialize(protocol_version=1)
        await proxy.new_session(cwd="/x", mcp_servers=[])
        await _pick_model(proxy, available[0])

    asyncio.run(_bind())
    client = _RecordingClient()
    proxy._client = client  # type: ignore[assignment]
    backend.calls.clear()
    return proxy, backend, client


class TestPostBindAgentForwarders:
    """Post-bind Agent methods forward to the backend with translated ids."""

    def test_cancel_forwards_with_translated_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.cancel(session_id=CLIENT_SESSION_ID))
        assert backend.calls[-1] == ("cancel", {"session_id": backend.session_id})

    def test_authenticate_forwards_method_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.authenticate(method_id="api_key"))
        assert backend.calls[-1] == ("authenticate", {"method_id": "api_key"})

    def test_set_session_mode_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.set_session_mode(mode_id="ask", session_id=CLIENT_SESSION_ID))
        call_kw = backend.calls[-1][1]
        assert call_kw == {"mode_id": "ask", "session_id": backend.session_id}

    def test_load_session_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.load_session(cwd="/x", session_id=CLIENT_SESSION_ID, mcp_servers=[]))
        assert backend.calls[-1][0] == "load_session"
        assert backend.calls[-1][1]["session_id"] == backend.session_id

    def test_list_sessions_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ACP 1.16 scopes listing by ``cwd`` alone — ``additionalDirectories``
        # is no longer part of the request.
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.list_sessions(cursor="abc"))
        assert backend.calls[-1] == ("list_sessions", {"cursor": "abc", "cwd": None})

    def test_fork_session_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.fork_session(cwd="/x", session_id=CLIENT_SESSION_ID, mcp_servers=[]))
        assert backend.calls[-1][1]["session_id"] == backend.session_id

    def test_resume_session_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.resume_session(cwd="/x", session_id=CLIENT_SESSION_ID, mcp_servers=[]))
        assert backend.calls[-1][1]["session_id"] == backend.session_id

    def test_ext_method_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        out = asyncio.run(proxy.ext_method("foo", {"bar": 1}))
        assert out == {"ok": True}
        assert backend.calls[-1] == ("ext_method", {"method": "foo", "params": {"bar": 1}})

    def test_ext_notification_forwards_when_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.ext_notification("evt", {"x": 1}))
        assert backend.calls[-1] == ("ext_notification", {"method": "evt", "params": {"x": 1}})

    def test_ext_notification_silent_pre_bind(self) -> None:
        """Pre-bind ext notifications drop silently — no exception, no backend touched."""
        proxy = _new_proxy(["claude:opus-4.6"])
        asyncio.run(proxy.ext_notification("evt", {"x": 1}))
        assert proxy._backend is None

    def test_set_config_option_non_model_post_bind_forwards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(
            proxy.set_config_option(config_id="theme", session_id=CLIENT_SESSION_ID, value="dark")
        )
        assert backend.calls[-1][0] == "set_config_option"
        assert backend.calls[-1][1]["config_id"] == "theme"
        assert backend.calls[-1][1]["session_id"] == backend.session_id

    def test_prompt_post_bind_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, backend, _ = _bound_proxy(monkeypatch)
        resp = asyncio.run(proxy.prompt(prompt=[], session_id=CLIENT_SESSION_ID))
        assert isinstance(resp, PromptResponse)
        assert backend.calls[-1][0] == "prompt"
        assert backend.calls[-1][1]["session_id"] == backend.session_id

    def test_prompt_does_not_relay_retired_message_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client still sending ``message_id`` gets it dropped, not forwarded.

        ACP 1.16 removed the field from ``PromptRequest``; relaying it to a
        0.11 backend would be a protocol violation.
        """
        proxy, backend, _ = _bound_proxy(monkeypatch)
        asyncio.run(proxy.prompt(prompt=[], session_id=CLIENT_SESSION_ID, message_id="m1"))
        assert "message_id" not in backend.calls[-1][1]


class TestClientSideForwarders:
    """Backend → proxy → client forwarders rewrite the session id to ``CLIENT_SESSION_ID``."""

    def test_session_update_non_config_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from acp.schema import AgentMessageChunk

        proxy, _backend, client = _bound_proxy(monkeypatch)
        chunk = AgentMessageChunk(
            session_update="agent_message_chunk",
            content={"type": "text", "text": "hi"},
        )
        asyncio.run(proxy.session_update(session_id="be-1", update=chunk))
        name, kw = client.calls[-1]
        assert name == "session_update"
        assert kw["session_id"] == CLIENT_SESSION_ID
        assert kw["update"] is chunk

    def test_request_permission_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from acp.schema import PermissionOption, ToolCallUpdate

        proxy, _backend, client = _bound_proxy(monkeypatch)
        options = [PermissionOption(option_id="approve", name="Approve", kind="allow_once")]
        tool_call = ToolCallUpdate(tool_call_id="t1")
        asyncio.run(
            proxy.request_permission(options=options, session_id="be-1", tool_call=tool_call)
        )
        name, kw = client.calls[-1]
        assert name == "request_permission"
        assert kw["session_id"] == CLIENT_SESSION_ID
        assert kw["options"] == options

    def test_read_text_file_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, _backend, client = _bound_proxy(monkeypatch)
        asyncio.run(proxy.read_text_file(path="/a", session_id="be-1", limit=10, line=2))
        name, kw = client.calls[-1]
        assert name == "read_text_file"
        assert kw == {"path": "/a", "session_id": CLIENT_SESSION_ID, "limit": 10, "line": 2}

    def test_write_text_file_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, _backend, client = _bound_proxy(monkeypatch)
        asyncio.run(proxy.write_text_file(content="hi", path="/a", session_id="be-1"))
        name, kw = client.calls[-1]
        assert name == "write_text_file"
        assert kw == {"content": "hi", "path": "/a", "session_id": CLIENT_SESSION_ID}

    def test_create_terminal_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proxy, _backend, client = _bound_proxy(monkeypatch)
        asyncio.run(proxy.create_terminal(command="ls", session_id="be-1", args=["-la"]))
        name, kw = client.calls[-1]
        assert name == "create_terminal"
        assert kw["session_id"] == CLIENT_SESSION_ID
        assert kw["command"] == "ls"
        assert kw["args"] == ["-la"]

    @pytest.mark.parametrize(
        "method",
        ["terminal_output", "release_terminal", "wait_for_terminal_exit", "kill_terminal"],
    )
    def test_terminal_method_forwards(self, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
        proxy, _backend, client = _bound_proxy(monkeypatch)
        coro = getattr(proxy, method)(session_id="be-1", terminal_id="term-1")
        asyncio.run(coro)
        name, kw = client.calls[-1]
        assert name == method
        assert kw == {"session_id": CLIENT_SESSION_ID, "terminal_id": "term-1"}
