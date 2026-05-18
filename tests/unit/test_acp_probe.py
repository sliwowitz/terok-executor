# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed ACP model-roster probe.

The probe stands a [`ClientSideConnection`][acp.ClientSideConnection]
up against the in-container wrapper (``terok-{agent}-acp``) and drives
the minimal handshake: ``initialize`` → ``session/new``.  These unit
tests patch [`spawn_agent_process`][acp.spawn_agent_process] so no
real subprocess ever starts; the canned backend returns the typed
responses the probe consumes.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from acp.schema import (
    InitializeResponse,
    ModelInfo,
    NewSessionResponse,
    SessionModelState,
)

from terok_executor.acp import probe as probe_module
from terok_executor.acp.probe import ProbeError, probe_agent_models


class _CannedBackend:
    """Stand-in for [`ClientSideConnection`][acp.ClientSideConnection].

    Drives the two-call handshake the probe runs.  Tests override
    ``new_session`` (or set ``raise_on_initialize``) to exercise
    failure paths.
    """

    def __init__(
        self,
        *,
        models: list[str] | None = ("opus-4.6", "haiku-4.5"),
        raise_on_initialize: BaseException | None = None,
        raise_on_new_session: BaseException | None = None,
        session_new_delay: float = 0.0,
    ) -> None:
        self._models = models
        self._raise_on_initialize = raise_on_initialize
        self._raise_on_new_session = raise_on_new_session
        self._session_new_delay = session_new_delay
        self.initialize_calls = 0
        self.new_session_calls = 0

    async def initialize(self, **kw: Any) -> InitializeResponse:
        """Record + canned reply, unless test asked for a failure."""
        self.initialize_calls += 1
        if self._raise_on_initialize is not None:
            raise self._raise_on_initialize
        return InitializeResponse(protocol_version=kw["protocol_version"])

    async def new_session(self, **_kw: Any) -> NewSessionResponse:
        """Record + canned reply, optionally delayed to exercise timeout."""
        self.new_session_calls += 1
        if self._session_new_delay:
            await asyncio.sleep(self._session_new_delay)
        if self._raise_on_new_session is not None:
            raise self._raise_on_new_session
        if not self._models:
            return NewSessionResponse(session_id="be-1")
        return NewSessionResponse(
            session_id="be-1",
            models=SessionModelState(
                available_models=[ModelInfo(model_id=m, name=m) for m in self._models],
                current_model_id=self._models[0],
            ),
        )


def _patch_spawn(monkeypatch: pytest.MonkeyPatch, backend: _CannedBackend) -> None:
    """Install *backend* as the next ``spawn_agent_process`` result.

    The fake asserts the unpacking contract: command + tuple-of-args
    rather than a single list.  A regression where ``probe_agent_models``
    accidentally passed ``wrapper_argv`` whole would otherwise silently
    pass the test.
    """

    @asynccontextmanager
    async def _fake_spawn(_client, command, *args, **_kw):
        assert command == "echo", f"expected command='echo', got {command!r}"
        assert args and args[0].startswith("terok-"), (
            f"expected first arg to be wrapper name, got {args!r}"
        )
        yield backend, None

    monkeypatch.setattr(probe_module, "spawn_agent_process", _fake_spawn)


class TestProbeAgentModels:
    """End-to-end probe with a patched ``spawn_agent_process``."""

    def test_happy_path_returns_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A normal handshake yields the model tuple from ``models.available_models``."""
        _patch_spawn(monkeypatch, _CannedBackend(models=["opus-4.6", "haiku-4.5"]))
        models = asyncio.run(
            probe_agent_models(
                agent_id="claude",
                wrapper_argv=["echo", "terok-claude-acp"],
                timeout=4.0,
            )
        )
        assert models == ("opus-4.6", "haiku-4.5")

    def test_no_models_block_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrapper that supports sessions but has no model picker → empty."""
        _patch_spawn(monkeypatch, _CannedBackend(models=None))
        models = asyncio.run(
            probe_agent_models(
                agent_id="claude",
                wrapper_argv=["echo", "terok-claude-acp"],
                timeout=4.0,
            )
        )
        assert models == ()

    def test_initialize_error_is_probe_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the backend rejects ``initialize``, we raise ``ProbeError``."""
        _patch_spawn(
            monkeypatch,
            _CannedBackend(raise_on_initialize=RuntimeError("wrapper rejected init")),
        )
        with pytest.raises(ProbeError):
            asyncio.run(
                probe_agent_models(
                    agent_id="codex",
                    wrapper_argv=["echo", "terok-codex-acp"],
                    timeout=2.0,
                )
            )

    def test_new_session_error_is_probe_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed ``session/new`` propagates as ``ProbeError``."""
        _patch_spawn(
            monkeypatch,
            _CannedBackend(raise_on_new_session=RuntimeError("no auth")),
        )
        with pytest.raises(ProbeError):
            asyncio.run(
                probe_agent_models(
                    agent_id="codex",
                    wrapper_argv=["echo", "terok-codex-acp"],
                    timeout=2.0,
                )
            )

    def test_timeout_is_probe_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend that hangs past *timeout* raises ``ProbeError``."""
        _patch_spawn(monkeypatch, _CannedBackend(session_new_delay=5.0))
        with pytest.raises(ProbeError):
            asyncio.run(
                probe_agent_models(
                    agent_id="codex",
                    wrapper_argv=["echo", "terok-codex-acp"],
                    timeout=0.2,
                )
            )


class TestProbeClient:
    """``_ProbeClient`` swallows informational frames and fast-fails real requests."""

    def test_session_update_swallowed(self) -> None:
        """A chatty wrapper's progress notifications are ignored, not raised."""
        from terok_executor.acp.probe import _ProbeClient

        asyncio.run(_ProbeClient().session_update(session_id="s", update=None))

    def test_ext_notification_swallowed(self) -> None:
        """Extension notifications during the probe are dropped silently."""
        from terok_executor.acp.probe import _ProbeClient

        asyncio.run(_ProbeClient().ext_notification("evt", {}))

    def test_on_connect_is_noop(self) -> None:
        """``on_connect`` is a protocol hook the probe doesn't use."""
        from terok_executor.acp.probe import _ProbeClient

        _ProbeClient().on_connect(object())

    @pytest.mark.parametrize(
        "method,args",
        [
            ("request_permission", {}),
            ("read_text_file", {}),
            ("write_text_file", {}),
            ("create_terminal", {}),
            ("terminal_output", {}),
            ("release_terminal", {}),
            ("wait_for_terminal_exit", {}),
            ("kill_terminal", {}),
            ("ext_method", {"_name": "x", "_payload": {}}),
        ],
    )
    def test_backend_initiated_request_fast_fails(self, method: str, args: dict[str, Any]) -> None:
        """Any backend-initiated request raises ``method_not_found``.

        Probes should not service requests the proxy would normally
        forward to the connected client; failing fast surfaces the
        misbehaving wrapper instead of letting the handshake hang.
        """
        from acp import RequestError

        from terok_executor.acp.probe import _ProbeClient

        with pytest.raises(RequestError) as exc:
            asyncio.run(getattr(_ProbeClient(), method)(**args))
        assert exc.value.code == -32601  # method_not_found
