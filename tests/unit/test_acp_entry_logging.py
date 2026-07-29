# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Entry-point logging seams for the ACP daemon.

Both the ``terok-executor acp`` verb ([`_handle_acp`][terok_executor.commands._handle_acp])
and the standalone daemon ([`main`][terok_executor.acp.daemon.main]) route
logging through [`configure`][terok_util.configure] before handing off to
[`serve_acp`][terok_executor.acp.daemon.serve_acp].  These smoke tests pin
that wiring with ``serve_acp`` stubbed, so no socket is ever bound.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_executor.acp import daemon
from terok_executor.commands import _handle_acp

_SOCK = "/tmp/terok-testing/acp.sock"


def test_daemon_main_configures_then_serves() -> None:
    """``daemon.main`` applies the ACP logging identity, then serves."""
    with (
        patch("terok_util.configure") as configure,
        patch.object(daemon, "serve_acp", return_value=0) as serve,
    ):
        rc = daemon.main(["ctr", _SOCK])

    assert rc == 0
    configure.assert_called_once()
    assert configure.call_args.kwargs["identifier"] == "terok-executor-acp"
    serve.assert_called_once_with("ctr", Path(_SOCK))


def test_daemon_main_debug_knob_raises_level() -> None:
    """``TEROK_ACP_DEBUG`` drops the configured level to DEBUG."""
    import logging

    with (
        patch("terok_util.configure") as configure,
        patch.object(daemon, "serve_acp", return_value=0),
        patch.dict("os.environ", {"TEROK_ACP_DEBUG": "1"}),
    ):
        daemon.main(["ctr", _SOCK])

    assert configure.call_args.kwargs["level"] == logging.DEBUG


def test_daemon_main_rejects_wrong_argument_count() -> None:
    """The usage guard returns 2 without touching logging or the socket."""
    with (
        patch("terok_util.configure") as configure,
        patch.object(daemon, "serve_acp") as serve,
    ):
        rc = daemon.main(["only-one-arg"])

    assert rc == 2
    configure.assert_not_called()
    serve.assert_not_called()


def test_handle_acp_configures_then_exits_with_serve_rc() -> None:
    """The ``acp`` verb configures logging and exits on ``serve_acp``'s code."""
    with (
        patch("terok_util.configure") as configure,
        patch.object(daemon, "serve_acp", return_value=3) as serve,
        pytest.raises(SystemExit) as exc,
    ):
        _handle_acp(container_name="ctr", socket_path=_SOCK)

    assert exc.value.code == 3
    configure.assert_called_once()
    assert configure.call_args.kwargs["identifier"] == "terok-executor-acp"
    serve.assert_called_once_with("ctr", Path(_SOCK))
