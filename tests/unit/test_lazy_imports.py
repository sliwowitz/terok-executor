# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Guards the import-laziness contract of the ``terok_executor`` barrel.

The host-side ACP names must resolve without dragging in the ``acp``
protocol library (a ~60 ms pydantic schema build): terok's executor
adapter imports them eagerly, so a regression here would tax every
terok CLI startup.  The check runs in a fresh interpreter because
``acp`` is very likely already resident in the test process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_acp_host_names_import_without_loading_acp() -> None:
    """``acp`` stays out of ``sys.modules`` after importing the ACP host names."""
    script = textwrap.dedent(
        """
        import sys

        from terok_executor import acp_socket_is_live, list_authenticated_agents

        assert callable(acp_socket_is_live)
        assert callable(list_authenticated_agents)
        leaked = sorted(m for m in sys.modules if m == "acp" or m.startswith("acp."))
        assert not leaked, f"acp eagerly imported: {leaked}"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bare_import_loads_neither_acp_nor_sandbox() -> None:
    """A bare ``import terok_executor`` touches neither ``acp`` nor ``terok_sandbox``."""
    script = textwrap.dedent(
        """
        import sys

        import terok_executor  # noqa: F401

        heavy = sorted(
            m
            for m in sys.modules
            if m == "acp" or m.startswith("acp.") or m == "terok_sandbox" or m.startswith("terok_sandbox.")
        )
        assert not heavy, f"bare import pulled heavy deps: {heavy}"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
