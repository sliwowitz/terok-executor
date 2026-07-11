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

import json
import subprocess
import sys
import textwrap

#: Verb modules that must stay unimported unless their verb is invoked.
_VERB_MODULES = [
    "terok_executor.commands",
    "terok_executor.credentials.vault_commands",
    "terok_sandbox.commands",
]

#: Every top-level verb the CLI advertises, for the ``--help`` listing check.
_ALL_VERBS = [
    "run",
    "run-tool",
    "auth",
    "agents",
    "build",
    "setup",
    "uninstall",
    "list",
    "start",
    "stop",
    "rm",
    "show-config",
    "acp",
    "sandbox",
    "vault",
]


def _cli_probe(args: list[str]) -> dict[str, object]:
    """Run ``cli.main(args)`` in a fresh interpreter; report imports + help text.

    Returns ``{"present": {module: bool}, "out": <captured stdout>}``.  A
    subprocess is required because the test process already has the verb
    modules resident.
    """
    script = textwrap.dedent(
        f"""
        import io, json, sys
        from contextlib import redirect_stdout

        import terok_executor.cli as cli

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                cli.main({args!r})
        except SystemExit:
            pass
        present = {{m: (m in sys.modules) for m in {_VERB_MODULES!r}}}
        print("PROBE" + json.dumps({{"present": present, "out": buf.getvalue()}}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("PROBE"))
    return json.loads(line[len("PROBE") :])


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


def test_run_verb_imports_only_its_own_stack() -> None:
    """``terok-executor run --help`` resolves ``commands`` but not the vault/sandbox tree."""
    present = _cli_probe(["run", "--help"])["present"]
    assert present["terok_executor.commands"], "the run verb should resolve its own module"
    assert not present["terok_executor.credentials.vault_commands"], (
        "run must not import the vault/sandbox command tree"
    )
    assert not present["terok_sandbox.commands"], (
        "run must not import the terok_sandbox command tree"
    )


def test_top_level_help_lists_all_verbs_without_importing_them() -> None:
    """Top-level ``--help`` renders every verb from lazy placeholders — no verb module loads."""
    probe = _cli_probe(["--help"])
    present = probe["present"]
    assert not any(present.values()), f"--help must not import any verb module: {present}"
    out = str(probe["out"])
    for verb in _ALL_VERBS:
        assert verb in out, f"--help listing is missing verb {verb!r}"


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
