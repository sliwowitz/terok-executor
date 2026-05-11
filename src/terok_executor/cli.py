# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-executor.

Built from the command registry in [`terok_executor.commands`][terok_executor.commands].
No command logic lives here — just argument wiring and dispatch.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as _meta_version
from typing import Any

from .commands import COMMANDS, ArgDef, CommandDef
from .credentials.vault_commands import VAULT_COMMANDS

try:
    __version__ = _meta_version("terok-executor")
except PackageNotFoundError:
    __version__ = "0.0.0"


# ── Public entry point ──────────────────────────────────────────────────


def main() -> None:
    """Run the terok-executor CLI."""
    parser = argparse.ArgumentParser(
        prog="terok-executor",
        description="Single-agent task runner for hardened Podman containers",
    )
    parser.add_argument("--version", action="version", version=f"terok-executor {__version__}")
    sub = parser.add_subparsers()

    for cmd in COMMANDS:
        _wire_command(sub, cmd)

    # -- vault --
    vault_p = sub.add_parser("vault", help="Vault management")
    vault_sub = vault_p.add_subparsers()
    for vault_cmd in VAULT_COMMANDS:
        _wire_command(vault_sub, vault_cmd)
    vault_p.set_defaults(_group_help=vault_p)

    args = parser.parse_args()
    if hasattr(args, "_cmd"):
        _dispatch(args)
    elif hasattr(args, "_group_help"):
        args._group_help.print_help()
    else:
        parser.print_help()
        raise SystemExit(1)


# ── Private helpers ─────────────────────────────────────────────────────


def _wire_command(sub: argparse._SubParsersAction, cmd: Any) -> None:
    """Add a [`CommandDef`][terok_executor.cli.CommandDef] to an argparse subparser group.

    ``cmd`` is intentionally ``Any``: this wires both
    :data:`terok_executor.COMMANDS` (`terok_executor.commands.CommandDef`)
    and :data:`VAULT_COMMANDS` (`terok_sandbox.commands.CommandDef`); the
    two have identical structural shapes but mypy treats them as distinct
    nominal types.  The Protocol surface lives in the duck-typed reads
    below (``cmd.name`` / ``cmd.args`` / ``arg.dest`` / …).
    """
    p = sub.add_parser(cmd.name, help=cmd.help)
    for arg in cmd.args:
        kwargs: dict = {}
        if arg.help:
            kwargs["help"] = arg.help
        if arg.type is not None:
            kwargs["type"] = arg.type
        if arg.default is not None:
            kwargs["default"] = arg.default
        if arg.action is not None:
            kwargs["action"] = arg.action
        if arg.dest is not None:
            kwargs["dest"] = arg.dest
        if arg.nargs is not None:
            kwargs["nargs"] = arg.nargs
        p.add_argument(arg.name, **kwargs)
    p.set_defaults(_cmd=cmd)


def _arg_key(arg: ArgDef) -> str:
    """Derive the kwarg name for an argument definition."""
    return arg.dest or arg.name.lstrip("-").replace("-", "_")


def _dispatch(args: argparse.Namespace) -> None:
    """Extract handler kwargs from parsed args and call the handler."""
    cmd: CommandDef = args._cmd
    if cmd.handler is None:
        raise SystemExit(f"Command '{cmd.name}' has no handler")
    kwargs = {_arg_key(arg): getattr(args, _arg_key(arg), arg.default) for arg in cmd.args}
    cmd.handler(**kwargs)


if __name__ == "__main__":
    main()
