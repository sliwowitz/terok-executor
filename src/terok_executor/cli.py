# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-executor.

Composes executor's own commands with sandbox's full tree (via
[`SANDBOX_TREE`][terok_executor.credentials.vault_commands.SANDBOX_TREE])
into a single
[`CommandTree`][terok_sandbox.commands.CommandTree], exposed two ways:

- *Deep path* — ``terok-executor sandbox <verb>`` reaches every
  sandbox verb verbatim, with executor's overlays applied where they
  exist (e.g. vault).
- *Shortcuts* — ``terok-executor vault <verb>`` resolves to the same
  ``CommandDef`` instance as ``terok-executor sandbox vault <verb>``,
  so wraps applied at one entry point apply at the other.

[`CommandTree.wire`][terok_sandbox.commands.CommandTree.wire] /
[`CommandTree.dispatch`][terok_sandbox.commands.CommandTree.dispatch]
do all the argparse plumbing.
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import PackageNotFoundError, version as _meta_version

from terok_sandbox.commands import CommandTree

from ._tree import COMMANDS

try:
    __version__ = _meta_version("terok-executor")
except PackageNotFoundError:
    __version__ = "0.0.0"


def main() -> None:
    """Run the terok-executor CLI.

    Top-level options must precede the subcommand
    (``terok-executor --config /path run claude .``) — standard argparse
    subparser convention, matching the placement used by ``docker`` and
    ``kubectl``.
    """
    parser = argparse.ArgumentParser(
        prog="terok-executor",
        description="Single-agent task runner for hardened Podman containers",
    )
    parser.add_argument("--version", action="version", version=f"terok-executor {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "Override the config.yml path (sets TEROK_CONFIG_FILE for this invocation). "
            "Bypasses the layered system/user lookup."
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Ignore any config.yml — use sandbox/executor dataclass defaults only. "
            "Equivalent to '--config /dev/null'."
        ),
    )
    COMMANDS.wire(parser)

    args = parser.parse_args()

    # Honour --config / --raw before dispatch so the very first
    # ``SandboxConfig()`` constructed by any handler sees the override.
    # SandboxConfig reads via ``terok_sandbox.paths._config_file_paths``,
    # which consults ``TEROK_CONFIG_FILE`` first.
    if args.raw:
        os.environ["TEROK_CONFIG_FILE"] = os.devnull
    elif args.config is not None:
        os.environ["TEROK_CONFIG_FILE"] = args.config

    if hasattr(args, "_cmd"):
        CommandTree.dispatch(args)
    elif hasattr(args, "_group_help"):
        args._group_help.print_help()
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
