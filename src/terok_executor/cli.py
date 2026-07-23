# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-executor.

Composes executor's own commands with sandbox's full tree (via
[`SANDBOX_TREE`][terok_executor.credentials.vault_commands.SANDBOX_TREE])
into a single
[`CommandTree`][terok_util.cli_types.CommandTree], exposed two ways:

- *Deep path* — ``terok-executor sandbox <verb>`` reaches every
  sandbox verb verbatim, with executor's overlays applied where they
  exist (e.g. vault).
- *Shortcuts* — ``terok-executor vault <verb>`` resolves to the same
  ``CommandDef`` instance as ``terok-executor sandbox vault <verb>``,
  so wraps applied at one entry point apply at the other.

[`CommandTree.wire`][terok_util.cli_types.CommandTree.wire] /
[`CommandTree.dispatch`][terok_util.cli_types.CommandTree.dispatch]
do all the argparse plumbing.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version as _meta_version

from terok_util import CommandTree

from . import _ensure_bootstrapped
from ._tree import COMMANDS

try:
    __version__ = _meta_version("terok-executor")
except PackageNotFoundError:
    __version__ = "0.0.0"


#: Spelling under which sandbox setup is reachable in this front-end;
#: declared at CLI entry so sandbox-composed re-run hints name a verb
#: this CLI actually exposes (protocol: env var name, shared by string).
_SETUP_INVOCATION_ENV = "TEROK_SETUP_INVOCATION"
_SETUP_INVOCATION = "terok-executor setup"


def main(argv: list[str] | None = None) -> None:
    """Run the terok-executor CLI.

    Top-level options must precede the subcommand
    (``terok-executor --config /path run claude .``) — standard argparse
    subparser convention, matching the placement used by ``docker`` and
    ``kubectl``.

    *argv* (default ``sys.argv[1:]``) is threaded into
    [`CommandTree.wire`][terok_util.cli_types.CommandTree.wire] so only
    the invoked verb's module is imported: ``terok-executor run …``
    pulls the run stack, not the whole ``terok_sandbox`` command tree.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # One-time unified logging: routes every getLogger(__name__) to journald
    # (when present) or stderr.  The ``acp`` daemon reconfigures with its own
    # identifier/format when dispatched (configure is idempotent).
    from terok_util import configure

    configure(identifier="terok-executor")
    os.environ.setdefault(_SETUP_INVOCATION_ENV, _SETUP_INVOCATION)
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
    COMMANDS.wire(parser, argv=raw_argv)

    args = parser.parse_args(raw_argv)

    # Honour --config / --raw before dispatch so the very first
    # ``SandboxConfig()`` constructed by any handler sees the override.
    # SandboxConfig reads via ``terok_sandbox.paths._config_file_paths``,
    # which consults ``TEROK_CONFIG_FILE`` first.
    if args.raw:
        os.environ["TEROK_CONFIG_FILE"] = os.devnull
    elif args.config is not None:
        os.environ["TEROK_CONFIG_FILE"] = args.config

    if hasattr(args, "_cmd"):
        # Populate the agent registry once before any handler runs — the
        # package defers this out of import so ``--version`` / ``--help``
        # (which exit above) never pay the roster YAML load.
        _ensure_bootstrapped()
        from terok_executor.integrations.sandbox import NoPassphraseError

        try:
            CommandTree.dispatch(args)
        except NoPassphraseError as exc:
            # sandbox#278 stripped CLI-hint text from the library raise
            # sites so they stay diagnostic-only.  Standalone runs have no
            # orchestrator to add the remediation, so this front-end is
            # the operator-facing surface (mirrors terok's own wrapper).
            print(
                f"error: {exc}\n"
                "hint:  run `terok-executor vault unlock` to provision the"
                " vault passphrase for this session.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
    elif hasattr(args, "_group_help"):
        args._group_help.print_help()
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
