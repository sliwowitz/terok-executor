# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Composes executor's full [`CommandTree`][terok_sandbox.commands.CommandTree].

Lives below the CLI surface so the package init can re-export the
composed tree as ``terok_executor.COMMANDS`` (the cli module is at
the top of the dependency graph; nothing below it may import it).

Three views over one underlying ``SANDBOX_TREE`` instance:

- ``terok-executor <own-verb>``        — executor's verbs (run, auth, …)
- ``terok-executor sandbox <verb>``    — full sandbox tree, deep path
- ``terok-executor vault <verb>``      — shortcut sharing identity with
  the corresponding subtree under ``sandbox``
"""

from __future__ import annotations

from terok_sandbox.commands import CommandDef, CommandTree

from .commands import COMMANDS as OWN_COMMANDS
from .credentials.vault_commands import SANDBOX_TREE, VAULT_COMMANDS

#: Executor's top-level command tree.  See module docstring.
COMMANDS: CommandTree = CommandTree(
    OWN_COMMANDS
    + (
        CommandDef(
            name="sandbox",
            help="Sandbox subsystem (full deep tree — same verbs as terok-sandbox)",
            children=SANDBOX_TREE.roots,
        ),
    )
    + VAULT_COMMANDS
)


__all__ = ["COMMANDS"]
