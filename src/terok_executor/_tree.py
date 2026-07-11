# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Composes executor's full [`CommandTree`][terok_util.cli_types.CommandTree].

Lives below the CLI surface so the package init can re-export the
composed tree as ``terok_executor.COMMANDS`` (the cli module is at
the top of the dependency graph; nothing below it may import it).

Every top-level verb is a **lazy root** — a
[`CommandDef`][terok_util.cli_types.CommandDef] carrying only
``name`` + ``help`` (enough for the top-level ``--help`` listing) and a
``source`` that resolves to the fully-populated verb definition in its
own module.  Building this tree therefore imports *none* of the verb
modules: neither [`commands`][terok_executor.commands] (which pulls the
container/build stack) nor
[`credentials.vault_commands`][terok_executor.credentials.vault_commands]
(which pulls the whole ``terok_sandbox`` command tree).
[`CommandTree.wire`][terok_util.cli_types.CommandTree.wire] with
``argv=`` resolves only the invoked verb, so ``terok-executor <verb>``
imports the one module that verb needs.

Three views over one underlying ``SANDBOX_TREE`` instance survive the
laziness (both ``sandbox`` and ``vault`` sources reference it, so the
resolved nodes still share identity):

- ``terok-executor <own-verb>``        — executor's verbs (run, auth, …)
- ``terok-executor sandbox <verb>``    — full sandbox tree, deep path
- ``terok-executor vault <verb>``      — shortcut sharing identity with
  the corresponding subtree under ``sandbox``
"""

from __future__ import annotations

from terok_util import CommandDef, CommandTree

#: Verb-module dotted paths for the lazy ``source`` references below.
_OWN = "terok_executor.commands"
_VAULT = "terok_executor.credentials.vault_commands"

#: Executor's top-level command tree — lazy roots only.  See module docstring.
COMMANDS: CommandTree = CommandTree(
    (
        CommandDef(
            "run",
            "Run an agent in a hardened container",
            source=f"{_OWN}:RUN_COMMAND",
        ),
        CommandDef(
            "run-tool",
            "Run a tool in a sidecar container (separate L1, real API key)",
            source=f"{_OWN}:RUN_TOOL_COMMAND",
        ),
        CommandDef(
            "auth",
            "Authenticate an agent",
            source=f"{_OWN}:AUTH_COMMAND",
        ),
        CommandDef(
            "agents",
            "Inspect the agent roster and set the build-time default selection",
            source=f"{_OWN}:AGENTS_COMMAND",
        ),
        CommandDef(
            "build",
            "Build L0+L1 container images",
            source=f"{_OWN}:BUILD_COMMAND",
        ),
        CommandDef(
            "setup",
            "Install sandbox services + container images (first-run bootstrap)",
            source=f"{_OWN}:SETUP_COMMAND",
        ),
        CommandDef(
            "uninstall",
            "Remove sandbox services + container images (mirror of setup)",
            source=f"{_OWN}:UNINSTALL_COMMAND",
        ),
        CommandDef(
            "list",
            "List containers",
            source=f"{_OWN}:LIST_COMMAND",
        ),
        CommandDef(
            "start",
            "Start a stopped container",
            source=f"{_OWN}:START_COMMAND",
        ),
        CommandDef(
            "stop",
            "Stop a container (kept for a later start)",
            source=f"{_OWN}:STOP_COMMAND",
        ),
        CommandDef(
            "rm",
            "Remove a container and its host-side state",
            source=f"{_OWN}:RM_COMMAND",
        ),
        CommandDef(
            "show-config",
            "Print the effective SandboxConfig (diffable against higher-layer orchestrators)",
            source=f"{_OWN}:SHOW_CONFIG_COMMAND",
        ),
        CommandDef(
            "acp",
            "Run the per-container ACP host-proxy daemon",
            source=f"{_OWN}:ACP_COMMAND",
        ),
        CommandDef(
            "sandbox",
            "Sandbox subsystem (full deep tree — same verbs as terok-sandbox)",
            source=f"{_VAULT}:SANDBOX_GROUP",
        ),
        CommandDef(
            "vault",
            "Vault credential lifecycle (shortcut for 'sandbox vault')",
            source=f"{_VAULT}:VAULT_GROUP",
        ),
    )
)


__all__ = ["COMMANDS"]
