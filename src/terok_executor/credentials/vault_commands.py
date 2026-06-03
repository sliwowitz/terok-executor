# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Executor-level vault helpers: route generation + credential-leak scan.

The vault is served per container: the supervisor spawns on container
start via the terok-sandbox OCI hook and reads the per-container
sidecar to bind its proxy.  Sandbox owns the ``unlock`` / ``lock`` /
``passphrase`` verbs (passphrase-tier CRUD on the DB).

What lives here:

- ``routes`` — regenerate ``routes.json`` from the YAML agent roster.
- ``clean`` — remove leaked credential files from shared config mounts.
- ``scan_leaked_credentials`` / ``_is_injected_credentials_file`` /
  ``_is_injected_codex_auth_file`` — primitives the scan + clean
  verbs share.

Both verbs operate on host-side files only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from terok_util import CommandDef, CommandTree

if TYPE_CHECKING:
    from terok_executor.integrations.sandbox import SandboxConfig


def _ensure_routes(cfg: SandboxConfig | None = None) -> Path:
    """Generate routes.json from the YAML agent roster."""
    from terok_executor.roster import AgentRoster

    return AgentRoster.shared().ensure_vault_routes(cfg=cfg)


def _is_injected_credentials_file(path: Path) -> bool:
    """Check whether *path* is a terok-injected ``.credentials.json``.

    Returns ``True`` only when **all** of these hold:

    - ``claudeAiOauth.accessToken`` equals `PHANTOM_CREDENTIALS_MARKER`
    - ``claudeAiOauth.refreshToken`` is empty or absent

    Any parse error or unexpected structure → ``False`` (treat as real leak).
    """
    from terok_executor.integrations.sandbox import PHANTOM_CREDENTIALS_MARKER

    from .vendor_files import RawClaudeCredentialsFile, load_vendor_json

    try:
        cred = load_vendor_json(RawClaudeCredentialsFile, path)
    except ValueError:
        return False
    if cred is None or cred.claudeAiOauth is None:
        return False
    oauth = cred.claudeAiOauth
    return oauth.accessToken == PHANTOM_CREDENTIALS_MARKER and not oauth.refreshToken


def _is_injected_codex_auth_file(path: Path) -> bool:
    """Check whether *path* is a terok-injected shared Codex ``auth.json``."""
    from terok_executor.integrations.sandbox import CODEX_SHARED_OAUTH_MARKER

    from .vendor_files import RawCodexAuthFile, load_vendor_json

    try:
        cred = load_vendor_json(RawCodexAuthFile, path)
    except ValueError:
        return False
    if cred is None or cred.tokens is None:
        return False
    tokens = cred.tokens
    return (
        tokens.access_token == CODEX_SHARED_OAUTH_MARKER
        and tokens.refresh_token == CODEX_SHARED_OAUTH_MARKER
        and not cred.OPENAI_API_KEY
    )


def scan_leaked_credentials(mounts_base: Path) -> list[tuple[str, Path]]:
    """Return ``(provider, host_path)`` for credential files found in shared mounts.

    When the vault is active, real secrets should only live in the
    vault's sqlite DB — not in the shared config directories that get mounted
    into containers.  This function checks each routed provider's mount for
    credential files that would leak real tokens alongside phantom ones.

    Files injected by `_write_claude_credentials_file`
    are recognised by their dummy ``accessToken`` marker and skipped.

    Symlinks are rejected to prevent a container from tricking the scan into
    reading arbitrary host files via a crafted symlink in the shared mount.
    """
    import stat

    from terok_executor.roster import AgentRoster

    roster = AgentRoster.shared()
    base_resolved = mounts_base.resolve(strict=False)
    leaked: list[tuple[str, Path]] = []
    for name, route in roster.vault_routes.items():
        if not route.credential_file:
            continue
        auth = roster.auth_providers.get(name)
        if not auth:
            continue
        try:
            path = mounts_base / auth.host_dir_name / route.credential_file
            # lstat: do not follow symlinks — reject them outright
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            # Ensure resolved path stays within the mounts base
            if base_resolved not in path.resolve(strict=True).parents:
                continue
            if st.st_size > 0 and not (
                _is_injected_credentials_file(path) or _is_injected_codex_auth_file(path)
            ):
                leaked.append((name, path))
        except (OSError, TypeError) as exc:
            # Silently skipping turns a real leak into a no-result: the
            # operator would believe the scan was clean.  Surface a
            # warning so it's obvious which provider was not checked
            # and why; the loop continues so other providers still get
            # scanned.
            print(
                f"Warning [vault]: credential leak scan skipped {name!r}: {exc}",
                file=sys.stderr,
            )
            continue
    return leaked


def _handle_routes(*, cfg: SandboxConfig | None = None) -> None:
    """Regenerate routes.json from the YAML agent roster."""
    path = _ensure_routes(cfg=cfg)
    if path:
        print(f"Routes written to {path}")


def _handle_clean(*, cfg: SandboxConfig | None = None) -> None:  # noqa: ARG001
    """Remove leaked credential files from shared config mounts."""
    from terok_executor.paths import mounts_dir

    leaked = scan_leaked_credentials(mounts_dir())
    if not leaked:
        print("No leaked credential files found.")
        return
    for provider, path in leaked:
        path.unlink()
        print(f"Removed {provider}: {path}")


def _build_sandbox_tree() -> CommandTree:
    """Extend sandbox's full command tree with executor-only vault verbs.

    Sandbox owns the verb set (``vault unlock`` / ``vault lock`` /
    ``vault passphrase {seal,to-keyring,reveal,acknowledge,destroy}``)
    plus argparse schema.  Executor adds two file-level verbs that
    don't make sense in sandbox itself because they depend on the
    executor's YAML roster + mounts layout: ``vault routes`` and
    ``vault clean``.

    No overlays: every sandbox verb flows through unchanged.  The vault
    is served per container; executor's vault surface is pure extension.
    """
    from terok_executor.integrations.sandbox import COMMANDS as SANDBOX_COMMANDS

    return SANDBOX_COMMANDS.extend_at(
        ("vault",),
        (
            CommandDef(
                name="routes",
                help="Regenerate routes.json from YAML roster",
                handler=_handle_routes,
            ),
            CommandDef(
                name="clean",
                help="Remove leaked credential files from shared mounts",
                handler=_handle_clean,
            ),
        ),
    )


#: Sandbox's full command tree with executor's vault extensions applied.
#: Wired in two positions of executor's CLI: under ``terok-executor sandbox``
#: (the full deep path) and a top-level ``terok-executor vault`` shortcut.
SANDBOX_TREE: CommandTree = _build_sandbox_tree()


#: Sandbox's vault group with executor's extensions applied — a 1-tuple
#: containing the modified vault ``CommandDef``.  Surfaced at executor's
#: top level as the ``terok-executor vault …`` shortcut; the same
#: ``CommandDef`` instance also reaches ``terok-executor sandbox vault …``
#: via [`SANDBOX_TREE`][terok_executor.credentials.vault_commands.SANDBOX_TREE].
VAULT_COMMANDS: tuple[CommandDef, ...] = (SANDBOX_TREE.find_at(("vault",)),)
