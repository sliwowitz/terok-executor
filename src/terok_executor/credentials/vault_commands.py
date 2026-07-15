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
- ``scan_leaked_credentials`` + the ``_BENIGN_CREDENTIAL_CHECKS``
  registry of per-provider recognizers for credential files that are
  legitimately non-empty — primitives the scan + clean verbs share.

Both verbs operate on host-side files only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
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


def _is_tokenless_glab_config_file(path: Path) -> bool:
    """Check whether *path* is a glab ``config.yml`` that carries no token.

    glab's ``config.yml`` doubles as a settings file the tool rewrites on
    startup (update-check metadata, ``git_protocol``, …), so its shared
    mount is writable by design and a non-empty file is the expected
    steady state — only a ``hosts.<host>.token`` entry makes it a leak.

    Any parse error or unexpected structure → ``False`` (treat as real leak).
    """
    from .vendor_files import RawGlabConfigFile, load_vendor_yaml

    try:
        cred = load_vendor_yaml(RawGlabConfigFile, path)
    except ValueError:
        return False
    if cred is None:
        return False
    return not any(block.token for block in cred.hosts.values())


_BENIGN_CREDENTIAL_CHECKS: dict[str, Callable[[Path], bool]] = {
    "claude": _is_injected_credentials_file,
    "codex": _is_injected_codex_auth_file,
    "glab": _is_tokenless_glab_config_file,
}
"""Per-provider recognizers for credential files that are legitimately non-empty.

Maps the mount's owning agent name to a predicate that returns ``True``
when the file holds no real secret: a terok-injected phantom (claude,
codex) or a settings-only vendor file (glab).  Providers without an
entry treat any non-empty credential file as a leak.
"""


def scan_leaked_credentials(mounts_base: Path) -> list[tuple[str, Path]]:
    """Return ``(provider, host_path)`` for credential files found in shared mounts.

    When the vault is active, real secrets should only live in the
    vault's sqlite DB — not in the shared config directories that get mounted
    into containers.  This function checks each routed provider's mount for
    credential files that would leak real tokens alongside phantom ones.

    Non-empty files recognised as benign by the provider's
    `_BENIGN_CREDENTIAL_CHECKS` entry — terok-injected phantoms, or
    glab's settings-only ``config.yml`` — are skipped.

    A credential file that does not exist is a definitive clean result,
    not a skipped check: an agent that ships in the image but was never
    authenticated on this host has nothing to leak.  Only genuine read
    failures (permissions, I/O errors) warrant the skip warning.

    Symlinks are rejected to prevent a container from tricking the scan into
    reading arbitrary host files via a crafted symlink in the shared mount.
    """
    import stat

    from terok_executor.roster import AgentRoster

    roster = AgentRoster.shared()
    base_resolved = mounts_base.resolve(strict=False)
    leaked: list[tuple[str, Path]] = []
    # Iterate the shared mounts directly: each MountDef already pairs the
    # agent's auth dir with the credential file its binding declares, so we
    # don't have to re-join provider-keyed routes against agent-keyed auth
    # providers.  ``mount.provider`` is the owning agent name (the label the
    # operator sees); mounts without a credential file (opencode state dirs,
    # explicit ``mounts:`` blocks) carry an empty string and are skipped.
    for mount in roster.mounts:
        if not mount.credential_file:
            continue
        try:
            path = mounts_base / mount.host_dir / mount.credential_file
            # lstat: do not follow symlinks — reject them outright
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            # Ensure resolved path stays within the mounts base
            if base_resolved not in path.resolve(strict=True).parents:
                continue
            is_benign = _BENIGN_CREDENTIAL_CHECKS.get(mount.provider)
            if st.st_size > 0 and not (is_benign and is_benign(path)):
                leaked.append((mount.provider, path))
        except FileNotFoundError:
            # No credential file at all — the agent was never authenticated
            # on this host, so there is nothing to scan and nothing to leak.
            continue
        except (OSError, TypeError) as exc:
            # Silently skipping turns a real leak into a no-result: the
            # operator would believe the scan was clean.  Surface a
            # warning so it's obvious which mount was not checked and why;
            # the loop continues so other mounts still get scanned.
            print(
                f"Warning [vault]: credential leak scan skipped {mount.provider!r}: {exc}",
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


#: The ``sandbox`` deep-path group — the lazy ``source`` target the
#: top-level tree ([`_tree`][terok_executor._tree]) references so that
#: building the tree doesn't import this module (and the whole
#: ``terok_sandbox`` command tree it pulls).  Resolving it — i.e. running
#: ``terok-executor sandbox …`` — is what loads the sandbox tree.
SANDBOX_GROUP: CommandDef = CommandDef(
    name="sandbox",
    help="Sandbox subsystem (full deep tree — same verbs as terok-sandbox)",
    children=SANDBOX_TREE.roots,
)


#: Sandbox's vault group with executor's extensions applied — the lazy
#: ``source`` target for the top-level ``terok-executor vault …``
#: shortcut.  The same ``CommandDef`` instance also reaches
#: ``terok-executor sandbox vault …`` via
#: [`SANDBOX_GROUP`][terok_executor.credentials.vault_commands.SANDBOX_GROUP]'s
#: children, so a wrap applied at one path applies at the other.
VAULT_GROUP: CommandDef = SANDBOX_TREE.find_at(("vault",))


#: Sandbox's vault group as a 1-tuple — retained for the terok adapter /
#: tests that consume the executor vault surface directly.
VAULT_COMMANDS: tuple[CommandDef, ...] = (VAULT_GROUP,)
