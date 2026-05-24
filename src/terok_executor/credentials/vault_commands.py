# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Manages the vault daemon lifecycle via CLI subcommands.

Wraps terok-sandbox vault lifecycle with agent-level concerns: route
generation from the YAML roster is performed before ``start`` and
``install`` so the vault always has up-to-date provider config.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from terok_util import CommandDef, CommandTree

if TYPE_CHECKING:
    from terok_executor.integrations.sandbox import SandboxConfig


def _ensure_routes(cfg: SandboxConfig | None = None) -> Path:
    """Generate routes.json from the YAML agent roster."""
    from terok_executor.roster import AgentRoster

    return AgentRoster.shared().ensure_vault_routes(cfg=cfg)


def _handle_start(*, cfg: SandboxConfig | None = None) -> None:
    """Generate routes and start the vault daemon."""
    from terok_executor.integrations.sandbox import VaultManager

    vault = VaultManager(cfg)
    if vault.is_daemon_running():
        print("Vault is already running.")
        sys.exit(1)
    _ensure_routes(cfg=cfg)
    vault.start_daemon()
    print("Vault started.")


def _handle_stop(*, cfg: SandboxConfig | None = None) -> None:
    """Stop the vault daemon."""
    from terok_executor.integrations.sandbox import VaultManager

    vault = VaultManager(cfg)
    if not vault.is_daemon_running():
        print("Vault is not running.")
        return
    vault.stop_daemon()
    print("Vault stopped.")


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


def _format_credentials(status: object, cfg: SandboxConfig | None = None) -> str:
    """Format stored credentials as ``name (type), ...`` for status display.

    *cfg* threads the caller's chosen passphrase-resolution knobs
    (session-unlock file, keyring opt-in, config-file fallback) into
    the DB open — without this hook the function would construct a
    fresh ``SandboxConfig()`` and silently miss any non-default tier
    the caller had set up.  Defaults to a fresh config for direct
    callers that don't have one handy.
    """
    from terok_executor.integrations.sandbox import SandboxConfig as _SandboxConfig, VaultStatus

    if cfg is None:
        cfg = _SandboxConfig()
    st: VaultStatus = status  # type: ignore[assignment]
    if not st.credentials_stored:
        return "none stored"
    try:
        # ``st.db_path`` is the running daemon's actual DB — may diverge
        # from ``cfg.db_path`` under test fixtures or multi-instance
        # hosts.  Pass it explicitly; *cfg* still owns the tier policy
        # so this caller never has to know about session-file /
        # systemd-creds / keyring / config.
        db = cfg.open_credential_db(st.db_path)
        try:
            parts = []
            for name in st.credentials_stored:
                cred = db.load_credential("default", name)
                ctype = cred.get("type", "unknown") if cred else "unknown"
                parts.append(f"{name} ({ctype})")
        finally:
            db.close()
        return ", ".join(parts)
    except Exception:  # noqa: BLE001
        print(
            "Warning [vault]: credential type lookup failed; showing names only",
            file=sys.stderr,
        )
        return ", ".join(st.credentials_stored)


def _handle_status(*, cfg: SandboxConfig | None = None) -> None:
    """Show vault status."""
    from terok_util import sanitize_tty

    from terok_executor.integrations.sandbox import (
        SandboxConfig as _SandboxConfig,
        VaultManager,
        systemd_creds_has_tpm2,
    )
    from terok_executor.paths import mounts_dir

    # Keep a concrete cfg in hand so the socket-mode branch can surface
    # the SSH-signer socket path alongside the broker socket without a
    # second sandbox round-trip; ``VaultManager.get_status`` does its own
    # resolution internally and is unaffected.
    working_cfg = cfg or _SandboxConfig()
    vault = VaultManager(working_cfg)
    status = vault.get_status()
    state = "running" if status.running else "stopped"
    # Path fields land in a terminal that interprets ANSI/control chars.
    # The values originate from the vault daemon's config / live state —
    # most installs trust those, but a malformed config layer or a
    # manipulated systemd unit could plant control sequences that would
    # otherwise spoof prompts or rewrite the screen.  Sanitise at the
    # render boundary; matches what ``terok-sandbox vault status`` does.
    #
    # Two orthogonal facts about the vault, each on its own line:
    # ``Activation:`` is the lifecycle channel (``systemd`` / ``daemon`` /
    # ``none``) — how the vault is *managed*.  ``Transport:`` is the wire
    # protocol containers use (``tcp`` / ``socket``) — how clients *reach*
    # it.  The historical single-line ``Mode:`` overloaded both axes onto
    # one label, which read as "transport" for almost every user and made
    # the TCP-mode rendering misleading (no port surfaced, only the
    # fast-path Unix probe — see commentary on the ``Socket:`` line
    # below).
    print(f"Activation:  {sanitize_tty(status.mode)}")
    # Tight allow-list — keeps an unexpected value (or a ``MagicMock`` in
    # tests that haven't bothered to set ``transport``) from leaking a
    # raw repr into operator-facing output.
    transport = status.transport if status.transport in ("tcp", "socket") else None
    print(f"Transport:   {transport or '(not configured)'}")
    print(f"Status:      {state}")
    # TCP mode: surface the actual listeners that container traffic
    # rides.  In TCP mode the daemon still binds ``vault.sock`` as a
    # local fast-path probe target (host-side ``ensure_reachable``
    # tries it before the TCP probe), but no container or external
    # client ever reaches that socket — flag it so the line doesn't
    # imply otherwise.
    if transport == "tcp":
        broker_port = vault.token_broker_port
        signer_port = vault.ssh_signer_port
        if broker_port is not None:
            print(f"TCP broker:  127.0.0.1:{broker_port}")
        if signer_port is not None:
            print(f"TCP signer:  127.0.0.1:{signer_port}")
        socket_annotation = "  (fast-path probe only)"
    else:
        socket_annotation = ""
    print(f"Socket:      {sanitize_tty(str(status.socket_path))}{socket_annotation}")
    # Socket-mode: the SSH signer rides a sibling Unix socket that
    # containers connect to directly via the per-task bind mount.
    # Surface it so the rendering is symmetric with the broker
    # ``Socket:`` line (both transports, both signer endpoints visible).
    if transport == "socket":
        print(f"SSH signer:  {sanitize_tty(str(working_cfg.ssh_signer_socket_path))}")
    print(f"DB:          {sanitize_tty(str(status.db_path))}")
    print(
        f"Routes:      {sanitize_tty(str(status.routes_path))}"
        f" ({status.routes_configured} configured)"
    )
    print(f"SSH keys:    {status.ssh_keys_stored}")
    # ``Locked:`` is the operator-facing question — the chain tier
    # ``Passphrase:`` answers WHICH tier resolved it; the explicit
    # ``Locked:`` line answers WHETHER it resolved at all so a
    # ``grep '^Locked:'`` is enough for scripts.
    print(f"Locked:      {'yes' if status.locked else 'no'}")
    if status.locked:
        print("Passphrase:  (no tier resolved — run `terok vault unlock`)")
    elif status.passphrase_source is not None:
        tier_line = f"Passphrase:  resolved via {status.passphrase_source}"
        # Surface the TPM2 binding when systemd-creds is the resolved
        # tier and the host actually has a TPM2 device — operators
        # using systemd-creds explicitly opted into the higher-end
        # sealing mode; showing the (+TPM2) annotation makes the
        # binding visible without needing to read ``systemd-creds
        # list`` manually.
        if status.passphrase_source == "systemd-creds":
            # Best-effort probe — a missing or hanging ``systemd-creds``
            # binary must not break ``vault status``.  ``suppress`` is
            # the explicit "intentional swallow" so static analysers
            # don't flag the bare pass.
            with suppress(Exception):
                if systemd_creds_has_tpm2():
                    tier_line = f"{tier_line} (+TPM2)"
        print(tier_line)
    if status.credentials_stored:
        print(f"Credentials: {_format_credentials(status, cfg)}")
    else:
        print("Credentials: none stored")
    if not status.running and status.mode == "none" and vault.is_systemd_available():
        print("\nHint: run 'install' to set up systemd socket activation.")

    plaintext_path = getattr(status, "plaintext_passphrase_path", None)
    if plaintext_path is not None:
        _print_plaintext_passphrase_warning(plaintext_path)

    leaked = scan_leaked_credentials(mounts_dir())
    if leaked:
        print("\nWARNING: Real credentials found in shared config mounts:")
        for provider, path in leaked:
            print(f"  {provider}: {path}")
        print("These files are mounted into containers alongside vault phantom tokens.")
        print("Run 'clean' to remove them.")


def _print_plaintext_passphrase_warning(path: Path) -> None:
    """Stderr WARNING that the vault passphrase lives in plaintext on disk.

    Mirrors [`terok_sandbox.commands._print_plaintext_passphrase_warning`][terok_sandbox.commands._print_plaintext_passphrase_warning]
    so ``terok vault status`` (executor-wrapped) and ``terok-sandbox vault status``
    surface the same operator-actionable warning.  ``getattr`` with a
    ``None`` default lets the call site work against older
    ``VaultStatus`` builds that pre-date sandbox#282.
    """
    from terok_util import sanitize_tty

    use_color = sys.stderr.isatty()
    red = "\033[1;31m" if use_color else ""
    reset = "\033[0m" if use_color else ""
    safe_path = sanitize_tty(str(path))
    print(
        f"{red}WARNING: vault passphrase stored in plaintext at {safe_path}{reset}\n"
        f"{red}         accept on-disk plaintext as your trust boundary,"
        f" or migrate to keyring/systemd-creds.{reset}",
        file=sys.stderr,
    )


def _handle_install(*, cfg: SandboxConfig | None = None) -> None:
    """Generate routes and install systemd socket activation."""
    from terok_executor.integrations.sandbox import VaultManager

    vault = VaultManager(cfg)
    if not vault.is_systemd_available():
        print(
            "Error: systemd user services are not available on this host.\n"
            "Use 'start' to run the vault without systemd."
        )
        sys.exit(1)
    _ensure_routes(cfg=cfg)
    vault.install_systemd_units()
    print("Vault installed via systemd socket activation.")


def _handle_uninstall(*, cfg: SandboxConfig | None = None) -> None:
    """Remove vault systemd units."""
    from terok_executor.integrations.sandbox import VaultManager

    vault = VaultManager(cfg)
    if not vault.is_systemd_available():
        print("Error: systemd user services are not available. Nothing to uninstall.")
        sys.exit(1)
    vault.uninstall_systemd_units()
    print("Vault systemd units removed.")


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


#: Paths into sandbox's vault subtree where executor swaps in enriched
#: handlers — leaked-credential scans, ``_ensure_routes`` before
#: installs, typed credential listings, ``SystemExit`` on already-
#: running.  Anything not in this map (``unlock`` / ``lock`` /
#: ``passphrase {seal,to-keyring,destroy}``) flows through unchanged —
#: that's how new sandbox verbs reach ``terok vault`` zero-edit.
_VAULT_OVERRIDES: dict[tuple[str, ...], Callable[..., None]] = {
    ("vault", "start"): _handle_start,
    ("vault", "stop"): _handle_stop,
    ("vault", "status"): _handle_status,
    ("vault", "install"): _handle_install,
    ("vault", "uninstall"): _handle_uninstall,
}


def _build_sandbox_tree() -> CommandTree:
    """Apply executor's overlays + extensions to sandbox's full command tree.

    Sandbox owns the verb set + argparse schema (one source of truth
    for ``--key=``, help text, structural nesting of ``vault
    passphrase``).  Executor overlays its enriched vault handlers at
    the five paths in
    [`_VAULT_OVERRIDES`][terok_executor.credentials.vault_commands._VAULT_OVERRIDES]
    and extends the vault subtree with two executor-only verbs
    (``routes`` / ``clean``).
    Identity is preserved for every untouched node so a downstream
    shortcut that splices the same subtree (terok's ``vault`` at
    top-level) shares the wrap with the deep ``terok executor sandbox
    vault`` path automatically.
    """
    from terok_executor.integrations.sandbox import COMMANDS as SANDBOX_COMMANDS

    # Both ``SANDBOX_COMMANDS`` and the locally-built ``CommandDef`` /
    # ``CommandTree`` now share the same ``terok_util.cli_types``
    # identity (sandbox adopted terok-util in v0.0.124a16+), so the
    # earlier ``cast`` + ``type: ignore`` bridge across two structurally-
    # identical-but-nominally-distinct classes is no longer needed.
    return SANDBOX_COMMANDS.overlay(_VAULT_OVERRIDES).extend_at(
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


#: Sandbox's full command tree with executor's overlays + extensions applied.
#: Wired in two positions of executor's CLI: under ``terok-executor sandbox``
#: (the full deep path) and selected subtrees promoted to top level as
#: shortcuts.  Both positions share ``CommandDef`` identity so wraps
#: applied here propagate through every entry point that references the
#: same node.
SANDBOX_TREE: CommandTree = _build_sandbox_tree()


#: Sandbox's vault group with executor's overlays applied — a 1-tuple
#: containing the modified vault ``CommandDef``.  Surfaced at executor's
#: top level as the ``terok-executor vault …`` shortcut; the same
#: ``CommandDef`` instance also reaches ``terok-executor sandbox vault …``
#: via [`SANDBOX_TREE`][terok_executor.credentials.vault_commands.SANDBOX_TREE].
VAULT_COMMANDS: tuple[CommandDef, ...] = (SANDBOX_TREE.find_at(("vault",)),)
