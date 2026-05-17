# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Krun (KVM-microVM) host-side provisioning: identity + runtime factory.

Two responsibilities, both intentionally outside terok:

- [`ensure_krun_host_keypair`][terok_executor.krun.ensure_krun_host_keypair]
  — mint or load the SSH keypair the L0 guest's ``sshd`` trusts, and
  materialise both halves onto tmpfs files: the private half so
  ``ssh -i`` can read it, and the public half so the orchestrator can
  bind-mount it into the running guest at
  ``/etc/ssh/authorized_keys.d/terok``.  The vault is the system of
  record; the tmpfs cache is a derived view rebuilt per process.

- [`make_krun_runtime`][terok_executor.krun.make_krun_runtime] — one-shot
  constructor for a production
  [`KrunRuntime`][terok_sandbox.KrunRuntime] backed by the TCP-over-passt
  SSH transport, with the keypair already wired in.  terok flips its
  runtime selector to ``krun`` and calls this; everything else is
  invisible.

Living in executor (rather than terok) keeps the rule honest: anything
that owns the L0 build path also owns the trust material that makes
the resulting guest reachable.  Selecting krun without provisioning
the key is a contradiction, so the two operations belong together.
"""

from __future__ import annotations

import dataclasses
import os
import stat
import tempfile
from pathlib import Path

from terok_sandbox import (
    KrunRuntime,
    PodmanRuntime,
    SandboxConfig,
    TcpSSHTransport,
    ensure_infra_keypair,
    namespace_runtime_dir,
    podman_port_resolver,
)

# Names matching the L0 guest's ``/etc/ssh/authorized_keys.d/terok``
# bind-mount target.  Keep both halves co-located so a future rename
# touches one place.
_HOST_KEYPAIR_BASENAME = "krun_host"


@dataclasses.dataclass(frozen=True, slots=True)
class KrunHostKeypair:
    """Materialised view of the ``%host`` infrastructure keypair.

    Returned by [`ensure_krun_host_keypair`][terok_executor.krun.ensure_krun_host_keypair].
    Carries the tmpfs path to the OpenSSH-PEM private key (ready for
    ``ssh -i``) and the matching public-key file (ready to bind-mount
    into the krun guest at ``/etc/ssh/authorized_keys.d/terok``), so
    callers don't have to redo the DER→PEM conversion or re-derive
    the public line from raw blobs.
    """

    private_path: Path
    """tmpfs path holding the OpenSSH-PEM private key (``0600`` perms)."""

    public_path: Path
    """Sibling ``.pub`` file (``0644`` perms) carrying the public line."""

    public_line: str
    """Single-line OpenSSH public key (``ssh-ed25519 AAAA… comment``)."""

    fingerprint: str
    """Canonical ``SHA256:…`` fingerprint over the SSH wire-format blob."""

    created: bool
    """``True`` when this call minted the key; ``False`` when it was loaded."""


def ensure_krun_host_keypair(
    *,
    cfg: SandboxConfig | None = None,
    runtime_dir: Path | None = None,
) -> KrunHostKeypair:
    """Load (or mint, first call) the ``%host`` keypair and materialise it to tmpfs.

    The vault is the system of record: the keypair lives in the sandbox
    credential DB under the ``%host`` infrastructure scope.  This
    helper opens the DB, calls
    [`ensure_infra_keypair`][terok_sandbox.ensure_infra_keypair] (which
    generates the key on first call and reloads it thereafter), and
    writes the OpenSSH-PEM private + the public-key line into
    *runtime_dir* (default:
    [`namespace_runtime_dir()`][terok_sandbox.namespace_runtime_dir]).

    The orchestrator bind-mounts ``public_path`` into the running
    krun guest at ``/etc/ssh/authorized_keys.d/terok`` so the
    guest's sshd accepts our private key.  The L0 image itself ships
    an empty placeholder at that path; the bind-mount overlays it.

    Rotation = clear the ``%host`` scope in the vault, then re-run.
    Typically called per task launch under krun (idempotent — loads
    on subsequent calls).  New tasks pick up the new key; in-flight
    tasks keep what they had until they're stopped.

    Requires the vault to be unlocked — the krun runtime is gated on
    ``experimental: true`` upstream and assumes the operator has the
    vault open for the session.  A ``NoPassphraseError`` propagates
    unchanged so the orchestrator can render its own remediation hint.

    Args:
        cfg: Sandbox config used to open the credential DB.  ``None``
            means use the zero-arg default — appropriate for standalone
            executor flows; terok injects its own enriched config when
            calling.
        runtime_dir: Override for the tmpfs cache directory.  ``None``
            uses [`namespace_runtime_dir`][terok_sandbox.namespace_runtime_dir],
            with a hard refusal to fall back to persistent disk.
    """
    target_dir = _ensure_safe_runtime_dir(runtime_dir)
    private = target_dir / f"{_HOST_KEYPAIR_BASENAME}.key"
    public = target_dir / f"{_HOST_KEYPAIR_BASENAME}.key.pub"

    db = (cfg or SandboxConfig()).open_credential_db(prompt_on_tty=False)
    try:
        infra = ensure_infra_keypair("%host", db=db, comment="krun-host (terok)")
    finally:
        db.close()

    _write_atomic(private, infra.private_pem, mode=0o600)
    _write_atomic(public, (infra.public_line + "\n").encode(), mode=0o644)
    return KrunHostKeypair(
        private_path=private,
        public_path=public,
        public_line=infra.public_line,
        fingerprint=infra.fingerprint,
        created=infra.created,
    )


def make_krun_runtime(*, cfg: SandboxConfig | None = None) -> KrunRuntime:
    """Construct a production [`KrunRuntime`][terok_sandbox.KrunRuntime] in one call.

    Wires together the three production pieces — the vault-backed host
    keypair, the TCP-over-passt SSH transport, and a fresh
    [`PodmanRuntime`][terok_sandbox.PodmanRuntime] for lifecycle —
    so the orchestrator's runtime selector reduces to a single call:
    ``_runtime = make_krun_runtime(cfg=...)``.  The experimental-flag
    gate stays on the orchestrator side (this factory is reachable
    only when the gate is open).
    """
    kp = ensure_krun_host_keypair(cfg=cfg)
    transport = TcpSSHTransport(
        identity_file=kp.private_path,
        endpoint_resolver=podman_port_resolver(),
    )
    return KrunRuntime(transport=transport, podman=PodmanRuntime())


# ── Private helpers ─────────────────────────────────────────────────────────


def _ensure_safe_runtime_dir(runtime_dir: Path | None) -> Path:
    """Resolve the krun runtime dir and refuse persistent-disk fallbacks.

    ``namespace_runtime_dir()`` falls back to ``$XDG_STATE_HOME/terok``
    (persistent disk) when ``$XDG_RUNTIME_DIR`` is unset.  Writing
    plaintext private-key material to persistent disk is the exact
    "vault → disk" leak the vault-backed flow was supposed to prevent,
    so refuse the fallback when no explicit *runtime_dir* is given.

    After ``mkdir + chmod``, ``lstat`` the result and refuse to
    proceed if it isn't a regular directory owned by the current user
    with ``0700`` (or stricter) permissions.  Three threats this
    closes:

    - **Symlink redirection** — ``mkdir(exist_ok=True)`` is a no-op when
      the target is a symlink to a directory, and a subsequent ``chmod``
      follows the symlink.  Without ``lstat`` we'd route the keypair
      write into whatever the symlink points at.
    - **Cross-user ownership** — a target someone else owns means
      they can read the key as soon as ``_write_atomic`` produces it.
    - **Group/world-readable mode** — the explicit ``chmod`` above
      caps the perms, but the ``lstat`` proves it actually took (no
      ACL surprises, no mode-bit weirdness on an unusual filesystem).

    Both XDG-derived and caller-supplied paths run through the same
    checks — they're the same trust tier (the operator running terok).
    """
    if runtime_dir is not None:
        target = runtime_dir
    else:
        if not os.environ.get("XDG_RUNTIME_DIR"):
            raise SystemExit(
                "krun host-key cache requires $XDG_RUNTIME_DIR (a tmpfs "
                "user-runtime dir) to be set so the vault-backed private "
                "key never lands on persistent disk.  Run terok-executor "
                "under a logind-managed session (the usual interactive "
                "shell), or set XDG_RUNTIME_DIR to a tmpfs path before "
                "launching."
            )
        target = namespace_runtime_dir()

    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
    _assert_owner_private_dir(target)
    return target


def _assert_owner_private_dir(path: Path) -> None:
    """Raise ``SystemExit`` unless *path* is a real dir, uid-owned, 0700-or-stricter.

    Uses ``lstat`` so a symlink at *path* is rejected outright (rather
    than silently followed).  Called after the ``mkdir`` + ``chmod`` in
    [`_ensure_safe_runtime_dir`][terok_executor.krun._ensure_safe_runtime_dir]
    to verify the mode actually took and the entry is genuinely the
    private directory we expected.
    """
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise SystemExit(
            f"krun host-key cache dir {path} is a symlink — refusing to "
            "write private key material via an indirection.  Remove the "
            "symlink and re-run."
        )
    if not stat.S_ISDIR(st.st_mode):
        raise SystemExit(f"krun host-key cache dir {path} is not a directory")
    if st.st_uid != os.getuid():
        raise SystemExit(
            f"krun host-key cache dir {path} is owned by uid {st.st_uid}, "
            f"not the current user (uid {os.getuid()}) — refusing to write."
        )
    if st.st_mode & 0o077:
        raise SystemExit(
            f"krun host-key cache dir {path} has group/world-accessible "
            f"mode {st.st_mode & 0o777:o}; expected 0700 or stricter."
        )


def _write_atomic(path: Path, data: bytes, *, mode: int) -> None:
    """Write *data* to *path* atomically with *mode* perms.

    Uses ``mkstemp`` (``O_EXCL`` — symlinks at the target are ignored)
    + ``fchmod`` on the descriptor + a short-write-safe ``os.write``
    loop + ``os.replace`` for the rename, so there's no TOCTOU window
    where an attacker could swap in a hardlink between the write and a
    chmod-by-path.  Tmp file is unlinked on any failure path so the
    runtime dir never accumulates stranded ``krun_host.key.<rand>``
    leftovers.  No ``fsync``: the path lives under a tmpfs (enforced
    by
    [`_ensure_safe_runtime_dir`][terok_executor.krun._ensure_safe_runtime_dir])
    where it would be a no-op cost.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        os.fchmod(fd, mode)
        # POSIX permits ``os.write`` to return fewer bytes than the
        # buffer length even for regular files; loop until everything
        # lands or the kernel signals an unrecoverable condition.
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise OSError(f"os.write made no progress at offset {offset}")
            offset += written
        os.close(fd)
        fd = -1
        os.replace(tmp_path, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        # Best-effort cleanup; if unlink itself fails we already have
        # an exception in flight and the leftover is the lesser issue.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
