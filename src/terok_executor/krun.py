# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Krun (KVM-microVM) host-side provisioning: identity + runtime factory.

Two responsibilities, both intentionally outside terok:

- [`ensure_l0g_host_keypair`][terok_executor.krun.ensure_l0g_host_keypair] —
  mint or load the host SSH keypair the L0G guest image trusts, and
  materialise the private half onto a tmpfs file ``ssh -i`` can read.
  The vault is the system of record; the tmpfs cache is a derived
  view rebuilt per process.

- [`make_krun_runtime`][terok_executor.krun.make_krun_runtime] — one-shot
  constructor for a production ``terok_sandbox.KrunRuntime`` backed by
  the vsock-SSH transport, with the keypair already wired in.  terok
  flips its runtime selector to ``krun`` and calls this; everything
  else is invisible.

Living in executor (rather than terok) keeps the rule honest: anything
that owns ``build_l0g_image`` also owns the trust material that makes
the resulting guest reachable.  Selecting krun without provisioning
the key is a contradiction, so the two operations belong together.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

from terok_sandbox import (
    KrunRuntime,
    PodmanRuntime,
    SandboxConfig,
    VsockSSHTransport,
    ensure_infra_keypair,
    namespace_runtime_dir,
    podman_annotation_resolver,
)

# Names matching the L0G guest image's baked-in trust file.  Keep both
# halves co-located so a future rename touches one place.
_HOST_KEYPAIR_BASENAME = "krun_host"


@dataclasses.dataclass(frozen=True, slots=True)
class L0GHostKeypair:
    """Materialised view of the ``%host`` infrastructure keypair.

    Returned by [`ensure_l0g_host_keypair`][terok_executor.krun.ensure_l0g_host_keypair].
    Carries the tmpfs path to the OpenSSH-PEM private key (ready for
    ``ssh -i``) and the matching public-key line (ready to bake into
    ``authorized_keys.d/terok`` at L0G build time), so callers don't
    have to redo the DER→PEM conversion or re-derive the public line
    from raw blobs.
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


def ensure_l0g_host_keypair(
    *,
    cfg: SandboxConfig | None = None,
    runtime_dir: Path | None = None,
) -> L0GHostKeypair:
    """Load (or mint, first call) the ``%host`` keypair and materialise it to tmpfs.

    The vault is the system of record: the keypair lives in the sandbox
    credential DB under the ``%host`` infrastructure scope.  This
    helper opens the DB, calls ``terok_sandbox.ensure_infra_keypair``
    (which generates the key on first call and reloads it thereafter),
    and writes the OpenSSH-PEM private + the public-key line into
    *runtime_dir* (default:
    [`namespace_runtime_dir()`][terok_sandbox.namespace_runtime_dir]).

    Rotation = clear the ``%host`` scope in the vault, then re-run.
    Typically called once per process from
    [`make_krun_runtime`][terok_executor.krun.make_krun_runtime] (the
    runtime handle is cached upstream), so the tmpfs cache lasts for
    the process lifetime; an out-of-band rotation propagates the next
    time terok-executor starts.  The public half
    (``krun_host.key.pub``) must be baked into the L0G guest image at
    build time so the guest accepts our auth.

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
    return L0GHostKeypair(
        private_path=private,
        public_path=public,
        public_line=infra.public_line,
        fingerprint=infra.fingerprint,
        created=infra.created,
    )


def make_krun_runtime(*, cfg: SandboxConfig | None = None) -> KrunRuntime:
    """Construct a production ``terok_sandbox.KrunRuntime`` in one call.

    Wires together the three production pieces — the vault-backed host
    keypair, the vsock-SSH transport, and a fresh
    ``terok_sandbox.PodmanRuntime`` for lifecycle — so the
    orchestrator's runtime selector reduces to a single call:
    ``_runtime = make_krun_runtime(cfg=...)``.  The experimental-flag
    gate stays on the orchestrator side (this factory is reachable
    only when the gate is open).
    """
    kp = ensure_l0g_host_keypair(cfg=cfg)
    transport = VsockSSHTransport(
        identity_file=kp.private_path,
        endpoint_resolver=podman_annotation_resolver(),
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

    Caller-supplied paths are trusted (tests, operator overrides).
    The chmod after ``mkdir`` is unconditional because ``mkdir(mode=…,
    exist_ok=True)`` is a no-op for an existing dir — a previous run
    under a more permissive umask could otherwise leave the cache dir
    world-listable.
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
    return target


def _write_atomic(path: Path, data: bytes, *, mode: int) -> None:
    """Write *data* to *path* atomically with *mode* perms.

    Uses ``mkstemp`` (``O_EXCL`` — symlinks at the target are ignored)
    + ``fchmod`` on the descriptor + ``os.replace`` for the rename, so
    there's no TOCTOU window where an attacker could swap in a
    hardlink between the write and a chmod-by-path.  No ``fsync``:
    the path lives under a tmpfs (enforced by
    [`_ensure_safe_runtime_dir`][terok_executor.krun._ensure_safe_runtime_dir])
    where it would be a no-op cost.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        os.fchmod(fd, mode)
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)
