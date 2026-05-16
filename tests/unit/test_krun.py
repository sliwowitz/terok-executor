# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for `terok_executor.krun`: %host keypair materialisation + factory.

The vault is the system of record (``%host`` infrastructure scope);
these tests use a per-test ``CredentialDB`` patched in via ``cfg`` so
they keep the production wiring honest while exercising real key
generation + storage.  No subprocess is run — the runtime factory is
unit-tested only for the wiring shape, not for talking to krun.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_executor.krun import (
    L0GHostKeypair,
    ensure_l0g_host_keypair,
    make_krun_runtime,
)


@pytest.fixture()
def _vault_backed(tmp_path: Path):
    """Build a ``cfg`` MagicMock whose ``open_credential_db`` yields a real DB.

    Mirrors the production flow where each call opens and closes its own
    ``CredentialDB`` handle — the test injects a per-test temp path with
    a known passphrase, no real vault unlock needed.  Yields the cfg so
    individual tests can pass it into the helper.
    """
    from terok_sandbox import CredentialDB

    db_path = tmp_path / "vault" / "credentials.db"

    def _open(*, prompt_on_tty: bool = False) -> CredentialDB:
        return CredentialDB(db_path, passphrase="test")

    cfg = MagicMock()
    cfg.open_credential_db = _open
    return cfg


class TestEnsureL0GHostKeypair:
    """`ensure_l0g_host_keypair` mints via the vault and materialises to tmpfs."""

    def test_creates_keypair_when_missing(self, tmp_path: Path, _vault_backed) -> None:
        """First call mints in the vault, writes 0600 OpenSSH PEM to tmpfs."""
        runtime_dir = tmp_path / "runtime"
        result = ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)

        assert isinstance(result, L0GHostKeypair)
        assert result.created is True
        assert result.private_path == runtime_dir / "krun_host.key"
        assert result.public_path == runtime_dir / "krun_host.key.pub"

        private = result.private_path.read_bytes()
        assert private.startswith(b"-----BEGIN OPENSSH PRIVATE KEY-----")
        # 0o600 = owner-only read/write; matches what ``ssh -i`` requires.
        assert (result.private_path.stat().st_mode & 0o777) == 0o600

        line = result.public_path.read_text()
        assert line.startswith("ssh-ed25519 ")
        assert line.rstrip().endswith("krun-host (terok)")
        # The dataclass mirrors the on-disk public line exactly.
        assert result.public_line == line.rstrip("\n")

    def test_refuses_persistent_disk_when_no_xdg_runtime_dir(
        self, _vault_backed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``$XDG_RUNTIME_DIR`` → refuse to write private bytes to disk.

        The default ``namespace_runtime_dir()`` would otherwise fall
        back to ``$XDG_STATE_HOME/terok`` (persistent disk).  Letting
        the vault-backed private key land there defeats the whole
        "vault is the system of record, tmpfs is a transient handle"
        property — fail closed instead.
        """
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        with pytest.raises(SystemExit, match="requires .*XDG_RUNTIME_DIR"):
            ensure_l0g_host_keypair(cfg=_vault_backed)  # no explicit runtime_dir

    def test_tightens_existing_dir_to_0700(self, tmp_path: Path, _vault_backed) -> None:
        """A pre-existing runtime dir wider than 0700 is re-tightened.

        ``mkdir(mode=0o700, exist_ok=True)`` is no-op for an existing
        dir, so a previous run under a more permissive umask could
        leave the cache dir world-listable.  Re-chmod every time.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(mode=0o755)  # too wide
        ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)
        assert (runtime_dir.stat().st_mode & 0o777) == 0o700

    def test_refuses_symlink_runtime_dir(self, tmp_path: Path, _vault_backed) -> None:
        """A symlink at the target path is refused before any key is written.

        Without ``lstat``, ``mkdir(exist_ok=True)`` is a no-op on a
        symlink-to-dir and ``chmod`` follows it — the keypair would be
        written into the symlink's target instead of the intended dir.
        ``_assert_owner_private_dir`` raises before the write happens.
        """
        real = tmp_path / "real"
        real.mkdir(mode=0o700)
        link = tmp_path / "via-symlink"
        link.symlink_to(real, target_is_directory=True)

        with pytest.raises(SystemExit, match="symlink"):
            ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=link)

        # The real target was never written into.
        assert list(real.iterdir()) == []

    def test_refuses_group_or_world_readable_runtime_dir(
        self, tmp_path: Path, _vault_backed
    ) -> None:
        """An ACL/filesystem oddity that prevents the chmod from taking is rejected.

        Hard to reproduce naturally on tmpfs, so simulate by stubbing
        ``os.chmod`` to a no-op for a directory that starts at 0755.
        The post-chmod ``lstat`` then sees the wide mode and refuses.
        """
        from unittest.mock import patch

        runtime_dir = tmp_path / "wide"
        runtime_dir.mkdir(mode=0o755)
        with (
            patch("terok_executor.krun.os.chmod"),  # chmod becomes a no-op
            pytest.raises(SystemExit, match="group/world-accessible"),
        ):
            ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)

    def test_private_write_is_atomic_no_symlink_clobber(
        self, tmp_path: Path, _vault_backed
    ) -> None:
        """A symlink at the target path is replaced atomically, not followed.

        ``os.replace`` is atomic and never follows a symlink at the
        destination — so an attacker who pre-creates ``krun_host.key``
        as a symlink to ``/etc/passwd`` can't trick us into writing
        the PEM through to that target.  The replace cuts the symlink
        out of the way, leaving a regular file with the PEM bytes.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(mode=0o700)
        decoy = tmp_path / "decoy-target"
        decoy.write_text("untouched")
        (runtime_dir / "krun_host.key").symlink_to(decoy)

        ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)

        priv = runtime_dir / "krun_host.key"
        assert not priv.is_symlink()
        assert priv.read_bytes().startswith(b"-----BEGIN OPENSSH PRIVATE KEY-----")
        assert decoy.read_text() == "untouched"

    def test_public_write_also_resists_symlink_clobber(self, tmp_path: Path, _vault_backed) -> None:
        """Same atomic-replace protection applies to the public key file."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(mode=0o700)
        decoy = tmp_path / "decoy-pub"
        decoy.write_text("untouched")
        (runtime_dir / "krun_host.key.pub").symlink_to(decoy)

        ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)

        pub = runtime_dir / "krun_host.key.pub"
        assert not pub.is_symlink()
        assert pub.read_text().startswith("ssh-ed25519 ")
        assert decoy.read_text() == "untouched"

    def test_idempotent_returns_same_key_material(self, tmp_path: Path, _vault_backed) -> None:
        """Second call reloads the existing %host key — same public line.

        The on-disk private bytes differ across calls because OpenSSH
        PEM serialisation embeds a random ``checkint`` — compare the
        public line (stable identity) instead.  The second call also
        reports ``created=False`` so callers can surface "minted just
        now" diagnostics from the first call alone.
        """
        runtime_dir = tmp_path / "runtime"
        first = ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)
        second = ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)

        assert first.public_line == second.public_line
        assert first.fingerprint == second.fingerprint
        assert first.created is True
        assert second.created is False

    def test_tmpfs_cache_rewritten_from_vault_on_every_call(
        self, tmp_path: Path, _vault_backed
    ) -> None:
        """Out-of-band tmpfs tampering is overwritten from the vault.

        The vault is the source of truth — if an operator (or anything
        else) modifies the tmpfs private file between calls, the next
        call must restore it.  This is what makes vault-side rotation
        propagate without manual intervention.
        """
        runtime_dir = tmp_path / "runtime"
        ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)
        priv = runtime_dir / "krun_host.key"
        priv.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\nGARBAGE\n")

        ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)

        restored = priv.read_bytes()
        assert restored.startswith(b"-----BEGIN OPENSSH PRIVATE KEY-----")
        assert b"GARBAGE" not in restored
        assert (priv.stat().st_mode & 0o777) == 0o600

    def test_pubkey_is_baked_in_authorized_keys_form(self, tmp_path: Path, _vault_backed) -> None:
        """The .pub file is exactly what L0G ``build_l0g_image`` consumes.

        Loose round-trip: parse the public line via cryptography to
        confirm it's a valid OpenSSH public key — that's the contract
        ``ssh`` and ``authorized_keys`` rely on.
        """
        from cryptography.hazmat.primitives import serialization

        runtime_dir = tmp_path / "runtime"
        ensure_l0g_host_keypair(cfg=_vault_backed, runtime_dir=runtime_dir)
        line = (runtime_dir / "krun_host.key.pub").read_text().strip()
        key_part = " ".join(line.split()[:2])
        serialization.load_ssh_public_key(key_part.encode())  # no raise


class TestMakeKrunRuntime:
    """`make_krun_runtime` wires the vault key into a VsockSSHTransport-backed runtime."""

    def test_returns_krun_runtime_with_vsock_transport(self, tmp_path: Path, _vault_backed) -> None:
        """Production factory: KrunRuntime + VsockSSHTransport, identity from %host."""
        from terok_sandbox import KrunRuntime, PodmanRuntime
        from terok_sandbox.runtime.krun_transport import VsockSSHTransport

        # Force the helper to use our temp runtime_dir by patching it at
        # call time — the factory itself doesn't expose runtime_dir.
        runtime_dir = tmp_path / "runtime"
        with patch("terok_executor.krun._ensure_safe_runtime_dir", return_value=runtime_dir):
            runtime_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            rt = make_krun_runtime(cfg=_vault_backed)

        assert isinstance(rt, KrunRuntime)
        assert isinstance(rt.transport, VsockSSHTransport)
        # The runtime composes a fresh PodmanRuntime for lifecycle verbs.
        assert isinstance(rt._podman, PodmanRuntime)
