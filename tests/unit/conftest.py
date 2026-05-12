# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit-test fixtures for terok-executor.

After at-rest encryption (terok-sandbox#268), opening the credentials
DB requires a passphrase that resolves through the chain.  Unit-test
runners have no passphrase anywhere on disk, so every test that
constructs a ``CredentialDB`` (directly or via ``cfg.open_credential_db``)
would raise ``NoPassphraseError``.

A single autouse fixture replaces ``SandboxConfig.open_credential_db``
with a thin wrapper that honours the config's own ``db_path`` but
applies ``TEST_VAULT_PASSPHRASE`` instead of walking the resolution
chain.  Tests that seed a real SQLCipher file at ``cfg.db_path``
should construct ``CredentialDB(cfg.db_path, passphrase=TEST_VAULT_PASSPHRASE)``
so the on-disk file uses the same key the autouse fixture opens it
with.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

TEST_VAULT_PASSPHRASE = "unit-test-passphrase"  # nosec: B105 — fixture, not a real secret
"""Passphrase used everywhere a unit test seeds or opens the vault DB."""


@pytest.fixture(autouse=True)
def _stub_credential_db_passphrase() -> Iterator[None]:
    """Open ``cfg.open_credential_db`` with ``TEST_VAULT_PASSPHRASE``.

    The wrapper honours each config's own ``db_path`` so tests that
    seed a DB at ``cfg.db_path`` see the production code read from
    exactly the same file.  ``prompt_on_tty`` is accepted (and
    ignored) to match the real signature.
    """
    from pathlib import Path

    from terok_sandbox import CredentialDB

    def _open_method(self, *, prompt_on_tty: bool = False) -> CredentialDB:
        return CredentialDB(self.db_path, passphrase=TEST_VAULT_PASSPHRASE)

    def _open_module(db_path: Path, **_kw: object) -> CredentialDB:
        return CredentialDB(db_path, passphrase=TEST_VAULT_PASSPHRASE)

    with (
        patch("terok_sandbox.config.SandboxConfig.open_credential_db", new=_open_method),
        patch("terok_sandbox.credentials.db.open_credential_db", new=_open_module),
    ):
        yield
