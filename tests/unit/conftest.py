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

A second autouse fixture (``_isolate_user_paths``) redirects ``HOME``
and every ``XDG_*`` / ``TEROK_*_DIR`` knob to a per-test ``tmp_path``
so that any code path constructing a default ``SandboxConfig()`` (the
``_mock_sandbox()`` helper in ``test_runner.py``, the helpers in
``test_acp_roster.py``, etc.) resolves under tmp instead of mutating
the operator's real ``~/.config/terok``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

TEST_VAULT_PASSPHRASE = "unit-test-passphrase"  # nosec: B105 — fixture, not a real secret
"""Passphrase used everywhere a unit test seeds or opens the vault DB."""


# Terok-specific env vars that override path resolution.  The autouse
# isolation fixture unsets each so resolution falls back through the
# tmp-rooted ``HOME`` / ``XDG_*`` chain — never to the operator's real
# state.
_TEROK_PATH_OVERRIDE_ENV_VARS = (
    "TEROK_CONFIG_DIR",
    "TEROK_STATE_DIR",
    "TEROK_VAULT_DIR",
    "TEROK_RUNTIME_DIR",
    "TEROK_ROOT",
    "TEROK_SANDBOX_LIVE_DIR",
    "TEROK_SANDBOX_STATE_DIR",
    "TEROK_SANDBOX_RUNTIME_DIR",
    "TEROK_EXECUTOR_STATE_DIR",
    "TEROK_PORT_REGISTRY_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_user_paths(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect ``HOME`` and every ``XDG_*`` / ``TEROK_*_DIR`` knob to a fresh tmp dir.

    Without this, tests that exercise default-config code paths
    (``SandboxConfig()`` with no overrides, ``_mock_sandbox()`` helpers,
    etc.) fall through to the operator's real ``~/.config/terok/`` and
    XDG state dirs — silently passing on a clean machine and mutating
    those files on a populated one.  ``test_setup_gate_calls_sandbox``
    in particular calls ``runner._setup_gate()`` which does
    ``cfg.gate_base_path.mkdir(parents=True)`` — without isolation
    that's a real ``~/.local/share/terok/gate/...`` creation.

    Uses ``tmp_path_factory`` rather than ``tmp_path`` so the fake home
    lives outside the per-test ``tmp_path``: storage tests like
    ``test_finds_both_tasks`` iterate their own ``tmp_path`` looking
    for task dirs and would otherwise see a stray ``fake-home`` entry.
    """
    fake_home = tmp_path_factory.mktemp("fake-home")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(fake_home / ".local" / "share"))
    monkeypatch.setenv("XDG_STATE_HOME", str(fake_home / ".local" / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_home / ".cache"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(fake_home / "run"))
    for var in _TEROK_PATH_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _stub_credential_db_passphrase() -> Iterator[None]:
    """Open ``cfg.open_credential_db`` with ``TEST_VAULT_PASSPHRASE``.

    The wrapper honours each config's own ``db_path`` so tests that
    seed a DB at ``cfg.db_path`` see the production code read from
    exactly the same file.  ``prompt_on_tty`` is accepted (and
    ignored) to match the real signature.
    """
    from terok_sandbox import CredentialDB

    def _open_method(
        self, db_path: Path | None = None, *, prompt_on_tty: bool = False
    ) -> CredentialDB:
        return CredentialDB(
            db_path if db_path is not None else self.db_path,
            passphrase=TEST_VAULT_PASSPHRASE,
        )

    def _open_module(db_path: Path, **_kw: object) -> CredentialDB:
        return CredentialDB(db_path, passphrase=TEST_VAULT_PASSPHRASE)

    with (
        patch("terok_sandbox.config.SandboxConfig.open_credential_db", new=_open_method),
        patch("terok_sandbox.vault.store.db.open_credential_db", new=_open_module),
    ):
        yield
