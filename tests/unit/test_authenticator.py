# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Authenticator class API surface.

The underlying ``authenticate`` free function has its own dedicated
tests in ``test_auth_capture.py``; this file exercises the
``Authenticator`` class that wraps it so the new bound-provider
surface is the unit of test.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from terok_executor.credentials.auth import Authenticator


def test_run_delegates_with_bound_provider() -> None:
    """``Authenticator(provider).run(...)`` forwards to the module-level fn."""
    with mock.patch("terok_executor.credentials.auth.authenticate") as auth:
        Authenticator("claude").run(
            "proj-123",
            mounts_dir=Path("/m"),
            image="some:tag",
            expose_token=True,
            oauth_enabled=False,
            credential_set="proj-123",
        )
    auth.assert_called_once_with(
        "proj-123",
        "claude",
        mounts_dir=Path("/m"),
        image="some:tag",
        expose_token=True,
        oauth_enabled=False,
        credential_set="proj-123",
        device_auth=False,
    )


def test_run_defaults_match_underlying_fn() -> None:
    """Default kwargs match the underlying ``authenticate`` defaults."""
    with mock.patch("terok_executor.credentials.auth.authenticate") as auth:
        Authenticator("codex").run(None, mounts_dir=Path("/m"))
    auth.assert_called_once_with(
        None,
        "codex",
        mounts_dir=Path("/m"),
        image=None,
        expose_token=False,
        oauth_enabled=True,
        credential_set="default",
        device_auth=False,
    )


def test_run_forwards_device_auth() -> None:
    """``device_auth=True`` rides through to the underlying fn."""
    with mock.patch("terok_executor.credentials.auth.authenticate") as auth:
        Authenticator("codex").run(None, mounts_dir=Path("/m"), device_auth=True)
    assert auth.call_args.kwargs["device_auth"] is True


def test_provider_is_frozen() -> None:
    """``Authenticator`` is a frozen dataclass — provider can't change after construction."""
    import dataclasses

    a = Authenticator("claude")
    assert a.provider == "claude"
    try:
        a.provider = "codex"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("expected FrozenInstanceError")
