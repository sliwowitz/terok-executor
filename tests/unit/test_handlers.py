# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CLI handler functions in ``commands.py``.

Each handler is invoked with the deepest collaborator mocked at the
ImageBuilder / Authenticator boundary so the test exercises the
W5.A class-API call sites added in this PR without paying for a real
podman / vault / OAuth stack.
"""

from __future__ import annotations

from unittest import mock

import pytest

from terok_executor.commands import (
    _build_images_with_banner,
    _handle_agents_set,
    _handle_auth,
    _handle_build,
    _remove_images,
)
from terok_executor.container.build import ImageSet

# ── _handle_auth ────────────────────────────────────────────


def test_handle_auth_api_key_path_stores_directly() -> None:
    """``_handle_auth(api_key=…)`` short-circuits to ``store_api_key`` without container."""
    with (
        mock.patch("terok_executor.credentials.auth.store_api_key") as store,
        mock.patch("terok_executor.credentials.auth.AUTH_PROVIDERS", {"claude": object()}),
    ):
        _handle_auth(agent="claude", api_key="sk-test-key")
    store.assert_called_once_with("claude", "sk-test-key")


def test_handle_auth_oauth_path_routes_through_authenticator() -> None:
    """``_handle_auth`` (no api_key) constructs an Authenticator and runs it."""
    with (
        mock.patch("terok_executor.credentials.auth.Authenticator") as cls,
        mock.patch("terok_executor.credentials.vault_config.write_vault_config"),
    ):
        _handle_auth(agent="claude")
    cls.assert_called_once_with("claude")
    cls.return_value.run.assert_called_once()


def test_handle_auth_device_auth_forwards_to_run() -> None:
    """``--device-auth`` rides through to ``Authenticator.run(device_auth=True)``."""
    with (
        mock.patch("terok_executor.credentials.auth.Authenticator") as cls,
        mock.patch("terok_executor.credentials.vault_config.write_vault_config"),
    ):
        _handle_auth(agent="codex", device_auth=True)
    assert cls.return_value.run.call_args.kwargs["device_auth"] is True


def test_handle_auth_api_key_and_device_auth_warns_and_takes_api_key(capsys) -> None:
    """Both flags given → API key wins, `--device-auth` is ignored with a warning."""
    with (
        mock.patch("terok_executor.credentials.auth.store_api_key") as store,
        mock.patch("terok_executor.credentials.auth.Authenticator") as cls,
        mock.patch("terok_executor.credentials.vault_config.write_vault_config"),
    ):
        _handle_auth(agent="codex", api_key="sk-test", device_auth=True)
    store.assert_called_once()  # API-key route taken
    cls.return_value.run.assert_not_called()  # no auth container
    assert "--device-auth is ignored" in capsys.readouterr().err


def test_handle_auth_empty_api_key_raises() -> None:
    """Empty api_key surfaces as a clean SystemExit, not an obscure failure."""
    with pytest.raises(SystemExit, match="cannot be empty"):
        _handle_auth(agent="claude", api_key="   ")


# ── _handle_build ───────────────────────────────────────────


def test_handle_build_routes_through_image_builder() -> None:
    """``_handle_build`` calls ``ImageBuilder(base, family).build_base()``."""
    images = ImageSet(l0="l0:tag", l1="l1:tag")
    with mock.patch("terok_executor.container.build.ImageBuilder") as cls:
        cls.return_value.build_base.return_value = images
        _handle_build(base="fedora:44", family="rpm", agents="claude", sidecar=False)
    cls.assert_called_once_with("fedora:44", family="rpm")
    cls.return_value.build_base.assert_called_once()


def test_handle_build_sidecar_flag_builds_sidecar() -> None:
    """``sidecar=True`` triggers ``build_sidecar`` after the L0/L1 build."""
    images = ImageSet(l0="l0", l1="l1")
    with mock.patch("terok_executor.container.build.ImageBuilder") as cls:
        cls.return_value.build_base.return_value = images
        cls.return_value.build_sidecar.return_value = "sidecar:tag"
        _handle_build(base="fedora:44", family=None, agents="all", sidecar=True)
    cls.return_value.build_sidecar.assert_called_once()


# ── _handle_agents_set ──────────────────────────────────────


def test_handle_agents_set_writes_through_config_view() -> None:
    """``_handle_agents_set`` validates then writes via ExecutorConfigView."""
    from pathlib import Path

    with (
        mock.patch(
            "terok_executor.roster.loader.AgentRoster.validate_selection", return_value=None
        ),
        mock.patch(
            "terok_executor.config_schema.ExecutorConfigView.set_image_agents",
            return_value=Path("/tmp/cfg.yml"),
        ) as setter,
    ):
        _handle_agents_set(selection="claude,codex")
    setter.assert_called_once_with("claude,codex")


# ── _build_images_with_banner ───────────────────────────────


def test_build_images_with_banner_routes_through_image_builder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Friendly first-run wrapper calls ``ImageBuilder.build_base()``."""
    with mock.patch("terok_executor.container.build.ImageBuilder") as cls:
        cls.return_value.build_base.return_value = ImageSet(l0="L0", l1="L1")
        _build_images_with_banner("fedora:44", None)
    cls.assert_called_once_with("fedora:44", family=None)
    out = capsys.readouterr().out
    assert "Building agent images" in out


# ── _remove_images ──────────────────────────────────────────


def test_remove_images_uses_image_builder_tags() -> None:
    """``_remove_images`` resolves L0 / L1 tags via ``ImageBuilder``."""
    with mock.patch("subprocess.run") as run:
        _remove_images("fedora:44")
    run.assert_called_once()
    argv = run.call_args.args[0]
    assert argv[:4] == ["podman", "image", "rm", "--force"]
    # L0 / L1 tags come from the ImageBuilder properties.
    assert "terok-l0:fedora-44" in argv
    assert "terok-l1-cli:fedora-44" in argv
