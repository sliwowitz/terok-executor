# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ImageBuilder class API surface.

The underlying free functions (``build_base_images``, ``render_l0``,
``stage_scripts``, …) have their own dedicated tests in
``test_build.py``; this file exercises the ``ImageBuilder`` methods
that wrap them, so the class-level surface is the unit of test rather
than the implementation it delegates to.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from terok_executor.container.build import (
    DEFAULT_BASE_IMAGE,
    ImageBuilder,
    ImageSet,
)

# ── Build delegation ──────────────────────────────────────────────


def test_build_base_delegates_with_self_state() -> None:
    """``build_base`` threads ``(base_image, family)`` from the instance."""
    builder = ImageBuilder("fedora:44", family="rpm")
    expected = ImageSet(l0="x", l1="y")
    with mock.patch(
        "terok_executor.container.build.build_base_images", return_value=expected
    ) as build:
        result = builder.build_base(agents=("claude",), rebuild=True)
    assert result is expected
    build.assert_called_once_with(
        "fedora:44",
        family="rpm",
        agents=("claude",),
        rebuild=True,
        full_rebuild=False,
        build_dir=None,
        tag_as_default=False,
    )


def test_build_sidecar_delegates_with_self_state() -> None:
    """``build_sidecar`` threads ``(base_image, family)`` and tool_name."""
    builder = ImageBuilder("ubuntu:24.04", family="deb")
    with mock.patch(
        "terok_executor.container.build.build_sidecar_image", return_value="tag"
    ) as build:
        result = builder.build_sidecar(tool_name="custom", rebuild=True)
    assert result == "tag"
    build.assert_called_once_with(
        "ubuntu:24.04",
        family="deb",
        tool_name="custom",
        rebuild=True,
        full_rebuild=False,
        build_dir=None,
    )


def test_ensure_default_l1_delegates() -> None:
    """``ensure_default_l1`` delegates to the module-level fn with self.state."""
    builder = ImageBuilder("fedora:44", family="rpm")
    with mock.patch(
        "terok_executor.container.build.ensure_default_l1", return_value="alias-tag"
    ) as fn:
        result = builder.ensure_default_l1(agents=("claude",))
    assert result == "alias-tag"
    fn.assert_called_once_with("fedora:44", family="rpm", agents=("claude",))


# ── Tag computation ───────────────────────────────────────────────


def test_l0_tag_uses_self_base_image() -> None:
    """``l0_tag`` returns the canonical L0 tag for the builder's base."""
    assert ImageBuilder("fedora:44").l0_tag == "terok-l0:fedora-44"


def test_l1_sidecar_tag_uses_self_base_image() -> None:
    """``l1_sidecar_tag`` returns the canonical sidecar tag for the builder's base."""
    assert ImageBuilder("fedora:44").l1_sidecar_tag == "terok-l1-sidecar:fedora-44"


def test_l1_tag_alias_when_agents_none() -> None:
    """``l1_tag(None)`` returns the default-alias tag."""
    assert ImageBuilder("ubuntu:24.04").l1_tag() == "terok-l1-cli:ubuntu-24.04"


def test_l1_tag_with_agents_appends_suffix() -> None:
    """``l1_tag(agents)`` appends a sorted ``-a-b-c`` suffix."""
    assert (
        ImageBuilder("ubuntu:24.04").l1_tag(("codex", "claude"))
        == "terok-l1-cli:ubuntu-24.04-claude-codex"
    )


# ── Template rendering ────────────────────────────────────────────


def test_render_l0_uses_self_base_and_family() -> None:
    """``render_l0`` threads the builder's base + resolved family."""
    builder = ImageBuilder("fedora:44", family="rpm")
    with mock.patch(
        "terok_executor.container.build.render_l0", return_value="Dockerfile content"
    ) as fn:
        result = builder.render_l0()
    assert result == "Dockerfile content"
    fn.assert_called_once_with("fedora:44", family="rpm")


def test_render_l1_takes_explicit_family() -> None:
    """``render_l1`` is a staticmethod — family must be passed explicitly."""
    with mock.patch("terok_executor.container.build.render_l1", return_value="L1 dockerfile") as fn:
        result = ImageBuilder.render_l1("l0-tag", family="rpm", agents=("claude",), cache_bust="42")
    assert result == "L1 dockerfile"
    fn.assert_called_once_with("l0-tag", family="rpm", agents=("claude",), cache_bust="42")


def test_render_l1_sidecar_takes_explicit_family() -> None:
    """``render_l1_sidecar`` is a staticmethod — family must be passed explicitly."""
    with mock.patch(
        "terok_executor.container.build.render_l1_sidecar", return_value="sidecar"
    ) as fn:
        result = ImageBuilder.render_l1_sidecar(
            "l0-tag", family="rpm", tool_name="ruff", cache_bust="7"
        )
    assert result == "sidecar"
    fn.assert_called_once_with("l0-tag", family="rpm", tool_name="ruff", cache_bust="7")


def test_family_property_auto_detects_when_unset() -> None:
    """``_family`` auto-detects from base_image when the override is None."""
    assert ImageBuilder("fedora:44")._family == "rpm"
    assert ImageBuilder("ubuntu:24.04")._family == "deb"


def test_family_property_honours_override() -> None:
    """``_family`` returns the explicit override verbatim when set."""
    assert ImageBuilder("fedora:44", family="deb")._family == "deb"


# ── Static utilities ──────────────────────────────────────────────


def test_detect_family_classmethod_delegates() -> None:
    """``ImageBuilder.detect_family`` is a re-export of the module fn."""
    assert ImageBuilder.detect_family("fedora:44") == "rpm"
    assert ImageBuilder.detect_family("ubuntu:24.04", "rpm") == "rpm"


def test_image_agents_classmethod_delegates() -> None:
    """``ImageBuilder.image_agents`` delegates to the module fn."""
    with mock.patch(
        "terok_executor.container.build.image_agents", return_value={"claude", "codex"}
    ) as fn:
        assert ImageBuilder.image_agents("some:tag") == {"claude", "codex"}
    fn.assert_called_once_with("some:tag")


def test_stage_scripts_delegates(tmp_path: Path) -> None:
    """``stage_scripts`` static method routes to the module fn."""
    with mock.patch("terok_executor.container.build.stage_scripts") as fn:
        ImageBuilder.stage_scripts(tmp_path)
    fn.assert_called_once_with(tmp_path)


def test_stage_tmux_config_delegates(tmp_path: Path) -> None:
    """``stage_tmux_config`` static method routes to the module fn."""
    with mock.patch("terok_executor.container.build.stage_tmux_config") as fn:
        ImageBuilder.stage_tmux_config(tmp_path)
    fn.assert_called_once_with(tmp_path)


def test_stage_toad_agents_delegates(tmp_path: Path) -> None:
    """``stage_toad_agents`` static method routes to the module fn."""
    with mock.patch("terok_executor.container.build.stage_toad_agents") as fn:
        ImageBuilder.stage_toad_agents(tmp_path)
    fn.assert_called_once_with(tmp_path)


# ── Default base image ────────────────────────────────────────────


def test_default_base_image() -> None:
    """A bare ``ImageBuilder()`` falls back to ``DEFAULT_BASE_IMAGE``."""
    assert ImageBuilder().base_image == DEFAULT_BASE_IMAGE
