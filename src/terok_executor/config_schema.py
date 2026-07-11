# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for the executor-owned slice of the shared ``config.yml``.

terok-executor owns one top-level section in the shared config:
``image:`` (base image, agent roster, Dockerfile snippets).  This
module defines that section's strict schema and composes it with
sandbox's [`SandboxConfigView`][terok_sandbox.config_schema.SandboxConfigView].

Standalone executor consumers (``terok-executor run``) validate the
file against [`ExecutorConfigView`][terok_executor.config_schema.ExecutorConfigView].  Sandbox-owned and
executor-owned sections are strict on their own keys; unknown
top-level sections (terok's ``tui:``, ``logs:`` …) pass through
silently because the view is itself ``extra="allow"``.

Higher layers (terok) inherit from [`ExecutorConfigView`][terok_executor.config_schema.ExecutorConfigView] and
flip the top level to ``extra="forbid"`` because they know the full
ecosystem set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from terok_executor.integrations.sandbox import SandboxConfigView

# ── Owned sub-section ─────────────────────────────────────────────────


class RawImageSection(BaseModel):
    """The ``image:`` section — base image, agent roster, Dockerfile snippets.

    Strict on its own keys (``extra="forbid"``).  Same shape used in both
    the global ``config.yml`` (defaults across projects) and per-project
    ``project.yml`` (project overrides).
    """

    model_config = ConfigDict(extra="forbid")

    base_image: str = Field(default="fedora:44", description="Base container image for builds")
    family: Literal["deb", "rpm"] | None = Field(
        default=None,
        description=(
            "Package family for the L0/L1 build (``deb`` or ``rpm``). "
            "Leave unset to auto-detect from *base_image*; set explicitly "
            "when the image is outside the known allowlist."
        ),
    )
    agents: str | None = Field(
        default=None,
        description=(
            'Comma-separated roster entries to install in L1, or "all". '
            'Prefix a name with "-" to exclude it from the selection '
            '(e.g. "all,-vibe" or just "-vibe" — both mean "everything '
            'except vibe").  Inherits from the global config when unset.'
        ),
    )
    user_snippet_inline: str | None = Field(
        default=None, description="Inline Dockerfile snippet injected into the project image"
    )
    user_snippet_file: str | None = Field(
        default=None, description="Path to a file containing a Dockerfile snippet"
    )


# ── Executor's view of the global config ──────────────────────────────


class ExecutorConfigView(SandboxConfigView):
    """The slice of ``config.yml`` executor owns + sandbox owns (transitively).

    Inherits all eight sandbox-owned sections from
    [`SandboxConfigView`][terok_sandbox.config_schema.SandboxConfigView] and adds
    the executor-owned ``image:`` section.  ``extra="allow"`` keeps the
    view tolerant of foreign top-level keys (terok's ``tui:`` /
    ``logs:`` / ``tasks:`` / ``git:`` / ``hooks:``) — standalone
    ``terok-executor run`` flows don't crash on a complete ecosystem
    config, no need to vendor a list of terok's section names here.

    terok's ``RawGlobalConfig`` inherits from this class and flips
    back to ``extra="forbid"``: the topmost layer knows every section,
    so a typo at the top level is caught there.

    The class also exposes staticmethods for reading and writing the
    ``image:`` section on disk: ``image_agents()``,
    ``image_base_image()``, and ``set_image_agents(selection)``.  The
    schema thus owns both the *shape* and the canonical *accessors*
    for its owned section, rather than scattering one helper per
    operation across a separate ``config`` module.
    """

    model_config = ConfigDict(extra="allow")

    image: RawImageSection = Field(default_factory=RawImageSection)

    @staticmethod
    def image_agents() -> str | None:
        """Return the effective ``image.agents``, or ``None`` when unset.

        ``None`` distinguishes "field absent" from ``"all"`` (the
        explicit "every roster entry" selector).
        """
        from terok_util import read_config_section

        return read_config_section("image").get("agents") or None

    @staticmethod
    def image_base_image() -> str | None:
        """Return the explicit ``image.base_image``, or ``None`` when unset.

        Callers apply the schema fallback themselves
        ([`DEFAULT_BASE_IMAGE`][terok_executor.DEFAULT_BASE_IMAGE]) —
        keeping that constant out of this module preserves the
        foundation-layer boundary (schema sits below container/build).
        """
        from terok_util import read_config_section

        return read_config_section("image").get("base_image") or None

    @staticmethod
    def set_image_agents(selection: str) -> Path:
        """Write *selection* into ``image.agents`` and return the file path.

        Caller validates *selection* up-front (typically via
        [`AgentRoster.validate_selection`][terok_executor.roster.loader.AgentRoster.validate_selection]).

        Invalidates terok-util's process-wide ``read_config_section``
        cache before returning so the next ``image_agents()`` /
        ``image_base_image()`` call observes the freshly-written value
        rather than the in-memory snapshot taken before the write.
        """
        from terok_util import paths as _util_paths

        from terok_executor.config import writable_config_path
        from terok_executor.integrations.sandbox import yaml_update_section

        path = writable_config_path()
        yaml_update_section(path, "image", {"agents": selection})
        _util_paths._config_section_cache.clear()
        return path


__all__ = [
    "ExecutorConfigView",
    "RawImageSection",
]
