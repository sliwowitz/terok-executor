# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Reads and writes the executor's slice of the global ``config.yml`` (``image:``).

The ``image:`` section is the executor's schema slice — ``base_image``
seeds every L0/L1 build's FROM line, ``agents`` picks which roster
entries are baked in.  Readers return the merged effective value;
``None`` distinguishes "operator hasn't chosen yet" from any explicit
value.  The writer goes through
[`yaml_update_section`][terok_sandbox.yaml_update_section] so the
on-disk replace is atomic and the file mode stays 0o600.
"""

from __future__ import annotations

import os
from pathlib import Path

from terok_sandbox import yaml_update_section
from terok_sandbox.paths import namespace_config_dir, read_config_section

__all__ = [
    "get_global_image_agents",
    "get_global_image_base_image",
    "set_global_image_agents",
    "writable_config_path",
]


def get_global_image_agents() -> str | None:
    """Return the effective ``image.agents``, or ``None`` when unset.

    ``None`` distinguishes "field absent" from ``"all"`` (the explicit
    "every roster entry" selector).
    """
    return read_config_section("image").get("agents") or None


def get_global_image_base_image() -> str | None:
    """Return the explicit ``image.base_image``, or ``None`` when unset.

    Callers apply the schema fallback themselves
    ([`DEFAULT_BASE_IMAGE`][terok_executor.DEFAULT_BASE_IMAGE]) — keeping
    that constant out of this module preserves the foundation-layer
    boundary (config sits below container/build).
    """
    return read_config_section("image").get("base_image") or None


def set_global_image_agents(selection: str) -> Path:
    """Write *selection* into ``image.agents`` and return the file path.

    Caller validates *selection* up-front (typically via
    [`validate_agent_selection`][terok_executor.validate_agent_selection]).
    """
    path = writable_config_path()
    yaml_update_section(path, "image", {"agents": selection})
    return path


def writable_config_path() -> Path:
    """Return the path the next config write should target.

    Honours ``TEROK_CONFIG_FILE`` when set; otherwise the user-scope
    file under [`namespace_config_dir`][terok_sandbox.paths.namespace_config_dir].
    """
    env = os.getenv("TEROK_CONFIG_FILE")
    if env:
        return Path(env).expanduser()
    return namespace_config_dir() / "config.yml"
