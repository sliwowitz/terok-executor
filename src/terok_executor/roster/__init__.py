# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Loads agent and tool definitions from layered YAML config into a queryable roster.

Delegates to `.loader` for YAML deserialization and roster construction,
and to `.config_stack` for generic layered config resolution.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import (
        AgentRoster as AgentRoster,
        MountDef as MountDef,
        SidecarSpec as SidecarSpec,
        VaultRoute as VaultRoute,
        load_roster as load_roster,
    )

#: Names resolvable through this package.  Only ``AgentRoster`` is part
#: of the stable public surface (re-exported as
#: ``terok_executor.AgentRoster``); the rest stay importable here for
#: internal callers but are absent from ``__all__`` — reach for them via
#: [`.loader`][terok_executor.roster.loader] when you need them.
_LAZY: dict[str, str] = {
    "AgentRoster": ".loader",
    "MountDef": ".loader",
    "SidecarSpec": ".loader",
    "VaultRoute": ".loader",
    "load_roster": ".loader",
}

__all__ = ["AgentRoster"]


def __getattr__(name: str) -> object:
    """Resolve a re-exported name to [`.loader`][terok_executor.roster.loader] on first access (PEP 562)."""
    try:
        module_path = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the lazy names to ``dir()`` / autocompletion."""
    return sorted({*globals(), *_LAZY})
