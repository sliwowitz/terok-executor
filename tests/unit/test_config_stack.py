# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests: ConfigStack re-exports from terok-sandbox are accessible."""

from __future__ import annotations

from terok_executor import ConfigScope, ConfigStack


def test_config_stack_reexported() -> None:
    """ConfigStack and ConfigScope are importable from terok_executor."""
    stack = ConfigStack()
    stack.push(ConfigScope("base", None, {"a": 1}))
    assert stack.resolve() == {"a": 1}
