# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests: ConfigStack / ConfigScope compose from their source modules."""

from __future__ import annotations

from terok_util import ConfigStack

from terok_executor.integrations.sandbox import ConfigScope


def test_config_stack_reexported() -> None:
    """ConfigStack and ConfigScope compose a layered config."""
    stack = ConfigStack()
    stack.push(ConfigScope("base", None, {"a": 1}))
    assert stack.resolve() == {"a": 1}
