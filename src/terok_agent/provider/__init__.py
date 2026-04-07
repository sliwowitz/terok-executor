# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""AI provider behavior — headless modes, wrapper generation, instructions.

Delegates to :mod:`.config` for provider-aware config value extraction,
:mod:`.headless` for the autopilot provider registry and CLI command building,
:mod:`.instructions` for per-provider instruction resolution, and
:mod:`.agents` for agent config directory preparation and wrapper scripts.
"""
