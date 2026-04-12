# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""AI provider behavior — provider definitions, headless modes, wrapper generation, instructions.

Delegates to :mod:`.providers` for the agent provider registry and environment
collection, :mod:`.wrappers` for shell wrapper generation, :mod:`.headless`
for headless command construction and config resolution, :mod:`.config` for
provider-aware config value extraction, :mod:`.instructions` for per-provider
instruction resolution, and :mod:`.agents` for agent config directory
preparation and wrapper scripts.
"""
