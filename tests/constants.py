# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared test constants: filesystem paths.

Centralizes hardcoded path literals so linters only flag the constant
definition, not every test assertion.
"""

from pathlib import Path

# ── Placeholder directories ──────────────────────────────────────────────────

MOCK_BASE = Path("/tmp/terok-testing")
"""Root for synthetic filesystem paths used by mocked tests."""

# ── Nonexistent / missing paths ──────────────────────────────────────────────

NONEXISTENT_DIR = Path("/nonexistent")
"""Guaranteed-missing absolute path used for missing-file behavior tests."""

NONEXISTENT_AGENT_PATH = NONEXISTENT_DIR / "agent.md"
"""Missing agent markdown path used by sub-agent parsing tests."""

NONEXISTENT_FILE_PATH = NONEXISTENT_DIR / "file.md"
"""Missing generic file path used by parse-md-agent tests."""

NONEXISTENT_CONFIG_YAML = NONEXISTENT_DIR / "config.yml"
"""Missing YAML config path used by config-stack tests."""

NONEXISTENT_CONFIG_JSON = NONEXISTENT_DIR / "config.json"
"""Missing JSON config path used by config-stack tests."""

NONEXISTENT_PROJECT_ROOT = MOCK_BASE / "does-not-exist"
"""Missing fake project root used by instruction-resolution tests."""

# ── Container/internal paths asserted in generated scripts ───────────────────

CONTAINER_HOME = Path("/home/dev")
"""Container home directory used in generated wrapper/config assertions."""

CONTAINER_TEROK_DIR = CONTAINER_HOME / ".terok"
"""Container terok state/config directory used by wrapper assertions."""

CONTAINER_INSTRUCTIONS_PATH = CONTAINER_TEROK_DIR / "instructions.md"
"""Container instructions file path injected into agent configs."""

WORKSPACE_ROOT = Path("/workspace")
"""Canonical workspace root referenced in bundled instructions assertions."""
