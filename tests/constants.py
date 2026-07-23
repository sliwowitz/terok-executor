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

CONTAINER_CLAUDE_SESSION_PATH = CONTAINER_TEROK_DIR / "claude-session.txt"
"""Container Claude session file path used by session-hook assertions."""

CONTAINER_AGENTS_MANIFEST_PATH = CONTAINER_TEROK_DIR / "agents.json"
"""Container readiness manifest path written by the ``terok-agents`` command."""

CONTAINER_CLAUDE_MEMORY_OVERRIDE = "/home/dev/.claude/projects/${PROJECT_ID}-workspace/memory"
"""Literal shell path used in generated Claude wrapper memory override assertions."""

CONTAINER_BIN_DIR = "/usr/local/bin"
"""Container directory the roster's install snippets symlink pinned aliases into."""

WORKSPACE_ROOT = Path("/workspace")
"""Canonical workspace root referenced in bundled instructions assertions."""

# ── Integration tests: real podman containers ────────────────────────────────
# Only the podman-backed suite under tests/integration/ reads these.  They
# name the one image, the one container-name prefix, and the file modes the
# credential contract is stated in — the suite hard-codes nothing itself.

PODMAN_BASE_IMAGE = "docker.io/library/alpine:latest"
"""Tiny base image the container tests launch.

Alpine carries busybox ``stat``/``printenv``/``sleep``, which is the whole
in-container vocabulary the suite needs.  Pulled once per session by the
``podman_image`` fixture; every launch afterwards finds it locally."""

PODMAN_CONTAINER_PREFIX = "terok-executor-itest"
"""Prefix of every container the suite creates.

Names are suffixed with a random token per test and removed in a ``finally``
— an interrupted run leaves at most one identifiable stray, and the prefix
makes it greppable in ``podman ps -a``."""

CONTAINER_KEEPALIVE_COMMAND = ("sleep", "300")

#: The sandbox hardcodes ``-w /workspace`` for every run, so a container
#: without the workspace bind-mount cannot start at all ("workdir does not
#: exist").  Production always mounts the task's workspace there; the
#: integration launches must too.
CONTAINER_WORKSPACE_DIR = "/workspace"
"""Entry command that keeps a test container up long enough to exec into."""

CREDENTIAL_FILE_MODE = 0o600
"""The mode every credential file executor places must land in.

Owner-only, host-side *and* as seen from inside the container: glab aborts
when its ``config.yml`` is looser than this (the 0644 bug class)."""

LOOSE_FILE_MODE = 0o644
"""World-readable mode a pre-0.3 launch left behind — the clamp's input."""

PERMISSIVE_UMASK = 0o000
"""Umask the credential-mode test runs under.

A ``touch()`` without an explicit mode inherits the umask, so a 0022 umask
masks the bug: only an all-permissive umask falsifies the claim that the
0600 comes from executor rather than from the process environment."""

INTEGRATION_VAULT_PASSPHRASE = "integration-test-passphrase"  # nosec: B105 — fixture key
"""Passphrase for the throwaway SQLCipher vault each integration test seeds."""

PODMAN_COMMAND_TIMEOUT = 60
"""Seconds any single podman invocation in the suite may take."""

PODMAN_PULL_TIMEOUT = 300
"""Seconds the once-per-session base-image pull may take."""
