# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Injection helpers for sealed containers.

In sealed isolation mode, the container has no bind mounts — files must be
injected via ``podman cp`` (before start) or ``podman exec`` (at runtime).
These helpers complement :func:`~terok_agent.provider.agents.prepare_agent_config_dir`
which prepares the files on the host side.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def inject_agent_config(container_name: str, config_dir: Path) -> None:
    """Copy a prepared agent-config directory into a sealed container.

    The container must be in the *created* (not running) state.  Delegates
    to :meth:`terok_sandbox.Sandbox.copy_to`.
    """
    from terok_sandbox import Sandbox

    Sandbox().copy_to(container_name, config_dir, "/home/dev/.terok")


def inject_prompt(container_name: str, prompt_text: str) -> None:
    """Write a prompt into a running sealed container.

    Uses ``podman exec`` to stream *prompt_text* into the agent-config
    prompt file.  Intended for headless follow-up prompts where the
    host-side bind mount is absent.
    """
    subprocess.run(
        ["podman", "exec", "-i", container_name, "sh", "-c", "cat > /home/dev/.terok/prompt.txt"],
        input=prompt_text.encode(),
        check=True,
        capture_output=True,
    )
