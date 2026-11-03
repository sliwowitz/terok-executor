# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Prepares agent config directories with wrappers, instructions, and session hooks.

Generates the ``terok-executor.sh`` wrapper that sets up git identity and CLI
flags inside task containers, writes the per-task instructions file, injects it
into the shared OpenCode configs, and installs the Claude SessionStart hook.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from terok_util import (
    open_untrusted_directory,
    read_untrusted_text,
    untrusted_file_lock,
    write_untrusted_text,
)

from terok_executor.task_files import CONTAINER_AGENT_INPUTS, MAX_PROMPT_BYTES, agent_input_dir

from .providers import OPENCODE_PROVIDERS

_MAX_CONFIG_BYTES = 4 * 1024 * 1024

# ---------------------------------------------------------------------------
# Dataclasses / types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfigSpec:
    """Groups parameters for preparing an agent-config directory."""

    tasks_root: Path
    task_id: str
    prompt: str | None = None
    agent: str = "claude"
    instructions: str | None = None
    default_agent: str | None = None
    mounts_base: Path | None = None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def prepare_agent_config_dir(spec: AgentConfigSpec) -> Path:
    """Create and populate the agent-config directory for a task.

    Host-authored files live in the sibling ``agent-inputs`` directory.
    The returned ``agent-config`` directory holds only writable task state.

    Writes:
    - terok-executor.sh (always) — wrapper functions with git env vars
    - prompt.txt (if prompt given, headless only)
    - instructions.md (always) — custom instructions or a neutral default
    - <envs>/_claude-config/settings.json — SessionStart hook (Claude only)
    - opencode.json entries — ``instructions`` path injected into shared
      OpenCode and Blablador configs

    Args:
        spec: All agent-config parameters bundled in an [`AgentConfigSpec`][terok_executor.provider.agents.AgentConfigSpec].

    Returns the agent_config_dir path.
    """
    from .providers import get_agent as _get_agent

    if spec.prompt is not None and len(spec.prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError(f"Headless prompt exceeds the {MAX_PROMPT_BYTES // 1024} KiB limit")
    resolved = _get_agent(spec.agent, default_agent=spec.default_agent)

    task_dir = spec.tasks_root / str(spec.task_id)
    agent_config_dir = task_dir / "agent-config"
    with open_untrusted_directory(agent_config_dir, create=True):
        pass
    inputs_dir = agent_input_dir(task_dir)

    # Write instructions file — always present so opencode.json `instructions`
    # references never point to a missing file.  When no custom instructions
    # are configured, a neutral default is used.
    _DEFAULT_INSTRUCTIONS = "Follow the project's coding conventions and existing patterns."

    instructions_text = spec.instructions or _DEFAULT_INSTRUCTIONS
    write_untrusted_text(inputs_dir / "instructions.md", instructions_text)

    # Inject instructions path into opencode.json configs on the host so
    # all OpenCode-based providers discover them natively (works for both
    # interactive and headless modes).
    mounts_base = spec.mounts_base
    if mounts_base is None:
        raise ValueError("mounts_base is required in AgentConfigSpec")
    _inject_opencode_instructions(mounts_base / "_opencode-config" / "opencode.json")
    for _name in OPENCODE_PROVIDERS:
        _inject_opencode_instructions(
            mounts_base / f"_{_name}-config" / "opencode" / "opencode.json"
        )

    # Write shell wrapper functions for ALL providers so interactive CLI users
    # can invoke any agent (each provider gets its own shell function).
    from .wrappers import generate_all_wrappers

    wrapper = generate_all_wrappers()
    write_untrusted_text(inputs_dir / "terok-executor.sh", wrapper)

    # Write SessionStart hook — only for providers that support it (Claude)
    if resolved.supports_session_hook:
        shared_claude_dir = mounts_base / "_claude-config"
        _write_session_hook(shared_claude_dir / "settings.json")

    # Prompt (headless only)
    if spec.prompt is not None:
        write_untrusted_text(inputs_dir / "prompt.txt", spec.prompt)

    return agent_config_dir


# ---------------------------------------------------------------------------
# Private helpers (in call order)
# ---------------------------------------------------------------------------


def _inject_opencode_instructions(config_path: Path) -> None:
    """Inject the instructions file path into an opencode.json config.

    Ensures the ``"instructions"`` key is a list containing the container-local
    path ``"/home/dev/.terok/input/instructions.md"``.  If the file does not exist it
    is created with the required ``$schema`` key.  If the instructions entry is already present the
    file is left untouched (idempotent).

    Uses the same inter-process file lock + atomic-replace pattern as
    `_write_session_hook` for concurrency safety.
    """
    instr_path = f"{CONTAINER_AGENT_INPUTS}/instructions.md"
    legacy_path = "/home/dev/.terok/instructions.md"
    schema_url = "https://opencode.ai/config.json"
    with untrusted_file_lock(config_path.with_name(config_path.name + ".lock")):
        existing = _read_config(config_path)
        schema_added = "$schema" not in existing
        existing.setdefault("$schema", schema_url)
        instructions = existing.get("instructions")
        legacy_removed = isinstance(instructions, list) and legacy_path in instructions
        if legacy_removed and isinstance(instructions, list):
            instructions = existing["instructions"] = [
                item for item in instructions if item != legacy_path
            ]
        if isinstance(instructions, list) and instr_path in instructions:
            if not schema_added and not legacy_removed:
                return
        elif isinstance(instructions, list):
            instructions.append(instr_path)
        else:
            existing["instructions"] = [instr_path]
        write_untrusted_text(config_path, json.dumps(existing, indent=2) + "\n")


def _write_session_hook(settings_path: Path) -> None:
    """Write a Claude project settings file with a SessionStart hook.

    ``settings_path`` currently points at the shared Claude config mount
    (``<envs>/_claude-config/settings.json``), so this function must be
    idempotent across many task launches/projects.

    The hook captures the session ID to ``/home/dev/.terok/claude-session.txt``
    on every session start.  That path is in the per-task ``agent-config`` mount,
    so session IDs remain task-local even though the hook definition is shared.
    The wrapper reads this file to add ``--resume`` on subsequent invocations,
    enabling session continuity across container restarts.

    If the settings file already exists, the hook config is merged into it
    (preserving any existing settings).

    Updates are serialized with an inter-process file lock and persisted via
    atomic replace to avoid clobbering concurrent task launches.
    """
    hook_command = (
        "python3 -c \"import json,sys; print(json.load(sys.stdin)['session_id'])\""
        " > /home/dev/.terok/claude-session.txt"
    )
    hook_entry = {"hooks": [{"type": "command", "command": hook_command}]}
    with untrusted_file_lock(settings_path.with_name(settings_path.name + ".lock")):
        existing = _read_config(settings_path)
        hooks = existing.get("hooks")
        if not isinstance(hooks, dict):
            hooks = existing["hooks"] = {}
        sessions = hooks.get("SessionStart")
        if not isinstance(sessions, list):
            sessions = hooks["SessionStart"] = []
        if any(
            isinstance(item, dict)
            and isinstance(item.get("hooks"), list)
            and any(
                isinstance(hook, dict) and hook.get("command") == hook_command
                for hook in item["hooks"]
            )
            for item in sessions
        ):
            return
        sessions.append(hook_entry)
        write_untrusted_text(settings_path, json.dumps(existing, indent=2) + "\n")


def _read_config(path: Path) -> dict:
    """Read ordinary shared JSON, preserving vendor fields but rejecting unsafe paths."""
    try:
        loaded = json.loads(read_untrusted_text(path, max_bytes=_MAX_CONFIG_BYTES))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
