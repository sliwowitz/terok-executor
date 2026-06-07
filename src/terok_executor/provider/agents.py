# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Prepares agent config directories with wrappers, instructions, and session hooks.

Generates the ``terok-executor.sh`` wrapper that sets up git identity and CLI
flags inside task containers, writes the per-task instructions file, injects it
into the shared OpenCode configs, and installs the Claude SessionStart hook.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from terok_util import ensure_dir, ensure_dir_writable

from .providers import OPENCODE_PROVIDERS

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

    resolved = _get_agent(spec.agent, default_agent=spec.default_agent)

    task_dir = spec.tasks_root / str(spec.task_id)
    agent_config_dir = task_dir / "agent-config"
    ensure_dir(agent_config_dir)

    # Write instructions file — always present so opencode.json `instructions`
    # references never point to a missing file.  When no custom instructions
    # are configured, a neutral default is used.
    _DEFAULT_INSTRUCTIONS = "Follow the project's coding conventions and existing patterns."

    instructions_text = spec.instructions or _DEFAULT_INSTRUCTIONS
    (agent_config_dir / "instructions.md").write_text(instructions_text, encoding="utf-8")

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
    (agent_config_dir / "terok-executor.sh").write_text(wrapper, encoding="utf-8")

    # Write SessionStart hook — only for providers that support it (Claude)
    if resolved.supports_session_hook:
        shared_claude_dir = mounts_base / "_claude-config"
        ensure_dir_writable(shared_claude_dir, "_claude-config")
        _write_session_hook(shared_claude_dir / "settings.json")

    # Prompt (headless only)
    if spec.prompt is not None:
        (agent_config_dir / "prompt.txt").write_text(spec.prompt, encoding="utf-8")

    return agent_config_dir


# ---------------------------------------------------------------------------
# Private helpers (in call order)
# ---------------------------------------------------------------------------


def _inject_opencode_instructions(config_path: Path) -> None:
    """Inject the instructions file path into an opencode.json config.

    Ensures the ``"instructions"`` key is a list containing the container-local
    path ``"/home/dev/.terok/instructions.md"``.  If the file does not exist it
    is created with the required ``$schema`` key.  If the instructions entry is already present the
    file is left untouched (idempotent).

    Uses the same inter-process file lock + atomic-replace pattern as
    `_write_session_hook` for concurrency safety.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - fcntl is unavailable on some platforms.
        fcntl = None  # type: ignore[assignment]

    instr_path = "/home/dev/.terok/instructions.md"
    _SCHEMA_URL = "https://opencode.ai/config.json"

    lock_path = config_path.with_suffix(config_path.suffix + ".lock")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if config_path.is_file():
                try:
                    loaded = json.loads(config_path.read_text(encoding="utf-8"))
                    existing = loaded if isinstance(loaded, dict) else {}
                except (json.JSONDecodeError, OSError):
                    existing = {}
            else:
                existing = {}

            # Ensure the $schema key is always present for a valid opencode.json.
            schema_added = "$schema" not in existing
            existing.setdefault("$schema", _SCHEMA_URL)

            instructions = existing.get("instructions")
            if isinstance(instructions, list) and instr_path in instructions:
                if not schema_added:
                    return  # fully up-to-date, nothing to write
            elif isinstance(instructions, list):
                instructions.append(instr_path)
            else:
                existing["instructions"] = [instr_path]

            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=config_path.parent,
                    delete=False,
                ) as tmp_file:
                    tmp_file.write(json.dumps(existing, indent=2) + "\n")
                    tmp_path = Path(tmp_file.name)
                os.replace(tmp_path, config_path)
            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
    try:
        import fcntl
    except ImportError:  # pragma: no cover - fcntl is unavailable on some platforms.
        fcntl = None  # type: ignore[assignment]

    hook_command = (
        "python3 -c \"import json,sys; print(json.load(sys.stdin)['session_id'])\""
        " > /home/dev/.terok/claude-session.txt"
    )
    hook_entry = {"hooks": [{"type": "command", "command": hook_command}]}
    lock_path = settings_path.with_suffix(settings_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if settings_path.is_file():
                try:
                    loaded = json.loads(settings_path.read_text(encoding="utf-8"))
                    existing = loaded if isinstance(loaded, dict) else {}
                except (json.JSONDecodeError, OSError):
                    existing = {}
            else:
                existing = {}

            changed = False

            hooks_obj = existing.get("hooks")
            if hooks_obj is None or not isinstance(hooks_obj, dict):
                hooks_obj = {}
                existing["hooks"] = hooks_obj
                changed = True

            session_hooks_obj = hooks_obj.get("SessionStart")
            if session_hooks_obj is None or not isinstance(session_hooks_obj, list):
                session_hooks_obj = []
                hooks_obj["SessionStart"] = session_hooks_obj
                changed = True

            session_hooks: list[object] = session_hooks_obj

            # Idempotent across equivalent forms: skip append if an existing SessionStart
            # command already writes session_id to claude-session.txt.
            hook_present = False
            for item in session_hooks:
                if item == hook_entry:
                    hook_present = True
                    break
                if not isinstance(item, dict):
                    continue
                nested = item.get("hooks")
                if not isinstance(nested, list):
                    continue
                for nested_item in nested:
                    if not isinstance(nested_item, dict):
                        continue
                    if nested_item.get("command") == hook_command:
                        hook_present = True
                        break
                if hook_present:
                    break

            if not hook_present:
                session_hooks.append(hook_entry)
                changed = True

            if changed:
                tmp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        "w",
                        encoding="utf-8",
                        dir=settings_path.parent,
                        delete=False,
                    ) as tmp_file:
                        tmp_file.write(json.dumps(existing, indent=2) + "\n")
                        tmp_path = Path(tmp_file.name)
                    os.replace(tmp_path, settings_path)
                finally:
                    if tmp_path is not None and tmp_path.exists():
                        tmp_path.unlink()
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
