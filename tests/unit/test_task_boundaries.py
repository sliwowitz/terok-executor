# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Task-owned entries cannot redirect host I/O or change published task instructions."""

from __future__ import annotations

import json
import os
import runpy
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest
from terok_util import UnsafePathError

from terok_executor import MAX_PROMPT_BYTES
from terok_executor.credentials.auth import _apply_post_capture_state
from terok_executor.credentials.extractors import extract_credential
from terok_executor.provider.agents import _inject_opencode_instructions, _write_session_hook
from terok_executor.task_files import (
    CONTAINER_AGENT_EVENTS,
    CONTAINER_AGENT_INPUTS,
    INITIAL_PROMPT_CONSUMED_FILENAME,
    INITIAL_PROMPT_FILENAME,
    INPUTS_PROTOCOL,
    INPUTS_PROTOCOL_LABEL,
    TASK_INPUT_PROTOCOL,
    agent_config_volumes,
    agent_event_dir,
    agent_input_dir,
    require_task_input_protocol,
    write_initial_prompt,
)


@pytest.fixture
def prompt_protocol():
    """Load the same standalone helper installed into agent images."""
    helper = resources.files("terok_executor") / "resources/scripts/terok-initial-prompt.py"
    return runpy.run_path(str(helper))["pending_prompt"]


def test_task_input_public_exports_keep_their_defining_objects():
    """Grouped lazy exports retain every task-input name and its defining object."""
    import terok_executor
    from terok_executor import task_files

    for name in task_files.__all__:
        assert name in terok_executor.__all__
        assert getattr(terok_executor, name) is getattr(task_files, name)


def test_inputs_and_acknowledgements_are_live_but_vendor_state_is_not(tmp_path):
    """Sealed tasks expose only their explicit input and acknowledgement channels."""
    state, inputs, events = agent_config_volumes(tmp_path / "agent-config")
    assert not state.read_only
    assert not state.live
    assert inputs.host_path == agent_input_dir(tmp_path)
    assert inputs.container_path == CONTAINER_AGENT_INPUTS
    assert inputs.read_only
    assert inputs.live
    assert events.host_path == agent_event_dir(tmp_path)
    assert events.container_path == CONTAINER_AGENT_EVENTS
    assert not events.read_only
    assert events.live


def test_initial_prompt_is_late_published_immutable_and_consumed_once(tmp_path, prompt_protocol):
    """Publication after mount preparation is visible without changing the input inode later."""
    agent_config_volumes(tmp_path / "agent-config")
    inputs, events = agent_input_dir(tmp_path), agent_event_dir(tmp_path)
    assert prompt_protocol(inputs, events, claim=False) is None
    write_initial_prompt(tmp_path, "First instruction")
    source = inputs / INITIAL_PROMPT_FILENAME
    identity = source.stat().st_ino
    assert prompt_protocol(inputs, events, claim=False) == "First instruction\n"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: prompt_protocol(inputs, events, claim=True), range(16)))
    assert results.count("First instruction\n") == 1
    assert results.count(None) == 15
    assert (events / INITIAL_PROMPT_CONSUMED_FILENAME).read_bytes() == b""
    assert source.read_text() == "First instruction\n"
    assert source.stat().st_ino == identity
    write_initial_prompt(tmp_path, "First instruction")
    with pytest.raises(ValueError, match="immutable"):
        write_initial_prompt(tmp_path, "Replacement")


@pytest.mark.parametrize("entry", ["symlink", "hardlink", "directory", "fifo"])
def test_consumption_refuses_unsafe_marker(tmp_path, prompt_protocol, entry):
    """A forged filesystem entry is an error, not a successful acknowledgement."""
    agent_config_volumes(tmp_path / "agent-config")
    write_initial_prompt(tmp_path, "Instruction")
    marker = agent_event_dir(tmp_path) / INITIAL_PROMPT_CONSUMED_FILENAME
    victim = tmp_path / "outside"
    victim.touch()
    _plant(marker, victim, entry)
    inputs, events = agent_input_dir(tmp_path), agent_event_dir(tmp_path)
    with pytest.raises(ValueError, match="regular file"):
        prompt_protocol(inputs, events, claim=True)
    assert victim.read_bytes() == b""


@pytest.mark.parametrize("writer", [_write_session_hook, _inject_opencode_instructions])
@pytest.mark.parametrize("entry", ["symlink", "hardlink", "directory", "fifo"])
def test_config_merges_refuse_host_disclosure(tmp_path, writer, entry):
    """Shared JSON merges never read an outside inode or a special file."""
    victim = tmp_path / "private.json"
    victim.write_text('{"private": "outside value"}')
    config = tmp_path / "mount" / "settings.json"
    config.parent.mkdir()
    _plant(config, victim, entry)
    with pytest.raises(UnsafePathError):
        writer(config)
    assert victim.read_text() == '{"private": "outside value"}'


@pytest.mark.parametrize(
    "provider,filename,payload",
    [
        ("codex", "auth.json", {"tokens": {"access_token": "outside-token"}}),
        ("claude", ".credentials.json", {"claudeAiOauth": {"accessToken": "outside-token"}}),
        ("gh", "hosts.yml", {"github.com": {"oauth_token": "outside-token"}}),
        ("vibe", ".env", "MISTRAL_API_KEY=outside-token"),
    ],
)
def test_capture_and_exposed_readers_refuse_symlinks(tmp_path, provider, filename, payload):
    """A credential-shaped outside file is not authorized by a task-planted link."""
    outside = tmp_path / "private"
    outside.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    mount = tmp_path / "mount"
    mount.mkdir()
    (mount / filename).symlink_to(outside)
    with pytest.raises(UnsafePathError):
        extract_credential(provider, mount)


def test_post_capture_does_not_resolve_destination_links(tmp_path):
    """Onboarding updates reject both leaf and intermediate host-target links."""
    outside = tmp_path / "outside.json"
    outside.write_text('{"unchanged": true}')
    mounts = tmp_path / "mounts"
    config = mounts / "_claude-config"
    config.mkdir(parents=True)
    (config / ".claude.json").symlink_to(outside)
    with pytest.raises(UnsafePathError):
        _apply_post_capture_state("_claude-config", {".claude.json": {"onboarded": True}}, mounts)
    assert outside.read_text() == '{"unchanged": true}'


def test_shared_config_rejects_symlinked_parent(tmp_path):
    """Reject directory indirection even when the final JSON file is regular."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "settings.json").write_text("{}")
    alias = tmp_path / "mount"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePathError):
        _write_session_hook(alias / "settings.json")
    assert (outside / "settings.json").read_text() == "{}"


def test_retained_input_protocol_requires_current_container_and_image():
    """An image update does not upgrade an existing container's frozen contract."""
    image = SimpleNamespace(labels=lambda: {INPUTS_PROTOCOL_LABEL: INPUTS_PROTOCOL})
    container = SimpleNamespace(name="task", image=image, env={})
    with pytest.raises(RuntimeError, match="recreate"):
        require_task_input_protocol(container)
    container.env["TEROK_CONTAINER_PROTOCOL"] = TASK_INPUT_PROTOCOL
    require_task_input_protocol(container)
    container.image = None
    with pytest.raises(RuntimeError, match="rebuild|Rebuild"):
        require_task_input_protocol(container)


def _plant(path: Path, victim: Path, entry: str) -> None:
    """Create a hostile entry entirely inside the temporary test tree."""
    match entry:
        case "symlink":
            path.symlink_to(victim)
        case "hardlink":
            os.link(victim, path)
        case "directory":
            path.mkdir()
        case "fifo":
            os.mkfifo(path)


def test_shared_opencode_instructions_upgrade_preserves_vendor_entries(tmp_path):
    """Replace only the obsolete terok-owned instruction path on the next launch."""
    config = tmp_path / "opencode.json"
    config.write_text(
        json.dumps({"instructions": ["custom.md", "/home/dev/.terok/instructions.md"]})
    )
    _inject_opencode_instructions(config)
    assert json.loads(config.read_text())["instructions"] == [
        "custom.md",
        f"{CONTAINER_AGENT_INPUTS}/instructions.md",
    ]
    inode = config.stat().st_ino
    _inject_opencode_instructions(config)
    assert config.stat().st_ino == inode


def test_shared_config_rejects_oversize_before_merge(tmp_path):
    """Oversized task output never becomes an unbounded host JSON input."""
    config = tmp_path / "settings.json"
    config.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(UnsafePathError, match="exceeds"):
        _write_session_hook(config)


@pytest.mark.parametrize("character", ["x", "é"])
def test_generated_headless_prompt_size_is_validated_before_publication(tmp_path, character):
    """An oversize prompt cannot leave a partial prepared task input directory."""
    from terok_executor.provider.agents import AgentConfigSpec, prepare_agent_config_dir

    prompt = character * (MAX_PROMPT_BYTES // len(character.encode("utf-8")) + 1)
    spec = AgentConfigSpec(tasks_root=tmp_path, task_id="task", prompt=prompt)
    with pytest.raises(ValueError, match="120 KiB"):
        prepare_agent_config_dir(spec)
    assert not (tmp_path / "task").exists()


@pytest.mark.parametrize("character", ["x", "é"])
def test_generated_headless_prompt_accepts_exact_byte_limit(tmp_path, character):
    """The documented byte boundary remains usable with multibyte text."""
    from terok_executor.provider.agents import AgentConfigSpec, prepare_agent_config_dir

    prompt = character * (MAX_PROMPT_BYTES // len(character.encode("utf-8")))
    config = prepare_agent_config_dir(
        AgentConfigSpec(
            tasks_root=tmp_path, task_id="task", prompt=prompt, mounts_base=tmp_path / "mounts"
        )
    )
    assert (agent_input_dir(config.parent) / "prompt.txt").read_text() == prompt


@pytest.mark.parametrize("character", ["x", "é"])
def test_initial_prompt_byte_limit_includes_its_newline(tmp_path, prompt_protocol, character):
    """A maximum-sized initial prompt is claimable; larger text is never published."""
    quotient, remainder = divmod(MAX_PROMPT_BYTES - 1, len(character.encode("utf-8")))
    prompt = character * quotient + "x" * remainder
    agent_config_volumes(tmp_path / "accepted" / "agent-config")
    write_initial_prompt(tmp_path / "accepted", prompt)
    assert (
        prompt_protocol(
            agent_input_dir(tmp_path / "accepted"),
            agent_event_dir(tmp_path / "accepted"),
            claim=True,
        )
        == prompt + "\n"
    )
    with pytest.raises(ValueError, match="120 KiB"):
        write_initial_prompt(tmp_path / "rejected", prompt + "x")
    assert not (tmp_path / "rejected").exists()


def test_prompt_helper_limit_matches_host_and_rejects_before_claim(tmp_path, prompt_protocol):
    """An oversized immutable input is not acknowledged by the standalone image helper."""
    assert prompt_protocol.__globals__["MAX_BYTES"] == MAX_PROMPT_BYTES < 128 * 1024
    agent_config_volumes(tmp_path / "agent-config")
    inputs, events = agent_input_dir(tmp_path), agent_event_dir(tmp_path)
    (inputs / INITIAL_PROMPT_FILENAME).write_bytes(b"x" * (MAX_PROMPT_BYTES + 1))
    with pytest.raises(ValueError, match="120 KiB"):
        prompt_protocol(inputs, events, claim=True)
    assert not (events / INITIAL_PROMPT_CONSUMED_FILENAME).exists()


def test_image_protocol_preflight_rejects_old_images():
    """An orchestrator can preserve its retained task until a suitable image exists."""
    from terok_executor.task_files import require_image_input_protocol

    image = SimpleNamespace(ref="old-agent", labels=lambda: {})
    with pytest.raises(RuntimeError, match="Rebuild"):
        require_image_input_protocol(image)
    image.labels = lambda: {INPUTS_PROTOCOL_LABEL: INPUTS_PROTOCOL}
    require_image_input_protocol(image)
