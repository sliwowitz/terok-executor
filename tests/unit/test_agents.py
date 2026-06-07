# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent configuration: wrapper generation, session hooks, and config dir."""

from __future__ import annotations

import json
import tempfile
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from terok_executor.provider.agents import (
    AgentConfigSpec,
    _inject_opencode_instructions,
    _write_session_hook,
    prepare_agent_config_dir,
)
from terok_executor.provider.providers import AGENTS
from terok_executor.provider.wrappers import generate_agent_wrapper
from tests.constants import (
    CONTAINER_CLAUDE_MEMORY_OVERRIDE,
    CONTAINER_CLAUDE_SESSION_PATH,
    CONTAINER_INSTRUCTIONS_PATH,
)


class TestGenerateClaudeWrapper:
    """Tests for _generate_claude_wrapper."""

    def test_basic_wrapper(self) -> None:
        """Wrapper includes add-dir / and git env vars."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert "claude()" in wrapper
        assert '--add-dir "/"' in wrapper
        assert "_terok_apply_git_identity Claude noreply@anthropic.com" in wrapper

    def test_wrapper_does_not_synthesize_agents_flag(self) -> None:
        """terok no longer injects --agents; native .claude/agents/ is discovered by Claude."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert "agents.json" not in wrapper
        assert "--agents" not in wrapper

    def test_wrapper_includes_append_system_prompt(self) -> None:
        """Wrapper injects --append-system-prompt via a runtime file guard."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert "[ -f /home/dev/.terok/instructions.md ]" in wrapper
        assert "--append-system-prompt" in wrapper

    def test_wrapper_timeout_support(self) -> None:
        """Wrapper parses --terok-timeout and wraps claude with timeout."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert "--terok-timeout" in wrapper
        assert 'timeout "$_timeout" claude' in wrapper
        assert 'command claude "${_args[@]}" "$@"' in wrapper

    def test_wrapper_resume_from_session_file(self) -> None:
        """Wrapper adds --resume from claude-session.txt when it exists."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert "claude-session.txt" in wrapper
        assert "--resume" in wrapper

    def test_wrapper_sets_memory_override(self) -> None:
        """Wrapper exports CLAUDE_COWORK_MEMORY_PATH_OVERRIDE."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert f'"{CONTAINER_CLAUDE_MEMORY_OVERRIDE}"' in wrapper

    def test_wrapper_picks_up_initial_prompt(self) -> None:
        """Wrapper consumes initial-prompt.txt one-shot, gated on no resume."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert "/home/dev/.terok/initial-prompt.txt" in wrapper
        assert "/home/dev/.terok/initial-prompt.consumed.txt" in wrapper
        # Resume always wins — pickup is skipped if the session file is present.
        assert f"[ ! -s {CONTAINER_CLAUDE_SESSION_PATH} ]" in wrapper


class TestRuntimeProviderWrappers:
    """Wrappers honor a runtime-selected provider (TEROK_PROVIDER / --provider)."""

    def test_claude_routes_via_environment(self) -> None:
        """Claude resolves the provider's vault endpoint and exports it."""
        wrapper = generate_agent_wrapper(AGENTS["claude"])
        assert 'local _provider="${TEROK_PROVIDER:-}"' in wrapper
        assert "--provider) _provider=" in wrapper
        # Var name is built from Claude's own protocol; bearer is its oauth var.
        assert "TEROK_PROVIDER_${_pu}_BASE_ANTHROPIC_MESSAGES" in wrapper
        assert 'export ANTHROPIC_BASE_URL="$_prov_base"' in wrapper
        assert 'export CLAUDE_CODE_OAUTH_TOKEN="$_prov_token"' in wrapper

    def test_codex_runs_bare_until_a_provider_is_selected(self) -> None:
        """Codex runs its binary by default (its config_patch routes the default);
        a selected provider re-points it through codex-provider."""
        wrapper = generate_agent_wrapper(AGENTS["codex"])
        assert 'local _provider="${TEROK_PROVIDER:-}"' in wrapper
        assert "_runner=(codex)" in wrapper
        assert "_runner=(codex-provider)" not in wrapper
        assert '_runner=(codex-provider --provider "$_provider")' in wrapper

    def test_vibe_runs_bare_until_a_provider_is_selected(self) -> None:
        """Vibe runs its binary by default (its config_patch routes mistral);
        a selected provider re-points it through vibe-provider."""
        wrapper = generate_agent_wrapper(AGENTS["vibe"])
        assert "_runner=(vibe)" in wrapper
        assert "_runner=(vibe-provider)" not in wrapper
        assert '_runner=(vibe-provider --provider "$_provider")' in wrapper

    def test_opencode_launcher_unchanged(self) -> None:
        """OpenCode keeps routing through its existing launcher (regression)."""
        wrapper = generate_agent_wrapper(AGENTS["opencode"])
        assert '_runner=(opencode-provider --provider "$_provider")' in wrapper

    def test_harnesses_declare_manifest_protocol(self) -> None:
        """opencode/pi declare openai-chat so the readiness manifest surfaces them
        paired with openai-chat providers — the universal cross-provider path."""
        assert AGENTS["opencode"].protocol == "openai-chat"
        assert AGENTS["pi"].protocol == "openai-chat"


class TestWriteSessionHook:
    """Tests for _write_session_hook."""

    def test_creates_settings_with_hook(self) -> None:
        """Creates settings.json with a SessionStart hook."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            _write_session_hook(settings_path)
            data = json.loads(settings_path.read_text())
            assert "SessionStart" in data["hooks"]
            command = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            assert "session_id" in command

    def test_merges_with_existing_settings(self) -> None:
        """Merges hook into existing settings.json without clobbering."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            settings_path.write_text('{"permissions": {"allow": ["Read"]}}', encoding="utf-8")
            _write_session_hook(settings_path)
            data = json.loads(settings_path.read_text())
            assert data["permissions"] == {"allow": ["Read"]}
            assert "SessionStart" in data["hooks"]

    def test_idempotent_hook_write(self) -> None:
        """Calling twice doesn't create duplicate hooks."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            _write_session_hook(settings_path)
            _write_session_hook(settings_path)
            data = json.loads(settings_path.read_text())
            assert len(data["hooks"]["SessionStart"]) == 1

    def test_does_not_rewrite_when_hook_present(self) -> None:
        """If equivalent hook exists, file is left untouched."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            hook_command = (
                "python3 -c \"import json,sys; print(json.load(sys.stdin)['session_id'])\""
                f" > {CONTAINER_CLAUDE_SESSION_PATH}"
            )
            original = json.dumps(
                {
                    "hooks": {
                        "SessionStart": [{"hooks": [{"type": "command", "command": hook_command}]}]
                    }
                },
                separators=(",", ":"),
            )
            settings_path.write_text(original, encoding="utf-8")
            _write_session_hook(settings_path)
            assert settings_path.read_text(encoding="utf-8") == original

    def test_concurrent_writes_keep_single_hook(self) -> None:
        """Concurrent writes produce a single valid SessionStart entry."""
        with tempfile.TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_write_session_hook, settings_path) for _ in range(48)]
                for f in futures:
                    f.result()
            data = json.loads(settings_path.read_text())
            assert len(data["hooks"]["SessionStart"]) == 1


class TestPrepareAgentConfigDir:
    """Tests for prepare_agent_config_dir."""

    @staticmethod
    def _make_spec(tasks_root: Path, task_id: str, **kwargs: object) -> AgentConfigSpec:
        """Build a minimal AgentConfigSpec for testing."""
        return AgentConfigSpec(
            tasks_root=tasks_root,
            task_id=task_id,
            default_agent=None,
            mounts_base=kwargs.pop("mounts_base", None),
            instructions=kwargs.pop("instructions", None),
            **kwargs,
        )

    @unittest.mock.patch("terok_executor.provider.agents._write_session_hook")
    def test_writes_instructions(self, _mock: object, tmp_path: Path) -> None:
        """Instructions text is written to instructions.md."""
        with tempfile.TemporaryDirectory() as envs:
            spec = self._make_spec(
                tmp_path / "tasks", "t1", instructions="Custom.", mounts_base=Path(envs)
            )
            (tmp_path / "tasks" / "t1").mkdir(parents=True)
            d = prepare_agent_config_dir(spec)
            assert (d / "instructions.md").read_text(encoding="utf-8") == "Custom."

    @unittest.mock.patch("terok_executor.provider.agents._write_session_hook")
    def test_default_instructions_when_none(self, _mock: object, tmp_path: Path) -> None:
        """Default instructions.md written when instructions is None."""
        with tempfile.TemporaryDirectory() as envs:
            spec = self._make_spec(tmp_path / "tasks", "t2", mounts_base=Path(envs))
            (tmp_path / "tasks" / "t2").mkdir(parents=True)
            d = prepare_agent_config_dir(spec)
            assert "conventions" in (d / "instructions.md").read_text(encoding="utf-8")

    @unittest.mock.patch("terok_executor.provider.agents._write_session_hook")
    def test_wrapper_has_append_system_prompt(self, _mock: object, tmp_path: Path) -> None:
        """Claude wrapper includes --append-system-prompt when instructions given."""
        with tempfile.TemporaryDirectory() as envs:
            spec = self._make_spec(
                tmp_path / "tasks", "t3", instructions="Test.", mounts_base=Path(envs)
            )
            (tmp_path / "tasks" / "t3").mkdir(parents=True)
            d = prepare_agent_config_dir(spec)
            wrapper = (d / "terok-executor.sh").read_text(encoding="utf-8")
            assert "--append-system-prompt" in wrapper


class TestInjectOpencodeInstructions:
    """Tests for _inject_opencode_instructions()."""

    def test_creates_file_if_missing(self) -> None:
        """Creates opencode.json with instructions entry and $schema."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["instructions"] == [str(CONTAINER_INSTRUCTIONS_PATH)]
            assert data["$schema"] == "https://opencode.ai/config.json"

    def test_idempotent_when_already_present(self) -> None:
        """Does not duplicate the instructions entry on repeated calls."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            _inject_opencode_instructions(config_path)
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["instructions"] == [str(CONTAINER_INSTRUCTIONS_PATH)]

    def test_preserves_existing_instructions(self) -> None:
        """Appends to existing instructions list."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            config_path.write_text(
                json.dumps({"instructions": ["/some/other/file.md"]}), encoding="utf-8"
            )
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert len(data["instructions"]) == 2

    def test_preserves_existing_config_keys(self) -> None:
        """Preserves other keys in the opencode.json file."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            config_path.write_text(json.dumps({"model": "test/model"}), encoding="utf-8")
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["model"] == "test/model"

    def test_creates_parent_directories(self) -> None:
        """Creates parent directories if they do not exist."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "nested" / "dir" / "opencode.json"
            _inject_opencode_instructions(config_path)
            assert config_path.is_file()

    def test_handles_invalid_json(self) -> None:
        """Overwrites file with valid config if existing JSON is invalid."""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "opencode.json"
            config_path.write_text("not valid json", encoding="utf-8")
            _inject_opencode_instructions(config_path)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            assert data["instructions"] == [str(CONTAINER_INSTRUCTIONS_PATH)]
