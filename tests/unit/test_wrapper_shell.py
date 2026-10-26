# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Run the rendered agent wrappers under bash against stub binaries.

The string-level tests check what the template *says*; these check what the
shell *does* on the paths a container user actually hits: the ``--help``
framing, ``--terok-new-session``, resume hints, and task-local agent history
reported by the login banner. Every container path the template bakes in is re-rooted
under ``tmp_path`` before sourcing.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import pytest

from terok_executor.provider.providers import AGENTS, OPENCODE_PROVIDERS
from terok_executor.provider.wrappers import generate_all_wrappers
from tests.constants import CONTAINER_TEROK_DIR, CONTAINER_TEROK_SHARE_DIR

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")

_HOST_TOOLS = ("awk", "bash", "cat", "cp", "env", "mkdir", "mv", "python3", "rm", "tee", "timeout")

_HISTORY_FILENAME = "agent-history"
_HISTORY_LABEL = "Agents previously active: "
_BANNER_RESOURCE = resources.files("terok_executor") / "resources/scripts/terok-bash-banner.sh"
_VIBE_ACP_RESOURCE = resources.files("terok_executor") / "resources/scripts/terok-vibe-acp"
_VIBE_STORE = "vibe-session.d"
_VIBE_RECORD = "vibe-session.txt"

_STUB = """#!__BASH__
printf 'ARGV:%s\\n' "$*" | tee -a "$STUB_LOG"
echo "VENDOR USAGE"
env  # so a leaked (exported) shell variable would show in the launched process
exit "${STUB_RC:-0}"
"""


class Harness:
    """A tmp-rooted stand-in for the container: wrappers, stub agents, state dir."""

    def __init__(self, tmp_path: Path) -> None:
        self.bin = tmp_path / "bin"
        self.home = tmp_path / "home"
        self.terok_dir = self.home / ".terok"
        for d in (self.home, self.terok_dir, self.bin):
            d.mkdir()
        for name in _HOST_TOOLS:
            executable = shutil.which(name)
            assert executable is not None, f"{name} is required"
            (self.bin / name).symlink_to(executable)
        self.log = tmp_path / "stub.log"
        # The share dir is re-rooted to a directory that does not exist, which
        # skips the optional git-identity / plugin sources; the identity helper
        # they would define is stubbed instead.
        script = (
            generate_all_wrappers()
            .replace(str(CONTAINER_TEROK_DIR), str(self.terok_dir))
            .replace(str(CONTAINER_TEROK_SHARE_DIR), str(tmp_path / "share"))
        )
        self.wrappers = tmp_path / "wrappers.sh"
        self.wrappers.write_text(script + "\n_terok_apply_git_identity() { :; }\n")

    def stub(self, name: str) -> None:
        """Put a recording stub named *name* on the harness PATH."""
        path = self.bin / name
        path.write_text(_STUB.replace("__BASH__", str(self.bin / "bash")))
        path.chmod(0o755)

    def session(self, filename: str, session_id: str) -> Path:
        """Record *session_id* the way the agent's capture would."""
        path = self.terok_dir / filename
        path.write_text(f"{session_id}\n")
        return path

    def run(self, *argv: str, rc: int = 0) -> subprocess.CompletedProcess[str]:
        """Invoke *argv* through the sourced wrappers; the stub exits with *rc*."""
        command = f"source {shlex.quote(str(self.wrappers))}; {shlex.join(argv)}"
        return subprocess.run(
            ["bash", "-c", command],
            env=self._environment(rc),
            capture_output=True,
            text=True,
            check=False,
        )

    def banner(
        self, *, tty: bool = True, already_shown: bool = False, hilfe_rc: int = 0
    ) -> subprocess.CompletedProcess[str]:
        """Source the real login banner, optionally on a terminal, with stubbed help."""
        source = _BANNER_RESOURCE.read_text().replace(str(CONTAINER_TEROK_DIR), str(self.terok_dir))
        command = f"hilfe() {{ printf 'HILFE\\n'; return {hilfe_rc}; }};\n{source}"
        env = self._environment()
        env["_TEROK_BANNER_SHOWN"] = "1" if already_shown else ""
        argv = ["bash", "-c", command]
        if not tty:
            return subprocess.run(argv, env=env, capture_output=True, text=True, check=False)

        master, slave = os.openpty()
        with os.fdopen(master, "rb") as output, os.fdopen(slave, "wb") as terminal:
            os.set_blocking(master, False)
            result = subprocess.run(
                argv, env=env, stdout=terminal, stderr=subprocess.PIPE, text=True, check=False
            )
            stdout = (output.read() or b"").decode().replace("\r\n", "\n")
        return subprocess.CompletedProcess(result.args, result.returncode, stdout, result.stderr)

    @property
    def calls(self) -> list[str]:
        """Argument vectors the stub was invoked with, in order."""
        return self.log.read_text().splitlines() if self.log.exists() else []

    @property
    def history(self) -> Path:
        """The same task-local history file the wrappers and greeter share."""
        return self.terok_dir / _HISTORY_FILENAME

    def _environment(self, rc: int = 0) -> dict[str, str]:
        """Isolate shell commands from real agent binaries and the operator's home."""
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("TEROK_", "CLAUDE", "VIBE_"))
        }
        env.update(
            # Only selected host tools and test stubs, never real agent launchers.
            PATH=str(self.bin),
            HOME=str(self.home),
            STUB_LOG=str(self.log),
            STUB_RC=str(rc),
            TASK_ID="test-task",
            # Satisfy the native-agent readiness guard (codex, vibe) without a
            # live vault: it only checks that this handle is non-empty.
            TEROK_PROVIDER_OPENAI_BASE_OPENAI_RESPONSES="ready",
            TEROK_PROVIDER_MISTRAL_BASE_OPENAI_CHAT="ready",
        )
        return env


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    """Fresh wrapper harness per test."""
    return Harness(tmp_path)


def test_harness_uses_non_fhs_tools_without_host_launchers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-FHS host supplies tools, but its agent launchers remain outside the harness."""
    host_bin = tmp_path / "host-tools"
    host_bin.mkdir()
    for name in _HOST_TOOLS:
        executable = shutil.which(name)
        assert executable is not None
        (host_bin / name).symlink_to(executable)
    (host_bin / "opencode-provider").symlink_to(host_bin / "bash")
    monkeypatch.setenv("PATH", str(host_bin))

    sandbox = Harness(tmp_path)
    sandbox.stub("claude")
    sandbox.session("claude-session.txt", "abc")
    result = sandbox.run("claude", "--terok-timeout", "5")

    assert result.returncode == 0, result.stderr
    assert sandbox.calls == ["ARGV:--add-dir / --resume abc"]
    assert (sandbox.bin / "bash").readlink() == host_bin / "bash"
    assert sandbox.run("command", "-v", "opencode-provider").returncode == 1


class TestHelpFraming:
    """``-h``/``--help`` wrap the vendor usage in terok's own summary."""

    def test_claude_help_is_framed(self, harness: Harness) -> None:
        """Header (terok flags) → vendor usage → footer, and the binary saw only --help."""
        harness.stub("claude")
        result = harness.run("claude", "--help")
        assert result.returncode == 0
        header = result.stdout.index("--terok-new-session")
        vendor = result.stdout.index("VENDOR USAGE")
        footer = result.stdout.index("terok wrapper", vendor)
        assert header < vendor < footer
        assert harness.calls == ["ARGV:--help"]

    def test_generic_help_short_flag(self, harness: Harness) -> None:
        """-h behaves the same on a roster-driven wrapper; the vendor's exit code passes through."""
        harness.stub("opencode")
        result = harness.run("opencode", "-h", rc=2)
        assert result.returncode == 2
        assert "terok wraps 'opencode' in this container" in result.stdout
        assert "--provider NAME" in result.stdout
        assert harness.calls == ["ARGV:-h"]

    def test_provider_alias_help_reaches_the_harness_wrapper(self, harness: Harness) -> None:
        """A pinned alias delegates to its harness wrapper, help included."""
        harness.stub("opencode")
        result = harness.run("blablador", "--help")
        assert result.returncode == 0
        assert "VENDOR USAGE" in result.stdout
        assert harness.calls == ["ARGV:--help"]


class TestNewSession:
    """``--terok-new-session`` skips the recorded session for one launch."""

    def test_bare_launch_resumes_recorded_session(self, harness: Harness) -> None:
        """Baseline: with a session file and no arguments, the wrapper resumes."""
        harness.stub("claude")
        harness.session("claude-session.txt", "abc")
        assert harness.run("claude").returncode == 0
        assert harness.calls == ["ARGV:--add-dir / --resume abc"]

    def test_flag_skips_resume_and_keeps_file(self, harness: Harness) -> None:
        """The flag drops --resume; the file stays for the agent's capture to overwrite."""
        harness.stub("claude")
        session = harness.session("claude-session.txt", "abc")
        assert harness.run("claude", "--terok-new-session").returncode == 0
        assert harness.calls == ["ARGV:--add-dir /"]
        assert session.read_text() == "abc\n"

    def test_generic_wrapper_honours_flag(self, harness: Harness) -> None:
        """Roster-driven wrappers behave the same with their own resume flag."""
        harness.stub("opencode")
        harness.session("opencode-session.txt", "abc")
        assert harness.run("opencode").returncode == 0
        assert harness.run("opencode", "--terok-new-session").returncode == 0
        assert harness.calls == ["ARGV:--session abc", "ARGV:"]


def _vibe_session(
    store: Path,
    session_id: str,
    *,
    modified: int = 1,
    parent: str | None = None,
    archived: bool = False,
) -> Path:
    """Persist a Vibe 3 root or child session in its unified storage layout."""
    directory = store / "unified" / session_id
    directory.mkdir(parents=True)
    (directory / "CURRENT").write_text("saved checkpoint")
    metadata = directory / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "parent_session_id": parent,
                "archived_at": "archived" if archived else None,
            }
        )
    )
    os.utime(metadata, (modified, modified))
    return metadata


class TestVibeTaskSessions:
    """Shared credentials never turn another task's conversation into our resume."""

    @pytest.mark.parametrize("args", [(), ("--terok-timeout", "5")])
    def test_tasks_share_config_but_resume_only_their_own_sessions(
        self, tmp_path: Path, args: tuple[str, ...]
    ) -> None:
        """CLI and headless select task roots, not shared, child, or archived sessions."""
        shared = tmp_path / "shared-vibe"
        shared.mkdir()
        _vibe_session(shared / "logs" / "session", "foreign", modified=100)
        for name in ("project-a", "project-b"):
            task = tmp_path / name
            task.mkdir()
            harness = Harness(task)
            (harness.home / ".vibe").symlink_to(shared, target_is_directory=True)
            harness.stub("vibe")
            store = harness.terok_dir / _VIBE_STORE
            _vibe_session(store, name)
            _vibe_session(store, "child", parent=name, modified=200)
            _vibe_session(store, "archived", archived=True, modified=300)
            harness.session(_VIBE_RECORD, "foreign")

            result = harness.run("vibe", *args)

            assert result.returncode == 0, result.stderr
            assert harness.calls == [f"ARGV:--resume {name}"]
            assert f"VIBE_SESSION_LOGGING__SAVE_DIR={store}" in result.stdout
            assert "VIBE_SESSION_LOGGING__ENABLED=true" in result.stdout
            assert (harness.terok_dir / _VIBE_RECORD).read_text() == name
            assert harness.history.read_text() == "vibe\n"

    def test_old_shared_pointer_does_not_block_initial_prompt(self, harness: Harness) -> None:
        """An unowned old pointer is discarded; task instructions and first prompt survive."""
        harness.stub("vibe")
        harness.session(_VIBE_RECORD, "foreign")
        harness.session("initial-prompt.txt", "FIRST PROMPT")
        harness.session("instructions.md", "TASK INSTRUCTIONS")
        stub = harness.bin / "vibe"
        stub.write_text(
            stub.read_text().replace(
                'echo "VENDOR USAGE"',
                'cat "$HOME/.vibe/prompts/$VIBE_SYSTEM_PROMPT_ID.md"',
            )
        )
        _vibe_session(harness.home / ".vibe" / "logs" / "session", "foreign")

        result = harness.run("vibe")

        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:FIRST PROMPT"]
        assert "TASK INSTRUCTIONS" in result.stdout
        assert (harness.terok_dir / "initial-prompt.consumed.txt").exists()
        assert (harness.terok_dir / _VIBE_RECORD).read_text() == ""
        assert harness.history.read_text() == "vibe\n"

    def test_capture_on_exit_and_new_session(self, harness: Harness) -> None:
        """A failed agent still records its saved session, and explicit fresh launch works."""
        harness.stub("vibe")
        stub = harness.bin / "vibe"
        stub.write_text(
            stub.read_text().replace(
                'echo "VENDOR USAGE"',
                'mkdir -p "$VIBE_SESSION_LOGGING__SAVE_DIR/unified/created"\n'
                'printf \'{"session_id":"created"}\' > '
                '"$VIBE_SESSION_LOGGING__SAVE_DIR/unified/created/meta.json"\n'
                ': > "$VIBE_SESSION_LOGGING__SAVE_DIR/unified/created/CURRENT"',
            )
        )
        assert harness.run("vibe", rc=7).returncode == 7
        assert (harness.terok_dir / _VIBE_RECORD).read_text() == "created"
        assert harness.run("vibe").returncode == 0
        assert harness.run("vibe", "--terok-new-session").returncode == 0
        assert harness.calls == ["ARGV:", "ARGV:--resume created", "ARGV:"]

    def test_incomplete_metadata_does_not_replace_resumable_root(self, harness: Harness) -> None:
        """Unreadable metadata and sessions without committed state cannot win selection."""
        harness.stub("vibe")
        store = harness.terok_dir / _VIBE_STORE
        _vibe_session(store, "saved")
        _vibe_session(store, "broken", modified=10).write_text("{")
        unsaved = _vibe_session(store, "unsaved", modified=20)
        (unsaved.parent / "CURRENT").unlink()
        result = harness.run("vibe")
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:--resume saved"]

    def test_acp_uses_the_same_task_private_store(self, harness: Harness) -> None:
        """The ACP entry point exposes only the CLI task's sessions, not shared history."""
        harness.stub("vibe-acp")
        share = harness.home.parent / "share"
        share.mkdir()
        (share / "terok-acp-env.sh").write_text(":\n")
        adapter = harness.bin / "terok-vibe-acp"
        adapter.write_text(
            _VIBE_ACP_RESOURCE.read_text()
            .replace(str(CONTAINER_TEROK_SHARE_DIR), str(share))
            .replace(str(CONTAINER_TEROK_DIR), str(harness.terok_dir))
        )
        adapter.chmod(0o755)
        result = harness.run("terok-vibe-acp")
        assert result.returncode == 0, result.stderr
        assert f"VIBE_SESSION_LOGGING__SAVE_DIR={harness.terok_dir / _VIBE_STORE}" in result.stdout
        assert "VIBE_SESSION_LOGGING__ENABLED=true" in result.stdout


class TestResumeHint:
    """A failed launch that the wrapper resumed ends in a hint — nothing more."""

    def test_failed_resume_hints_once_and_keeps_file(self, harness: Harness) -> None:
        """One launch, the agent's exit code, a hint on stderr, session file untouched."""
        harness.stub("claude")
        session = harness.session("claude-session.txt", "abc")
        result = harness.run("claude", rc=7)
        assert result.returncode == 7
        assert "resumed session abc" in result.stderr
        assert "claude --terok-new-session" in result.stderr
        assert harness.calls == ["ARGV:--add-dir / --resume abc"]
        assert session.read_text() == "abc\n"

    def test_headless_failed_resume_hints_too(self, harness: Harness) -> None:
        """The --terok-timeout path reports the same way (it lands in the task log)."""
        harness.stub("claude")
        harness.session("claude-session.txt", "abc")
        result = harness.run("claude", "--terok-timeout", "5", rc=3)
        assert result.returncode == 3
        assert "resumed session abc" in result.stderr
        assert harness.calls == ["ARGV:--add-dir / --resume abc"]

    def test_fast_failure_without_injected_resume_is_left_alone(self, harness: Harness) -> None:
        """The regression: a caller's own bad --resume must not trigger any wrapper reaction."""
        harness.stub("claude")
        session = harness.session("claude-session.txt", "abc")
        result = harness.run("claude", "--resume", "bogus", rc=1)
        assert result.returncode == 1
        assert "terok:" not in result.stderr
        assert harness.calls == ["ARGV:--add-dir / --resume bogus"]
        assert session.read_text() == "abc\n"

    def test_signal_and_timeout_exits_do_not_hint(self, harness: Harness) -> None:
        """A signal death (130) or timeout kill (124) of a resume is not a failed resume."""
        harness.stub("claude")
        harness.session("claude-session.txt", "abc")
        for rc in (124, 130):
            result = harness.run("claude", rc=rc)
            assert result.returncode == rc
            assert "terok:" not in result.stderr, rc

    def test_generic_wrapper_hint_names_its_binary(self, harness: Harness) -> None:
        """Roster-driven wrappers hint with their own command name."""
        harness.stub("opencode")
        harness.session("opencode-session.txt", "abc")
        result = harness.run("opencode", rc=2)
        assert result.returncode == 2
        assert "opencode exited with status 2" in result.stderr
        assert "resumed session abc" in result.stderr
        assert "opencode --terok-new-session" in result.stderr


class TestSubcommandResume:
    """Codex phrases resume as a subcommand; the wrapper places it accordingly."""

    def test_interactive_leads_with_resume(self, harness: Harness) -> None:
        """No arguments + recorded session → ``codex resume <id>``."""
        harness.stub("codex")
        harness.session("codex-session.txt", "abc")
        assert harness.run("codex").returncode == 0
        assert harness.calls == ["ARGV:resume abc"]

    def test_headless_nests_under_exec(self, harness: Harness) -> None:
        """``--terok-timeout … exec <prompt>`` → ``codex exec resume <id> <prompt>``."""
        harness.stub("codex")
        harness.session("codex-session.txt", "abc")
        assert harness.run("codex", "--terok-timeout", "5", "exec", "do it").returncode == 0
        assert harness.calls == ["ARGV:exec resume abc do it"]

    def test_new_session_flag_and_hint(self, harness: Harness) -> None:
        """The shared flag and hint apply to the subcommand form as well."""
        harness.stub("codex")
        harness.session("codex-session.txt", "abc")
        assert harness.run("codex", "--terok-new-session").returncode == 0
        result = harness.run("codex", rc=3)
        assert result.returncode == 3
        assert "codex exited with status 3" in result.stderr
        assert "resumed session abc" in result.stderr
        assert "codex --terok-new-session" in result.stderr
        assert harness.calls == ["ARGV:", "ARGV:resume abc"]


class TestProviderAliasSessions:
    """A pinned alias records and resumes its own session, under its own name."""

    def test_alias_uses_its_own_session_file(self, harness: Harness) -> None:
        """blablador resumes blablador-session.txt; plain opencode keeps its own file."""
        harness.stub("opencode")  # bare opencode runner
        harness.stub("opencode-provider")  # what a `--provider` alias execs
        harness.session("opencode-session.txt", "harness-id")
        harness.session("blablador-session.txt", "alias-id")
        assert harness.run("blablador").returncode == 0
        assert harness.run("opencode").returncode == 0
        assert harness.calls == [
            "ARGV:--provider blablador --session alias-id",
            "ARGV:--session harness-id",
        ]

    def test_alias_without_recorded_session_starts_fresh(self, harness: Harness) -> None:
        """A harness session on file does not leak into an alias that has none."""
        harness.stub("opencode-provider")
        harness.session("opencode-session.txt", "harness-id")
        assert harness.run("kisski").returncode == 0
        assert harness.calls == ["ARGV:--provider kisski"]

    def test_alias_speaks_under_its_own_name(self, harness: Harness) -> None:
        """Help header and resume hint name the alias, not the harness."""
        harness.stub("opencode")  # help execs `command opencode`
        harness.stub("opencode-provider")  # the resumed launch execs the launcher
        harness.session("blablador-session.txt", "alias-id")
        result = harness.run("blablador", "--help")
        assert "terok wraps 'blablador' in this container" in result.stdout
        assert "command blablador" in result.stdout
        result = harness.run("blablador", rc=2)
        assert "blablador exited with status 2" in result.stderr
        assert "resumed session alias-id" in result.stderr
        assert "blablador --terok-new-session" in result.stderr

    def test_alias_variables_do_not_leak_to_children(self, harness: Harness) -> None:
        """The alias hands its name/session file to the wrapper as plain (unexported) variables."""
        harness.stub("opencode-provider")
        harness.session("blablador-session.txt", "alias-id")
        result = harness.run("blablador", "--terok-new-session")
        assert result.returncode == 0
        # The stub prints its own environment; the alias's bookkeeping
        # variables are locals, so they never reach the launched process.
        assert "_terok_wrapper_name" not in result.stdout
        assert "_terok_session_file" not in result.stdout


class TestAgentHistory:
    """Every wrapped invocation appends its command name before launching the agent."""

    def test_every_agent_and_alias_records_help(self, harness: Harness) -> None:
        """Even help counts as a wrapper invocation; aliases record only their own name."""
        assert AGENTS and OPENCODE_PROVIDERS
        commands = [(agent.binary, agent.binary) for agent in AGENTS.values()]
        commands.extend((name, "opencode") for name in OPENCODE_PROVIDERS)
        recorded = []
        for name, binary in commands:
            harness.stub(binary)
            result = harness.run(name, "--help")
            assert result.returncode == 0, result.stderr
            recorded.append(name)
            assert harness.history.read_text().splitlines() == recorded

    @pytest.mark.parametrize("agent", ("claude", "codex", "vibe", "opencode", "copilot"))
    @pytest.mark.parametrize("headless", (False, True))
    def test_failed_agent_still_records_invocation(
        self, harness: Harness, agent: str, headless: bool
    ) -> None:
        """Interactive and timed launches retain history regardless of the agent's exit."""
        harness.stub(agent)
        flags = ("--terok-timeout", "5") if headless else ()
        result = harness.run(agent, *flags, rc=7)
        assert result.returncode == 7, result.stderr
        assert harness.history.read_text() == f"{agent}\n"

    @pytest.mark.parametrize("agent", ("blablador", "kisski"))
    def test_alias_launch_records_alias_not_harness(self, harness: Harness, agent: str) -> None:
        """A provider shortcut logs once, under the command the user actually entered."""
        harness.stub("opencode-provider")
        result = harness.run(agent)
        assert result.returncode == 0, result.stderr
        assert harness.history.read_text() == f"{agent}\n"

    @pytest.mark.parametrize(("agent", "command"), (("claude", "setup-token"), ("codex", "login")))
    def test_refused_command_still_records_invocation(
        self, harness: Harness, agent: str, command: str
    ) -> None:
        """The history records invocation, not successful agent startup."""
        result = harness.run(agent, command)
        assert result.returncode == 1
        assert harness.history.read_text() == f"{agent}\n"
        assert harness.calls == []

    def test_creates_missing_state_directory(self, harness: Harness) -> None:
        """The first invocation can establish its own task-state directory."""
        harness.terok_dir.rmdir()
        harness.stub("claude")
        result = harness.run("claude")
        assert result.returncode == 0, result.stderr
        assert harness.history.read_text() == "claude\n"

    @pytest.mark.parametrize("blocked", ("directory", "history"))
    def test_history_write_failure_does_not_prevent_agent(
        self, harness: Harness, blocked: str
    ) -> None:
        """Unavailable state is silent and cannot change the agent's own exit status."""
        if blocked == "directory":
            harness.terok_dir.rmdir()
            harness.terok_dir.write_text("not a directory")
        else:
            harness.history.mkdir()
        harness.stub("claude")
        result = harness.run("claude", rc=7)
        assert result.returncode == 7
        assert result.stderr == ""
        assert harness.calls == ["ARGV:--add-dir /"]

    def test_history_survives_separate_shells_and_greeter_reads(self, harness: Harness) -> None:
        """Fresh logins show unique agents in first-use order without rewriting history."""
        for agent in ("codex", "claude", "codex"):
            if not (harness.bin / agent).exists():
                harness.stub(agent)
            result = harness.run(agent)
            assert result.returncode == 0, result.stderr
        history = harness.history.read_text()
        assert history == "codex\nclaude\ncodex\n"
        for _ in range(2):
            result = harness.banner()
            assert result.returncode == 0, result.stderr
            assert result.stdout == f"HILFE\n{_HISTORY_LABEL}codex, claude\n"
            assert harness.history.read_text() == history


class TestAgentHistoryBanner:
    """The interactive greeter summarizes prior wrappers immediately after help."""

    @pytest.mark.parametrize("history", (None, "", "\n\n", " \n\t\n"))
    def test_no_agents_means_no_extra_output(self, harness: Harness, history: str | None) -> None:
        """Missing, empty, and blank-only histories leave the ordinary greeting unchanged."""
        if history is not None:
            harness.history.write_text(history)
        result = harness.banner()
        assert result.returncode == 0, result.stderr
        assert result.stdout == "HILFE\n"

    def test_history_follows_help_and_precedes_initial_prompt(self, harness: Harness) -> None:
        """The new line fits between existing help and initial-prompt sections."""
        harness.history.write_text("codex\n\nclaude\ncodex\nblablador\n")
        harness.session("initial-prompt.txt", "PROMPT")
        result = harness.banner()
        assert result.returncode == 0, result.stderr
        summary = f"{_HISTORY_LABEL}codex, claude, blablador"
        assert result.stdout.index("HILFE") < result.stdout.index(summary)
        assert result.stdout.index(summary) < result.stdout.index("Initial prompt")
        assert "PROMPT" in result.stdout

    @pytest.mark.parametrize(("tty", "already_shown"), ((False, False), (True, True)))
    def test_existing_greeting_guards_still_apply(
        self, harness: Harness, tty: bool, already_shown: bool
    ) -> None:
        """Non-terminal output and nested shells do not repeat either greeting section."""
        harness.history.write_text("claude\n")
        result = harness.banner(tty=tty, already_shown=already_shown)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""

    def test_failed_help_does_not_hide_history(self, harness: Harness) -> None:
        """The help command's existing best-effort behavior also permits the history line."""
        harness.history.write_text("claude\n")
        result = harness.banner(hilfe_rc=1)
        assert result.returncode == 0, result.stderr
        assert f"{_HISTORY_LABEL}claude" in result.stdout
