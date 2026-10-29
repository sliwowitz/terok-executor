# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Run the rendered agent wrappers under bash against stub binaries.

The string-level tests check what the template *says*; these check what the
shell *does* on the paths a container user actually hits: the ``--help``
framing, ``--terok-new-session``, resume hints, task-local agent history
reported by the login banner, and the provider-readiness pre-flight (which the
Claude ACP launcher repeats), including terok's flag for a stored Claude login
that is not routed to Claude. Every container path the template bakes in is
re-rooted under ``tmp_path`` before sourcing.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path

import pytest

from terok_executor import CLAUDE_OAUTH_UNROUTED_ENV, CLAUDE_OAUTH_UNROUTED_ON
from terok_executor.container.env import _materialize_exposed_providers, _set_provider_handle
from terok_executor.provider.providers import AGENTS, OPENCODE_PROVIDERS
from terok_executor.provider.wrappers import generate_all_wrappers
from terok_executor.roster import AgentRoster
from tests.constants import (
    CONTAINER_TEROK_DIR,
    CONTAINER_TEROK_SHARE_DIR,
    EXAMPLE_PROVIDER_UPSTREAM,
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")

_HOST_TOOLS = ("awk", "bash", "cat", "cp", "env", "mkdir", "mv", "python3", "rm", "tee", "timeout")

_HISTORY_FILENAME = "agent-history"
_HISTORY_LABEL = "Agents previously active: "
_BANNER_RESOURCE = resources.files("terok_executor") / "resources/scripts/terok-bash-banner.sh"
_VIBE_ACP_RESOURCE = resources.files("terok_executor") / "resources/scripts/terok-vibe-acp"
_VIBE_HOOK_RESOURCE = resources.files("terok_executor") / "resources/scripts/terok-vibe-session.py"
_VIBE_RECORD = "vibe-session-id.txt"
_SCRIPTS_RESOURCE = resources.files("terok_executor") / "resources/scripts"
_ACP_ENV_SCRIPT = "terok-acp-env.sh"
_TASK_WRAPPERS = "terok-env-wrappers.sh"
"""The L1 image's link to the per-task rendered wrappers, under the share dir."""

_CLAUDE_HANDLE = "TEROK_PROVIDER_ANTHROPIC_BASE_ANTHROPIC_MESSAGES"
"""Claude's default-provider handle, as the env builder names it (pinned to the
env builder by ``TestClaudeAcpReadiness.test_env_builder_handle_satisfies_the_check``)."""

_PHANTOM_TOKEN = "phantom"
"""Stand-in vault token a test task holds for each provider it authenticates."""

_UNROUTED_CAUSE = "It is an OAuth login, and OAuth routing is off in this configuration"
"""The line of Claude's refusal that names why its stored login cannot reach it."""

_WAY_ROUND = (
    "Or route this launch through the vault:  claude --provider NAME  (ready pairs: providers)"
)
"""The line of the shell refusal that names a selection, which works in the same task."""

_INITIAL_PROMPT = "initial-prompt.txt"
_INITIAL_PROMPT_CONSUMED = "initial-prompt.consumed.txt"

_STUB = """#!__BASH__
printf 'ARGV:%s\\n' "$*" | tee -a "$STUB_LOG"
echo "VENDOR USAGE"
env  # so a leaked (exported) shell variable would show in the launched process
exit "${STUB_RC:-0}"
"""


class Harness:
    """A tmp-rooted stand-in for the container: wrappers, stub agents, state dir."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
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

    def run(
        self, *argv: str, rc: int = 0, env: Mapping[str, str | None] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Invoke *argv* through the sourced wrappers; the stub exits with *rc*.

        *env* overlays the harness environment; a ``None`` value unsets the
        variable, e.g. to withdraw a provider handle the harness pre-seeds.
        """
        command = f"source {shlex.quote(str(self.wrappers))}; {shlex.join(argv)}"
        return subprocess.run(
            ["bash", "-c", command],
            env=self._environment(rc, env),
            capture_output=True,
            text=True,
            check=False,
        )

    def acp(
        self,
        agent: str,
        *argv: str,
        env: Mapping[str, str | None] | None = None,
        task_wrappers: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run the real ``terok-<agent>-acp`` launcher, re-rooted like the wrappers.

        The launcher and the ``terok-acp-env.sh`` it sources are copied into a
        tmp share dir of their own, next to this harness's rendered wrappers
        under the name the image links the per-task wrappers by (left out when
        *task_wrappers* is false, as in a container without them).  The
        git-identity helper ``terok-acp-env.sh`` would source stays absent, as
        it does for the wrappers.  *env* overlays the harness environment the
        same way as for ``run``.
        """
        share = self.root / "acp-share"
        share.mkdir(exist_ok=True)
        launcher = f"terok-{agent}-acp"
        for name in (_ACP_ENV_SCRIPT, launcher):
            source = (_SCRIPTS_RESOURCE / name).read_text()
            (share / name).write_text(
                source.replace(str(CONTAINER_TEROK_SHARE_DIR), str(share)).replace(
                    str(CONTAINER_TEROK_DIR), str(self.terok_dir)
                )
            )
        if task_wrappers:
            (share / _TASK_WRAPPERS).write_text(self.wrappers.read_text())
        return subprocess.run(
            ["bash", str(share / launcher), *argv],
            env=self._environment(overlay=env),
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

    def _environment(
        self, rc: int = 0, overlay: Mapping[str, str | None] | None = None
    ) -> dict[str, str]:
        """Isolate shell commands from real agent binaries and the operator's home.

        *overlay* is applied last: a string sets the variable, ``None`` unsets it.
        ``BASH_ENV`` / ``ENV`` are dropped too: inside a terok task they source
        the container's own wrappers into every ``bash -c``, ahead of the ones
        under test.
        """
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("TEROK_", "CLAUDE", "VIBE_")) and k not in ("BASH_ENV", "ENV")
        }
        env.update(
            # Only selected host tools and test stubs, never real agent launchers.
            PATH=str(self.bin),
            HOME=str(self.home),
            STUB_LOG=str(self.log),
            STUB_RC=str(rc),
            TASK_ID="test-task",
        )
        # Satisfy the native-agent readiness guard (claude, codex, vibe) without
        # a live vault: it only checks that each default's handle is non-empty.
        env.update(
            {
                _CLAUDE_HANDLE: "ready",
                "TEROK_PROVIDER_OPENAI_BASE_OPENAI_RESPONSES": "ready",
                "TEROK_PROVIDER_MISTRAL_BASE_OPENAI_CHAT": "ready",
            }
        )
        for key, value in (overlay or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
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


def _install_vibe_hook(harness: Harness) -> Path:
    """Stage the production helper as in the Vibe image."""
    share = harness.home.parent / "share"
    share.mkdir(exist_ok=True)
    helper = share / "terok-vibe-session.py"
    helper.write_text(_VIBE_HOOK_RESOURCE.read_text())
    return helper


def _stub_vibe_hook(harness: Harness, session_id: str, *, binary: str = "vibe") -> None:
    """Have the fake Vibe invoke the installed native hooks with its own session."""
    harness.stub(binary)
    stub = harness.bin / binary
    invocation = f"""python3 - <<'HOOK'
import json, os, subprocess, tomllib
from pathlib import Path
home = Path(os.environ.get('VIBE_HOME', str(Path.home() / '.vibe')))
for hook in tomllib.loads((home / 'hooks.toml').read_text())['hooks']:
    for parent in (None, 'root'):
        payload = {{'session_id': {session_id!r} if parent is None else 'child',
                   'parent_session_id': parent, 'hook_event_name': hook['type']}}
        subprocess.run(hook['command'], shell=True, check=True,
                       input=json.dumps(payload), text=True)
HOOK"""
    stub.write_text(stub.read_text().replace('echo "VENDOR USAGE"', invocation))


class TestVibeTaskSessions:
    """Transcripts stay shared; native hook IDs belong to the invoking task only."""

    @pytest.mark.parametrize("args", [(), ("--terok-timeout", "5")])
    def test_tasks_share_transcripts_but_resume_only_their_hook_ids(
        self, tmp_path: Path, args: tuple[str, ...]
    ) -> None:
        """Two tasks invoke hooks concurrently against one shared Vibe home."""
        shared = tmp_path / "shared-vibe"
        store = shared / "logs" / "session" / "unified" / "foreign"
        store.mkdir(parents=True)
        transcript = store / "meta.json"
        transcript.write_text('{"session_id":"foreign"}')
        tasks = []
        for name in ("project-a", "project-b"):
            task = tmp_path / name
            task.mkdir()
            harness = Harness(task)
            (harness.home / ".vibe").symlink_to(shared, target_is_directory=True)
            # In real tasks the helper has the same absolute path in each container.
            helper = _install_vibe_hook(harness)
            if tasks:
                helper.unlink()
                helper.symlink_to(tasks[0].home.parent / "share" / helper.name)
            _stub_vibe_hook(harness, name)
            tasks.append(harness)
        with ThreadPoolExecutor() as pool:
            results = list(pool.map(lambda task: task.run("vibe", *args), tasks))
        for name, harness, result in zip(("project-a", "project-b"), tasks, results, strict=True):
            assert result.returncode == 0, result.stderr
            assert harness.calls == ["ARGV:"]
            assert "VIBE_SESSION_LOGGING__" not in result.stdout
            assert (harness.terok_dir / _VIBE_RECORD).read_text().strip() == name
            assert harness.run("vibe", *args).returncode == 0
            assert harness.calls[-1] == f"ARGV:--resume {name}"
            assert harness.history.read_text() == "vibe\nvibe\n"
        assert transcript.read_text() == '{"session_id":"foreign"}'
        assert len(tomllib.loads((shared / "hooks.toml").read_text())["hooks"]) == 2

    def test_old_pointer_does_not_block_initial_prompt(self, harness: Harness) -> None:
        """Unverified legacy pointers are ignored; instructions and first prompt survive."""
        _install_vibe_hook(harness)
        harness.stub("vibe")
        harness.session("vibe-session.txt", "foreign")
        harness.session("initial-prompt.txt", "FIRST PROMPT")
        harness.session("instructions.md", "TASK INSTRUCTIONS")
        stub = harness.bin / "vibe"
        stub.write_text(
            stub.read_text().replace(
                'echo "VENDOR USAGE"', 'cat "$HOME/.vibe/prompts/$VIBE_SYSTEM_PROMPT_ID.md"'
            )
        )
        result = harness.run("vibe")
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:FIRST PROMPT"]
        assert "TASK INSTRUCTIONS" in result.stdout
        assert (harness.terok_dir / "initial-prompt.consumed.txt").exists()
        assert not (harness.terok_dir / _VIBE_RECORD).exists()
        assert "Automatic resume is unavailable" in result.stderr
        assert harness.history.read_text() == "vibe\n"

    def test_hook_capture_survives_failure_and_fresh_launch(self, harness: Harness) -> None:
        """An agent failure does not erase a captured ID; an explicit fresh run replaces it."""
        _install_vibe_hook(harness)
        _stub_vibe_hook(harness, "created")
        assert harness.run("vibe", rc=7).returncode == 7
        assert (harness.terok_dir / _VIBE_RECORD).read_text().strip() == "created"
        assert harness.run("vibe").returncode == 0
        _stub_vibe_hook(harness, "fresh")
        assert harness.run("vibe", "--terok-new-session").returncode == 0
        assert harness.calls == ["ARGV:", "ARGV:--resume created", "ARGV:"]
        assert (harness.terok_dir / _VIBE_RECORD).read_text().strip() == "fresh"

    def test_missing_hooks_warn_without_guessing(self, harness: Harness) -> None:
        """A missing helper cannot enable resume from legacy pointers or shared history."""
        harness.stub("vibe")
        harness.session("vibe-session.txt", "foreign")
        result = harness.run("vibe")
        assert result.returncode == 0
        assert harness.calls == ["ARGV:"]
        assert "Vibe session hooks unavailable" in result.stderr
        assert "Automatic resume is unavailable" in result.stderr

    def test_fresh_launch_without_hook_does_not_reuse_old_pointer(self, harness: Harness) -> None:
        """If no tool/turn completes in the new session, never silently resume old work."""
        _install_vibe_hook(harness)
        harness.stub("vibe")
        harness.session(_VIBE_RECORD, "old")
        result = harness.run("vibe", "--terok-new-session")
        assert "Automatic resume is unavailable" in result.stderr
        assert not (harness.terok_dir / _VIBE_RECORD).exists()
        harness.run("vibe")
        assert harness.calls == ["ARGV:", "ARGV:"]

    def test_native_resume_failure_does_not_fall_back(self, harness: Harness) -> None:
        """Vibe validates its store; rejected resumes preserve the pointer and exit code."""
        _install_vibe_hook(harness)
        harness.stub("vibe")
        harness.session(_VIBE_RECORD, "latest")
        result = harness.run("vibe", rc=7)
        assert result.returncode == 7
        assert harness.calls == ["ARGV:--resume latest"]
        assert (harness.terok_dir / _VIBE_RECORD).read_text().strip() == "latest"
        assert "resumed session latest" in result.stderr

    def test_version_leaves_resume_pointer_intact(self, harness: Harness) -> None:
        """Checking the installed version does not start a new conversation."""
        _install_vibe_hook(harness)
        harness.stub("vibe")
        harness.session(_VIBE_RECORD, "known")
        result = harness.run("vibe", "--version")
        assert result.returncode == 0
        assert not result.stderr
        assert (harness.terok_dir / _VIBE_RECORD).read_text().strip() == "known"

    def test_custom_storage_settings_are_not_overridden(self, harness: Harness) -> None:
        """An operator's Vibe home and logging policy reach Vibe unchanged."""
        _install_vibe_hook(harness)
        harness.stub("vibe")
        custom_home = harness.home / "custom-vibe"
        custom_store = harness.home / "custom-transcripts"
        result = harness.run(
            "vibe",
            env={
                "VIBE_HOME": str(custom_home),
                "VIBE_SESSION_LOGGING__SAVE_DIR": str(custom_store),
                "VIBE_SESSION_LOGGING__ENABLED": "false",
            },
        )
        assert result.returncode == 0, result.stderr
        assert f"VIBE_HOME={custom_home}" in result.stdout
        assert f"VIBE_SESSION_LOGGING__SAVE_DIR={custom_store}" in result.stdout
        assert "VIBE_SESSION_LOGGING__ENABLED=false" in result.stdout
        assert (custom_home / "hooks.toml").exists()

    def test_acp_captures_root_for_cli_without_moving_storage(self, harness: Harness) -> None:
        """ACP uses the same hooks and task pointer, but leaves native storage settings alone."""
        helper = _install_vibe_hook(harness)
        _stub_vibe_hook(harness, "acp-root", binary="vibe-acp")
        (helper.parent / "terok-acp-env.sh").write_text(":\n")
        adapter = harness.bin / "terok-vibe-acp"
        adapter.write_text(
            _VIBE_ACP_RESOURCE.read_text()
            .replace(str(CONTAINER_TEROK_SHARE_DIR), str(helper.parent))
            .replace(str(CONTAINER_TEROK_DIR), str(harness.terok_dir))
        )
        adapter.chmod(0o755)
        result = harness.run("terok-vibe-acp")
        assert result.returncode == 0, result.stderr
        assert "VIBE_SESSION_LOGGING__" not in result.stdout
        assert (harness.terok_dir / _VIBE_RECORD).read_text().strip() == "acp-root"
        harness.stub("vibe")
        assert harness.run("vibe").returncode == 0
        assert harness.calls[-1] == "ARGV:--resume acp-root"


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


def _authenticated(*providers: str) -> dict[str, str]:
    """The handles a task holds for *providers* when each is authenticated in it.

    Built by the env builder itself, so a test's "authenticated provider" is
    exactly what a task materializes: a token, plus one base URL per protocol
    the provider serves (none for a protocol it does not).
    """
    roster = AgentRoster.shared()
    env: dict[str, str] = {}
    for name in providers:
        _set_provider_handle(env, roster.providers[name], _PHANTOM_TOKEN, EXAMPLE_PROVIDER_UPSTREAM)
    return env


def _rebind_claude_default(monkeypatch: pytest.MonkeyPatch, provider: str) -> AgentRoster:
    """Re-bind Claude's default to *provider* through a user agent override.

    The override file goes where users put theirs (``~/.config/terok/agent/
    agents/``, tmp-rooted by the unit conftest) and a fresh roster load merges
    it, so the binding is the one the loader really produces.  The registry
    the wrappers render from is then pointed at it; the roster is returned for
    what else the override changed.
    """
    from terok_executor.roster.loader import _user_agents_dir

    user_dir = _user_agents_dir()
    assert user_dir.is_relative_to(Path.home()), "must stay inside the tmp-rooted HOME"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "claude.yaml").write_text(f"provider:\n  default: {provider}\n")
    roster = AgentRoster.load()
    claude = roster.agents["claude"]
    assert claude.provider_binding is not None
    assert claude.provider_binding.default == provider
    monkeypatch.setitem(AGENTS, "claude", claude)
    return roster


class TestClaudeReadiness:
    """Claude refuses to launch on an endpoint this task holds no credential for.

    terok-ai/terok#1273: without ``terok auth claude`` Claude started anyway,
    fell into its first-run login and met the egress firewall, which reserves
    Anthropic's hosts for the vault — a bare ``ConnectionRefused``, no hint.
    """

    def test_refuses_without_an_authenticated_default(self, harness: Harness) -> None:
        """No handle → exit 1 naming the host-side fix; claude never starts, but
        the invocation is still recorded."""
        harness.stub("claude")
        result = harness.run("claude", env={_CLAUDE_HANDLE: None})
        refusal = "claude has no authenticated anthropic-messages provider ('anthropic')"
        assert result.returncode == 1
        assert refusal in result.stderr
        assert "terok auth claude" in result.stderr
        assert "egress firewall" in result.stderr
        assert harness.calls == []
        assert harness.history.read_text() == "claude\n"

    def test_headless_launch_refuses_too(self, harness: Harness) -> None:
        """The --terok-timeout path refuses the same way (it lands in the task log)."""
        harness.stub("claude")
        result = harness.run(
            "claude", "--terok-timeout", "5", "-p", "do it", env={_CLAUDE_HANDLE: None}
        )
        assert result.returncode == 1
        assert "terok auth claude" in result.stderr
        assert harness.calls == []

    def test_refusal_keeps_the_initial_prompt(self, harness: Harness) -> None:
        """A refused launch leaves the one-shot initial prompt for the next attempt."""
        harness.stub("claude")
        prompt = harness.session(_INITIAL_PROMPT, "PROMPT")
        assert harness.run("claude", env={_CLAUDE_HANDLE: None}).returncode == 1
        assert prompt.read_text() == "PROMPT\n"
        assert not (harness.terok_dir / _INITIAL_PROMPT_CONSUMED).exists()

    def test_help_needs_no_credential(self, harness: Harness) -> None:
        """--help is answered before the pre-flight."""
        harness.stub("claude")
        result = harness.run("claude", "--help", env={_CLAUDE_HANDLE: None})
        assert result.returncode == 0, result.stderr
        assert "VENDOR USAGE" in result.stdout
        assert harness.calls == ["ARGV:--help"]

    def test_ready_selection_routes_without_the_default(self, harness: Harness) -> None:
        """A selected provider serving Claude's protocol carries the launch alone."""
        harness.stub("claude")
        result = harness.run(
            "claude",
            env={
                _CLAUDE_HANDLE: None,
                "TEROK_PROVIDER": "openrouter",
                "TEROK_PROVIDER_OPENROUTER_BASE_ANTHROPIC_MESSAGES": EXAMPLE_PROVIDER_UPSTREAM,
                "TEROK_PROVIDER_OPENROUTER_TOKEN": "phantom",
            },
        )
        assert result.returncode == 0, result.stderr
        assert f"ANTHROPIC_BASE_URL={EXAMPLE_PROVIDER_UPSTREAM}" in result.stdout
        assert harness.calls == ["ARGV:--add-dir /"]

    @pytest.mark.parametrize("provider", ["openrouter", "opper"])
    def test_unauthenticated_selection_is_refused_not_swapped(
        self, harness: Harness, provider: str
    ) -> None:
        """A selected provider that speaks Claude's protocol but holds no
        credential in the task is refused by name, even with the default ready:
        falling back would send the session to a provider nobody selected."""
        assert "anthropic-messages" in AgentRoster.shared().providers[provider].serves
        harness.stub("claude")
        result = harness.run("claude", "--provider", provider)
        assert result.returncode == 1
        assert f"claude's provider '{provider}' (from --provider) is not authenticated" in (
            result.stderr
        )
        assert f"terok auth {provider}" in result.stderr
        assert "egress firewall" in result.stderr
        assert "default endpoint" not in result.stderr
        assert harness.calls == []

    def test_unauthenticated_task_provider_is_refused(self, harness: Harness) -> None:
        """The task-wide selection (``TEROK_PROVIDER``) is held to the same rule,
        and the refusal says where it came from and how to pick another."""
        harness.stub("claude")
        result = harness.run("claude", env={"TEROK_PROVIDER": "openrouter"})
        assert result.returncode == 1
        assert "claude's provider 'openrouter' (from TEROK_PROVIDER)" in result.stderr
        assert "terok auth openrouter" in result.stderr
        assert "claude --provider NAME" in result.stderr
        assert harness.calls == []

    def test_refusal_names_the_binding_agents_auth_entry(self, harness: Harness) -> None:
        """A provider an agent binds is authenticated under that agent's entry:
        selecting Claude's own default points at ``terok auth claude``, not at a
        ``terok auth anthropic`` that does not exist."""
        harness.stub("claude")
        result = harness.run("claude", "--provider", "anthropic", env={_CLAUDE_HANDLE: None})
        assert result.returncode == 1
        assert "claude's provider 'anthropic' (from --provider)" in result.stderr
        assert "terok auth claude" in result.stderr
        assert "terok auth anthropic" not in result.stderr
        assert "does not serve" not in result.stderr
        assert harness.calls == []

    def test_unserved_selection_falls_back_to_a_ready_default(self, harness: Harness) -> None:
        """An authenticated task-wide provider Claude cannot speak to (one aimed
        at another harness) still leaves Claude on its authenticated default."""
        harness.stub("claude")
        result = harness.run(
            "claude", env={"TEROK_PROVIDER": "blablador", **_authenticated("blablador")}
        )
        assert result.returncode == 0, result.stderr
        assert "provider 'blablador' does not serve anthropic-messages" in result.stderr
        assert "using claude's default endpoint" in result.stderr
        assert harness.calls == ["ARGV:--add-dir /"]

    def test_unserved_selection_without_a_default_refuses(self, harness: Harness) -> None:
        """Falling back onto an unauthenticated default is refused, naming that default."""
        harness.stub("claude")
        result = harness.run(
            "claude",
            "--provider",
            "blablador",
            env={_CLAUDE_HANDLE: None, **_authenticated("blablador")},
        )
        assert result.returncode == 1
        assert "provider 'blablador' does not serve anthropic-messages" in result.stderr
        assert "provider ('anthropic')" in result.stderr
        assert harness.calls == []

    def test_exposed_oauth_credential_satisfies_the_guard(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        """Exposed-OAuth mode materializes the very handle the guard checks (real
        token, direct upstream), so it keeps launching Claude."""
        roster = AgentRoster.shared()
        claude_dir = tmp_path / "mounts" / roster.auth_providers["claude"].host_dir_name
        claude_dir.mkdir(parents=True)
        (claude_dir / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "real", "refreshToken": "r"}})
        )
        exposed = _materialize_exposed_providers(roster, tmp_path / "mounts", frozenset({"claude"}))
        assert exposed.get(_CLAUDE_HANDLE)
        harness.stub("claude")
        result = harness.run("claude", env={_CLAUDE_HANDLE: None, **exposed})
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:--add-dir /"]


class TestClaudeAcpReadiness:
    """``terok-claude-acp`` repeats the pre-flight, so ACP cannot bypass it."""

    def test_refuses_without_an_authenticated_default(self, harness: Harness) -> None:
        """No handle → exit 1 before the adapter starts; stdout, the ACP channel,
        stays clean."""
        harness.stub("claude-agent-acp")
        result = harness.acp("claude", env={_CLAUDE_HANDLE: None})
        assert result.returncode == 1
        assert "claude has no authenticated anthropic-messages provider" in result.stderr
        assert "terok auth claude" in result.stderr
        assert "egress firewall" in result.stderr
        assert result.stdout == ""
        assert harness.calls == []

    def test_ready_default_execs_the_adapter(self, harness: Harness) -> None:
        """With the handle the launcher execs claude-agent-acp, arguments intact."""
        harness.stub("claude-agent-acp")
        result = harness.acp("claude", "--some-flag")
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:--some-flag"]

    def test_selected_provider_does_not_stand_in(self, harness: Harness) -> None:
        """The adapter never routes to a selected provider, so a ready
        ``TEROK_PROVIDER`` cannot replace the default it actually uses."""
        harness.stub("claude-agent-acp")
        result = harness.acp(
            "claude",
            env={
                _CLAUDE_HANDLE: None,
                "TEROK_PROVIDER": "openrouter",
                "TEROK_PROVIDER_OPENROUTER_BASE_ANTHROPIC_MESSAGES": EXAMPLE_PROVIDER_UPSTREAM,
                "TEROK_PROVIDER_OPENROUTER_TOKEN": "phantom",
            },
        )
        assert result.returncode == 1
        assert harness.calls == []

    def test_env_builder_handle_satisfies_the_check(self, harness: Harness) -> None:
        """The handle the env builder materializes for Claude's roster default is
        the one the check reads — pinned by running it, not by spelling it."""
        claude = AGENTS["claude"]
        assert claude.provider_binding is not None and claude.provider_binding.default
        built = _authenticated(claude.provider_binding.default)
        harness.stub("claude-agent-acp")
        assert harness.acp("claude", env={_CLAUDE_HANDLE: None, **built}).returncode == 0
        assert harness.calls == ["ARGV:"]

    def test_follows_a_rebound_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user override that re-binds Claude's default moves the check on both
        entry points alike: neither keeps asking for Anthropic's handle."""
        roster = _rebind_claude_default(monkeypatch, "opper")
        harness = Harness(tmp_path)
        harness.stub("claude")
        harness.stub("claude-agent-acp")
        rebound = {_CLAUDE_HANDLE: None, **_authenticated("opper")}
        assert harness.acp("claude", env=rebound).returncode == 0
        assert harness.run("claude", env=rebound).returncode == 0
        refused = harness.acp("claude", env={_CLAUDE_HANDLE: "ready"})
        assert refused.returncode == 1
        assert "provider ('opper')" in refused.stderr
        assert "terok auth claude" in refused.stderr
        shell = harness.run("claude", env={_CLAUDE_HANDLE: "ready"})
        assert shell.returncode == 1
        assert "provider ('opper')" in shell.stderr
        # Unbound now, Anthropic is authenticated under its own name.
        assert "anthropic" in roster.auth_providers
        selected = harness.run("claude", "--provider", "anthropic", env=rebound)
        assert selected.returncode == 1
        assert "terok auth anthropic" in selected.stderr

    def test_skips_the_check_without_task_wrappers(self, harness: Harness) -> None:
        """Without the per-task wrappers (a container that predates the check)
        there is nothing to check against; the adapter starts as it used to."""
        harness.stub("claude-agent-acp")
        result = harness.acp("claude", env={_CLAUDE_HANDLE: None}, task_wrappers=False)
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:"]

    def test_launcher_names_no_provider(self) -> None:
        """The launcher borrows the rendered check instead of hardcoding a
        provider handle that a re-bound default would not match."""
        launcher = (_SCRIPTS_RESOURCE / "terok-claude-acp").read_text()
        assert "TEROK_PROVIDER_" not in launcher
        assert "_terok_claude_ready" in launcher
        assert f"{CONTAINER_TEROK_SHARE_DIR}/{_TASK_WRAPPERS}" in launcher


def _claude_default() -> str:
    """Claude's default provider, as its roster binding names it."""
    binding = AGENTS["claude"].provider_binding
    assert binding is not None and binding.default
    return binding.default


def _skipped_oauth_task() -> dict[str, str | None]:
    """The environment terok leaves a task whose Claude login is unrouted OAuth.

    terok's "skipped" mode: the vault holds a Claude OAuth login that terok
    neither routes to Claude nor exposes to it.  The login's provider handle
    stays — OpenCode, Pi and Toad reach the provider through the vault with
    it — and the flag, set through the package's public name and value as
    terok sets it, marks Claude's default endpoint as having no route for it.
    """
    return {
        **_authenticated(_claude_default()),
        CLAUDE_OAUTH_UNROUTED_ENV: CLAUDE_OAUTH_UNROUTED_ON,
    }


def _select(provider: str, selected_by: str) -> tuple[list[str], dict[str, str]]:
    """The ``claude`` argv and env overlay that select *provider* by *selected_by*.

    *selected_by* is ``--provider`` (one launch) or ``TEROK_PROVIDER`` (the
    task-wide selection).
    """
    if selected_by == "--provider":
        return ["claude", "--provider", provider], {}
    return ["claude"], {"TEROK_PROVIDER": provider}


class TestClaudeUnroutedOAuth:
    """A stored Claude login with no route from Claude's default endpoint.

    In terok's "skipped" OAuth mode the task keeps the login's provider handle
    for its other agents, so the handle alone passes Claude's readiness check,
    and Claude used to start on its default endpoint, go straight to the
    provider's hosts and meet the egress firewall's bare ``ConnectionRefused``.
    terok now flags the case, and the wrapper refuses a launch on that endpoint
    up front.  A selection that serves Claude's protocol still routes the
    launch through the vault, Claude's default provider included.
    """

    def test_refuses_on_the_default_path(self, harness: Harness) -> None:
        """Flag on, handle present → exit 1 naming the cause, both host remedies
        and the selection that works in this task; claude never starts, but the
        invocation is still recorded."""
        harness.stub("claude")
        result = harness.run("claude", env=_skipped_oauth_task())
        assert result.returncode == 1
        assert (
            f"claude cannot use the Claude login stored for its provider ('{_claude_default()}')"
            " on its default endpoint" in result.stderr
        )
        assert _UNROUTED_CAUSE in result.stderr
        assert "egress firewall" in result.stderr
        assert "Store an API key on the host instead:  terok auth claude" in result.stderr
        assert "experimental: true, agent.claude.allow_oauth: true" in result.stderr
        assert _WAY_ROUND in result.stderr
        assert "fresh task" in result.stderr
        assert harness.calls == []
        assert harness.history.read_text() == "claude\n"

    def test_the_named_way_round_launches_in_the_same_task(self, harness: Harness) -> None:
        """The refusal sends the user to ``providers``, which lists the pair of
        Claude and its default provider as ready; selecting that provider then
        launches on the very task that refused the bare command."""
        harness.stub("claude")
        task = _skipped_oauth_task()
        assert _WAY_ROUND in harness.run("claude", env=task).stderr
        result = harness.run("claude", "--provider", _claude_default(), env=task)
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:--add-dir /"]

    @pytest.mark.parametrize("value", [None, "", "0", "true"])
    def test_only_the_exported_value_turns_it_on(self, harness: Harness, value: str | None) -> None:
        """Unset, empty, ``0`` or another truthy spelling leave the launch as it
        was, on the same task: the check compares against the exported value."""
        assert value != CLAUDE_OAUTH_UNROUTED_ON
        harness.stub("claude")
        task = {**_skipped_oauth_task(), CLAUDE_OAUTH_UNROUTED_ENV: value}
        result = harness.run("claude", env=task)
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:--add-dir /"]

    def test_headless_launch_refuses_too(self, harness: Harness) -> None:
        """The --terok-timeout path refuses the same way (it lands in the task log)."""
        harness.stub("claude")
        result = harness.run(
            "claude", "--terok-timeout", "5", "-p", "do it", env=_skipped_oauth_task()
        )
        assert result.returncode == 1
        assert _UNROUTED_CAUSE in result.stderr
        assert harness.calls == []

    def test_refusal_keeps_the_initial_prompt(self, harness: Harness) -> None:
        """A refused launch leaves the one-shot initial prompt for the next attempt."""
        harness.stub("claude")
        prompt = harness.session(_INITIAL_PROMPT, "PROMPT")
        assert harness.run("claude", env=_skipped_oauth_task()).returncode == 1
        assert prompt.read_text() == "PROMPT\n"
        assert not (harness.terok_dir / _INITIAL_PROMPT_CONSUMED).exists()

    def test_help_needs_no_route(self, harness: Harness) -> None:
        """--help is answered before the pre-flight."""
        harness.stub("claude")
        result = harness.run("claude", "--help", env=_skipped_oauth_task())
        assert result.returncode == 0, result.stderr
        assert "VENDOR USAGE" in result.stdout
        assert harness.calls == ["ARGV:--help"]

    @pytest.mark.parametrize("provider", ["openrouter", "opper"])
    @pytest.mark.parametrize("selected_by", ["--provider", "TEROK_PROVIDER"])
    def test_a_routing_selection_still_launches(
        self, harness: Harness, provider: str, selected_by: str
    ) -> None:
        """A selected provider with a credential of its own that serves Claude's
        protocol routes the launch, so the flag on Claude's own login is moot."""
        assert "anthropic-messages" in AgentRoster.shared().providers[provider].serves
        harness.stub("claude")
        routed = _authenticated(provider)
        argv, selection = _select(provider, selected_by)
        result = harness.run(*argv, env={**_skipped_oauth_task(), **routed, **selection})
        assert result.returncode == 0, result.stderr
        base = routed[f"TEROK_PROVIDER_{provider.upper()}_BASE_ANTHROPIC_MESSAGES"]
        assert f"ANTHROPIC_BASE_URL={base}" in result.stdout
        assert harness.calls == ["ARGV:--add-dir /"]

    @pytest.mark.parametrize("flag", [CLAUDE_OAUTH_UNROUTED_ON, None], ids=["flagged", "unflagged"])
    @pytest.mark.parametrize(
        "spell", [str.lower, str.upper, str.capitalize], ids=["lower", "upper", "capitalized"]
    )
    @pytest.mark.parametrize("selected_by", ["--provider", "TEROK_PROVIDER"])
    def test_selecting_the_default_by_name_routes_through_the_vault(
        self, harness: Harness, spell: Callable[[str], str], selected_by: str, flag: str | None
    ) -> None:
        """Naming Claude's default provider, in any case, exports that provider's
        vault handle as Claude's endpoint and bearer, so Claude reaches the
        provider through the vault, not straight to its hosts: the flag, which
        concerns the default endpoint alone, leaves the selection as it was."""
        harness.stub("claude")
        argv, selection = _select(spell(_claude_default()), selected_by)
        task = {**_skipped_oauth_task(), CLAUDE_OAUTH_UNROUTED_ENV: flag, **selection}
        result = harness.run(*argv, env=task)
        assert result.returncode == 0, result.stderr
        assert f"ANTHROPIC_BASE_URL={EXAMPLE_PROVIDER_UPSTREAM}" in result.stdout
        assert f"CLAUDE_CODE_OAUTH_TOKEN={_PHANTOM_TOKEN}" in result.stdout
        assert _UNROUTED_CAUSE not in result.stderr
        assert harness.calls == ["ARGV:--add-dir /"]

    def test_an_unserved_selection_falls_back_into_the_refusal(self, harness: Harness) -> None:
        """A selection Claude cannot speak to (one aimed at another harness)
        leaves Claude on its default endpoint, which the flag marks as having no
        route, so the refusal and its way round apply as on a bare launch."""
        harness.stub("claude")
        task = {
            **_skipped_oauth_task(),
            "TEROK_PROVIDER": "blablador",
            **_authenticated("blablador"),
        }
        result = harness.run("claude", env=task)
        assert result.returncode == 1
        assert "provider 'blablador' does not serve anthropic-messages" in result.stderr
        assert _UNROUTED_CAUSE in result.stderr
        assert _WAY_ROUND in result.stderr
        assert harness.calls == []

    def test_a_missing_login_is_reported_as_missing(self, harness: Harness) -> None:
        """With no handle there is no stored login to be unrouted: the refusal
        asks for a login, whatever the flag says."""
        harness.stub("claude")
        result = harness.run(
            "claude",
            env={_CLAUDE_HANDLE: None, CLAUDE_OAUTH_UNROUTED_ENV: CLAUDE_OAUTH_UNROUTED_ON},
        )
        assert result.returncode == 1
        assert "claude has no authenticated anthropic-messages provider" in result.stderr
        assert _UNROUTED_CAUSE not in result.stderr
        assert harness.calls == []

    @pytest.mark.parametrize(
        ("agent", "selects_the_login", "launched"),
        [
            ("codex", False, "codex"),
            ("vibe", False, "vibe"),
            ("opencode", True, "opencode-provider"),
            ("pi", True, "pi-provider"),
        ],
    )
    def test_other_agents_are_unaffected(
        self, harness: Harness, agent: str, selects_the_login: bool, launched: str
    ) -> None:
        """The flag is Claude's alone: every other agent launches as before,
        including the harnesses that select the very same login."""
        argv = ["--provider", _claude_default()] if selects_the_login else []
        harness.stub(launched)
        result = harness.run(agent, *argv, env=_skipped_oauth_task())
        assert result.returncode == 0, result.stderr
        assert harness.calls == [f"ARGV:{shlex.join(argv)}"]
        assert _UNROUTED_CAUSE not in result.stderr


class TestClaudeAcpUnroutedOAuth:
    """``terok-claude-acp`` refuses an unrouted Claude login the same way."""

    def test_refuses_with_the_flag(self, harness: Harness) -> None:
        """Flag on → exit 1 before the adapter starts, the reason and the host
        remedies on stderr, and no selection offered, since the adapter honours
        none; stdout, the ACP channel, stays clean."""
        harness.stub("claude-agent-acp")
        result = harness.acp("claude", env=_skipped_oauth_task())
        assert result.returncode == 1
        assert _UNROUTED_CAUSE in result.stderr
        assert "terok auth claude" in result.stderr
        assert "allow_oauth: true" in result.stderr
        assert "--provider" not in result.stderr
        assert result.stdout == ""
        assert harness.calls == []

    def test_no_selection_routes_around_it(self, harness: Harness) -> None:
        """The adapter honours no selection, so even a task-wide provider that
        would route the shell wrapper cannot carry it past the flag."""
        harness.stub("claude-agent-acp")
        _, selection = _select("openrouter", "TEROK_PROVIDER")
        routed = _authenticated("openrouter")
        result = harness.acp("claude", env={**_skipped_oauth_task(), **routed, **selection})
        assert result.returncode == 1
        assert _UNROUTED_CAUSE in result.stderr
        assert harness.calls == []

    @pytest.mark.parametrize("value", [None, "0", "true"])
    def test_launches_when_the_flag_is_off(self, harness: Harness, value: str | None) -> None:
        """Without the flag (or with any value but the exported one) the adapter starts."""
        harness.stub("claude-agent-acp")
        result = harness.acp(
            "claude", "--some-flag", env={**_skipped_oauth_task(), CLAUDE_OAUTH_UNROUTED_ENV: value}
        )
        assert result.returncode == 0, result.stderr
        assert harness.calls == ["ARGV:--some-flag"]

    def test_launcher_leaves_the_flag_to_the_rendered_check(self) -> None:
        """The launcher reads no flag of its own: it borrows the per-task check,
        so the template stays the one place that knows the flag."""
        launcher = (_SCRIPTS_RESOURCE / "terok-claude-acp").read_text()
        assert CLAUDE_OAUTH_UNROUTED_ENV not in launcher
        assert "_terok_claude_ready" in launcher
