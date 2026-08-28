# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Run the rendered agent wrappers under bash against stub binaries.

The string-level tests check what the template *says*; these check what the
shell *does* on the paths a container user actually hits: the ``--help``
framing, ``--terok-new-session``, and the resume hint that replaced the old
fast-fail retry.  Every container path the template bakes in is re-rooted
under ``tmp_path`` before sourcing.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from terok_executor.provider.wrappers import generate_all_wrappers
from tests.constants import CONTAINER_TEROK_DIR, CONTAINER_TEROK_SHARE_DIR

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")

_STUB = """#!/usr/bin/env bash
printf 'ARGV:%s\\n' "$*" | tee -a "$STUB_LOG"
echo "VENDOR USAGE"
env  # so a leaked (exported) shell variable would show in the launched process
exit "${STUB_RC:-0}"
"""


class Harness:
    """A tmp-rooted stand-in for the container: wrappers, stub agents, state dir."""

    def __init__(self, tmp_path: Path) -> None:
        self.terok_dir = tmp_path / "terok"
        self.bin = tmp_path / "bin"
        self.home = tmp_path / "home"
        for d in (self.terok_dir, self.bin, self.home):
            d.mkdir()
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
        path.write_text(_STUB)
        path.chmod(0o755)

    def session(self, filename: str, session_id: str) -> Path:
        """Record *session_id* the way the agent's capture would."""
        path = self.terok_dir / filename
        path.write_text(f"{session_id}\n")
        return path

    def run(self, *argv: str, rc: int = 0) -> subprocess.CompletedProcess[str]:
        """Invoke *argv* through the sourced wrappers; the stub exits with *rc*."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(("TEROK_", "CLAUDE"))}
        env.update(
            # Hermetic PATH: the stubs plus coreutils only, so the container's
            # real launchers (opencode-provider, codex-provider, …) in
            # /usr/local/bin cannot shadow a stub the wrapper execs.
            PATH=f"{self.bin}:/usr/bin:/bin",
            HOME=str(self.home),
            STUB_LOG=str(self.log),
            STUB_RC=str(rc),
            # Satisfy the native-agent readiness guard (codex, vibe) without a
            # live vault: it only checks that this handle is non-empty.
            TEROK_PROVIDER_OPENAI_BASE_OPENAI_RESPONSES="ready",
        )
        command = f"source {shlex.quote(str(self.wrappers))}; {shlex.join(argv)}"
        return subprocess.run(
            ["bash", "-c", command], env=env, capture_output=True, text=True, check=False
        )

    @property
    def calls(self) -> list[str]:
        """Argument vectors the stub was invoked with, in order."""
        return self.log.read_text().splitlines() if self.log.exists() else []


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    """Fresh wrapper harness per test."""
    return Harness(tmp_path)


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
