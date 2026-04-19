# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentRunner facade."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from terok_executor.container.runner import AgentRunner, _generate_task_id, _resolve_repo


class TestResolveRepo:
    """Verify repo argument classification."""

    def test_local_dir(self, tmp_path: Path) -> None:
        code_repo, local_path = _resolve_repo(str(tmp_path))
        assert code_repo is None
        assert local_path == tmp_path

    def test_git_url_ssh(self) -> None:
        code_repo, local_path = _resolve_repo("git@github.com:user/repo.git")
        assert code_repo == "git@github.com:user/repo.git"
        assert local_path is None

    def test_git_url_https(self) -> None:
        code_repo, local_path = _resolve_repo("https://github.com/user/repo.git")
        assert code_repo == "https://github.com/user/repo.git"
        assert local_path is None

    def test_nonexistent_local_path_exits(self) -> None:
        with pytest.raises(SystemExit, match="not found"):
            _resolve_repo("./nonexistent-dir-xyz")


class TestTaskId:
    """Verify task ID generation."""

    def test_length(self) -> None:
        assert len(_generate_task_id()) == 12

    def test_unique(self) -> None:
        ids = {_generate_task_id() for _ in range(100)}
        assert len(ids) == 100


class TestAgentRunner:
    """Verify AgentRunner construction and delegation."""

    def test_default_construction(self) -> None:
        runner = AgentRunner()
        assert runner._base_image == "ubuntu:24.04"

    def test_custom_base_image(self) -> None:
        runner = AgentRunner(base_image="nvidia/cuda:12.4")
        assert runner._base_image == "nvidia/cuda:12.4"

    def test_lazy_roster(self) -> None:
        runner = AgentRunner()
        reg = runner.roster
        assert "claude" in reg.agent_names

    def test_mismatched_sandbox_and_runtime_rejected(self) -> None:
        """Passing a sandbox and a runtime from different backend types raises."""
        from terok_sandbox import NullRuntime, PodmanRuntime, Sandbox

        rt_a = PodmanRuntime()
        rt_b = NullRuntime()
        with pytest.raises(ValueError, match="same backend instance"):
            AgentRunner(sandbox=Sandbox(runtime=rt_a), runtime=rt_b)

    def test_mismatched_same_type_different_instance_rejected(self) -> None:
        """The check is identity-based — two NullRuntimes are still different backends."""
        from terok_sandbox import NullRuntime, Sandbox

        rt_a = NullRuntime()
        rt_b = NullRuntime()
        with pytest.raises(ValueError, match="same backend instance"):
            AgentRunner(sandbox=Sandbox(runtime=rt_a), runtime=rt_b)

    def test_matching_sandbox_and_runtime_accepted(self) -> None:
        """Passing a sandbox and a runtime pointing at the same backend is fine."""
        from terok_sandbox import NullRuntime, Sandbox

        rt = NullRuntime()
        runner = AgentRunner(sandbox=Sandbox(runtime=rt), runtime=rt)
        assert runner.runtime is rt
        assert runner.sandbox.runtime is rt

    # test_shared_mounts_from_roster, test_base_env_has_essentials,
    # test_base_env_opencode_vars moved to test_env_builder.py

    def test_run_headless_delegates_to_sandbox_run(self, tmp_path: Path) -> None:
        """Verify headless run delegates to sandbox.run() with correct RunSpec."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            cname = runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="Fix the bug",
                follow=False,
            )

        assert cname.startswith("terok-executor-")
        sandbox.run.assert_called_once()
        spec = sandbox.run.call_args[0][0]
        assert spec.image == "terok-l1-cli:test"
        assert any("/home/dev/.terok" in v.container_path for v in spec.volumes)

    def test_restricted_mode_sets_unrestricted_false(self, tmp_path: Path) -> None:
        """Restricted mode passes unrestricted=False in RunSpec."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless(
                "claude", str(tmp_path), prompt="test", follow=False, unrestricted=False
            )

        spec = sandbox.run.call_args[0][0]
        assert spec.unrestricted is False
        # TEROK_UNRESTRICTED should NOT be in env
        assert "TEROK_UNRESTRICTED" not in spec.env

    def test_unrestricted_mode_sets_env(self, tmp_path: Path) -> None:
        """Unrestricted spec includes TEROK_UNRESTRICTED=1."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless(
                "claude", str(tmp_path), prompt="test", follow=False, unrestricted=True
            )

        spec = sandbox.run.call_args[0][0]
        assert spec.unrestricted is True
        assert spec.env.get("TEROK_UNRESTRICTED") == "1"

    def test_gpu_flag_sets_gpu_enabled(self, tmp_path: Path) -> None:
        """GPU flag propagates to RunSpec.gpu_enabled."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False, gpu=True)

        spec = sandbox.run.call_args[0][0]
        assert spec.gpu_enabled is True

    def test_memory_limit_propagates(self, tmp_path: Path) -> None:
        """Memory limit flows through to RunSpec.memory_limit."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False, memory="4g")

        spec = sandbox.run.call_args[0][0]
        assert spec.memory_limit == "4g"

    def test_cpu_limit_propagates(self, tmp_path: Path) -> None:
        """CPU limit flows through to RunSpec.cpu_limit."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False, cpus="2.0")

        spec = sandbox.run.call_args[0][0]
        assert spec.cpu_limit == "2.0"

    def test_resource_limits_default_none(self, tmp_path: Path) -> None:
        """Resource limits default to None when not specified."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False)

        spec = sandbox.run.call_args[0][0]
        assert spec.memory_limit is None
        assert spec.cpu_limit is None

    def test_run_interactive_command(self, tmp_path: Path) -> None:
        """Interactive mode includes init-ssh-and-repo.sh in command."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_interactive("claude", str(tmp_path))

        spec = sandbox.run.call_args[0][0]
        cmd_str = " ".join(spec.command)
        assert "init-ssh-and-repo.sh" in cmd_str
        assert "__CLI_READY__" in cmd_str

    def test_run_web_publishes_port(self, tmp_path: Path) -> None:
        """Web mode includes port publishing in extra_args."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_web(str(tmp_path), port=9999)

        spec = sandbox.run.call_args[0][0]
        assert "-p" in spec.extra_args
        idx = list(spec.extra_args).index("-p")
        assert "9999:8080" in spec.extra_args[idx + 1]

    def test_run_web_auto_allocates_port(self, tmp_path: Path) -> None:
        """Web mode reserves a port via ``runtime.reserve_port`` when none is given."""
        from unittest.mock import MagicMock as _Mock

        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        reservation = _Mock()
        reservation.port = 12345
        reservation.__enter__.return_value = reservation
        reservation.__exit__.return_value = None
        runtime_mock = _Mock()
        runtime_mock.reserve_port.return_value = reservation

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch.object(AgentRunner, "runtime", runtime_mock),
        ):
            runner.run_web(str(tmp_path))  # no port= arg

        runtime_mock.reserve_port.assert_called_once()
        spec = sandbox.run.call_args[0][0]
        assert "-p" in spec.extra_args
        idx = list(spec.extra_args).index("-p")
        assert "12345:8080" in spec.extra_args[idx + 1]

    def test_hooks_passed_to_sandbox_run(self, tmp_path: Path) -> None:
        """Lifecycle hooks are forwarded to sandbox.run()."""
        from terok_sandbox import LifecycleHooks

        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        hooks = LifecycleHooks(pre_start=lambda: None)

        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False, hooks=hooks)

        assert sandbox.run.call_args.kwargs["hooks"] is hooks

    def test_lazy_sandbox_init(self) -> None:
        runner = AgentRunner()
        # Access sandbox property — should create a default Sandbox
        with patch("terok_sandbox.Sandbox") as mock_cls:
            mock_cls.return_value = _mock_sandbox()
            s = runner.sandbox
            assert s is not None

    def test_gpu_config_error_becomes_build_error(self, tmp_path: Path) -> None:
        """GpuConfigError from sandbox.run() is wrapped as BuildError."""
        from terok_sandbox import GpuConfigError

        from terok_executor.container.build import BuildError

        sandbox = _mock_sandbox()
        sandbox.run.side_effect = GpuConfigError("CDI broken")
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(BuildError, match="CDI broken"),
        ):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False)

    def test_shared_mount_must_be_absolute(self, tmp_path: Path) -> None:
        """Relative shared_mount is rejected with SystemExit."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(SystemExit, match="absolute path"),
        ):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=tmp_path / "ipc",
                shared_mount="relative/path",
            )

    def test_shared_mount_rejects_colon(self, tmp_path: Path) -> None:
        """shared_mount containing ':' is rejected (volume spec injection)."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(SystemExit, match="':'"),
        ):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=tmp_path / "ipc",
                shared_mount="/data:ro",
            )

    def test_shared_dir_file_rejected(self, tmp_path: Path) -> None:
        """shared_dir that exists as a file is rejected."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        existing_file = tmp_path / "not-a-dir"
        existing_file.touch()
        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            pytest.raises(SystemExit, match="exists as a file"),
        ):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=existing_file,
            )

    def test_shared_dir_in_container_env(self, tmp_path: Path) -> None:
        """shared_dir kwarg produces TEROK_SHARED_DIR and a volume mount."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        shared = tmp_path / "ipc"
        with patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"):
            runner.run_headless(
                "claude",
                str(tmp_path),
                prompt="test",
                follow=False,
                shared_dir=shared,
                shared_mount="/data",
            )

        spec = sandbox.run.call_args[0][0]
        assert spec.env["TEROK_SHARED_DIR"] == "/data"
        assert any(v.host_path == shared and v.container_path == "/data" for v in spec.volumes)


class TestLaunchPrepared:
    """Verify the library-level ``launch_prepared`` task primitive.

    ``launch_prepared`` is the public entry point terok uses to hand a
    caller-assembled env and volumes to the sandbox without reimplementing
    ``RunSpec`` construction.  Exercises the mapping from method args to
    :class:`~terok_sandbox.RunSpec` fields.
    """

    def test_returns_container_name(self, tmp_path: Path) -> None:
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        cname = runner.launch_prepared(
            env={"TEROK_TASK": "demo"},
            volumes=[],
            image="terok-l1-cli:test",
            command=["bash", "-lc", "echo hi"],
            name="terok-demo",
            task_dir=tmp_path,
        )

        assert cname == "terok-demo"
        sandbox.run.assert_called_once()

    def test_builds_runspec_from_args(self, tmp_path: Path) -> None:
        """Each kwarg maps to the matching RunSpec field."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        runner.launch_prepared(
            env={"FOO": "bar"},
            volumes=[],
            image="terok-l1-cli:test",
            command=["bash"],
            name="terok-x",
            task_dir=tmp_path,
            gpu=True,
            memory="4g",
            cpus="2.0",
            unrestricted=False,
            sealed=True,
            extra_args=["-p", "127.0.0.1:8080:8080"],
        )

        spec = sandbox.run.call_args[0][0]
        assert spec.container_name == "terok-x"
        assert spec.image == "terok-l1-cli:test"
        assert spec.env == {"FOO": "bar"}
        assert spec.command == ("bash",)
        assert spec.task_dir == tmp_path
        assert spec.gpu_enabled is True
        assert spec.memory_limit == "4g"
        assert spec.cpu_limit == "2.0"
        assert spec.unrestricted is False
        assert spec.sealed is True
        assert spec.extra_args == ("-p", "127.0.0.1:8080:8080")

    def test_sealed_defaults_false(self, tmp_path: Path) -> None:
        """Sealed isolation is opt-in — the default must be shared-mode."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        runner.launch_prepared(
            env={},
            volumes=[],
            image="img",
            command=[],
            name="c",
            task_dir=tmp_path,
        )

        spec = sandbox.run.call_args[0][0]
        assert spec.sealed is False

    def test_hooks_forwarded(self, tmp_path: Path) -> None:
        """Lifecycle hooks are passed through to sandbox.run()."""
        from terok_sandbox import LifecycleHooks

        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)
        hooks = LifecycleHooks(pre_start=lambda: None)

        runner.launch_prepared(
            env={},
            volumes=[],
            image="img",
            command=[],
            name="c",
            task_dir=tmp_path,
            hooks=hooks,
        )

        assert sandbox.run.call_args.kwargs["hooks"] is hooks

    def test_gpu_config_error_becomes_build_error(self, tmp_path: Path) -> None:
        """GpuConfigError from sandbox.run() surfaces as BuildError — same
        translation as the run_* entry points, so terok sees one failure type."""
        from terok_sandbox import GpuConfigError

        from terok_executor.container.build import BuildError

        sandbox = _mock_sandbox()
        sandbox.run.side_effect = GpuConfigError("no CDI")
        runner = AgentRunner(sandbox=sandbox)

        with pytest.raises(BuildError, match="no CDI"):
            runner.launch_prepared(
                env={},
                volumes=[],
                image="img",
                command=[],
                name="c",
                task_dir=tmp_path,
                gpu=True,
            )


class TestWaitForExit:
    """Verify task-level wait facade."""

    def test_returns_container_exit_code(self) -> None:
        """Happy path: ``podman wait`` reports the container's exit code."""
        runner = AgentRunner(sandbox=_mock_sandbox())
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="0\n", stderr="")
        with patch("subprocess.run", return_value=completed) as run_mock:
            assert runner.wait_for_exit("terok-x") == 0
        run_mock.assert_called_once()
        # Call shape: ["podman", "wait", "terok-x"] with timeout=None
        args, kwargs = run_mock.call_args
        assert args[0] == ["podman", "wait", "terok-x"]
        assert kwargs["timeout"] is None

    def test_returns_exit_code_124_distinctly(self) -> None:
        """A container that legitimately exits 124 is returned as 124 —
        not conflated with the wait-timeout signal."""
        runner = AgentRunner(sandbox=_mock_sandbox())
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="124\n", stderr="")
        with patch("subprocess.run", return_value=completed):
            assert runner.wait_for_exit("terok-x") == 124

    def test_timeout_raises(self) -> None:
        """``subprocess.TimeoutExpired`` surfaces as :class:`TimeoutError`
        so callers can signal the real exit code separately."""
        runner = AgentRunner(sandbox=_mock_sandbox())
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["podman", "wait"], timeout=30.0),
        ):
            with pytest.raises(TimeoutError, match="did not exit within 30"):
                runner.wait_for_exit("terok-x", timeout=30.0)

    def test_podman_wait_failure_raises(self) -> None:
        """Non-zero returncode from ``podman wait`` (e.g. unknown container)
        raises ``RuntimeError`` — never impersonated as a container exit code."""
        runner = AgentRunner(sandbox=_mock_sandbox())
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=125,
            stdout="",
            stderr="Error: no such container terok-x\n",
        )
        with patch("subprocess.run", return_value=completed):
            with pytest.raises(
                RuntimeError, match=r"podman wait .* failed .*returncode=125.*no such container"
            ):
                runner.wait_for_exit("terok-x")

    def test_unexpected_output_raises(self) -> None:
        """Non-numeric stdout raises ``RuntimeError`` with diagnostic context
        instead of leaking an obscure ``ValueError`` from ``int(...)``."""
        runner = AgentRunner(sandbox=_mock_sandbox())
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-a-number\n", stderr=""
        )
        with patch("subprocess.run", return_value=completed):
            with pytest.raises(RuntimeError, match=r"returned unexpected output.*not-a-number"):
                runner.wait_for_exit("terok-x")


class TestLogs:
    """Task-level log retrieval for the 'just show me what ran' case."""

    def test_one_shot_returns_combined_output(self) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="hello\n", stderr="")
        with patch("subprocess.run", return_value=completed) as run_mock:
            out = runner.logs("terok-x", tail=50, timestamps=True, since="1h")
        assert "hello" in out
        args, _ = run_mock.call_args
        assert args[0] == [
            "podman",
            "logs",
            "--timestamps",
            "--tail",
            "50",
            "--since",
            "1h",
            "terok-x",
        ]

    def test_podman_failure_raises(self) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        completed = subprocess.CompletedProcess(
            args=[], returncode=125, stdout="", stderr="Error: no such container\n"
        )
        with patch("subprocess.run", return_value=completed):
            with pytest.raises(RuntimeError, match=r"returncode=125.*no such container"):
                runner.logs("terok-x")


class TestCaptureLogs:
    """Archival path: stream stdout to a file, succeed-or-clean-up."""

    def test_writes_to_file_on_success(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        dest = tmp_path / "container.log"
        completed = subprocess.CompletedProcess(args=[], returncode=0, stderr=b"")
        with patch("subprocess.run", return_value=completed):
            assert runner.capture_logs("terok-x", dest) is True

    def test_failure_removes_dest(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        dest = tmp_path / "container.log"
        completed = subprocess.CompletedProcess(
            args=[], returncode=125, stderr=b"Error: no such container\n"
        )
        with patch("subprocess.run", return_value=completed):
            assert runner.capture_logs("terok-x", dest) is False
        assert not dest.exists()

    def test_timeout_returns_false(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        dest = tmp_path / "container.log"
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["podman"], timeout=60.0),
        ):
            assert runner.capture_logs("terok-x", dest) is False
        assert not dest.exists()

    def test_missing_podman_returns_false(self, tmp_path: Path) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        dest = tmp_path / "container.log"
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert runner.capture_logs("terok-x", dest) is False
        assert not dest.exists()


class TestStreamLogsProcess:
    """Streaming path: hand the caller a ``Popen`` they can select() on."""

    def test_returns_popen_with_stdout_pipe(self) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        fake_proc = Mock(spec=subprocess.Popen)
        with patch("subprocess.Popen", return_value=fake_proc) as popen_mock:
            proc = runner.stream_logs_process("terok-x", follow=True, tail=100)
        assert proc is fake_proc
        args, kwargs = popen_mock.call_args
        assert args[0] == ["podman", "logs", "-f", "--tail", "100", "terok-x"]
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE

    def test_merge_stderr_folds_streams(self) -> None:
        runner = AgentRunner(sandbox=_mock_sandbox())
        with patch("subprocess.Popen") as popen_mock:
            runner.stream_logs_process("terok-x", merge_stderr=True)
        assert popen_mock.call_args.kwargs["stderr"] is subprocess.STDOUT

    def test_propagates_file_not_found(self) -> None:
        """Missing podman surfaces as FileNotFoundError so callers can
        render a clear 'podman not installed' message in the TUI."""
        runner = AgentRunner(sandbox=_mock_sandbox())
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                runner.stream_logs_process("terok-x", follow=True)


class TestBuildLogsCmd:
    """Flag assembly helper — shared between all three log entry points."""

    def test_all_flags(self) -> None:
        from terok_executor.container.runner import _build_logs_cmd

        cmd = _build_logs_cmd("terok-x", follow=True, tail=10, timestamps=True, since="30m")
        assert cmd == [
            "podman",
            "logs",
            "-f",
            "--timestamps",
            "--tail",
            "10",
            "--since",
            "30m",
            "terok-x",
        ]

    def test_defaults_produce_minimal_cmd(self) -> None:
        from terok_executor.container.runner import _build_logs_cmd

        assert _build_logs_cmd("terok-x") == ["podman", "logs", "terok-x"]


class TestGateIntegration:
    """Verify gate wiring in AgentRunner."""

    def test_setup_gate_calls_sandbox(self) -> None:
        sandbox = _mock_sandbox()
        sandbox.ensure_gate.return_value = None
        sandbox.create_token.return_value = "tok123"
        sandbox.gate_url.return_value = "http://tok123@host:9418/repo"
        runner = AgentRunner(sandbox=sandbox)

        with patch("terok_sandbox.GitGate") as mock_gate_cls:
            mock_gate = Mock()
            mock_gate_cls.return_value = mock_gate
            url = runner._setup_gate("git@github.com:user/repo.git", "task1")

        mock_gate.sync.assert_called_once()
        sandbox.ensure_gate.assert_called_once()
        sandbox.create_token.assert_called_once()
        assert url == "http://tok123@host:9418/repo"

    def test_gate_true_with_git_url_uses_gate(self) -> None:
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch.object(
                runner, "_setup_gate", return_value="http://tok@host:9418/repo"
            ) as mock_gate,
        ):
            runner.run_headless(
                "claude",
                "git@github.com:user/repo.git",
                prompt="test",
                gate=True,
                follow=False,
            )

        # Verify _setup_gate was called with the repo URL
        mock_gate.assert_called_once()
        assert mock_gate.call_args[0][0] == "git@github.com:user/repo.git"

        # Verify the gate URL ended up in the RunSpec env as CODE_REPO
        spec = sandbox.run.call_args[0][0]
        assert spec.env.get("CODE_REPO") == "http://tok@host:9418/repo"

    def test_gate_false_skips_gate(self) -> None:
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch.object(runner, "_setup_gate") as mock_gate,
        ):
            runner.run_headless(
                "claude",
                "git@github.com:user/repo.git",
                prompt="test",
                gate=False,
                follow=False,
            )

        mock_gate.assert_not_called()
        # CODE_REPO should be the raw URL
        spec = sandbox.run.call_args[0][0]
        assert spec.env.get("CODE_REPO") == "git@github.com:user/repo.git"


class TestVaultEnv:
    """Verify vault integration (now via env_builder).

    Detailed token injection tests are in test_env_builder.py.
    These tests verify the runner delegates correctly.
    """

    def test_vault_not_running_no_tokens_in_run(self, tmp_path: Path) -> None:
        """When vault is not running, headless run has no TEROK_TOKEN_BROKER_PORT."""
        sandbox = _mock_sandbox()
        runner = AgentRunner(sandbox=sandbox)

        with (
            patch.object(runner, "_ensure_images", return_value="terok-l1-cli:test"),
            patch("terok_sandbox.is_vault_socket_active", return_value=False),
            patch("terok_sandbox.is_vault_running", return_value=False),
        ):
            runner.run_headless("claude", str(tmp_path), prompt="test", follow=False)

        spec = sandbox.run.call_args[0][0]
        assert "TEROK_TOKEN_BROKER_PORT" not in spec.env


class TestCommandRegistry:
    """Verify the command registry is well-formed."""

    def test_all_commands_have_handlers(self) -> None:
        from terok_executor.commands import COMMANDS

        for cmd in COMMANDS:
            assert cmd.handler is not None, f"Command '{cmd.name}' has no handler"

    def test_run_command_has_agent_arg(self) -> None:
        from terok_executor.commands import RUN_COMMAND

        arg_names = [a.name for a in RUN_COMMAND.args]
        assert "agent" in arg_names

    def test_commands_exported_from_package(self) -> None:
        from terok_executor import AGENT_COMMANDS

        assert len(AGENT_COMMANDS) >= 5  # run, auth, agents, build, ls, stop


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_sandbox() -> Mock:
    """Create a mock Sandbox with the right interface."""
    from terok_sandbox import SandboxConfig

    sandbox = Mock()
    sandbox.config = SandboxConfig()
    sandbox.pre_start_args.return_value = []
    sandbox.stream_logs.return_value = True
    sandbox.wait_for_exit.return_value = 0
    return sandbox
