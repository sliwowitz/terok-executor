# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Host subprocesses execute the current PATH choice, never a cwd shadow."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest

from terok_executor import commands
from terok_executor.acp.proxy import ACPProxy, AgentBindError
from terok_executor.acp.roster import ACPRoster
from terok_executor.container import build, cache
from terok_executor.container.runner import AgentRunner
from terok_executor.credentials import auth
from terok_executor.preflight import Preflight


@pytest.mark.parametrize(
    ("tool", "action"),
    [
        pytest.param("podman", lambda _: Preflight("claude").check_podman(), id="preflight"),
        pytest.param("podman", lambda _: Preflight("claude").check_images(), id="image-check"),
        pytest.param(
            "podman",
            lambda p: build.build_project_image(
                dockerfile=p / "Dockerfile", context_dir=p, target_tag="test:image"
            ),
            id="build",
        ),
        pytest.param("podman", lambda _: build._tag_image("test:image", "test:alias"), id="tag"),
        pytest.param("podman", lambda _: build._image_exists("test:image"), id="image-exists"),
        pytest.param("podman", lambda _: build.image_agents("test:image"), id="image-agents"),
        pytest.param("podman", lambda _: commands._remove_images("fedora:44"), id="remove-images"),
        pytest.param("git", lambda _: commands._resolve_host_git_identity(), id="git-identity"),
        pytest.param("cp", lambda p: cache._copy_tree(p / "source", p / "dest"), id="copy-cache"),
        pytest.param("git", lambda p: cache._rewrite_origin(p, "test:repo"), id="cache-origin"),
        pytest.param("podman", lambda _: AgentRunner().wait_for_exit("test"), id="wait"),
        pytest.param("podman", lambda _: AgentRunner().logs("test"), id="logs"),
        pytest.param(
            "podman", lambda p: AgentRunner().capture_logs("test", p / "log"), id="capture-logs"
        ),
        pytest.param(
            "podman", lambda _: AgentRunner().stream_logs_process("test"), id="stream-logs"
        ),
        pytest.param("podman", lambda _: AgentRunner._stream_headless("test", 1), id="headless"),
        pytest.param(
            "podman", lambda _: auth._cleanup_existing_container("test"), id="auth-cleanup"
        ),
    ],
)
def test_host_actions_resolve_current_absolute_path(
    tool: str,
    action: Callable[[Path], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated actions follow PATH changes while refusing relative lookup entries."""
    workspace = tmp_path / "workspace"
    bins = [tmp_path / "first-bin", tmp_path / "second-bin"]
    for directory in [workspace, *bins]:
        directory.mkdir()
        executable = directory / tool
        executable.write_text("#!/bin/sh\nexit 99\n")
        executable.chmod(0o755)
    monkeypatch.chdir(workspace)

    def completed(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 0, "0" if kwargs.get("text") else b"0", "")

    run = Mock(side_effect=completed)
    popen = Mock(return_value=Mock(stdout=[]))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    for directory in bins:
        monkeypatch.setenv("PATH", f".:{directory}")
        action(tmp_path)
        calls = [*run.call_args_list, *popen.call_args_list]
        assert calls
        assert all(call.args[0][0] == str(directory / tool) for call in calls)
        run.reset_mock()
        popen.reset_mock()


def test_missing_host_tools_preserve_optional_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing PATH tools still mean unavailable images, identity, and log capture."""
    monkeypatch.setenv("PATH", "")
    assert not Preflight("claude").check_images().ok
    assert build.image_agents("test:image") == set()
    assert commands._resolve_host_git_identity() == (None, None)
    dest = tmp_path / "old-log"
    dest.touch()
    assert not AgentRunner().capture_logs("test", dest)
    assert not dest.exists()

    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("copied without cp")
    cache._copy_tree(source, tmp_path / "dest")
    assert (tmp_path / "dest" / "file").read_text() == "copied without cp"


def test_acp_wrapper_selects_host_podman(
    fake_podman: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACP argv selects the host runtime without resolving the container-side wrapper."""
    roster = ACPRoster(container_name="test", image_id="image", sandbox=Mock())
    assert roster.wrapper_argv("claude") == [
        str(fake_podman),
        "exec",
        "-i",
        "test",
        "terok-claude-acp",
    ]
    monkeypatch.setenv("PATH", "")
    assert asyncio.run(roster.warm("claude")) == ()
    proxy = ACPProxy(roster=roster)
    bind = proxy._bind("claude", "test-model")
    with pytest.raises(AgentBindError, match="podman"):
        asyncio.run(bind)
