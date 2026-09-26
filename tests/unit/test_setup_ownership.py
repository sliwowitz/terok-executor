# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Executor owns routes and receipts, while child packages own their setup checks."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from terok_util import SetupCheck, SetupDowngradeError, SetupRequiredError, SetupStatus

from terok_executor import sandbox as setup
from terok_executor.commands import _handle_run, _handle_setup, _handle_start, _handle_uninstall
from terok_executor.container.runner import AgentRunner
from terok_executor.integrations.sandbox import SandboxConfig


@pytest.fixture
def cfg():
    """Use the unit suite's isolated HOME/XDG paths for every artifact."""
    return SandboxConfig()


@pytest.fixture(autouse=True)
def lower_setup():
    """Only the lower package is mocked; executor checks and files remain real."""
    with (
        patch.object(setup, "version", return_value="0.5.0"),
        patch.object(setup, "check_sandbox_setup", return_value=()) as checks,
        patch("terok_executor.integrations.sandbox._handle_sandbox_setup") as install,
    ):
        yield checks, install


def test_missing_setup_probe_is_read_only(cfg) -> None:
    """Checking never generates routes or writes a receipt on a fresh host."""
    checks = setup.check_setup(cfg)
    assert [check.status for check in checks] == [SetupStatus.MISSING, SetupStatus.MISSING]
    assert not cfg.routes_path.exists()
    assert not setup._receipt(cfg).path.exists()


def test_setup_writes_routes_before_child_setup_then_certifies(cfg, lower_setup) -> None:
    """The supervisor sees routes before it starts, and certification follows verification."""
    child_checks, install = lower_setup

    def provision(**kwargs):
        assert kwargs["cfg"] is cfg
        assert cfg.routes_path.is_file()
        assert setup._receipt(cfg).check().status is SetupStatus.MISSING

    install.side_effect = provision
    setup.ensure_sandbox_ready(cfg=cfg)
    assert all(check.status is SetupStatus.READY for check in setup.check_setup(cfg))
    child_checks.assert_called_with(cfg, live=False)


@pytest.mark.parametrize("owner", ["terok-executor", "child"])
def test_full_closure_downgrade_precedes_any_write(cfg, lower_setup, owner) -> None:
    """No route generation or receipt invalidation happens after a downgrade."""
    receipt = setup._receipt(cfg)
    replace(receipt, version="0.6.0" if owner == "terok-executor" else "0.5.0").write()
    before = receipt.path.read_bytes()
    child_checks, install = lower_setup
    if owner == "child":
        child_checks.return_value = (SetupCheck(owner, "receipt", SetupStatus.DOWNGRADE),)
    with (
        patch.object(setup.AgentRoster, "ensure_vault_routes") as routes,
        pytest.raises(SetupDowngradeError),
    ):
        setup.ensure_sandbox_ready(cfg=cfg)
    assert receipt.path.read_bytes() == before
    routes.assert_not_called()
    install.assert_not_called()


def test_failed_child_does_not_leave_upper_receipt(cfg, lower_setup) -> None:
    """Failed provisioning cannot retain a formerly valid executor receipt."""
    setup.ensure_sandbox_ready(cfg=cfg)
    lower_setup[1].side_effect = RuntimeError("lower setup failed")
    with pytest.raises(RuntimeError, match="lower setup failed"):
        setup.ensure_sandbox_ready(cfg=cfg)
    assert setup._receipt(cfg).check().status is SetupStatus.MISSING


def test_partial_setup_cannot_certify_missing_routes(cfg) -> None:
    """Skipping route generation is not permission to certify missing required work."""
    with pytest.raises(SetupRequiredError, match="Vault routes are missing"):
        setup.ensure_sandbox_ready(cfg=cfg, no_vault=True)
    assert not setup._receipt(cfg).path.exists()


def test_skipping_valid_routes_preserves_success(cfg) -> None:
    """Partial setup can retain a complete installation without rewriting routes."""
    setup.ensure_sandbox_ready(cfg=cfg)
    with patch.object(setup.AgentRoster, "ensure_vault_routes") as write_routes:
        setup.ensure_sandbox_ready(cfg=cfg, no_vault=True)
    write_routes.assert_not_called()
    assert setup._receipt(cfg).check().status is SetupStatus.READY


def test_child_pending_check_prevents_certification(cfg, lower_setup) -> None:
    """A child installer returning normally still must verify its own ready state."""
    lower_setup[0].return_value = (SetupCheck("child", "hooks", SetupStatus.MISSING),)
    with pytest.raises(SetupRequiredError, match="child/hooks"):
        setup.ensure_sandbox_ready(cfg=cfg)
    assert setup._receipt(cfg).check().status is SetupStatus.MISSING


def test_failed_upper_receipt_preserves_successful_child(cfg, lower_setup, tmp_path) -> None:
    """Upper certification failure never invalidates completed lower setup."""
    child_receipt = tmp_path / "child-receipt"
    lower_setup[1].side_effect = lambda **kw: child_receipt.write_text("ready")
    with patch.object(setup.SetupReceipt, "write", side_effect=OSError("write failed")):
        with pytest.raises(OSError, match="write failed"):
            setup.ensure_sandbox_ready(cfg=cfg)
    assert child_receipt.read_text() == "ready"
    assert setup._receipt(cfg).check().status is SetupStatus.MISSING


@pytest.mark.parametrize("damage", ["remove", "edit", "invalid-utf8"])
def test_receipt_does_not_hide_missing_or_changed_routes(cfg, damage) -> None:
    """Startup verifies essential owned artifacts, not only their receipt."""
    setup.ensure_sandbox_ready(cfg=cfg)
    if damage == "remove":
        cfg.routes_path.unlink()
    elif damage == "edit":
        cfg.routes_path.write_text("{}")
    else:
        cfg.routes_path.write_bytes(b"\xff")
    receipt, routes = setup.check_setup(cfg)
    assert receipt.status is SetupStatus.READY
    assert routes.status is not SetupStatus.READY


def test_fresh_roster_projection_invalidates_setup_without_writes(cfg) -> None:
    """Current roster configuration is checked rather than a process-wide cached snapshot."""
    setup.ensure_sandbox_ready(cfg=cfg)
    original = cfg.routes_path.read_bytes()
    with patch.object(setup.AgentRoster, "load") as load:
        load.return_value.generate_routes_json.return_value = "{}"
        assert all(check.status is SetupStatus.STALE for check in setup.check_setup(cfg))
    assert cfg.routes_path.read_bytes() == original


def test_invalid_roster_is_a_structured_check(cfg) -> None:
    """Configuration errors remain diagnostic setup results, not false readiness."""
    with patch.object(setup.AgentRoster, "load", side_effect=ValueError("bad provider")):
        assert setup.check_setup(cfg)[1] == SetupCheck(
            "terok-executor", "roster", SetupStatus.INVALID, "bad provider"
        )


def test_live_check_reaches_child(cfg, lower_setup) -> None:
    """Live launch validation propagates through the same downward public API."""
    setup.check_setup(cfg, live=True)
    lower_setup[0].assert_called_once_with(cfg, live=True)


@pytest.mark.parametrize("entry", ["run", "web", "prepared"])
@pytest.mark.parametrize("injected_sandbox", [False, True])
def test_library_guard_precedes_launch_mutation(
    cfg, tmp_path: Path, entry, injected_sandbox
) -> None:
    """Direct library callers cannot bypass setup via CLI-only checks."""
    runner = AgentRunner(sandbox=Mock(config=cfg)) if injected_sandbox else AgentRunner(cfg=cfg)
    checks = (SetupCheck("child", "nft", SetupStatus.MISSING),)
    with (
        patch("terok_executor.container.runner.check_setup", return_value=checks) as check,
        patch("terok_executor.integrations.sandbox.allocate_per_container_resources") as allocate,
        patch("terok_executor.integrations.sandbox.write_sidecar") as sidecar,
        patch("terok_executor.integrations.sandbox.Sandbox") as sandbox,
        pytest.raises(SetupRequiredError),
    ):
        if entry == "run":
            runner.run_headless(agent="claude", repo=None, prompt="test")
        elif entry == "web":
            runner.run_web(repo=None)
        else:
            runner.launch_prepared(
                env={}, volumes=[], image="fixture", command=[], name="fixture", task_dir=tmp_path
            )
    check.assert_called_once_with(cfg, live=True)
    allocate.assert_not_called()
    sidecar.assert_not_called()
    sandbox.assert_not_called()
    if injected_sandbox:
        runner.sandbox.runtime.reserve_port.assert_not_called()


def test_start_fails_live_check_before_constructing_runtime(cfg) -> None:
    """Starting a stopped container requires current executable/interpreter readiness."""
    checks = (SetupCheck("child", "nft", SetupStatus.MISSING),)
    with (
        patch.object(setup, "check_setup", return_value=checks) as check,
        patch("terok_executor.integrations.sandbox.Sandbox") as sandbox,
        pytest.raises(SystemExit) as exc,
    ):
        _handle_start(name="fixture")
    assert exc.value.code == 3
    check.assert_called_once_with(None, live=True)
    sandbox.assert_not_called()


def test_start_preserves_late_typed_setup_error() -> None:
    """The CLI dispatcher retains repair/downgrade exit codes if setup changes mid-start."""
    error = SetupRequiredError("setup changed")
    with (
        patch("terok_executor.commands._setup_verdict_or_exit"),
        patch("terok_executor.integrations.sandbox.Sandbox") as sandbox,
        pytest.raises(SetupRequiredError) as exc,
    ):
        sandbox.return_value.start.side_effect = error
        _handle_start(name="fixture")
    assert exc.value is error


def test_no_preflight_cannot_bypass_setup_downgrade(cfg) -> None:
    """The optional prompt switch does not disable the mandatory safety gate."""
    checks = (SetupCheck("child", "receipt", SetupStatus.DOWNGRADE),)
    with (
        patch.object(setup, "check_setup", return_value=checks) as check,
        patch("terok_executor.commands._preflight_or_exit") as preflight,
        pytest.raises(SystemExit) as exc,
    ):
        _handle_run(agent="claude", no_preflight=True, cfg=cfg)
    assert exc.value.code == 4
    check.assert_called_once_with(cfg, live=True)
    preflight.assert_not_called()


@pytest.mark.parametrize("entry", ["setup", "uninstall"])
def test_image_only_command_still_preflights_downgrade(cfg, entry) -> None:
    """Skipping sandbox provisioning cannot let an older executor mutate newer setup."""
    checks = (SetupCheck("child", "receipt", SetupStatus.DOWNGRADE),)
    with (
        patch.object(setup, "check_setup", return_value=checks),
        patch("terok_executor.commands._build_images_with_banner") as build,
        patch("terok_executor.commands._remove_images") as remove,
        pytest.raises(SetupDowngradeError),
    ):
        (_handle_setup if entry == "setup" else _handle_uninstall)(cfg=cfg, no_sandbox=True)
    build.assert_not_called()
    remove.assert_not_called()
