# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""First-run readiness gate for ``terok-executor run``.

Mandatory prerequisites (podman, sandbox services, container images)
block the launch if unmet after interactive remediation; optional
prerequisites (SSH key, per-agent credentials) print the consequence
of skipping and let the launch proceed.

The check-and-fix surface lives on the [`Preflight`][terok_executor.preflight.Preflight]
class: parameters that thread through every probe (provider, base
image, family, interactivity mode, ``--yes`` short-circuit) are held
once on the instance instead of being repeated in every free-function
signature.  Callers construct ``Preflight(provider="claude").run()``
in production; tests construct it with defaults and call individual
``check_*`` methods.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single prerequisite check."""

    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class Preflight:
    """Holds the parameters that thread through every prerequisite check.

    The orchestrator [`run`][terok_executor.preflight.Preflight.run]
    walks every gate / probe in order and reports a single
    "mandatory-everything-passed" verdict.  Individual probes
    ([`check_podman`][terok_executor.preflight.Preflight.check_podman]
    etc.) are exposed as methods so callers (doctor surfaces, tests)
    can ask narrow questions without paying for the full sweep.
    """

    provider: str
    base_image: str = "ubuntu:24.04"
    family: str | None = None
    interactive: bool = True
    assume_yes: bool = field(default=False)
    credential_set: str = "default"
    """Vault DB namespace to check for stored credentials.  Pairs with
    [`Authenticator.run`][terok_executor.Authenticator.run]'s
    ``credential_set`` — a project running with per-project credentials
    passes its own value so the preflight verdict reflects what the
    runtime will actually load, not the shared host-wide bucket."""

    mounts_dir: Path | None = None
    """Override for the agent-config mount tree.  ``None`` means use the
    global [`paths.mounts_dir`][terok_executor.paths.mounts_dir].  Callers
    that pair a non-``"default"`` ``credential_set`` with a per-project
    mount tree (terok in scope=project mode) must override this too —
    otherwise the captured OAuth credential's post-capture writer drops
    the phantom marker into the wrong tree and the runtime never sees it."""

    # ── Orchestrator ───────────────────────────────────────────────

    def run(self) -> bool:
        """Run every prerequisite check and return ``True`` iff mandatory items pass.

        In non-interactive mode, missing mandatory prerequisites are
        reported once and the return is ``False``; in interactive mode
        each one is offered up as a y/N fix before counting against
        readiness.  Optional items never turn the return into ``False`` —
        their consequence is printed and the launch proceeds.
        """
        print()
        all_ready = True

        if not self._require_podman():
            return False

        self._offer_git()

        if not self._require_sandbox_services():
            all_ready = False

        if not self._require_images():
            all_ready = False

        self._offer_ssh_key()
        self._offer_credentials()
        self._note_shield_bypass()

        if all_ready and self.interactive:
            self._provider_hints()

        print()
        return all_ready

    # ── Mandatory gates ────────────────────────────────────────────

    def _require_podman(self) -> bool:
        """Hard-stop when podman is missing — nothing terok can install it with."""
        r = self.check_podman()
        _print_step(r)
        if not r.ok:
            print(
                "      Install podman first: https://podman.io/docs/installation",
                file=sys.stderr,
            )
            return False
        return True

    def _require_sandbox_services(self) -> bool:
        """Install shield+vault+gate if needed; report remaining gap if not."""
        r = self.check_sandbox_services()
        if not r.ok and self.interactive:
            print(f"  {r.name}... {r.message}")
            if self._confirm("Install shield + vault + gate now?") and _fix_sandbox_services():
                r = self.check_sandbox_services()
        _print_step(r)
        if not r.ok:
            print("      Run: terok-executor setup", file=sys.stderr)
        return r.ok

    def _offer_git(self) -> None:
        """Surface a missing ``git`` binary — gate stops working but launches proceed.

        A container without a gate is just as secure as one with: the gate
        only provides the git push channel.  So this is informational, not
        a remediation step — there's no "fix it for me" prompt because
        installing git is the operator's call (distro-specific package
        manager, root needed).  The consequence is named explicitly so the
        operator can decide whether the gate matters to them.
        """
        r = self.check_git()
        _print_step(r)
        if not r.ok:
            print("      Without git, the host-side git gate is disabled —")
            print("      containers will run without the git push channel.")

    def _require_images(self) -> bool:
        """Build L0+L1 images if missing — mandatory, first-run-heavy."""
        r = self.check_images()
        if not r.ok and self.interactive:
            print(f"  {r.name}... {r.message}")
            if self._confirm("Build container images now?") and _fix_images(
                self.base_image, family=self.family
            ):
                r = self.check_images()
        _print_step(r)
        if not r.ok:
            print("      Run: terok-executor build", file=sys.stderr)
        return r.ok

    # ── Optional offers ────────────────────────────────────────────

    def _offer_ssh_key(self) -> None:
        """Generate a gate-signing SSH key when missing; gate push is the consequence."""
        r = self.check_ssh_key()
        if not r.ok and self.interactive:
            print(f"  {r.name}... {r.message}")
            if self._confirm("Generate an SSH key for gate signing?") and _fix_ssh_key():
                r = self.check_ssh_key()
        _print_step(r)
        if not r.ok:
            print("      Without a gate SSH key, git push via the gate won't work.")

    def _offer_credentials(self) -> None:
        """Authenticate *provider* when missing; login-on-first-turn is the consequence."""
        r = self.check_credentials()
        if not r.ok and self.interactive:
            print(f"  {r.name}... {r.message}")
            if self._confirm(f"Authenticate {self.provider} now?") and _fix_credentials(
                self.provider,
                base_image=self.base_image,
                family=self.family,
                credential_set=self.credential_set,
                mounts_dir=self.mounts_dir,
            ):
                r = self.check_credentials()
        _print_step(r)
        if not r.ok:
            print(
                f"      Without credentials, {self.provider} will prompt for login on first turn."
            )

    def _note_shield_bypass(self) -> None:
        """Surface the bypass override when set — regular shield state is in sandbox-services."""
        from terok_executor.integrations.sandbox import check_environment

        if check_environment().health == "bypass":
            print("\n  Note: shield is in bypass mode — containers have unrestricted network")

    # ── Prerequisite probes ────────────────────────────────────────

    def check_podman(self) -> CheckResult:  # noqa: PLR6301
        """Verify that podman is installed and responds to ``podman version``."""
        if not shutil.which("podman"):
            return CheckResult("podman", False, "not found on PATH")
        try:
            result = subprocess.run(
                ["podman", "version", "--format", "{{.Client.Version}}"],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            return CheckResult("podman", False, f"found but not responding: {exc}")
        if result.returncode != 0:
            detail = (result.stderr or b"").decode(errors="ignore").strip() or "non-zero exit"
            return CheckResult("podman", False, f"found but not responding: {detail}")
        return CheckResult("podman", True, "ok")

    def check_git(self) -> CheckResult:  # noqa: PLR6301
        """Report whether ``git`` is available on the host PATH.

        Informational only: terok-sandbox's git gate uses the host git
        binary to mirror upstream repositories, but a container without a
        gate is functionally identical from a security perspective — the
        gate exists to provide a *push channel*, not to enforce isolation.
        A missing git therefore degrades the workflow (no in-container
        ``git push``) but never blocks a launch.
        """
        if not shutil.which("git"):
            return CheckResult("git", False, "not found on PATH — git gate disabled")
        return CheckResult("git", True, "ok")

    def check_sandbox_services(self) -> CheckResult:  # noqa: PLR6301
        """Roll vault + shield-hooks + gate into a single readiness verdict.

        Treated as a unit because the first two — vault and shield — are
        installed by the sandbox aggregator and fail the same way on a
        fresh host.  Reporting each individually would just clutter the
        first-run summary.

        The gate is special-cased: when it's missing because the host has
        *no way to run it* (no systemd to install units, no git binary to
        drive mirrors), the verdict reports "gate unavailable: …" as a
        contextual note rather than a remediation item.  Operators on
        OpenRC / Gentoo or minimal images see the consequence named
        instead of being told to "run terok-executor setup" — which
        wouldn't fix the underlying gap.
        """
        from terok_executor.integrations.sandbox import (
            GateServerManager,
            SandboxConfig,
            VaultManager,
            check_environment,
        )

        # One SandboxConfig read covers every downstream probe — each of the
        # helpers below would otherwise rebuild it from layered YAML.
        cfg = SandboxConfig()
        vault = VaultManager(cfg)
        gate = GateServerManager(cfg)
        missing: list[str] = []
        if not (vault.is_socket_active() or vault.is_daemon_running()):
            missing.append("vault")
        if check_environment(cfg).health != "ok":
            missing.append("shield hooks")

        gate_running = gate.get_status().mode in ("systemd", "daemon")
        gate_reason: str | None = None
        if not gate_running:
            # Distinguish "operator hasn't installed it yet" (fixable via
            # `terok-executor setup`) from "the host can't host one" (don't
            # send the operator to a setup command that won't help).
            if not shutil.which("git"):
                gate_reason = "no git on PATH"
            elif not gate.is_systemd_available():
                gate_reason = "no systemd — gate has no managed-daemon fallback yet"
            else:
                missing.append("gate")

        if missing:
            return CheckResult("sandbox services", False, f"missing: {', '.join(missing)}")
        if gate_reason:
            return CheckResult(
                "sandbox services",
                True,
                f"shield + vault ready; gate unavailable: {gate_reason}",
            )
        return CheckResult("sandbox services", True, "shield + vault + gate ready")

    def check_images(self) -> CheckResult:
        """Check whether L0+L1 container images exist."""
        from terok_executor.container.build import ImageBuilder

        tag = ImageBuilder(self.base_image).l1_tag()
        try:
            result = subprocess.run(
                ["podman", "image", "exists", tag],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return CheckResult("container images", True, "ready")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return CheckResult("container images", False, "not built")

    def check_credentials(self) -> CheckResult:
        """Check whether credentials are stored for the configured *provider*."""
        from terok_executor.integrations.sandbox import SandboxConfig

        try:
            db = SandboxConfig().open_credential_db()
        except Exception:  # noqa: BLE001
            return CheckResult(
                f"{self.provider} credentials", False, "credential database unavailable"
            )
        try:
            cred = db.load_credential(self.credential_set, self.provider)
        finally:
            db.close()
        if cred:
            return CheckResult(f"{self.provider} credentials", True, "stored")
        return CheckResult(f"{self.provider} credentials", False, "not found")

    def check_ssh_key(self, scope: str = "standalone") -> CheckResult:  # noqa: PLR6301
        """Check whether a gate-signing SSH key exists for *scope*."""
        from terok_executor.integrations.sandbox import SandboxConfig

        try:
            db = SandboxConfig().open_credential_db()
        except Exception:  # noqa: BLE001
            return CheckResult("ssh key", False, "credential database unavailable")
        try:
            keys = db.list_ssh_keys_for_scope(scope)
        finally:
            db.close()
        if keys:
            return CheckResult("ssh key", True, f"{len(keys)} key(s) registered for '{scope}'")
        return CheckResult("ssh key", False, f"none registered for '{scope}'")

    def check_shield(self) -> CheckResult:  # noqa: PLR6301
        """Check whether shield OCI hooks are installed (informational)."""
        from terok_executor.integrations.sandbox import check_environment

        ec = check_environment()
        if ec.health == "ok":
            return CheckResult("shield", True, "active")
        return CheckResult("shield", False, "not installed (containers have unrestricted network)")

    # ── Interactive remediation ────────────────────────────────────

    def _confirm(self, prompt: str) -> bool:
        """Ask a yes/no question; ``self.assume_yes`` short-circuits with True."""
        if self.assume_yes:
            print(f"  {prompt} [Y/n] y")
            return True
        try:
            answer = input(f"  {prompt} [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer in ("", "y", "yes")

    def _provider_hints(self) -> None:
        """Print a hint about authenticating additional tools."""
        from terok_executor.roster import AgentRoster

        roster = AgentRoster.shared()
        others = sorted(name for name in roster.all_names if name != self.provider)
        if others:
            print("\n  Hint: authenticate additional tools with: terok-executor auth <name>")
            print(f"        Available: {', '.join(others)}")


# ── Remediation helpers (free; not parameterised by Preflight state) ───


def _fix_sandbox_services() -> bool:
    """Self-heal missing sandbox services via [`ensure_sandbox_ready`][terok_executor.ensure_sandbox_ready].

    Always per-user — the interactive preflight never escalates to
    sudo behind the operator's back.  ``--root`` is the explicit
    opt-in via ``terok-executor setup``.
    """
    from terok_executor.sandbox import ensure_sandbox_ready

    try:
        ensure_sandbox_ready()
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        print(f"  sandbox setup failed: {exc}", file=sys.stderr)
        return False
    return True


def _fix_images(base_image: str, family: str | None = None) -> bool:
    """Build L0+L1 container images with a friendly first-run banner."""
    from terok_executor.container.build import BuildError, ImageBuilder

    _print_first_build_preamble()
    try:
        ImageBuilder(base_image, family).build_base()
    except BuildError as exc:
        print(f"  Build failed: {exc}", file=sys.stderr)
        return False
    _print_first_build_postamble()
    return True


def _fix_ssh_key(scope: str = "standalone") -> bool:
    """Generate a gate-signing SSH key for *scope* in the credential DB."""
    from terok_executor.integrations.sandbox import SandboxConfig, SSHManager

    try:
        with SSHManager.open_for_config(scope=scope, cfg=SandboxConfig()) as mgr:
            result = mgr.init()
    except Exception as exc:  # noqa: BLE001
        print(f"  SSH key generation failed: {exc}", file=sys.stderr)
        return False
    print(f"  Generated {result['key_type']} key (fingerprint SHA256:{result['fingerprint']}).")
    print(f"  Public line: {result['public_line']}")
    return True


def _fix_credentials(
    provider: str,
    *,
    base_image: str,
    family: str | None = None,
    credential_set: str = "default",
    mounts_dir: Path | None = None,
) -> bool:
    """Run the interactive authentication flow for *provider*.

    *family* threads through to the lazy image resolver so the auth-time
    build matches the family the rest of preflight builds against (an
    unknown base requires the explicit override).  *credential_set*
    selects the vault DB namespace the captured token is stored under;
    *mounts_dir* must override the host-wide default whenever
    *credential_set* is non-default, otherwise the OAuth post-capture
    writer drops phantom markers into the wrong tree.
    """
    from terok_executor.container.build import ImageBuilder
    from terok_executor.credentials.auth import Authenticator
    from terok_executor.credentials.vault_config import write_vault_config
    from terok_executor.paths import mounts_dir as _default_mounts_dir

    # Lazy image resolution — picking API key from the OAuth-or-API-key prompt
    # short-circuits before we ever invoke ensure_default_l1.
    try:
        Authenticator(provider).run(
            None,
            mounts_dir=mounts_dir if mounts_dir is not None else _default_mounts_dir(),
            image=lambda: ImageBuilder(base_image, family=family).ensure_default_l1(),
            credential_set=credential_set,
        )
    except SystemExit:
        return False

    write_vault_config(provider)
    return True


# ── Printing ───────────────────────────────────────────────────────────


def _print_step(result: CheckResult) -> None:
    """Print a preflight check result."""
    marker = "ok" if result.ok else "FAIL"
    print(f"  {result.name:<22} {marker} ({result.message})")


def _print_first_build_preamble() -> None:
    """Announce the first-run image build so the wait doesn't look like a hang."""
    print()
    print("  ─ Building agent images ────────────────────────────────────")
    print("  This is a first-run step and usually takes a few minutes.")
    print("  Subsequent runs reuse the cached layers and start instantly.")
    print("  ────────────────────────────────────────────────────────────")


def _print_first_build_postamble() -> None:
    """Close the build banner once the images are ready."""
    print("  ────────────────────────────────────────────────────────────")
    print("  Images ready.  Next run will skip this step.")
    print()
