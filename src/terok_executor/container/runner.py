# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Launches AI agents in hardened Podman containers.

Builds the environment, prepares agent config, and launches a hardened
Podman container with the requested AI agent.  Three launch modes:

- **Headless**: fire-and-forget with a prompt (``run_headless``)
- **Interactive**: user logs in, agent is ready (``run_interactive``)
- **Web**: toad served over HTTP (``run_web``)

All user config is runtime (env vars + volumes) — no L2 image build needed.
Gate is on by default (safe-by-default egress control).
"""

from __future__ import annotations

import logging
import shlex
import sys
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from terok_executor._util import detect_host_timezone
from terok_executor.integrations.sandbox import SandboxConfig, Sharing, VolumeSpec

from .build import BuildError, ImageBuilder

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Mapping

    from terok_executor.integrations.sandbox import (
        ContainerRuntime,
        LifecycleHooks,
        PerContainerResources,
        Sandbox,
    )
    from terok_executor.roster.loader import AgentRoster

_logger = logging.getLogger(__name__)


class AgentRunner:
    """Composes sandbox + agent config into a single container launch.

    All three run methods follow the same flow:

    1. Ensure L0+L1 images exist (build if missing)
    2. Prepare agent-config directory (wrapper, instructions, prompt)
    3. Assemble environment variables and volume mounts
    4. Optionally set up gate (mirror repo, create token)
    5. Launch container via podman
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        sandbox: Sandbox | None = None,
        runtime: ContainerRuntime | None = None,
        roster: AgentRoster | None = None,
        base_image: str = "fedora:44",
        family: str | None = None,
        cfg: SandboxConfig | None = None,
    ) -> None:
        if sandbox is not None and runtime is not None and sandbox.runtime is not runtime:
            # Split backends would mean port reservations on one runtime
            # get used by containers launched via a different runtime —
            # a subtle class of bug (``run_web`` vs ``sandbox.run``) that
            # is easier to rule out at construction time.
            raise ValueError(
                "AgentRunner: sandbox.runtime and runtime must be the same backend "
                "instance; pass only one or ensure sandbox was constructed with runtime"
            )
        self._base_image = base_image
        self._family = family
        self._sandbox: Sandbox | None = sandbox
        self._runtime: ContainerRuntime | None = runtime
        self._roster: AgentRoster | None = roster
        self._cfg: SandboxConfig | None = cfg

    # ------------------------------------------------------------------
    # Properties (lazy init)
    # ------------------------------------------------------------------

    @property
    def sandbox(self) -> Sandbox:
        """Lazy-init sandbox facade.

        When an explicit ``runtime`` was supplied but no ``sandbox``, the
        sandbox is constructed with that same runtime so the two share
        one backend instance.
        """
        if self._sandbox is None:
            from terok_executor.integrations.sandbox import Sandbox

            self._sandbox = Sandbox(config=self._cfg, runtime=self._runtime)
        return self._sandbox

    @property
    def runtime(self) -> ContainerRuntime:
        """Return the container runtime used for observation and lifecycle.

        Falls back to the sandbox's runtime when the caller did not
        supply one — keeps the two in sync by construction.
        """
        if self._runtime is None:
            self._runtime = self.sandbox.runtime
        return self._runtime

    @property
    def roster(self) -> AgentRoster:
        """Lazy-init agent roster."""
        if self._roster is None:
            from terok_executor.roster import AgentRoster as _Roster

            self._roster = _Roster.shared()
        return self._roster

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_headless(
        self,
        agent: str,
        repo: str,
        *,
        prompt: str,
        branch: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        timeout: int = 1800,
        gate: bool = True,
        name: str | None = None,
        follow: bool = False,
        unrestricted: bool = True,
        gpu: bool = False,
        memory: str | None = None,
        cpus: str | None = None,
        hooks: LifecycleHooks | None = None,
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
        timezone: str | None = None,
        project_id: str = "",
        task_id: str = "",
        dossier_path: Path | str | None = None,
    ) -> str:
        """Launch a headless agent run. Returns container name.

        The agent executes the *prompt* against *repo* (local path or git URL)
        and exits when done or when *timeout* is reached.  Set *follow=True*
        to block until the agent finishes (the CLI does this by default).

        *project_id*, *task_id*, *dossier_path* propagate the terok
        orchestrator's identity into the per-container supervisor sidecar.
        Defaults preserve the standalone-executor case (no terok above).
        """
        return self._run(
            agent=agent,
            repo=repo,
            prompt=prompt,
            branch=branch,
            model=model,
            max_turns=max_turns,
            timeout=timeout,
            gate=gate,
            name=name,
            follow=follow,
            mode="headless",
            unrestricted=unrestricted,
            gpu=gpu,
            memory=memory,
            cpus=cpus,
            hooks=hooks,
            human_name=human_name,
            human_email=human_email,
            authorship=authorship,
            shared_dir=shared_dir,
            shared_mount=shared_mount,
            timezone=timezone,
            project_id=project_id,
            supervisor_task_id=task_id,
            dossier_path=dossier_path,
        )

    def run_interactive(
        self,
        agent: str,
        repo: str,
        *,
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
        unrestricted: bool = True,
        gpu: bool = False,
        memory: str | None = None,
        cpus: str | None = None,
        hooks: LifecycleHooks | None = None,
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
        timezone: str | None = None,
        project_id: str = "",
        task_id: str = "",
        dossier_path: Path | str | None = None,
    ) -> str:
        """Launch an interactive container. Returns container name.

        The container stays up after init; user logs in via ``podman exec``.

        See [`run_headless`][terok_executor.container.runner.AgentRunner.run_headless]
        for the *project_id* / *task_id* / *dossier_path* semantics.
        """
        return self._run(
            agent=agent,
            repo=repo,
            branch=branch,
            gate=gate,
            name=name,
            mode="interactive",
            unrestricted=unrestricted,
            gpu=gpu,
            memory=memory,
            cpus=cpus,
            hooks=hooks,
            human_name=human_name,
            human_email=human_email,
            authorship=authorship,
            shared_dir=shared_dir,
            shared_mount=shared_mount,
            timezone=timezone,
            project_id=project_id,
            supervisor_task_id=task_id,
            dossier_path=dossier_path,
        )

    def run_web(
        self,
        repo: str,
        *,
        port: int | None = None,
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
        public_url: str | None = None,
        unrestricted: bool = True,
        gpu: bool = False,
        memory: str | None = None,
        cpus: str | None = None,
        hooks: LifecycleHooks | None = None,
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
        timezone: str | None = None,
        project_id: str = "",
        task_id: str = "",
        dossier_path: Path | str | None = None,
    ) -> str:
        """Launch a toad web container. Returns container name.

        If *port* is None, an available port is auto-allocated.

        See [`run_headless`][terok_executor.container.runner.AgentRunner.run_headless]
        for the *project_id* / *task_id* / *dossier_path* semantics.
        """
        if port is None:
            with self.runtime.reserve_port() as reservation:
                port = reservation.port
        return self._run(
            agent="claude",  # toad uses claude as default
            repo=repo,
            branch=branch,
            gate=gate,
            name=name,
            mode="web",
            port=port,
            public_url=public_url,
            unrestricted=unrestricted,
            gpu=gpu,
            memory=memory,
            cpus=cpus,
            hooks=hooks,
            human_name=human_name,
            human_email=human_email,
            authorship=authorship,
            shared_dir=shared_dir,
            shared_mount=shared_mount,
            timezone=timezone,
            project_id=project_id,
            supervisor_task_id=task_id,
            dossier_path=dossier_path,
        )

    def run_tool(
        self,
        tool: str,
        repo: str,
        *,
        tool_args: tuple[str, ...] = (),
        branch: str | None = None,
        gate: bool = True,
        name: str | None = None,
        follow: bool = True,
        timeout: int = 600,
        timezone: str | None = None,
        project_id: str = "",
        task_id: str = "",
        dossier_path: Path | str | None = None,
    ) -> str:
        """Launch a sidecar tool container. Returns container name.

        Runs the named tool in a lightweight sidecar L1 image (no agent
        CLIs).  The tool receives the real API key from the credential
        store — not a phantom token.

        See [`run_headless`][terok_executor.container.runner.AgentRunner.run_headless]
        for the *project_id* / *task_id* / *dossier_path* semantics.
        """
        return self._run(
            agent=tool,
            repo=repo,
            mode="tool",
            gate=gate,
            name=name,
            follow=follow,
            timeout=timeout,
            tool_args=tool_args,
            branch=branch,
            timezone=timezone,
            project_id=project_id,
            supervisor_task_id=task_id,
            dossier_path=dossier_path,
        )

    def launch_prepared(
        self,
        *,
        env: dict[str, str],
        volumes: list[VolumeSpec],
        image: str,
        command: list[str],
        name: str,
        task_dir: Path,
        gpu: bool = False,
        memory: str | None = None,
        cpus: str | None = None,
        unrestricted: bool = True,
        sealed: bool = False,
        hooks: LifecycleHooks | None = None,
        extra_args: list[str] | None = None,
        hostname: str | None = None,
        annotations: Mapping[str, str] | None = None,
        runtime: str | None = None,
        project_id: str = "",
        task_id: str = "",
        dossier_path: Path | str | None = None,
        per_container: PerContainerResources | None = None,
    ) -> str:
        """Launch a container from a caller-prepared env, volumes, image, and command.

        Use this when the caller has already assembled the environment and
        volume specs — e.g. the terok orchestrator, which computes
        project-specific env via ``build_task_env_and_volumes`` and owns
        the container naming policy.  For end-to-end runs from a repo and
        prompt (CLI-style), use [`run_headless`][terok_executor.container.runner.AgentRunner.run_headless], [`run_interactive`][terok_executor.container.runner.AgentRunner.run_interactive],
        or [`run_web`][terok_executor.container.runner.AgentRunner.run_web] instead.

        In sealed isolation mode (*sealed=True*), the sandbox splits the
        launch into ``create`` → ``copy_to`` → ``start`` instead of a
        single ``run`` — no host↔container bind mounts remain after startup.

        Args:
            env: Environment variables injected into the container.
            volumes: Host↔container directory specs (sandbox decides mount vs inject).
            image: Image tag to run.
            command: Command + args to execute as PID 1.
            name: Container name (must be unique on the host).
            task_dir: Per-task directory used for per-container shield state.
            gpu: Pass GPU device args when True.
            memory: Podman ``--memory`` value (``"4g"`` etc.); ``None`` = unlimited.
            cpus: Podman ``--cpus`` value (``"2.0"`` etc.); ``None`` = unlimited.
            unrestricted: When False, adds ``--security-opt no-new-privileges``.
            sealed: Enable sealed isolation (no bind mounts).
            hooks: Optional lifecycle callbacks fired around the launch.
            extra_args: Additional raw ``podman run`` flags (e.g. port publishing).
            hostname: Override the in-container hostname (podman ``--hostname``).
                When ``None`` (default), podman assigns the short container ID.
            annotations: OCI annotations forwarded as ``--annotation k=v``;
                validated against
                [`SAFE_ANNOTATION_KEYS`][terok_sandbox.sandbox.SAFE_ANNOTATION_KEYS].
                Typed channel for orchestrator metadata the shield reads,
                distinct from the freeform *extra_args*.
            runtime: OCI runtime selector forwarded to
                [`RunSpec.runtime`][terok_sandbox.sandbox.RunSpec.runtime].
                ``None`` (default) leaves the choice to podman; ``"krun"``
                selects the libkrun microVM backend and also drives
                shield's dnsmasq bind selection.  Prefer this over
                passing ``--runtime`` via *extra_args* — sandbox emits
                the flag itself and shield reads the value to pick the
                right firewall topology.
            project_id: Identity written into the per-container
                supervisor sidecar so the supervisor can scope its
                state to the calling terok project.  Default ``""``
                preserves the standalone-executor case where no terok
                orchestrator sits above the runner.
            task_id: Per-task identity written into the supervisor
                sidecar alongside *project_id*.  Default ``""`` for
                the standalone case.
            dossier_path: Path to the per-task dossier file the
                shield reads at container start.  Default ``None``
                omits the field from the sidecar — only orchestrated
                runs carry a dossier.
            per_container: Pre-allocated per-container socket dir / TCP
                ports.  When provided, the launch uses these instead of
                allocating its own — so a caller that already threaded
                the same instance through env assembly
                ([`assemble_container_env`][terok_executor.container.env.assemble_container_env])
                keeps the vault-routing env vars and the supervisor
                binding on identical ports.  Default ``None`` allocates
                internally (the standalone path, and external callers
                that assemble env without per-container routing).

        Returns:
            The container name (same as *name*).

        Raises:
            BuildError: When GPU was requested but the host has no functioning
                NVIDIA CDI.
        """
        from terok_executor.integrations.sandbox import (
            GpuConfigError,
            RunSpec,
            Sharing,
            VolumeSpec,
            allocate_per_container_resources,
            write_sidecar,
        )

        cfg = self.sandbox.config

        # Per-container socket dir / TCP ports.  Allocated here so the
        # mount, the env vars the in-container bridge reads, and the
        # sidecar JSON the supervisor reads all see the same values —
        # the only path that keeps concurrent containers from colliding
        # on the singletons baked into ``cfg``.  When the caller already
        # allocated one (``_run`` threads it through env assembly too),
        # reuse that instance so the vault-routing env vars and this
        # binding land on identical ports — a second allocation here
        # would hand back different ports and re-introduce the TCP-mode
        # cross-container collision.
        if per_container is None:
            per_container = allocate_per_container_resources(cfg, name)

        # Bind-mount the per-container socket dir at /run/terok/.  The
        # supervisor's later-bound vault.sock + ssh-agent.sock surface
        # inside the container via this single mount (instead of two
        # singleton file-mounts that two containers would collide on).
        env = dict(env)
        volumes = list(volumes)
        volumes.append(
            VolumeSpec(
                per_container.container_runtime_dir,
                "/run/terok",
                sharing=Sharing.SHARED,
                live=True,
            )
        )
        # TCP-mode env vars carry the per-container port, not the
        # host-singleton ``cfg.token_broker_port`` — the launch flow
        # routes through the per-container ports only.
        if cfg.services_mode == "tcp":
            if per_container.token_broker_port is not None:
                env["TEROK_TOKEN_BROKER_PORT"] = str(per_container.token_broker_port)
            if per_container.ssh_signer_port is not None:
                env["TEROK_SSH_SIGNER_PORT"] = str(per_container.ssh_signer_port)
            if per_container.gate_port is not None:
                env["TEROK_GATE_PORT"] = str(per_container.gate_port)

        # The gate is wired when the prepared env carries a gate token
        # (set by ``_setup_gate`` / the orchestrator).  When active, the
        # supervisor needs the mirror base path, the token, and — in TCP
        # mode — the port to serve the gate in-process.
        gate_active = "TEROK_GATE_TOKEN" in env

        # Write the per-container supervisor sidecar before podman run.
        # The terok-sandbox OCI hook installed by ``terok-sandbox setup``
        # reads this file on container start and spawns one supervisor
        # per container; without it the supervisor refuses to start.
        # ``write_sidecar`` is sandbox's canonical writer — schema,
        # reader, and teardown live in one package, so nothing here can
        # drift from what the supervisor parses.
        sidecar_path = write_sidecar(
            name,
            cfg=cfg,
            per_container=per_container,
            project_id=project_id,
            task_id=task_id,
            dossier_path=dossier_path,
            gate_base_path=str(cfg.gate_base_path) if gate_active else None,
            gate_token=env["TEROK_GATE_TOKEN"] if gate_active else None,
            gate_port=per_container.gate_port if gate_active else None,
        )
        # Fail closed: a missing sidecar means the supervisor OCI hook
        # never fires, so the container would launch with no vault,
        # clearance, or signer behind it. Refuse the launch rather than
        # run unsupervised.
        if sidecar_path is None:
            raise BuildError(
                f"supervisor sidecar write failed for {name}; refusing to launch "
                "an unsupervised container (no vault/clearance/signer)"
            )
        # The supervisor OCI hook fires only when this annotation is
        # present (matched by ``when.annotations`` in the hook
        # descriptor) and reads its value as the sidecar location —
        # no XDG guessing, one anchor.
        spec_annotations = dict(annotations or {})
        spec_annotations["terok.sandbox.sidecar"] = str(sidecar_path)

        loopback_ports = tuple(
            p
            for p in (
                per_container.gate_port,
                per_container.token_broker_port,
                per_container.ssh_signer_port,
            )
            if p is not None
        )

        spec = RunSpec(
            container_name=name,
            image=image,
            env=env,
            volumes=tuple(volumes),
            command=tuple(command),
            task_dir=task_dir,
            gpu_enabled=gpu,
            memory=memory,
            cpus=cpus,
            extra_args=tuple(extra_args or ()),
            unrestricted=unrestricted,
            sealed=sealed,
            hostname=hostname,
            annotations=spec_annotations,
            runtime=runtime,
            loopback_ports=loopback_ports,
        )

        try:
            self.sandbox.run(spec, hooks=hooks)
        except GpuConfigError as exc:
            raise BuildError(str(exc)) from exc

        return name

    def wait_for_exit(
        self,
        container_name: str,
        timeout: float | None = None,
    ) -> int:
        """Block until *container_name* exits; return its exit code.

        Raises [`TimeoutError`][TimeoutError] when *timeout* elapses before the
        container exits — signalled out of band so a container that
        legitimately exits with code 124 (the ``timeout(1)`` convention)
        is returned unambiguously as its real exit code, not conflated
        with the wait timing out.

        Raises [`RuntimeError`][RuntimeError] when ``podman wait`` itself fails
        (non-zero returncode, e.g. unknown container) or returns output
        that is not a container exit code — the podman error is never
        impersonated as the container's exit code, which would let a
        "no such container" diagnostic leak out as exit code 125.

        Raises [`FileNotFoundError`][FileNotFoundError] when ``podman`` is not on PATH.
        Intentionally re-implements the wait loop instead of delegating
        to `Sandbox.wait_for_exit`, which swallows
        [`subprocess.TimeoutExpired`][subprocess.TimeoutExpired] and returns the 124 sentinel
        — fine for fire-and-forget generic waits, lossy for task-level
        callers that need to record the real exit code.
        """
        import subprocess

        try:
            proc = subprocess.run(
                ["podman", "wait", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"container {container_name!r} did not exit within {timeout}s"
            ) from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip() or "<no output>"
            raise RuntimeError(
                f"podman wait {container_name!r} failed (returncode={proc.returncode}): {detail}"
            )

        stdout = (proc.stdout or "").strip()
        try:
            return int(stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"podman wait {container_name!r} returned unexpected output: "
                f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
            ) from exc

    def logs(
        self,
        container_name: str,
        *,
        tail: int | None = None,
        timestamps: bool = False,
        since: str | None = None,
    ) -> str:
        """Return the container's logged output as a single string.

        One-shot retrieval for the "just show me what ran" case.  For live
        streaming (human watching), use [`stream_logs_process`][terok_executor.container.runner.AgentRunner.stream_logs_process]; for
        archival, use [`capture_logs`][terok_executor.container.runner.AgentRunner.capture_logs].

        Raises [`RuntimeError`][RuntimeError] when ``podman logs`` returns a non-zero
        status (e.g. unknown container) — the diagnostic is surfaced rather
        than impersonated as empty output.  [`FileNotFoundError`][FileNotFoundError]
        propagates when ``podman`` is not on PATH.
        """
        import subprocess

        cmd = _build_logs_cmd(container_name, tail=tail, timestamps=timestamps, since=since)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip() or "<no output>"
            raise RuntimeError(
                f"podman logs {container_name!r} failed (returncode={proc.returncode}): {detail}"
            )
        return (proc.stdout or "") + (proc.stderr or "")

    def capture_logs(
        self,
        container_name: str,
        dest: Path,
        *,
        timestamps: bool = True,
        timeout: float = 60.0,
    ) -> bool:
        """Capture a container's logs to *dest*; return ``True`` on success.

        Streams stdout directly to *dest* (bytes) so large logs do not need
        to fit in memory.  Used at task-archive time to freeze the
        container's output onto the host filesystem before removal.

        On any failure — missing podman, podman error, timeout — *dest* is
        removed and ``False`` is returned so the caller sees one signal,
        not a partially-written file.
        """
        import subprocess

        cmd = _build_logs_cmd(container_name, timestamps=timestamps)
        try:
            with dest.open("wb") as f:
                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            dest.unlink(missing_ok=True)
            return False

        if proc.returncode != 0:
            dest.unlink(missing_ok=True)
            return False
        return True

    def stream_logs_process(
        self,
        container_name: str,
        *,
        follow: bool = False,
        tail: int | None = None,
        timestamps: bool = False,
        merge_stderr: bool = False,
    ) -> subprocess.Popen[bytes]:
        """Spawn a long-running ``podman logs`` process; return the ``Popen``.

        The raw subprocess handle is exposed deliberately: live-log
        consumers (TUI log viewer, interactive ``task logs -f``) need
        fd-level control — ``select()`` between reads, SIGINT handling,
        stop-event polling — that a higher-level iterator abstraction
        would hide badly.  Every current caller's event loop already looks
        like ``select([proc.stdout], …) → read1()`` so returning the
        ``Popen`` matches existing patterns instead of fighting them.

        Caller owns the subprocess.  Typical pattern::

            proc = runner.stream_logs_process(cname, follow=True)
            try:
                for chunk in iter(proc.stdout.read1, b""):
                    ...
            finally:
                proc.terminate()
                proc.wait()

        When *merge_stderr* is True, stderr is folded into stdout
        (matches ``subprocess.STDOUT``); otherwise stderr is a separate
        pipe the caller can drain.

        [`FileNotFoundError`][FileNotFoundError] propagates when ``podman`` is not on
        PATH — callers handle it (usually as a user-facing "podman not
        installed" error).
        """
        import subprocess

        cmd = _build_logs_cmd(container_name, follow=follow, tail=tail, timestamps=timestamps)
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        )

    # ------------------------------------------------------------------
    # Internal orchestrator (all public entry points delegate here)
    # ------------------------------------------------------------------

    def _run(
        self,
        *,
        agent: str,
        repo: str,
        mode: str,
        prompt: str | None = None,
        branch: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        timeout: int = 1800,
        gate: bool = True,
        name: str | None = None,
        follow: bool = False,
        port: int | None = None,
        public_url: str | None = None,
        unrestricted: bool = True,
        gpu: bool = False,
        memory: str | None = None,
        cpus: str | None = None,
        hooks: LifecycleHooks | None = None,
        tool_args: tuple[str, ...] = (),
        human_name: str | None = None,
        human_email: str | None = None,
        authorship: str | None = None,
        shared_dir: Path | None = None,
        shared_mount: str = "/shared",
        timezone: str | None = None,
        project_id: str = "",
        supervisor_task_id: str = "",
        dossier_path: Path | str | None = None,
    ) -> str:
        """Unified launch flow for all modes (headless, interactive, web, tool)."""
        from terok_executor.integrations.sandbox import allocate_per_container_resources
        from terok_executor.paths import mounts_dir

        from .env import ContainerEnvSpec, assemble_container_env

        is_tool = mode == "tool"
        task_id = _generate_task_id()
        code_repo, local_path = _resolve_repo(repo)

        # Resolve the container name and allocate the per-container socket
        # dir / TCP ports ONCE, up front.  The same instance is threaded
        # into ``assemble_container_env`` (so vault routing — glab
        # ``GITLAB_API_HOST``, base-URL env, broker port — uses the
        # per-container port) and into ``launch_prepared`` (so the
        # supervisor binds those same ports).  Allocating twice would put
        # the routing env on the host-singleton ``cfg`` port while the
        # supervisor bound a different one, colliding across concurrent
        # containers in TCP mode (CWE-284).
        cname = name or f"terok-executor-{task_id}"
        per_container = allocate_per_container_resources(self.sandbox.config, cname)

        # Ensure images — sidecar L1 for tools, agent L1 for everything else
        if is_tool:
            sidecar_spec = self.roster.get_sidecar_spec(agent)
            image_tag = self._ensure_sidecar_image(sidecar_spec.tool_name)
        else:
            agent_spec = self.roster.get_agent(agent)
            image_tag = self._ensure_images()

        # Task directory (ephemeral for standalone runs)
        task_dir = Path(tempfile.mkdtemp(prefix=f"terok-executor-{task_id}-"))

        mounts_base = mounts_dir()

        if is_tool:
            # Sidecar tools: minimal env, no shared mounts, real API key
            env: dict[str, str] = {
                "TASK_ID": task_id,
                "REPO_ROOT": "/workspace",
                "GIT_RESET_MODE": "none",
            }
            if tz := timezone or detect_host_timezone():
                env["TZ"] = tz
            if branch:
                env["GIT_BRANCH"] = branch
            env.update(self._direct_credential_env(agent))

            volumes: list[VolumeSpec] = []
            if local_path:
                volumes.append(VolumeSpec(local_path, "/workspace", sharing=Sharing.PRIVATE))
            elif code_repo:
                if gate:
                    effective_repo, gate_token = self._setup_gate(code_repo)
                    # The launch path wires this into the supervisor sidecar.
                    env["TEROK_GATE_TOKEN"] = gate_token
                else:
                    effective_repo = code_repo
                env["CODE_REPO"] = effective_repo
                workspace = task_dir / "workspace"
                workspace.mkdir(parents=True, exist_ok=True)
                _seed_from_cache(
                    workspace, code_repo, self.sandbox.config, origin_url=effective_repo
                )
                volumes.append(VolumeSpec(workspace, "/workspace", sharing=Sharing.PRIVATE))
        else:
            # Agent modes: full env assembly via canonical builder
            agent_config_dir = self._prepare_agent_config(
                task_dir,
                task_id,
                agent,
                prompt=prompt,
                mounts_base=mounts_base,
                project_root=local_path,
            )

            # Resolve workspace and gate URL
            gate_token = None
            if local_path:
                ws_host = local_path
                resolved_code_repo = None
            elif code_repo:
                if gate:
                    effective_repo, gate_token = self._setup_gate(code_repo)
                else:
                    effective_repo = code_repo
                ws_host = task_dir / "workspace"
                ws_host.mkdir(parents=True, exist_ok=True)
                _seed_from_cache(ws_host, code_repo, self.sandbox.config, origin_url=effective_repo)
                resolved_code_repo = effective_repo
            else:
                ws_host = task_dir / "workspace"
                ws_host.mkdir(parents=True, exist_ok=True)
                resolved_code_repo = None

            spec_kwargs: dict = {
                "task_id": task_id,
                "agent_name": agent,
                "workspace_host_path": ws_host,
                "code_repo": resolved_code_repo,
                "branch": branch,
                "unrestricted": unrestricted,
                "agent_config_dir": agent_config_dir,
                "task_dir": task_dir,
                "envs_dir": mounts_base,
                "timezone": timezone,
            }
            if human_name:
                spec_kwargs["human_name"] = human_name
            if human_email:
                spec_kwargs["human_email"] = human_email
            if authorship:
                spec_kwargs["authorship"] = authorship
            if shared_dir:
                if not shared_mount.startswith("/") or ":" in shared_mount:
                    raise SystemExit(
                        f"--shared-mount must be an absolute path without ':', got: {shared_mount!r}"
                    )
                if shared_dir.is_file():
                    raise SystemExit(f"--shared-dir exists as a file: {shared_dir}")
                spec_kwargs["shared_dir"] = shared_dir
                spec_kwargs["shared_mount"] = shared_mount

            result = assemble_container_env(
                ContainerEnvSpec(**spec_kwargs),
                self.roster,
                per_container=per_container,
            )
            env = dict(result.env)
            if gate_token is not None:
                # The launch path wires this into the supervisor sidecar.
                env["TEROK_GATE_TOKEN"] = gate_token
            volumes = list(result.volumes)

        # Build command based on mode
        extra_args: list[str] = []
        if mode == "tool":
            tool_cmd = f"init-ssh-and-repo.sh && {shlex.quote(agent)}"
            if tool_args:
                tool_cmd += " " + " ".join(shlex.quote(a) for a in tool_args)
            command = ["bash", "-lc", tool_cmd]
        elif mode == "headless":
            cmd_str = agent_spec.build_headless_command(
                timeout=timeout, model=model, max_turns=max_turns
            )
            command = ["bash", "-lc", cmd_str]
        elif mode == "interactive":
            command = [
                "bash",
                "-lc",
                "init-ssh-and-repo.sh && echo __CLI_READY__; tail -f /dev/null",
            ]
        elif mode == "web":
            toad_cmd = "init-ssh-and-repo.sh && toad --serve -H 0.0.0.0 -p 8080"
            if public_url:
                toad_cmd += f" --public-url {shlex.quote(public_url)}"
            toad_cmd += " /workspace"
            command = ["bash", "-lc", toad_cmd]
            extra_args += ["-p", f"127.0.0.1:{port}:8080"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Launch — pass the SAME per_container allocated above so the
        # supervisor binds the exact ports the vault-routing env vars
        # point at (no second allocation).
        cname = self.launch_prepared(
            env=env,
            volumes=volumes,
            image=image_tag,
            command=command,
            name=cname,
            task_dir=task_dir,
            gpu=gpu,
            memory=memory,
            cpus=cpus,
            unrestricted=unrestricted,
            extra_args=extra_args or None,
            hooks=hooks,
            project_id=project_id,
            task_id=supervisor_task_id,
            dossier_path=dossier_path,
            per_container=per_container,
        )

        # Follow output if requested
        if follow and mode in ("headless", "tool"):
            self._stream_headless(cname, timeout=float(timeout + 60))
        elif mode == "interactive":
            from terok_executor.integrations.sandbox import READY_MARKER

            ready = self.sandbox.stream_logs(
                cname,
                timeout=120.0,
                ready_check=lambda line: "__CLI_READY__" in line or READY_MARKER in line,
            )
            if ready:
                # `login_command` is runtime-aware: PodmanContainer emits
                # `podman exec -it …`, KrunContainer emits `ssh -tt -i …
                # ProxyCommand=…`.  One code path serves both backends.
                import shlex as _shlex

                argv = self.runtime.container(cname).login_command(command=("bash", "-l"))
                print(f"\nContainer ready. Login with:\n  {_shlex.join(argv)}")
        elif mode == "web" and port:
            from terok_executor.integrations.sandbox import READY_MARKER

            self.sandbox.stream_logs(cname, timeout=120.0)
            url = public_url or f"http://127.0.0.1:{port}"
            print(f"\nToad available at: {url}")

        return cname

    # ------------------------------------------------------------------
    # Private helpers (in call order from _run)
    # ------------------------------------------------------------------

    def _ensure_images(self) -> str:
        """Ensure L0+L1 images exist, return L1 tag."""
        images = ImageBuilder(self._base_image, self._family).build_base()
        return images.l1

    def _ensure_sidecar_image(self, tool_name: str) -> str:
        """Ensure sidecar L1 exists for *tool_name*, return its tag."""

        return ImageBuilder(self._base_image, self._family).build_sidecar(tool_name=tool_name)

    def _setup_gate(self, repo_url: str) -> tuple[str, str]:
        """Mirror a repo via the gate and return ``(gate_url, gate_token)``.

        Steps:
        1. Create a bare git mirror under the gate base path
        2. Mint a per-container gate access token
        3. Construct the HTTP gate URL embedding that token

        The gate runs inside the per-container supervisor; the token
        travels to the container via the sidecar (the caller surfaces it as
        ``TEROK_GATE_TOKEN`` so the launch path wires it into the sidecar).
        The container clones from the returned URL; shield blocks all other
        egress.
        """
        from terok_executor.integrations.sandbox import GitGate

        cfg = self.sandbox.config
        gate_base = cfg.gate_base_path
        gate_base.mkdir(parents=True, exist_ok=True)

        # Derive a collision-free repo key from the full URL
        import hashlib

        url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
        basename = repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        repo_key = f"{basename}-{url_hash}"
        gate_path = gate_base / repo_key

        # Sync (creates bare mirror if missing, fetches if exists)
        gate = GitGate(
            scope=repo_key,
            gate_path=gate_path,
            upstream_url=repo_url,
            clone_cache_base=cfg.clone_cache_base_path,
        )
        gate.sync()

        # Mint the per-container token and build the URL that embeds it.
        token = self.sandbox.mint_gate_token()
        return self.sandbox.gate_url(gate_path, token), token

    def _direct_credential_env(self, tool_name: str) -> dict[str, str]:
        """Load the real API key for a sidecar tool and return as env dict.

        Unlike vault phantom-token injection (which creates phantom tokens),
        this injects the actual credential.  Safe because sidecar containers
        have no agent code that could leak it.
        """
        spec = self.roster.get_sidecar_spec(tool_name)
        # A tool's credential is stored under its default provider when it has
        # one (``gh`` → ``github``), mirroring how an agent keys to its provider;
        # a tool without a provider binding keys under its own name.
        auth_info = self.roster.auth_providers.get(tool_name)
        provider_key = (auth_info.credential_provider if auth_info else "") or tool_name
        cfg = self.sandbox.config
        try:
            db = cfg.open_credential_db()
        except Exception as exc:
            print(
                f"Warning [runner]: credential DB unavailable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return {}
        try:
            cred = db.load_credential("default", provider_key)
        finally:
            db.close()

        if cred is None:
            print(
                f"Warning [runner]: no credentials stored for {tool_name!r}. "
                f"Run: terok-executor auth {tool_name} --api-key <key>",
                file=sys.stderr,
            )
            return {}

        return {
            env_var: str(cred.get(cred_key, ""))
            for env_var, cred_key in spec.env_map.items()
            if cred.get(cred_key)
        }

    def _prepare_agent_config(
        self,
        task_dir: Path,
        task_id: str,
        agent: str,
        *,
        prompt: str | None = None,
        instructions: str | None = None,
        mounts_base: Path,
        project_root: Path | None = None,
    ) -> Path:
        """Prepare the agent-config directory for a task.

        *project_root* is passed to [`resolve_instructions`][terok_executor.resolve_instructions] so that
        ``<repo>/instructions.md`` is appended when present.
        """
        from terok_executor.provider.agents import AgentConfigSpec, prepare_agent_config_dir
        from terok_executor.provider.instructions import resolve_instructions

        resolved_instructions = instructions or resolve_instructions(
            {}, agent, project_root=project_root
        )

        spec = AgentConfigSpec(
            tasks_root=task_dir.parent,
            task_id=task_id,
            prompt=prompt,
            agent=agent,
            instructions=resolved_instructions,
            mounts_base=mounts_base,
        )
        return prepare_agent_config_dir(spec)

    @staticmethod
    def _stream_headless(cname: str, timeout: float) -> None:
        """Stream container logs to stdout and print exit code when done."""
        import subprocess
        import sys

        try:
            proc = subprocess.Popen(
                ["podman", "logs", "-f", cname],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            print("Agent timed out", file=sys.stderr)
        except (FileNotFoundError, OSError) as exc:
            print(
                f"Warning [runner]: failed to stream logs for {cname}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        # Retrieve exit code from the container itself
        try:
            result = subprocess.run(["podman", "wait", cname], capture_output=True, timeout=10)
            exit_code = int(result.stdout.decode().strip()) if result.stdout else 1
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as exc:
            print(
                f"Warning [runner]: failed to retrieve exit code for {cname}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            exit_code = 1

        if exit_code != 0:
            print(f"Agent exited with code {exit_code}")


# ── Module-level helpers ────────────────────────────────────────────────


def _build_logs_cmd(
    container_name: str,
    *,
    follow: bool = False,
    tail: int | None = None,
    timestamps: bool = False,
    since: str | None = None,
) -> list[str]:
    """Assemble a ``podman logs`` command with the given flags.

    Shared builder so the three log entry points ([`AgentRunner.logs`][terok_executor.container.runner.AgentRunner.logs],
    [`AgentRunner.capture_logs`][terok_executor.container.runner.AgentRunner.capture_logs], [`AgentRunner.stream_logs_process`][terok_executor.container.runner.AgentRunner.stream_logs_process])
    agree on flag order and naming.
    """
    cmd = ["podman", "logs"]
    if follow:
        cmd.append("-f")
    if timestamps:
        cmd.append("--timestamps")
    if tail is not None:
        cmd.extend(["--tail", str(tail)])
    if since:
        cmd.extend(["--since", since])
    cmd.append(container_name)
    return cmd


def _generate_task_id() -> str:
    """Generate a short unique task identifier."""
    return uuid.uuid4().hex[:12]


def _seed_from_cache(
    workspace: Path,
    repo_url: str,
    cfg: SandboxConfig | None,
    *,
    origin_url: str | None = None,
) -> None:
    """Seed *workspace* from the clone cache for *repo_url* (best-effort).

    *repo_url* is always the upstream URL (used to derive the cache scope).
    *origin_url* is written into the seeded ``.git`` as origin — typically
    the gate HTTP URL when gated, or *repo_url* itself otherwise.
    """
    import hashlib

    from .cache import seed_workspace_from_clone_cache

    url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    basename = repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    scope = f"{basename}-{url_hash}"
    seed_workspace_from_clone_cache(workspace, scope, origin_url=origin_url or repo_url, cfg=cfg)


def _resolve_repo(repo: str) -> tuple[str | None, Path | None]:
    """Classify *repo* as a git URL or local path.

    Returns ``(code_repo, local_path)`` — exactly one is non-None.
    Raises ``SystemExit`` for ambiguous local paths (look like paths but
    don't exist).
    """
    # Heuristic: if it looks like a local path (starts with /, ./, ~, or has
    # no : before /), check existence
    p = Path(repo).expanduser()
    if p.is_dir():
        return None, p.resolve()
    # If it looks like a local path but doesn't exist, fail early
    if repo.startswith(("/", "./", "../", "~")) or (
        not repo.startswith("git@") and "://" not in repo and ":" not in repo
    ):
        raise SystemExit(f"Local path not found: {repo}")
    # Treat as git URL (SSH, HTTPS, or file://)
    return repo, None
