# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""v1: run the *real* Codex CLI with a phantom token and trap its first authed request.

The whole chain, "as if a user ran it": real `codex` in a container, given only a
phantom API key (api_key mode) + the vault base URL via `config.toml` exactly as
terok provisions it, reaches the host-side vault, which swaps the phantom for the real
credential and forwards to a **request trap**.  The trap records what it saw and
answers with nothing useful — codex then errors, which is fine; we only need the first
authenticated request(s).

What this asserts (at the trap):
- **the phantom swapped to the real key** — proves phantom-health + the auth *mechanism*
  (codex sends `Authorization: Bearer <token>` in api_key mode) + end-to-end connectivity;
- **codex reached the vault at all** — a sudden client-side refusal of the phantom, or a
  base-url/config regression, shows up as *no capture*;
- **the WebSocket transport** — codex v0.146 on the default `openai` provider is
  WS-first (`wss://{base}/responses`, background prewarm at session spawn), so a WS
  upgrade should reach the trap; this is the leg terok's vault WS fix targets.

Scope (v1, per design): **api_key mode only** — that's what a current-shape phantom
(`terok-p-<hex>`) supports (oauth mode JWT-decodes the token and hits a hardcoded
auth.openai.com refresh). v2 will capture a WS *frame* to assert the sentinel prompt
survives the hop; the oauth/zstd path and the full `terok task` driver are later.

────────────────────────────────────────────────────────────────────────────────
MATRIX VALIDATION NOTES — this has only been collection+lint-checked off the test
machine.  Run it under the matrix and expect to tune:
  * Image: fedora + `npm i -g @openai/codex` + socat/curl/git + a uid-1000 `dev` user.
    Codex ships a glibc binary via npm → fedora base (not the alpine PODMAN_BASE_IMAGE).
    Verify the dev user / HOME=/home/dev / WORKDIR match what Sandbox.run expects.
  * Codex provisioning: `~/.codex/auth.json = {"OPENAI_API_KEY": <phantom>}` +
    `~/.codex/config.toml` with `openai_base_url`.  Confirm the exact auth.json shape
    codex v0.146 expects for api_key mode (spec says that field name; verify nesting).
  * Reachability: codex reads `openai_base_url=http://localhost:9419/v1`; the socat
    loopback bridge (same command ensure-bridges.sh uses) fronts the bind-mounted
    vault socket.  WS + HTTP both traverse socat.
  * Timing: the WS prewarm is a background task racing the /models GET — poll the trap
    for a few seconds.  If no WS upgrade appears, the swap assertion still holds.
  * OTEL: disabled via `-c otel.metrics_exporter=none` so the shield stays quiet
    (default-on egress to a hardcoded ab.chatgpt.com otherwise — see follow-up).
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

import pytest
from aiohttp import web

from terok_executor.integrations.sandbox import RunSpec, VolumeSpec
from terok_executor.vault_addr import CONTAINER_VAULT_SOCKET, LOOPBACK_VAULT_PORT
from tests.constants import CONTAINER_WORKSPACE_DIR, INTEGRATION_VAULT_PASSPHRASE

from .conftest import ExecutorEnv, hooks_missing, podman_missing
from .helpers import exec_in, podman, podman_rm, unique_container_name

pytestmark = [pytest.mark.needs_podman, podman_missing, hooks_missing]

PROVIDER = "openai"  # codex's vault route key (codex.yaml provider.default: openai)
REAL_SECRET = "sk-integration-codex-not-for-the-container"  # nosec B105 — fixture value
CREDENTIAL_SET = "default"
CREDENTIAL_SCOPE = "integration-scope"
_CODEX_IMAGE = "terok-executor-itest-codex:latest"
_FEDORA_BASE = "registry.fedoraproject.org/fedora:44"  # glibc base for codex's npm binary
_CAPTURE_TIMEOUT_S = 30


class _TrappingVault:
    """Vault broker + a request trap on a background event loop.

    The trap accepts anything: WS upgrades are prepared+recorded+closed; plain HTTP is
    recorded and answered with a minimal ``{"data": []}`` (enough for codex's cold
    ``GET /models``).  Every capture records the ``Authorization`` header the *vault
    forwarded* (i.e. the real key, post-swap) plus the path and transport kind, so a
    synchronous test can assert on them while codex runs.
    """

    def __init__(self, db_path: Path, routes_path: Path, socket_path: Path) -> None:
        self._db_path = db_path
        self._routes_path = routes_path
        self._socket_path = socket_path
        self.captures: list[dict[str, object]] = []
        self._loop = None
        self._thread: threading.Thread | None = None
        self._runners: list[web.AppRunner] = []

    def __enter__(self) -> _TrappingVault:
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self._thread.start()
        if not ready.wait(15):
            raise RuntimeError("vault/trap did not come up on the background loop")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(15)

    def _run(self, ready: threading.Event) -> None:
        import asyncio

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start())
        ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._cleanup())
            self._loop.close()

    async def _start(self) -> None:
        from terok_sandbox.vault.daemon.token_broker import _build_app

        async def _trap(request: web.Request) -> web.StreamResponse:
            rec: dict[str, object] = {
                "path": request.path,
                "auth": request.headers.get("Authorization", ""),
                "originator": request.headers.get("originator", ""),
                "user_agent": request.headers.get("User-Agent", ""),
                "version": request.headers.get("version", ""),
            }
            if request.headers.get("Upgrade", "").lower() == "websocket":
                rec["kind"] = "ws"
                self.captures.append(rec)
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                await ws.close()
                return ws
            rec["kind"] = "http"
            self.captures.append(rec)
            return web.json_response({"data": []})

        trap_app = web.Application()
        trap_app.router.add_route("*", "/{tail:.*}", _trap)
        trap_runner = web.AppRunner(trap_app)
        await trap_runner.setup()
        trap_site = web.TCPSite(trap_runner, host="127.0.0.1", port=0)
        await trap_site.start()
        trap_port = trap_site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
        self._routes_path.write_text(
            json.dumps(
                {
                    PROVIDER: {
                        "upstream": f"http://127.0.0.1:{trap_port}",
                        "auth_header": "Authorization",
                        "auth_prefix": "Bearer ",
                    }
                }
            )
        )

        broker_runner = web.AppRunner(_build_app(str(self._db_path), str(self._routes_path)))
        await broker_runner.setup()
        broker_site = web.UnixSite(broker_runner, path=str(self._socket_path))
        await broker_site.start()
        os.chmod(self._socket_path, 0o666)  # noqa: S103 — per-test tmp socket, container peer UID
        self._runners = [broker_runner, trap_runner]

    async def _cleanup(self) -> None:
        for runner in self._runners:
            await runner.cleanup()


@pytest.fixture(scope="session")
def codex_image() -> str:
    """Build (once) a fedora image with codex + socat/curl/git and a uid-1000 dev user."""
    import shutil
    import subprocess

    if not shutil.which("podman"):
        pytest.skip("podman not on PATH")
    if podman("image", "exists", _CODEX_IMAGE).returncode != 0:
        dockerfile = (
            f"FROM {_FEDORA_BASE}\n"
            "RUN dnf install -y --setopt=install_weak_deps=False nodejs npm socat curl git "
            "&& dnf clean all\n"
            "RUN npm install -g @openai/codex && npm cache clean --force\n"
            "RUN useradd -m -u 1000 dev && mkdir -p /workspace && chown dev:dev /workspace\n"
            "USER dev\n"
            "WORKDIR /workspace\n"
        )
        subprocess.run(
            ["podman", "build", "-t", _CODEX_IMAGE, "-f", "-", "."],
            input=dockerfile,
            check=True,
            text=True,
            timeout=600,
        )
    return _CODEX_IMAGE


def test_real_codex_phantom_swaps_and_reaches_the_vault(
    executor_env: ExecutorEnv,
    codex_image: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real codex, phantom-only, api_key mode → vault swaps to the real key at the trap."""
    from terok_sandbox.config import SandboxConfig

    # 1. Seed the real credential + mint a phantom for the openai route.
    db = executor_env.cfg.open_credential_db()
    try:
        db.store_credential(CREDENTIAL_SET, PROVIDER, {"type": "api_key", "key": REAL_SECRET})
        phantom = db.create_token(CREDENTIAL_SCOPE, "codex-handshake", CREDENTIAL_SET, PROVIDER)
    finally:
        db.close()
    assert phantom != REAL_SECRET

    # 2. Provision codex exactly as terok does for api_key mode (codex.yaml): auth.json
    #    holds the bearer, config.toml points openai_base_url at the in-container vault
    #    loopback that the socat bridge fronts.
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": phantom}))
    (codex_home / "config.toml").write_text(
        f'openai_base_url = "http://localhost:{LOOPBACK_VAULT_PORT}/v1"\n'
    )

    # 3. Broker (over executor's DB) + trap on a background loop; pin the passphrase the
    #    broker's _TokenDB resolves (executor_env patches only the adapter's SandboxConfig).
    monkeypatch.setattr(
        SandboxConfig,
        "resolve_passphrase_with_source",
        lambda _self: (INTEGRATION_VAULT_PASSPHRASE, "integration"),
    )
    host_socket = tmp_path / "vault.sock"
    name = unique_container_name("codex-handshake")

    with _TrappingVault(executor_env.cfg.db_path, tmp_path / "routes.json", host_socket) as vault:
        # 4. Launch real codex via the real Sandbox.run path, with .codex config + the
        #    vault socket bind-mounted where CONTAINER_VAULT_SOCKET expects it.
        executor_env.sandbox().run(
            RunSpec(
                container_name=name,
                image=codex_image,
                env={"TEROK_VAULT_LOOPBACK_PORT": str(LOOPBACK_VAULT_PORT)},
                volumes=(
                    VolumeSpec(executor_env.workspace_dir, CONTAINER_WORKSPACE_DIR),
                    VolumeSpec(host_socket, CONTAINER_VAULT_SOCKET),
                    VolumeSpec(codex_home, "/home/dev/.codex"),
                ),
                command=("sleep", "300"),
                task_dir=executor_env.task_dir,
            )
        )
        try:
            # 5. Start the loopback bridge (same command ensure-bridges.sh uses).
            exec_in(
                name,
                "sh",
                "-c",
                f"setsid socat TCP-LISTEN:{LOOPBACK_VAULT_PORT},bind=127.0.0.1,fork,reuseaddr "
                f"UNIX-CONNECT:{CONTAINER_VAULT_SOCKET},retry=300,interval=0.1 "
                ">/tmp/socat.log 2>&1 &",
            )
            # 6. Drive one codex turn, detached — we don't need it to succeed, only to
            #    emit its first authed request(s). OTEL off so the shield stays quiet.
            sentinel = f"TEROK-CANARY-{uuid.uuid4().hex[:8]}"
            exec_in(
                name,
                "sh",
                "-c",
                "setsid codex exec --skip-git-repo-check "
                f"-c otel.metrics_exporter=none '{sentinel}' >/tmp/codex.log 2>&1 &",
            )

            # 7. Wait for the trap to capture the first authed request(s).
            deadline = time.monotonic() + _CAPTURE_TIMEOUT_S
            while not vault.captures and time.monotonic() < deadline:
                time.sleep(0.5)
        finally:
            podman_rm(name)

    # 8. Assertions.
    assert vault.captures, (
        "codex never reached the vault — phantom refused client-side, or base-url/config "
        "regressed (see /tmp/codex.log, /tmp/socat.log in the container)"
    )
    assert any(c["auth"] == f"Bearer {REAL_SECRET}" for c in vault.captures), (
        f"phantom did not swap to the real key at the trap; captures={vault.captures}"
    )
    # codex identifies itself on every request — a cheap protocol-shape canary.
    assert any(c["originator"] == "codex_cli_rs" for c in vault.captures)
    # WS-first path: the prewarm should surface a WS upgrade (timing-dependent).
    assert any(c["kind"] == "ws" for c in vault.captures), (
        "no WebSocket upgrade reached the trap — codex may have gone straight to HTTP "
        f"fallback, or the prewarm didn't fire in time; captures={vault.captures}"
    )
