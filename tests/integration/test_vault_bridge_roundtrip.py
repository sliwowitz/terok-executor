# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The full agent transport: ``$PROVIDER_BASE_URL`` → in-container socat bridge → vault.

[`test_vault_route_roundtrip`][tests.integration.test_vault_route_roundtrip] proves the
generated phantom *functions* by calling the broker host-side.  This closes the last
gap — the exact path a real agent takes: the container reaches the vault at the
``ANTHROPIC_BASE_URL`` executor generated (``http://localhost:9419``), served by the
in-container socat loopback bridge that fronts the bind-mounted host vault socket
(``ensure-bridges.sh``, socket mode).  So it exercises executor's ``vault_addr`` wiring
and the bridge contract, not just phantom resolution.

Why the shape it has:

- The broker + mock upstream run on a **background event loop thread** because these
  integration tests are synchronous and the driving ``podman exec`` blocks — the broker
  must keep answering while it does.
- The container is launched through the real ``Sandbox.run`` (as executor launches
  one), but with an image carrying ``socat`` + ``curl`` and the host vault socket
  bind-mounted at [`CONTAINER_VAULT_SOCKET`][terok_executor.vault_addr.CONTAINER_VAULT_SOCKET];
  the keepalive command runs no init, so the test starts the socat bridge itself with
  the same command ``ensure-bridges.sh`` uses.

NOTE (validation): this is the highest-fidelity vault test but has the most moving
parts (background-thread broker, socat bridge timing, the socket bind-mount, an image
with socat).  It has only been checked by collection + lint off the test machine — run
it under the matrix (or a local podman host) to shake out timing/mount details before
relying on it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from aiohttp import web

from terok_executor.container.env import ContainerEnvSpec, assemble_container_env
from terok_executor.integrations.sandbox import RunSpec, VolumeSpec
from terok_executor.roster import AgentRoster
from terok_executor.vault_addr import CONTAINER_VAULT_SOCKET, LOOPBACK_VAULT_PORT
from tests.constants import (
    CONTAINER_KEEPALIVE_COMMAND,
    CONTAINER_WORKSPACE_DIR,
    INTEGRATION_VAULT_PASSPHRASE,
    PODMAN_BASE_IMAGE,
)

from .conftest import ExecutorEnv, hooks_missing, podman_missing
from .helpers import exec_in, podman, podman_rm, unique_container_name

pytestmark = [pytest.mark.needs_podman, podman_missing, hooks_missing]

PROVIDER = "anthropic"
REAL_SECRET = "sk-integration-bridge-not-for-the-container"  # nosec B105 — fixture value
CREDENTIAL_SCOPE = "integration-scope"
CREDENTIAL_SET = "default"
TASK_ID = "vault-bridge"
_SOCAT_CURL_IMAGE = "terok-executor-itest-socat:latest"


class _ThreadedVault:
    """A broker + mock upstream on a background event loop.

    A synchronous test can then drive a blocking ``podman exec`` against the broker
    (over the bind-mounted UNIX socket) while it keeps servicing requests.  Records the
    ``Authorization`` header the mock upstream ultimately saw, so the test can assert
    the phantom resolved to the real key.
    """

    def __init__(self, db_path: Path, routes_path: Path, socket_path: Path) -> None:
        self._db_path = db_path
        self._routes_path = routes_path
        self._socket_path = socket_path
        self.seen_auth: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runners: list[web.AppRunner] = []

    def __enter__(self) -> _ThreadedVault:
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self._thread.start()
        if not ready.wait(15):
            raise RuntimeError("vault broker did not come up on the background loop")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(15)

    def _run(self, ready: threading.Event) -> None:
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

        async def _echo(request: web.Request) -> web.Response:
            self.seen_auth = request.headers.get("Authorization", "")
            return web.json_response({"ok": True})

        mock_app = web.Application()
        mock_app.router.add_route("*", "/{tail:.*}", _echo)
        mock_runner = web.AppRunner(mock_app)
        await mock_runner.setup()
        mock_site = web.TCPSite(mock_runner, host="127.0.0.1", port=0)
        await mock_site.start()
        mock_port = mock_site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
        self._routes_path.write_text(
            json.dumps(
                {
                    PROVIDER: {
                        "upstream": f"http://127.0.0.1:{mock_port}",
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
        self._runners = [broker_runner, mock_runner]

    async def _cleanup(self) -> None:
        for runner in self._runners:
            await runner.cleanup()


@pytest.fixture(scope="session")
def socat_curl_image() -> str:
    """Build (once) an Alpine image with ``socat`` + ``curl`` for the bridge path."""
    import shutil

    if not shutil.which("podman"):
        pytest.skip("podman not on PATH")
    if podman("image", "exists", _SOCAT_CURL_IMAGE).returncode != 0:
        subprocess.run(
            ["podman", "build", "-t", _SOCAT_CURL_IMAGE, "-f", "-", "."],
            input=f"FROM {PODMAN_BASE_IMAGE}\nRUN apk add --no-cache socat curl\n",
            check=True,
            text=True,
            timeout=120,
        )
    return _SOCAT_CURL_IMAGE


@pytest.fixture
def launch_with_vault_socket(executor_env: ExecutorEnv, socat_curl_image: str) -> Iterator[object]:
    """Launch through the real ``Sandbox.run`` with a socat image + the vault socket mounted.

    Mirrors the shared ``launch`` fixture but overrides the image (the base image has no
    ``socat``) and bind-mounts the host broker socket at ``CONTAINER_VAULT_SOCKET`` — the
    hop the supervisor performs in production and that this test stands in for.
    """
    started: list[str] = []

    def _launch(host_socket: Path, env: dict[str, str], volumes: tuple[VolumeSpec, ...]) -> str:
        name = unique_container_name("vault-bridge")
        started.append(name)
        executor_env.sandbox().run(
            RunSpec(
                container_name=name,
                image=socat_curl_image,
                env=dict(env),
                volumes=(
                    VolumeSpec(executor_env.workspace_dir, CONTAINER_WORKSPACE_DIR),
                    VolumeSpec(host_socket, CONTAINER_VAULT_SOCKET),
                    *volumes,
                ),
                command=CONTAINER_KEEPALIVE_COMMAND,
                task_dir=executor_env.task_dir,
            )
        )
        return name

    try:
        yield _launch
    finally:
        for name in started:
            podman_rm(name)


def test_generated_base_url_reaches_the_vault_via_the_socat_bridge(
    executor_env: ExecutorEnv,
    roster: AgentRoster,
    launch_with_vault_socket,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request to ``$ANTHROPIC_BASE_URL`` from inside the container reaches the real key."""
    from terok_sandbox.config import SandboxConfig

    route = roster.vault_routes[PROVIDER]
    token_var = route.token_env["_default"]

    # 1. Seed the real credential in executor's vault DB and assemble the container env
    #    (mints the phantom + sets ANTHROPIC_BASE_URL to the loopback the bridge serves).
    db = executor_env.cfg.open_credential_db()
    try:
        db.store_credential(CREDENTIAL_SET, PROVIDER, {"type": "api_key", "key": REAL_SECRET})
    finally:
        db.close()
    spec = ContainerEnvSpec(
        task_id=TASK_ID,
        agent_name="claude",
        envs_dir=executor_env.mounts_dir,
        credential_scope=CREDENTIAL_SCOPE,
        credential_set=CREDENTIAL_SET,
        vault_required=True,
    )
    result = assemble_container_env(spec, roster, caller_manages_vault=False)
    assert route.base_url_env in result.env, "executor did not point the SDK at the vault"

    # 2. Broker (over executor's DB) + mock on a background loop; pin the passphrase the
    #    broker's _TokenDB resolves (executor_env patches only the adapter's SandboxConfig).
    monkeypatch.setattr(
        SandboxConfig,
        "resolve_passphrase_with_source",
        lambda _self: (INTEGRATION_VAULT_PASSPHRASE, "integration"),
    )
    host_socket = tmp_path / "vault.sock"
    with _ThreadedVault(
        executor_env.cfg.db_path, tmp_path / "broker-routes.json", host_socket
    ) as vault:
        name = launch_with_vault_socket(host_socket, result.env, result.volumes)

        # 3. Start the loopback bridge with the exact command ensure-bridges.sh uses, then
        #    hit $ANTHROPIC_BASE_URL from inside the container — the true agent path.
        bridge = (
            f"socat TCP-LISTEN:{LOOPBACK_VAULT_PORT},bind=127.0.0.1,fork,reuseaddr "
            f"UNIX-CONNECT:{CONTAINER_VAULT_SOCKET},retry=300,interval=0.1"
        )
        exec_in(name, "sh", "-c", f"setsid {bridge} >/tmp/socat.log 2>&1 &")
        # --retry-connrefused absorbs the race between the detached socat binding 9419
        # and curl connecting; the agent's own client would retry similarly.
        exec_in(
            name,
            "sh",
            "-c",
            "curl -sS --fail --retry 5 --retry-connrefused "
            '-H "Authorization: Bearer $' + token_var + '" '
            '"$' + route.base_url_env + '/v1/messages"',
        )

    assert vault.seen_auth == f"Bearer {REAL_SECRET}", (
        "the request to $ANTHROPIC_BASE_URL did not resolve the phantom to the real key"
    )
