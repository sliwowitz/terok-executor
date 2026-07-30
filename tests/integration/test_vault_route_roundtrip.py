# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""A provider route executor *generates* must yield a phantom that actually works.

[`test_vault_route_env`][tests.integration.test_vault_route_env] proves the phantom
executor mints for a route is *delivered* to the container and *resolves* in the vault
DB.  This adds the missing half: that same phantom, sent through a real token broker
wired to executor's vault DB, comes out the other side as the *real* provider key at a
mock upstream — the config-generation → working-credential contract, end to end, with
no real API key in the harness.

The container is launched exactly as executor launches one (``Sandbox.run`` via the
``launch`` fixture), and the token round-tripped through the broker is the one read
back out of the container's own environment — so delivery and function are asserted on
the *same* value.

The upstream leg (``ANTHROPIC_BASE_URL`` + the in-container socat loopback bridge that
the supervisor's init starts) is intentionally not exercised here: the keepalive
launch runs no init, and that socket-transport hop is already covered at its owning
layer by terok-sandbox's ``test_vault_container_transport``.  A follow-up that boots
the real init and curls ``$ANTHROPIC_BASE_URL`` from inside the container would close
that last gap (it needs ``socat`` in the image).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from terok_executor.container.env import ContainerEnvSpec, assemble_container_env
from terok_executor.roster import AgentRoster
from tests.constants import INTEGRATION_VAULT_PASSPHRASE

from .conftest import ExecutorEnv, Launcher, hooks_missing, podman_missing
from .helpers import exec_in

pytestmark = [pytest.mark.needs_podman, podman_missing, hooks_missing]

PROVIDER = "anthropic"
"""claude's default route — the roster's richest, matching test_vault_route_env."""

REAL_SECRET = "sk-integration-roundtrip-not-for-the-container"  # nosec B105 — fixture value
CREDENTIAL_SCOPE = "integration-scope"
CREDENTIAL_SET = "default"
TASK_ID = "vault-roundtrip"


async def _real_key_seen_by_upstream(db_path: Path, routes_path: Path, phantom: str) -> str:
    """Run a mock upstream + a broker over *db_path*; return the auth header the mock saw.

    The broker resolves the phantom against executor's real SQLCipher DB and injects the
    real credential, so the returned value is what a provider would actually receive.
    """
    from terok_sandbox.vault.daemon.token_broker import _build_app

    seen: dict[str, str] = {}

    async def _echo(request: web.Request) -> web.Response:
        seen["authorization"] = request.headers.get("Authorization", "")
        return web.json_response({"ok": True})

    upstream_app = web.Application()
    upstream_app.router.add_route("*", "/{tail:.*}", _echo)
    upstream_server = TestServer(upstream_app)
    await upstream_server.start_server()

    routes_path.write_text(
        json.dumps(
            {
                PROVIDER: {
                    "upstream": f"http://127.0.0.1:{upstream_server.port}",
                    "auth_header": "Authorization",
                    "auth_prefix": "Bearer ",
                }
            }
        )
    )

    broker_app = _build_app(str(db_path), str(routes_path))
    try:
        async with TestClient(TestServer(broker_app)) as client:
            resp = await client.get("/v1/messages", headers={"Authorization": f"Bearer {phantom}"})
            assert resp.status == 200, f"broker rejected the generated phantom: {resp.status}"
    finally:
        await upstream_server.close()
    return seen.get("authorization", "")


def test_generated_phantom_round_trips_to_the_real_key(
    executor_env: ExecutorEnv,
    roster: AgentRoster,
    launch: Launcher,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The phantom executor generates for the anthropic route works through the broker."""
    from terok_sandbox.config import SandboxConfig

    route = roster.vault_routes[PROVIDER]
    token_var = route.token_env["_default"]

    # 1. Seed the real credential in executor's own vault DB.
    db = executor_env.cfg.open_credential_db()
    try:
        db.store_credential(CREDENTIAL_SET, PROVIDER, {"type": "api_key", "key": REAL_SECRET})
    finally:
        db.close()

    # 2. Executor assembles the container env — this mints the phantom for the route.
    spec = ContainerEnvSpec(
        task_id=TASK_ID,
        agent_name="claude",
        envs_dir=executor_env.mounts_dir,
        credential_scope=CREDENTIAL_SCOPE,
        credential_set=CREDENTIAL_SET,
        vault_required=True,
    )
    result = assemble_container_env(spec, roster, caller_manages_vault=False)
    assert result.env[token_var] != REAL_SECRET

    # 3. Launch the container exactly as executor does, and read the phantom back out
    #    of its environment — delivery and function are asserted on the same value.
    name = launch("vault-roundtrip", env=result.env, volumes=result.volumes)
    container_phantom = exec_in(name, "printenv", token_var).strip()
    assert container_phantom == result.env[token_var]
    assert REAL_SECRET not in exec_in(name, "env"), "real credential leaked into the container"

    # 4. The broker opens executor's DB through the production passphrase chain; pin it
    #    to the config's throwaway passphrase (executor_env only patches the executor
    #    adapter's SandboxConfig, not the one the broker's _TokenDB constructs).
    monkeypatch.setattr(
        SandboxConfig,
        "resolve_passphrase_with_source",
        lambda _self, **_kw: (
            INTEGRATION_VAULT_PASSPHRASE,
            "integration",
        ),  # **_kw tolerates credentials_db/prompt_on_tty
    )

    seen_auth = asyncio.run(
        _real_key_seen_by_upstream(
            executor_env.cfg.db_path, tmp_path / "broker-routes.json", container_phantom
        )
    )
    assert seen_auth == f"Bearer {REAL_SECRET}", (
        "the generated phantom did not resolve to the real provider key at the upstream"
    )
