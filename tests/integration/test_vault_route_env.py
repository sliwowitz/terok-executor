# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""A configured provider's vault route must reach the container as env.

One provider, one route: a credential stored for ``anthropic`` becomes a
phantom token in the vault DB, the route names the env var that carries it
(``ANTHROPIC_API_KEY``) and the var that points the SDK at the vault proxy
(``ANTHROPIC_BASE_URL``), and ``podman run -e`` has to deliver both intact.

What a mock cannot check, and this does:

- the token in the container is the *phantom*, and the real secret never
  crosses the boundary — asserted against the container's whole env, not
  against the dict executor built;
- the phantom resolves back through the real SQLCipher vault to exactly the
  provider that was configured, so the route is wired to one credential and
  not another.
"""

from __future__ import annotations

import pytest

from terok_executor.container.env import ContainerEnvSpec, assemble_container_env
from terok_executor.roster import AgentRoster

from .conftest import ExecutorEnv, Launcher, hooks_missing, podman_missing
from .helpers import exec_in

pytestmark = [pytest.mark.needs_podman, podman_missing, hooks_missing]

PROVIDER = "anthropic"
"""The provider under test — claude's default, and the roster's richest route."""

REAL_SECRET = "sk-integration-not-for-the-container"  # nosec B105 — fixture, not a real key
"""The credential stored in the vault.  It must never appear in a container."""

CREDENTIAL_SCOPE = "integration-scope"
CREDENTIAL_SET = "default"
TASK_ID = "vault-route"


def test_provider_credential_reaches_the_container_as_a_phantom_token(
    executor_env: ExecutorEnv,
    roster: AgentRoster,
    launch: Launcher,
) -> None:
    """The anthropic route delivers a vault phantom token — never the secret."""
    route = roster.vault_routes[PROVIDER]
    token_var = route.token_env["_default"]

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

    phantom = result.env[token_var]
    assert phantom != REAL_SECRET

    name = launch("vault-route", env=result.env, volumes=result.volumes)

    assert exec_in(name, "printenv", token_var).strip() == phantom
    assert exec_in(name, "printenv", route.base_url_env).strip(), "route left the SDK unpointed"

    container_env = exec_in(name, "env")
    assert REAL_SECRET not in container_env, "the real credential leaked into the container"

    # The phantom the container holds resolves — through the real vault DB —
    # to the one provider that was configured.  This is the route.
    db = executor_env.cfg.open_credential_db()
    try:
        resolved = db.lookup_token(phantom)
    finally:
        db.close()

    assert resolved is not None, "the container's token is unknown to the vault"
    assert resolved["provider"] == PROVIDER
    assert resolved["scope"] == CREDENTIAL_SCOPE
    assert resolved["credential_set"] == CREDENTIAL_SET
