# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Assembles container environment variables and volume mounts for agent launches.

Both ``terok-executor run`` (standalone) and ``terok`` (project orchestrator)
construct identical container environments — shared config mounts, credential
proxy tokens, git identity, unrestricted-mode flags.  This module provides
the canonical assembly function so that logic lives in one place.

Usage::

    from terok_executor.container.env import ContainerEnvSpec, assemble_container_env
    from terok_executor import get_roster

    result = assemble_container_env(
        ContainerEnvSpec(task_id="abc", provider_name="claude", workspace_host_path=ws),
        get_roster(),
    )
    # result.env, result.volumes, result.task_dir
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from terok_sandbox import Sharing, VolumeSpec

if TYPE_CHECKING:
    from terok_sandbox import CredentialDB, SandboxConfig

    from terok_executor.roster.loader import AgentRoster

_logger = logging.getLogger(__name__)


# ── Vocabulary ──


@dataclass(frozen=True)
class ContainerEnvSpec:
    """Specification for container environment assembly.

    All fields use primitives or :class:`~pathlib.Path` — no terok-specific
    types.  Callers pre-resolve domain-specific decisions (security class,
    authorship mode, SSH mount, gate mirror creation) and pass results here.
    """

    # -- Required ----------------------------------------------------------

    task_id: str
    """Unique task identifier."""

    provider_name: str
    """Agent provider name (e.g. ``"claude"``, ``"codex"``)."""

    workspace_host_path: Path
    """Host-side workspace directory — caller pre-creates, mounted as ``/workspace:Z``."""

    # -- Repository setup --------------------------------------------------

    code_repo: str | None = None
    """Git URL to clone inside the container (→ ``CODE_REPO``)."""

    clone_from: str | None = None
    """Secondary clone source for online-mode gate optimization (→ ``CLONE_FROM``)."""

    branch: str | None = None
    """Git branch to check out (→ ``GIT_BRANCH``)."""

    # -- Git identity (container-level defaults) ---------------------------

    git_author_name: str | None = None
    """Resolved from roster provider if ``None``."""

    git_author_email: str | None = None
    git_committer_name: str | None = None
    git_committer_email: str | None = None

    # -- Authorship mode (for per-agent shell wrappers) --------------------

    authorship: str = "agent"
    """Authorship mode consumed by in-container wrappers (→ ``TEROK_GIT_AUTHORSHIP``)."""

    human_name: str = "Nobody"
    """Human operator name (→ ``HUMAN_GIT_NAME``).  terok resolves from project
    config / ``git config``; standalone uses the default or ``--git-identity-from-host``."""

    human_email: str = "nobody@localhost"
    """Human operator email (→ ``HUMAN_GIT_EMAIL``)."""

    # -- Credential proxy --------------------------------------------------

    credential_scope: str = "standalone"
    """Scope for proxy token creation.  terok passes ``project.id``."""

    proxy_transport: Literal["direct", "socket"] = "direct"
    """Proxy transport mode: ``"direct"`` (HTTP base URL) or ``"socket"``
    (Unix socket path via :attr:`~CredentialProxyRoute.socket_env`)."""

    proxy_required: bool = False
    """When ``True``, raise ``SystemExit`` if the credential proxy is
    unreachable.  When ``False`` (default), soft-fail to empty env."""

    scan_leaked_creds: bool = False
    """When ``True``, scan shared mounts for real credential files and emit
    warnings.  Standalone mode defaults to off; terok enables this."""

    # -- Permissions -------------------------------------------------------

    unrestricted: bool = True
    """Enable auto-approve flags for all agents."""

    # -- Agent config ------------------------------------------------------

    agent_config_dir: Path | None = None
    """Pre-prepared agent config directory (→ ``/home/dev/.terok:Z``)."""

    # -- Shared task directory (multi-agent IPC) ---------------------------

    shared_dir: Path | None = None
    """Host-side shared directory.  Created by the assembly function if set."""

    shared_mount: str = "/shared"
    """Container-side mount point for the shared directory."""

    # -- Directories -------------------------------------------------------

    task_dir: Path | None = None
    """Host-side task directory.  A temp dir is created if ``None``."""

    envs_dir: Path | None = None
    """Base directory for shared config mounts.  Uses :func:`paths.mounts_dir`
    if ``None``."""

    # -- Caller-specific mounts --------------------------------------------

    extra_volumes: tuple[VolumeSpec, ...] = ()
    """Additional volume specs from the caller (e.g. SSH mounts from terok)."""


@dataclass(frozen=True)
class ContainerEnvResult:
    """Assembled container environment ready for RunSpec construction.

    Not a ``RunSpec`` — omits launch-time concerns (container name, image,
    command, GPU, shield bypass).  Callers add those and construct ``RunSpec``.
    """

    env: dict[str, str]
    """Environment variables for the container."""

    volumes: tuple[VolumeSpec, ...]
    """Typed volume specs — the sandbox decides whether to mount or inject."""

    task_dir: Path
    """Host-side task directory.  When ``spec.task_dir`` was ``None``, this is
    an auto-created temporary directory — the **caller owns cleanup**."""


# ── Public API ──


def assemble_container_env(
    spec: ContainerEnvSpec,
    roster: AgentRoster,
    *,
    caller_manages_proxy: bool = False,
) -> ContainerEnvResult:
    """Assemble container environment variables and volume mounts.

    This is the **single source of truth** for container env/volume assembly.
    Both ``AgentRunner._run()`` and terok's ``build_task_env_and_volumes()``
    delegate here.

    Args:
        spec: What the caller wants — all host↔container contract fields.
        roster: Agent roster for shared mounts, proxy routes, provider identity.
        caller_manages_proxy: When ``True``, skip phantom-token injection
            here — the caller injects richer proxy tokens itself (e.g.
            terok's per-provider OAuth tiers, socket transport, SSH agent).
            Shared config patches (``api_base`` rewrites) still run because
            the credential proxy **is** in use; only token injection is
            delegated.

    Returns:
        Assembled env dict, volume tuple, and resolved task_dir.
    """
    from terok_executor.paths import mounts_dir as _mounts_dir

    env: dict[str, str] = {}
    volumes: list[VolumeSpec] = []

    # 1. Base env
    env["TASK_ID"] = spec.task_id
    env["REPO_ROOT"] = "/workspace"
    env["GIT_RESET_MODE"] = "none"
    env["CLAUDE_CONFIG_DIR"] = "/home/dev/.claude"

    # 2. OpenCode provider env
    env.update(roster.collect_opencode_provider_env())

    # 3. Git identity
    env.update(_resolve_git_identity(spec, roster))

    # 4. Authorship env (for per-agent wrappers inside container)
    env["TEROK_GIT_AUTHORSHIP"] = spec.authorship
    env["HUMAN_GIT_NAME"] = spec.human_name
    env["HUMAN_GIT_EMAIL"] = spec.human_email

    # 5. Branch
    if spec.branch:
        env["GIT_BRANCH"] = spec.branch

    # 6. Repo URLs
    if spec.code_repo:
        env["CODE_REPO"] = spec.code_repo
    if spec.clone_from:
        env["CLONE_FROM"] = spec.clone_from

    # 7. Workspace volume
    volumes.append(VolumeSpec(spec.workspace_host_path, "/workspace", sharing=Sharing.PRIVATE))

    # 8. Shared config mounts from roster
    mounts_base = spec.envs_dir or _mounts_dir()
    volumes += _shared_config_mounts(roster, mounts_base)

    # 8b. Re-apply proxy config patches (idempotent — ensures shared mount
    #     dirs contain correct proxy URLs even after state wipe).
    #
    #     NOT gated by caller_manages_proxy: that flag only skips
    #     phantom-token injection here because the caller (terok) injects
    #     richer tokens itself — the credential proxy is still in use and
    #     agents still need their config files rewritten to route through
    #     it.  Providers whose credential is exposed directly (Claude OAuth
    #     tier 3) are safe because they have no shared_config_patch.
    from terok_executor.credentials.proxy_config import apply_shared_config_patches

    apply_shared_config_patches(roster, mounts_base)

    # 9. Credential proxy
    if not caller_manages_proxy:
        env.update(
            _inject_proxy_tokens(
                roster,
                spec.credential_scope,
                spec.task_id,
                proxy_transport=spec.proxy_transport,
                proxy_required=spec.proxy_required,
            )
        )

    # 9b. Leaked credential scan (runs regardless of caller_manages_proxy —
    #     the shared mounts exist either way)
    if spec.scan_leaked_creds:
        from terok_executor.credentials.proxy_commands import scan_leaked_credentials

        leaked = scan_leaked_credentials(mounts_base)
        for provider, path in leaked:
            _logger.warning("Real credential in shared mount: %s: %s", provider, path)

    # 10. Agent config mount
    if spec.agent_config_dir:
        volumes.append(
            VolumeSpec(spec.agent_config_dir, "/home/dev/.terok", sharing=Sharing.PRIVATE)
        )

    # 11. Unrestricted mode
    if spec.unrestricted:
        env["TEROK_UNRESTRICTED"] = "1"
        env.update(roster.collect_all_auto_approve_env())

    # 12. Shared task directory
    if spec.shared_dir:
        spec.shared_dir.mkdir(parents=True, exist_ok=True)
        volumes.append(VolumeSpec(spec.shared_dir, spec.shared_mount))
        env["TEROK_SHARED_DIR"] = spec.shared_mount

    # 13. Extra volumes
    volumes.extend(spec.extra_volumes)

    # Resolve task_dir
    task_dir = spec.task_dir or Path(tempfile.mkdtemp(prefix=f"terok-executor-{spec.task_id}-"))

    return ContainerEnvResult(env=env, volumes=tuple(volumes), task_dir=task_dir)


# ── Private helpers ──


def _resolve_git_identity(spec: ContainerEnvSpec, roster: AgentRoster) -> dict[str, str]:
    """Resolve the four git identity env vars.

    Uses explicit spec fields when provided, otherwise falls back to the
    roster provider's configured identity (same name for both author and
    committer — standalone default).
    """
    provider = roster.providers.get(spec.provider_name)

    author_name = spec.git_author_name or (provider.git_author_name if provider else "AI Agent")
    author_email = spec.git_author_email or (
        provider.git_author_email if provider else "ai@localhost"
    )
    committer_name = spec.git_committer_name or author_name
    committer_email = spec.git_committer_email or author_email

    return {
        "GIT_AUTHOR_NAME": author_name,
        "GIT_AUTHOR_EMAIL": author_email,
        "GIT_COMMITTER_NAME": committer_name,
        "GIT_COMMITTER_EMAIL": committer_email,
    }


def _shared_config_mounts(roster: AgentRoster, mounts_base: Path) -> list[VolumeSpec]:
    """Derive shared volume specs from the agent roster.

    Uses ``roster.mounts`` — the already-deduplicated list that merges auth
    provider mounts and explicit ``mounts:`` YAML sections.  Creates host
    directories on demand.
    """
    specs: list[VolumeSpec] = []
    for m in roster.mounts:
        host_dir = mounts_base / m.host_dir
        host_dir.mkdir(parents=True, exist_ok=True)
        specs.append(VolumeSpec(host_dir, m.container_path))
    return specs


def _inject_proxy_tokens(
    roster: AgentRoster,
    scope: str,
    task_id: str,
    *,
    proxy_transport: Literal["direct", "socket"] = "direct",
    proxy_required: bool = False,
) -> dict[str, str]:
    """Inject credential proxy phantom tokens if the proxy is running.

    Handles three orthogonal concerns:

    - **Auth**: selects ``phantom_env`` vs ``oauth_phantom_env`` based on the
      stored credential type (Phase 1).
    - **Transport**: injects ``socket_env``/``socket_path`` when
      *proxy_transport* is ``"socket"`` (Phase 2).
    - **SSH agent**: creates a phantom token for the SSH agent proxy when the
      scope has registered SSH keys (Phase 3).

    When *proxy_required* is ``False`` (standalone default), soft-fails to
    empty dict.  When ``True`` (terok project mode), raises ``SystemExit``
    if the proxy is unreachable.
    """
    from terok_sandbox import (
        CredentialDB,
        SandboxConfig,
        get_proxy_port,
        get_ssh_agent_port,
        is_proxy_running,
        is_proxy_socket_active,
    )

    cfg = SandboxConfig()
    if not (is_proxy_socket_active() or is_proxy_running(cfg)):
        if proxy_required:
            raise SystemExit(
                "Credential proxy is not running.\n\n"
                "Start it with:\n"
                "  terok credential-proxy install   (systemd socket activation)\n"
                "  terok credential-proxy start     (manual daemon)"
            )
        return {}

    proxy_routes = roster.proxy_routes
    try:
        db = CredentialDB(cfg.proxy_db_path)
    except Exception:
        _logger.exception("Credential proxy DB unavailable")
        if proxy_required:
            raise SystemExit(
                "Credential proxy DB unavailable. Check logs for details.\n\n"
                "Start it with:\n"
                "  terok credential-proxy install   (systemd socket activation)\n"
                "  terok credential-proxy start     (manual daemon)"
            ) from None
        return {}

    try:
        credential_set = "default"
        stored = set(db.list_credentials(credential_set))
        routed = stored & proxy_routes.keys()

        # SSH agent token is independent of provider credentials — a project
        # with SSH keys but no API creds should still get TEROK_SSH_AGENT_*.
        ssh_token = _load_ssh_agent_token(db, cfg, scope, task_id)

        if not routed and not ssh_token:
            return {}

        credential_types: dict[str, str] = {}
        tokens: dict[str, str] = {}
        for name in routed:
            cred = db.load_credential(credential_set, name)
            credential_types[name] = (cred.get("type") if cred else None) or "api_key"
            tokens[name] = db.create_proxy_token(scope, task_id, credential_set, name)

        port = get_proxy_port(cfg)
    except Exception:
        _logger.exception("Credential proxy token injection failed")
        if proxy_required:
            raise SystemExit(
                "Credential proxy token injection failed. Check logs for details.\n\n"
                "Start it with:\n"
                "  terok credential-proxy install   (systemd socket activation)\n"
                "  terok credential-proxy start     (manual daemon)"
            ) from None
        return {}
    finally:
        db.close()

    use_socket = proxy_transport == "socket"
    proxy_base = f"http://host.containers.internal:{port}"
    env: dict[str, str] = {}

    for name, route in proxy_routes.items():
        if name not in routed:
            continue

        is_oauth = credential_types.get(name) == "oauth"
        token_vars = (
            route.oauth_phantom_env if (is_oauth and route.oauth_phantom_env) else route.phantom_env
        )
        for env_var in token_vars:
            env[env_var] = tokens[name]

        if use_socket and route.socket_path and route.socket_env:
            env[route.socket_env] = route.socket_path
        if route.base_url_env:
            env[route.base_url_env] = proxy_base

        # OpenCode base URL override for proxied providers
        provider = roster.providers.get(name)
        if provider and provider.opencode_config:
            env[f"TEROK_OC_{name.upper()}_BASE_URL"] = f"{proxy_base}/v1"
        # glab: redirect API to proxy
        if name == "glab":
            env["GITLAB_API_HOST"] = f"host.containers.internal:{port}"
            env["API_PROTOCOL"] = "http"

    if routed:
        env["TEROK_PROXY_PORT"] = str(port)

    if ssh_token:
        env["TEROK_SSH_AGENT_TOKEN"] = ssh_token
        if use_socket:
            env["TEROK_SSH_AGENT_SOCKET"] = str(cfg.ssh_agent_socket_path)
        else:
            env["TEROK_SSH_AGENT_PORT"] = str(get_ssh_agent_port(cfg))

    _logger.debug("Credential proxy: injected %d env vars for %s", len(env), routed)
    return env


def _load_ssh_keys_json(path: Path) -> dict:
    """Load the SSH key mapping JSON.  Returns empty dict on failure."""
    import json

    if not path.is_file():
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        _logger.warning("Cannot read SSH keys file %s: %s", path, exc)
        return {}


def _load_ssh_agent_token(
    db: CredentialDB, cfg: SandboxConfig, scope: str, task_id: str
) -> str | None:
    """Create an SSH agent phantom token if *scope* has valid keys registered."""
    ssh_keys = _load_ssh_keys_json(cfg.ssh_keys_json_path)
    ssh_entry = ssh_keys.get(scope)
    if not isinstance(ssh_entry, list):
        return None
    for entry in ssh_entry:
        if not isinstance(entry, dict):
            _logger.warning(
                "Malformed entry in ssh-keys.json for scope %r: expected dict, got %s",
                scope,
                type(entry).__name__,
            )
            return None
    if any(e.get("private_key") and e.get("public_key") for e in ssh_entry):
        return db.create_proxy_token(scope, task_id, scope, "ssh")
    return None
