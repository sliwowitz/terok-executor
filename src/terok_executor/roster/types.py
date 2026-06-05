# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Runtime dataclasses produced by the agent roster loader.

These are the immutable result types that consumers (env builder,
auth flow, image build) receive after a YAML file passes the
[`schema`][terok_executor.roster.schema] validation gate.  Kept in
their own module so both [`schema`][terok_executor.roster.schema]
(which projects onto them) and [`loader`][terok_executor.roster.loader]
(which orchestrates the projection) can import without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args


@dataclass(frozen=True)
class MountDef:
    """A shared directory mount derived from the agent roster."""

    host_dir: str
    """Directory name under ``mounts_dir()`` (e.g. ``"_codex-config"``)."""

    container_path: str
    """Mount point inside the container (e.g. ``"/home/dev/.codex"``)."""

    label: str
    """Human-readable label (e.g. ``"Codex config"``)."""

    credential_file: str = ""
    """Credential file path relative to the mount root (e.g. ``".credentials.json"``).

    Empty when the mount carries no auth artefact (e.g. opencode state dirs).
    Populated from the matching ``vault.credential_file`` so callers can
    layer a read-only shadow over the file without touching the rest of
    the shared mount.  See [terok-ai/terok#873](https://github.com/terok-ai/terok/issues/873).
    """

    provider: str = ""
    """Roster entry name that contributed this mount (e.g. ``"claude"``).

    Empty for explicit ``mounts:`` blocks that aren't tied to a single
    provider.  Used by the credential-shadow path to match against
    [`ContainerEnvSpec.expose_credential_providers`][terok_executor.ContainerEnvSpec.expose_credential_providers].
    """


@dataclass(frozen=True)
class VaultRoute:
    """Vault route config parsed from a ``vault:`` YAML section.

    Used to generate the ``routes.json`` that the vault server reads.
    """

    provider: str
    """Agent/tool name (e.g. ``"claude"``)."""

    route_prefix: str
    """Path prefix in the proxy (e.g. ``"claude"`` → ``/claude/v1/...``)."""

    upstream: str
    """Upstream API base URL (e.g. ``"https://api.anthropic.com"``)."""

    path_upstreams: dict[str, str] = field(default_factory=dict)
    """Optional request-path prefix → upstream-base overrides."""

    oauth_extra_headers: dict[str, str] = field(default_factory=dict)
    """Provider-specific headers added only when forwarding OAuth credentials."""

    auth_header: str = "Authorization"
    """HTTP header name for the real credential."""

    auth_prefix: str = "Bearer "
    """Prefix before the token value in the auth header."""

    credential_type: str = "api_key"
    """Type of credential: ``"oauth"``, ``"api_key"``, ``"oauth_token"``, ``"pat"``."""

    credential_file: str = ""
    """Credential file path relative to the auth mount."""

    token_env: dict[str, str] = field(default_factory=dict)
    """Phantom-token env var name, keyed by stored credential type.

    The named env var carries the phantom token the agent reads in place of
    the real credential.  Keys are credential types (``"oauth"``, ``"pat"``,
    …); ``"_default"`` is the fallback for any type without an explicit
    entry.  Most agents read one env var regardless of type
    (``{"_default": "MISTRAL_API_KEY"}``); Claude swaps the name when an
    OAuth token is stored
    (``{"oauth": "CLAUDE_CODE_OAUTH_TOKEN", "_default": "ANTHROPIC_API_KEY"}``).
    """

    base_url_env: str = ""
    """Env var to override with the vault's HTTP URL (e.g. ``"ANTHROPIC_BASE_URL"``)."""

    socket_env: str = ""
    """Env var that receives the container-side vault socket path.

    Set when the agent speaks HTTP-over-UNIX natively (e.g. Claude reads
    ``ANTHROPIC_UNIX_SOCKET``).  The resolved value is mode-dependent and
    injected centrally by the env builder.
    """

    shared_config_patch: dict | None = None
    """Optional shared config patch applied after auth (e.g. Vibe's config.toml)."""

    oauth_refresh: dict | None = None
    """OAuth refresh config: ``{token_url, client_id, scope}``."""

    shared_domain: bool = False
    """Whether the upstream host also serves non-API traffic.

    Set on entries whose ``upstream`` host is an apex (or otherwise mixed)
    domain that legitimately serves docs, dashboards, ``git push``, etc.
    Host-level egress denies can't separate paths, so terok's auth-protect
    layer skips these providers when re-applying denies after ``shield
    down`` — credential containment alone keeps the API safe.

    Examples: ``gitlab.com`` (API + ``git push``), ``sonarcloud.io``
    (API + project pages + docs + badges).
    """


@dataclass(frozen=True)
class InstallSpec:
    """Roster-driven install snippets emitted into the L1 Dockerfile.

    The build template loops over the resolved selection and concatenates
    ``run_as_root`` snippets in the root section, ``run_as_dev`` snippets
    in the dev-user section.  Both fields are raw Dockerfile fragments
    (``RUN``, ``COPY`` — anything valid at top level after ``USER ...``).
    ``depends_on`` lists other roster names that must be installed
    alongside this one (transitively resolved at selection time).
    """

    depends_on: tuple[str, ...] = ()
    """Other roster entries this install requires (e.g. ``blablador → opencode``)."""

    run_as_root: str = ""
    """Dockerfile fragment emitted in the root section of the L1 image."""

    run_as_dev: str = ""
    """Dockerfile fragment emitted in the dev-user section of the L1 image."""


HelpSection = Literal["agent", "dev_tool"]
"""Section in the in-container help banner that an entry belongs to."""

HELP_SECTIONS: tuple[HelpSection, ...] = get_args(HelpSection)
"""All valid [`HelpSection`][terok_executor.roster.types.HelpSection] values, as a tuple (single source of truth)."""


@dataclass(frozen=True)
class HelpSpec:
    """One-line entry shown in the in-container help banner."""

    label: str = ""
    """Raw banner line (the agent owns its formatting, including ANSI codes)."""

    section: HelpSection = "agent"


@dataclass(frozen=True)
class SidecarSpec:
    """Sidecar container configuration parsed from a ``sidecar:`` YAML section.

    Tools with sidecar specs run in a separate lightweight L1 image
    (no agent CLIs) and receive the real API key instead of phantom tokens.
    """

    tool_name: str
    """Tool identifier used to select the Jinja2 install block in the template."""

    env_map: dict[str, str] = field(default_factory=dict)
    """Maps container env var names to credential dict keys.

    Example: ``{"CODERABBIT_API_KEY": "key"}`` reads ``cred["key"]`` and
    injects it as ``CODERABBIT_API_KEY``.
    """


@dataclass(frozen=True)
class ProviderAuth:
    """How a provider attaches the real credential to an upstream request.

    One instance per supported mode.  A provider may carry an OAuth mode, an
    API-key mode, or both — see [`Provider.wire_auth`][terok_executor.roster.types.Provider.wire_auth].
    """

    header: str
    """HTTP header that carries the credential (e.g. ``"Authorization"``, ``"x-api-key"``)."""

    prefix: str = ""
    """String prepended to the token value (e.g. ``"Bearer "``, ``"token "``)."""

    extra_headers: dict[str, str] = field(default_factory=dict)
    """Headers added only for this mode (e.g. Anthropic's ``anthropic-beta``)."""


@dataclass(frozen=True)
class Provider:
    """A vault-routed upstream — an LLM endpoint or a tool API.

    Carries the *endpoint* concern lifted out of the agent ``vault:`` blocks:
    where requests go (``upstream`` / ``path_upstreams``) and how the real
    credential is attached (``oauth_auth`` / ``api_key_auth``).  This is the
    single source the ``routes.json`` consumed by the sandbox vault is
    generated from.  The per-agent *delivery* concern (which env var receives
    the phantom token or base URL) stays on the agent binding, not here.
    """

    name: str
    """Clean provider name — also the ``routes.json`` key and the vault DB
    ``credentials.provider`` value (e.g. ``"anthropic"``, ``"github"``)."""

    upstream: str
    """Upstream API base URL (e.g. ``"https://api.anthropic.com"``)."""

    api_key_auth: ProviderAuth | None = None
    """Wire auth used when an API key is the stored credential, if supported."""

    oauth_auth: ProviderAuth | None = None
    """Wire auth used when an OAuth token is the stored credential, if supported."""

    path_upstreams: dict[str, str] = field(default_factory=dict)
    """Request-path prefix → upstream-base overrides (e.g. Codex's ``/backend-api/``)."""

    oauth_refresh: dict[str, str] | None = None
    """OAuth refresh config: ``{token_url, client_id, scope}``."""

    shared_domain: bool = False
    """Whether the upstream host also serves non-API traffic (docs, ``git push``…).

    See [`VaultRoute.shared_domain`][terok_executor.roster.types.VaultRoute.shared_domain].
    """

    serves: dict[str, str] = field(default_factory=dict)
    """Wire protocol → container-facing base path (LLM providers only).

    Empty for tool providers (github, gitlab, …).  A provider that serves more
    than one protocol (openrouter: ``anthropic-messages`` + ``openai-chat``)
    backs several agent protocols from one endpoint.  Consumed by the wrapper
    when resolving an agent×provider combo; not part of the ``routes.json``
    contract.
    """

    def wire_auth(self) -> tuple[str, str, dict[str, str]]:
        """Resolve the ``routes.json`` ``(auth_header, auth_prefix, oauth_extra_headers)``.

        When the provider offers both OAuth and API-key auth under *different*
        headers (Anthropic: ``Authorization: Bearer`` for OAuth vs ``x-api-key``
        for the key), the vault must pick the header *and* prefix per stored
        credential type at request time — signalled by the sentinel header
        ``"dynamic"`` with an empty prefix.  A single mode emits its header and
        prefix verbatim.

        Two modes sharing a header are only representable if they're wire-
        identical: ``routes.json`` carries a single ``auth_prefix`` /
        ``oauth_extra_headers``, so modes on the same header but with different
        prefixes (or extra headers) can't both be serialised — that raises
        rather than silently wiring one mode with the other's prefix.

        This reproduces the historical ``auth_header: dynamic`` wire contract
        without the magic string ever appearing in a provider definition.
        """
        oauth, api_key = self.oauth_auth, self.api_key_auth
        if oauth is not None and api_key is not None:
            if oauth.header != api_key.header:
                # Distinct headers → the vault selects header + prefix by the
                # stored credential type at request time.
                return "dynamic", "", dict(oauth.extra_headers)
            if oauth.prefix != api_key.prefix or oauth.extra_headers != api_key.extra_headers:
                raise ValueError(
                    f"Provider {self.name!r} declares oauth and api_key auth on the same "
                    f"header ({oauth.header!r}) but with different prefix/extra_headers; the "
                    f"routes.json contract carries only one, so one mode would be wired with "
                    f"the other's prefix. Give the modes distinct headers or unify them."
                )
        mode = oauth or api_key
        if mode is None:
            raise ValueError(f"Provider {self.name!r} declares no auth mode")
        return mode.header, mode.prefix, dict(mode.extra_headers)
