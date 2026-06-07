# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Andreas Knüpfer
# SPDX-License-Identifier: Apache-2.0

"""Authenticates AI coding agents via OAuth or API key.

Two public entry points:

- ``authenticate(project_id, provider, *, mounts_dir, image)`` — dispatches
  based on the provider's ``modes`` field: prompts for an API key (no
  container) or launches an auth container with the vendor CLI.
- ``store_api_key(provider, api_key)`` — stores an API key directly in the
  credential DB (non-interactive fast path for CI).

``AUTH_PROVIDERS`` is a registry dict populated from the YAML roster at
package load time; ``authenticate`` looks up the provider by name and
delegates to the matching flow.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from terok_util import podman_userns_args

from terok_executor.integrations.sandbox import (
    CODEX_SHARED_OAUTH_MARKER,
    PHANTOM_CREDENTIALS_MARKER,
)

# Two consoles so colored success messages flow to stdout while
# colored errors / warnings flow to stderr.  Rich auto-disables colors
# on non-TTY streams and honors ``NO_COLOR=1``.
_out = Console()
_err = Console(stderr=True)

# ── Vocabulary ──


@dataclass(frozen=True)
class AuthProvider:
    """Describes how to authenticate one tool/agent."""

    name: str
    """Short key used in CLI and TUI dispatch (e.g. ``"codex"``)."""

    label: str
    """Human-readable display name (e.g. ``"Codex"``)."""

    host_dir_name: str
    """Single-segment directory name under ``mounts_dir()`` (e.g. ``"_codex-config"``)."""

    container_mount: str
    """Mount point inside the container (e.g. ``"/home/dev/.codex"``)."""

    command: list[str]
    """Command to execute inside the container (OAuth mode only)."""

    banner_hint: str
    """Provider-specific help text shown before the container runs."""

    extra_run_args: tuple[str, ...] = field(default_factory=tuple)
    """Additional ``podman run`` arguments (e.g. port forwarding)."""

    modes: tuple[str, ...] = ("api_key",)
    """Supported auth modes: ``"oauth"`` (container), ``"api_key"`` (fast path)."""

    device_auth: bool = False
    """Whether the OAuth flow has a headless device-code variant.

    When set, the auth prompt offers a device-code method alongside the normal
    OAuth login: it runs ``<command> --device-auth`` with the normal flow's
    port-forwarding (``extra_run_args``) dropped — the device-code flow polls a
    remote endpoint and needs no localhost callback, so it works where the
    operator can't complete the forwarded-port login."""

    api_key_hint: str = ""
    """Hint shown when prompting for an API key (URL to get one)."""

    post_capture_state: dict[str, dict] = field(default_factory=dict)
    """JSON state files to write after credential capture.

    Maps filename → key-value dict to merge into a JSON file in the auth
    mount directory.  Example: ``{".claude.json": {"hasCompletedOnboarding": true}}``
    marks Claude Code onboarding as complete so the first-run wizard is skipped.
    """

    credential_provider: str = ""
    """Provider name the captured credential is stored under (e.g. ``"anthropic"``).

    ``terok-executor auth claude`` authenticates Claude's *default provider*, so
    the credential is keyed under ``anthropic`` — matching the vault route and
    the v3-migrated DB.  Populated by the roster loader from the agent's
    ``provider.default`` binding; empty falls back to the entry's own name.
    """

    def __post_init__(self) -> None:
        """Validate fields that become filesystem paths."""
        p = Path(self.host_dir_name)
        if p.is_absolute() or ".." in p.parts or len(p.parts) != 1:
            raise ValueError(
                f"host_dir_name must be a single directory segment, got {self.host_dir_name!r}"
            )

    @property
    def supports_oauth(self) -> bool:
        """Whether this provider supports OAuth (container-based) auth."""
        return "oauth" in self.modes

    @property
    def supports_api_key(self) -> bool:
        """Whether this provider supports direct API key entry."""
        return "api_key" in self.modes

    @property
    def supports_device_auth(self) -> bool:
        """Whether this provider offers a headless device-code login variant."""
        return self.device_auth


@dataclass(frozen=True)
class AuthKeyConfig:
    """Describes how to prompt for and store an API key."""

    label: str
    """Human name shown in the prompt (e.g. ``"Claude"``)."""

    key_url: str
    """URL where the user can obtain the key."""

    env_var: str
    """Name shown in the ``read -p`` prompt (e.g. ``"ANTHROPIC_API_KEY"``)."""

    config_path: str
    """Destination inside the container (e.g. ``"~/.claude/config.json"``)."""

    printf_template: str
    """``printf`` format string (e.g. ``'{\"api_key\": \"%s\"}'``)."""

    tool_name: str
    """Name shown in the success message (e.g. ``"claude"``)."""


AUTH_PROVIDERS: dict[str, AuthProvider] = {}
"""All known auth providers (agents + tools), keyed by name.  Loaded from ``resources/agents/*.yaml``."""


# ── Public API ──


@dataclass(frozen=True)
class Authenticator:
    """Vendor-credential acquisition for a single agent.

    Wraps the ``authenticate`` flow behind a stable class so callers
    that orchestrate a multi-step setup (terok project init, the
    standalone ``terok-executor auth`` command, the TUI auth flow)
    talk to one named surface bound to ``self.provider``.

    The discovery counterparts
    ([`list_authenticated_agents`][terok_executor.list_authenticated_agents],
    [`scan_leaked_credentials`][terok_executor.scan_leaked_credentials])
    stay as module-level fns in their owning submodules — folding them
    in here would create a tach cycle through ``terok_executor.acp``
    and ``terok_executor.credentials.vault_commands``, which already
    depend on this module transitively.
    """

    provider: str
    """Auth provider name (e.g. ``"claude"``)."""

    def run(
        self,
        project_id: str | None,
        *,
        mounts_dir: Path,
        image: str | Callable[[], str] | None = None,
        expose_token: bool = False,
        oauth_enabled: bool = True,
        credential_set: str = "default",
    ) -> None:
        """Run the auth flow for ``self.provider``; see module-level docs.

        Mirrors the parameters of the underlying ``authenticate`` free
        function — instance-bound ``self.provider`` replaces the old
        positional ``provider`` arg.  The device-code login is offered as a
        method in that flow's prompt, so it needs no parameter here.
        """
        authenticate(
            project_id,
            self.provider,
            mounts_dir=mounts_dir,
            image=image,
            expose_token=expose_token,
            oauth_enabled=oauth_enabled,
            credential_set=credential_set,
        )

    def prepare_oauth(
        self,
        project_id: str | None,
        *,
        mounts_dir: Path,
        image: str,
        expose_token: bool = False,
        credential_set: str = "default",
        device_auth: bool = False,
    ) -> AuthSession:
        """Build an [`AuthSession`][terok_executor.AuthSession] without running it.

        Frontends that own their own UI loop (e.g. the terok Textual TUI,
        which wants to dispatch the OAuth container into a new terminal
        tab or via tmux instead of inline) build the session here, run
        ``session.argv`` however they like, then call ``session.capture()``
        on success.  The CLI's blocking ``authenticate`` path is just
        another such caller — see ``_run_auth_container``.

        With *device_auth*, the session runs the provider's headless
        device-code login (see [`prepare_oauth_session`][terok_executor.prepare_oauth_session]).
        """
        info = AUTH_PROVIDERS.get(self.provider)
        if not info:
            available = ", ".join(AUTH_PROVIDERS)
            raise SystemExit(f"Unknown auth provider: {self.provider}. Available: {available}")
        if not info.supports_oauth:
            raise SystemExit(
                f"Provider {self.provider!r} does not support OAuth — use store_api_key() instead."
            )
        if device_auth and not info.supports_device_auth:
            raise SystemExit(f"Provider {self.provider!r} has no device-auth login.")
        return prepare_oauth_session(
            info,
            project_id,
            mounts_dir=mounts_dir,
            image=image,
            expose_token=expose_token,
            credential_set=credential_set,
            device_auth=device_auth,
        )


def authenticate(
    project_id: str | None,
    provider: str,
    *,
    mounts_dir: Path,
    image: str | Callable[[], str] | None = None,
    expose_token: bool = False,
    oauth_enabled: bool = True,
    credential_set: str = "default",
) -> None:
    """Run the auth flow for *provider*, optionally scoped to a project.

    Offers every auth method the provider supports — OAuth (vendor CLI in a
    container), the OAuth device-code variant (for providers that declare
    ``device_auth``), and a direct API key — as the *effective* mode set: what
    the roster declares (``modes:``, ``device_auth``) intersected with what the
    caller permits via *oauth_enabled*.  One method runs straight through; two or
    more prompt the user to choose.

    Args:
        project_id: Project identifier used for container naming and the
            banner line.  Pass ``None`` for host-wide auth — the banner
            drops the project reference and the container gets a neutral
            ``host-auth-<provider>`` name.
        provider: Auth provider name (e.g. ``"claude"``).
        mounts_dir: Base directory for shared config bind-mounts.
        image: Container image for the OAuth container.  Either a tag
            string (eager) or a zero-arg callable returning the tag
            (lazy — invoked only when the user actually chooses the
            OAuth path).  ``None`` is fine for API-key-only providers,
            where no container is launched.  Use the lazy form to avoid
            paying the L1 build cost when the user might pick API key
            from the OAuth-or-API-key prompt.
        expose_token: When True, copy the real credential files into
            the shared mount instead of writing a phantom marker.  Used
            by tier 3 (``expose_oauth_token``) where containers need
            the actual token.
        oauth_enabled: External gate for the OAuth path.  ``True``
            (default) means the roster's ``modes`` list is honored
            verbatim.  ``False`` instructs the function to skip the
            OAuth prompt and go straight to the API-key flow regardless
            of what the roster declares — terok passes ``False`` for
            providers whose OAuth path requires unset config flags
            (e.g. ``agent.codex.allow_oauth=true`` plus ``experimental:
            true``).  When the provider declares only OAuth and the
            gate is closed, raises ``SystemExit`` with a clear hint.
        credential_set: Storage namespace in the vault DB.  Defaults to
            ``"default"`` — the shared host-wide bucket every standalone
            and pre-existing terok caller uses.  Per-project callers
            pass a project-specific value (e.g. ``project.id``) to
            keep each project's tokens isolated.  The DB schema keys on
            ``(credential_set, provider)``, so two projects can hold
            independent logins for the same provider side-by-side.

    Raises ``SystemExit`` if the provider name is unknown or no usable
    auth mode remains after gating.
    """
    info = AUTH_PROVIDERS.get(provider)
    if not info:
        available = ", ".join(AUTH_PROVIDERS)
        raise SystemExit(f"Unknown auth provider: {provider}. Available: {available}")

    # Gating: a provider's roster may declare OAuth, but the deployment
    # may not allow it (terok's ``allow_oauth`` + ``experimental`` gate).
    has_oauth = info.supports_oauth and oauth_enabled
    has_api_key = info.supports_api_key

    # OAuth (and its device-code variant) run the vendor CLI in a container; the
    # image is resolved lazily so the OAuth-or-API-key prompt never pays the L1
    # build cost when the user picks the container-free API-key path.
    def _oauth(*, device: bool) -> None:
        _run_auth_container(
            project_id,
            info,
            mounts_dir=mounts_dir,
            image=_resolve_image(image, provider),
            expose_token=expose_token,
            credential_set=credential_set,
            device_auth=device,
        )

    def _api_key() -> None:
        store_api_key(provider, _prompt_api_key(info), credential_set=credential_set)

    methods: list[tuple[str, Callable[[], None]]] = []
    if has_oauth:
        methods.append(("OAuth login", lambda: _oauth(device=False)))
        if info.supports_device_auth:
            # Same OAuth credential, obtained via a device code the operator
            # enters on a second device instead of a forwarded callback port.
            methods.append(("OAuth login — device code", lambda: _oauth(device=True)))
    if has_api_key:
        methods.append(("API key", _api_key))

    if not methods:
        # Provider declares only OAuth and the caller's gate is closed.
        raise SystemExit(
            f"Auth for {provider!r} requires OAuth, but it is disabled by "
            f"the caller's gating policy.  For terok this typically means "
            f"the experimental flag and/or the provider-specific "
            f"allow_oauth/expose_oauth_token config keys are unset."
        )

    if len(methods) == 1:
        methods[0][1]()
        return

    print(f"Authenticate {info.label}:\n")
    for i, (label, _) in enumerate(methods, 1):
        print(f"  {i}. {label}")
    print()
    choice = input(f"Choose [1-{len(methods)}] (default 1): ").strip() or "1"
    try:
        index = int(choice) - 1
        if not 0 <= index < len(methods):
            raise ValueError
    except ValueError:
        raise SystemExit(f"Invalid choice: {choice!r}") from None
    methods[index][1]()


def _resolve_image(image: str | Callable[[], str] | None, provider: str) -> str:
    """Coerce *image* — eager string, lazy callable, or ``None`` — to a tag.

    ``None`` means the caller didn't supply one and the OAuth path is
    actually about to launch a container; that is a programming error,
    not a user-recoverable one — raise.
    """
    if image is None:
        raise ValueError(
            f"OAuth auth for {provider!r} needs an L1 image; "
            "pass image=<tag> or image=<callable returning tag>."
        )
    return image() if callable(image) else image


def credential_provider(name: str) -> str:
    """Resolve an auth target (agent name) to the provider its credential is keyed under.

    ``terok-executor auth claude`` authenticates Claude's *default provider*, so
    the credential is stored under ``anthropic`` — the same key the vault route
    and the v3-migrated DB use, so ``routed = stored & routes`` intersects at
    runtime.  Tools resolve the same way (``gh`` → ``github``).  Names not in the
    auth registry (harnesses with no credentials, or already a provider) pass
    through unchanged.

    Reads the auth registry rather than the roster so the credentials layer
    stays below it — the resolved provider rides on each
    [`AuthProvider`][terok_executor.credentials.auth.AuthProvider], populated by
    the loader.
    """
    info = AUTH_PROVIDERS.get(name)
    return info.credential_provider if (info and info.credential_provider) else name


def store_api_key(
    provider: str,
    api_key: str,
    credential_set: str = "default",
) -> None:
    """Store an API key directly in the credential DB (no container needed).

    This is the non-interactive fast path for automated workflows and CI.
    The key is stored as ``{"type": "api_key", "key": "<value>"}`` under the
    resolved provider name (see [`credential_provider`][terok_executor.credentials.auth.credential_provider]).
    """
    from terok_executor.integrations.sandbox import SandboxConfig

    cred_provider = credential_provider(provider)
    cfg = SandboxConfig()
    db = cfg.open_credential_db(prompt_on_tty=True)
    try:
        db.store_credential(credential_set, cred_provider, {"type": "api_key", "key": api_key})
        print(f"API key stored for {cred_provider} (set: {credential_set})")
    finally:
        db.close()


# ── Private helpers ──


def _prompt_api_key(info: AuthProvider) -> str:
    """Interactively prompt for an API key (echoes ``*`` per character).

    Uses ``prompt_toolkit.prompt(is_password=True)`` for the TTY path —
    proper terminal raw-mode handling, ``Ctrl+C`` raises
    ``KeyboardInterrupt`` cleanly, and every character is reliably
    masked (the previous ``pwinput`` implementation occasionally let a
    keystroke echo through and swallowed ``SIGINT``).  Non-TTY input
    (``terok auth … < keyfile.txt``) falls back to a plain
    ``readline`` so pipe-fed automation still works.
    """
    import sys

    if info.api_key_hint:
        print(info.api_key_hint)
    prompt_text = f"{info.label} API key: "
    if sys.stdin.isatty():
        from prompt_toolkit import prompt as ptk_prompt

        try:
            key = ptk_prompt(prompt_text, is_password=True).strip()
        except (KeyboardInterrupt, EOFError):
            raise SystemExit("API key entry cancelled.") from None
    else:
        print(prompt_text, end="", flush=True)
        key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("No API key entered.")
    return key


@dataclass
class AuthSession:
    """A prepared-but-not-run OAuth auth container session.

    Built by [`Authenticator.prepare_oauth`][terok_executor.Authenticator.prepare_oauth]
    (or the module-level [`prepare_oauth_session`][terok_executor.prepare_oauth_session]
    helper).  Hold-don't-call: the caller is responsible for running
    ``argv`` (synchronously, in a new terminal tab, suspended TUI, etc.)
    and calling ``capture()`` afterwards.  Use as a context manager so
    the temp dir and any dangling container are cleaned up on exit.
    """

    provider: AuthProvider
    """Provider descriptor (label, banner hint, mount points)."""

    project_id: str | None
    """Project scope for the banner; ``None`` for host-wide auth."""

    container_name: str
    """Podman container name (used for cleanup and ``-it`` log clarity)."""

    argv: list[str]
    """The ``podman run …`` command line — run this however you like."""

    banner: str
    """Banner text to display before launching ``argv``."""

    auth_dir: Path
    """Temp dir bind-mounted as the container's auth config target.

    Lives until ``cleanup()`` (or ``__exit__``).  Credential extraction
    in ``capture()`` reads from here, so don't remove it manually.
    """

    mounts_dir: Path
    """Base directory for the shared post-capture mount (OAuth providers only)."""

    credential_set: str = "default"
    """Which credential set in the vault DB receives the captured token."""

    expose_token: bool = False
    """When True, real credential files are copied into the shared mount (tier 3)."""

    _tmpdir: tempfile.TemporaryDirectory[str] | None = field(
        default=None, repr=False, compare=False
    )
    """Internal: backing temp-dir handle, released by ``cleanup``."""

    @property
    def title(self) -> str:
        """Short human-readable title (``"Authenticating Claude (host-wide)"``)."""
        scope = f"for project: {self.project_id}" if self.project_id else "(host-wide)"
        return f"Authenticating {self.provider.label} {scope}"

    def capture(self) -> None:
        """Extract credentials from ``auth_dir``, store them in the vault DB.

        Call after ``argv`` exits successfully.  Safe to call multiple
        times (the underlying extractor is idempotent on a stable
        credential file).
        """
        _capture_credentials(
            self.provider.name,
            self.auth_dir,
            self.credential_set,
            mounts_base=self.mounts_dir,
            auth_provider=self.provider,
            expose_token=self.expose_token,
        )

    def cleanup(self) -> None:
        """Release the temp dir and force-remove any lingering container.

        Idempotent.  ``__exit__`` calls this automatically.
        """
        subprocess.run(
            ["podman", "rm", "-f", self.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self) -> AuthSession:
        """Return self; the heavy lifting already happened in the factory."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Run ``cleanup`` on context-manager exit."""
        self.cleanup()


def prepare_oauth_session(
    provider: AuthProvider,
    project_id: str | None,
    *,
    mounts_dir: Path,
    image: str,
    expose_token: bool = False,
    credential_set: str = "default",
    device_auth: bool = False,
) -> AuthSession:
    """Build an [`AuthSession`][terok_executor.AuthSession] without running it.

    Creates a fresh temp dir, computes the ``podman run`` argv, and
    cleans up any leftover container of the same name (so re-auth
    after a previous abort isn't blocked).  The caller drives execution
    and credential capture; see [`AuthSession`][terok_executor.AuthSession].

    The temp dir uses a clean slate so the vendor auth flow re-runs end
    to end — no stale config, no cached sessions.

    With *device_auth*, the command gains ``--device-auth`` and the
    port-forwarding ``extra_run_args`` are dropped: the device-code flow
    polls a remote endpoint, so there is no localhost callback to forward.
    """
    _check_podman()

    tmpdir = tempfile.TemporaryDirectory(prefix=f"terok-auth-{provider.name}-")
    host_dir = Path(tmpdir.name)

    # ``project_id`` must lead the container name; Podman rejects names
    # starting with ``_`` or other non-alphanumeric chars, so the
    # host-wide caller passes ``None`` and we fall back to ``host``.
    name_prefix = project_id or "host"
    container_name = f"{name_prefix}-auth-{provider.name}"
    _cleanup_existing_container(container_name)

    command = [*provider.command, "--device-auth"] if device_auth else list(provider.command)
    run_args = () if device_auth else provider.extra_run_args

    cmd = ["podman", "run", "--rm", *podman_userns_args(), "-it"]
    if run_args:
        cmd.extend(run_args)
    cmd.extend(["-v", f"{host_dir}:{provider.container_mount}:Z"])
    cmd.extend(["--name", container_name])
    cmd.append(image)
    cmd.extend(command)

    scope = f"for project: {project_id}" if project_id else "(host-wide)"
    banner_lines = [
        f"Authenticating {provider.label} {scope}",
        "",
        *provider.banner_hint.splitlines(),
        "",
        f"$ {' '.join(map(str, cmd))}",
        "",
    ]

    return AuthSession(
        provider=provider,
        project_id=project_id,
        container_name=container_name,
        argv=cmd,
        banner="\n".join(banner_lines),
        auth_dir=host_dir,
        mounts_dir=mounts_dir,
        credential_set=credential_set,
        expose_token=expose_token,
        _tmpdir=tmpdir,
    )


def _run_auth_container(
    project_id: str | None,
    provider: AuthProvider,
    *,
    mounts_dir: Path,
    image: str,
    credential_set: str = "default",
    expose_token: bool = False,
    device_auth: bool = False,
) -> None:
    """Synchronous CLI helper: prepare a session, run it inline, capture.

    Thin wrapper around [`prepare_oauth_session`][terok_executor.prepare_oauth_session]
    that preserves the original ``terok auth`` behaviour — print banner,
    run ``podman`` in the foreground, capture on success, swallow 130
    (Ctrl-C inside the container) without surfacing as failure.
    """
    with prepare_oauth_session(
        provider,
        project_id,
        mounts_dir=mounts_dir,
        image=image,
        expose_token=expose_token,
        credential_set=credential_set,
        device_auth=device_auth,
    ) as session:
        print(session.banner)
        try:
            # argv is built by prepare_oauth_session from the bundled roster
            # command + a resolved image tag, no shell, no untrusted input;
            # the foreground run is intentional for the CLI.
            subprocess.run(session.argv, check=True)  # nosec B603
        except subprocess.CalledProcessError as e:
            if e.returncode == 130:
                print("\nAuthentication container stopped.")
                return  # user cancelled — don't capture stale pre-seeded files
            raise SystemExit(f"Auth failed: {e}")
        except KeyboardInterrupt:
            print("\nAuthentication interrupted.")
            return  # session.cleanup() runs via __exit__ and removes the container

        session.capture()


def _check_podman() -> None:
    """Verify podman is available."""
    if shutil.which("podman") is None:
        raise SystemExit("podman not found; please install podman")


def _cleanup_existing_container(container_name: str) -> None:
    """Remove an existing container if it exists."""
    result = subprocess.run(
        ["podman", "container", "exists", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        print(f"Removing existing auth container: {container_name}")
        subprocess.run(
            ["podman", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _capture_credentials(
    provider_name: str,
    auth_dir: Path,
    credential_set: str,
    mounts_base: Path | None = None,
    auth_provider: AuthProvider | None = None,
    *,
    expose_token: bool = False,
) -> None:
    """Extract credentials from *auth_dir* and store in the credential DB.

    Uses the per-provider extractors from `credential_extractors`.
    If extraction fails (no credential file, malformed), prints a warning
    but does not raise — the auth flow succeeded, the user can retry.

    When *expose_token* is ``True`` for Claude OAuth (tier 3), the credential
    is **not** stored in the vault DB — Claude manages its own token lifecycle
    directly, and vault-side refresh would invalidate the exposed token.
    """
    from .extractors import extract_credential

    try:
        cred_data = extract_credential(provider_name, auth_dir)
    except Exception as exc:
        _err.print(
            f"\n[red]Error [auth]: could not extract credentials for {provider_name} "
            f"from {auth_dir}: {type(exc).__name__}: {exc}[/red]"
        )
        _err.print("[red]The auth flow completed but credentials were not captured.[/red]")
        print(
            "You may need to re-authenticate or check the credential file format.",
            file=sys.stderr,
        )
        # List files in the auth dir to aid debugging
        files = sorted(p.relative_to(auth_dir) for p in auth_dir.rglob("*") if p.is_file())
        if files:
            print(f"\nFiles found in auth dir ({len(files)}):")
            for f in files[:20]:
                print(f"  {f}")
        return

    is_oauth = cred_data.get("type") == "oauth"
    post_capture = _OAUTH_MOUNT_WRITERS.get(provider_name) if is_oauth else None
    # Tier 3 bypass: the host-side vault never refreshes the token, so the
    # container must own its lifecycle.  Storing in the DB would let the
    # background refresher rotate a token nobody brokers back to the mount.
    exposed_directly = expose_token and post_capture is not None

    if exposed_directly:
        _out.print(
            f"\n[green]Credentials for {provider_name} "
            "bypassing vault DB (exposed directly)[/green]"
        )
    else:
        try:
            from terok_executor.integrations.sandbox import SandboxConfig

            cfg = SandboxConfig()
            db = cfg.open_credential_db(prompt_on_tty=True)
            try:
                # Store under the resolved provider name (claude → anthropic);
                # the mount writer above stays keyed by the agent's auth dir.
                db.store_credential(credential_set, credential_provider(provider_name), cred_data)
                _out.print(
                    f"\n[green]Credentials captured for {provider_name} "
                    f"(set: {credential_set})[/green]"
                )
            finally:
                db.close()
        except Exception as exc:
            _err.print(
                f"\n[red]Error [auth]: failed to store credentials for {provider_name} "
                f"in vault DB: {type(exc).__name__}: {exc}[/red]"
            )
            _err.print(
                "[red]The auth flow completed but credentials were not saved to the vault DB.[/red]"
            )
            return

    # Reconcile the shared mount with the captured credential.  Provider-specific
    # writers drop a phantom marker (proxied mode) or copy the real file (exposed).
    if post_capture is not None:
        if mounts_base is None:
            from terok_executor.paths import mounts_dir

            mounts_base = mounts_dir()
        try:
            post_capture(auth_dir, mounts_base, cred_data, expose_token)
        except Exception as exc:  # noqa: BLE001
            _err.print(
                f"[yellow]Warning: could not reconcile {provider_name} mount: {exc}[/yellow]"
            )

    # Apply declarative post-capture state from roster YAML
    if auth_provider and auth_provider.post_capture_state:
        try:
            _apply_post_capture_state(
                auth_provider.host_dir_name,
                auth_provider.post_capture_state,
                mounts_base,
            )
        except Exception as exc:  # noqa: BLE001
            _err.print(
                f"[yellow]Warning: could not apply post_capture_state for "
                f"{provider_name}: {exc}[/yellow]"
            )


def _claude_oauth_mount_writer(
    auth_dir: Path, mounts_base: Path, cred_data: dict, expose_token: bool
) -> None:
    """Reconcile Claude's shared mount after an OAuth capture.

    Default path writes a phantom ``.credentials.json`` with subscription
    metadata only; tier 3 copies the real credential file so Claude Code
    can reach ``api.anthropic.com`` directly (its hardcoded OAuth host).
    """
    if expose_token:
        src = auth_dir / ".credentials.json"
        if not src.is_file():
            raise FileNotFoundError(f"No .credentials.json in {auth_dir}")
        dest_dir = mounts_base / "_claude-config"
        dest_dir.mkdir(parents=True, exist_ok=True)
        _write_bytes_nofollow(dest_dir / ".credentials.json", src.read_bytes())
        print("Real .credentials.json copied to shared Claude config mount.")
        print(
            "\nNote: Claude OAuth token is EXPOSED in the shared mount."
            "\n      Every task container can read the real token."
            "\n      The vault does NOT protect it."
        )
    else:
        _write_claude_credentials_file(cred_data, mounts_base)
        print("Subscription metadata written to shared Claude config mount.")
        print(
            "\nNote: Claude OAuth credential is shared across all task containers."
            "\n      API calls are routed through the vault — the real"
            "\n      token stays on the host."
        )


def _codex_oauth_mount_writer(
    auth_dir: Path, mounts_base: Path, cred_data: dict, expose_token: bool
) -> None:
    """Reconcile Codex's shared mount after an OAuth capture.

    Two modes:

    - **Exposed** (``expose_token=True``): copy the real ``auth.json``
      verbatim so the in-container Codex reads the live OAuth token
      (tier 3, unsafe — every task container can read it).
    - **Default**: drop a shared synthetic ``auth.json`` — the real
      ``id_token`` JWT (for plan-tier + workspace UI, public claims only)
      and ``account_id`` survive, but ``access_token`` and ``refresh_token``
      are replaced with `CODEX_SHARED_OAUTH_MARKER`.  The vault translates
      the marker back to the real token on inference requests; the CLI
      itself never sees the live bearer.  This is the fallback for tier 2
      (proxied) and also a no-harm default for tier 1 (vault stores creds
      but nothing wires the proxy yet).
    """
    dest_dir = mounts_base / "_codex-config"
    dest_file = dest_dir / "auth.json"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if expose_token:
        src = auth_dir / "auth.json"
        if not src.is_file():
            raise FileNotFoundError(f"No auth.json in {auth_dir}")
        _write_bytes_nofollow(dest_file, src.read_bytes())
        _out.print("[green]Real auth.json copied to shared Codex config mount.[/green]")
        _out.print(
            "[yellow]\nNote: Codex OAuth token is EXPOSED in the shared mount."
            "\n      Every task container can read the real token."
            "\n      The vault does NOT protect it.[/yellow]"
        )
    else:
        _write_codex_phantom_auth_json(cred_data, dest_file)
        _out.print("[green]Phantom auth.json written to shared Codex config mount.[/green]")
        print(
            "\nNote: Codex OAuth credential is shared across all task containers."
            "\n      API calls are routed through the vault — the real"
            "\n      OAuth tokens stay on the host.  Standalone executor"
            "\n      uses the shared Codex config rewrite directly;"
            "\n      terok project mode gates that rewrite via"
            "\n      agent.codex.allow_oauth."
        )


#: Static far-future timestamp for the phantom ``auth.json``'s
#: ``last_refresh`` field.  Codex's CLI fires a client-side token
#: refresh against the hardcoded ``auth.openai.com`` whenever
#: ``now - last_refresh > 8 days`` (manager.rs:1743).  Those attempts
#: would arrive at the upstream with the phantom refresh token and come
#: back as ``refresh_token_invalidated``, prompting the user to re-login
#: even though vault-side refresh keeps the real credential fresh.
#: Pinning ``last_refresh`` to year 9999 keeps the check permanently
#: satisfied — the same trick Claude's phantom file uses with its
#: ``"expiresAt": null`` sentinel.
_CODEX_PHANTOM_LAST_REFRESH = "9999-01-01T00:00:00Z"


def _write_codex_phantom_auth_json(cred_data: dict, dest: Path) -> None:
    """Write a shared synthetic ``auth.json`` without real bearer tokens.

    Codex's ``TokenData`` serde contract (codex-rs/login/src/token_data.rs)
    requires ``id_token`` to be a parseable JWT string.  We synthesize a
    minimal JWT that preserves non-PII account metadata while dropping the
    original opaque token body entirely.

    The access/refresh tokens are the actual bearer secrets; both are
    replaced with the Codex-specific shared vault marker.  Inference
    rides through the vault, which substitutes the live access token.
    """
    import json

    tokens: dict = {
        "id_token": _build_codex_shared_id_token(cred_data.get("id_token", "")),
        "access_token": CODEX_SHARED_OAUTH_MARKER,
        "refresh_token": CODEX_SHARED_OAUTH_MARKER,
    }
    account_id = cred_data.get("account_id")
    if account_id:
        tokens["account_id"] = account_id

    payload = {
        "OPENAI_API_KEY": None,
        "tokens": tokens,
        "last_refresh": _CODEX_PHANTOM_LAST_REFRESH,
    }
    _write_bytes_nofollow(dest, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _build_codex_shared_id_token(raw_jwt: str) -> str:
    """Return a synthetic JWT with only the Codex-local claims we need."""
    import base64
    import binascii
    import json

    def _b64url(data: dict) -> str:
        encoded = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")

    def _jwt_payload(token: str) -> dict:
        try:
            _header, payload, _sig = token.split(".", 2)
            padded = payload + "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
            parsed = json.loads(decoded)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, binascii.Error):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    payload = _jwt_payload(raw_jwt)
    auth = payload.get("https://api.openai.com/auth")

    safe_payload: dict[str, object] = {"exp": 253402300799}
    # Codex 0.123+ TUI bootstrap calls ``account/read`` (an internal
    # JSON-RPC), which fails unless both ``email`` AND ``chatgpt_plan_type``
    # parse out of the id_token JWT (codex-rs/login/src/token_data.rs:139).
    # ``chatgpt_plan_type`` already rides through under
    # ``https://api.openai.com/auth`` below; here we additionally preserve
    # the top-level ``email`` claim so the TUI's "logged in as ..." surface
    # populates and bootstrap doesn't error out.
    if isinstance(payload.get("email"), str):
        safe_payload["email"] = payload["email"]
    if isinstance(auth, dict):
        safe_auth = {
            key: auth[key]
            for key in (
                "chatgpt_plan_type",
                "chatgpt_user_id",
                "user_id",
                "chatgpt_account_id",
                "chatgpt_account_is_fedramp",
            )
            if key in auth
        }
        if safe_auth:
            safe_payload["https://api.openai.com/auth"] = safe_auth

    return ".".join((_b64url({"alg": "none", "typ": "JWT"}), _b64url(safe_payload), "terok"))


def _write_bytes_nofollow(path: Path, data: bytes) -> None:
    """Atomically write *data* without following a pre-existing target symlink."""
    import os
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(tmp, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


#: Maps provider name → post-capture mount reconciler.  OAuth-capable
#: providers register here to share the expose/proxy dispatch in
#: `_capture_credentials`.
_OAUTH_MOUNT_WRITERS: dict[str, Callable[[Path, Path, dict, bool], None]] = {
    "claude": _claude_oauth_mount_writer,
    "codex": _codex_oauth_mount_writer,
}


def _write_claude_credentials_file(cred_data: dict, mounts_base: Path) -> None:
    """Write a static ``.credentials.json`` with subscription metadata.

    The file lets Claude Code determine the subscription tier locally
    (``subscriptionType``, ``scopes``, ``rateLimitTier``) without
    exposing the real OAuth token.  ``accessToken`` is set to a dummy
    marker — actual API auth uses the per-task phantom token from the
    ``CLAUDE_CODE_OAUTH_TOKEN`` env var.

    Onboarding state (``.claude.json`` / ``hasCompletedOnboarding``) is
    applied separately via ``_apply_post_capture_state`` after capture.
    """
    import json

    claude_dir = mounts_base / "_claude-config"
    claude_dir.mkdir(parents=True, exist_ok=True)

    creds = {
        "claudeAiOauth": {
            "accessToken": PHANTOM_CREDENTIALS_MARKER,
            "refreshToken": "",
            "expiresAt": None,
            "scopes": cred_data.get("scopes", ""),
            "subscriptionType": cred_data.get("subscription_type"),
            "rateLimitTier": cred_data.get("rate_limit_tier"),
        }
    }
    _write_bytes_nofollow(
        claude_dir / ".credentials.json",
        (json.dumps(creds, indent=2) + "\n").encode("utf-8"),
    )


def _apply_post_capture_state(
    host_dir_name: str,
    patches: dict[str, dict],
    mounts_base: Path | None,
) -> None:
    """Apply ``post_capture_state`` after credential capture.

    Merges key-value pairs into JSON files in the provider's auth mount
    directory.  Declared in ``auth.post_capture_state`` in the agent YAML.
    Takes resolved data directly — no roster lookup (avoids circular dep).
    """
    import json

    if mounts_base is None:
        from terok_executor.paths import mounts_dir

        mounts_base = mounts_dir()

    mounts_root = mounts_base.resolve()
    host_rel = Path(host_dir_name)
    if host_rel.is_absolute() or ".." in host_rel.parts:
        raise ValueError(f"Invalid host_dir_name: {host_dir_name!r}")
    target_dir = (mounts_root / host_rel).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, patch in patches.items():
        rel = Path(filename)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Invalid post_capture_state filename: {filename!r}")
        path = (target_dir / rel).resolve()
        state: dict = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                state = loaded if isinstance(loaded, dict) else {}
            except (json.JSONDecodeError, OSError):
                state = {}

        if all(state.get(k) == v for k, v in patch.items()):
            continue  # already up to date

        state.update(patch)
        _write_bytes_nofollow(path, (json.dumps(state, indent=2) + "\n").encode("utf-8"))


def api_key_command(cfg: AuthKeyConfig) -> list[str]:
    """Build a bash command that prompts for an API key and writes it to a config file."""
    config_dir = cfg.config_path.rsplit("/", 1)[0]
    parts = [
        f"echo 'Enter your {cfg.label} API key (get one at {cfg.key_url}):'",
        f"read -r -p '{cfg.env_var}=' api_key",
        f"mkdir -p {config_dir}",
        f"printf '{cfg.printf_template}\\n' \"$api_key\" > {cfg.config_path}",
        "echo",
        f"echo 'API key saved to {cfg.config_path}'",
        f"echo 'You can now use {cfg.tool_name} in task containers.'",
    ]
    return ["bash", "-c", " && ".join(parts)]
