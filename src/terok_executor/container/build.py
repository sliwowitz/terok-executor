# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Builds L0 (base dev) and L1 (agent CLI) container images via podman.

Owns the L0 (base dev) and L1 (agent CLI) Dockerfile templates, resource
staging, image naming, and ``podman build`` invocation.

**Image layer architecture**::

    L0  (base)   — Distro + dev tools + init script + dev user
    L1  (agent)  — All AI agent CLIs, shell environment, ACP wrappers
                   L1 is self-sufficient for standalone use — all user
                   config (repo URL, SSH, branch, gate) is runtime.
    ─── boundary: above owned by terok-executor, below by terok ───
    L2  (project)— Optional: user Dockerfile snippet (custom packages)
                   Only built when project has docker snippet config.

``terok-executor run claude .`` launches directly on the L1 image — no L2
build needed.  terok adds L2 only for project-specific image customisation.

Usage as a library::

    from terok_executor import build_base_images

    images = build_base_images("fedora:44")
    # images.l0 = "terok-l0:fedora-44"
    # images.l1 = "terok-l1-cli:fedora-44"

The L0/L1 templates select between Debian/Ubuntu (``apt``) and Fedora-like
(``dnf``) package managers via a ``family`` Jinja2 variable resolved by
[`detect_family`][terok_executor.container.build.detect_family] from the base image name (or an explicit override).

L1 is roster-driven: each agent's install steps live in its YAML file
(``install.run_as_root`` / ``install.run_as_dev``), and the L1 template
loops over the resolved selection.  Build emits an OCI label
``ai.terok.agents=<csv>``, an in-container manifest
``/etc/terok/installed.env``, and pre-rendered ``hilfe`` help fragments —
all derived from the same selection.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment

# ── Vocabulary ──

DEFAULT_BASE_IMAGE = "fedora:44"
"""Default base OS image when none is specified."""

AGENTS_LABEL = "ai.terok.agents"
"""OCI label naming the roster entries baked into an L1 image."""

INSTALLED_ENV_PATH = "/etc/terok/installed.env"
"""In-container env file that scripts source to learn what's installed."""

_HELP_SECTION_FILES: dict[str, str] = {"agent": "agents.txt", "dev_tool": "dev-tools.txt"}
"""Maps each [`HelpSection`][terok_executor.roster.loader.HelpSection] to its fragment filename."""

_ESCAPE_RE = re.compile(r"\\(?:[0-7]{1,3}|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|[nrtbfav\\'\"])")
"""Backslash escapes recognised in roster ``help.label`` strings.

Matches octal (``\\033``), hex (``\\x1b``), 4-digit Unicode (``\\u1234``),
and the standard one-letter forms — nothing wider, so the surrounding
text (which may contain non-ASCII characters) is preserved verbatim.
"""


def _decode_label_escapes(text: str) -> str:
    """Expand backslash escapes in *text* without disturbing non-ASCII content.

    The straight ``bytes(text, "utf-8").decode("unicode_escape")`` round-trip
    re-interprets every UTF-8 byte as Latin-1 and mojibakes anything outside
    ASCII (e.g. ``"ä"`` becomes ``"Ã¤"``).  We only want to expand explicit
    backslash escapes, so we substitute one match at a time — each match
    is pure ASCII, making the per-match round-trip safe.
    """
    return _ESCAPE_RE.sub(lambda m: m.group().encode("ascii").decode("unicode_escape"), text)


_DEFAULT_TAG = "fedora-44"
"""Pre-sanitized tag fragment for the default base image.

Must stay in sync with the slug of
[`DEFAULT_BASE_IMAGE`][terok_executor.container.build.DEFAULT_BASE_IMAGE]
— i.e. the result of feeding ``DEFAULT_BASE_IMAGE`` through the same
sanitiser ``_base_tag`` applies.  This fallback is reached only when
the input collapses to an empty string after sanitisation (pathological
inputs such as ``_base_tag("???")``); normal empty/whitespace input
short-circuits earlier via ``_normalize_base_image``.
"""

_MAX_TAG_LEN = 120
"""Cap on the tag portion of an OCI image reference.

OCI spec allows 128; the extra headroom absorbs future suffix changes
without rebuilding history.  Both `_base_tag` (for the L0 tag)
and [`l1_image_tag`][terok_executor.container.build.l1_image_tag] (for the L1 base+agents tag) respect this.
"""

_AGENT_DIGEST_LEN = 12
"""SHA1-prefix length for the fallback agent-suffix digest."""

# Map of known base-image prefixes to their package family.  Each entry
# is either a literal ``"deb"``/``"rpm"`` or a tag-aware resolver — used
# for NVIDIA, where the same repo path ships both Ubuntu (apt) and UBI
# (dnf) variants and only the tag distinguishes them.
# "Officially tested" (per AGENTS.md): fedora:44, ubuntu:24.04,
# quay.io/podman/stable, nvcr.io/nvidia/nvhpc.  Other images in the
# same family path will match but are unsupported.
_NVIDIA_UBI_TAG_RE: re.Pattern[str] = re.compile(r"ubi\d+", re.IGNORECASE)


def _nvidia_family(tag: str) -> str:
    """Pick the family for a matched NVIDIA image from its *tag*.

    NVIDIA tags carry an explicit ``ubuntu`` or ``ubi[N]`` marker; absence
    of either is treated as the historical default of Ubuntu (``deb``).
    """
    return "rpm" if _NVIDIA_UBI_TAG_RE.search(tag) else "deb"


_KNOWN_FAMILIES: tuple[tuple[str, str | Callable[[str], str]], ...] = (
    ("registry.fedoraproject.org/fedora", "rpm"),
    ("quay.io/podman", "rpm"),
    ("nvcr.io/nvidia", _nvidia_family),
    ("nvidia", _nvidia_family),
    ("ubuntu", "deb"),
    ("debian", "deb"),
    ("fedora", "rpm"),
)


class BuildError(RuntimeError):
    """Raised when base-image construction cannot complete.

    The CLI maps this to a user-facing error message; library callers
    can catch it without being terminated by ``SystemExit``.
    """


@dataclass(frozen=True)
class ImageSet:
    """L0 + L1 image tags produced by a build."""

    l0: str
    """L0 base dev image tag (e.g. ``terok-l0:fedora-44``)."""

    l1: str
    """L1 agent CLI image tag (e.g. ``terok-l1-cli:fedora-44``)."""

    l1_sidecar: str | None = None
    """L1 sidecar image tag, if built (e.g. ``terok-l1-sidecar:fedora-44``)."""


# ── Public entry points ──


def detect_family(base_image: str, override: str | None = None) -> str:
    """Resolve the package family (``deb`` or ``rpm``) for *base_image*.

    *override* — when set, must be ``"deb"`` or ``"rpm"`` and wins over
    detection (used to support unknown bases via project config).

    Detection matches a small allowlist of known image prefixes
    (Ubuntu/Debian, Fedora, the official Podman container, NVIDIA CUDA/HPC
    SDK).  NVIDIA images are inspected at the tag level so UBI variants
    (e.g. ``…:13.0.0-devel-ubi9``) resolve to ``rpm`` while Ubuntu
    variants resolve to ``deb``.  Unknown images raise [`BuildError`][terok_executor.container.build.BuildError]
    with a hint to set ``family:`` explicitly.
    """
    if override is not None:
        if override not in {"deb", "rpm"}:
            raise BuildError(f"family must be 'deb' or 'rpm', got {override!r}")
        return override
    name, tag = _split_image_ref(_normalize_base_image(base_image))
    name_lc = name.lower()
    for prefix, fam in _KNOWN_FAMILIES:
        if name_lc == prefix or name_lc.startswith(prefix + "/"):
            return fam(tag) if callable(fam) else fam
    raise BuildError(
        f"Cannot infer package family for base image {base_image!r}. "
        "Set `family: deb` or `family: rpm` under image: in project.yml."
    )


def build_project_image(
    *,
    dockerfile: Path,
    context_dir: Path,
    target_tag: str,
    extra_tags: tuple[str, ...] = (),
    build_args: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    no_cache: bool = False,
    pull_always: bool = False,
) -> None:
    """Build an OCI image from a pre-rendered Dockerfile.

    The thin ``podman build`` invoker that the three opinionated factories
    in this module ([`build_base_images`][terok_executor.container.build.build_base_images], [`build_sidecar_image`][terok_executor.container.build.build_sidecar_image],
    and terok's project/L2 build) share.  Callers own Dockerfile
    rendering, tag naming, label computation, and build-context staging —
    this function only assembles flags and shells out.

    Args:
        dockerfile: Path to the pre-rendered Dockerfile (``-f``).
        context_dir: Build context directory (final positional argument).
        target_tag: Primary image tag (``-t``).
        extra_tags: Additional tags applied to the same build (each becomes
            another ``-t`` on the command line — podman builds once and
            tags the result multiple times).
        build_args: ``--build-arg KEY=VALUE`` pairs.
        labels: ``--label KEY=VALUE`` pairs recorded in the OCI config.
        no_cache: Force full rebuild.
        pull_always: Pull the base image even if a local copy exists.

    Raises:
        BuildError: When podman is not on PATH or the build exits non-zero.
    """
    cmd = ["podman", "build", "-f", str(dockerfile)]
    for key, value in (build_args or {}).items():
        cmd += ["--build-arg", f"{key}={value}"]
    for key, value in (labels or {}).items():
        cmd += ["--label", f"{key}={value}"]
    cmd += ["-t", target_tag]
    for tag in extra_tags:
        cmd += ["-t", tag]
    if no_cache:
        cmd.append("--no-cache")
    if pull_always:
        cmd.append("--pull=always")
    cmd.append(str(context_dir))

    print("$", shlex.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise BuildError("podman not found; please install podman") from exc
    except subprocess.CalledProcessError as exc:
        raise BuildError(f"Image build failed: {exc}") from exc


def build_base_images(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    family: str | None = None,
    agents: str | tuple[str, ...] = "all",
    rebuild: bool = False,
    full_rebuild: bool = False,
    build_dir: Path | None = None,
    tag_as_default: bool = False,
) -> ImageSet:
    """Build L0 + L1 container images and return their tags.

    Skips building if images already exist locally (unless *rebuild* or
    *full_rebuild* is set).  Uses a temporary directory for the build
    context by default; pass *build_dir* to use a specific (empty or
    non-existent) directory instead.

    Args:
        base_image: Base OS image (e.g. ``fedora:44``, ``nvidia/cuda:...``).
        family: Override for the package family (``"deb"`` or ``"rpm"``).
            ``None`` means detect from *base_image* via [`detect_family`][terok_executor.container.build.detect_family].
        agents: Roster entries to install, as the literal string ``"all"``
            (every entry) or a tuple of names (transitively expanded by
            ``depends_on``).  Same selection drives the OCI label, the L1
            tag suffix, the in-container manifest, and the help fragments.
        rebuild: Force rebuild with cache bust (refreshes agent installs).
        full_rebuild: Force rebuild with ``--no-cache --pull=always``.
        build_dir: Build context directory (must be empty or absent).
        tag_as_default: When ``True``, additionally tag the L1 with the
            unsuffixed default-alias [`l1_image_tag(base_image)`][terok_executor.container.build.l1_image_tag].
            Set by [`ensure_default_l1`][terok_executor.container.build.ensure_default_l1]
            when this build represents the user's *configured* default
            agent selection.  Project / per-agent / partial builds leave
            it ``False`` so the alias keeps pointing at the user's
            default L1, not at whatever was last built.

    Returns:
        [`ImageSet`][terok_executor.container.build.ImageSet] with the L0 and L1 image tags.

    Raises:
        BuildError: If podman is missing, the family cannot be resolved,
            or a build step fails.
        ValueError: If *build_dir* is a file or a non-empty directory,
            or if *agents* contains unknown roster entries.
    """
    from terok_executor.roster.loader import get_roster

    _validate_build_dir(build_dir)
    _check_podman()

    base_image = _normalize_base_image(base_image)
    selected = get_roster().resolve_selection(agents)

    l0_tag = l0_image_tag(base_image)
    l1_tag = l1_image_tag(base_image, selected)
    l1_alias = l1_image_tag(base_image)
    extra_tags: tuple[str, ...] = (l1_alias,) if tag_as_default else ()

    # Skip if both images exist and no forced rebuild — done before
    # detect_family() so cached images for unknown bases (built earlier
    # with explicit family) can still be reused without supplying it again.
    if not rebuild and not full_rebuild:
        if _image_exists(l0_tag) and _image_exists(l1_tag):
            return ImageSet(l0=l0_tag, l1=l1_tag)

    fam = detect_family(base_image, override=family)

    # Prepare build context in a safe directory
    import tempfile

    own_tmp = build_dir is None
    context = build_dir or Path(tempfile.mkdtemp(prefix="terok-executor-build-"))

    try:
        try:
            prepare_build_context(context)
            stage_help_fragments(context / "help.d", selected)

            # Single timestamp for both render and build-arg consistency
            cache_bust = str(int(time.time()))

            # Render and write Dockerfiles into the build context
            (context / "L0.Dockerfile").write_text(render_l0(base_image, family=fam))
            (context / "L1.cli.Dockerfile").write_text(
                render_l1(l0_tag, family=fam, agents=selected, cache_bust=cache_bust)
            )
        except OSError as exc:
            raise BuildError(
                f"Image build failed preparing base build context for "
                f"{l1_tag!r} at {context}: {exc}"
            ) from exc

        # Build L0 — base dev image (Ubuntu + git + SSH + init script)
        build_project_image(
            dockerfile=context / "L0.Dockerfile",
            context_dir=context,
            target_tag=l0_tag,
            build_args={"BASE_IMAGE": base_image},
            no_cache=full_rebuild,
            pull_always=full_rebuild,
        )

        # The unsuffixed alias is reserved for the user's default L1
        # (whatever they have configured as ``image.agents``).  It must
        # not be retagged on partial / project / per-agent builds —
        # otherwise `terok auth <X>` would silently end up running
        # against an L1 missing X.  Callers that ARE building the
        # user's default selection pass ``tag_as_default=True``.
        build_project_image(
            dockerfile=context / "L1.cli.Dockerfile",
            context_dir=context,
            target_tag=l1_tag,
            extra_tags=extra_tags,
            build_args={"BASE_IMAGE": l0_tag, "AGENT_CACHE_BUST": cache_bust},
            no_cache=full_rebuild,
        )

    finally:
        if own_tmp:
            shutil.rmtree(context, ignore_errors=True)

    return ImageSet(l0=l0_tag, l1=l1_tag)


def build_l0g_image(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    host_pubkey: str,
    family: str | None = None,
    rebuild: bool = False,
    full_rebuild: bool = False,
    build_dir: Path | None = None,
) -> str:
    """Build the L0G (krun guest) image and return its tag.

    Skips when the image already exists locally unless *rebuild* or
    *full_rebuild* is set.  *host_pubkey* is the host-side SSH public
    key baked into ``/etc/ssh/authorized_keys.d/terok``; empty values
    are rejected because a guest that accepts no connections is
    silently useless.

    Returns:
        The L0G image tag (e.g. ``terok-l0g:fedora-44``).

    Raises:
        BuildError: If podman is missing, the family cannot be resolved,
            ``host_pubkey`` is empty, or the build step fails.
        ValueError: If *build_dir* is a file or a non-empty directory.
    """
    host_pubkey = _validate_host_pubkey(host_pubkey)
    _validate_build_dir(build_dir)
    _check_podman()
    base_image = _validate_base_image(_normalize_base_image(base_image))
    tag = l0g_image_tag(base_image)
    if not rebuild and not full_rebuild and _image_exists(tag):
        return tag

    fam = detect_family(base_image, override=family)

    import tempfile

    own_tmp = build_dir is None
    context = build_dir or Path(tempfile.mkdtemp(prefix="terok-executor-l0g-"))
    try:
        try:
            context.mkdir(parents=True, exist_ok=True)
            (context / "L0G.Dockerfile").write_text(
                render_l0g(base_image, family=fam, host_pubkey=host_pubkey)
            )
        except OSError as exc:
            raise BuildError(
                f"Image build failed preparing L0G build context at {context}: {exc}"
            ) from exc

        build_project_image(
            dockerfile=context / "L0G.Dockerfile",
            context_dir=context,
            target_tag=tag,
            build_args={"BASE_IMAGE": base_image, "KRUN_HOST_PUBKEY": host_pubkey},
            no_cache=full_rebuild,
            pull_always=full_rebuild,
        )
    finally:
        if own_tmp:
            shutil.rmtree(context, ignore_errors=True)

    return tag


def build_sidecar_image(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    family: str | None = None,
    tool_name: str = "coderabbit",
    rebuild: bool = False,
    full_rebuild: bool = False,
    build_dir: Path | None = None,
) -> str:
    """Build the L1 sidecar image for a specific tool. Returns the image tag.

    Ensures L0 exists first (builds it if missing), then builds the
    sidecar image FROM L0.  The sidecar contains only the named tool —
    no agent CLIs, no LLMs.

    Args:
        base_image: Base OS image (passed through to L0 build).
        family: Override for the package family (``"deb"`` or ``"rpm"``).
            ``None`` means detect from *base_image* via [`detect_family`][terok_executor.container.build.detect_family].
        tool_name: Tool to install (selects Jinja2 conditional in template).
        rebuild: Force rebuild with cache bust.
        full_rebuild: Force rebuild with ``--no-cache``.
        build_dir: Build context directory (must be empty or absent).

    Returns:
        The sidecar image tag (e.g. ``terok-l1-sidecar:fedora-44``).

    Raises:
        BuildError: If podman is missing, the family cannot be resolved,
            or a build step fails.
        ValueError: If *build_dir* is a file or a non-empty directory.
    """
    _validate_build_dir(build_dir)
    _check_podman()

    base_image = _normalize_base_image(base_image)
    l0_tag = l0_image_tag(base_image)
    sidecar_tag = l1_sidecar_image_tag(base_image)

    # Same fast-path as build_base_images: defer detect_family until we
    # know we actually need to render Dockerfiles, so cached sidecars
    # for unknown bases can be reused without re-supplying ``family``.
    if not rebuild and not full_rebuild and _image_exists(sidecar_tag) and _image_exists(l0_tag):
        return sidecar_tag

    fam = detect_family(base_image, override=family)

    # Ensure L0 exists (build if needed)
    if not _image_exists(l0_tag) or full_rebuild:
        build_base_images(base_image, family=fam, rebuild=rebuild, full_rebuild=full_rebuild)

    import tempfile

    own_tmp = build_dir is None
    context = build_dir or Path(tempfile.mkdtemp(prefix="terok-executor-sidecar-"))

    try:
        try:
            prepare_build_context(context)
            cache_bust = str(int(time.time()))

            (context / "L1.sidecar.Dockerfile").write_text(
                render_l1_sidecar(l0_tag, family=fam, tool_name=tool_name, cache_bust=cache_bust)
            )
        except OSError as exc:
            raise BuildError(
                f"Image build failed preparing sidecar build context for "
                f"{sidecar_tag!r} at {context}: {exc}"
            ) from exc

        build_project_image(
            dockerfile=context / "L1.sidecar.Dockerfile",
            context_dir=context,
            target_tag=sidecar_tag,
            build_args={"BASE_IMAGE": l0_tag, "TOOL_CACHE_BUST": cache_bust},
            no_cache=full_rebuild,
        )
    finally:
        if own_tmp:
            shutil.rmtree(context, ignore_errors=True)

    return sidecar_tag


# ── Build context ──


def prepare_build_context(dest: Path) -> None:
    """Stage auxiliary resources into a build context directory.

    After calling this, *dest* contains the resources that Dockerfile
    ``COPY`` directives reference:

    - ``scripts/``     — container helper scripts (init, env, ACP wrappers)
    - ``toad-agents/`` — ACP agent TOML definitions
    - ``tmux/``        — container tmux config

    Dockerfiles themselves are **not** written here — they are rendered
    and placed by [`build_base_images`][terok_executor.container.build.build_base_images] (which calls this function
    internally).
    """
    dest.mkdir(parents=True, exist_ok=True)
    stage_scripts(dest / "scripts")
    stage_toad_agents(dest / "toad-agents")
    stage_tmux_config(dest / "tmux")


# ── Dockerfile rendering ──


def render_l0(base_image: str = DEFAULT_BASE_IMAGE, *, family: str | None = None) -> str:
    """Render the L0 (base dev) Dockerfile.

    The *base_image* is normalised before rendering so that blank or
    whitespace-only values produce a valid Dockerfile.  *family*
    (``"deb"`` or ``"rpm"``) selects the package-manager branch of the
    template; ``None`` resolves it via [`detect_family`][terok_executor.container.build.detect_family].
    """
    base_image = _normalize_base_image(base_image)
    fam = detect_family(base_image, override=family)
    return _render_template(
        "l0.dev.Dockerfile.template",
        {"BASE_IMAGE": base_image, "family": fam},
    )


# Map of package family → systemd unit names for the L0G guest's sshd.
# Two units per family: the **socket** unit (our vsock-only override
# attaches to it) and the **service** unit (the distro's
# ``openssh-server`` post-install enables this; we mask it so a
# default-enabled TCP listener can never come up behind our back).
# The regular L0 image doesn't run sshd as a system service so it has
# no equivalent need.
_FAMILY_SSH_UNITS: dict[str, dict[str, str]] = {
    "deb": {"socket": "ssh.socket", "service": "ssh.service"},
    "rpm": {"socket": "sshd.socket", "service": "sshd.service"},
}


def render_l0g(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    family: str | None = None,
    host_pubkey: str = "",
) -> str:
    """Render the L0G (krun guest) Dockerfile.

    L0G is the base image for terok krun microVM guests — parallel to
    L0 but swapped for an in-guest workload: socket-activated sshd
    bound to AF_VSOCK with *host_pubkey* baked into authorized_keys.

    *base_image* and *host_pubkey* are validated in-place — the function
    is a public security boundary, not a thin formatting wrapper, so a
    caller that hands untrusted input straight to it and pipes the
    result into ``podman build`` can't smuggle in an extra
    ``RUN curl … | sh`` line or sneak ``command=…`` options into
    ``authorized_keys``.  *host_pubkey* must be a single OpenSSH
    public-key line (``"ssh-… …"``); an empty value is rejected.
    """
    base_image = _validate_base_image(_normalize_base_image(base_image))
    host_pubkey = _validate_host_pubkey(host_pubkey)
    fam = detect_family(base_image, override=family)
    units = _FAMILY_SSH_UNITS[fam]
    return _render_template(
        "l0g.guest.Dockerfile.template",
        {
            "BASE_IMAGE": base_image,
            "family": fam,
            "ssh_socket_name": units["socket"],
            "ssh_service_name": units["service"],
            "KRUN_HOST_PUBKEY": host_pubkey,
        },
    )


def render_l1(
    l0_image: str,
    *,
    family: str,
    agents: tuple[str, ...] | str = "all",
    cache_bust: str = "0",
) -> str:
    """Render the L1 (agent CLI) Dockerfile for the given agent selection.

    *l0_image* is the tag of the L0 image to build on top of.  *family*
    (``"deb"`` or ``"rpm"``) selects the package-manager branch and is
    required — there is no L0 reference to detect from at this point, so
    callers must supply the value resolved at the L0 level (typically via
    [`detect_family`][terok_executor.container.build.detect_family]).  Each roster install snippet is itself rendered
    as a Jinja template with ``family`` in scope, so snippets can carry
    ``{% if family == "deb" %}…{% else %}…{% endif %}`` branches for
    package-manager-specific commands.  *agents* is a tuple of
    already-resolved roster names (or the literal string ``"all"``); the
    template loops over them and emits each one's install snippets.
    *cache_bust* invalidates the per-agent install layers when changed
    (typically set to a Unix timestamp).
    """
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    selected = roster.resolve_selection(agents)
    installs = roster.installs

    root_snippets = [
        _render_snippet(installs[n].run_as_root, family)
        for n in selected
        if installs[n].run_as_root
    ]
    dev_snippets = [
        _render_snippet(installs[n].run_as_dev, family) for n in selected if installs[n].run_as_dev
    ]

    return _render_template(
        "l1.agent-cli.Dockerfile.template",
        {
            "BASE_IMAGE": l0_image,
            "AGENT_CACHE_BUST": cache_bust,
            "family": family,
            "install_root_snippets": root_snippets,
            "install_dev_snippets": dev_snippets,
            "installed_agents_csv": ",".join(selected),
            "agents_label": AGENTS_LABEL,
            "installed_env_path": INSTALLED_ENV_PATH,
        },
    )


def render_l1_sidecar(
    l0_image: str,
    *,
    family: str,
    tool_name: str = "coderabbit",
    cache_bust: str = "0",
) -> str:
    """Render the L1 sidecar (tool-only) Dockerfile.

    The sidecar image is built FROM L0 (not L1) and installs a single
    tool binary — no agent CLIs, no LLMs.  *family* (required) selects
    the package-manager branch; *tool_name* selects which tool install
    block to activate via Jinja2 conditional.
    """
    return _render_template(
        "l1.sidecar.Dockerfile.template",
        {
            "BASE_IMAGE": l0_image,
            "TOOL_CACHE_BUST": cache_bust,
            "tool_name": tool_name,
            "family": family,
        },
    )


# ── Resource staging ──


def stage_scripts(dest: Path) -> None:
    """Stage container helper scripts into *dest*.

    Copies executor's own ``resources/scripts/`` then overlays the
    socat-based bridge scripts that ship with ``terok_sandbox``
    (``ensure-bridges.sh`` + ``ssh-agent-bridge.sh``).  The bridges
    live in sandbox because they encode sandbox-level concerns with
    no executor-specific logic; executor still bundles them into the
    container image so the Dockerfile's ``COPY scripts/…`` lines keep
    finding them at their established names.

    Raises [`BuildError`][terok_executor.container.build.BuildError]
    when ``terok_sandbox`` is not importable — sandbox is a hard
    dependency in ``pyproject.toml``, but a broken install would
    otherwise surface as a raw ``ModuleNotFoundError`` traceback.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_executor", "resources/scripts", dest)
    try:
        _copy_package_tree("terok_sandbox", "resources/bridges", dest)
    except ModuleNotFoundError as exc:
        raise BuildError(
            "terok_sandbox is not importable — its bridge resources "
            "(resources/bridges/) could not be staged into the build "
            "context.  Reinstall terok-executor's dependencies to fix."
        ) from exc
    _clean_packaging_artifacts(dest)


def stage_toad_agents(dest: Path) -> None:
    """Stage Toad ACP agent TOML definitions into *dest*.

    These describe OpenCode-based agents (Blablador, KISSKI, etc.) that are
    injected into Toad's bundled agent directory at container build time.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_executor", "resources/toad-agents", dest)
    _clean_packaging_artifacts(dest)


def stage_help_fragments(dest: Path, agents: tuple[str, ...]) -> None:
    """Render per-section ``hilfe`` help fragments into *dest*.

    Writes one file per section (currently ``agent`` and ``dev_tool``)
    containing the labels of the selected agents that have a ``help:``
    section, with backslash escapes (``\\033[...]``) interpreted into
    real ANSI sequences so that ``hilfe`` only needs to ``cat`` them.
    Empty sections are omitted entirely; ``hilfe`` skips missing files.
    """
    from terok_executor.roster.loader import get_roster

    roster = get_roster()
    helps = roster.helps

    by_section: dict[str, list[str]] = {}
    for name in agents:
        spec = helps.get(name)
        if spec is None or not spec.label:
            continue
        by_section.setdefault(spec.section, []).append(spec.label)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for section, lines in by_section.items():
        decoded = "".join(_decode_label_escapes(line) + "\n" for line in lines)
        (dest / _HELP_SECTION_FILES[section]).write_text(decoded, encoding="utf-8")


def stage_tmux_config(dest: Path) -> None:
    """Stage the container tmux configuration into *dest*.

    Copies ``container-tmux.conf`` — the green-status-bar config that
    distinguishes container tmux sessions from host tmux.
    """
    if dest.exists():
        shutil.rmtree(dest)
    _copy_package_tree("terok_executor", "resources/tmux", dest)
    _clean_packaging_artifacts(dest)


# ── Image naming ──


def l0_image_tag(base_image: str) -> str:
    """Return the L0 base dev image tag for *base_image*."""
    return f"terok-l0:{_base_tag(base_image)}"


def l1_image_tag(base_image: str, agents: tuple[str, ...] | None = None) -> str:
    """Return the L1 agent CLI image tag for *base_image* and a selection.

    When *agents* is ``None``, returns the unsuffixed **default-alias**
    (e.g. ``terok-l1-cli:fedora-44``).  This alias points at whichever
    L1 was last built with ``tag_as_default=True`` — i.e. the L1 that
    holds the user's configured default agent selection.  Project /
    per-agent / partial builds get only their suffixed tag and never
    touch the alias, so ``terok auth <provider>`` can rely on the alias
    actually containing every agent the user configured.

    When *agents* is a tuple of names, appends a sorted ``-a-b-c``
    suffix (``-`` is the only spec-valid separator that ``_base_tag``
    already uses) so multiple selections coexist in the local image
    store and stay individually addressable.  Agent name fragments are
    passed through the same ``_base_tag`` sanitiser to keep the final
    tag within the OCI tag charset (``[A-Za-z0-9_.-]``).

    The full tag (after ``:``) is bounded by `_MAX_TAG_LEN`.  When
    the readable ``base-a-b-c`` form would overflow, the agent portion
    is replaced with a SHA1 digest of the sorted selection — same
    collision-resistant fallback pattern `_base_tag` uses
    internally for overlong image names.
    """
    base_tag = _base_tag(base_image)
    if agents is None:
        return f"terok-l1-cli:{base_tag}"
    readable_suffix = "-".join(_base_tag(a) for a in sorted(agents)) if agents else "empty"
    if len(base_tag) + 1 + len(readable_suffix) <= _MAX_TAG_LEN:
        return f"terok-l1-cli:{base_tag}-{readable_suffix}"
    suffix = hashlib.sha1(
        ",".join(sorted(agents)).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:_AGENT_DIGEST_LEN]
    # Pathological case: a base already near _MAX_TAG_LEN leaves no room
    # even for the digest.  Trim base_tag further — same collision-resistant
    # shape as the digest fallback itself.
    max_base = _MAX_TAG_LEN - 1 - _AGENT_DIGEST_LEN
    if len(base_tag) > max_base:
        base_tag = base_tag[:max_base]
    return f"terok-l1-cli:{base_tag}-{suffix}"


def l1_sidecar_image_tag(base_image: str) -> str:
    """Return the L1 sidecar (tool-only) image tag for *base_image*."""
    return f"terok-l1-sidecar:{_base_tag(base_image)}"


def l0g_image_tag(base_image: str) -> str:
    """Return the L0G (krun guest) image tag for *base_image*."""
    return f"terok-l0g:{_base_tag(base_image)}"


# ── Private helpers ──


def _validate_build_dir(build_dir: Path | None) -> None:
    """Reject *build_dir* if it is a file or a non-empty directory."""
    if build_dir is None:
        return
    if build_dir.is_file():
        raise ValueError(f"build_dir is a file, not a directory: {build_dir}")
    if build_dir.exists() and any(build_dir.iterdir()):
        raise ValueError(f"build_dir must be empty or absent: {build_dir}")


def _normalize_base_image(base_image: str | None) -> str:
    """Normalize a base image string, falling back to the default."""
    return (base_image or "").strip() or DEFAULT_BASE_IMAGE


# Permissive OCI-ish reference shape; the WHY (Dockerfile-injection
# avoidance) lives in ``_validate_base_image``'s docstring.
#
# Three optional pieces compose: an upfront ``host[:port]/`` registry
# component (the ``:port`` is what made ``localhost:5000/ubuntu:24.04``
# previously fail — ``:`` was only allowed once, in the tag slot), the
# name body, and the usual ``:tag`` / ``@sha256:digest`` trailers.
_BASE_IMAGE_RE = re.compile(
    r"""^
        (?:                                         # optional registry component
            [A-Za-z0-9]                             #   host: leading alnum
            (?:[A-Za-z0-9.-]*[A-Za-z0-9])?          #   host body (dots/dashes, no slash)
            (?::[0-9]+)?                            #   optional :port (digits)
            /                                       #   trailing slash separates from name
        )?
        [A-Za-z0-9]                                 # name: leading alnum
        (?:[A-Za-z0-9._/-]*[A-Za-z0-9])?            # name body, last char alnum
        (?::[A-Za-z0-9_][A-Za-z0-9_.-]*)?           # optional :tag
        (?:@sha256:[A-Fa-f0-9]{64})?                # optional @digest
        $
    """,
    re.VERBOSE,
)


def _validate_base_image(ref: str) -> str:
    """Reject base images that could inject Dockerfile directives at render time.

    ``ARG BASE_IMAGE={{ BASE_IMAGE }}`` interpolates *ref* into the
    rendered Dockerfile.  A value carrying ``\\n`` or other control
    bytes would terminate the ARG line and let the rest of the string
    be parsed as additional instructions (``RUN curl ... | sh``,
    ``ADD ...``).  Validate up-front; the loose OCI shape covers the
    real-world bases that ``detect_family`` already accepts without
    pretending to be a complete parser.
    """
    if any(c.isspace() or ord(c) < 0x20 for c in ref):
        raise BuildError(f"base_image {ref!r}: contains whitespace or control characters")
    if len(ref) > 256:
        raise BuildError(f"base_image {ref!r}: exceeds 256-character bound")
    if not _BASE_IMAGE_RE.fullmatch(ref):
        raise BuildError(f"base_image {ref!r}: not a recognisable OCI reference shape")
    return ref


# OpenSSH public-key line: ``<type> <base64> [comment]``.  The comment
# may contain spaces but never a newline (RFC 4253 §6.6).  Keep the key
# types tight to the ones terok actually ships; expand the list when
# the orchestrator starts generating other types.
_HOST_PUBKEY_RE = re.compile(
    r"^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521))"
    r"\s+[A-Za-z0-9+/=]+"
    r"(?:\s+\S(?:.*\S)?)?$"  # optional comment (no leading/trailing space)
)


def _validate_host_pubkey(host_pubkey: str) -> str:
    """Return a single-line OpenSSH public-key string, or raise.

    The value is written verbatim into ``authorized_keys.d/terok`` via
    ``printf '%s\\n' "${KRUN_HOST_PUBKEY}"`` at image build time.  A
    multi-line value would add extra trust entries or inject
    ``authorized_keys`` options (``command=``, ``environment=``,
    ``from=``) that change the guest's security posture.  Reject both
    cases up front.
    """
    if not isinstance(host_pubkey, str):
        raise BuildError(f"host_pubkey must be a string, got {type(host_pubkey).__name__}")
    key = host_pubkey.strip()
    if not key:
        raise BuildError(
            "host_pubkey must be a non-empty SSH public-key line; "
            "an empty value would build an image that accepts no connections."
        )
    if "\n" in key or "\r" in key:
        raise BuildError(
            "host_pubkey must be a single line — multi-line values would "
            "add extra trust entries to authorized_keys.d/terok"
        )
    if len(key) > 8192:
        raise BuildError(f"host_pubkey: {len(key)} chars exceeds 8192-byte cap")
    if not _HOST_PUBKEY_RE.fullmatch(key):
        raise BuildError(
            "host_pubkey: not a recognisable OpenSSH public-key line "
            "(expected '<type> <base64> [comment]' with a supported key type)"
        )
    return key


def _split_image_ref(ref: str) -> tuple[str, str]:
    """Split an OCI image reference into ``(name_without_tag, tag)``.

    Strips an optional ``@digest`` suffix first, then peels off the
    trailing ``:tag`` only when the last ``:`` lies after the last ``/``
    — so ``localhost:5000/fedora:44`` keeps the registry port intact
    in *name* and yields ``"24.04"`` as *tag*.  Refs without a tag
    return an empty string for *tag*.
    """
    name = ref.split("@", 1)[0]  # drop digest
    if name.rfind(":") > name.rfind("/"):
        name, _, tag = name.rpartition(":")
        return name, tag
    return name, ""


def _base_tag(base_image: str) -> str:
    """Derive a safe OCI tag fragment from an arbitrary base image string.

    Replaces non-alphanumeric characters (except ``_``, ``.``, ``-``) with
    dashes, lowercases, and truncates with a SHA1 suffix if too long.
    """
    raw = _normalize_base_image(base_image)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.").lower() or _DEFAULT_TAG
    if len(tag) > 120:
        digest = hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        tag = f"{tag[:111]}-{digest}"
    return tag


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    """Shared stateless Jinja2 environment for all Dockerfile rendering."""
    return Environment(  # nosec B701 — Dockerfile output, not HTML
        loader=BaseLoader(), keep_trailing_newline=True, autoescape=False
    )


def _render_template(template_name: str, variables: dict[str, Any]) -> str:
    """Render a Jinja2 Dockerfile template from package resources.

    Templates live in ``resources/templates/``.  The L0/L1 templates
    branch on a ``family`` variable (``deb``/``rpm``); the L1 template
    also iterates over pre-rendered per-agent install snippets.
    """
    raw = (resources.files("terok_executor") / "resources" / "templates" / template_name).read_text(
        encoding="utf-8"
    )
    return _jinja_env().from_string(raw).render(**variables)


def _render_snippet(snippet: str, family: str) -> str:
    """Render a per-agent install snippet with *family* in Jinja scope.

    Roster install snippets may contain ``{% if family == "deb" %}…{% else %}…
    {% endif %}`` branches; non-family-aware snippets pass through unchanged.
    """
    return _jinja_env().from_string(snippet).render(family=family)


def _copy_package_tree(package: str, rel_path: str, dest: Path) -> None:
    """Copy a directory tree from package resources to a filesystem path.

    Uses ``importlib.resources`` Traversable API so it works from
    wheels and zip installs.
    """
    root = resources.files(package) / rel_path

    def _recurse(src, dst: Path) -> None:  # type: ignore[no-untyped-def]
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            out = dst / child.name
            if child.is_dir():
                _recurse(child, out)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(child.read_bytes())

    _recurse(root, dest)


def _clean_packaging_artifacts(dest: Path) -> None:
    """Remove __pycache__ dirs and __init__.py from a staged directory."""
    for unwanted in dest.rglob("__pycache__"):
        shutil.rmtree(unwanted)
    init = dest / "__init__.py"
    if init.exists():
        init.unlink()


def _check_podman() -> None:
    """Raise [`BuildError`][terok_executor.container.build.BuildError] if podman is not on PATH."""
    if shutil.which("podman") is None:
        raise BuildError("podman not found; please install podman")


def _image_exists(image: str) -> bool:
    """Check if a container image exists locally."""
    result = subprocess.run(
        ["podman", "image", "exists", image],
        capture_output=True,
    )
    return result.returncode == 0


def image_agents(image: str) -> set[str]:
    """Return the set of agent names installed in *image*.

    Reads the ``ai.terok.agents`` OCI label baked into the L1 image at
    build time (see [`AGENTS_LABEL`][terok_executor.container.build.AGENTS_LABEL]).
    Returns an empty set if the image is missing, has no label, or its
    label is empty — never raises on inspection failure, since callers
    use the result to make a "image good enough?" decision and an
    empty set always means "no, rebuild".
    """
    try:
        result = subprocess.run(
            [
                "podman",
                "image",
                "inspect",
                image,
                "--format",
                f'{{{{ index .Config.Labels "{AGENTS_LABEL}" }}}}',
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    csv = result.stdout.strip()
    return {a for a in csv.split(",") if a} if csv else set()


def ensure_default_l1(
    base_image: str = DEFAULT_BASE_IMAGE,
    *,
    family: str | None = None,
    agents: str | tuple[str, ...] = "all",
) -> str:
    """Return the default-alias L1 tag, building the user's default L1 if absent.

    Used by ``terok auth`` (and the equivalent standalone-executor flow)
    to resolve a host-wide L1 image that contains every agent the user
    has configured.  If the alias already exists locally, it is trusted
    and returned as-is — the alias is reserved for the user's default
    selection (see [`l1_image_tag`][terok_executor.container.build.l1_image_tag]),
    so its contents are well-defined.  When the alias is missing the
    function builds it via [`build_base_images`][terok_executor.container.build.build_base_images]
    with ``tag_as_default=True``.

    *agents* defaults to the literal string ``"all"`` so standalone
    callers get the whole roster.  terok passes the user's configured
    ``image.agents`` value here so the alias means *"every agent the
    user has enabled"* rather than the implementation-default roster.
    """
    alias = l1_image_tag(base_image)
    if _image_exists(alias):
        return alias
    build_base_images(
        base_image=base_image,
        family=family,
        agents=agents,
        tag_as_default=True,
    )
    return alias
