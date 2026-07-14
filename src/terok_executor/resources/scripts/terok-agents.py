#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

# terok:container — this file is deployed into task containers, not used on the host.

"""Report which agents are usable, and which (agent, provider) pairs are ready.

Writes ``/home/dev/.terok/agents.json`` with two views of the container's
*actual runtime state*:

  * ``pairs`` — every installed protocol-speaking agent paired with every
    authenticated provider, flagged ``ready`` when the provider serves the
    agent's wire protocol.  Consumed by the ``providers`` command.
  * ``agents`` — one rollup per installed agent: its banner ``label``, whether
    it is ``usable`` at all (has *some* authenticated provider it can talk to),
    and a short ``reason`` when it is not.  Consumed by the ``hilfe`` banner,
    which dims unusable agents.
  * ``protocols`` — one row per wire protocol in play: its candidate providers
    and which are authenticated.  Consumed by the ``hilfe`` providers section so
    a dimmed ``needs an openai-chat provider`` maps to a list to act on.

The views are derived from three in-container sources:

  * installed agents       — ``TEROK_INSTALLED_AGENTS`` in
    ``/etc/terok/installed.env``
  * authenticated providers — a ``TEROK_PROVIDER_<NAME>_TOKEN`` per provider
  * served protocols        — a ``TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL>`` per
    protocol the provider actually serves

plus two baked maps the environment cannot reconstruct: per-agent facts
(protocol, label) in ``agent-protocols.json``, and the protocol → candidate
providers universe in ``provider-protocols.json``.

Run with no arguments to (re)write the manifest — done at container startup,
and any time to refresh.  Run with ``--banner`` / ``--protocols`` to print just
the agent list / candidate-provider list for the ``hilfe`` login banner,
deriving them live without touching the file.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

# ── Container contract (paths and names the image build agrees on) ──

MANIFEST_VERSION = 2
"""Schema version of the emitted ``agents.json`` (2 added the ``agents`` rollup)."""

INSTALLED_ENV_PATH = Path("/etc/terok/installed.env")
"""Image manifest carrying ``TEROK_INSTALLED_AGENTS=<comma,separated>``."""

AGENT_FACTS_PATH = Path("/usr/local/share/terok/agent-protocols.json")
"""Baked per-agent facts (protocol, label) — a non-environment input.
Filename is historical; see the build-side constant ``AGENT_PROTOCOLS_PATH``
for why the name outlived the protocols-only contents."""

PROVIDER_PROTOCOLS_PATH = Path("/usr/local/share/terok/provider-protocols.json")
"""Baked ``protocol → [providers that serve it]`` universe — lets the providers
banner list the candidate providers a user could authenticate to enable a
dimmed agent, including providers not yet authenticated."""

_DIM = "\033[2m"
_BOLD = "\033[1m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
"""Matches the SGR escapes embedded in a baked banner label, so a dimmed line
can be flattened to plain text before being wrapped in a single dim span."""

MANIFEST_PATH = Path("/home/dev/.terok/agents.json")
"""Where the readiness manifest is written for agents and the operator to read."""

_TOKEN_PREFIX = "TEROK_PROVIDER_"  # nosec B105 — env-var name fragment, not a secret
"""Env-var prefix shared by every per-provider token and base-URL variable."""

_TOKEN_SUFFIX = "_TOKEN"  # nosec B105 — env-var name fragment, not a secret
"""Env-var suffix that marks a provider's phantom-token variable."""

_BASE_MARKER = "_BASE_"
"""Separator between the provider name and protocol in a base-URL variable."""


# ── Entry point ──


def main(argv: list[str]) -> None:
    """Write the readiness manifest, or render a ``hilfe`` section.

    ``--banner`` prints the dimmed/bright agent list; ``--protocols`` prints the
    per-protocol candidate-provider list for the providers section.  Both derive
    live and leave the manifest file untouched — a login banner is a read, not a
    checkpoint.  With no flag the manifest is (re)written.
    """
    flags = argv[1:]
    installed = _read_installed_agents(INSTALLED_ENV_PATH)
    facts = _read_agent_facts(AGENT_FACTS_PATH)
    universe = _read_provider_protocols(PROVIDER_PROTOCOLS_PATH)
    manifest = build_manifest(installed, facts, universe, os.environ)

    if "--banner" in flags:
        _print_if_nonempty(render_agent_banner(manifest["agents"]))
    elif "--protocols" in flags:
        _print_if_nonempty(render_provider_protocols(manifest["protocols"]))
    else:
        _write_manifest(MANIFEST_PATH, manifest)
        _print_summary(manifest)


def _print_if_nonempty(text: str) -> None:
    """Print *text* only when it has content (an empty section prints nothing)."""
    if text:
        print(text)


# ── Derivation (pure: three runtime inputs → the JSON shape) ──


def build_manifest(
    installed_agents: set[str],
    agent_facts: dict[str, dict[str, str | None]],
    provider_protocols: dict[str, list[str]],
    env: Mapping[str, str],
) -> dict[str, object]:
    """Return the readiness manifest for the given runtime state.

    Produces three views over the same facts:

    - ``pairs``: every *installed* protocol-speaking agent paired with every
      authenticated provider, flagged ``ready`` when the provider serves the
      agent's wire protocol.  Installed-ness and authentication are the
      enumeration domain (filters, not fields); readiness is the only per-pair
      fact worth recording.
    - ``agents``: one rollup per installed agent — ``label``, whether it is
      ``usable`` at all, and a ``reason`` when not.  This is what lets ``hilfe``
      dim an agent the container can't actually run.
    - ``protocols``: one row per wire protocol in play — its candidate providers
      and which of them are authenticated.  This is what lets ``hilfe`` turn a
      dimmed agent's ``needs an openai-chat provider`` into an actionable list.

    Args:
        installed_agents: Agent names present in this image (from
            ``TEROK_INSTALLED_AGENTS``) — the enumeration filter.
        agent_facts: Agent name → ``{protocol, label}``, baked from the roster.
        provider_protocols: Protocol → candidate provider names, baked from the
            roster (the providers a user *could* authenticate, beyond the ones
            the environment already carries).
        env: The container environment, scanned for authenticated providers
            and the protocols each one serves.

    Returns:
        ``{"version", "pairs", "agents", "protocols"}`` — see above per view.
    """
    authenticated = _authenticated_providers(env)
    served = _served_protocols(env)

    pairs: list[dict[str, object]] = []
    agents: list[dict[str, object]] = []
    for name in sorted(agent_facts):
        if name not in installed_agents:
            continue
        protocol = agent_facts[name].get("protocol")
        label = agent_facts[name].get("label") or name

        if protocol:
            wire = _protocol_token(protocol)
            for prov in sorted(authenticated):
                ready = wire in served.get(prov, frozenset())
                pairs.append(
                    {"agent": name, "provider": prov, "protocol": protocol, "ready": ready}
                )

        usable, reason = _assess(protocol, authenticated, served)
        agents.append({"name": name, "label": label, "usable": usable, "reason": reason})

    protocols = [
        {
            "protocol": protocol,
            "candidates": candidates,
            "authenticated": [p for p in candidates if p in authenticated],
        }
        for protocol, candidates in sorted(provider_protocols.items())
    ]
    return {"version": MANIFEST_VERSION, "pairs": pairs, "agents": agents, "protocols": protocols}


def _assess(
    protocol: str | None,
    authenticated: set[str],
    served: dict[str, set[str]],
) -> tuple[bool, str]:
    """Return ``(usable, reason)`` for one agent given the live provider state.

    A protocol-speaking agent is usable when *some* authenticated provider's
    ``serves`` covers its wire protocol — any compatible provider, not a single
    default.  An agent with no fixed protocol (a frontend like ``toad``) has
    nothing to assess, so it is always shown bright.

    The ``reason`` (only meaningful when not usable) names the protocol no
    authenticated provider speaks — ``needs an openai-chat provider`` — and the
    providers section lists which providers would satisfy it.
    """
    if not protocol:
        return True, ""
    wire = _protocol_token(protocol)
    if any(wire in served.get(p, frozenset()) for p in authenticated):
        return True, ""
    return False, f"needs an {protocol} provider"


def render_agent_banner(agents: object) -> str:
    """Render the ``agents`` rollup as ``hilfe``'s bright/dimmed agent list.

    Usable agents print their baked label verbatim (the command stays cyan);
    unusable ones are flattened to plain text, wrapped in a single dim span, and
    suffixed with ``(reason)`` so the *why* travels with the *what*.
    """
    if not isinstance(agents, list):
        return ""
    lines: list[str] = []
    for entry in agents:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", ""))
        if entry.get("usable"):
            lines.append(label)
        else:
            plain = _ANSI_RE.sub("", label)
            lines.append(f"{_DIM}{plain}  ({entry.get('reason', '')}){_RESET}")
    return "\n".join(lines)


def render_provider_protocols(protocols: object) -> str:
    """Render the ``protocols`` rollup as ``hilfe``'s candidate-provider list.

    One line per wire protocol: its candidate providers, with the authenticated
    ones in magenta (matching the providers section's colour) and the rest
    dimmed — so an agent dimmed with ``needs an openai-chat provider`` maps to a
    concrete row of providers to authenticate.
    """
    if not isinstance(protocols, list):
        return ""
    rows = [r for r in protocols if isinstance(r, dict) and r.get("candidates")]
    if not rows:
        return ""
    width = max(len(str(r["protocol"])) for r in rows)
    lines = [f"{_DIM}Providers by wire protocol (authenticate one to enable its agents):{_RESET}"]
    for row in rows:
        authed = set(row.get("authenticated") or [])
        names = ", ".join(
            f"{_MAGENTA}{p}{_RESET}" if p in authed else f"{_DIM}{p}{_RESET}"
            for p in row["candidates"]
        )
        lines.append(f"  {str(row['protocol']):<{width}}  {names}")
    return "\n".join(lines)


def _authenticated_providers(env: Mapping[str, str]) -> set[str]:
    """Return the providers this container holds a phantom token for.

    Reads one ``TEROK_PROVIDER_<NAME>_TOKEN`` per authenticated provider and
    returns the lower-cased clean names (``ANTHROPIC`` → ``anthropic``).
    """
    providers: set[str] = set()
    for key in env:
        if key.startswith(_TOKEN_PREFIX) and key.endswith(_TOKEN_SUFFIX):
            name = key[len(_TOKEN_PREFIX) : -len(_TOKEN_SUFFIX)]
            if name and _BASE_MARKER not in key:
                providers.add(name.lower())
    return providers


def _served_protocols(env: Mapping[str, str]) -> dict[str, set[str]]:
    """Return provider name → the wire-protocol tokens it serves.

    Each ``TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL>`` variable announces one
    protocol the provider serves; the protocol token is kept in its env-var
    form (upper snake) so it compares directly against the token returned by
    ``_protocol_token``.
    """
    served: dict[str, set[str]] = {}
    for key in env:
        if not key.startswith(_TOKEN_PREFIX) or _BASE_MARKER not in key:
            continue
        name, _, protocol = key[len(_TOKEN_PREFIX) :].partition(_BASE_MARKER)
        if name and protocol:
            served.setdefault(name.lower(), set()).add(protocol)
    return served


def _protocol_token(protocol: str) -> str:
    """Return *protocol* in the upper-snake form used in base-URL var names.

    ``anthropic-messages`` → ``ANTHROPIC_MESSAGES``, matching how the env
    builder encodes the protocol into ``TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL>``.
    """
    return protocol.upper().replace("-", "_")


# ── Sources and sink (the I/O around the pure core) ──


def _read_installed_agents(path: Path) -> set[str]:
    """Return the agent names from ``TEROK_INSTALLED_AGENTS`` in *path*.

    A missing or unreadable file yields an empty set — the manifest then has
    no agents to pair, rather than the container failing to start.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "TEROK_INSTALLED_AGENTS":
            return {name.strip() for name in value.split(",") if name.strip()}
    return set()


def _read_agent_facts(path: Path) -> dict[str, dict[str, str | None]]:
    """Return the baked per-agent facts map, or empty when absent.

    Each value is normalised to ``{protocol, label}`` with missing keys
    defaulted to ``None``.  A missing file or malformed JSON yields an empty map
    (nothing resolves as usable) rather than raising at startup.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    facts: dict[str, dict[str, str | None]] = {}
    for name, fact in data.items():
        if not isinstance(fact, dict):
            continue
        facts[str(name)] = {"protocol": fact.get("protocol"), "label": fact.get("label")}
    return facts


def _read_provider_protocols(path: Path) -> dict[str, list[str]]:
    """Return the baked ``protocol → [providers]`` universe, or empty when absent.

    A missing file or malformed JSON yields an empty map (the providers section
    simply lists no candidates) rather than raising at startup.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    universe: dict[str, list[str]] = {}
    for protocol, providers in data.items():
        if isinstance(providers, list):
            universe[str(protocol)] = [str(p) for p in providers]
    return universe


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write *manifest* to *path* as stable, pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(manifest: dict[str, object]) -> None:
    """Print a short, human-readable summary of the ready pairs."""
    pairs = manifest.get("pairs", [])
    ready = [p for p in pairs if isinstance(p, dict) and p.get("ready")]
    if ready:
        print("Ready agent/provider pairs:")
        for pair in ready:
            print(f"  {pair['agent']} -> {pair['provider']} ({pair['protocol']})")
    else:
        print("No agent/provider pairs are ready in this container.")
    print(f"\nFull readiness manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    # Advisory metadata must never abort container init, so any failure is
    # reported and swallowed — the command always exits 0.
    try:
        main(sys.argv)
    except Exception as exc:  # noqa: BLE001 — advisory metadata must not abort init
        print(f"terok-agents: could not write readiness manifest: {exc}", file=sys.stderr)
    sys.exit(0)
