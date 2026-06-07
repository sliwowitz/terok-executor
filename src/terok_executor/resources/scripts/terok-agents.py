#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

# terok:container — this file is deployed into task containers, not used on the host.

"""Report which (agent, provider) pairs are ready to use in this container.

Writes ``/home/dev/.terok/agents.json`` listing, for every installed agent
paired with every provider this container is authenticated for, whether the
agent can actually use that provider (``ready`` — the provider serves the
agent's wire protocol).  Installed-ness and authentication define which pairs
*appear* at all, so the only per-pair fact worth recording is readiness.

The manifest reflects the container's *actual runtime state*, read from three
in-container sources:

  * installed agents       — ``TEROK_INSTALLED_AGENTS`` in
    ``/etc/terok/installed.env``
  * authenticated providers — a ``TEROK_PROVIDER_<NAME>_TOKEN`` per provider
  * served protocols        — a ``TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL>`` per
    protocol the provider actually serves

The agent→protocol vocabulary is the one fact the environment does not carry;
it is baked into the image at build time and read from
``/usr/local/share/terok/agent-protocols.json``.

The manifest is generated at container startup; re-run the ``terok-agents``
command at any time to refresh it.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

# ── Container contract (paths and names the image build agrees on) ──

MANIFEST_VERSION = 1
"""Schema version of the emitted ``agents.json``."""

INSTALLED_ENV_PATH = Path("/etc/terok/installed.env")
"""Image manifest carrying ``TEROK_INSTALLED_AGENTS=<comma,separated>``."""

AGENT_PROTOCOLS_PATH = Path("/usr/local/share/terok/agent-protocols.json")
"""Baked agent→wire-protocol map — the only non-environment input."""

MANIFEST_PATH = Path("/home/dev/.terok/agents.json")
"""Where the readiness manifest is written for agents and the operator to read."""

_TOKEN_PREFIX = "TEROK_PROVIDER_"  # nosec B105 — env-var name fragment, not a secret
"""Env-var prefix shared by every per-provider token and base-URL variable."""

_TOKEN_SUFFIX = "_TOKEN"  # nosec B105 — env-var name fragment, not a secret
"""Env-var suffix that marks a provider's phantom-token variable."""

_BASE_MARKER = "_BASE_"
"""Separator between the provider name and protocol in a base-URL variable."""


# ── Entry point ──


def main() -> int:
    """Derive the readiness manifest from the live container state and write it."""
    installed = _read_installed_agents(INSTALLED_ENV_PATH)
    protocols = _read_agent_protocols(AGENT_PROTOCOLS_PATH)
    manifest = build_manifest(installed, protocols, os.environ)
    _write_manifest(MANIFEST_PATH, manifest)
    _print_summary(manifest)
    return 0


# ── Derivation (pure: three runtime inputs → the JSON shape) ──


def build_manifest(
    installed_agents: set[str],
    agent_protocols: dict[str, str],
    env: Mapping[str, str],
) -> dict[str, object]:
    """Return the readiness manifest for the given runtime state.

    Pairs every *installed* protocol-speaking agent with every provider this
    container is authenticated for, and records whether the pair is ``ready``
    — i.e. the provider serves the agent's wire protocol.  Installed-ness and
    authentication are the *enumeration domain* (an agent must be installed to
    appear; a provider must be authenticated for the container to know it
    exists at all), so they are filters, not per-pair fields: the only thing
    that varies across a pair, and the only thing worth recording, is protocol
    compatibility.

    Args:
        installed_agents: Agent names present in this image (from
            ``TEROK_INSTALLED_AGENTS``) — the agent enumeration filter.
        agent_protocols: Agent name → wire protocol, for the agents that speak
            one (baked from the roster).
        env: The container environment, scanned for authenticated providers
            and the protocols each one serves.

    Returns:
        ``{"version": int, "pairs": [...]}`` where each pair carries
        ``agent``, ``provider``, ``protocol``, and ``ready``.
    """
    authenticated = _authenticated_providers(env)
    served = _served_protocols(env)

    pairs: list[dict[str, object]] = []
    for agent in sorted(agent_protocols):
        if agent not in installed_agents:
            continue
        protocol = agent_protocols[agent]
        wire = _protocol_token(protocol)
        for provider in sorted(authenticated):
            ready = wire in served.get(provider, frozenset())
            pairs.append(
                {"agent": agent, "provider": provider, "protocol": protocol, "ready": ready}
            )
    return {"version": MANIFEST_VERSION, "pairs": pairs}


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


def _read_agent_protocols(path: Path) -> dict[str, str]:
    """Return the baked agent→wire-protocol map, or empty when absent.

    A missing file or malformed JSON yields an empty map (the readiness check
    then resolves nothing as compatible) rather than raising at startup.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(name): str(protocol) for name, protocol in data.items() if protocol}


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
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — advisory metadata must not abort init
        print(f"terok-agents: could not write readiness manifest: {exc}", file=sys.stderr)
        sys.exit(0)
