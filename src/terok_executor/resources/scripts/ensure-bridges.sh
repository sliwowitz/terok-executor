# shellcheck shell=bash
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
# terok:container — this file is deployed into task containers, not used on the host.

# Idempotent socat bridge launcher for container ↔ host-side services.
#
# Manages four bridges:
#   1. SSH agent   — UNIX socket → ssh-agent-bridge.sh → TCP or host socket
#   2. gh proxy    — UNIX socket → TCP (plain relay to credential proxy)
#   3. Claude proxy — UNIX socket → TCP (enables OAuth subscription mode)
#   4. Gate server — TCP listener → host UNIX socket (git HTTP-over-socket)
#
# Transport selection is env-var driven — set at container creation, not
# auto-detected.  Socket mode mounts the host runtime dir at /run/terok/.
#
# Uses PID files (not socket existence) to detect dead bridges — stale
# socket files persist after process death and are unreliable sentinels.
#
# Designed to be *sourced* (not executed) so SSH_AUTH_SOCK propagates
# to the caller.  Called from:
#   - init-ssh-and-repo.sh  (first boot)
#   - terok-env.sh          (every shell — self-heal after restart)

_TEROK_PIDDIR=/tmp/.terok
mkdir -p "$_TEROK_PIDDIR" 2>/dev/null

_terok_bridge_alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null
}

# ── SSH agent bridge ─────────────────────────────────────────────────────
# Requires a phantom token.  Transport: TEROK_SSH_AGENT_SOCKET (mounted
# host socket) or TEROK_SSH_AGENT_PORT (TCP to host loopback).
if [[ -n "${TEROK_SSH_AGENT_TOKEN:-}" ]] \
   && { [[ -n "${TEROK_SSH_AGENT_SOCKET:-}" ]] || [[ -n "${TEROK_SSH_AGENT_PORT:-}" ]]; } \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/ssh-agent.pid"; then
  rm -f /tmp/ssh-agent.sock
  socat "UNIX-LISTEN:/tmp/ssh-agent.sock,fork" "SYSTEM:ssh-agent-bridge.sh" &
  echo $! > "$_TEROK_PIDDIR/ssh-agent.pid"
  export SSH_AUTH_SOCK=/tmp/ssh-agent.sock
fi

# ── gh credential proxy bridge ───────────────────────────────────────────
if [[ -n "${TEROK_PROXY_PORT:-}" ]] && [[ -n "${GH_TOKEN:-}" ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/gh-proxy.pid"; then
  rm -f /tmp/terok-gh-proxy.sock
  socat UNIX-LISTEN:/tmp/terok-gh-proxy.sock,fork \
    TCP:host.containers.internal:"${TEROK_PROXY_PORT}" &
  echo $! > "$_TEROK_PIDDIR/gh-proxy.pid"
fi

# ── Claude credential proxy bridge (ANTHROPIC_UNIX_SOCKET transport) ────
# Routes Claude API traffic through the credential proxy via a Unix socket
# instead of ANTHROPIC_BASE_URL (enables OAuth subscription mode in Claude Code).
if [[ -n "${TEROK_PROXY_PORT:-}" ]] && [[ -n "${ANTHROPIC_UNIX_SOCKET:-}" ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/claude-proxy.pid"; then
  rm -f "${ANTHROPIC_UNIX_SOCKET}"
  socat UNIX-LISTEN:"${ANTHROPIC_UNIX_SOCKET}",fork \
    TCP:host.containers.internal:"${TEROK_PROXY_PORT}" &
  echo $! > "$_TEROK_PIDDIR/claude-proxy.pid"
fi

# ── Gate server bridge (socket mode) ─────────────────────────────────────
# In socket mode the gate HTTP server listens on a host Unix socket mounted
# into the container.  Git needs HTTP URLs, so we bridge localhost:9418 to
# the mounted socket.  CODE_REPO / CLONE_FROM point to http://localhost:9418/.
if [[ -n "${TEROK_GATE_SOCKET:-}" ]] \
   && command -v socat >/dev/null 2>&1 \
   && ! _terok_bridge_alive "$_TEROK_PIDDIR/gate.pid"; then
  socat TCP-LISTEN:9418,fork,reuseaddr \
    UNIX-CONNECT:"${TEROK_GATE_SOCKET}" &
  echo $! > "$_TEROK_PIDDIR/gate.pid"
fi
