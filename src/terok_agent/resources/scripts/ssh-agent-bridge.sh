#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
# terok:container — this file is deployed into task containers, not used on the host.

# Relay between an SSH agent client (via socat SYSTEM: stdin/stdout) and
# the host-side SSH agent proxy over TCP, injecting the phantom token
# handshake as the first bytes on the TCP connection.
#
# Called by socat:
#   socat UNIX-LISTEN:...,fork "SYSTEM:ssh-agent-bridge.sh"
#
# stdin/stdout = the SSH client's Unix socket side (provided by socat)
#
# Expects env:
#   TEROK_SSH_AGENT_TOKEN  - phantom token (terok-p-<32hex> or raw 32-char hex)
#   TEROK_SSH_AGENT_PORT   - TCP port on host.containers.internal

set -euo pipefail

: "${TEROK_SSH_AGENT_TOKEN:?missing}"
: "${TEROK_SSH_AGENT_PORT:?missing}"

[[ "${TEROK_SSH_AGENT_PORT}" =~ ^[0-9]+$ ]] || {
  echo "TEROK_SSH_AGENT_PORT must be numeric" >&2
  exit 2
}

# Compute 4-byte big-endian length header dynamically — the server-side
# _read_handshake() accepts any token length 1–1024 and does a DB lookup.
TOKEN_LEN=${#TEROK_SSH_AGENT_TOKEN}

# The nested socat connects to the TCP server, but we need to send the
# token handshake before relaying SSH agent traffic.
# Length header: 4 bytes big-endian, then the token as ASCII.
{
  printf "\\x$(printf '%02x' $((TOKEN_LEN >> 24 & 0xFF)))"
  printf "\\x$(printf '%02x' $((TOKEN_LEN >> 16 & 0xFF)))"
  printf "\\x$(printf '%02x' $((TOKEN_LEN >> 8  & 0xFF)))"
  printf "\\x$(printf '%02x' $((TOKEN_LEN       & 0xFF)))"
  printf '%s' "${TEROK_SSH_AGENT_TOKEN}"
  cat
} | socat - "TCP:host.containers.internal:${TEROK_SSH_AGENT_PORT}"
