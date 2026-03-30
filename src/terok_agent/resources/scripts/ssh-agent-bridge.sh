#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

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
#   TEROK_SSH_AGENT_TOKEN  - phantom token (32-char hex)
#   TEROK_SSH_AGENT_PORT   - TCP port on host.containers.internal

set -euo pipefail

: "${TEROK_SSH_AGENT_TOKEN:?missing}"
: "${TEROK_SSH_AGENT_PORT:?missing}"

[[ "${TEROK_SSH_AGENT_TOKEN}" =~ ^[[:xdigit:]]{32}$ ]] || {
  echo "TEROK_SSH_AGENT_TOKEN must be 32 hex characters" >&2
  exit 2
}
[[ "${TEROK_SSH_AGENT_PORT}" =~ ^[0-9]+$ ]] || {
  echo "TEROK_SSH_AGENT_PORT must be numeric" >&2
  exit 2
}

# The nested socat connects to the TCP server, but we need to send the
# token prefix before relaying.  Use printf piped into socat's stdin
# concatenated with our own stdin (the SSH client data).
{
  printf '\x00\x00\x00\x20%s' "${TEROK_SSH_AGENT_TOKEN}"
  cat
} | socat - "TCP:host.containers.internal:${TEROK_SSH_AGENT_PORT}"
