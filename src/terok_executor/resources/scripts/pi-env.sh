# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
#
# /etc/profile.d/ snippet for Pi (https://pi.dev).  Sourced by every
# login shell — including the ``bash -lc`` wrappers terok uses for
# headless task runs, so the symlink + env aliases are in place before
# ``pi`` is invoked.

# Skip Pi's startup version check and the fd/rg auto-download from
# GitHub releases.  The L1 image ships fd/rg in PATH and image rebuild
# is terok's update channel — no startup egress needed.
export PI_OFFLINE=1

# Co-installed Claude exposes its phantom (or, in exposed-credentials
# mode, real) Anthropic token under CLAUDE_CODE_OAUTH_TOKEN; Pi reads
# the Anthropic OAuth token under ANTHROPIC_OAUTH_TOKEN.  Aliasing
# bridges the name mismatch in both modes — in phantom mode the alias
# lets the vault-routing extension forward the phantom under the env
# name Pi looks for, in exposed mode the real token reaches the
# Anthropic API direct.
if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_OAUTH_TOKEN:-}" ]; then
    export ANTHROPIC_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN"
fi

# Symlink terok's vault-routing extension into Pi's auto-discovery dir
# (~/.pi/agent/extensions/).  We ship the extension at a system-wide
# path because the user's config dir is a bind mount we don't own.
_pi_ext_link="${HOME}/.pi/agent/extensions/terok-vault-routes.mjs"
_pi_ext_src="/usr/local/share/terok/pi-extensions/vault-routes.mjs"
if [ -f "$_pi_ext_src" ] && [ ! -L "$_pi_ext_link" ]; then
    mkdir -p "$(dirname "$_pi_ext_link")"
    ln -sf "$_pi_ext_src" "$_pi_ext_link"
fi
unset _pi_ext_link _pi_ext_src
