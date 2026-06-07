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

# No per-provider token bridging here: the env builder materializes every
# authenticated provider (vault-routed AND exposed) as TEROK_PROVIDER_<NAME>_*,
# and the vault-routing extension below registers each from those handles — so
# Pi reaches Anthropic (and the rest) without an env alias or a credential-file
# read in this snippet.

# Symlink terok's vault-routing extension into Pi's auto-discovery dir
# (~/.pi/agent/extensions/).  Pi only auto-discovers ``*.ts`` extensions, so the
# link must be ``.ts`` (the file is plain ESM, which is valid TypeScript).  We
# ship it at a system-wide path because the user's config dir is a bind mount we
# don't own.
_pi_ext_link="${HOME}/.pi/agent/extensions/terok-vault-routes.ts"
_pi_ext_src="/usr/local/share/terok/pi-extensions/vault-routes.ts"
if [ -f "$_pi_ext_src" ] && [ ! -L "$_pi_ext_link" ]; then
    mkdir -p "$(dirname "$_pi_ext_link")"
    ln -sf "$_pi_ext_src" "$_pi_ext_link"
fi
unset _pi_ext_link _pi_ext_src
