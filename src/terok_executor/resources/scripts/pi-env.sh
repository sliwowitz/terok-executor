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

# Symlink terok's Pi extensions into Pi's auto-discovery dir
# (~/.pi/agent/extensions/).  Pi only auto-discovers ``*.ts`` extensions, so the
# links must be ``.ts`` (the files are plain ESM, which is valid TypeScript).  We
# ship them at a system-wide path because the user's config dir is a bind mount
# we don't own.  Linking the whole directory keeps this snippet agnostic to how
# many extensions terok ships (vault-routes, git-identity, …).
_pi_ext_dir="/usr/local/share/terok/pi-extensions"
_pi_ext_link_dir="${HOME}/.pi/agent/extensions"
if [ -d "$_pi_ext_dir" ]; then
    mkdir -p "$_pi_ext_link_dir"
    for _pi_ext_src in "$_pi_ext_dir"/*.ts; do
        [ -f "$_pi_ext_src" ] || continue
        _pi_ext_link="${_pi_ext_link_dir}/terok-$(basename "$_pi_ext_src")"
        [ -L "$_pi_ext_link" ] || ln -sf "$_pi_ext_src" "$_pi_ext_link"
    done
fi
unset _pi_ext_dir _pi_ext_link_dir _pi_ext_src _pi_ext_link
