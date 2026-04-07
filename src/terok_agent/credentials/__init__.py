# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Authentication, credential extraction, and credential proxy.

Delegates to :mod:`.auth` for auth provider registry and container-based
auth flows, :mod:`.extractors` for per-provider credential file parsing,
:mod:`.proxy_commands` for proxy CLI lifecycle, and :mod:`.proxy_config`
for post-auth config file patching.
"""
