# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Re-exports the executor-only timezone helper.

Standalone — no terok-executor domain imports, safe to use from any layer.
Cross-package helpers (``ensure_dir``, ``podman_userns_args``, the round-trip
YAML facade in [`terok_util.yaml`][terok_util.yaml], ...) live in the shared
[`terok_util`][terok_util] package and are imported from there directly.
"""

from ._timezone import detect_host_timezone

__all__ = [
    "detect_host_timezone",
]
