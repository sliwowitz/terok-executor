# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Writable-path resolution for the executor's slice of the global ``config.yml``.

The read/write accessors for the ``image:`` section live on
[`ExecutorConfigView`][terok_executor.config_schema.ExecutorConfigView]
— this module holds only the path-resolution helper they call when
they need to write.
"""

from __future__ import annotations

import os
from pathlib import Path

from terok_util import namespace_config_dir

__all__ = ["writable_config_path"]


def writable_config_path() -> Path:
    """Return the path the next config write should target.

    Honours ``TEROK_CONFIG_FILE`` when set; otherwise the user-scope
    file under [`namespace_config_dir`][terok_util.paths.namespace_config_dir].
    """
    env = os.getenv("TEROK_CONFIG_FILE")
    if env:
        return Path(env).expanduser()
    return namespace_config_dir() / "config.yml"
