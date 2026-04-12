# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container lifecycle — image building, environment assembly, agent launch.

Delegates to :mod:`.build` for Dockerfile rendering and ``podman build``,
:mod:`.env` for container environment and volume assembly, and
:mod:`.runner` for the high-level agent runner that composes all three.
"""
