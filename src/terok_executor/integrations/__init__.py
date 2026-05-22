# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Cross-package adapters — one module per sibling wheel.

Every ``from terok_sandbox …`` import in ``terok_executor`` must go
through the adapter in this package; the ``import-linter``
``protected_modules`` contract on the sibling package root enforces
that.  Convention shared with terok-sandbox (which adapts terok-shield
and terok-clearance the same way) and terok-main (where the same
pattern lives at ``terok.lib.integrations.*``).
"""

#: Adapter sub-modules carry the public surface; the package root re-
#: exports nothing on its own.  Importers reference
#: ``terok_executor.integrations.sandbox.<symbol>`` explicitly.
__all__: list[str] = []
