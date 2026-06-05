# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Regression for #86 — ``scan_leaked_credentials`` warns when it skips a provider.

The scan used to ``except (OSError, TypeError): continue`` silently, so
an operator running it on a not-yet-mounted shared dir would see an
empty result and conclude "no leaks found" — when in fact zero
providers had been checked.  The fix surfaces a per-provider warning
on stderr while keeping the loop going so other providers still get
scanned.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_executor.credentials.vault_commands import scan_leaked_credentials


class TestScanLeakedCredentialsWarnsOnSkip:
    """A non-existent provider dir surfaces a warning and doesn't break the scan."""

    def test_missing_mount_warns_and_continues(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A provider whose mount dir is missing yields a warning naming it."""
        # Two mounts: one has a credential file present, one's dir is missing.
        roster = MagicMock()
        mount_present = MagicMock()
        mount_present.provider = "claude"
        mount_present.host_dir = "_claude-config"
        mount_present.credential_file = ".credentials.json"
        mount_missing = MagicMock()
        mount_missing.provider = "codex"
        mount_missing.host_dir = "_codex-config"
        mount_missing.credential_file = ".auth.json"
        roster.mounts = [mount_present, mount_missing]

        # Set up only the claude mount with a leaked-looking file.
        (tmp_path / "_claude-config").mkdir()
        (tmp_path / "_claude-config" / ".credentials.json").write_text(
            '{"accessToken": "real-secret"}'
        )
        # The codex provider's mount dir is intentionally absent → lstat raises.

        with (
            patch("terok_executor.roster.loader._shared_roster", return_value=roster),
            patch(
                "terok_executor.credentials.vault_commands._is_injected_credentials_file",
                return_value=False,
            ),
            patch(
                "terok_executor.credentials.vault_commands._is_injected_codex_auth_file",
                return_value=False,
            ),
        ):
            leaked = scan_leaked_credentials(tmp_path)

        # The present provider still gets reported.
        assert ("claude", tmp_path / "_claude-config" / ".credentials.json") in leaked
        # The missing one produced a warning naming it; the scan kept going.
        err = capsys.readouterr().err
        assert "codex" in err
        assert "vault" in err.lower()
