# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""``scan_leaked_credentials`` skip semantics: warn on failures, not on absence.

Regression for #86 — the scan used to ``except (OSError, TypeError):
continue`` silently, so an operator would see an empty result and
conclude "no leaks found" when a provider's credential file existed
but could not be read.  The fix surfaces a per-provider warning on
stderr while keeping the loop going so other providers still get
scanned.

That warning must not fire for a credential file that simply does not
exist: an agent that ships in the image but was never authenticated on
the host has nothing to leak, so absence is a definitive clean result
— not a skipped check.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_executor.credentials.vault_commands import scan_leaked_credentials


def _mount(provider: str, host_dir: str, credential_file: str) -> MagicMock:
    """Build a roster mount stub with the three fields the scan reads."""
    mount = MagicMock()
    mount.provider = provider
    mount.host_dir = host_dir
    mount.credential_file = credential_file
    return mount


class TestScanLeakedCredentialsSkipSemantics:
    """Missing credential files pass silently; unreadable ones warn."""

    def test_missing_mount_is_silent_and_scan_continues(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A provider whose credential file doesn't exist produces no warning."""
        roster = MagicMock()
        roster.mounts = [
            _mount("claude", "_claude-config", ".credentials.json"),
            _mount("glab", "_glab-config", "config.yml"),
        ]

        # Only the claude mount exists, with a leaked-looking file; the
        # glab agent ships in the image but was never set up on the host.
        (tmp_path / "_claude-config").mkdir()
        (tmp_path / "_claude-config" / ".credentials.json").write_text(
            '{"accessToken": "real-secret"}'
        )

        with (
            patch("terok_executor.roster.loader._shared_roster", return_value=roster),
            patch(
                "terok_executor.credentials.vault_commands._is_injected_credentials_file",
                return_value=False,
            ),
        ):
            leaked = scan_leaked_credentials(tmp_path)

        # The present provider still gets reported; the absent one is clean.
        assert leaked == [("claude", tmp_path / "_claude-config" / ".credentials.json")]
        assert capsys.readouterr().err == ""

    def test_unreadable_file_warns_and_scan_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A credential file that exists but can't be read warns, naming the provider."""
        roster = MagicMock()
        roster.mounts = [
            _mount("codex", "_codex-config", ".auth.json"),
            _mount("claude", "_claude-config", ".credentials.json"),
        ]

        for host_dir, name in (
            ("_codex-config", ".auth.json"),
            ("_claude-config", ".credentials.json"),
        ):
            (tmp_path / host_dir).mkdir()
            (tmp_path / host_dir / name).write_text('{"accessToken": "real-secret"}')

        real_lstat = Path.lstat

        def deny_codex(self: Path, *args: object, **kwargs: object) -> object:
            if self.name == ".auth.json":
                raise PermissionError(13, "Permission denied")
            return real_lstat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", deny_codex)

        with (
            patch("terok_executor.roster.loader._shared_roster", return_value=roster),
            patch(
                "terok_executor.credentials.vault_commands._is_injected_credentials_file",
                return_value=False,
            ),
        ):
            leaked = scan_leaked_credentials(tmp_path)

        # The readable provider still gets reported.
        assert leaked == [("claude", tmp_path / "_claude-config" / ".credentials.json")]
        # The unreadable one produced a warning naming it; the scan kept going.
        err = capsys.readouterr().err
        assert "codex" in err
        assert "vault" in err.lower()
