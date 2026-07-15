# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-container runtime-provider scripts.

Covers the native-provider launcher (``terok-native-provider``) — the per-agent
override *deliveries* (codex ``-c`` flags, vibe ``VIBE_PROVIDERS`` env), endpoint
resolution, and argument parsing — plus the pinned-alias git-identity
restoration in the ``opencode-provider`` launcher.  Both scripts are loaded by
path because they ship into task containers rather than being exposed as Python
modules.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

import terok_executor.resources.scripts as _scripts_pkg

# Container-side loopback the env builder would materialize into the
# TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL> handles; realistic values keep the
# fixtures honest without hard-coding a production address.
_LOOPBACK = "http://localhost:9419"
_OPENAI_RESPONSES_BASE = f"{_LOOPBACK}/v1"
_OPENAI_CHAT_BASE = f"{_LOOPBACK}/api/v1"


def _load_script(filename: str, module_name: str) -> ModuleType:
    """Load a staged container script as an importable module.

    Registered in ``sys.modules`` so a frozen dataclass resolves its own module
    the way direct execution (as ``__main__``) would.
    """
    script_path = Path(_scripts_pkg.__file__).parent / filename
    loader = SourceFileLoader(module_name, str(script_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def np() -> ModuleType:
    """The loaded native-provider launcher module."""
    return _load_script("terok-native-provider", "terok_native_provider")


@pytest.fixture(scope="module")
def ocp() -> ModuleType:
    """The loaded opencode-provider launcher module."""
    return _load_script("opencode-provider", "terok_opencode_provider")


@pytest.fixture(scope="module")
def pp() -> ModuleType:
    """The loaded pi-provider launcher module."""
    return _load_script("pi-provider", "terok_pi_provider")


def _c_settings(args: list[str]) -> dict[str, str]:
    """Collapse a ``-c key=value -c …`` list into a ``{key: value}`` map."""
    assert all(flag == "-c" for flag in args[::2]), args
    return dict(pair.split("=", 1) for pair in args[1::2])


class TestCodexDelivery:
    """``_deliver_codex`` renders ``-c`` overrides; no file is written."""

    def test_defines_and_selects_a_custom_provider(self, np: ModuleType) -> None:
        """The ``-c`` flags add a ``terok-<name>`` provider and select it."""
        args, env = np._deliver_codex(
            "openai", _OPENAI_RESPONSES_BASE, "TEROK_PROVIDER_OPENAI_TOKEN"
        )
        assert env == {}
        settings = _c_settings(args)
        assert settings["model_provider"] == '"terok-openai"'
        assert settings["model_providers.terok-openai.base_url"] == f'"{_OPENAI_RESPONSES_BASE}"'
        # env_key names the existing phantom-token var — the secret never lands on argv.
        assert settings["model_providers.terok-openai.env_key"] == '"TEROK_PROVIDER_OPENAI_TOKEN"'
        assert settings["model_providers.terok-openai.wire_api"] == '"responses"'
        assert settings["model_providers.terok-openai.name"] == '"openai"'

    def test_values_are_toml_quoted(self, np: ModuleType) -> None:
        """Values are TOML-quoted strings (codex parses ``-c`` values as TOML)."""
        args, _ = np._deliver_codex("openai", _OPENAI_RESPONSES_BASE, "TEROK_PROVIDER_OPENAI_TOKEN")
        assert all(
            value.startswith('"') and value.endswith('"') for value in _c_settings(args).values()
        )


class TestVibeDelivery:
    """``_deliver_vibe`` renders a ``VIBE_PROVIDERS`` env entry; no file is written."""

    def test_repoints_active_provider_via_env(self, np: ModuleType) -> None:
        """A single JSON providers entry aims the active provider through the vault."""
        args, env = np._deliver_vibe(
            "openrouter", _OPENAI_CHAT_BASE, "TEROK_PROVIDER_OPENROUTER_TOKEN"
        )
        assert args == []
        entry = json.loads(env["VIBE_PROVIDERS"])
        assert entry == [
            {
                "name": "mistral",  # active-provider name is preserved
                "api_base": _OPENAI_CHAT_BASE,
                "api_key_env_var": "TEROK_PROVIDER_OPENROUTER_TOKEN",
            }
        ]


class TestOverride:
    """``_override`` resolves the materialized handle and renders the delivery."""

    def test_renders_delivery_for_served_protocol(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A served protocol yields the agent's override (here, codex ``-c`` flags)."""
        monkeypatch.setenv("TEROK_PROVIDER_OPENAI_BASE_OPENAI_RESPONSES", _OPENAI_RESPONSES_BASE)
        args, env = np._override(np._NATIVE_AGENTS["codex"], "openai")
        assert _c_settings(args)["model_provider"] == '"terok-openai"'
        assert env == {}

    def test_empty_override_when_protocol_unserved(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unserved protocol yields no override — the agent keeps its default."""
        monkeypatch.delenv("TEROK_PROVIDER_OPENROUTER_BASE_OPENAI_RESPONSES", raising=False)
        assert np._override(np._NATIVE_AGENTS["codex"], "openrouter") == ([], {})


class TestArgumentResolution:
    """Provider selection and agent naming from the invocation."""

    def test_invoked_agent_strips_suffix(self, np: ModuleType) -> None:
        """``codex-provider`` resolves to the ``codex`` agent."""
        assert np._invoked_agent("/usr/local/bin/codex-provider") == "codex"

    def test_leading_provider_flag_wins(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``--provider`` is consumed and overrides the env default."""
        monkeypatch.setenv("TEROK_PROVIDER", "mistral")
        provider, rest = np._split_provider_flag(["--provider", "openrouter", "exec", "hi"])
        assert provider == "openrouter"
        assert rest == ["exec", "hi"]

    def test_env_fallback_when_flag_absent(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the flag, ``TEROK_PROVIDER`` selects the provider."""
        monkeypatch.setenv("TEROK_PROVIDER", "mistral")
        provider, rest = np._split_provider_flag(["exec", "hi"])
        assert provider == "mistral"
        assert rest == ["exec", "hi"]


class TestNoSelection:
    """With no provider selected the launcher applies no override.

    The agent then runs on its config_patch'd default endpoint (which already
    points at the vault); the wrapper, in fact, never even invokes the launcher
    in that case.  The launcher only re-points to an explicitly selected
    *non-default* provider.
    """

    def test_registry_mirrors_roster_binding(self, np: ModuleType) -> None:
        """The hardcoded binary/protocol must not drift from the roster.

        The launcher runs in task containers where terok isn't importable, so it
        hardcodes these instead of reading the roster; this guards that copy.
        """
        from terok_executor.provider.providers import AGENTS

        for name, native in np._NATIVE_AGENTS.items():
            agent = AGENTS[name]
            assert native.binary == agent.binary, name
            assert native.protocol == agent.protocol, name

    def test_main_applies_no_override_without_selection(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No flag and no ``TEROK_PROVIDER`` → codex execs bare, no ``-c`` overrides."""
        monkeypatch.delenv("TEROK_PROVIDER", raising=False)
        launched: dict[str, object] = {}
        monkeypatch.setattr(
            np,
            "_exec",
            lambda binary, args, env: launched.update(binary=binary, args=args, env=env) or 0,
        )
        np.main(["/usr/local/bin/codex-provider", "exec", "hi"])
        assert launched["binary"] == "codex"
        assert launched["args"] == ["exec", "hi"]  # forwarded verbatim, nothing prepended
        assert launched["env"] == {}


class TestEmit:
    """``--emit`` prints an agent's override args for an external (ACP) launcher."""

    def test_emits_codex_override_when_provider_selected(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--emit codex`` with a selected provider prints the ``-c`` flags, NUL-joined."""
        monkeypatch.setenv("TEROK_PROVIDER", "openai")
        monkeypatch.setenv("TEROK_PROVIDER_OPENAI_BASE_OPENAI_RESPONSES", _OPENAI_RESPONSES_BASE)
        assert np._emit(["codex"]) == 0
        settings = _c_settings(capsys.readouterr().out.split("\0"))
        assert settings["model_provider"] == '"terok-openai"'
        assert settings["model_providers.terok-openai.env_key"] == '"TEROK_PROVIDER_OPENAI_TOKEN"'

    def test_emits_nothing_without_selection(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No ``TEROK_PROVIDER`` → nothing emitted; the adapter keeps its config_patch default."""
        monkeypatch.delenv("TEROK_PROVIDER", raising=False)
        assert np._emit(["codex"]) == 0
        assert capsys.readouterr().out == ""

    def test_emits_nothing_when_provider_unserved(
        self, np: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unserved selection yields no flags, so the adapter keeps its default."""
        monkeypatch.setenv("TEROK_PROVIDER", "openrouter")  # serves openai-chat, not -responses
        monkeypatch.delenv("TEROK_PROVIDER_OPENROUTER_BASE_OPENAI_RESPONSES", raising=False)
        assert np._emit(["codex"]) == 0
        assert capsys.readouterr().out == ""

    def test_unknown_agent_errors(self, np: ModuleType) -> None:
        """``--emit`` of an unregistered agent fails with the launcher's usage hint."""
        with pytest.raises(SystemExit):
            np._emit(["bogus"])


class TestPinnedAliasGitIdentity:
    """``opencode-provider`` restores git authorship for symlinked pinned aliases."""

    def test_applies_identity_via_shell_helper(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the helper is present, the launch runs through it with the alias identity."""
        calls: list[list[str]] = []
        monkeypatch.setattr(ocp.os.path, "exists", lambda _p: True)
        monkeypatch.setattr(ocp.subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)
        ocp._launch_with_git_identity("blablador", ["opencode", "run"], {})
        cmd = calls[0]
        # Provider name doubles as the git display name (blablador → Blablador).
        assert cmd[0] == "bash"
        assert "Blablador" in cmd
        assert "noreply@blablador.localhost" in cmd
        assert cmd[-2:] == ["opencode", "run"]

    def test_falls_back_to_plain_launch_without_helper(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the helper (standalone image), the binary launches directly."""
        calls: list[list[str]] = []
        monkeypatch.setattr(ocp.os.path, "exists", lambda _p: False)
        monkeypatch.setattr(ocp.subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)
        ocp._launch_with_git_identity("blablador", ["opencode", "run"], {})
        assert calls[0] == ["opencode", "run"]


class TestOpencodeProviderSelection:
    """``opencode-provider --provider X`` must resolve X, not the argv[0] name.

    Regression: the wrapper invokes the launcher as ``opencode-provider --provider
    blablador``, but ``main`` resolved the provider from ``argv[0]`` *before*
    reading ``--provider`` — so it died with ``Unknown provider: opencode-provider``.
    The pinned-alias symlink (``argv[0]=blablador``) hid the bug.
    """

    def test_provider_flag_resolves_over_argv0(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invoked as ``opencode-provider --provider blablador`` → resolves blablador."""
        monkeypatch.setattr(ocp.sys, "argv", ["opencode-provider", "--provider", "blablador"])
        monkeypatch.setenv("TEROK_OC_BLABLADOR_BASE_URL", _LOOPBACK + "/v1")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_ENV_VAR_PREFIX", "BLABLADOR")
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_TOKEN", "tok")
        monkeypatch.setattr(ocp, "_fetch_models", lambda *a: None)
        monkeypatch.setattr(ocp, "_write_opencode_config", lambda *a: None)
        launched: dict[str, object] = {}
        monkeypatch.setattr(
            ocp.subprocess, "call", lambda cmd, env=None: launched.update(cmd=cmd, env=env) or 0
        )
        assert ocp.main() == 0
        # Reaching the launch at all means resolution didn't raise on the argv[0]
        # name; the config it picked is blablador's, not "opencode-provider".
        assert "blablador" in launched["env"]["OPENCODE_CONFIG"]


class TestPiProvider:
    """``pi-provider`` scopes Pi by provider: explicit ``--provider`` filters the
    picker, a container default opens Pi on it, unusable explicit picks are
    rejected with a clear error.  It also injects the per-task instructions."""

    @pytest.fixture(autouse=True)
    def _no_instructions(
        self, pp: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default to *no* instructions file so argv assertions ignore the host.

        ``main`` prepends ``--append-system-prompt`` only when the file exists;
        pointing the constant at an absent path keeps the provider-scoping
        assertions independent of whatever lives at the real container path.
        The injection itself is covered by its own test, which overrides this.
        """
        monkeypatch.setattr(pp, "_INSTRUCTIONS_PATH", str(tmp_path / "absent.md"))

    def test_instruction_args_passes_existing_path(self, pp: ModuleType, tmp_path: Path) -> None:
        """An existing instructions file is handed to ``--append-system-prompt`` verbatim."""
        instr = tmp_path / "instructions.md"
        instr.write_text("be terse", encoding="utf-8")
        assert pp.instruction_args(str(instr)) == ["--append-system-prompt", str(instr)]

    def test_instruction_args_empty_when_absent(self, pp: ModuleType, tmp_path: Path) -> None:
        """A missing file yields no args — never injected as literal prompt text."""
        assert pp.instruction_args(str(tmp_path / "nope.md")) == []

    def test_main_prepends_instructions_when_present(
        self, pp: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A present instructions file rides ahead of the provider flag and prompt."""
        instr = tmp_path / "instructions.md"
        instr.write_text("be terse", encoding="utf-8")
        monkeypatch.setattr(pp, "_INSTRUCTIONS_PATH", str(instr))
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_BASE_OPENAI_CHAT", "http://x")
        captured: dict[str, object] = {}
        monkeypatch.setattr(pp.os, "execvpe", lambda f, a, e: captured.update(argv=a))
        pp.main(["--provider", "blablador", "hi"])
        assert captured["argv"] == [
            "pi",
            "--append-system-prompt",
            str(instr),
            "--provider",
            "blablador",
            "hi",
        ]

    def test_peel_explicit_provider(self, pp: ModuleType) -> None:
        """An explicit ``--provider`` (space or ``=`` form) is peeked, not consumed."""
        assert pp.peel_provider(["--provider", "blablador", "hi"]) == "blablador"
        assert pp.peel_provider(["--provider=openrouter"]) == "openrouter"
        assert pp.peel_provider(["hi"]) is None

    def test_usable_set_is_just_the_materialized_handles(self, pp: ModuleType) -> None:
        """Usable = providers with a materialized ``_BASE_`` handle.

        Exposed providers are now materialized the same way as vault-routed ones,
        so there is no ANTHROPIC_OAUTH_TOKEN special-case — a token without a
        handle does not count as usable.
        """
        env = {
            "TEROK_PROVIDER_BLABLADOR_BASE_OPENAI_CHAT": "http://x",
            "TEROK_PROVIDER_ANTHROPIC_BASE_ANTHROPIC_MESSAGES": "https://api.anthropic.com",
            "ANTHROPIC_OAUTH_TOKEN": "sk-ant-oat-x",  # bare token — not a usability signal
            "PATH": "/usr/bin",
        }
        assert pp.usable_providers(env) == {"blablador", "anthropic"}  # both from handles

    def test_explicit_scopes_via_env_and_forwards_flag(
        self, pp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--provider X`` (usable) exports TEROK_PI_PROVIDER and forwards argv intact."""
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_BASE_OPENAI_CHAT", "http://x")
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            pp.os, "execvpe", lambda f, a, e: captured.update(file=f, argv=a, env=e)
        )
        pp.main(["--provider", "blablador", "hi"])
        assert captured["argv"] == ["pi", "--provider", "blablador", "hi"]
        assert captured["env"]["TEROK_PI_PROVIDER"] == "blablador"  # type: ignore[index]

    def test_default_opens_on_it_without_scoping(
        self, pp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No flag + a usable container default → prepend --provider, no scope env."""
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_BASE_OPENAI_CHAT", "http://x")
        monkeypatch.setenv("TEROK_PROVIDER", "blablador")
        captured: dict[str, object] = {}
        monkeypatch.setattr(pp.os, "execvpe", lambda f, a, e: captured.update(argv=a, env=e))
        pp.main(["hi"])
        assert captured["argv"] == ["pi", "--provider", "blablador", "hi"]
        assert "TEROK_PI_PROVIDER" not in captured["env"]  # type: ignore[operator]

    def test_rejects_unusable_explicit_provider(
        self, pp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit provider that isn't usable here is rejected with a clear error."""
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_BASE_OPENAI_CHAT", "http://x")
        with pytest.raises(SystemExit, match="not available to pi"):
            pp.main(["--provider", "bogus"])

    def test_ignores_unusable_default(
        self, pp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing/unusable container default is ignored — bare pi still launches."""
        monkeypatch.setenv("TEROK_PROVIDER", "bogus")  # not materialized
        captured: dict[str, object] = {}
        monkeypatch.setattr(pp.os, "execvpe", lambda f, a, e: captured.update(argv=a))
        pp.main(["hi"])
        assert captured["argv"] == ["pi", "hi"]


class TestOpencodeModelFetchFeedback:
    """The model-list refresh announces itself and fails with a reason.

    Regression: the refresh can stall for its full 30s timeout (e.g. a
    half-dead vault bridge that accepts but never answers), and the
    socket-level ``TimeoutError`` escaped the ``URLError``-only handler —
    a silent freeze followed by a raw traceback.
    """

    def test_timeout_returns_none_with_reason(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """A socket-level timeout → ``None`` plus a stderr warning, no traceback."""

        def _hang(*_a, **_k):
            raise TimeoutError("timed out")

        monkeypatch.setattr(ocp.request, "urlopen", _hang)
        assert ocp._fetch_models(_LOOPBACK + "/v1", "tok") is None
        err = capsys.readouterr().err
        assert "model-list refresh" in err
        assert "timed out" in err

    def test_main_announces_the_refresh(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """``main`` prints the update notice before fetching — no silent freeze."""
        monkeypatch.setattr(ocp.sys, "argv", ["blablador"])
        monkeypatch.setenv("TEROK_OC_BLABLADOR_BASE_URL", _LOOPBACK + "/v1")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_DISPLAY_NAME", "Helmholtz Blablador")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_ENV_VAR_PREFIX", "BLABLADOR")
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_TOKEN", "tok")
        monkeypatch.setattr(ocp, "_fetch_models", lambda *a: None)
        monkeypatch.setattr(ocp, "_write_opencode_config", lambda *a: None)
        monkeypatch.setattr(ocp.os.path, "exists", lambda _p: False)
        monkeypatch.setattr(ocp.subprocess, "call", lambda cmd, env=None: 0)
        assert ocp.main() == 0
        assert "Updating the model list from Helmholtz Blablador" in capsys.readouterr().err
