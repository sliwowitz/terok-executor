# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-container runtime-provider scripts.

Covers the native-provider launcher (``terok-native-provider``) — the per-agent
override *deliveries* (codex ``-c`` flags, vibe ``VIBE_PROVIDERS`` env), endpoint
resolution, and argument parsing — plus the ``opencode-provider`` launcher's
provider resolution.  Both scripts are loaded by path because they ship into
task containers rather than being exposed as Python modules.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
_EXAMPLE_CHAT_BASE = f"{_LOOPBACK}/v1"
_BLABLADOR_MODEL = "alias-huge"
_EXAMPLE_MODEL = "example-chat"

_PROVIDER_ENV_NAMES = frozenset({"TEROK_PROVIDER", "TEROK_PI_PROVIDER"})
_PROVIDER_ENV_PREFIXES = ("TEROK_OC_", "TEROK_PROVIDER_")


def _run_pi_extension(
    env: dict[str, str], discovered_models: list[dict[str, object]] | None = None
) -> dict:
    """Execute the staged Pi extension under Node and return its registrations."""
    extension = Path(_scripts_pkg.__file__).parent / "pi-vault-routes.ts"
    discovery_payload = json.dumps({"data": discovered_models or []})
    source = f"""
        import registerProviders from {json.dumps(extension.as_uri())};
        const registrations = [];
        let fetchCalls = 0;
        globalThis.fetch = async () => {{
            fetchCalls += 1;
            return {{ ok: true, json: async () => ({discovery_payload}) }};
        }};
        await registerProviders({{
            registerProvider: (name, config) => registrations.push({{ name, config }}),
        }});
        process.stdout.write(JSON.stringify({{ registrations, fetchCalls }}));
    """
    try:
        completed = subprocess.run(  # noqa: S603
            ["node", "--experimental-strip-types", "--input-type=module", "-e", source],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": os.environ["PATH"], "NODE_NO_WARNINGS": "1"} | env,
        )
    except FileNotFoundError:
        pytest.skip("Node is unavailable for the staged Pi extension test")
    except subprocess.CalledProcessError as exc:
        if "experimental-strip-types" in exc.stderr and "bad option" in exc.stderr:
            pytest.skip("Node 22+ is required to execute the staged TypeScript directly")
        raise
    return json.loads(completed.stdout)


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


@pytest.fixture(autouse=True)
def _isolate_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep staged-script tests independent of the enclosing Terok task."""
    for name in tuple(os.environ):
        if name in _PROVIDER_ENV_NAMES or name.startswith(_PROVIDER_ENV_PREFIXES):
            monkeypatch.delenv(name)


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
        monkeypatch.setenv("TEROK_OC_BLABLADOR_PREFERRED_MODEL", _BLABLADOR_MODEL)
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

    def test_launch_is_plain_even_for_a_pinned_alias(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The launcher never applies git identity itself — that is the caller's job.

        Invoked as the bare ``blablador`` symlink (``argv[0]=blablador``), it must
        still exec ``opencode`` directly, with no ``bash`` identity shim wrapped
        around it, so it stays consistent with ``pi-provider`` /
        ``terok-native-provider``.  The generated ``blablador()`` wrapper and the
        ``*-acp`` env scripts own authorship on the paths terok actually drives.
        """
        monkeypatch.setattr(ocp.sys, "argv", ["blablador"])
        monkeypatch.setenv("TEROK_OC_BLABLADOR_BASE_URL", _LOOPBACK + "/v1")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_ENV_VAR_PREFIX", "BLABLADOR")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_PREFERRED_MODEL", _BLABLADOR_MODEL)
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_TOKEN", "tok")
        monkeypatch.setattr(ocp, "_fetch_models", lambda *a: None)
        monkeypatch.setattr(ocp, "_write_opencode_config", lambda *a: None)
        launched: dict[str, object] = {}
        monkeypatch.setattr(
            ocp.subprocess, "call", lambda cmd, env=None: launched.update(cmd=cmd) or 0
        )
        assert ocp.main() == 0
        assert launched["cmd"][0] == "opencode"
        assert "bash" not in launched["cmd"]

    def test_failed_discovery_validates_preferred_against_cached_models(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed refresh selects the cached fallback instead of restoring a stale default."""
        monkeypatch.setattr(ocp.sys, "argv", ["opencode-provider", "--provider", "blablador"])
        monkeypatch.setenv("TEROK_OC_BLABLADOR_BASE_URL", _LOOPBACK + "/v1")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_ENV_VAR_PREFIX", "BLABLADOR")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_PREFERRED_MODEL", _BLABLADOR_MODEL)
        monkeypatch.setenv("TEROK_OC_BLABLADOR_FALLBACK_MODEL", "alias-code")
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_TOKEN", "tok")
        existing = {
            "model": f"blablador/{_BLABLADOR_MODEL}",
            "provider": {
                "blablador": {
                    "options": {"baseURL": _LOOPBACK + "/v1", "apiKey": "tok"},
                    "models": {"alias-code": {"name": "Cached fallback"}},
                }
            },
        }
        monkeypatch.setattr(ocp, "_fetch_models", lambda *_args: None)
        monkeypatch.setattr(ocp, "_load_opencode_config", lambda *_args: existing)
        written: dict[str, object] = {}
        monkeypatch.setattr(
            ocp,
            "_write_opencode_config",
            lambda _config, content: written.update(content=content),
        )
        monkeypatch.setattr(ocp.subprocess, "call", lambda _cmd, env=None: 0)

        assert ocp.main() == 0
        assert written["content"]["model"] == "blablador/alias-code"
        assert set(written["content"]["provider"]["blablador"]["models"]) == {"alias-code"}

    def test_failed_discovery_persists_preferred_into_an_empty_cache(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A selected preferred model is written without mutating the comparison baseline."""
        monkeypatch.setattr(ocp.sys, "argv", ["opencode-provider", "--provider", "blablador"])
        monkeypatch.setenv("TEROK_OC_BLABLADOR_BASE_URL", _LOOPBACK + "/v1")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_DISPLAY_NAME", "Blablador")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_ENV_VAR_PREFIX", "BLABLADOR")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_PREFERRED_MODEL", _BLABLADOR_MODEL)
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_TOKEN", "tok")
        existing = {
            "model": f"blablador/{_BLABLADOR_MODEL}",
            "provider": {
                "blablador": {
                    "name": "Blablador",
                    "options": {"baseURL": _LOOPBACK + "/v1", "apiKey": "tok"},
                    "models": {},
                }
            },
        }
        monkeypatch.setattr(ocp, "_fetch_models", lambda *_args: None)
        monkeypatch.setattr(ocp, "_load_opencode_config", lambda *_args: existing)
        written: dict[str, object] = {}
        monkeypatch.setattr(
            ocp,
            "_write_opencode_config",
            lambda _config, content: written.update(content=content),
        )
        monkeypatch.setattr(ocp.subprocess, "call", lambda _cmd, env=None: 0)

        assert ocp.main() == 0
        assert set(written["content"]["provider"]["blablador"]["models"]) == {_BLABLADOR_MODEL}

    def test_list_models_uses_cache_after_failed_discovery(
        self,
        ocp: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--list-models`` prints cached model IDs when live discovery fails."""
        monkeypatch.setattr(
            ocp.sys,
            "argv",
            ["opencode-provider", "--provider", "blablador", "--list-models"],
        )
        monkeypatch.setenv("TEROK_OC_BLABLADOR_BASE_URL", _LOOPBACK + "/v1")
        monkeypatch.setenv("TEROK_OC_BLABLADOR_ENV_VAR_PREFIX", "BLABLADOR")
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_TOKEN", "tok")
        existing = {
            "provider": {
                "blablador": {
                    "models": {"cached-model": {"name": "Cached model"}},
                }
            },
        }
        monkeypatch.setattr(ocp, "_fetch_models", lambda *_args: None)
        monkeypatch.setattr(ocp, "_load_opencode_config", lambda *_args: existing)
        monkeypatch.setattr(
            ocp.subprocess,
            "call",
            lambda *_args, **_kwargs: pytest.fail("listing models must not launch OpenCode"),
        )

        assert ocp.main() == 0
        assert capsys.readouterr().out == "cached-model\n"


class TestProviderModelMetadata:
    """Provider-neutral model declarations project into each harness schema."""

    def test_opencode_skips_discovery_for_declared_context_only_model(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared context-only model avoids optional ``/models`` discovery.

        OpenCode cannot represent a context-only ``limit`` object, so it keeps
        the declared model and name while omitting the incomplete limit pair.
        """
        monkeypatch.setattr(
            ocp.sys,
            "argv",
            ["opencode-provider", "--provider", "example"],
        )
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_BASE_OPENAI_CHAT", _EXAMPLE_CHAT_BASE)
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_TOKEN", "tok")
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_LABEL", "Example")
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_DEFAULT_MODEL", _EXAMPLE_MODEL)
        monkeypatch.setenv(
            "TEROK_PROVIDER_EXAMPLE_MODELS",
            json.dumps(
                {
                    _EXAMPLE_MODEL: {
                        "name": "Example Chat",
                        "context_limit": 120_000,
                    }
                }
            ),
        )
        monkeypatch.setattr(
            ocp,
            "_fetch_models",
            lambda *_args: pytest.fail("declared models must suppress network discovery"),
        )
        monkeypatch.setattr(ocp, "_load_opencode_config", lambda *_args: None)
        written: dict[str, object] = {}
        monkeypatch.setattr(
            ocp,
            "_write_opencode_config",
            lambda _config, content: written.update(content=content),
        )
        monkeypatch.setattr(ocp.subprocess, "call", lambda _cmd, env=None: 0)

        assert ocp.main() == 0
        provider = written["content"]["provider"]["example"]
        assert provider["name"] == "Example"
        assert provider["models"] == {_EXAMPLE_MODEL: {"name": "Example Chat"}}

    def test_opencode_emits_complete_official_limit_pair(self, ocp: ModuleType) -> None:
        """Known context and output limits map to OpenCode's required pair."""
        models = {
            _EXAMPLE_MODEL: {
                "name": "Example Chat",
                "context_limit": 120_000,
                "output_limit": 8_192,
            }
        }
        assert ocp._opencode_models(models) == {
            _EXAMPLE_MODEL: {
                "name": "Example Chat",
                "limit": {"context": 120_000, "output": 8_192},
            }
        }

    def test_opencode_rejects_invalid_cached_limits(self, ocp: ModuleType) -> None:
        """Cached limits obey the same positive, non-boolean contract as live metadata."""
        config = {"name": "example"}
        existing = {
            "provider": {
                "example": {
                    "models": {
                        "valid": {"limit": {"context": 120_000, "output": 8_192}},
                        "boolean": {"limit": {"context": True, "output": False}},
                        "nonpositive": {"limit": {"context": 0, "output": -1}},
                        "noninteger": {"limit": {"context": 120_000.0, "output": "8192"}},
                    }
                }
            }
        }

        assert ocp._get_configured_models(config, existing) == {
            "valid": {"context_limit": 120_000, "output_limit": 8_192},
            "boolean": {},
            "nonpositive": {},
            "noninteger": {},
        }

    def test_pi_skips_discovery_and_preserves_declared_context(self) -> None:
        """Pi receives a declared 120k context and defaults only its unknown output."""
        result = _run_pi_extension(
            {
                "TEROK_PROVIDER_EXAMPLE_BASE_OPENAI_CHAT": _EXAMPLE_CHAT_BASE,
                "TEROK_PROVIDER_EXAMPLE_TOKEN": "tok",
                "TEROK_PROVIDER_EXAMPLE_LABEL": "Example",
                "TEROK_PROVIDER_EXAMPLE_DEFAULT_MODEL": _EXAMPLE_MODEL,
                "TEROK_PROVIDER_EXAMPLE_MODELS": json.dumps(
                    {
                        _EXAMPLE_MODEL: {
                            "name": "Example Chat",
                            "context_limit": 120_000,
                        }
                    }
                ),
            }
        )

        assert result["fetchCalls"] == 0
        assert result["registrations"] == [
            {
                "name": "example",
                "config": {
                    "baseUrl": _EXAMPLE_CHAT_BASE,
                    "api": "openai-completions",
                    "name": "Example",
                    "apiKey": "$TEROK_PROVIDER_EXAMPLE_TOKEN",
                    "models": [
                        {
                            "id": _EXAMPLE_MODEL,
                            "name": "Example Chat",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": 120_000,
                            "maxTokens": 4_096,
                        }
                    ],
                },
            }
        ]

    def test_pi_prefers_default_model_after_discovery(self) -> None:
        """The declared default leads a model list discovered from the endpoint."""
        result = _run_pi_extension(
            {
                "TEROK_PROVIDER_EXAMPLE_BASE_OPENAI_CHAT": _EXAMPLE_CHAT_BASE,
                "TEROK_PROVIDER_EXAMPLE_DEFAULT_MODEL": _EXAMPLE_MODEL,
            },
            discovered_models=[{"id": "another-model"}, {"id": _EXAMPLE_MODEL}],
        )

        models = result["registrations"][0]["config"]["models"]
        assert [model["id"] for model in models] == [_EXAMPLE_MODEL, "another-model"]

    def test_opencode_refreshes_a_changed_provider_label(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A label-only change updates the persistent OpenCode provider name."""
        monkeypatch.setattr(ocp.sys, "argv", ["opencode-provider", "--provider", "example"])
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_BASE_OPENAI_CHAT", _EXAMPLE_CHAT_BASE)
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_TOKEN", "tok")
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_LABEL", "New Example")
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_DEFAULT_MODEL", _EXAMPLE_MODEL)
        monkeypatch.setenv(
            "TEROK_PROVIDER_EXAMPLE_MODELS",
            json.dumps({_EXAMPLE_MODEL: {"name": "Example Chat"}}),
        )
        existing = {
            "model": f"example/{_EXAMPLE_MODEL}",
            "provider": {
                "example": {
                    "name": "Old Example",
                    "options": {"baseURL": _EXAMPLE_CHAT_BASE, "apiKey": "tok"},
                    "models": {_EXAMPLE_MODEL: {"name": "Example Chat"}},
                }
            },
        }
        monkeypatch.setattr(ocp, "_load_opencode_config", lambda *_args: existing)
        written: dict[str, object] = {}
        monkeypatch.setattr(
            ocp,
            "_write_opencode_config",
            lambda _config, content: written.update(content=content),
        )
        monkeypatch.setattr(ocp.subprocess, "call", lambda _cmd, env=None: 0)

        assert ocp.main() == 0
        assert written["content"]["provider"]["example"]["name"] == "New Example"

    @pytest.mark.parametrize("drift", ["npm", "schema", "permission"])
    def test_opencode_repairs_managed_config_drift(
        self, ocp: ModuleType, monkeypatch: pytest.MonkeyPatch, drift: str
    ) -> None:
        """Repair managed fields and preserve an existing user permission."""
        monkeypatch.setattr(ocp.sys, "argv", ["opencode-provider", "--provider", "example"])
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_BASE_OPENAI_CHAT", _EXAMPLE_CHAT_BASE)
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_TOKEN", "tok")
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_LABEL", "Example")
        monkeypatch.setenv("TEROK_PROVIDER_EXAMPLE_DEFAULT_MODEL", _EXAMPLE_MODEL)
        monkeypatch.setenv(
            "TEROK_PROVIDER_EXAMPLE_MODELS",
            json.dumps({_EXAMPLE_MODEL: {"name": "Example Chat"}}),
        )
        user_permission = {"edit": "ask"}
        existing = {
            "$schema": ocp._OPENCODE_SCHEMA,
            "model": f"example/{_EXAMPLE_MODEL}",
            "permission": user_permission,
            "provider": {
                "example": {
                    "npm": ocp._OPENAI_COMPATIBLE_NPM,
                    "name": "Example",
                    "options": {"baseURL": _EXAMPLE_CHAT_BASE, "apiKey": "tok"},
                    "models": {_EXAMPLE_MODEL: {"name": "Example Chat"}},
                }
            },
        }
        if drift == "npm":
            existing["provider"]["example"]["npm"] = "wrong-package"
        elif drift == "schema":
            existing["$schema"] = "wrong-schema"
        else:
            del existing["permission"]

        monkeypatch.setattr(ocp, "_load_opencode_config", lambda *_args: existing)
        written: dict[str, object] = {}
        monkeypatch.setattr(
            ocp,
            "_write_opencode_config",
            lambda _config, content: written.update(content=content),
        )
        monkeypatch.setattr(ocp.subprocess, "call", lambda _cmd, env=None: 0)

        assert ocp.main() == 0
        content = written["content"]
        assert content["$schema"] == ocp._OPENCODE_SCHEMA
        assert content["provider"]["example"]["npm"] == ocp._OPENAI_COMPATIBLE_NPM
        expected_permission = {"*": "allow"} if drift == "permission" else user_permission
        assert content["permission"] == expected_permission

    def test_pi_rejects_array_model_metadata(self) -> None:
        """Array metadata does not register a declared model or suppress discovery."""
        result = _run_pi_extension(
            {
                "TEROK_PROVIDER_EXAMPLE_BASE_OPENAI_CHAT": _EXAMPLE_CHAT_BASE,
                "TEROK_PROVIDER_EXAMPLE_TOKEN": "tok",
                "TEROK_PROVIDER_EXAMPLE_MODELS": json.dumps({_EXAMPLE_MODEL: []}),
            }
        )

        assert result["fetchCalls"] == 1
        assert result["registrations"] == [
            {
                "name": "example",
                "config": {
                    "baseUrl": _EXAMPLE_CHAT_BASE,
                    "api": "openai-completions",
                    "apiKey": "$TEROK_PROVIDER_EXAMPLE_TOKEN",
                },
            }
        ]


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
        monkeypatch.setenv("TEROK_OC_BLABLADOR_PREFERRED_MODEL", _BLABLADOR_MODEL)
        monkeypatch.setenv("TEROK_PROVIDER_BLABLADOR_TOKEN", "tok")
        monkeypatch.setattr(ocp, "_fetch_models", lambda *a: None)
        monkeypatch.setattr(ocp, "_write_opencode_config", lambda *a: None)
        monkeypatch.setattr(ocp.os.path, "exists", lambda _p: False)
        monkeypatch.setattr(ocp.subprocess, "call", lambda cmd, env=None: 0)
        assert ocp.main() == 0
        assert "Updating the model list from Helmholtz Blablador" in capsys.readouterr().err
