# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for headless provider registry and dispatch functions."""

from dataclasses import FrozenInstanceError

import pytest

from terok_executor.provider.agents import _generate_claude_wrapper
from terok_executor.provider.config import resolve_provider_value
from terok_executor.provider.headless import (
    apply_provider_config,
    build_headless_command,
)
from terok_executor.provider.providers import (
    AGENT_PROVIDERS,
    PROVIDER_NAMES,
    collect_all_auto_approve_env,
    get_provider,
)
from terok_executor.provider.wrappers import (
    INITIAL_PROMPT_CONSUMED_PATH,
    INITIAL_PROMPT_PATH,
    generate_agent_wrapper,
    generate_all_wrappers,
)
from tests.constants import CONTAINER_TEROK_DIR


def _provider_wrapper(
    name: str,
    *,
    has_agents: bool = False,
) -> str:
    """Generate a wrapper for a provider under test."""
    kwargs = {"claude_wrapper_fn": _generate_claude_wrapper} if name == "claude" else {}
    return generate_agent_wrapper(
        AGENT_PROVIDERS[name],
        has_agents=has_agents,
        **kwargs,
    )


def _all_wrappers(*, has_agents: bool = False) -> str:
    """Generate the combined multi-provider wrapper file."""
    return generate_all_wrappers(
        has_agents=has_agents,
        claude_wrapper_fn=_generate_claude_wrapper,
    )


class TestAgentProviderRegistry:
    """Tests for the AGENT_PROVIDERS registry."""

    def test_registry_contains_expected_providers(self) -> None:
        """Registry contains exactly the expected set of bundled providers."""
        expected = {
            "claude",
            "codex",
            "copilot",
            "vibe",
            "blablador",
            "opencode",
            "kisski",
            "openrouter",
        }
        assert set(AGENT_PROVIDERS.keys()) == expected

    def test_provider_names_tuple(self) -> None:
        """PROVIDER_NAMES is a tuple matching registry keys."""
        assert isinstance(PROVIDER_NAMES, tuple)
        assert set(PROVIDER_NAMES) == set(AGENT_PROVIDERS.keys())

    def test_providers_are_frozen(self) -> None:
        """AgentProvider instances are immutable."""
        provider = AGENT_PROVIDERS["claude"]
        with pytest.raises(FrozenInstanceError):
            provider.name = "changed"  # type: ignore[misc]


class TestGetProvider:
    """Tests for get_provider() resolution."""

    def test_explicit_name(self) -> None:
        """Explicit provider name resolves correctly."""
        p = get_provider("codex")
        assert p.name == "codex"

    def test_none_falls_back_to_default(self) -> None:
        """None name uses default_agent."""
        p = get_provider(None, default_agent="copilot")
        assert p.name == "copilot"

    def test_none_falls_back_to_claude(self) -> None:
        """None name with no default resolves to claude."""
        p = get_provider(None, default_agent=None)
        assert p.name == "claude"

    def test_invalid_name_raises_system_exit(self) -> None:
        """Unknown provider name raises SystemExit."""
        with pytest.raises(SystemExit) as ctx:
            get_provider("nonexistent")
        assert "nonexistent" in str(ctx.value)


class TestBuildHeadlessCommand:
    """Tests for build_headless_command() per provider."""

    def test_all_commands_start_with_init(self) -> None:
        """All provider commands start with init-ssh-and-repo.sh."""
        for _name, p in AGENT_PROVIDERS.items():
            cmd = build_headless_command(p, timeout=1800)
            assert cmd.startswith("init-ssh-and-repo.sh")


class TestGenerateAgentWrapper:
    """Tests for generate_agent_wrapper() per provider."""

    def test_claude_wrapper_uses_claude_function(self) -> None:
        """Claude wrapper defines a claude() function with --add-dir."""
        wrapper = _provider_wrapper("claude")
        assert "claude()" in wrapper
        assert "--add-dir" in wrapper

    def test_claude_wrapper_requires_fn(self) -> None:
        """Claude provider without claude_wrapper_fn raises ValueError."""
        p = AGENT_PROVIDERS["claude"]
        with pytest.raises(ValueError):
            generate_agent_wrapper(p, has_agents=False)

    def test_generic_wrapper_has_timeout_support(self) -> None:
        """All non-Claude wrappers support --terok-timeout."""
        for name in AGENT_PROVIDERS:
            if name == "claude":
                continue
            wrapper = _provider_wrapper(name)
            assert "--terok-timeout" in wrapper, f"{name} missing timeout support"

    def test_session_resume_uses_explicit_id(self) -> None:
        """Providers with session_file use --session/--resume with explicit ID."""
        for name in ("vibe", "opencode", "blablador", "kisski", "openrouter"):
            p = AGENT_PROVIDERS[name]
            wrapper = _provider_wrapper(name)
            assert p.resume_flag in wrapper, f"{name} missing resume flag"
            assert f"cat {CONTAINER_TEROK_DIR}/{p.session_file}" in wrapper

    def test_vibe_wrapper_has_lazy_model_sync(self) -> None:
        """Vibe wrapper includes lazy Mistral model sync with mtime check."""
        wrapper = _provider_wrapper("vibe")
        assert "vibe-model-sync" in wrapper
        assert "mmin +1440" in wrapper

    def test_non_vibe_wrappers_lack_model_sync(self) -> None:
        """Only vibe gets the model sync block."""
        for name in AGENT_PROVIDERS:
            if name == "vibe":
                continue
            wrapper = _provider_wrapper(name)
            assert "vibe-model-sync" not in wrapper, f"{name} should not have model sync"

    def test_vibe_wrapper_forces_yolo_unconditionally(self) -> None:
        """``vibe()`` exports VIBE_BYPASS_TOOL_PERMISSIONS inside its subshells.

        ``container/env.py:339-341`` only applies the roster-declared
        ``auto_approve_env`` when ``spec.unrestricted`` is True — which
        excludes interactive ``vibe`` invocations.  The wrapper unblocks
        every tool prompt by setting the real Vibe field
        (``VibeConfig.bypass_tool_permissions``) on the function's own
        subshell, so the user never sees an approval popup in a terok
        task.  Pinned per #47 follow-up after we discovered the env was
        absent in interactive containers.
        """
        wrapper = _provider_wrapper("vibe")
        assert "export VIBE_BYPASS_TOOL_PERMISSIONS=true" in wrapper

    def test_vibe_wrapper_injects_per_task_system_prompt(self) -> None:
        """``vibe()`` writes a per-task prompt + exports VIBE_SYSTEM_PROMPT_ID.

        Pinned per #47 follow-up — the CLI path needs the same prompt-id
        injection the ACP wrapper does.  Per-task naming keeps parallel
        containers from clobbering each other in the shared ~/.vibe
        prompts dir; the EXIT trap removes the file when the subshell
        exits so finished tasks don't leave cruft.
        """
        wrapper = _provider_wrapper("vibe")
        assert 'cp "/home/dev/.terok/instructions.md"' in wrapper
        assert 'export VIBE_SYSTEM_PROMPT_ID="${_vibe_prompt_id}"' in wrapper
        # Per-task id and EXIT-cleanup trap.
        assert '_vibe_prompt_id="terok-task-${TASK_ID}"' in wrapper
        assert "trap 'rm -f \"${_vibe_prompt_file}\"' EXIT" in wrapper

    def test_vibe_wrapper_trusts_workspace(self) -> None:
        """``vibe()`` adds /workspace to ~/.vibe/trusted_folders.toml.

        Without the trust marker, ``HarnessFilesManager.load_project_docs``
        returns early (``trusted_workdir`` is None) and the project's
        ``AGENTS.md`` chain never composes into the prompt.  Vibe's CLI
        has a ``--trust`` flag for this; ACP and our wrapper have no
        equivalent so we write the TOML file directly with a flock-
        guarded helper.
        """
        wrapper = _provider_wrapper("vibe")
        assert "_terok_trust_workspace_for_vibe" in wrapper
        assert '_terok_trust_workspace_for_vibe "/workspace"' in wrapper

    def test_trust_workspace_helper_is_defined_once(self) -> None:
        """The shared ``_terok_trust_workspace_for_vibe`` lives at the top of the file.

        Defined in ``_TRUST_WORKSPACE_FN`` and emitted alongside
        ``_RESUME_FALLBACK_FN`` by ``generate_all_wrappers`` so multiple
        provider sections (today: just vibe; future: anyone else that
        needs trust) reference one helper instead of inlining the
        Python + flock block per wrapper.
        """
        from terok_executor.provider.wrappers import generate_all_wrappers

        all_wrappers = generate_all_wrappers(has_agents=True, claude_wrapper_fn=lambda _c: "")
        # Definition appears exactly once (no duplicate inlines)…
        assert all_wrappers.count("_terok_trust_workspace_for_vibe()") == 1
        # …and the flock guard is wired up.
        assert "flock -x 200" in all_wrappers

    def test_non_vibe_wrappers_lack_vibe_setup_block(self) -> None:
        """Only vibe gets the yolo / prompt-id / trust block.

        Empty list for non-vibe providers in
        ``_vibe_subshell_setup_block`` keeps the rendered wrapper
        unchanged for every other agent.
        """
        for name in AGENT_PROVIDERS:
            if name == "vibe":
                continue
            wrapper = _provider_wrapper(name)
            assert "VIBE_BYPASS_TOOL_PERMISSIONS" not in wrapper, (
                f"{name} should not touch Vibe env"
            )
            assert "VIBE_SYSTEM_PROMPT_ID" not in wrapper, f"{name} should not touch Vibe env"
            assert "_terok_trust_workspace_for_vibe" not in wrapper, (
                f"{name} should not call Vibe's trust helper"
            )

    def test_all_wrappers_pick_up_initial_prompt(self) -> None:
        """Every provider's wrapper consumes initial-prompt.txt one-shot."""
        for name in AGENT_PROVIDERS:
            wrapper = _provider_wrapper(name)
            assert INITIAL_PROMPT_PATH in wrapper, f"{name} missing initial-prompt pickup"
            assert INITIAL_PROMPT_CONSUMED_PATH in wrapper, f"{name} missing one-shot rename"
            assert f'set -- "$(cat {INITIAL_PROMPT_PATH})"' in wrapper, (
                f"{name} should set positional args from the prompt file"
            )

    def test_initial_prompt_skipped_when_session_present(self) -> None:
        """Wrappers with a session_file gate the prompt pickup on no resume."""
        for name in ("vibe", "opencode", "blablador", "kisski"):
            p = AGENT_PROVIDERS[name]
            wrapper = _provider_wrapper(name)
            assert f"[ ! -s {CONTAINER_TEROK_DIR}/{p.session_file} ]" in wrapper, (
                f"{name} initial-prompt block should defer to session resume"
            )

    def test_initial_prompt_skipped_in_headless(self) -> None:
        """Pickup block requires `_timeout` empty so headless never picks up the file."""
        for name in AGENT_PROVIDERS:
            wrapper = _provider_wrapper(name)
            assert '[ -z "$_timeout" ]' in wrapper, f"{name} missing headless guard"

    def test_claude_wrapper_refuses_setup_token(self) -> None:
        """Claude wrapper short-circuits setup-token with terok auth hint."""
        wrapper = _provider_wrapper("claude")
        assert "setup-token)" in wrapper
        assert "terok auth claude" in wrapper

    def test_codex_wrapper_refuses_login_logout(self) -> None:
        """Codex wrapper refuses login and logout with terok auth hint."""
        wrapper = _provider_wrapper("codex")
        assert "login|logout)" in wrapper
        assert "terok auth codex" in wrapper

    def test_wrappers_without_refuse_have_no_guard(self) -> None:
        """Providers without wrapper.refuse_subcommands emit no guard block."""
        # vibe is API-key only — no login subcommand exists, no refuse list.
        wrapper = _provider_wrapper("vibe")
        assert "Login is unavailable" not in wrapper


class TestResolveProviderValue:
    """Tests for resolve_provider_value() config resolution."""

    def test_flat_string_value(self) -> None:
        """Flat string value is returned for any provider."""
        config = {"model": "opus"}
        assert resolve_provider_value("model", config, "claude") == "opus"
        assert resolve_provider_value("model", config, "codex") == "opus"

    def test_per_provider_dict(self) -> None:
        """Per-provider dict returns provider-specific value."""
        config = {"model": {"claude": "opus", "codex": "o3"}}
        assert resolve_provider_value("model", config, "claude") == "opus"
        assert resolve_provider_value("model", config, "codex") == "o3"

    def test_per_provider_dict_with_default(self) -> None:
        """Per-provider dict falls back to _default for unlisted providers."""
        config = {"model": {"claude": "opus", "_default": "fast"}}
        assert resolve_provider_value("model", config, "codex") == "fast"

    def test_missing_key_returns_none(self) -> None:
        """Missing key returns None."""
        assert resolve_provider_value("model", {}, "claude") is None


class TestApplyProviderConfig:
    """Tests for apply_provider_config() best-effort feature mapping."""

    def test_model_from_config(self) -> None:
        """Model value is read from config when no CLI override."""
        p = AGENT_PROVIDERS["claude"]
        pcfg = apply_provider_config(p, {"model": "opus"})
        assert pcfg.model == "opus"

    def test_timeout_default(self) -> None:
        """Missing timeout defaults to 1800."""
        p = AGENT_PROVIDERS["claude"]
        pcfg = apply_provider_config(p, {})
        assert pcfg.timeout == 1800

    def test_max_turns_unsupported_injects_prompt(self) -> None:
        """Provider without max_turns_flag gets prompt injection + warning."""
        p = AGENT_PROVIDERS["codex"]
        pcfg = apply_provider_config(p, {"max_turns": 30})
        assert pcfg.max_turns is None
        assert "30 steps" in pcfg.prompt_extra


class TestGenerateAllWrappers:
    """Tests for generate_all_wrappers() multi-provider file."""

    def test_all_providers_in_output(self) -> None:
        """Output contains wrapper functions for all providers."""
        wrapper = _all_wrappers()
        for name, p in AGENT_PROVIDERS.items():
            assert f"{p.binary}()" in wrapper, f"Missing wrapper for {name}"

    def test_all_wrappers_valid_bash_syntax(self) -> None:
        """Combined wrapper output passes bash -n syntax check."""
        import shutil
        import subprocess

        if shutil.which("bash") is None:
            pytest.skip("bash is required for wrapper syntax validation")
        wrapper = _all_wrappers(has_agents=True)
        result = subprocess.run(["bash", "-n"], input=wrapper, capture_output=True, text=True)
        assert result.returncode == 0, f"bash syntax error:\n{result.stderr}"

    def test_collect_all_auto_approve_env(self) -> None:
        """The merged auto-approve env map contains all provider env vars."""
        merged = collect_all_auto_approve_env()
        assert merged["OPENCODE_PERMISSION"] == '{"*":"allow"}'
        # VIBE_BYPASS_TOOL_PERMISSIONS is the real Vibe field
        # (vibe.core.config._settings.VibeConfig.bypass_tool_permissions;
        # ACP gates the approval callback on it at acp_agent_loop.py:330).
        # The earlier VIBE_AUTO_APPROVE was a no-op — Vibe's pydantic-settings
        # loads with extra="ignore" and silently drops unknown env.
        assert merged["VIBE_BYPASS_TOOL_PERMISSIONS"] == "true"
        assert merged["COPILOT_ALLOW_ALL"] == "true"
