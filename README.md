<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://terok-ai.github.io/terok/terok-logo-w.svg">
    <img src="https://terok-ai.github.io/terok/terok-logo-b.svg" alt="terok-executor" width="120">
  </picture>
</p>

# terok-executor

[![PyPI](https://img.shields.io/pypi/v/terok-executor)](https://pypi.org/project/terok-executor/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/terok-ai/terok-executor)](https://api.reuse.software/info/github.com/terok-ai/terok-executor)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=terok-ai_terok-executor&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=terok-ai_terok-executor)

One command to run an AI coding agent inside a hardened, rootless
Podman container.

`terok-executor` builds the container, launches the agent against
the directory you point it at, and keeps real credentials on the
host.  Use it on its own as a CLI, or import its `AgentRunner` from
Python when you want library-grade control.

<p align="center">
  <img src="https://terok-ai.github.io/terok/img/architecture.svg" alt="terok ecosystem — terok-executor sits between project orchestration and the hardened runtime">
</p>

## Quick start

```bash
pip install terok-executor
terok-executor run claude ~/my-workspace -p "Fix the bug"
```

The first `run` interactively offers any missing prerequisites — sandbox
services, container images, agent credentials.
Mandatory items (services, images) block the launch if
declined; optional ones (SSH key, auth) print the consequence and
proceed.

Individual steps would be:

```bash
terok-executor setup                               # install sandbox services + build base images
terok-executor auth claude                         # authenticate (OAuth or API key)
terok-executor run claude <dir> -p "Fix the bug"   # run the agent with an initial prompt
```

## Use as a library

```python
from terok_executor import AgentRunner

runner = AgentRunner()
runner.run_headless(
    agent="claude",
    repo=".",
    prompt="Fix the failing test in test_auth.py",
    max_turns=25,
)
```

`AgentRunner` exposes four launch methods — `run_headless`,
`run_interactive`, `run_web`, `run_tool` — all with the same
hardening guarantees.

## Supported agents

| Agent | Auth | Description |
|-------|------|-------------|
| Claude Code | OAuth*, API key | Anthropic Claude Code |
| Codex | OAuth*, API key | OpenAI Codex CLI |
| Vibe | API key | Mistral Vibe |
| Copilot | — | GitHub Copilot (no vault route yet) |
| OpenCode | — (uses provider keys) | Harness that drives any OpenAI-compatible provider — curated configs for Helmholtz Blablador, KISSKI AcademicCloud, and OpenRouter (each authenticated with its own API key) |
| Pi | — (uses provider keys) | Multi-provider harness; routes through the phantom tokens of co-installed providers |
| gh | OAuth, API key | GitHub CLI |
| glab | API key | GitLab CLI |
| SonarCloud | API key | SonarCloud scanner |
| CodeRabbit | API key | CodeRabbit (sidecar tool) |

\* Claude and Codex OAuth are experimental.

`terok-executor agents list` lists the live roster (add `--all` to
include tools and harness-driven providers).

## Where it sits in the stack

terok-executor is the per-task layer.  Above it,
[terok](https://github.com/terok-ai/terok) composes many concurrent
runs across many projects.  Below it, terok-executor delegates the
host-side security boundary
([terok-sandbox](https://github.com/terok-ai/terok-sandbox)): the
credential vault, the git gate, the egress firewall hooks, the
per-container supervisor lifecycle (OCI `createRuntime`/`poststop`
hooks).

## Commands

| Command | Description |
|---------|-------------|
| `run` | Launch an agent (headless, interactive, or web) |
| `setup` | Bootstrap sandbox services + container images |
| `uninstall` | Remove sandbox services + container images |
| `auth` | Authenticate a provider |
| `agents` | Inspect the agent roster (`list`) and set the build-time default selection (`set`) |
| `build` | Build base + agent images explicitly |
| `run-tool` | Run a sidecar tool (CodeRabbit) |
| `list` | List containers |
| `start` | Start a stopped container |
| `stop` | Stop a container (kept for a later `start`) |
| `rm` | Remove a container and its host-side state |
| `show-config` | Print the effective `SandboxConfig` as YAML (diffable across orchestrators) |
| `vault` | Vault management (status, unlock, lock, list, passphrase, routes, clean) |
| `sandbox` | Full terok-sandbox command tree (shield, vault, ssh, doctor, …) |

### Config override

Two top-level flags (precede the subcommand, like `docker --config`):

- `--config PATH` — read this `config.yml` instead of the layered system/user paths (sets `TEROK_CONFIG_FILE` for the invocation).
- `--raw` — ignore any `config.yml`; use sandbox/executor dataclass defaults only.

Higher-layer orchestrators (such as `terok`) typically construct a `SandboxConfig` from their own resolution chain and pass it into the executor as a library; the public expectation is that, for the fields they own in `config.yml`, the resulting sub-environment matches what standalone `terok-executor` would produce against the same file.  Use `show-config` on both sides to verify.

## Development

See the [Developer Guide](https://terok-ai.github.io/terok-executor/developer/).

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)
