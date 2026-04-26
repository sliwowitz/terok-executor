# terok-executor

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/terok-ai/terok-executor)](https://api.reuse.software/info/github.com/terok-ai/terok-executor)

Per-task agent runner for hardened Podman containers.

`terok-executor` builds container images, launches a single AI
coding agent inside a rootless container with default-deny egress
and vault-isolated credentials, and exposes the same lifecycle as
both a standalone CLI and a Python library.  It is the layer that
turns *"give me an agent on this repo"* into a running, bounded
container.

<p align="center">
  <img src="docs/img/architecture.svg" alt="terok ecosystem — terok-executor sits between project orchestration and the hardened runtime">
</p>

## What it provides

- **`AgentRunner` Python API** — one object, four launch methods:
  `run_headless`, `run_interactive`, `run_web`, `run_tool`.  Same
  hardening guarantees regardless of mode.
- **Image factory** — builds a layered image stack (base distro,
  agent CLIs, optional sidecar tools) on top of any allowed base
  image; cached and reused.
- **Auth flows** — OAuth and API-key flows for every supported
  provider.  Real credentials stay on the host; containers see
  phantom tokens that the vault resolves per request.
- **Roster + provider registry** — agents are declared in YAML
  (bundled defaults plus user overlays under
  `~/.config/terok/agents/`); add a new endpoint without touching
  Python.
- **Sidecar tools** — non-agent helpers (CodeRabbit, SonarCloud) run
  in their own image with the same vault model.
- **Doctor + setup** — `terok-executor setup` brings up the
  underlying sandbox services and image cache; `terok-executor doctor`
  reports drift.

## Where it sits in the stack

terok-executor is the per-task layer.  Above it, multi-task
orchestration ([terok](https://github.com/terok-ai/terok)) composes
many concurrent runs across many projects.  Below it, terok-executor
delegates the entire host-side security boundary
([terok-sandbox](https://github.com/terok-ai/terok-sandbox)): the
vault, the git gate, the egress firewall hooks, the systemd service
lifecycle.  The split keeps the executor focused on what an agent
needs at runtime, and the sandbox focused on what the host needs to
trust the container.

You can use terok-executor entirely on its own — point it at a
directory, give it a prompt, get a run.  When you reach for project
config, presets, or multiple parallel agents, that's the moment to
add terok on top.

## Supported agents

| Agent | Auth | Description |
|-------|------|-------------|
| Claude Code | OAuth, API key | Anthropic Claude Code |
| Codex | OAuth, API key | OpenAI Codex CLI |
| Vibe | API key | Mistral Vibe |
| Copilot | OAuth | GitHub Copilot |
| OpenCode | API key | Generic LLM endpoint driver — bundled defaults for Helmholtz Blablador, KISSKI AcademicCloud, and your own endpoint |
| gh | OAuth, API key | GitHub CLI |
| glab | API key | GitLab CLI |
| CodeRabbit | API key | CodeRabbit (sidecar tool) |
| SonarCloud | API key | SonarCloud scanner (sidecar tool) |

## Quick start

Two paths converge on the same ready state.  Pick whichever fits.

### Explicit bootstrap (recommended for first install)

```bash
pip install terok-executor        # requires Python 3.12+, Podman (rootless)

terok-executor setup              # install shield hooks + vault + gate, build images
terok-executor auth claude        # authenticate (OAuth or API key)

terok-executor run claude . -p "Fix the failing test in test_auth.py"
```

`terok-executor setup` is idempotent — safe to re-run after upgrades.
The image build step is minutes-long only on first install; subsequent
runs reuse the cached layers.

### Lazy first run

```bash
pip install terok-executor
terok-executor run claude .       # prompts for each missing prerequisite
```

Missing pieces are offered one at a time with `[Y/n]` prompts:
sandbox services, container images, gate SSH key, and agent
credentials.  Mandatory items (services, images) block the launch if
declined; optional ones (SSH key, auth) print the consequence and
proceed.

Non-interactive environments (CI, scripts) should either run
`terok-executor setup` first or pass `--yes` / `--no-preflight` on
the `run` invocation.

### Use as a library

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

### Uninstall

```bash
terok-executor uninstall              # removes services + image cache
terok-executor uninstall --keep-images  # leave the image cache for fast re-install
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Launch an agent (headless, interactive, or web) |
| `setup` | Bootstrap sandbox services + container images |
| `uninstall` | Remove sandbox services + container images |
| `auth` | Authenticate a provider (OAuth, API key, or `--api-key`) |
| `agents` | List registered agents (`--all` includes tool entries) |
| `build` | Build base + agent container images explicitly |
| `run-tool` | Run a sidecar tool (CodeRabbit, SonarCloud) |
| `list` | List running terok-executor containers |
| `stop` | Stop a running container |
| `vault` | Vault management (start, stop, status, install, routes) |

## Documentation

- [Getting started](https://terok-ai.github.io/terok-executor/) — install, build, authenticate, first run
- [Agents](https://terok-ai.github.io/terok-executor/agents/) — catalog, custom definitions, auth flows
- [Launch modes](https://terok-ai.github.io/terok-executor/launch-modes/) — headless, interactive, web, tool
- [Security](https://terok-ai.github.io/terok-executor/security/) — firewall, vault, restricted mode
- [API Reference](https://terok-ai.github.io/terok-executor/reference/) — Python API docs

## Development

```bash
poetry install --with dev,test,docs
make check    # lint + test + tach + security + docstrings + deadcode + reuse
```

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)
