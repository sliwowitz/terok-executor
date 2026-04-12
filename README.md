# terok-executor

Single-agent task runner for hardened Podman containers.

## What it does

**terok-executor** builds container images, launches instrumented Podman
containers, and manages the lifecycle of one AI coding agent at a time.
Every container runs rootless behind an egress firewall with credential
isolation — real API keys and SSH private keys never enter the container.
Use it standalone from the CLI or as a Python library for
[terok](https://github.com/terok-ai/terok) orchestration.

## Ecosystem

```text
terok-shield    nftables egress firewall (security boundary)
terok-sandbox   hardened container runtime (isolation + credential proxy)
terok-executor     single-agent task runner (this package)
terok           project orchestration (TUI, presets, multi-agent)
```

Each layer depends only on the one below it.

## Quick start

```bash
pip install terok-executor        # requires Python 3.12+, Podman (rootless)

terok-executor build              # build base + agent images
terok-executor auth claude        # authenticate (OAuth or API key)

terok-executor run claude . -p "Fix the failing test in test_auth.py"
terok-executor run claude . --interactive   # shell into the container
terok-executor run claude . --web           # toad web UI
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Run an agent in a hardened container (headless, interactive, or web) |
| `auth` | Authenticate a provider (OAuth, API key, or `--api-key` direct) |
| `agents` | List registered agents (`--all` includes tools like gh, glab) |
| `build` | Build base + agent container images |
| `run-tool` | Run a sidecar tool (e.g. CodeRabbit, SonarCloud) |
| `ls` | List running terok-executor containers |
| `stop` | Stop a running container |
| `proxy` | Credential proxy management (start, stop, status, install, routes) |

## Supported agents

| Agent | Auth | Description |
|-------|------|-------------|
| Claude | OAuth, API key | Anthropic Claude Code |
| Codex | OAuth, API key | OpenAI Codex CLI |
| Vibe | API key | Mistral Vibe |
| Copilot | — | GitHub Copilot |
| Blablador | API key | Helmholtz Blablador (OpenCode) |
| KISSKI | API key | KISSKI AcademicCloud (OpenCode) |
| gh | OAuth, API key | GitHub CLI |
| glab | API key | GitLab CLI |
| CodeRabbit | API key | CodeRabbit (sidecar tool) |
| SonarCloud | API key | SonarCloud scanner (sidecar tool) |

## Documentation

- [Getting started](https://terok-ai.github.io/terok-executor/) — install, build, authenticate, first run
- [Agents](https://terok-ai.github.io/terok-executor/agents/) — catalog, custom definitions, auth flows
- [Launch modes](https://terok-ai.github.io/terok-executor/launch-modes/) — headless, interactive, web, tool
- [Security](https://terok-ai.github.io/terok-executor/security/) — firewall, credential proxy, restricted mode
- [API Reference](https://terok-ai.github.io/terok-executor/reference/) — Python API docs

## Development

```bash
poetry install --with dev,test,docs
make check    # lint + test + tach + security + docstrings + deadcode + reuse
```

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)
