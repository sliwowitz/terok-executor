# terok-agent

Single-agent task runner for hardened Podman containers.

**terok-agent** builds container images, launches instrumented Podman
containers, and manages the lifecycle of one AI coding agent at a time.
Designed for standalone use (`terok-agent run claude .`) and as a library
for [terok](https://github.com/terok-ai/terok) project orchestration.

## Ecosystem

```text
terok-shield    nftables egress firewall (security boundary)
terok-sandbox   hardened container runtime (isolation + credential proxy)
terok-agent     single-agent task runner (this package)
terok           project orchestration (TUI, presets, multi-agent)
```

Each layer depends only on the one below it.

## Installation

```bash
pip install terok-agent
```

Requires Python 3.12+ and Podman (rootless).

## Quick start

```bash
# Build L0 (base) + L1 (agent CLI) images
terok-agent build

# Authenticate with a provider
terok-agent auth claude

# Run an agent — headless with a prompt
terok-agent run claude . -p "Fix the failing test in test_auth.py"

# Run interactively (user logs into the container)
terok-agent run claude . --interactive

# Run toad web UI
terok-agent run claude . --web
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Run an agent in a hardened container (headless, interactive, or web) |
| `auth` | Authenticate a provider (OAuth, API key, or `--api-key` direct) |
| `agents` | List registered agents (`--all` includes tools like gh, glab) |
| `build` | Build L0+L1 container images |
| `run-tool` | Run a sidecar tool (e.g. CodeRabbit) |
| `ls` | List running terok-agent containers |
| `stop` | Stop a running container |
| `proxy start\|stop\|status\|install\|uninstall\|routes\|clean` | Credential proxy management |

## Supported agents

| Agent | Type | Auth | Description |
|-------|------|------|-------------|
| Claude | native | OAuth, API key | Anthropic Claude Code |
| Codex | native | OAuth, API key | OpenAI Codex CLI |
| Vibe | native | API key | Mistral Vibe |
| Copilot | native | — | GitHub Copilot (no proxy yet) |
| Blablador | OpenCode | API key | Helmholtz Blablador |
| KISSKI | OpenCode | API key | KISSKI (AcademicCloud) |
| gh | tool | OAuth, API key | GitHub CLI |
| glab | tool | API key | GitLab CLI |
| CodeRabbit | tool | API key | CodeRabbit (sidecar) |
| SonarCloud | tool | API key | SonarCloud scanner |

User-defined agents go in `~/.config/terok/agent/agents/` as YAML files.

## Security

- **Egress firewall** — gate is on by default; agents cannot reach the
  internet except through allowed domains
- **Credential proxy** — no real API keys enter containers; phantom tokens
  are resolved host-side (see [Credential Proxy](https://terok-ai.github.io/terok-agent/credential-proxy/))
- **Restricted mode** (`--restricted`) — disables auto-approve and sets
  `--no-new-privileges`
- **Rootless Podman** — containers run without root privileges

## Development

```bash
poetry install --with dev,test,docs
make check    # lint + test + tach + security + docstrings + deadcode + reuse
```

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)
