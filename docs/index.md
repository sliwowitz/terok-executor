# terok-agent

Single-agent task runner for hardened Podman containers.

## Overview

**terok-agent** builds container images, launches instrumented Podman
containers, and manages the lifecycle of one AI coding agent at a time.
It sits between [terok-sandbox](https://github.com/terok-ai/terok-sandbox)
(container isolation) and [terok](https://github.com/terok-ai/terok)
(project orchestration):

```text
terok  ->  terok-agent  ->  terok-sandbox  ->  terok-shield
```

### Standalone vs. library

`terok-agent run claude .` is a self-contained command — it builds images,
prepares config, and launches a container. The same functionality is
available as a Python library for terok to compose into multi-agent
workflows:

```python
from terok_agent import AgentRunner

runner = AgentRunner()
cname = runner.run_headless("claude", "/path/to/repo", prompt="Fix the bug")
```

## Quick start

```bash
pip install terok-agent        # requires Python 3.12+, Podman (rootless)
terok-agent build              # build L0 (base) + L1 (agent CLI) images
terok-agent auth claude        # authenticate (OAuth or API key)
terok-agent run claude . -p "Fix the failing test"
```

## Image layer architecture

```text
L0 (base)    Ubuntu + dev tools + init script + dev user
L1 (agent)   All AI agent CLIs, shell wrappers, ACP config
--- boundary: above owned by terok-agent, below by terok ---
L2 (project) Optional user Dockerfile snippet (custom packages)
```

L1 is self-sufficient for standalone use. All user config (repo URL, SSH
keys, branch, gate) is injected at runtime via environment variables and
volume mounts — no L2 build needed.

## Launch modes

| Mode | Flag | Description |
|------|------|-------------|
| Headless | `-p "prompt"` | Fire-and-forget with a prompt; streams output |
| Interactive | `--interactive` | User logs into the container; agent is ready |
| Web | `--web` | Toad multi-agent TUI served over HTTP |
| Tool | `run-tool <name>` | Run a sidecar tool (CodeRabbit, SonarCloud) |

## Agent registry

Agents are defined in YAML files under `resources/agents/`. Each file
declares the provider binary, headless flags, credential proxy routing,
auth modes, and git identity. Users can extend the registry by placing
YAML files in `~/.config/terok/agent/agents/`.

See the [API Reference](reference/) for the full agent roster schema.

## Security model

- **Egress firewall (gate)** — on by default. Containers cannot reach the
  internet except through explicitly allowed domains. Disable with
  `--no-gate` for development.
- **[Credential proxy](credential-proxy.md)** — no real API keys enter
  containers. Phantom tokens are resolved host-side by the credential
  proxy. See the dedicated page for architecture details.
- **Restricted mode** (`--restricted`) — disables auto-approve flags and
  sets `--no-new-privileges` on the container.
- **Rootless Podman** — all containers run without root privileges.
- **Config isolation** — vendor config directories are bind-mounted
  read-only where possible. Secrets are never written to shared mounts.

## Configuration

terok-agent uses a layered config stack (global -> project -> preset -> CLI)
resolved via `config_stack`. The roster merges bundled agent definitions
with user overrides using `_inherit` splicing for lists and deep merge
for dicts.

Key config paths:

| Path | Purpose |
|------|---------|
| `~/.config/terok/agent/agents/` | User agent YAML overrides |
| `~/.local/share/terok/agent/` | State root (credentials, tasks) |
| `TEROK_AGENT_STATE_DIR` | Override state root via env var |
