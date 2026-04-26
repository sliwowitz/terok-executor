# Getting started

`terok-executor` is the per-task agent runner for hardened Podman
containers.  One CLI command — or one Python class — to build, set
up, authenticate, and launch an AI coding agent inside a rootless
container with default-deny egress and vault-isolated credentials.

![terok ecosystem — terok-executor sits between project orchestration and the hardened runtime](img/architecture.svg)

## Why terok-executor

AI coding agents need network access and credentials to do useful
work, but giving them uncontrolled access to either is a risk: a
prompt-injected or supply-chain-compromised agent can exfiltrate
API keys, push to arbitrary remotes, or reach services it
shouldn't.

terok-executor runs each agent in an isolated rootless Podman
container with an egress firewall and a vault that keeps real
secrets off the container filesystem.  One command to build,
authenticate, and launch.

## Prerequisites

- Python 3.12+
- Podman (rootless) — `podman machine init` on macOS

## Install

```bash
pip install terok-executor
```

## Bootstrap and build

```bash
terok-executor setup
```

`setup` is idempotent and combines image building (base layer + agent
CLIs) with sandbox service installation (vault, gate, shield hooks,
clearance notifier).  Re-run safely after upgrades.

If you prefer to do these steps individually, `terok-executor build`
builds images and `terok-executor vault install` provisions just the
vault.

## Authenticate

```bash
terok-executor auth claude              # OAuth login
terok-executor auth vibe                # interactive API key prompt
terok-executor auth gh --api-key ghp_…  # non-interactive
```

Credentials are stored in a host-side database.  Containers never
see real keys — they receive phantom tokens that the vault resolves
per request.  See [Security](security.md) for details.

## First run

```bash
terok-executor run claude . -p "Add type hints to utils.py"
```

This clones the current directory into a hardened container,
launches Claude in headless mode, and streams its output.  The
egress firewall and vault are active by default.

See [Launch modes](launch-modes.md) for interactive, web, and
tool modes.

## Use as a library

```python
from terok_executor import AgentRunner

runner = AgentRunner()
runner.run_headless(
    agent="claude",
    repo=".",
    prompt="Add type hints to utils.py",
    max_turns=25,
)
```

`AgentRunner` is the same entry point that `terok-executor run` uses
under the hood.  Multi-task orchestrators
([terok](https://github.com/terok-ai/terok)) build on it directly.

## Next steps

- [Agents](agents.md) — supported agents, custom definitions, auth flows
- [Security](security.md) — firewall, vault, restricted mode
- [Launch modes](launch-modes.md) — headless, interactive, web, tool
- [Vault internals](vault.md) — architecture deep dive
