# Launch modes

terok-executor supports three ways to run an agent — headless,
interactive, and web — plus a separate tool runner for sidecars.

## Headless

```bash
terok-executor run claude . -p "Fix the failing test"
terok-executor run claude . -p "Refactor auth" -m sonnet --max-turns 10
```

Fire-and-forget. The agent runs autonomously and streams output to the
terminal. Exits when the agent finishes, hits `--max-turns`, or reaches
`--timeout` (default 1800 s).

## Interactive

```bash
terok-executor run claude . --interactive
```

Starts the container and keeps it running; once it is ready, the CLI
prints the login command (e.g. `podman exec -it <name> bash -l`). The
agent CLI is installed and ready; credentials and the repository are
pre-configured. Use this to drive the agent manually.

## Web

```bash
terok-executor run claude . --web
terok-executor run claude . --web --port 8080
```

Launches [toad](https://github.com/terok-ai/toad), a multi-agent TUI
served over HTTP on `127.0.0.1` (auto-allocated port unless `--port`
is given). Access it in a browser at the printed URL.

## Tool mode

```bash
terok-executor run-tool coderabbit . -- --pr 42
terok-executor run-tool coderabbit . --timeout 300
```

Runs a sidecar tool in its own container (default timeout: 600 s).
Arguments after `--` are passed to the tool binary. CodeRabbit is
currently the only sidecar tool — see
[Agents](agents.md#sidecar-tools).

## Container lifecycle

Containers follow podman's own lifecycle: `run` creates one and keeps it
after exit, `start` resumes it, `stop` halts it without removing
anything, and `rm` removes it together with its host-side state.  Add
`--rm` to `run` for a disposable container podman removes on exit.

```bash
terok-executor list             # list containers
terok-executor start my-task    # resume a stopped container
terok-executor stop my-task     # halt, kept for a later start
terok-executor rm my-task       # remove container + host-side state
```

The workspace lives inside the container by default — the repo is
cloned in through the gate, so your source checkout stays untouched and
the work survives stop/start in podman storage.  Mount a host directory
with `--workspace` when you want to work directly in it (it outlives
even `rm`).

## Common flags

| Flag | Description |
|------|-------------|
| `--gate` / `--no-gate` | Route the git clone through the per-container gate mirror (default: on; the shield firewall is unaffected).  `--no-gate` makes the container clone the upstream URL directly — a local directory repo is unreachable without the gate and is refused |
| `--restricted` | No auto-approve, no-new-privileges |
| `--branch <ref>` | Check out a specific git branch |
| `--name <name>` | Container name override |
| `--gpus <spec>` | GPU passthrough: `all`, or vendors `nvidia`/`amd`/`intel` (comma-separated) |
| `--memory <limit>` / `--cpus <n>` | Container memory / CPU limits (e.g. `4g`, `2.0`) |
| `--workspace <dir>` | Mount a host directory at `/workspace` (default: the workspace lives in the container) |
| `--rm` | Remove the container when it exits (podman `--rm`) |
| `--git-identity-from-host` | Use the host's git user.name/email as the human committer identity |
| `--shared-dir <dir>` | Mount a host directory as shared IPC space (at `--shared-mount`, default `/shared`) |
| `--timeout <seconds>` | Override the default timeout |
| `-m <name>` | Model override (headless mode) |
| `--max-turns <n>` | Limit agent turns (headless mode) |
