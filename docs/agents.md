# Agents

## Supported agents

| Agent | Auth | Description |
|-------|------|-------------|
| Claude | OAuth\*, API key | Anthropic Claude Code |
| Codex | OAuth\* (browser or device code), API key | OpenAI Codex CLI |
| Vibe | API key | Mistral Vibe |
| Copilot | — | GitHub Copilot (no vault route yet) |
| OpenCode | — (uses provider keys) | OpenCode harness; drives any authenticated OpenAI-compatible provider |
| Pi | — (uses provider keys) | [Pi](https://pi.dev/) multi-provider harness; routes through the phantom tokens of co-installed providers |

\* OAuth support for Claude and Codex is experimental.

### Updating Vibe

The Vibe roster installs `mistral-vibe` as `dev` with uv and a GIL-enabled
Python 3.14 interpreter. System Python is unchanged. Inside a container built
with this roster, run `uv tool upgrade mistral-vibe` without sudo, then restart
Vibe. The bulk `sudo update-all-the-things` helper uses the same uv environment.

Older containers retain their root-owned pipx installation. Rebuild the agent
image and create a new container to adopt the uv installation. Updating a tool
inside one container does not update the image or other containers.

### Vibe model discovery

The Vibe image enables its native experimental model catalog for CLI,
headless, and ACP sessions. The Mistral provider retains `backend = "mistral"`
when routed through the vault, so Vibe can fetch its catalog using the phantom
token. Explicitly selected non-Mistral providers use the generic backend and
do not request Mistral's catalog.

Vibe fetches the catalog at session start and caches it in its own shared
configuration directory. Fetch failures fall back to local model definitions.
Set `VIBE_EXPERIMENTAL_ENABLE_MODEL_CATALOG=false` to disable discovery.
The old `vibe-model-sync` helper and its unused `mistral-models.txt` cache are
no longer used. Rebuild the agent image and create a new container to adopt
the new image default and launcher.

For manual model definitions, set both `max_context_length` and
`max_output_tokens`: the total context window and per-response output limit
are distinct. In Vibe 3.0.2, omitting the output cap can request all remaining
context space as output and exceed the provider's completion-token limit.

### Vibe task sessions

Vibe keeps its native transcript storage: by default `~/.vibe/logs/session`,
inside the shared Vibe mount. Custom `VIBE_HOME` and session-logging settings
are not overridden. Mount sharing is chosen by the caller; terok defaults to
host-wide sharing and supports project-scoped mounts via `credentials.scope`.

The CLI wrapper and ACP adapter install native `pre_tool` and `post_agent`
[hooks](https://github.com/mistralai/mistral-vibe#hooks) in Vibe's `hooks.toml`,
preserving existing hooks and comments. Hooks
receive the session ID from Vibe itself and write only root-session IDs to
`/home/dev/.terok/vibe-session-id.txt`, the task-private resume pointer.
Subagent hooks cannot replace it. The installer locks and atomically updates
the shared hooks file, so simultaneous task launches cannot clobber it.

Automatic CLI resume uses only that captured ID, never the newest shared
transcript. `--terok-new-session` starts fresh; explicit `--resume` and
`--continue` retain Vibe's native behavior and can select shared history.
ACP clients control their own session selection; their root-session hooks
also capture the pointer for a subsequent CLI launch.

There is no session-start hook in Vibe 3.0.2: capture happens before a tool
call or after a completed agent turn. If a fresh run exits before either
hook fires, or hooks are unavailable/disabled, the wrapper warns that
automatic resume is unavailable. It never guesses from shared history.
Use `vibe --resume` to select a transcript manually in that case. Session
logging must remain enabled for Vibe to persist a resumable transcript.

Vibe validates checkpoint contents when resuming. A rejected resume keeps
its error, pointer, and fresh-session hint; there is no silent fallback to
older work. Old `vibe-session.txt` pointers came from unverified discovery
and are ignored. Existing transcripts are neither deleted nor relocated.
Rebuild the agent image and recreate the task container to install the hook
helper and ACP adapter. Initial instructions, first-prompt delivery, and
wrapper agent history remain unchanged.

### Harness-driven providers

Curated OpenAI-compatible endpoints driven through the OpenCode
harness — authenticated with their own API key, launched with a
one-word command (`blablador`, `kisski`, `openrouter`, `opper`):

| Provider | Auth | Description |
|----------|------|-------------|
| Blablador | API key | Helmholtz Blablador |
| KISSKI | API key | KISSKI AcademicCloud (GWDG) |
| OpenRouter | API key | OpenRouter model aggregator |
| Opper | API key | Opper EU-hosted AI gateway |

### Tools

Optionally available in the container:

| Tool | Auth | Description |
|------|------|-------------|
| gh | OAuth, API key | GitHub CLI |
| glab | API key | GitLab CLI |
| SonarCloud (`sonar`) | API key | SonarCloud scanner (`sonar-scanner`) |

### Sidecar tools

Tools run in a separate container:

| Tool | Auth | Description |
|------|------|-------------|
| CodeRabbit | API key | CodeRabbit code review |


## Listing agents

```bash
terok-executor agents list           # coding agents only
terok-executor agents list --all     # include tools (gh, glab, coderabbit, sonar) and harness-driven providers
```

## Setting the global default

The same selection string that `terok-executor build --agents …` accepts
also drives the global default that's baked into L1 images when a
project does not override `image.agents`:

```bash
terok-executor agents set                # interactive picker
terok-executor agents set all            # every roster entry
terok-executor agents set claude,vibe    # explicit list
terok-executor agents set all,-vibe      # everything except vibe
```

The value lands in `~/.config/terok/config.yml` under `image.agents` by
default — `/etc/terok/config.yml` when running as root, or whatever
`TEROK_CONFIG_FILE` points at when that env var is set.
Validation runs against the installed roster up front, so the file
never references a name that won't resolve at build time.

## Authentication

Three auth paths depending on the provider:

**Vendor login** (Claude, Codex, gh, Vibe) — launches a temporary
container with the vendor CLI and runs its own login. After login, the
credential it produced (an OAuth token, or the key Vibe's browser sign-in
issues for your plan) is captured to the host-side credential database.

```bash
terok-executor auth claude
```

Vibe's browser sign-in needs no browser in the container: press `c` in
its setup to show the sign-in URL and open it on any machine.

Codex also has a headless device-code variant for hosts without a
browser callback: `terok-executor auth codex --device-auth`.

**Interactive API key prompt** (Vibe, Blablador, KISSKI, OpenRouter,
Opper, glab, CodeRabbit, SonarCloud) — prompts for a key on the terminal.
No container needed.

```bash
terok-executor auth vibe
```

**Non-interactive** (any provider with an auth flow) — pass the key
directly:

```bash
terok-executor auth gh --api-key ghp_…
```

After authentication, containers receive phantom tokens instead of real
credentials. See [Security](security.md) for how this works.

## Running sidecar tools

Sidecar tools like CodeRabbit run via `run-tool`. Arguments after
`--` are passed to the tool binary:

```bash
terok-executor run-tool coderabbit . -- --pr 42
```

## Custom agents

Place YAML files in `~/.config/terok/agent/agents/`. The roster merges
user definitions with bundled ones using deep merge for dicts and
`_inherit` splicing for lists.

See the bundled definitions in `resources/agents/` for the schema:
binary, headless flags, provider binding, auth modes, and git
identity.  Endpoint definitions (upstream URL, wire auth) live
separately in `resources/providers/`, with user overrides in
`~/.config/terok/providers/`.

## Custom providers

A provider is an LLM endpoint. It is not an agent that Terok installs. The YAML
file name, without `.yaml`, is the provider name. The name must match
`[a-z0-9]+`. Thus, use only lowercase ASCII letters and digits. Terok uses the
name in environment variables and provider selectors. A new provider name must
not match an existing agent or tool name.

This example defines the `example` provider:

```yaml title="~/.config/terok/providers/example.yaml"
label: Example
upstream: https://api.example.com

auth:
  api_key: {}

serves:
  openai-chat: /v1

default_model: example-chat
models:
  example-chat:
    name: Example Chat
    limit:
      context: 120000
```

`auth.api_key: {}` uses the standard `Authorization: Bearer <key>` format.
Specify `header` or `prefix` only if the endpoint uses a different format. In
legacy files, an explicit header and no prefix means that there is no prefix.
Specify `prefix: "Bearer "` if you need this prefix. The `serves` map assigns an
API base path to each supported protocol.

When `models` is not empty, OpenCode and Pi do not request `/models`. This map
is the source of model data. `default_model` specifies the preferred model. The
optional model fields are `name`, `limit.context`, and `limit.output`. Pi uses
each limit independently. OpenCode requires both limits in one `limit` block.
Therefore, Terok writes this block only when both limits are available. In this
example, OpenCode registers the model without a `limit` block.

Authenticate on the host. Then start a new task:

```bash
terok-executor auth example

# Inside a task that includes OpenCode or Pi:
opencode --provider example
pi --provider example
```

After successful authentication, Terok updates the vault routes automatically.
Terok updates the routes again when a task starts. You do not have to run the
`vault routes` command.

You do not have to rebuild an image after you add or change a provider file.
Add `opencode` or `pi` to `image.agents`. Do not add `example` to
`image.agents`. Select the provider at run time. Start a new task to load the
changed provider settings.

Terok continues to read the legacy `~/.config/terok/agent/providers/`
directory. Terok also reads provider files that contain the legacy `opencode:`
block. Terok loads the legacy directory before the current provider directory.
Thus, a file in the current directory overrides a legacy file that has the same
name. Use the provider-neutral format for new files. For all fields, see the generated
[Provider YAML reference](roster-reference.md#provider-yaml) and the
[provider JSON Schema](schemas/provider.schema.json).

## Wrapper flags

Inside a task, each agent command (`claude`, `codex`, `opencode`, …) is a
shell function that terok-executor generates. The wrapper sets the git
identity and sends the task's initial prompt; where the agent supports it, the
wrapper also resumes the recorded session and routes the provider. It reads its
own flags too. Put them before the agent flags:

| Flag | Effect | Available on |
|------|--------|--------------|
| `--terok-timeout SECS` | Run without a terminal; stop the agent after `SECS` seconds | every wrapper |
| `--provider NAME` | Route the agent through the authenticated provider `NAME` (`providers` lists the ready ones) | Claude and any agent with a provider launcher (`opencode`, `codex`, `vibe`, …) |
| `--terok-new-session` | Start a new session; do not resume the recorded one | agents that resume a session (not `copilot`) |

Each wrapper accepts only the flags it acts on, so `<agent> --help` lists that
agent's own subset around the agent's usage text. If a resumed agent exits with
an error, the wrapper reports it and points to `--terok-new-session`. The
wrapper never retries. To skip the wrapper, run `command <agent>`.

Before launch, the `claude`, `codex` and `vibe` wrappers check that the task
holds a credential for the provider the agent will use. They read the same
snapshot as `providers`, taken when the task started. Claude's check on its
default endpoint also reads one flag that terok sets, described below; the
readiness views do not. Without a credential, the wrapper stops and names the
command to run on the host, for example `terok auth claude`. Do not log in
inside the task: the egress firewall reserves the provider's hosts for the
vault. A login on the host takes effect in a new task. `--help` works without a
credential.

Claude refuses a selected provider (`--provider`, or the task's
`TEROK_PROVIDER`) that has no credential in the task. It does not switch to its
default provider instead, because that would send the session to a provider
nobody selected. A selected provider that has a credential but does not speak
Claude's protocol, for example one selected for OpenCode, leaves Claude on its
default provider. The check then covers that default. The Claude ACP launcher,
`terok-claude-acp`, runs the default-provider check from the task's wrappers,
so Toad and terok's ACP proxy cannot start Claude without a credential. If a
user agent override re-binds Claude's default provider, both paths check the
new default.

A stored credential does not always give Claude's default endpoint a route.
The vault can hold a Claude OAuth login that terok's configuration neither
routes to Claude nor exposes to it. Routing needs `experimental: true` and
`agent.claude.allow_oauth: true` in terok's `config.yml`; exposure needs
`experimental: true` and `agent.claude.expose_oauth_token: true`. With neither,
terok removes Claude's own route to Anthropic but keeps the login's provider
handle, because other agents, such as OpenCode and Pi, still reach Anthropic
through the vault with it. A bare `claude` would go straight to Anthropic's
hosts and meet the egress firewall. So terok also sets
`TEROK_CLAUDE_OAUTH_UNROUTED=1` in the task, in that case only. It never sets
the variable while routing is on or the login is exposed, because Claude's
default endpoint then works. With that value, both Claude entry points refuse
a launch on the default endpoint up front. The refusal names the cause and the
remedies: store an API key with `terok auth claude`, or turn OAuth routing on,
then start a new task.

A selected provider that has a credential and speaks Claude's protocol still
routes the shell wrapper through the vault. This includes Claude's default
provider: `claude --provider anthropic` reaches Anthropic through its vault
handle, as `claude --provider openrouter` reaches OpenRouter with a credential
of its own. The shell refusal names this way round, which works in the same
task. The ACP launcher honours no selection, so it refuses whenever the flag is
on. Only Claude's entry points read the variable, and any value other than `1`
leaves them unchanged. `providers` and `hilfe` do not read it: they list the
pairs that a selection can use, so they keep showing Claude and its default
provider as ready. terok takes the name and the value from
`terok_executor.CLAUDE_OAUTH_UNROUTED_ENV` and
`terok_executor.CLAUDE_OAUTH_UNROUTED_ON` rather than spelling them.

Every wrapper invocation attempts to append its command name to
`~/.terok/agent-history` in that task, regardless of the agent's exit status.
History writes are best-effort and do not prevent agents from starting.
Provider aliases such as `blablador` record their own names.
When a fresh interactive shell opens, the greeter
lists unique names after the help banner, in first-use order:
`Agents previously active: codex, claude`. Tasks without history show no extra
line. Reattaching an existing tmux shell does not rerun the greeter. Commands run
without the shell wrapper do not record history.

Each agent records its last session in its own file under
`/home/dev/.terok/` (`claude-session.txt`, `codex-session.txt`,
`opencode-session.txt`, …), and the pinned provider aliases (`blablador`,
`kisski`) keep their own too. So several agents in one task never resume each
other's conversation. Codex resumes through its `resume` subcommand
(`codex resume <id>`, headless `codex exec resume <id> …`). A `SessionStart`
hook, installed as `/etc/codex/config.toml` — Codex's trusted system config
layer — records the id.

## Git identity

By default, agents commit under a built-in AI identity. To record the
host machine's git identity as the human committer alongside the agent
author:

```bash
terok-executor run claude . --git-identity-from-host -p "…"
```

This reads `user.name` and `user.email` from the host's global git
config and injects them as the human committer identity
(`HUMAN_GIT_NAME` / `HUMAN_GIT_EMAIL`); the agent remains the author.
