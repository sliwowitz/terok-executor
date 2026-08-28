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

### Harness-driven providers

Curated OpenAI-compatible endpoints driven through the OpenCode
harness — authenticated with their own API key, launched with a
one-word command (`blablador`, `kisski`, `openrouter`):

| Provider | Auth | Description |
|----------|------|-------------|
| Blablador | API key | Helmholtz Blablador |
| KISSKI | API key | KISSKI AcademicCloud (GWDG) |
| OpenRouter | API key | OpenRouter model aggregator |

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

**OAuth / interactive login** (Claude, Codex, gh) — launches a temporary
container with the vendor CLI. After login, the OAuth token is captured
to the host-side credential database.

```bash
terok-executor auth claude
```

Codex also has a headless device-code variant for hosts without a
browser callback: `terok-executor auth codex --device-auth`.

**Interactive API key prompt** (Vibe, Blablador, KISSKI, OpenRouter,
glab, CodeRabbit, SonarCloud) — prompts for a key on the terminal.
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
