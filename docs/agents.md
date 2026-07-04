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
`~/.config/terok/agent/providers/`.

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
