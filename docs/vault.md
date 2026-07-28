# Vault Integration

## Problem

terok bind-mounts vendor config directories into task containers. A
prompt-injected or supply-chain-compromised agent can read and exfiltrate
API keys, OAuth tokens, or SSH private keys from these shared mounts.

## Solution: Token Broker

No real secret enters a task container. Instead:

1. **Credential DB** (host-side) stores API keys and OAuth tokens
2. **Token broker** ([terok-sandbox](https://terok-ai.github.io/terok-sandbox/))
   resolves phantom tokens to real credentials and forwards requests upstream.
   It runs inside each container's supervisor (spawned by the terok-sandbox
   OCI hook) and listens on a Unix socket (default) or a per-container host
   TCP port (legacy)
3. **Per-provider phantom tokens** (per-task, per-provider) are what containers see
4. **SSH signer** ([terok-sandbox](https://terok-ai.github.io/terok-sandbox/))
   lets containers sign with host-side SSH keys without the private keys
   entering the container — via the bind-mounted signer socket
   (`TEROK_SSH_SIGNER_SOCKET`) or, in TCP mode, a per-container port
   (`TEROK_SSH_SIGNER_PORT`)

## Architecture

```text
HOST                                      CONTAINER
-------------------------------           -----------------------------------
Credential DB (encrypted SQLite,          Per-provider phantom tokens (env vars)
SQLCipher)                                  ANTHROPIC_API_KEY=<anthropic-phantom>
  credentials table                         MISTRAL_API_KEY=<mistral-phantom>
  proxy_tokens table                        GH_TOKEN=<github-phantom>
    (token, scope, subject,                 GITLAB_TOKEN=<gitlab-phantom>
     credential_set, provider)
                                          Agent / tool makes API request
Token Broker (terok-sandbox,                with phantom token in auth header
per-container supervisor)
  Unix socket (default) or TCP listener   Routing is by token, not URL path.
  Resolves phantom → real credential       Token encodes which provider it's for.
  Injects auth header, forwards
  to upstream over TLS
```

### SELinux considerations

SELinux blocks `connect()` on host Unix sockets mounted into rootless
Podman containers (`container_t -> unconfined_t` denied).  terok-sandbox
ships a policy module (`terok_socket.te`, installed by `setup`) that
allows exactly this connect.  Where the module cannot be installed, the
containers reach the token broker via TCP
(`host.containers.internal:<port>`) instead; [terok-shield](https://terok-ai.github.io/terok-shield/)
allows the broker port through the nftables firewall via its per-container
`loopback_ports` state.

### Per-provider phantom token routing

Each routed provider gets its own phantom token, minted at container
launch:

```python
tokens = {name: db.create_token(scope, task_id, credential_set, name) for name in routed_providers}
```

The token broker resolves the route from the token's `provider` field, not from
the URL path. This is essential because some SDKs (Vibe's Mistral SDK,
gh CLI) strip or ignore URL path components.

## Per-Agent Traffic Routing

Inside the container, HTTP clients always reach the vault at
`http://localhost:9419` — an in-container loopback bridge that runs in
both transports and forwards to the host socket (socket mode) or the
per-container broker port (TCP mode).  Different agents are pointed at
it in different ways, depending on what their SDK supports:

| Agent | How it reaches the token broker | Notes |
|-------|-------------------------|-------|
| **Claude** | `ANTHROPIC_BASE_URL=http://localhost:9419` (+ `ANTHROPIC_UNIX_SOCKET` pointing at the vault socket) | Anthropic SDK respects these env vars |
| **Codex** | Shared `~/.codex/config.toml` rewrite (`openai_base_url`, `chatgpt_base_url`) | Codex's built-in first-party auth is file/config based, so terok patches the shared Codex config instead of relying on env vars.  `openai_base_url` is keyed by stored credential type: OAuth → `{vault_url}/backend-api/codex`, API key → `{vault_url}/v1` |
| **Vibe** | `config.toml` with `api_base` (+ `api_key_env_var`) in shared `~/.vibe` mount | Mistral SDK ignores the URL path in api_base, only uses host:port. Written by the `provider.config_patch` in YAML |
| **Blablador / KISSKI / OpenRouter** | `TEROK_OC_<NAME>_BASE_URL` env var override | The OpenCode wrapper reads this; computed at launch as the vault URL plus the provider's served path (e.g. `/v1`, `/api/v1` for OpenRouter) |
| **gh** | `http_unix_socket` in `~/.config/gh/config.yml` | gh routes ALL API traffic through a Unix socket. See below. |
| **glab** | `GITLAB_API_HOST` + `API_PROTOCOL=http` env vars | glab sends to `http://<api_host>/api/v4/...`; the host is `localhost:9419` in socket mode, `host.containers.internal:<port>` in TCP mode |
| **CodeRabbit** | Real API key via sidecar `env_map` | CLI has no base URL override, so token broker routing is not possible. The sidecar receives the real key directly from the credential DB. |
| **SonarCloud** | `SONAR_HOST_URL` + `SONAR_TOKEN` phantom env | Tool; scanner uses the host URL override |

In addition, every routed provider with a `serves:` protocol map is
materialized as a generic handle — `TEROK_PROVIDER_<NAME>_TOKEN` plus
`TEROK_PROVIDER_<NAME>_BASE_<PROTOCOL>` — so harnesses (OpenCode, Pi)
can select any authenticated, protocol-compatible provider at runtime
via `--provider`.

### gh: Unix-socket pattern

gh has no env var for base URL. It supports `http_unix_socket` in its
config file, which routes all API traffic through a Unix socket.

The config-patch mechanism writes the vault socket path into
`~/.config/gh/config.yml` after `terok-executor auth gh` — the
`{vault_socket}` template token, resolved per transport mode:

- **socket mode**: `/run/terok/vault.sock` — the supervisor's vault
  socket, bind-mounted into the container.
- **TCP mode**: `/tmp/terok-vault.sock` — a socat bridge started by
  terok-sandbox's `ensure-bridges.sh`, forwarding to
  `host.containers.internal:${TEROK_TOKEN_BROKER_PORT}`.

Either way gh's API traffic reaches the vault token broker, which
substitutes the phantom token before forwarding upstream.

### YAML-driven config patches

Agents that need config file changes (not just env vars) declare a
`config_patch` under their `provider:` binding:

```yaml
# Vibe: TOML patch
provider:
  config_patch:
    file: config.toml
    toml_table: providers
    toml_match: {name: mistral}
    toml_set:
      api_base: "{vault_url}/v1"
      api_key_env_var: MISTRAL_API_KEY

# gh: YAML patch
provider:
  config_patch:
    file: config.yml
    yaml_set: {http_unix_socket: "{vault_socket}"}
```

The patch is applied after auth and reconciled on every task launch.  Only
non-secret values (URLs, socket paths) are written to shared mounts.  The
patcher also writes a `.terok-managed-config.json` sidecar beside the config
file so callers can later disable a provider and remove only values still
owned by terok; user-edited values are preserved.

## YAML Registry: Providers and Bindings

Vault integration is split across two YAML files.  The **endpoint**
half lives in `resources/providers/<name>.yaml` — where requests go
and how the real credential is attached:

```yaml
# resources/providers/anthropic.yaml
upstream: https://api.anthropic.com
auth:                            # wire auth per credential type
  oauth:
    header: Authorization
    prefix: "Bearer "
    extra_headers: {anthropic-beta: oauth-2025-04-20}
  api_key:
    header: x-api-key
    prefix: ""
oauth_refresh: {token_url: ..., client_id: ...}   # optional
serves:                          # protocol → served path (harness selection)
  anthropic-messages: ""
```

The **delivery** half is the agent's `provider:` binding in
`resources/agents/<name>.yaml`:

```yaml
# resources/agents/claude.yaml
provider:
  default: anthropic             # binds the agent to one provider
  token_env:                     # phantom-token env var, keyed by credential type
    oauth: CLAUDE_CODE_OAUTH_TOKEN
    _default: ANTHROPIC_API_KEY  # fallback for any non-OAuth credential
  base_url_env: ANTHROPIC_BASE_URL   # optional: env var for the vault URL
  socket_env: ANTHROPIC_UNIX_SOCKET  # optional: env var for the vault socket
  credential_file: .credentials.json
  credential_type: oauth
  config_patch: ...              # optional: file patch for the vault address
```

A provider maps to exactly one vault route: the roster loader rejects
a provider bound by more than one agent.  `routes.json` (regenerated
by `terok-executor vault routes` and by `setup`) carries one entry per
provider, keyed by its clean name — which is also what keeps a
provider routable for harnesses even when no agent binds it (the
curated OpenCode providers declare their endpoint plus an `opencode:`
block and have no agent YAML at all).

### Agent-specific settings not in YAML

- **OpenCode base URL override**: For Blablador, KISSKI, and OpenRouter,
  the environment builder sets `TEROK_OC_<NAME>_BASE_URL` when the vault
  is active. This is computed at container launch, not declared in YAML.

- **glab env vars**: `GITLAB_API_HOST` and `API_PROTOCOL=http` are injected
  by the environment builder for glab specifically. glab has no YAML field
  for this because it's a routing concern, not a credential concern.

- **socat bridges**: terok-sandbox's `ensure-bridges.sh` runs in-container.
  In socket mode the broker socket is bind-mounted at
  `/run/terok/vault.sock` and the bridge fronts it as TCP on
  `localhost:9419` for HTTP-only clients.  In TCP mode the bridge is
  reversed: it presents `/tmp/terok-vault.sock` for socket-only clients
  and forwards `localhost:9419` to the per-container host port.

## Auth Flow

### Three auth paths

**1. OAuth / interactive login** (Claude, Codex, gh):
Launches a container with the vendor CLI and a temporary config
directory.  After exit, the extractor captures the OAuth token to the
DB.  Codex additionally offers a headless device-code variant
(`--device-auth`).

**2. API key -- interactive prompt** (Vibe, Blablador, KISSKI,
OpenRouter, glab, CodeRabbit, SonarCloud): Prompts for an API key on
the terminal. No container needed.

**3. API key -- non-interactive** (any provider with an auth flow):
`terok-executor auth <provider> --api-key <key>`

Credentials are stored keyed by the entry's *provider* (`claude` →
`anthropic`, `gh` → `github`), matching the vault route names.

### Post-auth config patching

After storing credentials, `write_vault_config()` applies any
`provider.config_patch` from the YAML registry. This writes vault
addresses (not secrets) to the entry's shared config mount.  Task launch
re-applies enabled patches and removes disabled provider patches using
the managed sidecar (`.terok-managed-config.json`), which keeps global
shared config directories from retaining stale vault routing after a
feature mode changes.

## Per-Provider Credential Extractors

| Entry      | File                | Key fields                    |
|------------|---------------------|-------------------------------|
| Claude     | `.credentials.json` (fallback: `config.json`) | access_token, refresh_token (fallback: api_key) |
| Codex      | `auth.json`         | access_token, refresh_token, id_token, account_id |
| Vibe       | `.env`              | key (MISTRAL_API_KEY)         |
| Blablador  | `config.json`       | key (api_key)                 |
| KISSKI     | `config.json`       | key (api_key)                 |
| gh         | `hosts.yml`         | token (oauth_token)           |
| glab       | `config.yml`        | token (per-host)              |
| OpenRouter | —                   | API key via prompt/`--api-key` |
| CodeRabbit | —                   | API key via prompt/`--api-key` |
| SonarCloud | —                   | API key via prompt/`--api-key` |

## Limitations

- **Claude**: OAuth logins do not support full routing. You can either expose the
  real token to the agent to get the full Claude CLI user experience, or route
  what's officially routable, but miss out on some subscription features.
- **Codex**: ChatGPT/backend-api and realtime websocket traffic are routed
  through the token broker, but any newly added Codex-specific upstream
  surfaces still need explicit vault route coverage.
- **Copilot**: Not proxied yet — no `provider:` binding in its YAML, so no
  vault route and no `auth` flow.

## Package Boundaries

- **[terok-sandbox](https://terok-ai.github.io/terok-sandbox/)**: Credential
  DB, token broker (HTTP/WebSocket forwarding, phantom token resolution,
  OAuth token refresh, SSH signer), socket/TCP listeners, per-container
  supervisor lifecycle
- **terok-executor** (this package): YAML agent + provider registry,
  credential extractors, auth CLI, canonical container environment assembly
  (phantom env vars, base URL overrides, config patches)
- **[terok](https://terok-ai.github.io/terok/)**: Multi-task orchestration on
  top — delegates env assembly here and layers per-project credential sets
  and OAuth exposure tiers
