# Security

terok-executor isolates each agent behind four layers: an egress firewall,
a vault, optional restricted mode, and rootless containers.

## Egress firewall

On by default for every container that starts through terok-executor.
The firewall ([terok-shield](https://terok-ai.github.io/terok-shield/))
restricts outbound traffic to explicitly allowed domains — the agent's
API endpoint, package registries, and git hosts.  Everything else is
blocked at the nftables level.

The firewall is attached via OCI hooks at install time (``terok-executor
setup`` / ``terok-sandbox setup``); ``run`` has no per-run opt-out flag.
To loosen it for development, use the live shield verbs — per container:
``terok-executor sandbox shield allow|deny|down|up`` — or remove the
hooks entirely with ``terok-executor sandbox shield uninstall-hooks``.

The git gate mirror (``--gate`` / ``--no-gate``) is a separate concern
from the firewall — see [Launch modes](launch-modes.md) for that flag.

## Vault

No real API keys, OAuth tokens, or SSH private keys enter containers.
Instead, each container receives per-task **phantom tokens**. A host-side
token broker ([terok-sandbox](https://terok-ai.github.io/terok-sandbox/))
resolves phantom tokens to real credentials and forwards requests
upstream over TLS.

SSH keys are handled the same way: a host-side SSH signer lets
containers sign git operations without the private key crossing the
container boundary.

This means a compromised agent cannot read, copy, or exfiltrate real
credentials — they exist only on the host and are never written to
container-accessible mounts.

See [Vault internals](vault.md) for the full
architecture, per-agent routing table, and YAML configuration.

### Managing the vault

The vault is served per container: the supervisor spawns on container
start via the terok-sandbox OCI hook and reads the per-container
sidecar to bind its proxy.  The vault verbs are DB inspection and
passphrase-tier management (owned by terok-sandbox) plus two
executor-only file-level helpers:

```bash
terok-executor vault status      # lock state, passphrase chain, stored secrets
terok-executor vault unlock      # provision the credential-DB passphrase for this session
terok-executor vault lock        # clear every stored copy of the passphrase
terok-executor vault list        # inventory stored credentials
terok-executor vault passphrase  # manage passphrase storage (seal, to-desktop-keyring, …)
terok-executor vault routes      # regenerate routes.json from the YAML roster
terok-executor vault clean       # remove leaked credential files from mounts
```

## Restricted mode

```bash
terok-executor run claude . --restricted -p "…"
```

Disables auto-approve flags and sets `--no-new-privileges` on the
container. Use for untrusted prompts or when the agent should confirm
every action with the user.

## Rootless containers

All containers run under rootless Podman — no daemon, no root privileges.
Combined with SELinux labeling, this limits what a compromised agent can
reach on the host filesystem.


## Host inputs and shared vendor state

The host treats container-writable paths as untrusted. Shared vendor
configuration and transcripts remain shared and writable. Host readers and
writers reject symlinks in any path component, hardlinked files, directories,
FIFOs, and other nonregular entries. Reads are bounded: credential files to
1 MiB and merged vendor configuration to 4 MiB. Writes replace files atomically
rather than modifying an existing inode. Errors identify the unsafe entry;
repair it as an ordinary file rather than linking it to another host path.

| Host task directory | Container location | Access |
| --- | --- | --- |
| `agent-inputs/` | `/home/dev/.terok/input/` | Host-written, container read-only |
| `agent-events/` | `/home/dev/.terok/events/` | Narrow writable acknowledgement channel |
| `agent-config/` | `/home/dev/.terok/` | Writable session pointers and status |

Prepared agent launches validate their structured volume specs before host setup.
The reserved input mount must appear once, use the task's `agent-inputs/` source,
and be read-only and live. Aliases of its destination and mounts beneath it are
rejected. The parent configuration mount and overrides of other `.terok`
subdirectories remain supported. Tools without task inputs do not require this
mount.

Raw Podman arguments, including an orchestrator's `run.podman_args`, remain an
advanced-user escape hatch. Input-mount validation does not parse or reinterpret
them; operators are responsible for their mount overrides. Existing passthrough
policies are unchanged.

Wrappers, instructions, initial and headless prompts, review notices, and the
Toad access token belong in `agent-inputs/`. Read-only credential shadows are
bounded snapshots under `agent-inputs/credential-shadows/`; Podman never binds
a credential file from the writable vendor directory. Swapping the vendor
entry after its safe read cannot redirect the later bind. Exposed credentials
and vendor settings that require writes retain their ordinary shared mount.
Snapshots preserve vendor data; they do not grant broker authorization.
Credential snapshots are read-only bind mounts in shared mode and file overlays
after vendor-directory copies in sealed mode; their host source remains outside
writable mounts in both modes. An initial prompt is immutable.
The wrapper atomically creates the empty `initial-prompt.consumed` marker in
`agent-events/` before passing the prompt to the agent. The banner only displays
pending prompts; it does not claim them. The marker acknowledges delivery, not
successful agent execution. Deleting or forging it can affect delivery but
cannot replace the host-authored prompt.

Prompts are limited to 120 KiB of UTF-8 data, including the appended newline
for initial prompts. This leaves headroom below the Linux per-argument limit:
the existing agent launch paths deliver a prompt as one command-line argument.
Oversized prompts fail before publication or acknowledgement.

Input and event **directory** mounts remain live in sealed mode, so publication
after background startup and prompt acknowledgement are visible immediately.
This is a narrow exception to copied state, alongside the existing live service
channels. Other task state and vendor files remain isolated copies in sealed
mode. Atomic host publication is visible through directory mounts; an individual
file mount would retain an obsolete inode.

### Upgrade requirements

Rebuild the agent and project images and recreate retained task containers.
Restarting cannot replace their old writable-input mounts. Old images and
containers fail with an actionable recreation error. Shared vendor settings and
sessions can be retained, but symlinked or hardlinked host-consumed files must be
replaced with ordinary files. There is no unsafe one-time migration exception.
Protocol checks use strict runtime observations: only a confirmed protocol mismatch
justifies recreation. Failed inspection or unavailable container/image metadata
aborts the operation instead of treating uncertainty as an obsolete task.
The orchestrator must not import legacy agent-writable prompt text as authoritative
host instructions. Authorization sidecars also require the explicit credential
scope, subject, and credential set from token issuance; legacy sidecars are not
assigned a guessed namespace.
