// SPDX-FileCopyrightText: 2026 Jiri Vyskocil
// SPDX-License-Identifier: Apache-2.0
//
// Pi extension that points every built-in provider at terok's vault.
// The vault is a single HTTP endpoint per container; routing to the
// real upstream (api.anthropic.com, api.openai.com, …) happens vault-
// side based on path + auth header.  Pi reads its usual per-provider
// env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, …) for the API key —
// those carry phantom tokens placed by whichever vendor CLIs are
// co-installed (claude.yaml, codex.yaml, vibe.yaml, …).  With baseUrl
// pointing at the vault, the phantom flows through and the vault swaps
// it for the real credential in flight.
//
// Loaded automatically because pi-env.sh symlinks this file into
// ~/.pi/agent/extensions/ on shell startup.

const VAULT_PROVIDERS = [
    "anthropic",
    "openai",
    "mistral",
    "google",
    "google-vertex",
    "groq",
    "cerebras",
    "xai",
    "openrouter",
    "deepseek",
];

export default function (pi) {
    const loopback = process.env.TEROK_VAULT_LOOPBACK_PORT;
    const broker = process.env.TEROK_TOKEN_BROKER_PORT;
    const baseUrl = loopback
        ? `http://localhost:${loopback}`
        : broker
            ? `http://host.containers.internal:${broker}`
            : null;
    // No vault env vars set means no routes are active — direct-API mode.
    // Leave Pi's built-in baseUrls alone; the user can still talk direct
    // to providers if they have real keys in the env.
    if (!baseUrl) return;

    for (const provider of VAULT_PROVIDERS) {
        pi.registerProvider(provider, { baseUrl });
    }
}
