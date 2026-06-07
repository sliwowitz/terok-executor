// SPDX-FileCopyrightText: 2026 Jiri Vyskocil
// SPDX-License-Identifier: Apache-2.0
//
// Pi extension that registers each terok-materialized provider against the vault.
//
// terok injects, per authenticated + protocol-compatible provider P:
//   TEROK_PROVIDER_<P>_TOKEN             — the phantom bearer
//   TEROK_PROVIDER_<P>_BASE_<PROTOCOL>   — the vault loopback URL, path included
//                                          (e.g. .../v1, .../api/v1)
// For each such P we fetch its model list through the vault and register the
// provider with that base URL + the phantom (as a "$VAR" reference) + the
// fetched models.  Fetching matters because Pi only ships built-in models for a
// handful of providers (openai/mistral/openrouter/...); a provider Pi doesn't
// know (blablador, kisski, …) would otherwise show zero models even with its
// baseUrl set.  On a fetch failure we still register the baseUrl so Pi's
// built-in models for that provider (if any) route through the vault.
//
// Self-contained: must be a ``.ts`` file — Pi only auto-discovers ``*.ts``
// extensions in ~/.pi/agent/extensions/.  Plain ESM is valid TypeScript.

const BASE_VAR = /^TEROK_PROVIDER_(.+)_BASE_(.+)$/;

// terok wire protocol → Pi API type.
const PROTOCOL_API: Record<string, string> = {
    "openai-chat": "openai-completions",
    "openai-responses": "openai-responses",
    "anthropic-messages": "anthropic-messages",
};

// A provider may serve several protocols; pick the one Pi's provider speaks.
const PROTOCOL_PREFERENCE = ["openai-chat", "openai-responses", "anthropic-messages"];

// First array among the candidates — response shapes vary ({data:[]} vs {models:[]}).
const firstArray = (...candidates: unknown[]): any[] =>
    (candidates.find(Array.isArray) as any[]) ?? [];

async function fetchModels(baseUrl: string, token: string | undefined) {
    let base = baseUrl;
    while (base.endsWith("/")) base = base.slice(0, -1);
    try {
        const resp = await fetch(`${base}/models`, {
            headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!resp.ok) return [];
        const payload = await resp.json();
        const items = firstArray(payload?.data, payload?.models);
        return items
            .filter((m: any) => m && typeof m.id === "string")
            .map((m: any) => ({
                id: m.id,
                name: typeof m.name === "string" ? m.name : m.id,
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: typeof m.context_window === "number" ? m.context_window : 128000,
                maxTokens: typeof m.max_tokens === "number" ? m.max_tokens : 4096,
            }));
    } catch {
        return [];
    }
}

export default async function registerProviders(pi: any) {
    const env = process.env;

    // When the wrapper passed an explicit --provider, pi-provider exports it here
    // so we register ONLY that provider — scoping pi's picker to it.  Unset means
    // register every materialized provider (the full picker).
    const only = (env.TEROK_PI_PROVIDER ?? "").toLowerCase();

    // provider name (lower-case) -> { protocol -> vault base URL }
    const basesByProvider: Record<string, Record<string, string>> = {};
    for (const [key, value] of Object.entries(env)) {
        const match = BASE_VAR.exec(key);
        if (!match || !value) continue;
        const provider = match[1].toLowerCase();
        const protocol = match[2].toLowerCase().replaceAll("_", "-");
        basesByProvider[provider] ??= {};
        basesByProvider[provider][protocol] = value;
    }

    for (const [provider, byProtocol] of Object.entries(basesByProvider)) {
        if (only && provider !== only) continue;
        const protocol =
            PROTOCOL_PREFERENCE.find((p) => byProtocol[p]) ?? Object.keys(byProtocol)[0];
        const baseUrl = byProtocol[protocol];
        const tokenVar = `TEROK_PROVIDER_${provider.toUpperCase()}_TOKEN`;
        const config: Record<string, unknown> = {
            baseUrl,
            api: PROTOCOL_API[protocol] ?? "openai-completions",
        };
        if (env[tokenVar]) config.apiKey = `$${tokenVar}`;
        const models = await fetchModels(baseUrl, env[tokenVar]);
        if (models.length) config.models = models;
        pi.registerProvider(provider, config);
    }
}
