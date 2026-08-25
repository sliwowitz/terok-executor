// SPDX-FileCopyrightText: 2026 Jiri Vyskocil
// SPDX-License-Identifier: Apache-2.0
//
// Pi extension that registers each terok-materialized provider against the vault.
//
// terok injects, per authenticated + protocol-compatible provider P:
//   TEROK_PROVIDER_<P>_TOKEN             — the phantom bearer
//   TEROK_PROVIDER_<P>_BASE_<PROTOCOL>   — the vault loopback URL, path included
//                                          (e.g. .../v1, .../api/v1)
//   TEROK_PROVIDER_<P>_LABEL             — optional human-readable provider name
//   TEROK_PROVIDER_<P>_MODELS            — provider-neutral static model data
//   TEROK_PROVIDER_<P>_DEFAULT_MODEL     — optional model preferred by the provider
// For each P, register declared models if they are available. Otherwise, request
// the model list through the vault. Declared data supports a provider that has
// no /models endpoint. It also keeps limits that discovery does not return. If
// the request fails, register baseUrl. Pi can then route its built-in models for
// that provider through the vault.
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

const positiveInteger = (value: unknown): number | undefined =>
    typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;

const piModel = (id: string, metadata: Record<string, unknown>) => ({
    id,
    name: typeof metadata.name === "string" && metadata.name ? metadata.name : id,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: positiveInteger(metadata.context_limit) ?? 128000,
    maxTokens: positiveInteger(metadata.output_limit) ?? 4096,
});

const preferredFirst = <T extends { id: string }>(models: T[], preferred?: string): T[] =>
    preferred
        ? [...models].sort(
              (left, right) => Number(right.id === preferred) - Number(left.id === preferred),
          )
        : models;

function declaredModels(provider: string, env: Record<string, string | undefined>) {
    const handle = provider.toUpperCase();
    const raw = env[`TEROK_PROVIDER_${handle}_MODELS`];
    if (!raw) return [];

    let payload: unknown;
    try {
        payload = JSON.parse(raw);
    } catch {
        return [];
    }
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return [];

    const entries = Object.entries(payload as Record<string, unknown>).filter(
        (entry): entry is [string, Record<string, unknown>] =>
            Boolean(entry[0]) &&
            Boolean(entry[1]) &&
            typeof entry[1] === "object" &&
            !Array.isArray(entry[1]),
    );
    return preferredFirst(
        entries.map(([id, metadata]) => piModel(id, metadata)),
        env[`TEROK_PROVIDER_${handle}_DEFAULT_MODEL`],
    );
}

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
            .map((m: any) =>
                piModel(m.id, {
                    name: m.name,
                    context_limit: m.context_window,
                    output_limit: m.max_tokens,
                }),
            );
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
        const providerVar = `TEROK_PROVIDER_${provider.toUpperCase()}`;
        const config: Record<string, unknown> = {
            baseUrl,
            api: PROTOCOL_API[protocol] ?? "openai-completions",
        };
        if (env[`${providerVar}_LABEL`]) config.name = env[`${providerVar}_LABEL`];
        if (env[tokenVar]) config.apiKey = `$${tokenVar}`;
        const declared = declaredModels(provider, env);
        const models = declared.length
            ? declared
            : preferredFirst(
                  await fetchModels(baseUrl, env[tokenVar]),
                  env[`${providerVar}_DEFAULT_MODEL`],
              );
        if (models.length) config.models = models;
        pi.registerProvider(provider, config);
    }
}
