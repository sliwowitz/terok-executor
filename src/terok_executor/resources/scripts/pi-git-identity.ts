// SPDX-FileCopyrightText: 2026 Jiri Vyskocil
// SPDX-License-Identifier: Apache-2.0
//
// Pi extension that specializes terok's git author identity to the *live*
// provider/model, without disturbing the human terok records as committer.
//
// terok bakes GIT_AUTHOR_* / GIT_COMMITTER_* into the container env per its
// authorship policy (``agent-human`` → author=agent, committer=human, etc.),
// using the placeholder ``noreply@terok.ai`` for whichever slot it considers
// "the agent".  Pi is multi-provider and the model can change mid-session, so a
// static value can't say *which* backend actually wrote a commit.  This
// extension closes that gap: on every bash spawn it rewrites only the slot(s)
// still holding terok's placeholder, turning
//
//     Pi <noreply@terok.ai>                       (terok's honest placeholder)
//
// into a provider-tagged pair derived from the model in play at commit time
//
//     Pi (claude-opus-4-8) <pi@anthropic>         (anthropic backend)
//     Pi (devstral-2)       <pi@mistral>          (after a mid-session switch)
//
// The ``pi@<provider>`` address is deliberately domain-less: non-routable,
// git-accepted, and honest — it names the provider that did the work rather
// than impersonating a vendor's real ``noreply@`` mailbox.  A slot terok set to
// anything else (the human committer, or a real configured identity) never
// matches the sentinel and is passed through untouched, so terok's
// author/committer policy is preserved across every authorship mode.
//
// Adapted from Christian Tietze's "Make the pi Coding Agent Identify the Model
// in Commits" (https://christiantietze.de/posts/2026/03/pi-coding-agent-git-commit-identify-the-model-audit-trail/),
// narrowed to rewrite only terok's placeholder so the human committer survives.
//
// Self-contained: must be a ``.ts`` file — Pi only auto-discovers ``*.ts``
// extensions in ~/.pi/agent/extensions/.  Plain ESM is valid TypeScript.

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createBashTool } from "@earendil-works/pi-coding-agent";

// The placeholder terok bakes into the "agent" slot of GIT_AUTHOR_* /
// GIT_COMMITTER_*.  Must stay in lockstep with ``git_identity.email`` in
// ``resources/agents/pi.yaml`` — the two are shipped together.
const TEROK_SENTINEL_EMAIL = "noreply@terok.ai";

const GIT_IDENTITY_ROLES = ["GIT_AUTHOR", "GIT_COMMITTER"] as const;

/** Derive Pi's provider-tagged git identity for the active model. */
function piIdentity(provider: string, modelId: string): { name: string; email: string } {
	return { name: `Pi (${modelId})`, email: `pi@${provider}` };
}

/**
 * Return ``env`` with every terok-placeholder identity slot specialized to the
 * live provider/model, and all other slots left exactly as terok set them.
 *
 * Pure and side-effect-free so the rewrite rule can be reasoned about (and
 * tested) without standing up Pi or a container.
 */
export function specializeGitIdentity(
	env: NodeJS.ProcessEnv,
	provider: string,
	modelId: string,
): NodeJS.ProcessEnv {
	const { name, email } = piIdentity(provider, modelId);
	const next = { ...env };
	for (const role of GIT_IDENTITY_ROLES) {
		// Exact-match the placeholder only: an empty, unset, human, or otherwise
		// real identity never equals the sentinel and is left exactly as terok
		// set it.
		if (env[`${role}_EMAIL`] === TEROK_SENTINEL_EMAIL) {
			next[`${role}_NAME`] = name;
			next[`${role}_EMAIL`] = email;
		}
	}
	return next;
}

export default function (pi: ExtensionAPI): void {
	const cwd = process.cwd();
	let provider = "unknown";
	let modelId = "unknown";

	// Seed from the model resolved at startup, then track every switch so a
	// mid-session provider change is reflected in the very next commit.
	pi.on("session_start", async (_event, ctx) => {
		if (ctx.model) {
			provider = ctx.model.provider;
			modelId = ctx.model.id;
		}
	});
	pi.on("model_select", async (event) => {
		provider = event.model.provider;
		modelId = event.model.id;
	});

	// Pi builds each bash subprocess env from a getShellEnv() snapshot and then
	// applies this hook — so the spawn hook, not process.env, is the reliable
	// place to reach the git the agent runs.
	pi.registerTool(
		createBashTool(cwd, {
			spawnHook: ({ command, cwd, env }) => ({
				command,
				cwd,
				env: specializeGitIdentity(env, provider, modelId),
			}),
		}),
	);
}
