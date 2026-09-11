# Changelog
## v0.4.0 — Past Prologue

## What's Changed
* fix(acp): migrate to agent-client-protocol 0.11 (ACP schema 1.16) in https://github.com/terok-ai/terok-executor/pull/460
* fix(vault): stop flagging glab's settings-only config.yml as a leaked credential in https://github.com/terok-ai/terok-executor/pull/464
* feat(instructions): render the bundled default per package family in https://github.com/terok-ai/terok-executor/pull/465
* feat(gpu): device-grant selectors pass through (sandbox gpu-device-grants chain) in https://github.com/terok-ai/terok-executor/pull/479
* feat(tmux): show the host-written review-lag warning in the container status line in https://github.com/terok-ai/terok-executor/pull/489
* feat!: simplify custom LLM providers in https://github.com/terok-ai/terok-executor/pull/525
* fix(l0): git over HTTP/1.1 on Ubuntu 24.04 bases in https://github.com/terok-ai/terok-executor/pull/538
* fix: route Codex Apps authentication through vault in https://github.com/terok-ai/terok-executor/pull/544


**Full Changelog**: https://github.com/terok-ai/terok-executor/compare/v0.3.1...v0.4.0

## v0.3.1 — You Exist Here

* Shared credential files permissions for glab, https://github.com/terok-ai/terok-executor/pull/437
* Resilient gate restart after upgrade, https://github.com/terok-ai/terok-executor/pull/440

**Full Changelog**: https://github.com/terok-ai/terok-executor/compare/v0.3.0...v0.3.1

## v0.3.0 — The Celestial Temple

* Codex device-code login, https://github.com/terok-ai/terok-executor/pull/410
* Mount glab's config writable, https://github.com/terok-ai/terok-executor/pull/412
* Dim unusable agents in `hilfe`, list candidate providers by protocol, https://github.com/terok-ai/terok-executor/pull/411
* Container workspace default, and start/stop/rm verbs, https://github.com/terok-ai/terok-executor/pull/421
* API-key extractor for OpenRouter, https://github.com/terok-ai/terok-executor/pull/428
* Expose --passphrase-tier and declare the setup invocation, https://github.com/terok-ai/terok-executor/pull/429

**Full Changelog**: https://github.com/terok-ai/terok-executor/compare/v0.2.1...v0.3.0

## v0.2.2 — We are constantly searching

## What's Changed
* Codex device-code login UI fix, https://github.com/terok-ai/terok-executor/pull/410
* Fix `glab` config permissions, https://github.com/terok-ai/terok-executor/pull/412
* Dim unusable agents, list candidate providers by protocol in `hilfe`, https://github.com/terok-ai/terok-executor/pull/411

**Full Changelog**: https://github.com/terok-ai/terok-executor/compare/v0.2.1...v0.2.2

## v0.2.1 — Locks and Hooks

Hotfix for supervisor restart [#406](https://github.com/terok-ai/terok-executor/pull/406)

**Full Changelog**: https://github.com/terok-ai/terok-executor/compare/v0.2.0...v0.2.1

## v0.2.0 — Emissary, Part II

## What's Changed
* Jinja-templated agent wrappers + instruction-injection, https://github.com/terok-ai/terok-executor/pull/392
* Suppor LLM providers × agents combinations, https://github.com/terok-ai/terok-executor/pull/393, https://github.com/terok-ai/terok-executor/pull/396
* Provider-aware Pi git identity, https://github.com/terok-ai/terok-executor/pull/397
* terok-sandbox vault lock and SSH routing, https://github.com/terok-ai/terok-executor/pull/403, https://github.com/terok-ai/terok-executor/pull/402

**Full Changelog**: https://github.com/terok-ai/terok-executor/compare/v0.1.0...v0.2.0

## v0.1.0 — The Emissary

**First public PyPi release**

## What's Changed

* rewrite proxy + probe on top of typed ACP SDK, https://github.com/terok-ai/terok-executor/pull/348
* add Pi coding agent to the roster, https://github.com/terok-ai/terok-executor/pull/370
* seed initial prompt through the right argv surface per provider, https://github.com/terok-ai/terok-executor/pull/380
* per-container supervisor sidecar; drop --root install path, https://github.com/terok-ai/terok-executor/pull/384


**Full Changelog**: https://github.com/terok-ai/terok-executor/compare/v0.0.148...v0.1.0

