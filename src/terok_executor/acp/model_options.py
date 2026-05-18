# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Aggregate and namespace ACP model selectors for the host proxy.

The proxy hides multiple in-container agents behind a single ACP
endpoint by namespacing each agent's model ids as ``agent:model``.
Three operations live here, layered on top of the ACP SDK's pydantic
models:

- [`build_aggregated_session_new`][terok_executor.acp.model_options.build_aggregated_session_new]
  — the pre-bind ``session/new`` reply that advertises the union of
  every authenticated agent's models under one selector.
- [`build_model_option`][terok_executor.acp.model_options.build_model_option]
  — the ``configOptions[category=model]`` entry that mirrors the
  aggregate selector for clients that read it instead of ``models``.
- [`namespace_model_options_in_place`][terok_executor.acp.model_options.namespace_model_options_in_place]
  — the post-bind rewrite that puts the ``agent:`` prefix back on the
  bare model ids a bound backend emits in its own config options.

Plus the small vocabulary helpers ([`split_namespaced`][terok_executor.acp.model_options.split_namespaced],
[`humanise_model_id`][terok_executor.acp.model_options.humanise_model_id])
that callers across the proxy share.
"""

from __future__ import annotations

from acp import NewSessionResponse
from acp.schema import (
    ModelInfo,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SessionModelState,
)

MODEL_OPTION_CATEGORY = "model"
"""ACP semantic category for the model selector configOption.

Used as both the ``category`` and the ``id`` field of the
`acp.schema.SessionConfigOptionSelect` we
build — keeping them in sync prevents drift between the discriminator
the proxy emits and the one downstream code matches on.
"""

MODEL_NAMESPACE_SEP = ":"
"""Separator between agent and model in the namespaced id (e.g.
``claude:opus-4.6``).  Chosen over ``/`` to avoid collisions with
OpenRouter-style ids like ``anthropic/claude-opus-4``."""


def build_aggregated_session_new(session_id: str, models: list[str]) -> NewSessionResponse:
    """Build the pre-bind ``session/new`` reply for *models*.

    Empty *models* yields a schema-valid response with no ``models`` or
    ``configOptions`` block — both have non-nullable required fields the
    proxy can't fill in for an empty list, and modelling "no models" as
    an empty selector trips client validation.
    """
    if not models:
        return NewSessionResponse(session_id=session_id)
    current = models[0]
    return NewSessionResponse(
        session_id=session_id,
        models=SessionModelState(
            available_models=[
                ModelInfo(model_id=ident, name=humanise_model_id(ident)) for ident in models
            ],
            current_model_id=current,
        ),
        config_options=[build_model_option(models, current=current)],
    )


def build_model_option(namespaced_models: list[str], *, current: str) -> SessionConfigOptionSelect:
    """Build a ``category: "model"`` select option with namespaced ids."""
    return SessionConfigOptionSelect(
        id=MODEL_OPTION_CATEGORY,
        name="Model",
        type="select",
        description="AI model to use",
        category=MODEL_OPTION_CATEGORY,
        current_value=current,
        options=[
            SessionConfigSelectOption(value=ident, name=humanise_model_id(ident))
            for ident in namespaced_models
        ],
    )


def namespace_model_options_in_place(
    config_options: list[SessionConfigOptionSelect | SessionConfigOptionBoolean] | None,
    bound_agent: str,
) -> None:
    """Prefix bare model ids in *config_options* with ``bound_agent:``.

    Used on every backend → client frame that carries
    ``configOptions[*]`` post-bind (``ConfigOptionUpdate`` notification,
    ``SetSessionConfigOptionResponse``).  Already-namespaced values are
    left alone so the function is idempotent — paths the proxy itself
    constructed (e.g. ack of a model pick) round-trip cleanly.
    """
    if not config_options or not bound_agent:
        return
    prefix = f"{bound_agent}{MODEL_NAMESPACE_SEP}"
    for opt in config_options:
        if isinstance(opt, SessionConfigOptionSelect) and opt.category == MODEL_OPTION_CATEGORY:
            _namespace_select_in_place(opt, prefix)


def _namespace_select_in_place(opt: SessionConfigOptionSelect, prefix: str) -> None:
    """Apply *prefix* to the select's current value and every option value.

    Idempotency uses ``startswith(prefix)`` rather than "contains a colon" —
    backend model ids that legitimately carry colons (e.g. ``azure:gpt-4.1``)
    must still be prefixed; a bare ``:``-test would mis-classify them as
    already-namespaced and let bare ids leak back to the client.
    """
    if not opt.current_value.startswith(prefix):
        opt.current_value = prefix + opt.current_value
    for entry in opt.options:
        if isinstance(entry, SessionConfigSelectOption):
            _maybe_prefix(entry, prefix)
        elif isinstance(entry, SessionConfigSelectGroup):
            for sub in entry.options:
                _maybe_prefix(sub, prefix)


def _maybe_prefix(entry: SessionConfigSelectOption, prefix: str) -> None:
    """Add *prefix* to ``entry.value`` unless it already carries it."""
    if not entry.value.startswith(prefix):
        entry.value = prefix + entry.value


def split_namespaced(namespaced: str) -> tuple[str, str]:
    """Split ``agent:model`` into ``(agent, model)``.

    Empty halves signal a malformed id — callers that need to validate
    do so by checking both halves.  Centralises the partition so the
    proxy's bind/lookup paths stop spelling
    ``.partition(MODEL_NAMESPACE_SEP)`` themselves.
    """
    agent, _, model = namespaced.partition(MODEL_NAMESPACE_SEP)
    return agent, model


def humanise_model_id(namespaced: str) -> str:
    """Render ``claude:opus-4.6`` as ``Claude: opus-4.6`` for the picker.

    Colon matches the wire-level [`MODEL_NAMESPACE_SEP`][terok_executor.acp.model_options.MODEL_NAMESPACE_SEP]
    so an OpenCode-style ``opencode:opencode/big-pickle`` reads as one
    provider plus one slash-bearing model id.  Forwards verbatim if the
    input isn't a namespaced pair.
    """
    agent, model = split_namespaced(namespaced)
    if not agent or not model:
        return namespaced
    return f"{agent.capitalize()}: {model}"
