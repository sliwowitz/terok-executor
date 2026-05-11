# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary, schema-walker, and frame builders for ACP's model selector.

The model selector lives in ``configOptions[category=model]`` (and,
in modern ACP, mirrored under ``result.models`` of ``session/new``).
Three consumers share this vocabulary: the probe reads it to learn
what an agent can serve, the proxy builds aggregated responses
pre-bind, and the proxy rewrites it post-bind so the client sees
the namespaced ``agent:model`` ids it expects.  All three sit on
top of :func:`iter_model_choice_dicts`, which tolerates the in-flight
ACP schema variants in one place.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from acp import NewSessionResponse
from acp.schema import (
    ModelInfo,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionModelState,
)

MODEL_OPTION_CATEGORY = "model"
"""ACP semantic category for the model selector configOption."""

MODEL_NAMESPACE_SEP = ":"
"""Separator between agent and model in the namespaced id (e.g.
``claude:opus-4.6``).  Chosen over ``/`` to avoid collisions with
OpenRouter-style ids like ``anthropic/claude-opus-4``."""


def iter_model_choice_dicts(result: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield each dict-shaped choice from ``result.configOptions[category=model]``.

    Tolerant of the in-flight ACP schema variants we observed during
    design — the model selector can nest its choices under ``select``
    or directly, and use ``options`` / ``values`` / ``choices`` as the
    container key.  Skips non-dict entries; consumers that need to
    mutate in place can rely on dict-only semantics.
    """
    options = result.get("configOptions") or []
    for opt in options:
        if not isinstance(opt, dict) or opt.get("category") != MODEL_OPTION_CATEGORY:
            continue
        select = opt.get("select")
        nested = select if isinstance(select, dict) else opt
        if not isinstance(nested, dict):
            continue  # type: ignore[unreachable]
        for key in ("options", "values", "choices"):
            choices = nested.get(key)
            if not isinstance(choices, list):
                continue
            for entry in choices:
                if isinstance(entry, dict):
                    yield entry


def _humanise_model_id(namespaced: str) -> str:
    """Render ``claude:opus-4.6`` as ``Claude: opus-4.6`` for the picker.

    Colon (matching the wire-level :data:`MODEL_NAMESPACE_SEP`) keeps
    slashes free for model ids that legitimately contain them, e.g.
    OpenCode's ``opencode:opencode/big-pickle`` — humanising that to
    ``OpenCode: opencode/big-pickle`` reads as one provider plus one
    model id, while using ``/`` as the separator would chop the model
    id in the middle.  We only mutate names *we* synthesise;
    descriptions / labels coming back from real backend agents are
    forwarded verbatim so upstream formatting decisions stand.
    """
    agent, _, model = namespaced.partition(MODEL_NAMESPACE_SEP)
    if not agent or not model:
        return namespaced
    return f"{agent.capitalize()}: {model}"


def _build_model_config_option(
    namespaced_models: list[str],
    *,
    current: str,
) -> SessionConfigOptionSelect:
    """Build a ``category: "model"`` ``select`` configOption.

    Returns the SDK pydantic model (not a dict) so the caller can drop
    it straight into a :class:`NewSessionResponse` /
    :class:`SetSessionConfigOptionResponse`.  ``current`` is required
    by the schema (``current_value`` is a non-nullable ``str``); the
    caller is expected to handle the empty-models case by skipping
    the option entirely rather than passing a placeholder.
    """
    return SessionConfigOptionSelect(
        id="model",
        name="Model",
        type="select",
        description="AI model to use",
        category=MODEL_OPTION_CATEGORY,
        current_value=current,
        options=[
            SessionConfigSelectOption(value=ident, name=_humanise_model_id(ident))
            for ident in namespaced_models
        ],
    )


def _build_session_new_response(session_id: str, models: list[str]) -> NewSessionResponse:
    """Construct the pre-bind ``session/new`` reply for *models*.

    Splits the empty-models case so the response stays schema-valid:
    when no agents probed successfully, the ``models`` block and the
    model ``configOption`` are omitted entirely (both have non-nullable
    required fields the proxy can't fill in for an empty list).  The
    ``modes`` block is also omitted — :class:`SessionModeState` requires
    a non-null ``current_mode_id`` and the proxy doesn't manage modes.
    """
    if not models:
        return NewSessionResponse(session_id=session_id)
    current = models[0]
    return NewSessionResponse(
        session_id=session_id,
        models=SessionModelState(
            available_models=[
                ModelInfo(model_id=ident, name=_humanise_model_id(ident)) for ident in models
            ],
            current_model_id=current,
        ),
        config_options=[_build_model_config_option(models, current=current)],
    )


def _rewrite_model_options_in_place(frame: dict[str, Any], bound_agent: str) -> None:
    """Mutate *frame* so any model ids are namespaced as ``agent:model``.

    Backends emit bare model ids (``opus-4.6``); clients expect
    namespaced ids (``claude:opus-4.6``).  After bind, only the bound
    agent's models should appear; the proxy adds the prefix on every
    place a model id can show up:

    - ``configOptions[category=model].currentValue``
    - ``configOptions[category=model].options[*].id`` / ``.value``
    - ``models.currentModelId``
    - ``models.availableModels[*].modelId``

    The two ``models.*`` paths matter for the post-bind ``session/new``
    reply: claude-agent-acp v0.32+ carries the picker primarily in
    ``result.models``, and Zed reads from there — leaving those bare
    would surface ``opus-4.6`` in the picker even though the client
    has to send ``claude:opus-4.6`` back.
    """
    result = frame.get("result")
    if not isinstance(result, dict):
        return

    def _prefix(value: str) -> str:
        return f"{bound_agent}{MODEL_NAMESPACE_SEP}{value}"

    for opt in result.get("configOptions") or []:
        if not isinstance(opt, dict) or opt.get("category") != MODEL_OPTION_CATEGORY:
            continue
        current = opt.get("currentValue")
        if isinstance(current, str) and MODEL_NAMESPACE_SEP not in current:
            opt["currentValue"] = _prefix(current)
    for entry in iter_model_choice_dicts(result):
        ident = entry.get("id") or entry.get("value")
        if not isinstance(ident, str) or MODEL_NAMESPACE_SEP in ident:
            continue
        prefixed = _prefix(ident)
        if "id" in entry:
            entry["id"] = prefixed
        if "value" in entry:
            entry["value"] = prefixed

    models = result.get("models")
    if isinstance(models, dict):
        current = models.get("currentModelId")
        if isinstance(current, str) and MODEL_NAMESPACE_SEP not in current:
            models["currentModelId"] = _prefix(current)
        for entry in models.get("availableModels") or []:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("modelId")
            if isinstance(mid, str) and MODEL_NAMESPACE_SEP not in mid:
                entry["modelId"] = _prefix(mid)
