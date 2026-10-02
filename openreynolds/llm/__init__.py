"""Bring your own model.

The loop asks for a `Provider` and never learns which one it got. `make_provider`
reads the family, key and endpoint from the configuration and hands back the right
adapter; `PRESETS` is the list a person picks from at `openreynolds config`.
"""

from __future__ import annotations

from typing import Any

from .base import (
    BadRequest,
    Listener,
    Provider,
    ProviderError,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolUseBlock,
    Turn,
    refusal_line,
)
from .presets import (
    FALLBACK_CONTEXT_WINDOW,
    FAMILIES,
    PRESETS,
    REYNOLDS,
    Preset,
    fallbacks_for,
    family_for,
    family_of,
    preset_for,
)

__all__ = [
    "BadRequest",
    "FALLBACK_CONTEXT_WINDOW",
    "FAMILIES",
    "Listener",
    "PRESETS",
    "REYNOLDS",
    "Preset",
    "Provider",
    "ProviderError",
    "TextBlock",
    "ThinkingBlock",
    "ToolCall",
    "ToolUseBlock",
    "Turn",
    "endpoint_key",
    "fallback_models",
    "family_for",
    "family_of",
    "make_provider",
    "preset_for",
    "refusal_line",
]

_FALLBACKS_OFF = frozenset({"0", "off", "none", "false", "no"})


def fallback_models(cfg: Any, model: str = "") -> tuple[str, ...]:
    """The fallback chain for this configuration: `cfg.fallback_models` when it names
    one (`OPENREYNOLDS_FALLBACK_MODELS`, comma-separated; `0`/`off` for none at all),
    otherwise the preset's for `model`'s family (`presets.fallbacks_for`)."""
    raw = str(getattr(cfg, "fallback_models", "") or "").strip()
    if raw.lower() in _FALLBACKS_OFF:
        return ()
    provider = getattr(cfg, "provider", "") or ""
    model = model or getattr(cfg, "model", "") or ""
    if raw:
        named = tuple(m.strip() for m in raw.split(",") if m.strip())
        preset = preset_for(provider)
        if preset is not None and preset.name == REYNOLDS:
            # One provider speaks one family's API, so a chain naming both families
            # keeps only the half this provider can send.
            family = family_for(provider, model)
            named = tuple(m for m in named if family_for(provider, m) == family)
        return named
    return fallbacks_for(provider, model)


def endpoint_key(cfg: Any, model: str) -> tuple:
    """What a provider built for `model` under `cfg` depends on: when this changes, the
    client has to be rebuilt. The provider, key and endpoint, and -- because `reynolds`
    picks its API by model -- the family that model is spoken in."""
    provider = getattr(cfg, "provider", "") or ""
    try:
        family = family_for(provider, model)
    except ValueError:  # not a provider at all; `make_provider` is what says so
        family = ""
    return (provider, getattr(cfg, "llm_api_key", ""), getattr(cfg, "llm_base_url", None),
            family)


def make_provider(
    cfg: Any,
    *,
    model: str | None = None,
    timeout: float | None = None,
    default_headers: dict[str, str] | None = None,
) -> Provider:
    """The adapter for `cfg.provider`, pointed at `cfg.llm_base_url` with `cfg.llm_api_key`.

    A preset without an explicit base URL supplies its own; a bare family without one
    means the vendor's default endpoint (Anthropic, OpenAI).

    `model` is the one the provider will be asked for, `cfg.model` when not named. It
    matters on `reynolds` only, where it picks the API: the front desk, the CAD desk and
    the reviewer may each run a model other than the study's, and each is built for its
    own. The fallback chain is that model's family's.
    """
    model = model if model is not None else (getattr(cfg, "model", "") or "")
    family = family_for(cfg.provider, model)
    preset = preset_for(cfg.provider)
    base_url = cfg.llm_base_url or (preset.base_url if preset else None)
    api_key = cfg.llm_api_key
    if preset is not None and preset.name == REYNOLDS:
        # The workspace service fronts the model: same address, same key, and the
        # tokens land on the account's ledger next to the compute. Always -- a model
        # key left in the config from a bring-your-own setup must not be sent to the
        # service, which would (rightly) refuse it. Which of its two routes depends on
        # the model: Claude over the Messages API at `/v1/llm` (the Anthropic client
        # appends `/v1/messages`, and sends the key as `x-api-key`), GPT over the
        # Responses API at `/v1/llm/v1` (the OpenAI client appends `/responses`, and
        # sends the key as a Bearer token, which the service takes as readily).
        root = f"{cfg.foamd_url.rstrip('/')}/v1/llm"
        base_url = root if family == "anthropic" else f"{root}/v1"
        api_key = cfg.foamd_api_key
    seconds = timeout if timeout is not None else getattr(cfg, "llm_timeout_s", None)
    provider: Provider
    if family == "openai-responses":
        from .responses_api import ResponsesProvider

        provider = ResponsesProvider(api_key, base_url, seconds, default_headers)
    elif family == "openai":
        from .openai_api import OpenAIProvider

        provider = OpenAIProvider(api_key, base_url, seconds, default_headers)
    else:
        from .anthropic_api import AnthropicProvider

        provider = AnthropicProvider(api_key, base_url, seconds, default_headers)
    # A property of every provider built here -- the main loop's, the CAD desk's, the
    # reviewer's, the front desk's -- so no request path is left without it.
    provider.fallbacks = fallback_models(cfg, model)
    return provider
