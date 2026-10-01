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
    "fallback_models",
    "family_of",
    "make_provider",
    "preset_for",
    "refusal_line",
]

_FALLBACKS_OFF = frozenset({"0", "off", "none", "false", "no"})


def fallback_models(cfg: Any) -> tuple[str, ...]:
    """The fallback chain for this configuration: `cfg.fallback_models` when it names
    one (`OPENREYNOLDS_FALLBACK_MODELS`, comma-separated; `0`/`off` for none at all),
    otherwise the preset's (`presets.fallbacks_for`)."""
    raw = str(getattr(cfg, "fallback_models", "") or "").strip()
    if raw.lower() in _FALLBACKS_OFF:
        return ()
    if raw:
        return tuple(m.strip() for m in raw.split(",") if m.strip())
    return fallbacks_for(getattr(cfg, "provider", "") or "")


def make_provider(
    cfg: Any,
    *,
    timeout: float | None = None,
    default_headers: dict[str, str] | None = None,
) -> Provider:
    """The adapter for `cfg.provider`, pointed at `cfg.llm_base_url` with `cfg.llm_api_key`.

    A preset without an explicit base URL supplies its own; a bare family without one
    means the vendor's default endpoint (Anthropic, OpenAI).
    """
    family = family_of(cfg.provider)
    preset = preset_for(cfg.provider)
    base_url = cfg.llm_base_url or (preset.base_url if preset else None)
    api_key = cfg.llm_api_key
    if preset is not None and preset.name == REYNOLDS:
        # The workspace service fronts the model: same address, same key, and the
        # tokens land on the account's ledger next to the compute. Always -- a model
        # key left in the config from a bring-your-own setup must not be sent to the
        # service, which would (rightly) refuse it.
        base_url = f"{cfg.foamd_url.rstrip('/')}/v1/llm"
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
    provider.fallbacks = fallback_models(cfg)
    return provider
