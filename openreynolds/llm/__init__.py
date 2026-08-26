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
)
from .presets import FALLBACK_CONTEXT_WINDOW, FAMILIES, PRESETS, Preset, family_of, preset_for

__all__ = [
    "BadRequest",
    "FALLBACK_CONTEXT_WINDOW",
    "FAMILIES",
    "Listener",
    "PRESETS",
    "Preset",
    "Provider",
    "ProviderError",
    "TextBlock",
    "ThinkingBlock",
    "ToolCall",
    "ToolUseBlock",
    "Turn",
    "family_of",
    "make_provider",
    "preset_for",
]


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
    seconds = timeout if timeout is not None else getattr(cfg, "llm_timeout_s", None)
    if family == "openai":
        from .openai_api import OpenAIProvider

        return OpenAIProvider(cfg.llm_api_key, base_url, seconds, default_headers)
    from .anthropic_api import AnthropicProvider

    return AnthropicProvider(cfg.llm_api_key, base_url, seconds, default_headers)
