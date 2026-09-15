"""Named starting points for bring-your-own-key.

A preset is a convenience, not a contract: it fills in which API family a vendor
speaks, where, and a model id that existed when this file was written. Vendors rename
models; `openreynolds doctor` is what says whether the id still answers, and every
field can be overridden in the environment or the config file. The two families are
the only thing the code actually depends on: `anthropic` (the Messages API, which
several vendors expose as a compatible endpoint) and `openai` (Chat Completions, the
lingua franca of everything else, local models included).
"""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES = ("anthropic", "openai")

FALLBACK_CONTEXT_WINDOW = 200_000
"""Assumed window for a vendor whose model is not in the table. Conservative on
purpose: refreshing a thread early costs a briefing; overrunning a window costs the
turn."""


@dataclass(frozen=True)
class Preset:
    name: str
    family: str
    base_url: str | None
    model: str
    desk_model: str
    context_window: int
    key_env: str
    note: str
    needs_key: bool = True


REYNOLDS = "reynolds"
"""The preset that needs no key of its own: the workspace service proxies Claude and
meters the tokens to the account, so the service key is the model key. The endpoint is
derived from the service URL at run time (`make_provider`), never stored."""

PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            # Sonnet 5 by default: a study is mostly tool calls and re-read context,
            # where Sonnet's 2.5x lower price buys the same work; Opus 5 is a choice
            # (`OPENREYNOLDS_MODEL=claude-opus-5`) for the hard, ambiguous ones.
            REYNOLDS, "anthropic", None,
            "claude-sonnet-5", "claude-haiku-4-5", 1_000_000,
            "", "Reynolds' model: Claude through the workspace service, metered to your account.",
            needs_key=False,
        ),
        Preset(
            "anthropic", "anthropic", None,
            "claude-opus-5", "claude-haiku-4-5", 1_000_000,
            "ANTHROPIC_API_KEY", "Anthropic, directly.",
        ),
        Preset(
            "openai", "openai", None,
            "gpt-5", "gpt-5-mini", 400_000,
            "OPENAI_API_KEY", "OpenAI, directly.",
        ),
        Preset(
            "zai", "anthropic", "https://api.z.ai/api/anthropic",
            "glm-4.6", "glm-4.5-air", 200_000,
            "ZAI_API_KEY", "Z.ai (GLM) through its Anthropic-compatible endpoint.",
        ),
        Preset(
            "deepseek", "anthropic", "https://api.deepseek.com/anthropic",
            "deepseek-chat", "deepseek-chat", 128_000,
            "DEEPSEEK_API_KEY", "DeepSeek through its Anthropic-compatible endpoint.",
        ),
        Preset(
            "moonshot", "anthropic", "https://api.moonshot.ai/anthropic",
            "kimi-k2-thinking", "kimi-k2-turbo-preview", 256_000,
            "MOONSHOT_API_KEY", "Moonshot (Kimi) through its Anthropic-compatible endpoint.",
        ),
        Preset(
            "minimax", "anthropic", "https://api.minimax.io/anthropic",
            "MiniMax-M2", "MiniMax-M2", 200_000,
            "MINIMAX_API_KEY", "MiniMax through its Anthropic-compatible endpoint.",
        ),
        Preset(
            "openrouter", "openai", "https://openrouter.ai/api/v1",
            "anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5", 200_000,
            "OPENROUTER_API_KEY", "OpenRouter: any model it lists, one key.",
        ),
        Preset(
            "ollama", "openai", "http://localhost:11434/v1",
            "qwen3", "qwen3", 32_000,
            "OLLAMA_API_KEY", "A model running on this machine. No key needed.",
            needs_key=False,
        ),
    )
}


EFFORTS = ("low", "medium", "high")
"""The reasoning efforts a session can be set to (`--effort`, `/effort`).

The same three the hosted app offers (`reynolds_app/sessions.py`). Sent as the
Messages API's `output_config.effort` and as Chat Completions' `reasoning_effort`, both
of which accept these words."""

KNOWN_MODELS: dict[str, tuple[str, ...]] = {
    # The two the workspace service meters (`reynolds_app/sessions.py` REYNOLDS_MODELS).
    REYNOLDS: ("claude-sonnet-5", "claude-opus-5"),
    "anthropic": ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"),
}
"""Models worth offering at `/model` beyond a preset's own two. Like the presets, a
list of ids that existed when it was written, not a limit: any id the vendor answers
to can be typed, and the probe is what decides."""


MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Smaller than the window of the preset that offers them, so a switch to one of
    # these on the same provider must not keep the preset's window.
    "claude-haiku-4-5": 200_000,
    "anthropic/claude-haiku-4.5": 200_000,
}
"""Context windows of models whose window differs from their preset's. A model not
here is taken to have its provider's configured window."""


def context_window_for(model: str) -> int | None:
    """The window known for this model id, or None when only the provider's is known."""
    return MODEL_CONTEXT_WINDOWS.get((model or "").strip())


def models_for(provider: str) -> tuple[str, ...]:
    """The model ids known for a provider, its default first. Empty for a bare family.

    Where `KNOWN_MODELS` names a provider's models, that list is the whole of it: the
    `reynolds` service meters two models, and its desk model is not one to switch to.
    Elsewhere the preset's own two are offered."""
    preset = preset_for(provider)
    if preset is None:
        return ()
    listed = KNOWN_MODELS.get(preset.name)
    out: list[str] = []
    for model in (preset.model, *(listed if listed is not None else (preset.desk_model,))):
        if model not in out:
            out.append(model)
    return tuple(out)


def preset_for(name: str) -> Preset | None:
    return PRESETS.get((name or "").strip().lower())


def family_of(provider: str) -> str:
    """Which API family a `provider` setting means.

    The setting may be a preset name or a bare family. Anything else is an error the
    caller should surface as a list of what is accepted.
    """
    key = (provider or "anthropic").strip().lower()
    preset = PRESETS.get(key)
    if preset is not None:
        return preset.family
    if key in FAMILIES:
        return key
    raise ValueError(
        f"unknown provider {provider!r}; one of {', '.join(sorted(PRESETS))} "
        f"or a family ({', '.join(FAMILIES)}) with OPENREYNOLDS_LLM_BASE_URL set"
    )
