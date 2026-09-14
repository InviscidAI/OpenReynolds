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


PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    # Anthropic first-party API rates, per million tokens. Cache reads are 0.1x input
    # and cache writes 1.25x, so those are derived rather than typed twice.
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    # Aster's rates, read off its own `/v1/models` (which reports them per model) rather
    # than off the marketing page, whose gpt-oss figure disagreed with the API's. Chat
    # Completions never bills a cache *write* -- `openai_api._token_classes` always
    # reports that class as 0 -- so 0.0 here is the true rate, not a missing one.
    "kimi-k3": {"input": 2.50, "output": 12.50, "cache_read": 0.25, "cache_write": 0.0},
    "glm-5.2": {"input": 1.00, "output": 4.00, "cache_read": 0.20, "cache_write": 0.0},
    # gpt-oss publishes no cached rate, so a cached prefix is priced at the full input
    # rate: an unbilled discount we do not know about understates nothing.
    "gpt-oss-120b": {"input": 0.15, "output": 0.60, "cache_read": 0.15, "cache_write": 0.0},
    "gpt-oss-120b-fast": {"input": 0.15, "output": 0.60, "cache_read": 0.15, "cache_write": 0.0},
}
"""What a token costs, **by model**, next to the models themselves.

It used to be one untagged table in `scripts/cad_accept.py` holding Sonnet 5's rates,
because Sonnet 5 is what the default preset below runs. The build-up sweep sets
`mesher_model` to Opus 5, nothing reconciled the two, and every dollar the first baseline
reported was understated by exactly 2.5x -- `$3.00` for a corpus that cost `$7.49`.

Nothing was wrong with the arithmetic and nothing was wrong with the desk. The table was
a fact about one model kept somewhere that did not say which, and the sweep quietly used
it for another. So it lives here, keyed, where a model id is already the unit.

The ranking in that report survives, because one sweep runs one model and a uniform
scalar cannot reorder anything. What did not survive is every absolute figure, and any
comparison between two sweeps whose models differ -- which is exactly what the baseline
chain exists to make."""


def prices(model: str) -> dict[str, float] | None:
    """The rates for a model, or `None` when we do not know them.

    `None` rather than an empty dict on purpose: an empty dict prices a run at zero, and
    a zero that means "unpriced" is indistinguishable from a zero that means "free" --
    which is the shape of the bug this table was moved to fix."""
    return PRICE_PER_MTOK.get((model or "").strip())


def spend(tokens: dict[str, int] | None, model: str) -> float:
    """What a run cost, at this model's rates. Unknown model prices at 0.0.

    Callers that report a number to somebody should refuse an unpriced model up front
    rather than let this return zero -- `scripts/cad_buildup.py` does, beside its key
    check, because a sweep ranks its findings by cost."""
    rates = prices(model) or {}
    return sum(rates.get(name, 0.0) * int(count or 0) / 1e6
               for name, count in (tokens or {}).items())


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
            "aster", "openai", "https://api.asterlab.ai/v1",
            "kimi-k3", "gpt-oss-120b-fast", 1_048_576,
            "ASTER_API_KEY", "Aster's serving stack (Kimi K3, GLM 5.2, gpt-oss) over its "
            "OpenAI-compatible endpoint.",
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
