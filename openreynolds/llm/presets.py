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

FAMILIES = ("anthropic", "openai", "openai-responses")

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
    # The 5.5 generation, Anthropic's list (claude-api reference, 2026-09-25). Opus 5.5
    # is cheaper than Opus 5 and its cache read is 0.05x input, not the usual 0.1x, so
    # it is typed rather than derived. These two are what `reynolds` serves for Claude;
    # the service relays them to Amazon Bedrock, which bills Claude at partner prices
    # that may differ from these, and the service meters at its own rates besides --
    # so this is what a run costs at Anthropic's list, not what the ledger says.
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    # Aster's rates, read off its own `/v1/models` (which reports them per model) rather
    # than off the marketing page, whose gpt-oss figure disagreed with the API's. Chat
    # Completions never bills a cache *write* -- `openai_api._token_classes` always
    # reports that class as 0 -- so 0.0 here is the true rate, not a missing one.
    # Keyed by vendor as well as model, because a model id does not determine a price:
    # the same open weights cost different amounts at different vendors, and kimi-k3 is
    # 20% dearer at Moonshot than at Aster. An unqualified `kimi-k3` is deliberately not
    # here -- "some vendor's kimi-k3" has no price, and `prices()` returning None is what
    # stops a run that cannot say what it cost.
    "aster:kimi-k3": {"input": 2.50, "output": 12.50, "cache_read": 0.25, "cache_write": 0.0},
    "moonshot:kimi-k3": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 0.0},
    "glm-5.2": {"input": 1.00, "output": 4.00, "cache_read": 0.20, "cache_write": 0.0},
    # gpt-oss publishes no cached rate, so a cached prefix is priced at the full input
    # rate: an unbilled discount we do not know about understates nothing.
    "gpt-oss-120b": {"input": 0.15, "output": 0.60, "cache_read": 0.15, "cache_write": 0.0},
    "gpt-oss-120b-fast": {"input": 0.15, "output": 0.60, "cache_read": 0.15, "cache_write": 0.0},
    # OpenAI's own rates, standard tier. Chat Completions bills no cache write either.
    # `gpt-5.6-sol` is reachable only through `openai-responses` (ResponsesProvider):
    # OpenAI refuses function tools together with a reasoning effort on
    # /v1/chat/completions for every model above gpt-5.2. Its cache write is left at 0
    # as the sweeps priced it, though GPT-5.6 and later do bill one (see below).
    "gpt-5.6-sol": {"input": 4.00, "output": 20.00, "cache_read": 0.40, "cache_write": 0.0},
    # The Responses-API generation, list prices from OpenAI's pricing page (checked
    # 2026-10-02). These bill a cache write, at 1.25x input -- OpenAI does for GPT-5.6
    # and later; the gpt-5.6-sol row above predates knowing that and is left as the
    # sweeps priced it -- and `responses_api._token_classes` reports the class when the
    # usage carries it. These three are what `reynolds` serves for GPT; the service
    # meters at its own rates, so these are what a run costs at list, not what the
    # ledger says.
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00, "cache_read": 0.10, "cache_write": 2.50},
    "gpt-6-astra": {"input": 10.00, "output": 50.00, "cache_read": 1.00, "cache_write": 12.50},
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "gpt-5.2": {"input": 1.75, "output": 14.00, "cache_read": 0.175, "cache_write": 0.0},
    "gpt-5.1": {"input": 1.25, "output": 10.00, "cache_read": 0.125, "cache_write": 0.0},
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


def prices(model: str, provider: str = "") -> dict[str, float] | None:
    """The rates for a model, or `None` when we do not know them.

    `None` rather than an empty dict on purpose: an empty dict prices a run at zero, and
    a zero that means "unpriced" is indistinguishable from a zero that means "free" --
    which is the shape of the bug this table was moved to fix."""
    key = (model or "").strip()
    who = (provider or "").strip().lower()
    if who:
        qualified = PRICE_PER_MTOK.get(f"{who}:{key}")
        if qualified is not None:
            return qualified
    return PRICE_PER_MTOK.get(key)


def spend(tokens: dict[str, int] | None, model: str, provider: str = "") -> float:
    """What a run cost, at this model's rates. Unknown model prices at 0.0.

    Callers that report a number to somebody should refuse an unpriced model up front
    rather than let this return zero -- `scripts/cad_buildup.py` does, beside its key
    check, because a sweep ranks its findings by cost."""
    rates = prices(model, provider) or {}
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
    cad_model: str = ""
    """The model the CAD desk builds geometry with when none is configured. Empty
    means the main model."""


REYNOLDS = "reynolds"
"""The preset that needs no key of its own: the workspace service fronts two model
families and meters the tokens to the account, so the service key is the model key.
GPT is spoken over the Responses API at `{service}/v1/llm/v1`, Claude over the
Messages API at `{service}/v1/llm`, and which one a request goes to is decided by the
model it names (`family_for`), not by the preset. The endpoints are derived from the
service URL at run time (`make_provider`), never stored. Which cloud the service relays
each family to is its business; the agent sees one OpenAI-shaped and one
Anthropic-shaped endpoint either way."""

PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            # The family here is the default model's. A Claude id on this preset is
            # served over the Messages API instead (`family_for`), with Haiku at the
            # desk (`desk_model_for`) and its own window (`MODEL_CONTEXT_WINDOWS`).
            REYNOLDS, "openai-responses", None,
            "gpt-6.1-sol", "gpt-6-luna",
            # Not the model's own window. Above 272K input tokens these models bill at a
            # long-context rate the service's rate table does not carry, so a thread is
            # compacted before it crosses that line rather than metered at a price
            # nobody wrote down.
            272_000,
            "", "Reynolds' model: GPT or Claude through the workspace service, metered to "
            "your account.",
            needs_key=False,
            # Empty: the CAD desk builds with the main model, whichever family it is.
            # On GPT that is measured: gpt-6.1-sol had 22 of 26 survive the mesh vet
            # (`docs/cad-buildup/sweeps/core-gpt-6.1-sol-20261002-075149-9301`, $3.05),
            # against gpt-5.6-sol's 16 of 26. Neither Claude 5.5 model has been swept on
            # the desk; Sonnet 5 finished 0 of 16 in its 30 steps, so a Sonnet 5.5 study
            # building on Sonnet 5.5 is a default, not a finding.
            cad_model="",
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
            # Required, not preferred: OpenAI refuses function tools beside a reasoning
            # effort on Chat Completions for every model above gpt-5.2, so the newer
            # models are reachable only here. `desk_model` stays on a Chat Completions
            # generation because the front desk asks for no tools and no reasoning.
            "openai-responses", "openai-responses", None,
            "gpt-5.6-sol", "gpt-5.2", 400_000,
            "OPENAI_API_KEY", "OpenAI through the Responses API: tools with reasoning.",
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


EFFORTS = ("low", "medium", "high")
"""The reasoning efforts a session can be set to (`--effort`, `/effort`).

The same three the hosted app offers (`reynolds_app/sessions.py`). Sent as the
Messages API's `output_config.effort`, as Chat Completions' `reasoning_effort` and as
the Responses API's `reasoning.effort`, all of which accept these words."""

KNOWN_MODELS: dict[str, tuple[str, ...]] = {
    # The four the workspace service meters for a study (`reynolds_app/sessions.py`
    # REYNOLDS_MODELS), two per family; the default first.
    REYNOLDS: ("gpt-6.1-sol", "gpt-6-astra", "claude-sonnet-5-5", "claude-opus-5-5"),
    "anthropic": ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"),
}
"""Models worth offering at `/model` beyond a preset's own two. Like the presets, a
list of ids that existed when it was written, not a limit: any id the vendor answers
to can be typed, and the probe is what decides."""


MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Smaller than the window of the preset that offers them, so a switch to one of
    # these on the same provider must not keep the preset's window.
    "claude-haiku-4-5": 200_000,
    # GPT's window as `reynolds` uses it: the line above which these bill at a
    # long-context rate the service does not price (see the preset), not the models'
    # own. Listed per model so that a switch on `reynolds` from Claude back to GPT
    # takes it again rather than keeping Claude's million.
    "gpt-6.1-sol": 272_000,
    "gpt-6-astra": 272_000,
    "gpt-6-luna": 272_000,
    # Larger than the preset's, which is GPT's: the Claude 5.5 models hold 1M.
    "claude-opus-5-5": 1_000_000,
    "claude-sonnet-5-5": 1_000_000,
    "anthropic/claude-haiku-4.5": 200_000,
}
"""Context windows of models whose window differs from their preset's. A model not
here is taken to have its provider's configured window."""


def context_window_for(model: str) -> int | None:
    """The window known for this model id, or None when only the provider's is known."""
    return MODEL_CONTEXT_WINDOWS.get((model or "").strip())


def fallbacks_for(provider: str, model: str = "") -> tuple[str, ...]:
    """The models a refused request on `provider` for `model` is retried on, in order.

    Only the models the provider is known to serve (`KNOWN_MODELS`), and on `reynolds`
    only those of `model`'s family (the default's when none is named): gpt-6.1-sol and
    gpt-6-astra fall to each other, claude-sonnet-5-5 and claude-opus-5-5 likewise, and
    never across -- a provider is built for one family's endpoint, and a Claude id sent
    to the Responses endpoint is a 4xx, not a second opinion. A direct Anthropic key
    falls between Opus and Sonnet. The desk model is never one of
    them: a desk model is picked for being cheap and quick, not for holding a study's
    thread, and a 400 for an oversized request is a worse answer than the refusal it
    replaced. A preset with no list has no fallback, because guessing a second model id
    at a vendor we know one id for is how a session ends on a 404 instead.
    `OPENREYNOLDS_FALLBACK_MODELS` names a chain of its own.

    The requested model is left in: `Provider.chain` takes it out at request time, so
    the same tuple serves whichever of the two is the session's model."""
    preset = preset_for(provider)
    if preset is None:
        return ()
    listed = KNOWN_MODELS.get(preset.name) or ()
    out = tuple(m for m in listed if m != preset.desk_model)
    if preset.name == REYNOLDS:
        family = family_for(provider, model or preset.model)
        out = tuple(m for m in out if family_for(provider, m) == family)
    return out


def models_for(provider: str) -> tuple[str, ...]:
    """The model ids known for a provider, its default first. Empty for a bare family.

    Where `KNOWN_MODELS` names a provider's models, that list is the whole of it: the
    `reynolds` service meters four models, and its desk models are not ones to switch to.
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


REYNOLDS_DESK_MODELS: dict[str, str] = {
    "openai-responses": "gpt-6-luna",
    "anthropic": "claude-haiku-4-5",
}
"""The front desk's model on `reynolds`, by the family of the study's model. The desk
always speaks the main model's family: it shares the session's endpoint choice, and a
Claude study narrated by GPT would need a second client for no gain."""

REYNOLDS_SUCCESSORS: dict[str, str] = {
    "claude-sonnet-5": "claude-sonnet-5-5",
    "claude-opus-5": "claude-opus-5-5",
}
"""The ids `reynolds` served before, mapped to the ones it serves now. A person who
chose Claude stays on Claude's successor. Anything else it no longer serves goes to the
preset's default (`successor`)."""


def is_claude(model: str) -> bool:
    return (model or "").strip().lower().startswith("claude-")


def family_for(provider: str, model: str) -> str:
    """The API family a request for `model` on `provider` is spoken in.

    The preset's (or the bare family's) everywhere but `reynolds`, where the model
    decides: a `claude-` id goes over the Messages API, anything else over the
    Responses API."""
    preset = preset_for(provider)
    if preset is not None and preset.name == REYNOLDS:
        return "anthropic" if is_claude(model) else "openai-responses"
    return family_of(provider)


def desk_model_for(provider: str, model: str) -> str:
    """The front desk's model for a session on `provider` running `model`.

    The preset's desk model, except on `reynolds`, where it follows the family of the
    main model (`REYNOLDS_DESK_MODELS`). Empty for a bare family."""
    preset = preset_for(provider)
    if preset is None:
        return ""
    if preset.name == REYNOLDS:
        return REYNOLDS_DESK_MODELS[family_for(provider, model)]
    return preset.desk_model


def serves(provider: str, model: str) -> bool:
    """Whether `provider` can be asked for `model` at all, as far as this file knows.

    True everywhere but `reynolds`: any id a vendor answers to can be typed, and the
    probe is what decides. The workspace service is the exception because its list is
    closed -- it meters its four models and the two desk models, and nothing else is
    behind it. That matters for what this machine remembers rather than what anyone
    types: a config file or a study written before the 5.5 models still names
    `claude-sonnet-5`, which the service no longer has (`successor`)."""
    preset = preset_for(provider)
    if preset is None or preset.name != REYNOLDS:
        return True
    return (model or "").strip() in (*models_for(provider), *REYNOLDS_DESK_MODELS.values())


def successor(provider: str, model: str) -> str:
    """The model to run on `provider` in place of `model`, which it may not serve.

    `model` itself where `serves` says so; on `reynolds`, a retired Claude id's
    successor (`REYNOLDS_SUCCESSORS`), and the preset's default for anything else."""
    if serves(provider, model):
        return model
    preset = preset_for(provider)
    assert preset is not None  # `serves` is False only on a preset
    return REYNOLDS_SUCCESSORS.get((model or "").strip(), preset.model)


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
