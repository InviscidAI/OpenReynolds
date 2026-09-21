"""Changing the model, the provider or the effort in the middle of a study.

The person asked for it, so this is their choice about how the session is run, the same
kind of thing as a mode. Two rules make it safe:

* **checked before it is accepted** -- the candidate is probed exactly as `doctor`
  probes a configuration, image included, so a typo, a missing key or a text-only
  model is refused at the moment it is typed rather than one turn later;
* **applied between turns** -- `/model` only records what was asked for
  (`loop.pending_model`). `Loop.run` applies it before its first request, when every
  earlier assistant turn is complete, so no turn is ever half one model and half
  another.

Effort needs neither: it is read on every request, so setting it is immediate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .config import CONTEXT_REFRESH_FRACTION, CONTEXT_WINDOW_TOKENS
from .llm import ProviderError, make_provider
from .llm.presets import (
    EFFORTS,
    FALLBACK_CONTEXT_WINDOW,
    PRESETS,
    REYNOLDS,
    context_window_for,
    models_for,
    preset_for,
)

THINKING_BLOCKS = ("thinking", "redacted_thinking")


@dataclass
class Pending:
    """A switch that has been checked and is waiting for the next turn."""

    provider: str
    model: str
    llm_api_key: str
    llm_base_url: str | None
    context_window: int
    desk_model: str


def _mode(cfg: Any) -> str:
    return str(getattr(cfg, "mode", "") or "auto")


def status_lines(loop: Any) -> list[str]:
    """What `/model` on its own answers."""
    cfg = loop.cfg
    lines = [
        f"provider {cfg.provider}   model {cfg.model}   effort {cfg.effort}   mode {_mode(cfg)}",
    ]
    known = models_for(cfg.provider)
    if known:
        lines.append("known models here: " + ", ".join(known))
    pending = getattr(loop, "pending_model", None)
    if pending is not None:
        lines.append(f"switching to {pending.provider}:{pending.model} when the next turn starts")
    others = [name for name in sorted(PRESETS) if name != cfg.provider and key_for(name, cfg) is not None]
    if others:
        lines.append("other providers with a key here: " + ", ".join(others))
    lines.append("/model <model>, /model <provider>:<model>, /effort <"
                 + "|".join(EFFORTS) + ">")
    return lines


def key_for(provider: str, cfg: Any, env: Mapping[str, str] | None = None) -> str | None:
    """The model key a provider would use from here, or None when there is none.

    "" is a real answer: the preset needs no key of its own (a local model, or
    `reynolds`, where the service key stands in), or the provider asked about is the
    one this session is already running and an explicit endpoint stands in for a key.
    That last case is `Config.model_key_missing`'s rule, and it has to be the same
    rule: a gateway configuration with an endpoint and no key is legal -- `missing()`
    says so and the session is running on it -- so answering None for the provider in
    use would refuse a switch, or a resume, onto the very provider about to be used."""
    env = os.environ if env is None else env
    preset = preset_for(provider)
    if preset is None:
        return cfg.llm_api_key if provider == cfg.provider else None
    if preset.name == REYNOLDS:
        return "" if cfg.foamd_api_key else None
    if preset.name == cfg.provider:
        if cfg.llm_api_key:
            return cfg.llm_api_key
        if cfg.llm_base_url:
            return ""
    if preset.key_env and env.get(preset.key_env):
        return env[preset.key_env]
    if not preset.needs_key:
        return ""
    return None


def parse(text: str, cfg: Any) -> tuple[str, str]:
    """(provider, model) for what was typed after `/model`.

    `<preset>:<model>` and a bare `<preset>` name a provider; anything else is a model
    on the current one. Only a real preset name counts before the colon, because model
    ids have colons of their own (`qwen3:8b`)."""
    text = text.strip()
    head, colon, tail = text.partition(":")
    if colon and preset_for(head) is not None:
        preset = preset_for(head)
        return preset.name, tail.strip() or preset.model
    preset = preset_for(text)
    if preset is not None and text not in models_for(cfg.provider):
        return preset.name, preset.model
    return cfg.provider, text


def candidate(cfg: Any, provider: str, model: str, key: str) -> Any:
    """The configuration the session would have after the switch.

    The window is the new model's where one is known for it (`context_window_for`),
    otherwise the provider's: the configured one on the same provider, the preset's on
    another. Leaving a model with a window of its own for one without goes back to the
    preset's, so Haiku's 200k does not follow the session back to Opus."""
    preset = preset_for(provider)
    known = context_window_for(model)
    if provider == cfg.provider:
        new = replace(cfg, model=model)
        # `replace` re-runs `__post_init__`, which swaps `claude-opus-5` for the
        # preset's model on a non-anthropic preset. That swap is for a provider named
        # on its own; a model named here is the opposite of that, and without this line
        # `/model claude-opus-5` on `reynolds` -- one of the two models that service
        # meters -- quietly stayed on Sonnet, as did a resume restoring it. The model
        # asked for wins, exactly as it does across providers below. Nothing about the
        # desk was asked for, so it keeps the one the session already had.
        new.model = model
        new.desk_model = cfg.desk_model
        if known:
            new.context_window = known
        elif context_window_for(cfg.model):
            new.context_window = preset.context_window if preset else FALLBACK_CONTEXT_WINDOW
        return new
    new = replace(cfg, provider=provider, model=model, llm_api_key=key,
                  llm_base_url=None, context_window=0)
    # `__post_init__` swaps a default model id for the preset's; the one asked for wins.
    new.model = model
    if known:
        new.context_window = known
    if preset is not None:
        new.desk_model = preset.desk_model
    return new


def restore(cfg: Any, provider: str, model: str, key: str) -> None:
    """Put a configuration onto a (provider, model) pair recorded earlier.

    What `apply` does between turns, done before a session has started: there is no
    client to rebuild, no thread to strip and nothing on screen to correct, because
    none of them exists yet. It goes through `candidate` for the same reason `/model`
    does -- a pair from another provider has to bring that provider's key, endpoint,
    window and desk model with it, and a model id set on its own would leave this
    provider being asked for another vendor's model.
    """
    new = candidate(cfg, provider, model, key)
    cfg.provider = new.provider
    cfg.model = new.model
    cfg.llm_api_key = new.llm_api_key
    cfg.llm_base_url = new.llm_base_url
    cfg.context_window = new.context_window
    cfg.desk_model = new.desk_model


def request(loop: Any, text: str, env: Mapping[str, str] | None = None) -> list[str]:
    """Handle `/model [spec]`: answer, or check a switch and leave it pending."""
    cfg = loop.cfg
    if not text.strip():
        return status_lines(loop)
    provider, model = parse(text, cfg)
    if not model:
        return [f"no model named for {provider}; /model {provider}:<model>"]
    key = key_for(provider, cfg, env)
    if key is None:
        preset = preset_for(provider)
        where = f"{preset.key_env} in the environment, or " if preset and preset.key_env else ""
        return [
            f"not switching: there is no key for {provider} here.",
            f"set {where}run openreynolds config --provider {provider}",
        ]
    if provider == cfg.provider and model == cfg.model:
        # Cancels a pending switch, and with it any refresh that switch needed: the
        # thread fits the model it is already on.
        loop.pending_model = None
        loop.refresh_due = False
        return [f"already on {model} ({provider})"]
    new = candidate(cfg, provider, model, key)
    try:
        detail = make_provider(new).probe(model, vision=True)
    except (ProviderError, ValueError) as exc:
        reason = getattr(exc, "message", "") or str(exc)
        return [f"not switching to {model}: {reason}"]
    loop.pending_model = Pending(
        provider=new.provider,
        model=new.model,
        llm_api_key=new.llm_api_key,
        llm_base_url=new.llm_base_url,
        context_window=new.context_window,
        desk_model=new.desk_model,
    )
    window = new.context_window or CONTEXT_WINDOW_TOKENS
    lines = [detail]
    # Set from this candidate alone: a refresh an earlier, replaced request needed must
    # not outlive it and throw the thread away for a switch that fits.
    loop.refresh_due = loop.context_tokens > window * CONTEXT_REFRESH_FRACTION
    if loop.refresh_due:
        # The thread would not fit the new window comfortably, so it is refreshed on
        # the model that built it before the new one sees it (the interactive loop
        # honours this flag the same way it honours `needs_refresh`).
        lines.append(
            f"the thread ({loop.context_tokens:,} tokens) is too large for {model}'s "
            f"{window:,}-token window, so it is refreshed on {cfg.model} first"
        )
    if getattr(loop, "running", False):
        lines.append(f"{model} takes over when the current turn ends")
    else:
        lines.append(f"{model} answers from your next message")
    return lines


def apply(loop: Any) -> bool:
    """Apply `loop.pending_model`, if there is one. Called by `Loop.run` before its
    first request, when every earlier assistant turn is complete."""
    pending = getattr(loop, "pending_model", None)
    if pending is None:
        return False
    loop.pending_model = None
    cfg = loop.cfg
    rebuild = (pending.provider, pending.llm_api_key, pending.llm_base_url) != (
        cfg.provider, cfg.llm_api_key, cfg.llm_base_url)
    model_changed = pending.model != cfg.model
    cfg.provider = pending.provider
    cfg.model = pending.model
    cfg.llm_api_key = pending.llm_api_key
    cfg.llm_base_url = pending.llm_base_url
    cfg.context_window = pending.context_window
    cfg.desk_model = pending.desk_model
    if rebuild:
        loop.provider = make_provider(
            cfg, timeout=cfg.llm_timeout_s,
            default_headers={"X-Study-Id": loop.store.session.study_id},
        )
        # Whether a mid-conversation system turn is accepted is a fact about an
        # endpoint, and this is a different one.
        loop._no_system_role = False
    elif model_changed:
        # Same client, another model. What a client latched after a refused request
        # (`lean`: no thinking, effort or cache; `legacy_max_tokens`) was learned about
        # the old model, and kept it would silently drop caching and effort for the new
        # one. Cleared, it is learned again in one request if it still applies.
        for latch in ("lean", "legacy_max_tokens"):
            if getattr(loop.provider, latch, False):
                setattr(loop.provider, latch, False)
    loop.window = cfg.context_window or CONTEXT_WINDOW_TOKENS
    if model_changed:
        strip_thinking(loop.messages)
    loop.store.session.model = cfg.model
    # With the model, because neither is worth anything without the other: a resume
    # reads the pair back, and an id alone could be any provider's.
    loop.store.session.provider = cfg.provider
    # And where it was served from. A provider name is not an endpoint: a gateway or a
    # router in front of one family lists other vendors' ids, so a pair restored onto a
    # direct vendor key of the same family fails a turn later with that vendor's 400.
    loop.store.session.base_url = cfg.llm_base_url or ""
    loop.store.save()
    loop.view.model(cfg.model, cfg.effort, cfg.provider)
    loop.view.info(f"now on {cfg.model} ({cfg.provider})")
    return True


def strip_thinking(messages: list[dict[str, Any]]) -> int:
    """Take earlier reasoning blocks out of the thread; returns how many.

    A thinking block carries a signature only the model that wrote it accepts, and the
    reasoning has already become the words and tool calls beside it."""
    removed = 0
    for message in messages:
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
            continue
        kept = []
        for block in message["content"]:
            kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if kind in THINKING_BLOCKS:
                removed += 1
            else:
                kept.append(block)
        if not kept:
            # A turn that was only reasoning (cut off by max_tokens, say). The APIs
            # refuse an assistant turn with no content, and dropping the turn would
            # break the user/assistant alternation, so a marker stands in for it.
            kept = [{"type": "text", "text": "(reasoning from an earlier model, omitted)"}]
        message["content"] = kept
    return removed


def effort(loop: Any, text: str) -> list[str]:
    """Handle `/effort [level]`. Effort is read per request, so this is immediate."""
    cfg = loop.cfg
    level = text.strip().lower()
    if not level:
        return [f"effort {cfg.effort}; one of {', '.join(EFFORTS)}"]
    if level not in EFFORTS:
        return [f"effort is one of {', '.join(EFFORTS)}; not {text.strip()!r}"]
    cfg.effort = level
    loop.view.model(cfg.model, cfg.effort, cfg.provider)
    return [f"effort {level} from the next request"]
