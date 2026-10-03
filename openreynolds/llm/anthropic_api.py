"""The Messages API: Anthropic itself, and every vendor that speaks its dialect.

The Claude-only extras -- adaptive thinking, an effort setting, prompt caching -- are
sent on the first try and dropped for the rest of the session the moment an endpoint
rejects them, so a compatible vendor costs one failed request rather than a config
flag someone has to know about.

Claude's 5.5 generation (and Fable, Mythos) narrows what a request may say
(`_always_thinks`): thinking cannot be disabled -- `{type: "disabled"}` and a
`budget_tokens` are a 400 on Opus 5.5 at every effort, and `disabled` is a 400 on
Sonnet 5.5, which turns thinking off with `{type: "between_tools"}` instead -- forced
`tool_choice` (`any`, `tool`) is a 400, as are non-default sampling parameters and an
assistant prefill, and Opus 5.5 defaults to `medium` effort. This module sends none of
the refused fields to any model: thinking is adaptive, absent, or (Sonnet 5.5's short
answers) `between_tools`; `tool_choice` is never sent, which is the API's `auto`;
sampling is left at its defaults; and no request ends on an assistant turn. What it
adds for these models is the effort on every request, because leaving it out is not
neutral on Opus 5.5; and a short answer (`complete`, `probe`) asks for `low` effort --
on Sonnet 5.5 with thinking only between tools -- so reasoning does not spend it.
"""

from __future__ import annotations

import time

from typing import Any

import anthropic

from .. import trace
from .base import (
    PROBE_PNG,
    BadRequest,
    Listener,
    Provider,
    ProviderError,
    Turn,
    cannot_see,
    neutral_blocks,
)


_ALWAYS_THINKING = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-", "claude-mythos-")
"""Ids (as prefixes, so a dated or suffixed variant matches) of models whose thinking
cannot be turned off and that take an effort on every request."""

COMPLETE_FLOOR_TOKENS = 2_048
"""`max_tokens` for a short answer from a model that always thinks. The cap counts the
thinking too, and a sixty-token budget is spent before a word of the answer."""


def _always_thinks(model: str) -> bool:
    name = (model or "").strip().lower()
    return any(name.startswith(tag) for tag in _ALWAYS_THINKING)


def _thinks_between_tools(model: str) -> bool:
    """Whether `{type: "between_tools"}` -- thinking off but for tool rounds -- is
    accepted. Sonnet 5.5 only; on Opus 5.5 effort is the one control."""
    return (model or "").strip().lower().startswith("claude-sonnet-5-5")


def _quiet(model: str, max_tokens: int) -> dict[str, Any]:
    """The extras for a short, tool-less answer: nothing for a model that thinks only
    when asked, the least effort (and on Sonnet 5.5 no thinking before the answer) and
    room for what thinking remains for one that always thinks."""
    if not _always_thinks(model):
        return {"max_tokens": max_tokens}
    extras: dict[str, Any] = {"output_config": {"effort": "low"}}
    if _thinks_between_tools(model):
        # Takes no other field, and is accepted at effort `high` or below.
        extras["thinking"] = {"type": "between_tools"}
        extras["max_tokens"] = max_tokens
    else:
        extras["max_tokens"] = max(max_tokens, COMPLETE_FLOOR_TOKENS)
    return extras


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self.client = anthropic.Anthropic(
            api_key=api_key or None,
            base_url=base_url or None,
            default_headers=default_headers,
            timeout=timeout,
        )
        self.lean = False
        """Whether the endpoint has refused the Claude-only request fields."""

    # -- rendering -------------------------------------------------------------

    def _system(self, text: str) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": text}
        if not self.lean:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def render(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") != "assistant":
                out.append({"role": m["role"], "content": m["content"]})
                continue
            if m.get("provider", self.name) == self.name:
                # Its own blocks, signatures and all: what it asked to see again.
                out.append({"role": "assistant", "content": m["content"]})
            else:
                out.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": b.text}
                            if b.type == "text"
                            else {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                            for b in neutral_blocks(m["content"])
                        ],
                    }
                )
        return out

    # -- the calls -------------------------------------------------------------

    def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        effort: str,
        max_tokens: int,
        listener: Listener,
    ) -> Turn:
        return self.with_fallback(
            model,
            lambda asked: self._stream_model(
                asked, system=system, messages=messages, tools=tools,
                effort=effort, max_tokens=max_tokens, listener=listener,
            ),
        )

    def _stream_model(
        self,
        model: str,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        effort: str,
        max_tokens: int,
        listener: Listener,
    ) -> Turn:
        """One model, asked once -- twice when the first try is what teaches `lean`."""
        for attempt in (1, 2):
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                system=self._system(system),
                messages=self.render(messages),
                tools=tools,
            )
            always = _always_thinks(model)
            if not self.lean:
                kwargs.update(
                    # `summarized`, because these models default to `omitted`: the
                    # thinking blocks would stream empty and the UI would show a pause.
                    thinking={"type": "adaptive", "display": "summarized"},
                    output_config={"effort": effort or "high"},
                    cache_control={"type": "ephemeral"},
                )
            elif always:
                # Lean drops what a compatible endpoint refused, but on a model that
                # always thinks the effort is not optional: left out, Opus 5.5 runs at
                # `medium` rather than the effort the session asked for. Thinking is
                # left out, which on these models is adaptive anyway.
                kwargs["output_config"] = {"effort": effort or "high"}
            try:
                return self._stream_once(kwargs, listener)
            except anthropic.BadRequestError as exc:
                text = _message(exc)
                if "system" in text.lower() or self.lean or attempt == 2:
                    raise BadRequest(text) from exc
                # Not the system role, and the extras were on: this endpoint does
                # not know them. Try once more without, and stay that way.
                self.lean = True
            except anthropic.APIStatusError as exc:
                raise ProviderError(_message(exc), exc.status_code) from exc
            except anthropic.APIError as exc:
                raise ProviderError(str(exc)) from exc
        raise AssertionError("unreachable")

    def _stream_once(self, kwargs: dict[str, Any], listener: Listener) -> Turn:
        _started = time.monotonic() if trace.on else 0.0
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "thinking":
                    listener.on_thinking_begin()
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        listener.on_thinking(event.delta.thinking)
                    elif event.delta.type == "text_delta":
                        listener.on_text(event.delta.text)
            response = stream.get_final_message()
        if trace.on:
            trace.event(
                "turn",
                seconds=round(time.monotonic() - _started, 3),
                model=kwargs.get("model"),
                messages=len(kwargs.get("messages") or []),
                stop=response.stop_reason,
                **trace.usage(response),
            )
        return self._turn(response, kwargs["model"])

    def _turn(self, response: Any, asked: str) -> Turn:
        detail = getattr(response, "stop_details", None)
        return Turn(
            content=list(response.content),
            stop_reason=response.stop_reason or "end_turn",
            stop_explanation=getattr(detail, "explanation", None) or "",
            stop_category=getattr(detail, "category", None) or "",
            # The id asked for, not the dated one the response resolves it to: it is
            # what the next request will name, and what a hosted allowlist knows.
            model=asked,
            context_tokens=_context_tokens(getattr(response, "usage", None)),
            tokens=_token_classes(getattr(response, "usage", None)),
            provider=self.name,
        )

    def complete(self, *, model: str, system: str, prompt: str, max_tokens: int) -> str:
        turn = self.with_fallback(
            model, lambda asked: self._complete_model(asked, system, prompt, max_tokens),
        )
        return turn.text.strip()

    def _complete_model(self, model: str, system: str, prompt: str, max_tokens: int) -> Turn:
        try:
            response = self.client.messages.create(
                model=model,
                system=self._system(system),
                messages=[{"role": "user", "content": prompt}],
                **_quiet(model, max_tokens),
            )
        except anthropic.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except anthropic.APIError as exc:
            raise ProviderError(str(exc)) from exc
        return self._turn(response, model)

    def probe(self, model: str, vision: bool = False) -> str:
        """Counting tokens validates the key, the endpoint and the model id in one
        free call -- on Anthropic, and with an image in the count it also proves the
        model accepts pictures. A compatible vendor may not have the endpoint, or the
        key may not be allowed to use it (Bedrock counts on a separate service, under
        its own IAM action), in which case the cheapest real request stands in: a
        count is a convenience, and a key that can run the model can run the study."""
        messages = [{"role": "user", "content": _probe_content(vision)}]
        try:
            counted = self.client.messages.count_tokens(model=model, messages=messages)
            sees = " and can see images" if vision else ""
            return f"{model} reachable ({counted.input_tokens} tokens for a ping){sees}"
        except (anthropic.NotFoundError, anthropic.PermissionDeniedError):
            pass
        except anthropic.BadRequestError as exc:
            if vision and _about_images(_message(exc)):
                raise cannot_see(model, _message(exc)) from exc
            raise ProviderError(_message(exc), exc.status_code) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except anthropic.APIError as exc:
            raise ProviderError(str(exc)) from exc
        try:
            self.client.messages.create(
                model=model, system=self._system("Reply with one word."), messages=messages,
                **_quiet(model, 5),
            )
        except anthropic.BadRequestError as exc:
            if vision and _about_images(_message(exc)):
                raise cannot_see(model, _message(exc)) from exc
            raise ProviderError(_message(exc), exc.status_code) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except anthropic.APIError as exc:
            raise ProviderError(str(exc)) from exc
        sees = " and can see images" if vision else ""
        return f"{model} reachable (answered a ping){sees}"


def _probe_content(vision: bool) -> Any:
    if not vision:
        return "ping"
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PROBE_PNG}},
        {"type": "text", "text": "Reply with one word."},
    ]


def _about_images(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("image", "vision", "multimodal", "content type", "unsupported"))


def _token_classes(usage: Any) -> dict[str, int]:
    """The four counts, kept apart, because their prices span 250x.

    Summed into one integer -- which is all this module used to report -- a study whose
    prompt cache is working perfectly and one whose cache is being invalidated on every
    turn produce the same number. Cache reads are 68-80% of a study's model bill, so
    that one number hid the only thing worth watching."""
    if usage is None:
        return {}
    return {
        "input": int(getattr(usage, "input_tokens", 0) or 0),
        "cache_read": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cache_write": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "output": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _context_tokens(usage: Any) -> int:
    """What the request occupied in the window: the four classes added up."""
    return sum(_token_classes(usage).values())


def _message(exc: Exception) -> str:
    return str(getattr(exc, "message", None) or exc)
