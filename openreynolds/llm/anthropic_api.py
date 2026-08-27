"""The Messages API: Anthropic itself, and every vendor that speaks its dialect.

The Claude-only extras -- adaptive thinking, an effort setting, prompt caching -- are
sent on the first try and dropped for the rest of the session the moment an endpoint
rejects them, so a compatible vendor costs one failed request rather than a config
flag someone has to know about.
"""

from __future__ import annotations

from typing import Any

import anthropic

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
        for attempt in (1, 2):
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                system=self._system(system),
                messages=self.render(messages),
                tools=tools,
            )
            if not self.lean:
                kwargs.update(
                    thinking={"type": "adaptive", "display": "summarized"},
                    output_config={"effort": effort},
                    cache_control={"type": "ephemeral"},
                )
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
        detail = getattr(response, "stop_details", None)
        return Turn(
            content=list(response.content),
            stop_reason=response.stop_reason or "end_turn",
            stop_explanation=getattr(detail, "explanation", None) or "",
            context_tokens=_context_tokens(getattr(response, "usage", None)),
            provider=self.name,
        )

    def complete(self, *, model: str, system: str, prompt: str, max_tokens: int) -> str:
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system(system),
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except anthropic.APIError as exc:
            raise ProviderError(str(exc)) from exc
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def probe(self, model: str, vision: bool = False) -> str:
        """Counting tokens validates the key, the endpoint and the model id in one
        free call -- on Anthropic, and with an image in the count it also proves the
        model accepts pictures. A compatible vendor may not have the endpoint, in
        which case the cheapest real request stands in."""
        messages = [{"role": "user", "content": _probe_content(vision)}]
        try:
            counted = self.client.messages.count_tokens(model=model, messages=messages)
            sees = " and can see images" if vision else ""
            return f"{model} reachable ({counted.input_tokens} tokens for a ping){sees}"
        except anthropic.NotFoundError:
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
                model=model, max_tokens=5, system=self._system("Reply with one word."), messages=messages
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


def _context_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    return int(
        (getattr(usage, "input_tokens", 0) or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + (getattr(usage, "output_tokens", 0) or 0)
    )


def _message(exc: Exception) -> str:
    return str(getattr(exc, "message", None) or exc)
