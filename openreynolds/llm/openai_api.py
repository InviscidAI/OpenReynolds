"""Chat Completions: OpenAI, and everything that imitates it -- gateways, local servers.

Chosen over newer OpenAI-only APIs because it is the one dialect a local model, a
router and a cloud vendor all speak. The costs of the translation are known and
accepted: a tool result becomes a `tool` message, an image inside one has to ride in a
following `user` message because `tool` messages cannot carry pictures, and reasoning
arrives (from the vendors that stream it) as a non-standard `reasoning_content` delta
that is echoed back only to the vendor that sent it.
"""

from __future__ import annotations

import json
from typing import Any

import openai

from .base import (
    PROBE_PNG,
    BadRequest,
    Listener,
    Provider,
    ProviderError,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Turn,
    cannot_see,
    neutral_blocks,
    split_result,
)


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self.client = openai.OpenAI(
            # A local server wants no key at all; the SDK insists on a string.
            api_key=api_key or "none",
            base_url=base_url or None,
            default_headers=default_headers,
            timeout=timeout,
        )
        self.lean = False
        """Whether the endpoint has refused the reasoning-effort field."""
        self.legacy_max_tokens = False
        """Whether the endpoint knows only the older `max_tokens` name."""
        self.echoes_reasoning = False
        """Set once the endpoint has streamed `reasoning_content`: those vendors want
        it back in the assistant message of a tool-call round."""

    # -- rendering -------------------------------------------------------------

    def render(self, system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            role = m.get("role")
            if role == "system":
                out.append({"role": "system", "content": _as_text(m["content"])})
            elif role == "user":
                out.extend(self._user(m["content"]))
            elif role == "assistant":
                out.append(self._assistant(m))
        return out

    def _user(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        tool_messages: list[dict[str, Any]] = []
        trailing: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_result":
                text, images = split_result(block.get("content"))
                if block.get("is_error"):
                    text = f"error: {text}" if text else "error"
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": text or "(no output)",
                    }
                )
                for image in images:
                    source = image.get("source", {})
                    media = source.get("media_type", "image/png")
                    trailing.append(
                        {"type": "text", "text": f"(image returned by tool call {block.get('tool_use_id', '')})"}
                    )
                    trailing.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media};base64,{source.get('data', '')}"},
                        }
                    )
            elif kind == "text":
                trailing.append({"type": "text", "text": block.get("text", "")})
        out = tool_messages
        if trailing:
            out.append({"role": "user", "content": trailing})
        return out

    def _assistant(self, m: dict[str, Any]) -> dict[str, Any]:
        if m.get("provider") == self.name and m.get("raw") is not None:
            return m["raw"]
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        for b in neutral_blocks(m["content"]):
            if b.type == "text":
                text_parts.append(b.text)
            else:
                calls.append(_tool_call(b.id, b.name, b.input))
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        if calls:
            message["tool_calls"] = calls
        return message

    @staticmethod
    def tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    # -- the calls -------------------------------------------------------------

    def _limit(self, max_tokens: int) -> dict[str, Any]:
        return {"max_tokens": max_tokens} if self.legacy_max_tokens else {"max_completion_tokens": max_tokens}

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
        for attempt in range(1, 4):
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=self.render(system, messages),
                tools=self.tools(tools),
                stream=True,
                stream_options={"include_usage": True},
                **self._limit(max_tokens),
            )
            if effort and not self.lean:
                kwargs["reasoning_effort"] = effort
            try:
                return self._stream_once(kwargs, listener)
            except openai.BadRequestError as exc:
                text = _message(exc)
                lowered = text.lower()
                if "system" in lowered and "role" in lowered:
                    raise BadRequest(text) from exc
                if "max_completion_tokens" in lowered and not self.legacy_max_tokens:
                    self.legacy_max_tokens = True
                elif not self.lean:
                    self.lean = True
                else:
                    raise BadRequest(text) from exc
                if attempt == 3:
                    raise BadRequest(text) from exc
            except openai.APIStatusError as exc:
                raise ProviderError(_message(exc), exc.status_code) from exc
            except openai.APIError as exc:
                raise ProviderError(str(exc)) from exc
        raise AssertionError("unreachable")

    def _stream_once(self, kwargs: dict[str, Any], listener: Listener) -> Turn:
        text = ""
        reasoning = ""
        thinking_started = False
        calls: dict[int, dict[str, str]] = {}
        finish = None
        usage = None
        for chunk in self.client.chat.completions.create(**kwargs):
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                thought = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if thought:
                    if not thinking_started:
                        thinking_started = True
                        listener.on_thinking_begin()
                    reasoning += thought
                    listener.on_thinking(thought)
                piece = getattr(delta, "content", None)
                if piece:
                    text += piece
                    listener.on_text(piece)
                for tc in getattr(delta, "tool_calls", None) or []:
                    entry = calls.setdefault(int(getattr(tc, "index", 0) or 0), {"id": "", "name": "", "args": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            entry["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            entry["args"] += fn.arguments
            if getattr(choice, "finish_reason", None):
                finish = choice.finish_reason

        content: list[Any] = []
        if reasoning:
            self.echoes_reasoning = True
            content.append(ThinkingBlock(thinking=reasoning))
        if text:
            content.append(TextBlock(text=text))
        raw_calls: list[dict[str, Any]] = []
        for index in sorted(calls):
            entry = calls[index]
            call_id = entry["id"] or f"call_{index}"
            content.append(ToolUseBlock(id=call_id, name=entry["name"], input=_arguments(entry["args"])))
            raw_calls.append(_tool_call(call_id, entry["name"], entry["args"], raw=True))

        raw: dict[str, Any] = {"role": "assistant", "content": text or None}
        if raw_calls:
            raw["tool_calls"] = raw_calls
        if reasoning and self.echoes_reasoning:
            raw["reasoning_content"] = reasoning

        if finish == "length":
            stop = "max_tokens"
        elif finish == "content_filter":
            stop = "refusal"
        elif raw_calls:
            stop = "tool_use"
        else:
            stop = "end_turn"
        return Turn(
            content=content,
            stop_reason=stop,
            stop_explanation="the endpoint's content filter" if stop == "refusal" else "",
            context_tokens=_context_tokens(usage),
            provider=self.name,
            raw=raw,
        )

    def complete(self, *, model: str, system: str, prompt: str, max_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                **self._limit(max_tokens),
            )
        except openai.BadRequestError as exc:
            if "max_completion_tokens" in _message(exc).lower() and not self.legacy_max_tokens:
                self.legacy_max_tokens = True
                return self.complete(model=model, system=system, prompt=prompt, max_tokens=max_tokens)
            raise BadRequest(_message(exc)) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except openai.APIError as exc:
            raise ProviderError(str(exc)) from exc
        choice = (getattr(response, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
        return (getattr(message, "content", None) or "").strip()

    def probe(self, model: str, vision: bool = False) -> str:
        """Looking the model up is free where the endpoint has a model list; where it
        does not -- or when the question is whether it can see -- the smallest real
        request answers."""
        if not vision:
            try:
                self.client.models.retrieve(model)
                return f"{model} reachable"
            except openai.NotFoundError:
                pass
            except openai.APIStatusError as exc:
                raise ProviderError(_message(exc), exc.status_code) from exc
            except openai.APIError as exc:
                raise ProviderError(str(exc)) from exc
            self.complete(model=model, system="Reply with one word.", prompt="ping", max_tokens=5)
            return f"{model} reachable (answered a ping)"
        content = [
            {"type": "text", "text": "Reply with one word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PROBE_PNG}"}},
        ]
        try:
            self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}], **self._limit(5)
            )
        except openai.BadRequestError as exc:
            text = _message(exc)
            if "max_completion_tokens" in text.lower() and not self.legacy_max_tokens:
                self.legacy_max_tokens = True
                return self.probe(model, vision=True)
            raise cannot_see(model, text) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except openai.APIError as exc:
            raise ProviderError(str(exc)) from exc
        return f"{model} reachable and can see images"


def _tool_call(call_id: str, name: str, arguments: Any, raw: bool = False) -> dict[str, Any]:
    text = arguments if (raw and isinstance(arguments, str)) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": text}}


def _arguments(text: str) -> dict[str, Any]:
    """The tool's input, or -- when the model produced something that is not JSON --
    a marker the tool handler will refuse in words the model can act on."""
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"__invalid_json__": text}
    return value if isinstance(value, dict) else {"__invalid_json__": text}


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _context_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    return int((getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0))


def _message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])
    return str(getattr(exc, "message", None) or exc)
