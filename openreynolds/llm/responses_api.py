"""The Responses API: OpenAI's own dialect, and the only one its current models answer.

Chat Completions stays the default and stays the lingua franca -- a local server, a
router and most vendors speak that and nothing else, which is why `openai_api.py` is
written against it. But OpenAI now refuses function tools together with a reasoning
effort on `/v1/chat/completions` for every model above `gpt-5.2`:

    Function tools with reasoning_effort are not supported for gpt-5.6-sol in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

Both branches of that sentence are dead ends for this desk. `reasoning_effort: 'none'`
buys the tool call by turning the reasoning off, which is the one thing a long-horizon
build-up run cannot spare; and the adapter that did not speak `/v1/responses` simply
could not run the model at all.

**What this dialect gives back that Chat Completions cannot is the reasoning item
itself.** With `store=False` the chain of thought returns as `encrypted_content`, and
feeding those items into the next request is what lets a reasoning model keep its own
reasoning across a tool-call round. Chat Completions has no way to express that: the
`reasoning_content` echo is a vendor extension that carries a summary at best. So the
whole assistant turn rides back as `raw` here, and `_assistant` splices it in verbatim
rather than rebuilding it -- a rebuilt turn is a turn whose reasoning was dropped, and
on a thirty-step mesh that is the difference the API exists to make.
"""

from __future__ import annotations

import importlib
import json
import time
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

MIN_OUTPUT_TOKENS = 16
"""The API's own floor. A probe asking for fewer is a 400 about the wrong thing."""


class ResponsesProvider(Provider):
    name = "openai-responses"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self.client = openai.OpenAI(
            api_key=api_key or "none",
            base_url=base_url or None,
            default_headers=default_headers,
            timeout=timeout,
        )
        self.lean = False
        """Whether the endpoint has refused the `reasoning` block."""
        self.plain = False
        """Whether it has refused `include=[reasoning.encrypted_content]`. A gateway
        that imitates this API may serve the shape without the encrypted echo; losing
        the reasoning chain is worse than not running, but it is not a reason to fail
        the turn."""

    # -- rendering -------------------------------------------------------------

    def render(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The thread as `input` items. The system prompt goes in `instructions`, not
        here, which is why this takes no `system` argument."""
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                out.append(_text_item("user", _as_text(m["content"])))
            elif role == "user":
                out.extend(self._user(m["content"]))
            elif role == "assistant":
                out.extend(self._assistant(m))
        return out

    def _user(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [_text_item("user", content)]
        items: list[dict[str, Any]] = []
        trailing: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_result":
                text, images = split_result(block.get("content"))
                if block.get("is_error"):
                    text = f"error: {text}" if text else "error"
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.get("tool_use_id", ""),
                        "output": text or "(no output)",
                    }
                )
                # Unlike a `tool` message in Chat Completions, a `function_call_output`
                # carries no picture either, so the render still rides in a following
                # user turn -- named, so the model can tell which call drew it.
                for image in images:
                    trailing.append(
                        {"type": "input_text",
                         "text": f"(image returned by tool call {block.get('tool_use_id', '')})"}
                    )
                    trailing.append(_image_part(image))
            elif kind == "text":
                trailing.append({"type": "input_text", "text": block.get("text", "")})
            elif kind == "image":
                trailing.append(_image_part(block))
        if trailing:
            items.append({"role": "user", "content": trailing})
        return items

    def _assistant(self, m: dict[str, Any]) -> list[dict[str, Any]]:
        """The turn's own output items when we produced them, a rebuild otherwise.

        The verbatim path is the one that matters: it carries the `reasoning` items,
        and with them the encrypted chain of thought the next request continues from.
        The rebuild exists so a thread that started on another provider still renders
        -- it loses the reasoning, which `neutral_blocks` drops for everyone."""
        if m.get("provider") == self.name and m.get("raw") is not None:
            return [dict(item) for item in m["raw"]]
        out: list[dict[str, Any]] = []
        for b in neutral_blocks(m["content"]):
            if b.type == "text":
                if b.text:
                    out.append(
                        {"type": "message", "role": "assistant",
                         "content": [{"type": "output_text", "text": b.text}]}
                    )
            else:
                out.append(
                    {"type": "function_call", "call_id": b.id, "name": b.name,
                     "arguments": json.dumps(b.input or {})}
                )
        return out

    @staticmethod
    def tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flat, unlike Chat Completions' `{"type": "function", "function": {...}}`."""
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

    # -- the calls -------------------------------------------------------------

    def _kwargs(self, model: str, system: str, items: list[dict[str, Any]],
                tools: list[dict[str, Any]], effort: str, max_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": items,
            "max_output_tokens": max(MIN_OUTPUT_TOKENS, max_tokens),
            # Nothing is kept on OpenAI's side: the thread lives in this process, and a
            # stored response is a copy of a customer's geometry we did not need to make.
            "store": False,
        }
        if tools:
            kwargs["tools"] = self.tools(tools)
        if effort and not self.lean:
            kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
            if not self.plain:
                kwargs["include"] = ["reasoning.encrypted_content"]
        return kwargs

    @staticmethod
    def _transport_errors() -> tuple[type[BaseException], ...]:
        """The exceptions a stream that dies in the middle actually raises.

        A request that fails before it opens comes back wrapped as
        `openai.APIConnectionError`. One that dies *during* iteration does not: it raises
        from inside the SDK's own `with` block, below the layer that does the wrapping, so
        the raw transport exception escapes. The httpx package is `httpx2` in this
        environment and plain `httpx` in others, so probe for whichever is installed
        rather than importing one and failing on the other.
        """
        found: list[type[BaseException]] = [openai.APIConnectionError]
        for name in ("httpx2", "httpx"):
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            base = getattr(module, "TransportError", None)
            if isinstance(base, type) and issubclass(base, BaseException):
                found.append(base)
        return tuple(found)

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
        items = self.render(messages)
        transport = self._transport_errors()
        progress: dict[str, bool] = {}
        for attempt in (1, 2, 3):
            kwargs = self._kwargs(model, system, items, tools, effort, max_tokens)
            try:
                return self._stream_once(kwargs, listener, progress)
            except transport as exc:
                # A reset mid-stream used to end the whole session: nothing between here
                # and the top of the loop caught it, so one dropped connection threw away
                # an hour of solved case and the report that went with it. Nothing has
                # been committed to the thread yet -- the turn only lands when
                # `_stream_once` returns -- so the request is safe to send again.
                if attempt == 3:
                    raise ProviderError(f"connection lost mid-stream: {exc}") from exc
                if progress.get("emitted"):
                    # Say so rather than silently printing the reply twice.
                    listener.on_text(
                        "\n[connection dropped mid-reply; retrying, "
                        "the text above may repeat]\n"
                    )
                progress.clear()
                time.sleep(2 ** attempt)
            except openai.BadRequestError as exc:
                text = _message(exc)
                lowered = text.lower()
                if "encrypted_content" in lowered or "include" in lowered:
                    self.plain = True
                elif "reasoning" in lowered and not self.lean:
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

    def _stream_once(
        self, kwargs: dict[str, Any], listener: Listener,
        progress: dict[str, bool] | None = None,
    ) -> Turn:
        final: Any = None
        opened = False
        if progress is None:
            progress = {}
        with self.client.responses.stream(**kwargs) as stream:
            for event in stream:
                kind = getattr(event, "type", "")
                if kind == "response.output_text.delta":
                    progress["emitted"] = True
                    listener.on_text(getattr(event, "delta", "") or "")
                elif kind == "response.reasoning_summary_text.delta":
                    if not opened:
                        opened = True
                        listener.on_thinking_begin()
                    listener.on_thinking(getattr(event, "delta", "") or "")
            final = stream.get_final_response()
        return self._turn(final)

    def _turn(self, response: Any) -> Turn:
        content: list[Any] = []
        raw: list[dict[str, Any]] = []
        calls = 0
        for item in getattr(response, "output", None) or []:
            raw.append(_as_dict(item))
            kind = getattr(item, "type", "")
            if kind == "reasoning":
                summary = "".join(
                    getattr(part, "text", "") or ""
                    for part in (getattr(item, "summary", None) or [])
                )
                if summary:
                    content.append(ThinkingBlock(thinking=summary))
            elif kind == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", "") == "output_text":
                        content.append(TextBlock(text=getattr(part, "text", "") or ""))
            elif kind == "function_call":
                calls += 1
                content.append(
                    ToolUseBlock(
                        id=getattr(item, "call_id", "") or "",
                        name=getattr(item, "name", "") or "",
                        input=_arguments(getattr(item, "arguments", "") or ""),
                    )
                )
        stop = "tool_use" if calls else "end_turn"
        detail = getattr(response, "incomplete_details", None)
        if getattr(response, "status", "") == "incomplete":
            if getattr(detail, "reason", "") == "max_output_tokens":
                stop = "max_tokens"
        tokens = _token_classes(getattr(response, "usage", None))
        return Turn(
            content=content,
            stop_reason=stop,
            context_tokens=sum(tokens.values()),
            tokens=tokens,
            provider=self.name,
            raw=raw,
        )

    def complete(self, *, model: str, system: str, prompt: str, max_tokens: int) -> str:
        try:
            response = self.client.responses.create(
                model=model, instructions=system, input=prompt, store=False,
                max_output_tokens=max(MIN_OUTPUT_TOKENS, max_tokens))
        except openai.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except openai.APIError as exc:
            raise ProviderError(str(exc)) from exc
        return (getattr(response, "output_text", "") or "").strip()

    def probe(self, model: str, vision: bool = False) -> str:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": "ping"}]
        if vision:
            content.append(
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{PROBE_PNG}"}
            )
        try:
            self.client.responses.create(
                model=model, input=[{"role": "user", "content": content}],
                store=False, max_output_tokens=MIN_OUTPUT_TOKENS)
        except openai.BadRequestError as exc:
            text = _message(exc)
            if vision and ("image" in text.lower() or "vision" in text.lower()):
                raise cannot_see(model, text) from exc
            raise BadRequest(text) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(_message(exc), exc.status_code) from exc
        except openai.APIError as exc:
            raise ProviderError(str(exc)) from exc
        return f"{model} answered on the Responses API"


# -- helpers -------------------------------------------------------------------


def _text_item(role: str, text: str) -> dict[str, Any]:
    key = "input_text" if role == "user" else "output_text"
    return {"role": role, "content": [{"type": key, "text": text}]}


def _image_part(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source", {})
    media = source.get("media_type", "image/png")
    return {
        "type": "input_image",
        "image_url": f"data:{media};base64,{source.get('data', '')}",
    }


def _as_dict(item: Any) -> dict[str, Any]:
    """The item as plain JSON, because it goes back out in the next request.

    `model_dump` where the SDK gives a model, and the object itself when a test hands
    us a dict -- the thread is written to disk between turns and has to survive it."""
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return dict(item) if isinstance(item, dict) else {}


def _arguments(text: str) -> dict[str, Any]:
    """The tool's input, or a marker the tool handler refuses in words the model can
    act on -- the same contract as the Chat Completions adapter."""
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"__invalid_json__": text}
    return value if isinstance(value, dict) else {"__invalid_json__": text}


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _token_classes(usage: Any) -> dict[str, int]:
    """The counts this API reports, under the names the loop uses.

    `output_tokens` already includes the reasoning tokens, which is the honest total:
    they are generated, they are billed, and a run that spends its budget thinking has
    spent it. `input` is the uncached remainder so the classes still sum to the whole
    request, and a cache write is never billed here."""
    if usage is None:
        return {}
    total_in = int(getattr(usage, "input_tokens", 0) or 0)
    details = getattr(usage, "input_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return {
        "input": max(0, total_in - cached),
        "cache_read": cached,
        "cache_write": 0,
        "output": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])
    return str(getattr(exc, "message", None) or exc)
