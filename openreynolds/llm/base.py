"""What the loop needs from a model, and nothing about whose model it is.

The agent was written against one provider's SDK, and its shapes -- content blocks
with a `type`, `stop_reason`, tool-use ids -- turned out to be a good neutral
vocabulary, so they are kept here as the contract every provider renders into. A
provider owns three things: turning the thread into its own wire format, turning the
answer back into a `Turn`, and mapping its own exceptions onto `ProviderError` so the
loop has one thing to catch.

The thread itself stays provider-agnostic. Assistant turns carry the provider that
produced them and, when the provider needs its own message back verbatim (reasoning
signatures, tool-call ids in its own format), the native message rides along as
`raw`. Switching providers mid-study therefore works: the other provider rebuilds the
turn from the neutral blocks and simply loses the parts only the original could use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

STOP_REASONS = ("end_turn", "tool_use", "max_tokens", "refusal")


class ProviderError(Exception):
    """A model API failed. `status_code` is set when the API answered at all."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    @property
    def body(self) -> dict[str, Any]:
        """The JSON error body the API answered with, or {} when there was none.

        Read off the SDK error this was raised from (every provider raises `from` it,
        the same chain `retry_after` reads), so no provider has to carry it across. It
        matters on the `reynolds` preset, where the model is fronted by the workspace
        service and a refusal is one of its own: 402 `allowance_exhausted`, 403
        `plan_limit`, 409 `study_paused`, each with fields beyond the message."""
        cause = self.__cause__
        response = getattr(cause, "response", None)
        try:
            parsed = response.json() if response is not None else None
        except Exception:  # noqa: BLE001 - a body that is not JSON is simply no body
            parsed = None
        if not isinstance(parsed, dict):
            parsed = getattr(cause, "body", None)
        return parsed if isinstance(parsed, dict) else {}

    @property
    def code(self) -> str:
        """The service's machine-readable `error` token, when it sent one as a string
        (the workspace service does; a vendor's `error` is an object, and is not one)."""
        code = self.body.get("error")
        return code if isinstance(code, str) else ""


TRANSIENT_STATUSES = frozenset({408, 409, 425, 429})
"""The 4xx answers worth asking again: a timeout, a conflict, a rate limit. Every other
4xx is about this account or this request -- the budget, the key, the model id -- and
answers the same way in a minute. A 5xx, and a call the API never answered at all, are
worth asking again too (`transient`)."""

TRANSIENT_WAITS = (5.0, 15.0, 30.0, 60.0, 120.0)
"""Seconds before each retry of a request that failed transiently: about four minutes
in all, then the failure is the caller's to report.

The SDK's own retries come first and give up within seconds, which is too soon for the
limiter that ended five studies on 2026-10-19: foamd's per-key bucket
(`too many requests; retry in 1s`), drained by two or three sessions on one account
calling it at once. Its refill is steady, so a wait measured in tens of seconds finds
it full where a second's does not. A turn that ended on it instead never resumed: with
no job running there was nothing left to wake the agent, and the study sat in "your
turn" with nothing said."""

MAX_RETRY_AFTER = 300.0
"""The most a `Retry-After` header may stretch one wait to."""


NOT_TRANSIENT_CODES = frozenset({"study_paused"})
"""Service error codes that waiting does not change, whatever their status says.

`study_paused` is a 409, and a 409 is in `TRANSIENT_STATUSES`; but a study paused at
its spending ceiling stays paused until a person raises the ceiling, so asking again
over four minutes would only be refused five more times (`ceiling.py`)."""


def transient(exc: ProviderError) -> bool:
    """Whether asking again could get a different answer."""
    if getattr(exc, "code", "") in NOT_TRANSIENT_CODES:
        return False
    status = exc.status_code
    return status is None or status >= 500 or status in TRANSIENT_STATUSES


def retry_after(exc: BaseException) -> float | None:
    """The `Retry-After` seconds of the SDK error a `ProviderError` was raised from,
    when it had one. Every provider raises from the SDK's own error (`raise ... from
    exc`), and the SDK's status errors keep the response."""
    response = getattr(exc.__cause__, "response", None)
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None  # the HTTP-date form: the scheduled wait stands


def retry_delay(exc: ProviderError, attempt: int,
                waits: tuple[float, ...] = TRANSIENT_WAITS) -> float | None:
    """How long to wait before asking again after `exc`, on retry number `attempt`
    (0-based), or None when it should not be asked again: a refusal, or the waits
    are spent. A server's `Retry-After` lengthens a wait, never shortens it."""
    if not transient(exc) or attempt >= len(waits):
        return None
    return max(waits[attempt], min(retry_after(exc) or 0.0, MAX_RETRY_AFTER))


def nap(seconds: float, stop: Callable[[], bool] = lambda: False) -> bool:
    """Sleep `seconds`, a second at a time, unless `stop()` comes true first.
    Returns whether the whole wait was slept."""
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        if stop():
            return False
        time.sleep(min(1.0, left))
    return not stop()


class BadRequest(ProviderError):
    """The API rejected the request as malformed -- the one failure the loop reacts
    to, because "no mid-conversation system role" is a request shape it can change."""

    def __init__(self, message: str):
        super().__init__(message, 400)


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ThinkingBlock:
    thinking: str
    type: str = "thinking"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Turn:
    """One assistant turn, as the loop sees it.

    `content` is a list of blocks, each with a `.type` -- the provider's own objects
    when it has them (they are what it wants sent back), the dataclasses above
    otherwise. The loop only ever reads `.type`, `.text`, `.name`, `.input`, `.id`.
    """

    content: list[Any]
    stop_reason: str = "end_turn"
    stop_explanation: str = ""
    stop_category: str = ""
    """On a `refusal`, the policy area the provider's classifier named (Anthropic's
    `stop_details.category`: `cyber`, `bio`, `frontier_llm`, `reasoning_extraction`,
    `general_harms`), or empty when it named none. Shown, never parsed: the vendor says
    the wording is not stable."""
    model: str = ""
    """The model that served this turn, by the id it was asked for. Usually the
    session's; after a refusal it is the fallback that answered
    (`Provider.with_fallback`)."""
    declined: list[tuple[str, str]] = field(default_factory=list)
    """`(model, why)` for every model that refused this request before `model` was
    asked. Empty on an ordinary turn. When `stop_reason` is itself `refusal`, these are
    the models that declined before the one whose refusal this is -- the whole chain
    said no."""
    context_tokens: int = 0
    """Everything this request occupied in the model's window -- input, cached input
    and output together -- so the loop can tell when a refresh is due. Each provider
    counts differently; this is the one number they agree to report."""
    tokens: dict[str, int] = field(default_factory=dict)
    """The same request split by what each class of token costs: `input`, `cache_read`,
    `cache_write`, `output`.

    `context_tokens` sums them, which is the right number for "is a refresh due" and the
    wrong one for anything about money: on Opus a cache read and an output token differ
    in price by 250x, so a healthy cache and a completely broken one produce the same
    figure. That is why an eviction bug could cost real money on every study and never
    show up anywhere -- cache reads are 68-80% of a study's model bill, and nothing
    reported them separately. Providers that do not break usage down leave this empty."""
    provider: str = ""
    raw: Any = None
    """The provider-native assistant message, when it differs from `content`."""

    @property
    def text(self) -> str:
        return "".join(getattr(b, "text", "") for b in self.content if _kind(b) == "text")

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(id=_get(b, "id"), name=_get(b, "name"), input=dict(_get(b, "input") or {}))
            for b in self.content
            if _kind(b) == "tool_use"
        ]

    def as_message(self) -> dict[str, Any]:
        """The thread entry for this turn."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content, "provider": self.provider}
        if self.raw is not None:
            message["raw"] = self.raw
        return message


@dataclass
class Listener:
    """Where a streaming turn reports as it arrives. All optional."""

    thinking_begin: Callable[[], None] | None = None
    thinking: Callable[[str], None] | None = None
    text: Callable[[str], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def on_thinking_begin(self) -> None:
        if self.thinking_begin:
            self.thinking_begin()

    def on_thinking(self, delta: str) -> None:
        if self.thinking and delta:
            self.thinking(delta)

    def on_text(self, delta: str) -> None:
        if self.text and delta:
            self.text(delta)


class Provider:
    """The contract. Subclasses set `name` and implement the three calls."""

    name = ""

    fallbacks: tuple[str, ...] = ()
    """The models a refused request is retried on, in order (`make_provider` sets it
    from the configuration or the preset). Empty means a refusal is final.

    A refusal here is the vendor's own safety classifier declining a request --
    Anthropic answers it as a normal 200 with `stop_reason: "refusal"`, and its
    documentation says the remedy is to send the same request to another Claude model.
    One did, live, on 2026-10-01: a CFD study was declined under the "reverse
    engineering or duplicating model outputs" rule, and the session stopped dead with
    the sentence on screen and nothing else to do. The classifier is per model, so the
    same request on the next model usually goes through."""

    @property
    def pinned(self) -> dict[str, str]:
        """`requested model -> the fallback that answered`, for the rest of this
        provider's life. The vendor's advice is to stay on the model that accepted
        rather than re-ask the one that refused on every turn, and a refusal that
        arrives before any output is still billed in some categories. A desk running
        its own thirty-step loop on the model that just declined would otherwise pay
        for thirty refusals. `switch.apply` clears it: a switch the person asked for
        outranks a pin a refusal made."""
        pins = self.__dict__.get("_pinned")
        if pins is None:
            pins = self.__dict__["_pinned"] = {}
        return pins

    def chain(self, model: str) -> list[str]:
        """The models to ask, in order: the one named (or the fallback pinned for it),
        then every fallback not yet in the list."""
        asked = self.pinned.get(model, model)
        out = [asked]
        for alt in self.fallbacks:
            if alt not in out and alt != model:
                out.append(alt)
        return out

    def with_fallback(self, model: str, send: Callable[[str], "Turn"]) -> "Turn":
        """`send(model)`; on a refusal, `send` again down `fallbacks`, pinning the
        model that answers. The turn that comes back names the model that served it and
        every model that declined before it. When the whole chain declines, the last
        refusal is returned with the earlier ones listed on it."""
        declined: list[tuple[str, str]] = []
        turn: Turn | None = None
        for asked in self.chain(model):
            turn = send(asked)
            # The id as it was asked for, not the dated one the vendor resolves it to:
            # this is what the next request (and a hosted allowlist) will be given.
            turn.model = asked
            if turn.stop_reason != "refusal":
                if asked != model:
                    self.pinned[model] = asked
                break
            declined.append((asked, _why(turn)))
        assert turn is not None
        if turn.stop_reason == "refusal":
            # The last refusal is the turn; the models before it are the record.
            declined = declined[:-1]
        turn.declined = declined
        return turn

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
        raise NotImplementedError

    def complete(self, *, model: str, system: str, prompt: str, max_tokens: int) -> str:
        """One short, unstreamed answer -- what the front desk needs."""
        raise NotImplementedError

    def probe(self, model: str, vision: bool = False) -> str:
        """Confirm the key, the endpoint and the model id together, as cheaply as the
        API allows. With `vision`, also that the model accepts an image: the agent
        looks at geometry and mesh renders, and a model that cannot see them works
        blind without saying so. Returns a one-line description; raises
        `ProviderError` -- with `CANNOT_SEE` in the message when it is the image that
        was refused."""
        raise NotImplementedError


def _why(turn: "Turn") -> str:
    """A refusal's reason as one line: the category, then the vendor's sentence."""
    parts = [p for p in (turn.stop_category, turn.stop_explanation) if p]
    return ": ".join(parts) if parts else "no explanation given"


def refusal_line(turn: "Turn") -> str:
    """What to tell a person about a turn that is a refusal, or that was served only
    after one. Empty for an ordinary turn."""
    if turn.stop_reason == "refusal":
        chain = [*turn.declined, (turn.model or "the model", _why(turn))]
        said = "; ".join(f"{model} ({why})" for model, why in chain)
        if len(chain) == 1:
            return f"The model declined this request: {_why(turn)}"
        return f"Every model declined this request -- {said}"
    if turn.declined:
        said = "; ".join(f"{model} ({why})" for model, why in turn.declined)
        return f"{said} declined this request; {turn.model} answered it instead."
    return ""


CANNOT_SEE = "cannot see images"

PROBE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
"""A 1x1 PNG: the smallest thing that is unmistakably an image."""


def cannot_see(model: str, detail: str = "") -> "ProviderError":
    why = f" ({detail})" if detail else ""
    return ProviderError(
        f"{model} {CANNOT_SEE}{why}. Reynolds looks at geometry and mesh renders, so it "
        "needs a model that can -- Claude, GPT-5, or another vision model.",
        400,
    )


def _kind(block: Any) -> str:
    return _get(block, "type") or ""


def _get(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def neutral_blocks(content: list[Any]) -> list[Any]:
    """A provider's own blocks reduced to the dataclasses every provider can read.

    Thinking is dropped on purpose: another provider cannot verify it, and several
    refuse a turn that carries reasoning they did not produce.
    """
    out: list[Any] = []
    for block in content:
        kind = _kind(block)
        if kind == "text":
            out.append(TextBlock(text=_get(block, "text") or ""))
        elif kind == "tool_use":
            out.append(
                ToolUseBlock(
                    id=_get(block, "id") or "",
                    name=_get(block, "name") or "",
                    input=dict(_get(block, "input") or {}),
                )
            )
    return out


def split_result(content: Any) -> tuple[str, list[dict[str, Any]]]:
    """A tool result's text and its images, whichever shape it came in."""
    if isinstance(content, str) or content is None:
        return content or "", []
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "image":
            images.append(block)
    return "\n".join(texts), images
