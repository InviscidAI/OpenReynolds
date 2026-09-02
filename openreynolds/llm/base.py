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

from dataclasses import dataclass, field
from typing import Any, Callable

STOP_REASONS = ("end_turn", "tool_use", "max_tokens", "refusal")


class ProviderError(Exception):
    """A model API failed. `status_code` is set when the API answered at all."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


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
