"""The Anthropic tool-use loop.

A manual loop rather than the SDK's tool runner: watch mode, capture hooks and the
mid-conversation operator channel all want control the runner does not expose.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

from .config import CONTEXT_REFRESH_FRACTION, CONTEXT_WINDOW_TOKENS, Config
from .prompt import system_prompt
from .store import Store
from .tools import TOOLS, ToolContext, dispatch
from .view import View

MAX_TOKENS = 64_000


class Loop:
    """One conversation thread against one workspace."""

    def __init__(
        self,
        cfg: Config,
        ctx: ToolContext,
        store: Store,
        view: View,
        capture: Any | None = None,
        interject: Callable[[], str | None] | None = None,
    ):
        self.cfg = cfg
        self.ctx = ctx
        self.store = store
        self.view = view
        self.capture = capture
        self.interject = interject

        headers = {"X-Study-Id": store.session.study_id}
        self.client = anthropic.Anthropic(
            api_key=cfg.anthropic_api_key or None,
            base_url=cfg.llm_base_url or None,
            default_headers=headers,
        )
        self.system = [
            {"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}
        ]
        self.messages: list[dict[str, Any]] = []
        self.context_tokens = 0

    # -- inbound ---------------------------------------------------------------

    def say(self, text: str) -> None:
        """Add a turn from the user."""
        self.messages.append({"role": "user", "content": text})
        self._record("user", text)

    def brief(self, text: str) -> None:
        """Open a thread with harness-assembled facts.

        Sent as a user turn because a thread cannot begin with anything else, but
        recorded as an event: the user did not say this, and a transcript that
        claims otherwise misleads whatever reads it later.
        """
        self.messages.append({"role": "user", "content": text})
        self._record("event", text)

    def inform(self, text: str) -> None:
        """Add harness-authored facts.

        Preferred channel is a mid-conversation `role: "system"` message: it carries
        operator authority rather than impersonating the user, and it sits after the
        cached history so the prefix survives. The API only accepts one directly after
        a user turn, so elsewhere this falls back to a marked user message — and
        `_send` degrades again if the model does not support the role at all.
        """
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.append({"role": "system", "content": text})
        else:
            self.messages.append({"role": "user", "content": _as_operator_text(text)})
        self._record("event", text)

    # -- the turn --------------------------------------------------------------

    def run(self) -> Any:
        """Stream turns and dispatch tools until the model ends its turn."""
        while True:
            response = self._send()

            if response.stop_reason == "refusal":
                detail = getattr(response, "stop_details", None)
                reason = getattr(detail, "explanation", None) or "no explanation given"
                self.view.notice(f"The model declined this request: {reason}")
                return response

            self.messages.append({"role": "assistant", "content": response.content})
            self._record("assistant", _text_of(response))

            if response.stop_reason == "max_tokens":
                # Otherwise a turn cut off at the output cap is indistinguishable from
                # a finished one. Say so; whether to carry on is the model's call and
                # the user's, not the harness's.
                self.view.notice(
                    f"This turn stopped at the {MAX_TOKENS:,}-token output cap, "
                    "so it is incomplete."
                )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return response

            results: list[Any] = []
            for block in tool_uses:
                results.append(self._run_tool(block))

            # Tool results have to come first in this message, but a text block may
            # follow them. That is how something typed while the model is working
            # reaches it at the next turn instead of sitting unread until the whole
            # turn ends -- the difference between being heard and being ignored.
            said = self.interject() if self.interject else None
            if said:
                results.append({"type": "text", "text": said})
                self.view.interjection(said)
                self._record("user", said)

            self.messages.append({"role": "user", "content": results})

    def _send(self) -> Any:
        """One streamed request, printing as it arrives."""
        try:
            return self._stream()
        except anthropic.BadRequestError as exc:
            if "system" not in str(exc).lower():
                raise
            # This model has no mid-conversation system role — fold those turns into
            # user messages and carry on.
            self.messages = [
                {"role": "user", "content": _as_operator_text(m["content"])}
                if m["role"] == "system"
                else m
                for m in self.messages
            ]
            return self._stream()

    def _stream(self) -> Any:
        with self.client.messages.stream(
            model=self.cfg.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            messages=self.messages,
            tools=TOOLS,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.cfg.effort},
            cache_control={"type": "ephemeral"},
        ) as stream:
            for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "thinking":
                    self.view.thinking_begin()
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        self.view.thinking_delta(event.delta.thinking)
                    elif event.delta.type == "text_delta":
                        self.view.text_delta(event.delta.text)
            response = stream.get_final_message()

        self.view.turn_end()
        self._account(response)
        return response

    def _run_tool(self, block: Any) -> dict[str, Any]:
        self.view.tool(block.name, _summarize(block.input))
        content, is_error = dispatch(self.ctx, block.name, dict(block.input))
        self._record(
            "tool",
            {"tool": block.name, "input": dict(block.input), "output": content, "error": is_error},
        )
        if is_error:
            self.view.tool_error(content.splitlines()[0] if content else "failed")
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content or "(no output)",
            **({"is_error": True} if is_error else {}),
        }

    # -- context ---------------------------------------------------------------

    def _account(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.context_tokens = (
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (usage.output_tokens or 0)
        )
        self.view.usage(self.context_tokens, self.context_tokens / CONTEXT_WINDOW_TOKENS)

    @property
    def needs_refresh(self) -> bool:
        return self.context_tokens > CONTEXT_WINDOW_TOKENS * CONTEXT_REFRESH_FRACTION

    def refresh(self, blurb: str) -> None:
        """Rebuild the thread — the same move a resume makes.

        The model is told first, so it can put anything it wants to keep on disk. What
        survives is the filesystem plus whatever notes it chose to write; nothing here
        summarizes its reasoning for it.
        """
        self.view.info("- refreshing the thread -")
        self.inform(
            "This conversation thread is being refreshed to free up context. The "
            "workspace is untouched and this session continues. Anything you want to "
            "carry forward can go on disk now; the next message starts a fresh thread."
        )
        self.run()
        self.messages = []
        self.context_tokens = 0
        self.brief(blurb)

    def settle(self) -> None:
        """Answer any tool call left dangling by an interrupted turn.

        The API refuses a thread whose last assistant turn asks for a tool and never
        gets an answer, so a turn cut short by a network failure would otherwise be
        unresumable. What the model gets back is the fact: it did not run.
        """
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return
        content = self.messages[-1]["content"]
        pending = [b for b in content if getattr(b, "type", None) == "tool_use"]
        if not pending:
            return
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "This call did not run: the turn was interrupted.",
                        "is_error": True,
                    }
                    for block in pending
                ],
            }
        )

    def restart(self, blurb: str) -> None:
        """Begin a fresh thread from a factual situation blurb."""
        self.messages = []
        self.context_tokens = 0
        self.brief(blurb)

    # -- capture ---------------------------------------------------------------

    def _record(self, role: str, content: Any) -> None:
        seq = self.store.append_message(role, content)
        if self.capture:
            self.capture.message(seq, role, content)


def _text_of(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def _as_operator_text(text: Any) -> str:
    body = text if isinstance(text, str) else json.dumps(text, default=str)
    return f"[from the harness, not the user]\n{body}"


def _summarize(tool_input: Any, width: int = 100) -> str:
    """A one-line echo of a tool call for the terminal."""
    if isinstance(tool_input, dict):
        for key in ("cmd", "path", "job_id", "paths"):
            if key in tool_input:
                value = tool_input[key]
                text = " ".join(value) if isinstance(value, list) else str(value)
                break
        else:
            text = json.dumps(tool_input, default=str)
    else:
        text = str(tool_input)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


ToolFactory = Callable[[], ToolContext]
