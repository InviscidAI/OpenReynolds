"""The tool-use loop.

A manual loop rather than any SDK's tool runner: watch mode, capture hooks and the
mid-conversation operator channel all want control a runner does not expose. Which
model answers is the provider's business (`llm/`); the loop sees turns, blocks and
stop reasons and nothing about whose API produced them.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from .config import CONTEXT_REFRESH_FRACTION, CONTEXT_WINDOW_TOKENS, Config
from .llm import BadRequest, Listener, Turn, make_provider
from .prompt import system_prompt
from .store import Store
from .tools import TOOLS, ToolContext, describe, dispatch
from .view import View

MAX_TOKENS = 64_000

KEEP_LIVE_IMAGES = 5
"""How many of the most recently read images keep their pixels in the thread. Older
ones are reduced to their text description; the model has already reasoned on them and
re-sending megabytes of base64 every turn is what made requests large enough to be
refused. The path survives, so a picture can be re-read deliberately if it matters.

Two, until it was measured. Carrying five images costs about 5k extra tokens a turn at
the cache-read rate -- a tenth of a cent -- and each eviction it avoids rewrites 3k to
12k tokens at the cache-write rate. Keeping more is the cheap side of that trade."""

LIVE_IMAGE_BUDGET = 3_000_000
"""Bytes of image data the thread may carry before any eviction happens at all.

The guard exists for a real incident: a session accumulated twenty-one images, five
megabytes of them, and the API began refusing the requests. But it was triggered by a
*count*, so it fired on the third picture of a study and every picture after -- and each
firing rewrites a block in the middle of the conversation, which invalidates the prompt
cache from that point on. Measured over one study: five evictions forced 32,920 tokens
to be re-written to save re-reading about 107,000 cached ones, a net loss of roughly
3.5x, and it got worse the longer the study ran because the prefix being destroyed grew.

Bytes are what actually failed, so bytes are what triggers it now. Under the budget
nothing is evicted and the cache prefix is never touched."""


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
        progress: Any | None = None,
    ):
        self.cfg = cfg
        self.ctx = ctx
        self.store = store
        self.view = view
        self.capture = capture
        self.interject = interject
        self.progress = progress
        """Told what this thread is doing -- thinking, writing, in a tool -- so the
        bar can say so with a clock on it. Presentation; it hears, never speaks."""

        headers = {"X-Study-Id": store.session.study_id}
        # Without a timeout a stalled connection is indistinguishable from a model
        # thinking hard, and the session stops dead with nothing said. A failure is
        # recoverable -- it is reported and the thread survives -- silence is not.
        self.provider = make_provider(cfg, timeout=cfg.llm_timeout_s, default_headers=headers)
        self.window = cfg.context_window or CONTEXT_WINDOW_TOKENS
        """How much thread this model can hold; a refresh is due at a fraction of it."""
        self.messages: list[dict[str, Any]] = []
        self.context_tokens = 0
        self.token_totals: dict[str, int] = {}
        """Tokens so far, split by what each class costs -- input, cache_read,
        cache_write, output.

        `context_tokens` is one number for four things whose prices span 250x, so it
        cannot say whether the prompt cache is working; on this workload caching is
        worth 7.6-8x, and cache reads are 68-80% of a study's model bill. Kept here so
        `/status` can show the share, which is the cheapest possible alarm on the one
        mechanism holding the bill down."""
        self._no_system_role = False
        """Whether this endpoint has already refused a mid-conversation `system` turn.

        Set the first time one is rejected, so the cost of finding out is one round trip
        per session rather than one per harness fact. Not a provider capability flag:
        the same vendor accepts it on one endpoint and not another, so it is learned
        from the answer rather than declared in a table."""
        self.api_failures = 0
        """Consecutive model-API failures. Reset on any turn that completes; used to
        escalate from "the thread is intact" to a plain explanation once it is clearly
        not a one-off (a rate limit, a usage cap) rather than a blip."""
        self.blocked_reason: str | None = None
        """Why the model service refused the last call, when waiting cannot fix it.

        A 429 or a dropped connection is worth trying again; a 402, a 401 or a bad
        model id is the service answering a question about the account or the request
        and it will answer the same way in a minute. A live session spent twenty-six
        minutes discovering that: the account budget ran out mid-study, and the
        harness woke the model ninety more times, each one refused, each one silent
        on the page while the person typed "whats going on?". Set here so the caller
        can stop asking; cleared by any turn that completes and by the user speaking.
        """

    @property
    def client(self) -> Any:
        """The provider's SDK client -- what a test replaces with a scripted one."""
        return self.provider.client

    @client.setter
    def client(self, value: Any) -> None:
        self.provider.client = value

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
        if self.messages and self.messages[-1]["role"] == "user" and not self._no_system_role:
            self.messages.append({"role": "system", "content": text})
        else:
            self.messages.append({"role": "user", "content": _as_operator_text(text)})
        self._record("event", text)

    @staticmethod
    def _fold_system(message: dict[str, Any]) -> dict[str, Any]:
        """A `system` turn as a marked user turn; anything else passed straight through.

        Passed through as the *same object*, so the bytes of every other message are
        untouched and the cached prefix survives the rewrite."""
        if message.get("role") != "system":
            return message
        return {"role": "user", "content": _as_operator_text(message["content"])}

    # -- the turn --------------------------------------------------------------

    def run(self) -> Turn:
        """Stream turns and dispatch tools until the model ends its turn."""
        step = 0
        while True:
            step += 1
            started = time.monotonic()
            response = self._send()

            if response.stop_reason == "refusal":
                reason = response.stop_explanation or "no explanation given"
                self.view.notice(f"The model declined this request: {reason}")
                return response

            self.messages.append(response.as_message())
            self._record("assistant", response.text)

            if response.stop_reason == "max_tokens":
                # Otherwise a turn cut off at the output cap is indistinguishable from
                # a finished one. Say so; whether to carry on is the model's call and
                # the user's, not the harness's.
                self.view.notice(
                    f"This turn stopped at the {MAX_TOKENS:,}-token output cap, "
                    "so it is incomplete."
                )

            tool_uses = response.tool_calls
            if not tool_uses:
                self.view.step(step, time.monotonic() - started, 0)
                return response

            results: list[Any] = []
            for block in tool_uses:
                results.append(self._run_tool(block))

            # One round of think-then-act is over. Marking where each ends is what
            # makes the loop legible: without it the activity pane is an undivided
            # column of tool calls, and there is no telling a turn that took three
            # rounds from one that took thirty.
            self.view.step(step, time.monotonic() - started, len(tool_uses))

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

    def _send(self) -> Turn:
        """One streamed request, printing as it arrives."""
        try:
            return self._stream()
        except BadRequest as exc:
            if "system" not in str(exc).lower():
                raise
            # This endpoint has no mid-conversation system role — fold those turns into
            # user messages and carry on. Remembered, so it is learned once: `inform()`
            # appends a fresh `role: "system"` every time it is called, so without the
            # latch every job-end wake and every thread refresh paid for its own
            # rejected request, forever. The Anthropic Messages API is one of these --
            # `messages` takes only user and assistant -- so this is the ordinary path
            # for every Anthropic-family provider, not an exotic one.
            self._no_system_role = True
            self.messages = [self._fold_system(m) for m in self.messages]
            return self._stream()

    def _evict_old_images(self, keep: int = KEEP_LIVE_IMAGES) -> None:
        """Drop the pixels of images the model has already looked at.

        A `read_file` on a render comes back as an image block, and it stays in the
        thread and is re-sent, in full, on every turn after it — a two-megabyte PNG
        looked at once becomes two megabytes re-uploaded fifty times. One live run
        ended up carrying twenty-one images, five megabytes of them, in every
        request; the requests got large enough that the model API began refusing
        them and the session stopped answering entirely.

        The picture only has to be in the thread while the model is reasoning about
        it. After that its own words about it are what carry forward, so the base64
        is replaced with the one-line description that rode alongside it — the path
        stays, so it can be looked at again deliberately if it ever matters. The most
        recent `keep` images are left whole, because those are the ones a turn in
        flight is most likely still working from. Idempotent: an evicted result is
        text and is not found again.
        """
        image_results = []
        carried = 0
        for message in self.messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                inner = block.get("content")
                if isinstance(inner, list) and any(
                    isinstance(b, dict) and b.get("type") == "image" for b in inner
                ):
                    image_results.append(block)
                    carried += _image_bytes(inner)
        # Nothing is evicted while the thread is under the byte budget. Every eviction
        # rewrites a block in the middle of the conversation and so invalidates the
        # prompt cache from there on; doing it on a count meant paying that on the third
        # picture of a study, to save re-reading tokens that cost a tenth as much.
        if carried <= LIVE_IMAGE_BUDGET:
            return
        for block in image_results[: max(0, len(image_results) - keep)]:
            inner = block["content"]
            note = next(
                (b.get("text", "") for b in inner if isinstance(b, dict) and b.get("type") == "text"),
                "",
            )
            block["content"] = f"[image no longer in context to save space] {note}".strip()

    def _busy(self, kind: str, label: str = "", **facts: Any) -> None:
        if self.progress is not None:
            self.progress.begin(kind, label, **facts)

    def _unbusy(self) -> None:
        if self.progress is not None:
            self.progress.idle()

    def _stream(self) -> Turn:
        # Shed the pixels of images already looked at before building the request,
        # so the thread does not grow without bound and the requests stay a size the
        # API will accept.
        self._evict_old_images()
        # The round trip before the first event is thinking as far as anyone
        # watching can tell, and a clock on it is what tells a slow model from a
        # dead connection.
        self._busy("thinking")
        writing = False

        def text(delta: str) -> None:
            nonlocal writing
            if not writing:
                writing = True
                self._busy("writing")
            self.view.text_delta(delta)

        response = self.provider.stream(
            model=self.cfg.model,
            system=system_prompt(),
            messages=self.messages,
            tools=TOOLS,
            effort=self.cfg.effort,
            max_tokens=MAX_TOKENS,
            listener=Listener(
                thinking_begin=lambda: self.view.thinking_begin(),
                thinking=lambda delta: self.view.thinking_delta(delta),
                text=text,
            ),
        )

        self.view.turn_end()
        self._unbusy()
        self._account(response)
        return response

    def _run_tool(self, block: Any) -> dict[str, Any]:
        self.view.tool(block.name, _summarize(block.input))
        tool_input = dict(block.input)
        self._busy(
            "tool",
            block.name,
            cmd=str(tool_input.get("cmd") or ""),
            cwd=str(tool_input.get("cwd") or self.ctx.home or ""),
        )
        try:
            with _ticking(self.view, block.name):
                content, is_error = dispatch(self.ctx, block.name, tool_input)
        finally:
            self._unbusy()
        # A tool result can be content blocks rather than text -- an image, for one --
        # and those go to the model as they are. What gets written down is a
        # description: a megabyte of base64 in the message log helps nobody read it.
        written = describe(content)
        self._record(
            "tool",
            {"tool": block.name, "input": dict(block.input), "output": written, "error": is_error},
        )
        if is_error:
            self.view.tool_error(written.splitlines()[0] if written else "failed")
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content or "(no output)",
            **({"is_error": True} if is_error else {}),
        }

    # -- context ---------------------------------------------------------------

    def _account(self, response: Turn) -> None:
        for name, count in (response.tokens or {}).items():
            self.token_totals[name] = self.token_totals.get(name, 0) + count
        if not response.context_tokens:
            return
        self.context_tokens = response.context_tokens
        self.view.usage(self.context_tokens, self.context_tokens / self.window)

    @property
    def needs_refresh(self) -> bool:
        return self.context_tokens > self.window * CONTEXT_REFRESH_FRACTION

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
        self.restart(blurb)

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
        """Begin a fresh thread from a factual situation blurb.

        Everything the old thread carried goes here, and `ctx.echoes` is part of what
        it carried. The echo table maps a digest to the call whose bytes are still in
        the conversation, and it is the one piece of thread state that does not live
        on the thread: `ToolContext` is built once a session and never replaced. Left
        standing, every digest recorded before the reset stays matchable after it, so
        the first thing a freshly briefed model does -- go back and read the solver
        log it no longer remembers -- is answered with a pointer to a call that is no
        longer in any conversation, and asking again returns the same pointer for the
        rest of the session.
        """
        self.messages = []
        self.context_tokens = 0
        self.ctx.echoes.clear()
        self.brief(blurb)

    # -- capture ---------------------------------------------------------------

    def _record(self, role: str, content: Any) -> None:
        seq = self.store.append_message(role, content)
        if self.capture:
            self.capture.message(seq, role, content)


TICK_EVERY_S = 10.0
"""How often a running tool call says it is still running."""


@contextmanager
def _ticking(view: View, name: str, every: float = TICK_EVERY_S):
    """Say that a slow tool call is still going, for as long as it goes on.

    A command may take five minutes. One line when it starts and nothing after that is
    indistinguishable from a hang: the user reaches for ctrl+C, and anything watching
    the terminal concludes the turn ended and talks over it. The elapsed count is a
    fact about the harness, not about the work, and nothing depends on it being read.
    """
    stop = threading.Event()
    started = time.monotonic()

    def tick() -> None:
        while not stop.wait(every):
            view.stage(f"{name} still running, {time.monotonic() - started:.0f}s")

    thread = threading.Thread(target=tick, name=f"tick-{name}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()


def _image_bytes(blocks: list) -> int:
    """Roughly how many bytes of picture a tool result is carrying.

    base64 is four characters per three bytes, and it is the encoded length that
    travels, so the encoded length is what the budget is measured in."""
    total = 0
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "image":
            source = block.get("source") or {}
            total += len(source.get("data") or "")
    return total


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
