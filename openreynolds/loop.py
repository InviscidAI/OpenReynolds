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
from contextlib import contextmanager, nullcontext
from typing import Any, Callable

from . import modes
from .config import CONTEXT_REFRESH_FRACTION, CONTEXT_WINDOW_TOKENS, Config
from .llm import BadRequest, Listener, Turn, make_provider
from .prompt import system_prompt
from .store import Store
from .tools import TOOLS, ToolContext, describe, dispatch, tools_for
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
        self.gate: Any | None = None
        """Held around each tool call (`mirror.Gate`), so the background mirror can
        stand aside for it: its transfers share the container with the command and
        the command waited behind them -- a 27 s finish step took five minutes."""
        self.approver: Any | None = None
        """Puts a call to the person when they chose to be consulted (`modes.py`,
        `approval.Approver`). None when nobody can answer. Never consulted in auto."""
        self._notes: list[str] = []
        """Harness facts waiting for a place in the thread. Said mid-turn (a `/mode`
        switch typed while tools run) they cannot be a message of their own: the next
        message has to be the tool results. They ride in that message instead."""
        self._typed: list[str] = []
        """What the person typed mid-turn that is for the model, waiting to ride after
        this batch's tool results (`_turns`). Filled by `_gather`, which drains the
        inbox before each tool call, after the batch, and -- through `heard` -- every
        second a tool is waiting. One list rather than a local, because a waiting tool
        has to be able to ask whether it is empty."""
        self._mid_turn = False
        self._posted: list[str] = []
        """Harness facts said from another thread (`post`), waiting for this thread to
        pick them up at one of its own safe points. `messages` is read while a request
        is being built and appended to when the answer lands, and neither of those
        may be interleaved with a write from elsewhere."""
        self._posted_lock = threading.Lock()

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
        self.images_dropped = False
        """Whether this stretch of failures has already had its pictures stripped.

        `drop_images` is a one-shot repair, not a retry policy. If a turn still fails
        after every image is gone then the images were not the problem, and trying the
        same thing again would be the twenty-six-minute loop `blocked_reason` exists to
        prevent. Cleared by any turn that completes, so a later refusal in a longer
        session gets its own attempt."""
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
        self.pending_model: Any | None = None
        """A model or provider switch the person asked for (`/model`), already checked,
        waiting for the next `run` to start (`switch.apply`). Applied there and nowhere
        else, so a turn is never half one model and half another."""
        self.refresh_due = False
        """The pending switch's window cannot hold this thread, so it is refreshed on
        the current model first; `pending_model` waits until that is done."""
        self.running = False
        """Whether a `run` is in flight -- what tells the person a switch applies now
        or when the current turn ends."""

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

    def tell(self, text: str) -> None:
        """`inform`, from wherever it is safe to be.

        Between a turn's tool calls the next message must be their results, so a fact
        said then waits and rides in that message as a marked text block. Anywhere
        else it is an ordinary `inform`."""
        if self._mid_turn:
            self._notes.append(text)
            self._record("event", text)
        else:
            self.inform(text)

    def post(self, text: str) -> None:
        """`tell`, from a thread that is not this loop's.

        The workspace coming up, and the facts about it that the briefing could not
        wait for, are learned on a background thread while the model may be mid-turn
        on this one. `tell` straight from there would race the request being built;
        this only queues, and `_drain_posted` hands the fact over at the next point
        where this thread would have said it itself -- before a request goes out, or
        between one tool call and the next. Nothing is lost if no turn is running:
        the note is the first thing the next turn's request carries."""
        with self._posted_lock:
            self._posted.append(text)

    def _drain_posted(self) -> None:
        with self._posted_lock:
            posted, self._posted = self._posted, []
        for text in posted:
            self.tell(text)

    def _take_notes(self) -> list[dict[str, Any]]:
        notes = [{"type": "text", "text": _as_operator_text(note)} for note in self._notes]
        self._notes = []
        return notes

    def set_mode(self, mode: str) -> str | None:
        """Switch how much the person is consulted. Returns the mode, or None if the
        name is not one.

        Takes effect at the next tool call: `_consult` reads `ctx.mode` every time.
        Entering structured mode starts it afresh, so a plan has to be approved in it
        before compute is spent. The model is told in the harness's voice, because the
        tool list and what gets held both change under it."""
        canonical = modes.normalize(mode)
        if canonical is None:
            return None
        previous = self.ctx.mode
        self.ctx.mode = canonical
        self.cfg.mode = canonical
        if canonical == modes.STRUCTURED and previous != modes.STRUCTURED:
            self.ctx.plan_approved = False
        self.store.session.mode = canonical
        self.store.save()
        self.view.mode(canonical)
        if canonical != previous:
            self.tell(modes.switched(canonical))
        return canonical

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
        if self.pending_model is not None and not self.refresh_due:
            # Every earlier assistant turn is complete here, so this is the one place a
            # switch cannot split a turn between two models.
            from . import switch

            switch.apply(self)
        self.running = True
        try:
            return self._turns()
        finally:
            self.running = False

    def _turns(self) -> Turn:
        step = 0
        while True:
            step += 1
            started = time.monotonic()
            # Anything another thread has learned since the last request rides in
            # this one: a `system` turn here, after the user's message or the tool
            # results, is where `inform` would have put it.
            self._drain_posted()
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
            self._mid_turn = True
            try:
                for block in tool_uses:
                    # What was typed so far is read before each call, not only after the
                    # batch. Two things depend on it: a `/mode` typed while an earlier
                    # call ran governs this one (a switch applies from the next tool
                    # call), and a line typed before a question exists is an
                    # interjection, never the answer to a question the person has not
                    # seen yet (`Approver.ask` reads only what arrives after this).
                    self._gather()
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
                self._gather()
                said = "\n".join(self._typed) or None
                self._typed = []
                if said:
                    results.append({"type": "text", "text": said})
                    self.view.interjection(said)
                    self._record("user", said)
                results.extend(self._take_notes())
            finally:
                self._mid_turn = False

            self.messages.append({"role": "user", "content": results})

    def _gather(self) -> None:
        """Drain what has been typed (`interject`): commands are answered on the spot,
        and words for the model are kept in `_typed` to ride with this batch's results.
        What other threads posted meanwhile rides with them too (`_notes`)."""
        self._drain_posted()
        said = self.interject() if self.interject else None
        if said:
            self._typed.append(said)

    def heard(self) -> bool:
        """Whether the person has said something the model has not seen yet.

        What a tool that holds its answer asks, once a second, to know whether to stop
        holding it (`ToolContext.on_wait_input`; `job_check` with `wait_s`, `mesh_wait`).
        It drains the inbox the way the loop does between tool calls -- commands are
        answered on the spot, words for the model are kept for this batch's results --
        and answers whether anything is kept. So a wait that ends on it ends for words
        the model is about to read, in the same message as the result that ended.

        WHY THIS IS THE LOOP'S TO ANSWER. It used to be the reader's `pending()`: is
        there anything in the queue, without taking it. But the loop's own drain does
        not take everything. `/exit` is put back for whoever waits at the prompt, an
        EOF likewise, and a `/status` is answered without a word reaching the model.
        Measured in production (study 20260921-033019-e1b4): a line the drain put back
        sat in the queue for the rest of the turn, `pending()` answered true to every
        wait from then on, and `mesh_wait` returned three times in nine seconds with
        `[waited 0s] [the user said something]` -- and the model, handed nothing the
        person had said, concluded the tool did not wait and paced the remaining
        thirty-seven minutes with `bash sleep`. The same shape ran side by side with
        an unaffected session (20260921-033356-076b) whose `job_check(wait_s=...)`
        waited its full 120-300 s throughout. A line that is nobody's to deliver to
        the model cannot end a wait on the model's behalf, and the only way to know
        which kind of line it is, is to drain it.
        """
        self._gather()
        return bool(self._typed)

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
            tools=tools_for(self.ctx),
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

    def _consult(self, name: str, tool_input: dict[str, Any]) -> str | None:
        """None when this call may run; otherwise the tool result saying why it did not.

        The only place a tool call is ever held, and it holds only what the person
        chose to have held (`modes.py`). In auto this returns None without looking.

        The mesh desk runs its own bash loop on its own model client; the gate is on
        the outer `mesh` call, which is enough -- approving that call is approving the
        mesh desk's work, and nothing it does inside spends a job."""
        mode = self.ctx.mode
        if not modes.gated(mode, name):
            return None
        if mode == modes.STRUCTURED:
            return None if self.ctx.plan_approved else modes.held(name)
        if self.approver is None:
            return (f"This {name} call did not run: the session is in ask-before-compute "
                    "mode and nobody is here to answer.")
        title, detail = _question(name, tool_input, self.ctx.home)
        self._busy("waiting")
        try:
            decision = self.approver.ask("job" if name == "job_start" else "mesh", title, detail)
        finally:
            self._unbusy()
        if decision.approved:
            if decision.all:
                self.set_mode(modes.AUTO)
            return None
        if decision.note:
            return f"This {name} call did not run: the person declined it. What they said: {decision.note}"
        return f"This {name} call did not run: the person declined it and gave no reason."

    def _run_tool(self, block: Any) -> dict[str, Any]:
        self.view.tool(block.name, _summarize(block.input))
        tool_input = dict(block.input)
        refused = self._consult(block.name, tool_input)
        if refused is not None:
            content, is_error = refused, True
        elif block.name == modes.CHECKPOINT:
            # A question to a person: no mirror gate held while they think, and no
            # "still running" ticks for a wait that is theirs.
            self._busy("waiting")
            try:
                content, is_error = dispatch(self.ctx, block.name, tool_input, call_id=block.id)
            finally:
                self._unbusy()
        else:
            content, is_error = self._dispatch(block, tool_input)
        return self._result(block, content, is_error)

    def _dispatch(self, block: Any, tool_input: dict[str, Any]) -> tuple[Any, bool]:
        self._busy(
            "tool",
            block.name,
            cmd=str(tool_input.get("cmd") or ""),
            cwd=str(tool_input.get("cwd") or self.ctx.home or ""),
        )
        try:
            with _holding(self.gate), _ticking(self.view, block.name):
                return dispatch(self.ctx, block.name, tool_input, call_id=block.id)
        finally:
            self._unbusy()

    def _result(self, block: Any, content: Any, is_error: bool) -> dict[str, Any]:
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

    def add_tokens(self, tokens: dict) -> None:
        """Count model usage made on this session's behalf outside the loop -- the
        mesh desk's steps -- so `status` shows what the study actually spent."""
        for name, count in (tokens or {}).items():
            self.token_totals[name] = self.token_totals.get(name, 0) + int(count)

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
        self.messages = []
        self.context_tokens = 0
        self.brief(blurb)
        # A switch that was waiting for a smaller thread can go ahead at the next run.
        self.refresh_due = False

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
                ]
                # A fact said during the interrupted turn still belongs in the thread.
                + self._take_notes(),
            }
        )

    def drop_images(self) -> int:
        """Replace every image in the thread with a note saying it was dropped.

        Returns how many were replaced, so the caller can tell "there was something to
        fix" from "this 400 was about something else".

        WHY THIS EXISTS. One picture the API will not accept used to cost a whole
        session. On 2026-09-12 a 2 h 23 m run ended on `400 invalid_request_error:
        Could not process image`, and the harness was right that a 400 is not
        survivable by WAITING -- the same bytes get the same answer forever -- but
        wrong that it is not survivable at all. The bad bytes are sitting in the
        thread, and a thread is a thing this process owns and can edit. Take the image
        out and the very next call goes through.

        EVERY image goes, not the guilty one, because the API does not say which it
        objected to and guessing wrong means another refused call. Images are the
        cheapest thing in a thread to lose: the file is still on the instance and the
        path is still in the text beside it, so the model can look again on purpose.
        What it must NOT do is silently lose the knowledge that it ever looked, which
        is why each one leaves a sentence behind rather than a hole.

        `images.incomplete` (images.py) is the other half of the same fix and the one
        that should keep this from being needed: it stops a half-written figure being
        attached in the first place. This is the backstop for every other reason an
        image can be refused, including the ones nobody has met yet.
        """
        dropped = 0

        def clean(content: Any) -> Any:
            nonlocal dropped
            if not isinstance(content, list):
                return content
            out = []
            for block in content:
                kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if kind == "image":
                    dropped += 1
                    out.append({
                        "type": "text",
                        "text": "[an image was here. The model API refused it, so it was "
                                "removed to keep this session alive. The file is still on "
                                "the instance: read the path again to look at it.]",
                    })
                elif kind == "tool_result" and isinstance(block, dict):
                    out.append({**block, "content": clean(block.get("content"))})
                else:
                    out.append(block)
            return out

        for message in self.messages:
            if isinstance(message, dict):
                message["content"] = clean(message.get("content"))
        return dropped

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


TICK_EVERY_S = 10.0
"""How often a running tool call says it is still running."""


def _holding(gate: Any | None):
    """The gate held for the duration, or nothing at all when there is no mirror."""
    return gate.held() if gate is not None else nullcontext()


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


def _question(name: str, tool_input: dict[str, Any], home: str) -> tuple[str, str]:
    """The title and detail a gated call is put to the person with: what would run."""
    if name == "job_start":
        cmd = str(tool_input.get("cmd") or "")
        label = tool_input.get("name") or (cmd.splitlines()[0][:60] if cmd else "")
        return f"Start a job: {label}", f"cmd: {cmd}\ncwd: {tool_input.get('cwd') or home}"
    request = str(tool_input.get("request") or "")
    return "Build a mesh", f"{request}\ncase: {tool_input.get('case') or 'mesh'}"


ToolFactory = Callable[[], ToolContext]
