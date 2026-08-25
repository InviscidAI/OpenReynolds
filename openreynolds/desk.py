"""The front desk: a second, cheap agent whose only job is to talk to you.

The problem it solves, in the words of the person who hit it: you send a message and
"it still doesn't stop and respond -- it keeps doing its own work." That is not the
main agent being rude. It is structural. The main agent runs one thing at a time on one
thread: a `bash` call can hold that thread for up to 300 seconds, a thinking phase for
minutes, and until the current step returns there is nothing on that thread that can
read what you typed, let alone answer it. A screenshot that prompted this had the agent
134 seconds into a single blocking `bash` step with the user's "how long will this
take?" sitting unread beneath it.

Only a *different thread* can answer while the main one is busy, so this is a different
agent. It is deliberately small (Claude Haiku by default), it is **read-only**, and it
never touches the main agent's conversation, its tools, or its files. It reads what is
already on disk -- the transcript the main agent is writing, and the live job facts the
tracker keeps -- and it answers you from that, in seconds, labelled `desk` so it is
never mistaken for the agent itself. Your message still goes to the main agent unchanged
and it will answer in its own time; the desk just means the screen is no longer silent
while it does.

Why this does not break the free-will contract: the contract governs what may influence
the *main model's* decisions, and the desk influences nothing. It does not write to the
main agent's thread, cannot call a tool, cannot stop or steer a job. It is a narrator
with a phone, facing the user. The one thing it must never do is pretend to be the agent
or promise on its behalf -- so it speaks in the third person about "the agent," and when
a real answer needs the agent it says so rather than inventing one.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

import anthropic

DESK_SYSTEM = """\
You are the front desk of a CFD assistant running a long OpenFOAM job on a remote \
machine. A separate, more capable agent is doing the actual work -- meshing, solving, \
writing and reading files. You cannot do any of that and must never claim to. Your \
only job is to keep the person informed while the agent is busy, because the agent \
runs one thing at a time and cannot answer them until its current step finishes.

You are given a recent slice of what the agent has said and done, and live facts about \
any running job. From that, answer the person's question directly and briefly -- two or \
three sentences, plain language, numbers when you have them. Speak about the agent in \
the third person ("the agent is...", "it launched..."). If you can answer from what you \
were given -- how far along a solve is, what the agent just did, what a file says -- do. \
If the question needs the agent to actually do something, or needs information you were \
not given, say plainly that you have passed it to the agent and it will see it at its \
next step; do not guess and do not promise on the agent's behalf. Never invent results, \
file contents, or timings. No preamble, no sign-off."""

NOW_SYSTEM = """\
You narrate, in one short present-tense clause, what a CFD agent is doing right now, \
for someone watching who wants to know at a glance. You are given the last few things \
the agent said and did and any live job facts. Reply with a single clause, lower case, \
no trailing period, under about 90 characters, plain language -- e.g. "reworking the \
near-wake mesh before spending solver time" or "solving the steady case, ~14 min in". \
Describe what is happening now, not what already finished. No preamble."""

MAX_REPLY_TOKENS = 400
MAX_NOW_TOKENS = 60
TRANSCRIPT_ROWS = 24
NOW_EVERY_S = 20.0
"""How often the desk refreshes its one-line 'what's happening now', while the agent
is busy. Each refresh is one cheap Haiku call; when nothing is happening there are
none."""

NOW_MIN_GAP_S = 8.0
"""Never two 'now' calls closer than this, however often events nudge it."""


class Concierge:
    """A background agent that answers the user while the main one is busy.

    One thread, one queue. Replies are urgent (the user is waiting) and jump ahead of
    the periodic 'now' refresh. Everything it needs it reads fresh each time from the
    store and the tracker, so it never holds a stale view and never shares mutable
    state with the main loop.
    """

    def __init__(self, cfg: Any, store: Any, view: Any, tracker: Any = None):
        self.store = store
        self.view = view
        self.tracker = tracker
        self.model = cfg.desk_model or "claude-haiku-4-5"
        self._client = anthropic.Anthropic(
            api_key=cfg.anthropic_api_key or None,
            base_url=cfg.llm_base_url or None,
            # Short calls, but a stalled one must not wedge the desk thread forever.
            timeout=min(60.0, cfg.llm_timeout_s or 60.0),
        )
        self._system = [
            {"type": "text", "text": DESK_SYSTEM, "cache_control": {"type": "ephemeral"}}
        ]
        self._q: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._last_now = 0.0
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="desk", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # -- what the session tells it ---------------------------------------------

    def working(self, yes: bool = True) -> None:
        """Whether the main agent is mid-turn. The 'now' line only refreshes while it
        is: an idle session answers for itself and needs no narrator."""
        if yes:
            self._busy.set()
            self.nudge()
        else:
            self._busy.clear()

    def ask(self, text: str) -> None:
        """The user said something while the agent is busy. Answer it, soon."""
        if text and text.strip():
            self._q.put(("reply", text.strip()))

    def nudge(self) -> None:
        """Something changed worth re-narrating (a tool ran, a step ended)."""
        self._q.put(("now", ""))

    # -- the thread ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=NOW_EVERY_S)
            except queue.Empty:
                item = ("now", "")  # the periodic refresh
            if item is None or self._stop.is_set():
                return
            kind, text = item
            try:
                if kind == "reply":
                    self._reply(text)
                elif self._busy.is_set():
                    self._now()
            except Exception:  # noqa: BLE001 - the desk may never end the session
                pass

    def _reply(self, text: str) -> None:
        answer = self._call(self._system, self._prompt(user=text), MAX_REPLY_TOKENS)
        if answer:
            self.view.desk(answer)

    def _now(self) -> None:
        now = time.monotonic()
        if now - self._last_now < NOW_MIN_GAP_S:
            return
        self._last_now = now
        line = self._call(
            [{"type": "text", "text": NOW_SYSTEM}], self._prompt(user=None), MAX_NOW_TOKENS
        )
        if line:
            self.view.narration(_one_line(line))

    # -- talking to the model --------------------------------------------------

    def _call(self, system: list, prompt: str, max_tokens: int) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError:
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def _prompt(self, user: str | None) -> str:
        parts = ["Recent activity (oldest first):", self._transcript()]
        facts = self._facts()
        if facts:
            parts.append("\nLive job facts:\n" + facts)
        if user is not None:
            parts.append(f'\nThe person watching just said: "{user}"\n\nAnswer them.')
        else:
            parts.append("\nWhat is the agent doing right now?")
        return "\n".join(parts)

    def _transcript(self) -> str:
        rows = self.store.recent_messages(TRANSCRIPT_ROWS)
        lines = []
        for row in rows:
            role = row.get("role", "")
            body = _render(row.get("content"))
            if not body:
                continue
            who = {
                "assistant": "agent",
                "user": "person",
                "event": "system",
                "tool": "tool",
            }.get(role, role)
            lines.append(f"[{who}] {body[:600]}")
        return "\n".join(lines[-TRANSCRIPT_ROWS:]) or "(nothing yet)"

    def _facts(self) -> str:
        if self.tracker is None:
            return ""
        try:
            snap = self.tracker.snapshot()
            facts = list(self.tracker.facts_for_wake())
        except Exception:  # noqa: BLE001
            return ""
        out = list(facts)
        if snap.busy and snap.headline:
            head = snap.headline
            if snap.detail:
                head += f" ({snap.detail})"
            out.insert(0, head)
        return "\n".join(out)


def _render(content: Any) -> str:
    """One line describing a transcript row's content, whatever shape it is."""
    if isinstance(content, str):
        return " ".join(content.split())
    if isinstance(content, dict):
        if "tool" in content:
            inp = content.get("input") or {}
            arg = inp.get("cmd") or inp.get("path") or ""
            arg = " ".join(str(arg).split())
            out = _render(content.get("output"))
            tail = f" -> {out[:160]}" if out else ""
            return f"{content['tool']}({arg[:160]}){tail}"
        return " ".join(json.dumps(content, default=str).split())[:400]
    if isinstance(content, list):
        return " ".join(_render(item) for item in content if _render(item))[:400]
    return ""


def _one_line(text: str) -> str:
    line = " ".join(text.split())
    return line[:110]
