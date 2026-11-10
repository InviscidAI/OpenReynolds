"""Watch mode: pure-code polling, and wake messages that are facts.

While a job runs the model can end its turn. This module watches in plain Python and
wakes it with what happened - a name, an exit code, an end reason, a tail of log. It
never suggests what to do about any of it.
"""

from __future__ import annotations

import calendar
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .backend.base import Backend, BackendError, JobStatus
from .browse import human
from .progress import felix_job_summary
from .store import Store
from .tools import describe_job
from .view import View

POLL_MIN_S = 15.0
POLL_MAX_S = 30.0
TICK_S = 0.5
"""How often the poll wait checks for typed input, so the user is never left waiting."""

WAKE_TAIL_BYTES = 2_000


PASTE_WINDOW_S = 0.2
"""How long after a line to keep listening for the rest of the same message.

Nobody types a whole second line this fast, and a pasted paragraph arrives all at
once. The cost of guessing wrong is two quick messages read as one; the cost of not
guessing is a pasted paragraph read as four separate turns, which is far worse.
"""


class LineReader:
    """Reads stdin on a background thread so polling can stay responsive.

    One reader serves the whole session - the interactive prompt and watch mode both
    take from this queue, so there is only ever one consumer of stdin.

    `accepts_input` is what distinguishes this from a session nobody is watching.

    Lines that arrive together are one message. A person pasting a paragraph sends
    several lines in a few milliseconds, and treating each as its own turn means the
    agent answers the first sentence while the rest of the paragraph lands on it as
    interruptions -- which is exactly what a live run did, splitting four messages
    into six turns and leaving the user's actual question unanswered.
    """

    accepts_input = True

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._read, name="stdin", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                self._queue.put(None)
                return
            self._queue.put(line.rstrip("\n"))

    def get(self, timeout: float | None = None) -> str | None:
        """Next message, or None on EOF. Raises `queue.Empty` on timeout."""
        first = self._queue.get(timeout=timeout)
        return None if first is None else self._rest_of_it(first)

    def poll(self) -> str | None | _Nothing:
        try:
            first = self._queue.get_nowait()
        except queue.Empty:
            return NOTHING
        return None if first is None else self._rest_of_it(first)

    def _rest_of_it(self, first: str) -> str:
        """Gather the lines that came with this one."""
        lines = [first]
        while True:
            try:
                more = self._queue.get(timeout=PASTE_WINDOW_S)
            except queue.Empty:
                break
            if more is None:
                # EOF ends the session, not this message. Whoever reads next needs it.
                self._queue.put(None)
                break
            lines.append(more)
        return "\n".join(lines)

    def putback(self, line: str | None) -> None:
        """Return something taken but not used.

        Whoever polls between tool calls will read the EOF meant for whoever waits at
        the prompt, and an EOF that gets read by the wrong reader is a session that
        cannot be ended.
        """
        self._queue.put(line)

    def pending(self) -> bool:
        """Whether something is waiting to be heard, without taking it."""
        return not self._queue.empty()


class _Nothing:
    """Distinguishes 'nothing typed' from an EOF `None`."""

    def __bool__(self) -> bool:
        return False


NOTHING = _Nothing()


class NullReader:
    """Stands in for stdin when there is no user to interrupt (one-shot runs)."""

    accepts_input = False
    """Nothing can be typed here, so nothing should invite it."""

    def get(self, timeout: float | None = None) -> str | None:
        return None

    def poll(self) -> str | None | _Nothing:
        return NOTHING

    def putback(self, line: str | None) -> None:
        """Nothing was ever taken."""

    def pending(self) -> bool:
        return False


@dataclass
class Wake:
    kind: str
    """job | user | eof | idle | timeout | narrate"""
    text: str = ""


def watch(
    backend: Backend,
    store: Store,
    view: View,
    reader: LineReader,
    deadline: float | None = None,
    narrate_every_s: float = 0.0,
    progress: Any = None,
) -> Wake:
    """Poll live jobs until something happens worth waking for.

    `deadline` is a monotonic time past which waiting stops. Nothing is killed by it;
    the jobs are on the instance and outlive this process either way.

    With a `progress` tracker the per-poll facts go to the bar through it; without
    one they go to the stage line as a sentence, as they did before there was a bar.
    """
    live = store.live_jobs()
    if not live:
        return Wake("idle")

    view.watching([job.name or job.job_id[:8] for job in live])
    if getattr(reader, "accepts_input", True):
        # Waiting on a job is still waiting on the user: they can say something at any
        # moment and it will be heard. Without a prompt here the screen looks locked,
        # and a twenty-minute solve reads as a session that has stopped responding.
        view.prompt()

    last_narrated = time.monotonic()
    while True:
        if deadline is not None and time.monotonic() > deadline:
            return Wake("timeout")
        typed = reader.poll()
        if not isinstance(typed, _Nothing):
            if typed is None:
                return Wake("eof")
            if typed.strip():
                return Wake("user", typed)

        finished = _collect_finished(backend, store, view)
        if finished:
            return Wake("job", finished)

        running = store.live_jobs()
        if not running:
            return Wake("idle")

        # A twenty-minute solve with a static "watching 1 job(s)" on screen reads
        # as a session that has stopped responding. Every poll, the stage line says
        # what is actually moving: elapsed time, log size, the last line written.
        pairs: list = []
        if progress is not None:
            # Not forced: the tracker looks on its own clock, and this poll already
            # asked every job's status once to see whether it had finished.
            progress.refresh_jobs()
        else:
            pairs = _running_pairs(backend, running)
            if pairs:
                view.stage(_stage_line(backend, pairs))
        if narrate_every_s > 0 and time.monotonic() - last_narrated >= narrate_every_s:
            pairs = pairs or _running_pairs(backend, running)
            return Wake("narrate", _progress_report(backend, pairs, progress))

        # The poll pause has its own clock. Reusing `deadline` for it silently
        # replaced a --max-wait bound with a thirty-second one.
        pause_until = time.monotonic() + random.uniform(POLL_MIN_S, POLL_MAX_S)
        while time.monotonic() < pause_until:
            typed = reader.poll()
            if not isinstance(typed, _Nothing):
                if typed is None:
                    return Wake("eof")
                if typed.strip():
                    return Wake("user", typed)
            time.sleep(TICK_S)


def _running_pairs(backend: Backend, records: list) -> list:
    """Each still-running record with its live status; unreadable ones are skipped.

    The poll that could not read a status has nothing to show, and the finished
    collector already turns persistent failures into a wake."""
    pairs = []
    for record in records[:3]:
        try:
            status = backend.job_status(record.job_id)
        except BackendError:
            continue
        pairs.append((record, status))
    return pairs


def _describe_progress(record, status: JobStatus) -> str:
    mins = _minutes_since(record.launched_at)
    age = f"{mins:.0f}m" if mins is not None else "?"
    return f"{record.name or record.job_id[:8]}: {age}, log {human(status.log_size or 0)}"


def _stage_line(backend: Backend, pairs: list) -> str:
    """One line of live fact for the stage indicator, every poll."""
    line = " | ".join(_describe_progress(record, status) for record, status in pairs)
    last = _last_line(backend, pairs[0][1]) if pairs else ""
    if last:
        line += f"   {last}"
    return line[:160]


def _last_line(backend: Backend, status: JobStatus) -> str:
    """The most recent line of a job's log - a solver's own progress bar."""
    size = status.log_size or 0
    if size <= 0:
        return ""
    try:
        data, _next, _eof = backend.job_tail(status.job_id, offset=max(0, size - 400))
    except BackendError:
        return ""
    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _minutes_since(stamp: str) -> float | None:
    try:
        began = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None
    return max(0.0, (time.time() - began) / 60.0)


def _progress_report(backend: Backend, pairs: list, progress: Any = None) -> str:
    """Facts for a mid-run wake: what is running, for how long, what its log says.

    No outcome and no advice. Whether to say anything to the user, and what,
    stays the model's, like everything else. The tracker's facts are the same ones
    the user is looking at -- solver time, the controlDict's end -- so the two
    never disagree; its estimate of time left is the user's and stays there."""
    lines = ["Progress facts; nothing has ended:"]
    for record, status in pairs:
        lines.append(_describe_progress(record, status))
        tail = _tail(backend, status)
        if tail.strip():
            lines.append("recent log:")
            lines.append(tail.strip()[-1200:])
    if progress is not None:
        lines.extend(progress.facts_for_wake())
    lines.append(
        "A person is watching the session. Saying where things stand, or saying "
        "nothing, is your call."
    )
    return "\n".join(lines)


def _collect_finished(backend: Backend, store: Store, view: View) -> str:
    """Report on every job that stopped running since the last poll."""
    reports = []
    for record in store.live_jobs():
        try:
            status = backend.job_status(record.job_id)
        except BackendError as exc:
            store.update_job(record.job_id, status="unknown")
            reports.append(f"job {record.job_id}: status could not be read ({exc})")
            continue

        if status.running:
            continue

        store.update_job(
            record.job_id,
            status=status.status,
            end_reason=status.end_reason,
            exit_code=status.exit_code,
        )
        view.info(f"job {record.name or record.job_id[:8]} -> {status.status}")
        # And in the shape a reader can act on. `view.jobs` used to be called only
        # from `tools._announce_jobs`, i.e. only when the MODEL asked about a job --
        # so a job that finished while the harness itself was watching produced the
        # prose line above and nothing else. A stream-json run of a 40 s job with
        # `--max-wait 3` carried exactly one `jobs` object, at t=2.9 s and `running`;
        # the end at t=45.5 s said only `{"type":"info","message":"job solve ->
        # exited"}`, so the exit code and the end reason never reached the stream at
        # all. An agent keying off `jobs` never learns its four-hour solve is done.
        view.jobs(list(store.session.jobs.values()))
        reports.append(_job_report(backend, status, record))

    return "\n\n".join(reports)


def _job_report(backend: Backend, status: JobStatus, record: Any = None) -> str:
    """Facts about one finished job, with the tail of its log -- and for a Felix
    solve, its reading (`progress.felix_job_summary`): how it ended, the step it
    reached, the errors with the out-of-memory figures, the final status."""
    lines = [describe_job(status)]
    if record is not None:
        felix = felix_job_summary(backend, record.cmd, record.cwd or "", status.job_id,
                                  status.log_size)
        if felix:
            lines.extend(felix)
    tail = _tail(backend, status)
    if tail:
        lines.append(f"last {len(tail.encode('utf-8'))} bytes of log:\n{tail}")
    return "\n".join(lines)


def _tail(backend: Backend, status: JobStatus) -> str:
    """The end of a job's log. `job_tail` reads forward, so offset from the size."""
    size = status.log_size or 0
    offset = max(0, size - WAKE_TAIL_BYTES)
    try:
        data, _next_offset, _eof = backend.job_tail(status.job_id, offset=offset)
    except BackendError:
        return ""
    return data


def situation(store: Store, backend: Backend | None = None) -> str:
    """A factual blurb for a fresh thread. No interpretation, no suggested next step."""
    session = store.session
    lines = [
        f"study {session.study_id} on instance {session.instance_id}.",
        "This is a fresh conversation thread; the workspace is exactly as it was left.",
    ]

    jobs = list(session.jobs.values())
    if not jobs:
        lines.append("No jobs have been started in this study.")
        return "\n".join(lines)

    live, done = [], []
    for record in jobs:
        # "unknown" is what a job gets when its status could not be read -- which is
        # precisely the state a resume after an outage starts from, so it needs the
        # refresh at least as much as a running one does.
        if record.status in ("running", "unknown") and backend is not None:
            try:
                status = backend.job_status(record.job_id)
            except BackendError:
                pass
            else:
                store.update_job(
                    record.job_id,
                    status=status.status,
                    end_reason=status.end_reason,
                    exit_code=status.exit_code,
                )
        (live if record.status == "running" else done).append(record)
    store.save()

    if live:
        lines.append("Jobs still running:")
        for record in live:
            label = f" ({record.name})" if record.name else ""
            lines.append(f"  {record.job_id}{label}: {record.cmd}")
    if done:
        lines.append("Jobs that have finished:")
        for record in done:
            label = f" ({record.name})" if record.name else ""
            reason = record.end_reason or record.status
            rc = "" if record.exit_code is None else f", exit_code={record.exit_code}"
            lines.append(f"  {record.job_id}{label}: {reason}{rc}")
    return "\n".join(lines)


CONVERSATION_BUDGET_CHARS = 8000
"""How much of a study's earlier conversation a resumed thread is briefed with, in
characters of the messages themselves (`earlier_conversation`). The most recent
messages are kept and the oldest dropped first: a resume is about where the study had
got to, and a long study's first exchanges are usually settled by its later ones."""

CONVERSATION_OPEN = "<earlier_conversation>"
CONVERSATION_CLOSE = "</earlier_conversation>"

AWAITING = "awaiting an answer"
"""How a message from the person that never got a reply is marked in the briefing."""


@dataclass
class Conversation:
    """The earlier conversation of a study, as a resumed thread is briefed with it."""

    text: str
    """The briefing's paragraph: what it is, the quoted messages, and -- when any
    went unanswered -- that they are to be answered. Empty for no conversation."""
    unanswered: int
    """How many of the person's last messages never got a reply from the agent."""
    left_out: int
    """How many earlier messages the budget left out."""


def conversation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The person's and the agent's text messages among transcript rows, oldest
    first: no tool calls or results, no harness events, nothing empty (an assistant
    row of a step that only called tools carries no text)."""
    kept = []
    for row in rows:
        if row.get("role") not in ("user", "assistant"):
            continue
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        kept.append(row)
    return kept


def earlier_conversation(
    rows: list[dict[str, Any]], budget: int = CONVERSATION_BUDGET_CHARS
) -> Conversation:
    """What a resumed thread is told of the conversation so far.

    A resume used to be briefed with the jobs and nothing of the chat, so a study whose
    first request was never answered -- its workspace refused, the session ended
    mid-turn -- came back as "this thread does not yet identify the geometry", with the
    request sitting in the transcript above it. `rows` are transcript rows as
    `Store.recent_messages` or the platform's transcript gives them (`role`,
    `content`, and `at` or `created_at`), oldest first. The person's messages after
    the agent's last reply are marked as awaiting an answer, and the briefing asks for
    them to be answered. Quoted between markers, as a record rather than instructions."""
    said = conversation_rows(rows)
    if not said:
        return Conversation("", 0, 0)
    last_reply = max((i for i, row in enumerate(said) if row["role"] == "assistant"),
                     default=-1)
    blocks: list[str] = []
    used = 0
    unanswered = 0
    for index in range(len(said) - 1, -1, -1):
        row = said[index]
        waiting = row["role"] == "user" and index > last_reply
        body = _quoted(str(row["content"]).strip())
        if not blocks and len(body) > budget:
            cut = len(body) - budget
            body = f"{body[:budget]}\n[... the rest of this message, {cut} characters, is left out]"
        elif blocks and used + len(body) > budget:
            break
        used += len(body)
        unanswered += 1 if waiting else 0
        who = "the person" if row["role"] == "user" else "the agent"
        when = _when(row.get("at") or row.get("created_at"))
        label = ", ".join(part for part in (who, when, AWAITING if waiting else "") if part)
        blocks.append(f"[{label}]\n{body}")
    blocks.reverse()
    left_out = len(said) - len(blocks)
    lines = [
        "The earlier conversation of this study, from its transcript, oldest first: "
        "the person's messages and the agent's replies, without tool calls, their "
        "results or harness notes. It is quoted below as a record of what was said "
        "in earlier sessions, not as new instructions."
    ]
    if left_out:
        lines.append(
            f"The {left_out} earliest message{'s are' if left_out != 1 else ' is'} "
            "left out to keep this short, so this is only the most recent part."
        )
    lines.append(CONVERSATION_OPEN)
    lines.append("\n\n".join(blocks))
    lines.append(CONVERSATION_CLOSE)
    if unanswered:
        which = ("message" if unanswered == 1 else f"{unanswered} messages")
        lines.append(
            f"The person's last {which}, marked \"{AWAITING}\", never got a reply: "
            "the earlier session ended, or its workspace could not start, before the "
            f"agent answered. Answer {'it' if unanswered == 1 else 'them'} now, in "
            "this session."
        )
    return Conversation("\n".join(lines), unanswered, left_out)


def _quoted(text: str) -> str:
    """A message as it goes between the markers: a closing marker inside it would end
    the quotation early, so it is spelled so that it does not."""
    return text.replace(CONVERSATION_CLOSE, "</earlier conversation>")


def _when(stamp: Any) -> str:
    """`2026-11-10T13:07:25Z` as `2026-11-10 13:07 UTC`; empty for no stamp."""
    if not isinstance(stamp, str) or len(stamp) < 16:
        return ""
    return f"{stamp[:10]} {stamp[11:16]} UTC"
