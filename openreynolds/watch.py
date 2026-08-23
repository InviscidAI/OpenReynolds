"""Watch mode: pure-code polling, and wake messages that are facts.

While a job runs the model can end its turn. This module watches in plain Python and
wakes it with what happened - a name, an exit code, an end reason, a tail of log. It
never suggests what to do about any of it.
"""

from __future__ import annotations

import queue
import random
import sys
import threading
import time
from dataclasses import dataclass

from .backend.base import Backend, BackendError, JobStatus
from .store import Store
from .tools import describe_job
from .view import View

POLL_MIN_S = 15.0
POLL_MAX_S = 30.0
TICK_S = 0.5
"""How often the poll wait checks for typed input, so the user is never left waiting."""

WAKE_TAIL_BYTES = 2_000


class LineReader:
    """Reads stdin on a background thread so polling can stay responsive.

    One reader serves the whole session - the interactive prompt and watch mode both
    take from this queue, so there is only ever one consumer of stdin.
    """

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
        """Next line, or None on EOF. Raises `queue.Empty` on timeout."""
        return self._queue.get(timeout=timeout)

    def poll(self) -> str | None | _Nothing:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return NOTHING


class _Nothing:
    """Distinguishes 'nothing typed' from an EOF `None`."""

    def __bool__(self) -> bool:
        return False


NOTHING = _Nothing()


class NullReader:
    """Stands in for stdin when there is no user to interrupt (one-shot runs)."""

    def get(self, timeout: float | None = None) -> str | None:
        return None

    def poll(self) -> str | None | _Nothing:
        return NOTHING


@dataclass
class Wake:
    kind: str
    """job | user | eof | idle"""
    text: str = ""


def watch(
    backend: Backend,
    store: Store,
    view: View,
    reader: LineReader,
) -> Wake:
    """Poll live jobs until something happens worth waking for."""
    live = store.live_jobs()
    if not live:
        return Wake("idle")

    view.watching([job.name or job.job_id[:8] for job in live])

    while True:
        typed = reader.poll()
        if not isinstance(typed, _Nothing):
            if typed is None:
                return Wake("eof")
            if typed.strip():
                return Wake("user", typed)

        finished = _collect_finished(backend, store, view)
        if finished:
            return Wake("job", finished)

        if not store.live_jobs():
            return Wake("idle")

        deadline = time.monotonic() + random.uniform(POLL_MIN_S, POLL_MAX_S)
        while time.monotonic() < deadline:
            typed = reader.poll()
            if not isinstance(typed, _Nothing):
                if typed is None:
                    return Wake("eof")
                if typed.strip():
                    return Wake("user", typed)
            time.sleep(TICK_S)


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
        reports.append(_job_report(backend, status))

    return "\n\n".join(reports)


def _job_report(backend: Backend, status: JobStatus) -> str:
    """Facts about one finished job, with the tail of its log."""
    lines = [describe_job(status)]
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
