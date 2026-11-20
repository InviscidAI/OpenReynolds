"""Telling something else that a study happened, from this machine and nowhere else.

`openreynolds webhook URL` reads the session stream on stdin, writes every line back
out unchanged, and POSTs the few events worth waking a human or a pipeline for. It is
a filter, so it composes:

    openreynolds -p "drag on a sphere at Re=1000" --output-format stream-json \\
      | openreynolds webhook https://hooks.example/… --format slack

Three decisions, and each one is why this module is forty lines rather than a service.

**It sends from here.** The workspace has no outbound network at all: `pip` and `apt`
inside the image are symlinked to a refusal (`docker/workspace/install-guard.sh`), so a
notification raised on the instance could not leave it. The session's own process is on
the user's machine and already has the events, so that is where the POST belongs.

**It is a separate process, not a hook in the session.** A webhook endpoint that hangs
must not hold a study open, and `jsonview.emit` is deliberately a method that "never
raises: a view that can break a session is a view that decides something"
(`jsonview.py:102-107`). A pipe keeps that property for free -- the worst a dead
endpoint can do here is stall this filter -- and it costs the session nothing, not even
an import. It also means no new knowledge of the hosted service enters the package:
this reads a documented NDJSON stream and makes an HTTP request, which is what
`tests/test_negative_obligation.py` allows anywhere but the seven modules above the
backend protocol.

**It does not deliver reliably, and says so.** There is no queue, no retry schedule and
no store of what was sent. One attempt per event, a failure reported on stderr, the
stream carried on. A webhook service -- durable queue, redelivery, dead letters -- is
not this repository's job; a receiver that must not miss an event reads the stream
itself, which is still there on stdout.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Iterator, TextIO

DEFAULT_EVENTS = ("session_start", "session_end", "approval", "workspace_failed")
"""What is worth a notification, out of the thirty-odd types `jsonview` emits.

The two boundaries (`jsonview.py:153,169` -- `session_start` carries the study id, and
`session_end` carries the outcome that `ONE_SHOT_EXIT_CODES` maps, `cli.py:2985`), plus
the two events that mean the run has stopped and is waiting for a person: an approval
put to the user (`jsonview.py:377`) and a workspace that never came up
(`jsonview.py:275`). Everything else is narration at tens of events a minute --
`text` fires per 240 characters and `jobs` on every poll -- and a notification channel
that carries narration is a notification channel nobody reads. `--event` adds any type
by name and `--all` sends the lot.
"""

SIGNATURE_HEADER = "X-OpenReynolds-Signature"
"""HMAC-SHA256 of the exact bytes posted, hex, prefixed `sha256=`.

A webhook URL is a bearer secret in a query string, and the receiver of one has no
other way to tell this sender from anyone who has seen the URL in a CI log. Computed
over the body as sent rather than over a re-serialisation of it, because two JSON
encoders do not agree on whitespace and a signature over a different byte string than
the one delivered fails for the receiver who checks it properly."""

SECRET_ENV = "OPENREYNOLDS_WEBHOOK_SECRET"

TIMEOUT_S = 10.0
"""One attempt, and it gives up quickly. This filter sits in front of the user's own
stdout: a minute spent waiting on a dead endpoint is a minute the session's stream is
not being written through."""

MAX_LINE_BYTES = 1_000_000
"""A line longer than this is passed through and not posted.

The stream is NDJSON and a `message` event carries a whole model turn, which has no
documented ceiling. A notification is a sentence; posting a megabyte of transcript to a
chat webhook is a failed request at best."""


def summarise(event: dict[str, Any]) -> str:
    """One sentence about an event, for a channel a person reads.

    Slack and Discord take a single text field and render nothing else usefully, so a
    chat delivery is this line. Written from the event's own fields and never from a
    lookup table of outcomes: an outcome this function has not heard of has to arrive
    as itself rather than as "unknown".
    """
    kind = str(event.get("type") or "event")
    study = str(event.get("study") or "")
    where = f" ({study})" if study else ""
    if kind == "session_start":
        model = str(event.get("model") or "")
        return f"Study {study or '(unnamed)'} started" + (f" on {model}" if model else "")
    if kind == "session_end":
        return f"Study{where} ended: {event.get('outcome') or 'ok'}" + (
            f" -- {event['error']}" if event.get("error") else "")
    if kind == "approval":
        return f"Study{where} is waiting for an answer: {event.get('title') or ''}".strip()
    if kind == "workspace_failed":
        return f"Study{where} got no workspace: {event.get('reason') or ''}".strip()
    if kind in ("tool_error", "warn", "notice", "info"):
        return f"Study{where} {kind}: {event.get('message') or ''}".strip()
    return f"Study{where} {kind}"


def body_for(event: dict[str, Any], shape: str) -> bytes:
    """The bytes to POST: the event itself, or the one field a chat service reads.

    `slack` and `discord` differ only in the key name (`text` against `content`), and
    both carry the whole event beside the sentence so a receiver that wants the numbers
    is not forced to parse the sentence back apart.
    """
    if shape == "slack":
        payload: dict[str, Any] = {"text": summarise(event), "event": event}
    elif shape == "discord":
        payload = {"content": summarise(event), "event": event}
    else:
        payload = dict(event)
    return json.dumps(payload).encode("utf-8")


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def post(url: str, body: bytes, *, secret: str = "", timeout: float = TIMEOUT_S) -> str:
    """One attempt. An empty string means it was accepted; otherwise the reason.

    Never raises. This runs in a loop over a stream the user is also reading, and an
    endpoint being down is not a reason to stop writing their stream out.
    """
    try:
        # Built inside the try: `Request("not-a-url")` raises `ValueError: unknown url
        # type` from the constructor, not from `urlopen`, and a typo in a workflow file
        # must be a reported failure rather than a traceback through the user's stream.
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        if secret:
            request.add_header(SIGNATURE_HEADER, sign(body, secret))
        with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310
            answer.read(1024)
        return ""
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return str(exc) or type(exc).__name__


def wanted(event: dict[str, Any], events: Iterable[str] | None) -> bool:
    """Whether this event is one to send. `None` means every event."""
    return events is None or str(event.get("type") or "") in set(events)


def relay(lines: Iterator[str], urls: list[str], *, out: TextIO | None = sys.stdout,
          events: Iterable[str] | None = DEFAULT_EVENTS, shape: str = "json",
          secret: str = "", log: TextIO | None = sys.stderr,
          send=post) -> dict[str, int]:
    """Pass every line through and POST the ones that match. Returns a tally.

    A line that is not JSON is passed through and not posted. The stream is the
    session's and may carry anything a reader is expected to tolerate -- `jsonview`
    writes one object per line under a lock precisely because "a half-written line is
    not a parse error a reader can recover from" (`jsonview.py:17-26`) -- so this
    reader, which is not the one the stream exists for, declines to guess.
    """
    tally = {"lines": 0, "sent": 0, "failed": 0, "skipped": 0}
    for line in lines:
        tally["lines"] += 1
        if out is not None:
            out.write(line if line.endswith("\n") else line + "\n")
            out.flush()
        text = line.strip()
        if not text or len(text) > MAX_LINE_BYTES:
            continue
        try:
            event = json.loads(text)
        except ValueError:
            continue
        if not isinstance(event, dict) or not wanted(event, events):
            tally["skipped"] += 1
            continue
        body = body_for(event, shape)
        for url in urls:
            reason = send(url, body, secret=secret)
            if reason:
                tally["failed"] += 1
                if log is not None:
                    print(f"webhook: {event.get('type')} to {_safe(url)} failed: {reason}",
                          file=log, flush=True)
            else:
                tally["sent"] += 1
    return tally


def _safe(url: str) -> str:
    """A URL with its path and query cut off, for a log line.

    A Slack or Discord webhook URL *is* the credential -- `.../services/T0/B0/<secret>`
    -- so the failure report names the host and nothing else. stderr here goes to a CI
    log that is often public."""
    try:
        parts = urllib.parse.urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}"
    except ValueError:
        return "(unparseable url)"
