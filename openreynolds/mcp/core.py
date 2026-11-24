"""The nine capabilities, once, for every face that offers them.

A `Host` answers them from wherever studies run (`local.LocalHost` here; the hosted
service has its own), and the capabilities themselves -- `begin_study`, `look_once`,
`status_answer`, `said_text`, `approval_line`, `studies_list`, `study_results` -- are
plain blocking calls over one. Three faces put them in front of a caller:
`build_server(host)` as MCP tools, `openreynolds.serve` as HTTP, `openreynolds.sdk` as
a library. What they say -- their arguments, the states a study is in, how the
transcript is condensed for a calling model, what to do next -- is all here, so no two
faces can drift.

A host's methods are plain blocking calls, run on a worker thread for each tool call
(`asyncio.to_thread`, which carries the caller's context: a host that keeps who is
calling in a context variable still has it). Anything a host raises that a caller
should read is either `McpError` or something `Host.explain` turns into words.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

from ..study_style import NOTES_MAX

log = logging.getLogger("openreynolds.mcp")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
"""What one staged upload may be."""
UPLOAD_URL_TTL_S = 3600
DOWNLOAD_URL_TTL_S = 3600
MAX_WAIT_S = 50
"""The longest `study_status` holds a call open: under a load balancer's idle timeout
(60 s) with room to answer."""
POLL_S = 2.0
DELIVERY_WAIT_S = 20 * 60
"""How long a delivery is looked at every few seconds; after it, every `DELIVERY_SLOW_S`."""
DELIVERY_SLOW_S = 15.0
DELIVERY_GIVE_UP_S = 2 * 3600
"""How long staged files wait for a workspace before the task is said without them.

It was `DELIVERY_WAIT_S`, and a workspace that took longer (a busy region, a queue for
capacity) got the task with a note to upload again, and the files were thrown away
(#216). The files can only land on a workspace, so they wait for one: as long as the
session lives, up to this. A workspace that has failed is not waited for."""
DELIVERY_LOCK_S = 20 * 60
"""How long one placement may hold a delivery without news, refreshed after each file.
One archive can take minutes: the upload itself, then its unpacked tree, each with
the hosted backend's 300 s timeout and a retry. A lock that lapsed mid-placement
would let another face's pass place the same file again."""
PLACE_TRIES = 3
PLACE_GIVE_UP_S = 10 * 60
"""A file is given up on, and named as missing, once it has failed `PLACE_TRIES`
passes over at least this long: long enough that a workspace restarting or a network
blip is waited out, short enough that a file that cannot be placed is not retried
for hours."""
DELIVERY_TTL_S = 6 * 3600
SAID_TTL_S = 10 * 60
"""How long a delivered task is remembered after it is said, so the status reads
`working` until the session has taken it into the transcript."""
TEXT_BUDGET = 12_000
"""Roughly how many characters of transcript one status answer carries. The caller is
a model with a context window; the full rows are one more call away."""

CALLER_NOTE = (
    "This study was started by another AI agent through the Reynolds MCP server, on "
    "behalf of its user; there is no person watching this page. Do not wait for "
    "answers to questions unless the mode asks you to -- decide, and say what you "
    "assumed. When the work is done, end with one final message the calling agent can "
    "act on: what was simulated and how (mesh, solver, boundary conditions), the key "
    "numbers with units, whether the solution converged and how you judged that, the "
    "caveats, and the workspace paths of the important outputs. Also write the key "
    "numbers to results.json in the study directory."
)

STATES = ("starting", "working", "waiting_for_approval", "waiting_for_input", "ending", "ended", "error")

ACTIVE = ("starting", "running", "ending")
"""A session row's statuses while the session lives."""
WORKSPACE_STARTING = "starting"
WORKSPACE_READY = "ready"
WORKSPACE_FAILED = "failed"
"""`picture["session"]["workspace"]`: whether the study's workspace can run tools."""

UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")


class McpError(Exception):
    """A refusal the calling agent should read: raised in a tool or a host, answered
    as the tool's error text."""


@dataclass
class Staged:
    """One uploaded file, waiting for its study's workspace."""

    key: str
    name: str
    size: int


class Host(Protocol):
    """Where studies run, as the tools need it.

    A session is named by `session_id`, a study by `study_id`. `observe` answers with
    the session's row -- `{"status": one of ACTIVE or "ended"/"error", "study_id",
    "instance_id", "error"}` -- and its live picture, or None when nothing is live:

        {"session": {"status", "study_id", "instance_id", "home", "workspace", "style",
                     "error", ...},
         "turn": "user" | "agent",
         "items": [transcript items, `openreynolds.transcript`, plus approval cards],
         "jobs": [{"name", "status", "cmd", ...}], "stage": str, "narration": str,
         "side": [{"question", "answer", "forwarded"}]}
    """

    instructions: str
    descriptions: dict[str, str]
    """Tool descriptions that differ from the defaults (DESCRIPTIONS), by tool name."""
    kv: "Kv"
    """Where a task waiting for its workspace is kept (`Deliveries`)."""

    def stage_upload(self, filename: str) -> dict[str, Any]: ...
    def staged(self, upload_id: str) -> Staged | None: ...
    def place_files(self, session_id: str, instance_id: str, home: str,
                    files: list[dict[str, Any]]) -> list[str] | None: ...
    def discard_staged(self, files: list[dict[str, Any]]) -> None: ...
    def start_session(self, body: dict[str, Any]) -> dict[str, Any]: ...
    def check(self, session_id: str) -> None: ...
    def heartbeat(self, session_id: str) -> None: ...
    def observe(self, session_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]: ...
    def say(self, session_id: str, text: str) -> None: ...
    def end(self, session_id: str) -> str: ...
    def transcript(self, study_id: str) -> dict[str, Any]: ...
    def transcript_items(self, study_id: str, after_seq: int) -> list[dict[str, Any]]: ...
    def studies(self) -> dict[str, Any]: ...
    def read_file(self, study_id: str, path: str) -> bytes | None: ...
    def files(self, study_id: str) -> list[dict[str, Any]]: ...
    def renders(self, study_id: str) -> list[dict[str, Any]]: ...
    def render_link(self, study_id: str, name: str) -> str: ...
    def download(self, study_id: str, path: str) -> dict[str, Any]: ...
    def say_text(self, text: Any) -> str: ...
    def approval_card(self, title: str, detail: str, style: str) -> dict[str, Any] | None: ...
    def explain(self, exc: BaseException) -> str | None: ...


class HostDefaults:
    """What a host need not change: text shown as written, no special approval cards,
    nothing extra to explain. Mixed into a host."""

    descriptions: dict[str, str] = {}

    def say_text(self, text: Any) -> str:
        return str(text or "")

    def approval_card(self, title: str, detail: str, style: str) -> dict[str, Any] | None:  # noqa: ARG002
        return None

    def explain(self, exc: BaseException) -> str | None:  # noqa: ARG002
        return None


# -- a small store for the deliveries -----------------------------------------------


class Kv(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_s: float | None = None, only_new: bool = False) -> bool: ...
    def delete(self, key: str) -> None: ...


class MemoryKv:
    """A process's own dict. One process serves every call, so it is everybody's."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: str, ttl_s: float | None = None, only_new: bool = False) -> bool:  # noqa: ARG002
        with self._lock:
            if only_new and key in self._data:
                return False
            self._data[key] = value
            return True

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


# -- files in: staged uploads placed once the workspace is up ----------------------------


class Deliveries:
    """Words waiting for a workspace, with the files they name.

    `start_study` with files starts the session with no first message and records the
    task here; `send_message` with files records its words the same way (`add`). A
    thread waits for the workspace, places the files in the study's `uploads/` and says
    the words with their paths. `study_status` nudges it too, so a server whose waiting
    thread is gone does not strand a task.

    Two locks. The place lock is held by the one pass placing files, which may take
    minutes for an archive; it is refreshed after each file. The edit lock is held only
    to read-and-write the record, so a message sent meanwhile never waits on an unpack.
    Files are placed one at a time and each is crossed off as it lands, so a pass that
    fails part way does not place the first ones again on the next."""

    def __init__(self, host: Host):
        self.host = host
        self._watching: set[str] = set()
        self._watching_lock = threading.Lock()

    def _key(self, session_id: str) -> str:
        return f"mcp:deliver:{session_id}"

    def record(self, session_id: str, pending: dict[str, Any]) -> None:
        self.host.kv.set(self._key(session_id), json.dumps(pending), ttl_s=DELIVERY_TTL_S)

    def pending(self, session_id: str) -> dict[str, Any] | None:
        raw = self.host.kv.get(self._key(session_id))
        return json.loads(raw) if raw else None

    def _clear(self, session_id: str) -> None:
        self.host.kv.delete(self._key(session_id))

    def _lock(self, session_id: str) -> str | None:
        """The place lock, as a token only its holder has, or None when it is held."""
        token = uuid.uuid4().hex
        taken = self.host.kv.set(self._key(session_id) + ":lock", token, ttl_s=DELIVERY_LOCK_S, only_new=True)
        return token if taken else None

    def _refresh(self, session_id: str, token: str) -> bool:
        """Extend the place lock if it is still this holder's. False when it lapsed and
        someone else took it: this pass must stop placing."""
        key = self._key(session_id) + ":lock"
        if self.host.kv.get(key) != token:
            return False
        self.host.kv.set(key, token, ttl_s=DELIVERY_LOCK_S)
        return True

    def _unlock(self, session_id: str, token: str) -> None:
        key = self._key(session_id) + ":lock"
        if self.host.kv.get(key) == token:      # never another holder's
            self.host.kv.delete(key)

    @contextlib.contextmanager
    def _editing(self, session_id: str, wait_s: float = 10.0) -> Iterator[None]:
        """Hold the record for one read-and-write. Never held across a placement."""
        key = self._key(session_id) + ":edit"
        deadline = time.monotonic() + wait_s
        while not self.host.kv.set(key, "1", ttl_s=30, only_new=True):
            if time.monotonic() > deadline:
                raise McpError("this study's files are being recorded; send again in a moment")
            time.sleep(0.05)
        try:
            yield
        finally:
            self.host.kv.delete(key)

    def add(self, session_id: str, text: str, files: list[dict[str, Any]]) -> None:
        """Queue words, with or without files, on a running session (`send_message`).
        Words still waiting are added to rather than replaced, so nothing is lost and
        everything is said in the order it was sent."""
        with self._editing(session_id):
            pending = self.pending(session_id)
            if pending is None or "said" in pending:
                pending = {"task": text, "files": [], "placed": [], "at": time.time()}
            else:
                pending = {**pending, "task": f"{pending['task']}\n\n{text}"}
            known = {f["key"] for f in pending.get("files") or []} | set(pending.get("keys") or [])
            fresh = []
            for f in files:                     # an upload named twice is placed once
                if f["key"] not in known:
                    known.add(f["key"])
                    fresh.append(f)
            pending["files"] = list(pending.get("files") or []) + fresh
            pending["keys"] = sorted(known)
            self.record(session_id, pending)
        self.watch(session_id)

    def watch(self, session_id: str) -> None:
        """Wait for the workspace on a thread of this process, and deliver. One thread
        per session: a second call while one is watching leaves it to that one."""
        with self._watching_lock:
            if session_id in self._watching:
                return
            self._watching.add(session_id)

        def run() -> None:
            began = time.monotonic()
            try:
                while time.monotonic() < began + DELIVERY_GIVE_UP_S + 60:
                    try:
                        if self.attempt(session_id):
                            with self._watching_lock:
                                # Something `add`ed after the pass, while this thread
                                # still counted as watching, is this thread's to place.
                                again = self.pending(session_id)
                                if again is None or "said" in again:
                                    self._watching.discard(session_id)
                                    return
                            continue
                    except Exception:  # noqa: BLE001 - try again on the next tick
                        log.exception("mcp delivery for %s failed a pass", session_id[:8])
                    time.sleep(3.0 if time.monotonic() - began < DELIVERY_WAIT_S else DELIVERY_SLOW_S)
            finally:
                with self._watching_lock:
                    self._watching.discard(session_id)
        context = contextvars.copy_context()
        threading.Thread(target=context.run, args=(run,), name=f"mcp-deliver-{session_id[:8]}",
                         daemon=True).start()

    def attempt(self, session_id: str) -> bool:
        """Place the files and say the words, if the workspace is up. True when there is
        nothing left to do for this session."""
        pending = self.pending(session_id)
        if pending is None or "said" in pending:
            return True
        row, picture = self.host.observe(session_id)
        if str(row.get("status") or "") not in ACTIVE:
            self._drop(session_id)
            return True
        session = (picture or {}).get("session") or {}
        instance_id = str(session.get("instance_id") or row.get("instance_id") or "")
        study_id = str(session.get("study_id") or row.get("study_id") or "")
        ready = session.get("workspace") == WORKSPACE_READY
        failed = session.get("workspace") == WORKSPACE_FAILED
        waited = time.time() - float(pending.get("at") or time.time())
        if not (ready and instance_id and study_id):
            if waited < DELIVERY_GIVE_UP_S and not failed:
                return False
            # The workspace failed, or never came up: say the words anyway, and say why
            # the files are not there, rather than leave the session silent.
            token = self._lock(session_id)
            if token is None:
                return False
            try:
                return self._say(session_id, picture, gave_up=True)
            finally:
                self._unlock(session_id, token)
        token = self._lock(session_id)
        if token is None:
            return False
        try:
            home = str(session.get("home") or "").rstrip("/") or f"/work/{study_id}"
            while True:
                current = self.pending(session_id)
                if current is None or "said" in current:
                    return True
                files = list(current.get("files") or [])
                if not files:
                    if self._say(session_id, picture):
                        return True
                    continue                      # more files came in as it was said
                item = files[0]
                try:
                    lines = self.host.place_files(session_id, instance_id, home, [item])
                except Exception:
                    log.exception("mcp delivery for %s could not place %s", session_id[:8], item.get("name"))
                    self._failed_once(session_id, item)
                    return False
                if lines is None:
                    return False
                self._placed(session_id, item, lines)
                if not self._refresh(session_id, token):
                    return False
        finally:
            self._unlock(session_id, token)

    def _placed(self, session_id: str, item: dict[str, Any], lines: list[str]) -> None:
        # A long wait: a file placed and not crossed off would be placed again.
        with self._editing(session_id, wait_s=120):
            current = self.pending(session_id)
            if current is None or "said" in current:
                return                          # dropped meanwhile: the session ended
            files = list(current.get("files") or [])
            current["files"] = [f for f in files if f["key"] != item["key"]]
            current["placed"] = list(current.get("placed") or []) + list(lines)
            current["fails"] = 0
            current.pop("failing_since", None)
            self.record(session_id, current)
        self.host.discard_staged([item])

    def _failed_once(self, session_id: str, item: dict[str, Any]) -> None:
        """Count a failed placement; after `PLACE_TRIES` the file is given up on and
        named as missing when the words are said, rather than retried for hours."""
        gave_up = False
        with self._editing(session_id):
            current = self.pending(session_id)
            if current is None or "said" in current:
                return
            current["fails"] = int(current.get("fails") or 0) + 1
            since = float(current.setdefault("failing_since", time.time()))
            if current["fails"] >= PLACE_TRIES and time.time() - since >= PLACE_GIVE_UP_S:
                current["files"] = [f for f in current.get("files") or [] if f["key"] != item["key"]]
                current["failed"] = list(current.get("failed") or []) + [str(item.get("name") or "")]
                current["fails"] = 0
                current.pop("failing_since", None)
                gave_up = True
            self.record(session_id, current)
        if gave_up:
            self.host.discard_staged([item])

    def _say(self, session_id: str, picture: dict[str, Any] | None, gave_up: bool = False) -> bool:
        """Say the words with the files placed so far. False when files are still to be
        placed (one came in as this was called), unless the workspace was given up on."""
        with self._editing(session_id):
            current = self.pending(session_id)
            if current is None or "said" in current:
                return True
            left = list(current.get("files") or [])
            if left and not gave_up:
                return False
            text = current["task"]
            placed = list(current.get("placed") or [])
            if placed:
                text += "\n\nInput files, uploaded by the caller:\n" + "\n".join(f"- {p}" for p in placed)
            if left:
                text += ("\n\n(The caller also uploaded files, but the workspace did not start "
                         "to receive them; ask the caller to upload them again with upload_file.)")
            failed = [n for n in current.get("failed") or [] if n]
            if failed:
                text += (f"\n\n(These uploaded files could not be placed on the workspace: "
                         f"{', '.join(failed)}; ask the caller to upload them again with upload_file.)")
            self.host.say(session_id, text)
            # The words are on the session's queue but not yet in its transcript: the
            # session reads them a moment later. Until it does, the turn still says
            # "user", and a status read in that gap would tell the caller the agent had
            # finished a turn it had not begun. So where the transcript stood is kept.
            said = {"said": text, "at": time.time(), "seq": last_seq(picture),
                    "opening": opening(current)}
            self.host.kv.set(self._key(session_id), json.dumps(said), ttl_s=SAID_TTL_S)
        if left:
            self.host.discard_staged(left)
        return True

    def _drop(self, session_id: str) -> None:
        with self._editing(session_id):
            current = self.pending(session_id)
            if current is None or "said" in current:
                return
            self._clear(session_id)
        self.host.discard_staged(list(current.get("files") or []))

    def unseen(self, session_id: str, picture: dict[str, Any] | None) -> bool:
        """Whether words this class said have had no answer in the transcript yet.

        Any answering line past where the transcript stood counts, not the words
        themselves: the session records a message after parsing it."""
        said = self.pending(session_id)
        if not said or "said" not in said:
            return False
        after = int(said.get("seq") or 0)
        # An answering line, not just any row: an event the runner writes before it
        # records the words would otherwise end the wait while the turn still says
        # "user", and the caller would be told `waiting_for_input`. Words delivered
        # here are never a slash command (`send_words` sends those straight through),
        # so the person's line is always recorded.
        heard = any(isinstance(i, dict) and isinstance(i.get("seq"), int) and i["seq"] > after
                    and str(i.get("kind") or "") in ANSWERING_KINDS
                    for i in (picture or {}).get("items") or [])
        if time.time() - float(said.get("at") or 0) <= SAID_TTL_S and not heard:
            return True
        # Answered: forget it, but only if it is still that record. `add` may have put
        # new words in its place since it was read.
        try:
            with self._editing(session_id, wait_s=0):
                again = self.pending(session_id)
                if again is not None and "said" in again:
                    self._clear(session_id)
        except McpError:
            pass                                  # someone is editing it; leave it be
        return False


def opening(pending: dict[str, Any] | None) -> bool:
    """Whether a delivery carries the study's opening task rather than words sent to a
    study under way. Records written before the mark existed: `add` always sets `keys`
    and `begin_study` never did."""
    pending = pending or {}
    return bool(pending.get("opening", "keys" not in pending and "said" not in pending))


def last_seq(picture: dict[str, Any] | None) -> int:
    return max((i["seq"] for i in (picture or {}).get("items") or []
                if isinstance(i, dict) and isinstance(i.get("seq"), int)), default=0)


# -- what a status answer says ----------------------------------------------------------


def state_of(row: dict[str, Any], picture: dict[str, Any] | None, delivering: bool,
             opening: bool = False) -> str:
    status = str(row.get("status") or "")
    if status in ("ended", "error"):
        return status
    if status == "ending":
        return "ending"
    if not picture:
        return "starting"
    session = picture.get("session") or {}
    if session.get("status") in ("ended", "error"):
        return str(session["status"])
    if delivering and session.get("workspace") == WORKSPACE_READY:
        # Files on their way to a running study. An open approval shows, because no
        # message moves past one. Words sent to a study under way: working. The opening
        # task -- a resumed study's too, whose transcript already holds the last run --
        # falls through to `starting` until it is in the transcript, or a caller reads
        # a quiet prompt as the study being done with it (prod, 2026-10-30).
        if open_approvals(picture):
            return "waiting_for_approval"
        if not opening:
            return "working"
    if delivering or (session.get("workspace") != WORKSPACE_READY and picture.get("turn") != "user"):
        return "starting"
    if open_approvals(picture):
        return "waiting_for_approval"
    running = [j for j in picture.get("jobs") or [] if str(j.get("status") or "") in ("running", "starting", "queued")]
    if picture.get("turn") == "user" and not running and answered(picture):
        return "waiting_for_input"
    return "working"


ANSWERING_KINDS = frozenset({"user", "assistant", "turn_failed"})
"""The transcript items that say who spoke last: the person, the agent, or the failed
request that ended the agent's turn."""


def answered(picture: dict[str, Any] | None) -> bool:
    """Whether the agent's finished turn is in the transcript this same picture holds.

    `waiting_for_input` is read off `picture["turn"]`, which the session thread sets in
    `LocalView.prompt`, and the transcript and its cursor are built from
    `picture["items"]`, which `LocalView.refresh` fills from the transcript tail. Two
    fields, two threads, two moments, and the state is only true if both agree.

    Both directions of that disagreement are the same bug, and only one was covered.
    Where `items` is *ahead* of `turn` -- the idle prompt sets the turn and a line
    already waiting is read a moment later -- a caller told `waiting_for_input` is told
    the agent finished a turn it has not begun; a person's message nobody has answered
    is work in hand, whatever the turn flag says. Where `items` is *behind* `turn`, the
    answer the state says to read is not in the transcript the same call returns, and
    `transcript_since` leaves the cursor at `after_seq` when its slice is empty -- so
    the next poll is handed the whole transcript again as if it were new. Measured 1 in
    40 on `test_a_second_message_and_the_cursor` (#188).

    So an empty picture is not `waiting_for_input`: a transcript with nothing in it
    holds no message to read and no seq to move the cursor to. A picture that has items
    and none of them of either side's -- events only -- is left as it was, because the
    cursor does move over those, so the next poll starts after them rather than at the
    beginning again."""
    items = [i for i in ((picture or {}).get("items") or []) if isinstance(i, dict)]
    if not items:
        return False
    last = next((i for i in reversed(items) if i.get("kind") in ANSWERING_KINDS), None)
    return last is None or last.get("kind") != "user"


CEILING_ANSWERS = ("approve continues the study (its limit is doubled); decline stops it here. "
                   "Ask the person: it is their usage.")
"""How answer_approval's decisions read on a ceiling approval (`ceiling.py`: "/yes"
doubles the study's ceiling, "/no" stops)."""

CEILING_NEXT = ("The study has paused at its usage limit and needs the person's say. Put the "
                "question to them as it reads, then answer with answer_approval: approve to "
                "continue, decline to stop.")


def open_approvals(picture: dict[str, Any] | None, host: Host | None = None) -> list[dict[str, Any]]:
    """The approvals waiting for an answer. A ceiling (a study paused at its usage
    limit, asked in every mode: it is the person's usage) carries `answers`, what each
    decision does, and whatever card the host words it with."""
    style = str(((picture or {}).get("session") or {}).get("style") or "")
    say = host.say_text if host is not None else (lambda t: str(t or ""))
    out = []
    for i in (picture or {}).get("items") or []:
        if i.get("kind") == "approval" and i.get("state") == "open":
            approval = {"request_id": i.get("request_id"), "about": i.get("subject"), "title": i.get("title"),
                        "detail": i.get("detail"), "choices": i.get("choices")}
            if i.get("subject") == "ceiling":
                card = host.approval_card(str(i.get("title") or ""), str(i.get("detail") or ""), style) \
                    if host is not None else None
                if card:
                    approval.update(title=card["title"], detail=card["detail"], choices=card["choices"],
                                    question=card["question"])
                approval["answers"] = CEILING_ANSWERS
            else:
                approval.update(title=say(i.get("title")), detail=say(i.get("detail")))
            out.append(approval)
    return out


SIDE_SHOWN = 3
"""How many of the latest side questions study_status carries. A caller polling it
wants the answer to what it just asked, not the whole side chat again on every poll."""


def side_chat(picture: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The latest side questions (`/btw <question>`) and their answers: the side chat is
    in no transcript, so without this a caller that asked one through send_message
    could never read the answer. `answer` is null while it is coming; `forwarded` when
    the side chat passed the words on to the agent itself."""
    entries = [e for e in (picture or {}).get("side") or [] if isinstance(e, dict)]
    return [
        {"question": clip(e.get("question"), 500),
         "answer": None if e.get("answer") is None else clip(e.get("answer"), 4000),
         **({"forwarded": True} if e.get("forwarded") else {})}
        for e in entries[-SIDE_SHOWN:]
    ]


def clip(text: Any, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _shown(text: Any) -> str:
    return str(text or "")


def condense(item: dict[str, Any], say: Callable[[Any], str] = _shown) -> dict[str, Any] | None:
    """One transcript item the way a calling model wants it: who, what, briefly."""
    kind = item.get("kind")
    base: dict[str, Any] = {"seq": item.get("seq")}
    if kind == "user":
        return {**base, "from": "caller", "text": clip(item.get("text"), 2000)}
    if kind == "assistant":
        return {**base, "from": "reynolds", "text": clip(item.get("text"), 4000)}
    if kind == "tool":
        inp = item.get("input") or {}
        brief = inp.get("cmd") or inp.get("command") or inp.get("path") or ""
        out = str(item.get("output") or "").strip()
        return {**base, "tool": item.get("tool"), "input": clip(brief, 200),
                "output": clip(out, 300), **({"error": True} if item.get("error") else {})}
    if kind == "desk_step":
        return {**base, "desk": item.get("step_kind"), "text": clip(item.get("text"), 300),
                **({"blocking": True} if item.get("blocking") else {})}
    if kind == "turn_failed":
        # The reason in full (clipped): it is what a caller needs to decide whether
        # to say something again or to stop.
        return {**base, "event": clip(say(item.get("text")), 600), "error": True}
    if kind == "event":
        said = say(item.get("label") or item.get("text"))
        return {**base, "event": clip(said, 200)} if said else None
    return None


def transcript_since(items: list[dict[str, Any]], after_seq: int,
                     say: Callable[[Any], str] = _shown) -> tuple[list[dict[str, Any]], int, bool]:
    """(condensed rows after `after_seq`, the cursor to pass next, whether it was cut)."""
    rows = [i for i in items if isinstance(i.get("seq"), int) and i["seq"] > after_seq]
    rows.sort(key=lambda i: i["seq"])
    out: list[dict[str, Any]] = []
    used = 0
    cursor = after_seq
    for item in rows:
        small = condense(item, say)
        cursor = item["seq"]
        if small is None:
            continue
        size = len(json.dumps(small))
        if out and used + size > TEXT_BUDGET:
            return out, out[-1]["seq"], True
        out.append(small)
        used += size
    return out, cursor, False


def last_agent_message(items: list[dict[str, Any]]) -> str:
    for item in reversed(items):
        if item.get("kind") == "assistant" and str(item.get("text") or "").strip():
            return str(item["text"])
    return ""


NEXT = {
    "starting": "The workspace is starting (usually under a minute). Call study_status again.",
    "working": "Reynolds is working. Call study_status again; long solves can take many minutes.",
    "waiting_for_approval": "Reynolds is asking before it spends compute. Answer with answer_approval.",
    "waiting_for_input": "Reynolds has finished its turn. Read its last message: answer a question it asked "
                         "with send_message, or call get_results, then end_study when you are done.",
    "ending": "The session is closing down; its files are kept. Call study_status once more, then get_results.",
    "ended": "The session is over and its files are kept. get_results and download_url read them; "
             "start_study with this study_id continues the study.",
    "error": "The session stopped on an error; its files are kept. get_results shows what it got to; "
             "start_study with this study_id resumes it.",
}


# -- the tools -----------------------------------------------------------------------------

DESCRIPTIONS: dict[str, str] = {
    "upload_file": (
        "Get a URL to upload one input file (a geometry such as STL/STEP/OBJ, a drawing, a "
        "data file) for a study. Run the returned curl command from the directory that holds "
        "the file, then pass the upload_id to start_study. Up to 100 MB per file; the URL is "
        "valid for an hour."
    ),
    "start_study": (
        "Start a CFD study (or continue one) and return at once with its session_id.\n\n"
        "task: what to simulate and what you want to know, in plain words -- geometry, units, "
        "flow conditions, quantities of interest. Be specific.\n"
        "upload_ids: files staged with upload_file; they are placed in the study's uploads/ "
        "directory and named to Reynolds before it starts.\n"
        "study_id: continue an existing study (from list_studies or an earlier start).\n"
        "depth: \"quick\" (one coarse mesh, headline numbers, then asks before spending more), "
        "\"full\" (a careful study: mesh resolution, validation, as an engineer would), "
        "\"learn\" (quick, with the reasoning behind each choice said as it goes) or "
        "\"beginner\" (quick, in plain language with the terms spelled out and the limits of "
        "the answer stated). None of them holds a step: they are the user's standing note.\n"
        "mode: \"auto\" (default, no questions), \"partial\" (asks before each solve) or "
        "\"structured\" (asks to approve a plan first).\n"
        "solver: \"auto\" (default: Reynolds picks), \"openfoam\" (CPU) or \"felix\" (our GPU "
        "solver). Fixed for the study's life; ignored when continuing a study.\n"
        "Then call study_status with the session_id."
    ),
    "study_status": (
        "What the study is doing now, and what has been said since after_seq.\n\n"
        "Holds the call open for up to wait_seconds (max 50) and returns early when Reynolds "
        "needs you (a question, an approval) or the session ends. Pass the returned "
        "next_after_seq back as after_seq to read only what is new. Each call also tells "
        "Reynolds someone is still there, so keep polling while it works; a study left "
        "unpolled for 30 minutes with nothing running ends itself."
    ),
    "send_message": (
        "Say something to Reynolds in a running study: answer its question, add a requirement, "
        "or change course. It reads it at its next step. upload_ids: files staged with "
        "upload_file, placed in the study's uploads/ and named in the message (an archive is "
        "unpacked beside itself). Slash commands work too (/status, "
        "/model <id>, /mode auto|partial|structured), and \"/btw <question>\" asks a side "
        "question: answered without interrupting the study or entering its conversation, "
        "under `side` in study_status."
    ),
    "answer_approval": (
        "Answer the open approval in a partial or structured study.\n"
        "decision: \"approve\", \"decline\" (note says why or what to do instead) or "
        "\"approve_all\" (approve this and stop asking: the study goes to auto).\n"
        "A study paused at its usage limit (about \"ceiling\", asked in every mode): ask the "
        "person, then approve to continue (the limit is doubled) or decline to stop."
    ),
    "end_study": (
        "End a study's session: running jobs are stopped, the workspace is shut down and every "
        "file is kept. The study can be continued later with start_study."
    ),
    "list_studies": "This account's studies, newest first, with the live sessions.",
    "get_results": (
        "What a study came to: Reynolds' final message, results.json if it wrote one, the "
        "report and pictures as download links, and the study's file list. Works on a running "
        "study too (it answers with what exists so far)."
    ),
    "download_url": (
        "A link to download one file of a study (a path from get_results or the transcript, "
        "e.g. /work/<study>/case/postProcessing/forces/0/forces.dat). Valid for an hour; run "
        "the curl command to save it here."
    ),
    "get_results:next": "download_url(study_id, path) gives a link to fetch any of these files with curl.",
}
"""Each tool's description, and `get_results:next`, the line its answer ends with."""


def description(host: Host | None, tool: str) -> str:
    """What a tool, or a face's endpoint, says it does: the host's own words or the default."""
    return (host.descriptions.get(tool) if host is not None else None) or DESCRIPTIONS[tool]


# -- the capabilities, once ----------------------------------------------------------------
#
# Each of these is one of the nine things a caller can do, as a plain blocking call over a
# `Host`. They were the bodies of the MCP tools; they are out here because the MCP server is
# not the only face over a Host any more -- `openreynolds serve` answers the same nine over
# HTTP and `openreynolds.sdk` answers them to a library caller, and three faces deriving a
# state name or condensing a transcript each their own way is exactly the drift this module
# exists to prevent. `build_server` below is now a thin async wrapper over these.


def staged_files(host: Host, upload_ids: list[str] | None) -> list[dict[str, Any]]:
    """The staged uploads named by `upload_ids`, as a delivery carries them; refused
    with words when one is not an id, has no file yet, or is too big."""
    if upload_ids is not None and not isinstance(upload_ids, list):
        raise McpError("upload_ids is a list of ids from upload_file")
    files: list[dict[str, Any]] = []
    for upload_id in dict.fromkeys(upload_ids or []):   # an id named twice is one file
        if not UPLOAD_ID.match(str(upload_id or "")):
            raise McpError(f"{upload_id!r} is not an upload id")
        found = host.staged(upload_id)
        if found is None:
            raise McpError(f"upload {upload_id} has no file yet -- run its curl command first")
        if found.size > MAX_UPLOAD_BYTES:
            raise McpError(f"upload {upload_id} is {found.size} bytes; the limit is {MAX_UPLOAD_BYTES}")
        files.append({"key": found.key, "name": found.name})
    return files


def begin_study(host: Host, deliveries: "Deliveries", task: str, *,
                upload_ids: list[str] | None = None, study_id: str | None = None,
                depth: str = "quick", mode: str = "auto", model: str | None = None,
                effort: str | None = None, notes: str | None = None,
                solver: str = "auto") -> dict[str, Any]:
    """Start a study (or continue one) and answer with its session_id."""
    if not task or not task.strip():
        raise McpError("say what to simulate in task")
    files = staged_files(host, upload_ids)
    note = CALLER_NOTE + (f"\n\n{notes.strip()}" if notes and notes.strip() else "")
    body: dict[str, Any] = {
        "style": depth, "mode": mode, "notes": note[:NOTES_MAX],
        "message": None if files else task.strip(),
    }
    if study_id:
        body["study_id"] = study_id
    if model:
        body["model"] = model
    if effort:
        body["effort"] = effort
    if solver and solver != "auto":
        body["solver"] = solver
    elif files:
        # The task waits for the files, and the files for a workspace an open
        # solver only starts once chosen: choose it from the task now.
        body["route_on"] = task.strip()
    session = host.start_session(body)
    session_id = str(session.get("session_id") or "")
    if files:
        deliveries.record(session_id, {"task": task.strip(), "files": files, "at": time.time(),
                                       "opening": True})
        deliveries.watch(session_id)
    answer = {
        "session_id": session_id, "study_id": session.get("study_id") or study_id or None,
        "state": "starting", "model": session.get("model"),
        "next": "Call study_status with this session_id. The study_id appears once the "
                "study is named, within the first minute.",
    }
    if session.get("sharing_notice"):
        # A host that shares traces says what it shares the first time, and this
        # is the only place a person driving through an agent will see it.
        answer["sharing_notice"] = session["sharing_notice"]
        answer["next"] = ("Show the person sharing_notice, one line, word for word: it "
                          "says what this machine shares and how to turn it off. "
                          "Then " + answer["next"][0].lower() + answer["next"][1:])
    return answer


def look_once(host: Host, deliveries: "Deliveries",
              session_id: str) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """One look at a session: its row, its picture and the state it is in.

    Each face runs its own wait loop around this -- the MCP server on the event loop,
    the HTTP face on the thread serving the request -- and stops on the same condition:
    a state that is no longer `starting` or `working`."""
    row, picture = host.observe(session_id)
    delivering = False
    pending = deliveries.pending(session_id)
    first = opening(pending)
    if pending is not None:
        if "said" not in pending:
            # Placed by a watching thread, started here if this process has none (a
            # server restarted since the task was recorded). Never on this call's own
            # thread: unpacking an archive takes longer than a status call may.
            deliveries.watch(session_id)
            delivering = True
        else:
            delivering = deliveries.unseen(session_id, picture)
    return row, picture, state_of(row, picture, delivering, first)


def wait_for_change(host: Host, deliveries: "Deliveries", session_id: str,
                    wait_s: float = 0.0) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Look until the study wants something, or until `wait_s` is up: a long poll.

    The blocking twin of the loop `build_server`'s `study_status` runs on the event
    loop. The HTTP face and the SDK each hold a thread of their own open instead, so
    they sleep rather than await, and both stop on the same condition."""
    deadline = time.monotonic() + max(0.0, min(float(wait_s or 0.0), MAX_WAIT_S))
    while True:
        row, picture, state = look_once(host, deliveries, session_id)
        left = deadline - time.monotonic()
        if state not in ("starting", "working") or left <= 0:
            return row, picture, state
        time.sleep(min(POLL_S, left))


def status_answer(host: Host, session_id: str, row: dict[str, Any], picture: dict[str, Any] | None,
                  state: str, after_seq: int = 0) -> dict[str, Any]:
    """What the study is doing, and what has been said since `after_seq`."""
    say = host.say_text
    items = list((picture or {}).get("items") or [])
    session = (picture or {}).get("session") or {}
    study = str(session.get("study_id") or row.get("study_id") or "")
    if not items and study:
        items = list(host.transcript_items(study, int(after_seq or 0)) or [])
    rows, cursor, cut = transcript_since(items, int(after_seq or 0), say)
    jobs = [
        {"name": j.get("name") or j.get("job_id"), "status": j.get("status"),
         "cmd": clip(j.get("cmd"), 160)}
        for j in (picture or {}).get("jobs") or []
    ]
    answer: dict[str, Any] = {
        "state": state, "session_id": session_id, "study_id": study or None,
        "stage": (picture or {}).get("stage") or "", "narration": (picture or {}).get("narration") or "",
        "jobs": jobs[-12:], "transcript": rows, "next_after_seq": cursor,
        "next": NEXT.get(state, ""),
    }
    approvals = open_approvals(picture, host)
    if state == "waiting_for_approval" and any(a["about"] == "ceiling" for a in approvals):
        answer["next"] = CEILING_NEXT
    if cut:
        answer["more"] = "There is more transcript; call study_status again with next_after_seq."
    if approvals:
        answer["approvals"] = approvals
    side = side_chat(picture)
    if side:
        answer["side"] = side
    if state in ("waiting_for_input", "ended", "error"):
        answer["last_message"] = clip(last_agent_message(items), 8000)
    if row.get("error") or session.get("error"):
        answer["error"] = row.get("error") or session.get("error")
    return answer


def said_text(text: str) -> str:
    """The line `send_message` puts on a session, refused when there is nothing in it."""
    if not text or not text.strip():
        raise McpError("nothing to say")
    return text.strip()


def send_words(host: Host, deliveries: "Deliveries", session_id: str, text: str,
               upload_ids: list[str] | None = None) -> None:
    """`send_message`: say `text` on the session, or, with files, place them first and
    say it with their paths, the way `begin_study` says a task.

    Words with no files go straight to the session. With files they go through
    `deliveries`, which waits for the workspace if it is not up yet, so a file sent
    mid-study reaches the workspace rather than staying staged (#216)."""
    files = staged_files(host, upload_ids)
    if not files:
        words = said_text(text)
        waiting = deliveries.pending(session_id)
        if words.startswith("/") or waiting is None or "said" in waiting:
            host.say(session_id, words)
            return
    else:
        words = (text or "").strip() or "Here are more input files."
        if words.startswith("/"):
            raise McpError("a slash command cannot carry files; send the files with a message")
    host.check(session_id)
    row, _ = host.observe(session_id)
    if str(row.get("status") or "") not in ACTIVE:
        raise McpError("this session is not running; start_study with its study_id to continue it")
    deliveries.add(session_id, words, files)   # after any files still on their way


APPROVAL_WORDS = {"approve": "/yes", "decline": "/no", "approve_all": "/all"}


def approval_line(decision: str, note: str = "") -> str:
    """The line that answers an open approval, from the decision a caller names."""
    if decision not in APPROVAL_WORDS:
        raise McpError('decision is "approve", "decline" or "approve_all"')
    return APPROVAL_WORDS[decision] + (f" {note.strip()}" if decision == "decline" and note.strip() else "")


def studies_list(host: Host, limit: int = 20) -> dict[str, Any]:
    """This account's studies, newest first, with the live sessions."""
    got = host.studies()
    cards = [
        {"study_id": c.get("study_id"), "title": c.get("title"), "created_at": c.get("created_at")}
        for c in (got.get("studies") or [])[: max(1, min(int(limit or 20), 100))]
    ]
    live = [{"session_id": s.get("session_id"), "study_id": s.get("study_id") or None,
             "status": s.get("status")} for s in got.get("sessions") or []]
    return {"studies": cards, "live_sessions": live}


def study_results(host: Host, study_id: str) -> dict[str, Any]:
    """What a study came to: its final message, results.json, reports, pictures, files."""
    got = host.transcript(study_id)
    items = list(got.get("items") or [])
    session = got.get("session") or {}
    home = str(session.get("home") or f"/work/{study_id}").rstrip("/")
    answer: dict[str, Any] = {
        "study_id": study_id, "title": session.get("title"), "workspace": home,
        "final_message": last_agent_message(items),
        "jobs": [{"name": j.get("name") or j.get("job_id"), "status": j.get("status"),
                  "end_reason": j.get("end_reason"), "exit_code": j.get("exit_code")}
                 for j in session.get("jobs") or []],
    }
    raw = host.read_file(study_id, f"{home}/results.json")
    if raw is not None:
        try:
            answer["results_json"] = json.loads(raw)
        except ValueError:
            answer["results_json_text"] = clip(raw.decode("utf-8", "replace"), 8000)
    entries = [e for e in host.files(study_id) if not e.get("is_dir")]
    reports = [e["path"] for e in entries if str(e.get("path", "")).endswith("results.md")]
    for path in reports[:3]:
        raw = host.read_file(study_id, path)
        if raw is not None:
            answer.setdefault("reports", []).append(
                {"path": path, "text": clip(raw.decode("utf-8", "replace"), 6000)})
    answer["pictures"] = [
        {"name": r.get("name"), "url": host.render_link(study_id, str(r.get("name")))}
        for r in (host.renders(study_id) or [])[-20:] if r.get("name")
    ]
    answer["files"] = [{"path": e.get("path"), "size": e.get("size")} for e in entries[:300]]
    if len(entries) > 300:
        answer["files_note"] = f"{len(entries)} files; the first 300 are listed"
    answer["next"] = description(host, "get_results:next")
    return answer


def deliveries_for(host: Host) -> "Deliveries":
    """The host's own `Deliveries`, or one over it. What one has recorded is in the
    host's `kv`, so a task waiting for its workspace is nudged by whichever face asks
    for the status next."""
    return getattr(host, "deliveries", None) or Deliveries(host)


# -- the tools -------------------------------------------------------------------------------


def build_server(host: Host, *, name: str = "reynolds", title: str = "Reynolds CFD") -> Any:
    """The MCP server: every tool, answered by `host`."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    from .. import __version__

    server = MCPServer(name=name, title=title, instructions=host.instructions, version=__version__)
    deliveries = deliveries_for(host)

    def describe(tool: str) -> str:
        return description(host, tool)

    async def call(fn: Callable[[], Any]) -> Any:
        try:
            return await asyncio.to_thread(fn)
        except McpError as exc:
            raise ToolError(str(exc)) from exc
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - a host's own refusal, in its words
            words = host.explain(exc)
            if words is None:
                raise
            raise ToolError(words) from exc

    # -- files in ----------------------------------------------------------------

    @server.tool(description=describe("upload_file"))
    async def upload_file(filename: str) -> dict[str, Any]:
        return await call(lambda: host.stage_upload(filename))

    # -- the session -------------------------------------------------------------

    @server.tool(description=describe("start_study"))
    async def start_study(task: str, upload_ids: list[str] | None = None, study_id: str | None = None,
                          depth: str = "quick", mode: str = "auto", model: str | None = None,
                          effort: str | None = None, notes: str | None = None,
                          solver: str = "auto") -> dict[str, Any]:
        return await call(lambda: begin_study(
            host, deliveries, task, upload_ids=upload_ids, study_id=study_id, depth=depth,
            mode=mode, model=model, effort=effort, notes=notes, solver=solver))

    @server.tool(description=describe("study_status"))
    async def study_status(session_id: str, after_seq: int = 0, wait_seconds: int = 30) -> dict[str, Any]:
        wait = max(0, min(int(wait_seconds or 0), MAX_WAIT_S))
        deadline = time.monotonic() + wait
        # Whose session it is, before anything reaches it.
        await call(lambda: host.check(session_id))
        await call(lambda: host.heartbeat(session_id))

        while True:
            row, picture, state = await call(lambda: look_once(host, deliveries, session_id))
            if state not in ("starting", "working") or time.monotonic() >= deadline:
                break
            await asyncio.sleep(POLL_S)

        return await call(lambda: status_answer(host, session_id, row, picture, state,
                                                int(after_seq or 0)))

    @server.tool(description=describe("send_message"))
    async def send_message(session_id: str, text: str, upload_ids: list[str] | None = None) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            send_words(host, deliveries, session_id, text, upload_ids)
            return {"sent": True, "next": "Call study_status to see the answer."}
        return await call(run)

    @server.tool(description=describe("answer_approval"))
    async def answer_approval(session_id: str, decision: str, note: str = "") -> dict[str, Any]:
        def run() -> dict[str, Any]:
            line = approval_line(decision, note)
            host.say(session_id, line)
            return {"sent": line, "next": "Call study_status."}
        return await call(run)

    @server.tool(description=describe("end_study"))
    async def end_study(session_id: str) -> dict[str, Any]:
        return await call(lambda: {"status": host.end(session_id)})

    # -- studies and their files ---------------------------------------------------

    @server.tool(description=describe("list_studies"))
    async def list_studies(limit: int = 20) -> dict[str, Any]:
        return await call(lambda: studies_list(host, limit))

    @server.tool(description=describe("get_results"))
    async def get_results(study_id: str) -> dict[str, Any]:
        return await call(lambda: study_results(host, study_id))

    @server.tool(description=describe("download_url"))
    async def download_url(study_id: str, path: str) -> dict[str, Any]:
        return await call(lambda: host.download(study_id, path))

    return server
