"""Sharing: the trace of each study OpenReynolds sends to the service.

What this does is set by the privacy policy's "Traces from OpenReynolds"
(https://tryreynolds.com/privacy), and each rule there is a rule here:

* **Signed in only.** Nothing is recorded, queued or sent without a service key
  (`openreynolds login`). A machine signed out (`openreynolds logout`) has its queue emptied. Each queued
  trace is named for the account and service it was recorded under (`fingerprint`),
  and goes only to that one: a machine signed in to another account, or pointed at
  another service, drops what it queued before.
* **On by default, with a switch per machine.** `openreynolds sharing off` (kept in
  `sharing.json` beside the config) or `OPENREYNOLDS_SHARING=off` stops this machine
  sharing anything and empties what it has queued. The switch is asked again when a
  trace is queued and when it is sent. There is no account-wide switch. A
  `sharing.json` that cannot be read counts as off, not as the default.
* **Told once.** `NOTICE`, one line, is shown when the machine signs in, or at the start
  of the first study on a machine that signed in before this version. `DISCLOSURE_VERSION`
  says which notice; the service refuses a client whose notice is older than the one it
  requires, and that client empties its queue.
* **Credentials removed** before a trace is written to the queue (`redact.Redactor`):
  this machine's keys by value, then the patterns the service applies again. A
  string is redacted whole and cut afterwards, so a key across the cut is not left
  half there.
* **Bounded.** A tool's input and output are cut at `CONTENT_AT_MOST` characters and
  marked; images and other binary results are already their descriptions
  (`tools.describe`); a trace too large to send goes as its usage part.
* **Never in the way.** Nothing here fails a study: a config folder that cannot be
  written, a second OpenReynolds on the machine, or an unreachable service costs the
  trace at most. The close-down spends at most `END_BUDGET_S` sending.

A trace is its own record, apart from the capture plane (`capture.py`), which shows and
resumes a study on the service and has its own switch (`OPENREYNOLDS_CAPTURE`).

Which sessions share is `cli.session`'s `sharing`: OpenReynolds' own interfaces (the
terminal, `-p`, stream-json, the local MCP server), and the hosted web app, which passes
`"web"`: its studies always leave a trace, marked `source: web`, with no notice, since
the web app is the service itself and the policy says so. A web trace never touches this
machine's files (switch, notice, queue): one runner process holds many people's
sessions, so a queue there could go out under the wrong person's key. It is sent from
memory when the study ends (`send_now`), within the close-down's budget, or not at
all. The machine's switch is not asked: it is the service's own runner, and the
policy says a web study always leaves a trace.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import os
import platform
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import __version__
from .config import Config, config_path
from .redact import Redactor, known_values

DISCLOSURE_VERSION = 1
"""The notice this version shows, and the one the service must accept for it to send.
Raised with `NOTICE` whenever what a trace contains changes in a material way."""

PRIVACY_URL = "https://tryreynolds.com/privacy"

NOTICE = (
    "OpenReynolds shares a trace of each study with Reynolds to improve it, credentials "
    f"removed; `openreynolds sharing off` stops it on this machine. {PRIVACY_URL}"
)

CONTENT_AT_MOST = 20_000
"""Characters of any one string in a content record (a tool output, a command, a
message), the same cut the capture plane makes; the full text stays on this machine."""

SEND_AT_MOST = 4_500_000
"""Bytes of a gzipped trace; the service takes 5 MB. Over it, the usage part goes."""

JSON_AT_MOST = 15_000_000
"""Bytes of a trace before it is gzipped; the service unzips 16 MB. Over it, the usage
part goes too."""

_PRE_CUT = 1_000_000
"""A string longer than this is cut to it before it is redacted (then cut again to
`CONTENT_AT_MOST`): a key would have to straddle this cut and then have a megabyte
of redaction pull it under the second one to leave a piece behind."""

QUEUE_AT_MOST = 100
"""Traces kept waiting to be sent; the oldest go first when there are more."""

END_BUDGET_S = 8.0
"""How long the close-down of a session may spend sending, after which what is left
waits for the next session's start."""

START_BUDGET_S = 60.0

_OFF = ("0", "false", "no", "off")
_ON = ("1", "true", "yes", "on")

STALE_AFTER_S = 7 * 24 * 3600
"""A queued trace recorded under another account or service is left for that one (two
sessions on one machine can hold different keys) until it is this old, then dropped."""

WEB_BUDGET_S = 20.0
"""How long the web app's close-down may spend sending its trace: there is no queue
to fall back on, so a busy service is asked again for longer than a terminal waits."""

_SENT_AT = "X-OpenReynolds-Sent-At"
"""This machine's clock as it sends, so the service can correct `queued_at` for a
clock that is wrong (a trace queued before the account deleted its traces is not
kept, and which side of the deletion it is on is a matter of clocks)."""

USAGE_KINDS = frozenset({"header", "turn", "tool"})


# -- the machine's switch and notice -------------------------------------------------


@dataclass
class State:
    enabled: bool = True
    notice: int = 0
    """The newest notice this machine has shown (`DISCLOSURE_VERSION`), 0 for none."""
    unreadable: bool = False
    """`sharing.json` is there but could not be read: sharing is off until
    `openreynolds sharing on|off` writes it again. Not saved."""


def state_path() -> Path:
    return config_path().parent / "sharing.json"


def queue_dir() -> Path:
    return config_path().parent / "sharing-queue"


def load_state() -> State:
    """The machine's switch and notice. No file is the default (on, not yet told);
    a file that is there but not a switch this version can read is off: whoever
    wrote it was most likely turning sharing off, and guessing on is the wrong guess."""
    try:
        raw = json.loads(state_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return State()
    except (OSError, ValueError):
        return State(enabled=False, unreadable=True)
    if not isinstance(raw, dict):
        return State(enabled=False, unreadable=True)
    enabled = raw.get("enabled")
    readable = isinstance(enabled, bool) or (
        isinstance(enabled, str) and enabled.strip().lower() in _ON + _OFF)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in _ON
    notice = raw.get("notice") or 0
    return State(enabled=readable and enabled is True,
                 notice=notice if isinstance(notice, int) and not isinstance(notice, bool) else 0,
                 unreadable=not readable)


def save_state(state: State) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {"enabled": state.enabled, "notice": state.notice}
    _write_private(path, (json.dumps(saved, indent=2) + "\n").encode("utf-8"))


def env_off() -> bool:
    """OPENREYNOLDS_SHARING set to anything but empty or an on word is off: a value
    this version does not know (`disabled`, `never`) was meant as off."""
    value = (os.environ.get("OPENREYNOLDS_SHARING") or "").strip().lower()
    return bool(value) and value not in _ON


def signed_in(cfg: Config) -> bool:
    return bool(getattr(cfg, "foamd_api_key", "") and getattr(cfg, "foamd_url", ""))


def fingerprint(cfg: Config) -> str:
    """Which account on which service a trace was recorded under: a hash of the service
    URL and key, so a queued trace never goes under another account's key."""
    seen = f"{(cfg.foamd_url or '').rstrip('/')}\n{cfg.foamd_api_key or ''}"
    return hashlib.sha256(seen.encode("utf-8")).hexdigest()[:16]


def active(cfg: Config, state: State | None = None) -> bool:
    """Whether this machine shares at all: signed in, and its switch on."""
    state = state or load_state()
    return signed_in(cfg) and not env_off() and state.enabled


def set_enabled(enabled: bool) -> State:
    if not enabled:
        clear_queue()      # first: a switch that cannot be saved still empties the queue
    state = load_state()
    state.enabled = enabled
    state.unreadable = False
    save_state(state)
    return state


def take_notice(cfg: Config) -> str | None:
    """`NOTICE` when this machine shares and has not shown this version's notice yet,
    marking it shown; None otherwise. Whoever takes it shows it: `openreynolds login`, or
    the start of the first study on a machine signed in before this version. A machine
    that cannot save that it was told is told again next time. Never raises."""
    try:
        state = load_state()
        if not active(cfg, state) or state.notice >= DISCLOSURE_VERSION:
            return None
        state.notice = DISCLOSURE_VERSION
        try:
            save_state(state)
        except OSError:
            pass
        return NOTICE
    except Exception:  # noqa: BLE001 - sharing never fails a sign-in or a study
        return None


SOURCES = ("openreynolds", "web")


def begin(cfg: Config, study_id: str, *, backend: str = "", solver: str = "",
          source: str = "openreynolds") -> tuple[Recorder | None, str | None]:
    """At a session's start: (the recorder for this study or None, the notice to show
    or None). A machine that is not signed in has its queue emptied, which is what
    signing out does. A `web` session (the hosted app) is never shown the notice and
    reads and writes none of this machine's sharing files, the switch included.
    Never raises: a study that cannot be recorded is run unrecorded."""
    try:
        if source == "web":
            if not signed_in(cfg):
                return None, None
            return Recorder(cfg, study_id, backend=backend, solver=solver, source="web"), None
        if not signed_in(cfg):
            clear_queue()
            return None, None
        if not active(cfg):
            return None, None
        notice = take_notice(cfg)
        return Recorder(cfg, study_id, backend=backend, solver=solver, source=source), notice
    except Exception:  # noqa: BLE001 - sharing never fails a study
        return None, None


# -- recording one study ---------------------------------------------------------------


def _now() -> str:
    """To the microsecond: a trace queued in the same second as a deletion, but after
    it, must still compare as after it."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _cut(value: Any, at_most: int = CONTENT_AT_MOST) -> tuple[Any, bool]:
    """`value` with every string cut at `at_most`, and whether anything was."""
    cut = False

    def walk(v: Any) -> Any:
        nonlocal cut
        if isinstance(v, str):
            if len(v) > at_most:
                cut = True
                return v[:at_most]
            return v
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if v is None or isinstance(v, (bool, int, float)):
            return v
        return str(v)

    return walk(value), cut


def _block(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


def _quiet(method: Callable[..., None]) -> Callable[..., None]:
    """A recorder that fails stops recording; it never fails the loop that called it.
    A trace with a hole in it would misrepresent the study, so after the first failure
    the study is not sent at all (`Recorder.broken`)."""

    @functools.wraps(method)
    def guarded(self: "Recorder", *args: Any, **kwargs: Any) -> None:
        if self.broken:
            return
        try:
            method(self, *args, **kwargs)
        except Exception:  # noqa: BLE001 - see above
            self.broken = True

    return guarded


class Recorder:
    """The trace of one study session, built as the loop runs (`Loop.recorder`).

    Called from the loop's thread for turns and tool calls, and from wherever the loop
    records a message; one lock keeps the list whole."""

    def __init__(self, cfg: Config, study_id: str, *, backend: str = "", solver: str = "",
                 source: str = "openreynolds", environ: dict[str, str] | None = None):
        self.study_id = study_id
        self.source = source if source in SOURCES else "openreynolds"
        self.account = fingerprint(cfg)
        self.redact = Redactor(known_values([cfg.llm_api_key, cfg.foamd_api_key], environ))
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._calls = 0
        self._turns = 0
        self.broken = False
        self.header: dict[str, Any] = {
            "kind": "header", "backend": backend, "solver": solver, "provider": cfg.provider,
            "model": cfg.model, "started_at": _now(), "python": platform.python_version(),
            "platform": sys.platform,
        }

    def _add(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)

    def _clean(self, value: Any) -> tuple[Any, bool]:
        """`value` redacted whole, then cut at `CONTENT_AT_MOST`; and whether it was cut."""
        long, pre_cut = _cut(value, _PRE_CUT)
        cut, was_cut = _cut(self.redact.deep(long))
        return cut, pre_cut or was_cut

    @_quiet
    def turn(self, response: Any, seconds: float) -> None:
        """A model turn: its usage, and its reasoning as the provider returned it."""
        tokens = getattr(response, "tokens", None) or {}
        self._turns += 1
        self._add({
            "kind": "turn", "at": _now(), "model": getattr(response, "model", "") or "",
            "provider": getattr(response, "provider", "") or "", "seconds": round(seconds, 3),
            "stop": getattr(response, "stop_reason", "") or "",
            "input_tokens": int(tokens.get("input", 0)), "output_tokens": int(tokens.get("output", 0)),
            "cache_read_tokens": int(tokens.get("cache_read", 0)),
            "cache_write_tokens": int(tokens.get("cache_write", 0)),
        })
        for block in getattr(response, "content", None) or []:
            if _block(block, "type") == "thinking":
                text = _block(block, "thinking") or ""
                if text.strip():
                    clean, _ = self._clean(text)
                    self._add({"kind": "reasoning", "at": _now(), "text": clean})

    @_quiet
    def tool(self, name: str, tool_input: Any, output: str, is_error: bool,
             seconds: float | None = None) -> None:
        self._calls += 1
        call = self._calls
        output = output or ""
        clean_input, input_cut = self._clean(tool_input)
        clean_output, output_cut = self._clean(output)
        self._add({
            "kind": "tool", "at": _now(), "call": call, "tool": name, "ok": not is_error,
            "error_type": "error" if is_error else None,
            "seconds": round(seconds, 3) if seconds is not None else None,
            "output_chars": len(output), "truncated": input_cut or output_cut,
        })
        self._add({
            "kind": "tool_io", "call": call, "tool": name,
            "input": clean_input, "output": clean_output,
            "truncated": input_cut or output_cut,
        })

    @_quiet
    def message(self, role: str, content: Any) -> None:
        """What the loop records for the transcript, other than tool calls (`tool`)."""
        if role == "tool" or content in (None, ""):
            return
        clean, _ = self._clean(content if isinstance(content, str) else json.dumps(content, default=str))
        self._add({"kind": "message", "at": _now(), "role": role, "text": clean})

    def body(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
        header = {**self.header, "ended_at": _now(), "turns": self._turns, "tool_calls": self._calls}
        return {
            "id": uuid.uuid4().hex, "disclosure": DISCLOSURE_VERSION, "openreynolds": __version__,
            "study_id": self.study_id, "source": self.source, "queued_at": _now(),
            "parts": "usage+content",
            "records": [header, *records],
        }

    def finish(self) -> Path | None:
        """Write this study's trace to the queue; its path, or None when there was
        nothing worth sending (no model turn) or the machine stopped sharing meanwhile."""
        if not self._turns or self.broken:
            return None
        body = self.body()
        return enqueue(self.account, body["id"], packed(body))


def usage_only(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "parts": "usage",
            "records": [r for r in body.get("records") or [] if r.get("kind") in USAGE_KINDS]}


def packed(body: dict[str, Any]) -> bytes:
    """The gzipped bytes that are queued and sent as they are: the whole trace, or its
    usage part when the whole is over either of the service's limits."""
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    data = gzip.compress(raw)
    if len(raw) > JSON_AT_MOST or len(data) > SEND_AT_MOST:
        data = gzip.compress(json.dumps(usage_only(body), ensure_ascii=False).encode("utf-8"))
    return data


# -- the queue ---------------------------------------------------------------------------


def _write_private(path: Path, data: bytes) -> None:
    """Written whole or not at all, readable by this user only. The temporary name is
    this write's own, so two OpenReynolds on one machine never write the same one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows refuses a replace while another process is replacing or
                # reading the same file; it is free again in moments.
                if os.name != "nt" or attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        tmp.unlink(missing_ok=True)


def _queue() -> Path:
    folder = queue_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        folder.chmod(stat.S_IRWXU)
    return folder


def _age(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:     # sent or dropped by another OpenReynolds just now
        return 0.0


def queued() -> list[Path]:
    """What is waiting, oldest first: `<fingerprint>-<id>.json.gz` each."""
    folder = queue_dir()
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.json.gz"), key=_age)


def _account_of(path: Path) -> str:
    head, _, _ = path.name.partition("-")
    return head


def enqueue(account: str, trace_id: str, data: bytes) -> Path | None:
    state = load_state()
    if env_off() or not state.enabled:
        return None
    path = _queue() / f"{account}-{trace_id}.json.gz"
    _write_private(path, data)
    waiting = queued()
    for old in waiting[: max(0, len(waiting) - QUEUE_AT_MOST)]:
        old.unlink(missing_ok=True)
    return path


def clear_queue() -> int:
    gone = 0
    for path in queued():
        path.unlink(missing_ok=True)
        gone += 1
    return gone


# -- sending -------------------------------------------------------------------------------


def _sent_at() -> dict[str, str]:
    return {"content-type": "application/json", "content-encoding": "gzip", _SENT_AT: _now()}


def _post_within(http: Any, data: bytes, seconds: float) -> Any:
    """One POST that returns within `seconds` whatever the network does: httpx's
    timeouts are per phase and restart with each byte, and DNS has none. A post still
    running then is abandoned to its daemon thread (the service drops a duplicate);
    None means no answer in time."""
    result: dict[str, Any] = {}

    def post() -> None:
        try:
            result["sent"] = http.post("/v1/traces", content=data, headers=_sent_at(),
                                       timeout=max(0.5, seconds))
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    worker = threading.Thread(target=post, name="sharing-post", daemon=True)
    worker.start()
    worker.join(max(0.0, seconds))
    return result.get("sent")


def _client(cfg: Config, timeout: float) -> Any:
    import httpx2 as httpx

    # No redirects: a 307 would carry the trace to another host, and a 301 would turn
    # it into a GET whose 200 reads as "sent".
    return httpx.Client(base_url=cfg.foamd_url.rstrip("/"), timeout=timeout, follow_redirects=False,
                        headers={"Authorization": f"Bearer {cfg.foamd_api_key}",
                                 "User-Agent": f"openreynolds/{__version__}"})


def send(cfg: Config, budget_s: float = END_BUDGET_S) -> dict[str, int]:
    """Send what is queued, within `budget_s`. Counts of what was sent, dropped and left.

    Never raises: sharing must not fail a study or its close-down."""
    counts = {"sent": 0, "dropped": 0, "left": 0}
    waiting = queued()
    if not waiting:
        return counts
    state = load_state()
    if not signed_in(cfg) or env_off() or not state.enabled:
        counts["dropped"] = clear_queue()
        return counts
    account = fingerprint(cfg)
    deadline = time.monotonic() + budget_s
    try:
        with _client(cfg, max(1.0, min(budget_s, 20.0))) as http:
            for path in waiting:
                left = min(budget_s, deadline - time.monotonic())
                if left <= 0.5:
                    break
                if _account_of(path) != account:
                    # Recorded under another account or service (or by a version that
                    # did not say which): never sent under this one's key. Left for the
                    # session that holds that key, and dropped once it is stale.
                    if time.time() - _age(path) > STALE_AFTER_S:
                        path.unlink(missing_ok=True)
                        counts["dropped"] += 1
                    continue
                try:
                    data = path.read_bytes()
                except OSError:      # another OpenReynolds sent it meanwhile
                    continue
                sent = _post_within(http, data, min(left, 20.0))
                if sent is None:
                    break                          # no answer in time: later
                code = (sent.json().get("error") if sent.headers.get("content-type", "").startswith(
                    "application/json") and sent.status_code >= 400 else "") or ""
                if sent.status_code in (200, 201):
                    path.unlink(missing_ok=True)
                    counts["sent"] += 1
                elif code == "disclosure_outdated" or sent.status_code == 401:
                    # A notice this version never showed, or a revoked key (signed out).
                    counts["dropped"] += clear_queue()
                    break
                elif sent.status_code in (400, 410, 413, 422):
                    path.unlink(missing_ok=True)   # this one can never be taken
                    counts["dropped"] += 1
                else:
                    break                          # the service is struggling: later
    except Exception:  # noqa: BLE001 - offline, or a service that answered strangely
        pass
    counts["left"] = len(queued())
    return counts


_RETRY = frozenset({429, 502, 503, 504})


def send_now(cfg: Config, recorder: Recorder, budget_s: float = WEB_BUDGET_S) -> bool:
    """A web study's trace, sent straight from memory and never queued: whether the
    service took it. A busy service is asked again while the budget lasts; there is no
    queue to fall back on. Never raises."""
    if not recorder._turns or recorder.broken or not signed_in(cfg):
        return False
    deadline = time.monotonic() + budget_s
    try:
        data = packed(recorder.body())
        with _client(cfg, max(1.0, budget_s)) as http:
            while True:
                left = min(budget_s, deadline - time.monotonic())
                sent = _post_within(http, data, left)
                if sent is not None and sent.status_code in (200, 201):
                    return True
                if (sent is None or sent.status_code not in _RETRY
                        or deadline - time.monotonic() < 1.5):
                    return False
                time.sleep(1.0)
    except Exception:  # noqa: BLE001 - sharing never fails a study
        return False


def send_in_background(cfg: Config, budget_s: float = START_BUDGET_S) -> threading.Thread | None:
    """What earlier sessions left queued, sent while this one starts. A daemon: a
    process that exits first leaves the rest queued, which is all that is lost."""
    if not queued():
        return None
    worker = threading.Thread(target=send, args=(cfg, budget_s), name="sharing", daemon=True)
    worker.start()
    return worker


def describe(cfg: Config) -> list[str]:
    """`openreynolds sharing status`, line by line."""
    state = load_state()
    lines = []
    if not signed_in(cfg):
        clear_queue()      # signed out: nothing is kept for sending later
        lines.append("This machine is not signed in, so it shares nothing (`openreynolds login`).")
    elif state.unreadable:
        lines.append(f"Sharing is off on this machine: {state_path()} could not be read "
                     "(`openreynolds sharing on` or `off` writes it again).")
    elif env_off():
        lines.append("Sharing is off on this machine: OPENREYNOLDS_SHARING is set to off.")
    elif not state.enabled:
        lines.append("Sharing is off on this machine (`openreynolds sharing on` turns it back on).")
    elif state.notice < DISCLOSURE_VERSION:
        lines.append("Sharing is on on this machine. The next study says so once when it starts.")
    else:
        lines.append("Sharing is on on this machine.")
    if signed_in(cfg):
        lines.append("Your shared traces can be deleted on the web app's Account page.")
    waiting = len(queued())
    if waiting:
        lines.append(f"{waiting} trace{'s' if waiting != 1 else ''} waiting to be sent.")
    lines.append(f"What is shared: {PRIVACY_URL}")
    return lines
