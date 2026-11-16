"""Sharing: the trace of each study a signed-in OpenReynolds sends to the service.

What this does is set by the privacy policy's "Traces from OpenReynolds"
(https://tryreynolds.com/privacy), and each rule there is a rule here:

* **Signed in only.** Nothing is recorded, queued or sent without a service key
  (`openreynolds login`). A machine signed out has its queue emptied.
* **On by default, with a switch per machine.** `openreynolds sharing off` (kept in
  `sharing.json` beside the config) or `OPENREYNOLDS_SHARING=off` stops this machine
  sharing anything and empties what it has queued. The account's two switches, "Share
  usage" and "Share study content", are the service's (`GET /v1/account/sharing`), and
  are asked before every send: content off sends the usage part only, usage off sends
  nothing and empties the queue.
* **Told first.** Before the first trace this version would send, it shows `NOTICE`, and
  the study the notice is shown for is not recorded. `DISCLOSURE_VERSION` says which
  notice; the service refuses a client whose notice is older than the one it requires,
  and a client that meets one sends nothing more.
* **Credentials removed** before a trace is written to the queue (`redact.Redactor`):
  this machine's keys by value, then the patterns the service applies again.
* **Bounded.** A tool's input and output are cut at `CONTENT_AT_MOST` characters and
  marked; images and other binary results are already their descriptions
  (`tools.describe`); a trace too large to send goes as its usage part.

A trace is its own record, apart from the capture plane (`capture.py`), which shows and
resumes a study on the service and has its own switch (`OPENREYNOLDS_CAPTURE`).

Only OpenReynolds' own interfaces share -- the terminal, `-p`, stream-json, and the local
MCP server (`cli.session`'s `sharing`). The hosted web app runs sessions through the same
code under its own interface, and its studies are the service's, not traces.
"""

from __future__ import annotations

import functools
import gzip
import json
import os
import platform
import stat
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import __version__
from .config import ACCOUNT_PAGE_URL, Config, config_path
from .redact import Redactor, known_values

DISCLOSURE_VERSION = 1
"""The notice this version shows, and the one the service must accept for it to send.
Raised with `NOTICE` whenever what a trace contains changes in a material way."""

PRIVACY_URL = "https://tryreynolds.com/privacy"

NOTICE = (
    "OpenReynolds shares a trace of each study while this machine is signed in to your "
    "Reynolds account, and this is the first time this version would.\n"
    "  - Usage: the tools the agent called and whether they worked, the model, tokens, "
    "timings, error types and the OpenReynolds version.\n"
    "  - Study content: your messages, the agent's reasoning and replies, and every "
    "tool's inputs and outputs, which can include commands, logs and parts of your case "
    "files. Binary files go as names and sizes only.\n"
    "Credentials are removed before anything leaves this machine. Model keys and the "
    "service key are never sent. This study is not shared.\n"
    "Stop it on this machine with `openreynolds sharing off` (or OPENREYNOLDS_SHARING=off). "
    "\"Share usage\" and \"Share study content\" on your Account page "
    f"({ACCOUNT_PAGE_URL}) apply to every machine signed in to your account. "
    f"Details: {PRIVACY_URL}"
)

CONTENT_AT_MOST = 20_000
"""Characters of any one string in a content record (a tool output, a command, a
message), the same cut the capture plane makes; the full text stays on this machine."""

SEND_AT_MOST = 4_500_000
"""Bytes of a gzipped trace; the service takes 5 MB. Over it, the usage part goes."""

QUEUE_AT_MOST = 100
"""Traces kept waiting to be sent; the oldest go first when there are more."""

END_BUDGET_S = 8.0
"""How long the close-down of a session may spend sending, after which what is left
waits for the next session's start."""

START_BUDGET_S = 60.0

_OFF = ("0", "false", "no", "off")

USAGE_KINDS = frozenset({"header", "turn", "tool"})


# -- the machine's switch and notice -------------------------------------------------


@dataclass
class State:
    enabled: bool = True
    notice: int = 0
    """The newest notice this machine has shown (`DISCLOSURE_VERSION`), 0 for none."""


def state_path() -> Path:
    return config_path().parent / "sharing.json"


def queue_dir() -> Path:
    return config_path().parent / "sharing-queue"


def load_state() -> State:
    try:
        raw = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return State()
    if not isinstance(raw, dict):
        return State()
    return State(enabled=raw.get("enabled") is not False,
                 notice=int(raw.get("notice") or 0) if str(raw.get("notice") or "0").isdigit() else 0)


def save_state(state: State) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_private(path, (json.dumps(asdict(state), indent=2) + "\n").encode("utf-8"))


def env_off() -> bool:
    return (os.environ.get("OPENREYNOLDS_SHARING") or "").strip().lower() in _OFF


def signed_in(cfg: Config) -> bool:
    return bool(getattr(cfg, "foamd_api_key", "") and getattr(cfg, "foamd_url", ""))


def active(cfg: Config, state: State | None = None) -> bool:
    """Whether this machine shares at all: signed in, and its switch on."""
    state = state or load_state()
    return signed_in(cfg) and not env_off() and state.enabled


def set_enabled(enabled: bool) -> State:
    state = load_state()
    state.enabled = enabled
    save_state(state)
    if not enabled:
        clear_queue()
    return state


def take_notice(cfg: Config) -> str | None:
    """`NOTICE` when this machine shares and has not shown this version's notice yet,
    marking it shown; None otherwise. Whoever takes it shows it, and does not record the
    study it is shown for."""
    state = load_state()
    if not active(cfg, state) or state.notice >= DISCLOSURE_VERSION:
        return None
    state.notice = DISCLOSURE_VERSION
    save_state(state)
    return NOTICE


def begin(cfg: Config, study_id: str, *, backend: str = "", solver: str = "") -> tuple[Recorder | None, str | None]:
    """At a session's start: (the recorder for this study or None, the notice to show
    or None). A machine that is not signed in has its queue emptied, which is what
    signing out does."""
    if not signed_in(cfg):
        clear_queue()
        return None, None
    if not active(cfg):
        return None, None
    notice = take_notice(cfg)
    if notice:
        return None, notice
    return Recorder(cfg, study_id, backend=backend, solver=solver), None


# -- recording one study ---------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cut(value: Any) -> tuple[Any, bool]:
    """`value` with every string cut at `CONTENT_AT_MOST`, and whether anything was."""
    cut = False

    def walk(v: Any) -> Any:
        nonlocal cut
        if isinstance(v, str):
            if len(v) > CONTENT_AT_MOST:
                cut = True
                return v[:CONTENT_AT_MOST]
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
                 environ: dict[str, str] | None = None):
        self.study_id = study_id
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
                    cut, _ = _cut(text)
                    self._add({"kind": "reasoning", "at": _now(), "text": self.redact.text(cut)})

    @_quiet
    def tool(self, name: str, tool_input: Any, output: str, is_error: bool,
             seconds: float | None = None) -> None:
        self._calls += 1
        call = self._calls
        output = output or ""
        cut_input, input_cut = _cut(tool_input)
        cut_output, output_cut = _cut(output)
        self._add({
            "kind": "tool", "at": _now(), "call": call, "tool": name, "ok": not is_error,
            "error_type": "error" if is_error else None,
            "seconds": round(seconds, 3) if seconds is not None else None,
            "output_chars": len(output), "truncated": input_cut or output_cut,
        })
        self._add({
            "kind": "tool_io", "call": call, "tool": name,
            "input": self.redact.deep(cut_input), "output": self.redact.text(cut_output),
            "truncated": input_cut or output_cut,
        })

    @_quiet
    def message(self, role: str, content: Any) -> None:
        """What the loop records for the transcript, other than tool calls (`tool`)."""
        if role == "tool" or content in (None, ""):
            return
        cut, _ = _cut(content if isinstance(content, str) else json.dumps(content, default=str))
        self._add({"kind": "message", "at": _now(), "role": role, "text": self.redact.text(cut)})

    def body(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
        header = {**self.header, "ended_at": _now(), "turns": self._turns, "tool_calls": self._calls}
        return {
            "id": uuid.uuid4().hex, "disclosure": DISCLOSURE_VERSION, "openreynolds": __version__,
            "study_id": self.study_id, "queued_at": _now(), "parts": "usage+content",
            "records": [header, *records],
        }

    def finish(self) -> Path | None:
        """Write this study's trace to the queue; its path, or None when there was
        nothing worth sending (no model turn) or the machine stopped sharing meanwhile."""
        if not self._turns or self.broken:
            return None
        body = self.body()
        data = gzip.compress(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        if len(data) > SEND_AT_MOST:
            body = usage_only(body)
            data = gzip.compress(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        return enqueue(body["id"], data)


def usage_only(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "parts": "usage",
            "records": [r for r in body.get("records") or [] if r.get("kind") in USAGE_KINDS]}


# -- the queue ---------------------------------------------------------------------------


def _write_private(path: Path, data: bytes) -> None:
    """Written whole or not at all, readable by this user only."""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
    os.replace(tmp, path)


def _queue() -> Path:
    folder = queue_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        folder.chmod(stat.S_IRWXU)
    return folder


def queued() -> list[Path]:
    folder = queue_dir()
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.json.gz"), key=lambda p: p.stat().st_mtime)


def enqueue(trace_id: str, data: bytes) -> Path | None:
    state = load_state()
    if env_off() or not state.enabled:
        return None
    path = _queue() / f"{trace_id}.json.gz"
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


def _client(cfg: Config, timeout: float) -> Any:
    import httpx

    return httpx.Client(base_url=cfg.foamd_url.rstrip("/"), timeout=timeout, follow_redirects=True,
                        headers={"Authorization": f"Bearer {cfg.foamd_api_key}",
                                 "User-Agent": f"openreynolds/{__version__}"})


def account_switches(cfg: Config, timeout: float = 10.0) -> dict[str, Any] | None:
    """The account's switches and the notice it requires, or None when the service did
    not answer. A machine that is not signed in has none."""
    if not signed_in(cfg):
        return None
    try:
        with _client(cfg, timeout) as http:
            answer = http.get("/v1/account/sharing")
    except Exception:  # noqa: BLE001 - offline is an answer of its own
        return None
    if answer.status_code != 200:
        return None
    try:
        got = answer.json()
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


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
    deadline = time.monotonic() + budget_s
    try:
        with _client(cfg, max(2.0, min(budget_s, 20.0))) as http:
            answer = http.get("/v1/account/sharing")
            if answer.status_code == 401:
                counts["dropped"] = clear_queue()   # the key was revoked: signed out
                return counts
            if answer.status_code != 200:
                counts["left"] = len(waiting)
                return counts
            switches = answer.json()
            if switches.get("share_usage") is False or int(switches.get("disclosure") or 0) > DISCLOSURE_VERSION:
                counts["dropped"] = clear_queue()
                return counts
            with_content = switches.get("share_content") is not False
            for path in waiting:
                if time.monotonic() > deadline:
                    break
                try:
                    body = json.loads(gzip.decompress(path.read_bytes()))
                except (OSError, ValueError, EOFError):
                    path.unlink(missing_ok=True)
                    counts["dropped"] += 1
                    continue
                if not with_content:
                    body = usage_only(body)
                sent = http.post("/v1/traces", content=gzip.compress(json.dumps(body).encode("utf-8")),
                                 headers={"content-type": "application/json", "content-encoding": "gzip"})
                code = (sent.json().get("error") if sent.headers.get("content-type", "").startswith(
                    "application/json") and sent.status_code >= 400 else "") or ""
                if sent.status_code in (200, 201):
                    path.unlink(missing_ok=True)
                    counts["sent"] += 1
                elif code in ("sharing_off", "disclosure_outdated") or sent.status_code == 401:
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
        lines.append("This machine is not signed in, so it shares nothing (`openreynolds login`).")
    elif env_off():
        lines.append("Sharing is off on this machine: OPENREYNOLDS_SHARING is set to off.")
    elif not state.enabled:
        lines.append("Sharing is off on this machine (`openreynolds sharing on` turns it back on).")
    elif state.notice < DISCLOSURE_VERSION:
        lines.append("Sharing is on on this machine. The next study shows what is shared, "
                     "and is not shared itself.")
    else:
        lines.append("Sharing is on on this machine.")
    if signed_in(cfg):
        switches = account_switches(cfg)
        if switches is None:
            lines.append("The account's switches could not be read (the service did not answer).")
        else:
            on = lambda v: "on" if v else "off"  # noqa: E731
            lines.append(f"Account: Share usage {on(switches.get('share_usage'))}, "
                         f"Share study content {on(switches.get('share_content'))} "
                         f"({switches.get('traces', 0)} traces shared). Change them on the "
                         "web app's Account page.")
            if int(switches.get("disclosure") or 0) > DISCLOSURE_VERSION:
                lines.append("The service needs a newer OpenReynolds before this machine can "
                             "share; upgrade it.")
    waiting = len(queued())
    if waiting:
        lines.append(f"{waiting} trace{'s' if waiting != 1 else ''} waiting to be sent.")
    lines.append(f"What is shared: {PRIVACY_URL}")
    return lines
