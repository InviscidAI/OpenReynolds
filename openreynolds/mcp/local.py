"""The MCP tools answered on this machine: `openreynolds mcp`.

Each study a caller starts runs here, in this process, as `cli.session` -- the same
function `openreynolds` runs in a terminal, through the same seam the terminal uses
(`interface`): a `LocalView` that keeps the picture `study_status` reads, and a
`LocalReader` the caller's messages are put on. Where the workspace is, and which
model, is what the configuration says, exactly as for the terminal: a Reynolds
workspace when signed in, this machine's Docker with `--local`.

Studies live as long as this process. When the client that started the server goes
away, `LocalHost.close` ends every live study the way End does -- jobs stopped,
workspace stopped, files kept -- and `start_study` with its study_id continues it next
time.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .. import cli, modes
from .. import sharing as sharing_mod
from ..backend.base import WORKSPACE_ROOT, BackendError
from ..config import Config
from ..mirror import OutsideMirror, local_for
from ..store import Store, TornRecord, list_studies
from ..study_style import normalize_style, preferences_for
from ..transcript import TranscriptTail, read_items
from ..uploads import place_upload, placed_line, safe_filename
from ..view import View, plain_console
from ..watch import NOTHING
from .core import (ACTIVE, MAX_UPLOAD_BYTES, WORKSPACE_FAILED, WORKSPACE_READY, WORKSPACE_STARTING,
                   HostDefaults, McpError, MemoryKv, Staged)

IDLE_TIMEOUT_S = 1800.0
"""How long a study may go unpolled, waiting for input, before it ends itself -- as
`study_status` tells the caller. A caller that went away must not leave a workspace
running for ever."""
IDLE_TICK_S = 1.0
IDLE_NOTE = ("Nobody has polled this study for {minutes} minutes, so the session is ending "
             "on its own, as End would: its jobs and its workspace are stopped. The study is "
             "kept; start_study with its study_id carries on from here.")
READ_LIMIT = 4 * 1024 * 1024
"""The most of one file `read_file` hands a tool: results.json and a report, not a field."""
CLOSE_WAIT_S = 600.0
"""How long `close` waits for the live studies to finish ending."""
_STUDY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")

INSTRUCTIONS = (
    "Reynolds is a CFD agent with a real OpenFOAM workspace: in the Reynolds cloud when "
    "this machine is signed in, or this machine's own Docker when the server was started "
    "with --local. Give it a simulation task in words -- the geometry, the flow "
    "conditions, what you want to know -- and it sets up the case, meshes, solves and "
    "reports.\n\n"
    "Typical use: upload_file with the path of each input file, start_study with the task "
    "and the upload ids, then call study_status until the state is waiting_for_input or "
    "ended, then get_results, download_url for the local path of any file you want, and "
    "end_study. A study can take from minutes to hours; poll patiently, and answer "
    "questions with send_message. Studies run inside this server: if it stops, its "
    "studies end with their files kept, and start_study with the study_id continues one."
)

DESCRIPTIONS = {
    "upload_file": (
        "Stage one input file (a geometry such as STL/STEP/OBJ, a drawing, a data file) for "
        "a study, by its path on this machine (absolute, or relative to where this server "
        "runs). Then pass the upload_id to start_study. Up to 100 MB per file."
    ),
    "download_url": (
        "Where one file of a study is on this machine (a path from get_results or the "
        "transcript, e.g. /work/<study>/case/postProcessing/forces/0/forces.dat). A live "
        "study's file is copied out of its workspace first."
    ),
    "get_results:next": "download_url(study_id, path) gives the local path of any of these files.",
}


class LocalReader:
    """What the session reads typed lines from: a queue the tools put on.

    `get` -- the wait at the prompt -- also watches the idle clock: once nobody has
    polled for `idle_timeout_s`, it hands the loop `/exit`. `poll`, which the job
    watch uses, does not: a study watching a solve is doing the work it was left to do."""

    accepts_input = True

    def __init__(self, idle_timeout_s: float = IDLE_TIMEOUT_S, on_idle: Callable[[], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        import queue
        self._queue_mod = queue
        self.queue: Any = queue.Queue()
        self.idle_timeout_s = max(0.0, float(idle_timeout_s or 0.0))
        self.on_idle = on_idle
        self._clock = clock
        self.last_heartbeat = clock()
        self.idle_out = False

    def heartbeat(self) -> None:
        self.last_heartbeat = self._clock()

    def put(self, line: str | None) -> None:
        self.heartbeat()
        self.queue.put(line)

    def get(self, timeout: float | None = None) -> str | None:
        Empty = self._queue_mod.Empty
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                # A line already waiting is the caller, whatever the clock says.
                return self.queue.get_nowait()
            except Empty:
                pass
            if self.idle_timeout_s > 0 and not self.idle_out:
                remaining = self.idle_timeout_s - (self._clock() - self.last_heartbeat)
                if remaining <= 0:
                    self.idle_out = True
                    if self.on_idle is not None:
                        try:
                            self.on_idle()
                        except Exception:  # noqa: BLE001 - the note is a courtesy; the exit is the point
                            pass
                    return "/exit"
                slice_s = min(IDLE_TICK_S, max(remaining, 0.05))
            else:
                slice_s = IDLE_TICK_S
            if deadline is not None:
                slice_s = min(slice_s, max(0.0, deadline - time.monotonic()))
            try:
                return self.queue.get(timeout=slice_s)
            except Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    return None

    def poll(self) -> Any:
        try:
            return self.queue.get_nowait()
        except self._queue_mod.Empty:
            return NOTHING

    def putback(self, line: str | None) -> None:
        self.queue.put(line)

    def pending(self) -> bool:
        return not self.queue.empty()


class LocalView(View):
    """Every session event, kept as the picture `study_status` reads (`core.Host`)."""

    surface = "mcp"

    def __init__(self, studies_dir: Path, style: str):
        self.studies_dir = studies_dir
        self.browser: Any = None
        self.reader: LocalReader | None = None
        self.tail: TranscriptTail | None = None
        self._lock = threading.RLock()
        self.session: dict[str, Any] = {"status": "starting", "workspace": WORKSPACE_STARTING, "style": style}
        self.turn = "agent"
        self.stage_text = ""
        self.narration_text = ""
        self.job_rows: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self.side: list[dict[str, Any]] = []
        self.notes: list[str] = []

    # -- the picture --------------------------------------------------------------

    def picture(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            return {
                "session": dict(self.session), "turn": self.turn, "stage": self.stage_text,
                "narration": self.narration_text, "jobs": [dict(j) for j in self.job_rows],
                "items": [dict(i) for i in self.items], "side": [dict(s) for s in self.side],
            }

    def refresh(self) -> None:
        with self._lock:
            tail = self.tail
            if tail is None:
                return
            fresh = tail.read_new()
            if fresh:
                self.items.extend(fresh)

    def _attach(self, study_id: str) -> None:
        with self._lock:
            if self.tail is None and study_id and _STUDY_ID.match(study_id):
                self.tail = TranscriptTail(self.studies_dir / study_id / "messages.jsonl")

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self.session.update(changes)

    # -- View ------------------------------------------------------------------

    def header(self, study_id: str, instance_id: str, model: str, mirror: Path) -> None:
        self._set(status="running", study_id=study_id, instance_id=instance_id, model=model,
                  mirror_dir=str(mirror), started_at=time.time())
        self._attach(study_id)

    def thinking_begin(self) -> None:
        self.turn = "agent"
        self.stage_text = "thinking..."

    def thinking_delta(self, text: str) -> None:
        pass

    def text_delta(self, text: str) -> None:
        self.turn = "agent"

    def turn_end(self) -> None:
        self.stage_text = ""

    def tool(self, name: str, summary: str) -> None:
        self.turn = "agent"
        self.stage_text = f"{name}: {summary[:90]}"

    def tool_error(self, message: str) -> None:
        self._note(message)

    def notice(self, message: str) -> None:
        self._note(message)

    def warn(self, message: str) -> None:
        self._note(message)

    def info(self, message: str) -> None:
        self._note(message)

    def _note(self, message: str) -> None:
        with self._lock:
            self.notes.append(str(message))
            del self.notes[:-50]

    def usage(self, tokens: int, fraction: float) -> None:
        pass

    def watching(self, names: list[str]) -> None:
        self.stage_text = f"watching {len(names)} job(s)"

    def jobs(self, records: list[Any]) -> None:
        keys = ("job_id", "name", "cmd", "launched_at", "status", "end_reason", "exit_code", "cwd")
        rows = [{k: (r.get(k) if isinstance(r, dict) else getattr(r, k, None)) for k in keys} for r in records]
        with self._lock:
            self.job_rows = rows

    def stage(self, text: str) -> None:
        self.stage_text = text

    def step(self, number: int, seconds: float, tool_calls: int) -> None:
        pass

    def working(self, what: str) -> None:
        self.turn = "agent"
        self.stage_text = f"{what}..."

    def interjection(self, text: str) -> None:
        pass

    def workspace(self, browser: Any) -> None:
        self.browser = browser
        store = getattr(browser, "store", None)
        if store is not None:
            self._attach(store.session.study_id)
        self._set(home=str(getattr(browser, "home", "") or ""))

    def workspace_ready(self, instance_id: str, seconds: float) -> None:
        if instance_id:
            self._set(instance_id=instance_id)
        self._set(workspace=WORKSPACE_READY)

    def workspace_starting(self) -> None:
        self._set(workspace=WORKSPACE_STARTING)

    def workspace_failed(self, reason: str) -> None:
        self._set(workspace=WORKSPACE_FAILED)
        self.stage_text = ""
        self.narration_text = ""

    def show_files(self, path: str = "", depth: int = 0) -> None:
        pass

    def show_renders(self, renders_dir: Any) -> None:
        pass

    def status(self, lines: list[str]) -> None:
        pass

    def mirrored(self, report: Any) -> None:
        pass

    def prompt(self) -> None:
        # A line already waiting is the caller's turn taken: the loop reads it a moment
        # later. Saying "user" in that gap would tell a polling caller the agent has
        # finished a turn it has not begun (the hosted view's `first_held`).
        reader = self.reader
        self.turn = "agent" if reader is not None and reader.pending() else "user"

    def progress(self, snapshot: Any) -> None:
        pass

    def narration(self, text: str) -> None:
        self.narration_text = text

    def received(self, text: str) -> None:
        pass

    def side_question(self, text: str) -> None:
        with self._lock:
            self.side.append({"question": text, "answer": None})

    def side_answer(self, text: str, forwarded: bool = False) -> None:
        with self._lock:
            waiting = next((e for e in self.side if e.get("answer") is None), None)
            if waiting is None:
                waiting = {"question": ""}
                self.side.append(waiting)
            waiting["answer"] = text
            if forwarded:
                waiting["forwarded"] = True

    def desk_step(self, event: dict[str, Any]) -> None:
        pass

    def delivered(self, event: Any) -> None:
        pass

    def model(self, model: str, effort: str, provider: str) -> None:
        self._set(model=model, effort=effort, provider=provider)

    def mode(self, mode: str) -> None:
        self._set(mode=mode)

    def solver(self, solver: str) -> None:
        if solver:
            self._set(solver=str(solver))

    def approval(self, request_id: str, kind: str, title: str, detail: str, choices: list[str]) -> None:
        self.refresh()
        with self._lock:
            self.items.append({
                "id": f"approval-{request_id}", "kind": "approval", "request_id": request_id,
                "subject": kind, "title": title, "detail": detail, "choices": list(choices),
                "state": "open", "outcome": "", "note": "",
            })
        self.stage_text = f"waiting for you: {title[:90]}"
        self.turn = "user"

    def approval_done(self, request_id: str, outcome: str, note: str = "") -> None:
        with self._lock:
            for item in reversed(self.items):
                if item.get("id") == f"approval-{request_id}":
                    item.update(state="done", outcome=outcome, note=note)
                    break
        self.stage_text = ""
        self.turn = "agent"


class LocalSession:
    """One study running in this process."""

    def __init__(self, session_id: str, view: LocalView, reader: LocalReader, study_id: str | None):
        self.session_id = session_id
        self.view = view
        self.reader = reader
        self.status = "starting"
        self.error = ""
        self.study_id = study_id or ""
        self.thread: threading.Thread | None = None

    @property
    def live(self) -> bool:
        return self.status in ACTIVE

    def row(self) -> dict[str, Any]:
        session = self.view.session
        return {"status": self.status, "study_id": session.get("study_id") or self.study_id,
                "instance_id": session.get("instance_id") or "", "error": self.error}


SessionRunner = Callable[..., Any]


class LocalHost(HostDefaults):
    """`core.Host` for studies run in this process."""

    instructions = INSTRUCTIONS
    descriptions = DESCRIPTIONS

    def __init__(self, *, load_config: Callable[[], Config] = Config.load,
                 run_session: SessionRunner | None = None, local_workspace: Callable[[], Any] | None = None,
                 staging_dir: Path | None = None, cwd: Path | None = None,
                 idle_timeout_s: float = IDLE_TIMEOUT_S):
        self.load_config = load_config
        self.run_session = run_session or cli.session
        self.local_workspace = local_workspace or cli.local_workspace
        self.staging_dir = staging_dir or Path(tempfile.mkdtemp(prefix="reynolds-mcp-"))
        self.cwd = cwd or Path.cwd()
        self.idle_timeout_s = idle_timeout_s
        self.kv = MemoryKv()
        self.sessions: dict[str, LocalSession] = {}
        self._lock = threading.Lock()
        self._closing = False

    def _studies_dir(self) -> Path:
        return Path(self.load_config().studies_dir)

    # -- files in -----------------------------------------------------------------

    def stage_upload(self, filename: str) -> dict[str, Any]:
        source = Path(filename).expanduser()
        if not source.is_absolute():
            source = self.cwd / source
        if not source.is_file():
            raise McpError(f"no file at {source}")
        size = source.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise McpError(f"{source} is {size} bytes; the limit is {MAX_UPLOAD_BYTES}")
        upload_id = uuid.uuid4().hex
        name = safe_filename(source.name)
        folder = self.staging_dir / upload_id
        folder.mkdir(parents=True)
        shutil.copyfile(source, folder / name)
        return {"upload_id": upload_id, "name": name, "path": str(source), "size": size,
                "next": "Pass the upload_id to start_study."}

    def staged(self, upload_id: str) -> Staged | None:
        folder = self.staging_dir / upload_id
        files = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
        if not files:
            return None
        return Staged(key=upload_id, name=files[0].name, size=files[0].stat().st_size)

    def place_files(self, session_id: str, instance_id: str, home: str,
                    files: list[dict[str, Any]]) -> list[str] | None:
        session = self._session(session_id)
        backend = getattr(session.view.browser, "backend", None)
        if backend is None:
            return None
        placed = []
        for item in files:
            folder = self.staging_dir / str(item["key"])
            source = folder / str(item["name"])
            target, extras, made = place_upload(backend, home, item["name"], source.read_bytes())
            placed.append(placed_line(target, extras, made))
        return placed

    def discard_staged(self, files: list[dict[str, Any]]) -> None:
        for item in files:
            shutil.rmtree(self.staging_dir / str(item.get("key") or "_"), ignore_errors=True)

    # -- the session ---------------------------------------------------------------

    def start_session(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._closing:
            raise McpError("this server is shutting down")
        cfg = self.load_config()
        missing = cfg.missing(local=self.local_workspace() is not None)
        if missing:
            raise McpError(f"this machine is missing configuration: {', '.join(missing)}. "
                           "`openreynolds login` gets a Reynolds key; `openreynolds config` sets a "
                           "model key; `openreynolds mcp --local` runs on this machine's Docker.")
        try:
            style = normalize_style(body.get("style"))
        except ValueError as exc:
            raise McpError(f"depth: {exc}") from exc
        mode = modes.normalize(body.get("mode")) if body.get("mode") else None
        if body.get("mode") and mode is None:
            raise McpError(f"mode is one of {', '.join(modes.MODES)}")
        if mode:
            cfg.mode = mode
        if body.get("model"):
            cfg.model = str(body["model"])
        if body.get("effort"):
            cfg.effort = str(body["effort"]).lower()
        # The person's own standing note, then the caller's: after the depth.
        notes = "\n\n".join(t for t in (cfg.preferences.strip(), str(body.get("notes") or "").strip()) if t)
        cfg.preferences = preferences_for(style, notes)
        study_id = body.get("study_id") or None
        if study_id and not (_STUDY_ID.match(study_id) and (Path(cfg.studies_dir) / study_id).is_dir()):
            raise McpError(f"no study {study_id} on this machine (list_studies names them)")

        # The sharing notice, the first time this version would share: put in the answer,
        # because nobody reads this server's stderr. The study is shared like any other:
        # its trace goes after it ends, and `sharing off` before then shares nothing.
        notice = sharing_mod.take_notice(cfg)

        session_id = str(uuid.uuid4())
        view = LocalView(Path(cfg.studies_dir), style)
        reader = LocalReader(self.idle_timeout_s,
                             on_idle=lambda: view.info(IDLE_NOTE.format(minutes=max(1, round(self.idle_timeout_s / 60)))))
        view.reader = reader
        session = LocalSession(session_id, view, reader, study_id)
        if study_id:
            view._attach(study_id)
        message = body.get("message")
        if message:
            # Put before the loop starts: the first thing it reads at its prompt.
            reader.put(str(message))

        def interface(drive: Any) -> bool:
            drive(view, reader)
            return False

        def run() -> None:
            try:
                self.run_session(
                    cfg, study_id=study_id, instance_id=None, one_shot=None, keep_alive=False,
                    interface=interface, mode_explicit=mode is not None, model_explicit=bool(body.get("model")),
                    solver=body.get("solver") or None, route_on=body.get("route_on") or None,
                    sharing=True,
                )
                session.status = "ended"
            except SystemExit as exc:
                session.error = view.notes[-1] if view.notes else f"exit {exc.code}"
                session.status = "error"
            except BaseException as exc:  # noqa: BLE001 - said to the caller, never swallowed
                session.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                session.status = "error"
            finally:
                view._set(status=session.status)
                view.refresh()

        session.status = "running"
        session.thread = threading.Thread(target=run, name=f"study-{session_id[:8]}")
        with self._lock:
            self.sessions[session_id] = session
        session.thread.start()
        started = {"session_id": session_id, "study_id": study_id, "model": cfg.model}
        if notice:
            started["sharing_notice"] = notice
        return started

    def _session(self, session_id: str) -> LocalSession:
        try:
            uuid.UUID(session_id)
        except (ValueError, TypeError) as exc:
            raise McpError(f"{session_id!r} is not a session id (start_study returns one)") from exc
        with self._lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise McpError(f"no session {session_id} in this server")
        return session

    def check(self, session_id: str) -> None:
        self._session(session_id)

    def heartbeat(self, session_id: str) -> None:
        self._session(session_id).reader.heartbeat()

    def observe(self, session_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        session = self._session(session_id)
        return session.row(), session.view.picture()

    def say(self, session_id: str, text: str) -> None:
        session = self._session(session_id)
        if not session.live:
            raise McpError(f"session {session_id} has ended ({session.status}); "
                           "start_study with its study_id to resume the study")
        session.reader.put(text)

    def end(self, session_id: str) -> str:
        session = self._session(session_id)
        if not session.live:
            return session.status
        session.status = "ending"
        session.view._set(status="ending")
        session.reader.put("/exit")
        return "ending"

    def close(self, wait_s: float = CLOSE_WAIT_S) -> None:
        """End every live study, as End does, and wait for them to finish ending."""
        self._closing = True
        with self._lock:
            live = [s for s in self.sessions.values() if s.live]
        for session in live:
            self.end(session.session_id)
        deadline = time.monotonic() + wait_s
        for session in live:
            if session.thread is not None:
                session.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        shutil.rmtree(self.staging_dir, ignore_errors=True)

    # -- studies and their files ---------------------------------------------------

    def _live_for(self, study_id: str) -> LocalSession | None:
        with self._lock:
            for session in self.sessions.values():
                if session.live and session.row()["study_id"] == study_id:
                    return session
        return None

    def _store(self, study_id: str) -> Store:
        root = self._studies_dir()
        if not (_STUDY_ID.match(study_id or "") and (root / study_id / "session.json").is_file()):
            raise McpError(f"no study {study_id} on this machine (list_studies names them)")
        return Store(root, study_id)

    def studies(self) -> dict[str, Any]:
        cards = [{"study_id": s.study_id, "title": s.title, "created_at": s.created_at}
                 for s in list_studies(self._studies_dir())]
        with self._lock:
            live = [{"session_id": s.session_id, "study_id": s.row()["study_id"] or None, "status": s.status}
                    for s in self.sessions.values() if s.live]
        return {"studies": cards, "sessions": live}

    def transcript(self, study_id: str) -> dict[str, Any]:
        store = self._store(study_id)
        info = store.session
        session = {
            "study_id": info.study_id, "title": info.title, "home": info.home or f"{WORKSPACE_ROOT}/{study_id}",
            "jobs": [{"job_id": j.job_id, "name": j.name, "status": j.status, "end_reason": j.end_reason,
                      "exit_code": j.exit_code} for j in info.jobs.values()],
        }
        return {"items": read_items(store.dir / "messages.jsonl"), "session": session}

    def transcript_items(self, study_id: str, after_seq: int) -> list[dict[str, Any]]:
        try:
            store = self._store(study_id)
        except McpError:
            return []
        return [i for i in read_items(store.dir / "messages.jsonl")
                if isinstance(i.get("seq"), int) and i["seq"] > after_seq]

    def _check_path(self, path: str) -> str:
        """A workspace path, or a refusal.

        Two things are being protected and they need two different tests. The backend
        is asked for `path` verbatim, and nothing here can resolve a path on the far
        side of it, so that one stays a syntactic test -- now of `..` as a segment
        under either separator, because `\\` is what `Path` splits on when this runs
        on Windows (#180). The copy on this machine is protected by the join itself:
        `_local_for` resolves it and refuses anything that lands outside the study's
        mirror, which answers spellings this test has never heard of.
        """
        parts = path.replace("\\", "/").split("/")
        if not path.startswith(WORKSPACE_ROOT + "/") or ".." in parts:
            raise McpError(f"{path!r} is not a workspace path (they start {WORKSPACE_ROOT}/)")
        return path

    @staticmethod
    def _local_for(store: Store, path: str) -> Path:
        """Where `path` is on this machine -- inside this study's mirror, or nowhere."""
        try:
            return local_for(store.files_dir, path)
        except OutsideMirror as exc:
            raise McpError(f"{path!r} is not a workspace path "
                           f"(they start {WORKSPACE_ROOT}/)") from exc

    def files(self, study_id: str) -> list[dict[str, Any]]:
        store = self._store(study_id)
        live = self._live_for(study_id)
        browser = getattr(live.view, "browser", None) if live else None
        if browser is not None:
            try:
                return [{"path": e.path, "size": e.size, "is_dir": e.is_dir}
                        for e in browser.tree(browser.home, depth=12)]
            except Exception:  # noqa: BLE001 - the workspace is not answering; the copy here is
                pass
        root = store.files_dir
        if not root.is_dir():
            return []
        return [{"path": f"{WORKSPACE_ROOT}/{p.relative_to(root).as_posix()}", "size": p.stat().st_size,
                 "is_dir": False} for p in sorted(root.rglob("*")) if p.is_file()]

    def read_file(self, study_id: str, path: str) -> bytes | None:
        store = self._store(study_id)
        self._check_path(path)
        live = self._live_for(study_id)
        backend = getattr(getattr(live.view, "browser", None), "backend", None) if live else None
        if backend is not None:
            try:
                size = backend.stat(path).size
                return backend.get_file(path, 0, min(size, READ_LIMIT))
            except BackendError:
                pass
        local = self._local_for(store, path)
        if local.is_file():
            with local.open("rb") as fh:
                return fh.read(READ_LIMIT)
        return None

    def renders(self, study_id: str) -> list[dict[str, Any]]:
        store = self._store(study_id)
        if not store.renders_dir.is_dir():
            return []
        pics = sorted((p for p in store.renders_dir.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
        return [{"name": p.name} for p in pics]

    def render_link(self, study_id: str, name: str) -> str:
        return str(self._store(study_id).renders_dir / Path(name).name)

    def download(self, study_id: str, path: str) -> dict[str, Any]:
        store = self._store(study_id)
        self._check_path(path)
        live = self._live_for(study_id)
        browser = getattr(live.view, "browser", None) if live else None
        if browser is not None:
            try:
                browser.pull(path)
            except BackendError as exc:
                raise McpError(f"could not copy {path} out of the workspace: {exc}") from exc
        local = self._local_for(store, path)
        if not local.exists():
            raise McpError(f"{path} is not in this machine's copy of the study"
                           + ("" if browser is not None else
                              "; its workspace is stopped -- start_study with the study_id to reach it"))
        size = local.stat().st_size if local.is_file() else None
        return {"path": str(local), "size": size, "next": "The file is on this machine at path."}

    def explain(self, exc: BaseException) -> str | None:
        if isinstance(exc, BackendError):
            return f"the workspace refused: {exc}"
        if isinstance(exc, TornRecord):
            # In its own words, as a 400 rather than a 500: the record is the
            # caller's to repair, and nothing has been written to it (#167).
            return str(exc)
        return None


def serve() -> None:
    """`openreynolds mcp`: the tools over stdio, until the client goes away; then every
    live study ends as End would."""
    import sys

    import anyio

    from .core import build_server

    # Everything a session would print goes to stderr: stdout is the protocol.
    cli.console = plain_console(file=sys.stderr)
    host = LocalHost()
    server = build_server(host)
    try:
        anyio.run(server.run_stdio_async)
    except KeyboardInterrupt:
        pass
    finally:
        host.close()
