"""`openreynolds mcp`: the MCP tools answered on this machine.

The tools are `openreynolds/mcp/core.py`'s, the ones the hosted Reynolds MCP server
offers too; here they are answered by `LocalHost`, which runs each study in this
process through `cli.session`'s `interface` seam. The session itself is a stand-in
that drives the view and reads the reader the way `cli.session` does, and writes its
transcript to `messages.jsonl` through the real `Store`.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from openreynolds.backend.base import BackendError  # noqa: E402
from openreynolds.mcp import core  # noqa: E402
from openreynolds.mcp.local import LocalHost, LocalReader  # noqa: E402
from openreynolds.store import Store, new_study_id  # noqa: E402
from openreynolds.study_style import STYLES  # noqa: E402


@dataclass
class Cfg:
    studies_dir: Path
    model: str = "test-model"
    mode: str = "auto"
    effort: str = ""
    preferences: str = ""
    gaps: list[str] = field(default_factory=list)

    def missing(self, *, local: bool = False) -> list[str]:  # noqa: ARG002
        return list(self.gaps)


class FakeBackend:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def stat(self, path: str):
        if path not in self.files:
            raise BackendError("no such file")
        return type("S", (), {"size": len(self.files[path]), "is_dir": False})()

    def put_file(self, path: str, data: bytes) -> None:
        self.files[path] = data


    def get_file(self, path: str, offset: int = 0, limit: int | None = None, **_: Any) -> bytes:
        return self.files[path][offset: None if limit is None else offset + limit]


class FakeBrowser:
    def __init__(self, store: Store, home: str):
        self.store = store
        self.home = home
        self.backend = FakeBackend()

    def tree(self, path: str, depth: int = 0):
        entry = type("E", (), {})
        out = []
        for p, data in sorted(self.backend.files.items()):
            e = entry()
            e.path, e.size, e.is_dir = p, len(data), False
            out.append(e)
        return out

    def pull(self, path: str):
        from openreynolds.mirror import local_for
        target = local_for(self.store.files_dir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.backend.files[path])
        return [target]


class Studies:
    """A stand-in for `cli.session`: it answers each line with one reply, and asks an
    approval when told to."""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[dict[str, Any]] = []
        self.browsers: dict[str, FakeBrowser] = {}
        self.ready = threading.Event()

    def __call__(self, cfg: Cfg, *, study_id: str | None, interface: Any, **kwargs: Any) -> None:
        self.calls.append({"cfg": cfg, "study_id": study_id, **kwargs})
        sid = study_id or new_study_id()
        store = Store(self.root, sid)
        store.session.home = f"/work/{sid}"
        store.save()
        if study_id is None:
            # As `cli.session` does: the briefing is a new study's first row (seq 0).
            store.append_message("event", f"study {sid}: briefing")

        def drive(view: Any, reader: Any) -> None:
            view.header(sid, "inst-1", cfg.model, store.files_dir)
            browser = FakeBrowser(store, f"/work/{sid}")
            self.browsers[sid] = browser
            view.workspace(browser)
            self.ready.wait(5)
            view.workspace_ready("inst-1", 0.1)
            while True:
                view.prompt()
                line = reader.get()
                if line is None or line == "/exit":
                    return
                store.append_message("user", line)
                view.thinking_begin()
                if line == "ask me":
                    view.approval("r1", "job", "Run the solve?", "simpleFoam, 4 cores", ["yes", "no"])
                    answer = reader.get()
                    view.approval_done("r1", "approved" if answer == "/yes" else "declined")
                    store.append_message("assistant", f"approval: {answer}")
                    continue
                if line.startswith("/btw "):
                    view.side_question(line[5:])
                    view.side_answer("42", False)
                    continue
                browser.backend.put_file(f"/work/{sid}/results.json", b'{"drag_N": 1.5}')
                store.append_message("assistant", f"done: {line}")
                view.turn_end()

        interface(drive)


@pytest.fixture
def world(tmp_path: Path):
    root = tmp_path / "studies"
    root.mkdir()
    studies = Studies(root)
    cfg_box = {"gaps": []}
    host = LocalHost(load_config=lambda: Cfg(root, gaps=list(cfg_box["gaps"])), run_session=studies,
                     local_workspace=lambda: None, staging_dir=tmp_path / "staging", cwd=tmp_path)
    server = core.build_server(host)
    yield type("W", (), {"host": host, "server": server, "studies": studies, "root": root,
                         "tmp": tmp_path, "cfg": cfg_box})()
    studies.ready.set()
    host.close(wait_s=10)


def call(world: Any, tool: str, **args: Any) -> dict[str, Any]:
    from mcp.server.mcpserver.exceptions import ToolError
    try:
        result = asyncio.run(world.server.call_tool(tool, args))
    except ToolError as exc:
        raise core.McpError(str(exc)) from exc
    if result.is_error:
        raise core.McpError(result.content[0].text)
    if result.structured_content is not None:
        data = result.structured_content
        return data.get("result", data) if set(data) == {"result"} else data
    return json.loads(result.content[0].text)


def until(world: Any, session_id: str, want: str, after: int = 0) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while True:
        status = call(world, "study_status", session_id=session_id, after_seq=after, wait_seconds=0)
        if status["state"] == want or time.monotonic() > deadline:
            return status
        time.sleep(0.05)


def test_the_tools_are_the_hosted_servers_tools(world):
    tools = {t.name: t for t in asyncio.run(world.server.list_tools())}
    assert set(tools) == {k for k in core.DESCRIPTIONS if ":" not in k}
    start = tools["start_study"].input_schema["properties"]
    assert set(start) == {"task", "upload_ids", "study_id", "depth", "mode", "model", "effort", "notes", "solver"}


def test_a_study_runs_to_an_answer_and_ends(world):
    started = call(world, "start_study", task="flow past a cylinder at Re=100")
    sid = started["session_id"]
    assert started["state"] == "starting"
    assert until(world, sid, "starting")["state"] == "starting", "the workspace is not up yet"
    world.studies.ready.set()
    status = until(world, sid, "waiting_for_input")
    assert status["state"] == "waiting_for_input", status
    assert status["last_message"] == "done: flow past a cylinder at Re=100"
    assert [r.get("from") for r in status["transcript"]] == ["caller", "reynolds"], "after the briefing (seq 0)"
    study = status["study_id"]

    cfg = world.studies.calls[0]["cfg"]
    assert cfg.preferences.startswith(STYLES["quick"]), "the depth is the standing note"
    assert core.CALLER_NOTE in cfg.preferences
    assert world.studies.calls[0]["mode_explicit"] is True

    results = call(world, "get_results", study_id=study)
    assert results["final_message"] == "done: flow past a cylinder at Re=100"
    assert results["results_json"] == {"drag_N": 1.5}
    assert {"path": f"/work/{study}/results.json", "size": 15} in results["files"]

    got = call(world, "download_url", study_id=study, path=f"/work/{study}/results.json")
    assert Path(got["path"]).read_bytes() == b'{"drag_N": 1.5}'

    assert call(world, "end_study", session_id=sid) == {"status": "ending"}
    assert until(world, sid, "ended")["state"] == "ended"
    with pytest.raises(core.McpError, match="has ended"):
        call(world, "send_message", session_id=sid, text="more")
    listed = call(world, "list_studies")
    assert [s["study_id"] for s in listed["studies"]] == [study]
    assert listed["live_sessions"] == []


def test_a_second_message_and_the_cursor(world):
    world.studies.ready.set()
    sid = call(world, "start_study", task="first")["session_id"]
    first = until(world, sid, "waiting_for_input")
    # #188: `waiting_for_input` used to be reported off `picture["turn"]` alone, which
    # another thread sets, while the cursor came from `picture["items"]`, which had not
    # caught up -- so this came back 0 about 1 run in 40 and the next poll was handed
    # the whole transcript again as if it were new.
    assert first["state"] == "waiting_for_input", first
    assert first["next_after_seq"] > 0, first
    assert first["transcript"], "the turn it says to go and read is in this answer"

    call(world, "send_message", session_id=sid, text="second")
    deadline = time.monotonic() + 10
    later = None
    while time.monotonic() < deadline:
        later = call(world, "study_status", session_id=sid, after_seq=first["next_after_seq"], wait_seconds=0)
        if later["state"] == "waiting_for_input" and later["transcript"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"the second turn never came back in 10 s: {later}")
    assert [r["text"] for r in later["transcript"]] == ["second", "done: second"]
    assert later["next_after_seq"] > first["next_after_seq"], "the cursor has to move"


def test_waiting_for_input_is_never_said_before_the_answer_is_in_the_transcript():
    """#188, the defect under the flake, deterministically. The state and the cursor
    come from two fields of one picture that two threads fill at two moments: `turn`
    (the session thread, `LocalView.prompt`) and `items` (the refresh, from the
    transcript tail). `waiting_for_input`'s own `next` is "Reynolds has finished its
    turn. Read its last message" -- so the state is only true where that message is in
    the transcript the same call returns."""
    row = {"status": "running"}
    picture = {"session": {"status": "running", "workspace": core.WORKSPACE_READY},
               "turn": "user", "items": [], "jobs": []}

    assert core.state_of(row, picture, False) == "working", "an empty transcript"

    picture["items"] = [{"seq": 1, "kind": "user", "text": "first"}]
    assert core.state_of(row, picture, False) == "working", "nobody has answered it"

    picture["items"].append({"seq": 2, "kind": "assistant", "text": "done: first"})
    assert core.state_of(row, picture, False) == "waiting_for_input"
    # And the cursor the same call hands back covers what it just returned, which is
    # the invariant the flake broke: an empty slice leaves `cursor = after_seq`.
    rows, cursor, _cut = core.transcript_since(picture["items"], 0)
    assert rows and cursor == 2


def test_an_approval_is_asked_and_answered(world):
    world.studies.ready.set()
    sid = call(world, "start_study", task="ask me", mode="partial")["session_id"]
    status = until(world, sid, "waiting_for_approval")
    assert status["approvals"][0]["title"] == "Run the solve?"
    call(world, "answer_approval", session_id=sid, decision="approve")
    status = until(world, sid, "waiting_for_input")
    assert status["last_message"] == "approval: /yes"


def test_a_side_question_comes_back_under_side(world):
    world.studies.ready.set()
    sid = call(world, "start_study", task="first")["session_id"]
    until(world, sid, "waiting_for_input")
    call(world, "send_message", session_id=sid, text="/btw what is the answer")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = call(world, "study_status", session_id=sid, wait_seconds=0)
        if status.get("side"):
            break
    assert status["side"] == [{"question": "what is the answer", "answer": "42"}]


def test_an_uploaded_file_reaches_the_workspace_before_the_task(world):
    geometry = world.tmp / "wing section.stl"
    geometry.write_bytes(b"solid wing\nendsolid wing\n")
    up = call(world, "upload_file", filename="wing section.stl")
    assert up["name"] == "wing_section.stl" and "curl" not in up
    sid = call(world, "start_study", task="drag on this wing", upload_ids=[up["upload_id"]])["session_id"]
    assert world.studies.calls[0]["route_on"] == "drag on this wing", "the solver is chosen from the task"
    assert until(world, sid, "starting")["state"] == "starting"
    world.studies.ready.set()
    status = until(world, sid, "waiting_for_input")
    study = status["study_id"]
    placed = f"/work/{study}/uploads/wing_section.stl"
    assert world.studies.browsers[study].backend.files[placed] == geometry.read_bytes()
    said = [r["text"] for r in status["transcript"] if r.get("from") == "caller"]
    assert said == [f"drag on this wing\n\nInput files, uploaded by the caller:\n- {placed}"]
    assert not (world.tmp / "staging" / up["upload_id"]).exists(), "the staged copy is cleaned up"


def test_refusals_are_words(world):
    with pytest.raises(core.McpError, match="no file at"):
        call(world, "upload_file", filename="nowhere.stl")
    with pytest.raises(core.McpError, match="not an upload id"):
        call(world, "start_study", task="x", upload_ids=["../etc"])
    with pytest.raises(core.McpError, match="not a session id"):
        call(world, "study_status", session_id="nope")
    with pytest.raises(core.McpError, match="no study"):
        call(world, "get_results", study_id="../../etc")
    with pytest.raises(core.McpError, match="depth"):
        call(world, "start_study", task="x", depth="medium")
    world.cfg["gaps"] = ["OPENREYNOLDS_LLM_API_KEY"]
    with pytest.raises(core.McpError, match="missing configuration: OPENREYNOLDS_LLM_API_KEY"):
        call(world, "start_study", task="x")


def test_a_path_outside_the_workspace_is_refused(world):
    world.studies.ready.set()
    sid = call(world, "start_study", task="first")["session_id"]
    study = until(world, sid, "waiting_for_input")["study_id"]
    outside = [
        "/etc/passwd",
        f"/work/{study}/../../etc/passwd",
        # The backslash spelling of the same walk. `Path` splits on it when this runs
        # on Windows, so before #180 this one came back with the file's contents.
        f"/work/{study}\\..\\..\\..\\..\\outside-the-study.txt",
        f"/work/x\\../..\\../outside-the-study.txt",
    ]
    for path in outside:
        with pytest.raises(core.McpError, match="not a workspace path"):
            call(world, "download_url", study_id=study, path=path)
        with pytest.raises(core.McpError, match="not a workspace path"):
            world.host.read_file(study, path)


def test_closing_the_server_ends_every_live_study(world):
    world.studies.ready.set()
    sid = call(world, "start_study", task="first")["session_id"]
    until(world, sid, "waiting_for_input")
    world.host.close(wait_s=10)
    assert world.host.sessions[sid].status == "ended"
    with pytest.raises(core.McpError, match="shutting down"):
        call(world, "start_study", task="again")


def test_an_unpolled_study_ends_itself_at_the_prompt():
    now = [0.0]
    notes = []
    reader = LocalReader(idle_timeout_s=60, on_idle=lambda: notes.append("idle"), clock=lambda: now[0])
    reader.put("hello")
    now[0] = 1000.0
    assert reader.get() == "hello", "a line already waiting is the caller"
    assert reader.get() == "/exit" and notes == ["idle"]
    reader2 = LocalReader(idle_timeout_s=60, clock=lambda: now[0])
    now[0] += 30
    reader2.heartbeat()
    now[0] += 45
    assert reader2.get(timeout=0.05) is None, "a poll 45 s ago keeps it alive"


def test_the_sharing_notice_is_in_the_first_answer_and_that_study_is_shared(world, monkeypatch):
    """Nobody reads this server's stderr, so the notice a sharing machine shows before
    its first trace goes in `start_study`'s answer, with the caller told to show it; the
    study it was shown for is shared like the next one."""
    from openreynolds import sharing

    shown = iter([sharing.NOTICE, None])
    monkeypatch.setattr(sharing, "take_notice", lambda cfg: next(shown))
    first = call(world, "start_study", task="first")
    assert first["sharing_notice"] == sharing.NOTICE
    assert first["next"].startswith("Show the person sharing_notice")
    second = call(world, "start_study", task="second")
    assert "sharing_notice" not in second
    world.studies.ready.set()
    for _ in range(100):
        if len(world.studies.calls) == 2:
            break
        time.sleep(0.05)
    assert [c["sharing"] for c in world.studies.calls] == [True, True]


# -- files sent mid-study, and files that wait for a late workspace (#216) -------------------


def test_files_sent_with_a_message_reach_the_running_study(world):
    world.studies.ready.set()
    sid = call(world, "start_study", task="first")["session_id"]
    first = until(world, sid, "waiting_for_input")
    study = first["study_id"]

    (world.tmp / "flap.stl").write_bytes(b"solid flap")
    up = call(world, "upload_file", filename="flap.stl")
    sent = call(world, "send_message", session_id=sid, text="use this geometry", upload_ids=[up["upload_id"]])
    assert sent["sent"] is True

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        later = call(world, "study_status", session_id=sid, after_seq=first["next_after_seq"], wait_seconds=0)
        if later["state"] == "waiting_for_input" and later["transcript"]:
            break
        time.sleep(0.05)
    files = world.studies.browsers[study].backend.files
    assert files[f"/work/{study}/uploads/flap.stl"] == b"solid flap"
    said = [r["text"] for r in later["transcript"] if r.get("from") == "caller"]
    assert said == [f"use this geometry\n\nInput files, uploaded by the caller:\n- /work/{study}/uploads/flap.stl"]
    assert not (world.tmp / "staging" / up["upload_id"]).exists()

    with pytest.raises(core.McpError, match="not an upload id"):
        call(world, "send_message", session_id=sid, text="x", upload_ids=["../etc"])
    (world.tmp / "again.stl").write_bytes(b"solid")
    again = call(world, "upload_file", filename="again.stl")
    with pytest.raises(core.McpError, match="slash command"):
        call(world, "send_message", session_id=sid, text="/status", upload_ids=[again["upload_id"]])


class DeliveryHost(core.HostDefaults):
    """Just what `Deliveries` reads: a session whose workspace a test moves along."""

    def __init__(self) -> None:
        self.kv = core.MemoryKv()
        self.workspace = core.WORKSPACE_STARTING
        self.said: list[str] = []
        self.placed: list[list[str]] = []
        self.discarded: list[str] = []

    def observe(self, session_id: str):
        session = {"workspace": self.workspace, "instance_id": "i-1", "study_id": "s1", "home": "/work/s1"}
        return {"status": "running"}, {"session": session, "items": []}

    def place_files(self, session_id, instance_id, home, files):
        names = [f"{home}/uploads/{f['name']}" for f in files]
        self.placed.append(names)
        return names

    def say(self, session_id: str, text: str) -> None:
        self.said.append(text)

    def discard_staged(self, files) -> None:
        self.discarded += [f["key"] for f in files]


def test_files_wait_for_a_late_workspace_instead_of_being_dropped(monkeypatch):
    """Before #216 a workspace up after 20 minutes got the task with "upload them again",
    and the staged files were deleted."""
    host = DeliveryHost()
    deliveries = core.Deliveries(host)
    deliveries.record("sess", {"task": "drag", "files": [{"key": "k1", "name": "wing.stl"}],
                               "at": time.time() - core.DELIVERY_WAIT_S - 60})
    assert deliveries.attempt("sess") is False and host.said == [] and host.discarded == []
    host.workspace = core.WORKSPACE_READY
    assert deliveries.attempt("sess") is True
    assert host.said == ["drag\n\nInput files, uploaded by the caller:\n- /work/s1/uploads/wing.stl"]
    assert host.discarded == ["k1"]


def test_a_failed_workspace_or_a_long_wait_says_the_task_without_the_files():
    for workspace, waited in ((core.WORKSPACE_FAILED, 0), (core.WORKSPACE_STARTING, core.DELIVERY_GIVE_UP_S + 1)):
        host = DeliveryHost()
        host.workspace = workspace
        deliveries = core.Deliveries(host)
        deliveries.record("sess", {"task": "drag", "files": [{"key": "k1", "name": "wing.stl"}],
                                   "at": time.time() - waited})
        assert deliveries.attempt("sess") is True
        assert host.said[0].startswith("drag\n\n(The caller also uploaded files") and host.placed == []


def test_a_second_message_with_files_joins_one_still_waiting(monkeypatch):
    host = DeliveryHost()
    deliveries = core.Deliveries(host)
    monkeypatch.setattr(deliveries, "watch", lambda session_id: None)
    deliveries.add("sess", "first", [{"key": "k1", "name": "a.stl"}])
    deliveries.add("sess", "second", [{"key": "k2", "name": "b.stl"}])
    host.workspace = core.WORKSPACE_READY
    assert deliveries.attempt("sess") is True
    assert host.said == ["first\n\nsecond\n\nInput files, uploaded by the caller:\n"
                         "- /work/s1/uploads/a.stl\n- /work/s1/uploads/b.stl"]
    deliveries.add("sess", "third", [{"key": "k3", "name": "c.stl"}])
    assert deliveries.pending("sess")["task"] == "third", "a said one is not added to"


def test_files_on_their_way_do_not_hide_an_open_approval():
    picture = {"session": {"workspace": core.WORKSPACE_READY}, "turn": "user",
               "items": [{"kind": "approval", "state": "open", "request_id": "r1"}]}
    assert core.state_of({"status": "running"}, picture, delivering=True) == "waiting_for_approval"
    picture["items"] = [{"kind": "user", "text": "go"}, {"kind": "assistant", "text": "on it"}]
    assert core.state_of({"status": "running"}, picture, delivering=True) == "working"
    picture["items"] = []
    assert core.state_of({"status": "running"}, picture, delivering=True) == "starting", (
        "the opening task is not in the transcript yet, so the study has not started")


def test_words_without_files_wait_behind_files_still_on_their_way(monkeypatch):
    host = DeliveryHost()
    host.staged = lambda upload_id: core.Staged(key=upload_id, name="a.stl", size=1)
    host.check = lambda session_id: None
    deliveries = core.Deliveries(host)
    monkeypatch.setattr(deliveries, "watch", lambda session_id: None)
    core.send_words(host, deliveries, "sess", "use this", ["a" * 32])
    core.send_words(host, deliveries, "sess", "and make it fast")
    core.send_words(host, deliveries, "sess", "/status")
    assert host.said == ["/status"], "a slash command is not queued; words are"
    host.workspace = core.WORKSPACE_READY
    assert deliveries.attempt("sess") is True
    assert host.said[1] == ("use this\n\nand make it fast\n\nInput files, uploaded by the caller:\n"
                            "- /work/s1/uploads/a.stl")


def test_a_status_call_never_places_files_itself(monkeypatch):
    host = DeliveryHost()
    host.workspace = core.WORKSPACE_READY
    deliveries = core.Deliveries(host)
    watched = []
    monkeypatch.setattr(deliveries, "watch", watched.append)
    deliveries.record("sess", {"task": "drag", "files": [{"key": "k", "name": "w.zip"}], "at": time.time()})
    _, _, state = core.look_once(host, deliveries, "sess")
    assert host.placed == [] and watched == ["sess"] and state == "starting"


def test_a_file_that_fails_is_not_placed_twice_and_is_given_up_on(monkeypatch):
    monkeypatch.setattr(core, "PLACE_GIVE_UP_S", 0)
    host = DeliveryHost()
    host.workspace = core.WORKSPACE_READY
    calls = []

    def place(session_id, instance_id, home, files):
        calls.append(files[0]["name"])
        if files[0]["name"] == "bad.stl":
            raise OSError("disk full")
        return [f"{home}/uploads/{files[0]['name']}"]
    host.place_files = place
    deliveries = core.Deliveries(host)
    deliveries.record("sess", {"task": "drag", "at": time.time(),
                               "files": [{"key": "k1", "name": "good.stl"}, {"key": "k2", "name": "bad.stl"}]})
    for _ in range(core.PLACE_TRIES - 1):
        assert deliveries.attempt("sess") is False
    assert deliveries.attempt("sess") is False          # the last try gives the file up
    assert deliveries.attempt("sess") is True
    assert calls == ["good.stl"] + ["bad.stl"] * core.PLACE_TRIES, "good.stl was placed once"
    assert host.said == ["drag\n\nInput files, uploaded by the caller:\n- /work/s1/uploads/good.stl\n\n"
                         "(These uploaded files could not be placed on the workspace: bad.stl; ask the "
                         "caller to upload them again with upload_file.)"]
    assert sorted(host.discarded) == ["k1", "k2"]


def test_words_sent_during_a_long_placement_do_not_wait_for_it():
    host = DeliveryHost()
    host.workspace = core.WORKSPACE_READY
    deliveries = core.Deliveries(host)
    gate, inside = threading.Event(), threading.Event()

    def slow(session_id, instance_id, home, files):
        inside.set()
        gate.wait(10)
        return [f"{home}/uploads/{files[0]['name']}"]
    host.place_files = slow
    deliveries.record("sess", {"task": "first", "at": time.time(), "files": [{"key": "k1", "name": "a.zip"}]})
    worker = threading.Thread(target=deliveries.attempt, args=("sess",))
    worker.start()
    assert inside.wait(5)
    began = time.monotonic()
    deliveries.add("sess", "second", [])
    deliveries.add("sess", "third", [{"key": "k2", "name": "b.stl"}])
    assert time.monotonic() - began < 1, "add never waits on a placement"
    gate.set()
    worker.join(10)
    assert host.said == ["first\n\nsecond\n\nthird\n\nInput files, uploaded by the caller:\n"
                         "- /work/s1/uploads/a.zip\n- /work/s1/uploads/b.stl"]


def test_forgetting_an_answered_message_never_forgets_a_new_one():
    """`unseen` clears the record of words already answered. Words `add`ed between its
    read and its clear used to be cleared with it, files and all."""
    host = DeliveryHost()
    deliveries = core.Deliveries(host)
    deliveries.watch = lambda session_id: None
    host.kv.set(deliveries._key("sess"), json.dumps({"said": "old", "at": time.time(), "seq": 0}))
    real_pending = deliveries.pending
    raced = []

    def pending(session_id):
        answer = real_pending(session_id)
        if not raced and answer and "said" in answer:
            raced.append(1)
            deliveries.add("sess", "new", [{"key": "k9", "name": "n.stl"}])
        return answer
    deliveries.pending = pending
    assert deliveries.unseen("sess", {"items": [{"seq": 5}]}) is False
    assert real_pending("sess")["task"] == "new"


def test_a_short_outage_does_not_give_a_file_up():
    host = DeliveryHost()
    host.workspace = core.WORKSPACE_READY
    down = [True]

    def place(session_id, instance_id, home, files):
        if down[0]:
            raise OSError("502 from the workspace")
        return [f"{home}/uploads/{files[0]['name']}"]
    host.place_files = place
    deliveries = core.Deliveries(host)
    deliveries.record("sess", {"task": "drag", "at": time.time(), "files": [{"key": "k1", "name": "a.stl"}]})
    for _ in range(core.PLACE_TRIES * 3):
        assert deliveries.attempt("sess") is False
    down[0] = False
    assert deliveries.attempt("sess") is True
    assert host.said == ["drag\n\nInput files, uploaded by the caller:\n- /work/s1/uploads/a.stl"]


def test_an_upload_named_twice_is_placed_once():
    host = DeliveryHost()
    host.staged = lambda upload_id: core.Staged(key=upload_id, name="a.stl", size=1)
    files = core.staged_files(host, ["a" * 32, "a" * 32])
    assert len(files) == 1
    deliveries = core.Deliveries(host)
    deliveries.watch = lambda session_id: None
    deliveries.add("sess", "x", files + files)
    assert len(deliveries.pending("sess")["files"]) == 1


def test_a_lapsed_place_lock_is_neither_refreshed_nor_released_by_its_old_holder():
    host = DeliveryHost()
    deliveries = core.Deliveries(host)
    mine = deliveries._lock("sess")
    assert mine and deliveries._lock("sess") is None
    host.kv.delete(deliveries._key("sess") + ":lock")          # it lapsed
    theirs = deliveries._lock("sess")
    assert deliveries._refresh("sess", mine) is False
    deliveries._unlock("sess", mine)
    assert host.kv.get(deliveries._key("sess") + ":lock") == theirs
