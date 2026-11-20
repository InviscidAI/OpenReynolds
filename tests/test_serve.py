"""`openreynolds serve`: the capabilities over HTTP, on this machine.

The third face over `openreynolds/mcp/core.py`'s `Host`, so it is tested against the
same fakes as the first: `LocalHost` running the stand-in session in
`tests/test_mcp_local.py`. Nothing here reaches a network beyond a loopback socket this
file opens itself, and the two faces are asked for the same study's state side by side,
because the whole reason the capabilities live in `core` is that they must not differ.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_mcp_local as mcp_face
from openreynolds import serve
from openreynolds.mcp import core
from openreynolds.mcp.local import LocalHost

TOKEN = "a-token-for-this-test"


@pytest.fixture
def server(tmp_path: Path):
    root = tmp_path / "studies"
    root.mkdir()
    studies = mcp_face.Studies(root)
    host = LocalHost(load_config=lambda: mcp_face.Cfg(root), run_session=studies,
                     local_workspace=lambda: None, staging_dir=tmp_path / "staging",
                     cwd=tmp_path)
    listener = serve.build(host, token=TOKEN, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=listener.serve_forever, kwargs={"poll_interval": 0.02},
                              daemon=True, name="serve-test")
    thread.start()
    yield SimpleNamespace(listener=listener, thread=thread, host=host, studies=studies,
                          port=listener.server_port, root=root, tmp=tmp_path)
    listener.shutdown()
    listener.server_close()
    thread.join(10)
    studies.ready.set()
    host.close(wait_s=10)


def ask(server: SimpleNamespace, method: str, path: str, body: dict | None = None,
        token: str | None = TOKEN, timeout: float = 60.0,
        raw_body: bytes | None = None) -> SimpleNamespace:
    """One request, answered as (status, body). An error status is an answer here, not
    an exception: a refusal is what several of these tests are about."""
    data = raw_body if raw_body is not None else (json.dumps(body).encode()
                                                  if body is not None else None)
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=data,
                                     method=method)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310 - loopback
            raw, status, headers = answer.read(), answer.status, dict(answer.headers)
    except urllib.error.HTTPError as exc:
        raw, status, headers = exc.read(), exc.code, dict(exc.headers)
    body_json = json.loads(raw) if headers.get("Content-Type") == "application/json" else None
    return SimpleNamespace(status=status, raw=raw, json=body_json, headers=headers)


def until(server: SimpleNamespace, session_id: str, want: str, seconds: float = 10.0) -> dict:
    deadline = time.monotonic() + seconds
    while True:
        answer = ask(server, "GET", f"/sessions/{session_id}").json
        if answer["state"] == want or time.monotonic() > deadline:
            return answer
        time.sleep(0.05)


# -- the token -----------------------------------------------------------------------


def test_a_request_with_no_token_or_the_wrong_one_is_refused(server, monkeypatch):
    """This process starts compute that costs money. An unauthenticated listener that
    can do that is a defect, so there is no unauthenticated route at all."""
    for token in (None, "not-the-token", ""):
        refused = ask(server, "POST", "/studies", {"task": "spend my money"}, token=token)
        assert refused.status == 401, token
        assert "token is required" in refused.json["error"]
    assert server.studies.calls == [], "nothing was started"
    assert ask(server, "GET", "/").status == 200
    monkeypatch.delenv(serve.TOKEN_ENV, raising=False)
    assert serve.token_for(None) != serve.token_for(None), "a generated token is fresh"
    assert serve.token_for("mine") == "mine"
    monkeypatch.setenv(serve.TOKEN_ENV, "from-the-environment")
    assert serve.token_for(None) == "from-the-environment"


def test_the_index_says_what_this_is_and_lists_every_route(server):
    index = ask(server, "GET", "/").json
    assert "not the hosted" in index["about"].lower()
    assert index["states"] == list(core.STATES)
    assert {row["route"] for row in index["routes"]} == set(serve.ROUTES)
    assert {row["capability"] for row in index["routes"]} == {
        name for name in core.DESCRIPTIONS if ":" not in name
    }, "the nine capabilities, and nothing invented here"
    for row in index["routes"]:
        assert row["what"] == core.description(server.host, row["capability"])
    assert ask(server, "GET", "/nowhere").status == 404


# -- a study, over HTTP ---------------------------------------------------------------


def test_a_study_runs_to_an_answer_over_http(server):
    server.studies.ready.set()
    started = ask(server, "POST", "/studies", {"task": "flow past a cylinder at Re=100"})
    assert started.status == 201
    session_id = started.json["session_id"]
    assert started.json["state"] == "starting"

    answered = until(server, session_id, "waiting_for_input")
    assert answered["state"] == "waiting_for_input"
    assert answered["last_message"] == "done: flow past a cylinder at Re=100"
    assert answered["next"] == core.NEXT["waiting_for_input"]
    study_id = answered["study_id"]

    sent = ask(server, "POST", f"/sessions/{session_id}/messages", {"text": "now at Re=200"})
    assert sent.json == {"sent": True, "next": "Read the session to see the answer."}
    said: list[str] = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(said) < 2:
        rows = ask(server, "GET", f"/sessions/{session_id}?after_seq={answered['next_after_seq']}")
        said = [row.get("text") for row in rows.json["transcript"]]
    assert said == ["now at Re=200", "done: now at Re=200"]

    results = ask(server, "GET", f"/studies/{study_id}/results").json
    assert results["results_json"] == {"drag_N": 1.5}
    assert {"path": f"/work/{study_id}/results.json", "size": 15} in results["files"]

    got = ask(server, "GET", f"/studies/{study_id}/file?path=/work/{study_id}/results.json")
    assert got.raw == b'{"drag_N": 1.5}'
    assert got.headers["Content-Type"] == "application/octet-stream"
    assert "results.json" in got.headers["Content-Disposition"]

    listed = ask(server, "GET", "/studies").json
    assert [card["study_id"] for card in listed["studies"]] == [study_id]

    assert ask(server, "DELETE", f"/sessions/{session_id}").json == {"status": "ending"}
    assert until(server, session_id, "ended")["state"] == "ended"


def test_an_approval_is_asked_and_answered_over_http(server):
    server.studies.ready.set()
    session_id = ask(server, "POST", "/studies",
                     {"task": "ask me", "mode": "partial"}).json["session_id"]
    waiting = until(server, session_id, "waiting_for_approval")
    assert waiting["approvals"][0]["title"] == "Run the solve?"
    assert waiting["next"] == core.NEXT["waiting_for_approval"]
    answer = ask(server, "POST", f"/sessions/{session_id}/approvals", {"decision": "approve"})
    assert answer.json["sent"] == "/yes"
    assert until(server, session_id, "waiting_for_input")["last_message"] == "approval: /yes"
    assert "decision is" in ask(server, "POST", f"/sessions/{session_id}/approvals",
                                {"decision": "maybe"}).json["error"]


def test_an_uploaded_file_reaches_the_study_before_the_task(server):
    server.studies.ready.set()
    geometry = server.tmp / "wing section.stl"
    geometry.write_bytes(b"solid wing\nendsolid wing\n")
    staged = ask(server, "POST", "/uploads", {"path": "wing section.stl"}).json
    assert staged["name"] == "wing_section.stl"

    session_id = ask(server, "POST", "/studies",
                     {"task": "drag on this wing",
                      "upload_ids": [staged["upload_id"]]}).json["session_id"]
    answered = until(server, session_id, "waiting_for_input")
    study_id = answered["study_id"]
    placed = f"/work/{study_id}/uploads/wing_section.stl"
    assert server.studies.browsers[study_id].backend.files[placed] == geometry.read_bytes()
    said = [row["text"] for row in answered["transcript"] if row.get("from") == "caller"]
    assert said == [f"drag on this wing\n\nInput files, uploaded by the caller:\n- {placed}"]
    assert "not an upload id" in ask(server, "POST", "/studies",
                                     {"task": "x", "upload_ids": ["../etc"]}).json["error"]
    assert "is a list" in ask(server, "POST", "/studies",
                              {"task": "x", "upload_ids": "one"}).json["error"]
    assert "no file at" in ask(server, "POST", "/uploads", {"path": "nowhere.stl"}).json["error"]


def test_the_http_face_answers_the_same_states_as_the_tools(server):
    """The whole point of `core`: two faces over one Host cannot disagree about what a
    study is doing. Asked of the same session, in the same situation, side by side."""
    pytest.importorskip("mcp")
    tools = SimpleNamespace(server=core.build_server(server.host))
    session_id = ask(server, "POST", "/studies", {"task": "first"}).json["session_id"]

    over_http = ask(server, "GET", f"/sessions/{session_id}").json
    by_tool = mcp_face.call(tools, "study_status", session_id=session_id, wait_seconds=0)
    assert over_http["state"] == by_tool["state"] == "starting"
    assert set(over_http) == set(by_tool)

    server.studies.ready.set()
    assert until(server, session_id, "waiting_for_input")["state"] == "waiting_for_input"
    over_http = ask(server, "GET", f"/sessions/{session_id}").json
    by_tool = mcp_face.call(tools, "study_status", session_id=session_id, wait_seconds=0)
    for field in ("state", "next", "last_message", "study_id"):
        assert over_http[field] == by_tool[field], field
    assert over_http["state"] in core.STATES


# -- the refusals ---------------------------------------------------------------------


def test_a_file_outside_the_study_directory_is_refused(server):
    server.studies.ready.set()
    session_id = ask(server, "POST", "/studies", {"task": "first"}).json["session_id"]
    study_id = until(server, session_id, "waiting_for_input")["study_id"]
    for path in ("/etc/passwd", f"/work/{study_id}/../../etc/passwd",
                 "/work/x\\..\\..\\..\\..\\outside-the-study.txt"):  # #180
        refused = ask(server, "GET", f"/studies/{study_id}/file?path={path}")
        assert refused.status == 400, path
        assert "not a workspace path" in refused.json["error"]
    assert ask(server, "GET", f"/studies/{study_id}/file").status == 400, "no path at all"
    assert ask(server, "GET", "/studies/..%2F..%2Fetc/results").status == 400


def test_a_request_the_capabilities_refuse_is_four_hundred_in_their_own_words(server):
    assert "say what to simulate" in ask(server, "POST", "/studies", {"task": " "}).json["error"]
    assert "not a field" in ask(server, "POST", "/studies",
                                {"task": "x", "depht": "quick"}).json["error"]
    assert "depth" in ask(server, "POST", "/studies",
                          {"task": "x", "depth": "medium"}).json["error"]
    assert "not a session id" in ask(server, "GET", "/sessions/nope").json["error"]
    assert "nothing to say" in ask(server, "POST", "/sessions/nope/messages",
                                   {"text": ""}).json["error"]
    assert "not JSON" in ask(server, "POST", "/studies", raw_body=b"{oops").json["error"]
    assert "one JSON object" in ask(server, "POST", "/studies", raw_body=b"[1]").json["error"]
    assert "not a number" in ask(server, "GET", "/sessions/nope?wait=soon").json["error"]


# -- the long poll and the close-down -------------------------------------------------


def test_a_long_poll_returns_early_when_something_happens(server):
    session_id = ask(server, "POST", "/studies", {"task": "first"}).json["session_id"]
    threading.Timer(0.2, server.studies.ready.set).start()
    began = time.monotonic()
    answer = ask(server, "GET", f"/sessions/{session_id}?wait=40")
    took = time.monotonic() - began
    assert answer.json["state"] == "waiting_for_input", answer.json
    assert took < 20, f"held the call for {took:.1f}s with the answer in hand"


def test_a_long_poll_comes_back_on_time_when_nothing_does(server):
    """The workspace never comes up here, so the study stays `starting` and the call is
    held for as long as it asked for and not a request longer."""
    session_id = ask(server, "POST", "/studies", {"task": "first"}).json["session_id"]
    began = time.monotonic()
    answer = ask(server, "GET", f"/sessions/{session_id}?wait=1")
    took = time.monotonic() - began
    assert answer.json["state"] == "starting"
    assert 0.9 <= took < 15, f"asked for 1s, took {took:.1f}s"


def test_shutting_the_server_down_ends_every_live_study(server):
    """`serve`'s own close-down, which is `mcp/local.py`'s: jobs stopped, workspace
    stopped, files kept, and `start` with the study_id carries on later."""
    server.studies.ready.set()
    session_id = ask(server, "POST", "/studies", {"task": "first"}).json["session_id"]
    until(server, session_id, "waiting_for_input")

    server.listener.shutdown()
    server.thread.join(10)
    assert not server.thread.is_alive()
    server.host.close(wait_s=10)
    assert server.host.sessions[session_id].status == "ended"
    with pytest.raises(core.McpError, match="shutting down"):
        core.begin_study(server.host, core.deliveries_for(server.host), "again")
