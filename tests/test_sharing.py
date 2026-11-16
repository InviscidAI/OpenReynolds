"""Sharing (openreynolds/sharing.py): each rule the privacy policy's "Traces from
OpenReynolds" sets for this machine, against a temporary config folder and a scripted
service. No test here reaches a network."""

from __future__ import annotations

import gzip
import json
import os
import stat
from types import SimpleNamespace

import httpx2 as httpx
import pytest

from openreynolds import sharing
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.redact import MARK

from conftest import install_model as install, message, text_block, tool_block

KEY = "sk-ant-api03-" + "k" * 40
SERVICE = "of_live_" + "s" * 40


@pytest.fixture
def on(monkeypatch):
    """Sharing as a signed-in machine would have it: the suite's off switch lifted."""
    monkeypatch.delenv("OPENREYNOLDS_SHARING", raising=False)
    return Config(llm_api_key=KEY, foamd_api_key=SERVICE, foamd_url="https://api.example.test")


@pytest.fixture
def seen(on):
    """A machine that has already shown this version's notice."""
    sharing.save_state(sharing.State(enabled=True, notice=sharing.DISCLOSURE_VERSION))
    return on


def _turn(text="done", thinking="", tokens=None, stop="end_turn"):
    content = ([SimpleNamespace(type="thinking", thinking=thinking)] if thinking else []) + [text_block(text)]
    return SimpleNamespace(content=content, model="claude-opus-5-5", provider="anthropic",
                           stop_reason=stop, tokens=tokens or {"input": 10, "output": 5})


# -- who shares ------------------------------------------------------------------------


def test_a_machine_not_signed_in_records_nothing_and_drops_its_queue(on):
    sharing.enqueue("old", b"x")
    cfg = Config(llm_api_key=KEY)
    assert sharing.begin(cfg, "s1") == (None, None)
    assert sharing.queued() == []


def test_the_first_study_shows_the_notice_and_is_not_recorded(on):
    recorder, notice = sharing.begin(on, "s1")
    assert recorder is None and notice == sharing.NOTICE
    assert "openreynolds sharing off" in notice and sharing.PRIVACY_URL in notice
    again, none = sharing.begin(on, "s2")
    assert none is None and isinstance(again, sharing.Recorder)


def test_a_newer_notice_is_shown_again(seen, monkeypatch):
    monkeypatch.setattr(sharing, "DISCLOSURE_VERSION", sharing.DISCLOSURE_VERSION + 1)
    recorder, notice = sharing.begin(seen, "s1")
    assert recorder is None and notice


@pytest.mark.parametrize("how", ["env", "switch"])
def test_off_records_nothing(seen, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("OPENREYNOLDS_SHARING", "off")
    else:
        sharing.set_enabled(False)
    assert sharing.begin(seen, "s1") == (None, None)


def test_turning_it_off_discards_the_queue(seen):
    sharing.enqueue("a" * 32, b"x")
    assert len(sharing.queued()) == 1
    sharing.set_enabled(False)
    assert sharing.queued() == [] and sharing.load_state().enabled is False


def test_only_openreynolds_own_interfaces_share_unless_asked():
    """The hosted web app runs sessions through `interface`; its studies are the
    service's, so a session with an interface shares only when told to."""
    import inspect

    from openreynolds import cli

    source = inspect.getsource(cli.session)
    assert "shares = (interface is None) if sharing is None else bool(sharing)" in source


# -- what a trace holds ---------------------------------------------------------------


def test_credentials_are_removed_by_value_and_by_shape(seen):
    recorder = sharing.Recorder(seen, "s1", environ={"MY_SERVICE_TOKEN": "tok-abcdefgh1234"})
    recorder.tool("bash", {"cmd": f"curl -H 'x: {KEY}' https://x"},
                  f"used {SERVICE} and tok-abcdefgh1234 and AKIAABCDEFGHIJKLMNOP", False, 0.2)
    recorder.message("user", f"my key is {KEY}")
    dumped = json.dumps(recorder.records)
    for secret in (KEY, SERVICE, "tok-abcdefgh1234", "AKIAABCDEFGHIJKLMNOP"):
        assert secret not in dumped
    assert MARK in dumped


def test_usage_records_carry_no_text(seen):
    recorder = sharing.Recorder(seen, "s1")
    recorder.turn(_turn("the answer is 42", thinking="let me think"), 1.5)
    recorder.tool("bash", {"cmd": "blockMesh"}, "Mesh OK", False, 0.1)
    usage = [r for r in recorder.records if r["kind"] in sharing.USAGE_KINDS]
    dumped = json.dumps(usage)
    assert "42" not in dumped and "think" not in dumped and "blockMesh" not in dumped
    turn = next(r for r in usage if r["kind"] == "turn")
    assert turn["input_tokens"] == 10 and turn["output_tokens"] == 5 and turn["seconds"] == 1.5


def test_reasoning_is_kept_as_the_provider_returned_it(seen):
    recorder = sharing.Recorder(seen, "s1")
    recorder.turn(_turn(thinking="blockMesh first, then checkMesh"), 1.0)
    assert {"kind": "reasoning", "at": recorder.records[-1]["at"],
            "text": "blockMesh first, then checkMesh"} in recorder.records


def test_a_long_output_is_cut_and_marked(seen):
    recorder = sharing.Recorder(seen, "s1")
    recorder.tool("read_file", {"path": "/work/log"}, "x" * (sharing.CONTENT_AT_MOST + 10), False)
    tool, io = recorder.records
    assert tool["truncated"] is True and tool["output_chars"] == sharing.CONTENT_AT_MOST + 10
    assert len(io["output"]) == sharing.CONTENT_AT_MOST and io["truncated"] is True


def test_a_failing_recorder_never_fails_the_caller_and_sends_nothing(seen):
    recorder = sharing.Recorder(seen, "s1")
    recorder.turn(object(), 1.0)          # no content, no tokens: fine
    recorder.turn(SimpleNamespace(tokens={"input": "not a number"}), 1.0)
    assert recorder.broken is True
    recorder.tool("bash", {}, "ok", False)  # ignored, and does not raise
    assert recorder.finish() is None


def test_a_study_without_a_model_turn_queues_nothing(seen):
    assert sharing.Recorder(seen, "s1").finish() is None


def test_the_queue_is_private(seen):
    recorder = sharing.Recorder(seen, "s1")
    recorder.turn(_turn(), 1.0)
    path = recorder.finish()
    assert path is not None and path.parent == sharing.queue_dir()
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    body = json.loads(gzip.decompress(path.read_bytes()))
    assert body["disclosure"] == sharing.DISCLOSURE_VERSION and body["study_id"] == "s1"
    assert body["records"][0]["kind"] == "header" and body["records"][0]["turns"] == 1


def test_a_trace_too_large_goes_as_its_usage(seen, monkeypatch):
    monkeypatch.setattr(sharing, "SEND_AT_MOST", 10)
    recorder = sharing.Recorder(seen, "s1")
    recorder.turn(_turn(), 1.0)
    recorder.message("user", "mesh it")
    body = json.loads(gzip.decompress(recorder.finish().read_bytes()))
    assert body["parts"] == "usage"
    assert {r["kind"] for r in body["records"]} <= sharing.USAGE_KINDS


def test_the_loop_tells_the_recorder_everything(seen, ctx, store, view, backend):
    backend.files["/work/a"] = b"one"
    loop = Loop(Config(llm_api_key="test-key", model="claude-opus-5"), ctx, store, view)
    loop.recorder = sharing.Recorder(seen, store.session.study_id)
    install(loop, [
        message([tool_block("read_file", {"path": "/work/a"}, "tu_a")], stop_reason="tool_use"),
        message([text_block("it says one")]),
    ])
    loop.say("read a")
    loop.run()
    kinds = [r["kind"] for r in loop.recorder.records]
    assert kinds.count("turn") == 2 and "tool" in kinds and "tool_io" in kinds
    said = [r["text"] for r in loop.recorder.records if r["kind"] == "message"]
    assert "read a" in said and "it says one" in said
    tool = next(r for r in loop.recorder.records if r["kind"] == "tool")
    assert tool["tool"] == "read_file" and tool["ok"] is True and tool["seconds"] is not None


# -- sending -----------------------------------------------------------------------------


class Service:
    """The two routes a client calls, scripted."""

    def __init__(self, switches=None, post_status=201, post_error=""):
        self.switches = {"share_usage": True, "share_content": True,
                         "disclosure": sharing.DISCLOSURE_VERSION, **(switches or {})}
        self.post_status = post_status
        self.post_error = post_error
        self.posted: list[dict] = []
        self.get_status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/account/sharing":
            return httpx.Response(self.get_status, json=self.switches)
        body = json.loads(gzip.decompress(request.content))
        self.posted.append(body)
        if self.post_status >= 400:
            return httpx.Response(self.post_status, json={"error": self.post_error, "message": "no"})
        return httpx.Response(self.post_status, json={"trace_id": "t", "parts": body["parts"]})


@pytest.fixture
def service(monkeypatch):
    holder = {}

    def client(cfg, timeout):
        return httpx.Client(base_url=cfg.foamd_url, transport=httpx.MockTransport(holder["service"]))

    monkeypatch.setattr(sharing, "_client", client)

    def use(**kw):
        holder["service"] = Service(**kw)
        return holder["service"]

    return use


def _queue_one(cfg):
    recorder = sharing.Recorder(cfg, "s1")
    recorder.turn(_turn(), 1.0)
    recorder.message("user", "mesh the cavity")
    return recorder.finish()


def test_a_queued_trace_is_sent_and_removed(seen, service):
    svc = service()
    _queue_one(seen)
    assert sharing.send(seen)["sent"] == 1
    assert sharing.queued() == [] and svc.posted[0]["parts"] == "usage+content"


def test_content_off_sends_the_usage_part_only(seen, service):
    svc = service(switches={"share_content": False})
    _queue_one(seen)
    sharing.send(seen)
    (body,) = svc.posted
    assert body["parts"] == "usage"
    assert {r["kind"] for r in body["records"]} <= sharing.USAGE_KINDS


@pytest.mark.parametrize("switches", [{"share_usage": False},
                                      {"disclosure": sharing.DISCLOSURE_VERSION + 1}])
def test_usage_off_or_a_newer_notice_sends_nothing_and_empties_the_queue(seen, service, switches):
    svc = service(switches=switches)
    _queue_one(seen)
    sharing.send(seen)
    assert svc.posted == [] and sharing.queued() == []


@pytest.mark.parametrize("status,error,kept", [(500, "", True), (410, "deleted_since", False),
                                                (403, "sharing_off", False),
                                                (409, "disclosure_outdated", False)])
def test_what_a_refusal_does_to_the_queue(seen, service, status, error, kept):
    service(post_status=status, post_error=error)
    _queue_one(seen)
    sharing.send(seen)
    assert bool(sharing.queued()) is kept


def test_a_revoked_key_empties_the_queue(seen, service):
    svc = service()
    svc.get_status = 401
    _queue_one(seen)
    sharing.send(seen)
    assert sharing.queued() == [] and svc.posted == []


def test_offline_keeps_the_queue(seen, monkeypatch):
    def client(cfg, timeout):
        def down(request):
            raise httpx.ConnectError("offline")
        return httpx.Client(base_url=cfg.foamd_url, transport=httpx.MockTransport(down))

    monkeypatch.setattr(sharing, "_client", client)
    _queue_one(seen)
    assert sharing.send(seen)["left"] == 1


def test_status_says_what_this_machine_and_the_account_do(seen, service):
    service(switches={"share_content": False, "traces": 3})
    lines = "\n".join(sharing.describe(seen))
    assert "Sharing is on on this machine." in lines
    assert "Share usage on, Share study content off (3 traces shared)" in lines


def test_the_cli_switch(on, monkeypatch):
    from click.testing import CliRunner

    from openreynolds.cli import main

    monkeypatch.setattr(sharing, "account_switches", lambda cfg, timeout=10.0: None)
    runner = CliRunner()
    off = runner.invoke(main, ["sharing", "off"])
    assert off.exit_code == 0 and sharing.load_state().enabled is False
    back = runner.invoke(main, ["sharing", "on"])
    assert back.exit_code == 0 and sharing.load_state().enabled is True
