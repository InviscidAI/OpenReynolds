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
    sharing.enqueue(sharing.fingerprint(on), "old", b"x")
    cfg = Config(llm_api_key=KEY)
    assert sharing.begin(cfg, "s1") == (None, None)
    assert sharing.queued() == []


def test_the_first_study_shows_the_notice_and_is_recorded(on):
    recorder, notice = sharing.begin(on, "s1")
    assert isinstance(recorder, sharing.Recorder) and notice == sharing.NOTICE
    assert "openreynolds sharing off" in notice and sharing.PRIVACY_URL in notice
    again, none = sharing.begin(on, "s2")
    assert none is None and isinstance(again, sharing.Recorder)


def test_sharing_off_during_the_first_study_shares_nothing(on):
    """The notice is shown at the start; turning sharing off before the study ends
    keeps its trace from being queued."""
    recorder, notice = sharing.begin(on, "s1")
    assert notice
    recorder.message("user", "hello")
    recorder.turn(_turn(), 1.0)
    sharing.set_enabled(False)
    assert recorder.finish() is None
    assert sharing.queued() == []


def test_a_newer_notice_is_shown_again(seen, monkeypatch):
    monkeypatch.setattr(sharing, "DISCLOSURE_VERSION", sharing.DISCLOSURE_VERSION + 1)
    recorder, notice = sharing.begin(seen, "s1")
    assert recorder is not None and notice == sharing.NOTICE


@pytest.mark.parametrize("how", ["env", "switch"])
def test_off_records_nothing(seen, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("OPENREYNOLDS_SHARING", "off")
    else:
        sharing.set_enabled(False)
    assert sharing.begin(seen, "s1") == (None, None)


def test_turning_it_off_discards_the_queue(seen):
    sharing.enqueue(sharing.fingerprint(seen), "a" * 32, b"x")
    assert len(sharing.queued()) == 1
    sharing.set_enabled(False)
    assert sharing.queued() == [] and sharing.load_state().enabled is False


def test_only_openreynolds_own_interfaces_share_unless_asked():
    """An embedder's `interface` shares only when told to: the MCP server says True, and
    the hosted web app says "web"."""
    import inspect

    from openreynolds import cli

    source = inspect.getsource(cli.session)
    assert "shares = (interface is None) if sharing is None else bool(sharing)" in source
    assert 'trace_source = "web" if sharing == "web" else "openreynolds"' in source


def test_a_web_study_is_recorded_without_the_notice_and_says_so(on):
    """The hosted app is the service: its studies always leave a trace, and it never
    shows (or uses up) the machine's notice, whatever the machine's switch says."""
    sharing.set_enabled(False)
    recorder, notice = sharing.begin(on, "s1", source="web")
    assert notice is None and sharing.load_state().notice == 0
    recorder.turn(_turn(), 1.0)
    assert recorder.body()["source"] == "web"
    sharing.set_enabled(True)
    terminal, _ = sharing.begin(on, "s2")
    assert terminal.body()["source"] == "openreynolds"


def test_a_web_trace_is_sent_from_memory_and_never_queued(on, service):
    """One runner process holds many people's sessions: a queue there could send one
    person's trace with another's key, so a web trace never touches the queue, and a web
    session never sends what the queue holds."""
    svc = service()
    _queue_one(on)
    left = sharing.queued()
    recorder, _ = sharing.begin(on, "s1", source="web")
    recorder.turn(_turn(), 1.0)
    assert sharing.send_now(on, recorder) is True
    assert [b["source"] for b in svc.posted] == ["web"]
    assert sharing.queued() == left


def test_the_notice_is_one_line():
    assert "\n" not in sharing.NOTICE and len(sharing.NOTICE) < 200
    assert "openreynolds sharing off" in sharing.NOTICE and sharing.PRIVACY_URL in sharing.NOTICE


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
    recorder.turn(_turn("the answer is forty-two", thinking="let me think"), 1.5)
    recorder.tool("bash", {"cmd": "blockMesh"}, "Mesh OK", False, 0.1)
    usage = [r for r in recorder.records if r["kind"] in sharing.USAGE_KINDS]
    dumped = json.dumps(usage)
    assert "forty-two" not in dumped and "think" not in dumped and "blockMesh" not in dumped
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

    def __init__(self, post_status=201, post_error=""):
        self.post_status = post_status
        self.post_error = post_error
        self.posted: list[dict] = []
        self.paths: list[str] = []
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.requests.append(request)
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
    assert svc.paths == ["/v1/traces"], "there are no account switches to ask first"


@pytest.mark.parametrize("status,error,kept", [(500, "", True), (410, "deleted_since", False),
                                                (409, "disclosure_outdated", False)])
def test_what_a_refusal_does_to_the_queue(seen, service, status, error, kept):
    service(post_status=status, post_error=error)
    _queue_one(seen)
    sharing.send(seen)
    assert bool(sharing.queued()) is kept


def test_a_revoked_key_empties_the_queue(seen, service):
    service(post_status=401, post_error="unauthorized")
    _queue_one(seen)
    _queue_one(seen)
    sharing.send(seen)
    assert sharing.queued() == []


def test_offline_keeps_the_queue(seen, monkeypatch):
    def client(cfg, timeout):
        def down(request):
            raise httpx.ConnectError("offline")
        return httpx.Client(base_url=cfg.foamd_url, transport=httpx.MockTransport(down))

    monkeypatch.setattr(sharing, "_client", client)
    _queue_one(seen)
    assert sharing.send(seen)["left"] == 1


def test_status_says_what_this_machine_does(seen):
    lines = "\n".join(sharing.describe(seen))
    assert "Sharing is on on this machine." in lines and "Account page" in lines


def test_the_cli_switch(on, monkeypatch):
    from click.testing import CliRunner

    from openreynolds.cli import main

    runner = CliRunner()
    off = runner.invoke(main, ["sharing", "off"])
    assert off.exit_code == 0 and sharing.load_state().enabled is False
    back = runner.invoke(main, ["sharing", "on"])
    assert back.exit_code == 0 and sharing.load_state().enabled is True


def test_signing_in_says_it_shares_in_one_line_once(on, monkeypatch, tmp_path):
    """The notice is said at sign-in, so the first study is not interrupted by it."""
    from click.testing import CliRunner

    from openreynolds import cli

    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
    monkeypatch.setattr(cli.hosted, "auth_config", lambda url: {"supabase_url": "https://sb.example", "publishable_key": "pk"})
    monkeypatch.setattr(cli.hosted, "password_session", lambda *a: {"access_token": "jwt"})
    monkeypatch.setattr(cli.hosted, "accept_terms", lambda url, jwt: {})
    monkeypatch.setattr(cli.hosted, "mint_key", lambda url, jwt, name: {"key": "of_live_x", "name": name})

    result = CliRunner().invoke(cli.main, ["login"], input="a@example.com\npw123456\n")
    assert result.exit_code == 0, result.output
    assert sharing.NOTICE in result.output
    assert sharing.load_state().notice == sharing.DISCLOSURE_VERSION
    recorder, notice = sharing.begin(on, "s1")
    assert recorder is not None and notice is None


# -- the review's fixes ---------------------------------------------------------------


def test_a_queued_trace_goes_only_to_the_account_it_was_recorded_under(seen, service):
    """Signed in as someone else, or pointed at another service: what the machine
    queued before is never sent under the new key. It is left for a session that still
    holds the old one, and dropped once it is a week old."""
    svc = service()
    path = _queue_one(seen)
    other = Config(llm_api_key=KEY, foamd_api_key="of_live_" + "o" * 40, foamd_url=seen.foamd_url)
    staging = Config(llm_api_key=KEY, foamd_api_key=SERVICE, foamd_url="https://staging.example.test")
    assert sharing.send(other)["left"] == 1 and sharing.send(staging)["left"] == 1
    assert svc.posted == []
    old = path.stat().st_mtime - sharing.STALE_AFTER_S - 60
    os.utime(path, (old, old))
    assert sharing.send(other)["dropped"] == 1 and sharing.queued() == [] and svc.posted == []


def test_a_trace_queued_before_this_version_is_dropped(seen, service):
    """A queue file that does not say whose it is cannot be sent safely."""
    svc = service()
    sharing._queue()
    path = sharing.queue_dir() / ("a" * 32 + ".json.gz")
    path.write_bytes(b"x")
    os.utime(path, (1, 1))
    assert sharing.send(seen)["dropped"] == 1 and svc.posted == []


def test_the_bytes_queued_are_the_bytes_sent_with_the_clock(seen, service):
    svc = service()
    path = _queue_one(seen)
    queued = path.read_bytes()
    sharing.send(seen)
    (request,) = svc.requests
    assert request.content == queued
    assert request.headers["x-openreynolds-sent-at"].endswith("+00:00")


def test_stamps_carry_microseconds(seen):
    """A delete and a trace queued just after it, in the same second, must compare."""
    stamp = sharing._now()
    assert "." in stamp.split("T")[1]


@pytest.mark.parametrize("content", ["{not json", '{"enabled": "false"}', '{"enabled": 0}', "[]",
                                     '{"enabled": "disabled"}', '{"enabled": ""}', '{"Enabled": false}',
                                     '{"notice": 1}'])
def test_a_switch_file_that_cannot_be_read_is_off(on, content):
    sharing.state_path().parent.mkdir(parents=True, exist_ok=True)
    sharing.state_path().write_text(content)
    assert sharing.active(on) is False
    assert sharing.begin(on, "s1") == (None, None)
    if content != '{"enabled": "false"}':
        assert "could not be read" in "\n".join(sharing.describe(on))


def test_a_switch_written_true_is_on(on):
    sharing.state_path().parent.mkdir(parents=True, exist_ok=True)
    sharing.state_path().write_text('{"enabled": true, "notice": 1}')
    assert sharing.active(on) is True


def test_a_config_folder_that_cannot_be_written_never_fails_a_study(on, monkeypatch):
    """Told every time instead of once, and the study runs."""
    def refuse(state):
        raise PermissionError("read-only")

    monkeypatch.setattr(sharing, "save_state", refuse)
    recorder, notice = sharing.begin(on, "s1")
    assert recorder is not None and notice == sharing.NOTICE
    assert sharing.take_notice(on) == sharing.NOTICE


def test_whatever_goes_wrong_at_the_start_the_study_runs(on, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(sharing, "Recorder", broken)
    assert sharing.begin(on, "s1") == (None, None)
    assert sharing.begin(on, "s1", source="web") == (None, None)


def test_two_openreynolds_writing_at_once_do_not_collide(seen):
    import threading

    errors = []

    def write():
        try:
            for _ in range(20):
                sharing.save_state(sharing.State(enabled=True, notice=1))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert [p for p in sharing.state_path().parent.iterdir() if p.name.endswith(".tmp")] == []


def test_a_file_sent_by_another_openreynolds_meanwhile_is_not_an_error(seen, tmp_path):
    assert sharing._age(tmp_path / "gone.json.gz") == 0.0


def test_a_web_study_is_traced_whatever_the_runner_environment_says(seen, monkeypatch):
    """The web app's runner is the service's own: a stray OPENREYNOLDS_SHARING in its
    container does not turn off what the policy says always happens."""
    monkeypatch.setenv("OPENREYNOLDS_SHARING", "off")
    recorder, notice = sharing.begin(seen, "s1", source="web")
    assert recorder is not None and recorder.source == "web" and notice is None


def test_a_web_trace_asks_a_busy_service_again(seen, monkeypatch):
    answers = [503, 503, 201]
    sent = []

    def client(cfg, timeout):
        def reply(request):
            sent.append(request)
            return httpx.Response(answers.pop(0), json={"error": "busy"})
        return httpx.Client(base_url=cfg.foamd_url, transport=httpx.MockTransport(reply))

    monkeypatch.setattr(sharing, "_client", client)
    monkeypatch.setattr(sharing.time, "sleep", lambda s: None)
    recorder = sharing.Recorder(seen, "s1", source="web")
    recorder.turn(_turn(), 1.0)
    assert sharing.send_now(seen, recorder) is True and len(sent) == 3


def test_a_key_across_the_cut_leaves_no_piece_behind(seen):
    recorder = sharing.Recorder(seen, "s1")
    other = "sk-ant-api03-" + "z" * 60
    output = "x" * (sharing.CONTENT_AT_MOST - 21) + " " + other
    recorder.tool("bash", {"cmd": "env"}, output, False)
    io = recorder.records[-1]
    assert "zzzz" not in io["output"] and "sk-ant" not in io["output"]


def test_each_request_is_given_only_what_is_left_of_the_budget(seen, service):
    svc = service()
    _queue_one(seen)
    sharing.send(seen, budget_s=3.0)
    (request,) = svc.requests
    assert request.extensions["timeout"]["read"] <= 3.0


def test_ctrl_c_while_sending_leaves_the_trace_queued_and_the_close_down_running(seen, monkeypatch):
    from openreynolds import cli

    def interrupted(cfg, budget_s=8.0):
        raise KeyboardInterrupt

    monkeypatch.setattr(sharing, "send", interrupted)
    recorder = sharing.Recorder(seen, "s1")
    recorder.turn(_turn(), 1.0)
    cli._close_sharing(seen, recorder, web=False)   # does not raise
    assert len(sharing.queued()) == 1


@pytest.mark.parametrize("line,gone", [
    ('{"api_key": "abcd1234efgh5678"}', "abcd1234efgh5678"),
    ("'password': 'correct horse battery staple'", "correct horse battery staple"),
    ("postgres://reynolds:hunter2hunter2@db.example.com:5432/x", "hunter2hunter2"),
    ("curl -u alice:s3cr3tpass https://example.com", "s3cr3tpass"),
    ("Cookie: session=abcdef0123456789; theme=dark", "abcdef0123456789"),
    ('headers={"Authorization": "Bearer abcdefgh12345678"}', "abcdefgh12345678"),
])
def test_redaction_takes_the_common_shapes(line, gone):
    from openreynolds.redact import Redactor

    out = Redactor().text(line)
    assert gone not in out and MARK in out, out


def test_a_key_that_names_a_secret_marks_its_value():
    from openreynolds.redact import Redactor

    assert Redactor().deep({"password": "plainword", "case": "cavity"}) == {
        "password": MARK, "case": "cavity"}


@pytest.mark.parametrize("unit", [
    "TOKEN", "TOKEN=", "a_", "sk-", "eyJ-", "eyJaaaaaaaaa.", "xoxb-", "authorization:",
    "cookie:", "-----BEGIN PRIVATE KEY-----", "http://a:", "-u a:", '"password": "', " ",
])
def test_redaction_is_linear_in_the_text(unit):
    """A pattern that backtracks would hold the session: 40,000 characters built from each
    pattern's own prefix take well under a second (the old NAME=value pattern: 5.6 s)."""
    import time

    from openreynolds.redact import Redactor

    text = (unit * (40_000 // len(unit) + 1))[:40_000]
    started = time.perf_counter()
    Redactor().text(text)
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize("value,off", [("off", True), ("disabled", True), ("never", True),
                                       ("on", False), ("1", False), ("", False)])
def test_the_environment_switch_takes_anything_unknown_as_off(on, monkeypatch, value, off):
    monkeypatch.setenv("OPENREYNOLDS_SHARING", value)
    assert sharing.env_off() is off


def test_off_empties_the_queue_even_when_the_switch_cannot_be_saved(seen, monkeypatch):
    _queue_one(seen)

    def refuse(state):
        raise PermissionError("read-only")

    monkeypatch.setattr(sharing, "save_state", refuse)
    with pytest.raises(PermissionError):
        sharing.set_enabled(False)
    assert sharing.queued() == []


def test_status_while_signed_out_keeps_nothing_queued(seen):
    _queue_one(seen)
    sharing.describe(Config(llm_api_key=KEY))
    assert sharing.queued() == []


def test_a_redirect_is_not_followed(seen, monkeypatch):
    """A 307 would carry the trace to another host; a 301 would make it a GET whose 200
    reads as sent. Neither is followed, and the trace stays queued."""
    hosts = []

    def reply(request):
        hosts.append(request.url.host)
        if request.url.host == "api.example.test":
            return httpx.Response(307, headers={"location": "https://elsewhere.test/v1/traces"})
        return httpx.Response(201, json={})

    real = sharing._client
    monkeypatch.setattr(sharing, "_client", lambda cfg, timeout: _with_transport(real(cfg, timeout), reply))
    _queue_one(seen)
    assert sharing.send(seen)["left"] == 1 and hosts == ["api.example.test"]


def _with_transport(client, reply):
    """`sharing._client`'s own settings (redirects included), over a scripted transport."""
    client._transport = httpx.MockTransport(reply)
    return client


def test_the_budget_holds_whatever_the_network_does(seen, monkeypatch):
    """httpx's timeouts are per phase and restart with each byte; a post that never
    answers costs the budget, not more."""
    import threading
    import time as clock

    release = threading.Event()

    class Hung:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            release.set()

        def post(self, *a, **k):
            release.wait(30)
            raise httpx.ReadTimeout("trickled")

    monkeypatch.setattr(sharing, "_client", lambda cfg, timeout: Hung())
    _queue_one(seen)
    started = clock.monotonic()
    assert sharing.send(seen, budget_s=1.5)["left"] == 1
    recorder = sharing.Recorder(seen, "s1", source="web")
    recorder.turn(_turn(), 1.0)
    assert sharing.send_now(seen, recorder, budget_s=1.5) is False
    assert clock.monotonic() - started < 4.0


@pytest.mark.parametrize("line,expected", [
    ('curl --user "alice:s3cret pass" https://x', f'curl --user "alice:{MARK}" https://x'),
    ("postgres://bob:p@ss@db/x", f"postgres://bob:{MARK}@db/x"),
    ("DB_PASS=hunter2hunter2", f"DB_PASS={MARK}"),
    ("bypass_cache=enabled", "bypass_cache=enabled"),
])
def test_redaction_takes_the_last_shapes_and_leaves_lookalikes(line, expected):
    from openreynolds.redact import Redactor

    assert Redactor().text(line) == expected


def test_an_object_under_a_secret_key_goes_whole():
    from openreynolds.redact import Redactor

    assert Redactor().deep({"credentials": {"user": "bob", "pass": "hunter2hunter2"}}) == {"credentials": MARK}
