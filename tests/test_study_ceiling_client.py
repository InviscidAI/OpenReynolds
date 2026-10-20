"""The per-study spending ceiling, from the agent's side, against a fake foamd.

foamd pauses a study at 80% of its ceiling: the study's new sandboxes, model calls and
GPU solves answer `409 study_paused` with `{error, message, study_id, spent_cents,
cap_cents}`, and `POST /v1/studies/{id}/cap {cap_cents}` raises the ceiling (owner
only; it must exceed what is spent). What the agent does with that:

- the 409 becomes a question to the person (`approval.Approver`) showing what is spent,
  the ceiling and the estimate given before the first solve;
- yes raises the ceiling (double the old one, or what the person typed) and the call
  that met the pause goes again; no stops the study with a plain final message;
- before the first solve the person is told the estimate and the study's ceiling.

The 402/403/409 bodies carry more than `error` and `message`, and those fields survive
into `BackendError` (and a model call's `ProviderError`) instead of being dropped.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx as real_httpx
import httpx2 as httpx
import pytest

from conftest import RecordingView, ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import ceiling, cli
from openreynolds.approval import Approver
from openreynolds.backend.base import BackendError
from openreynolds.backend import hosted
from openreynolds.backend.hosted import FoamdClient, HostedBackend, _decode_error
from openreynolds.config import Config
from openreynolds.llm.base import ProviderError
from openreynolds.loop import Loop
from openreynolds.tools import ToolContext
from test_cli import finishing_jobs

pytestmark = pytest.mark.usefixtures("fast_polling")

STUDY = "20261020-101500-abcd"


# -- the fake foamd --------------------------------------------------------------


class FakeFoamd:
    """The routes this chunk touches, with the study's spend and ceiling as state.

    A study is paused while `spent >= 80% of cap`, as I1's reaper and proxy decide it.
    `POST .../cap` refuses a ceiling that does not exceed what is spent (409), and is
    owner only (a wrong key is a 403)."""

    def __init__(self, spent_cents=800, cap_cents=1000, style="quick", key="of_live_test"):
        self.spent_cents = spent_cents
        self.cap_cents = cap_cents
        self.style = style
        self.key = key
        self.caps_posted: list[int] = []
        self.execs: list[str] = []
        self.paused_answers = 0
        self.study_headers: list[tuple[str, str, str | None]] = []
        """(method, path, X-Study-Id) of every request, as the middleware reads it."""
        self.spend_at_raise: int | None = None
        """What the study has spent by the time a raise lands, when the service counts
        spend this side has not seen (an open sandbox priced to now)."""

    @property
    def paused(self) -> bool:
        return self.spent_cents >= 0.8 * self.cap_cents

    def paused_body(self) -> dict:
        return {
            "error": "study_paused",
            "message": (f"this study has spent ${self.spent_cents / 100:.2f} of its "
                        f"${self.cap_cents / 100:.2f} ceiling and is paused at 80%. "
                        "Raise its ceiling to continue."),
            "study_id": STUDY,
            "spent_cents": self.spent_cents,
            "cap_cents": self.cap_cents,
        }

    def handler(self, request):
        path = request.url.path
        if request.headers.get("authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"error": "unauthorized", "message": "no key"})
        self.study_headers.append((request.method, path, request.headers.get("x-study-id")))
        if path == f"/v1/studies/{STUDY}" and request.method == "GET":
            # foamd's get_study: the public row plus study_ceiling.view() -- style, the
            # cap in force, and spent/paused (None when the ledgers cannot be read).
            return httpx.Response(200, json={
                "id": STUDY, "title": "pipe", "solver": "openfoam", "style": self.style,
                "cap_cents": self.cap_cents, "spent_cents": self.spent_cents,
                "paused": self.paused if self.spent_cents is not None else None,
            })
        if path == f"/v1/studies/{STUDY}/cap" and request.method == "POST":
            cents = json.loads(request.content)["cap_cents"]
            self.caps_posted.append(cents)
            if self.spend_at_raise is not None:
                self.spent_cents = self.spend_at_raise
            if cents <= self.spent_cents:
                return httpx.Response(409, json={
                    "error": "conflict",
                    "message": (f"the new ceiling must exceed what the study has spent "
                                f"(${self.spent_cents / 100:.2f}); asked for "
                                f"${cents / 100:.2f}"),
                })
            self.cap_cents = cents
            # Accepted above the spend even when the spend is still past 80% of it,
            # answered `paused: true` (study_ceiling.set_cap).
            return httpx.Response(200, json={
                "study_id": STUDY, "cap_cents": cents, "spent_cents": self.spent_cents,
                "pause_at_cents": round(cents * 0.8, 2), "paused": self.paused,
            })
        if path == "/v1/instances/inst-1/start":
            return httpx.Response(200, json={"id": "inst-1", "started_new": True})
        if path == "/v1/instances/inst-1/files":
            if request.method == "PUT":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(200, content=b"hello")
        if path == "/v1/instances/inst-1/jobs" and request.method == "POST":
            return httpx.Response(201, json={"job_id": "job-1"})
        if path.endswith("/exec"):
            self.execs.append(json.loads(request.content)["cmd"])
            if self.paused:
                self.paused_answers += 1
                return httpx.Response(409, json=self.paused_body())
            return httpx.Response(200, json={"exit_code": 0, "output": "ran\n"})
        return httpx.Response(404, json={"error": "not_found", "message": path})

    def client(self) -> FoamdClient:
        client = FoamdClient("https://svc.example", self.key)
        client._client = httpx.Client(
            base_url="https://svc.example",
            headers={"Authorization": f"Bearer {self.key}"},
            transport=httpx.MockTransport(self.handler),
            follow_redirects=False,
        )
        return client


class AnsweringView(RecordingView):
    """Types `answer` only once a question is on screen, as a person would."""

    def __init__(self, reader, *answers):
        super().__init__()
        self.reader = reader
        self.answers_to_give = list(answers)
        self.questions = []
        self.answers = []

    def approval(self, request_id, kind, title, detail, choices):
        self.questions.append((request_id, kind, title, detail, choices))
        if self.answers_to_give:
            self.reader._lines.append(self.answers_to_give.pop(0))

    def approval_done(self, request_id, outcome, note=""):
        self.answers.append((request_id, outcome, note))


def _session(tmp_path, foamd, *answers, mode="auto", backend=None, approver=True):
    """A loop on a hosted workspace served by `foamd`, with someone at the keyboard
    typing `answers` to whatever is asked (or nobody, with `approver=False`)."""
    from openreynolds.store import Store

    store = Store(tmp_path / "studies", STUDY)
    store.session.remote_study_id = STUDY
    client = foamd.client()
    backend = backend if backend is not None else HostedBackend(client, "inst-1")
    ctx = ToolContext(backend=backend, store=store, max_output=4000, mode=mode)
    ctx.ceiling = ceiling.Ceiling(client=client, study_id=STUDY)
    reader = ScriptedReader([])
    view = AnsweringView(reader, *answers)
    loop = Loop(Config(llm_api_key="k", model="claude-opus-5", mode=mode), ctx, store, view)
    if approver:
        loop.approver = ctx.approver = Approver(view, reader)
    ctx.view = view
    return loop, ctx, view, store


# -- 1. the extra fields survive -----------------------------------------------------


def _response(status, body):
    return httpx.Response(status, json=body,
                          request=httpx.Request("POST", "https://svc.example/v1/x"))


def test_a_402_keeps_the_tier_the_period_and_whether_overage_can_be_turned_on():
    from datetime import datetime, timedelta, timezone

    ends = (datetime.now(timezone.utc) + timedelta(days=20)).isoformat()
    fields = {"tier": "starter", "period_end": ends, "renews_at": ends,
              "can_enable_overage": True, "can_buy_credits": True, "past_due": False}
    error = _decode_error(_response(402, {
        "error": "allowance_exhausted",
        "message": "this period's Starter allowance is used up.", **fields,
    }))
    assert (error.code, error.status) == ("allowance_exhausted", 402)
    assert error.extra == fields
    said = str(error)
    assert "Starter" in said and ends[:10] in said
    assert "credits can be bought" in said
    assert "pay-as-you-go overage (billed to your card)" in said


def test_a_403_plan_limit_names_the_tier_to_upgrade_to():
    error = _decode_error(_response(403, {
        "error": "plan_limit", "message": "max_cpu 16 is over the limit.",
        "limit": "max_cpu", "value": 16, "allowed": 8, "tier": "starter",
        "upgrade_to": "pro",
    }))
    assert error.extra["upgrade_to"] == "pro" and error.extra["allowed"] == 8
    said = str(error)
    assert said.startswith("plan_limit (403): max_cpu 16 is over the limit.")
    assert "Starter" in said and "upgrade to Pro" in said


def test_a_409_study_paused_keeps_the_study_its_spend_and_its_ceiling():
    error = _decode_error(_response(409, FakeFoamd().paused_body()))
    assert error.code == "study_paused"
    assert error.extra == {"study_id": STUDY, "spent_cents": 800, "cap_cents": 1000}
    assert "$8.00" in str(error) and "$10.00" in str(error)
    info = ceiling.paused(error)
    assert (info.study_id, info.spent_cents, info.cap_cents) == (STUDY, 800, 1000)


def test_an_error_with_no_extra_fields_reads_exactly_as_before():
    error = _decode_error(_response(404, {"error": "not_found", "message": "gone"}))
    assert error.extra == {} and str(error) == "not_found (404): gone"
    assert ceiling.paused(error) is None


def _sdk_error(status, body):
    """What the model client raises when foamd's proxy refuses: the SDK's own error,
    carrying the real response, wrapped the way every provider wraps it."""
    response = real_httpx.Response(
        status, json=body,
        request=real_httpx.Request("POST", "https://svc.example/v1/llm/v1/messages"),
    )
    cls = {402: anthropic.APIStatusError, 403: anthropic.PermissionDeniedError,
           409: anthropic.ConflictError}[status]
    sdk = cls(body.get("message", ""), response=response, body=body)
    try:
        raise ProviderError(body.get("message", ""), status) from sdk
    except ProviderError as exc:
        return exc


def test_a_model_call_refused_by_the_proxy_keeps_its_fields_too():
    exc = _sdk_error(403, {"error": "plan_limit", "message": "over the limit",
                           "tier": "starter", "upgrade_to": "pro"})
    assert exc.code == "plan_limit"
    assert exc.body["upgrade_to"] == "pro"
    info = ceiling.paused(_sdk_error(409, FakeFoamd().paused_body()))
    assert (info.spent_cents, info.cap_cents) == (800, 1000)


# -- 2. a pause on a tool call becomes a question -------------------------------------


def test_yes_raises_the_ceiling_to_double_and_the_paused_call_goes_again(tmp_path):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes")
    ctx.estimate_cents = 600
    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    (_, kind, title, detail, choices), = view.questions
    assert kind == "ceiling"
    assert "$8.00" in detail and "$10.00" in detail, "spent and the ceiling"
    assert "$6.00" in detail, "the estimate given before the first solve"
    assert "$20.00" in detail, "the default is double the old ceiling"
    assert foamd.caps_posted == [2000]
    assert foamd.execs == ["blockMesh", "blockMesh"], "the paused call went again"
    result = loop.messages[-2]["content"][0]
    assert not result.get("is_error") and "ran" in str(result["content"])
    assert len(fake.calls) == 2 and ctx.stopped is None


def test_a_typed_amount_is_the_new_ceiling_in_dollars(tmp_path):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, _ = _session(tmp_path, foamd, "$35")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert foamd.caps_posted == [3500]
    assert foamd.cap_cents == 3500 and foamd.execs == ["blockMesh", "blockMesh"]


def test_an_amount_that_would_pause_again_at_once_is_asked_again(tmp_path):
    """$9 exceeds the $8 spent -- the service would take it -- but 80% of it is $7.20,
    under what is spent, so the study would be paused again by the very next call."""
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, _ = _session(tmp_path, foamd, "9", "/yes")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert len(view.questions) == 2
    assert "at least $11.00" in view.questions[1][3], "says what the least useful ceiling is"
    assert foamd.caps_posted == [2000]


def test_no_stops_the_study_cleanly_and_the_model_is_not_asked_again(tmp_path):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, "/no")
    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"}, "tu_1"),
                 tool_block("bash", {"cmd": "checkMesh"}, "tu_2")], stop_reason="tool_use"),
        message([text_block("should never be asked for")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert foamd.caps_posted == [] and foamd.execs == ["blockMesh"]
    assert len(fake.calls) == 1, "the turn ended without another model request"
    results = loop.messages[-1]["content"]
    assert [r["tool_use_id"] for r in results if r.get("type") == "tool_result"] == ["tu_1", "tu_2"]
    assert all(r["is_error"] for r in results if r.get("type") == "tool_result")
    assert ctx.stopped and "$8.00" in ctx.stopped and "$10.00" in ctx.stopped
    assert any(ctx.stopped in n for n in view.notices), "the final message is shown"
    transcript = (store.dir / "messages.jsonl").read_text()
    assert "stopped" in transcript and "ceiling" in transcript, "and kept in the record"


def test_with_nobody_to_ask_the_study_stops_and_says_how_to_raise_it(tmp_path):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, _ = _session(tmp_path, foamd, approver=False)
    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("should never be asked for")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert foamd.caps_posted == [] and len(fake.calls) == 1
    assert ctx.stopped and f"/v1/studies/{STUDY}/cap" in ctx.stopped


def test_the_interactive_loop_does_not_wake_the_model_after_a_stop(tmp_path):
    """After a no, `blocked_reason` holds the stop, so a job finishing or a narration
    tick is not a reason to ask the model again -- it would be refused at the proxy
    exactly as the last call was."""
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/no")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("should never be asked for")]),
    ])
    loop.say("mesh it")
    assert cli._run_turn(loop, view) is True
    assert loop.blocked_reason == ctx.stopped


# -- 2b. a pause on the model call itself ------------------------------------------------


class PausedThenFine:
    """The model proxy, paused until the ceiling is raised on `foamd`."""

    def __init__(self, foamd, then):
        self.foamd = foamd
        self.then = then
        self.calls = []

    def stream(self, **kwargs):
        from conftest import FakeStream

        self.calls.append(kwargs)
        if self.foamd.paused:
            body = self.foamd.paused_body()
            response = real_httpx.Response(
                409, json=body,
                request=real_httpx.Request("POST", "https://svc.example/v1/llm/v1/messages"),
            )
            raise anthropic.ConflictError(body["message"], response=response, body=body)
        return FakeStream(self.then)


def test_a_paused_model_call_is_put_to_the_person_and_retried_once_raised(tmp_path):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes")
    fake = PausedThenFine(foamd, message([text_block("carrying on")]))
    loop.client = SimpleNamespace(messages=fake)
    loop.say("carry on")

    assert cli._run_turn(loop, view) is True
    assert [q[1] for q in view.questions] == ["ceiling"]
    assert foamd.caps_posted == [2000] and len(fake.calls) == 2


def test_a_paused_model_call_declined_is_not_retried_as_a_conflict(tmp_path):
    """409 is a transient status for a model call (`TRANSIENT_STATUSES`): without this,
    a paused study would be asked five more times over four minutes."""
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/no")
    fake = PausedThenFine(foamd, message([text_block("never")]))
    loop.client = SimpleNamespace(messages=fake)
    loop.say("carry on")

    assert cli._run_turn(loop, view) is False
    assert len(fake.calls) == 1 and foamd.caps_posted == []
    assert loop.blocked_reason and "ceiling" in loop.blocked_reason


# -- 3. the estimate before the first solve ----------------------------------------------


def test_the_first_solve_is_preceded_by_the_estimate_and_the_ceiling(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=120, cap_cents=5000, style="full")
    finishing_jobs(backend)
    loop, ctx, view, _ = _session(tmp_path, foamd, backend=backend)
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve",
                                          "estimate_usd": 4.5}, "tu_1")],
                stop_reason="tool_use"),
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "again",
                                          "estimate_usd": 4.5}, "tu_2")],
                stop_reason="tool_use"),
        message([text_block("running")]),
    ])
    loop.say("solve it")
    loop.run()

    said = [n for n in view.notices if "estimate" in n.lower()]
    assert len(said) == 1, "said once, before the first solve only"
    line = said[0]
    assert "$4.50" in line and "$50.00" in line and "$40.00" in line and "$1.20" in line
    assert view.questions == [], "a message, not a question, in auto"
    assert [s["cmd"] for s in backend.started] == ["simpleFoam", "simpleFoam"]
    assert ctx.estimate_cents == 450


def test_with_compute_approval_on_the_estimate_is_part_of_the_question(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=0, cap_cents=1000, style="quick")
    finishing_jobs(backend)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes", mode="partial", backend=backend)
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve",
                                          "estimate_usd": 2})], stop_reason="tool_use"),
        message([text_block("running")]),
    ])
    loop.say("solve it")
    loop.run()

    (_, kind, _, detail, _), = view.questions
    assert kind == "job" and "$2.00" in detail and "$10.00" in detail and "$8.00" in detail
    assert not [n for n in view.notices if "estimate" in n.lower()]


def test_a_solve_with_no_estimate_still_shows_the_ceiling_and_says_none_was_given(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=0, cap_cents=5000)
    finishing_jobs(backend)
    loop, ctx, view, _ = _session(tmp_path, foamd, backend=backend)
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})],
                stop_reason="tool_use"),
        message([text_block("running")]),
    ])
    loop.say("solve it")
    loop.run()

    line, = [n for n in view.notices if "ceiling" in n]
    assert "no estimate" in line.lower() and "$50.00" in line


def test_a_study_record_without_a_ceiling_falls_back_to_the_style_default():
    assert ceiling.default_cap_cents("quick") == 1000
    assert ceiling.default_cap_cents("full") == 5000
    assert ceiling.default_cap_cents(None) == 5000
    assert ceiling.default_cap_cents("") == 5000


def test_the_job_tools_take_an_estimate():
    from openreynolds.tools import FELIX_TOOLS, TOOLS

    for tools in (TOOLS, FELIX_TOOLS):
        job = next(t for t in tools if t["name"] == "job_start")
        assert job["input_schema"]["properties"]["estimate_usd"]["type"] == "number"
        assert "estimate_usd" not in job["input_schema"].get("required", [])


def test_raise_study_cap_posts_the_new_ceiling():
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    answer = foamd.client().raise_study_cap(STUDY, 2000)
    assert answer["cap_cents"] == 2000 and foamd.caps_posted == [2000]
    with pytest.raises(BackendError) as refused:
        foamd.client().raise_study_cap(STUDY, 700)
    assert refused.value.status == 409


def test_a_side_question_never_puts_the_ceiling_to_the_person(tmp_path):
    """`/btw` runs reads on a copy of the context; a pause met there is said to the side
    chat, and the question stays the conversation's."""
    from openreynolds.desk import _side_ctx

    loop, ctx, view, _ = _session(tmp_path, FakeFoamd())
    side = _side_ctx(ctx)
    assert side.approver is None and side.ceiling is None
    assert ctx.approver is not None and ctx.ceiling is not None


# -- 4. the study rides on every workspace call --------------------------------------


def _named(foamd, method, suffix):
    return [header for m, path, header in foamd.study_headers
            if m == method and path.endswith(suffix)]


def test_workspace_calls_name_the_study_once_the_session_knows_it():
    """foamd tags a sandbox start, and judges it against the study's ceiling, by the
    `X-Study-Id` it names; without one a start is tagged with the workspace's newest
    study, which on a reused workspace is the previous one."""
    foamd = FakeFoamd(spent_cents=0)
    client = foamd.client()
    backend = HostedBackend(client, "inst-1")
    backend.exec("true")
    assert _named(foamd, "POST", "/exec") == [None], "no study yet, none named"

    backend.study_id = STUDY
    backend.exec("ls")
    backend.put_file("/work/a.txt", b"a")
    backend.get_file("/work/a.txt")
    backend.job_start("felix run case --gpu A100")
    client.start_instance("inst-1")

    assert _named(foamd, "POST", "/exec")[-1] == STUDY
    assert _named(foamd, "PUT", "/files") == [STUDY]
    assert _named(foamd, "GET", "/files") == [STUDY]
    assert _named(foamd, "POST", "/jobs") == [STUDY], "a solve's job names its study"
    assert _named(foamd, "POST", "/start") == [STUDY]


def test_a_workspace_still_starting_hands_the_study_on_to_its_calls():
    from openreynolds.backend.pending import PendingBackend

    foamd = FakeFoamd(spent_cents=0)
    pending = PendingBackend("inst-1")
    pending.study_id = STUDY
    pending.resolve(HostedBackend(foamd.client(), "inst-1"))
    pending.exec("ls")
    assert _named(foamd, "POST", "/exec") == [STUDY]


def test_a_resumed_study_names_itself_on_the_start_too():
    """A resume knows its study before the workspace is started, so the start is
    exact rather than the service's guess."""
    foamd = FakeFoamd(spent_cents=0)
    client = foamd.client()
    client.study_id = STUDY
    hosted.Starter(client, "inst-1").result(timeout=5)
    assert _named(foamd, "POST", "/start") == [STUDY]


# -- 5. what foamd's study routes answer now --------------------------------------------


def test_a_raise_the_service_still_counts_as_paused_is_put_again_not_retried(tmp_path):
    """foamd takes a ceiling above the spend even when the spend is still past 80% of
    it -- spend this side had not seen, an open sandbox priced to now -- and answers
    `paused: true`. That is asked again at once, not met as a second 409."""
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    foamd.spend_at_raise = 900
    loop, ctx, view, _ = _session(tmp_path, foamd, "11", "/yes")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert foamd.caps_posted == [1100, 2200]
    assert len(view.questions) == 2
    again = view.questions[1][3]
    assert "$9.00" in again and "$11.00" in again, "says the service still counts it paused"
    assert foamd.execs == ["blockMesh", "blockMesh"], "no call went out into a known pause"
    assert not any("$11.00; carrying on" in n for n in view.notices)
    assert ctx.stopped is None


def test_the_preamble_reads_the_study_record_foamd_sends(tmp_path):
    """`spent_cents` and `paused` are None when foamd cannot read the ledgers."""
    foamd = FakeFoamd(spent_cents=None, cap_cents=5000, style="full")
    ctx = SimpleNamespace(ceiling=ceiling.Ceiling(client=foamd.client(), study_id=STUDY))
    said = ceiling.preamble(ctx, {"estimate_usd": 12})
    assert "$12.00" in said and "$50.00" in said and "spent so far" not in said

    foamd = FakeFoamd(spent_cents=850, cap_cents=1000)
    ctx = SimpleNamespace(ceiling=ceiling.Ceiling(client=foamd.client(), study_id=STUDY))
    said = ceiling.preamble(ctx, {"estimate_usd": 3})
    assert "$8.50" in said and "paused" in said.split("pauses at")[-1], \
        "a study already paused says so before the solve meets it"
