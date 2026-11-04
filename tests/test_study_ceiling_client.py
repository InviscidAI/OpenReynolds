"""The per-study spending ceiling, from the agent's side, against a fake foamd.

foamd pauses a study at 80% of its ceiling: the study's new sandboxes, model calls and
GPU solves answer `409 study_paused` with `{error, message, study_id, spent_cents,
cap_cents, times_normal, style}`, and `POST /v1/studies/{id}/cap {cap_cents}` raises
the ceiling (owner only; it must exceed what is spent). What the agent does with that:

- the 409 becomes a question to the person (`approval.Approver`): this study has used
  2x the usage of a normal study, do you want to continue?
- yes doubles the ceiling and the call that met the pause goes again; no stops the
  study with a plain final message;
- before the first solve the person is told how far the study may go before it asks.

Billing stays in cents between the agent and the service, and none of it reaches the
person as money: every figure they see is a multiple of a normal study, from the
service's own `times_normal` or, from an older service without it, from the cents.
The `$` tests below hold each surface to that.

The 402/403/409 bodies carry more than `error` and `message`, and those fields survive
into `BackendError` (and a model call's `ProviderError`) instead of being dropped.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx2 as real_httpx
import httpx2 as httpx
import pytest

from conftest import RecordingView, ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import ceiling, cli, solvers
from openreynolds.approval import Approver
from openreynolds.backend.base import BackendError, JobStatus
from openreynolds.backend import hosted
from openreynolds.backend.hosted import FoamdClient, HostedBackend, _decode_error
from openreynolds.config import Config
from openreynolds.llm.base import ProviderError
from openreynolds.loop import Loop
from openreynolds.tools import ToolContext, dispatch
from openreynolds.watch import Wake, _collect_finished, _job_report
from test_cli import finishing_jobs

pytestmark = pytest.mark.usefixtures("fast_polling")

STUDY = "20261020-101500-abcd"


# -- the fake foamd --------------------------------------------------------------


class FakeFoamd:
    """The routes this chunk touches, with the study's spend and ceiling as state.

    A study is paused while `spent >= 80% of cap`, as I1's reaper and proxy decide it.
    `POST .../cap` refuses a ceiling that does not exceed what is spent (409), and is
    owner only (a wrong key is a 403).

    The defaults are a full study at its default ceiling (1250 cents) that has used
    1000, twice a normal full study: where every default study pauses. `modern=False`
    is a foamd from before `times_normal`, whose prose named dollars."""

    def __init__(self, spent_cents=1000, cap_cents=1250, style="full", key="of_live_test",
                 modern=True):
        self.spent_cents = spent_cents
        self.cap_cents = cap_cents
        self.style = style
        self.key = key
        self.modern = modern
        self.caps_posted: list[int] = []
        self.execs: list[str] = []
        self.paused_answers = 0
        self.study_headers: list[tuple[str, str, str | None]] = []
        """(method, path, X-Study-Id) of every request, as the middleware reads it."""
        self.spend_at_raise: int | None = None
        """What the study has spent by the time a raise lands, when the service counts
        spend this side has not seen (an open sandbox priced to now)."""
        self.start_pauses = False
        """Whether a workspace start naming a paused study is refused with the pause,
        as foamd judges a resumed study's start against its ceiling."""
        self.starts = 0

    @property
    def paused(self) -> bool:
        return self.spent_cents >= 0.8 * self.cap_cents

    @property
    def normal(self) -> int:
        return 200 if self.style == "quick" else 500

    def multiples(self, cap=None) -> dict:
        """foamd's `times_normal` and `pause_times_normal`; nothing from an old one."""
        if not self.modern:
            return {}
        cap = self.cap_cents if cap is None else cap
        return {
            "times_normal": (round(self.spent_cents / self.normal, 1)
                             if self.spent_cents is not None else None),
            "pause_times_normal": round(cap * 0.8 / self.normal, 1),
        }

    def paused_body(self) -> dict:
        if not self.modern:
            return {
                "error": "study_paused",
                "message": (f"this study has spent ${self.spent_cents / 100:.2f} of its "
                            f"${self.cap_cents / 100:.2f} ceiling and is paused at 80%. "
                            "Raise its ceiling to continue."),
                "study_id": STUDY,
                "spent_cents": self.spent_cents,
                "cap_cents": self.cap_cents,
            }
        used = round(self.spent_cents / self.normal, 1)
        return {
            "error": "study_paused",
            "message": (f"study {STUDY} has used {used:g}x the usage of a normal "
                        f"{self.style} study and is paused. To continue, its owner "
                        f"raises the ceiling: POST /v1/studies/{STUDY}/cap "
                        '{"cap_cents": <more>}'),
            "study_id": STUDY,
            "spent_cents": self.spent_cents,
            "cap_cents": self.cap_cents,
            "times_normal": used,
            "style": self.style,
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
                **self.multiples(),
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
                                f"${cents / 100:.2f}") if not self.modern else
                               "the new ceiling must exceed what the study has used",
                })
            self.cap_cents = cents
            # Accepted above the spend even when the spend is still past 80% of it,
            # answered `paused: true` (study_ceiling.set_cap).
            return httpx.Response(200, json={
                "study_id": STUDY, "cap_cents": cents, "spent_cents": self.spent_cents,
                "pause_at_cents": round(cents * 0.8, 2), "paused": self.paused,
                **self.multiples(cents),
            })
        if path == "/v1/instances/inst-1/start":
            self.starts += 1
            if self.start_pauses and self.paused and request.headers.get("x-study-id"):
                return httpx.Response(409, json=self.paused_body())
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
    assert error.extra == {"study_id": STUDY, "spent_cents": 1000, "cap_cents": 1250,
                           "times_normal": 2.0, "style": "full"}
    assert "used 2x the usage of a normal full study" in str(error)
    assert "$" not in str(error)
    info = ceiling.paused(error)
    assert (info.study_id, info.spent_cents, info.cap_cents) == (STUDY, 1000, 1250)
    assert (info.times_normal, info.style) == (2.0, "full")


def test_the_backend_error_text_of_an_older_pause_names_no_money():
    """An older foamd said "this study has spent $10.00 of its $12.50 ceiling" in its
    prose and sent no `times_normal`. The error's text is what the person reads when a
    workspace start meets the pause, so the money is left out and the multiple is
    worked out from the cents."""
    old = FakeFoamd(modern=False)
    error = _decode_error(_response(409, old.paused_body()))
    said = str(error)
    assert "$" not in said, said
    assert "used 2x the usage of a normal full study" in said
    assert "Raise its ceiling to continue." in said, "the rest of the prose is kept"
    conflict = _decode_error(_response(409, {
        "error": "conflict", "message": "the new ceiling must exceed what the study has "
        "spent ($10.00); asked for $9.00", "study_id": STUDY}))
    assert "$" not in str(conflict) and "amount is not shown" in str(conflict)


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
    assert (info.spent_cents, info.cap_cents, info.times_normal) == (1000, 1250, 2.0)


# -- 2. a pause on a tool call becomes a question -------------------------------------


def test_yes_raises_the_ceiling_to_double_and_the_paused_call_goes_again(tmp_path):
    foamd = FakeFoamd()
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes")
    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    (_, kind, title, detail, choices), = view.questions
    assert kind == "ceiling"
    assert title == "This study has used 2x a normal study"
    assert detail.splitlines()[0] == ("This study has used 2x the usage of a normal full "
                                      "study. Do you want to continue?")
    assert "Yes lets it run to 4x before asking again; no stops it here." in detail
    assert choices == ["continue", "stop the study"]
    assert foamd.caps_posted == [2500], "yes is double the old ceiling"
    assert foamd.execs == ["blockMesh", "blockMesh"], "the paused call went again"
    result = loop.messages[-2]["content"][0]
    assert not result.get("is_error") and "ran" in str(result["content"])
    assert len(fake.calls) == 2 and ctx.stopped is None
    notice, = [n for n in view.notices if "carrying on" in n]
    assert notice == ("The study can now run to 4x a normal full study before asking "
                      "again; carrying on.")


@pytest.mark.parametrize("modern", [True, False])
def test_no_dollar_sign_reaches_the_person_at_a_pause(tmp_path, modern):
    """The question, the notice after a raise, and every line the view was handed:
    none of them names money, from a current foamd or from one that sends no
    `times_normal` and puts dollars in its prose."""
    foamd = FakeFoamd(modern=modern)
    loop, ctx, view, store = _session(tmp_path, foamd, "/yes")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    (_, _, title, detail, _), = view.questions
    assert "2x" in title and "4x" in detail
    for said in [title, detail, *view.notices]:
        assert "$" not in said, said
    assert foamd.caps_posted == [2500]


def test_a_quick_study_is_measured_against_a_normal_quick_study(tmp_path):
    """A quick study's default ceiling is 500 cents and a normal one uses 200, so it
    too pauses at 2x; an older foamd without `style` in its pause is read from the
    study's record."""
    foamd = FakeFoamd(spent_cents=400, cap_cents=500, style="quick", modern=False)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    (_, _, title, detail, _), = view.questions
    assert "normal quick study" in detail and title.endswith("2x a normal study")
    assert "run to 4x" in detail and foamd.caps_posted == [1000]


def test_a_raise_doubling_leaves_short_goes_to_what_is_needed_unsaid(tmp_path):
    """A study well past its ceiling (a service counting what this side had not seen)
    is raised to the least ceiling that would not pause again at once, not to a double
    that would, and nothing about that is shown in money."""
    foamd = FakeFoamd(spent_cents=3000, cap_cents=1250)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert foamd.caps_posted == [3800], "80% of 3800 is above the 3000 used"
    (_, _, title, detail, _), = view.questions
    assert "6x" in title and "run to 6.1x" in detail and "$" not in detail


def test_no_stops_the_study_cleanly_and_the_model_is_not_asked_again(tmp_path):
    foamd = FakeFoamd()
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
    assert ctx.stopped and ctx.stopped.startswith(
        "This study is paused: it has used 2x the usage of a normal full study.")
    assert "$" not in ctx.stopped
    assert any(ctx.stopped in n for n in view.notices), "the final message is shown"
    transcript = (store.dir / "messages.jsonl").read_text()
    assert "stopped" in transcript and "ceiling" in transcript, "and kept in the record"


@pytest.mark.parametrize("modern", [True, False])
def test_with_nobody_to_ask_the_study_stops_and_says_how_to_raise_it(tmp_path, modern):
    foamd = FakeFoamd(modern=modern)
    loop, ctx, view, _ = _session(tmp_path, foamd, approver=False)
    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("should never be asked for")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert foamd.caps_posted == [] and len(fake.calls) == 1
    assert ctx.stopped and f"/v1/studies/{STUDY}/cap" in ctx.stopped
    assert "2x the usage of a normal full study" in ctx.stopped and "$" not in ctx.stopped


def test_a_raise_the_service_refuses_stops_without_naming_money():
    """The stop message carries the raise's error, and an older foamd's refusal named
    the spend and the ceiling asked for in dollars."""
    info = ceiling.Paused(STUDY, 1000, 1250, times_normal=2.0, style="full")
    error = BackendError("the new ceiling must exceed what the study has spent ($10.00); "
                         "asked for $9.00", code="conflict", status=409)
    said = ceiling.stop_message(info, asked=True, error=str(error))
    assert "$" not in said and "could not be raised" in said


def test_the_interactive_loop_does_not_wake_the_model_after_a_stop(tmp_path):
    """After a no, `blocked_reason` holds the stop, so a job finishing or a narration
    tick is not a reason to ask the model again -- it would be refused at the proxy
    exactly as the last call was."""
    foamd = FakeFoamd()
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
    foamd = FakeFoamd()
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes")
    fake = PausedThenFine(foamd, message([text_block("carrying on")]))
    loop.client = SimpleNamespace(messages=fake)
    loop.say("carry on")

    assert cli._run_turn(loop, view) is True
    assert [q[1] for q in view.questions] == ["ceiling"]
    assert foamd.caps_posted == [2500] and len(fake.calls) == 2


def test_a_paused_model_call_declined_is_not_retried_as_a_conflict(tmp_path):
    """409 is a transient status for a model call (`TRANSIENT_STATUSES`): without this,
    a paused study would be asked five more times over four minutes."""
    foamd = FakeFoamd()
    loop, ctx, view, _ = _session(tmp_path, foamd, "/no")
    fake = PausedThenFine(foamd, message([text_block("never")]))
    loop.client = SimpleNamespace(messages=fake)
    loop.say("carry on")

    assert cli._run_turn(loop, view) is False
    assert len(fake.calls) == 1 and foamd.caps_posted == []
    assert loop.blocked_reason and "ceiling" in loop.blocked_reason


# -- 2c. a pause returned inside a Felix job -------------------------------------------


def _finished_paused_felix_job(ctx, backend, store, foamd):
    """A real Felix refusal's shape: exit 1, one ERROR line, then status JSON."""
    ctx.solver = solvers.FELIX
    cmd = "felix run cavity --gpu H100"
    job_id = backend.job_start(cmd, cwd="/work/study", name="cavity")
    store.record_job(job_id, cmd=cmd, name="cavity", cwd="/work/study")
    body = foamd.paused_body()
    log = (
        f"ERROR: {body['message']} (409 study_paused)\n"
        '{"state":"refused","status":409,"error":"study_paused",'
        f'"message":{json.dumps(body["message"])}' + "}\n"
    ).encode()
    backend.logs[job_id] = log
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, status="exited", name="cavity", end_reason="failed",
        exit_code=1, log_size=len(log),
    )
    return job_id


def test_a_felix_job_refused_at_the_ceiling_asks_and_can_be_raised(tmp_path, backend):
    """#79: the refusal is job output, not a BackendError caught by tools._handled."""
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    _loop, ctx, view, store = _session(tmp_path, foamd, "/yes", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)

    answer, is_error = dispatch(ctx, "job_check", {"job_id": job_id})

    assert not is_error
    assert [question[1] for question in view.questions] == ["ceiling"]
    assert foamd.caps_posted == [2000] and ctx.stopped is None
    assert "refused job is finished" in answer
    assert len(backend.started) == 1, "raising the cap does not silently submit a new solve"


def test_a_felix_job_refused_at_the_ceiling_stops_when_declined(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    _loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)

    _answer, is_error = dispatch(ctx, "job_check", {"job_id": job_id})

    assert not is_error
    assert [question[1] for question in view.questions] == ["ceiling"]
    assert foamd.caps_posted == []
    assert ctx.stopped and "1.6x" in ctx.stopped and "$" not in ctx.stopped

    dispatch(ctx, "job_check", {"job_id": job_id})
    assert len(view.questions) == 1, "the same job is put to the person only once"


def test_a_felix_refusal_still_matches_after_spend_moves_forward(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    _loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)
    foamd.spent_cents = 851

    _answer, is_error = dispatch(ctx, "job_check", {"job_id": job_id})

    assert not is_error
    assert [question[1] for question in view.questions] == ["ceiling"]
    assert "1.7x" in view.questions[0][3]
    assert "$" not in view.questions[0][3]


def test_the_question_uses_the_live_row_not_amounts_parsed_from_the_message(
    tmp_path, backend
):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    _loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)
    log = (
        b"ERROR: the refusal text changed and has no dollar amounts (409 study_paused)\n"
        b'{"state":"refused","status":409,"error":"study_paused"}\n'
    )
    backend.logs[job_id] = log
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, status="exited", name="cavity", end_reason="failed",
        exit_code=1, log_size=len(log),
    )

    dispatch(ctx, "job_check", {"job_id": job_id})

    assert [question[1] for question in view.questions] == ["ceiling"]
    assert "1.6x" in view.questions[0][3]
    assert "$" not in view.questions[0][3]
    assert store.session.jobs[job_id].ceiling_pause_handled is True


def test_an_unavailable_ledger_stays_eligible_until_it_can_be_read(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    _loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)
    foamd.spent_cents = None

    dispatch(ctx, "job_check", {"job_id": job_id})

    assert view.questions == []
    assert store.session.jobs[job_id].ceiling_pause_handled is False

    foamd.spent_cents = 803
    dispatch(ctx, "job_check", {"job_id": job_id})

    assert [question[1] for question in view.questions] == ["ceiling"]
    assert store.session.jobs[job_id].ceiling_pause_handled is True


def test_an_old_felix_pause_does_not_ask_again_after_the_ceiling_is_clear(
    tmp_path, backend
):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    _loop, ctx, view, store = _session(tmp_path, foamd, backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)
    foamd.cap_cents = 2000

    answer, is_error = dispatch(ctx, "job_check", {"job_id": job_id})

    assert not is_error and "study_paused" in answer
    assert "no longer paused at its ceiling" in answer
    assert view.questions == [] and ctx.stopped is None
    assert store.session.jobs[job_id].ceiling_pause_handled is True

    foamd.spent_cents = 1600
    dispatch(ctx, "job_check", {"job_id": job_id})
    assert view.questions == [], "the old refusal is not a later pause at the new cap"


def test_a_current_pause_under_a_different_cap_is_still_put_to_the_person(
    tmp_path, backend
):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    _loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)
    foamd.spent_cents = 1600
    foamd.cap_cents = 2000

    dispatch(ctx, "job_check", {"job_id": job_id})

    assert [question[1] for question in view.questions] == ["ceiling"]
    assert "3.2x" in view.questions[0][3] and "6.4x" in view.questions[0][3]
    assert "$" not in view.questions[0][3]


def test_a_finished_felix_pause_is_put_to_the_person_before_watch_wakes_the_model(
    tmp_path, backend
):
    """#79's ordinary path is watch mode, after the detached solve exits."""
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    _finished_paused_felix_job(ctx, backend, store, foamd)
    report = _collect_finished(backend, store, view)

    wake = cli._finished_job_wake(loop, view, report)

    assert wake is None
    assert [question[1] for question in view.questions] == ["ceiling"]
    assert loop.blocked_reason == ctx.stopped
    assert any(ctx.stopped in notice for notice in view.notices)
    assert any(ctx.stopped in str(message["content"]) for message in loop.messages)


def test_an_unhandled_finished_refusal_is_retried_before_the_next_turn(
    tmp_path, backend
):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, "/yes", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)
    foamd.spent_cents = None
    report = _collect_finished(backend, store, view)

    assert cli._finished_job_wake(loop, view, report) == report
    assert store.session.jobs[job_id].ceiling_pause_handled is False

    foamd.spent_cents = 800
    install_model(loop, [message([text_block("continuing")])])
    loop.say("continue")

    assert cli._run_turn(loop, view) is True
    assert [question[1] for question in view.questions] == ["ceiling"]
    assert store.session.jobs[job_id].ceiling_pause_handled is True
    assert foamd.caps_posted == [2000]


def test_a_raised_ceiling_is_in_the_watch_report_the_model_receives(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, "/yes", backend=backend)
    _finished_paused_felix_job(ctx, backend, store, foamd)
    report = _collect_finished(backend, store, view)

    wake = cli._finished_job_wake(loop, view, report)

    assert wake is not None and "ceiling was raised" in wake
    assert "409 study_paused" in wake
    assert foamd.caps_posted == [2000]


def test_watch_uses_the_finished_felix_facts_even_while_solver_is_auto(
    tmp_path, backend, monkeypatch
):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    _finished_paused_felix_job(ctx, backend, store, foamd)
    ctx.solver = solvers.AUTO
    finished = []
    report = _collect_finished(backend, store, view, finished)

    def unexpected_status(*_args, **_kwargs):
        raise AssertionError("watch already passed the finished status")

    def unexpected_tail(*_args, **_kwargs):
        raise AssertionError("watch already passed the parsed Felix tail")

    monkeypatch.setattr(backend, "job_status", unexpected_status)
    monkeypatch.setattr(backend, "job_tail", unexpected_tail)

    wake = cli._finished_job_wake(loop, view, report, tuple(finished))

    assert wake is None
    assert [question[1] for question in view.questions] == ["ceiling"]


def test_an_external_raise_is_explained_in_the_watch_report(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, backend=backend)
    _finished_paused_felix_job(ctx, backend, store, foamd)
    foamd.cap_cents = 2000
    finished = []
    report = _collect_finished(backend, store, view, finished)

    wake = cli._finished_job_wake(loop, view, report, tuple(finished))

    assert wake is not None and "no longer paused at its ceiling" in wake
    assert view.questions == [] and ctx.stopped is None


def test_unrelated_failed_output_cannot_trigger_the_ceiling_question(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, backend=backend)
    ctx.solver = solvers.FELIX
    job_id = backend.job_start("printf study_paused", cwd="/work", name="grep")
    store.record_job(job_id, cmd="printf study_paused", name="grep", cwd="/work")
    backend.logs[job_id] = b"409 study_paused\n"
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, status="exited", name="grep", exit_code=1, log_size=17,
    )
    report = _collect_finished(backend, store, view)

    assert cli._finished_job_wake(loop, view, report) == report
    assert view.questions == []


def test_felix_prose_without_a_rejected_final_status_cannot_ask(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, backend=backend)
    ctx.solver = solvers.FELIX
    cmd = "felix run cavity --gpu H100"
    job_id = backend.job_start(cmd, cwd="/work", name="cavity")
    store.record_job(job_id, cmd=cmd, name="cavity", cwd="/work")
    log = b"ERROR: notes mention 409 study_paused, but this is not a final status\n"
    backend.logs[job_id] = log
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, status="exited", name="cavity", exit_code=1,
        log_size=len(log),
    )
    report = _collect_finished(backend, store, view)

    assert cli._finished_job_wake(loop, view, report) == report
    assert view.questions == []


def test_a_felixd_rejection_is_not_foamd_refusing_the_study(tmp_path, backend):
    """`rejected` is felixd declining a created solve, not foamd's raw 409."""
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, backend=backend)
    cmd = "felix run cavity --gpu H100"
    job_id = backend.job_start(cmd, cwd="/work", name="cavity")
    store.record_job(job_id, cmd=cmd, name="cavity", cwd="/work")
    log = b'{"state":"rejected","status":409,"error":"study_paused"}\n'
    backend.logs[job_id] = log
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, status="exited", name="cavity", exit_code=1,
        log_size=len(log),
    )
    report = _collect_finished(backend, store, view)

    assert cli._finished_job_wake(loop, view, report) == report
    assert view.questions == []


def test_one_shot_returns_failed_when_the_person_stops_at_a_job_pause(
    tmp_path, backend, monkeypatch
):
    foamd = FakeFoamd(spent_cents=800, cap_cents=1000)
    loop, ctx, view, store = _session(tmp_path, foamd, "/no", backend=backend)
    job_id = _finished_paused_felix_job(ctx, backend, store, foamd)

    def finish(*_args, **_kwargs):
        status = backend.job_status(job_id)
        store.update_job(job_id, status=status.status, end_reason=status.end_reason,
                         exit_code=status.exit_code)
        return Wake("job", _job_report(backend, status, store.session.jobs[job_id]))

    monkeypatch.setattr(cli, "_run_turn", lambda *_args: True)
    monkeypatch.setattr(cli, "watch", finish)

    result = cli._run_one_shot(loop, backend, store, "solve", view,
                               ScriptedReader([]))

    assert result == "failed"
    assert len(view.questions) == 1


# -- 3. the line before the first solve --------------------------------------------------


def test_the_first_solve_is_preceded_by_how_far_the_study_may_go(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=150, cap_cents=1250, style="full")
    finishing_jobs(backend)
    loop, ctx, view, _ = _session(tmp_path, foamd, backend=backend)
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"}, "tu_1")],
                stop_reason="tool_use"),
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "again"}, "tu_2")],
                stop_reason="tool_use"),
        message([text_block("running")]),
    ])
    loop.say("solve it")
    loop.run()

    said = [n for n in view.notices if "normal" in n]
    assert said == ["This study will ask before it uses more than 2x a normal full study. "
                    "It has used 0.3x so far."], "said once, before the first solve only"
    assert "$" not in said[0]
    assert view.questions == [], "a message, not a question, in auto"
    assert [s["cmd"] for s in backend.started] == ["simpleFoam", "simpleFoam"]


def test_with_compute_approval_the_line_is_part_of_the_question(tmp_path, backend):
    foamd = FakeFoamd(spent_cents=0, cap_cents=500, style="quick")
    finishing_jobs(backend)
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes", mode="partial", backend=backend)
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})],
                stop_reason="tool_use"),
        message([text_block("running")]),
    ])
    loop.say("solve it")
    loop.run()

    (_, kind, _, detail, _), = view.questions
    assert kind == "job" and "more than 2x a normal quick study" in detail
    assert "$" not in detail and "so far" not in detail, "nothing used yet, nothing said"
    assert not [n for n in view.notices if "normal" in n]


def test_a_solve_against_an_older_service_works_the_multiple_out_from_the_cents(
    tmp_path, backend
):
    """A foamd without `pause_times_normal` is answered from the ceiling in force and
    the normals here, never from dollars: a 5000-cent full ceiling pauses at 8x."""
    foamd = FakeFoamd(spent_cents=0, cap_cents=5000, modern=False)
    finishing_jobs(backend)
    loop, ctx, view, _ = _session(tmp_path, foamd, backend=backend)
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})],
                stop_reason="tool_use"),
        message([text_block("running")]),
    ])
    loop.say("solve it")
    loop.run()

    line, = [n for n in view.notices if "normal" in n]
    assert line == "This study will ask before it uses more than 8x a normal full study."


def test_a_study_record_without_a_ceiling_falls_back_to_the_style_default():
    assert ceiling.default_cap_cents("quick") == 500
    assert ceiling.default_cap_cents("full") == 1250
    assert ceiling.default_cap_cents(None) == 1250
    assert ceiling.default_cap_cents("") == 1250
    for style in ("quick", "full", None):
        until = ceiling.default_cap_cents(style) * ceiling.PAUSE_AT / ceiling.normal_cents(style)
        assert until == 2.0, "every default study pauses at twice a normal one"
    row = {"style": "full"}
    said = ceiling.preamble(SimpleNamespace(ceiling=SimpleNamespace(study=lambda: row)))
    assert said == "This study will ask before it uses more than 2x a normal full study."


def test_the_job_tools_take_no_estimate_in_dollars():
    """`estimate_usd` existed to be shown to the person, and nothing in dollars is."""
    from openreynolds.tools import FELIX_TOOLS, TOOLS

    for tools in (TOOLS, FELIX_TOOLS):
        job = next(t for t in tools if t["name"] == "job_start")
        assert "estimate_usd" not in job["input_schema"]["properties"]


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
    foamd = FakeFoamd()
    foamd.spend_at_raise = 2100
    loop, ctx, view, _ = _session(tmp_path, foamd, "/yes", "/yes")
    install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("mesh it")
    loop.run()

    assert foamd.caps_posted == [2500, 5000]
    assert len(view.questions) == 2
    again = view.questions[1][3]
    assert "counts 4.2x the usage of a normal study" in again, \
        "says the service still counts it paused"
    assert "$" not in again
    assert foamd.execs == ["blockMesh", "blockMesh"], "no call went out into a known pause"
    assert [n for n in view.notices if "carrying on" in n] == [
        "The study can now run to 8x a normal full study before asking again; carrying on."]
    assert ctx.stopped is None


def test_the_preamble_reads_the_study_record_foamd_sends(tmp_path):
    """`spent_cents`, `times_normal` and `paused` are None when foamd cannot read the
    ledgers; then nothing is said about what has been used."""
    foamd = FakeFoamd(spent_cents=None, cap_cents=5000, style="full")
    ctx = SimpleNamespace(ceiling=ceiling.Ceiling(client=foamd.client(), study_id=STUDY))
    said = ceiling.preamble(ctx)
    assert said == "This study will ask before it uses more than 8x a normal full study."

    foamd = FakeFoamd(spent_cents=1050, cap_cents=1250)
    ctx = SimpleNamespace(ceiling=ceiling.Ceiling(client=foamd.client(), study_id=STUDY))
    said = ceiling.preamble(ctx)
    assert "It has used 2.1x so far." in said and "It is paused now" in said, \
        "a study already paused says so before the solve meets it"
    assert "$" not in said


@pytest.mark.parametrize("modern", [True, False])
def test_no_dollar_sign_reaches_the_person_before_the_first_solve(modern):
    """Whatever the record holds, current or from a service before `times_normal`,
    paused or not, nothing in the line is money."""
    for spent, cap, style in [(0, 1250, "full"), (1050, 1250, "full"), (450, 500, "quick"),
                              (None, 5000, "")]:
        foamd = FakeFoamd(spent_cents=spent, cap_cents=cap, style=style, modern=modern)
        ctx = SimpleNamespace(ceiling=ceiling.Ceiling(client=foamd.client(), study_id=STUDY))
        said = ceiling.preamble(ctx)
        assert said and "$" not in said, said
    unreadable = SimpleNamespace(ceiling=SimpleNamespace(study=lambda: None))
    assert "$" not in ceiling.preamble(unreadable)


def test_the_preamble_names_no_ceiling_for_an_exempt_study():
    """OFI#81: a staff study was told "This study's ceiling is $10.00; it pauses at
    $8.00", a pause that never applies to it. foamd's `exempt` says so, and an exempt
    study is told nothing."""
    row = {"style": "quick", "cap_cents": 500, "spent_cents": 24, "paused": False, "exempt": True}
    ctx = SimpleNamespace(ceiling=SimpleNamespace(study=lambda: row))
    assert ceiling.preamble(ctx) is None
    ctx = SimpleNamespace(ceiling=SimpleNamespace(study=lambda: {**row, "exempt": False}))
    said = ceiling.preamble(ctx)
    assert said == ("This study will ask before it uses more than 2x a normal quick study. "
                    "It has used 0.1x so far."), "a customer's still says it"
    assert ceiling.preamble(ctx) is None, "said once"


# -- 6. a resumed study whose workspace start met the ceiling -----------------------------


class _Bar:
    """The progress bar's two workspace calls, recorded."""

    def __init__(self):
        self.said = []

    def workspace_starting(self, eta_s=0.0):
        self.said.append("starting")

    def workspace_ready(self):
        self.said.append("ready")


def _resumed_start(tmp_path, monkeypatch, foamd, *answers):
    """A resume's start as `session` makes it: the study named on the client before
    the start, a `PendingBackend` the tools hold, and `_bring_up` settling it."""
    monkeypatch.setattr(cli, "_sync_toolbox_timed", lambda *a, **k: None)
    client = foamd.client()
    client.study_id = STUDY
    starter = hosted.Starter(client, "inst-1")
    starter.start()
    pending = starter.pending()
    loop, ctx, view, store = _session(tmp_path, foamd, *answers, backend=pending)
    bar = _Bar()
    cli._bring_up(pending, starter, store, view, loop, bar, None, True)
    return loop, ctx, view, pending, bar


def test_a_resumed_start_paused_at_its_ceiling_starts_again_once_it_is_raised(
    tmp_path, monkeypatch
):
    """foamd refuses a resumed study's workspace start with `409 study_paused`. The
    first tool call puts the pause to the person; yes raises the ceiling, the start is
    sent again, and the call runs on the workspace that comes -- not on the start's
    stored refusal, asked three times and then every call after it."""
    foamd = FakeFoamd()
    foamd.start_pauses = True
    loop, ctx, view, pending, bar = _resumed_start(tmp_path, monkeypatch, foamd, "/yes")
    assert pending.failed and foamd.starts == 1, "the start met the pause"

    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([tool_block("bash", {"cmd": "checkMesh"})], stop_reason="tool_use"),
        message([text_block("meshed")]),
    ])
    loop.say("carry on")
    loop.run()

    assert len(view.questions) == 1, "asked once"
    assert foamd.caps_posted == [2500], "and raised once"
    assert foamd.starts == 2, "the start went again after the raise"
    assert pending.ready() and not pending.failed
    assert [c for c in foamd.execs if c in ("blockMesh", "checkMesh")] == ["blockMesh", "checkMesh"]
    results = [block for m in loop.messages if m["role"] == "user"
               and isinstance(m["content"], list) for block in m["content"]
               if isinstance(block, dict) and block.get("type") == "tool_result"]
    assert len(results) == 2
    for result in results:
        assert not result.get("is_error") and "ran" in str(result["content"])
    assert ctx.stopped is None and ctx.on_ceiling_raised is None
    assert bar.said == ["ready", "starting", "ready"], "the bar was told it was coming again"
    assert len(fake.calls) == 3


def test_a_start_refused_for_another_reason_is_not_started_again(tmp_path, monkeypatch):
    """Only the pause is mended by a raise: any other refusal of the start stands, and
    nothing is set to start it again."""
    foamd = FakeFoamd(spent_cents=0)
    real = foamd.handler

    def no_capacity(request):
        if request.url.path.endswith("/start"):
            foamd.starts += 1
            return httpx.Response(402, json={"error": "allowance_exhausted",
                                             "message": "this period's allowance is used up."})
        return real(request)

    foamd.handler = no_capacity
    loop, ctx, view, pending, _bar = _resumed_start(tmp_path, monkeypatch, foamd)
    assert pending.failed and pending.error.code == "allowance_exhausted"
    assert ctx.on_ceiling_raised is None
    assert "until the session is resumed" in loop._posted[-1], "said as it always was"
    with pytest.raises(BackendError):
        pending.exec("ls")
    assert foamd.starts == 1
