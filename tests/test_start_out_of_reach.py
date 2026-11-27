"""A workspace start that could not reach the service.

Measured in production on 2026-11-22: ten seconds into a session the start call met a
504 from the load balancer while the service was restarting. The start was not tried
again, the person was shown the gateway's HTML page as the reason, every later message
was held back "until the session is resumed", and the second message got no answer in
the transcript at all. The person left.

Four changes, one test group each:
1. an HTML error page is read as its title, not passed on as markup;
2. the start is sent again for a while when the service could not be reached;
3. if it still fails, the person's next message sends it again;
4. a later message held back by a start that failed gets an answer in the transcript.
"""

from __future__ import annotations

import httpx2 as httpx
import pytest

from openreynolds import cli
from openreynolds.backend import hosted as hosted_mod
from openreynolds.backend.base import BackendError
from openreynolds.backend.hosted import FoamdClient, Starter, _decode_error, out_of_reach

from conftest import FakeBackend, ReadyStarter, message, text_block
from test_refused_start import CAVITY, PLAN_LIMIT, _refused_starter, _rows, _session

GATEWAY_504 = ("<html>\r\n<head><title>504 Gateway Time-out</title></head>\r\n<body>\r\n"
               "<center><h1>504 Gateway Time-out</h1></center>\r\n</body>\r\n</html>")


def _gateway_error(status=504, title="504 Gateway Time-out"):
    return BackendError(title, code="http_error", status=status)


# -- 1. an HTML error page is its title -----------------------------------------------


def test_a_gateway_page_is_read_as_its_title():
    error = _decode_error(httpx.Response(504, text=GATEWAY_504))
    assert error.code == "http_error" and error.status == 504
    assert error.message == "504 Gateway Time-out"
    assert str(error) == "http_error (504): 504 Gateway Time-out"


def test_a_page_without_a_title_is_its_text_and_plain_text_is_left_alone():
    page = _decode_error(httpx.Response(502, text="<html><body><h1>Bad</h1> gateway</body></html>"))
    assert page.message == "Bad gateway"
    plain = _decode_error(httpx.Response(500, text="x < y > z went wrong"))
    assert plain.message == "x < y > z went wrong"


# -- 2. the start is sent again while the service is out of reach -----------------------


def _starter_against(answers, monkeypatch):
    """A Starter whose service answers each start with the next of `answers` (a status,
    or a `(status, json)` pair); counts the starts sent."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    sent = []

    def handler(request):
        assert request.url.path == "/v1/instances/inst-1/start"
        sent.append(request)
        answer = answers[min(len(sent), len(answers)) - 1]
        if answer in (200, "joined"):
            return httpx.Response(200, json={"id": "inst-1", "status": "running",
                                             "started_new": answer == 200})
        if answer == "timeout":
            raise httpx.ReadTimeout("the gateway held the request", request=request)
        if isinstance(answer, tuple):
            return httpx.Response(answer[0], json=answer[1])
        return httpx.Response(answer, text=GATEWAY_504)

    client = FoamdClient("https://svc.example", "of_live_test")
    client._client = httpx.Client(base_url="https://svc.example",
                                  transport=httpx.MockTransport(handler),
                                  follow_redirects=False)
    return Starter(client, "inst-1"), sent


def test_a_start_that_met_a_gateway_error_is_sent_again_and_comes_up(monkeypatch):
    starter, sent = _starter_against([504, 502, 200], monkeypatch)
    backend = starter.result(timeout=5)
    assert backend.instance_id == "inst-1"
    assert len(sent) == 3


def test_a_start_that_never_gets_through_gives_up_with_the_gateways_words(monkeypatch):
    starter, sent = _starter_against([504], monkeypatch)
    with pytest.raises(BackendError) as raised:
        starter.result(timeout=5)
    assert len(sent) == 1 + len(hosted_mod.START_RETRY_DELAYS_S)
    assert str(raised.value) == "http_error (504): 504 Gateway Time-out"


def test_a_start_whose_answer_was_lost_is_still_the_sessions_to_stop(monkeypatch):
    """The 504 may have come while foamd was booting the workspace for this very
    start, so the start that gets through finds it running (`started_new: false`).
    Believed, the session would leave its own workspace up at End."""
    starter, sent = _starter_against([504, "joined"], monkeypatch)
    backend = starter.result(timeout=5)
    assert len(sent) == 2
    assert backend.was_already_running is False, "the listing said it was stopped"


def test_a_start_sent_again_on_the_next_message_still_distrusts_the_lost_answer(monkeypatch):
    """The resend minutes later finds the workspace up because this session's own
    504'd start booted it; `started_new: false` is not believed then either."""
    starter, sent = _starter_against([504] * 5 + ["joined"], monkeypatch)
    with pytest.raises(BackendError):
        starter.result(timeout=5)
    assert starter.restart()
    assert starter.result(timeout=5).was_already_running is False
    assert len(sent) == 6


def test_a_refusal_does_not_make_the_answer_untrusted(monkeypatch):
    starter, _sent = _starter_against(
        [(409, {"error": "study_paused", "message": "paused"}), "joined"], monkeypatch)
    with pytest.raises(BackendError):
        starter.result(timeout=5)
    assert starter._lost_answer is False


def test_the_starter_stops_resending_past_its_budget(monkeypatch):
    monkeypatch.setattr(hosted_mod, "START_RETRY_BUDGET_S", 1.0)
    starter, sent = _starter_against([504], monkeypatch)
    with pytest.raises(BackendError):
        starter.result(timeout=5)
    assert len(sent) == 1, "the first wait would already go past the budget"


def test_a_start_that_joined_at_once_is_still_believed(monkeypatch):
    starter, _sent = _starter_against(["joined"], monkeypatch)
    assert starter.result(timeout=5).was_already_running is True


def test_a_start_that_timed_out_is_not_resent_by_the_starter(monkeypatch):
    """Each timeout is the client's whole 180 s; the next message sends it again."""
    starter, sent = _starter_against(["timeout", 200], monkeypatch)
    with pytest.raises(BackendError) as raised:
        starter.result(timeout=5)
    assert raised.value.code == "timeout" and out_of_reach(raised.value)
    assert len(sent) == 1


def test_a_refused_start_is_not_sent_again(monkeypatch):
    starter, sent = _starter_against(
        [(403, {"error": "plan_limit", "message": "the Starter plan runs 1 study at a time"})],
        monkeypatch)
    with pytest.raises(BackendError) as raised:
        starter.result(timeout=5)
    assert raised.value.code == "plan_limit"
    assert len(sent) == 1


def test_what_counts_as_out_of_reach():
    assert out_of_reach(_gateway_error(504))
    assert out_of_reach(_gateway_error(502))
    assert out_of_reach(BackendError("refused", code="unreachable"))
    assert out_of_reach(BackendError("slow", code="timeout"))
    assert not out_of_reach(PLAN_LIMIT)
    assert not out_of_reach(BackendError("paused", code="study_paused", status=409))
    assert not out_of_reach(BackendError("x", code="http_error", status=404))
    assert not out_of_reach(BackendError("x", code="http_error", status=500))
    assert not out_of_reach(ValueError("not a backend error"))


# -- 3. the next message sends it again -------------------------------------------------


class FlakyStarter(ReadyStarter):
    """A start that fails with each of `errors` in turn and then comes up; `restart`
    as `hosted.Starter` has it."""

    def __init__(self, backend, errors):
        super().__init__(backend, "iid-9", error=errors[0])
        self.errors = list(errors[1:])
        self.starts = 1

    def restart(self):
        self.starts += 1
        self.error = self.errors.pop(0) if self.errors else None
        return True


def test_a_message_starts_the_workspace_again_and_is_answered(monkeypatch, tmp_path):
    """The start failed before the person's first message; that message sends it
    again, and once the workspace comes the agent answers it."""
    starter = FlakyStarter(FakeBackend(), [_gateway_error()])
    run = _session(monkeypatch, tmp_path, [message([text_block("the cavity, meshed")])],
                   lines=[CAVITY], starter=starter)

    assert starter.starts == 2, "sent again when the person spoke"
    assert len(run.calls) == 1, "and the message was answered"
    assert run.loop.start_refusal is None


def test_the_next_message_starts_it_again_when_this_ones_start_failed(monkeypatch, tmp_path):
    starter = FlakyStarter(FakeBackend(), [_gateway_error(), _gateway_error()])
    run = _session(monkeypatch, tmp_path, [message([text_block("the cavity, meshed")])],
                   lines=[CAVITY, "are you there?"], starter=starter)

    assert starter.starts == 3, "once with the first message, once with the second"
    assert len(run.calls) == 1, "the second message was answered"
    events = [str(r["content"]) for r in _rows(run.store) if r["role"] == "event"]
    first = next(e for e in events if "could not be started" in e)
    assert "504 Gateway Time-out" in first and "<html>" not in first
    assert "sent again when the person sends their next message" in first
    assert "until the session is resumed" not in first
    assert any("starting the workspace again" in n for n in run.view.notices)
    assert run.loop.start_refusal is None
    # Each failure is followed by a start said as starting, so an MCP caller does not
    # read "failed" (and end the study) while the new start comes up (#287).
    states = run.view.workspace_states
    assert states[-1] == "ready" and all(
        b == "starting" for a, b in zip(states, states[1:]) if a == "failed"), states


def test_a_start_that_fails_again_is_sent_once_per_message(monkeypatch, tmp_path):
    starter = FlakyStarter(FakeBackend(), [_gateway_error()] * 5)
    run = _session(monkeypatch, tmp_path, [message([text_block("no")])],
                   lines=[CAVITY, "hello?", "anyone?"], starter=starter)

    assert starter.starts == 4, "the first start, then once for each message, no more"
    assert run.calls == []
    users = [r["content"] for r in _rows(run.store) if r["role"] == "user"]
    assert users == [CAVITY, "hello?", "anyone?"]


# -- 4. a later message held back is answered in the transcript -------------------------


def test_every_message_held_back_gets_a_line_in_the_transcript(monkeypatch, tmp_path):
    run = _session(monkeypatch, tmp_path, [message([text_block("no")])],
                   lines=[CAVITY, "continue", "and now?"], starter=_refused_starter())

    rows = _rows(run.store)
    said = [str(r["content"]) for r in rows if r["role"] == "event"]
    refusals = [e for e in said if "could not be started" in e]
    assert len(refusals) == 3, "the start's own line, then one for each later message"
    assert sum("not sent to the agent" in e for e in refusals) == 2
    order = [(r["role"], str(r["content"])[:20]) for r in rows if r["role"] in ("user", "event")]
    for i, (role, _text) in enumerate(order):
        if role == "user" and i + 1 < len(order):
            assert order[i + 1][0] == "event", f"message {i} was answered: {order}"
