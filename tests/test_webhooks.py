"""`openreynolds webhook`: a study's events posted from this machine, and nothing else.

Every request here goes to a `http.server` this file opens on a loopback port, as
`tests/test_serve.py` does. No test reaches a real webhook endpoint: a test that posted
to a Slack URL would be a test that notified somebody's channel every time CI ran.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from openreynolds import webhooks


class Receiver(BaseHTTPRequestHandler):
    """A webhook endpoint: records the bytes and the headers, answers with the status
    the test asked for."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        length = int(self.headers.get("Content-Length") or 0)
        # `self.headers` rather than a dict of it: urllib sends `X-Openreynolds-...`
        # because `Request.add_header` capitalises, and a case-sensitive dict would make
        # this test assert the casing of a header HTTP says is case-insensitive.
        self.server.received.append(  # type: ignore[attr-defined]
            SimpleNamespace(body=self.rfile.read(length), headers=self.headers,
                            path=self.path))
        status = self.server.answer_with  # type: ignore[attr-defined]
        self.send_response(status)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def endpoint():
    listener = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    listener.received = []  # type: ignore[attr-defined]
    listener.answer_with = 200  # type: ignore[attr-defined]
    thread = threading.Thread(target=listener.serve_forever, kwargs={"poll_interval": 0.02},
                              daemon=True, name="webhook-test")
    thread.start()
    yield SimpleNamespace(listener=listener,
                          url=f"http://127.0.0.1:{listener.server_port}/hook",
                          received=listener.received)  # type: ignore[attr-defined]
    listener.shutdown()
    listener.server_close()
    thread.join(10)


STREAM = [
    json.dumps({"v": 1, "type": "session_start", "study": "st-1", "model": "claude-opus-5"}),
    json.dumps({"v": 1, "type": "text", "study": "st-1", "text": "meshing"}),
    json.dumps({"v": 1, "type": "tool", "study": "st-1", "name": "bash"}),
    json.dumps({"v": 1, "type": "session_end", "study": "st-1", "outcome": "ok"}),
]
"""Four lines in `jsonview`'s shape: the two boundaries and two of the narration events
between them (`openreynolds/jsonview.py:153,169,206,220`)."""


def relay(lines, urls, **kw):
    out = io.StringIO()
    log = io.StringIO()
    tally = webhooks.relay(iter(lines), urls, out=out, log=log, **kw)
    return SimpleNamespace(tally=tally, out=out.getvalue(), log=log.getvalue())


# -- what is sent ------------------------------------------------------------------


def test_only_the_events_worth_a_notification_are_sent(endpoint):
    """`text` fires per 240 characters and `tool` on every call. A channel that carries
    those is a channel nobody reads, so the default is the four in `DEFAULT_EVENTS`."""
    result = relay(STREAM, [endpoint.url])

    sent = [json.loads(r.body)["type"] for r in endpoint.received]
    assert sent == ["session_start", "session_end"]
    assert result.tally == {"lines": 4, "sent": 2, "failed": 0, "skipped": 2}


def test_asking_for_every_event_sends_every_event(endpoint):
    relay(STREAM, [endpoint.url], events=None)

    assert len(endpoint.received) == 4


def test_a_named_event_is_sent_even_when_it_is_not_a_default(endpoint):
    relay(STREAM, [endpoint.url], events=["tool"])

    assert [json.loads(r.body)["type"] for r in endpoint.received] == ["tool"]


def test_the_stream_is_passed_through_unchanged(endpoint):
    """It is a filter in a pipe. Taking the user's stream away in exchange for a
    notification is not a trade anybody asked for."""
    result = relay(STREAM, [endpoint.url])

    assert result.out.splitlines() == STREAM


def test_a_line_that_is_not_json_is_passed_through_and_not_posted(endpoint):
    """`jsonview` writes whole lines under a lock because "a half-written line is not a
    parse error a reader can recover from" (`jsonview.py:17-26`). This reader is not the
    one the stream exists for, so it declines to guess at one."""
    result = relay(["not json at all", STREAM[3]], [endpoint.url])

    assert "not json at all" in result.out
    assert len(endpoint.received) == 1


def test_a_very_long_line_is_passed_through_and_not_posted(endpoint):
    """A `message` event carries a whole model turn and has no documented ceiling.
    A notification is a sentence."""
    huge = json.dumps({"type": "session_end", "study": "st-1",
                       "outcome": "ok", "pad": "x" * (webhooks.MAX_LINE_BYTES + 10)})

    result = relay([huge], [endpoint.url], events=None)

    assert huge in result.out
    assert endpoint.received == []


def test_two_urls_each_get_the_event(endpoint):
    relay([STREAM[3]], [endpoint.url, endpoint.url])

    assert len(endpoint.received) == 2


# -- the shapes --------------------------------------------------------------------


def test_slack_gets_a_sentence_in_the_field_it_reads(endpoint):
    relay(STREAM, [endpoint.url], shape="slack")

    bodies = [json.loads(r.body) for r in endpoint.received]
    assert bodies[1]["text"] == "Study (st-1) ended: ok"
    assert bodies[1]["event"]["outcome"] == "ok", "the numbers travel beside the sentence"


def test_discord_gets_the_same_sentence_under_its_own_key(endpoint):
    relay([STREAM[3]], [endpoint.url], shape="discord")

    assert json.loads(endpoint.received[0].body)["content"] == "Study (st-1) ended: ok"


def test_a_failed_outcome_says_what_failed():
    """The reason is in the sentence, because a channel showing only "failed" sends the
    reader back to the terminal they were trying not to watch."""
    text = webhooks.summarise({"type": "session_end", "study": "st-2",
                               "outcome": "unreachable", "error": "no workspace service"})

    assert text == "Study (st-2) ended: unreachable -- no workspace service"


def test_an_outcome_nobody_listed_arrives_as_itself():
    assert "outcome-from-the-future" in webhooks.summarise(
        {"type": "session_end", "outcome": "outcome-from-the-future"})


def test_an_event_type_nobody_listed_still_makes_a_sentence():
    assert webhooks.summarise({"type": "weather", "study": "st-3"}) == "Study (st-3) weather"


# -- signing -----------------------------------------------------------------------


def test_the_signature_is_over_the_bytes_that_were_delivered(endpoint):
    """Signing a re-serialisation of the payload is the classic way to ship a signature
    that fails for every receiver who checks it properly: two JSON encoders do not agree
    on whitespace."""
    relay([STREAM[3]], [endpoint.url], secret="s3cret")

    got = endpoint.received[0]
    expected = hmac.new(b"s3cret", got.body, hashlib.sha256).hexdigest()
    assert got.headers[webhooks.SIGNATURE_HEADER] == f"sha256={expected}"


def test_nothing_is_signed_without_a_secret(endpoint):
    relay([STREAM[3]], [endpoint.url])

    assert endpoint.received[0].headers.get(webhooks.SIGNATURE_HEADER) is None


# -- failure -----------------------------------------------------------------------


def test_a_refused_endpoint_does_not_stop_the_stream(endpoint):
    endpoint.listener.answer_with = 503

    result = relay(STREAM, [endpoint.url])

    assert result.out.splitlines() == STREAM, "every line still went out"
    assert result.tally["failed"] == 2 and result.tally["sent"] == 0
    assert "HTTP 503" in result.log


def test_an_unreachable_endpoint_does_not_raise():
    """One attempt, no retries, no queue. A reader that must not miss an event reads
    the stream, which is still on stdout."""
    result = relay(STREAM, ["http://127.0.0.1:1/hook"], secret="k")

    assert result.out.splitlines() == STREAM
    assert result.tally["failed"] == 2


def test_the_failure_line_does_not_print_the_url(endpoint):
    """A Slack or Discord webhook URL is the credential, and this stderr goes into a CI
    log that is often public."""
    endpoint.listener.answer_with = 500

    result = relay([STREAM[3]], [endpoint.url + "?token=the-actual-secret"])

    assert "the-actual-secret" not in result.log
    assert "127.0.0.1" in result.log, "the host is still named, so it can be diagnosed"


def test_a_torn_url_is_still_reported_rather_than_raising():
    result = relay([STREAM[3]], ["not-a-url-at-all"])

    assert result.tally["failed"] == 1
    assert result.out.splitlines() == [STREAM[3]]


# -- the subcommand ----------------------------------------------------------------


def test_the_subcommand_relays_the_stream_on_stdin(monkeypatch, endpoint):
    from click.testing import CliRunner

    from openreynolds import cli

    result = CliRunner().invoke(cli.main, ["webhook", endpoint.url],
                                input="\n".join(STREAM) + "\n")

    assert result.exit_code == 0, result.output
    assert len(endpoint.received) == 2


def test_the_subcommand_takes_its_secret_from_the_environment(monkeypatch, endpoint):
    from click.testing import CliRunner

    from openreynolds import cli

    monkeypatch.setenv(webhooks.SECRET_ENV, "from-the-env")
    CliRunner().invoke(cli.main, ["webhook", endpoint.url], input=STREAM[3] + "\n")

    body = endpoint.received[0].body
    expected = hmac.new(b"from-the-env", body, hashlib.sha256).hexdigest()
    assert endpoint.received[0].headers[webhooks.SIGNATURE_HEADER] == f"sha256={expected}"


def test_the_subcommand_exits_nonzero_when_nothing_got_through(endpoint):
    from click.testing import CliRunner

    from openreynolds import cli

    endpoint.listener.answer_with = 500

    result = CliRunner().invoke(cli.main, ["webhook", endpoint.url], input=STREAM[3] + "\n")

    assert result.exit_code == 1
