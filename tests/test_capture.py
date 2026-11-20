"""Capture is invisible to the model and can never be in its way."""

from __future__ import annotations

import threading
import time

import pytest

from openreynolds import capture as capture_mod
from openreynolds.backend.base import BackendError
from openreynolds.capture import _CONTENT_CAP, Capture, _cap_content

from conftest import message, text_block
from test_refused_start import _Platform, _rows, _session


class FakeClient:
    def __init__(self, fail_times: int = 0, block: threading.Event | None = None):
        self.messages: list = []
        self.results: list = []
        self.artifacts: list = []
        self.studies: list = []
        self.attempts = 0
        self._fail_times = fail_times
        self._block = block

    def _maybe_fail(self):
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise RuntimeError("platform unreachable")

    def create_study(self, title, instance_id):
        self.studies.append((title, instance_id))
        self._maybe_fail()
        return "remote-study-1"

    def post_messages(self, study_id, messages):
        if self._block:
            self._block.wait(timeout=5)
        self._maybe_fail()
        self.messages.extend(messages)

    def post_result(self, study_id, payload):
        self._maybe_fail()
        self.results.append(payload)

    def post_artifact(self, study_id, filename, data, kind=None):
        self._maybe_fail()
        self.artifacts.append((filename, data, kind))


def drain(capture: Capture) -> None:
    capture.close(timeout=5)


def test_messages_results_and_artifacts_all_arrive(tmp_path):
    client = FakeClient()
    capture = Capture(client, "remote-1")
    png = tmp_path / "centerline.png"
    png.write_bytes(b"\x89PNG")

    capture.message(0, "user", "validate the solver")
    capture.artifact(png, kind="validation-plot")
    capture.result({"passed": True})
    drain(capture)

    assert client.messages == [{"seq": 0, "role": "user", "content": "validate the solver"}]
    assert client.artifacts == [("centerline.png", b"\x89PNG", "validation-plot")]
    assert client.results == [{"passed": True}]


def test_an_upload_is_attempted_once_and_not_repeated(tmp_path):
    """Retrying belongs to the transport, which knows what is safe to send twice.

    This worker cannot know: it holds an opaque callable, so its only move is to run
    the whole post again, and a post that had already been carried out appends the
    row a second time. Absorbing a blip is `FoamdClient.request`'s job and it is
    tested there (`test_hosted.py`); what is tested here is that nothing is added on
    top of it.
    """
    client = FakeClient(fail_times=1)
    capture = Capture(client, "remote-1")
    capture.message(0, "user", "hello")
    drain(capture)

    assert client.attempts == 1
    assert client.messages == []


def test_a_persistent_failure_is_dropped_with_a_warning():
    warnings: list[str] = []
    client = FakeClient(fail_times=99)
    capture = Capture(client, "remote-1", warn=warnings.append)

    capture.message(0, "user", "hello")
    capture.message(1, "assistant", "hi")
    capture.close(timeout=5)

    assert client.messages == []
    assert any("dropped 2 item(s)" in w for w in warnings)
    assert any("local mirror is complete" in w for w in warnings)


def test_the_warning_names_what_was_lost(tmp_path):
    """F-44. The count answered the wrong question: a dropped message leaves a gap
    in the transcript on the web, a dropped artifact leaves a picture that exists on
    the laptop and nowhere else, and "1 item" cannot tell them apart."""
    warnings: list[str] = []
    client = FakeClient(fail_times=99)
    capture = Capture(client, "remote-1", warn=warnings.append)
    png = tmp_path / "mesh_z.png"
    png.write_bytes(b"png-bytes")

    capture.message(41, "assistant", "hello")
    capture.artifact(png, kind="render")
    capture.result({"passed": True})
    capture.close(timeout=5)

    said = "; ".join(warnings)
    assert "message seq 41 (assistant)" in said
    assert "artifact mesh_z.png" in said
    assert "the end-of-study result" in said


def test_a_long_outage_is_counted_by_kind_rather_than_listed():
    """Past a handful the names stop being readable, and the kinds are what is left
    to act on. The first is still named, because it dates the outage."""
    warnings: list[str] = []
    capture = Capture(FakeClient(fail_times=99), "remote-1", warn=warnings.append)
    for seq in range(20):
        capture.message(seq, "assistant", "x")
    capture.close(timeout=5)

    said = warnings[0]
    assert "dropped 20 item(s)" in said
    assert "20 message(s)" in said
    assert "first was message seq 0 (assistant)" in said
    assert "seq 7" not in said, "the middle of a long outage is summarised, not listed"


def test_nothing_dropped_says_nothing():
    warnings: list[str] = []
    Capture(FakeClient(), "remote-1", warn=warnings.append).close(timeout=5)
    assert warnings == []


def test_submitting_never_blocks_the_caller():
    """The study must not wait on the platform, even when it hangs."""
    gate = threading.Event()
    capture = Capture(FakeClient(block=gate), "remote-1")

    start = time.monotonic()
    for seq in range(50):
        capture.message(seq, "user", "x")
    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    gate.set()
    drain(capture)


def test_close_is_bounded_when_the_platform_hangs():
    """A hung upload cannot hold the process open."""
    gate = threading.Event()
    capture = Capture(FakeClient(block=gate), "remote-1")
    capture.message(0, "user", "x")

    start = time.monotonic()
    capture.close(timeout=0.2)
    assert time.monotonic() - start < 2.0

    gate.set()


def test_capture_start_declines_rather_than_raising():
    warnings: list[str] = []
    assert Capture.start(FakeClient(fail_times=1), "t", "inst-1", warn=warnings.append) is None
    assert any("capture off" in w for w in warnings)


def test_capture_start_returns_a_live_capture():
    client = FakeClient()
    capture = Capture.start(client, "elbow study", "inst-1")
    assert capture is not None
    assert capture.study_id == "remote-study-1"
    assert client.studies == [("elbow study", "inst-1")]
    drain(capture)


def test_long_content_is_capped_for_upload():
    """The local mirror keeps the full text; the platform gets a sane size."""
    capped = _cap_content({"output": "x" * (_CONTENT_CAP + 5_000), "tool": "bash"})
    assert len(capped["output"]) == _CONTENT_CAP
    assert capped["tool"] == "bash"


def test_capping_walks_nested_structures():
    capped = _cap_content(["a" * (_CONTENT_CAP + 1), {"b": "c"}, 42, None])
    assert len(capped[0]) == _CONTENT_CAP
    assert capped[1:] == [{"b": "c"}, 42, None]


# -- the whole path, as the incident ran it --------------------------------------------


def test_one_recorded_message_reaches_the_service_exactly_once():
    """The incident, end to end: one reply recorded, three rows in the transcript.

    A `job_check` reply from a 3D transient run appeared three times in the captured
    transcript, minutes apart, all under message seq 197 — and five other messages of
    the same study twice. The agent produced that text once: `Store.append_message`
    increments a counter before it returns, so two records can never share a seq, and
    the reply's own `running_for=` and `[waited …s]` were computed when the string was
    built. What repeated was the upload. The service takes `seq` from this client and
    its transcript table indexes `(study_id, seq)` without a unique constraint, so a
    post that had already been carried out and whose answer was lost wrote the row
    again on every repeat — and there were up to fifteen: three here on top of five in
    `FoamdClient.request`.

    Driven through the real client so both layers are in the test, because either one
    alone is enough to produce the duplicate.
    """
    from openreynolds.backend.hosted import FoamdClient
    import httpx2 as httpx

    posted: list[bytes] = []

    def commits_then_loses_the_answer(request: httpx.Request) -> httpx.Response:
        posted.append(request.read())
        raise httpx.ReadTimeout("the row was written; the answer never came back")

    client = FoamdClient("https://svc.example", "of_live_test")
    client._client = httpx.Client(
        base_url="https://svc.example",
        transport=httpx.MockTransport(commits_then_loses_the_answer),
    )

    capture = Capture(client, "study-1")
    capture.message(197, "tool", {"tool": "job_check", "output": "status=running"})
    capture.close(timeout=5)

    assert len(posted) == 1, (
        f"one recorded message, {len(posted)} posts — each one appends another row "
        "under the same seq, and nothing downstream can tell them apart"
    )


# -- a study on this machine's workspace ----------------------------------------------
#
# `--local` and a native run used to upload nothing at all. `cli.session` leaves the
# service's client None for a workspace it does not reserve, and the capture block
# below it read that client, so the transcript, the renders, the results payload and
# the case bundle of every local study stayed on one laptop -- while `--no-capture`,
# the switch that is supposed to ask for exactly that, was never touched.


class Unreachable(_Platform):
    """A service a machine with no network cannot open a study on."""

    def create_study(self, title, instance_id, **extra):
        self.created.append((title, instance_id, extra))
        raise BackendError("cannot reach the service: getaddrinfo failed",
                           code="unreachable")


@pytest.mark.parametrize("kind", ["native", "docker"])
def test_a_study_on_this_machines_workspace_opens_a_row_and_posts_its_transcript(
        monkeypatch, tmp_path, kind):
    run = _session(monkeypatch, tmp_path, [message([text_block("meshed and solved")])],
                   one_shot="cavity at Re 100", capture=True, local=kind)

    assert len(run.platform.created) == 1, "one row, opened once"
    title, instance, extra = run.platform.created[0]
    assert title == run.store.session.title
    assert not instance, "there is no instance to bind a local study to"
    assert extra["study_id"] == run.store.session.study_id
    assert extra["home"] == run.store.session.home
    assert run.store.session.remote_study_id == "remote-1", (
        "recorded, so the next session of this study posts to the row it already has"
    )
    said = [str(m["content"]) for m in run.platform.posted]
    assert any("cavity at Re 100" in s for s in said), said
    assert any("meshed and solved" in s for s in said), said


def test_no_capture_keeps_a_local_study_off_the_service(monkeypatch, tmp_path):
    run = _session(monkeypatch, tmp_path, [message([text_block("done")])],
                   one_shot="cavity", capture=False, local="native")

    assert run.platform.created == []
    assert run.platform.posted == []


def test_a_local_study_with_no_network_says_so_and_finishes_anyway(monkeypatch, tmp_path):
    """The machine most likely to be running locally is the one that may be offline."""
    run = _session(monkeypatch, tmp_path, [message([text_block("solved")])],
                   one_shot="cavity", capture=True, local="native",
                   platform=Unreachable())

    assert run.outcome == "ok"
    assert "capture off" in run.said, run.said
    assert run.store.session.remote_study_id == ""
    assert any("solved" in str(r["content"]) for r in _rows(run.store)), (
        "the local record is complete whatever the service did"
    )


def test_a_row_that_opens_late_still_gets_what_was_held():
    """The close used to abandon a row that was one POST away from existing.

    A local row is opened on a thread (`cli._capture_locally`) so an offline machine
    does not wait a minute to be told what it knew. A `-p` run answered in a second
    then ends with that thread still in flight, and abandoning it there posts
    nothing and says nothing -- for a local study, its whole transcript.
    """
    client = FakeClient()
    capture = Capture.deferred()
    capture.message(0, "user", "validate the solver")

    def open_row() -> None:
        time.sleep(0.05)
        capture.open(client, "t", "")

    capture.opening = threading.Thread(target=open_row, daemon=True)
    capture.opening.start()
    capture.close(timeout=5)

    assert client.messages == [{"seq": 0, "role": "user", "content": "validate the solver"}]


def test_a_row_that_never_opens_is_said_rather_than_waited_for(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(capture_mod, "_OPENING_WAIT_S", 0.05)
    hold = threading.Event()
    capture = Capture.deferred(warn=warnings.append)
    capture.message(0, "user", "x")
    capture.opening = threading.Thread(target=lambda: hold.wait(5), daemon=True)
    capture.opening.start()

    start = time.monotonic()
    capture.close(timeout=1)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, "an exit is not held for a service that is not answering"
    assert any("stays on this machine" in w for w in warnings), warnings
    hold.set()
