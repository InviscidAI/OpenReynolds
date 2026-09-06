"""Capture is invisible to the model and can never be in its way."""

from __future__ import annotations

import threading
import time

from openreynolds.capture import _CONTENT_CAP, Capture, _cap_content


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
