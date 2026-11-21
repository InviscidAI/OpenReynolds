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

    # The dropped line, found by what it says rather than by its position: `close`
    # now says what it is about to send before it waits, so a session with a long
    # transcript and a slow link does not look like it has hung on exit, and that
    # notice is warnings[0].
    said = next(w for w in warnings if "dropped" in w)
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


# -- what never went, and why -------------------------------------------------
#
# These came out of one real local study, 20261121-033555-a8ea, run on a laptop with no
# OpenFOAM. The local mirror held 18 messages; production held 10, the eight absent
# being seqs 0, 1, 3, 4, 5, 6, 8 and 16; and the session printed `capture dropped 8
# item(s): 8 message(s)`. So the COUNT was right and nothing was lost in silence there
# -- what was missing was the reason, and posting two of those rows again by hand
# succeeded at once, which is the difference between a transient failure and a refusal
# and is the whole of what a person needs to know.
#
# `_unposted` is a second hole, found by reading `close` rather than by watching it: a
# queue still unfinished when the join timeout expires was abandoned without appearing
# in the number. The test below holds it. No production run has been seen to hit it,
# and this comment says so rather than borrowing the study above as evidence for it.


def test_items_still_queued_when_the_budget_runs_out_are_counted(monkeypatch):
    """They used to be lost in silence. `close` reported `_dropped`, which counts
    exceptions, so a queue abandoned at the join timeout did not appear in the number
    at all -- and that is the LOCAL study's normal case, not an edge case: its row is
    opened on a thread at session start, every message waits on that, and a short
    session reaches `close` with its whole transcript still queued."""
    warnings: list[str] = []
    held = threading.Event()
    capture = Capture(FakeClient(block=held), "remote-1", warn=warnings.append)
    monkeypatch.setattr(capture_mod, "_DRAIN_PER_ITEM_S", 0.0)
    for seq in range(5):
        capture.message(seq, "assistant", "x")

    capture.close(timeout=0.2)        # the worker is stuck on the first post
    held.set()

    said = next(w for w in warnings if "dropped" in w)
    assert "still queued when the session ended" in said
    assert "openreynolds capture" in said, "and what to do about it"
    counted = int(said.split("dropped ")[1].split(" item")[0])
    assert counted >= 4, f"only {counted} of 5 accounted for: {said}"


def test_the_first_failures_reason_is_said_not_only_counted():
    """`capture dropped 8 item(s): 8 message(s)` cannot tell a refusal from a dropped
    connection, and that is the first thing anyone debugging a gap needs."""
    warnings: list[str] = []
    capture = Capture(FakeClient(fail_times=99), "remote-1", warn=warnings.append)
    capture.message(0, "assistant", "x")
    capture.close(timeout=5)

    said = next(w for w in warnings if "dropped" in w)
    assert "first failure was RuntimeError: platform unreachable" in said


def test_the_drain_budget_grows_with_what_is_queued():
    """One item is one POST and the worker is serial, so a flat budget is a budget for
    a queue of a known size, and there is no such thing. 18 messages did not fit in
    the old flat 10 s and 16 of them were lost."""
    warnings: list[str] = []
    slow = FakeClient()
    capture = Capture(slow, "remote-1", warn=warnings.append)
    for seq in range(30):
        capture.message(seq, "assistant", "x")
    capture.close(timeout=1.0)

    assert len(slow.messages) == 30, "a full queue drained inside its own budget"
    assert not [w for w in warnings if "dropped" in w]
    said = next(w for w in warnings if "sending" in w)
    assert "sending 30 item(s)" in said, said


def test_the_budget_is_still_bounded_however_much_is_queued(monkeypatch):
    """The module's first promise is that nothing here can delay or fail a study, so
    an unreachable service must not hang an exit no matter how long the transcript."""
    monkeypatch.setattr(capture_mod, "_DRAIN_CEILING_S", 0.5)
    held = threading.Event()
    capture = Capture(FakeClient(block=held), "remote-1", warn=lambda _: None)
    for seq in range(500):
        capture.message(seq, "assistant", "x")

    began = time.monotonic()
    capture.close(timeout=0.1)
    waited = time.monotonic() - began
    held.set()

    assert waited < 5, f"close waited {waited:.1f}s against a 0.5s ceiling"


# -- sending again, by asking first -------------------------------------------


class _Holding:
    """A service that already has some seqs, and records what else it is given."""

    def __init__(self, held, refuse=()):
        self._held = sorted(held)
        self._refuse = set(refuse)
        self.posted: list[int] = []

    def list_messages(self, study_id, after=-1, limit=999, newest=False):
        rows = [{"seq": s, "role": "assistant"} for s in self._held if s > after]
        return {"messages": rows[:limit], "more": len(rows) > limit}

    def post_messages(self, study_id, messages):
        for m in messages:
            if m["seq"] in self._refuse:
                raise RuntimeError(f"seq {m['seq']} refused")
            self.posted.append(m["seq"])
            self._held.append(m["seq"])

    def close(self):
        pass


def _rows_for(seqs):
    return [{"seq": s, "role": "assistant", "content": "x", "at": "t"} for s in seqs]


def test_only_the_seqs_the_service_says_it_lacks_are_sent():
    """Which is what makes sending them safe at all. Capture never retries, because
    the service indexes `(study_id, seq)` without making it unique and a repeat of a
    post whose outcome is unknown appends the row twice. A reconciliation does not
    share that problem: it asks first."""
    client = _Holding(held=[0, 2, 7])
    local = _rows_for(range(8))

    gap = capture_mod.missing_from_service(client, "remote-1", local)

    assert [row["seq"] for row in gap] == [1, 3, 4, 5, 6]
    sent, refused = capture_mod.resend(client, "remote-1", gap)
    assert (sent, refused) == (5, [])
    assert client.posted == [1, 3, 4, 5, 6], "and nothing it already had"


def test_a_study_the_service_has_whole_sends_nothing():
    client = _Holding(held=range(5))
    gap = capture_mod.missing_from_service(client, "remote-1", _rows_for(range(5)))
    assert gap == []
    assert capture_mod.resend(client, "remote-1", gap) == (0, [])
    assert client.posted == []


def test_one_refused_row_does_not_take_the_rest_of_the_gap_with_it():
    """Posted one at a time for this reason -- the opposite trade from the live path,
    where a burst is what matters."""
    client = _Holding(held=[], refuse={2})
    gap = _rows_for(range(4))

    sent, refused = capture_mod.resend(client, "remote-1", gap)

    assert sent == 3
    assert client.posted == [0, 1, 3]
    assert refused == ["seq 2: RuntimeError: seq 2 refused"]


def test_a_long_transcript_is_asked_for_page_by_page():
    """`list_messages` caps at 999 and says `more`, so a transcript past one page has
    to be walked or the gap is computed against a truncated answer and rows the
    service already has get posted again -- the duplicate this is built to avoid."""
    client = _Holding(held=range(2500))
    seen = capture_mod.missing_from_service(client, "remote-1", _rows_for(range(2500)))
    assert seen == []


# -- the command ---------------------------------------------------------------


def _wired(monkeypatch, tmp_path, client, *, remote="remote-1", signed_in=True):
    """`openreynolds capture` against a study on disk and a stand-in service."""
    from click.testing import CliRunner

    from openreynolds import cli
    from openreynolds.store import Store

    store = Store(tmp_path / "studies", "20260101-000000-aaaa")
    store.session.remote_study_id = remote
    store.save()
    for seq in range(5):
        store.append_message("assistant", f"row {seq}")

    class _Cfg:
        studies_dir = tmp_path / "studies"
        foamd_url = "https://example.invalid" if signed_in else ""
        foamd_api_key = "key" if signed_in else ""

    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls: _Cfg()))
    monkeypatch.setattr(cli, "_the_study", lambda cfg, chosen: "20260101-000000-aaaa")
    monkeypatch.setattr(cli.hosted, "FoamdClient", lambda *a, **k: client)
    return CliRunner().invoke(cli.main, ["capture", "--study", "20260101-000000-aaaa"])


def test_the_command_sends_only_the_gap_and_says_what_it_sent(monkeypatch, tmp_path):
    client = _Holding(held=[0, 3])
    result = _wired(monkeypatch, tmp_path, client)

    assert result.exit_code == 0, result.output
    assert "5 row(s) here, 2 on the service" in result.output
    assert "missing seq: 1, 2, 4" in result.output
    assert "sent 3" in result.output
    assert client.posted == [1, 2, 4]


def test_the_command_sends_nothing_with_dry_run(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from openreynolds import cli

    client = _Holding(held=[0])
    _wired(monkeypatch, tmp_path, client)          # wires Config and the client
    # ...and sent the gap, so the service now holds everything. Put it back to one
    # row, which is the state a dry run is for.
    client._held = [0]
    client.posted.clear()
    result = CliRunner().invoke(cli.main, ["capture", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "missing seq" in result.output
    assert client.posted == [], "--dry-run named the gap and sent it"


def test_a_study_with_no_row_on_the_service_is_refused_not_opened(monkeypatch, tmp_path):
    """Nothing here can open a row: the id is the service's to accept at creation, so
    a study that ran with capture off has nothing to reconcile against."""
    client = _Holding(held=[])
    result = _wired(monkeypatch, tmp_path, client, remote="")

    assert result.exit_code == 1
    assert "no row on the service" in result.output
    assert client.posted == []


def test_a_machine_that_is_not_signed_in_is_told_so(monkeypatch, tmp_path):
    result = _wired(monkeypatch, tmp_path, _Holding(held=[]), signed_in=False)

    assert result.exit_code == 1
    assert "not signed in" in result.output
