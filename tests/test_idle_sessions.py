"""A session nobody is using stops claiming the workspace.

The mirror lists the workspace every twenty seconds for as long as the session
process is alive. Against a hosted backend each listing was an ordinary exec, and
an ordinary exec is a statement that the workspace is in use -- so the service's
idle reaper never fired and a compute instance billed on until its 24-hour ceiling.
In production that was ~150 listings an hour continuing twelve to twenty-one hours
past the last thing the person typed, on three separate accounts; one of them had
sent a single message and was billed for a full day.

The fix is one bit on the wire. What is checked here is that the bit is set on
exactly the calls that are polls, that it is *not* set on the calls that are work,
and that an idle answer is read as "nothing is running" rather than as "the
workspace is empty" -- which is the reading that would have the mirror delete or
re-fetch everything.
"""

from __future__ import annotations

import httpx2 as httpx
import pytest

from openreynolds import mirror
from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.backend.hosted import FoamdClient, HostedBackend
from openreynolds.browse import Browser

HOME = "/work/study-test"

IDLE = ExecResult(exit_code=-1, output="", truncated=False, log_path=None,
                  stderr="", idle=True)


# --- the wire -------------------------------------------------------------------

def _backend_recording_bodies(bodies: list, result: dict) -> HostedBackend:
    def handler(request):
        import json
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=result)

    client = FoamdClient("https://svc.example", "of_live_test")
    client._client = httpx.Client(base_url="https://svc.example",
                                  transport=httpx.MockTransport(handler),
                                  follow_redirects=True)
    return HostedBackend(client, "inst-1")


def test_a_poll_says_so_on_the_wire_and_ordinary_work_does_not():
    bodies: list = []
    backend = _backend_recording_bodies(bodies, {"exit_code": 0, "output": "ok"})

    backend.exec("find /work", background=True)
    backend.exec("blockMesh")

    assert bodies[0]["background"] is True
    assert "background" not in bodies[1], (
        "work must look exactly as it always did, so an older service sees the "
        "request it has always seen")


def test_an_idle_answer_is_not_a_successful_empty_one():
    """`exit_code 0` with no output would mean 'the workspace is empty', which a
    mirror acts on. Nothing ran, and the result has to say so."""
    backend = _backend_recording_bodies([], {"exit_code": None, "output": "", "idle": True})

    result = backend.exec("find /work", background=True)

    assert result.idle is True
    assert result.exit_code != 0 and result.output == ""


# --- the listing ----------------------------------------------------------------

def test_a_background_listing_of_a_stopped_workspace_refuses_to_guess(backend, store):
    """Not an empty tree, and no fallback to list_dir -- that fallback goes through
    a route that would start the very workspace this listing declined to start."""
    backend.exec_result = IDLE
    backend.dirs[HOME] = ["should-never-be-read"]

    with pytest.raises(BackendError) as caught:
        Browser(backend, store, home=HOME).tree(HOME, background=True)

    assert caught.value.code == "workspace_idle"
    assert len(backend.execs) == 1, "one listing, and no fallback behind it"


def test_an_ordinary_listing_is_not_marked_a_poll(backend, store):
    Browser(backend, store, home=HOME).tree(HOME)
    assert backend.exec_background == [False]


# --- the mirror -----------------------------------------------------------------

def test_the_unattended_cycle_marks_itself_a_poll(backend, store, monkeypatch):
    """The thread's own cycle, including a poked one: whatever provoked a poke was a
    real command a moment earlier, and that is what keeps the workspace alive."""
    thread = mirror.LiveMirror(Browser(backend, store, home=HOME), interval_s=20.0)

    thread._cycle(background=True)

    assert backend.exec_background == [True]


def test_a_sync_somebody_asked_for_is_not_a_poll(backend, store):
    """`sync_now` is foreground by definition -- something wanted it -- so it may
    start a workspace and does keep it alive."""
    thread = mirror.LiveMirror(Browser(backend, store, home=HOME), interval_s=20.0)

    thread.sync_now()

    assert backend.exec_background == [False]


def test_an_idle_cycle_is_silent(backend, store):
    """The expected steady state of an idle session, not a fault. Saying 'could not
    look at the workspace' every twenty seconds for a night would bury the warnings
    that mean something."""
    backend.exec_result = IDLE

    report = mirror.sync(Browser(backend, store, home=HOME), live=True, background=True)

    assert report.warnings == []
    assert report.pulled == []


def test_a_real_failure_is_still_reported(backend, store, monkeypatch):
    """The control: only `workspace_idle` is quiet. Anything else the mirror could
    not do still reaches the report."""
    def broken(cmd, cwd=None, timeout_s=120, *, background=False):
        raise BackendError("instance is not up", code="unavailable")

    backend.exec = broken

    report = mirror.sync(Browser(backend, store, home=HOME), live=True, background=True)

    assert report.warnings and "could not look at" in report.warnings[0]
