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


def stopped(backend, answer_once_started: ExecResult) -> None:
    """A workspace the service has stopped: a poll finds nothing running, and ordinary
    work starts a machine and gets its answer. The two are told apart by the bit."""

    def exec(cmd, cwd=None, timeout_s=120, *, background=False):
        backend.execs.append(cmd)
        backend.exec_background.append(background)
        return IDLE if background else answer_once_started

    backend.exec = exec


def test_an_ordinary_listing_is_asked_as_a_poll_first_and_is_work_only_when_it_must_be(
    backend, store
):
    """A foreground listing used to be an ordinary exec, and on a hosted workspace the
    service had already stopped an ordinary exec starts a machine: the close-down of
    an idle-timed-out session did exactly that on 2026-09-21, eighteen minutes of
    c7i.2xlarge for one `find`. So the listing polls first. On a workspace that is up
    the poll *is* the listing and nothing else is sent; only when nothing is running,
    and the backend has no copy of the workspace to read (`test_store_listing.py` has
    the copy), is it sent again as work -- the one case where starting the machine is
    what the person asked for."""
    Browser(backend, store, home=HOME).tree(HOME)
    assert backend.exec_background == [True], "up: the poll answered, and that was all"

    backend.exec_background.clear()
    stopped(backend, ExecResult(0, f"-\t9\t1700000000.0\t{HOME}/notes.md\n", False, None))
    listing = Browser(backend, store, home=HOME).tree(HOME)
    assert backend.exec_background == [True, False], "stopped, no copy: the poll, then the work"
    assert [e.path for e in listing] == [f"{HOME}/notes.md"], "and the work's answer is used"


# --- the mirror -----------------------------------------------------------------

def test_the_unattended_cycle_marks_itself_a_poll(backend, store, monkeypatch):
    """The thread's own cycle, including a poked one: whatever provoked a poke was a
    real command a moment earlier, and that is what keeps the workspace alive."""
    thread = mirror.LiveMirror(Browser(backend, store, home=HOME), interval_s=20.0)

    thread._cycle(background=True)

    assert backend.exec_background == [True]


def test_a_sync_somebody_asked_for_may_still_start_the_workspace(backend, store):
    """`sync_now` is foreground by definition -- something wanted it -- so it may
    start a workspace. Its listing is asked as a poll first, like every foreground
    listing now, and the start happens only when the poll finds nothing running and
    there is no copy of the workspace to read instead."""
    thread = mirror.LiveMirror(Browser(backend, store, home=HOME), interval_s=20.0)

    thread.sync_now()
    assert backend.exec_background == [True], "up: listed by the poll, nothing started"

    backend.exec_background.clear()
    stopped(backend, ExecResult(0, "", False, None))
    thread.sync_now()
    assert backend.exec_background == [True, False], "stopped, no copy: started, as before"


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
