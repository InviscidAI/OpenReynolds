"""The disk under a running job is filling: the agent hears it while the job runs.

A solve runs detached, so the agent is usually not running a command when its disk
fills -- it is holding a `job_check`, or its turn is over and the harness is watching.
The service puts a note on the running job's status; the agent is woken with it once
per disk size, and a held `job_check` ends early to deliver it.
"""
from __future__ import annotations

import pytest

from openreynolds import disknote
from openreynolds.backend.base import JobStatus
from openreynolds.tools import dispatch
from openreynolds.watch import watch

from conftest import ScriptedReader  # noqa: E402

pytestmark = pytest.mark.usefixtures("fast_polling")

NOTE = ("The workspace disk is 84% full (15.0 GB left for files) while this job runs. "
        "Past 90% it is grown once, to 200 GB, and not again.")


@pytest.fixture(autouse=True)
def _fresh():
    disknote._told.clear()
    yield
    disknote._told.clear()


def _filling(job_id, volume_gb=100, note=NOTE):
    return JobStatus(job_id=job_id, name="solve", status="running", log_size=10,
                     disk={"share": 0.84, "free_gb": 15.0, "volume_gb": volume_gb, "note": note})


def _start(backend, store):
    job_id = backend.job_start("pimpleFoam", name="solve")
    store.record_job(job_id, cmd="pimpleFoam", name="solve")
    return job_id


def test_the_harness_wakes_the_agent_when_the_disk_under_a_job_fills(backend, store, view):
    job_id = _start(backend, store)
    backend.jobs[job_id] = _filling(job_id)
    wake = watch(backend, store, view, ScriptedReader([]), narrate_every_s=0)
    assert wake.kind == "disk"
    assert "disk under job solve is filling" in wake.text and NOTE in wake.text
    assert "still running" in wake.text


def test_once_per_disk_size_not_every_poll(backend, store, view):
    job_id = _start(backend, store)
    backend.jobs[job_id] = _filling(job_id)
    assert watch(backend, store, view, ScriptedReader([])).kind == "disk"
    again = watch(backend, store, view, ScriptedReader(["what now?"]))
    assert again.kind == "user", "the same note does not wake it twice"
    backend.jobs[job_id] = _filling(job_id, volume_gb=200, note="It is as large as it is grown.")
    backend.jobs[job_id].disk["disk_gb"] = 210
    assert watch(backend, store, view, ScriptedReader([])).kind == "disk", (
        "after the grow, filling again is news")


def test_no_note_no_wake(backend, store, view):
    job_id = _start(backend, store)
    backend.jobs[job_id] = _filling(job_id, note=None)
    assert watch(backend, store, view, ScriptedReader(["hi"])).kind == "user"


def test_job_check_says_it_and_a_held_wait_ends_to_deliver_it(ctx, backend, store, view):
    dispatch(ctx, "job_start", {"cmd": "pimpleFoam"})
    backend.jobs["job-1"] = _filling("job-1")
    content, _ = dispatch(ctx, "job_check", {"job_id": "job-1", "wait_s": 300})
    assert "[disk] " + NOTE in content
    assert "[waited 300s]" not in content, "the wait ended early"
    assert watch(backend, store, view, ScriptedReader(["ok"])).kind == "user", (
        "seen in job_check, so the harness does not wake for it again")


def test_the_service_sending_nothing_is_no_disk():
    from openreynolds.backend.hosted import _job_status
    assert _job_status({"job_id": "j", "status": "running"}).disk is None
    st = _job_status({"job_id": "j", "status": "running", "disk": {"share": 0.5, "note": None}})
    assert st.disk_note is None and st.disk["share"] == 0.5


def test_a_grow_reaching_the_volume_before_the_files_is_not_a_new_disk(backend, store, view):
    """Keyed on the filesystem's size: the volume says 200 a step before the files have it."""
    job_id = _start(backend, store)
    backend.jobs[job_id] = _filling(job_id)
    backend.jobs[job_id].disk["disk_gb"] = 105
    assert watch(backend, store, view, ScriptedReader([])).kind == "disk"
    backend.jobs[job_id].disk["volume_gb"] = 200
    assert watch(backend, store, view, ScriptedReader(["hi"])).kind == "user"


def test_a_command_promoted_to_a_job_does_not_cut_its_wait_for_a_note_it_cannot_show(ctx, backend):
    from openreynolds import tools
    backend.jobs["job-9"] = _filling("job-9")
    began = __import__("time").monotonic()
    tools._hold(ctx, "job-9", backend.jobs["job-9"], 0.5)
    assert __import__("time").monotonic() - began >= 0.4, "only job_check ends its wait on a note"
