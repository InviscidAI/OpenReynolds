from __future__ import annotations

import time

import pytest

from openreynolds.backend.base import ExecResult, JobStatus
from openreynolds.backend.base import WORKSPACE_ROOT
from openreynolds.tools import TOOLS, ToolContext, dispatch


def test_tool_list_is_deterministic():
    """Tool order is prefix position 0 for the cache, so it stays sorted and fixed."""
    names = [tool["name"] for tool in TOOLS]
    assert names == sorted(names)
    assert names == [
        "bash",
        "fetch",
        "job_check",
        "job_kill",
        "job_start",
        "read_file",
        "request_workspace_size",
        "write_file",
    ]
    for tool in TOOLS:
        assert tool["input_schema"]["type"] == "object"
        assert tool["description"]


def test_unknown_tool_is_an_error_not_a_crash(ctx):
    content, is_error = dispatch(ctx, "run_gate", {})
    assert is_error
    assert "No such tool" in content


def test_bash_reports_exit_code(ctx, backend):
    backend.exec_result = ExecResult(3, "boom", False, None)
    content, is_error = dispatch(ctx, "bash", {"cmd": "false"})
    assert not is_error
    assert "exit_code: 3" in content
    assert "boom" in content


def test_bash_clamps_and_passes_through_timeout(ctx, backend):
    dispatch(ctx, "bash", {"cmd": "sleep 1", "timeout_s": 42, "cwd": "/work/case"})
    assert backend.last_exec == ("sleep 1", "/work/case", 42)


def test_truncation_marker_points_at_the_tail(ctx, backend):
    """The service returns the head of a long log, so the marker offers the far end."""
    backend.exec_result = ExecResult(0, "H" * 5000, True, "/work/.foamd/exec/abc.log")
    backend.files["/work/.foamd/exec/abc.log"] = b"L" * 1_000_000

    content, is_error = dispatch(ctx, "bash", {"cmd": "solve"})

    assert not is_error
    assert "[truncated" in content
    assert "of 1000000 shown" in content
    assert "/work/.foamd/exec/abc.log" in content
    assert "offset=996000" in content  # 1_000_000 - TAIL_HINT_BYTES
    # our own cap applied on top of the service's
    assert len(content.encode()) < 5000


def test_truncation_marker_without_a_total(ctx, backend):
    backend.exec_result = ExecResult(0, "H" * 5000, True, "/work/.foamd/exec/abc.log")
    content, _ = dispatch(ctx, "bash", {"cmd": "solve"})
    assert "read_file with an offset" in content


def test_our_cap_alone_still_marks(ctx, backend):
    backend.exec_result = ExecResult(0, "x" * 4000, False, None)
    content, _ = dispatch(ctx, "bash", {"cmd": "cat big"})
    assert "[truncated" in content


def test_write_then_read_round_trip(ctx, backend):
    dispatch(ctx, "write_file", {"path": "/work/a.txt", "content": "hello"})
    assert backend.files["/work/a.txt"] == b"hello"

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/a.txt"})
    assert not is_error
    assert "bytes 0–5 of 5" in content
    assert content.endswith("hello")


def test_read_file_window_reports_what_remains(ctx, backend):
    backend.files["/work/big"] = b"0123456789"
    content, _ = dispatch(ctx, "read_file", {"path": "/work/big", "offset": 2, "limit": 3})
    assert "bytes 2–5 of 10" in content
    assert "5 bytes remain past this window" in content
    assert content.endswith("234")


def test_read_file_on_a_directory_lists_it(ctx, backend):
    backend.dirs["/work/case"] = ["0", "constant", "system"]
    content, is_error = dispatch(ctx, "read_file", {"path": "/work/case"})
    assert not is_error
    assert "directory, 3 entries" in content
    assert "constant" in content


def test_missing_path_is_a_fact_not_a_traceback(ctx):
    content, is_error = dispatch(ctx, "read_file", {"path": "/work/nope"})
    assert is_error
    assert "not_found" in content


def test_job_start_records_it_locally(ctx, backend, store):
    """The service has no list-jobs endpoint, so a resume depends on this record."""
    content, is_error = dispatch(
        ctx, "job_start", {"cmd": "simpleFoam", "name": "solve", "kill_on": ["FOAM FATAL"]}
    )
    assert not is_error
    job_id = backend.started and "job-1"
    assert job_id in store.session.jobs
    record = store.session.jobs[job_id]
    assert record.name == "solve"
    assert record.cmd == "simpleFoam"
    assert backend.started[0]["kill_on"] == ["FOAM FATAL"]


def test_job_check_advances_the_offset(ctx, backend, store):
    dispatch(ctx, "job_start", {"cmd": "simpleFoam"})
    backend.logs["job-1"] = b"line one\nline two\n"

    first, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})
    assert "line one" in first
    assert store.session.jobs["job-1"].log_offset == 18

    backend.logs["job-1"] += b"line three\n"
    second, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})
    assert "line three" in second
    assert "line one" not in second


def test_job_check_surfaces_the_kill_on_line(ctx, backend):
    dispatch(ctx, "job_start", {"cmd": "simpleFoam", "kill_on": ["FOAM FATAL"]})
    backend.jobs["job-1"] = JobStatus(
        job_id="job-1",
        status="killed",
        end_reason="kill_on_match",
        killed_by="--> FOAM FATAL ERROR: keyword nu is undefined",
    )
    content, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})
    assert "end_reason=kill_on_match" in content
    assert "keyword nu is undefined" in content


def test_job_kill_updates_the_record(ctx, store):
    dispatch(ctx, "job_start", {"cmd": "simpleFoam"})
    content, _ = dispatch(ctx, "job_kill", {"job_id": "job-1"})
    assert "status=killed" in content
    assert store.session.jobs["job-1"].end_reason == "killed_by_client"


def test_fetch_writes_locally_and_notifies(ctx, backend, store):
    seen: list = []
    ctx.on_fetch = seen.extend
    backend.files["/work/case/renders/u.png"] = b"\x89PNG"

    content, is_error = dispatch(ctx, "fetch", {"paths": ["/work/case/renders/u.png"]})

    assert not is_error
    assert "copied 1 file" in content
    # The workspace shape is preserved, so two cases' renders cannot collide.
    assert (store.fetch_dir() / "case" / "renders" / "u.png").read_bytes() == b"\x89PNG"
    assert len(seen) == 1


def test_fetch_of_nothing_is_not_an_error(ctx):
    content, is_error = dispatch(ctx, "fetch", {"paths": []})
    assert not is_error
    assert "nothing was copied" in content


def test_job_start_output_is_parseable_by_the_smoke_script(ctx):
    """scripts/smoke.py reads the id out of this line; pin the shape."""
    content, _ = dispatch(ctx, "job_start", {"cmd": "simpleFoam", "name": "ticker"})
    assert content.split()[:2] == ["started", "job"]
    assert content.split()[2] == "job-1"


def test_an_over_long_timeout_is_reported_not_silently_clamped(ctx, backend):
    """A command cut off at a ceiling the caller did not know about reads as one
    that finished."""
    content, _ = dispatch(ctx, "bash", {"cmd": "simpleFoam", "timeout_s": 600})
    assert "exceeds the 300s ceiling" in content
    assert "job_start has no such limit" in content


def test_an_ordinary_timeout_says_nothing(ctx):
    content, _ = dispatch(ctx, "bash", {"cmd": "ls", "timeout_s": 60})
    assert "ceiling" not in content


def test_a_timed_out_command_explains_its_minus_one(ctx, backend):
    """A bare `exit_code: -1` reads like a command that merely produced nothing.
    Found by a live run, where the model had to re-issue the query to find out."""
    backend.exec_result = ExecResult(-1, "", False, None)
    content, _ = dispatch(ctx, "bash", {"cmd": "sleep 280", "timeout_s": 120})
    assert "exit_code -1 means no exit status was reported" in content
    assert "ran with 120s" in content
    assert "job_start has no time limit" in content


def test_an_ordinary_exit_says_nothing_about_timeouts(ctx, backend):
    backend.exec_result = ExecResult(0, "fine", False, None)
    content, _ = dispatch(ctx, "bash", {"cmd": "ls"})
    assert "exit_code -1" not in content


# -- a study works in its own directory ----------------------------------------


def test_a_command_runs_in_the_study_s_own_directory(ctx):
    """Otherwise every study's relative paths land in the same shared heap."""
    ctx.home = "/work/20260824-120000-abcd"

    dispatch(ctx, "bash", {"cmd": "ls"})

    assert ctx.backend.last_exec[1] == "/work/20260824-120000-abcd"


def test_an_explicit_directory_still_wins(ctx):
    ctx.home = "/work/mine"
    dispatch(ctx, "bash", {"cmd": "ls", "cwd": "/work/somewhere-else"})
    assert ctx.backend.last_exec[1] == "/work/somewhere-else"


def test_a_job_starts_in_the_study_s_own_directory_too(ctx):
    ctx.home = "/work/mine"
    dispatch(ctx, "job_start", {"cmd": "simpleFoam", "name": "solve"})
    assert ctx.backend.started[-1]["cwd"] == "/work/mine"


def test_the_default_is_the_whole_workspace_when_no_home_is_set(ctx):
    """Studies made before studies had a directory of their own keep what they had."""
    dispatch(ctx, "bash", {"cmd": "ls"})
    assert ctx.backend.last_exec[1] == WORKSPACE_ROOT


# -- how long things took ------------------------------------------------------


def test_a_slow_command_says_how_long_it_took(ctx, monkeypatch):
    """A four-minute command and a two-second one read identically otherwise, so the
    cost of what was just done is invisible to whoever chose to do it."""
    clock = iter([0.0, 42.0])
    monkeypatch.setattr("openreynolds.tools.time.monotonic", lambda: next(clock))

    out, _ = dispatch(ctx, "bash", {"cmd": "blockMesh"})

    assert "[took 42s]" in out


def test_a_quick_command_is_not_cluttered_with_a_duration(ctx, monkeypatch):
    clock = iter([0.0, 0.3])
    monkeypatch.setattr("openreynolds.tools.time.monotonic", lambda: next(clock))

    out, _ = dispatch(ctx, "bash", {"cmd": "ls"})

    assert "took" not in out


def test_a_running_job_says_how_long_it_has_been_running(ctx):
    """Two hours in and one minute in are the same line otherwise, and which of those
    it is changes what anyone would do about it."""
    ctx.backend.jobs["job-1"] = JobStatus(
        job_id="job-1",
        status="running",
        name="solve",
        started_at=time.time() - 3600,
    )
    ctx.store.record_job("job-1", cmd="simpleFoam", name="solve")

    out, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})

    assert "running_for=60." in out


def test_a_finished_job_says_how_long_it_ran(ctx):
    ctx.backend.jobs["job-1"] = JobStatus(
        job_id="job-1",
        status="exited",
        name="solve",
        exit_code=0,
        started_at="2026-08-24T10:00:00Z",
        ended_at="2026-08-24T10:03:00Z",
    )
    ctx.store.record_job("job-1", cmd="simpleFoam", name="solve")

    out, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})

    assert "ran_for=3.0min" in out


def test_a_job_with_no_timestamps_says_nothing_about_duration(ctx):
    """The service does not always send them, and a made-up number is worse than none."""
    ctx.backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")
    ctx.store.record_job("job-1", cmd="simpleFoam", name="solve")

    out, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})

    assert "running_for" not in out and "ran_for" not in out


# -- job_check can wait ---------------------------------------------------------


def test_job_check_waits_for_the_job_to_end(ctx, monkeypatch):
    """The model paced itself with `sleep` in bash, which tripped the bash cap and
    filled the transcript with timeout noise. Waiting is the harness's job."""
    import time as _time

    from openreynolds import tools as tools_mod
    from openreynolds.backend.base import JobStatus
    from openreynolds.tools import dispatch as _dispatch

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")
    calls = {"n": 0}
    real = ctx.backend.job_status

    def finishing(jid):
        calls["n"] += 1
        if calls["n"] >= 3:
            ctx.backend.jobs[jid] = JobStatus(
                job_id=jid, name="solve", status="exited", exit_code=0,
                end_reason="completed", log_size=0,
            )
        return real(jid)

    ctx.backend.job_status = finishing

    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id, "wait_s": 5})

    assert not is_error
    assert "exited" in out
    assert "waited" in out


def test_the_wait_ends_early_when_the_user_speaks(ctx, monkeypatch):
    import time as _time

    from openreynolds import tools as tools_mod
    from openreynolds.tools import dispatch as _dispatch

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")
    ctx.on_wait_input = lambda: True

    began = _time.monotonic()
    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id, "wait_s": 30})

    assert not is_error
    assert _time.monotonic() - began < 5, "it did not sit out the full wait"
    assert "the user said something" in out


def test_an_over_long_wait_is_clamped_and_says_so(ctx, monkeypatch):
    from openreynolds import tools as tools_mod
    from openreynolds.tools import dispatch as _dispatch

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    monkeypatch.setattr(tools_mod, "JOB_WAIT_MAX_S", 0.05)
    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")

    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id, "wait_s": 9999})

    assert not is_error
    assert "ceiling" in out


def test_no_wait_asked_means_no_waiting(ctx):
    import time as _time

    from openreynolds.tools import dispatch as _dispatch

    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")

    began = _time.monotonic()
    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id})

    assert not is_error
    assert _time.monotonic() - began < 1
    assert "waited" not in out


# -- the same bytes are not sent twice -----------------------------------------


@pytest.fixture
def roomy(backend, store):
    """A context whose output cap is above the echo threshold, so a large repeat is
    actually large by the time it is compared."""
    return ToolContext(backend=backend, store=store, max_output=50_000)


def test_a_byte_identical_repeat_points_at_the_first_one(roomy):
    """`bash` output is 78-82% of everything the model is sent, and one study's most
    expensive call -- `cat .toolbox/notes/*.md` -- was 4,177 tokens that then rode along
    in all 76 requests after it. Reading it again puts a second copy in the same thread."""
    roomy.backend.exec_result = ExecResult(0, "N" * 5_000, False, None)

    first, _ = dispatch(roomy, "bash", {"cmd": "cat notes.md"})
    assert "N" * 5_000 in first

    again, error = dispatch(roomy, "bash", {"cmd": "cat notes.md"})
    assert not error, "the call succeeded; only the second copy of the bytes is gone"
    assert "identical, byte for byte" in again
    assert "#1" in again, "and it says where the bytes already are"
    assert len(again) < 500


def test_output_that_changed_is_never_collapsed(roomy):
    """A command whose answer moved is exactly the interesting case."""
    roomy.backend.exec_result = ExecResult(0, "A" * 5_000, False, None)
    dispatch(roomy, "bash", {"cmd": "tail log"})
    roomy.backend.exec_result = ExecResult(0, "B" * 5_000, False, None)
    second, _ = dispatch(roomy, "bash", {"cmd": "tail log"})
    assert "B" * 5_000 in second


def test_a_short_repeat_is_left_exactly_as_it_is(ctx):
    """Below the threshold the sentence explaining the repeat costs what the repeat does."""
    ctx.backend.exec_result = ExecResult(0, "ok", False, None)
    dispatch(ctx, "bash", {"cmd": "ls"})
    again, _ = dispatch(ctx, "bash", {"cmd": "ls"})
    assert "identical" not in again and "ok" in again


def test_the_same_failure_twice_is_still_reported_twice(ctx):
    """A repeated error is a fact about the run, not a duplicate to fold away."""
    def refuse(cmd, cwd=None, timeout_s=120):
        raise BackendError("x" * 5_000, code="boom")

    ctx.backend.exec = refuse
    first, error1 = dispatch(ctx, "bash", {"cmd": "go"})
    second, error2 = dispatch(ctx, "bash", {"cmd": "go"})
    assert error1 and error2
    assert first == second and "identical" not in second
