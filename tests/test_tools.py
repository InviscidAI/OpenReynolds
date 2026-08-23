from __future__ import annotations

import pytest

from openreynolds.backend.base import ExecResult, JobStatus
from openreynolds.tools import TOOLS, dispatch


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
    assert (store.fetch_dir() / "u.png").read_bytes() == b"\x89PNG"
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
