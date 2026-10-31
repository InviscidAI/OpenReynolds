"""Reading a Felix job's log: step, time, residual trend, errors, how it ended (goal A4).

The fixtures under `tests/data/felix/logs/` are the stderr of real runs of the Felix
solver on this repository's development machine (one A100-SXM4-40GB, Felix build
a60138c7f780, 2026-10-07), with the scratch directory each ran in rewritten to
`/work/cases/<name>`:

- `cavity_small.stderr`: a quasi-2D cavity, 8x8x1 hexes, Re 100, 30 steps, snapshots,
  two probes and the lid force -- every line the run printed.
- `re1000_ptc.stderr`: the Re 1000 cavity on 128x128x1 hexes run to a steady state
  with PTC; it converged at step 624. Trimmed to the first 20 and last 25 `step=`
  lines; every other line is as printed.
- `nonfinite.stderr`: the same small cavity with the lid speed `sqrt(0.035 - t)`,
  which is NaN from step 4 (the stderr.md recipe).
- `oom_gcr_max.stderr`: `gcr.max: 2000000`, which does not fit on 40 GB.
- `oom_needs.stderr`: the small cavity started while another process held all but
  0.4 GB of the GPU.
- `case_error.stderr`: an unknown key in control.yaml.
- `felix_run.log`: what a `felix run` job's log holds -- felixd's log lines followed
  by the final status JSON. The solver lines are the real stderr of the
  Schafer-Turek cylinder run (trimmed to its first and last steps); the felixd NOTE
  line and the status object are composed in felixd's formats (`runner/wrapper.py`,
  `lifecycle.public`), because the `felix` command is being built alongside this.

The third out-of-memory form (`out of GPU memory: ... when a request for R GB
failed`) could not be provoked on demand here: holding the GPU's memory from another
process trips the earlier check instead. Its text is the captured example in
Felix's `docs/user/stderr.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openreynolds import felix_progress as fp
from openreynolds.progress import LogFacts, phase_from_cmd

LOGS = Path(__file__).parent / "data" / "felix" / "logs"


def log(name: str) -> str:
    return (LOGS / name).read_text(encoding="utf-8")


# -- the command ---------------------------------------------------------------


@pytest.mark.parametrize("cmd", [
    "felix run cavity --gpu A100",
    "cd /work/study/cavity && felix continue . --gpu H100 --wall 3600",
    "felix run /work/study/cyl --gpu A100 > run.log",
])
def test_a_felix_run_or_continue_is_recognised_as_a_felix_solve(cmd):
    assert fp.is_felix_cmd(cmd)
    assert phase_from_cmd(cmd) == ("solving", "felix")


@pytest.mark.parametrize("cmd", [
    "felix gpus",
    "felix-check-mesh --case cavity",
    "felix-tag-mesh tags.yaml",
    "simpleFoam -case cavity",
    "python3 felix_results.py summary cavity",
])
def test_other_commands_are_not_felix_solves(cmd):
    assert not fp.is_felix_cmd(cmd)


def test_the_case_directory_comes_from_the_command():
    assert fp.case_dir_from_felix_cmd("felix run cavity --gpu A100", "/work/s") == "/work/s/cavity"
    assert fp.case_dir_from_felix_cmd("cd /work/s/cyl && felix continue . --gpu H100", "/w") == "/work/s/cyl"
    assert fp.case_dir_from_felix_cmd("felix run --gpu A100 /abs/case", "/w") == "/abs/case"
    assert fp.case_dir_from_felix_cmd("felix gpus", "/w") is None


def test_nsteps_is_read_from_control_yaml():
    assert fp.nsteps_from_control("# control\ndt: 0.01\nnsteps: 3000\n") == 3000
    assert fp.nsteps_from_control("dt: 0.01\ncfl:\n  target: 1\n") is None


def test_a_warm_start_s_step_is_read_off_its_snapshot_name():
    assert fp.warm_start_step("output:\n  every: 1000\nwarm_start: output/snapshot_015000.vtu\n") == 15000
    assert fp.warm_start_step("warm_start: '../chunk1/output/snapshot_000120.vtu'\n") == 120
    assert fp.warm_start_step("output:\n  every: 1000\n") is None
    assert fp.warm_start_step("warm_start: initial.vtu\n") is None


def test_a_continued_run_ends_at_its_warm_start_plus_nsteps():
    """ui#44: `felix continue` sets `nsteps` to what is left (15000 of 30000) while
    the log counts on from the snapshot's step, so the run ends at 30000, not 15000."""
    files = {
        "/c/case.yaml": "mesh: mesh.vtu\ncontrol: control.yaml\nio: io.yaml\n",
        "/c/control.yaml": "dt: 0.2\nnsteps: 15000\n",
        "/c/io.yaml": "output:\n  every: 1000\nwarm_start: output/snapshot_015000.vtu\n",
    }
    assert fp.last_step(files.get, "/c") == 30000


def test_a_cold_run_ends_at_nsteps_and_case_yaml_names_the_files():
    files = {
        "/c/case.yaml": "control: ctl.yaml\nio: felixd_restart_io.yaml\n",
        "/c/ctl.yaml": "nsteps: 400\n",
        "/c/control.yaml": "nsteps: 9\n",
        "/c/felixd_restart_io.yaml": "warm_start: output/snapshot_000100.vtu\n",
    }
    assert fp.last_step(files.get, "/c") == 500
    assert fp.last_step({"/c/control.yaml": "nsteps: 60\n"}.get, "/c") == 60
    assert fp.last_step({}.get, "/c") is None


# -- a run that finished ---------------------------------------------------------


def test_step_time_and_residuals_come_from_the_last_step_line():
    facts = fp.parse_felix_log(log("cavity_small.stderr"))
    assert facts.step == 30
    assert facts.sim_time == pytest.approx(1.5)
    assert facts.dt == pytest.approx(0.05)
    assert facts.wall_time > 0
    assert set(facts.residuals) >= {"res0", "resF", "gcr_res", "cg_res"}
    assert facts.iterations["gcr"] > 0
    assert facts.finished and facts.outcome == "finished"
    assert facts.errors == []


def test_the_residual_trend_is_read_over_the_steps_in_the_window():
    """res0 falls steadily on the impulsively started cavity; the trend says so and
    carries the two numbers behind the word."""
    facts = fp.parse_felix_log(log("cavity_small.stderr"))
    assert facts.trend is not None
    assert facts.trend.word == "falling"
    assert facts.trend.recent < facts.trend.before


def test_a_ptc_run_that_converged_says_so():
    facts = fp.parse_felix_log(log("re1000_ptc.stderr"))
    assert facts.step == 623
    assert facts.ptc_converged_step == 624
    assert facts.finished
    assert facts.gpu_memory_gb == pytest.approx(0.96)
    lines = "\n".join(fp.summary_lines(facts, last_step=3000))
    assert "PTC converged at step 624" in lines
    assert "623" in lines and "3000" in lines


def test_end_of_run_paragraphs_are_kept_as_notes_and_warnings_are_listed():
    facts = fp.parse_felix_log(log("re1000_ptc.stderr"))
    assert facts.warnings == []
    assert any("without converging on 8 of 624" in note for note in facts.health_notes)


# -- runs that did not finish ------------------------------------------------------


def test_a_non_finite_stop_is_an_error_with_its_step():
    facts = fp.parse_felix_log(log("nonfinite.stderr"))
    assert facts.step == 3
    assert facts.nonfinite_step == 4
    assert facts.outcome == "stopped"
    assert any("non-finite" in e for e in facts.errors)
    assert len(facts.warnings) >= 4
    text = "\n".join(fp.summary_lines(facts))
    assert "non-finite" in text and "step 4" in text


def test_out_of_memory_for_gcr_max_says_how_much_is_needed():
    facts = fp.parse_felix_log(log("oom_gcr_max.stderr"))
    assert facts.outcome == "failed"
    oom = facts.oom
    assert oom is not None
    assert oom.needed_gb == pytest.approx(112.1)
    assert oom.gpu_gb == pytest.approx(42.4)
    assert oom.free_gb == pytest.approx(41.9)
    assert oom.largest_gcr_max == 735722
    text = "\n".join(fp.summary_lines(facts))
    assert "112.1 GB" in text and "42.4 GB" in text


def test_out_of_memory_for_the_case_itself_says_how_much_is_needed():
    facts = fp.parse_felix_log(log("oom_needs.stderr"))
    assert facts.oom is not None
    assert facts.oom.needed_gb == pytest.approx(1.1)
    assert facts.oom.free_gb == pytest.approx(0.4)
    assert facts.oom.largest_gcr_max is None


def test_the_allocation_failure_form_of_out_of_memory_is_read_too():
    """From Felix's docs/user/stderr.md: a request that failed mid-run. What was
    allocated plus what was asked for is the least the case needs."""
    line = ("ERROR: out of GPU memory: 1.85 GB allocated by this run when a request for "
            "0.09 GB failed; the GPU has 42.41 GB (1.87 GB free at start). Use a GPU with "
            "more memory; see gpu_sizing.md.")
    facts = fp.parse_felix_log(line + "\n")
    assert facts.oom.needed_gb == pytest.approx(1.94)
    assert facts.oom.gpu_gb == pytest.approx(42.41)


def test_a_case_error_is_surfaced_whole():
    facts = fp.parse_felix_log(log("case_error.stderr"))
    assert facts.step is None
    assert facts.outcome == "failed"
    assert facts.errors == [
        "control.yaml dt_typo: unknown key 'dt_typo'. Valid keys: dt, nsteps, rho_inf, "
        "corrector, cfl, frozen_velocity, tauc_scale."
    ]


def test_an_unprefixed_line_continues_the_message_above_it():
    """`error_continuation.stderr` is a real two-line ERROR: the small cavity started
    with all but 0.47 GB of the GPU held by another process, where Felix reported a
    zero element volume rather than running out of memory."""
    facts = fp.parse_felix_log(log("error_continuation.stderr"))
    assert len(facts.errors) == 1
    assert facts.errors[0].startswith("/work/cases/oom4/mesh.vtu:")
    assert "derived element volume 0 is 0" in facts.errors[0]
    assert facts.outcome == "failed"


def test_nan_spellings_in_a_step_line_do_not_break_the_parse():
    facts = fp.parse_felix_log(
        "step=7 t=0.07 dt=0.01 gcr=5 gmres=20 cg=22 wall_step=0.0083 wall_time=0.1 "
        "res0=-nan resF=inf gcr_res=3.2e-02 cg_res=8.2e-02 corrector=4\n")
    assert facts.step == 7
    assert facts.residuals["gcr_res"] == pytest.approx(0.032)


# -- through `felix run` -------------------------------------------------------------


def test_the_final_status_of_felix_run_is_read_from_the_end_of_the_log():
    facts = fp.parse_felix_log(log("felix_run.log"))
    assert facts.status is not None
    assert facts.status["state"] == "succeeded"
    assert facts.outcome == "finished"
    assert facts.ptc_converged_step == 111
    assert facts.step == 110
    text = "\n".join(fp.summary_lines(facts))
    assert "succeeded" in text
    assert "billed" in text and "$" not in text  # the GPU time, never its cost


def test_felixd_lines_are_told_apart_from_the_solver_s():
    """felixd writes its own NOTE and WARNING lines into the same log (a preempted
    runner continuing from a snapshot, a failed snapshot copy). They are about the
    service, not the solver, and a reader needs to know which is which."""
    text = log("cavity_small.stderr").replace(
        "NOTE: nu=", "NOTE: felixd: segment 2 continues from snapshot step 20 "
        "(the previous runner was preempted)\nNOTE: nu=", 1)
    facts = fp.parse_felix_log(text)
    assert facts.felixd_notes == [
        "felixd: segment 2 continues from snapshot step 20 (the previous runner was preempted)"]
    assert "preempted" in "\n".join(fp.summary_lines(facts))


def test_a_failed_solve_status_is_read_even_on_one_line():
    text = log("oom_gcr_max.stderr") + (
        '{"id": "x", "state": "failed", "exit_code": 1, "error": "linear_solvers.yaml '
        '`gcr.max: 2000000` needs 112.1 GB", "cost_usd": 0.0012, "billed_s": 2.0}\n')
    facts = fp.parse_felix_log(text)
    assert facts.status["state"] == "failed"
    assert facts.outcome == "failed"
    assert facts.oom.needed_gb == pytest.approx(112.1)


def test_the_status_line_says_the_gpu_time_but_never_the_cost():
    """The solve's status carries `cost_usd`, and the line made from it reaches the
    person as well as the model. What the hosted service costs is never shown to the
    person, so the line keeps the GPU and its billed seconds and leaves the dollars out."""
    text = log("cavity_small.stderr") + (
        '{"id": "x", "state": "succeeded", "gpu": "A100", "exit_code": 0, '
        '"cost_usd": 0.8561, "billed_s": 1290.0}\n')
    facts = fp.parse_felix_log(text)
    said = "\n".join(fp.summary_lines(facts))
    assert "solve succeeded, on A100, exit 0, 1290 s billed" in said
    assert "$" not in said and "0.856" not in said


def test_a_running_job_with_only_step_lines_is_running():
    text = "\n".join(log("cavity_small.stderr").splitlines()[:12]) + "\n"
    facts = fp.parse_felix_log(text)
    assert facts.outcome == "running"
    assert facts.step is not None and facts.step < 30


def test_a_tail_that_starts_mid_line_is_read_from_its_first_whole_line():
    text = log("cavity_small.stderr")
    tail = text[len(text) // 2 + 7:]
    facts = fp.parse_felix_log(tail)
    assert facts.step == 30


# -- what the bar and the wake get ----------------------------------------------------


def test_log_facts_feed_the_progress_bar():
    facts = fp.parse_felix_log(log("cavity_small.stderr"))
    lf = fp.log_facts(facts)
    assert isinstance(lf, LogFacts)
    assert lf.sim_time == pytest.approx(1.5)
    assert lf.delta_t == pytest.approx(0.05)
    assert "res0" in lf.residuals
    assert lf.ended is True
    assert lf.is_solver


# -- the tracker, on a `felix run` job ---------------------------------------------------

HOME = "/work/study-test"


def start_felix(backend, store, text, cmd="felix run cavity --gpu A100"):
    from openreynolds.backend.base import JobStatus

    job_id = backend.job_start(cmd, cwd=HOME, name="cavity")
    store.record_job(job_id, cmd=cmd, name="cavity", cwd=HOME)
    backend.logs[job_id] = text.encode()
    backend.jobs[job_id] = JobStatus(job_id=job_id, status="running", name="cavity",
                                     log_size=len(text))
    return job_id


def test_a_felix_job_shows_its_step_against_nsteps(backend, store, view):
    from openreynolds.progress import Tracker

    backend.files[f"{HOME}/cavity/control.yaml"] = b"dt: 0.05\nnsteps: 60\n"
    start_felix(backend, store, log("cavity_small.stderr"))
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    [job] = tracker.refresh_jobs(force=True)

    assert job.phase == "solving" and job.executable == "felix"
    assert job.step == 30 and job.end_step == 60
    assert job.fraction == pytest.approx(0.5)
    snap = tracker.snapshot()
    assert "step 30 / 60" in snap.headline
    assert "res0" in snap.detail


def test_a_continued_felix_job_counts_against_the_whole_run(backend, store, view):
    """ui#44, on the cavity: 30 steps, continued from snapshot 15, so `felix continue`
    left `nsteps: 15`. At step 20 the bar is two thirds of the way to 30 -- not past
    100% of 15 -- and the time left is ten steps at this solve's own pace."""
    from openreynolds.progress import Tracker

    steps = [ln for ln in log("cavity_small.stderr").splitlines() if ln.startswith("step=")]
    running = "\n".join(steps[15:20]) + "\n"
    backend.files[f"{HOME}/cavity/control.yaml"] = b"dt: 0.05\nnsteps: 15\n"
    backend.files[f"{HOME}/cavity/io.yaml"] = (
        b"output:\n  every: 5\nwarm_start: output/snapshot_000015.vtu\n")
    start_felix(backend, store, running, cmd="felix continue cavity --gpu A100")
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    [job] = tracker.refresh_jobs(force=True)

    assert job.step == 20 and job.end_step == 30
    assert job.fraction == pytest.approx(20 / 30)
    paces = [fp.parse_step_line(ln)["wall_step"] for ln in steps[15:20]]
    assert job.eta_s == pytest.approx(10 * sum(paces) / len(paces))
    assert "step 20 / 30" in tracker.snapshot().headline


def test_a_step_past_the_total_shows_no_total(backend, store, view):
    """A warm start the reader cannot place counts the run from 0; when the log has
    already passed that total, the bar shows the step and no percentage, rather than
    the 100% of ui#44."""
    from openreynolds.progress import Tracker

    steps = [ln for ln in log("cavity_small.stderr").splitlines() if ln.startswith("step=")]
    backend.files[f"{HOME}/cavity/control.yaml"] = b"dt: 0.05\nnsteps: 15\n"
    backend.files[f"{HOME}/cavity/io.yaml"] = b"warm_start: initial_state.vtu\n"
    start_felix(backend, store, "\n".join(steps[15:20]) + "\n")
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    [job] = tracker.refresh_jobs(force=True)

    assert job.step == 20 and job.end_step is None
    assert job.fraction is None and job.eta_s is None
    assert "step 20" in tracker.snapshot().headline
    assert "/ 15" not in tracker.snapshot().headline


def test_a_felix_job_tells_the_wake_its_step_and_any_error(backend, store, view):
    from openreynolds.progress import Tracker

    start_felix(backend, store, log("oom_gcr_max.stderr"))
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.refresh_jobs(force=True)

    [line] = tracker.facts_for_wake()

    assert line.startswith("cavity: ")
    assert "needs 112.1 GB" in line


def test_the_progress_seam_answers_a_felix_job_with_the_parser_s_reading():
    """`progress.felix_job_progress` is the one place the tracker asks about a Felix
    solve; it must answer (not None) so the generic OpenFOAM reading never runs."""
    from types import SimpleNamespace

    from openreynolds.progress import felix_job_progress, is_felix_solve

    record = SimpleNamespace(job_id="j1", name="cavity", cmd="felix run cavity --gpu A100",
                             cwd=HOME, launched_at="")
    assert is_felix_solve(record.cmd)
    progress = felix_job_progress(record, log("cavity_small.stderr"), 1234)
    assert progress is not None
    assert progress.executable == "felix" and progress.step == 30
    assert progress.felix.outcome == "finished"
    assert progress.log_size == 1234
