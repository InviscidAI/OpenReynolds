"""Felix mode, wired end to end: the Felix reading of a job in `job_check`, the job-end
wake and the bar; the toolbox a Felix study gets; and a session that runs `felix run`.

The readings themselves are `felix_progress`'s and `toolbox/felix_results.py`'s, with
their own tests. These check that a Felix study is given them, and an OpenFOAM study
is not.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from openreynolds import cli, felix_progress, watch
from openreynolds.backend.base import JobStatus
from openreynolds.prompt import FELIX_PROMPT, SYSTEM_PROMPT
from openreynolds.tools import dispatch

from conftest import FakeBackend, RecordingView, message, text_block, tool_block

DATA = Path(__file__).parent / "data" / "felix"
CASE = "/work/study/cav"
CMD = f"felix run {CASE} --gpu H100"


def _log(name: str) -> str:
    return (DATA / "logs" / name).read_text(encoding="utf-8")


def _felix_job(backend, store, text: str, *, status: str = "running", exit_code=None):
    backend.files[f"{CASE}/control.yaml"] = b"dt: 0.05\nnsteps: 30\n"
    job_id = backend.job_start(CMD, cwd="/work/study", name="cav")
    backend.logs[job_id] = text.encode("utf-8")
    backend.jobs[job_id] = JobStatus(job_id=job_id, status=status, name="cav",
                                     log_size=len(text.encode("utf-8")), exit_code=exit_code)
    store.record_job(job_id, cmd=CMD, name="cav", cwd="/work/study")
    return job_id


def _expected(text: str, nsteps: int | None = 30) -> list[str]:
    return felix_progress.summary_lines(felix_progress.parse_felix_log(text), nsteps)


@pytest.fixture
def felix(ctx):
    ctx.solver = "felix"
    return ctx


# -- job_check --------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["cavity_small.stderr", "oom_needs.stderr",
                                     "nonfinite.stderr", "felix_run.log"])
def test_job_check_gives_a_felix_job_the_felix_reading(felix, backend, store, fixture):
    text = _log(fixture)
    job_id = _felix_job(backend, store, text)

    content, is_error = dispatch(felix, "job_check", {"job_id": job_id})

    assert not is_error
    for line in _expected(text):
        assert line in content, line


def test_job_check_reads_nsteps_from_the_case_control_yaml(felix, backend, store):
    job_id = _felix_job(backend, store, _log("cavity_small.stderr"))
    content, _ = dispatch(felix, "job_check", {"job_id": job_id})
    assert "step 30 / 30" in content


def test_job_check_of_a_felix_job_says_nothing_of_cores(felix, backend, store):
    """`_running_on` is about ranks on the workspace's cores; a Felix solve runs on a
    cloud GPU and the workspace's cores are not what it is spending."""
    job_id = _felix_job(backend, store, _log("cavity_small.stderr"))
    content, _ = dispatch(felix, "job_check", {"job_id": job_id})
    assert "core" not in content.lower() and "rank" not in content.lower()


def test_job_check_of_an_openfoam_job_is_unchanged(ctx, backend, store):
    backend.job_start("simpleFoam", name="solve")
    store.record_job("job-1", cmd="simpleFoam", name="solve", cwd="/work")
    backend.logs["job-1"] = b"Time = 1\n"
    content, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})
    assert "felix" not in content


# -- the job-end wake --------------------------------------------------------------


def test_a_finished_felix_job_wakes_the_model_with_the_felix_reading(backend, store):
    text = _log("oom_needs.stderr")
    job_id = _felix_job(backend, store, text, status="exited", exit_code=1)

    report = watch._collect_finished(backend, store, RecordingView())

    assert job_id in report
    for line in _expected(text):
        assert line in report, line


def test_a_finished_openfoam_job_wakes_the_model_as_before(backend, store):
    backend.job_start("simpleFoam", name="solve")
    store.record_job("job-1", cmd="simpleFoam", name="solve", cwd="/work")
    backend.logs["job-1"] = b"End\n"
    backend.jobs["job-1"] = dataclasses.replace(backend.jobs["job-1"], status="exited",
                                                exit_code=0, log_size=4)
    report = watch._collect_finished(backend, store, RecordingView())
    assert "felix" not in report and "End" in report


# -- the toolbox -------------------------------------------------------------------


def _pushed(solver):
    backend = FakeBackend()
    cli._sync_toolbox(backend, solver=solver) if solver else cli._sync_toolbox(backend)
    (local_dir, remote), = backend.trees
    assert remote == cli.TOOLBOX_DEST
    return Path(local_dir)


def test_a_felix_study_gets_the_felix_toolbox():
    pushed = _pushed("felix")
    assert (pushed / "felix_results.py").is_file()
    assert not (pushed / "case_gen.py").exists(), "OpenFOAM's scripts are not offered"
    readme = (pushed / "README.md").read_text(encoding="utf-8")
    assert "felix_results.py" in readme
    assert "case_gen.py" not in readme and "checkMesh" not in readme


def test_an_openfoam_study_gets_its_toolbox_without_the_felix_rows():
    pushed = _pushed(None)
    assert (pushed / "log_digest.py").is_file() and (pushed / "case_gen.py").is_file()
    assert not (pushed / "felix_results.py").exists()
    readme = (pushed / "README.md").read_text(encoding="utf-8")
    assert "felix_results.py" not in readme and "case_gen.py" in readme


def test_the_felix_prompt_names_the_results_script():
    assert "felix_results.py" in FELIX_PROMPT and "/work/.toolbox" in FELIX_PROMPT
    for what in ("summary", "fields", "slice", "line", "probes", "forces", "log", "all"):
        assert f"`{what}`" in FELIX_PROMPT, what
    assert "felix_results" not in SYSTEM_PROMPT


# -- a Felix study's session, end to end -------------------------------------------


class _FelixWorkspace(FakeBackend):
    """A workspace whose `felix run` job has run, and written the cavity log."""

    def __init__(self, text):
        super().__init__()
        self.text = text
        self.files[f"{CASE}/control.yaml"] = b"dt: 0.05\nnsteps: 30\n"

    def job_start(self, cmd, cwd=None, name=None, kill_on=None):
        job_id = super().job_start(cmd, cwd=cwd, name=name, kill_on=kill_on)
        data = self.text.encode("utf-8")
        self.logs[job_id] = data
        # Finished by the time anyone looks: the run is the fixture's, start to end.
        self.jobs[job_id] = JobStatus(job_id=job_id, status="exited", name=name,
                                      log_size=len(data), exit_code=0)
        return job_id


def test_a_felix_session_reads_its_solve_the_felix_way(monkeypatch, tmp_path):
    from openreynolds.backend import pending as pending_mod
    from openreynolds.loop import Loop
    from openreynolds.progress import Tracker

    from conftest import ReadyStarter, ScriptedReader, install_model

    text = _log("cavity_small.stderr")
    backend = _FelixWorkspace(text)
    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    monkeypatch.setenv("OPENREYNOLDS_SOLVER", "felix")
    monkeypatch.setattr(cli.hosted, "reserve",
                        lambda url, key, iid=None, **kw: (None, "iid-f",
                                                          ReadyStarter(backend, "iid-f")))
    trackers: list[Tracker] = []

    class Kept(Tracker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trackers.append(self)

    monkeypatch.setattr(cli, "Tracker", Kept)
    loops: list[Loop] = []

    class Scripted(Loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fake = install_model(self, [
                message([tool_block("job_start", {"cmd": CMD, "name": "cav"}, "t1")],
                        stop_reason="tool_use"),
                message([tool_block("job_check", {"job_id": "job-1"}, "t2")],
                        stop_reason="tool_use"),
                message([text_block("done")]),
            ])
            loops.append(self)

    monkeypatch.setattr(cli, "Loop", Scripted)
    seen = SimpleNamespace(headline="", wake=[])

    class View(RecordingView):
        def tool(self, name, summary):
            super().tool(name, summary)
            if name == "job_check" and trackers:
                trackers[0].refresh_jobs(force=True)
                seen.headline = trackers[0].snapshot().headline
                seen.wake = trackers[0].facts_for_wake()

    cfg = cli.Config(foamd_url="u", foamd_api_key="k", llm_api_key="sk-test", model="m",
                     studies_dir=tmp_path / "studies", capture=False, desk=False,
                     mesh_tool=False, mirror_interval_s=0.0, workspace_eta_s=1.0)

    def interface(drive):
        drive(View(), ScriptedReader(["go"]))
        return False

    cli.session(cfg, study_id=None, instance_id=None, one_shot=None, interface=interface)

    (loop,) = loops
    results = [block for m in loop.messages if m["role"] == "user"
               and isinstance(m["content"], list) for block in m["content"]
               if isinstance(block, dict) and block.get("type") == "tool_result"]
    checked = str(results[-1]["content"])
    for line in _expected(text):
        assert line in checked, line
    assert "step 30 / 30" in seen.headline
    facts = felix_progress.parse_felix_log(text)
    assert felix_progress.wake_line("cav", facts, 30) in seen.wake
