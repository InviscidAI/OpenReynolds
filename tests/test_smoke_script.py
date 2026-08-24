"""scripts/smoke.py, driven against a simulated workspace.

The smoke script only runs when someone has credentials, which is the worst moment to
discover a typo in it. This drives the whole thing end to end against a fake that
behaves the way the real service does, so its control flow and its parsing of tool
output are exercised without a network.
"""

from __future__ import annotations

import base64
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from conftest import FakeBackend
from openreynolds.backend.base import ExecResult, JobStatus, Stat

PNG_HEADER = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    b"YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
"""A real PNG header, so the shape reported alongside the picture is a real shape."""

SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "smoke.py"


@pytest.fixture
def smoke():
    """Import the script fresh, with its pass/fail counters reset."""
    spec = importlib.util.spec_from_file_location("smoke_under_test", SMOKE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_under_test"] = module
    spec.loader.exec_module(module)
    module.PASSED = 0
    module.FAILED = 0
    yield module
    sys.modules.pop("smoke_under_test", None)


class SimulatedWorkspace(FakeBackend):
    """Answers the specific commands the smoke script issues."""

    BIG_LOG = "/work/.foamd/exec/deadbeef.log"

    LISTING = (
        "-\t2400\t1700000000.0\t/work/.smoke/plot.png\n"
        "-\t260\t1700000001.0\t/work/.smoke/plot.py\n"
    )

    GEOMETRY_DIGEST = """\
/work/.smoke/box.stl
  faces 12   points 8   bodies 1
  extent 0.06 x 0.02 x 0.01  (OpenFOAM reads these as metres)
  open edges 0   non-manifold edges 0   duplicate points 0
  enclosed volume 1.2e-05   surface area 0.004
/work/.smoke/lid.stl
  faces 200   points 121   bodies 1
  extent 0.01 x 0.01 x 0  (OpenFOAM reads these as metres)
  open edges 40   non-manifold edges 0   duplicate points 0
  surface area 0.0001   (not closed, so no volume)

r/geometry.png
"""

    def __init__(self):
        super().__init__()
        self.dirs["/work/.smoke"] = ["lines.txt", "plot.py", "plot.png"]
        self.dirs["/work/.toolbox"] = ["log_digest.py", "mesh_digest.py", "cells_estimate.py"]
        self.files[self.BIG_LOG] = b"A" * 200_000
        self.files["/work/.smoke/plot.png"] = PNG_HEADER + b"\x00" * 2000
        self.files["/work/.smoke/r/geometry.png"] = PNG_HEADER + b"\x00" * 4000
        self.loose_solver = False
        """A solver outside any job's process group, as `mpirun` leaves behind."""
        self._polls: dict[str, int] = {}

    def exec(self, cmd, cwd=None, timeout_s=120):
        if "WM_PROJECT_VERSION" in cmd:
            return ExecResult(0, "hello\n2512\n", False, None)
        if cmd.strip() == "exit 7":
            return ExecResult(7, "", False, None)
        if "200000" in cmd.replace(" ", ""):
            return ExecResult(0, "B" * 200_000, True, self.BIG_LOG)
        if "cells_estimate.py --help" in cmd:
            return ExecResult(0, "usage: cells_estimate.py", False, None)
        if "plot.py" in cmd:
            return ExecResult(0, "PLOT_OK\n", False, None)
        if cmd.strip() == "pwd":
            return ExecResult(0, f"{cwd}\n", False, None)
        if cmd.startswith("find "):
            # A listing is of somewhere in particular: answering every path with the
            # same contents would let a check pass while looking in the wrong place.
            # `-H` sits before the path, so the target is the first non-flag word.
            target = next(
                word.strip("'\"")
                for word in cmd.split()[1:]
                if not word.startswith("-")
            )
            if target.startswith("/work/smoke-"):
                return ExecResult(
                    0, f"-\t28\t1700000005.0\t{target}/notes.md\n", False, None
                )
            return ExecResult(0, self.LISTING, False, None)
        if "geo.py" in cmd:
            return ExecResult(0, "SURFACES_OK\n", False, None)
        if "geometry_view.py" in cmd:
            return ExecResult(0, self.GEOMETRY_DIGEST, False, None)
        if cmd.startswith("ps "):
            return ExecResult(0, "simpleFoam\n" if self.loose_solver else "", False, None)
        if cmd.startswith("pkill"):
            self.loose_solver = False
            return ExecResult(0, "", False, None)
        return ExecResult(0, "", False, None)

    def stat(self, path):
        if path == self.BIG_LOG:
            return Stat(path, "regular file", 200_000, 0, [])
        return super().stat(path)

    def job_start(self, cmd, cwd=None, name=None, kill_on=None):
        job_id = super().job_start(cmd, cwd=cwd, name=name, kill_on=kill_on)
        self.logs[job_id] = b"tick 1\ntick 2\n"
        self._polls[job_id] = 0
        if "simpleFoam" in cmd:
            # A solver started this way lands outside the job's process group, which
            # is the whole reason `stop --force` exists.
            self.loose_solver = True
        return job_id

    def job_status(self, job_id):
        """Run for one poll, then finish -- the way a short job behaves."""
        current = self.jobs[job_id]
        self._polls[job_id] += 1
        if self._polls[job_id] < 2:
            return current
        if current.name == "fatal":
            self.logs[job_id] = b"starting\n--> FOAM FATAL ERROR: smoke\n"
            self.jobs[job_id] = JobStatus(
                job_id=job_id,
                name=current.name,
                status="killed",
                end_reason="kill_on_match",
                killed_by="--> FOAM FATAL ERROR: smoke",
                log_size=len(self.logs[job_id]),
            )
        else:
            self.logs[job_id] = b"tick 1\ntick 2\ndone\n"
            self.jobs[job_id] = JobStatus(
                job_id=job_id,
                name=current.name,
                status="exited",
                exit_code=0,
                end_reason="completed",
                log_size=len(self.logs[job_id]),
            )
        return self.jobs[job_id]


@pytest.fixture
def wired(smoke, monkeypatch, tmp_path):
    """Point the script at a simulated workspace and a temporary studies directory."""
    backend = SimulatedWorkspace()
    monkeypatch.setattr(smoke.hosted, "acquire", lambda url, key, iid: (backend, None, "inst-1"))
    monkeypatch.setattr(
        smoke.Config,
        "load",
        classmethod(
            lambda cls: smoke.Config(
                foamd_url="https://svc",
                foamd_api_key="of_live_x",
                anthropic_api_key="",
                studies_dir=tmp_path / "studies",
            )
        ),
    )
    monkeypatch.setattr(smoke.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sys, "argv", ["smoke.py"])
    return smoke, backend


def test_the_smoke_script_passes_against_a_healthy_workspace(wired, capsys):
    smoke, _backend = wired

    exit_code = smoke.main()
    output = capsys.readouterr().out

    assert exit_code == 0, f"smoke reported failures:\n{output}"
    assert smoke.FAILED == 0
    assert smoke.PASSED > 15, "every check ran"
    assert "FAIL" not in output


def test_it_checks_the_things_it_claims_to(wired, capsys):
    smoke, _backend = wired
    smoke.main()
    output = capsys.readouterr().out

    for section in (
        "bash", "truncation", "files", "toolbox sync", "jobs", "kill_on", "fetch",
        "seeing", "looking at the workspace", "geometry", "stopping what outlived its job",
    ):
        assert f"== {section}" in output, f"missing section: {section}"


def test_it_reads_the_job_id_out_of_the_tool_output(wired):
    """The script parses `started job <id>`; a change to that line breaks it."""
    smoke, backend = wired
    smoke.main()
    launched = [j["name"] for j in backend.started]
    assert "fatal" in launched, "the kill_on job launched"
    assert "smoke-loose" in launched, "so did the one that outlives its group"
    assert backend.started[1]["kill_on"] == ["FOAM FATAL"]


def test_it_pulls_the_render_to_the_local_study(wired, tmp_path, monkeypatch):
    """--keep leaves the study behind, which is the only way to see what it pulled."""
    smoke, _backend = wired
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--keep"])

    smoke.main()

    pulled = list((tmp_path / "studies").rglob("plot.png"))
    assert pulled, "fetch landed the PNG under the study directory"
    assert pulled[0].stat().st_size > 1000


def test_it_cleans_up_after_itself(wired, tmp_path):
    smoke, backend = wired
    calls = []
    original = backend.exec
    backend.exec = lambda cmd, cwd=None, timeout_s=120: (
        calls.append(cmd) or original(cmd, cwd=cwd, timeout_s=timeout_s)
    )

    smoke.main()

    assert any("rm -rf /work/.smoke" in cmd for cmd in calls), "the workspace is tidied"
    assert not list((tmp_path / "studies").glob("smoke-*")), "the study directory is removed"


def test_a_failing_check_is_reported_and_exits_nonzero(wired, capsys):
    smoke, backend = wired
    backend.exec = lambda cmd, cwd=None, timeout_s=120: ExecResult(0, "wrong version", False, None)

    exit_code = smoke.main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert smoke.FAILED > 0
    assert "FAIL" in output


def test_it_refuses_to_run_without_credentials(smoke, monkeypatch):
    monkeypatch.setattr(
        smoke.Config, "load", classmethod(lambda cls: smoke.Config())
    )
    monkeypatch.setattr(sys, "argv", ["smoke.py"])
    assert smoke.main() == 2


def test_it_does_not_need_a_model_key(smoke, monkeypatch, tmp_path):
    """The smoke script drives the workspace with no model in the loop."""
    monkeypatch.setattr(
        smoke.Config,
        "load",
        classmethod(
            lambda cls: smoke.Config(foamd_url="u", foamd_api_key="k", anthropic_api_key="")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["smoke.py"])
    monkeypatch.setattr(
        smoke.hosted, "acquire", lambda url, key, iid: (SimulatedWorkspace(), None, "inst-1")
    )
    monkeypatch.setattr(smoke.time, "sleep", lambda _s: None)
    assert smoke.main() == 0


def test_the_script_never_deletes_an_instance():
    """Deleting one destroys its persistent volume."""
    source = SMOKE.read_text(encoding="utf-8")
    assert not re.search(r"\bdelete\b", source, re.IGNORECASE), "smoke must not delete anything"
