"""A log reader handed a case directory refuses, and says what the directory holds.

Found by sweeping the read-only toolbox scripts against a real case out of the corpus,
the way a session would. On `studies/20260907-005653-0902/.../Lduct`, both
`log_digest.py <case>` and `mesh_digest.py <case>` ended in

    PermissionError: [Errno 13] Permission denied: '...\\Lduct'

which is what Windows raises for opening a directory as a file. POSIX raises
`IsADirectoryError`, so the traceback differed by platform and neither was readable.

Both scripts take a log FILE, which is right. But several of their siblings take a case
DIRECTORY and find the file themselves -- `acoustics.py`, `first_look.py`,
`preflight.py` -- so a session holding a case will try it here, and a traceback is the
wrong answer to a reasonable guess.

The standard is set by the scripts beside them. On that same case directory
`domain_probe.py` answered "refused: no .stl files in <dir>" and `patch_entries.py`
"refused: no surface files in <dir>". These two now meet it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"

CHECKMESH_LOG = """\
Create polyMesh for time = 0

Mesh stats
    points:           638
    faces:            1234
    cells:            300

Checking geometry...
    Overall domain bounding box (0 0 0) (1 1 0.1)
    Max aspect ratio = 3.2 OK.
    Minimum face area = 1e-06. Maximum face area = 4e-05.
    Mesh OK.

End
"""

SOLVER_LOG = """\
Starting time loop

Time = 0.1

smoothSolver:  Solving for Ux, Initial residual = 0.5, Final residual = 1e-06, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.4, Final residual = 1e-06, No Iterations 3
GAMG:  Solving for p, Initial residual = 0.9, Final residual = 1e-05, No Iterations 7
time step continuity errors : sum local = 1e-09, global = 1e-12, cumulative = 1e-12
ExecutionTime = 0.5 s  ClockTime = 1 s

End
"""


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(TOOLBOX / script), *args],
                          capture_output=True, text=True, timeout=240)


@pytest.fixture
def a_case(tmp_path: Path) -> Path:
    """A case directory with the logs a real one has beside its dictionaries."""
    case = tmp_path / "Lduct"
    (case / "system").mkdir(parents=True)
    (case / "log.checkMesh").write_text(CHECKMESH_LOG, encoding="utf-8")
    (case / "log.simpleFoam").write_text(SOLVER_LOG, encoding="utf-8")
    (case / "log.gmshToFoam").write_text("End\n", encoding="utf-8")
    return case


@pytest.mark.parametrize("script, kind", [("log_digest.py", "solver"),
                                          ("mesh_digest.py", "checkMesh")])
def test_a_case_directory_is_refused_and_its_logs_are_named(script, kind, a_case):
    done = _run(script, str(a_case))

    assert done.returncode == 1, done.stdout + done.stderr
    said = done.stdout + done.stderr
    assert "Traceback" not in said, "a reasonable guess should not produce a traceback"
    assert "is a directory" in said
    assert f"reads one {kind} log" in said
    # The logs it holds, because that is what the caller needed and it was already in
    # the argument they got wrong.
    for name in ("log.checkMesh", "log.simpleFoam", "log.gmshToFoam"):
        assert name in said, f"{name} was not named: {said}"


@pytest.mark.parametrize("script", ["log_digest.py", "mesh_digest.py"])
def test_a_directory_with_no_logs_says_that_rather_than_listing_nothing(script, tmp_path):
    bare = tmp_path / "empty"
    bare.mkdir()

    said = _run(script, str(bare))
    blob = said.stdout + said.stderr

    assert said.returncode == 1
    assert "Traceback" not in blob
    assert "No `log.*` file in it either" in blob


@pytest.mark.parametrize("script", ["log_digest.py", "mesh_digest.py"])
def test_a_path_that_is_not_there_is_refused_by_name(script, tmp_path):
    said = _run(script, str(tmp_path / "log.nope"))
    blob = said.stdout + said.stderr

    assert said.returncode == 1
    assert "Traceback" not in blob
    assert "no such file" in blob


def test_the_checkmesh_log_itself_still_reads(a_case):
    done = _run("mesh_digest.py", str(a_case / "log.checkMesh"))

    assert done.returncode == 0, done.stdout + done.stderr
    assert "638" in done.stdout, "the point count off the log it was given"


def test_the_solver_log_itself_still_reads(a_case, tmp_path):
    done = _run("log_digest.py", str(a_case / "log.simpleFoam"),
                "-o", str(tmp_path / "residuals.png"))

    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "residuals.png").is_file(), "and drew its plot where asked"


def test_a_log_on_stdin_is_still_read_without_a_path(a_case):
    """`mesh_digest.py -` is in its own docstring, so the refusal must not reach it."""
    done = subprocess.run(
        [sys.executable, "-I", str(TOOLBOX / "mesh_digest.py"), "-"],
        input=CHECKMESH_LOG, capture_output=True, text=True, timeout=240)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "638" in done.stdout
