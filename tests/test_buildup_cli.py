"""The supervisor's commands, driven the way the subagent drives them.

End to end and through a subprocess, because that is the arrangement being tested: the
supervisor is a **different process** from the run, holding imports -- `cad_audit`,
`domain_probe`, `surfaces` -- that are deliberately absent from the workspace. A test that
called the functions in-process would prove the checks work and nothing about the boundary
that makes them legitimate.

The exit codes matter as much as the JSON: an agent acts on them. `0` clean, `1` the thing
it asked about went wrong, `2` it could not ask.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from openreynolds.buildup import heartbeat, record
from test_buildup_probes import CUBE, stl

CLI = Path(__file__).resolve().parents[1] / "scripts" / "cad_supervise.py"


def supervise(*argv):
    done = subprocess.run([sys.executable, str(CLI), *argv], capture_output=True,
                          text=True, timeout=300)
    return done.returncode, json.loads(done.stdout)


def staged(tmp_path: Path, point=(5, 5, 5)) -> tuple[Path, Path]:
    """A finished run: three turns, a mesh, and a meshing point outside the part."""
    case = tmp_path / "work" / "T1-case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "system").mkdir(parents=True)
    (case / "constant" / "triSurface" / "walls.stl").write_text(stl(CUBE), encoding="utf-8")
    (case / "system" / "snappyHexMeshDict").write_text(
        "locationInMesh (%s %s %s);\n" % tuple(point), encoding="utf-8")
    run = tmp_path / "run"
    pulse = heartbeat.Heartbeat(run)
    pulse.start()
    for turn in (1, 2, 3):
        pulse.beat(turn=turn, steps=turn, stop_reason="end_turn", text_chars=200,
                   fenced=True)
    pulse.done()
    record.save(run, record.Record(run_id="cli-1", case="T1", case_dir=str(case),
                                   mesh_exists=True, checkmesh_ok=True, n_steps=3))
    (run / "cells.log").write_text("print('bounds', 1.0)\n", encoding="utf-8")
    return run, case


def test_a_dirty_environment_is_refused_with_what_resolved(tmp_path):
    stale = tmp_path / "work" / ".toolbox"
    stale.mkdir(parents=True)
    code, answer = supervise("preflight", str(tmp_path / "work"))
    assert code == 1 and not answer["clean"] and ".toolbox" in answer["why"]


def test_a_run_that_finished_with_a_silent_failure_does_not_exit_clean(tmp_path):
    """`done` and a fired probe at once is the pairing the probes exist for: the run
    finished, `checkMesh` passed, and the mesh is of the volume around the part."""
    run, case = staged(tmp_path)
    code, answer = supervise("observe", str(run), "--case", str(case))
    assert code == 1
    assert answer["stopped"] == "done" and answer["fired"] == ["location_in_mesh"]
    assert answer["n_turns"] == 3 and not answer["contaminated"]


def test_the_same_run_with_the_point_inside_exits_clean(tmp_path):
    run, case = staged(tmp_path, point=(0.5, 0.5, 0.5))
    code, answer = supervise("observe", str(run), "--case", str(case))
    assert code == 0 and answer["fired"] == []


def test_the_graded_record_keeps_the_probe_evidence_and_the_workspace_is_untouched(tmp_path):
    run, case = staged(tmp_path)
    before = sorted(str(p.relative_to(case)) for p in case.rglob("*"))
    supervise("observe", str(run), "--case", str(case))
    graded = record.load(run)
    fired = [row for row in graded["probes"] if row["state"] == "fired"]
    assert fired[0]["measured"]["classification"] == "outside"
    assert sorted(str(p.relative_to(case)) for p in case.rglob("*")) == before


def test_the_registry_is_printable_so_a_report_can_cite_it():
    code, answer = supervise("registry")
    assert code == 0 and answer["active"] == [] and len(answer["dormant"]) == 6


def test_a_run_directory_with_no_record_says_so_rather_than_inventing_one(tmp_path):
    code, answer = supervise("report", str(tmp_path))
    assert code == 2 and "no record.json" in answer["why"]
