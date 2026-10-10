"""The Felix benchmark grader (goal B1).

`benchmarks/felix/grade.py` grades two cases from what the solver wrote:

- `cavity`: the lid-driven cavity at Re 1000 against Ghia, Ghia & Shin (1982),
  Tables I and II -- u on the vertical centreline, v on the horizontal one;
- `cylinder`: the Schafer & Turek (1996) 2D-1 cylinder in a channel, Re 20, drag
  coefficient 5.5795 (their reference interval 5.57-5.59).

The real fixtures are Felix runs on one A100 (build a60138c7f780, 2026-10-07):
a quasi-2D 128x128x1 hex cavity run to a steady state with PTC (converged at step
624, 5 min), and the 2D-1 channel on 6765 hexes one layer thick (converged at step
111, 1 min). Their `solution.vtu` files are cut down to the cells the grader samples
-- the two centreline strips of the cavity; the inlet and outlet columns and the
cells round the cylinder -- so the repository carries kilobytes, not megabytes.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GRADE = ROOT / "benchmarks" / "felix" / "grade.py"
DATA = Path(__file__).parent / "data" / "felix"
CAVITY = DATA / "cavity_re1000"
CYLINDER = DATA / "cylinder_2d1"


def _load():
    spec = importlib.util.spec_from_file_location("felix_grade", GRADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grade = _load()


def needs_pyvista():
    pytest.importorskip("pyvista", reason="reading .vtu needs pyvista", exc_type=ImportError)


# -- the published tables ----------------------------------------------------------


def test_the_ghia_tables_are_the_published_re_1000_columns():
    """17 points each, end points on the walls, spot values from Tables I and II."""
    u = dict(grade.GHIA_RE1000_U)
    v = dict(grade.GHIA_RE1000_V)
    assert len(u) == len(v) == 17
    assert u[1.0] == 1.0 and u[0.0] == 0.0
    assert u[0.1719] == -0.38289
    assert u[0.5] == -0.06080
    assert u[0.9766] == 0.65928
    assert v[0.9063] == -0.51550
    assert v[0.2344] == 0.32235
    assert v[0.1563] == 0.37095
    assert v[0.5] == 0.02526


def test_the_schafer_turek_reference_is_the_2d1_drag():
    assert grade.ST_2D1["cd"] == 5.5795
    assert grade.ST_2D1["cd_interval"] == (5.57, 5.59)
    assert grade.ST_2D1["re"] == 20


# -- the cavity checks, on numbers ---------------------------------------------------


def _ghia_values():
    return (np.array([val for _, val in grade.GHIA_RE1000_U]),
            np.array([val for _, val in grade.GHIA_RE1000_V]))


def test_ghia_s_own_numbers_pass():
    u, v = _ghia_values()
    checks = grade.cavity_checks(u, v, reynolds=1000.0)
    assert all(c["passed"] for c in checks)


def test_a_profile_off_by_one_and_a_half_percent_of_the_lid_passes_and_three_fails():
    u, v = _ghia_values()
    assert all(c["passed"] for c in grade.cavity_checks(u + 0.015, v, reynolds=1000.0))
    failed = [c["name"] for c in grade.cavity_checks(u + 0.03, v, reynolds=1000.0) if not c["passed"]]
    assert failed == ["u RMS error (vertical centreline)"]


def test_the_wrong_reynolds_number_fails_whatever_the_profiles_say():
    u, v = _ghia_values()
    failed = [c["name"] for c in grade.cavity_checks(u, v, reynolds=100.0) if not c["passed"]]
    assert failed == ["Reynolds number"]


# -- the cavity, on a real Felix run --------------------------------------------------


def test_a_real_felix_cavity_at_re_1000_passes():
    needs_pyvista()
    result = grade.grade_cavity(CAVITY)
    assert result["passed"], result
    by = {c["name"]: c for c in result["checks"]}
    assert by["Reynolds number"]["value"] == pytest.approx(1000.0)
    assert by["u RMS error (vertical centreline)"]["value"] == pytest.approx(0.0045, abs=0.0005)
    assert by["v RMS error (horizontal centreline)"]["value"] == pytest.approx(0.0106, abs=0.0005)
    assert result["frame"]["lid_axis"] == "y" and result["frame"]["lid_direction"] == "+x"


def _transformed_case(tmp_path, point_map, vector_map, nu=None):
    """The real cavity with its axes permuted or mirrored, as an agent might set it up."""
    import pyvista as pv

    case = tmp_path / "case"
    shutil.copytree(CAVITY, case)
    path = case / "output" / "solution.vtu"
    mesh = pv.read(path)
    mesh.points = point_map(np.asarray(mesh.points))
    mesh.point_data["velocity"] = vector_map(np.asarray(mesh.point_data["velocity"]))
    mesh.save(path)
    if nu is not None:
        text = (case / "case.yaml").read_text().replace("nu: 0.001", f"nu: {nu}")
        (case / "case.yaml").write_text(text)
    return case


def test_the_lid_is_found_wherever_the_case_put_it(tmp_path):
    """Lid on z = 1 moving in -y, span along x: the same flow, read the same way."""
    needs_pyvista()

    def points(p):  # (x, y, z) -> (span, -x + 1, y)
        return np.c_[p[:, 2], 1.0 - p[:, 0], p[:, 1]]

    def vectors(u):
        return np.c_[u[:, 2], -u[:, 0], u[:, 1]]

    result = grade.grade_cavity(_transformed_case(tmp_path, points, vectors))
    assert result["frame"]["lid_axis"] == "z" and result["frame"]["lid_direction"] == "-y"
    by = {c["name"]: c for c in result["checks"]}
    assert by["u RMS error (vertical centreline)"]["value"] == pytest.approx(0.0045, abs=0.0005)
    assert result["passed"]


def test_a_cavity_at_another_reynolds_number_fails(tmp_path):
    needs_pyvista()
    case = _transformed_case(tmp_path, lambda p: p, lambda u: u, nu=0.01)
    result = grade.grade_cavity(case)
    assert not result["passed"]
    by = {c["name"]: c for c in result["checks"]}
    assert by["Reynolds number"]["value"] == pytest.approx(100.0)


# -- the cylinder ----------------------------------------------------------------------


def test_the_drag_check_is_two_percent_of_the_reference():
    assert grade.cylinder_checks(cd=5.52, reynolds=20.0, um=0.3)[0]["passed"]
    assert not grade.cylinder_checks(cd=5.40, reynolds=20.0, um=0.3)[0]["passed"]


def test_a_real_felix_schafer_turek_run_passes():
    needs_pyvista()
    result = grade.grade_cylinder(CYLINDER)
    assert result["passed"], result
    by = {c["name"]: c for c in result["checks"]}
    assert by["drag coefficient"]["value"] == pytest.approx(5.521, abs=0.002)
    assert by["Reynolds number"]["value"] == pytest.approx(20.0, rel=0.01)
    assert result["geometry"]["diameter"] == pytest.approx(0.1, rel=0.02)
    assert result["geometry"]["depth"] == pytest.approx(0.02)
    assert result["geometry"]["um"] == pytest.approx(0.3, rel=0.01)
    assert result["steady"]


def test_a_drag_ten_percent_high_fails(tmp_path):
    needs_pyvista()
    case = tmp_path / "cyl"
    shutil.copytree(CYLINDER, case)
    csv = case / "output" / "forces.csv"
    lines = []
    for line in csv.read_text().splitlines():
        if line[:1].isdigit():
            cells = line.split(",")
            cells[2] = repr(float(cells[2]) * 1.1)
            line = ",".join(cells)
        lines.append(line)
    csv.write_text("\n".join(lines) + "\n")
    result = grade.grade_cylinder(case)
    assert not result["passed"]


def test_a_case_without_a_forces_csv_cannot_be_graded(tmp_path, capsys):
    needs_pyvista()
    case = tmp_path / "cyl"
    shutil.copytree(CYLINDER, case)
    (case / "output" / "forces.csv").unlink()
    assert grade.main(["cylinder", str(case)]) == 2
    assert "forces" in capsys.readouterr().err


def test_the_command_line_prints_json_and_exits_by_the_verdict(capsys):
    needs_pyvista()
    assert grade.main(["cavity", str(CAVITY), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["case"] == "cavity" and result["passed"]


# -- the prompts --------------------------------------------------------------------


def test_the_prompts_are_there_and_name_their_references():
    folder = ROOT / "benchmarks" / "felix"
    cavity = (folder / "cavity_re1000.txt").read_text(encoding="utf-8")
    cylinder = (folder / "cylinder_re20.txt").read_text(encoding="utf-8")
    assert "Ghia" in cavity and "1000" in cavity
    assert "Sch" in cylinder and "drag" in cylinder
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert "openreynolds --solver felix -p" in readme
    assert "grade.py cavity" in readme and "grade.py cylinder" in readme
