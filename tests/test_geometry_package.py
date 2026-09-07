"""The geometry package's skeleton: the lazy desk, the toolbox loader, the dependency
graph of DESIGN.md section 2, and the round trips of the shared dataclasses.

`openreynolds/geometry.py` became `openreynolds/geometry/desk.py` (D1) and the package
re-exports the desk's names on first use, never on import, so the kernel (everything but
`desk.py`) stays importable where only gmsh is (D27). These tests pin that shape module
by module in a fresh `-I` child, the way the cli shim runs the kernel, so a runtime
import that would create a cycle or drag the agent into the kernel fails here and not on
the integration day.
"""
from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from openreynolds import geometry
from openreynolds.geometry import _toolbox

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "openreynolds" / "geometry"
FIXTURES = ROOT / "tests" / "fixtures" / "geometry"

KERNEL = ["_toolbox", "sketch", "compile", "measure", "lint", "claims", "report", "preview",
          "fitness", "strategy", "runner", "case", "library", "cli"]

IMPORTS = {
    # DESIGN.md section 2: an arrow means "imports from" at runtime
    "_toolbox": [],
    "sketch": [],
    "compile": ["sketch", "_toolbox"],
    "measure": ["_toolbox"],
    "lint": ["measure"],
    "claims": ["measure", "lint"],
    "report": ["measure", "lint", "claims", "_toolbox"],
    "preview": ["measure", "lint", "claims", "_toolbox"],
    "fitness": [],
    "strategy": ["_toolbox"],
    "runner": [],
    "case": ["compile", "fitness", "_toolbox"],
    "library": ["sketch", "compile", "measure", "lint", "claims", "preview"],
    "cli": ["sketch", "compile", "measure", "lint", "claims", "report", "preview", "library", "runner"],
}

DESK_NAMES = {"GeometryAgent", "GeometryResult", "GEOMETRY_SYSTEM", "unavailable", "extract_spec",
              "grammar", "MAX_LAPS", "MAX_SECONDS", "MAX_REPLY_TOKENS", "FINISH_TIMEOUT_S", "COMMIT"}
"""The re-exported names today's desk carries (BRIEF, extract_reply and RUN_TIMEOUT_S
arrive with U5 and resolve the same way)."""


def closure(module: str) -> set[str]:
    seen: set[str] = set()
    todo = [module]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo.extend(IMPORTS[m])
    return seen


def child(code: str) -> subprocess.CompletedProcess:
    """A fresh interpreter under -I with the checkout root on sys.path, as the cli shim
    puts it there (D3)."""
    prelude = f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
    return subprocess.run([sys.executable, "-I", "-c", prelude + code], capture_output=True, text=True,
                          timeout=120, env={"PATH": "", "PYTHONUTF8": "1", "SYSTEMROOT": "C:\\Windows"})


# -- the lazy desk (3.1) -------------------------------------------------------------


def test_the_package_never_imports_the_desk_on_import():
    """Reloading re-executes `__init__.py` in the package's own namespace, so the `desk`
    attribute an earlier import bound there is dropped too: otherwise `from . import
    desk` would find it on the package and never touch `sys.modules`."""
    sys.modules.pop("openreynolds.geometry.desk", None)
    vars(geometry).pop("desk", None)
    package = importlib.reload(geometry)
    assert "openreynolds.geometry.desk" not in sys.modules and "desk" not in vars(package)
    assert package.GeometryAgent.__name__ == "GeometryAgent"
    assert "openreynolds.geometry.desk" in sys.modules


def test_the_desk_names_resolve_through_the_package_and_nothing_else_does():
    from openreynolds.geometry import desk
    for name in DESK_NAMES:
        assert getattr(geometry, name) is getattr(desk, name), name
        assert name in dir(geometry)
    with pytest.raises(AttributeError, match="has no attribute 'nothing_here'"):
        geometry.nothing_here
    # today's importers keep working unchanged: the package is what `openreynolds.cli`
    # and `tools.py` import, and its file is the package's, not the desk's
    assert Path(geometry.__file__).name == "__init__.py"
    assert (Path(geometry.__file__).parents[1] / "cli.py").exists()


def test_the_desk_still_finds_its_toolbox_after_the_move():
    from openreynolds.geometry import desk
    assert desk.TOOLBOX == ROOT / "openreynolds" / "toolbox"
    assert (desk.TOOLBOX / "mesh2d.py").exists()
    assert "mesh2d" in desk.build_args("2d", Path("s.json"), Path("out"), 0.001)[1]


# -- the toolbox loader (3.2) ---------------------------------------------------------


def test_load_returns_the_same_module_twice():
    assert _toolbox.load("mesh_digest") is _toolbox.load("mesh_digest")
    assert _toolbox.load("mesh_digest").__name__ == "toolbox_mesh_digest"
    assert _toolbox.TOOLBOX == ROOT / "openreynolds" / "toolbox"


def test_load_finds_mesh2d_build_face():
    mesh2d = _toolbox.load("mesh2d")
    assert callable(mesh2d.build_face) and callable(mesh2d.parse_spec) and callable(mesh2d.leg_lines)


def test_load_names_the_scripts_when_one_is_missing():
    with pytest.raises(FileNotFoundError, match="mesh2d, case_gen and mesh_digest"):
        _toolbox.load("no_such_script")


# -- the dependency graph (section 2) -------------------------------------------------


def test_every_module_of_the_layout_imports():
    for name in KERNEL + ["desk"]:
        importlib.import_module(f"openreynolds.geometry.{name}")
    assert (PACKAGE / "reference.md").exists()


def _escaping_imports(path: Path) -> list[str]:
    """Import lines that reach outside the geometry package: `openreynolds` by name, or
    enough dots to climb past `openreynolds.geometry` (one `..` from a module in the
    package, `...` from one in a subpackage). The cli's `__main__` shim is the one
    sanctioned absolute import (D3) and runs only in the child; it is skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.name == "cli.py":
        kept, in_shim = [], False
        for line in lines:
            if line.startswith('if __name__ == "__main__"'):
                in_shim = True
                continue
            if in_shim and (not line.strip() or line[0] in " \t"):
                continue
            in_shim = False
            kept.append(line)
        lines = kept
    depth = len(path.relative_to(PACKAGE).parts) - 1
    escaping = "." * (depth + 2)
    hits = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"(from|import)\s+openreynolds\b", stripped):
            hits.append(stripped)
        elif re.match(rf"from\s+{re.escape(escaping)}", stripped):
            hits.append(stripped)
    return hits


def test_the_kernel_imports_nothing_from_the_agent():
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "desk.py":
            continue
        assert _escaping_imports(path) == [], path.relative_to(ROOT)
    # and the desk does reach out, which is the point of keeping it apart
    assert _escaping_imports(PACKAGE / "desk.py")


@pytest.mark.parametrize("module", KERNEL)
def test_kernel_import_closure(module):
    """In a fresh -I child, importing one kernel module loads exactly the section-2
    closure for it: a runtime import of a lower module (the cycle compile <-> measure
    <-> claims an annotation import would create) fails here, and the child-side import
    name works on this platform."""
    proc = child(f"import openreynolds.geometry.{module}, sys; "
                 "print(sorted(k for k in sys.modules if k.startswith('openreynolds.geometry')))")
    assert proc.returncode == 0, proc.stderr
    loaded = set(json.loads(proc.stdout.strip().replace("'", '"')))
    # the library's own entries (tesla_valve, ...) are part of "library" in the table
    loaded = {re.sub(r"^openreynolds\.geometry\.library\..*$", "openreynolds.geometry.library", k) for k in loaded}
    expected = {"openreynolds.geometry"} | {f"openreynolds.geometry.{m}" for m in closure(module)}
    assert loaded == expected


@pytest.mark.parametrize("module", ["sketch", "fitness", "runner"])
def test_sketch_fitness_and_runner_import_no_gmsh_at_module_level(module):
    proc = child(f"import openreynolds.geometry.{module}, sys; "
                 "print('gmsh' in sys.modules, 'matplotlib' in sys.modules, 'numpy' in sys.modules)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["False", "False", "False"]


def test_importing_the_package_loads_no_kernel_and_no_agent():
    proc = child("import openreynolds.geometry, sys; "
                 "print(sorted(k for k in sys.modules if k.startswith('openreynolds')))")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "['openreynolds', 'openreynolds.geometry']"


def test_the_cli_runs_by_path_under_dash_i_through_the_shim():
    """`python -I openreynolds/geometry/cli.py` imports `openreynolds.geometry.cli` through
    the parents[2] shim: the failure it reaches is the cli's own (argparse refusing a
    `build` with no `--out` now that U4 built it; the skeleton's NotImplementedError
    before), never a ModuleNotFoundError for `openreynolds`."""
    proc = subprocess.run([sys.executable, "-I", str(PACKAGE / "cli.py"), "build"], capture_output=True,
                          text=True, timeout=120, env={"PATH": "", "PYTHONUTF8": "1", "SYSTEMROOT": "C:\\Windows"})
    assert proc.returncode != 0
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr
    assert "geometry build" in proc.stderr and "--out" in proc.stderr, proc.stderr


# -- the fixtures (8.1) -----------------------------------------------------------------


PROBE_SPECS = ["elbow.json", "lduct.json", "penne.json", "straight.json", "tesla_committed_4025.json",
               "tesla_real_attempt1.json", "tesla_real_attempt3.json", "uduct.json"]


def test_the_probe_specs_are_fixtures_under_their_assessment_names():
    for name in PROBE_SPECS:
        spec = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        assert isinstance(spec, (dict, list)), name
    two_d = [n for n in PROBE_SPECS if n.startswith("tesla")]
    for name in two_d:
        spec = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        assert spec["scale"] == 0.001 and spec["ops"][-1]["op"] == "fuse", name


# -- the shared dataclasses round-trip through JSON (section 3's convention) ------------


def roundtrip(value):
    return type(value).from_dict(json.loads(json.dumps(value.as_dict())))


def test_a_finding_round_trips_and_renders_the_section_5_shape():
    from openreynolds.geometry.lint import Finding, errors, has_errors, warnings
    f = Finding(level="error", code="E-OVERLAP", subject="Row 'loops'",
                what="loops[0] and loops[1] overlap by 26.7 mm2 at (17.7, 6.4)",
                fix="the pitch 12 is smaller than the footprint 19.74 plus a gap", where=(17.7, 6.4),
                numbers={"area": 26.7})
    assert roundtrip(f) == f
    assert f.text().startswith("!! ERROR  E-OVERLAP  Row 'loops': loops[0] and loops[1] overlap by 26.7")
    assert f.text().splitlines()[1] == "          the pitch 12 is smaller than the footprint 19.74 plus a gap"
    assert f.as_legacy() == {"level": "error", "where": "Row 'loops'", "what": f.what}
    w = Finding(level="warn", code="W-GAP", subject="Row 'loops'", what="0.2 apart", where=(1.0, 2.0))
    i = Finding(level="info", code="I-GAP", subject="Row 'loops'", what="2.44 apart")
    assert w.text().startswith("!!        W-GAP") and i.text().startswith("check     I-GAP")
    assert has_errors([f, w]) and not has_errors([w, i])
    assert errors([f, w, i]) == [f] and warnings([f, w, i]) == [w]


def test_measurements_round_trip_with_their_nested_records():
    from openreynolds.geometry import measure as M
    m = M.Measurements(
        units="mm", scale=0.001, bounds=(0.0, -1.5, 60.0, 12.75), extent=(60.0, 14.25), area=513.7, islands=4,
        n_curves=36, shortest_edge=(1.5, (0.75, 1.5)),
        patches={"inlet": {"curves": [1], "n": 1, "length": 3.0, "midpoints": [[0.0, 0.0]]}},
        legs={"main": [{"kind": "line", "length": 60}]}, features={"main": {"width": 3.0}},
        rows={"loops": M.RowMeasure(name="loops", count=4, pitch=14.66, footprint=(13.02, 11.25), gap=1.64,
                                    gap_measured=2.44, anchors=[7.24, 21.9, 36.56, 51.22], span=(1.5, 58.5))},
        open_ends=[M.OpenEnd(curve=1, centre=(0.0, 0.0), length=3.0, outward_normal=(-1.0, 0.0), depth=9.0,
                             corner_angles=(90.0, 90.0), name="inlet")],
        junctions=[M.Junction(channel="loops[0]", where=(7.24, 1.5), wall_line="y:1.5", kind="leave",
                              typed_angle=20.0, measured_angle=20.0, lip=((11.42, 1.5), 28.9), heading=20.0,
                              against_flow=False)],
        vertices=[M.VertexInfo(at=(0.0, 0.0), curves=(36, 1), interior_deg=90.0, solid_deg=270.0, kind="convex")],
        passage=M.PassageWidths(min=3.0, where_min=(30.0, 0.0), p10=3.0, median=3.0, n=7000),
        reference_width=3.0, reference_width_from="declared by 'main'",
        holes=[M.HoleMeasure(loop=1, area=8.6, centroid=(8.1, 6.0), radius=None, circumference=12.0)],
        flow=(1.0, 0.0))
    assert roundtrip(m) == m
    wk = M.Walk(loops=[[1, 2, 3]], curves={1: M.CurveInfo(
        tag=1, kind="line", length=3.0, start=(0.0, -1.5), end=(0.0, 1.5), heading_in=90.0, heading_out=90.0,
        radius=None, centre=None, sweep=None, patch="inlet", loop=0, midpoint=(0.0, 0.0),
        inward_normal=(1.0, 0.0), samples=[(0.0, -1.5), (0.0, 1.5)])},
        vertices=[], signed_area=651.0, repaired=False)
    assert roundtrip(wk) == wk


def test_claims_and_the_compliance_table_round_trip_and_count_by_kind():
    from openreynolds.geometry import claims as C
    cs = C.ClaimSet(unit="mm", kind="passage", flow="+x", claims=[
        C.Claim(id="c1", kind="measure", says="3 mm wide", measure="width", of="main", value=3),
        C.Claim(id="c12", kind="not_measurable", says="mesh it", not_measurable="the finish does it")],
        largest_length=60.0)
    assert roundtrip(cs) == cs
    assert C.WARNINGS_BLOCK_COMMIT is False
    assert "width" in C.LENGTH_MEASURES and "sweep" not in C.LENGTH_MEASURES
    rows = [C.ComplianceRow(id="c1", says="3 mm wide", kind="measure", expected="3 +/- 0.03",
                            measured="3.000 (main.width)", verdict="pass"),
            C.ComplianceRow(id="c10", says="sweep round", kind="report", expected="reported",
                            measured="240 deg (x4)", verdict="pass"),
            C.ComplianceRow(id="c9", says="outlet right", kind="patch", expected="right",
                            measured="outlet at (0, 18): the LEFT end", verdict="fail", detail="an even number of passes")]
    table = C.ComplianceTable(rows)
    assert roundtrip(table) == table
    assert (table.passed, table.failed, table.unmeasurable, table.disagreed, table.checkable) == (2, 1, 0, 0, 2)
    assert table.failing_ids() == ["c9"] and not table.ok()
    assert C.ComplianceTable(rows[:1]).ok()
    # D33: an empty or report-only table never passes
    assert not C.ComplianceTable([]).ok() and not C.ComplianceTable([rows[1]]).ok()


def test_a_fitness_table_refuses_placeholders_and_round_trips_unmeasured():
    from openreynolds.geometry.fitness import FitnessTable, Unmeasured
    absent = Unmeasured("no 2D passage measure for a 3D body in Phase 1")
    table = FitnessTable(cells=3750, hex_fraction=1.0, smallest_passage_m=absent, cell_m=4.75e-4,
                         cells_across_smallest_passage=absent, wall_cell_m=1e-4, first_cell_height_m=5e-5,
                         y_plus=Unmeasured("no flow speed in a mesh-only study"),
                         y_plus_verdict=Unmeasured("no flow speed in a mesh-only study"),
                         non_orth_max=42.32, non_orth_mean=7.26, skew_max=1.14,
                         aspect_max=Unmeasured("checkMesh's log has no aspect ratio line"), schemes="standard",
                         measured_from="log.checkMesh + mesh_sizes2d")
    back = roundtrip(table)
    assert back == table and back.smallest_passage_m == absent
    assert table.as_dict()["smallest_passage_m"] == {"unmeasured": absent.reason}
    assert table.quality_gate == "not applied"
    with pytest.raises(ValueError, match="FitnessTable.cells is required"):
        FitnessTable(**{**table.as_dict(), "cells": None, "smallest_passage_m": absent,
                        "cells_across_smallest_passage": absent, "y_plus": table.y_plus,
                        "y_plus_verdict": table.y_plus_verdict, "aspect_max": table.aspect_max})
    with pytest.raises(ValueError, match="skew_max is required"):
        FitnessTable(**{**table.as_dict(), "skew_max": float("nan"), "smallest_passage_m": absent,
                        "cells_across_smallest_passage": absent, "y_plus": table.y_plus,
                        "y_plus_verdict": table.y_plus_verdict, "aspect_max": table.aspect_max})
    with pytest.raises(ValueError, match="smallest_passage_m is required"):
        FitnessTable(**{**table.as_dict(), "smallest_passage_m": None,
                        "cells_across_smallest_passage": absent, "y_plus": table.y_plus,
                        "y_plus_verdict": table.y_plus_verdict, "aspect_max": table.aspect_max})
    with pytest.raises(ValueError, match="Unmeasured exactly when"):
        FitnessTable(**{**table.as_dict(), "smallest_passage_m": 0.00298,
                        "cells_across_smallest_passage": absent, "y_plus": table.y_plus,
                        "y_plus_verdict": table.y_plus_verdict, "aspect_max": table.aspect_max})
    with pytest.raises(ValueError, match="measured_from is required"):
        FitnessTable(**{**table.as_dict(), "measured_from": "", "smallest_passage_m": absent,
                        "cells_across_smallest_passage": absent, "y_plus": table.y_plus,
                        "y_plus_verdict": table.y_plus_verdict, "aspect_max": table.aspect_max})


def test_a_plan_and_a_run_outcome_round_trip():
    from openreynolds.geometry.compile import Plan, Solved
    from openreynolds.geometry.runner import RunOutcome
    from openreynolds.geometry.sketch import PortIntent
    plan = Plan(units="mm", scale=0.001, ops=[{"op": "rect", "name": "fluid", "origin": [0, 0], "size": [60, 3]}],
                rules=[{"name": "inlet", "at": "x:min"}],
                ports=[PortIntent(name="inlet", kind="inlet", edges=("x:min",))],
                features={"duct": Solved(kind="Rect", params={"size": [60, 3]}, solved={}, ops=["fluid"])},
                instances={}, outlines={"duct": [[(0.0, 0.0), (60.0, 0.0), (60.0, 3.0), (0.0, 3.0)]]},
                edge_targets={"duct.left": {"kind": "side", "points": [[0, 0], [0, 3]], "width": 3}},
                apart=[("a", "b", None)], declared_widths=[3.0], expected=(1, 1), notes=["a note"],
                clip_boxes=[("duct", (0.0, 0.0, 60.0, 3.0))])
    assert roundtrip(plan) == plan
    outcome = RunOutcome(rc=0, text="SCRIPT ran", png=b"\x89PNG", seconds=1.2, result={
        "compliance": {"rows": [{"id": "c1", "verdict": "pass"}, {"id": "c9", "verdict": "fail"}]},
        "lint": [{"level": "info"}]})
    assert roundtrip(outcome) == outcome
    assert outcome.failing_claims == ["c9"] and not outcome.ready and outcome.lint_clean
    assert RunOutcome(rc=0, text="", png=None, seconds=0.5, result={"compliance": {"rows": []}, "lint": []}).ready
    assert not RunOutcome(rc=0, text="", png=None, seconds=0.5, result={"compliance": None, "lint": []}).ready


def test_a_sketch_error_carries_its_code_and_prints_the_section_5_shape():
    from openreynolds.geometry.sketch import SCALE, SketchError
    err = SketchError("E-ROW-PITCH", "Row 'pins'", "pitch 2 is smaller than footprint 1.2 + gap 1.0 = 2.2",
                      "pitch >= 2.2, or gap <= 0.8, or leave pitch out and the gap sets it", numbers={"pitch": 2})
    assert err.code == "E-ROW-PITCH" and err.partial is None and err.numbers == {"pitch": 2}
    assert str(err) == ("!! ERROR  E-ROW-PITCH  Row 'pins': pitch 2 is smaller than footprint 1.2 + gap 1.0 = 2.2\n"
                        "          pitch >= 2.2, or gap <= 0.8, or leave pitch out and the gap sets it")
    assert isinstance(err, ValueError)
    assert SCALE == {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}
