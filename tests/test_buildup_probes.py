"""The silent failures, on surfaces built to hold each one.

Every probe here is `dormant`, which means the agent is told nothing about any of them.
That makes these tests the only demonstration that they work at all -- a dormant probe has
no run behind it yet, and the day it fires is not the day to find out it was measuring the
wrong thing. Each test builds the defect deliberately: a cube with a face missing, a cube
with one triangle wound backwards, a meshing point outside the part, a surface a thousand
times the size the request asked for.

The geometry is a unit cube because the numbers have to be checkable by eye. The machinery
underneath is the toolbox's -- `cad_audit.py`, `domain_probe.py`, `surfaces.py` -- and is
tested where it lives; what is pinned here is what counts as fired.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from openreynolds.buildup import probes

DOC = Path(__file__).resolve().parents[1] / "docs" / "cad-silent-failures.md"

CORNERS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

CUBE = [(0, 2, 1), (0, 3, 2),      # z = 0
        (4, 5, 6), (4, 6, 7),      # z = 1
        (0, 1, 5), (0, 5, 4),      # y = 0
        (1, 2, 6), (1, 6, 5),      # x = 1
        (2, 3, 7), (2, 7, 6),      # y = 1
        (3, 0, 4), (3, 4, 7)]      # x = 0
"""A closed unit cube, wound outwards. Twelve triangles, no free edges, one winding."""


def stl(triangles, scale: float = 1.0) -> str:
    lines = ["solid part"]
    for a, b, c in triangles:
        lines.append("facet normal 0 0 0")
        lines.append("  outer loop")
        for corner in (a, b, c):
            x, y, z = (value * scale for value in CORNERS[corner])
            lines.append(f"    vertex {x:.6f} {y:.6f} {z:.6f}")
        lines += ["  endloop", "endfacet"]
    return "\n".join(lines + ["endsolid part", ""])


def case(tmp_path: Path, triangles=None, *, scale: float = 1.0, point=None,
         manifest: dict | None = None) -> Path:
    """A case directory with an exported patch set, as a run would leave one."""
    root = tmp_path / "case"
    surface = root / "constant" / "triSurface"
    surface.mkdir(parents=True, exist_ok=True)
    (surface / "walls.stl").write_text(stl(CUBE if triangles is None else triangles,
                                           scale), encoding="utf-8")
    if manifest is not None:
        (surface / "patches.json").write_text(json.dumps(manifest), encoding="utf-8")
    if point is not None:
        system = root / "system"
        system.mkdir(parents=True, exist_ok=True)
        (system / "snappyHexMeshDict").write_text(
            "castellatedMeshControls\n{\n    locationInMesh (%s %s %s);\n}\n" % tuple(point),
            encoding="utf-8")
    return root


def one(results, probe_id):
    return next(result for result in results if result.id == probe_id)


# -- the surface itself ----------------------------------------------------------


def test_a_closed_cube_reads_as_closed_wound_and_clear(tmp_path):
    """The numbers a clean surface produces. Not "passes": these are instruments, and a
    reading of zero free edges is a reading, not a verdict."""
    found = probes.run_all(case(tmp_path, point=(0.5, 0.5, 0.5)), {"extent_m": 1.0})
    assert probes.fired(found) == []
    assert {one(found, name).state for name in
            ("union_closure", "normals", "self_intersection", "location_in_mesh",
             "scale")} == {probes.MEASURED}
    assert one(found, "union_closure").measured["open_edges"] == 0
    assert one(found, "normals").measured["flipped_edges"] == 0
    assert one(found, "self_intersection").measured["pairs"] == 0
    assert one(found, "location_in_mesh").measured["classification"] == "inside"


def test_a_face_missing_from_the_export_fires_union_closure(tmp_path):
    """The leak that gives the wrong fluid volume and still checks clean."""
    leaky = [tri for tri in CUBE if tri not in ((4, 5, 6), (4, 6, 7))]
    result = one(probes.run_all(case(tmp_path, leaky), {}), "union_closure")
    assert result.state == probes.MEASURED
    assert result.measured["open_edges"] == 4


def test_the_closure_check_is_on_the_union_and_not_per_file(tmp_path):
    """A patch file is an open surface by construction -- an inlet disc has a rim -- so a
    per-file check fails every correct export there has ever been. Split the same closed
    cube across two files and it stays closed."""
    root = case(tmp_path, CUBE[:6])
    (root / "constant" / "triSurface" / "lid.stl").write_text(stl(CUBE[6:]),
                                                              encoding="utf-8")
    result = one(probes.run_all(root, {}), "union_closure")
    assert result.measured["open_edges"] == 0 and len(result.measured["files"]) == 2


def test_one_triangle_wound_backwards_fires_normals(tmp_path):
    """What snappy reads as a hole, and why it meshes the outside of the object."""
    flipped = [(0, 1, 2) if tri == (0, 2, 1) else tri for tri in CUBE]
    result = one(probes.run_all(case(tmp_path, flipped), {}), "normals")
    assert result.state == probes.MEASURED and result.measured["flipped_edges"] > 0


def test_a_surface_that_crosses_itself_fires_self_intersection(tmp_path):
    root = case(tmp_path)
    crossing = ("solid x\n"
                "facet normal 0 0 0\n outer loop\n"
                "  vertex 0.2 0.5 0.2\n  vertex 0.8 0.5 0.2\n  vertex 0.5 0.5 0.8\n"
                " endloop\nendfacet\n"
                "facet normal 0 0 0\n outer loop\n"
                "  vertex 0.5 0.2 0.3\n  vertex 0.5 0.8 0.3\n  vertex 0.5 0.5 0.7\n"
                " endloop\nendfacet\nendsolid x\n")
    (root / "constant" / "triSurface" / "fin.stl").write_text(crossing, encoding="utf-8")
    result = one(probes.run_all(root, {}), "self_intersection")
    assert result.state == probes.MEASURED and result.measured["pairs"] >= 1


# -- the point, the scale, the manifest -------------------------------------------


def test_a_meshing_point_outside_the_part_fires(tmp_path):
    """`checkMesh` passes a perfectly valid mesh of the volume around the part, and
    nothing inside the run can see it."""
    result = one(probes.run_all(case(tmp_path, point=(5, 5, 5)), {}), "location_in_mesh")
    assert result.state == probes.MEASURED
    assert result.measured["classification"] == "outside"
    assert result.measured["source"] == "system/snappyHexMeshDict"
    # And `outside` is reported, never judged: T2 of the first baseline was a correct
    # external-flow case whose seed point is outside the part by definition.
    assert "outside" in result.why and "wrong" not in result.why


def test_the_point_is_read_off_the_case_and_not_off_the_conversation(tmp_path):
    """What was meshed is what the dictionary said. A desk that discussed one point and
    wrote another is exactly the run this probe is for."""
    root = case(tmp_path, point=(0.5, 0.5, 0.5),
                manifest={"location_in_mesh": [9, 9, 9], "patches": []})
    result = one(probes.run_all(root, {}), "location_in_mesh")
    assert result.measured["point"] == [0.5, 0.5, 0.5]
    assert result.measured["classification"] == "inside"


def test_a_case_with_no_meshing_point_is_not_applicable_rather_than_clean(tmp_path):
    """The instrument had nothing to read. Named `n/a` rather than `skipped` because a
    column of `skipped` reads as a check that ran -- which is how the first baseline came
    to report 79% of its probe verdicts as screening when nothing had been screened."""
    result = one(probes.run_all(case(tmp_path), {}), "location_in_mesh")
    assert result.state == probes.NOT_APPLICABLE


def test_millimetres_read_as_metres_show_up_as_a_ratio_of_a_thousand(tmp_path):
    result = one(probes.run_all(case(tmp_path, scale=1000.0), {"extent_m": 1.0}), "scale")
    assert result.state == probes.MEASURED and round(result.measured["ratio"]) == 1000


def test_scale_needs_a_stated_dimension_and_the_case_that_matters_has_none(tmp_path):
    """The probe written for the factor of a thousand cannot read the factor-of-a-thousand
    case. T6 of the first baseline is a STEP that declares no unit, the desk guessed
    millimetres and shipped a mesh checkMesh passed -- and this probe measures against
    "the extent the case states", which T6 by construction does not."""
    result = one(probes.run_all(case(tmp_path, scale=1000.0), {}), "scale")
    assert result.state == probes.NOT_APPLICABLE and "state" in result.why


def test_coverage_without_a_manifest_is_skipped_rather_than_guessed(tmp_path):
    """Nothing declares which face was meant to be whose, so double-assignment is
    unmeasurable -- and saying so beats reporting a pass nobody earned."""
    assert one(probes.run_all(case(tmp_path), {}),
               "coverage").state == probes.NOT_APPLICABLE


def test_a_probe_that_throws_is_a_probe_result_and_not_a_dead_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "_scale", lambda case, spec: 1 / 0)
    monkeypatch.setitem(probes._ONE, "scale", probes._scale)
    result = one(probes.run_all(case(tmp_path), {}), "scale")
    assert result.state == probes.ERROR and "ZeroDivisionError" in result.why


# -- the registry ----------------------------------------------------------------


def test_no_probe_carries_a_verdict_any_more():
    """The registry measures; the supervisor judges.

    Three probes fired across the first baseline and the supervisor overturned all three,
    while the two runs that were really wrong produced no probe signal at all. Every
    measurement was right and every verdict was wrong, so the verdicts are gone: nothing
    in the registry returns `fired` or `pass`, and no exit code turns on one.

    This used to assert every row was `dormant`, which was a census of the registry rather
    than the property the test is named for -- and it went false on 2026-09-13 when four
    rows were activated. **Activation is orthogonal to verdicts**, and that is the thing
    worth pinning: an active probe still only measures, and what its numbers mean is still
    the supervisor's to say and now also the desk's to answer at `declare_complete`."""
    assert {probe.state for probe in probes.REGISTRY} <= {probes.DORMANT, probes.ACTIVE}
    assert probes.ACTIVE in {probe.state for probe in probes.REGISTRY}, (
        "four rows are active; a registry that has gone all-dormant again has lost them")
    root = case(Path(tempfile.mkdtemp()), point=(5, 5, 5))
    found = probes.run_all(root, {"extent_m": 1.0})
    assert probes.fired(found) == []
    assert {r.state for r in found} <= {probes.MEASURED, probes.NOT_APPLICABLE,
                                        probes.ERROR}


def test_the_registry_document_and_the_code_hold_the_same_probes():
    """A row that has drifted from the code is a failing test rather than a stale
    document -- the registry is cited from run records, so it has to be true."""
    rows = re.findall(r"^\| `([a-z_]+)` \|.*\| (dormant|active) \|",
                      DOC.read_text(encoding="utf-8"), re.M)
    assert [row[0] for row in rows] == [probe.id for probe in probes.REGISTRY]
    assert [row[1] for row in rows] == [probe.state for probe in probes.REGISTRY]
