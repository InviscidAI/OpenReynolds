"""What "done" means for a CAD-and-mesh run, as a function of the facts.

Ported from `tests/test_mesher_check.py`, every criterion it pinned still pinned --
a case written and read as meshed, a mesh with failing checks handed to a solver,
`patch0`/`patch1` out of gmshToFoam with nobody able to say which end was the inlet, a
mesh nobody could rebuild, and a geometry left in millimetres.

One criterion is deliberately replaced. The old check failed a mesh with fewer than two
named patches, "because a flow case needs at least an inlet, an outlet and walls". That
is true of through-flow and false of buoyancy driven by volumetric sources on cell
zones: a sealed cavity with every boundary an adiabatic wall legitimately has one patch,
checkMesh passes, the mesh is right, and the old desk could not finish. The property is
**named, not counted**, and what stands in its place is three things: a default name
still fails, a single chosen name passes, and one patch draws a warning that says what
it is assuming and does not block.

The tests that need OpenFOAM build real meshes and skip cleanly where it is not on the
path -- a multi-region case out of `splitMeshRegions`, a sealed cavity with two cell
zones, and a sheared mesh whose non-orthogonality sits between the thresholds the
toolbox warns and fails at, which checkMesh accepts and this check must not refuse.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import pytest

from openreynolds.backend.base import ExecResult
from openreynolds.backend.local import LocalBackend, find_bashrc
from openreynolds.cad import gate
from openreynolds.cad.check import (
    ASPECT_WARN, LOOK, NON_ORTHO_FAIL, NON_ORTHO_WARN, SCALE_SLACK, SKEWNESS_FAIL,
    SKEWNESS_WARN, Check, Finding, fingerprint, largest_length, look_command,
    mesh_regions, read, scale_mismatch, verify,
)

REPO = Path(__file__).resolve().parents[1]
TOOLBOX = REPO / "openreynolds" / "toolbox"

GOOD = {
    "polymesh": True, "cells": 3750, "faces": 15000, "points": 7600,
    "bounds": [0, 0, 0, 0.1, 0.05, 0.001], "two_d": True,
    "patches": [{"name": "inlet", "type": "patch", "nFaces": 20},
                {"name": "outlet", "type": "patch", "nFaces": 20},
                {"name": "walls", "type": "wall", "nFaces": 400}],
    "build": ["Allmesh"], "checkmesh": "Mesh OK.", "checkmesh_ok": True,
    "render": "renders/mesh_look.png",
}


def facts(**changes):
    payload = {k: (list(v) if isinstance(v, list) else v) for k, v in GOOD.items()}
    payload.update(changes)
    return payload


def finding(check, name):
    return next((f for f in check.findings if f.check == name), None)


# -- the floor, inherited ------------------------------------------------------


def test_the_good_case_passes():
    check = read(facts(), "mesh", "/work/s/mesh")
    assert check.ok and check.missing == []
    assert check.render == "mesh/renders/mesh_look.png"
    assert check.render_abs == "/work/s/mesh/renders/mesh_look.png"
    assert check.regions == [""]


def test_no_polymesh_is_the_first_thing_said():
    check = read(facts(polymesh=False), "mesh")
    assert not check.ok
    assert "nothing has been meshed yet" in check.missing[0]
    assert check.regions == []


def test_a_mesh_that_checkmesh_refuses_is_not_done():
    check = read(facts(checkmesh_ok=False, checkmesh="failed 2 mesh checks"), "mesh")
    assert not check.ok
    assert any("failed 2 mesh checks" in m and "log.checkMesh" in m for m in check.missing)


def test_default_mesher_names_are_refused_by_name():
    """gmshToFoam types and names every patch itself; a case that reaches here with
    `patch0` on it is one where nobody said which surface was the inlet."""
    check = read(facts(patches=[{"name": "patch0", "nFaces": 20},
                                {"name": "defaultFaces", "nFaces": 400}]), "mesh")
    assert not check.ok
    named = next(m for m in check.missing if "default names" in m)
    assert "patch0" in named and "defaultFaces" in named


@pytest.mark.parametrize("name", ["inlet", "outlet", "walls", "frontAndBack", "wall2", "duct3"])
def test_a_name_somebody_chose_is_accepted_even_with_a_digit_in_it(name):
    check = read(facts(patches=[{"name": name, "nFaces": 4},
                                {"name": "rest", "nFaces": 4}]), "mesh")
    assert "default names" not in " ".join(check.missing)


def test_patches_with_no_faces_do_not_count():
    """A patch with no faces on it is not a patch somebody named the flow through."""
    check = read(facts(patches=[{"name": "inlet", "nFaces": 0},
                                {"name": "walls", "nFaces": 400}]), "mesh")
    count = finding(check, "patch_count")
    assert count is not None and count.status == "warn"
    assert "one boundary patch" in count.measured and "walls" in count.measured


def test_a_mesh_with_no_rebuild_script_listed_is_still_a_finish():
    """The gate that used to be here failed the desk for the harness's own omission.

    The brief says the accepted cells "are concatenated into `build.py` in the case
    directory" and nothing wrote that file -- `_replay` put it under `.replay/`, which it
    deletes and recreates every check, and `build_files` reads only the case root. So a
    binding gate demanded an artifact the desk was told it already had, and the desk had
    no move: the cell log is append-only, and no later cell rewrites an earlier one.

    It also passed for the wrong reason more often than it failed, because `build_files`
    counts `system/blockMeshDict` -- present whenever blockMesh was used, and not a thing
    that rebuilds the geometry.

    What replaced it is `_leave_script`, which writes the file the brief promised. The
    claim that rests on it is the replay, and that is where a broken script is reported.
    """
    check = read(facts(build=[]), "mesh")
    assert check.ok, check.missing
    assert not any("rebuilt" in m for m in check.missing)
    assert finding(check, "build") is None, "reported as a fact, not judged"


def test_a_mesh_nobody_drew_is_not_finished():
    check = read(facts(render=""), "mesh")
    assert any("no picture" in m for m in check.missing)


def test_the_refusal_reads_as_work_not_as_a_verdict():
    check = read(facts(polymesh=False), "mesh")
    text = check.as_refusal()
    assert text.startswith("The check did not pass")
    assert "nothing has been meshed yet" in text
    assert "Fix it and say done again" in text


# -- named, not counted --------------------------------------------------------


def test_one_patch_named_walls_is_a_finish():
    """The sealed cavity: every boundary an adiabatic wall, convection driven by
    volumetric sources on cell zones. checkMesh passes, the mesh is right, and the old
    check refused it."""
    check = read(facts(patches=[{"name": "walls", "nFaces": 400}]), "mesh")
    assert check.ok, check.missing


def test_the_one_patch_warn_is_present_reasoned_and_does_not_block():
    check = read(facts(patches=[{"name": "walls", "nFaces": 400}]), "mesh")
    warn = finding(check, "patch_count")
    assert warn is not None and warn.status == "warn"
    assert check.ok and warn.measured not in check.missing
    # It names both mechanisms one patch is inconsistent with, and neither is a guess
    # dressed as a gate.
    assert "buoyancy" in warn.meaning and "hot patch" in warn.meaning
    assert "through-flow" in warn.meaning and "inlet and an outlet" in warn.meaning
    assert "closed domain" in warn.meaning
    assert "topoSet" in warn.repair and "createPatch" in warn.repair


@pytest.mark.parametrize("name", ["defaultFaces", "patch0", "region3", "surface1", "volume0"])
def test_dropping_the_count_does_not_drop_the_naming_rule(name):
    check = read(facts(patches=[{"name": name, "nFaces": 400}]), "mesh")
    assert not check.ok
    assert any("default names" in m and name in m for m in check.missing)


def test_a_mesh_with_no_patch_carrying_a_face_fails():
    check = read(facts(patches=[{"name": "walls", "nFaces": 0}]), "mesh")
    assert not check.ok
    assert any("no boundary patch with any faces" in m for m in check.missing)
    assert finding(check, "patches").repair


# -- checkMesh is the sole authority ------------------------------------------


def test_a_number_between_the_thresholds_that_checkmesh_accepts_is_not_refused():
    """Two authorities on one number is the bug this forbids: the thresholds phrase the
    finding, checkMesh decides."""
    between = (NON_ORTHO_WARN + NON_ORTHO_FAIL) / 2
    check = read(facts(metrics={"max_non_orthogonality": between, "max_skewness": 0.6}), "mesh")
    assert check.ok, check.missing
    verdict = finding(check, "checkmesh")
    assert verdict.status == "ok"
    assert f"{between:g}" in verdict.measured and str(int(NON_ORTHO_WARN)) in verdict.measured
    assert "do not overrule" in verdict.meaning


def test_a_mesh_checkmesh_fails_is_not_passed_on_a_pretty_metric():
    check = read(facts(checkmesh_ok=False, checkmesh="failed 1 mesh checks",
                       metrics={"max_non_orthogonality": 2.0}), "mesh")
    assert not check.ok


def test_the_threshold_copies_are_the_numbers_preflight_uses():
    """This module cannot import the toolbox, so the copies are pinned to its source."""
    source = (TOOLBOX / "preflight.py").read_text()
    for name, ours in (("NON_ORTHO_WARN", NON_ORTHO_WARN), ("NON_ORTHO_FAIL", NON_ORTHO_FAIL),
                       ("SKEWNESS_WARN", SKEWNESS_WARN), ("SKEWNESS_FAIL", SKEWNESS_FAIL),
                       ("ASPECT_WARN", ASPECT_WARN)):
        match = re.search(rf"^{name}\s*=\s*([\d.]+)", source, re.M)
        assert match, f"{name} is no longer in preflight.py under that name"
        assert float(match.group(1)) == ours, f"{name} disagrees with preflight.py"
    theirs = re.search(r"^SCALE_FACTOR\s*=\s*([\d.]+)", source, re.M)
    assert theirs and float(theirs.group(1)) == SCALE_SLACK


# -- every finding is evidence -------------------------------------------------


@pytest.mark.parametrize("payload", [
    facts(),
    facts(polymesh=False),
    facts(checkmesh_ok=False, checkmesh="failed 2 mesh checks"),
    facts(patches=[{"name": "patch0", "nFaces": 4}]),
    facts(patches=[{"name": "walls", "nFaces": 400}]),
    facts(patches=[]),
    facts(build=[]),
    facts(build=["mesh.py"]),
    facts(render=""),
    facts(bounds=[0, 0, 0, 74, 24, 1]),
    facts(error="the picture could not be drawn"),
])
def test_no_finding_is_prose(payload):
    check = read(payload, "mesh", "/work/s/mesh", "a passage 8 mm wide, 60 mm long")
    assert check.findings
    for item in check.findings:
        assert item.measured.strip(), f"{item.check} has no measured value"
        if item.status == "fail":
            assert item.repair.strip(), f"{item.check} fails with no repair"


# -- cell zones, reported ------------------------------------------------------


def composite(regions, **extra):
    payload = {"regions": regions, "meshed": True, "cad": [], "replay": {}}
    payload.update(extra)
    return payload


def test_cellzones_are_reported_with_names_and_counts():
    check = read(composite({"": {"look": facts(), "cellzones": [["heater", 200],
                                                               ["cooler", 150]]}}), "mesh")
    zones = finding(check, "cellzones")
    assert zones.status == "ok"
    assert "heater 200 cells" in zones.measured and "cooler 150 cells" in zones.measured
    assert check.ok


def test_a_mesh_with_no_cellzones_says_so_rather_than_omitting_the_line():
    check = read(composite({"": {"look": facts(), "cellzones": []}}), "mesh")
    zones = finding(check, "cellzones")
    assert zones is not None and zones.status == "ok"
    assert "no cellZones" in zones.measured


def test_cellzones_nobody_read_are_skipped_rather_than_reported_as_none():
    check = read(facts(), "mesh")
    zones = finding(check, "cellzones")
    assert zones.status == "skipped" and "not read" in zones.measured


# -- the rebuild script the bundle will take -----------------------------------


def test_the_script_the_brief_promises_is_actually_left_in_the_case():
    """The brief's sentence, made true rather than deleted.

    `build.py` is in `casebundle.DEFINITION_NAMES`, so the bundle carries it, and
    `mesh_look.build_files` reads the case root, so the check reports it. Writing it is
    what the removed gate was reaching for and it needs no gate to happen.
    """
    from openreynolds.cad.check import REPLAY_SCRIPT, _leave_script
    from openreynolds.casebundle import DEFINITION_NAMES

    assert REPLAY_SCRIPT in DEFINITION_NAMES, "the bundle has to carry what we leave"

    written: dict[str, bytes] = {}

    class Workspace:
        def put_file(self, path, data):
            written[path] = data

    _leave_script(Workspace(), "/work/s/mesh", "PLATE_L_M = 0.12\n")
    assert written == {"/work/s/mesh/build.py": b"PLATE_L_M = 0.12\n"}

    # A workspace that will not take it is not a verdict about the mesh: the replay is
    # where a script that is missing or broken gets reported.
    class Refuses:
        def put_file(self, path, data):
            raise RuntimeError("read-only")

    _leave_script(Refuses(), "/work/s/mesh", "x = 1")


def test_the_check_finds_the_script_we_leave(tmp_path):
    """The other half: written under a name `mesh_look.build_files` reports."""
    (tmp_path / "build.py").write_text("x = 1", encoding="utf-8")
    assert "build.py" in mesh_look_names(tmp_path)


def mesh_look_names(tmp: Path) -> list[str]:
    """What `mesh_look.build_files` actually reports, run against a real directory.

    Run rather than read: the point is that the file we write is the file the check
    finds, and a regex over the source proves neither half of that.
    """
    import sys

    sys.path.insert(0, str(TOOLBOX))
    import mesh_look  # noqa: E402  (sibling script, not a package)

    return mesh_look.build_files(tmp)


def test_a_dictionary_under_system_is_captured_and_counts():
    check = read(facts(build=["system/blockMeshDict"]), "mesh")
    assert check.ok, check.missing


def test_the_definition_names_are_not_restated_here():
    """They were imported while this module gated on them; now it does not gate at all.

    What must not come back is a second copy of the list, which is what the import was
    guarding against.
    """
    source = (REPO / "openreynolds" / "cad" / "check.py").read_text()
    assert '"Allmesh"' not in source and "'Allmesh'" not in source
    assert '"Allrun"' not in source and "'Allrun'" not in source


# -- the CAD findings are consumed, not recomputed -----------------------------


AUDIT_FAIL = {
    "script": "cad_audit", "ok": False,
    "findings": [
        {"check": "closure", "status": "fail",
         "measured": "4 open edges on the union of 6 patch files",
         "means": "the exported surface is not closed, so snappy will leak into the solid",
         "repair": "the missing quad is on outlet.stl; re-export it"},
        {"check": "normals", "status": "ok", "measured": "0 same-direction edges",
         "means": "", "repair": ""},
    ],
    "measured": {"triangles": 1200},
}
PROBE_FAIL = {
    "script": "domain_probe", "ok": False,
    "findings": [{"check": "location_in_mesh", "status": "fail",
                  "measured": "(0.05 0.01 0.02) is 3.2 mm outside the fluid",
                  "means": "snappy would mesh the outside of the part",
                  "repair": "run domain_probe.py --suggest and write that point in"}],
    "measured": {},
}


def test_a_cad_fail_is_advisory_here_and_keeps_its_own_words():
    """It used to bind, and that was the defect.

    `ok` was `worst_status(findings) != "fail"` over every finding, and `cad_audit`'s are
    in that list, so an open surface failed the finish outright. A zero-thickness baffle
    is open by construction, `domain_probe` calls a seed point outside the surface wrong
    when external flow puts it there on purpose, and `coverage` warned on 19 of 20
    correct partitions in `core+cad_export-20260917-022129-dd05`. There was no waiver on
    this desk and re-declaring changed nothing, so a correct geometry could not be
    delivered at all.

    The words still arrive verbatim -- that half was always right, and `_cad_findings`
    carries `measured`, `means` and `repair` through without recomputing a triangle.
    What changed is that they reach the desk through `gate.evaluate` instead of through
    the verdict.
    """
    check = read(composite({"": {"look": facts(), "cellzones": []}},
                           cad=[AUDIT_FAIL, PROBE_FAIL]), "mesh")
    assert check.ok, "an advisory finding does not decide the finish"
    closure = finding(check, "cad_audit.closure")
    point = finding(check, "domain_probe.location_in_mesh")
    assert closure.measured == AUDIT_FAIL["findings"][0]["measured"]
    assert closure.repair == AUDIT_FAIL["findings"][0]["repair"]
    assert closure.meaning == AUDIT_FAIL["findings"][0]["means"]
    assert point.measured == PROBE_FAIL["findings"][0]["measured"]
    assert point.repair == PROBE_FAIL["findings"][0]["repair"]
    # Not in `missing`, which is the list the desk is handed as the reasons it is not
    # done. An advisory warning is not one of those.
    assert closure.measured not in check.missing
    assert point.measured not in check.missing
    # It is not dropped either: unwaived, each one bounces the declare.
    states = gate.evaluate(check.findings, [])
    warned = {s.check: s.concern for s in states if s.state == gate.WARNED}
    # The evidence and the interpretation, which is what a desk needs to act: `measured`
    # is the number it already has, `means` is why that number ends the run.
    for row, warning in ((AUDIT_FAIL, warned["closure"]),
                         (PROBE_FAIL, warned["location_in_mesh"])):
        assert row["findings"][0]["measured"] in warning
        assert row["findings"][0]["means"] in warning


def test_a_binding_fail_still_decides_the_finish():
    """The other half of the split, so "advisory" cannot quietly grow to mean everything.

    `scale` is the case worth pinning: `cad_audit.scale` reads the union's extent and is
    advisory, while `check.py`'s own `scale` compares the mesh bounds to the largest
    dimension the request names and binds. The first draft of the split made both
    advisory and a test caught it -- that check fires only past a factor of 100, where
    nothing legitimate lives, so what it catches is millimetres-for-metres.
    """
    check = read(facts(bounds=[0, 0, 0, 74, 24, 1]), "mesh", "/work/s/mesh",
                 "a passage 8 mm wide, two legs 60 mm long")
    assert not check.ok
    assert not gate.is_advisory("scale")
    assert gate.is_advisory("cad_audit.scale")
    # And nothing binding is offered to the desk as waivable.
    for name in gate.BINDING:
        assert name not in gate.WAIVABLE, name


def test_an_ok_cad_finding_comes_through_and_does_not_block():
    check = read(composite({"": {"look": facts(), "cellzones": []}},
                           cad=[{"script": "cad_audit", "ok": True,
                                 "findings": [{"check": "coverage", "status": "ok",
                                               "measured": "every triangle in exactly one patch",
                                               "means": "", "repair": ""}]}]), "mesh")
    assert check.ok
    assert finding(check, "cad_audit.coverage").measured == "every triangle in exactly one patch"


def test_an_audit_that_did_not_run_is_skipped_rather_than_failed():
    check = read(composite({"": {"look": facts(), "cellzones": []}},
                           cad=[{"script": "cad_audit", "unavailable": "no patch set"}]), "mesh")
    assert check.ok
    assert finding(check, "cad_audit").status == "skipped"


def test_the_check_computes_no_geometry_of_its_own():
    """The triangles are measured where they live. Nothing under `openreynolds/` outside
    the toolbox opens a CAD kernel, and a second opinion computed here is how two
    answers about one surface start."""
    import openreynolds.cad.check as module

    source = (REPO / "openreynolds" / "cad" / "check.py").read_text()
    tree = source.split("_FINGERPRINT_SNIPPET")[0]  # the snippet is text run on the instance
    for library in ("gmsh", "build123d", "OCP", "trimesh", "numpy", "pyvista"):
        assert f"import {library}" not in tree
        assert not hasattr(module, library)
    # And what it does measure, it measures over the backend.
    calls = []

    class Recording:
        workspace_root = "/work"

        def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
            calls.append(cmd)
            return ExecResult(0, "", False, None)

    assert fingerprint(Recording(), "/work/s/mesh") == {}
    assert calls and "python3" in calls[0]


# -- the size the request asked for --------------------------------------------


@pytest.mark.parametrize("request_text,expected", [
    ("a passage 8 mm wide, two legs 60 mm long, 20 mm apart", 0.06),
    ("a channel 300 mm long and 60 mm high", 0.3),
    ("a box 1 m long and 0.6 m high", 1.0),
    ("a pipe of 20 mm bore with a 4 cm bend radius", 0.04),
    ("something with no sizes in it at all", 0.0),
])
def test_the_largest_length_is_read_off_the_words(request_text, expected):
    assert largest_length(request_text) == pytest.approx(expected)


def test_a_mesh_left_in_millimetres_is_caught():
    """The T04 U-duct: gmsh built it in millimetres and nothing scaled it, so the
    mesh was 74 m across. checkMesh passed, the picture was right, and the case
    would have solved a duct a thousand times too big."""
    off = scale_mismatch("a passage 8 mm wide, two legs 60 mm long",
                         [0, -0.004, 0, 0.074, 0.024, 0.001])
    assert off == ""
    off = scale_mismatch("a passage 8 mm wide, two legs 60 mm long", [0, -4, 0, 74, 24, 1])
    assert "74 m across" in off and "1233x out" in off
    assert "transformPoints -scale" in off


def test_a_flow_box_round_a_small_body_is_not_a_scale_error():
    """A sphere of 20 mm in a box twenty times its size is the normal case, and a
    check that calls it wrong is a check nobody can leave switched on."""
    assert scale_mismatch("a sphere of 20 mm diameter in a flow box",
                          [-0.1, -0.1, -0.1, 0.3, 0.1, 0.1]) == ""


def test_no_sizes_and_no_bounds_mean_no_opinion():
    assert scale_mismatch("mesh me something nice", [0, 0, 0, 74, 24, 1]) == ""
    assert scale_mismatch("a duct 60 mm long", []) == ""
    assert scale_mismatch("a duct 60 mm long", [0, 0, 0, 0, 0, 0]) == ""


def test_the_scale_check_reaches_the_verdict():
    check = read(facts(bounds=[0, 0, 0, 74, 24, 1]), "mesh", "/work/s/mesh",
                 "a passage 8 mm wide, two legs 60 mm long")
    assert not check.ok
    assert any("1233x out" in m for m in check.missing)
    assert "transformPoints -scale" in finding(check, "scale").repair


# -- reaching the workspace ----------------------------------------------------


LOOK_JSON = json.dumps({"polymesh": True, "cells": 5, "faces": 5, "points": 5,
                        "checkmesh_ok": True, "checkmesh": "Mesh OK.",
                        "patches": [{"name": "a", "nFaces": 1}, {"name": "b", "nFaces": 1}],
                        "build": ["Allmesh"], "render": "r.png"})


class Answering:
    """A workspace that answers every command with the same thing, or refuses to."""

    workspace_root = "/work"

    def __init__(self, output, raises=None, raises_once=False, regions=("",),
                 raises_after=None):
        self.output = output
        self.raises = raises
        self.raises_once = raises_once
        self.raises_after = raises_after
        self.regions = regions
        self.calls: list[tuple] = []

    def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
        self.calls.append((cmd, cwd, timeout_s))
        if self.raises:
            raise self.raises
        if self.raises_once and len(self.calls) == 1:
            raise RuntimeError("sandbox recycled")
        if "REGION:" in cmd:
            listing = "".join(f"REGION:{r}\n" if r else "SINGLE:\n" for r in self.regions)
            return ExecResult(0, listing, False, None)
        if self.raises_after and self.raises_after in cmd:
            raise RuntimeError("sandbox gone mid-region")
        return ExecResult(0, self.output, False, None)

    def kernel_run(self, *args, **kwargs):  # pragma: no cover - the point is it is not used
        raise AssertionError("the finish check must not replay through a kernel")

    def put_file(self, path, data):
        self.calls.append(("put_file", path, len(data)))


def test_the_check_runs_the_look_in_the_case_and_reads_its_json(monkeypatch):
    backend = Answering("shell noise\n@@CELLZONES@@\n[]\n@@JSON@@\n" + LOOK_JSON)
    check = verify(backend, "/work/s/mesh", "mesh")
    look = next(call for call in backend.calls if LOOK in call[0])
    assert look[1] == "/work/s/mesh"
    assert "--json" in look[0]
    assert check.ok and check.cells == 5


def test_a_workspace_that_cannot_answer_is_a_failed_check_not_an_exception(monkeypatch):
    monkeypatch.setattr("openreynolds.cad.check.RETRY_PAUSE_S", 0)
    check = verify(Answering("", raises=RuntimeError("sandbox gone")), "/work/s/mesh", "mesh")
    assert not check.ok
    assert check.unreachable
    assert "sandbox gone" in check.missing[0]
    assert "may well be there" in check.missing[0]


def test_unreachable_survives_on_the_multi_region_path(monkeypatch):
    """A workspace that answers the region listing and then goes away is still
    unreachable, not a mesh that failed."""
    monkeypatch.setattr("openreynolds.cad.check.RETRY_PAUSE_S", 0)
    backend = Answering("", regions=("heater", "cooler"), raises_after=LOOK)
    check = verify(backend, "/work/s/mesh", "mesh")
    assert check.unreachable and not check.ok
    assert "sandbox gone mid-region" in check.error


def test_the_check_is_asked_twice_before_it_gives_up(monkeypatch):
    """A container that recycles mid-mesh is a preemption: it comes back in seconds and
    the Volume under it never went anywhere. Three real runs had a finished mesh
    reported as missing because the one attempt landed in that window."""
    monkeypatch.setattr("openreynolds.cad.check.RETRY_PAUSE_S", 0)
    backend = Answering("@@CELLZONES@@\n[]\n@@JSON@@\n" + LOOK_JSON, raises_once=True)
    check = verify(backend, "/work/s/mesh", "mesh")
    assert check.ok and not check.unreachable
    assert len(backend.calls) > 2


def test_a_mesh_that_could_not_be_checked_is_not_reported_as_missing(monkeypatch):
    """The tool result led with "NOT a usable mesh yet" when nothing at all was known,
    and the calling agent believed it and rebuilt a mesh that was already there."""
    from openreynolds.mesher.agent import MeshResult
    from openreynolds.tools import mesh_text

    monkeypatch.setattr("openreynolds.cad.check.RETRY_PAUSE_S", 0)
    check = verify(Answering("", raises=RuntimeError("sandbox recycled")), "/work/s/mesh", "mesh")
    text = mesh_text(MeshResult(case_rel="mesh", check=check, seconds=60.0))
    assert "could NOT BE CHECKED" in text
    assert "not a statement about the mesh" in text.lower()
    assert "NOT a usable mesh" not in text


def test_output_with_no_json_says_what_to_run_by_hand():
    check = verify(Answering("Traceback: pyvista exploded"), "/work/s/mesh", "mesh")
    assert not check.ok
    assert any("no readable answer" in m for m in check.missing)
    assert any("pyvista exploded" in m for m in check.missing)


def test_the_look_command_is_the_one_a_person_would_type():
    assert look_command("/work/s/mesh").startswith(f"python3 {LOOK} /work/s/mesh")


def test_the_regions_are_discovered_and_the_single_case_is_one_region():
    assert mesh_regions(Answering("", regions=("",)), "/work/s/mesh") == [""]
    assert mesh_regions(Answering("", regions=("heater", "cooler")), "/work/s/mesh") == [
        "cooler", "heater"]
    assert mesh_regions(Answering("", regions=()), "/work/s/mesh") == []


def test_a_split_case_that_kept_its_base_mesh_is_still_multi_region():
    backend = Answering("", regions=("heater", "cooler", ""))
    assert mesh_regions(backend, "/work/s/mesh") == ["cooler", "heater"]


# -- what the caller is shown --------------------------------------------------


def test_the_report_lines_carry_the_numbers_a_reader_needs():
    check = read(facts(patches=[{"name": "inlet", "type": "patch", "nFaces": 20,
                                 "area": 0.0001, "center": [0, 0.025, 0.0005],
                                 "normal": [-1, 0, 0]}],
                       metrics={"max_non_orthogonality": 12.4}), "mesh")
    text = "\n".join(check.lines())
    assert "3,750 cells" in text and "2D (one cell thick)" in text
    assert "bounds 0.1 x 0.05 x 0.001 m" in text
    assert "inlet" in text and "area 0.0001 m2" in text and "normal (-1.00 +0.00 +0.00)" in text
    assert "max_non_orthogonality 12.4" in text
    assert "rebuilds with: Allmesh" in text


def test_an_empty_check_reports_nothing_rather_than_zeroes():
    assert Check().lines() == []


def test_a_finding_serialises_with_means_rather_than_meaning():
    item = Finding("closure", "fail", "4 open edges", "not closed", "re-export")
    assert item.as_dict() == {"check": "closure", "status": "fail", "measured": "4 open edges",
                              "means": "not closed", "repair": "re-export"}
    assert Finding.from_dict(item.as_dict()) == item


# -- the script replays, and makes the same thing ------------------------------


BOX_WRITER = '''
def write_box(path, side):
    """An ASCII STL of a box at the origin, written without a CAD kernel."""
    corners = [(0, 0, 0), (side, 0, 0), (side, side, 0), (0, side, 0),
               (0, 0, side), (side, 0, side), (side, side, side), (0, side, side)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
    lines = ["solid box"]
    for a, b, c, d in quads:
        for tri in ((a, b, c), (a, c, d)):
            lines.append(" facet normal 0 0 0")
            lines.append("  outer loop")
            for index in tri:
                lines.append("   vertex %.9f %.9f %.9f" % corners[index])
            lines.append("  endloop")
            lines.append(" endfacet")
    lines.append("endsolid box")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\\n".join(lines))
'''

CLEAN_SCRIPT = f'''# --- cell 1 ---
import pathlib
SIDE = 0.02
{BOX_WRITER}
# --- cell 2 ---
write_box(pathlib.Path("constant/triSurface/walls.stl"), SIDE)
'''

BROKEN_SCRIPT = f'''# --- cell 1 ---
import pathlib
SIDE = 0.02
{BOX_WRITER}
# --- cell 2 ---
write_box(pathlib.Path("constant/triSurface/walls.stl"), SIDE)
# --- cell 3 ---
write_box(pathlib.Path("constant/triSurface/inner.stl"), WALL_THICKNESS)
'''

MUTATE_CELL = '''import pathlib
target = pathlib.Path("constant/triSurface/walls.stl")
# Doubles what is already on disk rather than rebuilding it from SIDE. Run once from
# empty it gives 2 x SIDE; a session that ran this cell twice leaves 4 x SIDE in the
# case, and the log says nothing about having run it twice.
out = []
for line in target.read_text().splitlines():
    bits = line.split()
    if bits[:1] == ["vertex"]:
        out.append("   vertex %.9f %.9f %.9f" % tuple(2 * float(v) for v in bits[1:4]))
    else:
        out.append(line)
target.write_text("\\n".join(out))
'''

MUTATING_SCRIPT = f'''# --- cell 1 ---
import pathlib
SIDE = 0.02
{BOX_WRITER}
# --- cell 2 ---
write_box(pathlib.Path("constant/triSurface/walls.stl"), SIDE)
# --- cell 3 ---
{MUTATE_CELL}'''


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    backend = LocalBackend(root=root)
    case = root / "s" / "mesh"
    case.mkdir(parents=True)
    return backend, str(case)


def replay_finding(backend, case_dir, script, supplied=()):
    check = verify(backend, case_dir, "mesh", script=script, supplied=supplied)
    item = finding(check, "replay")
    assert item is not None, [f.check for f in check.findings]
    return item


def test_a_clean_cell_log_replays_and_matches(workspace):
    backend, case_dir = workspace
    backend.exec(f"mkdir -p {case_dir} && cd {case_dir} && python3 -", cwd=case_dir)
    backend.put_file(f"{case_dir}/build.py", CLEAN_SCRIPT.encode())
    backend.exec("python3 build.py", cwd=case_dir)
    item = replay_finding(backend, case_dir, CLEAN_SCRIPT)
    assert item.status == "ok", item.measured
    assert "same geometry" in item.measured and "walls" not in item.repair


def test_a_cell_that_needs_a_rejected_cells_binding_fails_naming_the_cell(workspace):
    backend, case_dir = workspace
    backend.put_file(f"{case_dir}/build.py", BROKEN_SCRIPT.encode())
    item = replay_finding(backend, case_dir, BROKEN_SCRIPT)
    assert item.status == "fail"
    assert "NameError" in item.measured and "WALL_THICKNESS" in item.measured
    assert "cell 3" in item.measured
    assert item.repair


def test_a_log_that_replays_clean_and_makes_a_different_shape_fails_on_the_numbers(workspace):
    """The one an exit-status check passes. The script runs top to bottom without
    raising; what it produces is not what is on disk, because a cell mutated a file in
    place and the session ran it twice."""
    backend, case_dir = workspace
    backend.put_file(f"{case_dir}/build.py", MUTATING_SCRIPT.encode())
    backend.exec("python3 build.py", cwd=case_dir)
    # The desk ran the mutating cell a second time in the session that was already open.
    # The kernel held the result; the log holds one copy of the cell.
    backend.put_file(f"{case_dir}/again.py", MUTATE_CELL.encode())
    backend.exec("python3 again.py && rm again.py", cwd=case_dir)
    before = fingerprint(backend, case_dir)
    item = replay_finding(backend, case_dir, MUTATING_SCRIPT)
    assert item.status == "fail", item.measured
    assert "different geometry" in item.measured
    assert "union volume" in item.measured or "area" in item.measured
    assert "bounds" in item.measured
    # Not the exit code: the script itself ran clean.
    log = (Path(backend.workspace_root) / "s" / "mesh" / "log.replay").read_text()
    assert "Traceback" not in log
    assert before.get("volume_m3", 0) > 0


def test_the_replay_runs_the_script_as_a_script_not_through_a_kernel(workspace):
    backend, case_dir = workspace
    backend.put_file(f"{case_dir}/build.py", CLEAN_SCRIPT.encode())
    seen: list[str] = []
    real_exec = backend.exec

    def watched(cmd, cwd=None, timeout_s=120, *, background=False):
        seen.append(cmd)
        return real_exec(cmd, cwd=cwd, timeout_s=timeout_s, background=background)

    backend.exec = watched  # type: ignore[method-assign]

    def refuse(*args, **kwargs):  # pragma: no cover - the point is that it is not called
        raise AssertionError("the replay must not go through a kernel")

    backend.kernel_run = refuse  # type: ignore[attr-defined]
    verify(backend, case_dir, "mesh", script=CLEAN_SCRIPT)
    assert any(cmd.strip().startswith("python3 build.py") for cmd in seen), seen


def test_a_run_with_no_script_makes_no_replay_claim(workspace):
    backend, case_dir = workspace
    check = verify(backend, case_dir, "mesh")
    assert finding(check, "replay") is None


# -- a build that starts from a file the requester supplied --------------------

SUPPLIED_SCRIPT = f'''# --- cell 1 ---
import pathlib
SIDE = float(pathlib.Path("drawing.txt").read_text().split("=")[1])
{BOX_WRITER}
# --- cell 2 ---
write_box(pathlib.Path("constant/triSurface/walls.stl"), SIDE)
'''
"""A build whose first cell reads the file the requester sent, by the relative path it
was handed. The shape is a function of that file and of nothing else, which is the whole
point: no later cell can embed it, because embedding it would be the desk inventing the
input rather than reading it."""


def build_from_supplied(backend, case_dir, name="drawing.txt"):
    """Put the supplied file in the case and run the build against it, as a run would."""
    backend.put_file(f"{case_dir}/{name}", b"side=0.02\n")
    backend.put_file(f"{case_dir}/build.py", SUPPLIED_SCRIPT.encode())
    backend.exec("python3 build.py", cwd=case_dir)


def test_the_replay_is_given_back_the_file_the_requester_supplied(workspace):
    backend, case_dir = workspace
    build_from_supplied(backend, case_dir)
    item = replay_finding(backend, case_dir, SUPPLIED_SCRIPT,
                          supplied=[f"{case_dir}/drawing.txt"])
    assert item.status == "ok", item
    assert "drawing.txt" in item.measured, item.measured


def test_without_staging_the_same_build_is_failed_for_the_gate_s_own_reason(workspace):
    """The failure the staging closes, demonstrated in its absence.

    This is the run that meshed a floorplan cleanly and then could not say done: the
    replay's first line raises `FileNotFoundError` on a file the script was never meant
    to write, and the desk is told its build does not re-run from empty.
    """
    backend, case_dir = workspace
    build_from_supplied(backend, case_dir)
    item = replay_finding(backend, case_dir, SUPPLIED_SCRIPT)
    assert item.status == "fail", item
    assert "drawing.txt" in item.measured or "FileNotFoundError" in item.measured, item


def test_every_supplied_file_is_staged_not_only_the_first(workspace):
    """A build can read more than one handed-over file -- a drawing and the table of
    sizes beside it -- and a replay given only the first fails on the second."""
    backend, case_dir = workspace
    backend.put_file(f"{case_dir}/drawing.txt", b"side=0.02\n")
    backend.exec("mkdir -p refs", cwd=case_dir)
    backend.put_file(f"{case_dir}/refs/scale.txt", b"factor=1\n")
    script = SUPPLIED_SCRIPT.replace(
        'write_box(pathlib.Path("constant/triSurface/walls.stl"), SIDE)',
        'FACTOR = float(pathlib.Path("refs/scale.txt").read_text().split("=")[1])\n'
        'write_box(pathlib.Path("constant/triSurface/walls.stl"), SIDE * FACTOR)')
    backend.put_file(f"{case_dir}/build.py", script.encode())
    backend.exec("python3 build.py", cwd=case_dir)

    both = replay_finding(backend, case_dir, script,
                          supplied=[f"{case_dir}/drawing.txt",
                                    f"{case_dir}/refs/scale.txt"])
    assert both.status == "ok", both
    assert "drawing.txt" in both.measured and "refs/scale.txt" in both.measured

    only_one = replay_finding(backend, case_dir, script,
                              supplied=[f"{case_dir}/drawing.txt"])
    assert only_one.status == "fail", only_one
    assert "scale.txt" in only_one.measured


def test_a_supplied_file_under_the_case_keeps_the_path_the_script_names(workspace):
    """`geometry/part.step` is opened as `geometry/part.step`, so it is staged there."""
    backend, case_dir = workspace
    backend.exec("mkdir -p geometry", cwd=case_dir)
    script = SUPPLIED_SCRIPT.replace('"drawing.txt"', '"geometry/drawing.txt"')
    backend.put_file(f"{case_dir}/geometry/drawing.txt", b"side=0.02\n")
    backend.put_file(f"{case_dir}/build.py", script.encode())
    backend.exec("python3 build.py", cwd=case_dir)
    item = replay_finding(backend, case_dir, script,
                          supplied=[f"{case_dir}/geometry/drawing.txt"])
    assert item.status == "ok", item
    assert "geometry/drawing.txt" in item.measured, item.measured


def test_staging_an_input_cannot_pass_a_replay_that_makes_nothing(workspace):
    """Staging is not a way through the gate: the geometry still has to be rebuilt.

    The supplied file is handed back and the script no longer writes the patch set, so
    the fingerprints disagree and the replay fails on the numbers -- exactly as it would
    for a build with no supplied file at all."""
    backend, case_dir = workspace
    build_from_supplied(backend, case_dir)
    idle = '# --- cell 1 ---\nimport pathlib\npathlib.Path("drawing.txt").read_text()\n'
    item = replay_finding(backend, case_dir, idle,
                          supplied=[f"{case_dir}/drawing.txt"])
    assert item.status == "fail", item


# -- real meshes ---------------------------------------------------------------


CONTROL_DICT = """FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
application     scalarTransportFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
"""
FV_SCHEMES = """FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}
ddtSchemes{default steadyState;}
gradSchemes{default Gauss linear;}
divSchemes{default none;}
laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;}
snGradSchemes{default corrected;}
"""
FV_SOLUTION = """FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}
solvers{}
"""
CAVITY_BLOCKMESH = """FoamFile{version 2.0;format ascii;class dictionary;object blockMeshDict;}
scale 0.01;
vertices ((0 0 0)(10 0 0)(10 5 0)(0 5 0)(0 0 1)(10 0 1)(10 5 1)(0 5 1));
blocks (hex (0 1 2 3 4 5 6 7) (20 10 2) simpleGrading (1 1 1));
edges ();
boundary
(
    walls { type wall; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(0 4 7 3)(1 2 6 5)); }
);
mergePatchPairs ();
"""
ZONES_TOPOSET = """FoamFile{version 2.0;format ascii;class dictionary;object topoSetDict;}
actions
(
 { name heaterCells; type cellSet; action new; source boxToCell; box (-1 -1 -1)(0.05 1 1); }
 { name heater; type cellZoneSet; action new; source setToCellZone; set heaterCells; }
 { name coolerCells; type cellSet; action new; source boxToCell; box (0.05 -1 -1)(1 1 1); }
 { name cooler; type cellZoneSet; action new; source setToCellZone; set coolerCells; }
);
"""
SHEAR = math.tan(math.radians(73))
SHEARED_BLOCKMESH = """FoamFile{version 2.0;format ascii;class dictionary;object blockMeshDict;}
scale 0.01;
vertices ((0 0 0)(1 0 0)(1 1 0)(0 1 0)(%f 0 1)(%f 0 1)(%f 1 1)(%f 1 1));
blocks (hex (0 1 2 3 4 5 6 7) (4 4 4) simpleGrading (1 1 1));
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls { type wall; faces ((0 1 5 4)(2 3 7 6)(0 3 2 1)(4 5 6 7)); }
);
mergePatchPairs ();
""" % (SHEAR, SHEAR + 1, SHEAR + 1, SHEAR)


class Mapped:
    """A local workspace that answers to `/work` in command text as well as in paths.

    `LocalBackend` already treats `/work` as an alias for its root when it *resolves a
    path*, and this does the same for the text of a command, which nothing else does.
    It remains here for any command text that still names the hosted root; the finish
    check no longer needs it, because it now asks the backend where its toolbox is
    rather than assuming `/work/.toolbox`.

    The substitution is anchored deliberately. These roots are `tmp_path / "work"`, so
    an unanchored `.replace("/work/", root)` also rewrites the `/work/` *inside* the
    root it just substituted, and the command comes back pointing at a path made of two
    roots glued together. That is a shim bug, not a check bug, and it only became
    reachable once the check started naming real paths.
    """

    _HOSTED = re.compile(r"(?<![\w/])/work/")

    def __init__(self, backend):
        self._backend = backend
        self.workspace_root = backend.workspace_root

    def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
        return self._backend.exec(
            self._HOSTED.sub(f"{self.workspace_root}/", cmd),
            cwd=cwd, timeout_s=timeout_s, background=background)

    def __getattr__(self, name):
        return getattr(self._backend, name)


def openfoam_workspace(root: Path, blockmesh: str, toposet: str = ""):
    """A workspace with the toolbox on it and a case ready for blockMesh."""
    if not find_bashrc():
        pytest.skip("no OpenFOAM installation to source on this machine")
    root.mkdir(parents=True, exist_ok=True)
    backend = LocalBackend(root=root)
    if backend.exec("command -v blockMesh").exit_code != 0:
        pytest.skip("OpenFOAM is installed but blockMesh is not on the sourced path")
    toolbox = root / ".toolbox"
    toolbox.mkdir(exist_ok=True)
    shutil.copy(TOOLBOX / "mesh_look.py", toolbox / "mesh_look.py")
    backend = Mapped(backend)
    case = root / "s" / "mesh"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text(CONTROL_DICT)
    (case / "system" / "fvSchemes").write_text(FV_SCHEMES)
    (case / "system" / "fvSolution").write_text(FV_SOLUTION)
    (case / "system" / "blockMeshDict").write_text(blockmesh)
    if toposet:
        (case / "system" / "topoSetDict").write_text(toposet)
    return backend, case


def build(backend, case, *commands):
    for command in commands:
        outcome = backend.exec(command, cwd=str(case), timeout_s=280)
        assert outcome.exit_code == 0, f"{command}\n{outcome.output[-2000:]}"


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    """A sealed cavity: one patch named `walls`, two cell zones driving it from inside."""
    root = tmp_path_factory.mktemp("cavity") / "work"
    backend, case = openfoam_workspace(root, CAVITY_BLOCKMESH, ZONES_TOPOSET)
    build(backend, case, "blockMesh > log.blockMesh 2>&1", "topoSet > log.topoSet 2>&1")
    return backend, str(case)


@pytest.fixture(scope="module")
def conjugate(tmp_path_factory):
    """Two regions out of `splitMeshRegions`, the base mesh cleared away after."""
    root = tmp_path_factory.mktemp("conjugate") / "work"
    backend, case = openfoam_workspace(root, CAVITY_BLOCKMESH, ZONES_TOPOSET)
    build(backend, case,
          "blockMesh > log.blockMesh 2>&1", "topoSet > log.topoSet 2>&1",
          "splitMeshRegions -cellZones -overwrite > log.split 2>&1",
          "rm -rf constant/polyMesh")
    return backend, str(case)


def test_the_current_check_reports_nothing_meshed_about_a_conjugate_mesh(conjugate):
    """The before half of the deviation: today's check refuses a multi-region mesh
    however correctly it was built."""
    from openreynolds.mesher.check import verify as old_verify

    backend, case_dir = conjugate
    old = old_verify(backend, case_dir, "mesh")
    assert not old.ok
    assert any("nothing has been meshed yet" in m for m in old.missing), old.missing


def test_a_real_multi_region_case_passes(conjugate):
    backend, case_dir = conjugate
    check = verify(backend, case_dir, "mesh")
    assert check.regions == ["cooler", "heater"]
    assert check.ok, check.missing
    assert check.cells == 400
    assert "cooler" in check.checkmesh and "heater" in check.checkmesh
    assert check.render
    for item in check.findings:
        assert item.measured.strip()
        if item.status == "fail":
            assert item.repair.strip()


def test_a_broken_region_fails_and_the_verdict_names_that_region(tmp_path, conjugate):
    backend, case_dir = conjugate
    broken_root = tmp_path / "broken"
    shutil.copytree(Path(backend.workspace_root), broken_root)
    broken = Mapped(LocalBackend(root=broken_root))
    points = broken_root / "s" / "mesh" / "constant" / "cooler" / "polyMesh" / "points"
    moved = [0]

    def shift(match):
        moved[0] += 1
        if moved[0] > 5:
            return match.group(0)
        x, y, z = (float(v) for v in match.group(1).split())
        return f"({x + 0.4} {y + 0.4} {z + 0.4})"

    points.write_text(re.sub(r"\(([-\d.eE+ ]+)\)", shift, points.read_text()))
    check = verify(broken, str(broken_root / "s" / "mesh"), "mesh")
    assert not check.ok
    failed = [m for m in check.missing if "checkMesh does not pass" in m]
    assert failed, check.missing
    assert all("region cooler" in m for m in failed), failed
    assert not any("region heater" in m for m in check.missing)


def test_the_current_check_refuses_the_closed_cavity(cavity):
    """The before half again: one patch, and the old check says a flow case needs
    three."""
    from openreynolds.mesher.check import verify as old_verify

    backend, case_dir = cavity
    old = old_verify(backend, case_dir, "mesh")
    assert not old.ok
    assert any("at least an inlet" in m for m in old.missing), old.missing


def test_the_closed_cavity_passes_with_its_zones_reported(cavity):
    backend, case_dir = cavity
    check = verify(backend, case_dir, "mesh")
    assert check.ok, check.missing
    warn = finding(check, "patch_count")
    assert warn is not None and warn.status == "warn" and "walls" in warn.measured
    zones = finding(check, "cellzones")
    assert "heater 200 cells" in zones.measured and "cooler 200 cells" in zones.measured
    assert check.regions == [""]


def test_checkmesh_is_not_second_guessed_on_a_real_mesh(tmp_path):
    """A sheared mesh whose maximum non-orthogonality sits between the toolbox's warn
    and fail thresholds. checkMesh accepts it, so the finish check must too -- with the
    number and what it means in `measured`."""
    backend, case = openfoam_workspace(tmp_path / "work", SHEARED_BLOCKMESH)
    build(backend, case, "blockMesh > log.blockMesh 2>&1")
    check = verify(backend, str(case), "mesh")
    non_ortho = check.metrics.get("max_non_orthogonality")
    assert non_ortho is not None and NON_ORTHO_WARN < non_ortho < NON_ORTHO_FAIL, check.metrics
    assert check.ok, check.missing
    verdict = finding(check, "checkmesh")
    assert verdict.status == "ok"
    assert f"{non_ortho:g}" in verdict.measured
    assert "do not overrule" in verdict.meaning


# -- mesh_look's payload, with and without --region ----------------------------


PAYLOAD_KEYS = {"case", "polymesh", "patches", "build", "cells", "faces", "points",
                "bounds", "two_d", "checkmesh", "checkmesh_ok", "metrics", "render"}
"""The keys `case_gen.py` and the hosted Mesh panel read. `--region` changes which
polyMesh is read and nothing else, so the two payloads have the same shape."""


def load_mesh_look():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mesh_look_c6", TOOLBOX / "mesh_look.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_look_payload_keys_do_not_change_with_a_region(conjugate):
    mesh_look = load_mesh_look()
    backend, case_dir = conjugate
    case = Path(case_dir)
    # checkMesh is off here: this process has no OpenFOAM environment sourced, and the
    # point of this test is the shape of the payload rather than the verdict in it,
    # which the runs over the backend above already pin.
    heater = mesh_look.look(case, None, check=False, region="heater")
    cooler = mesh_look.look(case, None, check=False, region="cooler")
    assert set(heater) == set(cooler)
    assert PAYLOAD_KEYS <= set(heater) <= PAYLOAD_KEYS | {"failing", "error", "enclosure"}
    assert heater["polymesh"] and heater["cells"] == 200 and cooler["cells"] == 200
    assert [p["name"] for p in heater["patches"]][0] == "walls"
    assert "heater_to_cooler" in [p["name"] for p in heater["patches"]]


def test_the_look_payload_keys_are_the_same_with_and_without_a_region(tmp_path):
    """Pinned without OpenFOAM too, so the contract is checked wherever the suite runs."""
    mesh_look = load_mesh_look()
    case = tmp_path / "case"
    for where in (case / "constant" / "polyMesh", case / "constant" / "heater" / "polyMesh"):
        where.mkdir(parents=True)
        (where / "points").write_text("0\n(\n)\n")
        (where / "boundary").write_text(
            "FoamFile { object boundary; }\n1\n(\n walls { type wall; nFaces 6; startFace 0; }\n)\n")
    plain = mesh_look.look(case, None, check=False)
    region = mesh_look.look(case, None, check=False, region="heater")
    assert set(plain) == set(region)
    assert PAYLOAD_KEYS <= set(plain) <= PAYLOAD_KEYS | {"failing", "error", "enclosure"}
    assert region["polymesh"] and [p["name"] for p in region["patches"]] == ["walls"]
