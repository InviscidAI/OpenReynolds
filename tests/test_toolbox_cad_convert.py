"""STEP and IGES intake: the conversion, the two refusals, and the separation.

This is an integration gate rather than a suffix test. Asserting that `.step` is in a
tuple proves nothing about whether a converted part can be simulated, so the half of
this file that matters converts a real STEP with gmsh and pushes the result through
the path a study actually uses -- `geometry_view.py` for the numbers, `cells_estimate`
for the pre-mesh cost -- and measures what came out.

That half is skipped where gmsh is not importable, which is every dev machine and no
workspace image. Run it where it means something:

    docker run --rm -v $PWD:/repo <workspace image> python3 -m pytest -q \\
        /repo/tests/test_toolbox_cad_convert.py

What it does not cover, deliberately: STEP written by a real CAD system. Assemblies
with nested transforms and per-part units, several solids in one file, degenerate and
self-intersecting faces, and the repair loop those need are `OpenReynolds#6`, not this.
Every STEP here was written by OpenCASCADE itself, which is the friendly case.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"

HAS_GMSH = importlib.util.find_spec("gmsh") is not None
needs_gmsh = pytest.mark.skipif(not HAS_GMSH, reason="gmsh module not on this machine")


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    if str(TOOLBOX) not in sys.path:
        sys.path.insert(0, str(TOOLBOX))
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cad():
    return load("cad_convert")


# -- the units a file declares, read off the text -----------------------------------

STEP_SHELL = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('a fixture'),'2;1');
ENDSEC;
DATA;
{unit}
ENDSEC;
END-ISO-10303-21;
"""

UNIT_LINES = {
    "millimetre": "#436 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );",
    "metre": "#436 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );",
    "centimetre": "#436 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.CENTI.,.METRE.) );",
}


@pytest.mark.parametrize("name,line", sorted(UNIT_LINES.items()))
def test_a_declared_unit_is_read_off_the_file(cad, tmp_path, name, line):
    path = tmp_path / "declared.step"
    path.write_text(STEP_SHELL.format(unit=line))
    found = cad.declared_unit(path)
    assert found["unit"] == name
    assert found["metres"] == pytest.approx({"millimetre": 1e-3, "metre": 1.0, "centimetre": 1e-2}[name])
    assert "SI_UNIT" in found["evidence"]


def test_a_step_with_no_length_unit_reads_as_no_unit(cad, tmp_path):
    path = tmp_path / "silent.step"
    path.write_text(STEP_SHELL.format(unit="#436 = ( LENGTH_UNIT() NAMED_UNIT(*) );"))
    found = cad.declared_unit(path)
    assert found["unit"] is None and found["metres"] is None
    assert "no LENGTH_UNIT" in found["evidence"]


def test_an_inch_step_is_read_through_its_conversion_unit(cad, tmp_path):
    path = tmp_path / "imperial.step"
    path.write_text(STEP_SHELL.format(
        unit="#4362 = ( CONVERSION_BASED_UNIT('INCH',#4361) LENGTH_UNIT() NAMED_UNIT(#4360) );"
    ))
    found = cad.declared_unit(path)
    assert found["unit"] == "inch"
    assert found["metres"] == pytest.approx(0.0254)


# -- the two refusals ---------------------------------------------------------------


def test_no_clmax_refuses_and_names_what_to_pass(cad, tmp_path):
    path = tmp_path / "part.step"
    path.write_text(STEP_SHELL.format(unit=UNIT_LINES["millimetre"]))
    with pytest.raises(cad.Refused) as raised:
        cad.convert(path, tmp_path / "out.stl", clmax=None)
    message = str(raised.value)
    assert "--clmax" in message
    # The relationship stated as a fact, not as an instruction to follow.
    assert "0.5 * dx_surface" in message
    assert "L^2 / (8R)" in message
    assert "MeshSizeFromCurvature" in message


def test_no_unit_declared_and_none_supplied_refuses_and_names_the_override(cad, tmp_path):
    path = tmp_path / "silent.step"
    path.write_text(STEP_SHELL.format(unit="#436 = ( LENGTH_UNIT() NAMED_UNIT(*) );"))
    with pytest.raises(cad.Refused) as raised:
        cad.convert(path, tmp_path / "out.stl", clmax=0.001)
    message = str(raised.value)
    assert "declares no length unit" in message
    assert "--unit" in message
    assert "1000" in message  # what the wrong answer costs, said out loud


def test_the_refusals_come_before_gmsh_is_imported(cad, tmp_path):
    """Both refusals are readable on a machine with no gmsh at all -- which is what
    makes them testable here, and what stops a missing module reading as a bad file."""
    source = (TOOLBOX / "cad_convert.py").read_text()
    top = source.split("def convert(")[0]
    assert "\nimport gmsh" not in top
    assert "    import gmsh" in source


def test_a_bad_unit_name_refuses_rather_than_being_ignored(cad, tmp_path):
    path = tmp_path / "silent.step"
    path.write_text(STEP_SHELL.format(unit="#436 = ( LENGTH_UNIT() NAMED_UNIT(*) );"))
    with pytest.raises(cad.Refused) as raised:
        cad.convert(path, tmp_path / "out.stl", clmax=0.001, unit="furlong")
    assert "not a unit this knows" in str(raised.value)


# -- the five readers no longer treat CAD as an unreadable surface -------------------

SUFFIXES = (".step", ".stp", ".iges", ".igs")


def test_every_reader_knows_the_four_suffixes():
    for name in ("geometry_view", "first_look", "study_run", "preflight", "ladder"):
        module = load(name)
        known = tuple(s.lower() for s in module.CAD_SUFFIXES)
        assert set(SUFFIXES) <= set(known), name


def test_geometry_view_gathers_cad_out_of_a_directory(tmp_path):
    view = load("geometry_view")
    (tmp_path / "a.stl").write_text("solid a\nendsolid a\n")
    (tmp_path / "b.step").write_text("ISO-10303-21;\n")
    (tmp_path / "c.igs").write_text("S      1\n")
    (tmp_path / "notes.md").write_text("x")
    found = {p.name for p in view.gather([tmp_path])}
    assert found == {"a.stl", "b.step", "c.igs"}


def test_preflight_reports_cad_as_geometry_awaiting_tessellation(tmp_path):
    preflight = load("preflight")
    case = tmp_path / "case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "constant" / "triSurface" / "part.step").write_text(
        STEP_SHELL.format(unit=UNIT_LINES["millimetre"])
    )
    obj = preflight.Case(case)
    assert obj.surfaces == []            # nothing a mesher can read
    assert [p.name for p in obj.cad] == ["part.step"]

    findings = preflight.check_geometry(obj, preflight.Intent())
    assert len(findings) == 1
    finding = findings[0]
    assert finding.status == "fail"
    # Not "no geometry": the geometry is here, in a format the mesher does not read.
    assert "part.step" in finding.measured and "millimetre" in finding.measured
    assert "cad_convert.py" in finding.repair and "--clmax" in finding.repair


def test_study_run_counts_cad_as_geometry_that_exists(tmp_path):
    study_run = load("study_run")
    case = tmp_path / "case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "constant" / "triSurface" / "wing.iges").write_text("S      1\n")
    assert study_run.surfaces(case) == []
    assert [p.name for p in study_run.cad_files(case)] == ["wing.iges"]
    ctx = type("Ctx", (), {"case": case, "root": tmp_path})()
    present, evidence = study_run.geometry_evidence(ctx)
    assert present is True
    assert "wing.iges" in evidence and "tessellated it is not" in evidence


def test_ladder_sees_a_body_in_a_step_file(tmp_path):
    ladder = load("ladder")
    preflight = load("preflight")
    case = tmp_path / "case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "constant" / "triSurface" / "body.stp").write_text("ISO-10303-21;\n")
    assert ladder.has_body_surface(preflight.Case(case)) is True


def test_first_look_lists_cad_next_to_the_counts(tmp_path):
    first_look = load("first_look")
    case = tmp_path / "case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "constant" / "triSurface" / "part.step").write_text(
        STEP_SHELL.format(unit=UNIT_LINES["millimetre"])
    )
    stats = first_look.disk_stats(case)
    assert any("part.step is CAD (millimetre), not tessellated" in n for n in stats["notes"])


# -- the regression that a passing suite would otherwise hide -----------------------
#
# A render tessellation exists to be drawn and nothing else. The failure this guards
# against is somebody unifying it with the simulation tessellation so there is "one
# conversion", at which point the facet size a solver sees is chosen by whatever
# looked right in a picture -- which is precisely the accident #19 was filed about.

SOLVER_FED = (
    "preflight.py", "cells_estimate.py", "case_gen.py", "mesh_look.py",
    "study_run.py", "ladder.py", "layer_report.py", "mesh_digest.py",
)


def test_nothing_that_feeds_a_solver_mentions_the_render_tessellation():
    for name in SOLVER_FED:
        source = (TOOLBOX / name).read_text()
        assert "render_tessellation" not in source, name


def test_only_the_two_drawing_scripts_call_it():
    callers = sorted(
        path.name for path in TOOLBOX.glob("*.py")
        if "render_tessellation(" in path.read_text() and path.name != "cad_convert.py"
    )
    assert callers == ["first_look.py", "geometry_view.py"]


def test_both_callers_hand_it_a_temporary_directory():
    """Not a location a case could reach. An STL under `constant/triSurface` is an STL
    something will mesh, and this one must never be meshed."""
    for name in ("geometry_view.py", "first_look.py"):
        source = (TOOLBOX / name).read_text()
        assert "tempfile.mkdtemp" in source, name
        assert "render_tessellation(path, " in source, name


def test_the_render_size_is_not_derived_from_any_mesh_size(cad):
    """`clmax` never reaches the render path: its size comes from the bounding box."""
    source = (TOOLBOX / "cad_convert.py").read_text()
    body = source.split("def render_tessellation(")[1].split("\ndef ")[0]
    code = body.split('"""')[2]  # past the docstring, which talks about clmax on purpose
    assert "clmax" not in code
    assert "RENDER_CHORDS_PER_DIAGONAL" in body
    assert cad.RENDER_CHORDS_PER_DIAGONAL >= 20


# -- for real, with gmsh ------------------------------------------------------------


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    """A 100 x 40 x 20 mm block with a r=12 mm cylindrical boss, in four files:

    `mm.step` declaring millimetres, `nounit.step` declaring nothing at all (the same
    bytes with the length unit removed), `mm.iges`, and the same in metres. Written by
    OpenCASCADE, which is why this fixture is the friendly case and #6 is not closed.
    """
    if not HAS_GMSH:
        pytest.skip("gmsh module not on this machine")
    import gmsh

    directory = tmp_path_factory.mktemp("cad")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        box = gmsh.model.occ.addBox(0, 0, 0, 100, 40, 20)
        boss = gmsh.model.occ.addCylinder(50, 20, 20, 0, 0, 25, 12)
        gmsh.model.occ.fuse([(3, box)], [(3, boss)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(directory / "mm.step"))
        gmsh.write(str(directory / "mm.iges"))
    finally:
        gmsh.finalize()

    text = (directory / "mm.step").read_text()
    declared = "#436 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );"
    assert declared in text
    (directory / "nounit.step").write_text(
        text.replace(declared, "#436 = ( LENGTH_UNIT() NAMED_UNIT(*) );")
    )
    return directory


@needs_gmsh
def test_a_declared_unit_converts_without_ceremony(cad, parts, tmp_path):
    out = tmp_path / "part.stl"
    report = cad.convert(parts / "mm.step", out, clmax=0.002)

    assert out.exists() and out.stat().st_size > 0
    assert report["entities"]["volumes"] == 1
    assert report["entities"]["surfaces"] == 8
    # 100 mm read as millimetres and converted to metres.
    assert report["extent_m"] == pytest.approx([0.1, 0.04, 0.045], rel=1e-6)
    assert "millimetre" in report["unit_interpretation"]
    assert "OCCTargetUnit" in report["unit_interpretation"]
    # The B-rep's own circle, not a polygon's: r = 12 mm.
    assert report["tightest_curve"]["radius"] == pytest.approx(0.012, rel=1e-3)
    assert report["sagitta_m"] == pytest.approx(0.002 ** 2 / (8 * 0.012), rel=1e-3)
    assert report["facets"] > 1000
    # Every number the log promises is in it.
    text = cad.render(report)
    for expected in ("clmax", "sagitta", "facets", "entities", "gmsh calls"):
        assert expected in text
    assert "gmsh.model.mesh.generate(2)" in text


@needs_gmsh
def test_iges_converts_too(cad, parts, tmp_path):
    report = cad.convert(parts / "mm.iges", tmp_path / "part.stl", clmax=0.003)
    assert report["format"] == "IGES"
    assert report["entities"]["surfaces"] == 8
    assert report["facets"] > 100


@needs_gmsh
def test_the_conversion_feeds_the_existing_path_unchanged(cad, parts, tmp_path):
    """End to end: STEP -> chosen clmax -> the STL the ordinary tools already read.

    `geometry_view` for the four views and the numbers, `cells_estimate` for the
    pre-mesh cost. Neither knows a STEP file was involved, which is the claim.
    """
    pytest.importorskip("pyvista")
    out = tmp_path / "case" / "constant" / "triSurface" / "part.stl"
    report = cad.convert(parts / "mm.step", out, clmax=0.002)

    view = load("geometry_view")
    facts = view.measure(view.load(out))
    assert facts["faces"] == report["facets"]
    assert facts["open_edges"] == 0            # a closed solid stayed closed
    # The numbers the ordinary tool reports are the numbers the conversion reported:
    # same file, no second opinion about scale anywhere between them.
    for axis in range(3):
        assert facts["extent"][axis] == pytest.approx(report["extent_m"][axis], rel=1e-6)
    assert facts["extent"] == pytest.approx([0.1, 0.04, 0.045], rel=1e-6)

    cells = load("cells_estimate")
    area, extent, count = cells.stl_area(out)
    assert count == report["facets"]
    assert extent[0] == pytest.approx(0.1, rel=1e-3)
    # Surface area of a 100x40x20 mm block plus a r=12 boss, in metres squared.
    assert 0.014 < area < 0.020

    drawn = view.draw([("part.stl", view.load(out))], tmp_path / "renders" / "g.png", "part")
    assert drawn.exists() and drawn.stat().st_size > 0


@needs_gmsh
def test_the_scale_trap_shows_up_where_it_would_hurt(cad, parts, tmp_path):
    """The same file, two unit interpretations, and the difference must not be absorbed.

    This is the assertion worth the most in the file. A unitless STEP read as metres
    instead of millimetres is a 1000x error on every length, and it produces a
    perfectly plausible mesh: the bounding box looks fine on its own, the solve
    converges, and only the forces are wrong. So the check is not that the converter
    noticed -- it refuses, and that is asserted below -- but that when a unit *is*
    supplied the choice propagates all the way to the numbers a decision is made from.
    """
    cells = load("cells_estimate")

    as_mm = cad.convert(parts / "nounit.step", tmp_path / "mm.stl", clmax=0.002, unit="mm")
    as_m = cad.convert(parts / "nounit.step", tmp_path / "m.stl", clmax=2.0, unit="m")

    # 1. the reported bounding box
    assert as_mm["extent_m"] == pytest.approx([0.1, 0.04, 0.045], rel=1e-6)
    assert as_m["extent_m"] == pytest.approx([100.0, 40.0, 45.0], rel=1e-6)
    assert as_m["extent_m"][0] / as_mm["extent_m"][0] == pytest.approx(1000.0, rel=1e-2)
    assert "occ.dilate" in as_mm["unit_interpretation"]

    # 2. the pre-mesh cost, read by the tool that reads it for a real case
    area_mm, _extent, _n = cells.stl_area(tmp_path / "mm.stl")
    area_m, _extent, _n = cells.stl_area(tmp_path / "m.stl")
    assert area_m / area_mm == pytest.approx(1e6, rel=1e-2)

    # Same domain, same refinement, one surface a thousand times the other. The
    # surface term moves by 1e6; the total moves by ~1.2e3, because the background
    # mesh puts a floor under the millimetre case that absorbs some of it. That
    # smaller number is the honest one and it is still the difference between a mesh
    # you can build and one nobody can: 1.0e6 cells against 1.2e9.
    delta0, volume = 0.01, 1.0
    levels = {"body": 3}
    predicted_mm = cells.estimate(delta0, volume, levels, area_mm, 0)
    predicted_m = cells.estimate(delta0, volume, levels, area_m, 0)
    assert predicted_mm["total"] < 2e6
    assert predicted_m["total"] > 1e9
    assert predicted_m["total"] / predicted_mm["total"] > 1e3

    # 3. and with no unit at all, it does not happen: it refuses.
    with pytest.raises(cad.Refused) as raised:
        cad.convert(parts / "nounit.step", tmp_path / "no.stl", clmax=0.002)
    assert "--unit" in str(raised.value)
    assert not (tmp_path / "no.stl").exists()


@needs_gmsh
def test_the_files_own_declaration_wins_over_a_supplied_unit(cad, parts, tmp_path):
    """A flag cannot silently rescale a file that already says what it is."""
    report = cad.convert(parts / "mm.step", tmp_path / "p.stl", clmax=0.002, unit="m")
    assert report["extent_m"][0] == pytest.approx(0.1, rel=1e-3)
    assert "is ignored" in report["unit_interpretation"]


@needs_gmsh
def test_tessellation_follows_the_mesh_measured(cad, parts, tmp_path):
    """Facet count and sagitta against clmax, as numbers rather than as a claim.

    Two predictions: facets scale roughly as clmax^-2, and the implied sagitta as
    clmax^2. Both are checked as trends over a decade of clmax, not as point values --
    curvature refinement bends the first one and is supposed to.
    """
    measured = []
    for clmax in (0.008, 0.004, 0.002, 0.001):
        report = cad.convert(parts / "mm.step", tmp_path / f"c{clmax}.stl", clmax=clmax)
        measured.append((clmax, report["facets"], report["sagitta_m"],
                         report["edge_m"]["longest"]))

    for (c1, f1, s1, _e1), (c2, f2, s2, _e2) in zip(measured, measured[1:]):
        assert c2 == pytest.approx(c1 / 2)
        assert f2 > f1                                   # halving clmax adds facets
        assert 2.0 < f2 / f1 < 6.0                       # ~4x, loosened for curvature
        assert s2 / s1 == pytest.approx(0.25, rel=1e-6)  # sagitta is exactly clmax^2/8R

    # A tessellation coarser than the surface cell size against one that follows it.
    # dx_surface = 0.5 mm at the wall, so the rule of thumb asks for clmax = 0.25 mm;
    # 8 mm is what "gmsh's default looked fine" gets you.
    dx_surface = 0.0005
    coarse = cad.convert(parts / "mm.step", tmp_path / "coarse.stl", clmax=0.008)
    fine = cad.convert(parts / "mm.step", tmp_path / "fine.stl", clmax=dx_surface / 2)

    # Coarse: the flat spot on the r=12 mm boss is deeper than a cell, so the mesh
    # resolves the faceting and reproduces it -- the polygon becomes the geometry.
    assert coarse["sagitta_m"] > dx_surface
    assert coarse["edge_m"]["longest"] > 4 * dx_surface
    # Fine: the same departure is more than a decade below the cell size, which is
    # what "subgrid" means and is the whole content of the 0.5 * dx_surface rule.
    assert fine["sagitta_m"] < dx_surface / 10
    assert fine["edge_m"]["longest"] < 2 * dx_surface
    # And it is not free: the fine one costs an order of magnitude more triangles.
    assert fine["facets"] > 10 * coarse["facets"]


@needs_gmsh
def test_the_render_tessellation_is_coarse_temporary_and_never_a_surface(cad, parts, tmp_path):
    scratch = tmp_path / "scratch"
    drawn = cad.render_tessellation(parts / "mm.step", scratch)
    meshed = cad.convert(parts / "mm.step", tmp_path / "part.stl", clmax=0.001)

    cells = load("cells_estimate")
    _area, _extent, drawn_facets = cells.stl_area(drawn)
    assert drawn_facets < meshed["facets"] / 5       # coarse, deliberately
    assert drawn.parent == scratch                   # and nowhere a case would look
    assert drawn.name.endswith(".render.stl")
    assert not list((parts).glob("*.stl"))           # nothing beside the CAD file


@needs_gmsh
def test_the_cli_refuses_loudly_and_converts_quietly(parts, tmp_path):
    """The refusals as a user meets them: on stderr, with a non-zero exit code."""
    script = str(TOOLBOX / "cad_convert.py")

    missing = subprocess.run(
        [sys.executable, script, str(parts / "mm.step")],
        capture_output=True, text=True,
    )
    assert missing.returncode == 2
    assert "--clmax" in missing.stderr and missing.stdout == ""

    unitless = subprocess.run(
        [sys.executable, script, str(parts / "nounit.step"), "--clmax", "0.002"],
        capture_output=True, text=True,
    )
    assert unitless.returncode == 2
    assert "--unit" in unitless.stderr

    good = subprocess.run(
        [sys.executable, script, str(parts / "mm.step"), "--clmax", "0.002",
         "--out", str(tmp_path / "part.stl"), "--json"],
        capture_output=True, text=True,
    )
    assert good.returncode == 0, good.stderr
    assert "implied sagitta" in good.stdout
    assert "gmsh.model.mesh.generate(2)" in good.stdout
    assert (tmp_path / "part.stl").exists()


@needs_gmsh
def test_geometry_view_draws_a_step_file_and_says_what_the_counts_are(parts, tmp_path):
    """The wall the issue was filed about: a `.step` handed to the viewer used to be
    "not a surface file". It draws now, and labels the numbers as the picture's."""
    pytest.importorskip("pyvista")
    view = load("geometry_view")
    mesh = view.load(parts / "mm.step")
    assert mesh.n_cells > 100
    lines = view.report("mm.step", view.measure(mesh), cad=True)
    text = "\n".join(lines)
    assert "render tessellation" in text and "cad_convert.py --clmax" in text

    drawn = view.draw([("mm.step", mesh)], tmp_path / "g.png", "mm.step")
    assert drawn.exists() and drawn.stat().st_size > 0
    # And nothing was left beside the CAD file for a mesher to find.
    assert not list(parts.glob("*.stl"))


@needs_gmsh
def test_first_look_draws_a_case_that_holds_only_cad(parts, tmp_path):
    pytest.importorskip("pyvista")
    first_look = load("first_look")
    case = tmp_path / "case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    target = case / "constant" / "triSurface" / "part.step"
    target.write_bytes((parts / "mm.step").read_bytes())

    out = first_look.render_geometry(case, tmp_path / "geometry.png")
    assert out.exists() and out.stat().st_size > 0
    # The panel drew it; the case still has no surface anything would mesh.
    assert list(case.rglob("*.stl")) == []
    stats = first_look.disk_stats(case)
    assert any("part.step is CAD" in note for note in stats["notes"])
