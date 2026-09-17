"""`cad_export.export_patches`: the addition, and the failures it closes.

The rule for an addition is that it arrives with a test demonstrating the failure **in
its absence**, so most of what is here runs the thing the desk does today and asserts
that it breaks. Four failures, each with an id in some sweep's `findings.jsonl`:

* `exported_surface_winding_inconsistent` and `exported_surface_has_open_edges` --
  tessellating each patch separately tears the seam between them. `test_the_desk_s_own_
  pattern_tears_the_seams` parametrises the three ways it does so;
* `no_closure_assertion_between_export_and_meshing` -- nothing in the desk's path ever
  said what the union was, so a leak reached snappyHexMesh unremarked;
* `named_patches_land_empty_under_a_passing_checkmesh` -- a patch nobody filled;
* `exported_surface_duplicated_in_trisurface` -- a file from an abandoned route left in
  the directory, which every downstream reader welds in as though it were the surface.

**The weld here is exact, and that is deliberate.** `preflight.surface_topology` welds
within a tolerance, which is right for a surface of unknown provenance and wrong for
this question: it reports zero on a seam whose two sides differ by a micron, and a
micron is exactly what a second tessellation moves them by. The counts below are over
bit-identical vertices, so they say whether the two sides are *the same points* rather
than whether they are close enough to pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"

HAS_B123D = importlib.util.find_spec("build123d") is not None
needs_cad = pytest.mark.skipif(not HAS_B123D, reason="build123d is not on this machine")


@pytest.fixture(scope="module")
def cad_export():
    if str(TOOLBOX) not in sys.path:
        sys.path.insert(0, str(TOOLBOX))
    spec = importlib.util.spec_from_file_location("cad_export", TOOLBOX / "cad_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def finned():
    """A barrel with six fins, fused. Twenty-seven faces and a seam at every fin root.

    T18's geometry in miniature, which is where this failure was worst on record: a
    finned cylinder is nothing but patch boundaries, so a tessellation that does not
    weld has nowhere to hide.
    """
    import build123d as bd

    align = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
    shape = bd.Cylinder(0.03, 0.12, align=align)
    for i in range(6):
        shape = shape + bd.Pos(0, 0, 0.01 + i * 0.015) * bd.Cylinder(0.05, 0.003, align=align)
    return shape


def groups(shape):
    """Barrel faces and fin faces, by radius -- the split any desk would write."""
    barrel, fins = [], []
    for face in shape.faces():
        box = face.bounding_box()
        radius = max(abs(box.min.X), abs(box.max.X), abs(box.min.Y), abs(box.max.Y))
        (fins if radius > 0.03 + 1e-6 else barrel).append(face)
    return barrel, fins


def read_stl(path: Path) -> list[tuple]:
    """Triangles out of an ASCII or binary STL, as exact vertex tuples."""
    import struct

    raw = path.read_bytes()
    if raw[:5] != b"solid" or b"facet" not in raw[:2048]:
        count = struct.unpack("<I", raw[80:84])[0]
        out = []
        for i in range(count):
            chunk = raw[84 + i * 50:84 + i * 50 + 50]
            values = struct.unpack("<12fH", chunk)[:12]
            out.append(tuple(tuple(values[3 + 3 * k:6 + 3 * k]) for k in range(3)))
        return out
    out, current = [], []
    for line in raw.decode().splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            current.append(tuple(float(v) for v in line.split()[1:4]))
            if len(current) == 3:
                out.append(tuple(current))
                current = []
    return out


def seams(paths) -> dict:
    """Open and same-direction edges of the union, welded at exact coordinates."""
    vertices: dict[tuple, int] = {}
    undirected: Counter = Counter()
    directed: Counter = Counter()
    for path in paths:
        for triangle in read_stl(path):
            keys = [vertices.setdefault(v, len(vertices)) for v in triangle]
            for a, b in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
                undirected[(min(a, b), max(a, b))] += 1
                directed[(a, b)] += 1
    return {
        "open": sum(1 for n in undirected.values() if n == 1),
        "flipped": sum(1 for n in directed.values() if n > 1),
    }


# -- the failure, in the tool's absence ---------------------------------------------


@needs_cad
@pytest.mark.parametrize("how", ["per-face", "two-tolerances", "loose-face"])
def test_the_desk_s_own_pattern_tears_the_seams(finned, tmp_path, how):
    """`export_stl` per patch, the three ways it goes wrong, each on this geometry.

    The fourth way -- one call per group at one tolerance -- is left out on purpose and
    has its own test below, because it *passes*, and why it passes is the whole argument
    for the addition.
    """
    import build123d as bd

    barrel, fins = groups(finned)
    if how == "per-face":
        for i, face in enumerate(finned.faces()):
            bd.export_stl(bd.Compound([face]), str(tmp_path / f"{i:03d}.stl"),
                          tolerance=2.5e-4, angular_tolerance=0.08, ascii_format=True)
    elif how == "two-tolerances":
        bd.export_stl(bd.Compound(barrel), str(tmp_path / "barrel.stl"),
                      tolerance=5e-4, angular_tolerance=0.10, ascii_format=True)
        bd.export_stl(bd.Compound(fins), str(tmp_path / "fins.stl"),
                      tolerance=1e-4, angular_tolerance=0.05, ascii_format=True)
    else:
        # A patch drawn where the boundary is, instead of selected off the solid. This
        # is T18, which shipped 8,364 free edges: `ductWalls`, `inlet` and `outlet` were
        # `Face(Wire.make_polygon(...))` beside an engine built from primitives, so the
        # duct never sealed against anything. Here the duct is a box round the barrel
        # and its inlet is drawn rather than selected, which is the same mistake with
        # one face instead of six.
        # The duct floor cuts through the barrel, so the fluid's own inlet face is a
        # square with a circular hole in it. Drawn as a plain rectangle -- which is what
        # `Wire.make_polygon` gives you -- the circular rim has nothing to weld to.
        drawn_at = 0.02
        duct = bd.Pos(0, 0, drawn_at) * bd.Box(
            0.2, 0.2, 0.2, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        fluid = duct - finned
        walls = [f for f in fluid.faces() if abs(f.center().Z - drawn_at) > 1e-9]
        bd.export_stl(bd.Compound(walls), str(tmp_path / "walls.stl"),
                      tolerance=2.5e-4, angular_tolerance=0.08, ascii_format=True)
        inlet = bd.Face(bd.Wire.make_polygon(
            [(-0.1, -0.1, drawn_at), (-0.1, 0.1, drawn_at),
             (0.1, 0.1, drawn_at), (0.1, -0.1, drawn_at)], close=True))
        bd.export_stl(inlet, str(tmp_path / "inlet.stl"),
                      tolerance=2.5e-4, angular_tolerance=0.08, ascii_format=True)

    torn = seams(sorted(tmp_path.glob("*.stl")))
    assert torn["open"] + torn["flipped"] > 100, (
        f"{how} was expected to tear the seams and did not ({torn}). If OpenCASCADE has "
        "changed such that separate tessellations now agree, the addition's premise has "
        "to be re-measured rather than the assertion relaxed.")


@needs_cad
def test_the_way_that_works_works_only_by_cache(finned, tmp_path):
    """One `export_stl` per group at one tolerance welds -- and that is not correctness.

    OpenCASCADE caches a triangulation on the shape, so the second call reuses the edge
    polygons the first wrote. The test asserts both halves: it welds, and it stops
    welding the moment the cache is not shared, which is what makes it luck rather than
    a property anyone can rely on.
    """
    import build123d as bd

    barrel, fins = groups(finned)
    for name, faces in (("barrel", barrel), ("fins", fins)):
        bd.export_stl(bd.Compound(faces), str(tmp_path / f"{name}.stl"),
                      tolerance=2.5e-4, angular_tolerance=0.08, ascii_format=True)
    assert seams(sorted(tmp_path.glob("*.stl")))["open"] == 0

    # The same code, with the fins taken from a shape that has its own triangulation.
    # Nothing about the desk's cell changed; the seam tore anyway.
    other = tmp_path / "other"
    other.mkdir()
    copy = finned.moved(bd.Location((0, 0, 0)))
    barrel2, fins2 = groups(copy)
    bd.export_stl(bd.Compound(barrel), str(other / "barrel.stl"),
                  tolerance=2.5e-4, angular_tolerance=0.08, ascii_format=True)
    bd.export_stl(bd.Compound(fins2), str(other / "fins.stl"),
                  tolerance=1e-4, angular_tolerance=0.05, ascii_format=True)
    assert seams(sorted(other.glob("*.stl")))["open"] > 100


# -- and with it --------------------------------------------------------------------


@needs_cad
def test_export_patches_welds_the_same_geometry(cad_export, finned, tmp_path):
    """One tessellation, so the seam is the same points and not merely nearby ones."""
    barrel, fins = groups(finned)
    report = cad_export.export_patches(
        finned, {"barrel": barrel, "fins": fins}, tmp_path,
        tolerance=2.5e-4, angular_tolerance=0.08, quiet=True)
    assert seams(sorted(tmp_path.glob("*.stl"))) == {"open": 0, "flipped": 0}
    assert report["union"]["open_edges"] == 0
    assert report["triangles"] == sum(p["triangles"] for p in report["patches"])


@needs_cad
def test_the_report_says_what_the_union_is(cad_export, finned, tmp_path):
    """`no_closure_assertion_between_export_and_meshing`: now it is asserted, and said.

    Reported rather than enforced. A zero-thickness baffle is open on purpose and a
    refusal could not tell it from a leak -- so the number reaches the transcript, where
    the desk can explain it or fix it, and the `waive` channel already exists for the
    first.
    """
    barrel, fins = groups(finned)
    report = cad_export.export_patches(
        finned, {"barrel": barrel, "fins": fins}, tmp_path,
        tolerance=2.5e-4, quiet=True)
    text = cad_export.render(report)
    assert "open edges" in text and "flipped" in text
    assert "extent" in text and "m^2" in text

    # An open shell exports, and says so, rather than being refused. One face is the
    # smallest such shell and is a real case -- a zero-thickness baffle is exactly this.
    import build123d as bd
    baffle = bd.Face(bd.Wire.make_polygon(
        [(0, 0, 0), (0.1, 0, 0), (0.1, 0.1, 0), (0, 0.1, 0)], close=True))
    open_report = cad_export.export_patches(
        baffle, {"baffle": ...}, tmp_path / "shell", tolerance=1e-3, quiet=True)
    assert open_report["union"]["open_edges"] > 0
    assert "reported, not enforced" in cad_export.render(open_report)


@needs_cad
@pytest.mark.parametrize("bad,expect", [
    ("foreign", "not a face of the shape"),
    ("doubled", "in both"),
    ("unclaimed", "are in no patch"),
    ("empty", "was given no faces"),
    ("no-tolerance", "no `tolerance` given"),
])
def test_the_refusals(cad_export, finned, tmp_path, bad, expect):
    """Each refusal is a failure id that shipped. None of them guesses.

    `unclaimed` is `named_patches_land_empty_under_a_passing_checkmesh` at its source: a
    face nobody names is not dropped, it lands in the mesher's default patch and takes
    that patch's boundary condition, and the run finishes looking fine.
    """
    import build123d as bd

    barrel, fins = groups(finned)
    kwargs = {"tolerance": 2.5e-4, "quiet": True}
    if bad == "foreign":
        loose = bd.Face(bd.Wire.make_polygon(
            [(0, 0, 0), (0.01, 0, 0), (0.01, 0.01, 0)], close=True))
        patches = {"loose": [loose], "rest": ...}
    elif bad == "doubled":
        patches = {"a": barrel, "b": barrel[:1] + fins}
    elif bad == "unclaimed":
        patches = {"barrel": barrel}
    elif bad == "empty":
        patches = {"barrel": barrel, "nothing": [], "rest": ...}
    else:
        patches = {"all": ...}
        kwargs.pop("tolerance")
    with pytest.raises(cad_export.Refused, match=expect):
        cad_export.export_patches(finned, patches, tmp_path, **kwargs)


@needs_cad
def test_a_file_from_an_abandoned_route_is_cleared(cad_export, finned, tmp_path):
    """`exported_surface_duplicated_in_trisurface`, on T4, T22 and T25.

    Three of one sweep's five non-zero surface readings were a dead file rather than a
    defect in a delivered surface: the probes weld every STL in the directory, so a
    whole-surface export left beside the per-patch files is a doubled surface in which
    every edge walks twice the same way.
    """
    barrel, fins = groups(finned)
    stale = tmp_path / "fluid.stl"
    stale.write_text("solid stale\nendsolid stale\n")
    report = cad_export.export_patches(
        finned, {"barrel": barrel, "fins": fins}, tmp_path, tolerance=2.5e-4, quiet=True)
    assert not stale.exists()
    assert report["removed"] == ["fluid.stl"]
    assert "cleared" in cad_export.render(report)

    kept = tmp_path / "keep.stl"
    kept.write_text("solid keep\nendsolid keep\n")
    cad_export.export_patches(finned, {"barrel": barrel, "fins": fins}, tmp_path,
                              tolerance=2.5e-4, clean=False, quiet=True)
    assert kept.exists(), "clean=False is the escape and has to actually escape"


@needs_cad
def test_the_manifest_is_the_one_the_coverage_probe_reads(cad_export, finned, tmp_path):
    """`coverage` returned `n/a` on every case of every sweep for want of this file."""
    barrel, fins = groups(finned)
    cad_export.export_patches(finned, {"barrel": barrel, "fins": fins}, tmp_path,
                              tolerance=2.5e-4, location_in_mesh=(0, 0, 0.001), quiet=True)
    manifest = json.loads((tmp_path / "patches.json").read_text())
    assert manifest["unit_metres"] == 1.0
    assert manifest["location_in_mesh"] == [0.0, 0.0, 0.001]
    assert [p["name"] for p in manifest["patches"]] == ["barrel", "fins"]
    for patch in manifest["patches"]:
        assert (tmp_path / patch["file"]).is_file()
        assert patch["triangles"] > 0


@needs_cad
def test_binary_and_ascii_carry_the_same_triangles(cad_export, finned, tmp_path):
    """Binary by default: T23 welded 544,306 triangles, which is 270 MB of ASCII."""
    barrel, fins = groups(finned)
    both = {}
    for binary in (True, False):
        out = tmp_path / ("binary" if binary else "ascii")
        cad_export.export_patches(finned, {"barrel": barrel, "fins": fins}, out,
                                  tolerance=2.5e-4, binary=binary, quiet=True)
        both[binary] = seams(sorted(out.glob("*.stl")))
    assert both[True] == both[False] == {"open": 0, "flipped": 0}


# -- that the desk can actually reach it ---------------------------------------------


def test_the_core_desk_is_handed_it_and_the_brief_names_the_call():
    """Handed, named, and named with the two lines that make the first call work.

    The failure this guards is on record: the first attempt at the reference-file
    addition copied the file to the workspace root while the desk's working directory is
    the case directory, so the brief named a path no desk could reach and T16 ran the
    grep the brief suggests and got an empty string back.
    """
    from openreynolds.buildup import core

    assert "cad_export.py" in core.REFERENCE_FILES
    assert (TOOLBOX / "cad_export.py").is_file()

    brief = core.brief()
    assert f'sys.path.insert(0, "{core.REFERENCE_DIR}")' in brief
    assert "from cad_export import export_patches" in brief
    assert "help(export_patches)" in brief
    assert "Do not export the patches one at a time with `export_stl`" in brief


def test_reaching_for_it_is_not_contamination():
    """It is `given`, like `b123d_api.md`, so using it does not void the run.

    `isolation.house_names` reads every toolbox file off disk, so without this the first
    desk to import the module would grade contaminated and void the sweep -- which is
    the mechanism that made the reference-file addition's own sweep worth checking
    before anything else.
    """
    from openreynolds.buildup import core, isolation

    used = {"cells": "import sys; sys.path.insert(0, '.reference')\n"
                     "from cad_export import export_patches"}
    assert not isolation.scan(used, given=core.REFERENCE_FILES).contaminated
    assert isolation.scan({"cells": "grep -n patches cad_convert.py"},
                          given=core.REFERENCE_FILES).contaminated
