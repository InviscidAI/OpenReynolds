"""Per-patch export: the tessellation, the recipe beside it, and the three traps C1 found.

`export_patches()` is the one tessellation in the toolbox and `convert()` is its n = 1
caller, so almost everything here is a property of the pair rather than of either. What
is asserted, and why each is a way of being wrong quietly:

*Exhaustive and disjoint at the source.* A B-rep face nobody named is not dropped -- it
lands in whatever patch the mesher defaults to, takes that patch's boundary condition,
and the run finishes. So the partition is asserted over face **count**, before a triangle
exists, and cross-checked against `cad_audit.py` counting the same property downstream on
triangles. A disagreement between the two is a bug in one of them.

*Tag stability.* The same selectors on the same geometry twice, and on the geometry after
a rigid transform, give the same partition of face centroids. An implementation leaning
on OpenCASCADE's face ordering passes the first and fails the third.

*Repair that looks like success.* C1 measured `healShapes` at its defaults turning three
solids into none while the volume moved by five parts in ten million. The
round-trip gate here checks solid count and per-solid bounding boxes for exactly that
reason, and one test runs the damaging call on purpose to prove the gate bites.

The recipe halves need build123d and gmsh, and skip where either is absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"
FIXTURES = ROOT / "tests" / "data" / "cad"
SYNTHETIC = FIXTURES / "synthetic"
REAL = FIXTURES / "real" / "ldrobot_ld19_lidar.step"
PREP = TOOLBOX / "templates" / "prep" / "README.md"

HAS_GMSH = importlib.util.find_spec("gmsh") is not None
HAS_B123D = importlib.util.find_spec("build123d") is not None
needs_gmsh = pytest.mark.skipif(not HAS_GMSH, reason="gmsh module not on this machine")
needs_cad = pytest.mark.skipif(
    not (HAS_GMSH and HAS_B123D), reason="the recipe needs gmsh and build123d"
)


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


@pytest.fixture(scope="module")
def answers():
    return json.loads((SYNTHETIC / "answers.json").read_text())["fixtures"]


# -- the recipe, written here the way templates/prep/README.md writes it -------------
#
# Not a copy of a script: there is no script. These are the README's snippets as
# functions so the gate can run them three times and compare, which is the only thing
# a page cannot do for itself.


def heal_and_checkpoint(source: Path, checkpoint: Path) -> dict:
    """Import, repair with sewing off, and write the B-rep out. The expensive stage."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.add("prep")
        gmsh.model.occ.importShapes(str(source))
        gmsh.model.occ.synchronize()
        before = len(gmsh.model.getEntities(3))
        gmsh.model.occ.healShapes(
            fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
            sewFaces=False, makeSolids=True,
        )
        gmsh.model.occ.synchronize()
        after = len(gmsh.model.getEntities(3))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(checkpoint))
        return {"solids_before": before, "solids_after": after,
                "surfaces": len(gmsh.model.getEntities(2))}
    finally:
        gmsh.finalize()


def load_metres(path: Path):
    """The checkpoint is millimetres declaring millimetres; `import_step` reads neither."""
    from build123d import import_step

    return import_step(str(path)).scale(1e-3)


def tag_like_the_readme(shape) -> dict:
    """The README's selectors, re-derived against the geometry in front of them.

    Everything is expressed relative to the shape's own bounding box, so a rigid
    transform moves the selectors with the model instead of leaving them behind.
    """
    from build123d import Axis

    faces = shape.faces()
    box = shape.bounding_box()
    eps = 1e-6
    inlet = faces.filter_by_position(Axis.X, box.min.X - eps, box.min.X + eps)
    outlet = faces.filter_by_position(Axis.X, box.max.X - eps, box.max.X + eps)
    named = list(inlet) + list(outlet)
    walls = [face for face in faces if face not in named]
    return {"inlet": list(inlet), "outlet": list(outlet), "walls": walls}


def keys(faces) -> list:
    """Face centres of mass, which is what `export_patches` names a face by."""
    from build123d import CenterOf

    return [tuple(face.center(CenterOf.MASS)) for face in faces]


def groups_from(tagged: dict) -> dict:
    roles = {"inlet": "inlet", "outlet": "outlet", "walls": "wall"}
    return {
        name: {"role": roles[name], "faces": keys(faces)}
        for name, faces in tagged.items()
    }


def fluid_domain(part):
    """The passage of `box_with_duct.step`, by subtraction, with its ends capped."""
    from build123d import Cylinder, Location

    duct = Cylinder(radius=0.02, height=0.2, rotation=(0, 90, 0)).locate(
        Location((0.1, 0.05, 0.05))
    )
    return duct - part


@pytest.fixture(scope="module")
def duct_export(tmp_path_factory, cad):
    """The whole recipe run once on `box_with_duct.step`, for the tests that only read it."""
    if not (HAS_GMSH and HAS_B123D):
        pytest.skip("the recipe needs gmsh and build123d")
    from build123d import Unit, export_step

    directory = tmp_path_factory.mktemp("duct")
    heal_and_checkpoint(SYNTHETIC / "box_with_duct.step", directory / "prep" / "healed.step")
    part = load_metres(directory / "prep" / "healed.step")
    fluid = fluid_domain(part)
    export_step(fluid, str(directory / "prep" / "fluid.step"), unit=Unit.M)
    tagged = tag_like_the_readme(fluid)
    out = directory / "case" / "constant" / "triSurface"
    report = cad.export_patches(
        directory / "prep" / "fluid.step", groups_from(tagged), out, 0.01,
        location_in_mesh=[0.1, 0.05, 0.05],
    )
    return {"dir": directory, "out": out, "report": report,
            "fluid": fluid, "part": part, "tagged": tagged}


# -- exhaustive and disjoint, asserted over faces before anything is tessellated -----


@needs_cad
def test_every_brep_face_lands_in_exactly_one_patch(duct_export):
    report = duct_export["report"]
    coverage = report["coverage"]
    assert coverage["faces"] == report["entities"]["surfaces"] == 3
    assert coverage["claimed"] == coverage["faces"]
    assert coverage["unclaimed"] == 0
    assert sum(coverage["patches"].values()) == coverage["faces"]
    assert sum(patch["brep_faces"] for patch in report["patches"]) == coverage["faces"]
    # And the triangles follow the faces: nothing was written twice and nothing was lost.
    assert sum(patch["triangles"] for patch in report["patches"]) == report["facets"]


@needs_cad
def test_the_face_count_here_and_the_triangle_count_in_cad_audit_agree(duct_export):
    """C2 checks the same property downstream on triangles. Both must hold."""
    result = subprocess.run(
        [sys.executable, str(TOOLBOX / "cad_audit.py"), str(duct_export["out"]), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout[result.stdout.index("{"):])
    coverage = next(f for f in payload["findings"] if f["check"] == "coverage")
    assert coverage["status"] == "ok", coverage["measured"]
    per_patch = {p["name"]: p["triangles"] for p in duct_export["report"]["patches"]}
    for name, triangles in per_patch.items():
        assert f"{name} {triangles:,}" in coverage["measured"], coverage["measured"]


@needs_cad
def test_a_face_nobody_named_is_a_refusal_not_a_default_patch(cad, duct_export):
    tagged = duct_export["tagged"]
    partial = {"inlet": {"role": "inlet", "faces": keys(tagged["inlet"])}}
    with pytest.raises(cad.Refused) as exc:
        cad.export_patches(
            duct_export["dir"] / "prep" / "fluid.step", partial,
            duct_export["dir"] / "nope", 0.01,
        )
    assert "in no patch" in str(exc.value)
    assert "takes that patch's boundary condition" in str(exc.value)


@needs_cad
def test_a_face_in_two_patches_is_a_refusal(cad, duct_export):
    tagged = duct_export["tagged"]
    doubled = groups_from(tagged)
    doubled["walls"]["faces"] = doubled["walls"]["faces"] + keys(tagged["inlet"])
    with pytest.raises(cad.Refused) as exc:
        cad.export_patches(
            duct_export["dir"] / "prep" / "fluid.step", doubled,
            duct_export["dir"] / "nope", 0.01,
        )
    assert "claimed by both" in str(exc.value)


@needs_cad
def test_a_face_key_out_by_a_thousand_refuses_rather_than_matching_something_else(
    cad, duct_export
):
    tagged = duct_export["tagged"]
    wrong = groups_from(tagged)
    wrong["inlet"]["faces"] = [tuple(v * 1000 for v in p) for p in wrong["inlet"]["faces"]]
    with pytest.raises(cad.Refused) as exc:
        cad.export_patches(
            duct_export["dir"] / "prep" / "fluid.step", wrong,
            duct_export["dir"] / "nope", 0.01,
        )
    assert "centre of mass within" in str(exc.value)
    assert "factor of 1000" in str(exc.value)


# -- tag stability, which is what "re-derived on each run" means ---------------------


@needs_cad
def test_the_tagging_recipe_gives_the_same_partition_three_times(duct_export):
    """Twice from scratch, once after a rigid transform. The third is the one that bites.

    A rotation about X and a translation: the selectors are written against the model's
    own bounding box, so they move with it. What must not move is who is in which patch.
    """
    from build123d import Location
    from OCP.gp import gp_Pnt

    def partition(shape) -> dict:
        return {
            name: sorted(tuple(round(v, 9) for v in c) for c in keys(faces))
            for name, faces in tag_like_the_readme(shape).items()
        }

    first = partition(duct_export["fluid"])
    # From scratch: the same file read again, not the same Python object measured twice.
    again = partition(fluid_domain(load_metres(duct_export["dir"] / "prep" / "healed.step")))
    assert again == first

    location = Location((0.31, -0.22, 0.73), (90, 0, 0))
    transform = location.wrapped.Transformation()

    def mapped(point):
        moved = gp_Pnt(*point).Transformed(transform)
        return (round(moved.X(), 9), round(moved.Y(), 9), round(moved.Z(), 9))

    moved = partition(duct_export["fluid"].moved(location))
    expected = {name: sorted(mapped(c) for c in centres) for name, centres in first.items()}
    assert moved == expected
    assert {name: len(v) for name, v in moved.items()} == {"inlet": 1, "outlet": 1, "walls": 1}


@needs_cad
def test_the_exported_patch_set_is_the_same_after_a_rigid_transform(cad, duct_export, tmp_path):
    """The partition surviving is one thing; the export agreeing with it is the other."""
    from build123d import Location, Unit, export_step

    location = Location((0.31, -0.22, 0.73), (90, 0, 0))
    moved = duct_export["fluid"].moved(location)
    export_step(moved, str(tmp_path / "moved.step"), unit=Unit.M)
    report = cad.export_patches(
        tmp_path / "moved.step", groups_from(tag_like_the_readme(moved)), tmp_path / "out",
        0.01, location_in_mesh=[0.1, 0.05, 0.05],
    )
    before = {p["name"]: p["brep_faces"] for p in duct_export["report"]["patches"]}
    after = {p["name"]: p["brep_faces"] for p in report["patches"]}
    assert after == before
    for patch in report["patches"]:
        was = next(p for p in duct_export["report"]["patches"] if p["name"] == patch["name"])
        # The B-rep area is the same surface and is identical. The tessellated area is
        # not asserted that tightly: the mesher is free to lay its triangles down
        # differently on a rotated model, and it does, by two parts in a million. What
        # is being gated is which faces went where, not which triangles came out.
        assert patch["brep_area_m2"] == pytest.approx(was["brep_area_m2"], rel=1e-9)
        assert patch["area_m2"] == pytest.approx(was["area_m2"], rel=1e-4)


# -- round-trip conservation, and the gate biting on the call that damages -----------


def _occ_measurements(shape) -> dict:
    from OCP.BRepGProp import BRepGProp
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()

    def volume(one):
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(one, props)
        return props.Mass()

    def area(one):
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(one, props)
        return props.Mass()

    def bounds(one):
        box = Bnd_Box()
        BRepBndLib.Add_s(one, box)
        return tuple(round(v, 12) for v in box.Get())

    return {
        "solids": len(solids),
        # Two volumes on purpose. The whole-shape one is what a careless gate reads, and
        # it survives a repair that has dismantled every solid; the sum over solids is
        # the one that notices.
        "volume": volume(shape),
        "solids_volume": sum(volume(s) for s in solids),
        "area": area(shape),
        "boxes": sorted(bounds(s) for s in solids),
    }


def _round_trip(source: Path, checkpoint: Path, sew: bool) -> tuple[dict, dict]:
    """import -> healShapes -> export_step -> re-import, measured on both sides."""
    import gmsh
    from build123d import import_step

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.add("trip")
        gmsh.model.occ.importShapes(str(source))
        gmsh.model.occ.synchronize()
        before = _occ_measurements(import_step(str(source)).scale(1e-3).wrapped)
        gmsh.model.occ.healShapes(
            fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
            sewFaces=sew, makeSolids=True,
        )
        gmsh.model.occ.synchronize()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(checkpoint))
    finally:
        gmsh.finalize()
    after = _occ_measurements(import_step(str(checkpoint)).scale(1e-3).wrapped)
    return before, after


@needs_cad
def test_repair_and_a_step_round_trip_conserve_the_assembly(tmp_path):
    """The real three-solid file, out through STEP and back, with sewing off."""
    before, after = _round_trip(REAL, tmp_path / "healed.step", sew=False)
    assert before["solids"] == 3, before
    assert after["solids"] == before["solids"]
    assert after["volume"] == pytest.approx(before["volume"], rel=1e-6)
    assert after["solids_volume"] == pytest.approx(before["solids_volume"], rel=1e-6)
    assert after["area"] == pytest.approx(before["area"], rel=1e-6)
    assert len(after["boxes"]) == len(before["boxes"])
    for was, now in zip(before["boxes"], after["boxes"]):
        for a, b in zip(was, now):
            assert b == pytest.approx(a, rel=1e-6, abs=1e-9)


@needs_cad
def test_the_round_trip_gate_bites_on_healshapes_at_its_defaults(tmp_path):
    """C1's finding, run on purpose: the defaults lose every solid and keep the volume.

    A gate that checked volume alone would pass this. That is the whole point of
    checking solid count and per-solid boxes, so the damaging call is run here to prove
    the check is not decorative.
    """
    before, after = _round_trip(REAL, tmp_path / "sewn.step", sew=True)
    assert before["solids"] == 3
    assert after["solids"] == 0, (
        "healShapes at its defaults kept the solids on this machine; C1 measured 3 -> 0. "
        "If OCCT has changed, the recipe's sewFaces=False argument wants re-measuring, "
        "not deleting."
    )
    # The trap, stated as an assertion: the number a careless gate would have read moved
    # by five parts in ten million while every solid became a loose bag of faces. Volume
    # is not the check; solid count and the per-solid boxes are.
    assert after["volume"] == pytest.approx(before["volume"], rel=1e-5)
    assert after["solids_volume"] == 0.0
    assert after["boxes"] == []


# -- the fluid boolean is arithmetic -------------------------------------------------


@needs_cad
def test_the_fluid_volume_is_the_box_minus_the_solid(duct_export, answers):
    from build123d import GeomType

    exact = answers["box_with_duct"]
    fluid = duct_export["fluid"]
    part = duct_export["part"]
    box = 0.2 * 0.1 * 0.1
    assert fluid.volume == pytest.approx(box - part.volume, rel=1e-3)
    assert fluid.volume == pytest.approx(exact["fluid_volume_m3"], rel=1e-3)

    caps = [f for f in fluid.faces() if f.geom_type == GeomType.PLANE]
    assert len(caps) == 2
    analytic = math.pi * 0.02 ** 2
    for cap in caps:
        assert cap.is_planar
        assert cap.area == pytest.approx(analytic, rel=1e-3)


# -- defeaturing, against the closed form --------------------------------------------


@needs_cad
def test_defeaturing_recovers_the_corner_the_fillet_took(answers):
    """`(1 - pi/4) * r**2 * L`, true for a right angle and false for anything else."""
    from build123d import Solid
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing

    exact = answers["filleted_block"]
    radius = exact["fillet_radius_m"]
    length = exact["fillet_edge_length_m"]

    def cylinders(shape):
        return [
            face for face in shape.faces()
            if str(BRepAdaptor_Surface(face.wrapped).GetType()).rsplit("_", 1)[-1]
            == "Cylinder"
        ]

    # At the model's true scale: OCCT's defeaturing tolerances are absolute, and at the
    # file's own millimetre magnitudes C1 measured the same call not returning.
    part = load_metres(SYNTHETIC / "filleted_block.step")
    small = cylinders(part)
    assert len(small) == 1

    op = BRepAlgoAPI_Defeaturing()
    op.SetShape(part.wrapped)
    for face in small:
        op.AddFaceToRemove(face.wrapped)
    op.Build()
    assert op.IsDone()
    defeatured = Solid(op.Shape())

    increase = defeatured.volume - part.volume
    assert increase == pytest.approx((1 - math.pi / 4) * radius ** 2 * length, rel=0.01)
    assert len(cylinders(defeatured)) == len(small) - 1


# -- the two refusals, unchanged by the move -----------------------------------------
#
# C1 hashed these against `main`. The hashes are repeated here rather than computed,
# so a reworded refusal is a failure here and not a quietly re-recorded constant.

NO_CLMAX_SHA256 = "a6fac989f746fc0fab6e1302b0bc656e3afe1cee80d2274564bfb82d9e236c15"
NO_UNIT_SHA256 = "7ef1eb051e9f19fe48412865442a688af449cd954c7792b258fe490254998b58"


def test_the_refusal_texts_are_byte_identical_to_the_ones_on_main(cad):
    assert hashlib.sha256(cad.NO_CLMAX.encode()).hexdigest() == NO_CLMAX_SHA256
    assert hashlib.sha256(cad.NO_UNIT.encode()).hexdigest() == NO_UNIT_SHA256


def test_export_patches_refuses_without_clmax_and_says_what_to_pass(cad, tmp_path):
    with pytest.raises(cad.Refused) as exc:
        cad.export_patches(
            SYNTHETIC / "box_with_duct.step",
            [{"name": "all", "role": "wall", "faces": None}], tmp_path, None,
        )
    assert str(exc.value) == cad.NO_CLMAX.format(curvature=cad.CURVATURE_DEFAULT)
    assert str(exc.value).startswith("refused: no --clmax given")


def test_export_patches_refuses_a_file_with_no_unit_and_none_supplied(cad, tmp_path):
    stripped = tmp_path / "nounit.step"
    text = (SYNTHETIC / "box_with_duct.step").read_text(errors="replace")
    stripped.write_text(text.replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.STERADIAN.)"))
    declared = cad.declared_unit(stripped)
    assert declared["unit"] is None
    with pytest.raises(cad.Refused) as exc:
        cad.export_patches(
            stripped, [{"name": "all", "role": "wall", "faces": None}], tmp_path, 0.01,
        )
    assert str(exc.value) == cad.NO_UNIT.format(
        name=stripped.name, evidence=declared["evidence"]
    )


def test_both_refusals_come_back_as_exit_2_at_the_command_line(tmp_path):
    """The desk reports up and the main agent asks the user. It never guesses."""
    stripped = tmp_path / "nounit.step"
    text = (SYNTHETIC / "box_with_duct.step").read_text(errors="replace")
    stripped.write_text(text.replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.STERADIAN.)"))

    no_clmax = subprocess.run(
        [sys.executable, str(TOOLBOX / "cad_convert.py"),
         str(SYNTHETIC / "box_with_duct.step")],
        capture_output=True, text=True,
    )
    assert no_clmax.returncode == 2
    assert no_clmax.stderr.startswith("refused: no --clmax given")

    no_unit = subprocess.run(
        [sys.executable, str(TOOLBOX / "cad_convert.py"), str(stripped), "--clmax", "0.01"],
        capture_output=True, text=True,
    )
    assert no_unit.returncode == 2
    assert no_unit.stderr.startswith(f"refused: {stripped.name} declares no length unit")


def test_the_refusals_still_come_before_gmsh_is_imported(cad, tmp_path, monkeypatch):
    """They are cheap and they stay cheap: no kernel is started to be told no."""
    monkeypatch.setitem(sys.modules, "gmsh", None)
    with pytest.raises(cad.Refused):
        cad.export_patches(
            SYNTHETIC / "box_with_duct.step",
            [{"name": "all", "role": "wall", "faces": None}], tmp_path, None,
        )


def test_a_role_nothing_recognises_refuses(cad, tmp_path):
    with pytest.raises(cad.Refused) as exc:
        cad.export_patches(
            SYNTHETIC / "box_with_duct.step",
            {"side": {"role": "freestream", "faces": None}}, tmp_path, 0.01,
        )
    assert "freestream" in str(exc.value)
    assert "boundary condition nobody chose" in str(exc.value)


# -- one implementation, so the CLI cannot drift from it -----------------------------


@needs_gmsh
def test_the_cli_and_export_patches_write_the_same_bytes(cad, tmp_path):
    """Two tessellation paths that agree today are two that drift tomorrow."""
    through_cli = tmp_path / "cli" / "part.stl"
    through_cli.parent.mkdir()
    result = subprocess.run(
        [sys.executable, str(TOOLBOX / "cad_convert.py"),
         str(SYNTHETIC / "box_with_duct.step"), "--clmax", "0.01",
         "--out", str(through_cli), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    cli_report = json.loads(result.stdout[result.stdout.index("{"):])

    library = tmp_path / "library"
    report = cad.export_patches(
        SYNTHETIC / "box_with_duct.step",
        [{"name": "part", "role": "wall", "faces": None, "file": "part.stl"}],
        library, 0.01, manifest=False,
    )
    assert (library / "part.stl").read_bytes() == through_cli.read_bytes()
    for key in ("facets", "nodes", "sagitta_m", "extent_m", "diagonal_m", "clmax"):
        assert report[key] == pytest.approx(cli_report[key]), key
    assert report["edge_m"]["longest"] == pytest.approx(cli_report["edge_m"]["longest"])
    assert cli_report["coverage"]["unclaimed"] == 0


@needs_gmsh
def test_the_manifest_is_i3_verbatim(duct_export):
    manifest = json.loads((duct_export["out"] / "patches.json").read_text())
    assert set(manifest) == {"unit_metres", "source", "patches", "location_in_mesh"}
    assert manifest["unit_metres"] == 1.0
    assert manifest["source"] == "fluid.step"
    assert manifest["location_in_mesh"] == [0.1, 0.05, 0.05]
    for patch in manifest["patches"]:
        assert set(patch) == {"name", "file", "triangles", "area_m2", "role"}
        assert patch["role"] in ("inlet", "outlet", "wall", "symmetry", "interface")
        assert (duct_export["out"] / patch["file"]).is_file()


# -- the checkpoint is a cache, and deleting it costs time and nothing else ----------


@needs_cad
def test_deleting_the_checkpoint_changes_the_run_and_not_the_answer(cad, tmp_path):
    from build123d import Unit, export_step

    def run(directory: Path) -> dict:
        heal_and_checkpoint(SYNTHETIC / "box_with_duct.step", directory / "healed.step")
        part = load_metres(directory / "healed.step")
        fluid = fluid_domain(part)
        export_step(fluid, str(directory / "fluid.step"), unit=Unit.M)
        report = cad.export_patches(
            directory / "fluid.step", groups_from(tag_like_the_readme(fluid)),
            directory / "triSurface", 0.01, location_in_mesh=[0.1, 0.05, 0.05],
        )
        audit = subprocess.run(
            [sys.executable, str(TOOLBOX / "cad_audit.py"),
             str(directory / "triSurface"), "--json"],
            capture_output=True, text=True,
        )
        assert audit.returncode == 0, audit.stderr
        payload = json.loads(audit.stdout[audit.stdout.index("{"):])
        return {
            "report": report,
            "findings": {f["check"]: f["status"] for f in payload["findings"]},
        }

    warm = tmp_path / "warm"
    warm.mkdir()
    first = run(warm)
    assert (warm / "healed.step").is_file()

    cold = tmp_path / "cold"          # the checkpoint deleted, the recipe from empty
    cold.mkdir()
    second = run(cold)

    assert second["findings"] == first["findings"]
    was = {p["name"]: p for p in first["report"]["patches"]}
    now = {p["name"]: p for p in second["report"]["patches"]}
    assert set(now) == set(was)
    for name, patch in now.items():
        assert patch["triangles"] == was[name]["triangles"]
        assert patch["area_m2"] == pytest.approx(was[name]["area_m2"], rel=1e-9)
        assert patch["brep_faces"] == was[name]["brep_faces"]


# -- thin walls, reported either way -------------------------------------------------


@needs_cad
@pytest.mark.parametrize("thickness", [3e-4, 5e-5])
def test_the_fluid_boolean_on_thin_fins_and_what_it_did(tmp_path, capsys, thickness):
    """OCCT's known weak case, run and written down rather than avoided.

    A failure here is information -- it is the case the tessellated fallback was held
    in reserve for -- so the assertion is not "it worked". It is that whatever came out
    is either arithmetically right and a valid closed solid, or an outright failure.
    A boolean that silently returns a wall short of what it was given fails here, which
    is the only outcome that would be worse than an exception.

    Note that this is synthetic. C1 found the one real-CAD file in the tree to be clean
    -- no face OCCT's checker flags, no free edge -- so the thin-wall and degenerate-face
    outcomes remain **untested on real input**, which PROVENANCE.md says in as many
    words. This says what OCCT does to fins we built, and nothing about a customer's.
    """
    from build123d import Box, Location, Align
    from OCP.BRepCheck import BRepCheck_Analyzer

    # 0.3 mm and 0.05 mm fins: the regime that is known to be hard, and an order of
    # magnitude further into it.
    plate = Box(0.06, 0.02, 0.004, align=(Align.MIN, Align.MIN, Align.MIN))
    part = plate
    for i in range(8):
        fin = Box(thickness, 0.02, 0.02, align=(Align.MIN, Align.MIN, Align.MIN))
        part = part + fin.locate(Location((0.004 + i * 0.007, 0.0, 0.004)))

    flow = Box(0.06, 0.02, 0.03, align=(Align.MIN, Align.MIN, Align.MIN))
    outcome = {"thinnest_wall_m": thickness, "fins": 8}
    try:
        fluid = flow - part
        outcome["completed"] = True
        outcome["fluid_volume_m3"] = fluid.volume
        outcome["expected_m3"] = flow.volume - part.volume
        outcome["solids"] = len(fluid.solids())
        outcome["valid"] = bool(BRepCheck_Analyzer(fluid.wrapped).IsValid())
        outcome["faces"] = len(fluid.faces())
    except Exception as exc:            # noqa: BLE001 -- the outcome, not a surprise
        outcome["completed"] = False
        outcome["error"] = f"{type(exc).__name__}: {exc}"

    print("thin-wall fluid boolean:", json.dumps(outcome, indent=2))
    (tmp_path / f"thin_walls_{thickness:g}.json").write_text(json.dumps(outcome, indent=2))

    if outcome["completed"]:
        assert outcome["valid"], outcome
        assert outcome["fluid_volume_m3"] == pytest.approx(
            outcome["expected_m3"], rel=1e-3
        ), outcome
    else:
        assert outcome["error"], outcome
    assert "completed" in capsys.readouterr().out


# -- the unit static that outlives the session ---------------------------------------


LEAK_PROBE = """
import json, re, sys
from pathlib import Path
sys.path.insert(0, {toolbox!r})
import gmsh
import cad_convert

directory = Path({directory!r})

def write_box(name):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("leak")
    gmsh.model.occ.addBox(0, 0, 0, 100, 40, 20)
    gmsh.model.occ.synchronize()
    gmsh.write(str(directory / name))
    gmsh.finalize()
    text = (directory / name).read_text(errors="replace")
    xs = [float(m) for m in re.findall(r"CARTESIAN_POINT\\('',\\(([-0-9.E+]+),", text)]
    return max(xs)

before = write_box("before.step")
if {control!r}:
    # The bare sequence C1 measured, with nothing put back. The control that proves
    # this probe can see the leak at all.
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.add("q")
    gmsh.model.occ.importShapes({source!r})
    gmsh.model.occ.synchronize()
    gmsh.finalize()
    released = None
else:
    report = cad_convert.export_patches(
        {source!r}, [{{"name": "all", "role": "wall", "faces": None}}],
        str(directory / "out"), 0.01, manifest=False,
    )
    released = report["occ_target_unit_released"]
after = write_box("after.step")
print(json.dumps({{"before": before, "after": after, "ratio": after / before,
                   "released": released}}))
"""


@needs_gmsh
@pytest.mark.parametrize("control", [True, False])
def test_export_patches_puts_the_occ_unit_static_back(tmp_path, control):
    """`Geometry.OCCTargetUnit` outlives the session that set it, and must not outlive this.

    C1 probe 3b: the gmsh option is reset by `finalize()`, the OpenCASCADE static it
    writes through is not, and it is read by the STEP writer as well as the reader. A
    session that sets it and imports leaves every later STEP write in the process
    multiplied by a thousand while still declaring millimetres. STL is unitless, so
    nothing `export_patches` writes is wrong -- what would be wrong is what the next
    caller in a long-lived kernel writes.

    Parametrised against the bare sequence so that "no leak" cannot mean "no instrument".
    """
    directory = tmp_path / ("control" if control else "cured")
    directory.mkdir()
    script = tmp_path / "probe.py"
    script.write_text(LEAK_PROBE.format(
        toolbox=str(TOOLBOX), directory=str(directory), control=control,
        source=str(SYNTHETIC / "box_with_duct.step"),
    ))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    if control:
        assert measured["ratio"] == pytest.approx(1000.0), (
            "the leak C1 measured is gone, so this test proves nothing about the cure; "
            "re-measure probe 3b before deleting the release step"
        )
    else:
        assert measured["released"] is True
        assert measured["ratio"] == pytest.approx(1.0), measured


# -- the recipe folder ---------------------------------------------------------------


def test_the_prep_recipe_offers_rather_than_instructs():
    """The same rule the toolbox index is held to: a page, not a procedure."""
    page = " ".join(PREP.read_text(encoding="utf-8").lower().split())
    for imperative in ("you must", "always run", "before you", "step 1", "first,"):
        assert imperative not in page, imperative
    assert "ignore them" in page


def test_the_recipe_carries_the_gotchas_that_were_paid_for():
    page = PREP.read_text(encoding="utf-8")
    lowered = page.lower()
    assert "sewfaces=false" in lowered, "the argument that keeps the solids"
    assert "five parts in ten million" in lowered, "why volume is not the check"
    assert "b-rep survives all the way to export" in lowered
    assert "re-segmenting" in lowered, "why tessellating early forecloses the rest"
    assert "multi-solid from the start" in lowered
    assert "cache" in lowered and "checkpoint" in lowered
    assert "re-derived" in lowered and "face index" in lowered
    assert "no default" in lowered and "clmax" in lowered
    assert "thin wall" in lowered or "thin-wall" in lowered
    assert "outlives the session" in lowered, "the unit static C1 found"
    assert "absolute" in lowered, "the defeaturing scale trap"
    assert "untested" in lowered or "nothing wrong with it" in lowered


def test_the_recipe_has_no_script_to_copy_and_run():
    """An earlier draft was nine subcommands over the kernel. The page replaces it."""
    assert sorted(p.name for p in PREP.parent.iterdir()) == ["README.md"]


def test_the_toolbox_index_names_the_recipe_folder():
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    assert "templates/prep" in index
