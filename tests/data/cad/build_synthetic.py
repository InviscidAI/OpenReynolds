#!/usr/bin/env python3
"""Builds the four synthetic CAD fixtures, and checks its own arithmetic while it does.

The real fixture beside these (`real/`) is what a CAD system actually wrote, which is
the case that matters and the case whose answers nobody knows in closed form. These
four are the opposite trade: every number about them is exact by construction, so a
probe that measures one of them is measuring the measurement rather than the geometry.

    box_with_duct.step     a box with a round duct bored through it; the fluid volume
                           is pi*r^2*L exactly
    three_solids.step      three separate solids, two of which overlap by a known
                           rectangular volume -- the file is deliberately *not* fused
    hollow_box.step        a closed box with an internal void; the three wall
                           thicknesses differ so "minimum" is a real choice
    filleted_block.step    one fillet of radius r on one through edge of length L,
                           and the edge is a right angle **by construction**, because
                           (1 - pi/4)*r^2*L holds for 90 degrees and nothing else

Everything is built and reasoned about in **metres** -- `answers.json` is SI throughout --
and written out in **millimetres with the unit declared**, which is what a CAD system
actually emits and what `cad_convert.py`'s declared-unit path is for. Each file is read
back after writing and its volume checked against the analytic answer through that
conversion, so the unit hop is measured rather than assumed.

    python3 tests/data/cad/build_synthetic.py            # rebuild in place, re-check
    python3 tests/data/cad/build_synthetic.py --out DIR  # build somewhere else

Assertions are not a test-suite courtesy here. A downstream chunk quoting 2.513e-4 as
the duct volume is quoting this file; if the arithmetic and the solid ever disagree,
the build is the place that has both and it stops.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import json
import math
import sys
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.Interface import Interface_Static
from OCP.STEPControl import (
    STEPControl_AsIs,
    STEPControl_Controller,
    STEPControl_Reader,
    STEPControl_Writer,
)
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf


# -- the dimensions, in metres, and the only place they are written down -------------

DUCT_BOX = (0.20, 0.10, 0.10)
DUCT_RADIUS = 0.02
DUCT_AXIS_YZ = (0.05, 0.05)

OVERLAP_CUBE = 0.10
OVERLAP_SHIFT = 0.08          # so the overlap slab is 0.02 thick
OVERLAP_THIRD_AT = 0.40       # far enough away to share nothing

HOLLOW_OUTER = (0.100, 0.080, 0.060)
HOLLOW_INNER = (0.090, 0.074, 0.052)   # walls 5, 3 and 4 mm -- the minimum is 3 mm

FILLET_BOX = (0.08, 0.05, 0.06)
FILLET_RADIUS = 0.005
FILLET_EDGE_LENGTH = FILLET_BOX[2]     # the fillet runs the full height

TOL = 1e-9


# -- small helpers -------------------------------------------------------------------


@contextlib.contextmanager
def quiet_stdout():
    """OpenCASCADE's writer reports itself on file descriptor 1, under Python."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def volume_of(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def count(shape: TopoDS_Shape, kind) -> int:
    n = 0
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        n += 1
        explorer.Next()
    return n


def faces(shape: TopoDS_Shape) -> list:
    out = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        out.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    return out


def edges(shape: TopoDS_Shape) -> list:
    out = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        out.append(TopoDS.Edge_s(explorer.Current()))
        explorer.Next()
    return out


def compound(shapes: list) -> TopoDS_Compound:
    result = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, shape)
    return result


def write_step(shape: TopoDS_Shape, path: Path) -> float:
    """Write in millimetres, declaring millimetres, and return the read-back volume in m^3.

    The model is metres; the file is the millimetres a CAD system would have written.
    `STEPControl_Controller.Init_s()` first, because the static parameters do not exist
    until the controller has been initialised and `SetCVal_s` quietly returns False
    into the void if they have not -- which is how a file ends up declaring a unit that
    is not the one its numbers are in.
    """
    STEPControl_Controller.Init_s()
    if not Interface_Static.SetCVal_s("write.step.unit", "MM"):
        raise RuntimeError("could not set write.step.unit")

    to_mm = gp_Trsf()
    to_mm.SetScaleFactor(1000.0)
    in_mm = BRepBuilderAPI_Transform(shape, to_mm, True).Shape()

    writer = STEPControl_Writer()
    with quiet_stdout():
        writer.Transfer(in_mm, STEPControl_AsIs)
        status = writer.Write(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed for {path}")

    text = path.read_text(errors="replace")
    if "SI_UNIT(.MILLI.,.METRE.)" not in text.replace(" ", ""):
        raise RuntimeError(f"{path.name} does not declare millimetres")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"{path.name} did not read back")
    reader.TransferRoots()
    return volume_of(reader.OneShape()) * 1e-9


def close(got: float, want: float, rel: float, what: str) -> None:
    if want == 0:
        raise ValueError("close() is for non-zero targets")
    error = abs(got - want) / abs(want)
    if error > rel:
        raise AssertionError(
            f"{what}: measured {got!r}, analytic {want!r}, "
            f"relative error {error:.3e} > {rel:.3e}"
        )


# -- the four fixtures ---------------------------------------------------------------


def box_with_duct() -> tuple[TopoDS_Shape, dict]:
    lx, ly, lz = DUCT_BOX
    y, z = DUCT_AXIS_YZ
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), lx, ly, lz).Shape()
    axis = gp_Ax2(gp_Pnt(0, y, z), gp_Dir(1, 0, 0))
    duct = BRepPrimAPI_MakeCylinder(axis, DUCT_RADIUS, lx).Shape()
    solid = BRepAlgoAPI_Cut(box, duct).Shape()

    fluid = math.pi * DUCT_RADIUS ** 2 * lx
    bulk = lx * ly * lz
    close(volume_of(solid), bulk - fluid, 1e-9, "box_with_duct solid volume")
    close(volume_of(duct), fluid, 1e-9, "box_with_duct duct volume")
    if count(solid, TopAbs_SOLID) != 1:
        raise AssertionError("box_with_duct should be one solid")

    return solid, {
        "box_m": list(DUCT_BOX),
        "duct_radius_m": DUCT_RADIUS,
        "duct_axis_yz_m": list(DUCT_AXIS_YZ),
        "duct_length_m": lx,
        "fluid_volume_m3": fluid,
        "solid_volume_m3": bulk - fluid,
        "solids": 1,
        "faces": len(faces(solid)),
    }


def three_solids() -> tuple[TopoDS_Shape, dict]:
    a = OVERLAP_CUBE
    first = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), a, a, a).Shape()
    second = BRepPrimAPI_MakeBox(gp_Pnt(OVERLAP_SHIFT, 0, 0), a, a, a).Shape()
    third = BRepPrimAPI_MakeBox(gp_Pnt(OVERLAP_THIRD_AT, 0, 0), a, a, a).Shape()

    overlap = (a - OVERLAP_SHIFT) * a * a
    measured = volume_of(BRepAlgoAPI_Common(first, second).Shape())
    close(measured, overlap, 1e-9, "three_solids overlap volume")

    apart = volume_of(BRepAlgoAPI_Common(second, third).Shape())
    if apart > TOL:
        raise AssertionError(f"the third solid was meant to touch nothing, got {apart!r}")

    shape = compound([first, second, third])
    if count(shape, TopAbs_SOLID) != 3:
        raise AssertionError("three_solids must stay three solids -- do not fuse it")
    close(volume_of(shape), 3 * a ** 3, 1e-9, "three_solids summed volume")

    return shape, {
        "cube_edge_m": a,
        "second_offset_x_m": OVERLAP_SHIFT,
        "third_offset_x_m": OVERLAP_THIRD_AT,
        "solids": 3,
        "each_volume_m3": a ** 3,
        "summed_volume_m3": 3 * a ** 3,
        "overlap_volume_m3": overlap,
        "faces": len(faces(shape)),
    }


def hollow_box() -> tuple[TopoDS_Shape, dict]:
    ox, oy, oz = HOLLOW_OUTER
    ix, iy, iz = HOLLOW_INNER
    outer = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), ox, oy, oz).Shape()
    inner = BRepPrimAPI_MakeBox(
        gp_Pnt((ox - ix) / 2, (oy - iy) / 2, (oz - iz) / 2), ix, iy, iz
    ).Shape()
    solid = BRepAlgoAPI_Cut(outer, inner).Shape()

    walls = ((ox - ix) / 2, (oy - iy) / 2, (oz - iz) / 2)
    thinnest = min(walls)
    close(volume_of(solid), ox * oy * oz - ix * iy * iz, 1e-9, "hollow_box volume")

    # Measured, not just asserted from the construction: the shortest distance from the
    # void's boundary to the outside of the part is the minimum wall.
    distance = BRepExtrema_DistShapeShape(inner, outer)
    distance.Perform()
    if not distance.IsDone():
        raise AssertionError("hollow_box: the wall-thickness measurement did not run")
    # The void touches the outer *solid*, so measure against the outer shell's faces
    # one at a time and take the smallest gap that is not zero-by-containment.
    gaps = []
    for face in faces(outer):
        d = BRepExtrema_DistShapeShape(inner, face)
        d.Perform()
        if d.IsDone():
            gaps.append(d.Value())
    measured_wall = min(gaps)
    close(measured_wall, thinnest, 1e-6, "hollow_box minimum wall thickness")

    if count(solid, TopAbs_SOLID) != 1:
        raise AssertionError("hollow_box should be one solid with a void inside it")
    if len(faces(solid)) != 12:
        raise AssertionError(
            f"hollow_box should carry 12 faces (6 out, 6 in), got {len(faces(solid))}"
        )

    return solid, {
        "outer_m": list(HOLLOW_OUTER),
        "inner_m": list(HOLLOW_INNER),
        "wall_thicknesses_m": list(walls),
        "min_wall_thickness_m": thinnest,
        "min_wall_measured_m": measured_wall,
        "volume_m3": ox * oy * oz - ix * iy * iz,
        "solids": 1,
        "faces": 12,
    }


def _vertical_corner_edge(box: TopoDS_Shape) -> "TopoDS_Shape":
    """The edge at x=0, y=0 running the full height. A 90-degree convex edge, and the
    probe's formula is only true because of that."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_CurveType

    for edge in edges(box):
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_CurveType.GeomAbs_Line:
            continue
        start = curve.Value(curve.FirstParameter())
        end = curve.Value(curve.LastParameter())
        if abs(start.X()) < TOL and abs(end.X()) < TOL and \
           abs(start.Y()) < TOL and abs(end.Y()) < TOL and \
           abs(abs(end.Z() - start.Z()) - FILLET_EDGE_LENGTH) < TOL:
            return edge
    raise AssertionError("could not find the x=0, y=0 vertical edge of the block")


def filleted_block() -> tuple[TopoDS_Shape, dict]:
    lx, ly, lz = FILLET_BOX
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), lx, ly, lz).Shape()
    plain = volume_of(box)

    maker = BRepFilletAPI_MakeFillet(box)
    maker.Add(FILLET_RADIUS, _vertical_corner_edge(box))
    maker.Build()
    if not maker.IsDone():
        raise AssertionError("the fillet did not build")
    solid = maker.Shape()

    removed = (1 - math.pi / 4) * FILLET_RADIUS ** 2 * FILLET_EDGE_LENGTH
    close(plain - volume_of(solid), removed, 1e-6, "filleted_block removed volume")

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType

    cylinders = [
        f for f in faces(solid)
        if BRepAdaptor_Surface(f).GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder
    ]
    if len(cylinders) != 1:
        raise AssertionError(f"expected exactly one cylindrical face, got {len(cylinders)}")
    close(
        BRepAdaptor_Surface(cylinders[0]).Cylinder().Radius(),
        FILLET_RADIUS, 1e-9, "filleted_block fillet radius",
    )

    return solid, {
        "box_m": list(FILLET_BOX),
        "fillet_radius_m": FILLET_RADIUS,
        "fillet_edge_length_m": FILLET_EDGE_LENGTH,
        "fillet_edge_angle_deg": 90.0,
        "plain_box_volume_m3": plain,
        "volume_m3": volume_of(solid),
        "fillet_removed_volume_m3": removed,
        "cylindrical_faces": 1,
        "solids": 1,
        "faces": len(faces(solid)),
    }


BUILDERS = {
    "box_with_duct": box_with_duct,
    "three_solids": three_solids,
    "hollow_box": hollow_box,
    "filleted_block": filleted_block,
}


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    answers: dict = {
        "note": (
            "Analytic answers for the synthetic fixtures, in metres and cubic metres. "
            "Written and re-checked by tests/data/cad/build_synthetic.py; every number "
            "here was compared against the solid before the file was written."
        ),
        "unit": "metre",
        "fixtures": {},
    }
    for name, builder in BUILDERS.items():
        shape, facts = builder()
        path = out / f"{name}.step"
        read_back_m3 = write_step(shape, path)
        # Round trip: what the file says, not only what was in memory.
        analytic = facts.get("volume_m3", facts.get("solid_volume_m3", facts.get("summed_volume_m3")))
        close(read_back_m3, analytic, 1e-6, f"{name} volume after the write/read round trip")
        facts["file"] = path.name
        facts["file_unit"] = "millimetre, declared"
        facts["volume_read_back_m3"] = read_back_m3
        answers["fixtures"][name] = facts
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    (out / "answers.json").write_text(json.dumps(answers, indent=2) + "\n")
    print(f"wrote {out / 'answers.json'}")
    return answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "synthetic",
        help="where to write the fixtures (default: tests/data/cad/synthetic)",
    )
    args = parser.parse_args(argv)
    build(args.out)
    print("every analytic answer agreed with the solid it describes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
