#!/usr/bin/env python3
"""The four CAD probes, run and compared against the answers written down beside them.

`docs/cad-probes.md` is the prose record; `RECORDED` below is the machine copy, and the
two are meant to agree. The point of the script is not that it measures -- anybody can
measure once -- but that re-running it a version later says *which* answer moved, so a
downstream chunk built on one of these numbers finds out from a red run rather than from
a mesh that looks fine.

    python3 scripts/cad_probes.py                # run everything, compare, exit 0 or 1
    python3 scripts/cad_probes.py --dump         # print what was measured, as JSON
    python3 scripts/cad_probes.py --recorded F   # compare against F instead of RECORDED
    python3 scripts/cad_probes.py --only probe2  # one probe
    python3 scripts/cad_probes.py --no-pytest    # skip probe 3's suite re-run

**These answers are image-of-record and the ones below are not from the image.** They
were measured on the dev machine at OCC 7.9.3; the image pins 7.8.1. `#14` reserved the
unit-declaration re-run precisely because that class of behaviour moves between OCC
versions, so certifying 7.8.1 from a 7.9.3 run would repeat the error one version later.
Every entry carries the versions it was taken on, and `docs/cad-probes.md` says which
ones still owe an image run. All of them do.

Nothing under `openreynolds/` may import gmsh, OCP or build123d (Precondition 3). This
file lives under `scripts/`, is not imported by the package, and is not collected by the
test suite; it imports all three deliberately.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAD = ROOT / "tests" / "data" / "cad"
REAL = CAD / "real" / "ldrobot_ld19_lidar.step"
SYNTH = CAD / "synthetic"
PROVENANCE = CAD / "PROVENANCE.md"

# The sentence PROVENANCE.md must carry when the real fixture turns out to be clean --
# see `fixture_is_the_hard_case()`. Kept here so the two files cannot drift apart
# silently: if the wording changes on one side the probe run goes red.
CLEAN_FIXTURE_ADMISSION = (
    "OCCT's checker flags no face on this fixture before healShapes"
)


# -- the recorded answers ------------------------------------------------------------
#
# Measured 2026-09-11 on the dev machine, gmsh 4.15.2 / OCP 7.9.3.1.1, NOT on the image.
# `--dump` prints this shape exactly, so re-recording is a copy rather than a
# transcription. `versions` is compared like everything else: on the image it will not
# match, and that is the first thing the run should say.

RECORDED: dict = json.loads("""
{
  "versions": {
    "gmsh": "4.15.2",
    "cadquery_ocp": "7.9.3.1.1",
    "build123d": "0.11.1"
  },
  "fixture": {
    "real_file": "ldrobot_ld19_lidar.step",
    "real_sha256": "368bed304152e7b5669b1c60511aa2c82b8f152d0740c2d4a16db14562e00ab1",
    "importShapes_solids": 3,
    "importShapes_surfaces": 78,
    "synthetic": {
      "box_with_duct": {
        "solids": 1,
        "volume_m3": 0.001748672587712808,
        "faces": 7
      },
      "three_solids": {
        "solids": 3,
        "volume_m3": 0.003,
        "faces": 18
      },
      "hollow_box": {
        "solids": 1,
        "volume_m3": 0.00013367999999999998,
        "faces": 12
      },
      "filleted_block": {
        "solids": 1,
        "volume_m3": 0.00023967809724509637,
        "faces": 7
      }
    }
  },
  "probe1_defeaturing": {
    "symbol": "OCP.BRepAlgoAPI.BRepAlgoAPI_Defeaturing",
    "reachable": true,
    "synthetic": {
      "cylindrical_faces_before": 1,
      "cylindrical_faces_after": 0,
      "faces_before": 7,
      "faces_after": 6,
      "volume_before_m3": 0.00023967809724509637,
      "volume_after_m3": 0.00024,
      "volume_increase_m3": 3.2190275490364003e-07,
      "analytic_m3": 3.219027549038276e-07,
      "within_one_percent": true
    },
    "real": {
      "fillet_faces_offered": 9,
      "succeeded": true,
      "faces_before": 78,
      "faces_after": 73,
      "volume_before_m3": 4.119370080197598e-05,
      "volume_change_m3": -1.4593289323136657e-10,
      "note": "observation only -- no analytic answer applies to these edges"
    }
  },
  "probe2_deflection": {
    "clmax_m": 0.01,
    "surface": "the duct wall of tests/data/cad/synthetic/box_with_duct.step",
    "radius_m": 0.02,
    "gmsh_defaults": {
      "StlLinearDeflection": 0.001,
      "StlLinearDeflectionRelative": 1.0,
      "StlAngularDeflection": 0.3
    },
    "cells": {
      "defl=0.0001,curv=0": {
        "max_chord_deviation_m": 0.0008803844165120762,
        "triangles": 636,
        "surfaces_on_duct": 1
      },
      "defl=0.0001,curv=2": {
        "max_chord_deviation_m": 0.0008803844165120762,
        "triangles": 636,
        "surfaces_on_duct": 1
      },
      "defl=0.001,curv=0": {
        "max_chord_deviation_m": 0.0008803844165120762,
        "triangles": 636,
        "surfaces_on_duct": 1
      },
      "defl=0.001,curv=2": {
        "max_chord_deviation_m": 0.0008803844165120762,
        "triangles": 636,
        "surfaces_on_duct": 1
      },
      "defl=0.01,curv=0": {
        "max_chord_deviation_m": 0.0008803844165120762,
        "triangles": 636,
        "surfaces_on_duct": 1
      },
      "defl=0.01,curv=2": {
        "max_chord_deviation_m": 0.0008803844165120762,
        "triangles": 636,
        "surfaces_on_duct": 1
      }
    },
    "controls": {
      "clmax/4, curvature off": {
        "max_chord_deviation_m": 6.612081954640234e-05,
        "triangles": 9424,
        "surfaces_on_duct": 1
      },
      "clmax as given, curvature 20": {
        "max_chord_deviation_m": 0.00041265553836212707,
        "triangles": 1484,
        "surfaces_on_duct": 1
      }
    },
    "controls_move_the_deviation": true,
    "columns": {
      "curv=0": {
        "deviation_spread_across_deflection": 0.0,
        "responds": false
      },
      "curv=2": {
        "deviation_spread_across_deflection": 0.0,
        "responds": false
      }
    },
    "deflection_governs_occ_tessellation": false,
    "verdict": "Mesh.StlLinearDeflection does NOT govern OCC tessellation on this path: across two orders of magnitude the max chord deviation does not move, with MeshSizeFromCurvature off or at 2. It governs STL *import* only. C5 may not use the knob. MeshSizeFromCurvature is what moves the facets in curved regions."
  },
  "probe3_units": {
    "refusals": {
      "NO_CLMAX_sha256": "a6fac989f746fc0fab6e1302b0bc656e3afe1cee80d2274564bfb82d9e236c15",
      "NO_UNIT_sha256": "7ef1eb051e9f19fe48412865442a688af449cd954c7792b258fe490254998b58",
      "NO_CLMAX_first_line": "refused: no --clmax given, and there is no default worth guessing.",
      "NO_UNIT_first_line": "refused: {name} declares no length unit, and none was supplied."
    },
    "declaration_line_present_in_occ_output": true,
    "declared_mm_extent_m": [
      0.1000002,
      0.0400002,
      0.0450002
    ],
    "no_unit_extent_raw": [
      100.0000002,
      40.0000002,
      45.0000002
    ],
    "occ_target_unit_rescales_a_declared_file": true,
    "occ_target_unit_is_a_noop_without_a_declaration": true,
    "stripped_file_reads_as_no_unit": true,
    "real_fixture_declared_unit": "metre",
    "real_fixture_unit_evidence": "STEP SI_UNIT($,.METRE.)",
    "real_fixture_extent_in_file_units": [
      53.950047,
      46.83811,
      31.350743
    ],
    "real_fixture_declaration_is_contradicted_by_its_own_numbers": true,
    "suite": {
      "ran": true,
      "returncode": 0,
      "passed": 29,
      "skipped_for_missing_gmsh": false,
      "summary_has_no_failures": true
    }
  },
  "probe3b_unit_leak": {
    "before": {
      "largest_x_in_file": 100.0,
      "declares_millimetres": true
    },
    "after_a_session_that_set_OCCTargetUnit": {
      "largest_x_in_file": 100000.0,
      "declares_millimetres": true
    },
    "write_scale_ratio": 1000.0,
    "leaks_across_finalize": true
  },
  "probe4_healshapes": {
    "file": "ldrobot_ld19_lidar.step",
    "units": "file units (the file declares metres; see probe 3)",
    "before": {
      "faces": 78,
      "solids": 3,
      "volume_file_units3": 41193.70080197623,
      "free_edges": 0,
      "invalid_faces": 0
    },
    "after_defaults": {
      "faces": 78,
      "solids": 0,
      "volume_file_units3": 41193.72112077736,
      "free_edges": 354,
      "invalid_faces": 0,
      "gmsh_entities_before": {
        "volumes": 3,
        "surfaces": 78,
        "curves": 189
      },
      "gmsh_entities_after": {
        "volumes": 0,
        "surfaces": 78,
        "curves": 189
      }
    },
    "after_sewFaces_false": {
      "faces": 78,
      "solids": 3,
      "volume_file_units3": 41193.72112075599,
      "free_edges": 0,
      "invalid_faces": 0,
      "gmsh_entities_before": {
        "volumes": 3,
        "surfaces": 78,
        "curves": 189
      },
      "gmsh_entities_after": {
        "volumes": 3,
        "surfaces": 78,
        "curves": 189
      }
    },
    "defaults_lose_every_solid": true,
    "sewFaces_false_keeps_them": true,
    "nothing_was_flagged_to_begin_with": true,
    "note": "healShapes had no defect to repair here and made one: sewFaces=True is the argument that does it. See docs/cad-probes.md."
  }
}
""")


# -- comparison ----------------------------------------------------------------------

# Dotted paths whose value is an observation, not a claim: recorded so a change is
# visible, compared loosely or not at all where the plan says no assertion belongs.
OBSERVATION_ONLY: tuple[str, ...] = (
    "probe1_defeaturing.real.note",
    "probe2_deflection.verdict",
    "probe4_healshapes.note",
)

# Relative tolerance by dotted path prefix. Anything not listed is compared exactly,
# which is what counts, versions and booleans deserve.
TOLERANCES: tuple[tuple[str, float], ...] = (
    ("probe2_deflection.cells", 0.10),
    ("probe2_deflection.controls", 0.10),
    ("probe1_defeaturing.synthetic.volume_increase_m3", 1e-3),
    ("probe1_defeaturing.synthetic.analytic_m3", 1e-9),
    ("probe1_defeaturing.real.volume_change_m3", 0.05),
    ("probe1_defeaturing.real.volume_before_m3", 1e-6),
    ("probe3_units.real_fixture_extent_in_file_units", 1e-6),
    ("probe3_units.declared_mm_extent_m", 1e-6),
    ("probe3_units.no_unit_extent_raw", 1e-6),
    ("probe3b_unit_leak", 1e-9),
    ("probe4_healshapes.before.volume_file_units3", 1e-6),
    ("probe4_healshapes.after_defaults.volume_file_units3", 1e-6),
    ("probe4_healshapes.after_sewFaces_false.volume_file_units3", 1e-6),
    ("fixture.synthetic", 1e-6),
    ("probe1_defeaturing.synthetic.volume_before_m3", 1e-9),
    ("probe1_defeaturing.synthetic.volume_after_m3", 1e-9),
)


def tolerance_for(path: str) -> float | None:
    best = None
    for prefix, rel in TOLERANCES:
        if path == prefix or path.startswith(prefix + "."):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), rel)
    return None if best is None else best[1]


def flatten(value, prefix: str = "") -> dict:
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{i}]"))
    else:
        out[prefix] = value
    return out


def differences(measured: dict, recorded: dict) -> list[str]:
    """Every way the run disagrees with the record, named by path."""
    got = flatten(measured)
    want = flatten(recorded)
    problems = []

    for path in sorted(set(want) - set(got)):
        problems.append(f"{path}: recorded {want[path]!r}, but the run produced nothing there")
    for path in sorted(set(got) - set(want)):
        problems.append(f"{path}: the run produced {got[path]!r}, which is not in the record")

    for path in sorted(set(got) & set(want)):
        if any(path == o or path.startswith(o.rstrip("*")) for o in OBSERVATION_ONLY):
            continue
        a, b = got[path], want[path]
        rel = tolerance_for(path)
        if rel is not None and isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            if b == 0:
                if abs(a) > rel:
                    problems.append(f"{path}: measured {a!r}, recorded {b!r}")
            elif abs(a - b) / abs(b) > rel:
                problems.append(
                    f"{path}: measured {a!r}, recorded {b!r} "
                    f"(relative change {abs(a - b) / abs(b):.3e} > {rel:.3e})"
                )
        elif a != b:
            problems.append(f"{path}: measured {a!r}, recorded {b!r}")
    return problems


# -- the environment the answers belong to -------------------------------------------


def versions() -> dict:
    import gmsh

    gmsh.initialize()
    try:
        gmsh_version = gmsh.option.getString("General.Version")
    finally:
        gmsh.finalize()

    occ = "unknown"
    for name in ("cadquery-ocp", "cadquery-ocp-novtk", "cadquery-ocp-proxy"):
        try:
            from importlib.metadata import version as _v

            occ = _v(name)
            break
        except Exception:
            continue
    try:
        import build123d

        b123d = getattr(build123d, "__version__", "unknown")
    except Exception:
        b123d = "absent"
    return {"gmsh": gmsh_version, "cadquery_ocp": occ, "build123d": b123d}


# -- OCC helpers ---------------------------------------------------------------------


def occ_read(path: Path, scale: float = 1.0):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader
    from OCP.gp import gp_Trsf

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"could not read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if scale != 1.0:
        trsf = gp_Trsf()
        trsf.SetScaleFactor(scale)
        shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    return shape


def sub_shapes(shape, kind):
    from OCP.TopExp import TopExp_Explorer

    out = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        out.append(explorer.Current())
        explorer.Next()
    return out


def occ_faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS

    return [TopoDS.Face_s(f) for f in sub_shapes(shape, TopAbs_FACE)]


def occ_volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def surface_kind(face) -> str:
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    return str(BRepAdaptor_Surface(face).GetType()).rsplit("_", 1)[-1]


def cylindrical_faces(shape) -> list:
    return [f for f in occ_faces(shape) if surface_kind(f) == "Cylinder"]


def invalid_faces(shape) -> int:
    from OCP.BRepCheck import BRepCheck_Analyzer

    return sum(1 for f in occ_faces(shape) if not BRepCheck_Analyzer(f).IsValid())


def free_edges(shape) -> int:
    """Edges bounding exactly one face. On a closed solid this is zero."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    mapping = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, mapping)
    loose = 0
    for i in range(1, mapping.Extent() + 1):
        if mapping.FindFromIndex(i).Extent() == 1:
            loose += 1
    return loose


def solid_count(shape) -> int:
    from OCP.TopAbs import TopAbs_SOLID

    return len(sub_shapes(shape, TopAbs_SOLID))


@contextlib.contextmanager
def quiet_stdout():
    """OpenCASCADE's STEP writer reports itself on file descriptor 1, below Python's
    `sys.stdout`, so `--dump`'s JSON would arrive with a transfer banner in the middle
    of it. Redirect the descriptor, not the object."""
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


@contextlib.contextmanager
def gmsh_session(quiet: bool = True):
    import gmsh

    gmsh.initialize()
    try:
        if quiet:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
        yield gmsh
    finally:
        gmsh.finalize()


# -- the fixture gate ----------------------------------------------------------------


def fixture() -> dict:
    """Multi-solid, asserted through the same call the toolbox makes, plus the
    synthetic set re-measured off the committed files rather than off the builder."""
    if not REAL.exists():
        raise SystemExit(f"the real fixture is missing: {REAL}")

    with gmsh_session() as gmsh:
        gmsh.model.add("fixture")
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(str(REAL))
        gmsh.model.occ.synchronize()
        solids = len(gmsh.model.getEntities(3))
        surfaces = len(gmsh.model.getEntities(2))

    if solids < 3:
        raise SystemExit(
            f"gate: importShapes returned {solids} solids from {REAL.name}; "
            "the fixture is required to be multi-solid (>= 3)"
        )

    answers = json.loads((SYNTH / "answers.json").read_text())
    synthetic = {}
    for name, facts in answers["fixtures"].items():
        shape = occ_read(SYNTH / facts["file"], scale=1e-3)
        synthetic[name] = {
            "solids": solid_count(shape),
            "volume_m3": occ_volume(shape),
            "faces": len(occ_faces(shape)),
        }
        if synthetic[name]["solids"] != facts["solids"]:
            raise SystemExit(
                f"gate: {facts['file']} holds {synthetic[name]['solids']} solids, "
                f"answers.json says {facts['solids']}"
            )

    return {
        "real_file": REAL.name,
        "real_sha256": hashlib.sha256(REAL.read_bytes()).hexdigest(),
        "importShapes_solids": solids,
        "importShapes_surfaces": surfaces,
        "synthetic": synthetic,
    }


def fixture_is_the_hard_case(invalid_before: int) -> None:
    """The gate's second half. A real-CAD file whose faces OCCT already accepts proves
    nothing about repair, and the plan's answer to that is not to pretend otherwise but
    to make PROVENANCE.md say so in as many words -- checked here, not trusted."""
    if invalid_before > 0:
        return
    text = PROVENANCE.read_text() if PROVENANCE.exists() else ""
    if CLEAN_FIXTURE_ADMISSION not in text:
        raise SystemExit(
            "gate: OCCT's checker flags no face on the fixture before healShapes, so the "
            "hard case is unproven. PROVENANCE.md must say so in as many words and does "
            f"not. Expected to find, verbatim:\n\n    {CLEAN_FIXTURE_ADMISSION}\n"
        )


# -- probe 1: defeaturing, done rather than imported ---------------------------------


def defeature(shape, faces_to_remove):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
    from OCP.TopTools import TopTools_ListOfShape

    listing = TopTools_ListOfShape()
    for face in faces_to_remove:
        listing.Append(face)
    op = BRepAlgoAPI_Defeaturing()
    op.SetShape(shape)
    op.AddFacesToRemove(listing)
    op.Build()
    return op


def probe1_defeaturing() -> dict:
    answers = json.loads((SYNTH / "answers.json").read_text())["fixtures"]["filleted_block"]
    radius = answers["fillet_radius_m"]
    length = answers["fillet_edge_length_m"]
    analytic = (1 - math.pi / 4) * radius ** 2 * length

    # The symbol imports. That is the cheap half and not the probe.
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing  # noqa: F401

    shape = occ_read(SYNTH / "filleted_block.step", scale=1e-3)
    before_volume = occ_volume(shape)
    before_cylinders = cylindrical_faces(shape)
    if len(before_cylinders) != 1:
        raise SystemExit(
            f"gate: the synthetic fillet fixture should carry exactly one cylindrical "
            f"face, found {len(before_cylinders)}"
        )

    op = defeature(shape, before_cylinders)
    if not op.IsDone():
        raise SystemExit("gate: defeaturing the synthetic fillet did not complete")
    healed = op.Shape()
    after_volume = occ_volume(healed)
    after_cylinders = len(cylindrical_faces(healed))

    increase = after_volume - before_volume
    error = abs(increase - analytic) / analytic
    if error > 0.01:
        raise SystemExit(
            f"gate: removing the r={radius} m fillet on an L={length} m right-angle edge "
            f"changed the volume by {increase!r} m^3; (1 - pi/4)*r^2*L is {analytic!r} m^3, "
            f"a relative error of {error:.3%} (allowed 1%)"
        )
    if before_cylinders and after_cylinders != len(before_cylinders) - 1:
        raise SystemExit(
            f"gate: cylindrical faces went {len(before_cylinders)} -> {after_cylinders}; "
            "removing one fillet must drop exactly one"
        )

    synthetic = {
        "cylindrical_faces_before": len(before_cylinders),
        "cylindrical_faces_after": after_cylinders,
        "faces_before": len(occ_faces(shape)),
        "faces_after": len(occ_faces(healed)),
        "volume_before_m3": before_volume,
        "volume_after_m3": after_volume,
        "volume_increase_m3": increase,
        "analytic_m3": analytic,
        "within_one_percent": True,
    }

    # The real file, as an observation. No formula applies: the fillets there meet other
    # features and the edges are not ours to have chosen.
    # The real fixture is read at 1/1000, which is its true scale: it declares metres
    # and is 54 of them across (probe 3), so the numbers in it are millimetres whatever
    # the header says. The correction is not cosmetic here -- OCCT's defeaturing
    # tolerances are absolute, and at the file's own scale the same call had not
    # returned after fifteen minutes. That is itself worth knowing and is written down
    # in docs/cad-probes.md rather than paid for on every run.
    real = occ_read(REAL, scale=1e-3)
    small = []
    for face in occ_faces(real):
        kind = surface_kind(face)
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        adaptor = BRepAdaptor_Surface(face)
        if kind == "Torus" and adaptor.Torus().MinorRadius() <= 0.001:
            small.append(face)
        elif kind == "Cylinder" and adaptor.Cylinder().Radius() <= 0.0006:
            small.append(face)
    real_before_faces = len(occ_faces(real))
    real_before_volume = occ_volume(real)
    op2 = defeature(real, small)
    succeeded = bool(op2.IsDone())
    if succeeded:
        result = op2.Shape()
        real_after_faces = len(occ_faces(result))
        real_after_volume = occ_volume(result)
    else:
        real_after_faces = real_before_faces
        real_after_volume = real_before_volume

    return {
        "symbol": "OCP.BRepAlgoAPI.BRepAlgoAPI_Defeaturing",
        "reachable": True,
        "synthetic": synthetic,
        "real": {
            "fillet_faces_offered": len(small),
            "succeeded": succeeded,
            "faces_before": real_before_faces,
            "faces_after": real_after_faces,
            "volume_before_m3": real_before_volume,
            "volume_change_m3": real_after_volume - real_before_volume,
            "note": "observation only -- no analytic answer applies to these edges",
        },
    }


# -- probe 2: StlLinearDeflection, twice ---------------------------------------------

DEFLECTIONS = (1e-4, 1e-3, 1e-2)
CURVATURES = (0.0, 2.0)
PROBE2_CLMAX = 0.01


def _duct_deviation(gmsh, radius: float, axis_yz: tuple[float, float]) -> dict:
    """Max chord deviation on the duct wall, measured against the exact cylinder.

    The surface is a cylinder of a radius chosen by us, so the distance from any point
    to the B-rep is `radius - hypot(dy, dz)` in closed form and no projection is needed.
    Measured at the midpoint of every triangle edge, which is where a chord departs
    furthest from the arc it replaces."""
    y0, z0 = axis_yz
    worst = 0.0
    triangles = 0
    on_duct = 0
    for dim, tag in gmsh.model.getEntities(2):
        node_tags, coords, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
        if len(node_tags) == 0:
            continue
        points = [coords[i:i + 3] for i in range(0, len(coords), 3)]
        radii = [math.hypot(p[1] - y0, p[2] - z0) for p in points]
        if max(abs(r - radius) for r in radii) > 1e-6:
            continue
        on_duct += 1
        types, elements, nodes = gmsh.model.mesh.getElements(dim, tag)
        coord_of = {}
        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        for i, t in enumerate(all_tags):
            coord_of[int(t)] = all_coords[3 * i:3 * i + 3]
        for etype, elist, nlist in zip(types, elements, nodes):
            if etype != 2:  # 3-node triangle
                continue
            triangles += len(elist)
            for i in range(0, len(nlist), 3):
                tri = [coord_of[int(n)] for n in nlist[i:i + 3]]
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    my = (tri[a][1] + tri[b][1]) / 2
                    mz = (tri[a][2] + tri[b][2]) / 2
                    worst = max(worst, radius - math.hypot(my - y0, mz - z0))
    return {"max_chord_deviation_m": worst, "triangles": triangles, "surfaces_on_duct": on_duct}


def probe2_deflection() -> dict:
    answers = json.loads((SYNTH / "answers.json").read_text())["fixtures"]["box_with_duct"]
    radius = answers["duct_radius_m"]
    axis = tuple(answers["duct_axis_yz_m"])
    path = SYNTH / "box_with_duct.step"

    cells = {}
    defaults = {}
    for deflection in DEFLECTIONS:
        for curvature in CURVATURES:
            with gmsh_session() as gmsh:
                if not defaults:
                    defaults = {
                        "StlLinearDeflection": gmsh.option.getNumber("Mesh.StlLinearDeflection"),
                        "StlLinearDeflectionRelative": gmsh.option.getNumber(
                            "Mesh.StlLinearDeflectionRelative"
                        ),
                        "StlAngularDeflection": gmsh.option.getNumber("Mesh.StlAngularDeflection"),
                    }
                gmsh.model.add("duct")
                gmsh.option.setString("Geometry.OCCTargetUnit", "M")
                gmsh.model.occ.importShapes(str(path))
                gmsh.model.occ.synchronize()
                gmsh.option.setNumber("Mesh.StlLinearDeflection", deflection)
                gmsh.option.setNumber("Mesh.MeshSizeMax", PROBE2_CLMAX)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
                gmsh.model.mesh.generate(2)
                cell = _duct_deviation(gmsh, radius, axis)
            cells[f"defl={deflection:g},curv={curvature:g}"] = cell

    # Two controls, because "the number did not move" and "the measurement cannot see
    # movement" produce identical tables and only one of them is a finding.
    controls = {}
    for label, clmax, curvature in (
        ("clmax/4, curvature off", PROBE2_CLMAX / 4, 0.0),
        ("clmax as given, curvature 20", PROBE2_CLMAX, 20.0),
    ):
        with gmsh_session() as gmsh:
            gmsh.model.add("control")
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            gmsh.model.occ.importShapes(str(path))
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", clmax)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
            gmsh.model.mesh.generate(2)
            controls[label] = _duct_deviation(gmsh, radius, axis)

    by_curvature = {}
    for curvature in CURVATURES:
        column = [cells[f"defl={d:g},curv={curvature:g}"]["max_chord_deviation_m"]
                  for d in DEFLECTIONS]
        spread = (max(column) - min(column)) / max(column) if max(column) else 0.0
        by_curvature[f"curv={curvature:g}"] = {
            "deviation_spread_across_deflection": spread,
            "responds": spread > 0.05,
        }

    baseline = cells[f"defl={DEFLECTIONS[0]:g},curv=0"]["max_chord_deviation_m"]
    controls_move = all(
        abs(c["max_chord_deviation_m"] - baseline) / baseline > 0.05
        for c in controls.values()
    )
    if not controls_move:
        raise SystemExit(
            "gate: probe 2's controls did not move either, so the table says nothing "
            "about StlLinearDeflection -- the deviation measurement itself is blind"
        )

    moves = any(v["responds"] for v in by_curvature.values())
    verdict = (
        "Mesh.StlLinearDeflection moves the chord deviation on this path."
        if moves else
        "Mesh.StlLinearDeflection does NOT govern OCC tessellation on this path: across "
        "two orders of magnitude the max chord deviation does not move, with "
        "MeshSizeFromCurvature off or at 2. It governs STL *import* only. C5 may not use "
        "the knob. MeshSizeFromCurvature is what moves the facets in curved regions."
    )
    return {
        "clmax_m": PROBE2_CLMAX,
        "surface": "the duct wall of tests/data/cad/synthetic/box_with_duct.step",
        "radius_m": radius,
        "gmsh_defaults": defaults,
        "cells": cells,
        "controls": controls,
        "controls_move_the_deviation": controls_move,
        "columns": by_curvature,
        "deflection_governs_occ_tessellation": moves,
        "verdict": verdict,
    }


# -- probe 3: unit declaration on this gmsh/OCC --------------------------------------


def load_cad_convert():
    path = ROOT / "openreynolds" / "toolbox" / "cad_convert.py"
    spec = importlib.util.spec_from_file_location("cad_convert_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _near(got: list, want: list, rel: float = 1e-5) -> bool:
    """OCC's bounding box carries an absolute tolerance, so the box around a 100 mm part
    is 100.0000002 mm wide. Compare the way the number deserves."""
    return all(abs(a - b) <= rel * abs(b) for a, b in zip(got, want))


def _extent_of(shape) -> list:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return [round(xmax - xmin, 6), round(ymax - ymin, 6), round(zmax - zmin, 6)]


def probe3b_unit_leak() -> dict:
    """Does `Geometry.OCCTargetUnit` outlive the gmsh session that set it?

    Found the hard way: probe 3 run *after* probe 2 in one process measured a factor of
    1000 that probe 3 run alone does not. The option is a gmsh option, reset by
    `finalize()`; the OpenCASCADE static it writes through is not. Write a box, set the
    option in a session that does nothing else, finalize, write the same box again --
    and the second file carries coordinates a thousand times larger while still
    declaring millimetres. A file that is wrong by 1000 and says nothing about it is the
    exact failure `cad_convert.py`'s unit refusal exists to prevent, arriving from the
    other end.
    """
    import re

    def write_box(path: Path) -> dict:
        with gmsh_session() as gmsh:
            gmsh.model.add("leak")
            gmsh.model.occ.addBox(0, 0, 0, 100, 40, 20)
            gmsh.model.occ.synchronize()
            gmsh.write(str(path))
        text = path.read_text(errors="replace")
        xs = {float(m) for m in re.findall(r"CARTESIAN_POINT\('',\(([-0-9.E+]+),", text)}
        return {
            "largest_x_in_file": max(xs) if xs else None,
            "declares_millimetres": "SI_UNIT(.MILLI.,.METRE.)" in text.replace(" ", ""),
        }

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        clean = write_box(directory / "before.step")
        with gmsh_session() as gmsh:
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            gmsh.model.add("q")
            gmsh.model.occ.importShapes(str(directory / "before.step"))
            gmsh.model.occ.synchronize()
        after = write_box(directory / "after.step")

    ratio = (after["largest_x_in_file"] / clean["largest_x_in_file"]
             if clean["largest_x_in_file"] else None)
    return {
        "before": clean,
        "after_a_session_that_set_OCCTargetUnit": after,
        "write_scale_ratio": ratio,
        "leaks_across_finalize": ratio is not None and abs(ratio - 1.0) > 1e-9,
    }


def probe3_units(run_pytest: bool = True) -> dict:
    cad = load_cad_convert()
    refusals = {
        "NO_CLMAX_sha256": hashlib.sha256(cad.NO_CLMAX.encode()).hexdigest(),
        "NO_UNIT_sha256": hashlib.sha256(cad.NO_UNIT.encode()).hexdigest(),
        "NO_CLMAX_first_line": cad.NO_CLMAX.splitlines()[0],
        "NO_UNIT_first_line": cad.NO_UNIT.splitlines()[0],
    }

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # The same pair `test_toolbox_cad_convert.py`'s fixture builds: a file that
        # declares millimetres, and the same bytes with the declaration removed.
        with gmsh_session() as gmsh:
            gmsh.model.add("part")
            box = gmsh.model.occ.addBox(0, 0, 0, 100, 40, 20)
            boss = gmsh.model.occ.addCylinder(50, 20, 20, 0, 0, 25, 12)
            gmsh.model.occ.fuse([(3, box)], [(3, boss)])
            gmsh.model.occ.synchronize()
            gmsh.write(str(directory / "mm.step"))

        text = (directory / "mm.step").read_text()
        declared = "#436 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );"
        declaration_line_found = declared in text
        (directory / "nounit.step").write_text(
            text.replace(declared, "#436 = ( LENGTH_UNIT() NAMED_UNIT(*) );")
        )

        def extent(path: Path) -> list:
            with gmsh_session() as gmsh:
                gmsh.model.add(path.stem)
                gmsh.option.setString("Geometry.OCCTargetUnit", "M")
                gmsh.model.occ.importShapes(str(path))
                gmsh.model.occ.synchronize()
                lo = [math.inf] * 3
                hi = [-math.inf] * 3
                for dim, tag in gmsh.model.getEntities(3):
                    box = gmsh.model.getBoundingBox(dim, tag)
                    lo = [min(lo[i], box[i]) for i in range(3)]
                    hi = [max(hi[i], box[i + 3]) for i in range(3)]
                return [round(hi[i] - lo[i], 9) for i in range(3)]

        declared_extent = extent(directory / "mm.step")
        raw_extent = extent(directory / "nounit.step")

        reading = cad.declared_unit(directory / "nounit.step")
        real_reading = cad.declared_unit(REAL)

        # The real fixture's own declaration, against its own numbers. It says metres
        # and it is 54 of them across, which is a lidar the size of a house. Nothing in
        # `cad_convert.py` refuses this: the refusal fires on a *missing* declaration,
        # and a declaration that is simply wrong reads as "converts without ceremony".
        real_shape = occ_read(REAL)
        real_extent = _extent_of(real_shape)
        real_contradiction = max(real_extent) > 1.0 and real_reading["unit"] == "metre"

    suite = {"ran": False}
    if run_pytest:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_toolbox_cad_convert.py"],
            cwd=ROOT, capture_output=True, text=True,
        )
        tail = [line for line in result.stdout.strip().splitlines() if line.strip()]
        summary = tail[-1] if tail else ""
        passed = re.search(r"(\d+) passed", summary)
        suite = {
            "ran": True,
            "returncode": result.returncode,
            "passed": int(passed.group(1)) if passed else None,
            "skipped_for_missing_gmsh": "gmsh module not on this machine" in result.stdout,
            "summary_has_no_failures": " failed" not in summary,
        }

    return {
        "refusals": refusals,
        "declaration_line_present_in_occ_output": declaration_line_found,
        "declared_mm_extent_m": declared_extent,
        "no_unit_extent_raw": raw_extent,
        "occ_target_unit_rescales_a_declared_file": _near(declared_extent, [0.1, 0.04, 0.045]),
        "occ_target_unit_is_a_noop_without_a_declaration": _near(raw_extent, [100.0, 40.0, 45.0]),
        "stripped_file_reads_as_no_unit": reading["unit"] is None,
        "real_fixture_declared_unit": real_reading["unit"],
        "real_fixture_unit_evidence": real_reading["evidence"],
        "real_fixture_extent_in_file_units": real_extent,
        "real_fixture_declaration_is_contradicted_by_its_own_numbers": real_contradiction,
        "suite": suite,
    }


# -- probe 4: healShapes on a real file ----------------------------------------------


def _shape_facts(shape) -> dict:
    return {
        "faces": len(occ_faces(shape)),
        "solids": solid_count(shape),
        "volume_file_units3": occ_volume(shape),
        "free_edges": free_edges(shape),
        "invalid_faces": invalid_faces(shape),
    }


def _heal(tmp: Path, label: str, **kwargs) -> dict:
    """Run `healShapes` one way and read the result back through OCC."""
    healed_path = tmp / f"healed_{label}.step"
    with gmsh_session() as gmsh:
        gmsh.model.add("heal")
        gmsh.model.occ.importShapes(str(REAL))
        gmsh.model.occ.synchronize()
        entities_before = {
            "volumes": len(gmsh.model.getEntities(3)),
            "surfaces": len(gmsh.model.getEntities(2)),
            "curves": len(gmsh.model.getEntities(1)),
        }
        gmsh.model.occ.healShapes(**kwargs)
        gmsh.model.occ.synchronize()
        entities_after = {
            "volumes": len(gmsh.model.getEntities(3)),
            "surfaces": len(gmsh.model.getEntities(2)),
            "curves": len(gmsh.model.getEntities(1)),
        }
        gmsh.write(str(healed_path))
    facts = _shape_facts(occ_read(healed_path))
    facts["gmsh_entities_before"] = entities_before
    facts["gmsh_entities_after"] = entities_after
    return facts


def probe4_healshapes() -> dict:
    """What `healShapes` in fact does to a real file -- which here is damage.

    The plan asks for face count, solid count, volume and free-edge count before and
    after. On this fixture the answer is that the defaults **destroy the assembly**: the
    three solids become none and 354 free edges appear where there were zero, on a file
    OCCT's checker had no complaint about in the first place. `sewFaces=False` is the
    one argument that changes it; every other flag is innocent. Measured both ways here,
    because "before and after" with one setting would have recorded the damage without
    finding the cause.
    """
    before = _shape_facts(occ_read(REAL))

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        defaults = _heal(directory, "defaults")
        no_sew = _heal(directory, "nosew", sewFaces=False)

    fixture_is_the_hard_case(before["invalid_faces"])

    return {
        "file": REAL.name,
        "units": "file units (the file declares metres; see probe 3)",
        "before": before,
        "after_defaults": defaults,
        "after_sewFaces_false": no_sew,
        "defaults_lose_every_solid": defaults["solids"] == 0 and before["solids"] > 0,
        "sewFaces_false_keeps_them": no_sew["solids"] == before["solids"],
        "nothing_was_flagged_to_begin_with": before["invalid_faces"] == 0,
        "note": (
            "healShapes had no defect to repair here and made one: sewFaces=True is the "
            "argument that does it. See docs/cad-probes.md."
        ),
    }


# -- driver --------------------------------------------------------------------------

PROBES = {
    "fixture": lambda args: fixture(),
    "probe1_defeaturing": lambda args: probe1_defeaturing(),
    "probe2_deflection": lambda args: probe2_deflection(),
    "probe3_units": lambda args: probe3_units(run_pytest=not args.no_pytest),
    # Probe 3's second half, and in its own process because measuring the leak leaks:
    # everything written after it in one process is a thousand times too large.
    "probe3b_unit_leak": lambda args: probe3b_unit_leak(),
    "probe4_healshapes": lambda args: probe4_healshapes(),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", action="store_true", help="print the measurements as JSON")
    parser.add_argument("--recorded", type=Path, help="compare against this JSON instead")
    parser.add_argument("--only", action="append", choices=sorted(PROBES),
                        help="run only these probes (repeatable)")
    parser.add_argument("--no-pytest", action="store_true",
                        help="skip probe 3's re-run of tests/test_toolbox_cad_convert.py")
    parser.add_argument("--child", choices=sorted(PROBES), help=argparse.SUPPRESS)
    parser.add_argument("--in-process", action="store_true",
                        help="run every probe in this process. The answers are not "
                             "trustworthy that way -- see the comment below -- and this "
                             "exists to reproduce that, not to save time.")
    args = parser.parse_args(argv)

    if args.child:
        with quiet_stdout():
            result = PROBES[args.child](args)
        print(json.dumps({args.child: result}))
        return 0

    wanted = args.only or list(PROBES)
    with quiet_stdout():
        measured = {"versions": versions()}

    # One process per probe, and not for tidiness. `Geometry.OCCTargetUnit`, set in any
    # gmsh session, leaks through OpenCASCADE's statics into every later `gmsh.write`
    # in the same process -- `probe3b_unit_leak` measures it -- and it silently rescaled
    # probe 3's own fixture by a thousand when probe 3 happened to run after probe 2.
    # The probes are isolated so that they are measuring the kernel and not each other.
    for name in PROBES:
        if name not in wanted:
            continue
        if args.in_process:
            with quiet_stdout():
                measured[name] = PROBES[name](args)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--child", name]
        if args.no_pytest:
            command.append("--no-pytest")
        child = subprocess.run(command, capture_output=True, text=True)
        if child.returncode != 0:
            sys.stderr.write(child.stderr)
            raise SystemExit(f"{name} did not complete (exit {child.returncode})")
        measured.update(json.loads(child.stdout))

    if args.dump:
        print(json.dumps(measured, indent=2, sort_keys=False))
        return 0

    recorded = json.loads(args.recorded.read_text()) if args.recorded else RECORDED
    if not recorded:
        print("no recorded answers to compare against; run with --dump and record them",
              file=sys.stderr)
        return 2

    moved = []
    for name in measured:
        if name not in recorded:
            moved.append((name, [f"{name}: nothing recorded for it"]))
            continue
        problems = differences({name: measured[name]}, {name: recorded[name]})
        if problems:
            moved.append((name, problems))

    if not moved:
        print(f"all {len(wanted)} probes agree with the record "
              f"(gmsh {measured['versions']['gmsh']}, OCP {measured['versions']['cadquery_ocp']})")
        return 0

    print("PROBES MOVED:", ", ".join(name for name, _ in moved), file=sys.stderr)
    for name, problems in moved:
        print(f"\n  {name}:", file=sys.stderr)
        for problem in problems:
            print(f"    {problem}", file=sys.stderr)
    print(
        "\nA moved probe is a finding, not a flake. docs/cad-probes.md records what these "
        "answers were and on which gmsh/OCC; decide which of the two is now right before "
        "re-recording.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
