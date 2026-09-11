#!/usr/bin/env python3
"""STEP and IGES to STL, with the tessellation written down rather than assumed.

gmsh on this image is built against OpenCASCADE, so a `.step` or `.iges` file is
read as real B-rep -- trimmed surfaces, exact circles, declared units -- and not as
a tessellated approximation. What it is *not* is self-describing: nothing in the
file says how finely to facet it, and a tessellation chosen badly is a geometry
error that survives all the way into the force coefficients without ever looking
wrong. Tessellate finer than the mesh and you pay for triangles snappyHexMesh
cannot resolve; tessellate coarser and the faceting *becomes* the geometry -- the
mesh reproduces the flat spots faithfully and they reach the pressure field looking
like physics.

So this script decides nothing. You choose `--clmax`; it converts, and reports what
it actually did: the unit it read or was given, the entity counts off the B-rep, the
resolved sizes, the tightest curve it found and the sagitta that implies, the facet
count that came out, and the exact gmsh calls in order. The numbers are the point.
Convert some other way if you prefer -- what is not on offer is converting through
this one without the numbers being visible.

    python3 cad_convert.py part.step --clmax 0.002
    python3 cad_convert.py part.step --clmax 0.002 --out constant/triSurface/part.stl
    python3 cad_convert.py part.iges --clmax 0.002 --unit mm --curvature 20
    python3 cad_convert.py part.step --clmax 0.002 --json

It refuses in exactly two cases, both of them cases where a wrong answer looks like a
right one: no `--clmax`, and a file that declares no unit with no `--unit` given.

A case does not consume one merged STL, it consumes one STL per named patch plus the
manifest beside them, so the tessellation here is `export_patches()` and the command
line above is its n = 1 caller -- one group, every face in it. Import it from a cell
(`import cad_convert; cad_convert.export_patches(...)`) with the face groups the
selectors found; `templates/prep/README.md` is the recipe that gets you as far as
having them.

There is no statistics report here on purpose. The module is importable
(`import gmsh`), so the B-rep answers questions directly -- entity lists,
`getBoundingBox`, `getMass` for lengths and areas, `getCurvature`, `getEntities` --
and a fixed set of numbers chosen in advance would only tell you which numbers
somebody else thought mattered. The case that matters is the one with a feature
nobody thought to measure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

CAD_SUFFIXES = (".step", ".stp", ".iges", ".igs")
"""What this reads. The five toolbox surface readers know the same four suffixes, so
a CAD file lands as CAD awaiting conversion rather than as an unreadable surface."""

STEP_SUFFIXES = (".step", ".stp")
IGES_SUFFIXES = (".iges", ".igs")

CURVATURE_DEFAULT = 20.0
"""`Mesh.MeshSizeFromCurvature`: facets per 2*pi of turning, so curvature drives the
facet size in curved regions rather than the global cap. 20 puts roughly 20 chords
round a full circle wherever a circle is tighter than `clmax` would notice."""

TOLERANCE_DEFAULT = 1e-8
"""`Geometry.Tolerance` -- how close two B-rep entities must be to be the same one.
This is geometry fidelity, not facet density: the two are separate knobs.

The image now pins one gmsh, 4.15.2, for both the module and the CLI. 4.15 *does*
surface `Mesh.StlLinearDeflection` and `StlAngularDeflection`, which 4.12 did not --
a direct bound on chord deviation, i.e. on the sagitta this script computes and
reports as a consequence of `clmax`. It is **not used here, and the reason is now a
measurement rather than a reservation**: `Mesh.StlLinearDeflection` does not govern
OCC tessellation on this path. It governs STL *import*. Swept across two orders of
magnitude (1e-4, 1e-3, 1e-2 m) against a cylinder of known radius, the max chord
deviation came back bit-identical in all six cells and the triangle counts with it --
the same mesh six times -- while the controls in the same run moved the deviation by
more than an order of magnitude, so the instrument could see movement and there was
none to see. C1 probe 2, `docs/cad-probes.md`.

What does move the facets in curved regions is `Mesh.MeshSizeFromCurvature`, which is
the knob below and which this script already sets and already reports."""

METRES_PER_UNIT = {
    "m": 1.0, "metre": 1.0, "meter": 1.0,
    "dm": 0.1, "cm": 0.01, "mm": 0.001, "millimetre": 0.001, "millimeter": 0.001,
    "um": 1e-6, "micron": 1e-6,
    "in": 0.0254, "inch": 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
}

CURVE_SAMPLES = 24
"""Parametric samples per curve when looking for the tightest radius. The number is
a compromise and is reported with the answer, because "the tightest curve found"
and "the tightest curve there is" are not the same claim on a sampled search."""


# -- what the file says about its own units ----------------------------------------

_STEP_SI = re.compile(
    r"LENGTH_UNIT\s*\(\s*\)[^;]*?SI_UNIT\s*\(\s*(\.[A-Z]+\.|\$)\s*,\s*\.METRE\.\s*\)",
    re.S | re.I,
)
_STEP_SI_ALT = re.compile(
    r"SI_UNIT\s*\(\s*(\.[A-Z]+\.|\$)\s*,\s*\.METRE\.\s*\)[^;]*?LENGTH_UNIT\s*\(\s*\)",
    re.S | re.I,
)
_STEP_CONVERSION = re.compile(
    r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'", re.I
)

_SI_PREFIX = {
    "$": ("metre", 1.0),
    ".METRE.": ("metre", 1.0),
    ".DECI.": ("decimetre", 0.1),
    ".CENTI.": ("centimetre", 0.01),
    ".MILLI.": ("millimetre", 0.001),
    ".MICRO.": ("micrometre", 1e-6),
    ".KILO.": ("kilometre", 1000.0),
}


def declared_unit(path: Path) -> dict:
    """What the file declares its length unit to be, and where that was read.

    `{"unit": name or None, "metres": float or None, "evidence": str}`. A `None`
    unit is the case the conversion refuses on: OpenCASCADE will read such a file
    perfectly happily and hand back the raw numbers, and whether those numbers are
    millimetres or metres is a factor of 1000 on every length in the study.
    """
    suffix = path.suffix.lower()
    try:
        head = path.read_text(errors="replace")
    except OSError as exc:
        return {"unit": None, "metres": None, "evidence": f"unreadable ({exc})"}

    if suffix in STEP_SUFFIXES:
        for pattern in (_STEP_SI, _STEP_SI_ALT):
            match = pattern.search(head)
            if match:
                prefix = match.group(1).upper()
                name, metres = _SI_PREFIX.get(prefix, (None, None))
                if name:
                    return {
                        "unit": name, "metres": metres,
                        "evidence": f"STEP SI_UNIT({match.group(1)},.METRE.)",
                    }
        match = _STEP_CONVERSION.search(head)
        if match:
            name = match.group(1).strip().lower()
            metres = METRES_PER_UNIT.get(name)
            return {
                "unit": name, "metres": metres,
                "evidence": f"STEP CONVERSION_BASED_UNIT('{match.group(1)}')",
            }
        return {
            "unit": None, "metres": None,
            "evidence": "no LENGTH_UNIT SI_UNIT(...,.METRE.) and no CONVERSION_BASED_UNIT in the file",
        }

    if suffix in IGES_SUFFIXES:
        # The Global section is comma-delimited with a 'G' in column 73. Parameter 14
        # is the units flag and 15 the unit name; both are frequently 3/"MM" or 1/"IN".
        globals_text = "".join(
            line[:72] for line in head.splitlines() if len(line) > 72 and line[72] == "G"
        )
        fields = globals_text.split(",")
        if len(fields) > 14:
            name = fields[14].strip().strip("0123456789H").strip().lower()
            metres = METRES_PER_UNIT.get(name)
            if metres:
                return {
                    "unit": name, "metres": metres,
                    "evidence": f"IGES Global parameter 15 = {fields[14].strip()!r}",
                }
        return {
            "unit": None, "metres": None,
            "evidence": "IGES Global section names no unit this reads",
        }

    return {"unit": None, "metres": None, "evidence": f"{suffix} is not a CAD suffix this reads"}


# -- the refusals ------------------------------------------------------------------


class Refused(Exception):
    """A conversion that would have produced a plausible-looking wrong answer."""


NO_CLMAX = """\
refused: no --clmax given, and there is no default worth guessing.

The tessellation edge length is the decision this conversion exists to make explicit.
Facet the surface finer than the mesh and you pay for triangles snappyHexMesh cannot
resolve; facet it coarser and the faceting becomes the geometry -- the mesh reproduces
the flat spots faithfully and they arrive in the pressure field looking like physics.

Pass --clmax in metres. Tessellation follows the finest surface cell size at the wall:
clmax ~= 0.5 * dx_surface. Mesh.MeshSizeFromCurvature (--curvature, {curvature:g} unless
you say otherwise) then drives the facets in curved regions rather than the global cap.
The sagitta -- how far a chord of length L departs from a surface of radius R -- is
~= L^2 / (8R), which is the number that says whether the faceting is subgrid; this
reports it against the tightest curve it can find once you have chosen."""

NO_UNIT = """\
refused: {name} declares no length unit, and none was supplied.

{evidence}.

OpenCASCADE reads the file regardless and hands back the numbers as they are written.
Whether those numbers are millimetres or metres is a factor of 1000 on every length in
the study, and the wrong one produces a perfectly plausible mesh, a converged solve, and
completely wrong forces.

Pass --unit to say what the numbers mean: --unit mm (also m, cm, dm, um, in, ft). It is
applied as an explicit occ.dilate by that many metres per file unit, and reported.
Geometry.OCCTargetUnit is not used for this case: measured on gmsh 4.15.2 with OCC, it
rescales a file that *does* declare a unit and is a no-op on one that does not, so it
cannot carry an override.

A file that declares its unit needs none of this and converts without ceremony."""


# -- the conversion ----------------------------------------------------------------


def _sample_min_radius(gmsh, curves) -> dict:
    """The tightest curve found, by sampling curvature along each one.

    Reported as "found", not "is": this is a finite sample of the parametrisation and
    a fillet shorter than the sample spacing can hide between two samples.
    """
    tightest = math.inf
    where = None
    for dim, tag in curves:
        try:
            lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
        except Exception:
            continue
        # `lo` and `hi` come back as numpy arrays, where `not lo` is a value test
        # and not an emptiness test: on a curve starting at parameter 0 it is True.
        if len(lo) == 0 or len(hi) == 0:
            continue
        span = float(hi[0]) - float(lo[0])
        if span <= 0:
            continue
        coords = [float(lo[0]) + span * i / (CURVE_SAMPLES - 1) for i in range(CURVE_SAMPLES)]
        try:
            curvature = gmsh.model.getCurvature(dim, tag, coords)
        except Exception:
            continue
        for value in curvature:
            if value > 1e-12:
                radius = 1.0 / float(value)
                if radius < tightest:
                    tightest = radius
                    where = tag
    if where is None:
        return {"radius": None, "curve": None, "samples": CURVE_SAMPLES}
    return {"radius": tightest, "curve": where, "samples": CURVE_SAMPLES}


# -- per-patch export, which is what a case actually consumes ----------------------

PATCH_ROLES = ("inlet", "outlet", "wall", "symmetry", "interface")
"""The five roles I3's manifest carries. `patch_entries.py` turns them into OpenFOAM
patch types, so a role invented here becomes a boundary condition nobody chose."""

SEED_STEP = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('one point, so OpenCASCADE re-reads its own unit static'),'2;1');
FILE_NAME('occ-target-unit-seed','1970-01-01T00:00:00',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('',(0.,0.,0.));
#2 = VERTEX_POINT('',#1);
#3 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
ENDSEC;
END-ISO-10303-21;
"""
"""A one-point STEP, imported for its side effect and for nothing else. See
`_release_occ_target_unit`."""


def _release_occ_target_unit(gmsh, calls: list[str] | None = None) -> bool:
    """Put `Geometry.OCCTargetUnit` back where it was found, which `finalize()` does not.

    Measured (C1 probe 3b, `docs/cad-probes.md`): the gmsh *option* is reset by
    `gmsh.finalize()`, the OpenCASCADE static it writes through is not, and that static
    is read by the STEP **writer** as well as the reader. A session that sets the option
    to `M` and imports leaves every later STEP write in the process multiplied by a
    thousand and still declaring millimetres -- across two `finalize()` calls, in the
    probe that found it. STL is unitless and unaffected, so nothing this file writes is
    wrong; what is wrong is what the *next* caller writes, and a kernel that lives for a
    whole session is exactly such a next caller.

    gmsh only touches the static on the way into `importShapes`, so the option alone
    cannot put it back: setting it to `MM` and importing something is the only lever
    there is, and `SEED_STEP` is the smallest something. Cheap, and it leaves the static
    at the value a fresh process has.
    """
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / "occ_target_unit_seed.step"
            seed.write_text(SEED_STEP)
            gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
            gmsh.model.add("occ_target_unit_seed")
            gmsh.model.occ.importShapes(str(seed))
            gmsh.model.remove()
        if calls is not None:
            calls.append(
                "gmsh.option.setString('Geometry.OCCTargetUnit', 'MM'); "
                "gmsh.model.occ.importShapes(<one-point seed>)  # release the OCC static"
            )
        return True
    except Exception:
        return False


def _normalise_groups(groups) -> list[dict]:
    """Face groups as given -- a mapping or a sequence -- into one shape.

    Each group is `{"name", "role", "faces", "file"}`. `faces` is a list of face
    centroids **in metres**, or `None` for the one group that takes whatever is left.
    """
    out: list[dict] = []
    if isinstance(groups, dict):
        items = [(name, spec) for name, spec in groups.items()]
    else:
        items = []
        for spec in groups:
            if not isinstance(spec, dict) or "name" not in spec:
                raise Refused(
                    "refused: a face group given as a sequence needs a 'name' in each "
                    f"entry; got {spec!r}."
                )
            items.append((spec["name"], spec))
    for name, spec in items:
        if not isinstance(spec, dict):
            spec = {"faces": spec}
        role = str(spec.get("role", "wall"))
        if role not in PATCH_ROLES:
            raise Refused(
                f"refused: patch {name!r} has role {role!r}, which is not one of "
                f"{', '.join(PATCH_ROLES)}. The role is what patch_entries.py turns into "
                "an OpenFOAM patch type, so a role nothing recognises becomes a boundary "
                "condition nobody chose."
            )
        faces = spec.get("faces", None)
        if faces is not None:
            points = []
            for point in faces:
                values = [float(v) for v in point]
                if len(values) != 3:
                    raise Refused(
                        f"refused: patch {name!r} was given the face key {point!r}, which "
                        "is not an (x, y, z) centroid in metres."
                    )
                points.append(tuple(values))
            faces = points
        out.append({
            "name": str(name),
            "role": role,
            "faces": faces,
            "file": str(spec.get("file") or f"{name}.stl"),
        })
    if not out:
        raise Refused(
            "refused: no face groups given, so there is nothing to name and nothing to "
            "write. One group with faces=None exports the whole surface as one patch."
        )
    rest = [g["name"] for g in out if g["faces"] is None]
    if len(rest) > 1:
        raise Refused(
            f"refused: {', '.join(repr(n) for n in rest)} each ask for whatever is left. "
            "At most one group can be the catch-all, or which of them gets an unclaimed "
            "face is a coin toss."
        )
    names = [g["name"] for g in out]
    if len(set(names)) != len(names):
        raise Refused(f"refused: two face groups share a name in {names!r}.")
    files = [g["file"] for g in out]
    if len(set(files)) != len(files):
        raise Refused(f"refused: two face groups would write the same file in {files!r}.")
    return out


def _assign_faces(surfaces: list[dict], groups: list[dict], match_tolerance: float) -> dict:
    """Face centroids to patches: exhaustive, disjoint, and asserted before tessellation.

    `surfaces` is every B-rep face gmsh imported, with its centre of mass in metres.
    Each group names its faces by centroid rather than by index, because face order is
    OpenCASCADE's and changes when anything upstream of it changes -- which is the whole
    reason selectors are re-derived on each run rather than recorded.
    """
    claimed: dict[int, str] = {}
    for group in groups:
        group["tags"] = []
        if group["faces"] is None:
            continue
        for point in group["faces"]:
            best = None
            for surface in surfaces:
                distance = math.dist(point, surface["centre"])
                if best is None or distance < best[0]:
                    best = (distance, surface)
            if best is None or best[0] > match_tolerance:
                nearest = "" if best is None else (
                    f" The nearest face centre is {best[0]:.6g} m away, at "
                    f"({', '.join(f'{v:.6g}' for v in best[1]['centre'])})."
                )
                raise Refused(
                    f"refused: patch {group['name']!r} names a face at "
                    f"({', '.join(f'{v:.6g}' for v in point)}) and no face of this "
                    f"geometry has its centre of mass within {match_tolerance:.6g} m of "
                    f"it.{nearest}\n\n"
                    "Face keys are centres of mass in metres, in the coordinates of the "
                    "geometry as exported -- build123d's `face.center(CenterOf.MASS)`, "
                    "not `face.center()`, which is the bounding-box centre and lands off "
                    "a cylinder. A key that is out by a factor of 1000 is a model in "
                    "millimetres against a file that declares them."
                )
            tag = best[1]["tag"]
            if tag in claimed:
                raise Refused(
                    f"refused: the face at "
                    f"({', '.join(f'{v:.6g}' for v in best[1]['centre'])}) is claimed by "
                    f"both {claimed[tag]!r} and {group['name']!r}. A face in two patches "
                    "is two surfaces in one place, and snappyHexMesh will snap to "
                    "whichever it reads last."
                )
            claimed[tag] = group["name"]
            group["tags"].append(tag)

    unclaimed = [s for s in surfaces if s["tag"] not in claimed]
    rest = next((g for g in groups if g["faces"] is None), None)
    if rest is not None:
        rest["tags"] = [s["tag"] for s in unclaimed]
        for surface in unclaimed:
            claimed[surface["tag"]] = rest["name"]
        unclaimed = []
    if unclaimed:
        raise Refused(
            f"refused: {len(unclaimed)} of {len(surfaces)} B-rep faces are in no patch.\n\n"
            "Their centres of mass, in metres:\n"
            + "\n".join(
                f"  ({', '.join(f'{v:.6g}' for v in s['centre'])})  area {s['area']:.6g} m^2"
                for s in unclaimed[:20]
            )
            + ("\n  ..." if len(unclaimed) > 20 else "")
            + "\n\nAn unassigned face is not dropped: it lands in whatever patch the "
            "mesher defaults to and takes that patch's boundary condition, and the run "
            "finishes. Name it, or give one group faces=None to take the remainder."
        )
    empty = [g["name"] for g in groups if not g["tags"]]
    if empty:
        raise Refused(
            f"refused: {', '.join(repr(n) for n in empty)} matched no face at all. An "
            "empty patch file is a patch the mesher will not find, and the boundary it "
            "was meant to be will be somebody else's default."
        )
    return {
        "faces": len(surfaces),
        "claimed": len(claimed),
        "unclaimed": 0,
        "patches": {g["name"]: len(g["tags"]) for g in groups},
    }


def _triangles_by_surface(gmsh) -> dict:
    """Triangle count and area per B-rep face, measured off the mesh that was written.

    The areas here are sums over triangles rather than `occ.getMass`, because the number
    in the manifest should be a fact about the STL a mesher will read and not about the
    surface it approximates. The two differ by exactly the faceting error, which is the
    quantity `clmax` was chosen to control.
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(tag): i for i, tag in enumerate(node_tags)}
    out: dict[int, dict] = {}
    for dim, tag in gmsh.model.getEntities(2):
        count = 0
        area = 0.0
        types, _tags, connectivity = gmsh.model.mesh.getElements(dim, tag)
        for element_type, nodes in zip(types, connectivity):
            if gmsh.model.mesh.getElementProperties(element_type)[3] != 3:
                continue
            nodes = [int(n) for n in nodes]
            for i in range(0, len(nodes), 3):
                a, b, c = (index[n] for n in nodes[i:i + 3])
                pa = coords[3 * a:3 * a + 3]
                pb = coords[3 * b:3 * b + 3]
                pc = coords[3 * c:3 * c + 3]
                u = [pb[k] - pa[k] for k in range(3)]
                v = [pc[k] - pa[k] for k in range(3)]
                cross = (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )
                area += 0.5 * math.sqrt(sum(w * w for w in cross))
                count += 1
        out[tag] = {"triangles": count, "area": area}
    return out


def export_patches(
    path: Path,
    groups,
    out_dir: Path,
    clmax: float | None,
    unit: str | None = None,
    curvature: float = CURVATURE_DEFAULT,
    tolerance: float = TOLERANCE_DEFAULT,
    clmin: float | None = None,
    binary: bool = False,
    location_in_mesh=None,
    manifest: bool = True,
    match_tolerance: float | None = None,
    source: str | None = None,
) -> dict:
    """One CAD file to one STL per patch, plus I3's `patches.json`, at a size you chose.

    This is the tessellation, and there is one of it. `convert()` is this function with a
    single group that takes every face, so the CLI and a twenty-patch chassis export go
    down the same code path and carry the same two refusals -- two tessellation paths
    that agree today are two that drift apart on the first bug fix to one of them.

    `groups` maps patch names to face groups, either as `{name: spec}` or as a sequence
    of specs each carrying its own `name`. A spec is
    `{"role": one of PATCH_ROLES, "faces": [(x, y, z), ...]}`, where each `(x, y, z)` is
    a face's **centre of mass in metres**, in the coordinates of the geometry as
    exported. `"faces": None` means "whatever is left", and at most one group may say it.

    Faces are named by centroid rather than by index on purpose: OpenCASCADE's face order
    is a property of the file and of the operations that produced it, so an export keyed
    on it is correct until the model changes and silently wrong afterwards. The selectors
    that produced these centroids are re-derived on each run; the centroids are what the
    run found.

    Every face is asserted into exactly one patch **before** anything is tessellated, so
    a face nobody named is a refusal rather than a surface that quietly takes a default
    boundary condition. `cad_audit.py` checks the same property downstream on triangles;
    both must hold, and a disagreement between them is a bug in one of the two.

    Raises `Refused` -- exit 2 at the command line -- rather than guessing.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    normalised = _normalise_groups(groups)

    if clmax is None or clmax <= 0:
        raise Refused(NO_CLMAX.format(curvature=CURVATURE_DEFAULT))

    declared = declared_unit(path)
    supplied = None
    if unit:
        key = unit.strip().lower()
        supplied = METRES_PER_UNIT.get(key)
        if supplied is None:
            raise Refused(
                f"refused: --unit {unit!r} is not a unit this knows. "
                f"One of: {', '.join(sorted(set(METRES_PER_UNIT)))}."
            )

    if declared["metres"] is None and supplied is None:
        raise Refused(NO_UNIT.format(name=path.name, evidence=declared["evidence"]))

    if location_in_mesh is not None:
        location_in_mesh = [float(v) for v in location_in_mesh]
        if len(location_in_mesh) != 3:
            raise Refused(
                f"refused: location_in_mesh is {location_in_mesh!r}, which is not a point. "
                "It is one (x, y, z) in metres, inside the fluid; domain_probe.py "
                "--suggest proposes one and checks it."
            )

    import gmsh  # deferred: everything above is readable without the module present

    calls: list[str] = []

    def call(text: str) -> None:
        calls.append(text)

    report: dict = {
        "input": str(path),
        "output": str(out_dir),
        "format": "STEP" if path.suffix.lower() in STEP_SUFFIXES else "IGES",
        "declared_unit": declared,
        "supplied_unit": {"unit": unit, "metres": supplied} if supplied else None,
        "clmax": float(clmax),
        "clmin": float(clmin) if clmin else None,
        "curvature": float(curvature),
        "tolerance": float(tolerance),
        "calls": calls,
    }

    gmsh.initialize()
    call("gmsh.initialize()")
    target_unit_set = False
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.Tolerance", tolerance)
        call(f"gmsh.option.setNumber('Geometry.Tolerance', {tolerance!r})")

        # Units, one of two ways, and never both.
        #
        # A declared unit is converted by OpenCASCADE's own reader: OCCTargetUnit = M
        # makes it hand back metres whatever the file said. A file that declares
        # nothing gets an explicit dilate instead -- measured, OCCTargetUnit is a
        # no-op on such a file (it converts *from* a declared unit and there is none),
        # so using it as an override would silently do nothing at all.
        if declared["metres"] is not None:
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            target_unit_set = True
            call("gmsh.option.setString('Geometry.OCCTargetUnit', 'M')")
            factor = None
            interpretation = (
                f"file declares {declared['unit']} ({declared['evidence']}); "
                f"Geometry.OCCTargetUnit = M converts it to metres"
            )
            if supplied is not None:
                interpretation += (
                    f"; --unit {unit} was given and is ignored -- the file's own "
                    "declaration wins over a flag"
                )
        else:
            factor = supplied
            interpretation = (
                f"file declares no unit ({declared['evidence']}); --unit {unit} taken as "
                f"{supplied:g} m per file unit and applied as occ.dilate"
            )

        gmsh.model.add(path.stem)
        gmsh.model.occ.importShapes(str(path))
        call(f"gmsh.model.occ.importShapes({str(path)!r})")

        if factor is not None:
            entities = gmsh.model.occ.getEntities()
            gmsh.model.occ.dilate(entities, 0, 0, 0, factor, factor, factor)
            call(f"gmsh.model.occ.dilate(<all>, 0, 0, 0, {factor!r}, {factor!r}, {factor!r})")

        gmsh.model.occ.synchronize()
        call("gmsh.model.occ.synchronize()")

        report["unit_interpretation"] = interpretation
        report["entities"] = {
            "volumes": len(gmsh.model.getEntities(3)),
            "surfaces": len(gmsh.model.getEntities(2)),
            "curves": len(gmsh.model.getEntities(1)),
            "points": len(gmsh.model.getEntities(0)),
        }
        if report["entities"]["surfaces"] == 0:
            raise Refused(
                f"refused: {path.name} imported with no surfaces at all. "
                "OpenCASCADE read the file and found no B-rep faces in it -- an empty "
                "assembly, a wireframe-only export, or a file this reader does not "
                "understand. There is nothing here to tessellate."
            )

        surfaces = [
            {
                "tag": tag,
                "centre": tuple(float(v) for v in gmsh.model.occ.getCenterOfMass(2, tag)),
                "area": float(gmsh.model.occ.getMass(2, tag)),
            }
            for dim, tag in gmsh.model.getEntities(2)
        ]
        box = gmsh.model.getBoundingBox(-1, -1)
        diagonal = math.dist(box[0:3], box[3:6]) or 1.0
        if match_tolerance is None:
            match_tolerance = 1e-6 * diagonal
        report["match_tolerance_m"] = float(match_tolerance)

        # Exhaustive and disjoint, asserted here -- before a single triangle exists.
        report["coverage"] = _assign_faces(surfaces, normalised, float(match_tolerance))

        tight = _sample_min_radius(gmsh, gmsh.model.getEntities(1))
        report["tightest_curve"] = tight
        if tight["radius"]:
            report["sagitta_m"] = clmax * clmax / (8.0 * tight["radius"])
        else:
            report["sagitta_m"] = None

        gmsh.option.setNumber("Mesh.MeshSizeMax", clmax)
        call(f"gmsh.option.setNumber('Mesh.MeshSizeMax', {float(clmax)!r})")
        if clmin:
            gmsh.option.setNumber("Mesh.MeshSizeMin", clmin)
            call(f"gmsh.option.setNumber('Mesh.MeshSizeMin', {float(clmin)!r})")
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        call(f"gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', {float(curvature)!r})")
        if binary:
            gmsh.option.setNumber("Mesh.Binary", 1)
            call("gmsh.option.setNumber('Mesh.Binary', 1)")

        gmsh.model.mesh.generate(2)
        call("gmsh.model.mesh.generate(2)")

        types, tags, _nodes = gmsh.model.mesh.getElements(2)
        facets = sum(len(t) for t in tags)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        report["facets"] = facets
        report["nodes"] = len(node_tags)

        # The bounding box is measured off the mesh nodes rather than off
        # `getBoundingBox`, so that the number reported is a fact about the STL that
        # was written and not about OpenCASCADE's internal box. Those two are not
        # always the same: OCC's box carries an absolute tolerance that survives a
        # rescale unscaled, and on a dilated import it read 41.6 mm across a part
        # that is 40.0 mm across. A scale report that is itself 4% out is worse than
        # no scale report, since the whole point of it is catching factors.
        bounds = [
            [min(node_coords[axis::3]) for axis in range(3)],
            [max(node_coords[axis::3]) for axis in range(3)],
        ] if len(node_coords) else [[0.0] * 3, [0.0] * 3]
        extent = [float(bounds[1][axis] - bounds[0][axis]) for axis in range(3)]
        report["bounds_m"] = [float(v) for v in bounds[0] + bounds[1]]
        report["extent_m"] = extent
        report["diagonal_m"] = math.sqrt(sum(v * v for v in extent))

        report["edge_m"] = _edge_lengths(gmsh)

        per_surface = _triangles_by_surface(gmsh)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[dict] = []
        for group in normalised:
            physical = gmsh.model.addPhysicalGroup(2, group["tags"], name=group["name"])
            target = out_dir / group["file"]
            gmsh.write(str(target))
            call(
                f"gmsh.model.addPhysicalGroup(2, {group['tags']!r}, "
                f"name={group['name']!r}); gmsh.write({str(target)!r})"
            )
            gmsh.model.removePhysicalGroups([(2, physical)])
            written.append({
                "name": group["name"],
                "file": group["file"],
                "role": group["role"],
                "brep_faces": len(group["tags"]),
                "brep_area_m2": sum(
                    s["area"] for s in surfaces if s["tag"] in set(group["tags"])
                ),
                "triangles": sum(per_surface[t]["triangles"] for t in group["tags"]),
                "area_m2": sum(per_surface[t]["area"] for t in group["tags"]),
                "bytes": target.stat().st_size if target.exists() else 0,
            })
        report["patches"] = written
        report["bytes"] = sum(p["bytes"] for p in written)

        report["location_in_mesh"] = location_in_mesh
        if manifest:
            manifest_body = {
                "unit_metres": 1.0,
                "source": source or path.name,
                "patches": [
                    {
                        "name": p["name"],
                        "file": p["file"],
                        "triangles": p["triangles"],
                        "area_m2": p["area_m2"],
                        "role": p["role"],
                    }
                    for p in written
                ],
                "location_in_mesh": location_in_mesh,
            }
            manifest_path = out_dir / "patches.json"
            manifest_path.write_text(json.dumps(manifest_body, indent=2) + "\n")
            report["manifest"] = str(manifest_path)
            call(f"<wrote {str(manifest_path)!r}>")
        else:
            report["manifest"] = None
    finally:
        if target_unit_set:
            report["occ_target_unit_released"] = _release_occ_target_unit(gmsh, calls)
        gmsh.finalize()
        call("gmsh.finalize()")
    return report


def convert(
    path: Path,
    out: Path,
    clmax: float | None,
    unit: str | None = None,
    curvature: float = CURVATURE_DEFAULT,
    tolerance: float = TOLERANCE_DEFAULT,
    clmin: float | None = None,
    binary: bool = False,
) -> dict:
    """Tessellate one CAD file to one STL at a size you chose, and report what happened.

    The n = 1 case of `export_patches()`: one group, every face in it, no manifest. Not a
    second implementation of the same thing -- it is a call, so the two cannot drift and
    the refusals are the same objects.

    Raises `Refused` rather than guessing, in the two cases where guessing produces
    something that looks right. Returns the report as a dict; `render()` prints it.
    """
    out = Path(out)
    report = export_patches(
        path, [{"name": out.stem, "role": "wall", "faces": None, "file": out.name}],
        out.parent, clmax,
        unit=unit, curvature=curvature, tolerance=tolerance, clmin=clmin, binary=binary,
        manifest=False,
    )
    report["output"] = str(out)
    report["bytes"] = report["patches"][0]["bytes"]
    return report


def _edge_lengths(gmsh) -> dict:
    """Longest and mean triangle edge actually produced.

    `clmax` is a request; this is what came out. They differ -- gmsh's size field is
    a target, curvature refinement pulls edges below it, and a short curve can force
    one above it -- and the difference is exactly the thing a reported request would
    hide.
    """
    try:
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(tag): i for i, tag in enumerate(node_tags)}
        types, _tags, connectivity = gmsh.model.mesh.getElements(2)
        longest = 0.0
        total = 0.0
        count = 0
        for element_type, nodes in zip(types, connectivity):
            per = gmsh.model.mesh.getElementProperties(element_type)[3]
            if per != 3:
                continue
            nodes = [int(n) for n in nodes]
            for i in range(0, len(nodes), 3):
                triangle = nodes[i:i + 3]
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    ia, ib = index[triangle[a]], index[triangle[b]]
                    d = math.dist(coords[3 * ia:3 * ia + 3], coords[3 * ib:3 * ib + 3])
                    longest = max(longest, d)
                    total += d
                    count += 1
        return {"longest": longest, "mean": total / count if count else 0.0, "edges": count}
    except Exception as exc:
        return {"longest": None, "mean": None, "edges": 0, "note": str(exc)}


# -- the render tessellation, which is a different thing entirely -------------------

RENDER_CHORDS_PER_DIAGONAL = 25
"""How coarse a picture is allowed to be: `clmax = diagonal / 25`.

Set by what a 700-pixel panel can show and by nothing else. 60 was tried first and
was wrong for the reason that matters here: on a 117 mm part it lands within 3% of a
sensible *simulation* clmax, so the picture and the mesh happened to agree and the
distinction this function exists to draw became invisible in its own test."""


def render_tessellation(path: Path, directory: Path) -> Path:
    """Triangles for *drawing* a CAD file. Not a geometry decision.

    `geometry_view.py`'s four views need something to draw, and a picture of a STEP
    file is worth having before anyone has chosen a mesh size. That is the whole
    reason this exists, and it is worth being blunt about what separates it from
    `convert()`:

    - nothing downstream consumes it. It never reaches snappyHexMesh, never reaches a
      solver, never lands in `constant/triSurface`. It is written into a caller-owned
      temporary directory and is expected to be thrown away with it.
    - so it may be as coarse as looks right, and it is: `diagonal / 60`, chosen by
      this function and reported by it, not by anyone.
    - and so it does not refuse a file that declares no unit. A picture at the wrong
      scale is still the right picture; a *mesh* at the wrong scale is wrong forces.
      `geometry_view.py` says which case it is looking at.

    If you find yourself about to make a case read this output, or about to unify its
    size with `convert()`'s `--clmax` so there is "one tessellation", stop: that
    unification is precisely the accident this file exists to prevent. The simulation
    tessellation follows the mesh; the render tessellation follows the page.
    """
    import gmsh

    path = Path(path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / (path.stem + ".render.stl")

    declared = declared_unit(path)
    target_unit_set = False
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if declared["metres"] is not None:
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            target_unit_set = True
        gmsh.model.add(path.stem)
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        box = gmsh.model.getBoundingBox(-1, -1)
        diagonal = math.dist(box[0:3], box[3:6]) or 1.0
        gmsh.option.setNumber("Mesh.MeshSizeMax", diagonal / RENDER_CHORDS_PER_DIAGONAL)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 8)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out))
    finally:
        # The same leak `export_patches()` cures, for the same reason and by the same
        # lever. Drawing a picture is the most innocent thing in this file and it sets
        # the identical OpenCASCADE static: a session that renders a STEP declaring
        # millimetres and later writes one gets coordinates a thousand times too large
        # with `MILLI` still written in the header. Four scripts import this function
        # and none of them is in a position to know that happened.
        if target_unit_set:
            _release_occ_target_unit(gmsh)
        gmsh.finalize()
    return out


# -- printing ----------------------------------------------------------------------


def render(report: dict) -> str:
    """The report as text, which is the only thing that leaves this script."""
    lines: list[str] = []
    add = lines.append
    add(f"{report['input']}  ->  {report['output']}")
    add(f"  format              {report['format']}")
    add(f"  unit                {report['unit_interpretation']}")
    entities = report["entities"]
    add(
        f"  entities            {entities['volumes']} volumes, {entities['surfaces']} surfaces, "
        f"{entities['curves']} curves, {entities['points']} points"
    )
    extent = report["extent_m"]
    add(
        f"  bounding box        {extent[0]:.6g} x {extent[1]:.6g} x {extent[2]:.6g} m "
        f"(diagonal {report['diagonal_m']:.6g} m)"
    )
    add(f"  clmax               {report['clmax']:.6g} m  (Mesh.MeshSizeMax)")
    if report.get("clmin"):
        add(f"  clmin               {report['clmin']:.6g} m  (Mesh.MeshSizeMin)")
    add(f"  curvature           {report['curvature']:g}  (Mesh.MeshSizeFromCurvature)")
    add(f"  tolerance           {report['tolerance']:g}  (Geometry.Tolerance)")
    tight = report["tightest_curve"]
    if tight["radius"]:
        add(
            f"  tightest curve      R = {tight['radius']:.6g} m on curve {tight['curve']} "
            f"({tight['samples']} samples per curve, so: found, not proved)"
        )
        add(
            f"  implied sagitta     {report['sagitta_m']:.6g} m  = clmax^2 / (8R) -- the worst a "
            "chord of clmax could depart from that curve"
        )
        add(
            "                      curvature refinement puts the real chords below clmax there, "
            "so this is a ceiling"
        )
    else:
        add("  tightest curve      none found -- no curve in this file has measurable curvature")
    edge = report["edge_m"]
    add(f"  facets              {report['facets']:,} triangles, {report['nodes']:,} nodes")
    if edge.get("longest") is not None:
        add(
            f"  edges produced      longest {edge['longest']:.6g} m, mean {edge['mean']:.6g} m "
            f"over {edge['edges']:,} triangle edges"
        )
    patches = report.get("patches") or []
    if len(patches) > 1 or report.get("manifest"):
        coverage = report.get("coverage") or {}
        add(
            f"  patches             {len(patches)} over "
            f"{coverage.get('faces', '?')} B-rep faces, every face in exactly one"
        )
        for patch in patches:
            add(
                f"    {patch['name']:<18} {patch['role']:<10} "
                f"{patch['brep_faces']:>4} faces  {patch['triangles']:>8,} triangles  "
                f"{patch['area_m2']:.6g} m^2  -> {patch['file']}"
            )
        point = report.get("location_in_mesh")
        add(
            "  location_in_mesh    "
            + (", ".join(f"{v:.6g}" for v in point) if point else
               "not given -- domain_probe.py --suggest proposes one and checks it")
        )
        if report.get("manifest"):
            add(f"  manifest            {report['manifest']}")
    add(f"  written             {report['bytes']:,} bytes")
    add("  gmsh calls, in order:")
    for text in report["calls"]:
        add(f"    {text}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="STEP/IGES -> STL with the tessellation reported, not assumed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "clmax is yours to choose and is not defaulted. Tessellation follows the\n"
            "finest surface cell size at the wall: clmax ~= 0.5 * dx_surface, with\n"
            "--curvature ~ 20 so curvature drives the facets where the surface bends.\n"
            "Sagitta for a chord L on radius R is ~ L^2 / 8R."
        ),
    )
    parser.add_argument("path", type=Path, help="a .step, .stp, .iges or .igs file")
    parser.add_argument("--clmax", type=float, default=None, help="max facet edge, metres")
    parser.add_argument("--clmin", type=float, default=None, help="min facet edge, metres")
    parser.add_argument("--unit", default=None, help="what the numbers mean, if the file does not say")
    parser.add_argument("--curvature", type=float, default=CURVATURE_DEFAULT)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_DEFAULT)
    parser.add_argument("--out", type=Path, default=None, help="STL to write (default: alongside)")
    parser.add_argument("--binary", action="store_true", help="write binary STL")
    parser.add_argument("--json", action="store_true", help="the report as JSON as well")
    args = parser.parse_args(argv)

    if args.path.suffix.lower() not in CAD_SUFFIXES:
        print(
            f"refused: {args.path.name} is not one of {', '.join(CAD_SUFFIXES)}. "
            "A tessellated surface needs no conversion -- geometry_view.py reads it "
            "as it stands.",
            file=sys.stderr,
        )
        return 2
    if not args.path.is_file():
        print(f"refused: {args.path} does not exist", file=sys.stderr)
        return 2

    out = args.out or args.path.with_suffix(".stl")
    try:
        report = convert(
            args.path, out, args.clmax,
            unit=args.unit, curvature=args.curvature, tolerance=args.tolerance,
            clmin=args.clmin, binary=args.binary,
        )
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(render(report))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
