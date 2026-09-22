#!/usr/bin/env python3
"""Look at a mesh and measure it: one picture, one table, one verdict.

    python3 mesh_look.py /work/study/mesh                      # picture + report
    python3 mesh_look.py . --out look.png --json look.json     # and the machine-readable one
    python3 mesh_look.py . --no-check                          # skip checkMesh (faster)
    python3 mesh_look.py . --region heater                     # one region of a CHT case

`--out` and `--json` are taken relative to where you run this from -- the working
directory, as for every other toolbox script -- and NOT relative to the case. Without
`--out` the picture is `<case>/look.png`. The report's `picture:` line is the absolute
path that was written, so there is nothing to work out. (Until 2026-09-21 a relative
`--out` was joined to the case instead: `mesh_look.py mesh --out look.png` run from a
study directory wrote `mesh/look.png` and printed `picture: look.png`, and the caller
read `look.png` where it stood -- a 404 and three turns to find the file, in study
20260920-161908-c7ef.)

Every panel is drawn from `constant/polyMesh` -- no solve, no fields, no time
directory needed. `--region` reads `constant/<name>/polyMesh` instead, which is the
only thing it changes: one region per call, the same report with the same keys, so a
multi-region case is looked at one payload at a time rather than in a shape nothing
downstream knows how to read. The boundary is coloured **one colour per patch**, because the
question that gets answered wrong most often is not "is this the right shape" but "is
the inlet the end I think it is": a picture where the inlet patch is a different colour
from the outlet answers it in one look, and the table under it gives each patch's area,
its centre and the direction it faces, which answers it in numbers.

What it reports:

* cells, faces, points, the bounding box, and whether the mesh is one cell thick (a
  plane case) or a volume;
* every patch: type, face count, area, centre, mean unit normal;
* every cell zone and face zone in `constant/polyMesh`, with its count -- and for a cell
  zone its bounding box and centroid where the mesh is ascii. A sliding interface, a
  frozen rotor and an overset region are all named against a zone, and a dictionary that
  names a zone the mesh does not have fails at the first solver step;
* `checkMesh`'s own verdict and the three metrics a solver actually minds -- maximum
  non-orthogonality, maximum skewness, maximum aspect ratio;
* which files in the case would rebuild it.

It measures the mesh. It cannot measure the *request* -- whether the branch is at 20
degrees, whether the gap is half a diameter -- and does not pretend to: that number has
to be computed from the geometry by whoever asked for it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import math
from pathlib import Path

PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#eeca3b",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#1b9e77", "#d95f02",
]

CHECK_TIMEOUT_S = 240


# -- the files ----------------------------------------------------------------


def mesh_dir(case: Path, region: str = "") -> Path:
    """Where this case's mesh files are: `constant/polyMesh`, or one region's."""
    return case / "constant" / region / "polyMesh" if region else case / "constant" / "polyMesh"


def boundary_entries(case: Path, region: str = "") -> list[dict]:
    """Patch name, type and face count straight out of `constant/polyMesh/boundary`.

    Read as text on purpose: it works with no OpenFOAM environment, no reader and no
    time directory, so a mesh that nothing else can open still reports its patches.
    """
    path = mesh_dir(case, region) / "boundary"
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    body = text[text.find("("): text.rfind(")") + 1] if "(" in text else ""
    out: list[dict] = []
    for match in re.finditer(r"(\w+)\s*\{([^}]*)\}", body):
        name, block = match.group(1), match.group(2)
        entry = {"name": name, "type": _key(block, "type") or "patch",
                 "nFaces": int(_key(block, "nFaces") or 0)}
        if _key(block, "inGroups"):
            entry["inGroups"] = _key(block, "inGroups")
        out.append(entry)
    return out


def _key(block: str, key: str) -> str:
    match = re.search(rf"\b{key}\s+([^;]+);", block)
    return match.group(1).strip() if match else ""


# -- the zones ----------------------------------------------------------------
#
# Zones were invisible here until 2026-09-12. Everything that moves a mesh or freezes a
# rotor names one -- `cyclicAMI` couples a pair of face zones, `MRFProperties` names a
# cell zone, an overset region is a cell zone -- and a dictionary naming a zone the mesh
# does not have fails at the first solver step with an error about a name nobody typed
# twice. The mesh was already being walked; this is the read that makes the zone
# checkable before the dictionary is written against it.

ZONE_FILES = (("cellZones", "cell"), ("faceZones", "face"))

_ZONE_ENTRY = re.compile(
    # name { ... cellLabels List<label> N ( ... )
    # `[^}]` keeps the span inside one block, so the `FoamFile { ... }` header cannot
    # reach forward into the first zone's label list and be reported as a zone.
    r"([A-Za-z_][\w.\-]*)\s*\{([^}]{0,400}?)\b(cellLabels|faceLabels)\s*"
    r"(?:List<label>\s*)?(\d+)",
    re.S,
)


def zone_entries(case: Path) -> list[dict]:
    """Every cell zone and face zone: name, kind and count.

    Read as text like `boundary` is, and for the same reason -- no OpenFOAM environment,
    no reader. It works on a binary polyMesh too, because the label count is written in
    plain text immediately before the binary block: only the labels themselves are
    unreadable that way, and those are needed for the extents rather than for the name.
    """
    out: list[dict] = []
    poly = case / "constant" / "polyMesh"
    for filename, kind in ZONE_FILES:
        path = poly / filename
        if not path.is_file():
            continue
        try:
            text = path.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue
        wanted = "cellLabels" if kind == "cell" else "faceLabels"
        for match in _ZONE_ENTRY.finditer(text):
            name, _between, keyword, count = match.groups()
            if keyword != wanted:
                continue
            out.append({"name": name, "kind": kind, "count": int(count),
                        "file": f"constant/polyMesh/{filename}"})
    return out


def _zone_labels(case: Path, kind: str) -> dict[str, list[int]]:
    """`{zone name: cell or face labels}` for one zone file, ascii only.

    A binary zone file returns nothing rather than a guess: the count is still reported
    from `zone_entries`, and a zone whose extents could not be measured says so.
    """
    filename = "cellZones" if kind == "cell" else "faceZones"
    path = case / "constant" / "polyMesh" / filename
    if not path.is_file():
        return {}
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return {}
    if not re.search(r"\bformat\s+ascii\s*;", text[:2000]):
        return {}
    wanted = "cellLabels" if kind == "cell" else "faceLabels"
    found: dict[str, list[int]] = {}
    for match in _ZONE_ENTRY.finditer(text):
        name, _between, keyword, count = match.groups()
        if keyword != wanted:
            continue
        open_at = text.find("(", match.end())
        if open_at < 0:
            continue
        depth, close_at = 0, -1
        for index in range(open_at, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    close_at = index
                    break
        if close_at < 0:
            continue
        labels = [int(token) for token in text[open_at + 1:close_at].split()]
        if len(labels) == int(count):
            found[name] = labels
    return found


def measure_cell_zones(case: Path, zones: list[dict]) -> None:
    """Bounding box and centroid onto each cell zone entry, in place.

    A zone's extent is what says whether the rotating region actually surrounds the
    blade, and it is the one fact about a zone that a name and a count cannot carry.

    The box is the box containing the zone's cell *centres*, not its outer vertices, so
    it is short of the true extent by about half a cell on each face. That is stated
    here and in the report rather than corrected, because the correction would need a
    per-cell vertex walk and the half-cell is far below the question the number is asked
    for -- whether the zone is where it was meant to be.
    The arithmetic is `layer_report.py`'s -- it already reads `points`, `faces` and
    `owner` and builds cell centres by OpenFOAM's own pyramid decomposition, and having
    two polyMesh readers in this toolbox that disagree would be worse than having one
    that refuses a binary mesh out loud.
    """
    cells = [zone for zone in zones if zone.get("kind") == "cell" and zone.get("count")]
    if not cells:
        return

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:  # the sibling import works when run from anywhere
        sys.path.insert(0, here)
    try:
        import numpy as np

        import layer_report
    except Exception as exc:  # noqa: BLE001 - a mesh with no extents is still a mesh
        for zone in cells:
            zone["unmeasured"] = f"the zone extents need numpy ({type(exc).__name__}: {exc})"
        return

    labels = _zone_labels(case, "cell")
    try:
        mesh = layer_report.load(case)
    except Exception as exc:  # noqa: BLE001 - MeshError on a binary or partial polyMesh
        for zone in cells:
            zone["unmeasured"] = str(exc)
        return

    for zone in cells:
        wanted = labels.get(zone["name"])
        if wanted is None:
            zone["unmeasured"] = (
                f"{zone['file']} is not ascii, so the labels could not be read -- "
                "`foamFormatConvert` on a copy puts the extents within reach"
            )
            continue
        index = np.asarray(wanted, dtype=np.int64)
        if index.size == 0 or int(index.max()) >= int(mesh["n_cells"]):
            zone["unmeasured"] = "the zone names cells this polyMesh does not have"
            continue
        try:
            centres = layer_report.cell_centres(mesh, index)
        except Exception as exc:  # noqa: BLE001
            zone["unmeasured"] = f"{type(exc).__name__}: {exc}"
            continue
        low, high = centres.min(axis=0), centres.max(axis=0)
        zone["bounds"] = [float(v) for v in (*low, *high)]
        zone["centre"] = [float(v) for v in centres.mean(axis=0)]


def build_files(case: Path) -> list[str]:
    """What in this case would rebuild the mesh from nothing."""
    names = ["Allmesh", "Allrun", "build.py", "geometry.py", "mesh.py",
             "system/blockMeshDict", "system/snappyHexMeshDict", "system/meshDict"]
    found = [n for n in names if (case / n).exists()]
    found += sorted(p.name for p in case.glob("*.geo"))
    return found


# -- checkMesh ----------------------------------------------------------------


def run_check(case: Path, region: str = "") -> dict:
    """`checkMesh`, its verdict, its counts and the three metrics that matter.

    The log is kept at `log.checkMesh` whatever happens: the summary here is the news,
    and the failing checks in full are what somebody reads next.
    """
    out = {"ok": False, "verdict": "", "metrics": {}, "counts": {}, "bounds": []}
    try:
        command = ["checkMesh", "-case", str(case)]
        if region:
            command += ["-region", region]
        proc = subprocess.run(command, cwd=str(case),
                              capture_output=True, text=True, timeout=CHECK_TIMEOUT_S)
        log = proc.stdout + proc.stderr
    except FileNotFoundError:
        out["verdict"] = "checkMesh is not on PATH here"
        return out
    except subprocess.TimeoutExpired:
        out["verdict"] = f"checkMesh did not finish in {CHECK_TIMEOUT_S} s"
        return out
    name = f"log.checkMesh.{region}" if region else "log.checkMesh"
    (case / name).write_text(log, errors="replace")
    return parse_check(log)


def parse_check(log: str) -> dict:
    """The verdict, counts, bounds and metrics in a checkMesh log."""
    out: dict = {"ok": False, "verdict": "", "metrics": {}, "counts": {}, "bounds": []}
    if "Mesh OK." in log:
        out["ok"] = True
        out["verdict"] = "Mesh OK."
    else:
        failed = re.search(r"Failed (\d+) mesh checks?", log)
        stars = [line.strip() for line in log.splitlines() if line.strip().startswith("***")]
        if failed:
            out["verdict"] = f"failed {failed.group(1)} mesh checks"
        elif stars:
            out["verdict"] = f"{len(stars)} failing checks"
        elif "FOAM FATAL" in log:
            # An empty reason after the colon is what this reported on a live run --
            # "checkMesh stopped: " and nothing after it, which tells the reader less
            # than the raw log would have. The message can sit lines below the banner,
            # so the tail stands in when the pattern catches nothing.
            fatal = re.search(r"FOAM FATAL[^\n]*\n+(.{0,300})", log, re.S)
            said = " ".join((fatal.group(1) if fatal else "").split())
            if not said:
                said = " ".join(log.strip().splitlines()[-3:])[:300]
            out["verdict"] = (f"checkMesh stopped: {said}" if said else
                              "checkMesh stopped and said nothing readable; read log.checkMesh")
        else:
            out["verdict"] = "no verdict in the checkMesh log"
        if stars:
            out["failing"] = [" ".join(s.split())[:160] for s in stars[:8]]
    for key, pattern in (("cells", r"\bcells:\s+(\d+)"), ("faces", r"\bfaces:\s+(\d+)"),
                         ("points", r"\bpoints:\s+(\d+)")):
        match = re.search(pattern, log)
        if match:
            out["counts"][key] = int(match.group(1))
    box = re.search(r"Overall domain bounding box \(([^)]*)\) \(([^)]*)\)", log)
    if box:
        try:
            out["bounds"] = [float(v) for v in (box.group(1) + " " + box.group(2)).split()]
        except ValueError:
            pass
    for key, pattern in (
        ("max_non_orthogonality", r"non-orthogonality Max:\s*([-\d.eE+]+)"),
        ("max_skewness", r"[Mm]ax skewness\s*=\s*([-\d.eE+]+)"),
        ("max_aspect_ratio", r"Max aspect ratio\s*=\s*([-\d.eE+]+)"),
        # The smallest cell in the mesh, which is what a Courant time step and a y+
        # estimate are actually sized on. checkMesh prints it as a volume.
        ("min_volume", r"Min volume\s*=\s*([-\d.eE+]+)"),
    ):
        match = re.search(pattern, log)
        if match:
            try:
                # checkMesh ends its sentences: `Min volume = 4.4444e-09. Max volume ...`
                # and the trailing full stop is inside the number's own character class,
                # so `float` refused it and the smallest cell went unmeasured -- which
                # is a time step and a y+ estimate quietly falling back to an average.
                out["metrics"][key] = float(match.group(1).rstrip("."))
            except ValueError:
                pass
    return out


# -- the picture --------------------------------------------------------------


def open_mesh(case: Path, region: str = ""):
    """(internal mesh, {patch name: surface}) from `constant/polyMesh`, or one region's.

    An empty `0/` is made when the case has no time directory at all: the reader wants
    one time to exist and a mesh-only case has none, which is the normal state of a
    case that has been meshed and not yet set up.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    times = [p for p in case.iterdir() if p.is_dir() and _is_time(p.name)]
    if not times:
        # The reader wants one time directory to exist and a mesh-only case has none.
        # Made here and taken away again in `look`, because looking at a case must not
        # change it: a `--dry-run` that leaves a `0/` behind is a side effect nobody
        # asked for, and an empty `0/` is a case that looks set up and is not.
        (case / "0").mkdir(exist_ok=True)
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    try:
        reader.enable_all_patch_arrays()
    except Exception:  # noqa: BLE001 - older pyvista names it differently; internal is enough
        pass
    block = reader.read()
    if region and region in block.keys():
        # A multi-region case comes back with one block per region, named for it. Asking
        # for the region by name is the whole of what `--region` does to the picture.
        block = block[region]
    internal = block["internalMesh"] if "internalMesh" in block.keys() else None
    patches: dict = {}
    if "boundary" in block.keys():
        boundary = block["boundary"]
        for name in boundary.keys():
            surface = boundary[name]
            if surface is not None and surface.n_cells:
                patches[str(name)] = surface
    return internal, patches


def _is_time(name: str) -> bool:
    try:
        float(name)
    except ValueError:
        return False
    return True


def inward_sign(internal, centre, normal, area: float) -> int:
    """+1 if the patch's normal points into the fluid, -1 if out of it, 0 if unknown.

    Asked of the mesh rather than reasoned about: step a little way off the patch
    along its normal, both ways, and see which point lands inside a cell. What this
    replaces is "point it at the middle of the bounding box", which is right for a
    straight duct and for a U-bend and wrong for anything concave enough that the
    middle of the box is not in the fluid -- a C-shaped passage, a spiral, one loop
    of a Tesla valve. Getting it wrong sets the inlet velocity backwards, and every
    other check in the chain passes happily.
    """
    import numpy as np

    if internal is None or not centre or not normal or area <= 0:
        return 0
    step = 0.25 * math.sqrt(area) if area > 0 else 0.0
    if step <= 0:
        return 0
    point = np.asarray(centre, dtype=float)
    direction = np.asarray(normal, dtype=float)
    try:
        forward = internal.find_containing_cell(point + step * direction)
        backward = internal.find_containing_cell(point - step * direction)
    except Exception:  # noqa: BLE001 - an answer of "unknown" is honest and cheap
        return 0
    forward = int(np.asarray(forward).ravel()[0])
    backward = int(np.asarray(backward).ravel()[0])
    if forward >= 0 and backward < 0:
        return 1
    if backward >= 0 and forward < 0:
        return -1
    return 0


def measure_patches(entries: list[dict], surfaces: dict, bounds=None,
                    internal=None) -> list[dict]:
    """The boundary table: the dictionary's names and counts, plus what the geometry says.

    Area, centre and mean normal come from the faces themselves. The normal is the one
    number that tells an inlet from an outlet without looking at where the patch sits,
    which is exactly the mistake that bounding-box classification makes.
    """
    import numpy as np

    out = []
    for entry in entries:
        row = dict(entry)
        surface = surfaces.get(entry["name"])
        if surface is not None and surface.n_cells:
            row["touches"] = box_contact(surface.points, bounds)
            try:
                sized = surface.compute_cell_sizes(length=False, area=True, volume=False)
                areas = np.asarray(sized.cell_data["Area"], dtype=float)
                row["area"] = float(areas.sum())
                centre = np.asarray(surface.cell_centers().points, dtype=float)
                weight = areas / areas.sum() if areas.sum() else None
                row["center"] = [float(v) for v in (
                    np.average(centre, axis=0, weights=weight) if weight is not None
                    else centre.mean(axis=0))]
                # Consistent ordering matters more here than it looks. Without it the
                # faces of a curved patch come back with their normals pointing
                # whichever way each cell happened to be wound, and the mean of a
                # cylinder's faces -- which must cancel to nothing, because the patch
                # closes on itself -- came out as a confident unit vector at 45
                # degrees, reported next to "flat". Oriented consistently, a closed
                # patch cancels and says so, and a flat one keeps its direction.
                # Consistent, but NOT auto-oriented: `auto_orient_normals` is defined
                # for a closed surface and a boundary patch is not one, so it flipped
                # an outlet to face the same way as the inlet -- two opposing flat
                # faces reported with the same normal, which is geometrically
                # impossible and was noticed and dismissed as cosmetic in a real run.
                # The sign is fixed below by asking the mesh which side the fluid is
                # on, which is a measurement rather than a convention.
                normals = surface.extract_surface().compute_normals(
                    cell_normals=True, point_normals=False, consistent_normals=True)
                vectors = np.asarray(normals.cell_data["Normals"], dtype=float)
                weight = areas / areas.sum() if areas.sum() and len(areas) == len(vectors) else None
                mean = (np.average(vectors, axis=0, weights=weight) if weight is not None
                        else vectors.mean(axis=0))
                norm = float(np.linalg.norm(mean))
                row["normal"] = [float(v) for v in (mean / norm)] if norm > 1e-9 else [0.0, 0.0, 0.0]
                row["flat"] = norm > 0.98
                if row["normal"] != [0.0, 0.0, 0.0]:
                    inward = inward_sign(internal, row["center"], row["normal"],
                                         float(row["area"]))
                    if inward:
                        # Reported outward, away from the fluid, whichever way the
                        # winding happened to run: the probe says which side the fluid
                        # is on, so the normal is signed against it.
                        row["normal"] = [-inward * v for v in row["normal"]]
                        row["inward"] = -1
                    else:
                        row["inward"] = 0
            except Exception as exc:  # noqa: BLE001 - a measurement missing beats a report missing
                row["measure_error"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
    return out


def as_box(bounds) -> list:
    """VTK's (xmin, xmax, ymin, ymax, zmin, zmax) as OpenFOAM's two corners.

    checkMesh prints its bounding box as (min) (max) and pyvista hands its own back
    interleaved, and mixing the two silently produced a domain 0.06 x -0.3 x 0.001 m
    -- a negative span nothing complained about. One convention, converted once: the
    two corners, which is what every reader of this file expects.
    """
    if not bounds or len(bounds) != 6:
        return []
    x0, x1, y0, y1, z0, z1 = (float(v) for v in bounds)
    return [x0, y0, z0, x1, y1, z1]


def box_contact(points, bounds) -> int:
    """How many of the domain's axes this patch reaches the end of.

    The measurement that separates a body in the flow from the wall of a passage, and
    it took three wrong answers to arrive at. A cylinder in a channel touches the
    bounding box on no axis; a square body sitting on the floor touches it on one (the
    floor); the wall of an L-duct runs into the ends and the sides and touches it on
    two. So: a wall patch reaching one axis or none is something in the flow, and one
    reaching two or more is the domain's own boundary.

    Axes the domain is only one cell thick in are not counted -- in a plane case every
    point in the mesh lies on both z faces, which made every patch look like the box.
    """
    import numpy as np

    if not bounds or len(bounds) != 6:
        return 0
    array = np.asarray(points, dtype=float)
    if not len(array):
        return 0
    lo = np.array(bounds[:3], dtype=float)
    hi = np.array(bounds[3:], dtype=float)
    span = hi - lo
    widest = float(span.max()) if span.size else 0.0
    tol = 1e-6 + 0.002 * widest
    reached = 0
    for axis in range(3):
        if span[axis] <= 0.05 * widest:
            continue
        column = array[:, axis]
        if (np.abs(column - lo[axis]) <= tol).any() or (np.abs(column - hi[axis]) <= tol).any():
            reached += 1
    return reached


def enclosure_patches(surfaces: dict, bounds, types: dict | None = None) -> set:
    """Which patches are the outside of the box rather than the thing inside it.

    A patch is part of the enclosure when, for some axis, every one of its points sits
    on that axis' minimum or maximum -- an inlet plane, a farfield, a floor, or the
    two z faces of a plane case together. A body immersed in the flow is on no such
    face, and neither is the wall of a passage that bends.

    The axis test is per-axis rather than per-point for a reason that cost a wrong
    answer: in a plane case every point in the mesh lies on one of the two z faces,
    so a "does this point touch the box anywhere" rule made a cylinder in mid-channel
    part of the enclosure. A direction the domain is only one cell thick in is not a
    direction anything can be classified by, so it is left out of the test entirely.

    It matters twice: an opaque flow box hides everything the picture was drawn to
    show, and a body in open flow carries a hundredth of the free-stream turbulence a
    passage does.
    """
    import numpy as np

    if not bounds or len(bounds) != 6:
        return set()
    lo = np.array(bounds[:3], dtype=float)
    hi = np.array(bounds[3:], dtype=float)
    span = hi - lo
    widest = float(span.max()) if span.size else 0.0
    tol = 1e-6 + 0.002 * widest
    # A direction the domain is barely thick in tells nothing apart.
    axes = [i for i in range(3) if span[i] > 0.05 * widest]
    out = set()
    for name, surface in surfaces.items():
        if types and types.get(name) == "empty":
            out.add(name)
            continue
        points = np.asarray(surface.points, dtype=float)
        if not len(points) or not axes:
            continue
        for axis in axes:
            column = points[:, axis]
            on_face = (np.abs(column - lo[axis]) <= tol) | (np.abs(column - hi[axis]) <= tol)
            if on_face.all():
                out.add(name)
                break
    return out


def draw(case: Path, internal, surfaces: dict, out_png: Path, two_d: bool,
         types: dict | None = None) -> str:
    """One PNG: the boundary by patch, and the cells inside it.

    2D gets two panels (the patches, and the cells looking down z); 3D gets four (the
    patches, and a cut on each axis through the middle). A panel that fails to draw
    leaves the others alone -- a partial picture is worth more than an exception where
    a picture was supposed to be.

    Two things the first version of this got wrong, both of which made a correct
    picture useless. In 2D the `empty` front and back are the whole area of the view
    and painted over every other patch, so a plane case showed one colour and nothing
    else; they are left out of the patch panel now (they are named in the table, and
    nobody has ever needed to see where the front of a one-cell-thick case is). In 3D
    the flow box did the same to the body inside it, so the enclosure is drawn faint
    and the body opaque.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    types = types or {}
    colours = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(sorted(surfaces))}
    domain = as_box(internal.bounds) if internal is not None else []
    enclosure = set() if two_d else enclosure_patches(surfaces, domain, types)
    if len(enclosure) == len(surfaces):
        enclosure = set()  # a bare box: there is nothing else to see past it
    shown = {name: s for name, s in surfaces.items()
             if not (two_d and types.get(name) == "empty")}
    shape = (1, 2) if two_d else (2, 2)
    size = (1600, 760) if two_d else (1500, 1150)
    plotter = pv.Plotter(shape=shape, off_screen=True, window_size=size, border=False)

    def frame(view: str) -> None:
        """Fill the panel with the mesh rather than with white space."""
        try:
            plotter.camera.tight(padding=0.06, view=view)
        except Exception:  # noqa: BLE001 - older pyvista: aim and zoom by hand
            plotter.camera_position = view
            plotter.reset_camera()
            plotter.camera.zoom(1.3)

    def patches_panel(title: str) -> None:
        plotter.add_text(title, font_size=9)
        if two_d and internal is not None:
            plotter.add_mesh(internal.extract_surface(), color="#ececec",
                             show_edges=False, lighting=False)
        for name, surface in shown.items():
            faint = name in enclosure
            if two_d:
                # Looking down z, a side patch is a quad seen exactly edge-on: as a
                # filled surface it is zero pixels wide and the panel comes back blank.
                # Drawn as a thick wireframe it is the coloured line it should be.
                plotter.add_mesh(surface, color=colours[name], style="wireframe",
                                 line_width=7, render_lines_as_tubes=True,
                                 lighting=False, label=name)
            else:
                plotter.add_mesh(surface, color=colours[name],
                                 opacity=0.10 if faint else 1.0, show_edges=False,
                                 label=name)
        if shown:
            try:
                plotter.add_legend(bcolor="white", size=(0.26, 0.045 * max(2, len(shown))),
                                   loc="upper right", face="rectangle")
            except Exception:  # noqa: BLE001 - the legend is a nicety
                pass
        frame("xy" if two_d else "yz")
        if not two_d:
            plotter.camera_position = "iso"
            plotter.camera.zoom(1.2)

    def cut_panel(normal: str, title: str) -> None:
        plotter.add_text(title, font_size=9)
        if internal is None:
            return
        try:
            cut = internal.slice(normal=normal)
            plotter.add_mesh(cut, color="#dddddd", show_edges=True, edge_color="#3b3b3b",
                             line_width=1)
        except Exception:  # noqa: BLE001
            plotter.add_mesh(internal.extract_surface(), color="#dddddd", show_edges=True)
        frame({"x": "yz", "y": "zx", "z": "xy"}[normal])

    if two_d:
        plotter.subplot(0, 0)
        patches_panel("boundary patches (looking down z; front and back left out)")
        plotter.subplot(0, 1)
        cut_panel("z", "cells")
    else:
        plotter.subplot(0, 0)
        patches_panel("boundary patches" + (" (the enclosure faint)" if enclosure else ""))
        for index, normal in enumerate("xyz"):
            plotter.subplot((index + 1) // 2, (index + 1) % 2)
            cut_panel(normal, f"cells, cut through {normal}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_png))
    plotter.close()
    return str(out_png)


# -- several views, for a reviewer who did not build it -----------------------

VIEWS_2D = ("overview.png", "cells.png", "detail.png")
VIEWS_3D = ("iso.png", "ortho.png", "cuts.png")
"""What `draw_views` writes, by name, so a reader knows what to expect before it runs.

Three composite pictures either way, and no more: the reviewer that reads them pays
for every pixel, and three at about a thousand pixels across is the budget it has."""

# Looking at the mesh from one side: pyvista's `tight` names a view by the axis that
# points right and the axis that points up, and `negative` puts the camera on the far
# side. `+x` is the camera on the +x side looking back toward -x, and so on.
_AXIS_VIEWS = {
    "+x": ("yz", False, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "-x": ("yz", True, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "+y": ("xz", True, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "-y": ("xz", False, (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "+z": ("xy", False, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "-z": ("xy", True, (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}

_ISO_CORNERS = (
    ((1.0, 1.0, 1.0), "from +x +y +z"),
    ((-1.0, 1.0, 1.0), "from -x +y +z"),
    ((1.0, -1.0, 1.0), "from +x -y +z"),
    ((-1.0, -1.0, -1.0), "from -x -y -z (underneath)"),
)


def draw_views(case: Path, internal, surfaces: dict, out_dir: Path, two_d: bool,
               types: dict | None = None) -> list[str]:
    """Three composite PNGs into `out_dir`, for someone judging the shape who was not
    there when it was built. Returns the paths written, as strings.

    `draw()` is one picture for the person who made the mesh and knows what they are
    looking at. These are for a reviewer who does not: the same shape from enough
    sides that a missing feature, a polygonal corner on what should be a curve, or an
    inlet on the wrong end is visible in at least one of them.

    2D (`two_d`): `overview.png` -- the patches coloured, looking down z, the `empty`
    front and back left out as `draw()` leaves them out; `cells.png` -- every cell with
    its edges, looking down z; `detail.png` -- a 2x2 grid of that cell view, one
    quadrant of the domain's bounding box per panel, which is where a curve built from
    six straight segments stops looking like a curve.

    3D: `iso.png` -- 2x2, the patches from four corners, the enclosure faint at the
    opacity `draw()` uses so the body inside it is what shows; `ortho.png` -- 2x3, the
    patches from +x, -x, +y, -y, +z, -z with parallel projection, so proportions can be
    read off; `cuts.png` -- 2x2, the cells on the mid-plane through x, y and z, and one
    more cut a quarter of the way along the longest axis, which is where a passage that
    the mid-plane happens to miss shows up.

    Tolerance is `draw()`'s: a panel that fails leaves the others alone, and a whole
    picture that fails is said on stderr and leaves the other two alone. The caller
    compares what came back against `VIEWS_2D` / `VIEWS_3D` to know what is missing.
    """
    import numpy as np
    import pyvista as pv

    pv.OFF_SCREEN = True
    types = types or {}
    colours = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(sorted(surfaces))}
    domain = as_box(internal.bounds) if internal is not None else []
    enclosure = set() if two_d else enclosure_patches(surfaces, domain, types)
    if len(enclosure) == len(surfaces):
        enclosure = set()  # a bare box: there is nothing else to see past it
    shown = {name: s for name, s in surfaces.items()
             if not (two_d and types.get(name) == "empty")}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def aim(plotter, side: str, padding: float = 0.06) -> None:
        """Axis-aligned framing that fills the panel; never resizes the window, which
        `camera.tight` does by default and which a grid of panels cannot survive."""
        view, negative, direction, up = _AXIS_VIEWS[side]
        try:
            plotter.camera.tight(padding=padding, adjust_render_window=False,
                                 view=view, negative=negative)
        except Exception:  # noqa: BLE001 - older pyvista: aim and zoom by hand
            aim_from(plotter, direction, up, zoom=1.3)
            try:
                plotter.enable_parallel_projection()
            except Exception:  # noqa: BLE001
                pass

    def aim_from(plotter, direction, up, zoom: float = 1.05) -> None:
        """Look at whatever is in the panel from `direction`, and fit it."""
        x0, x1, y0, y1, z0, z1 = plotter.renderer.bounds
        focal = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        radius = max(x1 - x0, y1 - y0, z1 - z0, 1e-9) * 3.0
        vector = np.asarray(direction, dtype=float)
        vector = vector / (np.linalg.norm(vector) or 1.0)
        plotter.camera_position = [tuple(focal + radius * vector), tuple(focal), tuple(up)]
        plotter.reset_camera()
        plotter.camera.zoom(zoom)

    def add_patches(plotter, legend: bool) -> None:
        if two_d and internal is not None:
            plotter.add_mesh(internal.extract_surface(), color="#ececec",
                             show_edges=False, lighting=False)
        for name, surface in shown.items():
            if two_d:
                # Edge-on looking down z: a filled quad is zero pixels wide, a thick
                # line is the coloured outline it should be (see `draw`).
                plotter.add_mesh(surface, color=colours[name], style="wireframe",
                                 line_width=7, render_lines_as_tubes=True,
                                 lighting=False, label=name)
            else:
                plotter.add_mesh(surface, color=colours[name],
                                 opacity=0.10 if name in enclosure else 1.0,
                                 show_edges=False, label=name)
        if legend and shown:
            # The patch names in their own colours, stacked top-right. Not `add_legend`:
            # VTK's legend box scales its text to the box and on this stack it comes
            # out cut off at the edge of the panel, which for a reviewer telling the
            # inlet from the outlet by colour is the one thing that cannot be cut off.
            for index, name in enumerate(sorted(shown)):
                try:
                    plotter.add_text(name, position=(0.66, 0.90 - 0.05 * index),
                                     viewport=True, font_size=10, color=colours[name],
                                     shadow=False, name=f"legend_{index}")
                except Exception:  # noqa: BLE001 - the legend is a nicety
                    break

    def add_cut(plotter, normal: str, origin=None) -> None:
        if internal is None:
            return
        try:
            cut = internal.slice(normal=normal, origin=origin)
            plotter.add_mesh(cut, color="#dddddd", show_edges=True, edge_color="#3b3b3b",
                             line_width=1)
        except Exception:  # noqa: BLE001
            plotter.add_mesh(internal.extract_surface(), color="#dddddd", show_edges=True)

    def composite(name: str, shape: tuple, size: tuple, panels: list) -> None:
        """One PNG of `panels`, each `(row, col, title, fill)`; a panel that fails is
        left blank under its title, and a picture that fails is said and skipped."""
        try:
            plotter = pv.Plotter(shape=shape, off_screen=True, window_size=size,
                                 border=len(panels) > 1, border_color="#c8c8c8")
        except Exception as exc:  # noqa: BLE001
            print(f"mesh_look: {name} could not be drawn ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            return
        try:
            for row, col, title, fill in panels:
                plotter.subplot(row, col)
                plotter.add_text(title, font_size=9)
                try:
                    fill(plotter)
                except Exception:  # noqa: BLE001 - a partial picture beats no picture
                    continue
            target = out_dir / name
            plotter.screenshot(str(target))
            written.append(str(target))
        except Exception as exc:  # noqa: BLE001 - the other pictures are still worth having
            print(f"mesh_look: {name} could not be drawn ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
        finally:
            try:
                plotter.close()
            except Exception:  # noqa: BLE001
                pass

    if two_d:
        def overview(plotter) -> None:
            add_patches(plotter, legend=True)
            aim(plotter, "+z")

        def cells(plotter) -> None:
            add_cut(plotter, "z")
            aim(plotter, "+z")

        def quadrant(x_lo: bool, y_lo: bool):
            def within(dataset):
                """The cells of `dataset` whose centres are in this quadrant -- whole
                cells, because a cell cut in half looks like a mesh that is wrong."""
                x0, y0, _z0, x1, y1, _z1 = domain
                xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
                centres = np.asarray(dataset.cell_centers().points, dtype=float)
                if not len(centres):
                    return None
                keep = (((centres[:, 0] <= xm) if x_lo else (centres[:, 0] >= xm)) &
                        ((centres[:, 1] <= ym) if y_lo else (centres[:, 1] >= ym)))
                return dataset.extract_cells(np.flatnonzero(keep)) if keep.any() else None

            def fill(plotter) -> None:
                if internal is None or len(domain) != 6:
                    return
                part = within(internal.slice(normal="z"))
                if part is not None:
                    plotter.add_mesh(part, color="#dddddd", show_edges=True,
                                     edge_color="#3b3b3b", line_width=1)
                for name, surface in shown.items():
                    # The patch outline in its colour, so the corner being looked at
                    # is known to be a wall and not the inlet. Clipped to the quadrant
                    # too, or the framing would take in the whole domain again.
                    edge = within(surface)
                    if edge is not None and edge.n_cells:
                        plotter.add_mesh(edge, color=colours[name], style="wireframe",
                                         line_width=4, lighting=False)
                aim(plotter, "+z", padding=0.03)
            return fill

        composite("overview.png", (1, 1), (1500, 1000), [
            (0, 0, "boundary patches (looking down z; front and back left out)", overview)])
        composite("cells.png", (1, 1), (1500, 1000), [(0, 0, "cells (looking down z)", cells)])
        composite("detail.png", (2, 2), (1500, 1250), [
            (0, 0, "detail: upper-left quadrant (low x, high y)", quadrant(True, False)),
            (0, 1, "detail: upper-right quadrant (high x, high y)", quadrant(False, False)),
            (1, 0, "detail: lower-left quadrant (low x, low y)", quadrant(True, True)),
            (1, 1, "detail: lower-right quadrant (high x, low y)", quadrant(False, True)),
        ])
    else:
        def corner(direction, legend: bool):
            def fill(plotter) -> None:
                add_patches(plotter, legend=legend)
                aim_from(plotter, direction, (0.0, 0.0, 1.0))
            return fill

        def side(name: str):
            def fill(plotter) -> None:
                add_patches(plotter, legend=False)
                aim(plotter, name)
            return fill

        def cut(normal: str, origin=None, side_of: str = ""):
            def fill(plotter) -> None:
                add_cut(plotter, normal, origin)
                aim(plotter, side_of or {"x": "+x", "y": "-y", "z": "+z"}[normal])
            return fill

        faint = " (the enclosure faint)" if enclosure else ""
        composite("iso.png", (2, 2), (1500, 1300), [
            (index // 2, index % 2, f"boundary patches {label}{faint}", corner(vector, index == 0))
            for index, (vector, label) in enumerate(_ISO_CORNERS)])
        sides = ("+x", "-x", "+y", "-y", "+z", "-z")
        composite("ortho.png", (2, 3), (1600, 1000), [
            (index // 3, index % 3, f"patches seen from {name} (orthographic){faint}", side(name))
            for index, name in enumerate(sides)])
        panels = [(index // 2, index % 2, f"cells, cut through {axis} (mid-plane)", cut(axis))
                  for index, axis in enumerate("xyz")]
        if len(domain) == 6:
            span = [domain[3] - domain[0], domain[4] - domain[1], domain[5] - domain[2]]
            longest = int(np.argmax(span))
            axis = "xyz"[longest]
            origin = [(domain[i] + domain[i + 3]) / 2 for i in range(3)]
            origin[longest] = domain[longest] + 0.25 * span[longest]
            panels.append((1, 1, f"cells, cut through {axis} at 25% along ({axis} = {origin[longest]:.4g})",
                           cut(axis, origin)))
        composite("cuts.png", (2, 2), (1500, 1300), panels)
    return written


# -- the report ---------------------------------------------------------------


def report(payload: dict) -> str:
    lines = [f"{payload['case']}"]
    if not payload.get("polymesh"):
        lines.append("  no constant/polyMesh -- nothing has been meshed here yet")
        return "\n".join(lines)
    counts = (f"  {payload.get('cells', 0):,} cells, {payload.get('faces', 0):,} faces, "
              f"{payload.get('points', 0):,} points")
    if payload.get("two_d"):
        counts += "  (one cell thick: a plane case)"
    lines.append(counts)
    bounds = payload.get("bounds") or []
    if len(bounds) == 6:
        x0, y0, z0, x1, y1, z1 = bounds
        span = (x1 - x0, y1 - y0, z1 - z0)
        line = (f"  bounds {span[0]:.4g} x {span[1]:.4g} x {span[2]:.4g} m"
                f"   x {x0:.4g}..{x1:.4g}   y {y0:.4g}..{y1:.4g}   z {z0:.4g}..{z1:.4g}")
        # In millimetres too, when it is small enough that the request was probably
        # written in them. OpenFOAM has no units and reads these numbers as metres, so
        # a geometry built in millimetres and never scaled is a mesh a thousand times
        # too big with nothing anywhere saying so.
        if 0 < max(span) < 10:
            line += f"   [{span[0] * 1000:.4g} x {span[1] * 1000:.4g} x {span[2] * 1000:.4g} mm]"
        lines.append(line)
    lines.append(f"  checkMesh: {payload.get('checkmesh', '(not run)')}")
    for key, value in sorted((payload.get("metrics") or {}).items()):
        lines.append(f"    {key.replace('_', ' ')}: {value:g}")
    for failing in payload.get("failing") or []:
        lines.append(f"    {failing}")
    if payload.get("patches"):
        lines.append("  patches:")
        width = max(len(p["name"]) for p in payload["patches"])
        for patch in payload["patches"]:
            row = (f"    {patch['name']:<{width}}  {patch.get('type', ''):<12}"
                   f"{patch.get('nFaces', 0):>9,} faces")
            if patch.get("area") is not None:
                row += f"  area {patch['area']:.4g} m2"
            if patch.get("center"):
                c = patch["center"]
                row += f"  at ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})"
            if patch.get("normal") and any(patch["normal"]):
                n = patch["normal"]
                row += f"  faces ({n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f})"
                if not patch.get("flat", False):
                    row += " (curved)"
            lines.append(row)
    zones = payload.get("zones")
    if zones:
        lines.append("  zones:")
        width = max(len(z["name"]) for z in zones)
        for zone in zones:
            row = (f"    {zone['name']:<{width}}  {zone.get('kind', ''):<5} zone"
                   f"{zone.get('count', 0):>9,} {'cells' if zone.get('kind') == 'cell' else 'faces'}")
            if zone.get("centre"):
                c = zone["centre"]
                row += f"  centroid ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})"
            bounds = zone.get("bounds") or []
            if len(bounds) == 6:
                row += (f"  cell centres span x {bounds[0]:.4g}..{bounds[3]:.4g}"
                        f"  y {bounds[1]:.4g}..{bounds[4]:.4g}"
                        f"  z {bounds[2]:.4g}..{bounds[5]:.4g}")
            if zone.get("unmeasured"):
                row += f"  (extents not measured: {zone['unmeasured']})"
            lines.append(row)
    elif payload.get("polymesh"):
        # Said out loud, because "no zones" and "zones were not looked for" are
        # different facts and only the first one means an AMI or MRF dictionary has
        # nothing to name yet.
        lines.append("  zones: none -- constant/polyMesh has no cellZones or faceZones")
    if payload.get("build"):
        lines.append("  rebuilds with: " + ", ".join(payload["build"]))
    else:
        lines.append("  no Allmesh or build script in this case")
    if payload.get("render"):
        # The absolute path, always: the report is read by whoever ran the script,
        # from wherever they ran it, and a path relative to the case read as one
        # relative to the caller's directory cost a 404 and three turns (c7ef).
        lines.append(f"  picture: {payload.get('render_abs') or payload['render']}")
    if payload.get("views"):
        lines.append("  views: " + ", ".join(payload["views"]))
    if payload.get("error"):
        lines.append(f"  {payload['error']}")
    return "\n".join(lines)


def look(case: Path, out_png: Path | None, check: bool = True, region: str = "",
         views: Path | None = None) -> dict:
    """Everything this script knows about the mesh, as one dictionary.

    `region` selects `constant/<region>/polyMesh`. The dictionary's keys do not change
    with it: `case_gen.py` and the hosted Mesh panel both read this shape, so a region
    is a different payload rather than a different payload shape.

    `views` is a directory: when given, `draw_views` writes its three composite
    pictures there and the payload carries `"views": [paths]`, the paths as written.
    Without it the key is absent and the payload is exactly what it always was.
    """
    case = case.resolve()
    entries = boundary_entries(case, region)
    payload: dict = {
        "case": str(case),
        "polymesh": (mesh_dir(case, region) / "points").exists(),
        "patches": entries,
        # Additive: `mesher/check.py` and `case_gen.py` read this payload by key and
        # ignore what they do not know, so an older reader of a newer mesh_look keeps
        # working and inherits the zone facts the day it asks for them.
        "zones": [],
        "build": build_files(case),
        "cells": 0, "faces": 0, "points": 0, "bounds": [], "two_d": False,
        "checkmesh": "", "checkmesh_ok": False, "metrics": {}, "render": "",
    }
    if not payload["polymesh"]:
        return payload

    payload["zones"] = zone_entries(case)
    measure_cell_zones(case, payload["zones"])

    if check:
        verdict = run_check(case, region)
        payload["checkmesh"] = verdict.get("verdict", "")
        payload["checkmesh_ok"] = bool(verdict.get("ok"))
        payload["metrics"] = verdict.get("metrics") or {}
        if verdict.get("failing"):
            payload["failing"] = verdict["failing"]
        payload.update({k: v for k, v in (verdict.get("counts") or {}).items()})
        if verdict.get("bounds"):
            payload["bounds"] = verdict["bounds"]

    payload["two_d"] = any(e.get("type") == "empty" for e in entries)
    internal, surfaces = None, {}
    made_time = not any(p.is_dir() and _is_time(p.name) for p in case.iterdir())
    try:
        internal, surfaces = open_mesh(case, region)
    except Exception as exc:  # noqa: BLE001 - the numbers survive a reader that will not open
        payload["error"] = f"the mesh could not be opened for drawing ({type(exc).__name__}: {exc})"
    finally:
        if made_time:
            try:
                (case / "0").rmdir()  # only ever the empty one this call made
            except OSError:
                pass
    if internal is not None:
        payload["cells"] = payload.get("cells") or int(internal.n_cells)
        payload["points"] = payload.get("points") or int(internal.n_points)
        if not payload["bounds"]:
            payload["bounds"] = as_box(internal.bounds)
    if surfaces:
        payload["patches"] = measure_patches(entries, surfaces, payload["bounds"], internal)
        # Which patches are the walls of the box rather than something inside it.
        # The picture uses it to draw the enclosure faint; `case_gen.py` uses it to
        # tell a body in open flow from a passage, which sets the free-stream
        # turbulence. Measured once, here, while the surfaces are open.
        payload["enclosure"] = sorted(enclosure_patches(
            surfaces, payload["bounds"], {e["name"]: e.get("type", "") for e in entries}))
    if out_png is not None and surfaces:
        try:
            payload["render"] = draw(case, internal, surfaces, out_png, payload["two_d"],
                                     {e["name"]: e.get("type", "") for e in entries})
            # Additive, like `zones`: the absolute path beside the one `render` has
            # always carried, so a reader that wants no arithmetic has none to do and
            # a reader that joins `render` to the case keeps working.
            payload["render_abs"] = str(Path(payload["render"]).resolve())
        except Exception as exc:  # noqa: BLE001
            payload["error"] = f"the picture could not be drawn ({type(exc).__name__}: {exc})"
    if views is not None:
        payload["views"] = []
        if surfaces or internal is not None:
            expected = VIEWS_2D if payload["two_d"] else VIEWS_3D
            try:
                payload["views"] = draw_views(case, internal, surfaces, views, payload["two_d"],
                                              {e["name"]: e.get("type", "") for e in entries})
            except Exception as exc:  # noqa: BLE001
                _add_error(payload, f"the views could not be drawn ({type(exc).__name__}: {exc})")
            missing = [name for name in expected
                       if not any(Path(p).name == name for p in payload["views"])]
            if missing:
                _add_error(payload, f"{len(missing)} of the {len(expected)} views could not "
                                    f"be drawn: {', '.join(missing)}")
    return payload


def _add_error(payload: dict, text: str) -> None:
    """One more thing that went wrong, kept beside what already did rather than over it."""
    payload["error"] = f"{payload['error']}; {text}" if payload.get("error") else text


def from_cwd(path: Path) -> Path:
    """A path the caller typed, made absolute the way the shell would read it: against
    the working directory, never against the case.

    Every other toolbox script that takes `--out` (`render.py`, `results.py`,
    `showcase.py`, `geometry_view.py`, `animate.py`) hands a relative one straight to
    `Path`, which is this. This one alone joined it to the case, and a caller who typed
    `mesh_look.py mesh --out look.png` from a study directory found no `look.png` where
    they stood (study 20260920-161908-c7ef: `read_file .../look.png -> 404`, three
    turns to recover). One convention across the toolbox, and this is the one."""
    return path if path.is_absolute() else Path.cwd() / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--out", type=Path, default=None,
                        help="where the picture goes; a relative path is relative to where "
                             "you run this from, not to the case (default <case>/look.png). "
                             "The report prints the absolute path written.")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the machine-readable report here (relative to where "
                             "you run this from)")
    parser.add_argument("--no-check", action="store_true", help="skip checkMesh")
    parser.add_argument("--region", default="",
                        help="read constant/<region>/polyMesh instead (a CHT case has one "
                             "mesh per region; this looks at one of them)")
    parser.add_argument("--views", type=Path, default=None,
                        help="also write three composite pictures of the mesh from several "
                             "sides into this directory (2D: overview, cells, detail; 3D: "
                             "iso, ortho, cuts); relative to where you run this from, like "
                             "--out. The JSON lists them under `views`.")
    args = parser.parse_args()

    case = from_cwd(args.case)
    out = from_cwd(args.out) if args.out is not None else case / "look.png"
    views = from_cwd(args.views) if args.views is not None else None
    payload = look(case, out, check=not args.no_check, region=args.region, views=views)
    if payload.get("render"):
        # `render` stays relative to the case when the picture is inside it -- that is
        # what `mesher/check.py` has always read and joins to the case's two paths --
        # and absolute when it is not. `render_abs` is absolute either way, and the
        # printed report uses it.
        try:
            payload["render"] = str(Path(payload["render"]).relative_to(case.resolve()))
        except ValueError:
            pass
    if payload.get("views"):
        # The same rule as `render`: relative to the case when inside it, absolute when not.
        relative: list[str] = []
        for written in payload["views"]:
            try:
                relative.append(str(Path(written).relative_to(case.resolve())))
            except ValueError:
                relative.append(written)
        payload["views"] = relative
    if args.json is not None:
        target = from_cwd(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(report(payload))
    sys.exit(0 if payload.get("polymesh") else 1)


if __name__ == "__main__":
    main()
