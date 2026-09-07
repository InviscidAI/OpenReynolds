#!/usr/bin/env python3
"""Look at a mesh and measure it: one picture, one table, one verdict.

    python3 mesh_look.py /work/study/mesh                      # picture + report
    python3 mesh_look.py . --out look.png --json look.json     # and the machine-readable one
    python3 mesh_look.py . --no-check                          # skip checkMesh (faster)

Every panel is drawn from `constant/polyMesh` -- no solve, no fields, no time
directory needed. The boundary is coloured **one colour per patch**, because the
question that gets answered wrong most often is not "is this the right shape" but "is
the inlet the end I think it is": a picture where the inlet patch is a different colour
from the outlet answers it in one look, and the table under it gives each patch's area,
its centre and the direction it faces, which answers it in numbers.

What it reports:

* cells, faces, points, the bounding box, and whether the mesh is one cell thick (a
  plane case) or a volume;
* every patch: type, face count, area, centre, mean unit normal;
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


def boundary_entries(case: Path) -> list[dict]:
    """Patch name, type and face count straight out of `constant/polyMesh/boundary`.

    Read as text on purpose: it works with no OpenFOAM environment, no reader and no
    time directory, so a mesh that nothing else can open still reports its patches.
    """
    path = case / "constant" / "polyMesh" / "boundary"
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


def build_files(case: Path) -> list[str]:
    """What in this case would rebuild the mesh from nothing."""
    names = ["Allmesh", "Allrun", "build.py", "geometry.py", "mesh.py",
             "system/blockMeshDict", "system/snappyHexMeshDict", "system/meshDict"]
    found = [n for n in names if (case / n).exists()]
    found += sorted(p.name for p in case.glob("*.geo"))
    return found


# -- checkMesh ----------------------------------------------------------------


def run_check(case: Path) -> dict:
    """`checkMesh`, its verdict, its counts and the three metrics that matter.

    The log is kept at `log.checkMesh` whatever happens: the summary here is the news,
    and the failing checks in full are what somebody reads next.
    """
    out = {"ok": False, "verdict": "", "metrics": {}, "counts": {}, "bounds": []}
    try:
        proc = subprocess.run(["checkMesh", "-case", str(case)], cwd=str(case),
                              capture_output=True, text=True, timeout=CHECK_TIMEOUT_S)
        log = proc.stdout + proc.stderr
    except FileNotFoundError:
        out["verdict"] = "checkMesh is not on PATH here"
        return out
    except subprocess.TimeoutExpired:
        out["verdict"] = f"checkMesh did not finish in {CHECK_TIMEOUT_S} s"
        return out
    (case / "log.checkMesh").write_text(log, errors="replace")
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
            fatal = re.search(r"--> FOAM FATAL ERROR:?\s*\n?(.{0,200})", log, re.S)
            out["verdict"] = "checkMesh stopped: " + " ".join((fatal.group(1) if fatal else "").split())
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


def open_mesh(case: Path):
    """(internal mesh, {patch name: surface}) from `constant/polyMesh`.

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
    if payload.get("build"):
        lines.append("  rebuilds with: " + ", ".join(payload["build"]))
    else:
        lines.append("  no Allmesh or build script in this case")
    if payload.get("render"):
        lines.append(f"  picture: {payload['render']}")
    if payload.get("error"):
        lines.append(f"  {payload['error']}")
    return "\n".join(lines)


def look(case: Path, out_png: Path | None, check: bool = True) -> dict:
    """Everything this script knows about the mesh, as one dictionary."""
    case = case.resolve()
    entries = boundary_entries(case)
    payload: dict = {
        "case": str(case),
        "polymesh": (case / "constant" / "polyMesh" / "points").exists(),
        "patches": entries,
        "build": build_files(case),
        "cells": 0, "faces": 0, "points": 0, "bounds": [], "two_d": False,
        "checkmesh": "", "checkmesh_ok": False, "metrics": {}, "render": "",
    }
    if not payload["polymesh"]:
        return payload

    if check:
        verdict = run_check(case)
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
        internal, surfaces = open_mesh(case)
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
        except Exception as exc:  # noqa: BLE001
            payload["error"] = f"the picture could not be drawn ({type(exc).__name__}: {exc})"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("look.png"),
                        help="where the picture goes (default look.png in the case)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the machine-readable report here")
    parser.add_argument("--no-check", action="store_true", help="skip checkMesh")
    args = parser.parse_args()

    case = args.case if args.case.is_absolute() else Path.cwd() / args.case
    out = args.out if args.out.is_absolute() else case / args.out
    payload = look(case, out, check=not args.no_check)
    if payload.get("render"):
        try:
            payload["render"] = str(Path(payload["render"]).relative_to(case.resolve()))
        except ValueError:
            pass
    if args.json is not None:
        target = args.json if args.json.is_absolute() else case / args.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(report(payload))
    sys.exit(0 if payload.get("polymesh") else 1)


if __name__ == "__main__":
    main()
