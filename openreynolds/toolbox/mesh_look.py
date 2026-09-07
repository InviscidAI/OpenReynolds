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
    ):
        match = re.search(pattern, log)
        if match:
            try:
                out["metrics"][key] = float(match.group(1))
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


def measure_patches(entries: list[dict], surfaces: dict) -> list[dict]:
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
            try:
                sized = surface.compute_cell_sizes(length=False, area=True, volume=False)
                areas = np.asarray(sized.cell_data["Area"], dtype=float)
                row["area"] = float(areas.sum())
                centre = np.asarray(surface.cell_centers().points, dtype=float)
                weight = areas / areas.sum() if areas.sum() else None
                row["center"] = [float(v) for v in (
                    np.average(centre, axis=0, weights=weight) if weight is not None
                    else centre.mean(axis=0))]
                normals = surface.extract_surface().compute_normals(
                    cell_normals=True, point_normals=False, consistent_normals=False)
                vectors = np.asarray(normals.cell_data["Normals"], dtype=float)
                mean = vectors.mean(axis=0)
                norm = float(np.linalg.norm(mean))
                row["normal"] = [float(v) for v in (mean / norm)] if norm > 1e-9 else [0.0, 0.0, 0.0]
                row["flat"] = norm > 0.98
            except Exception as exc:  # noqa: BLE001 - a measurement missing beats a report missing
                row["measure_error"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
    return out


def enclosure_patches(surfaces: dict, bounds) -> set:
    """Which patches are the outside of the box rather than the thing inside it.

    A patch every one of whose points lies on the domain's bounding box is a wall of
    the enclosure -- an inlet plane, a farfield, a floor. A body immersed in the flow
    is not, and neither is the wall of a passage that bends. It matters because an
    opaque flow box hides everything the picture was drawn to show, and "make the big
    ones transparent" mislabels a wide floor. This is measured, not guessed.
    """
    import numpy as np

    if not bounds or len(bounds) != 6:
        return set()
    lo = np.array(bounds[:3], dtype=float)
    hi = np.array(bounds[3:], dtype=float)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    tol = 1e-6 + 0.002 * float(span.max())
    out = set()
    for name, surface in surfaces.items():
        points = np.asarray(surface.points, dtype=float)
        if not len(points):
            continue
        on_face = ((np.abs(points - lo) <= tol) | (np.abs(points - hi) <= tol)).any(axis=1)
        if on_face.all():
            out.add(name)
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
    domain = list(internal.bounds) if internal is not None else []
    enclosure = set() if two_d else enclosure_patches(surfaces, domain)
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
        lines.append(f"  bounds {x1 - x0:.4g} x {y1 - y0:.4g} x {z1 - z0:.4g} m"
                     f"   x {x0:.4g}..{x1:.4g}   y {y0:.4g}..{y1:.4g}   z {z0:.4g}..{z1:.4g}")
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
    try:
        internal, surfaces = open_mesh(case)
    except Exception as exc:  # noqa: BLE001 - the numbers survive a reader that will not open
        payload["error"] = f"the mesh could not be opened for drawing ({type(exc).__name__}: {exc})"
    if internal is not None:
        payload["cells"] = payload.get("cells") or int(internal.n_cells)
        payload["points"] = payload.get("points") or int(internal.n_points)
        if not payload["bounds"]:
            payload["bounds"] = [float(v) for v in internal.bounds]
    if surfaces:
        payload["patches"] = measure_patches(entries, surfaces)
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
