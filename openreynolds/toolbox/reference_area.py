#!/usr/bin/env python3
"""How much of the body this mesh models, and the area its forces divide by.

R01, the DARPA SUBOFF hull, 2026-08-29 (F-43): a half model on a symmetry plane,
OpenFOAM's `forces` reported 43.74 N on that half, and the agent divided it by the
wetted area of the WHOLE hull. The solve was 0.09% off the towing tank (87.48 N
doubled, 87.4 N published); the delivered Ct, 0.00158, was half of it. It then
computed the ITTC-57 friction line, 0.00291 -- above its own total, which no body
with any form drag can do -- and explained the contradiction away.

The fix went into `snappy_gen.py`, which halved the area in the same function that
cut the mesh, and printed the friction floor on every case. `snappy_gen.py` was
deleted with the rest of the geometry stack on 2026-09-07 (`b6ac498`) and the fix
went with it, while meshes started arriving from the CAD desk, cfMesh and hand-written
blockMesh -- none of which says whether it cut the body. So this reads that off the
mesh itself, whoever made it:

* **A symmetry patch that shares points with the body patch cuts the body.** Each
  such plane doubles the body's force when mirrored back. A symmetry plane the body
  does not reach -- a far-field side, a floor below a car -- cuts nothing and counts
  for nothing. The count is printed with the plane and the number of shared points,
  so a reading that is wrong can be seen to be. The one case the mesh cannot tell
  apart is a double-body waterline, where the plane closes a hull whose upper half
  never existed: `--not-mirror <patch>` says so.

* **The areas are measured off the body patch as meshed**, so on a half model they
  are half areas. A force on the meshed part divided by the meshed part's area is
  the full body's coefficient, with no factor to remember; the full body's force
  and area are the meshed ones times the mirror factor, and both are printed.
  Wetted area is the face-area sum; frontal area is the silhouette along the drag
  direction, rasterised -- `0.5 * sum(A |n.d|)` overcounts anything with a second
  part behind the first.

* **A total drag coefficient below flat-plate friction is impossible**, not tight,
  whenever the area it is divided by is no larger than the wetted area -- which
  frontal and wetted both are. The floor is ITTC-57 for a turbulent case and
  Blasius, 1.328/sqrt(Re), for a laminar one, which is the lower of the two lines.

`results.py` and `case_gen.py` import this; run on its own it prints the reading,
and `--cd X --re R` checks a coefficient against the floor, exiting 1 when it is
below it, because that is arithmetic rather than judgement. The mesh must be ascii,
like `layer_report.py`, whose reader this uses.

    python3 reference_area.py <case> --body hull
    python3 reference_area.py <case> --body hull --drag-dir 1 0 0 --json
    python3 reference_area.py <case> --body hull --cd 0.00158 --re 1.2e7
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_report  # noqa: E402  (sibling script, not a package)

MeshError = layer_report.MeshError

SYMMETRY_TYPES = ("symmetry", "symmetryPlane")

PLANAR_TOL = 1e-3
"""A symmetry patch whose faces stray from one plane by more than this fraction of
the domain is not a mirror anything can be reflected in."""

MIN_SHARED_POINTS = 3
"""Fewer points in common than this is a body touching a plane, not cut by it."""

MIN_CUT_EXTENT = 0.01
"""The shared points must span at least this fraction of the body's size: a cut
leaves an outline across the plane, a graze leaves a point."""

TRANSITION_RE = 5e5
"""Below this a laminar flat plate is the physical floor even for a turbulent model,
which is what a user quoting a low-Re case against ITTC would otherwise trip over."""


# ------------------------------------------------------------------- the mesh --


def load_surface(case: Path) -> dict[str, Any]:
    """Points, the boundary faces and the patches -- no owner, no internal faces.

    Face indices are kept relative to the first boundary face (`face_base`), because
    the internal faces are most of the file and none of the answer: the ONERA M6 mesh's
    `faces` is 184 MB, and walking every face of it took minutes where the boundary
    alone takes seconds."""
    poly = Path(case) / "constant" / "polyMesh"
    if not poly.is_dir():
        raise MeshError(f"{Path(case).as_posix()}: no constant/polyMesh")
    boundary = layer_report.read_boundary(poly / "boundary")
    base = min((e["startFace"] for e in boundary.values()), default=0)
    end = max((e["startFace"] + e["nFaces"] for e in boundary.values()), default=0)
    verts, offsets = _boundary_faces(poly / "faces", base, end)
    return {
        "points": _points(poly / "points"),
        "verts": verts,
        "offsets": offsets,
        "boundary": boundary,
        "face_base": base,
    }


def _points(path: Path) -> np.ndarray:
    """`points` -> (n, 3), without walking the file a character at a time.

    The list is everything between the `(` under its count and the file's last `)`,
    so it is read with one `fromstring` and checked against the count; a file that
    does not add up goes through `layer_report.read_points`. 68 s -> seconds on the
    ONERA M6 mesh's 70 MB."""
    if not path.is_file():
        raise MeshError(f"no {path.as_posix()}")
    text = path.read_text(errors="replace")
    head = text[:4096]
    if "binary" not in head:
        lines = text[:8192].split("\n")
        opening = next((i for i, line in enumerate(lines) if line.strip() == "("), None)
        if opening is not None and opening > 0 and lines[opening - 1].strip().isdigit():
            count = int(lines[opening - 1].strip())
            start = len("\n".join(lines[:opening + 1]))
            body = text[start:text.rindex(")")].replace("(", " ").replace(")", " ")
            values = np.array(body.split(), dtype=np.float64)
            if values.size == 3 * count:
                return values.reshape(-1, 3)
    return layer_report.read_points(path)


def _boundary_faces(path: Path, base: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Faces `base` to `end` of a polyMesh `faces` file, as (verts, offsets).

    An ascii `faceList` is written one face to a line, `4(0 1 2 3)`, so the boundary
    is a slice of lines. Anything else -- the compact form, or a file that does not
    look like that -- goes through `layer_report.read_faces` whole and is sliced after."""
    if not path.is_file():
        raise MeshError(f"no {path.as_posix()}")
    text = path.read_text(errors="replace")
    head = text[:4096]
    if "faceCompactList" not in head and "binary" not in head:
        lines = text.split("\n")
        opening = next((i for i, line in enumerate(lines[:400]) if line.strip() == "("), None)
        if opening is not None and opening > 0 and lines[opening - 1].strip().isdigit():
            rows = lines[opening + 1 + base: opening + 1 + end]
            if len(rows) == end - base and all(r.rstrip().endswith(")") for r in rows[:1] + rows[-1:]):
                sizes = [int(r.split("(", 1)[0]) for r in rows]
                flat = " ".join(r.split("(", 1)[1] for r in rows).replace(")", " ")
                verts = np.array(flat.split(), dtype=np.int64)
                offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
                if verts.size == offsets[-1]:
                    return verts, offsets
    verts, offsets = layer_report.read_faces(path)
    return verts[offsets[base]:offsets[end]], offsets[base:end + 1] - offsets[base]


def patch_faces(mesh: dict, name: str) -> np.ndarray:
    entry = mesh["boundary"].get(name)
    if entry is None:
        raise MeshError(f"no patch named {name!r}; the mesh has "
                        + ", ".join(mesh["boundary"]))
    start = entry["startFace"] - mesh.get("face_base", 0)
    return np.arange(start, start + entry["nFaces"], dtype=np.int64)


def patch_points(mesh: dict, faces: np.ndarray) -> np.ndarray:
    off = mesh["offsets"]
    if faces.size == 0:
        return np.zeros(0, dtype=np.int64)
    pieces = [mesh["verts"][off[f]:off[f + 1]] for f in faces]
    return np.unique(np.concatenate(pieces))


def symmetry_patches(mesh: dict) -> list[str]:
    return [name for name, entry in mesh["boundary"].items()
            if entry["type"] in SYMMETRY_TYPES and entry["nFaces"] > 0]


# ------------------------------------------------------------ what is mirrored --


def cuts(mesh: dict, bodies: Sequence[str], not_mirror: Sequence[str] = ()) -> list[dict]:
    """Every symmetry patch, and whether it cuts the body.

    One entry per symmetry patch with the plane it lies in, how many points it shares
    with the body, and `mirrors` -- True when it cuts the body and nobody said
    otherwise. `why` is the sentence that goes beside the number."""
    body_faces = np.concatenate([patch_faces(mesh, b) for b in bodies])
    body_pts = patch_points(mesh, body_faces)
    pts = mesh["points"]
    body_size = float(np.ptp(pts[body_pts], axis=0).max()) if body_pts.size else 0.0
    domain = float(np.ptp(pts, axis=0).max()) or 1.0
    found = []
    for name in symmetry_patches(mesh):
        faces = patch_faces(mesh, name)
        centres, areas = layer_report.face_geometry(mesh, faces)
        total = areas.sum(axis=0)
        norm = float(np.linalg.norm(total))
        entry: dict[str, Any] = {"patch": name, "mirrors": False}
        if norm <= 0.0:
            entry["why"] = f"{name} has no area to find a plane in"
            found.append(entry)
            continue
        normal = total / norm
        offset = float(np.dot(centres, normal).mean())
        stray = float(np.abs(centres @ normal - offset).max())
        entry.update(normal=[round(float(c), 6) for c in normal], offset=offset)
        if stray > PLANAR_TOL * domain:
            entry["why"] = (f"{name} is not one plane (faces stray {stray:.3g} m from it), "
                            "so nothing is mirrored in it")
            found.append(entry)
            continue
        shared = np.intersect1d(body_pts, patch_points(mesh, faces), assume_unique=True)
        entry["shared_points"] = int(shared.size)
        extent = float(np.ptp(pts[shared], axis=0).max()) if shared.size else 0.0
        plane = _plane_words(normal, offset)
        if shared.size < MIN_SHARED_POINTS or extent < MIN_CUT_EXTENT * max(body_size, 1e-30):
            entry["why"] = f"{name} ({plane}) does not reach the body, so it cuts nothing"
        elif name in not_mirror:
            entry["why"] = (f"{name} ({plane}) meets the body along {shared.size} points "
                            "but was named with --not-mirror: the body ends there")
        else:
            entry["mirrors"] = True
            entry["why"] = (f"{name} ({plane}) cuts the body: {shared.size} points in common "
                            "along the cut, so the mesh holds half of what it mirrors")
        found.append(entry)
    return found


def mirror_factor(found: Sequence[dict]) -> int:
    """2 to the number of planes that cut the body: the meshed part is 1/factor of it."""
    return 2 ** sum(1 for c in found if c.get("mirrors"))


def _plane_words(normal: np.ndarray, offset: float) -> str:
    axis = int(np.argmax(np.abs(normal)))
    if abs(normal[axis]) > 1 - 1e-6:
        value = offset * math.copysign(1.0, normal[axis])
        value = 0.0 if abs(value) < 1e-12 else value
        return f"{'xyz'[axis]} = {value:.6g} m"
    return f"n = ({normal[0]:.3g} {normal[1]:.3g} {normal[2]:.3g}), d = {offset:.6g} m"


# ------------------------------------------------------------------ the areas --


def wetted_area(mesh: dict, bodies: Sequence[str]) -> float:
    faces = np.concatenate([patch_faces(mesh, b) for b in bodies])
    _, areas = layer_report.face_geometry(mesh, faces)
    return float(np.linalg.norm(areas, axis=1).sum())


def frontal_area(mesh: dict, bodies: Sequence[str], drag_dir: Sequence[float] = (1, 0, 0),
                 grid: int = 500) -> float:
    """The silhouette along `drag_dir`, rasterised on square pixels.

    Each face is fanned into triangles about its first vertex and every pixel whose
    centre falls inside any of them is counted once, so a second part behind the
    first adds nothing -- which is where the half-sum of |A.d| goes wrong."""
    d = np.asarray(drag_dir, dtype=float)
    d /= np.linalg.norm(d)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    eu = np.cross(d, helper)
    eu /= np.linalg.norm(eu)
    ev = np.cross(d, eu)
    faces = np.concatenate([patch_faces(mesh, b) for b in bodies])
    off, verts, pts = mesh["offsets"], mesh["verts"], mesh["points"]
    tris = []
    for f in faces:
        ring = verts[off[f]:off[f + 1]]
        for k in range(1, len(ring) - 1):
            tris.append((ring[0], ring[k], ring[k + 1]))
    if not tris:
        return 0.0
    tri = pts[np.asarray(tris)]                          # (n, 3, 3)
    u, v = tri @ eu, tri @ ev                             # (n, 3) each
    lo_u, hi_u, lo_v, hi_v = u.min(), u.max(), v.min(), v.max()
    span_u, span_v = hi_u - lo_u, hi_v - lo_v
    if span_u <= 0 or span_v <= 0:
        return 0.0
    step = max(span_u, span_v) / grid
    nu = max(1, int(math.ceil(span_u / step)))
    nv = max(1, int(math.ceil(span_v / step)))
    covered = np.zeros((nv, nu), dtype=bool)
    pu, pv = (u - lo_u) / step, (v - lo_v) / step
    for a, b in zip(pu, pv):
        r0 = max(0, int(math.floor(b.min())))
        r1 = min(nv - 1, int(math.ceil(b.max())))
        for row in range(r0, r1 + 1):
            yc = row + 0.5
            spans = []
            for i in range(3):
                j = (i + 1) % 3
                if (b[i] <= yc < b[j]) or (b[j] <= yc < b[i]):
                    t = (yc - b[i]) / (b[j] - b[i])
                    spans.append(a[i] + t * (a[j] - a[i]))
            if len(spans) < 2:
                continue
            c0 = max(0, int(math.floor(min(spans) + 0.5)))
            c1 = min(nu - 1, int(math.ceil(max(spans) - 0.5)))
            if c1 >= c0:
                covered[row, c0:c1 + 1] = True
    return float(covered.sum()) * step * step


# ------------------------------------------------------------------ the floor --


def ittc57(reynolds: float) -> float:
    """The ITTC-57 model-ship correlation line, Cf = 0.075/(log10(Re)-2)^2."""
    return 0.075 / (math.log10(max(reynolds, 11.0)) - 2.0) ** 2


def blasius(reynolds: float) -> float:
    """Laminar flat plate, whole-plate average: Cf = 1.328/sqrt(Re)."""
    return 1.328 / math.sqrt(max(reynolds, 1.0))


def friction_floor(reynolds: float, turbulent: bool = True) -> tuple[float, str]:
    """The skin friction no total drag coefficient can be below, and its name.

    ITTC-57 for a turbulent case past transition; Blasius otherwise, since a laminar
    plate carries less friction than the turbulent line and quoting ITTC there would
    call a correct laminar answer impossible."""
    if turbulent and reynolds >= TRANSITION_RE:
        return ittc57(reynolds), "ITTC-57"
    return blasius(reynolds), "Blasius laminar"


def below_floor(cd: float, reynolds: float, turbulent: bool = True) -> str:
    """The sentence to print when `cd` is impossible, or "" when it is not."""
    floor, name = friction_floor(reynolds, turbulent)
    if cd >= floor:
        return ""
    return (f"Cd {cd:.5g} is below flat-plate friction alone at Re {reynolds:.3g} "
            f"({name} Cf = {floor:.5g}). No body with form drag can do that on an area no "
            "larger than its wetted area: the coefficient is wrong, most often a half "
            "model's force over the whole body's area.")


# ------------------------------------------------------------------ one reading --


def assess(case: Path, bodies: Sequence[str], drag_dir: Sequence[float] = (1, 0, 0),
           not_mirror: Sequence[str] = (), frontal: bool = True) -> dict[str, Any]:
    """Everything above for one case, as the dict the scripts and `--json` share."""
    mesh = load_surface(case)
    found = cuts(mesh, bodies, not_mirror)
    factor = mirror_factor(found)
    wetted = wetted_area(mesh, bodies)
    out: dict[str, Any] = {
        "case": Path(case).as_posix(),
        "bodies": list(bodies),
        "planes": found,
        "mirror_factor": factor,
        "wetted_meshed_m2": wetted,
        "wetted_body_m2": wetted * factor,
    }
    if frontal:
        area = frontal_area(mesh, bodies, drag_dir)
        out["frontal_meshed_m2"] = area
        out["frontal_body_m2"] = area * factor
        out["drag_dir"] = [float(c) for c in drag_dir]
    return out


def lines(found: dict[str, Any]) -> list[str]:
    factor = found["mirror_factor"]
    out = [f"body       {', '.join(found['bodies'])}"]
    planes = found["planes"]
    if not planes:
        out.append("symmetry   none: the mesh holds the whole body")
    for plane in planes:
        out.append(f"symmetry   {plane['why']}")
    if factor > 1:
        out.append(f"mirrored   the mesh holds 1/{factor} of the body. Forces reported on "
                   f"it are 1/{factor} of the body's: multiply by {factor}.")
    out.append(f"wetted     {found['wetted_meshed_m2']:.6g} m2 as meshed"
               + (f", {found['wetted_body_m2']:.6g} m2 for the whole body" if factor > 1 else ""))
    if "frontal_meshed_m2" in found:
        out.append(f"frontal    {found['frontal_meshed_m2']:.6g} m2 as meshed"
                   + (f", {found['frontal_body_m2']:.6g} m2 for the whole body"
                      if factor > 1 else "")
                   + f" (along {tuple(found['drag_dir'])})")
    out.append("Aref       the as-meshed area: meshed force over meshed area is the whole "
               "body's coefficient.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--body", action="append", required=True,
                        help="the patch forces are taken on; repeatable")
    parser.add_argument("--drag-dir", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--not-mirror", action="append", default=[],
                        help="a symmetry patch where the body ends rather than is mirrored "
                             "(a double-body waterline)")
    parser.add_argument("--cd", type=float, help="a total drag coefficient to check")
    parser.add_argument("--re", type=float, help="its Reynolds number, with --cd")
    parser.add_argument("--laminar", action="store_true",
                        help="the case is laminar: the floor is Blasius")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        found = assess(args.case, args.body, args.drag_dir, args.not_mirror)
    except MeshError as exc:
        print(f"unmeasured: {exc}")
        return 0
    verdict = ""
    if args.cd is not None:
        if args.re is None:
            parser.error("--cd needs --re")
        verdict = below_floor(args.cd, args.re, turbulent=not args.laminar)
        floor, name = friction_floor(args.re, turbulent=not args.laminar)
        found["floor"] = {"cf": floor, "line": name, "cd": args.cd, "impossible": bool(verdict)}
    if args.json:
        print(json.dumps(found, indent=2))
    else:
        print("\n".join(lines(found)))
        if "floor" in found:
            f = found["floor"]
            print(verdict or f"floor      Cd {f['cd']:.5g} is above {f['line']} "
                             f"Cf = {f['cf']:.5g}: possible")
    return 1 if verdict else 0


if __name__ == "__main__":
    sys.exit(main())
