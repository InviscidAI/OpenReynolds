#!/usr/bin/env python3
"""How much of this wall actually carries a prism layer, measured off the mesh.

Three meshers were asked the same question about the same hull on 2026-08-31 and
none of the three answers could be put next to each other. snappyHexMesh prints a
layer table and believes it. cfMesh's `cartesianMesh` prints nothing comparable.
The hybrid route -- snappy's geometry with cfMesh's `generateBoundaryLayers` --
prints nothing at all. Until the meshes themselves were measured there was no
number that meant the same thing twice, and a study was being decided on
self-reports written to three different scales.

So this ignores what the mesher said and measures the mesh. The quantity is the
first-cell height at each patch face,

    h = 2 |C_face - C_owner|

which is exact for a prismatic layer cell, whose centre sits at mid-height, and
close enough on the squashed hexes that are left where nothing was extruded. The
centres are computed from `points`, `faces`, `owner` and `neighbour` by OpenFOAM's
own pyramid decomposition, so nothing has to have been written and no solver has
to have run -- a mesh straight out of the mesher can be measured.

A height on its own says nothing: 1 mm is a layer on a hull and a coarse cell in a
duct. Coverage is therefore h against a REFERENCE WALL SPACING -- the same wall
meshed with no layer request (`--ref`), or a spacing already known
(`--wall-spacing`). With `--ref` the lookup is per face, which is what lets it
survive a graded wall. With neither, the threshold falls back to half this mesh's
own median h, which is only meaningful when coverage happens to be near a half,
and the fallback is named wherever it is used.

The calibration that licenses the rest: on the round-4 snappy mesh, the one case
where snappy does report a number, snappy said 66.7% and this said 65.2%. That
agreement is the only reason to believe the 100.0% it reports on the cfMesh mesh,
where there is no self-report to check it against.

This reads and prints. It writes nothing, changes nothing, and exits 0 whether or
not there was anything to measure -- the reading can also be wrong, so an
unreadable or binary mesh is reported as unmeasured rather than as zero coverage.

    python3 layer_report.py <case>                       # every wall patch
    python3 layer_report.py <case> --patch hull
    python3 layer_report.py <case> --patch hull --ref ../hull-nolayers
    python3 layer_report.py <case> --wall-spacing 0.027  # a spacing you know
    python3 layer_report.py <case> --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Where the first-cell height stops looking like a layer. A face carrying n layers
# of expansion q is thinned by (1 + q + ... + q^(n-1)) relative to the same wall
# with no layer on it; a face carrying none is unchanged, so its ratio is about 1.
# 1.5 sits in the empty gap between "none" and "one or more".
THINNING_LAYERED = 1.5


class MeshError(Exception):
    """The mesh could not be read. Not the same as a mesh with no layers on it."""


# ------------------------------------------------------------------ reading --


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _header(text: str) -> tuple[str, str, str]:
    """`(format, class, body)` -- the body being everything after `FoamFile { }`."""
    fmt = re.search(r"\bformat\s+(\w+)\s*;", text)
    cls = re.search(r"\bclass\s+(\w+)\s*;", text)
    end = 0
    at = text.find("FoamFile")
    if at >= 0:
        depth, j = 0, text.index("{", at)
        for j in range(j, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        end = j + 1
    return (
        (fmt.group(1) if fmt else "ascii"),
        (cls.group(1) if cls else ""),
        text[end:],
    )


def _read(path: Path) -> tuple[str, str, str]:
    if not path.is_file():
        raise MeshError(f"no {path.as_posix()}")
    try:
        text = _strip(path.read_text(errors="replace"))
    except OSError as exc:  # unreadable is a fact about the run, not a crash
        raise MeshError(f"{path.as_posix()}: {exc}") from exc
    fmt, cls, body = _header(text)
    if fmt != "ascii":
        raise MeshError(
            f"{path.as_posix()}: written {fmt}, and this reads ascii only -- "
            "`foamFormatConvert` on a copy, or `writeFormat ascii;` in controlDict"
        )
    return fmt, cls, body


def _list_body(body: str, start: int = 0) -> tuple[str, int]:
    """The text inside the next balanced `( ... )`, and where it ended."""
    i = body.index("(", start)
    depth, j = 0, i
    for j in range(i, len(body)):
        if body[j] == "(":
            depth += 1
        elif body[j] == ")":
            depth -= 1
            if depth == 0:
                break
    else:
        raise MeshError("unterminated list")
    return body[i + 1 : j], j + 1


def _ints(chunk: str) -> np.ndarray:
    flat = chunk.replace("(", " ").replace(")", " ")
    return np.fromstring(flat, dtype=np.int64, sep=" ")


def read_labels(path: Path) -> np.ndarray:
    """A polyMesh labelList (`owner`, `neighbour`) as an int array."""
    _, _, body = _read(path)
    inner, _ = _list_body(body)
    return _ints(inner)


def read_points(path: Path) -> np.ndarray:
    """`points` -> (nPoints, 3)."""
    _, _, body = _read(path)
    inner, _ = _list_body(body)
    flat = inner.replace("(", " ").replace(")", " ")
    a = np.fromstring(flat, dtype=np.float64, sep=" ")
    if a.size % 3:
        raise MeshError(f"{path.as_posix()}: point list is not a multiple of three")
    return a.reshape(-1, 3)


def read_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """`faces` -> (vertex labels, offsets), where face i is verts[off[i]:off[i+1]].

    Both spellings are read: the plain `faceList` (`4(0 1 2 3)`) and the
    `faceCompactList` (an offset list then one flat vertex list) that newer ESI
    versions write.
    """
    _, cls, body = _read(path)
    if cls == "faceCompactList":
        first, after = _list_body(body)
        second, _ = _list_body(body, after)
        return _ints(second), _ints(first)

    inner, _ = _list_body(body)
    flat = _ints(inner)
    verts: list[np.ndarray] = []
    offsets = [0]
    i, n = 0, flat.size
    while i < n:
        size = int(flat[i])
        if size <= 0 or i + 1 + size > n:
            raise MeshError(f"{path.as_posix()}: face list is malformed at label {i}")
        verts.append(flat[i + 1 : i + 1 + size])
        offsets.append(offsets[-1] + size)
        i += 1 + size
    if not verts:
        return np.zeros(0, np.int64), np.zeros(1, np.int64)
    return np.concatenate(verts), np.asarray(offsets, dtype=np.int64)


def read_boundary(path: Path) -> dict[str, dict[str, Any]]:
    """`boundary` -> {patch: {type, nFaces, startFace}}, in file order."""
    _, _, body = _read(path)
    inner, _ = _list_body(body)
    out: dict[str, dict[str, Any]] = {}
    for m in re.finditer(r"([\w.\-]+)\s*\{(.*?)\}", inner, flags=re.S):
        name, blk = m.group(1), m.group(2)
        nf = re.search(r"nFaces\s+(\d+)\s*;", blk)
        sf = re.search(r"startFace\s+(\d+)\s*;", blk)
        ty = re.search(r"\btype\s+(\w+)\s*;", blk)
        if nf and sf:
            out[name] = {
                "type": ty.group(1) if ty else "unknown",
                "nFaces": int(nf.group(1)),
                "startFace": int(sf.group(1)),
            }
    return out


def load(case: Path) -> dict[str, Any]:
    """The parts of `constant/polyMesh` this needs, as arrays."""
    poly = case / "constant" / "polyMesh"
    if not poly.is_dir():
        raise MeshError(f"{case.as_posix()}: no constant/polyMesh")
    verts, offsets = read_faces(poly / "faces")
    owner = read_labels(poly / "owner")
    neighbour = (
        read_labels(poly / "neighbour")
        if (poly / "neighbour").is_file()
        else np.zeros(0, dtype=np.int64)
    )
    n_cells = 1 + int(max(owner.max(initial=-1), neighbour.max(initial=-1)))
    return {
        "case": case,
        "points": read_points(poly / "points"),
        "verts": verts,
        "offsets": offsets,
        "owner": owner,
        "neighbour": neighbour,
        "boundary": read_boundary(poly / "boundary"),
        "n_cells": n_cells,
    }


# ----------------------------------------------------------------- geometry --


def face_geometry(mesh: dict, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centres and area vectors of the named faces, by OpenFOAM's decomposition.

    A polygon is fanned into triangles about the average of its own vertices and
    the centre is the area-weighted mean of theirs, which is what OpenFOAM does
    and what makes `h` here the same number the solver would see. Faces are
    handled in groups of equal vertex count so the arithmetic stays vectorised.
    """
    faces = np.asarray(faces, dtype=np.int64)
    ctr = np.zeros((faces.size, 3))
    area = np.zeros((faces.size, 3))
    if faces.size == 0:
        return ctr, area
    off = mesh["offsets"]
    sizes = off[faces + 1] - off[faces]
    for size in np.unique(sizes):
        rows = np.flatnonzero(sizes == size)
        cols = off[faces[rows]][:, None] + np.arange(int(size))[None, :]
        p = mesh["points"][mesh["verts"][cols]]  # (m, size, 3)
        if size == 3:
            ctr[rows] = p.mean(axis=1)
            area[rows] = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            continue
        avg = p.mean(axis=1)
        nxt = np.roll(p, -1, axis=1)
        tri_area = 0.5 * np.cross(nxt - p, avg[:, None, :] - p)
        tri_ctr = (p + nxt + avg[:, None, :]) / 3.0
        mag = np.linalg.norm(tri_area, axis=2)
        total = mag.sum(axis=1)
        safe = np.where(total > 0.0, total, 1.0)
        weighted = (tri_ctr * mag[:, :, None]).sum(axis=1) / safe[:, None]
        # A face of zero total area has no weighted centre; its vertex average is
        # the only answer left, and it is the right one for a degenerate sliver.
        ctr[rows] = np.where((total > 0.0)[:, None], weighted, avg)
        area[rows] = tri_area.sum(axis=1)
    return ctr, area


def cell_centres(mesh: dict, cells: np.ndarray) -> np.ndarray:
    """Centres of the named cells, by OpenFOAM's pyramid decomposition.

    Only the cells asked for are built, because on a real hull the wall cells are
    a percent of the mesh and walking all of them in Python is the difference
    between a second and a minute.
    """
    cells = np.asarray(cells, dtype=np.int64)
    if cells.size == 0:
        return np.zeros((0, 3))

    wanted = np.zeros(mesh["n_cells"], dtype=bool)
    wanted[cells] = True
    owner, neighbour = mesh["owner"], mesh["neighbour"]
    cell_of = np.concatenate([owner, neighbour])
    face_of = np.concatenate([np.arange(owner.size), np.arange(neighbour.size)])
    keep = wanted[cell_of]
    cell_of, face_of = cell_of[keep], face_of[keep]

    row_of_cell = np.full(mesh["n_cells"], -1, dtype=np.int64)
    row_of_cell[cells] = np.arange(cells.size)
    row = row_of_cell[cell_of]

    fc, sf = face_geometry(mesh, face_of)
    n = cells.size

    def summed(weights: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.bincount(row, weights=weights[:, k], minlength=n) for k in range(3)],
            axis=1,
        )

    count = np.bincount(row, minlength=n)
    est = summed(fc) / np.maximum(count, 1)[:, None]

    sign = np.where(owner[face_of] == cell_of, 1.0, -1.0)
    pyr_vol = (sf * (fc - est[row])).sum(axis=1) * sign / 3.0
    pyr_ctr = 0.75 * fc + 0.25 * est[row]
    vol = np.bincount(row, weights=pyr_vol, minlength=n)
    moment = summed(pyr_ctr * pyr_vol[:, None])
    scale = np.abs(vol).max(initial=0.0)
    good = np.abs(vol) > 1e-12 * max(scale, 1e-300)
    return np.where(good[:, None], moment / np.where(good, vol, 1.0)[:, None], est)


def first_cell_heights(mesh: dict, patch: str) -> tuple[np.ndarray, np.ndarray]:
    """`(h, face centres)` for one patch: h = 2 |C_face - C_owner| per face."""
    entry = mesh["boundary"].get(patch)
    if entry is None:
        have = ", ".join(sorted(mesh["boundary"])) or "none"
        raise MeshError(f"no patch '{patch}' in this mesh (patches: {have})")
    start, n = entry["startFace"], entry["nFaces"]
    faces = np.arange(start, start + n)
    if n and faces[-1] >= mesh["offsets"].size - 1:
        raise MeshError(f"patch '{patch}' names faces the face list does not have")
    fc, _ = face_geometry(mesh, faces)
    cc = cell_centres(mesh, mesh["owner"][faces])
    return 2.0 * np.linalg.norm(fc - cc, axis=1), fc


# -------------------------------------------------------------- referencing --


def face_spacing(mesh: dict, patch: str) -> float:
    """A patch's in-plane length scale, `sqrt(median face area)`.

    Used only to size the buckets for the reference lookup, where the answer only
    has to be right to a factor of two.
    """
    entry = mesh["boundary"][patch]
    faces = np.arange(entry["startFace"], entry["startFace"] + entry["nFaces"])
    _, sf = face_geometry(mesh, faces)
    mag = np.linalg.norm(sf, axis=1)
    return float(np.sqrt(np.median(mag))) if mag.size else 0.0


def nearest(
    query: np.ndarray, ref: np.ndarray, ref_val: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face nearest-neighbour lookup by voxel hash (scipy is not on the image).

    Reference and query are the same wall meshed twice, so the two face-centre
    clouds sit on top of one another and a bucket of side ~`cell` always contains
    the match. Returns `(value, distance)`; the distance is infinite where nothing
    was found in the 27-bucket neighbourhood, which is how an unmatched face stays
    visible instead of being silently assigned someone else's spacing.
    """
    out = np.full(len(query), np.nan)
    dist = np.full(len(query), np.inf)
    if len(ref) == 0 or len(query) == 0 or not np.isfinite(cell) or cell <= 0:
        return out, dist

    origin = ref.min(axis=0)
    key = np.floor((ref - origin) / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, k in enumerate(map(tuple, key)):
        buckets.setdefault(k, []).append(i)

    qkey = np.floor((query - origin) / cell).astype(np.int64)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for i in range(len(query)):
        k = qkey[i]
        cand: list[int] = []
        for o in offsets:
            cand.extend(buckets.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]), ()))
        if not cand:
            continue
        idx = np.asarray(cand)
        d = np.linalg.norm(ref[idx] - query[i], axis=1)
        j = int(np.argmin(d))
        out[i] = ref_val[idx[j]]
        dist[i] = d[j]
    return out, dist


# --------------------------------------------------------------- measuring ---


def _stats(h: np.ndarray) -> dict[str, float]:
    return {
        "min": float(h.min()),
        "mean": float(h.mean()),
        "max": float(h.max()),
        "median": float(np.median(h)),
    }


def measure_patch(
    mesh: dict,
    patch: str,
    ref_mesh: dict | None = None,
    wall_spacing: float | None = None,
    cell: float | None = None,
) -> dict[str, Any]:
    """One patch, measured. Raises `MeshError` and nothing else."""
    entry = mesh["boundary"][patch]
    record: dict[str, Any] = {
        "patch": patch,
        "type": entry["type"],
        "faces": entry["nFaces"],
        "coverage_pct": None,
        "layered_faces": None,
        "basis": None,
        "first_cell_m": None,
        "reference_spacing_m": None,
        "thinning": None,
        "note": "",
    }
    if entry["nFaces"] == 0:
        record["note"] = "no faces on this patch"
        return record

    h, fc = first_cell_heights(mesh, patch)
    record["first_cell_m"] = _stats(h)

    per_face: np.ndarray | None = None
    if ref_mesh is not None:
        if patch not in ref_mesh["boundary"]:
            raise MeshError(f"the reference mesh has no patch '{patch}'")
        h_ref, fc_ref = first_cell_heights(ref_mesh, patch)
        side = cell if cell else face_spacing(ref_mesh, patch)
        value, distance = nearest(fc, fc_ref, h_ref, side or 0.0)
        matched = np.isfinite(distance)
        if matched.sum() > 0.9 * h.size:
            per_face = np.where(matched, value, np.nan)
        record["reference_spacing_m"] = float(np.median(h_ref))
        if per_face is None:
            record["note"] = (
                f"only {int(matched.sum())} of {h.size} faces found a match in the "
                "reference mesh, so the per-face comparison was not used -- is it "
                "the same wall, and is --cell near the wall spacing?"
            )
    elif wall_spacing:
        record["reference_spacing_m"] = float(wall_spacing)

    if per_face is not None:
        ratio = per_face / np.maximum(h, 1e-30)
        finite = np.isfinite(ratio)
        layered = finite & (ratio > THINNING_LAYERED)
        record["basis"] = "per face against the reference mesh"
        record["layered_faces"] = int(layered.sum())
        record["coverage_pct"] = round(100.0 * float(layered.mean()), 2)
        r = ratio[finite]
        record["thinning"] = {
            "median": float(np.median(r)),
            "min": float(r.min()),
            "max": float(r.max()),
            "p5": float(np.percentile(r, 5)),
            "p95": float(np.percentile(r, 95)),
            "matched_faces": int(finite.sum()),
            "median_over_layered": (
                float(np.median(ratio[layered])) if layered.any() else None
            ),
        }
        return record

    reference = record["reference_spacing_m"]
    if reference:
        record["basis"] = "half the reference wall spacing"
    else:
        reference = float(np.median(h))
        record["basis"] = "half this mesh's own median (no reference given)"
        record["note"] = (
            "with no --ref and no --wall-spacing the threshold comes from this mesh, "
            "so it can only see a split that is somewhere near half: a wall that is "
            "wholly layered and one that is wholly bare look alike to it"
        )
    threshold = 0.5 * reference
    layered = h < threshold
    record["layered_faces"] = int(layered.sum())
    record["coverage_pct"] = round(100.0 * float(layered.mean()), 2)
    record["threshold_m"] = float(threshold)
    if record["reference_spacing_m"]:
        ratio = record["reference_spacing_m"] / np.maximum(h, 1e-30)
        record["thinning"] = {
            "median": float(np.median(ratio)),
            "min": float(ratio.min()),
            "max": float(ratio.max()),
            "p5": float(np.percentile(ratio, 5)),
            "p95": float(np.percentile(ratio, 95)),
            "matched_faces": int(h.size),
            "median_over_layered": (
                float(np.median(ratio[layered])) if layered.any() else None
            ),
        }
    return record


def wall_patches(mesh: dict) -> list[str]:
    """Patches worth measuring. A layer on an inlet is not a thing, so walls first;
    if the mesh names none, every patch is offered rather than none."""
    walls = [n for n, e in mesh["boundary"].items() if e["type"] == "wall"]
    return walls or list(mesh["boundary"])


def measure(
    case: Path,
    patches: list[str] | None = None,
    ref: Path | None = None,
    wall_spacing: float | None = None,
    cell: float | None = None,
) -> dict[str, Any]:
    """Everything this script knows about a case, as plain data. `report` formats it."""
    found: dict[str, Any] = {
        "case": case.as_posix(),
        "reference": ref.as_posix() if ref else None,
        "wall_spacing_m": wall_spacing,
        "patches": [],
        "problems": [],
    }
    try:
        mesh = load(case)
    except MeshError as exc:
        found["problems"].append(str(exc))
        return found

    ref_mesh = None
    if ref is not None:
        try:
            ref_mesh = load(ref)
        except MeshError as exc:
            found["problems"].append(f"reference not read: {exc}")

    wanted = patches or wall_patches(mesh)
    for name in wanted:
        if name not in mesh["boundary"]:
            have = ", ".join(sorted(mesh["boundary"])) or "none"
            found["problems"].append(f"no patch '{name}' in this mesh (patches: {have})")
            continue
        try:
            found["patches"].append(
                measure_patch(mesh, name, ref_mesh, wall_spacing, cell)
            )
        except MeshError as exc:
            found["problems"].append(f"patch '{name}': {exc}")
    return found


# ---------------------------------------------------------------- reporting --


def histogram(h: np.ndarray, bins: int = 20) -> list[str]:
    lo, hi = float(h.min()), float(h.max())
    if hi <= lo:
        return [f"    every face at {lo * 1e3:.4f} mm"]
    edges = np.geomspace(max(lo, hi * 1e-4), hi, bins + 1)
    counts, _ = np.histogram(h, bins=edges)
    top = int(counts.max()) or 1
    return [
        f"    {a * 1e3:9.4f} - {b * 1e3:9.4f} mm | {c:7d} {'#' * int(round(40 * c / top))}"
        for c, a, b in zip(counts, edges[:-1], edges[1:])
    ]


def report(found: dict[str, Any], heights: dict[str, np.ndarray] | None = None) -> str:
    lines = [f"# layers on {found['case']}"]
    if found["reference"]:
        lines.append(f"  reference wall: {found['reference']}")
    elif found["wall_spacing_m"]:
        lines.append(f"  reference wall spacing: {found['wall_spacing_m'] * 1e3:.4f} mm")
    lines.append("")

    for p in found["patches"]:
        lines.append(f"# patch '{p['patch']}' ({p['type']})")
        lines.append(f"  faces                 {p['faces']}")
        if p["first_cell_m"] is None:
            lines.append(f"  {p['note'] or 'nothing to measure'}")
            lines.append("")
            continue
        fc = p["first_cell_m"]
        lines.append(f"  COVERAGE              {p['coverage_pct']} %"
                     f"  ({p['layered_faces']} / {p['faces']} faces)")
        lines.append(f"    measured as         {p['basis']}")
        lines.append(
            f"  first-cell height     min {fc['min'] * 1e3:.4f}"
            f"   mean {fc['mean'] * 1e3:.4f}"
            f"   max {fc['max'] * 1e3:.4f} mm"
        )
        if p["reference_spacing_m"]:
            lines.append(
                f"  reference spacing     {p['reference_spacing_m'] * 1e3:.4f} mm"
            )
        t = p["thinning"]
        if t:
            lines.append(
                f"  thinning vs reference median {t['median']:.2f}x"
                f"   p5 {t['p5']:.2f}   p95 {t['p95']:.2f}"
                f"   ({t['matched_faces']} faces)"
            )
            if t["median_over_layered"] is not None:
                lines.append(
                    f"    over layered faces  median {t['median_over_layered']:.2f}x"
                )
        else:
            lines.append(
                "  thinning              not derivable without a reference wall "
                "(--ref or --wall-spacing)"
            )
        if p["note"]:
            lines.append(f"  note: {p['note']}")
        if heights is not None and p["patch"] in heights:
            lines.append("  distribution of first-cell height (log bins):")
            lines.extend(histogram(heights[p["patch"]]))
        lines.append("")

    if found["problems"]:
        lines.append("# not measured")
        lines.extend(f"  {problem}" for problem in found["problems"])
        lines.append("")
    if not found["patches"] and not found["problems"]:
        lines.append("  no patches to measure")
    lines.append(
        "  what a coverage number is worth depends on the reference it was measured "
        "against; nothing here decides whether this mesh is good enough"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path, help="the case whose mesh is measured")
    parser.add_argument(
        "--patch",
        action="append",
        default=None,
        help="patch to measure; repeatable (default: every wall patch)",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=None,
        help="the same wall meshed with no layer request; its spacing sets the threshold",
    )
    parser.add_argument(
        "--wall-spacing",
        type=float,
        default=None,
        help="reference wall spacing in metres, when the no-layer mesh is not around",
    )
    parser.add_argument(
        "--cell",
        type=float,
        default=None,
        help="bucket side for the reference lookup (default: the reference face spacing)",
    )
    parser.add_argument("--json", action="store_true", help="the same facts as JSON")
    args = parser.parse_args(argv)

    found = measure(args.case, args.patch, args.ref, args.wall_spacing, args.cell)

    if args.json:
        print(json.dumps(found, indent=2))
    else:
        heights = {}
        try:
            mesh = load(args.case)
            for p in found["patches"]:
                if p["first_cell_m"] is not None:
                    heights[p["patch"]] = first_cell_heights(mesh, p["patch"])[0]
        except MeshError:
            heights = {}
        print(report(found, heights))

    # A mesh with no layers on it is an answer, and so is a mesh that could not be
    # read. Neither is this script failing, so neither is a non-zero exit.
    return 0


if __name__ == "__main__":
    sys.exit(main())
