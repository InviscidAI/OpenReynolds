#!/usr/bin/env python3
"""The exported patch set, measured -- closure of the union, not of the parts.

    python3 cad_audit.py <case>/constant/triSurface
    python3 cad_audit.py <case>/constant/triSurface --json
    python3 cad_audit.py <case>/constant/triSurface --surface-check

It reads `patches.json` and the per-patch STL files beside it, composes their union,
and reports what the union is: closed or not, manifold or not, wound one way or two,
covered by the patches exhaustively and disjointly or not, crossing itself or not, and
plausibly in metres or not.

**The distinction this whole script turns on.** An individual patch file is an *open*
surface -- an inlet disc has a rim, a wall has the edges where the inlet and the outlet
were cut away from it -- and a closure check run file by file fails on every correct
export there has ever been. It is the *union* that has to be closed, manifold and
consistently wound. So every topological number here is taken on the concatenation of
all the patch files at once, and the per-file numbers are printed beside it only so a
reader can see that the parts being open is the normal state of affairs.

**It reimplements nothing.** `preflight.read_triangles()` reads the STL, binary or
ASCII; `preflight.surface_topology()` counts open edges, non-manifold edges,
same-direction edges and degenerate triangles, on welded vertices, because without the
weld every edge looks open; `preflight.scale_diagnosis()` is the millimetre ladder;
`surfaces.weld()`, `surfaces.Index` and `surfaces.tri_tri_intersect()` are the triangle
machinery. What is written here is the audit logic on top of them -- composing the
union, partitioning it by patch, and turning the numbers into findings. A second edge
counter that is nearly the same as the first is the failure this arrangement exists to
prevent.

**What degrades, and how.** Above `preflight.TOPOLOGY_TRIANGLE_LIMIT` the edge
bookkeeping and the self-intersection sweep are not run, and the findings that depend
on them come back `skipped` with the reason in `measured`. They never come back `ok`:
a surface nobody looked at is not a surface that passed.

Exit code is 0 whatever is found. It is 2 only where the inputs are refused -- a
directory that is not there, or one with no `patches.json` in it -- because a surface
audit of a directory this script had to guess about is worth nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight  # noqa: E402  (sibling script, not a package)
import surfaces  # noqa: E402  (sibling script, not a package)

Finding = preflight.Finding

MANIFEST_NAME = "patches.json"
"""I3's manifest. Its absence is a refusal rather than a finding: without it there is
no patch set, only a directory with some triangles in it, and which of those files the
mesher will read is not a thing to guess at."""

SURFACE_CHECK_TIMEOUT = 600
"""`surfaceCheck` on a large surface is not fast, and a hung utility should come back
as a skipped finding rather than as a script that never returns."""


# -- what was read off the disk ----------------------------------------------------


class PatchSurface(NamedTuple):
    """One named patch and the triangles in its file."""

    name: str
    file: str
    path: Path
    triangles: np.ndarray


class Union(NamedTuple):
    """Every patch's triangles in one array, and who owns each row.

    `owner[i]` indexes `patches`. The union is what the topology is measured on; the
    owner column is what makes the coverage question askable at all.
    """

    patches: list[PatchSurface]
    triangles: np.ndarray
    owner: np.ndarray


class Refused(Exception):
    """Inputs this cannot audit. Carries the message printed on stderr."""


# -- reading the patch set ---------------------------------------------------------


def read_manifest(directory: Path) -> dict[str, Any]:
    """I3's `patches.json`, or a refusal naming what to pass."""
    path = Path(directory)
    if not path.is_dir():
        raise Refused(
            f"refused: {path} is not a directory. Pass the triSurface directory of a "
            "case -- <case>/constant/triSurface -- not a single STL."
        )
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise Refused(
            f"refused: no {MANIFEST_NAME} in {path}. The patch set is the manifest plus "
            "the files it names; without it there is no way to tell which STLs are "
            "patches of one surface. Write one, or export with cad_convert.py."
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused(
            f"refused: {manifest_path} does not read as JSON ({exc}). Fix the manifest; "
            "auditing the STLs without it would report on a patch set nobody declared."
        ) from None
    if not isinstance(data, dict):
        raise Refused(
            f"refused: {manifest_path} is not a JSON object. It should carry "
            '"patches": [...] as I3 describes.'
        )
    return data


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("patches")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def surface_files(directory: Path) -> list[Path]:
    """The STLs actually sitting in the directory, in a stable order."""
    found: list[Path] = []
    for suffix in preflight.SURFACE_SUFFIXES:
        found.extend(Path(directory).glob(f"*{suffix}"))
    return sorted(set(found), key=lambda path: path.name)


def load_union(directory: Path, manifest: dict[str, Any]) -> tuple[Union, list[str]]:
    """The manifest's patches read off disk, plus the complaints reading them raised.

    The complaints are strings rather than findings because `manifest_finding` below
    turns the whole lot into one verdict; a patch set that is wrong in three ways
    should say so in one place.
    """
    directory = Path(directory)
    complaints: list[str] = []
    patches: list[PatchSurface] = []
    named: set[str] = set()

    for entry in manifest_entries(manifest):
        name = str(entry.get("name") or "").strip()
        file = str(entry.get("file") or (f"{name}.stl" if name else "")).strip()
        if not file:
            complaints.append("a manifest entry names neither a patch nor a file")
            continue
        if not name:
            name = Path(file).stem
        named.add(file)
        path = directory / file
        if not path.is_file():
            complaints.append(f"{file} is named in the manifest and is not on disk")
            continue
        triangles = preflight.read_triangles(path)
        if triangles is None or len(triangles) == 0:
            complaints.append(f"{file} read as no triangles at all")
            continue
        declared = entry.get("triangles")
        if isinstance(declared, int) and declared != len(triangles):
            complaints.append(
                f"{file} holds {len(triangles):,} triangles, the manifest says "
                f"{declared:,}"
            )
        patches.append(PatchSurface(name, file, path, triangles))

    for path in surface_files(directory):
        if path.name not in named:
            complaints.append(f"{path.name} is on disk and in no manifest entry")

    if patches:
        triangles = np.concatenate([patch.triangles for patch in patches], axis=0)
        owner = np.concatenate([
            np.full(len(patch.triangles), index, dtype=np.int64)
            for index, patch in enumerate(patches)
        ])
    else:
        triangles = np.zeros((0, 3, 3), dtype=float)
        owner = np.zeros(0, dtype=np.int64)
    return Union(patches, triangles, owner), complaints


# -- self-intersection -------------------------------------------------------------


def _separating_axes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Which of these pairs no cheap axis separates -- a broadphase, not a verdict.

    `surfaces.tri_tri_intersect` tries twenty-three candidate axes one pair at a time,
    which is exact and costs a Python call per pair; on a two-hundred-thousand-triangle
    surface the index hands back half a million box-overlapping pairs and that call is
    the whole run. So eight of those same twenty-three axes -- the two face normals and
    each triangle's three in-plane edge normals -- are tried here for every pair at once
    in numpy, and the pairs an axis separates are dropped.

    It cannot disagree with `surfaces.tri_tri_intersect`, and that is the point of
    taking a *subset* of its axes rather than writing a second test: a separating axis
    proves disjointness outright, so a pair dropped here is disjoint, and a pair kept
    here still gets its answer from `surfaces.tri_tri_intersect` and from nowhere else.
    Every verdict this script reports comes from that function.
    """
    keep = np.ones(len(a), dtype=bool)
    if not len(a):
        return keep
    edges_a = a[:, [1, 2, 0]] - a
    edges_b = b[:, [1, 2, 0]] - b
    normal_a = np.cross(edges_a[:, 0], -edges_a[:, 2])
    normal_b = np.cross(edges_b[:, 0], -edges_b[:, 2])

    reach = np.maximum(
        np.abs(a).reshape(len(a), -1).max(axis=1),
        np.abs(b).reshape(len(b), -1).max(axis=1),
    )
    floor = (np.maximum(reach, 1e-300) * 1e-12) ** 2

    axes = [normal_a, normal_b]
    for k in range(3):
        axes.append(np.cross(edges_a[:, k], normal_a))
        axes.append(np.cross(edges_b[:, k], normal_b))

    for axis in axes:
        length2 = np.einsum("ij,ij->i", axis, axis)
        usable = length2 > floor  # a degenerate axis separates nothing
        pa = np.einsum("ijk,ik->ij", a, axis)
        pb = np.einsum("ijk,ik->ij", b, axis)
        apart = usable & (
            (pa.max(axis=1) < pb.min(axis=1)) | (pb.max(axis=1) < pa.min(axis=1))
        )
        keep &= ~apart
    return keep


def crossing_pairs(triangles: np.ndarray, corners: np.ndarray) -> dict[str, Any]:
    """Triangles that meet triangles they are not neighbours of.

    Two triangles that share a corner meet at that corner, and `tri_tri_intersect`
    counts touching as meeting -- correctly, because on a welded surface sharing an edge
    is what being a surface means. So a self-intersection sweep that did not exclude
    neighbours would report every surface as intersecting itself. Neighbours are
    excluded by welded vertex index, which is preflight's own notion of one vertex
    rather than a second one.
    """
    count = len(triangles)
    if count == 0:
        return {"pairs": [], "triangles": [], "tested": 0, "candidates": 0}

    index = surfaces.Index(triangles)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for tri_id in range(count):
        near = index.candidates_overlapping(tri_id)
        near = near[near > tri_id]
        if len(near):
            left.append(np.full(len(near), tri_id, dtype=np.int64))
            right.append(near)
    if not left:
        return {"pairs": [], "triangles": [], "tested": 0, "candidates": 0}

    first = np.concatenate(left)
    second = np.concatenate(right)
    candidates = len(first)

    shares = np.zeros(candidates, dtype=bool)
    corners_first = corners[first]
    corners_second = corners[second]
    for one in range(3):
        for other in range(3):
            shares |= corners_first[:, one] == corners_second[:, other]
    first, second = first[~shares], second[~shares]

    keep = _separating_axes(triangles[first], triangles[second])
    first, second = first[keep], second[keep]

    pairs = [
        (int(i), int(j))
        for i, j in zip(first, second)
        if surfaces.tri_tri_intersect(triangles[i], triangles[j])
    ]
    involved = sorted({tri for pair in pairs for tri in pair})
    return {
        "pairs": pairs,
        "triangles": involved,
        "tested": int(len(first)),
        "candidates": candidates,
    }


# -- coverage ----------------------------------------------------------------------


def _shared_labels(corners: np.ndarray) -> np.ndarray:
    """One label per triangle, equal where the triangles occupy the same corners.

    The corner triple is sorted first, so a triangle exported into two patches with
    its corners written in a different order is still recognised as the same face --
    which is exactly how a double assignment arrives in practice.
    """
    if not len(corners):
        return np.zeros(0, dtype=np.int64)
    ordered = np.sort(corners, axis=1)
    _unique, labels = np.unique(ordered, axis=0, return_inverse=True)
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def _connected_parts(corners: np.ndarray) -> int:
    """How many separate pieces the surface is in, by shared corners.

    Needed only to read a Euler characteristic, and only ever called when the surface
    has a hole in it, so the cost lands on the broken case rather than the good one.
    """
    if not len(corners):
        return 0
    parent = list(range(int(corners.max()) + 1))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for triple in corners.tolist():
        root = find(triple[0])
        for other in triple[1:]:
            second = find(other)
            if second != root:
                parent[second] = root
    return len({find(node) for node in set(corners.reshape(-1).tolist())})


def missing_regions(topology: dict[str, Any], corners: np.ndarray, vertices: int) -> int | None:
    """How many patches of surface are missing, off the Euler characteristic.

    A face nobody exported leaves a hole, and a hole is not a number
    `surface_topology()` reports -- it reports the open edges that bound it, which is
    four for one missing quad and four hundred for one missing cylinder wall. The count
    that matters to a person is how many pieces of surface are absent, and Euler gives
    it without enumerating a single edge a second time: a triangulated surface has
    `2E = 3F + open_edges` when no edge carries more than two triangles, so

        chi = V - (3F + open_edges) / 2 + F

    and a genus-zero surface in `c` pieces with `h` holes has `chi = 2c - h`.

    Returns `None` where the arithmetic does not apply -- a non-manifold edge, a
    degenerate triangle, or a hole count that comes out impossible -- because a made-up
    number here would be worse than saying the open edges and stopping.
    """
    open_edges = int(topology.get("open_edges", 0))
    if open_edges <= 0:
        return 0
    if topology.get("non_manifold_edges") or topology.get("degenerate_triangles"):
        return None
    faces = int(topology.get("triangles", 0))
    if (3 * faces + open_edges) % 2:
        return None
    edges = (3 * faces + open_edges) // 2
    chi = vertices - edges + faces
    holes = 2 * _connected_parts(corners) - chi
    return holes if holes >= 1 else None


# -- findings ----------------------------------------------------------------------


def manifest_finding(directory: Path, union: Union, complaints: list[str]) -> Finding:
    if not union.patches and not complaints:
        return Finding(
            "manifest", "fail",
            f"{MANIFEST_NAME} names no patches and {len(surface_files(directory))} "
            "STL files sit beside it",
            "the manifest is the patch set; an empty one means every file here is "
            "unnamed, and snappyHexMesh will take whatever the STL's own solid names "
            "happen to be",
            "list every exported patch in patches.json with its file and its role",
        )
    if complaints:
        return Finding(
            "manifest", "fail",
            f"{len(union.patches)} patch files read; " + "; ".join(complaints),
            "the manifest and the directory disagree, so the surface the mesher reads "
            "is not the surface this audit measured -- a file named and absent is a "
            "patch with no geometry, and a file present and unnamed is geometry with "
            "no boundary condition",
            "make patches.json and the directory name the same set of files",
        )
    return Finding(
        "manifest", "ok",
        f"{len(union.patches)} patches, each with its file on disk and no unlisted "
        f"STL beside them: {', '.join(patch.name for patch in union.patches)}",
        "every triangle in this directory belongs to a named patch",
    )


def scale_finding(union: Union, manifest: dict[str, Any]) -> Finding:
    if not len(union.triangles):
        return Finding("scale", "skipped", "no triangles read", "nothing to measure")
    flat = union.triangles.reshape(-1, 3)
    unit = manifest.get("unit_metres")
    factor = float(unit) if isinstance(unit, (int, float)) and unit > 0 else 1.0
    extent = (flat.max(axis=0) - flat.min(axis=0)) * factor
    verdict = preflight.scale_diagnosis(extent)
    declared = f"; manifest declares unit_metres {factor:g}" if factor != 1.0 else ""
    return Finding(
        "scale", verdict["status"],
        f"the union spans {extent[0]:.4g} x {extent[1]:.4g} x {extent[2]:.4g} m"
        + declared,
        verdict["note"] or "the bounding box is the size of a thing OpenFOAM meshes in "
        "metres",
        verdict["repair"],
    )


def _per_file(union: Union, limit: int) -> list[dict[str, Any]]:
    """Each patch file's own topology, which is what the union's is not.

    Reported so that the union-versus-parts distinction is visible rather than merely
    asserted: a correct six-patch cube has four open edges in every file and none at
    all in their union.
    """
    rows = []
    for patch in union.patches:
        row: dict[str, Any] = {"name": patch.name, "file": patch.file,
                               "triangles": int(len(patch.triangles))}
        if len(union.triangles) <= limit:
            topology = preflight.surface_topology(patch.triangles)
            if topology.get("computed"):
                row["open_edges"] = topology["open_edges"]
                row["non_manifold_edges"] = topology["non_manifold_edges"]
                row["flipped_edges"] = topology["flipped_edges"]
        rows.append(row)
    return rows


def _skipped(check: str, topology: dict[str, Any], meaning: str) -> Finding:
    return Finding(
        check, "skipped",
        f"not computed: {topology.get('note') or 'no triangles read'}",
        meaning,
        "audit the patches in groups, or coarsen the tessellation -- clmax follows the "
        "finest surface cell size and a surface this dense is usually finer than the "
        "mesh can resolve",
    )


def closure_finding(topology: dict[str, Any], per_file: list[dict[str, Any]]) -> Finding:
    if not topology.get("computed"):
        return _skipped(
            "closure", topology,
            "whether the union is closed is unknown here, which is not the same as it "
            "being closed",
        )
    open_edges = int(topology["open_edges"])
    parts = ", ".join(
        f"{row['name']} {row.get('open_edges', '?')}" for row in per_file
    )
    measured = (
        f"{preflight.count_phrase(open_edges, 'open edge')} in the union of "
        f"{len(per_file)} patch files ({topology['triangles']:,} triangles); "
        f"per file: {parts}"
    )
    if open_edges == 0:
        return Finding(
            "closure", "ok", measured,
            "the union is watertight -- the per-file counts are the seams between "
            "patches and are what a correct export looks like",
        )
    return Finding(
        "closure", "fail", measured,
        "the exported surface has a hole in it, so snappyHexMesh cannot tell inside "
        "from outside and castellation leaks out through it",
        "close the hole in the CAD, or export the face that is missing -- the union of "
        "the patches, not any one file, is what has to be closed",
    )


def manifold_finding(topology: dict[str, Any]) -> Finding:
    if not topology.get("computed"):
        return _skipped("manifold", topology,
                        "whether any edge carries more than two triangles is unknown here")
    count = int(topology["non_manifold_edges"])
    measured = f"{preflight.count_phrase(count, 'non-manifold edge')} in the union"
    if count == 0:
        return Finding("manifold", "ok", measured,
                       "every edge has exactly one or two triangles on it")
    return Finding(
        "manifold", "fail", measured,
        "an edge with more than two triangles on it is not the boundary of a solid: "
        "usually two patches exported over the same face, or two parts sharing a wall "
        "rather than one body",
        "export each face into exactly one patch, or split genuinely separate bodies "
        "into their own patch sets",
    )


def normals_finding(topology: dict[str, Any]) -> Finding:
    if not topology.get("computed"):
        return _skipped("normals", topology,
                        "whether the winding is consistent is unknown here")
    count = int(topology["flipped_edges"])
    measured = f"{preflight.count_phrase(count, 'same-direction edge')} in the union"
    if count == 0:
        return Finding("normals", "ok", measured,
                       "every shared edge is walked one way by one triangle and the "
                       "other way by its neighbour, so the normals agree")
    return Finding(
        "normals", "fail", measured,
        "some facets face inwards: an edge walked the same way by both its triangles "
        "is two triangles facing opposite ways, which reads to snappy as a hole even "
        "where the surface is geometrically watertight, and turns a solid into a void",
        "surfaceOrient the offending file against a point outside the body, and "
        "re-export with one coordinate frame and no per-file transforms",
    )


def degenerate_finding(topology: dict[str, Any]) -> Finding:
    if not topology.get("computed"):
        return _skipped("degenerate", topology,
                        "whether any triangle has a repeated corner is unknown here")
    count = int(topology["degenerate_triangles"])
    measured = f"{preflight.count_phrase(count, 'triangle')} with a repeated corner"
    if count == 0:
        return Finding("degenerate", "ok", measured, "every triangle has three corners "
                                                     "and some area")
    return Finding(
        "degenerate", "warn", measured,
        "a triangle with two corners at one point has no area and no normal; most "
        "meshers drop it, and the ones that do not read it as a crack",
        "surfaceClean the file, or tessellate again with a coarser clmax",
    )


def coverage_finding(
    union: Union, topology: dict[str, Any], corners: np.ndarray, vertices: int
) -> Finding:
    if not topology.get("computed"):
        return _skipped(
            "coverage", topology,
            "whether every face is in exactly one patch is unknown here",
        )

    labels = _shared_labels(corners)
    order = np.argsort(labels, kind="stable")
    grouped = labels[order]
    boundaries = np.flatnonzero(np.diff(grouped)) + 1
    reasons: list[str] = []

    doubled: list[str] = []
    doubled_faces = 0
    for group in np.split(order, boundaries):
        if len(group) < 2:
            continue
        owners = sorted({union.patches[o].name for o in union.owner[group]})
        doubled_faces += 1
        if len(doubled) < 6:
            centre = union.triangles[group[0]].mean(axis=0)
            where = f"({centre[0]:.4g}, {centre[1]:.4g}, {centre[2]:.4g})"
            doubled.append(
                f"the face at {where} is in " + (
                    " and ".join(owners) if len(owners) > 1
                    else f"{owners[0]} {len(group)} times"
                )
            )
    if doubled_faces:
        reasons.append(
            f"double-assigned: {preflight.count_phrase(doubled_faces, 'face is', 'faces are')}"
            " exported more than once -- " + "; ".join(doubled)
        )

    holes = missing_regions(topology, corners, vertices)
    open_edges = int(topology["open_edges"])
    if open_edges and holes:
        reasons.append(
            f"unassigned: {preflight.count_phrase(holes, 'face of the exported surface is', 'faces of the exported surface are')}"
            f" in no patch file, leaving {preflight.count_phrase(open_edges, 'open edge')}"
            " around the gap"
        )
    elif open_edges:
        reasons.append(
            f"unassigned: {preflight.count_phrase(open_edges, 'open edge')} bound part "
            "of the surface that is in no patch file, and how many faces are missing "
            "cannot be read off a surface that is also non-manifold or degenerate"
        )

    empty = [patch.name for patch in union.patches if not len(patch.triangles)]
    if empty:
        reasons.append(f"empty: {', '.join(empty)} carry no triangles")

    counts = ", ".join(
        f"{patch.name} {len(patch.triangles):,}" for patch in union.patches
    )
    head = (
        f"{len(union.patches)} patches over {len(union.triangles):,} triangles "
        f"({counts})"
    )
    if not reasons:
        return Finding(
            "coverage", "ok", head + "; every face in exactly one patch",
            "the partition is exhaustive and disjoint, so every face of the meshed "
            "domain takes the boundary condition somebody chose for it",
        )
    return Finding(
        "coverage", "fail", head + "; " + "; ".join(reasons),
        "an unassigned face lands silently in a default patch and takes whatever "
        "boundary condition it carries -- a wrong answer rather than an error -- and a "
        "double-assigned face is two surfaces in one place that snap inconsistently",
        "tag every face where the surface is created and export one file per patch, "
        "exhaustive and disjoint",
    )


def self_intersection_finding(
    union: Union, topology: dict[str, Any], corners: np.ndarray, limit: int
) -> tuple[Finding, dict[str, Any]]:
    if len(union.triangles) == 0 or len(union.triangles) > limit:
        note = topology.get("note") or "no triangles read"
        return _skipped(
            "self_intersection", {"note": note},
            "whether the surface passes through itself is unknown here",
        ), {}

    found = crossing_pairs(union.triangles, corners)
    pairs = found["pairs"]
    measured_head = (
        f"{len(found['triangles']):,} triangles in "
        f"{preflight.count_phrase(len(pairs), 'crossing pair')}, out of "
        f"{found['candidates']:,} pairs whose boxes overlap"
    )
    if not pairs:
        return Finding(
            "self_intersection", "ok", measured_head,
            "no triangle meets a triangle it is not a neighbour of",
        ), found
    examples = []
    for i, j in pairs[:5]:
        centre = (union.triangles[i].mean(axis=0) + union.triangles[j].mean(axis=0)) / 2
        examples.append(
            f"{union.patches[union.owner[i]].name}/{union.patches[union.owner[j]].name} "
            f"at ({centre[0]:.4g}, {centre[1]:.4g}, {centre[2]:.4g})"
        )
    return Finding(
        "self_intersection", "fail",
        measured_head + "; first at " + "; ".join(examples),
        "the surface passes through itself, so there is no consistent inside: snapping "
        "produces cells on both sides of a wall that is in two places at once",
        "repair the overlap in the CAD -- a boolean union of the parts rather than two "
        "parts left interpenetrating -- and re-export",
    ), found


def surface_check_finding(union: Union) -> tuple[Finding, dict[str, Any]]:
    """OpenFOAM's own opinion of each file, when OpenFOAM is on the PATH.

    Per file rather than on a merged surface, because merging would mean writing
    geometry, and this script writes none: what it wants from `surfaceCheck` is the
    second opinion on triangle quality and illegal triangles, which is a per-file
    question. Closure is answered on the union above and `surfaceCheck`'s own "not
    closed" line is reported but never failed on -- an open patch file is the normal
    state of a correct export.
    """
    binary = shutil.which("surfaceCheck")
    if binary is None:
        return Finding(
            "surface_check", "skipped",
            "surfaceCheck is not on PATH",
            "OpenFOAM's own surface utility was not run, which is not the same as it "
            "having found nothing",
            "source the OpenFOAM bashrc and run again with --surface-check",
        ), {}
    if not union.patches:
        return Finding("surface_check", "skipped", "no patch files to check",
                       "nothing to run it on"), {}

    rows: list[dict[str, Any]] = []
    status = "ok"
    notes: list[str] = []
    for patch in union.patches:
        try:
            done = subprocess.run(
                [binary, patch.file],
                cwd=str(patch.path.parent),
                capture_output=True, text=True, timeout=SURFACE_CHECK_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            status = preflight.escalate(status, "warn")
            notes.append(f"{patch.file}: surfaceCheck did not run ({exc})")
            continue
        output = done.stdout + done.stderr
        illegal = [
            line.strip() for line in output.splitlines()
            if "illegal" in line.lower() or "zero area" in line.lower()
        ]
        rows.append({
            "file": patch.file,
            "returncode": done.returncode,
            "illegal": illegal,
            "closed": "Surface is not closed" not in output,
        })
        if done.returncode != 0:
            status = preflight.escalate(status, "fail")
            notes.append(f"{patch.file}: surfaceCheck exited {done.returncode}")
        for line in illegal:
            if any(char.isdigit() for char in line):
                status = preflight.escalate(status, "warn")
                notes.append(f"{patch.file}: {line}")

    measured = f"surfaceCheck ran on {preflight.count_phrase(len(rows), 'patch file')}"
    if notes:
        measured += "; " + "; ".join(notes[:8])
    else:
        measured += " and named no illegal triangle in any of them"
    return Finding(
        "surface_check", status, measured,
        "OpenFOAM reading the same files with its own reader; its closure verdict is "
        "per file and so is expected to say a patch is open",
        "surfaceClean, or repair in the CAD, whatever it named" if status != "ok" else "",
    ), {"surface_check": rows}


# -- the audit ---------------------------------------------------------------------


def audit(directory: Path, surface_check: bool = False,
          limit: int | None = None) -> dict[str, Any]:
    """The I2 envelope for one triSurface directory. Raises `Refused` on bad inputs."""
    directory = Path(directory)
    limit = preflight.TOPOLOGY_TRIANGLE_LIMIT if limit is None else int(limit)
    manifest = read_manifest(directory)
    union, complaints = load_union(directory, manifest)

    topology = preflight.surface_topology(union.triangles if len(union.triangles) else None)
    if len(union.triangles) and len(union.triangles) <= limit:
        corners, vertices = surfaces.weld(union.triangles)
        vertex_count = int(len(vertices))
    else:
        corners = np.zeros((0, 3), dtype=np.int64)
        vertex_count = 0

    per_file = _per_file(union, limit)
    findings = [
        manifest_finding(directory, union, complaints),
        scale_finding(union, manifest),
        closure_finding(topology, per_file),
        manifold_finding(topology),
        normals_finding(topology),
        degenerate_finding(topology),
        coverage_finding(union, topology, corners, vertex_count),
    ]
    crossings, crossing_data = self_intersection_finding(union, topology, corners, limit)
    findings.append(crossings)

    measured: dict[str, Any] = {
        "directory": str(directory),
        "manifest": {
            "source": manifest.get("source"),
            "unit_metres": manifest.get("unit_metres"),
            "location_in_mesh": manifest.get("location_in_mesh"),
            "patches": len(manifest_entries(manifest)),
        },
        "patches": per_file,
        "triangles": int(len(union.triangles)),
        "vertices": vertex_count,
        "union_topology": topology,
        "topology_triangle_limit": limit,
        "worst": None,
        "counts": None,
    }
    if len(union.triangles):
        flat = union.triangles.reshape(-1, 3)
        measured["bounding_box"] = {
            "min": flat.min(axis=0).tolist(),
            "max": flat.max(axis=0).tolist(),
        }
    if crossing_data:
        measured["self_intersection"] = {
            "pairs": len(crossing_data["pairs"]),
            "triangles": len(crossing_data["triangles"]),
            "pairs_tested": crossing_data["tested"],
            "box_overlapping_pairs": crossing_data["candidates"],
            "triangle_ids": crossing_data["triangles"][:256],
        }

    if surface_check:
        finding, extra = surface_check_finding(union)
        findings.append(finding)
        measured.update(extra)

    measured["worst"] = preflight.worst_status(findings)
    measured["counts"] = preflight.summarise(findings)
    return {
        "script": "cad_audit",
        "ok": preflight.worst_status(findings) != "fail",
        "findings": [finding.as_dict() for finding in findings],
        "measured": measured,
    }


def findings_of(report: dict[str, Any]) -> list[Finding]:
    return [
        Finding(row["check"], row["status"], row["measured"],
                row.get("means", ""), row.get("repair", ""))
        for row in report["findings"]
    ]


def render(report: dict[str, Any]) -> str:
    findings = findings_of(report)
    counts = preflight.summarise(findings)
    lines = [f"# cad_audit {report['measured']['directory']}"]
    lines.append(
        "  ".join(
            f"{status} {counts[status]}"
            for status in preflight.STATUSES if counts.get(status)
        )
        or "nothing measured"
    )
    for finding in findings:
        lines.append("")
        lines.append(f"{finding.status.upper():<9}{finding.check}")
        lines.append(f"  measured  {finding.measured}")
        if finding.meaning:
            lines.append(f"  means     {finding.meaning}")
        if finding.repair:
            lines.append(f"  repair    {finding.repair}")
    lines.append("")
    lines.append(
        "Every topological number above is the union's. The patch files are open "
        "surfaces one at a time, and that is what a correct export looks like."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The exported patch set measured -- closure of the union, not of "
                    "the parts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The union of the patch files is what has to be closed, manifold and\n"
            "consistently wound; the files one at a time are open surfaces and their\n"
            "open-edge counts are printed only so that is visible.\n"
            "Exit code is 0 whatever is found."
        ),
    )
    parser.add_argument("directory", type=Path,
                        help="a <case>/constant/triSurface with patches.json in it")
    parser.add_argument("--json", action="store_true", help="the report as JSON")
    parser.add_argument("--surface-check", action="store_true",
                        help="also run OpenFOAM's surfaceCheck on each patch file")
    args = parser.parse_args(argv)

    try:
        report = audit(args.directory, surface_check=args.surface_check)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
