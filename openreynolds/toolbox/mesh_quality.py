#!/usr/bin/env python3
"""The bad cells themselves: where they are, what they look like, and a picture.

The quality numbers were already here three times over and the cells were here
nowhere. `mesh_digest.py` turns a `checkMesh` log into a metric table and attaches no
thresholds on purpose; `preflight.py` applies the conventional thresholds to the same
numbers and suggests a repair; `results.py` draws histograms of whatever quality
measures the VTK build will compute. All three answer "how bad", and none of them
answers "bad where", which is the question that decides what to do about it: max
skewness 11 in four cells on a trailing edge is a local snap artefact a solver will
survive, and the same 11 in a band across the middle of the domain is a mesh to throw
away. The two look identical in every report in this toolbox and completely different
in a picture.

So this is the locator for a mesh metric, built the way `locate.py` is the locator for
a diverging field. `locate.py` reads `constant/polyMesh` outright, ascii or binary,
and reports a cell index, its coordinates, the patch it belongs to and its distance to
the boundary; `heat.py` reuses those readers for hot spots and reads no mesh file
itself. This reads no mesh file itself either -- `locate.read_mesh` loads it,
`layer_report.face_geometry` and `layer_report.cell_centres` do the geometry by
OpenFOAM's own decompositions, and `heat.nearest_patch` answers "near what" for a cell
in the middle of the domain.

which route produced the picture, said on the picture

There are two ways to get the offending cells and they are not equally good:

1. **the set route.** `checkMesh -writeSets vtk` leaves the cells and faces that failed
   its own checks as `cellSet`/`faceSet` files under `constant/polyMesh/sets` (or under
   a time directory's `polyMesh/sets`), plus a surface copy under `VTK/face-set/<name>/`.
   Three names are attested on this image from live runs in the wing campaign of
   2026-10-09 -- `skewFaces`, `nonOrthoFaces` and `wrongOrientedFaces`, all read out of
   `constant/polyMesh/sets` -- and whether that is the complete list was not
   established, which is the whole reason nothing here is looked up by name. These are
   the cells that failed, by definition, with no second opinion involved. Preferred
   wherever they are on disk.
2. **the computed route.** Skewness and non-orthogonality are recomputed here from the
   mesh, by the formulae in OpenFOAM's `primitiveMeshTools` -- the skewness vector
   normalised by the face's own extent along it, and the angle between the face area
   vector and the owner-to-neighbour vector. Available always, and it is its own
   measurement rather than `checkMesh`'s.

Which one drew the picture is printed in the report, in the JSON and as a caption on
the PNG, because the two can disagree: a recomputed measure that happens not to be the
one a check failed on would put a red cell in the wrong place and look just as
convincing. The set names are read off whatever is in the sets directory rather than
hardcoded, since which names an ESI v2512 `checkMesh` writes was not determined
without running it -- so a name this file has never heard of is still read, still
counted and still drawn, and only its mapping onto a named metric is guessed, by
substring, and reported as a guess.

Note what is NOT recomputed: `results.py` asks VTK for `skew`, `aspect_ratio`,
`scaled_jacobian` and `max_angle`, which are VTK's definitions on VTK's cell types.
They are fine for a histogram and they are not OpenFOAM's numbers, so they cannot be
compared against a `checkMesh` threshold. Aspect ratio here is read from the log or
from a set and is never recomputed, for exactly that reason.

conversion, and what this image cannot reach

The second half is mesh conversion, and the honest part of it is the list of formats
that are out of reach. `gmshToFoam` and `foamToVTK` are in use elsewhere in the
toolbox already; Fluent and CGNS appear in this repository only as strings in a phase
classifier's vocabulary. So the converters are probed with `shutil.which` the way
`cfmesh.py`'s `where` probes for its meshers, in both directions, and what is missing
is named rather than guessed around. CGNS in particular has no converter in the
OpenFOAM distribution at all, in either direction: that is a statement about OpenFOAM
and not about this image, and no flag here will make one appear.

Nothing here edits a mesh, runs a solver or decides anything. It reports and it draws.
Exit code is 0 whatever it finds, including when it finds nothing wrong.

    python3 mesh_quality.py report /work/case
    python3 mesh_quality.py report /work/case --png bad.png --json bad.json
    python3 mesh_quality.py report /work/case --skewness 6 --worst 10
    python3 mesh_quality.py formats
    python3 mesh_quality.py convert duct.msh --from fluent --to openfoam --case /work/duct
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heat  # noqa: E402  (sibling scripts, not a package)
import layer_report  # noqa: E402
import locate  # noqa: E402
import mesh_digest  # noqa: E402
import preflight  # noqa: E402

Finding = preflight.Finding

TINY = 1e-300
"""Stands in for OpenFOAM's ROOTVSMALL in the same denominators it guards there. A
degenerate face gives a zero denominator in both skewness and non-orthogonality, and
a nan in a metric silently drops the cell out of every comparison that follows."""

WORST = 5
"""How many cells per metric are located in full. Each one costs a pass over every
boundary face to find the patch nearest it, so this is a count rather than a cap on a
fraction: on a 10M-cell mesh the hundredth worst cell is not news."""

MAX_DRAW = 20000
"""Bad cells drawn before the picture is truncated. 20k cells is already a solid red
region at any sensible window size, and the faces of more than that cost more to build
than the picture adds. A truncated picture says so on itself."""


class QualityError(Exception):
    """A measurement that could not be made, reported rather than raised past `main`."""


# -- the metrics, and whose thresholds they are ------------------------------------


METRICS: tuple[tuple[str, str, float, float | None, bool], ...] = (
    # (mesh_digest key, label, warn, fail, recomputable here)
    ("skewness", "max skewness", preflight.SKEWNESS_WARN, preflight.SKEWNESS_FAIL, True),
    ("non_ortho", "non-orthogonality", preflight.NON_ORTHO_WARN, preflight.NON_ORTHO_FAIL, True),
    ("aspect_ratio", "max aspect ratio", preflight.ASPECT_WARN, None, False),
)
"""The thresholds are `preflight.py`'s constants, imported rather than copied: a second
copy of SKEWNESS_WARN is a second number to disagree with the gate about. Aspect ratio
is marked unrecomputable because OpenFOAM's cell aspect ratio is not the ratio of a
bounding box and guessing it would put a red cell where checkMesh never found one."""

SET_HINTS: tuple[tuple[str, str], ...] = (
    ("skew", "skewness"),
    ("nonortho", "non_ortho"),
    ("aspectratio", "aspect_ratio"),
)
"""Substring guesses from a set's name onto a metric, matched case-folded with the
non-letters removed. A set whose name matches nothing is still read and still drawn --
it is named cells that failed a check, which is the whole point -- it is just reported
under its own name with no metric attached."""


def grade(value: float | None, warn: float, fail: float | None) -> str:
    """`ok`, `warn` or `fail` for one number, on preflight's thresholds."""
    if value is None:
        return "unknown"
    if fail is not None and value > fail:
        return "fail"
    return "warn" if value > warn else "ok"


def _first(value: Any) -> float | None:
    """The leading number of a `mesh_digest` entry, which is a tuple for the metrics
    it reports as a pair (max and average non-orthogonality, for instance)."""
    if isinstance(value, tuple):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        return float(str(value).rstrip("."))
    except ValueError:
        return None


def metric_from_set(name: str) -> str:
    folded = re.sub(r"[^a-z]", "", name.lower())
    for hint, metric in SET_HINTS:
        if hint in folded:
            return metric
    return ""


# -- the log ----------------------------------------------------------------------


def find_log(case: Path, region: str = "") -> Path | None:
    """The `checkMesh` log a case already has, by the names the toolbox writes.

    `mesh_look.py` writes `log.checkMesh` (and `log.checkMesh.<region>`) beside the
    case and `preflight.py` reads the same name, so a case that has been looked at has
    one of these and nothing needs to be re-run to read it.
    """
    names = [f"log.checkMesh.{region}", "log.checkMesh"] if region else ["log.checkMesh"]
    for name in names:
        path = case / name
        if path.is_file():
            return path
    return None


def read_log(path: Path | None) -> dict[str, Any]:
    """`mesh_digest.parse` on the log, or an empty parse. Not a second parser: the
    digest already reads every number checkMesh prints, including the one-star
    advisories that used to be missed."""
    if path is None:
        return {"counts": {}, "patches": [], "failures": [], "topology": []}
    return mesh_digest.parse(path.read_text(encoding="utf-8", errors="replace"))


# -- the set route ----------------------------------------------------------------


def set_dirs(case: Path, region: str = "") -> list[Path]:
    """Every `polyMesh/sets` directory a case could be carrying them in.

    `constant/polyMesh/sets` is where a mesh-only case keeps them; a case that has run
    keeps topoSet output under the time directory the sets were written at, so the time
    directories are looked in too rather than only the constant one.
    """
    mesh_dir = Path("constant") / region / "polyMesh" if region else Path("constant/polyMesh")
    out = [case / mesh_dir / "sets"]
    for _, folder in locate.time_dirs(case):
        candidate = folder / region / "polyMesh" / "sets" if region else folder / "polyMesh" / "sets"
        out.append(candidate)
    return [path for path in out if path.is_dir()]


def read_sets(case: Path, region: str = "") -> list[dict[str, Any]]:
    """Every cellSet and faceSet on disk, by whatever name it is under.

    The names an ESI v2512 `checkMesh -writeSets` writes were not established here, so
    nothing is looked up by name: the directory is listed, each file's `class` is read
    out of its own header, and the labels are read with `locate.read_list`, which
    handles the binary case a `writeFormat binary` mesh leaves them in. A set this file
    has never heard of therefore still arrives, with `metric` empty.
    """
    out: list[dict[str, Any]] = []
    for folder in set_dirs(case, region):
        for path in sorted(folder.iterdir()):
            if path.is_dir() or path.name.endswith(".orig"):
                continue
            name = path.name[:-3] if path.name.endswith(".gz") else path.name
            fobj = locate.foam_file(path)
            if fobj is None or fobj.cls not in ("cellSet", "faceSet"):
                continue
            try:
                labels, _ = locate.read_list(fobj, fobj.body_at, "label")
            except locate.ListError as error:
                out.append({"set": name, "kind": fobj.cls, "where": folder.as_posix(),
                            "count": None, "metric": "", "problem": str(error)})
                continue
            out.append({
                "set": name,
                "kind": fobj.cls,
                "where": folder.as_posix(),
                "count": int(labels.size),
                "metric": metric_from_set(name),
                "labels": np.asarray(labels, dtype=np.int64),
            })
    return out


def cells_of_set(mesh: dict[str, Any], entry: dict[str, Any]) -> np.ndarray:
    """The cells a set names, whether it named cells or faces.

    A `faceSet` is turned into the cells on both sides of its faces, because a face is
    not a thing anybody can see in a picture of a mesh and the cell it belongs to is.
    """
    labels = entry.get("labels")
    if labels is None or labels.size == 0:
        return np.zeros(0, dtype=np.int64)
    if entry["kind"] == "cellSet":
        return np.unique(labels[(labels >= 0) & (labels < mesh["n_cells"])])
    owner, neighbour = mesh["owner"], mesh["neighbour"]
    faces = labels[(labels >= 0) & (labels < owner.size)]
    cells = [owner[faces]]
    internal = faces[faces < neighbour.size]
    if internal.size:
        cells.append(neighbour[internal])
    return np.unique(np.concatenate(cells))


# -- the computed route -----------------------------------------------------------


def cell_centre_field(mesh: dict[str, Any]) -> np.ndarray:
    """Every cell centre, which both face metrics need on both sides of every face.

    `layer_report.cell_centres` builds only the cells asked for, and here that is all
    of them, so this is the one unavoidably whole-mesh step.
    """
    # ponytail: one pass over every face in numpy, which is seconds at a few million
    # cells and minutes beyond that. A metric computed on a subset of the mesh is the
    # upgrade path, and it needs a cell selection this script has no way to be given.
    return layer_report.cell_centres(mesh, np.arange(mesh["n_cells"], dtype=np.int64))


def _vertex_extent(mesh: dict[str, Any], faces: np.ndarray, centres: np.ndarray,
                   direction: np.ndarray) -> np.ndarray:
    """How far a face reaches from its own centre along `direction`, per face.

    This is the normalisation in OpenFOAM's skewness: the skewness vector is divided by
    the face's extent in the skewness direction, so the same absolute offset counts for
    more on a small face than on a large one. Grouped by vertex count so the arithmetic
    stays vectorised, which is `layer_report.face_geometry`'s own shape.
    """
    out = np.zeros(faces.size)
    offsets = mesh["offsets"]
    sizes = offsets[faces + 1] - offsets[faces]
    for size in np.unique(sizes):
        rows = np.flatnonzero(sizes == size)
        cols = offsets[faces[rows]][:, None] + np.arange(int(size))[None, :]
        points = mesh["points"][mesh["verts"][cols]]
        offset = points - centres[rows][:, None, :]
        out[rows] = np.abs((offset * direction[rows][:, None, :]).sum(axis=2)).max(axis=1)
    return out


def face_metrics(mesh: dict[str, Any]) -> dict[str, np.ndarray]:
    """Skewness per face and non-orthogonality per internal face, OpenFOAM's way.

    Both are the formulae in `primitiveMeshTools`, not approximations of them, because
    the number has to be comparable with the threshold `checkMesh` failed on. Skewness
    is the component of the face-centre offset that the owner-to-neighbour line does
    not explain, normalised by the face's extent along it; a boundary face has no
    neighbour, so the wall-normal projection of the offset stands in for that line,
    which is what OpenFOAM does for the same reason. Non-orthogonality is the angle
    between the face area vector and the owner-to-neighbour vector, and exists only
    where there is a neighbour.
    """
    n_faces = int(mesh["offsets"].size - 1)
    n_internal = int(mesh["neighbour"].size)
    faces = np.arange(n_faces, dtype=np.int64)
    face_centres, areas = layer_report.face_geometry(mesh, faces)
    centres = cell_centre_field(mesh)

    owner_centres = centres[mesh["owner"]]
    offset = face_centres - owner_centres

    line = np.zeros((n_faces, 3))
    line[:n_internal] = centres[mesh["neighbour"]] - owner_centres[:n_internal]
    if n_faces > n_internal:
        magnitude = np.linalg.norm(areas[n_internal:], axis=1)
        normals = areas[n_internal:] / np.maximum(magnitude, TINY)[:, None]
        wall = offset[n_internal:]
        line[n_internal:] = normals * (normals * wall).sum(axis=1)[:, None]

    along = (areas * line).sum(axis=1)
    across = (areas * offset).sum(axis=1)
    skew_vector = offset - (across / (along + TINY))[:, None] * line
    skew_magnitude = np.linalg.norm(skew_vector, axis=1)
    direction = skew_vector / np.maximum(skew_magnitude, TINY)[:, None]

    line_length = np.linalg.norm(line, axis=1)
    weight = np.where(np.arange(n_faces) < n_internal, 0.2, 0.4)
    extent = np.maximum(weight * line_length + TINY,
                        _vertex_extent(mesh, faces, face_centres, direction))
    skewness = skew_magnitude / extent

    cosine = along[:n_internal] / np.maximum(
        np.linalg.norm(areas[:n_internal], axis=1) * line_length[:n_internal], TINY
    )
    non_ortho = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {"skewness": skewness, "non_ortho": non_ortho}


def cells_above(mesh: dict[str, Any], values: np.ndarray, threshold: float) -> np.ndarray:
    """The cells on either side of every face whose value is over the threshold.

    Both sides, deliberately: a skewed face is shared, and a picture that drew only the
    owner would show half of a bad region for no reason a reader could see.
    """
    bad = np.flatnonzero(values > threshold)
    if bad.size == 0:
        return np.zeros(0, dtype=np.int64)
    owner, neighbour = mesh["owner"], mesh["neighbour"]
    cells = [owner[bad]]
    internal = bad[bad < neighbour.size]
    if internal.size:
        cells.append(neighbour[internal])
    return np.unique(np.concatenate(cells))


# -- where the cells are ----------------------------------------------------------


def patch_tally(mesh: dict[str, Any], cells: np.ndarray) -> dict[str, int]:
    """How many of these cells have a face on each patch, worst patch first.

    One pass per patch over its own faces, rather than one pass over every boundary
    face per cell: the tally is over the whole selection, which on a real mesh is
    thousands of cells, and the per-cell question is only asked of the worst few.
    """
    wanted = np.zeros(mesh["n_cells"], dtype=bool)
    wanted[cells] = True
    owner = mesh["owner"]
    tally: dict[str, int] = {}
    for name, entry in mesh["boundary"].items():
        low = int(entry["startFace"])
        owners = owner[low: low + int(entry["nFaces"])]
        if owners.size == 0:
            continue
        hit = np.unique(owners[wanted[owners]])
        if hit.size:
            tally[name] = int(hit.size)
    return dict(sorted(tally.items(), key=lambda item: -item[1]))


def place(mesh: dict[str, Any], cells: np.ndarray, values: np.ndarray | None,
          worst: int = WORST) -> dict[str, Any]:
    """Coordinates, patches and extent for a set of cells -- `locate.py`'s answer.

    The per-cell part is the same three measurements `locate.py` makes of a diverging
    field's extremum: the cell centre, the patch its own boundary faces belong to (and
    the nearest patch when it has none, which is `heat.nearest_patch`), and its
    distance to the domain bounding box. The whole-set part is the box the bad cells
    span, as a fraction of the domain in each direction, which is the number that says
    sliver against band without anybody having to look at the picture first.
    """
    cells = np.asarray(cells, dtype=np.int64)
    out: dict[str, Any] = {"cells": int(cells.size), "worst": [], "patches": {}}
    if cells.size == 0:
        return out
    # `locate.cell_patches` walks every boundary face to answer for ONE cell, which is
    # the right shape for an extremum and quadratic over a selection of thousands. It
    # is kept for the worst few, where its answer is the ordered patch list, and the
    # tally over the whole selection goes the other way round.
    out["fraction_of_mesh"] = round(float(cells.size) / max(mesh["n_cells"], 1), 6)
    centres = layer_report.cell_centres(mesh, cells)
    low = mesh["points"].min(axis=0)
    high = mesh["points"].max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    box_low, box_high = centres.min(axis=0), centres.max(axis=0)
    out["box"] = {
        "min": [round(float(v), 9) for v in box_low],
        "max": [round(float(v), 9) for v in box_high],
        "span_fraction_of_domain": [round(float(v), 4) for v in (box_high - box_low) / span],
    }

    order = np.argsort(values)[::-1] if values is not None and values.size == cells.size \
        else np.arange(cells.size)
    out["patches"] = patch_tally(mesh, cells)

    for row in order[:max(worst, 0)]:
        cell = int(cells[row])
        centre = centres[row]
        on = locate.cell_patches(mesh, cell)
        record: dict[str, Any] = {
            "cell": cell,
            "centre": [round(float(v), 9) for v in centre],
            "on_patches": on,
        }
        if values is not None and values.size == cells.size:
            record["value"] = round(float(values[row]), 4)
        if on:
            record["patch"] = on[0]
            record["distance_to_patch"] = 0.0
        else:
            near, gap = heat.nearest_patch(mesh, centre)
            record["patch"] = near or "(no patch)"
            record["distance_to_patch"] = round(gap, 9) if near else None
        record.update(locate.bbox_facts(mesh, centre))
        out["worst"].append(record)
    return out


def cell_values(mesh: dict[str, Any], cells: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The worst face value touching each cell, so the worst cells can be ordered.

    The value array is as long as the faces it exists for, which is every face for
    skewness and the internal faces alone for non-orthogonality: a boundary face has
    no neighbour and so no angle to be off. Both are therefore taken against the
    length of the values rather than the length of the mesh.
    """
    owner, neighbour = mesh["owner"], mesh["neighbour"]
    count = int(values.size)
    per_cell = np.zeros(mesh["n_cells"])
    np.maximum.at(per_cell, owner[:count], values[:count])
    shared = min(int(neighbour.size), count)
    if shared:
        np.maximum.at(per_cell, neighbour[:shared], values[:shared])
    return per_cell[cells]


# -- the measurement --------------------------------------------------------------


def measure(case: Path, *, region: str = "", log: Path | None = None,
            thresholds: dict[str, float] | None = None,
            worst: int = WORST) -> dict[str, Any]:
    """The whole report as data: the log's numbers, the sets on disk, the cells.

    The two routes are tried in that order per metric and the one that answered is
    recorded on the metric, never averaged with the other: they are different
    measurements and a reader who cannot tell which they are looking at cannot check
    either.
    """
    case = Path(case)
    found: dict[str, Any] = {
        "case": case.as_posix(), "region": region, "problems": [],
        "log": None, "metrics": [], "sets": [], "picture": None,
    }
    log_path = log if log is not None else find_log(case, region)
    if log_path is not None and not Path(log_path).is_file():
        found["problems"].append(f"no such checkMesh log: {Path(log_path).as_posix()}")
        log_path = None
    found["log"] = Path(log_path).as_posix() if log_path else None
    if log_path is None:
        found["problems"].append(
            "no log.checkMesh here, so the log's own numbers are absent and only the "
            "recomputed metrics below are reported"
        )
    data = read_log(Path(log_path) if log_path else None)
    found["counts"] = data.get("counts", {})
    found["verdict"] = preflight.checkmesh_verdict(data)
    found["failures"] = data.get("failures", [])
    found["warnings"] = data.get("warnings", [])

    poly = case / "constant" / region / "polyMesh" if region else case / "constant" / "polyMesh"
    found["polyMesh"] = poly.as_posix()
    mesh = locate.read_mesh(poly)
    if isinstance(mesh, str):
        found["problems"].append(f"the mesh could not be read: {mesh}")
        found["mesh"] = mesh
        return found
    found["mesh"] = f"{mesh['n_cells']:,} cells, {len(mesh['boundary'])} patches"
    found["n_cells"] = int(mesh["n_cells"])

    sets = read_sets(case, region)
    found["sets"] = [
        {key: value for key, value in entry.items() if key != "labels"} for entry in sets
    ]
    by_metric = {entry["metric"]: entry for entry in sets if entry["metric"]}

    computed: dict[str, np.ndarray] = {}
    wanted = thresholds or {}
    # Recomputed wherever a set did not already name the cells, log or no log: a
    # report that waited for `checkMesh` to have printed a number first would have
    # nothing to say about a mesh nobody has run it on, which is the mesh most worth
    # looking at.
    if any(recomputable and key not in by_metric for key, _, _, _, recomputable in METRICS):
        try:
            computed = face_metrics(mesh)
        except (ValueError, MemoryError) as error:
            found["problems"].append(f"the face metrics could not be computed: {error}")

    for key, label, warn, fail, recomputable in METRICS:
        value = _first(data.get(key))
        threshold = wanted.get(key, warn)
        row: dict[str, Any] = {
            "metric": key, "label": label, "log_value": value,
            "warn": warn, "fail": fail, "status": grade(value, warn, fail),
            "threshold": threshold, "route": "none", "cells": 0,
        }
        entry = by_metric.get(key)
        if entry is not None and entry.get("labels") is not None:
            cells = cells_of_set(mesh, entry)
            row["route"] = "set"
            row["from_set"] = entry["set"]
            row["detail"] = (
                f"{entry['set']} ({entry['kind']}) in {entry['where']}, "
                f"{entry['count']} label(s)"
            )
            row.update(place(mesh, cells, None, worst))
            row["drawable"] = cells
        elif recomputable and key in computed:
            values = computed[key]
            cells = cells_above(mesh, values, threshold)
            row["route"] = "computed"
            row["measured_max"] = round(float(values.max(initial=0.0)), 4)
            row["detail"] = (
                f"recomputed here: max {row['measured_max']:g}, "
                f"{int((values > threshold).sum())} face(s) over {threshold:g}"
            )
            row.update(place(mesh, cells, cell_values(mesh, cells, values), worst))
            row["drawable"] = cells
            if value is None:
                # No log to grade, so the status is graded on the number measured
                # here and is labelled as such. A metric reported `unknown` beside
                # forty located cells is a row that argues with itself.
                row["status"] = grade(row["measured_max"], warn, fail)
                row["status_from"] = "recomputed here"
        else:
            row["detail"] = (
                "no set on disk and not recomputed here"
                if recomputable else
                "read from the log or from a set only; OpenFOAM's cell aspect ratio is "
                "not recomputed here, since a different definition would place cells "
                "checkMesh never flagged"
            )
            if not recomputable:
                row["note"] = "checkMesh -writeSets vtk leaves the cells it flagged in polyMesh/sets"
        found["metrics"].append(row)

    for entry in sets:
        if entry["metric"] or entry.get("labels") is None:
            continue
        cells = cells_of_set(mesh, entry)
        found["metrics"].append({
            "metric": entry["set"], "label": f"{entry['set']} ({entry['kind']})",
            "log_value": None, "warn": None, "fail": None, "status": "set",
            "threshold": None, "route": "set", "from_set": entry["set"],
            "detail": f"a set under {entry['where']} that maps to no named metric here",
            "drawable": cells, **place(mesh, cells, None, worst),
        })

    found["_mesh"] = mesh
    return found


def worst_metric(found: dict[str, Any]) -> dict[str, Any] | None:
    """The metric the picture is drawn for: the one with cells, worst status first."""
    rank = {"fail": 0, "warn": 1, "set": 2, "ok": 3, "unknown": 4}
    have = [row for row in found.get("metrics", []) if row.get("cells")]
    if not have:
        return None
    return sorted(have, key=lambda row: (rank.get(row["status"], 5), -row["cells"]))[0]


# -- the picture ------------------------------------------------------------------


def face_polydata(mesh: dict[str, Any], faces: np.ndarray):
    """The named faces as a pyvista `PolyData`, straight out of the polyMesh.

    Built here rather than read back through `pyvista.OpenFOAMReader` on purpose. The
    reader decomposes polyhedra by default, so its cell numbering is not the mesh's
    wherever a snappy cell was split, and a red cell drawn at the wrong index is a
    picture that lies. These are the mesh's own faces at the mesh's own indices.
    """
    import pyvista as pv

    faces = np.asarray(faces, dtype=np.int64)
    offsets, verts = mesh["offsets"], mesh["verts"]
    if faces.size == 0:
        return pv.PolyData()
    sizes = offsets[faces + 1] - offsets[faces]
    total = int(sizes.sum())
    flat = np.empty(total + faces.size, dtype=np.int64)
    starts = np.cumsum(np.concatenate([[0], sizes + 1]))[:-1]
    flat[starts] = sizes
    position = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    gather = np.repeat(offsets[faces], sizes) + position
    body = np.ones(flat.size, dtype=bool)
    body[starts] = False
    flat[body] = verts[gather]
    return pv.PolyData(mesh["points"], faces=flat)


def faces_of_cells(mesh: dict[str, Any], cells: np.ndarray) -> np.ndarray:
    """Every face of the named cells, interior faces included."""
    cells = np.asarray(cells, dtype=np.int64)
    if cells.size == 0:
        return np.zeros(0, dtype=np.int64)
    wanted = np.zeros(mesh["n_cells"], dtype=bool)
    wanted[cells] = True
    owner, neighbour = mesh["owner"], mesh["neighbour"]
    hit = wanted[owner]
    if neighbour.size:
        hit[:neighbour.size] |= wanted[neighbour]
    return np.flatnonzero(hit)


def boundary_faces(mesh: dict[str, Any]) -> np.ndarray:
    """The faces on the domain boundary, which is the context the picture needs."""
    n_internal = int(mesh["neighbour"].size)
    return np.arange(n_internal, int(mesh["offsets"].size - 1), dtype=np.int64)


def one_ring(mesh: dict[str, Any], cells: np.ndarray) -> np.ndarray:
    """The cells sharing a face with the named ones, so the close-up has a mesh in it.

    A bad cell drawn alone is a red blob at an unknown scale. Its neighbours give the
    reader the local cell size to judge it against, which is the difference between
    "a sliver" and "every cell here looks like that".
    """
    cells = np.asarray(cells, dtype=np.int64)
    if cells.size == 0:
        return cells
    wanted = np.zeros(mesh["n_cells"], dtype=bool)
    wanted[cells] = True
    owner, neighbour = mesh["owner"], mesh["neighbour"]
    internal = np.arange(neighbour.size)
    pick = wanted[owner[internal]] | wanted[neighbour[internal]]
    return np.unique(np.concatenate([cells, owner[internal][pick], neighbour[internal][pick]]))


def draw(found: dict[str, Any], out_png: Path, *, max_draw: int = MAX_DRAW) -> dict[str, Any]:
    """Two panels: the bad cells in the whole mesh, and the same cells close up.

    Two, because one cannot carry both facts. The wide panel answers "where in the
    domain", which is what separates a trailing-edge artefact from a band through the
    middle; the close-up answers "what do they look like next to their neighbours",
    which is what separates a sliver from a region that is simply coarse. The route
    that produced the selection is written on the image, so a picture that outlives
    this report still says where its red came from.

    The check is `docker/workspace/check.sh`'s: a render whose pixels have zero spread
    is a blank image, and a blank PNG written without complaint is worse than no PNG.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    mesh = found.get("_mesh")
    row = worst_metric(found)
    if not isinstance(mesh, dict) or row is None:
        raise QualityError("no cells to draw")
    cells = np.asarray(row.get("drawable", np.zeros(0, np.int64)), dtype=np.int64)
    if cells.size == 0:
        raise QualityError("no cells to draw")
    truncated = cells.size > max_draw
    if truncated:
        cells = cells[:max_draw]

    bad = face_polydata(mesh, faces_of_cells(mesh, cells))
    context = face_polydata(mesh, boundary_faces(mesh))
    ring = face_polydata(mesh, faces_of_cells(mesh, one_ring(mesh, cells)))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 800), border=False)

    plotter.subplot(0, 0)
    if context.n_cells:
        plotter.add_mesh(context, color="white", opacity=0.22, show_edges=False)
    plotter.add_mesh(bad, color="crimson", show_edges=True, edge_color="black", line_width=0.4)
    plotter.add_text(
        f"{row['label']}: {row['cells']} cell(s) in the domain", font_size=9
    )
    plotter.view_isometric()
    plotter.reset_camera()

    plotter.subplot(0, 1)
    if ring.n_cells:
        plotter.add_mesh(ring, color="white", opacity=0.45, show_edges=True,
                         edge_color="dimgray", line_width=0.4)
    plotter.add_mesh(bad, color="crimson", show_edges=True, edge_color="black", line_width=0.6)
    caption = f"close up, selected by the {row['route']} route"
    if row.get("from_set"):
        caption += f" ({row['from_set']})"
    elif row.get("threshold") is not None:
        caption += f" (over {row['threshold']:g})"
    if truncated:
        caption += f"; drawn {max_draw} of {row['cells']}"
    plotter.add_text(caption, font_size=9)
    plotter.view_isometric()
    plotter.reset_camera()

    plotter.screenshot(str(out_png))
    plotter.close()
    return {
        "picture": out_png.resolve().as_posix(),
        "drawn_metric": row["metric"],
        "drawn_route": row["route"],
        "drawn_cells": int(cells.size),
        "truncated": truncated,
    }


# -- conversion -------------------------------------------------------------------


CONVERTERS: dict[str, dict[str, tuple[str, ...]]] = {
    # format: {to_foam: (candidate binaries), from_foam: (candidate binaries)}
    "fluent": {
        "to_foam": ("fluentMeshToFoam", "fluent3DMeshToFoam"),
        "from_foam": ("foamMeshToFluent",),
    },
    "gmsh": {"to_foam": ("gmshToFoam",), "from_foam": ()},
    "vtk": {"to_foam": ("vtkUnstructuredToFoam",), "from_foam": ("foamToVTK",)},
    "cgns": {"to_foam": (), "from_foam": ()},
    "unv": {"to_foam": ("ideasUnvToFoam",), "from_foam": ()},
    "cfx4": {"to_foam": ("cfx4ToFoam",), "from_foam": ()},
    "star-ccm": {"to_foam": ("star4ToFoam",), "from_foam": ("foamToStarMesh",)},
}
"""Candidate binaries per format and direction, probed rather than assumed.

An empty tuple is a claim about OpenFOAM and not about this image: CGNS has no
converter in the distribution in either direction, and there is no `foamToGmsh`. Those
stay empty however the probe comes out, because no binary on any PATH will fill them.
"""

NO_CONVERTER_ANYWHERE: dict[str, str] = {
    "cgns": "OpenFOAM ships no CGNS converter in either direction; cgns appears nowhere "
            "in this repository either, and nothing here will make one appear",
}
"""Formats whose absence is not this image's doing, said once and said plainly."""


def probe() -> dict[str, Any]:
    """What of the mesh converters is on this instance, by `shutil.which`.

    `cfmesh.py`'s `where` is the pattern: ask the PATH rather than the documentation,
    report the answer including the empty parts, and name what was looked for so a
    reader can check the probe itself.
    """
    out: dict[str, Any] = {"formats": {}, "reachable": [], "unreachable": []}
    for name, directions in sorted(CONVERTERS.items()):
        entry: dict[str, Any] = {}
        for direction, candidates in directions.items():
            found = [
                {"name": binary, "path": shutil.which(binary)}
                for binary in candidates
            ]
            entry[direction] = {
                "looked_for": list(candidates),
                "found": [row for row in found if row["path"]],
                "missing": [row["name"] for row in found if not row["path"]],
                "reachable": any(row["path"] for row in found),
            }
        entry["note"] = NO_CONVERTER_ANYWHERE.get(name, "")
        out["formats"][name] = entry
        if entry["to_foam"]["reachable"] or entry["from_foam"]["reachable"]:
            out["reachable"].append(name)
        else:
            out["unreachable"].append(name)
    return out


def report_formats(found: dict[str, Any]) -> str:
    lines = ["mesh conversion, as this instance's PATH answers it", ""]
    lines.append(f"  {'format':<10}{'to OpenFOAM':<46}{'from OpenFOAM'}")
    for name, entry in found["formats"].items():
        def column(direction: str) -> str:
            side = entry[direction]
            if side["found"]:
                return ", ".join(row["name"] for row in side["found"])
            if not side["looked_for"]:
                return "no converter exists"
            return "missing: " + ", ".join(side["missing"])
        lines.append(f"  {name:<10}{column('to_foam'):<46}{column('from_foam')}")
    lines.append("")
    lines.append(f"  reachable here:   {', '.join(found['reachable']) or 'none'}")
    lines.append(f"  not reachable:    {', '.join(found['unreachable']) or 'none'}")
    for name, entry in found["formats"].items():
        if entry["note"]:
            lines.append(f"  {name}: {entry['note']}")
    lines.append("")
    lines.append("  openfoam is the hub: every row above is a conversion through it, "
                 "and a format with")
    lines.append("  no binary in either column cannot be reached from here by any "
                 "route in this image.")
    return "\n".join(lines)


def convert(source: Path, source_format: str, target_format: str, *,
            case: Path | None = None, timeout: int = 1800) -> dict[str, Any]:
    """One converter run, or the reason there is none to run.

    A missing converter is reported by the name that was looked for and exits 0: a
    format that is not in this image is a fact about the image, not a failure of the
    call that asked about it.
    """
    out: dict[str, Any] = {
        "source": Path(source).as_posix(), "from": source_format, "to": target_format,
        "converter": None, "command": [], "ok": False, "problems": [],
    }
    if source_format == target_format:
        out["problems"].append(f"{source_format} to {source_format} is not a conversion")
        return out
    if "openfoam" not in (source_format, target_format):
        out["problems"].append(
            f"{source_format} to {target_format} is not a direct conversion here; "
            "OpenFOAM is the hub, so it is two calls through it"
        )
        return out

    other = target_format if source_format == "openfoam" else source_format
    direction = "from_foam" if source_format == "openfoam" else "to_foam"
    directions = CONVERTERS.get(other)
    if directions is None:
        out["problems"].append(f"unknown format '{other}'; known: "
                               + ", ".join(sorted(CONVERTERS)) + ", openfoam")
        return out
    candidates = directions[direction]
    if not candidates:
        note = NO_CONVERTER_ANYWHERE.get(other, "")
        out["problems"].append(
            f"no OpenFOAM converter exists for {other} in this direction"
            + (f": {note}" if note else "")
        )
        out["unreachable"] = True
        return out

    chosen = next((name for name in candidates if shutil.which(name)), None)
    if chosen is None:
        out["problems"].append(
            "none of " + ", ".join(candidates) + " is on PATH here, so "
            f"{source_format} to {target_format} is not reachable on this instance"
        )
        out["unreachable"] = True
        out["looked_for"] = list(candidates)
        return out

    out["converter"] = chosen
    if direction == "to_foam":
        # `source` is the foreign mesh file and the case is where the polyMesh lands,
        # defaulting to the directory the file sits in.
        folder = Path(case) if case is not None else Path(source).resolve().parent
        folder.mkdir(parents=True, exist_ok=True)
        command = [chosen, str(Path(source).resolve()), "-case", str(folder)]
    else:
        # Out of OpenFOAM, `source` IS the case, so it is also the working directory:
        # a converter run from the parent writes its output next to the case rather
        # than inside it, which is the one way these utilities are quietly wrong.
        folder = Path(case) if case is not None else Path(source).resolve()
        command = [chosen, "-case", str(folder)]
    out["command"] = command
    out["case"] = folder.as_posix()
    try:
        done = subprocess.run(command, cwd=str(folder), capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        out["problems"].append(f"{chosen} did not run: {error}")
        return out
    log = (done.stdout or "") + (done.stderr or "")
    out["returncode"] = done.returncode
    out["ok"] = done.returncode == 0
    out["tail"] = "\n".join(log.strip().splitlines()[-12:])
    if not out["ok"]:
        out["problems"].append(f"{chosen} exited {done.returncode}; its last lines are in `tail`")
    return out


def report_convert(found: dict[str, Any]) -> str:
    lines = [f"convert: {found['from']} to {found['to']}", f"  source:    {found['source']}"]
    if found.get("converter"):
        lines.append(f"  converter: {found['converter']}")
        lines.append(f"  ran:       {' '.join(found['command'])}")
        lines.append(f"  exit:      {found.get('returncode')}")
    for problem in found["problems"]:
        lines.append(f"  {problem}")
    if found.get("tail"):
        lines.append("  last lines:")
        lines.extend(f"    {line}" for line in found["tail"].splitlines())
    return "\n".join(lines)


# -- the report -------------------------------------------------------------------


def findings(found: dict[str, Any]) -> list[Finding]:
    """The same facts in preflight's measured/means/repair register.

    The thresholds and the repairs are `preflight.py`'s own, reached through
    `checkmesh_verdict`, so a study reading both does not get two different opinions
    about the same number. What this adds to them is the location.
    """
    out: list[Finding] = []
    verdict = found.get("verdict") or {}
    if verdict.get("numbers"):
        numbers = ", ".join(f"{key} {value:g}" for key, value in verdict["numbers"].items())
        out.append(Finding(
            "checkmesh", verdict.get("status", "unknown"), numbers,
            "; ".join(verdict.get("problems", [])) or
            "every metric read off the log is inside preflight's thresholds",
            "; ".join(verdict.get("repairs", [])),
        ))
    for row in found.get("metrics", []):
        if not row.get("cells"):
            continue
        box = row.get("box", {})
        where = ", ".join(f"{name} ({count})" for name, count in
                          list(row.get("patches", {}).items())[:4]) or "no patch"
        out.append(Finding(
            f"cells:{row['metric']}", row.get("status", "set"),
            f"{row['cells']} cell(s) by the {row['route']} route, "
            f"{row.get('fraction_of_mesh', 0) * 100:.3f}% of the mesh, spanning "
            f"{box.get('span_fraction_of_domain')} of the domain; on {where}",
            "where the flagged cells are, which is what says whether they sit on one "
            "feature or run through the domain",
            "",
        ))
    for problem in found.get("problems", []):
        out.append(Finding("mesh_quality", "skipped", problem))
    return out


def report(found: dict[str, Any]) -> str:
    lines = [f"mesh quality: {found['case']}"]
    if found.get("region"):
        lines.append(f"  region:   {found['region']}")
    lines.append(f"  mesh:     {found.get('mesh', 'unread')}")
    lines.append(f"  log:      {found.get('log') or 'none here'}")
    verdict = found.get("verdict") or {}
    if verdict.get("numbers"):
        lines.append("  checkMesh: " + ", ".join(
            f"{key} {value:g}" for key, value in verdict["numbers"].items()
        ) + f"   [{verdict.get('status')}]")
    for line in found.get("failures", [])[:6]:
        lines.append(f"    *** {line}")
    for line in found.get("warnings", [])[:6]:
        lines.append(f"    *   {line}")

    if found.get("sets"):
        lines.append("")
        lines.append("  sets on disk (the cells checkMesh itself flagged):")
        for entry in found["sets"]:
            metric = entry["metric"] or "no named metric here"
            lines.append(f"    {entry['set']:<24}{entry['kind']:<10}"
                         f"{str(entry['count']):>10}  {metric}")
    else:
        lines.append("")
        lines.append("  no cellSet or faceSet on disk, so nothing came by the set route."
                     " checkMesh -writeSets vtk")
        lines.append("  leaves the cells it flagged under polyMesh/sets.")

    lines.append("")
    for row in found.get("metrics", []):
        head = f"  {row['label']}"
        if row.get("log_value") is not None:
            head += f" = {row['log_value']:g}"
        head += f"   [{row['status']}]"
        lines.append(head)
        lines.append(f"      {row['detail']}")
        if not row.get("cells"):
            continue
        box = row.get("box", {})
        lines.append(
            f"      {row['cells']} cell(s), {row.get('fraction_of_mesh', 0) * 100:.3f}%"
            f" of the mesh, by the {row['route']} route"
        )
        lines.append(f"      they span {box.get('span_fraction_of_domain')} of the domain,"
                     f" from {box.get('min')} to {box.get('max')}")
        if row.get("patches"):
            lines.append("      on patches: " + ", ".join(
                f"{name} ({count})" for name, count in row["patches"].items()
            ))
        for record in row.get("worst", []):
            value = f" value {record['value']:g}" if "value" in record else ""
            gap = record.get("distance_to_patch")
            near = (f"on {record['patch']}" if record.get("on_patches")
                    else f"nearest {record['patch']}"
                          + (f" at {gap:g} m" if isinstance(gap, float) else ""))
            lines.append(
                f"        cell {record['cell']:<10}{record['centre']}{value}  {near},"
                f" {record['distance_to_bbox']:g} m from the bounding box"
            )
    if found.get("picture"):
        lines.append("")
        lines.append(f"  drawn: {found.get('drawn_metric')} by the "
                     f"{found.get('drawn_route')} route")
    if not any(row.get("cells") for row in found.get("metrics", [])):
        lines.append("")
        lines.append("  no cells were flagged by either route: nothing on disk names any,"
                     " and the metrics")
        lines.append("  recomputed here are under the thresholds. There is no picture"
                     " because there is")
        lines.append("  nothing in it.")
    for problem in found.get("problems", []):
        lines.append(f"  note: {problem}")
    return "\n".join(lines)


def as_json(found: dict[str, Any]) -> str:
    payload = {key: value for key, value in found.items() if not key.startswith("_")}
    payload["metrics"] = [
        {key: value for key, value in row.items() if key != "drawable"}
        for row in payload.get("metrics", [])
    ]
    made = findings(found)
    payload["findings"] = [item.as_dict() for item in made]
    payload["worst"] = preflight.worst_status(made)
    return json.dumps(payload, indent=2, default=float)


# -- the command line -------------------------------------------------------------


def _emit(payload: str, text: str, json_out: str | None) -> None:
    if json_out and json_out != "-":
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(text)
        print(f"json:    {path.resolve().as_posix()}")
    elif json_out:
        print(payload)
    else:
        print(text)


def _cmd_report(args: argparse.Namespace) -> int:
    thresholds: dict[str, float] = {}
    if args.skewness is not None:
        thresholds["skewness"] = args.skewness
    if args.non_ortho is not None:
        thresholds["non_ortho"] = args.non_ortho
    try:
        found = measure(args.case, region=args.region or "", log=args.log,
                        thresholds=thresholds, worst=args.worst)
    except Exception as error:  # noqa: BLE001 - the last net under the exit-0 contract
        found = {
            "case": Path(args.case).as_posix(), "region": args.region or "",
            "metrics": [], "sets": [],
            "problems": [f"mesh_quality.py raised {type(error).__name__}: {error}"],
        }

    out_png = Path(args.png) if args.png else (
        Path(args.out) / "mesh_quality.png" if args.out else None
    )
    if out_png is not None:
        try:
            found.update(draw(found, out_png, max_draw=args.max_draw))
        except QualityError as error:
            found.setdefault("problems", []).append(f"nothing was drawn: {error}")
        except Exception as error:  # noqa: BLE001 - a picture is never the point
            found.setdefault("problems", []).append(
                f"the picture could not be drawn: {type(error).__name__}: {error}"
            )

    json_out = args.json_out
    if json_out is None and args.out:
        json_out = str(Path(args.out) / "mesh_quality.json")
    _emit(as_json(found), report(found), json_out)
    if found.get("picture"):
        print(f"picture: {found['picture']}")
    return 0


def _cmd_formats(args: argparse.Namespace) -> int:
    found = probe()
    _emit(json.dumps(found, indent=2), report_formats(found), args.json_out)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    found = convert(args.source, args.source_format, args.target_format, case=args.case)
    _emit(json.dumps(found, indent=2), report_convert(found), args.json_out)
    return 0


FORMAT_NAMES = sorted(set(CONVERTERS) | {"openfoam"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="the bad cells: how many, where, and a picture")
    p.add_argument("case", type=Path, help="the case directory")
    p.add_argument("--region", default=None, help="one mesh region of a multi-region case")
    p.add_argument("--log", type=Path, default=None,
                   help="a checkMesh log, instead of the case's own log.checkMesh")
    p.add_argument("--skewness", type=float, default=None,
                   help=f"the skewness a cell is selected at (preflight's warn is "
                        f"{preflight.SKEWNESS_WARN:g})")
    p.add_argument("--non-ortho", dest="non_ortho", type=float, default=None,
                   help=f"the non-orthogonality in degrees a cell is selected at "
                        f"(preflight's warn is {preflight.NON_ORTHO_WARN:g})")
    p.add_argument("--worst", type=int, default=WORST,
                   help=f"how many cells per metric are located in full (default {WORST})")
    p.add_argument("--png", default=None, metavar="PATH", help="the bad-cell picture")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="a directory for mesh_quality.png and mesh_quality.json")
    p.add_argument("--max-draw", type=int, default=MAX_DRAW,
                   help=f"cells drawn before the picture is truncated (default {MAX_DRAW})")
    p.add_argument("--json", dest="json_out", nargs="?", const="-", default=None,
                   metavar="PATH", help="the same facts as JSON, on stdout or to a file")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("formats", help="which mesh formats are reachable on this instance")
    p.add_argument("--json", dest="json_out", nargs="?", const="-", default=None,
                   metavar="PATH")
    p.set_defaults(func=_cmd_formats)

    p = sub.add_parser("convert", help="one OpenFOAM mesh converter, run or reported absent")
    p.add_argument("source", type=Path, help="the mesh file, or the case for --from openfoam")
    p.add_argument("--from", dest="source_format", required=True, choices=FORMAT_NAMES)
    p.add_argument("--to", dest="target_format", required=True, choices=FORMAT_NAMES)
    p.add_argument("--case", type=Path, default=None,
                   help="the case the converted mesh is written into")
    p.add_argument("--json", dest="json_out", nargs="?", const="-", default=None,
                   metavar="PATH")
    p.set_defaults(func=_cmd_convert)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
