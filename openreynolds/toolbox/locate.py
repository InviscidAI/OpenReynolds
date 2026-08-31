#!/usr/bin/env python3
"""Where a diverging run went wrong, measured off the case rather than argued.

On the Wigley free-surface case, eight hypotheses about a divergence were proposed
and all eight were falsified, over five rounds and two days. What settled it in one
step was a measurement with three parts: *where* the extremum was (the first cell row
at the inlet, x = -4.4375), *which field moved first* (p_rgh, at the first write,
before omega and before U), and *whether the value was mesh-invariant* (-4723 Pa on
two domains with completely different cell sizes, agreeing to four significant
figures -- a quantity invariant to the mesh is set by the boundary specification, not
the grid). Location plus field-ordering plus mesh-invariance is diagnostic; a
mechanism that explains every symptom is not, because a wrong mechanism explains
them too. This script makes the three measurements in one call, and it deliberately
proposes no mechanism at all: what it prints is where, which, and whether -- never
why.

what comes out

Findings in preflight's register -- what was **measured**, what that **means**, a
suggested **repair** -- worst first. The `means` on each is kept interpretation-light
on purpose: the whole point of this tool is location over explanation, and a locator
that editorialised would be a ninth hypothesis.

- the last written time (serial or `processor*`), and for each field present
  (p, p_rgh, U, alpha.*, k, omega, nut): min, max, mean, and the **location** of the
  extremum -- cell index, cell-centre coordinates read out of `constant/polyMesh`
  (ascii or binary), the patch (if any) the cell's boundary faces belong to, and its
  distance to the domain bounding box
- the solver log read for ordering: over the run-up to the end, **which field's
  residual or bounding warning degrades first**, and at what time, with the Courant
  number tail
- `--compare <other-case>`: the same last-write scan on a second case, and per field
  whether the extremum's VALUE and its LOCATION (normalised to each domain's bounding
  box) agree -- a value that does not move when the mesh changes is set by the
  boundary specification, not the grid

This script edits nothing, refuses nothing, and blocks nothing. It writes no file,
runs no solver, and there is no exit code that means "you may not proceed": the
exit code is 0 whatever it finds, including a case it could not read. Every reading here
can be wrong -- an extremum against the inlet may be a fault or a jet -- and what to
make of any of it is yours to decide.

    python3 locate.py /work/case
    python3 locate.py /work/case --log log.interFoam
    python3 locate.py /work/case --time 0.5
    python3 locate.py /work/case --compare /work/case_3L
    python3 locate.py /work/case --json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_report  # noqa: E402  (sibling script, not a package)
import preflight  # noqa: E402

Finding = preflight.Finding


# -- which fields are worth localising ---------------------------------------------


FIELD_NAMES = ("p", "p_rgh", "U", "k", "omega", "nut")
"""The fields a divergence announces itself in. `alpha.*` is matched by prefix
because the phase is named after the fluid (`alpha.water`, `alpha.air`)."""


def is_field_of_interest(name: str) -> bool:
    return name in FIELD_NAMES or name.startswith("alpha.")


# -- reading OpenFOAM files that may be binary -------------------------------------
#
# `layer_report.py` reads the polyMesh ascii-only and declines binary by design; a
# crashed production case is routinely `writeFormat binary`, and telling its author
# to foamFormatConvert a mesh before finding out where the extremum sits would price
# the measurement out of the moment it is needed. So the readers here parse the
# FoamFile header for `format` and the `arch` note (label and scalar widths) and read
# either spelling. Bytes are decoded latin-1, which is lossless byte-for-byte, so one
# string serves for both the header regexes and the byte offsets into the raw data.


_FORMAT = re.compile(r"\bformat\s+(\w+)\s*;")
_CLASS = re.compile(r"\bclass\s+(\w+)\s*;")
_ARCH = re.compile(r'arch\s+"[^"]*?label=(\d+);scalar=(\d+)')
_LIST_COUNT = re.compile(r"(\d+)\s*\(")


class FoamData(NamedTuple):
    """One OpenFOAM file: header facts plus the same content twice, as latin-1 text
    (for regexes and offsets) and as bytes (for `np.frombuffer` on a binary list)."""

    fmt: str
    cls: str
    label_bytes: int
    scalar_bytes: int
    text: str
    data: bytes
    body_at: int


def read_bytes(path: Path) -> bytes:
    path = Path(path)
    try:
        if path.is_file():
            return path.read_bytes()
        packed = Path(str(path) + ".gz")
        if packed.is_file():
            return gzip.decompress(packed.read_bytes())
    except OSError:
        return b""
    return b""


def foam_file(path: Path) -> FoamData | None:
    data = read_bytes(path)
    if not data:
        return None
    text = data.decode("latin-1")
    head = text[:4096]
    fmt = _FORMAT.search(head)
    cls = _CLASS.search(head)
    arch = _ARCH.search(head)
    body_at = 0
    at = text.find("FoamFile")
    if at >= 0:
        brace = text.find("{", at)
        depth = 0
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    body_at = index + 1
                    break
    return FoamData(
        fmt=fmt.group(1) if fmt else "ascii",
        cls=cls.group(1) if cls else "",
        label_bytes=int(arch.group(1)) // 8 if arch else 4,
        scalar_bytes=int(arch.group(2)) // 8 if arch else 8,
        text=text,
        data=data,
        body_at=body_at,
    )


class ListError(Exception):
    """The list could not be read. A fact about the file, reported, never raised
    past `scan`."""


def _ascii_span(text: str, open_at: int) -> tuple[str, int]:
    """The text inside the balanced `( ... )` opening at `open_at`, and the index
    just past its close."""
    depth = 0
    for index in range(open_at, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:index], index + 1
    return text[open_at + 1:], len(text)


def read_list(fobj: FoamData, start: int, kind: str, per_item: int = 1):
    """The next `<count> ( ... )` list at or after `start`, ascii or binary.

    Returns `(flat array, index past the list)`. `kind` is `label` or `scalar` and
    decides both the dtype and, for a binary file, the item width the `arch` note
    promised.
    """
    match = _LIST_COUNT.search(fobj.text, start)
    if not match:
        raise ListError("no list found")
    count = int(match.group(1))
    open_at = match.end() - 1
    if fobj.fmt == "binary":
        width = fobj.label_bytes if kind == "label" else fobj.scalar_bytes
        dtype = f"<i{width}" if kind == "label" else f"<f{width}"
        nbytes = count * per_item * width
        raw = fobj.data[open_at + 1: open_at + 1 + nbytes]
        if len(raw) < nbytes:
            raise ListError(f"binary list truncated: wanted {nbytes} bytes, have {len(raw)}")
        return np.frombuffer(raw, dtype=dtype), open_at + 1 + nbytes + 1
    inner, end = _ascii_span(fobj.text, open_at)
    flat = inner.replace("(", " ").replace(")", " ")
    dtype = np.int64 if kind == "label" else np.float64
    return np.fromstring(flat, dtype=dtype, sep=" "), end


# -- the mesh, both formats --------------------------------------------------------


def read_boundary(path: Path) -> dict[str, dict[str, Any]]:
    """`boundary` -> {patch: {type, nFaces, startFace}}. Not `layer_report`'s reader
    on purpose: a `writeFormat binary` case stamps `format binary;` on this file too,
    though its entries are textual, and refusing it there would cost the patch
    attribution exactly on the cases this script exists for."""
    fobj = foam_file(path)
    if fobj is None:
        return {}
    open_at = fobj.text.find("(", fobj.body_at)
    if open_at < 0:
        return {}
    inner, _ = _ascii_span(fobj.text, open_at)
    out: dict[str, dict[str, Any]] = {}
    for match in re.finditer(r"([\w.\-]+)\s*\{(.*?)\}", inner, flags=re.S):
        name, block = match.group(1), match.group(2)
        faces = re.search(r"nFaces\s+(\d+)\s*;", block)
        start = re.search(r"startFace\s+(\d+)\s*;", block)
        kind = re.search(r"\btype\s+(\w+)\s*;", block)
        if faces and start:
            out[name] = {
                "type": kind.group(1) if kind else "unknown",
                "nFaces": int(faces.group(1)),
                "startFace": int(start.group(1)),
            }
    return out


def read_mesh(poly: Path) -> dict[str, Any] | str:
    """`constant/polyMesh` as the dict `layer_report`'s geometry expects, or a
    sentence saying why not. The geometry itself -- face centres by fan, cell centres
    by pyramid decomposition -- is `layer_report`'s and is reused, not rewritten."""
    poly = Path(poly)
    if not poly.is_dir():
        return f"no {poly.as_posix()}"
    try:
        points_file = foam_file(poly / "points")
        owner_file = foam_file(poly / "owner")
        faces_file = foam_file(poly / "faces")
        if not points_file or not owner_file or not faces_file:
            return "points, owner or faces missing or unreadable"

        flat_points, _ = read_list(points_file, points_file.body_at, "scalar", per_item=3)
        if flat_points.size % 3:
            return "point list is not a multiple of three"
        points = flat_points.astype(np.float64).reshape(-1, 3)

        owner, _ = read_list(owner_file, owner_file.body_at, "label")
        owner = owner.astype(np.int64)

        neighbour = np.zeros(0, dtype=np.int64)
        neighbour_file = foam_file(poly / "neighbour")
        if neighbour_file:
            neighbour, _ = read_list(neighbour_file, neighbour_file.body_at, "label")
            neighbour = neighbour.astype(np.int64)

        if faces_file.cls == "faceCompactList":
            offsets, after = read_list(faces_file, faces_file.body_at, "label")
            verts, _ = read_list(faces_file, after, "label")
            offsets = offsets.astype(np.int64)
            verts = verts.astype(np.int64)
        elif faces_file.fmt == "binary":
            return "binary faceList (pre-compact) is not readable here"
        else:
            flat, _ = read_list(faces_file, faces_file.body_at, "label")
            verts_parts: list[np.ndarray] = []
            offsets_list = [0]
            index, total = 0, flat.size
            while index < total:
                size = int(flat[index])
                if size <= 0 or index + 1 + size > total:
                    return f"face list is malformed at label {index}"
                verts_parts.append(flat[index + 1: index + 1 + size])
                offsets_list.append(offsets_list[-1] + size)
                index += 1 + size
            verts = (
                np.concatenate(verts_parts).astype(np.int64)
                if verts_parts else np.zeros(0, np.int64)
            )
            offsets = np.asarray(offsets_list, dtype=np.int64)

        boundary = read_boundary(poly / "boundary")

        n_cells = 1 + int(max(owner.max(initial=-1), neighbour.max(initial=-1)))
        return {
            "points": points,
            "verts": verts,
            "offsets": offsets,
            "owner": owner,
            "neighbour": neighbour,
            "boundary": boundary,
            "n_cells": n_cells,
        }
    except (ListError, ValueError, OSError) as error:
        return f"mesh unreadable: {error}"


def cell_patches(mesh: dict[str, Any], cell: int) -> list[str]:
    """The patches the cell's own boundary faces belong to. Empty means interior."""
    n_internal = int(mesh["neighbour"].size)
    owner = mesh["owner"]
    faces = np.flatnonzero(owner[n_internal:] == cell) + n_internal
    names: list[str] = []
    for name, entry in mesh["boundary"].items():
        lo, hi = entry["startFace"], entry["startFace"] + entry["nFaces"]
        if np.any((faces >= lo) & (faces < hi)) and name not in names:
            names.append(name)
    return names


def bbox_facts(mesh: dict[str, Any], centre: np.ndarray) -> dict[str, Any]:
    """Distance to the domain bounding box and the centre normalised into it. The
    box, not the true boundary: exact wall distance costs a search this does not
    need, and 'first cell row at the inlet' is a bounding-box sentence."""
    lo = mesh["points"].min(axis=0)
    hi = mesh["points"].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    distance = float(np.minimum(centre - lo, hi - centre).min())
    normalised = (centre - lo) / span
    return {
        "distance_to_bbox": distance,
        "normalised": [round(float(value), 4) for value in normalised],
    }


# -- the fields at the last write --------------------------------------------------


_UNIFORM_SCALAR = re.compile(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;")
_UNIFORM_VECTOR = re.compile(
    r"internalField\s+uniform\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
)
_NONUNIFORM = re.compile(r"internalField\s+nonuniform\s+List<(\w+)>")


def read_internal_field(path: Path) -> dict[str, Any] | None:
    """`internalField` of one field file: uniform or nonuniform, scalar or vector,
    ascii or binary. None when there is nothing readable."""
    fobj = foam_file(path)
    if fobj is None:
        return None
    match = _NONUNIFORM.search(fobj.text)
    if match:
        per_item = 3 if match.group(1) == "vector" else 1
        try:
            flat, _ = read_list(fobj, match.end(), "scalar", per_item=per_item)
        except ListError:
            return None
        values = flat.astype(np.float64)
        if per_item == 3:
            if values.size % 3:
                return None
            values = values.reshape(-1, 3)
        return {"kind": "nonuniform", "components": per_item, "values": values}
    match = _UNIFORM_VECTOR.search(fobj.text)
    if match:
        return {
            "kind": "uniform", "components": 3,
            "values": np.array([float(part) for part in match.groups()]),
        }
    match = _UNIFORM_SCALAR.search(fobj.text)
    if match:
        return {"kind": "uniform", "components": 1, "values": float(match.group(1))}
    return None


def is_time_name(name: str) -> bool:
    try:
        float(name)
    except ValueError:
        return False
    return True


def time_dirs(root: Path) -> list[tuple[float, Path]]:
    found = []
    if root.is_dir():
        for entry in root.iterdir():
            if entry.is_dir() and is_time_name(entry.name):
                found.append((float(entry.name), entry))
    return sorted(found)


def regions(case_path: Path) -> list[tuple[str, Path]]:
    """Where the meshes and times live: the case itself, or each `processor*`. A
    decomposed case's fields never left the processor directories if the run died
    before `reconstructPar`, which is exactly when this gets run."""
    processors = sorted(
        (entry for entry in case_path.glob("processor*") if entry.is_dir()),
        key=lambda entry: int(re.sub(r"\D", "", entry.name) or 0),
    )
    if processors:
        return [(entry.name, entry) for entry in processors]
    return [("", case_path)]


def pick_time(case_path: Path, requested: str | None) -> tuple[str | None, str]:
    """The time directory name to scan: `--time` as given, else the latest written.
    Returns `(name or None, why-not)`."""
    _, first_root = regions(case_path)[0]
    times = time_dirs(first_root)
    if requested:
        for _, entry in times:
            if entry.name == requested or (
                is_time_name(requested) and float(entry.name) == float(requested)
            ):
                return entry.name, ""
        return None, f"no time directory {requested!r}"
    times = [(value, entry) for value, entry in times if any(entry.iterdir())]
    if not times:
        return None, "no time directories with anything in them"
    return times[-1][1].name, ""


def field_stats(values: Any, components: int) -> dict[str, Any]:
    """min/max/mean and the extremum's local cell index. For a vector the quantity
    is the magnitude and the extremum is its maximum; for a scalar the extremum is
    whichever of min and max sits further from the mean -- the outlier, which is the
    thing worth localising. A non-finite value outranks everything: the first NaN or
    inf cell is the extremum in every sense that matters here."""
    if components == 3:
        magnitude = np.linalg.norm(np.atleast_2d(values), axis=1)
    else:
        magnitude = np.atleast_1d(np.asarray(values, dtype=np.float64))
    finite = np.isfinite(magnitude)
    bad = int(magnitude.size - finite.sum())
    if not finite.any():
        return {"non_finite": bad, "all_bad": True, "extremum_cell": 0,
                "extremum_value": float("nan"), "min": None, "max": None, "mean": None}
    good = magnitude[finite]
    vmin, vmax, vmean = float(good.min()), float(good.max()), float(good.mean())
    if bad:
        cell = int(np.flatnonzero(~finite)[0])
        value = float(magnitude[cell])
    elif components == 3:
        cell = int(np.argmax(magnitude))
        value = vmax
    else:
        cell = int(np.argmax(np.abs(magnitude - vmean)))
        value = float(magnitude[cell])
    return {"non_finite": bad, "all_bad": False, "min": vmin, "max": vmax,
            "mean": vmean, "extremum_cell": cell, "extremum_value": value}


def scan(case_path: Path, requested_time: str | None = None) -> dict[str, Any]:
    """The last-write scan: every field of interest at the chosen time, with its
    extremum located in the mesh. Never raises; what could not be read says so."""
    case_path = Path(case_path)
    out: dict[str, Any] = {"case": str(case_path), "time": None, "fields": {}, "errors": []}
    if not case_path.is_dir():
        out["errors"].append(f"{case_path.as_posix()} is not a directory")
        return out
    time_name, why_not = pick_time(case_path, requested_time)
    if time_name is None:
        out["errors"].append(why_not)
        return out
    out["time"] = time_name
    parts = regions(case_path)
    out["parallel"] = bool(parts[0][0])

    meshes: dict[str, Any] = {}

    def mesh_for(label: str, root: Path):
        if label not in meshes:
            meshes[label] = read_mesh(root / "constant" / "polyMesh")
        return meshes[label]

    names: set[str] = set()
    for _, root in parts:
        directory = root / time_name
        if directory.is_dir():
            for entry in directory.iterdir():
                name = entry.name[:-3] if entry.name.endswith(".gz") else entry.name
                if entry.is_file() and is_field_of_interest(name):
                    names.add(name)

    for name in sorted(names):
        record: dict[str, Any] = {"field": name}
        best: dict[str, Any] | None = None
        totals: list[np.ndarray] = []
        uniform = None
        components = 1
        for label, root in parts:
            data = read_internal_field(root / time_name / name)
            if data is None:
                continue
            components = data["components"]
            if data["kind"] == "uniform":
                uniform = data["values"]
                continue
            stats = field_stats(data["values"], components)
            stats["region"] = label
            magnitude = (
                np.linalg.norm(np.atleast_2d(data["values"]), axis=1)
                if components == 3 else np.atleast_1d(data["values"])
            )
            totals.append(magnitude)
            if best is None or _worse(stats, best):
                best = stats
        if best is None:
            if uniform is None:
                record["error"] = "internalField unreadable"
            else:
                record["uniform"] = (
                    [float(part) for part in np.atleast_1d(uniform)]
                    if components == 3 else float(uniform)
                )
            out["fields"][name] = record
            continue
        merged = np.concatenate(totals)
        finite = merged[np.isfinite(merged)]
        record.update({
            "components": components,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "non_finite": int(merged.size - finite.size),
            "extremum": {
                "value": best["extremum_value"],
                "cell": best["extremum_cell"],
                "region": best.get("region", ""),
            },
        })
        label = best.get("region", "")
        root = dict(parts)[label] if label else parts[0][1]
        mesh = mesh_for(label, root)
        if isinstance(mesh, str):
            record["extremum"]["mesh"] = mesh
        elif best["extremum_cell"] < mesh["n_cells"]:
            centre = layer_report.cell_centres(
                mesh, np.array([best["extremum_cell"]])
            )[0]
            record["extremum"]["centre"] = [round(float(part), 6) for part in centre]
            record["extremum"]["patches"] = cell_patches(mesh, best["extremum_cell"])
            record["extremum"].update(bbox_facts(mesh, centre))
        else:
            record["extremum"]["mesh"] = (
                f"cell {best['extremum_cell']} is outside a mesh of {mesh['n_cells']}"
            )
        out["fields"][name] = record
    return out


def _worse(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Across processor pieces, the extremum worth reporting: a non-finite value
    beats any finite one, and further from zero beats nearer."""
    if bool(candidate["non_finite"]) != bool(incumbent["non_finite"]):
        return bool(candidate["non_finite"])
    def weight(stats):
        value = stats["extremum_value"]
        return abs(value) if math.isfinite(value) else math.inf
    return weight(candidate) > weight(incumbent)


# -- the log: which field moves first ----------------------------------------------


_TIME_LINE = re.compile(r"^Time = ([0-9.eE+-]+)\s*$")
_SOLVING = re.compile(
    r"Solving for (\S+?),\s*Initial residual = ([0-9.eE+-]+),"
    r"\s*Final residual = ([0-9.eE+-]+),\s*No Iterations (\d+)"
)
_BOUNDING = re.compile(r"^bounding (\S+?),")
_COURANT = re.compile(r"Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)")

DEGRADE_FACTOR = 100.0
"""How far above its own median a field's outer residual has to climb before the
climb is called an onset. A factor, not an absolute, because 1e-2 is a crisis for a
pressure that lived at 1e-7 and a Tuesday for one that lived at 1e-3."""


def parse_log(path: Path) -> dict[str, Any]:
    """Outer-loop residuals per field per time step, bounding warnings with the time
    they appear, and the Courant history. The FIRST solve of a field in a step is the
    outer residual -- the lesson `log_digest.py` carries -- so inner PIMPLE correctors
    do not flatter the series."""
    residuals: dict[str, list[tuple[float, float]]] = {}
    bounding: dict[str, list[float]] = {}
    courant: list[tuple[float, float, float]] = []
    time_value = 0.0
    seen_this_step: set[str] = set()
    steps = 0
    try:
        handle = Path(path).open("r", errors="replace")
    except OSError:
        return {"residuals": {}, "bounding": {}, "courant": [], "steps": 0}
    with handle:
        for line in handle:
            match = _TIME_LINE.match(line)
            if match:
                time_value = float(match.group(1))
                seen_this_step = set()
                steps += 1
                continue
            match = _SOLVING.search(line)
            if match:
                field, initial = match.group(1), float(match.group(2))
                if field not in seen_this_step:
                    seen_this_step.add(field)
                    residuals.setdefault(field, []).append((time_value, initial))
                continue
            match = _BOUNDING.match(line)
            if match:
                bounding.setdefault(match.group(1), []).append(time_value)
                continue
            match = _COURANT.search(line)
            if match:
                courant.append((time_value, float(match.group(1)), float(match.group(2))))
    return {"residuals": residuals, "bounding": bounding, "courant": courant, "steps": steps}


def degradation_onsets(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per field, the earliest sign of trouble: the first `bounding` line, or the
    first outer residual at `DEGRADE_FACTOR` times the field's own median. A field
    that was bad from its first step has a bad median and therefore no residual
    onset, which is honest: a series with no good stretch has no 'first' to find."""
    onsets: dict[str, dict[str, Any]] = {}
    for field, times in parsed["bounding"].items():
        onsets[field] = {"time": times[0], "how": f"bounding ({len(times)} lines)"}
    for field, series in parsed["residuals"].items():
        values = np.array([value for _, value in series])
        if values.size < 3:
            continue
        baseline = float(np.median(values))
        if baseline <= 0.0:
            continue
        above = np.flatnonzero(values >= DEGRADE_FACTOR * baseline)
        if not above.size:
            continue
        time_value = series[int(above[0])][0]
        ratio = values[int(above[0])] / baseline
        entry = {"time": time_value, "how": f"residual {ratio:.0f}x its median {baseline:.2e}"}
        if field not in onsets or time_value < onsets[field]["time"]:
            onsets[field] = entry
    return onsets


def first_degrading(parsed: dict[str, Any]) -> dict[str, Any] | None:
    onsets = degradation_onsets(parsed)
    if not onsets:
        return None
    field = min(onsets, key=lambda name: onsets[name]["time"])
    return {"field": field, **onsets[field], "onsets": onsets}


# -- mesh-invariance ---------------------------------------------------------------


VALUE_AGREEMENT = 1e-3
"""Relative difference below which two extrema are called the same value. The Wigley
pair agreed to four significant figures across a 1 L and a 3 L domain; one part in a
thousand is the loose edge of that."""

LOCATION_AGREEMENT = 0.05
"""How far apart, as a fraction of each domain's bounding box, two extremum
locations may sit and still be called the same place."""


def compare_fields(here: dict[str, Any], there: dict[str, Any]) -> list[dict[str, Any]]:
    """Per field present in both scans: does the extremum's value survive the change
    of mesh, and does its normalised location."""
    rows: list[dict[str, Any]] = []
    for name in sorted(set(here.get("fields", {})) & set(there.get("fields", {}))):
        a, b = here["fields"][name], there["fields"][name]
        ea, eb = a.get("extremum"), b.get("extremum")
        if not ea or not eb:
            continue
        va, vb = ea["value"], eb["value"]
        row: dict[str, Any] = {"field": name, "value_here": va, "value_there": vb}
        if not (math.isfinite(va) and math.isfinite(vb)):
            row["value_invariant"] = bool(not math.isfinite(va) and not math.isfinite(vb))
            row["figures"] = None
        else:
            scale = max(abs(va), abs(vb))
            relative = abs(va - vb) / scale if scale else 0.0
            row["value_invariant"] = relative <= VALUE_AGREEMENT
            row["figures"] = (
                None if relative == 0.0 else max(0, int(math.floor(-math.log10(relative))))
            )
        na, nb = ea.get("normalised"), eb.get("normalised")
        if na and nb:
            gap = max(abs(x - y) for x, y in zip(na, nb))
            row["location_here"], row["location_there"] = na, nb
            row["location_agrees"] = gap <= LOCATION_AGREEMENT
        rows.append(row)
    return rows


# -- findings ----------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:.6g}"


def field_finding(name: str, record: dict[str, Any]) -> Finding:
    if "error" in record:
        return Finding("last-write", "skipped", f"{name}: {record['error']}")
    if "uniform" in record:
        return Finding(
            "last-write", "ok", f"{name}: uniform {record['uniform']}",
            "a uniform field has no extremum to place",
        )
    quantity = f"|{name}|" if record.get("components") == 3 else name
    extremum = record["extremum"]
    pieces = [
        f"{quantity}: min {_fmt(record['min'])} max {_fmt(record['max'])} "
        f"mean {_fmt(record['mean'])}",
    ]
    where = f"extremum {_fmt(extremum['value'])} at cell {extremum['cell']}"
    if extremum.get("region"):
        where += f" of {extremum['region']}"
    if "centre" in extremum:
        centre = ", ".join(f"{part:.6g}" for part in extremum["centre"])
        where += f", centre ({centre})"
    pieces.append(where)
    if record.get("non_finite"):
        pieces.append(f"{preflight.count_phrase(record['non_finite'], 'non-finite value')}")
    status = "ok"
    meaning = "measured, not interpreted; the location is the finding"
    repair = ""
    if extremum.get("patches"):
        joined = ", ".join(extremum["patches"])
        pieces.append(
            f"the extremum cell has boundary faces on {joined}; "
            f"{_fmt(extremum.get('distance_to_bbox'))} from the domain bounding box, "
            f"at ({', '.join(f'{part:g}' for part in extremum['normalised'])}) of it"
        )
        status = "warn"
        meaning = (
            f"the extremum sits in a cell against {joined}. Where it sits is measured; "
            "why is not, and is not offered here"
        )
        repair = (
            f"read the {name} boundary specification on {joined} before proposing a "
            "mechanism; --compare a differently meshed sibling says whether the value "
            "is set there"
        )
    elif "distance_to_bbox" in extremum:
        pieces.append(
            f"interior cell, {_fmt(extremum['distance_to_bbox'])} from the domain "
            f"bounding box, at ({', '.join(f'{part:g}' for part in extremum['normalised'])}) of it"
        )
    elif "mesh" in extremum:
        pieces.append(f"no coordinates: {extremum['mesh']}")
    if record.get("non_finite"):
        status = "fail"
        meaning = (
            "the field holds values that are not numbers; the first bad cell is the "
            "extremum reported"
        )
    return Finding("last-write", status, "; ".join(pieces), meaning, repair)


def ordering_finding(parsed: dict[str, Any], log_path: Path | None) -> Finding:
    if log_path is None:
        return Finding(
            "ordering", "skipped", "no solver log found",
            "field ordering needs a log; --log names one",
        )
    if not parsed["residuals"] and not parsed["bounding"]:
        return Finding(
            "ordering", "skipped",
            f"{log_path.name}: no residual or bounding lines parsed",
            "either the log is not a solver log or it died before the first step",
        )
    first = first_degrading(parsed)
    tail = "; ".join(
        f"t={time_value:g} Co max {maximum:g}" for time_value, _, maximum in parsed["courant"][-3:]
    )
    last_residuals = ", ".join(
        f"{field} {series[-1][1]:.2e}" for field, series in sorted(parsed["residuals"].items())
    )
    if first is None:
        measured = (
            f"{log_path.name}: {parsed['steps']} steps; no field crosses "
            f"{DEGRADE_FACTOR:g}x its own median and nothing was bounded"
            + (f"; last residuals {last_residuals}" if last_residuals else "")
            + (f"; {tail}" if tail else "")
        )
        return Finding(
            "ordering", "ok", measured,
            "no degradation ordering to report; whatever ended this run did not "
            "announce itself in the residuals",
        )
    others = [
        f"{field} at t={entry['time']:g}"
        for field, entry in sorted(first["onsets"].items(), key=lambda kv: kv[1]["time"])
        if field != first["field"]
    ]
    measured = (
        f"{log_path.name}: {first['field']} degrades first, {first['how']} at "
        f"t={first['time']:g}"
        + (f"; then {', '.join(others)}" if others else "")
        + (f"; last residuals {last_residuals}" if last_residuals else "")
        + (f"; Courant tail {tail}" if tail else "")
    )
    return Finding(
        "ordering", "warn", measured,
        f"{first['field']} is the field that moves first. Which field, and when, is "
        "measured; the mechanism is not",
        f"put the {first['field']} onset time next to the last-write extremum location "
        "before proposing anything",
    )


def compare_finding(row: dict[str, Any], other_case: str) -> Finding:
    name = row["field"]
    figures = row.get("figures")
    agreement = (
        "identical" if figures is None and row["value_invariant"]
        else f"agreeing to {figures} significant figures" if row["value_invariant"]
        else f"differing beyond {VALUE_AGREEMENT:g} relative"
    )
    measured = (
        f"{name}: extremum {_fmt(row['value_here'])} here vs {_fmt(row['value_there'])} "
        f"in {other_case} -- {agreement}"
    )
    if "location_agrees" in row:
        here = ", ".join(f"{part:g}" for part in row["location_here"])
        there = ", ".join(f"{part:g}" for part in row["location_there"])
        placed = "the same place" if row["location_agrees"] else "different places"
        measured += (
            f"; normalised location ({here}) vs ({there}) -- {placed} in the "
            "bounding box"
        )
    if row["value_invariant"]:
        if row.get("location_agrees") is False:
            return Finding(
                "invariance", "warn", measured,
                "the extremum value recurs on the other mesh but in a different "
                "place. A recurring value is not the grid's; where it recurs is a "
                "separate measurement and here the two disagree",
                f"read where each mesh puts its {name} extremum before treating the "
                "pair as one fault",
            )
        return Finding(
            "invariance", "warn", measured,
            "the extremum did not move when the mesh changed. A value invariant to "
            "the mesh is set by the boundary specification, not the grid",
            f"read the {name} boundary entries; the mesh is exonerated for this one",
        )
    return Finding(
        "invariance", "ok", measured,
        "the extremum moved when the mesh changed, so it is not pinned by the "
        "boundary specification alone. A value that moves with the mesh is the "
        "grid's, and refining is a legitimate next measurement",
    )


# -- putting it together -----------------------------------------------------------


def run(
    case_path: Path,
    *,
    log: str | None = None,
    time: str | None = None,
    compare: str | None = None,
) -> tuple[list[Finding], dict[str, Any]]:
    """All three measurements, as findings plus the raw numbers for `--json`."""
    findings: list[Finding] = []
    data: dict[str, Any] = {}

    scanned = scan(Path(case_path), time)
    data["scan"] = scanned
    for problem in scanned["errors"]:
        findings.append(Finding(
            "last-write", "skipped", problem,
            "nothing was written, or the case is not where this was pointed",
        ))
    for name, record in scanned["fields"].items():
        try:
            findings.append(field_finding(name, record))
        except Exception as error:  # one bad field must not cost the others
            findings.append(Finding(
                "last-write", "skipped", f"{name}: reading raised {type(error).__name__}: {error}",
                "this is a bug in locate.py, not necessarily a problem with the case",
            ))

    log_path = None
    if Path(case_path).is_dir():
        log_path = preflight.Case(case_path).solver_log(log)
    parsed = parse_log(log_path) if log_path else {
        "residuals": {}, "bounding": {}, "courant": [], "steps": 0,
    }
    data["log"] = {
        "path": str(log_path) if log_path else None,
        "first_degrading": first_degrading(parsed),
        "courant_tail": parsed["courant"][-5:],
        "last_residuals": {
            field: series[-1] for field, series in parsed["residuals"].items()
        },
    }
    findings.append(ordering_finding(parsed, log_path))

    if compare:
        other = scan(Path(compare), None)
        data["compare"] = {"case": str(compare), "scan": other, "fields": []}
        if other["errors"]:
            findings.append(Finding(
                "invariance", "skipped",
                f"{compare}: {'; '.join(other['errors'])}",
                "the second case could not be scanned, so invariance has no answer",
            ))
        else:
            rows = compare_fields(scanned, other)
            data["compare"]["fields"] = rows
            if not rows:
                findings.append(Finding(
                    "invariance", "skipped",
                    f"no field with a located extremum is present in both this case "
                    f"and {compare}",
                ))
            for row in rows:
                findings.append(compare_finding(row, Path(compare).name))

    findings.sort(key=lambda finding: -preflight._SEVERITY.get(finding.status, 0))
    return findings, data


def render(findings: list[Finding], case_path: Path | str) -> str:
    counts = preflight.summarise(findings)
    lines = [f"# locate {case_path}"]
    lines.append(
        "  ".join(
            f"{status} {counts[status]}" for status in preflight.STATUSES if counts.get(status)
        )
        or "nothing measured"
    )
    for finding in findings:
        lines.append("")
        lines.append(f"{finding.status.upper():<8}{finding.check}")
        lines.append(f"  measured  {finding.measured}")
        if finding.meaning:
            lines.append(f"  means     {finding.meaning}")
        if finding.repair:
            lines.append(f"  repair    {finding.repair}")
    lines.append("")
    lines.append(
        "Locations are measurements, not mechanisms. Nothing here has changed the case."
    )
    return "\n".join(lines)


def as_json(findings: list[Finding], data: dict[str, Any], case_path: Path | str) -> str:
    return json.dumps(
        {
            "case": str(case_path),
            "worst": preflight.worst_status(findings),
            "counts": preflight.summarise(findings),
            "findings": [finding.as_dict() for finding in findings],
            "data": data,
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, help="the case directory")
    parser.add_argument("--log", default=None, help="the solver log to read the ordering from")
    parser.add_argument("--time", default=None, help="scan this time directory, not the latest")
    parser.add_argument(
        "--compare", default=None,
        help="a second case of the same problem on a different mesh; per field, says "
             "whether the extremum's value and location survive the change",
    )
    parser.add_argument("--json", action="store_true", help="findings and numbers as JSON")
    args = parser.parse_args(argv)

    try:
        findings, data = run(
            args.case, log=args.log, time=args.time, compare=args.compare
        )
    except Exception as error:  # pragma: no cover - the last net under the contract
        findings, data = [Finding(
            "locate", "skipped", f"locate raised {type(error).__name__}: {error}",
            "this is a bug in locate.py, not necessarily a problem with the case",
        )], {}
    print(as_json(findings, data, args.case) if args.json else render(findings, args.case))
    return 0


if __name__ == "__main__":
    sys.exit(main())
