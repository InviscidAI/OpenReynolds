#!/usr/bin/env python3
"""The wall resolution a run actually achieved, measured off what it wrote.

Everything in here that touched y+ until now was a PREDICTION. `case_gen.py`'s
`estimate_y_plus` puts the first cell centre on a flat-plate skin-friction
correlation and prints an order of magnitude before the solver starts, and it says
so: its own note records a generated aerofoil O-grid whose first cell centre sat at
y+ about 1200 under a wall function valid to 300, and the line beside it says the
real number is measurable afterwards with a y+ function object. Nothing read that
function object's output back. `layer_report.py` measures the mesh -- the first-cell
height at every patch face -- which is the geometry half and carries no velocity, so
it cannot tell a 4 mm cell in a duct from a 4 mm cell on a hull at 20 knots. This is
the other half: the velocity field's own answer, per patch, after the run.

three ways to the same number, in order of how directly the run said it

1. `postProcessing/yPlus/<time>/yPlus.dat`, which is OpenFOAM's `yPlus` function
   object reporting min, max and average per patch per write. The most direct
   answer there is, and the cheapest: it is already computed.
2. a written `yPlus` field in a time directory, whose `boundaryField` carries one
   value per wall face -- so the distribution is there, not only its three summary
   numbers, and the regime fractions below are real fractions.
3. `wallShearStress`, where that is what the run wrote. u_tau = sqrt(|tau_wall|)
   for the kinematic field OpenFOAM writes for an incompressible solver, and
   y+ = u_tau y / nu with y the distance to the first cell centre -- half the
   first-cell height out of `layer_report.py`, which is the y OpenFOAM's own yPlus
   uses -- and nu read off `constant/transportProperties`. nu is REFUSED rather than
   assumed: a y+ computed from a guessed viscosity is a number that looks like a
   measurement and is not one, so a case whose nu cannot be read is reported with
   the flag that would supply it and no number at all.

Whatever is present is reported, each labelled with where it came from, because the
three disagreeing is itself worth seeing.

why the field is parsed and not read through pyvista

`ENVIRONMENT.md` has pyvista, and `render.py` is the right way to read a solved
volume. A `boundaryField` is not a volume: it is one value per boundary face,
already keyed by patch name in the file, and getting it back out of a reader means
reconstructing the patch-to-block mapping that the text already states. The reader
also chokes on fields carrying `$variable` and `#includeEtc`, which `render.py`
works around with `foamToVTK` -- a conversion of the whole case to answer a question
about a few hundred numbers. So this parses, with `preflight.py`'s brace matching,
and needs no render stack: it runs on a login node, and on a case the reader cannot
open.

the three regions

y+ < 5 is the viscous sublayer, where the velocity profile is linear; 5 to 30 is the
buffer layer, where neither law holds; above 30 is the log layer. A patch is reported
as the FRACTIONS of its faces in each, not as one verdict, because a wall is almost
never in one region -- a hull's bow and its midbody are different flows, and a single
average hides exactly the part that is wrong. Where only a `.dat` summary exists
there are no per-face values, so what is reported is which regions the min-to-max
range spans, and the fractions are stated as unavailable rather than invented.

what it says about the wall treatment, and what it does not

Every turbulence model `case_gen.py` writes gets `nutUSpaldingWallFunction`, which
blends through all three regions and is therefore valid at whatever y+ came out; for
those cases this finding is informational and says so. It is sharper where a case
used a plain high-Re form: `nutkWallFunction` assumes the log law holds at the first
cell, so a wall measured in the sublayer is outside what that form is for. What is
printed is the measurement and what it means. The reading, and whatever follows from
it, are yours; this edits nothing and exits 0 whatever it finds, including a case
that wrote no y+ data at all.

    python3 wall.py /work/case
    python3 wall.py /work/case --time 0.5
    python3 wall.py /work/case --nu 1e-06
    python3 wall.py /work/case --json wall.json --png wall.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_report  # noqa: E402  (sibling scripts, not a package)
import locate  # noqa: E402
import preflight  # noqa: E402
import results  # noqa: E402

Finding = preflight.Finding


class WallError(Exception):
    """A measurement that could not be made. Not the same as a measurement of zero."""


def _need(value: Any, what: str, where: str, flag: str) -> Any:
    """`benchmarks/moving_mesh/grade.py`'s rule: a missing constant stops and names
    the flag that would supply it. A default here is a wrong y+ with a right format."""
    if value is None:
        raise WallError(f"could not read {what} from {where}; pass {flag}")
    return value


# ------------------------------------------------------------------- regions --


REGIMES: tuple[tuple[str, float, float], ...] = (
    ("viscous sublayer", 0.0, 5.0),
    ("buffer layer", 5.0, 30.0),
    ("log layer", 30.0, math.inf),
)
"""The three wall regions by y+, and the boundaries every wall function is written
against. 5 and 30 are the conventional edges; nothing here pretends they are sharp."""


def regime_of(y_plus: float) -> str:
    for name, low, high in REGIMES:
        if low <= y_plus < high:
            return name
    return REGIMES[-1][0]


def regimes_spanned(low: float, high: float) -> list[str]:
    """Which regions a min-to-max range touches. The honest answer from a summary
    that carries no distribution: three numbers cannot say how much sits where."""
    return [name for name, a, b in REGIMES if high >= a and low < b]


def regime_fractions(values: np.ndarray) -> dict[str, float]:
    """The fraction of faces in each region, which is what a distribution can say."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: 0.0 for name, _, _ in REGIMES}
    return {
        name: round(100.0 * float(((finite >= a) & (finite < b)).mean()), 2)
        for name, a, b in REGIMES
    }


def patch_record(patch: str, values: np.ndarray | None,
                 summary: tuple[float, float, float] | None = None,
                 faces: int | None = None) -> dict[str, Any]:
    """One patch's row, from per-face values where there are any and from a
    summary's three numbers where that is all the run wrote."""
    if values is not None and np.asarray(values).size:
        values = np.asarray(values, dtype=float)
        low, high, mean = float(values.min()), float(values.max()), float(values.mean())
        fractions: dict[str, float] | None = regime_fractions(values)
        faces = int(values.size) if faces is None else faces
    elif summary is not None:
        low, high, mean = summary
        fractions = None
    else:
        raise WallError(f"no y+ values and no summary for patch '{patch}'")
    return {
        "patch": patch,
        "min": low,
        "max": high,
        "mean": mean,
        "faces": faces,
        "spans": regimes_spanned(low, high),
        "fractions": fractions,
    }


# ------------------------------------------- the function object's own output --


_COLUMN_ALIASES = {
    "time": ("time",),
    "patch": ("patch", "patchname"),
    "min": ("min", "minimum"),
    "max": ("max", "maximum"),
    "mean": ("average", "mean", "avg"),
}
_POSITIONAL = {"time": 0, "patch": 1, "min": 2, "max": 3, "mean": 4}


def dat_columns(text: str) -> dict[str, int]:
    """Which column is which, read off the `# Time patch min max average` header.

    The header is split by `results.py`'s own `split_header_names`, so a `.dat`
    header is read one way in the toolbox rather than two. `results.parse_dat` is
    the function that would normally do the whole file and it cannot do this one:
    its row reader returns nothing for any line carrying a non-numeric token -- a
    guard against a job killed mid-write shifting every column -- and every row of
    a `yPlus.dat` carries a patch NAME in its second column. So the header comes
    from there and the rows are read here, where a string column is expected.
    """
    found: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            if found:
                break
            continue
        names = [name.lower() for name in results.split_header_names(stripped)]
        if "patch" not in names:
            continue
        found = {}
        for key, aliases in _COLUMN_ALIASES.items():
            for index, name in enumerate(names):
                if name in aliases:
                    found[key] = index
                    break
    return found if set(found) >= set(_POSITIONAL) else dict(_POSITIONAL)


def dat_rows(path: Path) -> list[dict[str, Any]]:
    """`(time, patch, min, max, mean)` per row of one yPlus `.dat`."""
    text = preflight.read_text(path)
    columns = dat_columns(text)
    width = max(columns.values()) + 1
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < width:
            continue
        try:
            row = {
                key: float(tokens[index])
                for key, index in columns.items()
                if key != "patch"
            }
        except ValueError:
            continue
        row["patch"] = tokens[columns["patch"]]
        rows.append(row)
    return rows


def dat_files(case: Path) -> dict[str, list[Path]]:
    """Every yPlus `.dat` under `postProcessing`, grouped into one series each.

    A restart writes a second time directory beside the first, and a restart into a
    directory that already has the file writes `yPlus_0.2.dat` beside `yPlus.dat`.
    Both are one history, as `results.py` has it, and reading only the newest
    directory is how half a run's wall resolution goes missing.
    """
    base = Path(case) / "postProcessing"
    grouped: dict[str, list[Path]] = {}
    for dat in sorted(base.rglob("*.dat")) if base.is_dir() else []:
        if not dat.is_file():
            continue
        name, _ = results.series_name(dat)
        if not name.lower().startswith("yplus"):
            continue
        parts = dat.relative_to(base).parts
        function = "/".join(parts[:-2]) if len(parts) >= 3 else dat.parent.name
        grouped.setdefault(f"{function}/{name}", []).append(dat)
    for paths in grouped.values():
        paths.sort(key=results.written_order)
    return grouped


def from_dat(case: Path, requested: str | None = None) -> list[dict[str, Any]]:
    """One source record per yPlus series the case logged, histories stitched."""
    sources: list[dict[str, Any]] = []
    for key, paths in sorted(dat_files(case).items()):
        rows: list[dict[str, Any]] = []
        for path in paths:
            start = results.restart_point(path)
            if start is not None:
                edge = start - results.time_tolerance(start)
                rows = [row for row in rows if row["time"] < edge]
            rows.extend(dat_rows(path))
        if not rows:
            continue
        times = sorted({row["time"] for row in rows})
        if requested is not None:
            wanted = float(requested)
            times = [t for t in times if abs(t - wanted) <= results.time_tolerance(wanted)]
            if not times:
                continue
        at = times[-1]
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            if abs(row["time"] - at) <= results.time_tolerance(at):
                latest[row["patch"]] = row
        sources.append({
            "source": "yPlus function object",
            "detail": ", ".join(results.post_relative(path) for path in paths),
            "series": key,
            "time": at,
            "times": sorted({row["time"] for row in rows}),
            "patches": [
                patch_record(name, None, (row["min"], row["max"], row["mean"]))
                for name, row in sorted(latest.items())
            ],
        })
    return sources


# ------------------------------------------------------- a written field's wall --


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def field_components(text: str) -> int:
    """3 for a `volVectorField`, 1 for a `volScalarField`, off the FoamFile class."""
    match = re.search(r"\bclass\s+(\w+)", text)
    return 3 if match and "Vector" in match.group(1) else 1


def boundary_values(text: str) -> dict[str, np.ndarray | None]:
    """`{patch: magnitudes}` out of a field's `boundaryField`, one per wall face.

    The brace matching is `preflight.py`'s, so a nested patch entry is not read as
    the end of the block. A `uniform` entry comes back as a single value rather than
    tiled: numpy broadcasts it against the per-face heights later, and a size of one
    is also the honest record of what the file says. A patch that wrote no `value`
    at all -- `calculated` on a patch the function object skipped -- comes back
    `None` rather than as an empty array, which would read as a measured nothing.
    """
    components = field_components(text)
    found: dict[str, np.ndarray | None] = {}
    body = preflight.block_body(text, "boundaryField")
    for name, inner in preflight.block_entries(body):
        if not name:
            continue
        patch = name.strip('"')
        match = re.search(r"\bvalue\b", inner)
        if not match:
            found[patch] = None
            continue
        tail = inner[match.end():].split(";", 1)[0]
        # The count in `nonuniform List<scalar> 8(...)` is not a datum; everything
        # from the opening parenthesis on is.
        if "(" in tail:
            tail = tail[tail.index("("):]
        numbers = [float(token) for token in _NUMBER.findall(tail)]
        if not numbers:
            found[patch] = None
            continue
        values = np.asarray(numbers, dtype=float)
        if components == 3:
            if values.size % 3:
                found[patch] = None
                continue
            values = np.linalg.norm(values.reshape(-1, 3), axis=1)
        found[patch] = values
    return found


def field_file(case: Path, name: str, requested: str | None) -> Path | None:
    """The latest written `name`, or the one in `--time`. Time directories are
    `locate.py`'s. Only the case's own times are read: in a decomposed run that was
    never reconstructed each `processor*` holds a slice of every patch, and a
    per-patch range taken off one slice is a range over part of a wall."""
    times = locate.time_dirs(Path(case))
    if requested is not None:
        times = [
            (value, folder) for value, folder in times
            if folder.name == requested
            or (locate.is_time_name(requested) and value == float(requested))
        ]
    for _, folder in reversed(times):
        for candidate in (folder / name, folder / f"{name}.gz"):
            if candidate.is_file():
                return candidate
    return None


def from_field(case: Path, requested: str | None = None) -> dict[str, Any] | None:
    """A written `yPlus` field, which carries the distribution and not only its
    summary -- so the regime fractions off this source are real fractions."""
    path = field_file(case, "yPlus", requested)
    if path is None:
        return None
    values = boundary_values(preflight.read_text(path))
    patches = [
        patch_record(name, array)
        for name, array in sorted(values.items())
        if array is not None and array.size
    ]
    if not patches:
        return None
    return {
        "source": "written yPlus field",
        "detail": f"{path.parent.name}/{path.name}",
        "time": float(path.parent.name) if locate.is_time_name(path.parent.name) else None,
        "patches": patches,
    }


# -------------------------------------------------- y+ derived from the shear --


def viscosity(case: Path, given: float | None) -> tuple[float | None, str]:
    """nu and where it came from. `preflight.py`'s reader, which already knows the
    old and new spellings of the dictionary that holds it."""
    if given is not None:
        return given, "--nu"
    for relative in ("constant/transportProperties", "constant/physicalProperties",
                     "constant/momentumTransport"):
        value = preflight.parse_nu(preflight.read_text(Path(case) / relative))
        if value is not None:
            return value, relative
    return None, ""


def from_wall_shear(case: Path, requested: str | None = None,
                    nu: float | None = None) -> dict[str, Any] | None:
    """y+ built from `wallShearStress`, the mesh and nu, for a run that wrote the
    shear and never asked for y+.

    u_tau = sqrt(|tau_wall|) because the field an incompressible OpenFOAM solver
    writes is already kinematic -- tau/rho, m2/s2 -- and y is HALF the first-cell
    height, the distance to the cell centre, which is the y OpenFOAM's own yPlus
    uses. `layer_report.py` supplies that height per face straight out of
    `constant/polyMesh`, so this needs nothing the run did not already write.
    """
    path = field_file(Path(case), "wallShearStress", requested)
    if path is None:
        return None
    nu = _need(nu, "nu", "constant/transportProperties", "--nu")
    if not nu > 0:
        raise WallError(
            f"nu read as {nu:g}, which no y+ can be divided by; pass --nu"
        )
    mesh = layer_report.load(Path(case))
    values = boundary_values(preflight.read_text(path))
    patches: list[dict[str, Any]] = []
    problems: list[str] = []
    for name, tau in sorted(values.items()):
        if tau is None or tau.size == 0:
            continue
        if name not in mesh["boundary"]:
            problems.append(f"the field has a patch '{name}' the mesh does not")
            continue
        height, _ = layer_report.first_cell_heights(mesh, name)
        if tau.size not in (1, height.size):
            problems.append(
                f"patch '{name}': {tau.size} shear values against {height.size} faces"
            )
            continue
        y_plus = np.sqrt(np.abs(tau)) * (height / 2.0) / nu
        patches.append(patch_record(name, y_plus, faces=int(height.size)))
    if not patches:
        if problems:
            raise WallError("; ".join(problems))
        return None
    return {
        "source": "derived from wallShearStress",
        "detail": (
            f"{path.parent.name}/{path.name}, nu = {nu:g} m2/s, "
            "y = half the first-cell height from layer_report.py"
        ),
        "time": float(path.parent.name) if locate.is_time_name(path.parent.name) else None,
        "nu": float(nu),
        "patches": patches,
        "problems": problems,
    }


# --------------------------------------------------------- the wall treatment --


WALL_FUNCTIONS: dict[str, tuple[float, float, str]] = {
    "nutUSpaldingWallFunction": (
        0.0, math.inf,
        "Spalding's law blended through the sublayer, the buffer layer and the log "
        "layer, so it is valid at any y+",
    ),
    "nutUBlendedWallFunction": (
        0.0, math.inf, "a blended law, valid through all three regions"),
    "nutkWallFunction": (
        30.0, 300.0,
        "the plain high-Re Launder-Spalding form, which assumes the log law holds at "
        "the first cell centre",
    ),
    "nutkRoughWallFunction": (
        30.0, 300.0, "the high-Re form with a roughness height, log law at the first cell"),
    "nutkAtmRoughWallFunction": (
        30.0, 300.0, "the atmospheric high-Re form, log law at the first cell"),
    "nutUWallFunction": (
        30.0, 300.0, "the high-Re form solved on U rather than k, log law at the first cell"),
    "nutLowReWallFunction": (
        0.0, 5.0, "a zero wall viscosity, which assumes the sublayer is resolved"),
}
"""The validity band each `nut` wall condition is written for. The full-range blends
are first because they are what `case_gen.py` writes for every model it supports."""


def nut_types(case: Path) -> tuple[dict[str, Any], str]:
    """`{patch: the nut condition}` and the file it was read from.

    `0/nut` first: the wall treatment is chosen there and a later time directory is
    a copy of it, so the earliest statement is the one a run was started on.
    """
    candidates = [Path(case) / "0" / "nut", Path(case) / "0" / "nut.orig"]
    latest = field_file(Path(case), "nut", None)
    if latest is not None:
        candidates.append(latest)
    for path in candidates:
        text = preflight.read_text(path)
        if not text:
            continue
        parsed = preflight.parse_boundary_field(text)
        if parsed["types"]:
            return parsed, f"{path.parent.name}/{path.name}"
    return {"names": [], "patterns": [], "types": {}}, ""


def resolve_type(parsed: dict[str, Any], patch: str) -> str:
    """A literal patch name wins over a pattern, as it does in OpenFOAM."""
    types = parsed.get("types", {})
    if patch in types:
        return types[patch]
    for pattern in parsed.get("patterns", []):
        try:
            if re.fullmatch(pattern, patch):
                return types.get(pattern, "")
        except re.error:
            continue
    return ""


# -------------------------------------------------------------- measuring it --


def measure(case: Path, time: str | None = None, nu: float | None = None) -> dict[str, Any]:
    """Every source of y+ this case carries, as plain data. `report` formats it."""
    case = Path(case)
    read_nu, nu_from = viscosity(case, nu)
    found: dict[str, Any] = {
        "case": case.as_posix(),
        "time": time,
        "nu": read_nu,
        "nu_from": nu_from,
        "sources": [],
        "wall_treatment": {},
        "nut_from": "",
        "problems": [],
    }

    found["sources"].extend(from_dat(case, time))
    for producer in (lambda: from_field(case, time),
                     lambda: from_wall_shear(case, time, read_nu)):
        try:
            source = producer()
        except (WallError, layer_report.MeshError) as error:
            found["problems"].append(str(error))
            continue
        if source is not None:
            found["problems"].extend(source.pop("problems", []))
            found["sources"].append(source)

    parsed, nut_from = nut_types(case)
    found["nut_from"] = nut_from
    walls = {row["patch"] for source in found["sources"] for row in source["patches"]}
    found["wall_treatment"] = {
        patch: resolve_type(parsed, patch) for patch in sorted(walls)
    }
    if not found["sources"]:
        found["problems"].append(
            "no y+ anywhere: no postProcessing/yPlus/*/yPlus*.dat, no written yPlus "
            "field, and no wallShearStress to derive one from"
        )
    return found


def primary(found: dict[str, Any]) -> dict[str, Any] | None:
    """The source to draw and to lead the report with: the one the run computed
    itself, before one computed here."""
    order = ("yPlus function object", "written yPlus field", "derived from wallShearStress")
    for name in order:
        for source in found["sources"]:
            if source["source"] == name:
                return source
    return found["sources"][0] if found["sources"] else None


def _range(row: dict[str, Any]) -> str:
    return f"y+ {row['min']:.3g} to {row['max']:.3g} (mean {row['mean']:.3g})"


def _regions(row: dict[str, Any], brief: bool = False) -> str:
    """Fractions where there is a distribution to take them over; otherwise the
    regions the range reaches, labelled as spanned so the two are never read alike."""
    if row["fractions"] is not None:
        parts = [f"{name} {value:g}%" for name, value in row["fractions"].items() if value]
        return ", ".join(parts) or "no faces in any region"
    tail = " (spanned)" if brief else " (spanned; a summary carries no fractions)"
    return ", ".join(row["spans"]) + tail


def findings(found: dict[str, Any]) -> list[Finding]:
    """The measurement in preflight's register: what was measured, what it means."""
    out: list[Finding] = []
    for source in found["sources"]:
        measured = "; ".join(
            f"{row['patch']}: {_range(row)}"
            + (f", {row['faces']} faces" if row["faces"] else "")
            for row in source["patches"]
        )
        where = source["detail"]
        at = source.get("time")
        if at is not None:
            where += f" at time {at:g}"
        out.append(Finding(
            "y+", "ok", f"{source['source']}, {where} -- {measured}",
            "; ".join(f"{row['patch']}: {_regions(row)}" for row in source["patches"]),
        ))

    treatment = found["wall_treatment"]
    lead = primary(found)
    if treatment and lead:
        rows = {row["patch"]: row for row in lead["patches"]}
        measured_parts: list[str] = []
        meaning_parts: list[str] = []
        status = "ok"
        for patch, kind in sorted(treatment.items()):
            row = rows.get(patch)
            if row is None:
                continue
            band = WALL_FUNCTIONS.get(kind)
            measured_parts.append(f"{patch}: {kind or 'no nut entry'}, {_range(row)}")
            if band is None:
                meaning_parts.append(
                    f"{patch}: {kind or 'nothing'} is not one of the wall conditions "
                    "this knows a validity band for, so the two are not compared here"
                )
                continue
            low, high, description = band
            inside = row["min"] >= low and row["max"] <= high
            if math.isinf(high):
                meaning_parts.append(
                    f"{patch}: {description}. The y+ measured here is inside what "
                    "the case's own treatment covers whatever it came out as, so "
                    "this is informational rather than a finding about this mesh"
                )
            elif inside:
                meaning_parts.append(
                    f"{patch}: {description}; the measured range sits inside the "
                    f"y+ {low:g} to {high:g} that form is written for"
                )
            else:
                status = "warn"
                meaning_parts.append(
                    f"{patch}: {description}; the measured range runs outside the "
                    f"y+ {low:g} to {high:g} that form is written for, so part of "
                    "this wall was treated by a law that does not hold there"
                )
        if measured_parts:
            out.append(Finding(
                "wall-treatment", status,
                f"{found['nut_from'] or 'nut'}: " + "; ".join(measured_parts),
                " | ".join(meaning_parts),
            ))

    for problem in found["problems"]:
        out.append(Finding("y+", "skipped", problem))
    return out


# ----------------------------------------------------------------- reporting --


def report(found: dict[str, Any]) -> str:
    lines = [f"wall resolution: {found['case']}"]
    for source in found["sources"]:
        at = source.get("time")
        head = f"  {source['source']} -- {source['detail']}"
        if at is not None:
            head += f"   time {at:g}"
        lines.append("")
        lines.append(head)
        if source.get("times") and len(source["times"]) > 1:
            lines.append(
                f"    stitched over {source['times'][0]:g} .. {source['times'][-1]:g}"
                f" ({len(source['times'])} writes)"
            )
        lines.append(
            f"    {'patch':<16}{'faces':>8}{'y+ min':>11}{'y+ mean':>11}{'y+ max':>11}"
            "   regions"
        )
        for row in source["patches"]:
            faces = str(row["faces"]) if row["faces"] else "-"
            lines.append(
                f"    {row['patch']:<16}{faces:>8}{row['min']:>11.4g}"
                f"{row['mean']:>11.4g}{row['max']:>11.4g}   {_regions(row, brief=True)}"
            )
    if found["nu"] is not None:
        lines.append("")
        lines.append(f"  nu {found['nu']:g} m2/s   ({found['nu_from']})")
    if found["wall_treatment"]:
        lines.append(f"  nut {found['nut_from'] or 'not found'}   " + ", ".join(
            f"{patch}: {kind or 'no entry'}"
            for patch, kind in found["wall_treatment"].items()
        ))

    lines.append("")
    lines.append("findings")
    for finding in sorted(findings(found),
                          key=lambda f: -preflight._SEVERITY.get(f.status, 0)):
        lines.append(f"  [{finding.status}] {finding.check}")
        lines.append(f"      measured: {finding.measured}")
        if finding.meaning:
            lines.append(f"      means:    {finding.meaning}")
        if finding.repair:
            lines.append(f"      repair:   {finding.repair}")
    lines.append("")
    lines.append("nothing here decides whether this wall resolution is good enough.")
    return "\n".join(lines)


def plot(found: dict[str, Any], out: Path) -> Path | None:
    """One bar per patch from its minimum y+ to its maximum, with the mean marked
    and the two region boundaries drawn, because the range is the finding and a bar
    chart of means would hide it."""
    source = primary(found)
    if source is None or not source["patches"]:
        return None
    rows = source["patches"]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 1.2 + 0.5 * len(rows)))
    floor = min([row["min"] for row in rows if row["min"] > 0] or [0.1]) * 0.5
    for index, row in enumerate(rows):
        low = max(row["min"], floor)
        axes.plot([low, max(row["max"], low)], [index, index], lw=8,
                  color="#3b6ea5", solid_capstyle="butt")
        axes.plot([max(row["mean"], floor)], [index], "o", color="#f0f0f0",
                  markeredgecolor="#1f3d5c", zorder=3)
    for edge in (5.0, 30.0):
        axes.axvline(edge, color="#999999", ls="--", lw=1)
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([row["patch"] for row in rows])
    axes.set_xscale("log")
    axes.set_xlabel("y+   (5 and 30: sublayer | buffer | log layer)")
    axes.set_title(f"{source['source']} -- {Path(found['case']).name}")
    axes.invert_yaxis()
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    plt.close(figure)
    return out


def as_json(found: dict[str, Any]) -> str:
    payload = dict(found)
    found_findings = findings(found)
    payload["findings"] = [finding.as_dict() for finding in found_findings]
    payload["worst"] = preflight.worst_status(found_findings)
    return json.dumps(payload, indent=2, sort_keys=False, default=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, help="the case directory")
    parser.add_argument("--time", default=None,
                        help="read this time, not the latest written")
    parser.add_argument("--nu", type=float, default=None,
                        help="kinematic viscosity, where the case does not state one")
    parser.add_argument("--json", dest="json_out", nargs="?", const="-", default=None,
                        metavar="PATH",
                        help="the same facts as JSON, on stdout or to a file")
    parser.add_argument("--png", default=None, metavar="PATH",
                        help="a per-patch y+ range plot")
    args = parser.parse_args(argv)

    try:
        found = measure(args.case, time=args.time, nu=args.nu)
    except Exception as error:  # pragma: no cover - the last net under the contract
        found = {
            "case": Path(args.case).as_posix(), "time": args.time, "nu": None,
            "nu_from": "", "sources": [], "wall_treatment": {}, "nut_from": "",
            "problems": [f"wall.py raised {type(error).__name__}: {error}"],
        }

    if args.png:
        written = plot(found, Path(args.png))
        found["picture"] = written.resolve().as_posix() if written else None

    payload = as_json(found)
    if args.json_out and args.json_out != "-":
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(report(found))
        print(f"json:    {path.resolve().as_posix()}")
    elif args.json_out:
        print(payload)
    else:
        print(report(found))
    if args.png and found.get("picture"):
        print(f"picture: {found['picture']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
