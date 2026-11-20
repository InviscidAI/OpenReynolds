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
   the flag that would supply it and no number at all. The KINEMATIC reading of
   that field is checked against the field's own `dimensions` rather than assumed:
   a compressible solver writes the wall shear as a stress in Pa, where mass appears
   in the dimensions, and `sqrt(|tau|)` is not a velocity for it. No case in the
   corpus wrote one, so that path was never exercised; it is refused by name against
   its dimensions instead of being reported in the wrong units.

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

what this reports about a patch, and what it refuses to say about one

The four rules below are all one lesson, and it was paid for after this script
shipped on a green suite: issues #138 and #144, found by running it over the real
corpus. Every one of them was a silent guess dressed as a measurement.

* A `format binary;` field is DECODED, never regexed. 5 of the 74 yPlus and
  wallShearStress files in the corpus are binary -- `writeFormat binary` in
  `controlDict` is ordinary -- and running a number regex over an IEEE-754 payload
  finds whatever digits happen to fall out of the bytes. On the hull case that
  produced `hull 4 faces, 0 / 1.5 / 3, viscous sublayer 100%` while the run's own
  function object, at the same time 1500, said `32.86 / 83.73 / 223.5, log layer`:
  a factor of 75, reported at status `ok`, with a fabricated per-face distribution
  behind it. So the payloads are lifted out through `locate.py`'s binary reader --
  the `arch` note gives the byte order and the scalar width, the list head gives the
  count, and the byte just past the payload has to be the closing parenthesis or the
  file is refused by name rather than guessed at. A binary file never produces a
  number presented as measured: either the decode is exact (it reproduces the hull's
  32.86 / 83.73 / 223.5 to every digit) or the source is declined and says so.
* Only a WALL is reported. y+ is a wall quantity, and `boundaryField` carries an
  entry for every patch -- an inlet's `calculated` / `uniform 0` was being reported
  as a measured `y+ = 0, viscous sublayer 100%`, on the cylinder case as the whole
  report and the whole plot. The patch type comes off `constant/polyMesh/boundary`
  through `locate.read_boundary` (which reads the binary spelling too), with the
  `nut` conditions as the fallback where there is no mesh to read. Non-wall patches
  are LISTED as excluded with their type, because dropping a source without saying
  so is the pattern behind both of these defects.
* A `uniform` entry is reported as uniform. One value is not a distribution: it was
  becoming `faces: 1` and then a 100% per-face fraction. What is printed is the
  value, the word uniform, and the patch's real face count where the mesh states one.
* Time 0 is not a result. `0/yPlus` is the initial condition the function object
  wrote before the run started, and it was being reported as a measurement "at time
  0" beside a function object saying the same wall ran at y+ 2.5 to 1289. It is
  passed over by default, with a note saying so, and read only if `--time 0` asks.

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
                 faces: int | None = None, uniform: bool = False) -> dict[str, Any]:
    """One patch's row, from per-face values where there are any and from a
    summary's three numbers where that is all the run wrote.

    `uniform` is the third state, and it exists because of #144: a `boundaryField`
    entry written `uniform 0` is ONE value for the whole patch, and it was being
    turned into `faces: 1` and from there into a 100% per-face fraction. A single
    value carries no distribution, so the fractions are stated as unavailable the
    way a `.dat` summary's are, and the face count is the mesh's where the mesh
    states one rather than the length of a one-element array.
    """
    if values is not None and np.asarray(values).size:
        values = np.asarray(values, dtype=float)
        low, high, mean = float(values.min()), float(values.max()), float(values.mean())
        fractions: dict[str, float] | None = (
            None if uniform else regime_fractions(values)
        )
        if faces is None and not uniform:
            faces = int(values.size)
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
        "uniform": bool(uniform),
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


def from_dat(case: Path, requested: str | None = None,
             dropped: list[str] | None = None) -> list[dict[str, Any]]:
    """One source record per yPlus series the case logged, histories stitched.

    A `--time` that no row sits on drops the series, and until #138 and #144 were
    fixed it dropped it in silence -- `--time 1075` on the wing case left the
    function object's own answer out of the report with nothing said. What was
    passed over, and the times that were there instead, go into `dropped`.
    """
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
            matched = [
                t for t in times if abs(t - wanted) <= results.time_tolerance(wanted)
            ]
            if not matched:
                if dropped is not None:
                    near = min(times, key=lambda t: abs(t - wanted))
                    dropped.append(
                        f"{key} holds no row at time {requested}: {len(times)} writes "
                        f"from {times[0]:g} to {times[-1]:g}, the nearest at {near:g}. "
                        "This source was passed over rather than reported at a time "
                        "that was not asked for"
                    )
                continue
            times = matched
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
_FORMAT = re.compile(r"\bformat\s+(\w+)\s*;")
_ARCH = re.compile(r'\barch\s+"([^"]*)"')
_LIST_HEAD = re.compile(r"(\d+)\s*\(")
_LIST_OF = re.compile(r"List<(\w+)>")
_PAYLOAD = re.compile(r"@payload (\d+)@")

_LIST_ITEMS: dict[str, int] = {
    "scalar": 1, "sphericalTensor": 1, "vector": 3, "symmTensor": 6, "tensor": 9,
}
"""How many scalars one item of each list type is written as. The list head states
the item count, not the scalar count, so the payload width needs both."""


def field_components(text: str) -> int:
    """3 for a `volVectorField`, 1 for a `volScalarField`, off the FoamFile class."""
    match = re.search(r"\bclass\s+(\w+)", text)
    return 3 if match and "Vector" in match.group(1) else 1


def header_format(text: str) -> str:
    """The `format` entry of the FoamFile header: `ascii` or `binary`.

    Read out of the header and not assumed, which is the whole of #138: nothing
    looked at this entry, so a binary payload went through a number regex and the
    digits that fell out of the bytes were reported as measured y+. Only the header
    region is searched, because a binary payload can contain the word anywhere.
    """
    match = _FORMAT.search(text[:4096])
    return match.group(1).lower() if match else "ascii"


def decode_binary(path: Path) -> tuple[str, list[np.ndarray]]:
    """A binary field as its text with every payload lifted out, and the payloads.

    `locate.py` already reads both spellings of an OpenFOAM file -- it decodes the
    bytes latin-1, which is lossless, so one string serves for the regexes and for
    the byte offsets -- and this is that reader applied to a field rather than a
    mesh. Each `<count> ( <bytes> )` list is replaced in the text by an
    `@payload <n>@` marker, so `preflight.py`'s brace matching then runs over text
    that holds no payload bytes at all: a `{`, a `}` or a `;` inside an IEEE-754
    double cannot be mistaken for structure, and the `value` keyword is never looked
    for inside a payload.

    Every assumption the layout rests on is checked and the file is REFUSED by name
    rather than guessed at when one fails -- the byte just past a payload has to be
    its closing parenthesis, which is the check that would have caught #138 on the
    first run. Raises `WallError`, which `measure` reports as a skipped source.
    """
    fobj = locate.foam_file(path)
    if fobj is None:
        raise WallError(f"{path.name} declares format binary and could not be read")
    arch = _ARCH.search(fobj.text[:4096])
    if arch is None:
        raise WallError(
            f"{path.parent.name}/{path.name} declares format binary but carries no "
            "arch note, so the byte order and scalar width it was written with are "
            "not stated; foamFormatConvert would rewrite it as ascii"
        )
    if not arch.group(1).upper().startswith("LSB"):
        raise WallError(
            f"{path.parent.name}/{path.name} is binary and big-endian "
            f"(arch \"{arch.group(1)}\"); only the little-endian spelling is decoded "
            "here, so this source is refused rather than byte-swapped on a guess"
        )
    if fobj.scalar_bytes not in (4, 8):
        raise WallError(
            f"{path.parent.name}/{path.name} is binary with a "
            f"{fobj.scalar_bytes * 8}-bit scalar, which is neither of the widths "
            "decoded here"
        )
    default_items = 3 if "Vector" in fobj.cls else 1
    # The header comes back with the body: `field_components` reads `class` out of the
    # returned text, and without it every binary volVectorField decodes as a scalar.
    pieces: list[str] = [fobj.text[:fobj.body_at]]
    payloads: list[np.ndarray] = []
    at = fobj.body_at
    while True:
        head = _LIST_HEAD.search(fobj.text, at)
        if head is None:
            pieces.append(fobj.text[at:])
            break
        count = int(head.group(1))
        opened = head.end() - 1
        kinds = _LIST_OF.findall(fobj.text[max(at, head.start() - 60):head.start()])
        items = _LIST_ITEMS.get(kinds[-1], default_items) if kinds else default_items
        width = count * items * fobj.scalar_bytes
        closes = opened + 1 + width
        if fobj.data[closes:closes + 1] != b")":
            raise WallError(
                f"{path.parent.name}/{path.name} is binary and its payload of "
                f"{count} items does not end where the header's widths say it should "
                f"(byte {closes} is not a closing parenthesis), so it is refused "
                "rather than read at an offset that is already known to be wrong; "
                "foamFormatConvert would rewrite it as ascii"
            )
        payloads.append(
            np.frombuffer(fobj.data[opened + 1:closes], dtype=f"<f{fobj.scalar_bytes}")
        )
        pieces.append(fobj.text[at:opened + 1] + f"@payload {len(payloads) - 1}@")
        at = closes
    return "".join(pieces), payloads


def read_field(path: Path) -> tuple[str, list[np.ndarray] | None]:
    """A field's text, and its binary payloads where it has any.

    `None` for an ascii file, which is the common case and is read exactly as
    before. A binary file is decoded here or refused here; it never reaches the
    number regex, which is what #138 was.
    """
    text = preflight.read_text(path)
    if header_format(text) != "binary":
        return text, None
    return decode_binary(path)


def boundary_values(text: str,
                    payloads: list[np.ndarray] | None = None
                    ) -> dict[str, np.ndarray | None]:
    """`{patch: magnitudes}` out of a field's `boundaryField`, one per wall face.

    The brace matching is `preflight.py`'s, so a nested patch entry is not read as
    the end of the block. A `uniform` entry comes back as a single value rather than
    tiled: numpy broadcasts it against the per-face heights later, and a size of one
    is also the honest record of what the file says -- `patch_record` is told so
    rather than counting it as one face. A patch that wrote no `value` at all --
    `calculated` on a patch the function object skipped -- comes back `None` rather
    than as an empty array, which would read as a measured nothing.

    `payloads` is `decode_binary`'s, and an `@payload n@` marker is resolved from it
    instead of being handed to the number regex. Without it the reading is the ascii
    one it has always been, which is what `heat.py` calls this for.
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
        marker = _PAYLOAD.search(tail)
        if marker is not None and payloads is not None:
            index = int(marker.group(1))
            values = payloads[index].astype(float) if index < len(payloads) else None
            if values is None:
                found[patch] = None
                continue
        else:
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


def is_wall_type(kind: str) -> bool:
    """Whether a patch type out of a mesh `boundary` is a wall.

    `wall` is what OpenFOAM writes; `mappedWall` and the other derived spellings end
    in it. Everything else -- `patch`, `empty`, `symmetryPlane`, `slip`, `cyclic` --
    has no wall and therefore no wall resolution.
    """
    return kind == "wall" or kind.lower().endswith("wall")


def patch_types(case: Path) -> tuple[dict[str, str], dict[str, int], str]:
    """`{patch: type}`, `{patch: face count}` and where they were read.

    #144: `boundaryField` holds an entry per patch and every one of them was being
    reported as a measured wall, so an inlet's `calculated` / `uniform 0` came out
    as `y+ 0, viscous sublayer 100%`. The type was available three ways over and
    none was used. This is the mesh's own answer, through `locate.read_boundary`
    rather than a reader written here, because that one reads the binary spelling of
    `boundary` as well -- the same `writeFormat binary` case that produced #138
    stamps the header on this file too. It also carries `nFaces`, which is the honest
    face count for a `uniform` entry.

    The `nut` conditions are the fallback where there is no mesh to read: a
    `nut*WallFunction` is on a wall by construction. A laminar case has no `nut` at
    all, which is why the mesh comes first.
    """
    boundary = locate.read_boundary(Path(case) / "constant" / "polyMesh" / "boundary")
    if boundary:
        return (
            {name: str(entry["type"]) for name, entry in boundary.items()},
            {name: int(entry["nFaces"]) for name, entry in boundary.items()},
            "constant/polyMesh/boundary",
        )
    parsed, where = nut_types(Path(case))
    if not where:
        return {}, {}, ""
    types: dict[str, str] = {}
    for name in parsed.get("names", []):
        kind = resolve_type(parsed, name)
        types[name] = "wall" if kind.endswith("WallFunction") else kind
    return types, {}, where


def field_file(case: Path, name: str, requested: str | None,
               dropped: list[str] | None = None) -> Path | None:
    """The latest written `name`, or the one in `--time`. Time directories are
    `locate.py`'s. Only the case's own times are read: in a decomposed run that was
    never reconstructed each `processor*` holds a slice of every patch, and a
    per-patch range taken off one slice is a range over part of a wall.

    Time 0 is passed over unless `--time 0` asks for it: `0/yPlus` is the field as
    the function object wrote it before the run started, and #144 had it reported as
    a measurement "at time 0" -- hull y+ 0 -- next to a function object saying the
    same hull ran at y+ 2.5 to 1289. Whatever is passed over is appended to
    `dropped`, because a source that disappears without a word is the shape of both
    of the defects this file was fixed for.
    """
    times = locate.time_dirs(Path(case))

    def holds(folder: Path) -> Path | None:
        for candidate in (folder / name, folder / f"{name}.gz"):
            if candidate.is_file():
                return candidate
        return None

    if requested is not None:
        matched = [
            (value, folder) for value, folder in times
            if folder.name == requested
            or (locate.is_time_name(requested) and value == float(requested))
        ]
        if not matched and dropped is not None:
            elsewhere = [folder.name for _, folder in times if holds(folder)]
            if elsewhere:
                dropped.append(
                    f"--time {requested} names no time directory in this case, so the "
                    f"written {name} field was not read; it is there at "
                    + ", ".join(elsewhere)
                )
        times = matched

    for value, folder in reversed(times):
        found = holds(folder)
        if found is None:
            continue
        if requested is None and value == 0.0:
            if dropped is not None:
                dropped.append(
                    f"{found.parent.name}/{found.name} is the field as it stood before "
                    "the run, written at time 0: an initial condition and not a "
                    "result, so it is not reported as a measurement here (--time 0 "
                    "reads it anyway, labelled)"
                )
            continue
        return found
    return None


def from_field(case: Path, requested: str | None = None,
               dropped: list[str] | None = None) -> dict[str, Any] | None:
    """A written `yPlus` field, which carries the distribution and not only its
    summary -- so the regime fractions off this source are real fractions.

    Walls only, binary decoded or refused, and a `uniform` entry reported as one
    value rather than as a one-face histogram. The three rules are #138 and #144,
    and the module docstring has the numbers they were found with.
    """
    path = field_file(case, "yPlus", requested, dropped)
    if path is None:
        return None
    text, payloads = read_field(path)
    values = boundary_values(text, payloads)
    types, faces, typed_from = patch_types(Path(case))
    patches: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    problems: list[str] = []
    for name, array in sorted(values.items()):
        if array is None or not array.size:
            continue
        kind = types.get(name)
        if kind is not None and not is_wall_type(kind):
            excluded.append({"patch": name, "type": kind})
            continue
        uniform = array.size == 1
        expected = faces.get(name)
        # The cheap guard #138 asked for, kept even though the decode makes it
        # unreachable on the corpus: a count that is neither one nor the patch's
        # faces is a read that went wrong, and `4 faces` on a wall of 1008 was all
        # it ever took to know.
        if expected is not None and not uniform and array.size != expected:
            problems.append(
                f"patch '{name}' in {path.parent.name}/{path.name}: {array.size} "
                f"values against {expected} faces in constant/polyMesh/boundary, so "
                "the file and the mesh do not describe the same patch; not reported "
                "as a measurement"
            )
            continue
        patches.append(patch_record(
            name, array, faces=expected if uniform else None, uniform=uniform
        ))
    if not patches and not excluded:
        if problems:
            raise WallError("; ".join(problems))
        return None
    at = float(path.parent.name) if locate.is_time_name(path.parent.name) else None
    detail = f"{path.parent.name}/{path.name}"
    if payloads is not None:
        detail += ", binary, payloads decoded"
    if at == 0.0:
        detail += ", AN INITIAL CONDITION AND NOT A RESULT (--time 0 was asked for)"
    return {
        "source": "written yPlus field",
        "detail": detail,
        "time": at,
        "initial": at == 0.0,
        "patches": patches,
        "excluded": excluded,
        "typed_from": typed_from,
        "problems": problems,
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


def kinematic_or_refused(path: Path, text: str) -> str:
    """How the shear field is dimensioned, or a refusal of it.

    `u_tau = sqrt(|tau|)` holds only for the KINEMATIC wall shear -- tau/rho, m2/s2
    -- which is what an incompressible OpenFOAM solver writes. A compressible solver
    writes a stress in Pa, kg m-1 s-2, and the square root of that is not a
    velocity: it is out by sqrt(rho), about 1.2 for air at room conditions and far
    more in a gas at pressure. No case in the corpus wrote a compressible
    `wallShearStress`, so that path was never exercised by the audit that found #138
    and #144 -- untested, which is not the same as clean. Rather than leave the
    conversion assumed, the field's own `dimensions` are read with `preflight.py`'s
    reader and a stress that carries MASS is refused by name. Converting it properly
    needs rho at the wall, which a compressible case does not state as a constant,
    and a guessed rho is exactly the kind of number this script refuses to print.
    """
    dimensions = preflight.parse_dimensions(text)
    if dimensions is None:
        return "no dimensions stated in the field, so it is read as kinematic"
    if dimensions[0]:
        shown = " ".join(f"{value:g}" for value in dimensions)
        raise WallError(
            f"{path.parent.name}/{path.name} has dimensions [{shown}]: mass appears, "
            "so this is a dynamic wall shear stress in Pa and not the kinematic "
            "tau/rho an incompressible solver writes, and u_tau = sqrt(|tau|) is not "
            "a velocity for it. Converting it needs rho at the wall, which this does "
            "not read, so the source is refused rather than reported out by a factor "
            "of sqrt(rho); pass a case whose shear is kinematic, or read y+ off a "
            "yPlus function object"
        )
    return "kinematic (no mass in the field's own dimensions)"


def from_wall_shear(case: Path, requested: str | None = None,
                    nu: float | None = None,
                    dropped: list[str] | None = None) -> dict[str, Any] | None:
    """y+ built from `wallShearStress`, the mesh and nu, for a run that wrote the
    shear and never asked for y+.

    u_tau = sqrt(|tau_wall|) because the field an incompressible OpenFOAM solver
    writes is already kinematic -- tau/rho, m2/s2 -- and y is HALF the first-cell
    height, the distance to the cell centre, which is the y OpenFOAM's own yPlus
    uses. `layer_report.py` supplies that height per face straight out of
    `constant/polyMesh`, so this needs nothing the run did not already write.

    Walls only, since #144: on the cylinder case this source is the only one, so its
    `inlet`, `outlet`, `topWall` and `bottomWall` rows at y+ 0 were the whole report
    and the whole plot. The mesh `layer_report.py` already loaded here types them.

    And a wall whose shear comes back all zero is skipped by name rather than reported
    at y+ 0, because that is the entry written for a patch the function object did not
    evaluate. The type filter cannot catch it: `stepWall` on the backward-facing step
    is a `wall`, with 20 faces and `value uniform (0 0 0)`.
    """
    path = field_file(Path(case), "wallShearStress", requested, dropped)
    if path is None:
        return None
    nu = _need(nu, "nu", "constant/transportProperties", "--nu")
    if not nu > 0:
        raise WallError(
            f"nu read as {nu:g}, which no y+ can be divided by; pass --nu"
        )
    text, payloads = read_field(path)
    convention = kinematic_or_refused(path, text)
    mesh = layer_report.load(Path(case))
    values = boundary_values(text, payloads)
    patches: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    problems: list[str] = []
    for name, tau in sorted(values.items()):
        if tau is None or tau.size == 0:
            continue
        if name not in mesh["boundary"]:
            problems.append(f"the field has a patch '{name}' the mesh does not")
            continue
        kind = str(mesh["boundary"][name]["type"])
        if not is_wall_type(kind):
            excluded.append({"patch": name, "type": kind})
            continue
        height, _ = layer_report.first_cell_heights(mesh, name)
        if tau.size not in (1, height.size):
            problems.append(
                f"patch '{name}': {tau.size} shear values against {height.size} faces"
            )
            continue
        # A wall whose shear is identically zero was not measured, it was defaulted.
        # `type calculated; value uniform (0 0 0)` is what OpenFOAM writes for a patch
        # the function object did not evaluate, and no wall under a moving fluid has
        # zero shear over the whole of it. #144's non-walls are filtered above, but
        # `20260829-061843-9483/bfs` still reported the real wall `stepWall` as
        # `20 faces, y+ 0 to 0, viscous sublayer 100%` at status ok: the same
        # fabricated distribution over the same placeholder, on a patch the type
        # filter cannot catch because it IS a wall. Skipped by name instead, which is
        # this script's own answer to a source it cannot measure.
        if not np.any(tau):
            problems.append(
                f"patch '{name}' in {path.parent.name}/{path.name} is a wall whose "
                f"shear is written as {tau.size} value(s), every one zero -- what is "
                "written for a patch wallShearStress did not evaluate, rather than a "
                "measured shear -- so no y+ was derived from it"
            )
            continue
        y_plus = np.sqrt(np.abs(tau)) * (height / 2.0) / nu
        patches.append(patch_record(name, y_plus, faces=int(height.size)))
    if not patches and not excluded:
        if problems:
            raise WallError("; ".join(problems))
        return None
    detail = (
        f"{path.parent.name}/{path.name}, nu = {nu:g} m2/s, "
        "y = half the first-cell height from layer_report.py, "
        f"tau {convention}"
    )
    if payloads is not None:
        detail += ", binary, payloads decoded"
    return {
        "source": "derived from wallShearStress",
        "detail": detail,
        "time": float(path.parent.name) if locate.is_time_name(path.parent.name) else None,
        "initial": False,
        "nu": float(nu),
        "patches": patches,
        "excluded": excluded,
        "typed_from": "constant/polyMesh/boundary",
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

    dropped: list[str] = []
    found["sources"].extend(from_dat(case, time, dropped))
    for producer in (lambda: from_field(case, time, dropped),
                     lambda: from_wall_shear(case, time, read_nu, dropped)):
        try:
            source = producer()
        except (WallError, layer_report.MeshError) as error:
            found["problems"].append(str(error))
            continue
        if source is not None:
            found["problems"].extend(source.pop("problems", []))
            found["sources"].append(source)
    # What was passed over, said out loud. A source that disappears without a word
    # is the shape of both #138 and #144, so this is reported and not swallowed.
    found["problems"].extend(dropped)

    parsed, nut_from = nut_types(case)
    found["nut_from"] = nut_from
    walls = {row["patch"] for source in found["sources"] for row in source["patches"]}
    found["wall_treatment"] = {
        patch: resolve_type(parsed, patch) for patch in sorted(walls)
    }
    # Only where nothing was found AND nothing was refused: a case whose one shear
    # file was read and then declined by name has not got "no wallShearStress", and
    # saying it would talk over the refusal that is the actual answer.
    if not found["sources"] and not found["problems"]:
        found["problems"].append(
            "no y+ anywhere: no postProcessing/yPlus/*/yPlus*.dat, no written yPlus "
            "field, and no wallShearStress to derive one from"
        )
    return found


def primary(found: dict[str, Any]) -> dict[str, Any] | None:
    """The source to draw and to lead the report with: the one the run computed
    itself, before one computed here.

    A source at time 0 is last whatever it is, since #144: an initial condition led
    the report on one case, and a y+ of 0 from before the run started is not the
    number anything should be read against. It is still returned when it is all
    there is, because `--time 0` asking for it is a different thing from it being
    picked.
    """
    order = ("yPlus function object", "written yPlus field", "derived from wallShearStress")
    candidates = [s for s in found["sources"] if not s.get("initial")]
    for pool in (candidates, found["sources"]):
        for name in order:
            for source in pool:
                if source["source"] == name:
                    return source
        if pool:
            return pool[0]
    return None


def _range(row: dict[str, Any]) -> str:
    if row.get("uniform"):
        return f"y+ {row['mean']:.3g} uniform over the patch"
    return f"y+ {row['min']:.3g} to {row['max']:.3g} (mean {row['mean']:.3g})"


def _regions(row: dict[str, Any], brief: bool = False) -> str:
    """Fractions where there is a distribution to take them over; otherwise the
    regions the range reaches, labelled as spanned so the two are never read alike.

    A `uniform` entry gets a third wording, because of #144: one value for a whole
    patch was being printed as `viscous sublayer 100%`, which reads as a per-face
    fraction over a distribution that the file does not contain.
    """
    if row.get("uniform"):
        tail = " (uniform)" if brief else (
            " (uniform; one value for the whole patch, which is not a distribution)"
        )
        return regime_of(row["mean"]) + tail
    if row["fractions"] is not None:
        parts = [f"{name} {value:g}%" for name, value in row["fractions"].items() if value]
        return ", ".join(parts) or "no faces in any region"
    tail = " (spanned)" if brief else " (spanned; a summary carries no fractions)"
    return ", ".join(row["spans"]) + tail


def _excluded(source: dict[str, Any]) -> str:
    """The non-wall patches this source left out, with the type that says why.

    Listed and not merely dropped: #144 was non-wall patches reported as measured
    walls, and the fix for it must not become a second silent drop."""
    rows = source.get("excluded") or []
    if not rows:
        return ""
    named = ", ".join(f"{row['patch']} (type {row['type']})" for row in rows)
    where = source.get("typed_from") or "the case"
    return (
        f"excluded, not walls and y+ is a wall quantity: {named}"
        f" -- typed from {where}"
    )


def findings(found: dict[str, Any]) -> list[Finding]:
    """The measurement in preflight's register: what was measured, what it means."""
    out: list[Finding] = []
    for source in found["sources"]:
        measured = "; ".join(
            f"{row['patch']}: {_range(row)}"
            + (f", {row['faces']} faces" if row["faces"] else "")
            for row in source["patches"]
        ) or "no wall patch in this source"
        where = source["detail"]
        at = source.get("time")
        if at is not None:
            where += f" at time {at:g}"
        left_out = _excluded(source)
        if left_out:
            measured += f" | {left_out}"
        meaning = "; ".join(
            f"{row['patch']}: {_regions(row)}" for row in source["patches"]
        )
        if source.get("initial"):
            meaning = (
                "time 0 is the initial condition the field was written from and not "
                "a result, so this is not a measurement of what the run reached"
                + (f". {meaning}" if meaning else "")
            )
        out.append(Finding(
            "y+", "skipped" if source.get("initial") else "ok",
            f"{source['source']}, {where} -- {measured}", meaning,
        ))

    treatment = found["wall_treatment"]
    lead = primary(found)
    if treatment and lead and lead.get("initial"):
        out.append(Finding(
            "wall-treatment", "skipped",
            f"{found['nut_from'] or 'nut'}: " + ", ".join(
                f"{patch}: {kind or 'no nut entry'}"
                for patch, kind in sorted(treatment.items())
            ),
            "the only y+ this case offers is the field at time 0, which is the "
            "initial condition and not what the run reached, so there is nothing "
            "here to put beside the band that wall condition is written for",
        ))
    elif treatment and lead:
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
        if not source["patches"]:
            lines.append("    (no wall patch in this source)")
        left_out = _excluded(source)
        if left_out:
            lines.append(f"    {left_out}")
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
