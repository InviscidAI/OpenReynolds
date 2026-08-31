#!/usr/bin/env python3
"""The cheap questions to ask before an expensive run, each answered with a diagnosis.

A CFD run that fails usually fails for one of a small number of dull reasons, and
almost all of them are visible before the solver starts: a patch in the mesh that no
field has an entry for, an STL still in millimetres sitting in a domain measured in
metres, an `empty` patch that one field calls `zeroGradient`, a deltaT that puts the
Courant number at forty, a write cadence that needs more disk than the volume has.
Each of those costs the same thing when it is found late -- a queued hour, a job that
dies twenty minutes in with four lines of fatal error, or worse, a run that finishes
and is quietly wrong. Finding them costs seconds.

what comes out

One finding per check, and each finding says three separate things: what was
**measured** (a number or a list, so it can be argued with), what that **means**, and
a **suggested repair**. Not a dumped log. The three are kept apart on purpose,
because a log line pasted into a report tells the next reader what the tool saw and
not what it thought, and a bare verdict tells them what it thought and not what it
saw.

The repair is a suggestion. This script edits nothing, refuses nothing, and blocks
nothing; a `fail` is its reading of what it measured, and the reading can be wrong --
a mesh whose non-orthogonality is 78 is fine with the right schemes, an STL that is
genuinely 200 m across is a building. The exit code (0 when nothing failed, 1 when
something did) exists so a shell script can gate on it if somebody wants that. What
to do about any of this is yours to decide, including deciding it does not apply.

the checks

- `geometry`   STL bounding box, the millimetres-read-as-metres heuristic, open
               edges, non-manifold edges, and triangles wound against their neighbours
- `patches`    every patch in `constant/polyMesh/boundary` against every `0/` field,
               both directions. This one alone catches a large share of failed starts
- `empty`      a 2D study's one cell in z, and `empty` declared in the mesh *and* in
               every field. A single field left on `zeroGradient` is the whole bug
- `reynolds`   the Re implied by U, L and nu, against the Re you say you meant
- `cells`      the snappy prediction (via `cells_estimate.py`) against the mesh built
- `checkmesh`  quality numbers out of a checkMesh log (via `mesh_digest.py`), with
               thresholds attached -- which is the one thing mesh_digest declines to do
- `probe`      the solver run for a single step in a *copy* of the case. The real
               `controlDict` is never touched
- `courant`    U * deltaT / dx for the chosen deltaT, against what the solver is
- `residuals`  divergence and continuity drift in a running or finished log
- `units`      incompressible OpenFOAM `p` is kinematic (m2/s2); force coefficients
               computed as though it were Pa are wrong by a factor of rho
- `disk`       bytes per write time times the number of write times, against `df`

    python3 preflight.py /work/case
    python3 preflight.py /work/case --checks patches,empty,units
    python3 preflight.py /work/case --re 3900 --u 1.0 --l 0.1
    python3 preflight.py /work/case --json --out /work/study/preflight.json
    python3 preflight.py /work/case --no-probe --log log.simpleFoam

Exit code is 0 when nothing failed and 1 when something did.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells_estimate  # noqa: E402  (sibling script, not a package)
import log_digest  # noqa: E402
import mesh_digest  # noqa: E402
import study_state  # noqa: E402


# -- what a finding is -------------------------------------------------------------


STATUSES = ("fail", "warn", "ok", "skipped")
"""In the order they are printed. Worst first, because a report is read from the top
and the reason the run is about to die should not be below eight lines of `ok`."""

_SEVERITY = {"fail": 3, "warn": 2, "ok": 1, "skipped": 0}


class Finding(NamedTuple):
    """One check's answer.

    `measured` is evidence and should survive being quoted on its own; `meaning` is
    the interpretation and is allowed to be wrong; `repair` is a suggestion and is
    allowed to be ignored.
    """

    check: str
    status: str
    measured: str
    meaning: str = ""
    repair: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "check": self.check,
            "status": self.status,
            "measured": self.measured,
            "means": self.meaning,
            "repair": self.repair,
        }


class Intent(NamedTuple):
    """What the person running this says the case is supposed to be.

    All optional. Nothing is guessed in their place: a length scale invented by
    a script is a Reynolds number invented by a script, and a made-up number that
    disagrees with the case is worse than no number.

    `resolve` is different in kind from the other three. They describe the flow; it
    describes the *answer* -- the feature the run has to show for the run to have been
    worth doing. "shock", "interface", "wake". It is the one piece of information that
    lets a check ask whether the discretisation can produce the deliverable, rather
    than only whether the case will run.
    """

    reynolds: float | None = None
    velocity: float | None = None
    length: float | None = None
    resolve: str = ""


def worst_status(findings) -> str:
    return max((f.status for f in findings), key=lambda s: _SEVERITY.get(s, 0), default="ok")


def count_phrase(number: int, noun: str, plural: str = "") -> str:
    """`1 edge` / `8 edges`. Small, but a finding is read by a person and "1 edges"
    reads as a bug in the tool and puts the rest of the line in doubt."""
    return f"{number} {noun}" if number == 1 else f"{number} {plural or noun + 's'}"


def escalate(current: str, candidate: str) -> str:
    """The worse of two statuses. Findings accumulate reasons and the status has to
    follow the worst of them, never the last one appended."""
    return candidate if _SEVERITY.get(candidate, 0) > _SEVERITY.get(current, 0) else current


def summarise(findings) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    return counts


# -- reading OpenFOAM dictionaries -------------------------------------------------


_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_ENTRY_LINE = re.compile(r"^\s*(\w+)\s+([^;{}]+?)\s*;", re.M)
_ENTRY_ANY = re.compile(r"(\w+)\s+([^;{}]+?)\s*;")
"""Two spellings of the same `key value;` entry. The line-anchored one is for
rewriting a dictionary, where a match has to correspond to a line that can be
replaced; the loose one is for reading, because a hand-written `boundary` file puts
a whole patch on one line and reading only the first entry on it loses `nFaces`."""
_NOTE = re.compile(r'note\s+"([^"]*)"')
_NOTE_FIELD = re.compile(r"(\w+):\s*(\d+)")
_DIMENSIONS = re.compile(r"^\s*dimensions\s*\[([^\]]*)\]\s*;", re.M)
_NU = re.compile(
    r"^\s*nu\s+(?:nu\s+)?(?:\[[^\]]*\]\s*)?([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*;", re.M
)
_UNIFORM_VECTOR = re.compile(r"uniform\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")

REGEX_CHARACTERS = set(".*|()[]?+^$\\")
"""What makes a `boundaryField` key a pattern rather than a patch name. OpenFOAM
matches unquoted keys literally first and only then as regexes, but in practice a key
containing any of these was written to be a pattern."""

CONSTRAINT_TYPES = frozenset({
    "empty", "symmetry", "symmetryPlane", "wedge", "cyclic", "cyclicAMI",
    "cyclicACMI", "processor", "processorCyclic", "nonConformalCyclic",
})
"""Patch types whose field entry is decided by the mesh, not by the physics. A
`#includeEtc "caseDicts/setConstraintTypes"` in a boundaryField covers exactly these,
which is why their absence from the written-out entries is not a missing entry."""

STEADY_APPLICATIONS = frozenset({
    "simpleFoam", "potentialFoam", "boundaryFoam", "porousSimpleFoam",
    "SRFSimpleFoam", "adjointShapeOptimizationFoam", "buoyantSimpleFoam",
    "rhoSimpleFoam", "rhoPorousSimpleFoam", "overSimpleFoam",
})
"""Solvers whose `deltaT` counts iterations rather than seconds. Reporting a Courant
number for one of these would be reporting a number that does not exist."""


def strip_comments(text: str) -> str:
    return _COMMENTS.sub("", text or "")


def read_text(path: Path) -> str:
    """The file, or its `.gz`, or an empty string. Never raises: a check that cannot
    read its input reports `skipped`, and a missing file is not an exception."""
    path = Path(path)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
        packed = Path(str(path) + ".gz")
        if packed.is_file():
            with gzip.open(packed, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.read()
    except OSError:
        return ""
    return ""


def block_body(text: str, keyword: str) -> str:
    """The inside of `keyword { ... }`, brace-matched.

    Brace-matched rather than regex-delimited because `boundaryField` blocks nest --
    every patch entry is itself a block -- and a non-greedy match to the first `}`
    stops inside the first patch.
    """
    body = strip_comments(text)
    match = re.search(r"(?<![\w.])" + re.escape(keyword) + r"\s*\{", body)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    for index in range(start, len(body)):
        if body[index] == "{":
            depth += 1
        elif body[index] == "}":
            depth -= 1
            if depth == 0:
                return body[start + 1:index]
    return body[start + 1:]


def block_entries(body: str) -> list[tuple[str, str]]:
    """The `name { ... }` entries at the top level of a block, in file order.

    The name is taken as the last whitespace-separated token before the brace, so a
    `#includeEtc "..."` line or a stray `;` entry sitting above a patch does not get
    glued onto that patch's name.
    """
    entries: list[tuple[str, str]] = []
    depth = 0
    pending = 0
    name = ""
    inner = 0
    for index, char in enumerate(body):
        if char == "{":
            if depth == 0:
                tokens = body[pending:index].split()
                name = tokens[-1] if tokens else ""
                inner = index + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth <= 0:
                if depth == 0:
                    entries.append((name, body[inner:index]))
                depth = 0
                pending = index + 1
        elif char == ";" and depth == 0:
            pending = index + 1
    return entries


def outer_text(body: str) -> str:
    """The block's own text with every nested block removed.

    A `;` is left where each nested block was, so the name that introduced it cannot
    run into the entry that follows and be read as one long `key value;` pair.
    """
    kept: list[str] = []
    depth = 0
    for char in body:
        if char == "{":
            depth += 1
            if depth == 1:
                kept.append(";")
        elif char == "}":
            depth = max(0, depth - 1)
        elif depth == 0:
            kept.append(char)
    return "".join(kept)


def entry_values(body: str) -> dict[str, str]:
    """The `key value;` pairs directly inside a block, ignoring nested ones.

    A repeated key takes its last value, which is what OpenFOAM does with one.
    """
    found: dict[str, str] = {}
    for match in _ENTRY_ANY.finditer(outer_text(strip_comments(body))):
        found[match.group(1)] = match.group(2).strip()
    return found


def parse_control(text: str) -> dict[str, str]:
    """Top-level entries of a `controlDict`, as written.

    Only the top level: a `functions {}` block routinely carries its own
    `writeControl` and `writeInterval`, and reading a function object's write cadence
    as the case's would put the disk estimate out by whatever factor separates them.
    """
    return entry_values(text)


def as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_boundary(text: str) -> list[dict[str, Any]]:
    """Patch name, type, group and face count out of `constant/polyMesh/boundary`."""
    patches: list[dict[str, Any]] = []
    for name, body in block_entries(strip_comments(text)):
        if not name or name == "FoamFile":
            continue
        values = entry_values(body)
        faces = as_float(values.get("nFaces"))
        patches.append({
            "name": name,
            "type": values.get("type", ""),
            "nFaces": int(faces) if faces is not None else 0,
            "inGroups": values.get("inGroups", ""),
        })
    return patches


def parse_block_mesh_boundary(text: str) -> list[dict[str, Any]]:
    """Patch names and types out of a `blockMeshDict`'s own `boundary` list.

    The same information `constant/polyMesh/boundary` carries, read from the file
    that will produce it. This exists because of when preflight is worth running:
    before the mesh is built, which is exactly when `constant/polyMesh/boundary`
    does not exist yet. Reading only the built mesh made the `empty` check report
    "no patch is declared empty" for every case that had not been meshed -- the
    check that most wants to run early was the one that could not.

    Face counts are not available here (they are the mesher's output, not its
    input), so they come back as 0 and only the name and the type are used.
    """
    # `boundary ( ... );` -- a parenthesised list, unlike polyMesh/boundary's braces,
    # so it is matched here rather than with block_body.
    stripped = strip_comments(text)
    match = re.search(r"(?<![\w.])boundary\s*\(", stripped)
    if not match:
        return []
    depth = 0
    start = match.end() - 1
    body = ""
    for index in range(start, len(stripped)):
        if stripped[index] == "(":
            depth += 1
        elif stripped[index] == ")":
            depth -= 1
            if depth == 0:
                body = stripped[start + 1:index]
                break
    if not body:
        return []
    patches: list[dict[str, Any]] = []
    for name, entry in block_entries(body):
        if not name:
            continue
        values = entry_values(entry)
        patches.append({
            "name": name,
            "type": values.get("type", ""),
            "nFaces": 0,
            "inGroups": values.get("inGroups", ""),
        })
    return patches


def parse_boundary_field(text: str) -> dict[str, Any]:
    """What a `0/` field says about patches: literal names, patterns, types.

    `includes` records whether the block pulls entries in with a `#include*`
    directive. It almost always means `caseDicts/setConstraintTypes`, and a field
    that does that has covered its empty/cyclic/wedge patches without naming them --
    so the difference matters to whether a missing name is a bug.
    """
    body = block_body(text, "boundaryField")
    names: list[str] = []
    patterns: list[str] = []
    types: dict[str, str] = {}
    for name, inner in block_entries(body):
        if not name:
            continue
        bare = name.strip('"')
        values = entry_values(inner)
        types[bare] = values.get("type", "")
        if name.startswith('"') or (set(bare) & REGEX_CHARACTERS):
            patterns.append(bare)
        else:
            names.append(bare)
    return {
        "names": names,
        "patterns": patterns,
        "types": types,
        "includes": "#include" in body,
        "present": bool(body.strip()),
    }


def parse_dimensions(text: str) -> tuple[float, ...] | None:
    """The seven exponents of a field's `dimensions [...]` entry."""
    match = _DIMENSIONS.search(strip_comments(text))
    if not match:
        return None
    parts = match.group(1).replace(",", " ").split()
    values = [as_float(part) for part in parts]
    if len(values) != 7 or any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def parse_nu(text: str) -> float | None:
    """Kinematic viscosity from transportProperties, old or new spelling.

    A zero or negative value is returned as it is rather than treated as absent: `nu
    0;` is a real thing people write while sweeping a parameter, and "there is no
    viscosity here" and "the viscosity is zero" want different answers.
    """
    match = _NU.search(strip_comments(text))
    return as_float(match.group(1)) if match else None


def parse_uniform_velocity(text: str) -> float | None:
    """The largest uniform vector magnitude in a `0/U`.

    Which is the inlet in every case shaped like flow past something: the internal
    field and the walls are zero and the inlet is not. A heuristic, and only ever
    used as a stand-in for a `--u` nobody supplied.
    """
    best = 0.0
    for match in _UNIFORM_VECTOR.finditer(strip_comments(text)):
        try:
            vector = [float(part) for part in match.groups()]
        except ValueError:
            continue
        best = max(best, float(np.linalg.norm(vector)))
    return best if best > 0 else None


def parse_owner_note(text: str) -> dict[str, int]:
    """The counts OpenFOAM writes into the header note of `constant/polyMesh/owner`.

    They are already there, so the cheapest cell count available is a regex over the
    first few hundred bytes rather than a mesh read.
    """
    match = _NOTE.search(text or "")
    if not match:
        return {}
    return {key: int(value) for key, value in _NOTE_FIELD.findall(match.group(1))}


def field_components(text: str) -> int:
    """3 for a vector field, 1 for a scalar one, judged by its internalField."""
    body = strip_comments(text)
    match = re.search(r"internalField\s+\w+\s*(\(?)", body)
    if match and match.group(1) == "(":
        return 3
    if "List<vector>" in body or "volVectorField" in body:
        return 3
    return 1


# -- reading the case as a whole ---------------------------------------------------


FIELD_DIRS = ("0", "0.orig", "0.org")

SURFACE_DIRS = ("constant/triSurface", "constant/geometry", "constant/trisurface")

SURFACE_SUFFIXES = (".stl", ".stlb")


class Case:
    """A case directory, read once and remembered.

    Ten checks over one case otherwise re-read `controlDict` ten times and the STL
    twice, and on a mesh directory that is measured in gigabytes the difference
    between reading once and reading per check is the difference between a preflight
    and a wait.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._text: dict[str, str] = {}
        self._cache: dict[str, Any] = {}

    def read(self, relative: str) -> str:
        if relative not in self._text:
            self._text[relative] = read_text(self.path / relative)
        return self._text[relative]

    def _memo(self, key: str, build: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def control(self) -> dict[str, str]:
        return self._memo("control", lambda: parse_control(self.read("system/controlDict")))

    @property
    def application(self) -> str:
        return self.control.get("application", "")

    @property
    def boundary(self) -> list[dict[str, Any]]:
        return self._memo(
            "boundary", lambda: parse_boundary(self.read("constant/polyMesh/boundary"))
        )

    @property
    def field_dir(self) -> Path | None:
        def find() -> Path | None:
            for name in FIELD_DIRS:
                candidate = self.path / name
                if candidate.is_dir():
                    return candidate
            return None

        return self._memo("field_dir", find)

    @property
    def field_texts(self) -> dict[str, str]:
        def read_all() -> dict[str, str]:
            directory = self.field_dir
            if directory is None:
                return {}
            found: dict[str, str] = {}
            for entry in sorted(directory.iterdir()):
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                if entry.name.endswith(".gz"):
                    # `read_text` finds the .gz for a plain name; here the plain name
                    # is what has to be derived from the file that exists.
                    found[entry.name[:-3]] = read_text(entry.parent / entry.name[:-3])
                else:
                    found[entry.name] = read_text(entry)
            return found

        return self._memo("field_texts", read_all)

    @property
    def fields(self) -> dict[str, dict[str, Any]]:
        return self._memo(
            "fields",
            lambda: {name: parse_boundary_field(text) for name, text in self.field_texts.items()},
        )

    @property
    def viscosity(self) -> float | None:
        def find() -> float | None:
            for name in (
                "constant/transportProperties",
                "constant/physicalProperties",
                "constant/momentumTransport",
            ):
                value = parse_nu(self.read(name))
                if value is not None:
                    return value
            return None

        return self._memo("nu", find)

    @property
    def surfaces(self) -> list[Path]:
        def find() -> list[Path]:
            found: list[Path] = []
            seen: set[str] = set()
            for relative in SURFACE_DIRS:
                directory = self.path / relative
                if not directory.is_dir():
                    continue
                for entry in sorted(directory.iterdir()):
                    if not entry.is_file() or entry.suffix.lower() not in SURFACE_SUFFIXES:
                        continue
                    key = str(entry.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        found.append(entry)
            return found

        return self._memo("surfaces", find)

    @property
    def cell_count(self) -> int | None:
        """Cells in the built mesh, from the `owner` header note or a checkMesh log."""

        def find() -> int | None:
            note = parse_owner_note(self.read("constant/polyMesh/owner"))
            if note.get("nCells"):
                return int(note["nCells"])
            digest = mesh_digest.parse(self.check_mesh_log_text)
            cells = digest.get("counts", {}).get("cells")
            return int(cells) if cells else None

        return self._memo("cells", find)

    @property
    def check_mesh_log_text(self) -> str:
        def find() -> str:
            for name in ("log.checkMesh", "log.checkmesh", "checkMesh.log", "log/checkMesh"):
                text = self.read(name)
                if text:
                    return text
            return ""

        return self._memo("checkmesh_text", find)

    def solver_log(self, explicit: str | Path | None = None) -> Path | None:
        """The log the residual check should read: the one named, else the newest
        `log.<solver>` that is not one of the meshing utilities' logs."""
        if explicit:
            candidate = Path(explicit)
            if not candidate.is_absolute():
                candidate = self.path / candidate
            return candidate if candidate.is_file() else None
        if self.application:
            named = self.path / f"log.{self.application}"
            if named.is_file():
                return named
        skip = {"blockMesh", "checkMesh", "snappyHexMesh", "surfaceFeatureExtract",
                "decomposePar", "reconstructPar", "extrudeMesh", "topoSet"}
        candidates = [
            path for path in self.path.glob("log.*")
            if path.is_file() and path.name.split(".", 1)[-1] not in skip
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)


# -- geometry ----------------------------------------------------------------------


MILLIMETRE_SUSPICION = 100.0
"""A bounding box wider than this many metres. Wind tunnels and buildings are real,
so this is a warning on its own; it becomes a failure only when the domain the STL
sits in disagrees with it by about the factor that separates mm from m."""

SCALE_FACTOR = 100.0
"""How far the STL and the background mesh have to disagree before the disagreement
is called a units error rather than a modelling choice. 1000 is the actual factor;
100 leaves room for an STL that really is a tenth of the domain."""

TOPOLOGY_TRIANGLE_LIMIT = 400_000
"""Above this the edge bookkeeping starts costing more than the preflight is worth.
It is reported as not computed rather than silently skipped."""


def read_triangles(path: Path) -> np.ndarray | None:
    """The triangles of an STL, binary or ASCII, as (n, 3, 3).

    `cells_estimate.stl_area` reads the same files but returns only summary numbers,
    and the edge bookkeeping below needs the vertices themselves.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    if len(raw) >= 84:
        count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        if len(raw) == 84 + count * 50 and count:
            data = np.frombuffer(raw[84:], dtype=np.uint8).reshape(count, 50)
            floats = data[:, :48].copy().view("<f4").reshape(count, 4, 3)
            return floats[:, 1:, :].astype(np.float64)
    text = raw.decode("utf-8", errors="replace")
    values = re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text)
    if len(values) < 3:
        return None
    points = np.array(values, dtype=float)
    usable = len(points) - len(points) % 3
    return points[:usable].reshape(-1, 3, 3)


def surface_topology(triangles: np.ndarray | None) -> dict[str, Any]:
    """Open edges, non-manifold edges and edges walked the same way twice.

    Vertices are welded on rounded coordinates first, because an STL stores each
    triangle's corners independently and a surface that is closed geometrically has
    three copies of every vertex on disk. Without the weld every edge looks open.

    The third number is the one that is hard to get any other way: on a closed
    surface with consistent winding each directed edge (a -> b) occurs exactly once,
    the neighbouring triangle walking it back as (b -> a). An edge that appears twice
    in the same direction is two triangles facing opposite ways, which is what
    snappyHexMesh reads as a hole and what makes it mesh the outside of an object.
    """
    if triangles is None or len(triangles) == 0:
        return {"triangles": 0, "computed": False, "note": "no triangles read"}
    if len(triangles) > TOPOLOGY_TRIANGLE_LIMIT:
        return {
            "triangles": int(len(triangles)),
            "computed": False,
            "note": f"over {TOPOLOGY_TRIANGLE_LIMIT:,} triangles, edges not counted here",
        }

    flat = triangles.reshape(-1, 3)
    span = float(np.max(flat.max(axis=0) - flat.min(axis=0)))
    scale = span if span > 0 else 1.0
    welded = np.round(flat / scale, 9)
    _unique, index = np.unique(welded, axis=0, return_inverse=True)
    corners = np.asarray(index, dtype=np.int64).reshape(-1, 3)
    vertices = int(corners.max()) + 1

    starts = np.concatenate([corners[:, 0], corners[:, 1], corners[:, 2]])
    ends = np.concatenate([corners[:, 1], corners[:, 2], corners[:, 0]])
    keep = starts != ends  # a degenerate triangle contributes no real edge
    starts, ends = starts[keep], ends[keep]

    directed = starts * vertices + ends
    low = np.minimum(starts, ends)
    high = np.maximum(starts, ends)
    undirected = low * vertices + high

    _edges, edge_counts = np.unique(undirected, return_counts=True)
    _walks, walk_counts = np.unique(directed, return_counts=True)

    repeated_corner = (
        (corners[:, 0] == corners[:, 1])
        | (corners[:, 1] == corners[:, 2])
        | (corners[:, 0] == corners[:, 2])
    )

    return {
        "triangles": int(len(triangles)),
        "computed": True,
        "open_edges": int((edge_counts == 1).sum()),
        "non_manifold_edges": int((edge_counts > 2).sum()),
        "flipped_edges": int((walk_counts > 1).sum()),
        "degenerate_triangles": int(repeated_corner.sum()),
    }


def scale_diagnosis(
    extent, domain_extent=None, length: float | None = None
) -> dict[str, Any]:
    """Whether the surface looks like it is in the units OpenFOAM will read it in.

    OpenFOAM reads an STL as metres, full stop, and CAD exports millimetres by
    default. The two facts together are the single most common way a snappy build is
    wasted: the geometry lands a thousand times too big, misses the refinement region
    entirely, and the mesh comes back either empty or enormous.
    """
    span = float(max(extent)) if len(extent) else 0.0
    result: dict[str, Any] = {"span": span, "status": "ok", "note": "", "repair": ""}
    if span <= 0:
        result.update(
            status="fail",
            note="the bounding box has no extent -- the file read as zero-sized",
            repair="check the file is a real STL and not empty or truncated",
        )
        return result

    domain_span = float(max(domain_extent)) if domain_extent is not None and len(domain_extent) else 0.0
    if domain_span > 0:
        ratio = span / domain_span
        if ratio >= SCALE_FACTOR:
            result.update(
                status="fail",
                note=(
                    f"the surface is {ratio:.4g}x the background mesh ({span:.4g} m against "
                    f"{domain_span:.4g} m), which is the shape of a millimetre file read as metres"
                ),
                repair=(
                    "scale the surface by 0.001 -- surfaceTransformPoints -scale "
                    "'(0.001 0.001 0.001)' in.stl out.stl -- or set scale in the "
                    "snappyHexMeshDict geometry entry"
                ),
            )
            return result
        if ratio <= 1.0 / SCALE_FACTOR:
            result.update(
                status="fail",
                note=(
                    f"the surface is {ratio:.4g}x the background mesh ({span:.4g} m against "
                    f"{domain_span:.4g} m); it will be a speck in the domain"
                ),
                repair=(
                    "scale the surface up, or shrink the blockMeshDict domain -- one of "
                    "the two is in the wrong units"
                ),
            )
            return result

    if length and length > 0:
        ratio = span / length
        if ratio >= SCALE_FACTOR or ratio <= 1.0 / SCALE_FACTOR:
            result.update(
                status="warn",
                note=(
                    f"the surface spans {span:.4g} m against a stated length scale of "
                    f"{length:.4g} m, a factor of {ratio:.4g}"
                ),
                repair="check which of the two is in the wrong units before meshing",
            )
            return result

    if span > MILLIMETRE_SUSPICION:
        result.update(
            status="warn",
            note=(
                f"the surface spans {span:.4g} m; OpenFOAM reads STL coordinates as "
                "metres, and a CAD export in millimetres looks exactly like this"
            ),
            repair="if it was exported in millimetres, scale it by 0.001",
        )
        return result
    if span < 1e-3:
        result.update(
            status="warn",
            note=f"the surface spans {span:.4g} m, under a millimetre end to end",
            repair="check the export units if the object is not genuinely that small",
        )
    return result


def check_geometry(case: Case, intent: Intent) -> list[Finding]:
    surfaces = case.surfaces
    if not surfaces:
        return [Finding(
            "geometry", "skipped",
            f"no .stl under {', '.join(SURFACE_DIRS)}",
            "nothing to check -- a blockMesh-only case has no surface geometry",
        )]

    _delta, _volume, domain_extent = cells_estimate.block_mesh_delta(
        case.read("system/blockMeshDict")
    )
    findings: list[Finding] = []
    for path in surfaces:
        area, extent, count = cells_estimate.stl_area(path)
        scale = scale_diagnosis(extent, domain_extent or None, intent.length)
        measured = (
            f"{path.name}: {count:,} triangles, bbox "
            f"{extent[0]:.4g} x {extent[1]:.4g} x {extent[2]:.4g} m, area {area:.4g} m2"
        )
        topology = surface_topology(read_triangles(path))
        if topology.get("computed"):
            measured += (
                f"; open edges {topology['open_edges']}, non-manifold "
                f"{topology['non_manifold_edges']}, same-direction (flipped) "
                f"{topology['flipped_edges']}"
            )
        elif topology.get("note"):
            measured += f"; topology not computed ({topology['note']})"

        status = scale["status"]
        meanings = [scale["note"]] if scale["note"] else []
        repairs = [scale["repair"]] if scale["repair"] else []

        if topology.get("open_edges"):
            status = escalate(status, "fail")
            meanings.append(
                count_phrase(topology["open_edges"], "open edge")
                + ": the surface is not closed, so snappyHexMesh cannot tell inside "
                "from outside and castellation leaks out through the hole"
            )
            repairs.append(
                "surfaceCheck the file, then close it in the CAD tool, or run "
                "surfaceClean on it before meshing"
            )
        if topology.get("flipped_edges"):
            status = escalate(status, "fail")
            meanings.append(
                count_phrase(topology["flipped_edges"], "edge is", "edges are")
                + " walked the same way by both of its triangles: some facets face "
                "inwards, which reads to snappy as a hole even where the surface is "
                "geometrically watertight"
            )
            repairs.append(
                f"surfaceOrient {path.name} {path.name} to make the winding consistent"
            )
        if topology.get("non_manifold_edges"):
            status = escalate(status, "warn")
            meanings.append(
                count_phrase(topology["non_manifold_edges"], "edge has", "edges have")
                + " more than two triangles on it, usually two parts sharing a face "
                "rather than one solid"
            )
            repairs.append("split the parts into separate STL files, or merge them properly")
        if topology.get("degenerate_triangles"):
            meanings.append(
                count_phrase(topology["degenerate_triangles"], "triangle has", "triangles have")
                + " a repeated corner and no area"
            )

        findings.append(Finding(
            "geometry", status, measured,
            "; ".join(meanings) or "bounding box and edge topology look ordinary",
            "; ".join(repairs),
        ))
    return findings


# -- patch-name consistency --------------------------------------------------------


def compile_pattern(pattern: str):
    try:
        return re.compile(pattern)
    except re.error:
        return None


def pattern_covers(spec: dict[str, Any], name: str) -> bool:
    """Whether one of a field's regex keys matches a patch name."""
    for text in spec.get("patterns", []):
        compiled = compile_pattern(text)
        if compiled is not None and compiled.fullmatch(name):
            return True
    return False


def entry_type_for(spec: dict[str, Any], name: str) -> str | None:
    """The patchField type a field gives one patch, or None if it gives it none.

    A literal key beats a pattern, which is OpenFOAM's own order, and it matters
    here: a field with `".*" { type zeroGradient; }` and an explicit entry for the
    empty patch is correct, and one with only the wildcard is the bug.
    """
    types = spec.get("types", {})
    if name in spec.get("names", []):
        return types.get(name, "")
    for text in spec.get("patterns", []):
        compiled = compile_pattern(text)
        if compiled is not None and compiled.fullmatch(name):
            return types.get(text, "")
    return None


def patch_consistency(
    mesh_patches: list[dict[str, Any]], fields: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per field: which mesh patches it has no entry for, and which entries it has
    for patches that do not exist.

    Both directions matter and they fail differently. A missing entry stops the
    solver on the first read with `Cannot find patchField entry for <patch>`; a
    spare entry is silently ignored, which is worse, because the boundary condition
    somebody thought they set is simply not applied and the run completes.
    """
    mesh_names = [patch["name"] for patch in mesh_patches]
    constraint = {
        patch["name"] for patch in mesh_patches if patch.get("type") in CONSTRAINT_TYPES
    }
    rows: list[dict[str, Any]] = []
    for field in sorted(fields):
        spec = fields[field]
        literals = set(spec.get("names", []))
        patterns = [compile_pattern(text) for text in spec.get("patterns", [])]
        patterns = [pattern for pattern in patterns if pattern is not None]
        covered: set[str] = set()
        for name in mesh_names:
            if name in literals:
                covered.add(name)
            elif any(pattern.fullmatch(name) for pattern in patterns):
                covered.add(name)
            elif spec.get("includes") and name in constraint:
                # `#includeEtc "caseDicts/setConstraintTypes"` writes the entries for
                # exactly these, so their absence by name is not an absence.
                covered.add(name)
        rows.append({
            "field": field,
            "missing": [name for name in mesh_names if name not in covered],
            "extra": sorted(literals - set(mesh_names)),
            "has_boundary_field": bool(spec.get("present")),
        })
    return rows


def check_patches(case: Case, intent: Intent) -> list[Finding]:
    mesh_patches = case.boundary
    fields = case.fields
    if not mesh_patches:
        return [Finding(
            "patches", "skipped",
            "constant/polyMesh/boundary is missing or unreadable",
            "the mesh has not been built yet, so there is nothing to compare fields against",
        )]
    if not fields:
        return [Finding(
            "patches", "fail",
            f"{len(mesh_patches)} patches in the mesh, no readable field files in "
            f"{'/'.join(FIELD_DIRS)}",
            "the solver has no initial conditions to read and will stop immediately",
            "create a 0/ directory with one file per field the solver needs",
        )]

    rows = patch_consistency(mesh_patches, fields)
    broken = [row for row in rows if row["missing"] or row["extra"]]
    names = ", ".join(patch["name"] for patch in mesh_patches)
    if not broken:
        return [Finding(
            "patches", "ok",
            f"{len(mesh_patches)} patches ({names}) all covered by {len(fields)} fields",
            "every patch has an entry in every field and no entry names a patch that "
            "does not exist",
        )]

    measured_parts = []
    meaning_parts = []
    repair_parts = []
    for row in broken:
        if not row["has_boundary_field"]:
            measured_parts.append(f"0/{row['field']}: no boundaryField block at all")
            meaning_parts.append(f"{row['field']} is not a field file, or it is malformed")
            continue
        if row["missing"]:
            measured_parts.append(
                f"0/{row['field']} has no entry for {', '.join(row['missing'])}"
            )
            repair_parts.append(
                f"add a boundaryField entry for {', '.join(row['missing'])} to 0/{row['field']}"
            )
        if row["extra"]:
            measured_parts.append(
                f"0/{row['field']} has entries for {', '.join(row['extra'])}, "
                "which are not patches in this mesh"
            )
            repair_parts.append(
                f"remove or rename {', '.join(row['extra'])} in 0/{row['field']} -- "
                "check it against the names in constant/polyMesh/boundary"
            )
    if any(row["missing"] for row in broken):
        meaning_parts.append(
            "a patch with no entry stops the solver on its first read with "
            "'Cannot find patchField entry for <patch>'"
        )
    if any(row["extra"] for row in broken):
        meaning_parts.append(
            "an entry naming no patch is ignored without a message, so that boundary "
            "condition is not applied and the run finishes looking fine"
        )
    return [Finding(
        "patches", "fail",
        f"mesh patches: {names}. " + "; ".join(measured_parts),
        "; ".join(meaning_parts),
        "; ".join(repair_parts),
    )]


# -- 2D and the empty patch --------------------------------------------------------


def empty_diagnosis(
    mesh_patches: list[dict[str, Any]],
    fields: dict[str, dict[str, Any]],
    block_counts: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Whether this looks like a 2D study and whether `empty` is honoured everywhere.

    OpenFOAM has no 2D solver. A 2D case is a 3D case one cell thick whose two
    end faces are declared `empty`, and every one of those words has to line up: one
    cell in the third direction in the blockMeshDict, `type empty;` on the patch in
    the mesh, and `type empty;` for that patch in every single field. Any field left
    on `zeroGradient` or `fixedValue` there stops the solver with 'incompatible
    patch and patchField types', and a mesh with two cells in z simply solves a very
    thin 3D problem while everyone believes it is 2D.
    """
    empty_patches = [patch["name"] for patch in mesh_patches if patch.get("type") == "empty"]
    thin_axis = None
    if block_counts and len(block_counts) == 3:
        ones = [axis for axis, count in enumerate(block_counts) if count == 1]
        thin_axis = ones[0] if len(ones) == 1 else None

    result: dict[str, Any] = {
        "empty_patches": empty_patches,
        "thin_axis": thin_axis,
        "block_counts": tuple(block_counts) if block_counts else None,
        "two_dimensional": bool(empty_patches) or thin_axis is not None,
        "wrong_types": [],
        "missing_entries": [],
        "thickness_disagrees": False,
    }
    if not result["two_dimensional"]:
        return result

    if empty_patches and block_counts and thin_axis is None:
        result["thickness_disagrees"] = True

    for field in sorted(fields):
        spec = fields[field]
        if not spec.get("present"):
            continue
        for patch in empty_patches:
            declared = entry_type_for(spec, patch)
            if declared is None:
                if spec.get("includes"):
                    continue
                result["missing_entries"].append((field, patch))
            elif declared != "empty":
                result["wrong_types"].append((field, patch, declared or "(no type entry)"))
    return result


def check_empty(case: Case, intent: Intent) -> list[Finding]:
    block_mesh = case.read("system/blockMeshDict")
    blocks = cells_estimate.HEX_BLOCK.findall(block_mesh)
    counts = tuple(int(value) for value in blocks[0]) if blocks else None

    # Before blockMesh runs there is no constant/polyMesh/boundary, and that is the
    # moment this check is worth most: an `empty` patch missing from blockMeshDict
    # costs a mesh build to discover, and the case that has not been meshed is
    # exactly the one still cheap to fix. So the built mesh is preferred and the
    # dictionary that will produce it is the fallback, with the source named in the
    # finding -- reading only the built mesh made every unmeshed 2D case report
    # "no patch is declared empty", which is the opposite of the truth.
    patches = case.boundary
    source = "constant/polyMesh/boundary"
    if not patches:
        patches = parse_block_mesh_boundary(block_mesh)
        source = "system/blockMeshDict"

    result = empty_diagnosis(patches, case.fields, counts)
    result["source"] = source

    if not result["two_dimensional"]:
        return [Finding(
            "empty", "skipped",
            "no patch of type empty and no single-cell direction in the blockMeshDict",
            "nothing here says this is a 2D study, so the empty rules do not apply",
        )]

    measured = []
    if result["empty_patches"]:
        measured.append(f"empty patches in {source}: {', '.join(result['empty_patches'])}")
    else:
        measured.append(f"no patch is declared empty in {source}")
    if result["block_counts"]:
        measured.append("blockMeshDict cells (%d %d %d)" % result["block_counts"])

    meanings: list[str] = []
    repairs: list[str] = []
    status = "ok"

    if not result["empty_patches"]:
        status = "fail"
        meanings.append(
            "the blockMeshDict is one cell thick in a direction but no patch is "
            "empty, so the solver will discretise across that single cell as though "
            "it were a real third dimension"
        )
        repairs.append(
            "declare the two end faces as one patch of type empty in the "
            "blockMeshDict boundary block and rebuild the mesh"
        )
    if result["empty_patches"] and source == "system/blockMeshDict":
        meanings.append(
            "read from the blockMeshDict, because the mesh has not been built yet; "
            "checking again after blockMesh confirms the mesh agrees with it"
        )
    if result["thickness_disagrees"]:
        status = "fail"
        meanings.append(
            "an empty patch exists but no direction has exactly one cell: an empty "
            "patch on a mesh more than one cell thick is rejected by checkMesh and "
            "by the solver"
        )
        repairs.append("set the cell count in the empty direction to 1 and rebuild")
    for field, patch, declared in result["wrong_types"]:
        status = "fail"
        measured.append(f"0/{field} gives {patch} type {declared}")
        repairs.append(f"change 0/{field}'s {patch} entry to 'type empty;'")
    if result["wrong_types"]:
        meanings.append(
            "a field that does not call an empty patch empty stops the solver with "
            "'incompatible patch and patchField types' -- one field out of ten is enough"
        )
    for field, patch in result["missing_entries"]:
        status = "fail"
        measured.append(f"0/{field} has no entry for {patch}")
        repairs.append(f"add 'type empty;' for {patch} to 0/{field}")

    checked = [name for name in sorted(case.fields) if case.fields[name].get("present")]
    if status == "ok":
        where = "the mesh" if source == "constant/polyMesh/boundary" else "the blockMeshDict"
        if checked:
            # Naming the fields, because "every field agrees" over none of them is
            # the same sentence and a completely different fact.
            meanings.append(
                f"one cell in the thin direction, empty declared in {where}, and all "
                f"{count_phrase(len(checked), 'field')} with a boundaryField agree "
                f"({', '.join(checked)})"
            )
        else:
            # No 0/ yet is a stage, not a fault, so this stays `ok` -- but it says
            # which half of the check ran. "Every field agrees" over no fields is a
            # true sentence and a false reassurance, and this check exists precisely
            # to catch the one field that does not agree.
            measured.append("no readable field files to check")
            meanings.append(
                f"one cell in the thin direction and empty declared in {where}; there "
                "are no 0/ fields on disk yet, so the field half of this check has not "
                "been done"
            )
            repairs.append(
                "run this again once 0/ is written -- a single field left on "
                "zeroGradient at the empty patch is the whole bug this looks for"
            )
    return [Finding("empty", status, "; ".join(measured), "; ".join(meanings), "; ".join(repairs))]


# -- Reynolds number ---------------------------------------------------------------


RE_TOLERANCE = 0.05
"""Within 5 per cent the difference is rounding in whatever length scale was used."""

RE_FACTOR = 2.0
"""Beyond a factor of two it is not a rounding difference, it is a different case."""


def reynolds_diagnosis(
    velocity: float | None,
    length: float | None,
    viscosity: float | None,
    stated: float | None = None,
) -> dict[str, Any]:
    """The Re implied by U, L and nu, and how far it sits from the stated intent.

    Worth doing because the three numbers live in three different files -- U in
    `0/U`, nu in `constant/transportProperties`, L nowhere at all -- and nobody
    recomputes the product after changing one of them. A case set up for Re 3900 and
    left at the nu of the Re 100 case it was copied from runs perfectly and answers a
    question nobody asked.
    """
    result: dict[str, Any] = {
        "velocity": velocity,
        "length": length,
        "viscosity": viscosity,
        "stated": stated,
        "implied": None,
        "ratio": None,
        "status": "skipped",
    }
    if viscosity is not None and viscosity <= 0:
        result["status"] = "fail"
        return result
    if not (velocity and length and viscosity):
        return result
    implied = float(velocity) * float(length) / float(viscosity)
    result["implied"] = implied
    result["status"] = "ok"
    if stated and stated > 0:
        ratio = implied / float(stated)
        result["ratio"] = ratio
        if abs(ratio - 1.0) > RE_TOLERANCE:
            result["status"] = "fail" if (ratio > RE_FACTOR or ratio < 1 / RE_FACTOR) else "warn"
    return result


def suggested_viscosity(velocity: float, length: float, reynolds: float) -> float:
    """The nu that would make the stated Re true, which is what the repair line needs."""
    return float(velocity) * float(length) / float(reynolds)


def check_reynolds(case: Case, intent: Intent) -> list[Finding]:
    velocity = intent.velocity or parse_uniform_velocity(case.field_texts.get("U", ""))
    viscosity = case.viscosity
    result = reynolds_diagnosis(velocity, intent.length, viscosity, intent.reynolds)

    source = "--u" if intent.velocity else "the largest uniform vector in 0/U"
    if viscosity is None:
        return [Finding(
            "reynolds", "skipped",
            "no nu found in constant/transportProperties, physicalProperties or "
            "momentumTransport",
            "either this is a compressible case, where the viscosity is a function of "
            "state, or the transport properties have not been written yet",
        )]
    if result["status"] == "fail" and result["implied"] is None:
        return [Finding(
            "reynolds", "fail",
            f"nu = {viscosity}",
            "a zero or negative kinematic viscosity is not a fluid; the momentum "
            "equation loses its diffusive term",
            "set nu to the real value for the fluid, or to U*L/Re for the Re you want",
        )]
    if result["implied"] is None:
        detail = f"nu = {viscosity:.6g}"
        if velocity:
            detail += f", U = {velocity:.6g} m/s (from {source}), Re/L = {velocity / viscosity:.6g} per metre"
        return [Finding(
            "reynolds", "skipped",
            detail,
            "without a length scale there is no Reynolds number to check -- no length "
            "is guessed here, because a made-up L makes a made-up Re",
            "pass --l with the characteristic length (chord, diameter, height)",
        )]

    measured = (
        f"U = {velocity:.6g} m/s (from {source}), L = {intent.length:.6g} m, "
        f"nu = {viscosity:.6g} m2/s -> Re = {result['implied']:.6g}"
    )
    if not intent.reynolds:
        return [Finding(
            "reynolds", "ok", measured,
            "this is the Reynolds number the case is actually set up for; nothing was "
            "stated to compare it against",
            "pass --re with the intended Reynolds number to have the two compared",
        )]

    measured += f", stated Re = {intent.reynolds:.6g} (ratio {result['ratio']:.4g})"
    if result["status"] == "ok":
        return [Finding(
            "reynolds", "ok", measured,
            "the case is set up for the Reynolds number it was asked for",
        )]
    wanted_nu = suggested_viscosity(velocity, intent.length, intent.reynolds)
    wanted_u = float(intent.reynolds) * viscosity / float(intent.length)
    return [Finding(
        "reynolds", result["status"], measured,
        f"the case will run at Re {result['implied']:.6g}, not the {intent.reynolds:.6g} "
        "it is supposed to be; at this separation the flow regime itself can differ",
        f"set nu to {wanted_nu:.6g} m2/s in constant/transportProperties, or set the "
        f"inlet U to {wanted_u:.6g} m/s, or correct --l if the length scale is wrong",
    )]


# -- predicted against actual cell count -------------------------------------------


CELL_RATIO_WARN = 3.0
"""The estimate is documented as order-of-magnitude and lands within about a factor
of two on ordinary geometries, so three is where a disagreement starts meaning
something."""

CELL_RATIO_FAIL = 20.0

BIG_MESH = 20_000_000
"""Cells. Past this a build is an overnight job on one machine, which is worth saying
out loud before it starts rather than after."""


def cell_count_diagnosis(predicted: float | None, actual: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "predicted": predicted, "actual": actual, "ratio": None, "status": "skipped",
    }
    if predicted is None or predicted <= 0:
        return result
    if actual is None:
        result["status"] = "warn" if predicted > BIG_MESH else "ok"
        return result
    ratio = actual / predicted
    result["ratio"] = ratio
    if ratio > CELL_RATIO_FAIL or ratio < 1 / CELL_RATIO_FAIL:
        result["status"] = "fail"
    elif ratio > CELL_RATIO_WARN or ratio < 1 / CELL_RATIO_WARN:
        result["status"] = "warn"
    else:
        result["status"] = "ok"
    return result


def block_mesh_cells(text: str) -> int | None:
    """The exact cell count a `blockMeshDict` will produce, or None.

    blockMesh makes the product of the divisions written on each block and nothing
    else, so this is not an estimate: it is the number `checkMesh` will report.
    """
    blocks = cells_estimate.HEX_BLOCK.findall(text or "")
    if not blocks:
        return None
    total = 0
    for counts in blocks:
        try:
            x, y, z = (int(value) for value in counts)
        except (TypeError, ValueError):
            return None
        total += x * y * z
    return total


def check_cells(case: Case, intent: Intent) -> list[Finding]:
    block_text = case.read("system/blockMeshDict")
    snappy_text = case.read("system/snappyHexMeshDict")
    actual = case.cell_count

    # No snappyHexMeshDict means blockMesh builds the mesh on its own, and then the
    # count is arithmetic rather than a guess. The estimator below is built for the
    # other case -- a uniform background mesh refined around an STL -- and applying
    # it here read a 12-block graded O-grid as 105 background cells against 4,238
    # real ones, then advised checking `refinementRegions` a case with no snappy
    # dictionary does not have. A gatekeeper that fails a healthy case teaches the
    # agent to stop reading it.
    if not snappy_text.strip():
        exact = block_mesh_cells(block_text)
        if exact is None:
            return [Finding(
                "cells", "skipped", "no snappyHexMeshDict and no readable blocks in blockMeshDict",
                "there is nothing here to predict a cell count from",
            )]
        if actual is None:
            return [Finding(
                "cells", "ok", f"blockMesh will build exactly {exact:,} cells; no mesh built yet",
                f"this is the product of the block divisions, so it is what checkMesh will report"
                + (", which is an overnight job on one machine" if exact > BIG_MESH else ""),
                "coarsen the block divisions if that is more than the study needs"
                if exact > BIG_MESH else "",
            )]
        if actual == exact:
            return [Finding(
                "cells", "ok", f"blockMeshDict predicts {exact:,} cells and the mesh has {actual:,}",
                "the mesh is exactly what the dictionary asks for",
            )]
        return [Finding(
            "cells", "warn",
            f"blockMeshDict predicts {exact:,} cells and the mesh has {actual:,}",
            "blockMesh produces the product of the block divisions, so these should "
            "agree exactly; a difference means the mesh on disk was not built from "
            "this dictionary, or blockMesh merged faces (mergePatchPairs)",
            "rebuild with blockMesh and compare again",
        )]

    delta0, volume, _extent = cells_estimate.block_mesh_delta(block_text)

    if delta0 is None:
        if actual:
            return [Finding(
                "cells", "ok", f"{actual:,} cells in the built mesh",
                "no readable blockMeshDict, so there is no prediction to compare against",
            )]
        return [Finding(
            "cells", "skipped", "no readable blockMeshDict and no built mesh",
            "nothing to predict from and nothing to compare with",
        )]

    area = sum(cells_estimate.stl_area(path)[0] for path in case.surfaces)
    levels = cells_estimate.surface_levels(snappy_text)
    layers_match = cells_estimate.N_LAYERS.search(snappy_text)
    layers = int(layers_match.group(1)) if layers_match else 0
    prediction = cells_estimate.estimate(delta0, volume, levels, area, layers)
    predicted = prediction["total"]

    result = cell_count_diagnosis(predicted, actual)
    measured = (
        f"predicted ~{predicted:,.0f} cells (background {prediction['background']:,.0f}, "
        f"surface {prediction['surface']:,.0f}, layers {prediction['layers']:,.0f}, "
        f"base cell {delta0:.4g} m, finest level {prediction['finest_level']})"
    )
    if actual is None:
        measured += "; no mesh built yet"
        meaning = (
            f"this build is predicted at {predicted:,.0f} cells"
            + (", which is an overnight job on one machine" if predicted > BIG_MESH else "")
        )
        repair = (
            "drop a refinement level or coarsen the background mesh if that is more "
            "than the study needs" if predicted > BIG_MESH else ""
        )
        return [Finding("cells", result["status"], measured, meaning, repair)]

    measured += f"; the mesh has {actual:,} cells (ratio {result['ratio']:.3g})"
    if result["status"] == "ok":
        return [Finding(
            "cells", "ok", measured,
            "the mesh that was built is the size the dictionaries predict, so the "
            "refinement that was asked for is the refinement that happened",
        )]
    if result["ratio"] < 1:
        meaning = (
            "the mesh is far smaller than the dictionaries predict, which is what a "
            "snappy run that failed to find the surface looks like -- the refinement "
            "never happened and the background mesh is all that is left"
        )
        repair = (
            "check the snappy log for 'Did not find' or a zero-cell refinement, and "
            "check locationInMesh is inside the fluid and the surface is closed"
        )
    else:
        meaning = (
            "the mesh is far larger than the dictionaries predict; the refinement "
            "regions are catching more of the domain than intended"
        )
        repair = "check refinementRegions and the refinement levels against the geometry"
    return [Finding("cells", result["status"], measured, meaning, repair)]


# -- checkMesh quality -------------------------------------------------------------


NON_ORTHO_WARN = 70.0
"""OpenFOAM's own threshold for 'severe non-orthogonality'; above it the correction
loops in the pressure equation stop being optional."""

NON_ORTHO_FAIL = 85.0
SKEWNESS_WARN = 4.0
SKEWNESS_FAIL = 10.0
ASPECT_WARN = 1000.0


def checkmesh_verdict(data: dict[str, Any]) -> dict[str, Any]:
    """Thresholds against the numbers `mesh_digest` extracts.

    `mesh_digest` deliberately attaches no verdicts, which is right for a digest and
    useless for a gate, so the thresholds live here. They are the conventional ones
    and they are not the last word: a 78-degree mesh runs fine with enough
    nonOrthogonalCorrectors and the right limited scheme.
    """
    verdict: dict[str, Any] = {"status": "ok", "problems": [], "repairs": [], "numbers": {}}

    def number(value):
        return as_float(value[0] if isinstance(value, tuple) else value)

    non_ortho = number(data.get("non_ortho"))
    if non_ortho is not None:
        verdict["numbers"]["non_orthogonality_max"] = non_ortho
        if non_ortho > NON_ORTHO_FAIL:
            verdict["problems"].append(
                f"maximum non-orthogonality {non_ortho:g} degrees, past the point where "
                "the pressure correction stops converging reliably"
            )
            verdict["repairs"].append(
                "improve the mesh, or set nonOrthogonalCorrectors to 2 or 3 in "
                "system/fvSolution and use 'laplacian ... limited 0.33' in fvSchemes"
            )
        elif non_ortho > NON_ORTHO_WARN:
            verdict["problems"].append(
                f"maximum non-orthogonality {non_ortho:g} degrees, over OpenFOAM's own "
                f"severe threshold of {NON_ORTHO_WARN:g}"
            )
            verdict["repairs"].append(
                "add nonOrthogonalCorrectors 1-2 in system/fvSolution and use a limited "
                "laplacian scheme"
            )

    skewness = number(data.get("skewness"))
    if skewness is not None:
        verdict["numbers"]["skewness_max"] = skewness
        if skewness > SKEWNESS_FAIL:
            verdict["problems"].append(f"maximum skewness {skewness:g}")
            verdict["repairs"].append(
                "the skewed cells are usually where layers met a sharp feature; reduce "
                "the layer count there or relax the snap controls"
            )
        elif skewness > SKEWNESS_WARN:
            verdict["problems"].append(f"maximum skewness {skewness:g}, above the usual limit of 4")
            verdict["repairs"].append("check where the skewed cells sit before trusting results there")

    aspect = number(data.get("aspect_ratio"))
    if aspect is not None:
        verdict["numbers"]["aspect_ratio_max"] = aspect
        if aspect > ASPECT_WARN:
            verdict["problems"].append(f"maximum aspect ratio {aspect:g}")
            verdict["repairs"].append(
                "very stretched cells slow the pressure solve; check the layer expansion"
            )

    failures = data.get("failures") or []
    if failures:
        verdict["problems"].extend(f"checkMesh flagged: {line}" for line in failures)
        verdict["repairs"].append("read the flagged lines in the checkMesh log in full")

    if failures or (non_ortho is not None and non_ortho > NON_ORTHO_FAIL) or (
        skewness is not None and skewness > SKEWNESS_FAIL
    ):
        verdict["status"] = "fail"
    elif verdict["problems"]:
        verdict["status"] = "warn"
    return verdict


def check_checkmesh(case: Case, intent: Intent) -> list[Finding]:
    text = case.check_mesh_log_text
    if not text:
        return [Finding(
            "checkmesh", "skipped", "no log.checkMesh in the case directory",
            "checkMesh has not been run, or its output was not kept",
            "run checkMesh and keep the output: checkMesh > log.checkMesh 2>&1",
        )]
    data = mesh_digest.parse(text)
    verdict = checkmesh_verdict(data)
    numbers = ", ".join(f"{key} {value:g}" for key, value in verdict["numbers"].items())
    cells = data.get("counts", {}).get("cells")
    measured = f"{cells:,} cells; {numbers}" if cells else numbers or "no metrics parsed"

    # A log with no metrics in it is a checkMesh that did not get as far as
    # measuring anything -- almost always because the mesh was not there to read.
    # Reporting `ok` for that says "the quality numbers are fine" about numbers
    # that were never read, which is the one answer a gate must not give.
    if not verdict["numbers"] and not verdict["problems"] and cells is None:
        fatal = fatal_error_text(text)
        return [Finding(
            "checkmesh", "skipped",
            f"{len(text.splitlines())} line(s) of log.checkMesh, no quality metrics in it"
            + (f". Fatal error: {fatal}" if fatal else ""),
            "checkMesh did not get as far as measuring the mesh, so there is nothing "
            "here to judge -- this is not a mesh that passed"
            + (", and the log ends in a fatal error" if fatal else ""),
            "run checkMesh again once the mesh exists and keep the output: "
            "checkMesh > log.checkMesh 2>&1",
        )]
    if verdict["status"] == "ok":
        return [Finding(
            "checkmesh", "ok", measured,
            "every quality metric this reads is inside the conventional limits",
        )]
    return [Finding(
        "checkmesh", verdict["status"], measured,
        "; ".join(verdict["problems"]),
        "; ".join(verdict["repairs"]),
    )]


# -- the one-iteration solver probe ------------------------------------------------


PROBE_SKIP = ("processor", "postProcessing", "dynamicCode", "log", "system")
"""Top-level entries the staged copy does not take from the real case. `system` is
excluded here because it is copied rather than linked -- it is the one directory the
probe rewrites, and rewriting a link would rewrite the real controlDict."""

PROBE_TIMEOUT = 300

SOLVER_HINTS: tuple[tuple[str, str], ...] = (
    (r"Cannot find patchField entry for (\S+)",
     "the field named in the error has no boundaryField entry for that patch -- add one"),
    (r"incompatible|inconsistent.*patch.*patchField",
     "a field gives a patch a type the mesh does not allow there, usually a "
     "constraint patch (empty, cyclic, wedge) given an ordinary condition"),
    (r"keyword (\S+) is undefined in dictionary \"?([^\"\s]+)",
     "the dictionary named is missing the keyword named -- add it"),
    (r"cannot find file \"?([^\"\s]+)",
     "the file named does not exist; check the path and whether an earlier step wrote it"),
    (r"Unknown (?:patchField|function|fvPatchField|solver) type (\S+)",
     "the type named is not in this OpenFOAM build -- check the spelling and the version"),
    (r"number of (?:cells|faces|points).*not equal|size \d+ is not equal",
     "a field's internal list is a different length from the mesh, which is what a "
     "0/ directory left over from a different mesh looks like"),
    (r"Maximum number of iterations exceeded",
     "a bounded quantity ran away on the first step -- check the initial conditions"),
    (r"floating point exception|Foam::sigFpe",
     "a division by zero on the first step, usually a zero-thickness cell or a zero "
     "reference value"),
)
"""Fatal-error shapes worth naming. Anything not matched is reported with the error
text itself and no interpretation, which is the honest answer for an unknown one."""


def rewrite_control_dict(text: str, entries: dict[str, str]) -> str:
    """A `controlDict` with some top-level entries replaced, others appended.

    Only top-level ones: the `functions` block has its own `writeInterval` and
    rewriting that instead would leave the run bounds untouched and quietly change a
    function object.
    """
    lines = strip_comments(text).splitlines()
    remaining = dict(entries)
    depth = 0
    output: list[str] = []
    for line in lines:
        replaced = False
        if depth == 0:
            match = _ENTRY_LINE.match(line)
            if match and match.group(1) in remaining:
                key = match.group(1)
                output.append(f"{key}    {remaining.pop(key)};")
                replaced = True
        depth = max(0, depth + line.count("{") - line.count("}"))
        if not replaced:
            output.append(line)
    for key, value in remaining.items():
        output.append(f"{key}    {value};")
    return "\n".join(output) + "\n"


def probe_control_dict(text: str) -> tuple[str, dict[str, float]]:
    """The probe's `controlDict`: stop one step past where the run starts.

    `startFrom startTime` and an explicit `startTime` rather than `latestTime`,
    because the probe is asking whether the case can take its first step, and a
    `latestTime` that finds a half-written time directory answers a different
    question. `writeControl timeStep; writeInterval 1;` so the write path is
    exercised too -- a case that solves and then dies writing is still a dead case.
    """
    control = parse_control(text)
    start = as_float(control.get("startTime")) or 0.0
    delta = as_float(control.get("deltaT")) or 1.0
    end = start + delta
    entries = {
        "startFrom": "startTime",
        "startTime": repr(start),
        "stopAt": "endTime",
        "endTime": repr(end),
        "writeControl": "timeStep",
        "writeInterval": "1",
        "purgeWrite": "0",
        "adjustTimeStep": "no",
        "runTimeModifiable": "false",
    }
    return rewrite_control_dict(text, entries), {"startTime": start, "deltaT": delta, "endTime": end}


def stage_probe_case(case: Path, destination: Path) -> Path:
    """A case that shares the real one's mesh and fields but owns its own `system`.

    `constant` and the time directories are linked rather than copied: a polyMesh is
    routinely gigabytes and copying one to ask a question that takes two seconds is
    the wrong trade. `system` is copied, because that is the directory the probe
    rewrites, and the entire point is that the real `controlDict` is never touched.
    On a filesystem that refuses links the copy is made instead.

    The case path is resolved first, and it has to be. A symlink stores its target
    verbatim and a relative one resolves against the *link's* directory, not the
    working directory -- so `preflight.py case`, the ordinary way to call this,
    linked `constant` to `<tempdir>/case/case/constant` and staged a case with
    nothing in it. The solver then said it could not find the mesh and the probe
    reported `fail` on a case that was fine.
    """
    case = Path(case).resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(case.iterdir()):
        if entry.name in PROBE_SKIP or entry.name.startswith("processor") or entry.name.endswith("_frames"):
            continue
        if entry.name.startswith("log"):
            continue
        target = destination / entry.name
        try:
            os.symlink(entry, target, target_is_directory=entry.is_dir())
            continue
        except (OSError, NotImplementedError, AttributeError):
            pass
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
    system = case / "system"
    if system.is_dir():
        shutil.copytree(system, destination / "system", dirs_exist_ok=True)
    return destination


def subprocess_runner(command: list[str], cwd: Path) -> dict[str, Any]:
    """The default probe runner. Injected in tests, so no test ever needs a solver."""
    try:
        completed = subprocess.run(
            command, cwd=str(cwd), capture_output=True, text=True,
            timeout=PROBE_TIMEOUT, check=False,
        )
    except FileNotFoundError:
        return {"returncode": 127, "output": f"{command[0]}: not found on PATH"}
    except subprocess.TimeoutExpired:
        return {
            "returncode": 124,
            "output": f"{command[0]} did not finish one step in {PROBE_TIMEOUT} s",
        }
    return {"returncode": completed.returncode, "output": (completed.stdout or "") + (completed.stderr or "")}


def fatal_error_text(output: str) -> str:
    """The fatal-error block from a solver's output, without the banner around it."""
    match = re.search(r"--> FOAM FATAL (?:IO )?ERROR:?(.*?)(?:FOAM exiting|$)", output or "", re.S)
    if not match:
        return ""
    body = match.group(1)
    body = re.sub(r"^\s*#\d+.*$", "", body, flags=re.M)  # the stack trace says nothing useful
    body = re.sub(r"^\s*-+\s*$", "", body, flags=re.M)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return " ".join(lines[:6])


def probe_verdict(result: dict[str, Any]) -> dict[str, Any]:
    """What the one step said, turned into a diagnosis.

    A solver that reaches its first `Time =` and exits cleanly has proved the whole
    read path: dictionaries parse, patches line up, fields match the mesh, the
    schemes exist. That is most of what preflight is guessing at everywhere else,
    confirmed by the only authority that counts.
    """
    output = result.get("output") or ""
    returncode = int(result.get("returncode", 0))
    fatal = fatal_error_text(output)
    times = re.findall(r"^Time = (\S+)", output, re.M)
    finished = bool(re.search(r"^End\b", output, re.M))

    verdict: dict[str, Any] = {
        "returncode": returncode,
        "steps": len(times),
        "finished": finished,
        "fatal": fatal,
        "hint": "",
        "status": "ok",
    }
    if fatal or returncode not in (0,):
        verdict["status"] = "fail"
        for pattern, hint in SOLVER_HINTS:
            if re.search(pattern, fatal or output, re.I):
                verdict["hint"] = hint
                break
    elif not times and not finished:
        verdict["status"] = "warn"
    return verdict


def check_probe(
    case: Case,
    intent: Intent,
    runner: Callable[[list[str], Path], dict[str, Any]] | None = None,
    workdir: Path | None = None,
) -> list[Finding]:
    application = case.application
    if not application:
        return [Finding(
            "probe", "skipped", "system/controlDict names no application",
            "there is no solver to run for a step",
            "set 'application <solver>;' in system/controlDict",
        )]
    if not case.boundary:
        return [Finding(
            "probe", "skipped", "no constant/polyMesh/boundary",
            "there is no mesh for the solver to read, so a probe would only report that",
        )]
    if runner is None:
        if shutil.which(application) is None:
            return [Finding(
                "probe", "skipped", f"{application} is not on PATH",
                "the solver binary is not available here, so it cannot be asked to take a step",
                "run this inside the OpenFOAM environment, or source its bashrc first",
            )]
        runner = subprocess_runner

    temporary = None
    try:
        if workdir is None:
            temporary = tempfile.TemporaryDirectory(prefix="preflight-probe-")
            workdir = Path(temporary.name) / case.path.name
        staged = stage_probe_case(case.path, Path(workdir))
        control_path = staged / "system" / "controlDict"
        patched, bounds = probe_control_dict(read_text(control_path))
        control_path.write_text(patched, encoding="utf-8")
        result = runner([application], staged)
    except OSError as error:
        return [Finding(
            "probe", "skipped", f"the case could not be staged for a probe ({error})",
            "the probe never runs against the real case, so a staging failure stops it here",
        )]
    finally:
        if temporary is not None:
            try:
                temporary.cleanup()
            except OSError:
                pass

    verdict = probe_verdict(result)
    measured = (
        f"{application} run in a copy from t={bounds['startTime']:g} to "
        f"{bounds['endTime']:g} (one deltaT): exit {verdict['returncode']}, "
        f"{verdict['steps']} time step(s)"
        + (", reached End" if verdict["finished"] else "")
    )
    if verdict["fatal"]:
        measured += f". Fatal error: {verdict['fatal']}"
    if verdict["status"] == "ok":
        return [Finding(
            "probe", "ok", measured,
            "the solver read the case and took a step, so the dictionaries parse, the "
            "patches line up and the fields match the mesh",
        )]
    if verdict["status"] == "warn":
        return [Finding(
            "probe", "warn", measured,
            "the solver exited without an error and without reaching a time step, "
            "which usually means it stopped in setup",
            "run the solver by hand in the case to see the full output",
        )]
    return [Finding(
        "probe", "fail", measured,
        verdict["hint"] or "the solver stopped before completing one step; the fatal "
        "error text above is what it said",
        "fix what the error names, then re-run this probe -- it costs one time step",
    )]


# -- Courant number and timestep ---------------------------------------------------


COURANT_WARN = 1.0
COURANT_FAIL = 5.0


def courant_estimate(velocity: float | None, cell_size: float | None, delta_t: float | None):
    """U * deltaT / dx. The convective Courant number for the smallest cell."""
    if not velocity or not cell_size or not delta_t:
        return None
    if cell_size <= 0:
        return None
    return float(velocity) * float(delta_t) / float(cell_size)


def is_steady(application: str) -> bool:
    return bool(application) and (application in STEADY_APPLICATIONS or "Simple" in application)


def courant_verdict(courant: float | None, adjusting: bool, max_co: float | None) -> dict[str, Any]:
    result: dict[str, Any] = {"courant": courant, "status": "skipped"}
    if adjusting:
        result["status"] = "ok" if (max_co and max_co <= 5) else "warn"
        return result
    if courant is None:
        return result
    if courant > COURANT_FAIL:
        result["status"] = "fail"
    elif courant > COURANT_WARN:
        result["status"] = "warn"
    else:
        result["status"] = "ok"
    return result


def smallest_cell_size(case: Case) -> tuple[float | None, str]:
    """The smallest cell edge available, and where it came from.

    checkMesh's minimum cell volume first, because that is the mesh that exists; the
    blockMeshDict base size only describes the mesh before snappy refined it, and
    using it on a refined case reports a Courant number several factors too low --
    the exact direction that makes an unstable run look safe.
    """
    data = mesh_digest.parse(case.check_mesh_log_text) if case.check_mesh_log_text else {}
    volumes = data.get("cell_volume")
    if volumes:
        smallest = as_float(volumes[0])
        if smallest and smallest > 0:
            return float(smallest) ** (1.0 / 3.0), "the minimum cell volume in log.checkMesh"
    delta0, _volume, _extent = cells_estimate.block_mesh_delta(case.read("system/blockMeshDict"))
    if delta0:
        levels = cells_estimate.surface_levels(case.read("system/snappyHexMeshDict"))
        finest = max(levels.values()) if levels else 0
        if finest:
            return delta0 / (2 ** finest), (
                f"the blockMeshDict base cell divided by the finest snappy level ({finest})"
            )
        return delta0, "the blockMeshDict base cell size"
    return None, ""


def check_courant(case: Case, intent: Intent) -> list[Finding]:
    application = case.application
    control = case.control
    delta_t = as_float(control.get("deltaT"))
    adjusting = str(control.get("adjustTimeStep", "no")).lower() in ("yes", "true", "on", "1")
    max_co = as_float(control.get("maxCo"))

    if is_steady(application):
        return [Finding(
            "courant", "skipped",
            f"application {application}, deltaT {delta_t}",
            "this solver is steady, so deltaT counts iterations and there is no "
            "physical Courant number to check",
        )]
    velocity = intent.velocity or parse_uniform_velocity(case.field_texts.get("U", ""))
    cell_size, source = smallest_cell_size(case)
    courant = courant_estimate(velocity, cell_size, delta_t)
    verdict = courant_verdict(courant, adjusting, max_co)

    if adjusting:
        measured = f"adjustTimeStep is on with maxCo {max_co if max_co is not None else 'unset'}"
        if verdict["status"] == "ok":
            return [Finding(
                "courant", "ok", measured,
                "the solver picks deltaT itself and holds the Courant number at maxCo, "
                "so the value in controlDict is only a starting point",
            )]
        return [Finding(
            "courant", "warn", measured,
            "with adjustTimeStep on, maxCo is the number that matters, and it is unset "
            "or high enough to lose accuracy in the transient",
            "set maxCo to about 1 (0.5 for LES or an interface) in system/controlDict",
        )]

    if courant is None:
        missing = [
            label for label, value in
            (("U", velocity), ("a cell size", cell_size), ("deltaT", delta_t))
            if not value
        ]
        return [Finding(
            "courant", "skipped",
            f"could not compute a Courant number: no {', no '.join(missing)}",
            "one of the three numbers it needs is not readable from the case",
            "pass --u, or run checkMesh so the minimum cell volume is on disk",
        )]

    measured = (
        f"U {velocity:.4g} m/s, smallest cell ~{cell_size:.4g} m (from {source}), "
        f"deltaT {delta_t:g} -> Co ~ {courant:.4g}"
    )
    if verdict["status"] == "ok":
        return [Finding(
            "courant", "ok", measured,
            "the fluid crosses less than a cell per time step in the smallest cell",
        )]
    suggested = float(delta_t) * (0.8 / courant)
    return [Finding(
        "courant", verdict["status"], measured,
        "the fluid crosses several cells per time step in the smallest cell; an "
        "explicit or PISO scheme goes unstable, and even PIMPLE loses the transient",
        f"drop deltaT to about {suggested:.3g} s, or set adjustTimeStep yes with "
        "maxCo 1 and let the solver pick",
    )]


# -- residuals in a running or finished log ----------------------------------------


DIVERGED = 1e3
"""An initial residual above this is not a slow convergence, it is a blow-up."""

RESIDUAL_RISE = 10.0
"""How much worse than its own best a residual has to get to count as rising."""

RESIDUAL_FLOOR = 1e-6
"""Below this a residual is converged and the ratio to its own minimum stops meaning
anything. A run that bottoms out at 1e-12 and settles at 3e-11 is thirty times its
best and is not going anywhere; calling that "rising" puts a warn on the healthiest
log there is, and a gate that warns about healthy runs stops being read."""

CONTINUITY_WARN = 1e-3
CONTINUITY_FAIL = 1.0


def residual_verdict(data: dict[str, Any]) -> dict[str, Any]:
    """Divergence and continuity drift out of what `log_digest` read.

    Two separate failures with two separate repairs. A residual that climbs is a
    stability problem -- relaxation, schemes, timestep. A cumulative continuity error
    that grows while residuals fall is a mass-conservation problem -- boundary
    conditions that do not balance, or a pressure solve stopping too early.
    """
    verdict: dict[str, Any] = {
        "status": "ok", "diverging": [], "rising": [], "problems": [], "repairs": [],
    }
    residuals = data.get("residuals") or {}
    if not residuals:
        verdict["status"] = "skipped"
        return verdict

    for field, series in sorted(residuals.items()):
        values = [value for _step, value in series]
        if not values:
            continue
        last = values[-1]
        if not math.isfinite(last) or last > DIVERGED:
            verdict["diverging"].append((field, last))
        elif (
            len(values) > 3
            and last > RESIDUAL_FLOOR
            and last > min(values) * RESIDUAL_RISE
        ):
            verdict["rising"].append((field, min(values), last))

    if verdict["diverging"]:
        verdict["problems"].append(
            "diverged: " + ", ".join(f"{field} at {value:.3e}" for field, value in verdict["diverging"])
        )
        verdict["repairs"].append(
            "lower the relaxation factors in system/fvSolution, drop deltaT, or move to "
            "a bounded/limited divergence scheme in system/fvSchemes"
        )
    if verdict["rising"]:
        verdict["problems"].append(
            "rising: " + ", ".join(
                f"{field} best {best:.2e}, now {last:.2e}" for field, best, last in verdict["rising"]
            )
        )
        verdict["repairs"].append(
            "a residual climbing back off its own floor usually starts at a boundary; "
            "look at where the field is worst before changing the numerics"
        )

    continuity = data.get("continuity")
    if continuity:
        cumulative = abs(float(continuity[2]))
        verdict["cumulative_continuity"] = cumulative
        if not math.isfinite(cumulative) or cumulative > CONTINUITY_FAIL:
            verdict["problems"].append(f"cumulative continuity error {cumulative:.3e}")
            verdict["repairs"].append(
                "mass is not conserved: check the inlet and outlet conditions balance "
                "and that the pressure equation is solving to a tight enough tolerance"
            )
        elif cumulative > CONTINUITY_WARN:
            verdict["problems"].append(f"cumulative continuity error {cumulative:.3e}, drifting")
            verdict["repairs"].append(
                "tighten the p solver tolerance or add a pressure corrector"
            )

    bounding = data.get("bounding") or {}
    if bounding:
        verdict["problems"].append(
            "bounded " + ", ".join(f"{field} x{count}" for field, count in sorted(bounding.items()))
        )
        verdict["repairs"].append(
            "a field being bounded is the solver clipping a value it computed as "
            "negative; frequent bounding of k or omega means the turbulence inlet "
            "values or the wall functions want checking"
        )

    if verdict["diverging"] or (
        verdict.get("cumulative_continuity", 0) > CONTINUITY_FAIL
    ):
        verdict["status"] = "fail"
    elif verdict["problems"]:
        verdict["status"] = "warn"
    return verdict


def check_residuals(case: Case, intent: Intent, log: str | Path | None = None) -> list[Finding]:
    path = case.solver_log(log)
    if path is None:
        return [Finding(
            "residuals", "skipped", "no solver log found in the case directory",
            "nothing has run yet, or its output was not kept",
            "keep the solver output: <solver> > log.<solver> 2>&1",
        )]
    try:
        data = log_digest.digest(path)
    except OSError as error:
        return [Finding("residuals", "skipped", f"{path.name} could not be read ({error})", "")]

    verdict = residual_verdict(data)
    times = data.get("times") or []
    header = f"{path.name}: {len(times)} time step(s)"
    if times:
        header += f", last Time = {times[-1]:g}"
    finals = data.get("final_residual") or {}
    if finals:
        header += "; last residuals " + ", ".join(
            f"{field} {value:.2e}" for field, value in sorted(finals.items())
        )
    if verdict["status"] == "skipped":
        return [Finding(
            "residuals", "skipped", header,
            "the log has no 'Solving for' lines yet, so there are no residuals to read",
        )]
    if verdict["status"] == "ok":
        return [Finding(
            "residuals", "ok", header,
            "residuals are falling or flat, continuity is not drifting, and nothing is "
            "being bounded",
        )]
    return [Finding(
        "residuals", verdict["status"],
        header + ". " + "; ".join(verdict["problems"]),
        "the run is not converging in the way the log would show if it were healthy",
        "; ".join(verdict["repairs"]),
    )]


# -- force and pressure units ------------------------------------------------------


KINEMATIC_PRESSURE = (0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0)
"""m2/s2 -- pressure divided by density, which is what every incompressible OpenFOAM
solver actually solves for."""

STATIC_PRESSURE = (1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0)

FORCE_TYPES = ("forces", "forceCoeffs")


def find_force_objects(text: str, name_if_bare: str = "") -> list[dict[str, Any]]:
    """Function objects of type `forces` or `forceCoeffs`, with their settings.

    Three layouts, because all three are ordinary. Written out inside
    `controlDict`'s `functions` block; nested one level inside a wrapper block in an
    included file; and -- the one this used to miss -- a whole file that *is* the
    function object, `system/forceCoeffs` with `type forceCoeffs;` at its top level,
    which is what `#includeFunc forceCoeffs` reads. A case in that layout got
    "no forces or forceCoeffs function object" and its missing `rhoInf` was never
    looked at, which is the exact error this check exists for.
    """
    found: list[dict[str, Any]] = []
    functions = block_body(text, "functions")
    body = functions or strip_comments(text)
    for name, inner in block_entries(body):
        values = entry_values(inner)
        kind = values.get("type", "")
        if kind not in FORCE_TYPES:
            # A function object can be one level down, inside an #includeFunc-style
            # wrapper block; one level of nesting is worth looking through.
            for sub_name, sub_inner in block_entries(inner):
                sub_values = entry_values(sub_inner)
                if sub_values.get("type") in FORCE_TYPES:
                    found.append(_force_entry(sub_name, sub_values))
            continue
        found.append(_force_entry(name, values))
    if not found and not functions:
        # The file is the function object. Only ever considered when the text has no
        # `functions` block of its own, so a controlDict is never read this way.
        top = entry_values(body)
        if top.get("type") in FORCE_TYPES:
            found.append(_force_entry(name_if_bare or top.get("type", ""), top))
    return found


def _force_entry(name: str, values: dict[str, str]) -> dict[str, Any]:
    return {
        "name": name.strip('"'),
        "type": values.get("type", ""),
        "rho": values.get("rho", ""),
        "rhoInf": as_float(values.get("rhoInf")),
        "magUInf": as_float(values.get("magUInf")),
        "lRef": as_float(values.get("lRef")),
        "Aref": as_float(values.get("Aref")),
    }


def force_units_diagnosis(
    pressure_dimensions: tuple[float, ...] | None,
    objects: list[dict[str, Any]],
    has_fields: bool = True,
) -> dict[str, Any]:
    """Whether force output will come out in newtons or in newtons per unit density.

    Incompressible OpenFOAM solves for p in m2/s2, not Pa. The `forces` function
    object knows this and asks for the density separately: `rho rhoInf; rhoInf
    1.225;`. Left out, or left at the placeholder 1, the numbers it writes are
    forces divided by density -- correct to a factor of rho, wrong by 1.2 in air and
    by a thousand in water, and they look entirely plausible either way. This is the
    error that survives all the way into a report.
    """
    kinematic = pressure_dimensions == KINEMATIC_PRESSURE
    result: dict[str, Any] = {
        "kinematic_pressure": kinematic,
        "pressure_dimensions": pressure_dimensions,
        "problems": [],
        "repairs": [],
        "status": "ok",
    }
    if not objects:
        result["status"] = "skipped"
        return result
    if pressure_dimensions is None:
        # No fields written yet is a stage, not a fault: a controlDict can name its
        # force objects before the 0/ directory exists, and warning about a units
        # error nobody can have made yet is how a gate gets ignored. A 0/ that
        # exists and still has no readable p is a different thing and stays a warn.
        result["status"] = "warn" if has_fields else "skipped"
        result["problems"].append(
            "0/p has no readable dimensions entry" if has_fields
            else "no field files written yet, so p's dimensions cannot be read"
        )
        result["repairs"].append(
            "check whether p is kinematic (m2/s2) before reading any force output"
            if has_fields else
            "run this again once 0/ is written -- the density setting can only be "
            "judged against p's dimensions"
        )
        return result

    for entry in objects:
        label = f"{entry['type']} '{entry['name']}'"
        if kinematic:
            if entry["rho"] != "rhoInf" or entry["rhoInf"] is None:
                result["problems"].append(
                    f"{label} does not set 'rho rhoInf;' with an rhoInf value, while p is "
                    "kinematic (m2/s2)"
                )
                result["repairs"].append(
                    f"add 'rho rhoInf; rhoInf <density>;' to {entry['name']} -- 1.225 for "
                    "air at sea level, 998 for water"
                )
                result["status"] = "fail"
            elif entry["rhoInf"] == 1.0:
                result["problems"].append(
                    f"{label} has rhoInf 1, so its forces come out per unit density"
                )
                result["repairs"].append(
                    f"if the fluid is not density 1, set rhoInf in {entry['name']} to the "
                    "real density; if it is deliberate, say so where the numbers are reported"
                )
                if result["status"] != "fail":
                    result["status"] = "warn"
        elif entry["rho"] == "rhoInf":
            result["problems"].append(
                f"{label} sets rho rhoInf while p already carries pressure units"
            )
            result["repairs"].append(
                f"for a compressible case {entry['name']} should use 'rho rho;' and read "
                "the density field"
            )
            if result["status"] != "fail":
                result["status"] = "warn"

        if entry["type"] == "forceCoeffs":
            missing = [
                key for key in ("magUInf", "lRef", "Aref")
                if not entry.get(key)
            ]
            if missing:
                result["problems"].append(f"{label} is missing {', '.join(missing)}")
                result["repairs"].append(
                    f"set {', '.join(missing)} in {entry['name']}; a coefficient divided by "
                    "the wrong reference area is off by exactly that ratio"
                )
                result["status"] = "fail"
    return result


def check_units(case: Case, intent: Intent) -> list[Finding]:
    objects = find_force_objects(case.read("system/controlDict"))
    system = case.path / "system"
    # A function object is as often in its own file under system/ and pulled in with
    # #includeFunc as it is written out inside controlDict.
    for entry in sorted(system.glob("*")) if system.is_dir() else []:
        if entry.is_file() and entry.name != "controlDict":
            objects.extend(find_force_objects(read_text(entry), name_if_bare=entry.name))
    dimensions = parse_dimensions(case.field_texts.get("p", ""))
    result = force_units_diagnosis(dimensions, objects, has_fields=bool(case.field_texts))

    if dimensions == KINEMATIC_PRESSURE:
        kind = "kinematic (m2/s2)"
    elif dimensions == STATIC_PRESSURE:
        kind = "static (Pa)"
    elif dimensions is None:
        kind = "unreadable"
    else:
        kind = " ".join(f"{value:g}" for value in dimensions)
    measured = f"0/p dimensions {kind}; force function objects: " + (
        ", ".join(f"{entry['name']} ({entry['type']})" for entry in objects) or "none"
    )
    if result["status"] == "skipped":
        if result["problems"]:
            return [Finding(
                "units", "skipped", measured + ". " + "; ".join(result["problems"]),
                "the force objects are declared but no field is on disk yet, so there "
                "is nothing to judge them against",
                "; ".join(result["repairs"]),
            )]
        return [Finding(
            "units", "skipped", measured,
            "no forces or forceCoeffs function object, so nothing is computing a force "
            "from p here",
        )]
    if result["status"] == "ok":
        return [Finding(
            "units", "ok", measured,
            "the force objects account for density the way this pressure field requires",
        )]
    if dimensions is None:
        # The reading below is about a density setting judged against p's units.
        # With p unreadable there is no such judgement to report, and printing one
        # anyway is a finding whose `means` does not follow from its `measured`.
        return [Finding(
            "units", result["status"], measured + ". " + "; ".join(result["problems"]),
            "without p's dimensions there is no telling whether these force objects "
            "need a density -- incompressible p is kinematic (m2/s2) and compressible "
            "p is not, and the setting differs between them",
            "; ".join(result["repairs"]),
        )]
    return [Finding(
        "units", result["status"], measured + ". " + "; ".join(result["problems"]),
        "forces and coefficients from an incompressible run are wrong by a factor of "
        "rho unless the function object is told the density, and the wrong numbers "
        "look exactly as plausible as the right ones",
        "; ".join(result["repairs"]),
    )]


# -- disk ---------------------------------------------------------------------------


ASCII_BYTES_PER_VALUE = 20
"""A written scalar in ASCII is a line like '1.234567e-03\\n'."""

BINARY_BYTES_PER_VALUE = 8

DISK_WARN_FRACTION = 0.8


def expected_write_count(control: dict[str, str]) -> int | None:
    """How many time directories the run as configured will write."""
    start = as_float(control.get("startTime")) or 0.0
    end = as_float(control.get("endTime"))
    interval = as_float(control.get("writeInterval"))
    if end is None or not interval or interval <= 0:
        return None
    duration = end - start
    if duration <= 0:
        return 0
    write_control = control.get("writeControl", "timeStep")
    if write_control == "timeStep":
        delta = as_float(control.get("deltaT"))
        if not delta or delta <= 0:
            return None
        count = duration / delta / interval
    else:
        count = duration / interval
    count = int(math.floor(count))
    purge = as_float(control.get("purgeWrite"))
    if purge and purge > 0:
        count = min(count, int(purge))
    return max(0, count)


def estimate_write_bytes(
    cells: int | None, components: int, write_format: str = "ascii"
) -> int | None:
    """Bytes one time directory costs, from the cell count and the fields written."""
    if not cells or components <= 0:
        return None
    per_value = ASCII_BYTES_PER_VALUE if write_format != "binary" else BINARY_BYTES_PER_VALUE
    return int(cells * components * per_value * 1.1)


def measured_write_bytes(case_path: Path) -> tuple[int | None, str]:
    """The size of a time directory that already exists, which beats any estimate."""
    case_path = Path(case_path)
    times: list[tuple[float, Path]] = []
    for entry in case_path.iterdir() if case_path.is_dir() else []:
        if not entry.is_dir():
            continue
        value = as_float(entry.name)
        if value is None or value <= 0:
            continue
        times.append((value, entry))
    if not times:
        return None, ""
    _value, newest = max(times, key=lambda pair: pair[0])
    total = 0
    for path in newest.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return (total, f"the size of the existing time directory {newest.name}") if total else (None, "")


def disk_verdict(needed: int | None, free: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {"needed": needed, "free": free, "status": "skipped"}
    if needed is None or free is None:
        return result
    result["fraction"] = needed / free if free else float("inf")
    if needed > free:
        result["status"] = "fail"
    elif needed > free * DISK_WARN_FRACTION:
        result["status"] = "warn"
    else:
        result["status"] = "ok"
    return result


def human_bytes(value: float | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.4g} {unit}"
        size /= 1024
    return f"{size:.4g} TB"


def check_disk(case: Case, intent: Intent, free_bytes: int | None = None) -> list[Finding]:
    control = case.control
    writes = expected_write_count(control)
    per_write, source = measured_write_bytes(case.path)
    if per_write is None:
        components = sum(field_components(text) for text in case.field_texts.values())
        per_write = estimate_write_bytes(
            case.cell_count, components, control.get("writeFormat", "ascii")
        )
        source = (
            f"{case.cell_count:,} cells x {components} components written as "
            f"{control.get('writeFormat', 'ascii')}" if per_write else ""
        )
    if free_bytes is None:
        try:
            free_bytes = shutil.disk_usage(str(case.path)).free
        except OSError:
            free_bytes = None

    if writes is None or per_write is None:
        return [Finding(
            "disk", "skipped",
            f"write times {writes if writes is not None else 'unknown'}, bytes per write "
            f"{human_bytes(per_write)}, free {human_bytes(free_bytes)}",
            "not enough is known about the run or the mesh to size its output",
            "check endTime, writeInterval and writeControl in system/controlDict",
        )]

    needed = writes * per_write
    verdict = disk_verdict(needed, free_bytes)
    measured = (
        f"{writes} write time(s) x {human_bytes(per_write)} each "
        f"(from {source}) = {human_bytes(needed)}; free here {human_bytes(free_bytes)}"
    )
    if verdict["status"] == "ok":
        return [Finding(
            "disk", "ok", measured,
            "the run's output fits in the space available with room to spare",
        )]
    if verdict["status"] == "skipped":
        return [Finding("disk", "skipped", measured, "free space could not be read")]
    fewer = max(1, int(free_bytes * 0.5 / per_write)) if per_write else 1
    return [Finding(
        "disk", verdict["status"], measured,
        "a solver that fills the volume mid-run dies at the write, and what it leaves "
        "behind is a half-written time directory that the next reconstruct trips over",
        f"write less often -- about {fewer} write time(s) fits in half the free space -- "
        "or set purgeWrite, or write in binary (writeFormat binary in controlDict)",
    )]


# -- putting the checks together ---------------------------------------------------


SHOCK_CAPTURING = ("rhocentralfoam", "rhocentraldymfoam", "sonicfoam", "sonicdymfoam", "hisa")
"""Solvers that carry a discontinuity as a discontinuity. They are density-based: the
flux itself is reconstructed with a limiter, so a compression lands in a few cells and
stays there. All but `hisa` are in the image; `hisa` was built onto the workspace volume
and so is a property of an instance rather than of the image -- `hisa_env.py` says
whether this one has it."""

PRESSURE_BASED_STEADY = ("rhosimplefoam", "simplefoam", "sonicfoamsteady")
"""Solvers that reach a shock through a pressure equation. They can run transonic and
give sensible integrated loads; what they cannot do is hold the front sharp, because the
limiters that keep the pressure equation stable are the same terms that smear it."""

VOF = ("interfoam", "interisofoam", "compressibleinterfoam", "interphasechangefoam", "overinterdymfoam")


def _scheme_entries(body: str) -> list[tuple[str, str]]:
    """`key value;` pairs from a schemes block, where the key may hold brackets.

    `entry_values` cannot be used here and the difference is not cosmetic: its key
    pattern is `\\w+`, so on `div(phid,p) Gauss upwind;` it matches the last bare word
    before the value and returns `{"Gauss": "upwind"}` -- a parse that looks like it
    worked, drops the entry actually being looked for, and lets the check pass. Scheme
    keys are single tokens with no spaces (`div(phi,U)`,
    `div((phi|interpolate(rho)),p)`), so splitting each statement at its first run of
    whitespace is both simpler and right.
    """
    out: list[tuple[str, str]] = []
    for statement in strip_comments(body).split(";"):
        parts = statement.split(None, 1)
        if len(parts) == 2:
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def check_method(case: Case, intent: Intent) -> list[Finding]:
    """Can this discretisation represent the thing the run is being asked to show?

    Every other check in this file asks whether the case will *run*. This one asks
    whether, having run, it can *answer the question* -- which is a different question,
    and it is the one nobody asked on the ONERA M6 study of 2026-08-30 (FINDINGS.md
    F-52). That study was commissioned to show a lambda-shock. It got a 1.79M-cell mesh
    with a refinement box driven through the supersonic pocket, a farfield at sixteen
    chords, and a converged solve -- 3228 iterations, drag flat to six significant
    figures -- and it showed no shock at any of seven span stations, because
    `rhoSimpleFoam` with `cellLimited` gradients and an upwinded pressure flux spreads a
    compression over something like ten cells. Every input was defensible. The
    deliverable was unreachable from the first line of `fvSchemes`, and that was
    knowable in a second, before the mesh, for free.

    The competitor whose study prompted the replication put it in one sentence: "a
    smeared shock defeats the purpose of the benchmark", and they picked a density-based
    solver with AUSM+up and a van Leer limiter so the compression stayed inside about
    three cells. Choosing the solver *class from the deliverable* is the move, and it
    happens before meshing or not at all.

    Nothing here is guessed. With no `--resolve` the check says so and stops: a script
    that decides on its own what a run is for would be inventing the one input that
    makes the rest of it mean anything. A feature with no rule is reported as having no
    rule rather than passed, because silence from a checker reads as approval.

    Like the rest of preflight this edits nothing and blocks nothing. A `fail` here is a
    reading, the reading can be wrong, and there are good reasons to run a case whose
    method cannot resolve the headline feature -- getting the loads, debugging a mesh,
    establishing a starting field. The point is that the trade is made knowingly.
    """
    want = (intent.resolve or "").strip().lower()
    if not want:
        return [Finding(
            "method", "skipped",
            "no feature named",
            "nothing said what this run has to show, so there is nothing to check the "
            "discretisation against",
            "pass --resolve shock|interface|wake to have this checked",
        )]

    application = (case.application or "").strip()
    if not application:
        return [Finding(
            "method", "skipped",
            f"asked to resolve {want!r}; system/controlDict names no application",
            "the solver is what decides whether the feature survives, and it is not stated",
            "set 'application <solver>;' in system/controlDict",
        )]

    solver = application.lower()
    findings: list[Finding] = []

    if want in ("shock", "shocks", "lambda", "lambda-shock", "compression"):
        findings.extend(_shock_findings(case, application, solver))
    elif want in ("interface", "free-surface", "freesurface", "wave", "kelvin"):
        findings.extend(_interface_findings(case, application, solver))
    else:
        findings.append(Finding(
            "method", "skipped",
            f"asked to resolve {want!r}; solver {application}",
            "there is no rule here for that feature, so this check has no opinion and "
            "is not evidence that the method is adequate",
            "check by hand how many cells wide the feature is in the scheme you chose, "
            "or add a rule to preflight.check_method",
        ))
    return findings


def _shock_findings(case: Case, application: str, solver: str) -> list[Finding]:
    """Three separate ways a transonic setup fails to hold a front, reported apart.

    They are kept apart because the repairs are different: one is a change of solver,
    one is a single scheme line, one is a limiter coefficient. Rolled into a single
    verdict they would read as "transonic is hard", which is not actionable.
    """
    out: list[Finding] = []

    if solver in SHOCK_CAPTURING:
        out.append(Finding(
            "method", "ok",
            f"solver {application} is density-based",
            "the flux is reconstructed with a limiter, so a compression can stay inside "
            "a few cells",
        ))
    elif solver in PRESSURE_BASED_STEADY:
        out.append(Finding(
            "method", "fail",
            f"solver {application} is pressure-based and steady; "
            "shock-capturing alternatives: rhoCentralFoam and sonicFoam are in the image, "
            "hisa is a volume build that may or may not be on this instance (hisa_env.py "
            "answers that in a second)",
            "this family reaches a transonic state through a pressure equation whose "
            "stability depends on the limiters that smear a discontinuity, so the front "
            "arrives ten or so cells wide however fine the mesh is. Integrated loads can "
            "still be useful; a shock position, a shock structure or a merge point is not "
            "reachable this way",
            "for a shock as the deliverable, use rhoCentralFoam (transient, density-based) "
            "on the same mesh; keep this solver only if the loads, not the structure, are "
            "what is wanted, and say which in the report",
        ))
    else:
        out.append(Finding(
            "method", "warn",
            f"solver {application} is not in either table",
            "it may or may not carry a front sharply; this check does not know it",
            "confirm the solver is density-based, or that its flux scheme reconstructs "
            "with a limiter, before treating a shock position as a result",
        ))

    schemes = case.read("system/fvSchemes")
    if not schemes:
        out.append(Finding(
            "method", "skipped", "system/fvSchemes is not readable",
            "the scheme half of this cannot be checked",
        ))
        return out

    # entry_values, not block_entries: a schemes block is `key value;` pairs, where a
    # boundaryField is nested blocks. Reaching for the wrong one returns an empty list
    # and the check silently passes, which is the failure mode a checker can least
    # afford -- it was found here by asserting on the ONERA case that prompted it.
    div = _scheme_entries(block_body(schemes, "divSchemes") or "")
    pressure_flux = [
        (key, value) for key, value in div
        if "phid" in key.replace(" ", "") or (",p)" in key.replace(" ", "") and "phi" in key)
    ]
    first_order = [(k, v) for k, v in pressure_flux if re.search(r"\bupwind\b", v) and "linearUpwind" not in v]
    if first_order:
        shown = "; ".join(f"{k} -> {v.strip()}" for k, v in first_order[:2])
        out.append(Finding(
            "method", "fail",
            f"pressure flux is first order: {shown}",
            "this is the term that carries the compression, and first-order upwind on it "
            "spreads the front regardless of what the momentum schemes do. A momentum "
            "scheme upgraded to linearUpwind while this stays upwind reads as 'less "
            "dissipative' and is not, where it matters",
            "use a limited second-order interpolation on the pressure flux, or move to a "
            "density-based solver where the flux is reconstructed properly",
        ))
    elif pressure_flux:
        out.append(Finding(
            "method", "ok",
            "; ".join(f"{k} -> {v.strip()}" for k, v in pressure_flux[:2]),
            "the pressure flux is at least second order",
        ))

    grad = _scheme_entries(block_body(schemes, "gradSchemes") or "")
    fully_limited = [
        (key, value) for key, value in grad
        if "cellLimited" in value and re.search(r"cellLimited.*?\s1(\.0*)?\s*$", value.strip())
    ]
    if fully_limited:
        shown = "; ".join(f"{k} -> {v.strip()}" for k, v in fully_limited[:2])
        out.append(Finding(
            "method", "warn",
            f"gradients fully limited: {shown}",
            "a limiter coefficient of 1 clips the gradient hardest exactly at an extremum, "
            "and a shock is an extremum, so the scheme reverts to first order at the one "
            "place the answer lives",
            "relax toward 'cellLimited Gauss linear 0.33' once the run is past its "
            "start-up transient, or accept it and do not report a shock position",
        ))
    return out


def _interface_findings(case: Case, application: str, solver: str) -> list[Finding]:
    """A free surface is a discontinuity that moves, so the Courant number on it is
    the thing that blurs it, not the mesh."""
    out: list[Finding] = []
    if solver not in VOF:
        out.append(Finding(
            "method", "fail",
            f"solver {application} does not transport a phase fraction",
            "an interface cannot be resolved by a solver that has no interface; a "
            "single-phase run models the water surface as a wall or not at all",
            f"use one of {', '.join(sorted(VOF)[:3])} for a free surface, or state that "
            f"the run is a double-body model and that wave-making is excluded",
        ))
        return out

    out.append(Finding("method", "ok", f"solver {application} transports a phase fraction",
                       "the interface exists in the formulation"))
    control = case.control
    alpha_co = as_float(control.get("maxAlphaCo"))
    if alpha_co is not None and alpha_co > 1.0:
        out.append(Finding(
            "method", "warn",
            f"maxAlphaCo = {alpha_co:g}",
            "the interface is allowed to cross more than one cell per step, which smears "
            "it; the tutorials that ship this value are usually chasing throughput on a "
            "case whose answer is a force, not a wave profile",
            "set maxAlphaCo to 1 or below when the free-surface shape is the deliverable",
        ))
    elif alpha_co is not None:
        out.append(Finding("method", "ok", f"maxAlphaCo = {alpha_co:g}",
                           "the interface moves at most about one cell per step"))
    return out


CHECKS: dict[str, Callable[[Case, Intent], list[Finding]]] = {
    "method": check_method,
    "geometry": check_geometry,
    "patches": check_patches,
    "empty": check_empty,
    "reynolds": check_reynolds,
    "cells": check_cells,
    "checkmesh": check_checkmesh,
    "probe": check_probe,
    "courant": check_courant,
    "residuals": check_residuals,
    "units": check_units,
    "disk": check_disk,
}

CHECK_ORDER = tuple(CHECKS)


def select_checks(requested: str | None, skip_probe: bool = False) -> list[str]:
    """Which checks to run, in the fixed order, from a comma-separated selection."""
    if not requested:
        names = [name for name in CHECK_ORDER if not (skip_probe and name == "probe")]
        return names
    wanted = [part.strip().lower() for part in requested.split(",") if part.strip()]
    unknown = [name for name in wanted if name not in CHECKS]
    if unknown:
        raise ValueError(
            f"unknown check(s): {', '.join(unknown)}. Known: {', '.join(CHECK_ORDER)}"
        )
    return [name for name in CHECK_ORDER if name in wanted]


def run_checks(
    case_path: Path | str,
    names: list[str] | None = None,
    intent: Intent = Intent(),
    *,
    runner: Callable[[list[str], Path], dict[str, Any]] | None = None,
    log: str | Path | None = None,
    free_bytes: int | None = None,
) -> list[Finding]:
    """Every requested check against one case, worst findings first.

    A check that raises is reported as a `skipped` finding naming the exception
    rather than taking the report down with it: nine useful answers and one crash is
    a better preflight than no preflight, and the crash is itself worth reporting.
    """
    case = Case(case_path)
    findings: list[Finding] = []
    for name in names or list(CHECK_ORDER):
        function = CHECKS[name]
        try:
            if name == "probe":
                findings.extend(check_probe(case, intent, runner=runner))
            elif name == "residuals":
                findings.extend(check_residuals(case, intent, log=log))
            elif name == "disk":
                findings.extend(check_disk(case, intent, free_bytes=free_bytes))
            else:
                findings.extend(function(case, intent))
        except Exception as error:  # a broken check must not cost the other ten
            findings.append(Finding(
                name, "skipped", f"the check raised {type(error).__name__}: {error}",
                "this is a bug in preflight.py, not necessarily a problem with the case",
            ))
    return sorted(findings, key=lambda finding: -_SEVERITY.get(finding.status, 0))


def render(findings: list[Finding], case_path: Path | str) -> str:
    counts = summarise(findings)
    lines = [f"# preflight {case_path}"]
    lines.append(
        "  ".join(f"{status} {counts[status]}" for status in STATUSES if counts.get(status))
        or "nothing checked"
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
    lines.append("Repairs are suggestions. Nothing here has changed the case.")
    return "\n".join(lines)


def as_json(findings: list[Finding], case_path: Path | str) -> str:
    return json.dumps(
        {
            "case": str(case_path),
            "worst": worst_status(findings),
            "counts": summarise(findings),
            "findings": [finding.as_dict() for finding in findings],
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, nargs="?", help="the case directory")
    parser.add_argument("--checks", default="", help=f"comma-separated: {', '.join(CHECK_ORDER)}")
    parser.add_argument("--list-checks", action="store_true", help="name the checks and stop")
    parser.add_argument("--json", action="store_true", help="findings as JSON")
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    parser.add_argument("--re", type=float, default=None, help="the Reynolds number intended")
    parser.add_argument("--u", type=float, default=None, help="reference velocity, m/s")
    parser.add_argument("--l", type=float, default=None, help="characteristic length, m")
    parser.add_argument(
        "--resolve", default="",
        help="the feature the run has to show (shock, interface); checked against the "
             "solver and schemes before anything is meshed",
    )
    parser.add_argument("--log", default=None, help="the solver log to read residuals from")
    parser.add_argument("--no-probe", action="store_true", help="skip the one-step solver probe")
    args = parser.parse_args(argv)

    if args.list_checks:
        for name in CHECK_ORDER:
            print(name)
        return 0
    if args.case is None:
        parser.error("a case directory is required")

    try:
        names = select_checks(args.checks, skip_probe=args.no_probe)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    intent = Intent(reynolds=args.re, velocity=args.u, length=args.l, resolve=args.resolve)
    findings = run_checks(args.case, names, intent, log=args.log)
    text = as_json(findings, args.case) if args.json else render(findings, args.case)
    print(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        try:
            # Resolved, because `--out` is relative to the working directory and
            # `record` stores it relative to the study root. A relative one handed
            # over as written is joined to the root instead, and the manifest ends up
            # pointing at a file that is not there.
            study_state.record(
                "report", out.resolve(), root=args.case, case=Path(args.case).name,
                label="preflight", worst=worst_status(findings),
                checks=names, counts=summarise(findings),
            )
        except Exception as error:
            print(f"(the report was written but not registered: {error})", file=sys.stderr)

    return 1 if worst_status(findings) == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
