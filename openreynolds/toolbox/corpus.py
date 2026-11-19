#!/usr/bin/env python3
"""The corpus: tutorials and finished studies, indexed so they can be asked for.

`$FOAM_TUTORIALS` holds 556 working cases and `/work/<study-id>/` the studies this
instance has finished, and nothing indexed either. This builds two indexes and keeps
them two -- `tutorials.index.jsonl` (vendor) and `studies.index.jsonl` (earned) -- with
the tier on every row.

There is no benchmark tier under them, so nothing here is an anchor: tutorials carry no
reference value, an earned row's verdict says how far a study got rather than whether
it was right, and a value that was not read is null rather than guessed.

The reader is a keyword scraper, not a parser: no `#include`, no macros.

    python3 corpus.py build                                   # both tiers, to /work/.toolbox/corpus
    python3 corpus.py build --if-stale                        # only when stale (the harness runs this)
    python3 corpus.py build --tutorials DIR --work DIR --out DIR
    python3 corpus.py read system/controlDict                 # every entry
    python3 corpus.py read constant/turbulenceProperties --key RASModel
    python3 corpus.py boundaries 0.orig/U                     # patch -> BC type

From another script: `entries(read(path))`, `entry(path, key)`,
`boundary_types(read(path))`."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state  # noqa: E402  (sibling script, not a package)

BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT = re.compile(r"//[^\n]*")

ENTRY = re.compile(r"^[ \t]*([^\s{};#][^\s{};]*)[ \t]+([^;{}\n]+);", re.MULTILINE)
"""One `key value;` on one line. The key is any non-space run, because `fvSchemes` is
keyed by terms (`div(phi,U)`) and `fvSolution` by quoted regexes (`".*"`). An entry
written across two lines is not read -- a limitation, not a guess."""

TOKENS = re.compile(r'"[^"\n]*"|[{};]|[^\s{};]+')
"""Words, braces and semicolons -- enough to walk a block without parsing it.
Quoted names come through whole, because patch names in the shipped tree are
regularly regular expressions: `"wall.*"`, `"(lowerWall|upperWall)"`."""


def strip_comments(text: str) -> str:
    """Block comments first, then line comments: four banners carry `https://`, and a
    `//` pass run first leaves them unterminated. A block becomes a space, not nothing."""
    return LINE_COMMENT.sub("", BLOCK_COMMENT.sub(" ", text))


def entries(text: str) -> dict[str, str]:
    """Every `key value;` in comment-stripped text, whatever block it sits in.

    Flat, because ESI keeps the turbulence model a block down. The first occurrence wins,
    and whitespace in values is collapsed."""
    found: dict[str, str] = {}
    for key, value in ENTRY.findall(text):
        if key not in found:
            found[key] = " ".join(value.split())
    return found


def block(text: str, name: str) -> str | None:
    """The inside of `name { ... }`, or None if there is no such block."""
    for match in re.finditer(rf"\b{re.escape(name)}\b", text):
        opening = text.find("{", match.end())
        if opening < 0 or text[match.end() : opening].strip():
            continue
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[opening + 1 : index]
        return text[opening + 1 :]  # unbalanced; the rest of the file is the block
    return None


def boundary_types(text: str) -> dict[str, str]:
    """`boundaryField` read as patch name -> BC type. Scoped to the block, so `FoamFile`
    is not a patch and a patch's `value` is not its type."""
    inner = block(text, "boundaryField")
    if inner is None:
        return {}
    patches: dict[str, str] = {}
    depth = 0
    candidate = ""
    patch = ""
    key = ""
    words: list[str] = []
    for match in TOKENS.finditer(inner):
        token = match.group(0)
        if token == "{":
            depth += 1
            if depth == 1:
                patch = candidate.strip('"')
                key, words = "", []
            candidate = ""
        elif token == "}":
            depth -= 1
            if depth <= 0:
                depth, patch = 0, ""
            key, words = "", []
        elif token == ";":
            if depth == 1 and patch and key == "type" and words and patch not in patches:
                patches[patch] = " ".join(words)
            key, words = "", []
        elif depth == 0:
            candidate = token
        elif not key:
            key = token
        else:
            words.append(token)
    return patches


def read(path: Path | str) -> str:
    """One dictionary file, comment-stripped, or `""` when it will not read: one binary
    file or dangling link costs that case, not the harvest."""
    try:
        return strip_comments(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return ""


def entry(path: Path | str, key: str) -> str | None:
    """One value out of one file, or None -- absent and unreadable read the same."""
    return entries(read(path)).get(key)


# -- the vendor tier ---------------------------------------------------------------


SCHEMA_VERSION = 4
"""Bumped when the shape on disk changes, a row's or the stamp's: `search.py` rebuilds
an index whose schema it does not recognise. Left unbumped once, an old index passed
the check and answered a real `failure` query with nothing. 4 added `studies` to the
stamp. `tests/test_toolbox_search.py` pins both shapes to this number."""

CORPUS_DIR = "/work/.toolbox/corpus"
TUTORIALS_INDEX = "tutorials.index.jsonl"
STUDIES_INDEX = "studies.index.jsonl"
STAMP = "corpus.stamp.json"
RETRIEVALS = "retrievals.jsonl"

MESHERS = frozenset({"snappyHexMesh", "blockMesh", "foamyHexMesh", "foamyQuadMesh",
                     "cartesianMesh", "tetMesh", "extrudeMesh", "refineMesh"})
"""Applications that build a mesh rather than solve on one, so `runs` can keep them
out of a solver query. Only `snappyHexMesh` appears in v2512 (18 cases)."""

STEADY_DDT = frozenset({"steadyState", "none"})
"""`none` drops the time derivative altogether -- `potentialFoam`'s way of being steady."""

TRANSIENT_DDT = frozenset({"Euler", "backward", "CrankNicolson", "localEuler", "bounded"})
"""A scheme on neither list reads as null, not transient."""

GEOMETRY_SUFFIXES = frozenset({".stl", ".obj", ".ply", ".vtk", ".stlb", ".gz"})

NO_TURBULENCE: dict[str, str | None] = {"simulation_type": None, "model": None}
NO_REGIME: dict[str, object] = {"class": None, "compressible": None, "steady": None,
                                "Re": None, "Ma": None, "shedding_risk": None}
"""What a row says when there was nothing to read: the same shape, with nulls in it."""

TIME_DIRS = ("0.orig", "0")
"""In that order: in a case that has run, `0/` is output and `0.orig/` the specification."""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def foam_version() -> str:
    """`$WM_PROJECT_VERSION`, or `unknown`. Never guessed from anything else."""
    return os.environ.get("WM_PROJECT_VERSION") or "unknown"


def foam_tutorials() -> Path | None:
    """`$FOAM_TUTORIALS` as a path, or None. Not `Path("")`, which is `.` and truthy: as
    an argparse default it made `build` index the working directory and report success."""
    value = os.environ.get("FOAM_TUTORIALS", "").strip()
    return Path(value) if value else None


def foam_fork(tree: Path | str) -> str:
    """Which fork's tree this is: the Foundation fork keeps turbulence in
    `momentumTransport`, ESI in `turbulenceProperties`, and `unknown` when neither shows."""
    for _ in Path(tree).rglob("constant/momentumTransport"):
        return "foundation"
    for _ in Path(tree).rglob("constant/turbulenceProperties"):
        return "esi"
    return "unknown"


def turbulence_of(directory: Path) -> dict[str, str | None]:
    """`{simulation_type, model}` from whichever fork's file is in `directory`. ESI writes
    `RASModel` / `LESModel`, the Foundation fork `model`; both are tried."""
    for name in ("turbulenceProperties", "momentumTransport"):
        found = entries(read(directory / name))
        if not found:
            continue
        simulation = found.get("simulationType")
        candidates = [f"{simulation}Model"] if simulation else []
        candidates += ["RASModel", "LESModel", "model"]
        model = next((found[key] for key in candidates if key in found), None)
        return {"simulation_type": simulation, "model": model}
    return NO_TURBULENCE.copy()


def regions_of(case: Path) -> dict[str, dict[str, str | None]]:
    """Per-region turbulence. 16 of the 18 multi-region cases keep it only under
    `constant/<region>/`, and read at the case level would index as null."""
    if not (case / "constant" / "regionProperties").exists():
        return {}
    found: dict[str, dict[str, str | None]] = {}
    try:
        children = sorted(child for child in (case / "constant").iterdir() if child.is_dir())
    except OSError:
        return {}
    for child in children:
        turbulence = turbulence_of(child)
        if turbulence["simulation_type"] or turbulence["model"]:
            found[child.name] = turbulence
    return found


def has_geometry(case: Path) -> bool:
    for holder in ("triSurface", "geometry"):
        try:
            children = list((case / "constant" / holder).iterdir())
        except OSError:
            continue
        if any(child.suffix.lower() in GEOMETRY_SUFFIXES for child in children):
            return True
    return False


def mesh_type_of(case: Path) -> str:
    """How this case's mesh is built. snappy is checked before blockMesh: 62 of the 64
    snappy cases also carry a `blockMeshDict` for their background mesh."""
    system = case / "system"
    if (system / "snappyHexMeshDict").exists():
        return "snappyHexMesh"
    if (system / "blockMeshDict").exists():
        return "blockMesh"
    if has_geometry(case):
        return "surface"
    return "unknown"


def steady_of(case: Path) -> bool | None:
    """Steady or transient, read off the `default` in `ddtSchemes` -- scoped to the block,
    since every schemes block has a `default`."""
    schemes = block(read(case / "system" / "fvSchemes"), "ddtSchemes")
    if schemes is None:
        return None
    default = entries(schemes).get("default")
    if not default:
        return None
    scheme = default.split()[0]
    if scheme in STEADY_DDT:
        return True
    if scheme in TRANSIENT_DDT:
        return False
    return None


def compressible_of(case: Path) -> bool | None:
    """Compressibility, from which properties file the case carries.

    `physicalProperties` is opened rather than taken as incompressible: on the Foundation
    fork it replaced both other files, and the one in v2512 is electrostatics."""
    constant = case / "constant"
    if (constant / "thermophysicalProperties").exists():
        return True
    if (constant / "transportProperties").exists():
        return False
    physical = constant / "physicalProperties"
    if physical.exists():
        found = entries(read(physical))
        if "thermoType" in found or block(read(physical), "thermoType") is not None:
            return True
        if "nu" in found or "viscosityModel" in found or "transportModel" in found:
            return False
    return None


def regime_of(case: Path) -> dict[str, object]:
    """The regime, carrying only what was read. Internal versus external is not readable
    from a tutorial, and `Re`, `Ma` and `shedding_risk` need scales a tutorial does not
    reliably state, so they stay null."""
    steady = steady_of(case)
    compressible = compressible_of(case)
    parts = []
    if compressible is not None:
        parts.append("compressible" if compressible else "incompressible")
    if steady is not None:
        parts.append("steady" if steady else "transient")
    return {**NO_REGIME, "class": "-".join(parts) or None, "compressible": compressible,
            "steady": steady}


def bc_map_of(case: Path) -> dict[str, dict[str, str]]:
    """Every field in the first initial-condition directory that exists, as patch -> BC type."""
    for name in TIME_DIRS:
        directory = case / name
        if not directory.is_dir():
            continue
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return {}
        fields: dict[str, dict[str, str]] = {}
        for child in children:
            if not child.is_file():
                continue
            types = boundary_types(read(child))
            if types:
                fields[child.name] = types
        return fields
    return {}


def case_roots(tree: Path | str) -> list[Path]:
    """Every distinct case under `tree`: the grandparent of a `system/controlDict`.

    A symlinked file is a case (7 are). A symlinked directory is the same case by another
    name, so cases are de-duplicated by resolved path."""
    seen: dict[Path, Path] = {}
    for control in sorted(Path(tree).rglob("system/controlDict")):
        case = control.parent.parent
        try:
            key = case.resolve()
        except OSError:
            key = case
        seen.setdefault(key, case)
    return list(seen.values())


def harvest_case(case: Path, *, of_version: str, of_fork: str) -> dict[str, object] | None:
    """One vendor row, or None when no entry at all could be read from `controlDict`. A
    `controlDict` naming no `application` is still a case (the `foamyHexMesh` tutorials)."""
    found = entries(read(case / "system" / "controlDict"))
    if not found:
        return None
    executable = found.get("application")
    return {
        "path": str(case),
        "tier": "vendor",
        "solver": {"executable": executable, "module": None},
        "runs": bool(executable) and executable not in MESHERS,
        "turbulence": turbulence_of(case / "constant"),
        "regions": regions_of(case),
        "regime": regime_of(case),
        "mesh_type": mesh_type_of(case),
        "bc_map": bc_map_of(case),
        "of_version": of_version,
        "of_fork": of_fork,
        "verdict": None,
        "provenance": {"indexed_at": now_iso(), "schema_version": SCHEMA_VERSION},
    }


def harvest_tutorials(
    tree: Path | str, *, of_version: str | None = None, of_fork: str | None = None
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Every case under `tree`, and how many were indexed and skipped. Nothing raises, and
    the counts are what make a corpus that silently halved visible."""
    version = foam_version() if of_version is None else of_version
    fork = foam_fork(tree) if of_fork is None else of_fork
    rows: list[dict[str, object]] = []
    skipped = 0
    for case in case_roots(tree):
        row = harvest_case(case, of_version=version, of_fork=fork)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    return rows, {"indexed": len(rows), "skipped": skipped}


# -- the earned tier ---------------------------------------------------------------


WORK_ROOT = "/work"

RUNG_FIELDS = ("class", "rung", "name", "status", "value", "known")
"""What a rung row carries out of the manifest, always all six."""


def phase_table(study: Path) -> dict[str, object] | None:
    """The study's phase table as written, or None. Not `study_state.load_phases`, which
    returns a blank table for a missing file -- no record and no progress are different."""
    path = study / study_state.STATE_DIR / study_state.PHASES_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def verdict_of(table: dict[str, object] | None) -> str | None:
    """How far the study got, read off the phase table: how far, not whether it was right.

    `failed` outranks everything, then `running`, then `pending` (`incomplete`). `skipped`
    does not count against completion, but an all-`skipped` table is not completed. A
    status this does not know reads `unrecognised` rather than the nearest known one."""
    if table is None:
        return None
    rows = table.get("phases")
    if not isinstance(rows, list):
        return None
    statuses = [str(row.get("status") or "") for row in rows if isinstance(row, dict)]
    if not statuses:
        return None
    if set(statuses) - set(study_state.STATUSES):
        return "unrecognised"
    if "failed" in statuses:
        return "failed"
    if "running" in statuses:
        return "running"
    if "pending" in statuses:
        return "incomplete"
    return "completed" if "done" in statuses else "incomplete"


def within(case: Path, study: Path) -> str:
    """`case` written relative to `study`, in posix form, as the manifest writes paths."""
    try:
        return case.resolve().relative_to(study.resolve()).as_posix()
    except (ValueError, OSError):
        return case.name


def primary_case(study: Path, cases: list[Path], recorded: str) -> Path | None:
    """Which case represents this study: the one `phases.json` records, when it is there;
    otherwise the only one; otherwise None rather than whichever sorted first."""
    if recorded:
        target = (study / recorded).resolve()
        for case in cases:
            try:
                if case.resolve() == target:
                    return case
            except OSError:
                continue
    return cases[0] if len(cases) == 1 else None


def artifacts_of(study: Path) -> dict[str, int]:
    """The manifest counted by kind, rungs left out (they are reported under `rungs`)."""
    counted: dict[str, int] = {}
    for row in study_state.artifacts(root=study, exists=False):
        kind = str(row.get("kind") or "")
        if not kind or kind == study_state.RUNG_KIND:
            continue
        counted[kind] = counted.get(kind, 0) + 1
    return dict(sorted(counted.items()))


def rungs_of(study: Path) -> list[dict[str, object]]:
    """Recorded ladder rungs: the measured value beside its known answer -- the one thing in
    the corpus checked against something outside it. Not a benchmark tier."""
    found = []
    for row in study_state.rung_evidence(root=study):
        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        found.append({field: meta.get(field) for field in RUNG_FIELDS})
    return found


def notes_of(study: Path, table: dict[str, object] | None) -> list[str]:
    """The free text a study left behind -- phase notes, artifact labels, rung notes -- which
    is what `search.py failure` matches on."""
    found: list[str] = []
    rows = (table or {}).get("phases")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        note = str(row.get("note") or "").strip()
        if note:
            name = str(row.get("name") or "").strip()
            found.append(f"{name}: {note}" if name else note)
    for row in study_state.artifacts(root=study, exists=False):
        label = str(row.get("label") or "").strip()
        if label:
            found.append(label)
        meta = row.get("meta")
        if isinstance(meta, dict):
            note = str(meta.get("note") or "").strip()
            if note:
                found.append(note)
    return found


def study_roots(work: Path | str) -> list[Path]:
    """Every study under `work`: a directory one level down carrying a `.reynolds/`."""
    root = Path(work)
    try:
        children = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        return []
    return [child for child in children if (child / study_state.STATE_DIR).is_dir()]


def harvest_study(study: Path, *, of_version: str, of_fork: str) -> dict[str, object] | None:
    """One earned row: a vendor row's fields plus the study's own.

    R2 lives here: `tier` is `earned` and stays that way, and a study's own result is not
    a reference value."""
    table = phase_table(study)
    recorded = str((table or {}).get("case") or "")
    cases = case_roots(study)
    case = primary_case(study, cases, recorded)
    executable = entry(case / "system" / "controlDict", "application") if case else None
    return {
        "path": str(study),
        "study_id": study.name,
        "tier": "earned",
        "case": within(case, study) if case else None,
        "cases": sorted(within(found, study) for found in cases),
        "solver": {"executable": executable, "module": None},
        "runs": bool(executable) and executable not in MESHERS,
        "turbulence": turbulence_of(case / "constant") if case else NO_TURBULENCE.copy(),
        "regions": regions_of(case) if case else {},
        "regime": regime_of(case) if case else NO_REGIME.copy(),
        "mesh_type": mesh_type_of(case) if case else "unknown",
        "bc_map": bc_map_of(case) if case else {},
        "of_version": of_version,
        "of_fork": of_fork,
        "verdict": verdict_of(table),
        "artifacts": artifacts_of(study),
        "rungs": rungs_of(study),
        "notes": notes_of(study, table),
        "provenance": {"indexed_at": now_iso(), "schema_version": SCHEMA_VERSION},
    }


def harvest_studies(
    work: Path | str = WORK_ROOT, *, of_version: str | None = None, of_fork: str | None = None
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Every study under `work`, and what was indexed and skipped. A missing `/work` is an
    empty index, not an error."""
    version = foam_version() if of_version is None else of_version
    fork = foam_fork(work) if of_fork is None else of_fork
    rows: list[dict[str, object]] = []
    skipped = 0
    for study in study_roots(work):
        row = harvest_study(study, of_version=version, of_fork=fork)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    return rows, {"indexed": len(rows), "skipped": skipped}


def studies_fingerprint(work: Path | str) -> str:
    """Each study's name and the modification times of its two record files, hashed.
    A study finishing moves none of schema, tree or version; this is what it does move."""
    parts = []
    for study in study_roots(work):
        state = study / study_state.STATE_DIR
        times = []
        for name in (study_state.PHASES_FILE, study_state.MANIFEST):
            try:
                times.append(str((state / name).stat().st_mtime_ns))
            except OSError:
                times.append("-")
        parts.append(f"{study.name}:{','.join(times)}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _resolved(path: Path | str) -> str:
    """An absolute path for the stamp, so two spellings of one tree compare equal."""
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def write_index(path: Path, rows: list[dict[str, object]]) -> Path:
    """One JSON object per line, through a temporary file. Replaced rather than appended, so
    a case deleted upstream leaves the index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def build(
    tree: Path | str, out_dir: Path | str = CORPUS_DIR, *, work: Path | str = WORK_ROOT
) -> dict[str, object]:
    """Harvest both tiers into `out_dir` and write the stamp. Returns the stamp.

    Two files, so "never merged" is a property of the corpus rather than a promise. The
    stamp records what it was built from, which is what `search.py` checks."""
    out = Path(out_dir)
    version, fork = foam_version(), foam_fork(tree)
    rows, counts = harvest_tutorials(tree, of_version=version, of_fork=fork)
    write_index(out / TUTORIALS_INDEX, rows)
    earned, earned_counts = harvest_studies(work, of_version=version, of_fork=fork)
    write_index(out / STUDIES_INDEX, earned)
    stamp = {
        "of_version": version,
        "of_fork": fork,
        "built_at": now_iso(),
        "counts": {"tutorials": counts, "studies": earned_counts},
        "schema_version": SCHEMA_VERSION,
        # What it was built from, so a switched tree, another `--work` or a study that
        # changed since is a reason for `search.py` to rebuild.
        "tutorials": _resolved(tree),
        "work": _resolved(work),
        "studies": studies_fingerprint(work),
    }
    tmp = (out / STAMP).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out / STAMP)
    return stamp


# -- the command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    reading = sub.add_parser("read", help="Every entry in one dictionary.")
    reading.add_argument("path", type=Path)
    reading.add_argument("--key", default="", help="Print one value instead of all of them.")
    reading.add_argument("--json", action="store_true", help="Machine-readable.")

    bounds = sub.add_parser("boundaries", help="A field file's patches and their BC types.")
    bounds.add_argument("path", type=Path)
    bounds.add_argument("--json", action="store_true")

    building = sub.add_parser("build", help="Harvest the tutorial tree into an index.")
    building.add_argument("--tutorials", type=Path, default=foam_tutorials(),
                          help="The tree to walk (default: $FOAM_TUTORIALS).")
    building.add_argument("--work", type=Path, default=Path(WORK_ROOT),
                          help=f"Where the studies are (default: {WORK_ROOT}).")
    building.add_argument("--out", type=Path, default=Path(CORPUS_DIR),
                          help=f"Where to write (default: {CORPUS_DIR}).")
    building.add_argument("--json", action="store_true")
    building.add_argument("--if-stale", action="store_true",
                          help="Build only when the index is missing or stale.")

    args = ap.parse_args(argv)

    if args.command == "build":
        if args.if_stale:
            import search  # sibling; imported here because it imports this module

            reason = search.rebuild_reason(args.out, tree=args.tutorials, work=args.work)
            if reason is None:
                print(f"index is current: {args.out}")
                return 0
            print(f"rebuilding: {reason}")
        if args.tutorials is None or not args.tutorials.is_dir():
            print(f"no tutorial tree at {args.tutorials or '$FOAM_TUTORIALS (unset)'}"
                  " -- pass --tutorials, or source the OpenFOAM environment", file=sys.stderr)
            return 1
        stamp = build(args.tutorials, args.out, work=args.work)
        if args.json:
            print(json.dumps(stamp, indent=2, sort_keys=True))
            return 0
        for index, tier in ((TUTORIALS_INDEX, "tutorials"), (STUDIES_INDEX, "studies")):
            counts = stamp["counts"][tier]
            print(f"{args.out / index}: indexed {counts['indexed']}, "
                  f"skipped {counts['skipped']}")
        print(f"built against {stamp['of_fork']} {stamp['of_version']}")
        return 0

    text = read(args.path)
    if not text:
        print(f"nothing read from {args.path}", file=sys.stderr)
        return 1
    found = entries(text) if args.command == "read" else boundary_types(text)
    if args.command == "read" and args.key:
        value = found.get(args.key)
        if args.json:
            print(json.dumps({args.key: value}))
        elif value is None:
            print(f"{args.key} is not set in {args.path}", file=sys.stderr)
            return 1
        else:
            print(value)
        return 0
    if args.json:
        print(json.dumps(found, indent=2, sort_keys=True))
    elif not found and args.command == "boundaries":
        print(f"no boundaryField in {args.path}")
    else:
        width = max((len(key) for key in found), default=0)
        for key in (sorted(found) if args.command == "read" else found):
            print(f"{key:<{width}}  {found[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
