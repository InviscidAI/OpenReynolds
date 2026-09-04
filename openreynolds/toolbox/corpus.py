#!/usr/bin/env python3
"""The corpus: tutorials and finished studies, indexed so they can be asked for.

`$FOAM_TUTORIALS` holds 556 working cases and every one of them is a precedent --
a solver choice, a scheme set and a boundary specification that somebody got to
run. Nothing indexes them, so the only way to reach one is to `find` the tree by
hand and read what turns up. Completed studies under `/work/<study-id>/` are in the
same position: they are on the Volume, they are readable, and nothing points at a
relevant one.

This module builds the indexes. Two of them, and they stay two: `tutorials.index.jsonl`
for what the vendor shipped, `studies.index.jsonl` for what this instance has done.
Every row says which tier it is from, and they are never merged.

That separation is the load-bearing part. The corpus design has four tiers and this
builds the bottom two, neither of which is an anchor: a benchmark tier holds published
experimental values, and there is not one here. Without it the corpus is a closed loop
-- a convention that is subtly wrong passes review, enters the earned tier, is
retrieved into the next study, and becomes the norm by repetition, at which point the
correct value is the one that looks unusual. Nothing here fixes that. What it can do is
refuse to hide it: tutorials never carry a reference value, earned rows carry a verdict
saying how far a study got rather than whether it was right, a value that was not read
is null rather than guessed, and the tier is on every row.

The reader underneath is a keyword scraper and not a parser, deliberately. It does not
expand `#include`, it does not resolve macros, and it does not know a dimension set
from a velocity. It answers "what does this file say `application` is" across 556 cases
in under a second, and where it cannot answer it says so.

    python3 corpus.py build                                   # both tiers, to /work/.toolbox/corpus
    python3 corpus.py build --tutorials DIR --work DIR --out DIR
    python3 corpus.py read system/controlDict                 # every entry, as JSON
    python3 corpus.py read constant/turbulenceProperties --key RASModel
    python3 corpus.py boundaries 0.orig/U                     # patch -> BC type

The distinction that matters is between a value and a guess. `None` is a real
answer here and is written into the index as null. The corpus this feeds has no
benchmark tier under it, so an invented value that looks measured is the failure
mode with no floor: it would be retrieved into the next study, and the next, and
become the convention by repetition. Nothing in this file infers a value it did not
read.

Reading it from another script is the same three calls the command line makes:
`entries(read(path))`, `entry(path, key)`, `boundary_types(read(path))`.
"""

from __future__ import annotations

import argparse
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
"""One `key value;` on one line.

The key is any run of non-space that is not a brace or a semicolon, because a key in
this format is not an identifier. `fvSchemes` is mostly keyed by the term being
discretised -- `div(phi,U)` in **299 cases**, `div(phi,k)` in 209, and
`div(((rho*nuEff)*dev2(T(grad(U)))))` in 186 -- and `fvSolution` is keyed by quoted
regular expressions over field names: `".*"`, `"pa.*"`, `"Ua.*"`. A key pattern
restricted to identifiers reads none of those, which is most of what a query about
schemes is asking about.

A leading `#` is excluded so `#include` and `#includeEtc` cannot match. They carry no
semicolon and so would not match anyway; the exclusion says why on purpose rather
than by accident.

A bare block name (`RAS`, `boundaryField`) still cannot match, because it has no value
beside it. Excluding `{}` from the value is what keeps `RAS` from capturing the block
that follows it, and excluding the newline makes this a line scanner: an entry written
across two lines is not read, which is a limitation and not a guess. Nothing in a list
body matches either -- `(0 0 0)` and `hex (0 1 2 3 4 5 6 7) (20 20 1)` carry no
semicolon of their own, and the `);` that closes them has no value."""

TOKENS = re.compile(r'"[^"\n]*"|[{};]|[^\s{};]+')
"""Words, braces and semicolons -- enough to walk a block without parsing it.
Quoted names come through whole, because patch names in the shipped tree are
regularly regular expressions: `"wall.*"`, `"(lowerWall|upperWall)"`."""


def strip_comments(text: str) -> str:
    """Block comments first, then line comments.

    The order is load-bearing, for four files. Every dictionary in the tree opens
    with the OpenFOAM banner, which is one block comment; v2512 writes the website
    in it without a scheme, but four files still carry `https://www.openfoam.com`.
    A `//` pass run first cuts those four banners at the scheme and leaves them
    unterminated, and the block pass then eats the file down to whatever `*/` comes
    next -- taking the real entries in between with it.

    A block becomes a single space rather than nothing, so
    `application /* was: icoFoam */ simpleFoam;` does not read as
    `applicationsimpleFoam`.
    """
    return LINE_COMMENT.sub("", BLOCK_COMMENT.sub(" ", text))


def entries(text: str) -> dict[str, str]:
    """Every `key value;` in comment-stripped text, whatever block it sits in.

    Flat on purpose. ESI keeps the turbulence model one level down --
    `RAS { RASModel kEpsilon; }` -- and the Foundation fork keys it differently
    again, so a reader that only sees the top level of `turbulenceProperties` finds
    the simulation type and never the model. Flattening costs the block context,
    which none of the fields this index carries needs.

    The first occurrence of a key wins. A dictionary that sets one key twice has
    been edited rather than authored, and the earlier line is the one the file
    reads as.

    Values have their whitespace collapsed: the shipped tree writes
    `application     interFoam;` and `application       adjointOptimisationFoam;`
    and nothing normalises them upstream.
    """
    found: dict[str, str] = {}
    for key, value in ENTRY.findall(text):
        if key not in found:
            found[key] = " ".join(value.split())
    return found


def block(text: str, name: str) -> str | None:
    """The inside of `name { ... }`, or None if there is no such block.

    Braces are counted rather than matched by regex, because patch entries nest.
    An occurrence of `name` that is not followed by a brace is passed over, so a
    word appearing in a value does not open a phantom block.
    """
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
    """`boundaryField` read as patch name -> BC type.

    The one thing here that needs block awareness rather than a flat scan. `type`
    appears once per patch and `value` sits beside it, so a flat `entries()` would
    report one type for the whole file and might report `uniform (10 0 0)` as it.
    Scoping to `boundaryField` is also what keeps `FoamFile` -- a word followed by
    a brace at the top of every field file -- from being indexed as a boundary on
    every case in the corpus.

    A patch's own nested block (`codeOptions { ... }`) is walked through and its
    entries ignored, and the first `type` a patch states is the one recorded.
    """
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
    """One dictionary file, comment-stripped. `""` when it will not read.

    A case that cannot be read costs that case and not the harvest: 556 of these
    are opened in one pass, some of them are binary (`points`, `owner`), some are
    dangling symlinks into a build tree, and one unreadable file must not take the
    index down. Decoding errors are replaced rather than raised on, for the same
    reason.
    """
    try:
        return strip_comments(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return ""


def entry(path: Path | str, key: str) -> str | None:
    """One value out of one file, or None -- absent and unreadable read the same.

    They read the same because the index records the same thing for both: null. The
    harvest counts the files it could not read separately, so a corpus that
    silently halved is visible in the stamp rather than only in the rows.
    """
    return entries(read(path)).get(key)


# -- the vendor tier ---------------------------------------------------------------


SCHEMA_VERSION = 3
"""Bumped when anything about the shape on disk changes -- a row's fields or the
stamp's. `search.py` rebuilds rather than reads an index whose schema it does not
recognise, for the same reason it rebuilds on a version mismatch: a stale index is
worse than no index, because it looks authoritative.

Version 2 added `notes` to the earned row, for `search.py failure` to match on. The
bump is the point of the mechanism and forgetting it defeats the mechanism entirely:
with the number left at 1, an index built before the field was added stayed on disk,
passed the staleness check, and answered a real `failure` query with nothing at all --
because those rows had no `notes` and there was no way to tell.

Version 3 added `tutorials` and `work` to the stamp, so a switched tree is detectable.
An index stamped 2 has neither, and its tree therefore cannot be checked at all --
which is the reason to make it rebuild once rather than to special-case it forever.

`tests/test_toolbox_search.py` pins both shapes against this number so the next change
has to move it."""

CORPUS_DIR = "/work/.toolbox/corpus"
TUTORIALS_INDEX = "tutorials.index.jsonl"
STUDIES_INDEX = "studies.index.jsonl"
STAMP = "corpus.stamp.json"
RETRIEVALS = "retrievals.jsonl"

MESHERS = frozenset(
    {
        "snappyHexMesh",
        "blockMesh",
        "foamyHexMesh",
        "foamyQuadMesh",
        "cartesianMesh",
        "tetMesh",
        "extrudeMesh",
        "refineMesh",
    }
)
"""Applications that build a mesh rather than solve on one.

Only `snappyHexMesh` is actually observed in the `application` field of v2512 -- in 18
cases -- and the rest are here because they are OpenFOAM executables that could
legitimately appear there. A mesher in the solver field would make those cases
retrievable as seeds for a solve they cannot perform, which is the whole reason
`runs` exists."""

STEADY_DDT = frozenset({"steadyState", "none"})
"""`steadyState` is 165 of the tree's `fvSchemes`. `none` -- 43 more -- drops the time
derivative from the equation altogether, which is `potentialFoam`'s way of being
steady; it is read here rather than inferred, and it is not the same as absent."""

TRANSIENT_DDT = frozenset({"Euler", "backward", "CrankNicolson", "localEuler", "bounded"})
"""294 Euler, 18 backward, 9 localEuler, 3 CrankNicolson. A scheme on neither list
reads as null rather than as transient: a scheme shipped after this was written is
something not known about, not something known to evolve in time."""

GEOMETRY_SUFFIXES = frozenset({".stl", ".obj", ".ply", ".vtk", ".stlb", ".gz"})

NO_TURBULENCE: dict[str, str | None] = {"simulation_type": None, "model": None}
NO_REGIME: dict[str, object] = {
    "class": None,
    "compressible": None,
    "steady": None,
    "Re": None,
    "Ma": None,
    "shedding_risk": None,
}
"""What a row says when there was nothing to read. The shape stays the same whether
or not anything was found, so `search.py` never has to ask whether a field is a dict
or a null before it can rank on it -- and a null inside the shape is a stated fact,
where a missing field would be a silence."""

TIME_DIRS = ("0.orig", "0")
"""In that order. 369 cases keep their initial conditions in `0.orig/` and 122 in `0/`,
and no case in the shipped tree has both -- so the preference never has to choose there.
It matters for a case that has been copied out and run once, where `0/` is then the
solver's output and `0.orig/` is still the specification."""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def foam_version() -> str:
    """`$WM_PROJECT_VERSION`, or `unknown`. Never guessed from anything else."""
    return os.environ.get("WM_PROJECT_VERSION") or "unknown"


def foam_tutorials() -> Path | None:
    """`$FOAM_TUTORIALS` as a path, or None when it is not set.

    None and not `Path("")`, which is the whole point. `Path("")` normalises to
    `Path(".")` -- the current directory -- and a `Path` defines no `__bool__`, so it is
    truthy and `is_dir()` is True. Used as an argparse default it made the guard that
    checks for a tutorial tree unreachable, and `build` with the variable unset walked
    the working directory, indexed whatever `system/controlDict` was under it, and
    reported success. An index of the wrong tree that says `indexed 1` is worse than a
    refusal, because nothing downstream can tell.
    """
    value = os.environ.get("FOAM_TUTORIALS", "").strip()
    return Path(value) if value else None


def foam_fork(tree: Path | str) -> str:
    """Which fork's tree this is, decided by the filename turbulence lives under.

    ESI keeps it in `turbulenceProperties`; the Foundation fork uses
    `momentumTransport`. v2512 has zero `momentumTransport` files at any depth, so the
    question is settled by whichever appears -- and by `unknown` when neither does,
    rather than by assuming the fork this was written on.
    """
    for _ in Path(tree).rglob("constant/momentumTransport"):
        return "foundation"
    for _ in Path(tree).rglob("constant/turbulenceProperties"):
        return "esi"
    return "unknown"


def turbulence_of(directory: Path) -> dict[str, str | None]:
    """`{simulation_type, model}` from whichever fork's file is in `directory`.

    The model key is the one divergence a single schema has to absorb: ESI writes
    `RASModel` / `LESModel` inside a block named for the simulation type, and the
    Foundation fork writes `model`. Both spellings are tried, in the order the
    simulation type implies, so neither fork indexes as null on the other's tree.

    A `laminar` file names no model at all -- 179 cases -- and comes back with the
    type set and the model null, which is a different fact from both being null.
    """
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
    """Per-region turbulence, for the cases that keep it per region.

    None of the 18 multi-region cases carries a case-level `turbulenceProperties`, and
    16 of them keep one under `constant/<region>/`. Read only at the case level, every
    one of them indexes with turbulence nulled out -- and "nulled out" would then mean
    two different things in one index: the tutorial does not say, and nobody looked
    where it says it.

    The regions are discovered by looking for the file rather than by parsing
    `regionProperties`, whose list syntax this reader does not pretend to handle. A
    directory under `constant/` with no turbulence file in it is not reported as a
    region, so `polyMesh/` and `triSurface/` do not become ones.
    """
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
    """How this case's mesh is built.

    snappy is checked *before* blockMesh, and the order is the whole point: 62 of the
    64 `snappyHexMeshDict` cases also carry a `blockMeshDict`, because snappy cuts its
    mesh out of a background block that blockMesh builds. Checking blockMesh first --
    which is how the design document's table reads -- labels 97% of the snappy tier
    `blockMesh`, and a query for a snappy precedent then finds two cases out of
    sixty-four.
    """
    system = case / "system"
    if (system / "snappyHexMeshDict").exists():
        return "snappyHexMesh"
    if (system / "blockMeshDict").exists():
        return "blockMesh"
    if has_geometry(case):
        return "surface"
    return "unknown"


def steady_of(case: Path) -> bool | None:
    """Steady or transient, read off `ddtSchemes` rather than guessed from the solver.

    Scoped to the block. `fvSchemes` carries a `default` in `gradSchemes`,
    `divSchemes`, `laplacianSchemes` and more, so a flat read returns whichever comes
    first in the file -- and a case whose `gradSchemes` precedes its `ddtSchemes` would
    be classified by `Gauss linear`. Five files in the tree have no `ddtSchemes` at
    all, and they come back null.
    """
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
    """Which properties file the case carries, and for one of them what it says.

    `thermophysicalProperties` (118 cases) and `transportProperties` (333) are
    unambiguous on both forks and settle it by name alone.

    `physicalProperties` does not, and taking its presence as proof of
    incompressibility was wrong: on the Foundation fork that one filename **replaced
    both** of the others, so it is exactly as likely to hold a compressible
    thermophysical model. That reading stated a fact nobody had read, which is the one
    thing this module is not allowed to do. So the file is opened: a `thermoType` in it
    is a thermophysical model, a `nu` or a `viscosityModel` is the incompressible form,
    and anything else is null rather than a coin toss.
    """
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
    """The regime, carrying only what was measured.

    The design document's example class is `internal-incompressible-steady`, and two of
    those three are readable off the case: compressibility from which properties file
    is present, and steadiness from `ddtSchemes`. **Internal versus external is not**,
    not from any file a tutorial reliably has -- patch names hint at it and hints are
    guesses -- so that half is left out of the string rather than invented into it. A
    class here is a join of the parts that were read, and null when none were.

    `Re`, `Ma` and `shedding_risk` stay null for the vendor tier. Deriving them needs a
    velocity scale, a length scale and a viscosity that a tutorial does not reliably
    state, and a corpus with no benchmark tier under it cannot afford a number that
    looks measured and was not.
    """
    steady = steady_of(case)
    compressible = compressible_of(case)
    parts = []
    if compressible is not None:
        parts.append("compressible" if compressible else "incompressible")
    if steady is not None:
        parts.append("steady" if steady else "transient")
    return {
        "class": "-".join(parts) or None,
        "compressible": compressible,
        "steady": steady,
        "Re": None,
        "Ma": None,
        "shedding_risk": None,
    }


def bc_map_of(case: Path) -> dict[str, dict[str, str]]:
    """Every field in the case's initial-condition directory, as patch -> BC type.

    The first directory in `TIME_DIRS` that exists is the one read, even if it turns
    out to hold nothing -- preferring `0.orig/` means preferring it, not falling
    through to the solver's output when the specification is thin. 65 cases have
    neither directory, because their `Allrun` generates the fields, and those come
    back empty.
    """
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

    Symlinks appear twice in this tree and mean two different things, so both are
    handled here rather than left to the walker's defaults.

    A symlinked *file* is a case. 7 of the 556 are relative symlinks into a shared
    `common/` directory (`basic/laplacianFoam/multiWorld2/*/system/controlDict`), all
    resolving, and each names a real and separate case. `rglob` matches them and
    `read_text` follows them, so they cost nothing -- unless something actively
    excludes links, which `find -type f` does, losing those 7 and reporting nothing.

    A symlinked *directory* is not a case. Exactly one exists:
    `mesh/foamyHexMesh/straightDuctImplicit` points at
    `incompressible/porousSimpleFoam/straightDuctImplicit`, so the same case is
    reachable by two paths. Python 3.12's `rglob` happens not to descend into it, which
    is why the tree counts 556 and not 557 -- but that is a default, and 3.13 moved it
    behind `recurse_symlinks`. Resolving before de-duplicating settles it either way:
    one directory is one case, however many names it answers to. Indexing it twice
    would put one precedent in the corpus twice, and then the value distribution
    §5.4 reports would count it twice as well.
    """
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
    """One vendor row, or None when this is not a case that can be read.

    None means skipped, and the only thing that earns it is a `controlDict` from which
    no entry at all could be read -- a binary file, a dangling link, a truncated write.
    A `controlDict` that reads but names no `application` is still a case: the four
    `foamyHexMesh` and `foamyQuadMesh` tutorials are exactly that, and they are real
    meshing precedents. So `runs: false` has two causes, an `application` that is a
    mesher and no `application` to read, and neither is a reason to lose the row.
    """
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
    """Every case under `tree`, and a count of what was indexed and what was skipped.

    Nothing raises. A case that cannot be read is counted and passed over, the way
    `study_state.artifacts` skips a manifest line that will not parse: 556 of these are
    opened in one pass, and one bad file must not take the index down. The counts are
    what make a corpus that silently halved visible -- an index of 300 rows looks
    perfectly healthy until something says 256 were skipped.
    """
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
"""What a rung row carries out of the manifest. Always all six, so a rung recorded
without a measured value is a row with `value: null` rather than a row with a
different shape."""


def phase_table(study: Path) -> dict[str, object] | None:
    """The study's phase table as it is written, or None when there is none.

    Deliberately not `study_state.load_phases`, which is right for its own callers and
    wrong here: it returns a blank all-`pending` table when the file is missing, so a
    study that recorded no phases at all would be indistinguishable from one that
    recorded getting nowhere. Those are different facts and the index keeps them apart.
    """
    path = study / study_state.STATE_DIR / study_state.PHASES_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def verdict_of(table: dict[str, object] | None) -> str | None:
    """How far the study got, read off the phase table rather than invented.

    The order is the point. `failed` outranks everything: a study that meshed, solved
    and rendered and then failed its report is not one to seed from without knowing
    that. `pending` outranks `done` for the opposite reason -- a session that ended
    mid-study leaves the later phases pending, and that is neither a failure nor
    something finished.

    `skipped` does not count against completion. `study_state.py` records a phase as
    `skipped` rather than leaving it `pending` precisely so that "not done" and "not
    wanted" do not read the same, and a mesh-only study that stops after `checkMesh`
    is complete. But a table where *everything* is skipped is not a completed study,
    so `completed` also requires that something was actually done.

    A status this was not written against reads `unrecognised`. Picking the nearest
    known verdict would be inventing one, and the verdict is the only field an earned
    row has that a vendor row does not -- it carries the whole weight of R2.
    """
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
    """Which case represents this study.

    A study holds a primary case and often a copy of it per attempt --
    `runs/01-coarse`, `runs/02-medium` -- so the directory listing has several answers
    and the record has one: `phases.json` carries the case, because `set_phase` is
    given it. The record wins when it points at a case that is actually there; a table
    naming a case since deleted or renamed is a stale record, and turning it into a
    path in the index would put a path in there that nothing can open.

    With no record and exactly one case, there is no guess to make. With no record and
    several, the case fields stay null rather than being filled from whichever sorted
    first -- and `cases` lists what was found, so the null is explicable rather than
    just empty.
    """
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
    """The manifest counted by kind.

    Read through `study_state.artifacts`, which skips a line that will not parse --
    a manifest is appended to by several scripts and a job killed mid-write leaves
    half a line behind. Inheriting that behaviour is better than re-deciding it here
    and disagreeing with the writer.

    Rung rows are manifest lines too and are left out of this count: they are
    evidence rather than artifacts, and they are reported in full under `rungs`.
    """
    counted: dict[str, int] = {}
    for row in study_state.artifacts(root=study, exists=False):
        kind = str(row.get("kind") or "")
        if not kind or kind == study_state.RUNG_KIND:
            continue
        counted[kind] = counted.get(kind, 0) + 1
    return dict(sorted(counted.items()))


def rungs_of(study: Path) -> list[dict[str, object]]:
    """Recorded ladder rungs: the measured value and the known answer beside it.

    This is the one thing in the corpus that was checked against something outside the
    corpus. A rung's answer never comes from another solve -- it comes from
    Archimedes, the ITTC-57 line, Kelvin's 19.47 degrees, Hagen-Poiseuille -- so a
    recorded rung is a claim with an external referent, which no other earned field is.

    It is not a benchmark tier and it is not a substitute for one: it says a reduced
    case reproduced a known answer, not that the study's own result is right. Carrying
    it is what makes the difference legible instead of collapsing it into the verdict.
    """
    found = []
    for row in study_state.rung_evidence(root=study):
        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        found.append({field: meta.get(field) for field in RUNG_FIELDS})
    return found


def notes_of(study: Path, table: dict[str, object] | None) -> list[str]:
    """The free text a study left behind: phase notes, artifact labels, rung notes.

    Without this an earned row has no words in it, and `search.py failure` -- "find me
    a study that hit this" -- has nothing to match on but a one-word verdict. The
    sentence that actually identifies a failure is written by hand into a phase note
    (`set_phase("solve", "failed", note="diverged at t=0.31")`) or carried on a rung,
    and that is the text a later session is trying to find.

    Labels are included whole rather than parsed. They were written for a person to
    read and a query is a person asking; taking them apart would be guessing at a
    structure nobody promised.
    """
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
    """Every study under `work`: a directory carrying a `.reynolds/`.

    One level down, because that is where studies are -- `/work/<study-id>/`. `/work`
    also holds whatever else the agent put there, geometry and scratch directories and
    unpacked archives, and a loose case sitting in it is not a study. Carrying
    `.reynolds/` is what makes a directory one, which is the same test
    `study_state.find_root` applies when it walks up looking for one.
    """
    root = Path(work)
    try:
        children = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        return []
    return [child for child in children if (child / study_state.STATE_DIR).is_dir()]


def harvest_study(study: Path, *, of_version: str, of_fork: str) -> dict[str, object] | None:
    """One earned row.

    The same fields a vendor row carries, so a query ranks both on the same things,
    plus the five an earned row has of its own: the study it came from, which case
    represents it, what else was in it, what it produced, and what its rungs showed.

    **R2 lives here.** `tier` is `earned` and stays that way: an earned row is never
    promoted, merged into, or presented as equivalent to another tier. It carries a
    verdict, which no vendor row does, and it carries no reference value, which no
    vendor row does either -- a study's own result is not a reference, it is the thing
    a reference would check.
    """
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
    """Every study under `work`, and what was indexed and skipped.

    A `/work` that is not there is an empty index and not an error: the first session
    on a fresh instance has no studies yet, and asking for the earned tier then is a
    reasonable thing to do.
    """
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


def _resolved(path: Path | str) -> str:
    """An absolute path for the stamp, so two spellings of one tree compare equal."""
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def write_index(path: Path, rows: list[dict[str, object]]) -> Path:
    """One JSON object per line, written through a temporary file and moved into place.

    Replaced rather than appended: the index is a rebuild artifact and not a log. An
    append would keep a case that has been deleted upstream in the index forever, and a
    query would then hand back a path that is not there.
    """
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

    Two indexes, two files. **R2 at the level of the filesystem**: keeping the tiers in
    separate files means "never merged" is a property of the corpus rather than a
    promise the reader has to keep. A reader that wants both asks for both and is told
    which is which; there is no arrangement of these files in which a vendor row and an
    earned row are indistinguishable.

    The stamp is what makes the index answerable-about: the version and fork it was
    built against, when, how many rows in each tier, and how many were passed over.
    `search.py` reads it to decide whether what is on disk is worth reading at all.
    """
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
        # Which trees this was built from. Without them a switched `--tutorials`, or a
        # `$FOAM_TUTORIALS` corrected mid-session, is invisible to `search.py`'s
        # staleness check: the stamp still matches on version and schema, so the index
        # built from the wrong tree goes on answering. The tree was the one thing the
        # stamp described nothing about.
        "tutorials": _resolved(tree),
        "work": _resolved(work),
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
    building.add_argument(
        "--tutorials",
        type=Path,
        default=foam_tutorials(),
        help="The tree to walk (default: $FOAM_TUTORIALS).",
    )
    building.add_argument(
        "--work",
        type=Path,
        default=Path(WORK_ROOT),
        help=f"Where the studies are (default: {WORK_ROOT}).",
    )
    building.add_argument(
        "--out", type=Path, default=Path(CORPUS_DIR), help=f"Where to write (default: {CORPUS_DIR})."
    )
    building.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)

    if args.command == "build":
        if args.tutorials is None or not args.tutorials.is_dir():
            print(
                f"no tutorial tree at {args.tutorials or '$FOAM_TUTORIALS (unset)'}"
                " -- pass --tutorials, or source the OpenFOAM environment",
                file=sys.stderr,
            )
            return 1
        stamp = build(args.tutorials, args.out, work=args.work)
        if args.json:
            print(json.dumps(stamp, indent=2, sort_keys=True))
            return 0
        for index, tier in ((TUTORIALS_INDEX, "tutorials"), (STUDIES_INDEX, "studies")):
            counts = stamp["counts"][tier]
            print(f"{args.out / index}")
            print(f"  indexed  {counts['indexed']}")
            print(f"  skipped  {counts['skipped']}")
        print(f"built against {stamp['of_fork']} {stamp['of_version']}")
        return 0

    text = read(args.path)
    if not text:
        # Not an error: "this file says nothing I can read" is an answer, and the
        # harvest treats it as one. The exit code says which happened.
        print(f"nothing read from {args.path}", file=sys.stderr)
        return 1

    if args.command == "read":
        found = entries(text)
        if args.key:
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
            return 0
        width = max((len(key) for key in found), default=0)
        for key in sorted(found):
            print(f"{key:<{width}}  {found[key]}")
        return 0

    if args.command == "boundaries":
        patches = boundary_types(text)
        if args.json:
            print(json.dumps(patches, indent=2, sort_keys=True))
            return 0
        if not patches:
            print(f"no boundaryField in {args.path}")
            return 0
        width = max(len(name) for name in patches)
        for name, kind in patches.items():
            print(f"{name:<{width}}  {kind}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
