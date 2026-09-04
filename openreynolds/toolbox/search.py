#!/usr/bin/env python3
"""Ask the corpus for a precedent: a seed case, a scheme's usual value, a past failure.

`corpus.py` builds two indexes -- the tutorials the vendor shipped, and the studies
this instance has finished. This asks them questions.

    python3 search.py regime  "internal incompressible steady"   # ranked seed cases
    python3 search.py keyword "div(phi,U)"                       # what cases set it to
    python3 search.py failure "trailing-edge collapse"           # studies that hit it
    python3 search.py took /path/to/the/case/you/used            # say what you adopted
    python3 search.py log                                        # what has been asked
    python3 search.py --json regime "RAS kEpsilon"               # for another script

No embeddings and no vector store. Plain token matching, where an exact match on an
indexed field outranks the same word appearing somewhere in a path. That is a weaker
retriever than an embedding, and it is deliberate for a reason that outlives the
convenience: **every hit prints why it matched.** The corpus has no benchmark tier
under it, so nothing here can demonstrate that a ranking is *right*; the most it can
be is inspectable, and an unexplainable score is exactly where a closed loop hides.

Three things every answer carries.

**The tier of every hit, always.** A tutorial is a demonstration of a feature. A past
study is this system's own output. Neither is evidence that the other is correct, and
a result set that does not say which is which is how the two become one thing.

**Why it matched**, per hit, as the fields that matched and the free text that did.

**How concentrated a distribution is.** `keyword` answers "what do cases set this to",
and the honest version of that answer says who is holding the majority. 89 of the 557
tutorial cases are one solver from one tutorial family; an unweighted count reports
that family's house style as the consensus of the corpus, and there is no benchmark
tier to contradict it. The count and the family spread are printed together.

**What was passed over.** Every query is appended to `retrievals.jsonl` with its whole
ranked list, not just its winner, because the failure worth catching is specific: a
vendor hit and an earned hit answer the same query, they disagree, and the earned one
is adopted. That is the system preferring its own previous output, and a log of winners
cannot show it. `took` records what was actually used, against the retrieval that
offered it; nothing is inferred, and a query that returned exactly one case is not a
case adopted.

The index is rebuilt when either of its files is missing, when it was built from a
different tutorial tree, when it was built against a different OpenFOAM, or when its
schema is older than this code -- because a stale index is worse than no index: it
answers, and it answers about something else. Building needs a tutorial tree and says
so rather than harvesting the working directory; asking an index that is already there
needs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402  (sibling script, not a package)

FIELD_SCORE = 3
TEXT_SCORE = 1
"""An exact match on an indexed field beats the same word turning up in a path.

`incompressible/simpleFoam/pitzDaily` contains the word `simpleFoam`, and so does the
`application` entry of a case that actually runs it. The second is an answer to
"simpleFoam" and the first is a coincidence of filing. Three-to-one rather than
something finer because the numbers are a stated rule and not a tuned parameter: two
field matches beat one, and no amount of free text adds up to a field."""

DEFAULT_LIMIT = 10


# -- reading what was built ----------------------------------------------------


def read_stamp(out_dir: Path | str) -> dict | None:
    try:
        return json.loads((Path(out_dir) / corpus.STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def staleness(stamp: dict | None, live_version: str, *, tree: Path | str | None = None) -> str | None:
    """Why the index should be rebuilt, or None if it should not.

    The **tree is checked before the version**, and the order is the fix rather than a
    detail. Not knowing the live version returns None below, so with
    `$WM_PROJECT_VERSION` unset an index built from the wrong tree used to pass every
    check the stamp could make: an agent that indexed the working directory by mistake,
    then corrected its environment, went on being answered about the wrong corpus for
    the rest of the session. Checking the recorded tree first closes that and the
    switched-`--tutorials` case together.

    An old stamp records no tree. That is something not known rather than a mismatch,
    and treating it as one would rebuild on every query -- so it is passed over, and
    the schema bump is what gets those indexes rebuilt once instead.

    Not knowing the live version is likewise not a reason. `$WM_PROJECT_VERSION` unset
    means the harness cannot tell, and rebuilding on that would rebuild forever -- each
    rebuild producing an index stamped `unknown`, stale by the same rule that triggered
    it. An index stamped `unknown` when the version *is* now known is different:
    something was built blind and can now be built properly.
    """
    if stamp is None:
        return "no index has been built yet"
    if stamp.get("schema_version") != corpus.SCHEMA_VERSION:
        return (f"index schema {stamp.get('schema_version')}, "
                f"this reads {corpus.SCHEMA_VERSION}")
    recorded_tree = stamp.get("tutorials")
    if tree is not None and recorded_tree:
        try:
            wanted = Path(tree).resolve()
        except OSError:
            wanted = Path(tree)
        if not _same_path(recorded_tree, wanted):
            return f"index was built from another tree ({recorded_tree}); this is {wanted}"
    if live_version == "unknown":
        return None
    recorded = str(stamp.get("of_version") or "unknown")
    if recorded == "unknown":
        return f"index was built without a version; this is {live_version}"
    if recorded != live_version:
        return f"index was built against {recorded}; this is {live_version}"
    return None


def rebuild_reason(
    out_dir: Path | str, *, tree: Path | str | None = None, live_version: str | None = None
) -> str | None:
    """Everything that makes an index unusable, not only what the stamp says.

    The stamp is not the index. `staleness` reads `corpus.stamp.json` and nothing else,
    and `load_rows` passes over a file it cannot open -- so deleting one `.jsonl` and
    leaving a valid stamp made every query answer "nothing matched", with exit 0 and no
    rebuild, for as long as it took somebody to notice. The docstring of this module
    said the index is rebuilt when it is missing, and it was not.
    """
    version = corpus.foam_version() if live_version is None else live_version
    reason = staleness(read_stamp(out_dir), version, tree=tree)
    if reason:
        return reason
    missing = [name for name in (corpus.TUTORIALS_INDEX, corpus.STUDIES_INDEX)
               if not (Path(out_dir) / name).exists()]
    if missing:
        return f"the index is incomplete: {' and '.join(missing)} not there"
    return None


def load_rows(out_dir: Path | str) -> list[dict]:
    """Both indexes, in one list. Every row already carries its own tier."""
    rows: list[dict] = []
    for name in (corpus.TUTORIALS_INDEX, corpus.STUDIES_INDEX):
        try:
            text = (Path(out_dir) / name).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a half-written line costs its row, not the query
            if isinstance(row, dict):
                rows.append(row)
    return rows


def ensure(
    out_dir: Path | str,
    *,
    tutorials: Path | str,
    work: Path | str = corpus.WORK_ROOT,
    rebuild: bool = False,
) -> tuple[list[dict], str | None]:
    """The rows, building them first if they are missing or stale.

    Returns the reason it rebuilt, or None. The reason is printed rather than kept
    quiet: a query that silently took a second longer than usual, on an index that
    silently changed underneath it, is a thing the reader should be told about.
    """
    reason = "asked for" if rebuild else rebuild_reason(out_dir, tree=tutorials)
    if reason:
        corpus.build(tutorials, out_dir, work=work)
    return load_rows(out_dir), reason


# -- what a query can match ----------------------------------------------------


def fields_of(row: dict) -> list[tuple[str, str]]:
    """The values a token can match exactly, as (what it is, what it says).

    The regime class is offered whole *and* in parts, because the design document's
    own example query is "internal incompressible steady" -- three words against a
    field that holds `incompressible-steady`. Matching only the whole string would
    make the documented query return nothing.
    """
    pairs: list[tuple[str, str]] = [("tier", str(row.get("tier") or ""))]
    solver = row.get("solver") or {}
    turbulence = row.get("turbulence") or {}
    regime = row.get("regime") or {}
    for name, value in (
        ("solver", solver.get("executable")),
        ("turbulence", turbulence.get("simulation_type")),
        ("model", turbulence.get("model")),
        ("mesh", row.get("mesh_type")),
        ("verdict", row.get("verdict")),
        ("regime", regime.get("class")),
    ):
        if value:
            pairs.append((name, str(value)))
    for part in str(regime.get("class") or "").split("-"):
        if part:
            pairs.append(("regime", part))
    for region in (row.get("regions") or {}).values():
        for value in (region.get("simulation_type"), region.get("model")):
            if value:
                pairs.append(("region", str(value)))
    # De-duplicated, and this is load-bearing rather than tidiness. The class is
    # offered whole and in parts, so a class that is a single word -- `steady`, which
    # 21 rows carry because their properties file is missing and compressibility is
    # therefore null -- yields `regime=steady` twice. Scored twice, it is worth six
    # points for one concept, and a case matching only `steady` as a field outranks a
    # case matching both `incompressible` and `steady` as fields. Which is backwards,
    # and is what running the design document's own query against the real tree showed.
    return list(dict.fromkeys(pairs))


def text_of(row: dict) -> str:
    """Everything else a token may appear in: the path, and whatever words the study
    left behind. Bounded on purpose -- the boundary map and the scheme set are not in
    here, because a free-text hit on `zeroGradient` would match most of the corpus."""
    parts = [str(row.get("path") or "")]
    parts += [str(note) for note in row.get("notes") or []]
    parts += [str(kind) for kind in row.get("artifacts") or {}]
    for rung in row.get("rungs") or []:
        parts += [str(rung.get(field) or "") for field in ("name", "status", "known")]
    return " ".join(parts)


def score(row: dict, tokens: list[str]) -> tuple[int, list[str]]:
    """How well this row answers, and the reason, in that order of importance."""
    total = 0
    why: list[str] = []
    fields = fields_of(row)
    text = text_of(row).lower()
    for token in tokens:
        lowered = token.lower()
        matched = [f"{name}={value}" for name, value in fields if value.lower() == lowered]
        if matched:
            total += FIELD_SCORE * len(matched)
            why += matched
            continue
        if lowered in text:
            total += TEXT_SCORE
            why.append(f"text~{token}")
    return total, list(dict.fromkeys(why))


def unmatched_tokens(rows: list[dict], text: str, *, tier: str | None = None) -> list[str]:
    """The words in the query that matched nothing anywhere in the corpus.

    "internal incompressible steady" is the query the design document writes, and
    `internal` matches nothing: internal versus external is not derivable from a
    tutorial, so it was never indexed. Two thirds of that query scored and the reader
    was not told which third did not. A retriever with no benchmark tier behind it can
    at least be clear about what it did not understand.

    `tier` has to be the same tier the search was held to, and leaving it out was a
    bug: a `failure` query searches only the earned tier, so `failure "pitzDaily"` --
    a word that appears solely in a vendor path -- found nothing and reported nothing
    unmatched, telling both the reader and the log that the query had been understood.
    """
    tokens = [token for token in text.split() if token]
    seen: set[str] = set()
    for row in rows:
        if tier and row.get("tier") != tier:
            continue
        values = {value.lower() for _, value in fields_of(row)}
        text_of_row = text_of(row).lower()
        for token in tokens:
            lowered = token.lower()
            if lowered in values or lowered in text_of_row:
                seen.add(lowered)
    return [token for token in tokens if token.lower() not in seen]


def hit(row: dict, points: int, why: list[str]) -> dict:
    """One result, in the shape the provenance log records -- including, when the time
    comes, the hits that were not taken."""
    return {"path": str(row.get("path") or ""), "tier": str(row.get("tier") or ""),
            "score": points, "why": why}


# -- the three questions -------------------------------------------------------


def search_with_held(
    rows: list[dict],
    text: str,
    *,
    tier: str | None = None,
    runnable_only: bool = False,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    """Ranked hits, and how many were held back for not being able to solve.

    Every hit by default, and the caller truncates. That split matters here: a
    two-word regime query ties enormously -- **146 vendor rows carry the class
    `incompressible-steady`** -- and the tie is broken by path, alphabetically. A
    function that returned ten of them would be handing back an arbitrary slice with
    nothing to say it was one, and `IO/fileHandler` would look like the best available
    seed for steady incompressible flow when all it is is first in the alphabet.
    Ranking and presenting are kept apart so the presenter can say how big the tie was.

    The held-back count is returned rather than swallowed for the same reason. 22 cases
    in the tree do not solve -- 18 name a mesher as their `application` and 4 name
    nothing -- and dropping them from a solver query is right, but dropping them
    silently is the search deciding something for the reader without saying it did.
    """
    tokens = [token for token in text.split() if token]
    scored: list[tuple[int, str, dict]] = []
    held = 0
    for row in rows:
        if tier and row.get("tier") != tier:
            continue
        points, why = score(row, tokens)
        if not points:
            continue
        if runnable_only and not row.get("runs"):
            held += 1
            continue
        scored.append((points, str(row.get("path") or ""), hit(row, points, why)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [item[2] for item in scored]
    return (ranked[:limit] if limit else ranked), held


def search(rows: list[dict], text: str, **kwargs) -> list[dict]:
    return search_with_held(rows, text, **kwargs)[0]


def failures(rows: list[dict], text: str, limit: int | None = None) -> list[dict]:
    """Studies that hit something like this before.

    The earned tier only. "Has this gone wrong before" is a question about this
    system's own history, and a tutorial cannot answer it: tutorials do not fail,
    they ship.
    """
    return search(rows, text, tier="earned", limit=limit)


def case_of(row: dict) -> Path:
    """The case directory a row points at. For a study that is the case it recorded;
    for a tutorial the row already is one."""
    path = Path(str(row.get("path") or ""))
    case = row.get("case")
    return path / str(case) if case else path


FAMILY_DEPTH = 2
"""How many directory levels below the corpus root name a family:
`incompressible/simpleFoam`, `incompressible/adjointOptimisationFoam`,
`heatTransfer/chtMultiRegionFoam`. That is how the tutorial tree is organised at the
top, whatever it does further down."""


def family_names(cases: list[Path], root: Path | str | None = None) -> dict[Path, str]:
    """Each case grouped by the tutorial family it belongs to.

    Measured **down from the tree root**, not up from the case, and that distinction
    was found the hard way. Taking the two directory levels immediately above each
    case looks equivalent and is not: the tree is not uniformly deep, and
    `adjointOptimisationFoam` nests its cases several levels further down than
    `simpleFoam` does. Grouping upwards reported the 69 cases of one adjoint key as
    **47 families**, named things like `1_Inlet_2_Outlet/levelSet` -- so the proxy
    fell apart on the single family the whole concentration warning exists for, and
    fell apart in the direction that hides the problem.

    `root` is the tree the cases came from, and the caller knows it: the query was
    given a `--tutorials` to build from. Deriving it instead from the common ancestor
    of the cases is the fallback and not the default, because it is wrong exactly when
    the answer matters -- a set of cases that all sit under `incompressible/` has that
    directory as its common ancestor, and the two levels below it are then case names
    rather than family names.

    A case that is not under `root` falls back with the rest of its group. A group with
    no common ancestor at all -- different drives on Windows -- keeps its own names
    rather than raising.
    """
    if not cases:
        return {}
    base = Path(root) if root is not None else None
    if base is None or not all(_under(case, base) for case in cases):
        try:
            base = Path(os.path.commonpath([str(case) for case in cases]))
        except ValueError:
            base = None
    names: dict[Path, str] = {}
    for case in cases:
        parts: tuple[str, ...] = ()
        if base is not None and _under(case, base):
            parts = case.relative_to(base).parts
        names[case] = "/".join(parts[:FAMILY_DEPTH]) if parts else case.name
    return names


def _under(case: Path, base: Path) -> bool:
    try:
        case.relative_to(base)
        return True
    except ValueError:
        return False


def distribution(
    rows: list[dict],
    key: str,
    *,
    tier: str | None = None,
    limit: int | None = None,
    root: Path | str | None = None,
) -> dict:
    """What the corpus sets `key` to, with the counts and the family spread.

    Read out of the cases rather than out of the index. A case has hundreds of scheme
    and solver keys and indexing them all would swamp the row; the index says which
    cases exist and this reads them, which also means the answer is current with what
    is on disk rather than with what was true when the index was built.

    Only the dictionaries directly under `system/` and `constant/` are read. That is
    where schemes, solution settings and material properties live, and it keeps a
    query bounded at a handful of files per case rather than a walk of the whole tree.

    **The family spread is the part that matters.** A value held by 5 cases from one
    directory of siblings and a value held by 5 unrelated cases are different claims,
    and the counts alone cannot tell them apart. Both numbers are reported so the
    reader can see which they have.
    """
    values: dict[str, list[Path]] = {}
    scanned: dict[str, list[Path]] = {}
    for row in rows:
        if tier and row.get("tier") != tier:
            continue
        case = case_of(row)
        scanned.setdefault(str(row.get("tier") or ""), []).append(case)
        value = None
        for holder in ("system", "constant"):
            try:
                children = sorted(child for child in (case / holder).iterdir()
                                  if child.is_file())
            except OSError:
                continue
            for child in children:
                found = corpus.entries(corpus.read(child)).get(key)
                if found is not None:
                    value = found
                    break
            if value is not None:
                break
        if value is not None:
            values.setdefault(value, []).append(case)

    # Family names are derived from every case that was *scanned*, per tier, not from
    # the ones that happened to hold the key. A key only one family sets would
    # otherwise have its common root land inside that family, and the one answer the
    # concentration report exists to give -- "these are all siblings" -- would come
    # back as a dozen families with plausible-looking names.
    names: dict[Path, str] = {}
    for cases in scanned.values():
        names.update(family_names(cases, root))

    total = sum(len(cases) for cases in values.values())
    families: dict[str, int] = {}
    for cases in values.values():
        for case in cases:
            name = names.get(case, case.name)
            families[name] = families.get(name, 0) + 1
    largest = max(families.items(), key=lambda item: (item[1], item[0]), default=None)

    entries = [
        {
            "value": value,
            "count": len(cases),
            "families": len({names.get(case, case.name) for case in cases}),
            "paths": [str(case) for case in cases[:5]],
        }
        for value, cases in values.items()
    ]
    entries.sort(key=lambda entry: (-entry["count"], entry["value"]))
    return {
        "key": key,
        "total": total,
        "values": entries[:limit] if limit else entries,
        "families": {
            "total": len(families),
            "largest": None if largest is None else {
                "name": largest[0],
                "count": largest[1],
                "share": largest[1] / total if total else 0.0,
            },
        },
    }


# -- provenance ----------------------------------------------------------------


def record_retrieval(
    out_dir: Path | str,
    *,
    kind: str,
    text: str,
    hits: list[dict],
    matched: int,
    unmatched: list[str] | tuple[str, ...] = (),
    tier: str | None = None,
    values: list[dict] | None = None,
) -> str | None:
    """Write down one query and everything it returned. Returns the record's id.

    **The ranked list goes in whole, including every hit that was passed over**, and
    that is the entire reason this file exists. The failure it is meant to catch is
    specific: a hit from the vendor tier and a hit from the earned tier answer the
    same query, they disagree, and the earned one is adopted. That is the system
    preferring its own previous output to the vendor's, which is the closed loop
    tightening by a notch -- and it is completely invisible if only the winner is
    recorded, because the winner looks the same either way.

    `matched` is recorded beside the hits because 358 matched and 10 shown is a
    different retrieval from 10 matched and 10 shown, and later there is no way to
    tell them apart from the hits alone.

    Returns None rather than raising if the log cannot be written. Provenance is a
    record of the work and not a precondition for it: a read-only corpus directory is
    not a reason for a query to fail.
    """
    record = {
        "t": corpus.now_iso(),
        "id": f"{corpus.now_iso()}-{os.getpid()}-{_next_serial()}",
        "query": {"kind": kind, "text": text, "tier": tier,
                  "unmatched": list(unmatched)},
        "matched": matched,
        "hits": hits,
        "taken": None,
    }
    if values is not None:
        record["values"] = values
    return record["id"] if _append(out_dir, record) else None


def record_taken(out_dir: Path | str, *, path: str, of: str | None = None) -> bool:
    """Write down that something was adopted, against the retrieval it came from.

    Appended as its own line rather than written back into the earlier one. The
    retrieval happened and then the adoption happened; they are two events with two
    times, and a log that edits its own history to look tidier is worth less than one
    that does not.

    `of` defaults to **the most recent retrieval whose hits actually contained this
    path** -- a fact about what was on the screen, not a guess about what was recent.
    Attaching to whichever retrieval came last looks equivalent and is not: three
    queries were run for real, the case adopted came from the first, and the record
    hung it on the third. The log would then have said a `failure` query produced a
    tutorial it never returned.

    With no such retrieval, the adoption is recorded unattached. That is the more
    interesting record rather than a degenerate one: the agent used a case the corpus
    never handed it, and hanging that on an unrelated query would make the log claim a
    retrieval succeeded when it did not.
    """
    if of is None:
        for entry in reversed(_lines(out_dir)):
            if entry.get("kind") == "taken" or "query" not in entry:
                continue
            if any(_same_path(found.get("path"), path)
                   for found in entry.get("hits") or []):
                of = str(entry.get("id"))
                break
    return _append(out_dir, {"t": corpus.now_iso(), "kind": "taken", "of": of,
                             "path": path})


def read_retrievals(out_dir: Path | str) -> list[dict]:
    """The log, with each adoption folded onto the retrieval it belongs to.

    Folding is done on read rather than on write, so the file stays append-only and
    the folded view stays a view. An adoption with no retrieval to attach to is not
    invented into one; it simply does not appear here, and the raw line is still in
    the file.
    """
    retrievals: list[dict] = []
    by_id: dict[str, dict] = {}
    for entry in _lines(out_dir):
        if entry.get("kind") == "taken":
            owner = by_id.get(str(entry.get("of")))
            if owner is not None:
                owner["taken"] = entry.get("path")
            continue
        if "query" not in entry:
            continue
        retrievals.append(entry)
        by_id[str(entry.get("id"))] = entry
    return retrievals


def _same_path(one: object, other: object) -> bool:
    """Whether two written paths name the same case.

    Compared as paths and not as strings, which is a distinction that cost a real
    record: the hits carried `full\\incompressible\\simpleFoam\\motorBike` and the
    adoption arrived from a shell as `full/incompressible/simpleFoam/motorBike`. As
    strings those are two cases and the log reported that the corpus had never offered
    the one that was used. `normpath` also settles a `.` or a doubled-back `..` that a
    hand-typed path picks up.
    """
    if not one or not other:
        return False
    try:
        return Path(os.path.normpath(str(one))) == Path(os.path.normpath(str(other)))
    except (TypeError, ValueError):
        return False


_serial = 0


def _next_serial() -> int:
    """Distinguishes two records written inside the same second."""
    global _serial
    _serial += 1
    return _serial


LOCK_ATTEMPTS = 100
LOCK_WAIT_S = 0.005


def _append(out_dir: Path | str, record: dict) -> bool:
    """One record, one line, under a lock.

    The obvious implementation is `open(path, "a")` and one `write`, which is what this
    was and what `study_state.record` still does for the manifest, on the reasoning that
    concurrent appends interleave whole lines but never halves of one. **That reasoning
    is wrong, and it was inherited rather than measured.** Six processes appending 150
    records each produced 763 lines instead of 900, 23 of them cut mid-JSON: records
    lost outright.

    Rewriting it as a single `os.write` to an `O_APPEND` descriptor -- one syscall,
    which is atomic for a regular file on POSIX -- did not fix it either: 796 of 900.
    Append mode on Windows is a seek-to-end followed by a write with nothing joining
    them, so two processes can agree on the same offset and one overwrites the other.

    So a lock file, which needs no platform branch and no third-party anything: create
    it `O_EXCL`, append, remove it. Under contention past `LOCK_ATTEMPTS` the write goes
    ahead unlocked, because a record written into a possible race is better than a
    record dropped for certain -- and a crashed process that left its lock behind must
    not silence the log forever.

    A provenance log that quietly loses an eighth of its records is worth less than no
    log, for the same reason a stale index is worth less than no index: it answers, and
    the answer is incomplete in a way nothing on the surface reveals.
    """
    path = Path(out_dir) / corpus.RETRIEVALS
    try:
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    except (TypeError, ValueError):
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    lock = path.with_name(path.name + ".lock")
    held = False
    for _ in range(LOCK_ATTEMPTS):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            held = True
            break
        except FileExistsError:
            time.sleep(LOCK_WAIT_S)
        except OSError:
            break  # a directory that will not take a lock file still takes the record
    try:
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            os.write(handle, line)
        finally:
            os.close(handle)
        return True
    except OSError:
        return False
    finally:
        if held:
            try:
                os.unlink(lock)
            except OSError:
                pass


def _lines(out_dir: Path | str) -> list[dict]:
    try:
        text = (Path(out_dir) / corpus.RETRIEVALS).read_text(encoding="utf-8")
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a line truncated mid-write costs its line and no more
        if isinstance(entry, dict):
            found.append(entry)
    return found


def distributions(
    rows: list[dict],
    key: str,
    *,
    tier: str | None = None,
    limit: int | None = None,
    root: Path | str | None = None,
) -> list[dict]:
    """One distribution per tier, each labelled, never one table over both.

    **This is R2 at the one place it was not being kept.** `distribution` has always
    taken a `tier`, and the `keyword` command passed the command-line default of None
    -- so a tutorial and this instance's own finished study were counted into a single
    table, and the footer called all of them "tutorial families". A 50/50 split with
    nothing on screen saying that half of it was the system's own prior output is
    exactly the merge the whole design exists to forbid. The capability to hold a
    distribution to one tier existed and was defaulted off.

    Reported separately rather than restricted to the vendor tier, because "what have
    my own studies used" is a real question. It is just a different one, and the answer
    has to say which it is answering.
    """
    tiers = [tier] if tier else sorted({str(row.get("tier") or "") for row in rows},
                                       key=lambda name: (name != "vendor", name))
    found = []
    for name in tiers:
        if not name:
            continue
        entry = distribution(rows, key, tier=name, limit=limit, root=root)
        entry["tier"] = name
        found.append(entry)
    return found


# -- the command line ----------------------------------------------------------


def show_hits(hits: list[dict], held: int, *, runnable_only: bool, limit: int) -> None:
    shown = hits[:limit]
    if not shown:
        print("nothing matched")
    else:
        width = max(len(hit["tier"]) for hit in shown)
        for found in shown:
            print(f"{found['score']:>3}  {found['tier']:<{width}}  {found['path']}")
            print(f"     {', '.join(found['why'])}")
        if len(hits) > len(shown):
            tied = sum(1 for found in hits if found["score"] == shown[-1]["score"])
            print()
            print(f"{len(hits)} matched; showing {len(shown)}.")
            print(f"{tied} of them share the lowest score displayed, and ties break by "
                  "path rather than")
            print("by merit -- so those are a slice of that group and not the best of it. "
                  "Naming a")
            print("solver, a turbulence model or a mesh type ranks on more fields than a "
                  "regime alone.")
    if held and runnable_only:
        print(f"\n{held} mesh-only case(s) matched and are not shown; they cannot seed a "
              f"solve. --include-mesh-only to see them.")


def show_distributions(found: list[dict]) -> None:
    """Each tier's table, named. Two tiers are two answers, never one."""
    if not any(entry["total"] for entry in found):
        key = found[0]["key"] if found else ""
        print(f"nothing in the corpus sets {key}")
        return
    for index, entry in enumerate(found):
        if index:
            print()
        show_distribution(entry)


def show_distribution(found: dict) -> None:
    tier = found.get("tier")
    if not found["total"]:
        where = f" in the {tier} tier" if tier else " in the corpus"
        print(f"nothing{where} sets {found['key']}")
        return
    label = f"  [{tier} tier]" if tier else ""
    print(f"{found['key']}  --  {found['total']} case(s){label}")
    width = max(len(entry["value"]) for entry in found["values"])
    for entry in found["values"]:
        share = entry["count"] / found["total"]
        print(f"  {entry['value']:<{width}}  {entry['count']:>4}  {share:>5.0%}"
              f"   {entry['families']} {'family' if entry['families'] == 1 else 'families'}")
    largest = found["families"]["largest"]
    if largest:
        # "tutorial families" only if these are tutorials. An earned distribution is
        # groups of the instance's own studies, which is a different claim entirely.
        total = found["families"]["total"]
        if tier == "vendor":
            kind = "tutorial family" if total == 1 else "tutorial families"
        else:
            kind = "group" if total == 1 else "groups"
        print(f"\n{total} {kind} in all. The largest, "
              f"{largest['name']}, holds {largest['count']} of the {found['total']} "
              f"({largest['share']:.0%}).")
        if largest["share"] > 0.5:
            print("A majority of these cases are siblings in one directory, so this is "
                  "one group's house style rather than a corpus-wide consensus.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=Path(corpus.CORPUS_DIR),
                    help=f"Where the indexes are (default: {corpus.CORPUS_DIR}).")
    ap.add_argument("--tutorials", type=Path, default=corpus.foam_tutorials(),
                    help="The tree to harvest if a build is needed (default: $FOAM_TUTORIALS).")
    ap.add_argument("--work", type=Path, default=Path(corpus.WORK_ROOT),
                    help=f"Where the studies are (default: {corpus.WORK_ROOT}).")
    ap.add_argument("--rebuild", action="store_true", help="Harvest again before asking.")
    ap.add_argument("--tier", choices=["vendor", "earned"], default=None,
                    help="Hold the answer to one tier.")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--json", action="store_true", help="Machine-readable.")
    sub = ap.add_subparsers(dest="command", required=True)

    seeds = sub.add_parser("regime", help="Ranked cases to seed from.")
    seeds.add_argument("text")
    seeds.add_argument("--include-mesh-only", action="store_true",
                       help="Include cases that mesh but cannot solve.")

    values = sub.add_parser("keyword", help="What the corpus sets one key to.")
    values.add_argument("key")

    hit_failures = sub.add_parser("failure", help="Studies that hit something like this.")
    hit_failures.add_argument("text")

    adopted = sub.add_parser("took", help="Record that you used one of the last hits.")
    adopted.add_argument("path")

    listing = sub.add_parser("log", help="What has been retrieved, and what was taken.")
    listing.add_argument("--limit", type=int, default=20, dest="log_limit")

    args = ap.parse_args(argv)

    if args.command == "took":
        # No index needed, and no query run: this is the agent saying what it used.
        if not record_taken(args.corpus, path=args.path):
            print(f"could not write to {args.corpus / corpus.RETRIEVALS}", file=sys.stderr)
            return 1
        print(f"recorded: took {args.path}")
        return 0

    if args.command == "log":
        entries = read_retrievals(args.corpus)[-args.log_limit:]
        if args.json:
            print(json.dumps(entries, indent=2))
            return 0
        if not entries:
            print("nothing retrieved yet")
            return 0
        for entry in entries:
            query = entry.get("query", {})
            print(f"{entry.get('t')}  {query.get('kind')}  {query.get('text')!r}"
                  f"  ({entry.get('matched')} matched)")
            for found in entry.get("hits", []):
                mark = "->" if _same_path(found.get("path"), entry.get("taken")) else "  "
                print(f"   {mark} {found.get('score'):>3}  {found.get('tier'):<7} "
                      f"{found.get('path')}")
            for value in entry.get("values", [])[:5]:
                print(f"      {value.get('count'):>4}  {value.get('value')} "
                      f"({value.get('families')} "
                      f"{'family' if value.get('families') == 1 else 'families'})")
            if entry.get("taken") is None and entry.get("hits"):
                print("      (nothing recorded as taken)")
        unattached = [line for line in _lines(args.corpus)
                      if line.get("kind") == "taken" and line.get("of") is None]
        if unattached:
            # Worth surfacing rather than leaving in the file: a case was adopted that
            # no query in this log returned, so the corpus did not supply it.
            print(f"\n{len(unattached)} adoption(s) matched no retrieval in this log:")
            for line in unattached[-args.log_limit:]:
                print(f"   {line.get('t')}  {line.get('path')}")
        return 0
    needed = "asked for" if args.rebuild else rebuild_reason(args.corpus,
                                                             tree=args.tutorials)
    if needed and (args.tutorials is None or not args.tutorials.is_dir()):
        # The guard belongs on building, not on asking: an index already on disk and
        # not stale is answerable with no tree in sight, which is every query after
        # the first. Without it, `ensure` built a corpus out of the working directory.
        print(f"the index needs building ({needed}) and there is no tutorial tree at "
              f"{args.tutorials or '$FOAM_TUTORIALS (unset)'} -- pass --tutorials, or "
              "source the OpenFOAM environment", file=sys.stderr)
        return 1
    rows, reason = ensure(args.corpus, tutorials=args.tutorials, work=args.work,
                          rebuild=args.rebuild)
    if reason and not args.json:
        print(f"(rebuilt the index: {reason})\n")

    def say_unmatched(text: str, tier: str | None = None) -> None:
        missing = unmatched_tokens(rows, text, tier=tier)
        if missing and not args.json:
            print(f"(nothing in the corpus matches: {', '.join(missing)})\n")

    if args.command == "keyword":
        found = distributions(rows, args.key, tier=args.tier, root=args.tutorials)
        for entry in found:
            record_retrieval(args.corpus, kind="keyword", text=args.key, hits=[],
                             matched=entry["total"], tier=entry["tier"],
                             values=entry["values"])
        if args.json:
            print(json.dumps(found, indent=2))
        else:
            show_distributions(found)
        return 0

    if args.command == "failure":
        say_unmatched(args.text, tier="earned")
        hits = failures(rows, args.text)
        record_retrieval(args.corpus, kind="failure", text=args.text,
                         hits=hits[:args.limit], matched=len(hits), tier="earned",
                         unmatched=unmatched_tokens(rows, args.text, tier="earned"))
        if args.json:
            print(json.dumps(hits[:args.limit], indent=2))
        else:
            show_hits(hits, 0, runnable_only=False, limit=args.limit)
        return 0

    say_unmatched(args.text, tier=args.tier)
    runnable_only = not args.include_mesh_only
    hits, held = search_with_held(rows, args.text, tier=args.tier,
                                  runnable_only=runnable_only)
    record_retrieval(args.corpus, kind="regime", text=args.text,
                     hits=hits[:args.limit], matched=len(hits), tier=args.tier,
                     unmatched=unmatched_tokens(rows, args.text, tier=args.tier))
    if args.json:
        print(json.dumps(hits[:args.limit], indent=2))
    else:
        show_hits(hits, held, runnable_only=runnable_only, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
