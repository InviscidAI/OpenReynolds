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

Plain token matching, no embeddings, and every hit prints its tier and why it
matched: with no benchmark tier under the corpus a ranking cannot be shown right, so
it is kept inspectable. Every query goes to `retrievals.jsonl` with its whole ranked
list, so a vendor hit losing to an earned one stays visible; `took` records what was
used, and nothing is inferred. A stale index (`staleness`) is rebuilt before asking.
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
"""A field match beats a path match, and no amount of path text adds up to a field:
`incompressible/simpleFoam/pitzDaily` contains `simpleFoam` by filing, not by running
it."""

DEFAULT_LIMIT = 10


# -- reading what was built ----------------------------------------------------


def read_stamp(out_dir: Path | str) -> dict | None:
    try:
        return json.loads((Path(out_dir) / corpus.STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def staleness(
    stamp: dict | None,
    live_version: str,
    *,
    tree: Path | str | None = None,
    work: Path | str | None = None,
) -> str | None:
    """Why the index should be rebuilt, or None if it should not.

    The trees are checked before the version, because an unknown live version returns
    None and would otherwise wave through an index built from the wrong tree. A stamp
    that records no tree is passed over; the schema bump rebuilds those once. A study
    finishing moves none of schema, tree or version, which is what the `studies`
    fingerprint is for.
    """
    if stamp is None:
        return "no index has been built yet"
    if stamp.get("schema_version") != corpus.SCHEMA_VERSION:
        return (f"index schema {stamp.get('schema_version')}, "
                f"this reads {corpus.SCHEMA_VERSION}")
    recorded_tree = stamp.get("tutorials")
    if tree is not None and recorded_tree:
        wanted = corpus._resolved(tree)
        if not _same_path(recorded_tree, wanted):
            return f"index was built from another tree ({recorded_tree}); this is {wanted}"
    if work is not None:
        recorded_work = stamp.get("work")
        wanted_work = corpus._resolved(work)
        if recorded_work and not _same_path(recorded_work, wanted_work):
            return (f"index was built from other studies ({recorded_work}); "
                    f"this is {wanted_work}")
        recorded_studies = stamp.get("studies")
        if recorded_studies is not None and recorded_studies != corpus.studies_fingerprint(work):
            return "the studies have changed since the index was built"
    # Not knowing the live version is not a reason: rebuilding on it would rebuild
    # forever, each index stamped `unknown` and stale by the same rule.
    if live_version == "unknown":
        return None
    recorded = str(stamp.get("of_version") or "unknown")
    if recorded == "unknown":
        return f"index was built without a version; this is {live_version}"
    if recorded != live_version:
        return f"index was built against {recorded}; this is {live_version}"
    return None


def rebuild_reason(
    out_dir: Path | str,
    *,
    tree: Path | str | None = None,
    work: Path | str | None = None,
    live_version: str | None = None,
) -> str | None:
    """Everything that makes an index unusable: the stamp, and the files it describes
    actually being there."""
    version = corpus.foam_version() if live_version is None else live_version
    reason = staleness(read_stamp(out_dir), version, tree=tree, work=work)
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
            if not line.strip():
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
    """The rows, building them first if they are missing or stale, and why it built."""
    reason = "asked for" if rebuild else rebuild_reason(out_dir, tree=tutorials, work=work)
    if reason:
        corpus.build(tutorials, out_dir, work=work)
    return load_rows(out_dir), reason


# -- what a query can match ----------------------------------------------------


def fields_of(row: dict, *, earned: bool = False) -> list[tuple[str, str]]:
    """The values a token can match exactly, as (what it is, what it says).

    The regime class is offered whole and in parts, so "incompressible steady" matches
    `incompressible-steady`. `verdict` exists only on earned rows and is offered only
    when the search is held to that tier, so a study cannot win a tie on it.
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
        ("verdict", row.get("verdict") if earned else None),
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
    # De-duplicated: a one-word class (`steady`) is offered whole and as its only part,
    # and scoring it twice outranked a case matching two distinct fields.
    return list(dict.fromkeys(pairs))


def text_of(row: dict, *, earned: bool = False) -> str:
    """The path, and -- only when the search is held to the earned tier -- the words a
    study left behind. Notes, artifacts and rungs exist only on earned rows, so scoring
    them across both tiers ranked a study above an identical tutorial."""
    parts = [str(row.get("path") or "")]
    if earned:
        parts += [str(note) for note in row.get("notes") or []]
        parts += [str(kind) for kind in row.get("artifacts") or {}]
        for rung in row.get("rungs") or []:
            parts += [str(rung.get(field) or "") for field in ("name", "status", "known")]
    return " ".join(parts)


def score(row: dict, tokens: list[str], *, earned: bool = False) -> tuple[int, list[str]]:
    """How well this row answers, and the reason."""
    total = 0
    why: list[str] = []
    fields = fields_of(row, earned=earned)
    text = text_of(row, earned=earned).lower()
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
    """The query words that matched nothing in the tier that was searched."""
    tokens = [token for token in text.split() if token]
    seen: set[str] = set()
    earned = tier == "earned"
    for row in rows:
        if tier and row.get("tier") != tier:
            continue
        values = {value.lower() for _, value in fields_of(row, earned=earned)}
        text_of_row = text_of(row, earned=earned).lower()
        for token in tokens:
            lowered = token.lower()
            if lowered in values or lowered in text_of_row:
                seen.add(lowered)
    return [token for token in tokens if token.lower() not in seen]


def hit(row: dict, points: int, why: list[str]) -> dict:
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

    Every hit unless `limit` is given, so the presenter can say how big a tie it is
    showing a slice of: 146 vendor rows share `incompressible-steady`, and ties break
    by path, not merit.
    """
    tokens = [token for token in text.split() if token]
    earned = tier == "earned"
    scored: list[tuple[int, str, dict]] = []
    held = 0
    for row in rows:
        if tier and row.get("tier") != tier:
            continue
        points, why = score(row, tokens, earned=earned)
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
    """Studies that hit something like this before. The earned tier only: tutorials do
    not fail, they ship."""
    return search(rows, text, tier="earned", limit=limit)


def case_of(row: dict) -> Path:
    """The case directory a row points at: a study's recorded case, or the tutorial."""
    path = Path(str(row.get("path") or ""))
    case = row.get("case")
    return path / str(case) if case else path


def distribution(
    rows: list[dict], key: str, *, tier: str | None = None, limit: int | None = None
) -> dict:
    """What the corpus sets `key` to, with counts.

    Read out of the cases rather than the index -- a case has hundreds of keys -- and
    only from the files directly under `system/` and `constant/`, which keeps a query
    to a handful of files per case.
    """
    values: dict[str, list[Path]] = {}
    for row in rows:
        if tier and row.get("tier") != tier:
            continue
        case = case_of(row)
        value = None
        for holder in ("system", "constant"):
            try:
                children = sorted(child for child in (case / holder).iterdir()
                                  if child.is_file())
            except OSError:
                continue
            for child in children:
                value = corpus.entries(corpus.read(child)).get(key)
                if value is not None:
                    break
            if value is not None:
                break
        if value is not None:
            values.setdefault(value, []).append(case)
    entries = [{"value": value, "count": len(cases), "paths": [str(c) for c in cases[:5]]}
               for value, cases in values.items()]
    entries.sort(key=lambda entry: (-entry["count"], entry["value"]))
    return {"key": key, "total": sum(len(cases) for cases in values.values()),
            "values": entries[:limit] if limit else entries}


def distributions(
    rows: list[dict], key: str, *, tier: str | None = None, limit: int | None = None
) -> list[dict]:
    """One distribution per tier, each labelled, never one table over both: a
    tutorial and this instance's own study counted together is the merge R2 forbids."""
    tiers = [tier] if tier else sorted({str(row.get("tier") or "") for row in rows},
                                       key=lambda name: (name != "vendor", name))
    found = []
    for name in tiers:
        if name:
            entry = distribution(rows, key, tier=name, limit=limit)
            entry["tier"] = name
            found.append(entry)
    return found


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
    """Write down one query and everything it returned, and return the record's id.

    The whole ranked list goes in: a vendor hit and an earned hit disagreeing, and the
    earned one adopted, is invisible in a log of winners. Returns None rather than
    raising -- a read-only corpus directory is not a reason for a query to fail.
    """
    record = {
        "t": corpus.now_iso(),
        "id": f"{corpus.now_iso()}-{os.getpid()}-{_next_serial()}",
        "query": {"kind": kind, "text": text, "tier": tier, "unmatched": list(unmatched)},
        "matched": matched,
        "hits": hits,
        "taken": None,
    }
    if values is not None:
        record["values"] = values
    return record["id"] if _append(out_dir, record) else None


def record_taken(out_dir: Path | str, *, path: str, of: str | None = None) -> bool:
    """Write down that something was adopted, as its own line.

    `of` defaults to the most recent retrieval whose hits contained this path, not the
    most recent retrieval. With none, the adoption is recorded unattached: the agent
    used a case the corpus never handed it.
    """
    if of is None:
        for entry in reversed(_lines(out_dir)):
            if entry.get("kind") == "taken" or "query" not in entry:
                continue
            if any(_same_path(found.get("path"), path) for found in entry.get("hits") or []):
                of = str(entry.get("id"))
                break
    return _append(out_dir, {"t": corpus.now_iso(), "kind": "taken", "of": of, "path": path})


def read_retrievals(out_dir: Path | str) -> list[dict]:
    """The log, with each adoption folded onto its retrieval. Folded on read, so the
    file stays append-only."""
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
    """Whether two written paths name the same case, as paths rather than strings: the
    hits said `a\\b` and the adoption arrived from a shell as `a/b`."""
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
    """One record, one line, under a lock file.

    A bare append loses records under concurrency (763 of 900 from six processes, and
    796 with a single `O_APPEND` write: Windows append is a seek then a write). Past
    `LOCK_ATTEMPTS` the write goes ahead unlocked, so a stale lock cannot silence the
    log.
    """
    path = Path(out_dir) / corpus.RETRIEVALS
    try:
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
    except (TypeError, ValueError, OSError):
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
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a line truncated mid-write costs its line and no more
        if isinstance(entry, dict):
            found.append(entry)
    return found


# -- the command line ----------------------------------------------------------


def show_hits(hits: list[dict], held: int, *, runnable_only: bool, limit: int) -> None:
    shown = hits[:limit]
    if not shown:
        print("nothing matched")
    else:
        width = max(len(found["tier"]) for found in shown)
        for found in shown:
            print(f"{found['score']:>3}  {found['tier']:<{width}}  {found['path']}")
            print(f"     {', '.join(found['why'])}")
        if len(hits) > len(shown):
            tied = sum(1 for found in hits if found["score"] == shown[-1]["score"])
            print(f"\n{len(hits)} matched; showing {len(shown)}. {tied} share the lowest "
                  "score shown, and ties break by path, not merit.")
            print("Naming a solver, a turbulence model or a mesh type ranks on more fields.")
    if held and runnable_only:
        print(f"\n{held} mesh-only case(s) matched and are not shown; they cannot seed a "
              "solve. --include-mesh-only to see them.")


def show_distributions(found: list[dict]) -> None:
    """Each tier's table, named. Two tiers are two answers, never one."""
    if not any(entry["total"] for entry in found):
        print(f"nothing in the corpus sets {found[0]['key'] if found else ''}")
        return
    for index, entry in enumerate(found):
        if index:
            print()
        tier = entry.get("tier")
        if not entry["total"]:
            print(f"nothing in the {tier} tier sets {entry['key']}")
            continue
        print(f"{entry['key']}  --  {entry['total']} case(s)  [{tier} tier]")
        width = max(len(value["value"]) for value in entry["values"])
        for value in entry["values"]:
            share = value["count"] / entry["total"]
            print(f"  {value['value']:<{width}}  {value['count']:>4}  {share:>5.0%}")


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
                print(f"      {value.get('count'):>4}  {value.get('value')}")
            if entry.get("taken") is None and entry.get("hits"):
                print("      (nothing recorded as taken)")
        unattached = [line for line in _lines(args.corpus)
                      if line.get("kind") == "taken" and line.get("of") is None]
        if unattached:
            print(f"\n{len(unattached)} adoption(s) matched no retrieval in this log:")
            for line in unattached[-args.log_limit:]:
                print(f"   {line.get('t')}  {line.get('path')}")
        return 0

    needed = "asked for" if args.rebuild else rebuild_reason(args.corpus, tree=args.tutorials,
                                                             work=args.work)
    if needed and (args.tutorials is None or not args.tutorials.is_dir()):
        # The guard is on building, not asking: without it, `ensure` built a corpus
        # out of the working directory.
        print(f"the index needs building ({needed}) and there is no tutorial tree at "
              f"{args.tutorials or '$FOAM_TUTORIALS (unset)'} -- pass --tutorials, or "
              "source the OpenFOAM environment", file=sys.stderr)
        return 1
    rows, reason = ensure(args.corpus, tutorials=args.tutorials, work=args.work,
                          rebuild=args.rebuild)
    if reason and not args.json:
        print(f"(rebuilt the index: {reason})\n")

    def say_unmatched(text: str, tier: str | None) -> list[str]:
        missing = unmatched_tokens(rows, text, tier=tier)
        if missing and not args.json:
            print(f"(nothing in the corpus matches: {', '.join(missing)})\n")
        return missing

    if args.command == "keyword":
        found = distributions(rows, args.key, tier=args.tier)
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
        missing = say_unmatched(args.text, "earned")
        hits = failures(rows, args.text)
        record_retrieval(args.corpus, kind="failure", text=args.text, hits=hits[:args.limit],
                         matched=len(hits), tier="earned", unmatched=missing)
        if args.json:
            print(json.dumps(hits[:args.limit], indent=2))
        else:
            show_hits(hits, 0, runnable_only=False, limit=args.limit)
        return 0

    missing = say_unmatched(args.text, args.tier)
    runnable_only = not args.include_mesh_only
    hits, held = search_with_held(rows, args.text, tier=args.tier, runnable_only=runnable_only)
    record_retrieval(args.corpus, kind="regime", text=args.text, hits=hits[:args.limit],
                     matched=len(hits), tier=args.tier, unmatched=missing)
    if args.json:
        print(json.dumps(hits[:args.limit], indent=2))
    else:
        show_hits(hits, held, runnable_only=runnable_only, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
