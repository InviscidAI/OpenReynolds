#!/usr/bin/env python3
"""What a study has produced and how far it has got, written down where the next
agent can find it.

Two files under `<study>/.reynolds/`, both plain text, both safe to read with `cat`:

- `manifest.jsonl` -- one line per artifact, append-only. Every picture, plot,
  animation and report gets a line saying *what it is for* (`mesh-full`,
  `vorticity`, `residuals`, ...), not just where it landed. "Show me the latest
  mesh" is then a lookup instead of a hunt through directories. Ladder rung
  outcomes recorded by `ladder.py --record` are lines here too, under the kind
  `rung` -- evidence, like a picture, is a fact a later session would otherwise
  pay to re-derive.
- `phases.json` -- the study's pipeline, one row per phase, each with a status.
  A session that ends mid-solve leaves this behind; the next one reads it and
  picks up at the first phase that is not `done` rather than re-deriving the
  whole case from the transcript.

Why files and not a service: the instance can be reaped at any moment (the 24 h
Sandbox ceiling, the idle reaper, a preemption), and everything under `/work` is
on the Volume and survives that. The same reasoning `jobd` uses for job state --
all of it plain files on disk, so the answer is re-derived by reading rather than
remembered by a process that may be gone.

Nothing here decides anything. It records what happened and answers questions
about it; what to do next is the agent's call.

    python3 study_state.py list                     # every artifact
    python3 study_state.py list --kind vorticity    # just those
    python3 study_state.py latest mesh-full         # one path, for a read_file
    python3 study_state.py phases                   # the pipeline and its statuses
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

STATE_DIR = ".reynolds"
MANIFEST = "manifest.jsonl"
PHASES_FILE = "phases.json"

KINDS = (
    "geometry-preview",
    "mesh-full",
    "mesh-closeup",
    "mesh-patches",
    "contact-sheet",
    "velocity",
    "pressure",
    "vorticity",
    "streamlines",
    "residuals",
    "forces",
    "animation",
    "report",
    "gallery",
    "other",
)
"""The purposes an artifact can be registered under. `other` is the escape hatch:
a kind that is not on this list is recorded as it is given rather than refused --
a new sort of picture is not a reason to lose the row -- but the listed ones are
what the query side is built around."""

PHASES = (
    "geometry",
    "preview",
    "mesh",
    "checkMesh",
    "probe",
    "solve",
    "reconstruct",
    "render",
    "animate",
    "report",
)
"""The pipeline in order. A study may skip any of them (a mesh-only study stops
after `checkMesh`; a case with no moving parts never animates), and skipping is
recorded as `skipped` rather than left `pending`, so "not done" and "not wanted"
do not read the same."""

STATUSES = ("pending", "running", "done", "failed", "skipped")

RUNG_KIND = "rung"
"""The manifest kind for ladder rung evidence, written by `ladder.py --record`.

A rung outcome is not a picture, but it earns its manifest line the same way one
does: it is a fact about the study that a later session would otherwise pay to
re-derive. The outcome lives under `meta` -- the class, the rung number and name,
pass/fail/skipped, the measured value and the known answer it was set against."""

RUNG_STATUSES = ("pass", "fail", "skipped")
"""What can happen on a rung. Deliberately not the phase statuses: a rung is never
`pending` or `running` in the manifest, because only outcomes are worth a line."""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -- where the state lives ---------------------------------------------------------


def find_root(start: Path | str = ".") -> Path:
    """The study directory for `start`: the nearest ancestor that already has a
    `.reynolds/`, else `start` itself.

    A case usually sits inside the study home (`/work/<study>/<case>`), and both
    the case scripts and the study-level ones want the same state. Walking up to
    an existing directory means a script pointed at the case and a script pointed
    at the study home write to one place instead of two.
    """
    here = Path(start).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / STATE_DIR).is_dir():
            return candidate
        # Do not walk out past the workspace root into /work itself: two studies
        # would then share one manifest.
        if candidate.name == "work" or candidate == candidate.parent:
            break
    return here


def state_dir(start: Path | str = ".", *, create: bool = True) -> Path:
    root = find_root(start) / STATE_DIR
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


# -- the artifact manifest ---------------------------------------------------------


def record(
    kind: str,
    path: Path | str,
    *,
    root: Path | str = ".",
    case: str = "",
    label: str = "",
    **meta: Any,
) -> dict[str, Any]:
    """Register one artifact and return the row that was written.

    `path` is stored relative to the study root when it is underneath it, so a
    manifest copied home still points at something. Appending is one `open(...,
    "a")` and one line: two scripts writing at once interleave lines, never
    halves of a line, which is why this is JSONL and not a JSON array that has to
    be read, edited and rewritten.
    """
    study = find_root(root)
    target = Path(path)
    stored = str(target)
    try:
        absolute = target if target.is_absolute() else (study / target)
        stored = str(absolute.resolve().relative_to(study.resolve())).replace(os.sep, "/")
    except (ValueError, OSError):
        stored = str(target).replace(os.sep, "/")
    row: dict[str, Any] = {
        "kind": kind,
        "path": stored,
        "case": case,
        "label": label,
        "at": now_iso(),
    }
    if meta:
        row["meta"] = meta
    directory = study / STATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / MANIFEST).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
    return row


def artifacts(
    *,
    root: Path | str = ".",
    kind: str | Iterable[str] | None = None,
    case: str = "",
    exists: bool = False,
) -> list[dict[str, Any]]:
    """Every registered artifact, oldest first, optionally filtered.

    A line that will not parse is skipped rather than raised on: a manifest is
    written by several scripts and a truncated last line (a job killed mid-write)
    must not take the whole listing down with it.
    """
    study = find_root(root)
    manifest = study / STATE_DIR / MANIFEST
    if not manifest.exists():
        return []
    wanted = {kind} if isinstance(kind, str) else set(kind) if kind else set()
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "path" not in row:
            continue
        if wanted and row.get("kind") not in wanted:
            continue
        if case and row.get("case") != case:
            continue
        row["abspath"] = str(study / row["path"])
        if exists and not Path(row["abspath"]).exists():
            continue
        rows.append(row)
    return rows


def latest(kind: str, *, root: Path | str = ".", case: str = "", exists: bool = True) -> dict[str, Any] | None:
    """The most recently registered artifact of a kind, or None.

    `exists` defaults to True here and False in `artifacts()` on purpose: asking
    for "the latest mesh picture" wants one that is still on disk, while listing
    the manifest is also how you find out that something was deleted.
    """
    rows = artifacts(root=root, kind=kind, case=case, exists=exists)
    return rows[-1] if rows else None


# -- rung evidence -----------------------------------------------------------------


def _drop_rung_rows(study: Path, class_key: str, number: int) -> None:
    """Remove any earlier record of the same class and rung from the manifest.

    Re-recording a rung is a correction, not an addition -- two lines for one rung
    would leave a later reader to guess which is current. Every other line is kept
    byte-for-byte, including ones that do not parse, so the manifest stays the same
    file it was; the rewrite goes through a temporary file and `os.replace` like
    `save_phases` does, so a reader never sees half of it.
    """
    manifest = study / STATE_DIR / MANIFEST
    if not manifest.exists():
        return
    kept: list[str] = []
    dropped = False
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = None
        meta = row.get("meta") if isinstance(row, dict) else None
        if (
            isinstance(meta, dict)
            and row.get("kind") == RUNG_KIND
            and str(meta.get("class", "")) == class_key
            and meta.get("rung") == number
        ):
            dropped = True
            continue
        kept.append(line)
    if not dropped:
        return
    tmp = manifest.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(part + "\n" for part in kept), encoding="utf-8")
    os.replace(tmp, manifest)


def record_rung(
    number: int,
    status: str,
    *,
    root: Path | str = ".",
    case: str = "",
    class_key: str = "",
    name: str = "",
    value: Any = None,
    known: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Record one ladder rung's outcome and return the row that was written.

    One more manifest line, in the same shape every artifact gets, because the
    manifest is already the place a resumed session looks. The class and rung
    number identify the record: writing the same pair again replaces the earlier
    line rather than sitting beside it, so the manifest answers "what happened on
    rung 2" with one row instead of a history to reconcile.
    """
    if status not in RUNG_STATUSES:
        raise ValueError(f"status must be one of {', '.join(RUNG_STATUSES)}; got {status!r}")
    number = int(number)
    _drop_rung_rows(find_root(root), class_key, number)
    meta: dict[str, Any] = {"class": class_key, "rung": number, "name": name, "status": status}
    if value is not None:
        meta["value"] = value
    if known:
        meta["known"] = known
    if note:
        meta["note"] = note
    label = f"rung {number} ({name}): {status}" if name else f"rung {number}: {status}"
    where = Path(root)
    if where.is_file():
        where = where.parent
    try:
        where = where.resolve()
    except OSError:
        pass
    return record(RUNG_KIND, where, root=root, case=case, label=label, **meta)


def rung_evidence(*, root: Path | str = ".", case: str = "") -> list[dict[str, Any]]:
    """The recorded outcome of every ladder rung, one row per class and rung.

    Rows whose `meta` does not carry a class and a rung number are passed over
    rather than raised on, for the same reason `artifacts` skips a line that will
    not parse. Should two rows for one rung survive in the manifest anyway, the
    later one wins, because it is the correction.
    """
    latest_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in artifacts(root=root, kind=RUNG_KIND, case=case, exists=False):
        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        try:
            number = int(meta["rung"])
        except (KeyError, TypeError, ValueError):
            continue
        latest_rows[(str(meta.get("class", "")), number)] = row
    return [latest_rows[key] for key in sorted(latest_rows)]


# -- the phase table ---------------------------------------------------------------


def _blank_phases() -> dict[str, Any]:
    return {
        "updated_at": now_iso(),
        "case": "",
        "phases": [{"name": name, "status": "pending", "note": "", "started_at": "", "ended_at": ""} for name in PHASES],
    }


def load_phases(root: Path | str = ".") -> dict[str, Any]:
    """The phase table, or a blank one. Never raises: a corrupt file is a reason
    to start the table again, not a reason to stop the study."""
    path = find_root(root) / STATE_DIR / PHASES_FILE
    if not path.exists():
        return _blank_phases()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _blank_phases()
    if not isinstance(data, dict) or not isinstance(data.get("phases"), list):
        return _blank_phases()
    known = {row.get("name") for row in data["phases"] if isinstance(row, dict)}
    for name in PHASES:
        if name not in known:
            data["phases"].append({"name": name, "status": "pending", "note": "", "started_at": "", "ended_at": ""})
    return data


def save_phases(data: dict[str, Any], root: Path | str = ".") -> Path:
    """Written to a temporary file and moved into place, so a reader never sees
    half a table -- the one thing JSONL gets for free and JSON does not."""
    directory = find_root(root) / STATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now_iso()
    path = directory / PHASES_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def set_phase(
    name: str,
    status: str,
    *,
    root: Path | str = ".",
    note: str = "",
    case: str = "",
) -> dict[str, Any]:
    """Move one phase to a status and stamp the time. Returns the whole table."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}; got {status!r}")
    data = load_phases(root)
    if case:
        data["case"] = case
    for row in data["phases"]:
        if row.get("name") != name:
            continue
        row["status"] = status
        if note:
            row["note"] = note
        if status == "running" and not row.get("started_at"):
            row["started_at"] = now_iso()
        if status in ("done", "failed", "skipped"):
            row["ended_at"] = now_iso()
        break
    else:
        data["phases"].append({
            "name": name, "status": status, "note": note,
            "started_at": now_iso() if status == "running" else "",
            "ended_at": now_iso() if status in ("done", "failed", "skipped") else "",
        })
    save_phases(data, root)
    return data


def next_phase(root: Path | str = ".") -> str:
    """The first phase that is neither done nor skipped -- where a resumed study
    picks up. Empty when every phase is settled."""
    for row in load_phases(root)["phases"]:
        if row.get("status") not in ("done", "skipped"):
            return str(row.get("name") or "")
    return ""


def phase_status(name: str, root: Path | str = ".") -> str:
    for row in load_phases(root)["phases"]:
        if row.get("name") == name:
            return str(row.get("status") or "pending")
    return "pending"


# -- the command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."), help="A path inside the study (default: here).")
    sub = ap.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="Every registered artifact.")
    listing.add_argument("--kind", default="", help=f"One of: {', '.join(KINDS)}")
    listing.add_argument("--case", default="")
    listing.add_argument("--json", action="store_true")
    listing.add_argument("--missing", action="store_true", help="Include rows whose file is gone.")

    newest = sub.add_parser("latest", help="The newest artifact of a kind: its path.")
    newest.add_argument("kind")
    newest.add_argument("--case", default="")

    sub.add_parser("phases", help="The pipeline and where it got to.")
    sub.add_parser("next", help="The first phase that is not done or skipped.")

    mark = sub.add_parser("set", help="Move a phase to a status.")
    mark.add_argument("phase")
    mark.add_argument("status", choices=list(STATUSES))
    mark.add_argument("--note", default="")
    mark.add_argument("--case", default="")

    add = sub.add_parser("record", help="Register an artifact by hand.")
    add.add_argument("kind")
    add.add_argument("path")
    add.add_argument("--case", default="")
    add.add_argument("--label", default="")

    args = ap.parse_args(argv)
    root = args.root

    if args.command == "list":
        rows = artifacts(root=root, kind=args.kind or None, case=args.case, exists=not args.missing)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        if not rows:
            print("nothing registered yet")
            return 0
        width = max(len(str(row.get("kind", ""))) for row in rows)
        for row in rows:
            label = f"  {row['label']}" if row.get("label") else ""
            print(f"{str(row.get('kind','')):<{width}}  {row['path']}{label}")
        return 0

    if args.command == "latest":
        row = latest(args.kind, root=root, case=args.case)
        if row is None:
            print(f"no {args.kind} registered", file=sys.stderr)
            return 1
        print(row["abspath"])
        return 0

    if args.command == "phases":
        data = load_phases(root)
        print(f"case {data.get('case') or '-'}   updated {data.get('updated_at')}")
        for row in data["phases"]:
            note = f"   {row['note']}" if row.get("note") else ""
            print(f"  {row.get('status',''):<8} {row.get('name','')}{note}")
        return 0

    if args.command == "next":
        print(next_phase(root))
        return 0

    if args.command == "set":
        set_phase(args.phase, args.status, root=root, note=args.note, case=args.case)
        print(f"{args.phase} -> {args.status}")
        return 0

    if args.command == "record":
        row = record(args.kind, args.path, root=root, case=args.case, label=args.label)
        print(json.dumps(row))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
