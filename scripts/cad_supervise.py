#!/usr/bin/env python3
"""The out-of-band observer, as commands. The supervisor subagent drives this file.

    python3 scripts/cad_supervise.py preflight <workspace>        # before the run
    python3 scripts/cad_supervise.py watch <run-dir> [--deadline 1200]
    python3 scripts/cad_supervise.py probe <case-dir> [--spec spec.json]
    python3 scripts/cad_supervise.py contamination <run-dir> [--expected mesh_look.py]
    python3 scripts/cad_supervise.py observe <run-dir> --case <case-dir> [--spec s.json]
    python3 scripts/cad_supervise.py registry                     # the probes, as they stand
    python3 scripts/cad_supervise.py report <run-dir>             # the record's flat scalars

Every command prints JSON on stdout and nothing else, because its caller is an agent that
has to act on the answer rather than read it. The exit code carries the verdict too: `0`
clean, `1` the thing it was asked about went wrong (dirty workspace, alarm raised, probe
fired, contaminated run), `2` it could not be asked.

**It never writes into the workspace and never speaks to the run.** `observe` writes the
run record, which is the delivery side of §2; the detection side is here, in a different
process from the desk, holding imports -- `cad_audit`, `domain_probe`, `surfaces` -- that
are deliberately absent from the workspace the desk works in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openreynolds.buildup import (  # noqa: E402
    alarms, heartbeat, isolation, probes, record, supervise)


def emit(payload: dict, code: int = 0) -> int:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return code


def cmd_preflight(args) -> int:
    try:
        return emit({"clean": True, **supervise.preflight(Path(args.workspace))})
    except isolation.Dirty as exc:
        return emit({"clean": False, "why": str(exc)}, 1)


def cmd_watch(args) -> int:
    watcher = supervise.Supervisor(Path(args.run), case_dir=Path(args.case) if args.case
                                   else None, stale_s=args.stale, k=args.k,
                                   poll_s=args.poll)
    try:
        seen = watcher.watch(deadline_s=args.deadline)
    except supervise.Watched as exc:
        # Exit 2: this command could not do its job, as against exit 1 where it did and the
        # answer was bad. A second watcher is a mistake to fix, not a verdict on the run.
        return emit({"why": str(exc)}, 2)
    return emit(seen.as_dict(), 1 if seen.alarm else 0)


def cmd_probe(args) -> int:
    """The instruments, read off one case. Exit 0 either way.

    A non-zero exit would be a verdict, and these do not carry one: the numbers go to the
    supervisor, which has the case in front of it and can tell an external-flow seed point
    from an inverted mesh. See `probes.FIRED` for what the first baseline showed."""
    results = probes.run_all(Path(args.case), _spec(args.spec))
    return emit({"case": str(args.case),
                 "measured": probes.measured(results),
                 "probes": [result.as_dict() for result in results]}, 0)


VERDICTS = ("holds", "differs", "unmeasurable")
"""What the grader can conclude about one named property.

`unmeasurable` is a real answer and not a failure to try: T4 names a wall thickness
"measured, as the minimum over the solid" while its request says to mesh the water side
only, so no delivered artifact carries it. Saying so beats both guessing and silence."""

DESK = ("printed", "printed-from-input", "absent")
"""What the desk did about the property, which is a separate question from whether the
property holds. `printed-from-input` is the corpus's most common failure and the one no
instrument catches: eleven of twenty-six cases in the sol sweep printed a
`requested / measured` pair whose two sides derive from the same constant, so the pair
cannot disagree with itself whatever was built."""


def cmd_grade(args) -> int:
    """Write the supervisor's own property measurements into a finished run's record.

    The measuring is done by whatever is reading the delivered mesh -- in practice the
    `cad-supervisor` subagent, which did exactly this by hand across the sol corpus:
    slicing T11 to count 105 open cells, reading T19's radii off `constant/polyMesh/points`,
    measuring T23's gap on an axis and on a diagonal. This is only the writer.

    It is a *writer* and not a grader because the alternative was structured targets in
    the case files, and those cannot be made to work here: the desk chooses its own node
    ordering, patch ordering and tessellation, so a numeric comparison keyed on any of
    them measures the desk's incidentals rather than its geometry. A reader that measures
    the property afresh off the delivered artifacts does not care how they are ordered.

    The record is merged rather than rewritten, and only `properties` is touched. The
    observer used to recompute `stopped` and save the whole record over the top of it,
    which silently rewrote `refused` on every refusal case in every sweep on disk.
    """
    run = Path(args.run)
    data = record.load(run)
    if not data:
        return emit({"why": f"no {record.RECORD} under {run}"}, 2)
    named = data.get("properties") or []
    if not named:
        return emit({"why": f"{run} names no properties, so there is nothing to grade"}, 2)

    try:
        raw = (sys.stdin.read() if args.grades in ("-", "") else
               Path(args.grades).read_text(encoding="utf-8"))
        given = json.loads(raw)
    except (OSError, ValueError) as exc:
        return emit({"why": f"could not read the grades: {exc}"}, 2)
    if isinstance(given, dict):
        given = given.get("properties") or []
    if not isinstance(given, list):
        return emit({"why": "the grades should be a list, or an object carrying one "
                            "under `properties`"}, 2)

    wanted = [str(entry.get("property") or "") for entry in named]
    by_text = {text: i for i, text in enumerate(wanted)}
    seen: dict[int, dict] = {}
    for entry in given:
        if not isinstance(entry, dict):
            return emit({"why": f"a grade is not an object: {entry!r}"}, 2)
        text = str(entry.get("property") or "")
        index = by_text.get(text)
        if index is None:
            return emit({"why": f"no such property on this run: {text!r}",
                         "named": wanted}, 2)
        verdict = str(entry.get("verdict") or "")
        if verdict not in VERDICTS:
            return emit({"why": f"verdict {verdict!r} is not one of {VERDICTS}"}, 2)
        desk = str(entry.get("desk") or "")
        if desk not in DESK:
            return emit({"why": f"desk {desk!r} is not one of {DESK}"}, 2)
        if verdict != "unmeasurable" and not str(entry.get("measured") or "").strip():
            return emit({"why": f"{text!r} is graded {verdict!r} with no measurement; "
                                "a verdict without a number is the thing this replaces"},
                        2)
        seen[index] = {"measured": entry.get("measured"),
                       "source": entry.get("source") or "",
                       "verdict": verdict, "desk": desk,
                       "note": entry.get("note") or ""}

    missing = [wanted[i] for i in range(len(wanted)) if i not in seen]
    if missing and not args.partial:
        # Silence is what the inert field already gave us. A grader that skips a property
        # has to say so with `unmeasurable`, or pass --partial and own the gap.
        return emit({"why": "these properties were not graded; grade them or pass "
                            "--partial", "missing": missing}, 2)

    for index, found in seen.items():
        named[index].update(found)
    data["properties"] = named
    record.save(run, data)
    graded = sum(1 for entry in named if entry.get("verdict"))
    return emit({"run": str(run), "named": len(named), "graded": graded,
                 "holds": sum(1 for e in named if e.get("verdict") == "holds"),
                 "differs": sum(1 for e in named if e.get("verdict") == "differs"),
                 "unmeasurable": sum(1 for e in named
                                     if e.get("verdict") == "unmeasurable"),
                 "desk_printed_from_input": sum(1 for e in named
                                                if e.get("desk") == "printed-from-input"),
                 "ungraded": len(named) - graded}, 0)


def cmd_contamination(args) -> int:
    found = isolation.scan_run(Path(args.run), expected=args.expected or [])
    return emit({**found.as_dict(), "lines": found.lines()},
                1 if found.contaminated else 0)


def cmd_observe(args) -> int:
    watcher = supervise.Supervisor(Path(args.run))
    data = watcher.observe(case_dir=Path(args.case) if args.case else None,
                           spec=_spec(args.spec), expected=args.expected or [])
    flat = {key: value for key, value in data.items()
            if not isinstance(value, (list, dict))}
    # The exit code answers "did this run end well", and nothing else. It used to also
    # answer "did a probe fire", on the argument that the pairing -- run finishes,
    # checkMesh passes, answer is the wrong volume -- is what the probes exist for. The
    # first baseline tested that pairing three times and the supervisor overturned all
    # three, while the two runs that were really wrong produced no probe signal at all.
    # A number that is right and an implication that is wrong should not become an exit
    # code; it should become something a reader looks at.
    return emit({**flat, "measured": [row["id"] for row in data.get("probes", [])
                                      if row.get("state") == probes.MEASURED]},
                0 if data.get("stopped") == "done" else 1)


def cmd_registry(args) -> int:
    return emit({"probes": [vars(probe) for probe in probes.REGISTRY],
                 "dormant": [p.id for p in probes.REGISTRY if p.state == probes.DORMANT],
                 "active": [p.id for p in probes.REGISTRY if p.state == probes.ACTIVE]})


def cmd_report(args) -> int:
    data = record.load(Path(args.run))
    if not data:
        return emit({"why": f"no {record.RECORD} under {args.run}"}, 2)
    flat = {key: value for key, value in data.items()
            if not isinstance(value, (list, dict))}
    beats = heartbeat.read(Path(args.run))
    return emit({**flat, "beats": len(beats),
                 "last_beat": vars(beats[-1]) if beats else None,
                 "replies": len(record.replies(Path(args.run)))},
                0 if data.get("stopped") == "done" else 1)


def _spec(path: str | None) -> dict:
    """The case's own statement of what it asked for -- `extent_m` and the rest.

    The `scale` probe is measured against the request, which means the request has to
    state its dimensions; a case that states none leaves that probe skipped rather than
    passed, and the difference is the whole point of recording `skipped`."""
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read the spec at {path}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)

    one = subs.add_parser("preflight", help="assert a workspace is clean, or refuse")
    one.add_argument("workspace")
    one.set_defaults(run_command=cmd_preflight)

    two = subs.add_parser("watch", help="poll a live run for the three alarms")
    two.add_argument("run")
    two.add_argument("--case", default="",
                     help="the case directory, watched for when a mesh first appears")
    two.add_argument("--deadline", type=float, default=None,
                     help="seconds before the supervisor calls it wedged itself")
    two.add_argument("--stale", type=float, default=alarms.HEARTBEAT_STALE_S,
                     help="heartbeat age that counts as wedged")
    two.add_argument("--k", type=int, default=alarms.K,
                     help="consecutive turns before no-progress or starved fires")
    two.add_argument("--poll", type=float, default=supervise.POLL_S)
    two.set_defaults(run_command=cmd_watch)

    three = subs.add_parser("probe", help="the silent-failure probes, on a case on disk")
    three.add_argument("case")
    three.add_argument("--spec", default="")
    three.set_defaults(run_command=cmd_probe)

    four = subs.add_parser("contamination", help="grep a run's own output for house surfaces")
    four.add_argument("run")
    four.add_argument("--expected", action="append",
                      help="a house name a tooled arm was supposed to reach")
    four.set_defaults(run_command=cmd_contamination)

    five = subs.add_parser("observe", help="grade a finished run into its record")
    five.add_argument("run")
    five.add_argument("--case", default="")
    five.add_argument("--spec", default="")
    five.add_argument("--expected", action="append")
    five.set_defaults(run_command=cmd_observe)

    grade = subs.add_parser(
        "grade", help="write the supervisor's own property measurements into a record")
    grade.add_argument("run")
    grade.add_argument("--grades", default="-",
                       help="a JSON file of measurements, or `-` for stdin")
    grade.add_argument("--partial", action="store_true",
                       help="accept a run where some named properties are ungraded; "
                            "without it, every property must carry a verdict")
    grade.set_defaults(run_command=cmd_grade)

    six = subs.add_parser("registry", help="the probes and their states")
    six.set_defaults(run_command=cmd_registry)

    seven = subs.add_parser("report", help="a finished run's flat scalars")
    seven.add_argument("run")
    seven.set_defaults(run_command=cmd_report)

    # The thresholds are `alarms`' and are defaulted from it rather than restated, so
    # that a number in a report can be re-derived from the module that decided it.
    args = parser.parse_args(argv)
    return args.run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
