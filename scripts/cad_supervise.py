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
    results = probes.run_all(Path(args.case), _spec(args.spec))
    went_off = probes.fired(results)
    return emit({"case": str(args.case), "fired": went_off,
                 "probes": [result.as_dict() for result in results]},
                1 if went_off else 0)


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
    went_off = [row["id"] for row in data.get("probes", [])
                if row.get("state") == probes.FIRED]
    # A fired probe is a failed exit even when the run ended `done`, because that pairing
    # is the whole reason the probes exist: the silent failures are the ones where the
    # run finishes, `checkMesh` passes, and the answer is of the wrong volume.
    return emit({**flat, "fired": went_off},
                0 if data.get("stopped") == "done" and not went_off else 1)


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
