#!/usr/bin/env python3
"""Does a run produce what a healthy run produces? Artifacts, not outcomes.

Written after a sweep spent an hour measuring a broken heartbeat. Three smoke runs had
been called clean beforehand, and every one of them was missing `heartbeat.jsonl` --
because the checks were "did the case pass" and "did the gate produce states", which the
bug did not touch. The runs were short enough to finish inside the watcher's 420 s
threshold, so the harness never had to be right for them to look right.

**An outcome check cannot see instrumentation that has stopped working.** A run that is
not observed still passes; it just stops being evidence. So this asks the other question:
against a sweep that is known good, does this run write the same set of files, and is each
of them non-empty?

    python3 scripts/cad_smoke.py <sweep-dir> [--against <sweep-dir>]

Exit 0 if every run carries every expected artifact, 1 otherwise, with the missing ones
named. It reads only what is on disk and runs nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED = ("record.json", "cells.log", "heartbeat.jsonl", "replies.jsonl", "watch.json")
"""What a run that was driven and watched leaves behind.

`heartbeat.jsonl` is the one that was missing and is the reason this file exists: it is
written by the runner and read by the supervisor, so its absence is invisible to both the
desk's result and the sweep's table. `replies.jsonl` goes with it -- same `on_turn`, so
the same silence takes both.
"""

TOLERATED = {"refused": ("cells.log",)}
"""Endings that legitimately lack an artifact, by the ending's own name.

A run that refuses on its first turn may never execute a cell, so an empty `cells.log` is
the correct record of it. Nothing else is tolerated: the list is short on purpose, and a
missing file that is not in here is a finding rather than a footnote."""


def runs(sweep: Path) -> list[Path]:
    return sorted((sweep / "runs").glob("*/"), key=lambda p: p.name)


def ending(run: Path) -> str:
    try:
        return str(json.loads((run / "record.json").read_text(encoding="utf-8"))
                   .get("stopped") or "")
    except Exception:  # noqa: BLE001 - a record we cannot read is its own finding
        return ""


def check(sweep: Path, expected: tuple[str, ...]) -> list[str]:
    """One line per run that is missing something, empty when every run is whole."""
    problems: list[str] = []
    for run in runs(sweep):
        allowed = set(TOLERATED.get(ending(run), ()))
        missing = [name for name in expected
                   if name not in allowed
                   and not (run / name).is_file() or
                   (name not in allowed and (run / name).is_file()
                    and (run / name).stat().st_size == 0)]
        if missing:
            problems.append(f"{run.name}: missing or empty {', '.join(sorted(set(missing)))}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--against", type=Path, default=None,
                        help="a known-good sweep; its artifact set is used instead of "
                             "the built-in one, so the two cannot drift apart")
    args = parser.parse_args(argv)

    expected = EXPECTED
    if args.against:
        good = runs(args.against)
        if not good:
            print(f"{args.against} has no runs to learn from", file=sys.stderr)
            return 1
        # Everything every healthy run over there has, which is stricter than a list
        # somebody remembered to update.
        common = set.intersection(*[{p.name for p in run.glob("*") if p.is_file()}
                                    for run in good])
        expected = tuple(sorted(common))
        print(f"expecting, from {args.against.name}: {', '.join(expected)}")

    problems = check(args.sweep, expected)
    for line in problems:
        print(line)
    if problems:
        print(f"\n{len(problems)} of {len(runs(args.sweep))} runs are not whole. "
              "Their numbers are not evidence until this is fixed.")
        return 1
    print(f"all {len(runs(args.sweep))} runs carry {', '.join(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
