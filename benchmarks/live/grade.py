#!/usr/bin/env python3
"""Grade the live backend's cases, and report what each one's agreement is worth.

    python benchmarks/live/grade.py list
    python benchmarks/live/grade.py smoke
    python benchmarks/live/grade.py poiseuille --resolution 32
    python benchmarks/live/grade.py cavity --resolution 128 --json
    python benchmarks/live/grade.py crosscheck
    python benchmarks/live/grade.py all --milestone M1

This grader **solves**, like `../lbm/grade.py` and unlike `../felix/grade.py` and
`../moving_mesh/grade.py`. Those two read a finished run: the cases they grade cost a
cluster an hour, so the run happens once and trimmed output is committed under
`tests/data/`. This solver is in this repository and its small cases finish in seconds,
so committing its output would replace a check that re-derives the number with a number
nobody can re-derive -- which is the argument `../lbm/README.md` already makes for the
same reason.

The one exception is the cross-check, and it is the interesting case: it solves one side
and *reads* the other. The OpenFOAM half is `tests/data/moving_mesh/viv_fixed/`, a
finished `pimpleFoam` run committed in September, and it is read through
`../moving_mesh/grade.py` rather than copied into a constant here, so if that run is ever
re-trimmed the comparison moves with it.

Exit status: 0 when every graded check passed, 1 when one failed, 2 when the case could
not be run at all (no Warp, no Newton, an unphysical setup). The same three the sibling
graders use.

## What the --device flag is for, and why a GPU is not optional for most of these

Every case here runs correctly on Warp's CPU device and most of them run slowly enough
there to be useless. The physics is device-independent -- `tests/test_live_cases.py`
asserts that the CPU and the GPU agree on a small case to single precision -- but a
lid-driven cavity at the resolution Ghia used is of order 10^11 lattice updates, which
is seconds on an H100 and hours on one CPU core, because Warp's CPU backend is
single-threaded.

So: `--device cpu` is for the cases marked otherwise in `list`, and
`benchmarks/live/modal_run.py` is how the rest are actually run. The numbers in
`PROGRESS_LIVE.md` say which device produced each of them, every time, because a rate
without a device is not a measurement and an agreement with a published value is not
improved by the hardware that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class GradeError(Exception):
    """The case cannot be run as it stands; the message says what to do.

    The same contract as `../felix/grade.py` and `../moving_mesh/grade.py`: a constant
    the setup does not supply is never assumed, and a refusal names the flag that
    supplies it.
    """


def _cases():
    try:
        from openreynolds.live import cases
    except ImportError as exc:
        raise GradeError(
            "the live backend needs NVIDIA Warp, which is an optional extra so that "
            "core OpenReynolds does not pull a CUDA toolchain in:\n"
            "    pip install 'openreynolds[live]'\n"
            f"(the import said: {exc})"
        ) from exc
    return cases


def _listing() -> int:
    cases = _cases()
    print("# the live backend's cases, and what each is measured against\n")
    width = max(len(name) for name in cases.CASES)
    for case in cases.CASES.values():
        marker = "GPU" if case.heavy else "cpu"
        newton = " (needs newton)" if case.needs_newton else ""
        print(f"  {case.name.ljust(width)}  {case.milestone:>2}  {marker}  "
              f"{case.about}{newton}")
    print("\n# the references, where each was read, and what is wrong with it\n")
    for name, entry in cases.REFERENCES.items():
        print(f"  {name}")
        print(f"    source:    {entry['source']}")
        print(f"    read:      {entry['read']}")
        print(f"    tolerance: {entry['tolerance']}  on {entry['tolerance_is']}")
        print(f"    caveat:    {entry['caveat']}\n")
    return 0


def _run(name: str, args: argparse.Namespace) -> dict:
    cases = _cases()
    overrides: dict = {}
    if args.resolution:
        overrides["resolution"] = args.resolution
    if args.steps:
        overrides["steps"] = args.steps
    if args.collision:
        overrides["collision"] = args.collision
    if args.smagorinsky is not None:
        overrides["smagorinsky"] = args.smagorinsky
    if args.reynolds:
        overrides["reynolds"] = args.reynolds

    progress = None if args.quiet else (lambda line: print(line, file=sys.stderr))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if name == "crosscheck":
            return cases.crosscheck(device=args.device, out_dir=out, progress=progress,
                                    **overrides)
        if name == "benchmark":
            layouts = tuple(args.layouts.split(",")) if args.layouts else None
            return cases.benchmark(device=args.device, progress=progress,
                                   layouts=layouts,
                                   resolutions=tuple(int(v) for v in
                                                     (args.resolution and [args.resolution])
                                                     or (64, 128)),
                                   steps=args.steps or 200)
        return cases.run_case(name, device=args.device, out_dir=out, progress=progress,
                              **overrides)
    except Exception as exc:
        raise GradeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="a case name, `list`, `benchmark`, or `all`")
    parser.add_argument("--resolution", type=int, default=0,
                        help="cells across whatever this case's grid parameter is")
    parser.add_argument("--reynolds", type=float, default=0.0,
                        help="Reynolds number, for the cases that take one")
    parser.add_argument("--steps", type=int, default=0,
                        help="run exactly this many steps instead of to convergence")
    parser.add_argument("--collision", default="",
                        choices=("", "trt", "bgk", "regularized"))
    parser.add_argument("--smagorinsky", type=float, default=None,
                        help="the Smagorinsky constant. A laminar case REFUSES a "
                             "non-zero value; see cases.LAMINAR")
    parser.add_argument("--device", default=None,
                        help="a Warp device, `cpu` or `cuda:0`. Default: the preferred one")
    parser.add_argument("--layouts", default="",
                        help="benchmark only: `soa,aos` to measure both, each in its "
                             "own process")
    parser.add_argument("--milestone", default="",
                        help="all only: run just M0, M1, M2 or M3")
    parser.add_argument("-o", "--out", default=str(Path(__file__).parent / "results"),
                        help="where figures go")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="one JSON object, for a script")
    parser.add_argument("--quiet", action="store_true", help="no progress on stderr")
    args = parser.parse_args(argv)

    try:
        cases = _cases()
        if args.case == "list":
            return _listing()
        if args.case == "all":
            wanted = [c.name for c in cases.CASES.values()
                      if not args.milestone or c.milestone == args.milestone]
            if not wanted:
                raise GradeError(
                    f"no cases in milestone {args.milestone!r}; they are "
                    f"{sorted({c.milestone for c in cases.CASES.values()})}")
            results = []
            for name in wanted:
                print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", file=sys.stderr)
                results.append(_run(name, args))
                if not args.as_json:
                    print(cases.report(results[-1]))
            if args.as_json:
                print(json.dumps({"results": results}, indent=2, default=str))
            failed = [r["case"] for r in results if not r["passed"]]
            print(f"\n{len(results) - len(failed)}/{len(results)} cases passed"
                  + (f"; failed: {', '.join(failed)}" if failed else ""),
                  file=sys.stderr)
            return 1 if failed else 0

        result = _run(args.case, args)
        if args.as_json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(cases.report(result))
        return 0 if result["passed"] else 1
    except GradeError as exc:
        print(f"cannot grade: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
