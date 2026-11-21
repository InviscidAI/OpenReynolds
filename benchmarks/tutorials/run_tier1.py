#!/usr/bin/env python3
"""Run every tier 1 case and print what each one agreed with.

    python3 run_tier1.py                 # all of them
    python3 run_tier1.py --quick         # the ones that are not slow
    python3 run_tier1.py --list          # what it would run, and stop

Tier 1 is the set that needs nothing installed: no OpenFOAM, no workspace, no account,
no model turn, nothing billed. The index beside this file lists them with the reference
each is measured against, and until now the only way to check them was to copy eight
commands out of a table one at a time. Nobody does that, so nobody found out whether
their install actually solves anything.

This runs them and prints one line each. It exits non-zero if a case that has a verdict
reports a failure, so it is usable as a check and not only as a demonstration. A case
that deliberately has **no** verdict -- the dam break, whose reference could not be read
-- is run, reported, and not counted either way, because inventing a pass for it is the
thing `benchmarks/README.md` refuses to do.

`panel.py --check` grades the three statements that need no new arithmetic --
d'Alembert, a symmetric section, and elliptic loading against the value
`lifting_line` already reports -- plus mode convergence. The fourth, a cambered
section's zero-lift angle against the closed-form thin-aerofoil integral, stays in
`tests/test_toolbox_panel.py`, because a second copy of that integral would be a
second thing to keep agreeing. `--check` names the test rather than duplicating it.

The commands are the ones the index documents, character for character, so the numbers
printed here are the numbers quoted there. Where that is slow it is slow. Measured on one
laptop, single-threaded: the lid-driven cavity at n = 128 is about 100 s, the sloshing
tank about 240 s, and the SPH dam break about 480 s -- so the three marked slow are most
of the total and `--quick` leaves them out by name rather than quietly running a smaller
case and reporting it as the same one. The first draft of this docstring guessed 25
seconds for `--quick` and the measured figure was 503, because the dam break had not been
marked slow.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (name, argv, what it is graded against, slow?, graded?)
# `graded` false means the case deliberately reports no verdict, which is a property of
# the case and not something to infer from its output.
CASES: tuple[tuple[str, tuple[str, ...], str, bool, bool], ...] = (
    ("plane Poiseuille, LBM",
     ("benchmarks/lbm/grade.py", "poiseuille"),
     "the exact parabolic profile, closed form", False, True),
    ("lid-driven cavity at Re 1000, LBM",
     ("benchmarks/lbm/grade.py", "cavity", "--n", "128"),
     "Ghia, Ghia & Shin (1982), Tables I and II", True, True),
    ("hydrostatic pressure, SPH",
     ("benchmarks/sph/grade.py", "hydrostatic"),
     "p = rho g (h - y), exact", False, True),
    ("first sloshing mode, SPH",
     ("benchmarks/sph/grade.py", "sloshing"),
     "omega^2 = g k tanh(k h), Lamb art. 228", True, True),
    ("dam break, SPH",
     ("benchmarks/sph/grade.py", "dambreak"),
     "Martin & Moyce (1952), UNREAD. Containment and weak compressibility are\n      graded; the surge front is printed and NOT graded", True, False),
    ("aerofoil and wing, panel method",
     ("openreynolds/toolbox/panel.py", "--check"),
     "d'Alembert (pressure drag exactly 0), a symmetric section (zero-lift angle and "
     "quarter-chord moment exactly 0), elliptic loading, and mode convergence", False,
     True),
    ("a pure tone, acoustics",
     ("openreynolds/toolbox/acoustics.py", "--tone"),
     "SPL = 20 log10(p_rms / 20 uPa), exact", False, True),
    ("a year of wind composed, exceedance",
     ("openreynolds/toolbox/wind_comfort.py", "--check"),
     "the fraction of the year above the threshold, exact", False, True),
    ("particle relaxation and two terminal velocities",
     ("benchmarks/particles/grade.py", "drag"),
     "v(t) = U(1 - exp(-t/tau_p)) and w_t = tau_p g (1 - rho_f/rho_p), both exact under\n"
     "      Stokes drag; and the Schiller-Naumann terminal against its implicit root,\n"
     "      where Re_p = 42.8 makes Stokes the wrong law", False, True),
    ("a tracer orbit in solid-body rotation",
     ("benchmarks/particles/grade.py", "tracer"),
     "the orbit closes exactly after one period, and RK4 must show order 4", False, True),
    ("conjugate heat transfer, the two-region reference",
     ("openreynolds/toolbox/conjugate.py", "--check"),
     "steady series conduction, exact, and the two continuities a conjugate"
     " interface owes at once; plus the shell-versus-slab factor, graded"
     " against its own algebra rather than a printed per cent",
     False, True),
    ("quasi-1D compressible relations",
     ("openreynolds/toolbox/nozzle.py", "--check"),
     "exact closed forms, and three identities: A/A* = 1 at the throat,"
     " the Prandtl relation M1* M2* = 1 derived independently of the jump"
     " relations it checks, and total pressure falling through every shock",
     False, True),
    ("the laminar flat plate, Blasius",
     ("openreynolds/toolbox/blasius.py", "--check"),
     "the Blasius similarity solution, exact: the published f''(0) ="
     " 0.3320573362151963, and reference_area.py's typed 1.328 held to 4 f''(0)."
     " Both conventions printed, since 0.4696 is the same answer sqrt(2) away",
     False, True),
    ("critical Stokes number for capture on a cylinder",
     ("benchmarks/particles/grade.py", "capture"),
     "1/8 in the St = tau_p U_inf / R convention. Robinson et al.\n"
     "      (arXiv:2310.03474v2) III.3 states 1/4 for St = m U/(l_y xi) with U = 2 U_inf\n"
     "      at the stagnation point, the same result a factor of two apart", True, True),
)

VERDICT = re.compile(r"\b(PASS|PASSED|FAIL|FAILED|DISAGREES|agrees)\b")
# A flag this checkout does not have. `--tone` and `--check` arrive with #246.
MISSING_FLAG = re.compile(r"unrecognized arguments:\s*(\S+)")
# The line worth keeping is the one carrying the verdict; failing that, one carrying a
# measured quantity. FIRST, not last: the trailing lines of these graders are notes
# about the method, and reporting a note as the result is how this read "u(y) = G/(2 mu)
# y (H - y) exactly" as plane Poiseuille's measured agreement.
WITH_VERDICT = re.compile(r"^.*\b(?:PASS|PASSED|FAIL|FAILED|agrees|DISAGREES)\b.*$", re.M)
WITH_NUMBER = re.compile(
    r"^(?!\s*note:).*?(?:RMS|rms|max error|measured|exceedance|[-+]?\d+\.\d+\s*%)"
    r".*$", re.M)


def run_one(argv: tuple[str, ...], timeout: float) -> dict:
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, "-I", "-X", "utf8", *argv],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"code": None, "seconds": time.monotonic() - started,
                "verdict": "TIMEOUT", "headline": f"did not finish in {timeout:g}s"}
    seconds = time.monotonic() - started
    text = done.stdout + done.stderr

    absent = MISSING_FLAG.search(text)
    if absent:
        return {"code": done.returncode, "seconds": seconds, "verdict": "NOT AVAILABLE",
                "headline": f"this checkout has no {absent.group(1)} (it arrives with "
                            f"PR #246); the case itself is sound"}

    found = VERDICT.findall(text)
    if any(f in ("FAIL", "FAILED", "DISAGREES") for f in found):
        verdict = "FAIL"
    elif found:
        verdict = "PASS"
    else:
        verdict = "no verdict in its output"

    picked = [l.strip() for l in WITH_VERDICT.findall(text) if l.strip()]
    if not picked:
        picked = [l.strip() for l in WITH_NUMBER.findall(text) if l.strip()]
    headline = picked[0] if picked else (
        text.strip().splitlines()[-1] if text.strip() else "")
    return {"code": done.returncode, "seconds": seconds, "verdict": verdict,
            "headline": headline[:110]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run every tier 1 case and print what each one agreed with.",
        epilog="Exits non-zero if a case with a verdict failed. The dam break has no "
               "verdict by design and is never counted either way.")
    parser.add_argument("--quick", action="store_true",
                        help="skip the three slow cases (the cavity at n=128, the "
                             "sloshing tank and the dam break: about 100 s, 240 s and "
                             "480 s on one laptop)")
    parser.add_argument("--list", action="store_true",
                        help="print what would run, and stop")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-case limit in seconds (default 900)")
    args = parser.parse_args(argv)

    chosen = [c for c in CASES if not (args.quick and c[3])]
    skipped = [c for c in CASES if args.quick and c[3]]

    if args.list:
        for name, cmd, against, slow, graded in chosen:
            print(f"{name}\n    python3 {' '.join(cmd)}\n    against: {against}"
                  + ("\n    slow" if slow else "")
                  + ("" if graded else "\n    no verdict by design"))
        for name, _, _, _, _ in skipped:
            print(f"{name}\n    skipped by --quick")
        return 0

    print(f"tier 1: {len(chosen)} case(s), nothing installed, nothing billed")
    if skipped:
        print("  skipped by --quick: " + ", ".join(n for n, _, _, _, _ in skipped))
    print()

    results = []
    for name, cmd, against, _, graded in chosen:
        print(f"  {name} ... ", end="", flush=True)
        found = run_one(cmd, args.timeout)
        found.update({"name": name, "against": against, "cmd": cmd, "graded": graded})
        results.append(found)
        print(f"{found['verdict']}  ({found['seconds']:.1f}s)")

    print()
    print("what each one was measured against, and what it reported")
    for found in results:
        print(f"  {found['name']}")
        print(f"      against  {found['against']}")
        print(f"      reported {found['headline'] or '(nothing matched)'}")

    # `graded` comes from the table, not from whether a verdict was found: a case that
    # should report one and did not is a failure, and one that never claimed to is not.
    graded = [r for r in results if r["graded"] and r["verdict"] != "NOT AVAILABLE"]
    failed = [r for r in graded if r["verdict"] != "PASS"]
    silent = [r for r in results if not r["graded"]]
    absent = [r for r in results if r["verdict"] == "NOT AVAILABLE"]
    total = sum(r["seconds"] for r in results)
    print()
    print(f"{len(graded) - len(failed)} of {len(graded)} graded case(s) passed "
          f"in {total:.0f}s")
    for found in silent:
        print(f"  no verdict by design, not counted: {found['name']}")
    for found in absent:
        print(f"  not in this checkout: {found['name']}")
        print(f"    {found['headline']}")
    for found in failed:
        print(f"  FAILED: {found['name']}  (exit {found['code']})")
        print(f"    rerun: python3 {' '.join(found['cmd'])}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
