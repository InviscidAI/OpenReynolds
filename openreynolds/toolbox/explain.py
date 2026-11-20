#!/usr/bin/env python3
"""What a finished case's numbers mean, in sentences, with what would make them
wrong put first.

Nothing here measures anything. Every number and every reading in the output comes
from an instrument in this directory that already produced it, and that is the point
rather than a shortcut: a second opinion about convergence computed by a different
file is how a study ends up with two answers and no way to tell them apart, which is
the incident `claims.py` was written for. So this is a collector. What it adds is the
one thing no single instrument can say -- the order to read them in.

what was already here, and what was not

`log_digest.how_it_ended` already writes the convergence reading in words whose shape
depends on why the residuals stopped (`log_digest.py:260-291`). `wall.py` already
reports the achieved y+ per patch with the regimes it spans. `results.force_headline`
already pairs each coefficient's last value with the mean over its tail, and says why
the tail mean is the one people quote. `mesh_quality.measure` already grades the mesh
off checkMesh's own numbers. Four good paragraphs that have never been in one place,
in an order, with a sentence at the end saying what they jointly permit.

the order, and why it is that order

Caveats first, numbers second. A document that leads with `Cd = 1.34` and mentions on
page two that the run was still falling has told the reader the number; the caveat
arrives after they have written it down. So this prints what would make the numbers
wrong before it prints the numbers, and the verdict line is a sentence about the
numbers rather than a grade on the study.

Three verdicts, and none of them is the word "converged":

- **do not quote these numbers** -- something makes them not an answer at all: the
  solver stopped itself, the residuals are climbing, or no time but the initial
  condition was ever written.
- **a reading, not an answer** -- the run produced numbers and at least one thing
  about it is unresolved: still falling, a plateau under a steady solver, a wall
  outside the range its model is written for, no mesh grade available.
- **nothing here contradicts them** -- the weakest positive statement that is true.
  Not "validated", not "converged": this reads four files and none of them is a
  benchmark. A number agreeing with a published one is `benchmarks/`, and a number
  agreeing with its own figure is `claims.py`.

A reading that cannot be made is printed as a reading that cannot be made. The
alternative -- a quiet gap where the mesh grade should be -- reads as a mesh that was
checked and found fine, which is the failure mode `claims.py` and `report.py` have
each already paid for once.

    python3 explain.py /work/case
    python3 explain.py /work/case --json /work/case/explain.json
    python3 explain.py /work/case --log /work/case/log.simpleFoam
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import log_digest  # noqa: E402  (sibling scripts, not a package)
import results  # noqa: E402
import study_state  # noqa: E402

STOP = "stop"
CAVEAT = "caveat"
CLEAR = "clear"
"""How much a reading costs the numbers. `STOP` means they are not an answer; `CAVEAT`
means they are a reading; `CLEAR` contributes nothing either way and is printed because
a reader who is told only the bad news cannot tell a quiet instrument from a happy one."""

VERDICTS = {
    STOP: "do not quote these numbers",
    CAVEAT: "a reading, not an answer",
    CLEAR: "nothing here contradicts them",
}


def note(level: str, said: str, where: str = "") -> dict[str, str]:
    return {"level": level, "said": said, "where": where}


# -- the four readings -----------------------------------------------------------


def convergence(case: Path, log: Path | None) -> list[dict[str, str]]:
    """What the solver log says, in `log_digest`'s own words.

    The sentences are not rewritten here. `log_digest._residual_reading` distinguishes
    a plateau on a shedding wake (the physics) from a climb towards a floating point
    exception (a failure) and says each in its own terms, after a version that called
    both "did not report convergence" and taught a reader that nothing converges. That
    distinction is the whole value and it is not worth paraphrasing.
    """
    if log is None:
        return [note(CAVEAT, "no solver log in this case, so nothing here can say "
                             "whether the run converged, diverged or was cut off",
                     "results.find_solver_log")]
    data = log_digest.digest(log)
    said = log_digest.how_it_ended(data, log_digest.requested_end_time(log))
    shape = log_digest.residual_shape(data.get("residuals") or {})["shape"]
    residual_level = {"diverging": STOP, "falling": CAVEAT, "levelled": CAVEAT,
                      "short": CAVEAT}.get(shape, CAVEAT)
    # Levelled per sentence, not per log. `how_it_ended` returns how the run stopped and
    # then what the residuals were doing, and those are two facts with two costs: a run
    # that reached the end of its controlDict with climbing residuals should not mark
    # "ran to the end" as the alarming half. Marking both at the worse of them is what
    # makes a reader stop reading the markers.
    out: list[dict[str, str]] = []
    for line in said:
        if line.startswith("residuals:"):
            out.append(note(residual_level, line, str(log.name)))
        elif data.get("fatal") or line.startswith("ended: no End line"):
            out.append(note(STOP, line, str(log.name)))
        else:
            out.append(note(CLEAR, line, str(log.name)))
    return out


def times(case: Path) -> list[dict[str, str]]:
    """Whether anything but the initial condition was written.

    A case whose only time directory is `0` draws, plots and reports perfectly well,
    and every number in it is the initial condition with a confident label on it --
    the trap `showcase.open_case` refuses by name for a decomposed case and that
    nothing refuses for a reconstructed one that never got going."""
    latest = study_state.latest_time(case)
    if not latest or float(latest) <= 0.0:
        return [note(STOP, "no time directory but 0: every field here is the initial "
                           "condition, whatever a picture of it looks like", "the case")]
    return [note(CLEAR, f"a solution was written, latest t = {float(latest):g}",
                 "the case")]


FROM_STATUS = {"ok": CLEAR, "warn": CAVEAT, "fail": STOP, "skipped": CAVEAT}
"""`preflight.Finding`'s four statuses, which `wall.py` reuses, mapped onto what a
reading costs the numbers. `skipped` is a caveat rather than nothing: a measurement
that could not be made is a thing the reader has to know, and the gap where it would
have gone is otherwise read as a measurement that came back fine."""


def wall_resolution(case: Path) -> list[dict[str, str]]:
    """The achieved y+ and the wall treatment, as `wall.findings` already pairs them.

    The judgement is not made here. `wall.WALL_FUNCTIONS` carries the validity band
    each `nut` condition is written for -- 30 to 300 for the plain high-Re forms, all
    y+ for the Spalding blend `case_gen.py` writes -- and `wall.findings:1123-1163`
    already puts the measured range beside the band and marks it `warn` where part of
    a wall was treated by a law that does not hold there. That comparison is the thing
    the aerofoil O-grid at y+ about 1200 paid for; forming a second opinion about it
    here would be a second number to reconcile, which is how this goes wrong.
    """
    try:
        import wall  # noqa: PLC0415 - matplotlib at its module level, so loaded lazily
    except Exception as exc:  # noqa: BLE001
        return [note(CAVEAT, f"the y+ reader would not load ({type(exc).__name__}), so "
                             "the wall resolution is unknown here", "wall.py")]
    try:
        found = wall.measure(case)
        rows = wall.findings(found)
    except Exception as exc:  # noqa: BLE001 - a case with no wall data is not a failure
        return [note(CAVEAT, f"no achieved y+ could be read ({exc})", "wall.py")]
    if not rows:
        return [note(CAVEAT, "nothing in this case says what y+ the walls reached",
                     "wall.py")]
    out: list[dict[str, str]] = []
    for row in rows:
        meaning = getattr(row, "meaning", "")
        said = f"{row.check}: {row.measured}" + (f" -- {meaning}" if meaning else "")
        out.append(note(FROM_STATUS.get(row.status, CAVEAT), _brief(said), "wall.py"))
    return out


def _brief(text: str, at_most: int = 260) -> str:
    """One instrument's sentence, cut to fit beside three others.

    `wall.py` prints the full reading -- every excluded patch, the regime percentages,
    the band each wall condition is written for -- and it is right to, because that is
    the whole of what it has to say. Here it is one of four readings and the reader has
    not asked for the wall report, so the verdict and the first clause of the evidence
    are what is carried and `wall.py` is named as where the rest is. Cut on a word, so
    the line never ends mid-number."""
    flat = " ".join(str(text).split())
    if len(flat) <= at_most:
        return flat
    return flat[:at_most].rsplit(" ", 1)[0] + " ... (wall.py prints the rest)"


def mesh(case: Path) -> list[dict[str, str]]:
    """checkMesh's own verdict, through `mesh_quality.measure`.

    Reported and not re-derived: the thresholds in `preflight.py:1632` and the grades in
    `mesh_quality.py` are the repository's one opinion about what a bad cell is, and a
    second opinion formed here would be a second number to reconcile."""
    try:
        import mesh_quality  # noqa: PLC0415
        found = mesh_quality.measure(case)
    except Exception as exc:  # noqa: BLE001
        return [note(CAVEAT, f"the mesh is not graded here ({type(exc).__name__}: {exc}) "
                             "-- run checkMesh and mesh_quality.py if it matters",
                     "mesh_quality.py")]
    worst = None
    try:
        worst = mesh_quality.worst_metric(found)
    except Exception:  # noqa: BLE001
        worst = None
    if not worst:
        return [note(CAVEAT, "no checkMesh numbers in this case, so the mesh is "
                             "ungraded here", "mesh_quality.py")]
    grade = str(worst.get("grade") or "").lower()
    level = STOP if grade == "fail" else (CAVEAT if grade == "warn" else CLEAR)
    return [note(level, f"worst mesh metric: {worst.get('metric')} = "
                        f"{worst.get('value')} ({grade or 'ungraded'})",
                 "mesh_quality.py, checkMesh")]


# -- the numbers, each with the file it came out of ------------------------------


def numbers(case: Path) -> tuple[list[dict[str, str]], list[str]]:
    """The headline coefficients, as `results.py` reads them, each with its source file.

    No value is computed here and none is rounded differently from the way `results.py`
    rounds it, so a number in this document and the same number in `results.md` are the
    same string. The tail mean sits beside the last value rather than replacing it,
    which is `results.tail_mean`'s own rule and the reason it is quoted: for a shedding
    case the last row is whichever phase of the cycle the final write landed on.
    """
    try:
        grouped = results.find_force_files(case)
        key = results.choose_force_series(grouped)
        paths = grouped[key]
        history = results.read_history(paths)
    except Exception as exc:  # noqa: BLE001 - a case with no forces is a normal case
        return [], [f"no force or coefficient history ({exc})"]
    if not history.get("rows"):
        return [], [f"{key} has a header and no rows yet"]
    columns = results.plot_columns(history["columns"])
    if not columns:
        return [], [f"{key} has no column but time"]
    where = results.post_relative(paths[-1])
    if len(paths) > 1:
        where += f" (+{len(paths) - 1} more, stitched)"
    out = [{"name": name, "value": value, "where": where}
           for name, value in results.force_headline(history, columns).items()]
    return out, []


# -- putting it in an order ------------------------------------------------------


def explain(case: Path, log: Path | None = None) -> dict[str, Any]:
    case = Path(case)
    if log is None:
        found = results.find_solver_log(case)
        log = Path(found) if found else None
    readings: list[dict[str, str]] = []
    readings += times(case)
    readings += convergence(case, log)
    readings += mesh(case)
    readings += wall_resolution(case)
    rows, unread = numbers(case)
    levels = {row["level"] for row in readings}
    verdict = STOP if STOP in levels else (CAVEAT if CAVEAT in levels else CLEAR)
    if not rows:
        unread = unread or ["no headline number was found in this case"]
    return {
        "case": str(case),
        "name": case.name,
        "log": str(log) if log else None,
        "verdict": verdict,
        "says": VERDICTS[verdict],
        "readings": readings,
        "numbers": rows,
        "unread": unread,
    }


def _wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(text, width=92, initial_indent=indent,
                         subsequent_indent=indent + "  ")


MARKERS = {STOP: "!!", CAVEAT: " ?", CLEAR: "  "}
"""Two characters, not colour and not an emoji: this is read in a terminal, pasted
into a message, and sometimes written to a file."""


def report(found: dict[str, Any]) -> str:
    out = [f"{found['name']} -- what these numbers mean", ""]
    out.append("  read this first")
    for row in found["readings"]:
        out.append(_wrap(f"{MARKERS[row['level']]} {row['said']}"
                         + (f"  [{row['where']}]" if row["where"] else ""), "    "))
    out.append("")
    out.append(f"  so: {found['says']}")
    out.append("")
    if found["numbers"]:
        out.append("  what it reports")
        width = max(len(row["name"]) for row in found["numbers"])
        for row in found["numbers"]:
            out.append(f"    {row['name']:<{width}}  {row['value']}")
        out.append(f"    read from {found['numbers'][0]['where']}")
        out.append("")
    for line in found["unread"]:
        out.append(_wrap(f"not read: {line}", "    "))
    if found["unread"]:
        out.append("")
    out.append("  Nothing above is a comparison with a published value. "
               "`benchmarks/` is where a number meets a reference, and `claims.py` is "
               "where it meets the figure it was drawn from.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path)
    parser.add_argument("--log", type=Path, default=None,
                        help="the solver log, where it is not in the case directory")
    parser.add_argument("--json", type=Path, default=None, dest="json_out",
                        help="also write the reading as JSON")
    args = parser.parse_args(argv)

    if not args.case.is_dir():
        raise SystemExit(f"no such case directory: {args.case}")
    found = explain(args.case, args.log)
    print(report(found))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
        print(f"\n{args.json_out}")
    # Zero whatever the verdict: this reports, it does not gate. `preflight.py` and
    # `claims.py` exit non-zero because they are checks somebody asked to be stopped
    # by; a reading of a finished run is not one, and a non-zero exit here would make
    # "explain the result" look like a failed command in every log that ran it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
