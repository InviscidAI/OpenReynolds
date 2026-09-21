#!/usr/bin/env python3
"""Solver log -> residual plot, last-iteration table, continuity summary.

Reads a log of any size in one pass without holding it in memory. Reports numbers and
draws them, and says what the residuals did -- levelled off, still falling, climbing --
because those are three different facts about a run and only one of them is a failure.
Whether a level is low enough for the question is still yours to judge.

    python3 log_digest.py log.simpleFoam [-o residuals.png] [--csv residuals.csv]
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOLVING = re.compile(
    r"Solving for (\S+?),\s*Initial residual = ([0-9.eE+-]+),"
    r"\s*Final residual = ([0-9.eE+-]+),\s*No Iterations (\d+)"
)
TIME = re.compile(r"^Time = ([0-9.eE+-]+)\s*$")
CONTINUITY = re.compile(
    r"time step continuity errors : sum local = ([0-9.eE+-]+), "
    r"global = ([0-9.eE+-]+), cumulative = ([0-9.eE+-]+)"
)
BOUNDING = re.compile(r"^bounding (\S+),")
COURANT = re.compile(r"Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)")
EXEC_TIME = re.compile(r"ExecutionTime = ([0-9.eE+-]+) s")
FATAL = re.compile(r"^--> FOAM FATAL (ERROR|IO ERROR)")
CONVERGED = re.compile(r"solution converged in (\d+) iterations")
END = re.compile(r"^End\s*$")


def digest(path: Path):
    residuals: dict[str, list[tuple[int, float]]] = defaultdict(list)
    final_residual: dict[str, float] = {}
    iterations: dict[str, int] = {}
    bounding = defaultdict(int)
    continuity = None
    courant = None
    times: list[float] = []
    exec_time = None
    step = 0
    seen_at: dict[str, set] = {}
    outer_residual: dict[str, float] = {}
    continuity_series: list[tuple[int, float]] = []
    fatal: str | None = None
    converged_at: int | None = None
    ended = False

    with path.open("r", errors="replace") as handle:
        for line in handle:
            match = SOLVING.search(line)
            if match:
                field, initial, final, iters = match.groups()
                # The FIRST solve of a field in a step is the outer-loop residual, which
                # is what measures convergence. `step` only advances on a `Time =` line,
                # so every PIMPLE inner corrector landed on the same step and the last
                # one -- typically one to two orders lower -- was reported as that step's
                # residual. A transient table looked immaculate whether or not the outer
                # loop had converged at all.
                if step not in seen_at.get(field, ()):
                    residuals[field].append((step, float(initial)))
                    seen_at.setdefault(field, set()).add(step)
                    outer_residual[field] = float(initial)
                final_residual[field] = float(final)
                iterations[field] = int(iters)
                continue
            match = TIME.match(line)
            if match:
                times.append(float(match.group(1)))
                step += 1
                continue
            match = CONTINUITY.search(line)
            if match:
                continuity = tuple(float(v) for v in match.groups())
                # The whole series, not only the last one. The field notes list a
                # *growing* cumulative continuity error as a failure signature, and
                # keeping one value made exactly that invisible.
                continuity_series.append((step, continuity[2]))
                continue
            match = COURANT.search(line)
            if match:
                courant = tuple(float(v) for v in match.groups())
                continue
            match = BOUNDING.match(line)
            if match:
                bounding[match.group(1)] += 1
                continue
            match = EXEC_TIME.search(line)
            if match:
                exec_time = float(match.group(1))
                continue
            # How the run ended is the first thing anybody asks and nothing here read
            # it: a solve that died at iteration 37 printed a normal-looking table
            # headed "time steps parsed: 37", indistinguishable from one that finished.
            if FATAL.match(line):
                fatal = line.strip()
                continue
            match = CONVERGED.search(line)
            if match:
                converged_at = int(match.group(1))
                continue
            if END.match(line):
                ended = True

    return {
        "residuals": residuals,
        "outer_residual": outer_residual,
        "continuity_series": continuity_series,
        "fatal": fatal,
        "converged_at": converged_at,
        "ended": ended,
        "final_residual": final_residual,
        "iterations": iterations,
        "bounding": dict(bounding),
        "continuity": continuity,
        "courant": courant,
        "times": times,
        "exec_time": exec_time,
    }


def plot(residuals, out: Path) -> None:
    if not residuals:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for field, series in sorted(residuals.items()):
        steps = [s for s, _ in series]
        values = [max(v, 1e-30) for _, v in series]
        ax.semilogy(steps, values, label=field, linewidth=1.0)
    ax.set_xlabel("outer iteration / time step")
    ax.set_ylabel("initial residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


DIVERGED = 1e3
"""An initial residual above this is not a slow convergence, it is a blow-up."""

RISE = 10.0
"""How much worse than its own best a residual has to get to count as climbing."""

FLOOR = 1e-6
"""Below this a residual has done its job and the ratio to its own minimum means
nothing: 3e-11 against a best of 1e-12 is thirty times its best and going nowhere."""

STILL_FALLING = 2.0
"""A last window this many times lower than the window before it is a residual that
is still coming down; less than that, and it has levelled off."""

MIN_STEPS = 8
"""Fewer steps than this and there is no trend to read, only numbers."""


def _geometric_mean(values: list[float]) -> float:
    positive = [v for v in values if v > 0 and math.isfinite(v)]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def residual_shape(residuals: dict, window: int | None = None) -> dict:
    """Why the residuals stopped falling -- or whether they have.

    A run that ends says one of four things about its residuals, and each wants
    different words: `diverging` (a field climbing off its own best, or past any
    sensible value), `falling` (the last window still lower than the one before it),
    `levelled` (flat: a plateau, which under a steady solver on an unsteady flow is the
    flow being reported, not a defect), or `short` (too few steps to say). Per field,
    and overall as the worst of them: diverging beats falling beats levelled.

    `level` is the geometric mean of each levelled field's last window, and `since` is
    the step from which the series last stayed within a factor of three of that level,
    so the report can say "levelled off at ~2.5e-2 from step ~250" -- which is what a
    person reading the plot would say, and what the digest said nothing about.
    """
    shape: dict = {"shape": "short", "fields": {}, "level": {}, "best": {}, "last": {}, "since": {}}
    ranking = {"diverging": 3, "falling": 2, "levelled": 1, "short": 0}
    for field, series in sorted(residuals.items()):
        values = [float(v) for _step, v in series]
        steps = [s for s, _v in series]
        if not values:
            continue
        last = values[-1]
        shape["last"][field] = last
        finite = [v for v in values if math.isfinite(v)]
        best = min(finite) if finite else float("nan")
        shape["best"][field] = best
        if not math.isfinite(last) or last > DIVERGED or (
            len(values) > 3 and last > FLOOR and last > best * RISE
        ):
            kind = "diverging"
        elif len(values) < MIN_STEPS:
            kind = "short"
        else:
            size = window or max(4, len(values) // 5)
            recent = _geometric_mean(values[-size:])
            before = _geometric_mean(values[-2 * size:-size])
            if math.isfinite(recent) and math.isfinite(before) and recent * STILL_FALLING <= before:
                kind = "falling"
            else:
                kind = "levelled"
                shape["level"][field] = recent
                since = len(values) - 1
                while since > 0 and recent / 3 <= values[since - 1] <= recent * 3:
                    since -= 1
                shape["since"][field] = steps[since]
        shape["fields"][field] = kind
        if ranking[kind] > ranking[shape["shape"]]:
            shape["shape"] = kind
    return shape


def _residual_reading(data) -> str:
    """The residuals in one sentence whose words depend on why they stopped falling.

    The old line said "did not report convergence" whatever the reason, and a person
    who heard it on five studies concluded that nothing converges. A plateau, a climb
    and a series still on its way down are three different facts about a run, and only
    one of them is a failure."""
    shape = residual_shape(data.get("residuals") or {})
    kind = shape["shape"]
    if kind == "diverging":
        worst = [f for f, k in shape["fields"].items() if k == "diverging"]
        detail = ", ".join(
            f"{f} best {shape['best'][f]:.1e}, last {shape['last'][f]:.1e}" for f in worst
        )
        return (f"residuals: climbing ({detail}) -- this run is diverging, not converging "
                "slowly, and the last field is not one to show; the usual causes are the "
                "mesh where the field is worst, a boundary condition, or the timestep")
    if kind == "falling":
        moving = [f for f, k in shape["fields"].items() if k == "falling"]
        detail = ", ".join(f"{f} {shape['last'][f]:.1e}" for f in moving)
        return (f"residuals: still falling ({detail}) -- more iterations would tighten "
                "the answer; nothing here says it is wrong")
    if kind == "levelled":
        flat = [f for f, k in shape["fields"].items() if k == "levelled"]
        detail = ", ".join(f"{f} ~{shape['level'][f]:.1e}" for f in flat)
        since = min(shape["since"][f] for f in flat)
        return (f"residuals: levelled off ({detail}) from about step {since:g} and stayed "
                "there -- a plateau, not a divergence. Under a steady solver on a flow that "
                "is unsteady (a bluff body, a shedding wake) this is the residual reporting "
                "the flow, and the last field is a usable snapshot of it; a plateau on a "
                "flow that should be steady points at the mesh or a boundary condition")
    return ""


def how_it_ended(data, end_time: float | None = None) -> list[str]:
    """What happened to the run, before any table of numbers, in words that depend
    on why: how it stopped, then what its residuals were doing when it did.

    Nothing here read `FOAM FATAL`, `End`, or `solution converged`, so a solve that
    diverged at iteration 37 produced a normal-looking table headed "time steps parsed:
    37", and one that stopped at its iteration cap without converging looked exactly
    like one that had finished. Then it said "did not report convergence" about every
    run that reached its end without the solver's own message -- the same words for a
    residual levelled off on a shedding wake (the physics) as for one climbing towards
    a floating point exception (a failure), and a person who read that on five studies
    heard that nothing converges. The end is still a fact; what the residuals did is
    read from the series (`residual_shape`) and said in its own terms.
    """
    times = data.get("times") or []
    reached = times[-1] if times else None
    if data.get("fatal"):
        return [f"ended: FOAM FATAL — {data['fatal']} (the solver stopped itself: this run failed)"]
    if data.get("converged_at") is not None:
        return [f"ended: the solver reported convergence at iteration {data['converged_at']}"]
    if end_time is not None and reached is not None and reached < end_time:
        lines = [f"ended: stopped at {reached:g} of a requested {end_time:g}"]
    elif data.get("ended"):
        lines = ["ended: ran to the end of controlDict"]
    else:
        lines = ["ended: no End line — the log is still being written, or the run was cut off"]
    reading = _residual_reading(data)
    if reading:
        lines.append(reading)
    return lines


def requested_end_time(log: Path) -> float | None:
    """`endTime` from the controlDict beside this log, when there is one to read."""
    for candidate in (log.parent / "system" / "controlDict",
                      log.parent.parent / "system" / "controlDict"):
        try:
            text = candidate.read_text(errors="replace")
        except OSError:
            continue
        match = re.search(r"^\s*endTime\s+([0-9.eE+-]+)\s*;", text, re.M)
        if match:
            return float(match.group(1))
    return None


def report(data, log: Path, png: Path | None) -> str:
    lines = [f"# {log.name}"]
    lines += how_it_ended(data, requested_end_time(log))
    times = data["times"]
    if times:
        lines.append(f"time steps parsed: {len(times)}  (last Time = {times[-1]:g})")
    if data["exec_time"] is not None:
        lines.append(f"ExecutionTime at last write: {data['exec_time']:g} s")

    if data["final_residual"]:
        lines.append("\n## last iteration")
        lines.append("initial is the OUTER-loop residual, which is what measures convergence;")
        lines.append("final is after the inner correctors and reaches its relTol by construction.")
        lines.append(f"{'field':<12}{'initial':>14}{'final':>14}{'iters':>8}")
        for field in sorted(data["final_residual"]):
            outer = data.get("outer_residual", {}).get(field)
            if outer is None:
                series = data["residuals"].get(field) or [(0, float('nan'))]
                outer = series[-1][1]
            lines.append(
                f"{field:<12}{outer:>14.4e}"
                f"{data['final_residual'][field]:>14.4e}"
                f"{data['iterations'].get(field, 0):>8}"
            )

    if data["continuity"]:
        local, global_, cumulative = data["continuity"]
        lines.append("\n## continuity (most recent)")
        lines.append(f"sum local {local:.4e}   global {global_:.4e}   cumulative {cumulative:.4e}")
        series = data.get("continuity_series") or []
        if len(series) >= 4:
            # A *growing* cumulative error is a documented failure signature, and keeping
            # only the most recent value made exactly that impossible to see.
            first, last = series[0][1], series[-1][1]
            trend = ("growing" if abs(last) > abs(first) * 2
                     else "shrinking" if abs(last) * 2 < abs(first) else "steady")
            lines.append(f"cumulative over the run: {first:.4e} -> {last:.4e}  ({trend})")

    if data["courant"]:
        lines.append(f"\nCourant number — mean {data['courant'][0]:g}, max {data['courant'][1]:g}")

    if data["bounding"]:
        lines.append("\n## bounding messages")
        for field, count in sorted(data["bounding"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {field}: {count}")

    if png:
        lines.append(f"\nresidual plot: {png}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None, help="residual plot path")
    parser.add_argument("--csv", type=Path, default=None, help="also write the residual series")
    args = parser.parse_args()

    data = digest(args.log)
    png = args.out or args.log.with_suffix(".residuals.png")
    plot(data["residuals"], png)

    if args.csv:
        import pandas as pd

        frame = pd.DataFrame(
            {field: pd.Series(dict(series)) for field, series in data["residuals"].items()}
        )
        frame.index.name = "step"
        frame.to_csv(args.csv)

    print(report(data, args.log, png if data["residuals"] else None))


if __name__ == "__main__":
    main()
