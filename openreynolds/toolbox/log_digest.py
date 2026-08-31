#!/usr/bin/env python3
"""Solver log -> residual plot, last-iteration table, continuity summary.

Reads a log of any size in one pass without holding it in memory. Reports numbers and
draws them; it does not say whether anything is converged — that reading is yours.

    python3 log_digest.py log.simpleFoam [-o residuals.png] [--csv residuals.csv]
"""

from __future__ import annotations

import argparse
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


def how_it_ended(data, end_time: float | None = None) -> list[str]:
    """One line saying what happened to the run, before any table of numbers.

    Nothing here read `FOAM FATAL`, `End`, or `solution converged`, so a solve that
    diverged at iteration 37 produced a normal-looking table headed "time steps parsed:
    37", and one that stopped at its iteration cap without converging looked exactly
    like one that had finished. Whether a residual is *low enough* is still a judgement
    about the case; whether the solver said it was finished is a fact, and it was missing.
    """
    times = data.get("times") or []
    reached = times[-1] if times else None
    if data.get("fatal"):
        return [f"ended: FOAM FATAL — {data['fatal']}"]
    if data.get("converged_at") is not None:
        return [f"ended: the solver reported convergence at iteration {data['converged_at']}"]
    if end_time is not None and reached is not None and reached < end_time:
        return [f"ended: stopped at {reached:g} of a requested {end_time:g}, "
                "and did not report convergence"]
    if data.get("ended"):
        return ["ended: ran to the end of controlDict without reporting convergence"]
    return ["ended: no End line — the log is still being written, or the run was cut off"]


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
