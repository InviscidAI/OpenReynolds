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

    with path.open("r", errors="replace") as handle:
        for line in handle:
            match = SOLVING.search(line)
            if match:
                field, initial, final, iters = match.groups()
                residuals[field].append((step, float(initial)))
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

    return {
        "residuals": residuals,
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


def report(data, log: Path, png: Path | None) -> str:
    lines = [f"# {log.name}"]
    times = data["times"]
    if times:
        lines.append(f"time steps parsed: {len(times)}  (last Time = {times[-1]:g})")
    if data["exec_time"] is not None:
        lines.append(f"ExecutionTime at last write: {data['exec_time']:g} s")

    if data["final_residual"]:
        lines.append("\n## last iteration")
        lines.append(f"{'field':<12}{'initial':>14}{'final':>14}{'iters':>8}")
        for field in sorted(data["final_residual"]):
            series = data["residuals"].get(field) or [(0, float('nan'))]
            lines.append(
                f"{field:<12}{series[-1][1]:>14.4e}"
                f"{data['final_residual'][field]:>14.4e}"
                f"{data['iterations'].get(field, 0):>8}"
            )

    if data["continuity"]:
        local, global_, cumulative = data["continuity"]
        lines.append("\n## continuity (most recent)")
        lines.append(f"sum local {local:.4e}   global {global_:.4e}   cumulative {cumulative:.4e}")

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
