#!/usr/bin/env python3
"""The lid-driven cavity at Re 100, graded against Ghia, Ghia and Shin (1982).

    python examples/live/cavity.py

What it shows: the graded case `openreynolds live run cavity` runs, in Python, at a
resolution a laptop can afford. It calls `openreynolds.live.cases.run_cavity` rather
than building the lattice itself, because the grading is the point of this one -- the
straddling-line centreline extraction, Ghia's two tables, and the two boundary rows
that come out of the RMS all live in there and are worth reading rather than copying.
`sphere_in_a_stream.py` is the example that drives `FluidWorld` directly.

What it costs: 32 cells a side is 3,072 nodes (the cavity is quasi-two-dimensional:
three lattice planes, periodic in z), and 20,000 lattice steps of it took **34 s** on
one CPU core of the machine this was written on, at 2.1 MLUPS. A GPU runs the same
kernels; `openreynolds live bench` is what measures the rate on yours.

What it is not: 32 cells is not the graded resolution. `CASES["cavity"]` defaults to
128 and `benchmarks/live/` runs it at 128 and 256, because one coarse grid that agrees
with a published table is evidence that one grid agreed. What 32 cells is good for is
seeing the whole path work in under a minute -- and the agreement it gets (u RMS 0.22%,
v RMS 0.26% of the lid speed, against a 2% bound) is better than it deserves for a
reason `REFERENCES["ghia"]`'s own caveat states: Ghia's numbers carry more error than
that.

Two PNGs are written into the current directory, and their paths are printed.
"""

from pathlib import Path

from openreynolds.live.cases import report, run_cavity

CELLS = 32
REYNOLDS = 100.0
STEPS = 20_000
"""Steps rather than a convergence tolerance, so the cost of this file is a number its
docstring can state. `run_cavity`'s own default is to run until the field stops moving
-- `cases._converge`, whose residual floor on this case is single precision and not the
scheme -- which is the right default for a graded run and the wrong one for an example
that promises to finish."""


def main() -> int:
    result = run_cavity(cells=CELLS, reynolds=REYNOLDS, steps=STEPS,
                        out_dir=Path.cwd(), progress=print)
    print(report(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
