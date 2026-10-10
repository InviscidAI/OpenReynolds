# Felix benchmarks

Two prompts with published answers and a grader for both, so a change to the Felix
prompt or tools is judged by what the solver produced rather than by how the session
read.

| prompt | what it tests | the reference |
|---|---|---|
| `cavity_re1000.txt` | a 2D lid-driven cavity at Re 1000 to a steady state: building a quasi-2D case for a 3D solver, steady-state control, sampling a solution along lines | Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, Table I (u on the vertical centreline) and Table II (v on the horizontal one), embedded in `grade.py` |
| `cylinder_re20.txt` | external flow with a drag coefficient: an unstructured mesh round a body, a parabolic inflow expression, a pressure outlet, force integration | Schafer & Turek (1996), DFG benchmark 2D-1: Cd = 5.5795 (interval 5.57-5.59), Cl 0.0104-0.0110 |

## Why the Schafer-Turek case for drag

The external-flow case had to be one Felix can run as it ships and one whose answer
does not depend on choices the prompt leaves open. 2D-1 is a channel, so the domain,
the blockage and the inflow are part of the benchmark's definition; an unbounded body
(the stock `sphere_re300` example, say) has a drag that moves with the domain size
and a reference that varies between sources by more than the tolerance. It is
steady at Re 20, so a PTC run settles in about a hundred steps and the answer is one
number, not a mean over shedding cycles. And it uses only what Felix documents: an
expression inflow (`boundary_conditions.md`, `expressions.md`), `slip` front and back
to make one cell layer behave as 2D, a pressure `dirichlet` outlet, and io.yaml
`forces:` on the cylinder group.

## Running one

```sh
openreynolds --solver felix -p "$(cat benchmarks/felix/cavity_re1000.txt)"
openreynolds --solver felix -p "$(cat benchmarks/felix/cylinder_re20.txt)"
openreynolds pull --study <study-id>          # the case directory, with its output/

python benchmarks/felix/grade.py cavity <case>
python benchmarks/felix/grade.py cylinder <case>
```

`--file` grades a snapshot instead of `output/solution.vtu`, `--nu` supplies the
viscosity when case.yaml's is not a plain number, and `--json` prints the result for
a machine. The exit status is 0 when every graded check passes, 1 when one fails,
and 2 when the case could not be read. The grader needs pyvista (it is in the
`dev` and `toolbox` extras).

## What the grader reads

Only what the solver wrote: `case.yaml` for `nu`, `solution.vtu` (in the output
directory io.yaml names) for the field and the geometry, and the forces CSV. The
lid is found as the face of the domain whose nodes move fastest, so a cavity with
its lid on z = 1 moving in -y is graded the same as one on y = 1 moving in +x; the
lid speed and side give Re. For the cylinder, the inflow peak is read on x = 0,
and the cylinder is the set of no-slip nodes off the channel walls, which gives its
diameter; the depth of the slab turns the force into a per-unit-depth coefficient.

The thresholds are in `THRESHOLDS` at the top of `grade.py`, each with where it came
from: 2% RMS of the lid speed on each cavity profile (the Felix-mode capstone's
bound), and 2% on Cd.

## The known result, 2026-10-07

Both were run on one A100 with the Felix solver directly (not through the agent), to
check that the cases are within Felix's reach and the grader reads them; the cut-down
outputs are the test fixtures in `tests/data/felix/`.

**Cavity, 128 x 128 x 1 hexes, slip front and back, PTC.** Steady at step 624
(about five minutes). u RMS 0.0045 of the lid speed, v RMS 0.0106; pass.

**2D-1, 6765 hexes one layer thick (cylinder edge 0.004), PTC.** Steady at step 111
(one minute). Cd 5.522, -1.04% of 5.5795, outside the 5.57-5.59 interval and inside
the 2% bound; Cl 0.0010 against 0.0104-0.0110 -- the lift needs a much finer mesh
round the cylinder, which is why it is reported and not graded. A finer mesh (29k
hexes, edge 0.0015) at the same PTC settings stalled its linear solver at 25 s per
step and was stopped; the coarse result is the one recorded.
