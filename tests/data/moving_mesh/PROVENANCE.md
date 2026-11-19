# `tests/data/moving_mesh/` -- real solver output, cut down to what the grader reads

Five cases from the two moving-mesh runs of 2026-09-12, the runs whose numbers
`benchmarks/moving_mesh/README.md` records under "The known result". They are here so
`benchmarks/moving_mesh/grade.py` is exercised against output a solver actually wrote,
not only against the synthetic `.dat` files `tests/test_benchmark_moving_mesh.py`
writes. Those synthetic fixtures stay: they can be made to hold a restart overlap or a
swapped column on purpose, and real output cannot.

| directory | the run it came from | what it is |
|---|---|---|
| `couette/` | `couette` | the 2D circular-Couette annulus, 50 x 240, inner wall turning, cut by a `cyclicAMI` sliding interface |
| `couette_conformal/` | `noAMI` | the same annulus meshed in one piece: the control the sliding interface is measured against |
| `viv_fixed/` | `fixed` | the fixed cylinder at Re = 100, whose shedding frequency is the Strouhal gate |
| `viv_loose/` | `viv` | the cylinder released on its spring, body solved outside the PIMPLE loop |
| `viv_tight/` | `tight` | the continuation with the body inside the PIMPLE loop and no acceleration relaxation |

## What was kept

Only the files the grader opens: `postProcessing/<fo>/<time>/*.dat`,
`constant/polyMesh/points`, `constant/transportProperties`,
`constant/dynamicMeshDict`, `system/controlDict`, and -- for the AMI weight check --
the solver log. Byte-identical to the run's own files, with the two exceptions below.

**The solver log is grepped, not copied.** `couette/log.pimpleFoam` is the run's 13 MB
`log.pimpleFoam` reduced to the two kinds of line `_ami_weights` reads: the
`sum(weights)` lines and the `Time = ` lines. All 8,152 weight lines are there and the
last `Time` is still 20, so the check reports exactly what it reported on the whole log.

**Three `.dat` files start later than the solver started them.** The VIV grader measures
the last few cycles, so the early rows are dead weight: `viv_fixed/.../26`,
`viv_loose/.../75` and `viv_tight/.../206.2` were cut to their rows at or after
t = 90, 150 and 230 with their headers intact. Each case still holds a whole cycle
margin beyond the window the grader takes, and `viv_loose` still holds the
`75`/`200` restart pair with its overlap, so the stitching is still under test.

Every number the two graders print from these five directories is identical, digit for
digit, to what they print from the full cases -- which is the check that the trim took
nothing load-bearing. The recorded values are asserted in
`tests/test_benchmark_moving_mesh.py`.

## What was dropped, and why none of it can come back

The two studies these came from are 348 MB and 666 MB of fields, meshes, renders and
restart directories, and each names the account that ran it, the instance it ran on
and what it cost. `OpenReynolds/.gitignore` keeps studies out
of the repository for that reason. Nothing here came from a study's own analysis,
`REPORT.md` or JSON summaries either, on the same principle the grader itself follows:
grading a run's arithmetic grades the arithmetic.

What is left is solver output and OpenFOAM dictionaries. The whole directory holds no
path, host, account, identifier or cost: the only names in it are OpenFOAM's own
(`pimpleFoam`, `innerWall`, `cylinder`, `transverseSpring`).

## Regenerating it

With the two study directories at `$A` (`couette`, `noAMI`) and `$B` (`fixed`, `viv`,
`tight`):

Copy, per case, `system/controlDict`, whichever of
`constant/{transportProperties,dynamicMeshDict,polyMesh/points}` the case has, and
every `postProcessing/<fo>/<time>/` directory of the function objects below, keeping
only the one file named:

| directory | function objects, and the file taken from each |
|---|---|
| `couette`, `couette_conformal` | `forces/moment.dat`, `torqueOuter/moment.dat` |
| `viv_fixed` | `forceCoeffs/coefficient.dat` |
| `viv_loose`, `viv_tight` | `forceCoeffs/coefficient.dat`, `bodyState/sixDoFRigidBodyState.dat` |

`couette_conformal` has no `dynamicMeshDict` (its wall moves by a boundary condition),
which is why the grader reports that its omega came from the graded case. `viv_fixed`
keeps only restarts `26` and `126`, `viv_loose` only `75` and `200`.

Then the log, and the three cuts:

```sh
D=tests/data/moving_mesh
grep -E '^Time = |sum\(weights\)' $A/couette/log.pimpleFoam > $D/couette/log.pimpleFoam

# cut_at <file> <t0>: drop the rows before t0, keep the header
cut_at() { awk -v t0="$2" '/^#/ {print; next} $1+0 >= t0 {print}' "$1" > "$1.cut" && mv "$1.cut" "$1"; }

cut_at $D/viv_fixed/postProcessing/forceCoeffs/26/coefficient.dat 90
for f in $D/viv_loose/postProcessing/forceCoeffs/75/coefficient.dat \
         $D/viv_loose/postProcessing/bodyState/75/sixDoFRigidBodyState.dat; do cut_at "$f" 150; done
for f in $D/viv_tight/postProcessing/*/206.2/*; do cut_at "$f" 230; done
```

Then check the trim with the graders themselves:

```sh
python benchmarks/moving_mesh/grade.py couette $D/couette --control $D/couette_conformal
python benchmarks/moving_mesh/grade.py viv $D/viv_fixed $D/viv_loose $D/viv_tight
```

Both exit 1, and the failing check is the finding in each case: the sliding interface's
0.91% torque imbalance, and the loosely coupled body's 10% net fluid work per cycle.
