# Lattice Boltzmann: plane Poiseuille, and the cavity at Re 1000

The two cases `openreynolds/toolbox/lbm.py` is validated on, and what each one is worth.
`grade.py` runs them:

```
python benchmarks/lbm/grade.py poiseuille
python benchmarks/lbm/grade.py cavity --n 128
```

Both run in the ordinary `benchmarks` job on every pull request. This grader **solves**,
where `../felix/grade.py` and `../moving_mesh/grade.py` read a finished run: those grade
an OpenFOAM or Felix case that cost a cluster an hour, so the run happens once and
trimmed output is committed under `tests/data/`. This solver is a few hundred lines of
numpy in this repository and both cases finish inside the test job, so committing its
output would replace a check that re-derives the number with a number nobody can
re-derive.

## Plane Poiseuille, which is exact

A periodic channel `n` cells tall driven by a uniform body force `G`, against

    u(y) = G/(2 mu) y (H - y),    u_max = G H^2 / (8 mu),    mu = rho nu

which is closed form and needs no paper. The force is set so `u_max` is 0.05 in lattice
units, which keeps the scheme's `O(Ma^2)` compressibility error small.

What makes this case worth having is that it is not graded as an approximation. With two
relaxation times and the product

    LAMBDA = (tau_plus - 1/2)(tau_minus - 1/2) = 3/16

the no-slip wall of a halfway bounce-back sits exactly on the lattice link midpoint for a
straight wall, at any viscosity -- Ginzburg, I. & d'Humieres, D. (2003), "Multireflection
boundary conditions for lattice Boltzmann models", *Phys. Rev. E* **68**, 066614. The
discrete solution is then the analytic parabola to the convergence residual rather than
to second order. The grader runs three viscosities a factor of twenty apart, because one
viscosity cannot tell "exact" from "exact at this tau".

**The contrast is graded too, and it is what stops the above being a tautology.** BGK is
the same code with `tau_minus = tau_plus`, which satisfies the 3/16 condition at exactly
one viscosity, `nu = sqrt(3)/12 = 0.14434`. The grader asserts that BGK is exact there
*and measurably wrong a factor of four away*, so a grader that passed because it was
comparing the solver with itself would fail. A validation whose reference is the thing
being validated is the failure mode `../README.md` was written about, and this is the
cheapest available guard against it.

## The lid-driven cavity at Re 1000

A square box with the top wall sliding, `nu = u_lid n / Re`, against Ghia, Ghia & Shin
(1982), *J. Comput. Phys.* **48**, 387-411, Table I (`u` along the vertical centreline)
and Table II (`v` along the horizontal centreline).

**Ghia's table is imported from `../felix/grade.py` rather than typed again.** Two copies
of a reference table is two things to keep right, and the copy already in this repository
has been through the re-checking `../README.md` describes -- which changed two of the
eleven reference values in that file. Importing it also means both solvers are graded
against the same 34 numbers with the same 2% RMS threshold, which is the only way a
comparison between them means anything.

The lattice carries node values at cell centres and the walls carry the boundary
condition, so the profile handed to the comparison is the nodes with the two wall values
on the ends -- `u = 1` at the lid, 0 at the stationary wall, `v = 0` at both sides -- and
is interpolated to Ghia's 17 stations. That is the same comparison `felix/grade.py`
makes, which samples its continuous field a nanometre inside the wall and reads the
boundary condition there too. The only interval where the linear interpolation is a real
approximation is the half cell between the last node and the wall, 0.4% of the cavity at
`n = 128`.

### What the agreement is worth, which is less than its digits suggest

**Ghia's own error is larger than the agreement being measured.** His Re 1000 centreline
extrema sit 1.46%, 1.59% and 2.20% from Botella & Peyret's (1998) spectral benchmark, so
anything much inside 2% has agreed to inside his uncertainty and part of it is luck. The
same caveat sits on the Felix row in `../README.md` and for the same reason. And neither
side of the comparison is the converged answer: Ghia solved on a 129x129 grid to second
order in 1982, and this runs on 128 cells across.

## Cost

Every run prints its own step count, wall-clock time and rate in million lattice updates
per second, so the cost is measured on the machine that ran it rather than quoted from
here. The step count dominates and scales with the lattice: the residual has to fall
across the whole cavity, so steps grow roughly linearly in cells across while each step
grows as the square. Halving `--n` is therefore about eight times cheaper.
