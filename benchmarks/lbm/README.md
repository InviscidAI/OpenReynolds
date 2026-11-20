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
straight wall, at any viscosity. The discrete solution is then the analytic parabola to
the convergence residual rather than to second order. The grader runs three viscosities a
factor of twenty apart, because one viscosity cannot tell "exact" from "exact at this
tau".

**This property is held by the measurement and not by a citation, deliberately.** The
result is attributed in the literature to Ginzburg and d'Humieres. The obvious reference
for it, Ginzburg, I. & d'Humieres, D. (2003), "Multireflection boundary conditions for
lattice Boltzmann models", *Phys. Rev. E* **68**, 066614, was checked and **its abstract
does not state it**: that paper is about multireflection boundary conditions, and its
nearest claim is that "Couette and Poiseuille flows are exact solutions of the lattice
Boltzmann models for a Reynolds number Re=0" using those conditions, which is a different
statement. The paper whose title names the "magic" collision numbers is d'Humieres, D. &
Ginzburg, I. (2009), *Computers and Mathematics with Applications* **58**, 823-840.
Neither full text was readable from here, so neither is quoted as though it had been, and
the citation is given as attribution rather than as evidence. `../README.md` is blunt
about what this repository has already paid for quoting a reference it had not read, and
the cheap alternative was available here: the property is checkable in seconds, so it is
checked.

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

## What it measured

On a GitHub Actions `ubuntu-latest` runner, so a floor rather than a flattering number.
The agreements and the step counts are deterministic and repeat exactly between runs; the
wall-clock times do not -- the same cavity run measured 103 s and 160 s on two runners, a
factor of 1.5 -- so the times below are what a slow shared machine gives and not a
property of the solver.

| run | agreement | cost |
|---|---|---|
| Poiseuille, TRT, nu = 0.02 (tau_plus 0.56) | max error **5.17e-11** of u_max | 478,600 steps, 100.4 s |
| Poiseuille, TRT, nu = 0.1 (tau_plus 0.80) | **1.22e-11** | 102,600 steps, 21.4 s |
| Poiseuille, TRT, nu = 0.4 (tau_plus 1.70) | **3.30e-12** | 27,200 steps, 5.7 s |
| Poiseuille, BGK, nu = sqrt(3)/12 | **9.04e-12** | -- |
| Poiseuille, BGK, nu = 4 sqrt(3)/12 | **3.66e-03**, and that is the point | -- |
| Cavity Re 1000, n = 128 | u RMS **0.736%**, v RMS **0.749%** of the lid speed | 123,500 steps, 103-160 s, 12.6-19.6 MLUPS |

The channel is exact to between 3e-12 and 5e-11 of u_max across a factor of twenty in
viscosity, and the residual it stopped at was 5e-14 -- so what is being measured there is
the convergence stop and not a discretisation error. **The BGK row is what makes that
mean something**: the same code with one relaxation time, four times away from its single
exact viscosity, is wrong by 3.7e-3 -- eight orders of magnitude worse, from a one-line
change in how the collision is split.

The cavity is worth comparing with the Felix row in `../README.md`, because the two were
graded against the same 34 numbers with the same 2% bound: Felix gives u RMS 0.448% and v
RMS 1.063%, this gives 0.736% and 0.749%. Neither is better than the other in any sense
that survives the next paragraph, which is the honest summary -- both are inside Ghia's
own uncertainty.

**Where the remaining difference comes from.** For the channel, the convergence stop, and
nothing else: the scheme has no discretisation error on this flow. For the cavity, in the
order they matter: Ghia's own error against a later spectral solution is 1.46% to 2.20%
on the centreline extrema, which is larger than the 0.74% being reported; 128 cells
against his 129x129 grid, so neither side is converged; and the residual stop at 9.3e-8,
which is now small enough to be the least of the three. The first of
those three dominates, and it means the right reading of this row is "agrees with Ghia to
inside Ghia's uncertainty", not "0.74% accurate".

## Cost

Every run prints its own step count, wall-clock time and rate in million lattice updates
per second, so the cost is measured on the machine that ran it rather than quoted from
here. The step count dominates and scales with the lattice: the residual has to fall
across the whole cavity, so steps grow roughly linearly in cells across while each step
grows as the square. Halving `--n` is therefore about eight times cheaper.
