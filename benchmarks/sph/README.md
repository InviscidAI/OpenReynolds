# SPH: hydrostatic pressure, a sloshing period, and a dam break with no verdict

The three cases `openreynolds/toolbox/sph.py` is run on. Two are graded against closed
forms. The third is the one the solver was asked for, and it is reported without a
verdict; the reason is the whole subject of this file.

```
python benchmarks/sph/grade.py hydrostatic
python benchmarks/sph/grade.py sloshing
python benchmarks/sph/grade.py dambreak
```

Like `../lbm/grade.py`, this grader solves rather than reading a finished run, for the
same reason: the solver is in this repository and finishes inside the test job, so the
number is re-derived on every pull request instead of being quoted from a commit.

## Why the dam break is not graded

A dam break is the canonical SPH validation and the reason is that somebody measured one:
Martin, J. C. & Moyce, W. J. (1952), "Part IV. An experimental study of the collapse of
liquid columns on a rigid horizontal plane", *Phil. Trans. R. Soc. London A* **244**
(882), 312-324, doi 10.1098/rsta.1952.0006, which published surge-front position against
time for a column of width *a* and height *na* in dimensionless form.

**That paper could not be read.** The publisher returns HTTP 403 to an automated request
and no open repository copy was found. So three things that matter are unverified: which
non-dimensionalisation the paper actually defines (`T = t sqrt(2g/a)` and `T = t sqrt(g/a)`
are both in circulation in the secondary literature, and they differ by a factor of
1.414 on every point of the curve), which figure or table the data is in, and what the
experimental scatter was.

Quoting a number from that paper here would be quoting a secondary source. `../README.md`
has a long account of what that costs: of its eleven reference values, re-checking
changed two, and **both errors flattered the result**, which is why neither had been
noticed. One of them -- Gartling's reattachment length, read as step heights when the
paper states channel heights -- was a factor of two, and it reached a customer-facing
page. The same file's rule is that where the source could not be read, the absence is
stated instead of a figure being asserted.

So the surge front is **printed and carries no verdict**. `../README.md` already has two
rows that do exactly this: an amplitude with no published value for the configuration,
and a lift coefficient that needs resolution the drag does not.

**What would fix this** is an open-access dam-break experiment whose data is in the paper
rather than behind a request. Lobovsky, L. et al. (2014), "Experimental investigation of
dynamic pressure loads during dam break", *J. Fluids and Structures* **48**, 407-434 is
the modern candidate and is widely used for exactly this reason; whether its full text
and its front-position data are openly readable was not established here either, so it is
recorded as the next step and not as a reference.

## Hydrostatic pressure, which is exact

Still water of depth `h` in a tank, against `p = rho g (h - y)`. Everything this case
produces is error, which is what makes it the sharpest cheap check in the solver: it
exercises the equation of state, the density evolution and the wall treatment at once,
and the exact answer is arithmetic rather than an experiment.

Sampled over the middle half of the tank only. The ends carry the wall's own error, which
is a different quantity and is what the sloshing case measures instead. The worst
individual particle is always at the free surface, where the kernel support is truncated
and there is no fluid above to sum -- that is a property of SPH and not a defect of this
implementation, and it is graded on a separate, looser bound so that it is visible rather
than averaged away.

## The sloshing period, which is also exact

A first-mode standing wave in a closed tank of length `L` and depth `h`, against the
linear water-wave dispersion relation

    omega^2 = g k tanh(k h),    k = pi/L,    period = 2 pi / omega

which is classical closed form (Lamb, H., *Hydrodynamics*, 6th ed., art. 228). This
repository already grades against exact answers of this kind -- the Graetz eigenvalue on
the heated pipe, a closed-form torque on circular Couette -- so it is the established
convention here and not a substitute invented to avoid a paywall.

It is a real test of the boundary treatment and not only of the time integration. The
period depends on `k = pi/L`, so a wall that holds the fluid a fraction of a particle
spacing away from where the tank says it is changes `L` and changes the period. That is
precisely the error dynamic boundary particles have, and this is what sizes it.

Two choices in the setup are there to keep the reference honest:

* **The amplitude is 1% of the depth.** The reference is linear theory, so an amplitude
  large enough to be easy to measure would be compared against the wrong formula.
* **It is released from the mode's own velocity field with a flat surface**, so the
  particle lattice starts uniform. Displacing the surface instead would start the run
  with a disturbed lattice and measure that as well as the period.

The period is measured from the fluid's **horizontal centroid**, not a surface probe:
every particle contributes to it, and it does not care which particle happens to be
highest when the sample is taken. `_period` interpolates each zero crossing and averages
over all of them, and returns `nan` rather than a number if there are fewer than three.

## What the dam break *is* graded on

Two conditions that decide whether the scheme was valid during the run, both of which
fail silently otherwise:

* **Containment.** Dynamic boundary particles are the cheap boundary treatment and this
  is the failure they have: a fast particle passes through a wall thinner than the kernel
  support and the run continues without complaint. Three wall layers against a `2h = 2.6
  dx` support, and the check is the deepest penetration in particle spacings.
* **Weak compressibility.** Tait's equation of state is a linearisation about `rho0`
  dressed up as a seventh power. At a few percent compression the pressure it returns is
  still water's; well past that it is a numerical fluid's. A run that exceeded this did
  not fail to match a reference -- it stopped being a simulation of water, and no
  agreement it reports afterwards means anything.

## What it measured

On a GitHub Actions `ubuntu-latest` runner, at the configuration the graders ship with:

| run | what is graded | agreement | cost |
|---|---|---|---|
| hydrostatic, depth 0.1 m, dx = h/20 | RMS pressure error / rho g h | **2.50%** | 800 fluid + 318 wall, 15,238 steps, 285 s |
| " | worst single particle | **8.61%**, at the free surface | " |
| sloshing, 1.0 x 0.5 m, dx = h/20 | period against `2 pi / sqrt(g k tanh(k h))` | **-4.75%** (1.1257 s against 1.1818 s) | 800 fluid + 330 wall, 12,081 steps, 225 s |
| dam break, a = 0.1 m, n = 2 | **nothing**; surge front reported only | Z = 0.975 at T = 0, 3.943 at T = 3.00 | 800 fluid + 606 wall, 4,616 steps, 105 s |
| all three | weak compressibility, `max rho/rho0 - 1` | **0.25%, 1.03%, 0.25%** | -- |
| all three | wall penetration, in particle spacings | **none**: -0.30, -0.19, -0.33 | -- |

**Where the difference comes from, hydrostatic.** The density noise, multiplied by
`c0^2`. Tait is stiff -- `dp/drho` is about `c0^2` -- so at `c0 = 20 sqrt(g h)` a density
error of `drho/rho` is a pressure error of `400 drho/rho` as a fraction of the `rho g h`
being measured. Before the density diffusion was added this read 67%, with operators that
balance gravity to 3% and a density rate identically zero on a static field: nothing was
wrong with the discretisation, the noise was simply being amplified. The worst particle
is always at the free surface, where the kernel support is truncated and there is no
fluid above to sum, which is a property of SPH and not of this implementation -- hence a
separate, looser bound on it rather than averaging it into the RMS.

**Where the difference comes from, sloshing: the sign is the diagnosis.** The period
comes out **short**, and short means the effective tank is shorter than the one the case
asked for -- `omega^2 = g k tanh(k h)` with `k = pi/L`, so a wall that holds the fluid a
fraction of a particle spacing inside where the tank says it is raises `k` and raises the
frequency. That is exactly the error dynamic boundary particles have, and -4.75% at
`dx/L = 0.025` is the right order for it. Two smaller terms sit underneath: the amplitude
is 5% of the depth, whose first nonlinear correction to the frequency is `O((k a)^2)`,
about 0.6%, and 20 particles across the depth is a coarse resolution of a dispersion
relation. The resolution trend is the obvious next measurement and is not in this table.

**A measurement artefact, in the dam break, that is not physics.** Z starts at 0.975 and
not at 1, because the front is the outermost particle's *centre* and that sits half a
spacing inside the column's edge: `1 - dx/(2a)`, exactly 0.025 at the shipping spacing.
It is worth knowing before it is read as a slow start.

## Cost

Every run prints its particle count, step count, wall-clock time and steps per second.
The time step is acoustic and not physical -- `dt = 0.2 h / c0` with `c0` ten to twenty
times the fastest flow -- so a tenth of a second of a dam break is thousands of steps
however few particles there are. That floor is the scheme's and not this implementation's.
`--dx` is the knob and it is expensive in both directions at once: halving it doubles the
particles per metre in two directions and halves the time step, so a fixed physical
duration costs as `1/dx^3`.
