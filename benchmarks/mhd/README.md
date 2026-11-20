# Hartmann flow: a conducting liquid across a magnetic field

What `mhdFoam` is held to, where the answer comes from, and what a real run measured.

The case is `openreynolds/toolbox/templates/mhd_hartmann.py`, the grader is `grade.py`
beside this file, and the solver is held to both by `tests/test_journey_mhd.py` in the
`journey` job, which meshes, solves and grades inside the workspace image.

## Why this case and not a liquid-metal experiment

Because its answer is a formula. The first argument in
[`benchmarks/README.md`](../README.md) is that the reference value is what goes wrong
most often and is last on everyone's list: of the eleven published reference numbers in
that file, re-checking two against their actual sources changed them, and one of the two
was wrong by a factor of two. A closed form derived from the solver's own equations has
nothing to misquote, no secondary citation to inherit and no experimental uncertainty to
agree with by luck. `benchmarks/thermal_stress/` was chosen on the same grounds.

What that choice costs is worth stating: this case tells you that `mhdFoam` solves the
equations it claims to. It does not tell you that incompressible, laminar,
single-fluid MHD with a uniform scalar conductivity is the right model for a real
liquid-metal duct, and nothing here claims it does.

## The derivation

From `applications/solvers/electromagnetics/mhdFoam/mhdFoam.C` in OpenFOAM v2512, with
`DBU = 1/(2*mu*rho)` and `DB = 1/(mu*sigma)` out of `createFields.H` in the same
directory:

```
ddt(U) + div(phi,U) - div(phiB, 2*DBU*B) - laplacian(nu,U) + grad(DBU*magSqr(B))
    == -grad(p)
ddt(B) + div(phi,B) - div(phiB,U) - laplacian(DB,B) == 0
```

`p` is kinematic (`0/p` carries `[0 2 -2 0 0 0 0]`). Since `2*DBU = 1/(mu*rho)`, the two
magnetic terms in the momentum equation are the Lorentz force per unit mass,
`((B.grad)B - grad(|B|^2/2))/(rho*mu)`, written as a flux divergence minus a magnetic
pressure gradient. `mu` is the magnetic **permeability** in H/m, not a dynamic
viscosity: the dimension set `createFields.H` constructs it with is
`[1 1 -2 0 0 -2 0]`. `sigma` is the electrical conductivity in S/m.

Take flow along x between walls at `y = -L` and `y = +L`, fully developed so nothing
depends on x, with a uniform applied field `B0` along y. The velocity is `u(y)*xhat`;
the field is `b(y)*xhat + B0*yhat`, because the flow drags the transverse field into an
x-component.

**Induction.** With both fields divergence free,
`div(U B) - div(B U) = (U.grad)B - (B.grad)U`. Nothing depends on x, so `(U.grad)B = 0`
while `(B.grad)U = B0*du/dy`. The x-component of the steady induction equation is then
`-B0*du/dy - DB*d2b/dy2 = 0`, and integrating once with `DB = 1/(mu*sigma)`:

```
db/dy = -mu*sigma*B0*u + C                                               (1)
```

**Momentum.** `(B.grad)B = B0*(db/dy)*xhat`, and `grad(|B|^2/2)` has no x-component
since `b` and `B0` depend on y alone. So

```
0 = -dp/dx + nu*d2u/dy2 + B0*(db/dy)/(rho*mu)
```

and substituting (1) turns the Lorentz term into a term proportional to `u`:

```
nu*d2u/dy2 - (sigma*B0^2/rho)*u = dp/dx - B0*C/(rho*mu)  =  constant
```

That is the whole of it. **In a transverse field the Lorentz force on this flow is a
linear drag**, and the equation is the Poiseuille equation with a `-k*u` added. Writing
`Ha = B0*L*sqrt(sigma/(rho*nu))`, it is `u'' - (Ha/L)^2*u = const`, whose solution with
`u(-L) = u(L) = 0` and even symmetry is a constant minus a cosh:

```
u(y)/u_c   = ( cosh(Ha) - cosh(Ha*y/L) ) / ( cosh(Ha) - 1 )              (2)
u_c/u_mean = ( cosh(Ha) - 1 ) / ( cosh(Ha) - sinh(Ha)/Ha )               (3)
```

(3) follows from integrating (2) across the channel. The thickness of the two layers
is `L/Ha`, which is the length the `cosh` decays over.

### What each of the two measures, and why both

(2) contains Ha and **nothing else** -- not the pressure gradient, not the viscosity,
not the flow rate, not even L except through `y/L`. So a solved profile divided by its
own centreline value lies on (2) with no constant fitted anywhere. That makes it a
statement about the shape alone, which passes or fails independently of whether the case
conserved mass.

(3) is **absolute**. The template's inlet is a plug at `u_mean` and the case is 2D with
both z faces `empty`, so continuity fixes the mean at every station exactly and the
centreline speed is `u_mean*(3)` outright. Values:

| Ha | u_c/u_mean | L/Ha |
|---|---|---|
| 0 | 1.5 (Poiseuille) | - |
| 5 | 1.23312791 | 0.200 |
| 10 | 1.11101022 | 0.100 |
| 20 | 1.05263157 | 0.050 |
| 40 | 1.02564103 | 0.025 |

At Ha 0, (2) degenerates to `1 - (y/L)^2` and (3) to 3/2, which is the limit the
derivation has to reproduce and does. For large Ha, (3) approaches `Ha/(Ha-1)`: a flat
core with two thin layers, the core moving barely faster than the average. **The
flattening is the physics**, and it is 29% between Ha 0 and Ha 20.

## What the grader checks, and what each check is responsible for

`grade.py hartmann <case> --hartmann H --mean-velocity U --half-height L`. It reads the
`raw` profile `postProcess -func sample` wrote and nothing else.

| check | what it is responsible for on its own |
|---|---|
| `samples`, `y_range` | enough points, and a line that reaches **both** walls. Half a channel is symmetric about its own end and smooth, so a reach taken as `max|y|` would pass it and every check after it would be grading half a profile against a whole closed form |
| `symmetry` | `max\|u(y) - u(-y)\|/u_c`. A flow still developing is asymmetric while its shape error can still be small, which separates "not developed" from "wrong profile" |
| `shape_rms`, `shape_max` | the profile divided by its own centreline against (2). No constant fitted |
| `flow_rate` | the trapezoidal mean of the profile against the stated inlet mean -- the assumption `centreline` rests on, checked rather than assumed |
| `centreline` | the measured centreline against `u_mean*(3)`, absolute |
| `parabola_contrast` | the same profile against the Ha 0 parabola, which has to be **very much worse**. Reported without a verdict where Ha is small enough that the two curves are inside the shape threshold of each other, since there is then no contrast to measure |

Ha, the mean velocity and the half-height are **required flags and are not inferred**.
Ha needs B0, sigma, rho and nu together; the mean velocity is the inlet condition; and L
cannot be read off the sampled line, because the template's line stops a thousandth of a
half-height short of each wall and the profile's slope there is about 20 `u_c` per L at
Ha 20 -- inferring L from the samples would be a 2% error exactly where the check is
tightest. `heat.py` and `balance.py` refuse a missing constant by the flag that would
supply it for the same reason.

## The mesh, the time step, and what Ha costs

`blockMesh`, not gmsh, which is what every other template in this repository uses: the
domain is a box, so gmsh would contribute a recombination and a non-orthogonality to
argue about and nothing else.

The Hartmann layer is `L/Ha` thick and the profile's entire departure from a parabola
lives in it, so resolving it is the one mesh fact that decides whether the case can be
graded. **The obvious way to do that does not work, and this is the most useful thing
this case has produced.**

### The graded mesh that diverged

The ESI tutorial runs Ha 20 on a uniform 40-cell mesh, which puts **one** cell inside
its `L/Ha = 0.05` layer. One cell cannot resolve the only interesting part of the
profile, so the first version of this template graded toward the walls: `simpleGrading`
multi-graded, reciprocal expansion ratios of 10, 80 cells, a 0.00633 m wall cell and
seven cells inside the layer. That is the right instinct for a boundary layer, and
`mhdFoam` died on it inside eighty time steps with a SIGFPE in
`symGaussSeidelSmoother` on the vector equation. From the `journey` job's log:

```
Time = 0.07
Courant Number mean: 0.4238678571 max: 23.59931818
```

**A Courant number of 23.6 cannot come from the flow.** The advective one on that mesh
is `u*dt/dx = 1.05*0.005/0.2 = 0.026` everywhere, and 23.6 needs a transverse velocity
of about 30 m/s. 30 m/s is the **Alfven speed**, `c_A = B0/sqrt(mu*rho) = 20 m/s`, which
names what was growing. The `pB` projection in the same log was converging cleanly to
1e-08, so nothing was wrong with the mesh or the properties as such.

The solver documents the mechanism without being asked. From `mhdFoam.C`'s own
`Description` block:

> However, in this formulation discretisation error prevents the normal stresses in UB
> from cancelling with those from BU, but it is unknown whether this is a serious error.

Both of those stresses are `fvc::` and not `fvm::` -- `fvc::div(phiB, 2.0*DBU*B)` and
`fvc::grad(DBU*magSqr(B))`. Each is of order `B0^2/(mu*rho*dy)` alone and they are meant
to cancel; what survives is a spurious **explicit** force that grows as the cell shrinks.
Grading puts the smallest cells exactly where the field gradient is largest, which is the
worst place to put them for this scheme. So the fix is not to grade and then fight it: it
is to keep the mesh uniform and pick Ha so that the uniform cell resolves `L/Ha`.

### The stability bracket, measured and not derived

With `c_A*dt/dy` as the magnetic Courant number:

| mesh | `c_A*dt/dy` | outcome |
|---|---|---|
| ESI tutorial, uniform `dy` 0.05 | **2.0** | runs |
| graded, `dy` 0.00633 at the wall | **15.8** | diverges, SIGFPE at `t` 0.07 |

The limit is somewhere between 2 and 15.8 and nothing here knows where. The template's
`delta_t_s()` asks for **0.5**, inside the one stable datum by a factor of four, and
prints what it got, so a future divergence can be compared against this bracket rather
than guessed at. `tests/test_journey_mhd.py` asserts both that ratio and that the
reported Courant number stayed below one -- the two checks that would have caught this
on the run that produced it instead of leaving a stack trace to be read backwards.

### And the cost: Ha cubed

Holding cells-per-layer fixed makes `dy` fall as `1/Ha` while `c_A` rises as `Ha`, so the
step falls as `1/Ha^2` and the cell count rises as `Ha`. **The total work goes as `Ha^3`.**

| Ha | `dy` for 8 cells in the layer | cells | steps to 2 s | work |
|---|---|---|---|---|
| 5 | 0.025 | 8,000 | 800 | 1x |
| 20 | 0.00625 | 32,000 | 12,800 | **64x** |

Sixty-four times is more than the whole `journey` job's budget, which is why the graded
points are **Ha 5 and Ha 0** and not the tutorial's 20. At Ha 5 the layer is 0.2 m, a
uniform 80-cell mesh puts eight cells in it, and the profile has structure across the
whole channel rather than a flat core with two thin layers -- so an RMS over a uniformly
sampled line measures something everywhere. That is a property of this solver's explicit
magnetic coupling, not of Hartmann flow, and it is the kind of thing that is cheap to
learn once and expensive to rediscover.

### The transport numbers

`rho = nu = mu = sigma = 1`, which is the ESI tutorial's choice at
`$FOAM_TUTORIALS/electromagnetics/mhdFoam/hartmann` and is deliberate rather than lazy.
The magnetic Prandtl number `Pm = nu*mu*sigma` is then 1, so the viscous and magnetic
diffusion times across the channel are equal and the transient settles in a couple of
`L^2/nu`. A real liquid metal has `Pm` of order 1e-5 -- six orders between the two
timescales -- and **the same graded answer**, because (2) and (3) contain Ha alone. A
metal's numbers would buy nothing and cost a transient nobody can afford. Ha maps onto a
real duct through `Ha = B0*L*sqrt(sigma/(rho*nu))` and nothing else.

`Re = u_mean*2L/nu` is 2, which is why `laminar` is not a modelling assumption being
smuggled in: `mhdFoam` has no turbulence model at all, and at Re 2 none is wanted. It is
also why a channel twenty half-heights long, sampled at fifteen, is far past developed.

## The three things that go quietly wrong

* **`mu` is a permeability.** A dynamic viscosity written in
  `constant/transportProperties` is read without complaint, because the dimensions come
  from `createFields.H` and not from the file, and it enters the answer through Ha and
  through `DB`. The dictionary the template writes says so on the line above the entry.
* **`0/pB` has to exist.** It is a fictitious magnetic flux pressure whose only job is
  to project `phiB` divergence free -- the solver's own `Description` block says as
  much -- so there is no physical condition to get right, but `createFields.H` reads it
  `MUST_READ` and a case without one does not start. `magneticFieldErr.H` prints
  `magnetic flux divergence error = ...` every B-PISO pass, which is
  `deltaT*weightedAverage(mag(div(phiB)))`, and a growing one is the projection failing
  rather than a physics result. The journey test differences the first ten against the
  last ten for that reason.
* **`fvSolution` needs `BPISO` as well as `PISO`.** `mhdFoam` runs two PISO loops per
  time step and reads their corrector counts from two separate sub-dictionaries.
  `nCorrectors` in the wrong one is read as a default.

## The recorded run

Measured in the `journey` job of `.github/workflows/tests.yml`, which builds the
workspace image, runs the template at Ha 5 and Ha 0, solves both, grades both and
cross-grades them.

**No agreement number is recorded here yet.** The first run of this case diverged, for
the reason set out above, and the uniform-mesh version has not completed a `journey` job
at the time of writing. What the section above records is a measurement -- of a failure,
with the Courant number, the stack frame and the solver's own documentation of the
mechanism -- and it is kept because the next person to reach for a graded mesh here
should not have to pay for it again. The agreement columns in
[`benchmarks/README.md`](../README.md) stay empty until something has solved it, which is
the same rule that file applies to the flat plate.
The numbers are printed by `tests/test_journey_mhd.py` on every run of that job
(`-s` is deliberate: on a green run that log is the evidence), and the agreement it
measured is recorded in the `mhd` rows of [`benchmarks/README.md`](../README.md).

There is no fixture under `tests/data/` behind this case, for the same reason
`benchmarks/thermal_stress/` has none: it needs OpenFOAM to produce one and the
`benchmarks` job has none. What stands in for it is
`tests/test_benchmark_mhd.py`, which writes the exact profile and five wrong ones --
the parabola, the right shape at the wrong Ha, the right shape at the wrong flow rate,
a tilted profile and half a channel -- and requires the grader to pass the first and
fail each of the others at a different check.
