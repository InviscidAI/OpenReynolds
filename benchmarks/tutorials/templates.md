# The templates, as tutorials

Tier 2 of [`README.md`](README.md): every recipe in
[`../../openreynolds/toolbox/templates/`](../../openreynolds/toolbox/templates/). Each one
is a single Python file you copy into an empty directory and run. It builds a mesh with
gmsh, converts it with `gmshToFoam`, retypes the patches, runs `checkMesh` and finishes by
calling `mesh_look.py`, so the result is looked at rather than only built.

```sh
mkdir slab && cd slab
cp /work/.toolbox/templates/porous.py .      # on the instance; from the repo, copy from
python3 porous.py                            # openreynolds/toolbox/templates/
```

They need an OpenFOAM, which in practice means the workspace image. Nothing in here needs
a model turn or a hosted study.

**What this page is for.** The template files themselves are long and carry the full
account of every trap baked into them; that is the right place for it and this page does
not repeat it. What this page adds is the thing a template cannot say about itself: **what
is known about the answer before the solver starts**, stated with the length and velocity
scales it is known on, and **what has actually been measured** against it in this
repository. Two of the seven end in an exact comparison. The other five do not, and
saying so is the point rather than the omission.

| template | edit these | what it writes | the answer before the solve | measured here |
|---|---|---|---|---|
| `duct2d.py` | `LENGTH_M`, `HEIGHT_M` | a mesh and **stub** dictionaries | **exact**, three lines of algebra | **yes, in CI** |
| `body_in_box.py` | `BODY_RADIUS_M`, `CELL_SIZE_M` | a mesh and **stub** dictionaries | none shipped | the mesh only |
| `free_surface.py` | `TANK_WIDTH_M`, `COLUMN_WIDTH_M` | a solvable `interFoam` case | one exact pre-release check | the mesh only |
| `rotating.py` | `VESSEL_RADIUS_M`, `ROTOR_RPM` | a solvable `simpleFoam` + MRF case | two exact conservation statements | the mesh only |
| `porous.py` | `SLAB_THICKNESS_M`, `PRESSURE_DROP_CURVE` | a solvable `porousSimpleFoam` case | **exact by construction** | the coefficients, not the solve |
| `thermal_stress.py` | `BAR_HEIGHT_M`, `DELTA_T_K` | a solvable `solidDisplacementFoam` case | **exact**, closed form | **yes, in CI** |
| `mhd_hartmann.py` | `HARTMANN_NUMBER`, `HALF_HEIGHT_M` | a solvable `mhdFoam` case | **exact**, closed form | the grader in CI; the solve's first `journey` run diverged and is recorded |

The stub/solvable split matters and is the first thing to know. `duct2d.py` and
`body_in_box.py` write a `controlDict`, `fvSchemes` and `fvSolution` that are just enough
for `gmshToFoam` and `checkMesh` to build a `Foam::Time` and an `fvMesh`, and they expect
to be overwritten by whatever dresses the case -- `case_gen.py`, usually. The other four
write the real dictionaries and are meant to be solved as they stand. All six stop short
of launching the solver, because what a solve costs and whether it is wanted is not a
template's call; each prints the command it would be.

## duct2d.py -- a plane duct, against an exact answer

A straight rectangular duct, 0.10 m x 0.02 m as it ships, meshed in quads and extruded
one cell into hexahedra. This is the 2D recipe: OpenFOAM has no 2D mesh, only a 3D mesh
one cell thick with the two faces normal to the thickness typed `empty`.

**The answer is exact and the derivation is three lines.** Far enough from the inlet the
flow is fully developed plane Poiseuille flow, which is the exact solution of the steady
Navier-Stokes equations rather than a published measurement with an uncertainty on it:

```
developed, 2D, steady:  nu d2u/dy2 = (1/rho) dp/dx = const,  u(0) = u(H) = 0

  u(y)          = -(1/(2 nu)) (dp/dx)_kin y (H - y)      a parabola
  u_mean        = -(H^2/(12 nu)) (dp/dx)_kin             integrate over H
  u_max         = u(H/2) = 1.5 u_mean                    put y = H/2 in the first
  (dp/dx)_kin   = -12 nu u_mean / H^2                    rearrange the second
  tau_wall/rho  = nu |du/dy|_0 = 6 nu u_mean / H         differentiate the first
```

**The normalisation, which is where the one available factor of two lives.** `H` is the
**channel height**, 0.02 m. `u_mean` is 1 m/s exactly, because the inlet is a uniform
1 m/s `fixedValue` and mass is conserved, so nothing below is fitted. But the Reynolds
number `case_gen.py` is asked for is **Re 10 on the hydraulic diameter**, which for a
plane passage is `2H` = 0.04 m, so `nu = U * 2H / Re` = 4e-3 m^2/s. Putting `H` into the
Reynolds number instead of `2H` halves `nu` and moves every number above by a factor of
two while leaving the case looking entirely reasonable. White, *Viscous Fluid Flow*, 3rd
ed., section 3-3.2 derives the same expressions; no constant has to be looked up.

**What has been measured.** `tests/test_journey_duct.py` builds this mesh, dresses it,
solves it, reads the field back with the product's own readers and asserts the physics, in
the `journey` job of `.github/workflows/tests.yml` on every pull request:

| quantity | exact | measured |
|---|---|---|
| peak of the node-sampled profile | 1.5 m/s | **1.4863 m/s**, about a per cent low because each node averages its two cells |
| peak / mean of the same samples | 1.5 | **1.532** |
| `(dp/dx)` over x = 0.06 to 0.09 m | -120 m/s^2, kinematic | **+2.9%**, second order on ten cells across |
| the solver's own global continuity error | 0 | **-9.04e-08** against a 4e-05 m^3/s throughflow |
| `balance.py`'s boundary-integral closure | 0 | **0.84%**, a different and looser quantity: it integrates the boundary `U` field rather than the solver's `phi` |
| wall shear, from four independent y+ sources | 1.2 m^2/s^2, kinematic | **-9.7% and -8.7%** from the two `yPlus` sources, **+10.8% and +13.5%** from `wallShearStress`, bracketing the exact answer from both sides |
| the two walls' shear against each other | equal, by symmetry | **0.6%** apart |

Every tolerance in that file carries the reason it is the number it is
(`tests/test_journey_duct.py:142-180`), and none of them is a published band: the
reference is exact, so each is the discretisation ten cells across is allowed.

**The one thing in there that is not a constant.** `CELL_SIZE_M` is a background size
handed to gmsh, not a division count, and the Blossom recombination that makes the quads
does not honour it as one. The first measured run came out with 1000 cells whose first row
at the wall is **1.491 mm on a 2 mm request**, so a y+ computed from an assumed 2 mm first
cell was out by a third and looked like a solver problem. The first-cell height is
therefore measured off `constant/polyMesh` with `layer_report.py` and never assumed.

## thermal_stress.py -- a constrained bar, against a closed form with no error bar

A 2D steel bar, 0.04 m x 0.02 m as it ships, heated 100 K on one face and held in plane
stress or plane strain, solved by `solidDisplacementFoam` -- which is standard OpenFOAM
and has been in the image since the first build. What was missing was that nothing here
knew it.

**The answer is exact and has no uncertainty to hide inside**, which is why this case was
chosen. [`../thermal_stress/README.md`](../thermal_stress/README.md) derives it from the
three kinematic facts the boundaries impose plus the solver's own constitutive law:

```
plane stress:  sigma_xx(y) = -E alpha T(y)
plane strain:  sigma_xx(y) = -E alpha T(y) / (1 - nu)
with T(y) = dT (1 - y/H)
```

For the template's steel (`E` 2.0e11 Pa, `alpha` 1.1e-5 /K, `nu` 0.3, `dT` 100 K) that is
**-220.0000 MPa** on the hot face under plane stress and **-314.2857 MPa** under plane
strain. Pascals, not a normalised group: there is nothing to get the scale of wrong here,
which is a pleasant change.

**Why both plane states are graded rather than just one.** The plane-stress answer
contains `E` and `alpha` only as their product, so it cannot distinguish a wrong `nu` --
or a wrong `E` compensated by a wrong `alpha`. The plane-strain answer is the same algebra
down the other branch of `readMechanicalProperties.H` and is the only place `nu` appears
in the stress. The two are 43% apart, and their **ratio** is the cleanest number in the
case because whatever the mesh does to one state it does to the other.

| quantity | exact | measured |
|---|---|---|
| `sigma_xx(y)` against `-E alpha T` at the solver's own T, plane stress | 0 error | RMS **0.094%**, max **0.300%** of the 220.0000 MPa peak |
| the same, plane strain | 0 error | RMS **0.162%**, max **0.556%** of 314.2857 MPa |
| hot face, extrapolated | -220.0000 / -314.2857 MPa | **-220.2858** and **-314.6932 MPa**, **+0.130%** both |
| the two states' ratio | 1.428571 = 1/(1-nu) | **1.429471**, **+0.063%** |
| `max sigmaEq` | 220.0000 MPa | **+0.261%** on the gmsh quad mesh, **0.00%** on a uniform `blockMesh` grid |
| the temperature's e-folding time | 2.2834 s from the bar height and the diffusivity | which is why `END_TIME_S` is what it is rather than a round number |

**Where the remaining tenths of a per cent come from: the mesh, and only the mesh.** The
answer is mesh-independent, so a uniform grid reproduces it to 0.00% and what is left is
the gmsh quad mesh -- Blossom recombination leaves it mildly non-orthogonal, and `grad(D)`
is `leastSquares` on it.

Held in three places, deliberately: `tests/test_benchmark_thermal_stress.py` holds the
grader to the closed form and feeds it four wrong profiles it has to reject;
`tests/test_journey_thermal_stress.py` runs the template, solves both plane states in the
image and grades what came out; and the `solidDisplacementFoam` gate in
`docker/workspace/check.sh` asserts the same number out of the solver's own log.

## porous.py -- a Darcy-Forchheimer slab, exact by construction

A 2D duct with a porous slab across it: a filter, a radiator, a screen or a packed bed.
The subject of this template is **the two coefficients nobody knows.** Rather than
demanding `d` and `f`, it derives them from what a datasheet or a rig actually gives.

OpenFOAM's sink is `S = -(mu d + (rho |U|/2) f) U`, so the drop across a slab of thickness
`L` is

```
dp = (mu d U + 0.5 rho f U^2) L
```

and dividing through by `L U` makes that a **straight line in U** whose intercept is
`mu d` and whose slope is `0.5 rho f`. A Forchheimer plot, fitted with
`statistics.linear_regression` from the standard library. So the reference here is not a
published measurement at all: it is OpenFOAM's own source term integrated across the slab,
which makes it **exact by construction**, and what is being checked is that the fit and
the solver agree about the same algebra.

**The normalisation, and the two ways to get it wrong.** `U` is the **face (superficial)
velocity** -- volume flow divided by the whole slab face, including the solid -- and not
the pore velocity, which is larger by `1/e`. `L` is the slab thickness **along the flow**;
`d` and `f` are per unit length, so a pressure drop quoted without the thickness it was
measured across is half a measurement. And **the case's pressure is not in pascals**:
`porousSimpleFoam` is incompressible and solves `p/rho` in m^2/s^2, so a 503 Pa
measurement compared against a solved field reading 418 and called a 17% error is the
mistake. They agree exactly, by a factor of rho.

**What has been measured.** Running the shipped template's own arithmetic
(`python -c "import porous; print(porous.summary())"`, no OpenFOAM needed):

| check | exact | measured |
|---|---|---|
| the fit run backwards against the four points it came from | the points themselves | worst **7.6e-5%**, at 1.5 m/s |
| `d` from the independent Ergun reduction, against the fit | identical, since the shipped curve *is* Ergun's prediction rounded to six figures | **-2.67e-4%** |
| `f`, the same | identical | **+2.11e-4%** |
| the drop across the slab at the 1 m/s design point | **503.34 Pa** = **418.057 m^2/s^2** | **not solved here** |

That last row is the honest state of this case. The coefficient route is exact and checked
in both directions; **nothing in this repository has solved the case and quoted a drop
against the 503.34 Pa the template prints.** What would close it is one solve of the
shipped case and a `probe.py op_between` across the slab, which is the same measurement
`balance.py` and `probe.py` already make on other cases. Until that exists this is a
tutorial about getting the coefficients and the units right, which is the part that is
usually wrong, rather than a validation of `porousSimpleFoam`.

**Where the default curve comes from**, since a shipped number with no provenance is the
thing this directory exists to avoid: `PRESSURE_DROP_CURVE` is **not a measurement** and
is not invented either. It is the Ergun form's prediction for a bed of 3 mm spheres at a
void fraction of 0.38, in air at 20 C, through the default 50 mm slab, rounded to six
figures, so that the two routes in the file check each other. The Ergun form as it is
universally written is

```
dp/L = 150 mu (1-e)^2/(e^3 Dp^2) U  +  1.75 rho (1-e)/(e^3 Dp) U^2
```

with `e` the void fraction, `Dp` the particle diameter and `U` the superficial velocity.
Term by term against OpenFOAM's, `d = 150(1-e)^2/(e^3 Dp^2)` and
`f = 3.5(1-e)/(e^3 Dp)` -- the factor of 2 on Ergun's 1.75 is OpenFOAM's own half running
the other way, and dropping it is a factor-of-two error in the term that dominates a
packed bed. **Ergun (1952), *Chem. Eng. Prog.* **48**(2), 89-94 was not read here.** The
two constants are quoted from the form as OpenFOAM's and the wider literature's secondary
sources write it, so replace the shipped curve with your own measured points and the
provenance question disappears: the fit does not care where the points came from.

## free_surface.py -- a two-phase dam break, with the reference it cannot have

A square tank one cell thick with a water column in the bottom left corner, meshed,
`setFields` run so the column is a real interface rather than a uniform field, and the
whole `interFoam` case written. As it ships it is the classic `damBreak` box: a 0.584 m
tank with a 0.1461 m column standing twice as tall, which is Martin & Moyce's `n = 2`.

**The reference this case wants is the one the SPH row declines to quote, for the same
reason.** Martin, J. C. & Moyce, W. J. (1952), *Phil. Trans. R. Soc. A* **244** (882),
312-324 published surge-front position against time for a collapsing column, and **that
paper could not be read**: the publisher returns HTTP 403 to an automated request and no
open repository copy was found. Three things that matter are therefore unverified -- which
non-dimensionalisation it defines (`t sqrt(2g/a)` and `t sqrt(g/a)` are both in
circulation in the secondary literature and differ by a factor of **1.414** at every
point of the curve), which figure or table the data is in, and what the experimental
scatter was. [`../sph/README.md`](../sph/README.md) sets out the full argument for not
quoting it, and the open-access alternative that would settle it is any later paper that
reproduces the original table with its own axes stated, or a primary copy of Martin &
Moyce itself.

**What is exact here and free, before any of that matters.** Two things, and neither needs
a paper:

- **The undisturbed column is hydrostatic.** Before release, `p_rgh` at the base of the
  column is `rho g h` with `h` the column height, exactly. A column started at a uniform
  pressure is not a column at rest, and the first thing a bad initialisation does is to
  show up here. This is the same check [`../sph/`](../sph/) grades at **2.50%** RMS, so
  the two solvers can be put beside each other on it.
- **Water volume is conserved.** The integral of `alpha.water` over the domain is fixed
  for all time until water leaves through a boundary, and the VOF solver's own `alpha`
  bounds (`MULESCorr`, `nAlphaSubCycles`) are what decide how well. A dam break that gains
  or loses a per cent of its water has an answer about something else.

**What is measured here: the mesh, and `setFields` writing a real interface.** No solved
result of this case exists in this repository, so **no agreement is quoted**. What would
close it is one `interFoam` run and the surge front reported with no verdict, exactly as
[`../sph/`](../sph/) reports it -- which would also make the two solvers comparable on the
same configuration, and that comparison is worth more than either number alone.

`../../openreynolds/toolbox/notes/free-surface-field-notes.md` is the companion read: most
of the ways a two-phase case is wrong leave a run that finishes and logs nothing.

## rotating.py -- a frozen rotor, and the two things conservation still gives you

A 2D vessel with a six-bladed impeller at its centre, cut into a `rotor` cell zone and a
`stator` zone, with the rotation entered as an MRF source term rather than as a moving
mesh. 1000 rpm as it ships.

**There is no closed form for a six-bladed stirred vessel, and this page does not invent
one.** Worse, a frozen rotor is one blade angle rather than an average, so even a perfect
solve of it answers a slightly different question from the one usually asked.

**Two statements are still exact, and both are conservation rather than correlation:**

- **In a steady state with no through-flow, the torque on the impeller and the torque on
  the vessel wall are equal and opposite exactly**, by conservation of angular momentum.
  Not approximately, not to the mesh: exactly. This is the check
  [`../moving_mesh/`](../moving_mesh/) found teeth in, and the finding is the reason to
  write it down here: a `cyclicAMI` sliding interface held its interpolation **area** to
  43 ppm of unity while the two wall torques **disagreed by 0.911%**, against 0.011% on
  the same annulus meshed conformally. The check every rotating-zone case does run passed
  a case whose torque was a per cent wrong; the check nobody runs is the one that caught
  it.
- **Power = torque x omega.** With `omega` in rad/s, which is **104.719755 rad/s** for the
  shipped 1000 rpm -- a factor of **9.5493** from the nameplate units, and the template
  prints the round trip back to rpm for exactly that reason.

**And one check that is not exact but is the one that matters most for an MRF case.** The
zone face is a surface where nothing physical happens, so moving `ZONE_FRACTION` off its
0.7 by about a quarter and re-running should move the torque by no more than a few per
cent. A torque that tracks the zone radius is a number set by a modelling choice rather
than by the flow. `ladder.py`'s frozen-rotor rung states this, and the template prints it.

**What is measured here: the mesh and its two cell zones.** Nothing has solved this
template and quoted a torque, so no agreement is quoted. What the template is explicit
about not being able to tell you is as useful as what it can: the 100 Hz blade-passing
tone (6 blades x 1000 rpm / 60) is **absent, not small**; there is no unsteady wake, no
shedding and no rotating stall; and the spread over several blade angles is the error bar
on the single angle it solved. Any of those wants a `cyclicAMI` sliding mesh and
`pimpleFoam`, transient, which is what [`../moving_mesh/couette_ami.txt`](../moving_mesh/)
grades.

## viv_cylinder.py -- a cylinder on a spring, and the two stages that order the work

An O-grid around a cylinder, written as **two complete `pimpleFoam` cases sharing one
mesh**:

* `fixed/` -- the cylinder bolted still (`staticFvMesh`, `noSlip`). Its job is the
  Strouhal number, and its job is to be run **first**: a mesh that does not land on the
  published shedding frequency is not ready to carry a moving-body answer, and finding
  that out after a 400 s moving-mesh run is the expensive order to find it out in.
* `released/` -- the same mesh under `dynamicMotionSolverFvMesh` and
  `sixDoFRigidBodyMotion`, the body on a linear spring with zero structural damping.
  `--dof 2` frees it in x as well, on an isotropic spring.

Then `blockMesh`, `checkMesh` and `mesh_look.py`. `viv.py` is the instrument for the
finished run: amplitude, frequency, lock-in and a per-cycle energy balance.

**Edit two numbers:** `DIAMETER_M` and `USTAR`, the reduced velocity `U/(f_n D)`, which
is what sweeps the response. `--re` sets the free stream from the diameter and
viscosity, `--mstar` sets the body mass from the displaced fluid mass, and the spring
stiffness falls out of `USTAR`.

### What is known before the solver starts

**The fixed stage has a reference.** Williamson's `St = 0.164` at these Reynolds
numbers, and the graded run measures **0.166141**, which is **+1.10%** on the 0.1643
fit. That is the gate on the mesh.

**The released stage's energy balance is exact and is zero.** Over a cycle in a settled
state, the work the flow does on the body equals the work the body does on the flow,
because a linear spring with zero damping stores and returns everything. So the audit
has no reference to misquote: it compares against nought. `benchmarks/moving_mesh/`
found teeth in exactly this -- a loose coupling passed every other check while the audit
read **-10.37% to -10.09%**, and tightening it (`moveMeshOuterCorrectors yes`,
`accelerationRelaxation 1.0`, which is what this template ships) took it to **+0.044% to
+0.176%**.

**The natural frequency in water is not the natural frequency on paper, and the
difference changes the story.** A circular cylinder's 2D added mass is the displaced
fluid mass exactly -- an added-mass coefficient of 1, Lamb art. 68 -- so in still water
the body rings at

    f_n,water / f_n,vacuum = sqrt( m* / (m* + 1) )

which at the shipped `m* = 10` is **0.95346**. The graded tight run's `f/f_n = 0.9697`
is therefore **1.70% above** the frequency the body would ring at in water, not 3% below
a resonance. Those are different statements about the same number, and only the second
one is about the physics.

**A/D carries no verdict.** Williamson & Govardhan (2004) and Khalak & Williamson
(1996, 1999) are the sources for the lock-in amplitude range and **none of them could be
read** -- Annual Reviews and ScienceDirect both answer 403. The graded runs report
**0.6401929** loose and **0.5670282** tight and stop there. `benchmarks/moving_mesh/`'s
own README had quoted "about 0.56 at U* = 5" from Menon & Mittal as though the paper
states it; it does not -- the number is only readable off Figure 5(a) -- and that row now
says so, with the journal reference added.

One normalisation trap recorded with it: Menon & Mittal define `m* = 2m/(rho D^2)`, so
their `m* = 10` is **6.37** on displaced mass. Two papers both saying "m* = 10" can mean
a 57% different body.

## evaluate_steady.py -- one round of a parametric study, and who owns which number

A cambered 2D section at incidence above a ground plane, in a box, meshed in quads by
gmsh, extruded one cell, converted by `gmshToFoam`, retyped, `checkMesh`ed and looked at
with `mesh_look.py`. Patches: `inlet`, `outlet`, `ground`, `top`, `body`,
`frontAndBack`.

What makes it different from the seven above is that **it is not meant to be edited.**
Every other template here says "change the two numbers at the top and run it". This one
is the build.py of a round, and the numbers come from a design_constants.py the
caller writes beside it:

```sh
python3 evaluate_steady.py --write-round /work/study/rounds/000
# -> build.py (this file), hooks.py, goal.lock.json, design_constants.py

python3 /work/.toolbox/parametric.py /work/study/cand/007 \
    --round /work/study/rounds/000 --lock /work/study/goal.lock.json --mode solve
```

`CHORD_M`, `CAMBER_1`, `CAMBER_POS`, `THICKNESS`, `ALPHA_DEG` and `RIDE_HEIGHT_M` have
**no defaults here**, deliberately: an optimizer varies them and the lock's trim varies
the incidence, so a candidate that arrives without the file fails loudly rather than
meshing the same shape under a new name and reporting it as a new design. That is the
one failure mode a parametric study cannot recover from, because the result looks
plausible.

The mesh sizes *are* this file's own, and they are the fidelity. `goal.lock.json`'s
`fidelity.cells: {min, max}` is what holds them to account, and `fidelity.yplus` the
wall treatment: the sample leaves `yplus.min` out, because the stagnation point puts y+
near zero on any body and a floor there would fail every candidate for a reason that is
not about the mesh.

So three owners, and the split is the point: the **caller** owns the shape, this
**template** owns the mesh, and the **lock** owns the physics and the bands. Nothing
here reads a number it also writes.

## mhd_hartmann.py -- a conducting liquid across a magnetic field, against a formula

A 2D channel of half-height `L`, twenty times that long, one cell thick in `z`, with a
uniform field `B0` applied across the flow and a uniform mesh whose cell height is chosen
against the Hartmann layer rather than graded into it. It writes the whole `mhdFoam`
case -- `0/U`, `0/p`, `0/B`, `0/pB`, `constant/transportProperties` and the four
`system/` dictionaries -- runs `blockMesh` and `checkMesh`, finishes with
`mesh_look.py`, and prints the profile the solved case has to reproduce. Like
`free_surface.py`, `rotating.py` and `porous.py`, everything it writes is meant to be
solved as it stands; it stops short of launching the solver and prints the command.

**Edit two numbers:** `HARTMANN_NUMBER` and `HALF_HEIGHT_M`. The applied field, the
channel length, the grading, the layer thickness and the printed profile all follow.

**Why this case and not a prettier one.** Its answer is a formula, not a figure in a
paper. The derivation in the file is taken from `mhdFoam`'s own two equations in the
v2512 source, so there is nothing to misquote -- the same choice
`benchmarks/thermal_stress/` made, for the reason `benchmarks/README.md` argues at
length: of its eleven reference numbers, re-checking two against their actual sources
changed them, one by a factor of two.

**What is known before the solver starts.** Normalised on its own centreline speed, the
profile is a constant minus a `cosh`:

    u(y)/u_c = ( cosh(Ha) - cosh(Ha*y/L) ) / ( cosh(Ha) - 1 )

and the centreline-to-mean ratio closes it:

    u_c/u_mean = ( cosh(Ha) - 1 ) / ( cosh(Ha) - sinh(Ha)/Ha )

Both are exact. At the shipped `Ha = 5` the second gives **1.233128**, against
Poiseuille's 1.5 -- the field flattens the core until it moves closer to the mean, which
is the whole physical content of the case and is visible in one number. The two Hartmann
layers are `L/Ha` thick, so at `Ha = 5` they are 20% of the half-height each and a
uniform 80-cell mesh puts eight cells in each.

**Why 5 and not the ESI tutorial's 20.** Resolving `L/Ha` forces the cell height down as
`1/Ha` while the Alfven speed `c_A = B0/sqrt(mu*rho)` rises as `Ha`, and `mhdFoam`
differences its magnetic stresses explicitly, so the time step is capped by `c_A*dt/dy`
and falls as `1/Ha^2`. Total work goes as `Ha^3`: Ha 20 is sixty-four times Ha 5. The
first version of this template graded toward the walls to reach Ha 20 and the solver
diverged, with a max Courant number of 23.6 and a SIGFPE in `symGaussSeidelSmoother`.
[`../mhd/README.md`](../mhd/) has the log, the stability bracket and the solver's own
`Description` block saying those stresses do not cancel discretely.

The first form is the better check of the two: dividing a solved profile by **its own**
centreline value tests the shape without the solver's pressure gradient, its mean
velocity or its units entering the comparison at all. The second then tests that the
flow rate the case actually carries is the one the profile implies.

**No reference is quoted and none is needed.** Both expressions come from the equations
`mhdFoam` solves, so this case has no error bar to hide inside, and nothing in this
repository has solved it yet -- the agreement column stays empty until something has.
What *is* recorded is the failure: a wall-graded mesh at Ha 20 diverged, and the Courant
number, the stack frame and the `Ha^3` cost law are written down so the next person does
not pay for them again.

## body_in_box.py -- a body-fitted mesh, and a reference it does not claim

An OpenCASCADE primitive cut out of a rectangular flow box with a boolean, meshed
body-fitted with tetrahedra -- no STL, no `snappyHexMesh`. A sphere of radius 0.02 m as it
ships. The box runs `UPSTREAM_R` body-radii ahead of the body, `DOWNSTREAM_R` behind it
(a wake needs more room than an approach) and `SIDE_R` to each side, in multiples of the
body radius, which is the rule of thumb in
`../../openreynolds/toolbox/notes/openfoam-field-notes.md` for how far a farfield boundary
has to sit before it starts shaping the answer it was meant to measure.

**No reference is shipped and none is claimed.** The template's job is the mesh, and
`checkMesh` plus `mesh_look.py` is the only verdict it offers. Faces are named right after
the boolean, from the numbers the box was built with, never by looking at the finished
mesh.

**The reference that would make this a graded case, and why it is a gap rather than a
claim.** A sphere's drag coefficient is **exactly `Cd = 24/Re`** in the Stokes limit, on
the **diameter** and the **free-stream speed**, with `Cd` referenced to the **projected
area** `pi R^2` and not the wetted area -- a factor of four waiting to be picked up by
anyone normalising on the wrong one. That is an exact result, it needs no paper, and it
would make this the best first external-flow case in the repository. The shipped box is
not sized or dressed for Re below about 1, and no run of it at that Reynolds number exists
here, so the honest state is: **listed as a gap**. Shipping `24/Re` with no measurement
beside it is what [`../README.md`](../README.md) does not do with Blasius either, and for
the same reason.

## cht_wall.py -- a two-region mesh, and the answer a conjugate solve owes

`advise.py` offers a `conjugate-heat-transfer` family, `chtMultiRegionFoam` and its
steady twin, and until this there was **no case for either** in this repository: no
template, no row here, nothing graded. It was the only named family in `advise.FAMILIES`
with nothing behind it.

**What it builds.** A fluid layer then a solid layer, heat along x, the span two cells
each way so it is a real 3D mesh solving a 1D problem. Two `blockMesh` blocks carrying
the cell-zone names `fluid` and `solid`, then `splitMeshRegions -cellZones -overwrite`,
then `checkMesh -allRegions`.

**The mesh recipe is not invented.** It follows
[`docs/cad-acceptance/artifacts/desk/T8/build.py`](../../docs/cad-acceptance/artifacts/desk/T8/build.py),
a recorded run against this image, and three of its findings are the reason this template
is short:

- the zone name is a token in the `hex` entry, so **no `topoSet` is needed**;
- the interface faces are named in **no patch at all** -- they merge as internal and
  `splitMeshRegions` turns them into the mappedWall pair by itself;
- the unsplit `constant/polyMesh`, `constant/cellToRegion` and `0/` must be **removed**
  afterwards, because `cellToRegion` is a field of neither region and anything reading
  the base mesh is reading the wrong mesh. The template removes them and says so.

**The sides are named adiabatic walls, and that is physics.** The reference is 1D series
conduction, which is only the answer if no heat leaves through the span faces. Left
unnamed they fall into `defaultFaces`, whose type is a setting rather than a statement,
and a side that conducts makes the reference wrong while the run still converges.

**What is known before any solver starts.** With gravity set to zero the fluid cannot
convect, so both regions conduct and the steady answer is exact -- series resistance,
derived and graded in [`../../openreynolds/toolbox/conjugate.py`](../../openreynolds/toolbox/conjugate.py):

| | |
|---|---|
| flux, one value through both regions | **2944.78527607 W/m^2** |
| temperature at the interface | **301.84049080 K** |
| drop across the fluid / the solid | **98.159509 K** / **1.840491 K** |

Gravity is off deliberately and that is a modelling choice: a fluid layer with a
side-to-side temperature difference convects at any Rayleigh number, so a case left under
gravity has no closed form to be graded against. `constant/g` carries it where a reader
can see it.

**The trap, and it is a real one.** OpenFOAM's fluid `thermophysicalProperties` has **no
`kappa` entry**: conductivity comes out of the Prandtl number as `kappa = mu Cp / Pr`.
Asking for a fluid conductivity therefore means setting `Pr`, and a template that wrote
`kappa` somewhere would be setting nothing at all. This one inverts it --
`Pr = 0.001 * 4181 / 0.6 = 6.968333` -- and prints the arithmetic, because `Pr` read as a
property of the fluid rather than as the place `kappa` is specified is how the wrong
conductivity gets solved with nobody noticing.

**It does not invent the solver dictionaries -- it copies them off the image.** The
per-region thermophysical files and the coupled thermal boundary condition are spelled
differently in the ESI and Foundation builds (`thermophysicalProperties` against
`physicalProperties`, `turbulenceProperties` against `momentumTransport`) and nothing
here had run `chtMultiRegionSimpleFoam` to settle which this image takes. Guessing them
would be a case nobody has run that looks like one that works.

So they come from `$FOAM_TUTORIALS/heatTransfer/chtMultiRegionFoam/multiRegionHeater`,
which is `chem0d.py`'s pattern and for its reason -- that template copies `reactions` and
`thermo` out of the image's own tutorial so the data is the image's file byte for byte.
Here:

- **every file** in the tutorial's fluid-region and solid-region `constant/` directories
  is copied, *without enumerating their names*, because the names are the thing not
  known. Whatever this image's tutorial carries, the case gets;
- the **coupled thermal condition is read out of the tutorial's own `T` field**, by
  finding the entry on its `<region>_to_<region>` patch and keeping that block verbatim.
  That is the single entry a guess is most likely to get wrong;
- only `Pr` (the fluid) and `kappa` (the solid) are then patched, with **asserted**
  replacements: a file with no such entry, or two of them, is refused by name. A property
  silently keeping the tutorial's value would be a case solved for somebody else's fluid,
  converging happily with the wrong answer.

Off the image it refuses and says which path it wanted, so one run on the image finds a
moved tutorial. **No solved result exists in this repository yet**, no agreement is
quoted, and a journey test that solves this mesh and grades the interface temperature
against the number above is the next item.

## What every one of them has baked in

Four things the templates handle identically, which is most of what makes a hand-built
case wrong:

**Metres, always.** OpenFOAM has no units. A duct authored in millimetres and never
divided by 1000 is a mesh a thousand times too big, and `checkMesh` passes it without a
word.

**The 2D recipe, if the case is 2D.** OpenFOAM has no 2D mesh, only a 3D mesh one cell
thick with the two faces normal to the thickness typed `empty`. The common way a "2D" case
is actually a slow, thin 3D one is a mesh built several cells deep, or a side patch left
`patch`/`wall` where `empty` was meant. `mesh_look.py` lists what the patches actually
are.

**`gmshToFoam` reads only msh format 2.2 ASCII**, and it types every patch it creates
`patch`. So `Mesh.MshFileVersion` is set to 2.2 immediately before `gmsh.write`, and the
walls and the `empty` faces are retyped with `foamDictionary` afterwards. A physical
*volume* in the msh becomes a `cellZone` of the same name; a physical *surface* becomes a
patch. A `porosityProperties` or an `MRFProperties` naming a zone the mesh does not have
fails at the first solver step.

**The pressure unit.** Every incompressible OpenFOAM solver here solves `p/rho` in
m^2/s^2, not pascals. A force computed from it is out by rho; a pressure drop compared
against a datasheet in pascals is out by rho. The templates that report a drop print both
numbers for that reason.
