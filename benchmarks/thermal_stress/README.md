# Thermal stress in a constrained bar

A case whose answer is derived here rather than quoted from anywhere, which is the
point of it. `benchmarks/README.md` records that of the eleven reference numbers in
this repository, re-checking them against their sources changed two, and both errors
flattered the result. This case has no source to misread: every constant in its answer
is read out of the case's own dictionaries, and the solution is four lines of algebra
anybody can check without a library card.

Graded by `grade.py`, which reads only files the solver wrote.

## What is being validated, and what was already there

`solidDisplacementFoam` -- a transient segregated finite-volume solver for
linear-elastic small-strain deformation **with optional thermal diffusion and thermal
stresses**, in OpenFOAM's own words. It is a standard ESI OpenFOAM solver, at
`applications/solvers/stressAnalysis/solidDisplacementFoam`, and it is in the workspace
image already: the image is built `FROM opencfd/openfoam-default:2512`, a full ESI
distribution installed from the `openfoam2512` Debian packages. Nothing was installed
to make this benchmark possible and the image is not a byte larger for it.

What the distribution does **not** ship is a worked thermal case. The solver's only
tutorial, `tutorials/stressAnalysis/solidDisplacementFoam/plateHole`, carries a
`constant/thermalProperties` with `thermalStress no`: it is a mechanical case, a plate
in uniaxial tension, and the thermal half of the solver has no example in the
distribution at all. So the gap was never the solver. It was that nothing here knew the
solver existed, nothing wrote a case for it, and nothing checked it.

## The case

A bar of steel lying along x, of height H in y, one cell thick in z:

| boundary | displacement D | temperature T |
|---|---|---|
| x = 0 and x = L | `symmetryPlane` -- a frictionless rigid wall | `symmetryPlane`, insulated |
| y = 0, `hot` | `fixedValue (0 0 0)` | `fixedValue` dT |
| y = H, `cold` | `tractionDisplacement`, zero traction | `fixedValue` 0 |
| the two z faces | `empty` | `empty` |

`openreynolds/toolbox/templates/thermal_stress.py` writes it, mesh and all.

**T is a temperature rise, not an absolute temperature.** This is the single thing
most likely to produce a wrong answer from this solver and it is not optional
bookkeeping. Read `readThermalProperties.H` and `calculateStress.H`: the thermal term
is `3*K*alpha*T` with T taken raw, there is no `T0` or `TRef` entry anywhere in
`constant/thermalProperties` to subtract, so the stress-free state of the body is at
T = 0. A case that puts `internalField uniform 300` in `0/T` because the metal is at
300 K is asking for the stress of a 300 K rise from an unstressed state at absolute
zero, which on steel is about -660 MPa conjured out of nothing. The plateHole tutorial
ships exactly that T field and gets away with it only because its `thermalStress` is
`no`.

## The answer

**Conduction.** Steady state between two fixed values with the sides insulated is the
linear profile, exactly:

    T(y) = dT * (1 - y/H)

**Elasticity.** Three kinematic facts come from the boundaries: the ends hold
`ex_xx = 0`, the top is traction-free so `sigma_yy = 0`, and the mesh is one cell thick
so `ex_zz = 0`. Put them into the solver's own constitutive law, which
`calculateStress.H` writes as

    sigma = 2*mu*symm(grad(D)) + lambda*I*tr(grad(D)) - I*3*K*alpha*T

with `ex_yy = e` the one non-zero strain:

    sigma_yy = (2*mu + lambda)*e - 3*K*alpha*T = 0   =>   e = 3*K*alpha*T/(2*mu + lambda)
    sigma_xx = lambda*e - 3*K*alpha*T                =   -3*K*alpha*T * 2*mu/(2*mu + lambda)

Everything now depends on which pair of Lame coefficients
`readMechanicalProperties.H` selected, and that is the `planeStress` switch:

| `planeStress` | lambda | 3K | 2mu/(2mu+lambda) | sigma_xx(y) | ex_yy(y) |
|---|---|---|---|---|---|
| `yes` | nu E/((1+nu)(1-nu)) | E/(1-nu) | 1-nu | **-E alpha T(y)** | (1+nu) alpha T(y) |
| `no` | nu E/((1+nu)(1-2nu)) | E/(1-2nu) | (1-2nu)/(1-nu) | **-E alpha T(y)/(1-nu)** | (1+nu)/(1-nu) alpha T(y) |

In the plane-stress row the two factors of (1-nu) cancel and no elastic constant
survives in the stress but E: it is the textbook result for a bar prevented from
expanding along its length, `sigma = -E alpha dT`. The plane-strain row is the same
algebra down the other branch and differs by exactly 1/(1-nu), which is 1.4286 at
nu = 0.3. Integrating `ex_yy` up from the floor,

    D_y(y) = (1+nu) * alpha * dT * (y - y^2/(2H))        D_y(H) = (1+nu) * alpha * dT * H/2

**That these are exact and not asymptotic.** `sigma_xx` varies with y and only with y,
so x-equilibrium holds; `sigma_yy` and `sigma_xy` are identically zero, so
y-equilibrium holds; `D = (0, D_y(y), 0)` is compatible with those strains; and D is
identically zero on the y = 0 face, which is why `fixedValue (0 0 0)` can be imposed
there without over-constraining anything and without a boundary layer near it. The
answer does not depend on the mesh, on the bar's length, or on how long the run took.

**The solver's own `Max sigmaEq` line is also exact.** Here sigma is
`diag(s, 0, s)` -- the plane-stress `sigma_zz` the solver prints is not the true zero
of plane stress but `lambda*e - 3*K*alpha*T`, which equals `sigma_xx`; in plane strain
that same equality is the correct answer. Either way
`sqrt(1.5*magSqr(dev(sigma)))` works out to exactly `|s|`, so the solver's own
per-write line should print `E*alpha*dT` (plane stress) or `E*alpha*dT/(1-nu)` (plane
strain) and nothing else. `docker/workspace/check.sh` gates the image on that single
number, which is why the gate needs no file parsing at all.

For the template's steel (E = 2e11 Pa, nu = 0.3, alpha = 1.1e-5 /K) and dT = 100 K:

| | plane stress | plane strain |
|---|---|---|
| sigma_xx on the hot face | **-220.0000 MPa** | **-314.2857 MPa** |
| max sigmaEq | 220.0000 MPa | 314.2857 MPa |
| D_y at the free top face, H = 0.02 m | 14.30 um | 20.43 um |

## Why this case and not a tutorial comparison

A uniformly heated bar would be a weaker test of the same solver. With T uniform,
`grad(3*K*alpha*T)` is zero in every interior cell and the entire thermal load arrives
through `tractionDisplacement`'s boundary term; the field is uniform, so every gradient
scheme in `fvSchemes` is exercised on a function it reproduces trivially. With T linear
in y, `grad(3*K*alpha*T)` is non-zero everywhere and is cancelled, to the solver's
tolerance, by the gradient of `div(sigmaD)`. Both terms are individually large and the
answer is their difference, which is the arrangement in which a sign error or an
inconsistent pair of gradient operators shows up rather than cancelling.

Grading **both** plane states out of the same template is what pins nu. The stress
answer in plane stress contains only E and alpha, as the product `E*alpha`, so a run
graded in that state alone cannot tell a wrong nu from a right one; the plane-strain
answer carries 1/(1-nu) and the two are 43% apart.

## What this does not establish

That linear small-strain elasticity is the right model for anyone's problem. It is not
for large rotations -- which is also why preCICE's perpendicular-flap FSI tutorial,
whose `solid-openfoam` participant is this same solver, disagrees with its CalculiX
reference rather than reproducing it. It establishes that this solver solves the model
it says it solves, on a case where the model's own answer is known exactly.

Nor does it establish anything about a real material. E, nu, alpha, C and k are
settings here, three of them lifted from the plateHole tutorial so a reader can check
them against something that ships, and the case is a verification against mathematics
rather than a validation against a measurement.

## Running it

In the workspace image, from an empty directory:

    cp /work/.toolbox/templates/thermal_stress.py .
    python3 thermal_stress.py          # mesh, case, checkMesh, mesh_look
    solidDisplacementFoam
    postProcess -func "components(sigma)" -noZero
    postProcess -func singleGraph -noZero

then, from a checkout of this repository:

    python benchmarks/thermal_stress/grade.py bar <case>

`sigma` is a symmetric tensor and the sampler wants scalars, so `components(sigma)`
has to run before `singleGraph`; that is the order the plateHole tutorial's own
`Allrun` uses, for the same reason.

`tests/test_journey_thermal_stress.py` is that whole sequence as a test, and it is what
the `journey` job in `.github/workflows/tests.yml` runs inside the image. The run
recorded in `benchmarks/README.md` is that job's.
