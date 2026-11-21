# Benchmarks and case studies

Every case in this repository whose answer is known from outside a solver: what it
measures, the published or closed-form value it is measured against, how close the real
run got, and where the remaining difference comes from.

The last column is the one worth reading. "Agrees to 3%" with no account of the 3% is not
a result, it is a coincidence that has not been examined yet, and on three of the cases
below the examination is what found the defect.

| case | measures | the answer, and where it comes from | the recorded run | where the difference comes from |
|---|---|---|---|---|
| [lid-driven cavity, Re 1000](felix/) | `u` and `v` on the centrelines | Ghia, Ghia & Shin (1982), *J. Comput. Phys.* **48**, 387-411, Table I (u on x = 0.5) and Table II (v on y = 0.5) | u RMS 0.45%, v RMS 1.06% of the lid speed | **the reference's own error is larger than the agreement.** Ghia solved Re 1000 on a 129x129 grid to second order, and his centreline extrema sit 1.46%, 1.59% and 2.20% from Botella & Peyret's (1998) spectral benchmark. Agreeing with Ghia to 0.45% is agreeing to inside his uncertainty, so part of that agreement is luck |
| [cylinder in a channel, Re 20](felix/) | drag coefficient | Schafer & Turek (1996) DFG benchmark 2D-1, Cd 5.5795, their interval 5.57-5.59 | Cd 5.522, **-1.04%** | 6765 hexes one layer thick: inside the 2% bound, outside the reference interval |
| " | lift coefficient | the same, Cl 0.0104-0.0110 | Cl 0.0010, **reported and not graded** | Cl here is produced by a 5 mm offset of the cylinder from the channel midline, so it is a small difference of two large surface integrals and needs resolution the drag does not |
| [circular Couette, sliding interface](moving_mesh/) | inner-wall torque | closed form for circular Couette flow | +0.43% | one 50 x 240 mesh |
| " | the two wall torques, which conservation makes equal and opposite **exactly** | 0, from conservation of angular momentum | **-0.911%, a failure** | the `cyclicAMI` interface, and first order in the angular slide per step rather than in cell size |
| [cylinder on a spring, Re 100](moving_mesh/) | shedding frequency, cylinder fixed | **0.1643**: Williamson (1989) *JFM* **206**, eq. (6), St = -3.3265/Re + 0.1816 + 1.6e-4 Re, the parallel-shedding fit reproduced as Fig. 7(a) of Williamson (1996) | St 0.166141, **+1.10%** on that fit, +1.31% on a bare 0.164 | 2.5% blockage, expected to raise St about a percent. But **the published correlations disagree by more than the error**: Roshko (1954) and Williamson & Brown (1998) both give 0.1671 at Re 100, against which the same result is **-0.55%**. Williamson's own tunnel data span 0.1598 to 0.1648 depending on whether the ends were manipulated into parallel shedding, and 0.164 is the parallel value, which is the one a 2D simulation should be held to |
| " | net fluid work per cycle, cylinder released | 0, because the structural damping is zero | **-10.1 to -10.4%, a failure**; +0.04 to +0.18% once tightly coupled | the coupling scheme, not the physics: it held a limit cycle 13% too wide |
| " | amplitude A/D | **no published value for this configuration**, and the nearest one is not a number: Menon & Mittal (arXiv:2006.11649, published as *J. Fluid Mech.* **907**, A37, 2021) is 2D, Re 100, zero damping, and states no A/D in its text -- the value at U\* = 5 is only readable off its Figure 5(a). Its mass ratio is also a different quantity: `m* = 2m/(rho D^2)` verbatim, so its m\* = 10 is **6.37** on displaced mass, not 10 | 0.640 loose, 0.567 tight | reported without a verdict, which is the honest thing to do with it. The definition check is the part worth keeping: two papers quoting "m\* = 10" at Re 100 can be 57% apart in body mass, and nothing in either paper flags it |
| [laminar heated pipe](tutorials/) | fully developed Nusselt number | the Graetz problem, Nu = lambda_1^2/2 = 3.6568, **exact** | 3.677 as reported, 3.63 to 3.65 recomputed, **sub-percent either way** | the wall-gradient stencil, which moves the answer 0.7% across four defensible conventions and so is larger than the 0.55% being quoted |
| [natural convection, Ra 1e5](tutorials/) | mean Nusselt number on the hot wall | de Vahl Davis (1983), *Int. J. Numer. Methods Fluids* **3**, 249-264, Table V: **4.509** on the hot wall, 4.519 for the cavity average | 4.5335, **+0.54%** | a 40 x 40 mesh, a one-sided wall gradient, and 0.3% of reference uncertainty. Against the 4.519 that secondary sources misattribute to the hot wall it reads +0.32%, which is how a validation gets half a percent better without the solver changing |
| [backward-facing step](tutorials/) | reattachment length on the stepped wall | Gartling (1990), 6.1 **channel** heights = 12.2 step heights, at **Re 800 on the channel height** with the mean inlet velocity (the paper itself is unread: see below) | 6.42 step heights = 3.21 channel heights, **about 47% short**, and still drifting | the run is Gartling's geometry scaled by two at the same `nu`, so **Re 1600 on the channel height** against the benchmark's 800 on that same length, and its lengths are step heights where the benchmark's are channel heights. One mistake, two factors of two, and a printed "+5%" that was neither |
| [Poiseuille, force-driven](lbm/) | the velocity profile across a channel | `u = G/(2 mu) y (H - y)`, **exact**, and no paper is needed for it | max error **5.2e-11** of u_max at nu 0.02, **1.2e-11** at 0.1, **3.3e-12** at 0.4 | the convergence stop, and nothing else. With two relaxation times at `(tau_plus - 1/2)(tau_minus - 1/2) = 3/16` the bounce-back wall sits on the lattice link midpoint at any viscosity, so this scheme has no discretisation error on this flow -- the residual it stopped at was 5e-14. **The same code with one relaxation time, four times from its one exact viscosity, is wrong by 3.7e-3**, which is what makes the row evidence rather than a solver compared with itself |
| [lid-driven cavity, Re 1000](lbm/) | `u` and `v` on the centrelines | the same Ghia, Ghia & Shin (1982) Tables I and II as the Felix row above, **imported from `felix/grade.py` rather than retyped** | u RMS **0.736%**, v RMS **0.749%** of the lid speed | **the reference's own error is larger than the agreement**, exactly as on the Felix row and for the same reason: Ghia's extrema sit 1.46% to 2.20% from Botella & Peyret (1998). Then 128 cells against his 129x129. Graded against the same 34 numbers and the same 2% bound as Felix (0.448% / 1.063%), which is the only thing that makes the two comparable |
| [hydrostatic pressure](sph/) | pressure against depth in still water | `p = rho g (h - y)`, **exact** | RMS **2.50%** of rho g h, worst particle **8.61%** | the density noise, multiplied by `c0^2`. `dp/drho` is about `c0^2`, so a density error is 400 times itself as a fraction of the `rho g h` being measured -- it read **67%** before a density diffusion term was added, with operators that balance gravity to 3%. The worst particle is at the free surface, where the kernel support is truncated and there is no fluid above to sum |
| [sloshing, first mode](sph/) | the oscillation period | `omega^2 = g k tanh(k h)`, `k = pi/L`: linear theory, **exact** (Lamb, *Hydrodynamics*, 6th ed., art. 228) | **-4.75%**, 1.1257 s against 1.1818 s | **the sign is the diagnosis.** Short means the effective tank is shorter than the one asked for, which is the wall-offset error dynamic boundary particles have. Underneath it, 0.6% from the 5%-of-depth amplitude against a linear reference, and 20 particles across the depth |
| [dam break](sph/) | the surge front position | Martin & Moyce (1952), *Phil. Trans. R. Soc. A* **244**, 312-324 -- **the paper could not be read**, the publisher refuses the request | **reported, not graded**: Z = 0.975 at T = 0 rising to 3.943 at T = 3.00. The run is graded only on containment and on weak compressibility holding | which non-dimensionalisation the paper defines is unverified, and `t sqrt(2g/a)` and `t sqrt(g/a)` are both in circulation and differ by 1.414 at every point. Quoting it from a secondary source is the error the rest of this file argues hardest against, so it is not quoted |

| [relaxation and settling](particles/) | the velocity of a particle released into a uniform stream, and its terminal velocity under gravity | `v = U (1 - exp(-t/tau_p))` and `w_t = tau_p g (1 - rho_f/rho_p)`, **exact** for linear drag, with `tau_p = rho_p d^2/18 mu`. No paper is needed for either | max error **1.9e-12** of U over five relaxation times; terminal velocity **2.2e-14** as a fraction | the convergence stop, and nothing else. These are the exact solutions of the equation the integrator integrates, so an agreement at 1e-12 is RK4 at `dt = tau_p/200` and not a physical claim. **What makes the row evidence is the contrast beside it**: the same integrator with the Schiller-Naumann correction at a particle Reynolds number of 42.8 settles at **0.4619** of the Stokes velocity, and matches the independently solved root of `w = tau_p g'/phi(Re_p(w))` to 5.1e-15. A drag law that quietly did nothing would pass the two exact checks and fail that one |
| [tracer orbit](particles/) | a massless tracer's path in solid-body rotation | it closes **exactly** after one period, `2 pi / Omega` | **5.1e-8** of the radius at 200 steps per revolution, observed order **4.00** over three refinements | the time step, and only the time step. There is no reference, no interpolation and no stopping criterion in this one, which is why the observed order is worth reading: it is 4.00 to three figures at every refinement, so the integrator is the RK4 the docstring claims and a first-order one fails the check (`tests/test_benchmark_particles.py` hands it forward Euler to prove the check bites) |
| [capture on a cylinder](particles/) | the critical Stokes number below which no particle reaches the wall, in inviscid flow | **1/8** in the `tau_p U_inf / R` convention. Read at source: Robinson, Warren, Turner & Sear, arXiv:2310.03474v2 (25 Feb 2025), Section III.3, "the critical Stokes number St_c = 1/4 for a cylinder" -- 1/4 in *its* convention, whose `U` is the local stagnation-point speed `2 U_inf` and whose length is the radius, and which says in the same paragraph that Langmuir and Blodgett "actually quote St_c = 1/8" "because of their alternative choice to set U = U_inf". **Taylor (1940) and Langmuir & Blodgett (1946), the primary attributions, were not read** | **0.1262909, +1.03%**, approached monotonically from above: +5.671, +3.152, +2.000, +1.375, +1.033% as the stopping speed tightens from 1e-6 to 1e-14 of U | **the stopping criterion, and its sign is not negotiable.** The threshold is an infinite-time statement -- below it the particle approaches the wall forever and never arrives -- so any tracker that has to stop waiting calls some sub-critical particles captured, and can only measure the threshold **high**. A measurement below 1/8 would be a bug and the grader fails it as one. The 1% floor is double precision: below 1e-15 of U the position increment is under the ULP of the radius, and 1e-16 measures the same 0.1262909. Not discretisation (a factor of five in `dt` moves it 0.12%, the wrong way) and not the far field (release at -3, -5, -10, -20 radii spans 0.01%), both of which were measured rather than argued |
| [a reacting 0-D reactor](combustion/) | the end temperature of a closed, adiabatic, constant-pressure reactor | **the solver's own thermodynamics, exact**: `chemFoam` conserves absolute enthalpy, so `sum Y_i(end) Ha_i(T_end) = sum Y_i(0) Ha_i(T0)` is one equation in one unknown over the case's own JANAF coefficients, derived in [combustion/README.md](combustion/README.md) from `hEqn.H`, `solveChemistry.H` and `StandardChemistryModel.C:544-556`. **2877.7712 K** for stoichiometric CH4/air from 1000 K at 1 atm | **2877.6383 K, +0.0046%** after 5,468 steps on one cell in 0.4 s | **the unburnt tail, and nothing else.** Graded a second time against the composition the solver itself ended on, the same balance closes to **+1.1e-06%** -- eight significant figures -- so the whole 0.0046% is the 1.6e-05 mass fraction of fuel still unconsumed at 0.5 s. A one-step irreversible third-order reaction approaches completion as a power law, not exponentially, so the tail is the error |
| " | the thermo data the answer above rests on | NIST Chemistry WebBook, gas-phase enthalpies of formation, read off its own `Mask=1` pages: CH4 **-74.6 +/- 0.3** (Manion 2002, adopting Gurvich 1991), CO2 **-393.51 +/- 0.13** and H2O **-241.826 +/- 0.040** (CODATA, Cox & Wagman 1984) kJ/mol | the image's own JANAF fit at 298.15 K: CH4 **-74.5992**, CO2 **-393.5487**, H2O **-241.8465**; heat of reaction **-802.641** against Hess's law's -802.562 kJ/mol, **+0.010%**; methane's lower heating value **50.0312** against **50.0260 MJ/kg** | **the reference's own uncertainty is larger than the agreement**, as on four other rows in this file. Every difference is inside NIST's stated error bar -- by 375x on methane, 3.4x on CO2, 1.95x on water -- and NIST's four methane determinations themselves span 0.37 kJ/mol, 0.50%, a hundred times the agreement quoted. Against the Chase (1998) value on the same page the fit reads +0.27 rather than +0.0008, which is de Vahl Davis's 4.509-against-4.519 in another costume |
| " | that the agreement above is about combustion and not about arithmetic | 0 temperature rise with `chemistry off`, one word changed and nothing else | **999.9985 K from 1000 K**, a drift of 0.0015 K, and the grader fails it | the energy equation is still integrated with the chemistry off, so 1.5 ppm over 5,009 steps is discretisation. The reacting run rises 1877.6 K, so the control and the case are three orders apart |
| [thermal stress, constrained bar](thermal_stress/) | `sigma_xx` through the bar, plane stress | **closed form, exact**: `sigma_xx(y) = -E alpha T(y)` with `T(y) = dT (1 - y/H)`, derived in [thermal_stress/README.md](thermal_stress/README.md) from the three kinematic facts the boundaries impose and the solver's own constitutive law. -220.0000 MPa on the hot face for the template's steel | `sigma_xx` vs `-E alpha T` at the solver's own T: **RMS 0.094%, max 0.300%** of the 220.0000 MPa peak. Extrapolated to the hot face **-220.2858 MPa, +0.130%**. `max sigmaEq` **+0.261%** on the gmsh mesh and **0.00%** on the gate's uniform grid | **the mesh, and only the mesh.** The answer is mesh-independent, so the gate's `blockMesh` grid reproduces it to 0.00% and these few tenths of a percent are the gmsh quad mesh: Blossom recombination leaves it mildly non-orthogonal (`checkMesh` 27.5 deg on the comparable duct), and `grad(D)` is `leastSquares` on it. The reference itself has no error bar to hide inside, which is the point of choosing it |
| " | the same under `planeStress no` | the same algebra down the other branch of `readMechanicalProperties.H`: `-E alpha T(y)/(1-nu)`, **exact**, -314.2857 MPa | the same: **RMS 0.162%, max 0.556%** of 314.2857 MPa, hot face **-314.6932 MPa, +0.130%**, `max sigmaEq` **+0.324%** and **0.00%**. The two states' ratio is **1.429471** against an exact 1.428571, **+0.063%** | the same mesh, and the ratio is the cleanest number here because whatever the mesh does to one state it does to the other. Graded as well as the plane-stress run because that one contains E and alpha only as their product and so cannot distinguish a wrong `nu`; these two are 43% apart |

| [plane duct, Re 10](tutorials/templates.md) | `u_max`, the developed pressure gradient and the wall shear | **exact**, three lines of algebra: `u_max = 1.5 u_mean`, `(dp/dx)_kin = -12 nu u_mean / H^2`, `tau_wall/rho = 6 nu u_mean / H` on the **channel height** `H`, with `u_mean` 1 m/s fixed by a uniform inlet and continuity. The case's **Re 10 is on the hydraulic diameter** `2H`, which is the only factor of two available | node-sampled peak **1.4863** against 1.5; `(dp/dx)` **+2.9%** on -120 m/s^2; wall shear **-9.7%** and **-8.7%** from the two `yPlus` sources and **+10.8%** and **+13.5%** from `wallShearStress`; the solver's own continuity error **-9.04e-08** on a 4e-05 m^3/s throughflow | the discretisation, ten cells across, and nothing else: the reference is exact and `u_mean` is set by a boundary condition rather than fitted. The y+ band is wide on purpose -- y+ is squared into tau and the first-cell height varies face to face on a recombined quad mesh -- and what it still catches is a y+ out by sqrt(2) or a defaulted `nu`. Solved end to end in the `journey` job on every pull request |
| [porous slab coefficients](tutorials/templates.md) | `d` and `f` recovered from a measured pressure-drop curve | **exact by construction**: `dp = (mu d U + 0.5 rho f U^2) L` is OpenFOAM's own sink integrated across the slab, so `dp/(L U)` against `U` is a straight line with intercept `mu d` and slope `0.5 rho f`. `U` is the **face (superficial) velocity**, not the pore velocity, and `L` the thickness **along the flow** | the fit run backwards against the four points it came from: worst **7.6e-5%**. The independent Ergun reduction recovers the same `d` to **-2.67e-4%** and `f` to **+2.11e-4%** | nothing, at the coefficient level: both routes are arithmetic and they agree to a rounding error because the shipped curve *is* Ergun's prediction rounded to six figures. **The solve is ungraded**: nothing here has solved the case and quoted a drop against the **503.34 Pa = 418.057 m^2/s^2** the template prints, and the factor between those two is rho, which is the error this case exists to prevent |

[`tutorials/README.md`](tutorials/README.md) is the **tutorial index**: every case above
plus the ones that are not benchmarks, sorted by what it costs to run one, with the
normalisation basis in every row. Three tiers -- numpy only with nothing installed, a case
built from a template where there is an OpenFOAM, and a prompt answered by an agent in a
workspace. [`tutorials/templates.md`](tutorials/templates.md) is the tier 2 page: the
seven recipes in `../openreynolds/toolbox/templates/`, what each builds, and what is known
about the answer before the solver starts.

`tutorials/` also holds the three cheap prompt cases you can run as they are.
`felix/` and `moving_mesh/` hold the four with graders, and both graders run on every
pull request against trimmed copies of the real runs in `tests/data/`.

`thermal_stress/` is the one with no fixture behind it, because a solid-mechanics case
needs OpenFOAM to produce one and the `benchmarks` job has none. Its grader is held to
the closed form by `tests/test_benchmark_thermal_stress.py`, which writes the exact
profile and also writes four wrong ones it has to reject; the solver is held to it by
`tests/test_journey_thermal_stress.py` in the `journey` job, which runs the template,
solves both plane states in the image and runs the grader on what came out, and by the
`solidDisplacementFoam` gate in `docker/workspace/check.sh`, which asserts the same
number out of the solver's own log. Nothing was installed for it: `solidDisplacementFoam`
is standard OpenFOAM and has been in the image since the first build. What was missing
was that nothing here knew it.
`mhd/` and `combustion/` are the two added for the multiphysics items, and both were
chosen for the same reason `thermal_stress/` was: their answer is a formula rather than a
figure in somebody's paper. `mhd/` is Hartmann flow, where a transverse magnetic field
makes the Lorentz force on a fully developed flow a **linear drag**, so the profile is a
constant minus a cosh and `u_c/u_mean = (cosh Ha - 1)/(cosh Ha - sinh Ha/Ha)` -- derived
in `mhd/README.md` from `mhdFoam`'s own two equations and graded by `mhd/grade.py`.
`combustion/` is a `chemFoam` well-stirred reactor, where the solver's own `hEqn.H` and
`solveChemistry.H` conserve the mixture's absolute enthalpy, so the end temperature is one
scalar equation in one unknown over the case's own JANAF coefficients; it has no `grade.py`
beside it because that root find already lives in `openreynolds/toolbox/chem0d.py` and a
second copy would be a second thing to keep agreeing.

**The Hartmann case solves, as of 2026-11-21.** It did not for two rounds: `mhdFoam`
exited -8 on a SIGFPE inside the vector solve, with the Courant number running 15.1 at
t = 0.27 and 38.9 one step later. The cause was the mesh, not the time step. `mhdFoam`'s
two magnetic normal stresses are both `fvc::` and do not cancel discretely, so the
surviving explicit force is of order `B0^2/(mu*rho*dy)` and **contains no `dt`** -- which
is why two reductions of the step did not reach it. The ESI tutorial is stable at a
magnetic Courant number of 2.0 where this case diverged at 0.5, so that ratio is not the
governing quantity either.

What governs it is `dy`, and the tutorial's own `dy` of 0.05 is the one value `mhdFoam` is
known to be stable at -- stable **because** at its Ha of 20 the Hartmann layer `L/Ha` is
exactly one cell, so the tutorial does not resolve the layer at all. This template now
uses that cell height at Ha 5, where four cells still fall inside the layer: resolved, at
a size the scheme tolerates. Measured in the `journey` job:

| | |
|---|---|
| magnetic Courant `c_A*dt/dy` | **0.5000** at Ha 5, against the tutorial's stable 2.0 |
| flow Courant, max over 400 reported steps | **0.0589** at Ha 5, **0.0618** at Ha 0 |
| magnetic flux divergence error | 0 throughout |
| the fourteen journey tests | all pass, graders included |

**The graded profile agreement is not quoted here**, and that is deliberate: pytest
captures a passing test's output, so the log carries the fact that `grade.py`'s bounds
were met and not the numbers. The bounds are `shape_rms < 1.5e-2` and
`shape_max < 5.0e-2` of the centreline speed, and they are discretisation bounds rather
than reference uncertainties, because the profile is exact. A per-cent figure will go here
when a run prints one.

Neither has a fixture under `tests/data/`, for the reason `thermal_stress/` has none: both
need OpenFOAM to produce one and the `benchmarks` job has none. Their graders are held to
their closed forms by `tests/test_benchmark_mhd.py` and `tests/test_toolbox_chem0d.py` in
the suite -- the first writes the exact profile at three Hartmann numbers and five wrong
ones it has to reject at a different check each, the second writes a constant-Cp mixture
whose adiabatic temperature is one line of algebra -- and the solvers are held to them by
`tests/test_journey_mhd.py` and `tests/test_journey_chem0d.py` in the `journey` job, which
mesh, solve and grade inside the workspace image. Nothing was installed for either:
`mhdFoam` and `chemFoam` are standard OpenFOAM and have been in the image since the first
build. What was missing, again, was that nothing here knew it.

The combustion case is the one row in this file whose **published** reference is not the
graded quantity. What is graded is the solver against its own thermodynamics; what is
compared with a measurement is the thermo data itself, species by species, against NIST's
Chemistry WebBook gas-phase formation enthalpies. That is deliberate, and
`combustion/README.md` says why: a single irreversible global step makes no CO, OH, H, O or
NO, so the temperature it produces is the complete-combustion one and a real flame of the
same mixture is cooler. How much cooler is **not asserted**, because no primary source for
it has been read here, which is the same position the dam-break row above is in.

`lbm/`, `sph/` and `particles/` are the three whose graders **solve** rather than reading a
finished run,
and the difference is worth stating rather than hiding. The four above grade an OpenFOAM
or Felix case that cost a cluster an hour, so the run happens once and its output is
trimmed into `tests/data/`. Those three grade `openreynolds/toolbox/lbm.py`, `sph.py` and `tracks.py`,
which are numpy in this repository and finish inside the test job, so committing their
output would replace a check that re-derives the number with a number nobody can
re-derive. The whole of the three costs about sixteen minutes on a runner, which is the
largest single thing in that job and the first place to look if it needs to be cheaper.

## The five ways a validated case misses, in the order they are usually blamed

Across these cases the blame falls in a very different place from where it is first put,
and the order matters because the first explanation is the expensive one to believe.

**The reference value, which is last on everyone's list and was wrong twice here.** Of the
eleven reference numbers in the table above, re-checking them against their sources
changed two. de Vahl Davis's 4.519 is his cavity average, not his hot-wall value, which is
4.509; the misattribution is near-universal in secondary sources and it is worth 0.22%,
against a paper whose own stated accuracy is 0.3%. Gartling's 6.1 is in channel heights
and had been read here as step heights, which is a factor of two. Both errors flatter the
result, which is why neither was noticed: a reference that is wrong in the direction of
agreement is invisible, and the only defence is reading the source rather than the
secondary citation. Where the source could not be read, as with Gartling, the table above
says so instead of asserting a figure, and the gap is tracked as issue #205 rather than
left to be rediscovered.

**And the reference's own uncertainty, which on four of these cases is larger than the
agreement being claimed.** Ghia's Re 1000 centreline extrema are 1.5% to 2.2% from a later
spectral solution, so a run agreeing with him to 0.45% has agreed to inside his error.
Williamson's parallel-shedding fit gives 0.1643 at Re 100 while Roshko's and Williamson &
Brown's give 0.1671, a 1.7% disagreement among published correlations against a result
reported as 1.31% high. de Vahl Davis states 0.3% for himself. On the heated pipe the
reference is exact and the uncertainty moves to the measurement instead: four defensible
wall-gradient stencils span 0.7% against an error of 0.55%. A validation that quotes a
tighter agreement than the reference can support is not more accurate than one that does
not, it is just less careful, and the way to tell is to put the reference's own error bar
next to the number every time.

**Discretisation, which is the first guess and is usually spent.** On the submarine hull
below, two meshes 3.04x apart in cell count agreed to 0.97%, and Richardson extrapolation
on the pair moved the answer by a little over a percent. The mesh was already fine enough
and the remaining 4% was somewhere else. The only way to know that is a second mesh, and
a case with one mesh has no claim about discretisation at all. Note also that the
reference can be the discretised one: the lid-driven cavity is checked against numbers
computed in 1982 on a 129x129 grid, so part of the disagreement there belongs to Ghia.

**The reference experiment's own assumptions.** A towing tank trips its boundary layer;
fully turbulent k-omega SST from the nose does not, and at Re 1.3e7 that is worth a few
percent in the direction observed. A correlation fitted to banks of twenty tube rows is
being asked about seven. A propeller database's chord and twist are measured and its
aerofoil sections are a reconstruction. None of these is a solver error and all of them
look like one.

**The turbulence model and the near-wall treatment, which is where a y+ in the buffer
layer hides.** Between y+ 5 and 30 neither a wall function nor a resolved sublayer
applies, so a sequence of meshes landing in that band produces a spread that looks like a
convergence trend and is three invalid points. On a hull here it moved a form factor by
29% while the published round-robin sees at most 4.5% across a valid y+ range, which says
the mesh was implicated and not only the target.

**The quantity's own conditioning.** Drag converges early because it is dominated by one
region; lift on the same case is a small difference of large integrals and converges far
later. Reporting both to the same confidence is the error, and the 2D-1 row above is what
it looks like when it is caught.

## A case study: an impossible number, and a reference somebody typed

A submarine hull, the DARPA SUBOFF bare hull (AFF-1), 4.356 m long, towed at 3.046 m/s,
Re about 1.2e7, uploaded as a surface and asked about in plain English.

The towing-tank resistance at that speed is **87.4 N**. The locator is worth having
exactly, because the appended configuration at the same speed is 102.3 N and picking the
wrong row is a 17% error that looks like a solver problem: Liu & Huang (1998), *Summary
of DARPA SUBOFF Experimental Program Data*, CRDKNSWC/HD-1298-11, **Table 14, p. 23**,
configuration 1, bare hull, 5.92 knots, 19.65 lbf. The pound reading is the measurement
and 87.40 N is its conversion, and 5.92 knots is 3.0455 m/s. That table is reproduced
from Crook (1990), DTRC/SHD-1298-07, which is the primary source. The wetted area is
5.988 m2 for the bare hull, from Groves, Huang & Chang (1989), DTRC/SHD-1298-01; the
appended hull's 6.338 m2 is the other number not to normalise on.

A total resistance coefficient of 3.15e-3 follows from those, and it is worth saying that
this is **derived and not published**: Table 14 tabulates a residual resistance
coefficient of 0.00030 and no C_T at all.

The solve was good. It measured the hull's wetted area off the uploaded surface as
5.98757 m2 against the report's 5.988, which is the check that the upload, the units and
the geometry handling were all right at once. The force on the half model it meshed was
43.74 N, and doubled for the symmetry plane that is **87.48 N**, 0.09% from the published
figure. A grid check later put the mesh-converged coefficient at 3.02e-3 against 3.15e-3,
so about 4% low, in the direction and of the size that fully turbulent RANS against a
tripped experiment is expected to be.

The delivered answer was half of that, because the force on the half model was divided by
the wetted area of the whole hull: half the numerator, all of the denominator.

The part worth keeping is not the slip, it is what failed to catch it. The same analysis
computed the ITTC-57 friction line at that Reynolds number, 2.907e-3, and reported a
**total** drag coefficient of 1.58e-3 beside it. No body with any form drag can have a
total below the flat-plate friction coefficient at the same Reynolds number and the same
area; it is the fluid-dynamics equivalent of negative fuel consumption. Both halves of
the contradiction were derived in the same paragraph, and the contradiction was resolved
in favour of the wrong number, because the prompt had supplied a reference value that was
itself wrong and the wrong answer sat neatly inside it.

So the lesson is about reference values and not about arithmetic: **an independently
derived physical bound is not negotiable against a number somebody typed, including a
number in this file.** The ITTC-57 line, Archimedes, a closed-form torque, a conservation
statement and the divergence theorem are all cheap, and all of them outrank a quoted
expectation. `openreynolds/toolbox/ladder.py` exists for that reason: it names, for the
class of case it detects, the answers that are known without running anything.

## What is not here, and why

**No aerofoil case.** A NACA 0012 at 6 degrees and Re 3e6 was run against Abbott & von
Doenhoff and Ladson, and it came back with a drag coefficient roughly seven times the
published one on an O-grid whose y+ reached the hundreds under a wall function that does
not apply there. The run said so about itself. It is a real finding about meshing and a
useless tutorial, because nothing in it would teach a reader anything except that the
mesh was wrong, so it is recorded here as a gap rather than shipped as an example.

**No flat plate.** Blasius gives an exact skin friction and would make an excellent
first case, and no run of that configuration exists in this repository to report a result
from. Rather than ship a reference with no measurement beside it, it is listed as missing.

**No figures.** The cases carry numbers, not pictures. A figure that disagrees with the
number beside it is worse than no figure, and that is not a hypothetical: the
backward-facing step run in `tutorials/` delivered a plot annotated 7.57 beside a written
answer of 6.4, because the script that drew the picture took the third wall-shear sign
change where the answer took the second. Neither number was reconciled with the other,
and the figure is what a reader looks at.

**No grader for the three `tutorials/` cases.** The quantities they turn on are already
measured by instruments in `openreynolds/toolbox/` -- `heat.py` for a wall Nusselt number
against a bulk reference temperature, `reattach.py` for a reattachment length off the
wall shear, `balance.py` for whether an energy or mass budget closes. A fourth
implementation inside a grader would be a fourth thing to keep right. What is missing
instead is trimmed output from those three runs in `tests/data/`, the way `felix/` and
`moving_mesh/` carry theirs, which is what would let the comparison be re-derived here
rather than quoted.
