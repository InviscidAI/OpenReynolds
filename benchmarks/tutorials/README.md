# Tutorial cases

**The index of every case this repository can run whose answer comes from somewhere other
than a solver**, sorted by what it costs to run one. Each row carries the command, the
reference, **the length and velocity scales that reference is normalised on**, and the
agreement a run really measured. Rows with no verdict say so in words.

Three of these cases are written out in full further down this page, because they are the
three cheapest routes in and because one of them is a worked failure worth more than its
agreement. Every other row points at the directory that carries its own account: one
account per case, because fifteen cases summarised twice is fifteen chances for the two
summaries to drift apart, and the thing that drifts is always the reference value.

**Why the normalisation basis is in every row rather than in a footnote.** Of the
reference values this repository quotes, re-checking them against their sources changed
two, and **both errors flattered the result**, which is why neither had been noticed. One
was a factor of two: Gartling's reattachment length of 6.1 is in channel heights and had
been read here as step heights, and the wrong reading reached a customer-facing page. The
other was a misattribution: de Vahl Davis's 4.519 is his cavity average, not his hot-wall
value, which is 4.509. Neither was a solver problem and neither was arithmetic. Both were
a number read without its scale. [`../README.md`](../README.md) carries the full account
and the five ways a validated case misses, in the order they are usually blamed.

## Tier 1: numpy only, nothing installed, no account

`pip install openreynolds` and these run on a laptop. No OpenFOAM, no workspace, no
model, nothing billed. They are the first things to run if the question is "does any of
this work", because the answers are exact and the whole chain is in this repository.

| case | run it | the answer, and what it is normalised on | what a run measured |
|---|---|---|---|
| plane Poiseuille in a lattice Boltzmann channel, [`../lbm/`](../lbm/) | `python benchmarks/lbm/grade.py poiseuille` | `u(y) = G/(2 mu) y (H - y)`, **exact, closed form, no paper needed**. Lattice units throughout: `dx = dt = 1`, `c_s^2 = 1/3`, `mu = rho nu` with `nu` in lattice units, and the force scaled so `u_max` is 0.05 so the scheme's `O(Ma^2)` error stays small | max error **5.17e-11** of `u_max` at nu 0.02, **1.22e-11** at 0.1, **3.30e-12** at 0.4. The same code with one relaxation time, four times from its single exact viscosity, is wrong by **3.66e-3** -- which is what makes the first three numbers evidence rather than a solver agreeing with itself |
| the lid-driven cavity at Re 1000, [`../lbm/`](../lbm/) | `python benchmarks/lbm/grade.py cavity --n 128` | Ghia, Ghia & Shin (1982), *J. Comput. Phys.* **48**, 387-411, Table I (`u` on the vertical centreline) and Table II (`v` on the horizontal). `Re = u_lid L / nu` on the **cavity side** `L` with the **lid speed**; the profiles are normalised on the lid speed and the side | u RMS **0.736%**, v RMS **0.749%** of the lid speed, against the same 34 numbers and the same 2% bound the Felix solver is held to. **Inside Ghia's own error**: his extrema sit 1.46% to 2.20% from Botella & Peyret's (1998) spectral benchmark |
| hydrostatic pressure in still SPH water, [`../sph/`](../sph/) | `python benchmarks/sph/grade.py hydrostatic` | `p = rho g (h - y)`, **exact**. Normalised on `rho g h` with `h` the still-water depth, which is the scale that makes a density error visible: `dp/drho` is about `c0^2`, so a 0.1% density error is tens of per cent of `rho g h` | RMS **2.50%** of `rho g h`, worst particle **8.61%** at the free surface, where the kernel support is truncated. It read **67%** before a density diffusion term was added |
| the first sloshing mode in a tank, [`../sph/`](../sph/) | `python benchmarks/sph/grade.py sloshing` | `omega^2 = g k tanh(k h)` with `k = pi/L`: linear theory, **exact** (Lamb, *Hydrodynamics*, 6th ed., art. 228). `L` is the tank length, `h` the still depth, and the amplitude has to be small for linear theory to be the right reference at all -- 5% of the depth here | **-4.75%**, 1.1257 s against 1.1818 s. **The sign is the diagnosis**: short means the effective tank is shorter than the one asked for, which is the wall-offset error dynamic boundary particles have |
| a dam break, [`../sph/`](../sph/) | `python benchmarks/sph/grade.py dambreak` | Martin & Moyce (1952), *Phil. Trans. R. Soc. A* **244**, 312-324. **The paper could not be read** -- the publisher returns 403 and no open copy was found -- so which non-dimensionalisation it defines is unverified, and `t sqrt(2g/a)` and `t sqrt(g/a)` are both in circulation and differ by 1.414 at every point | **no verdict, deliberately.** The surge front is printed: Z = 0.975 at T = 0 rising to 3.943 at T = 3.00. What *is* graded is containment and weak compressibility holding, which are the solver's own business and need no paper |
| an aerofoil and a wing by panel method, [`../../openreynolds/toolbox/panel.py`](../../openreynolds/toolbox/panel.py) | `python openreynolds/toolbox/panel.py --naca 0012 --alpha 5` | Four statements, all exact and none of them a correlation: **d'Alembert** makes the pressure drag of a closed body in steady potential flow exactly zero; a **symmetric** section's zero-lift angle is exactly 0 and its quarter-chord moment exactly 0; a **NACA 4-digit** camber line's zero-lift angle is the closed-form thin-aerofoil integral over its own ordinates; and an **elliptically loaded** wing's induced drag is exactly `Cl^2/(pi AR)`. Chord-normalised throughout, inviscid, no Reynolds number anywhere in it | NACA 0012 at alpha 5: pressure drag **2.19e-04** against an exact 0, zero-lift angle **+0.000 deg** against an exact 0, `Cm_c/4` **-0.0065** against an exact 0. NACA 2412 at the default 160 panels: zero-lift angle **-2.118 deg** against the thin-aerofoil integral over that same camber line, **-2.0772 deg** (computed independently to seven figures, and again in `tests/test_toolbox_panel.py`), so **+2.0%** more negative -- signed the way thickness requires, since the integral has none. **And the panel count is worth the same as the thickness here**: the same section reads -2.138 at 400 panels and -2.144 at 800, so the honest statement is a couple of per cent and not the third figure. Wing, AR 8, square tip: induced drag **0.01771** against the elliptic **0.0166971**, a span efficiency of **0.9428**, which is the square tip and not an error, and mode-converged -- identical to four figures at 24, 48 and 96 Fourier modes. The **lift slope is 1.100 x 2 pi** and is reported rather than graded, because 2 pi is the zero-thickness limit and a 12% section belongs above it |

## Tier 2: a case built from a template, solved where there is an OpenFOAM

[`templates.md`](templates.md) is the page for these: the seven recipes in
`openreynolds/toolbox/templates/`, each one copied into an empty directory and run, what
each builds, and **what is known about the answer before the solver starts**. Two of the
seven end in an exact comparison that CI runs on every pull request; the other five are
honest about having no reference, which is a different thing from having no value.

| case | run it | the answer, and what it is normalised on | what is measured today |
|---|---|---|---|
| a plane duct, `duct2d.py` -> `case_gen.py` -> `simpleFoam` | `python3 duct2d.py`, then `case_gen.py --speed 1 --reynolds 10 --length 0.04` | developed plane Poiseuille flow, **exact**: `u_max = 1.5 u_mean`, `(dp/dx)_kin = -12 nu u_mean / H^2`, `tau_wall/rho = 6 nu u_mean / H`. Three lines of algebra on the channel height `H` with `u_mean` fixed at 1 m/s by a uniform inlet and continuity; the case's `Re = 10` is on the **hydraulic diameter** `2H`, which is the one place a factor of two could hide | solved end to end in the `journey` job on every pull request: the node-sampled peak **1.4863 m/s** against the exact 1.5 (the nodes average their two cells, so it reads low) and the same samples' peak-to-mean **1.532**, `(dp/dx)` **+2.9%** on the exact -120 m/s^2, wall shear bracketed **-9.7% to +13.5%** across four independent y+ sources, the solver's own continuity error **-9.04e-08** on a 4e-05 throughflow. `tests/test_journey_duct.py:107-180` carries every tolerance with the reason it is that number |
| thermal stress in a constrained bar, `thermal_stress.py` -> `solidDisplacementFoam` | `python3 thermal_stress.py`, then `solidDisplacementFoam` | `sigma_xx(y) = -E alpha T(y)` under plane stress and `-E alpha T(y)/(1-nu)` under plane strain, **exact**, derived in [`../thermal_stress/README.md`](../thermal_stress/README.md) from the three kinematic facts the boundaries impose. **-220.0000 MPa** and **-314.2857 MPa** on the hot face for the template's steel. Normalised on nothing: these are pascals, and the reference has no error bar to hide inside | RMS **0.094%** and **0.162%** of the peak, hot face **+0.130%** either way, and the two states' ratio **1.429471** against an exact 1.428571 (**+0.063%**), which is the cleanest number here because whatever the mesh does to one state it does to the other. The temperature field in the same solve has its own exact scale: the bar's thermal e-folding time is **2.2834 s** for the shipped steel, which is why `END_TIME_S` is what it is rather than a round number |
| a Darcy-Forchheimer porous slab, `porous.py` -> `porousSimpleFoam` | `python3 porous.py`, then `porousSimpleFoam`; the coefficient arithmetic alone needs no OpenFOAM at all | `dp = (mu d U + 0.5 rho f U^2) L`, which is OpenFOAM's own sink integrated across the slab, so it is **exact by construction** rather than a model. `U` is the **face (superficial) velocity**, not the pore velocity, and `L` is the slab thickness along the flow -- a drop quoted without its thickness is half a measurement. The shipped curve is Ergun's prediction for 3 mm spheres at a void fraction of 0.38 | the derivation run backwards against the four points it came from: worst **7.6e-5%**, at 1.5 m/s. The independent Ergun route recovers the same `d` to **-2.67e-4%** and `f` to **+2.11e-4%**. **The solve itself is ungraded here**: nothing in this repository has solved this case and quoted a drop against the 503.34 Pa the template prints, and until something has, that is the state |
| a sphere in a flow box, `body_in_box.py` | `python3 body_in_box.py` | no reference is shipped and **none is claimed**. The template's job is the body-fitted mesh, and `checkMesh` is the only verdict it offers | the mesh, by `checkMesh` and `mesh_look.py`. Stokes drag `Cd = 24/Re` is exact below Re about 1 and would make this a graded case; the shipped box is not sized or dressed for that regime, so it is listed as a gap rather than quoted as a check |
| a two-phase dam break, `free_surface.py` -> `interFoam` | `python3 free_surface.py`, then `interFoam` | the same Martin & Moyce (1952) the SPH row above declines to quote, for the same reason. What **is** exact and free before release: the hydrostatic pressure at the base of the undisturbed column, `p_rgh = rho g h`, and total water volume conserved to the VOF solver's own `alpha` bounds | `setFields` writing a real interface, and `checkMesh`. **No solved result exists in this repository**, so no agreement is quoted |
| a frozen-rotor bladed mixer, `rotating.py` -> `simpleFoam` + MRF | `python3 rotating.py`, then `simpleFoam` | no closed form for a six-bladed vessel, and a frozen rotor is one blade angle rather than an average. Two statements are still exact: in a steady state with no through-flow the **torque on the impeller and the torque on the vessel wall are equal and opposite exactly**, by conservation of angular momentum, and **power = torque x omega**. `omega` is 104.719755 rad/s for the shipped 1000 rpm, a factor 9.5493 from the nameplate units | the mesh and its two cell zones. The torque balance is the check [`../moving_mesh/`](../moving_mesh/) found teeth in: a `cyclicAMI` interface held its *area* to 43 ppm while the two wall torques disagreed by **0.911%**, so the check every rotating case runs passed one whose torque was a per cent wrong. Nothing has run it on this template yet |

## Tier 3: a prompt, answered by an agent in a workspace

These cost a model turn and a workspace, so they are **not** run by CI and not run by
anything automatically. Each is one prompt in a user's own words, with the reference named
inside the prompt, so the comparison is part of the answer rather than something done to
it afterwards.

| prompt | what it tests | the reference, and what it is normalised on | what a run measured |
|---|---|---|---|
| `pipe_nu.txt` | an axisymmetric wedge, a scalar transported on a converged flow, and a Nusselt number taken against a mixed-mean reference temperature | the Graetz problem, Nu = `lambda_1^2/2` = **3.6568**, **exact**. `Nu = h D / k` on the **diameter**, with `h = q_wall/(T_wall - T_bulk)` and `T_bulk` the **mixed mean**, not the area average. Re 100 is on the diameter | **3.677 as reported, +0.55%**; 3.628 to 3.654 recomputed four defensible ways. Sub-percent either way, and **the third figure is not established** |
| `cavity_ra1e5.txt` | buoyancy-driven flow, a graded mesh, and an energy balance across two walls | de Vahl Davis (1983), *Int. J. Numer. Methods Fluids* **3**, 249-264, Table V: **4.509** on the hot wall, **4.519** for the cavity average. `Ra = g beta dT L^3/(nu alpha)` on the **cavity side** `L`; his velocity maxima are non-dimensionalised on **alpha/L**, not `nu/L`, and mixing those up is a factor of Pr | hot wall **4.5335, +0.54%**, and the cold wall agreeing to seven figures, which is a closed energy balance and is free |
| `bfs_re800.txt` | a separating and reattaching flow, a reattachment length off the wall shear, and which length a benchmark is normalised on | Gartling (1990), *Int. J. Numer. Methods Fluids* **11**, 953-967. **6.1 is in channel heights**, which is 12.2 step heights, and his Re 800 is on the **channel height** with the **mean inlet velocity**. **No primary access**: the paper is paywalled and was not read, and the normalisation rests on reproductions, which are listed below | **6.42 step heights = 3.21 channel heights, about 47% short**, and still climbing when the run stopped. **Not a validation.** It is kept because working out what it should be compared against found the factor of two |
| [`../felix/cavity_re1000.txt`](../felix/) | the Felix solver on the same cavity the LBM row above solves | the same Ghia Tables I and II, the same 34 numbers, the same 2% bound -- and therefore the same normalisation: `Re = u_lid L / nu` on the **cavity side** with the **lid speed**, and both profiles scaled on the lid speed. Importing one table rather than typing two is what makes the comparison between the two solvers mean anything | u RMS **0.448%**, v RMS **1.063%**, and the same caveat: inside Ghia's own 1.46% to 2.20% |
| [`../felix/cylinder_re20.txt`](../felix/) | drag and lift on a cylinder in a channel | Schafer & Turek (1996) DFG benchmark **2D-1**, Cd 5.5795 with their interval 5.57-5.59, Cl 0.0104-0.0110. Cd is on the **cylinder diameter** and the **mean inlet velocity** | Cd **5.522, -1.04%**: inside the 2% bound, outside the reference interval. Cl **0.0010, reported and not graded**, because lift here is a small difference of two large surface integrals and needs resolution the drag does not |
| [`../moving_mesh/couette_ami.txt`](../moving_mesh/) | a `cyclicAMI` sliding interface cut through a circular-Couette annulus | the closed form for circular Couette torque, **exact**, and -- better -- conservation of angular momentum, which makes the two wall torques **equal and opposite exactly**, with no paper and no constant | inner-wall torque **+0.43%**; the two torques disagreeing by **-0.911%**, **a failure**, against 0.011% on the same annulus meshed conformally |
| [`../moving_mesh/viv.txt`](../moving_mesh/) | a cylinder on a spring, free to move across the stream at Re 100 | St **0.1643** from Williamson (1989) *JFM* **206**, eq. (6), the parallel-shedding fit, at Re 100 on the **diameter**; and net fluid work per cycle **exactly 0**, because the structural damping is zero | St **+1.10%** on that fit, and **the published correlations disagree by more than that**: Roshko and Williamson & Brown both give 0.1671, against which the same result is -0.55%. The energy audit **failed at -10.1%** and found a limit cycle 13% too wide; A/D is **reported with no verdict** because no published value for this configuration was found |

## What is in the tiers and what is not

**Instruments, not cases.** `grid.py` (Richardson, observed order, GCI), `spectra.py`
(FFT, Strouhal, settling), `wall.py` (y+ measured off a finished run), `heat.py`,
`balance.py`, `reattach.py`, `acoustics.py` and the renderers `showcase.py`,
`render.py` and `animate.py` measure or draw the cases above rather than being cases
themselves, and each is pinned by its own tests against an exact statement -- a
manufactured series of known order, a synthetic tone of known frequency, a wall shear of
known value. They are listed in
[`../../openreynolds/toolbox/README.md`](../../openreynolds/toolbox/README.md) with what
each is for, and the table further down this page names the four that the three Tier 3
prompts turn on.

**Generators, not cases.** `case_gen.py`, `compressible.py`, `site_abl.py` and
`site_stl.py` write cases, inlets and geometry rather than being cases themselves, and
`site_stl.py`'s own check is a geometric one rather than a physical one: the terrain and
building surfaces it writes are checked for closure and winding by `cad_audit.py`, not
against a measured flow. `site_abl.py` has the nearest thing to a graded
claim: the Richards & Hoxey profile it writes is held to the solver's own expression
to **1e-12**, and `k` is held constant with height to the same, by
`tests/test_toolbox_site_abl.py:79-111`. That is an identity between two pieces of
arithmetic and not a validation against the atmosphere, and it is quoted that way. No
`compressible.py` case is graded against anything yet; that is a gap and is named as
one.

**Cases that were run and are deliberately not shipped.** A NACA 0012 at 6 degrees and
Re 3e6 came back with a drag coefficient roughly seven times Abbott & von Doenhoff's, on
an O-grid whose y+ reached the hundreds under a wall function that does not apply there.
The run said so about itself. It is a real finding about meshing and a useless tutorial,
so [`../README.md`](../README.md) records it as a gap rather than shipping it as an
example. There is also no flat plate, which is the most annoying absence here: Blasius
gives an exact skin friction and would make the best first case of all, and no run of
that configuration exists to report a result from.

## Running one from each tier

```sh
# tier 1: nothing but this package. exact answers, no workspace, no model, nothing billed
python benchmarks/lbm/grade.py poiseuille
python benchmarks/sph/grade.py sloshing
python openreynolds/toolbox/panel.py --naca 2412 --alpha 5

# tier 2: an OpenFOAM, which in practice means the workspace image. see templates.md
cp openreynolds/toolbox/templates/porous.py /tmp/slab/ && cd /tmp/slab && python3 porous.py

# tier 3: a workspace and a model turn, which cost money
openreynolds -p "$(cat benchmarks/tutorials/pipe_nu.txt)"
openreynolds pull --study <study-id>          # the case directory, home
```

The five grader commands in tier 1 are the ones the `benchmarks` job runs on every pull
request, and `panel.py`'s four exact checks are asserted by `tests/test_toolbox_panel.py`
in the `test` job rather than as a command, so a tier 1 row that stops being true turns a
build red rather than aging quietly in a document.
The tier 2 pair that end in an exact comparison run in the `journey` job, inside the
image. **Nothing runs tier 3 automatically and nothing should**: those cost a workspace
and a model turn against a real account.

## What measures the answer

The three tier 3 prompts turn on quantities the toolbox already measures, which is why
there is no grader in this directory:

| the quantity | what reads it | the part that is easy to get wrong |
|---|---|---|
| wall Nusselt number | `heat.py --reynolds --prandtl` | which reference temperature. The same wall gives three different Nusselt numbers against a free-stream, an inlet and a mixed-mean reference, and only the mixed mean is what an internal flow's Nu is defined on. |
| whether the heat budget closes | `heat.py` per patch | a hot wall and a cold wall that disagree have not converged, whatever the residuals say. It costs one extra integral. |
| reattachment length | `reattach.py` | which sign change, and in what unit. The first is the corner eddy; a later one may be a secondary bubble. |
| whether a transient has settled | `spectra.py` | a record that is stationary and spectrally clean can still be held there by the numerics rather than by the physics. |

## The three tier 3 cases in full

The rest of this page is the long account of the three prompts above, which is where the
reference values earned their re-checking. The numbers were recomputed for this file from
each run's own final field and mesh, not copied from the run's summary, and the reference
values were re-checked against the sources rather than against the repository. That
exercise changed three of them, so the recomputation is what is quoted and the
disagreement is noted where it exists.

### The heated pipe, against an exact answer

A 150 x 40 axisymmetric wedge, half-angle 2.5 degrees, radius 0.01 m and length 0.3 m,
graded 0.4 toward the wall, the collapsed axis edge typed `empty` and the two wedge faces
`wedge`. `simpleFoam` to converge the flow, then a scalar transport for the energy
equation, which is legitimate here because the properties are constant and the temperature
does not feed back into the momentum. The case ran with nu = 1.5e-5 and DT = 2.152e-5, so
Pr = 0.697, with the wall at 350 K against a 300 K inlet.

**The run reported Nu = 3.677 over x/D > 10, which is +0.55% on the exact 3.6568.**

The flow is genuinely developed where that average was taken. The hydrodynamic entrance
length is about 0.05 Re D = 0.1 m and the thermal entrance length about
0.05 Re Pr D = 0.07 m against a 0.3 m pipe, and the centreline velocity in the final field
is 1.991 times the mean against a parabolic 2.

**What the case actually teaches is that the third significant figure is not established.**
Recomputing the same quantity from the run's own final `T` and `U`:

| wall gradient | radial cell position taken as | Nu over x/D > 10 |
|---|---|---|
| secant to the wall | annular centroid | 3.654 |
| secant to the wall | midpoint of the radial levels | 3.650 |
| quadratic through the wall and two cells | annular centroid | 3.635 |
| quadratic through the wall and two cells | midpoint of the radial levels | 3.628 |

Every one of those is a defensible way to take the number, they span 0.7%, and the
discrepancy being reported is 0.55%. **The post-processing convention moves the answer by
more than the error does.** So the honest claim is sub-percent agreement with an exact
result, which is a real and useful statement, and not a figure in the third place. There
is a second trap in the reference itself: the textbook rounding 3.66 is 0.09% above
3.6568, so comparing against 3.66 manufactures a bias of its own at this accuracy, and
4.364 (= 48/11) is the constant-heat-flux answer rather than this one.

Reference: Nu = lambda_1^2 / 2 with the first Graetz eigenvalue lambda_1^2 = 7.3136, so
Nu = 3.6568; solving that eigenvalue problem directly gives 3.65679. The problem is
Graetz (1883, 1885) and Nusselt (1910), properly the Graetz-Nusselt problem; the accurate
eigenvalues are Brown, *AIChE J.* **6**, 179-183 (1960). Incropera & DeWitt Table 8.4
carries it as Nu = 3.66 (Eq. 8.55) against 4.36 for constant flux (Eq. 8.53).

### Natural convection at Ra 1e5, and a reference that is widely misquoted

A 40 x 40 mesh symmetrically graded toward all four walls, differentially heated vertical
walls, adiabatic top and bottom, solved steady with a Boussinesq buoyant solver.
`checkMesh` clean: maximum non-orthogonality 0, skewness about 7e-14.

The setup is worth copying because the Rayleigh number was reached by choosing the one
free quantity rather than by bending a property. With g = 9.81, beta = 3.3333e-3,
nu = 1.5e-5, Pr = 0.71 and L = 0.1 m, the walls were set to 300.485 K and 299.515 K, a
difference of 0.970 K, giving Ra = g beta dT L^3 / (nu alpha) = 9.999e4 and Pr = 0.710.

**Mean Nu on the hot wall 4.5335, against de Vahl Davis's hot-wall 4.509: +0.54%.**

The reference needed correcting and this is the part worth reading. de Vahl Davis reports
*several* Nusselt numbers at Ra 1e5 in his Table V, and they are not interchangeable:

| his symbol | what it is | Ra 1e5 |
|---|---|---|
| `Nu_0` | the average on the vertical boundary at x = 0, the **hot wall** | **4.509** |
| `Nu` | the average throughout the cavity | 4.519 |
| `Nu_1/2` | the average on the vertical mid-plane | 4.519 |

`Nu` and `Nu_1/2` coincide to four figures here, and secondary sources routinely cite
**"de Vahl Davis hot-wall Nu = 4.519"**, which is his cavity-average and not his wall
value. The two are 0.22% apart, and his own estimated accuracy at Ra 1e5 is 0.3%, so the
misattribution is the same size as the reference's uncertainty and larger than it is
comfortable with. A wall-integrated Nusselt number, which is what this run computes and
what `heat.py` computes, belongs against **4.509**. Measured against the commonly quoted
4.519 the same result reads +0.32%, which is how a validation can be half a percent
wrong in the flattering direction without anyone touching the solver. He also recommends
`Nu_1/2` as the number to describe heat flow across a cavity with adiabatic ends, on the
grounds that it contains both wall values.

The check that makes the result believable is not the agreement with the benchmark, it is
the agreement with itself: the mean Nusselt number on the **cold** wall recomputes to
4.5335 as well, matching the hot wall to seven significant figures. With adiabatic top and
bottom in a steady state every watt entering one vertical wall leaves through the other
exactly, so the two walls agreeing is a closed energy balance, and it is free. The local
Nusselt number along the hot wall runs from 0.720 at the top to 7.791 at the bottom, which
is the textbook shape, and de Vahl Davis's own extremes are Nu_max 7.717 at height 0.081
and Nu_min 0.729 at the top.

Convergence as a measurement rather than a tolerance: at the final iteration the initial
residuals were 5.2e-8 on Ux, 3.9e-8 on Uy, 3.3e-8 on T and 6.7e-8 on p_rgh, with the local
continuity error at 8.3e-11. Ra 1e5 is far below the transition where this geometry needs
transient treatment.

Where the remaining 0.54% comes from: a 40 x 40 mesh, a one-sided wall gradient at the
first cell centre, which is the same stencil question the pipe case is dominated by, and
0.3% of reference uncertainty. Two more free checks are in the prompt: his mid-plane
velocity maxima are 34.73 and 68.59, non-dimensionalised on **alpha/L** and not nu/L, and
mixing those two scales up is a factor of Pr and the usual reason a correct solve appears
to fail this benchmark.

Reference: de Vahl Davis, *Int. J. Numer. Methods Fluids* **3**, 249-264 (1983), Table V
on p. 256, with the definitions on p. 252 and the section "Which Nusselt number?" on
pp. 262-263. The error estimates are his Table VII, p. 262.

### The backward-facing step, where the units are the whole case

This one is not shipped as a passing validation. It is shipped because working out what
it should be compared against found an error that this repository asserts in three places,
and the error is a single mistake that produces two factors of two.

The run built, in its own `blockMeshDict` comment, "step height h=1, expansion ratio 2":
an inlet duct over y in [1, 2], a step face at x = 0 over y in [0, 1], a downstream channel
over y in [0, 2] and 30 length units long, with an inlet profile u = 6 y'(1 - y') whose
mean over the inlet is exactly 1, and nu = 0.00125. Its stated Reynolds number is
`Re = Umean*h/nu = 1*1/nu = 800` on the **inlet** channel height.

Gartling's case is the same shape with the unit length chosen differently: his downstream
channel is height 1 (y from -0.5 to 0.5), his inlet is the upper half, his step is
therefore 0.5, his inlet mean velocity is 1 and his viscosity is 0.00125, so his
Re = u_mean H / nu = 800 is on the **downstream channel height**.

Set those side by side and the run's case is Gartling's geometry scaled up by a factor of
two with the same viscosity and the same velocity. That is one mistake, taking Gartling's
unit length to be the step height when it is the channel height, and it has two
consequences:

**The Reynolds number is twice the benchmark's.** Same nu, same u_mean, every length
doubled, so Re is doubled on whichever length you measure it: 1600 on the downstream
channel height where Gartling's is 800. This part needs no reading of Gartling's text at
all, only his geometry, and it is a geometric fact rather than a naming dispute.

**The reattachment length is in a different unit.** Recomputed from the final
`wallShearStress` field and the mesh, the streamwise wall shear changes sign at 0.21,
6.42, 7.57, 9.70 and further downstream; 0.21 is the counter-rotating corner eddy and
6.42 closes the primary bubble. Those are in the run's length unit, which is the step
height, so 6.42 step heights, or 3.21 channel heights. Gartling's 6.1 is in his unit
length, the channel height, which is 12.2 step heights.

So the comparison the run made, and that `reattach.py`'s docstring, `claims.py` and the
internal record all repeat, is **6.42 step heights against 6.1 channel heights**. Taken
in one unit it is not "+5%, good agreement": it is 3.21 against 6.10, about 47% short.
And because the run is also at twice the Reynolds number, where the bubble is longer
rather than shorter in this range, a correct solve of the run's own case should give more
than 12.2 step heights, not less. The two factors of two do not cancel. The unit error
manufactured an agreement and the Reynolds error made the real shortfall worse.

**What the normalisation rests on, since it is not read from Gartling.** Gartling (1990),
*Int. J. Numer. Methods Fluids* **11**, 953-967, is behind the Wiley paywall with no open
copy, so his own table was never opened. The chain is this, and it is worth following
because the first attempt at it reached the opposite answer from a search-engine
paraphrase:

- **MFIX-Exa's reproduction of the case states its own units outright**: `rho = 1`,
  `mu = 0.00125`, `H = 1`, "where H is the expanded channel height", the mean velocity
  over the inlet opening `u_bar = 1`, and `Re = rho u_bar H / mu = 800`. Its domain is
  x in [0, 30] with the expanded channel over -0.5 <= y <= 0.5 and the inlet over
  0 < y <= 0.5, so the channel height is 1, the step height is 0.5, and every length on
  that page is in channel heights.
- Its grid table is headed "Lower-wall eddy vs. mesh resolution (cf. Gartling, Table
  II)", and it matches its own Richardson-extrapolated **6.097** to "Gartling's reported
  value of **6.10**" to within 0.05%, **with no conversion**. That only works if
  Gartling's 6.10 is in those same channel-height units.
- **Abaqus/CFD's sample-problem manual says the convention outright** for the same case:
  "this example simulates the laminar flow over a backward facing step at a Reynolds
  number of 800 **based on channel height**", with the domain `0 < x < 30`,
  `-0.5 < y < 0.5`, density 1 and "viscosity = 0.00125 units (the viscosity is chosen so
  as to set the flow Reynolds number to 800)". Its comparison table then lists the
  Gartling (1990) row as **6.1, 4.85, 10.48**.
- **An independent solver lands on the same number under the same stated convention.**
  Bourantas et al. (arXiv:1906.05387), a meshless point-collocation scheme, define
  `Re = u_avg H / nu` with `H = 1` "ranging from (0, -0.5) to (0, 0.5)" and the step at
  `H/2`, and compute the lower-wall reattachment at **6.1**.
- A published MHD benchmarking table corroborates by doubling, listing the Gartling row
  as `x_r/h = 12.20` and `11.26` with `h` the step height: exactly 2 x 6.10 and
  2 x 5.63. The same table lists Gresho et al. (1993), which Gartling co-authored, with
  identical values.
- The 5.63 is the **upper-wall bubble length**, 10.48 - 4.85, not a rival value for the
  lower-wall reattachment, which is what made the pair of numbers confusing.
- **And the conventions never cross.** Across the accessible literature, every source
  that normalises on the channel height reports 6.10 and every source that normalises on
  the step height reports 12.2. There is no source anywhere reporting 6.10 in step
  heights or 12.2 in channel heights, so there is no genuine dispute about the value,
  only two self-consistent conventions and a factor of two between them.
- Armaly's experiment sits where this reading predicts: at the same flow rate it measures
  about 14 step heights, and a 2D steady computation is expected to underpredict a flow
  Armaly states is three-dimensional above Re about 400. That puts 12.2 in the right
  place and 6.1 nowhere near it.

**So Gartling's 6.1 is in channel heights and his Re 800 is on the channel height with
the mean inlet velocity.** The paper is still unread, and that is said here rather than
implied away.

Two traps for anyone checking this against those sources. Abaqus heads its third column
"length of the upper separation bubble" and puts 10.48 in it, but 10.48 is the upper
reattachment *position*; the bubble length is 5.63. And Teruel & Fogliatto (2013),
*Mecanica Computacional* **XXXII**, 3265-3278, quote Gartling as 11.9 / 9.5 / 20.6 step
heights, which is 5.95 / 4.75 / 10.3 channel heights: they agree on the unit and sit 2%
to 2.5% below everyone else on the value, which looks like numbers read off a figure.
Neither affects the normalisation, and both are the sort of thing that turns into a
third circulating number if nobody writes it down.

**The unread paper is tracked as issue #205**, which records what would close it and what
has already been tried: reading Gartling, or Gresho et al. (1993), which he co-authored
and which carries the same values; and then re-running this case at Re 800 on the channel
height so it becomes a validation instead of a worked units failure. He reports an
upper-wall bubble from 4.85 to 10.48 channel heights as well, which nothing here has
measured and which would be a second independent check on the same solve.

The other two things the run did are worth keeping regardless of the units. It reported
that the pressure residual plateaued around 0.25 to 0.29 out to 15,000 iterations,
oscillating rather than decaying, and that at this Reynolds number the 2D laminar step is
at or past the threshold where a steady solution stops existing, so the "steady, laminar"
premise was itself questionable. Recomputing the reattachment at each of its writes gives
5.92, 6.18, 6.32 and 6.42 at t = 2500, 3000, 14500 and 15000: still climbing when it
stopped, so the number was never converged in any unit.

And the figure it delivered is annotated 7.5667 while its text said 6.4. The 7.5667 is not
a second estimate that disagreed; it is the **third** sign change in the same list at the
same time, so the script that drew the picture took `crossings[2]` where the answer took
`crossings[1]`. `reattach.py` exists because of that: one implementation, which says which
crossing it chose and why, and which carries its own number into the figure it draws.

## What these three deliberately leave out

**No grader, and no fixtures.** All three graded quantities are recoverable from solver
output alone, and were recovered that way for this file: the pipe's Nusselt number from
`T` and `U` on the wedge, the cavity's from `T` and the mesh grading, the step's from the
`wallShearStress` field's `boundaryField`. What stops them being graded here the way
`felix/` and `moving_mesh/` are graded is that none of that output is in this repository,
because `studies/` is not committed. Trimmed copies in `tests/data/`, on the pattern
`tests/data/moving_mesh/PROVENANCE.md` sets out, are what would close the gap; until they
exist these numbers are quoted rather than re-derivable here.

**No grid study on any of the three.** Each is one mesh, so none has a claim about how much
of its error is discretisation, and on two of them the post-processing stencil is already
shown to matter as much as the mesh might.

**No turbulent case.** All three are laminar or buoyancy-driven at a Rayleigh number where
steady laminar is correct. Nothing here exercises a turbulence model.

**The step case is not a validation.** It ran at twice the benchmark's Reynolds number
and never converged, so it could not be one whatever the units turned out to be. It is a
worked example of a units failure, labelled that way above rather than counted as a pass.
Its reference value is settled, from reproductions rather than from the source.

**No turbulent case anywhere in tier 3.** All three are laminar or buoyancy-driven at a
Rayleigh number where steady laminar is correct, and nothing in tier 1 or tier 2 has a
turbulence model being graded either. The nearest thing is the DARPA SUBOFF hull in
[`../README.md`](../README.md), which is a case study rather than a case here, and whose
lesson is that a y+ in the buffer layer moved a form factor by 29% while the published
round-robin sees at most 4.5%. A turbulent case with a reference and a measured
agreement is the largest single gap in this index.
