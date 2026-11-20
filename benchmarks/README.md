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
| " | amplitude A/D | **no published value for this configuration** | 0.640 loose, 0.567 tight | reported without a verdict, which is the honest thing to do with it |
| [laminar heated pipe](tutorials/) | fully developed Nusselt number | the Graetz problem, Nu = lambda_1^2/2 = 3.6568, **exact** | 3.677 as reported, 3.63 to 3.65 recomputed, **sub-percent either way** | the wall-gradient stencil, which moves the answer 0.7% across four defensible conventions and so is larger than the 0.55% being quoted |
| [natural convection, Ra 1e5](tutorials/) | mean Nusselt number on the hot wall | de Vahl Davis (1983), *Int. J. Numer. Methods Fluids* **3**, 249-264, Table V: **4.509** on the hot wall, 4.519 for the cavity average | 4.5335, **+0.54%** | a 40 x 40 mesh, a one-sided wall gradient, and 0.3% of reference uncertainty. Against the 4.519 that secondary sources misattribute to the hot wall it reads +0.32%, which is how a validation gets half a percent better without the solver changing |
| [backward-facing step](tutorials/) | reattachment length on the stepped wall | Gartling (1990), 6.1 **channel** heights = 12.2 step heights, at **Re 800 on the channel height** with the mean inlet velocity (the paper itself is unread: see below) | 6.42 step heights = 3.21 channel heights, **about 47% short**, and still drifting | the run is Gartling's geometry scaled by two at the same `nu`, so **Re 1600 on the channel height** against the benchmark's 800 on that same length, and its lengths are step heights where the benchmark's are channel heights. One mistake, two factors of two, and a printed "+5%" that was neither |
| [Poiseuille, force-driven](lbm/) | the velocity profile across a channel | `u = G/(2 mu) y (H - y)`, **exact**, and no paper is needed for it | max error **5.2e-11** of u_max at nu 0.02, **1.2e-11** at 0.1, **3.3e-12** at 0.4 | the convergence stop, and nothing else. With two relaxation times at `(tau_plus - 1/2)(tau_minus - 1/2) = 3/16` the bounce-back wall sits on the lattice link midpoint at any viscosity, so this scheme has no discretisation error on this flow -- the residual it stopped at was 5e-14. **The same code with one relaxation time, four times from its one exact viscosity, is wrong by 3.7e-3**, which is what makes the row evidence rather than a solver compared with itself |
| [lid-driven cavity, Re 1000](lbm/) | `u` and `v` on the centrelines | the same Ghia, Ghia & Shin (1982) Tables I and II as the Felix row above, **imported from `felix/grade.py` rather than retyped** | u RMS **0.736%**, v RMS **0.749%** of the lid speed | **the reference's own error is larger than the agreement**, exactly as on the Felix row and for the same reason: Ghia's extrema sit 1.46% to 2.20% from Botella & Peyret (1998). Then 128 cells against his 129x129. Graded against the same 34 numbers and the same 2% bound as Felix (0.448% / 1.063%), which is the only thing that makes the two comparable |
| [hydrostatic pressure](sph/) | pressure against depth in still water | `p = rho g (h - y)`, **exact** | RMS **2.50%** of rho g h, worst particle **8.61%** | the density noise, multiplied by `c0^2`. `dp/drho` is about `c0^2`, so a density error is 400 times itself as a fraction of the `rho g h` being measured -- it read **67%** before a density diffusion term was added, with operators that balance gravity to 3%. The worst particle is at the free surface, where the kernel support is truncated and there is no fluid above to sum |
| [sloshing, first mode](sph/) | the oscillation period | `omega^2 = g k tanh(k h)`, `k = pi/L`: linear theory, **exact** (Lamb, *Hydrodynamics*, 6th ed., art. 228) | **-4.75%**, 1.1257 s against 1.1818 s | **the sign is the diagnosis.** Short means the effective tank is shorter than the one asked for, which is the wall-offset error dynamic boundary particles have. Underneath it, 0.6% from the 5%-of-depth amplitude against a linear reference, and 20 particles across the depth |
| [dam break](sph/) | the surge front position | Martin & Moyce (1952), *Phil. Trans. R. Soc. A* **244**, 312-324 -- **the paper could not be read**, the publisher refuses the request | **reported, not graded**: Z = 0.975 at T = 0 rising to 3.943 at T = 3.00. The run is graded only on containment and on weak compressibility holding | which non-dimensionalisation the paper defines is unverified, and `t sqrt(2g/a)` and `t sqrt(g/a)` are both in circulation and differ by 1.414 at every point. Quoting it from a secondary source is the error the rest of this file argues hardest against, so it is not quoted |

`tutorials/` holds the three cheap cases with prompts you can run as they are.
`felix/` and `moving_mesh/` hold the four with graders, and both graders run on every
pull request against trimmed copies of the real runs in `tests/data/`.

`lbm/` and `sph/` are the two whose graders **solve** rather than reading a finished run,
and the difference is worth stating rather than hiding. The four above grade an OpenFOAM
or Felix case that cost a cluster an hour, so the run happens once and its output is
trimmed into `tests/data/`. Those two grade `openreynolds/toolbox/lbm.py` and `sph.py`,
which are numpy in this repository and finish inside the test job, so committing their
output would replace a check that re-derives the number with a number nobody can
re-derive. The whole of both costs about fifteen minutes on a runner, which is the
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
