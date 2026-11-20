# A reacting 0-D reactor: what is graded, and what is deliberately not

What `chemFoam` is held to, where the answer comes from, and which number in it is a
measurement.

The case and the measurement are both `openreynolds/toolbox/chem0d.py`:
`write` writes the case, `predict` computes the target, `grade` puts the run beside it.
The solver is held to it by `tests/test_journey_chem0d.py` in the `journey` job and the
arithmetic by `tests/test_toolbox_chem0d.py` in the suite.

## There is no grade.py in this directory, on purpose

[`benchmarks/README.md`](../README.md) argues it, about the three `tutorials/` cases:
the quantities they turn on are already measured by instruments in
`openreynolds/toolbox/`, and "a fourth implementation inside a grader would be a fourth
thing to keep right". The target temperature here is a root find over the case's own
JANAF coefficients. Written twice it would be two root finds to keep agreeing, so it is
written once, in `chem0d.py`, and the `--grade` path reads the same function the
`predict` path prints. `benchmarks/mhd/grade.py` next door is a separate file because
there the grader reads a finished OpenFOAM case and the template writes one -- two
different jobs.

## Why chemFoam and not a flame

`chemFoam` is OpenFOAM's single-cell chemistry solver. `createSingleCellMesh.H` builds
one hexahedron and `createFields.H` refuses a mesh with any other number of cells, so
there is no transport, no mesh quality, no boundary condition and no turbulence-chemistry
interaction in it at all: a closed reactor held at constant pressure or constant volume
while the chemistry integrates. The whole run costs a fraction of a second.

That is the point. A 3D flame has an answer that depends on the mesh, the inflow, the
combustion model and the mechanism together, and no part of it can be checked on its
own. A well-stirred reactor has an answer that depends only on thermodynamics, and
thermodynamics can be computed independently.

## The target, and why it needs no paper

`chemFoam` carries sensible enthalpy and adds the chemical release as a source. From
`applications/solvers/combustion/chemFoam/` in OpenFOAM v2512:

```
readInitialConditions.H   h0 = sum_i Y0_i * Hs_i(p, T0)
solveChemistry.H          integratedHeat += (Qdot/rho) * dt
hEqn.H                    h = h0 + integratedHeat                (constant pressure)
                          h = u0 + p/rho + integratedHeat        (constant volume)
```

`Qdot` in `chemistryModel` is `-sum_i Hf_i*RR_i`, and `Hs_i = Ha_i - Hc_i` with
`Hc_i = Ha_i(Tstd)` (`janafThermoI.H:207,212`). Adding those up, the `Hf`/`Hc` terms
cancel and what the solver conserves at constant pressure is the mixture's **absolute**
enthalpy:

```
sum_i Y_i(end) * Ha_i(T_end) = sum_i Y_i(0) * Ha_i(T0)                     (1)
```

One scalar equation in one unknown. Under `constantProperty volume` the same algebra
with `u = h - R_mix*T` gives

```
sum_i Y_i(end)*(Ha_i(T_end) - R_i*T_end) = sum_i Y_i(0)*(Ha_i(T0) - R_i*T0)  (2)
```

and `p_end = rho0*R_mix(end)*T_end` at the unchanged density, which is a second
independent number carrying the mixture's molecular weight as well as its temperature.

`Ha_i` is exactly what `janafThermoI.H:185-196` evaluates, with the dictionary's
coefficients scaled to a mass basis by `R_i = RR/W_i` (`janafThermo.C:74-75`,
`specieI.H:87-89`):

```
Ha_i(T) = (RR/W_i) * ( ((((a4/5*T + a3/4)*T + a2/3)*T + a1/2)*T + a0)*T + a5 )
```

with `RR = 1e3*NA*k` (`thermodynamicConstants.C`), `NA = 6.0221417930e+23`
(`fundamentalConstants.C`) and `k = 1.38065e-23` (`etc/controlDict`), so
`RR = 8314.47006650545 J/(kmol.K)` -- OpenFOAM's are the pre-2019 values and the product
is 1e-6 relative from the current SI 8314.462618, which matters to nothing here and
would matter if the wrong one were used somewhere it does. `perfectGas::H` is zero
(`perfectGasI.H:83-86`), so `Ha` has no pressure term.

(1) is solved by **bisection** over the narrowest interval every species' fit is valid
on, not by Newton: the JANAF fit is piecewise at `Tcommon` and its derivative jumps
there, scipy is not on the image and is forbidden (`tests/test_toolbox.py`), and a root
outside `[max Tlow, min Thigh]` is reported as a refusal rather than returned --
`janafThermo::limit` would clamp it and the answer would be a boundary value dressed as
a solution.

This is the `benchmarks/thermal_stress/` move: the reference is the solver's own
constitutive data re-evaluated independently, so there is nothing to misquote.

## Where the real-world data comes in: the thermo data, not the temperature

(1) is only as true as the coefficients in it. For a NASA/JANAF polynomial fit,
`Ha_i(298.15 K)` **is** the standard enthalpy of formation, and those are measured
quantities with published uncertainties. So that is what is compared with a
measurement.

`chem0d.py predict` prints each species' `Ha_i(298.15 K)` beside NIST's Chemistry
WebBook gas-phase value. The table is in `chem0d.NIST_FORMATION_KJ_MOL` and every entry
carries the WebBook's compound id, its method and its reference line, read off the
WebBook's own gas-phase thermochemistry pages
(`https://webbook.nist.gov/cgi/cbook.cgi?ID=<id>&Units=SI&Mask=1`, read 2026-11-21) and
not from a secondary source:

| species | NIST ΔfH°(gas), kJ/mol | reference |
|---|---|---|
| CH4 | **-74.6 ± 0.3** | C74828, Manion 2002, adopting Gurvich, Veyts et al. 1991. The page also lists -74.87 (Chase 1998, reviewed 1961), -74.85 ± 0.31 (Prosen and Rossini 1945) and -74.5 ± 0.4 (Pittam and Pilcher 1972) |
| CO2 | **-393.51 ± 0.13** | C124389, Cox, Wagman et al. 1984, CODATA review value. The page also lists -393.52 (Chase 1998) |
| H2O | **-241.826 ± 0.040** | C7732185, Cox, Wagman et al. 1984, CODATA review value. The page also lists -241.83 (Chase 1998) |
| O2, N2 | 0 | elements in their standard state, by definition |

The spread across NIST's own four methane determinations is 0.37 kJ/mol, which is larger
than any of the differences below, so **the reference's own uncertainty is larger than
the agreement** -- the second argument in [`benchmarks/README.md`](../README.md), and it
applies here as it does to Ghia and to de Vahl Davis.

Hess's law on those three gives the heat of reaction of `CH4 + 2O2 = CO2 + 2H2O` with
gaseous water as **-802.562 kJ/mol**, and dividing by methane's 16.0428 g/mol gives a
lower heating value of **50.026 MJ/kg** -- lower and not higher because the species list
has gaseous water in it and no liquid. The uncertainties add in quadrature to 0.33
kJ/mol, 0.04%, which is the tightest claim this table can support.

### The distribution's own data against that

Read from
`$FOAM_TUTORIALS/combustion/reactingFoam/laminar/counterFlowFlame2D/constant/thermo.compressibleGas`
and evaluated through the polynomial above:

| species | the image's JANAF fit at 298.15 K | NIST | difference |
|---|---|---|---|
| CH4 | -74.5992 | -74.6 | **+0.0008 kJ/mol, +0.001%** |
| CO2 | -393.5487 | -393.51 | **-0.0387 kJ/mol, -0.010%** |
| H2O | -241.8465 | -241.826 | **-0.0205 kJ/mol, -0.008%** |
| O2 | -0.0008 | 0 | -0.0008, rounding of seven-figure coefficients |
| N2 | +0.0014 | 0 | +0.0014, the same |
| heat of reaction | -802.641 kJ/mol | -802.562 | **-0.079 kJ/mol, -0.010%** |
| lower heating value of CH4 | 50.0312 MJ/kg | 50.0263 | **+0.0049 MJ/kg, +0.010%** |

So the thermo data the solver reads reproduces NIST's measured formation enthalpies to
one part in ten thousand, and **every difference is inside the reference's own stated
uncertainty** -- by a factor of 375 on methane, 3.4 on CO2 and 1.95 on water, which is
the smallest margin here and still a margin. The CH4 row agreeing to 0.001% while the
other two sit at 0.01% is not a sign that methane is better known: the fit was built
against the Gurvich value NIST now adopts, and against the Chase value on the same page
it reads +0.27 kJ/mol. NIST's own four methane determinations span 0.37 kJ/mol, which is
0.50% and is a hundred times the agreement being quoted. **Which of the reference's own numbers
is quoted is worth more than the solver's third digit**, and that is the whole of the
argument in `benchmarks/README.md` about de Vahl Davis's 4.509 against 4.519.

These numbers are derived rather than published and are checked two ways:
`tests/test_toolbox_chem0d.py` asserts the NIST table reproduces 50.03 MJ/kg by Hess's
law, and `tests/test_journey_chem0d.py` asserts the image's own data against the table,
in the image, byte-identical to the file the solver read.

## What is deliberately not graded

* **A flame temperature.** The shipped mechanism is one irreversible global step with
  five species. It cannot make CO, OH, H, O or NO, and all five are endothermic sinks a
  real stoichiometric hydrocarbon flame spends part of its heat release on. So (1) gives
  the **complete-combustion** temperature, the run reproduces it, and a real flame of
  the same mixture is **cooler**. The size of that gap is not asserted anywhere here,
  because no primary source for it has been read in this repository, and
  `benchmarks/sph/` makes exactly the same call about Martin & Moyce's dam-break
  non-dimensionalisation. NASA CEA (the Glenn equilibrium code, documented in NASA
  RP-1311) and the NIST-JANAF tables are the open-access places to get it with
  dissociation included; a number from either belongs in this file once somebody has
  read it there.

* **An ignition delay.** The obvious thing to measure in a 0-D reactor and the wrong
  thing to measure here: with a one-step global mechanism the delay is a property of `A`
  and `Ta`, not of combustion, and a published delay would be for a detailed mechanism
  this case does not contain. Grading it would be grading an Arrhenius fit against
  somebody's shock tube.

* **Rich mixtures.** `--phi` above 1 is refused by name. With no CO and no H2 in the
  species list, a one-step mechanism burns a rich mixture to leftover methane, which is
  not a thing that happens.

The run starts at **1000 K, not 300 K**, and that is the mechanism's property and not a
convenience: the rate is `A*exp(-Ta/T)*[CH4]*[O2]^2` with `Ta = 14906 K`, so at 300 K
the exponential is about 3e-22 and nothing happens in any finite run. A one-step
mechanism has no ignition chemistry and cannot ignite itself from cold. At 1000 K and
1 atm the fuel consumption time is of order 10 ms. The preheat adds its own sensible
enthalpy to the end temperature and changes nothing about (1).

## The two checks that make the agreement evidence

The end temperature is graded **twice**. Against the complete-combustion composition the
check is the whole case: thermodynamics plus the mechanism having actually run to
completion. Against the mass fractions **the solver itself ended on**, read out of the
last time directory, it is thermodynamics alone -- the same trick
`test_journey_thermal_stress.py` uses when it grades `sigma` against `-E*alpha*T` at the
solver's own T, so that an incomplete burn cannot be mistaken for an energy error or
hide one. Whether the burn completed is then its own check: a one-step irreversible
third-order reaction approaches complete consumption as a power law rather than
exponentially, so the tail is what decides whether the complete-combustion target
applies at all.

And the control: `tests/test_journey_chem0d.py` writes a second case with one word
changed, `chemistry off`, and nothing else. It has to come back at its initial
temperature and the grader has to fail it. Without that, "the final temperature matched
the prediction" is a claim about a number rather than about combustion --
`benchmarks/lbm/` makes the same argument about its one-relaxation-time contrast.

## The recorded run

Measured in the `journey` job of `.github/workflows/tests.yml`. The numbers are printed
by `tests/test_journey_chem0d.py` on every run of that job, and the agreement it
measured is recorded in the `combustion` rows of [`benchmarks/README.md`](../README.md).
