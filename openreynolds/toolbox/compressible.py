#!/usr/bin/env python3
r"""The case files for a compressible run on a mesh that already exists: p in Pa, T in K.

`case_gen.py` writes an incompressible case and nothing else can write any other kind.
Its `SOLVERS` are `simpleFoam` and `pimpleFoam`, its `constant/transportProperties`
carries one kinematic `nu`, and its `0/p` is p/rho in m2/s2. Meanwhile the image has
`rhoSimpleFoam`, `rhoCentralFoam`, `sonicFoam` and -- since the workspace image baked it
in -- HiSA, the implicit density-based shock-capturing solver, with its own ONERA M6
example sitting at `/opt/hisa/examples/oneraM6/simulation`. Four shock-capturing solvers
and no way to generate a case for any of them. This is that way.

It is a sibling rather than a `--study compressible` inside `case_gen.py` because a
compressible case differs from an incompressible one in almost every file -- p's
dimensions, T existing at all, thermophysicalProperties instead of transportProperties,
`transonic yes`, the acoustic time step, the `rho` entry in `forceCoeffs` -- so the
branch would have run through a dozen writers that 500 lines of tests already pin for
the incompressible path. What it does share it imports: the mesh reading, the patch-role
resolution, the `boundaryField` walker that raises on an undescribed patch, the
turbulence fields, and the file furniture all come from `case_gen`.

why this exists, in one number

    The ONERA M6 replication (study 20260830-164816-eee0) converged beautifully on a
    1.79M-cell mesh that was never the problem and produced no shock at any of seven
    span stations: CL 22% low, CD 50% high. The cause was in `fvSchemes` -- both
    compressible pressure-flux terms first order and `grad(U)` fully limited, so the
    local numerical diffusivity u*h/2 = 0.63 m2/s against a physical nu of 1.48e-5.
    Round 3 on the same mesh with HiSA resolved the lambda structure and landed CD
    within 0.3% of a five-code band. Every scheme choice below is downstream of that.

what is derived, and from what

    Nothing about air is a bare constant here. The molar mass and the Sutherland
    coefficients are stated with their source and printed; Cp is computed from gamma
    and R; and then gamma, R, the speed of sound, the Mach number, the density, the
    viscosity and the Reynolds number are all read back from the literals this run
    actually wrote into `constant/thermophysicalProperties`, so the reported Mach
    number cannot drift from the file it describes.

what it will not do

    HiSA's `fvSchemes` and `fvSolution` are not written. Its dictionary set is its own
    -- `fluxScheme AUSMPlusUp`, `wVanLeer` reconstruction, dual-time `rPseudoDeltaT`,
    GMRES under LU-SGS -- and the only authority for the exact block nesting is the
    example case in the image, which this script cannot read from a laptop. It writes
    `0/`, `constant/` and `controlDict` for HiSA and prints the two `cp` commands and
    the entries that matter, which is a refusal that names what is missing rather than
    a guess that parses.

    python3 compressible.py /work/study/case --mach 0.84 --t 255.6 --p 80459
    python3 compressible.py /work/m6/case --mach 0.8395 --t 255.6 --p 80459 \
        --solver hisa --farfield farfield --body wing --aref 0.7532 --lref 0.64607
    python3 compressible.py /work/wedge/case --mach 2.0 --study transient --end-time 0.01
    python3 compressible.py /work/study/case --mach 0.3 --dry-run
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_gen
import hisa_env

# -- air, with a source for every number -------------------------------------------
#
# The brief these came from said "every number must be a stated, sourced value for air
# at the conditions, not a magic constant". A constant with a docstring is still a magic
# constant if nothing prints it, so each one is a row in SOURCES and the summary prints
# the table beside the case it wrote.

UNIVERSAL_GAS_CONSTANT = 8314.46261815324
"""J/(kmol K). N_A * k_B, both exact since the 2019 SI redefinition, times 1000."""

MOLAR_MASS_AIR = 28.9644
"""kg/kmol. Dry air's mean molar mass, US Standard Atmosphere 1976 (M0 = 28.9644)."""

GAMMA_AIR = 1.4
"""Cp/Cv for a diatomic calorically perfect gas, which air is to within 0.3% from 200
to 600 K. Above roughly 1000 K the vibrational modes come in and `hConst` stops being
the right thermo -- `janaf` is, and this script does not write it."""

SUTHERLAND_AS = 1.458e-6
"""kg/(m s sqrt(K)). mu = As*sqrt(T)/(1 + Ts/T). Air's standard pair, US Standard
Atmosphere 1976 (beta) and White, Viscous Fluid Flow 3e Table 1-2."""

SUTHERLAND_TS = 110.4
"""K. Sutherland's effective temperature for air, same two sources."""

ENTHALPY_OF_FORMATION = 0.0
"""J/kg. `hConst` asks for Hf and a single non-reacting specie has no use for it: only
differences of h enter the energy equation, so any constant offset cancels."""

STANDARD_PRESSURE = 101325.0
"""Pa. Sea-level static pressure, ICAO standard atmosphere / US Std Atm 1976."""

STANDARD_TEMPERATURE = 288.15
"""K. Sea-level static temperature, same."""


def foam_file(cls: str, obj: str, body: str, location: str = "") -> str:
    """`case_gen.foam_file`, with the banner saying which script actually wrote it.

    Small, and worth the line: `case_gen`'s banner has `Written: openreynolds
    case_gen.py` baked into it, and a compressible case whose every file says it came
    from the incompressible generator is a header that will be believed by somebody
    reading the case six weeks from now.
    """
    return case_gen.foam_file(cls, obj, body, location).replace(
        "openreynolds case_gen.py", "openreynolds compressible.py"
    )


def sources(thermo: dict) -> list[tuple[str, str, str, str]]:
    """(name, value, unit, where it came from), for the rows the summary prints.

    Split into what was written and what was read back, because the two are different
    kinds of claim: the first four are choices with citations, and everything after
    them is arithmetic on the literals in the file and cannot be argued with.
    """
    rows = [
        ("molWeight", f"{thermo['molar_mass']:g}", "kg/kmol",
         "US Standard Atmosphere 1976, dry air M0 = 28.9644"),
        ("Cp", thermo["cp_text"], "J/(kg K)",
         f"gamma*R/(gamma-1) with gamma = {GAMMA_AIR:g} and R below -- "
         "calorically perfect air"),
        ("Hf", f"{ENTHALPY_OF_FORMATION:g}", "J/kg",
         "one non-reacting specie: only differences of h enter the energy equation"),
        ("As", thermo["as_text"], "kg/(m s sqrt(K))", thermo["sutherland_from"]),
        ("Ts", f"{SUTHERLAND_TS:g}", "K",
         "Sutherland temperature for air, US Std Atm 1976 / White, Viscous Fluid Flow 3e"),
    ]
    rows += [
        ("R", f"{thermo['R']:.6g}", "J/(kg K)",
         f"{UNIVERSAL_GAS_CONSTANT:.9g} / molWeight; the numerator is N_A*k_B, exact in SI"),
        ("gamma", f"{thermo['gamma']:.6g}", "-",
         "Cp/(Cp - R), both read back from the file this run wrote"),
        ("a", f"{thermo['sound']:.6g}", "m/s", "sqrt(gamma*R*T) with that gamma and R"),
        ("rho", f"{thermo['rho']:.6g}", "kg/m3", "p/(R*T), perfectGas on the stated p and T"),
        ("mu", f"{thermo['mu']:.6g}", "Pa s", "As*sqrt(T)/(1 + Ts/T) at the stated T"),
        ("nu", f"{thermo['nu']:.6g}", "m2/s",
         "mu/rho -- a compressible case has no nu in any file, so Re is an output"),
    ]
    return rows


def air(pressure: float, temperature: float, viscosity: float | None = None) -> dict:
    """Everything the case needs about air, with the file's own literals as the source.

    The order matters and is the point. Cp is computed from gamma and R and then
    *formatted*; gamma is recovered from the formatted Cp. So the gamma that sets the
    speed of sound, the Mach number, the stagnation ratios and `waveTransmissive`'s
    `gamma` entry is the gamma the written file implies, to the digit, rather than a
    1.4 sitting in the reporting code where nothing can check it against the case.

    `viscosity` is the calibration knob: a reference mu measured at this T refits As
    rather than being written somewhere the Sutherland law would contradict it. The M6
    conditions wanted 1.6266e-5 at 255.6 K where the standard pair gives 1.6279e-5 --
    0.08%, well under anything a shock position resolves, and the knob is there because
    a published condition set is allowed to disagree with a correlation.
    """
    if pressure <= 0 or temperature <= 0:
        raise SystemExit(
            f"p = {pressure:g} Pa and T = {temperature:g} K: both are absolute and both "
            "must be positive. A compressible case's p is static pressure in pascals, "
            "not the gauge pressure an incompressible case sets to zero at the outlet"
        )

    gas_constant = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS_AIR
    cp_text = f"{GAMMA_AIR * gas_constant / (GAMMA_AIR - 1.0):.6g}"
    cp = float(cp_text)
    gamma = cp / (cp - gas_constant)

    if viscosity is not None:
        if viscosity <= 0:
            raise SystemExit("--mu is a dynamic viscosity in Pa s and must be positive")
        refitted = viscosity * (1.0 + SUTHERLAND_TS / temperature) / math.sqrt(temperature)
        as_text = f"{refitted:.6g}"
        standard = SUTHERLAND_AS * math.sqrt(temperature) / (1.0 + SUTHERLAND_TS / temperature)
        sutherland_from = (
            f"refitted so mu({temperature:g} K) = {viscosity:g} Pa s as --mu stated; "
            f"the standard 1.458e-06 gives {standard:.6g}, "
            f"{abs(standard / viscosity - 1.0) * 100:.2f}% away"
        )
    else:
        as_text = f"{SUTHERLAND_AS:g}"
        sutherland_from = (
            "Sutherland constant for air, US Std Atm 1976 (beta) / White, Viscous Fluid "
            "Flow 3e Table 1-2"
        )
    sutherland_as = float(as_text)
    mu = sutherland_as * math.sqrt(temperature) / (1.0 + SUTHERLAND_TS / temperature)
    rho = pressure / (gas_constant * temperature)

    return {
        "pressure": float(pressure),
        "temperature": float(temperature),
        "molar_mass": MOLAR_MASS_AIR,
        "cp": cp,
        "cp_text": cp_text,
        "as": sutherland_as,
        "as_text": as_text,
        "sutherland_from": sutherland_from,
        "R": gas_constant,
        "gamma": gamma,
        "sound": math.sqrt(gamma * gas_constant * temperature),
        "rho": rho,
        "mu": mu,
        "nu": mu / rho,
    }


def stagnation(thermo: dict, mach: float) -> dict:
    """Total conditions, isentropic, from the same gamma the file implies.

    Printed because they are what a published condition set is usually quoted in -- a
    wind tunnel states p0 and T0 and the model sees static -- and converting between
    the two with a gamma that is not the case's gamma is a quiet 1% in the Mach number.
    """
    ratio = 1.0 + 0.5 * (thermo["gamma"] - 1.0) * mach * mach
    exponent = thermo["gamma"] / (thermo["gamma"] - 1.0)
    total_t = thermo["temperature"] * ratio
    total_p = thermo["pressure"] * ratio ** exponent
    return {
        "T0": total_t,
        "p0": total_p,
        "rho0": total_p / (thermo["R"] * total_t),
        "dynamic": 0.5 * thermo["gamma"] * thermo["pressure"] * mach * mach,
    }


# -- which solver, and whether it is here ------------------------------------------

SOLVERS = {
    "rhoSimpleFoam": "steady, pressure-based (SIMPLE on a compressible pressure equation)",
    "rhoCentralFoam": "transient, explicit density-based (Kurganov central-upwind flux)",
    "hisa": "pseudo-transient, implicit density-based (AUSM+up, GMRES under LU-SGS)",
}
"""The three this writes a dictionary set for, and what each one is.

`rhoPimpleFoam` and `sonicFoam` are in the image and are deliberately absent: a
transient pressure-based transonic run is strictly worse than either density-based
solver at the one thing this script exists for, and a fourth half-tested dictionary set
is a fourth thing to be wrong. `--solver` reports what is on PATH either way.
"""

SHOCK_CAPTURING = ("hisa", "rhoCentralFoam")


def available(name: str) -> dict:
    """Is this solver on this machine, as a fact rather than an assumption.

    HiSA is asked of `hisa_env.probe`, which looks at the image's site directories and
    then at the volume and link-checks what it finds, because a HiSA binary that is
    present and whose `libhisa*.so` does not resolve reads like a solver crash from the
    middle of a job log. Everything else is a `$PATH` lookup.

    Absence is never a refusal here. The dictionaries are the deliverable and the
    machine that writes them does not have to be the machine that runs them -- writing
    a HiSA case on a laptop is an ordinary thing to do. What would be wrong is writing
    it and implying the solver is there, so this is reported in the summary every time.
    """
    if name == "hisa":
        found = hisa_env.probe(Path(hisa_env.DEFAULT_USER_DIR), Path(hisa_env.DEFAULT_SOURCE))
        return {"present": bool(found["available"]), "where": found["reason"]}
    found = shutil.which(name)
    return {
        "present": found is not None,
        "where": found or "not on $PATH here (the workspace image has it at $FOAM_APPBIN)",
    }


def choose_solver(asked: str, mach: float, study: str) -> tuple[str, str]:
    """(solver, the sentence saying why). `--solver` wins and is never second-guessed.

    `auto` prefers HiSA when it is actually here, for the reason the M6 post-mortem
    recorded: an explicit solver's timestep is set by the smallest cell in the domain,
    and on a layered mesh that is the prism layer rather than the shock cell -- 165 um
    against 4.7 mm, so about 92,000 explicit steps against 6,000 implicit pseudo-steps
    for the same answer. Where HiSA is absent the choice falls to the regime: a steady
    pressure-based run below M 1, and the explicit density-based one above it, where a
    pressure equation has no business being.
    """
    if asked != "auto":
        return asked, f"--solver {asked}"
    if available("hisa")["present"]:
        return "hisa", "auto: hisa is on this machine and is the implicit density-based one"
    if mach >= 1.0 or study == "transient":
        return ("rhoCentralFoam",
                f"auto: no hisa here, and M = {mach:.3f} with a {study} study wants a "
                "density-based solver rather than a pressure equation")
    return ("rhoSimpleFoam",
            f"auto: no hisa here and M = {mach:.3f} is subsonic, so the steady "
            "pressure-based solver is the cheap instrument")


# -- the regime, which is what picks the boundary conditions -----------------------


def regime(mach: float) -> tuple[str, str]:
    """(label, what the label means for the case). The label is for the report.

    The boundary conditions below branch on `mach >= 1` and on nothing else, because
    that is the only place the physics of the boundary actually changes: below it one
    characteristic runs upstream and the far field gets to set a pressure, above it
    none does. The four labels are for whoever reads the summary and wants to know
    whether they are in the band where this script is the right instrument at all.
    """
    if mach < 0.3:
        return ("low-speed",
                "M < 0.3, where density varies by under 5% and case_gen.py's "
                "incompressible case is the cheaper instrument for the same answer")
    if mach < 0.8:
        return ("subsonic compressible",
                "compressibility is in the answer and there is no shock to capture")
    if mach < 1.2:
        return ("transonic",
                "a supersonic pocket terminated by a shock sits on a subsonic "
                "freestream; this is the regime the pressure-based solver failed in")
    return ("supersonic", "the whole field is supersonic and the inflow is fully determined")


def outlet_choice(mach: float) -> tuple[str, str]:
    """(kind, why), for the outlet. Printed, because it changes the answer.

    A `fixedValue` p at a subsonic outlet is a perfectly reflecting wall for pressure
    waves: a shock that reaches it comes back into the domain as an expansion and the
    run either oscillates forever or converges onto a field with a reflection standing
    in it. `waveTransmissive` solves an advection equation for p on the patch with the
    far-field value and a relaxation length, so an outgoing wave leaves. Above M 1
    neither question arises -- every characteristic points out of the domain, so every
    field is `zeroGradient` and the outlet has nothing to say.
    """
    if mach >= 1.0:
        return ("zeroGradient",
                "supersonic outflow: all five characteristics leave the domain, so "
                "nothing may be imposed here -- a fixedValue p would be information "
                "travelling upstream through a supersonic flow")
    return ("waveTransmissive",
            "subsonic outflow: one characteristic runs back upstream, so a pressure has "
            "to be set -- waveTransmissive rather than fixedValue, which reflects an "
            "arriving shock back into the domain as an expansion")


def inlet_choice(mach: float) -> tuple[str, str]:
    if mach >= 1.0:
        return ("fixedValue on p, T and U",
                "supersonic inflow: every characteristic enters, so the whole upstream "
                "state is imposed and the domain has no say in the inlet pressure")
    return ("fixedValue on T and U, zeroGradient on p",
            "subsonic inflow: one characteristic runs upstream, so the inlet pressure "
            "is the domain's to settle and fixing it over-determines the boundary")


# -- the patches ------------------------------------------------------------------

FARFIELD_KIND = "farfield"
"""A role `case_gen.roles_for` has no flag for, and an external compressible case is
mostly made of it. A patch standing in for unbounded fluid is neither an inlet nor an
outlet -- flow crosses it both ways over the same run -- and `roles_for` would read the
name `farfield` as saying nothing and call it a wall, which is the safe reading for an
incompressible duct and a no-slip surface 16 chords from the wing here."""


def resolve_roles(mesh, opts) -> dict:
    """`case_gen.roles_for`, then the far-field patches on top of it.

    The role resolution is not reimplemented: what is named wins, then the mesh's own
    constraint types, then the name, and a name that says nothing is a wall -- all of
    that is `roles_for`'s policy and stays there. This only adds the one role it has no
    vocabulary for, and validates the names the same way it does, because a `--farfield`
    naming a patch that is not in the mesh has to be as loud as an `--inlet` that does.
    """
    roles = case_gen.roles_for(mesh, opts)
    stated = case_gen.named(opts, FARFIELD_KIND)
    unknown = [name for name in stated if name not in mesh.patch_faces]
    if unknown:
        raise SystemExit(
            f"the mesh has no patch called {', '.join(sorted(unknown))}; it has "
            + ", ".join(sorted(mesh.patch_faces))
        )
    for name in stated:
        roles[name] = {"kind": FARFIELD_KIND}
    return roles


def up_axis(mesh, opts) -> tuple[tuple[float, float, float], str]:
    """Which way is up, for the lift direction. Asked for, or the thin axis's complement.

    There is no measuring this. A wing at incidence and a wing with sideslip are the
    same mesh and the same freestream vector, and only a stated up axis tells them
    apart -- the M6 round 3 had `liftDir (-0.0533104 0 0.9985782)` because that mesh is
    z up, and taking the axis least aligned with the freestream instead would have
    picked y and reported sideslip as lift. For a plane case the thin axis is the one
    thing that cannot be up, which is a measurement and is used.
    """
    axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    stated = (opts.get("up") or "").strip().lower()
    if stated:
        if stated not in axes:
            raise SystemExit(f"--up takes x, y or z, not {stated!r}")
        return axes[stated], f"--up {stated}"
    if mesh.two_d and mesh.thin_axis == 2:
        return axes["y"], "the mesh is one cell thick in z, so up is y"
    return axes["z"], "z by default; --up says otherwise"


def wind_axes(stream, up) -> tuple[tuple, tuple, str]:
    """(dragDir, liftDir, the sentence). Wind axes from the freestream, not from alpha.

    dragDir is the freestream direction as measured off the inlet patch, and liftDir is
    the up axis with its streamwise part taken out and renormalised. That is the same
    rotation an `--alpha` would apply, derived from the flow the case actually has
    rather than from a second number that can disagree with it. Writing body axes
    instead -- `liftDir (0 1 0)`, which is what the incompressible generator does --
    puts cos(alpha) of the lift into drag and sin(alpha) of the drag into lift, which at
    3 degrees is a 5% error in CD on an aerofoil and looks like a mesh problem.
    """
    along = sum(up[i] * stream[i] for i in range(3))
    lift = tuple(up[i] - along * stream[i] for i in range(3))
    length = math.sqrt(sum(c * c for c in lift))
    if length < case_gen.TOLERANCE:
        raise SystemExit(
            "the freestream is along the up axis, so there is no lift direction to "
            "derive -- pass --up with an axis the flow is not parallel to"
        )
    lift = tuple(c / length for c in lift)
    angle = math.degrees(math.asin(max(-1.0, min(1.0, along))))
    return (stream, lift,
            f"wind axes off the measured freestream: incidence to the up axis is "
            f"{angle:.3f} deg")


# -- 0/ ---------------------------------------------------------------------------
#
# One `entry` per field, each a flat branch on the patch role. They are written out
# rather than generated from a table because the table would have to carry the same
# number of special cases and would read as if the cases were interchangeable, which is
# the thing a compressible boundary set is not. `case_gen.boundary_field` walks the
# mesh's own patch list and raises on a patch no role describes, so an undescribed
# patch stays a loud error here exactly as it is there.


def hisa_case(solver: str) -> bool:
    return solver == "hisa"


def field_U(plan, thermo: dict, mach: float, solver: str) -> str:
    stream = case_gen.stream_direction(plan)
    speed = mach * thermo["sound"]
    free = tuple(component * speed for component in stream)
    hisa = hisa_case(solver)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == FARFIELD_KIND:
            if hisa:
                return ["type            characteristicFarfieldVelocity;",
                        f"U               {case_gen.vector(free)};",
                        f"p               {thermo['pressure']:.6g};",
                        f"T               {thermo['temperature']:.6g};",
                        f"value           uniform {case_gen.vector(free)};"]
            return ["type            freestreamVelocity;",
                    f"freestreamValue uniform {case_gen.vector(free)};"]
        if kind == "inlet":
            direction = role.get("direction", (1.0, 0.0, 0.0))
            value = tuple(component * speed for component in direction)
            if hisa:
                return ["type            characteristicVelocityInletOutletVelocity;",
                        f"U               {case_gen.vector(value)};",
                        f"p               {thermo['pressure']:.6g};",
                        f"T               {thermo['temperature']:.6g};",
                        f"value           uniform {case_gen.vector(value)};"]
            return ["type            fixedValue;",
                    f"value           uniform {case_gen.vector(value)};"]
        if kind == "outlet":
            if mach >= 1.0:
                return ["type            zeroGradient;"]
            if hisa:
                return ["type            characteristicPressureInletOutletVelocity;",
                        f"U               {case_gen.vector(free)};",
                        f"p               {thermo['pressure']:.6g};",
                        f"T               {thermo['temperature']:.6g};",
                        f"value           uniform {case_gen.vector(free)};"]
            return ["type            inletOutlet;",
                    "inletValue      uniform (0 0 0);",
                    f"value           uniform {case_gen.vector(free)};"]
        # HiSA's own wall velocity: it corrects the boundary face value for the
        # density-based flux rather than setting it to zero and leaving the flux to
        # discover the wall. The ESI solvers want plain no-slip.
        if hisa:
            return ["type            boundaryCorrectedFixedValue;",
                    "value           uniform (0 0 0);"]
        return ["type            noSlip;"]

    body = ("dimensions      [0 1 -1 0 0 0 0];\n\n"
            f"internalField   uniform {case_gen.vector(free)};\n\n"
            + case_gen.boundary_field(plan, entry))
    return foam_file("volVectorField", "U", body, "0")


def field_p(plan, thermo: dict, mach: float, solver: str, length: float) -> str:
    """Static pressure in PASCALS, dimensions [1 -1 -2 0 0 0 0].

    This is the whole reason the file exists. `case_gen.field_p` writes
    `[0 2 -2 0 0 0 0]` and says so in a comment, because every incompressible OpenFOAM
    solver solves p/rho in m2/s2. A compressible solver solves p, and the two cases are
    one dimensions line apart in the text and a factor of rho apart in every force,
    every Cp and every pressure coefficient computed afterwards -- 1.2 in air, which
    hides inside any tolerance band wide enough to be useful. `preflight.py`'s
    `force_units_diagnosis` reads this exact line to decide which way round to judge
    the `forceCoeffs` density entry.
    """
    hisa = hisa_case(solver)
    kind, _ = outlet_choice(mach)
    free = tuple(c * mach * thermo["sound"] for c in case_gen.stream_direction(plan))

    def entry(name: str, role: dict) -> list[str]:
        role_kind = role["kind"]
        if role_kind == "empty":
            return ["type            empty;"]
        if role_kind == "symmetry":
            return ["type            symmetry;"]
        if role_kind == "slip":
            return ["type            slip;"]
        if role_kind == FARFIELD_KIND:
            if hisa:
                return ["type            characteristicFarfieldPressure;",
                        f"U               {case_gen.vector(free)};",
                        f"p               {thermo['pressure']:.6g};",
                        f"T               {thermo['temperature']:.6g};",
                        f"value           uniform {thermo['pressure']:.6g};"]
            return ["type            freestreamPressure;",
                    f"freestreamValue uniform {thermo['pressure']:.6g};"]
        if role_kind == "inlet":
            if mach >= 1.0:
                return ["type            fixedValue;",
                        f"value           uniform {thermo['pressure']:.6g};"]
            return ["type            zeroGradient;"]
        if role_kind == "outlet":
            if kind == "zeroGradient":
                return ["type            zeroGradient;"]
            if hisa:
                return ["type            characteristicPressureInletOutletPressure;",
                        f"p               {thermo['pressure']:.6g};",
                        f"T               {thermo['temperature']:.6g};",
                        f"value           uniform {thermo['pressure']:.6g};"]
            # lInf is the relaxation length the outgoing wave is damped over. The
            # domain's own streamwise extent is the honest reading: shorter and the
            # patch still reflects, much longer and it drags the far-field value in
            # too slowly for a steady run to settle.
            return ["type            waveTransmissive;",
                    "field           p;",
                    "phi             phi;",
                    "rho             rho;",
                    "psi             thermo:psi;",
                    f"gamma           {thermo['gamma']:.6g};",
                    f"fieldInf        {thermo['pressure']:.6g};",
                    f"lInf            {length:.6g};",
                    f"value           uniform {thermo['pressure']:.6g};"]
        return ["type            zeroGradient;"]

    body = (f"dimensions      [1 -1 -2 0 0 0 0];   // ABSOLUTE static pressure, Pa\n\n"
            f"internalField   uniform {thermo['pressure']:.6g};\n\n"
            + case_gen.boundary_field(plan, entry))
    return foam_file("volScalarField", "p", body, "0")


def field_T(plan, thermo: dict, mach: float, solver: str) -> str:
    """Static temperature in kelvin. A field the incompressible case does not have.

    Every wall is adiabatic (`zeroGradient`). That is a choice and not a default: on a
    transonic wing the recovery temperature at the wall is T0 to within the recovery
    factor, so an isothermal wall at the freestream T would pull 35 K out of the
    boundary layer at M 0.84 and thin it. An isothermal wall is a `fixedValue` here and
    this script does not offer the flag, because nothing has run one.
    """
    hisa = hisa_case(solver)
    temperature = f"{thermo['temperature']:.6g}"
    stream = case_gen.stream_direction(plan)
    free = tuple(c * mach * thermo["sound"] for c in stream)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == FARFIELD_KIND:
            if hisa:
                return ["type            characteristicFarfieldTemperature;",
                        f"U               {case_gen.vector(free)};",
                        f"p               {thermo['pressure']:.6g};",
                        f"T               {temperature};",
                        f"value           uniform {temperature};"]
            return ["type            freestream;",
                    f"freestreamValue uniform {temperature};"]
        if kind == "inlet":
            return ["type            fixedValue;", f"value           uniform {temperature};"]
        if kind == "outlet":
            if mach >= 1.0:
                return ["type            zeroGradient;"]
            if hisa:
                return ["type            characteristicPressureInletOutletTemperature;",
                        f"p               {thermo['pressure']:.6g};",
                        f"T               {temperature};",
                        f"value           uniform {temperature};"]
            return ["type            inletOutlet;",
                    f"inletValue      uniform {temperature};",
                    f"value           uniform {temperature};"]
        return ["type            zeroGradient;"]

    body = (f"dimensions      [0 0 0 1 0 0 0];   // static temperature, K\n\n"
            f"internalField   uniform {temperature};\n\n"
            + case_gen.boundary_field(plan, entry))
    return foam_file("volScalarField", "T", body, "0")


def field_alphat(plan) -> str:
    """Turbulent thermal diffusivity. The compressible solvers read it and refuse to
    start without it when a RAS model is on, and the incompressible case has no such
    field -- there is no energy equation for it to diffuse anything in."""

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind in ("inlet", "outlet", FARFIELD_KIND):
            return ["type            calculated;", "value           uniform 0;"]
        return ["type            compressible::alphatWallFunction;",
                "Prt             0.85;",
                "value           uniform 0;"]

    body = ("dimensions      [1 -1 -1 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
            + case_gen.boundary_field(plan, entry))
    return foam_file("volScalarField", "alphat", body, "0")


# -- constant/ --------------------------------------------------------------------


def thermophysical_properties(thermo: dict, solver: str) -> str:
    """`thermoType` plus one specie's numbers, in place of a kinematic nu.

    `sensibleInternalEnergy` for the density-based solvers and `sensibleEnthalpy` for
    the pressure-based one, which is not cosmetic: rhoCentralFoam transports rhoE and
    HiSA's own example is written in internal energy, while rhoSimpleFoam's energy
    equation is in h. Asking a solver for the energy variable it does not solve stops
    it at construction.

    `hePsiThermo` rather than `heRhoThermo`: psiThermo carries compressibility psi =
    1/(R T) and is what the compressible pressure equation and `waveTransmissive`'s
    `thermo:psi` entry both read. `perfectGas` is the equation of state that makes psi
    mean that.
    """
    energy = "sensibleEnthalpy" if solver == "rhoSimpleFoam" else "sensibleInternalEnergy"
    lines = [
        "thermoType", "{",
        "    type            hePsiThermo;",
        "    mixture         pureMixture;",
        "    transport       sutherland;",
        "    thermo          hConst;",
        "    equationOfState perfectGas;",
        "    specie          specie;",
        f"    energy          {energy};",
        "}", "",
        "mixture", "{",
        "    specie", "    {",
        f"        molWeight       {thermo['molar_mass']:g};   // kg/kmol, US Std Atm 1976",
        "    }",
        "    thermodynamics", "    {",
        f"        Cp              {thermo['cp_text']};   // J/(kg K) = gamma*R/(gamma-1), "
        f"gamma {GAMMA_AIR:g}",
        f"        Hf              {ENTHALPY_OF_FORMATION:g};",
        "    }",
        "    transport", "    {",
        f"        As              {thermo['as_text']};   // mu = As*sqrt(T)/(1 + Ts/T)",
        f"        Ts              {SUTHERLAND_TS:g};",
        "    }",
        "}", "",
        f"// R = {thermo['R']:.6g} J/(kg K), gamma = {thermo['gamma']:.6g} read back from the",
        f"// Cp and molWeight above; at p = {thermo['pressure']:.6g} Pa and "
        f"T = {thermo['temperature']:.6g} K that is",
        f"// rho = {thermo['rho']:.6g} kg/m3, a = {thermo['sound']:.6g} m/s, "
        f"mu = {thermo['mu']:.6g} Pa s, nu = {thermo['nu']:.6g} m2/s",
    ]
    return foam_file("dictionary", "thermophysicalProperties", "\n".join(lines),
                              "constant")


# -- system/ ----------------------------------------------------------------------


def fv_schemes(solver: str, model: str) -> str:
    """The schemes, and the one place the M6 replication was actually lost.

    Round 2 ran with `div(phid,p) Gauss upwind` and `grad(U) cellLimited Gauss linear 1`
    -- the compressible pressure flux first order, and the velocity gradient fully
    limited, which is first order at every extremum, and a shock is an extremum. At
    269 m/s through a 4.7 mm shock cell the local numerical diffusivity u*h/2 is
    0.63 m2/s against a physical nu of 1.48e-5. The shock was diffused into a ramp and
    the run converged cleanly onto it. So: `limitedLinear 1` on the convected terms,
    which is TVD and second order away from extrema, and an unlimited `Gauss linear`
    gradient. That bound is local, at the extrema where the limiter clips -- it is not
    an effective Reynolds number for the case, and quoting it as one overstates it.

    rhoCentralFoam has no pressure equation and no `div` terms to limit: its capturing
    is in the flux, `Kurganov` central-upwind with `vanLeer` reconstruction of rho, U
    and T onto the faces.
    """
    if solver == "rhoCentralFoam":
        lines = ["fluxScheme      Kurganov;", "",
                 "ddtSchemes", "{", "    default         Euler;", "}", "",
                 "gradSchemes", "{", "    default         Gauss linear;", "}", "",
                 "divSchemes", "{", "    default         none;",
                 "    div(tauMC)      Gauss linear;", "}", "",
                 "laplacianSchemes", "{", "    default         Gauss linear corrected;", "}", "",
                 "interpolationSchemes", "{", "    default         linear;",
                 "    reconstruct(rho) vanLeer;",
                 "    reconstruct(U)   vanLeerV;",
                 "    reconstruct(T)   vanLeer;", "}", "",
                 "snGradSchemes", "{", "    default         corrected;", "}"]
        return foam_file("dictionary", "fvSchemes", "\n".join(lines), "system")

    bounded = "bounded " if solver == "rhoSimpleFoam" else ""
    time_scheme = "steadyState" if solver == "rhoSimpleFoam" else "Euler"
    lines = ["ddtSchemes", "{", f"    default         {time_scheme};", "}", "",
             "gradSchemes", "{", "    default         Gauss linear;", "}", "",
             "divSchemes", "{", "    default         none;",
             f"    div(phi,U)      {bounded}Gauss limitedLinearV 1;",
             # The transonic pressure flux. Present whether or not `transonic yes` is
             # set: naming a term the solver does not form is harmless, and leaving out
             # one it does form stops the run on the first iteration.
             f"    div(phid,p)     {bounded}Gauss limitedLinear 1;",
             f"    div(phi,e)      {bounded}Gauss limitedLinear 1;",
             f"    div(phi,h)      {bounded}Gauss limitedLinear 1;",
             f"    div(phi,K)      {bounded}Gauss limitedLinear 1;",
             f"    div(phi,Ekp)    {bounded}Gauss limitedLinear 1;"]
    for field in case_gen.turbulence_fields(model):
        lines.append(f"    {f'div(phi,{field})':<16}{bounded}Gauss limitedLinear 1;")
    lines += ["    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;", "}", "",
              "laplacianSchemes", "{", "    default         Gauss linear corrected;", "}", "",
              "interpolationSchemes", "{", "    default         linear;", "}", "",
              "snGradSchemes", "{", "    default         corrected;", "}", "",
              "wallDist", "{", "    method          meshWave;", "}"]
    return foam_file("dictionary", "fvSchemes", "\n".join(lines), "system")


def fv_solution(solver: str, model: str, mach: float) -> str:
    """The linear solvers, and `transonic`.

    `transonic yes;` is what switches rhoSimpleFoam's pressure equation to the form
    that carries `div(phid,p)` -- the compressible flux of pressure. Left at `no` the
    solver forms the subsonic pressure equation and a transonic case runs to
    convergence on the wrong equation, which is the quietest of the failure modes in
    here: nothing in the log says so. It is set from the Mach number rather than
    offered as a flag, and the summary says which way it went.
    """
    fields = case_gen.turbulence_fields(model)
    if solver == "rhoCentralFoam":
        lines = ["solvers", "{"]
        lines += case_gen.solver_entry('"(rho|rhoU|rhoE)"', ["solver          diagonal;"])
        smooth = ["solver          smoothSolver;", "smoother        GaussSeidel;",
                  "nSweeps         2;", "tolerance       1e-09;", "relTol          0;"]
        lines += case_gen.solver_entry("U", smooth)
        lines += case_gen.solver_entry("e", smooth)
        lines += ["}"]
        return foam_file("dictionary", "fvSolution", "\n".join(lines), "system")

    energy = '"(e|h)"'
    group = '"(' + "|".join(fields) + ')"' if fields else ""
    names = ["U", energy] + ([group] if group else [])
    lines = ["solvers", "{"]
    lines += case_gen.solver_entry("p", ["solver          GAMG;", "tolerance       1e-08;",
                                         "relTol          0.01;",
                                         "smoother        GaussSeidel;"])
    for name in names:
        lines += case_gen.solver_entry(name, case_gen.SMOOTH_SOLVER)
    lines += case_gen.solver_entry("rho", ["solver          diagonal;"])
    lines += ["}", ""]

    transonic = "yes" if mach >= 0.7 else "no"
    lines += ["SIMPLE", "{", "    nNonOrthogonalCorrectors 0;",
              f"    transonic       {transonic};",
              "    residualControl", "    {",
              "        p               0.0001;",
              "        U               1e-05;",
              f"        {energy:<15} 1e-05;"]
    if group:
        lines.append(f"        {group:<15} 1e-05;")
    lines += ["    }", "}", "",
              # rho is relaxed hard and p softly: the compressible SIMPLE loop couples
              # them through the equation of state, and an under-relaxed density is what
              # keeps a transonic start from going negative on the first few iterations.
              "relaxationFactors", "{", "    fields", "    {",
              "        p               0.3;", "        rho             0.05;", "    }",
              "    equations", "    {", "        U               0.7;",
              f"        {energy:<15} 0.7;"]
    if group:
        lines.append(f"        {group:<15} 0.7;")
    lines += ["    }", "}"]
    return foam_file("dictionary", "fvSolution", "\n".join(lines), "system")


def force_objects(plan, thermo: dict, mach: float, opts) -> tuple[str, list[str]]:
    """(the `functions` block, the notes about it). `rho rho`, which is the whole point.

    The incompressible generator writes `rho rhoInf; rhoInf 1;` because its p is
    kinematic and the force object has to be handed a density it cannot read. A
    compressible case has a density field, so `rho rho;` tells the object to read it
    and the forces come out in newtons. Carrying the incompressible pair over scales
    every coefficient by 1/rhoInf -- a clean factor of 1.2 in air at sea level, or of
    rho itself wherever the altitude is not sea level. `preflight.py`'s
    `force_units_diagnosis` already judges the two cases by p's dimensions and warns
    about exactly this pairing, so a case written here and read there agree.

    Aref is the one number not derived. On a plane case it is the chord times the one
    cell's thickness and that is measured; on a 3D wing it is a planform area no script
    can see, and `case_gen`'s own docstring records the easy factor of ten that comes of
    defaulting it. So a 3D case with no `--aref` gets the `forces` object -- newtons,
    which need no reference area -- and a note saying why forceCoeffs was left out.
    """
    body = str(opts.get("body_patch") or "body")
    notes: list[str] = []
    if body not in plan.mesh.patch_face_counts():
        return "", notes

    speed = mach * thermo["sound"]
    stream = case_gen.stream_direction(plan)
    up, up_from = up_axis(plan.mesh, opts)
    drag, lift, axes_note = wind_axes(stream, up)
    notes.append(f"forces on '{body}': rho rho (the density field, {thermo['rho']:.6g} "
                 f"kg/m3 at these conditions) -- not rho rhoInf, which is the "
                 "kinematic-pressure form and would scale every coefficient")
    notes.append(f"           {axes_note}; up is {up_from}")

    lref = float(opts.get("lref") or plan.length)
    aref = opts.get("aref")
    if aref is None and plan.mesh.two_d:
        aref = plan.length * plan.mesh.thickness
        notes.append(f"           Aref {aref:.6g} m2 = {plan.length:g} m chord x "
                     f"{plan.mesh.thickness:g} m, the one cell's thickness (measured)")
    elif aref is not None:
        aref = float(aref)
        notes.append(f"           Aref {aref:.6g} m2 from --aref")

    shared = [
        "        libs            (forces);",
        f"        patches         ({body});",
        "        rho             rho;",
        f"        liftDir         {case_gen.vector(lift)};",
        f"        dragDir         {case_gen.vector(drag)};",
        "        CofR            (0 0 0);",
        f"        pitchAxis       {case_gen.vector(_pitch_axis(drag, lift))};",
        f"        magUInf         {speed:.6g};",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
    ]
    entries = ["functions", "{"]
    if aref is None:
        notes.append(
            "           forceCoeffs is NOT written: Aref is a planform area on a 3D "
            "case and nothing here can measure it, so `forces` is written instead "
            "(newtons, no reference area) -- pass --aref for the coefficients"
        )
        entries += ["    forces", "    {", "        type            forces;"] + shared + \
                   [f"        lRef            {lref:.6g};", "    }"]
    else:
        entries += ["    forceCoeffs", "    {", "        type            forceCoeffs;"] + \
                   shared + [f"        lRef            {lref:.6g};",
                             f"        Aref            {aref:.6g};", "    }"]
    entries += ["    residuals", "    {", "        type            solverInfo;",
                "        libs            (utilityFunctionObjects);",
                "        fields          (U p T);",
                "        writeResidualFields false;",
                "        writeControl    timeStep;", "        writeInterval   1;", "    }",
                "}"]
    return "\n" + "\n".join(entries), notes


def _pitch_axis(drag, lift) -> tuple[float, float, float]:
    """The remaining axis, taken positive, which is the sign convention Cm is quoted in.

    It is deliberately not a cross product. `lift x drag` and `drag x lift` differ by a
    sign and OpenFOAM's own tutorials use one of each -- `airFoil2D` has dragDir
    (1 0 0), liftDir (0 1 0), pitchAxis (0 0 1), which is `drag x lift`, while the 3D
    motorBike-style cases have liftDir (0 0 1) and pitchAxis (0 1 0), which is
    `lift x drag`. Both are really "the third coordinate axis, positive", because that
    is what makes a nose-up moment positive in each layout, and a computed cross
    product gets one of the two backwards -- which does not change |Cm| and does flip
    its sign, so every pitching moment reads the wrong way round.

    So: the global axis the wind-axis pair leans on least, pointed positive. For a
    plane case with the flow along x and lift along y that is z; for the M6 layout
    (z up, flow 3 degrees off x) it is y, which is the axis CMy is named after.
    """
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def leaning(axis) -> float:
        return abs(sum(axis[i] * drag[i] for i in range(3))) + \
            abs(sum(axis[i] * lift[i] for i in range(3)))

    return min(axes, key=leaning)


def control_dict(plan, thermo: dict, mach: float, opts, solver: str, study: str) -> tuple[str, list[str]]:
    """The run's clock, and the acoustic time step.

    This is the compressible difference that bites hardest. An incompressible transient
    step is Co*h/|U|; a compressible one is Co*h/(|U| + a), because the pressure waves
    travel at |U| + a and an explicit density-based solver is limited by them, not by
    the convection. At M 0.84 that is a factor of 2.2 and at M 0.1 it is a factor of
    11 -- an incompressible step handed to rhoCentralFoam diverges on the first few
    steps and reads like a bad mesh.

    HiSA counts pseudo-time iterations rather than seconds, so its clock is the steady
    one: deltaT 1 and endTime the iteration count, as its own example runs it.
    """
    notes: list[str] = []
    functions, force_notes = force_objects(plan, thermo, mach, opts)
    notes += force_notes
    speed = mach * thermo["sound"]

    if study == "transient" and solver != "hisa":
        end = float(opts["end_time"])
        cell = plan.mesh.smallest_cell
        wave = speed + thermo["sound"]
        fixed = opts.get("delta_t") is not None
        delta = float(opts["delta_t"]) if fixed else float(opts["courant"]) * cell / wave
        interval = end / max(1, int(opts["writes"]))
        which = "the average cell" if plan.mesh.cell_is_estimate else "the shortest cell"
        notes.append(
            f"time step  {delta:.4g} s = Co {float(opts['courant']):g} x {cell:.4g} m "
            f"({which}) / (U + a) = {wave:.4g} m/s"
        )
        notes.append(
            f"           the acoustic speed is in the denominator, not just U -- "
            f"{wave / speed:.2f}x smaller a step than the incompressible rule gives"
        )
        start, start_notes = case_gen.start_entries(opts.get("case"), end)
        notes += start_notes
        entries = ([f"application     {solver};"] + start
                   + ["stopAt          endTime;",
                   f"endTime         {end:g};", f"deltaT          {delta:.6g};",
                   f"writeControl    {'runTime' if fixed else 'adjustableRunTime'};",
                   f"writeInterval   {interval:.6g};",
                   f"purgeWrite      {case_gen.purge_write(study, opts)};",
                   "writeFormat     binary;", "writePrecision  6;",
                   "writeCompression off;", "timeFormat      general;",
                   "timePrecision   6;", "runTimeModifiable true;"])
        if fixed:
            entries += ["adjustTimeStep  no;",
                        f"// --delta-t {delta:.6g} s; acoustic Co = "
                        f"{delta * wave / cell:.3g} at U + a = {wave:.4g} m/s"]
        else:
            entries += ["adjustTimeStep  yes;",
                        f"maxCo           {float(opts['courant']):g};",
                        f"maxDeltaT       {delta * 5:.6g};"]
    else:
        end = float(opts["iterations"])
        start, start_notes = case_gen.start_entries(opts.get("case"), end)
        notes += start_notes
        entries = ([f"application     {solver};"] + start
                   + ["stopAt          endTime;",
                   f"endTime         {end:g};", "deltaT          1;",
                   "writeControl    timeStep;",
                   f"writeInterval   {max(1, int(end / max(1, int(opts['writes']))))};",
                   f"purgeWrite      {case_gen.purge_write('steady', opts)};",
                   "writeFormat     binary;", "writePrecision  6;",
                   "writeCompression off;", "timeFormat      general;",
                   "timePrecision   6;", "runTimeModifiable true;"])
        if solver == "hisa":
            entries.append("// hisa counts dual-time pseudo-iterations, so deltaT 1 and "
                           "endTime is the iteration count")
    body = "\n".join(entries) + functions + case_gen.io_optimisation()
    return foam_file("dictionary", "controlDict", body, "system"), notes


# -- the case ---------------------------------------------------------------------

HISA_SCHEME_ENTRIES = (
    "fluxScheme AUSMPlusUp with lowMachAusm false, wVanLeer reconstruction, bounded "
    "dualTime rPseudoDeltaT steadyState, faceLeastSquares gradients with cellLimited "
    "Gauss linear 0.1 on the turbulence, GMRES under LU-SGS, pseudoCoNum 0.1 rising to "
    "pseudoCoNumMax 20, localTimestepping true"
)
"""What round 3 of the M6 replication ran with, recorded from its own brief. The knob
names are known; the exact block nesting HiSA's dictionaries use is not knowable from
here, and a dictionary that does not parse is worse than no dictionary."""


def case_files(plan, thermo: dict, mach: float, opts, solver: str, study: str,
               model: str) -> tuple[dict[str, str], list[str]]:
    """Every file, path relative to the case, plus the notes the summary prints.

    Returned rather than written, the way `case_gen.case_files` is, so `--dry-run` and
    the writer cannot disagree about what a real run would produce.
    """
    length = max(plan.mesh.span) if plan.mesh.span else plan.length
    control, notes = control_dict(plan, thermo, mach, opts, solver, study)
    files = {
        "system/controlDict": control,
        "constant/thermophysicalProperties": thermophysical_properties(thermo, solver),
        "constant/turbulenceProperties": case_gen.turbulence_properties(model),
        "0/U": field_U(plan, thermo, mach, solver),
        "0/p": field_p(plan, thermo, mach, solver, length),
        "0/T": field_T(plan, thermo, mach, solver),
    }
    if solver == "hisa":
        notes.append(
            "hisa       system/fvSchemes and system/fvSolution are NOT written -- HiSA's "
            "dictionary set is its own and the only authority for it is the example "
            "case in the image:"
        )
        found = hisa_env.probe(Path(hisa_env.DEFAULT_USER_DIR), Path(hisa_env.DEFAULT_SOURCE))
        example = found["example"]
        notes.append(f"           cp {example}/system/fvSchemes "
                     f"{opts.get('case')}/system/")
        notes.append(f"           cp {example}/system/fvSolution "
                     f"{opts.get('case')}/system/")
        notes.append(f"           ({'present' if found['example_present'] else 'not on this machine'}"
                     "; hisa_env.py prints where it is on the instance)")
        notes.append(f"           what those files carry: {HISA_SCHEME_ENTRIES}")
    else:
        files["system/fvSchemes"] = fv_schemes(solver, model)
        files["system/fvSolution"] = fv_solution(solver, model, mach)

    fields = case_gen.turbulence_fields(model)
    if fields:
        flow = case_gen.Flow(mach * thermo["sound"], plan.length, thermo["nu"],
                             mach * thermo["sound"] * plan.length / thermo["nu"], "reynolds")
        intensity = case_gen.free_stream_intensity(plan, opts)
        mixing = 0.07 * plan.length
        ratio = case_gen.viscosity_ratio(plan, opts)
        writers = {
            "k": lambda: case_gen.field_k(plan, flow, intensity),
            "omega": lambda: case_gen.field_omega(plan, flow, intensity, mixing,
                                                  nu=thermo["nu"], ratio=ratio),
            "epsilon": lambda: case_gen.field_epsilon(plan, flow, intensity, mixing),
            "nuTilda": lambda: case_gen.field_nu_tilda(plan, flow),
        }
        for field in fields:
            files[f"0/{field}"] = writers[field]()
        files["0/nut"] = case_gen.field_nut(plan, model)
        files["0/alphat"] = field_alphat(plan)
    return files, notes


def turbulence_for(solver: str, opts, reynolds: float) -> tuple[str, str]:
    """The model, and why. rhoCentralFoam is laminar here and that is a scoping choice.

    rhoCentralFoam's dictionary set in this script is the inviscid/laminar
    shock-capturing one -- the shock tube, the wedge, the forward step -- which is what
    an explicit central-upwind solver is the cheap instrument for. A turbulent transonic
    wing on a layered mesh is the case where its explicit step is set by the 165 um
    prism layer rather than the 4.7 mm shock cell, and that is HiSA's job or
    rhoSimpleFoam's, not this one's.
    """
    asked = (opts.get("turbulence") or "auto").strip()
    if solver == "rhoCentralFoam":
        if asked not in ("auto", "laminar"):
            raise SystemExit(
                f"--turbulence {asked} with rhoCentralFoam: this script writes "
                "rhoCentralFoam as the laminar shock-capturing case only (its explicit "
                "step is set by the smallest cell, which on a layered mesh is the prism "
                "layer, not the shock cell). Use --solver hisa or rhoSimpleFoam for a "
                "turbulent compressible run"
            )
        return "laminar", "rhoCentralFoam is written here as the laminar case (see --help)"
    if asked != "auto":
        return asked, f"--turbulence {asked}"
    if reynolds < case_gen.LAMINAR_BELOW:
        return "laminar", f"Re = {reynolds:.0f} is below {case_gen.LAMINAR_BELOW:g}"
    # Spalart-Allmaras rather than kOmegaSST: it is what the M6 round 3 ran, it is one
    # transported field instead of two, and it needs no k at the wall -- which matters
    # on an external compressible case where the free-stream k and omega are the pair
    # that silently sets a free-stream eddy viscosity thousands of times the molecular.
    return "SpalartAllmaras", (
        f"Re = {reynolds:.0f}; one transported field and no free-stream k to get wrong, "
        "which is what the M6 run used"
    )


def summary(case: Path, plan, thermo: dict, mach: float, solver: str, why: str,
            study: str, model: str, model_why: str, notes: list[str]) -> list[str]:
    """What was decided, in the order somebody checking it would ask.

    The constants table is at the bottom and is printed every time, not only on request.
    The asymmetry the incompressible generator's own notes record applies here twice
    over: a bad thermophysical number is not measurable from anything the product
    reports afterwards, so a study inherits it and explains the answer it produces.
    """
    speed = mach * thermo["sound"]
    label, what = regime(mach)
    inlet_kind, inlet_why = inlet_choice(mach)
    out_kind, out_why = outlet_choice(mach)
    total = stagnation(thermo, mach)
    counts = plan.mesh.patch_face_counts()
    span = plan.mesh.span
    reynolds = speed * plan.length / thermo["nu"]

    lines = [
        f"case       {case}",
        f"solver     {solver} -- {SOLVERS[solver]}",
        f"           {why}",
        "installed  " + "; ".join(
            f"{name}: {'here' if state['present'] else 'NOT here'}"
            for name, state in ((n, available(n)) for n in SOLVERS)
        ),
        f"regime     {label}: M = {mach:.4f} from U = {speed:.6g} m/s and "
        f"a = {thermo['sound']:.6g} m/s",
        f"           {what}",
        f"state      p = {thermo['pressure']:.6g} Pa (absolute), "
        f"T = {thermo['temperature']:.6g} K",
        f"           rho = {thermo['rho']:.6g} kg/m3, mu = {thermo['mu']:.6g} Pa s, "
        f"nu = {thermo['nu']:.6g} m2/s",
        f"stagnation T0 = {total['T0']:.6g} K, p0 = {total['p0']:.6g} Pa, "
        f"rho0 = {total['rho0']:.6g} kg/m3, q = {total['dynamic']:.6g} Pa",
        f"flow       U = {speed:.6g} m/s, L = {plan.length:.6g} m, "
        f"nu = {thermo['nu']:.6g} m2/s, Re = {reynolds:.0f}   (Re = U*L/nu, nu = mu/rho)",
        f"turbulence {model}   ({model_why})",
        f"inlet      {inlet_kind}",
        f"           {inlet_why}",
        f"outlet     {out_kind}",
        f"           {out_why}",
        f"mesh       {plan.mesh.cells:,} cells, "
        f"{span[0]:.4g} x {span[1]:.4g} x {span[2]:.4g} m"
        + ("   (2D, one cell thick)" if plan.mesh.two_d else "")
        + (f"   {plan.mesh.checkmesh}" if plan.mesh.checkmesh else ""),
        "patches    " + ", ".join(
            f"{patch} ({counts[patch]}, {plan.roles[patch]['kind']})"
            for patch in case_gen.patch_order(counts)),
    ]
    lines += notes
    lines += [f"           {note}" for note in plan.notes]
    lines.append("")
    lines.append("constants  every number this run wrote, and where it came from")
    for name, value, unit, source in sources(thermo):
        lines.append(f"  {name:<10} {value:<12} {unit:<22} {source}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("case", nargs="?", type=Path,
                    help="A case directory that already holds constant/polyMesh.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Say what would be written; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Write over 0/ and the dictionaries if they are there.")

    state = ap.add_argument_group("the freestream")
    state.add_argument("--mach", type=float, default=None,
                       help="Freestream Mach number. Either this or --speed.")
    state.add_argument("--speed", type=float, default=None,
                       help="Freestream speed in m/s; the Mach number follows from the "
                            "speed of sound at --t.")
    state.add_argument("--p", type=float, default=STANDARD_PRESSURE, dest="pressure",
                       help=f"Static pressure, Pa ABSOLUTE (default {STANDARD_PRESSURE:g}, "
                            "sea level).")
    state.add_argument("--t", type=float, default=STANDARD_TEMPERATURE, dest="temperature",
                       help=f"Static temperature, K (default {STANDARD_TEMPERATURE:g}).")
    state.add_argument("--mu", type=float, default=None, dest="mu",
                       help="Dynamic viscosity at --t, Pa s. Refits the Sutherland As so "
                            "the law reproduces it, rather than contradicting it.")
    state.add_argument("--length", type=float, default=None,
                       help="Characteristic length for the reported Re. Default: measured "
                            "off the inlet patch.")
    state.add_argument("--direction", default="",
                       help="Which way the inlet flows: x, y, z, -x, -y, -z. Default: the "
                            "inlet patch's own normal, pointed into the domain.")
    state.add_argument("--up", default="",
                       help="Which axis lift is measured towards: x, y or z. Default z, or "
                            "y on a mesh one cell thick in z. There is no measuring this.")
    state.add_argument("--turbulence", default="auto",
                       choices=["auto", "laminar", "kOmegaSST", "kEpsilon",
                                "SpalartAllmaras"],
                       help="auto: laminar below Re 2300, SpalartAllmaras above.")
    state.add_argument("--turbulent-intensity", type=float, default=None,
                       dest="turbulent_intensity",
                       help="Free-stream turbulence intensity. Default 0.001 for a body in "
                            "open flow, 0.05 inside a passage.")
    state.add_argument("--viscosity-ratio", type=float, default=None, dest="viscosity_ratio",
                       help="Target free-stream nu_t/nu; omega follows from it.")
    state.add_argument("--external", action="store_true",
                       help="Treat this as a body in open flow even when the mesh does not "
                            "say so.")

    run = ap.add_argument_group("the run")
    run.add_argument("--solver", default="auto", choices=["auto", *SOLVERS],
                     help="auto prefers hisa where it is actually installed, then picks by "
                          "regime. A named solver is written whether or not it is on this "
                          "machine, and the summary says which.")
    run.add_argument("--study", default="steady", choices=["steady", "transient"])
    run.add_argument("--iterations", type=int, default=2000,
                     help="Steady or pseudo-time: how many (default 2000).")
    run.add_argument("--end-time", type=float, default=1.0, dest="end_time",
                     help="Transient: seconds.")
    run.add_argument("--delta-t", type=float, default=None, dest="delta_t",
                     help="Transient: a fixed step, which turns adjustTimeStep off.")
    run.add_argument("--courant", type=float, default=0.5,
                     help="Transient: target max ACOUSTIC Courant number (default 0.5); "
                          "the step is Co*h/(|U| + a), not Co*h/|U|.")
    run.add_argument("--writes", type=int, default=40, help="How many times to write.")
    run.add_argument("--purge", type=int, default=None, dest="purge",
                     help="Keep only the last N time directories (0 keeps all).")

    patches = ap.add_argument_group("what each patch is for (names, comma-separated)")
    patches.add_argument("--inlet", action="append", default=[])
    patches.add_argument("--outlet", action="append", default=[])
    patches.add_argument("--farfield", action="append", default=[],
                         help="Patches standing in for unbounded fluid. Flow crosses them "
                              "both ways, so they are neither inlet nor outlet -- and "
                              "nothing guesses the role from the name.")
    patches.add_argument("--wall", action="append", default=[], help="No-slip walls.")
    patches.add_argument("--slip", action="append", default=[], help="Free-slip walls.")
    patches.add_argument("--symmetry", action="append", default=[], help="Symmetry planes.")
    patches.add_argument("--body", default="body", dest="body_patch",
                         help="The patch forces are taken on (default 'body').")
    patches.add_argument("--aref", type=float, default=None,
                         help="Reference area for forceCoeffs, m2. Measured on a plane "
                              "case; on a 3D case nothing can see a planform area, so "
                              "without this only `forces` (newtons) is written.")
    patches.add_argument("--lref", type=float, default=None,
                         help="Reference length for forceCoeffs, m. Default: the "
                              "characteristic length.")

    args = ap.parse_args(argv)
    if args.case is None:
        ap.error("a case directory that already holds constant/polyMesh")
    if args.mach is None and args.speed is None:
        ap.error("--mach or --speed: a compressible case is defined by one of them, and "
                 "the Mach number is what picks every boundary condition below")
    if args.mach is not None and args.speed is not None:
        ap.error("give --mach or --speed, not both: the other follows from the speed of "
                 "sound at --t")

    opts = vars(args)
    thermo = air(args.pressure, args.temperature, args.mu)
    mach = float(args.mach) if args.mach is not None else float(args.speed) / thermo["sound"]
    if mach <= 0:
        raise SystemExit("--mach and --speed must be positive")

    mesh = case_gen.read_mesh(Path(args.case))
    roles = resolve_roles(mesh, opts)
    length, length_from = case_gen.characteristic_length(mesh, roles, opts)
    bodies = mesh.bodies()
    plan = case_gen.Plan(
        mesh, roles, length,
        {"external": bool(args.external) or bool(bodies) or bool(case_gen.named(opts, "farfield")),
         "bodies": bodies, "length_from": length_from},
        [f"length {length:.4g} m from {length_from}"]
        + (["body in the flow: " + ", ".join(bodies)] if bodies else []),
    )

    solver, why = choose_solver(args.solver, mach, args.study)
    reynolds = mach * thermo["sound"] * length / thermo["nu"]
    model, model_why = turbulence_for(solver, opts, reynolds)
    files, notes = case_files(plan, thermo, mach, opts, solver, args.study, model)

    for line in summary(Path(args.case), plan, thermo, mach, solver, why, args.study,
                        model, model_why, notes):
        print(line)
    print()

    if args.dry_run:
        print(f"would write {len(files)} files into {args.case}:")
        for relative in sorted(files):
            print(f"  {relative}   ({len(files[relative].splitlines())} lines)")
        return 0

    target = Path(args.case)
    if (target / "0" / "U").exists() and not args.force:
        raise SystemExit(f"{target} already has a 0/ directory; --force to write over it")
    written = case_gen.write_case(target, files)
    print(f"wrote {len(written)} files into {target}")
    print(f"next: {solver} -case {target}   (the mesh is already there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
