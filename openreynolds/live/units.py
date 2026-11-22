"""Physical units in, lattice units out, and the three checks that stop a bad run.

Nothing in this repository converted SI to lattice units before this file. It did not
have to: the OpenFOAM path is dimensional end to end, Felix's `case.yaml` is
dimensional, and `openreynolds/toolbox/lbm.py` is written entirely in lattice units and
leaves the conversion to whoever chose `--re`. A lattice solver driven from a robot
simulator cannot leave it to anybody -- the robot's timestep arrives in seconds and its
geometry in metres, and there is no Reynolds number anywhere in the request.

A lattice Boltzmann scheme has exactly three free numbers and one of them is not free:

    dx      metres per lattice cell          chosen (it is the resolution)
    dt      seconds per lattice step         chosen
    nu_l    viscosity in lattice units       NOT chosen: nu_l = nu * dt / dx^2

and the whole art of setting up an LBM run is that the third one lands somewhere the
scheme can actually work, because `tau = nu_l/c_s^2 + 1/2` and both ends of tau's range
are failures: at tau -> 1/2 the collision stops relaxing and the run is stiff, and at
large tau the viscosity is so large that the lattice Mach number needed to reach a given
Reynolds number pushes the scheme's own `O(Ma^2)` compressibility error into the answer.

So `Scaling` is the conversion and the three refusals together, and it refuses at
construction rather than advising at run time. The three are the ones the brief for this
work named, with the fourth that falls out of the same arithmetic:

    Ma = u_l / c_s < 0.1        the scheme's error is O(Ma^2)
    tau_plus > STIFF_TAU        `toolbox/lbm.py`'s own floor, and its own wording
    u_l < U_LATTICE_MAX         where the second-order equilibrium's truncation bites
    dx > 0, dt > 0              a cell and a step, not a point and an instant

Each refusal names the knob that moves it, which is the convention
`benchmarks/*/grade.py` follows for the same reason: a message that says what is wrong
and not what to do about it costs a round of guessing.

## The constants are `toolbox/lbm.py`'s, held by a test rather than by an import

`CS2`, `LAMBDA` and `STIFF_TAU` below are the same numbers, for the same reasons, as the
ones in `openreynolds/toolbox/lbm.py`. They are copied and not imported, because
`tests/test_toolbox.py` asserts that every toolbox script imports only the standard
library, a closed set of third-party modules and its siblings -- the toolbox is copied
verbatim into a workspace image that has no Warp in it, so a toolbox script that
imported this package would be a script nobody could run, and this package importing a
toolbox script would make the dependency run the wrong way.

`tests/test_live_units.py` parses the toolbox script and asserts the values agree, which
is the device `tests/test_benchmarks_index.py` already uses to stop two indexes drifting
apart. Copied constants with no check between them is how two files come to hold two
different opinions about the speed of sound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CS2", "CS", "LAMBDA", "STIFF_TAU", "MACH_MAX", "U_LATTICE_MAX", "TAU_WINDOW",
    "Fluid", "FLUIDS", "fluid", "Scaling", "UnphysicalScaling", "rates",
    "LiveUnavailable", "require",
    "WARP_TESTED", "NEWTON_TESTED", "version_note",
]


class LiveUnavailable(ImportError):
    """The `[live]` extra is not installed, with the one command that fixes it.

    Raised from `require()` rather than from an import, so that
    `from openreynolds.live import FluidWorld` succeeds on a machine without Warp and
    fails with this message when something tries to build a world. That is what keeps
    `openreynolds --help` working everywhere (`tests/test_live_optional.py`).

    It lives in this module, the only one in the package that imports nothing but the
    standard library, for a reason that is not organisational. It was in `world.py`
    first, beside the rest of the public API -- and `world.py` imports `lattice.py`,
    which imports Warp when it is read. So on exactly the machine the message is
    written for, `from .world import require` raised a bare
    `ModuleNotFoundError: No module named 'warp'` from `lattice.py`'s own import line,
    and the sentence naming the install command never got to run. A refusal reachable
    only when it is not needed is not a refusal.
    """


def require() -> Any:
    """Warp, or `LiveUnavailable` naming the install command.

    Every entry point that is about to touch the device calls this first:
    `FluidWorld.__init__`, `openreynolds live`'s subcommands through
    `cli._live_cases`, and `live_tool.handler`. The point is that the failure is one
    sentence about an optional extra rather than a traceback through a lattice kernel.
    """
    try:
        import warp as wp
    except Exception as exc:  # pragma: no cover - exercised on a machine without it
        raise LiveUnavailable(
            "the live backend needs NVIDIA Warp, which is not installed. It is an "
            "optional extra, so that core OpenReynolds does not pull a CUDA toolchain "
            "in:\n    pip install 'openreynolds[live]'\n"
            f"(the import said: {exc})"
        ) from exc
    return wp


WARP_TESTED = ("1.17", "1.20")
NEWTON_TESTED = ("1.6", "1.9")
"""The dependency band this extension was measured across: `[lower, upper)` by minor.

Here and not only in `pyproject.toml` because `pyproject.toml` cannot be enforced where
it matters. The case this exists for is a user running the extension **inside Isaac
Lab**: Isaac Lab 3.0.0's own metadata says `warp-lang==1.17.0` and `newton[sim]==1.6.1`,
so whatever `[live]` asks for, pip has already resolved those and the extension has to
work on them or say why not. `docs/live-version-matrix.md` is the measurement; the short
version is that it works, and that nothing in this file, `lattice.py`, `world.py` or
`couple.py` needed a shim to make it:

* every Warp symbol the kernels use behaves identically on 1.17.0 and 1.19.0, including
  `wp.where`'s argument order, `wp.set_module_options`' two keys, the 3x19 `wp.constant`
  matrix, `wp.mesh_query_point_sign_normal`'s result fields and the "mutating a constant
  inside a dynamic loop" rule that `lattice.stream` is written around;
* the Re 100 lid-driven cavity comes out **bit-identical** on the two (same SHA-1 over
  the saved velocity plane), as does a three-step `NewtonCoupling` run;
* every `newton.Model` / `newton.State` field `couple.py` reads exists on 1.6.1 with the
  same dtype, `newton.GeoType`'s member names and values are unchanged, and
  `shape_scale`'s packing -- the one `couple.py::_SCALE` records because Newton's own
  comments get it wrong -- is the same on both.

So the band is advisory rather than a refusal: there is nothing to refuse. It is carried
in code anyway for two reasons. `tests/test_live_versions.py` asserts the installed
versions are inside it and that `pyproject.toml`'s `live` extra agrees with it, so the
declared range and the tested range cannot drift. And `version_note()` puts one line in
`FluidWorld.report()` when a run is outside it, so a number measured on an untested
Warp is labelled as such in the report that carries it, which is the habit
`benchmarks/README.md` is written around.

The upper bounds are exclusive minors rather than open ends because an open end is a
claim about software that does not exist yet. 1.18 was not installed; it sits between
two releases that agree bit for bit, which is weaker evidence than a run and stronger
than nothing, and the matrix says exactly that.
"""


def version_note() -> str | None:
    """One line naming a dependency outside `WARP_TESTED` / `NEWTON_TESTED`, or None.

    Reads the versions through `importlib.metadata` rather than through `warp.config`
    and `newton.__version__`, so this module keeps importing nothing but the standard
    library -- the property `openreynolds/live/__init__.py`'s lazy `__getattr__` and
    `require()` both depend on, and the one `tests/test_live_optional.py` checks by
    hiding Warp from the import system.

    A missing distribution is not reported: `newton` is only needed by `couple.py`, and
    a pure-fluid run on a machine with no robot simulator is a supported case.
    """
    from importlib.metadata import PackageNotFoundError, version

    outside = []
    for dist, (low, high) in (("warp-lang", WARP_TESTED), ("newton", NEWTON_TESTED)):
        try:
            found = version(dist)
        except PackageNotFoundError:
            continue
        parts = tuple(int(p) for p in found.split(".")[:2] if p.isdigit())
        if len(parts) < 2:
            continue
        if not (_minor(low) <= parts < _minor(high)):
            outside.append(f"{dist} {found} (tested >={low},<{high})")
    if not outside:
        return None
    return (
        "  NOTE:       outside the tested dependency range: " + ", ".join(outside)
        + ". The extension may well work; nothing here has measured it. "
        + "docs/live-version-matrix.md is what was measured"
    )


def _minor(spec: str) -> tuple[int, int]:
    major, _, minor = spec.partition(".")
    return int(major), int(minor or 0)


CS2 = 1.0 / 3.0
"""The lattice speed of sound squared, for every velocity set here (D3Q19, D3Q27, D2Q9).

The same value as `toolbox/lbm.py`'s `CS2`. It is a property of the quadrature the
velocity set is, not a choice.
"""

CS = math.sqrt(CS2)

LAMBDA = 3.0 / 16.0
"""The two-relaxation-time "magic parameter": (tau_plus - 1/2)(tau_minus - 1/2).

The same value, and the same reason, as `toolbox/lbm.py`'s `LAMBDA`: it puts the no-slip
wall of a halfway bounce-back exactly on the lattice link midpoint for a straight wall,
at any viscosity. `benchmarks/lbm/` grades that at three viscosities a factor of twenty
apart, and grades single-relaxation-time beside it to show the property is the scheme's
and not the solver agreeing with itself.
"""

STIFF_TAU = 0.52
"""Below this tau_plus a two-relaxation-time or single-relaxation-time run is refused.

`toolbox/lbm.py` uses the same number to *report* rather than to refuse, and the
difference is deliberate: that script is handed a viscosity in lattice units by someone
who has already done this arithmetic, so reporting is all it can usefully do. Here the
caller handed over metres and seconds and a fluid, so a stiff tau is this file's fault
to catch, and it is catchable before a single step runs.

The number is where the trouble starts and not a cliff. `toolbox/lbm.py` records the
measurement behind it: a graded cavity at tau_plus 0.538 converges and a 16-cell cavity
at the same Reynolds number sits at 0.5048 and does not converge at all.

**It is not applied to a large-eddy run, and that is the whole reason the LES is here.**
A robot in air is the case: at dx = 20 mm and dt_robot = 2 ms, air's molecular viscosity
gives nu*dt/dx^2 = 7.6e-05 and tau_plus = 0.50002, and no knob in this file fixes it.
dt cannot rise -- the largest step is the robot's own, since the fluid has to arrive at
the same instant -- and a finer dx makes it worse, not better. The resolved-viscosity
run simply does not exist at any resolution a real-time loop can afford, which is what
it means to say a flow is under-resolved. What carries such a run is the subgrid
viscosity, and the collision that survives it is the regularised one. So a stiff tau
with `les=True` is reported as an under-resolved large-eddy run -- stated in the report,
every time, in those words -- and a stiff tau without it is refused.
"""

MACH_MAX = 0.1
"""The lattice Mach number a run is allowed to reach, `u_l / c_s`.

The scheme solves the compressible equations and is used for the incompressible ones;
the error in that substitution is `O(Ma^2)`, so 0.1 is a 1% ceiling on it. Every
published value this extension is graded against -- Ghia's cavity, Schafer and Turek's
cylinder, the sphere drag correlations -- is an incompressible answer, and a run above
this ceiling is not solving the equations it is being compared with.
"""

U_LATTICE_MAX = 0.4
"""The largest lattice velocity allowed anywhere, independent of the Mach check.

`Ma < 0.1` is `u_l < 0.0577`, which is the stricter of the two for a uniform flow, so
this bound exists for the other case: a velocity field whose *peak* is far above its
reference speed, which is where the second-order Maxwellian's own truncation stops being
a small correction and starts being the answer. It is checked against whatever peak the
caller declares rather than against the reference speed.
"""

TAU_WINDOW = (0.55, 1.5)
"""The band `Scaling.for_coupling` aims tau_plus into when it is free to choose.

Not a stability bound -- tau is stable well outside it -- but the band where neither end
of the trade-off is biting: comfortably above `STIFF_TAU` at the bottom, and at the top
low enough that the lattice velocity needed for a given Reynolds number is still inside
`MACH_MAX`. Only used where there is a free number to spend, which is the coupled case:
`dt_robot` is given, so the substep count is the knob.
"""


class UnphysicalScaling(ValueError):
    """The run cannot be set up as asked; the message says which knob moves it.

    The same contract as `GradeError` in `benchmarks/*/grade.py`: a refusal that does
    not name the fix costs a round of guessing, and the knobs here are always the same
    four -- resolution, timestep, substeps, reference speed.
    """


@dataclass(frozen=True)
class Fluid:
    """A Newtonian fluid, as the two numbers a lattice run needs and nothing else.

    `nu` is kinematic viscosity in m^2/s and `rho` is density in kg/m^3, both at the
    stated temperature, which is carried only so that a report can say which water it
    meant. Compressibility, temperature dependence and any equation of state are absent
    on purpose: this solver is weakly compressible by construction and is held below
    `MACH_MAX` precisely so that none of them is in the answer.
    """

    name: str
    nu: float
    rho: float
    about: str

    def reynolds(self, speed: float, length: float) -> float:
        """U*L/nu, in SI, which is the only dimensionless group this solver has."""
        return abs(speed) * abs(length) / self.nu


FLUIDS: dict[str, Fluid] = {
    "air": Fluid(
        "air", 1.516e-5, 1.204,
        "dry air at 20 C and 101.325 kPa (Sutherland's law for mu, ideal gas for rho)",
    ),
    "water": Fluid(
        "water", 1.0034e-6, 998.21,
        "liquid water at 20 C and 101.325 kPa",
    ),
}
"""The two fluids the brief names, at one temperature each.

Two and not twenty. A table of fluids is a table of numbers somebody has to be right
about, and the failure `toolbox/references/sphere_drag.py` was written against -- a
published number recalled under pressure -- is available here too. These two are the
ones the demos use, and `Fluid` is public so a caller with a third one passes it in with
its own provenance rather than asking this file to have been right about it.

Both are the 20 C values. A run at another temperature passes its own `Fluid`; the
`about` field exists so the report can say which one it was.
"""


def fluid(spec: str | Fluid) -> Fluid:
    """A `Fluid` from a `Fluid` or from a name in `FLUIDS`, else a refusal that lists them."""
    if isinstance(spec, Fluid):
        return spec
    try:
        return FLUIDS[str(spec).strip().lower()]
    except KeyError:
        known = ", ".join(sorted(FLUIDS))
        raise UnphysicalScaling(
            f"no fluid called {spec!r}: known fluids are {known}, or pass a "
            f"units.Fluid(name, nu, rho, about) of your own"
        ) from None


def rates(nu_lattice: float, scheme: str = "trt") -> tuple[float, float]:
    """(omega_plus, omega_minus) for a kinematic viscosity in lattice units.

    Identical in form and in reasoning to `toolbox/lbm.py::rates`, which is where the
    argument for `LAMBDA` lives. `nu_lattice` fixes the even rate and nothing else:
    `tau_plus = nu/c_s^2 + 1/2`. The odd rate is the free one, and the two schemes are
    the two ways of spending it -- `bgk` sets it equal, so the wall position depends on
    the viscosity, and `trt` sets it from `LAMBDA`, so it does not.

    `regularized` returns the same pair as `bgk`: the regularised non-equilibrium is
    even in c, so there is no odd part left for a second rate to relax, and reporting
    two different rates for it would be a fiction. See `DECISIONS_LIVE.md` D3.
    """
    tau_plus = nu_lattice / CS2 + 0.5
    if scheme in ("bgk", "regularized"):
        tau_minus = tau_plus
    elif scheme == "trt":
        tau_minus = 0.5 + LAMBDA / (tau_plus - 0.5)
    else:
        raise UnphysicalScaling(
            f"relaxation scheme {scheme!r} is none of 'trt', 'bgk', 'regularized'"
        )
    return 1.0 / tau_plus, 1.0 / tau_minus


@dataclass(frozen=True)
class Scaling:
    """The conversion between SI and lattice units for one run, and its own verdict.

    Built through one of the four classmethods below rather than directly, because the
    useful question is never "what is dt" -- it is one of:

        for_reynolds    I have a resolution and a Reynolds number (every graded case)
        from_velocity   I have a resolution and a flow speed      (a case with a speed)
        for_coupling    I have a resolution and a robot timestep  (the demos)
        from_dt         I have both and I know what I am doing    (the tests)

    `for_reynolds` is the one to reach for first and was missing from this list: a
    benchmark is specified by a Reynolds number and a geometry, never by a timestep.

    Every field is SI except the three named `_lattice`. `peak_velocity` is the largest
    speed the caller expects anywhere, which is what `U_LATTICE_MAX` is checked against;
    it defaults to the reference speed, and a case that knows better (a rotor tip, a
    cavity corner) says so.
    """

    dx: float
    """Metres per lattice cell."""
    dt: float
    """Seconds per lattice step."""
    fluid: Fluid
    reference_velocity: float
    """The speed the Reynolds number and the Mach check are taken on, in m/s."""
    peak_velocity: float
    """The largest speed expected anywhere in the domain, in m/s."""
    scheme: str = "trt"
    nu_lattice_override: float = 0.0
    """A lattice viscosity chosen for stability rather than derived from the fluid.

    Zero means "derive it", which is what every single-phase case does:
    `nu_lattice = nu dt / dx^2` and the fluid decides. A free-surface case cannot work
    that way and the reason is arithmetic, not taste.

    A free-surface run's timestep is set by **gravity**, because gravity is what drives
    it: `g_lattice = g dt^2 / dx` has to stay small (of order 1e-5 to 1e-4) or the
    scheme goes unstable within a few hundred steps. That fixes `dt` at
    `sqrt(g_lattice dx / g)`, which for a 3 mm cell is about 6e-5 s -- and water's
    viscosity at that timestep is `nu dt / dx^2 = 5e-6` in lattice units, a tau_plus of
    0.500016. There is no resolution that fixes it: `dt` goes as `sqrt(dx)`, so
    `nu dt/dx^2` goes as `dx^-1.5` and refining makes it worse.

    So a free-surface run matches the **Froude** number, which is the one that governs a
    gravity-driven free surface, and accepts a Reynolds number lower than the real
    fluid's. That is standard for this method and it is what `for_froude` does. What
    this repository insists on is that the number be said out loud: `reynolds_effective`
    computes it, `report` prints it beside the physical one, and every free-surface case
    carries both. A dam break at Re 200 against an experiment at Re 50,000 is a
    defensible comparison for a front position, which is Froude-scaled; it is not a
    defensible comparison for anything viscous, and the way to keep that distinction
    alive is to print both numbers every time.
    """
    les: bool = False
    """Whether a subgrid (Smagorinsky) viscosity will be added on top of `nu_lattice`.

    It changes no conversion in this file -- the eddy viscosity is a field, computed per
    node per step in the collision kernel, and no single number here could stand for it.
    What it changes is the verdict: see `STIFF_TAU`. It is carried on the scaling rather
    than passed to `complaints()` because the scaling is what gets reported, and a
    report that does not say the run was under-resolved is the one this file exists to
    prevent.
    """

    # -- the derived lattice quantities ----------------------------------------------

    @property
    def nu_lattice(self) -> float:
        """nu * dt / dx^2, or the override a free-surface run chose instead.

        "The one number that is not a choice" is what this said, and for a single-phase
        run it is still true. `nu_lattice_override` is the free surface's exception and
        its docstring is the argument for it.
        """
        if self.nu_lattice_override:
            return float(self.nu_lattice_override)
        return self.fluid.nu * self.dt / (self.dx * self.dx)

    @property
    def nu_effective(self) -> float:
        """The physical viscosity this lattice is actually running, in m^2/s.

        `nu_lattice dx^2 / dt`. Equal to the fluid's own viscosity whenever nothing was
        overridden, and larger -- usually by orders of magnitude -- when a free-surface
        run has chosen its lattice viscosity for stability.
        """
        return self.nu_lattice * self.dx * self.dx / self.dt

    @property
    def viscosity_ratio(self) -> float:
        """How many times the fluid's own viscosity this run is using. 1.0 if none."""
        return self.nu_effective / self.fluid.nu if self.fluid.nu else float("nan")

    def reynolds_effective(self, length: float) -> float:
        """The Reynolds number this lattice actually achieves on a physical length."""
        return abs(self.reference_velocity) * abs(length) / self.nu_effective

    def froude(self, length: float, gravity: float = 9.81) -> float:
        """`u / sqrt(g L)`, the number a gravity-driven free surface is governed by.

        Matched exactly by construction in a `for_froude` run, because the timestep is
        derived from gravity and the velocity scale from the same length -- which is
        the whole point of giving up on the Reynolds number.
        """
        denominator = math.sqrt(abs(gravity) * abs(length))
        return abs(self.reference_velocity) / denominator if denominator else float("nan")

    @property
    def tau_plus(self) -> float:
        return self.nu_lattice / CS2 + 0.5

    @property
    def tau_minus(self) -> float:
        return 1.0 / rates(self.nu_lattice, self.scheme)[1]

    @property
    def u_lattice(self) -> float:
        """The reference speed in lattice units: `reference_velocity * dt / dx`."""
        return self.reference_velocity * self.dt / self.dx

    @property
    def u_lattice_peak(self) -> float:
        return self.peak_velocity * self.dt / self.dx

    @property
    def mach(self) -> float:
        return self.u_lattice / CS

    @property
    def rates(self) -> tuple[float, float]:
        return rates(self.nu_lattice, self.scheme)

    # -- SI <-> lattice, in both directions ------------------------------------------
    #
    # Written out one method per quantity rather than as a dictionary of exponents. The
    # exponents are the whole content of a unit conversion and a table of them is one
    # transcription away from a force that is off by a factor of dx; named methods are
    # what a caller reads, and `tests/test_live_units.py` round-trips every pair.

    def length_to_lattice(self, metres: float) -> float:
        return metres / self.dx

    def length_to_si(self, cells: float) -> float:
        return cells * self.dx

    def time_to_lattice(self, seconds: float) -> float:
        return seconds / self.dt

    def time_to_si(self, steps: float) -> float:
        return steps * self.dt

    def velocity_to_lattice(self, metres_per_second: float) -> float:
        return metres_per_second * self.dt / self.dx

    def velocity_to_si(self, lattice: float) -> float:
        return lattice * self.dx / self.dt

    def acceleration_to_lattice(self, si: float) -> float:
        return si * self.dt * self.dt / self.dx

    def acceleration_to_si(self, lattice: float) -> float:
        return lattice * self.dx / (self.dt * self.dt)

    def force_to_si(self, lattice: float) -> float:
        """rho * dx^4 / dt^2.

        A force is mass times acceleration; the lattice's mass unit is
        `rho_physical * dx^3` because the lattice density is 1 by construction, and its
        acceleration unit is `dx/dt^2`. Hence the fourth power, which is the single
        easiest exponent in this file to get wrong and the one every drag coefficient
        depends on.
        """
        return lattice * self.fluid.rho * self.dx ** 4 / (self.dt * self.dt)

    def force_to_lattice(self, si: float) -> float:
        return si * (self.dt * self.dt) / (self.fluid.rho * self.dx ** 4)

    def torque_to_si(self, lattice: float) -> float:
        """rho * dx^5 / dt^2: a force times a length."""
        return lattice * self.fluid.rho * self.dx ** 5 / (self.dt * self.dt)

    def torque_to_lattice(self, si: float) -> float:
        return si * (self.dt * self.dt) / (self.fluid.rho * self.dx ** 5)

    def pressure_to_si(self, lattice: float) -> float:
        """rho * dx^2 / dt^2. The lattice's pressure is `rho_l * c_s^2`."""
        return lattice * self.fluid.rho * self.dx * self.dx / (self.dt * self.dt)

    def body_force_to_lattice(self, si_per_volume: float) -> float:
        """A force per unit volume (N/m^3) to its lattice equivalent: dt^2/(rho*dx)."""
        return si_per_volume * (self.dt * self.dt) / (self.fluid.rho * self.dx)

    # -- the verdict -----------------------------------------------------------------

    def reynolds(self, length: float) -> float:
        """The Reynolds number this scaling implies on a physical length, in metres."""
        return self.fluid.reynolds(self.reference_velocity, length)

    def complaints(self) -> list[str]:
        """Everything wrong with this scaling, in full sentences, each naming its knob.

        A list rather than one message because a scaling that is both stiff and
        supersonic should say both: fixing one of them by itself will not help, and a
        caller told only about the first will change the resolution twice.
        """
        said = []
        if not (self.dx > 0.0 and math.isfinite(self.dx)):
            said.append(f"dx is {self.dx}: a cell has a positive finite size")
        if not (self.dt > 0.0 and math.isfinite(self.dt)):
            said.append(f"dt is {self.dt}: a step has a positive finite duration")
        if said:
            return said  # nothing below is meaningful without these

        if self.mach >= MACH_MAX:
            said.append(
                f"lattice Mach number is {self.mach:.3f}, at or above the {MACH_MAX:g} "
                f"ceiling: the scheme's compressibility error is O(Ma^2) and the "
                f"published values this is graded against are incompressible. Ma falls "
                f"with dt at fixed dx, so take smaller steps (more substeps per robot "
                f"step), or raise the resolution"
            )
        if self.u_lattice_peak >= U_LATTICE_MAX:
            said.append(
                f"peak lattice velocity is {self.u_lattice_peak:.3f}, at or above "
                f"{U_LATTICE_MAX:g}, where the second-order equilibrium's own "
                f"truncation stops being a correction. Same knobs as the Mach ceiling: "
                f"smaller dt, or finer dx"
            )
        if self.tau_plus <= STIFF_TAU and not self.les:
            said.append(
                f"tau_plus is {self.tau_plus:.5f}, within {STIFF_TAU - 0.5:g} of the "
                f"1/2 at which the collision stops relaxing anything: nu*dt/dx^2 = "
                f"{self.nu_lattice:.3e} is too small for this lattice. tau_plus rises "
                f"with dt/dx^2, so coarsen dx or take longer steps -- a finer lattice "
                f"at a fixed Reynolds number makes this worse, not better. If no "
                f"affordable dx and dt exist, which is usual for air at a robot's "
                f"timestep, the run is under-resolved and wants the large-eddy path: "
                f"collision='regularized' with smagorinsky on"
            )
        return said

    def notes(self) -> list[str]:
        """What is true about this scaling and is not a refusal. Stated, never implied.

        One entry today, and it is the important one: a large-eddy run below `STIFF_TAU`
        is a run whose molecular viscosity is not resolved anywhere, and whose answer
        away from a wall is the subgrid model's rather than the Navier-Stokes equations'.
        That is a legitimate thing to do and an illegitimate thing to leave unsaid.
        """
        said = []
        if self.les and self.tau_plus <= STIFF_TAU:
            said.append(
                f"under-resolved: tau_plus {self.tau_plus:.5f} is inside "
                f"{STIFF_TAU - 0.5:g} of 1/2, so the molecular viscosity "
                f"(nu*dt/dx^2 = {self.nu_lattice:.3e}) is not resolved on this lattice "
                f"and the subgrid viscosity is carrying the run. Graded answers are "
                f"not taken from a run in this state"
            )
        return said

    def check(self) -> Scaling:
        """Itself if it is runnable, else `UnphysicalScaling` listing every complaint."""
        said = self.complaints()
        if said:
            raise UnphysicalScaling(
                "this run cannot be set up as asked:\n  - " + "\n  - ".join(said)
            )
        return self

    def report(self) -> list[str]:
        """The scaling as a human reads it. One line per fact, no interpretation."""
        omega_plus, omega_minus = self.rates
        return [
            f"  fluid:      {self.fluid.name} "
            f"(nu {self.fluid.nu:.4g} m2/s, rho {self.fluid.rho:.4g} kg/m3)",
            f"  lattice:    dx {self.dx:.6g} m   dt {self.dt:.6g} s   "
            f"dx/dt {self.dx / self.dt:.6g} m/s",
            f"  viscosity:  nu_lattice {self.nu_lattice:.6g}   "
            f"tau_plus {self.tau_plus:.5f}   tau_minus {self.tau_minus:.5f}"
            + ("" if not self.nu_lattice_override else
               f"   (chosen, not derived: nu_eff {self.nu_effective:.4g} m2/s is "
               f"{self.viscosity_ratio:.0f}x the fluid's)"),
            f"  relaxation: {self.scheme}"
            f"{' + smagorinsky' if self.les else ''}  "
            f"omega_plus {omega_plus:.6f}  omega_minus {omega_minus:.6f}",
            f"  speeds:     u_ref {self.reference_velocity:.6g} m/s -> "
            f"{self.u_lattice:.6g} lattice (Ma {self.mach:.4f}), "
            f"peak {self.u_lattice_peak:.6g} lattice",
            *(f"  NOTE:       {note}" for note in self.notes()),
        ]

    # -- the three ways in ------------------------------------------------------------

    @classmethod
    def from_dt(cls, dx: float, dt: float, fluid_spec: str | Fluid = "air", *,
                reference_velocity: float = 0.0, peak_velocity: float | None = None,
                scheme: str = "trt", les: bool = False, check: bool = True) -> Scaling:
        """Both numbers given. The honest primitive; the other two call it."""
        medium = fluid(fluid_spec)
        made = cls(
            dx=float(dx), dt=float(dt), fluid=medium,
            reference_velocity=float(reference_velocity),
            peak_velocity=float(reference_velocity if peak_velocity is None
                                else peak_velocity),
            scheme=scheme, les=bool(les),
        )
        return made.check() if check else made

    @classmethod
    def from_velocity(cls, dx: float, reference_velocity: float,
                      fluid_spec: str | Fluid = "air", *, u_lattice: float = 0.05,
                      peak_velocity: float | None = None, scheme: str = "trt",
                      les: bool = False, check: bool = True) -> Scaling:
        """A resolution and a flow speed: `dt` follows from the lattice velocity asked for.

        `u_lattice = 0.05` is the default because it is what `toolbox/lbm.py`'s
        Poiseuille case uses and for the same stated reason -- it keeps `Ma = 0.0866`,
        inside the `MACH_MAX` ceiling with room for a peak above the reference speed.
        This is the constructor the graded cases use: they are given a Reynolds number
        and a geometry, which fixes `reference_velocity`, and a cell count, which fixes
        `dx`.
        """
        if not reference_velocity:
            raise UnphysicalScaling(
                "from_velocity needs a non-zero reference_velocity: it is what sets dt. "
                "For a case with no free-stream (a hovering rotor, a falling body) use "
                "Scaling.for_coupling or Scaling.from_dt"
            )
        dt = u_lattice * float(dx) / abs(float(reference_velocity))
        return cls.from_dt(dx, dt, fluid_spec, reference_velocity=reference_velocity,
                           peak_velocity=peak_velocity, scheme=scheme, les=les,
                           check=check)

    @classmethod
    def for_reynolds(cls, dx: float, length: float, reynolds: float,
                     fluid_spec: str | Fluid = "water", *, u_lattice: float = 0.05,
                     peak_factor: float = 1.0, scheme: str = "trt", les: bool = False,
                     check: bool = True) -> Scaling:
        """The scaling that puts `reynolds` on `length` with this cell size.

        The constructor every graded case uses, because a benchmark is specified by a
        Reynolds number and a geometry and nothing else: Ghia's cavity is Re 100 on the
        cavity side, Schafer and Turek's cylinder is Re 100 on the diameter, the sphere
        drag curve is Re on the diameter. None of them says what the fluid or the speed
        is, and any pair that gives the right Reynolds number is the same benchmark.

        The arithmetic, which is the one place in this file that runs backwards:

            cells  = length / dx              the geometry in lattice units
            nu_l   = u_lattice * cells / Re   Re is u*L/nu, in lattice units
            dt     = nu_l * dx^2 / nu         because nu_l = nu dt / dx^2
            u_ref  = u_lattice * dx / dt

        so `u_lattice` is the free parameter and `dt` falls out of it. The default 0.05
        is `toolbox/lbm.py`'s, for its stated reason: `Ma = 0.0866`, inside the ceiling
        with room for a peak above the reference speed. `peak_factor` multiplies the
        reference speed to get the peak -- 1.5 for a parabolic channel inflow, where the
        centreline is half again the mean.

        The fluid barely matters here and that is not a bug: it sets `dt` and nothing
        else, because the Reynolds number has already absorbed it. Water is the default
        only so that the reported `dt` and `u_ref` are numbers a person recognises.
        """
        medium = fluid(fluid_spec)
        dx = float(dx)
        cells = float(length) / dx
        if cells <= 0 or reynolds <= 0:
            raise UnphysicalScaling(
                f"for_reynolds needs a positive length and Reynolds number, got "
                f"{length} and {reynolds}"
            )
        nu_lattice = u_lattice * cells / float(reynolds)
        dt = nu_lattice * dx * dx / medium.nu
        u_ref = u_lattice * dx / dt
        return cls.from_dt(dx, dt, medium, reference_velocity=u_ref,
                           peak_velocity=u_ref * float(peak_factor), scheme=scheme,
                           les=les, check=check)

    @classmethod
    def for_froude(cls, dx: float, length: float, fluid_spec: str | Fluid = "water", *,
                   gravity: float = 9.81, g_lattice: float | None = None,
                   mach: float = 0.07, tau: float = 0.53, scheme: str = "trt",
                   les: bool = False, check: bool = True) -> Scaling:
        """The scaling a free-surface case needs: gravity sets the step, tau sets the rest.

        Two numbers are chosen and everything follows:

            g_lattice   gravity in lattice units, `g dt^2 / dx`. The stability limit of
                        a free-surface LBM, and 1e-4 is the usual working value; at 1e-3
                        the interface starts to ring and at 1e-2 it diverges.
            tau         the relaxation time, chosen for stability rather than derived
                        from the fluid, for the reason `nu_lattice_override` gives.

        then

            dt    = sqrt(g_lattice * dx / g)
            nu_l  = (tau - 1/2) c_s^2
            u_ref = sqrt(g L)                 the gravity-wave speed on `length`

        and the Froude number comes out matched while the Reynolds number comes out
        whatever it comes out -- `reynolds_effective` says what, and the report prints
        it next to what the real fluid would have been.

        `u_ref` is `sqrt(g L)` because that is the velocity scale of every case this
        constructor is for: a dam break's front, a sloshing tank's wave, a hull's bow
        wave all move at the shallow-water speed, and it is the right thing to check the
        Mach ceiling against.

        **`g_lattice` defaults to whatever holds the Mach number, and that is not a
        convenience.** Substituting the two definitions gives

            Ma = u_ref dt / (dx c_s) = sqrt(L g_lattice / dx) / c_s

        so the lattice Mach number **rises** as the lattice is refined at a fixed
        `g_lattice` -- the opposite of every single-phase case here, where refining
        lowers it. A 30-cell tank at `g_lattice = 1e-4` sits at Ma 0.0735 and the same
        tank at 60 cells sits at 0.104 and is refused. Anyone refining a free-surface
        case has to halve `g_lattice` when they halve `dx`, and a default that does the
        arithmetic is better than a footnote that says to. Pass `g_lattice` explicitly
        to override it; `mach` sets the target the default aims at.

        The compensation is worth stating because it is the reason refining is still
        worth doing: `dt` falls as `sqrt(g_lattice dx)`, so the effective viscosity
        `nu_l dx^2/dt` falls as `dx^1.5` and the **Reynolds number rises**. A 30-cell
        tank runs at Re 76 and a 60-cell one at Re 153.
        """
        medium = fluid(fluid_spec)
        dx = float(dx)
        if dx <= 0 or length <= 0 or gravity <= 0:
            raise UnphysicalScaling(
                f"for_froude needs positive dx, length and gravity; got "
                f"{dx}, {length}, {gravity}"
            )
        if g_lattice is None:
            # Ma = sqrt(L g_lattice / dx) / c_s, inverted.
            g_lattice = (float(mach) * CS) ** 2 * dx / float(length)
        if g_lattice <= 0:
            raise UnphysicalScaling(f"g_lattice must be positive, got {g_lattice}")
        if tau <= 0.5:
            raise UnphysicalScaling(
                f"tau must exceed 1/2, got {tau}: at 1/2 the collision stops relaxing"
            )
        dt = math.sqrt(g_lattice * dx / gravity)
        nu_lattice = (float(tau) - 0.5) * CS2
        u_ref = math.sqrt(gravity * float(length))
        made = cls(
            dx=dx, dt=dt, fluid=medium, reference_velocity=u_ref,
            peak_velocity=2.0 * u_ref, scheme=scheme,
            nu_lattice_override=nu_lattice, les=bool(les),
        )
        return made.check() if check else made

    @classmethod
    def for_coupling(cls, dx: float, dt_robot: float, fluid_spec: str | Fluid = "air", *,
                     substeps: int | None = None, reference_velocity: float = 0.0,
                     peak_velocity: float | None = None, scheme: str = "regularized",
                     les: bool = True, u_lattice: float = 0.05,
                     max_substeps: int = 4000, check: bool = True
                     ) -> tuple[Scaling, int]:
        """A resolution and the robot's timestep: choose how many lattice steps fit in it.

        The coupled case has exactly one free number and it is the substep count.
        `dt_robot` is the physics engine's step and cannot move -- it is the robot's,
        not the fluid's -- and `dx` is the resolution asked for. So `dt = dt_robot / N`,
        and N is what is chosen here.

        **N is chosen from the velocity ceiling, not from tau, and the difference is
        physical rather than a matter of taste.** `tau_plus = nu*dt/(dx^2 c_s^2) + 1/2`
        *decreases* with N, so there is no N that makes a stiff tau comfortable: the
        largest dt available is `dt_robot` itself (the fluid has to arrive at the same
        instant the robot does), and a finer dx makes tau worse. For air at a
        millisecond timestep and a centimetre lattice there is no resolved-viscosity run
        at any affordable resolution, which is simply what it means for a flow to be
        under-resolved -- see `STIFF_TAU`. What N does control is the lattice velocity,
        `u = u_peak * dt_robot / (N * dx)`, which has to stay under the Mach ceiling or
        the scheme is solving the wrong equations. So:

            N = ceil(u_peak * dt_robot / (u_lattice * dx))

        the smallest N that puts the peak velocity at `u_lattice`, and the smallest N is
        the cheapest run. The defaults are the large-eddy path (`regularized` collision,
        Smagorinsky on) because that is the only path a robot-timestep run has; a caller
        who has a resolved case passes `scheme="trt", les=False` and finds out from
        `check()` whether it is really resolved.

        Returns the scaling and N together, because the caller needs both and deriving
        one from the other at three call sites is how they come to disagree.
        """
        dx = float(dx)
        dt_robot = float(dt_robot)
        if not (dx > 0.0 and dt_robot > 0.0):
            raise UnphysicalScaling(
                f"for_coupling needs a positive dx and dt_robot, got {dx} and {dt_robot}"
            )
        medium = fluid(fluid_spec)
        speed = abs(float(peak_velocity if peak_velocity is not None
                          else reference_velocity))
        if substeps is None:
            if not speed:
                raise UnphysicalScaling(
                    "for_coupling needs a reference_velocity or a peak_velocity to "
                    "choose the substep count from: it is the speed the lattice "
                    "velocity ceiling applies to. Give the fastest motion you expect "
                    "(a rotor's induced velocity, a terminal velocity, a free stream), "
                    "or pass substeps= yourself"
                )
            ideal = speed * dt_robot / (u_lattice * dx)
            substeps = max(1, min(int(max_substeps), int(math.ceil(ideal))))
        substeps = int(substeps)
        if substeps < 1:
            raise UnphysicalScaling(f"substeps must be at least 1, got {substeps}")
        made = cls.from_dt(dx, dt_robot / substeps, medium,
                           reference_velocity=reference_velocity,
                           peak_velocity=peak_velocity, scheme=scheme, les=les,
                           check=False)
        if check:
            said = made.complaints()
            if said:
                raise UnphysicalScaling(
                    f"this coupled run cannot be set up as asked (dx {dx:g} m, "
                    f"dt_robot {dt_robot:g} s, {substeps} substeps, so dt "
                    f"{made.dt:.4g} s):\n  - " + "\n  - ".join(said)
                    + "\n  substeps is the only knob that does not change the robot, "
                      "and it lowers both Ma and tau_plus together: it buys a Mach "
                      "ceiling and it cannot buy a tau"
                )
        return made, substeps
