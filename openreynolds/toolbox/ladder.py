#!/usr/bin/env python3
"""A ladder of reduced cases whose answers are known without running anything.

The expensive failure in CFD is not a case that dies. It is a case that dies after
four things were changed at once, so that six mechanisms can be proposed for it and
all six can be falsified, none of them cheaply. That happened on a free-surface hull
this week: the internal field started at rest instead of at tow speed, the inlet was
ramped instead of held constant, the outlet type changed, and the phase fraction was
hand-written instead of set by `setFields`. None of the four was tested on its own,
and the fault -- a bad pressure at the inlet -- was present in a tank with no hull and
no motion, which is a two-minute run nobody made.

A ladder is the other order. Before the case you were asked for, a short sequence of
**reduced** cases, each adding exactly one piece of physics, each with an answer that
is known **from outside** -- hydrostatics, a symmetry argument, a closed-form integral,
a towing-tank correlation. A rung that fails localises the fault to the one thing that
rung introduced. A rung that passes is evidence you can point at later.

the rule that makes it worth anything

Every rung's expected answer comes from somewhere that is not a solver. Archimedes,
the ITTC-57 line, Kelvin's 19.47 degrees, Hagen-Poiseuille, the Graetz solution, the
divergence theorem. If a rung's answer could only come from another solve, it is not a
rung -- it is the same unknown, cheaper, and it belongs nowhere near a ladder. The
`known` field on every rung in here says the answer *and* where it comes from, and a
test in this repository fails the build if one of them reads like a solver result.

what comes out

The class this case appears to be, the evidence that led to that reading, and the
rungs for that class in order: what each adds, what to measure, the known answer, what
counts as passing, and an honest cost so that skipping is an informed choice rather
than a default.

Detection reads the case rather than asking, and nothing is guessed. A feature that
cannot be detected is reported as undetected, in the same register `preflight.py` uses
for a missing `--resolve`: a script that invented the answer would be inventing the one
input that makes the rest of it mean anything.

This script edits nothing, refuses nothing, and blocks nothing. It writes no case, runs
no solver, and there is no exit code that means "you may not proceed" -- the exit code
is 0 whatever the report says. The rungs are offered. Climbing none of them is a
legitimate choice, and there are good reasons for it: a case with a tutorial behind it
is already a known-good point, and a ladder for a case you have run fifty times is a
tax. The point is to make the question cheap, not mandatory.

    python3 ladder.py /work/case
    python3 ladder.py /work/case --json
    python3 ladder.py /work/case --rung 1
    python3 ladder.py /work/case --record 1 --status pass --value 0.0007
    python3 ladder.py /work/case --list-classes

`--record n` writes a rung's outcome -- pass, fail or skipped, the measured value and
the known answer it was set against -- as one line in the study manifest that
`study_state.py` keeps on the volume. Results recorded there survive the session and
the sandbox, which is how a solver choice established in round 1 of a study was lost
to round 2: the evidence lived only in a transcript a fresh session cannot see. The
report reads the manifest back as an evidence column, and a rung with no record shows
"-" rather than a verdict, because absence of evidence is only absence. Recording the
same rung again replaces the earlier line, so a rung retried after a fix has one
answer and not a history to reconcile.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight  # noqa: E402  (sibling script, not a package)
import study_state  # noqa: E402


# -- what a rung is ----------------------------------------------------------------


class Rung(NamedTuple):
    """One reduced case, and the reason its answer is knowable before it is run.

    `known` is the load-bearing field and the only one with a rule attached: it states
    the answer and where the answer comes from, and the source may never be another
    solve. Everything else on a rung is bookkeeping around that one sentence.

    `overrides` is the dictionary surgery that turns the case as asked into this
    reduced one. This version prints it and applies none of it -- `--rung n` shows it
    so the edits can be made by hand -- but it is carried as data rather than prose so
    that it can be applied mechanically later. Its shape, every key optional:

        {"edit":   {"<relative path>": {"<entry path>": "<new value>"}},
         "remove": ["<relative path>", ...],
         "note":   "what cannot be said as a dictionary edit"}

    An entry path is slash-separated into nested blocks, so `boundaryField/inlet/type`
    is the `type` inside the `inlet` block inside `boundaryField`.
    """

    name: str
    adds: str
    check: str
    known: str
    tolerance: str
    cost: str
    overrides: dict[str, Any] = {}

    def as_dict(self, number: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "adds": self.adds,
            "check": self.check,
            "known": self.known,
            "tolerance": self.tolerance,
            "cost": self.cost,
            "overrides": dict(self.overrides),
        }
        if number is not None:
            out = {"number": number, **out}
        return out


NOT_FROM_A_SOLVE = ("simulation", "cfd", "another run", "a previous run", "a prior run")
"""Words that, in a `known`, mean the rung has no independent answer.

Kept as a constant rather than left in the test because it is the definition of the
idea and not an implementation detail: a rung whose expected value comes from a solve
is a cheaper copy of the same unknown, and the ladder stops meaning anything the
moment one is allowed on. `tests/test_toolbox_ladder.py` reads this list.
"""


def known_is_independent(rung: Rung) -> bool:
    """Whether a rung's stated answer claims a source outside a solver."""
    text = (rung.known or "").lower()
    return bool(text.strip()) and not any(word in text for word in NOT_FROM_A_SOLVE)


# -- reading the case --------------------------------------------------------------


class Signal(NamedTuple):
    """One thing looked for in the case, and what was there.

    `detected` false is a real answer and is printed as one. The alternative -- leaving
    out what was not found -- reads as though it was not looked for, and the difference
    between "there is no gravity in this case" and "nobody checked" is the difference
    between a hydrostatic rung that is available and one that is not.
    """

    name: str
    source: str
    value: str
    detected: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.name,
            "source": self.source,
            "value": self.value,
            "detected": self.detected,
        }


VOF_APPLICATIONS = frozenset({
    "interfoam", "interisofoam", "compressibleinterfoam", "interphasechangefoam",
    "overinterdymfoam", "interfoamdymfoam", "multiphaseinterfoam", "intermixingfoam",
})

CHT_APPLICATIONS = frozenset({
    "chtmultiregionfoam", "chtmultiregionsimplefoam", "chtmultiregiontwophaseeulerfoam",
})

COMPRESSIBLE_APPLICATIONS = frozenset({
    "rhocentralfoam", "rhosimplefoam", "rhopimplefoam", "sonicfoam", "hisa",
    "rhocentraldymfoam", "sonicdymfoam", "rhoporoussimplefoam",
})

FARFIELD_NAMES = re.compile(
    r"far.?field|free.?stream|atmosphere|^sky$|^outer|^ambient|^external", re.I
)
"""Patch names that only appear on a boundary standing in for an unbounded fluid.
An internal-flow case does not have one; an external one nearly always does, whatever
the solver."""

INLET_NAMES = re.compile(r"inlet|^in$|intake|^upstream", re.I)
OUTLET_NAMES = re.compile(r"outlet|^out$|exhaust|^downstream", re.I)

SURFACE_SUFFIXES = (".stl", ".stlb", ".obj")


def vector_entry(text: str, key: str) -> tuple[float, float, float] | None:
    """A `key (a b c);` entry, ignoring any `dimensions [...]` beside it."""
    body = preflight.strip_comments(text or "")
    match = re.search(
        re.escape(key) + r"\s*(?:\[[^\]]*\]\s*)?\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
        body,
    )
    if not match:
        return None
    try:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
    except ValueError:
        return None


def case_patches(case: preflight.Case) -> tuple[list[dict[str, Any]], str]:
    """Patch names and types, from the built mesh if there is one and from the
    dictionary that would build it if there is not.

    The same fallback `preflight.check_empty` makes, for the same reason: a ladder is
    worth most before the mesh exists, and reading only `constant/polyMesh/boundary`
    would make every unmeshed case look like it had no boundary at all.
    """
    patches = case.boundary
    if patches:
        return patches, "constant/polyMesh/boundary"
    block = preflight.parse_block_mesh_boundary(case.read("system/blockMeshDict"))
    if block:
        return block, "system/blockMeshDict"
    return [], ""


def has_body_surface(case: preflight.Case) -> bool:
    """Whether something is immersed in the domain rather than bounding it."""
    if case.surfaces:
        return True
    for relative in ("constant/triSurface", "constant/geometry"):
        directory = case.path / relative
        if directory.is_dir():
            for entry in directory.iterdir():
                if entry.is_file() and entry.suffix.lower() in SURFACE_SUFFIXES:
                    return True
    return bool(case.read("system/snappyHexMeshDict").strip())


def read_signals(case: preflight.Case) -> list[Signal]:
    """Everything the class is decided from, in the order it is printed.

    One pass over the case, one `Signal` per row of the table in the plan. Nothing in
    here interprets: `read_signals` says what is on disk and `classify` says what it
    means, kept apart so that a detection that comes out wrong can be argued with
    against the evidence rather than against the verdict.
    """
    signals: list[Signal] = []

    application = (case.application or "").strip()
    signals.append(Signal(
        "application",
        "system/controlDict",
        application or "no application entry",
        bool(application),
    ))

    alphas = sorted(
        name for name in case.field_texts
        if name == "alpha" or name.startswith("alpha.")
    )
    field_dir = case.field_dir.name if case.field_dir else "0"
    signals.append(Signal(
        "phase fraction",
        f"{field_dir}/",
        ", ".join(alphas) + " -- a volume-of-fluid case, so a free surface"
        if alphas else "no alpha field, so nothing here transports an interface",
        bool(alphas),
    ))

    motion_text = case.read("constant/dynamicMeshDict") or case.read("system/dynamicMeshDict")
    motion_values = preflight.entry_values(motion_text)
    solver_name = (
        motion_values.get("motionSolver")
        or motion_values.get("solver")
        or motion_values.get("dynamicFvMesh")
        or ""
    )
    six_dof = "sixDoFRigidBodyMotion" in motion_text
    signals.append(Signal(
        "body motion",
        "constant/dynamicMeshDict",
        (
            f"{solver_name or 'a motion solver'} -- the body is free to move"
            if six_dof else
            (f"{solver_name} -- the mesh moves, but not as a body responding to the flow"
             if solver_name else "no dynamicMeshDict")
        ),
        bool(motion_text.strip()),
    ))

    gravity = vector_entry(case.read("constant/g"), "value")
    magnitude = math.sqrt(sum(component * component for component in gravity)) if gravity else 0.0
    signals.append(Signal(
        "gravity",
        "constant/g",
        (
            f"({gravity[0]:g} {gravity[1]:g} {gravity[2]:g}), |g| = {magnitude:g} m/s2"
            if gravity and magnitude > 0 else
            ("(0 0 0) -- gravity is present and switched off" if gravity else
             "no constant/g, so no buoyancy term and no hydrostatic reference")
        ),
        bool(gravity) and magnitude > 0,
    ))

    mrf_text = case.read("constant/MRFProperties") or case.read("system/MRFProperties")
    fv_options = case.read("constant/fvOptions") or case.read("system/fvOptions")
    rotating = bool(mrf_text.strip()) or "MRFSource" in fv_options or "rotorDisk" in fv_options
    signals.append(Signal(
        "rotating frame",
        "constant/MRFProperties, fvOptions",
        "MRF zone declared -- rotating machinery" if rotating else "no MRF zone and no rotor source",
        rotating,
    ))

    thermo_text = (
        case.read("constant/thermophysicalProperties")
        or case.read("constant/physicalProperties")
    )
    thermo = preflight.entry_values(preflight.block_body(thermo_text, "thermoType"))
    thermo_type = thermo.get("type", "")
    signals.append(Signal(
        "thermophysics",
        "constant/thermophysicalProperties",
        (
            f"{thermo_type}"
            + (f", {thermo.get('equationOfState')}" if thermo.get("equationOfState") else "")
            if thermo_type else
            "no thermoType, so this is incompressible or isothermal as far as the files say"
        ),
        bool(thermo_type),
    ))

    turbulence_text = (
        case.read("constant/momentumTransport")
        or case.read("constant/turbulenceProperties")
    )
    turbulence_values = preflight.entry_values(turbulence_text)
    simulation_type = turbulence_values.get("simulationType", "")
    model = ""
    for keyword, entry in (("RAS", "RASModel"), ("LES", "LESModel")):
        body = preflight.block_body(turbulence_text, keyword)
        if body:
            model = preflight.entry_values(body).get(entry, "") or preflight.entry_values(body).get("model", "")
            if model:
                break
    signals.append(Signal(
        "turbulence",
        "constant/momentumTransport",
        (f"{simulation_type}" + (f" {model}" if model else "")) if simulation_type
        else "no simulationType entry",
        bool(simulation_type),
    ))

    regions = case.read("constant/regionProperties")
    signals.append(Signal(
        "regions",
        "constant/regionProperties",
        "more than one mesh region -- solid and fluid solved together"
        if regions.strip() else "one region",
        bool(regions.strip()),
    ))

    patches, source = case_patches(case)
    if patches:
        shown = ", ".join(
            f"{patch['name']} ({patch.get('type') or 'no type'})" for patch in patches[:10]
        )
        if len(patches) > 10:
            shown += f", and {len(patches) - 10} more"
    else:
        shown = "no boundary file and no blockMeshDict boundary list"
    signals.append(Signal(
        "patches", source or "constant/polyMesh/boundary", shown, bool(patches)
    ))

    signals.append(Signal(
        "immersed body",
        "constant/triSurface, system/snappyHexMeshDict",
        "a surface is cut into the mesh, so something sits inside the domain"
        if has_body_surface(case) else "no surface geometry, so the mesh is its own boundary",
        has_body_surface(case),
    ))
    return signals


def signal_named(signals: list[Signal], name: str) -> Signal | None:
    for signal in signals:
        if signal.name == name:
            return signal
    return None


# -- deciding which ladder applies -------------------------------------------------


class Detection(NamedTuple):
    """The class, and the sentence that has to survive being disagreed with."""

    key: str
    title: str
    generic: bool
    reason: str
    signals: list[Signal]

    def as_dict(self) -> dict[str, Any]:
        return {
            "class": self.key,
            "class_name": self.title,
            "generic": self.generic,
            "reason": self.reason,
            "evidence": [signal.as_dict() for signal in self.signals],
        }


def classify(case: preflight.Case, signals: list[Signal] | None = None) -> Detection:
    """Which of the four families this looks like, and why.

    The order matters and it runs from the least ambiguous signal to the most. A phase
    fraction field is unarguable; a second mesh region is unarguable; whether a case is
    external or internal is a judgement made from patch names and an immersed body, and
    it is the one that can be wrong. When neither reading is supported the answer is
    "unrecognised" and the generic ladder, because two suggested rungs honestly labelled
    are worth more than six confident ones aimed at the wrong physics.
    """
    signals = signals if signals is not None else read_signals(case)
    application = (case.application or "").strip()
    solver = application.lower()

    phase = signal_named(signals, "phase fraction")
    regions = signal_named(signals, "regions")
    gravity = signal_named(signals, "gravity")
    motion = signal_named(signals, "body motion")

    if solver in CHT_APPLICATIONS or (regions is not None and regions.detected):
        return Detection(
            "conjugate-heat-transfer", "conjugate heat transfer", False,
            "solid and fluid regions are solved together"
            + (f" ({application})" if solver in CHT_APPLICATIONS else
               " (constant/regionProperties names more than one region)"),
            signals,
        )

    if (phase is not None and phase.detected) or solver in VOF_APPLICATIONS:
        parts = []
        if phase is not None and phase.detected:
            parts.append(f"a phase fraction field is present ({phase.value.split(' --')[0]})")
        if solver in VOF_APPLICATIONS:
            parts.append(f"{application} transports an interface")
        if gravity is not None and gravity.detected:
            parts.append(f"gravity is on at {gravity.value.split(', ')[-1]}")
        if motion is not None and "free to move" in motion.value:
            parts.append("the body is free to move")
        return Detection(
            "free-surface-marine", "free-surface marine", False, "; ".join(parts), signals,
        )

    patches, _source = case_patches(case)
    names = [patch["name"] for patch in patches]
    types = {patch["name"]: (patch.get("type") or "") for patch in patches}

    farfield = [name for name in names if FARFIELD_NAMES.search(name)]
    inlets = [name for name in names if INLET_NAMES.search(name)]
    outlets = [name for name in names if OUTLET_NAMES.search(name)]
    body = has_body_surface(case)
    open_types = [
        name for name in names
        if types.get(name) in ("symmetry", "symmetryPlane", "wedge")
    ]

    external_evidence: list[str] = []
    if farfield:
        external_evidence.append(f"a far-field boundary ({', '.join(farfield)})")
    if body:
        external_evidence.append("a surface immersed in the domain rather than bounding it")
    if open_types and not (inlets and outlets):
        external_evidence.append(f"open outer boundaries ({', '.join(open_types)})")

    walls = [name for name in names if types.get(name) == "wall"]
    internal_evidence: list[str] = []
    if inlets and outlets:
        internal_evidence.append(f"one way in and one way out ({inlets[0]} -> {outlets[0]})")
    if walls and not farfield and len(walls) >= max(1, len(names) - len(inlets) - len(outlets) - 1):
        internal_evidence.append("everything that is not an inlet or an outlet is a wall")

    if len(external_evidence) > len(internal_evidence) and external_evidence:
        compressible = solver in COMPRESSIBLE_APPLICATIONS
        thermo = signal_named(signals, "thermophysics")
        if compressible or (thermo is not None and thermo.detected):
            external_evidence.append(f"a compressible solver or thermophysics ({application or 'unnamed'})")
        return Detection(
            "external-aerodynamics", "external aerodynamics", False,
            "; ".join(external_evidence), signals,
        )

    if internal_evidence and len(internal_evidence) >= len(external_evidence):
        return Detection(
            "internal-flow", "internal flow", False, "; ".join(internal_evidence), signals,
        )

    return Detection(
        "generic", "unrecognised", True,
        "nothing in the case decides between a free surface, an external flow, an "
        "internal one and a conjugate problem -- "
        + (f"the patches are {', '.join(names)}" if names else "there are no patches to read")
        + (f" and the application is {application}" if application else " and no application is named"),
        signals,
    )


# -- the catalogues ----------------------------------------------------------------
#
# Every `known` below names a source outside a solver, and the sources are of four
# kinds: an exact solution of the governing equations (Hagen-Poiseuille, the Graetz
# problem, a fluid at rest), a conservation or symmetry argument that is arithmetic on
# the boundary data (Archimedes, Borda-Carnot, zero lift on a symmetric section), a
# closed-form result of a linearised theory that is honest about being one (thin
# aerofoil, Kelvin's wedge), and a measured correlation with a citation (ITTC-57,
# Blasius, Dittus-Boelter). The last kind carries a scatter, and the tolerance says so;
# a correlation quoted without its scatter is a number pretending to be a law.


MARINE = (
    Rung(
        name="still tank",
        adds="nothing: water at rest, no hull in the mesh, no motion, U = 0 everywhere",
        check="the deviation of p_rgh from uniform, and how far the interface moves from "
              "where setFields put it",
        known="Hydrostatics. A fluid at rest under gravity has p = rho*g*h, and p_rgh is "
              "that pressure with the hydrostatic part already subtracted, so p_rgh is "
              "uniform and the interface does not move. Both answers are zero, and they "
              "are zero by the definition of hydrostatic equilibrium rather than by "
              "anyone's prediction.",
        tolerance="the interface within one cell height over the run, and p_rgh varying by "
                  "less than about 1% of rho*g*d across the tank. One cell, because that is "
                  "the resolution of the interface reconstruction and nothing finer is "
                  "meaningful; 1%, because the pressure equation is solved to a tolerance "
                  "and not exactly.",
        cost="minutes. A coarse box from blockMesh, no body to snap to, a few seconds of "
             "physical time. This is the cheapest rung on any ladder in this file.",
        overrides={
            "edit": {
                "system/controlDict": {
                    "endTime": "5", "writeInterval": "1", "adjustTimeStep": "yes",
                },
                "0/U": {"internalField": "uniform (0 0 0)"},
                "0/p_rgh": {"internalField": "uniform 0"},
            },
            "remove": ["constant/dynamicMeshDict"],
            "note": "mesh the tank with blockMesh alone and run no snappyHexMesh stage; keep "
                    "setFields and the same water depth, because the depth is what the "
                    "hydrostatic number is built from",
        },
    ),
    Rung(
        name="the hull at its design draught",
        adds="the body, meshed and held fixed, still water around it",
        check="the vertical component of the pressure force integrated over the hull patch",
        known="Archimedes: the upward force is rho*g*V, with V the volume below the "
              "waterline. For an analytic hull V is closed-form -- a Wigley form "
              "y = (B/2)(1-(2x/L)^2)(1-(z/T)^2) displaces B*(2L/3)*(2T/3) -- and for a "
              "scanned hull it comes from clipping the STL at the waterline and "
              "integrating the closed surface. Both are arithmetic on the geometry.",
        tolerance="within 2-3% of rho*g*V. The gap is the faceting of the hull and the "
                  "cells straddling the waterline, both of which shrink with the mesh; a "
                  "10% or 20% error does not, and means the displacement or the ballast is "
                  "wrong. That exact error -- 20% -- was found reactively on a Wigley case "
                  "after two rounds of chasing it as a meshing defect.",
        cost="tens of minutes: the first rung that needs the hull meshed, so it pays for "
             "the snappy build the rest of the ladder reuses.",
        overrides={
            "edit": {
                "system/controlDict": {"endTime": "10", "writeInterval": "2"},
                "0/U": {
                    "internalField": "uniform (0 0 0)",
                    "boundaryField/inlet/type": "fixedValue",
                    "boundaryField/inlet/value": "uniform (0 0 0)",
                },
            },
            "remove": ["constant/dynamicMeshDict"],
            "note": "keep the hull at the design draught and add a forces function object on "
                    "the hull patch with the correct rhoInf, so the vertical force is written "
                    "every step",
        },
    ),
    Rung(
        name="floating free in still water",
        adds="heave and pitch released; surge, sway, roll and yaw still constrained",
        check="the draught and trim the body settles at, once the transient has rung down",
        known="The design waterline. A body ballasted to displace its own mass floats where "
              "rho*g*V(z) equals its weight, and V(z) is the same closed form used at the "
              "rung below, so the equilibrium draught is the root of a one-line static "
              "equation. The trim is the same statement about the moment.",
        tolerance="draught within a few millimetres on a metre-scale model, and trim within "
                  "a few hundredths of a degree. Worth setting against the signal being "
                  "chased later: a sinkage measurement of 7.5 mm cannot survive a datum that "
                  "is 18 mm out, which is how one case spent two rounds blaming its mesh.",
        cost="tens of minutes, and it needs the mesh motion to be stable before it means "
             "anything -- which is itself the thing this rung tests.",
        overrides={
            "edit": {
                "system/controlDict": {"endTime": "30", "writeInterval": "1"},
                "0/U": {"internalField": "uniform (0 0 0)"},
                "constant/dynamicMeshDict": {
                    "sixDoFRigidBodyMotionCoeffs/constraints": "heave and pitch only",
                },
            },
            "note": "the water stays at rest: no inlet velocity, no towing. Constrain every "
                    "degree of freedom except heave and pitch, so a failure here is about "
                    "buoyancy and not about a yaw instability",
        },
    ),
    Rung(
        name="towed, single phase, no free surface",
        adds="flow at the tow speed, with the interface taken out entirely",
        check="the skin-friction coefficient on the hull",
        known="The ITTC-57 correlation line, Cf = 0.075/(log10(Re) - 2)^2, adopted in 1957 "
              "as a fit to towing-tank flat-plate measurements. It is a measurement "
              "reduced to a formula, and for a slender hull the frictional resistance sits "
              "within a form factor of it -- typically (1+k) with k around 0.1 to 0.3.",
        tolerance="within about 10% of the line before the form factor, and within a few "
                  "per cent once (1+k) is applied. Wider than the rungs above because a "
                  "correlation carries scatter, and quoting one without its scatter turns a "
                  "fit into a law.",
        cost="an hour or two, but on half the cells: with no interface the mesh does not "
             "need the free-surface refinement band, which is usually a third of the count.",
        overrides={
            "edit": {
                "system/controlDict": {
                    "application": "simpleFoam", "endTime": "2000", "writeInterval": "500",
                },
            },
            "remove": [
                "constant/dynamicMeshDict", "system/setFieldsDict", "0/alpha.water",
            ],
            "note": "the double-body model: the hull mirrored about the undisturbed "
                    "waterline, one phase, no wave-making. It excludes wave resistance by "
                    "construction, which is what makes the friction number readable on its own",
        },
    ),
    Rung(
        name="towed with the free surface, body fixed",
        adds="wave-making: the interface is back and the body is moving through it",
        check="the half-angle of the wedge containing the wave pattern, measured off a "
              "plan view of the free surface",
        known="Kelvin's 1887 result: the wake wedge has a half-angle of arcsin(1/3) = "
              "19.47 degrees for any disturbance travelling over deep water. It falls out "
              "of the deep-water dispersion relation and a stationary-phase argument, and "
              "nothing about the hull, the speed or the fluid enters it. A lovely check "
              "precisely because it is so nearly content-free about the case.",
        tolerance="19.5 degrees plus or minus a degree or two on a plan view, provided the "
                  "water is deep relative to the wavelength -- the angle narrows in shallow "
                  "water, and a Froude depth number above about 0.7 invalidates it rather "
                  "than failing it.",
        cost="hours: the full mesh, the full physics, minus only the motion. This is the "
             "first rung that costs what the real case costs.",
        overrides={
            "edit": {"system/controlDict": {"writeInterval": "0.5"}},
            "remove": ["constant/dynamicMeshDict"],
            "note": "hold the body: tow speed at the inlet and on the internal field, no "
                    "degrees of freedom released. A wake angle that is wrong here with the "
                    "body fixed cannot be a motion problem",
        },
    ),
    Rung(
        name="the case as asked",
        adds="the released body in the towed free-surface flow: the deliverable",
        check="whatever the study was commissioned to measure -- resistance, sinkage, trim",
        known="Nothing external. This is the question itself, and it is the one rung on the "
              "ladder whose answer is not knowable in advance, which is the entire reason "
              "the five below it exist. If it disagrees with a published value, the rungs "
              "beneath are where the disagreement is already located.",
        tolerance="whatever the study set, and it belongs in the report next to the number.",
        cost="the cost of the study.",
        overrides={"note": "the case as written; nothing is overridden"},
    ),
)


AERO = (
    Rung(
        name="empty tunnel",
        adds="nothing: the freestream on the background mesh, with no body in it",
        check="the largest departure of U and p from their freestream values anywhere in "
              "the domain",
        known="An undisturbed uniform stream is an exact solution of the governing "
              "equations: with nothing in the domain, the freestream written into the 0/ "
              "directory satisfies them everywhere, for all time. Any gradient that appears "
              "is numerical -- a far-field condition that does not pass a uniform state, a "
              "non-orthogonality artefact, an initialisation that disagrees with its own "
              "boundaries.",
        tolerance="under 0.1% of the freestream in velocity magnitude, and under a few "
                  "hundredths of a per cent in pressure. Not a physical tolerance: it is "
                  "the round-off and linear-solver floor, because the exact answer is zero "
                  "departure.",
        cost="minutes on the background mesh alone, before any snapping.",
        overrides={
            "edit": {"system/controlDict": {"endTime": "100", "writeInterval": "50"}},
            "note": "delete the geometry entry from snappyHexMeshDict, or run blockMesh only. "
                    "The far-field conditions stay exactly as the real case has them, since "
                    "they are what is on trial",
        },
    ),
    Rung(
        name="symmetric body at zero incidence",
        adds="the body, meshed, at zero angle of attack",
        check="lift and side force",
        known="Symmetry. A body symmetric about a plane, in a stream aligned with that "
              "plane, has a solution symmetric about it, so the force component normal to "
              "the plane is identically zero. The argument is about the geometry and the "
              "boundary conditions and holds at every Reynolds number and every Mach "
              "number below the point where the flow becomes unsteady and asymmetric.",
        tolerance="lift coefficient under about 1e-3, and worth reading as a fraction of "
                  "the lift the real case is expected to make rather than in the absolute. "
                  "A non-zero value here is a mesh that is not symmetric, an incidence that "
                  "was applied twice, or a far-field that is too close.",
        cost="hours: the same mesh and the same solve as the real case, at one condition.",
        overrides={
            "edit": {
                "0/U": {
                    "internalField": "uniform (Uinf 0 0)",
                    "boundaryField/farfield/freestreamValue": "uniform (Uinf 0 0)",
                },
            },
            "note": "the freestream vector aligned with the symmetry plane and no component "
                    "across it. If the body is not symmetric this rung does not apply and is "
                    "worth skipping rather than reinterpreting",
        },
    ),
    Rung(
        name="incidence, coarse mesh",
        adds="an angle of attack, on a mesh deliberately too coarse to trust in detail",
        check="the lift slope dCl/dalpha over two or three small angles",
        known="Thin-aerofoil theory: 2*pi per radian, or 0.11 per degree, from the "
              "closed-form solution of the linearised potential equation. It is an "
              "order-of-magnitude reference and says so: real sections come in 5-10% below "
              "it because of the boundary layer, and compressibility raises it by the "
              "Prandtl-Glauert factor 1/sqrt(1 - M^2). A finite wing is lower again by the "
              "aspect-ratio correction.",
        tolerance="within about 20% of the corrected value. Wide on purpose -- this rung is "
                  "asked whether the incidence is being applied at all and in the right "
                  "direction, not whether the lift is right. A slope of the wrong sign, or "
                  "half the expected magnitude, is a rotated freestream or a reference area.",
        cost="under an hour, deliberately: a coarse mesh is enough to get a slope, and a "
             "slope is what is being checked.",
        overrides={
            "edit": {
                "system/controlDict": {"endTime": "500"},
                "0/U": {"internalField": "uniform (Uinf*cos(a) Uinf*sin(a) 0)"},
            },
            "note": "rotate the freestream rather than the mesh, and run two or three small "
                    "angles. Coarsen the refinement levels by one -- the slope survives it "
                    "and the cost falls by about eight",
        },
    ),
    Rung(
        name="the case as asked",
        adds="the full mesh at the requested condition: the deliverable",
        check="whatever the study was commissioned to measure",
        known="Nothing external. This rung is the question. What the three below it buy is "
              "that a disagreement here is already narrowed to the physics this rung adds, "
              "rather than to the far field, the incidence convention or the mesh symmetry.",
        tolerance="whatever the study set.",
        cost="the cost of the study.",
        overrides={"note": "the case as written; nothing is overridden"},
    ),
)


INTERNAL = (
    Rung(
        name="straight duct, fully developed, laminar",
        adds="nothing but the cross-section: a straight length of the same hydraulic "
             "diameter, at a Reynolds number low enough to be laminar",
        check="the velocity profile across the duct, and the ratio of centreline speed to "
              "bulk mean speed",
        known="Hagen-Poiseuille. Fully developed laminar flow in a round pipe has "
              "u(r) = 2*U_bulk*(1 - (r/R)^2), so the centreline is exactly twice the bulk "
              "mean; in a plane channel the parabola gives exactly 3/2. These are exact "
              "solutions of the Navier-Stokes equations for that geometry, not "
              "approximations of one.",
        tolerance="the ratio within 1%, and the profile shape within a per cent or two away "
                  "from the wall. Tight, because the answer is exact: a ratio near 1.2 means "
                  "the flow is not developed yet and the duct is too short, and a ratio near "
                  "1.0 means a turbulence model is switched on when it should not be.",
        cost="minutes. A blockMesh duct, a few thousand cells, and a steady solve.",
        overrides={
            "edit": {
                "system/controlDict": {"application": "simpleFoam", "endTime": "1000"},
                "constant/momentumTransport": {"simulationType": "laminar"},
            },
            "note": "a straight duct of the same cross-section and at least 60 hydraulic "
                    "diameters long, or a cyclic pair driven by a pressure gradient, which "
                    "is developed by construction and much shorter",
        },
    ),
    Rung(
        name="straight duct at the working Reynolds number",
        adds="turbulence: the same duct at the Reynolds number the real case runs at",
        check="the Darcy friction factor from the streamwise pressure gradient, "
              "f = (dp/dx)*D/(0.5*rho*U^2)",
        known="For laminar flow f = 64/Re exactly, from the profile above. For a smooth "
              "turbulent pipe the Blasius fit f = 0.316*Re^-0.25 holds to about Re 1e5, and "
              "the Colebrook-White equation beyond it; both are fits to Nikuradse's pipe "
              "measurements from the 1930s, which is to say they are experimental data with "
              "a curve through them.",
        tolerance="within 5-10% of Blasius or Colebrook. That band is the scatter in the "
                  "underlying measurements plus the wall function; outside it, the y+ is in "
                  "the buffer layer where no wall treatment is valid, which is the usual "
                  "cause and is visible directly from the first cell height.",
        cost="tens of minutes, and the mesh from the rung below is reused unchanged apart "
             "from the near-wall spacing.",
        overrides={
            "edit": {
                "system/controlDict": {"endTime": "3000"},
                "constant/momentumTransport": {"simulationType": "RAS"},
            },
            "note": "same duct, the real Reynolds number, and a near-wall spacing chosen for "
                    "the wall treatment actually in use. Report the y+ range with the "
                    "friction factor: the two are read together or not at all",
        },
    ),
    Rung(
        name="one geometric feature",
        adds="the single feature the real duct has that a straight pipe does not -- the "
             "bend, the expansion, the junction",
        check="the pressure-loss coefficient K = dp_loss/(0.5*rho*U^2) across the feature",
        known="For a sudden expansion, the Borda-Carnot result K = (1 - A1/A2)^2, which is a "
              "control-volume momentum balance across the step and needs nothing else. For "
              "bends, tees and contractions, the handbook K-factors -- Idelchik, or Crane "
              "Technical Paper 410 -- measured in rigs and tabulated by ratio.",
        tolerance="within 10-15% for a handbook K, which is roughly the spread between "
                  "sources for the same fitting; within a few per cent for Borda-Carnot, "
                  "which is derived rather than measured. A K that is out by a factor is "
                  "usually a reference velocity taken at the wrong section.",
        cost="an hour or so: a short duct with the feature in the middle and developed flow "
             "arriving at it.",
        overrides={
            "note": "trim the geometry to an inlet leg, the feature, and an outlet leg long "
                    "enough to re-develop. Measure the loss between two fully developed "
                    "sections and subtract the straight-pipe friction over that length, "
                    "otherwise the friction is counted as part of the fitting",
        },
    ),
    Rung(
        name="the case as asked",
        adds="the full geometry at the working condition: the deliverable",
        check="whatever the study was commissioned to measure",
        known="Nothing external. This rung is the question. The three below it mean that a "
              "total pressure drop which comes out wrong is already separated into the part "
              "that is friction, the part that is a fitting, and the part that is neither.",
        tolerance="whatever the study set.",
        cost="the cost of the study.",
        overrides={"note": "the case as written; nothing is overridden"},
    ),
)


CHT = (
    Rung(
        name="the solid alone, pure conduction",
        adds="nothing but the solid region, with a fixed temperature on each of two faces",
        check="the temperature profile through the solid and the heat flux leaving it",
        known="The one-dimensional steady conduction solution. Across a plane slab T is "
              "linear and q = k*(T1 - T2)/L; through a cylindrical wall T is logarithmic "
              "and q = 2*pi*k*(T1 - T2)/ln(r2/r1). Both are Fourier's law integrated once, "
              "with k read straight out of the material entry in the case.",
        tolerance="the flux within 1% and the profile within a fraction of a degree. The "
                  "answer is exact, so the tolerance is the linear solver's, and a "
                  "disagreement larger than that is nearly always k in the wrong units or a "
                  "thickness taken from the wrong drawing.",
        cost="minutes: one region, no flow, a handful of cells through the thickness.",
        overrides={
            "edit": {
                "system/controlDict": {"application": "laplacianFoam", "endTime": "1000"},
            },
            "note": "solve the solid region on its own with fixedValue temperatures on two "
                    "opposite faces and zeroGradient elsewhere, so the problem really is "
                    "one-dimensional and the closed form really does apply",
        },
    ),
    Rung(
        name="the fluid alone, isothermal wall",
        adds="the flow, with the wall held at a fixed temperature instead of coupled",
        check="the Nusselt number on the heated wall",
        known="For fully developed laminar pipe flow the Graetz problem gives Nu = 3.66 for "
              "a constant wall temperature and 4.36 for a constant wall flux -- exact "
              "eigenvalue results. For turbulent flow the Dittus-Boelter correlation "
              "Nu = 0.023*Re^0.8*Pr^n, with n = 0.4 heating and 0.3 cooling, which is a fit "
              "to measured heat-transfer data over Re > 1e4 and 0.6 < Pr < 160.",
        tolerance="within 2% of the Graetz values, which are exact; within 20-25% of "
                  "Dittus-Boelter, which is the stated scatter of the correlation itself and "
                  "not a slack tolerance.",
        cost="tens of minutes, and it reuses the fluid mesh the real case needs anyway.",
        overrides={
            "edit": {
                "0/T": {
                    "boundaryField/wall/type": "fixedValue",
                    "boundaryField/wall/value": "uniform 350",
                },
            },
            "remove": ["constant/regionProperties"],
            "note": "one region: the fluid, with the coupling replaced by a fixed wall "
                    "temperature. This separates the heat transfer coefficient from the "
                    "interface treatment, which is what the rung above it tests",
        },
    ),
    Rung(
        name="the two regions coupled, steady",
        adds="the interface: the solid and the fluid solved together",
        check="the heat flux integrated on the solid side of the interface against the same "
              "integral on the fluid side, and the overall temperature drop",
        known="Energy conservation across the interface, plus resistances in series. In "
              "steady state the two integrals are the same number, because there is nowhere "
              "for energy to accumulate; and the total drop is q*(L/k + 1/h), the conduction "
              "and film resistances added, with h taken from the rung below. Both are "
              "arithmetic on numbers already in hand.",
        tolerance="the two interface fluxes within 1-2% of each other -- a larger gap is the "
                  "interpolation between non-matching region meshes and is a real finding, "
                  "not noise. The series resistance within 10%, inheriting the correlation's "
                  "scatter from the rung below.",
        cost="hours: both regions, and the coupled solve converges more slowly than either "
             "region alone.",
        overrides={
            "edit": {"system/controlDict": {"endTime": "2000"}},
            "note": "both regions, steady, with the geometry simplified to the slab or "
                    "cylinder the closed form describes. The real geometry comes at the rung "
                    "above",
        },
    ),
    Rung(
        name="the case as asked",
        adds="the real geometry and the real condition: the deliverable",
        check="whatever the study was commissioned to measure",
        known="Nothing external. This rung is the question. What the three below it buy is "
              "that a temperature which comes out wrong is already separated into the "
              "conduction, the film and the coupling.",
        tolerance="whatever the study set.",
        cost="the cost of the study.",
        overrides={"note": "the case as written; nothing is overridden"},
    ),
)


GENERIC = (
    Rung(
        name="does the trivial state stay trivial",
        adds="nothing: the case with its forcing removed, started from the state it should "
             "sit in forever",
        check="any drift at all -- in velocity, in pressure, in whatever the case's primary "
              "field is",
        known="An exact solution stays put. A fluid at rest with uniform properties and no "
              "forcing has zero velocity and uniform pressure for all time, and that is a "
              "statement about the equations rather than about this case, so it holds "
              "whatever the geometry is. Every unit of drift is numerical, and it comes "
              "from the boundary conditions, the initialisation or the discretisation -- a "
              "much smaller set of suspects than the full case has.",
        tolerance="drift under about 1e-6 of the case's own velocity or pressure scale over "
                  "a few hundred steps. There is no physical tolerance here because the "
                  "exact answer is exactly nothing.",
        cost="minutes on the coarsest mesh that has the real boundary conditions on it.",
        overrides={
            "edit": {"system/controlDict": {"endTime": "100", "writeInterval": "50"}},
            "note": "keep the boundary types and the mesh topology, remove the driving: no "
                    "inlet velocity, no pressure difference, no motion, no source terms",
        },
    ),
    Rung(
        name="does a static balance close",
        adds="the imposed state, held: whatever is being driven is set and not solved for",
        check="a conserved quantity summed over the boundary -- mass in against mass out, or "
              "the force on a body against the pressure imposed on it",
        known="Conservation, via the divergence theorem. Over a closed boundary with no "
              "sources inside it the net mass flux is zero, and the force on a body under a "
              "known pressure field is the integral of that field over its surface. Both "
              "are arithmetic on the boundary data and the geometry; neither is a prediction "
              "about the flow.",
        tolerance="the imbalance under 0.1% of the throughput, or the force within the "
                  "faceting error of the surface. Larger than that is a boundary condition "
                  "that is not doing what it is written to do, which is the cheapest fault "
                  "there is to fix and the most expensive to find late.",
        cost="minutes, and often no solve at all -- the mass balance can be read off a "
             "single step, and the force integral off the geometry.",
        overrides={
            "note": "cannot be written as a fixed set of dictionary edits without knowing the "
                    "case class, which is why this ladder is labelled generic. The shape of "
                    "the edit is: impose the driving state everywhere, take one step, and "
                    "sum the boundary fluxes",
        },
    ),
)


CATALOGUE: dict[str, tuple[Rung, ...]] = {
    "free-surface-marine": MARINE,
    "external-aerodynamics": AERO,
    "internal-flow": INTERNAL,
    "conjugate-heat-transfer": CHT,
    "generic": GENERIC,
}

CLASS_TITLES: dict[str, str] = {
    "free-surface-marine": "free-surface marine",
    "external-aerodynamics": "external aerodynamics",
    "internal-flow": "internal flow",
    "conjugate-heat-transfer": "conjugate heat transfer",
    "generic": "unrecognised",
}


GENERIC_CAVEAT = (
    "This case was not recognised as any of the classes this script has a ladder for, "
    "so what is offered is the two rungs that are available almost anywhere: does the "
    "trivial state stay trivial, and does a static balance close. They are genuinely "
    "generic and they are not a ladder for this case in particular -- a class-specific "
    "one would be better and is not being claimed."
)

OFFER = (
    "These rungs are offered, not owed. Nothing here has been run, nothing has been "
    "changed on disk, and no exit code from this script means you may not proceed. "
    "Climbing none of them is a legitimate choice -- a case adapted from a worked "
    "tutorial already sits on a known-good point, and a ladder for a case you have run "
    "fifty times is a tax. The reason to climb one is that a rung which fails names the "
    "single thing it introduced, and a rung which passes is evidence that survives into "
    "the next session."
)


def ladder_for(key: str) -> tuple[Rung, ...]:
    return CATALOGUE.get(key, GENERIC)


# -- recorded evidence -------------------------------------------------------------


def parse_value(text: str | None) -> Any:
    """`--value` as a number when it is one, and as itself when it is not.

    A measured value is usually a float, but "19.4 deg, shallow side" is a
    legitimate reading too, and refusing it would lose the record over its format.
    """
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def recorded_evidence(case_path: Path | str, class_key: str) -> dict[int, dict[str, Any]]:
    """What the study manifest already holds for this ladder, by rung number.

    Read back from `<study>/.reynolds/manifest.jsonl`, the file `study_state.py`
    keeps on the volume, so a rung climbed in an earlier session shows up here
    without anyone being told to look for it. Only rows recorded under the same
    class count: evidence for a different ladder is not evidence for this one.
    A manifest that cannot be read is answered with no evidence, not an error --
    the ladder is still worth printing.
    """
    rows: dict[int, dict[str, Any]] = {}
    try:
        recorded = study_state.rung_evidence(root=case_path)
    except OSError:
        return rows
    for row in recorded:
        meta = row.get("meta") or {}
        if str(meta.get("class", "")) != class_key:
            continue
        try:
            rows[int(meta["rung"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def value_phrase(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def evidence_phrase(row: dict[str, Any] | None) -> str:
    """One recorded outcome as a line, or "-" for none.

    "-" and never "fail": a rung nobody has recorded is absence of evidence, and
    printing absence as a verdict would be inventing the one thing this column
    exists to keep honest.
    """
    if not row:
        return "-"
    meta = row.get("meta") or {}
    parts = [str(meta.get("status", "?"))]
    if meta.get("value") is not None:
        parts.append(f"value {value_phrase(meta['value'])}")
    if row.get("at"):
        parts.append(f"recorded {row['at']}")
    if meta.get("note"):
        parts.append(str(meta["note"]))
    return "   ".join(parts)


def evidence_as_json(evidence: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """The recorded rows flattened for `--json`, in rung order."""
    out: list[dict[str, Any]] = []
    for number in sorted(evidence):
        row = evidence[number]
        meta = row.get("meta") or {}
        out.append({
            "rung": number,
            "name": str(meta.get("name", "")),
            "class": str(meta.get("class", "")),
            "status": str(meta.get("status", "")),
            "value": meta.get("value"),
            "known": str(meta.get("known", "")),
            "note": str(meta.get("note", "")),
            "at": str(row.get("at", "")),
        })
    return out


# -- the report --------------------------------------------------------------------


def render(
    detection: Detection,
    rungs: tuple[Rung, ...],
    case_path: Path | str,
    evidence: dict[int, dict[str, Any]] | None = None,
) -> str:
    """The detected class, the evidence for it, then the rungs in order.

    Laid out like `preflight.render`: a heading, a one-line summary, then labelled
    fields per item, in a fixed column so two reports diff against each other. The
    evidence comes before the rungs on purpose -- the class is the one thing in here
    that can be wrong, and it should be arguable before anything downstream of it is
    read.

    `evidence` is what the study manifest already records for these rungs, printed
    as a column so a fresh session sees what an earlier one established without
    asking. A rung without a record shows "-".
    """
    evidence = evidence or {}
    lines = [f"# ladder {case_path}"]
    lines.append(
        f"{detection.title}  --  {preflight.count_phrase(len(rungs), 'rung')} offered"
    )
    lines.append("")
    lines.append("read from the case")
    for signal in detection.signals:
        mark = " " if signal.detected else "-"
        lines.append(f"  {mark} {signal.name:<16}{signal.value}")
        lines.append(f"    {'':<16}({signal.source})")
    lines.append("")
    lines.append(f"so: {detection.reason}")
    if detection.generic:
        lines.append("")
        lines.append(GENERIC_CAVEAT)

    for number, rung in enumerate(rungs, start=1):
        lines.append("")
        lines.append(f"{number}. {rung.name}")
        lines.append(f"   adds       {rung.adds}")
        lines.append(f"   check      {rung.check}")
        lines.append(f"   known      {rung.known}")
        lines.append(f"   tolerance  {rung.tolerance}")
        lines.append(f"   cost       {rung.cost}")
        lines.append(f"   evidence   {evidence_phrase(evidence.get(number))}")

    lines.append("")
    lines.append(OFFER)
    lines.append("")
    lines.append(
        f"`--rung n` prints one of these in full, with the dictionary edits that would "
        f"turn {Path(str(case_path)).name} into it. `--record n --status pass|fail|skipped "
        f"--value X` writes a rung's outcome to the study manifest on the volume, which is "
        f"where the evidence column above is read from, in this session or the next."
    )
    return "\n".join(lines)


def render_overrides(overrides: dict[str, Any]) -> list[str]:
    """The edits a rung would need, as lines a person can work from by hand."""
    lines: list[str] = []
    edits = overrides.get("edit") or {}
    for relative in sorted(edits):
        for entry in sorted(edits[relative]):
            lines.append(f"   edit       {relative}: {entry} -> {edits[relative][entry]}")
    for relative in overrides.get("remove") or []:
        lines.append(f"   remove     {relative}")
    if overrides.get("note"):
        lines.append(f"   note       {overrides['note']}")
    if not lines:
        lines.append("   (no overrides: this rung is the case as it stands)")
    return lines


def render_rung(
    detection: Detection,
    rungs: tuple[Rung, ...],
    number: int,
    case_path: Path | str,
    evidence: dict[int, dict[str, Any]] | None = None,
) -> str:
    """One rung, everything it holds, including what it would take to build it."""
    evidence = evidence or {}
    if not rungs:
        return f"# ladder {case_path}\nthere are no rungs for {detection.title}."
    if number < 1 or number > len(rungs):
        return (
            f"# ladder {case_path}\n"
            f"{detection.title} has rungs 1 to {len(rungs)}; there is no rung {number}. "
            f"Run without --rung to see them all."
        )
    rung = rungs[number - 1]
    lines = [f"# ladder {case_path}  --  {detection.title}, rung {number} of {len(rungs)}"]
    if detection.generic:
        lines.append("")
        lines.append(GENERIC_CAVEAT)
    lines.append("")
    lines.append(f"{number}. {rung.name}")
    lines.append(f"   adds       {rung.adds}")
    lines.append(f"   check      {rung.check}")
    lines.append(f"   known      {rung.known}")
    lines.append(f"   tolerance  {rung.tolerance}")
    lines.append(f"   cost       {rung.cost}")
    lines.append(f"   evidence   {evidence_phrase(evidence.get(number))}")
    lines.append("")
    lines.append("to build it, from the case as it stands")
    lines.extend(render_overrides(rung.overrides))
    lines.append("")
    lines.append(
        "Nothing above has been applied. This version of the script writes no files; the "
        "edits are printed so they can be made by hand, in a copy of the case."
    )
    return "\n".join(lines)


def as_json(
    detection: Detection,
    rungs: tuple[Rung, ...],
    case_path: Path | str,
    evidence: dict[int, dict[str, Any]] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "case": str(case_path),
        "offered": True,
        "rungs": [rung.as_dict(number) for number, rung in enumerate(rungs, start=1)],
        "recorded": evidence_as_json(evidence or {}),
    }
    payload.update(detection.as_dict())
    return json.dumps(payload, indent=2)


def inspect(case_path: Path | str) -> tuple[Detection, tuple[Rung, ...]]:
    """The whole of the read-only half: detect, then look the ladder up."""
    case = preflight.Case(case_path)
    detection = classify(case)
    return detection, ladder_for(detection.key)


def record_outcome(
    case_path: Path | str,
    detection: Detection,
    rungs: tuple[Rung, ...],
    number: int,
    status: str,
    value: Any = None,
    note: str = "",
) -> int:
    """Write one rung's outcome to the study manifest and say what was written.

    The row carries the rung's own `known` alongside the measured value, so the
    record states what the number was compared against and not only what it was --
    a "pass" whose yardstick has to be re-derived is half a record. Always exits 0:
    a manifest that cannot be written is reported as a fact about the disk, not
    turned into a refusal.
    """
    if number < 1 or number > len(rungs):
        print(
            f"{detection.title} has rungs 1 to {len(rungs)}; there is no rung {number} "
            f"to record. Run without --record to see them."
        )
        return 0
    rung = rungs[number - 1]
    try:
        row = study_state.record_rung(
            number,
            status,
            root=case_path,
            case=Path(str(case_path)).name,
            class_key=detection.key,
            name=rung.name,
            value=value,
            known=rung.known,
            note=note,
        )
    except OSError as error:
        print(f"the outcome was not recorded: {error}")
        print(
            "nothing else is affected -- the report still works, and the record can "
            "be made again once the manifest is writable"
        )
        return 0
    print(json.dumps(row, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, nargs="?", help="the case directory")
    parser.add_argument(
        "--rung", type=int, default=None,
        help="print one rung in full, with the dictionary edits that would produce it",
    )
    parser.add_argument("--json", action="store_true", help="the ladder as JSON")
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    parser.add_argument(
        "--record", type=int, default=None, metavar="N",
        help="write rung N's outcome to the study manifest on the volume",
    )
    parser.add_argument(
        "--status", choices=list(study_state.RUNG_STATUSES), default=None,
        help="what happened on the recorded rung",
    )
    parser.add_argument(
        "--value", default=None,
        help="the measured number, kept next to the known answer it was set against",
    )
    parser.add_argument("--note", default="", help="anything worth carrying with the record")
    parser.add_argument(
        "--list-classes", action="store_true",
        help="name the case classes that have a ladder here, and stop",
    )
    args = parser.parse_args(argv)

    if args.list_classes:
        for key in CATALOGUE:
            print(f"{key:<26}{preflight.count_phrase(len(CATALOGUE[key]), 'rung')}")
        return 0
    if args.case is None:
        parser.error("a case directory is required")

    detection, rungs = inspect(args.case)

    if args.record is not None:
        if args.status is None:
            print(
                f"--record {args.record} needs --status "
                f"{'|'.join(study_state.RUNG_STATUSES)}: the outcome is the record."
            )
            return 0
        return record_outcome(
            args.case, detection, rungs, args.record, args.status,
            parse_value(args.value), args.note,
        )

    evidence = recorded_evidence(args.case, detection.key)
    if args.json:
        text = as_json(detection, rungs, args.case, evidence)
    elif args.rung is not None:
        text = render_rung(detection, rungs, args.rung, args.case, evidence)
    else:
        text = render(detection, rungs, args.case, evidence)
    print(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")

    # Always 0. A ladder that could fail a build would be a workflow with a gate on it,
    # which is the one thing this may not be.
    return 0


if __name__ == "__main__":
    sys.exit(main())
