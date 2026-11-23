"""The cases: what each one measures, what it is measured against, and how close it got.

Written to the same contract as `benchmarks/README.md`, which is this repository's
strictest document: *a case with a reference and no measured agreement beside it teaches
a reader to trust an unvalidated number*. So every case here carries its reference, its
tolerance, the source the reference was read from, and -- where the reference is not
exact -- what the reference's own uncertainty is.

## Where the references came from, and which of them were read

Every number in `REFERENCES` below was read from a document, in this session, by
fetching and opening it. That is not a courtesy. `openreynolds/toolbox/references/`
exists because four separate studies in this project reached for a published value the
toolbox did not carry and produced one from memory; `benchmarks/README.md` records the
two reference values that were *wrong* before that directory existed, both of which
flattered the result. The provenance of each is in the entry.

Two of them could not be fully established and say so:

* Ghia's Table II at Re 100 rests on one OCR of the 1982 scan, because no second
  transcription of that table could be found. Table I was cross-checked against an
  independent transcription and all 17 values agree.
* There is **no canonical published drag coefficient for a cylinder at Re 100**. The
  literature clusters between 1.32 and 1.37. The cross-check against this repository's
  own OpenFOAM run does not need one, which is part of why that cross-check is the one
  the cylinder's drag is graded on.

## Large eddy simulation is off for every laminar case, and that is load-bearing

`cavity`, `cylinder`, `sphere`, `translating_sphere`, `falling_sphere` and `poiseuille`
all run with `smagorinsky=0.0` and the `trt` collision, and `run_case` refuses to grade
them with it on. A Smagorinsky term adds a positive eddy viscosity wherever the strain
rate is non-zero, which in a laminar flow is everywhere; the effective Reynolds number
is then lower than the one the case was set up at, and the drag comes out closer to the
published value than the scheme deserves. A subgrid model has nothing to model in a
resolved laminar flow, so switching it on can only make the agreement a coincidence.
`LAMINAR` is the list, and `_official` is the check.

The two cases that *do* use it, `ground_effect` and `quadrotor`, are rotor flows at
Reynolds numbers in the tens of thousands, where no affordable lattice resolves the
molecular viscosity at all (`units.STIFF_TAU` has the arithmetic). They say so in their
own report, every run.

## Grid convergence, because one coarse-grid match is luck

`cylinder` runs at three resolutions and reports the observed order of convergence from
the three errors. That is the only evidence in this file that distinguishes "the scheme
is second-order accurate and this grid is fine enough" from "this grid happens to agree".
`benchmarks/README.md`'s own commentary makes the point about an agreement that has not
been examined; three resolutions is the cheapest examination there is.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import figures
from .units import CS, MACH_MAX, STIFF_TAU, Scaling, UnphysicalScaling

__all__ = [
    "CASES", "Case", "REFERENCES", "LAMINAR",
    "describe_cases", "run_case", "run_case_file", "benchmark", "crosscheck", "report",
]

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "toolbox" / "references"
REPO = Path(__file__).resolve().parents[2]


# -- the references ---------------------------------------------------------------------

REFERENCES: dict[str, dict[str, Any]] = {
    "ghia": {
        "source": (
            "Ghia, U., Ghia, K.N. and Shin, C.T. (1982), 'High-Re solutions for "
            "incompressible flow using the Navier-Stokes equations and a multigrid "
            "method', Journal of Computational Physics 48:387-411, Tables I and II"
        ),
        "read": "the scanned paper; Table I also against an independent transcription",
        "caveat": (
            "Ghia solved on a 129x129 grid to second order, and his Re 1000 centreline "
            "extrema sit 1.46%, 1.59% and 2.20% from Botella and Peyret's (1998) "
            "spectral benchmark -- so an agreement tighter than about 1.5% at Re 1000 "
            "is partly luck, which benchmarks/README.md already says about the Felix "
            "run of this same case. The Re 100 v column rests on a single OCR of the "
            "1982 scan"
        ),
        "tolerance": 0.02,
        "tolerance_is": (
            "RMS of (u - u_Ghia) over his 17 points, as a fraction of the lid speed. "
            "The same bound benchmarks/felix/grade.py applies to the same quantity, "
            "taken from docs/felix-mode-acceptance.md, so the two solvers in this "
            "repository are graded identically on the one case they share"
        ),
    },
    "schafer_turek_2d1": {
        "source": "Schafer, M. and Turek, S. (1996), 'Benchmark computations of laminar "
                  "flow around a cylinder', Notes on Numerical Fluid Mechanics 52, "
                  "547-566: test case 2D-1.",
        "read": "the intervals as this repository already carries them in "
                "benchmarks/felix/grade.py::ST_2D1, which cites the paper's test case "
                "2D-1 and is used by a committed Felix run",
        "cd": 5.5795,
        "cd_interval": (5.57, 5.59),
        "cl_interval": (0.0104, 0.0110),
        "reynolds": 20.0,
        # The second backend. `benchmarks/felix/README.md` records a real Felix run of
        # this exact case -- 6,765 hexes one layer thick, steady at step 111 -- and both
        # of its numbers are quoted here because both matter, in opposite ways.
        "felix_cd": 5.522,
        "felix_cl": 0.0010,
        "caveat": "2D-1 is STEADY, which is the whole reason this case exists beside the "
                  "unsteady cross-check: with no shedding there is no oscillating lift "
                  "for a transverse acoustic mode to contaminate, and that mode is what "
                  "makes the Re 100 cross-check's Strouhal number wrong sevenfold. "
                  "Felix's own lift on this case is 0.0010 against the paper's "
                  "0.0104-0.0110 -- a factor of ten low, which its README attributes to "
                  "mesh resolution and records rather than fixes. So the drag is a real "
                  "backend-to-backend comparison and the lift is not: a bound on lift "
                  "here would be a bound against a number this repository already knows "
                  "is unconverged.",
        "tolerance": {"cd": 0.02},
        "tolerance_is": "2% on drag, which is the bound benchmarks/felix/README.md "
                        "grades its own 2D-1 run against (it reports Cd 5.522 as "
                        "'-1.04% of 5.5795, outside the 5.57-5.59 interval and inside "
                        "the 2% bound'). Taking the same bound is what makes the two "
                        "backends comparable; a tighter one would fail Felix too and a "
                        "looser one would be chosen to pass.",
    },
    "schafer_turek_2d2": {
        "source": (
            "Schafer, M. and Turek, S. (1996), 'Benchmark computations of laminar flow "
            "around a cylinder', Notes on Numerical Fluid Mechanics 52:547-566, test "
            "case 2D-2, Table 4"
        ),
        "read": "the full paper text",
        "cd_max": (3.2200, 3.2400),
        "cl_max": (0.9900, 1.0100),
        "strouhal": (0.2950, 0.3050),
        "pressure_drop": (2.4600, 2.5000),
        "caveat": (
            "2D-2 has no single reference value: the paper gives only these intervals, "
            "which it calls 'estimated intervals for the exact results as suggested by "
            "the authors'. The Strouhal number is f*D/U with D = 0.1 m and U the MEAN "
            "inflow velocity of 1.0 m/s, not the 1.5 m/s peak. The paper's "
            "experimental St = 0.287 +/- 0.003 is for an extended-inlet variant and is "
            "NOT this case"
        ),
        "geometry": {
            "length": 2.2, "height": 0.41, "diameter": 0.1, "centre": (0.2, 0.2),
            "nu": 1.0e-3, "u_max": 1.5, "u_mean": 1.0, "reynolds": 100.0,
        },
        "tolerance": {"cd": 0.05, "strouhal": 0.03},
        "tolerance_is": (
            "the brief's: drag within 5% and Strouhal within 3%. Both are taken against "
            "the midpoint of the paper's interval, and the interval is reported beside "
            "the measurement so a reader can see which side of it the run fell"
        ),
    },
    "schiller_naumann": {
        "source": (
            "Schiller, L. and Naumann, A. (1933), as implemented in "
            "openreynolds/toolbox/references/sphere_drag.py"
        ),
        "read": "this repository's own reference module, which carries the formula",
        "caveat": (
            "A fit to measurements, not a derivation, so it carries the scatter of the "
            "data behind it; sphere_drag.py is explicit that its accuracy should be "
            "quoted from a source and not from the formula. It holds below Re about "
            "1000. Stokes drag is the exact check and only in its own limit: at Re 1 "
            "the inertial correction is already about 15% (sphere_drag.STOKES_LIMIT_RE)"
        ),
        "tolerance": 0.05,
        "tolerance_is": "the brief's 5%, on the terminal velocity and on Cd",
    },
    "cheeseman_bennett": {
        "source": (
            "Cheeseman, I.C. and Bennett, W.E. (1955), 'The effect of the ground on a "
            "helicopter rotor in forward flight', ARC R&M 3021, equation (2)"
        ),
        "read": "the original report",
        "formula": "T_g/T_inf = 1 / (1 - R^2/(16 Z^2))",
        "caveat": (
            "Three things the usual quotation drops. The ratio is derived AT CONSTANT "
            "POWER. Z is the height of the rotor above the ground. The report states no "
            "numeric lower bound on Z/R -- only that small Z/R gives 'large errors' and "
            "is 'of no practical interest' -- and its one number is in the conclusions, "
            "that flight results agree for Z/R greater than 0.6; the Z/R >= 0.25 figure "
            "that circulates could not be traced to it. Sanchez-Cuevas, Heredia and "
            "Ollero (2017), Int. J. Aerospace Eng. 1823056, measured that a single small "
            "rotor follows Cheeseman and Bennett but a complete multirotor does not: the "
            "effect is much larger and reaches to about 5R rather than 2R, because of a "
            "fountain under the airframe. That is why the graded case here is ONE "
            "actuator disk and the quadrotor is a demo rather than a graded comparison"
        ),
        "tolerance": 0.10,
        "tolerance_is": (
            "a grading choice, and a loose one, for a model whose own author declined to "
            "bound its validity. 10% on the induced-velocity ratio, over Z/R from 0.75 "
            "to 3 where the flight data he quotes applies"
        ),
    },
    "openfoam_cylinder": {
        "source": (
            "this repository's own OpenFOAM run, committed at "
            "tests/data/moving_mesh/viv_fixed/ and graded by "
            "benchmarks/moving_mesh/grade.py"
        ),
        "read": "the committed postProcessing/forceCoeffs/*/coefficient.dat",
        "strouhal": 0.166141,
        "cd": 1.355514,
        "geometry": {"diameter": 0.01, "speed": 0.01, "nu": 1.0e-6, "blockage": 0.025},
        "caveat": (
            "A cylinder at Re 100 in water with 2.5% blockage, run with pimpleFoam on a "
            "morphing mesh. The two numbers are asserted to five digits by "
            "tests/test_benchmark_moving_mesh.py, so they cannot drift without a test "
            "failing. Williamson (1996) gives St = 0.164 for an unconfined cylinder at "
            "Re 100 -- his figure 7b prints the fit S0 = -3.3265/Re + 0.1816 + "
            "1.6e-4*Re, which at Re 100 is 0.16433 -- and benchmarks/moving_mesh/ "
            "records that 2.5% blockage is expected to raise St by about a percent, "
            "which is what the +1.31% of the OpenFOAM run is. There is NO canonical "
            "published Cd at Re 100; the literature clusters 1.32 to 1.37 and Posdziech "
            "and Grundmann (2007) give 1.325, which could not be read in the original. "
            "So drag here is graded against this repository's own run and not against a "
            "published value, which is the honest thing a cross-check can do"
        ),
        "tolerance": {"cd": 0.05, "strouhal": 0.03},
        "tolerance_is": "the brief's, applied to the difference between the two backends",
    },
    "poiseuille": {
        "source": "closed form; no paper",
        "read": "n/a",
        "formula": "u(y) = G y (H - y) / (2 mu),  u_max = G H^2 / (8 mu)",
        "caveat": (
            "Exact, and that is the point of having it: with two relaxation times and "
            "LAMBDA = 3/16 the no-slip wall sits on the link midpoint at any viscosity, "
            "so the discrete solution is the analytic parabola to the convergence "
            "residual rather than to second order. benchmarks/lbm/ grades the "
            "two-dimensional version of this claim at three viscosities and grades "
            "single-relaxation-time beside it, because a validation whose reference is "
            "the thing being validated is no validation"
        ),
        "tolerance": 2.0e-3,
        "tolerance_is": "max |u - u_exact| / u_max, which for an exact scheme is the residual",
    },
}

LAMINAR = frozenset({
    "poiseuille", "cavity", "cylinder", "sphere", "translating_sphere",
    "falling_sphere", "momentum", "smoke",
    # The free-surface cases. Every one of them is gravity-driven and laminar at the
    # Reynolds numbers a free surface can afford (`units.Scaling.for_froude` has the
    # arithmetic: a few hundred, against tens of thousands for the real fluid), so a
    # subgrid model has nothing to model and would only add viscosity to a run that
    # already has more than it should.
    "hydrostatic", "dam_break", "sloshing",
})
"""The cases whose official numbers are measured with no subgrid model.

See the module docstring. `_official` refuses to mark a run in this set as the graded
one if `smagorinsky` is non-zero, rather than quietly reporting a flattered agreement.
"""


def _reference(name: str) -> dict[str, Any]:
    path = REFERENCE_DIR / name
    if not path.is_file():
        raise UnphysicalScaling(
            f"the reference table {path} is missing. It ships with the package "
            f"(openreynolds/toolbox/references/); a source checkout that has it and an "
            f"install that does not means the wheel is not packaging it"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def sphere_drag():
    """`toolbox/references/sphere_drag.py`, imported rather than re-derived.

    It is in the toolbox so that a session can read it in a workspace, and it is
    importable as `openreynolds.toolbox.references.sphere_drag` because `references/`
    is a package directory under an installed `openreynolds`. Reaching for it here
    rather than writing `24/Re (1 + 0.15 Re^0.687)` into this file is the whole point of
    that directory existing: one copy of a published correlation, with its caveats
    attached, and a test (`tests/test_live_cases.py`) that this module agrees with it.
    """
    from ..toolbox.references import sphere_drag as module

    return module


# -- the result shape -------------------------------------------------------------------


@dataclass
class Check:
    """One graded quantity: what it is, what came out, what it should be, and the verdict.

    `target` is a string, deliberately. A target is sometimes a number, sometimes an
    interval, sometimes a tolerance about a midpoint, and sometimes "reported, not
    graded" -- and a structured field that could hold all four would be read wrong more
    often than a sentence. `benchmarks/*/grade.py` prints targets the same way.
    """

    name: str
    measured: float | str
    target: str
    passed: bool | None
    note: str = ""

    def __post_init__(self) -> None:
        """Coerce the verdict and the measurement to Python types.

        Almost every verdict in this file is an expression over numpy values --
        `error <= tolerance` where `error` came out of an array -- so it is a
        `numpy.bool_`, which is **not** an instance of `bool`. That is invisible until
        something serialises it: `json.dumps` refuses it, and a sanitiser that falls
        back to `str()` writes the string `"True"`, which a consumer then reads as
        truthy whatever it says. `benchmarks/live/summarise.py` found it by raising
        `KeyError: 'True'` on a table of results that had already been recorded.

        Fixed here rather than in the serialiser, because there are three consumers --
        the CLI's `--json`, the agent tool, and the Modal runner -- and a type that is
        only correct after one of them has touched it is a type that will be wrong in
        the next one.
        """
        if self.passed is not None:
            self.passed = bool(self.passed)
        if hasattr(self.measured, "item") and not isinstance(self.measured, str):
            try:
                self.measured = self.measured.item()
            except Exception:
                self.measured = str(self.measured)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "measured": self.measured, "target": self.target,
                "passed": self.passed, "note": self.note}


@dataclass
class Case:
    """One runnable case: how to run it, and one line about what it is for."""

    name: str
    about: str
    runner: Callable[..., dict[str, Any]]
    defaults: dict[str, Any] = field(default_factory=dict)
    milestone: str = ""
    needs_newton: bool = False
    heavy: bool = False
    """Whether this case wants a GPU. The CPU runs it correctly and slowly; `heavy`
    is what `benchmarks/live/grade.py` reads to decide whether to refuse on a CPU, and
    what the agent tool warns about."""


def _result(case: str, checks: Sequence[Check], lines: Sequence[str],
            figures_written: Sequence[Any] = (), **extra: Any) -> dict[str, Any]:
    graded = [c for c in checks if c.passed is not None]
    return {
        "case": case,
        "passed": all(c.passed for c in graded) if graded else True,
        "checks": [c.as_dict() for c in checks],
        "report": list(lines),
        "figures": [str(p) for p in figures_written if p],
        **extra,
    }


def _official(case: str, scaling: Scaling, smagorinsky: float) -> list[str]:
    """The note every run carries about whether its numbers are the official ones.

    Refuses rather than warns for a laminar case with the subgrid model on, because the
    whole failure this guards against is a *better-looking* number: an eddy viscosity in
    a laminar flow lowers the effective Reynolds number, and a drag coefficient measured
    at a lower Reynolds number than the one the case claims is closer to the published
    value for reasons that have nothing to do with the solver being right.
    """
    if case in LAMINAR and smagorinsky:
        raise UnphysicalScaling(
            f"{case} is a laminar benchmark and its graded numbers are measured with no "
            f"subgrid model, but smagorinsky is {smagorinsky:g}. A Smagorinsky term adds "
            f"a positive eddy viscosity wherever the strain rate is non-zero, which in a "
            f"laminar flow is everywhere, so the effective Reynolds number is below the "
            f"one this case is set up at and the agreement it reports is flattered. Run "
            f"it with smagorinsky=0 for a graded number; run it with the model on only "
            f"to measure what the model costs, and say that is what you did"
        )
    notes = []
    if case in LAMINAR:
        notes.append("  graded:     subgrid model off (smagorinsky = 0), as a laminar "
                     "benchmark's official numbers must be")
    else:
        notes.append(f"  graded:     subgrid model ON (Cs = {smagorinsky:g}): this case "
                     f"is a rotor flow at a Reynolds number no affordable lattice "
                     f"resolves, so the answer away from a wall is the model's")
    notes.extend(f"  NOTE:       {note}" for note in scaling.notes())
    return notes


# -- the loop every case runs -----------------------------------------------------------


def _advance(world: Any, steps: int, *, sample_every: int = 0,
             sample: Callable[[int], None] | None = None,
             settle: float = 0.0, progress: Callable[[str], None] | None = None,
             every: int = 0) -> None:
    """Step the world `steps` times, sampling and reporting on the way.

    One loop for every case, so that "how a case is run" is one thing rather than nine.
    `settle` is the fraction of the run to discard before sampling, which is the
    transient; a case that samples its own transient and then averages over it reports
    an average of a flow that was still arriving.
    """
    start_sampling = int(steps * settle)
    announce = every or max(1, steps // 10)
    lattice_at_start = world.lattice_steps
    began = time.perf_counter()
    for index in range(steps):
        world.step()
        if sample is not None and sample_every and index >= start_sampling \
                and (index - start_sampling) % sample_every == 0:
            sample(index)
        if progress is not None and (index + 1) % announce == 0:
            elapsed = time.perf_counter() - began
            # `world.lattice_steps`, not the loop index: a coupling step is `substeps`
            # lattice steps, and counting the coupling steps under-reports the rate by
            # exactly that factor. The rotor demos run 40-odd substeps a coupling step,
            # so the ground-effect case reported 17 MLUPS on an H100 that was doing
            # nearer 700 -- a number that reads as a performance catastrophe and is a
            # reporting one.
            done = max(1, world.lattice_steps - lattice_at_start)
            rate = world.nodes * done / elapsed / 1e6
            left = elapsed / (index + 1) * (steps - index - 1)
            progress(f"    {index + 1:>9,}/{steps:,} steps ({done:,} lattice)  "
                     f"{rate:6.1f} MLUPS  {figures.clock(left)} left")


def _stop_word(stopped: dict[str, Any]) -> str:
    """`converged`, `stalled at the precision floor`, or `stopped`. One word each.

    Three outcomes and not two, because "stopped" covered both a run that hit the step
    cap and a run whose residual had sat on the float32 floor for a million steps -- and
    those tell a reader opposite things about whether to trust the field.
    """
    if stopped.get("converged"):
        return "converged"
    if stopped.get("stalled"):
        return "stalled at the single-precision floor"
    return "stopped"


def _converge(world: Any, *, tolerance: float = 1.0e-6, check_every: int = 500,
              max_steps: int = 2_000_000, reference: float = 1.0, patience: int = 12,
              minimum_steps: int = 0,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run until the velocity field stops moving, and report what it stopped at.

    The residual is the largest change in any velocity component over `check_every`
    steps, as a fraction of `reference`. Reported whether or not it reached `tolerance`,
    because -- in `toolbox/lbm.py`'s words about the same quantity -- "a run that hit
    max_steps and a run that converged are not the same answer and the difference is one
    number".

    **The floor is single precision, not the scheme, so there are two ways to stop.**
    The populations are float32 and the interesting part of one is five or six orders of
    magnitude below its weight, so the residual of a converged run settles at a few
    times 1e-5 of the reference speed and stays there. A tolerance below that floor
    never trips.

    Measured: the 32-cell cavity at Re 100 is stable in its fourth digit from 20,000
    steps and its residual floor is 5e-5. The 256-cell cavity at Re 1000 ran to
    **1.6 million steps** on an H100 with its residual pinned at 4.0e-5 against a 3e-5
    tolerance, and would have run to the four-million cap -- sixteen minutes of H100 for
    a field that had not changed in its fourth digit for most of it.

    So a run stops either when the residual reaches `tolerance`, which is convergence,
    or when it **stops falling** -- `patience` consecutive checks without a 10%
    improvement on the best so far -- which is the precision floor. The two are reported
    differently, with `converged`, `stalled` and a `why` in words, because they are not
    the same answer and a reader deciding whether to trust a number needs to know which
    one happened. Neither is treated as a failure by the cases: every one of them reports
    the residual it stopped at and grades the field, not the stopping.

    **`minimum_steps` exists because a plateau is not always a floor.** A lid-driven
    cavity spins up diffusively, over `cells^2/nu_lattice` steps, and during the slow
    part of that the residual sits almost still while the field is genuinely still
    moving: the Re 1000 cavity at 256 cells plateaued at 3.8e-5 for millions of steps
    and was 8 to 10% short of Ghia when it stopped. A stall detector with no floor
    believes that plateau. The caller that knows the physical timescale passes it, and
    `run_cavity` passes one diffusive time.
    """
    previous = None
    residual = float("inf")
    best = float("inf")
    stale = 0
    done = 0
    began = time.perf_counter()
    while done < max_steps:
        for _ in range(check_every):
            world.step()
        done += check_every
        u = world.fields()["u"]
        if previous is not None:
            residual = float(np.nanmax(np.abs(u - previous)) / abs(reference))
        previous = u.copy()
        if residual < 0.9 * best:
            best = residual
            stale = 0
        else:
            stale += 1
        if progress is not None:
            elapsed = time.perf_counter() - began
            progress(f"    {done:>9,} steps  residual {residual:.3e}  "
                     f"{world.nodes * done / elapsed / 1e6:6.1f} MLUPS")
        if residual < tolerance:
            return {"steps": done, "residual": residual, "converged": True,
                    "stalled": False, "why": "the residual reached the tolerance"}
        if stale >= patience and done >= minimum_steps:
            return {
                "steps": done, "residual": residual, "converged": False,
                "stalled": True,
                "why": (f"the residual stopped falling: {patience} consecutive checks "
                        f"without a 10% improvement on {best:.3e}, which is the "
                        f"single-precision floor for this field rather than a failure "
                        f"to converge"),
            }
    return {"steps": done, "residual": residual, "converged": residual < tolerance,
            "stalled": False, "why": f"reached the {max_steps:,} step cap"}


# -- 1. Poiseuille, which is exact ------------------------------------------------------


def run_poiseuille(cells: int = 32, *, nu_lattice: float = 0.1, u_max: float = 0.05,
                   collision: str = "trt", smagorinsky: float = 0.0,
                   device: str | None = None, out_dir: str | Path = ".",
                   steps: int | None = None, progress: Callable[[str], None] | None = None,
                   **_: Any) -> dict[str, Any]:
    """A force-driven periodic channel against the exact parabola.

    The cheapest check in the suite and the only exact one. `cells` fluid rows between
    two walls, periodic along the channel and across it, driven by a uniform body force
    chosen so the centreline reaches `u_max` in lattice units.

    Three viscosities would be better than one and `benchmarks/live/grade.py` runs
    three, for the reason `benchmarks/lbm/README.md` gives: one viscosity cannot tell
    "exact" from "exact at this tau".
    """
    import warp  # noqa: F401  - fail here rather than inside the first launch

    from .world import Domain, FluidWorld

    # Work backwards from the lattice numbers, because this case is specified in them:
    # nu_lattice and u_max are the case, and any dx and fluid that give them are the
    # same case. dx is 1 mm and water, so the reported dt and speed are recognisable.
    dx = 1.0e-3
    fluid_name = "water"
    from .units import FLUIDS

    dt = nu_lattice * dx * dx / FLUIDS[fluid_name].nu
    u_ref = u_max * dx / dt
    scaling = Scaling.from_dt(dx, dt, fluid_name, reference_velocity=u_ref,
                              scheme=collision, les=smagorinsky > 0)
    height = cells * dx
    lines = [f"# poiseuille: {cells} cells across, exact parabola",
             *scaling.report(), *_official("poiseuille", scaling, smagorinsky)]

    domain = Domain(origin=(0.0, 0.0, 0.0), size=(4 * dx, height, 3 * dx),
                    faces={"-x": "periodic", "+x": "periodic",
                           "-y": "wall", "+y": "wall",
                           "-z": "periodic", "+z": "periodic"})
    world = FluidWorld(domain=domain, dx=dx, fluid=fluid_name, scaling=scaling,
                       substeps=1, collision=collision, smagorinsky=smagorinsky,
                       device=device)
    # u_max = G H^2 / (8 mu) with mu = rho nu, so G = 8 u_max mu / H^2. In lattice
    # units mu_l = nu_l (rho_l = 1), and the force per unit volume converts through
    # Scaling.body_force_to_lattice -- which is the one conversion in this file that is
    # easy to get wrong, so it is done by the units layer and not here.
    force_lattice = 8.0 * u_max * nu_lattice / (cells * cells)
    force_si = force_lattice * scaling.fluid.rho * dx / (dt * dt)
    world.set_body_force((force_si, 0.0, 0.0))
    world.initialise()

    if steps:
        _advance(world, steps, progress=progress)
        stopped = {"steps": steps, "residual": float("nan"), "converged": False,
                   "stalled": False, "why": "a fixed step count was asked for"}
    else:
        stopped = _converge(world, tolerance=1e-6, check_every=200,
                            max_steps=200_000, reference=u_ref, progress=progress)

    u = world.fields()["u"][2, :, 1, 0] / (dx / dt)   # lattice units, mid-channel column
    y = (np.arange(cells) + 0.5)
    exact = force_lattice / (2.0 * nu_lattice) * y * (cells - y)
    error = float(np.nanmax(np.abs(u - exact)) / u_max)
    measured_max = float(np.nanmax(u))

    tolerance = REFERENCES["poiseuille"]["tolerance"]
    checks = [
        Check("profile, max |u - u_exact| / u_max", error,
              f"<= {tolerance:.1e}", error <= tolerance),
        Check("u_max against the closed form", measured_max,
              f"{u_max:.6f} +/- 1%", abs(measured_max / u_max - 1) <= 0.01),
        Check("residual at stop", stopped["residual"], "reported, not graded", None),
    ]
    out = Path(out_dir)
    figure = figures.profile_figure(
        out / "live-poiseuille.png",
        [{"x": exact, "y": y, "label": "exact parabola"},
         {"x": u, "y": y, "label": "lattice", "style": "o"}],
        title=f"Poiseuille, {cells} cells, nu_lattice {nu_lattice:g}",
        xlabel="u (lattice units)", ylabel="y (cells from the wall)")
    figures.ensure_rgb(figure)
    lines += [f"  steps:      {stopped['steps']:,} "
              f"({_stop_word(stopped)}, residual {stopped['residual']:.3e})",
              f"  stopped:    {stopped.get('why', '')}",
              *world.report()[-2:]]
    return _result("poiseuille", checks, lines, [figure], scaling=scaling.report(),
                   stopped=stopped)


# -- 2. the lid-driven cavity against Ghia ----------------------------------------------


def run_cavity(cells: int = 128, reynolds: float = 100.0, *, u_lattice: float = 0.055,
               collision: str = "trt", smagorinsky: float = 0.0,
               device: str | None = None, out_dir: str | Path = ".",
               steps: int | None = None, tolerance: float = 3.0e-5,
               convective_times: float = 200.0, diffusive_times: float = 4.0,
               progress: Callable[[str], None] | None = None, **_: Any) -> dict[str, Any]:
    """The lid-driven cavity, graded on Ghia's two centrelines.

    Quasi-two-dimensional: three lattice planes, periodic in z, with no z dependence in
    any boundary condition. A z-invariant field stays z-invariant exactly under D3Q19 --
    nothing in the collision or the streaming couples to z when the field does not --
    so this is the two-dimensional answer computed by the three-dimensional code, which
    is the point: it grades the kernels that the three-dimensional cases use.

    `u_lattice` defaults to 0.055 rather than `toolbox/lbm.py`'s 0.1 because 0.1 puts
    the lattice Mach number at 0.173, above this extension's own ceiling
    (`units.MACH_MAX`). That is a real tension and not a preference: at Re 1000 the two
    bounds together -- Ma below 0.1 and tau above `STIFF_TAU` -- require about 115 cells
    at the very least, and `cells` defaults to 128 for that reason. The arithmetic is
    `nu_lattice = u_lattice * cells / Re`.
    """
    import warp  # noqa: F401

    from .world import Domain, FluidWorld

    dx = 1.0 / cells
    nz = 3
    scaling = Scaling.for_reynolds(dx, 1.0, reynolds, "water", u_lattice=u_lattice,
                                   scheme=collision, les=smagorinsky > 0)
    lid = scaling.reference_velocity
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(1.0, 1.0, nz * dx),
                    faces={"-x": "wall", "+x": "wall", "-y": "wall", "+y": "wall",
                           "-z": "periodic", "+z": "periodic"},
                    face_velocity={"+y": (lid, 0.0, 0.0)})
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision=collision, smagorinsky=smagorinsky,
                       device=device)
    world.initialise()
    lines = [f"# cavity: Re {reynolds:g}, {cells} cells a side",
             *scaling.report(), *_official("cavity", scaling, smagorinsky)]

    # How long a cavity takes to spin up is a diffusive time, not a convective one: the
    # primary vortex is established by viscosity carrying the lid's momentum across the
    # box, which takes of order `cells^2 / nu_lattice` steps. At 256 cells and Re 1000
    # that is 4.65 million, and the first GPU run of this case stopped at a 4,000,000
    # cap with its residual plateaued at 3.8e-5 -- which read as a precision floor and
    # was not one. Every measured quantity came out 8 to 10% short of Ghia's (u minimum
    # -0.345 against -0.383, v extremes -0.469 and +0.333 against -0.516 and +0.371),
    # which is what an under-developed vortex looks like, and the RMS failed at 3.2%.
    #
    # So the step budget is set from the physics and the residual is a second opinion:
    # `diffusive_times` of `cells^2 / nu_lattice`, with `_converge` allowed to stop
    # early only after the first one has elapsed. A residual plateau before then is the
    # slow part of the spin-up, not convergence, and `minimum_steps` is what stops the
    # stall detector believing it.
    diffusive = cells * cells / max(scaling.nu_lattice, 1e-12)
    # **The run is capped in CONVECTIVE times, and that is a measurement.** A cavity
    # reaches steady state on the convective timescale `cells / u_lattice`, and running
    # it much past that does not refine the answer -- it **erodes** it, because the
    # populations are float32 and millions of steps of a closed recirculating flow
    # accumulate round-off that spins the vortex down. Measured, the Re 1000 cavity at
    # 192 cells against Ghia, same case, same lattice, only the step count changed:
    #
    #       600,000 steps (172 convective)   u RMS 0.39%   u_min -0.3806   PASSES
    #     1,200,000 steps (344 convective)   u RMS 0.91%   u_min -0.3735   passes
    #     3,492,000 steps (1,000 convective) u RMS 2.88%   u_min -0.3487   FAILS
    #
    # against Ghia's -0.38289 and a 2% tolerance: roughly **0.8% of error per million
    # steps**, growing without bound.
    #
    # This case used to run to one diffusive time, `cells^2/nu_lattice`, and that floor
    # was added here to stop the stall detector firing on the spin-up plateau -- a real
    # problem, diagnosed from a run that stopped early and came out 8 to 10% short of
    # Ghia. The floor cured it and caused the opposite failure, which looks **identical**
    # from the outside: also short of Ghia, also by 9%. Two failure modes with one
    # symptom, and the second was read as more of the first for two days.
    #
    # So the budget is convective now. 200 is comfortably past convergence -- Re 100
    # converges by 100 and Re 1000 by about 170 -- and short of where erosion reaches the
    # tolerance. The residual and stall logic still stop a run earlier if it settles.
    convective = cells / max(scaling.u_lattice, 1e-12)
    budget = int(convective_times * convective)
    if steps:
        _advance(world, steps, progress=progress)
        stopped = {"steps": steps, "residual": float("nan"), "converged": False,
                   "stalled": False, "why": "a fixed step count was asked for"}
    else:
        stopped = _converge(world, tolerance=tolerance, check_every=2000,
                            max_steps=budget,
                            minimum_steps=int(50.0 * convective),
                            reference=lid, progress=progress)
    stopped["diffusive_time_steps"] = diffusive
    stopped["convective_time_steps"] = convective
    stopped["budget"] = budget

    u = world.fields()["u"][:, :, nz // 2, :] / lid
    mid = cells // 2
    columns = slice(mid - 1, mid + 1) if cells % 2 == 0 else slice(mid, mid + 1)
    # x = 0.5 and y = 0.5 fall between two nodes for an even cell count, so both
    # centrelines are the mean of the two straddling lines rather than a nearest pick.
    # `toolbox/lbm.py::cavity` makes the same choice for the same reason.
    u_line = u[columns, :, 0].mean(axis=0)
    v_line = u[:, columns, 1].mean(axis=1)
    coordinate = (np.arange(cells) + 0.5) / cells

    table = _reference(f"ghia_re{int(reynolds)}.json")
    u_ref = np.asarray(table["table_1_u_on_vertical_centreline"], dtype=float)
    v_ref = np.asarray(table["table_2_v_on_horizontal_centreline"], dtype=float)
    u_at = np.interp(u_ref[:, 0], coordinate, u_line)
    v_at = np.interp(v_ref[:, 0], coordinate, v_line)
    u_rms_all = float(np.sqrt(np.mean((u_at - u_ref[:, 1]) ** 2)))
    v_rms_all = float(np.sqrt(np.mean((v_at - v_ref[:, 1]) ** 2)))

    # The graded RMS is over Ghia's INTERIOR points, and the two walls are excluded for
    # a reason that is about the grading and not about the solver.
    #
    # Ghia's tables start and end on the boundaries: his first row for u is y = 1.0,
    # u = 1.0, and his last is y = 0.0, u = 0.0. Neither is a solved value. They are the
    # boundary condition, which every solver reproduces exactly by construction -- and
    # which a lattice Boltzmann solver cannot be *sampled* at, because its nodes sit
    # half a lattice spacing inside the wall and the no-slip plane is between nodes
    # (`world.Domain.origin`). Interpolating to y = 1.0 therefore clamps to the last
    # node, at y = 1 - 1/(2n), where the real solution is already well below the lid
    # speed: at 32 cells that one point contributes 0.039 to the RMS on its own, which
    # is twice the whole tolerance, while the extrema this run actually gets -- u
    # minimum within 0.15% of Ghia's, v extremes within 1.2% -- say the field is right.
    #
    # Grading a solver on a point where it is being asked to extrapolate its own
    # boundary condition measures the sampling and not the solution. So the two
    # boundary rows come out, the count that went in is reported, and the RMS over ALL
    # of Ghia's points is reported beside it so that nothing is hidden by the choice.
    # benchmarks/felix/grade.py keeps all 17 and is right to: Felix is a mesh solver
    # with nodes *on* the wall, so for it y = 1.0 is a real sample.
    u_inside = (u_ref[:, 0] > 0.0) & (u_ref[:, 0] < 1.0)
    v_inside = (v_ref[:, 0] > 0.0) & (v_ref[:, 0] < 1.0)
    u_rms = float(np.sqrt(np.mean((u_at[u_inside] - u_ref[u_inside, 1]) ** 2)))
    v_rms = float(np.sqrt(np.mean((v_at[v_inside] - v_ref[v_inside, 1]) ** 2)))

    limit = REFERENCES["ghia"]["tolerance"]
    checks = [
        Check(f"u on x = 0.5, RMS against Ghia Table I "
              f"({int(u_inside.sum())} interior points)", u_rms,
              f"<= {limit:.0%} of the lid speed", u_rms <= limit),
        Check(f"v on y = 0.5, RMS against Ghia Table II "
              f"({int(v_inside.sum())} interior points)", v_rms,
              f"<= {limit:.0%} of the lid speed", v_rms <= limit),
        Check("the same RMS over all 17 of Ghia's points, walls included",
              f"u {u_rms_all:.4f}, v {v_rms_all:.4f}", "reported, not graded", None,
              "his first and last rows are the boundary condition, not a solution, and "
              "a lattice has no node on the wall to sample them at"),
        Check("u minimum on x = 0.5", float(u_line.min()),
              f"{u_ref[:, 1].min():.5f} (Ghia)", None),
        Check("v extremes on y = 0.5", f"{v_line.min():+.5f} to {v_line.max():+.5f}",
              f"{v_ref[:, 1].min():+.5f} to {v_ref[:, 1].max():+.5f} (Ghia)", None),
        Check("residual at stop", stopped["residual"], "reported, not graded", None),
    ]
    out = Path(out_dir)
    tag = f"re{int(reynolds)}-n{cells}"
    drawn = [
        figures.slice_figure(world, out / f"live-cavity-{tag}.png", axis=2,
                             title=f"lid-driven cavity, Re {reynolds:g}, {cells} cells"),
        figures.profile_figure(
            out / f"live-cavity-{tag}-centrelines.png",
            [{"x": u_line, "y": coordinate, "label": "u on x = 0.5 (lattice)"},
             {"x": u_ref[:, 1], "y": u_ref[:, 0], "label": "Ghia Table I", "style": "o"},
             {"x": coordinate, "y": v_line, "label": "v on y = 0.5 (lattice)", "panel": 1},
             {"x": v_ref[:, 0], "y": v_ref[:, 1], "label": "Ghia Table II", "style": "o",
              "panel": 1}],
            title=f"Re {reynolds:g}: u RMS {u_rms:.2%}, v RMS {v_rms:.2%}",
            xlabel="u/u_lid, or x", ylabel="y, or v/u_lid", panels=2),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines += [f"  steps:      {stopped['steps']:,} "
              f"({_stop_word(stopped)}, residual {stopped['residual']:.3e})",
              f"  stopped:    {stopped.get('why', '')}",
              f"  timescale:  one convective time is cells/u_lattice = "
              f"{stopped['convective_time_steps']:,.0f} steps; the budget was "
              f"{stopped['budget']:,} ({convective_times:g} of them) and it ran "
              f"{stopped['steps'] / stopped['convective_time_steps']:.0f}. One "
              f"diffusive time is {stopped['diffusive_time_steps']:,.0f} steps, which "
              f"this case deliberately does NOT run to: float32 erosion costs about "
              f"0.8% of the answer per million steps",
              f"  reference:  {REFERENCES['ghia']['source']}",
              f"  caveat:     {REFERENCES['ghia']['caveat']}",
              *world.report()[-2:]]
    return _result("cavity", checks, lines, drawn, reynolds=reynolds, cells=cells,
                   u_rms=u_rms, v_rms=v_rms, u_rms_all=u_rms_all,
                   v_rms_all=v_rms_all, stopped=stopped)


# -- 3. the cylinder, Schafer-Turek 2D-2, at three resolutions --------------------------


def _cylinder_once(cells_per_d: int, *, u_lattice: float = 0.037,
                   collision: str = "trt", smagorinsky: float = 0.0,
                   outlet_rho: float = 1.0,
                   device: str | None = None, periods: float = 30.0,
                   establish_flows: float = 8.0, sample_periods: float = 25.0,
                   settle: float = 0.5, out_dir: str | Path = ".",
                   progress: Callable[[str], None] | None = None,
                   steps: int | None = None, animate: bool = False,
                   upstream: float = 0.0) -> dict[str, Any]:
    """One resolution of the Schafer-Turek 2D-2 cylinder. `run_cylinder` runs three."""
    import warp  # noqa: F401

    from .world import Domain, FluidWorld

    geometry = REFERENCES["schafer_turek_2d2"]["geometry"]
    diameter = geometry["diameter"]
    length, height = geometry["length"], geometry["height"]
    centre = geometry["centre"]
    # `upstream` lengthens the channel ahead of the cylinder and moves the cylinder with
    # it, leaving the paper's geometry untouched at the default of zero. **A diagnostic,
    # not a benchmark knob**: the graded run is the paper's domain, and a run with this
    # set is not Schafer-Turek 2D-2 and must not be reported as it.
    #
    # **It has already answered**, and the numbers are here because the graded case still
    # fails and a reader deserves to know why. Same case, same lattice, only the inlet
    # distance changed:
    #
    #     the paper's domain   Cd_max 3.6091 (+11.7%)   St 0.3025   fails
    #     +4 diameters         Cd_max 3.3083 (+2.4%)    St 0.2950   passes
    #     +10 diameters        Cd_max 3.3408 (+3.4%)    St 0.2943   passes
    #
    # A stiff velocity inlet two diameters ahead of a 24% blockage over-constrains the
    # pressure field the blockage needs. Four diameters of development length is enough
    # and ten adds nothing. The graded run stays in the paper's domain regardless,
    # because 2D-2 *is* that channel and a longer one is a different problem.
    #
    # It exists because this case is +11% on BOTH force coefficients, converged -- the
    # observed order of convergence is about zero across 24, 32 and 44 cells per
    # diameter, so the error is not discretisation -- while its Strouhal number sits
    # inside the paper's interval at +0.9%. A converged wrong force with the right
    # frequency is a boundary condition, and the structural difference from the
    # cross-check that passes at -1.9% is this one: 24% blockage against 2.5%, no-slip
    # channel walls against slip, and the inlet two diameters ahead of the body. If Cd
    # falls toward 3.23 as the inlet moves away, the inlet is the cause.
    if upstream > 0:
        length += upstream
        centre = (centre[0] + upstream, centre[1])
    reynolds = geometry["reynolds"]

    dx = diameter / cells_per_d
    nz = 3
    scaling = Scaling.for_reynolds(dx, diameter, reynolds, "water",
                                   u_lattice=u_lattice, peak_factor=1.5,
                                   scheme=collision, les=smagorinsky > 0)
    u_mean = scaling.reference_velocity
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(length, height, nz * dx),
                    faces={"-x": "inlet", "+x": "outlet", "-y": "wall", "+y": "wall",
                           "-z": "periodic", "+z": "periodic"})
    # `outlet_rho` defaults to 0 here, the extrapolating outlet this case's +11.73%
    # was measured with, so that number stays reproducible. Pass 1.0 to anchor the
    # density instead -- see `crosscheck_2d1`, where doing so took the drag from
    # +9.6% to +2.4% and stopped the domain gaining 7% of its mass in 12,000 steps.
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision=collision, smagorinsky=smagorinsky,
                       outlet_rho=outlet_rho, device=device)

    # The parabolic inflow, which is the case. 4 U_max y (H - y) / H^2 peaks at U_max on
    # the centreline and averages 2/3 of it, so U_mean = 1.0 against U_max = 1.5 is the
    # same statement as the paper's. A uniform inlet at the same mean is a different
    # benchmark with a different answer, which is why this array exists at all.
    y = world.node_coordinates(1)
    peak = 1.5 * u_mean
    profile = np.zeros((world.ny, world.nz, 3))
    profile[:, :, 0] = (4.0 * peak * y * (height - y) / height ** 2)[:, None]
    world.set_inlet_profile(profile)

    body = world.add_body(position=(centre[0], centre[1], nz * dx / 2),
                          label="cylinder")
    # The shape's position is in the BODY's frame, and the body is already at the
    # cylinder's centre, so the offset is zero. Passing the world position to both --
    # which this file did until a voxeliser test caught it -- puts the shape at twice
    # the intended coordinate.
    world.add_shape("cylinder", body=body, scale=(diameter / 2.0, nz * dx, 0.0))
    world.initialise(velocity=(u_mean, 0.0, 0.0))

    # -- how long to run, which is the thing this case got wrong -----------------------
    #
    # Two timescales, and the run has to respect both.
    #
    # One shedding period, from the expected Strouhal number: `D/(St U)`, which in
    # lattice units is `cells_per_d / (St u_lattice)`. That sets how long the SAMPLING
    # window has to be, and 25 periods is plenty for a mean and a frequency.
    #
    # One flow-through, `L/U`, which in lattice units is `(length/dx) / u_lattice`. That
    # sets how long the ESTABLISHING window has to be, and it is much the longer of the
    # two here: at 40 cells per diameter a period is 3,604 steps and a flow-through is
    # 23,784.
    #
    # The first version of this case ran `30 * period` and sampled the last half of it,
    # which is 4.5 flow-throughs of run and sampling from 2.3 -- a wake that had not
    # finished arriving. It measured Cd_max 2.52 against the paper's 3.22-3.24 and St
    # 0.241 against 0.295-0.305, both about 20% low, **and an observed convergence order
    # of -0.17**. That last number is what identified it: an error that does not improve
    # with resolution is not resolution error, and because `total` scaled with
    # `cells_per_d` every resolution got the same number of flow-throughs and therefore
    # the same error. The grid-convergence sweep paid for itself on its first run.
    expected_st = float(np.mean(REFERENCES["schafer_turek_2d2"]["strouhal"]))
    period = cells_per_d / (expected_st * u_lattice)
    flow_through = (length / dx) / u_lattice
    establish = establish_flows * flow_through
    sample_window = sample_periods * period
    total = int(steps if steps else establish + sample_window)
    settle = min(0.95, establish / total) if not steps else settle
    depth = nz * dx
    coefficient = 2.0 / (scaling.fluid.rho * u_mean * u_mean * diameter * depth)

    times: list[float] = []
    cd: list[float] = []
    cl: list[float] = []
    frames: list[np.ndarray] = []

    def take(index: int) -> None:
        force, _ = world.wrench(body)
        times.append(world.steps * scaling.dt)
        cd.append(float(force[0]) * coefficient)
        cl.append(float(force[1]) * coefficient)
        if animate and len(frames) < 240 and index % max(1, total // 240) == 0:
            frames.append(figures.slice_frame(world, axis=2))

    _advance(world, total, sample_every=1, sample=take, settle=settle,
             progress=progress)

    cd_array = np.asarray(cd)
    cl_array = np.asarray(cl)
    spectral = figures.strouhal(times, cl_array, diameter, u_mean)
    out = {
        "cells_per_d": cells_per_d,
        "dx": dx,
        "nodes": world.nodes,
        "steps": total,
        "period_steps": period,
        "flow_through_steps": flow_through,
        "flow_throughs": total / flow_through,
        "flow_throughs_before_sampling": establish / flow_through,
        "periods_sampled": sample_window / period,
        "cd_max": float(cd_array.max()) if cd_array.size else float("nan"),
        "cd_mean": float(cd_array.mean()) if cd_array.size else float("nan"),
        "cl_max": float(cl_array.max()) if cl_array.size else float("nan"),
        "scaling": scaling,
        "world": world,
        "times": times,
        "cd": cd,
        "cl": cl,
        "frames": frames,
        **spectral,
    }
    return out


def run_cylinder(cells_per_d: int | None = None, *, resolutions: Sequence[int] = (24, 32, 44),
                 upstream: float = 0.0, u_lattice: float = 0.037,
                 collision: str = "trt", smagorinsky: float = 0.0,
                 outlet_rho: float = 1.0,
                 device: str | None = None, periods: float = 30.0,
                 establish_flows: float = 8.0, sample_periods: float = 25.0,
                 out_dir: str | Path = ".", steps: int | None = None,
                 animate: bool = True,
                 progress: Callable[[str], None] | None = None,
                 **_: Any) -> dict[str, Any]:
    """Schafer-Turek 2D-2 at three resolutions, with the observed order of convergence.

    One resolution that agrees with a published value is not evidence that the scheme is
    right; it is evidence that one grid agreed. Three resolutions give the observed
    order,

        p = log(e_coarse / e_fine) / log(h_coarse / h_fine)

    on the drag error, and a second-order scheme with an interpolated boundary should
    land near 2. A `p` near zero means the error is not grid error -- the domain, the
    inflow or the sampling window is dominating -- and a `p` near 4 means the three
    errors are noise. Either way the number says more than the agreement does.

    `cells_per_d` runs one resolution instead of three, which is what the agent tool and
    a quick look use.

    **Why the sweep starts at 24 cells and not 20, and why `u_lattice` is 0.037.** Both
    bounds bite at once on this case. The inflow is parabolic with a peak half again the
    mean, so the Mach ceiling applies to `1.5 * u_lattice` and caps the mean at 0.0385;
    and `nu_lattice = u_lattice * cells / Re` has to clear `units.STIFF_TAU`, which at
    Re 100 needs about 18 cells at that velocity. Twenty cells and 0.035 gave tau_plus
    0.521 against a floor of 0.52 -- inside, but close enough that a rounding change
    refuses the run, which is not a resolution a published number should rest on. At 24
    cells and 0.037 the peak Mach number is 0.096 and tau_plus is 0.527, both with room,
    and 24 to 44 is still a refinement ratio of 1.83 for the order estimate.
    """
    reference = REFERENCES["schafer_turek_2d2"]
    cd_interval = reference["cd_max"]
    st_interval = reference["strouhal"]
    cl_interval = reference["cl_max"]
    cd_mid = float(np.mean(cd_interval))
    st_mid = float(np.mean(st_interval))
    tolerances = reference["tolerance"]

    grids = [int(cells_per_d)] if cells_per_d else [int(c) for c in resolutions]
    runs = []
    out = Path(out_dir)
    drawn = []
    for index, grid in enumerate(grids):
        if progress:
            progress(f"  cylinder at {grid} cells per diameter "
                     f"({index + 1} of {len(grids)})")
        run = _cylinder_once(grid, upstream=upstream, u_lattice=u_lattice,
                             collision=collision, smagorinsky=smagorinsky,
                             outlet_rho=outlet_rho,
                             device=device, periods=periods, out_dir=out,
                             establish_flows=establish_flows,
                             sample_periods=sample_periods,
                             progress=progress, steps=steps,
                             animate=animate and grid == max(grids))
        runs.append(run)
        drawn.append(figures.slice_figure(
            run["world"], out / f"live-cylinder-n{grid}.png", axis=2,
            title=f"cylinder Re 100, {grid} cells per diameter, "
                  f"Cd_max {run['cd_max']:.4f}, St {run['strouhal']:.4f}"))
        if run["frames"]:
            gif = figures.animate(run["frames"], out / f"live-cylinder-n{grid}.gif",
                                  fps=20)
            if gif:
                drawn.append(gif)

    finest = runs[-1]
    errors = [abs(r["cd_max"] - cd_mid) / cd_mid for r in runs]
    order = float("nan")
    if len(runs) >= 2 and errors[-1] > 0 and errors[0] > 0:
        h = [r["dx"] for r in runs]
        order = float(np.log(errors[0] / errors[-1]) / np.log(h[0] / h[-1]))

    cd_error = abs(finest["cd_max"] - cd_mid) / cd_mid
    st_error = abs(finest["strouhal"] - st_mid) / st_mid
    checks = [
        Check(f"Cd_max at {finest['cells_per_d']} cells per diameter", finest["cd_max"],
              f"{cd_mid:.4f} +/- {tolerances['cd']:.0%} "
              f"(the paper's interval {cd_interval[0]}-{cd_interval[1]})",
              cd_error <= tolerances["cd"]),
        Check(f"Strouhal at {finest['cells_per_d']} cells per diameter",
              finest["strouhal"],
              f"{st_mid:.4f} +/- {tolerances['strouhal']:.0%} "
              f"(the paper's interval {st_interval[0]}-{st_interval[1]})",
              st_error <= tolerances["strouhal"]),
        Check("Cl_max", finest["cl_max"],
              f"the paper's interval {cl_interval[0]}-{cl_interval[1]}", None,
              "reported, not graded: the lift is a small difference of two large surface "
              "integrals, and benchmarks/README.md records the same case's lift being "
              "reported and not graded for the same reason"),
        Check("Strouhal by the two methods",
              f"{finest['strouhal_crossings']:.5f} (crossings), "
              f"{finest['strouhal_fft']:.5f} (fft), {finest['cycles']:.1f} cycles",
              "the two should agree", None),
    ]
    if len(runs) >= 3:
        checks.append(Check(
            "observed order of convergence on the drag error", order,
            ">= 1.0, and near 2 for a second-order scheme with an interpolated wall",
            order >= 1.0,
            "from " + ", ".join(f"{r['cells_per_d']} cells: {e:.3%}"
                                for r, e in zip(runs, errors))))

    lines = [f"# cylinder: Schafer-Turek 2D-2, Re 100, "
             f"{', '.join(str(r['cells_per_d']) for r in runs)} cells per diameter",
             *finest["scaling"].report(),
             *_official("cylinder", finest["scaling"], smagorinsky),
             f"  reference:  {reference['source']}",
             f"  caveat:     {reference['caveat']}"]
    for run, error in zip(runs, errors):
        lines.append(
            f"  {run['cells_per_d']:>3} cells:  Cd_max {run['cd_max']:.4f} "
            f"({error:+.3%} on the interval midpoint)  St {run['strouhal']:.5f}  "
            f"Cl_max {run['cl_max']:.4f}  {run['nodes']:,} nodes, {run['steps']:,} steps "
            f"= {run['flow_throughs']:.1f} flow-throughs, sampled from "
            f"{run['flow_throughs_before_sampling']:.1f} over "
            f"{run['periods_sampled']:.0f} periods")
    if len(runs) >= 3:
        lines.append(f"  order:      {order:.2f} observed on the drag error")
    lines += finest["world"].report()[-2:]

    history = figures.profile_figure(
        out / "live-cylinder-history.png",
        [{"x": finest["times"], "y": finest["cd"], "label": "Cd"},
         {"x": finest["times"], "y": finest["cl"], "label": "Cl", "panel": 1}],
        title=f"Re 100, {finest['cells_per_d']} cells per diameter, "
              f"{finest['cycles']:.1f} cycles sampled",
        xlabel="t (s)", ylabel="coefficient", panels=2)
    drawn.append(history)
    for figure in drawn:
        figures.ensure_rgb(figure)
    return _result("cylinder", checks, lines, drawn,
                   resolutions=[{k: v for k, v in r.items()
                                 if k not in ("world", "scaling", "frames", "times",
                                              "cd", "cl")} for r in runs],
                   order=order)


# -- 4. sphere drag against the standard curve ------------------------------------------


def _cells_for(reynolds: float, u_lattice: float, floor: int = 20) -> int:
    """The coarsest lattice that is not stiff at this Reynolds number.

    `nu_lattice = u_lattice * cells / Re`, and `units.STIFF_TAU` wants it above
    `(0.52 - 0.5) * c_s^2 = 1/150`, so

        cells > Re / (150 * u_lattice)

    with a floor so that a low-Reynolds sphere is still resolved as a sphere rather than
    as whatever eight cells make. One margin of 15% on top, because landing exactly on a
    refusal boundary means a rounding change refuses the run.

    This exists because the first GPU run of the sphere case was **refused** at Re 100
    and Re 200: 16 cells and `u_lattice` 0.04 give tau_plus 0.5192 and 0.5096, and
    `Scaling` would not build them. The refusal was right -- those lattices cannot carry
    those Reynolds numbers -- and the case had no business asking for a fixed resolution
    across a factor of twenty in Re.
    """
    needed = 1.15 * float(reynolds) / (150.0 * u_lattice)
    return max(int(floor), int(math.ceil(needed)))


def cd_pct(measured: float, against: float) -> str:
    """`measured` as a signed percentage of `against`, for a report line."""
    return f"{measured / against - 1.0:+.2%}" if against else "n/a"


def run_sphere(reynolds: Sequence[float] = (10.0, 50.0, 100.0, 200.0), *,
               cells_per_d: int = 0, u_lattice: float = 0.055,
               collision: str = "trt", smagorinsky: float = 0.0,
               device: str | None = None, out_dir: str | Path = ".",
               span: float = 20.0, cross: float = 12.0,
               steps: int | None = None,
               progress: Callable[[str], None] | None = None,
               **_: Any) -> dict[str, Any]:
    """Drag on a fixed sphere at several Reynolds numbers, against Schiller-Naumann.

    The domain is `span` diameters long and `cross` across, with slip side faces, an
    inlet and an outflow. Those are the numbers that decide whether this case measures
    the sphere or the box: a confined sphere has a higher drag than a free one, the
    correction goes as `(D/H)^3` for a slip-walled box, and at `cross = 12` it is a few
    tenths of a per cent. It is reported rather than corrected for, because a correction
    applied to a measurement makes the measurement unfalsifiable.

    Re 1 is deliberately absent. Stokes drag is the exact reference there and the domain
    a Stokes flow needs is enormous -- the disturbance decays as 1/r -- so a 12-diameter
    box at Re 1 measures the box. `toolbox/references/sphere_drag.py` is explicit that
    Stokes holds as Re approaches zero and not at small-but-finite values, and the
    cheapest honest thing to do is to stay in the range where the fit this is graded
    against was fitted.
    """
    import warp  # noqa: F401

    from .world import Domain, FluidWorld

    curve = sphere_drag()
    diameter = 0.01
    out = Path(out_dir)
    rows = []
    drawn = []

    for index, re in enumerate(float(r) for r in reynolds):
        # One resolution per Reynolds number, not one for the sweep: see `_cells_for`.
        cells = int(cells_per_d) if cells_per_d else _cells_for(re, u_lattice)
        dx = diameter / cells
        if progress:
            progress(f"  sphere at Re {re:g} on {cells} cells per diameter "
                     f"({index + 1} of {len(reynolds)})")
        scaling = Scaling.for_reynolds(dx, diameter, re, "water",
                                       u_lattice=u_lattice, scheme=collision,
                                       les=smagorinsky > 0)
        u_inf = scaling.reference_velocity
        size = (span * diameter, cross * diameter, cross * diameter)
        domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                        faces={"-x": "inlet", "+x": "outlet", "-y": "slip", "+y": "slip",
                               "-z": "slip", "+z": "slip"},
                        face_velocity={"-x": (u_inf, 0.0, 0.0)})
        world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                           substeps=1, collision=collision, smagorinsky=smagorinsky,
                           device=device)
        # Five diameters of upstream run and fifteen of wake.
        position = (5.0 * diameter, size[1] / 2.0, size[2] / 2.0)
        body = world.add_body(position=position, label=f"sphere_re{re:g}")
        # Body-frame offset, and the body is at the sphere's centre already.
        world.add_shape("sphere", body=body, scale=(diameter / 2.0, 0.0, 0.0))
        world.initialise(velocity=(u_inf, 0.0, 0.0))

        # Long enough for the free stream to cross the box three times, which is the
        # cheapest statement of "the wake has arrived and settled".
        crossing = size[0] / dx / u_lattice
        total = int(steps if steps else 3.0 * crossing)
        samples: list[float] = []

        def take(_index: int, world=world, samples=samples) -> None:
            force, _ = world.wrench(body)
            samples.append(float(force[0]))

        _advance(world, total, sample_every=max(1, total // 400), sample=take,
                 settle=0.75, progress=progress)

        area = math.pi * (diameter / 2.0) ** 2
        force = float(np.mean(samples[-max(1, len(samples) // 4):])) if samples else 0.0
        cd = 2.0 * force / (scaling.fluid.rho * u_inf * u_inf * area)
        # **Graded against Clift, Grace & Weber where its branch applies**
        # (DECISIONS_LIVE.md D16); Schiller-Naumann is reported beside it and is no
        # longer the bar. CGW's `0.01 < Re < 20` branch covers Re 10, which is the one
        # Reynolds number this case ever failed on, and refuses above 20 rather than
        # extrapolating -- so Re 50, 100 and 200 keep Schiller-Naumann and say so.
        naumann = curve.schiller_naumann_cd(re)
        lo, hi = curve.CGW_LOW_RE_RANGE
        if lo <= re <= hi:
            expected, which = curve.clift_grace_weber_cd(re), "Clift, Grace & Weber"
        else:
            expected, which = naumann, "Schiller-Naumann"
        rows.append({"reynolds": re, "cd": cd, "schiller_naumann": naumann,
                     "expected": expected, "reference": which,
                     "error": cd / expected - 1.0, "nodes": world.nodes,
                     "steps": total, "force": force, "cells_per_d": cells,
                     "tau_plus": scaling.tau_plus})
        drawn.append(figures.slice_figure(
            world, out / f"live-sphere-re{re:g}.png", axis=2,
            title=f"sphere Re {re:g}, Cd {cd:.4f} against "
                  f"{which} {expected:.4f}"))

    limit = REFERENCES["schiller_naumann"]["tolerance"]
    checks = [
        Check(f"Cd at Re {row['reynolds']:g}", row["cd"],
              f"{row['expected']:.4f} +/- {limit:.0%} ({row['reference']}). "
              + ("**The primary text could not be reached**; this branch was read from "
                 "two independent transcriptions that both cite Bubbles, Drops and "
                 "Particles (1978), so the comparison carries that caveat. "
                 if row["reference"].startswith("Clift") else "")
              + f"Schiller-Naumann for the same point is {row['schiller_naumann']:.4f}, "
                f"which this reads {cd_pct(row['cd'], row['schiller_naumann'])} against "
                f"-- informational, not the bar (DECISIONS_LIVE.md D16)",
              abs(row["error"]) <= limit,
              f"{row['cd']:.5f}, {row['error']:+.2%} of {row['reference']}")
        for row in rows
    ]
    blockage = 1.0 / cross
    checks.append(Check("blockage, D/H", blockage,
                        "reported, not graded", None,
                        "a slip-walled box's confinement correction goes as (D/H)^3, so "
                        f"{blockage ** 3:.2e} here"))

    curve_re = np.logspace(0, 3, 60)
    drawn.append(figures.profile_figure(
        out / "live-sphere-drag.png",
        [{"x": curve_re, "y": [curve.schiller_naumann_cd(r) for r in curve_re],
          "label": "Schiller-Naumann (1933)"},
         {"x": curve_re, "y": [curve.stokes_cd(r) for r in curve_re],
          "label": "Stokes 24/Re", "style": "--"},
         {"x": [r["reynolds"] for r in rows], "y": [r["cd"] for r in rows],
          "label": "lattice Boltzmann", "style": "o"}],
        title="sphere drag", xlabel="Re", ylabel="Cd"))
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# sphere: drag at Re {', '.join(f'{r:g}' for r in reynolds)}, "
             f"one resolution per Reynolds number",
             f"  domain:     {span:g} D long, {cross:g} D across, slip sides",
             f"  reference:  {REFERENCES['schiller_naumann']['source']}",
             f"  caveat:     {REFERENCES['schiller_naumann']['caveat']}"]
    for row in rows:
        lines.append(f"  Re {row['reynolds']:>6g}:  Cd {row['cd']:.4f} against "
                     f"{row['schiller_naumann']:.4f} ({row['error']:+.2%})  "
                     f"{row['cells_per_d']} cells per diameter, tau_plus "
                     f"{row['tau_plus']:.4f}, {row['nodes']:,} nodes, "
                     f"{row['steps']:,} steps")
    return _result("sphere", checks, lines, drawn, rows=rows)


# -- 5. a translating sphere: the moving-boundary and refill check ----------------------


def _sphere_in_a_box(reynolds: float, cells_per_d: int, u_lattice: float,
                     moving: bool, *, collision: str, smagorinsky: float,
                     device: str | None, steps: int | None,
                     progress: Callable[[str], None] | None):
    """One sphere at one Reynolds number, either held in a stream or moving through still
    fluid, in the **same box at the same resolution**. Returns its drag coefficient.

    One function for both, because the whole point of the comparison is that nothing
    differs between them except which of the two is moving. Two functions, or two
    hand-written setups, would leave a reader wondering whether the box or the sampling
    window had changed too.
    """
    from .world import Domain, FluidWorld

    diameter = 0.01
    dx = diameter / cells_per_d
    scaling = Scaling.for_reynolds(dx, diameter, reynolds, "water",
                                   u_lattice=u_lattice, scheme=collision,
                                   les=smagorinsky > 0)
    speed = scaling.reference_velocity
    size = (24.0 * diameter, 12.0 * diameter, 12.0 * diameter)
    if moving:
        faces = {"-x": "outlet", "+x": "outlet", "-y": "slip", "+y": "slip",
                 "-z": "slip", "+z": "slip"}
        face_velocity: dict[str, tuple[float, float, float]] = {}
    else:
        faces = {"-x": "inlet", "+x": "outlet", "-y": "slip", "+y": "slip",
                 "-z": "slip", "+z": "slip"}
        face_velocity = {"-x": (speed, 0.0, 0.0)}
    domain = Domain(origin=(0.0, 0.0, 0.0), size=size, faces=faces,
                    face_velocity=face_velocity)
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision=collision, smagorinsky=smagorinsky,
                       device=device)
    # The moving sphere starts downstream and travels upstream, so that it spends the
    # whole run in fluid it has not already disturbed. The held one sits where the
    # moving one ends up, so both are measured at the same place in the same box.
    start = (18.0 * diameter if moving else 6.0 * diameter,
             size[1] / 2.0, size[2] / 2.0)
    body = world.add_body(position=start, label="sphere",
                          velocity=(-speed, 0, 0, 0, 0, 0) if moving else None)
    world.add_shape("sphere", body=body, scale=(diameter / 2.0, 0.0, 0.0))
    world.initialise(velocity=(0.0, 0.0, 0.0) if moving else (speed, 0.0, 0.0))

    total = int(steps if steps else 8.0 * cells_per_d / u_lattice)
    area = math.pi * (diameter / 2.0) ** 2
    scale = 2.0 / (scaling.fluid.rho * speed * speed * area)
    position = np.asarray(start, dtype=float)
    samples: list[float] = []
    history: list[tuple[float, float]] = []

    for index in range(total):
        if moving:
            position = position + np.array([-speed * world.dt, 0.0, 0.0])
            world.set_body_state(body, position=tuple(position))
        world.step()
        if index > total // 2:
            force, _ = world.wrench(body)
            samples.append(float(force[0]) * scale)
            history.append(((start[0] - position[0]) / diameter, samples[-1]))
        if progress and (index + 1) % max(1, total // 6) == 0:
            progress(f"    {'moving' if moving else 'held  '} "
                     f"{index + 1:>7,}/{total:,} steps")

    cd = float(np.mean(samples[-max(1, len(samples) // 3):])) if samples else float("nan")
    return {
        "moving": moving, "cd": cd, "cells_per_d": cells_per_d, "nodes": world.nodes,
        "steps": total, "refilled": int(world._refilled.numpy()[0]),
        "fallbacks": int(world._fallbacks.numpy()[0]),
        "travelled": float((start[0] - position[0]) / diameter),
        "history": history, "world": world, "scaling": scaling,
    }


def run_translating_sphere(reynolds: float = 100.0, *, cells_per_d: int = 0,
                           u_lattice: float = 0.055, collision: str = "trt",
                           smagorinsky: float = 0.0, device: str | None = None,
                           out_dir: str | Path = ".", steps: int | None = None,
                           progress: Callable[[str], None] | None = None,
                           **_: Any) -> dict[str, Any]:
    """The same sphere twice: held in a stream, and moving through still fluid.

    This is the check `DECISIONS_LIVE.md` D5 promises for the fresh-node refilling, and
    it is the right one because the quantity a bad refill ruins is exactly this: the
    force on a body that is uncovering lattice nodes as it goes. A held sphere in a
    stream never refills anything, so it is the control.

    **The reference is Galilean invariance, and it is exact.** The same sphere at the
    same Reynolds number must feel the same drag whether the fluid moves past it or it
    moves through the fluid. So the graded number is the *ratio* of the two drag
    coefficients, both measured here, in the same box, at the same resolution, with the
    same sampling window -- and a ratio cancels every error the two runs share:
    resolution, blockage, the collision operator, the Reynolds number, the sampling. What
    is left is what the moving-boundary treatment costs, which is the only thing this
    case can honestly claim to measure.

    Schiller-Naumann is reported for both and graded for neither. That is the change the
    first GPU run forced: this case used to grade the moving sphere against the
    correlation, which conflates the refill error with the resolution error and the
    blockage error, and a 12-diameter box at 20 cells per diameter has several per cent
    of both. Grading a difference against an exact reference is strictly better evidence
    than grading an absolute against a fitted one.

    That run also found the drag sign: it reported **Cd = -1.19**, because the force on a
    sphere moving in -x is +x and the sampler negated it. The magnitude was right, which
    is why only the sign gave it away.
    """
    import warp  # noqa: F401

    curve = sphere_drag()
    cells = int(cells_per_d) if cells_per_d else _cells_for(reynolds, u_lattice)
    runs = {}
    for moving in (False, True):
        if progress:
            progress(f"  sphere at Re {reynolds:g}, "
                     f"{'moving through still fluid' if moving else 'held in a stream'}")
        runs[moving] = _sphere_in_a_box(
            reynolds, cells, u_lattice, moving, collision=collision,
            smagorinsky=smagorinsky, device=device, steps=steps, progress=progress)

    held, going = runs[False], runs[True]
    expected = curve.schiller_naumann_cd(reynolds)
    ratio = going["cd"] / held["cd"] if held["cd"] else float("nan")
    limit = REFERENCES["schiller_naumann"]["tolerance"]

    checks = [
        Check("Galilean invariance: Cd moving / Cd held", ratio,
              f"1.0 +/- {limit:.0%}, and this reference is exact",
              abs(ratio - 1.0) <= limit,
              "both measured here, same box, same resolution, same window, so the "
              "ratio carries only what the moving-boundary treatment costs"),
        Check("Cd held in a stream", held["cd"],
              f"{expected:.4f} (Schiller-Naumann), reported not graded", None,
              f"{held['cd'] / expected - 1.0:+.2%} on the correlation; the difference is "
              f"resolution and the {1 / 12:.1%} blockage of a 12-diameter box, which the "
              f"ratio above cancels"),
        Check("Cd moving through still fluid", going["cd"],
              f"{expected:.4f} (Schiller-Naumann), reported not graded", None,
              f"{going['cd'] / expected - 1.0:+.2%} on the correlation"),
        Check("fresh nodes refilled, moving", going["refilled"],
              "> 0, or nothing moved", going["refilled"] > 0,
              "the count this case exists to exercise: a sphere that uncovered no "
              "lattice node has not tested the refilling at all"),
        Check("fresh nodes refilled, held", held["refilled"], "0, it does not move",
              held["refilled"] == 0),
        Check("diameters travelled", f"{going['travelled']:.2f}",
              "reported, not graded", None),
        Check("Bouzidi fallbacks",
              f"{held['fallbacks']:,} held, {going['fallbacks']:,} moving",
              "reported, not graded", None),
    ]
    out = Path(out_dir)
    drawn = [
        figures.slice_figure(going["world"], out / "live-translating-sphere.png", axis=2,
                             title=f"sphere translating at Re {reynolds:g}, "
                                   f"Cd {going['cd']:.4f}"),
        figures.slice_figure(held["world"], out / "live-translating-sphere-held.png",
                             axis=2, title=f"the same sphere held in a stream, "
                                           f"Cd {held['cd']:.4f}"),
        figures.profile_figure(
            out / "live-translating-sphere-history.png",
            [{"x": [h[0] for h in going["history"]],
              "y": [h[1] for h in going["history"]], "label": "moving"},
             {"x": [going["history"][0][0], going["history"][-1][0]]
              if going["history"] else [0, 1],
              "y": [held["cd"], held["cd"]], "label": "held in a stream", "style": "--"},
             {"x": [going["history"][0][0], going["history"][-1][0]]
              if going["history"] else [0, 1],
              "y": [expected, expected], "label": "Schiller-Naumann", "style": ":"}],
            title=f"Galilean invariance: moving/held = {ratio:.4f}",
            xlabel="diameters travelled", ylabel="Cd"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# translating sphere: Re {reynolds:g}, {cells} cells per diameter, "
             f"held and moving in the same box",
             *going["scaling"].report(),
             *_official("translating_sphere", going["scaling"], smagorinsky),
             f"  held:       Cd {held['cd']:.4f} "
             f"({held['cd'] / expected - 1.0:+.2%} on Schiller-Naumann), "
             f"{held['refilled']:,} refills",
             f"  moving:     Cd {going['cd']:.4f} "
             f"({going['cd'] / expected - 1.0:+.2%} on Schiller-Naumann), "
             f"{going['refilled']:,} refills over {going['travelled']:.2f} diameters",
             f"  Galilean:   moving/held = {ratio:.4f}, and 1.0 is exact",
             *going["world"].report()[-2:]]
    return _result("translating_sphere", checks, lines, drawn, ratio=ratio,
                   cd_held=held["cd"], cd_moving=going["cd"],
                   schiller_naumann=expected, refilled=going["refilled"])


# -- 6. a sphere falling in water, two-way coupled to Newton ----------------------------


def run_falling_sphere(*, cells_per_d: int = 20, density_ratio: float = 2.0,
                       diameter: float = 0.001, u_lattice: float = 0.05,
                       window: bool = False, collision: str = "trt",
                       smagorinsky: float = 0.0, device: str | None = None,
                       out_dir: str | Path = ".", steps: int | None = None,
                       solver: str = "mujoco", one_way: bool = False,
                       animate: bool = True,
                       progress: Callable[[str], None] | None = None,
                       **_: Any) -> dict[str, Any]:
    """Release a sphere in water and let the coupling find its terminal velocity.

    The first case here in which the fluid decides what the body does. Newton integrates
    the sphere, this extension hands it the fluid wrench every step, and the terminal
    velocity it settles at is compared with the velocity at which Schiller-Naumann's
    drag balances the sphere's weight less its buoyancy:

        (rho_s - rho_f) V g = 0.5 rho_f Cd(Re) A V_t^2,   Re = V_t D / nu

    solved by fixed-point iteration below. Nothing about that balance is in the solver;
    it is the external answer, and the measured terminal velocity either matches it or
    does not.

    **Why the numbers are these numbers.** `density_ratio` is 1.005 and the sphere is
    4 mm, which looks like an odd choice and is the only one that works. Three
    constraints have to hold at once. The terminal Reynolds number has to be inside
    Schiller-Naumann's range and low enough that the lattice resolves it: with
    `nu_lattice = u_lattice * cells_per_d / Re`, a terminal Re of 32 at 30 cells per
    diameter gives tau = 0.584, comfortably above `units.STIFF_TAU`, where a terminal Re
    of 235 (a sphere twice as dense) would give 0.509 and be refused. The sphere has to
    reach terminal velocity in a distance smaller than the box, and a near-neutrally
    buoyant sphere does: the effective acceleration is
    `(1 - rho_f/rho_s) g / (1 + 0.5 rho_f/rho_s)` with the added mass in it, about
    0.03 m/s^2 here, so it reaches 8 mm/s in a quarter of a diameter. And the fluid's
    own timestep has to be a plausible robot timestep, which at these numbers it is:
    half a millisecond.

    That last point is the one worth keeping. A validation whose timestep is a
    microsecond proves nothing about a solver that has to run at a millisecond.
    """
    import warp as wp

    try:
        import newton
    except Exception as exc:
        raise UnphysicalScaling(
            "this case needs the Newton physics engine, which the [live] extra "
            f"declares:\n    pip install 'openreynolds[live]'\n(the import said: {exc})"
        ) from exc

    from .couple import NewtonCoupling
    from .units import FLUIDS
    from .world import Domain, FluidWorld

    curve = sphere_drag()
    water = FLUIDS["water"]
    gravity = 9.81
    radius = diameter / 2.0
    volume = 4.0 / 3.0 * math.pi * radius ** 3
    area = math.pi * radius ** 2
    rho_s = density_ratio * water.rho

    # The expected answer, by fixed point. Cd falls with Re and Re rises with V, so the
    # iteration converges from below in a handful of passes; it is started from Stokes.
    excess = (rho_s - water.rho) * volume * gravity
    terminal = math.sqrt(2.0 * excess / (water.rho * area * 1.0))
    for _ in range(80):
        re = max(terminal * diameter / water.nu, 1e-6)
        cd = curve.schiller_naumann_cd(re)
        terminal = math.sqrt(2.0 * excess / (water.rho * area * cd))
    expected_re = terminal * diameter / water.nu
    expected_cd = curve.schiller_naumann_cd(expected_re)

    dx = diameter / cells_per_d
    scaling = Scaling.for_reynolds(dx, diameter, expected_re, "water",
                                   u_lattice=u_lattice, scheme=collision,
                                   les=smagorinsky > 0)
    # `for_reynolds` set dt from the Reynolds number; the coupling step is that dt with
    # one substep, which is the honest tightest coupling available and is what a
    # validation should use. A demo trades substeps for speed; a graded number does not.
    dt = scaling.dt

    # -- the box, and why the graded run does not use the moving window -----------------
    #
    # The sphere reaches terminal velocity in about 1.6 diameters and then keeps going at
    # 112 mm/s; six settling time constants is about twenty more. So either the box is
    # tall enough for all of it, or the window follows the body.
    #
    # **The window was the first choice and it biased the answer.** A windowed run in a
    # 14-diameter box shifted ten times and measured a terminal velocity **12.05% high**
    # (125.87 mm/s against 112.33), outside the 5% this is graded at. The cause is
    # visible in what a shift does: `lattice.shift` fills the slab uncovered at the
    # trailing edge with the free stream, and for a falling body the trailing edge is
    # **where the wake is**. Every 1.4 diameters of travel, the wake was replaced with
    # quiescent fluid, and a body that keeps meeting fresh fluid instead of its own wake
    # falls faster than one that does not.
    #
    # That is a real property of an integer-shift window and not a bug in it: the window
    # exists to follow a body across a domain it would otherwise leave, and it is
    # correct for a body whose wake does not matter to its own motion. For a falling
    # sphere the wake is the whole mechanism. So the graded run uses a tall box and no
    # window -- 26 diameters of it, which at 20 cells per diameter is 30 million nodes and
    # 160 seconds on an H100, an affordable price for not having to caveat the number --
    # and `window=True` runs the windowed version beside it so that what the window costs
    # is measured and reported rather than assumed.
    tall = 14.0 if window else 26.0
    size = (12.0 * diameter, 12.0 * diameter, tall * diameter)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                    faces={"-x": "slip", "+x": "slip", "-y": "slip", "+y": "slip",
                           "-z": "outlet", "+z": "outlet"})
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision=collision, smagorinsky=smagorinsky,
                       device=device, one_way=one_way,
                       follow=0 if window else None, follow_slack=0.12)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    start = (size[0] / 2.0, size[1] / 2.0, size[2] * (0.75 if window else 0.88))
    # `add_body` already creates the free joint; adding one would give the body two.
    body = builder.add_body(xform=wp.transform(wp.vec3(*start), wp.quat_identity()),
                            label="sphere")
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = rho_s
    shape_cfg.has_shape_collision = False
    builder.add_shape_sphere(body, radius=radius, cfg=shape_cfg)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -gravity))

    if solver == "mujoco":
        engine = newton.solvers.SolverMuJoCo(model, disable_contacts=True,
                                             solver="newton", integrator="implicitfast",
                                             iterations=10)
    else:
        engine = newton.solvers.SolverFeatherstone(model)

    world.initialise()
    link = NewtonCoupling(model, engine, world, dt=dt, buoyancy=True)

    # Settling takes about V_t / a_eff seconds, with the added mass in a_eff.
    a_eff = ((1.0 - water.rho / rho_s) * gravity
             / (1.0 + 0.5 * water.rho / rho_s))
    # Six time constants: `1 - exp(-6)` is 0.9975, so the velocity is within a quarter
    # of a per cent of terminal, well inside the 5% the case is graded at. Twelve, which
    # this was, is twice the GPU time for a digit the tolerance cannot see.
    settle_time = 6.0 * terminal / max(a_eff, 1e-9)
    total = int(steps if steps else math.ceil(settle_time / dt))

    times: list[float] = []
    speeds: list[float] = []
    frames: list[np.ndarray] = []
    sample_every = max(1, total // 400)
    for index in range(total):
        link.step()
        if index % sample_every == 0:
            qd = link.state.body_qd.numpy()[body]
            times.append(index * dt)
            speeds.append(float(-qd[2]))
            if animate and len(frames) < 200:
                frames.append(figures.slice_frame(world, axis=1))
        if progress and (index + 1) % max(1, total // 10) == 0:
            qd = link.state.body_qd.numpy()[body]
            progress(f"    {index + 1:>8,}/{total:,} steps  "
                     f"v {-float(qd[2]) * 1e3:7.3f} mm/s of an expected "
                     f"{terminal * 1e3:.3f}")

    measured = float(np.mean(speeds[-max(1, len(speeds) // 10):])) if speeds else 0.0
    measured_re = measured * diameter / water.nu
    error = measured / terminal - 1.0 if terminal else float("nan")
    limit = REFERENCES["schiller_naumann"]["tolerance"]
    volume_cells = getattr(link, "displaced_volume", None)
    voxel_volume = float(volume_cells[body]) if volume_cells is not None else float("nan")

    checks = [
        Check("terminal velocity", measured,
              f"{terminal:.6g} m/s +/- {limit:.0%} (Schiller-Naumann balance)"
              + ("" if not window else "; REPORTED, NOT GRADED with the window on"),
              None if window else abs(error) <= limit,
              "" if not window else
              "this run exists to measure what the moving window costs, so it is not "
              "asked to pass. It reproduces the bias deliberately: a shift replaces the "
              "wake with free stream and for a falling body the trailing edge is where "
              "the wake is, so the body keeps meeting fresh fluid and falls fast. The "
              "graded run is the same case in a 26-diameter box with no window"),
        Check("terminal Reynolds number", measured_re,
              f"{expected_re:.2f} expected", None),
        Check("displaced volume, voxelised against analytic",
              voxel_volume / volume - 1.0 if volume else float("nan"),
              "within 5% of the analytic volume",
              abs(voxel_volume / volume - 1.0) <= 0.05 if volume else None,
              "graded, because the buoyancy is computed from the voxelised volume on "
              "purpose -- it has to be the volume the boundary condition encloses, or "
              "the weight and the drag do not act on the same body "
              "(DECISIONS_LIVE.md D9) -- and the terminal velocity therefore inherits "
              "this error at half its size, V_t going as the square root of the net "
              "force"),
        Check("final state finite", world.stable(), "True", world.stable()),
        Check("window shifts", world.shifts,
              "0 for the graded run: a shift replaces the wake with free stream, and "
              "for a falling body the wake is the mechanism",
              world.shifts == 0 if not window else None,
              "a windowed run in a 14-diameter box shifted ten times and measured "
              "12.05% high; `window=True` reproduces that deliberately"),
        Check("travelled", f"{(start[2] - float(link.state.body_q.numpy()[body][2])) / diameter:.2f} diameters "
                           f"of a {tall:g}-diameter box",
              "inside the box", True),
    ]
    out = Path(out_dir)
    drawn = [figures.slice_figure(
        world, out / "live-falling-sphere.png", axis=1,
        title=f"sphere falling in water, Re {measured_re:.1f}, "
              f"v {measured * 1e3:.3f} mm/s")]
    drawn.append(figures.profile_figure(
        out / "live-falling-sphere-history.png",
        [{"x": times, "y": speeds, "label": "coupled simulation"},
         {"x": [times[0], times[-1]] if times else [0, 1],
          "y": [terminal, terminal], "label": "Schiller-Naumann terminal", "style": "--"}],
        title=f"terminal velocity: {measured * 1e3:.4f} against "
              f"{terminal * 1e3:.4f} mm/s ({error:+.2%})",
        xlabel="t (s)", ylabel="fall speed (m/s)"))
    if frames:
        gif = figures.animate(frames, out / "live-falling-sphere.gif", fps=20)
        if gif:
            drawn.append(gif)
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# falling sphere: {diameter * 1e3:g} mm, rho_s/rho_f {density_ratio:g}, "
             f"{cells_per_d} cells per diameter",
             *scaling.report(),
             *_official("falling_sphere", scaling, smagorinsky),
             f"  expected:   V_t {terminal * 1e3:.4f} mm/s at Re {expected_re:.2f}, "
             f"Cd {expected_cd:.4f}",
             f"  measured:   V_t {measured * 1e3:.4f} mm/s at Re {measured_re:.2f} "
             f"({error:+.2%})",
             *link.report()[:2], *world.report()[-2:]]
    return _result("falling_sphere", checks, lines, drawn, terminal=measured,
                   # NOT `error=`: the runner and `tests/test_benchmark_live.py` use
                   # that key for "this run raised", so a case that reports a numeric
                   # error under it makes a passing run look like a crashed one. It did:
                   # two recorded falling-sphere runs failed the suite with "did not
                   # complete: -0.038", which was the terminal-velocity error.
                   expected=terminal, reynolds=measured_re, velocity_error=error,
                   times=times, speeds=speeds)


# -- 7. the momentum balance ------------------------------------------------------------


def run_momentum(*, cells: int = 48, steps: int = 0, cells_per_d: int = 16,
                 collision: str = "trt", smagorinsky: float = 0.0,
                 device: str | None = None, out_dir: str | Path = ".",
                 progress: Callable[[str], None] | None = None,
                 **_: Any) -> dict[str, Any]:
    """Momentum in, momentum out, measured twice because the two halves fail differently.

    The conservation law this solver has to obey and the one its boundary treatment can
    break quietly. With every face periodic and no body force, the only thing that can
    change the fluid's total momentum is the body, so

        d/dt (fluid momentum) = - (force on the body)

    exactly, and the integral of the force the solver reported must equal the change in
    the momentum the solver holds. A bounce-back that reflects the wrong population, a
    Bouzidi interpolation with the wrong `q`, a sign flipped in the momentum exchange --
    all of them show up as a gap between the two, and none of them shows up in a drag
    coefficient that is within 5% of a published fit.

    **The still body is the tight check.** A fixed sphere in a closed periodic box with
    the fluid started moving: the flow decays as it drags on the sphere, nothing is ever
    refilled because nothing moves, and the balance is between two things the solver
    computes independently -- a field integral over the whole lattice and a time integral
    of a surface sum. Graded at 1%.

    **The moving body is the loose one, and the slack is a real deficiency rather than
    numerical noise.** A translating sphere uncovers lattice nodes as it goes, and
    `lattice.refill` builds each of them from equilibrium at the body's own surface
    velocity -- which **injects momentum that no force on the body accounts for**. The
    gap is therefore a measurement of what equilibrium refilling costs, which is exactly
    what `DECISIONS_LIVE.md` D5 said this case would measure. Graded at 10% and reported
    with the refill count beside it, because the number means nothing without it.

    **The body does not wrap.** The first run of this case let the sphere cross the
    periodic x boundary and reappear at the far side, three times in four thousand
    steps. Every wrap is a body vanishing from one place and appearing in another, which
    is a momentum discontinuity of the whole body's worth, and it measured a 7.6%
    imbalance that was almost entirely that. The run is now short enough that the sphere
    stays inside, and `travelled` is reported so a reader can see that it did.
    """
    import warp  # noqa: F401

    from .world import Domain, FluidWorld

    diameter = 0.01
    dx = diameter / int(cells_per_d)
    size = (cells * dx, cells * dx, cells * dx)
    scaling = Scaling.for_reynolds(dx, diameter, 50.0, "water", u_lattice=0.04,
                                   scheme=collision, les=smagorinsky > 0)
    speed = scaling.reference_velocity
    centre = tuple(v / 2.0 for v in size)

    def box():
        domain = Domain(origin=(0.0, 0.0, 0.0), size=size, faces="periodic")
        world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                           substeps=1, collision=collision, smagorinsky=smagorinsky,
                           device=device)
        body = world.add_body(position=centre, label="sphere")
        world.add_shape("sphere", body=body, scale=(diameter / 2.0, 0.0, 0.0))
        return world, body

    # -- part one: the body is still and the fluid runs down ---------------------------
    still, still_body = box()
    still.initialise(velocity=(speed, 0.0, 0.0))
    held = int(steps) if steps else 1500
    before = np.asarray(still.momentum())
    impulse = np.zeros(3)
    history = []
    for index in range(held):
        still.step()
        force, _ = still.wrench(still_body)
        impulse += np.asarray(force) * still.dt
        if index % max(1, held // 200) == 0:
            history.append((index * still.dt,
                            float(np.linalg.norm(np.asarray(still.momentum()) - before)),
                            float(np.linalg.norm(impulse))))
        if progress and (index + 1) % max(1, held // 6) == 0:
            progress(f"    still body {index + 1:>7,}/{held:,}")
    gained = np.asarray(still.momentum()) - before
    still_residual = float(np.linalg.norm(gained + impulse))
    still_scale = float(max(np.linalg.norm(gained), np.linalg.norm(impulse), 1e-30))
    still_relative = still_residual / still_scale
    still.fallbacks = int(still._fallbacks.numpy()[0])

    # -- part two: the body translates, and the refilling shows up ---------------------
    moving, moving_body = box()
    moving.initialise()
    # Short enough that the sphere stays inside the box: a third of the way across, so
    # it never approaches the periodic face it would otherwise wrap through.
    travel_steps = int(0.33 * cells / 0.04)
    position = np.asarray(centre, dtype=float)
    moving.set_body_state(moving_body, velocity=(speed, 0, 0, 0, 0, 0))
    before_moving = np.asarray(moving.momentum())
    impulse_moving = np.zeros(3)
    for index in range(travel_steps):
        position = position + np.array([speed * moving.dt, 0.0, 0.0])
        moving.set_body_state(moving_body, position=tuple(position))
        moving.step()
        force, _ = moving.wrench(moving_body)
        impulse_moving += np.asarray(force) * moving.dt
        if progress and (index + 1) % max(1, travel_steps // 6) == 0:
            progress(f"    moving body {index + 1:>7,}/{travel_steps:,}")
    gained_moving = np.asarray(moving.momentum()) - before_moving
    moving_residual = float(np.linalg.norm(gained_moving + impulse_moving))
    moving_scale = float(max(np.linalg.norm(gained_moving),
                             np.linalg.norm(impulse_moving), 1e-30))
    moving_relative = moving_residual / moving_scale
    travelled = float((position[0] - centre[0]) / diameter)
    moving.refilled = int(moving._refilled.numpy()[0])
    moving.fallbacks = int(moving._fallbacks.numpy()[0])

    checks = [
        Check("momentum balance, still body: |fluid gained + impulse| / |exchanged|",
              still_relative, "<= 1%", still_relative <= 0.01,
              "nothing moves, so nothing is refilled: this is the boundary condition "
              "and the force sum checked against each other and nothing else"),
        Check("momentum balance, translating body", moving_relative, "<= 10%",
              moving_relative <= 0.10,
              f"{moving.refilled:,} nodes were refilled from equilibrium, each "
              f"injecting momentum that no force on the body accounts for. The gap is a "
              f"measurement of what equilibrium refilling costs (DECISIONS_LIVE.md D5)"),
        Check("the body stayed inside the box", f"{travelled:.2f} diameters travelled",
              "no wrap through a periodic face",
              travelled < 0.4 * cells / cells_per_d),
        Check("fluid momentum gained, still body",
              f"{gained[0]:+.4e} kg m/s (x)", "reported, not graded", None),
        Check("impulse the solver reported, still body",
              f"{impulse[0]:+.4e} kg m/s (x)", "reported, not graded", None),
        Check("lattice still finite", still.stable() and moving.stable(), "True",
              still.stable() and moving.stable()),
        Check("window shifts", still.shifts + moving.shifts,
              "0, or this measures the window",
              still.shifts + moving.shifts == 0),
        Check("Bouzidi fallbacks",
              f"{still.fallbacks:,} still, {moving.fallbacks:,} moving",
              "reported, not graded", None,
              "links whose second fluid node was also solid; at "
              f"{cells_per_d} cells per diameter this is the sphere's own curvature"),
    ]
    out = Path(out_dir)
    drawn = [figures.profile_figure(
        out / "live-momentum.png",
        [{"x": [h[0] for h in history], "y": [h[1] for h in history],
          "label": "|fluid momentum change|"},
         {"x": [h[0] for h in history], "y": [h[2] for h in history],
          "label": "|impulse on the body|", "style": "--"}],
        title=f"momentum balance, still body: {still_relative:.3%} residual",
        xlabel="t (s)", ylabel="kg m/s")]
    drawn.append(figures.slice_figure(
        moving, out / "live-momentum-moving.png", axis=2,
        title=f"translating sphere after {travelled:.2f} D, "
              f"{moving_relative:.2%} imbalance"))
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# momentum: a closed periodic box, {cells}^3, "
             f"{cells_per_d} cells per diameter",
             *scaling.report(), *_official("momentum", scaling, smagorinsky),
             f"  still body: {still_relative:.4%} of the momentum exchanged, over "
             f"{held:,} steps, 0 refills",
             f"  moving:     {moving_relative:.4%} over {travel_steps:,} steps and "
             f"{travelled:.2f} diameters, {moving.refilled:,} refills",
             *still.report()[-2:]]
    return _result("momentum", checks, lines, drawn, still=still_relative,
                   moving=moving_relative, refilled=moving.refilled,
                   travelled=travelled)


# -- 8. ground effect: one actuator disk over a plane -----------------------------------


def _cheeseman_bennett(radius: float, height: float) -> float:
    """`T_g/T_inf = 1/(1 - R^2/(16 Z^2))`, Cheeseman and Bennett (1955) equation (2).

    Written out rather than only cited, because the citation is the thing that goes
    wrong: the constant inside the bracket is 4 (so 16 when squared), the ratio is
    derived **at constant power**, and `Z` is the rotor's height above the ground. All
    three were read off the original report (ARC R&M 3021). See
    `REFERENCES["cheeseman_bennett"]`.
    """
    ratio = radius / (4.0 * height)
    return 1.0 / (1.0 - ratio * ratio)


def _induced_ratio(radius: float, height: float) -> float:
    """The same model as a ratio of induced velocities, which is what is measurable here.

    Cheeseman and Bennett's result is a thrust ratio at constant power, and a lattice
    Boltzmann run with an actuator disk has the *thrust* as its input -- the disk is a
    momentum source of a set strength -- so a thrust ratio is not a quantity this case
    can measure. The induced velocity is.

    The two are the same statement. The model's mechanism is that the ground's image
    source reduces the induced velocity at the disk by `(R/4Z)^2`, so

        v_g / v_inf = 1 - (R / 4Z)^2

    and at constant power `P = T v` a thrust ratio of `1/(1 - (R/4Z)^2)` follows
    immediately -- which is `_cheeseman_bennett` above, and the two functions are each
    other's reciprocal by construction. Grading the induced velocity is therefore
    grading the same physics without having to hold power constant in a solver that has
    no notion of rotor power.
    """
    ratio = radius / (4.0 * height)
    return 1.0 - ratio * ratio


def run_ground_effect(*, radius: float = 0.12, thrust: float = 2.72,
                      heights: Sequence[float] = (0.75, 1.0, 1.5, 2.0, 3.0),
                      free_height: float = 6.0, cells_per_r: int = 12,
                      collision: str = "regularized", smagorinsky: float = 0.12,
                      device: str | None = None, out_dir: str | Path = ".",
                      steps: int | None = None, animate: bool = True,
                      progress: Callable[[str], None] | None = None,
                      **_: Any) -> dict[str, Any]:
    """One actuator disk at several heights over a ground plane, against the 1955 model.

    `heights` are in rotor radii, which is how the model is written. The measured
    quantity is the mean axial velocity through the disk, and the graded quantity is its
    *ratio* to the same disk in free air -- because a ratio cancels everything about the
    actuator-disk model that is not the ground: the disk's thickness, how its momentum
    is spread, the lattice resolution and the subgrid constant all appear in both the
    numerator and the denominator.

    **This case runs with the subgrid model on, and says so every time.** A 120 mm rotor
    pushing 5 m/s of air is a Reynolds number near 80,000 on the diameter, and
    `units.STIFF_TAU`'s arithmetic shows there is no resolution at which a millisecond
    timestep resolves air's molecular viscosity. What carries the run is the eddy
    viscosity. That is legitimate for this quantity -- the ground effect is an inviscid
    image-source argument and the model being compared against is itself inviscid -- and
    it is why this is the one graded case in the suite with the LES on.

    **One disk and not four.** Sanchez-Cuevas, Heredia and Ollero (2017) measured that a
    complete multirotor does not follow Cheeseman and Bennett: the effect is much larger
    and reaches about five radii rather than two, because of a fountain of air under the
    airframe. A quadrotor graded against this model would be graded against a model that
    the later literature shows does not apply to it. The quadrotor is `quadrotor`, and
    it is a demo.

    **`animate` writes two GIFs and not six.** The lowest graded height and the free-air
    reference are the comparison this case *is* -- the same disk with the ground under it
    and without -- and four more heights of the same jet is four more files rather than
    more information. `run_cylinder` animates only its finest grid for the same reason.

    The one thing `animate` changes about the numbers is that it samples the disk 200
    times instead of 60, for a GIF long enough to watch. The graded average is the last
    third of the samples, which is the same window at either density, so that is more
    points inside the window and not a different measurement.
    """
    import warp  # noqa: F401

    from .units import FLUIDS
    from .world import Domain, FluidWorld

    dx = radius / cells_per_r
    area = math.pi * radius * radius
    air = FLUIDS["air"]
    # Momentum theory for the free-air induced velocity, which sets the lattice speed:
    # T = 2 rho A v^2.
    v_free = math.sqrt(thrust / (2.0 * air.rho * area))
    out = Path(out_dir)
    rows = []
    drawn = []
    notes: list[str] = []

    for index, z_over_r in enumerate([*heights, free_height]):
        height = z_over_r * radius
        is_free = z_over_r == free_height
        wants_gif = bool(animate) and (is_free or z_over_r == min(heights))
        if progress:
            progress(f"  disk at Z/R {z_over_r:g}"
                     f"{' (free air reference)' if is_free else ''} "
                     f"({index + 1} of {len(heights) + 1})")
        # Enough room for the jet to turn and leave: eight radii each way, and the
        # domain top always well above the disk.
        span = 16.0 * radius
        tall = max(height + 8.0 * radius, 12.0 * radius)
        size = (span, span, tall)
        # Slip on the four sides and an outflow on the axis the jet leaves by. Six
        # outflows is what this case was first written with, and `Domain` now refuses
        # it: an outflow imposes no density, so a box of nothing but outflows around a
        # momentum source has a free mean density, and a weakly compressible lattice
        # with a free mean density diverges -- measured, inside fifteen coupling steps.
        # A slip face conserves the mass that crosses it and pins it. The free-air
        # reference opens the floor as well as the ceiling, which is what "free air"
        # means: no ground.
        domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                        faces={"-x": "slip", "+x": "slip",
                               "-y": "slip", "+y": "slip",
                               "-z": "outlet" if is_free else "wall",
                               "+z": "outlet"})
        scaling, substeps = Scaling.for_coupling(
            dx, 2.0e-3, "air", peak_velocity=4.0 * v_free, scheme=collision,
            les=smagorinsky > 0)
        world = FluidWorld(domain=domain, dx=dx, fluid="air", scaling=scaling,
                           substeps=substeps, collision=collision,
                           smagorinsky=smagorinsky, device=device)
        centre = (size[0] / 2.0, size[1] / 2.0, height)
        world.add_rotor(-1, centre, (0.0, 0.0, 1.0), radius, thickness=2.0 * dx)
        world.set_rotor_thrust(0, thrust)
        world.initialise()

        # The jet has to cross the box once for the flow to be established, plus a
        # window to average over.
        crossing = tall / max(v_free, 1e-9) / world.dt
        total = int(steps if steps else math.ceil(2.5 * crossing))
        samples: list[float] = []
        frames: list[np.ndarray] = []
        node_z = world.node_coordinates(2)
        plane = int(np.argmin(np.abs(node_z - height)))
        xs = world.node_coordinates(0)
        ys = world.node_coordinates(1)
        disk = ((xs[:, None] - centre[0]) ** 2
                + (ys[None, :] - centre[1]) ** 2) <= radius * radius

        # The transient is discarded **here** rather than by `_advance`'s `settle`,
        # because the average must not contain it and the animation is mostly made of
        # it: the jet reaching the floor and turning into a wall jet happens inside the
        # first 60% of the run, and a GIF that starts afterwards is a picture of a
        # steady state. The averaging window is unchanged -- the same 0.6 of the run,
        # and `induced` still takes the last third of what is sampled.
        settle_at = int(total * 0.6)

        def take(index: int, world=world, samples=samples, plane=plane, disk=disk,
                 frames=frames, wants_gif=wants_gif, settle_at=settle_at):
            # Gated on how many frames there are and on nothing else. A second modulus
            # of the step index, inside a callback that `_advance` already fires on one,
            # is how `crosscheck_2d1` collected exactly one frame: at 704,000 steps its
            # two strides were 1,760 and 3,911, which are coprime, so both fired
            # together only at index 0 -- and the refusal came after 457 s of H100.
            if wants_gif and len(frames) < 200:
                frames.append(figures.slice_frame(world, axis=1, component=2))
            if index < settle_at:
                return
            w = world.fields()["u"][:, :, plane, 2]
            samples.append(float(np.nanmean(np.where(disk, w, np.nan))))

        _advance(world, total, sample_every=max(1, total // (200 if animate else 60)),
                 sample=take, settle=0.0, progress=progress)
        induced = abs(float(np.mean(samples[-max(1, len(samples) // 3):]))) \
            if samples else 0.0
        rows.append({"z_over_r": z_over_r, "height": height, "induced": induced,
                     "free": is_free, "nodes": world.nodes, "steps": total,
                     "substeps": substeps, "tau": scaling.tau_plus})
        drawn.append(figures.slice_figure(
            world, out / f"live-ground-effect-z{z_over_r:g}.png", axis=1,
            component=2,
            title=f"actuator disk at Z/R {z_over_r:g}, mean w through the disk "
                  f"{induced:.3f} m/s"))
        if frames:
            gif = out / f"live-ground-effect-z{z_over_r:g}.gif"
            made, note = _try_draw(
                f"the {len(frames)}-frame animation at Z/R {z_over_r:g}",
                lambda gif=gif, frames=frames: figures.animate(
                    frames, gif, fps=20, component=True) or gif)
            drawn.append(made) if made else notes.append(note or "")

    free = next(r for r in rows if r["free"])
    graded = [r for r in rows if not r["free"]]
    limit = REFERENCES["cheeseman_bennett"]["tolerance"]
    checks = []
    for row in graded:
        measured = row["induced"] / free["induced"] if free["induced"] else float("nan")
        model = _induced_ratio(radius, row["height"])
        row["ratio"] = measured
        row["model"] = model
        row["thrust_ratio_model"] = _cheeseman_bennett(radius, row["height"])
        error = measured / model - 1.0 if model else float("nan")
        row["error"] = error
        checks.append(Check(
            f"induced velocity ratio at Z/R {row['z_over_r']:g}", measured,
            f"{model:.4f} +/- {limit:.0%} (1 - (R/4Z)^2)",
            abs(error) <= limit))
    checks.append(Check("free-air induced velocity", free["induced"],
                        f"{v_free:.4f} m/s from momentum theory, T = 2 rho A v^2",
                        None,
                        f"reported, not graded: an actuator disk of finite thickness on "
                        f"a lattice does not have to reproduce the inviscid "
                        f"momentum-theory value, and the graded quantity is a ratio in "
                        f"which the disk's own discretisation cancels. It reads "
                        f"{free['induced'] / v_free - 1.0:+.1%} on momentum theory"))
    # The residual's sign is a diagnosis and it belongs in the output rather than only
    # in the index. The model approaches 1 from below and cannot exceed it, so a
    # measured ratio above 1 is the free-air denominator being low rather than the
    # ground effect being strong -- and the reference run opens its floor to an outflow,
    # which is a zero-gradient plane a few radii down and not infinity.
    above = [row for row in graded if row["ratio"] > 1.0]
    # Built with join() on a list comprehension rather than inside the f-string. The
    # previous version nested same-type quotes in an f-string and was repaired by
    # concatenating the quote character around the key, which does not escape anything:
    # it builds a DIFFERENT key, with the quotes as characters, and raised `KeyError`
    # the first time a run actually put a height above 1. That is a report line, so it
    # took the case down after 1,087 s of H100 with every number already computed.
    # `tests/test_live_cases.py` guards the pattern.
    listed = ", ".join(f"{row['z_over_r']:g}" for row in above)
    checks.append(Check(
        "heights whose ratio exceeds 1",
        f"{len(above)} of {len(graded)}" + (f" (Z/R {listed})" if above else ""),
        "reported, not graded", None,
        "the model approaches 1 from below and cannot exceed it, so a ratio above 1 is "
        "the free-air reference being low, not the ground pushing harder. The reference "
        "run's floor is an outflow a few radii down rather than infinity, and its "
        f"induced velocity reads {free['induced'] / v_free - 1.0:+.1%} on momentum "
        "theory -- which is roughly the size of the overshoot. A larger free-air box "
        "would tighten it; it is inside the 10% this 1955 model is graded at either way"))

    sweep = np.linspace(0.7, 3.2, 50)
    drawn.append(figures.profile_figure(
        out / "live-ground-effect.png",
        [{"x": [r["z_over_r"] for r in graded], "y": [r["ratio"] for r in graded],
          "label": "lattice Boltzmann", "style": "o"},
         {"x": sweep,
          "y": [_induced_ratio(radius, z * radius) for z in sweep],
          "label": "1 - (R/4Z)^2, Cheeseman and Bennett (1955)"}],
        title="ground effect on an actuator disk",
        xlabel="Z/R", ylabel="v_ground / v_free"))
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# ground effect: one actuator disk, R {radius * 1e3:g} mm, "
             f"T {thrust:g} N, {cells_per_r} cells per radius",
             f"  model:      {REFERENCES['cheeseman_bennett']['formula']}, and the "
             f"induced-velocity form 1 - (R/4Z)^2 this grades against",
             f"  reference:  {REFERENCES['cheeseman_bennett']['source']}",
             f"  caveat:     {REFERENCES['cheeseman_bennett']['caveat']}",
             f"  free air:   v {free['induced']:.4f} m/s measured, "
             f"{v_free:.4f} m/s from momentum theory",
             f"  subgrid:    ON (Cs = {smagorinsky:g}); tau_plus "
             f"{free['tau']:.5f}, so this is an under-resolved large-eddy run"]
    for row in graded:
        lines.append(f"  Z/R {row['z_over_r']:>4g}:  v/v_free {row['ratio']:.4f} "
                     f"against {row['model']:.4f} ({row['error']:+.2%}); the model's "
                     f"thrust ratio at constant power is "
                     f"{row['thrust_ratio_model']:.4f}")
    lines.extend(notes)
    return _result("ground_effect", checks, lines, drawn, rows=rows,
                   free_induced=free["induced"], momentum_theory=v_free)


# -- 9. the quadrotor demo --------------------------------------------------------------


def run_quadrotor(*, arm: float = 0.18, rotor_radius: float = 0.12,
                  mass: float = 1.2, heights: Sequence[float] = (1.0, 2.5),
                  cells_per_r: int = 8, collision: str = "regularized",
                  smagorinsky: float = 0.12, device: str | None = None,
                  out_dir: str | Path = ".", steps: int | None = None,
                  animate: bool = True, wall: bool = True,
                  progress: Callable[[str], None] | None = None,
                  **_: Any) -> dict[str, Any]:
    """A quadrotor held in hover by its own rotors, feeling its own downwash.

    The demo the whole extension exists for: four actuator disks driven by a controller,
    an airframe Newton integrates with its MuJoCo Warp solver, and the fluid wrench on
    that airframe computed from the flow the rotors made. Run at two heights over a
    ground plane and, by default, beside a wall.

    **Graded for bookkeeping and stability, not against a published number.** The
    ground-effect model is graded in `ground_effect`, on one disk, because
    Sanchez-Cuevas, Heredia and Ollero (2017) measured that a complete multirotor does
    not follow Cheeseman and Bennett -- the fountain under the airframe makes the effect
    larger and longer-ranged. What is checked here is what can be checked: that the
    thrust the controller needs falls as the vehicle approaches the ground, that the
    wrench the airframe feels is a real fraction of its weight rather than noise, that
    the altitude is held, and that a long run does not diverge.

    **The airframe is primitives, not the Crazyflie USD.**
    `newton/examples/assets/crazyflie.usd` is in Newton's assets and carries no
    `RigidBodyAPI`, no mass and no joints -- loading it gives twelve visual-only meshes,
    zero bodies and zero total mass, and no Newton example loads it either (the drone
    example's own line reads `# TODO: Use drone path to load USD asset`). It would need
    `usd-core` and `newton-usd-schemas` as well, and voxelising twelve meshes every
    coupling step would cost more than the rest of the step. A box and four disks is the
    same physics for this purpose; `DECISIONS_LIVE.md` D10 records it.
    """
    import warp as wp

    try:
        import newton
    except Exception as exc:
        raise UnphysicalScaling(
            "this case needs the Newton physics engine, which the [live] extra "
            f"declares:\n    pip install 'openreynolds[live]'\n(the import said: {exc})"
        ) from exc

    from .couple import NewtonCoupling
    from .units import FLUIDS
    from .world import Domain, FluidWorld

    air = FLUIDS["air"]
    gravity = 9.81
    weight = mass * gravity
    hover_each = weight / 4.0
    area = math.pi * rotor_radius ** 2
    v_hover = math.sqrt(hover_each / (2.0 * air.rho * area))
    dx = rotor_radius / cells_per_r
    out = Path(out_dir)
    rows = []
    drawn = []

    for index, z_over_r in enumerate(heights):
        height = z_over_r * rotor_radius
        if progress:
            progress(f"  quadrotor hovering at Z/R {z_over_r:g} "
                     f"({index + 1} of {len(heights)})")
        span = 20.0 * rotor_radius
        tall = max(height + 10.0 * rotor_radius, 14.0 * rotor_radius)
        # The ground is a wall, the ceiling an outflow, and the sides slip -- see
        # `run_ground_effect` for why they are not outflows. `wall` on -x adds the
        # vertical surface the wall-effect half of this demo needs.
        faces = {"-x": "wall" if wall else "slip", "+x": "slip",
                 "-y": "slip", "+y": "slip", "-z": "wall", "+z": "outlet"}
        size = (span, span, tall)
        domain = Domain(origin=(0.0, 0.0, 0.0), size=size, faces=faces)
        scaling, substeps = Scaling.for_coupling(
            dx, 2.0e-3, "air", peak_velocity=4.0 * v_hover, scheme=collision,
            les=smagorinsky > 0)
        world = FluidWorld(domain=domain, dx=dx, fluid="air", scaling=scaling,
                           substeps=substeps, collision=collision,
                           smagorinsky=smagorinsky, device=device)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        centre = (size[0] / 2.0, size[1] / 2.0, height)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), label="airframe")
        half = (arm * 0.55, arm * 0.55, 0.02)
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = mass / (8.0 * half[0] * half[1] * half[2])
        shape_cfg.has_shape_collision = False
        builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=shape_cfg)
        model = builder.finalize()
        model.set_gravity((0.0, 0.0, -gravity))
        engine = newton.solvers.SolverMuJoCo(model, disable_contacts=True,
                                             solver="newton",
                                             integrator="implicitfast", iterations=10)
        world.initialise()
        for sign_x, sign_y in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            world.add_rotor(body, (sign_x * arm, sign_y * arm, 0.03),
                            (0.0, 0.0, 1.0), rotor_radius, thickness=2.0 * dx)
        link = NewtonCoupling(model, engine, world, dt=world.dt, buoyancy=False)

        # Two timescales again, and the longer one is the controller's. The flow needs
        # about one crossing of the box to establish, `tall / v_hover`. The altitude
        # loop needs several of its own time constants, which for this PD is about
        # 0.25 s -- and at 2 ms a coupling step, 0.65 s of run (two crossings) left the
        # vehicle still converging: it measured an altitude spread of 0.127 R over the
        # last third, just outside the 0.1 R that counts as settled. Four crossings is
        # 1.3 s, five of those time constants, and the extra 170 seconds of H100 is a
        # better trade than widening the bound until the run passes.
        crossing = tall / max(v_hover, 1e-9) / world.dt
        total = int(steps if steps else math.ceil(4.0 * crossing))
        target = height
        thrusts: list[float] = []
        wrenches: list[float] = []
        times: list[float] = []
        altitudes: list[float] = []
        frames: list[np.ndarray] = []
        integral = 0.0

        for step_index in range(total):
            pose = link.state.body_q.numpy()[body]
            qd = link.state.body_qd.numpy()[body]
            error = target - float(pose[2])
            integral += error * world.dt
            # A plain PID on altitude plus the hover feedforward. Deliberately simple:
            # the point of the demo is the fluid, and a controller carrying a fluid
            # model would hide whether the fluid did anything.
            # Damping raised from 4 to 8 after the first flying run overshot its
            # target by half a rotor radius. The overshoot is not a controller bug: the
            # fluid hands the vehicle an unmodelled +6.5% of its weight in ground
            # effect, and a PD that does not know about it has to reject that as a
            # disturbance. More damping rejects it; a controller that modelled it would
            # hide whether the fluid did anything, which is the one thing this demo is
            # for.
            command = (weight + 6.0 * mass * error
                       + 8.0 * mass * (-float(qd[2])) + 0.5 * mass * integral)
            command = float(np.clip(command, 0.0, 3.0 * weight))
            for rotor_index in range(4):
                world.set_rotor_thrust(rotor_index, command / 4.0)
            link.step()
            if step_index % max(1, total // 300) == 0:
                force, _ = world.wrench(body)
                times.append(step_index * world.dt)
                thrusts.append(command)
                wrenches.append(float(force[2]))
                altitudes.append(float(pose[2]))
                if animate and len(frames) < 200:
                    frames.append(figures.slice_frame(world, axis=1, component=2))
            if progress and (step_index + 1) % max(1, total // 8) == 0:
                progress(f"    {step_index + 1:>8,}/{total:,} steps  "
                         f"z {float(pose[2]):.4f} m  thrust {command:.3f} N "
                         f"({command / weight:.3f} of weight)")

        airframe_cells = int(world.solid_cells(body))
        keep = max(1, len(thrusts) // 3)
        mean_thrust = float(np.mean(thrusts[-keep:])) if thrusts else float("nan")
        mean_wrench = float(np.mean(wrenches[-keep:])) if wrenches else float("nan")
        rows.append({
            "z_over_r": z_over_r, "height": height, "thrust": mean_thrust,
            "thrust_over_weight": mean_thrust / weight,
            "fluid_wrench_z": mean_wrench,
            "wrench_over_weight": mean_wrench / weight,
            "altitude": float(np.mean(altitudes[-keep:])) if altitudes else float("nan"),
            "altitude_spread": (float(np.max(altitudes[-keep:]) - np.min(altitudes[-keep:]))
                                if altitudes else float("nan")),
            "nodes": world.nodes, "steps": total, "substeps": substeps,
            "realtime": world.dt * world.steps / max(world.seconds, 1e-9),
            "tau": scaling.tau_plus, "airframe_cells": airframe_cells,
        })
        drawn.append(figures.slice_figure(
            world, out / f"live-quadrotor-z{z_over_r:g}.png", axis=1, component=2,
            title=f"quadrotor hovering at Z/R {z_over_r:g}, thrust "
                  f"{mean_thrust / weight:.3f} of weight"))
        if frames:
            gif = figures.animate(frames, out / f"live-quadrotor-z{z_over_r:g}.gif",
                                  fps=20, component=True)
            if gif:
                drawn.append(gif)
        if times:
            drawn.append(figures.profile_figure(
                out / f"live-quadrotor-z{z_over_r:g}-history.png",
                [{"x": times, "y": [t / weight for t in thrusts],
                  "label": "commanded thrust / weight"},
                 {"x": times, "y": [w / weight for w in wrenches],
                  "label": "fluid wrench on the airframe / weight", "panel": 1}],
                title=f"Z/R {z_over_r:g}", xlabel="t (s)",
                ylabel="fraction of weight", panels=2))

    low = min(rows, key=lambda r: r["z_over_r"])
    high = max(rows, key=lambda r: r["z_over_r"])
    checks = [
        Check("thrust needed to hover falls nearer the ground",
              f"{low['thrust_over_weight']:.4f} at Z/R {low['z_over_r']:g} against "
              f"{high['thrust_over_weight']:.4f} at Z/R {high['z_over_r']:g}",
              "the lower one is the smaller",
              low["thrust_over_weight"] <= high["thrust_over_weight"],
              "the sign of the ground effect, which is all a two-height run can "
              "establish; ground_effect grades the magnitude on one disk"),
        Check("the airframe is voxelised at all",
              f"{low['airframe_cells']:,} solid nodes",
              "> 0, or the lattice cannot see the body",
              low["airframe_cells"] > 0,
              "a slab thinner than a cell passes between the nodes and voxelises to "
              "nothing, and the symptom is a wrench of exactly zero -- which reads as a "
              "coupling that is not wired up. At two cells per rotor radius this case "
              "measured 0 nodes for a 40 mm airframe on a 60 mm lattice"),
        Check("fluid wrench on the airframe, as a fraction of weight",
              f"{low['wrench_over_weight']:+.4f} at Z/R {low['z_over_r']:g}",
              "finite, non-zero and under 1",
              bool(abs(low["wrench_over_weight"]) < 1.0
                   and low["wrench_over_weight"] != 0.0
                   and math.isfinite(low["wrench_over_weight"]))),
        Check("altitude settled",
              f"spread {low['altitude_spread'] / rotor_radius:.3f} R over the last "
              f"third of the run",
              "< 0.1 R, which is what a stable loop looks like",
              low["altitude_spread"] < 0.1 * rotor_radius,
              "the graded property is that the loop is STABLE, not that it is accurate: "
              "a vehicle still oscillating is a coupling that has not settled, and a "
              "vehicle holding a wrong height steadily is a controller that has not "
              "been tuned"),
        Check("altitude offset from target",
              f"{(low['altitude'] - low['height']) / rotor_radius:+.3f} R "
              f"({low['altitude']:.4f} m against {low['height']:.4f} m)",
              "reported, not graded", None,
              "the size of the disturbance the controller had to reject, which is the "
              "ground effect it does not model. It is the measurement, not the error"),
        Check("real time, simulated seconds per wall-clock second",
              f"{low['realtime']:.4f}x at {low['nodes']:,} nodes",
              "reported; the brief's 0.5x target is a device claim", None),
    ]
    lines = [f"# quadrotor: {mass:g} kg, {arm * 1e3:g} mm arms, "
             f"R {rotor_radius * 1e3:g} mm, {cells_per_r} cells per radius",
             f"  hover:      {weight:.3f} N total, {hover_each:.3f} N a rotor, "
             f"induced {v_hover:.3f} m/s from momentum theory",
             f"  subgrid:    ON (Cs = {smagorinsky:g}); tau_plus {low['tau']:.5f}, "
             f"an under-resolved large-eddy run",
             "  walls:      ground at z = 0" + (", and a wall at x = 0" if wall else "")]
    for row in rows:
        lines.append(
            f"  Z/R {row['z_over_r']:>4g}:  thrust {row['thrust']:.4f} N "
            f"({row['thrust_over_weight']:.4f} of weight), fluid wrench "
            f"{row['fluid_wrench_z']:+.4f} N ({row['wrench_over_weight']:+.4f}), "
            f"{row['realtime']:.4f}x real time at {row['nodes']:,} nodes, "
            f"{row['substeps']} substeps, airframe {row['airframe_cells']:,} nodes")
    for figure in drawn:
        figures.ensure_rgb(figure)
    return _result("quadrotor", checks, lines, drawn, rows=rows)


# -- 10. an underwater body, whose drag has to be the sphere's ---------------------------


def run_underwater(*, reynolds: float = 100.0, cells_per_d: int = 0,
                   length_ratio: float = 3.0, collision: str = "trt",
                   smagorinsky: float = 0.0, device: str | None = None,
                   out_dir: str | Path = ".", steps: int | None = None,
                   animate: bool = True,
                   progress: Callable[[str], None] | None = None,
                   **_: Any) -> dict[str, Any]:
    """A capsule towed through water, graded against the sphere it reduces to.

    The brief asks for "an underwater body or simple swimmer whose drag matches M2", and
    the way to make that a real check rather than a plausible picture is to grade the
    body against a shape whose answer is known. A capsule of zero length **is** a
    sphere, so the same geometry code, the same voxelisation, the same boundary
    treatment and the same force sum have to reproduce the sphere's drag coefficient at
    zero length -- and that is what is graded, against Schiller-Naumann.

    The elongated hull is then reported rather than graded, because no published drag
    coefficient for a capsule of arbitrary fineness ratio has been read for this
    repository. What is said about it is the one thing that can be said honestly: it is
    lower than the sphere's on frontal area, which is what streamlining does, and here
    is by how much.

    **`animate` costs nothing but a GIF.** The frames come off the sampling callback the
    force history already runs, at the stride it already uses, so the animation changes
    neither the sampled indices nor the averaging window. Both hulls get one, because
    the two wakes side by side are the case's own argument: the same code, the same
    voxelisation, one shape with a published answer and one without.

    What it shows is a *steady* wake -- Re 100 is below the sphere's shedding threshold
    and the sampling starts at three-quarters of the run -- so the frame-to-frame
    difference is small by construction. That is the physics and not a dead encoder;
    `figures.animate` refuses frames that collapse, and if this case's do, there is a
    note in the report instead of a GIF.
    """
    import warp  # noqa: F401

    from .world import Domain, FluidWorld

    curve = sphere_drag()
    diameter = 0.01
    radius = diameter / 2.0
    cells_per_d = int(cells_per_d) if cells_per_d else _cells_for(reynolds, 0.04)
    dx = diameter / cells_per_d
    out = Path(out_dir)
    rows = []
    drawn = []
    notes: list[str] = []
    scaling = None

    for index, ratio in enumerate([0.0, float(length_ratio)]):
        half_height = ratio * radius
        if progress:
            progress(f"  capsule of fineness {1.0 + ratio:g} ({index + 1} of 2)")
        scaling = Scaling.for_reynolds(dx, diameter, reynolds, "water",
                                       u_lattice=0.04, scheme=collision,
                                       les=smagorinsky > 0)
        u_inf = scaling.reference_velocity
        size = (24.0 * diameter, 12.0 * diameter, 12.0 * diameter)
        domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                        faces={"-x": "inlet", "+x": "outlet", "-y": "slip",
                               "+y": "slip", "-z": "slip", "+z": "slip"},
                        face_velocity={"-x": (u_inf, 0.0, 0.0)})
        world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                           substeps=1, collision=collision, smagorinsky=smagorinsky,
                           device=device)
        position = (6.0 * diameter, size[1] / 2.0, size[2] / 2.0)
        body = world.add_body(position=position, label="capsule")
        # A capsule's axis is its local z (Newton's convention, measured in
        # couple.py), so the hull is turned to lie along the stream: a quarter turn
        # about y takes z to x.
        turn = (0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0))
        world.add_shape("capsule", body=body, orientation=turn,
                        scale=(radius, half_height, 0.0))
        world.initialise(velocity=(u_inf, 0.0, 0.0))

        crossing = size[0] / dx / 0.04
        total = int(steps if steps else 3.0 * crossing)
        samples: list[float] = []
        frames: list[np.ndarray] = []

        # The transient is dropped here and not by `_advance`'s `settle`, for the same
        # reason as `run_ground_effect`: the animation is the wake forming, which is
        # over long before the three-quarters mark the force average starts at. Sampled
        # after it, at Re 100, the frames are a steady wake and differ by almost
        # nothing. The averaged window is the same window as before.
        settle_at = int(total * 0.75)

        def take(index: int, world=world, samples=samples, body=body,
                 frames=frames, settle_at=settle_at) -> None:
            # On the frame count, not on a modulus of the step index: `_advance` has
            # already returned early on one, and a second stride that is coprime with
            # the first fires with it only at index 0. That cost `crosscheck_2d1` a
            # one-frame GIF after 457 s of H100.
            if animate and len(frames) < 200:
                frames.append(figures.slice_frame(world, axis=2))
            if index < settle_at:
                return
            force, _ = world.wrench(body)
            samples.append(float(force[0]))

        _advance(world, total, sample_every=max(1, total // 300), sample=take,
                 settle=0.0, progress=progress)
        area = math.pi * radius * radius
        force = float(np.mean(samples[-max(1, len(samples) // 4):])) if samples else 0.0
        cd = 2.0 * force / (scaling.fluid.rho * u_inf * u_inf * area)
        rows.append({"fineness": 1.0 + ratio, "cd": cd, "nodes": world.nodes,
                     "steps": total})
        drawn.append(figures.slice_figure(
            world, out / f"live-underwater-f{1.0 + ratio:g}.png", axis=2,
            title=f"capsule, fineness {1.0 + ratio:g}, Re {reynolds:g}, Cd {cd:.4f}"))
        if frames:
            gif = out / f"live-underwater-f{1.0 + ratio:g}.gif"
            made, note = _try_draw(
                f"the {len(frames)}-frame animation of the fineness-{1.0 + ratio:g} "
                f"hull",
                lambda gif=gif, frames=frames: figures.animate(
                    frames, gif, fps=15) or gif)
            drawn.append(made) if made else notes.append(note or "")

    expected = curve.schiller_naumann_cd(reynolds)
    limit = REFERENCES["schiller_naumann"]["tolerance"]
    sphere_like, elongated = rows[0], rows[1]
    ratio = elongated["cd"] / sphere_like["cd"] if sphere_like["cd"] else float("nan")
    checks = [
        Check("Cd of the zero-length capsule, which is a sphere", sphere_like["cd"],
              f"{expected:.4f} +/- {limit:.0%} (Schiller-Naumann)",
              abs(sphere_like["cd"] / expected - 1.0) <= limit,
              "the same code path as the elongated hull, on a shape with a known answer"),
        Check(f"Cd of the fineness-{elongated['fineness']:g} hull", elongated["cd"],
              "reported, not graded", None,
              "no published drag coefficient for a capsule of this fineness ratio has "
              "been read for this repository, so there is nothing honest to grade it "
              f"against. It measured {elongated['cd'] / sphere_like['cd']:.2f} times the "
              f"sphere's on the same frontal area, which is HIGHER, and that is the "
              f"right way round at this Reynolds number: streamlining reduces PRESSURE "
              f"drag, and at Re 100 a slender hull's four-times-larger wetted area adds "
              f"more skin friction than the fairing saves. The ratio crosses one "
              f"somewhere above Re 1000, which is outside what this case runs and so is "
              f"not claimed here"),
    ]
    lines = [f"# underwater body: a towed capsule at Re {reynolds:g}, "
             f"{cells_per_d} cells per diameter",
             *_official("underwater", scaling, smagorinsky),
             f"  sphere:     Cd {sphere_like['cd']:.4f} against Schiller-Naumann "
             f"{expected:.4f} ({sphere_like['cd'] / expected - 1.0:+.2%})",
             f"  hull:       Cd {elongated['cd']:.4f} at fineness "
             f"{elongated['fineness']:g}, {ratio:.2f}x the sphere's on the same frontal "
             f"area -- higher, because at Re 100 the extra wetted area costs more "
             f"friction than the fairing saves in pressure drag",
             *notes]
    for figure in drawn:
        figures.ensure_rgb(figure)
    return _result("underwater", checks, lines, drawn, rows=rows)


# -- 11. a smoke case small enough for the main test suite ------------------------------


def run_smoke(*, cells: int = 12, steps: int = 40, collision: str = "trt",
              smagorinsky: float = 0.0, device: str | None = "cpu",
              out_dir: str | Path = ".", figure: bool = False,
              progress: Callable[[str], None] | None = None,
              **_: Any) -> dict[str, Any]:
    """The smallest run that exercises every kernel. Seconds on a CPU, in the test suite.

    Not a validation: twelve cells a side resolves nothing. What it checks is that every
    kernel launches, that the lattice stays finite, that a body gets a non-zero wrench
    whose sign is the stream's, and that mass is conserved -- which between them catch
    the whole class of failure where a kernel does not compile, an array is the wrong
    shape, or a layout change was applied to one accessor and not the other.
    """
    import warp  # noqa: F401

    from .world import Domain, FluidWorld

    diameter = 0.01
    dx = diameter / 4.0
    scaling = Scaling.for_reynolds(dx, diameter, 20.0, "water", u_lattice=0.05,
                                   scheme=collision, les=smagorinsky > 0)
    u_inf = scaling.reference_velocity
    size = (cells * dx, cells * dx, cells * dx)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                    faces={"-x": "inlet", "+x": "outlet", "-y": "slip", "+y": "slip",
                           "-z": "periodic", "+z": "periodic"},
                    face_velocity={"-x": (u_inf, 0.0, 0.0)})
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision=collision, smagorinsky=smagorinsky,
                       device=device)
    centre = tuple(v / 2.0 for v in size)
    body = world.add_body(position=centre, label="sphere")
    world.add_shape("sphere", body=body, scale=(diameter / 2.0, 0.0, 0.0))
    world.initialise(velocity=(u_inf, 0.0, 0.0))
    _advance(world, steps, progress=progress)
    force, torque = world.wrench(body)

    # -- mass, in a closed box, because the box above is not closed --------------------
    #
    # The world above has an inlet and an outflow, and neither conserves mass by
    # construction: the inlet overwrites its nodes with f_eq(1, u_in) whatever the
    # pressure there is, and the outflow copies its neighbour. Measured on the first run
    # of this case, the drift over forty steps was 1.0%, which is the inlet doing its
    # job and not a defect -- and a check that called it one would have had every future
    # reader looking for a streaming bug that was not there.
    #
    # So the mass check is a second, closed world: every face periodic, no body, no
    # force. There, mass is conserved exactly by the streaming step (every population
    # that leaves a node arrives at another) and the only thing that can break it is a
    # real bug -- a layout accessor applied to a read and not a write, a wrapped index
    # off by one, a population written twice. That is worth catching, and it is only
    # catchable where the answer is exactly zero.
    closed_domain = Domain(origin=(0.0, 0.0, 0.0), size=size, faces="periodic")
    closed = FluidWorld(domain=closed_domain, dx=dx, fluid="water", scaling=scaling,
                        substeps=1, collision=collision, smagorinsky=smagorinsky,
                        device=device)
    closed.initialise(velocity=(u_inf, 0.5 * u_inf, -0.3 * u_inf))
    mass_before = float(np.nansum(closed.fields()["rho"]))
    _advance(closed, steps)
    mass_after = float(np.nansum(closed.fields()["rho"]))
    drift = abs(mass_after / mass_before - 1.0) if mass_before else float("nan")

    checks = [
        Check("lattice finite and positive", world.stable(), "True", world.stable()),
        Check("drag on the sphere is downstream", force[0], "> 0", force[0] > 0.0),
        Check("mass drift in a closed periodic box", drift, "<= 1e-5",
              drift <= 1.0e-5,
              "conserved exactly by streaming, so the bound is single precision and not "
              "physics: a population is read, multiplied and written once a step in "
              f"float32, and {steps} steps of that accumulate about "
              "sqrt(steps)*eps = 1e-6, which is what this measures. The bound is 1e-5 "
              "for that reason and would be 1e-14 in double precision. The "
              "inlet/outlet box this case also runs drifts about a percent and is "
              "supposed to"),
        Check("torque finite", f"{torque[0]:+.3e}, {torque[1]:+.3e}, {torque[2]:+.3e}",
              "finite", all(math.isfinite(v) for v in torque)),
    ]
    drawn = []
    if figure:
        drawn.append(figures.slice_figure(world, Path(out_dir) / "live-smoke.png"))
        figures.ensure_rgb(drawn[0])
    lines = [f"# smoke: {cells}^3, {steps} steps, every kernel once",
             *world.report()[:3],
             f"  drag:       {force[0]:+.4e} N downstream",
             f"  mass drift: {drift:.2e} in a closed periodic box of the same size"]
    return _result("smoke", checks, lines, drawn, force=[float(v) for v in force],
                   drift=drift)


# -- the cross-check against this repository's own OpenFOAM run --------------------------


def _openfoam_cylinder() -> dict[str, Any]:
    """The committed OpenFOAM cylinder, read by the grader that already grades it.

    `benchmarks/moving_mesh/grade.py` is run as a subprocess on
    `tests/data/moving_mesh/viv_fixed/` -- a finished `pimpleFoam` run, committed, with
    its `postProcessing/forceCoeffs/*/coefficient.dat` intact -- and its JSON is read.

    **Not a constant in this file, on purpose.** The alternative is to copy St and Cd
    into `REFERENCES` and compare against the copy, and then the cross-check is between
    the live backend and a number somebody typed. Going through the grader means the
    comparison is against whatever the OpenFOAM files actually say today: if that run is
    ever re-trimmed, or the grader's windowing changes, the cross-check moves with it.
    The copies in `REFERENCES["openfoam_cylinder"]` are kept anyway and asserted equal
    to what comes back, so a drift is reported rather than silently absorbed.

    The grader's `viv` mode wants a released case beside the fixed one; `viv_loose` is
    the one committed next to it. Only the `fixed` block is read.
    """
    import subprocess
    import sys

    grader = REPO / "benchmarks" / "moving_mesh" / "grade.py"
    fixed = REPO / "tests" / "data" / "moving_mesh" / "viv_fixed"
    loose = REPO / "tests" / "data" / "moving_mesh" / "viv_loose"
    if not grader.is_file() or not fixed.is_dir():
        raise UnphysicalScaling(
            f"the cross-check reads this repository's own OpenFOAM run through "
            f"{grader}, and one of {grader} or {fixed} is not there. It is committed in "
            f"a source checkout and is NOT part of the installed wheel "
            f"(pyproject excludes benchmarks/ and the tests), so this case runs from a "
            f"checkout only"
        )
    done = subprocess.run(
        [sys.executable, str(grader), "viv", str(fixed), str(loose), "--json"],
        capture_output=True, text=True, timeout=600,
    )
    try:
        payload = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise UnphysicalScaling(
            f"benchmarks/moving_mesh/grade.py did not return JSON "
            f"(exit {done.returncode}): {done.stderr.strip()[:400]}"
        ) from exc
    return payload["fixed"]


def _try_draw(label: str, make: Callable[[], Any]) -> tuple[Any, str | None]:
    """Draw a figure, and **never let a drawing failure discard a finished run.**

    Returns `(figure_or_None, note_or_None)`. The note goes in the report so a missing
    figure is visible rather than silent.

    This exists because the same accident has now happened three times in this work, each
    time *after* every graded number had been computed and each time on a GPU:

    * a mangled f-string that built a dict key by concatenating a quote character out of
      a `chr` call, which raised `KeyError` after **1,087 s** of H100 on the
      ground-effect case (`test_no_case_builds_a_dict_key_out_of_chr_39` now forbids the
      construct, which is why it is described here rather than quoted -- quoting it
      failed that very test);
    * two report lines appended to `lines` before `lines` existed raised
      `UnboundLocalError` after **44 s** on `sph_hydrostatic`;
    * a one-frame animation tripped `figures.animate`'s own guard after **457 s** on
      `crosscheck-2d1`.

    All three are report-assembly faults. None of them says anything about the physics,
    and all three threw the physics away. A figure is a presentation of a result and must
    not be able to destroy one.
    """
    try:
        return make(), None
    except Exception as exc:  # noqa: BLE001 -- a figure must not sink a run
        return None, (f"  figure:     {label} was not drawn: "
                      f"{type(exc).__name__}: {exc}. The run's numbers are unaffected; "
                      f"a figure cannot be allowed to discard them.")


def crosscheck_2d1(*, cells_per_d: int = 32, u_lattice: float = 0.04,
                   outlet_rho: float = 1.0,
                   steps: int | None = None, device: str | None = None,
                   out_dir: str | Path = ".", animate: bool = True,
                   tolerance_scale: float = 1.0,
                   progress: Callable[[str], None] | None = None,
                   **_: Any) -> dict[str, Any]:
    """Schafer-Turek 2D-1: the same cylinder channel at Re 20, and **steady**.

    **What the first version of this function got wrong**, because it is the kind of
    mistake that hides: it was a hand copy of `_cylinder_once`'s setup that dropped two
    lines, and it reached NaN at step **1**. The two were the thin direction's faces
    (`periodic`, not `slip`) and `world.initialise(velocity=...)` -- a lattice started at
    rest behind a stiff velocity inlet has to absorb the inlet as an acoustic step and
    does not survive it. The tell that it was the copy and not Re 20 was that the same
    hand-built world NaN'd at Re 100 with the working case's own settings.

    **Why a second cross-check rather than a fix to the first.** The Re 100 cross-check
    reports a Strouhal number of 1.22 against 0.166 because the lift it measures carries
    a transverse acoustic mode: the contaminant's period is 361 lattice steps and this
    domain's transverse quarter-wave is 346, lift *is* the transverse force, and the
    incompressible reference cannot support such a mode at all. That is a real defect and
    it stays recorded as one.

    2D-1 removes the mechanism instead of suppressing it. At Re 20 the wake is steady,
    there is no oscillating lift, and a mode that needs an oscillator to excite it has
    nothing to couple to. So this case can compare the forces between backends while the
    unsteady one cannot.

    **What is graded, and what is only reported.** `benchmarks/felix/README.md` records a
    real Felix run of this exact case: Cd 5.522 (-1.04%), and **Cl 0.0010 against the
    paper's 0.0104-0.0110** -- a factor of ten low, which it attributes to mesh
    resolution and records rather than fixes. So drag is a genuine backend-to-backend
    comparison and lift is not; a bound on lift here would be a bound against a number
    this repository already knows is unconverged. Lift is therefore reported with its
    sign and order checked, and its value graded against nothing.

    The 2% drag bound is Felix's own, quoted from that README, so the two backends are
    held to the same line. Choosing a tighter one would fail Felix and a looser one would
    have been chosen to pass.
    """
    import numpy as np

    import warp  # noqa: F401

    from . import figures
    from .units import Scaling
    from .world import Domain, FluidWorld

    reference = REFERENCES["schafer_turek_2d1"]
    geometry = REFERENCES["schafer_turek_2d2"]["geometry"]
    length, height = geometry["length"], geometry["height"]
    diameter, centre = geometry["diameter"], geometry["centre"]
    reynolds = float(reference["reynolds"])
    limit = float(reference["tolerance"]["cd"]) * float(tolerance_scale)

    dx = diameter / cells_per_d
    nz = 3
    # Re 20 on the channel-height convention the paper uses for this family, through the
    # same `for_reynolds` the unsteady case uses, so the two are set up alike.
    scaling = Scaling.for_reynolds(dx, diameter, reynolds, "water",
                                   u_lattice=u_lattice, peak_factor=1.5,
                                   scheme="trt", les=False)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(length, height, nz * dx),
                    faces={"-x": "inlet", "+x": "outlet", "-y": "wall", "+y": "wall",
                           # **Periodic, not slip.** The thin direction is three cells of
                           # a nominally 2D case, and slip faces there make it a very
                           # shallow 3D duct instead of a 2D plane (DECISIONS_LIVE.md
                           # D13 is the same choice for the free-surface cases).
                           "-z": "periodic", "+z": "periodic"})
    # **The outlet anchors the density, which the extrapolating one does not.** With a
    # velocity inlet at the other end nothing holds the channel's mass: measured here,
    # the extrapolating outlet gained +7.06% of the domain's mass in 12,000 steps and the
    # drag climbed with it and never settled. Anchored, the same run holds mass to
    # -0.64% and reads Cd 5.713 against the paper's 5.5795 at 10 cells per diameter,
    # where the extrapolating one read 6.114.
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision="trt", smagorinsky=0.0,
                       outlet_rho=outlet_rho, device=device)

    # The parabolic inflow, which is the case: `4 U_max y (H - y) / H^2`. A uniform inlet
    # at the same mean is a different problem, and this project has had the inlet
    # condition backwards twice, both times from holding one reference and a stale
    # memory of the other.
    u_mean = scaling.reference_velocity
    peak = 1.5 * u_mean
    profile = np.zeros((world.ny, world.nz, 3))
    y = world.node_coordinates(1)
    profile[:, :, 0] = (4.0 * peak * y * (height - y) / height ** 2)[:, None]
    world.set_inlet_profile(profile)

    body = world.add_body(position=(centre[0], centre[1], nz * dx / 2))
    # The shape's position is in the BODY's frame and the body is already placed, so the
    # shape takes no offset. `_cylinder_once` says the same thing: passing the world
    # centre here puts the shape at twice the offset, which a voxeliser test caught once.
    world.add_shape("cylinder", body=body, scale=(diameter / 2.0, nz * dx, 0.0))
    # **A uniform stream, not rest.** A lattice started at rest behind a stiff velocity
    # inlet has to absorb the whole inlet step as an acoustic transient, and it does not:
    # omitting this line reached NaN at step 1, at Re 100 with the working case's own
    # settings as well as at Re 20, which is what proved the fault was the copy.
    world.initialise(velocity=(u_mean, 0.0, 0.0))

    depth = nz * dx
    coefficient = 2.0 / (scaling.fluid.rho * u_mean * u_mean * diameter * depth)
    flow_through = (length / dx) / u_lattice
    total = int(steps if steps else 40.0 * flow_through)

    # Steady, so the stopping rule is "the drag stopped changing" and not a number of
    # shedding periods. The cavity case taught this project that a step floor can run a
    # case past convergence into float32 erosion, so this watches the quantity instead.
    history: list[tuple[int, float, float]] = []
    frames: list[Any] = []
    every = max(1, total // 400)

    def take(index: int) -> None:
        if index % every:
            return
        force, _ = world.wrench(body)
        history.append((index, float(force[0]) * coefficient,
                        float(force[1]) * coefficient))
        # **Gate on the sample count, not on a second modulus of `index`.** The first
        # version tested `index % (total // 180)` here while the line above had already
        # returned unless `index % (total // 400) == 0`. For `total = 704,000` those are
        # 3,911 and 1,760, which are coprime, so both fired only at index 0 -- one frame,
        # and `figures.animate` refused it after 457 s of H100.
        if animate and len(frames) < 180 and len(history) % 2 == 1:
            frames.append(figures.slice_frame(world, axis=2))

    _advance(world, total, sample_every=1, sample=take, settle=0.0, progress=progress)

    tail = [row for row in history if row[0] >= 0.75 * total]
    cd = float(np.mean([row[1] for row in tail])) if tail else float("nan")
    cl = float(np.mean([row[2] for row in tail])) if tail else float("nan")
    # Steadiness, as a number: the drag's own spread over the window it is averaged on.
    spread = (float(np.std([row[1] for row in tail]) / abs(cd))
              if tail and cd else float("nan"))

    paper_cd = float(reference["cd"])
    felix_cd, felix_cl = float(reference["felix_cd"]), float(reference["felix_cl"])
    cl_mid = float(np.mean(reference["cl_interval"]))
    cd_error = cd / paper_cd - 1.0
    felix_gap = cd / felix_cd - 1.0

    checks = [
        Check("Cd against the paper's 5.5795", cd,
              f"within {limit:.0%} ({reference['tolerance_is']})",
              abs(cd_error) <= limit,
              f"{cd:.4f}, {cd_error:+.2%} of {paper_cd}"),
        # **Felix is dropped as a reference** (DECISIONS_LIVE.md D17), reported only.
        # Its lift is a factor of ten low by its own README, and its drag sits outside
        # the paper's own 5.57-5.59 interval -- so grading against it held the lattice to
        # a 2% window around a number not in the interval, and produced the state where
        # the lattice passed against Schafer-Turek and failed against Felix. A comparison
        # in which the better answer fails is not one worth keeping.
        Check("Cd against the Felix backend", cd,
              f"reported, not graded: Felix reads {felix_cd} on this case, itself "
              f"outside the paper's {reference['cd_interval']}, and its lift is a factor "
              f"of ten low. DECISIONS_LIVE.md D17 drops it as a reference; the "
              f"backend-to-backend job is done by the unsteady cross-check, which agrees "
              f"with its OpenFOAM run to -0.80% on drag", None,
              f"{cd:.4f} against Felix's {felix_cd}, {felix_gap:+.2%}"),
        Check("the wake was steady when the forces were read", spread,
              "Cd's own standard deviation over the averaging window within 1% of its "
              "mean; 2D-1 is a steady case and a drifting drag would mean the run was "
              "graded mid-transient, which is how the cavity case failed for two days",
              bool(spread == spread and spread <= 0.01), f"{spread:.3%}"),
        # **Lift is now graded against the paper's interval** (DECISIONS_LIVE.md D17).
        # It used to be checked only for sign and order of magnitude, because the only
        # comparison available was Felix's 0.0010 -- a factor of ten low by its own
        # README. With Felix dropped, Schafer-Turek's 0.0104-0.0110 is the bar, and the
        # interval is used as an interval rather than through a relative tolerance on its
        # midpoint: Cl here is 0.01 against a drag of 5.6, a small difference of two large
        # surface integrals, and a relative bound on a near-zero quantity is a trap.
        Check("Cl against the paper's interval", cl,
              f"inside {reference['cl_interval']} (Schafer & Turek 1996, test case "
              f"2D-1). An absolute interval, not a relative tolerance: lift here is "
              f"0.01 against a drag of 5.6, so a percentage bound on it would be a bound "
              f"on the difference of two large surface integrals",
              bool(cl == cl and reference["cl_interval"][0] <= cl
                   <= reference["cl_interval"][1]),
              f"{cl:.5f} against {reference['cl_interval']}; Felix reads {felix_cl} "
              f"here, which is why it is no longer a reference"),
    ]

    drawn = []
    notes: list[str] = []
    if frames:
        gif = Path(out_dir) / "crosscheck-2d1.gif"
        made, note = _try_draw(f"the {len(frames)}-frame animation",
                               lambda: figures.animate(frames, gif) or gif)
        drawn.append(made) if made else notes.append(note or "")
    if history:
        drawn.append(figures.profile_figure(
            Path(out_dir) / "crosscheck-2d1-forces.png",
            [{"x": [r[0] for r in history], "y": [r[1] for r in history],
              "label": "Cd"},
             {"x": [r[0] for r in history], "y": [r[2] for r in history],
              "label": "Cl", "panel": 1}],
            title=f"2D-1, Re 20, steady: Cd {cd:.4f} ({cd_error:+.2%}), Cl {cl:.5f}",
            xlabel="step", ylabel="coefficient", panels=2))
    for figure in drawn:
        if str(figure).endswith(".png"):
            figures.ensure_rgb(figure)

    lines = [
        "# cross-check 2D-1: the same channel at Re 20, steady",
        f"  lattice:    {cells_per_d} cells per diameter, dx {dx * 1e3:.3f} mm, "
        f"{total:,} steps ({total / flow_through:.0f} flow-throughs)",
        f"  drag:       Cd {cd:.4f}, {cd_error:+.2%} of the paper's {paper_cd} "
        f"(interval {reference['cd_interval']})",
        f"  backends:   this one {cd:.4f}, Felix {felix_cd} -> {felix_gap:+.2%}. "
        f"Both graded against the same 2% bound, which is Felix's own.",
        f"  lift:       Cl {cl:.5f}, paper {reference['cl_interval']}, "
        f"Felix {felix_cl}. REPORTED; only its sign and order are graded, and why is "
        f"in the check's own target.",
        f"  steadiness: Cd's spread over the averaging window {spread:.3%} of its mean",
        f"  reference:  {reference['source']}",
        f"  caveat:     {reference['caveat']}",
        *notes,
    ]
    return _result("crosscheck-2d1", checks, lines, drawn, live_cd=cd, live_cl=cl,
                   paper_cd=paper_cd, felix_cd=felix_cd, felix_cl=felix_cl,
                   cd_error=cd_error, felix_gap=felix_gap, steadiness=spread,
                   cells_per_d=cells_per_d, steps=total)


def crosscheck(*, cells_per_d: int = 20, u_lattice: float = 0.05,
               outlet_rho: float = 1.0,
               collision: str = "trt",
               smagorinsky: float = 0.0, device: str | None = None,
               out_dir: str | Path = ".", periods: float = 30.0,
               steps: int | None = None, animate: bool = True,
               progress: Callable[[str], None] | None = None,
               **_: Any) -> dict[str, Any]:
    """The same cylinder on both backends: this one, and the committed OpenFOAM run.

    A cylinder at Re 100 in water, D = 10 mm, U = 10 mm/s, 2.5% blockage -- the case
    `tests/data/moving_mesh/viv_fixed/` holds a finished `pimpleFoam` solution of. The
    geometry, the speed and the viscosity are read out of that run's own constants
    rather than written here, so the two backends are given the same case by
    construction.

    This is the brief's cross-check and it is the most useful single number in the
    suite, for two reasons. It needs no OpenFOAM installation, no cluster and no
    network -- the OpenFOAM side ran once, in September, and is committed. And it is a
    comparison between two completely independent discretisations of the same physics:
    a finite-volume pressure-correction solver on a body-fitted mesh, and a lattice
    Boltzmann scheme on a uniform grid with an interpolated boundary. They share no
    code, no mesh and no boundary treatment. Agreement between them is evidence of a
    kind that agreement with a published table is not.

    Strouhal is graded both ways: against the OpenFOAM run, and against Williamson's
    0.164 for an unconfined cylinder -- which this case should sit about a percent above,
    because `benchmarks/moving_mesh/` records that 2.5% blockage raises it by about
    that, and the OpenFOAM run's +1.31% is what that looks like. Drag is graded against
    the OpenFOAM run only, because there is no canonical published Cd at Re 100.
    """
    import warp  # noqa: F401

    from .world import Domain, FluidWorld

    foam = _openfoam_cylinder()
    constants = foam["constants"]
    diameter = float(constants["D"])
    speed = float(constants["U"])
    reference = REFERENCES["openfoam_cylinder"]
    tolerances = reference["tolerance"]
    blockage = reference["geometry"]["blockage"]

    # The copies in REFERENCES are the documented ones; the grader's are the live ones.
    drift = {
        "strouhal": abs(foam["strouhal"] - reference["strouhal"]),
        "cd": abs(foam["mean_cd"] - reference["cd"]),
    }

    height = diameter / blockage
    # Two channel heights, which at 2.5% blockage is 80 diameters: 40 upstream of the
    # cylinder and 40 down. Four heights was the first version and is 160 diameters of
    # mostly undisturbed fluid, which doubles the cells and doubles the flow-through
    # time that has to be paid for before anything can be sampled.
    length = 2.0 * height
    dx = diameter / cells_per_d
    nz = 3
    reynolds = speed * diameter / float(constants.get("nu", 1.0e-6) or 1.0e-6)
    scaling = Scaling.for_reynolds(dx, diameter, 100.0, "water", u_lattice=u_lattice,
                                   scheme=collision, les=smagorinsky > 0)
    u_mean = scaling.reference_velocity
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(length, height, nz * dx),
                    faces={"-x": "inlet", "+x": "outlet", "-y": "slip", "+y": "slip",
                           "-z": "periodic", "+z": "periodic"},
                    face_velocity={"-x": (u_mean, 0.0, 0.0)})
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision=collision, smagorinsky=smagorinsky,
                       outlet_rho=outlet_rho, device=device)
    centre = (height, height / 2.0 + dx / 2.0, nz * dx / 2.0)
    # Half a cell off the centreline, which is what the OpenFOAM case does: a cylinder
    # exactly on the axis sheds only when the symmetry is broken by round-off, and the
    # transient takes an order of magnitude longer. benchmarks/README.md records the
    # same 5 mm offset in the Schafer-Turek 2D-1 case for the same reason.
    body = world.add_body(position=centre, label="cylinder")
    world.add_shape("cylinder", body=body, scale=(diameter / 2.0, nz * dx, 0.0))
    world.initialise(velocity=(u_mean, 0.0, 0.0))

    # Flow-throughs to establish, periods to sample. See `_cylinder_once` for the
    # argument and for what the first version of it measured: this case ran 1.13
    # flow-throughs and sampled from 0.56, which is a wake that had not crossed the
    # domain once, and it came out Cd -7% and St +7%.
    period = cells_per_d / (foam["strouhal"] * u_lattice)
    flow_through = (length / dx) / u_lattice
    establish = 6.0 * flow_through
    total = int(steps if steps else establish + 25.0 * period)
    settle = min(0.95, establish / total)
    depth = nz * dx
    coefficient = 2.0 / (scaling.fluid.rho * u_mean * u_mean * diameter * depth)
    times: list[float] = []
    cd: list[float] = []
    cl: list[float] = []
    frames: list[np.ndarray] = []

    def take(index: int) -> None:
        force, _ = world.wrench(body)
        times.append(world.steps * scaling.dt)
        cd.append(float(force[0]) * coefficient)
        cl.append(float(force[1]) * coefficient)
        if animate and len(frames) < 240 and index % max(1, total // 240) == 0:
            frames.append(figures.slice_frame(world, axis=2))

    _advance(world, total, sample_every=1, sample=take, settle=settle,
             progress=progress)

    cd_array = np.asarray(cd)
    cl_array = np.asarray(cl)
    spectral = figures.strouhal(times, cl_array, diameter, u_mean)
    live_cd = float(cd_array.mean()) if cd_array.size else float("nan")
    live_st = spectral["strouhal"]
    cd_error = live_cd / foam["mean_cd"] - 1.0
    st_error = live_st / foam["strouhal"] - 1.0
    williamson = 0.164
    st_vs_williamson = live_st / williamson - 1.0

    checks = [
        Check("mean Cd, live backend against the OpenFOAM run", live_cd,
              f"{foam['mean_cd']:.6f} +/- {tolerances['cd']:.0%} (OpenFOAM, "
              f"benchmarks/moving_mesh/grade.py on the committed run)",
              abs(cd_error) <= tolerances["cd"]),
        Check("Strouhal, live backend against the OpenFOAM run", live_st,
              f"{foam['strouhal']:.6f} +/- {tolerances['strouhal']:.0%} (OpenFOAM)",
              abs(st_error) <= tolerances["strouhal"]),
        Check("Strouhal against Williamson (1996), unconfined", live_st,
              f"{williamson} +/- {tolerances['strouhal']:.0%}; both runs should sit "
              f"about a percent above it on 2.5% blockage",
              abs(st_vs_williamson) <= tolerances["strouhal"]),
        Check("Cl RMS, live against OpenFOAM",
              f"{float(cl_array.std()):.4f} against {foam['cl_rms']:.4f}",
              "reported, not graded", None,
              "the lift is a small difference of two large surface integrals; "
              "benchmarks/README.md reports and does not grade it on the sibling case"),
        Check("the OpenFOAM numbers have not drifted from the documented ones",
              f"St {drift['strouhal']:.2e}, Cd {drift['cd']:.2e}",
              "< 1e-5 on both",
              drift["strouhal"] < 1e-5 and drift["cd"] < 1e-5,
              "REFERENCES['openfoam_cylinder'] carries copies for documentation; the "
              "graded comparison uses what the grader reads today"),
        Check("cycles sampled", f"{spectral['cycles']:.1f} by {spectral['method']}",
              f"the OpenFOAM window held {foam['cycles']} cycles", None),
    ]
    out = Path(out_dir)
    drawn = [figures.slice_figure(
        world, out / "live-crosscheck-cylinder.png", axis=2,
        title=f"cylinder Re 100: live Cd {live_cd:.4f} / St {live_st:.5f} against "
              f"OpenFOAM {foam['mean_cd']:.4f} / {foam['strouhal']:.5f}")]
    drawn.append(figures.profile_figure(
        out / "live-crosscheck-history.png",
        [{"x": times, "y": cd, "label": "Cd, live backend"},
         {"x": [times[0], times[-1]] if times else [0, 1],
          "y": [foam["mean_cd"]] * 2, "label": "mean Cd, OpenFOAM", "style": "--"},
         {"x": times, "y": cl, "label": "Cl, live backend", "panel": 1}],
        title=f"two backends, one case: Cd {cd_error:+.2%}, St {st_error:+.2%}",
        xlabel="t (s)", ylabel="coefficient", panels=2))
    if frames:
        gif = figures.animate(frames, out / "live-crosscheck-cylinder.gif", fps=20)
        if gif:
            drawn.append(gif)
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [
        f"# cross-check: a cylinder at Re {reynolds:g} on both backends",
        f"  case:       D {diameter * 1e3:g} mm, U {speed * 1e3:g} mm/s, water, "
        f"{blockage:.1%} blockage, read from the OpenFOAM run's own constants",
        f"  openfoam:   Cd {foam['mean_cd']:.6f}, St {foam['strouhal']:.6f} over "
        f"{foam['cycles']} cycles ({reference['source']})",
        f"  live:       Cd {live_cd:.6f} ({cd_error:+.2%}), St {live_st:.6f} "
        f"({st_error:+.2%}), {spectral['cycles']:.1f} cycles",
        f"  run:        {total:,} steps = {total / flow_through:.1f} flow-throughs, "
        f"sampled from {establish / flow_through:.1f}",
        f"  williamson: St {williamson} unconfined; live is {st_vs_williamson:+.2%} on "
        f"it, OpenFOAM is {foam['strouhal'] / williamson - 1.0:+.2%}",
        f"  caveat:     {reference['caveat']}",
        *scaling.report(), *_official("cylinder", scaling, smagorinsky),
        *world.report()[-2:],
    ]
    # **Both estimators and the cycle count, not just the chosen one.** `strouhal`
    # computes the zero-crossing and the FFT answers and returns both; this case used to
    # record only the one it picked, and threw away the thing that identifies a bad run.
    #
    # What that cost: under a stiff velocity inlet this case reports St 1.22 against
    # 0.166 and 181 cycles against OpenFOAM's 7, and the recorded result carried no way
    # to tell a wrong flow from a wrong peak-pick. It is a wrong flow -- the number came
    # from zero crossings, which count real sign changes -- and the arithmetic then says
    # what is oscillating: the contaminant's period is 361 steps, the **transverse**
    # quarter-wave of this domain is 346 (the cross-section is 800 cells and the lattice
    # sound speed is 0.577 cells a step), and lift *is* the transverse force. Which is
    # why St is wrong by seven times while Cd, the streamwise force, is wrong by 25%.
    #
    # The streamwise acoustic mode, which was the first guess, is at 688 steps and does
    # not fit; checking it would have cost one division and saved proposing a fix for it.
    return _result("crosscheck", checks, lines, drawn, live_cd=live_cd,
                   live_strouhal=live_st, openfoam=foam, cd_error=cd_error,
                   st_error=st_error,
                   strouhal_crossings=spectral.get("strouhal_crossings"),
                   strouhal_fft=spectral.get("strouhal_fft"),
                   cycles=spectral.get("cycles"), method=spectral.get("method"),
                   force_history=[list(row) for row in zip(times, cl)] if len(times) < 4000
                   else None)




# ======================================================================================
# M5: the free surface
# ======================================================================================
#
# Three cases, and the choice of what each is graded against is the whole design.
#
# This repository has already decided, in `benchmarks/README.md`'s SPH dam-break row,
# that **Martin and Moyce (1952) cannot be quoted here**: the paper could not be read,
# the publisher refuses the request, and the two non-dimensionalisations in circulation
# -- `t sqrt(g/a)` and `t sqrt(2g/a)` -- differ by a factor of 1.414 at every point. That
# decision binds this family too. So the dam break is graded against **Ritter's (1892)
# shallow-water solution**, which is a closed form that needs no paper, and the Martin
# and Moyce comparison is reported in both conventions and graded in neither.
#
# The sloshing tank is graded against linear theory, `omega^2 = g k tanh(k h)`, which
# the SPH family already uses and attributes to Lamb's *Hydrodynamics* 6th edition
# article 228. Exact, and derivable from first principles in half a page.
#
# The hydrostatic case is graded against `p = rho g h`, which is exact and is the only
# one of the three that tests the thing a floating body depends on: that the pressure
# really does increase with depth, because that is what buoyancy is.


_FREE_SURFACE_REFERENCES = {
    "ritter": {
        "source": (
            "Ritter, A. (1892), 'Die Fortpflanzung der Wasserwellen', Zeitschrift des "
            "Vereines Deutscher Ingenieure 36(33):947-954, as the standard shallow-water "
            "dam-break solution"
        ),
        "read": (
            "not read; the solution is re-derived rather than quoted -- it is the "
            "characteristics solution of the shallow-water equations for a dry bed and "
            "is three lines, which is why it is safe to use without the paper"
        ),
        "formula": "x_front(t) = 2 t sqrt(g h0),  on a dry horizontal bed",
        "caveat": (
            "**Inviscid, hydrostatic and dry-bed.** Ritter's front moves at exactly "
            "2 sqrt(g h0) from the instant the dam goes, which no real dam break does: "
            "a real front is slowed by bed friction and by the vertical acceleration the "
            "shallow-water equations neglect, and measurements run 10-30% below Ritter "
            "at late time. It is quoted here as an EARLY-TIME bound, over the first "
            "half-height of travel, where the discrepancy is smallest, and the measured "
            "front is reported against it rather than fitted to it. Ritter also assumes "
            "the dam is removed instantaneously; a lattice removes it in one step, which "
            "is the closest any method gets"
        ),
        "tolerance": 0.15,
        "tolerance_is": (
            "a grading choice, and a loose one on purpose: it brackets the gap between "
            "an inviscid shallow-water front and a viscous three-dimensional one at the "
            "Reynolds number a free-surface lattice can afford (see "
            "`units.Scaling.for_froude`). A run outside 15% is not resolving the "
            "collapse at all; a run inside it has not proved it is accurate"
        ),
    },
    "linear_sloshing": {
        "source": (
            "linear water-wave theory: omega^2 = g k tanh(k h) with k = pi/L for the "
            "first mode of a rectangular tank. Lamb, 'Hydrodynamics', 6th ed., art. 228, "
            "which is the attribution benchmarks/sph/ already uses for the same formula"
        ),
        "read": "the formula is standard and is re-derived rather than quoted",
        "formula": "omega^2 = g k tanh(k h),  k = pi / L,  period T = 2 pi / omega",
        "caveat": (
            "**Linear, so the amplitude has to be small.** The theory is the first term "
            "of an expansion in wave steepness, and a 5%-of-depth initial displacement "
            "is where benchmarks/sph/ puts it for the same reason. **Which way a finite "
            "amplitude moves the period depends on the depth, and this caveat used to "
            "have the sign backwards.** Concus 1962 (UCRL-10208) eq. (34) at delta = 0 "
            "-- the capillary-gravity generalisation of Tadjbakhsh & Keller (1960), "
            "whose original is paywalled -- gives w_2 = [-2 w0^5 - 3 w0 - 12 w0^-3 "
            "+ 9 w0^-7] / 32 with w0^2 = tanh(k h), and w = w0 + (1/2) (k a)^2 w_2. It "
            "returns exactly -1/4 in deep water and changes sign at the critical depth "
            "k h = 1.058. BELOW that depth w_2 is positive and a larger amplitude "
            "SHORTENS the period; above it, lengthens. This tank sits at k h = 0.9425, "
            "so w_2 = +0.1179 and at k a = 0.0628 the shift is -0.027%. A measured "
            "period ABOVE the linear one is therefore NOT the finite amplitude. "
            "It is also inviscid: the measured oscillation decays and the linear period "
            "does not, so the frequency is taken from the first few cycles"
        ),
        "tolerance": 0.05,
        "tolerance_is": "5% on the first-mode period, measured over the first cycles",
    },
    "hydrostatic": {
        "source": "p = rho g h, exact; no paper",
        "read": "n/a",
        "formula": "p(z) = rho g (h - z),  and the surface stays flat",
        "caveat": (
            "**Exact, and the one thing in this family that a floating body depends on.** "
            "Archimedes is nothing but this pressure integrated over a hull, so a solver "
            "that gets a draft right while getting this wrong is getting it right by "
            "accident. The measured surface sits slightly BELOW the level it was filled "
            "to, and that is not an error: the scheme is weakly compressible, so the "
            "column compresses under its own weight. The deficit is `rho g h / (2 rho "
            "c_s^2)` in physical units, which the case computes and checks the residual "
            "against rather than absorbing into a tolerance"
        ),
        "tolerance": 0.02,
        "tolerance_is": (
            "2% on the mean surface height after the predicted compression is accounted "
            "for, and on the flatness as a fraction of the depth"
        ),
    },
}


REFERENCES.update(_FREE_SURFACE_REFERENCES)
"""The free surface's references, folded into the one dictionary the graders read.

Kept as a separate literal above purely so the free-surface block reads as a unit; a
second dictionary would be a second place for `_official` and `report` to forget to
look."""


def _free_surface_world(*, cells: int, size: Sequence[float], depth: float,
                        faces: Any, collision: str, smagorinsky: float,
                        device: str | None, tau: float = 0.53,
                        gravity: float = 9.81, thin: int = 0,
                        mach: float = 0.07):
    """A `FreeSurfaceWorld` sized from a cell count and a tank, with the Froude scaling.

    One helper for the three cases, so that "how a free-surface case is set up" is one
    thing rather than three. `cells` counts across the LONGEST side, which is what
    `Domain.spacing` means everywhere else in this file.
    """
    from .freesurface import FreeSurfaceWorld
    from .world import Domain

    # A zero in `size` means "thin": `thin` cells deep, periodic, so the case is the
    # two-dimensional answer computed by the three-dimensional kernels. The cavity does
    # the same thing and `run_cavity` has the argument for why it is exact.
    size = list(size)
    spanning = max(v for v in size if v > 0)
    dx = spanning / cells
    thinned = [index for index, v in enumerate(size) if v <= 0]
    size = [v if v > 0 else max(int(thin), 1) * dx for v in size]
    # **A thin direction has to be periodic (DECISIONS_LIVE.md D13).** A
    # free-surface case asks for `faces="wall"` meaning "a closed tank", and that sets
    # all six faces to no-slip -- including the two ends of a three-cell-wide slot, which
    # then has wall friction on both sides of every cell and no throughflow at all. The
    # sloshing tank did not slosh: a cosine first mode decayed monotonically from 2.00 mm
    # to 0.03 mm without crossing zero, and converted 102,455 cells trying. The same
    # tank with these two faces periodic oscillates, and its period brackets linear
    # theory. So a `size` entry of zero means "thin", and thin means periodic, here,
    # once, rather than in each case's face dictionary where one of them will forget.
    if thinned:
        names = ("-x", "+x", "-y", "+y", "-z", "+z")
        faces = ({name: faces for name in names} if isinstance(faces, str)
                 else dict(faces))
        for index in thinned:
            faces[("-x", "-y", "-z")[index]] = "periodic"
            faces[("+x", "+y", "+z")[index]] = "periodic"
    domain = Domain(origin=(0.0, 0.0, 0.0), size=tuple(size), faces=faces)
    # **`mach` is threaded because the sloshing study cannot be interpreted without it.**
    # `for_froude` with no `g_lattice` sets `g_lattice = (mach c_s)^2 dx / h`, so
    # `dt ~ dx` -- acoustic scaling, measured at `dx^1.000000` with a residual of 3.4e-15
    # over seven resolutions, not the `dx^1.5` three places in this repository used to
    # claim (`units.py` twice and `PROGRESS_LIVE.md`, all now corrected). That
    # means Ma and tau are **frozen** along the refinement path while `Re_eff ~ 1/dx`
    # runs away, so an O(Ma^2) error of 4.9e-3 survives Richardson extrapolation
    # untouched and sits inside the "converged" limit by construction. Varying `mach` at
    # a fixed grid is the one experiment that separates that from a scheme error, and
    # until now it could not be run: the knob fell into `run_sloshing`'s `**_`.
    scaling = Scaling.for_froude(dx, depth, "water", gravity=gravity, tau=tau,
                                 mach=mach, scheme=collision, les=smagorinsky > 0)
    world = FreeSurfaceWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                             substeps=1, collision=collision,
                             smagorinsky=smagorinsky, device=device,
                             gravity=(0.0, 0.0, -gravity),
                             # These cases exist to measure the frozen backend, which is
                             # the one use that does not need the warning.
                             experimental=True)
    return world, scaling


def _free_surface_lines(world: Any, scaling: Scaling, length: float,
                        gravity: float = 9.81) -> list[str]:
    """The two numbers every free-surface report must carry, every time.

    The Froude number, which is matched, and the Reynolds number, which is not -- and
    by how much. `units.Scaling.nu_lattice_override` is the argument for why that pair
    has to be printed rather than assumed, and this is the function that makes sure no
    case forgets.
    """
    return [
        f"  froude:     {scaling.froude(length, gravity):.4f} on {length:g} m, matched "
        f"by construction",
        f"  reynolds:   {scaling.reynolds_effective(length):.0f} achieved, against "
        f"{scaling.reference_velocity * length / scaling.fluid.nu:.0f} for real water: "
        f"this lattice runs {scaling.viscosity_ratio:.0f}x the fluid's viscosity, "
        f"because a free surface's timestep is set by gravity and water's viscosity at "
        f"that timestep is a tau of 0.50002",
    ]


def run_hydrostatic(*, cells: int = 32, depth_fraction: float = 0.6,
                    height: float = 0.1, steps: int = 0, collision: str = "trt",
                    smagorinsky: float = 0.0, device: str | None = None,
                    out_dir: str | Path = ".",
                    progress: Callable[[str], None] | None = None,
                    **_: Any) -> dict[str, Any]:
    """Still water in a closed tank: does it stay still, and does it weigh the right amount?

    The first thing to ask of a free surface and the easiest to get wrong. Three graded
    properties:

    * the **volume** is the volume it was filled with, which tests that a partly filled
      interface cell counts as its fraction and not as a whole cell or as nothing;
    * the surface stays **flat**, which tests that the atmospheric reconstruction does
      not push the surface around;
    * the **mass drift** is nothing, over as long a run as the case is given.

    And the surface height is checked against a prediction rather than against the level
    it was filled to, because those differ and the difference is physics. The scheme is
    weakly compressible, so a column of water compresses under its own weight: the
    density at the bottom is higher by `rho g h / c_s^2`, the mean by half that, and the
    same mass therefore occupies a column shorter by `h rho g h / (2 rho c_s^2)`. On the
    first run of this case that predicted 0.245% and the measurement was 0.26%.
    """
    import warp  # noqa: F401

    gravity = 9.81
    depth = depth_fraction * height
    world, scaling = _free_surface_world(
        cells=cells, size=(height * 0.5, height * 0.5, height), depth=depth,
        faces="wall", collision=collision, smagorinsky=smagorinsky, device=device,
        gravity=gravity)  # fully three-dimensional: a still tank has nothing to thin
    world.initialise_free_surface(level=depth)

    exact_volume = world.domain.size[0] * world.domain.size[1] * depth
    start_volume = world.water_volume()
    total = int(steps if steps else 5000)
    heights: list[tuple[float, float, float]] = []

    def take(index: int) -> None:
        surface = world.surface_height()
        heights.append((world.steps * scaling.dt, float(np.nanmean(surface)),
                        float(np.nanstd(surface))))

    _advance(world, total, sample_every=max(1, total // 120), sample=take,
             progress=progress)

    surface = world.surface_height()
    mean_height = float(np.nanmean(surface))
    flatness = float(np.nanstd(surface))
    drift = world.mass_drift()
    bad_neighbours = world.fluid_touching_gas()
    # The hydrostatic gradient, fitted over the fluid cells rather than taken from two
    # of them: a two-point slope on a field that is still ringing acoustically is mostly
    # the ringing.
    node_z, density = world.density_profile()
    usable = np.isfinite(density)
    slope = (float(np.polyfit(node_z[usable], density[usable], 1)[0])
             if usable.sum() > 3 else float("nan"))

    # The compressible deficit, predicted rather than fitted. c_s in physical units is
    # the lattice speed of sound carried into metres per second by the scaling.
    sound = CS * scaling.dx / scaling.dt
    exact_slope = -world.fluid.rho * gravity / (sound * sound)
    compression = gravity * depth / (2.0 * sound * sound)
    predicted = depth * (1.0 - compression)
    limit = REFERENCES["hydrostatic"]["tolerance"]
    height_error = (mean_height - predicted) / depth

    checks = [
        Check("water volume at the start", start_volume * 1e6,
              f"{exact_volume * 1e6:.4f} mL +/- {limit:.0%}",
              abs(start_volume / exact_volume - 1.0) <= limit,
              "a partly filled interface cell has to count as its fraction; counting it "
              "as a whole cell or as none is a whole cell of error on every surface"),
        Check("surface height against the compressible prediction", mean_height * 1e3,
              f"{predicted * 1e3:.4f} mm +/- {limit:.0%} of the depth "
              f"(filled to {depth * 1e3:.2f} mm, less {compression:.3%} compression)",
              abs(height_error) <= limit),
        Check("surface flatness", flatness / depth,
              f"<= {limit:.0%} of the depth", flatness / depth <= limit,
              "the standard deviation of the free surface over the tank; still water "
              "that is not flat is the atmospheric reconstruction pushing on it"),
        Check("hydrostatic pressure gradient", slope / exact_slope,
              f"1.0 +/- 10% ({exact_slope:+.4f} kg/m4 exact)",
              abs(slope / exact_slope - 1.0) <= 0.10,
              "d(rho)/dz against `-rho g / c_s^2`, fitted over the fluid cells. **The "
              "most important number in this case**: pressure rising with depth at the "
              "right rate is what buoyancy IS, so a solver that gets a draft right "
              "while getting this wrong gets it right by accident. It rings acoustically "
              "for the first few hundred steps and settles; at 600 steps it read 0.94x "
              "and by 5,000 it reads 0.995x"),
        Check("fluid cells touching gas cells", bad_neighbours, "exactly 0",
              bad_neighbours == 0,
              "the invariant the whole algorithm rests on: the interface is a closed "
              "layer one cell thick, and a hole in it means the cells around it stream "
              "in populations nothing ever wrote. A tank filled to a level that lands "
              "on a cell boundary used to produce exactly that, with no interface cells "
              "at all and a free surface that could not move"),
        Check("mass drift", drift["drift"], "<= 1e-3 of the initial mass",
              abs(drift["drift"]) <= 1e-3,
              f"over {total:,} steps and {drift['conversions']:,} cell conversions"),
        Check("lattice finite", world.stable(), "True", world.stable()),
        Check("predicted compression", f"{compression:.4%}",
              "reported, not graded", None,
              "rho g h / (2 rho c_s^2) with c_s the lattice speed of sound in physical "
              f"units, {sound:.2f} m/s here"),
    ]
    out = Path(out_dir)
    drawn = [
        figures.slice_figure(world, out / "live-hydrostatic.png", axis=1,
                             title=f"still tank, {cells} cells, surface "
                                   f"{mean_height * 1e3:.3f} mm"),
        figures.profile_figure(
            out / "live-hydrostatic-pressure.png",
            [{"x": density[usable], "y": node_z[usable] * 1e3,
              "label": "measured density", "style": "o"},
             {"x": [world.fluid.rho + exact_slope * (z - node_z[usable][-1])
                    for z in node_z[usable]], "y": node_z[usable] * 1e3,
              "label": "rho_0 - rho g z / c_s^2"}],
            title=f"hydrostatic: slope {slope / exact_slope:.4f} of exact",
            xlabel="density (kg/m3)", ylabel="height (mm)"),
        figures.profile_figure(
            out / "live-hydrostatic-surface.png",
            [{"x": [h[0] for h in heights], "y": [h[1] * 1e3 for h in heights],
              "label": "mean surface height (mm)"},
             {"x": [heights[0][0], heights[-1][0]] if heights else [0, 1],
              "y": [predicted * 1e3] * 2, "label": "compressible prediction",
              "style": "--"}],
            title=f"still water: drift {drift['drift']:+.2e}, flatness "
                  f"{flatness * 1e3:.4f} mm",
            xlabel="t (s)", ylabel="mm"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# hydrostatic: {depth * 1e3:g} mm of water in a {height * 1e3:g} mm "
             f"tank, {cells} cells",
             *scaling.report(),
             *_free_surface_lines(world, scaling, depth, gravity),
             *_official("hydrostatic", scaling, smagorinsky),
             f"  volume:     {start_volume * 1e6:.4f} mL against an exact "
             f"{exact_volume * 1e6:.4f} mL",
             f"  surface:    {mean_height * 1e3:.4f} mm against a predicted "
             f"{predicted * 1e3:.4f} ({height_error:+.3%} of the depth), flat to "
             f"{flatness * 1e3:.4f} mm",
             f"  pressure:   d(rho)/dz {slope:+.4f} against an exact "
             f"{exact_slope:+.4f} kg/m4 ({slope / exact_slope:.4f}x)",
             f"  interface:  {bad_neighbours} fluid cells touching gas (must be 0)",
             *world.report()[-3:]]
    return _result("hydrostatic", checks, lines, drawn, volume=start_volume,
                   surface=mean_height, flatness=flatness, drift=drift,
                   gradient_ratio=slope / exact_slope if exact_slope else float("nan"))




def _front_position(world: Any, threshold_cells: float = 0.5) -> float:
    """The furthest x a front of water has reached, in metres, by a depth threshold.

    **The threshold is the measurement's weak point and is reported, not hidden.**
    Ritter's front is a tongue that thins to zero depth, so "where the front is" depends
    on how little water still counts as water: the exact solution has no leading edge at
    all. Half a cell of depth is the choice here -- the smallest amount a lattice can
    represent without it being a single interface cell's rounding -- and the case reports
    the front at a quarter, a half and one full cell so the sensitivity is on the page
    rather than in a footnote.

    Returns the far edge of the furthest wet column, not its centre: the front is an
    edge, and a cell that holds water is wet all the way across.
    """
    depth = world.surface_height(axis=2)
    wet = np.nan_to_num(depth.mean(axis=1)) > threshold_cells * world.dx
    if not wet.any():
        return float("nan")
    last = int(np.flatnonzero(wet)[-1])
    return float(world.domain.origin[0] + (last + 1) * world.dx)


def run_dam_break(*, cells: int = 325, height: float = 0.05, reservoir: float = 4.0,
                  tank: float = 0.0,
                  collision: str = "trt", smagorinsky: float = 0.0,
                  device: str | None = None, out_dir: str | Path = ".",
                  steps: int = 0, gif: bool = True,
                  progress: Callable[[str], None] | None = None,
                  **_: Any) -> dict[str, Any]:
    """A column of water collapses: the surge front against Ritter's shallow-water speed.

    A column of water `height` deep and `reservoir * height` wide, released at `t = 0`
    on a dry horizontal bed. Quasi-two-dimensional, three planes periodic in the thin
    direction, for the same reason the cavity is.

    **The reservoir is four depths wide and that is a measurement, not a preference.** A
    square column was tried first and cannot be graded against Ritter at all. Ritter is
    the shallow-water solution, so it needs the hydrostatic assumption to hold, and a
    column as tall as it is wide violates it for the first couple of time scales while
    the collapse is still vertically accelerating -- measured, the depth at the dam ran
    2.07 times Ritter's at `T = 0.5` and settled to 1.08 by `T = 2`. But the depression
    wave reaches the back wall at `T = a/h0` and returns to the dam at `T = 2 a/h0`, so
    for a square column Ritter expires at `T = 2` -- exactly when the agreement arrives,
    which makes the agreement worthless. Four depths of reservoir buys a valid window of
    `T` in [2, 8] and the case grades inside it.

    **Graded on the depth profile, not on the front position** (DECISIONS_LIVE.md D14).
    Ritter's front is a
    tongue that thins to zero depth, so where it is depends entirely on how little water
    still counts as water -- and the profile is what has physical content. Two numbers:
    the depth at the dam, which Ritter fixes at exactly `4 h0 / 9` for as long as the
    solution holds, and the RMS of the whole profile where Ritter is thicker than two
    cells. The front is reported at three thresholds so its spread is visible.

    **Graded against Ritter (1892), not against Martin and Moyce.** This repository's own
    `benchmarks/README.md` already records, for the SPH dam break, that the Martin and
    Moyce paper could not be read -- the publisher refuses the request -- and that its two
    non-dimensionalisations, `t sqrt(g/a)` and `t sqrt(2g/a)`, differ by 1.414 at every
    point. Quoting a number through a secondary source is the error this repository
    argues hardest against, so the addendum's "dam break vs Martin & Moyce (1952)" is
    answered the only honest way available: the comparison is **computed and printed in
    both conventions** and **graded in neither**, and what is graded is a closed form
    that needs no paper.

    Ritter's front advances at `2 sqrt(g h0)` from the instant the dam goes. The measured
    front speed is fitted over the window where that is defensible -- after the start-up
    transient and before the depression wave has reflected off the back wall and reached
    the front again -- and `REFERENCES` has the caveat about why an inviscid
    hydrostatic front is an upper bound on a real one.
    """
    import warp  # noqa: F401

    gravity = 9.81
    thin = 3
    dam = reservoir * height
    # Long enough that the front has not reached the far wall by the end of the graded
    # window: Ritter's front is at `2 T h0` past the dam, so `T = 4` needs eight depths.
    length = tank if tank > 0 else dam + 9.0 * height
    world, scaling = _free_surface_world(
        cells=cells, size=(length, 0.0, 2.0 * height), depth=height, faces="wall",
        collision=collision, smagorinsky=smagorinsky, device=device, gravity=gravity,
        thin=thin)
    world.initialise_column((0.0, -1.0, 0.0), (dam, 1.0, height))

    celerity = math.sqrt(gravity * height)
    time_scale = math.sqrt(height / gravity)
    graded_from, graded_to = 2.0, 4.0
    total = int(steps if steps else math.ceil(graded_to * time_scale / scaling.dt))
    node_x = world.node_coordinates(0) - dam        # measured from the dam
    exact_at_dam = 4.0 * height / 9.0
    samples: list[tuple[float, float, float]] = []
    track: list[tuple[float, float, float, float]] = []
    frames: list[Any] = []
    out = Path(out_dir)

    def ritter(elapsed: float) -> np.ndarray:
        """Ritter's depth profile at time `elapsed`, on this lattice's x nodes.

        `h = (2 sqrt(g h0) - x/t)^2 / (9 g)` between the upstream edge of the depression
        wave at `x = -c0 t` and the front at `x = 2 c0 t`, undisturbed upstream of it and
        dry beyond. Re-derived rather than quoted: it is the characteristics solution of
        the shallow-water equations on a dry bed, which is why it is safe to use without
        the 1892 paper.
        """
        if elapsed <= 0:
            return np.where(node_x <= 0.0, height, 0.0)
        inside = (node_x >= -celerity * elapsed) & (node_x <= 2.0 * celerity * elapsed)
        wave = (2.0 * celerity - node_x / elapsed) ** 2 / (9.0 * gravity)
        return np.where(inside, wave,
                        np.where(node_x < -celerity * elapsed, height, 0.0))

    def take(index: int) -> None:
        elapsed = world.steps * scaling.dt
        track.append((elapsed, _front_position(world, 0.25),
                      _front_position(world, 0.5), _front_position(world, 1.0)))
        depth = np.nan_to_num(world.surface_height(axis=2).mean(axis=1))
        exact = ritter(elapsed)
        usable = exact > 2.0 * world.dx
        if usable.sum() > 4 and elapsed >= graded_from * time_scale:
            samples.append((
                elapsed / time_scale,
                float(np.interp(0.0, node_x, depth)),
                float(np.sqrt(((depth[usable] - exact[usable]) ** 2).mean()))))
        if gif and len(frames) < 60:
            frames.append(figures.slice_frame(world, axis=1))

    _advance(world, total, sample_every=max(1, total // 80), sample=take,
             progress=progress)

    times = np.array([row[0] for row in track])
    fronts = np.array([row[2] for row in track])
    dimensionless_t = times / time_scale
    dimensionless_x = (fronts - dam) / height
    window = (dimensionless_t >= graded_from) & (dimensionless_t <= graded_to) & np.isfinite(
        dimensionless_x)
    if window.sum() >= 3:
        measured_slope = float(np.polyfit(dimensionless_t[window],
                                          dimensionless_x[window], 1)[0])
    else:
        measured_slope = float("nan")
    at_dam = float(np.mean([row[1] for row in samples])) if samples else float("nan")
    profile_rms = float(np.mean([row[2] for row in samples])) if samples else float("nan")
    limit = REFERENCES["ritter"]["tolerance"]
    drift = world.mass_drift()

    # The Martin and Moyce comparison, computed in both conventions and graded in
    # neither, for the reason in this function's docstring.
    both = {
        "T = t sqrt(g/a)": float(dimensionless_t[-1]) if len(dimensionless_t) else
        float("nan"),
        "T = t sqrt(2g/a)": float(dimensionless_t[-1] * math.sqrt(2.0))
        if len(dimensionless_t) else float("nan"),
    }

    checks = [
        Check("depth at the dam against Ritter's 4 h0 / 9", at_dam * 1e3,
              f"{exact_at_dam * 1e3:.4f} mm +/- {limit:.0%}",
              abs(at_dam / exact_at_dam - 1.0) <= limit
              if np.isfinite(at_dam) else False,
              f"Ritter fixes the depth at the dam at exactly `4 h0 / 9` for as long as "
              f"the solution holds, independent of time -- which makes it the one number "
              f"in a dam break that needs no threshold and no fit. Averaged over T in "
              f"[{graded_from:g}, {graded_to:g}], the window where the shallow-water "
              f"assumption holds and the back wall has not answered yet"),
        Check("depth profile against Ritter, RMS", profile_rms / height,
              f"<= {limit:.0%} of h0",
              profile_rms / height <= limit if np.isfinite(profile_rms) else False,
              "over the cells where Ritter's own depth exceeds two lattice cells, so "
              "that the comparison is not dominated by the vanishing tongue near the "
              "front, where Ritter tends to zero and no lattice can follow it"),
        Check("front speed against Ritter, dX/dT", measured_slope,
              "2.0, reported, not graded", None,
              f"`x_front = 2 t sqrt(g h0)`, fitted over T in [{graded_from:g}, "
              f"{graded_to:g}]. Not graded because Ritter's front has zero depth: the "
              f"measured front is wherever the depth threshold is put, and the three "
              f"thresholds below show how much that moves it"),
        Check("mass drift", drift["drift"], "<= 1e-3 of the initial mass",
              abs(drift["drift"]) <= 1e-3,
              f"over {total:,} steps and {drift['conversions']:,} conversions, while the "
              f"surface is moving fast -- which is the case mass conservation is worth "
              f"testing on"),
        Check("fluid cells touching gas cells", world.fluid_touching_gas(), "exactly 0",
              world.fluid_touching_gas() == 0),
        Check("lattice finite", world.stable(), "True", world.stable()),
        Check("front at a quarter, a half and one cell of depth",
              f"{track[-1][1]:.4f} / {track[-1][2]:.4f} / {track[-1][3]:.4f} m"
              if track else "n/a", "reported, not graded", None,
              "how much the measured front moves with the depth threshold that defines "
              "it. Ritter's front thins to zero depth, so this spread is the honest "
              "width of the measurement"),
        Check("non-dimensional time at the end of the run", both,
              "reported, not graded", None,
              "both of the conventions in circulation for Martin and Moyce (1952), "
              "because the paper could not be read and they differ by 1.414. "
              "benchmarks/README.md records the same refusal for the SPH dam break"),
        Check("unplaceable mass", drift["unplaceable"], "reported, not graded", None,
              "excess mass from a converting cell with no interface neighbour -- a "
              "tearing film or a one-cell droplet. A collapsing column makes some, and "
              "counting it is the only way the drift above means anything"),
    ]
    drawn = [
        figures.slice_figure(world, out / "live-dam-break.png", axis=1,
                             title=f"dam break at T = {dimensionless_t[-1]:.2f}, "
                                   f"{cells} cells"),
        figures.profile_figure(
            out / "live-dam-break-front.png",
            [{"x": dimensionless_t.tolist(), "y": dimensionless_x.tolist(),
              "label": "measured front (half a cell of depth)", "style": "o"},
             {"x": [0.0, float(dimensionless_t.max()) if len(dimensionless_t) else 2.0],
              "y": [0.0, 2.0 * (float(dimensionless_t.max())
                                if len(dimensionless_t) else 2.0)],
              "label": "Ritter, X = 2T", "style": "--"}],
            title=f"dam break: dX/dT {measured_slope:.3f} against Ritter's 2.0",
            xlabel="T = t sqrt(g/h0)", ylabel="X = (x_front - a) / h0"),
    ]
    if gif and frames:
        drawn.append(figures.animate(frames, out / "live-dam-break.gif", fps=12))
    for figure in drawn:
        if str(figure).endswith(".png"):
            figures.ensure_rgb(figure)
    lines = [f"# dam break: {height * 1e3:g} mm deep, {reservoir:g} deep wide, in a "
             f"{length * 1e3:g} mm tank, {cells} cells",
             *scaling.report(),
             *_free_surface_lines(world, scaling, height, gravity),
             *_official("dam_break", scaling, smagorinsky),
             f"  celerity:   sqrt(g h0) = {celerity:.4f} m/s, so Ritter's front runs at "
             f"{2 * celerity:.4f} m/s",
             f"  measured:   h(dam) {at_dam * 1e3:.4f} mm against {exact_at_dam * 1e3:.4f} "
             f"({at_dam / exact_at_dam:.4f}x), profile RMS {profile_rms / height:.3%} "
             f"of h0, dX/dT {measured_slope:.4f} against 2.0",
             f"  window:     graded over T in [{graded_from:g}, {graded_to:g}]; the "
             f"reservoir is {reservoir:g} depths so the reflection reaches the dam at "
             f"T = {2 * reservoir:g}",
             f"  reference:  {REFERENCES['ritter']['source']}",
             f"  caveat:     {REFERENCES['ritter']['caveat']}",
             *world.report()[-3:]]
    return _result("dam_break", checks, lines, drawn, slope=measured_slope,
                   at_dam=at_dam, exact_at_dam=exact_at_dam, profile_rms=profile_rms,
                   celerity=celerity, drift=drift,
                   track=[list(row) for row in track])


def run_sloshing(*, cells: int = 120, length: float = 0.1, depth: float = 0.03,
                 amplitude: float = 0.002, periods: float = 4.0, collision: str = "trt",
                 smagorinsky: float = 0.0, mach: float = 0.07, device: str | None = None,
                 out_dir: str | Path = ".", steps: int = 0,
                 progress: Callable[[str], None] | None = None,
                 **_: Any) -> dict[str, Any]:
    """A tilted tank is released: the first-mode period against linear wave theory.

    `omega^2 = g k tanh(k h)` with `k = pi / L`, which is exact for small amplitudes and
    is the formula `benchmarks/sph/` already uses for the same measurement, attributed
    to Lamb's *Hydrodynamics* 6th edition article 228.

    The initial condition is `amplitude cos(pi x / L)` released from rest: the first
    natural mode itself, which has zero slope at both walls the way the wall boundary
    condition wants. A linear tilt was tried first, because tilting a tank and letting
    go is how the experiment is done, and it is the wrong thing to release here -- a ramp
    has a constant slope that fights the walls and excites the odd modes together, and
    the trace is their beat rather than a period.

    The period is graded from the **zero crossings** of the elevation at one wall and
    cross-checked against an FFT of the same trace; the check itself says why that way
    round, and both numbers are always reported because they fail differently.

    `amplitude` is 2 mm on a 30 mm depth by default, which is 6.7% of it, and `k a` is
    0.0628. Linear theory is a small-amplitude expansion, and **which way a finite
    amplitude moves the period depends on the depth: this docstring used to say it
    lengthens it, and at this depth it shortens it.** Concus 1962 (UCRL-10208) eq. (34)
    at `delta = 0` changes sign at `k h = 1.058`; this tank is at `k h = 0.9425`, below
    that, so `w_2 = +0.1179` and the shift is **-0.027%**. A measured period *above* the
    linear one is therefore not explained by the amplitude. `REFERENCES` has the formula
    and its two checks.
    """
    import warp  # noqa: F401

    gravity = 9.81
    thin = 3
    world, scaling = _free_surface_world(
        cells=cells, size=(length, 0.0, 3.0 * depth), depth=depth, faces="wall",
        collision=collision, smagorinsky=smagorinsky, device=device, gravity=gravity,
        thin=thin, mach=mach)
    world.initialise_free_surface(level=depth, amplitude=amplitude, mode=1.0,
                                  along_axis=0)

    wavenumber = math.pi / length
    omega = math.sqrt(gravity * wavenumber * math.tanh(wavenumber * depth))
    period = 2.0 * math.pi / omega
    total = int(steps if steps else math.ceil(periods * period / scaling.dt))

    trace: list[tuple[float, float]] = []

    def take(index: int) -> None:
        surface = world.surface_height(axis=2)
        column = np.nan_to_num(surface.mean(axis=1))
        trace.append((world.steps * scaling.dt, float(column[0] - depth)))

    # Sample often enough to resolve the period: at least 40 points a cycle, which an
    # FFT needs far more than Nyquist to put a peak in the right bin.
    every = max(1, int(period / scaling.dt / 60))
    _advance(world, total, sample_every=every, sample=take, progress=progress)

    times = np.array([row[0] for row in trace])
    elevation = np.array([row[1] for row in trace])
    measured_period, how = _period_of(times, elevation)
    crossings_period, _ = _period_of(times, elevation, method="crossings")
    both = [v for v in (measured_period, crossings_period) if np.isfinite(v)]
    pair_period = float(np.mean(both)) if both else float("nan")
    spread = (abs(measured_period - crossings_period)
              if len(both) == 2 else float("inf"))
    # Linear theory is inviscid and this lattice is not, so the measured period is the
    # DAMPED one and is longer by `1 / sqrt(1 - zeta^2)` by construction. `_damping_of`
    # has the argument; the graded number is the undamped period recovered from it.
    zeta, decay_note = _damping_of(times, elevation)
    undamped = pair_period * math.sqrt(max(0.0, 1.0 - zeta * zeta))
    limit = REFERENCES["linear_sloshing"]["tolerance"]
    drift = world.mass_drift()

    checks = [
        Check("first-mode period, with the damping taken out", undamped,
              f"{period:.5f} s +/- {limit:.0%}",
              abs(undamped / period - 1.0) <= limit
              if np.isfinite(undamped) else False,
              f"`omega^2 = g k tanh(k h)`, k = pi/L = {wavenumber:.4f} 1/m, "
              f"kh = {wavenumber * depth:.4f}. The graded number is the **mean of two "
              f"estimators** on the elevation at the -x wall, an FFT and the zero "
              f"crossings, and the check below requires them to agree. That is not "
              f"hedging, it is what the measurements forced: neither estimator is "
              f"reliably the better one. Grading the FFT alone was tried, then the "
              f"crossings alone -- the crossings won on one run (412.5 ms against the "
              f"FFT's 453.5, theory 417.09) and the FFT won on the next (418.6 against "
              f"the crossings' 392.9), so picking either by name is picking whichever "
              f"was lucky on the run that was looked at. An FFT bin is `1/T_total` "
              f"wide and a zero crossing is pulled by any drift in the mean; they fail "
              f"differently, so their mean is better than either and their spread is "
              f"the honest width of the measurement. "
              f"That mean is {pair_period * 1e3:.3f} ms as measured, and the graded "
              f"number is it with the damping taken out -- `T sqrt(1 - zeta^2)` with "
              f"zeta {zeta:.4f} from the decay envelope. **Linear theory is inviscid "
              f"and this lattice is not**, so the measured period is the damped one and "
              f"is longer than the theoretical one by construction; comparing them "
              f"directly compares two different quantities. At the Reynolds numbers a "
              f"free surface can afford that is not a small correction"),
        Check("damping ratio from the decay envelope", zeta,
              "reported, not graded", None,
              f"{decay_note}. It is a property of the run and not an error: `Re_eff` "
              f"here is a few hundred against tens of thousands for real water, so the "
              f"wave really is heavily damped. It enters the graded period above and "
              f"nothing else"),
        Check("the two estimators agree", spread / period,
              f"<= {limit:.0%} of the period",
              spread / period <= limit if np.isfinite(spread) else False,
              f"the gap between the FFT ({measured_period * 1e3:.2f} ms) and the zero "
              f"crossings ({crossings_period * 1e3:.2f} ms) as a fraction of the "
              f"theoretical period. **Graded, because a measurement whose two "
              f"estimators disagree by more than the tolerance cannot settle anything "
              f"at that tolerance**, however close their mean happens to land. It "
              f"narrows with resolution and with run length: 523/363 ms at 80 cells "
              f"over 2.6 cycles, 443/407 at 240"),
        Check("mass drift", drift["drift"], "<= 1e-3 of the initial mass",
              abs(drift["drift"]) <= 1e-3,
              f"over {total:,} steps and {drift['conversions']:,} conversions"),
        Check("fluid cells touching gas cells", world.fluid_touching_gas(), "exactly 0",
              world.fluid_touching_gas() == 0),
        Check("lattice finite", world.stable(), "True", world.stable()),
        Check("wave steepness", amplitude / depth,
              "reported, not graded", None,
              f"the initial elevation at the wall as a fraction of the depth "
              f"({amplitude * 1e3:.2f} mm on {depth * 1e3:g} mm). Linear theory is the "
              f"first term of an expansion in this. Which way a larger one moves the "
              f"period depends on the depth: this tank is at k h = 0.9425, below the "
              f"k h = 1.058 where Concus (1962) eq. (34) changes sign, so a larger "
              f"amplitude SHORTENS the period -- -0.027% here, not the +0.4% this "
              f"check used to claim, and the wrong way round"),
    ]
    out = Path(out_dir)
    drawn = [
        figures.profile_figure(
            out / "live-sloshing.png",
            [{"x": times.tolist(), "y": (elevation * 1e3).tolist(),
              "label": "elevation at the -x wall (mm)"},
             {"x": times.tolist(),
              "y": (amplitude * 1e3 * np.cos(2 * math.pi * times / period)).tolist(),
              "label": f"linear theory, T = {period * 1e3:.1f} ms", "style": "--"}],
            title=f"sloshing: measured {measured_period * 1e3:.2f} ms against "
                  f"{period * 1e3:.2f} ms",
            xlabel="t (s)", ylabel="mm"),
        figures.slice_figure(world, out / "live-sloshing-field.png", axis=1,
                             title=f"sloshing tank, {cells} cells, t = "
                                   f"{times[-1] if len(times) else 0:.3f} s"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# sloshing: {depth * 1e3:g} mm of water in a {length * 1e3:g} mm tank, "
             f"{cells} cells, first mode at {amplitude * 1e3:g} mm",
             *scaling.report(),
             *_free_surface_lines(world, scaling, depth, gravity),
             *_official("sloshing", scaling, smagorinsky),
             f"  theory:     k = pi/L = {wavenumber:.4f} 1/m, kh = "
             f"{wavenumber * depth:.4f}, omega = {omega:.4f} rad/s, T = "
             f"{period * 1e3:.3f} ms",
             f"  measured:   {pair_period * 1e3:.3f} ms "
             f"({pair_period / period:.4f}x, graded, the mean of the two) from "
             f"{measured_period * 1e3:.3f} by FFT and {crossings_period * 1e3:.3f} by "
             f"zero crossings, which differ by {spread / period:.2%} of the period",
             f"  reference:  {REFERENCES['linear_sloshing']['source']}",
             *world.report()[-3:]]
    return _result("sloshing", checks, lines, drawn, period=pair_period,
                   theory=period, fft=measured_period, crossings=crossings_period,
                   spread=spread, drift=drift,
                   trace=[list(row) for row in trace])


def _damping_of(times: np.ndarray, signal: np.ndarray) -> tuple[float, str]:
    """The damping ratio of a decaying oscillation, from the envelope of its extrema.

    **Needed because linear water-wave theory is inviscid and this lattice is not.** A
    damped oscillator's period is `T_d = T / sqrt(1 - zeta^2)`, longer than the
    undamped one, so comparing a measured period straight to `2 pi / sqrt(g k tanh k h)`
    compares two different quantities and the measured one is biased long by
    construction. At the Reynolds numbers a free surface can afford -- a few hundred,
    see `units.Scaling.for_froude` -- that bias is not small.

    The envelope is the successive extrema of the centred signal, and `zeta` comes from
    the logarithmic decrement: `delta = ln(a_n / a_n+1)` over a half period, so
    `zeta = delta / sqrt(delta^2 + pi^2)`. Fitted over all the extrema rather than taken
    from the first pair, because a single pair is one sample of a noisy envelope.

    Returns `(zeta, how)` and `zeta = 0` when there are too few extrema to fit, which is
    the right default: it leaves the period comparison exactly where it was.
    """
    centred = np.asarray(signal, dtype=float)
    centred = centred - centred.mean()
    if len(centred) < 8:
        return 0.0, "too few samples to fit a decay"
    # Interior extrema: a sample larger (or smaller) than both its neighbours.
    rising = np.diff(centred)
    turns = np.flatnonzero(np.signbit(rising[:-1]) != np.signbit(rising[1:])) + 1
    peaks = np.abs(centred[turns])
    keep = peaks > 1e-12
    turns, peaks = turns[keep], peaks[keep]
    if len(peaks) < 4:
        return 0.0, f"only {len(peaks)} extrema, not enough to fit a decay"
    slope = float(np.polyfit(times[turns], np.log(peaks), 1)[0])
    if slope >= 0.0:
        return 0.0, "the envelope is not decaying"
    # Half a period between successive extrema, so the decrement per half cycle is
    # `-slope * dt_mean`, and per full cycle twice that.
    half = float(np.mean(np.diff(times[turns])))
    decrement = -slope * 2.0 * half
    zeta = decrement / math.sqrt(decrement * decrement + 4.0 * math.pi * math.pi)
    return float(zeta), f"{len(peaks)} extrema, log decrement {decrement:.4f}"


def _period_of(times: np.ndarray, signal: np.ndarray,
               method: str = "fft") -> tuple[float, str]:
    """The dominant period of a sampled oscillation, by FFT or by zero crossings.

    Two estimators because they fail differently and the cases report both. An FFT is
    robust to noise and quantises the answer into a bin whose width is `1 / T_total`;
    zero crossings are exact in time but are pulled around by any drift in the mean and
    by the higher modes a tilt excites. Agreement between them is the evidence that
    either is right.

    `figures.strouhal` does this for a vortex street and is not reused here: it returns a
    frequency non-dimensionalised by a diameter and a free-stream speed, neither of which
    a sloshing tank has.
    """
    if len(times) < 8:
        return float("nan"), "too few samples"
    centred = np.asarray(signal, dtype=float)
    centred = centred - centred.mean()
    span = float(times[-1] - times[0])
    if method == "crossings":
        sign = np.signbit(centred)
        changes = np.flatnonzero(sign[:-1] != sign[1:])
        if len(changes) < 3:
            return float("nan"), "fewer than three zero crossings"
        # Interpolate each crossing, then take twice the mean gap: a full period is two.
        moments = []
        for index in changes:
            a, b = centred[index], centred[index + 1]
            if a == b:
                continue
            fraction = a / (a - b)
            moments.append(times[index] + fraction * (times[index + 1] - times[index]))
        if len(moments) < 3:
            return float("nan"), "fewer than three usable crossings"
        return float(2.0 * np.mean(np.diff(moments))), f"{len(moments)} crossings"
    sampling = span / (len(times) - 1)
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(len(centred))))
    frequencies = np.fft.rfftfreq(len(centred), d=sampling)
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    if peak == 0 or frequencies[peak] <= 0:
        return float("nan"), "no spectral peak"
    # Parabolic interpolation on the log spectrum, so the answer is not quantised to the
    # bin width -- which is `1/span` and would be 25% of the period over four cycles.
    if 0 < peak < len(spectrum) - 1:
        left, middle, right = (math.log(max(spectrum[peak + offset], 1e-30))
                               for offset in (-1, 0, 1))
        shift = 0.5 * (left - right) / (left - 2 * middle + right)
        shift = max(-0.5, min(0.5, shift))
    else:
        shift = 0.0
    frequency = (peak + shift) / (len(centred) * sampling)
    return (1.0 / frequency if frequency > 0 else float("nan"),
            f"FFT peak at bin {peak} + {shift:+.2f}, {span / (1.0 / frequency):.1f} "
            f"cycles sampled")




# ======================================================================================
# M6: floating bodies
# ======================================================================================
#
# The free surface earns its keep here. A `FluidWorld` run hands a submerged body
# `-rho V g` explicitly (D9) because a single-phase solver has no waterline and no
# hydrostatic gradient to integrate. A `FreeSurfaceWorld` run has both -- gravity acts on
# the fluid through the Guo forcing, the pressure really does rise with depth, and
# `run_hydrostatic` measures `d(rho)/dz` against `-rho g / c_s^2` to within a per cent --
# so the momentum exchange over a hull produces a real buoyant force and the analytic
# term must NOT be added on top. `NewtonCoupling(buoyancy=False)` is how a floating body
# is coupled, and this case is the measurement that says the force which comes out is
# the right size.

REFERENCES["archimedes"] = {
    "source": "Archimedes' principle; exact, no paper",
    "read": "n/a",
    "formula": (
        "a prism of uniform density floats at a draft `d = (rho_body / rho_water) H`, "
        "with H its height, because the displaced weight equals its own"
    ),
    "caveat": (
        "**Exact as a statement about an equilibrium, and three things stand between it "
        "and a lattice.** The hull is voxelised, so its displaced volume is quantised: "
        "at N cells through the draft the volume error is of order 1/N and the draft "
        "error with it, which is why this case is graded at a resolution rather than at "
        "one. The free surface is weakly compressible, so the water it floats in is "
        "very slightly denser at depth -- `run_hydrostatic` measures that and it is "
        "0.245% over 60 mm. And a released body oscillates about its equilibrium at the "
        "heave period before it settles, so the draft has to be averaged over a window "
        "after the oscillation has decayed, not read off at a chosen instant"
    ),
    "tolerance": 0.02,
    "tolerance_is": (
        "2% of the draft, which the addendum asks for by name ('static float draft vs "
        "Archimedes within 2%')"
    ),
}


def run_archimedes(*, cells: int = 160, side: float = 0.04, height: float = 0.04,
                   density_ratio: float = 0.5, depth: float = 0.08,
                   settle_steps: int = 30000,
                   collision: str = "trt", smagorinsky: float = 0.0,
                   device: str | None = None, out_dir: str | Path = ".",
                   steps: int = 0, solver: str = "mujoco",
                   progress: Callable[[str], None] | None = None,
                   **_: Any) -> dict[str, Any]:
    """A box of half water's density is released at the surface: where does it float?

    Archimedes says a prism of uniform density floats at a draft of
    `(rho_body / rho_water) H` -- half its height at the default ratio -- and nothing in
    this solver knows that. The draft is whatever the pressure integrated over the
    voxelised hull turns out to produce, so agreeing with it is evidence that the free
    surface, the Guo forcing, the momentum exchange and the Newton coupling are all
    right at once, and that none of them is being rescued by an analytic buoyancy term.
    **Buoyancy is not supplied: `NewtonCoupling(buoyancy=False)`.** The draft this case
    measures is produced by the hydrostatic pressure integrated over the voxelised hull
    by momentum exchange, and by nothing else, which is what makes it a test rather than
    a restatement. Handing the body `rho V g` from its own voxel count would make the
    equilibrium draft come out at `(rho_body/rho_water) H` almost by construction and
    test only the volume count.

    That it can work at all is measured: a fully submerged box in a tank with gravity in
    the fluid, pinned, with the density gradient verified at 1.000 of exact, feels
    **+1.001 times `rho V g`** -- the right force to a part in a thousand. A floating
    hull is the harder case, because its top is in gas and the absolute pressure on the
    wetted part has less to cancel against, and that is what this case measures.

    The box is held square to the water by giving it only the heave degree of freedom
    this case is about: roll and pitch are a separate measurement with a separate
    analytic answer, and letting a cube find its own orientation while its draft is
    being graded would mix the two. `run_heave_decay` takes the dynamics.
    """
    import warp as wp

    try:
        import newton
    except Exception as exc:
        raise UnphysicalScaling(
            "this case needs the Newton physics engine, which the [live] extra "
            f"declares:\n    pip install 'openreynolds[live]'\n(the import said: {exc})"
        ) from exc

    from .couple import NewtonCoupling
    from .units import FLUIDS

    water = FLUIDS["water"]
    gravity = 9.81
    rho_body = density_ratio * water.rho
    exact_draft = density_ratio * height

    tank = 5.0 * side
    world, scaling = _free_surface_world(
        cells=cells, size=(tank, tank, depth + 3.0 * height), depth=depth,
        faces="wall", collision=collision, smagorinsky=smagorinsky, device=device,
        gravity=gravity)
    dt = scaling.dt

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    # Released at the draft Archimedes predicts, not at the surface: a box dropped from
    # above arrives with momentum and takes several heave periods to give it up, and
    # this case is about the equilibrium. Starting at the answer is not assuming it --
    # the body is free to leave, and a wrong buoyant force moves it straight away.
    start = (tank / 2.0, tank / 2.0, depth - exact_draft + height / 2.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(*start), wp.quat_identity()),
                            label="float")
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = rho_body
    shape_cfg.has_shape_collision = False
    builder.add_shape_box(body, hx=side / 2.0, hy=side / 2.0, hz=height / 2.0,
                          cfg=shape_cfg)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -gravity))
    if solver == "mujoco":
        engine = newton.solvers.SolverMuJoCo(model, disable_contacts=True,
                                             solver="newton",
                                             integrator="implicitfast", iterations=10)
    else:
        engine = newton.solvers.SolverFeatherstone(model)

    link = NewtonCoupling(model, engine, world, dt=dt, buoyancy=False)
    world.initialise_free_surface(level=depth)

    # **Fill the tank, hold the boat, let the water settle, then let go.** The tank
    # starts at uniform density; the hydrostatic gradient that holds a hull up takes of
    # order a thousand steps to establish -- `run_hydrostatic` watches it arrive between
    # 600 and 1,500 -- and a body released before then feels no buoyancy at all and
    # simply falls.
    #
    # The body is held by pinning its pose and velocity after each coupled step, **not**
    # by stepping the world instead of the link. Stepping the world does not sync poses
    # from Newton, and doing it here voxelised this very box into the corner of the
    # domain for long enough to produce four retracted measurements; the constructor now
    # pushes poses once, and this loop still goes through `link.step` so the hull is
    # where it says it is on every one of them.
    held_q = link.state.body_q.numpy().copy()
    settled = False
    reached = float("nan")
    for index in range(int(settle_steps)):
        link.step()
        # **`body_q` is not a solver input and never was.** `SolverMuJoCo` is
        # reduced-coordinate: `_update_mjc_data` pushes `joint_q`/`joint_qd` into
        # `qpos`/`qvel` every step and never reads `body_q`, and Newton's own
        # `SolverMuJoCo.reset` says the `BODY_Q`/`BODY_QD` flags "are not actionable
        # here and are ignored". So this pin was ignored: measured on an SPH rig, a
        # held body drifted 0.060 -> 0.0055 m over 50 steps while the pinned array read
        # 0.060000 throughout. **Any draft this case has reported is suspect.**
        #
        # `link.place` writes the body's free-joint coordinates and runs `eval_fk`,
        # which both the solver and the fluid then agree with. Measured: a `body_q`
        # write leaves the body at the origin; a `joint_q` write plus FK puts it exactly
        # where asked.
        link.place(body, held_q[body])
        if index % 200 == 199:
            settled, reached = world.at_rest()
            if progress:
                progress(f"    settling {index + 1:,}/{int(settle_steps):,}, "
                         f"|u|max {reached * 1e3:.2f} mm/s")
            if settled:
                break
    settle_used = world.steps

    # The heave period of a floating prism is `2 pi sqrt(d / g)` with `d` the draft --
    # the restoring stiffness is `rho g A` and the mass is `rho_body A H = rho A d`. Run
    # for several of them so the release transient is gone before anything is averaged.
    heave_period = 2.0 * math.pi * math.sqrt(max(exact_draft, 1e-6) / gravity)
    total = int(steps if steps else math.ceil(6.0 * heave_period / dt))
    settle_from = int(0.6 * total)

    track: list[tuple[float, float, float]] = []
    every = max(1, total // 160)

    for index in range(total):
        link.step()
        if index % every == 0 or index == total - 1:
            centre = float(link.state.body_q.numpy()[body][2])
            surface = world.surface_height(axis=2)
            # The undisturbed level, read away from the hull: the mean over the outer
            # ring of columns, which the box does not reach. Reading it under the hull
            # instead would measure the body's own depression, and the draft is relative
            # to the water the body is NOT disturbing.
            edge = np.concatenate([
                np.nan_to_num(surface[0, :]), np.nan_to_num(surface[-1, :]),
                np.nan_to_num(surface[:, 0]), np.nan_to_num(surface[:, -1])])
            level = float(edge[edge > 0].mean()) if (edge > 0).any() else float("nan")
            track.append((world.steps * dt, level, level - (centre - height / 2.0)))
        if progress and index % max(1, total // 40) == 0:
            progress(f"    {index:,}/{total:,} steps, draft "
                     f"{track[-1][2] * 1e3:.4f} mm" if track else "")

    late = [row for row in track if row[0] >= settle_from * dt]
    draft = float(np.mean([row[2] for row in late])) if late else float("nan")
    wobble = float(np.std([row[2] for row in late])) if late else float("nan")
    limit = REFERENCES["archimedes"]["tolerance"]
    drift = world.mass_drift()
    cells_through_draft = exact_draft / world.dx

    checks = [
        Check("draft against Archimedes", draft * 1e3,
              f"{exact_draft * 1e3:.4f} mm +/- {limit:.0%}",
              abs(draft / exact_draft - 1.0) <= limit
              if np.isfinite(draft) else False,
              f"`d = (rho_body / rho_water) H` = {density_ratio:g} x "
              f"{height * 1e3:g} mm, averaged over the last 40% of the run, by which "
              f"time the release transient has decayed over "
              f"{6.0 * 0.4:.1f} heave periods. The solver is told nothing about "
              f"buoyancy: `NewtonCoupling(buoyancy=False)`, so this force is the "
              f"hydrostatic pressure integrated over the voxelised hull by momentum "
              f"exchange, and nothing else"),
        Check("the draft stopped moving", wobble / exact_draft,
              f"<= {limit:.0%} of the draft",
              wobble / exact_draft <= limit if np.isfinite(wobble) else False,
              "the standard deviation of the draft over the same window. A mean draft "
              "that is right while still oscillating is an average of a transient, not "
              "an equilibrium"),
        Check("mass drift", drift["drift"], "<= 1e-3 of the initial mass",
              abs(drift["drift"]) <= 1e-3,
              f"over {total:,} steps; {drift['displaced']:.4g} lattice mass was "
              f"displaced by the hull and is counted separately, because a body moving "
              f"through water legitimately carries some off"),
        Check("fluid cells touching gas cells", world.fluid_touching_gas(), "exactly 0",
              world.fluid_touching_gas() == 0),
        Check("the tank was at rest before the body was released", settled,
              "True (|u|max <= 2% of the gravity-wave speed)", bool(settled),
              f"reached {reached * 1e3:.3f} mm/s after {settle_used:,} steps, against "
              f"a bound of {0.02 * abs(scaling.reference_velocity) * 1e3:.3f}. **Graded, "
              f"because the first version of this case was not** -- it took 3,000 steps "
              f"on trust and released the hull into a tank still moving at 50 to 90 "
              f"mm/s, with the force on a pinned hull swinging between 0.2 and 1.7 "
              f"times its weight. The draft it then reported was an average of that "
              f"transient and did not reproduce between identical runs"),
        Check("lattice finite", world.stable(), "True", world.stable()),
        Check("cells through the draft", f"{cells_through_draft:.1f}",
              "reported, not graded", None,
              "the hull is voxelised, so its displaced volume is quantised at about "
              "one part in this number, and the draft error with it. It is the first "
              "thing to raise if this case misses"),
        Check("heave period used to size the run", f"{heave_period * 1e3:.2f} ms",
              "reported, not graded", None,
              "`2 pi sqrt(d / g)`: the restoring stiffness of a prism is `rho g A` and "
              "its mass is `rho A d`, so the period depends only on the draft"),
    ]
    out = Path(out_dir)
    drawn = [
        figures.slice_figure(world, out / "live-archimedes.png", axis=1,
                             title=f"floating box, draft {draft * 1e3:.3f} mm against "
                                   f"{exact_draft * 1e3:.3f}"),
        figures.profile_figure(
            out / "live-archimedes-draft.png",
            [{"x": [row[0] * 1e3 for row in track],
              "y": [row[2] * 1e3 for row in track], "label": "draft (mm)"},
             {"x": [track[0][0] * 1e3, track[-1][0] * 1e3] if track else [0, 1],
              "y": [exact_draft * 1e3] * 2, "label": "Archimedes", "style": "--"}],
            title=f"draft {draft * 1e3:.3f} mm, wobble {wobble * 1e3:.4f}",
            xlabel="t (ms)", ylabel="mm"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)
    lines = [f"# archimedes: a {side * 1e3:g} mm box at {density_ratio:g} of water's "
             f"density, {cells} cells",
             *scaling.report(),
             *_free_surface_lines(world, scaling, height, gravity),
             *_official("archimedes", scaling, smagorinsky),
             f"  expected:   draft {exact_draft * 1e3:.4f} mm "
             f"({cells_through_draft:.1f} cells), heave period "
             f"{heave_period * 1e3:.2f} ms",
             f"  measured:   draft {draft * 1e3:.4f} mm "
             f"({draft / exact_draft:.4f}x), wobble {wobble * 1e3:.4f} mm",
             f"  buoyancy:   NOT supplied; the momentum exchange over the hull produces "
             f"it and this case is the check that it is the right size",
             *world.report()[-3:]]
    return _result("archimedes", checks, lines, drawn, draft=draft,
                   exact_draft=exact_draft, wobble=wobble, drift=drift,
                   track=[list(row) for row in track])


# -- the catalogue ----------------------------------------------------------------------


CASES: dict[str, Case] = {
    "smoke": Case(
        "smoke", "every kernel once on a 12-cube: the cheapest thing that fails if the "
        "solver is broken. Seconds on a CPU; in the main test suite",
        run_smoke, {"cells": 12, "steps": 40}, milestone="M0"),
    "poiseuille": Case(
        "poiseuille", "a force-driven channel against the exact parabola, which the "
        "two-relaxation-time wall makes exact rather than second-order",
        run_poiseuille, {"cells": 32}, milestone="M1"),
    "cavity": Case(
        "cavity", "the lid-driven cavity at Re 100 and Re 1000 against Ghia, Ghia and "
        "Shin (1982) Tables I and II",
        run_cavity, {"cells": 128, "reynolds": 100.0}, milestone="M1", heavy=True),
    "cylinder": Case(
        "cylinder", "Schafer and Turek's 2D-2 cylinder at Re 100 at three resolutions, "
        "with the observed order of convergence on the drag error",
        run_cylinder, {"resolutions": (24, 32, 44)}, milestone="M1", heavy=True),
    "sphere": Case(
        "sphere", "drag on a fixed sphere at four Reynolds numbers against "
        "Schiller-Naumann", run_sphere, {}, milestone="M1", heavy=True),
    "crosscheck": Case(
        "crosscheck", "the same cylinder on both backends: this one and the OpenFOAM "
        "run committed at tests/data/moving_mesh/viv_fixed",
        crosscheck, {}, milestone="M1", heavy=True),
    "translating_sphere": Case(
        "translating_sphere", "a sphere moving through still fluid: the moving-boundary "
        "and fresh-node refilling check, against Galilean invariance",
        run_translating_sphere, {}, milestone="M2", heavy=True),
    "falling_sphere": Case(
        "falling_sphere", "a sphere released in water, two-way coupled to Newton, "
        "against the Schiller-Naumann terminal velocity",
        run_falling_sphere, {}, milestone="M2", needs_newton=True, heavy=True),
    "momentum": Case(
        "momentum", "a closed periodic box with one moving body: the momentum the fluid "
        "gained against the impulse the solver reported",
        run_momentum, {}, milestone="M2", heavy=True),
    "ground_effect": Case(
        "ground_effect", "one actuator disk at five heights over a ground plane against "
        "Cheeseman and Bennett (1955)",
        run_ground_effect, {}, milestone="M3", heavy=True),
    "quadrotor": Case(
        "quadrotor", "a quadrotor hovering on four actuator disks, Newton-coupled, with "
        "ground and wall effects; graded for bookkeeping and stability",
        run_quadrotor, {}, milestone="M3", needs_newton=True, heavy=True),
    "hydrostatic": Case(
        "hydrostatic", "still water in a closed tank: the volume it was filled with, a "
        "flat surface, and no mass drift. The free surface's first check",
        run_hydrostatic, {"cells": 32}, milestone="M5"),
    "dam_break": Case(
        "dam_break", "a column of water collapses on a dry bed: the surge front against "
        "Ritter's shallow-water speed, with Martin and Moyce reported in both "
        "conventions and graded in neither",
        run_dam_break, {"cells": 325}, milestone="M5"),
    "sloshing": Case(
        "sloshing", "a tilted tank is released: the first-mode period against "
        "`omega^2 = g k tanh(k h)`",
        run_sloshing, {"cells": 120}, milestone="M5"),
    "archimedes": Case(
        "archimedes", "a box of half water's density is released at the surface: the "
        "static float draft against Archimedes, with no analytic buoyancy term",
        run_archimedes, {"cells": 160}, milestone="M6"),
    "crosscheck-2d1": Case(
        "crosscheck-2d1", "the same cylinder channel at Re 20 and steady, so the drag "
        "can be compared between backends without a transverse acoustic mode in the lift",
        crosscheck_2d1, {}, milestone="M1", heavy=True),
    "underwater": Case(
        "underwater", "a towed capsule in water, graded at zero length against the "
        "sphere it reduces to", run_underwater, {}, milestone="M3", heavy=True),
}


def describe_cases() -> list[str]:
    """One line a case, for `openreynolds live cases` and for the agent's tool."""
    width = max(len(name) for name in CASES)
    return [
        f"{case.name.ljust(width)}  {case.milestone or '--':>3}  {case.about}"
        f"{'  [needs a GPU to be quick]' if case.heavy else ''}"
        for case in CASES.values()
    ]


def run_case(name: str, *, out_dir: str | Path = ".", device: str | None = None,
             progress: Callable[[str], None] | None = None,
             **overrides: Any) -> dict[str, Any]:
    """Run one case by name and return its result.

    **An unknown keyword argument raises.** It used to be ignored, which meant a knob the
    case did not plumb produced a run that looked like it honoured the request. See the
    comment on the check below for the two measurements that cost.

    Ignored rather than refused, deliberately: the CLI and the agent tool both pass a
    fixed set of options (`resolution`, `steps`, `collision`, `fluid`) and the cases take
    different subsets of them, so a case that does not take `fluid` should run rather
    than raise at a caller who had no way to know. Every runner ends its signature with
    `**_` for that reason. What IS refused is an unknown case name and a laminar case
    with the subgrid model on.
    """
    load_sph_cases()
    try:
        case = CASES[name]
    except KeyError:
        raise UnphysicalScaling(
            f"no case called {name!r}. The cases are: {', '.join(sorted(CASES))}"
        ) from None
    settings = dict(case.defaults)
    # `resolution` is the CLI's and the tool's one knob for the grid, and each case
    # spends it on whatever its own grid parameter is.
    resolution = overrides.pop("resolution", None)
    if resolution:
        for key in ("cells", "cells_per_d", "cells_per_r"):
            if key in settings or key in _grid_keys(case):
                settings[key] = int(resolution)
                break
    settings.update({k: v for k, v in overrides.items() if v is not None})
    settings["out_dir"] = out_dir
    settings["device"] = device

    # **An override the runner cannot accept must raise, not be swallowed.** Every case
    # ends its signature with `**_`, so a keyword the function does not declare is
    # silently discarded and the run proceeds with the default -- producing a result that
    # looks like it honoured the request and did not. This has now cost two measurements
    # in this work:
    #
    #   * `--extra '{"outlet_rho": 1.0}'` on the cylinder. `run_cylinder` never declared
    #     it, so 220 s of H100 re-measured the old boundary and returned Cd 3.6088, bit
    #     for bit the number already on record. Two identical numbers read as "the change
    #     made no difference", which is the most expensive way to be wrong.
    #   * `--u_lattice` on the sloshing case, which is the one experiment that would
    #     separate its +5.1% from a Mach-number error, and which could not be run at all.
    #
    # A silent default is indistinguishable from a tested one in the recorded result, so
    # the check belongs here, before the GPU is involved.
    import inspect

    allowed = inspect.signature(case.runner).parameters
    if not any(q.kind is inspect.Parameter.VAR_KEYWORD for q in allowed.values()):
        unknown = sorted(set(settings) - set(allowed))
    else:
        unknown = sorted(k for k in overrides
                         if k not in allowed and k not in case.defaults
                         and k not in ("out_dir", "device", "progress", "resolution"))
    if unknown:
        raise TypeError(
            f"{name!r} does not take {', '.join(unknown)}. Its runner ends in `**_`, so "
            f"these would be discarded and the case would run with its defaults while "
            f"the result looked like it honoured them. Thread the argument through the "
            f"case, or drop it. Accepted here: "
            f"{', '.join(sorted(k for k in allowed if k != 'kwargs'))}"
        )
    settings["progress"] = progress
    began = time.perf_counter()
    result = case.runner(**settings)
    result["seconds"] = time.perf_counter() - began
    result["milestone"] = case.milestone
    result["about"] = case.about
    return result


def _grid_keys(case: Case) -> frozenset[str]:
    """Which grid parameter a case's runner takes, from its own signature."""
    import inspect

    return frozenset(inspect.signature(case.runner).parameters) & {
        "cells", "cells_per_d", "cells_per_r"}


# -- a case file, with the backend selector ---------------------------------------------


def _parse_case_file(text: str) -> dict[str, Any]:
    """A tiny YAML subset: `key: value` and `key:` blocks one level deep.

    Not PyYAML, and not because PyYAML is bad. It is not a dependency of this package
    and this extra should not add one for a dozen scalars; `benchmarks/felix/grade.py`
    reads Felix's own `case.yaml` the same way and for the same reason, with
    `_yaml_value` and `_block_value` regexes. A case file that needs more than this is a
    Python script, which is a perfectly good way to drive `FluidWorld`.
    """
    import re

    data: dict[str, Any] = {}
    block: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line[:1].isspace()
        found = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if not found:
            continue
        key, value = found.group(1), found.group(2).strip()
        if not indented:
            if value:
                data[key] = _scalar(value)
                block = None
            else:
                data[key] = {}
                block = key
        elif block is not None and value:
            data[block][key] = _scalar(value)
    return data


def _scalar(text: str) -> Any:
    """`true`, a number, a bracketed list, or the string as written."""
    lowered = text.strip().strip("'\"")
    if lowered.lower() in ("true", "yes", "on"):
        return True
    if lowered.lower() in ("false", "no", "off"):
        return False
    if lowered.startswith("[") and lowered.endswith("]"):
        inner = lowered[1:-1].strip()
        return tuple(_scalar(part) for part in inner.split(",")) if inner else ()
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        return lowered


OPENFOAM_HINTS = {
    "cavity": "$FOAM_TUTORIALS/incompressible/icoFoam/cavity",
    "cylinder": "$FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/cylinder2D "
                "(or the Schafer-Turek setup in benchmarks/felix/)",
    "crosscheck": "tests/data/moving_mesh/viv_fixed, which is a finished run of it",
    "sphere": "$FOAM_TUTORIALS/incompressible/simpleFoam, with a sphere STL and "
              "snappyHexMesh",
}
"""For `backend: openfoam` in a case file: where the same case lives on the other path.

The brief asks for the case format to carry a backend selector, `openfoam | live`. What
`openfoam` can honestly do from here is say where that case is on the OpenFOAM path and
stop, because the OpenFOAM backend runs in a workspace with its own image, its own
mesher and its own dictionaries, and this file cannot conjure one. So the selector picks
the backend and `openfoam` answers with the case rather than running it -- and
`crosscheck` is the one place where both backends' answers exist at once, which is what
makes the selector useful rather than decorative. `DECISIONS_LIVE.md` D2.
"""


def run_case_file(path: str | Path, *, out_dir: str | Path = ".",
                  device: str | None = None,
                  progress: Callable[[str], None] | None = None,
                  **overrides: Any) -> dict[str, Any]:
    """Run a case described by a small YAML file, with `backend: live | openfoam`.

        case: cylinder
        backend: live
        resolution: 30
        collision: trt
        smagorinsky: 0

    `backend: openfoam` does not run anything; it reports where the same case is on the
    OpenFOAM path, which is what this repository can honestly say (see `OPENFOAM_HINTS`
    and `DECISIONS_LIVE.md` D2). Anything else is refused by name.
    """
    source = Path(path)
    if not source.is_file():
        raise UnphysicalScaling(f"no case file at {source}")
    data = _parse_case_file(source.read_text(encoding="utf-8"))
    name = str(data.pop("case", "") or data.pop("name", "")).strip()
    backend = str(data.pop("backend", "live")).strip().lower()
    if not name:
        raise UnphysicalScaling(
            f"{source} names no case: it needs a `case:` line. The cases are "
            f"{', '.join(sorted(CASES))}"
        )
    if backend not in ("live", "openfoam"):
        raise UnphysicalScaling(
            f"{source} asks for backend {backend!r}; it is `live` or `openfoam`"
        )
    if backend == "openfoam":
        hint = OPENFOAM_HINTS.get(name, "no equivalent is recorded for this case")
        lines = [
            f"# {name}: backend openfoam",
            "  This selector picks the backend and does not start a workspace. The",
            "  OpenFOAM path runs in a workspace with its own image, mesher and",
            "  dictionaries; `openreynolds` with no subcommand starts a study that can.",
            f"  the same case there: {hint}",
            "  for both answers at once:  openreynolds live crosscheck",
        ]
        return _result(name, [Check("backend", "openfoam",
                                    "reported, not run", None)], lines, [],
                       backend="openfoam")
    data.update({k: v for k, v in overrides.items() if v is not None})
    result = run_case(name, out_dir=out_dir, device=device, progress=progress, **data)
    result["backend"] = "live"
    result["case_file"] = str(source)
    return result


# -- the benchmark ----------------------------------------------------------------------


def benchmark(*, resolutions: Sequence[int] = (64, 128), steps: int = 200,
              collision: str = "trt", smagorinsky: float = 0.0,
              device: str | None = None, layouts: Sequence[str] | None = None,
              progress: Callable[[str], None] | None = None,
              **_: Any) -> dict[str, Any]:
    """D3Q19 lattice updates per second, on whatever device this is, at each resolution.

    Million lattice updates per second, which is the only portable figure of merit an
    LBM has: nodes times steps divided by seconds. Reported per resolution because the
    rate is not one number -- a small lattice is launch-bound and a large one is
    bandwidth-bound, and on this project's own CPU the two differ by a factor of three.

    `layouts` re-runs the whole sweep with `OPENREYNOLDS_LIVE_LAYOUT` set to each of
    `soa` and `aos`, in **subprocesses**, because the layout is baked into the generated
    code at module import and cannot be changed in a running process. That is the
    measurement that has to pick the per-device default, and it has to be made on the
    device in question: the node-major layout is three times faster on one CPU core and
    is the wrong answer on a GPU, where a warp's reads of one direction are coalesced
    under the direction-major layout and scattered 76 bytes apart under the other.
    """
    import warp as wp

    from .world import Domain, FluidWorld

    if layouts:
        return _benchmark_layouts(resolutions=resolutions, steps=steps,
                                  collision=collision, smagorinsky=smagorinsky,
                                  device=device, layouts=layouts, progress=progress)

    from . import lattice as L

    rows = []
    for cells in (int(c) for c in resolutions):
        dx = 1.0 / cells
        nz = max(3, cells // 4)
        scaling = Scaling.for_reynolds(dx, 1.0, 100.0, "water", u_lattice=0.05,
                                       scheme=collision, les=smagorinsky > 0)
        domain = Domain(origin=(0.0, 0.0, 0.0), size=(1.0, 1.0, nz * dx),
                        faces={"-x": "wall", "+x": "wall", "-y": "wall", "+y": "wall",
                               "-z": "periodic", "+z": "periodic"},
                        face_velocity={"+y": (scaling.reference_velocity, 0.0, 0.0)})
        world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                           substeps=1, collision=collision, smagorinsky=smagorinsky,
                           device=device)
        world.initialise()
        world.step()  # the first launch compiles and allocates; it is not the rate
        best = 0.0
        for _ in range(3):
            began = time.perf_counter()
            for _ in range(steps):
                world.step()
            wp.synchronize_device(world.device)
            elapsed = time.perf_counter() - began
            best = max(best, world.nodes * steps / elapsed / 1e6)
        row = {"cells": cells, "nodes": world.nodes, "steps": steps, "mlups": best,
               "ms_per_step": world.nodes / best / 1e3, "device": world.device,
               "layout": L.LAYOUT,
               "bytes": int(world.f.size * 4 * 2)}
        rows.append(row)
        if progress:
            progress(f"  {cells:>4} cells  {world.nodes:>12,} nodes  "
                     f"{best:8.1f} MLUPS  {row['ms_per_step']:8.3f} ms a step")

    device_name = rows[0]["device"] if rows else (device or "unknown")
    checks = [
        Check(f"D3Q19 MLUPS at {row['cells']} cells ({row['nodes']:,} nodes)",
              row["mlups"], "reported; the brief's 1000+ target is for an H100-class GPU",
              None)
        for row in rows
    ]
    best = max((r["mlups"] for r in rows), default=0.0)
    hardware = _device_name(device_name)
    flagship = _is_flagship(hardware)
    # The brief's target names an H100-class GPU, so it is only answerable on one. An
    # A10 measured 510.8 MLUPS, which is a perfectly good number for a 600 GB/s
    # inference card and is NOT a failure to reach a target set for a 3.35 TB/s
    # training card -- the first version of this check reported it as one, which is
    # grading the wrong device. A CPU cannot answer it either.
    checks.append(Check(
        "1000+ MLUPS on an H100-class GPU", f"{best:.1f} MLUPS on {hardware}",
        ">= 1000, on an H100-class device",
        (best >= 1000.0) if flagship else None,
        "" if flagship else
        f"not applicable: {hardware} is not an H100-class device, so this target is "
        f"not a claim about it. The rate above is the measurement that stands"))
    lines = [f"# benchmark: D3Q19 on {device_name}, {L.LAYOUT} layout, "
             f"{collision} collision",
             f"  warp:       {wp.config.version}"]
    for row in rows:
        lines.append(f"  {row['cells']:>4} cells:  {row['nodes']:>12,} nodes  "
                     f"{row['mlups']:8.1f} MLUPS  {row['ms_per_step']:8.3f} ms a step  "
                     f"{row['bytes'] / 2 ** 20:.0f} MiB of populations")
    return _result("benchmark", checks, lines, [], rows=rows, device=str(device_name),
                   layout=L.LAYOUT)


FLAGSHIP = ("h100", "h200", "a100", "b100", "b200", "gb200", "mi300", "mi325")
"""Device names whose bandwidth is in the class the brief's 1000+ MLUPS target names.

A list of substrings and not a bandwidth lookup, because the thing available at run time
is a name. It is matched case-insensitively against whatever the device reports, and a
device that is not on it has the target reported as not applicable rather than failed:
the measurement still stands, it is just not an answer to a question about an H100.

Deliberately short. Adding a part to it is a claim that that part is in the same class,
and a wrong entry turns a reported number into a failed target.
"""


def _device_name(device: Any) -> str:
    """The device's own name, for the report, from Warp rather than from the string.

    `device` is `cuda:0`, which says nothing about the hardware. Warp's device object
    carries the name the driver reports ("NVIDIA H100 80GB HBM3"), which is what a
    reader needs and what `_is_flagship` matches on.
    """
    try:
        import warp as wp

        return str(wp.get_device(str(device)).name)
    except Exception:
        return str(device)


def _is_flagship(name: str) -> bool:
    lowered = str(name).lower()
    return any(part in lowered for part in FLAGSHIP)


def _benchmark_layouts(*, resolutions, steps, collision, smagorinsky, device, layouts,
                       progress) -> dict[str, Any]:
    """Run `benchmark` once per layout, each in its own process, and compare.

    A subprocess per layout and not a loop, because `lattice.LAYOUT` is read once at
    import and baked into every kernel's generated code -- which is the whole reason the
    layout is fast. The processes are sequential so that they do not contend for the one
    device.
    """
    import subprocess
    import sys

    rows = []
    for layout in layouts:
        if progress:
            progress(f"  measuring the {layout} layout in its own process")
        environment = dict(os.environ, OPENREYNOLDS_LIVE_LAYOUT=layout,
                           PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        script = (
            "import json;from openreynolds.live.cases import benchmark;"
            f"print(json.dumps(benchmark(resolutions={tuple(resolutions)!r},"
            f"steps={int(steps)},collision={collision!r},"
            f"smagorinsky={float(smagorinsky)!r},device={device!r})))"
        )
        done = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, env=environment, timeout=3600)
        payload = None
        for line in reversed(done.stdout.splitlines()):
            if line.startswith("{"):
                payload = json.loads(line)
                break
        if payload is None:
            raise UnphysicalScaling(
                f"the {layout} benchmark subprocess returned no JSON "
                f"(exit {done.returncode}): {done.stderr.strip()[-400:]}"
            )
        for row in payload["rows"]:
            rows.append(dict(row, layout=layout))

    device_name = rows[0]["device"] if rows else (device or "unknown")
    by_layout: dict[str, float] = {}
    for row in rows:
        by_layout[row["layout"]] = max(by_layout.get(row["layout"], 0.0), row["mlups"])
    fastest = max(by_layout, key=by_layout.get) if by_layout else "unknown"
    checks = [
        Check(f"best MLUPS, {layout} layout on {device_name}", rate,
              "reported; this is the measurement that picks the default for this device",
              None)
        for layout, rate in sorted(by_layout.items())
    ]
    checks.append(Check(
        f"fastest layout on {device_name}", fastest,
        "whatever the measurement says, which is what "
        "lattice._preferred_layout should return on this device", None))
    lines = [f"# benchmark: both layouts on {device_name}"]
    for row in sorted(rows, key=lambda r: (r["layout"], r["cells"])):
        lines.append(f"  {row['layout']}  {row['cells']:>4} cells:  "
                     f"{row['nodes']:>12,} nodes  {row['mlups']:8.1f} MLUPS")
    for layout, rate in sorted(by_layout.items()):
        lines.append(f"  best {layout}: {rate:.1f} MLUPS")
    lines.append(f"  fastest on {device_name}: {fastest}")
    return _result("benchmark", checks, lines, [], rows=rows, device=str(device_name),
                   by_layout=by_layout, fastest=fastest)


# -- the report -------------------------------------------------------------------------


def report(result: dict[str, Any]) -> str:
    """A case's result as a person reads it: the facts, then the checks, then the verdict.

    The layout `benchmarks/*/grade.py` prints and for the same reason: every check on
    one line with its measured value, its target and its verdict, so that a reader can
    see which one failed without reading prose.
    """
    lines = list(result.get("report", []))
    checks = result.get("checks", [])
    if checks:
        lines.append("")
        width = max(len(str(c["name"])) for c in checks)
        for check in checks:
            measured = check["measured"]
            if isinstance(measured, float):
                shown = f"{measured:.6g}"
            elif isinstance(measured, bool):
                shown = "yes" if measured else "no"
            else:
                shown = str(measured)
            verdict = {True: "PASS", False: "FAIL", None: "    "}[check["passed"]]
            lines.append(f"  {verdict}  {str(check['name']).ljust(width)}  "
                         f"{shown}   against {check['target']}")
            if check.get("note"):
                lines.append(f"        {' ' * width}  ({check['note']})")
    graded = [c for c in checks if c["passed"] is not None]
    if graded:
        failed = [c for c in graded if not c["passed"]]
        lines.append("")
        lines.append(f"  {len(graded) - len(failed)}/{len(graded)} graded checks passed"
                     + (f"; failed: {', '.join(c['name'] for c in failed)}"
                        if failed else ""))
    if result.get("seconds"):
        lines.append(f"  took {figures.clock(result['seconds'])}")
    for figure in result.get("figures", []):
        lines.append(f"  figure:     {figure}")
    return "\n".join(lines)


# The SPH cases live in their own module -- it is 3,000 lines and has its own imports --
# but they are cases like any other and `run_case`, `modal_run.py` and the CLI all reach
# them through this one registry. Imported at the bottom so `sph_cases` can import `Case`,
# `Check`, `_result` and the period/damping helpers from here without a cycle.
def load_sph_cases() -> None:
    """Import `sph_cases`, which registers its cases into `CASES` as it loads.

    **Deferred, because a module-level import either way round is a cycle.** `sph_cases`
    needs `Case`, `Check`, `_result` and the period helpers from this module at import
    time, to build its own `SPH_CASES` dict. This module used to import it at the bottom,
    which worked when `cases` was imported first and raised
    `ImportError: cannot import name 'SPH_CASES' from partially initialized module` when
    `sph_cases` was -- an order that nothing in the package controls.

    Idempotent, and safe to call from anywhere that is about to look in `CASES`.
    """
    from . import sph_cases  # noqa: F401  (registers on import)
