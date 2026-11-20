"""The MHD journey: template -> blockMesh -> mhdFoam -> sample -> graded.

The third test in this repository that actually solves something, after
`test_journey_duct.py` and `test_journey_thermal_stress.py`, and it exists for the same
reason they do: every other test of this capability passes on a laptop with no OpenFOAM
on it. `test_toolbox_templates.py` parses the template and
`test_benchmark_mhd.py` feeds the grader profiles written from the closed form. Neither
can tell you that `mhdFoam` is in the image at all -- nothing in this repository
referenced it before this item -- that the `BPISO` dictionary it demands is the one the
template writes, or that the profile it produces is the right one.

the case, and why its answer is known exactly

`openreynolds/toolbox/templates/mhd_hartmann.py` as it ships: a channel two half-heights
across and twenty long, one cell thick in z, graded toward both walls, with a uniform
magnetic field across it. In a transverse field the Lorentz force on a fully developed
flow reduces to a linear drag, so the momentum equation is the Poiseuille equation with
a `-k*u` in it and its solution is a constant minus a cosh. The derivation is in the
template's docstring and in `benchmarks/mhd/README.md`, and it starts from `mhdFoam`'s
own two equations:

    u(y)/u_c   = ( cosh(Ha) - cosh(Ha*y/L) ) / ( cosh(Ha) - 1 )
    u_c/u_mean = ( cosh(Ha) - 1 ) / ( cosh(Ha) - sinh(Ha)/Ha )
    Ha         = B0*L*sqrt(sigma/(rho*nu))

1.23312791 at Ha 5 and 1.5 at Ha 0. No published value, no reference uncertainty,
nothing to misquote: `Ha` is the case's own and the algebra is beside the grader.

Two Hartmann numbers are run, 5 and 0, and the second one is the point. At Ha 0 the field
is switched off and the answer has to be the parabola; at Ha 5 the same mesh, the same
schemes and the same solver have to produce a profile 0.176 RMS away from that parabola
with a centreline 17.8% closer to the mean. A single run agreeing with a cosh is a weaker
statement than two runs 18% apart agreeing with two different formulae, and
`benchmarks/lbm/` makes the same argument about its one-relaxation-time contrast.

**Ha 20 is not run, and that is measured rather than chosen.** It is the ESI tutorial's
value and this test did run it, on a mesh graded to resolve its 0.05 L layer. `mhdFoam`
diverged: a max Courant number of 23.6 inside eighty steps and a SIGFPE in
`symGaussSeidelSmoother` on the vector equation, because the solver differences its
magnetic stresses explicitly and the spurious part of them grows as the cell shrinks --
the solver's own `Description` block says those stresses do not cancel discretely.
Resolving `L/Ha` forces `dy` down as `1/Ha` while `c_A` rises as `Ha`, so the step falls
as `1/Ha^2` and the work goes as `Ha^3`: Ha 20 is sixty-four times Ha 5, which is more
than this whole job's budget. `mhd_hartmann.py`'s docstring and
`benchmarks/mhd/README.md` carry the evidence and the stability bracket.

what is asserted at each seam, and not at the end

* `blockMesh` built the mesh the template describes, with its cell count and the two z
  faces `empty` -- a "2D" case that is really a thin 3D one has a different answer
  rather than a failure;
* the **Courant number stayed below one**, and `c_A*dt/dy` inside the measured bracket.
  These are the two checks that would have caught the divergence above on the run that
  produced it, rather than leaving a SIGFPE to be read backwards from a stack trace;
* `checkMesh` says the mesh is OK, which on a graded box it has every right to;
* the solver reached `End`, and its own `magnetic flux divergence error` line did not
  grow across the run. That line is `magneticFieldErr.H` reporting
  `deltaT*weightedAverage(mag(div(phiB)))`, and a growing one is the `pB` projection
  failing rather than a physics result;
* the profile stopped moving, by sampling the last two written times and differencing
  them. A transient that has not finished looks exactly like a converged answer in a
  single output, which is the failure this separates out;
* the centreline speed is `u_mean*(3)` **absolutely**, asserted here against a formula
  written out in this file rather than imported from the template or the grader;
* the shape is the closed form, and at Ha 5 it is very much further from the parabola
  than from the cosh;
* and `benchmarks/mhd/grade.py` agrees, through its own command line, with the exit
  status a person would read.

Skipped where there is no OpenFOAM. The place this runs is the `journey` job in
`.github/workflows/tests.yml`, inside the workspace image.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
TEMPLATE = ROOT / "openreynolds" / "toolbox" / "templates" / "mhd_hartmann.py"
GRADER = ROOT / "benchmarks" / "mhd" / "grade.py"

BINARIES = ("blockMesh", "checkMesh", "mhdFoam", "postProcess")


def _absent() -> list[str]:
    return [name for name in BINARIES if shutil.which(name) is None]


pytestmark = pytest.mark.skipif(
    bool(_absent()),
    reason="not an OpenFOAM machine, so no case can be solved here: "
           + ", ".join(_absent() or ["-"]) + " missing",
)


# -- the case, exactly as the template defines it ------------------------------------

HALF_HEIGHT_M = 1.0          # mhd_hartmann.py  HALF_HEIGHT_M
MEAN_VELOCITY = 1.0          # mhd_hartmann.py  MEAN_VELOCITY_M_S
CELLS_ALONG = 100            # mhd_hartmann.py  CELLS_ALONG
CELLS_ACROSS = 80            # mhd_hartmann.py  CELLS_ACROSS
END_TIME_S = 2.0             # mhd_hartmann.py  END_TIME_S
WRITE_INTERVAL_S = 0.5       # mhd_hartmann.py  WRITE_INTERVAL_S

# c_A*dt/dy, the ratio the graded mesh got wrong: it ran at 15.8 and diverged, while the
# ESI tutorial runs at 2.0. `mhd_hartmann.py`'s MAGNETIC_CFL asks for 0.5.
MAGNETIC_COURANT_LIMIT = 0.75


def delta_t(case: Path) -> float:
    """The step out of the case the template wrote.

    The template computes it from the mesh and the field rather than carrying a
    constant, so this test reads it back instead of duplicating the formula -- which
    also means a change to `delta_t_s()` cannot silently desynchronise the two.
    """
    text = (case / "system" / "controlDict").read_text(encoding="utf-8")
    return float(re.search(r"^deltaT\s+([\d.eE+-]+);", text, re.M).group(1))

HARTMANN_NUMBERS = (5.0, 0.0)


def centreline_over_mean(hartmann: float) -> float:
    """Equation (3), written out here rather than imported.

    A test that gets its expected value from the code under test asserts only that the
    code is self-consistent -- the rule `test_journey_thermal_stress.py` states for the
    same reason.
    """
    if hartmann == 0.0:
        return 1.5
    cosh_ha = math.cosh(hartmann)
    return (cosh_ha - 1.0) / (cosh_ha - math.sinh(hartmann) / hartmann)


def shape(eta: np.ndarray, hartmann: float) -> np.ndarray:
    if hartmann == 0.0:
        return 1.0 - eta * eta
    cosh_ha = math.cosh(hartmann)
    return (cosh_ha - np.cosh(hartmann * eta)) / (cosh_ha - 1.0)


def run(cmd: list[str], cwd: Path, label: str = "") -> tuple[str, float]:
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    print(f"  {label or cmd[0]}: {elapsed:.1f} s, exit {proc.returncode}")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        pytest.fail(f"{cmd} failed (exit {proc.returncode})")
    return proc.stdout, elapsed


def edited_copy(root: Path, hartmann: float) -> Path:
    """The template with HARTMANN_NUMBER edited, in a directory shaped like the toolbox.

    Editing a constant at the top and re-running it is the whole interface a template
    has, so the test does that rather than importing it and passing an argument. The
    copy cannot go just anywhere: `find_mesh_look` falls back to
    `Path(__file__).parent.parent / "mesh_look.py"`, so the two-deep layout is
    reproduced with `mesh_look.py` beside it -- the same thing
    `test_journey_thermal_stress.py:edited_copy` does and for the same reason.
    """
    toolbox = root / f"toolbox_ha{hartmann:g}"
    (toolbox / "templates").mkdir(parents=True)
    shutil.copy(ROOT / "openreynolds" / "toolbox" / "mesh_look.py",
                toolbox / "mesh_look.py")
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "HARTMANN_NUMBER = 5.0" in text, "the template's knob has been renamed"
    text = text.replace("HARTMANN_NUMBER = 5.0",
                        f"HARTMANN_NUMBER = {hartmann!r}", 1)
    script = toolbox / "templates" / "mhd_hartmann.py"
    script.write_text(text, encoding="utf-8")
    return script


def read_profile(case: Path, time_name: str) -> tuple[np.ndarray, np.ndarray]:
    """`(y, Ux)` out of one written time's sampled profile.

    Read here as well as in the grader, because the steadiness check is a comparison
    between two times and the grader only ever looks at the newest.
    """
    where = case / "postProcessing" / "sample" / time_name
    # Globbed rather than named: how the `raw` set writer joins a set name and its
    # fields into a filename is its decision, and a test that hard-codes the spelling
    # fails on a change that is not a change in the physics.
    found = sorted(where.glob("*.xy")) if where.is_dir() else []
    assert len(found) == 1, f"{where} holds {[p.name for p in found]}, wanted one .xy"
    path = found[0]
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    data = np.array([[float(value) for value in row] for row in rows], dtype=float)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def build_and_solve(case: Path, hartmann: float) -> dict:
    """The template, run as a person runs it, then the solver and the sampler."""
    case.mkdir(parents=True)
    script = edited_copy(case.parent, hartmann)

    print(f"\n[Ha = {hartmann:g}] {case}")
    template_log, _ = run([sys.executable, str(script)], case, "template")
    solve_log, seconds = run(["mhdFoam"], case, "mhdFoam")
    run(["postProcess", "-func", "sample", "-noZero"], case, "sample")

    cells = CELLS_ALONG * CELLS_ACROSS
    steps = int(round(END_TIME_S / delta_t(case)))
    print(f"  cost: {cells:,} cells, {steps:,} steps, {seconds:.1f} s "
          f"({1e6 * seconds / (cells * steps):.2f} us per cell-step)")
    return {"template": template_log, "solve": solve_log, "seconds": seconds,
            "cells": cells, "steps": steps}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[float, dict]:
    root = tmp_path_factory.mktemp("mhd")
    out = {}
    for hartmann in HARTMANN_NUMBERS:
        case = root / f"ha{hartmann:g}"
        result = build_and_solve(case, hartmann)
        result["case"] = case
        out[hartmann] = result
    return out


# -- the seams ---------------------------------------------------------------------


def test_block_mesh_built_the_two_dimensional_case_the_template_describes(runs):
    """`empty` on the z faces is what makes this 2D. A pair of plain `patch` faces
    there is a thin 3D case with two free-slip walls a cell apart, which has a
    different answer rather than a failure."""
    case = runs[5.0]["case"]
    boundary = (case / "constant" / "polyMesh" / "boundary").read_text()
    index = boundary.index("frontAndBack")
    assert "empty" in boundary[index:index + 200], "the case is a thin 3D one, not 2D"
    for patch in ("lowerWall", "upperWall"):
        index = boundary.index(patch)
        assert "wall" in boundary[index:index + 200], f"{patch} is not a wall patch"

    log = runs[5.0]["template"]
    assert str(CELLS_ALONG * CELLS_ACROSS) in log, (
        "neither blockMesh nor checkMesh reported the cell count the mesh implies")
    assert "Mesh OK" in log, "checkMesh did not pass on a uniform box"


def test_the_template_resolved_the_hartmann_layer_rather_than_straddling_it(runs):
    """The one mesh fact that decides whether the case can be graded at all: the layer
    is `L/Ha` thick, 0.2 m at Ha 5, and the uniform cell has to fit several times inside
    it. The ESI tutorial's own 40-cell mesh at Ha 20 fits **one**, which is what made
    the first version of this template reach for grading and diverge."""
    log = runs[5.0]["template"]
    assert "cells in it" in log
    line = next(line for line in log.splitlines() if "cells in it" in line)
    count = float(line.split()[3])
    print(f"  {line.strip()}")
    assert count >= 4.0, f"only {count} cells inside the Hartmann layer"
    assert "WARNING" not in log


def test_the_courant_number_stayed_below_one(runs):
    """The check that would have caught the graded mesh's divergence on the run that
    produced it. `CourantNo.H` prints a line per step, and the advective Courant number
    on this mesh is about 0.015, so anything approaching 1 is not the flow: it is the
    explicitly differenced magnetic stresses growing. The graded run reached 23.6 before
    `symGaussSeidelSmoother` took a SIGFPE."""
    for hartmann, result in runs.items():
        peaks = [float(line.rsplit("max:", 1)[1])
                 for line in result["solve"].splitlines()
                 if line.startswith("Courant Number")]
        assert peaks, f"Ha {hartmann:g}: the solver printed no Courant number"
        print(f"  Ha {hartmann:g}: Courant max over {len(peaks)} reported steps is "
              f"{max(peaks):.4f}")
        assert max(peaks) < 1.0


def test_the_magnetic_courant_number_is_inside_the_measured_bracket(runs):
    """`c_A*dt/dy`, out of the case's own dictionaries rather than the template's
    constants. 15.8 diverged and the ESI tutorial runs at 2.0, so this is the number to
    look at first if a run of this ever blows up again."""
    for hartmann, result in runs.items():
        case = result["case"]
        field = (case / "0" / "B").read_text(encoding="utf-8")
        applied = float(re.search(
            r"internalField\s+uniform\s+\(\S+\s+(\S+)\s+\S+\)", field).group(1))
        # rho and mu are 1 in this case, so c_A is B0 numerically. Read them anyway:
        # they are what the ratio is made of, and a template that changed them would
        # otherwise be graded against the wrong wave speed.
        properties = (case / "constant" / "transportProperties").read_text(
            encoding="utf-8")
        rho = float(re.search(r"^rho\s+(\S+);", properties, re.M).group(1))
        permeability = float(re.search(r"^mu\s+(\S+);", properties, re.M).group(1))
        speed = applied / math.sqrt(permeability * rho)
        cell = 2.0 * HALF_HEIGHT_M / CELLS_ACROSS
        ratio = speed * delta_t(case) / cell
        print(f"  Ha {hartmann:g}: B0 {applied:g}, c_A {speed:g} m/s, dt "
              f"{delta_t(case):g} s, c_A*dt/dy {ratio:.4f}")
        assert ratio < MAGNETIC_COURANT_LIMIT


def test_the_magnetic_flux_error_did_not_grow(runs):
    """`magneticFieldErr.H` prints `magnetic flux divergence error = ...` every B-PISO
    pass, which is `deltaT*weightedAverage(mag(div(phiB)))`. The `pB` projection exists
    only to keep it near zero, and a growing one means the projection is failing --
    which a converged-looking velocity field will not tell you."""
    for hartmann, result in runs.items():
        errors = []
        for line in result["solve"].splitlines():
            if "magnetic flux divergence error" in line:
                errors.append(float(line.rsplit("=", 1)[1]))
        assert errors, f"Ha {hartmann:g}: the solver printed no magnetic flux error"
        first, last = max(errors[:10]), max(errors[-10:])
        print(f"  Ha {hartmann:g}: magnetic flux error {first:.3e} early, "
              f"{last:.3e} late over {len(errors)} passes")
        assert last <= max(10.0 * first, 1e-8), "the div(B) projection is diverging"


def test_both_runs_reached_the_end(runs):
    for hartmann, result in runs.items():
        assert "End" in result["solve"], f"Ha {hartmann:g} did not finish"


def test_the_profile_stopped_moving(runs):
    """Two viscous times at nu = 1, differenced across the last written half-second.

    A single output cannot say whether it is converged, and a profile still drifting
    toward the closed form agrees with it to whatever tolerance the drift happens to
    have reached when the clock ran out.
    """
    last = f"{END_TIME_S:g}"
    previous = f"{END_TIME_S - WRITE_INTERVAL_S:g}"
    for hartmann, result in runs.items():
        y_last, u_last = read_profile(result["case"], last)
        y_prev, u_prev = read_profile(result["case"], previous)
        assert np.allclose(y_last, y_prev), "the two times sampled different points"
        drift = float(np.max(np.abs(u_last - u_prev))) / MEAN_VELOCITY
        print(f"  Ha {hartmann:g}: max drift between t = {previous} and {last} is "
              f"{drift:.3e} of u_mean")
        assert drift < 2e-3


@pytest.mark.parametrize("hartmann", HARTMANN_NUMBERS)
def test_the_centreline_speed_is_the_closed_form_absolutely(hartmann, runs):
    """The strongest single number here, because nothing is fitted: the inlet is a plug
    at u_mean, the case is 2D so continuity fixes the mean at every station, and (3)
    then gives the centreline speed outright. 1.23312791 m/s at Ha 5 against
    1.5 m/s at Ha 0, which is the 17.8% flattening the field is responsible for."""
    case = runs[hartmann]["case"]
    y, ux = read_profile(case, f"{END_TIME_S:g}")
    centre = float(np.interp(0.0, y, ux))
    expected = MEAN_VELOCITY * centreline_over_mean(hartmann)
    print(f"  Ha {hartmann:g}: u_c {centre:.6f} m/s against an exact {expected:.8f} "
          f"({abs(centre - expected) / expected:+.3%})")
    assert centre == pytest.approx(expected, rel=2e-2)


@pytest.mark.parametrize("hartmann", HARTMANN_NUMBERS)
def test_the_shape_is_the_closed_form_and_not_the_other_one(hartmann, runs):
    """Both profiles against both formulae, which is what makes either one evidence.

    At Ha 5 the run has to be close to the cosh and far from the parabola; at Ha 0,
    where the two coincide, the parabola IS the closed form and the contrast does not
    exist, so only the first half is asserted.
    """
    case = runs[hartmann]["case"]
    y, ux = read_profile(case, f"{END_TIME_S:g}")
    eta = y / HALF_HEIGHT_M
    centre = float(np.interp(0.0, y, ux))
    own = float(np.sqrt(np.mean((ux / centre - shape(eta, hartmann)) ** 2)))
    other = float(np.sqrt(np.mean((ux / centre - shape(eta, 0.0)) ** 2)))
    print(f"  Ha {hartmann:g}: RMS {own:.6f} from its own closed form, "
          f"{other:.6f} from the Ha 0 parabola")
    assert own < 1.5e-2
    if hartmann >= 5.0:
        assert other > 10.0 * own, (
            "the run is no further from the parabola than from the cosh, so agreeing "
            "with the cosh says nothing")


@pytest.mark.parametrize("hartmann", HARTMANN_NUMBERS)
def test_the_benchmark_grader_agrees_through_its_own_command_line(hartmann, runs):
    """The grader as a person runs it, exit status and all. The assertions above are
    this test's own reading of the same file; this is the published grader's."""
    case = runs[hartmann]["case"]
    proc = subprocess.run(
        [sys.executable, str(GRADER), "hartmann", str(case),
         "--hartmann", f"{hartmann:g}", "--mean-velocity", f"{MEAN_VELOCITY:g}",
         "--half-height", f"{HALF_HEIGHT_M:g}"],
        text=True, capture_output=True)
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout


def test_the_grader_rejects_each_run_graded_as_the_other(runs):
    """The two runs cross-graded. A grader that passed a Ha 0 run against Ha 5 would
    be measuring nothing, and this is the cheapest way to show it does not -- the two
    cases differ only in one number at the top of the template."""
    for solved, graded_as in ((0.0, 5.0), (5.0, 0.0)):
        proc = subprocess.run(
            [sys.executable, str(GRADER), "hartmann", str(runs[solved]["case"]),
             "--hartmann", f"{graded_as:g}", "--mean-velocity", f"{MEAN_VELOCITY:g}",
             "--half-height", f"{HALF_HEIGHT_M:g}"],
            text=True, capture_output=True)
        print(f"  Ha {solved:g} graded as Ha {graded_as:g}: exit {proc.returncode}")
        assert proc.returncode == 1, proc.stdout
        assert "FAILED" in proc.stdout
