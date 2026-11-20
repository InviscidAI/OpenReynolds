"""`viv.py`, the instrument for a finished moving-body run.

Two kinds of check. The synthetic ones drive it with an oscillator whose amplitude,
frequency and energy balance are known in closed form, so a wrong answer is a wrong
answer and not a disagreement with another measurement. The last one drives it with the
2026-09-12 OpenFOAM runs already committed under `tests/data/moving_mesh/` and holds it
to the numbers `benchmarks/moving_mesh/grade.py` reports from the same files: the two
read the same raw output through two implementations, and this is what makes the second
one safe to keep.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "moving_mesh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def viv():
    return _load(ROOT / "openreynolds" / "toolbox" / "viv.py", "toolbox_viv")


@pytest.fixture(scope="module")
def grade():
    return _load(ROOT / "benchmarks" / "moving_mesh" / "grade.py", "moving_mesh_grade")


class Args:
    """The flag bundle `viv.py` reads, with everything unset so it has to read the case."""

    def __init__(self, **kw):
        self.diameter = self.u = self.aref = self.rho = self.mass = self.stiffness = None
        self.cycles, self.st_cycles, self.energy_cycles = 8, 7, 6
        self.body_fo, self.fixed, self.out, self.json = "bodyState", None, None, False
        self.__dict__.update(kw)


# -- the primitives ------------------------------------------------------------


def test_upward_crossings_are_interpolated_not_snapped_to_a_sample(viv):
    """A frequency taken from the nearest sample is wrong by up to half a step on every
    crossing, which on eight cycles of a 200-step oscillation is a 0.25% error in f."""
    # Deliberately off-grid: the crossings sit at 0.37, 1.37, ... while the samples are
    # at multiples of 0.01, so a nearest-sample frequency would be wrong here and an
    # interpolated one is not.
    t = np.linspace(0.0, 4.0, 401)
    crossings = viv.upward_crossings(t, np.sin(2 * math.pi * (t - 0.373)))
    assert len(crossings) == 4
    assert crossings == pytest.approx([0.373, 1.373, 2.373, 3.373], abs=1e-5)
    assert not np.allclose(crossings, np.round(crossings, 2))


def test_a_restart_directory_cuts_the_run_it_replaced(viv, tmp_path):
    """A restart from `latestTime` writes into a directory named for its start time, and
    the run it replaced usually went on past that before it was abandoned. Those rows
    belong to a solution nobody kept."""
    for start, rows in ((0, [(0.0, 0.0), (1.0, 1.0), (2.0, 99.0)]),
                        (1, [(1.0, 1.0), (2.0, 2.0)])):
        d = tmp_path / "postProcessing" / "bodyState" / str(start)
        d.mkdir(parents=True)
        (d / "sixDoFRigidBodyState.dat").write_text(
            "# Time centreOfMass\n"
            + "".join(f"{t:g} ({v:g} 0 0)\n" for t, v in rows)
        )
    series = viv.read_series(tmp_path, "bodyState", "sixDoFRigidBodyState.dat")
    assert 99.0 not in series["centreOfMass_x"]
    assert series["centreOfMass_x"][-1] == 2.0


def test_a_missing_constant_names_the_flag_that_supplies_it(viv, tmp_path):
    """Assuming a density or a diameter is how a graded case ends up graded against a
    number nobody set, so the refusal has to say what to pass."""
    with pytest.raises(viv.CaseError) as caught:
        viv.read_series(tmp_path, "bodyState", "sixDoFRigidBodyState.dat")
    assert "bodyState" in str(caught.value)


# -- the closed forms ----------------------------------------------------------


def _write_oscillator(case: Path, *, amplitude, frequency, mass, stiffness,
                      diameter, speed, span, rho, cl_amplitude, cl_phase):
    """A case directory holding nothing but a synthetic motion and lift history.

    y(t) = A sin(2 pi f t) and Cl(t) = C sin(2 pi f t + phi). The net work per cycle of
    that pair is exactly `pi A C q Aref sin(phi)`, so a phase of zero is a limit cycle
    that dissipates nothing and any other phase is a known non-zero number.
    """
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "dynamicMeshDict").write_text(
        "dynamicFvMesh dynamicMotionSolverFvMesh;\n"
        "motionSolver sixDoFRigidBodyMotion;\n"
        f"mass {mass!r};\n"
        "restraints { spring { sixDoFRigidBodyMotionRestraint linearSpring;\n"
        f"  anchor (0 0 0); refAttachmentPt (0 0 0); stiffness {stiffness!r};\n"
        "  damping 0; restLength 0; } }\n"
    )
    (case / "system" / "controlDict").write_text(
        "application pimpleFoam;\n"
        "functions { forceCoeffs { type forceCoeffs;\n"
        f"  rhoInf {rho!r}; magUInf {speed!r}; lRef {diameter!r};\n"
        "  Aref " + repr(diameter * span) + "; } }\n"
    )
    t = np.linspace(0.0, 12.0 / frequency, int(round(12 * 400)) + 1)
    omega = 2 * math.pi * frequency
    body = case / "postProcessing" / "bodyState" / "0"
    body.mkdir(parents=True)
    (body / "sixDoFRigidBodyState.dat").write_text(
        "# Time centreOfMass velocity\n"
        + "".join(
            f"{tt:.10g} (0 {amplitude * math.sin(omega * tt):.10g} 0) "
            f"(0 {amplitude * omega * math.cos(omega * tt):.10g} 0)\n"
            for tt in t
        )
    )
    forces = case / "postProcessing" / "forceCoeffs" / "0"
    forces.mkdir(parents=True)
    (forces / "coefficient.dat").write_text(
        "# Time Cd Cl\n"
        + "".join(f"{tt:.10g} 2 {cl_amplitude * math.sin(omega * tt + cl_phase):.10g}\n"
                  for tt in t)
    )


def test_it_recovers_the_amplitude_and_frequency_of_a_known_oscillation(viv, tmp_path):
    frequency, amplitude, diameter = 0.25, 0.004, 0.01
    _write_oscillator(tmp_path, amplitude=amplitude, frequency=frequency, mass=1e-2,
                      stiffness=1e-2 * (2 * math.pi * frequency) ** 2, diameter=diameter,
                      speed=0.01, span=diameter, rho=1000.0, cl_amplitude=0.5,
                      cl_phase=0.0)
    out = viv.released_body(tmp_path, Args())
    assert out["amplitude_over_D"] == pytest.approx(amplitude / diameter, rel=1e-4)
    assert out["frequency"] == pytest.approx(frequency, rel=1e-5)
    assert out["f_over_fn"] == pytest.approx(1.0, rel=1e-5)


def test_the_added_mass_frequency_is_the_vacuum_one_scaled_by_sqrt_mstar_over_mstar_plus_one(
    viv, tmp_path
):
    """A circular cylinder's added mass is exactly its own displaced mass (Lamb,
    *Hydrodynamics*, 6th ed., art. 68), so the still-fluid natural frequency is
    f_n sqrt(m*/(m* + 1)) and not f_n. At m* = 10 that is 4.7%, which is larger than the
    agreement most of these cases are reported to.
    """
    frequency, diameter, rho, span = 0.2, 0.01, 1000.0, 0.01
    displaced = rho * math.pi * diameter**2 / 4 * span
    mass = 10.0 * displaced
    _write_oscillator(tmp_path, amplitude=0.005, frequency=frequency, mass=mass,
                      stiffness=mass * (2 * math.pi * frequency) ** 2, diameter=diameter,
                      speed=0.01, span=span, rho=rho, cl_amplitude=0.5, cl_phase=0.0)
    out = viv.released_body(tmp_path, Args())
    assert out["mass_ratio"] == pytest.approx(10.0, rel=1e-9)
    assert out["natural_frequency_added_mass"] == pytest.approx(
        out["natural_frequency"] * math.sqrt(10.0 / 11.0), rel=1e-9
    )


@pytest.mark.parametrize("phase, sign", [(0.0, 0), (0.4, +1), (-0.4, -1)])
def test_the_energy_audit_is_zero_only_when_lift_and_displacement_are_in_phase(
    viv, tmp_path, phase, sign
):
    """The whole point of the audit. Work per cycle is pi A C q Aref sin(phi): zero at
    phi = 0, and of the sign of phi otherwise. A loosely coupled run reads -10% here
    while its displacement trace is a clean sinusoid.
    """
    frequency, amplitude, diameter, rho, span, speed = 0.25, 0.004, 0.01, 1000.0, 0.01, 0.01
    cl = 0.5
    _write_oscillator(tmp_path, amplitude=amplitude, frequency=frequency, mass=1e-2,
                      stiffness=1e-2 * (2 * math.pi * frequency) ** 2, diameter=diameter,
                      speed=speed, span=span, rho=rho, cl_amplitude=cl, cl_phase=phase)
    out = viv.released_body(tmp_path, Args())
    works = [row["work"] for row in out["energy_audit"]]
    expected = math.pi * amplitude * cl * 0.5 * rho * speed**2 * diameter * span * math.sin(phase)
    assert works
    for work in works:
        assert work == pytest.approx(expected, abs=1e-3 * abs(expected) + 1e-12)
    percents = [row["work_pct"] for row in out["energy_audit"]]
    if sign == 0:
        assert max(abs(p) for p in percents) < 1e-6
    else:
        assert all(p * sign > 0.5 for p in percents)


# -- against the real runs -----------------------------------------------------


@pytest.mark.parametrize("name, amplitude, work_low, work_high", [
    # The recorded 2026-09-12 result, from benchmarks/moving_mesh/README.md. The loose
    # row is the failure the case exists for: a 13% wider limit cycle held there by the
    # coupling, which only the energy audit sees.
    ("viv_loose", 0.6401929, -10.37341, -10.09327),
    ("viv_tight", 0.5670282, 0.04367032, 0.1755677),
])
def test_it_reproduces_the_recorded_run(viv, name, amplitude, work_low, work_high):
    out = viv.released_body(DATA / name, Args())
    assert out["amplitude_over_D"] == pytest.approx(amplitude, rel=1e-6)
    works = [row["work_pct"] for row in out["energy_audit"]]
    assert min(works) == pytest.approx(work_low, rel=1e-5)
    assert max(works) == pytest.approx(work_high, rel=1e-5)
    # Derived from the case files rather than from a flag: 10 and 5 are what the case
    # was asked for, so recovering them is a check on the reading, not on the run.
    assert out["mass_ratio"] == pytest.approx(10.0, rel=1e-4)
    assert out["reduced_velocity"] == pytest.approx(5.0, rel=1e-4)


@pytest.mark.parametrize("name", ["viv_loose", "viv_tight"])
def test_it_agrees_digit_for_digit_with_the_benchmark_grader(viv, grade, name):
    """Two implementations over the same committed output. `benchmarks/README.md` is
    blunt about a second implementation being a second thing to keep right; this is
    what keeps it right, and it is cheap because the trimmed run is already here.
    """
    class GradeArgs(Args):
        def __init__(self):
            super().__init__()
            self.inner_fo = self.outer_fo = None

    fixed = grade.fixed_cylinder(DATA / "viv_fixed", GradeArgs())
    theirs = grade.released_cylinder(DATA / name, GradeArgs(), fixed)
    mine = viv.released_body(DATA / name, Args(fixed=None))
    for key in ("amplitude_over_D", "frequency", "natural_frequency", "f_over_fn",
                "mean_cd", "cl_rms", "mean_y_over_D", "mass", "stiffness"):
        assert mine[key] == pytest.approx(theirs[key], rel=1e-12), key
    assert [row["work_pct"] for row in mine["energy_audit"]] == pytest.approx(
        [row["work_pct"] for row in theirs["energy_audit"]], rel=1e-12
    )


def test_the_fixed_case_gives_the_strouhal_number_the_mesh_was_graded_on(viv):
    out = viv.fixed_body(DATA / "viv_fixed", Args())
    assert out["strouhal"] == pytest.approx(0.166141, rel=1e-5)
    assert out["mean_cd"] == pytest.approx(1.355514, rel=1e-5)


def test_lock_in_is_decided_against_the_fixed_shedding_and_not_asserted(viv):
    """f/f_n alone cannot say a body is locked in: it has to be nearer the structure's
    frequency than the frequency the same mesh sheds at with the body bolted still."""
    fixed = viv.fixed_body(DATA / "viv_fixed", Args())
    out = viv.released_body(DATA / "viv_tight", Args(), fixed)
    assert out["locked_in"] is True
    assert out["f_over_fixed_shedding"] == pytest.approx(1.167306, rel=1e-5)
