"""`benchmarks/particles/grade.py` must reject a wrong answer, not only report a right one.

The `benchmarks` job runs this grader for real, so the numbers are measured there. What
this file holds is the other half, the half the thermal-stress grader's test established:
a grader that only ever sees a good run is a grader nobody has watched fail. Each test
here hands it a broken integrator and asserts the verdict goes to FAIL.

The capture case is slow (a bisection per stopping speed), so it is driven here with a
single stopping speed and a short horizon; the full convergence sweep is the CI step.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grade():
    return _load(ROOT / "benchmarks" / "particles" / "grade.py", "particles_grade")


class Args:
    def __init__(self, **kw):
        self.steps_per_tau = 200
        self.revolution_steps = 200
        self.dt = 0.01
        self.horizon = 50.0
        self.monotone_samples = 5
        self.stop_speeds = [1e-10]
        self.json = False
        self.__dict__.update(kw)


def _verdicts(result: dict) -> dict[str, str]:
    return {check["name"]: check["verdict"] for check in result["checks"]}


def test_the_published_critical_stokes_number_is_recorded_in_the_radius_convention(grade):
    """1/8 and 1/4 are the same published result two conventions apart, and the
    convention is the whole risk: `benchmarks/README.md` records that of its eleven
    reference values, re-checking changed two, and one was a factor of two that reached
    a customer-facing page. The number stored here is the tau_p U_inf / R one.
    """
    assert grade.THRESHOLDS["stokes_critical_over_R"] == 0.125
    assert "arXiv:2310.03474" in grade.__doc__
    assert "1/8" in grade.__doc__ and "1/4" in grade.__doc__
    # And the source that could not be read is named as unread rather than cited.
    assert "Langmuir" in grade.__doc__ and "not read" in grade.__doc__


def test_the_drag_case_passes_on_the_shipping_integrator(grade):
    result = grade.grade_drag(Args())
    assert all(v in ("PASS", "-") for v in _verdicts(result).values())


def test_the_tracer_case_passes_on_the_shipping_integrator(grade):
    result = grade.grade_tracer(Args())
    assert all(v in ("PASS", "-") for v in _verdicts(result).values())


def test_a_first_order_integrator_fails_the_tracer_order_check(grade, monkeypatch):
    """Forward Euler closes the orbit too, just an order worse. If the grader passed it,
    the fourth-order claim in the docstring would be unchecked."""
    tracks = grade.tracks()

    def euler(seeds, velocity, tau, dt, steps, **kw):
        import numpy as np

        position = np.array(seeds, dtype=float).reshape(-1, 3)
        for _ in range(steps):
            position = position + dt * np.nan_to_num(velocity(position))
        fate = np.array(["running"] * len(position), dtype=object)
        return fate, np.full(len(position), steps * dt), position, None

    monkeypatch.setattr(grade, "tracks", lambda: _patched(tracks, euler))
    result = grade.grade_tracer(Args())
    assert _verdicts(result)["observed order of convergence"] == "FAIL"


def _patched(module, integrate):
    class Shim:
        def __getattr__(self, name):
            return integrate if name == "integrate" else getattr(module, name)

    return Shim()


def test_a_drag_law_that_ignores_the_particle_reynolds_number_fails_the_contrast(
    grade, monkeypatch
):
    """The anti-tautology check. If `--drag schiller-naumann` quietly behaved like
    Stokes drag, both exact checks above would still pass and the grader would be
    comparing the integrator with itself."""
    tracks = grade.tracks()

    def flat(slip, diameter, nu, law):
        import numpy as np

        return np.ones(len(slip))

    class Shim:
        def __getattr__(self, name):
            return flat if name == "drag_factor" else getattr(tracks, name)

    monkeypatch.setattr(grade, "tracks", lambda: Shim())
    result = grade.grade_drag(Args())
    verdicts = _verdicts(result)
    failed = [name for name, verdict in verdicts.items() if verdict == "FAIL"]
    assert any("Schiller-Naumann terminal / Stokes terminal" in name for name in failed)


def test_a_capture_threshold_below_one_eighth_is_failed_in_both_directions(
    grade, monkeypatch
):
    """Not just "within 3%": below 1/8 is a bug, because the exact statement is about
    infinite time and a finite stopping criterion can only over-estimate. A grader that
    accepted 0.1240 would accept a stopping rule that fires early."""
    tracks = grade.tracks()

    class Shim:
        def __getattr__(self, name):
            if name == "capture_threshold":
                return lambda **kw: {"stokes_critical": 0.1240}
            return getattr(tracks, name)

    monkeypatch.setattr(grade, "tracks", lambda: Shim())
    result = grade.grade_capture(Args())
    verdicts = _verdicts(result)
    assert verdicts["the error is positive, which it has to be"] == "FAIL"


def test_a_capture_threshold_that_does_not_converge_is_failed(grade, monkeypatch):
    """The convergence is what makes the residual +1% a diagnosis. A threshold that sat
    at the same wrong place however tight the stopping rule got would mean the error was
    something else, and the grader has to say so rather than pass on the 3% bound."""
    tracks = grade.tracks()

    class Shim:
        def __getattr__(self, name):
            if name == "capture_threshold":
                return lambda **kw: {"stokes_critical": 0.1275}
            return getattr(tracks, name)

    monkeypatch.setattr(grade, "tracks", lambda: Shim())
    result = grade.grade_capture(Args(stop_speeds=[1e-8, 1e-10, 1e-12]))
    failed = [name for name, verdict in _verdicts(result).items() if verdict == "FAIL"]
    assert any(name.startswith("error [%] at stop speeds") for name in failed)
