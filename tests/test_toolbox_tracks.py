"""`tracks.py`, the particle and tracer integrator.

The quantitative grading lives in `benchmarks/particles/grade.py`, which solves in the
`benchmarks` job. What is here is the behaviour that grader takes for granted: that the
closed-form field is the field it says it is, that a tracer and a particle are different
things, that the drag law does something, and that a fate is decided rather than
defaulted. The `case` path is not exercised: it needs pyvista and an OpenFOAM case, which
this environment is not promised to have, so it follows the toolbox convention for that
(see `test_render_is_valid_python` in `test_toolbox.py`) of a parse check plus the
import-allowlist rules the rest of the toolbox is held to.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tracks():
    path = ROOT / "openreynolds" / "toolbox" / "tracks.py"
    spec = importlib.util.spec_from_file_location("toolbox_tracks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_cylinder_field_is_tangent_to_the_wall_and_uniform_far_away(tracks):
    """Potential flow past a cylinder, checked at the two places it is known exactly:
    nothing goes through the wall, and the far field is the free stream."""
    velocity = tracks.potential_cylinder(1.0, 2.0)
    angles = np.linspace(0.0, 2 * math.pi, 37)[:-1]
    wall = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)
    u = velocity(wall)
    normal_component = u[:, 0] * wall[:, 0] + u[:, 1] * wall[:, 1]
    assert np.max(np.abs(normal_component)) < 1e-12

    far = np.array([[-1000.0, 3.0, 0.0], [1000.0, -7.0, 0.0]])
    assert velocity(far)[:, 0] == pytest.approx([2.0, 2.0], rel=1e-5)
    assert velocity(far)[:, 1] == pytest.approx([0.0, 0.0], abs=1e-5)

    # The forward stagnation point: u = 0 there, and one radius-fraction upstream of it
    # the speed toward the wall is 2U/R times the distance. That strain rate is the only
    # property of the flow the capture threshold turns on, so it is checked here rather
    # than trusted: 1/8 is 1/4 of 1/(2U/R) and nothing else.
    stand_off = 1e-6
    axis = np.array([[-1.0, 0.0, 0.0], [-1.0 - stand_off, 0.0, 0.0]])
    assert velocity(axis)[0, 0] == pytest.approx(0.0, abs=1e-12)
    strain = velocity(axis)[1, 0] / stand_off
    assert strain == pytest.approx(2 * 2.0 / 1.0, rel=1e-5)


def test_a_tracer_follows_the_field_and_a_heavy_particle_does_not(tracks):
    """Same seed, same field, tau_p = 0 against tau_p large. A tracer in solid-body
    rotation stays on its circle; a particle with inertia spirals out of it."""
    omega = 1.0

    def velocity(points):
        return np.stack([-omega * points[:, 1], omega * points[:, 0],
                         np.zeros(len(points))], axis=-1)

    seed = np.array([[1.0, 0.0, 0.0]])
    steps, dt = 400, 2 * math.pi / 400
    _, _, tracer, _ = tracks.integrate(seed, velocity, 0.0, dt, steps)
    _, _, heavy, _ = tracks.integrate(seed, velocity, 5.0, dt, steps)
    assert np.hypot(tracer[0, 0], tracer[0, 1]) == pytest.approx(1.0, abs=1e-8)
    assert np.hypot(heavy[0, 0], heavy[0, 1]) > 1.5


def test_the_schiller_naumann_correction_is_one_at_zero_slip_and_grows_with_it(tracks):
    slip = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    phi = tracks.drag_factor(slip, 1e-4, 1.5e-5, "schiller-naumann")
    assert phi[0] == pytest.approx(1.0)
    # Re_p = 1 * 1e-4 / 1.5e-5 = 6.667, so phi = 1 + 0.15 * 6.667^0.687.
    assert phi[1] == pytest.approx(1 + 0.15 * (1e-4 / 1.5e-5) ** 0.687, rel=1e-12)
    assert tracks.drag_factor(slip, 1e-4, 1.5e-5, "stokes") == pytest.approx([1.0, 1.0])


def test_stokes_relaxation_time_is_rho_d_squared_over_eighteen_mu(tracks):
    assert tracks.relaxation_time(2000.0, 1e-4, 1.8e-5) == pytest.approx(
        2000.0 * 1e-8 / (18 * 1.8e-5), rel=1e-12
    )


def test_a_particle_that_stops_approaching_is_recorded_as_stopped_not_captured(tracks):
    """The distinction the capture benchmark rests on. Below the critical Stokes number
    the particle approaches the wall for the rest of time and never reaches it, so a
    distance test alone eventually reports a capture that is a floating-point underflow.
    """
    fates = {}
    for stokes in (0.05, 0.5):
        fate, _, _, _ = tracks.integrate(
            np.array([[-5.0, 0.0, 0.0]]), tracks.potential_cylinder(1.0, 1.0),
            stokes, 0.005, 10000, capture=tracks.cylinder_capture(1.0),
            exit_test=lambda p: p[:, 0] > 2.0, stop_speed=1e-10,
        )
        fates[stokes] = fate[0]
    assert fates[0.05] == "stopped"
    assert fates[0.5] == "captured"


def test_the_capture_threshold_refuses_a_bracket_that_does_not_bracket(tracks):
    """A bisection on a bracket that is wrong at one end returns a number anyway, and
    that number is the end of the bracket rather than a threshold."""
    with pytest.raises(SystemExit):
        tracks.capture_threshold(low=0.3, high=0.5, horizon=20.0)
    with pytest.raises(SystemExit):
        tracks.capture_threshold(low=0.01, high=0.05, horizon=20.0)


def test_the_capture_threshold_lands_above_one_eighth_and_not_below(tracks):
    """The sign is not negotiable: a finite stopping criterion can only over-estimate a
    threshold whose exact statement is about infinite time. A measurement below 1/8 is a
    bug and not a better answer. The published value is 1/8 in this tau_p U/R convention
    (arXiv:2310.03474v2 section III.3, which states it as 1/4 in a convention whose U is
    the local stagnation-point speed, twice the free stream).
    """
    measured = tracks.capture_threshold(stop_speed=1e-10, dt=0.01, horizon=50.0)
    assert measured["stokes_critical"] > 0.125
    assert measured["stokes_critical"] < 0.125 * 1.03
