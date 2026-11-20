"""`sph.py`: a smoothed-particle hydrodynamics solver, checked on its exact identities.

An SPH run cannot be unit-tested by its answer: the cheapest physically meaningful case
is thousands of acoustic time steps, which does not belong in the matrix. What can be
tested, and is, are the identities the scheme has to satisfy exactly whatever the
physics does -- the kernel's normalisation and its gradient, the neighbour table against
a brute-force search, and Newton's third law across the particle pairs. Each of those is
a place a sign error or an indexing error hides while the run still produces a plausible
picture, which is the failure mode this file exists for.

The cases themselves are graded in `benchmarks/sph/`, against the hydrostatic pressure
law and the linear sloshing period, both closed form.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def sph():
    spec = importlib.util.spec_from_file_location("toolbox_sph", TOOLBOX / "sph.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- the kernel ---------------------------------------------------------------------


def test_the_kernel_integrates_to_one(sph):
    """A kernel that does not integrate to 1 puts a constant factor on every density in
    the run, so the equation of state reads a pressure that is wrong by that factor and
    nothing in the output says so."""
    h = 0.03
    step = h / 60.0
    axis = np.arange(-2.5 * h, 2.5 * h, step)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    r = np.hypot(grid[..., 0], grid[..., 1])
    w, _dwdr = sph.kernel(r, h)
    assert w.sum() * step * step == pytest.approx(1.0, rel=1e-3)


def test_the_kernel_is_compact_and_smooth_at_the_support(sph):
    h = 0.02
    w, dwdr = sph.kernel(np.array([0.0, h, 1.999 * h, 2.0 * h, 3.0 * h]), h)
    assert w[0] > w[1] > w[2] > 0.0
    assert w[3] == 0.0 and w[4] == 0.0, "the support is not compact"
    assert dwdr[3] == 0.0 and dwdr[4] == 0.0
    # Wendland C2: W and dW/dr both go to zero at q = 2, which is the property the
    # cubic spline lacks and the reason for choosing this one.
    assert abs(w[2]) < 1e-6 * w[0]
    # dW/dr vanishes at r = 0 as well, because the kernel is smooth at its own centre --
    # a self-force is what a non-zero value there would be.
    assert dwdr[0] == 0.0
    assert (dwdr[1:3] < 0.0).all(), "the kernel must fall away from the particle"


def test_the_kernel_gradient_matches_a_finite_difference(sph):
    h = 0.025
    r = np.linspace(0.1 * h, 1.9 * h, 25)
    delta = 1e-7
    high, _ = sph.kernel(r + delta, h)
    low, _ = sph.kernel(r - delta, h)
    _w, dwdr = sph.kernel(r, h)
    assert dwdr == pytest.approx((high - low) / (2.0 * delta), rel=1e-4)


# -- the neighbour table ------------------------------------------------------------


def test_the_table_finds_exactly_the_brute_force_neighbours(sph):
    """The one piece of this solver with no physics in it and the most ways to be
    quietly wrong: an off-by-one in the cell offsets drops a row of neighbours, and the
    run still looks like a run."""
    rng = np.random.default_rng(20261120)
    pos = rng.random((300, 2)) * np.array([0.4, 0.2])
    radius = 0.035
    table = sph.candidates(pos, radius)

    distance = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    for i in range(pos.shape[0]):
        expected = set(np.flatnonzero((distance[i] < radius) & (distance[i] > 0.0)).tolist())
        found = {int(j) for j in table[i] if j >= 0}
        assert expected <= found, f"particle {i} is missing neighbours"


def test_the_table_holds_no_index_out_of_range(sph):
    rng = np.random.default_rng(7)
    pos = rng.random((120, 2))
    table = sph.candidates(pos, 0.2)
    assert table.min() >= -1
    assert table.max() < pos.shape[0]
    # Every particle finds itself, which is what proves the centre cell is included.
    assert all(i in {int(j) for j in table[i]} for i in range(pos.shape[0]))


def test_a_diverged_run_is_refused_rather_than_swapping_the_machine(sph):
    """The neighbour grid is sized from the bounding box, so a divergence that stays
    finite asks for an array of astronomically many cells and numpy obliges by taking
    the machine into swap. The run then looks like a hang instead of an instability,
    which is how an afternoon goes missing."""
    with pytest.raises(ValueError, match="diverged"):
        sph.candidates(np.array([[0.0, 0.0], [1e8, 1e8]]), 0.01)
    with pytest.raises(ValueError, match="not finite"):
        sph.candidates(np.array([[0.0, 0.0], [np.nan, 0.5]]), 0.01)


def test_a_single_particle_has_a_table_and_no_neighbours(sph):
    table = sph.candidates(np.array([[0.5, 0.5]]), 0.1)
    assert table.shape[0] == 1
    assert {int(j) for j in table[0] if j >= 0} == {0}


# -- the forces ---------------------------------------------------------------------


def test_the_pair_forces_cancel(sph):
    """Newton's third law, which Monaghan's symmetric pressure term gives exactly.

    Gravity is the only external force, so the total momentum rate must be exactly
    -g times the total mass. Any asymmetry in the pressure or viscous sum -- a dropped
    neighbour, a sign error on one side of a pair -- shows up here and nowhere else in a
    short run.
    """
    rng = np.random.default_rng(3)
    pos = rng.random((200, 2)) * 0.1
    vel = rng.standard_normal((200, 2)) * 0.05
    rho = sph.RHO0 * (1.0 + 0.01 * rng.standard_normal(200))
    mass = np.full(200, sph.RHO0 * 0.005**2)
    h = 1.3 * 0.005
    table = sph.candidates(pos, 2.0 * h)
    acceleration, _drho = sph.accelerations(pos, vel, rho, mass, table, h, 20.0, 0.02)
    total = (mass[:, None] * acceleration).sum(axis=0)
    assert total[0] == pytest.approx(0.0, abs=1e-9 * mass.sum() * sph.G)
    assert total[1] == pytest.approx(-sph.G * mass.sum(), rel=1e-9)


def test_still_uniform_fluid_has_no_pressure_force_and_no_density_change(sph):
    """At rho = rho0 the equation of state gives p = 0 everywhere, so the only
    acceleration is gravity and the density does not move."""
    pos = sph.block(0.0, 0.0, 0.08, 0.08, 0.01)
    vel = np.zeros_like(pos)
    rho = np.full(pos.shape[0], sph.RHO0)
    mass = np.full(pos.shape[0], sph.RHO0 * 0.01**2)
    h = 1.3 * 0.01
    table = sph.candidates(pos, 2.0 * h)
    acceleration, drho = sph.accelerations(pos, vel, rho, mass, table, h, 20.0, 0.02)
    assert acceleration[:, 0] == pytest.approx(0.0, abs=1e-9)
    assert acceleration[:, 1] == pytest.approx(-sph.G, abs=1e-9)
    assert drho == pytest.approx(0.0, abs=1e-9)


def _lattice(sph):
    pos = sph.block(0.0, 0.0, 0.1, 0.1, 0.01)
    mass = np.full(pos.shape[0], sph.RHO0 * 0.01**2)
    h = 1.3 * 0.01
    return pos, mass, h, sph.candidates(pos, 2.0 * h), np.zeros_like(pos)


def test_density_diffusion_flattens_a_bump_and_conserves_the_right_thing(sph):
    """Two properties, and the second one is not the obvious one.

    The sign first: backwards, this term is an anti-diffusion that sharpens the noise it
    exists to remove, and the run then blows up for a reason that looks nothing like a
    sign error. `r_ij . grad_i W` is negative everywhere live, which is why the sum is
    subtracted.

    Then conservation. The bare sum of the density rates is **not** zero here -- it
    measures -8.3 on this lattice -- and that is not a defect. The pair term is
    antisymmetric but it is weighted by the neighbour's volume, so what telescopes is the
    volume-weighted sum, and that one is zero to round-off (-3e-17 against a scale of
    0.41). Asserting the bare sum was the first version of this test and it failed
    correctly.
    """
    pos, mass, h, table, still = _lattice(sph)
    middle = np.argmin(((pos - pos.mean(axis=0)) ** 2).sum(axis=1))
    rho = np.full(pos.shape[0], sph.RHO0)
    rho[middle] = sph.RHO0 * 1.002

    _a, with_it = sph.accelerations(pos, still, rho, mass, table, h, 20.0, 0.0, 0.3)
    _a, without = sph.accelerations(pos, still, rho, mass, table, h, 20.0, 0.0, 0.0)
    assert without[middle] == pytest.approx(0.0, abs=1e-9), (
        "nothing is moving, so only the diffusion may change a density here")
    assert with_it[middle] < 0.0, "the bump is being sharpened, not flattened"
    volume = mass / rho
    assert (volume * with_it).sum() == pytest.approx(
        0.0, abs=1e-9 * volume[middle] * abs(with_it[middle]))


def test_density_diffusion_is_second_order_in_a_linear_gradient(sph):
    """Why the term is safe on the case it is graded on, as a scaling law.

    It is a Laplacian, and a Laplacian annihilates a linear field -- so the hydrostatic
    density profile, which is linear to within its own curvature at the one percent
    compression this runs at, survives while the noise on top of it goes. The residual
    is not exactly zero, because the neighbour's volume in the weight varies along the
    gradient too, but it is second order: measured here, halving the slope divides the
    residual by 3.99, 4.00 and 4.00. A first-order residual would mean the diffusion was
    eating the physical gradient and the hydrostatic grade was measuring the diffusion
    coefficient.
    """
    pos, mass, h, table, still = _lattice(sph)
    inside = ((pos[:, 0] > 2 * h) & (pos[:, 0] < 0.1 - 2 * h)
              & (pos[:, 1] > 2 * h) & (pos[:, 1] < 0.1 - 2 * h))
    assert inside.any()
    residuals = []
    for slope in (0.004, 0.002, 0.001):
        ramp = sph.RHO0 * (1.0 + slope * (0.1 - pos[:, 1]) / 0.1)
        _a, rate = sph.accelerations(pos, still, ramp, mass, table, h, 20.0, 0.0, 0.3)
        residuals.append(float(np.abs(rate[inside]).max()))
    for coarse, fine in zip(residuals, residuals[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.05), (
            f"residual fell by {coarse / fine:.2f} for half the slope, not by 4: the "
            "diffusion is first order in the gradient and is eating it")


def test_viscosity_damps_approach_and_ignores_separation(sph):
    """Monaghan's artificial viscosity is one-sided on purpose: it is there to stop
    particles interpenetrating, and a two-sided version would damp the free-surface
    motion these cases are entirely about."""
    pos = np.array([[0.0, 0.0], [0.01, 0.0]])
    rho = np.full(2, sph.RHO0)
    mass = np.full(2, sph.RHO0 * 0.01**2)
    h = 1.3 * 0.01
    table = sph.candidates(pos, 2.0 * h)

    closing = np.array([[1.0, 0.0], [-1.0, 0.0]])
    apart = -closing
    toward, _ = sph.accelerations(pos, closing, rho, mass, table, h, 20.0, 0.1)
    away, _ = sph.accelerations(pos, apart, rho, mass, table, h, 20.0, 0.1)
    # Approaching: the left particle is pushed back to the left (negative x).
    assert toward[0, 0] < 0.0
    assert toward[1, 0] > 0.0
    # Separating: nothing at all, because p = 0 here and the viscous branch is off.
    assert away[:, 0] == pytest.approx(0.0, abs=1e-12)


def test_tait_is_monotonic_above_the_reference_density(sph):
    rho = np.array([sph.RHO0, sph.RHO0 * 1.005, sph.RHO0 * 1.01])
    p = sph.pressure(rho, 20.0)
    assert p[0] == pytest.approx(0.0)
    assert p[0] < p[1] < p[2]
    # The bulk modulus is c0^2*rho0/gamma, so a 1% density rise is about gamma% of it.
    assert p[2] == pytest.approx(20.0**2 * sph.RHO0 / sph.GAMMA * (1.01**sph.GAMMA - 1.0))


def test_the_pressure_is_floored_at_zero(sph):
    """The one line without which this solver blows up. Tait is stiff through rho0, so
    a tenth of a percent of density noise is hundreds of pascals either way; where it
    goes negative the pressure term pulls neighbours together instead of pushing them
    apart, and the clumping has no limit. Measured before the floor existed: the first
    negative pressure by the hundredth step, -22,366 Pa six hundred steps in, and the
    fluid outside the tank after 729."""
    rarefied = np.array([sph.RHO0 * 0.9, sph.RHO0 * 0.999, sph.RHO0])
    assert (sph.pressure(rarefied, 20.0) == 0.0).all()


def test_a_rarefied_particle_is_not_pulled_towards_its_neighbour(sph):
    """The floor, as a force rather than as a number: two under-dense particles at rest
    exert nothing on each other. Without it they attract, which is the instability."""
    pos = np.array([[0.0, 0.0], [0.01, 0.0]])
    rho = np.full(2, sph.RHO0 * 0.99)
    mass = np.full(2, sph.RHO0 * 0.01**2)
    h = 1.3 * 0.01
    table = sph.candidates(pos, 2.0 * h)
    acceleration, _ = sph.accelerations(pos, np.zeros_like(pos), rho, mass, table,
                                        h, 20.0, 0.02)
    assert acceleration[:, 0] == pytest.approx(0.0, abs=1e-12)
    assert acceleration[:, 1] == pytest.approx(-sph.G, abs=1e-12)


# -- geometry and the integrator ----------------------------------------------------


def test_a_block_puts_its_surface_where_the_fluid_ends(sph):
    """Half a spacing in, so a column of height H has its free surface at H. Getting
    this wrong is a dx/2 error in every length the dam break is graded on."""
    particles = sph.block(0.0, 0.0, 0.04, 0.02, 0.01)
    assert particles.shape == (4 * 2, 2)
    assert particles[:, 0].min() == pytest.approx(0.005)
    assert particles[:, 0].max() == pytest.approx(0.035)
    assert particles[:, 1].max() == pytest.approx(0.015)


def test_the_tank_is_three_layers_thick_and_outside_the_fluid(sph):
    walls = sph.tank(0.1, 0.05, 0.01, layers=3)
    assert walls[:, 1].min() < 0.0, "the floor is not below the fluid domain"
    assert walls[:, 1].min() == pytest.approx(-0.025)
    # Nothing in the walls sits inside the box the fluid occupies.
    inside = (walls[:, 0] > 0.0) & (walls[:, 0] < 0.1) & (walls[:, 1] > 0.0)
    assert not inside.any()


def test_wall_particles_never_move(sph):
    """Dynamic boundary particles evolve a density and keep their position. A wall that
    drifts is a tank that leaks, and it would read as a free-surface instability."""
    result = sph.hydrostatic(depth=0.04, dx=0.01, end=0.004, progress=False)
    walls = ~result["moving"]
    assert (result["vel"][walls] == 0.0).all()
    assert (result["rho"][walls] >= sph.RHO0).all(), (
        "a wall below the reference density would pull fluid into itself")


def test_the_run_reports_the_step_it_actually_took(sph):
    result = sph.hydrostatic(depth=0.04, dx=0.01, end=0.004, progress=False)
    assert result["dt"] == pytest.approx(0.2 * result["h"] / result["c0"])
    assert result["steps"] == math.ceil(result["end"] / result["dt"])
    assert result["history"], "no history was recorded"
    assert result["history"][0]["t"] == 0.0


def test_the_period_of_a_clean_sine_is_recovered(sph):
    """`_period` is the measurement the sloshing grade turns on, so it is checked
    against a signal whose period is known rather than only against a run."""
    period = 1.25
    t = np.linspace(0.0, 4.0 * period, 400)
    history = [{"t": float(time), "centroid_x": 0.3 + 0.01 * math.sin(2 * math.pi * time / period)}
               for time in t]
    measured, cycles = sph._period(history)
    assert measured == pytest.approx(period, rel=2e-3)
    assert cycles == pytest.approx(4.0, rel=0.05)


def test_noise_does_not_shorten_the_period(sph):
    """The failure that replaced zero crossings with a periodogram, as a test.

    Counting crossings, every spurious pair of them shortens the mean interval, so the
    answer comes out short with nothing to show it: a real 3-period run reported 7
    crossings where 6 were due and a period 8.1% low. A projection onto sin and cos uses
    all 400 samples, and noise a third of the amplitude moves it by well under a
    percent -- where a crossing count, on this very signal, is out by more than ten.
    """
    period = 1.25
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 3.0 * period, 400)
    clean = 0.01 * np.sin(2.0 * math.pi * t / period)
    noisy = clean + 0.0033 * rng.standard_normal(t.size)
    history = [{"t": float(a), "centroid_x": float(b)} for a, b in zip(t, noisy)]
    measured, _cycles = sph._period(history)
    assert measured == pytest.approx(period, rel=0.01)

    # The control: the crossing count this replaced, on the same signal.
    crossings = np.flatnonzero(np.diff(np.sign(noisy - noisy.mean())) != 0)
    by_crossings = 2.0 * np.mean(np.diff(t[crossings]))
    assert abs(by_crossings / period - 1.0) > 0.1, (
        "the noise is too mild for this test to demonstrate anything")


def test_a_signal_with_no_oscillation_reports_no_period(sph):
    """Better than a number fitted to the shape of the window: the sloshing grade would
    then be graded against noise and would pass or fail at random."""
    history = [{"t": float(i) * 0.01, "centroid_x": 0.5} for i in range(50)]
    measured, cycles = sph._period(history)
    assert math.isnan(measured)
    assert cycles == 0.0


def test_less_than_two_cycles_is_refused(sph):
    """A frequency fitted to one cycle is fitted to the window."""
    t = np.linspace(0.0, 0.9, 200)
    history = [{"t": float(x), "centroid_x": math.sin(2.0 * math.pi * x)} for x in t]
    measured, cycles = sph._period(history)
    assert math.isnan(measured)
    assert cycles < 1.5


# -- the picture ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny(sph):
    """One very short run of each case, shared, because each is thousands of acoustic
    steps even at an absurd spacing and the drawing path is what is being tested."""
    return {
        "hydrostatic": sph.hydrostatic(depth=0.04, dx=0.01, end=0.004, progress=False),
        "sloshing": sph.sloshing(length=0.4, depth=0.2, dx=0.05, periods=0.02,
                                 progress=False),
        "dambreak": sph.dambreak(column=0.04, dx=0.01, end=0.004, progress=False),
    }


@pytest.mark.parametrize("case", ["hydrostatic", "sloshing", "dambreak"])
def test_each_case_draws_its_figure(sph, tmp_path, tiny, case):
    """One branch per case in `figure`, and the first version of the hydrostatic branch
    referred to a key the result did not carry. Nothing caught it, because nothing asked
    for the picture."""
    path = tmp_path / f"{case}.png"
    sph.figure(tiny[case], path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "not a png"


@pytest.mark.parametrize("case", ["hydrostatic", "sloshing", "dambreak"])
def test_each_case_reports_without_raising(sph, tiny, case):
    text = sph.report(tiny[case])
    assert tiny[case]["case"] in text
    assert "steps/s" in text


def test_the_dam_break_reports_martin_and_moyce_coordinates(sph, tiny):
    """Z = x/a against T = t sqrt(2g/a), which is the form the comparison would be made
    in. It is reported and not graded; `benchmarks/sph/README.md` is why."""
    front = tiny["dambreak"]["front"]
    assert front and {"T", "Z", "t", "x"} <= set(front[0])
    assert front[0]["T"] == 0.0
    # The column starts a particle's width in from the wall and the front only advances.
    assert front[-1]["Z"] >= front[0]["Z"] > 0.0


# -- the contract the toolbox holds every script to ---------------------------------


def test_the_docstring_shows_how_to_run_it(sph):
    doc = ast.get_docstring(ast.parse((TOOLBOX / "sph.py").read_text(encoding="utf-8")))
    assert "python3 sph.py" in doc


def test_it_offers_rather_than_instructs(sph):
    text = (TOOLBOX / "sph.py").read_text(encoding="utf-8").lower()
    for imperative in ("you must", "you should", "always run", "do not proceed",
                       "before you run", "step 1"):
        assert imperative not in text, imperative


def test_it_says_what_it_is_not_for(sph):
    doc = ast.get_docstring(ast.parse((TOOLBOX / "sph.py").read_text(encoding="utf-8")))
    assert "not for" in doc.lower()
    assert "interfoam" in doc.lower(), "the alternative that already exists is not named"
