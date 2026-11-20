"""`lbm.py`: a lattice Boltzmann solver, checked where it can be checked exactly.

The whole reason this solver is worth having in the repository is that one of its cases
has a closed-form answer, so the tests here are not "it ran and produced numbers". The
sharp one is `test_the_trt_channel_is_exact_at_every_viscosity`: with the magic parameter
at 3/16 the halfway bounce-back wall sits on the link midpoint whatever the viscosity,
so a force-driven channel reproduces the analytic parabola to the convergence residual
and not merely to second order. The companion test is the one that gives that meaning --
BGK, the same code with one relaxation rate, is exact at one viscosity and measurably
wrong either side of it, which is the property the default scheme exists to remove.

Everything here is sized to run in the matrix: the largest lattice is 16 cells across.
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
def lbm():
    spec = importlib.util.spec_from_file_location("toolbox_lbm", TOOLBOX / "lbm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- the lattice's own identities ---------------------------------------------------


def test_the_velocity_set_is_consistent(lbm):
    """A D2Q9 set that does not satisfy these is not D2Q9, and every error below it
    would read as a physics problem."""
    assert lbm.W.sum() == pytest.approx(1.0)
    assert (lbm.W * lbm.CX).sum() == pytest.approx(0.0)
    assert (lbm.W * lbm.CY).sum() == pytest.approx(0.0)
    # Second moment: sum w c_a c_b = c_s^2 delta_ab, which is what makes the scheme
    # recover the Navier-Stokes pressure.
    assert (lbm.W * lbm.CX * lbm.CX).sum() == pytest.approx(lbm.CS2)
    assert (lbm.W * lbm.CY * lbm.CY).sum() == pytest.approx(lbm.CS2)
    assert (lbm.W * lbm.CX * lbm.CY).sum() == pytest.approx(0.0)
    # OPP is an involution that negates the velocity.
    assert np.array_equal(lbm.OPP[lbm.OPP], np.arange(9))
    assert np.array_equal(lbm.CX[lbm.OPP], -lbm.CX)
    assert np.array_equal(lbm.CY[lbm.OPP], -lbm.CY)
    assert np.array_equal(lbm.W[lbm.OPP], lbm.W)
    # Every pair in PAIRS is an opposite pair, and between them they cover 1-8.
    assert sorted(i for pair in lbm.PAIRS for i in pair) == list(range(1, 9))
    for i, j in lbm.PAIRS:
        assert lbm.OPP[i] == j


def test_the_equilibrium_carries_the_density_and_the_momentum(lbm):
    """Whatever else it is, the equilibrium's zeroth and first moments are rho and
    rho*u, to second order in u. The third digit of that is the scheme's Mach error."""
    rho = np.full((3, 3), 1.05)
    ux = np.full((3, 3), 0.06)
    uy = np.full((3, 3), -0.02)
    feq = lbm.equilibrium(rho, ux, uy)
    assert feq.sum(axis=0) == pytest.approx(rho)
    assert (lbm.CX[:, None, None] * feq).sum(axis=0) == pytest.approx(rho * ux)
    assert (lbm.CY[:, None, None] * feq).sum(axis=0) == pytest.approx(rho * uy)
    assert (feq > 0).all(), "a negative equilibrium population is an unstable scheme"


@pytest.mark.parametrize("nu", [0.005, 0.0128, 0.1, 0.5])
def test_the_magic_parameter_holds_at_every_viscosity(lbm, nu):
    """TRT spends its free rate on fixing the product at 3/16; BGK sets the two equal
    and so hits 3/16 at exactly one viscosity."""
    omega_plus, omega_minus = lbm.rates(nu, "trt")
    tau_plus, tau_minus = 1.0 / omega_plus, 1.0 / omega_minus
    assert tau_plus == pytest.approx(nu / lbm.CS2 + 0.5)
    assert (tau_plus - 0.5) * (tau_minus - 0.5) == pytest.approx(lbm.LAMBDA)
    same_plus, same_minus = lbm.rates(nu, "bgk")
    assert same_plus == same_minus == omega_plus


def test_an_unknown_relaxation_scheme_is_refused(lbm):
    with pytest.raises(ValueError, match="neither"):
        lbm.rates(0.1, "mrt")


def test_an_all_solid_lattice_is_refused_rather_than_divided_by(lbm):
    with pytest.raises(ValueError, match="no fluid"):
        lbm.solve(np.ones((4, 4), dtype=bool), 0.1)


# -- the closed-form case -----------------------------------------------------------


def test_centres_put_the_walls_on_the_link_midpoints(lbm):
    """Half a cell, and every length in every case depends on it."""
    assert lbm.centres(4) == pytest.approx([0.125, 0.375, 0.625, 0.875])
    assert lbm.centres(2)[0] == pytest.approx(0.25)


@pytest.mark.parametrize("nu", [0.02, 0.1, 0.4])
def test_the_trt_channel_is_exact_at_every_viscosity(lbm, nu):
    """The claim the default scheme is chosen for, as a measurement.

    Force-driven plane Poiseuille has u(y) = G/(2*mu)*y*(H - y) exactly. With TRT at
    LAMBDA = 3/16 the discrete solution is that parabola to the convergence residual at
    any viscosity -- the error below is 1e-9 of u_max, which is the residual and not a
    discretisation error, and it does not grow as nu changes by a factor of twenty.
    """
    result = lbm.poiseuille(n=16, nu=nu, steps=None, tolerance=1e-10,
                            check_every=200, max_steps=200_000, progress=False)
    error = np.abs(result["u_profile"] - result["u_exact"]).max()
    assert error / result["u_max_exact"] < 1e-7, f"nu {nu}: {error:.3e}"


def test_bgk_is_exact_at_one_viscosity_and_not_at_the_others(lbm):
    """What makes the test above worth having.

    BGK is TRT with tau_minus = tau_plus, so it satisfies the 3/16 condition only where
    (tau - 1/2)^2 = 3/16, at tau = 1/2 + sqrt(3)/4 and so nu = sqrt(3)/12 = 0.14434.
    There BGK is exact too; a factor of four away in either direction the wall has moved
    off the midpoint and the profile carries it.
    """
    exact_nu = math.sqrt(3.0) / 12.0
    at_the_magic_value = lbm.poiseuille(n=16, nu=exact_nu, tolerance=1e-10,
                                        check_every=200, max_steps=200_000,
                                        scheme="bgk", progress=False)
    error = (np.abs(at_the_magic_value["u_profile"] - at_the_magic_value["u_exact"]).max()
             / at_the_magic_value["u_max_exact"])
    assert error < 1e-7, f"BGK at nu = sqrt(3)/12 should be exact too: {error:.3e}"

    for nu in (exact_nu / 4.0, exact_nu * 4.0):
        off = lbm.poiseuille(n=16, nu=nu, tolerance=1e-10, check_every=200,
                             max_steps=200_000, scheme="bgk", progress=False)
        wrong = (np.abs(off["u_profile"] - off["u_exact"]).max() / off["u_max_exact"])
        assert wrong > 1e-4, f"BGK at nu {nu} looks exact, so the test proves nothing"


def test_the_channel_is_driven_by_the_force_it_reports(lbm):
    """The reported force and the reported exact peak are the same statement twice,
    and a template that disagreed with its own arithmetic is how a unit error ships."""
    result = lbm.poiseuille(n=8, nu=0.1, u_max=0.03, steps=10, progress=False)
    assert result["u_max_exact"] == pytest.approx(0.03)
    assert result["u_max_exact"] == pytest.approx(
        result["force"] * result["height"] ** 2 / (8.0 * result["nu"]))


# -- the cavity ---------------------------------------------------------------------


def test_the_cavity_is_closed_and_the_lid_is_the_only_thing_moving(lbm):
    result = lbm.cavity(n=8, re=100.0, steps=200, progress=False)
    assert result["solid"][0].all() and result["solid"][-1].all()
    assert result["solid"][:, 0].all() and result["solid"][:, -1].all()
    # The fluid is driven: positive u near the lid, and the recirculation sends it back.
    assert result["u_centreline"][-1] > 0.1
    assert result["u_centreline"][0] < 0.0
    # Ghia's normalisation: the lid is 1, so nothing in the cavity exceeds it.
    assert np.nanmax(np.abs(result["u_centreline"])) <= 1.0
    assert result["nu"] == pytest.approx(0.1 * 8 / 100.0)


def test_the_cavity_conserves_its_mass(lbm):
    """A closed box with bounce-back walls leaks no mass. It is the cheapest check that
    the streaming and the boundary condition agree about which nodes are which: a
    bounce-back applied on the wrong side loses population every step."""
    result = lbm.cavity(n=12, re=100.0, steps=400, progress=False)
    mass = np.nansum(result["rho"])
    assert mass == pytest.approx(np.count_nonzero(~result["solid"]), rel=2e-3)


def test_the_centrelines_are_sampled_either_side_of_the_middle(lbm):
    """`n` is even in both benchmark runs, so x = 0.5 is between two columns. A
    nearest-node pick would be half a cell off Ghia's station."""
    result = lbm.cavity(n=8, re=100.0, steps=50, progress=False)
    assert len(result["u_centreline"]) == 8
    assert len(result["coordinate"]) == 8
    assert result["coordinate"][0] == pytest.approx(0.0625)


def test_a_step_count_overrides_the_residual(lbm):
    result = lbm.cavity(n=6, re=100.0, steps=37, progress=False)
    assert result["steps"] == 37
    assert result["converged"] is False


# -- the reason this solver is here at all -------------------------------------------


def test_an_arbitrary_mask_is_the_geometry(lbm):
    """The claim the whole design rests on: the geometry is a boolean array.

    A cylinder in a channel, written as `(x-a)^2 + (y-b)^2 < r^2` and nothing else -- no
    mesh, no snappyHexMesh, no checkMesh, no cell quality. Three things are asserted and
    each would fail differently if the bounce-back were wrong:

    * mass is conserved over a closed, obstructed domain, to round-off;
    * the solid carries no velocity, so the mask and the field agree about where the
      fluid is;
    * the flow through the constriction speeds up by exactly what continuity demands.
      The cylinder blocks 8 of the channel's 20 fluid rows at its widest, leaving 0.6 of
      the area, so the peak velocity there must be about 1/0.6 = 1.67 times the peak in
      the clear channel. That number comes from the geometry, not from the solver, which
      is what makes it a check.
    """
    ny, nx = 22, 60
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    rows, columns = np.mgrid[0:ny, 0:nx]
    solid |= ((columns - 18) ** 2 + (rows - 11) ** 2) < 4**2

    result = lbm.solve(solid, nu=0.1, force=(2e-6, 0.0), reference_velocity=0.02,
                       tolerance=1e-8, check_every=200, max_steps=120_000,
                       progress=False)
    assert result["converged"], f"residual {result['residual']:.2e}"

    fluid = ~solid
    assert np.nansum(result["rho"]) == pytest.approx(fluid.sum(), rel=1e-6)
    assert np.isnan(result["ux"][solid]).all(), "the solid is carrying a velocity"

    clear = np.nanmax(result["ux"][:, 3])
    past = np.nanmax(result["ux"][:, 18])
    assert clear > 0, "the channel is not flowing"
    assert past / clear == pytest.approx(1.67, rel=0.2), (
        f"{past / clear:.2f}x through a constriction that leaves 0.6 of the area")


# -- the picture ---------------------------------------------------------------------


@pytest.mark.parametrize("case", ["poiseuille", "cavity"])
def test_each_case_draws_its_figure(lbm, tmp_path, case):
    """`--out` is the half of this script a person actually looks at, and it has a
    branch per case. An untested drawing path is where a reference to a key the result
    does not carry sits until somebody asks for the picture."""
    result = lbm.CASES[case](n=6, steps=20, progress=False)
    path = tmp_path / f"{case}.png"
    lbm.figure(result, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "not a png"


@pytest.mark.parametrize("case", ["poiseuille", "cavity"])
def test_each_case_reports_without_raising(lbm, case):
    result = lbm.CASES[case](n=6, steps=20, progress=False)
    text = lbm.report(result)
    assert result["case"] in text
    assert "million lattice updates" in text


def test_a_stiff_lattice_says_so_in_the_report(lbm):
    """nu = u_lid*n/Re, so a small cavity at a large Reynolds number puts tau_plus next
    to the 1/2 where the collision stops relaxing. That run cannot converge, and saying
    nothing about it is how it reads as a solver defect instead of a configuration."""
    stiff = lbm.cavity(n=8, re=5000.0, steps=10, progress=False)
    assert 1.0 / stiff["omega_plus"] < lbm.STIFF_TAU
    assert "NOTE:" in lbm.report(stiff)
    comfortable = lbm.cavity(n=8, re=100.0, steps=10, progress=False)
    assert 1.0 / comfortable["omega_plus"] > lbm.STIFF_TAU
    assert "NOTE:" not in lbm.report(comfortable)


# -- the contract the toolbox holds every script to ---------------------------------


def test_the_docstring_shows_how_to_run_it(lbm):
    doc = ast.get_docstring(ast.parse((TOOLBOX / "lbm.py").read_text(encoding="utf-8")))
    assert "python3 lbm.py" in doc


def test_it_offers_rather_than_instructs(lbm):
    """docs/design.md section 1: a toolbox script states facts and does not direct the
    session's work."""
    text = (TOOLBOX / "lbm.py").read_text(encoding="utf-8").lower()
    for imperative in ("you must", "you should", "always run", "do not proceed",
                       "before you run", "step 1"):
        assert imperative not in text, imperative


def test_it_says_what_it_is_not_for(lbm):
    """The honest half of adding a solver. An index entry that only lists strengths
    leaves the choosing to be done by whoever reads it last."""
    doc = ast.get_docstring(ast.parse((TOOLBOX / "lbm.py").read_text(encoding="utf-8")))
    assert "not for" in doc.lower()
    assert "turbulence" in doc.lower()
