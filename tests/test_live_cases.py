"""The live backend, on grids small enough to run in the ordinary test job.

Three kinds of test, and the split is deliberate.

**Things that are exactly true.** The velocity set's moments, the involutions
(`OPP[OPP[i]] == i`), mass conservation in a closed periodic box, the two layouts
agreeing node for node. These are worth testing because they are the ones a refactor
breaks silently: a layout change applied to a read and not a write, a wrapped index off
by one, a sign flipped in a pair. Each has an exact answer, so the bound is single
precision and not judgement.

**Things that are exactly true and were nearly lost.** `test_a_diagonal_link_through_two_faces_obeys_both`
is here because it did not hold: the first version of `lattice.stream` applied the first
out-of-range axis's rule and skipped the rest, which meant a wall was ignored on every
diagonal link out of an edge node, *and* the kernel then read out of bounds. It showed
up as a 2.4% error on the one case in the suite whose answer is exact. A test that
fails only on a case nobody runs on a laptop is a test that will not catch it again, so
the property is asserted directly here on a nine-cell lattice.

**The cases themselves, at resolutions that prove nothing about physics.** `smoke` and
a 16-cell Poiseuille run in seconds and check that the whole pipeline works end to end.
The graded physics is in `benchmarks/live/`, on a GPU, because a lid-driven cavity at
the resolution Ghia used is four hundred thousand lattice steps and does not belong in a
test job. What *is* asserted here is that the grading machinery is honest: that a
laminar case refuses to be graded with the subgrid model on, that every case carries a
reference, and that the tolerances in the code are the tolerances in the prose.

GPU tests are behind `needs_a_gpu`, the same `skipif` shape `conftest.needs_a_kernel`
uses, and are additionally marked `gpu` so `-m gpu` selects them on a machine that has
one.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

SPEED = 1.0e-3
"""A velocity the tiny worlds below can all carry, in m/s.

Not a round number by taste: `_tiny_world`'s scaling is built for Re 50 on ten cells,
which fixes the reference velocity, and anything above a few per cent of the lattice
speed of sound is refused. Writing 0.03 m/s here, which an earlier version of this file
did, asks for a lattice Mach number of 16.
"""

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"
REFERENCES = TOOLBOX / "references"


def _warp_is_here() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("warp") is not None
    except (ImportError, ValueError):
        return False


def _gpu_is_here() -> bool:
    if not _warp_is_here():
        return False
    try:
        import warp as wp

        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


needs_warp = pytest.mark.skipif(not _warp_is_here(),
                                reason="the [live] extra is not installed here")
needs_a_gpu = pytest.mark.skipif(
    not _gpu_is_here(),
    reason="no CUDA device here, so nothing can be measured on one")


# -- the units layer, which holds constants a sibling file also holds -------------------


def _toolbox_constant(name: str) -> float:
    """A module-level constant out of `toolbox/lbm.py`, by parsing rather than importing.

    The toolbox is not an importable package -- `tests/test_toolbox.py` asserts that
    every script in it imports only the standard library, a closed set of third-party
    modules and its siblings, because it is copied verbatim into a workspace image that
    has no Warp in it. So the agreement between the two files is checked by reading the
    source, which is what `tests/test_benchmarks_index.py` does to the same end between
    two indexes.
    """
    tree = ast.parse((TOOLBOX / "lbm.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return float(eval(compile(ast.Expression(node.value), "<lbm>", "eval")))
    raise AssertionError(f"{name} is not a module-level constant of toolbox/lbm.py")


@pytest.mark.parametrize("name", ["CS2", "LAMBDA", "STIFF_TAU"])
def test_the_units_layer_holds_the_toolbox_solvers_constants(name):
    """Copied constants with nothing checking them is how two files come to hold two
    different opinions about the speed of sound. `units.py`'s docstring promises this
    test exists; this is it."""
    from openreynolds.live import units

    assert getattr(units, name) == _toolbox_constant(name), (
        f"openreynolds/live/units.py and openreynolds/toolbox/lbm.py disagree about "
        f"{name}. They are the same quantity for the same reason and one of them has "
        f"been changed without the other"
    )


def test_the_magic_parameter_is_the_product_the_two_rates_make():
    """LAMBDA is defined as (tau_+ - 1/2)(tau_- - 1/2), so `rates` must reproduce it at
    every viscosity and not only at the one it was checked at."""
    from openreynolds.live.units import CS2, LAMBDA, rates

    for nu in (0.001, 0.01, 0.1, 0.5, 2.0):
        omega_plus, omega_minus = rates(nu, "trt")
        tau_plus, tau_minus = 1.0 / omega_plus, 1.0 / omega_minus
        assert tau_plus == pytest.approx(nu / CS2 + 0.5)
        assert (tau_plus - 0.5) * (tau_minus - 0.5) == pytest.approx(LAMBDA)


def test_bgk_satisfies_the_magic_product_at_exactly_one_viscosity():
    """The contrast that stops the above being a tautology, and the same one
    `benchmarks/lbm/` grades: with both rates equal the product is 3/16 at
    nu = sqrt(3)/12 and nowhere else, so the wall position depends on the viscosity."""
    from openreynolds.live.units import CS2, LAMBDA, rates

    exact = math.sqrt(3.0) / 12.0
    tau = exact / CS2 + 0.5
    assert (tau - 0.5) ** 2 == pytest.approx(LAMBDA)
    for nu in (exact / 4.0, exact * 4.0):
        omega, _ = rates(nu, "bgk")
        tau = 1.0 / omega
        assert (tau - 0.5) ** 2 != pytest.approx(LAMBDA, rel=0.1)


def test_every_unit_conversion_round_trips():
    from openreynolds.live.units import Scaling

    # check=False: these assert the arithmetic of the conversions, not that this
    # particular dx and dt make a runnable lattice. They do not, and need not.
    scaling = Scaling.from_dt(1e-3, 2e-4, "water", reference_velocity=0.05,
                              check=False)
    pairs = [("length", "length"), ("time", "time"), ("velocity", "velocity"),
             ("acceleration", "acceleration"), ("force", "force"), ("torque", "torque")]
    for a, _ in pairs:
        to_si = getattr(scaling, f"{a}_to_si")
        to_lattice = getattr(scaling, f"{a}_to_lattice")
        assert to_lattice(to_si(1.2345)) == pytest.approx(1.2345, rel=1e-12)
        assert to_si(to_lattice(1.2345)) == pytest.approx(1.2345, rel=1e-12)


def test_the_force_exponent_is_the_fourth_power_of_dx():
    """The one exponent in `units.py` that every drag coefficient depends on, and the
    easiest to get wrong. A force is a mass times an acceleration; the lattice's mass
    unit is rho*dx^3 because the lattice density is 1, and its acceleration unit is
    dx/dt^2."""
    from openreynolds.live.units import Scaling

    dx, dt = 2.5e-3, 1.5e-4
    scaling = Scaling.from_dt(dx, dt, "water", reference_velocity=0.01, check=False)
    rho = scaling.fluid.rho
    assert scaling.force_to_si(1.0) == pytest.approx(rho * dx ** 4 / dt ** 2)
    assert scaling.torque_to_si(1.0) == pytest.approx(rho * dx ** 5 / dt ** 2)


def test_a_supersonic_scaling_is_refused_and_says_which_knob():
    from openreynolds.live.units import MACH_MAX, Scaling, UnphysicalScaling

    with pytest.raises(UnphysicalScaling) as raised:
        Scaling.from_velocity(1e-3, 1.0, "water", u_lattice=0.5)
    said = str(raised.value)
    assert "Mach" in said and f"{MACH_MAX:g}" in said
    assert "dt" in said or "resolution" in said, "the refusal names no knob"


def test_a_stiff_scaling_is_refused_unless_it_declares_a_subgrid_model():
    """The distinction `STIFF_TAU` exists for: a resolved run below the floor is a
    mistake, and an under-resolved large-eddy run below it is the only thing a robot's
    timestep allows. The second is permitted and reports itself."""
    from openreynolds.live.units import STIFF_TAU, Scaling, UnphysicalScaling

    hard = dict(dx=0.02, dt_robot=2.0e-3, fluid_spec="air", peak_velocity=10.0)
    with pytest.raises(UnphysicalScaling) as raised:
        Scaling.for_coupling(**hard, scheme="trt", les=False)
    assert "tau_plus" in str(raised.value)
    assert "smagorinsky" in str(raised.value), "the refusal does not name the way out"

    scaling, substeps = Scaling.for_coupling(**hard, scheme="regularized", les=True)
    assert substeps >= 1
    assert scaling.tau_plus <= STIFF_TAU
    notes = " ".join(scaling.notes())
    assert "under-resolved" in notes, "a stiff large-eddy run does not say so"
    assert any("under-resolved" in line for line in scaling.report())


# -- the velocity set, which is arithmetic -----------------------------------------------


@needs_warp
def test_the_velocity_set_satisfies_the_moments_it_is_a_quadrature_for():
    """D3Q19's weights and directions are a quadrature of the Maxwellian, and these four
    moments are what being one means. If any fails, the equilibrium is not second-order
    accurate and nothing downstream is worth measuring."""
    from openreynolds.live.lattice import C_NP, CS2, W_NP

    assert W_NP.sum() == pytest.approx(1.0, abs=1e-14)
    assert (W_NP[None, :] * C_NP).sum(axis=1) == pytest.approx(np.zeros(3), abs=1e-14)
    second = (W_NP[None, None, :] * C_NP[:, None, :] * C_NP[None, :, :]).sum(axis=2)
    assert second == pytest.approx(CS2 * np.eye(3), abs=1e-14)
    third = (W_NP[None, None, None, :] * C_NP[:, None, None, :]
             * C_NP[None, :, None, :] * C_NP[None, None, :, :]).sum(axis=3)
    assert third == pytest.approx(np.zeros((3, 3, 3)), abs=1e-14), (
        "the third moment is not zero, so the set is not symmetric under c -> -c"
    )


@needs_warp
def test_the_opposite_and_mirror_tables_are_involutions():
    from openreynolds.live.lattice import C_NP, MIRROR_NP, OPP_NP, Q

    assert OPP_NP[OPP_NP].tolist() == list(range(Q))
    assert (C_NP[:, OPP_NP] == -C_NP).all()
    for axis in range(3):
        mirror = MIRROR_NP[axis]
        assert mirror[mirror].tolist() == list(range(Q))
        flipped = C_NP.copy()
        flipped[axis] = -flipped[axis]
        assert (C_NP[:, mirror] == flipped).all()


@needs_warp
def test_the_two_layouts_allocate_the_same_number_of_cells():
    from openreynolds.live.lattice import Q, lattice_shape

    shape = lattice_shape((4, 5, 6))
    assert math.prod(shape) == Q * 4 * 5 * 6
    assert set(shape) == {Q, 4, 5, 6} or sorted(shape) == sorted((Q, 4, 5, 6))


# -- the kernels, on lattices of nine cells ---------------------------------------------


def _tiny_world(faces="periodic", cells=9, **kwargs):
    from openreynolds.live.units import Scaling
    from openreynolds.live.world import Domain, FluidWorld

    dx = 1.0e-3
    # Through `for_reynolds` and not `from_dt`: a timestep picked by hand is almost
    # always a stiff one. dt = 1e-4 s at dx = 1 mm in water is nu*dt/dx^2 = 1e-4, a
    # tau_plus of 0.5003, which `units.Scaling` refuses -- and it is right to. Asking
    # for Re 50 on ten cells gives nu_lattice 0.01 and tau_plus 0.53.
    scaling = Scaling.for_reynolds(dx, 10 * dx, 50.0, "water", u_lattice=0.05)
    size = (cells * dx, cells * dx, cells * dx)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=size, faces=faces,
                    **kwargs.pop("domain", {}))
    return FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                      substeps=1, device="cpu", **kwargs), scaling


@needs_warp
def test_a_closed_periodic_box_conserves_mass_to_single_precision():
    world, _ = _tiny_world()
    world.initialise(velocity=(0.02 * SPEED, -0.01 * SPEED, 0.005 * SPEED))
    before = float(np.nansum(world.fields()["rho"]))
    for _ in range(50):
        world.step()
    after = float(np.nansum(world.fields()["rho"]))
    assert abs(after / before - 1.0) < 1e-5


@needs_warp
def test_a_closed_periodic_box_with_nothing_in_it_conserves_momentum():
    """No body, no force, every face periodic: the momentum is set by the initial
    condition and nothing can change it. Exactly, to round-off, and this is the cheapest
    check that the streaming step is a permutation."""
    world, _ = _tiny_world()
    world.initialise(velocity=(0.02 * SPEED, -0.01 * SPEED, 0.005 * SPEED))
    before = np.asarray(world.momentum())
    for _ in range(50):
        world.step()
    after = np.asarray(world.momentum())
    assert after == pytest.approx(before, rel=2e-4), (
        "a closed box with no body and no force changed its own momentum"
    )


@needs_warp
def test_a_uniform_flow_in_a_periodic_box_stays_uniform():
    """Nothing to disturb it, so every node must hold the same velocity for ever. Catches
    a streaming index that is off by one, which shows up as a travelling ripple."""
    world, scaling = _tiny_world()
    speed = 0.6 * scaling.reference_velocity
    world.initialise(velocity=(speed, 0.0, 0.0))
    for _ in range(40):
        world.step()
    u = world.fields()["u"]
    assert float(np.nanstd(u[..., 0])) < 1e-4 * scaling.reference_velocity + 1e-12
    assert float(np.nanmean(u[..., 0])) == pytest.approx(speed, rel=1e-4)


@needs_warp
def test_a_diagonal_link_through_two_faces_obeys_both():
    """The bug this file's docstring is about, asserted directly.

    A box walled on y and periodic on x and z, filled with fluid moving along x. The
    walls must stop nothing in x (a flow parallel to a no-slip wall is still sheared by
    it, so the check is not "the velocity is unchanged") but they must not leak: the
    total mass is conserved and no velocity is non-finite. The specific failure the old
    code produced was an out-of-bounds read on every diagonal link out of an edge node,
    which at nine cells is a large fraction of them, and the symptom was a mass drift
    and a 2.4% error on the exact Poiseuille case.

    So: mass to round-off, every value finite, and -- the part that is only true when
    both faces are honoured -- the flow is symmetric about the channel's mid-plane,
    because the two walls are identical and the initial condition is.
    """
    world, _ = _tiny_world(faces={"-x": "periodic", "+x": "periodic",
                                  "-y": "wall", "+y": "wall",
                                  "-z": "periodic", "+z": "periodic"})
    world.initialise(velocity=(0.6 * world.scaling.reference_velocity, 0.0, 0.0))
    before = float(np.nansum(world.fields()["rho"]))
    for _ in range(60):
        world.step()
    got = world.fields()
    assert np.isfinite(got["rho"]).all(), "a density went non-finite"
    assert np.isfinite(got["u"]).all(), "a velocity went non-finite"
    assert abs(float(np.nansum(got["rho"])) / before - 1.0) < 1e-5

    profile = np.nanmean(got["u"][..., 0], axis=(0, 2))
    assert profile == pytest.approx(profile[::-1], rel=1e-3, abs=1e-9), (
        "the channel is not symmetric about its mid-plane, so the two walls are not "
        "being treated alike"
    )


@needs_warp
def test_a_slip_face_does_not_slow_a_flow_along_it():
    """Free slip is specular reflection: a uniform flow parallel to the face must be
    untouched by it, which a no-slip wall would shear and a broken mirror table would
    scramble."""
    world, _ = _tiny_world(faces={"-x": "periodic", "+x": "periodic",
                                  "-y": "slip", "+y": "slip",
                                  "-z": "periodic", "+z": "periodic"})
    speed = 0.6 * world.scaling.reference_velocity
    world.initialise(velocity=(speed, 0.0, 0.0))
    for _ in range(60):
        world.step()
    u = world.fields()["u"][..., 0]
    assert float(np.nanmin(u)) == pytest.approx(speed, rel=2e-3), (
        "a slip face slowed a flow running along it"
    )


@needs_warp
def test_a_moving_wall_drags_the_fluid_the_way_it_moves():
    """The Ladd term's sign, which is the one thing in the boundary condition that a
    plausible-looking field hides: a lid moving in +x that dragged the fluid in -x would
    still produce a tidy recirculation."""
    world, scaling = _tiny_world(
        faces={"-x": "periodic", "+x": "periodic", "-y": "wall", "+y": "wall",
               "-z": "periodic", "+z": "periodic"},
        domain={"face_velocity": {"+y": (1.0e-3, 0.0, 0.0)}})
    world.initialise()
    for _ in range(200):
        world.step()
    u = np.nanmean(world.fields()["u"][..., 0], axis=(0, 2))
    assert u[-1] > 0.0, "the fluid next to the lid is not moving the way the lid is"
    assert u[-1] > u[0], "the profile does not increase towards the lid"


@needs_warp
def test_a_body_in_a_stream_feels_a_downstream_force_and_no_lift():
    """A sphere on the axis of a symmetric domain: the drag is positive and the two
    transverse components are zero by symmetry. The second half is the better test --
    a sign error or a mis-indexed direction in the momentum exchange shows up as lift
    on a symmetric body long before it shows up in the drag."""
    from openreynolds.live.units import Scaling
    from openreynolds.live.world import Domain, FluidWorld

    diameter = 4.0e-3
    dx = diameter / 6.0
    scaling = Scaling.for_reynolds(dx, diameter, 20.0, "water", u_lattice=0.05)
    speed = scaling.reference_velocity
    size = (24 * dx, 16 * dx, 16 * dx)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                    faces={"-x": "inlet", "+x": "outlet", "-y": "slip", "+y": "slip",
                           "-z": "slip", "+z": "slip"},
                    face_velocity={"-x": (speed, 0.0, 0.0)})
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, device="cpu")
    centre = (8 * dx, size[1] / 2.0, size[2] / 2.0)
    body = world.add_body(position=centre, label="sphere")
    # No `position=`: a shape's offset is in its BODY's frame, and the body is at the
    # sphere's centre. Giving both the world position puts the sphere at twice the
    # coordinate, which is what `test_the_voxeliser_puts_the_right_number_of_cells...`
    # caught in `cases.py` -- one octant of the sphere inside a corner of the box, and
    # a transverse force 220 times the drag.
    world.add_shape("sphere", body=body, scale=(diameter / 2.0, 0.0, 0.0))
    world.initialise(velocity=(speed, 0.0, 0.0))
    for _ in range(300):
        world.step()
    force, torque = world.wrench(body)
    assert force[0] > 0.0, "the drag is not downstream"
    transverse = max(abs(force[1]), abs(force[2]))
    assert transverse < 0.05 * abs(force[0]), (
        f"a sphere on the axis feels {transverse / abs(force[0]):.1%} of its drag "
        f"sideways; the momentum exchange is not symmetric"
    )
    assert all(abs(t) < 0.05 * abs(force[0]) * size[1] for t in torque)


@needs_warp
def test_the_voxeliser_puts_the_right_number_of_cells_inside_a_sphere():
    """`phi < 0` is the solid, and its volume has to be the sphere's to the
    discretisation. The buoyancy of a falling body is computed from this count
    (`couple.py`), so an error here is an error in a graded terminal velocity."""
    world, _ = _tiny_world(cells=24)
    radius = 6.0 * world.dx
    centre = tuple(12 * world.dx for _ in range(3))
    body = world.add_body(position=centre)
    world.add_shape("sphere", body=body, scale=(radius, 0.0, 0.0))
    world.initialise()
    cells = int((world.phi.numpy() < 0.0).sum())
    exact = 4.0 / 3.0 * math.pi * 6.0 ** 3
    assert cells == pytest.approx(exact, rel=0.08), (
        f"{cells} solid nodes against an exact {exact:.0f} at six cells of radius"
    )


@needs_warp
def test_a_body_that_moves_off_a_node_gets_it_refilled():
    """The counter `DECISIONS_LIVE.md` D5 promises. A body that uncovers no node has not
    exercised the refilling, so the count being non-zero is the precondition for the
    translating-sphere case meaning anything."""
    world, _ = _tiny_world(cells=20)
    radius = 4.0 * world.dx
    start = [10 * world.dx, 10 * world.dx, 10 * world.dx]
    body = world.add_body(position=tuple(start))
    world.add_shape("sphere", body=body, scale=(radius, 0.0, 0.0))
    world.initialise()
    for _ in range(12):
        start[0] += world.dx
        world.set_body_state(body, position=tuple(start))
        world.step()
    assert int(world._refilled.numpy()[0]) > 0, "a body moved and refilled nothing"


# -- the two layouts give the same answer ------------------------------------------------


@needs_warp
def test_both_array_layouts_give_the_same_field():
    """The layout is baked into the generated code at import, so this runs the same case
    in two subprocesses and compares. It is the only test that can catch an accessor
    applied to a read and not to a write, which is a one-character mistake that leaves
    one layout correct and the other quietly wrong."""
    import os
    import subprocess
    import sys

    script = (
        "import json,numpy as np;"
        "from openreynolds.live.units import Scaling;"
        "from openreynolds.live.world import Domain, FluidWorld;"
        "dx=1e-3;s=Scaling.for_reynolds(dx,10*dx,50.0,'water',u_lattice=0.05);"
        "d=Domain(origin=(0,0,0),size=(9*dx,9*dx,9*dx),faces={'-x':'periodic',"
        "'+x':'periodic','-y':'wall','+y':'wall','-z':'periodic','+z':'periodic'},"
        "face_velocity={'+y':(s.reference_velocity,0,0)});"
        "w=FluidWorld(domain=d,dx=dx,fluid='water',scaling=s,substeps=1,device='cpu');"
        "w.initialise()\n"
        "for _ in range(80): w.step()\n"
        "u=w.fields()['u'];print(json.dumps([float(np.nansum(u)),"
        "float(np.nansum(np.abs(u))),float(np.nanmax(u[...,0]))]))"
    )
    answers = {}
    for layout in ("soa", "aos"):
        done = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", script], capture_output=True, text=True,
            env=dict(os.environ, OPENREYNOLDS_LIVE_LAYOUT=layout, PYTHONUTF8="1"),
            timeout=900)
        assert done.returncode == 0, f"the {layout} run failed: {done.stderr[-500:]}"
        line = [row for row in done.stdout.splitlines() if row.startswith("[")][-1]
        answers[layout] = json.loads(line)
    assert answers["soa"] == pytest.approx(answers["aos"], rel=1e-5), (
        f"the two layouts disagree: {answers}"
    )


# -- the cases, and whether their grading is honest --------------------------------------


def test_every_case_carries_a_milestone_and_a_sentence():
    from openreynolds.live import cases

    for name, case in cases.CASES.items():
        assert case.name == name
        assert case.about and len(case.about) > 30, f"{name} has no useful description"
        assert case.milestone in ("M0", "M1", "M2", "M3", "M5", "M6", "M7"), name
        assert callable(case.runner)


def test_the_catalogue_lists_every_case():
    from openreynolds.live import cases

    listed = "\n".join(cases.describe_cases())
    for name in cases.CASES:
        assert name in listed, f"{name} is a case and the catalogue does not name it"


def test_an_unknown_case_is_refused_by_name():
    from openreynolds.live import cases
    from openreynolds.live.units import UnphysicalScaling

    with pytest.raises(UnphysicalScaling) as raised:
        cases.run_case("not_a_case")
    assert "not_a_case" in str(raised.value)
    for name in cases.CASES:
        assert name in str(raised.value), "the refusal does not list the cases"


@pytest.mark.parametrize("name", sorted({"poiseuille", "cavity", "cylinder", "sphere",
                                         "translating_sphere", "falling_sphere",
                                         "momentum", "smoke"}))
def test_a_laminar_case_refuses_to_be_graded_with_the_subgrid_model_on(name):
    """The rule the module docstring is about. An eddy viscosity in a laminar flow
    lowers the effective Reynolds number, so the drag comes out closer to the published
    value for a reason that has nothing to do with the solver being right. A flattering
    number that nothing refuses is the worst failure mode this suite has."""
    from openreynolds.live import cases
    from openreynolds.live.units import Scaling, UnphysicalScaling

    assert name in cases.LAMINAR
    scaling = Scaling.for_reynolds(1e-3, 2e-2, 100.0, "water")
    with pytest.raises(UnphysicalScaling) as raised:
        cases._official(name, scaling, 0.12)
    said = str(raised.value)
    assert "laminar" in said and "smagorinsky" in said
    assert "flattered" in said or "effective Reynolds" in said


def test_the_rotor_cases_declare_the_subgrid_model_rather_than_hiding_it():
    from openreynolds.live import cases
    from openreynolds.live.units import Scaling

    scaling, _ = Scaling.for_coupling(0.01, 2e-3, "air", peak_velocity=20.0,
                                      scheme="regularized", les=True)
    notes = " ".join(cases._official("ground_effect", scaling, 0.12))
    assert "ON" in notes and "0.12" in notes
    assert "model's" in notes or "Reynolds number no affordable lattice" in notes


def test_every_reference_says_where_it_was_read_and_what_is_wrong_with_it():
    """The rule `benchmarks/README.md` exists for, applied to this file's own table. A
    reference with no provenance is a number somebody remembered, and this repository
    has four studies on record that did exactly that."""
    from openreynolds.live.cases import REFERENCES

    for name, entry in REFERENCES.items():
        assert entry.get("source"), f"{name} names no source"
        assert entry.get("read"), f"{name} does not say what was actually read"
        assert entry.get("caveat"), f"{name} states no caveat, which no reference has"
        assert entry.get("tolerance") is not None, f"{name} has no tolerance"
        assert entry.get("tolerance_is"), f"{name} does not say what its tolerance is on"


def test_the_schafer_turek_intervals_are_the_papers_and_not_a_midpoint():
    """2D-2 has no single reference value, only intervals, and the code must carry them
    as intervals: a midpoint hard-coded as "the" answer is how an interval becomes a
    number nobody can argue with."""
    from openreynolds.live.cases import REFERENCES

    entry = REFERENCES["schafer_turek_2d2"]
    assert entry["cd_max"] == (3.2200, 3.2400)
    assert entry["cl_max"] == (0.9900, 1.0100)
    assert entry["strouhal"] == (0.2950, 0.3050)
    assert "no single reference value" in entry["caveat"]
    assert "MEAN" in entry["caveat"], (
        "the Strouhal number's reference velocity is the mean and not the peak, and the "
        "caveat has to say so: using 1.5 instead of 1.0 is a 50% error"
    )


def test_the_cheeseman_bennett_constant_is_four_and_the_model_is_at_constant_power():
    from openreynolds.live import cases

    assert "16 Z^2" in cases.REFERENCES["cheeseman_bennett"]["formula"]
    assert "CONSTANT POWER" in cases.REFERENCES["cheeseman_bennett"]["caveat"]
    # The two forms are each other's reciprocal by construction, which is the whole
    # argument for grading the induced velocity instead of the thrust.
    for z_over_r in (0.8, 1.0, 2.0, 5.0):
        thrust = cases._cheeseman_bennett(1.0, z_over_r)
        induced = cases._induced_ratio(1.0, z_over_r)
        assert thrust * induced == pytest.approx(1.0, rel=1e-12)
    # Far from the ground the effect vanishes; near it the thrust ratio exceeds one.
    assert cases._induced_ratio(1.0, 50.0) == pytest.approx(1.0, abs=1e-4)
    assert cases._cheeseman_bennett(1.0, 1.0) > 1.0


def test_the_sphere_drag_curve_is_the_toolboxs_and_not_a_second_copy():
    from openreynolds.live.cases import sphere_drag

    curve = sphere_drag()
    for re in (1.0, 10.0, 100.0, 500.0):
        assert curve.schiller_naumann_cd(re) == pytest.approx(
            24.0 / re * (1.0 + 0.15 * re ** 0.687))
    assert curve.schiller_naumann_cd(1e-6) / curve.stokes_cd(1e-6) == pytest.approx(
        1.0, rel=1e-3), "Schiller-Naumann does not reduce to Stokes in its own limit"


# -- the documents the code points at ----------------------------------------------------


def test_every_decision_the_code_cites_is_written_down():
    """`DECISIONS_LIVE.md D9` in a docstring is a promise that D9 exists.

    It did not. Three decisions were cited from five files -- D8 on the velocity set, D9
    on gravity and buoyancy, D10 on the quadrotor's geometry -- while the document stopped
    at D7, so a reader following the pointer found nothing and had to take the
    surrounding sentence on trust. That is the same failure
    `tests/test_toolbox.py::test_the_toolbox_index_names_nothing_that_is_not_there` was
    written for: an index running ahead of what it indexes fails nothing unless something
    checks it.

    Both directions are checked. A cited decision must exist, and an existing one must be
    cited or referenced from somewhere -- a decision nobody points at is a decision nobody
    reads, and this file counts `DECISIONS_LIVE.md` itself as a referrer so that the
    document's own cross-references count.
    """
    import re

    document = ROOT / "DECISIONS_LIVE.md"
    assert document.is_file(), "DECISIONS_LIVE.md is gone"
    written = set(re.findall(r"^## (D\d+)\.", document.read_text(encoding="utf-8"),
                             re.M))
    assert written, "DECISIONS_LIVE.md has no numbered decisions"

    cited: dict[str, set[str]] = {}
    sources = [*(ROOT / "openreynolds").rglob("*.py"), *ROOT.glob("*.md"),
               *(ROOT / "docs").glob("*.md"), *(ROOT / "benchmarks").rglob("*.md"),
               *(ROOT / "benchmarks").rglob("*.py"), *(ROOT / "tests").glob("*.py")]
    for path in sources:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for found in re.finditer(r"DECISIONS_LIVE\.md\W{0,3}(D\d+)", text):
            cited.setdefault(found.group(1), set()).add(path.name)

    dangling = {name: sorted(where) for name, where in cited.items()
                if name not in written}
    assert not dangling, (
        "the code cites decisions that DECISIONS_LIVE.md does not contain: "
        + "; ".join(f"{name} from {', '.join(where)}"
                    for name, where in sorted(dangling.items()))
    )
    unreferenced = sorted(written - set(cited), key=lambda s: int(s[1:]))
    assert not unreferenced, (
        f"DECISIONS_LIVE.md contains {', '.join(unreferenced)}, which nothing points "
        f"at. A decision nobody references is a decision nobody reads: cite it from the "
        f"code it governs, or from another decision"
    )


# -- the Ghia tables ---------------------------------------------------------------------


@pytest.mark.parametrize("reynolds", [100, 1000])
def test_the_ghia_tables_are_there_and_have_seventeen_points_each(reynolds):
    data = json.loads((REFERENCES / f"ghia_re{reynolds}.json").read_text(encoding="utf-8"))
    for key in ("table_1_u_on_vertical_centreline",
                "table_2_v_on_horizontal_centreline"):
        rows = data[key]
        assert len(rows) == 17, f"Re {reynolds} {key} has {len(rows)} rows"
        assert [r[0] for r in rows] == sorted(r[0] for r in rows)
        assert rows[0][0] == 0.0 and rows[-1][0] == 1.0
    assert "Ghia" in data["source"] and "387-411" in data["source"]


def test_the_two_ghia_files_are_sampled_at_the_same_seventeen_ordinates():
    """The cheapest check that the Re 100 column was read off the right rows: Ghia's
    tables use one grid for both Reynolds numbers, so the coordinates must match the
    file that was already here. They do, which is why the Re 100 numbers are trusted
    despite Table II resting on a single OCR of a 1982 scan."""
    hundred = json.loads((REFERENCES / "ghia_re100.json").read_text(encoding="utf-8"))
    thousand = json.loads((REFERENCES / "ghia_re1000.json").read_text(encoding="utf-8"))
    for key in ("table_1_u_on_vertical_centreline",
                "table_2_v_on_horizontal_centreline"):
        assert [r[0] for r in hundred[key]] == [r[0] for r in thousand[key]]


def test_the_re100_file_records_that_table_two_has_one_source():
    """An honest "this rests on one reading" is worth more than a confident number, and
    the file has to carry it where a reader will see it."""
    data = json.loads((REFERENCES / "ghia_re100.json").read_text(encoding="utf-8"))
    note = data["note"]
    assert "scan" in note and ("one" in note or "single" in note)


def test_the_toolbox_index_names_the_new_reference_table():
    """`tests/test_toolbox.py` checks the index against the scripts; the reference tables
    are indexed in the same README and nothing checked those."""
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    for path in sorted(REFERENCES.glob("*.json")):
        assert f"references/{path.name}" in index, (
            f"{path.name} is in the toolbox's references and its index does not name it"
        )


# -- the cases actually run ---------------------------------------------------------------


@needs_warp
def test_the_smoke_case_passes_every_check(tmp_path):
    from openreynolds.live.cases import run_case

    result = run_case("smoke", device="cpu", out_dir=tmp_path)
    failed = [c for c in result["checks"] if c["passed"] is False]
    assert not failed, f"smoke failed: {[c['name'] for c in failed]}"
    assert result["passed"]


@needs_warp
def test_the_exact_case_is_exact(tmp_path):
    """Poiseuille at sixteen cells. The scheme's wall sits on the link midpoint at any
    viscosity, so this is the analytic parabola to the convergence residual and not to
    second order -- and it is the case that caught the two-face streaming bug."""
    from openreynolds.live.cases import run_case

    result = run_case("poiseuille", device="cpu", out_dir=tmp_path, cells=16,
                      nu_lattice=0.1)
    checks = {c["name"]: c for c in result["checks"]}
    error = checks["profile, max |u - u_exact| / u_max"]
    assert error["passed"], f"the exact case is not exact: {error['measured']}"
    assert float(error["measured"]) < 2e-3


@needs_warp
def test_a_figure_comes_out_as_rgb(tmp_path):
    """A saved PNG with an alpha channel renders as a black panel wherever a thumbnailer
    composites it, which this project has been bitten by on its own site. matplotlib
    writes RGBA by default."""
    from openreynolds.live.cases import run_case

    result = run_case("smoke", device="cpu", out_dir=tmp_path, figure=True)
    assert result["figures"], "the case drew nothing"
    pillow = pytest.importorskip("PIL.Image")
    for path in result["figures"]:
        if str(path).endswith(".png"):
            with pillow.open(path) as image:
                assert image.mode == "RGB", f"{path} is {image.mode}, not RGB"


# -- the case file and its backend selector -----------------------------------------------


def test_a_case_file_selects_the_backend(tmp_path):
    from openreynolds.live.cases import run_case_file

    path = tmp_path / "case.yaml"
    path.write_text("case: cylinder\nbackend: openfoam\n", encoding="utf-8")
    result = run_case_file(path)
    assert result["backend"] == "openfoam"
    text = "\n".join(result["report"])
    assert "crosscheck" in text, "the openfoam side does not point at the comparison"
    assert "does not start a workspace" in text


def test_a_case_file_with_an_unknown_backend_is_refused(tmp_path):
    from openreynolds.live.cases import run_case_file
    from openreynolds.live.units import UnphysicalScaling

    path = tmp_path / "case.yaml"
    path.write_text("case: cylinder\nbackend: xlb\n", encoding="utf-8")
    with pytest.raises(UnphysicalScaling) as raised:
        run_case_file(path)
    assert "live" in str(raised.value) and "openfoam" in str(raised.value)


def test_the_case_file_parser_reads_scalars_blocks_and_lists():
    from openreynolds.live.cases import _parse_case_file

    data = _parse_case_file(
        "case: cavity\n"
        "resolution: 64\n"
        "smagorinsky: 0\n"
        "animate: true\n"
        "resolutions: [20, 30, 40]\n"
        "# a comment\n"
        "domain:\n"
        "  size: 1.0\n"
        "  faces: wall\n"
    )
    assert data["case"] == "cavity"
    assert data["resolution"] == 64 and isinstance(data["resolution"], int)
    assert data["smagorinsky"] == 0
    assert data["animate"] is True
    assert data["resolutions"] == (20, 30, 40)
    assert data["domain"] == {"size": 1.0, "faces": "wall"}


# -- the GPU, behind a marker --------------------------------------------------------------


@pytest.mark.gpu
@needs_a_gpu
def test_the_smoke_case_passes_on_a_gpu(tmp_path):
    from openreynolds.live.cases import run_case

    result = run_case("smoke", device="cuda:0", out_dir=tmp_path)
    failed = [c for c in result["checks"] if c["passed"] is False]
    assert not failed, f"smoke failed on the GPU: {[c['name'] for c in failed]}"


@pytest.mark.gpu
@needs_a_gpu
def test_the_gpu_and_the_cpu_agree_on_the_same_small_case():
    """The one check that the CUDA kernels are the CPU kernels. They are generated from
    the same source but not into the same code, and the layout flips between devices, so
    bit-identity is not expected and agreement to single precision is."""
    from openreynolds.live.units import Scaling
    from openreynolds.live.world import Domain, FluidWorld

    answers = []
    for device in ("cpu", "cuda:0"):
        dx = 1.0e-3
        scaling = Scaling.for_reynolds(dx, 10 * dx, 50.0, "water", u_lattice=0.05)
        domain = Domain(origin=(0.0, 0.0, 0.0), size=(9 * dx, 9 * dx, 9 * dx),
                        faces={"-x": "periodic", "+x": "periodic", "-y": "wall",
                               "+y": "wall", "-z": "periodic", "+z": "periodic"},
                        face_velocity={"+y": (scaling.reference_velocity, 0.0, 0.0)})
        world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                           substeps=1, device=device)
        world.initialise()
        for _ in range(80):
            world.step()
        answers.append(float(np.nansum(np.abs(world.fields()["u"]))))
    assert answers[0] == pytest.approx(answers[1], rel=1e-4), (
        f"the CPU and the GPU disagree: {answers}"
    )


@pytest.mark.gpu
@needs_a_gpu
def test_the_benchmark_reports_a_rate_on_the_gpu():
    from openreynolds.live.cases import benchmark

    result = benchmark(resolutions=(32,), steps=20, device="cuda:0")
    assert result["rows"] and result["rows"][0]["mlups"] > 0.0
    assert "cuda" in result["device"]


def test_no_case_builds_a_dict_key_out_of_chr_39():
    """A narrow guard on one repair pattern, because it cost 1,087 seconds of H100.

    An f-string in `run_ground_effect` needed a single-quoted dict key inside a
    single-quoted f-string, and the repair was `r[chr(39) + 'z_over_r' + chr(39)]`. That
    does not escape a quote; it **builds a different key**, `"'z_over_r'"` with the
    quotes as characters, and raises `KeyError` the first time the branch containing it
    is reached. The branch was a report line that only fires when a measured ratio
    exceeds 1, so the case ran to completion, computed every graded number, and then
    died formatting them.

    The general lesson has no cheap test -- a report line executes after the expensive
    work, and nothing here can afford to run the GPU cases to find out. This guard is
    just the specific pattern, named so that the next person reaching for `chr(39)` to
    get a quote past an f-string parser hoists the value into a variable instead.
    """
    offenders = []
    for path in sorted((ROOT / "openreynolds" / "live").glob("*.py")):
        if "chr(39)" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, (
        f"{', '.join(offenders)} builds a string with chr(39). If that is standing in "
        f"for a quote inside an f-string, it is a different string than intended -- "
        f"assign the value to a name and interpolate the name"
    )


@needs_warp
def test_the_host_side_poses_are_not_conditional_on_which_feature_is_on():
    """`world._poses` must hold the body's real position, whatever else is enabled.

    It feeds three consumers -- an actuator disk's pose, the buoyancy's moment arm, and
    `FluidWorld._maybe_shift`, which re-centres a moving window on the body it follows --
    and it used to be filled only when `rotors or buoyancy`, which covers the first two
    and silently drops the third.

    A run with `follow` set, no rotors and `buoyancy=False` therefore measured its
    body's drift from the **origin** instead of from the body, fired one spurious shift
    of half the domain, and re-centred the window on empty water. The hull's
    voxelisation then eroded from 2,176 cells to 68 as it left the window, and an
    independent measurement on a larger hull saw `solid_cells` reach zero and the wrench
    return exactly 0.0 -- a body accelerating at `T/m` to four figures, flying through
    vacuum, with no error anywhere.

    This asserts the position, with every optional feature off, which is the
    configuration the old condition missed.
    """
    newton = pytest.importorskip("newton", reason="needs the Newton physics engine")
    import warp as wp

    from openreynolds.live.couple import NewtonCoupling
    from openreynolds.live.units import Scaling
    from openreynolds.live.world import Domain, FluidWorld

    # 16 cells a diameter, not 8: at Re 100 and u_lattice 0.05 the coarser lattice puts
    # tau_plus at 0.512, under `units.STIFF_TAU`, and the scaling refuses it.
    diameter, cells_per_d = 0.01, 16
    dx = diameter / cells_per_d
    scaling = Scaling.for_reynolds(dx, diameter, 100.0, "water", u_lattice=0.05)
    size = (8 * diameter, 6 * diameter, 6 * diameter)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=size, faces="slip")
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, device="cpu", follow=0, follow_slack=0.12)
    centre = tuple(v / 2.0 for v in size)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), label="sphere")
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 3000.0
    shape_cfg.has_shape_collision = False
    builder.add_shape_sphere(body, radius=diameter / 2.0, cfg=shape_cfg)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    engine = newton.solvers.SolverFeatherstone(model)

    # No rotors, buoyancy off: exactly the case the old condition skipped.
    link = NewtonCoupling(model, engine, world, dt=scaling.dt, buoyancy=False)
    assert not world.rotors
    assert link.buoyancy is False

    held = np.asarray(world._poses[0][:3])
    assert held == pytest.approx(np.asarray(centre), abs=1e-9), (
        f"world._poses[0] is {held}, not the body's position {centre}. _maybe_shift "
        f"measures drift against this, so a zero here moves the window to the wrong "
        f"place on the first step"
    )

    world.initialise()
    before = int(world.solid_cells(0))
    assert before > 0
    for _ in range(40):
        link.step()
    assert world.shifts == 0, (
        f"{world.shifts} window shifts on a body sitting still at the centre; the "
        f"drift it measured was not zero"
    )
    after = int(world.solid_cells(0))
    assert after == pytest.approx(before, rel=0.05), (
        f"the voxelised body went from {before} cells to {after}: it is leaving the "
        f"window, which is what a wrongly re-centred window looks like"
    )
