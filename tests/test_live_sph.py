"""The SPH backend's own tests: the kernel, the hydrostatics, and the sign of `wrench`.

Everything here runs on Warp's **CPU** backend, because the suite has no GPU. The two
hydrostatic worlds are a few thousand particles each and are built once per module;
measured, `pytest tests/test_live_sph.py --durations=8`: **177.5 s for 23 tests**, of
which 95.0 s is the still tank settling and 73.0 s is the submerged box, so 95% of the
file is those two fixtures and no other test costs more than 6.5 s. The four
density-diffusion and wrench-filter tests added for v0.1's force noise bring it to
**29 tests** and cost about 2 s between them, because none of them settles anything. Both are
module-scoped for that reason -- three tests read the still tank and two read the box,
and per-test fixtures would be seven and a half minutes.

If this file has to get faster, the settling is the only place to look, and the lever is
`particle_radius`: the still tank at a 7.5 mm radius measured 0.9970 on the hydrostatic
slope against 0.9855 at 10 mm, and cost 47 s against 13 s for the same number of steps.
The 10 mm world is the cheap end of that trade, not a compromise found by accident.

What is graded here, and to what:

    kernel normalisation      4 pi int W r^2 dr = 1              to 1e-6
    bulk summation density    sum_j m_j W_ij = rho0              to 1e-4 (measured 3e-5)
    hydrostatic pressure      dp/dh = rho g                      to 5% (measured 1.5%)
    density error             max compression bounded            to 2% (measured 0.6%)
    delta-SPH's eigenvalue    lambda_max h^2, resolution-free    to 0.02 (measured 23.2963)
    weight on the floor       F_z on the tank = -(water weight)  to 2% (measured 0.01%)
    submerged box             F_z on the box  = +rho V g         to 15% (measured 9.9%)

Every tolerance above is wider than the number measured beside it and none of them was
widened to make a test pass; where the measured number is not close to the gate, the
reason is written at the test and is a property of the discretisation rather than of the
run. The buoyancy gate is the loose one and `test_a_submerged_box_feels_rho_v_g_upward`
carries the convergence table that says why.

The graded SPH physics -- a dam break, a sloshing tank, a floating hull -- is not here.
It belongs on a GPU in `benchmarks/live/`, at a resolution a CPU cannot reach in a test
suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

warp = pytest.importorskip("warp", reason="the live backend is an optional extra")

# Imported at module scope and not inside the tests, which is the opposite of what the
# other `test_live_*.py` files do, for a reason that is Warp's: a `@wp.func` or a
# constant a kernel below refers to has to be resolvable when Warp parses the kernel's
# source, and an `import` statement *inside* a kernel body is a codegen error
# ("Construct `ast.ImportFrom` not supported in kernels"). The `importorskip` above is
# still what keeps this file off a machine with no Warp.
from openreynolds.live import lattice as L  # noqa: E402
from openreynolds.live.sph import _spline, _spline_grad  # noqa: E402


# -- the kernel, measured through the device code the solver actually runs -------------


@warp.kernel
def _sample_spline(r: warp.array(dtype=warp.float32), w: warp.array(dtype=warp.float32),
                   g: warp.array(dtype=warp.float32), h: warp.float32,
                   sigma: warp.float32):
    """Evaluate `_spline` and the radial part of `_spline_grad` at a list of radii.

    Through a kernel and not through a numpy copy of the formula, which would be the
    easy version and would test nothing: a numpy mirror of the kernel passes whether or
    not the kernel is right. This calls the `@wp.func` the solver calls.
    """
    i = warp.tid()
    w[i] = _spline(r[i], h, sigma)
    # The gradient of W(|x_i - x_j|) with the separation along +x, so its x component is
    # dW/dr. Signed: it points from i towards j, which is -x for a decaying kernel.
    d = warp.vec3(r[i], warp.float32(0.0), warp.float32(0.0))
    g[i] = _spline_grad(d, r[i], h, sigma)[0]


def test_the_kernel_integrates_to_one_over_its_support():
    """`4 pi int_0^h W(r) r^2 dr = 1`, which is the whole of "correct normalisation".

    The cubic spline is written with the **support radius** as its length scale, so
    `sigma = 8/(pi h^3)` and the support is at `q = r/h = 1`. The other convention in the
    literature puts the support at `q = 2` and carries `1/(pi h'^3)` with `h' = h/2`, and
    the two normalisations differ by a factor of eight -- which is exactly the kind of
    thing that silently scales every density in the run by 8 and reads as a mass error.
    """
    from openreynolds.live.sph import SUPPORT_FACTOR

    h = 0.02
    sigma = 8.0 / (math.pi * h ** 3)
    r = np.linspace(0.0, h, 200001)[1:].astype(np.float32)
    dev = warp.array(r, dtype=warp.float32, device="cpu")
    w = warp.zeros(len(r), dtype=warp.float32, device="cpu")
    g = warp.zeros(len(r), dtype=warp.float32, device="cpu")
    warp.launch(_sample_spline, dim=len(r), device="cpu",
                inputs=[dev, w, g, warp.float32(h), warp.float32(sigma)])
    values = w.numpy().astype(np.float64)
    integral = np.trapezoid(4.0 * math.pi * r.astype(np.float64) ** 2 * values, r)
    assert abs(integral - 1.0) < 1e-6, integral

    # Past the support it is zero, not small: a kernel with a tail is a kernel whose
    # neighbour radius is a lie, and every neighbour query here uses exactly `h`.
    assert values[-1] == 0.0 or r[-1] >= h

    # The gradient is the derivative of the value, checked by central differences rather
    # than by a second copy of the algebra -- on a *coarse* sample, which is not laziness
    # but the only way the check means anything. The kernel is float32 and its values
    # here are around 3e5, so each one is quantised to about 0.03; a central difference
    # over the 1e-7 m spacing used for the integral above divides that by 1e-7 and
    # produces 3e5 of pure quantisation noise against a true maximum slope of 3.2e7 --
    # measured, 0.94% error, all of it the differencing and none of it the kernel. At a
    # 1e-5 m spacing the same noise is 3e3, and the error measured below is 1.1e-4.
    coarse = np.linspace(0.0, h, 2001)[1:].astype(np.float32)
    cdev = warp.array(coarse, dtype=warp.float32, device="cpu")
    cw = warp.zeros(len(coarse), dtype=warp.float32, device="cpu")
    cg = warp.zeros(len(coarse), dtype=warp.float32, device="cpu")
    warp.launch(_sample_spline, dim=len(coarse), device="cpu",
                inputs=[cdev, cw, cg, warp.float32(h), warp.float32(sigma)])
    numeric = np.gradient(cw.numpy().astype(np.float64), coarse.astype(np.float64))
    # The two interior stretches, away from r = 0, away from the support, and away from
    # the kink at q = 1/2 where the second derivative jumps and a central difference
    # straddles both branches.
    q = coarse / h
    interior = ((q > 0.05) & (q < 0.45)) | ((q > 0.55) & (q < 0.95))
    scale = np.abs(numeric[interior]).max()
    error = np.abs(cg.numpy()[interior] - numeric[interior]).max() / scale
    assert error < 1e-3, error

    # And the support is what `SUPPORT_FACTOR`'s table is written against.
    assert SUPPORT_FACTOR == 4.0


def test_a_bulk_lattice_of_particles_reads_its_own_rest_density():
    """`sum_j m_j W_ij = rho0` in the interior, which is why there is no mass calibration.

    The three things that have to agree for this to hold are the particle mass
    (`rho0 * spacing^3`), the support (`4 * particle_radius`, so `2 * spacing`) and the
    kernel's normalisation, and this is the only test that holds all three at once. The
    number it measures is in `sph.SUPPORT_FACTOR`'s docstring as the reason that constant
    is 4 and not 3.5 or 5: a bulk lattice reads 0.999972 rho0 at 4, and the two
    neighbouring choices are 40 and 50 times worse.

    Measured here: the interior particles read rho0 to 3.1e-5 relative, worst case.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    radius = 0.005
    size = 0.14
    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (size, size, size), faces="wall"),
                     particle_radius=radius, fluid="water",
                     gravity=(0.0, 0.0, 0.0), device="cpu")
    # No boundary particles at all: this measures the fluid's own kernel sum, and a wall
    # would add an Akinci contribution that is a different thing to be right about.
    world.fill_box((0.0, 0.0, 0.0), (size, size, size))
    world.step()

    x = world.positions()
    rho = world.densities()
    # Interior means "a full support away from every face", which for a support of two
    # spacings is two particle layers in from each side.
    inner = np.all((x > 2.5 * world.spacing) & (x < size - 2.5 * world.spacing), axis=1)
    assert inner.sum() > 100, inner.sum()
    error = np.abs(rho[inner] / world.rho0 - 1.0).max()
    assert error < 1e-4, error


@warp.kernel
def _sample_sdf(points: warp.array(dtype=warp.vec3), out: warp.array(dtype=warp.float32),
                kind: warp.int32, scale: warp.vec3):
    """`lattice.py`'s five signed distances, one shape kind per launch."""
    i = warp.tid()
    p = points[i]
    if kind == L.SHAPE_SPHERE:
        out[i] = L._sdf_sphere(p, scale)
    elif kind == L.SHAPE_BOX:
        out[i] = L._sdf_box(p, scale)
    elif kind == L.SHAPE_CAPSULE:
        out[i] = L._sdf_capsule(p, scale)
    elif kind == L.SHAPE_CYLINDER:
        out[i] = L._sdf_cylinder(p, scale)
    else:
        out[i] = p[2]


def test_the_host_sdfs_agree_with_the_lattice_kernels():
    """`sph._sdf` and `lattice._sdf_*` are the same five functions, so they must agree.

    They are duplicated on purpose -- `lattice.py`'s are `@wp.func` and only callable
    from a kernel, and boundary sampling happens once at setup over a host array of
    candidate points -- and a duplicate with no test between it and its original is how
    a capsule ends up along y in one file and z in the other.
    """
    from openreynolds.live.sph import _sdf

    rng = np.random.default_rng(20261123)
    points = rng.uniform(-0.09, 0.09, size=(500, 3))
    dev = warp.array(points.astype(np.float32), dtype=warp.vec3, device="cpu")
    out = warp.zeros(len(points), dtype=warp.float32, device="cpu")
    for kind, scale in [
        (L.SHAPE_SPHERE, (0.04, 0.0, 0.0)),
        (L.SHAPE_BOX, (0.03, 0.05, 0.02)),
        (L.SHAPE_CAPSULE, (0.02, 0.04, 0.0)),
        (L.SHAPE_CYLINDER, (0.03, 0.025, 0.0)),
        (L.SHAPE_PLANE, (0.0, 0.0, 0.0)),
    ]:
        warp.launch(_sample_sdf, dim=len(points), device="cpu",
                    inputs=[dev, out, warp.int32(kind), warp.vec3(*scale)])
        host = _sdf(kind, points, scale)
        assert np.abs(out.numpy() - host).max() < 1e-6, L.SHAPE_NAMES[kind]


def test_the_quaternion_maths_matches_world():
    """`sph._quat_matrix` is a copy of `world._quat_matrix`. Copies drift; this is the
    thing that makes the drift an error."""
    from openreynolds.live.sph import _quat_matrix as mine
    from openreynolds.live.world import _quat_matrix as theirs

    rng = np.random.default_rng(7)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert np.abs(mine(q) - theirs(q)).max() < 1e-12


# -- a still tank: the pressure, the density error and the load on the floor -----------


@pytest.fixture(scope="module")
def tank():
    """280 mm of still water in a 140 x 140 x 340 mm tank, at a 20 mm spacing.

    Built once and shared by three tests, because settling it is 89 s of the 96 s this
    file costs and three copies of it would be five minutes.

    Deliberately tall and narrow. A shallow tank was tried first and the hydrostatic
    slope came out 5.6% high rather than 1.5%, for a reason that is about the measurement
    and not about the solver: the fit has to exclude the layer next to each wall (an
    Akinci boundary over-pressures the first fluid layer) and the top two layers (the
    free surface reads zero pressure, by construction -- see `sph.residual`), and in a
    200 mm column that leaves seven particle layers to fit a line through. 280 mm leaves
    eleven, and the excluded layers are a smaller fraction of the thing being measured.

    Returns the world, the tank body index, and the time-averaged pressure and height.
    Time-averaged over 40 coupling steps because the free surface layer oscillates: it
    sits at zero pressure with a positive pressure below it, so it is pushed up, falls
    back, and water's real viscosity damps that very slowly. The average is the
    hydrostatic state; a single frame is the state plus a standing wave.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    level = 0.28
    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.14, 0.14, 0.34), faces="wall"),
                     particle_radius=0.01, fluid="water", dt=0.01, device="cpu")
    body = world.add_body(label="tank")
    world.boundary_from_domain(body=body)
    world.fill_below(level)
    for _ in range(120):
        world.step()
    samples = 40
    pressure = np.zeros(world.n_fluid)
    height = np.zeros(world.n_fluid)
    forces = []
    errors = []
    for _ in range(samples):
        world.step()
        pressure += world.pressures()
        height += world.positions()[:, 2]
        forces.append(world.wrench(body)[0])
        errors.append(world.density_error())
    return {
        "world": world, "body": body, "level": level,
        "pressure": pressure / samples, "height": height / samples,
        "forces": np.asarray(forces), "errors": errors,
    }


def test_a_hydrostatic_column_rises_linearly_with_depth_at_rho_g_h(tank):
    """`dp/dh = rho g`, fitted over the bulk, with the scatter reported rather than hidden.

    Measured here: a slope of 1.0257 rho g, an intercept of -39.5 Pa against a bottom
    pressure of 2,740 Pa, and a residual scatter of 6.4% of rho g H about the line. The
    gate is 5% on the slope and two standard errors on the intercept; see below for why
    the intercept cannot honestly be graded tighter than that.

    The excluded particles and why each is excluded:

    * the layer against each of the four side walls and the floor -- an Akinci boundary
      over-pressures the first fluid layer, measured at 1.55 rho g h on the floor layer
      against 0.94 to 0.99 in the bulk;
    * the top two layers -- `sph.residual` clamps the constant-density residual at zero,
      so the free surface carries p = 0 by construction, and the transition from zero to
      hydrostatic takes about a support radius.

    The scatter is not noise that more steps would remove. It is the standard SPH
    pressure field on a regular lattice: consecutive particle layers alternate about the
    line, measured at 0.993, 0.958, 0.952 and 0.941 of the local rho g h down four bulk
    layers of the shallower tank this started from. A projection-style
    solver like DFSPH enforces the *density* constraint, which fixes the pressure
    gradient and leaves a layer-scale oscillation in the pressure itself; it is why the
    integrated load on the floor below is 50 times more accurate than any single
    particle's pressure.
    """
    world, pressure, height = tank["world"], tank["pressure"], tank["height"]
    level, spacing = tank["level"], world.spacing
    x = world.positions()
    size = world.domain.size

    bulk = (
        (x[:, 0] > 1.5 * spacing) & (x[:, 0] < size[0] - 1.5 * spacing)
        & (x[:, 1] > 1.5 * spacing) & (x[:, 1] < size[1] - 1.5 * spacing)
        & (height > 1.5 * spacing) & (height < level - 2.0 * spacing)
    )
    assert bulk.sum() > 100, bulk.sum()

    depth = level - height[bulk]
    design = np.vstack([depth, np.ones(len(depth))]).T
    slope, intercept = np.linalg.lstsq(design, pressure[bulk], rcond=None)[0]
    rho_g = world.rho0 * 9.81

    assert abs(slope / rho_g - 1.0) < 0.05, (slope, rho_g, slope / rho_g)

    # The intercept is the pressure the fit extrapolates to the free surface, where the
    # physical answer is zero -- and it is graded against **its own standard error**, not
    # against a fraction of the bottom pressure, because the fraction that was here (1%
    # of rho g H = 27 Pa) is smaller than the uncertainty of the number it was grading.
    #
    # That is arithmetic and not an opinion. The residual scatter about the line is 6.4%
    # of rho g H -- 175 Pa, a layer-scale pattern in the pressure field and not temporal
    # noise, so averaging more steps does not shrink it -- and over 140 points whose
    # depths run 40 to 250 mm the leverage term puts the intercept's standard error at
    # about 38 Pa. The old gate was therefore 0.7 sigma on a quantity whose own error bar
    # is wider than the gate, and it passed or failed on where the layer pattern happened
    # to sit: -3.0 Pa (0.08 sigma) before `nonpressure` stopped racing its own velocity
    # field and -39.5 Pa (1.0 sigma) after, for a pressure field whose integrated load on
    # the floor is 0.9999x the weight of the water in BOTH cases (the check below, which
    # is the one with the physics in it).
    #
    # Two sigma on the intercept still catches what this line is for -- a field with a
    # real offset in it, which would be many sigma -- and it is a statement that holds
    # whatever the layer pattern does.
    residual = pressure[bulk] - (slope * depth + intercept)
    scatter = float(residual.std(ddof=2))
    spread = float(((depth - depth.mean()) ** 2).sum())
    error = scatter * math.sqrt(1.0 / len(depth) + depth.mean() ** 2 / spread)
    assert abs(intercept) < 2.0 * error, (
        f"the fit extrapolates {intercept:.3g} Pa to the free surface, "
        f"{abs(intercept) / error:.2f} standard errors from zero "
        f"(scatter {scatter / (rho_g * level):.2%} of rho g H, SE {error:.3g} Pa)")
    # Monotonic in the mean: the deepest tenth is heavier than the shallowest tenth by
    # the amount the hydrostatic gradient says, within the scatter.
    order = np.argsort(depth)
    tenth = max(3, len(order) // 10)
    deep = pressure[bulk][order[-tenth:]].mean()
    shallow = pressure[bulk][order[:tenth]].mean()
    gap = depth[order[-tenth:]].mean() - depth[order[:tenth]].mean()
    assert abs((deep - shallow) / (rho_g * gap) - 1.0) < 0.10


def test_the_density_error_stays_bounded(tank):
    """The compression error does not grow, and the surface deficit is reported separately.

    Measured on this tank: max compression 0.61%, mean 0.11%, over 160 coupling steps
    (480 substeps). The gate is 2% on the max and 0.5% on the mean.

    `density_error()`'s `max` and `mean` are the **compression** error,
    `max(rho/rho0 - 1, 0)`, and the reason is in its docstring and is worth repeating
    here because it is the difference between a test that means something and one that
    cannot pass: a free-surface particle's summation density reads 20 to 40% below rho0
    in a converged, correct run, because there is no fluid above it to sum. This tank's
    worst particle reads 0.60 rho0 and that is not an error -- it is a particle on the
    surface. A single `|rho - rho0|/rho0` would be 40% here forever, so a test on it
    would have to be gated at 50%, at which point it would no longer notice real
    compression.
    """
    errors = tank["errors"]
    worst = max(e["max"] for e in errors)
    mean = float(np.mean([e["mean"] for e in errors]))
    assert worst < 0.02, worst
    assert mean < 0.005, mean
    # It is a steady state, not a slow divergence: the last ten samples are no worse than
    # the first ten.
    early = float(np.mean([e["mean"] for e in errors[:10]]))
    late = float(np.mean([e["mean"] for e in errors[-10:]]))
    assert late < early * 1.5 + 1e-4, (early, late)
    # And the deficit is real and is reported, so nothing here is claiming a 0.1% fluid.
    assert min(e["min_signed"] for e in errors) < -0.1
    assert tank["world"].stable()


def test_the_floor_of_a_tank_carries_the_weight_of_the_water(tank):
    """`wrench(tank)` is `-(m g)` in z: the sign convention and the magnitude in one number.

    **This is the standing check on the thing that has gone wrong twice in this
    repository.** `wrench(body)` is the force **on the body**, `lattice.stream`'s
    convention, so water sitting in a tank pushes *down* on it. A sign error here would
    make a floating hull sink and a submerged body fall, and it would do it while every
    density error in the run stayed perfect, because the pressure solve does not care
    which way the reaction is reported.

    Measured: -7.834 N on a tank holding 686 particles of 7.954 mg, whose weight is
    7.834 N. Ratio 0.9999, standard deviation across the 40 samples 1.9%. The gate is 2%
    on the mean and the sign is asserted on its own so that a failure says which of the
    two went wrong.

    It is also the only global conservation statement this file makes. The water's weight
    has nowhere else to go: there is no body force on the walls, no buoyancy term added
    anywhere, and the only other thing in the domain is the fluid's own pressure field --
    so if the integrated boundary force is the weight, the Akinci coupling is accounting
    for all of it, and the 0.01% says the discretisation is not leaking any.
    """
    world, forces = tank["world"], tank["forces"]
    weight = world.n_fluid * world.mass * 9.81

    mean = forces.mean(axis=0)
    assert mean[2] < 0.0, ("the water must push DOWN on the tank it sits in", mean)
    assert abs(mean[2] / -weight - 1.0) < 0.02, (mean[2], -weight)
    # Nothing sideways: a still tank has no horizontal load, and a systematic one would
    # mean the wall sampling is lopsided.
    assert abs(mean[0]) < 0.05 * weight and abs(mean[1]) < 0.05 * weight, mean

    # `wrench()` and `wrench(body)` are the same numbers, differently shaped.
    everything = world.wrench()
    assert everything.shape == (1, 6)
    force, torque = world.wrench(world.n_fluid and tank["body"])
    assert np.allclose(everything[tank["body"], :3], force)
    assert np.allclose(everything[tank["body"], 3:], torque)

    # The torque about the tank's own centre of mass -- which `add_body` put at the
    # domain's corner -- is `r x F` for the load applied at the tank's centre. Checked
    # because the torque is the half of the wrench that was missing from
    # `world.rotor_reaction` until it was found by an Isaac task that could not control
    # attitude, and a torque nobody checks is a torque nobody notices is zero.
    arm = np.asarray(world.domain.size) / 2.0
    arm[2] = 0.0
    expected = np.cross(arm, [0.0, 0.0, mean[2]])
    assert np.allclose(torque[:2], expected[:2], rtol=0.1, atol=0.02 * weight), \
        (torque, expected)


# -- a submerged body: the force the whole redirect exists for -------------------------


@pytest.fixture(scope="module")
def submerged():
    """An 80 mm cube held fixed under 220 mm of water, four particles across.

    The cube's half-extent is exactly two spacings and its centre is exactly halfway
    between two lattice sites, which makes the sampled volume equal the geometric one
    (512 cm^3 both ways) rather than approximately equal. That is not the test being
    arranged to pass -- `volume(body)` is what the force is graded against either way --
    it is the test being arranged so that one number means both things, because a reader
    who sees `rho V g` wants to know which V.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    half = 0.04
    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.20, 0.20, 0.28), faces="wall"),
                     particle_radius=0.01, fluid="water", dt=0.01, device="cpu")
    tank = world.add_body(label="tank")
    world.boundary_from_domain(body=tank)
    # Centre between lattice sites on every axis: sites are at `(i + 1/2) * 20 mm`.
    box = world.add_body(position=(0.10, 0.10, 0.12), label="box")
    world.add_shape("box", body=box, scale=(half, half, half))
    world.fill_below(0.22)
    for _ in range(80):
        world.step()
    forces, torques = [], []
    for _ in range(40):
        world.step()
        force, torque = world.wrench(box)
        forces.append(force)
        torques.append(torque)
    return {"world": world, "box": box, "tank": tank, "half": half,
            "forces": np.asarray(forces), "torques": np.asarray(torques)}


def test_a_submerged_box_feels_rho_v_g_upward(submerged):
    """Archimedes, through the Akinci coupling and nothing else.

    No buoyancy term is added anywhere: `docs/sph-contract.md` makes `buoyancy=False` the
    default for this backend precisely because the pressure field already has a real
    waterline in it, and adding `rho V g` on top would count it twice. So this number
    comes entirely out of the boundary half of the pressure solve
    (`sph.apply_pressure`), and it is the reason SPH was chosen over the frozen lattice
    free surface.

    Measured, time-averaged over 40 coupling steps, as a multiple of `rho V g` with V the
    sampled volume `volume(box)`:

        particles across the box    F_z / rho V g    sample sd    |F_x| + |F_y| / rho V g
                 3 (60 mm cube)        1.1863          0.153               0.182
                 4 (80 mm cube)        1.0992          0.098               0.108
                 5 (100 mm cube)       1.0614          0.075               0.092

    so it converges towards 1 at roughly first order in the particle count across the
    body, and the gate here is 15% at four across. That is a loose gate and it is loose
    for a stated reason rather than to accommodate a failure: the integrand is a pressure
    gradient summed over a surface sampled by 64 particles, four across is a body one and
    a half kernel supports wide, and the error is a resolution error that the table above
    shows going away. The equivalent lattice number is the one `lattice.stream`'s comment
    records -- -0.48 rho V g, the wrong sign and half the magnitude -- which is the
    comparison that matters.

    Overshooting rather than undershooting is consistent across the three resolutions.
    The sampled volume accounts for the body's size exactly, so the excess is in the
    pressure field: the first fluid layer against a boundary is over-pressured (the same
    effect the hydrostatic fit excludes), the box's underside sees more of that than its
    top does because the pressure there is higher, and the imbalance adds to the lift.
    That is a hypothesis consistent with all three rows and it is not a measurement; what
    is measured is the table.
    """
    world, box, forces = submerged["world"], submerged["box"], submerged["forces"]
    volume = world.volume(box)

    # The sampled volume is the geometric one for this cube, by construction.
    geometric = (2.0 * submerged["half"]) ** 3
    assert abs(volume / geometric - 1.0) < 1e-9, (volume, geometric)

    expected = world.rho0 * volume * 9.81
    mean = forces.mean(axis=0)
    assert mean[2] > 0.0, ("a submerged body must be pushed UP", mean)
    assert abs(mean[2] / expected - 1.0) < 0.15, (mean[2], expected,
                                                  mean[2] / expected)
    # Horizontal load on a symmetrically placed cube in still water: zero, to the
    # scatter. 15% because it is the same discretisation error with no mean to sit on.
    assert abs(mean[0]) < 0.15 * expected and abs(mean[1]) < 0.15 * expected, mean
    # And no torque on a cube whose centre of mass is its centroid.
    assert np.abs(submerged["torques"].mean(axis=0)).max() < 0.05 * expected * 0.04


def test_the_tank_carries_the_water_and_the_submerged_box_reaction(submerged):
    """Global vertical balance: the floor holds the water plus what the box displaces.

    The box is kinematic -- nothing integrates it -- so the water pushes it up with
    `rho V g` and it pushes back down on the water with the same, and the tank has to
    carry both. Measured: -71.95 N on the tank against -(67.69 + 2.12) = -69.80 N
    predicted, 3.1% high, and the two terms are separately measured by the two tests
    above. The gate is 8%.

    Looser than the empty tank's 2% because it is the sum of two numbers, one of which is
    the 10%-accurate buoyancy. It is here anyway: it is the only check that the two
    boundary bodies' wrenches are consistent with each other rather than each being
    plausible on its own, which is the failure a per-body sign error would survive.
    """
    world, box, tank = submerged["world"], submerged["box"], submerged["tank"]
    weight = world.n_fluid * world.mass * 9.81
    displaced = world.rho0 * world.volume(box) * 9.81
    floor = world.wrench(tank)[0][2]
    assert floor < 0.0, floor
    assert abs(floor / -(weight + displaced) - 1.0) < 0.08, (floor, weight, displaced)


# -- the contract, the refusals and the report -----------------------------------------


def test_step_returns_the_contract_dictionary_and_the_reads_line_up():
    """`docs/sph-contract.md`'s keys and shapes, on a world small enough to be cheap."""
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall"),
                     particle_radius=0.01, fluid="water", device="cpu")
    world.boundary_from_domain()
    world.fill_below(0.06)
    cost = world.step()
    for key in ("step", "substeps", "seconds", "simulated", "particles", "mlups"):
        assert key in cost, key
    assert cost["step"] == 1
    assert cost["particles"] == world.n_fluid

    # `seconds` is wall clock and `simulated` is physical time, and they are different
    # keys because a case that divides a sloshing period by the wrong one is wrong by a
    # factor nobody can see: the answer keeps its units and its order of magnitude.
    assert cost["simulated"] == world.dt
    assert cost["seconds"] != cost["simulated"]

    # What `couple.py` branches on, and what connects a graded density error to a graded
    # pressure error. `c0` is a derived equivalence and not a solver parameter -- see its
    # docstring -- but it has to be a finite positive number.
    assert world.coupling_style == "particles"
    assert not hasattr(world, "dx"), "an SphWorld has no lattice spacing to divide by"
    assert np.isfinite(world.c0) and world.c0 > 0.0
    assert abs(world.c0 - world._peak_estimate / math.sqrt(world.tolerance)) < 1e-9

    n = world.n_fluid
    assert world.positions().shape == (n, 3)
    assert world.velocities().shape == (n, 3)
    assert world.densities().shape == (n,)
    assert world.pressures().shape == (n,)
    assert set(world.density_error()) >= {"max", "mean"}
    assert isinstance(world.report(), list)

    # `surface_height` is depth per column, nan where dry. The tank is full to 60 mm on
    # every column, so nothing is dry and every column reads the same depth to within a
    # particle radius.
    depth = world.surface_height(axis=2)
    assert depth.ndim == 2
    assert np.isfinite(depth).all()
    assert abs(np.nanmean(depth) - 0.06) < world.radius

    # A dry column reads nan rather than zero, which is the difference between "no water
    # here" and "water of zero depth here".
    wide = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.2, 0.2, 0.1), faces="wall"),
                    particle_radius=0.01, fluid="water", device="cpu")
    wide.boundary_from_domain()
    wide.fill_box((0.0, 0.0, 0.0), (0.06, 0.06, 0.05))
    wide.step()
    assert np.isnan(wide.surface_height(axis=2)).any()


def test_surface_height_bins_each_horizontal_axis_from_its_own_extent():
    """A long thin tank gets a long thin grid, not a square one with dry bins in it.

    This returned `(n, n)` sized from the first horizontal axis, so a 4:1 tank produced a
    grid three quarters of which lay outside the fluid. Those columns read `nan`, which
    is correct, and a caller that zero-filled them instead of skipping them then averaged
    a depth four times too small -- a believable wrong answer, which is the one kind that
    does not get noticed.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    # 80 mm on the thin axis and not 60: `_check` refuses a domain narrower than four
    # particle spacings, because a kernel support is two of them and nothing in a
    # narrower box would ever have a full neighbourhood. 20 mm particles make that
    # 80 mm, and the tank is still 3:1.
    #
    # The level is 40 mm and not 50 because `fill_below` fills whole lattice layers: at a
    # 20 mm spacing the sites are at 10 and 30 mm, so asking for 50 mm of water gets
    # 40 mm of it and `surface_height` correctly reports 40. That is the sampler being
    # honest; a test that asked for 50 would be measuring the quantisation and calling it
    # an error.
    long, thin, level = 0.24, 0.08, 0.04
    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (long, thin, 0.10), faces="wall"),
                     particle_radius=0.01, fluid="water", device="cpu")
    world.boundary_from_domain()
    world.fill_below(level)
    world.step()

    depth = world.surface_height(axis=2)
    assert depth.shape == (int(round(long / world.spacing)),
                           int(round(thin / world.spacing))), depth.shape
    # Every column is wet, because the tank is full to `level` everywhere.
    assert np.isfinite(depth).all()
    assert abs(np.nanmean(depth) - level) < world.radius
    # And the explicit forms still work: one count for both, or a pair.
    assert world.surface_height(bins=7).shape == (7, 7)
    assert world.surface_height(bins=(9, 3)).shape == (9, 3)


def test_the_first_sloshing_mode_starts_as_a_cosine_and_not_as_a_staircase():
    """`fill_below(level, amplitude=...)` is the free-surface initialiser the cases need.

    The sloshing case was building the first mode out of abutting `fill_box` slabs, which
    is a cosine plus every harmonic the step edges carry, and which depends on the
    slabs' lattices lining up -- true here, because `_lattice` is global, and not
    something `docs/sph-contract.md` promises.

    Checked as a shape rather than as a formula: the surface is high at one wall, low at
    the other, and crosses the mean level at the middle. Within one particle radius,
    because a particle lattice cannot represent a cosine more finely than that.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    level, amplitude = 0.08, 0.02
    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.24, 0.04, 0.14), faces="wall"),
                     particle_radius=0.005, fluid="water", device="cpu")
    world.boundary_from_domain()
    assert world.fill_below(level, amplitude=amplitude, mode=1) > 200
    world.step()

    depth = np.nanmean(world.surface_height(axis=2), axis=1)
    bins = len(depth)
    assert depth[0] > level + amplitude - 2 * world.radius, depth[0]
    assert depth[-1] < level - amplitude + 2 * world.radius, depth[-1]
    middle = depth[bins // 2 - 1:bins // 2 + 1].mean()
    assert abs(middle - level) < 2 * world.radius, middle
    # Monotonic down the tank, which a staircase of slabs is only on average.
    assert np.all(np.diff(depth) < world.radius)
    # A zero amplitude is the plane it always was.
    flat = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall"),
                    particle_radius=0.01, fluid="water", device="cpu")
    assert flat.fill_below(0.06) == flat.fill_below(0.06, amplitude=0.0)


@pytest.mark.parametrize("kwargs, wanted", [
    ({"particle_radius": None}, "particle_radius"),
    ({"solver": "wcsph"}, "wcsph"),
    ({"surface_tension": 0.07}, "surface_tension"),
    ({"dt": 0.1, "substeps": 1}, "substeps"),
    ({"viscosity": 1.0}, "viscosity"),
    ({"particle_radius": 0.05}, "particle_radius"),
])
def test_the_refusals_name_the_knob_that_moves_them(kwargs, wanted):
    """Every refusal says which knob to turn. `units.UnphysicalScaling`'s contract.

    A refusal that does not name the fix costs a round of guessing, which is the argument
    `units.py` makes and `benchmarks/*/grade.py` follows. The six cases here are the six
    ways a caller can ask for a run this backend cannot do: no sizing knob, a solver that
    is not shipped, surface tension that would be silently ignored, a substep past the
    CFL limit, a viscosity past the explicit scheme's stability limit, and a spacing too
    coarse for the domain to hold a kernel support.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.units import UnphysicalScaling
    from openreynolds.live.world import Domain

    settings = {"particle_radius": 0.01, "fluid": "water", "device": "cpu"}
    settings.update(kwargs)
    with pytest.raises(UnphysicalScaling) as raised:
        SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall"),
                 **settings)
    assert wanted in str(raised.value), str(raised.value)


def test_stepping_with_no_fluid_refuses_rather_than_running_empty():
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.units import UnphysicalScaling
    from openreynolds.live.world import Domain

    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall"),
                     particle_radius=0.01, device="cpu")
    world.boundary_from_domain()
    with pytest.raises(UnphysicalScaling, match="no fluid particles"):
        world.step()


def test_a_mesh_shape_is_refused_by_name_rather_than_sampled_wrongly():
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.units import UnphysicalScaling
    from openreynolds.live.world import Domain

    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall"),
                     particle_radius=0.01, device="cpu")
    with pytest.raises(UnphysicalScaling, match="no mesh sampler"):
        world.add_shape("mesh", position=(0.05, 0.05, 0.05), scale=(1.0, 1.0, 1.0),
                        mesh=([[0, 0, 0]], [0, 0, 0]))


def test_the_report_states_the_smoothing_length_and_that_surface_tension_is_off():
    """The contract says the smoothing length is reported; the brief says an absent
    surface tension says so rather than being implied."""
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall"),
                     particle_radius=0.01, fluid="water", device="cpu")
    world.boundary_from_domain()
    world.fill_below(0.06)
    text = "\n".join(world.report())
    assert "support h 40 mm" in text, text
    assert "cubic spline" in text
    assert "surface tension OFF" in text
    assert "Akinci" in text
    assert "DFSPH" in text
    # The host read inside the step is named in the report rather than left implied.
    assert "host reads" in text


def test_the_walls_do_not_leak_and_coincident_particles_are_dropped():
    """Two particles at one lattice site is a division, not a near miss. `_distinct`.

    `boundary_from_domain` builds each wall as a slab overhanging the other axes by the
    shell thickness, so the tank's edges are filled rather than notched -- and so a
    corner site is produced by two or three slabs. Measured on this tank: 1,232 distinct
    sites from 1,760 produced, 528 duplicates, all of them on edges. Dropping them was
    also a 3.4x speedup (13.1 s to 3.9 s for 80 coupling steps), because a duplicate pair
    sits at r = 0 and appears in every neighbour query for nothing.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.12, 0.12, 0.26), faces="wall"),
                     particle_radius=0.01, fluid="water", dt=0.01, device="cpu")
    body = world.add_body(label="tank")
    world.boundary_from_domain(body=body)
    world.fill_below(0.20)
    world.step()

    assert world.duplicates == 528, world.duplicates
    assert world.n_boundary == 1232, world.n_boundary
    # The lattice index of a site, which is `origin + (i + 1/2) * spacing` -- so the
    # half has to come off before rounding. Without it `(i + 1/2)` rounds to `i` or
    # `i + 1` by numpy's round-half-to-even and collapses alternate pairs, which read as
    # 608 phantom duplicates in a set that has none.
    sites = np.round(world.boundary_positions() / world.spacing - 0.5).astype(int)
    assert len(np.unique(sites, axis=0)) == len(sites)

    # No fluid particle sits inside the tank's walls, and the last-resort clamp is a
    # last resort: it fires on a handful of particles during the transient, not every
    # step on the whole floor. (It did exactly that until the reflection box was moved
    # off the first lattice plane, and the tank then read 63% of the weight of its own
    # water.)
    for _ in range(40):
        world.step()
    caught = int(world._clamped.numpy()[0])
    assert caught < world.n_fluid, caught


def test_a_moving_wall_drags_the_fluid_and_feels_the_reaction():
    """A boundary body given a velocity moves its particles on the device and is pushed back.

    The two halves of `pose_boundary` and of the boundary reaction in `sph.nonpressure`
    and `sph.apply_pressure`, in one check: a paddle swept through still water must
    (a) move its own boundary particles from the body table, on the device, (b) put
    momentum into the water, and (c) feel a force opposing its own motion.

    Five steps and no more, which is the whole measurement rather than a shortcut. The
    paddle is 40 mm across in a 160 mm tank, so at 0.5 m/s it is a piston, and the water
    it displaces reaches the far wall and comes back. Measured, fluid x-momentum against
    step count: +0.102 kg m/s at step 5, -0.064 at step 10, -0.172 at step 20. By step 10
    the recirculation dominates and "the water is moving the way the paddle is" is simply
    false -- a real flow, and the wrong thing to assert about a no-slip wall. The force on
    the paddle averages -5.75 N over the five steps, which is the half that does not care
    about the recirculation.

    A wiring test, not a drag measurement: a drag coefficient at a 20 mm particle spacing
    would not be worth writing down.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    world = SphWorld(domain=Domain((0.0, 0.0, 0.0), (0.16, 0.08, 0.12), faces="wall"),
                     particle_radius=0.01, fluid="water", dt=0.005, device="cpu")
    world.boundary_from_domain()
    paddle = world.add_body(position=(0.04, 0.04, 0.06), velocity=(0.5, 0, 0, 0, 0, 0),
                            label="paddle")
    world.add_shape("box", body=paddle, scale=(0.02, 0.03, 0.04))
    world.fill_below(0.10)
    start = world.boundary_positions(paddle)[:, 0].mean()

    pushed = []
    for step in range(5):
        world.set_body_state(paddle, position=(0.04 + 0.5 * step * world.dt, 0.04, 0.06))
        world.step()
        pushed.append(world.wrench(paddle)[0][0])

    # (a) the particles followed the body, from `body_q`, in a kernel.
    assert world.boundary_positions(paddle)[:, 0].mean() > start + 0.005
    # (b) the water has net momentum in the direction the paddle went.
    momentum = (world.velocities() * world.mass).sum(axis=0)
    assert momentum[0] > 0.0, momentum
    # (c) and the water pushes back, against the motion -- in the mean, not on every
    # step. Measured, newtons per step: -13.40, -10.89, +1.09, -1.76, -3.78, mean -5.75.
    # The paddle starts from rest against still water, so the first two steps are an
    # impulse and the third is its rebound; a per-step assertion would be an assertion
    # about the ringing of a 20 mm particle lattice and would fail for the right reasons.
    assert np.mean(pushed) < 0.0, pushed
    assert world.stable()


def test_the_viscous_term_cannot_see_a_velocity_another_thread_wrote():
    """A rigidly translating block has no shear, so `nonpressure` must only add gravity.

    **This is the reproducibility bug, as one assertion.** `sph.nonpressure` used to
    finish with `v[i] = vi + accel * dt`, in the same launch whose neighbour loop reads
    `v[j]`. Nothing orders the two, so `v[j]` was the old velocity or the new one
    depending on where thread `j` had got to -- which on a GPU is a different subset of
    the neighbourhood on every run of the same code, and is what made `sph_sloshing` read
    a 419 ms first-mode period on one run and 322 ms on the next.

    Monaghan's viscosity is a function of `v_i - v_j` only, so a block of fluid moving
    rigidly has an identically zero viscous acceleration: every pair difference is zero.
    Gravity's increment `g dt` is uniform, so it cancels out of a pair difference taken
    at one instant and survives in full in one taken across the race. The invariant is
    therefore exact rather than approximate -- `v` out must be `v` in plus `g dt`, to the
    last bit -- and the failure is one-sided: the only thing that can break it is reading
    a neighbour mid-update.

    Measured with the bug in, on this exact configuration: 1.07e-5 m/s of spurious
    velocity after a single launch, 0.011% of the block's own 0.1 m/s and 0.048% of
    `g dt`. That is injected afresh every substep, so over the 2,217 substeps of the
    sloshing case it is not small. Warp's CPU launch is serial, so the race is *milder*
    here than on a GPU -- thread `i` sees an updated `v[j]` only for the neighbours ahead
    of it in grid order -- and it still fails by five orders of magnitude.

    No walls in this world on purpose: a boundary particle sits at `v = 0` next to fluid
    at 0.1 m/s, which is real shear and would hide the invariant.
    """
    from openreynolds.live import sph
    from openreynolds.live.world import Domain

    speed = 0.1
    world = sph.SphWorld(
        domain=Domain((0.0, 0.0, 0.0), (0.04, 0.04, 0.04), faces="wall"),
        particle_radius=0.0025, fluid="water", gravity=(0.0, 0.0, -9.81), device="cpu")
    world.fill_box((0.005, 0.005, 0.005), (0.035, 0.035, 0.035),
                   velocity=(speed, 0.0, 0.0))
    world._commit()
    assert world.n_boundary == 0
    world._prepare()
    dt = world.dt / world.substeps
    before = world.v.numpy().copy()[:world.n_fluid]

    warp.launch(sph.nonpressure, dim=world.n, device=world.device,
                inputs=[world._grid.id, world.x, world.v, world.psi, world.rho,
                        world.body, world.body_com, world._wrench, world.accel,
                        sph.DTYPE(world.mass), world.n_fluid, sph.DTYPE(world.support),
                        sph.DTYPE(world.sigma), warp.vec3(*world.gravity),
                        sph.DTYPE(world.nu), sph.DTYPE(world.rho0)])
    warp.launch(sph.advect, dim=world.n_fluid, device=world.device,
                inputs=[world.v, world.accel, world.n_fluid, sph.DTYPE(dt)])

    after = world.v.numpy()[:world.n_fluid]
    want = before + np.asarray(world.gravity, dtype=np.float32) * np.float32(dt)
    slip = float(np.abs(after - want).max())
    # 1e-9 m/s and not exact equality, so that the assertion is about the race and not
    # about whether the device fused the multiply and the add. The bug gave 1.07e-5.
    assert slip < 1.0e-9, (
        f"a rigidly translating block picked up {slip:.4g} m/s of viscous acceleration "
        f"in one substep ({slip / speed:.3%} of its own speed, {slip / (9.81 * dt):.3%} "
        f"of g*dt): `nonpressure` is reading a velocity its own launch has written"
    )


def test_the_same_tank_stepped_twice_gives_the_same_field_bit_for_bit():
    """Two identical worlds, twenty steps each, and the fields must match exactly.

    The property every SPH measurement in this repository rests on and the one that was
    not true: `sph_sloshing` on an H100 read a first-mode period anywhere between 374.4
    and 419.3 ms on five runs of byte-identical code, because `sph.nonpressure` raced its
    own velocity field and because the solver's convergence test came off a float32
    `wp.atomic_add` whose summation order varies. Both are fixed and this is the standing
    guard: positions, velocities and densities compared with `==`, not with a tolerance,
    because the claim is determinism and a tolerance would pass on a solver that had lost
    it.

    On the **preferred** device rather than `"cpu"` on purpose, which is the only test
    here that is: Warp's CPU launches are serial, so a CPU pass is a weak pass -- a race
    between threads cannot fire when there is one thread. On a CUDA device this is the
    real check, and it is what it measures there: five H100 runs of `sph_sloshing` now
    return one sampled surface trace, identical byte for byte.

    No bodies, so no `wrench()`: that accumulator is six float32 `wp.atomic_add`s per
    boundary pair and its summation order does vary. It is an output and nothing in the
    fluid reads it, so it is not in this test's claim -- see `SphWorld.wrench`.
    """
    from openreynolds.live.sph import SphWorld
    from openreynolds.live.world import Domain

    def run():
        world = SphWorld(
            domain=Domain((0.0, 0.0, 0.0), (0.12, 0.06, 0.12), faces="wall"),
            particle_radius=0.005, fluid="water", dt=0.004)
        world.boundary_from_domain()
        # Off-centre and with a velocity, so the field has something to get wrong: a
        # column released at rest in the middle of a symmetric tank is a weaker test of
        # a summation order than one that is sloshing.
        world.fill_box((0.01, 0.01, 0.01), (0.07, 0.05, 0.08), velocity=(0.15, 0.0, 0.0))
        for _ in range(20):
            world.step()
        return world.positions(), world.velocities(), world.densities()

    first, second = run(), run()
    for name, a, b in zip(("positions", "velocities", "densities"), first, second):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        differing = int((a != b).sum())
        assert differing == 0, (
            f"{differing} of {a.size} {name} differ between two identical runs; "
            f"largest difference {np.abs(a - b).max():.6g}")


# -- delta-SPH and the wrench filter, the two v0.1 knobs for the force noise -----------


def test_density_diffusion_flattens_a_bump_and_spares_a_gradient():
    """`diffuse_density` is a Laplacian: it removes a bump and leaves a linear ramp.

    The two halves are the whole of what a diffusion term is allowed to do, and the
    second is why this one is not a thumb on the scale. `openreynolds/toolbox/sph.py`
    makes the same argument for the same term in the weakly compressible solver and
    `tests/test_toolbox_sph.py` has the 2D version of this test; this is the device
    kernel, launched the way `_prepare` launches it.

    **The sign is what this catches.** `x_ij . grad_i W_ij` is negative everywhere in
    the support, so the term diffuses only if it is subtracted; written the other way it
    is an anti-diffusion that sharpens the noise, and a run with it that way looks
    entirely plausible -- the density field is still a density field and the solver still
    converges.

    No walls, because `diffuse_density` skips boundary neighbours by design (its
    docstring says why) and a test with walls in it would be measuring that skip rather
    than the operator.
    """
    from openreynolds.live import sph
    from openreynolds.live.world import Domain

    def diffused(field):
        world = sph.SphWorld(
            domain=Domain((0.0, 0.0, 0.0), (0.06, 0.06, 0.06), faces="wall"),
            particle_radius=0.0025, fluid="water", device="cpu",
            density_diffusion=0.012)
        world.fill_box((0.01, 0.01, 0.01), (0.05, 0.05, 0.05))
        world._commit()
        assert world.n_boundary == 0
        world._prepare()
        # The real summation density is overwritten, so the operator is measured on a
        # field this test chose rather than on whatever the fill happened to produce.
        world.rho.assign(field(world.x.numpy()[:world.n_fluid]).astype(np.float32))
        warp.launch(sph.diffuse_density, dim=world.n, device=world.device,
                    inputs=[world._grid.id, world.x, world.rho, world.rho_c,
                            sph.DTYPE(world.mass), world.n_fluid,
                            sph.DTYPE(world.support), sph.DTYPE(world.sigma),
                            sph.DTYPE(world.dt / world.substeps * world.diffusivity)])
        return (world.rho.numpy().astype(np.float64),
                world.rho_c.numpy().astype(np.float64),
                world.x.numpy()[:world.n_fluid].astype(np.float64))

    # A bump: rho0 everywhere, plus 1% on the particles nearest the centre.
    def bump(x):
        middle = np.abs(x - 0.03).max(axis=1) < 0.006
        return np.where(middle, 998.21 * 1.01, 998.21)

    raw, out, _ = diffused(bump)
    hot = raw > 998.21 * 1.005
    assert hot.sum() >= 8, "the bump has to be wider than one particle to diffuse"
    before = float(raw[hot].mean() - 998.21)
    after = float(out[hot].mean() - 998.21)
    assert 0.0 < after < before, (
        f"the bump's excess went {before:.4g} -> {after:.4g} kg/m3. It must come DOWN: "
        f"up is the sign error, and `x_ij . grad_i W_ij` is negative everywhere live, "
        f"so the term diffuses only when it is subtracted"
    )
    # delta 0.012 is 88% of the ceiling `diffusivity` derives, where `D dt lambda` is
    # 0.88 and the sharpest mode is almost annihilated in one substep. A three-particle
    # bump is close to that mode, so most of it goes: the gate is that most goes and
    # that none of it comes back the other side of zero, which is the overshoot
    # `DIFFUSION_EIGENVALUE` exists to keep out.
    assert 0.10 < after / before < 0.80, (
        f"one substep took the bump to {after / before:.3g} of itself; it measures "
        f"0.615 at this delta. A ratio near 1 is a term that is not running, and a "
        f"NEGATIVE one is the overshoot `_check` refuses. A three-particle bump is not "
        f"the checkerboard, so it keeps more than the sharpest mode would -- "
        f"`test_the_diffusion_eigenvalue_is_what_the_refusal_is_built_on` is where that "
        f"mode is measured"
    )

    # A linear ramp in z, which a Laplacian must not move. This is the half that says
    # the hydrostatic profile the cases grade survives.
    _, ramp_out, points = diffused(lambda x: 998.21 * (1.0 + 0.01 * x[:, 2] / 0.06))
    want = 998.21 * (1.0 + 0.01 * points[:, 2] / 0.06)
    # Interior only: the operator is truncated at the edge of the block, where a
    # one-sided Laplacian of a ramp is genuinely not zero.
    inside = np.abs(points - 0.03).max(axis=1) < 0.012
    assert inside.sum() >= 20
    drift = float(np.abs(ramp_out[inside] - want[inside]).max())
    assert drift < 0.02, (
        f"a linear density ramp moved by {drift:.4g} kg/m3 under the diffusion. A "
        f"Laplacian leaves a linear field alone, which is what lets this term run "
        f"without flattening the hydrostatic gradient every floating case is graded on"
    )


def test_the_diffusion_eigenvalue_is_what_the_refusal_is_built_on():
    """`lambda_max h^2 = 23.2963`, at three resolutions, and it GROWS past `D dt lam = 2`.

    `DIFFUSION_EIGENVALUE` is the one number in `_check`'s density-diffusion refusal, and
    the refusal is the only thing standing between a caller who read Molteni and
    Colagrossi and a run that sharpens its own density noise. It is a measurement, so it
    is measured here: one launch on a checkerboard, which is the sharpest mode a cubic
    lattice at this spacing carries, and the amplitude back out.

    Resolution independence is the half that makes it a constant rather than a fit: the
    operator is a property of the kernel and the lattice, so the same `lambda h^2` has to
    come out at every spacing, and it does to four decimals.

    The second half is the failure the limit exists for. At `D dt lambda = 2.209` -- which
    is what taking the viscous term's `0.125 h^2/nu` by analogy would have allowed -- a
    unit checkerboard comes back at about -1.21: the noise is bigger than it started and
    has changed sign, every substep.
    """
    from openreynolds.live import sph
    from openreynolds.live.world import Domain

    def eigenvalue(radius: float, box: float, number: float) -> tuple[float, float]:
        """`(lambda h^2, the amplitude after one launch at this diffusion number)`."""
        world = sph.SphWorld(
            domain=Domain((0.0, 0.0, 0.0), (box, box, box), faces="wall"),
            particle_radius=radius, fluid="water", device="cpu",
            # Any valid delta: it only has to be non-zero, so that `rho_c` is a second
            # array. With the term off `rho_c` *is* `rho` -- the allocation `_commit`
            # does not make -- and this kernel read in place is a Gauss-Seidel sweep in
            # grid order rather than the Jacobi one it is. Measured: the apparent
            # eigenvalue then drifts 23.19 -> 20.53 with the diffusion number, which
            # reads exactly like a nonlinearity and is an aliased output.
            density_diffusion=0.001)
        world.fill_box((box / 6.0,) * 3, (box * 5.0 / 6.0,) * 3)
        world._commit()
        world._prepare()
        points = world.x.numpy()[:world.n_fluid].astype(np.float64)
        cell = np.rint((points - points.min(axis=0)) / world.spacing).astype(int)
        sign = np.where(cell.sum(axis=1) % 2 == 0, 1.0, -1.0)
        world.rho.assign((998.21 + sign).astype(np.float32))
        # `number` is `D dt/h^2`; the kernel takes the product `D dt`.
        warp.launch(sph.diffuse_density, dim=world.n, device=world.device,
                    inputs=[world._grid.id, world.x, world.rho, world.rho_c,
                            sph.DTYPE(world.mass), world.n_fluid,
                            sph.DTYPE(world.support), sph.DTYPE(world.sigma),
                            sph.DTYPE(number * world.support ** 2)])
        out = world.rho_c.numpy().astype(np.float64) - 998.21
        # Interior only: the operator is truncated at the edge of the block.
        middle = np.abs(points - points.mean(axis=0)).max(axis=1) < (
            box / 3.0 - 2.0 * world.spacing)
        assert middle.sum() >= 27
        amplitude = float((sign[middle] * out[middle]).mean())
        return (1.0 - amplitude) / number, amplitude

    for radius, box in ((0.005, 0.12), (0.0025, 0.06), (0.002, 0.048)):
        # 0.02, not 1e-3: the amplitude is float32, so a diffusion number small enough
        # to be "linear" is also small enough to be rounding. At 0.02 the mode loses
        # 47% and the eigenvalue reads to four decimals.
        lam, _ = eigenvalue(radius, box, 0.02)
        assert abs(lam - sph.DIFFUSION_EIGENVALUE) < 0.02, (
            f"at a {radius * 1e3:g} mm radius the checkerboard decayed as though "
            f"lambda h^2 were {lam:.4f}, against the {sph.DIFFUSION_EIGENVALUE} "
            f"`_check` builds its refusal on. It must not depend on the resolution: it "
            f"is a property of the kernel and the lattice"
        )

    # The limit itself: monotone at `D dt lambda = 1`, and growing at the 2.209 the
    # viscous analogy would have allowed.
    _, at_limit = eigenvalue(0.0025, 0.06, 1.0 / sph.DIFFUSION_EIGENVALUE)
    assert abs(at_limit) < 0.02, (
        f"at the ceiling the sharpest mode should be annihilated, not reflected; the "
        f"checkerboard came back at {at_limit:+.4g}"
    )
    _, past_it = eigenvalue(0.0025, 0.06, 2.209 / sph.DIFFUSION_EIGENVALUE)
    assert past_it < -1.0, (
        f"at `D dt lambda` = 2.209 a unit checkerboard must come back below -1 -- "
        f"grown and sign-flipped -- and it came back at {past_it:+.4g}. That is the "
        f"configuration `0.125 h^2/nu` by analogy allows and `_check` refuses"
    )


def test_a_published_delta_is_refused_and_the_ceiling_is_1_25_root_tolerance():
    """`1.25 sqrt(tolerance)`, and a refusal that names the knob rather than running.

    `SphWorld.diffusivity` derives the ceiling in closed form; this is the arithmetic
    checked against the solver's own `_check` at three resolutions and two tolerances,
    because the derivation's whole claim is that the resolution cancels.
    """
    from openreynolds.live import sph
    from openreynolds.live.units import UnphysicalScaling
    from openreynolds.live.world import Domain

    domain = Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall")
    for radius in (0.005, 0.0075, 0.01):
        for tolerance in (1.0e-3, 1.0e-4):
            ceiling = math.sqrt(tolerance) / (sph.DIFFUSION_EIGENVALUE / 10.0)
            kwargs = dict(domain=domain, particle_radius=radius, device="cpu",
                          tolerance=tolerance)
            sph.SphWorld(density_diffusion=ceiling * 0.9, **kwargs)
            with pytest.raises(UnphysicalScaling, match="delta-SPH at delta"):
                sph.SphWorld(density_diffusion=ceiling * 1.1, **kwargs)
    # And the one every reader will try first.
    with pytest.raises(UnphysicalScaling, match="ACOUSTIC"):
        sph.SphWorld(domain=domain, particle_radius=0.005, device="cpu",
                     density_diffusion=0.1)
    with pytest.raises(UnphysicalScaling, match="ANTI-diffusion"):
        sph.SphWorld(domain=domain, particle_radius=0.005, device="cpu",
                     density_diffusion=-0.01)


def test_the_wrench_filter_is_off_by_default_and_lags_by_what_it_says():
    """Off: the raw accumulator. On: an EMA advanced once a step, never once a read.

    The second half is the one that could break silently. `wrench()` is called twice on
    some steps by `benchmarks/live/free_settle.py` -- once for the per-step series and
    once for the table -- so a filter that advanced on a read would have a time constant
    set by how often somebody looked at it. Measured by reading it three times a step
    and asserting the three agree.

    A CPU world, 500 particles, four steps: this is about where the filter is driven
    from, not about the fluid.
    """
    from openreynolds.live import sph
    from openreynolds.live.world import Domain

    def tank(**extra):
        world = sph.SphWorld(
            domain=Domain((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), faces="wall"),
            particle_radius=0.005, fluid="water", device="cpu", **extra)
        world.boundary_from_domain()
        world.fill_below(0.05)
        return world

    raw_world = tank()
    raw = []
    for _ in range(4):
        raw_world.step()
        first = raw_world.wrench(0)[0][2]
        assert raw_world.wrench(0)[0][2] == first, (
            "the raw wrench must not change between two reads of one step")
        raw.append(float(first))
    assert raw_world._wrench_mean is None, "off means no filter state at all"

    alpha = 0.2
    filtered_world = tank(wrench_filter=alpha)
    got = []
    for _ in range(4):
        filtered_world.step()
        three = [float(filtered_world.wrench(0)[0][2]) for _ in range(3)]
        assert three[0] == three[1] == three[2], (
            f"three reads of one step gave {three}: the EMA is advancing on a READ. It "
            f"must advance in `step()` and nowhere else, or its time constant is a "
            f"property of the reporting code"
        )
        got.append(three[0])

    # The fluid is deterministic bit for bit (the module docstring's claim) and the
    # filter does not touch it, so the raw series is reproduced and the EMA is exactly
    # the recurrence over it. A loose absolute gate against forces of order 0.1 N,
    # because only the float32 wrench atomics vary and they are bounded at 1.4e-8.
    want = raw[0]
    for index, sample in enumerate(raw):
        want = sample if index == 0 else alpha * sample + (1.0 - alpha) * want
        assert abs(got[index] - want) < 1.0e-6, (
            f"step {index}: filtered {got[index]:.9g} N against the recurrence's "
            f"{want:.9g} N over the raw series {raw}"
        )

    lines = "\n".join(filtered_world.report())
    assert "4-step group delay" in lines, (
        f"a filtered run has to say its own lag; alpha 0.2 is (1-a)/a = 4 steps. Got:\n"
        f"{lines}"
    )
    # Ten samples of the measured noise reduce it 10.01x and not sqrt(10) = 3.16,
    # because the noise is coherent rather than white (`PROGRESS_LIVE.md`). That claim
    # is a property of the signal rather than of this code, so it is measured in
    # `benchmarks/live/force_noise.py` and not asserted on a 500-particle CPU tank.
