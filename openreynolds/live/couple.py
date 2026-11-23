"""The Newton side: read the bodies, hand back the wrench, step both in lockstep.

The only file in this extension that knows Newton's field names. Everything it uses was
read out of the installed package rather than guessed, and the ones that are easy to get
wrong are recorded here with what they actually are, because three of them are not what
the documentation beside them says.

## What was read, and where

`newton._src.sim.state.State`
    `body_q`   `wp.array(dtype=wp.transform)`, the body frame, position in metres.
    `body_qd`  `wp.array(dtype=wp.spatial_vector)`. Its own docstring: "First three
               entries: linear velocity [m/s] relative to the body's center of mass in
               world frame; last three: angular velocity [rad/s] in world frame."
    `body_f`   `wp.array(dtype=wp.spatial_vector)`, and its docstring is the whole
               contract this file depends on: "Rigid body forces (spatial) [N, N.m] ...
               First three entries: linear force [N] in world frame applied at the
               body's center of mass (COM). Last three: torque (moment) [N.m] in world
               frame ... body_f represents an external wrench in world frame with the
               body's center of mass (COM) as reference point."
    `clear_forces()` zeroes `body_f` and `particle_f`.

`newton.Model`
    `body_com`          `wp.array(dtype=wp.vec3)`, the centre of mass **in body-local
                        coordinates**, so the world one is
                        `transform_point(body_q[b], body_com[b])`.
    `body_mass`         `wp.array(dtype=wp.float32)`. Note: a mass passed to
                        `add_body(mass=...)` is overwritten at `finalize()` by the one
                        the shape densities imply.
    `shape_type`        `wp.array(dtype=wp.int32)`, values from `newton.GeoType`.
    `shape_body`        `wp.array(dtype=wp.int32)`, `-1` for a static shape.
    `shape_transform`   `wp.array(dtype=wp.transform)`, the shape in its body's frame.
    `shape_scale`       `wp.array(dtype=wp.vec3)`, per-type (see `_SCALE` below).
    `shape_source`      a plain Python **list**, not a `wp.array`: unusable in a kernel.
    `shape_source_ptr`  `wp.array(dtype=wp.uint64)`, the same meshes as ids a kernel can
                        use. This is the one to pass to `voxelize`.
    `body_label` / `shape_label`  plain `list[str]`. There is no `body_key`.

`newton.solvers.SolverMuJoCo` -- the MuJoCo Warp solver the brief asks for.
    `step(state_in, state_out, control, contacts, dt, *, observables=None)`. Five
    required positionals, so `contacts=None` is passed explicitly and not omitted. It
    runs on the Warp CPU device here, with `mujoco_warp` installed; Newton warns that it
    wants mujoco ~=3.17 and works with 4.0.1.

The loop is Newton's own, from `newton/examples/basic/example_basic_dzhanibekov.py`:
`state.clear_forces()`, then every force source adds into `body_f`, then
`solver.step(state_0, state_1, control, contacts, dt)`, then swap the states.

## Three conventions that are documented wrongly in Newton and were measured instead

`shape_scale` is not `(r, r, r)` for a sphere and not `(r, h, r)` for a capsule, which
is what the comments at `newton/_src/sim/builder.py` lines 15385 and 15392 say. The
writers at lines 9188 and 9372 store zeros in the unused components, the decode in
`newton/_src/geometry/support_function.py` reads only the components it needs, and a
model built with one of each and printed gives:

    SPHERE     (radius, 0, 0)
    BOX        (half_x, half_y, half_z)
    CAPSULE    (radius, half_height, 0)          about the local z axis
    CYLINDER   (radius, half_height, barrel_r)   about the local z axis
    CONE       (radius, half_height, 0)           apex towards +z
    ELLIPSOID  (semi_x, semi_y, semi_z)
    PLANE      (half_width_x, half_width_y, 0)    0 meaning infinite
    MESH       (scale_x, scale_y, scale_z)

`_SCALE` below is that table, and `tests/test_live_couple.py` asserts it against a model
Newton actually builds, so if a future release changes the packing the test says so
rather than the sphere quietly becoming a point.

## The same, re-read on newton 1.6.1, because Isaac Lab pins it

Isaac Lab 3.0.0's metadata is `warp-lang==1.17.0` and `newton[sim]==1.6.1`, so a user
running this inside Isaac Lab gets 1.6.1 whatever `pyproject.toml` asks for. Everything
above was therefore read again out of 1.6.1, by building the same one-of-each model and
printing it. `docs/live-version-matrix.md` has the commands and the output. Nothing
moved, and the parts worth naming because they are the ones that would have been silent:

* **`shape_scale`'s packing is identical on 1.6.1 and 1.8.0**, component for component,
  for sphere, box, capsule, cylinder, plane and mesh. The sphere is still
  `(radius, 0, 0)` and not `(r, r, r)`. This is the one that had to be checked rather
  than trusted: `_SCALE` is the identity, so a changed packing would not raise, it would
  put the radius in the wrong slot and the body would voxelise as a point or a slab.
* every field the step loop reads exists on 1.6.1 with the same dtype:
  `Model.shape_source_ptr` (`uint64`), `shape_type`, `shape_body`, `shape_transform`,
  `shape_scale`, `body_com`, `body_mass`, `gravity`, the `shape_label` / `body_label`
  lists, and `State.body_q` / `body_qd` / `body_f` / `clear_forces()`.
* `newton.GeoType`'s member **names and values** are unchanged across the two, all
  twelve of them. `GEO_TO_SHAPE` maps by name on purpose and so would have survived a
  renumbering; it did not have to.
* `newton.solvers.SolverMuJoCo` exists on both, and so do `SolverXPBD`,
  `SolverSemiImplicit`, `SolverFeatherstone`, `SolverVBD`, `SolverStyle3D`,
  `SolverKamino` and `SolverImplicitMPM`. `SolverSteel` is new in 1.8.0 and is not used
  here.
* **`solver.step()` gained a keyword-only `observables=None` in 1.8.0.** On 1.6.1 the
  signature is the five positionals and nothing else. `step()` below passes exactly
  those five positionally, which is why it works on both -- and why `contacts=None` is
  still passed explicitly rather than omitted.
* a three-step `NewtonCoupling` run over a voxelised sphere produces the same 552 solid
  nodes, the same body pose and the same wrench to the last float on both.

The one thing that is *not* the same is packaging rather than API: `newton[sim]` asks
for `mujoco-warp ~=3.12.0` at 1.6.1 and `~=3.17.0` at 1.8.0, so the MuJoCo backend's own
version band moves with Newton. Nothing in this file imports mujoco -- the solver is the
caller's object -- so that band belongs to whoever chooses `SolverMuJoCo`, and
`pyproject.toml`'s `live` extra says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import lattice as L
from .world import FluidWorld, LiveUnavailable

__all__ = ["NewtonCoupling", "newton_shapes", "GEO_TO_SHAPE"]


GEO_TO_SHAPE = {
    "SPHERE": L.SHAPE_SPHERE,
    "BOX": L.SHAPE_BOX,
    "CAPSULE": L.SHAPE_CAPSULE,
    "CYLINDER": L.SHAPE_CYLINDER,
    "PLANE": L.SHAPE_PLANE,
    "MESH": L.SHAPE_MESH,
    "CONVEX_MESH": L.SHAPE_MESH,
}
"""Newton's `GeoType` to this extension's shape codes, **by name and not by value**.

Newton's integers today are NONE 0, PLANE 1, HFIELD 2, SPHERE 3, CAPSULE 4,
ELLIPSOID 5, CYLINDER 6, BOX 7, MESH 8, CONE 9, CONVEX_MESH 10, GAUSSIAN 11. None of
that is a documented API, and a renumbering would silently turn every sphere into a
capsule. Looking the name up costs one dictionary access per shape per coupling step
and cannot go wrong quietly: a type that is not in here raises, naming the shape.

What is deliberately absent, and what happens if a model has one: `HFIELD`,
`ELLIPSOID`, `CONE` and `GAUSSIAN`. A height field is a terrain and wants its own
sampling kernel rather than a generic SDF; an ellipsoid has no closed-form exact signed
distance (the usual one is a bound, not a distance, and `q` is a distance ratio -- see
`lattice.stream`); a cone and a Gaussian splat are not shapes a robot's collision
geometry usually contains. Each of them raises with that reason and the suggestion that
fits: convert it to a mesh, which this extension does support, and which `ModelBuilder`
can do.
"""

_SCALE = {
    L.SHAPE_SPHERE: lambda s: (s[0], 0.0, 0.0),
    L.SHAPE_BOX: lambda s: (s[0], s[1], s[2]),
    L.SHAPE_CAPSULE: lambda s: (s[0], s[1], 0.0),
    L.SHAPE_CYLINDER: lambda s: (s[0], s[1], 0.0),
    L.SHAPE_PLANE: lambda s: (s[0], s[1], 0.0),
    L.SHAPE_MESH: lambda s: (s[0], s[1], s[2]),
}
"""How Newton's `shape_scale` maps onto `lattice.py`'s SDF parameters.

The identity for every type that this extension supports, which is the point: the SDFs
in `lattice.py` were written to Newton's measured packing rather than to a convention of
their own, so there is no conversion to get wrong. The table exists anyway because
`tests/test_live_couple.py` uses it to assert the packing against a model Newton builds,
and because `cylinder`'s third component (the barrel radius, for a rounded cylinder)
is dropped here rather than silently ignored in the kernel.
"""


def _rotate(quaternion: Any, vector: Any) -> np.ndarray:
    """Rotate `vector` by a quaternion in Warp and Newton's `(x, y, z, w)` order.

    Four lines rather than a dependency, and the same arithmetic `world._quat_matrix`
    does; it is here rather than imported so that this file's one host-side rotation
    does not reach into another module's private helper.
    """
    x, y, z, w = (float(v) for v in quaternion)
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    # v + 2 w (q x v) + 2 q x (q x v), the standard reduced form.
    axis = np.array([x, y, z])
    first = np.cross(axis, vector)
    return vector + 2.0 * w * first + 2.0 * np.cross(axis, first)


def _as_numpy(value: Any, length: int) -> tuple[float, ...]:
    """A Newton model attribute as a tuple of floats, whether it is on the host or not.

    `Model.gravity` is a `wp.array` or a `wp.vec3` depending on the Newton version, and
    on a CUDA device Warp refuses the implicit conversion that works on the CPU:
    "Warp does not support implicit NumPy conversion for arrays on device cuda:0."
    So `np.asarray(model.gravity)` is correct on the machine this was written on and a
    `TypeError` on the one it is meant for -- which is exactly how the first GPU run of
    the falling-sphere case died, forty-four seconds in, after the lattice had compiled.

    One helper rather than a `.numpy()` at the call site, because every Newton attribute
    this file reads from the host has the same shape of problem and there is no reason
    for the next one to rediscover it.
    """
    if hasattr(value, "numpy"):
        value = value.numpy()
    flat = np.asarray(value, dtype=np.float64).reshape(-1)[:length]
    return tuple(float(v) for v in flat)


def newton_shapes(model: Any) -> list[dict[str, Any]]:
    """Newton's shape table as plain Python, for inspection and for error messages.

    Not what the step loop uses -- that is `NewtonCoupling._shape_arrays`, which keeps
    everything on the device. This is for the report, for the tests and for the refusal
    when a model holds a shape type this extension cannot voxelise, which has to name
    the shape and so has to read the labels.
    """
    import newton

    kinds = model.shape_type.numpy()
    bodies = model.shape_body.numpy()
    transforms = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    labels = list(getattr(model, "shape_label", []) or [])
    out = []
    for index in range(len(kinds)):
        name = newton.GeoType(int(kinds[index])).name
        label = labels[index] if index < len(labels) else f"shape_{index}"
        if name not in GEO_TO_SHAPE:
            raise LiveUnavailable(
                f"shape {index} ({label!r}) is a Newton {name}, which the live fluid "
                f"backend cannot voxelise. It voxelises "
                f"{', '.join(sorted(GEO_TO_SHAPE))}; everything else should be a mesh, "
                f"which ModelBuilder can make from any of them. See "
                f"openreynolds/live/couple.py::GEO_TO_SHAPE for why each of the others "
                f"is left out"
            )
        code = GEO_TO_SHAPE[name]
        out.append({
            "index": index, "label": label, "geo": name, "kind": code,
            "body": int(bodies[index]),
            "position": tuple(float(v) for v in transforms[index][:3]),
            "orientation": tuple(float(v) for v in transforms[index][3:]),
            "scale": _SCALE[code](scales[index]),
        })
    return out


class NewtonCoupling:
    """One Newton model, one `FluidWorld`, stepped together.

        link = NewtonCoupling(model, solver, world)
        for _ in range(steps):
            link.step()

    and after each step `link.state.body_q` is the new pose and
    `link.world.wrench(body)` is the fluid wrench that produced it.

    ## What one `step()` does, in order

        1  poses        state.body_q / body_qd -> the world's lattice body table
        2  fluid        world.step(): voxelise, refill, N substeps, wrench accumulated
        3  clear        state.clear_forces()
        4  wrench       the fluid's surface wrench added into state.body_f
        5  buoyancy     rho_fluid * V * g added, with V the voxelised volume
        6  rotors       the actuator disks' reaction added
        7  newton       solver.step(state_0, state_1, control, contacts, dt)
        8  swap         state_0, state_1 = state_1, state_0

    Steps 1, 4, 5 and 6 are single Warp kernel launches on arrays that already live on
    the device, so **nothing in this loop touches the host**. That is the brief's one
    hard constraint on the step loop and it is why `poses_from_newton` and
    `wrench_to_newton` exist at all: the obvious implementation, `numpy()` out and
    `assign()` back, is four PCIe transfers per coupling step, and at a two-millisecond
    robot step that is most of the step.

    The fluid runs **before** Newton rather than after, and the wrench Newton gets is
    the mean over the interval the fluid has just crossed. That is a first-order
    explicit coupling: the force is evaluated at the start of the robot's step and held
    across it. The alternative -- iterating the pair to convergence within a step -- is
    what a tightly coupled fluid-structure solver does, and `benchmarks/moving_mesh/`
    in this repository records what it is worth there: a loosely coupled
    vortex-induced-vibration run measured a net fluid work per cycle of about -10% of
    the body's energy while that energy held constant, which is impossible, and the
    tightly coupled continuation measured +0.1%. That matters for a body on a spring at
    resonance. It does not matter at the mass ratios a robot in air has -- a quadrotor
    is a thousand times denser than the air it displaces, so the added-mass term the
    iteration exists to capture is a thousandth of its inertia. For a **neutrally
    buoyant** body in water it matters a great deal, and `one_way` is not the answer;
    the answer is a smaller `dt`, and `docs/live-sync.md` says so with the measurement
    from `benchmarks/live/` beside it.

    ## Buoyancy, and why the fluid has no gravity in it

    `FluidWorld` never applies gravity to the fluid. In a weakly compressible solver a
    hydrostatic pressure gradient is a real density gradient, and at the Mach numbers
    this runs at it would be a large one -- a metre of water is 10 kPa, which against
    `rho c_s^2` in lattice units is not a small perturbation. Every incompressible
    solver folds gravity into the pressure instead, and the lattice Boltzmann equivalent
    is to leave it out of the fluid and give the body its weight minus the weight of the
    fluid it displaces.

    So this class adds `-rho_fluid * V * g` to each body, with `V` the volume the
    **voxelisation** encloses (`lattice.volumes`) rather than the volume the geometry
    has. Those differ by a few per cent at twenty cells across a sphere, and a falling
    sphere's terminal velocity is set by the difference between two numbers of the same
    size, so using the analytic volume for the buoyancy and the voxelised surface for
    the drag does not balance and the error lands squarely on the quantity being graded.
    `DECISIONS_LIVE.md` D9.

    `buoyancy=False` turns it off, for a body in air where it is a part in a thousand,
    or for a caller who is applying it themselves.
    """

    def __init__(
        self,
        model: Any,
        solver: Any,
        world: FluidWorld,
        *,
        state_0: Any = None,
        state_1: Any = None,
        control: Any = None,
        contacts: Any = None,
        one_way: bool | None = None,
        buoyancy: bool = True,
        dt: float | None = None,
    ) -> None:
        import warp as wp

        self.wp = wp
        self.model = model
        self.solver = solver
        self.world = world
        self.state_0 = state_0 if state_0 is not None else model.state()
        self.state_1 = state_1 if state_1 is not None else model.state()
        self.control = control if control is not None else model.control()
        self.contacts = contacts
        self.one_way = world.one_way if one_way is None else bool(one_way)
        self.buoyancy = bool(buoyancy)
        self.dt = float(dt) if dt is not None else float(world.dt)
        self.steps = 0

        if self.state_0.body_f is None:
            raise LiveUnavailable(
                "this Newton model has no bodies, so there is nothing for a fluid to "
                "push on. Add a body with ModelBuilder.add_body() before finalizing -- "
                "note that add_body() already creates the free joint, so do not also "
                "call add_joint_free()"
            )

        device = world.device
        count = int(model.body_count)
        self.world.model = model
        # The world's body table is sized to Newton's and is written by a kernel from
        # here on. `_device_bodies_dirty` stays False for exactly that reason: the host
        # table in `world.py` is not the truth any more, this is.
        self.world.body_pose = wp.zeros(count, dtype=wp.transform, device=device)
        self.world.body_vel = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        self.world.body_com = wp.zeros(count, dtype=wp.vec3, device=device)
        self.world._wrench = wp.zeros((count, 6), dtype=L.DTYPE, device=device)
        self.world._device_bodies_dirty = False
        self.world._poses = [(0.0,) * 3 + (0.0, 0.0, 0.0, 1.0)] * count
        self.world._vels = [(0.0,) * 6] * count
        self.world._coms = [(0.0,) * 3] * count
        self._com_local = np.asarray(model.body_com.numpy(), dtype=np.float64).reshape(
            -1, 3).copy()
        """`Model.body_com`, the centre of mass in BODY-LOCAL coordinates, on the host.

        Read once, because it does not move: a rigid body's centroid is fixed in its own
        frame. The world centre of mass is `transform_point(body_q, body_com)` and is
        recomputed every step in `_push_poses` from this and the pose.

        It exists because of a bug it caused by being absent. `world._coms` was seeded
        with zeros here and never refreshed, and `FluidWorld.rotor_reaction` takes every
        rotor's moment arm **to the centre of mass** -- so the arm was measured from the
        world origin instead. For a quadrotor hovering 0.84 m along x from that origin,
        four rotors whose moments cancel exactly about the true centroid produced
        `r0 x F_total` about the wrong point: 9.9 N.m on a body with about 0.01 kg.m^2 of
        inertia, which is a thousand radians per second squared. It tumbled inside a
        dozen coupling steps, the thrust vector tipped over with it, and the symptom was
        a drone that fell faster than free fall with its controller railed at three times
        its weight -- a failure that looks like a controller problem, a solver problem, or
        a sign error anywhere in the chain, and is none of those.
        """
        self.world._labels = list(getattr(model, "body_label", []) or
                                  [f"body_{i}" for i in range(count)])

        self._shape_kinds, self._shape_body, self._shape_local, self._shape_scale = \
            self._static_shape_arrays()
        self._shape_pose = wp.zeros(max(1, int(model.shape_count)), dtype=wp.transform,
                                    device=device)
        self._shape_mesh = getattr(model, "shape_source_ptr", None)
        if self._shape_mesh is None:
            self._shape_mesh = wp.zeros(max(1, int(model.shape_count)),
                                        dtype=wp.uint64, device=device)
        self._volumes = wp.zeros(count, dtype=wp.int32, device=device)
        # A free-surface run's submerged volume, in fractional cells, so a draft is not
        # quantised to the lattice. `_push_extras` says why it is not the whole volume.
        self._submerged = wp.zeros(count, dtype=L.DTYPE, device=device)
        self._buoyancy = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        self._gravity = _as_numpy(model.gravity, 3)

        # The world voxelises from its own host shape list; with a model behind it the
        # shapes are Newton's and the poses are composed on the device every step. The
        # hand-off is this one assignment, and `world._voxelize` honours it.
        self.world._newton_shapes = (
            self._shape_kinds, self._shape_body, self._shape_pose, self._shape_scale,
            self._shape_mesh,
        )
        # **A particle backend cannot use that handoff and must be told so out loud.**
        # The lattice re-voxelises Newton's shapes every step, because a moving body
        # changes which cells are solid. SPH samples boundary particles once per shape,
        # on the host, at `add_shape` time -- a rigid shape's particles never move in its
        # own frame -- so it stores this tuple and ignores it. A caller who builds a
        # Newton model, hands it here and expects the fluid to find the bodies gets
        # bodies with no wetted surface and a wrench of **exactly zero**, which reads as
        # a coupling that is running and finding no force rather than as a mistake.
        # **Fail open.** The first version of this guard asked for `world.shapes`, an
        # attribute no `SphWorld` has; `getattr` returned `None`, `not None` was true, and
        # it raised on every correctly built world -- four H100 validation runs died on it
        # before anything was simulated. A guard that cannot name the state it checks must
        # refuse only when it positively establishes the bad case, never when it simply
        # fails to find what it was looking for.
        #
        # The real state: `add_shape` appends to `_shapes`, `boundary_from_domain` appends
        # to `_bdy_x`, and `n_boundary` is only summed once the world is prepared. So a
        # world with any of the three is fine, and only one with none of them is the
        # zero-force trap this is here to catch.
        _bdy = (len(getattr(self.world, "_shapes", ()) or ())
                + len(getattr(self.world, "_bdy_x", ()) or ())
                + int(getattr(self.world, "n_boundary", 0) or 0))
        if self._style == "particles" and _bdy == 0:
            raise ValueError(
                "this SPH world has no boundary particles, so every body would feel "
                "exactly zero force and the run would look like it was working. "
                "An SPH world samples a shape's boundary particles when the shape is "
                "added, not from Newton each step: call world.add_shape(...) for each "
                "coupled body (and world.boundary_from_domain() for the walls) before "
                "constructing the coupling. docs/sph-contract.md has the signatures"
            )
        # **Push the poses once, here, so the world is voxelised correctly before anybody
        # steps anything.** Until 2026-11-23 this happened only inside `step`, and the
        # consequence was quiet and expensive: `world.step()` syncs from the world's own
        # pose arrays, so a body that exists only in Newton sat at the identity transform
        # -- the origin -- for every world-only step.
        #
        # That is not a hypothetical. A floating body has to be released into water that
        # has already settled hydrostatically, and the obvious way to settle it is to
        # step the world while leaving the body alone. Doing that voxelised a 40 mm box
        # into the corner of the domain, 216 cells of it instead of 1,728, straddling two
        # walls; the measured "buoyancy" was then a force on a box wedged in a corner,
        # and it did not change when the box was moved, because the box was never where
        # it was put. Four separate measurements were taken that way before the voxel
        # count gave it away.
        self._push_poses()

    # -- the shapes, once --------------------------------------------------------------

    def _static_shape_arrays(self):
        """Types, bodies, local transforms and scales: everything that does not move.

        Read through `newton_shapes` rather than straight off the device arrays, so
        that a shape type this extension cannot voxelise is refused by name here, at
        construction, instead of silently becoming whatever the kernel's `else` branch
        is. A model is finalised once and this runs once.
        """
        import warp as wp

        shapes = newton_shapes(self.model)
        count = max(1, len(shapes))
        kinds = np.zeros(count, dtype=np.int32)
        bodies = np.full(count, -1, dtype=np.int32)
        local = np.zeros((count, 7), dtype=np.float32)
        local[:, 6] = 1.0
        scales = np.ones((count, 3), dtype=np.float32)
        for shape in shapes:
            index = shape["index"]
            kinds[index] = shape["kind"]
            bodies[index] = shape["body"]
            local[index, :3] = shape["position"]
            local[index, 3:] = shape["orientation"]
            scales[index] = shape["scale"]
        device = self.world.device
        return (
            wp.array(kinds, dtype=wp.int32, device=device),
            wp.array(bodies, dtype=wp.int32, device=device),
            wp.array(local, dtype=wp.transform, device=device),
            wp.array(scales / self._to_solver, dtype=wp.vec3, device=device),
        )

    # -- the step ----------------------------------------------------------------------

    def step(self, dt: float | None = None) -> dict[str, Any]:
        """One Newton step and the fluid substeps inside it. Returns the fluid's cost."""
        import warp as wp

        step_dt = float(self.dt if dt is None else dt)
        self._push_poses()
        cost = self.world.step()

        self.state_0.clear_forces()
        if not self.one_way:
            self._push_wrench()
        self.steps += 1

        # Five positionals and no keywords. newton 1.8.0's `step` takes a keyword-only
        # `observables=None` that 1.6.1 (which Isaac Lab 3.0.0 pins) does not have, so
        # naming anything here would break on the older one; `contacts` stays an
        # explicit positional for the same reason, since on 1.6.1 it has no default.
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts,
                         step_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        del wp
        return cost

    @property
    def state(self) -> Any:
        """The current state: the one `step()` will read next, and the one just written."""
        return self.state_0

    @property
    def _style(self) -> str:
        """`"lattice"` or `"particles"`: which backend's units the coupling is talking to.

        **Five lines in this file divided by `world.dx` and that made it a lattice-only
        file.** The lattice solves in index space, so a shape's scale, a body's pose and
        the wrench it accumulates all need converting by the spacing; SPH solves in
        metres, so all three are already SI and every one of those conversions is the
        identity. Reported by the world rather than sniffed from its type, so a third
        backend says which arithmetic it wants instead of being guessed at.

        Measured, before this existed: an `SphWorld` reached
        `AttributeError: 'SphWorld' object has no attribute 'dx'` from
        `_static_shape_arrays`, which blocked all four floating-body cases on a line that
        had nothing to do with floating bodies.
        """
        return str(getattr(self.world, "coupling_style", "lattice"))

    @property
    def _to_solver(self) -> float:
        """Divide an SI length by this to get the solver's own coordinates."""
        return float(self.world.dx) if self._style == "lattice" else 1.0

    @property
    def _step_seconds(self) -> float:
        """The solver's own timestep, for turning a velocity into solver units.

        The lattice keeps this on a `Scaling`; a particle world has no lattice to scale
        to and works in SI, so the conversion is 1. Read through this rather than
        `world.scaling.dt` because an `SphWorld` has no `scaling` at all -- measured, not
        predicted: a coupling raised `AttributeError: no attribute 'scaling'` from
        `_push_poses` before it managed a single step.
        """
        return (float(self.world.scaling.dt) if self._style == "lattice" else 1.0)

    def _push_poses(self) -> None:
        """Newton's body state and shape poses to the lattice, on the device."""
        import warp as wp

        origin = wp.vec3(*[float(v) for v in self.world.domain.origin])
        inv_dx = L.DTYPE(1.0 / self._to_solver)
        # Half a cell for the lattice, nothing for particles. See `poses_from_newton`.
        node_offset = L.DTYPE(0.5 if self._style == "lattice" else 0.0)
        wp.launch(
            L.poses_from_newton, dim=int(self.model.body_count),
            inputs=[self.state_0.body_q, self.state_0.body_qd, self.model.body_com,
                    self.world.body_pose, self.world.body_vel, self.world.body_com,
                    origin, inv_dx, L.DTYPE(self._step_seconds), node_offset],
            device=self.world.device)
        wp.launch(
            L.shapes_from_newton, dim=int(self.model.shape_count),
            inputs=[self._shape_body, self._shape_local, self.state_0.body_q,
                    self._shape_pose, origin, inv_dx, node_offset],
            device=self.world.device)

        # **Unconditional, and it was conditional until 2026-11-23.** The host needs
        # these poses for three things: an actuator disk's position and axis, the
        # buoyancy's moment arm, and `FluidWorld._maybe_shift`, which re-centres a
        # moving window on the body it follows. The condition used to be
        # `rotors or buoyancy`, which covers the first two and silently drops the third.
        #
        # What that cost: a run with `follow=0`, no rotors and `buoyancy=False` -- a
        # submarine, say -- left `world._poses` at the identity transform. `_maybe_shift`
        # then measured the body's drift from the ORIGIN rather than from its real
        # position, fired one spurious shift of the whole half-domain on the first step,
        # and re-centred the window on the wrong place. Measured: the hull's voxelisation
        # collapsed from 2,176 cells to 224 and eroded to 68 over 600 steps as it left
        # the window, and the drag it felt fell from about 3 uN to 0.5. An independent
        # measurement of the same thing on a 23,184-cell hull saw `solid_cells` reach
        # **zero** and `world.wrench` return exactly 0.0, with the body then accelerating
        # at `T/m` to four significant figures -- a submarine flying through vacuum.
        #
        # The windowed falling-sphere row is NOT affected and was checked rather than
        # assumed: it runs `buoyancy=True`, so the old condition happened to cover it.
        #
        # The readback does not weaken the rule it was conditioned for. "No host round
        # trips inside the step loop" is about the lattice -- the substeps, the
        # voxelisation and the wrench never leave the device, and those are the three
        # costs that scale with the grid. This one is 28 bytes a body, once per coupling
        # step, and conditioning it on which feature happens to be enabled traded that
        # for a silent wrong answer.
        poses = self.state_0.body_q.numpy()
        self.world._poses = [tuple(float(v) for v in row) for row in poses]
        # And the centres of mass, which is the half that was missing. Newton's
        # `body_com` is body-local, so the world one is the pose applied to it; a
        # rotor's moment arm is measured from it, and a stale or zeroed one puts the
        # arm at the world origin. See `_com_local`.
        self.world._coms = [
            tuple(float(v) for v in (np.asarray(row[:3])
                                     + _rotate(row[3:], self._com_local[index])))
            for index, row in enumerate(poses)
        ]

    def _push_wrench(self) -> None:
        """The fluid's wrench, the buoyancy and the rotor reaction, into `body_f`."""
        import warp as wp

        # `dt2` below is only used by the lattice force scales; a particle world keeps
        # its wrench in SI and never reaches them, so this must not demand a `Scaling`
        # it does not have.
        dt_step = self._step_seconds
        rho = self.world.fluid.rho
        dt2 = dt_step * dt_step
        # The wrench accumulated one contribution per lattice substep; Newton wants the
        # mean over the interval it is about to integrate.
        per_substep = 1.0 / max(1, self.world.substeps)
        if self._style == "lattice":
            force_scale = rho * self.world.dx ** 4 / dt2
            torque_scale = rho * self.world.dx ** 5 / dt2
        else:
            force_scale = torque_scale = 1.0
        wp.launch(
            L.wrench_to_newton, dim=int(self.model.body_count),
            inputs=[self.world._wrench, self.state_0.body_f,
                    # The lattice accumulates its wrench in lattice units, so it
                    # carries `rho dx^4/dt^2` for a force and one more `dx` for a
                    # moment. SPH accumulates newtons and newton-metres directly, so
                    # both scales are 1 and only the substep mean remains.
                    L.DTYPE(force_scale * per_substep),
                    L.DTYPE(torque_scale * per_substep),
                    1],
            device=self.world.device)

        # `rotors` is a lattice-world concept -- a particle world has no actuator disks
        # -- and the buoyancy branch beside it already refuses on the particle path, so
        # this must not demand the attribute to decide it is not needed.
        if self.buoyancy or getattr(self.world, "rotors", ()):
            self._push_extras()

    def _push_extras(self) -> None:
        """Buoyancy from the voxelised volume, and the rotors' reaction. One launch each.

        This is the one place in the loop that reads a device array back to the host,
        and it is a single `int32` per body -- `body_count` integers, not a field. It is
        here and not in a kernel because the rotor thrusts are set from the host by
        whatever is flying the vehicle, so the host already has to compose that vector
        each step; adding the buoyancy to it costs the volume count and nothing else.
        A model with no rotors and `buoyancy=False` never runs this.
        """
        import warp as wp

        count = int(self.model.body_count)
        extras = np.zeros((count, 6), dtype=np.float32)
        if self.buoyancy and self._style == "particles":
            # SPH has a real waterline and real hydrostatic pressure, so buoyancy comes
            # out of the pressure field through the boundary particles and adding
            # `rho V g` on top would count it twice. `docs/sph-contract.md` says
            # `buoyancy=False` is the default there and should stay it; this refuses
            # rather than silently double-counting, because a draft that comes out
            # halved looks like a solver fault and is not one.
            raise ValueError(
                "buoyancy=True is for the lattice backend, which has no free surface "
                "and therefore no hydrostatic gradient to integrate (DECISIONS_LIVE.md "
                "D9). An SPH world produces its own buoyancy from the pressure field; "
                "adding rho V g on top counts it twice. Pass buoyancy=False"
            )
        if self.buoyancy:
            level = getattr(self.world, "waterline", None)
            if level is None:
                self._volumes.zero_()
                wp.launch(L.volumes, dim=self.world.shape,
                          inputs=[self.world.owner, self._volumes],
                          device=self.world.device)
                cells = self._volumes.numpy().astype(np.float64)
            else:
                # **A free-surface run buoys on its SUBMERGED volume, sub-cell
                # accurate**, because a floating body's draft would otherwise be
                # quantised to the lattice: `volumes` counts whole cells, and at twelve
                # cells through a draft that is 8% steps in the one number the
                # Archimedes case grades to 2%. `submerged_volumes` weights each solid
                # cell by the fraction of its height below `world.waterline`, so the
                # displaced volume -- and the draft -- is continuous in the body's
                # position.
                #
                # Whether the momentum exchange could supply this force instead, from
                # the pressure field it already has, is **open**. It was measured as
                # failing and that measurement was invalid: the body had been voxelised
                # into the corner of the domain, because `world.step()` does not sync
                # poses from Newton and the settling loop used it. The constructor now
                # pushes poses once so that cannot recur, and the question has not been
                # re-measured. Nothing here rests on the answer: a sub-cell displaced
                # volume is needed for the draft either way.
                self._submerged.zero_()
                wp.launch(L.submerged_volumes, dim=self.world.shape,
                          inputs=[self.world.owner, self._submerged, 2,
                                  L.DTYPE(self.world._to_lattice(
                                      (0.0, 0.0, float(level)))[2])],
                          device=self.world.device)
                cells = self._submerged.numpy().astype(np.float64)
            volume = cells * self.world.dx ** 3
            displaced = -self.world.fluid.rho * volume[:, None] * np.asarray(self._gravity)
            extras[:, :3] += displaced.astype(np.float32)
            self.displaced_volume = volume
        for body in {r.body for r in self.world.rotors}:
            if 0 <= body < count:
                # All six components: a rotor off the centre of mass applies a moment,
                # and dropping it leaves a multirotor with no attitude control at all.
                # `FluidWorld.rotor_reaction` has the history.
                extras[body] += np.asarray(self.world.rotor_reaction(body),
                                           dtype=np.float32)
        if not extras.any():
            return
        self._buoyancy.assign(extras)
        wp.launch(_add_wrench, dim=count,
                  inputs=[self._buoyancy, self.state_0.body_f],
                  device=self.world.device)

    # -- what the caller reads ---------------------------------------------------------

    def report(self) -> list[str]:
        """The pair as a human reads it: the model, the solver, and the coupling."""
        lines = [
            f"  newton:     {self.model.body_count} bodies, "
            f"{self.model.shape_count} shapes, "
            f"{type(self.solver).__name__}",
            f"  gravity:    {self._gravity} m/s2"
            f"{'  (fluid buoyancy on, from the voxelised volume)' if self.buoyancy else ''}",
        ]
        if self.one_way:
            lines.append(
                "  NOTE:       one-way: the fluid wrench is computed and is NOT applied "
                "to the bodies. A debug mode; nothing measured here is a coupled result"
            )
        return lines + self.world.report()


# `lattice.py` imports Warp at module scope, so importing this module already needs it.
# What keeps `import openreynolds.live` free of Warp is `__init__.py`'s lazy
# `__getattr__`, not anything here, so this is an ordinary module-level kernel.
import warp as wp  # noqa: E402


@wp.kernel
def _add_wrench(extra: wp.array(dtype=wp.spatial_vector),
                body_f: wp.array(dtype=wp.spatial_vector)):
    """Add a precomputed per-body wrench into Newton's accumulator.

    `wp.atomic_add` on a whole `wp.array(dtype=wp.spatial_vector)` element does work --
    it was measured, 3000 threads into 3 bodies summing exactly -- and it is what
    Newton's own propeller examples use. A plain add is enough here because this is one
    thread per body with no contention; the atomics are in `lattice.stream`, where
    thousands of boundary links hit the same body.
    """
    b = wp.tid()
    body_f[b] = body_f[b] + extra[b]
