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
        self._buoyancy = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        self._gravity = tuple(float(v) for v in np.asarray(model.gravity).reshape(3))

        # The world voxelises from its own host shape list; with a model behind it the
        # shapes are Newton's and the poses are composed on the device every step. The
        # hand-off is this one assignment, and `world._voxelize` honours it.
        self.world._newton_shapes = (
            self._shape_kinds, self._shape_body, self._shape_pose, self._shape_scale,
            self._shape_mesh,
        )

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
            wp.array(scales / self.world.dx, dtype=wp.vec3, device=device),
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

    def _push_poses(self) -> None:
        """Newton's body state and shape poses to the lattice, on the device."""
        import warp as wp

        origin = wp.vec3(*[float(v) for v in self.world.domain.origin])
        inv_dx = L.DTYPE(1.0 / self.world.dx)
        wp.launch(
            L.poses_from_newton, dim=int(self.model.body_count),
            inputs=[self.state_0.body_q, self.state_0.body_qd, self.model.body_com,
                    self.world.body_pose, self.world.body_vel, self.world.body_com,
                    origin, inv_dx, L.DTYPE(self.world.scaling.dt)],
            device=self.world.device)
        wp.launch(
            L.shapes_from_newton, dim=int(self.model.shape_count),
            inputs=[self._shape_body, self._shape_local, self.state_0.body_q,
                    self._shape_pose, origin, inv_dx],
            device=self.world.device)

        if self.world.rotors or self.buoyancy:
            # An actuator disk's position and axis are the body's, and the thrust that
            # drives it comes from the host -- whatever is flying the vehicle sets it
            # every step. So the host needs this body's pose anyway, and reading
            # `body_count` transforms back is the cheapest way to get it: 28 bytes a
            # body, once per coupling step.
            #
            # This is the one readback in the loop and it does not weaken the rule the
            # rule is for. "No host round trips inside the step loop" is about the
            # lattice: the substeps, the voxelisation and the wrench never leave the
            # device, and those are the three things whose cost scales with the grid.
            # This one scales with the body count, and a model with no rotors and
            # `buoyancy=False` does not run it at all.
            poses = self.state_0.body_q.numpy()
            self.world._poses = [tuple(float(v) for v in row) for row in poses]

    def _push_wrench(self) -> None:
        """The fluid's wrench, the buoyancy and the rotor reaction, into `body_f`."""
        import warp as wp

        scaling = self.world.scaling
        rho = self.world.fluid.rho
        dt2 = scaling.dt * scaling.dt
        # The wrench accumulated one contribution per lattice substep; Newton wants the
        # mean over the interval it is about to integrate.
        per_substep = 1.0 / max(1, self.world.substeps)
        wp.launch(
            L.wrench_to_newton, dim=int(self.model.body_count),
            inputs=[self.world._wrench, self.state_0.body_f,
                    L.DTYPE(rho * self.world.dx ** 4 / dt2 * per_substep),
                    L.DTYPE(rho * self.world.dx ** 5 / dt2 * per_substep),
                    1],
            device=self.world.device)

        if self.buoyancy or self.world.rotors:
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
        if self.buoyancy:
            self._volumes.zero_()
            wp.launch(L.volumes, dim=self.world.shape,
                      inputs=[self.world.owner, self._volumes],
                      device=self.world.device)
            cells = self._volumes.numpy().astype(np.float64)
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
