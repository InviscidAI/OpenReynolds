"""`FluidWorld`: the lattice, the bodies in it, and the step a caller actually calls.

Everything stateful lives here. `lattice.py` is kernels with no memory; this file owns
the arrays, decides what runs in what order, and is the only place that knows how many
substeps a coupling step has.

## The step, in order, and why that order

    step()
      1  poses        body transforms and velocities in, from whatever drives them
      2  window       shift the lattice if the followed body has drifted (optional)
      3  voxelize     phi and owner, from the shapes where they are now
      4  refill       nodes that have just left a body get a state
      5  rotors       the actuator-disk force field, if there are any
      6  zero wrench  the force accumulator for this coupling step
      7  N times:     collide, stream (which accumulates the wrench), apply_faces
      8  wrenches     available to the caller, and pushed into Newton by `couple.py`

Steps 1 to 6 happen once per *coupling* step and 7 happens `substeps` times, which is
the whole shape the brief asks for: "each Newton step (dt_robot, e.g. 1 to 5 ms) runs N
LBM substeps". The body is held fixed across the substeps. That is a first-order
coupling in time and it is the honest one to write down: the alternative, interpolating
the pose across the substeps, needs the pose at the *end* of the Newton step, which does
not exist until Newton has run, which needs the wrench, which needs the substeps.

`voxelize` before `refill` because `refill` reads both the new `phi` and the old one.
`refill` before the substeps because a fresh node with garbage in it is read by its
neighbours on the first `stream`. The wrench is zeroed once per coupling step and
accumulated over all `substeps`, then divided by `substeps`: what Newton wants is the
mean wrench over the interval it is about to integrate, not the instantaneous one at
either end.

## Everything outside is SI; everything inside is lattice units

The boundary is exactly this file. `units.Scaling` does the conversion, every public
method takes and returns metres, seconds, newtons and kilograms, and every array the
kernels see is in lattice units. A body's pose is converted on the way in and its wrench
on the way out, and nowhere else.

## Bodies the world owns, and bodies Newton owns

`FluidWorld` keeps its own body table: pose, velocity and centre of mass, in lattice
units. When there is a Newton model, its bodies are the first `model.body_count` entries
and `couple.NewtonCoupling` refreshes them from a `State` every step. Bodies added with
`add_body` after that are the world's own -- a fixed cylinder, a sphere held still in a
stream, a ground plane -- and nobody refreshes them, which is what "static" means here.

The point of one table is that `stream` does not know or care which is which: it reads
`owner`, looks up a pose and a velocity, and accumulates a wrench. A static cylinder
gets its drag measured by exactly the code that puts a force on a quadrotor.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import lattice as L
from .units import Scaling, UnphysicalScaling, fluid as resolve_fluid
# Defined in `units.py`, which imports nothing but the standard library, and re-exported
# here because this is where the public API lives and `FluidWorld` is its first caller.
# The move was forced: reading this module imports Warp (through `lattice.py`), so a
# refusal defined here could only be raised on a machine that did not need it. The
# names are unchanged, and `from .world import require` still works.
from .units import LiveUnavailable, require

__all__ = ["Domain", "FluidWorld", "Rotor", "LiveUnavailable", "require"]


FACE_CODES = {
    "periodic": L.FACE_PERIODIC, "wall": L.FACE_WALL, "slip": L.FACE_SLIP,
    "inlet": L.FACE_INLET, "outlet": L.FACE_OUTLET, "open": L.FACE_OUTLET,
}
"""Face rules by the name a case file or a CLI flag writes. `open` is `outlet`: a
zero-gradient outflow is what "open" means for a box with no imposed flow through it,
and two names for it is cheaper than explaining the difference at every call site."""

FACE_ORDER = ("-x", "+x", "-y", "+y", "-z", "+z")


@dataclass
class Domain:
    """The box, in metres, and what each of its six faces does.

    `origin` is the world position of lattice node `(0, 0, 0)` and `size` is the extent,
    so the box is `[origin, origin + size]`. A `Domain` is immutable to the user and is
    moved only by the moving window, which rewrites `origin` and nothing else.

    Faces are named rather than indexed in the constructor, because `faces=(1, 4, 0, 0,
    2, 2)` is unreadable and is the sort of thing that silently puts the cavity's lid on
    the floor. `faces` accepts a dict keyed by `-x`, `+x`, `-y`, `+y`, `-z`, `+z`, or a
    single name applied to all six.
    """

    origin: tuple[float, float, float]
    """The low corner of the box: the plane a `wall` face puts its no-slip surface on.

    **Not** the position of node `(0, 0, 0)`, and the difference is half a cell. A
    halfway bounce-back puts the wall half a lattice spacing outside the last fluid
    node, so with `n` nodes across an axis the wall-to-wall extent is `n * dx` and node
    `i` sits at `origin + (i + 0.5) * dx`. Getting that half cell wrong is a `1/(2n)`
    error in every length in the problem, and it looks exactly like a solver defect --
    `toolbox/lbm.py::centres` exists for the same reason and says the same thing. It is
    written down here, and `node_coordinates` is the only place anything reads it from.
    """
    size: tuple[float, float, float]
    """Wall-to-wall extent in metres, or the period on a periodic axis."""
    faces: Any = "outlet"
    face_velocity: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    """Wall velocities in m/s, by face name. A moving wall face is how the lid-driven
    cavity's lid is expressed: it is a boundary condition, not a body, so it needs no
    shape, no voxelisation and no force accumulation."""

    def __post_init__(self) -> None:
        self.origin = tuple(float(v) for v in self.origin)  # type: ignore[assignment]
        self.size = tuple(float(v) for v in self.size)  # type: ignore[assignment]
        if any(v <= 0 for v in self.size):
            raise UnphysicalScaling(f"a domain has positive extent; got size {self.size}")

    def face_rules(self) -> list[int]:
        """The six face codes in `lattice.py`'s order (`2 * axis + side`)."""
        spec = self.faces
        if isinstance(spec, str):
            spec = dict.fromkeys(FACE_ORDER, spec)
        if isinstance(spec, dict):
            unknown = set(spec) - set(FACE_ORDER)
            if unknown:
                raise UnphysicalScaling(
                    f"a domain has six faces, named {', '.join(FACE_ORDER)}; "
                    f"got {', '.join(sorted(unknown))}"
                )
            named = [spec.get(name, "outlet") for name in FACE_ORDER]
        else:
            named = list(spec)
            if len(named) != 6:
                raise UnphysicalScaling(f"six face rules, got {len(named)}")
        out = []
        for name in named:
            if isinstance(name, int):
                out.append(int(name))
                continue
            try:
                out.append(FACE_CODES[str(name).strip().lower()])
            except KeyError:
                raise UnphysicalScaling(
                    f"no face rule called {name!r}: known rules are "
                    f"{', '.join(sorted(FACE_CODES))}"
                ) from None
        self._check_periodic_pairs(out)
        return out

    @staticmethod
    def _check_periodic_pairs(rules: list[int]) -> None:
        """Periodic is a property of an axis, not of a face.

        A box periodic on `-x` and walled on `+x` is not a thing: streaming wraps one
        way and reflects the other, so mass leaves at one end and is reflected at the
        other, and the symptom is a slow drift in total density that looks like a
        solver bug. Refused here, where it is one comparison.
        """
        for axis in range(3):
            low, high = rules[2 * axis], rules[2 * axis + 1]
            if (low == L.FACE_PERIODIC) != (high == L.FACE_PERIODIC):
                name = "xyz"[axis]
                raise UnphysicalScaling(
                    f"the {name} axis is periodic on one face and not the other "
                    f"(-{name} is {L.FACE_NAMES[low]}, +{name} is {L.FACE_NAMES[high]}). "
                    f"Periodic is a property of the axis: set both or neither"
                )

    def resolution(self, dx: float) -> tuple[int, int, int]:
        """Node counts for a cell size, at least 3 a side so D3Q19 has a neighbour.

        `n = round(size / dx)`, not `+ 1`: see `origin`. The nodes do not sit on the
        walls, they sit half a cell inside them.
        """
        return tuple(max(3, int(round(extent / dx))) for extent in self.size)

    def spacing(self, cells: int) -> float:
        """The cell size that puts `cells` cells along the longest axis."""
        return max(self.size) / float(max(1, int(cells)))


@dataclass
class Rotor:
    """An actuator disk, in SI, attached to a body (or to nothing).

    `centre` and `axis` are in the body's own frame, so a rotor stays where it is bolted
    as the airframe moves. `axis` points the way the rotor pushes the *vehicle*: the
    fluid gets the opposite, and the pair is applied in `FluidWorld._apply_rotors` so
    that Newton's third law is visible in one place.

    `thickness` defaults to two cells at the world's resolution rather than to a
    physical number, because the disk is a numerical device: thinner than two cells and
    the momentum source falls between lattice nodes and the delivered thrust depends on
    where the rotor happens to sit.
    """

    body: int
    centre: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    thickness: float = 0.0
    thrust: float = 0.0
    """Total thrust in newtons. Set per step by whatever is flying the thing."""


class FluidWorld:
    """A lattice Boltzmann fluid, coupled to rigid bodies, stepped once per robot step.

    The minimal use, which is the brief's:

        world = FluidWorld(domain=Domain((0, 0, 0), (1, 1, 1)), resolution=64,
                           fluid="water", dt=1e-3)
        world.initialise(velocity=(0.1, 0, 0))
        for _ in range(1000):
            world.step()

    and with Newton on the other side, `couple.NewtonCoupling` wraps the pair so that
    one call advances both. See `docs/live-sync.md`.
    """

    def __init__(
        self,
        model: Any = None,
        domain: Domain | None = None,
        resolution: int | None = None,
        fluid: str | Any = "air",
        *,
        dx: float | None = None,
        dt: float | None = None,
        substeps: int | None = None,
        reference_velocity: float = 0.0,
        peak_velocity: float | None = None,
        collision: str = "trt",
        smagorinsky: float = 0.0,
        scaling: Scaling | None = None,
        device: str | None = None,
        one_way: bool = False,
        follow: int | None = None,
        follow_slack: float = 0.25,
        mesh_search_cells: float = 4.0,
        far_cells: float = 3.0,
    ) -> None:
        """
        `model` is a `newton.Model` or None. With one, its bodies and shapes are read
        into the world's tables by `couple.NewtonCoupling`; without one, bodies are
        added by hand, which is what the validation cases do.

        `resolution` is cells along the **longest** domain axis; `dx` gives the cell
        size directly and wins if both are given. One of them is required: a lattice
        without a resolution is the one thing this class cannot guess, because the whole
        trade-off between cost and accuracy lives in it.

        `dt` is the **coupling** step -- the robot's step, the interval one `step()`
        advances the fluid by. With `substeps` given, the lattice step is `dt/substeps`;
        without, `units.Scaling.for_coupling` picks the substep count from the velocity
        ceiling and explains itself if it cannot.

        `collision` is `trt`, `bgk` or `regularized`; `smagorinsky` is the Smagorinsky
        constant `C_s` (0 disables the LES, 0.1 to 0.17 is the usual range, and it is
        squared on the way into the kernel). `DECISIONS_LIVE.md` D3 is why there is a
        choice.

        `one_way` is the debug mode the brief asks for: the fluid feels the bodies and
        the bodies do not feel the fluid. It changes nothing in this file -- the wrench
        is still computed, because a one-way run whose forces you cannot see is not a
        debugging tool -- it only stops `couple.py` pushing it into Newton.

        `follow` is a body index for the moving window, and `follow_slack` the fraction
        of the box that body may drift from the centre before the window shifts.
        """
        self.wp = require()
        import warp as wp

        if domain is None:
            raise UnphysicalScaling("a FluidWorld needs a Domain: the box it solves in")
        if dx is None:
            if resolution is None:
                raise UnphysicalScaling(
                    "a FluidWorld needs a resolution: either `resolution=` (cells "
                    "along the longest axis of the domain) or `dx=` (metres per cell). "
                    "There is no default, because the whole cost of the run is in it"
                )
            dx = domain.spacing(resolution)

        self.domain = domain
        self.fluid = resolve_fluid(fluid)
        self.device = device or str(wp.get_preferred_device())
        self.collision = str(collision).strip().lower()
        if self.collision not in L.SCHEMES:
            raise UnphysicalScaling(
                f"no collision operator called {self.collision!r}: "
                f"{', '.join(sorted(L.SCHEMES))}"
            )
        self.smagorinsky = float(smagorinsky)
        self.one_way = bool(one_way)

        if scaling is not None:
            self.scaling = scaling
            self.substeps = int(substeps or 1)
            self.dt = self.scaling.dt * self.substeps
        elif dt is None:
            raise UnphysicalScaling(
                "a FluidWorld needs a coupling step: `dt=` in seconds (the robot's "
                "step), or a ready-made `scaling=`"
            )
        else:
            self.scaling, self.substeps = Scaling.for_coupling(
                dx, dt, self.fluid, substeps=substeps,
                reference_velocity=reference_velocity, peak_velocity=peak_velocity,
                scheme=self.collision, les=self.smagorinsky > 0.0,
            )
            self.dt = float(dt)

        self.dx = float(self.scaling.dx)
        self.shape = self.domain.resolution(self.dx)
        self.nx, self.ny, self.nz = self.shape
        self.nodes = self.nx * self.ny * self.nz

        # -- the lattice -------------------------------------------------------------
        allocation = L.lattice_shape(self.shape)
        self.f = wp.zeros(allocation, dtype=L.DTYPE, device=self.device)
        self.f_next = wp.zeros(allocation, dtype=L.DTYPE, device=self.device)
        self.phi = wp.full(self.shape, L.DTYPE(far_cells), dtype=L.DTYPE,
                           device=self.device)
        self.phi_prev = wp.full(self.shape, L.DTYPE(far_cells), dtype=L.DTYPE,
                                device=self.device)
        self.owner = wp.full(self.shape, -1, dtype=wp.int32, device=self.device)
        self.owner_prev = wp.full(self.shape, -1, dtype=wp.int32, device=self.device)
        self.far = float(far_cells)
        self.mesh_search = float(mesh_search_cells)

        rules = self.domain.face_rules()
        self.face_rules = rules
        self.faces = wp.array(np.asarray(rules, dtype=np.int32), dtype=wp.int32,
                              device=self.device)
        velocities = np.zeros((6, 3), dtype=np.float32)
        for name, value in self.domain.face_velocity.items():
            if name not in FACE_ORDER:
                raise UnphysicalScaling(
                    f"face_velocity names a face that does not exist: {name!r}"
                )
            velocities[FACE_ORDER.index(name)] = [
                self.scaling.velocity_to_lattice(v) for v in value
            ]
        self.face_velocity = wp.array(velocities, dtype=wp.vec3, device=self.device)

        # The inlet profile covers the -x face only; `set_inlet_profile` fills it. A
        # 1x1 placeholder keeps the kernel signature uniform -- Warp needs a real array
        # even when the flag says not to read it.
        self._inlet_profile = wp.zeros((1, 1), dtype=wp.vec3, device=self.device)
        self._use_profile = 0

        # -- the bodies --------------------------------------------------------------
        self._poses: list[Any] = []
        self._vels: list[Any] = []
        self._coms: list[Any] = []
        self._labels: list[str] = []
        self._shapes: list[dict[str, Any]] = []
        self._meshes: list[Any] = []
        self.rotors: list[Rotor] = []
        self.model = model
        self._newton_shapes = None
        """Set by `couple.NewtonCoupling` to the device shape arrays it maintains.

        With a Newton model behind the world, the shapes are Newton's and their poses
        are composed on the device by `lattice.shapes_from_newton` every coupling step.
        The host shape list this class otherwise keeps is then not the truth, and
        `_voxelize` reads this tuple instead. One attribute rather than a subclass: the
        difference between the two cases is five arrays, and a `FluidWorld` with a
        model and one without are the same object in every other respect.
        """
        self._device_bodies_dirty = True
        self._device_shapes_dirty = True
        self.body_pose = None
        self.body_vel = None
        self.body_com = None

        # -- accumulators and diagnostics ---------------------------------------------
        self._wrench = wp.zeros((1, 6), dtype=L.DTYPE, device=self.device)
        self._fallbacks = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._refilled = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._force_field = wp.zeros((1, 1, 1), dtype=wp.vec3, device=self.device)
        self._use_field = 0
        self._rho = None
        self._u = None

        self.steps = 0
        """Coupling steps taken."""
        self.lattice_steps = 0
        self.shifts = 0
        self.seconds = 0.0
        self.fallbacks = 0
        self.refilled = 0
        self.follow = follow
        self.follow_slack = float(follow_slack)
        self._body_force = (0.0, 0.0, 0.0)
        self._free_stream = (0.0, 0.0, 0.0)
        self._rho0 = 1.0
        self._blown_up = False

    # -- geometry in -------------------------------------------------------------------

    def add_body(self, position: Sequence[float] = (0.0, 0.0, 0.0),
                 orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
                 *, com: Sequence[float] | None = None, velocity: Sequence[float] | None = None,
                 label: str = "") -> int:
        """A body the world owns: pose and velocity in SI, nobody refreshes them.

        `orientation` is a quaternion `(x, y, z, w)`, Warp's and Newton's order.
        `com` defaults to `position`, which is what a single-shape probe body wants.
        `velocity` is a six-vector, linear then angular, matching Newton's `body_qd`.
        """
        import warp as wp

        index = len(self._poses)
        self._poses.append(tuple(float(v) for v in position)
                           + tuple(float(v) for v in orientation))
        self._coms.append(tuple(float(v) for v in (com if com is not None else position)))
        self._vels.append(tuple(float(v) for v in (velocity or (0.0,) * 6)))
        self._labels.append(label or f"body_{index}")
        self._device_bodies_dirty = True
        del wp
        return index

    def set_body_state(self, body: int, position: Sequence[float] | None = None,
                       orientation: Sequence[float] | None = None,
                       velocity: Sequence[float] | None = None,
                       com: Sequence[float] | None = None) -> None:
        """Move a world-owned body. SI in, lattice units on the device at the next step."""
        pose = list(self._poses[body])
        if position is not None:
            pose[:3] = [float(v) for v in position]
        if orientation is not None:
            pose[3:] = [float(v) for v in orientation]
        self._poses[body] = tuple(pose)
        if velocity is not None:
            self._vels[body] = tuple(float(v) for v in velocity)
        if com is not None:
            self._coms[body] = tuple(float(v) for v in com)
        self._device_bodies_dirty = True

    def add_shape(self, kind: str | int, *, body: int = -1,
                  position: Sequence[float] = (0.0, 0.0, 0.0),
                  orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
                  scale: Sequence[float] = (1.0, 1.0, 1.0),
                  mesh: Any = None) -> int:
        """A solid, in metres, in the frame of `body` (or in the world if `body < 0`).

        `scale` is the shape's own parameters, in metres, and the convention is the one
        `lattice.py`'s SDFs use and `couple.py` maps Newton onto:

            sphere      (radius, -, -)
            box         (half_x, half_y, half_z)
            capsule     (radius, half_height, -)     along the local z axis
            cylinder    (radius, half_height, -)     along the local z axis
            plane       ignored; the local z = 0 half-space, +z outward
            mesh        (sx, sy, sz) scaling applied to the mesh

        A shape with `body < 0` is scenery: it is a wall and its force is not
        accumulated anywhere. A ground plane is the usual one.
        """
        code = kind if isinstance(kind, int) else {
            v: k for k, v in L.SHAPE_NAMES.items()
        }[str(kind).strip().lower()]
        self._shapes.append({
            "kind": int(code), "body": int(body),
            "position": tuple(float(v) for v in position),
            "orientation": tuple(float(v) for v in orientation),
            "scale": tuple(float(v) for v in scale),
            "mesh": mesh,
        })
        self._device_shapes_dirty = True
        return len(self._shapes) - 1

    def add_rotor(self, body: int, centre: Sequence[float], axis: Sequence[float],
                  radius: float, thickness: float = 0.0) -> int:
        """An actuator disk on `body`, in the body's frame, in metres. Returns its index."""
        norm = math.sqrt(sum(float(v) ** 2 for v in axis))
        if norm <= 0.0:
            raise UnphysicalScaling("a rotor's axis cannot be the zero vector")
        self.rotors.append(Rotor(
            body=int(body),
            centre=tuple(float(v) for v in centre),
            axis=tuple(float(v) / norm for v in axis),
            radius=float(radius),
            thickness=float(thickness) if thickness else 2.0 * self.dx,
        ))
        self._ensure_force_field()
        return len(self.rotors) - 1

    def set_rotor_thrust(self, index: int, thrust: float) -> None:
        """Total thrust in newtons for one rotor, applied from the next `step()`.

        Newton's third law in one place: the fluid gets `-thrust * axis` as a momentum
        source (`lattice.disks_to_force`) and the body gets `+thrust * axis` as part of
        the wrench `couple.py` pushes (`FluidWorld.rotor_reaction`). Neither half is
        hidden in a kernel.
        """
        self.rotors[index].thrust = float(thrust)

    def set_inlet_profile(self, velocity: np.ndarray) -> None:
        """A per-node inflow velocity over the -x face, shape `(ny, nz, 3)`, in m/s.

        The Schafer-Turek channel needs this and nothing else does: its inflow is a
        parabola, and a uniform inlet at the same mean speed is a different benchmark
        with a different answer. `cases.py` builds the array; this only moves it.
        """
        import warp as wp

        array = np.asarray(velocity, dtype=np.float64)
        if array.shape != (self.ny, self.nz, 3):
            raise UnphysicalScaling(
                f"an inlet profile for this lattice is {(self.ny, self.nz, 3)}, "
                f"got {array.shape}"
            )
        lattice = (array * self.scaling.dt / self.scaling.dx).astype(np.float32)
        self._inlet_profile = wp.array(lattice, dtype=wp.vec3, device=self.device)
        self._use_profile = 1

    def set_body_force(self, force_per_volume: Sequence[float]) -> None:
        """A uniform body force in N/m^3, which is how a periodic channel is driven.

        Not gravity. Gravity is deliberately absent from the fluid: in a weakly
        compressible solver a hydrostatic gradient is a real density gradient, and at
        the Mach numbers this runs at it would be a large one. A body in a heavy fluid
        gets its weight and its buoyancy from `couple.py` instead, which is both correct
        and the standard way to do it. `DECISIONS_LIVE.md` D9.
        """
        self._body_force = tuple(float(v) for v in force_per_volume)

    # -- starting state ----------------------------------------------------------------

    def initialise(self, velocity: Sequence[float] = (0.0, 0.0, 0.0),
                   density: float = 1.0) -> None:
        """Every node at equilibrium with a uniform state. `velocity` in m/s."""
        import warp as wp

        self._free_stream = tuple(float(v) for v in velocity)
        self._rho0 = float(density)
        u = wp.vec3(*[self.scaling.velocity_to_lattice(v) for v in self._free_stream])
        wp.launch(L.initialise, dim=self.shape,
                  inputs=[self.f, L.DTYPE(density), u], device=self.device)
        wp.launch(L.initialise, dim=self.shape,
                  inputs=[self.f_next, L.DTYPE(density), u], device=self.device)
        self._sync_bodies()
        self._voxelize()
        # The first coupling step has no previous field to compare against, so nothing
        # is "fresh": copy, so the first `refill` finds no work rather than refilling
        # every node inside every body on step one.
        wp.copy(self.phi_prev, self.phi)
        wp.copy(self.owner_prev, self.owner)

    # -- the step ----------------------------------------------------------------------

    def step(self, substeps: int | None = None) -> dict[str, Any]:
        """Advance the fluid by one coupling step. Returns what the step cost.

        The returned dictionary is what `cases.py` logs and what the agent tool reports:
        the step index, the lattice steps taken, the rate, and the two counters that say
        whether the boundary treatment is in trouble (`fallbacks`, `refilled`).
        """
        import warp as wp

        count = int(self.substeps if substeps is None else substeps)
        started = time.perf_counter()

        self._sync_bodies()
        shifted = self._maybe_shift()
        self._voxelize()
        if self.steps or shifted:
            wp.launch(L.refill, dim=self.shape,
                      inputs=[self.f, self.phi, self.phi_prev, self.owner_prev,
                              self.body_pose, self.body_vel, self.body_com,
                              self._refilled],
                      device=self.device)
        wp.copy(self.phi_prev, self.phi)
        wp.copy(self.owner_prev, self.owner)
        self._apply_rotors()

        self._wrench.zero_()
        omega_plus, omega_minus = self.scaling.rates
        scheme = L.SCHEMES[self.collision]
        smag = L.DTYPE(self.smagorinsky ** 2)
        body_force = wp.vec3(*[self.scaling.body_force_to_lattice(v)
                               for v in self._body_force])
        moving = 0 if self._static else 1

        for _ in range(count):
            wp.launch(L.collide, dim=self.shape,
                      inputs=[self.f, self.phi, self._force_field, body_force,
                              self._use_field, L.DTYPE(omega_plus),
                              L.DTYPE(omega_minus), scheme, smag],
                      device=self.device)
            wp.launch(L.stream, dim=self.shape,
                      inputs=[self.f, self.f_next, self.phi, self.owner,
                              self.body_pose, self.body_vel, self.body_com,
                              self._wrench, self.faces, self.face_velocity,
                              self._fallbacks, moving],
                      device=self.device)
            self.f, self.f_next = self.f_next, self.f
            if self._needs_faces:
                wp.launch(L.apply_faces, dim=self.shape,
                          inputs=[self.f, self.phi, self.faces, self.face_velocity,
                                  self._inlet_profile, self._use_profile],
                          device=self.device)
            self.lattice_steps += 1

        wp.synchronize_device(self.device)
        elapsed = time.perf_counter() - started
        self.seconds += elapsed
        self.steps += 1
        self._rho = None  # the cached macroscopic fields are a step out of date now
        return {
            "step": self.steps,
            "lattice_steps": count,
            "seconds": elapsed,
            "mlups": self.nodes * count / elapsed / 1e6 if elapsed > 0 else 0.0,
            "shifted": shifted,
        }

    @property
    def _static(self) -> bool:
        """Whether any body can move. A wholly static scene skips the wall-velocity
        lookup in `stream`, which is a transform read and a cross product per boundary
        link, and on the cavity is most of the boundary work."""
        if self.model is not None:
            return False
        return not any(any(v) for v in self._vels)

    @property
    def _needs_faces(self) -> bool:
        return any(rule in (L.FACE_INLET, L.FACE_OUTLET) for rule in self.face_rules)

    # -- what the caller reads ---------------------------------------------------------

    def wrench(self, body: int | None = None, *, substeps: int | None = None):
        """The mean fluid wrench over the last coupling step, in SI.

        `(force, torque)` for one body as two 3-tuples in newtons and newton-metres, or
        an `(n, 6)` array for all of them. Divided by the substep count, because
        `stream` accumulated one contribution per lattice step and what the caller wants
        is the mean over the interval -- which is also what Newton wants, since that is
        the interval it is about to integrate over.

        A host read. It is not in the step loop: `couple.py` pushes the wrench into
        Newton with a kernel and never copies it. This exists for reports, for figures
        and for the force histories the validations are graded on.
        """
        count = float(self.substeps if substeps is None else substeps)
        raw = self._wrench.numpy() / max(count, 1.0)
        force = raw[:, :3] * (self.fluid.rho * self.dx ** 4
                              / (self.scaling.dt * self.scaling.dt))
        torque = raw[:, 3:] * (self.fluid.rho * self.dx ** 5
                               / (self.scaling.dt * self.scaling.dt))
        both = np.concatenate([force, torque], axis=1)
        if body is None:
            return both
        return tuple(both[body, :3]), tuple(both[body, 3:])

    def rotor_reaction(self, body: int) -> tuple[float, float, float]:
        """The sum of `thrust * axis` for the rotors on `body`, in world frame, newtons.

        The other half of the actuator disk. `disks_to_force` put `-thrust * axis` into
        the fluid; this is what the body is owed, and `couple.py` adds it to `body_f`
        beside the surface wrench.
        """
        total = np.zeros(3)
        pose = self._poses[body]
        rotation = _quat_matrix(pose[3:])
        for rotor in self.rotors:
            if rotor.body == body:
                total += rotation @ np.asarray(rotor.axis) * rotor.thrust
        return tuple(total)

    def fields(self) -> dict[str, np.ndarray]:
        """Density and velocity over the lattice, in SI, as numpy arrays.

        `rho` is `(nx, ny, nz)` in kg/m^3 and `u` is `(nx, ny, nz, 3)` in m/s, with
        solid nodes set to `nan` so that a figure does not draw the inside of a body as
        stationary fluid. A host read and not cheap: 2 million nodes is 32 MB of
        velocity. Cached until the next `step()`.
        """
        import warp as wp

        if self._rho is None:
            if self._rho_dev is None:
                self._rho_dev = wp.zeros(self.shape, dtype=L.DTYPE, device=self.device)
                self._u_dev = wp.zeros(self.shape, dtype=wp.vec3, device=self.device)
            body_force = wp.vec3(*[self.scaling.body_force_to_lattice(v)
                                   for v in self._body_force])
            wp.launch(L.macroscopic, dim=self.shape,
                      inputs=[self.f, self.phi, self._rho_dev, self._u_dev,
                              self._force_field, body_force, self._use_field],
                      device=self.device)
            solid = self.phi.numpy() < 0.0
            rho = self._rho_dev.numpy().astype(np.float64)
            u = self._u_dev.numpy().astype(np.float64)
            rho[solid] = np.nan
            u[solid] = np.nan
            self._rho = rho * self.fluid.rho
            self._u = u * (self.dx / self.scaling.dt)
        return {"rho": self._rho, "u": self._u, "solid": self.phi.numpy() < 0.0}

    _rho_dev = None
    _u_dev = None

    def momentum(self) -> tuple[float, float, float]:
        """Total fluid momentum, kg m/s, summed over the fluid nodes.

        The quantity the momentum balance is checked on: with every face periodic and
        no body force, a closed lattice must hold it constant to round-off, and any
        drift is the boundary treatment leaking. A host read, a diagnostic, never in the
        step loop. `shifts` has to be zero for this to mean anything -- the moving
        window does not conserve momentum and says so (`lattice.shift`).
        """
        got = self.fields()
        rho, u = got["rho"], got["u"]
        cell = self.dx ** 3
        mom = np.nansum(rho[..., None] * u, axis=(0, 1, 2)) * cell
        return tuple(float(v) for v in mom)

    def stable(self) -> bool:
        """Whether the lattice still holds finite, positive densities.

        Checked by the long-run tests rather than every step: it is a full host read of
        the density field, which at 2 million nodes is 8 MB, and a blow-up does not
        need to be caught within one step of happening. What it catches is the only way
        this solver fails -- a negative or non-finite density, which is what a lattice
        Boltzmann divergence looks like a step or two before every number becomes a nan.
        """
        rho = self.fields()["rho"]
        finite = np.isfinite(rho) | np.isnan(rho)  # nan is a solid node, not a blow-up
        positive = np.nan_to_num(rho, nan=self.fluid.rho) > 0.0
        self._blown_up = not (bool(finite.all()) and bool(positive.all()))
        return not self._blown_up

    def report(self) -> list[str]:
        """The world as a human reads it: the lattice, the scaling, and what it cost."""
        self.fallbacks = int(self._fallbacks.numpy()[0])
        self.refilled = int(self._refilled.numpy()[0])
        faces = "  ".join(f"{name}={L.FACE_NAMES[rule]}"
                          for name, rule in zip(FACE_ORDER, self.face_rules))
        lines = [
            f"  device:     {self.device}  (warp {self.wp.config.version}, "
            f"{L.LAYOUT} layout)",
            f"  lattice:    {self.nx} x {self.ny} x {self.nz} = {self.nodes:,} nodes, "
            f"D3Q19, {self.f.size * 4 * 2 / 2**20:.0f} MiB of populations",
            f"  domain:     {self.domain.size[0]:g} x {self.domain.size[1]:g} x "
            f"{self.domain.size[2]:g} m from {self.domain.origin}",
            f"  faces:      {faces}",
            *self.scaling.report(),
            f"  coupling:   dt {self.dt:.6g} s per step, {self.substeps} lattice "
            f"substeps of {self.scaling.dt:.6g} s"
            f"{'  (one-way: the bodies do not feel the fluid)' if self.one_way else ''}",
        ]
        if self.steps:
            rate = self.nodes * self.lattice_steps / self.seconds / 1e6
            lines.append(
                f"  cost:       {self.steps:,} coupling steps "
                f"({self.lattice_steps:,} lattice steps) in {self.seconds:.1f} s at "
                f"{rate:.1f} MLUPS, {self.seconds / self.steps * 1e3:.2f} ms a step"
            )
            lines.append(
                f"  realtime:   {self.dt * self.steps / self.seconds:.3f}x "
                f"(simulated seconds per wall-clock second)"
            )
        if self.refilled:
            lines.append(f"  refilled:   {self.refilled:,} fresh nodes since the start")
        if self.fallbacks:
            share = self.fallbacks / max(1, self.lattice_steps)
            lines.append(
                f"  NOTE:       {self.fallbacks:,} Bouzidi fallbacks "
                f"({share:.1f} a step): links whose second fluid node was also solid, "
                f"reflected at q = 1/2 instead of interpolated. A body thinner than a "
                f"cell, or a sharp edge; raise the resolution if the forces matter"
            )
        if self.shifts:
            lines.append(
                f"  window:     {self.shifts} shifts, so momentum is not conserved "
                f"across this run by construction"
            )
        return lines

    # -- the machinery ----------------------------------------------------------------

    def _to_lattice(self, point) -> np.ndarray:
        """A world point in metres to lattice coordinates. The half cell lives here.

        Node `i` is at `origin + (i + 0.5) * dx`, so a point's lattice coordinate is
        `(p - origin)/dx - 0.5`. One expression, one place, and `Domain.origin` says
        why. The inverse is `node_coordinates`.
        """
        return (np.asarray(point, dtype=np.float64)
                - np.asarray(self.domain.origin)) / self.dx - 0.5

    def node_coordinates(self, axis: int | None = None):
        """The world coordinates of the lattice nodes, in metres.

        Per axis with `axis=`, or all three as a tuple. What a figure's axes are and
        what a profile is sampled against; the only inverse of `_to_lattice` anything
        needs.
        """
        if axis is None:
            return tuple(self.node_coordinates(a) for a in range(3))
        return (self.domain.origin[axis]
                + (np.arange(self.shape[axis]) + 0.5) * self.dx)

    def _ensure_force_field(self) -> None:
        import warp as wp

        if self._use_field:
            return
        self._force_field = wp.zeros(self.shape, dtype=wp.vec3, device=self.device)
        self._use_field = 1

    def _sync_bodies(self) -> None:
        """Push the host body table to the device, in lattice units, if it has moved."""
        import warp as wp

        if not self._device_bodies_dirty and self.body_pose is not None:
            return
        count = max(1, len(self._poses))
        poses = np.zeros((count, 7), dtype=np.float32)
        poses[:, 6] = 1.0
        vels = np.zeros((count, 6), dtype=np.float32)
        coms = np.zeros((count, 3), dtype=np.float32)
        for index, pose in enumerate(self._poses):
            poses[index, :3] = self._to_lattice(pose[:3])
            poses[index, 3:] = pose[3:]
            coms[index] = self._to_lattice(self._coms[index])
            vels[index, :3] = np.asarray(self._vels[index][:3]) * self.scaling.dt / self.dx
            vels[index, 3:] = np.asarray(self._vels[index][3:]) * self.scaling.dt
        self.body_pose = wp.array(poses, dtype=wp.transform, device=self.device)
        self.body_vel = wp.array(vels, dtype=wp.spatial_vector, device=self.device)
        self.body_com = wp.array(coms, dtype=wp.vec3, device=self.device)
        if self._wrench.shape[0] != count:
            self._wrench = wp.zeros((count, 6), dtype=L.DTYPE, device=self.device)
        self._device_bodies_dirty = False

    def _shape_arrays(self):
        """The shape table on the device, in lattice units, rebuilt when it changes.

        The *poses* are recomposed every coupling step (bodies move) but the types, the
        scales and the mesh handles are not, so they are built once. The pose
        composition is here rather than in `voxelize` because it is one quaternion
        multiply per shape against one per lattice node.
        """
        import warp as wp

        count = max(1, len(self._shapes))
        kinds = np.zeros(count, dtype=np.int32)
        bodies = np.full(count, -1, dtype=np.int32)
        poses = np.zeros((count, 7), dtype=np.float32)
        poses[:, 6] = 1.0
        scales = np.ones((count, 3), dtype=np.float32)
        meshes = np.zeros(count, dtype=np.uint64)

        for index, shape in enumerate(self._shapes):
            kinds[index] = shape["kind"]
            bodies[index] = shape["body"]
            local_p = np.asarray(shape["position"], dtype=np.float64)
            local_q = np.asarray(shape["orientation"], dtype=np.float64)
            if shape["body"] >= 0:
                body_pose = self._poses[shape["body"]]
                body_p = np.asarray(body_pose[:3])
                body_q = np.asarray(body_pose[3:])
                world_p = body_p + _quat_matrix(body_q) @ local_p
                world_q = _quat_multiply(body_q, local_q)
            else:
                world_p, world_q = local_p, local_q
            poses[index, :3] = self._to_lattice(world_p)
            poses[index, 3:] = world_q
            if shape["kind"] == L.SHAPE_MESH:
                scales[index] = np.asarray(shape["scale"]) / self.dx
                meshes[index] = _mesh_id(shape["mesh"], self.dx, self.device,
                                         self._meshes)
            else:
                scales[index] = np.asarray(shape["scale"]) / self.dx
        return (
            wp.array(kinds, dtype=wp.int32, device=self.device),
            wp.array(bodies, dtype=wp.int32, device=self.device),
            wp.array(poses, dtype=wp.transform, device=self.device),
            wp.array(scales, dtype=wp.vec3, device=self.device),
            wp.array(meshes, dtype=wp.uint64, device=self.device),
        )

    def _voxelize(self) -> None:
        import warp as wp

        if self._newton_shapes is not None:
            kinds, bodies, poses, scales, meshes = self._newton_shapes
        elif not self._shapes:
            self.phi.fill_(L.DTYPE(self.far))
            self.owner.fill_(-1)
            return
        else:
            kinds, bodies, poses, scales, meshes = self._shape_arrays()
        wp.launch(L.voxelize, dim=self.shape,
                  inputs=[self.phi, self.owner, kinds, bodies, poses, scales, meshes,
                          L.DTYPE(self.mesh_search), L.DTYPE(self.far)],
                  device=self.device)

    def _apply_rotors(self) -> None:
        import warp as wp

        if not self.rotors:
            return
        self._ensure_force_field()
        wp.launch(L.zero_force, dim=self.shape, inputs=[self._force_field],
                  device=self.device)
        n = len(self.rotors)
        centres = np.zeros((n, 3), dtype=np.float32)
        axes = np.zeros((n, 3), dtype=np.float32)
        radii = np.zeros(n, dtype=np.float32)
        thick = np.zeros(n, dtype=np.float32)
        thrust = np.zeros(n, dtype=np.float32)
        for index, rotor in enumerate(self.rotors):
            if rotor.body >= 0:
                pose = self._poses[rotor.body]
                rotation = _quat_matrix(pose[3:])
                world = np.asarray(pose[:3]) + rotation @ np.asarray(rotor.centre)
                axis = rotation @ np.asarray(rotor.axis)
            else:
                world = np.asarray(rotor.centre)
                axis = np.asarray(rotor.axis)
            centres[index] = self._to_lattice(world)
            axes[index] = axis / max(np.linalg.norm(axis), 1e-12)
            radii[index] = rotor.radius / self.dx
            thick[index] = max(rotor.thickness / self.dx, 1.0)
            thrust[index] = self.scaling.force_to_lattice(rotor.thrust)
        wp.launch(
            L.disks_to_force, dim=self.shape,
            inputs=[self._force_field, self.phi,
                    wp.array(centres, dtype=wp.vec3, device=self.device),
                    wp.array(axes, dtype=wp.vec3, device=self.device),
                    wp.array(radii, dtype=L.DTYPE, device=self.device),
                    wp.array(thick, dtype=L.DTYPE, device=self.device),
                    wp.array(thrust, dtype=L.DTYPE, device=self.device)],
            device=self.device)

    def _maybe_shift(self) -> bool:
        """Re-centre the window on the followed body, if it has drifted far enough."""
        import warp as wp

        if self.follow is None:
            return False
        origin = np.asarray(self.domain.origin)
        size = np.asarray(self.domain.size)
        centre = origin + size / 2.0
        position = np.asarray(self._poses[self.follow][:3])
        drift = position - centre
        allow = self.follow_slack * size
        cells = np.where(np.abs(drift) > allow, np.round(drift / self.dx), 0.0)
        if not cells.any():
            return False
        u = wp.vec3(*[self.scaling.velocity_to_lattice(v) for v in self._free_stream])
        wp.launch(L.shift, dim=self.shape,
                  inputs=[self.f, self.f_next,
                          wp.vec3i(*[int(v) for v in cells]),
                          L.DTYPE(self._rho0), u],
                  device=self.device)
        self.f, self.f_next = self.f_next, self.f
        self.domain.origin = tuple(origin + cells * self.dx)
        self.shifts += 1
        self._device_bodies_dirty = True
        self._sync_bodies()
        return True


# -- quaternion arithmetic, in the (x, y, z, w) order Warp and Newton both use ---------
#
# Four lines rather than a dependency. scipy's Rotation would do it and scipy is not in
# this extra; numpy-quaternion is a dependency for sixteen multiplications.


def _quat_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _mesh_id(mesh: Any, dx: float, device: str, cache: list[Any]) -> int:
    """A `wp.Mesh` id for a mesh in lattice units, built once and kept alive.

    Accepts a `wp.Mesh` (used as is), a `newton.Mesh` (its `.vertices` and `.indices`),
    or a `(vertices, indices)` pair. The vertices are divided by `dx`, because
    `voxelize` works in lattice coordinates and scaling the query point instead would
    make a non-uniform `scale` wrong.

    The cache is not an optimisation: a `wp.Mesh` whose Python object is garbage
    collected frees its device memory, and the id in the shape array would then point at
    nothing. Holding a reference is the only way to keep it valid.
    """
    import warp as wp

    if isinstance(mesh, wp.Mesh):
        cache.append(mesh)
        return mesh.id
    vertices = getattr(mesh, "vertices", None)
    indices = getattr(mesh, "indices", None)
    if vertices is None and isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        vertices, indices = mesh
    if vertices is None:
        raise UnphysicalScaling(
            "a mesh shape needs a wp.Mesh, a newton.Mesh, or a (vertices, indices) "
            f"pair; got {type(mesh).__name__}"
        )
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3) / float(dx)
    faces = np.asarray(indices, dtype=np.int32).reshape(-1)
    built = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    cache.append(built)
    return built.id
