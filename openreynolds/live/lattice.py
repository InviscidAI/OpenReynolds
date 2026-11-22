"""D3Q19 on NVIDIA Warp: the velocity set, and every kernel the step loop launches.

This is the whole numerical core. Nothing above it (`world.py`, `couple.py`,
`cases.py`) contains a floating-point operation on a lattice; everything here runs on
whichever Warp device the arrays live on, and the step loop below `FluidWorld.step`
makes no host round trip.

## Why the whole step is two kernels and not one

A fused pull-stream-collide is the fastest shape for an LBM step and it cannot carry an
interpolated boundary. Bouzidi needs **post-collision** populations at the boundary node
*and* at its next neighbour away from the wall. A fused kernel at node x has only
pre-collision values of its neighbours, so a boundary node would have to recollide its
neighbour to find out what to reflect. So:

    collide(f)              in place, every fluid node, nothing streams
    stream(f -> f_new)      pull, with Bouzidi where the source node is solid,
                            and the momentum exchange in the same pass

The momentum exchange rides in `stream` for the same reason: it is a sum over exactly
the links the boundary treatment already visited, and a separate kernel would walk the
whole lattice again to find them.

## Why the solid is one float per node

`phi` is the signed distance from a node to the nearest solid surface, in lattice units,
negative inside. `owner` is which rigid body that surface belongs to. Everything else
about the boundary is derived in the kernel that needs it:

    solid       phi < 0
    q           phi[x] / (phi[x] - phi[x+c]),  the fraction of the link in the fluid
    u_wall      v_body + omega x (x_w - com),  read from the body arrays via `owner`
    fresh       phi >= 0 now and phi < 0 last coupling step

The alternative -- a per-link `q` array and a list of boundary links -- is what most LBM
codes do, and it is the wrong shape here for two measured reasons. A `q` array is
another 19 floats a node: 160 MB at 128^3, against 8 bytes a node for `phi` and `owner`
together. And a boundary *list* has to be rebuilt every time the robot moves, which is
every coupling step, which means a device-side compaction every one to five
milliseconds. Recomputing `phi` instead is one kernel over the lattice with no
compaction, no host round trip and no allocation.

## Conventions, which are the ones that go wrong silently

* `f` is `(Q, nx, ny, nz)` or `(nx, ny, nz, Q)`, chosen for the device and read in
  exactly two places (`pop` and `set_pop`). `_preferred_layout` has the measurement that
  decided it: on one CPU core the node-major layout is three times faster and on a GPU
  it is the direction-major one, by factors that large.
* Node `(i, j, k)` is at world position `origin + (i, j, k) * dx`. Nodes, not cell
  centres: a wall put on a node is `phi = 0` there, and the halfway bounce-back the
  `trt` scheme is built around puts the no-slip plane half a link *outside* the last
  fluid node. `cases.py` is where that half cell is written down, once, for the same
  reason `toolbox/lbm.py::centres` exists.
* `C[:, i]` is direction `i` as three integers; `OPP[i]` is the direction `i` reflects
  into; `MIRROR_x/y/z[i]` is `i` with that component negated, which is specular
  reflection off a free-slip face.
* Streaming is a **pull**: `f_new[i][x] = f_post[i][x - C[:, i]]`.
* Lattice units everywhere. `units.Scaling` is the only thing that knows about metres.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import warp as wp

from .units import CS2, LAMBDA  # noqa: F401  (LAMBDA is re-exported for the report)

__all__ = [
    "Q", "D", "C_NP", "W_NP", "OPP_NP", "DTYPE",
    "FACE_PERIODIC", "FACE_WALL", "FACE_SLIP", "FACE_INLET", "FACE_OUTLET", "FACE_NAMES",
    "SHAPE_SPHERE", "SHAPE_BOX", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE",
    "SHAPE_MESH", "SHAPE_NAMES",
    "SCHEME_TRT", "SCHEME_BGK", "SCHEME_REGULARIZED", "SCHEMES",
    "collide", "stream", "apply_faces", "voxelize", "refill", "macroscopic",
    "initialise", "disks_to_force", "wrench_to_newton", "zero_force", "shift",
    "volumes", "poses_from_newton", "shapes_from_newton",
    "LAYOUT", "SOA", "pop", "set_pop", "lattice_shape",
]

def _preferred_layout() -> str:
    """`soa` or `aos`, chosen for the device that is actually here, and why it matters.

    The populations are a four-dimensional array and there are two ways to index it:

        soa     pop(f, q, i, j, k)   direction-major, the three spatial indices adjacent
        aos     pop(f, i, j, k, q)   node-major, one node's 19 populations adjacent

    Both are correct and they differ only in the order of four subscripts. They do not
    differ a little in speed. Measured on this machine (one core, a streaming kernel
    over 221,184 nodes, best of three):

        layout  max_unroll 16   max_unroll 20
        soa          7.6 MLUPS      10.4 MLUPS
        aos          7.9 MLUPS      29.8 MLUPS

    Two effects, and the table separates them. The first is the unroll limit: Warp
    unrolls a `range` loop whose bound is a constant only up to `max_unroll`, which
    defaults to 16, and `Q` is 19 -- so the direction loop in every kernel below was a
    *dynamic* loop with a counter, a bounds check and no chance of vectorisation.
    Raising the limit past 19 unrolls it. The second effect only appears once it is
    unrolled: with the loop unrolled, `aos` reads one node's populations from two cache
    lines and `soa` reads them from nineteen, and the compiler can only see that when
    the nineteen reads are in front of it.

    So why is `soa` the default on a GPU? Because the same argument runs the other way
    there. A CUDA warp is 32 threads at 32 adjacent nodes, and under `soa` their reads
    of `f[q]` are 32 adjacent floats -- one coalesced transaction. Under `aos` they are
    32 reads 76 bytes apart, which is 32 transactions. The layout that halves the work
    on one CPU core multiplies it by up to sixteen on a GPU.

    Hence: chosen from the device, overridable by `OPENREYNOLDS_LIVE_LAYOUT`, and
    reported by `FluidWorld.report` so that no measurement in `PROGRESS_LIVE.md` is
    ambiguous about which one produced it. It is a module-level constant and not a
    runtime flag because it is baked into the generated code: a branch per array access
    would cost more than the layout saves.

    **The unroll table above is warp-lang 1.19.0's and does not transfer to 1.17.0.**
    Isaac Lab 3.0.0 pins 1.17.0, so the whole table was re-measured there (the numbers
    and the commands are in `docs/live-version-matrix.md`). `max_unroll` is honoured by
    both -- it is read, applied, and reported back by `get_module_options()` on 1.17.0
    exactly as on 1.19.0 -- but on 1.17.0 raising it from 16 to 24 measured *slower* on
    an isolated Q-loop kernel, 43.95 MLUPS against 38.84, where on 1.19.0 it measured
    faster, 49.71 against 55.29. On the real cavity 1.17.0 runs the step loop at
    1.11 MLUPS against 1.19.0's 3.51, a factor of 3.2, and that gap is not the unroll
    option and not launch overhead (8.06 against 8.52 microseconds a launch, which is
    the same number): it is 1.17.0's CPU code generation for the two large kernels.
    The value stays 24 because the answers are bit-identical either way and 1.19.0 and
    every GPU are what the step rate is for. A run on 1.17.0 gets the honest number
    from `FluidWorld.report`, next to the Warp version that produced it.

    One side effect of this function worth knowing, because something else depends on
    it: `wp.get_cuda_device_count()` creates Warp's runtime. So importing this module
    initialises Warp, and `wp.from_torch` / `wp.to_torch` -- which on **both** 1.17.0
    and 1.19.0 raise `AttributeError: 'NoneType' object has no attribute 'cpu_device'`
    if they are the first Warp call in a process, the runtime being lazy and
    `device_from_torch` not forcing it -- work from the moment this package is
    imported. That is the order an Isaac Lab user hits first and it is why the torch
    interop in `docs/live-version-matrix.md` is recorded as holding rather than as a
    trap they have to know about.
    """
    asked = os.environ.get("OPENREYNOLDS_LIVE_LAYOUT", "").strip().lower()
    if asked in ("soa", "aos"):
        return asked
    try:
        return "soa" if wp.get_cuda_device_count() > 0 else "aos"
    except Exception:  # pragma: no cover - a Warp too old to be asked
        return "aos"


LAYOUT = _preferred_layout()
SOA = LAYOUT == "soa"

wp.set_module_options({
    # Past `Q`, so the direction loop in every kernel here unrolls. See
    # `_preferred_layout` for the measurement: on this CPU it is the difference between
    # 7.9 and 29.8 MLUPS, and it is the single largest factor in the step rate.
    #
    # Both keys are accepted by warp-lang 1.17.0 and 1.19.0 -- measured, because Isaac
    # Lab 3.0.0 pins 1.17.0 and this call is the first thing the module does, so a
    # rejected key would be an import error rather than a slow kernel. 1.17.0 returns
    # the same `get_module_options()` dictionary with the same defaults, and neither
    # version validates the keys at all: an option name that does not exist is accepted
    # silently by both. That is why there is no try/except here and no per-key probing;
    # there is nothing to catch. `tests/test_live_versions.py` holds that fact, so if a
    # future Warp starts rejecting unknown keys the test says so before a user does.
    "max_unroll": 24,
    # Nothing in this extension differentiates through a lattice. The adjoint of every
    # kernel below is generated otherwise, which doubles the compile and the code size
    # for something no caller can reach.
    "enable_backward": False,
})

DTYPE = wp.float32
"""Single precision throughout.

A lattice Boltzmann population sits near its weight -- `f[0]` near 1/3 -- and the
interesting part of it is the non-equilibrium difference, which at the Mach numbers this
solver is held to (`units.MACH_MAX`) is five or six orders of magnitude smaller. That is
the classic argument for double precision, and it is why `rho` is carried as a deviation
from 1 in some codes. It is not done here: the validations below measure what single
precision actually costs on these cases, and `benchmarks/live/` reports it. Double
precision doubles the memory and halves the bandwidth-bound step rate, which is the
whole product.
"""

D = 3
Q = 19
"""D3Q19 only.

D3Q27 is "optional" in the brief and is not shipped; `DECISIONS_LIVE.md` D8 says why,
and the short version is that it is a second instantiation of every kernel below for 42%
more memory and no gain at the order the second-order equilibrium is accurate to.
"""

# The velocity set, built rather than typed: every ci with |ci|_1 <= 2, which is the 19.
# Built in the same order XLB's `d3q19.py` builds it (itertools.product over 0, -1, 1),
# so that a field written by one can be read by the other without a permutation.
_C = np.array(
    [ci for ci in itertools.product([0, -1, 1], repeat=3) if np.sum(np.abs(ci)) <= 2],
    dtype=np.int32,
).T  # (3, 19)
assert _C.shape == (D, Q)

_W = np.zeros(Q, dtype=np.float64)
for _i in range(Q):
    _order = int(np.sum(np.abs(_C[:, _i])))
    _W[_i] = (1.0 / 3.0, 1.0 / 18.0, 1.0 / 36.0)[_order]
assert abs(_W.sum() - 1.0) < 1e-14, "the weights are a quadrature and must sum to 1"

_OPP = np.zeros(Q, dtype=np.int32)
for _i in range(Q):
    _match = np.flatnonzero(np.all(_C == -_C[:, _i][:, None], axis=0))
    assert len(_match) == 1, "every direction has exactly one opposite"
    _OPP[_i] = _match[0]

_MIRROR = np.zeros((D, Q), dtype=np.int32)
for _a in range(D):
    for _i in range(Q):
        _want = _C[:, _i].copy()
        _want[_a] = -_want[_a]
        _match = np.flatnonzero(np.all(_C == _want[:, None], axis=0))
        assert len(_match) == 1
        _MIRROR[_a, _i] = _match[0]

C_NP = _C
W_NP = _W
OPP_NP = _OPP
MIRROR_NP = _MIRROR
"""The velocity set as numpy, for the host side: `cases.py` uses it to build equilibria
and `tests/test_live_lattice.py` checks the moments it is supposed to satisfy."""

# -- the same tables as Warp compile-time constants ------------------------------------
#
# The idiom is XLB's (`xlb/velocity_set/velocity_set.py::_init_warp_properties`): a
# `wp.constant` of a fixed-shape matrix or vector type is baked into the generated code,
# so the inner loops below index a register file rather than global memory.

_mat_c = wp.types.matrix(shape=(D, Q), dtype=wp.int32)
_vec_q = wp.types.vector(length=Q, dtype=DTYPE)
_vec_qi = wp.types.vector(length=Q, dtype=wp.int32)

C = wp.constant(_mat_c(_C))
W = wp.constant(_vec_q(_W.astype(np.float32)))
OPP = wp.constant(_vec_qi(_OPP))
MIRROR_X = wp.constant(_vec_qi(_MIRROR[0]))
MIRROR_Y = wp.constant(_vec_qi(_MIRROR[1]))
MIRROR_Z = wp.constant(_vec_qi(_MIRROR[2]))

INV_CS2 = wp.constant(DTYPE(1.0 / CS2))
CS2_W = wp.constant(DTYPE(CS2))

# -- what a domain face does -----------------------------------------------------------

FACE_PERIODIC = 0
FACE_WALL = 1
FACE_SLIP = 2
FACE_INLET = 3
FACE_OUTLET = 4
FACE_NAMES = {
    FACE_PERIODIC: "periodic",
    FACE_WALL: "wall",
    FACE_SLIP: "slip",
    FACE_INLET: "inlet",
    FACE_OUTLET: "outlet",
}
"""Faces are indexed `2 * axis + side`, side 0 for the low face and 1 for the high one,
so face 0 is -x, 1 is +x, 2 is -y, and so on. `wall` carries a velocity (that is how the
lid-driven cavity's lid is a face and not a body), `slip` is specular reflection, and
`outlet` is first-order extrapolation of the populations from one cell inside -- the
"do nothing" outflow, which is what XLB calls it too and is first-order accurate and
unconditionally stable."""

# -- what a solid is -------------------------------------------------------------------

SHAPE_SPHERE = 0
SHAPE_BOX = 1
SHAPE_CAPSULE = 2
SHAPE_CYLINDER = 3
SHAPE_PLANE = 4
SHAPE_MESH = 5
SHAPE_NAMES = {
    SHAPE_SPHERE: "sphere", SHAPE_BOX: "box", SHAPE_CAPSULE: "capsule",
    SHAPE_CYLINDER: "cylinder", SHAPE_PLANE: "plane", SHAPE_MESH: "mesh",
}
"""This extension's own shape codes, deliberately not Newton's `GeoType` integers.

`couple.py` maps Newton's enum to these **by name**, so a renumbering of `GeoType` in a
future Newton release changes one dictionary there and nothing here. Newton's integers
are not part of its documented API and this file should not depend on them."""

SCHEME_TRT = 0
SCHEME_BGK = 1
SCHEME_REGULARIZED = 2
SCHEMES = {"trt": SCHEME_TRT, "bgk": SCHEME_BGK, "regularized": SCHEME_REGULARIZED}


# -- the one place the layout is read ---------------------------------------------------


@wp.func
def pop(f: wp.array4d(dtype=DTYPE), q: wp.int32, i: wp.int32, j: wp.int32,
        k: wp.int32) -> DTYPE:
    """One population, whichever way round the array is. See `_preferred_layout`.

    `SOA` is a module-level Python bool, so Warp bakes it in at code generation and the
    branch below does not exist in the compiled kernel. That is the whole reason the
    layout is a module constant rather than an argument: a branch per array access,
    nineteen times a node, would cost more than the layout saves.
    """
    if SOA:
        return f[q, i, j, k]
    return f[i, j, k, q]


@wp.func
def set_pop(f: wp.array4d(dtype=DTYPE), q: wp.int32, i: wp.int32, j: wp.int32,
            k: wp.int32, value: DTYPE):
    """Write one population. The companion to `pop`, and the only other reader of `SOA`."""
    if SOA:
        f[q, i, j, k] = value
    else:
        f[i, j, k, q] = value


def lattice_shape(shape) -> tuple[int, ...]:
    """The allocation shape for a population array over a lattice of `shape`.

    `world.py` calls this rather than writing `(Q, *shape)`, so that the two places that
    would otherwise have to agree about the layout are one place.
    """
    nx, ny, nz = (int(v) for v in shape)
    return (Q, nx, ny, nz) if SOA else (nx, ny, nz, Q)


# -- small device functions ------------------------------------------------------------


@wp.func
def _equilibrium(i: wp.int32, rho: DTYPE, u: wp.vec3) -> DTYPE:
    """The second-order Maxwellian, which is what every collision here relaxes towards.

        f_i^eq = w_i rho (1 + (c_i.u)/c_s^2 + (c_i.u)^2/(2 c_s^4) - u.u/(2 c_s^2))
    """
    cu = (DTYPE(C[0, i]) * u[0] + DTYPE(C[1, i]) * u[1] + DTYPE(C[2, i]) * u[2]) * INV_CS2
    usq = wp.dot(u, u) * INV_CS2
    return W[i] * rho * (DTYPE(1.0) + cu + DTYPE(0.5) * cu * cu - DTYPE(0.5) * usq)


@wp.func
def _guo_source(i: wp.int32, u: wp.vec3, force: wp.vec3) -> DTYPE:
    """Guo's forcing term, before the `(1 - omega/2)` prefactor the caller applies.

        S_i = w_i [ (c_i.F)/c_s^2 + (c_i.u)(c_i.F)/c_s^4 - (u.F)/c_s^2 ]

    Guo, Zheng and Shi, Phys. Rev. E 65, 046308 (2002). The half-force has to already be
    in the `u` the equilibrium was built from, which is the caller's job and is done in
    `collide`; without it the force is first-order accurate instead of second.
    """
    cf = DTYPE(C[0, i]) * force[0] + DTYPE(C[1, i]) * force[1] + DTYPE(C[2, i]) * force[2]
    cu = DTYPE(C[0, i]) * u[0] + DTYPE(C[1, i]) * u[1] + DTYPE(C[2, i]) * u[2]
    return W[i] * (cf * INV_CS2 + cu * cf * INV_CS2 * INV_CS2 - wp.dot(u, force) * INV_CS2)


@wp.func
def _in_bounds(x: wp.int32, n: wp.int32) -> bool:
    return x >= 0 and x < n


# -- 1. collide ------------------------------------------------------------------------


@wp.kernel
def collide(
    f: wp.array4d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    force_field: wp.array3d(dtype=wp.vec3),
    body_force: wp.vec3,
    use_field: wp.int32,
    omega_plus: DTYPE,
    omega_minus: DTYPE,
    scheme: wp.int32,
    smagorinsky: DTYPE,
):
    """Relax every fluid node towards equilibrium, in place. Nothing streams.

    Three collision operators behind one `scheme`, and the reason there is more than one
    is in `DECISIONS_LIVE.md` D3:

    * `SCHEME_TRT` -- two relaxation times, the parts of `f - f^eq` even and odd under
      `c -> -c` relaxed at `omega_plus` and `omega_minus`. With
      `(tau_+ - 1/2)(tau_- - 1/2) = 3/16` a straight no-slip wall sits exactly on the
      link midpoint at any viscosity; `benchmarks/lbm/` grades that property for the
      two-dimensional version of this scheme at three viscosities.
    * `SCHEME_BGK` -- the same code with both rates equal. Here so the contrast above
      stays measurable rather than asserted.
    * `SCHEME_REGULARIZED` -- the non-equilibrium part of every population is thrown
      away and rebuilt from the deviatoric second moment alone,
      `f_i^neq = w_i Q_i : Pi^neq / (2 c_s^4)` with `Q_i = c_i c_i - c_s^2 I`, before
      being relaxed. Everything the second moment does not carry (the ghost modes) is
      annihilated every step, which is what keeps an under-resolved high-Reynolds run
      from diverging. It is even in `c` by construction, so there is no odd part left
      for a second rate: `omega_minus` is ignored and `units.rates` says so.

    Smagorinsky, when `smagorinsky > 0`, is the eddy viscosity `nu_t = (C_s Delta)^2 |S|`
    with the filter width `Delta` one lattice unit. The strain rate comes from the same
    `Pi^neq` the regularisation already forms, through
    `S_ab = -Pi^neq_ab / (2 rho c_s^2 tau)`, and substituting that into
    `tau_eff = tau_0 + nu_t/c_s^2` and solving the quadratic gives the closed form used
    below,

        tau_eff = (tau_0 + sqrt(tau_0^2 + 18 sqrt(2) C_s^2 |Pi^neq| / rho)) / 2

    with `|Pi^neq| = sqrt(Pi_ab Pi_ab)` and `c_s^2 = 1/3` giving the 18. The form is
    attributed to Hou, Sterling, Chen and Doolen (1996); it is written out here rather
    than cited as evidence because the derivation above is the whole of it, and this
    repository has paid before for quoting a reference it had not read.

    `smagorinsky` is `C_s^2`, not `C_s`: the square is what the formula uses and taking
    it here would be one square root per node per step for nothing.
    """
    i, j, k = wp.tid()
    if phi[i, j, k] < DTYPE(0.0):
        return  # a solid node does not collide; `stream` reflects what arrives at it

    fl = _vec_q()
    rho = DTYPE(0.0)
    mom = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for q in range(Q):
        value = pop(f, q, i, j, k)
        fl[q] = value
        rho += value
        mom += wp.vec3(DTYPE(C[0, q]), DTYPE(C[1, q]), DTYPE(C[2, q])) * value

    force = body_force
    if use_field != 0:
        force = force + force_field[i, j, k]

    # Guo: the body force puts half a step of momentum into the velocity the equilibrium
    # is built from. `toolbox/lbm.py` makes the same move for the same reason.
    inv_rho = DTYPE(1.0) / rho
    u = (mom + force * DTYPE(0.5)) * inv_rho

    feq = _vec_q()
    for q in range(Q):
        feq[q] = _equilibrium(q, rho, u)

    # Pi^neq, the second moment of the non-equilibrium part. Six independent components;
    # needed by the regularisation and by Smagorinsky, and cheap enough that it is
    # always computed -- the alternative is two kernels and a second pass over `f`.
    pxx = DTYPE(0.0)
    pyy = DTYPE(0.0)
    pzz = DTYPE(0.0)
    pxy = DTYPE(0.0)
    pxz = DTYPE(0.0)
    pyz = DTYPE(0.0)
    for q in range(Q):
        neq = fl[q] - feq[q]
        cx = DTYPE(C[0, q])
        cy = DTYPE(C[1, q])
        cz = DTYPE(C[2, q])
        pxx += neq * cx * cx
        pyy += neq * cy * cy
        pzz += neq * cz * cz
        pxy += neq * cx * cy
        pxz += neq * cx * cz
        pyz += neq * cy * cz

    op = omega_plus
    om = omega_minus
    if smagorinsky > DTYPE(0.0):
        # The deviatoric part: the trace of Pi^neq is a bulk mode and is not a strain.
        trace = (pxx + pyy + pzz) / DTYPE(3.0)
        dxx = pxx - trace
        dyy = pyy - trace
        dzz = pzz - trace
        norm = wp.sqrt(
            dxx * dxx + dyy * dyy + dzz * dzz
            + DTYPE(2.0) * (pxy * pxy + pxz * pxz + pyz * pyz)
        )
        tau0 = DTYPE(1.0) / op
        tau_eff = DTYPE(0.5) * (
            tau0 + wp.sqrt(tau0 * tau0
                           + DTYPE(18.0) * DTYPE(1.4142135623730951)
                           * smagorinsky * norm * inv_rho)
        )
        op = DTYPE(1.0) / tau_eff
        if scheme == SCHEME_TRT:
            # The odd rate follows the even one through the magic parameter, so the wall
            # stays on the link midpoint as the eddy viscosity moves. Doing anything
            # else here would make the wall position depend on the local strain rate.
            om = DTYPE(1.0) / (DTYPE(0.5) + DTYPE(LAMBDA) / (tau_eff - DTYPE(0.5)))

    if scheme == SCHEME_REGULARIZED:
        # Rebuild f^neq from Pi^neq alone, then relax it. `cc` below is Q_i : Pi^neq
        # with Q_i = c_i c_i - c_s^2 I; the c_s^2 I part contracts to the trace.
        trace = pxx + pyy + pzz
        half_inv4 = DTYPE(0.5) * INV_CS2 * INV_CS2
        source_scale = DTYPE(1.0) - DTYPE(0.5) * op
        for q in range(Q):
            cx = DTYPE(C[0, q])
            cy = DTYPE(C[1, q])
            cz = DTYPE(C[2, q])
            cc = (cx * cx * pxx + cy * cy * pyy + cz * cz * pzz
                  + DTYPE(2.0) * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz)
                  - CS2_W * trace)
            regular = W[q] * cc * half_inv4
            set_pop(f, q, i, j, k,
                    feq[q] + (DTYPE(1.0) - op) * regular
                    + source_scale * _guo_source(q, u, force))
        return

    # TRT (and BGK, which is TRT with both rates equal), written over opposite pairs.
    # The rest population is its own even part and takes the even rate.
    set_pop(f, 0, i, j, k,
            fl[0] - op * (fl[0] - feq[0])
            + (DTYPE(1.0) - DTYPE(0.5) * op) * _guo_source(0, u, force))
    for q in range(1, Q):
        p = OPP[q]
        if wp.int32(q) < p:  # each pair handled once, from its lower index
            di = fl[q] - feq[q]
            dj = fl[p] - feq[p]
            even = DTYPE(0.5) * (di + dj)
            odd = DTYPE(0.5) * (di - dj)
            # Guo's source split by the same parity: the (c.F)/c_s^2 term is odd in c
            # and takes the odd rate, the rest is even and takes the even one. Without
            # the split this is the BGK source bolted onto a TRT collision.
            si = _guo_source(q, u, force)
            sj = _guo_source(p, u, force)
            s_even = DTYPE(0.5) * (si + sj) * (DTYPE(1.0) - DTYPE(0.5) * op)
            s_odd = DTYPE(0.5) * (si - sj) * (DTYPE(1.0) - DTYPE(0.5) * om)
            set_pop(f, q, i, j, k,
                    fl[q] - (op * even + om * odd) + s_even + s_odd)
            set_pop(f, p, i, j, k,
                    fl[p] - (op * even - om * odd) + s_even - s_odd)


# -- 2. stream, reflect, and weigh the links -------------------------------------------


@wp.func
def _wall_velocity(
    body: wp.int32,
    point: wp.vec3,
    body_pose: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    inv_dx: DTYPE,
) -> wp.vec3:
    """The surface velocity at a lattice point, in lattice units per step.

    `point` is in lattice coordinates. `body_vel` is Newton's `body_qd`: linear velocity
    of the centre of mass in world frame in the first three components, angular velocity
    in world frame in the last three (the convention is in
    `newton._src.sim.state.State.body_qd`'s own docstring). `body_pose` and `body_com`
    are already scaled into lattice coordinates by `couple.py`, and `inv_dx` is absent
    from the arithmetic for that reason -- it is carried only to convert the velocity,
    which arrives in lattice units already, so this is a cross-product and nothing else.
    """
    if body < 0:
        return wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    vel = body_vel[body]
    linear = wp.vec3(vel[0], vel[1], vel[2])
    angular = wp.vec3(vel[3], vel[4], vel[5])
    arm = point - body_com[body]
    return linear + wp.cross(angular, arm)


@wp.kernel
def stream(
    f_post: wp.array4d(dtype=DTYPE),
    f_new: wp.array4d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    owner: wp.array3d(dtype=wp.int32),
    body_pose: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    wrench: wp.array2d(dtype=DTYPE),
    faces: wp.array(dtype=wp.int32),
    face_velocity: wp.array(dtype=wp.vec3),
    fallbacks: wp.array(dtype=wp.int32),
    moving: wp.int32,
):
    """Pull the post-collision lattice into `f_new`, reflecting where it meets a solid.

    For node `x` and direction `i` the source is `s = x - c_i`. Three cases:

    * `s` is fluid and inside the domain: a copy, which is the overwhelming majority of
      the work and is why this kernel is bandwidth-bound.
    * `s` is outside the domain: the face's own rule (`faces`), which for a `wall` is
      halfway bounce-back with that face's velocity -- so a lid-driven cavity's lid is
      a face and needs no body at all.
    * `s` is solid: **Bouzidi interpolated bounce-back**, and the momentum exchange.

    The Bouzidi arithmetic, with `j = OPP[i]` the direction pointing from `x` into the
    solid (so `x + c_j` is `s`) and `q` the fraction of that link lying in the fluid:

        q < 1/2   f_i(x) = 2q f_j*(x) + (1 - 2q) f_j*(x + c_i)
        q >= 1/2  f_i(x) = f_j*(x)/(2q) + (2q - 1)/(2q) f_i*(x)

    Both forms read only `x` and its two neighbours along the link, and both reduce at
    `q = 1/2` to halfway bounce-back, which is the consistency check worth remembering:
    at `q = 1/2` the first gives `f_j*(x)` and the second gives `f_j*(x)` too.

    A moving wall adds `-2 w_j (c_j . u_wall) / c_s^2` to the reflected population,
    scaled by `1` in the first branch and `1/(2q)`'s partner `1/q` in the second so that
    the two still agree at `q = 1/2`. That is the Ladd term, and it is the same term
    `toolbox/lbm.py` writes as `6 w_j (c_j . u_wall)` -- `6 = 2/c_s^2` -- with the sign
    flipped because it adds into direction `j` there and into `i` here.

    **The fallback, counted rather than hidden.** The `q < 1/2` branch reads `x + c_i`,
    the next fluid node away from the wall. For a body thinner than a lattice cell, or
    at a sharp edge, that node is solid too and there is nothing to interpolate from.
    The scheme then degrades to halfway bounce-back at `q = 1/2` and `fallbacks` is
    incremented. `FluidWorld.step` reports the count, because a body that is mostly
    sub-cell is a resolution problem the caller needs to be told about and not a quiet
    loss of order.

    **The momentum exchange** is the classical form: the momentum delivered to the solid
    along one link is `c_j (f_j*(x) + f_i(x, t+1))` -- what arrived, minus what left,
    and `c_i = -c_j` supplies the minus. Summed into `wrench[body]` with
    `wp.atomic_add` on plain floats rather than on a `spatial_vector`, because a
    six-float accumulator is guaranteed to have an atomic and a vector type is not.
    The torque arm is to the cut point `x + q c_j`, not to the lattice node: using the
    node instead is an error of up to half a cell in every moment arm, which is a
    systematic torque on a rotating body rather than noise.
    """
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]

    phi_here = phi[i, j, k]
    if phi_here < DTYPE(0.0):
        # A solid node holds no fluid. Its populations are never read by a fluid node
        # (Bouzidi reads only fluid nodes), and `refill` will build them from scratch if
        # it becomes fluid, so leaving them is correct and writing them is bandwidth.
        return

    # `continue` is avoided throughout: `range(Q)` with Q = 19 is past Warp's unroll
    # limit, so this is a dynamic loop, and the branches are written as nested ifs with
    # a `handled` flag instead. Every variable the loop mutates is declared with its
    # type for the same reason -- Warp refuses to mutate an inferred constant inside a
    # dynamic loop, and says so.
    for d in range(Q):
        si = int(i - C[0, d])
        sj = int(j - C[1, d])
        sk = int(k - C[2, d])

        # -- outside the domain: the face's rule --------------------------------------
        #
        # **Every axis is examined, not the first one that is out of range.** A diagonal
        # link from an edge or a corner node leaves the domain on two axes at once --
        # direction (-1, -1, 0) from node (0, 0, k) is the common case -- and the first
        # version of this kernel used `if/elif` here, applied only the x rule, and then
        # fell through to read `phi[si, sj, sk]` with `sj` still -1. Two things were
        # wrong with that: the wall rule the y face asked for was never applied, and the
        # read was out of bounds (Warp does not bounds-check an array read in release
        # mode, so it returned whatever was in memory rather than failing).
        #
        # It was not a subtle amount wrong. `cases.run_poiseuille` is the exact case --
        # a two-relaxation-time wall at Lambda = 3/16 reproduces the analytic parabola
        # to the convergence residual -- and it came out 2.4% off, on a channel whose
        # every wall node has four such diagonal links. That is the entire reason an
        # exact case is worth having: 2.4% on a drag coefficient would have looked like
        # a resolution problem and been tuned away.
        #
        # So: wrap each periodic axis, then take the first axis that is still outside
        # and apply its face's rule. A link leaving through two *non-periodic* faces at
        # once (a true corner) is a genuine ambiguity -- the node has two walls and one
        # population to write -- and the lower axis wins. That is what every lattice
        # Boltzmann code does with a corner and it is stated rather than hidden.
        outside = int(-1)
        axis = int(-1)
        if not _in_bounds(si, nx):
            low = wp.where(si < 0, int(0), int(1))
            if faces[low] == FACE_PERIODIC:
                si = (si + nx) % nx
            else:
                axis = int(0)
                outside = low
        if not _in_bounds(sj, ny):
            low = wp.where(sj < 0, int(2), int(3))
            if faces[low] == FACE_PERIODIC:
                sj = (sj + ny) % ny
            elif outside < 0:
                axis = int(1)
                outside = low
        if not _in_bounds(sk, nz):
            low = wp.where(sk < 0, int(4), int(5))
            if faces[low] == FACE_PERIODIC:
                sk = (sk + nz) % nz
            elif outside < 0:
                axis = int(2)
                outside = low

        handled = int(0)
        if outside >= 0:
            rule = faces[outside]
            if rule == FACE_SLIP:
                mirror = int(d)
                if axis == 0:
                    mirror = int(MIRROR_X[d])
                elif axis == 1:
                    mirror = int(MIRROR_Y[d])
                else:
                    mirror = int(MIRROR_Z[d])
                set_pop(f_new, d, i, j, k,
                        pop(f_post, mirror, i, j, k))
                handled = int(1)
            else:
                # wall, and the placeholder for inlet/outlet, which `apply_faces`
                # overwrites immediately after this kernel.
                back = int(OPP[d])
                uw = face_velocity[outside]
                cu = (DTYPE(C[0, back]) * uw[0] + DTYPE(C[1, back]) * uw[1]
                      + DTYPE(C[2, back]) * uw[2])
                set_pop(f_new, d, i, j, k,
                        pop(f_post, back, i, j, k)
                        - DTYPE(2.0) * W[back] * cu * INV_CS2)
                handled = int(1)

        if handled == 0:
            phi_there = phi[si, sj, sk]
            if phi_there >= DTYPE(0.0):
                set_pop(f_new, d, i, j, k,
                        pop(f_post, d, si, sj, sk))
            else:
                # -- the source node is solid: Bouzidi --------------------------------
                opp = int(OPP[d])
                body = int(owner[si, sj, sk])
                denom = phi_here - phi_there
                frac = DTYPE(0.5)
                if denom > DTYPE(1.0e-12):
                    frac = wp.clamp(phi_here / denom, DTYPE(0.01), DTYPE(0.99))

                cj = wp.vec3(DTYPE(C[0, opp]), DTYPE(C[1, opp]), DTYPE(C[2, opp]))
                wall_term = DTYPE(0.0)
                if moving != 0:
                    cut = wp.vec3(DTYPE(i), DTYPE(j), DTYPE(k)) + cj * frac
                    uw = _wall_velocity(body, cut, body_pose, body_vel, body_com,
                                        DTYPE(1.0))
                    wall_term = DTYPE(2.0) * W[opp] * wp.dot(cj, uw) * INV_CS2

                f_in = pop(f_post, opp, i, j, k)
                reflected = DTYPE(0.0)
                if frac < DTYPE(0.5):
                    # needs the next fluid node away from the wall, at x + c_d
                    ai = int(i + C[0, d])
                    aj = int(j + C[1, d])
                    ak = int(k + C[2, d])
                    usable = (_in_bounds(ai, nx) and _in_bounds(aj, ny)
                              and _in_bounds(ak, nz))
                    if usable:
                        usable = phi[ai, aj, ak] >= DTYPE(0.0)
                    if usable:
                        reflected = (
                            DTYPE(2.0) * frac * f_in
                            + (DTYPE(1.0) - DTYPE(2.0) * frac) * pop(f_post, opp, ai, aj, ak)
                            - wall_term
                        )
                    else:
                        reflected = f_in - wall_term
                        wp.atomic_add(fallbacks, 0, 1)
                else:
                    inv2q = DTYPE(1.0) / (DTYPE(2.0) * frac)
                    reflected = (inv2q * f_in
                                 + (DTYPE(1.0) - inv2q) * pop(f_post, d, i, j, k)
                                 - wall_term * DTYPE(2.0) * inv2q)

                set_pop(f_new, d, i, j, k,
                        reflected)

                if body >= 0:
                    # Momentum into the solid along this link, and its moment about the
                    # centre of mass.
                    impulse = cj * (f_in + reflected)
                    arm = ((wp.vec3(DTYPE(i), DTYPE(j), DTYPE(k)) + cj * frac)
                           - body_com[body])
                    moment = wp.cross(arm, impulse)
                    wp.atomic_add(wrench, body, 0, impulse[0])
                    wp.atomic_add(wrench, body, 1, impulse[1])
                    wp.atomic_add(wrench, body, 2, impulse[2])
                    wp.atomic_add(wrench, body, 3, moment[0])
                    wp.atomic_add(wrench, body, 4, moment[1])
                    wp.atomic_add(wrench, body, 5, moment[2])


# -- 3. the faces that are not reflections ---------------------------------------------


@wp.kernel
def apply_faces(
    f: wp.array4d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    faces: wp.array(dtype=wp.int32),
    face_velocity: wp.array(dtype=wp.vec3),
    inlet_profile: wp.array2d(dtype=wp.vec3),
    use_profile: wp.int32,
):
    """Inlet and outlet, imposed on the face node after streaming.

    `inlet` is a full-equilibrium condition: the node is overwritten with
    `f^eq(1, u_in)`. First order at the boundary and unconditionally stable, which is
    the right trade for a face that is a long way from anything being measured -- every
    case here puts its inlet several body diameters upstream. `use_profile` takes `u_in`
    from a two-dimensional array over the -x face instead of from `face_velocity`, which
    is what the Schafer-Turek channel needs: its inflow is a parabola, and a uniform
    inlet would be a different benchmark.

    `outlet` copies the populations from one cell inside -- zero gradient on `f`, the
    "do nothing" outflow. It lets vorticity leave without reflecting it, which a fixed
    pressure does not, and that is what a shedding wake downstream of a cylinder needs.

    Launched over the whole lattice with an early return rather than as six
    two-dimensional launches: the early return costs an index comparison per node and
    no memory traffic, and six launches cost six kernel launches per substep.
    """
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]

    face = -1
    inward_i = i
    inward_j = j
    inward_k = k
    if i == 0:
        face = 0
        inward_i = 1
    elif i == nx - 1:
        face = 1
        inward_i = nx - 2
    elif j == 0:
        face = 2
        inward_j = 1
    elif j == ny - 1:
        face = 3
        inward_j = ny - 2
    elif k == 0:
        face = 4
        inward_k = 1
    elif k == nz - 1:
        face = 5
        inward_k = nz - 2
    if face < 0:
        return
    rule = faces[face]
    if rule != FACE_INLET and rule != FACE_OUTLET:
        return
    if phi[i, j, k] < DTYPE(0.0):
        return

    if rule == FACE_OUTLET:
        for d in range(Q):
            set_pop(f, d, i, j, k,
                    pop(f, d, inward_i, inward_j, inward_k))
        return

    u = face_velocity[face]
    if use_profile != 0 and face == 0:
        u = inlet_profile[j, k]
    for d in range(Q):
        set_pop(f, d, i, j, k,
                _equilibrium(d, DTYPE(1.0), u))


# -- 4. the solid, rebuilt every coupling step -----------------------------------------


@wp.func
def _sdf_sphere(p: wp.vec3, scale: wp.vec3) -> DTYPE:
    return wp.length(p) - scale[0]


@wp.func
def _sdf_box(p: wp.vec3, scale: wp.vec3) -> DTYPE:
    q = wp.vec3(wp.abs(p[0]) - scale[0], wp.abs(p[1]) - scale[1], wp.abs(p[2]) - scale[2])
    outside = wp.length(wp.vec3(wp.max(q[0], DTYPE(0.0)), wp.max(q[1], DTYPE(0.0)),
                                wp.max(q[2], DTYPE(0.0))))
    inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), DTYPE(0.0))
    return outside + inside


@wp.func
def _sdf_capsule(p: wp.vec3, scale: wp.vec3) -> DTYPE:
    """Along the local z axis: radius `scale[0]`, half-height `scale[1]`.

    Newton's own convention, which is z for both capsules and cylinders -- see the
    comment in `newton/examples/basic/example_basic_dzhanibekov.py`, "Cylinders default
    to Z axis". `couple.py` is where that is read off the model rather than assumed.
    """
    dz = p[2] - wp.clamp(p[2], -scale[1], scale[1])
    return wp.length(wp.vec3(p[0], p[1], dz)) - scale[0]


@wp.func
def _sdf_cylinder(p: wp.vec3, scale: wp.vec3) -> DTYPE:
    dr = wp.sqrt(p[0] * p[0] + p[1] * p[1]) - scale[0]
    dz = wp.abs(p[2]) - scale[1]
    outside = wp.sqrt(wp.max(dr, DTYPE(0.0)) * wp.max(dr, DTYPE(0.0))
                      + wp.max(dz, DTYPE(0.0)) * wp.max(dz, DTYPE(0.0)))
    inside = wp.min(wp.max(dr, dz), DTYPE(0.0))
    return outside + inside


@wp.kernel
def voxelize(
    phi: wp.array3d(dtype=DTYPE),
    owner: wp.array3d(dtype=wp.int32),
    shape_type: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    shape_pose: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_mesh: wp.array(dtype=wp.uint64),
    mesh_max_dist: DTYPE,
    far: DTYPE,
):
    """`phi` and `owner` for the whole lattice, from the shapes as they are right now.

    One thread a node, a loop over shapes, minimum wins. `shape_pose` is the shape's
    pose in **lattice coordinates**, already composed from the body transform and the
    shape's local transform by `couple.py` -- composing it here would be the same
    quaternion multiply repeated once per node.

    Shapes are in lattice units, so an SDF in this frame is already a `q` numerator. The
    distance is clamped to `far` so a lattice far from everything does not carry
    enormous numbers through a float32 subtraction; `far` only has to exceed one lattice
    spacing for the `q` arithmetic in `stream` to be exact, and it is set to a few cells.

    A mesh goes through `wp.mesh_query_point_sign_normal`, which is Warp's own
    closest-point query with a sign from the surface normal. `mesh_max_dist` bounds the
    search: a node further than that from the mesh is reported as `far` fluid, which is
    true and is also the only way to keep the query cheap. The sign-from-normal variant
    is the one to use for a closed mesh with reasonable triangles; the winding-number
    variant is more robust and much slower, and a robot's collision mesh is not the case
    that needs it.
    """
    i, j, k = wp.tid()
    p = wp.vec3(DTYPE(i), DTYPE(j), DTYPE(k))

    best = DTYPE(far)
    best_body = int(-1)
    for s in range(shape_type.shape[0]):
        local = wp.transform_point(wp.transform_inverse(shape_pose[s]), p)
        kind = shape_type[s]
        scale = shape_scale[s]
        d = DTYPE(far)
        if kind == SHAPE_SPHERE:
            d = _sdf_sphere(local, scale)
        elif kind == SHAPE_BOX:
            d = _sdf_box(local, scale)
        elif kind == SHAPE_CAPSULE:
            d = _sdf_capsule(local, scale)
        elif kind == SHAPE_CYLINDER:
            d = _sdf_cylinder(local, scale)
        elif kind == SHAPE_PLANE:
            # Newton's plane is the local z = 0 surface with +z the outward normal. An
            # infinite half-space, which is what a ground plane is; a finite slab is a
            # box and should be built as one.
            d = local[2]
        else:
            query = wp.mesh_query_point_sign_normal(shape_mesh[s], local, mesh_max_dist)
            if query.result:
                closest = wp.mesh_eval_position(shape_mesh[s], query.face, query.u,
                                                query.v)
                d = wp.length(local - closest) * query.sign
        if d < best:
            best = d
            best_body = shape_body[s]

    # Clamped both ways. Above, so a node a long way from everything does not carry a
    # huge number into the float32 subtraction `stream` does to find `q`. Below, because
    # `q = phi[x]/(phi[x] - phi[s])` is only meaningful when `phi[s]` is the distance to
    # the *nearby* surface: deep inside a large body the true distance is metres and the
    # fraction it would produce is nonsense. Only the first cell inside matters, and
    # `far` of a few cells is past it.
    phi[i, j, k] = wp.clamp(best, -far, far)
    owner[i, j, k] = wp.where(best < DTYPE(0.0), best_body, -1)


@wp.kernel
def volumes(
    owner: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
):
    """Count the lattice nodes each body occupies, which is its displaced volume.

    Why measure it rather than compute it. A body's buoyancy is `rho_fluid * V * g` and
    `V` has to be the volume the *solver* thinks the body has, not the volume its
    geometry has: those differ by the voxelisation, and at twenty cells across a sphere
    the difference is a few per cent. A falling sphere's terminal velocity depends on
    the difference between its weight and its buoyancy, so a buoyancy computed from the
    analytic volume and a drag computed from the voxelised surface do not balance, and
    the error shows up as a terminal velocity that is wrong by more than the drag
    correlation's own scatter.

    Counting the nodes gives the volume the boundary condition actually encloses, from
    the same `owner` field the forces come from. One atomic per solid node, once per
    coupling step.
    """
    i, j, k = wp.tid()
    body = owner[i, j, k]
    if body >= 0:
        wp.atomic_add(counts, body, 1)


@wp.kernel
def poses_from_newton(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com_local: wp.array(dtype=wp.vec3),
    pose_out: wp.array(dtype=wp.transform),
    vel_out: wp.array(dtype=wp.spatial_vector),
    com_out: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    inv_dx: DTYPE,
    dt: DTYPE,
):
    """Newton's body state, in metres and seconds, to the lattice's own units.

    The one kernel that exists purely to avoid a host round trip. Newton's `body_q` and
    `body_qd` live on the same Warp device as the lattice, so the conversion is a launch
    of `body_count` threads and nothing crosses the PCIe bus; doing it in numpy would
    be two device-to-host copies and two host-to-device copies per coupling step, which
    at a two-millisecond robot step is most of the step.

    The conventions, all three read out of Newton's own source rather than guessed:

    * `body_q` is the body frame's transform, position in metres.
    * `body_com_local` is `Model.body_com`, the centre of mass **in body-local
      coordinates**, so the world centre of mass is `transform_point(body_q, body_com)`.
    * `body_qd` is a spatial vector whose first three components are the linear velocity
      **of the centre of mass** in world frame and whose last three are the angular
      velocity in world frame (`State.body_qd`'s docstring). That is why the arm in
      `_wall_velocity` is measured from the centre of mass and not from the body origin:
      get that wrong and a spinning body's surface velocity is wrong by
      `omega x (com - origin)`, which is a rigid rotation of the whole error field and
      looks like a spurious lift.

    `origin` is the domain's low corner in metres and the half cell is subtracted here,
    for the reason `world.Domain.origin` gives: node `i` is at `origin + (i+0.5) dx`.
    """
    b = wp.tid()
    pose = body_q[b]
    position = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    lattice_p = (position - origin) * inv_dx - wp.vec3(DTYPE(0.5), DTYPE(0.5), DTYPE(0.5))
    pose_out[b] = wp.transform(lattice_p, rotation)

    com_world = wp.transform_point(pose, body_com_local[b])
    com_out[b] = (com_world - origin) * inv_dx - wp.vec3(DTYPE(0.5), DTYPE(0.5),
                                                         DTYPE(0.5))

    vel = body_qd[b]
    scale = dt * inv_dx
    vel_out[b] = wp.spatial_vector(
        vel[0] * scale, vel[1] * scale, vel[2] * scale,
        vel[3] * dt, vel[4] * dt, vel[5] * dt,
    )


@wp.kernel
def shapes_from_newton(
    shape_body: wp.array(dtype=wp.int32),
    shape_local: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    pose_out: wp.array(dtype=wp.transform),
    origin: wp.vec3,
    inv_dx: DTYPE,
):
    """Compose each shape's world pose from its body's, in lattice units, on the device.

    The companion to `poses_from_newton` and the other half of the no-round-trip rule:
    `voxelize` wants the shape poses composed, and composing them on the host would copy
    `body_q` back every coupling step. A shape whose `shape_body` is negative is
    scenery -- a ground plane, a tunnel wall -- and its local transform is already its
    world one.
    """
    s = wp.tid()
    body = shape_body[s]
    local = shape_local[s]
    world = local
    if body >= 0:
        world = wp.transform_multiply(body_q[body], local)
    position = wp.transform_get_translation(world)
    lattice_p = (position - origin) * inv_dx - wp.vec3(DTYPE(0.5), DTYPE(0.5), DTYPE(0.5))
    pose_out[s] = wp.transform(lattice_p, wp.transform_get_rotation(world))


@wp.kernel
def shift(
    f_src: wp.array4d(dtype=DTYPE),
    f_dst: wp.array4d(dtype=DTYPE),
    offset: wp.vec3i,
    rho0: DTYPE,
    u0: wp.vec3,
):
    """Translate the whole lattice by a whole number of cells: the moving window.

    A robot that flies across a fixed box eventually leaves it, and a box big enough
    for the whole trajectory is mostly empty lattice being stepped for nothing. The
    window instead follows the body: once the body has drifted more than a set distance
    from the box centre, the lattice is shifted back by that many cells and the slab
    uncovered at the trailing edge is filled with the free stream.

    Whole cells only, and that is the point. A fractional shift would be an
    interpolation of every population every time, which is both expensive and
    diffusive -- it would smear the wake the window exists to keep. An integer shift is
    a gather, exact, and the only error it introduces is that the window's own position
    is quantised to `dx`, which the body's boundary condition already is.

    What it does cost: momentum is not conserved across a shift, because fluid leaves at
    one face and uniform free stream arrives at the other. `FluidWorld.momentum_balance`
    reports shifts separately from everything else for that reason -- a momentum check
    run across a shift is checking the window, not the solver.
    """
    i, j, k = wp.tid()
    nx = wp.where(SOA, f_src.shape[1], f_src.shape[0])
    ny = wp.where(SOA, f_src.shape[2], f_src.shape[1])
    nz = wp.where(SOA, f_src.shape[3], f_src.shape[2])
    si = int(i + offset[0])
    sj = int(j + offset[1])
    sk = int(k + offset[2])
    if _in_bounds(si, nx) and _in_bounds(sj, ny) and _in_bounds(sk, nz):
        for q in range(Q):
            set_pop(f_dst, q, i, j, k,
                    pop(f_src, q, si, sj, sk))
    else:
        for q in range(Q):
            set_pop(f_dst, q, i, j, k,
                    _equilibrium(q, rho0, u0))


# -- 5. fresh nodes --------------------------------------------------------------------


@wp.kernel
def refill(
    f: wp.array4d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    phi_prev: wp.array3d(dtype=DTYPE),
    owner_prev: wp.array3d(dtype=wp.int32),
    body_pose: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    refilled: wp.array(dtype=wp.int32),
):
    """Give a node that has just left a body somewhere to start from.

    A node inside a moving body holds populations from no physical state at all. When
    the body moves off it, the node is fluid and whatever is in it is garbage, so it has
    to be built: density averaged over the fluid neighbours that already had one, and
    velocity taken as the body's own surface velocity at that point, which is continuous
    with the boundary condition the node was subject to one step earlier.

    Equilibrium refill, which is the cheapest of the three usual schemes
    (`DECISIONS_LIVE.md` D5 has the other two and why they are not here). The quantity a
    bad refill ruins is the force on a translating body, so `benchmarks/live/` measures
    exactly that -- a sphere sliding through a quiescent lattice at constant velocity,
    whose drag is known -- rather than asserting this is good enough.

    Both `phi` arrays are read, so this must run after `voxelize` and before the first
    substep of the coupling step, with `phi_prev` still holding the previous step's
    field. `world.py` does the copy; doing it here would need a second kernel anyway.
    """
    i, j, k = wp.tid()
    if phi[i, j, k] < DTYPE(0.0):
        return
    if phi_prev[i, j, k] >= DTYPE(0.0):
        return  # it was already fluid; it has a history

    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]

    rho = DTYPE(0.0)
    found = DTYPE(0.0)
    for d in range(1, Q):
        ni = int(i + C[0, d])
        nj = int(j + C[1, d])
        nk = int(k + C[2, d])
        inside = (_in_bounds(ni, nx) and _in_bounds(nj, ny) and _in_bounds(nk, nz))
        if inside:
            # a neighbour that was solid too has populations no better than ours
            if phi_prev[ni, nj, nk] >= DTYPE(0.0):
                total = DTYPE(0.0)
                for q in range(Q):
                    total += pop(f, q, ni, nj, nk)
                rho += total
                found += DTYPE(1.0)

    if found > DTYPE(0.0):
        rho = rho / found
    else:
        rho = DTYPE(1.0)  # wholly enclosed a step ago: unit density is the only guess

    body = owner_prev[i, j, k]
    u = _wall_velocity(body, wp.vec3(DTYPE(i), DTYPE(j), DTYPE(k)), body_pose, body_vel,
                       body_com, DTYPE(1.0))
    for q in range(Q):
        set_pop(f, q, i, j, k,
                _equilibrium(q, rho, u))
    wp.atomic_add(refilled, 0, 1)


# -- 6. what a human or a figure reads -------------------------------------------------


@wp.kernel
def macroscopic(
    f: wp.array4d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    rho: wp.array3d(dtype=DTYPE),
    u: wp.array3d(dtype=wp.vec3),
    force_field: wp.array3d(dtype=wp.vec3),
    body_force: wp.vec3,
    use_field: wp.int32,
):
    """Density and velocity, with the same half-force correction `collide` applies.

    Not part of the step loop: called when something wants to look. The half-force
    matters even here -- without it a force-driven channel's reported profile is a half
    step out of phase with the one the solver is actually converging to, which shows up
    as a systematic error against the exact parabola and looks like a solver defect.
    """
    i, j, k = wp.tid()
    if phi[i, j, k] < DTYPE(0.0):
        rho[i, j, k] = DTYPE(0.0)
        u[i, j, k] = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
        return
    total = DTYPE(0.0)
    mom = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for q in range(Q):
        value = pop(f, q, i, j, k)
        total += value
        mom += wp.vec3(DTYPE(C[0, q]), DTYPE(C[1, q]), DTYPE(C[2, q])) * value
    force = body_force
    if use_field != 0:
        force = force + force_field[i, j, k]
    rho[i, j, k] = total
    u[i, j, k] = (mom + force * DTYPE(0.5)) / total


@wp.kernel
def initialise(
    f: wp.array4d(dtype=DTYPE),
    rho0: DTYPE,
    u0: wp.vec3,
):
    """Every node at equilibrium with a uniform state. Solid nodes included, on purpose:
    a node that is solid now may be fluid in ten steps, and starting it at equilibrium
    costs one write and saves `refill` from having nothing to average."""
    i, j, k = wp.tid()
    for q in range(Q):
        set_pop(f, q, i, j, k,
                _equilibrium(q, rho0, u0))


# -- 7. actuator disks -----------------------------------------------------------------


@wp.kernel
def zero_force(force_field: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    force_field[i, j, k] = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))


@wp.kernel
def disks_to_force(
    force_field: wp.array3d(dtype=wp.vec3),
    phi: wp.array3d(dtype=DTYPE),
    disk_centre: wp.array(dtype=wp.vec3),
    disk_axis: wp.array(dtype=wp.vec3),
    disk_radius: wp.array(dtype=DTYPE),
    disk_thickness: wp.array(dtype=DTYPE),
    disk_thrust: wp.array(dtype=DTYPE),
):
    """An actuator disk per rotor: thrust spread over a thin cylinder of lattice.

    The blades are not resolved and are not going to be. A 2-inch propeller at 20,000
    rpm has a tip Mach number around 0.15 and a blade chord of a few millimetres; a
    lattice that resolved the blade would need cells under a millimetre and a timestep
    under a microsecond, which is four orders of magnitude of work away from a
    millisecond robot step. What a hovering multirotor's airframe actually feels is the
    *induced flow*, and an actuator disk reproduces that: a momentum source of the right
    total thrust, in the right place, pointing the right way.

    `disk_thrust` is the total force in lattice units; it is divided by the number of
    lattice cells the disk covers, which is counted here rather than computed from
    `pi r^2 t` so that a disk clipped by the domain edge or intersecting the airframe
    still delivers the thrust it was told to. A disk is one kernel launch over the
    lattice per coupling step, and the alternative -- a per-cell weight list built on
    the host -- would be a host round trip inside exactly the loop that must not have
    one.

    Signs: the force on the fluid is `-thrust * axis`, because `axis` points the way the
    rotor pushes the *vehicle* and the fluid is pushed the other way. Newton's third
    law is applied by the caller putting `+thrust * axis` on the body, which keeps the
    pair visible in one place (`world.py::FluidWorld.set_rotor_thrust`) instead of
    hiding half of it in a kernel.
    """
    i, j, k = wp.tid()
    if phi[i, j, k] < DTYPE(0.0):
        return
    p = wp.vec3(DTYPE(i), DTYPE(j), DTYPE(k))
    total = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for s in range(disk_centre.shape[0]):
        axis = disk_axis[s]
        offset = p - disk_centre[s]
        along = wp.dot(offset, axis)
        if wp.abs(along) <= disk_thickness[s] * DTYPE(0.5):
            radial = wp.length(offset - axis * along)
            if radial <= disk_radius[s]:
                # cells covered: pi r^2 * thickness, in lattice cells. Counting them
                # exactly would need a reduction; this is the same number to within the
                # disk's own discretisation, and `world.py` rescales from a measured
                # count so the delivered thrust is exact rather than approximate.
                volume = (DTYPE(3.14159265358979) * disk_radius[s] * disk_radius[s]
                          * wp.max(disk_thickness[s], DTYPE(1.0)))
                total -= axis * (disk_thrust[s] / volume)
    force_field[i, j, k] = force_field[i, j, k] + total


# -- 8. handing the wrench to Newton ---------------------------------------------------


@wp.kernel
def wrench_to_newton(
    wrench: wp.array2d(dtype=DTYPE),
    body_f: wp.array(dtype=wp.spatial_vector),
    force_scale: DTYPE,
    torque_scale: DTYPE,
    accumulate: wp.int32,
):
    """Convert the lattice wrench to SI and add it to Newton's `body_f`.

    `body_f` is documented in `newton._src.sim.state.State` as "an external wrench in
    world frame with the body's center of mass (COM) as reference point", linear force
    in the first three components and torque in the last three -- which is the layout
    `stream` accumulated into, so this is two multiplications and an add.

    `accumulate` is on by default because Newton's own loop is
    `state.clear_forces()`, then every force source adds, then `solver.step`. Gravity,
    a controller's thrust and this are three sources, and a kernel that assigned would
    silently delete the other two.

    One launch of `body_count` threads per coupling step, which is the only time
    anything in this file touches an array Newton owns. No host round trip: both arrays
    live on the same device, and `wp.launch` on them is the whole handover.
    """
    b = wp.tid()
    force = wp.vec3(wrench[b, 0], wrench[b, 1], wrench[b, 2]) * force_scale
    torque = wp.vec3(wrench[b, 3], wrench[b, 4], wrench[b, 5]) * torque_scale
    value = wp.spatial_vector(force[0], force[1], force[2],
                              torque[0], torque[1], torque[2])
    if accumulate != 0:
        body_f[b] = body_f[b] + value
    else:
        body_f[b] = value
