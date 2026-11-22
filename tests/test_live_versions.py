"""The `[live]` extra's dependency range, held against the Warp and Newton installed.

`docs/live-version-matrix.md` is the measurement: the extension was run end to end on
warp-lang 1.19.0 + newton 1.8.0 (the development environment) and on
warp-lang 1.17.0 + newton 1.6.1 (what Isaac Lab 3.0.0's own metadata pins, so what a
user running this inside Isaac Lab gets whatever `pyproject.toml` asks for), and it
behaves the same on both -- the Re 100 cavity is bit-identical, the Newton coupling
gives the same wrench to the last float, and no source change was needed to get there.

A document cannot hold that. The point of this file is that the *facts* the matrix rests
on are checked against the packages actually installed, in the main suite, on every run.
Each test below is one claim that would have needed a compatibility shim had it gone the
other way, and the comment on each says what the shim would have had to be. They are
written so that a future Warp or Newton that breaks one fails *here*, with a sentence,
rather than inside a lattice kernel or -- worse -- silently, in a number.

Everything here runs on Warp's CPU device and allocates nothing larger than a few
kilobytes, so it belongs in the main suite rather than behind a GPU marker. The Newton
tests skip rather than fail when Newton is absent: a pure-fluid run on a machine with no
robot simulator is a supported case, which is why `couple.py` is the only file that
imports it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from openreynolds.live import units

warp = pytest.importorskip("warp", reason="the [live] extra is not installed")

ROOT = Path(__file__).resolve().parents[1]


# -- 1. the declared range and the tested range are the same thing ---------------------


def _minor(spec: str) -> tuple[int, int]:
    major, _, minor = spec.partition(".")
    return int(major), int(minor or 0)


def _installed_minor(dist: str) -> tuple[int, int] | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version(dist)
    except PackageNotFoundError:
        return None
    return tuple(int(p) for p in found.split(".")[:2])


def test_pyproject_live_extra_matches_units_band():
    """`pyproject.toml`'s `live` extra and `units.WARP_TESTED` cannot drift apart.

    The range exists in two places because each is unreachable from the other: pip reads
    the first and only at install time, and `FluidWorld.report` reads the second at run
    time, on a machine where pip has already resolved something else entirely (an Isaac
    Lab environment, which is the case the whole band is for). Two copies of a version
    range with nothing between them is how a package comes to declare one thing and
    measure another, so the test is the thing between them.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    live = re.search(r"^live = \[(.*?)^\]", text, re.S | re.M)
    assert live, "pyproject.toml has no `live = [...]` extra any more"
    for dist, band in (("warp-lang", units.WARP_TESTED), ("newton", units.NEWTON_TESTED)):
        found = re.search(rf'"{dist}>=([0-9.]+),<([0-9.]+)"', live.group(1))
        assert found, (
            f"the `live` extra no longer declares a bounded range for {dist}. The band "
            f"units.py measured is >={band[0]},<{band[1]}; see "
            f"docs/live-version-matrix.md for what that was measured on"
        )
        assert (found.group(1), found.group(2)) == band, (
            f"{dist}: pyproject declares >={found.group(1)},<{found.group(2)} but "
            f"units.py was measured over >={band[0]},<{band[1]}"
        )


@pytest.mark.parametrize("dist, band", [
    ("warp-lang", units.WARP_TESTED),
    ("newton", units.NEWTON_TESTED),
])
def test_installed_version_is_inside_the_tested_band(dist, band):
    """What is installed here is what the matrix measured, so the rest of this file means
    something. A version outside the band is not a bug; it is an untested configuration,
    and `units.version_note` is what tells a user that in the report their numbers come
    out of. This test exists so that nobody reads the other tests below as evidence
    about a version they were never run on."""
    found = _installed_minor(dist)
    if found is None:
        pytest.skip(f"{dist} is not installed")
    assert _minor(band[0]) <= found <= _minor(band[1]), (
        f"{dist} {'.'.join(str(p) for p in found)} is outside the measured band "
        f">={band[0]},<{band[1]}. Either it works and the band and "
        f"docs/live-version-matrix.md should grow to say so, or it does not and the "
        f"failure belongs in that file too"
    )


def test_version_note_is_silent_inside_the_band_and_speaks_outside_it():
    """The one user-visible consequence of the band: a line in `FluidWorld.report`.

    Checked in both directions, because a note that never fires and a note that always
    fires look identical from inside the band.
    """
    assert units.version_note() is None, (
        f"version_note() fired on a supported environment: {units.version_note()}"
    )
    band = units.WARP_TESTED
    try:
        units.WARP_TESTED = ("99.0", "99.1")
        note = units.version_note()
    finally:
        units.WARP_TESTED = band
    assert note is not None and "warp-lang" in note
    assert "live-version-matrix" in note, "the note has to name the measurement"


# -- 2. `wp.set_module_options`, which is the first thing `lattice.py` does ------------


def test_module_options_tolerate_a_key_this_warp_does_not_know():
    """A `wp.set_module_options` key the installed Warp has never heard of is accepted.

    This is the one that decided there is no shim around that call. `lattice.py` sets
    `max_unroll` and `enable_backward` at module scope, before any kernel is defined, so
    a Warp that rejected a key would turn the whole extension into an import error -- not
    a slow kernel, an `ImportError` on `import openreynolds.live.lattice`. Measured on
    1.17.0 and 1.19.0: neither validates the names at all, and an invented option is
    swallowed silently.

    If this ever fails, the shim is known and small: set the options one key at a time in
    a `try/except`, keeping `enable_backward` (correctness-adjacent -- it halves compile
    time and code size) ahead of `max_unroll` (speed only, and `_preferred_layout`
    already measures what losing it costs).
    """
    before = dict(warp.get_module_options())
    try:
        warp.set_module_options({"openreynolds_not_a_real_option": 7})
    except Exception as exc:  # pragma: no cover - the day this fires, read the docstring
        pytest.fail(
            f"warp {warp.config.version} rejects an unknown module option "
            f"({type(exc).__name__}: {exc}), so lattice.py's module-scope "
            f"set_module_options call now needs the per-key try/except this test's "
            f"docstring describes"
        )
    finally:
        warp.set_module_options({k: before[k] for k in ("max_unroll", "enable_backward")})


def test_max_unroll_is_honoured_and_is_past_Q():
    """`max_unroll` survives into the module options, and the value is past `Q`.

    Warp unrolls a constant-bound `range` only up to `max_unroll`, whose default is 16,
    and `Q` is 19 -- so without this the direction loop in every kernel in `lattice.py`
    is a dynamic loop. Both the setting and the arithmetic are checked: a future Warp
    that renamed the key would still pass a `hasattr`-style check and would silently cost
    the step rate, which is the kind of regression nobody notices.
    """
    from openreynolds.live import lattice as L

    options = L.wp.get_module_options(L)
    assert options["max_unroll"] >= L.Q, (
        f"max_unroll is {options['max_unroll']} and Q is {L.Q}: the direction loop in "
        f"every kernel is a dynamic loop. See lattice._preferred_layout for the cost"
    )
    assert options["enable_backward"] is False


# -- 3. the Warp symbols whose behaviour would be invisible if it changed --------------


def test_where_takes_the_condition_first():
    """`wp.where(cond, if_true, if_false)`, which is the order `lattice.py` relies on.

    `wp.where` replaced `wp.select`, whose arguments were the other way round
    (`select(cond, if_false, if_true)`). `lattice.stream` uses `wp.where` eleven times to
    pick a face index and a sign, and `lattice.voxelize` uses it to decide whether a node
    is owned by a body -- all of them in branches that are *correct either way as code*
    and simply reflect off the wrong face or own the wrong body if the order flips. There
    is no exception to catch and nothing in the output that looks wrong at a glance, so
    the order is asserted here rather than trusted.

    Measured identical on warp 1.17.0 and 1.19.0.
    """
    @warp.kernel
    def pick(out: warp.array(dtype=warp.int32)):
        out[0] = warp.where(True, 111, 222)
        out[1] = warp.where(False, 111, 222)

    got = warp.zeros(2, dtype=warp.int32, device="cpu")
    warp.launch(pick, dim=1, inputs=[got], device="cpu")
    assert got.numpy().tolist() == [111, 222], (
        "wp.where's argument order has changed: every branch in lattice.stream and "
        "lattice.voxelize that uses it now picks the other side"
    )


def test_mesh_query_point_sign_normal_returns_the_fields_voxelize_reads():
    """`.result`, `.sign`, `.face`, `.u`, `.v`, and a distance with the sign on it.

    `lattice.voxelize`'s `MESH` branch is the only path a robot made of triangles reaches,
    and it reads all five fields of the query struct plus `wp.mesh_eval_position`. A
    renamed field is an immediate compile error and would be caught by any test that ran
    a mesh; a *sign convention* flip is not, and would turn every mesh body inside out --
    `phi < 0` is the solid mask, so the fluid would be inside the robot. Hence the
    assertion is on the sign at a point known to be inside, not on the fields existing.

    The mesh is a unit cube and the point is its centre, so the distance is exactly 1 and
    the sign is negative.
    """
    corners = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0)
                        for z in (-1.0, 1.0)], dtype=np.float32)
    faces = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
                      2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3],
                     dtype=np.int32)
    mesh = warp.Mesh(points=warp.array(corners, dtype=warp.vec3, device="cpu"),
                     indices=warp.array(faces, dtype=warp.int32, device="cpu"))

    @warp.kernel
    def probe(m: warp.uint64, out_signed: warp.array(dtype=warp.float32)):
        p = warp.vec3(0.0, 0.0, 0.0)
        query = warp.mesh_query_point_sign_normal(m, p, 10.0)
        if query.result:
            closest = warp.mesh_eval_position(m, query.face, query.u, query.v)
            out_signed[0] = warp.length(p - closest) * query.sign

    out = warp.zeros(1, dtype=warp.float32, device="cpu")
    warp.launch(probe, dim=1, inputs=[mesh.id, out], device="cpu")
    assert float(out.numpy()[0]) == pytest.approx(-1.0, abs=1e-5), (
        "the centre of a unit cube is no longer at signed distance -1: either "
        "mesh_query_point_sign_normal's sign convention has flipped, in which case every "
        "mesh body in lattice.voxelize is inside out, or mesh_eval_position has"
    )


def test_a_constant_declared_inside_a_dynamic_loop_can_be_mutated():
    """Warp's "error mutating a constant inside a dynamic loop" rule, as `stream` needs it.

    `lattice.stream`'s direction loop is written without `continue` and with every
    mutated local declared as `int(...)` or `DTYPE(...)`, because Warp refuses to mutate
    a local whose type it inferred as constant inside a loop it could not unroll. That
    style is load-bearing and ugly, and the comment in `stream` says so; this test is
    what keeps it from being "cleaned up" against a Warp that tightened the rule, and
    what would catch a Warp that loosened it in a way that changed the generated code.

    The loop bound here is `Q = 19`, the same as `stream`'s, and the kernel mutates an
    int and a float declared the same way `stream` declares `outside`, `axis`, `handled`
    and `wall_term`.
    """
    from openreynolds.live import lattice as L

    @warp.kernel
    def dynamic(out: warp.array(dtype=warp.int32), acc: warp.array(dtype=warp.float32)):
        total = int(0)
        weight = warp.float32(0.0)
        for d in range(19):
            axis = int(-1)
            handled = int(0)
            if d % 2 == 0:
                axis = int(d)
                handled = int(1)
            if handled == 1:
                total += axis
            weight += warp.float32(0.5)
        out[0] = total
        acc[0] = weight

    o = warp.zeros(1, dtype=warp.int32, device="cpu")
    a = warp.zeros(1, dtype=warp.float32, device="cpu")
    warp.launch(dynamic, dim=1, inputs=[o, a], device="cpu")
    assert int(o.numpy()[0]) == sum(range(0, 19, 2))
    assert float(a.numpy()[0]) == pytest.approx(0.5 * L.Q)


def test_atomic_add_works_on_the_two_shapes_the_wrench_needs():
    """`wp.atomic_add` on `array2d(float32)` with two indices, and on `array(int32)`.

    These are the accumulators `lattice.stream` uses: the per-body wrench is
    `(body_count, 6)` floats rather than an array of `spatial_vector` precisely because a
    plain float is guaranteed an atomic and a vector type is not (the comment in `stream`
    says so), and the Bouzidi fallback counter is one `int32`. Thousands of boundary links
    hit the same six floats, so a missing atomic here is not an error, it is a wrong force
    that depends on thread scheduling.
    """
    @warp.kernel
    def accumulate(w: warp.array2d(dtype=warp.float32), n: warp.array(dtype=warp.int32)):
        t = warp.tid()
        warp.atomic_add(w, 0, t % 6, 1.0)
        warp.atomic_add(n, 0, 1)

    w = warp.zeros((2, 6), dtype=warp.float32, device="cpu")
    n = warp.zeros(1, dtype=warp.int32, device="cpu")
    warp.launch(accumulate, dim=600, inputs=[w, n], device="cpu")
    assert w.numpy()[0].tolist() == [100.0] * 6
    assert w.numpy()[1].tolist() == [0.0] * 6
    assert int(n.numpy()[0]) == 600


def test_the_lattice_tables_still_fit_a_19_wide_warp_constant():
    """A 3x19 `int32` matrix and a 19-long vector, as compile-time constants.

    Warp's fixed-shape vector and matrix types are generated per shape, and 19 is not one
    of the sizes anything else uses. `lattice.py` bakes the whole velocity set into the
    generated code this way; if a Warp release capped the generated sizes, the import
    would fail here rather than in a kernel, and the fallback (pass the tables as device
    arrays) would cost a global-memory read per direction per node.
    """
    from openreynolds.live import lattice as L

    assert L.C_NP.shape == (L.D, L.Q)

    @warp.kernel
    def read_tables(out: warp.array(dtype=warp.float32)):
        total = warp.float32(0.0)
        for q in range(19):
            total += L.W[q] * warp.float32(L.C[0, q] + L.OPP[q])
        out[0] = total

    out = warp.zeros(1, dtype=warp.float32, device="cpu")
    warp.launch(read_tables, dim=1, inputs=[out], device="cpu")
    expected = float((L.W_NP * (L.C_NP[0] + L.OPP_NP)).sum())
    assert float(out.numpy()[0]) == pytest.approx(expected, rel=1e-5)


def test_torch_interop_needs_the_runtime_and_importing_the_extension_creates_it():
    """`wp.from_torch` shares memory, and is reachable the moment this package is imported.

    Two claims in one test because the second is what makes the first true in practice.

    `wp.from_torch` and `wp.to_torch` wrap without copying: the pointers are equal and a
    write through either side is visible from the other. That is the property an Isaac Lab
    user depends on, because their observations are torch tensors and this extension's
    arrays are `wp.array`s.

    And on **both** 1.17.0 and 1.19.0 those functions raise
    `AttributeError: 'NoneType' object has no attribute 'cpu_device'` when they are the
    first Warp call in a process -- Warp's runtime is created lazily and
    `device_from_torch` does not force it. `lattice._preferred_layout` calls
    `wp.get_cuda_device_count()` at module scope, which does, so importing anything from
    this package leaves torch interop working. That is an accident of the layout choice
    and this test is what turns it into a property.
    """
    torch = pytest.importorskip("torch", reason="torch is not installed here")
    import openreynolds.live.lattice  # noqa: F401  -- the import under test

    import warp._src.context as context
    assert context.runtime is not None, (
        "importing openreynolds.live.lattice no longer creates Warp's runtime, so a "
        "caller whose first Warp call is wp.from_torch now gets an AttributeError about "
        "cpu_device. See lattice._preferred_layout"
    )

    t = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    a = warp.from_torch(t, dtype=warp.vec3)
    assert a.ptr == t.data_ptr(), "wp.from_torch copied instead of wrapping"
    t[0, 0] = -7.0
    assert float(a.numpy()[0][0]) == -7.0, "the torch write is not visible through warp"

    b = warp.zeros(4, dtype=warp.float32, device="cpu")
    tb = warp.to_torch(b)
    assert b.ptr == tb.data_ptr(), "wp.to_torch copied instead of wrapping"
    b.fill_(warp.float32(2.5))
    assert tb.tolist() == [2.5] * 4, "the warp write is not visible through torch"


# -- 4. Newton's field names and `shape_scale`'s packing ------------------------------


@pytest.fixture(scope="module")
def one_of_each():
    """A finalised Newton model with one of every shape type `couple.py` can voxelise.

    Built here rather than fetched from a fixture file because the thing under test is
    what *this* Newton writes into `shape_scale`, which a stored file could not tell us.
    One body so that `shape_body` is exercised, and the plane is static (`body=-1`) so
    that the scenery path is too.
    """
    newton = pytest.importorskip("newton", reason="newton is not installed")
    builder = newton.ModelBuilder()
    body = builder.add_body(
        xform=warp.transform(warp.vec3(0.0, 0.0, 1.0), warp.quat_identity()), mass=1.0)
    builder.add_shape_sphere(body, radius=0.3)
    builder.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
    builder.add_shape_capsule(body, radius=0.11, half_height=0.42)
    builder.add_shape_cylinder(body, radius=0.13, half_height=0.44)
    builder.add_shape_plane(xform=warp.transform_identity(), width=0.0, length=0.0)
    corners = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0)
                        for z in (-1.0, 1.0)], dtype=np.float32)
    faces = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
             2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    builder.add_shape_mesh(body, mesh=newton.Mesh(corners, faces), scale=(0.5, 0.6, 0.7))
    return newton, builder.finalize()


def test_every_newton_field_the_adapter_reads_exists(one_of_each):
    """The field names in `couple.py`'s header docstring, against the installed Newton.

    `couple.py` is the only file in the extension that knows Newton's names, and it reads
    them off `Model` and `State` by attribute. A rename is an `AttributeError` deep inside
    a coupling step; this is the same failure with a sentence on it, at collection time.
    `shape_source_ptr` is called out separately because it is the one whose *absence*
    `couple.py` papers over with a zeros array, which would voxelise every mesh body as
    empty space rather than raising.
    """
    _, model = one_of_each
    state = model.state()
    for name, dtype in (
        ("shape_type", warp.int32), ("shape_body", warp.int32),
        ("shape_transform", warp.transform), ("shape_scale", warp.vec3),
        ("body_com", warp.vec3), ("shape_source_ptr", warp.uint64),
    ):
        array = getattr(model, name, None)
        assert isinstance(array, warp.array), f"Model.{name} is {type(array).__name__}"
        assert array.dtype is dtype or array.dtype.__name__.startswith(dtype.__name__), (
            f"Model.{name} is now {array.dtype.__name__}, not {dtype.__name__}"
        )
    assert isinstance(getattr(model, "shape_label", None), list)
    assert isinstance(getattr(model, "body_label", None), list)
    for name in ("body_q", "body_qd", "body_f"):
        assert isinstance(getattr(state, name, None), warp.array), f"State.{name}"
    assert callable(state.clear_forces)
    assert model.shape_source_ptr is not None, (
        "Model.shape_source_ptr has gone; couple.py falls back to a zeros array when it "
        "is missing, which voxelises every MESH shape as empty space instead of failing"
    )


def test_geo_type_names_still_cover_the_adapter_map(one_of_each):
    """Every name in `GEO_TO_SHAPE` is a real `newton.GeoType` member.

    `couple.py` maps Newton's enum to the extension's shape codes **by name**, so that a
    renumbering of `GeoType` changes nothing. The cost of that choice is this: a *rename*
    now matters where a renumbering does not, and a name that quietly left the enum would
    make `GEO_TO_SHAPE` an entry that can never match, which looks like a model with an
    unsupported shape rather than like a version problem.
    """
    newton, _ = one_of_each
    from openreynolds.live.couple import GEO_TO_SHAPE

    members = {member.name for member in newton.GeoType}
    missing = sorted(set(GEO_TO_SHAPE) - members)
    assert not missing, (
        f"GEO_TO_SHAPE names shapes newton {getattr(newton, '__version__', '?')} does "
        f"not have: {missing}. They map by name on purpose (see couple.py), so this is a "
        f"rename and the dictionary is where it is fixed"
    )


def test_shape_scale_packing_matches_the_SCALE_table(one_of_each):
    """`shape_scale`'s packing, measured, against `couple.py::_SCALE`.

    The most serious thing in this file. Newton's own comments in `builder.py` say a
    sphere's scale is `(r, r, r)` and a capsule's is `(r, h, r)`; neither is true -- the
    writers store zeros in the unused components, and `couple.py`'s header docstring
    records what a model actually contains. The SDFs in `lattice.py` were written to that
    measured packing, so `_SCALE` is the identity and a changed packing **cannot raise**:
    it would put the radius where the half-height belongs and voxelise the body as a point
    or a slab, with forces that are wrong and plausible.

    So the radii and half-heights given to the builder are asserted back out of the model,
    component by component, on whatever Newton is installed. Measured identical on 1.6.1
    (Isaac Lab 3.0.0's pin) and 1.8.0.
    """
    newton, model = one_of_each
    from openreynolds.live.couple import GEO_TO_SHAPE, _SCALE

    kinds = model.shape_type.numpy()
    scales = model.shape_scale.numpy()
    by_name = {newton.GeoType(int(k)).name: s for k, s in zip(kinds, scales)}

    expected = {
        "SPHERE": (0.3, 0.0, 0.0),
        "BOX": (0.1, 0.2, 0.3),
        "CAPSULE": (0.11, 0.42, 0.0),
        "CYLINDER": (0.13, 0.44, 0.0),
        "MESH": (0.5, 0.6, 0.7),
    }
    for name, want in expected.items():
        assert name in by_name, f"the fixture model has no {name} shape"
        got = tuple(float(v) for v in by_name[name])
        assert got == pytest.approx(want, abs=1e-6), (
            f"newton {getattr(newton, '__version__', '?')} packs a {name}'s shape_scale "
            f"as {got}, not {want}. couple.py::_SCALE and the matching SDF in "
            f"lattice.py both assume the second; read couple.py's header docstring "
            f"before changing either, because the SDFs were written to this packing "
            f"rather than to a convention of their own"
        )
        # and the table that is supposed to be the identity for these types still is
        assert tuple(float(v) for v in _SCALE[GEO_TO_SHAPE[name]](by_name[name])) == got


def test_solver_step_takes_the_five_positionals_the_coupling_passes(one_of_each):
    """`solver.step(state_in, state_out, control, contacts, dt)`, positionally.

    newton 1.8.0 added a keyword-only `observables=None` that 1.6.1 does not have, which
    is why `NewtonCoupling.step` passes five positionals and names none of them. Checked
    by signature rather than by running a solver, so that it holds on a machine with no
    MuJoCo backend installed -- `newton[sim]` pins `mujoco-warp ~=3.12.0` at newton 1.6.1
    and `~=3.17.0` at 1.8.0, so the backend is not always there, and nothing in the
    extension imports it.
    """
    import inspect

    newton, _ = one_of_each
    for name in ("SolverMuJoCo", "SolverXPBD", "SolverSemiImplicit"):
        solver = getattr(newton.solvers, name, None)
        assert solver is not None, (
            f"newton.solvers.{name} has gone; couple.py's docstring and "
            f"docs/live-version-matrix.md both name it"
        )
        positional = [
            p for p in inspect.signature(solver.step).parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert [p.name for p in positional][:5] == [
            "state_in", "state_out", "control", "contacts", "dt"], (
            f"{name}.step's positional parameters have changed to "
            f"{[p.name for p in positional]}; NewtonCoupling.step passes five "
            f"positionally so that it works on newton 1.6.1, which has no `observables`"
        )


def test_newton_and_our_arrays_share_a_kernel_without_a_copy(one_of_each):
    """The handover: a Newton array and one of ours in one launch, by pointer.

    `couple.py`'s whole no-host-round-trip claim rests on this. The five arrays the step
    loop hands across are `state.body_f`, `state.body_q`, `state.body_qd`,
    `model.body_com` and `model.shape_source_ptr`, and the thing to establish is not that
    a launch works -- it would work over a copy too -- but that the pointer the kernel
    writes through is the one Newton keeps. So the pointers are taken before, a kernel
    that reads two of Newton's arrays and writes a third plus one of ours is launched, and
    the pointers are read again off freshly-accessed attributes, in case any of them is a
    property that rebuilds.
    """
    _, model = one_of_each
    state = model.state()
    state.clear_forces()
    watched = ("body_f", "body_q", "body_qd")
    before = {n: getattr(state, n).ptr for n in watched}
    before_model = {n: getattr(model, n).ptr for n in ("body_com", "shape_source_ptr")}
    ours = warp.zeros(model.body_count, dtype=warp.float32,
                      device=state.body_f.device)

    @warp.kernel
    def handover(body_f: warp.array(dtype=warp.spatial_vector),
                 body_q: warp.array(dtype=warp.transform),
                 body_com: warp.array(dtype=warp.vec3),
                 mine: warp.array(dtype=warp.float32)):
        b = warp.tid()
        com = warp.transform_point(body_q[b], body_com[b])
        body_f[b] = body_f[b] + warp.spatial_vector(com[0], com[1], com[2],
                                                    0.0, 0.0, 0.0)
        mine[b] = com[2]

    warp.launch(handover, dim=int(model.body_count),
                inputs=[state.body_f, state.body_q, model.body_com, ours],
                device=state.body_f.device)

    assert {n: getattr(state, n).ptr for n in watched} == before, (
        "a Newton State array moved under a launch, so couple.py is writing into a copy"
    )
    assert {n: getattr(model, n).ptr for n in ("body_com", "shape_source_ptr")} \
        == before_model
    # and the kernel really wrote through both arrays, Newton's and ours
    assert float(state.body_f.numpy()[0][2]) == pytest.approx(1.0, abs=1e-5)
    assert float(ours.numpy()[0]) == pytest.approx(1.0, abs=1e-5)
