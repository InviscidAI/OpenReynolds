#!/usr/bin/env python3
"""Hold a hull at five prescribed drafts and weigh the water: coupling, or particles?

    # wiring, on this machine's CPU -- the physics at this spacing is meaningless
    python -X utf8 benchmarks/live/draft_curve.py --radius 0.006 --settle 400 --samples 40

    # the measurement
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/draft_curve.py
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/draft_curve.py --order hull-first

## The question

Four M6 floating cases miss Archimedes, and after `_float_rig`'s held-body bug was fixed
the static float still reads 14.6x its target. Two things can produce that and the fix
for each is in a different file:

* the **boundary particles** are not carrying the hydrostatic pressure, in which case the
  vertical force on the hull is wrong at every depth and the coupling is innocent;
* the **coupling** is wrong -- a frame, a centre of mass, a double-counted gravity, the
  point the wrench is applied at -- in which case the force the fluid reports is right
  and what Newton does with it is not.

A held body separates them in one measurement, because a held body has an exact reference
and no dynamics: `F_z = rho g V_submerged`. Nothing about Newton, the body's mass, its
centre of mass or its integrator enters that. So this file never steps Newton at all --
`world.step()` advances the fluid and the hull's Akinci particles are posed from a
`body_q` that nothing is integrating, which is commit 84a2329's fix taken to its limit --
and reads `world.wrench(body)` at five drafts from barely submerged to fully submerged.

**What each shape of curve means**, and this is the whole reason for five points rather
than one:

    ratio ~ 1 across the range   the particles carry the pressure; the bug is downstream,
                                 in the coupling or in what is done with the wrench
    ratio ~ 2, flat              something is counted twice, and a constant factor cannot
                                 be a waterline error -- those are not constant in draft
    ratio drifting with draft    a free-surface or waterline problem: the force is right
                                 about some depth and wrong about where the water is
    constant offset in newtons   a force that does not scale with the submerged volume is
                                 not buoyancy -- it is being added somewhere

## The two orders, and why both are run

`_floating_tank` calls `fill_below` and *then* `_float_rig` adds the hull. `sph.fill_box`'s
own docstring is explicit about what that costs: "a `fill_box` before an `add_shape` leaves
fluid inside the hull". So the four floating cases start with fluid particles inside the
hull's outer surface, some of them on the **same lattice sites** as its boundary shell --
`_commit`'s `_distinct` de-duplicates the fluid against itself and the boundary against
itself, never one against the other. `--order fill-first` is that configuration, which is
the one being diagnosed. `--order hull-first` builds the same tank with the hull added
before the fill, so those sites are dropped, and is the control: the difference between the
two curves is what the construction order costs and nothing else.

## What is reported and what is not

The reference denominator is reported three ways, because at a draft of one particle
spacing they are not the same number and picking one silently would decide the answer:

* `V_geom`, the prescribed draft times the waterplane, which is what the brief asks for;
* `V_level`, the same thing measured from `_waterline`'s median over columns the hull is
  not standing in -- a hull displaces its own volume and a three-beam tank is only eight
  waterplanes wide, so the level sits about `d/8` above the fill line and the geometric
  draft is not the wetted one;
* `V_latt`, the hull's own lattice sites below that level, which is the volume the solver
  can actually push on. `sph.volume()` says why this is the honest denominator: a shape's
  surface does not fall on the lattice, so grading a sampled surface against an analytic
  volume grades the sampler. At four particles through the draft that is a 25% knob.

Nothing here is graded. It is a diagnostic and it prints what it measured.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DRAFT_FRACTIONS = (0.125, 0.25, 0.5, 0.75, 1.0)
"""Fractions of the hull's height, barely submerged to fully submerged.

0.5 is `run_sph_static_float`'s own target -- Archimedes at a density ratio of 0.5 -- so
the middle point of this curve is the number that case is failing to reach, measured with
the body held instead of released.
"""


# -- the measurement ---------------------------------------------------------------------


def _lattice_volume_below(world, centre, half, level: float) -> float:
    """The hull's own lattice sites below `level`, as a volume in m^3.

    `sph._lattice`'s formula, re-derived here rather than imported, because `_lattice`
    takes a box and returns every site in it and what is wanted is a count under a plane.
    Same expression: `origin + (i + 1/2) * spacing`, one global lattice for the fluid and
    every boundary shell.
    """
    import numpy as np

    origin = np.asarray(world.domain.origin, dtype=np.float64)
    spacing = world.spacing
    axes = []
    for a in range(3):
        lo = (centre[a] - half[a] - origin[a]) / spacing - 0.5
        hi = (centre[a] + half[a] - origin[a]) / spacing - 0.5
        axes.append(np.arange(int(math.ceil(lo - 1e-9)), int(math.floor(hi + 1e-9)) + 1))
    if any(len(a) == 0 for a in axes):
        return 0.0
    counts = [len(a) for a in axes]
    z = origin[2] + (axes[2] + 0.5) * spacing
    wet = int((z <= level).sum())
    return counts[0] * counts[1] * wet * spacing ** 3


def _fluid_inside(world, centre, half) -> int:
    """Fluid particles inside the hull's outer surface. The construction-order defect."""
    import numpy as np

    pos = world.positions()
    return int(np.all(np.abs(pos - np.asarray(centre)) <= np.asarray(half), axis=1).sum())


def _build(sc, box, *, order: str, particle_radius: float, depth: float, span: float,
           freeboard: float, start_z: float, device, solver: str, engine: str):
    """The tank and the hull, in one order or the other. Returns `(world, link, body, plan)`.

    `fill-first` is `_floating_tank` then `_float_rig`, which is what all four floating
    cases do. `hull-first` is the same tank with the fill moved after the hull, which is
    the order `sph.fill_box`'s docstring asks for; it is assembled here rather than by
    `_floating_tank` because that helper fills, and this file does not edit it.
    """
    from openreynolds.live import sph as sph_module
    from openreynolds.live.world import Domain

    if order == "fill-first":
        world = sc._floating_tank(box, particle_radius=particle_radius, depth=depth,
                                  span=span, freeboard=freeboard, device=device,
                                  solver=solver)
        link, body, plan = sc._float_rig(box, world, start_z=start_z, solver=engine)
        return world, link, body, plan

    domain = Domain(origin=(0.0, 0.0, 0.0),
                    size=(span * box.beam, span * box.length, depth + freeboard),
                    faces="wall")
    world = sph_module.SphWorld(domain=domain, particle_radius=particle_radius,
                                fluid="water", gravity=(0.0, 0.0, -sc.GRAVITY),
                                surface_tension=0.0, device=device, solver=solver)
    world.boundary_from_domain()
    link, body, plan = sc._float_rig(box, world, start_z=start_z, solver=engine)
    world.fill_below(depth, axis=2)
    return world, link, body, plan


def one_draft(*, draft: float, order: str = "fill-first", particle_radius: float = 0.002,
              beam: float = 0.064, height: float = 0.032, density_ratio: float = 0.5,
              depth: float = 0.06, span: float = 3.0, freeboard: float = 0.04,
              settle_steps: int = 6000, samples: int = 400, device=None,
              solver: str = "dfsph", engine: str = "mujoco", progress=print) -> dict:
    """One point of the curve: hold the hull at `draft`, settle, weigh the water.

    **Newton is never stepped.** `link.step()` is not called once -- not to settle and not
    to sample. `world.step()` advances the fluid, `pose_boundary` poses the hull's Akinci
    particles from a `body_q` that nothing is integrating, and `world.wrench(body)` is the
    mean fluid wrench over the last coupling step. That is the held-body fix of commit
    84a2329 taken all the way: the body is held by not being integrated, so there is no
    drift to correct and no hold to get wrong. (Verified rather than assumed -- the
    boundary particles' bounding box is reported, and a hull that had collapsed to the
    identity transform would show it there and not in the physics.)
    """
    import numpy as np

    from openreynolds.live import sph_cases as sc
    from openreynolds.live.units import FLUIDS

    box = sc._Box(beam=beam, length=beam, height=height, density_ratio=density_ratio)
    # The waterline the draft is prescribed against is the fill line. The hull's bottom
    # goes `draft` below it, so its centre goes half a height above its bottom.
    bottom = depth - draft
    start_z = bottom + height / 2.0
    world, link, body, plan = _build(sc, box, order=order,
                                     particle_radius=particle_radius, depth=depth,
                                     span=span, freeboard=freeboard, start_z=start_z,
                                     device=device, solver=solver, engine=engine)
    centre = (plan[0], plan[1], start_z)
    half = (beam / 2.0, beam / 2.0, height / 2.0)
    spacing = 2.0 * particle_radius
    started_with = len(world.positions())
    inside_at_start = _fluid_inside(world, centre, half)
    progress(f"  draft {draft * 1e3:6.2f} mm ({draft / spacing:.1f} spacings), "
             f"{order}: {started_with:,} fluid, {world.n_boundary:,} boundary, "
             f"{inside_at_start:,} fluid inside the hull, "
             f"{world.duplicates:,} de-duplicated")

    celerity = math.sqrt(sc.GRAVITY * depth)
    settled = sc._settle(world, reference_speed=celerity, max_steps=int(settle_steps),
                         step=world.step, progress=progress)

    series: list[float] = []
    levels: list[float] = []
    for index in range(int(samples)):
        world.step()
        force, _torque = world.wrench(body)
        series.append(float(np.asarray(force, dtype=float).ravel()[2]))
        if index % max(1, int(samples) // 20) == 0:
            levels.append(sc._waterline(world, box, plan, spacing))
    levels.append(sc._waterline(world, box, plan, spacing))

    raw = sc.noise_fraction(series, window=1)
    smoothed = sc.noise_fraction(series, window=10)
    level = float(np.nanmedian(levels))
    rho = FLUIDS["water"].rho
    waterplane = beam * beam

    def weight(volume: float) -> float:
        return rho * volume * sc.GRAVITY

    v_geom = waterplane * min(draft, height)
    v_level = waterplane * min(max(level - bottom, 0.0), height)
    v_latt = _lattice_volume_below(world, centre, half, level)

    bdy = world.boundary_positions(body)
    error = world.density_error()
    return {
        "order": order, "draft": draft, "draft_spacings": draft / spacing,
        "level": level, "fill_level": depth, "bottom": bottom, "start_z": start_z,
        "settled": bool(settled["settled"]), "reached": settled["reached"],
        "bound": settled["bound"], "settle_steps": settled["steps"],
        "fz": raw["mean"], "rms": raw["rms"], "fraction": raw["fraction"],
        "fraction_10": smoothed["fraction"], "samples": len(series),
        "v_geom": v_geom, "v_level": v_level, "v_latt": v_latt,
        "f_geom": weight(v_geom), "f_level": weight(v_level), "f_latt": weight(v_latt),
        "ratio_geom": raw["mean"] / weight(v_geom) if v_geom else float("nan"),
        "ratio_level": raw["mean"] / weight(v_level) if v_level else float("nan"),
        "ratio_latt": raw["mean"] / weight(v_latt) if v_latt else float("nan"),
        "sampled_volume": world.volume(body), "analytic_volume": box.volume,
        "fluid_inside_at_start": inside_at_start,
        "fluid_inside_at_end": _fluid_inside(world, centre, half),
        "fluid_particles": started_with,
        "fluid_particles_at_end": len(world.positions()),
        "boundary_particles": int(world.n_boundary),
        "duplicates": int(world.duplicates),
        "density_error_max": error["max"], "density_error_mean": error["mean"],
        "hull_box_min": bdy.min(axis=0).tolist() if len(bdy) else None,
        "hull_box_max": bdy.max(axis=0).tolist() if len(bdy) else None,
        "hull_asked_at": list(centre),
        "dt": sc._timestep(world), "spacing": spacing, "device": str(world.device),
        "series": series,
    }


def curve(*, fractions=DRAFT_FRACTIONS, height: float = 0.032, order: str = "fill-first",
          progress=print, on_row=None, **settings) -> list[dict]:
    """The five points, with each one handed to `on_row` as soon as it exists.

    `on_row` is not a hook looking for a use. A settle at this resolution is ten minutes
    of GPU and the five points are an hour, and the first attempt at this run lost all of
    it to a **local** DNS failure: `modal run` holds the app open with a heartbeat, the
    heartbeat could not resolve a hostname, and Modal tore down two containers that were
    four thousand steps into useful work. A point that has been measured should survive
    the client that asked for it.
    """
    rows = []
    for fraction in fractions:
        rows.append(one_draft(draft=fraction * height, height=height, order=order,
                              progress=progress, **settings))
        if on_row is not None:
            on_row(rows)
        progress(table(rows))
    return rows


def table(rows: list[dict]) -> str:
    """The five-point curve, and the three denominators it could be divided by."""
    if not rows:
        return "no rows"
    head = (f"{'draft':>8} {'sp':>4} {'|u|max':>9} {'settled':>7} "
            f"{'V_geom':>9} {'V_level':>9} {'V_latt':>9} {'F_z':>10} "
            f"{'/geom':>7} {'/level':>7} {'/latt':>7} {'RMS%':>6} {'RMS10%':>7} {'in':>5}")
    lines = [f"\n  {rows[0]['order']}, {rows[0]['boundary_particles']:,} boundary "
             f"particles, spacing {rows[0]['spacing'] * 1e3:g} mm, "
             f"dt {rows[0]['dt'] * 1e6:.2f} us, device {rows[0]['device']}",
             f"  {'mm':>8} {'':>4} {'mm/s':>9} {'':>7} "
             f"{'mm3':>9} {'mm3':>9} {'mm3':>9} {'mN':>10} "
             f"{'x':>7} {'x':>7} {'x':>7} {'':>6} {'':>7} {'n':>5}",
             "  " + head]
    for r in rows:
        lines.append(
            f"  {r['draft'] * 1e3:8.2f} {r['draft_spacings']:4.1f} "
            f"{r['reached'] * 1e3:9.3f} {str(r['settled']):>7} "
            f"{r['v_geom'] * 1e9:9.0f} {r['v_level'] * 1e9:9.0f} {r['v_latt'] * 1e9:9.0f} "
            f"{r['fz'] * 1e3:10.3f} "
            f"{r['ratio_geom']:7.3f} {r['ratio_level']:7.3f} {r['ratio_latt']:7.3f} "
            f"{r['fraction'] * 100:6.2f} {r['fraction_10'] * 100:7.2f} "
            f"{r['fluid_inside_at_start']:5d}")
    lines.append(f"  settle bound {rows[0]['bound'] * 1e3:.3f} mm/s "
                 f"(2% of the gravity-wave speed), "
                 f"{rows[0]['samples']} force samples per point")
    lines.append(f"  the force is the fluid's, with Newton never stepped: "
                 f"world.step() only, wrench() read after each")
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    """What the curve's shape says, stated as a shape and not as a conclusion."""
    import numpy as np

    if len(rows) < 3:
        return "too few points to call a shape"
    ratio = np.array([r["ratio_latt"] for r in rows], dtype=float)
    fz = np.array([r["fz"] for r in rows], dtype=float)
    reference = np.array([r["f_latt"] for r in rows], dtype=float)
    noise = np.array([r["fraction_10"] for r in rows], dtype=float)
    good = np.isfinite(ratio) & np.isfinite(fz)
    if good.sum() < 3:
        return "too few finite points to call a shape"
    ratio, fz, reference = ratio[good], fz[good], reference[good]
    slope, offset = np.polyfit(reference, fz, 1)
    spread = float(ratio.max() - ratio.min())
    lines = [
        f"  ratio against the lattice volume: {ratio.min():.3f} to {ratio.max():.3f}, "
        f"mean {ratio.mean():.3f}, spread {spread:.3f}",
        f"  F_z against rho g V as a line: slope {slope:.4f}, "
        f"offset {offset * 1e3:+.4f} mN "
        f"({offset / max(abs(fz).max(), 1e-12):+.2%} of the largest force)",
        f"  ten-sample RMS over the five points: {noise[np.isfinite(noise)].min() * 100:.2f}"
        f"% to {noise[np.isfinite(noise)].max() * 100:.2f}% of the mean",
    ]
    return "\n".join(lines)


# -- Modal ------------------------------------------------------------------------------
#
# Its own app and image rather than a function added to `modal_run.py`: this is a
# diagnostic that returns a curve, not a case that returns a graded result, and
# `modal_run.py` is held by other work. The image is that file's, pinned the same way and
# for the same reason -- a diagnostic whose dependency floats cannot be compared with the
# case it is diagnosing.

try:
    import modal
except ImportError:  # the container does not need the local entrypoint
    modal = None

if modal is not None:
    # `parents[2]` and not a module-scope constant: **this module is imported twice**,
    # once locally at `benchmarks/live/draft_curve.py` and once inside the container,
    # where Modal has put the entrypoint at `/root/draft_curve.py` and `parents[2]` is an
    # `IndexError` during startup, before a kernel compiles. `modal_run.py::_local_repo`
    # documents this exactly and it still cost one run to rediscover.
    _here = Path(__file__).resolve()
    REPO = _here.parents[2] if len(_here.parents) > 2 else Path("/repo")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("warp-lang==1.19.0", "newton==1.8.0", "mujoco-warp==4.0.1",
                     "numpy>=1.24", "matplotlib>=3.7", "imageio>=2.31", "pillow>=10")
        .env({"WARP_CACHE_PATH": "/tmp/warp-cache", "MPLCONFIGDIR": "/tmp/mpl",
              "PYTHONUNBUFFERED": "1"})
        .add_local_dir(str(REPO), remote_path="/repo",
                       ignore=["studies/**", ".git/**", "**/__pycache__/**", "*.log",
                               "launcher/**", ".pytest_cache/**",
                               "benchmarks/live/results/**", "docker/**"],
                       copy=True)
    )
    app = modal.App("openreynolds-draft-curve", image=image)

    SINK = "openreynolds-draft-curve"
    """A `modal.Dict` the remote writes each finished point into.

    The durable half of the lesson above: the return value of a `modal run` reaches the
    client over the same connection as the heartbeat that killed the first attempt, so it
    is not where an hour of GPU should live. `--detach` keeps the containers running when
    the client dies and this keeps their results; `--fetch` reads them back later from a
    process that has nothing to do with the run.
    """

    def _remote(settings: dict) -> dict:
        sys.path.insert(0, "/repo")
        import traceback

        import warp as wp

        wp.init()
        device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
        sink = modal.Dict.from_name(SINK, create_if_missing=True)
        order = settings.get("order", "fill-first")

        def keep(rows: list[dict]) -> None:
            sink[order] = {"rows": rows, "device": device, "settings": settings,
                           "complete": False}

        try:
            rows = curve(device=device, progress=print, on_row=keep, **settings)
        except Exception:
            payload = {"error": traceback.format_exc(), "rows": [], "device": device}
            sink[f"{order}-error"] = payload
            return payload
        payload = {"rows": rows, "device": device, "settings": settings,
                   "complete": True}
        sink[order] = payload
        return payload

    @app.function(gpu="H100", timeout=14400)
    def on_h100(settings: dict) -> dict:
        return _remote(settings)

    @app.function(gpu="A100-80GB", timeout=14400)
    def on_a100(settings: dict) -> dict:
        return _remote(settings)

    @app.function(gpu="L40S", timeout=14400)
    def on_l40s(settings: dict) -> dict:
        return _remote(settings)

    RUNNERS = {"h100": on_h100, "a100": on_a100, "l40s": on_l40s}

    @app.local_entrypoint()
    def main(gpu: str = "h100", order: str = "both", radius: float = 0.002,
             settle: int = 10000, samples: int = 400, depth: float = 0.06,
             span: float = 3.0, out: str = "", fetch: bool = False):
        if fetch:
            sink = modal.Dict.from_name(SINK, create_if_missing=True)
            done = {}
            for key in sink.keys():
                payload = sink[key]
                done[key] = payload
                if payload.get("error"):
                    print(f"\n{key} raised:\n{payload['error']}")
                    continue
                print(f"\n{'=' * 78}\n{key} on {payload.get('device')} "
                      f"({'complete' if payload.get('complete') else 'PARTIAL'})"
                      f"\n{'=' * 78}")
                print(table(payload["rows"]))
                print(verdict(payload["rows"]))
            target = Path(out) if out else None
            if target:
                target.write_text(json.dumps(done, indent=2, default=str) + "\n",
                                  encoding="utf-8")
                print(f"\n  saved: {target}")
            return
        orders = ["fill-first", "hull-first"] if order == "both" else [order]
        base = {"particle_radius": radius, "settle_steps": settle, "samples": samples,
                "depth": depth, "span": span}
        runner = RUNNERS[gpu]
        done = {}
        for name, payload in zip(
                orders, runner.map([{**base, "order": o} for o in orders])):
            done[name] = payload
            if payload.get("error"):
                print(f"\n{name} raised:\n{payload['error']}")
                continue
            rows = payload["rows"]
            print(f"\n{'=' * 78}\n{name} on {payload['device']}\n{'=' * 78}")
            print(table(rows))
            print(verdict(rows))
        target = Path(out) if out else (Path("/tmp") / "draft_curve.json")
        try:
            target.write_text(json.dumps(done, indent=2, default=str) + "\n",
                              encoding="utf-8")
            print(f"\n  saved: {target}")
        except OSError as exc:
            print(f"\n  not saved ({exc})")


# -- local ------------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--order", default="both",
                        choices=["fill-first", "hull-first", "both"])
    parser.add_argument("--radius", type=float, default=0.002)
    parser.add_argument("--settle", type=int, default=6000)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--depth", type=float, default=0.06)
    parser.add_argument("--span", type=float, default=3.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    orders = (["fill-first", "hull-first"] if args.order == "both" else [args.order])
    done = {}
    for order in orders:
        rows = curve(order=order, particle_radius=args.radius, settle_steps=args.settle,
                     samples=args.samples, depth=args.depth, span=args.span,
                     device=args.device)
        done[order] = {"rows": rows}
        print(f"\n{'=' * 78}\n{order}\n{'=' * 78}")
        print(table(rows))
        print(verdict(rows))
    if args.out:
        Path(args.out).write_text(json.dumps(done, indent=2, default=str) + "\n",
                                 encoding="utf-8")
        print(f"\n  saved: {args.out}")


def demo() -> None:
    """The one check this file leaves behind: the denominators, not the solver.

    `_lattice_volume_below` is the only non-trivial arithmetic here that is not somebody
    else's tested code, and it is the denominator of every ratio this file prints -- so a
    sign or an off-by-one in it would move the whole curve. Checked against a hand count
    on a lattice whose answer can be written down.
    """
    import numpy as np

    class _Domain:
        origin = (0.0, 0.0, 0.0)

    class _World:
        domain = _Domain()
        spacing = 0.01

    world = _World()
    # A 0.04 cube centred at (0.05, 0.05, 0.05) on a 0.01 lattice with sites at
    # 0.005 + 0.01 k: x and y hold sites 0.035..0.065, which is four of them; z likewise.
    centre, half = (0.05, 0.05, 0.05), (0.02, 0.02, 0.02)
    cell = 0.01 ** 3
    whole = _lattice_volume_below(world, centre, half, 1.0)
    assert abs(whole - 64 * cell) < 1e-15, whole / cell
    # Dry: the lowest site is at 0.035, so a level below it wets nothing.
    assert _lattice_volume_below(world, centre, half, 0.03) == 0.0
    # Half: a level at 0.05 wets the two site planes at 0.035 and 0.045.
    assert abs(_lattice_volume_below(world, centre, half, 0.05) - 32 * cell) < 1e-15
    # One plane: a level at 0.04 wets only 0.035.
    assert abs(_lattice_volume_below(world, centre, half, 0.04) - 16 * cell) < 1e-15
    # Outside the lattice altogether.
    assert _lattice_volume_below(world, (0.05, 0.05, 0.05), (0.001, 0.02, 0.02),
                                 1.0) == 0.0

    # `table` and `verdict` must survive a row set with a nan in it, because a draft that
    # does not settle produces one and the run must still print.
    rows = [{"order": "demo", "draft": 0.004 * k, "draft_spacings": k, "level": 0.06,
             "reached": 0.01, "settled": True, "bound": 0.0153, "spacing": 0.004,
             "dt": 1e-4, "device": "cpu", "boundary_particles": 100, "samples": 10,
             "v_geom": 1e-6 * k, "v_level": 1e-6 * k, "v_latt": 1e-6 * k,
             "f_latt": 9.81e-3 * k, "fz": 9.81e-3 * k,
             "ratio_geom": 1.0, "ratio_level": 1.0, "ratio_latt": 1.0,
             "fraction": 0.129, "fraction_10": 0.0129, "fluid_inside_at_start": 0}
            for k in (1, 2, 3)]
    rows[1]["ratio_geom"] = float("nan")
    assert "demo" in table(rows)
    assert "slope" in verdict(rows)
    print("draft_curve demo: the denominators and the table are right")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        _cli()
