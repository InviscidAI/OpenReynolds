#!/usr/bin/env python3
"""One settle, then every remedy for the force noise released from the SAME water.

    # wiring, on this machine's CPU -- the physics at this spacing is meaningless
    python -X utf8 benchmarks/live/force_noise.py --radius 0.008 --settle 1500 --release 200

    # the measurement
    MODAL_PROFILE=inviscid python -m modal run --detach benchmarks/live/force_noise.py
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/force_noise.py --fetch

## What this file is for

`benchmarks/live/free_settle.py` measures one configuration: settle to the bound, release
the hull, report the per-step force. It is the right shape for a *finding* and the wrong
shape for a *comparison*, because the settle is 19 of its 22 minutes and every variant
pays for it again. Five remedies, each repeated twice, is four hours of H100 for forty
minutes of release window.

So the settle happens once and the release happens many times from the state it left.
`_snapshot` keeps the fluid's positions and velocities; `_restore` writes them back and
puts the hull at rest at the pose it was built at, through `NewtonCoupling.place`, which
is the one way to move a body that `SolverMuJoCo` agrees with (`couple.place_body` has
the four-retracted-measurements story). Every variant therefore starts from **literally
the same water**, which `free_settle.py` run five times cannot promise: two of its runs
settled 1.9e-3 apart in draft.

The settle gate is `free_settle.py`'s and is just as hard: `_settle` in slices, and if
the bound is not met nothing is released and no noise number is reported.

## The variants, and the one that is a diagnostic rather than a remedy

`frozen_walls` is not a physics option and is not offered on `SphWorld`. It wraps
`world.step` to zero `world.body_vel` first, so the Akinci boundary particles stay posed
to the hull and moving with it in *position* while their *velocity* reads zero -- which
is exactly the held hull's condition on a hull Newton is integrating. `PROGRESS_LIVE.md`
names that term as the one open candidate for the gap between 23.0% held and 101.5% free,
and this is the one-line experiment that settles it. It is wrong as a remedy -- a heaving
hull with no surface velocity has no added mass and drags no fluid -- so it lives here,
in the file that measures it, and not in the solver.

## What is measured, and why the filter needs no second run

Every variant keeps `z`, `vz` and `Fz` for **every step** (`free_settle.py`'s docstring
has the 1.4736-against-0.6377 sampling trap that bought that rule), so the per-step RMS,
the near-zero fraction and the mean all come off the full series.

The low-pass filter is then measured **from the same series**, by running the EMA over it
offline at several alphas. `SphWorld(wrench_filter=)` is the same recurrence driven once
a step, and it changes nothing the solver does -- `couple.py` reads the raw device wrench
-- so a filtered run and an unfiltered one produce the same trajectory and the same raw
series by construction. Measuring the filter offline is therefore not an approximation to
a filtered run; it is the same arithmetic on the same numbers, and it costs no GPU.
`white_equivalent` is what makes the coherence claim checkable: the reduction a
ten-sample average of *white* noise would give is sqrt(10) = 3.16, and anything much
larger is a signal with structure in it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

SINK = "openreynolds-force-noise"

VARIANTS: dict[str, dict] = {
    # name                     what it changes
    "base": {},
    "frozen-walls": {"frozen_walls": True},
    "tol-1e-4": {"tolerance": 1.0e-4},
    "tol-1e-5": {"tolerance": 1.0e-5},
    "delta-0.012": {"density_diffusion": 0.012},
    "delta-0.004": {"density_diffusion": 0.004},
    "dt-half": {"dt_scale": 0.5},
    "dt-third": {"dt_scale": 1.0 / 3.0},
    "dt-quarter": {"dt_scale": 0.25},
    "dt-sixth": {"dt_scale": 1.0 / 6.0},
    "dt-eighth": {"dt_scale": 0.125},
    "delta+tol": {"density_diffusion": 0.004, "tolerance": 1.0e-4},
    "held": {"held": True},
    "held-delta-0.012": {"held": True, "density_diffusion": 0.012},
    "held-dt-quarter": {"held": True, "dt_scale": 0.25},
    "base-again": {},
}
"""The release windows, in the order they are run. `base` is first and is the control.

`delta-0.012` is 88% of the ceiling `SphWorld.diffusivity` derives for the default
tolerance (0.429 sqrt(tolerance) = 0.01357) and `delta-0.004` is 29% of it, so the pair
brackets the usable range rather than sampling one point in it. `delta+tol` exists
because the two knobs are **not independent**: a tighter DFSPH tolerance raises the
equivalent sound speed, which is this term's prefactor, so `tolerance=1e-4` drops the
ceiling to 0.00429 and 0.004 is what fits under both at once.

`dt_scale` is a scale on the coupling step, not on the substep: the whole suspicion about
the released hull is that the coupling is explicit -- the fluid sees `body_vel` from the
start of the step and the body sees the wrench from the end of it -- so the lag to halve
is the coupling's. One substep each, so the fluid's own CFL number halves with it.

`base-again` is `base` a second time, last, and it is the control on this file's own
method: eleven windows run in sequence from one snapshot, each restoring the fluid and
re-placing the hull, and if that restore leaks anything then the last window will not
reproduce the first. A difference between `base` and `base-again` is this harness, not
the solver, and it bounds every other row here.

`held` advances `world.step` instead of `link.step`, so Newton is never stepped and the
hull stays pinned at the Archimedes draft -- `draft_curve.py`'s and `free_settle.py`'s
held configuration, which measures 23.0%. It is here so that a remedy can be weighed
against the fluid alone *and* against the coupling: a term that moves the held number and
not the free one is doing something to the density field, and one that moves neither is
not the mechanism.
"""


def _series_stats(fz, weight: float, dt: float) -> dict:
    """RMS, near-zero fraction and the filter's whole table, off one per-step series."""
    import numpy as np

    fz = np.asarray(fz, dtype=float)
    mean = float(fz.mean())
    out = {
        "steps": int(len(fz)),
        "fz_mean": mean,
        "fz_mean_over_mg": mean / weight,
        "fz_rms_fraction": float(fz.std() / abs(mean)) if mean else float("nan"),
        "fz_rms_over_mg": float(fz.std() / weight),
        "fz_near_zero_fraction": float(np.mean(np.abs(fz) < 0.01 * weight)),
        "fz_min_over_mg": float(fz.min() / weight),
        "fz_max_over_mg": float(fz.max() / weight),
    }
    # The filter, offline, on exactly this series. One pass per alpha; the `n` column is
    # the equivalent averaging length `(2 - alpha)/alpha`, which is what a sqrt(n) white
    # expectation has to be compared against.
    out["filter"] = []
    for alpha in (0.5, 0.3, 0.2, 0.1, 0.05):
        filtered = np.empty_like(fz)
        state = fz[0]
        for index, sample in enumerate(fz):
            state = sample if index == 0 else alpha * sample + (1.0 - alpha) * state
            filtered[index] = state
        # Drop the filter's own start-up transient: `(1-alpha)^k` falls below 1% after
        # `ln(100)/-ln(1-alpha)` samples, so skip five times the time constant.
        warm = int(min(len(fz) // 4, math.ceil(5.0 / alpha)))
        tail = filtered[warm:]
        samples = (2.0 - alpha) / alpha
        out["filter"].append({
            "alpha": alpha,
            "equivalent_samples": samples,
            "lag_steps": (1.0 - alpha) / alpha,
            "lag_ms": (1.0 - alpha) / alpha * dt * 1e3,
            "corner_hz": alpha / (2.0 * math.pi * (1.0 - alpha) * dt),
            "fz_mean_over_mg": float(tail.mean() / weight),
            "fz_rms_fraction": float(tail.std() / abs(tail.mean()))
            if tail.mean() else float("nan"),
            "reduction": float(fz.std() / tail.std()) if tail.std() else float("inf"),
            "white_equivalent": math.sqrt(samples),
            "peak_over_mg": float(np.abs(tail).max() / weight),
        })
    return out


def one_run(*, particle_radius: float = 0.002, beam: float = 0.064,
            height: float = 0.032, density_ratio: float = 0.5, depth: float = 0.06,
            span: float = 3.0, freeboard: float = 0.04, settle_steps: int = 200_000,
            settle_slice: int = 2_000, release_steps: int = 0,
            variants: list[str] | None = None, device=None, engine: str = "mujoco",
            keep=None, progress=print) -> dict:
    """Settle once to the bound, then release each variant from that same state."""
    import numpy as np

    from openreynolds.live import sph as sph_module
    from openreynolds.live import sph_cases as sc
    from openreynolds.live.world import Domain

    from free_settle import mass_conserving_draft

    names = list(variants) if variants else list(VARIANTS)
    box = sc._Box(beam=beam, length=beam, height=height, density_ratio=density_ratio)
    start_z = depth - box.draft + height / 2.0

    # Hull before water, the order the four floating cases use. Same as
    # `free_settle.one_run`, deliberately: the comparison against its 101.5% is only a
    # comparison if the tank is the same tank.
    domain = Domain(origin=(0.0, 0.0, 0.0),
                    size=(span * beam, span * beam, depth + freeboard), faces="wall")
    world = sph_module.SphWorld(domain=domain, particle_radius=particle_radius,
                                fluid="water", gravity=(0.0, 0.0, -sc.GRAVITY),
                                surface_tension=0.0, device=device)
    world.boundary_from_domain()
    link, body, plan = sc._float_rig(box, world, start_z=start_z, solver=engine)
    world.fill_below(depth, axis=2)

    dt0 = sc._timestep(world)
    celerity = math.sqrt(sc.GRAVITY * depth)
    bound = 0.02 * celerity
    mass = float(link.model.body_mass.numpy()[body])
    weight = mass * sc.GRAVITY
    period = sc.heave_period(box.draft)
    spacing = 2.0 * particle_radius
    held_pose = tuple(float(v) for v in link.state.body_q.numpy()[body])

    out: dict = {
        "config": {
            "particle_radius": particle_radius, "beam": beam, "height": height,
            "density_ratio": density_ratio, "depth": depth, "span": span,
            "freeboard": freeboard, "order": "hull-first", "engine": engine,
            "dt": dt0, "substeps": int(world.substeps),
            "fluid_particles": int(len(world.positions())),
            "boundary_particles": int(world.n_boundary),
            "duplicates": int(world.duplicates),
            "spacings_through_draft": box.draft / spacing,
            "device": str(world.device), "archimedes_draft": box.draft,
            "start_z": start_z, "body_mass": mass, "weight": weight,
            "heave_period": period, "settle_bound": bound, "settle_cap": settle_steps,
            "delta_ceiling": 0.429 * math.sqrt(world.tolerance),
            "variants": names,
        },
        "settle_trace": [], "settled": False, "reached": None, "settle_steps": 0,
        "runs": {}, "complete": False,
    }
    cfg = out["config"]
    progress(f"  {cfg['fluid_particles']:,} fluid, {cfg['boundary_particles']:,} "
             f"boundary, dt {dt0 * 1e6:.3f} us, "
             f"{cfg['spacings_through_draft']:.1f} spacings through the draft")
    progress(f"  weight {weight * 1e3:.4f} mN, Archimedes draft "
             f"{box.draft * 1e3:.4f} mm, heave period {period * 1e3:.2f} ms, "
             f"settle bound {bound * 1e3:.3f} mm/s")
    progress(f"  delta ceiling at this tolerance: "
             f"{cfg['delta_ceiling']:.5f};  variants: {', '.join(names)}")

    warm = time.perf_counter()
    for _ in range(10):
        world.step()
    cfg["steps_per_second"] = 10.0 / (time.perf_counter() - warm)
    progress(f"  {cfg['steps_per_second']:.2f} world steps/s, so the "
             f"{settle_steps:,}-step cap is "
             f"{settle_steps / cfg['steps_per_second'] / 3600:.2f} h")
    if keep:
        keep(out)

    # -- settle, in slices, through the graded helper. `world.step`, so Newton is never
    #    stepped and the hull stays pinned at the Archimedes draft it was built at.
    done = 0
    while done < settle_steps:
        chunk = min(int(settle_slice), settle_steps - done)
        slice_result = sc._settle(world, reference_speed=celerity, max_steps=chunk,
                                  check_every=min(100, chunk), step=world.step)
        done += slice_result["steps"]
        out["settle_trace"].append({"steps": done, "reached": slice_result["reached"]})
        out["settle_steps"] = done
        out["reached"] = slice_result["reached"]
        progress(f"    settling {done:,}/{settle_steps:,}  |u|max "
                 f"{slice_result['reached'] * 1e3:8.3f} mm/s against {bound * 1e3:.3f}")
        if keep:
            keep(out)
        if slice_result["settled"]:
            out["settled"] = True
            break

    if not out["settled"]:
        out["complete"] = True
        out["verdict"] = (
            f"NOT SETTLED: |u|max reached {out['reached'] * 1e3:.3f} mm/s after "
            f"{done:,} steps against a bound of {bound * 1e3:.3f} mm/s "
            f"({out['reached'] / bound:.2f}x). No force noise is reported.")
        progress(f"  {out['verdict']}")
        if keep:
            keep(out)
        return out

    settled_x = world.x.numpy().copy()
    settled_v = world.v.numpy().copy()
    progress(f"  settled at {out['reached'] * 1e3:.3f} mm/s after {done:,} steps; "
             f"snapshot taken, releasing {len(names)} variants from it")

    for name in names:
        knobs = dict(VARIANTS[name])
        frozen = bool(knobs.pop("frozen_walls", False))
        held = bool(knobs.pop("held", False))
        dt_scale = float(knobs.pop("dt_scale", 1.0))

        # Restore: the fluid first, then the hull at rest at the pose it was settled
        # under. `link.place` goes through `joint_q` and re-pushes the poses, which is
        # the only way `SolverMuJoCo` agrees a body has moved.
        world.x.assign(settled_x)
        world.v.assign(settled_v)
        link.place(body, held_pose)
        world._wrench.zero_()
        world._wrench_mean = None

        # The knobs. All of them are plain attributes read at launch time, so a variant
        # is a few assignments and not a rebuilt world -- which is the whole point:
        # rebuilding the world would mean settling again.
        world.tolerance = float(knobs.get("tolerance", 1.0e-3))
        # And the equivalent sound speed with it, because `diffusivity` is proportional
        # to it: a tolerance set without this leaves delta-SPH's prefactor at the
        # tolerance the world was BUILT with, and the ceiling `_check` then holds is the
        # wrong one. This is the coupling between the two knobs, in one line.
        world.c0 = world._peak_estimate / math.sqrt(world.tolerance)
        world.density_diffusion = float(knobs.get("density_diffusion", 0.0))
        world.dt = dt0 * dt_scale
        world.substeps = 1
        link.dt = world.dt
        if world.density_diffusion > 0.0:
            if world.rho_c is world.rho:
                import warp as wp
                world.rho_c = wp.zeros(world.n_fluid, dtype=sph_module.DTYPE,
                                       device=world.device)
            # The ceiling is the solver's own; a variant past it is a configuration
            # error and must not quietly run. `_check` is what says so, and it reads
            # the attributes just set.
            world._check()
        else:
            # **Back to the alias, and this is not tidying.** `_prepare` only launches
            # `diffuse_density` when the term is on, so a variant with it off that
            # inherited a separate `rho_c` from the variant before would hand `residual`
            # the PREVIOUS variant's diffused density, frozen, for its whole window --
            # a stale constraint that still looks like a density field. Four of the
            # eleven variants sit after a delta one in this list.
            world.rho_c = world.rho
        real_step = world.step
        if frozen:
            def advance_world(*args, **kwargs):
                # Pose still follows the hull; only the surface VELOCITY is removed.
                world.body_vel.zero_()
                return real_step(*args, **kwargs)
            world.step = advance_world

        total = int(release_steps if release_steps
                    else math.ceil(6.0 * period / world.dt))
        advance = world.step if held else link.step
        progress(f"    {name}: dt {world.dt * 1e6:.3f} us, tolerance "
                 f"{world.tolerance:g}, delta {world.density_diffusion:g}, "
                 f"frozen_walls {frozen}, held {held} -- {total:,} steps")
        began = time.perf_counter()
        trace_z: list[float] = []
        trace_vz: list[float] = []
        trace_fz: list[float] = []
        iterations: list[float] = []
        for index in range(total):
            cost = advance()
            pose = link.state.body_q.numpy()[body]
            trace_z.append(float(pose[2]))
            trace_vz.append(float(link.state.body_qd.numpy()[body][2]))
            trace_fz.append(float(np.asarray(world.wrench(body)[0],
                                             dtype=float).ravel()[2]))
            iterations.append(float(cost.get("iterations", 0.0)))
            if index % max(1, total // 6) == 0:
                progress(f"      {index:,}/{total:,}  z {pose[2] * 1e3:.4f} mm  Fz/mg "
                         f"{trace_fz[-1] / weight:+.4f}  "
                         f"{(time.perf_counter() - began) / max(index, 1) * (total - index) / 60:.1f}"
                         f" min left")
        world.step = real_step

        # The late window, the same 40% tail `free_settle.py` reports on.
        cut = int(0.6 * total)
        z_late = np.asarray(trace_z[cut:], dtype=float)
        vz_late = np.asarray(trace_vz[cut:], dtype=float)
        span_s = max(len(z_late) - 1, 1) * world.dt
        stats = _series_stats(trace_fz[cut:], weight, world.dt)
        keel = float(z_late[-1]) - height / 2.0
        level_mass, draft_mass = mass_conserving_draft(
            fluid_particles=world.n_fluid, particle_radius=particle_radius,
            tank_area=(span * beam) ** 2, waterplane=beam * beam, keel=keel)
        stats.update({
            "dt": world.dt, "tolerance": world.tolerance,
            "density_diffusion": world.density_diffusion, "frozen_walls": frozen,
            "held": held, "release_steps": total,
            "density_iterations": float(np.mean(iterations)),
            "seconds": time.perf_counter() - began,
            # `m z'' = Fz - m g` over the same steps, which is what says a force is the
            # force the body actually felt rather than a number out of a reporting path.
            "accel_from_force": float((stats["fz_mean"] - weight) / mass),
            "accel_from_z": float(2.0 * (z_late[-1] - z_late[0] - vz_late[0] * span_s)
                                  / span_s ** 2) if len(z_late) > 1 else float("nan"),
            "vz_mean": float(vz_late.mean()),
            "vz_from_dz": float((z_late[-1] - z_late[0]) / span_s),
            "z_wobble": float(z_late.std()),
            "draft_mass": draft_mass,
            "draft_mass_over_archimedes": draft_mass / box.draft,
            "level_mass": level_mass,
            "density_error": world.density_error(),
            "speed_max": float(np.nanmax(np.abs(world.velocities()))),
            "series": {"z": trace_z, "vz": trace_vz, "fz": trace_fz},
        })
        out["runs"][name] = stats
        best = min(stats["filter"], key=lambda row: row["fz_rms_fraction"])
        progress(f"    {name}: RMS {stats['fz_rms_fraction']:.1%}, "
                 f"{stats['fz_near_zero_fraction']:.1%} near zero, mean "
                 f"{stats['fz_mean_over_mg']:.4f} mg, draft "
                 f"{stats['draft_mass_over_archimedes']:.4f}x, "
                 f"{stats['density_iterations']:.1f} density iterations; filtered best "
                 f"{best['fz_rms_fraction']:.1%} at alpha {best['alpha']:g}")
        if keep:
            keep(out)

    out["complete"] = True
    control = out["runs"].get("base", {})
    out["verdict"] = (
        f"SETTLED at {out['reached'] * 1e3:.3f} mm/s after {done:,} steps; "
        f"{len(out['runs'])} variants released from that one state. base RMS "
        f"{control.get('fz_rms_fraction', float('nan')):.1%}, best "
        + ", ".join(f"{n} {r['fz_rms_fraction']:.1%}"
                    for n, r in sorted(out["runs"].items(),
                                       key=lambda kv: kv[1]["fz_rms_fraction"])[:3]))
    progress(f"  {out['verdict']}")
    if keep:
        keep(out)
    return out


def table(payload: dict) -> str:
    """The settle trace, then one row per variant, then the filter on the best of them."""
    cfg = payload.get("config", {})
    lines = [
        f"  radius {cfg.get('particle_radius', 0) * 1e3:g} mm, "
        f"{cfg.get('fluid_particles', 0):,} fluid + "
        f"{cfg.get('boundary_particles', 0):,} boundary, "
        f"{cfg.get('spacings_through_draft', 0):.1f} spacings through the draft, "
        f"{cfg.get('device', '?')}",
        f"  dt {cfg.get('dt', 0) * 1e6:.3f} us, weight {cfg.get('weight', 0) * 1e3:.4f} mN, "
        f"Archimedes draft {cfg.get('archimedes_draft', 0) * 1e3:.4f} mm, "
        f"settle bound {cfg.get('settle_bound', 0) * 1e3:.3f} mm/s, "
        f"delta ceiling {cfg.get('delta_ceiling', 0):.5f}",
        "",
        f"  {'settle steps':>14} {'|u|max (mm/s)':>15} {'x bound':>9}",
    ]
    bound = cfg.get("settle_bound") or 1.0
    for row in payload.get("settle_trace", []):
        lines.append(f"  {row['steps']:>14,} {row['reached'] * 1e3:>15.3f} "
                     f"{row['reached'] / bound:>9.2f}")
    runs = payload.get("runs", {})
    if runs:
        lines += ["",
                  f"  {'variant':<14} {'dt (us)':>9} {'RMS':>8} {'<1% mg':>8} "
                  f"{'mean/mg':>9} {'min/mg':>8} {'max/mg':>8} {'iters':>6} "
                  f"{'draft/A':>8} {'rho err':>9}"]
        for name, row in runs.items():
            lines.append(
                f"  {name:<14} {row['dt'] * 1e6:>9.3f} {row['fz_rms_fraction']:>8.1%} "
                f"{row['fz_near_zero_fraction']:>8.1%} {row['fz_mean_over_mg']:>9.4f} "
                f"{row['fz_min_over_mg']:>8.3f} {row['fz_max_over_mg']:>8.3f} "
                f"{row['density_iterations']:>6.1f} "
                f"{row['draft_mass_over_archimedes']:>8.4f} "
                f"{row['density_error']['mean']:>9.3%}")
        lines += ["",
                  "  m z'' = Fz - m g, and the hull's own trace:",
                  f"  {'variant':<14} {'a from F':>10} {'a from z':>10} "
                  f"{'vz mean':>10} {'dz/dt':>10} {'z wobble':>10}"]
        for name, row in runs.items():
            lines.append(
                f"  {name:<14} {row['accel_from_force']:>10.4f} "
                f"{row['accel_from_z']:>10.4f} {row['vz_mean'] * 1e3:>10.4f} "
                f"{row['vz_from_dz'] * 1e3:>10.4f} {row['z_wobble'] * 1e6:>10.3f}")
        for name, row in runs.items():
            lines += ["",
                      f"  the EMA on {name}'s own per-step series "
                      f"(raw RMS {row['fz_rms_fraction']:.1%}):",
                      f"  {'alpha':>7} {'n':>7} {'lag':>7} {'lag ms':>8} "
                      f"{'corner Hz':>10} {'RMS':>8} {'mean/mg':>9} {'cut by':>8} "
                      f"{'white':>7} {'peak/mg':>8}"]
            for f in row["filter"]:
                lines.append(
                    f"  {f['alpha']:>7g} {f['equivalent_samples']:>7.1f} "
                    f"{f['lag_steps']:>7.1f} {f['lag_ms']:>8.2f} "
                    f"{f['corner_hz']:>10.1f} {f['fz_rms_fraction']:>8.1%} "
                    f"{f['fz_mean_over_mg']:>9.4f} {f['reduction']:>8.2f}x "
                    f"{f['white_equivalent']:>7.2f} {f['peak_over_mg']:>8.3f}")
    lines += ["", f"  {payload.get('verdict', 'no verdict')}"]
    return "\n".join(lines)


try:
    import modal
except ImportError:  # the container does not need the local entrypoint
    modal = None

if modal is not None:
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
    app = modal.App("openreynolds-force-noise", image=image)

    def _remote(settings: dict) -> dict:
        sys.path.insert(0, "/repo")
        sys.path.insert(0, "/repo/benchmarks/live")
        import traceback

        import warp as wp

        wp.init()
        device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
        label = settings.pop("label", "run")
        sink = modal.Dict.from_name(SINK, create_if_missing=True)

        def keep(payload: dict) -> None:
            sink[label] = payload

        try:
            return one_run(device=device, progress=print, keep=keep, **settings)
        except Exception:
            payload = {"error": traceback.format_exc(), "config": {"device": device}}
            sink[f"{label}-error"] = payload
            return payload

    @app.function(gpu="H100", timeout=86400)
    def on_h100(settings: dict) -> dict:
        return _remote(settings)

    @app.function(gpu="A100-80GB", timeout=86400)
    def on_a100(settings: dict) -> dict:
        return _remote(settings)

    RUNNERS = {"h100": on_h100, "a100": on_a100}

    @app.local_entrypoint()
    def main(gpu: str = "h100", radius: float = 0.002, settle: int = 200_000,
             chunk: int = 2_000, release: int = 0, variants: str = "",
             label: str = "", out: str = "", fetch: bool = False, keys: str = ""):
        if fetch:
            sink = modal.Dict.from_name(SINK, create_if_missing=True)
            wanted = [k.strip() for k in keys.split(",") if k.strip()]
            done = {}
            for key in sink.keys():
                if wanted and key not in wanted:
                    continue
                payload = sink[key]
                done[key] = payload
                if payload.get("error"):
                    print(f"\n{key} raised:\n{payload['error']}")
                    continue
                print(f"\n{'=' * 78}\n{key} on "
                      f"{payload.get('config', {}).get('device')} "
                      f"({'complete' if payload.get('complete') else 'PARTIAL'})"
                      f"\n{'=' * 78}")
                print(table(payload))
            if out:
                Path(out).write_text(json.dumps(done, indent=2, default=str) + "\n",
                                     encoding="utf-8")
                print(f"\n  saved: {out}")
            return
        chosen = [v.strip() for v in variants.split(",") if v.strip()] or None
        name = label or f"r{radius * 1e3:g}mm-{gpu}"
        payload = RUNNERS[gpu].remote({
            "particle_radius": radius, "settle_steps": settle, "settle_slice": chunk,
            "release_steps": release, "variants": chosen, "label": name})
        if payload.get("error"):
            print(f"\n{name} raised:\n{payload['error']}")
            return
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(table(payload))


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radius", type=float, default=0.008)
    parser.add_argument("--settle", type=int, default=1500)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--release", type=int, default=200)
    parser.add_argument("--variants", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    chosen = [v.strip() for v in args.variants.split(",") if v.strip()] or None
    payload = one_run(particle_radius=args.radius, settle_steps=args.settle,
                      settle_slice=args.chunk, release_steps=args.release,
                      variants=chosen, device=args.device)
    print(table(payload))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                  encoding="utf-8")
        print(f"\n  saved: {args.out}")


def demo() -> None:
    """The two checks this file leaves behind, both on the analysis rather than the run.

    `_series_stats` is where every number in the report comes from, and the two things
    that can be silently wrong in it are the EMA recurrence and the coherence claim --
    the second of which is the whole reason the filter is worth shipping, so it must not
    rest on a number nobody re-derived.

    The bimodal series is the measured one, not a plausible one: `PROGRESS_LIVE.md` has
    the free hull at 101.5% RMS with 4.1% of steps below 1% of `m g`, so the fixture is
    a two-state signal with that duty cycle and that RMS.
    """
    import numpy as np

    weight, dt = 0.6417574, 1142.274e-6

    # A coherent two-state signal: long runs of the high mode and the low one, which is
    # what "bimodal, 4.1% near zero" describes. A ten-sample average cuts coherent noise
    # far less than sqrt(10) when the runs are longer than ten samples, and far more when
    # they alternate every step -- so the measured 10.01x is a statement about the
    # period of the structure, and this fixture must reproduce the ARITHMETIC, not that.
    rng = np.random.default_rng(0)
    alternating = weight * (1.0 + np.where(np.arange(2000) % 2 == 0, 1.0, -1.0))
    stats = _series_stats(alternating, weight, dt)
    assert abs(stats["fz_mean_over_mg"] - 1.0) < 1e-12, stats["fz_mean_over_mg"]
    assert abs(stats["fz_rms_fraction"] - 1.0) < 1e-9, stats["fz_rms_fraction"]
    row = next(f for f in stats["filter"] if f["alpha"] == 0.2)
    # A one-pole EMA at alpha on a +/-1 square wave of period 2 leaves an amplitude of
    # alpha/(2 - alpha): 0.1111 at alpha 0.2, so the reduction is exactly 9x.
    assert abs(row["reduction"] - (2.0 - 0.2) / 0.2) < 0.02, row["reduction"]
    assert abs(row["lag_steps"] - 4.0) < 1e-12
    assert abs(row["lag_ms"] - 4.569096) < 1e-5, row["lag_ms"]
    assert abs(row["corner_hz"] - 34.83) < 0.01, row["corner_hz"]
    print(f"  alternating: RMS {stats['fz_rms_fraction']:.1%}, EMA(0.2) cuts it "
          f"{row['reduction']:.2f}x against a white sqrt(n) of "
          f"{row['white_equivalent']:.2f}, for a {row['lag_steps']:.0f}-step "
          f"({row['lag_ms']:.2f} ms) lag and a {row['corner_hz']:.1f} Hz corner")

    # White noise at the same RMS: the same filter must give sqrt(n) and not 9x, which
    # is the control that makes the coherence claim mean something.
    white = weight * (1.0 + rng.standard_normal(20_000))
    row = next(f for f in _series_stats(white, weight, dt)["filter"]
               if f["alpha"] == 0.2)
    assert abs(row["reduction"] / row["white_equivalent"] - 1.0) < 0.06, (
        f"on white noise the EMA cut the RMS {row['reduction']:.3f}x against the "
        f"sqrt((2-a)/a) = {row['white_equivalent']:.3f} it must: if this drifts, "
        f"`white_equivalent` is not the control the coherence claim is weighed against"
    )
    print(f"  white:       EMA(0.2) cuts it {row['reduction']:.2f}x against "
          f"sqrt(n) = {row['white_equivalent']:.2f} -- the control")

    # The near-zero fraction, on a series built to have one.
    # 960 and not 1000, so the duty cycle is exactly 1/24 rather than 42/1000.
    bimodal = np.where(np.arange(960) % 24 == 0, 0.0, weight * 1.0438)
    stats = _series_stats(bimodal, weight, dt)
    assert abs(stats["fz_near_zero_fraction"] - 1.0 / 24.0) < 1e-9
    assert abs(stats["fz_mean_over_mg"] - 1.0) < 2e-3, stats["fz_mean_over_mg"]
    print(f"  bimodal:     {stats['fz_near_zero_fraction']:.1%} of steps below 1% of "
          f"mg with the mean still at {stats['fz_mean_over_mg']:.4f} -- which is why a "
          f"per-step force must not be sampled sparsely")
    print("demo: the EMA recurrence, its lag, and the white control all hold")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        _cli()
