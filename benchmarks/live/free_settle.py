#!/usr/bin/env python3
"""Release the hull only after the tank has actually settled, and log the force.

    # wiring, on this machine's CPU -- the physics at this spacing is meaningless
    python -X utf8 benchmarks/live/free_settle.py --radius 0.008 --settle 2000 --release 400

    # the measurement
    MODAL_PROFILE=inviscid python -m modal run --detach benchmarks/live/free_settle.py
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/free_settle.py --fetch

## Why this file exists and `run_sph_static_float` is not enough

`_settle` targets 2% of the gravity-wave speed, 15.34 mm/s for this tank. **No SPH
floating measurement in this repository has ever reached it** -- the five-point
prescribed-draft curve finished at 18-33 mm/s at every point against a 10,000-step cap,
and a CPU probe finished at 93 mm/s. `run_sph_static_float` grades that as a failed check
and then reports its draft anyway, beside the failure. The instruction this file exists
to follow is the opposite: **only report a draft once the settle bound is reached**, so
the gate is hard here and a run that does not clear it reports how close it got and no
draft at all.

It also logs what the case does not: the fluid's vertical force against Newton's own
`m * g`, every `--every` steps through the release, with the ratio. That is the number
that says whether a hull which is sitting still is sitting still for the right reason.

## The settle trace, and why it is a trace

`_settle` returns only the velocity it finished at, which cannot distinguish "converging
slowly" from "converged to a floor above the bound". So it is called here in slices and
each slice's `|u|max` is kept -- the same graded function, the same bound, with the decay
visible. A run that plateaus is then a statement about the solver rather than about the
budget, and the trace is what makes that readable.

## The `modal.Dict`, and the DNS blip that paid for it

`draft_curve.py` records it: the return value of a `modal run` reaches the client over
the same connection as the heartbeat, so an hour of H100 must not live there. Every
slice of the settle and every block of the release is written into a `modal.Dict` as it
finishes, so `--detach` plus `--fetch` reads a partial run back out of a process that
has nothing to do with it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

SINK = "openreynolds-free-settle"


def one_run(*, particle_radius: float = 0.002, beam: float = 0.064,
            height: float = 0.032, density_ratio: float = 0.5, depth: float = 0.06,
            span: float = 3.0, freeboard: float = 0.04, settle_steps: int = 200_000,
            settle_slice: int = 2_000, release_steps: int = 0, every: int = 50,
            hold: bool = False, device=None, solver: str = "dfsph",
            engine: str = "mujoco", keep=None, progress=print) -> dict:
    """Settle to the bound, then release, and report the force against `m g` throughout.

    `hold=True` does not release: it advances `world.step` instead of `link.step`, so
    Newton is never stepped and the hull stays exactly at the Archimedes draft it was
    built at, which is `draft_curve.py`'s configuration. **The point of having both here
    is that they are then the same configuration** -- the same tank, the same hull-first
    fill, the same settle to the same bound, the same per-step sampling -- and this
    repository has retracted four findings that came from comparing a held number against
    a free one taken under different conditions. With `hold`, `fz_over_mg` is the force
    the fluid delivers at a draft of exactly `ratio * H`, and `1 / fz_over_mg` is the
    draft the free hull should then settle at.

    `keep` is called with the result-so-far after every settle slice and every release
    block, which is what survives a dead client.
    """
    import numpy as np

    from openreynolds.live import sph as sph_module
    from openreynolds.live import sph_cases as sc
    from openreynolds.live.world import Domain

    box = sc._Box(beam=beam, length=beam, height=height, density_ratio=density_ratio)
    hydro = box.hydrostatics()
    start_z = depth - box.draft + height / 2.0

    # Hull before water, which is `_floating_tank(fill=False)` plus `_float_rig` plus the
    # fill -- the order the four cases now use. Assembled here rather than through the
    # helper only because this file prints the counts between the three steps.
    domain = Domain(origin=(0.0, 0.0, 0.0),
                    size=(span * beam, span * beam, depth + freeboard), faces="wall")
    world = sph_module.SphWorld(domain=domain, particle_radius=particle_radius,
                                fluid="water", gravity=(0.0, 0.0, -sc.GRAVITY),
                                surface_tension=0.0, device=device, solver=solver)
    world.boundary_from_domain()
    link, body, plan = sc._float_rig(box, world, start_z=start_z, solver=engine)
    world.fill_below(depth, axis=2)

    dt = sc._timestep(world)
    celerity = math.sqrt(sc.GRAVITY * depth)
    bound = 0.02 * celerity
    mass = float(link.model.body_mass.numpy()[body])
    weight = mass * sc.GRAVITY
    period = sc.heave_period(box.draft)
    spacing = 2.0 * particle_radius

    out: dict = {
        "config": {
            "particle_radius": particle_radius, "beam": beam, "height": height,
            "density_ratio": density_ratio, "depth": depth, "span": span,
            "freeboard": freeboard, "order": "hull-first", "solver": solver,
            "engine": engine, "dt": dt, "substeps": int(world.substeps),
            "fluid_particles": int(len(world.positions())),
            "boundary_particles": int(world.n_boundary),
            "duplicates": int(world.duplicates),
            "spacings_through_draft": box.draft / spacing,
            "device": str(world.device),
            "archimedes_draft": box.draft, "start_z": start_z,
            "body_mass": mass, "weight": weight, "heave_period": period,
            "settle_bound": bound, "settle_cap": settle_steps,
            "buoyancy": bool(link.buoyancy), "one_way": bool(link.one_way),
            "GM": hydro["GM"],
        },
        "settle_trace": [], "settled": False, "reached": None, "settle_steps": 0,
        "release": [], "draft": None, "wobble": None, "complete": False,
    }
    progress(f"  {out['config']['fluid_particles']:,} fluid, "
             f"{out['config']['boundary_particles']:,} boundary, dt {dt * 1e6:.3f} us, "
             f"{box.draft / spacing:.1f} spacings through the draft")
    progress(f"  weight {weight * 1e3:.4f} mN, Archimedes draft "
             f"{box.draft * 1e3:.4f} mm, heave period {period * 1e3:.2f} ms")
    progress(f"  settle bound {bound * 1e3:.3f} mm/s, cap {settle_steps:,} steps")

    # -- the step rate, measured before anything is budgeted against it ---------------
    warm = time.perf_counter()
    for _ in range(10):
        world.step()
    rate = 10.0 / (time.perf_counter() - warm)
    out["config"]["steps_per_second"] = rate
    progress(f"  {rate:.2f} world steps/s, so the {settle_steps:,}-step cap is "
             f"{settle_steps / rate / 3600:.2f} h")
    if keep:
        keep(out)

    # -- settle, in slices, through the graded helper ---------------------------------
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
                 f"{slice_result['reached'] * 1e3:8.3f} mm/s against "
                 f"{bound * 1e3:.3f}")
        if keep:
            keep(out)
        if slice_result["settled"]:
            out["settled"] = True
            break

    if not out["settled"]:
        # **The hard gate.** A draft measured in a moving tank is the average of a
        # transient, and this repository has retracted two of those. Report the distance
        # to the bound and stop.
        out["complete"] = True
        out["verdict"] = (
            f"NOT SETTLED: |u|max reached {out['reached'] * 1e3:.3f} mm/s after "
            f"{done:,} steps against a bound of {bound * 1e3:.3f} mm/s "
            f"({out['reached'] / bound:.2f}x). No draft is reported."
        )
        progress(f"  {out['verdict']}")
        if keep:
            keep(out)
        return out

    # -- release ---------------------------------------------------------------------
    total = int(release_steps if release_steps else math.ceil(6.0 * period / dt))
    out["config"]["release_steps"] = total
    out["config"]["hold"] = bool(hold)
    progress(f"  settled at {out['reached'] * 1e3:.3f} mm/s after {done:,} steps; "
             f"{'HOLDING' if hold else 'releasing'} for {total:,} steps "
             f"({6.0 * period * 1e3:.1f} ms)")
    began = time.perf_counter()
    # **Every step, not every `every`-th.** The first run of this file sampled the force
    # at 25-step intervals and the late window read a mean of 1.4736 `m g` on a hull
    # whose draft had a 0.0068 mm wobble, which is impossible for a free body and was
    # entirely the sampling: the per-step force is bimodal, near zero on some steps and
    # 1.6 to 4.0 `m g` on others, and 23 samples spaced 25 apart landed 18 times on the
    # high mode. `z` and `fz` are therefore kept for every step -- 1,333 floats, nothing
    # -- and the mean, the displacement and the velocity are read off the full series.
    # The `every`-th rows stay, for the table a human reads.
    trace_z: list[float] = []
    trace_vz: list[float] = []
    trace_fz: list[float] = []
    advance = world.step if hold else link.step
    for index in range(total):
        advance()
        pose_all = link.state.body_q.numpy()[body]
        trace_z.append(float(pose_all[2]))
        trace_vz.append(float(link.state.body_qd.numpy()[body][2]))
        trace_fz.append(float(np.asarray(world.wrench(body)[0],
                                         dtype=float).ravel()[2]))
        if index % int(every) == 0 or index == total - 1:
            pose = link.state.body_q.numpy()[body]
            vel = link.state.body_qd.numpy()[body]
            level = sc._waterline(world, box, plan, spacing)
            draft = level - (float(pose[2]) - height / 2.0)
            heel, off_axis = sc._roll_of(pose[3:7])
            force = float(np.asarray(world.wrench(body)[0], dtype=float).ravel()[2])
            out["release"].append({
                "step": index, "t": (index + 1) * dt, "z": float(pose[2]),
                "vz": float(vel[2]), "level": level, "draft": draft,
                "draft_over_archimedes": draft / box.draft,
                "fz": force, "fz_over_mg": force / weight,
                "off_axis": off_axis, "speed": float(np.nanmax(np.abs(
                    world.velocities()))),
            })
            if keep and len(out["release"]) % 20 == 0:
                keep(out)
            if index % max(1, total // 20) == 0:
                row = out["release"][-1]
                progress(f"    {index:,}/{total:,}  draft {row['draft'] * 1e3:8.4f} mm "
                         f"({row['draft_over_archimedes']:.4f}x)  Fz/mg "
                         f"{row['fz_over_mg']:+.4f}  "
                         f"{(time.perf_counter() - began) / max(index, 1) * (total - index) / 60:.1f} min left")

    late = [row for row in out["release"] if row["t"] >= 0.6 * total * dt]
    drafts = [row["draft"] for row in late]
    ratios = [row["fz_over_mg"] for row in late]
    out["draft"] = float(np.mean(drafts)) if drafts else None
    out["wobble"] = float(np.std(drafts)) if drafts else None
    out["fz_over_mg_late"] = float(np.mean(ratios)) if ratios else None

    # -- the full series, and the one consistency test the coupling has to pass --------
    #
    # A free rigid body obeys `m z'' = Fz - m g` and `z' = vz`, both over the SAME steps.
    # The fluid is exonerated and the Newton half is measured clean in isolation, so this
    # is where a coupling fault has to show: a mean force that does not match the
    # displacement the body actually made, or a `body_qd` that does not match its own
    # `body_q`. Both are computed over the late window, from the per-step trace, so
    # neither can be a sampling artefact.
    cut = int(0.6 * total)
    z_late = np.asarray(trace_z[cut:], dtype=float)
    vz_late = np.asarray(trace_vz[cut:], dtype=float)
    fz_late = np.asarray(trace_fz[cut:], dtype=float)
    span = max(len(z_late) - 1, 1) * dt
    out["series"] = {"z": trace_z, "vz": trace_vz, "fz": trace_fz}
    out["consistency"] = {
        "steps": int(len(z_late)),
        "fz_mean": float(np.mean(fz_late)),
        "fz_mean_over_mg": float(np.mean(fz_late) / weight),
        "fz_rms_fraction": float(np.std(fz_late) / abs(np.mean(fz_late)))
        if np.mean(fz_late) else float("nan"),
        "fz_near_zero_fraction": float(np.mean(np.abs(fz_late) < 0.01 * weight)),
        # what the mean force says the body must have done, and what it did
        "accel_from_force": float((np.mean(fz_late) - weight) / mass),
        "accel_from_z": float(2.0 * (z_late[-1] - z_late[0] - vz_late[0] * span)
                              / span ** 2) if len(z_late) > 1 else float("nan"),
        "dz_observed": float(z_late[-1] - z_late[0]),
        "dz_from_vz": float(np.mean(vz_late) * span),
        "vz_mean": float(np.mean(vz_late)),
        "vz_from_dz": float((z_late[-1] - z_late[0]) / span),
        "z_wobble": float(np.std(z_late)),
        # The held hull's own reference: `rho g V_geom` at the draft it is pinned at,
        # which is what `draft_curve.py` divides by. `fz_over_rho_g_v` and `fz_over_mg`
        # differ only by the 0.18% between FLUIDS' 998.21 and the 1000 the prose used.
        "fz_over_rho_g_v": float(np.mean(fz_late)
                                 / (world.fluid.rho * sc.GRAVITY * beam * beam
                                    * box.draft)),
        # If the fluid delivers `k` times the weight at the Archimedes draft, a free
        # hull settles at about `1/k` of it -- the one line that makes a held number and
        # a free number the same claim.
        "implied_free_draft_over_archimedes": float(
            weight / np.mean(fz_late)) if np.mean(fz_late) else float("nan"),
    }
    # The draft with no probe in it, from the water's own volume. See
    # `mass_conserving_draft` -- this is the number the reported draft is wrong by.
    keel = float(z_late[-1]) - height / 2.0
    level_mass, draft_mass = mass_conserving_draft(
        fluid_particles=len(world.positions()), particle_radius=particle_radius,
        tank_area=(span * beam) * (span * beam), waterplane=beam * beam, keel=keel)
    out["consistency"].update({
        "keel": keel,
        "level_probe": out["release"][-1]["level"],
        "level_mass": level_mass,
        "level_bias": out["release"][-1]["level"] - level_mass,
        "draft_probe": out["release"][-1]["draft"],
        "draft_mass": draft_mass,
        "draft_mass_over_archimedes": draft_mass / box.draft,
    })
    con = out["consistency"]
    progress(
        f"  consistency over the last {con['steps']:,} steps:\n"
        f"    mean Fz {con['fz_mean'] * 1e3:+.4f} mN = {con['fz_mean_over_mg']:.4f} mg, "
        f"RMS {con['fz_rms_fraction']:.1%}, "
        f"{con['fz_near_zero_fraction']:.1%} of steps below 1% of mg\n"
        f"    a_z from the force {con['accel_from_force']:+.4f} m/s2, "
        f"from the z trace {con['accel_from_z']:+.4f}\n"
        f"    dz observed {con['dz_observed'] * 1e6:+.3f} um, "
        f"dz from mean vz {con['dz_from_vz'] * 1e6:+.3f} um\n"
        f"    mean vz {con['vz_mean'] * 1e3:+.4f} mm/s, "
        f"dz/dt {con['vz_from_dz'] * 1e3:+.4f} mm/s, "
        f"z wobble {con['z_wobble'] * 1e6:.3f} um")
    out["complete"] = True
    out["verdict"] = (
        f"SETTLED at {out['reached'] * 1e3:.3f} mm/s after {done:,} steps "
        f"(bound {bound * 1e3:.3f}). Draft {out['draft'] * 1e3:.4f} mm against "
        f"Archimedes {box.draft * 1e3:.4f} mm = {out['draft'] / box.draft:.4f}x, "
        f"wobble {out['wobble'] * 1e3:.4f} mm ({out['wobble'] / box.draft:.4%} of the "
        f"draft). Mean Fz/mg over the same window {out['fz_over_mg_late']:.4f}."
    )
    progress(f"  {out['verdict']}")
    if keep:
        keep(out)
    return out


def mass_conserving_draft(*, fluid_particles: int, particle_radius: float,
                          tank_area: float, waterplane: float, keel: float) -> tuple:
    """The waterline and draft from the water's own volume, which has no probe in it.

    **This is what found the last M6 fault.** `sph_cases._waterline` measures the free
    surface as the median topmost *particle* in each hull-free column, plus one particle
    radius for the half spacing between a particle's centre and the surface it stands for
    (`SphWorld.surface_height`). A settled SPH column's surface layer sits lower than
    that: the kernel is truncated at a free surface, so the outermost particles relax
    inward, and the probe reads **about 0.9 mm low in these tanks** -- 5.7% of a 16 mm
    draft, and the whole of the static float's reported miss.

    Mass conservation has no such convention in it. The fill places one particle per
    `dp^3` cell, and `n * dp^3` is the water's volume to the last digit: 33,536 particles
    at 4 mm is 2,146,304 mm3, which is exactly `36,864 x 60 - 4,096 x 16` for this tank
    and hull. So for a wall-sided prism, with `keel` the hull's bottom,

        V = A_tank * L - A_waterplane * (L - keel)
          => L = (V - A_waterplane * keel) / (A_tank - A_waterplane)

    is one linear equation in the level `L`, and the draft is `L - keel`. Both are
    returned, with the probe's bias, so a case reports the two and the difference rather
    than picking one silently.

    Lengths in metres, volumes implied; returns `(level, draft)`.
    """
    volume = float(fluid_particles) * (2.0 * particle_radius) ** 3
    level = (volume - waterplane * keel) / (tank_area - waterplane)
    return level, level - keel


def table(payload: dict) -> str:
    """The settle trace and the release, as a reader needs them."""
    cfg = payload.get("config", {})
    lines = [
        f"  radius {cfg.get('particle_radius', 0) * 1e3:g} mm, "
        f"{cfg.get('fluid_particles', 0):,} fluid + "
        f"{cfg.get('boundary_particles', 0):,} boundary, "
        f"{cfg.get('spacings_through_draft', 0):.1f} spacings through the draft, "
        f"{cfg.get('device', '?')}",
        f"  dt {cfg.get('dt', 0) * 1e6:.3f} us, weight "
        f"{cfg.get('weight', 0) * 1e3:.4f} mN, Archimedes draft "
        f"{cfg.get('archimedes_draft', 0) * 1e3:.4f} mm, bound "
        f"{cfg.get('settle_bound', 0) * 1e3:.3f} mm/s",
        "",
        f"  {'settle steps':>14} {'|u|max (mm/s)':>15} {'x bound':>9}",
    ]
    bound = cfg.get("settle_bound") or 1.0
    for row in payload.get("settle_trace", []):
        lines.append(f"  {row['steps']:>14,} {row['reached'] * 1e3:>15.3f} "
                     f"{row['reached'] / bound:>9.2f}")
    release = payload.get("release", [])
    if release:
        weight = cfg.get("weight") or 1.0
        arch = cfg.get("archimedes_draft") or 1.0
        lines += ["",
                  f"  {'step':>8} {'t (ms)':>9} {'draft (mm)':>11} {'x/Arch':>8} "
                  f"{'Fz (mN)':>10} {'Fz/mg':>8} {'vz (mm/s)':>10} {'|u|max':>9}"]
        for row in release[::max(1, len(release) // 40)]:
            lines.append(
                f"  {row['step']:>8} {row['t'] * 1e3:>9.3f} "
                f"{row['draft'] * 1e3:>11.4f} {row['draft'] / arch:>8.4f} "
                f"{row['fz'] * 1e3:>10.4f} {row['fz'] / weight:>8.4f} "
                f"{row['vz'] * 1e3:>10.4f} {row['speed'] * 1e3:>9.3f}")
    con = payload.get("consistency")
    if con:
        # `table` runs on the FETCHING client, not in the container, so these lines
        # render for a run that finished before they were written.
        lines += ["",
                  f"  consistency over the last {con['steps']:,} steps "
                  f"({'held' if cfg.get('hold') else 'free'}):",
                  f"    mean Fz {con['fz_mean'] * 1e3:+.4f} mN = "
                  f"{con['fz_mean_over_mg']:.4f} mg = "
                  f"{con.get('fz_over_rho_g_v', float('nan')):.4f} rho g V_geom",
                  f"    per-step RMS {con['fz_rms_fraction']:.1%}, "
                  f"{con['fz_near_zero_fraction']:.1%} of steps below 1% of mg",
                  f"    implied free draft "
                  f"{con.get('implied_free_draft_over_archimedes', float('nan')):.4f}"
                  f"x Archimedes",
                  f"    a_z from the force {con['accel_from_force']:+.4f} m/s2, "
                  f"from the z trace {con['accel_from_z']:+.4f}",
                  f"    dz {con['dz_observed'] * 1e6:+.3f} um observed, "
                  f"{con['dz_from_vz'] * 1e6:+.3f} um from mean vz; "
                  f"vz {con['vz_mean'] * 1e3:+.4f} mm/s, "
                  f"dz/dt {con['vz_from_dz'] * 1e3:+.4f} mm/s"]
        if "draft_mass" in con:
            lines += [
                f"    waterline: probe {con['level_probe'] * 1e3:.4f} mm, "
                f"mass-conserving {con['level_mass'] * 1e3:.4f} mm, "
                f"bias {con['level_bias'] * 1e3:+.4f} mm",
                f"    draft:     probe {con['draft_probe'] * 1e3:.4f} mm "
                f"({con['draft_probe'] / (cfg.get('archimedes_draft') or 1):.4f}x), "
                f"mass-conserving {con['draft_mass'] * 1e3:.4f} mm "
                f"({con['draft_mass_over_archimedes']:.4f}x)"]
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
    app = modal.App("openreynolds-free-settle", image=image)

    def _remote(settings: dict) -> dict:
        sys.path.insert(0, "/repo")
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
            payload = {"error": traceback.format_exc(), "device": device,
                       "config": {"device": device}}
            sink[f"{label}-error"] = payload
            return payload

    @app.function(gpu="H100", timeout=72000)
    def on_h100(settings: dict) -> dict:
        return _remote(settings)

    @app.function(gpu="A100-80GB", timeout=72000)
    def on_a100(settings: dict) -> dict:
        return _remote(settings)

    RUNNERS = {"h100": on_h100, "a100": on_a100}

    @app.local_entrypoint()
    def main(gpu: str = "h100", radius: float = 0.002, settle: int = 200_000,
             chunk: int = 2_000, release: int = 0, every: int = 50,
             hold: bool = False, label: str = "", out: str = "",
             fetch: bool = False):
        if fetch:
            sink = modal.Dict.from_name(SINK, create_if_missing=True)
            done = {}
            for key in sink.keys():
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
        name = label or f"r{radius * 1e3:g}mm-{'held' if hold else 'free'}-{gpu}"
        payload = RUNNERS[gpu].remote({
            "particle_radius": radius, "settle_steps": settle,
            "settle_slice": chunk, "release_steps": release, "every": every,
            "hold": hold, "label": name})
        if payload.get("error"):
            print(f"\n{name} raised:\n{payload['error']}")
            return
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(table(payload))


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radius", type=float, default=0.008)
    parser.add_argument("--settle", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--release", type=int, default=400)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = one_run(particle_radius=args.radius, settle_steps=args.settle,
                      settle_slice=args.chunk, release_steps=args.release,
                      every=args.every, hold=args.hold, device=args.device)
    print(table(payload))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                  encoding="utf-8")
        print(f"\n  saved: {args.out}")


def demo() -> None:
    """The two checks this file leaves behind: the gate, and the draft arithmetic.

    The solver is measured by the run. What can break silently here is the hard gate --
    an unsettled run that reports a draft anyway is the exact failure this file exists to
    prevent, and it is one `if` -- and `mass_conserving_draft`, which is the three lines
    that said the reported draft was wrong and the hull was not.

    The four H100 runs it is checked against, all settled, all hull-first, 64 x 64 x 32 mm
    hull at 0.5 in 60 mm of water in a three-beam tank. `z` is the hull's centre at the
    end of the release window; the held run was never stepped through Newton, so its `z`
    is the pose it was built at and its true draft is 16.0000 mm by construction.
    """
    fake = {"settled": False, "reached": 0.093, "draft": None,
            "verdict": "NOT SETTLED: ..."}
    assert fake["draft"] is None, "an unsettled run must not carry a draft"
    settled = {"settled": True, "reached": 0.010, "draft": 0.0137}
    assert settled["draft"] is not None
    # the bound itself
    assert abs(0.02 * math.sqrt(9.81 * 0.06) - 0.015344) < 1e-6, "the settle bound moved"

    # (radius, fluid particles, final z, the draft `_waterline` reported)
    runs = [("2 mm held", 0.002, 33_536, 0.0600000, 0.0150953),
            ("2 mm free", 0.002, 33_536, 0.0600938, 0.0149962),
            ("2 mm free, re-run", 0.002, 33_536, 0.0600855, 0.0149954),
            ("1.5 mm free", 0.0015, 79_500, 0.0600998, 0.0150632)]
    for label, radius, count, z, probe in runs:
        level, draft = mass_conserving_draft(
            fluid_particles=count, particle_radius=radius,
            tank_area=0.192 * 0.192, waterplane=0.064 * 0.064, keel=z - 0.016)
        bias = probe - draft
        print(f"  {label:18s} draft: probe {probe * 1e3:7.4f} mm, "
              f"mass-conserving {draft * 1e3:7.4f} mm ({draft / 0.016:.4f}x), "
              f"probe low by {-bias * 1e3:.4f} mm")
        assert 0.98 <= draft / 0.016 <= 1.01, (
            f"{label}: the mass-conserving draft came out {draft / 0.016:.4f}x "
            f"Archimedes. All four of these runs measured 0.993 to 1.000, which is the "
            f"finding -- the hull floats where Archimedes says and the probe does not"
        )
        assert -0.0012 < bias < -0.0007, (
            f"{label}: the probe's bias came out {bias * 1e3:+.4f} mm. All four runs "
            f"measured -0.83 to -0.91 mm, and that bias is the whole of the static "
            f"float's reported miss"
        )
    # the held run is the clean one: pinned at the pose it was built at, so mass
    # conservation must give exactly the Archimedes draft and nothing else
    _, held = mass_conserving_draft(fluid_particles=33_536, particle_radius=0.002,
                                    tank_area=0.192 * 0.192, waterplane=0.064 * 0.064,
                                    keel=0.060 - 0.016)
    assert abs(held - 0.016) < 1e-9, (
        f"the held hull's mass-conserving draft is {held * 1e3:.6f} mm and must be "
        f"exactly 16.000000: the fill placed 33,536 particles, which is exactly "
        f"(36,864 x 60 - 4,096 x 16) / 4^3 cells, so the level is exactly 60 mm"
    )
    print("demo: the gate holds, the bound is 15.344 mm/s, and the four runs reproduce")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        _cli()
