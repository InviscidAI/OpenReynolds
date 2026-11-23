#!/usr/bin/env python3
"""Measure the 2D cylinder path: the empty-channel inlet check, its step rate, and
what one long run will actually cost.

Three measurements, all on the real GPU, because the point of the file is to replace a
blanket timeout with a computed one.

    MODAL_PROFILE=inviscid python -X utf8 -u -m modal run         benchmarks/live/cylinder_profile.py::main --jobs channel,rate --gpu h100

    python -X utf8 benchmarks/live/cylinder_profile.py        # the arithmetic, locally

**1. `channel`.** The 2D-2 channel with no cylinder in it, a parabolic velocity inlet,
and the profile at the outlet against the analytic parabola. The clause that decides
whether this measures the flow or the measurer is *the effective wall position*: the
lattice nodes sit half a cell inside the domain faces (`Domain.resolution` says so), and
the `wall` face rule is halfway bounce-back, which with `units.LAMBDA = 3/16` puts the
no-slip plane exactly on the link midpoint at any viscosity -- that is, on the domain
face itself. So the hydrodynamic channel is `ny * dx` tall and the first fluid node is
at `dx/2`, not at the wall.

`ny * dx` is NOT 0.41: `ny = round(0.41/dx)` and 0.41 is not a whole number of cells at
any resolution this case uses. `cases._cylinder_once` builds its inlet parabola with
`height = 0.41`, so the two parabolas differ, and this job reports both against the
effective-wall reference so the size of that difference is a number rather than an
argument.

**2. `rate`.** Lattice updates per second on this path, and where they go. The cylinder
samples forces every step (`sample_every=1`) and `world.wrench()` is `self._wrench.numpy()`
-- a device-to-host copy -- so each sample is a hard serialisation point. Measured here
three ways at each grid: no sampling, every step, every fiftieth. `substeps` is swept
beside it because the body is static, so `step(substeps=k)` is physically the same run
with one synchronise and one voxelise per `k` lattice steps instead of per step; the
difference between `k = 1` and `k = 50` is everything the coupling step costs that is
not collide-and-stream.

**3. the projection.** Printed by both paths from the step counts
`cases._cylinder_once` computes and whatever MLUPS job 2 measured.

**The cost scales as N^4 on this sweep, not N^3.** `total = 8 * flow_through + 25 *
period` and both terms are `cells / u_lattice`, so steps scale as `N / u_lattice`. At
fixed `u_lattice` that is `N`, and with `nodes ~ N^2` the cost is `N^3` -- which is what
the measured 24/32/44 runs did. But holding `tau` while halving the Mach number requires
`u_lattice ~ 1/N` (`nu_lattice = u_lattice * N / Re`), so steps scale as `N^2` and the
cost as **`N^4`**: 16x per doubling, 256x from 32 cells to 128. That is the arithmetic
the 7200 s ceiling ran into.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# The image, the GPU table and the result writer are `modal_run.py`'s and are not worth
# a second copy -- the dependency pins in particular. Importable from two places because
# this module is read twice: locally, where it sits beside `modal_run.py`, and inside the
# container, where Modal has put the entrypoint at `/root/` and the checkout at `/repo`.
for _candidate in (Path(__file__).resolve().parent, Path("/repo/benchmarks/live")):
    if (_candidate / "modal_run.py").exists():
        sys.path.insert(0, str(_candidate))
        break

import modal  # noqa: E402

from modal_run import GPUS, RESULTS, _sanitise, image  # noqa: E402

app = modal.App("openreynolds-cylinder-profile")

# -- the case's own arithmetic, which both the local and the remote side need -----------
#
# Duplicated from `_cylinder_once` deliberately and in one function, so that a projection
# can be made without importing Warp and without running anything. The numbers are
# asserted against the recorded 24/32/44 runs in `_selfcheck`.

GEOMETRY = {"length": 2.2, "height": 0.41, "diameter": 0.1, "centre": (0.2, 0.2),
            "reynolds": 100.0}
EXPECTED_ST = 0.300
"""`mean(REFERENCES["schafer_turek_2d2"]["strouhal"])`, which is what the case uses to
size its sampling window."""

ESTABLISH_FLOWS = 8.0
SAMPLE_PERIODS = 25.0


def plan(cells_per_d: int, u_lattice: float) -> dict:
    """Nodes and steps for one point of the sweep, by the case's own formulas."""
    diameter, length, height = (GEOMETRY["diameter"], GEOMETRY["length"],
                               GEOMETRY["height"])
    dx = diameter / cells_per_d
    nx = max(3, round(length / dx))
    ny = max(3, round(height / dx))
    nz = 3
    period = cells_per_d / (EXPECTED_ST * u_lattice)
    flow_through = (length / dx) / u_lattice
    steps = int(ESTABLISH_FLOWS * flow_through + SAMPLE_PERIODS * period)
    return {
        "cells_per_d": cells_per_d,
        "u_lattice": u_lattice,
        "mach": u_lattice * 3.0 ** 0.5,
        "peak_mach": 1.5 * u_lattice * 3.0 ** 0.5,
        "nu_lattice": u_lattice * cells_per_d / GEOMETRY["reynolds"],
        "tau_plus": 0.5 + 3.0 * u_lattice * cells_per_d / GEOMETRY["reynolds"],
        "shape": (nx, ny, nz),
        "nodes": nx * ny * nz,
        "effective_height": ny * dx,
        "steps": steps,
        "updates": nx * ny * nz * steps,
    }


SWEEP = [(32, 0.0370), (64, 0.0185), (128, 0.00925)]
"""The three points, paired so that `tau_plus` is 0.5355 at all of them.

`u_lattice ~ 1/N` is what holds `nu_lattice = u_lattice * N / Re` fixed, and holding it
fixed is what makes the sweep a Mach sweep rather than a tau sweep. Ma = u_lattice / c_s
with c_s = 1/sqrt(3), so these are Ma 0.064, 0.032 and 0.016 -- the three points
`PROGRESS_LIVE.md` names.
"""


def _selfcheck() -> None:
    """The one runnable check: the duplicated arithmetic against the recorded runs.

    `benchmarks/live/results/cylinder/result.json` holds nodes and steps for 24, 32 and
    44 cells per diameter at `u_lattice` 0.037. If this function's formulas have drifted
    from `cases._cylinder_once`, these fail.
    """
    recorded = {24: (155_232, 168_216), 32: (276_672, 224_288), 44: (522_720, 308_396)}
    for cells, (nodes, steps) in recorded.items():
        got = plan(cells, 0.0370)
        assert got["nodes"] == nodes, (cells, got["nodes"], nodes)
        assert got["steps"] == steps, (cells, got["steps"], steps)
    # And the scaling claim this file rests on: N^4 across the Mach sweep, N^3 at fixed
    # lattice velocity.
    a, b = plan(32, 0.0370)["updates"], plan(64, 0.0185)["updates"]
    assert 15.5 < b / a < 16.5, b / a
    c, d = plan(32, 0.0370)["updates"], plan(64, 0.0370)["updates"]
    assert 7.5 < d / c < 8.5, d / c


# -- the remote measurements ------------------------------------------------------------


def _world(cells_per_d: int, u_lattice: float, *, cylinder: bool,
           inlet_height: str = "effective", device: str | None = None,
           outlet_rho: float = 0.0):
    """The 2D-2 world `_cylinder_once` builds, with the cylinder optional.

    `inlet_height` picks which channel height the inlet parabola is computed for:
    `geometric` is 0.41, which is what `cases.py` does, and `effective` is `ny * dx`,
    which is where the bounce-back walls actually are.

    `outlet_rho` is passed straight to `FluidWorld`: 0 is the first-order extrapolating
    outlet that anchors nothing, and 1.0 rescales the outlet node's populations to the
    lattice rest density.
    """
    import numpy as np

    from openreynolds.live.units import Scaling
    from openreynolds.live.world import Domain, FluidWorld

    diameter = GEOMETRY["diameter"]
    length, height = GEOMETRY["length"], GEOMETRY["height"]
    centre = GEOMETRY["centre"]
    dx = diameter / cells_per_d
    nz = 3
    scaling = Scaling.for_reynolds(dx, diameter, GEOMETRY["reynolds"], "water",
                                   u_lattice=u_lattice, peak_factor=1.5, scheme="trt")
    u_mean = scaling.reference_velocity
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(length, height, nz * dx),
                    faces={"-x": "inlet", "+x": "outlet", "-y": "wall", "+y": "wall",
                           "-z": "periodic", "+z": "periodic"})
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1, collision="trt", smagorinsky=0.0,
                       outlet_rho=outlet_rho, device=device)

    y = world.node_coordinates(1)
    peak = 1.5 * u_mean
    h = world.ny * dx if inlet_height == "effective" else height
    profile = np.zeros((world.ny, world.nz, 3))
    profile[:, :, 0] = (4.0 * peak * y * (h - y) / h ** 2)[:, None]
    world.set_inlet_profile(profile)

    body = -1
    if cylinder:
        body = world.add_body(position=(centre[0], centre[1], nz * dx / 2),
                              label="cylinder")
        world.add_shape("cylinder", body=body, scale=(diameter / 2.0, nz * dx, 0.0))
    world.initialise(velocity=(u_mean, 0.0, 0.0))
    return world, scaling, body, peak, h


def _probe(world, peak: float, h_eff: float, mu: float, step: int) -> dict:
    """The five quantities, read off one host copy of the macroscopic fields.

    **Every reduction is a `nan`-aware one.** `fields()["rho"]` is `nan` at solid nodes,
    and a plain `mean` or `sum` over it returns `nan` -- which is what made one earlier
    mass probe report nothing at all. The empty channel has no solid nodes, so the two
    agree here, but the same function has to survive a body being put back in.

    Pressure comes from the density the solver actually holds: an isothermal lattice has
    `p = c_s^2 rho` exactly, and in SI `c_s = (dx/dt)/sqrt(3)`, so
    `p = rho_SI (dx/dt)^2 / 3`. The constant drops out of the difference, which is all
    that is compared here.
    """
    import numpy as np

    got = world.fields()
    rho, u = got["rho"], got["u"]
    dx = world.dx
    cell = dx ** 3
    area = dx * dx                      # one node's share of an x-plane
    width = world.nz * dx               # the periodic span, the channel's "depth"
    c2 = (dx / world.scaling.dt) ** 2 / 3.0

    ux = u[..., 0]
    inlet_plane, outlet_plane = 0, world.nx - 1
    first, last = 1, world.nx - 2       # the first and last node no face rule sets

    def flux(i: int) -> float:
        return float(np.nansum(ux[i]) * area)

    def mass_flux(i: int) -> float:
        """**The conserved one.** Volume flux is not conserved down a weakly
        compressible channel and should not be: the density falls along the pressure
        gradient, so `u` rises to carry the same `rho u`. Reporting only the volume flux
        would therefore book a correct solver as leaking, and a mass-conserving outlet
        as leaking by exactly the compressibility. `rho u` summed over the plane is what
        has to match in and out, and what the drift in total mass has to integrate to.
        """
        return float(np.nansum(rho[i] * ux[i]) * area)

    def rho_mean(i: int) -> float:
        return float(np.nanmean(rho[i]))

    # p(x), averaged over each x-plane, then a least-squares slope over the middle half
    # of the channel -- away from the inlet node the Zou-He condition sets and the
    # outlet node the outflow rule sets, neither of which is a Poiseuille node.
    p_plane = np.nanmean(rho, axis=(1, 2)) * c2
    x = (np.arange(world.nx) + 0.5) * dx
    lo, hi = world.nx // 4, 3 * world.nx // 4
    slope = float(np.polyfit(x[lo:hi], p_plane[lo:hi], 1)[0])

    # The analytic plane-Poiseuille gradient, taken on the flux actually entering:
    # dp/dx = -12 mu U / h^2 with U the bulk mean speed. Written on the measurement
    # rather than on the imposed peak so that an inlet not delivering what it was asked
    # for shows up as a disagreement here instead of as a hidden premise.
    flux_in = flux(first)
    u_bulk = flux_in / (h_eff * width)
    dpdx_exact = -12.0 * mu * u_bulk / h_eff ** 2

    # The outlet profile against the effective-wall parabola.
    y = world.node_coordinates(1)
    got_profile = np.asarray(ux[last, :, world.nz // 2], dtype=np.float64)
    exact = 4.0 * peak * y * (h_eff - y) / h_eff ** 2
    error = got_profile - exact

    return {
        "step": int(step),
        "mass_kg": float(np.nansum(rho) * cell),
        "rho_mean_kg_m3": float(np.nanmean(rho)),
        "rho_inlet_plane": rho_mean(inlet_plane),
        "rho_outlet_plane": rho_mean(outlet_plane),
        "rho_first_interior": rho_mean(first),
        "rho_last_interior": rho_mean(last),
        "flux_inlet_plane_m3_s": flux(inlet_plane),
        "flux_outlet_plane_m3_s": flux(outlet_plane),
        "flux_first_interior_m3_s": flux_in,
        "flux_last_interior_m3_s": flux(last),
        "mass_flux_inlet_plane_kg_s": mass_flux(inlet_plane),
        "mass_flux_outlet_plane_kg_s": mass_flux(outlet_plane),
        "mass_flux_first_interior_kg_s": mass_flux(first),
        "mass_flux_last_interior_kg_s": mass_flux(last),
        "seconds_per_step": float(world.scaling.dt),
        "dpdx_measured_pa_m": slope,
        "dpdx_poiseuille_pa_m": dpdx_exact,
        "dp_measured_pa": slope * float(x[hi - 1] - x[lo]),
        "dp_poiseuille_pa": dpdx_exact * float(x[hi - 1] - x[lo]),
        "dp_span_m": float(x[hi - 1] - x[lo]),
        "profile_rms_pct": float(np.sqrt(np.mean(error ** 2)) / peak * 100.0),
        "profile_max_pct": float(np.max(np.abs(error)) / peak * 100.0),
        "u_max_m_s": float(np.nanmax(ux)),
        "finite": bool(np.isfinite(np.nanmax(rho)) and np.nanmin(rho) > 0.0),
    }


def _channel(cells_per_d: int, u_lattice: float, device: str | None,
             progress, inlets: tuple[str, ...] = ("effective",),
             steps: int = 0, outlet_rhos: tuple[float, ...] = (0.0, 1.0),
             samples: int = 140) -> dict:
    """The empty 2D-2 channel, run long, with and without the anchored outlet.

    Every combination of `outlet_rhos` and `inlets` is a separate run of `steps` lattice
    steps, probed `samples` times along the way. The probe is the whole point: an outlet
    profile that looks right says nothing about whether the box is conservative, and the
    2D-1 cylinder's drag drifted for 704,000 steps while its lift sat still.

    **Lattice steps are batched into `world.step(substeps=k)` between probes.** With no
    body in the domain nothing moves, so `substeps` is physically the same run -- the
    collide/stream pair is launched `k` times either way -- with one device
    synchronise, one voxelise and one refill per probe instead of per step. At this step
    count that is the difference between a measurement and a host-latency benchmark.
    """
    import numpy as np

    rows = []
    for outlet_rho in outlet_rhos:
        for inlet_height in inlets:
            world, scaling, _, peak, h_inlet = _world(
                cells_per_d, u_lattice, cylinder=False, inlet_height=inlet_height,
                device=device, outlet_rho=outlet_rho)
            dx = world.dx
            h_eff = world.ny * dx
            mu = world.fluid.rho * world.fluid.nu
            flow_through = int((GEOMETRY["length"] / dx) / u_lattice)
            budget = int(steps) if steps > 0 else 8 * flow_through
            chunk = max(1, budget // max(1, samples))
            tag = f"outlet_rho={outlet_rho:g} inlet={inlet_height}"

            series = [_probe(world, peak, h_eff, mu, 0)]
            began = time.time()
            taken = 0
            while taken < budget:
                take = min(chunk, budget - taken)
                world.step(substeps=take)
                taken += take
                row = _probe(world, peak, h_eff, mu, taken)
                row["seconds"] = time.time() - began
                row["mlups"] = world.nodes * taken / row["seconds"] / 1e6
                series.append(row)
                if progress:
                    # One JSON line per probe, printed as it happens, so that a run cut
                    # short by a timeout still leaves its numbers in the log rather than
                    # only in a return value that never arrives.
                    progress("PROBE " + json.dumps({"tag": tag,
                                                    "cells": cells_per_d, **row}))
                if not row["finite"]:
                    if progress:
                        progress(f"  {tag}: non-finite at {taken:,} steps")
                    break

            first_row, last_row = series[0], series[-1]
            y = world.node_coordinates(1)
            field = world.fields()["u"][:, :, world.nz // 2, 0]
            station = world.nx - 2
            outlet = np.asarray(field[station], dtype=np.float64)

            def against(h: float, got=outlet) -> dict:
                exact = 4.0 * peak * y * (h - y) / h ** 2
                error = got - exact
                return {"height": float(h),
                        "rms_pct": float(np.sqrt(np.mean(error ** 2)) / peak * 100.0),
                        "max_pct": float(np.max(np.abs(error)) / peak * 100.0)}

            stations = {}
            for name, i in (("inlet+1", 1), ("quarter", world.nx // 4),
                            ("half", world.nx // 2),
                            ("three-quarter", 3 * world.nx // 4),
                            ("outlet-1", station)):
                here = np.asarray(field[i], dtype=np.float64)
                stations[name] = {"i": int(i), "x": float((i + 0.5) * dx),
                                  "vs_effective": against(h_eff, here),
                                  "flux": float(np.sum(here) * dx)}

            rows.append({
                "outlet_rho": float(outlet_rho),
                "inlet_parabola_for": inlet_height,
                "inlet_height_used": float(h_inlet),
                "effective_height": float(h_eff),
                "geometric_height": float(GEOMETRY["height"]),
                "wall_offset_cells": 0.5,
                "ny": int(world.ny), "nx": int(world.nx), "nodes": int(world.nodes),
                "steps": int(last_row["step"]),
                "flow_throughs": last_row["step"] / flow_through,
                "seconds": last_row.get("seconds", float("nan")),
                "u_peak": float(peak),
                "mu_pa_s": float(mu),
                "mass_start_kg": first_row["mass_kg"],
                "mass_end_kg": last_row["mass_kg"],
                "mass_drift_pct": (last_row["mass_kg"] / first_row["mass_kg"] - 1.0)
                                  * 100.0,
                "mass_max_drift_pct": max(abs(r["mass_kg"] / first_row["mass_kg"] - 1.0)
                                          for r in series) * 100.0,
                "flux_imbalance_pct": (last_row["flux_last_interior_m3_s"]
                                       / last_row["flux_first_interior_m3_s"] - 1.0)
                                      * 100.0,
                "mass_flux_imbalance_pct":
                    (last_row["mass_flux_last_interior_kg_s"]
                     / last_row["mass_flux_first_interior_kg_s"] - 1.0) * 100.0,
                "dpdx_ratio": (last_row["dpdx_measured_pa_m"]
                               / last_row["dpdx_poiseuille_pa_m"]),
                "vs_effective": against(h_eff),
                "vs_geometric": against(GEOMETRY["height"]),
                "flux_in": float(np.sum(np.asarray(field[1], dtype=np.float64)) * dx),
                "flux_out": float(np.sum(outlet) * dx),
                "outlet_profile": outlet.tolist(),
                "y": y.tolist(),
                "stations": stations,
                "final": last_row,
                "series": series,
            })
            if progress:
                row = rows[-1]
                progress(f"  {tag}: {row['steps']:,} steps "
                         f"({row['flow_throughs']:.1f} flow-throughs) -- mass "
                         f"{row['mass_drift_pct']:+.3f}%, flux in/out "
                         f"{row['flux_imbalance_pct']:+.3f}%, dp/dx vs Poiseuille "
                         f"x{row['dpdx_ratio']:.4f}, outlet profile rms "
                         f"{row['vs_effective']['rms_pct']:.3f}% max "
                         f"{row['vs_effective']['max_pct']:.3f}%")
            del world
    return {"cells_per_d": cells_per_d, "u_lattice": u_lattice, "rows": rows,
            "tolerance_pct": 0.5,
            "tolerance_is": ("the user's 0.5%, taken on the peak inflow speed, which is "
                             "the normalisation REFERENCES['poiseuille'] already uses "
                             "for the same quantity")}


def _rate(cells_per_d: int, u_lattice: float, steps: int, device: str | None,
          progress) -> dict:
    """Deliverable 2: MLUPS on the cylinder path, and what the per-step sample costs.

    Same world as the graded run -- cylinder, parabolic inlet, TRT -- timed over `steps`
    lattice steps in five configurations. The sampling callback is `_cylinder_once`'s,
    including the `world.wrench(body)` host read, because that read is the thing under
    test.
    """
    import numpy as np
    import warp as wp

    world, scaling, body, _, _ = _world(cells_per_d, u_lattice, cylinder=True,
                                        device=device)
    depth = 3 * world.dx
    u_mean = scaling.reference_velocity
    coefficient = 2.0 / (scaling.fluid.rho * u_mean * u_mean
                         * GEOMETRY["diameter"] * depth)
    for _ in range(20):
        world.step()  # compile and warm; the first launches are not the rate

    cd: list[float] = []

    def take() -> None:
        force, _ = world.wrench(body)
        cd.append(float(force[0]) * coefficient)

    def timed(label: str, *, sample_every: int, substeps: int) -> dict:
        cd.clear()
        count = max(1, steps // substeps)
        wp.synchronize_device(world.device)
        began = time.perf_counter()
        for index in range(count):
            world.step(substeps=substeps)
            if sample_every and index % sample_every == 0:
                take()
        wp.synchronize_device(world.device)
        elapsed = time.perf_counter() - began
        done = count * substeps
        row = {"label": label, "sample_every": sample_every, "substeps": substeps,
               "lattice_steps": done, "seconds": elapsed,
               "mlups": world.nodes * done / elapsed / 1e6,
               "us_per_lattice_step": elapsed / done * 1e6,
               "samples": len(cd),
               "cd_mean": float(np.mean(cd)) if cd else float("nan")}
        if progress:
            progress(f"    {cells_per_d:>4} cells  {label:<28} "
                     f"{row['mlups']:8.1f} MLUPS  "
                     f"{row['us_per_lattice_step']:7.1f} us a lattice step")
        return row

    rows = [
        timed("no sampling", sample_every=0, substeps=1),
        timed("sample every step", sample_every=1, substeps=1),
        timed("sample every 50", sample_every=50, substeps=1),
        # The body is static, so these are the same physics with one synchronise, one
        # voxelise and one refill per k lattice steps instead of per step. The gap
        # between them and `no sampling` is the coupling step's own overhead.
        timed("substeps=10, no sampling", sample_every=0, substeps=10),
        timed("substeps=50, no sampling", sample_every=0, substeps=50),
    ]
    # And the host read on its own, away from the loop, so the per-step cost has a
    # direct measurement beside the differenced one.
    wp.synchronize_device(world.device)
    began = time.perf_counter()
    for _ in range(200):
        world.wrench(body)
    wrench_us = (time.perf_counter() - began) / 200 * 1e6
    return {"cells_per_d": cells_per_d, "u_lattice": u_lattice, "nodes": world.nodes,
            "shape": list(world.shape), "wrench_us": wrench_us, "rows": rows,
            **{k: v for k, v in plan(cells_per_d, u_lattice).items()
               if k in ("steps", "updates", "tau_plus", "mach")}}


def _remote(jobs: str, gpu_steps: int, channel_cells: str, inlets: str,
            outlet_rhos: str = "0,1.0", samples: int = 140) -> dict:
    """Everything, in one container, so the Warp compile is paid once."""
    import traceback

    sys.path.insert(0, "/repo")
    import warp as wp

    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    out: dict = {"device": device, "jobs": jobs}
    want = {j.strip() for j in jobs.split(",") if j.strip()}
    started = time.time()
    try:
        if "channel" in want:
            print("# the empty-channel inlet check")
            # The channel is run at the sweep's own pairings, so that a deviation that
            # scales with the Mach number is distinguishable from one that does not.
            paired = dict(SWEEP)
            out["channel"] = [
                _channel(int(c), paired[int(c)], device, print,
                         tuple(i.strip() for i in inlets.split(",") if i.strip()),
                         steps=int(gpu_steps),
                         outlet_rhos=tuple(float(r) for r in outlet_rhos.split(",")
                                           if r.strip()),
                         samples=int(samples))
                for c in channel_cells.split(",") if c.strip()
            ]
        if "rate" in want:
            print("# MLUPS on the cylinder path")
            out["rate"] = [_rate(cells, u, gpu_steps, device, print)
                           for cells, u in SWEEP]
    except Exception:
        out["error"] = traceback.format_exc()
        print(out["error"])
    out["seconds"] = time.time() - started
    return _sanitise(out)


@app.function(image=image, gpu="H100", timeout=21600)
def on_h100(jobs: str, gpu_steps: int, channel_cells: str, inlets: str,
          outlet_rhos: str = "0,1.0", samples: int = 140) -> dict:
    return _remote(jobs, gpu_steps, channel_cells, inlets, outlet_rhos,
                   samples)


@app.function(image=image, gpu="A100-80GB", timeout=21600)
def on_a100(jobs: str, gpu_steps: int, channel_cells: str, inlets: str,
          outlet_rhos: str = "0,1.0", samples: int = 140) -> dict:
    return _remote(jobs, gpu_steps, channel_cells, inlets, outlet_rhos,
                   samples)


@app.function(image=image, gpu="L40S", timeout=21600)
def on_l40s(jobs: str, gpu_steps: int, channel_cells: str, inlets: str,
          outlet_rhos: str = "0,1.0", samples: int = 140) -> dict:
    return _remote(jobs, gpu_steps, channel_cells, inlets, outlet_rhos,
                   samples)


RUNNERS = {"h100": on_h100, "a100": on_a100, "l40s": on_l40s}


# -- the projection, which is arithmetic and runs anywhere -------------------------------


def project(mlups: dict[int, float] | None = None) -> list[dict]:
    """Wall-clock seconds and a timeout for each point of the sweep.

    `mlups` maps cells-per-diameter to a measured rate. A point with no measured rate
    borrows the nearest larger measured one, which is the conservative direction: the
    rate rises with lattice size until the card is full, so a smaller grid's rate
    under-states a larger grid's and the projected time comes out long rather than short.
    """
    mlups = mlups or {}
    table = []
    for cells, u_lattice in SWEEP:
        row = plan(cells, u_lattice)
        rate = mlups.get(cells)
        source = "measured"
        if rate is None and mlups:
            nearest = min(mlups, key=lambda c: (abs(c - cells), -c))
            rate, source = mlups[nearest], f"borrowed from {nearest} cells"
        if rate is None:
            rate, source = 1000.0, "assumed, nothing measured"
        row["mlups"] = rate
        row["mlups_source"] = source
        row["seconds"] = row["updates"] / (rate * 1e6)
        row["timeout_s"] = int(row["seconds"] * 1.5)
        table.append(row)
    return table


def _print_projection(table: list[dict]) -> None:
    print(f"\n{'=' * 96}")
    print("projected runtimes, 2D-2 cylinder, tau_plus held at 0.5355")
    print(f"{'=' * 96}")
    print(f"  {'cells':>5} {'u_lat':>8} {'Ma':>7} {'nodes':>12} {'steps':>12} "
          f"{'updates':>11} {'MLUPS':>8} {'wall s':>10} {'x1.5':>10}")
    for row in table:
        print(f"  {row['cells_per_d']:>5} {row['u_lattice']:>8.5f} {row['mach']:>7.4f} "
              f"{row['nodes']:>12,} {row['steps']:>12,} {row['updates']:>11.4g} "
              f"{row['mlups']:>8.1f} {row['seconds']:>10,.0f} {row['timeout_s']:>10,}")
    for row in table:
        print(f"    {row['cells_per_d']:>4} cells: MLUPS {row['mlups_source']}")
    base = table[0]
    print("\n  scaling, against the first row:")
    for row in table:
        ratio = row["cells_per_d"] / base["cells_per_d"]
        print(f"    {row['cells_per_d']:>4} cells: N x{ratio:>4.0f}  "
              f"nodes x{row['nodes'] / base['nodes']:>6.1f} (N^2 = {ratio ** 2:.0f})  "
              f"steps x{row['steps'] / base['steps']:>6.1f} (N^2 = {ratio ** 2:.0f})  "
              f"updates x{row['updates'] / base['updates']:>7.1f} "
              f"(N^4 = {ratio ** 4:.0f})")


@app.local_entrypoint()
def main(jobs: str = "channel,rate", gpu: str = "h100", steps: int = 400,
         channel_cells: str = "32", inlets: str = "effective,geometric",
         outlet_rhos: str = "0,1.0", samples: int = 140,
         label: str = "cylinder-profile") -> None:
    _selfcheck()
    got = RUNNERS[gpu.lower()].remote(jobs, int(steps), channel_cells, inlets,
                                      outlet_rhos, int(samples))
    out = RESULTS / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")

    measured = {}
    for entry in got.get("rate", []) or []:
        for row in entry["rows"]:
            if row["label"] == "sample every step":
                measured[entry["cells_per_d"]] = row["mlups"]
    _print_projection(project(measured))
    print(f"\n  saved: {out / 'result.json'}")
    if got.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    _selfcheck()
    _print_projection(project())
    print("\n  no GPU here; the MLUPS column is an assumption. Run `modal run "
          "benchmarks/live/cylinder_profile.py::main` for the measured one.")
