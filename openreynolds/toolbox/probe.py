#!/usr/bin/env python3
"""Any solved field, sampled where you ask, as JSON (and a picture for lines and streamlines).

    python3 probe.py CASE --fields
    python3 probe.py CASE --line X0 Y0 X1 Y1 [--z Z] --field F [--n 200] [--png line.png]
    python3 probe.py CASE --points "x,y;x,y" --field F
    python3 probe.py CASE --region XMIN YMIN XMAX YMAX --field F --stat mean|min|max|integral
    python3 probe.py CASE --section-mean --field F --at-x X [--weighted flux]
    python3 probe.py CASE --between X0 X1 --field F [--weighted flux]
    python3 probe.py CASE --flux --at-x X
    python3 probe.py CASE --streamlines --seeds "x,y;x,y" [--direction forward|backward|both] [--png s.png]

    common: [--time latest|T] [--out probe.json]

A score says how good a candidate is and nothing about why. The seed-5 Tesla campaign
(Archimedes `results/rematch/REFLECTION-tesla_v3-s5.md`) ran on that one number and could
not ask where the reverse flow lost its pressure; this is the instrument for that question,
for any physics: which fields exist is read off the case (U, p, T, k, omega, species, ...),
never assumed.

CASE is a solved case, or a candidate directory holding case subdirectories (a lock with
`cases`, e.g. `forward/` and `reverse/`): one such subdirectory is probed through, several
must be named (solved subdirectories win over a mesh at the root). The case is read by
`render.py`'s `open_case` -- pyvista's OpenFOAMReader, falling back to `foamToVTK` -- at
the write nearest `--time` (default the latest). A decomposed case whose processor*/
holds a later time is refused: `reconstructPar` first. On a one-cell-thick 2D case `--z`
defaults to the mid-plane, and integrals and fluxes are per the case's depth, which `notes` states.

What each op measures:

- `--line`/`--points` interpolate the point field (VTK's probe); a point outside the mesh
  is `null`, not zero. A vector comes back as `F_x`, `F_y`, `F_z` and `F_mag`, signed.
- `--region` reduces the cells whose centres fall in the x-y box: `mean` is volume-weighted,
  `integral` is the sum of value x cell volume.
- `--section-mean` cuts the plane x = X and means the cut cells' values weighted by area, or
  by the volumetric flux through each (`--weighted flux`), which is the mixed-mean a loss
  between two sections is honestly measured with. `--between X0 X1` is mean(X1) - mean(X0).
- `--flux` is the integral of U_x over the section x = X.
- `--streamlines` integrates U from each seed; each polyline comes back with its length.

`p_total` is derived when the case did not write it: p + |U|^2/2 for kinematic pressure
(dimensions m^2 s^-2), p + rho |U|^2/2 when p is in Pa and the case wrote `rho`.

Output: `{"op", "field", "time", "units", "case", "data", "notes"}` on stdout (and to `--out`);
exit 0, or non-zero with a message saying what was missing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402

KINDS = {1: "scalar", 3: "vector", 6: "symmTensor", 9: "tensor"}
COMPONENTS = {3: ("x", "y", "z"), 6: ("xx", "xy", "xz", "yy", "yz", "zz"),
              9: ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")}
SI = ("kg", "m", "s", "K", "mol", "A", "cd")
READER_ARRAYS = {"vtkOriginalCellIds", "vtkValidPointMask", "vtkGhostType"}


# -- the case ----------------------------------------------------------------------------

def is_case(path: Path) -> bool:
    return (path / "constant" / "polyMesh").is_dir()


def times_in(path: Path) -> list[float]:
    """The numeric (time) directory names directly under PATH."""
    out = []
    for child in path.iterdir():
        try:
            if child.is_dir():
                out.append(float(child.name))
        except ValueError:
            continue
    return out


def resolve_case(path: Path) -> Path:
    """The case to read: its one solved case subdirectory, else PATH itself.

    Subdirectories come first: parametric.py leaves build.py's mesh at the candidate root,
    so the root passes `is_case` while the solves live in `forward/`, `reverse/`, ...
    """
    subs = sorted(p for p in path.iterdir()
                  if p.is_dir() and not p.name.startswith("processor")
                  and is_case(p) and times_in(p)) if path.is_dir() else []
    if len(subs) == 1:
        return subs[0]
    if subs:
        raise SystemExit(f"{path} holds several cases; name one: "
                         + ", ".join(str(p) for p in subs))
    if is_case(path):
        return path
    raise SystemExit(f"{path} is not a case (no constant/polyMesh) and holds none")


def refuse_decomposed(case: Path, chosen: float) -> None:
    """A parallel solve leaves its times in processor*/ until reconstructPar runs."""
    later = [t for d in case.glob("processor*") if d.is_dir() for t in times_in(d) if t > chosen]
    if later:
        raise SystemExit(f"{case} is decomposed: processor directories hold time {max(later):g}, "
                         f"later than the reconstructed {chosen:g}; run "
                         "`reconstructPar -latestTime` first")


def open_case(case: Path, time: float | None):
    """The internal mesh at the write nearest `time`, and that time.

    stdout is the JSON channel, and `render.open_case` says on stdout when it falls back
    to foamToVTK, so what it prints goes to stderr instead.
    """
    with contextlib.redirect_stdout(sys.stderr):
        block, chosen, _ = render.open_case(case, time)
        mesh = render.internal_mesh(block)
    refuse_decomposed(case, float(chosen))
    if mesh is None or mesh.n_cells == 0:
        raise SystemExit(f"no internal mesh in {case}")
    if not mesh.point_data.keys() and mesh.cell_data.keys():
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    return mesh, float(chosen)


def time_dir(case: Path, time: float) -> Path | None:
    for child in case.iterdir():
        try:
            if child.is_dir() and float(child.name) == time:
                return child
        except ValueError:
            continue
    return None


def units_of(case: Path, time: float, field: str) -> str | None:
    """The field's `dimensions [...]` as SI text, read off its file header."""
    directory = time_dir(case, time)
    path = directory / field if directory else None
    if path is None or not path.is_file():
        return None
    with path.open(errors="replace") as handle:
        head = handle.read(4096)
    match = re.search(r"dimensions\s*\[([^\]]*)\]", head)
    if not match:
        return None
    try:
        powers = [float(v) for v in match.group(1).split()][:7]
    except ValueError:
        return None
    parts = [unit if p == 1 else f"{unit}^{p:g}" for unit, p in zip(SI, powers) if p]
    return " ".join(parts) or "1"


def fields_of(mesh) -> list[str]:
    return sorted((set(mesh.cell_data.keys()) | set(mesh.point_data.keys())) - READER_ARRAYS)


def need(mesh, field: str) -> None:
    if field not in fields_of(mesh):
        raise SystemExit(f"the case has no field '{field}' at this time; it has: "
                         + (", ".join(fields_of(mesh)) or "nothing"))


def derive_total_pressure(mesh, case: Path, time: float, notes: list[str]) -> str | None:
    """Add `p_total` to both cell and point data; returns its units."""
    need(mesh, "p")
    need(mesh, "U")
    units = units_of(case, time, "p")
    kinematic = units == "m^2 s^-2"
    if not kinematic and "rho" not in fields_of(mesh):
        raise SystemExit("p_total: p is not kinematic (m^2 s^-2) and the case wrote no rho")
    for data in (mesh.cell_data, mesh.point_data):
        if "p" in data and "U" in data:
            half_u2 = 0.5 * np.sum(np.asarray(data["U"], float) ** 2, axis=1)
            rho = 1.0 if kinematic else np.asarray(data["rho"], float)
            data["p_total"] = np.asarray(data["p"], float) + rho * half_u2
    notes.append("p_total derived here: p + |U|^2/2" if kinematic
                 else "p_total derived here: p + rho |U|^2/2")
    return units


# -- values ------------------------------------------------------------------------------

def columns(field: str, array) -> dict[str, np.ndarray]:
    """A scalar as `value`, anything else as named components (plus `_mag` for a vector)."""
    array = np.asarray(array, dtype=float)
    if array.ndim == 1 or array.shape[1] == 1:
        return {"value": array.reshape(-1)}
    names = COMPONENTS.get(array.shape[1], tuple(str(i) for i in range(array.shape[1])))
    out = {f"{field}_{name}": array[:, i] for i, name in enumerate(names)}
    if array.shape[1] == 3:
        out[f"{field}_mag"] = np.linalg.norm(array, axis=1)
    return out


def clean(value):
    """JSON-safe: numpy to Python, NaN and inf to null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    value = float(value)  # 7 digits: the reader hands back float32, the rest is noise
    return float(f"{value:.7g}") if np.isfinite(value) else None


def reduce_columns(cols: dict[str, np.ndarray], fn) -> dict | float:
    out = {name: fn(values) for name, values in cols.items()}
    return out["value"] if list(out) == ["value"] else out


def parse_points(text: str, z: float) -> np.ndarray:
    try:
        pts = [[float(v) for v in chunk.split(",")] for chunk in text.split(";") if chunk.strip()]
    except ValueError:
        raise SystemExit(f"points are 'x,y;x,y' (or x,y,z); got {text!r}") from None
    if not pts or any(len(p) not in (2, 3) for p in pts):
        raise SystemExit(f"points are 'x,y;x,y' (or x,y,z); got {text!r}")
    return np.array([p if len(p) == 3 else p + [z] for p in pts], dtype=float)


def sampled(probed, field: str) -> dict[str, list]:
    """A probe result's field as columns, with points outside the mesh as None."""
    valid = np.asarray(probed.point_data.get("vtkValidPointMask", np.ones(probed.n_points)), bool)
    cols = columns(field, probed.point_data[field])
    return {name: [v if ok else None for v, ok in zip(values.tolist(), valid)]
            for name, values in cols.items()}


def section(mesh, x: float, field: str):
    """The cut x = X as (cell values, areas, U_x per cut cell)."""
    lo, hi = mesh.bounds[0], mesh.bounds[1]
    if not lo <= x <= hi:
        raise SystemExit(f"x = {x:g} is outside the mesh (x from {lo:g} to {hi:g})")
    centre = np.array(mesh.center)
    cut = mesh.slice(normal=(1, 0, 0), origin=(x, centre[1], centre[2]))
    if cut.n_cells == 0:
        raise SystemExit(f"the plane x = {x:g} cuts no cells")
    if field not in cut.cell_data or "U" not in cut.cell_data:
        cut = cut.point_data_to_cell_data(pass_point_data=True)
    area = np.asarray(cut.compute_cell_sizes(length=False, volume=False)["Area"], float)
    ux = np.asarray(cut.cell_data["U"], float)[:, 0] if "U" in cut.cell_data else None
    return np.asarray(cut.cell_data[field], float), area, ux


def section_mean(mesh, x: float, field: str, weighted: str) -> dict | float:
    values, area, ux = section(mesh, x, field)
    if weighted == "flux":
        if ux is None:
            raise SystemExit("--weighted flux needs U, which this case did not write")
        weights = ux * area
    else:
        weights = area
    if abs(weights.sum()) == 0:
        raise SystemExit(f"the {weighted} weights sum to zero at x = {x:g}")
    return reduce_columns(columns(field, values),
                          lambda v: float(np.sum(v * weights) / np.sum(weights)))


# -- ops ---------------------------------------------------------------------------------

def op_fields(mesh, case, time, args, notes):
    if not fields_of(mesh):
        raise SystemExit(f"{case} has no solved fields at time {time:g}")
    data = {}
    for name in fields_of(mesh):
        array = np.asarray(mesh.cell_data[name] if name in mesh.cell_data
                           else mesh.point_data[name], float)
        ncomp = 1 if array.ndim == 1 else array.shape[1]
        magnitude = array if ncomp == 1 else np.linalg.norm(array, axis=1)
        data[name] = {"kind": KINDS.get(ncomp, f"{ncomp} components"),
                      "units": units_of(case, time, name),
                      "min": float(np.nanmin(magnitude)), "max": float(np.nanmax(magnitude))}
    if any(data[n]["kind"] != "scalar" for n in data):
        notes.append("min/max of a vector or tensor are of its magnitude")
    return data


def op_line(mesh, case, time, args, notes):
    x0, y0, x1, y1 = args.line
    a, b = np.array([x0, y0, args.z]), np.array([x1, y1, args.z])
    probed = mesh.sample_over_line(a, b, resolution=max(args.n, 2) - 1)
    pts = np.asarray(probed.points)
    data = {"s": np.linalg.norm(pts - a, axis=1).tolist(),
            "x": pts[:, 0].tolist(), "y": pts[:, 1].tolist(), "z": pts[:, 2].tolist()}
    data.update(sampled(probed, args.field))
    return data


def op_points(mesh, case, time, args, notes):
    pv = render._pyvista()
    pts = parse_points(args.points, args.z)
    probed = pv.PolyData(pts).sample(mesh)
    return {"points": pts.tolist(), **sampled(probed, args.field)}


def op_region(mesh, case, time, args, notes):
    xmin, ymin, xmax, ymax = args.region
    centres = np.asarray(mesh.cell_centers().points)
    inside = ((centres[:, 0] >= xmin) & (centres[:, 0] <= xmax)
              & (centres[:, 1] >= ymin) & (centres[:, 1] <= ymax))
    if not inside.any():
        raise SystemExit("no cell centre falls inside that region")
    if args.field not in mesh.cell_data:
        mesh = mesh.point_data_to_cell_data(pass_point_data=True)
    volume = np.abs(np.asarray(
        mesh.compute_cell_sizes(length=False, area=False)["Volume"], float))[inside]
    cols = columns(args.field, np.asarray(mesh.cell_data[args.field], float)[inside])
    stat = {"mean": lambda v: float(np.sum(v * volume) / np.sum(volume)),
            "min": lambda v: float(np.min(v)), "max": lambda v: float(np.max(v)),
            "integral": lambda v: float(np.sum(v * volume))}[args.stat]
    if args.stat == "integral":
        notes.append("integral is value x cell volume; divide by the depth for per-unit-depth")
    return {"stat": args.stat, "value": reduce_columns(cols, stat),
            "n_cells": int(inside.sum()), "volume": float(volume.sum())}


def op_section_mean(mesh, case, time, args, notes):
    return {"at_x": args.at_x, "weighted": args.weighted,
            "value": section_mean(mesh, args.at_x, args.field, args.weighted)}


def op_between(mesh, case, time, args, notes):
    x0, x1 = args.between
    m0 = section_mean(mesh, x0, args.field, args.weighted)
    m1 = section_mean(mesh, x1, args.field, args.weighted)
    diff = ({k: m1[k] - m0[k] for k in m0} if isinstance(m0, dict) else m1 - m0)
    return {"x0": x0, "x1": x1, "weighted": args.weighted,
            "mean_x0": m0, "mean_x1": m1, "difference": diff}


def op_flux(mesh, case, time, args, notes):
    need(mesh, "U")
    _, area, ux = section(mesh, args.at_x, "U")
    return {"at_x": args.at_x, "value": float(np.sum(ux * area)), "area": float(area.sum())}


def op_streamlines(mesh, case, time, args, notes):
    pv = render._pyvista()
    field = args.field
    need(mesh, field)
    if field not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    seeds = parse_points(args.seeds, args.z)
    b = np.asarray(mesh.bounds, float)
    diagonal = float(np.linalg.norm(b[1::2] - b[0::2]))
    # pyvista renamed max_time to max_length (and made the old name an error); the
    # image's version is not pinned here, so ask the signature which one it takes.
    import inspect
    reach = ("max_length" if "max_length" in
             inspect.signature(mesh.streamlines_from_source).parameters else "max_time")
    lines = mesh.streamlines_from_source(
        pv.PolyData(seeds), vectors=field, integration_direction=args.direction,
        initial_step_length=0.2, max_step_length=0.5, max_steps=50000,  # cell lengths
        **{reach: 3 * diagonal})
    # With `both`, VTK writes a backward and a forward piece per seed, each starting at
    # the seed; ordering every point by its integration time (negative going backward)
    # joins them into one polyline upstream to downstream, whatever order they came in.
    seed_ids = (np.asarray(lines.cell_data["SeedIds"]) if "SeedIds" in lines.cell_data
                else np.arange(lines.n_cells))
    when = np.asarray(lines.point_data["IntegrationTime"], float)
    by_seed: dict[int, dict[float, np.ndarray]] = {}
    for i in range(lines.n_cells):
        ids = lines.get_cell(i).point_ids
        by_seed.setdefault(int(seed_ids[i]), {}).update(
            {float(when[j]): np.asarray(lines.points[j]) for j in ids})
    out = []
    for sid, seed in enumerate(seeds):
        timed = by_seed.get(sid, {})
        pts = np.array([timed[t] for t in sorted(timed)]).reshape(-1, 3)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
        if len(pts) > args.n:
            pts = pts[np.linspace(0, len(pts) - 1, args.n).round().astype(int)]
        out.append({"seed": seed.tolist(), "length": length, "points": pts.tolist()})
    if any(not line["points"] for line in out):
        notes.append("a seed with no points is outside the mesh or in still flow")
    return {"field": field, "direction": args.direction, "lines": out}


OPS = {"fields": op_fields, "line": op_line, "points": op_points, "region": op_region,
       "section_mean": op_section_mean, "between": op_between, "flux": op_flux,
       "streamlines": op_streamlines}


# -- pictures ----------------------------------------------------------------------------

def picture(result: dict, mesh, z: float, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    data = result["data"]
    title = f"{result['op']}  {result.get('field') or ''}  t = {result['time']:g}"
    if result["op"] == "line":
        for name, values in data.items():
            if name in ("s", "x", "y", "z"):
                continue
            ax.plot(data["s"], [np.nan if v is None else v for v in values], label=name)
        ax.set_xlabel("distance along the line")
        ax.set_ylabel(f"{result['field']} [{result.get('units') or '?'}]")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        centres = np.asarray(mesh.cell_centers().points)
        ax.scatter(centres[:, 0], centres[:, 1], s=1, color="0.85", zorder=0)
        for line in data["lines"]:
            pts = np.asarray(line["points"])
            if len(pts):  # a seed off the mesh is left out so it cannot reframe the plot
                ax.plot(pts[:, 0], pts[:, 1], lw=1.2)
                ax.plot(line["seed"][0], line["seed"][1], "k.", ms=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# -- the command -------------------------------------------------------------------------

def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path)
    parser.add_argument("--time", default="latest", help="latest (default) or a time")
    ops = parser.add_mutually_exclusive_group(required=True)
    ops.add_argument("--fields", action="store_true")
    ops.add_argument("--line", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    ops.add_argument("--points")
    ops.add_argument("--region", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ops.add_argument("--section-mean", action="store_true")
    ops.add_argument("--between", nargs=2, type=float, metavar=("X0", "X1"))
    ops.add_argument("--flux", action="store_true")
    ops.add_argument("--streamlines", action="store_true")
    parser.add_argument("--field")
    parser.add_argument("--z", type=float, default=None, help="default: the mesh mid-plane")
    parser.add_argument("--n", type=int, default=200, help="points on a line / per streamline")
    parser.add_argument("--stat", choices=("mean", "min", "max", "integral"), default="mean")
    parser.add_argument("--at-x", type=float)
    parser.add_argument("--weighted", choices=("area", "flux"), default="area")
    parser.add_argument("--seeds")
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="both")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args(argv)
    op = next(name for name in OPS if getattr(args, name) not in (None, False))
    if op in ("line", "points", "region", "section_mean", "between") and not args.field:
        parser.error(f"--{op.replace('_', '-')} needs --field")
    if op in ("section_mean", "flux") and args.at_x is None:
        parser.error(f"--{op.replace('_', '-')} needs --at-x")
    if op == "streamlines" and not args.seeds:
        parser.error("--streamlines needs --seeds")
    if op == "streamlines":
        args.field = args.field or "U"
    return args, op


def main(argv: list[str] | None = None) -> int:
    args, op = parse(argv)
    case = resolve_case(args.case)
    wanted = None if str(args.time).lower() in ("latest", "last", "") else float(args.time)
    mesh, time = open_case(case, wanted)
    if args.z is None:
        args.z = float(mesh.center[2])
    notes: list[str] = []
    units = None
    if args.field == "p_total" and "p_total" not in fields_of(mesh):
        units = derive_total_pressure(mesh, case, time, notes)
    elif args.field:
        need(mesh, args.field)
        units = units_of(case, time, args.field)
    if op == "flux":
        units = "m^3 s^-1" if units_of(case, time, "U") in ("m s^-1", None) else None
    if op in ("region", "flux") and mesh.bounds[5] - mesh.bounds[4] > 0:
        depth = mesh.bounds[5] - mesh.bounds[4]
        notes.append(f"the mesh is {depth:g} deep in z; a 2D case's totals are over that depth")
    result = {"op": op, "field": args.field, "time": time, "units": units, "case": str(case),
              "data": OPS[op](mesh, case, time, args, notes), "notes": notes}
    result = clean(result)
    if args.png:
        if op in ("line", "streamlines"):
            result["png"] = str(picture(result, mesh, args.z, args.png))
        else:
            notes.append(f"no picture for --{op.replace('_', '-')}; --png draws lines and streamlines")
    text = json.dumps(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
