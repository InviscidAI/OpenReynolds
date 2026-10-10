#!/usr/bin/env python3
"""A Felix case's `output/` read back: the run summary, fields and slices from the
`.vtu` files, values along a line, and the probes and forces CSVs as plots.

Felix writes everything a run produced into one directory (`<case>/output/` unless
io.yaml `output.dir` says otherwise): `summary.json`, `solution.vtu` (the final
state, or the last finite one after a NaN stop), `snapshot_NNNNNN.vtu` every
`output.every` steps, `probes.csv` and the forces CSV. The `.vtu` files carry point
arrays `velocity` and `pressure` (p/rho), plus `temperature`, `nu_tilde`, `nut` when
those equations ran, and `step` and `time` as field data. The CSVs open with `#`
lines saying what the numbers are, then a header; columns are found by that header.

Nothing here judges the run. The summary prints the health counters that are not
zero by name, and a force coefficient is printed only for a reference speed and area
you give -- the solver does not know them, and neither does this script.

    python3 felix_results.py summary /work/cavity
    python3 felix_results.py fields /work/cavity --file latest
    python3 felix_results.py slice /work/cavity --field velocity --component x --normal z -o u.png
    python3 felix_results.py line /work/cavity --start 0.5 0 0.5 --end 0.5 1 0.5 \\
        --field velocity --component x -n 129 --csv u_centreline.csv -o u_centreline.png
    python3 felix_results.py probes /work/cavity -o probes.png
    python3 felix_results.py forces /work/cyl --uref 0.2 --aref 0.002 -o forces.png
    python3 felix_results.py log felix.log -o residuals.png
    python3 felix_results.py all /work/cavity --normal z      # the standard set, into <case>/results

`--file` picks the result file: `solution` (default), `latest` (the newest
snapshot), a step number, a snapshot name, or a path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results  # noqa: E402  (aim, scalar_bar, colour limits, force_plot, tail_mean)

SNAPSHOT = re.compile(r"^snapshot_(\d+)\.vtu$")
HEALTH_ORDER = (
    "corrector_nonconverged", "energy_corrector_nonconverged", "sa_corrector_nonconverged",
    "gcr_nonconverged", "cg_nonconverged", "gram_breakdowns", "sa_gmres_nonconverged",
    "energy_gmres_nonconverged", "spalding_nonconverged", "gmres_capped",
)


# -- where things are -------------------------------------------------------------


def _yaml_block_value(text: str, block: str, key: str) -> str | None:
    """`key:` inside the top-level `block:` mapping of a small YAML file.

    The toolbox has no YAML library on the image, and Felix's io.yaml is a flat
    two-level file; this reads exactly that much and nothing more."""
    inside = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            inside = line.strip() == f"{block}:"
            continue
        if inside:
            found = re.match(rf"^\s+{re.escape(key)}\s*:\s*(.+?)\s*$", line)
            if found:
                return found.group(1).strip("'\"")
    return None


def _io_text(case: Path) -> str:
    case = Path(case)
    yaml_case = case / "case.yaml"
    name = "io.yaml"
    if yaml_case.is_file():
        found = re.search(r"^io\s*:\s*(\S+)", yaml_case.read_text(encoding="utf-8"), re.M)
        if found:
            name = found.group(1).strip("'\"")
    path = case / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def output_dir(case: Path | str) -> Path:
    """io.yaml `output.dir` resolved against the case, else `<case>/output`."""
    case = Path(case)
    found = _yaml_block_value(_io_text(case), "output", "dir")
    if not found:
        return case / "output"
    path = Path(found)
    return path if path.is_absolute() else case / path


def forces_csv(case: Path | str) -> Path:
    """io.yaml `forces.csv`, else `<output dir>/forces.csv`."""
    case = Path(case)
    found = _yaml_block_value(_io_text(case), "forces", "csv")
    if not found:
        return output_dir(case) / "forces.csv"
    path = Path(found)
    return path if path.is_absolute() else case / path


def snapshots(case: Path | str) -> list[tuple[int, Path]]:
    folder = output_dir(case)
    found = []
    if folder.is_dir():
        for path in folder.iterdir():
            match = SNAPSHOT.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
    return sorted(found)


def choose_file(case: Path | str, which: str | None = "solution") -> Path:
    """The result file `which` names: solution, latest, a step, a snapshot name, a path."""
    case = Path(case)
    which = (which or "solution").strip()
    folder = output_dir(case)
    snaps = snapshots(case)
    if which == "solution":
        path = folder / "solution.vtu"
    elif which == "latest":
        if not snaps:
            raise SystemExit(f"no snapshot_*.vtu in {folder}; io.yaml output.every was 0 or the run "
                             "stopped before the first one")
        path = snaps[-1][1]
    elif which.isdigit():
        wanted = [p for step, p in snaps if step == int(which)]
        if not wanted:
            have = ", ".join(str(step) for step, _ in snaps) or "none"
            raise SystemExit(f"no snapshot at step {which}; snapshots exist at steps: {have}")
        path = wanted[0]
    elif SNAPSHOT.match(which + ("" if which.endswith(".vtu") else ".vtu")):
        path = folder / (which if which.endswith(".vtu") else which + ".vtu")
    else:
        path = Path(which)
        if not path.is_absolute() and not path.exists():
            path = case / which
    if not path.is_file():
        raise SystemExit(f"{path} does not exist")
    return path


# -- summary.json ---------------------------------------------------------------------


def read_summary(case: Path | str) -> dict[str, Any]:
    path = output_dir(case) / "summary.json"
    if not path.is_file():
        raise SystemExit(
            f"no summary.json in {path.parent}: the run never wrote one (it failed before the "
            "first step, or was killed -- its step= lines are then the record of how far it got)")
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summary_lines(summary: dict[str, Any]) -> list[str]:
    """The run summary in a dozen lines: how it ended first, then how far it got."""
    lines = []
    run = summary.get("run", {})
    if summary.get("error"):
        lines.append(f"ERROR: {summary['error']}")
    lines.append(
        f"run: step {_num(run.get('terminated_step'))}, t = {_num(run.get('t_final'))}, "
        f"wall {_num(run.get('wall_time'))} s, snapshots: {_num(run.get('snapshots_written'))}")
    if "last_finite_step" in run:
        lines.append(f"last finite step {run['last_finite_step']} (solution.vtu holds it)")
    if run.get("resumed_from_step") is not None:
        lines.append(f"continued from step {run['resumed_from_step']}")
    if "dt_min" in run:
        lines.append(f"dt: last {_num(run.get('dt_last'))}, min {_num(run.get('dt_min'))}, "
                     f"max {_num(run.get('dt_max'))}")
    ptc = summary.get("ptc")
    if isinstance(ptc, dict) and "converged" in ptc:
        state = "PTC converged" if ptc["converged"] else "PTC did NOT converge (nsteps ran out)"
        rel = ", ".join(f"{k} {_num(v)}" for k, v in ptc.items() if k.endswith("_rel"))
        lines.append(f"{state}: {rel}" if rel else state)
    mesh = summary.get("mesh", {})
    if mesh:
        lines.append(f"mesh: {_num(mesh.get('n_nodes'))} nodes, {_num(mesh.get('n_elems'))} elements")
    health = summary.get("health", {})
    nonzero = [(k, health[k]) for k in HEALTH_ORDER if health.get(k)]
    nonzero += [(k, v) for k, v in health.items()
                if k not in HEALTH_ORDER and not k.endswith("ratio") and v]
    if nonzero:
        lines.append("health: " + ", ".join(f"{k} {v}" for k, v in nonzero))
    elif health:
        lines.append("health: every counter is 0")
    if health.get("corrector_worst_res_ratio") is not None:
        lines.append(f"corrector worst residual ratio {_num(health['corrector_worst_res_ratio'])}")
    iters = summary.get("iters", {})
    if iters:
        lines.append("iterations: " + ", ".join(f"{k} {v}" for k, v in iters.items()))
    for probe in summary.get("probes", []) or []:
        lines.append(f"probe {probe.get('name')} ({probe.get('var')} at {probe.get('location')}): "
                     f"{_num(probe.get('value'))}")
    files = summary.get("files", []) or []
    if files:
        kinds: dict[str, int] = {}
        for item in files:
            kinds[item.get("kind", "?")] = kinds.get(item.get("kind", "?"), 0) + 1
        lines.append("files: " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
    return lines


# -- CSVs -----------------------------------------------------------------------------


def read_csv(path: Path | str) -> dict[str, Any]:
    """A Felix CSV as {columns, rows, comments}: the `#` preamble, the header, and
    one row of floats per line. The shape is results.py's history, so its
    `column`, `tail_mean` and `force_plot` work on it."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{path} does not exist")
    comments, columns, rows = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            comments.append(line[1:].strip())
        elif not columns:
            columns = [c.strip() for c in line.split(",")]
        else:
            rows.append([_float(c) for c in line.split(",")])
    return {"columns": columns, "rows": rows, "comments": comments, "meta": {}}


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def column(table: dict[str, Any], name: str) -> np.ndarray:
    return results.column(table, name)


def _time_table(table: dict[str, Any]) -> dict[str, Any]:
    """The same table with `time` first and `step` dropped, the shape force_plot reads."""
    names = [c for c in table["columns"] if c != "step"]
    index = [table["columns"].index(c) for c in names]
    return {"columns": names, "rows": [[row[i] for i in index] for row in table["rows"]],
            "meta": {}}


def force_coefficients(table: dict[str, Any], uref: float, aref: float,
                       fraction: float = 0.25) -> dict[str, dict[str, float]]:
    """C = F / (0.5 uref^2 aref) for Fx, Fy, Fz (rho = 1: Felix forces are kinematic).
    Named Cd, Cl, Cz for flow along x with lift along y; rename them if the case is
    set up otherwise."""
    q = 0.5 * uref * uref * aref
    out = {}
    for name, label in (("Fx", "Cd"), ("Fy", "Cl"), ("Fz", "Cz")):
        values = column(table, name)
        if values.size:
            out[label] = {"last": float(values[-1] / q),
                          "tail_mean": float(results.tail_mean(values, fraction) / q)}
    return out


def forces_lines(table: dict[str, Any], uref: float | None = None, aref: float | None = None,
                 fraction: float = 0.25) -> list[str]:
    percent = int(round(fraction * 100))
    lines = [c for c in table.get("comments", []) if c.startswith("F = force")]
    steps = column(table, "step")
    times = column(table, "time")
    if times.size:
        lines.append(f"{times.size} rows, step {steps[0]:g}..{steps[-1]:g}, "
                     f"t = {times[0]:g}..{times[-1]:g}")
    for name in table["columns"]:
        if name in ("step", "time"):
            continue
        values = column(table, name)
        if values.size:
            lines.append(f"{name}: last {values[-1]:.6g}, mean of last {percent}% "
                         f"{results.tail_mean(values, fraction):.6g}")
    if uref and aref:
        for label, value in force_coefficients(table, uref, aref, fraction).items():
            lines.append(f"{label} (uref {uref:g}, aref {aref:g}): last {value['last']:.6g}, "
                         f"mean of last {percent}% {value['tail_mean']:.6g}")
    return lines


def plot_table(table: dict[str, Any], out: Path | str, title: str = "",
               columns: Sequence[str] | None = None) -> Path:
    """Every data column (or the ones named) against time, tail mean dashed."""
    timed = _time_table(table)
    chosen = list(columns) if columns else [c for c in timed["columns"][1:]
                                            if not (c.startswith("Fz") or c.endswith("_pres")
                                                    or c.endswith("_visc"))]
    return results.force_plot(timed, chosen, Path(out), title=title)


# -- the stderr record -------------------------------------------------------------------


_PAIR = re.compile(r"(\w+)=(\S+)")


def read_step_lines(text: str) -> dict[str, list[float]]:
    """Every `step=` record of a Felix log as columns, by field name. A field absent
    from a step (it did not run) is NaN there."""
    records = []
    for line in text.splitlines():
        if line.startswith("step="):
            records.append({k: _float(v) for k, v in _PAIR.findall(line)})
    names: list[str] = []
    for record in records:
        for name in record:
            if name not in names:
                names.append(name)
    return {name: [r.get(name, math.nan) for r in records] for name in names}


def plot_steps(series: dict[str, list[float]], out: Path | str) -> Path:
    """Residuals per step on a log axis, and dt beneath them."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(series.get("step", []), dtype=float)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    for name in ("res0", "resF", "gcr_res", "cg_res", "sa_gmres_res", "energy_gmres_res"):
        if name in series:
            values = np.asarray(series[name], dtype=float)
            values = np.where(values > 0, values, np.nan)
            top.semilogy(steps, values, linewidth=1.0, label=name)
    top.set_ylabel("residual")
    top.grid(True, which="both", alpha=0.3)
    top.legend(loc="best", fontsize="small")
    if "dt" in series:
        bottom.plot(steps, series["dt"], linewidth=1.0)
    bottom.set_ylabel("dt")
    bottom.set_xlabel("step")
    bottom.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


# -- fields (pyvista) ------------------------------------------------------------------


def read_vtu(path: Path | str):
    pv = results._pyvista()
    return pv.read(str(path))


def vtu_step_time(mesh) -> tuple[int | None, float | None]:
    step = mesh.field_data.get("step")
    when = mesh.field_data.get("time")
    return (int(np.asarray(step).ravel()[0]) if step is not None else None,
            float(np.asarray(when).ravel()[0]) if when is not None else None)


def field_table(mesh) -> dict[str, dict[str, Any]]:
    """Point arrays with their component count and range (of the magnitude for a vector)."""
    out = {}
    for name in mesh.point_data.keys():
        values = np.asarray(mesh.point_data[name], dtype=float)
        comps = 1 if values.ndim == 1 else values.shape[1]
        scalar = values if comps == 1 else np.linalg.norm(values, axis=1)
        finite = scalar[np.isfinite(scalar)]
        out[name] = {"components": comps,
                     "min": float(finite.min()) if finite.size else math.nan,
                     "max": float(finite.max()) if finite.size else math.nan}
    return out


def _component(values: np.ndarray, field: str, component: str | None) -> np.ndarray:
    if values.ndim == 1 or values.shape[1] == 1:
        return values.ravel()
    if component in (None, "mag"):
        return np.linalg.norm(values, axis=1)
    index = {"x": 0, "y": 1, "z": 2}.get(str(component))
    if index is None or index >= values.shape[1]:
        raise SystemExit(f"'{field}' has no component '{component}'")
    return values[:, index]


def sample_line(mesh, start: Sequence[float], end: Sequence[float], n: int,
                field: str, component: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The field interpolated at n points from start to end; s is the distance along.
    A point outside the mesh is NaN rather than the zero VTK fills in."""
    pv = results._pyvista()
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t = np.linspace(0.0, 1.0, n)
    points = start[None, :] + t[:, None] * (end - start)[None, :]
    probe = pv.PolyData(points).sample(mesh)
    if field not in probe.point_data:
        have = ", ".join(sorted(mesh.point_data.keys()))
        raise SystemExit(f"no point array '{field}'; the file has: {have}")
    values = _component(np.asarray(probe.point_data[field], dtype=float), field, component)
    valid = np.asarray(probe.point_data.get("vtkValidPointMask", np.ones(n)), dtype=bool)
    values = np.where(valid, values, np.nan)
    return t * float(np.linalg.norm(end - start)), values


def render_slice(mesh, field: str, component: str | None, normal: str, out: Path | str,
                 origin: Sequence[float] | None = None, title: str = "") -> Path:
    """One slice through the mesh coloured by a field, aimed and scaled the way
    results.py draws OpenFOAM slices. A signed component gets a colour range centred
    on zero; a magnitude or pressure a percentile range."""
    pv = results._pyvista()
    centre = np.asarray(origin if origin is not None else mesh.center, dtype=float)
    cut = mesh.slice(normal=results.NORMALS[normal], origin=centre)
    if cut.n_points == 0:
        raise SystemExit(f"the {normal}-normal plane through {tuple(centre)} misses the mesh")
    scalars = results.scalar_from(cut, field, component)  # Unavailable names what is there
    values = np.asarray(cut.point_data.get(scalars, cut.cell_data.get(scalars)), dtype=float)
    signed = component in ("x", "y", "z") or field in ("vorticity",)
    clim = results.symmetric_clim(values) if signed else results.robust_clim(values)
    cmap = "RdBu_r" if signed else "viridis"
    del pv
    return results.render_scalar(cut, scalars, Path(out), normal=normal, cmap=cmap, clim=clim,
                                 title=title or scalars)


def plot_line(s: np.ndarray, values: np.ndarray, out: Path | str, label: str) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(s, values, linewidth=1.2)
    axes.set_xlabel("distance along the line")
    axes.set_ylabel(label)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


# -- the command line -------------------------------------------------------------------


def _cmd_summary(args) -> int:
    print("\n".join(summary_lines(read_summary(args.case))))
    snaps = snapshots(args.case)
    if snaps:
        print(f"snapshots on disk: steps {', '.join(str(s) for s, _ in snaps)}")
    return 0


def _cmd_fields(args) -> int:
    path = choose_file(args.case, args.file)
    mesh = read_vtu(path)
    step, when = vtu_step_time(mesh)
    print(f"{path}: step {step}, t = {when}, {mesh.n_points} points, {mesh.n_cells} cells")
    print(f"bounds: {tuple(round(b, 6) for b in mesh.bounds)}")
    for name, info in field_table(mesh).items():
        kind = "vector |.|" if info["components"] > 1 else "scalar"
        print(f"  {name}: {info['components']} component(s), {kind} range "
              f"{info['min']:.6g} .. {info['max']:.6g}")
    return 0


def _cmd_slice(args) -> int:
    path = choose_file(args.case, args.file)
    mesh = read_vtu(path)
    step, when = vtu_step_time(mesh)
    label = f"{args.field}" + (f" {args.component}" if args.component else "")
    out = args.output or Path(args.case) / "results" / f"{args.field}_{args.component or 'mag'}.png"
    render_slice(mesh, args.field, args.component, args.normal, out, args.origin,
                 title=f"{label}  step {step}  t = {when:g}" if when is not None else label)
    print(out)
    return 0


def _cmd_line(args) -> int:
    mesh = read_vtu(choose_file(args.case, args.file))
    s, values = sample_line(mesh, args.start, args.end, args.n, args.field, args.component)
    name = f"{args.field}_{args.component}" if args.component else args.field
    start, end = np.asarray(args.start, float), np.asarray(args.end, float)
    if args.csv:
        rows = ["s,x,y,z," + name]
        for si, vi in zip(s, values):
            p = start + (end - start) * (si / max(float(np.linalg.norm(end - start)), 1e-300))
            rows.append(f"{si:.8g},{p[0]:.8g},{p[1]:.8g},{p[2]:.8g},{vi:.8g}")
        Path(args.csv).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(args.csv)
    if args.output:
        print(plot_line(s, values, args.output, name))
    if not args.csv and not args.output:
        for si, vi in zip(s, values):
            print(f"{si:.6g} {vi:.6g}")
    return 0


def _cmd_probes(args) -> int:
    table = read_csv(output_dir(args.case) / "probes.csv")
    for name in table["columns"][2:]:
        values = column(table, name)
        print(f"{name}: last {values[-1]:.6g}, mean of last 25% {results.tail_mean(values):.6g}")
    out = args.output or Path(args.case) / "results" / "probes.png"
    print(plot_table(table, out, "probes"))
    return 0


def _cmd_forces(args) -> int:
    path = Path(args.csv) if args.csv else forces_csv(args.case)
    table = read_csv(path)
    print("\n".join(forces_lines(table, args.uref, args.aref)))
    out = args.output or Path(args.case) / "results" / "forces.png"
    print(plot_table(table, out, "forces"))
    return 0


def _cmd_log(args) -> int:
    series = read_step_lines(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    if not series:
        raise SystemExit(f"no step= lines in {args.log}")
    print(f"{len(series['step'])} step lines, last step {series['step'][-1]:g}")
    out = args.output or Path(args.log).with_suffix(".residuals.png")
    print(plot_steps(series, out))
    return 0


def _cmd_all(args) -> int:
    """summary + |u| and pressure slices + probes and forces plots, each attempted on
    its own: what cannot be made is said and the rest still are."""
    out = Path(args.out) if args.out else Path(args.case) / "results"
    out.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []
    try:
        print("\n".join(summary_lines(read_summary(args.case))))
    except SystemExit as why:
        skipped.append(f"summary: {why}")
    try:
        path = choose_file(args.case, args.file)
        mesh = read_vtu(path)
        step, when = vtu_step_time(mesh)
        tag = f"step {step}  t = {when:g}" if when is not None else path.name
        for field, component, name in (("velocity", "mag", "velocity_mag.png"),
                                       ("pressure", None, "pressure.png")):
            try:
                made.append(render_slice(mesh, field, component, args.normal, out / name,
                                         title=f"{field} {component or ''}  {tag}"))
            except (SystemExit, results.Unavailable) as why:
                skipped.append(f"{name}: {why}")
    except SystemExit as why:
        skipped.append(f"slices: {why}")
    for name, path in (("probes", output_dir(args.case) / "probes.csv"),
                       ("forces", forces_csv(args.case))):
        if path.is_file():
            table = read_csv(path)
            made.append(plot_table(table, out / f"{name}.png", name))
            if name == "forces":
                print("\n".join(forces_lines(table)))
        else:
            skipped.append(f"{name}: no {path.name} (the case declares no {name})")
    for path in made:
        print(path)
    for line in skipped:
        print(f"skipped {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def case_parser(name: str, help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("case", type=Path, help="the case directory (holding case.yaml)")
        return p

    case_parser("summary", "summary.json in words").set_defaults(run=_cmd_summary)
    p = case_parser("fields", "arrays in a result file, with ranges")
    p.add_argument("--file", default="solution")
    p.set_defaults(run=_cmd_fields)
    p = case_parser("slice", "one coloured slice to PNG")
    p.add_argument("--field", default="velocity")
    p.add_argument("--component", default=None, help="x, y, z or mag (vectors)")
    p.add_argument("--normal", default="z", choices=sorted(results.NORMALS))
    p.add_argument("--origin", type=float, nargs=3, default=None)
    p.add_argument("--file", default="solution")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.set_defaults(run=_cmd_slice)
    p = case_parser("line", "a field sampled along a straight line")
    p.add_argument("--start", type=float, nargs=3, required=True)
    p.add_argument("--end", type=float, nargs=3, required=True)
    p.add_argument("--field", default="velocity")
    p.add_argument("--component", default=None)
    p.add_argument("-n", type=int, default=101)
    p.add_argument("--file", default="solution")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.set_defaults(run=_cmd_line)
    p = case_parser("probes", "probes.csv: last values and a plot")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.set_defaults(run=_cmd_probes)
    p = case_parser("forces", "the forces CSV: last values, tail means, a plot")
    p.add_argument("--csv", default=None, help="the CSV, if not where io.yaml says")
    p.add_argument("--uref", type=float, default=None, help="reference speed for coefficients")
    p.add_argument("--aref", type=float, default=None, help="reference area for coefficients")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.set_defaults(run=_cmd_forces)
    p = sub.add_parser("log", help="step= lines of a Felix log: residual and dt plot")
    p.add_argument("log", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.set_defaults(run=_cmd_log)
    p = case_parser("all", "summary, slices, probes and forces plots")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--normal", default="z", choices=sorted(results.NORMALS))
    p.add_argument("--file", default="solution")
    p.set_defaults(run=_cmd_all)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except results.Unavailable as missing:
        print(f"felix_results: {missing}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
