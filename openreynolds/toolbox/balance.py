#!/usr/bin/env python3
"""Where the flow goes inside a duct and what it costs: the per-patch split, whether it
closes, and the two dimensionless numbers.

    python3 balance.py CASE
    python3 balance.py CASE --drop inlet outlet [--u-ref 2.0] [--rho 998.2]
    python3 balance.py CASE --sections 0.2 0.8 [--u-ref 2.0]
    python3 balance.py CASE --drop inlet outlet --u-ref 2.0 --length 1.5 --dh 0.05

    common: [--time latest|T] [--zero-tol 1e-6] [--json balance.json] [--png balance.png]

`probe.py` already measures a pressure drop between two x-sections (`--between`, with the
flux-weighted mixed mean that is the honest average for a duct) and the flux through one
(`--flux`), and it is what this reads through. What it has no way to say is how the
boundary divides: a manifold with three outlets gets one number per section, nothing that
says 14/33/52, nothing that says whether what went in came out, and no coefficient -- so a
loss coefficient was quoted from Idelchik rather than measured off the solve that was just
paid for. The missing three are what is here: the split, the closure, and K and f.

Inflow and outflow are told apart by the sign of U . A over each patch and never by a
patch's name, because a name is a convention somebody else chose and a sign is a
measurement. OpenFOAM winds a boundary face out of the domain, so outflow is positive and
an inlet comes out negative however it is spelled -- including a patch called `outlet`
that this case happens to be running backwards through, which is the case where a
name-based reading is both wrong and plausible.

A closure error is the most useful single diagnostic an internal-flow case has, so it is
reported as a number -- absolute and against the throughflow -- with no verdict attached:
what counts as closed depends on the solver's own continuity tolerance, and a case that
does not close is usually either unconverged or has a patch nobody told the analysis
about, which is why the patches that were counted are listed by name beside it.

Units, which is where a duct number most often goes quietly wrong. An incompressible
OpenFOAM `p` is p/rho in m^2 s^-2 and a compressible one is absolute static pressure in
Pa; both print the same field name and both look plausible. Which one this case wrote is
read off `p`'s own `dimensions` entry (`preflight.py` parses the same entry; this goes
through `probe.units_of`) and stated in the report, and every number printed carries its
unit. rho cancels in K -- dp_total is already dp/rho when p is kinematic -- which is why a
kinematic case gives K with no density anywhere in it, and why a dp asked for in Pa off
the same case does need one. Static and total pressure are always reported together,
because quoting one as the other is the classic error and the gap between them is the
velocity head that tells you which you are looking at.

What it will not do is default a constant. rho, U_ref, L and D are refused by name, each
naming the flag that would supply it, on `benchmarks/moving_mesh/grade.py`'s rule: a K off
an assumed reference velocity is out by a factor and looks measured. The balance itself
exits 0 whatever it finds; a refusal is a missing constant, not a finding.

Output: the table on stdout, `--json` the same facts as a sidecar, `--png` the per-patch
split as a bar chart with the drop summary under it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402
import render  # noqa: E402

KINEMATIC = "m^2 s^-2"
PASCAL = "kg m^-1 s^-2"


# -- reading the boundary ----------------------------------------------------------------

def patch_surfaces(case: Path, time: float | None):
    """Every boundary patch as its own surface carrying the fields on it, and the time.

    `render.open_case` reads the internal mesh and leaves the boundary behind, so the
    patches are read here: `enable_all_patch_arrays` is the one reader switch it does not
    touch. The fallback is the same one and for the same reason -- the VTK reader cannot
    expand a `$variable`, a `flowVelocity` symbol or an `#includeEtc` in `0/`, and a case
    written that way is not a broken case.
    """
    pv = render._pyvista()
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    try:
        reader = pv.OpenFOAMReader(str(foam))
        times = list(reader.time_values)
        chosen = times[-1] if times else 0.0
        if time is not None and times:
            chosen = min(times, key=lambda t: abs(t - time))
        reader.set_active_time_value(chosen)
        reader.enable_all_patch_arrays()
        block = reader.read()
        boundary = block["boundary"] if "boundary" in (block.keys() or []) else None
        if boundary is None or not list(boundary.keys()):
            raise ValueError("the reader returned no boundary blocks")
        return {name: boundary[name] for name in boundary.keys()}, float(chosen), "OpenFOAMReader"
    except Exception as exc:  # the reader choked -- go through foamToVTK, as render.py does
        return patches_via_foamtovtk(case, time, exc)


def patches_via_foamtovtk(case: Path, time: float | None, why: Exception):
    """The patches through `foamToVTK`, which reads with OpenFOAM's own libraries.

    It writes `VTK/<name>_<n>/boundary/<patch>.vtp` beside the `internal.vtu`
    `render.newest_internal_vtk` already knows how to find, so the newest write is located
    by that and the patches taken from the directory next to it rather than from a guessed
    layout.
    """
    import subprocess

    pv = render._pyvista()
    argv = ["foamToVTK", "-case", str(case)]
    argv += ["-latestTime"] if time is None else ["-time", f"{time:g}"]
    proc = subprocess.run(argv, cwd=str(case), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"could not read {case}: OpenFOAMReader failed ({type(why).__name__}: {why}) "
            f"and foamToVTK also failed (exit {proc.returncode}). Last lines:\n"
            + "\n".join(proc.stderr.splitlines()[-8:]))
    newest = render.newest_internal_vtk(case / "VTK")
    folder = newest.parent if newest is not None else None
    vtps = sorted((folder / "boundary").glob("*.vtp")) if folder else []
    if not vtps:
        raise SystemExit(
            f"foamToVTK wrote no boundary patches under {case / 'VTK'}, and the per-patch "
            "split is the whole point of this; `foamToVTK -help` says whether this build "
            "writes them")
    return ({path.stem: pv.read(str(path)) for path in vtps},
            float(render.time_from_stem(folder.name)), "foamToVTK")


def pressure_convention(case: Path, time: float):
    """Which `p` this case wrote, off its own `dimensions` entry rather than by guessing.

    Both conventions print the same field name and plausible numbers, and reading one as
    the other is out by rho -- a factor of about 1000 for water, in both directions.
    `0/p` is read when the solved time does not carry the field.
    """
    units = probe.units_of(case, time, "p") or probe.units_of(case, 0.0, "p")
    if units == KINEMATIC:
        return "kinematic", units
    if units == PASCAL:
        return "Pa", units
    return "unknown", units


# -- the arithmetic, which needs no mesh -------------------------------------------------

def face_area_vectors(surf):
    """Each triangle's outward area vector, taken from the winding OpenFOAM wrote.

    Triangulated first so one cross product covers quads, triangles and the odd polygon,
    and because the area vector of a triangle is unambiguous where a normal filter is not:
    `vtkPolyDataNormals` has a consistency pass that will quietly re-orient a patch to
    agree with itself, which on a wall band wrapping a duct is exactly the wrong answer,
    and its pyvista keyword names have moved between releases while a cross product has
    not. Triangulation carries the cell data down to each piece, so a triangle keeps its
    face's U and p and the sums below are the face sums.
    """
    tri = surf.triangulate()
    if tri.n_cells == 0:
        return tri, np.zeros((0, 3))
    points = np.asarray(tri.points, float)
    flat = np.asarray(tri.faces).reshape(-1)
    # VTK's own layout is [3, i, j, k] per triangle; newer pyvista hands back the regular
    # faces with no leading count. Ask the size which one arrived.
    conn = (flat.reshape(-1, 4)[:, 1:] if flat.size == 4 * tri.n_cells
            else flat.reshape(tri.n_cells, 3))
    a, b, c = points[conn[:, 0]], points[conn[:, 1]], points[conn[:, 2]]
    return tri, 0.5 * np.cross(b - a, c - a)


def weighted(values: np.ndarray, weights: np.ndarray) -> float | None:
    """A weighted mean, or None where the weights sum to nothing.

    A wall carries no flux, so it has no flux-weighted mean; that is reported as absent
    rather than as zero, which is a pressure.
    """
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if abs(total) > 0 else None


def patch_record(name: str, surf, rho: float | None, case: Path, time: float,
                 notes: list[str]) -> dict:
    """One patch measured: area, signed flux, and its two pressure averages.

    The area average is what a patch looks like; the flux-weighted (mixed) mean is the one
    a loss between two stations is honestly taken with, which is the average `probe.py`
    already offers at a section and the one every pressure drop below uses.
    """
    tri, areas = face_area_vectors(surf)
    area = np.linalg.norm(areas, axis=1)
    if "U" not in tri.cell_data or "p" not in tri.cell_data:
        missing = [f for f in ("U", "p") if f not in tri.cell_data]
        raise SystemExit(f"patch {name} carries no {' and no '.join(missing)}; "
                         "a flow balance needs both on every patch")
    volume_flux = np.sum(np.asarray(tri.cell_data["U"], float) * areas, axis=1)
    density = (np.asarray(tri.cell_data["rho"], float) if "rho" in tri.cell_data
               else (rho if rho is not None else None))
    mass_flux = None if density is None else volume_flux * density
    total_pressure(tri, case, time, rho, notes)
    weights = volume_flux if mass_flux is None else mass_flux
    pressure = np.asarray(tri.cell_data["p"], float)
    total = np.asarray(tri.cell_data["p_total"], float)
    return {
        "patch": name,
        "faces": int(surf.n_cells),
        "area": float(area.sum()),
        "volume_flux": float(volume_flux.sum()),
        "mass_flux": None if mass_flux is None else float(np.sum(mass_flux)),
        "p_area": weighted(pressure, area),
        "p_mixed": weighted(pressure, weights),
        "p_total_area": weighted(total, area),
        "p_total_mixed": weighted(total, weights),
    }


def total_pressure(tri, case: Path, time: float, rho: float | None,
                   notes: list[str]) -> None:
    """`p_total` on the patch, through `probe.py`, which already owns both conventions.

    The one thing it cannot do is take rho from a flag, because nothing on a case's disk
    carries it: it refuses a Pa case that wrote no `rho` field, and that refusal is where
    `--rho` gets its chance before being handed back as the named flag it is.
    """
    try:
        probe.derive_total_pressure(tri, case, time, notes)
    except SystemExit as exc:
        if rho is None:
            raise SystemExit(f"{exc}; pass --rho to say which density") from None
        half_u2 = 0.5 * np.sum(np.asarray(tri.cell_data["U"], float) ** 2, axis=1)
        tri.cell_data["p_total"] = np.asarray(tri.cell_data["p"], float) + rho * half_u2
        notes.append("p_total derived here: p + rho |U|^2/2, rho from --rho")


def split(rows: list[dict], zero_tol: float) -> dict:
    """Each patch's direction and share, and the closure over all of them.

    The share is of the total inflow for an inflow patch and of the total outflow for an
    outflow patch, so each set sums to 100% and neither is diluted by the other. A patch
    whose flux is under `zero_tol` of the largest is reported as closed with no share at
    all: a wall's 1e-18 is not a 0.0000001% outlet, and a share list that includes it is a
    share list nobody can read.
    """
    key = "mass_flux" if rows and rows[0]["mass_flux"] is not None else "volume_flux"
    biggest = max((abs(row[key]) for row in rows), default=0.0)
    cut = zero_tol * biggest
    for row in rows:
        flux = row[key]
        row["flux"] = flux
        row["direction"] = ("closed" if abs(flux) <= cut
                            else "inflow" if flux < 0 else "outflow")
    inflow = -sum(row["flux"] for row in rows if row["direction"] == "inflow")
    outflow = sum(row["flux"] for row in rows if row["direction"] == "outflow")
    for row in rows:
        total = inflow if row["direction"] == "inflow" else outflow
        row["share_pct"] = (None if row["direction"] == "closed" or not total
                            else 100.0 * abs(row["flux"]) / total)
    throughflow = 0.5 * (inflow + outflow)
    error = inflow - outflow
    return {
        "flux_quantity": "mass" if key == "mass_flux" else "volume",
        "flux_units": "kg s^-1" if key == "mass_flux" else "m^3 s^-1",
        "inflow": inflow, "outflow": outflow, "throughflow": throughflow,
        "closure_error": error,
        "closure_error_relative": error / throughflow if throughflow else None,
        "inflow_patches": [r["patch"] for r in rows if r["direction"] == "inflow"],
        "outflow_patches": [r["patch"] for r in rows if r["direction"] == "outflow"],
        "closed_patches": [r["patch"] for r in rows if r["direction"] == "closed"],
        "zero_tol": zero_tol,
    }


def _need(value, what: str, where: str, flag: str):
    """grade.py:125's rule: stop and name the flag, rather than default a constant."""
    if value is None:
        raise SystemExit(f"could not read {what} from {where}; pass {flag}")
    return value


def patch_drop(rows: list[dict], a: str, b: str) -> dict:
    """The drop between two patches, static and total, on the mixed mean of each.

    Both are reported because quoting one as the other is the classic error: across a
    constant-area duct they agree and the friction is all of it, across an expansion the
    static pressure can rise while the total pressure falls, and a K taken off the static
    drop there comes out negative.
    """
    found = {row["patch"]: row for row in rows}
    for name in (a, b):
        if name not in found:
            raise SystemExit(f"this case has no patch '{name}'; it has: "
                             + ", ".join(sorted(found)))
    up, down = found[a], found[b]
    for row, name in ((up, a), (down, b)):
        if row["p_mixed"] is None:
            raise SystemExit(
                f"patch '{name}' carries no flux, so it has no mixed mean and no drop can "
                "be referred to it; name two patches the flow crosses, or use --sections")
    return {
        "from": a, "to": b, "weighted": "flux (mixed mean)",
        "p_from": up["p_mixed"], "p_to": down["p_mixed"],
        "dp_static": up["p_mixed"] - down["p_mixed"],
        "p_total_from": up["p_total_mixed"], "p_total_to": down["p_total_mixed"],
        "dp_total": up["p_total_mixed"] - down["p_total_mixed"],
    }


def section_drop(case: Path, time: float, x0: float, x1: float, notes: list[str]) -> dict:
    """The same drop between two x-sections, straight through `probe.py`'s section mean.

    Sections rather than patches wherever the loss wanted is across a feature in the middle
    of a duct: `ladder.py`'s one-geometric-feature rung measures between two developed
    sections on purpose, so that the straight-pipe friction can be subtracted and the
    fitting is not charged for it.
    """
    mesh, chosen = probe.open_case(case, time)
    if "p_total" not in probe.fields_of(mesh):
        probe.derive_total_pressure(mesh, case, chosen, notes)
    out = {"x0": x0, "x1": x1, "weighted": "flux (mixed mean)"}
    for field, label in (("p", "static"), ("p_total", "total")):
        upstream = probe.section_mean(mesh, x0, field, "flux")
        downstream = probe.section_mean(mesh, x1, field, "flux")
        out[f"{field}_x0"], out[f"{field}_x1"] = upstream, downstream
        out[f"dp_{label}"] = upstream - downstream
    return out


def coefficients(drop: dict, convention: str, units: str | None, rho: float | None,
                 u_ref: float | None, length: float | None, dh: float | None) -> dict:
    """K and the Darcy friction factor, each refused rather than assumed.

    K = dp_total/(0.5 rho U_ref^2), the number `ladder.py`'s one-geometric-feature rung
    checks against Borda-Carnot's K = (1 - A1/A2)^2 for a sudden expansion and against the
    handbook tables for a bend or a tee. f = dp/((L/D)(0.5 rho U^2)) is the Darcy factor,
    taken off the static drop, which in the constant-area duct the definition assumes is
    also the total drop. rho cancels in both when p is kinematic, since dp is then already
    dp/rho; nothing else does. U_ref is not read off the case even though a mean speed
    could be formed from a flux and an area: "a K that is out by a factor is usually a
    reference velocity taken at the wrong section", so which section is the caller's to
    say.
    """
    out: dict = {}
    if u_ref is None:
        return out
    if convention == "kinematic":
        dynamic = 0.5 * u_ref ** 2
        basis = f"0.5 U_ref^2 = {dynamic:.6g} m^2 s^-2 (rho cancels: p is kinematic)"
    else:
        _need(rho, "a density", f"the case (p is in {units or 'unknown units'}, not "
              f"{KINEMATIC}, so rho does not cancel)", "--rho")
        dynamic = 0.5 * rho * u_ref ** 2
        basis = f"0.5 rho U_ref^2 = {dynamic:.6g} Pa"
    out["u_ref"] = u_ref
    out["dynamic_head"] = dynamic
    out["dynamic_head_basis"] = basis
    out["K"] = drop["dp_total"] / dynamic
    out["K_definition"] = "K = dp_total / (0.5 rho U_ref^2), on the mixed means"
    if length is not None or dh is not None:
        _need(length, "a duct length", "the case", "--length")
        _need(dh, "a hydraulic diameter", "the case", "--dh")
        out["length"] = length
        out["hydraulic_diameter"] = dh
        out["f_darcy"] = drop["dp_static"] / ((length / dh) * dynamic)
        out["f_definition"] = ("f = dp_static / ((L/D) 0.5 rho U_ref^2); the static drop, "
                               "which in the constant-area duct the definition assumes is "
                               "also the total drop")
    return out


# -- the report --------------------------------------------------------------------------

def table(result: dict) -> str:
    """The same facts as `--json`, laid out to be read."""
    balance = result["balance"]
    flux_units = balance["flux_units"]
    pressure_units = result["pressure_units"] or "unknown"
    lines = [
        f"case: {result['case']}   time {result['time']:g}   read by {result['reader']}",
        f"p is {result['pressure_convention']} ({pressure_units}); every pressure below is "
        f"in {pressure_units}",
        f"flux is {balance['flux_quantity']} flux in {flux_units}, signed: negative is into "
        "the domain",
        "",
        f"{'patch':<18}{'faces':>8}{'area m^2':>14}{'flux':>14}"
        f"{'share':>10}  {'direction':<9}{'p area':>13}{'p mixed':>13}{'p_total mixed':>15}",
    ]
    for row in result["patches"]:
        def number(value, width):  # a patch with no flux has no mixed mean, and says so
            return f"{'--':>{width}}" if value is None else f"{value:>{width}.6g}"

        share = "" if row["share_pct"] is None else f"{row['share_pct']:.2f}%"
        lines.append(
            f"{row['patch']:<18}{row['faces']:>8}{number(row['area'], 14)}"
            f"{number(row['flux'], 14)}{share:>10}  {row['direction']:<9}"
            f"{number(row['p_area'], 13)}{number(row['p_mixed'], 13)}"
            f"{number(row['p_total_mixed'], 15)}")
    relative = balance["closure_error_relative"]
    lines += [
        "",
        f"in {balance['inflow']:.6g}, out {balance['outflow']:.6g} {flux_units}; "
        f"closure error {balance['closure_error']:.6g} {flux_units}"
        + ("" if relative is None else f", {100 * relative:.4g}% of the throughflow "
                                       f"({balance['throughflow']:.6g} {flux_units})"),
        f"  counted as inflow: {', '.join(balance['inflow_patches']) or 'none'}",
        f"  counted as outflow: {', '.join(balance['outflow_patches']) or 'none'}",
        f"  closed (|flux| under {balance['zero_tol']:g} of the largest): "
        f"{', '.join(balance['closed_patches']) or 'none'}",
    ]
    drop = result.get("drop")
    if drop:
        where = (f"patches {drop['from']} -> {drop['to']}" if "from" in drop
                 else f"x = {drop['x0']:g} -> x = {drop['x1']:g}")
        lines += [
            "",
            f"pressure drop, {where}, on the {drop['weighted']}:",
            f"  static {drop['dp_static']:.6g} {pressure_units}",
            f"  total  {drop['dp_total']:.6g} {pressure_units}   "
            f"(the gap is the velocity head: "
            f"{drop['dp_total'] - drop['dp_static']:.6g})",
        ]
    coef = result.get("coefficients") or {}
    if coef:
        lines += ["", f"  {coef['dynamic_head_basis']}",
                  f"  K = {coef['K']:.6g}   ({coef['K_definition']})"]
        if "f_darcy" in coef:
            lines.append(f"  f = {coef['f_darcy']:.6g}   (L = {coef['length']:g} m, "
                         f"D = {coef['hydraulic_diameter']:g} m)")
    for note in result["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def picture(result: dict, path: Path) -> Path:
    """The split as a bar chart, with the closure and the drop written under it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    balance = result["balance"]
    rows = [row for row in result["patches"] if row["share_pct"] is not None]
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {"inflow": "#2b6cb0", "outflow": "#c05621"}
    ax.bar([row["patch"] for row in rows], [row["share_pct"] for row in rows],
           color=[colours[row["direction"]] for row in rows])
    for index, row in enumerate(rows):
        ax.text(index, row["share_pct"], f"{row['share_pct']:.1f}%\n{row['flux']:.3g}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"share of total {balance['flux_quantity']} inflow / outflow  [%]")
    ax.set_ylim(0, 118)
    ax.set_title(f"{Path(result['case']).name}  t = {result['time']:g}   "
                 f"blue into the domain, orange out of it")
    ax.grid(axis="y", alpha=0.3)
    relative = balance["closure_error_relative"]
    caption = (f"in {balance['inflow']:.6g} / out {balance['outflow']:.6g} "
               f"{balance['flux_units']}; closure error {balance['closure_error']:.6g}"
               + ("" if relative is None else f" ({100 * relative:.3g}%)"))
    drop = result.get("drop")
    if drop:
        caption += (f"\ndp static {drop['dp_static']:.6g}, total {drop['dp_total']:.6g} "
                    f"{result['pressure_units'] or '?'}")
    coef = result.get("coefficients") or {}
    if "K" in coef:
        caption += f"   K = {coef['K']:.4g}"
    if "f_darcy" in coef:
        caption += f"   f = {coef['f_darcy']:.4g}"
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=9)
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
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--drop", nargs=2, metavar=("FROM", "TO"),
                       help="pressure drop between two patches, on their mixed means")
    where.add_argument("--sections", nargs=2, type=float, metavar=("X0", "X1"),
                       help="pressure drop between two x-sections, through probe.py")
    parser.add_argument("--rho", type=float,
                        help="density in kg/m^3; needed for a mass flux, and for K and f "
                             "wherever p is in Pa rather than kinematic")
    parser.add_argument("--u-ref", type=float,
                        help="the reference velocity K and f are referred to, in m/s")
    parser.add_argument("--length", type=float, help="duct length L in m, for f")
    parser.add_argument("--dh", type=float, help="hydraulic diameter D in m, for f")
    parser.add_argument("--zero-tol", type=float, default=1e-6,
                        help="a patch under this fraction of the largest |flux| is "
                             "reported as closed (default 1e-6)")
    parser.add_argument("--json", type=Path, dest="json_out")
    parser.add_argument("--png", type=Path)
    args = parser.parse_args(argv)
    if (args.length is not None or args.dh is not None) and args.u_ref is None:
        parser.error("f needs --u-ref as well as --length and --dh")
    if args.u_ref is not None and not (args.drop or args.sections):
        parser.error("K and f are a drop divided by a head; name the drop with --drop or "
                     "--sections")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    case = probe.resolve_case(args.case)
    wanted = None if str(args.time).lower() in ("latest", "last", "") else float(args.time)
    notes: list[str] = []
    with contextlib.redirect_stdout(sys.stderr):  # stdout is the report channel
        surfaces, time, reader = patch_surfaces(case, wanted)
        rows = [patch_record(name, surfaces[name], args.rho, case, time, notes)
                for name in sorted(surfaces)]
    convention, units = pressure_convention(case, time)
    balance = split(rows, args.zero_tol)
    if balance["flux_quantity"] == "volume":
        notes.append("no density: the balance is on volume flux; --rho makes it mass flux")
    drop = None
    if args.drop:
        drop = patch_drop(rows, *args.drop)
    elif args.sections:
        with contextlib.redirect_stdout(sys.stderr):
            drop = section_drop(case, time, *args.sections, notes)
    result = {
        "case": str(case), "time": time, "reader": reader,
        "pressure_convention": convention, "pressure_units": units,
        "patches": rows, "balance": balance, "drop": drop,
        "coefficients": coefficients(drop, convention, units, args.rho, args.u_ref,
                                     args.length, args.dh) if drop else {},
        "notes": sorted(set(notes)),
    }
    if args.png:
        result["png"] = str(picture(result, args.png))
    result = probe.clean(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(table(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
