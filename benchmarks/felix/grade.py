#!/usr/bin/env python3
"""Grade a Felix benchmark run from what the solver wrote.

    python benchmarks/felix/grade.py cavity <case> [--file solution.vtu] [--json]
    python benchmarks/felix/grade.py cylinder <case> [--json]

`cavity`: the lid-driven cavity at Re 1000 against Ghia, Ghia & Shin (1982), J.
Comput. Phys. 48, 387-411, Table I (u along the vertical centreline) and Table II
(v along the horizontal centreline). `cylinder`: the Schafer & Turek (1996) DFG
benchmark 2D-1 -- a cylinder in a channel at Re 20 -- whose drag coefficient is
5.5795 (their reference interval 5.57-5.59).

Only the solver's own files are read: case.yaml for the viscosity, the output
directory's `solution.vtu` for the field and the geometry, and the forces CSV.
Never the run's own analysis. Where the lid is, which way it moves, the channel's
inflow peak and the cylinder's diameter are all read off the solution, so a case set
up in other axes or another orientation is graded the same way; a constant that
cannot be found stops the grader with the flag that supplies it.

Exit status: 0 when every check passes, 1 when one fails, 2 when the case cannot be
read.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

THRESHOLDS = {
    # Capstone 1 of docs/felix-mode-acceptance.md: "centerline u within 2% RMS of
    # Ghia" -- RMS of (u - u_Ghia) over Ghia's 17 points, as a fraction of the lid
    # speed. The same bound is applied to v. A 128x128 quasi-2D Felix run (the test
    # fixture) gives 0.45% on u and 1.06% on v.
    "cavity_rms": 0.02,
    # Re is a setting, not a result: a run at another Re is another benchmark.
    "reynolds_rel": 0.01,
    # Schafer & Turek give Cd in [5.57, 5.59] as the band their contributors' best
    # solutions fell in; 2% admits a well-resolved engineering mesh (6765 hexes give
    # -1.04%) and fails one that is not.
    "cd_rel": 0.02,
    # The inflow is set, not computed: Um = 0.3 to round-off.
    "um_rel": 0.01,
    "diameter_rel": 0.02,
    # A force record whose last 10% of rows still move by more than this (relative)
    # is not reported as steady. Reported, not graded: a PTC run stops when it is.
    "steady_rel": 1e-3,
}

# Ghia, Ghia & Shin (1982), Table I, Re = 1000: (y, u) along x = 0.5.
GHIA_RE1000_U = (
    (1.0000, 1.00000), (0.9766, 0.65928), (0.9688, 0.57492), (0.9609, 0.51117),
    (0.9531, 0.46604), (0.8516, 0.33304), (0.7344, 0.18719), (0.6172, 0.05702),
    (0.5000, -0.06080), (0.4531, -0.10648), (0.2813, -0.27805), (0.1719, -0.38289),
    (0.1016, -0.29730), (0.0703, -0.22220), (0.0625, -0.20196), (0.0547, -0.18109),
    (0.0000, 0.00000),
)
# Table II, Re = 1000: (x, v) along y = 0.5.
GHIA_RE1000_V = (
    (1.0000, 0.00000), (0.9688, -0.21388), (0.9609, -0.27669), (0.9531, -0.33714),
    (0.9453, -0.39188), (0.9063, -0.51550), (0.8594, -0.42665), (0.8047, -0.31966),
    (0.5000, 0.02526), (0.2344, 0.32235), (0.2266, 0.33075), (0.1563, 0.37095),
    (0.0938, 0.32627), (0.0781, 0.30353), (0.0703, 0.29012), (0.0625, 0.27485),
    (0.0000, 0.00000),
)

# Schafer, M. & Turek, S. (1996), "Benchmark computations of laminar flow around a
# cylinder", Notes on Numerical Fluid Mechanics 52, 547-566: test case 2D-1.
ST_2D1 = {
    "cd": 5.5795, "cd_interval": (5.57, 5.59), "cl_interval": (0.0104, 0.0110),
    "re": 20, "um": 0.3, "diameter": 0.1, "centre": (0.2, 0.2), "length": 2.2,
    "height": 0.41, "nu": 1e-3,
}

AXES = "xyz"


class GradeError(Exception):
    """The case cannot be graded; the message says what is missing and which flag
    supplies it."""


# -- reading the case ---------------------------------------------------------------


def _yaml_value(text: str, key: str) -> str | None:
    found = re.search(rf"^{re.escape(key)}\s*:\s*([^#\n]+)", text, re.M)
    return found.group(1).strip().strip("'\"") if found else None


def _block_value(text: str, block: str, key: str) -> str | None:
    inside = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            inside = line.strip() == f"{block}:"
            continue
        if inside:
            found = re.match(rf"^\s+{re.escape(key)}\s*:\s*(.+?)\s*$", line)
            if found:
                return found.group(1).strip("'\"")
    return None


def _case_text(case: Path) -> str:
    path = case / "case.yaml"
    if not path.is_file():
        raise GradeError(f"{path} does not exist: give the case directory (the one holding case.yaml)")
    return path.read_text(encoding="utf-8")


def _io_text(case: Path) -> str:
    name = _yaml_value(_case_text(case), "io") or "io.yaml"
    path = case / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def read_nu(case: Path, override: float | None) -> float:
    if override is not None:
        return override
    raw = _yaml_value(_case_text(case), "nu")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise GradeError(f"case.yaml `nu` is {raw!r}, not a number; pass --nu") from None


def output_dir(case: Path) -> Path:
    found = _block_value(_io_text(case), "output", "dir")
    if not found:
        return case / "output"
    path = Path(found)
    return path if path.is_absolute() else case / path


def result_file(case: Path, file: str | None) -> Path:
    path = Path(file) if file else output_dir(case) / "solution.vtu"
    if not path.is_absolute() and file and not path.exists():
        path = case / file
    if not path.is_file():
        raise GradeError(f"{path} does not exist: the run wrote no solution there (pass --file)")
    return path


def read_solution(path: Path):
    import pyvista as pv

    mesh = pv.read(str(path))
    if "velocity" not in mesh.point_data:
        raise GradeError(f"{path} has no point array 'velocity'")
    return mesh


def _summary(case: Path) -> dict[str, Any] | None:
    path = output_dir(case) / "summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sample(mesh, points: np.ndarray) -> np.ndarray:
    """Velocity at the points; a point the mesh does not contain is NaN."""
    import pyvista as pv

    probe = pv.PolyData(np.asarray(points, dtype=float)).sample(mesh)
    values = np.asarray(probe.point_data["velocity"], dtype=float)
    mask = np.asarray(probe.point_data.get("vtkValidPointMask", np.ones(len(points))), dtype=bool)
    values[~mask] = np.nan
    return values


def _check(name: str, value: float, reference: Any, passed: bool, note: str = "") -> dict:
    return {"name": name, "value": value, "reference": reference, "passed": bool(passed),
            "note": note}


def rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(math.sqrt(float(np.mean(d * d))))


# -- the cavity ----------------------------------------------------------------------


def cavity_checks(u: np.ndarray, v: np.ndarray, reynolds: float) -> list[dict]:
    """The three graded numbers, given u at Table I's points and v at Table II's
    (both as fractions of the lid speed)."""
    u_ref = np.array([val for _, val in GHIA_RE1000_U])
    v_ref = np.array([val for _, val in GHIA_RE1000_V])
    limit = THRESHOLDS["cavity_rms"]
    u_err = rms(u, u_ref) if np.all(np.isfinite(u)) else math.inf
    v_err = rms(v, v_ref) if np.all(np.isfinite(v)) else math.inf
    return [
        _check("Reynolds number", reynolds, 1000.0,
               abs(reynolds - 1000.0) <= THRESHOLDS["reynolds_rel"] * 1000.0),
        _check("u RMS error (vertical centreline)", u_err, f"<= {limit}", u_err <= limit,
               f"max |du| {np.nanmax(np.abs(u - u_ref)):.4f}" if np.any(np.isfinite(u)) else ""),
        _check("v RMS error (horizontal centreline)", v_err, f"<= {limit}", v_err <= limit,
               f"max |dv| {np.nanmax(np.abs(v - v_ref)):.4f}" if np.any(np.isfinite(v)) else ""),
    ]


def cavity_frame(mesh) -> dict[str, Any]:
    """Where the lid is and which way it moves, read off the field.

    The lid is the face of the bounding box whose nodes move fastest; its axis is
    the wall-normal ("vertical") one, the dominant component of their mean velocity
    is the lid direction, and the remaining axis is the span."""
    points = np.asarray(mesh.points, dtype=float)
    vel = np.asarray(mesh.point_data["velocity"], dtype=float)
    speed = np.linalg.norm(vel, axis=1)
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = hi - lo
    tol = 1e-6 * float(size.max())
    best = None
    for axis in range(3):
        for side, value in (("min", lo[axis]), ("max", hi[axis])):
            on = np.abs(points[:, axis] - value) <= tol
            if not np.any(on):
                continue
            mean = float(np.mean(speed[on]))
            if best is None or mean > best[0]:
                best = (mean, axis, side, on)
    if best is None or best[0] <= 0:
        raise GradeError("no face of the domain moves: no lid to grade against")
    _, normal, side, on = best
    lid_speed = float(np.median(speed[on]))
    mean_vel = vel[on].mean(axis=0)
    candidates = [a for a in range(3) if a != normal]
    along = max(candidates, key=lambda a: abs(mean_vel[a]))
    span = next(a for a in candidates if a != along)
    sign = 1.0 if mean_vel[along] > 0 else -1.0
    return {
        "lid_axis": AXES[normal], "lid_side": side,
        "lid_direction": ("+" if sign > 0 else "-") + AXES[along],
        "span_axis": AXES[span], "lid_speed": lid_speed,
        "side_length": float(size[along]), "height": float(size[normal]),
        "span": float(size[span]),
        "_normal": normal, "_along": along, "_span": span, "_sign": sign,
        "_lo": lo, "_hi": hi,
    }


def cavity_profiles(mesh, frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """u at Ghia's y* on the vertical centreline and v at his x* on the horizontal
    one, in his frame: y* from the stationary wall to the lid, x* along the lid's
    motion, u along the lid's motion and v toward the lid, divided by the lid speed."""
    n, a, s = frame["_normal"], frame["_along"], frame["_span"]
    lo, hi = frame["_lo"], frame["_hi"]
    sign = frame["_sign"]
    up = 1.0 if frame["lid_side"] == "max" else -1.0
    eps = 1e-9

    def at(along_star: np.ndarray, normal_star: np.ndarray) -> np.ndarray:
        along_star = np.clip(along_star, eps, 1 - eps)
        normal_star = np.clip(normal_star, eps, 1 - eps)
        pts = np.zeros((len(along_star), 3))
        pts[:, a] = lo[a] + along_star * (hi[a] - lo[a]) if sign > 0 else hi[a] - along_star * (hi[a] - lo[a])
        pts[:, n] = lo[n] + normal_star * (hi[n] - lo[n]) if up > 0 else hi[n] - normal_star * (hi[n] - lo[n])
        pts[:, s] = 0.5 * (lo[s] + hi[s])
        return pts

    ys = np.array([y for y, _ in GHIA_RE1000_U])
    xs = np.array([x for x, _ in GHIA_RE1000_V])
    vel_u = sample(mesh, at(np.full_like(ys, 0.5), ys))
    vel_v = sample(mesh, at(xs, np.full_like(xs, 0.5)))
    speed = frame["lid_speed"]
    u = sign * vel_u[:, a] / speed
    v = up * vel_v[:, n] / speed
    # Ghia's wall values are the boundary conditions themselves; the clipped sample
    # a nanometre inside the wall reads them to round-off.
    return u, v


def grade_cavity(case: Path | str, file: str | None = None, nu: float | None = None) -> dict:
    case = Path(case)
    path = result_file(case, file)
    mesh = read_solution(path)
    viscosity = read_nu(case, nu)
    frame = cavity_frame(mesh)
    reynolds = frame["lid_speed"] * frame["side_length"] / viscosity
    u, v = cavity_profiles(mesh, frame)
    checks = cavity_checks(u, v, reynolds)
    summary = _summary(case) or {}
    ptc = summary.get("ptc") if isinstance(summary.get("ptc"), dict) else None
    public = {k: (round(val, 9) if isinstance(val, float) else val)
              for k, val in frame.items() if not k.startswith("_")}
    return {
        "case": "cavity", "file": str(path), "passed": all(c["passed"] for c in checks),
        "checks": checks, "frame": public,
        "notes": [
            f"span {frame['span']:.4g} = {frame['span'] / frame['side_length']:.3g} of the side: "
            "a slab one cell thick with slip (or periodic) faces is the 2D flow Ghia solved; "
            "a deep cavity is a 3D flow, and its midplane is not Ghia's profile",
            f"steady state: {'PTC converged' if ptc and ptc.get('converged') else 'not stated by the summary'}",
        ] + ([] if abs(frame["height"] - frame["side_length"]) <= 1e-6 * frame["side_length"]
             else [f"the cavity is not square: {frame['side_length']:g} x {frame['height']:g}"]),
        "profiles": {
            "u": [{"y": y, "ghia": g, "felix": float(val)} for (y, g), val in zip(GHIA_RE1000_U, u)],
            "v": [{"x": x, "ghia": g, "felix": float(val)} for (x, g), val in zip(GHIA_RE1000_V, v)],
        },
    }


# -- the cylinder --------------------------------------------------------------------


def cylinder_checks(cd: float, reynolds: float, um: float, diameter: float | None = None) -> list[dict]:
    ref = ST_2D1
    lo, hi = ref["cd_interval"]
    checks = [
        _check("drag coefficient", cd, ref["cd"],
               abs(cd - ref["cd"]) <= THRESHOLDS["cd_rel"] * ref["cd"],
               f"{(cd - ref['cd']) / ref['cd'] * 100:+.2f}% of {ref['cd']}; "
               f"{'inside' if lo <= cd <= hi else 'outside'} Schafer-Turek's interval {lo}-{hi}"),
        _check("Reynolds number", reynolds, float(ref["re"]),
               abs(reynolds - ref["re"]) <= THRESHOLDS["reynolds_rel"] * ref["re"]),
        _check("inflow peak Um", um, ref["um"],
               abs(um - ref["um"]) <= THRESHOLDS["um_rel"] * ref["um"]),
    ]
    if diameter is not None:
        checks.append(_check("cylinder diameter", diameter, ref["diameter"],
                             abs(diameter - ref["diameter"]) <= THRESHOLDS["diameter_rel"] * ref["diameter"]))
    return checks


def forces_csv(case: Path) -> Path:
    found = _block_value(_io_text(case), "forces", "csv")
    if found:
        path = Path(found)
        return path if path.is_absolute() else case / path
    return output_dir(case) / "forces.csv"


def read_forces(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise GradeError(f"no forces CSV at {path}: the run must integrate the force on the "
                         "cylinder (io.yaml `forces: {groups: [<cylinder group>]}`)")
    header: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if not header:
            header = [c.strip() for c in line.split(",")]
            continue
        rows.append([float(c) for c in line.split(",")])
    if not rows:
        raise GradeError(f"{path} has no rows")
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(header)}


def cylinder_geometry(mesh) -> dict[str, float]:
    """Channel extents, the inflow peak on x = xmin, and the cylinder read off the
    no-slip nodes that are not on the channel walls."""
    points = np.asarray(mesh.points, dtype=float)
    vel = np.asarray(mesh.point_data["velocity"], dtype=float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = hi - lo
    tol = 1e-6 * float(size.max())
    inlet = np.abs(points[:, 0] - lo[0]) <= tol
    if not np.any(inlet):
        raise GradeError("no nodes on x = xmin: the grader expects the inflow at the low-x end")
    um = float(np.max(vel[inlet, 0]))
    speed = np.linalg.norm(vel, axis=1)
    on_wall = (np.abs(points[:, 1] - lo[1]) <= tol) | (np.abs(points[:, 1] - hi[1]) <= tol)
    at_ends = (np.abs(points[:, 0] - lo[0]) <= tol) | (np.abs(points[:, 0] - hi[0]) <= tol)
    still = (speed <= 1e-9 * max(um, 1e-30)) & ~on_wall & ~at_ends
    if not np.any(still):
        raise GradeError("no no-slip nodes inside the channel: no cylinder found in the solution")
    body = points[still]
    extent = body.max(axis=0) - body.min(axis=0)
    return {
        "length": float(size[0]), "height": float(size[1]), "depth": float(size[2]),
        "um": um, "diameter": float(max(extent[0], extent[1])),
        "centre_x": float(0.5 * (body[:, 0].max() + body[:, 0].min())),
        "centre_y": float(0.5 * (body[:, 1].max() + body[:, 1].min())),
    }


def grade_cylinder(case: Path | str, file: str | None = None, nu: float | None = None) -> dict:
    case = Path(case)
    forces = read_forces(forces_csv(case))
    path = result_file(case, file)
    mesh = read_solution(path)
    viscosity = read_nu(case, nu)
    geometry = cylinder_geometry(mesh)
    mean = 2.0 / 3.0 * geometry["um"]
    area = geometry["diameter"] * geometry["depth"]
    q = 0.5 * mean * mean * area
    fx, fy = forces["Fx"], forces["Fy"]
    cd, cl = float(fx[-1] / q), float(fy[-1] / q)
    tail = fx[-max(2, len(fx) // 10):]
    drift = float(np.max(np.abs(tail - fx[-1])) / max(abs(fx[-1]), 1e-300))
    reynolds = mean * geometry["diameter"] / viscosity
    checks = cylinder_checks(cd, reynolds, geometry["um"], geometry["diameter"])
    lo, hi = ST_2D1["cl_interval"]
    return {
        "case": "cylinder", "file": str(path), "passed": all(c["passed"] for c in checks),
        "checks": checks, "geometry": geometry,
        "steady": drift <= THRESHOLDS["steady_rel"],
        "notes": [
            f"Cl {cl:.5f} (Schafer-Turek {lo}-{hi}; reported, not graded: it is a small "
            "difference of large pressure forces and needs a far finer mesh than Cd)",
            f"Fx moved by {drift:.2e} (relative) over the last {len(tail)} of {len(fx)} rows",
            f"Cd = 2 Fx / (Umean^2 D depth) with Umean = 2/3 Um = {mean:.6g}, D = "
            f"{geometry['diameter']:.6g}, depth = {geometry['depth']:.6g} (rho = 1)",
        ],
    }


# -- output ------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(result: dict) -> None:
    print(f"{result['case']}: {result['file']}")
    for check in result["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        line = f"  {mark}  {check['name']}: {_fmt(check['value'])} (reference {_fmt(check['reference'])})"
        if check.get("note"):
            line += f"  -- {check['note']}"
        print(line)
    for note in result.get("notes", []):
        print(f"  note: {note}")
    if "profiles" in result:
        print("  y       u Ghia    u Felix  |  x       v Ghia    v Felix")
        for pu, pv in zip(result["profiles"]["u"], result["profiles"]["v"]):
            print(f"  {pu['y']:.4f}  {pu['ghia']:+.5f}  {pu['felix']:+.5f}  |  "
                  f"{pv['x']:.4f}  {pv['ghia']:+.5f}  {pv['felix']:+.5f}")
    print("PASSED" if result["passed"] else "FAILED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="which", required=True)
    for name in ("cavity", "cylinder"):
        p = sub.add_parser(name)
        p.add_argument("case", type=Path)
        p.add_argument("--file", help="the result .vtu (default: <output dir>/solution.vtu)")
        p.add_argument("--nu", type=float, help="kinematic viscosity, if case.yaml's is an expression")
        p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        grade = grade_cavity if args.which == "cavity" else grade_cylinder
        result = grade(args.case, args.file, args.nu)
    except GradeError as why:
        print(f"grade.py: {why}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, default=float))
    else:
        print_table(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
