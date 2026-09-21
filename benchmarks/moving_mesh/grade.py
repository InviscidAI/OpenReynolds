#!/usr/bin/env python3
"""Grade the two moving-mesh benchmarks from a finished case's raw OpenFOAM output.

    python benchmarks/moving_mesh/grade.py couette <case> [--control <case>] [--json]
    python benchmarks/moving_mesh/grade.py viv <fixed_case> <released_case>... [--json]

Everything is read from what the solver wrote: `postProcessing/*/*.dat`, the solver
log, `constant/` and `system/`. Nothing is read from the agent's own analysis, its
REPORT.md or its JSON summaries, because the first time these cases were graded four of
the run's own numbers disagreed with its raw files, and one of its scripts had quietly
read a sample from t = 13 and called it t = 20 (it never opened the restart directory).
A grader that trusts the run's arithmetic grades the arithmetic.

A constant the case files do not state is never assumed. The grader stops, says which
file it looked in, and names the flag that supplies it.

Exit status: 0 every graded check passed, 1 at least one failed, 2 the case could not
be read.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# Every threshold, and where it came from. A threshold marked "grading choice" is a line
# drawn by whoever set the benchmark, not a published tolerance; it is written down so it
# can be argued with.
THRESHOLDS = {
    # Critical Taylor number for the onset of Taylor vortices in the narrow-gap limit
    # (Taylor 1923). Above it the closed-form Couette profile is no longer the answer.
    "taylor_critical": 1708.0,
    # Grading choice. The inner-wall torque against the exact circular-Couette value.
    # The first graded run landed at +0.43% on a single 50 x 240 mesh.
    "couette_torque_error_pct": 1.0,
    # Grading choice, anchored on a measurement. In steady circular Couette the two wall
    # torques are equal and opposite exactly (conservation of angular momentum), so any
    # imbalance is numerical. A conformal ring at the same 50 x 240 resolution balanced
    # to 0.011%; this allows ten times that. The first sliding-interface run measured
    # 0.91% and FAILS this on purpose: that failure is the finding the case exists for.
    "couette_torque_imbalance_pct": 0.1,
    # From the prompt itself: "a sliding interface that has lost a percent of its area
    # quietly corrupts every torque downstream of it". The deviation of the solver's
    # printed AMI sum(weights) from 1, as a fraction.
    "ami_weight_deviation": 0.01,
    # Williamson (1996), the value the prompt names for a fixed cylinder at Re = 100.
    "strouhal_reference": 0.164,
    # Grading choice. The prompt says the mesh must "land near" 0.164 before a moving
    # body is attempted. 2.5% blockage alone is expected to raise St by about a percent.
    "strouhal_error_pct": 2.0,
    # Grading choice. With zero structural damping a limit cycle needs zero net fluid
    # work per cycle. A loosely coupled run measured about -10% of the body's energy
    # per cycle while that energy held constant, which is impossible; the tightly
    # coupled continuation measured about +0.1%. 1% separates the two by an order of
    # magnitude either side.
    "energy_work_per_cycle_pct": 1.0,
}


class CaseError(Exception):
    """The case cannot be graded as it stands; the message says what to do."""


# --------------------------------------------------------------------------- dictionaries

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _read(path: Path) -> str | None:
    try:
        return _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def _scalar(text: str | None, key: str) -> float | None:
    """`key value;` or `key [0 2 -1 0 0 0 0] value;`, first occurrence."""
    if text is None:
        return None
    m = re.search(
        rf"(?m)(?:^|[\s{{;]){re.escape(key)}\s+(?:\[[^\]]*\]\s*)?({_NUMBER})\s*;", text
    )
    return float(m.group(1)) if m else None


def _vector(text: str | None, key: str) -> tuple[float, float, float] | None:
    if text is None:
        return None
    m = re.search(
        rf"(?m)(?:^|[\s{{;]){re.escape(key)}\s*\(\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*\)",
        text,
    )
    return tuple(float(g) for g in m.groups()) if m else None


def _block(text: str | None, name: str) -> str | None:
    """The body of `name { ... }`, braces matched."""
    if text is None:
        return None
    m = re.search(rf"(?m)(?:^|\s){re.escape(name)}\s*\{{", text)
    if not m:
        return None
    depth, start = 1, m.end()
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def _need(value, what: str, where: str, flag: str):
    if value is None:
        raise CaseError(f"could not read {what} from {where}; pass {flag}")
    return value


# --------------------------------------------------------------------------- time series


def _numeric_dirs(root: Path) -> list[tuple[float, Path]]:
    out = []
    for p in root.iterdir() if root.is_dir() else []:
        try:
            out.append((float(p.name), p))
        except ValueError:
            continue
    return sorted(out)


def _parse_dat(path: Path) -> tuple[list[str], list[list[float]]]:
    """Columns named from the `# Time ...` header, rows with vectors flattened.

    A vector column (`centreOfMass`, written `(x y z)`) becomes `centreOfMass_x/_y/_z`;
    `moment.dat` already names its components, so it passes through unchanged."""
    names: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#"):
            body = line[1:].strip()
            if body.split()[:1] == ["Time"]:
                names = body.split()[1:]
            continue
        values = line.replace("(", " ").replace(")", " ").split()
        if not values:
            continue
        try:
            rows.append([float(v) for v in values])
        except ValueError:
            continue
    if rows and names:
        width = len(rows[0]) - 1
        if width == 3 * len(names):
            names = [f"{n}_{c}" for n in names for c in "xyz"]
    return names, rows


def read_series(case: Path, function_object: str, filename: str) -> dict[str, np.ndarray]:
    """Stitch every restart directory of one function object into a single history.

    A restart from `latestTime` writes into a directory named for its start time, and
    the run it replaces usually went on past that time before it died. Those later
    rows belong to a solution that was thrown away, so each directory is cut at the
    start of the next one rather than merged and de-duplicated."""
    root = case / "postProcessing" / function_object
    dirs = [(t, d) for t, d in _numeric_dirs(root) if (d / filename).is_file()]
    if not dirs:
        raise CaseError(f"no {root / '*' / filename} in {case}")
    names: list[str] = []
    stitched: list[list[float]] = []
    for i, (_, d) in enumerate(dirs):
        cols, rows = _parse_dat(d / filename)
        if not names:
            names = cols
        elif cols and cols != names:
            raise CaseError(f"{d / filename} has different columns from the first restart")
        stop = dirs[i + 1][0] if i + 1 < len(dirs) else math.inf
        stitched.extend(r for r in rows if r[0] <= stop + 1e-12)
    if not names:
        raise CaseError(f"{root} has no '# Time' header naming its columns")
    data = np.array(stitched, dtype=float)
    data = data[np.argsort(data[:, 0], kind="stable")]
    series = {"Time": data[:, 0]}
    for j, name in enumerate(names, start=1):
        if j < data.shape[1]:
            series[name] = data[:, j]
    return series


def _column(series: dict[str, np.ndarray], name: str, where: str) -> np.ndarray:
    if name not in series:
        raise CaseError(f"no column '{name}' in {where}; it has {sorted(series)}")
    return series[name]


def _dat_header_scalar(case: Path, function_object: str, filename: str, key: str) -> float | None:
    """`# magUInf : 1.0e-02` style values the forceCoeffs writer puts in its header."""
    for _, d in _numeric_dirs(case / "postProcessing" / function_object):
        f = d / filename
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("#"):
                break
            m = re.match(rf"#\s*{re.escape(key)}\s*:\s*({_NUMBER})\s*$", line.strip())
            if m:
                return float(m.group(1))
    return None


def upward_crossings(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Times where x crosses zero going up, linearly interpolated between samples."""
    below, above = x[:-1] < 0, x[1:] >= 0
    i = np.nonzero(below & above)[0]
    return t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])


def _last_cycles(crossings: np.ndarray, n: int, what: str) -> tuple[float, float, int]:
    if len(crossings) < 2:
        raise CaseError(f"{what}: fewer than two upward zero crossings, no whole cycle to measure")
    n = min(n, len(crossings) - 1)
    return float(crossings[-1 - n]), float(crossings[-1]), n


# --------------------------------------------------------------------------- couette


def _function_objects(control_dict: str | None, case: Path, kind: str) -> list[tuple[str, str]]:
    """(name, patches) for every function object that wrote `moment.dat`."""
    out = []
    for _, d in [(0, p) for p in sorted((case / "postProcessing").glob("*")) if p.is_dir()]:
        if not any((sub / "moment.dat").is_file() for _, sub in _numeric_dirs(d)):
            continue
        block = _block(control_dict, d.name) or ""
        m = re.search(r"patches\s*\(([^)]*)\)", block) or re.search(r"patch\s+(\S+)\s*;", block)
        out.append((d.name, m.group(1).strip() if m else ""))
    return out


def _pick_walls(fos, inner_fo, outer_fo, case: Path) -> tuple[str, str]:
    if inner_fo and outer_fo:
        return inner_fo, outer_fo
    inner = [n for n, p in fos if "inner" in p.lower() or "inner" in n.lower()]
    outer = [n for n, p in fos if "outer" in p.lower() or "outer" in n.lower()]
    inner_fo = inner_fo or (inner[0] if len(inner) == 1 else None)
    outer_fo = outer_fo or (outer[0] if len(outer) == 1 else None)
    if not inner_fo or not outer_fo:
        raise CaseError(
            f"cannot tell which forces function object is the inner and which the outer wall "
            f"in {case} (found {fos}); pass --inner-fo and --outer-fo"
        )
    return inner_fo, outer_fo


def _mesh_radii(case: Path) -> tuple[float, float, float] | None:
    """Inner radius, outer radius and slab thickness off constant/polyMesh/points."""
    path = case / "constant" / "polyMesh" / "points"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if re.search(r"format\s+binary", text[:2000]):
        return None
    pts = np.array(re.findall(rf"\(({_NUMBER}) ({_NUMBER}) ({_NUMBER})\)", text), dtype=float)
    if len(pts) == 0:
        return None
    r = np.hypot(pts[:, 0], pts[:, 1])
    return float(r.min()), float(r.max()), float(pts[:, 2].max() - pts[:, 2].min())


def _ami_weights(case: Path) -> dict | None:
    """The solver's printed AMI weight sums, from the solver's own log.

    Only the log named for `application` in controlDict is read when there is one: a
    `-postProcess` pass rebuilds the interface too and prints its own weights at its own
    times, which would both mix a second mesh motion into the extremes and make the
    "last time" belong to a utility rather than the solve."""
    pattern = re.compile(rf"sum\(weights\)\s*min:\s*({_NUMBER})\s*max:\s*({_NUMBER})")
    lo, hi, count, last_time = math.inf, -math.inf, 0, None
    app = re.search(r"(?m)^\s*application\s+(\w+)\s*;", _read(case / "system" / "controlDict") or "")
    logs = sorted(case.glob(f"log.{app.group(1)}*")) if app else []
    for log in logs or sorted(case.glob("log.*")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Time = "):
                last_time = line.split("=", 1)[1].strip()
                continue
            m = pattern.search(line)
            if m:
                lo, hi, count = min(lo, float(m.group(1))), max(hi, float(m.group(2))), count + 1
    if count == 0:
        return None
    return {"min": lo, "max": hi, "lines": count, "last_time_in_log": last_time}


def _couette_torques(case, args, need_omega: bool, inherited: dict | None = None):
    control_dict = _read(case / "system" / "controlDict")
    fos = _function_objects(control_dict, case, "moment")
    inner_fo, outer_fo = _pick_walls(fos, args.inner_fo, args.outer_fo, case)

    geometry = _mesh_radii(case)
    ri = args.ri if args.ri is not None else (geometry[0] if geometry else None)
    ro = args.ro if args.ro is not None else (geometry[1] if geometry else None)
    zt = args.thickness if args.thickness is not None else (geometry[2] if geometry else None)
    where_mesh = f"{case / 'constant/polyMesh/points'} (missing or binary)"
    ri = _need(ri, "the inner radius", where_mesh, "--ri")
    ro = _need(ro, "the outer radius", where_mesh, "--ro")
    zt = _need(zt, "the slab thickness", where_mesh, "--thickness")

    nu = args.nu if args.nu is not None else (
        _scalar(_read(case / "constant" / "transportProperties"), "nu")
        or _scalar(_read(case / "constant" / "physicalProperties"), "nu")
    )
    nu = _need(nu, "nu", f"{case / 'constant/transportProperties'}", "--nu")
    rho = args.rho if args.rho is not None else _scalar(_block(control_dict, inner_fo), "rhoInf")
    rho = _need(rho, "rhoInf", f"the '{inner_fo}' block of {case / 'system/controlDict'}", "--rho")

    omega_source = "flag"
    omega = args.omega
    if omega is None:
        dyn = _read(case / "constant" / "dynamicMeshDict")
        omega = _scalar(dyn, "omega")
        omega_source = "constant/dynamicMeshDict"
        if omega is None and _scalar(dyn, "rpm") is not None:
            omega = _scalar(dyn, "rpm") * 2 * math.pi / 60
        if omega is None and inherited is not None:
            omega, omega_source = inherited["omega"], "the graded case (this case has no dynamicMeshDict)"
    if need_omega:
        omega = _need(omega, "omega", f"{case / 'constant/dynamicMeshDict'}", "--omega")

    torques = {}
    for role, fo in (("inner", inner_fo), ("outer", outer_fo)):
        s = read_series(case, fo, "moment.dat")
        column = "viscous_z" if "viscous_z" in s else "total_z"
        t, m = s["Time"], _column(s, column, f"{fo}/moment.dat")
        window = args.window if args.window is not None else 0.25 * t[-1]
        sel = t >= t[-1] - window - 1e-9
        torques[role] = {
            "function_object": fo,
            "column": column,
            "per_length": float(abs(m[sel].mean()) / zt),
            "sd_per_length": float(m[sel].std() / zt),
            "samples": int(sel.sum()),
            "window": [float(t[sel][0]), float(t[-1])],
        }
    tin, tout = torques["inner"]["per_length"], torques["outer"]["per_length"]
    return {
        "ri": ri, "ro": ro, "thickness": zt, "nu": nu, "rho": rho,
        "omega": omega, "omega_source": omega_source,
        "torque": torques,
        "imbalance_pct_of_inner": 100.0 * (tout - tin) / tin,
    }


def grade_couette(args) -> dict:
    case = Path(args.case)
    g = _couette_torques(case, args, need_omega=True)
    ri, ro, nu, rho, omega = g["ri"], g["ro"], g["nu"], g["rho"], g["omega"]
    mu = rho * nu
    exact = 4 * math.pi * mu * omega * ri**2 * ro**2 / (ro**2 - ri**2)
    tin = g["torque"]["inner"]["per_length"]
    gap = ro - ri
    re_i = omega * ri * gap / nu
    taylor = re_i**2 * (gap / ri)
    weights = _ami_weights(case)

    checks = [
        _check("Taylor number (Re_i^2 d/r_i)", taylor, f"< {THRESHOLDS['taylor_critical']:g}",
               taylor < THRESHOLDS["taylor_critical"]),
        _check("inner torque per length [N m/m]", tin, f"exact {exact:.10e}", None),
        _check("inner torque error [%]", 100 * (tin - exact) / exact,
               f"|err| <= {THRESHOLDS['couette_torque_error_pct']:g}",
               abs(100 * (tin - exact) / exact) <= THRESHOLDS["couette_torque_error_pct"]),
        _check("outer torque per length [N m/m]", g["torque"]["outer"]["per_length"], "", None),
        _check("inner/outer imbalance [% of inner]", g["imbalance_pct_of_inner"],
               f"|x| <= {THRESHOLDS['couette_torque_imbalance_pct']:g} (exact 0)",
               abs(g["imbalance_pct_of_inner"]) <= THRESHOLDS["couette_torque_imbalance_pct"]),
    ]
    if weights is None:
        checks.append(_check("AMI sum(weights)", None, "no AMI lines in log.*", None))
    else:
        dev = max(abs(weights["min"] - 1), abs(weights["max"] - 1))
        checks.append(_check(
            f"AMI sum(weights) min..max over {weights['lines']} lines (log to t = {weights['last_time_in_log']})",
            [weights["min"], weights["max"]], f"|w-1| <= {THRESHOLDS['ami_weight_deviation']:g}",
            dev <= THRESHOLDS["ami_weight_deviation"]))

    result = {"case": str(case), "benchmark": "couette", "constants": {
        k: g[k] for k in ("ri", "ro", "thickness", "nu", "rho", "omega")},
        "exact_torque_per_length": exact, "taylor": taylor, "re_inner": re_i,
        "torque": g["torque"], "ami_weights": weights, "checks": checks}

    if args.control:
        c = _couette_torques(Path(args.control), args, need_omega=False, inherited=g)
        ratio = abs(g["imbalance_pct_of_inner"]) / abs(c["imbalance_pct_of_inner"]) if c["imbalance_pct_of_inner"] else math.inf
        cin = c["torque"]["inner"]["per_length"]
        checks.append(_check("control inner/outer imbalance [% of inner]", c["imbalance_pct_of_inner"], "", None))
        if c["omega"] is not None:
            c_exact = 4 * math.pi * c["rho"] * c["nu"] * c["omega"] * c["ri"]**2 * c["ro"]**2 / (c["ro"]**2 - c["ri"]**2)
            checks.append(_check(f"control inner torque error [%] (omega from {c['omega_source']})",
                                 100 * (cin - c_exact) / c_exact, "", None))
        checks.append(_check("imbalance, graded case / control", ratio, "", None))
        result["control"] = {"case": args.control, "torque": c["torque"],
                             "imbalance_pct_of_inner": c["imbalance_pct_of_inner"], "imbalance_ratio": ratio}
    return result


# --------------------------------------------------------------------------- viv


def _force_constants(case: Path, args) -> dict:
    control_dict = _read(case / "system" / "controlDict")
    block = _block(control_dict, "forceCoeffs") or control_dict

    def pick(flag_value, key, flag):
        if flag_value is not None:
            return flag_value
        v = _dat_header_scalar(case, "forceCoeffs", "coefficient.dat", key)
        return v if v is not None else _scalar(block, key)

    where = f"the forceCoeffs header or {case / 'system/controlDict'}"
    return {
        "D": _need(pick(args.diameter, "lRef", "--diameter"), "lRef (the diameter)", where, "--diameter"),
        "U": _need(pick(args.u, "magUInf", "--u"), "magUInf", where, "--u"),
        "Aref": _need(pick(args.aref, "Aref", "--aref"), "Aref", where, "--aref"),
        "rho": _need(args.rho if args.rho is not None else _scalar(block, "rhoInf"),
                     "rhoInf", where, "--rho"),
    }


def _coeffs(case: Path) -> dict[str, np.ndarray]:
    s = read_series(case, "forceCoeffs", "coefficient.dat")
    where = f"{case / 'postProcessing/forceCoeffs/*/coefficient.dat'}"
    return {"Time": s["Time"], "Cl": _column(s, "Cl", where), "Cd": _column(s, "Cd", where)}


def fixed_cylinder(case: Path, args) -> dict:
    k = _force_constants(case, args)
    c = _coeffs(case)
    t0, t1, n = _last_cycles(upward_crossings(c["Time"], c["Cl"]), args.st_cycles, f"{case} Cl")
    f = n / (t1 - t0)
    sel = (c["Time"] >= t0) & (c["Time"] <= t1)
    return {"case": str(case), "cycles": n, "window": [t0, t1], "frequency": f,
            "strouhal": f * k["D"] / k["U"], "mean_cd": float(c["Cd"][sel].mean()),
            "cl_rms": float(np.sqrt(np.mean(c["Cl"][sel] ** 2))), "constants": k}


def energy_audit(t, y, v, cl, t_cl, cycles_t, m, k_spring, q_aref, y_rest=0.0) -> list[dict]:
    """Net fluid work over each whole cycle, as a fraction of that cycle's peak mechanical energy.

    With zero structural damping the spring stores and returns energy and nothing
    dissipates it, so on a genuine limit cycle the fluid can do no net work. A clean,
    stationary amplitude with non-zero work per cycle is being held there by something
    other than the physics, which on a partitioned FSI scheme is the coupling."""
    fy = np.interp(t, t_cl, cl) * q_aref
    power = fy * v
    out = []
    for a, b in zip(cycles_t[:-1], cycles_t[1:]):
        sel = (t >= a) & (t <= b)
        if sel.sum() < 3:
            continue
        work = float(np.trapezoid(power[sel], t[sel]) if hasattr(np, "trapezoid") else np.trapz(power[sel], t[sel]))
        energy = float(np.max(0.5 * m * v[sel] ** 2 + 0.5 * k_spring * (y[sel] - y_rest) ** 2))
        out.append({"from": float(a), "to": float(b), "work": work, "energy": energy,
                    "work_pct": 100.0 * work / energy if energy else math.nan})
    return out


def released_cylinder(case: Path, args, fixed: dict) -> dict:
    k = _force_constants(case, args)
    dyn = _read(case / "constant" / "dynamicMeshDict")
    where = f"{case / 'constant/dynamicMeshDict'}"
    mass = _need(args.mass if args.mass is not None else _scalar(dyn, "mass"), "mass", where, "--mass")
    stiffness = _need(args.stiffness if args.stiffness is not None else _scalar(dyn, "stiffness"),
                      "the spring stiffness", where, "--stiffness")
    damping = _scalar(dyn, "damping")
    anchor = _vector(dyn, "refAttachmentPt") or _vector(dyn, "anchor")

    s = read_series(case, "bodyState", "sixDoFRigidBodyState.dat")
    t = s["Time"]
    y = _column(s, "centreOfMass_y", "bodyState")
    v = _column(s, "velocity_y", "bodyState")
    y_rest = anchor[1] if anchor else float(np.mean(y[t >= t[len(t) // 2]]))
    crossings = upward_crossings(t, y - y_rest)
    t0, t1, n = _last_cycles(crossings, args.cycles, f"{case} displacement")
    sel = (t >= t0) & (t <= t1)
    D = k["D"]
    amplitude = float((y[sel].max() - y[sel].min()) / 2 / D)
    f = n / (t1 - t0)
    fn = math.sqrt(stiffness / mass) / (2 * math.pi)

    c = _coeffs(case)
    csel = (c["Time"] >= t0) & (c["Time"] <= t1)
    q_aref = 0.5 * k["rho"] * k["U"] ** 2 * k["Aref"]
    ecycles = crossings[crossings <= t1][-(args.energy_cycles + 1):]
    audit = energy_audit(t, y, v, c["Cl"], c["Time"], ecycles, mass, stiffness, q_aref, y_rest)
    return {"case": str(case), "cycles": n, "window": [t0, t1], "amplitude_over_D": amplitude,
            "frequency": f, "natural_frequency": fn, "f_over_fn": f / fn,
            "f_over_fixed_shedding": f / fixed["frequency"],
            "mean_y_over_D": float((y[sel] - y_rest).mean() / D),
            "mean_cd": float(c["Cd"][csel].mean()), "cl_rms": float(np.sqrt(np.mean(c["Cl"][csel] ** 2))),
            "damping": damping, "mass": mass, "stiffness": stiffness, "energy_audit": audit}


def grade_viv(args) -> dict:
    fixed = fixed_cylinder(Path(args.fixed), args)
    st_err = 100 * (fixed["strouhal"] - THRESHOLDS["strouhal_reference"]) / THRESHOLDS["strouhal_reference"]
    checks = [
        _check(f"Strouhal number, fixed ({fixed['cycles']} cycles)", fixed["strouhal"],
               f"{THRESHOLDS['strouhal_reference']:g} +/- {THRESHOLDS['strouhal_error_pct']:g}%",
               abs(st_err) <= THRESHOLDS["strouhal_error_pct"]),
        _check("mean Cd, fixed", fixed["mean_cd"], "", None),
        _check("Cl rms, fixed", fixed["cl_rms"], "", None),
    ]
    released = []
    for path in args.released:
        r = released_cylinder(Path(path), args, fixed)
        released.append(r)
        name = Path(path).name
        checks.append(_check(f"[{name}] A/D, half peak-to-peak ({r['cycles']} cycles)", r["amplitude_over_D"],
                             "no validated reference", None))
        checks.append(_check(f"[{name}] frequency [Hz]", r["frequency"], f"f_n {r['natural_frequency']:.6g}", None))
        locked = abs(r["frequency"] - r["natural_frequency"]) < abs(r["frequency"] - fixed["frequency"])
        checks.append(_check(f"[{name}] lock-in (f nearer f_n than fixed shedding)",
                             [r["f_over_fn"], r["f_over_fixed_shedding"]], "f/f_n, f/f_shed", locked))
        checks.append(_check(f"[{name}] mean Cd", r["mean_cd"], "", None))
        works = [c["work_pct"] for c in r["energy_audit"]]
        if r["damping"] not in (None, 0.0):
            checks.append(_check(f"[{name}] energy audit", None,
                                 "structural damping is non-zero; zero net work does not apply", None))
        elif works:
            checks.append(_check(f"[{name}] net fluid work per cycle [% of energy], last {len(works)}",
                                 [min(works), max(works)],
                                 f"|W/E| <= {THRESHOLDS['energy_work_per_cycle_pct']:g}",
                                 max(abs(w) for w in works) <= THRESHOLDS["energy_work_per_cycle_pct"]))
    return {"benchmark": "viv", "fixed": fixed, "released": released, "checks": checks}


# --------------------------------------------------------------------------- output


def _check(name, value, reference, passed):
    return {"name": name, "value": value, "reference": reference,
            "verdict": "-" if passed is None else ("PASS" if passed else "FAIL")}


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return " .. ".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


def print_table(result: dict) -> None:
    rows = [(c["name"], _fmt(c["value"]), c["reference"], c["verdict"]) for c in result["checks"]]
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    for name, value, ref, verdict in rows:
        print(f"{name:<{widths[0]}}  {value:>{widths[1]}}  {ref:<{widths[2]}}  {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="benchmark", required=True)

    cp = sub.add_parser("couette", help="the sliding-interface Couette annulus")
    cp.add_argument("case")
    cp.add_argument("--control", help="the same annulus without an interface")
    for flag in ("--nu", "--rho", "--omega", "--ri", "--ro", "--thickness"):
        cp.add_argument(flag, type=float)
    cp.add_argument("--window", type=float, help="trailing averaging window in seconds (default: last quarter of the run)")
    cp.add_argument("--inner-fo")
    cp.add_argument("--outer-fo")

    vp = sub.add_parser("viv", help="the fixed-then-released vortex-induced vibration case")
    vp.add_argument("fixed")
    vp.add_argument("released", nargs="+")
    for flag in ("--diameter", "--u", "--aref", "--rho", "--mass", "--stiffness"):
        vp.add_argument(flag, type=float)
    vp.add_argument("--st-cycles", type=int, default=7)
    vp.add_argument("--cycles", type=int, default=8)
    vp.add_argument("--energy-cycles", type=int, default=6)

    for p in (cp, vp):
        p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        result = grade_couette(args) if args.benchmark == "couette" else grade_viv(args)
    except CaseError as exc:
        print(f"grade: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)
    return 1 if any(c["verdict"] == "FAIL" for c in result["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
