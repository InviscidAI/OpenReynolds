#!/usr/bin/env python3
"""Read a finished moving-body case and report amplitude, frequency and the energy balance.

    python3 viv.py <released case> [--fixed <bolted-still case>] [--json] [--out viv.png]

For an elastically mounted body on `sixDoFRigidBodyMotion`: the transverse amplitude as
A/D, the response frequency, where that frequency sits relative to the structure's own,
and whether the fluid did any net work on the body over each whole cycle. Everything
comes out of what the solver wrote -- `postProcessing/bodyState/*/sixDoFRigidBodyState.dat`,
`postProcessing/forceCoeffs/*/coefficient.dat`, `constant/dynamicMeshDict` -- and nothing
out of a run's own analysis.

**The energy balance is the check worth running.** With zero structural damping nothing
can dissipate energy, so on a real limit cycle the fluid does no net work on the body
per cycle. A loosely coupled partitioned scheme settles into an amplitude that is
stationary, sinusoidal and spectrally clean, and is still being held there by the
coupling rather than by the physics, and this is the only instrument here that sees it.
Measured on the two runs in `tests/data/moving_mesh/`, which are the same case with the
same mesh and differ only in the coupling: the loose one reports net work between
-10.37% and -10.09% of the body's energy per cycle at A/D = 0.640, the one with the body
inside the PIMPLE loop reports +0.04% to +0.18% at A/D = 0.567. The 13% of amplitude
between them is coupling error, and the amplitude alone cannot tell you which run you
have.

Second-order: the natural frequency printed here is also given with the added mass in
it. A circular cylinder accelerating in an unbounded fluid carries an added mass per
unit length of rho pi a^2, exactly its own displaced mass (Lamb, *Hydrodynamics*, 6th
ed., art. 68), so the frequency a body in still water would ring at is
f_n sqrt(m*/(m*+1)) and not f_n. For m* = 10 that is 0.953 f_n, which is 5% -- larger
than most of the agreement anybody quotes on one of these cases, so a measured f/f_n of
0.97 is not 3% from resonance in water, it is above it.

A published amplitude for a given (Re, m*, U*, zeta) is harder to come by than it
looks, and `benchmarks/moving_mesh/README.md` says which values were and were not
readable at source. This script prints A/D and does not grade it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


class CaseError(Exception):
    """The case cannot be read as it stands; the message says what to pass instead."""


# --------------------------------------------------------------------------- dictionaries


def _read(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def scalar(text: str | None, key: str) -> float | None:
    """`key value;` or `key [0 2 -1 0 0 0 0] value;`, first occurrence."""
    if text is None:
        return None
    m = re.search(rf"(?m)(?:^|[\s{{;]){re.escape(key)}\s+(?:\[[^\]]*\]\s*)?({NUMBER})\s*;", text)
    return float(m.group(1)) if m else None


def vector(text: str | None, key: str) -> tuple[float, float, float] | None:
    if text is None:
        return None
    m = re.search(
        rf"(?m)(?:^|[\s{{;]){re.escape(key)}\s*\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)",
        text,
    )
    return tuple(float(g) for g in m.groups()) if m else None


def block(text: str | None, name: str) -> str | None:
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
    """Columns named from the `# Time ...` header, with vector columns flattened.

    `sixDoFRigidBodyState.dat` writes `(x y z)` triples under single names, so
    `centreOfMass` becomes `centreOfMass_x/_y/_z`; `coefficient.dat` already names its
    own columns and passes through. Columns are found by name and never by position:
    the writers have reordered them between releases.
    """
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
    if rows and names and len(rows[0]) - 1 == 3 * len(names):
        names = [f"{n}_{c}" for n in names for c in "xyz"]
    return names, rows


def read_series(case: Path, function_object: str, filename: str) -> dict[str, np.ndarray]:
    """Stitch every restart directory of one function object into a single history.

    A restart from `latestTime` writes into a directory named for its start time, and
    the run it replaces usually carried on past that time before it was abandoned.
    Those later rows belong to a solution nobody kept, so each directory is cut at the
    start of the next rather than merged and de-duplicated. Getting this wrong is not
    hypothetical: an analysis of the 2026-09-12 run read a sample from t = 13 and
    labelled it t = 20 because it never opened the restart's directory
    (`benchmarks/moving_mesh/grade.py`, module docstring).
    """
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


def column(series: dict[str, np.ndarray], name: str, where: str) -> np.ndarray:
    if name not in series:
        raise CaseError(f"no column '{name}' in {where}; it has {sorted(series)}")
    return series[name]


def dat_header_scalar(case: Path, function_object: str, filename: str, key: str) -> float | None:
    """`# magUInf : 1.0e-02`, the way the forceCoeffs writer records what it used."""
    for _, d in _numeric_dirs(case / "postProcessing" / function_object):
        f = d / filename
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("#"):
                break
            m = re.match(rf"#\s*{re.escape(key)}\s*:\s*({NUMBER})\s*$", line.strip())
            if m:
                return float(m.group(1))
    return None


def upward_crossings(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Times where x crosses zero going up, linearly interpolated between samples."""
    below, above = x[:-1] < 0, x[1:] >= 0
    i = np.nonzero(below & above)[0]
    return t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])


def last_cycles(crossings: np.ndarray, n: int, what: str) -> tuple[float, float, int]:
    if len(crossings) < 2:
        raise CaseError(f"{what}: fewer than two upward zero crossings, no whole cycle")
    n = min(n, len(crossings) - 1)
    return float(crossings[-1 - n]), float(crossings[-1]), n


# --------------------------------------------------------------------------- measurement


def force_constants(case: Path, args) -> dict:
    """D, U, Aref and rho, from the forceCoeffs header first and controlDict second.

    The header is preferred because it is what the writer actually used; a controlDict
    edited after the run says what somebody meant.
    """
    control_dict = _read(case / "system" / "controlDict")
    fo = block(control_dict, "forceCoeffs") or control_dict

    def pick(given, key):
        if given is not None:
            return given
        found = dat_header_scalar(case, "forceCoeffs", "coefficient.dat", key)
        return found if found is not None else scalar(fo, key)

    where = f"the forceCoeffs header or {case / 'system/controlDict'}"
    return {
        "D": _need(pick(args.diameter, "lRef"), "lRef (the diameter)", where, "--diameter"),
        "U": _need(pick(args.u, "magUInf"), "magUInf", where, "--u"),
        "Aref": _need(pick(args.aref, "Aref"), "Aref", where, "--aref"),
        "rho": _need(args.rho if args.rho is not None else scalar(fo, "rhoInf"),
                     "rhoInf", where, "--rho"),
    }


def coefficients(case: Path) -> dict[str, np.ndarray]:
    s = read_series(case, "forceCoeffs", "coefficient.dat")
    where = f"{case / 'postProcessing/forceCoeffs/*/coefficient.dat'}"
    return {"Time": s["Time"], "Cl": column(s, "Cl", where), "Cd": column(s, "Cd", where)}


def fixed_body(case: Path, args) -> dict:
    """The Strouhal number of the shedding with the body held still."""
    k = force_constants(case, args)
    c = coefficients(case)
    t0, t1, n = last_cycles(upward_crossings(c["Time"], c["Cl"]), args.st_cycles, f"{case} Cl")
    f = n / (t1 - t0)
    window = (c["Time"] >= t0) & (c["Time"] <= t1)
    return {"case": str(case), "cycles": n, "window": [t0, t1], "frequency": f,
            "strouhal": f * k["D"] / k["U"],
            "mean_cd": float(c["Cd"][window].mean()),
            "cl_rms": float(np.sqrt(np.mean(c["Cl"][window] ** 2))), "constants": k}


def energy_audit(t, y, v, cl, t_cl, cycle_times, mass, stiffness, q_aref, rest=0.0) -> list[dict]:
    """Net fluid work over each whole cycle, as a fraction of that cycle's peak energy.

    With zero structural damping the spring stores and returns energy and nothing
    dissipates it, so on a genuine limit cycle the fluid can do no net work. A clean,
    stationary amplitude with non-zero work per cycle is being held there by something
    that is not the physics, and on a partitioned scheme that is the coupling.
    """
    force = np.interp(t, t_cl, cl) * q_aref
    power = force * v
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out = []
    for a, b in zip(cycle_times[:-1], cycle_times[1:]):
        window = (t >= a) & (t <= b)
        if window.sum() < 3:
            continue
        work = float(trapezoid(power[window], t[window]))
        energy = float(np.max(0.5 * mass * v[window] ** 2
                              + 0.5 * stiffness * (y[window] - rest) ** 2))
        out.append({"from": float(a), "to": float(b), "work": work, "energy": energy,
                    "work_pct": 100.0 * work / energy if energy else math.nan})
    return out


def released_body(case: Path, args, fixed: dict | None = None) -> dict:
    """Amplitude, frequency, lock-in and the energy balance of the released body."""
    k = force_constants(case, args)
    dyn = _read(case / "constant" / "dynamicMeshDict")
    where = f"{case / 'constant/dynamicMeshDict'}"
    mass = _need(args.mass if args.mass is not None else scalar(dyn, "mass"),
                 "mass", where, "--mass")
    stiffness = _need(args.stiffness if args.stiffness is not None else scalar(dyn, "stiffness"),
                      "the spring stiffness", where, "--stiffness")
    damping = scalar(dyn, "damping")
    anchor = vector(dyn, "refAttachmentPt") or vector(dyn, "anchor")

    s = read_series(case, args.body_fo, "sixDoFRigidBodyState.dat")
    t = s["Time"]
    y = column(s, "centreOfMass_y", args.body_fo)
    v = column(s, "velocity_y", args.body_fo)
    x = s.get("centreOfMass_x")
    rest = anchor[1] if anchor else float(np.mean(y[t >= t[len(t) // 2]]))
    crossings = upward_crossings(t, y - rest)
    t0, t1, cycles = last_cycles(crossings, args.cycles, f"{case} displacement")
    window = (t >= t0) & (t <= t1)

    D, U = k["D"], k["U"]
    # The span is Aref/D for a one-cell-thick 2D case, which is how the displaced mass
    # and therefore the mass ratio come out of the case rather than out of a flag.
    span = k["Aref"] / D
    displaced = k["rho"] * math.pi * D * D / 4.0 * span
    f_n = math.sqrt(stiffness / mass) / (2 * math.pi)
    f_added = math.sqrt(stiffness / (mass + displaced)) / (2 * math.pi)
    f = cycles / (t1 - t0)

    c = coefficients(case)
    c_window = (c["Time"] >= t0) & (c["Time"] <= t1)
    q_aref = 0.5 * k["rho"] * U**2 * k["Aref"]
    audit_cycles = crossings[crossings <= t1][-(args.energy_cycles + 1):]
    audit = energy_audit(t, y, v, c["Cl"], c["Time"], audit_cycles, mass, stiffness,
                         q_aref, rest)

    out = {
        "case": str(case), "cycles": cycles, "window": [t0, t1],
        "amplitude_over_D": float((y[window].max() - y[window].min()) / 2 / D),
        "mean_y_over_D": float((y[window] - rest).mean() / D),
        "frequency": f, "natural_frequency": f_n,
        "natural_frequency_added_mass": f_added,
        "f_over_fn": f / f_n, "f_over_fn_added_mass": f / f_added,
        "reduced_velocity": U / (f_n * D),
        "mass": mass, "stiffness": stiffness, "damping": damping,
        "displaced_mass": displaced, "mass_ratio": mass / displaced,
        "span": span, "constants": k,
        "mean_cd": float(c["Cd"][c_window].mean()),
        "cl_rms": float(np.sqrt(np.mean(c["Cl"][c_window] ** 2))),
        "energy_audit": audit,
    }
    if x is not None:
        out["inline_amplitude_over_D"] = float((x[window].max() - x[window].min()) / 2 / D)
    if fixed is not None:
        out["f_over_fixed_shedding"] = f / fixed["frequency"]
        out["locked_in"] = bool(abs(f - f_n) < abs(f - fixed["frequency"]))
    return out


# --------------------------------------------------------------------------- figure


def draw(case: Path, released: dict, out: Path) -> None:
    """Displacement, lift and the per-cycle work, because a limit cycle that is wrong
    looks right on the displacement trace alone."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = read_series(case, "bodyState", "sixDoFRigidBodyState.dat")
    c = coefficients(case)
    D = released["constants"]["D"]
    t, y = s["Time"], s["centreOfMass_y"]
    rest = y[len(y) // 2:].mean()

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=False)
    axes[0].plot(t, (y - rest) / D, lw=0.8, color="#2166ac")
    axes[0].axhspan(-released["amplitude_over_D"], released["amplitude_over_D"],
                    color="#2166ac", alpha=0.10)
    axes[0].set_ylabel("y / D")
    axes[0].set_title(
        f"A/D = {released['amplitude_over_D']:.4f}   f = {released['frequency']:.5f} Hz"
        f"   f/f_n = {released['f_over_fn']:.4f}"
        f"   f/f_n(added mass) = {released['f_over_fn_added_mass']:.4f}", fontsize=9)

    axes[1].plot(c["Time"], c["Cl"], lw=0.6, color="#b2182b", label="Cl")
    axes[1].plot(c["Time"], c["Cd"], lw=0.6, color="#4d4d4d", label="Cd")
    axes[1].set_ylabel("coefficient")
    axes[1].legend(fontsize=8)

    works = [row["work_pct"] for row in released["energy_audit"]]
    centres = [0.5 * (row["from"] + row["to"]) for row in released["energy_audit"]]
    axes[2].axhline(0.0, color="#999999", lw=0.8)
    axes[2].plot(centres, works, "o-", lw=1.0, ms=4, color="#1b7837")
    axes[2].set_ylabel("net fluid work\nper cycle [% of E]")
    axes[2].set_xlabel("t [s]")
    damping = released["damping"]
    axes[2].set_title(
        "zero structural damping, so this is zero on a real limit cycle"
        if damping in (None, 0.0) else f"structural damping {damping:g}: not zero",
        fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- main


def report(args) -> dict:
    fixed = fixed_body(Path(args.fixed), args) if args.fixed else None
    released = released_body(Path(args.case), args, fixed)
    works = [row["work_pct"] for row in released["energy_audit"]]
    return {
        "fixed": fixed, "released": released,
        "work_per_cycle_pct": [min(works), max(works)] if works else None,
    }


def print_table(result: dict) -> None:
    fixed, r = result["fixed"], result["released"]
    rows = []
    if fixed:
        rows += [("Strouhal number, body held still", f"{fixed['strouhal']:.6f}"),
                 ("shedding frequency, still [Hz]", f"{fixed['frequency']:.6f}"),
                 ("mean Cd, still", f"{fixed['mean_cd']:.4f}")]
    rows += [
        ("mass ratio m* (mass / displaced mass)", f"{r['mass_ratio']:.4f}"),
        ("reduced velocity U/(f_n D)", f"{r['reduced_velocity']:.4f}"),
        ("structural damping", "0 (none)" if r["damping"] in (None, 0.0) else f"{r['damping']:g}"),
        (f"A/D, half peak-to-peak over {r['cycles']} cycles", f"{r['amplitude_over_D']:.6f}"),
        ("mean y/D", f"{r['mean_y_over_D']:+.6f}"),
        ("response frequency [Hz]", f"{r['frequency']:.6f}"),
        ("f_n in vacuo [Hz]", f"{r['natural_frequency']:.6f}"),
        ("f_n with the added mass [Hz]", f"{r['natural_frequency_added_mass']:.6f}"),
        ("f / f_n", f"{r['f_over_fn']:.6f}"),
        ("f / f_n with the added mass", f"{r['f_over_fn_added_mass']:.6f}"),
        ("mean Cd", f"{r['mean_cd']:.4f}"),
        ("Cl rms", f"{r['cl_rms']:.4f}"),
    ]
    if "inline_amplitude_over_D" in r:
        rows.append(("inline A/D (x), if the body has that freedom",
                     f"{r['inline_amplitude_over_D']:.6f}"))
    if "f_over_fixed_shedding" in r:
        rows.append(("f / fixed shedding frequency", f"{r['f_over_fixed_shedding']:.6f}"))
        rows.append(("locked in (f nearer f_n than the fixed shedding)",
                     "yes" if r["locked_in"] else "no"))
    if result["work_per_cycle_pct"]:
        low, high = result["work_per_cycle_pct"]
        rows.append((f"net fluid work per cycle [% of E], last {len(r['energy_audit'])}",
                     f"{low:+.4f} .. {high:+.4f}"))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value:>16}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case", help="the case with the body released")
    parser.add_argument("--fixed", help="the same mesh with the body bolted still")
    for flag in ("--diameter", "--u", "--aref", "--rho", "--mass", "--stiffness"):
        parser.add_argument(flag, type=float)
    parser.add_argument("--cycles", type=int, default=8,
                        help="whole displacement cycles to measure over")
    parser.add_argument("--st-cycles", type=int, default=7,
                        help="whole lift cycles for the fixed case's Strouhal number")
    parser.add_argument("--energy-cycles", type=int, default=6)
    parser.add_argument("--body-fo", default="bodyState",
                        help="the sixDoFRigidBodyState function object's name")
    parser.add_argument("--out", help="write a PNG of displacement, forces and work here")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        result = report(args)
        if args.out:
            draw(Path(args.case), result["released"], Path(args.out))
    except CaseError as exc:
        print(f"viv: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
