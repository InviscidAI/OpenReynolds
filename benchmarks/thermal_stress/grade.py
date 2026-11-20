#!/usr/bin/env python3
"""Grade a `solidDisplacementFoam` thermal-stress run against the closed form.

    python benchmarks/thermal_stress/grade.py bar <case> [--json]

`bar`: a bar held between two frictionless rigid walls, resting on a frictionless
floor, free on top, conducting heat from a hot face to a cold one. Its answer is not a
published measurement with an uncertainty on it but the exact solution of the steady
thermoelastic equations, derived in `README.md` beside this file and again in
`openreynolds/toolbox/templates/thermal_stress.py`, which is the template that writes
the case:

    T(y)        = dT * (1 - y/H)                           exactly linear
    sigma_xx(y) = -E * alpha * T(y) * f                     f = 1 (plane stress)
    sigma_yy    = sigma_xy = 0                              f = 1/(1-nu) (plane strain)
    ex_yy(y)    = (1 + nu) * f * alpha * T(y)

There is no reference paper to misquote here and no reference uncertainty to hide
inside: every constant in those four lines is read out of the case's own
`constant/mechanicalProperties`, `constant/thermalProperties` and `0/T`, and the only
thing the solver supplies is the field. That is the whole reason this case was chosen
over preCICE's or anyone else's tutorial comparison. What it cannot tell you is
whether linear small-strain elasticity is the right model for your problem; it tells
you that this solver solves the model it claims to.

Only the solver's own files are read: the two property dictionaries, the `0/T` the
case was started from, and the `raw` profile `postProcess -func singleGraph` wrote
under `postProcessing/`. Never the run's own analysis. The bar's height is measured
off the sampled line rather than taken from the template, so a bar of another size,
or sampled along another axis, is graded the same way.

Exit status: 0 when every check passes, 1 when one fails, 2 when the case cannot be
read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

THRESHOLDS = {
    # The exact solution is uniform in x and linear in y, and the discretisation is
    # exact for both, so what is left is the linear solver's own tolerance and the
    # sampler's interpolation onto the line. The template asks GAMG for 1e-10 and the
    # displacement residual for 1e-9. 0.5% of the peak stress is therefore loose by
    # three orders of magnitude on a converged run and still fails anything that has
    # the constitutive law, the plane-stress switch or the sign of the thermal term
    # wrong -- the plane-stress and plane-strain answers here differ by 43%, and
    # dropping the thermal term altogether gives zero.
    "sigma_xx_rms_rel": 0.005,
    "sigma_xx_max_rel": 0.02,
    # sigma_yy and sigma_xy are identically zero in the exact solution. They are
    # graded against the peak of sigma_xx, because a "small" stress is only small
    # against the one the case actually produces.
    "traction_free_rel": 0.02,
    # T is the solution of a Laplace equation between two fixed values, which second
    # order differencing reproduces exactly; this is the residual and the sampler.
    "temperature_rel": 0.002,
}


class GradeError(Exception):
    """The case cannot be graded; the message says what is missing."""


# -- reading the case ---------------------------------------------------------------


def _text(path: Path) -> str:
    if not path.is_file():
        raise GradeError(f"{path} is not there, so the case cannot be graded")
    return path.read_text(encoding="utf-8", errors="replace")


def _sub_value(text: str, name: str) -> float:
    """`name { type uniform; value <x>; }` -- the shape both property dictionaries use.

    The sub-dictionary form is the only one read: a `type field` entry means the
    property is a whole volScalarField on disk and the closed form above, which has
    one E and one alpha in it, does not describe that case at all.
    """
    match = re.search(
        name + r"\s*\{[^}]*?type\s+(\w+)\s*;[^}]*?value\s+([-+0-9.eE]+)\s*;[^}]*?\}",
        text, re.S)
    if not match:
        raise GradeError(
            f"no `{name} {{ type uniform; value ...; }}` in the dictionary. A "
            f"`type field` property is a field on disk, which this closed form does "
            f"not cover")
    if match.group(1) != "uniform":
        raise GradeError(f"{name} is `type {match.group(1)}`, not uniform")
    return float(match.group(2))


def _switch(text: str, name: str) -> bool:
    match = re.search(rf"\b{name}\s+(yes|no|true|false|on|off|1|0)\s*;", text)
    if not match:
        raise GradeError(f"no `{name}` switch in the dictionary")
    return match.group(1) in ("yes", "true", "on", "1")


def _hot_face_rise(case: Path) -> float:
    """The temperature rise the case was started with: the largest `fixedValue` in
    `0/T`'s boundaryField.

    Read from the field rather than assumed, and read as a **rise** rather than an
    absolute temperature, because `solidDisplacementFoam` subtracts no reference
    temperature anywhere: `calculateStress.H` uses T raw, so the stress-free state of
    the body is T = 0 and the T field is a rise by construction. A case written with
    an ambient 300 in it is not this case and the number this returns would be that
    300, which is the honest reading of what was asked for.
    """
    text = _text(case / "0" / "T")
    body = text[text.index("boundaryField"):] if "boundaryField" in text else text
    values = [float(v) for v in re.findall(r"value\s+uniform\s+([-+0-9.eE]+)\s*;", body)]
    if not values:
        raise GradeError("0/T has no `value uniform ...;` in its boundaryField, so the "
                         "temperature rise the case was given cannot be read")
    return max(values)


def read_case(case: Path) -> dict[str, Any]:
    """E, nu, alpha, the plane-stress switch and the temperature rise, from the case."""
    mechanical = _text(case / "constant" / "mechanicalProperties")
    thermal = _text(case / "constant" / "thermalProperties")
    if not _switch(thermal, "thermalStress"):
        raise GradeError(
            "constant/thermalProperties has `thermalStress no`, so this run solved "
            "pure elasticity and produced no thermal stress to grade. That switch is "
            "the one the plateHole tutorial ships as `no`")
    plane_stress = _switch(mechanical, "planeStress")
    nu = _sub_value(mechanical, "nu")
    return {
        "E_pa": _sub_value(mechanical, "E"),
        "nu": nu,
        "alpha_per_k": _sub_value(thermal, "alpha"),
        "plane_stress": plane_stress,
        # The one place the two states differ, and the reason grading the plane-strain
        # run is also a check on nu: 1/(1-nu) is 1.4286 at nu = 0.3.
        "factor": 1.0 if plane_stress else 1.0 / (1.0 - nu),
        "delta_t_k": _hot_face_rise(case),
    }


# -- reading the sampled profile ----------------------------------------------------

DEFAULT_COLUMNS = ("y", "T", "sigmaxx", "sigmayy", "sigmaxy")


def read_profile(case: Path) -> dict[str, np.ndarray]:
    """`{column: values}` out of the newest `raw` graph `postProcess` wrote.

    The `raw` set writer puts a `#` header with the column names on the first line and
    one row per sample point after it. The header is used where it is there; where it
    is not, the columns are taken to be `DEFAULT_COLUMNS`, which is the order
    `thermal_stress.py` asks for them in, and the count has to agree or nothing is
    read -- guessing at a column is how a grader comes to compare the wrong field.
    """
    found = sorted((case / "postProcessing").rglob("*.xy")) if (
        case / "postProcessing").is_dir() else []
    if not found:
        raise GradeError(
            "no postProcessing/**/*.xy: the profile is written by\n"
            '    postProcess -func "components(sigma)" -noZero\n'
            "    postProcess -func singleGraph -noZero\n"
            "run in the case after the solver, in that order -- `sigma` is a "
            "symmetric tensor and the sampler wants the scalar components")
    # The newest time directory, by the number its name is.
    def when(path: Path) -> float:
        try:
            return float(path.parent.name)
        except ValueError:
            return -1.0
    path = max(found, key=lambda p: (when(p), len(p.name)))

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    header = [line for line in lines if line.startswith("#")]
    rows = [line.split() for line in lines if line and not line.startswith("#")]
    if not rows:
        raise GradeError(f"{path} has a header and no rows")
    width = len(rows[0])

    names: tuple[str, ...] = ()
    if header:
        tokens = header[-1].lstrip("#").split()
        if len(tokens) == width:
            names = tuple(tokens)
    if not names:
        if width != len(DEFAULT_COLUMNS):
            raise GradeError(
                f"{path} has {width} columns and no usable header, and the fields "
                f"{DEFAULT_COLUMNS} would be {len(DEFAULT_COLUMNS)}")
        names = DEFAULT_COLUMNS

    values = np.array([[float(token) for token in row] for row in rows if len(row) == width])
    columns = {name: values[:, index] for index, name in enumerate(names)}
    missing = [n for n in ("T", "sigmaxx") if n not in columns]
    if missing:
        raise GradeError(f"{path} has no {missing} column; it has {sorted(columns)}")
    if "y" not in columns:
        # `axis y` names the coordinate column `y`; another axis names it after itself,
        # and whichever it is, it is the first column the writer puts down.
        columns["y"] = values[:, 0]
    return columns | {"_path": path}


# -- the closed form ----------------------------------------------------------------


def exact(case: dict[str, Any], y: np.ndarray) -> dict[str, np.ndarray]:
    """T, sigma_xx and ex_yy at the sampled heights, from the case's own constants.

    The derivation is in README.md. In one line: the ends hold ex_xx = 0, the top is
    traction-free so sigma_yy = 0, and the mesh is one cell thick so ex_zz = 0; put
    those into the solver's own constitutive law and every elastic constant cancels
    out of sigma_xx but E.
    """
    height = float(np.max(y))
    if height <= 0:
        raise GradeError("the sampled line has no extent, so there is no profile")
    temperature = case["delta_t_k"] * (1.0 - y / height)
    thermal = case["E_pa"] * case["alpha_per_k"] * case["factor"]
    return {
        "height_m": height,
        "T": temperature,
        "sigma_xx": -thermal * temperature,
        "strain_yy": (1.0 + case["nu"]) * case["factor"] * case["alpha_per_k"] * temperature,
        "peak_pa": thermal * case["delta_t_k"],
    }


# -- grading ------------------------------------------------------------------------


def check(name: str, value: float, limit: float, detail: str) -> dict[str, Any]:
    return {"name": name, "value": value, "limit": limit, "detail": detail,
            "pass": bool(value <= limit)}


def grade_bar(case_path: Path) -> dict[str, Any]:
    settings = read_case(case_path)
    profile = read_profile(case_path)
    y = profile["y"]
    answer = exact(settings, y)
    peak = answer["peak_pa"]
    if peak <= 0:
        raise GradeError("E, alpha or the temperature rise is zero, so the case has no "
                         "thermal stress in it and there is nothing to grade")

    temperature_error = float(np.max(np.abs(profile["T"] - answer["T"]))) / settings["delta_t_k"]
    residual = profile["sigmaxx"] - answer["sigma_xx"]
    checks = [
        check("temperature profile, max error / dT",
              temperature_error, THRESHOLDS["temperature_rel"],
              f"linear between {settings['delta_t_k']:g} K and 0 over "
              f"{answer['height_m']:.6g} m"),
        check("sigma_xx, RMS error / peak",
              float(np.sqrt(np.mean(residual**2))) / peak,
              THRESHOLDS["sigma_xx_rms_rel"],
              f"-E*alpha*T*{settings['factor']:.6g}, peak {peak / 1e6:.4f} MPa "
              f"({'plane stress' if settings['plane_stress'] else 'plane strain'})"),
        check("sigma_xx, max error / peak",
              float(np.max(np.abs(residual))) / peak, THRESHOLDS["sigma_xx_max_rel"],
              "pointwise, including the hot face where the stress is largest"),
    ]
    for component in ("sigmayy", "sigmaxy"):
        if component in profile:
            checks.append(check(
                f"{component} / peak sigma_xx",
                float(np.max(np.abs(profile[component]))) / peak,
                THRESHOLDS["traction_free_rel"],
                "zero in the exact solution: the top face carries no traction"))

    measured = {
        "sigma_xx_hot_pa": float(profile["sigmaxx"][int(np.argmin(y))]),
        "sigma_xx_hot_exact_pa": float(-peak),
        "height_m": answer["height_m"],
        "points": int(y.size),
        "profile": str(profile["_path"]),
    }
    return {"case": str(case_path), "settings": settings, "measured": measured,
            "checks": checks}


def report(result: dict[str, Any]) -> None:
    settings, measured = result["settings"], result["measured"]
    print(f"# thermal stress, constrained bar -- {result['case']}")
    print(f"#   {'plane stress' if settings['plane_stress'] else 'plane strain'}, "
          f"E {settings['E_pa']:g} Pa, nu {settings['nu']:g}, "
          f"alpha {settings['alpha_per_k']:g} /K, dT {settings['delta_t_k']:g} K")
    print(f"#   {measured['points']} points over {measured['height_m']:.6g} m, "
          f"from {measured['profile']}")
    print(f"#   sigma_xx on the hot face: {measured['sigma_xx_hot_pa'] / 1e6:+.4f} MPa "
          f"against an exact {measured['sigma_xx_hot_exact_pa'] / 1e6:+.4f} MPa")
    print()
    for item in result["checks"]:
        verdict = "PASS" if item["pass"] else "FAIL"
        print(f"{verdict}  {item['name']}: {item['value']:.3e} "
              f"(limit {item['limit']:.3e}) -- {item['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("which", choices=["bar"])
    parser.add_argument("case", type=Path)
    parser.add_argument("--json", action="store_true", help="the result as JSON")
    args = parser.parse_args(argv)

    try:
        result = grade_bar(args.case)
    except GradeError as error:
        print(f"cannot grade: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        report(result)
    return 0 if all(item["pass"] for item in result["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
