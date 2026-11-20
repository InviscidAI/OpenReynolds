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

The two halves are graded separately and each against what it is responsible for.
`sigma_xx` is graded against `-E*alpha*T` at the **solver's own T at the same sample
point**, which is exact because all three kinematic facts hold cell by cell, so a
conduction that has not converged cannot be mistaken for an elasticity error (or hide
one). The conduction is graded as the linearity of T, which is a statement that needs
neither H nor any knowledge of where the sampler put its 100 points.

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
    # The stress is graded against the temperature the solver itself produced at the
    # same sample point, not against dT*(1 - y/H) at that point. The algebra holding
    # sigma_xx = -E*alpha*T*f is per-cell -- ex_xx = 0, sigma_yy = 0 and ex_zz = 0 hold
    # cell by cell -- so the relation is exact wherever T is read, and grading it this
    # way measures the elasticity without the conduction error or the sampler's
    # interpolation onto the line in the way. `setFormat raw` with `type uniform` and
    # `nPoints 100` over 20 cells returns each cell's value about five times, and a
    # comparison against the analytic line at the sample's own y would be reading that
    # staircase as a 2.5% error in the physics. The conduction is graded separately,
    # as the linearity of T, which is the statement that does not need H.
    #
    # What is left in this number is the linear solver's tolerance: the template asks
    # GAMG for 1e-10 and the displacement residual for 1e-9. 0.5% is loose by orders
    # of magnitude on a converged run and still fails everything that matters -- the
    # plane-stress and plane-strain answers are 43% apart, a missing thermal term
    # gives zero, and a flipped sign gives a tension.
    "sigma_vs_t_rms_rel": 0.005,
    "sigma_vs_t_max_rel": 0.02,
    # sigma_yy and sigma_xy are identically zero in the exact solution. They are
    # graded against the peak of sigma_xx, because a "small" stress is only small
    # against the one the case actually produces.
    "traction_free_rel": 0.02,
    # T solves a Laplace equation between two fixed values, so it is linear in y and
    # second-order differencing reproduces that exactly. Graded as the residual of a
    # straight-line fit, which needs neither H nor where the sampler put its points.
    #
    # This threshold is set by the sampling and not by the solver, which is the honest
    # way round. `thermal_stress.py` asks for `type midPoint`, one sample per cell at
    # the cell centre, where a converged linear field is reproduced to round-off. But
    # `sampleDict.cfg`'s own default is `type uniform; nPoints 100;`, which over 20
    # cells returns each cell's value about five times and makes a staircase whose
    # RMS departure from the line is half a cell step over root three -- 1.44% of dT
    # on this mesh, measured in `test_a_staircase_profile_still_passes`. 2% admits
    # either sampling choice. It is still a real check: an unconverged profile is
    # curved, and a quadratic one departs from its best line by 7.4% of dT.
    "temperature_linearity_rel": 0.02,
    # And it has to actually vary, or a solver that never ran the conduction at all
    # would pass the line fit perfectly with a constant.
    "temperature_span_min_rel": 0.5,
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


def exact(case: dict[str, Any], temperature: np.ndarray) -> dict[str, np.ndarray]:
    """sigma_xx and ex_yy at a given temperature, from the case's own constants.

    The derivation is in README.md. In one line: the ends hold ex_xx = 0, the top is
    traction-free so sigma_yy = 0, and the mesh is one cell thick so ex_zz = 0; put
    those into the solver's own constitutive law and every elastic constant cancels
    out of sigma_xx but E.

    Taking T as the argument rather than y is the point: all three of those facts hold
    cell by cell, so the relation holds wherever T is read, and the grade is then a
    statement about the elasticity alone. It is also what makes the grading immune to
    where the sampler put its points.
    """
    thermal = case["E_pa"] * case["alpha_per_k"] * case["factor"]
    return {
        "T": temperature,
        "sigma_xx": -thermal * temperature,
        "strain_yy": (1.0 + case["nu"]) * case["factor"] * case["alpha_per_k"] * temperature,
        "peak_pa": thermal * case["delta_t_k"],
    }


def line_fit_residual(y: np.ndarray, values: np.ndarray) -> float:
    """RMS departure of `values` from the best straight line through them.

    The conduction check. T between two fixed values with the sides insulated is
    linear in y, exactly, and a straight-line fit says so without needing the bar's
    height, the position of the first cell centre, or anything else about where the
    samples landed.
    """
    if y.size < 3:
        raise GradeError(f"{y.size} sample points is not a profile")
    slope, intercept = np.polyfit(y, values, 1)
    return float(np.sqrt(np.mean((values - (slope * y + intercept)) ** 2)))


# -- grading ------------------------------------------------------------------------


def check(name: str, value: float, limit: float, detail: str) -> dict[str, Any]:
    return {"name": name, "value": value, "limit": limit, "detail": detail,
            "pass": bool(value <= limit)}


def grade_bar(case_path: Path) -> dict[str, Any]:
    settings = read_case(case_path)
    profile = read_profile(case_path)
    y, temperature = profile["y"], profile["T"]
    answer = exact(settings, temperature)
    peak = answer["peak_pa"]
    if peak <= 0:
        raise GradeError("E, alpha or the temperature rise is zero, so the case has no "
                         "thermal stress in it and there is nothing to grade")

    span = float(np.max(temperature) - np.min(temperature))
    residual = profile["sigmaxx"] - answer["sigma_xx"]
    state = "plane stress" if settings["plane_stress"] else "plane strain"
    checks = [
        check("temperature, departure from a straight line / dT",
              line_fit_residual(y, temperature) / settings["delta_t_k"],
              THRESHOLDS["temperature_linearity_rel"],
              "T solves a Laplace equation between two fixed values, so it is linear "
              "in y exactly"),
        check("temperature, dT / span sampled", settings["delta_t_k"] / max(span, 1e-30),
              1.0 / THRESHOLDS["temperature_span_min_rel"],
              f"the profile spans {span:.4g} K of the {settings['delta_t_k']:g} K the "
              f"case was given: a conduction that never ran would fit a line perfectly"),
        check("sigma_xx against -E*alpha*T, RMS / peak",
              float(np.sqrt(np.mean(residual**2))) / peak,
              THRESHOLDS["sigma_vs_t_rms_rel"],
              f"-E*alpha*T*{settings['factor']:.6g} at the solver's own T, peak "
              f"{peak / 1e6:.4f} MPa ({state})"),
        check("sigma_xx against -E*alpha*T, max / peak",
              float(np.max(np.abs(residual))) / peak, THRESHOLDS["sigma_vs_t_max_rel"],
              "pointwise, every sample, including the hot end"),
    ]
    for component in ("sigmayy", "sigmaxy"):
        if component in profile:
            checks.append(check(
                f"{component} / peak sigma_xx",
                float(np.max(np.abs(profile[component]))) / peak,
                THRESHOLDS["traction_free_rel"],
                "zero in the exact solution: the top face carries no traction"))

    hottest = int(np.argmax(temperature))
    measured = {
        "sigma_xx_hottest_pa": float(profile["sigmaxx"][hottest]),
        "t_hottest_k": float(temperature[hottest]),
        # What the hot face itself carries, which is the number README.md quotes. The
        # hottest sample is a cell centre rather than the face, so this is stated as
        # the closed form at the case's own dT and the measurement is beside it.
        "sigma_xx_hot_face_exact_pa": float(-peak),
        "temperature_span_k": span,
        "points": int(y.size),
        "sampled_extent_m": float(np.max(y) - np.min(y)),
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
    print(f"#   {measured['points']} points over {measured['sampled_extent_m']:.6g} m, "
          f"from {measured['profile']}")
    print(f"#   hottest sample {measured['t_hottest_k']:.4f} K carries "
          f"{measured['sigma_xx_hottest_pa'] / 1e6:+.4f} MPa; the hot face's closed "
          f"form is {measured['sigma_xx_hot_face_exact_pa'] / 1e6:+.4f} MPa")
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
