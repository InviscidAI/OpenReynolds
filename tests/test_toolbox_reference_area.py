"""The SUBOFF half model, and the factor of two that shipped once already.

R01 (F-43, 2026-08-29): a half model on a symmetry plane, 43.74 N on the half, and a
delivered Ct of 0.00158 -- the half force over the whole hull's area, below the ITTC
line the agent had computed itself. The fix lived in `snappy_gen.py` and was deleted
with it on 2026-09-07 without anything failing. These tests are what fails now: they
put R01's own mesh shape and R01's own recorded force through the code that reports a
drag, and hold the answer to the towing tank.

What they cannot do is re-run the solve; the suite has no OpenFOAM in it. They pin
everything after the solver -- which is where the error was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import suboff

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"))
import case_gen  # noqa: E402
import reference_area  # noqa: E402
import results  # noqa: E402

Q = 0.5 * suboff.RHO * suboff.SPEED ** 2


@pytest.fixture(scope="module")
def half_hull(tmp_path_factory):
    return suboff.write_half_hull(tmp_path_factory.mktemp("suboff"))


@pytest.fixture(scope="module")
def reading(half_hull):
    return reference_area.assess(half_hull, ["hull"])


def within(value, reference, fraction):
    return abs(value - reference) <= fraction * abs(reference)


# -- the mesh says it is half a hull ------------------------------------------


def test_the_symmetry_plane_is_read_as_cutting_the_hull(reading):
    assert reading["mirror_factor"] == 2
    (plane,) = reading["planes"]
    assert plane["patch"] == "symmetry" and plane["mirrors"]
    assert "y = 0 m" in plane["why"]


def test_the_measured_areas_are_the_hulls(reading):
    """Areas measured off the mesh, doubled, against the report's and against pi r^2."""
    assert within(reading["wetted_body_m2"], suboff.PUBLISHED_WETTED_M2, 0.005)
    assert within(reading["wetted_meshed_m2"] * 2, reading["wetted_body_m2"], 1e-12)
    assert within(reading["frontal_body_m2"], 3.141592653589793 * 0.254 ** 2, 0.005)


def test_the_drag_delivered_is_the_towing_tanks(reading):
    """THE regression: the half model's force, mirrored, against 87.4 N; and the
    coefficient on the as-meshed area against Ct 3.15e-3."""
    drag = reading["mirror_factor"] * suboff.HALF_MODEL_FORCE_N
    ct = suboff.HALF_MODEL_FORCE_N / (Q * reading["wetted_meshed_m2"])

    assert within(drag, suboff.PUBLISHED_DRAG_N, 0.01), f"{drag:.2f} N vs 87.4 N"
    assert within(ct, suboff.PUBLISHED_CT, 0.01), f"Ct {ct:.5f} vs 3.15e-3"
    assert not reference_area.below_floor(ct, suboff.REYNOLDS)


def test_r01s_delivered_answer_is_called_impossible():
    """Half the force over the whole hull: 0.00158, under ITTC-57's 0.002907."""
    delivered = suboff.HALF_MODEL_FORCE_N / (Q * suboff.R01_AREA_M2)
    assert within(delivered, 0.00158, 0.01)
    floor, name = reference_area.friction_floor(suboff.REYNOLDS)
    assert name == "ITTC-57" and within(floor, 0.002907, 0.001)
    assert "below flat-plate friction" in reference_area.below_floor(delivered, suboff.REYNOLDS)


def test_the_cli_exits_non_zero_on_the_impossible_number(half_hull, capsys):
    assert reference_area.main([str(half_hull), "--body", "hull",
                                "--cd", "0.00158", "--re", "1.2e7"]) == 1
    assert "below flat-plate friction" in capsys.readouterr().out
    assert reference_area.main([str(half_hull), "--body", "hull",
                                "--cd", "0.00316", "--re", "1.2e7"]) == 0


# -- what does not cut the body counts for nothing -------------------------------


def test_a_plane_that_is_not_a_symmetry_patch_mirrors_nothing(tmp_path):
    case = suboff.write_half_hull(tmp_path, symmetry_type="patch")
    assert reference_area.assess(case, ["hull"], frontal=False)["mirror_factor"] == 1


def test_not_mirror_is_how_a_double_body_waterline_is_said(half_hull):
    found = reference_area.assess(half_hull, ["hull"], not_mirror=["symmetry"], frontal=False)
    assert found["mirror_factor"] == 1
    assert "--not-mirror" in found["planes"][0]["why"]


def test_laminar_cases_are_held_to_blasius_not_ittc():
    """ITTC-57 is a turbulent line: a correct laminar Cd sits under it."""
    floor, name = reference_area.friction_floor(1e5, turbulent=True)
    assert name == "Blasius laminar" and within(floor, 1.328 / 1e5 ** 0.5, 1e-9)
    assert reference_area.friction_floor(1e7, turbulent=False)[1] == "Blasius laminar"


# -- the path a study actually reports through: results.py ------------------------


def _forces_case(case: Path, *, aref: float, cd: float) -> Path:
    """R01's force output around the half hull: raw forces and forceCoeffs."""
    system, constant = case / "system", case / "constant"
    system.mkdir(exist_ok=True)
    constant.mkdir(exist_ok=True)
    (system / "controlDict").write_text(
        "application simpleFoam;\nfunctions\n{\n"
        "    forces\n    {\n        type forces;\n        libs (forces);\n"
        "        patches (hull);\n        rho rhoInf;\n        rhoInf 998;\n        CofR (0 0 0);\n    }\n"
        "    forceCoeffs\n    {\n        type forceCoeffs;\n        libs (forces);\n"
        f"        patches (hull);\n        rho rhoInf;\n        rhoInf 998;\n"
        f"        dragDir (1 0 0);\n        magUInf {suboff.SPEED};\n"
        f"        lRef {suboff.LENGTH_M:.6g};\n        Aref {aref:.6g};\n    }}\n}}\n",
        encoding="utf-8")
    (constant / "transportProperties").write_text(
        f"transportModel Newtonian;\nnu {suboff.NU:.6g};\n", encoding="utf-8")
    (constant / "turbulenceProperties").write_text("simulationType RAS;\n", encoding="utf-8")
    forces = case / "postProcessing" / "forces" / "0"
    forces.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{t} ({suboff.HALF_MODEL_FORCE_N} 0 0) (20.1 0 0) (23.64 0 0)"
                     for t in range(1, 41))
    (forces / "force.dat").write_text(
        "# Forces\n# CofR : (0 0 0)\n"
        "# Time (total_x total_y total_z) (pressure_x pressure_y pressure_z) "
        "(viscous_x viscous_y viscous_z)\n" + rows + "\n", encoding="utf-8")
    coeffs = case / "postProcessing" / "forceCoeffs" / "0"
    coeffs.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{t} {cd} 0 0" for t in range(1, 41))
    (coeffs / "coefficient.dat").write_text(
        f"# Force coefficients\n# magUInf : {suboff.SPEED}\n# lRef : {suboff.LENGTH_M:.6g}\n"
        f"# Aref : {aref:.6g}\n# Time Cd Cl CmPitch\n" + rows + "\n", encoding="utf-8")
    return case


def _history(case: Path, kind: str):
    grouped = results.find_force_files(case)
    key = next(k for k in grouped if kind in k.lower())
    return results.read_history(grouped[key])


def test_results_reports_the_whole_hulls_drag(tmp_path, reading):
    case = _forces_case(suboff.write_half_hull(tmp_path), aref=reading["wetted_meshed_m2"],
                        cd=suboff.HALF_MODEL_FORCE_N / (Q * reading["wetted_meshed_m2"]))
    notes = results.body_checks(case, _history(case, "forces"))

    assert "1/2 of the body" in notes["mirrored"]
    drag = float(notes["whole-body drag"].split("=")[1].split()[0])
    assert within(drag, suboff.PUBLISHED_DRAG_N, 0.01), notes["whole-body drag"]


def test_results_passes_the_right_coefficient(tmp_path, reading):
    ct = suboff.HALF_MODEL_FORCE_N / (Q * reading["wetted_meshed_m2"])
    case = _forces_case(suboff.write_half_hull(tmp_path), aref=reading["wetted_meshed_m2"], cd=ct)
    notes = results.body_checks(case, _history(case, "coeff"))

    assert "AREA WRONG" not in notes
    assert notes["friction floor"].startswith(f"Cd {ct:.5g} is above ITTC-57")


def test_results_flags_r01_as_delivered(tmp_path):
    """R01's forceCoeffs: the whole hull's area on the half model, Cd 0.00158."""
    whole = suboff.R01_AREA_M2
    case = _forces_case(suboff.write_half_hull(tmp_path), aref=whole,
                        cd=suboff.HALF_MODEL_FORCE_N / (Q * whole))
    notes = results.body_checks(case, _history(case, "coeff"))

    assert "whole body's wetted area" in notes["AREA WRONG"]
    assert notes["friction floor"].startswith("IMPOSSIBLE")


def test_results_says_when_it_could_not_check(tmp_path):
    notes = results.body_checks(tmp_path, {"columns": ["Time", "Cd"], "rows": [], "meta": {}})
    assert notes["body check"].startswith("not made")


# -- the path a case is written through: case_gen.py ------------------------------


def test_case_gen_writes_the_as_meshed_area_on_a_half_model(half_hull):
    ref = case_gen.body_reference(half_hull, "hull", {"area": "wetted"})
    mesh = case_gen.MeshFacts([{"name": "hull", "nFaces": 1}], [0, 0, -1.5, 4.4, 0.3, 1.5],
                              1, 0.0, two_d=False)
    plan = case_gen.Plan(mesh, {}, suboff.LENGTH_M, {"reference": ref})

    area, why = case_gen.reference_aref(plan, {"area": "wetted"})

    assert within(area * 2, suboff.PUBLISHED_WETTED_M2, 0.005)
    assert "1/2 of the body's" in why
