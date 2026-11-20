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

import numpy as np
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


def test_r01s_delivered_answer_is_flagged_against_the_friction_line():
    """Half the force over the whole hull: 0.00158, under ITTC-57's 0.002907."""
    delivered = suboff.HALF_MODEL_FORCE_N / (Q * suboff.R01_AREA_M2)
    assert within(delivered, 0.00158, 0.01)
    floor, name = reference_area.friction_floor(suboff.REYNOLDS)
    assert name == "ITTC-57" and within(floor, 0.002907, 0.001)
    assert "check the area" in reference_area.below_floor(delivered, suboff.REYNOLDS)


def test_the_line_is_not_called_a_bound_where_it_is_a_correlation():
    """ITTC-57 is a model-ship correlation line, so a transitional case can be under
    it honestly and the reading says to check the area rather than "impossible"
    (review of #61, item 6). Blasius, which is the laminar plate itself, carries no
    such caveat."""
    ittc = reference_area.below_floor(1e-4, suboff.REYNOLDS, turbulent=True)
    assert "correlation line rather than a bound" in ittc
    assert "impossible" not in ittc.lower()
    assert "correlation line" not in reference_area.below_floor(1e-4, 1e4, turbulent=False)


def test_the_cli_exits_non_zero_below_the_friction_line(half_hull, capsys):
    assert reference_area.main([str(half_hull), "--body", "hull",
                                "--cd", "0.00158", "--re", "1.2e7"]) == 1
    assert "check the area" in capsys.readouterr().out
    assert reference_area.main([str(half_hull), "--body", "hull",
                                "--cd", "0.00316", "--re", "1.2e7"]) == 0


# -- each plane doubles only what survives it ------------------------------------


def test_only_the_components_that_survive_the_mirror_are_doubled(reading):
    """The SUBOFF fixture is cut at y = 0 and `case_gen.py` writes `liftDir (0 1 0)`
    and `pitchAxis (0 0 1)` there, so the whole hull carries no lift and no pitching
    moment however large the half model's Cl is. Doubling every force was the half of
    F-43 that doubling the area does not fix (review of #61, item 1)."""
    factors = {c["name"]: c["factor"] for c in reading["components"]}

    assert factors == {"drag": 2, "lift": 0, "pitch moment": 0}
    lift = next(c for c in reading["components"] if c["name"] == "lift")
    assert "equal and opposite" in lift["why"] and "zero on the whole body" in lift["why"]


def test_a_plane_normal_to_the_drag_direction_does_not_double_drag(half_hull):
    """The y = 0 plane with `dragDir (0 1 0)`: the mirrored half's drag is equal and
    opposite, so the whole body's is zero, not twice the file's (item 2)."""
    found = reference_area.assess(half_hull, ["hull"], drag_dir=(0.0, 1.0, 0.0),
                                  frontal=False)
    drag = next(c for c in found["components"] if c["name"] == "drag")

    assert drag["factor"] == 0
    assert "equal and opposite" in drag["why"]


def test_a_moment_about_an_axis_in_the_plane_cancels_and_about_the_normal_doubles():
    """A moment is a pseudovector: M + M' = 2(n.M)n, the opposite way round from a
    force. So on the y = 0 plane a yaw moment about y doubles and a pitching moment
    about z cancels."""
    planes = [{"patch": "symmetry", "mirrors": True, "normal": [0.0, 1.0, 0.0]}]

    assert reference_area.component_factor(planes, (0, 1, 0), moment=True)[0] == 2
    assert reference_area.component_factor(planes, (0, 0, 1), moment=True)[0] == 0
    assert reference_area.component_factor(planes, (0, 1, 0))[0] == 0
    assert reference_area.component_factor(planes, (0, 0, 1))[0] == 2


def test_the_factors_are_what_mirroring_the_half_body_actually_sums_to():
    """The arithmetic, checked by doing it rather than by asserting the algebra: take a
    half body's faces with a force on each, reflect positions AND forces in the plane,
    and sum both halves. `component_factor`'s answer has to be that ratio."""
    n = np.array([0.0, 1.0, 0.0])
    reflect = np.eye(3) - 2 * np.outer(n, n)
    rng = np.random.default_rng(0)
    arm = np.abs(rng.normal(size=(500, 3))) * np.array([1.0, 1.0, -1.0])
    arm[:, 1] = np.abs(arm[:, 1]) + 0.1          # the meshed half, on y >= 0
    force = rng.normal(size=(500, 3))
    mirrored_arm, mirrored_force = arm @ reflect.T, force @ reflect.T
    whole_force = force.sum(axis=0) + mirrored_force.sum(axis=0)
    whole_moment = (np.cross(arm, force).sum(axis=0)
                    + np.cross(mirrored_arm, mirrored_force).sum(axis=0))
    half_force, half_moment = force.sum(axis=0), np.cross(arm, force).sum(axis=0)
    planes = [{"patch": "symmetry", "mirrors": True, "normal": list(n)}]

    # An absolute tolerance, not `within`: the right answer is 0 for two of the six,
    # and a fraction of 0 admits nothing.
    for axis in range(3):
        direction = np.eye(3)[axis]
        for whole, half, moment in ((whole_force, half_force, False),
                                    (whole_moment, half_moment, True)):
            factor = reference_area.component_factor(planes, direction, moment=moment)[0]
            ratio = (whole @ direction) / (half @ direction)
            assert abs(ratio - factor) < 1e-9, f"{'xyz'[axis]} moment={moment}: {ratio}"


def test_a_direction_oblique_to_the_plane_gets_no_factor_rather_than_a_guess():
    """The mirror sends it to a mixture of the other components, so no single number
    is the answer and None says so."""
    planes = [{"patch": "symmetry", "mirrors": True, "normal": [0.0, 1.0, 0.0]}]
    factor, (why,) = reference_area.component_factor(planes, (1.0, 1.0, 0.0))

    assert factor is None and "oblique" in why


# -- what does not cut the body counts for nothing -------------------------------


def _cubes_on_a_floor(case: Path, kind: str = "symmetry") -> Path:
    """Four closed cubes resting on a plane at z = 0, as one `body` patch.

    The review's case (item 4): a car on a slip floor of type `symmetry` touches it at
    four tyre patches, and those span the wheelbase, so how far the shared points reach
    cannot tell this from a cut. Each cube is watertight, so the body is open nowhere
    and the plane crosses no edge of it. The floor patch is the four contact squares
    themselves, which is what a coincident contact looks like in a mesh."""
    points: list[tuple[float, float, float]] = []
    body, floor = [], []
    for x0, y0 in ((0.0, 0.0), (2.4, 0.0), (0.0, 1.5), (2.4, 1.5)):
        base = len(points)
        for z in (0.0, 0.3):
            for x, y in ((x0, y0), (x0 + 0.3, y0), (x0 + 0.3, y0 + 0.3), (x0, y0 + 0.3)):
                points.append((x, y, z))
        b = [base + i for i in range(4)]
        t = [base + 4 + i for i in range(4)]
        body.append([b[0], b[3], b[2], b[1]])                     # the contact square
        body.append(t)
        for i in range(4):
            j = (i + 1) % 4
            body.append([b[i], b[j], t[j], t[i]])
        floor.append([b[0], b[1], b[2], b[3]])
    return suboff.write_mesh(case, points, [("body", "wall", body), ("floor", kind, floor)])


def test_a_closed_body_resting_on_a_symmetry_plane_is_not_cut_by_it(tmp_path):
    """Four tyre patches across the wheelbase, not a cut (review of #61, item 4)."""
    found = reference_area.assess(_cubes_on_a_floor(tmp_path), ["body"], frontal=False)
    (plane,) = found["planes"]

    assert found["mirror_factor"] == 1, plane["why"]
    assert plane["shared_points"] == 16, "the points ARE shared, right across the body"
    assert plane["rim_points"] == 0 and "crosses no open edge" in plane["why"]


def test_a_plane_the_body_is_open_along_is_still_a_cut(tmp_path):
    """The other side of item 4: the discriminator is the open edge, and a body cut by
    a plane is open along it, so a half cube on x = 0 still reads as half a body."""
    points = [(x, y, z) for z in (0.0, 1.0) for x, y in
              ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
    b, t = [0, 1, 2, 3], [4, 5, 6, 7]
    body = [[b[0], b[3], b[2], b[1]], t]
    body += [[b[i], b[(i + 1) % 4], t[(i + 1) % 4], t[i]] for i in (0, 1, 2)]  # no x = 0
    cut = [[b[3], b[0], t[0], t[3]]]
    case = suboff.write_mesh(tmp_path, points, [("body", "wall", body),
                                                ("symmetry", "symmetry", cut)])
    (plane,) = reference_area.assess(case, ["body"], frontal=False)["planes"]

    assert plane["mirrors"] and plane["rim_points"] == 4, plane["why"]




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
    assert notes["friction floor"].startswith("below the friction line, check the area")


def test_results_flags_the_whole_bodys_frontal_area_too(tmp_path, reading):
    """Only wetted area was compared, and the whole body's FRONTAL area is smaller than
    the meshed wetted area, so a half force over it cleared both checks and the friction
    line as well (review of #61, item 5)."""
    whole = reading["frontal_body_m2"]
    case = _forces_case(suboff.write_half_hull(tmp_path), aref=whole,
                        cd=suboff.HALF_MODEL_FORCE_N / (Q * whole))
    notes = results.body_checks(case, _history(case, "coeff"))

    assert "whole body's frontal area" in notes["AREA WRONG"], notes
    assert "friction line" not in notes["friction floor"], "which is why wetted missed it"


def test_results_takes_not_mirror_as_the_script_does(tmp_path, reading):
    """`body_checks` called `cuts()` without it, so a double-body waterline was reported
    as half a hull on the one path a study reports through (review of #61, item 3)."""
    case = _forces_case(suboff.write_half_hull(tmp_path), aref=reading["wetted_meshed_m2"],
                        cd=0.003)
    history = _history(case, "coeff")

    assert "mirrored" in results.body_checks(case, history)
    waterline = results.body_checks(case, history, not_mirror=["symmetry"])
    assert "mirrored" not in waterline and "AREA WRONG" not in waterline
    assert not [key for key in waterline if key.startswith(("cut ", "mirror "))]


def test_results_says_the_lift_and_moment_are_not_doubled(tmp_path, reading):
    """What the half model reports as a large Cl, the whole hull does not carry at all
    (review of #61, item 1), said on the series the study reads."""
    case = _forces_case(suboff.write_half_hull(tmp_path), aref=reading["wetted_meshed_m2"],
                        cd=0.003)
    notes = results.body_checks(case, _history(case, "coeff"))

    assert "x 2" in notes["mirror drag"]
    assert "zero on the whole body" in notes["mirror lift"]
    assert "zero on the whole body" in notes["mirror pitch moment"]


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
