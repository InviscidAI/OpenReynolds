"""The moving-mesh grader reads raw solver output, so its tests write raw solver output.

The fixtures copy the exact layout of what OpenFOAM v2512 writes -- `# Time` headers,
vector columns in parentheses, one directory per restart -- because the grader's first
real job was a case with five restart directories that overlapped, and a column order
nobody should have to assume.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

GRADE = Path(__file__).resolve().parents[1] / "benchmarks" / "moving_mesh" / "grade.py"


def _load():
    spec = importlib.util.spec_from_file_location("moving_mesh_grade", GRADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grade = _load()

MOMENT_HEADER = (
    "# Moment        \n"
    "# CofR          : (0.00000000e+00 0.00000000e+00 0.00000000e+00)\n"
    "#\n"
    "# Time          \ttotal_x total_y total_z\tpressure_x pressure_y pressure_z\tviscous_x viscous_y viscous_z\n"
)


def _moment(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [MOMENT_HEADER]
    for t, mz in rows:
        lines.append(f"{t:<16g} 0 0 {mz:.8e} 0 0 0 0 0 {mz:.8e}\n")
    path.write_text("".join(lines))


def _coefficients(path: Path, t, cl, cd, cl_first=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["Cl", "Cd"] if cl_first else ["Cd", "Cl"]
    head = (
        "# Force and moment coefficients\n"
        "# magUInf       : 1.00000000e-02\n"
        "# lRef          : 1.00000000e-02\n"
        "# Aref          : 1.00000000e-04\n"
        "#\n"
        "# Time          \t" + "\t".join(f"{n:<16}" for n in names + ["CmPitch"]) + "\n"
    )
    body = []
    for ti, l, d in zip(t, cl, cd):
        cols = [l, d] if cl_first else [d, l]
        body.append(f"{ti:<16.10g}\t" + "\t".join(f"{c:.8e}" for c in cols + [0.0]) + "\n")
    path.write_text(head + "".join(body))


def _body_state(path: Path, t, y, v):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = (
        "# Motion State  \n# Angle Units   : degrees\n"
        "# Time          \tcentreOfRotation\tcentreOfMass\trotation\tvelocity\tomega\n"
    )
    body = [
        f"{ti:<16.10g}\t(0 {yi:.10e} 0.005)\t(0 {yi:.10e} 0.005)\t(0 0 0)\t(0 {vi:.10e} 0)\t(0 0 0)\n"
        for ti, yi, vi in zip(t, y, v)
    ]
    path.write_text(head + "".join(body))


# --------------------------------------------------------------------------- stitching


def test_a_restart_replaces_the_rows_the_dead_run_wrote_past_its_start(tmp_path):
    case = tmp_path / "case"
    # The first run reached t = 3 before dying; the restart began from the t = 2 write.
    _moment(case / "postProcessing/forces/0/moment.dat", [(t, -1.0) for t in (1.0, 2.0, 2.5, 3.0)])
    _moment(case / "postProcessing/forces/2/moment.dat", [(t, -5.0) for t in (2.5, 3.0, 3.5)])
    s = grade.read_series(case, "forces", "moment.dat")
    assert list(s["Time"]) == [1.0, 2.0, 2.5, 3.0, 3.5]
    assert list(s["viscous_z"]) == [-1.0, -1.0, -5.0, -5.0, -5.0]


def test_restart_directories_are_ordered_by_time_not_by_name(tmp_path):
    case = tmp_path / "case"
    _moment(case / "postProcessing/forces/13/moment.dat", [(13.5, -2.0)])
    _moment(case / "postProcessing/forces/2/moment.dat", [(2.5, -1.0)])
    assert list(grade.read_series(case, "forces", "moment.dat")["Time"]) == [2.5, 13.5]


def test_a_column_is_found_by_its_header_name(tmp_path):
    case = tmp_path / "case"
    _coefficients(case / "postProcessing/forceCoeffs/0/coefficient.dat",
                  [0.1, 0.2], [0.3, 0.4], [1.3, 1.4], cl_first=True)
    s = grade._coeffs(case)
    assert list(s["Cl"]) == [0.3, 0.4]
    assert list(s["Cd"]) == [1.3, 1.4]


def test_vector_columns_are_split_into_components(tmp_path):
    case = tmp_path / "case"
    _body_state(case / "postProcessing/bodyState/0/sixDoFRigidBodyState.dat", [0.1], [2e-3], [3e-4])
    s = grade.read_series(case, "bodyState", "sixDoFRigidBodyState.dat")
    assert s["centreOfMass_y"][0] == pytest.approx(2e-3)
    assert s["velocity_y"][0] == pytest.approx(3e-4)


# --------------------------------------------------------------------------- strouhal


def test_the_strouhal_number_of_a_known_sine(tmp_path):
    case = tmp_path / "fixed"
    f = 0.17
    t = np.arange(0.005, 80, 0.05)
    _coefficients(case / "postProcessing/forceCoeffs/0/coefficient.dat",
                  t, 0.3 * np.sin(2 * math.pi * f * t + 0.4), 1.35 + 0 * t)
    args = grade.main.__globals__["argparse"].Namespace(
        diameter=None, u=None, aref=None, rho=1000.0, st_cycles=7)
    r = grade.fixed_cylinder(case, args)
    assert r["cycles"] == 7
    assert r["strouhal"] == pytest.approx(f * 0.01 / 0.01, rel=1e-4)
    assert r["mean_cd"] == pytest.approx(1.35)


# --------------------------------------------------------------------------- energy


def _oscillator(force_per_velocity):
    m, k = 7.85398163e-3, 1.24025107e-2
    omega = math.sqrt(k / m)
    t = np.linspace(0, 10 * 2 * math.pi / omega, 20001)
    y = 0.005 * np.sin(omega * t)
    v = 0.005 * omega * np.cos(omega * t)
    q_aref = 0.5 * 1000 * 0.01**2 * 1e-4
    cl = force_per_velocity * v / q_aref
    crossings = grade.upward_crossings(t, y)
    return grade.energy_audit(t, y, v, cl, t, crossings, m, k, q_aref)


def test_a_fluid_that_does_no_work_closes_the_budget():
    audit = _oscillator(0.0)
    assert audit and all(abs(c["work_pct"]) < 1e-9 for c in audit)


def test_the_sign_of_the_work_says_who_is_feeding_whom():
    # A force opposing the velocity takes energy out of the body; one along it puts it in.
    assert all(c["work_pct"] < 0 for c in _oscillator(-1e-4))
    assert all(c["work_pct"] > 0 for c in _oscillator(+1e-4))


# --------------------------------------------------------------------------- couette


def _couette(case: Path, *, with_transport=True, torque_in=None, torque_out=None, weights=1.0000428):
    ri, ro, zt, nu, rho, omega = 0.02, 0.04, 0.001, 1e-4, 900.0, 2.0
    exact = 4 * math.pi * rho * nu * omega * ri**2 * ro**2 / (ro**2 - ri**2)
    tin = exact if torque_in is None else torque_in
    tout = exact if torque_out is None else torque_out
    (case / "system").mkdir(parents=True)
    (case / "constant/polyMesh").mkdir(parents=True)
    (case / "system/controlDict").write_text(
        "application     pimpleFoam;\n\nfunctions\n{\n    forces\n    {\n        type forces;\n        patches ( innerWall );\n"
        "        rhoInf          900;\n    }\n    torqueOuter\n    {\n        type forces;\n"
        "        patches ( outerWall );\n        rhoInf 900;\n    }\n}\n")
    if with_transport:
        (case / "constant/transportProperties").write_text("// oil\nnu              1e-04;\n")
    (case / "constant/dynamicMeshDict").write_text("motionSolver solidBody;\nomega           2;   // rad/s\n")
    pts = [(ri, 0, 0), (ro, 0, 0), (0, ri, zt), (0, -ro, zt)]
    (case / "constant/polyMesh/points").write_text(
        "FoamFile\n{\n    format      ascii;\n}\n\n4\n(\n" + "".join(f"({x} {y} {z})\n" for x, y, z in pts) + ")\n")
    times = np.round(np.arange(0.05, 20.0001, 0.05), 6)
    _moment(case / "postProcessing/forces/0/moment.dat", [(t, -tin * zt) for t in times])
    _moment(case / "postProcessing/torqueOuter/0/moment.dat", [(t, tout * zt) for t in times])
    (case / "log.pimpleFoam").write_text(
        "Time = 0.005\n"
        "AMI: Patch source sum(weights) min:1 max:1 average:1\n"
        f"AMI: Patch target sum(weights) min:1 max:{weights} average:1\n")
    return exact


def _couette_args(case, **kw):
    ns = dict(case=str(case), control=None, nu=None, rho=None, omega=None, ri=None, ro=None,
              thickness=None, window=None, inner_fo=None, outer_fo=None, json=False)
    ns.update(kw)
    return grade.main.__globals__["argparse"].Namespace(**ns)


def test_an_exact_couette_run_passes_every_check(tmp_path):
    exact = _couette(tmp_path / "c")
    r = grade.grade_couette(_couette_args(tmp_path / "c"))
    assert r["exact_torque_per_length"] == pytest.approx(exact)
    assert r["taylor"] == pytest.approx(64.0)
    assert r["ami_weights"]["max"] == pytest.approx(1.0000428)
    assert {c["verdict"] for c in r["checks"]} <= {"PASS", "-"}


def test_an_imbalanced_interface_fails_even_when_its_weights_are_perfect(tmp_path):
    exact = _couette(tmp_path / "c", torque_out=None, torque_in=None)
    case = tmp_path / "c"
    _moment(case / "postProcessing/torqueOuter/0/moment.dat",
            [(t, exact * 0.991 * 0.001) for t in np.arange(0.05, 20.0001, 0.05)])
    r = grade.grade_couette(_couette_args(case))
    verdicts = {c["name"]: c["verdict"] for c in r["checks"]}
    assert verdicts["inner/outer imbalance [% of inner]"] == "FAIL"
    assert r["torque"] and grade.main(["couette", str(case)]) == 1


def test_the_weights_come_from_the_solver_log_not_a_post_processing_pass(tmp_path):
    case = tmp_path / "c"
    _couette(case)
    (case / "log.probe").write_text("Time = 0.05\nAMI: Patch source sum(weights) min:0.9 max:1.2 average:1\n")
    r = grade.grade_couette(_couette_args(case))
    assert r["ami_weights"]["min"] == 1.0
    assert r["ami_weights"]["max"] == pytest.approx(1.0000428)
    assert r["ami_weights"]["last_time_in_log"] == "0.005"


def test_a_missing_constant_names_the_flag_instead_of_guessing(tmp_path, capsys):
    _couette(tmp_path / "c", with_transport=False)
    assert grade.main(["couette", str(tmp_path / "c")]) == 2
    assert "--nu" in capsys.readouterr().err
    assert grade.main(["couette", str(tmp_path / "c"), "--nu", "1e-4"]) == 0


# ------------------------------------------------------------------- the real 2026-09-12 runs

# `tests/data/moving_mesh/` holds what the solver wrote on the two runs
# `benchmarks/moving_mesh/README.md` records under "The known result", cut down to the
# files the grader opens (PROVENANCE.md says exactly what was cut and what was not).
# The synthetic fixtures above can be made to hold a swapped column or an overlapping
# restart on purpose; these cannot, and catch the opposite mistake -- a grader that
# stops reading real output the way it used to. The values are the README's.
REAL = Path(__file__).parent / "data" / "moving_mesh"


def _graded(argv):
    """The grader's own CLI, as the benchmark job runs it: (exit status, the JSON)."""
    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        status = grade.main([*argv, "--json"])
    return status, json.loads(out.getvalue())


def test_the_real_sliding_interface_run_grades_as_it_did_by_hand():
    status, result = _graded(["couette", str(REAL / "couette"),
                              "--control", str(REAL / "couette_conformal")])
    by = {c["name"]: c for c in result["checks"]}
    assert result["taylor"] == pytest.approx(64.0, rel=0.01)
    assert by["inner torque per length [N m/m]"]["value"] == pytest.approx(1.211542e-3, rel=1e-5)
    assert by["inner torque error [%]"]["value"] == pytest.approx(0.4286, abs=1e-3)
    assert by["inner torque error [%]"]["verdict"] == "PASS"
    # The finding the case exists for: the weights held and the torques did not.
    assert by["inner/outer imbalance [% of inner]"]["value"] == pytest.approx(-0.91057, abs=1e-4)
    assert by["inner/outer imbalance [% of inner]"]["verdict"] == "FAIL"
    assert result["ami_weights"] == {"min": 1.0, "max": pytest.approx(1.0000428),
                                     "lines": 8152, "last_time_in_log": "20"}
    assert result["control"]["imbalance_ratio"] == pytest.approx(84.73, abs=0.01)
    assert status == 1


def test_the_real_viv_runs_grade_as_they_did_by_hand():
    status, result = _graded(["viv", str(REAL / "viv_fixed"),
                              str(REAL / "viv_loose"), str(REAL / "viv_tight")])
    assert result["fixed"]["cycles"] == 7
    assert result["fixed"]["strouhal"] == pytest.approx(0.1661407, rel=1e-5)
    assert result["fixed"]["mean_cd"] == pytest.approx(1.355514, rel=1e-5)
    loose, tight = result["released"]
    assert loose["cycles"] == tight["cycles"] == 8
    assert loose["amplitude_over_D"] == pytest.approx(0.6401929, rel=1e-5)
    assert tight["amplitude_over_D"] == pytest.approx(0.5670282, rel=1e-5)
    assert tight["frequency"] == pytest.approx(0.1939371, rel=1e-5)
    # The 13% amplitude drop the energy audit explains: the loose run is held at its
    # amplitude by its coupling, taking 10% of the body's energy out of it per cycle.
    assert [c["work_pct"] for c in loose["energy_audit"]] == pytest.approx(
        [-10.3708, -10.0933, -10.3723, -10.0974, -10.3734, -10.1011], abs=0.002)
    assert max(abs(c["work_pct"]) for c in tight["energy_audit"]) < 0.2
    by = {c["name"]: c["verdict"] for c in result["checks"]}
    assert by["Strouhal number, fixed (7 cycles)"] == "PASS"
    assert by["[viv_loose] net fluid work per cycle [% of energy], last 6"] == "FAIL"
    assert by["[viv_tight] net fluid work per cycle [% of energy], last 6"] == "PASS"
    assert status == 1
