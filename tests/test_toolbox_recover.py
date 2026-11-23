"""`recover.py`: the numerical moves on a diverged run, read out of the case, mesh last.

It exists because a diverged case was remeshed again and again while the relaxation
factors, the schemes and the time step it started with were never changed. So what is
tested is that each move reads this case's own dictionaries and log -- a factor
already low is not lowered again, a scheme already upwind is not proposed, a steady
solver gets no Courant advice -- and that the mesh comes last. The rest is the
contract `locate.py` keeps: advisory, exit code 0, and nothing beyond its siblings.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def recover():
    sys.path.insert(0, str(TOOLBOX))
    spec = importlib.util.spec_from_file_location("toolbox_recover", TOOLBOX / "recover.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HEADER = "FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n"


def write_case(root: Path, *, application: str, control: str = "", solution: str,
               schemes: str = "", times=(0,), u_internal: str = "uniform (1 0 0)",
               log: str = "", check_mesh: str = "") -> Path:
    case = root / "case"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text(
        HEADER % "controlDict" + f"application {application};\n{control}\n")
    (case / "system" / "fvSolution").write_text(HEADER % "fvSolution" + solution)
    (case / "system" / "fvSchemes").write_text(HEADER % "fvSchemes" + (schemes or (
        "divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U);"
        " div(phi,k) Gauss limitedLinear 1;"
        " div((nuEff*dev2(T(grad(U))))) Gauss linear; }")))
    for t in times:
        (case / f"{t:g}").mkdir()
    (case / "0" / "U").write_text(HEADER % "U" + f"internalField {u_internal};\n")
    if log:
        (case / f"log.{application}").write_text(log)
    if check_mesh:
        (case / "log.checkMesh").write_text(check_mesh)
    return case


def transient_log(times, courant, u_residuals, tail: str = "") -> str:
    lines = []
    for t, co, u in zip(times, courant, u_residuals):
        lines += [f"Courant Number mean: {co / 4:g} max: {co:g}", f"Time = {t:g}", "",
                  f"smoothSolver:  Solving for Ux, Initial residual = {u:g}, "
                  f"Final residual = 1e-08, No Iterations 3",
                  "DICPCG:  Solving for p, Initial residual = 0.001, "
                  "Final residual = 1e-07, No Iterations 20"]
    return "\n".join(lines) + "\n" + tail


PIMPLE_ONE = "PIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; }\n"
FPE_TAIL = "#0  Foam::error::printStack(Foam::Ostream&)\n#1  Foam::sigFpe::sigHandler(int)\n"


def moves(report) -> dict:
    return {m["move"]: m for m in report["moves"]}


def diverged_transient(recover, tmp_path):
    times = [0.05 * i for i in range(1, 9)]
    log = transient_log(times, [0.8, 0.8, 0.9, 0.9, 1.2, 4.0, 15.0, 39.0],
                        [1e-3, 8e-4, 7e-4, 6e-4, 5e-4, 2.0, 40.0, 900.0], FPE_TAIL)
    case = write_case(tmp_path, application="pimpleFoam", control="deltaT 0.002;",
                      solution=PIMPLE_ONE, times=(0, 0.1, 0.2, 0.3), log=log)
    return recover.run(case)


def test_a_transient_blow_up_is_read_and_the_restart_is_the_last_good_write(recover, tmp_path):
    report = diverged_transient(recover, tmp_path)
    assert report["diverged"] is True
    assert report["log"]["fpe"] is True
    assert report["onset"] == pytest.approx(0.3)
    restart = moves(report)["restart"]
    assert restart["status"] == "try"
    assert "startTime 0.2;" in restart["edit"], "0.3 is the onset itself, not before it"


def test_a_fixed_step_is_scaled_from_the_courant_seen_before_the_onset(recover, tmp_path):
    step = moves(diverged_transient(recover, tmp_path))["time step"]
    assert step["status"] == "try"
    # The Courant line above `Time = 0.3` is the onset step's own (4.0), so the last
    # healthy one is 1.2, printed above `Time = 0.25`: 0.002 * 0.5 / 1.2
    assert "deltaT 0.000833;" in step["edit"]
    assert "39" in step["read"], "the peak belongs in what was read"


def test_relaxation_is_not_offered_to_pimple_with_one_outer_corrector(recover, tmp_path):
    relax = moves(diverged_transient(recover, tmp_path))["under-relaxation"]
    assert relax["status"] == "n/a"
    assert "nOuterCorrectors" in relax["why"]


def test_only_the_higher_order_convection_terms_are_proposed_for_upwind(recover, tmp_path):
    conv = moves(diverged_transient(recover, tmp_path))["first-order convection"]
    assert conv["status"] == "try"
    assert "div(phi,U) Gauss upwind;" in conv["edit"]
    assert "div(phi,k) Gauss upwind;" in conv["edit"]
    assert "bounded" not in conv["edit"], "bounded is for a steady solver"
    assert "nuEff" not in conv["edit"], "the viscous term is not convection"


def test_the_mesh_is_last_and_points_at_locate(recover, tmp_path):
    report = diverged_transient(recover, tmp_path)
    names = [m["move"] for m in report["moves"]]
    assert names[-1] == "mesh"
    assert names.index("restart") == 0
    assert "locate.py" in report["moves"][-1]["edit"]
    assert "Ux degrades first" in report["moves"][-1]["read"]


def test_a_steady_case_gets_relaxation_and_no_courant(recover, tmp_path):
    case = write_case(
        tmp_path, application="simpleFoam",
        solution="SIMPLE { nNonOrthogonalCorrectors 0; }\n"
                 "relaxationFactors { fields { p 0.3; } equations { U 0.9; k 0.7; } }\n",
        schemes="divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);"
                " div(phi,k) bounded Gauss upwind; }")
    found = moves(recover.run(case))
    assert found["time step"]["status"] == "n/a"
    relax = found["under-relaxation"]
    assert relax["status"] == "try"
    assert "p 0.2;" in relax["edit"], "at the classic value: two thirds of it"
    assert "U 0.7;" in relax["edit"], "above the classic value: down to it"
    conv = found["first-order convection"]
    assert "div(phi,U) bounded Gauss upwind;" in conv["edit"]
    assert "div(phi,k)" not in conv["edit"], "already upwind"


def test_relaxation_already_far_below_classic_is_not_lowered_again(recover, tmp_path):
    case = write_case(
        tmp_path, application="simpleFoam",
        solution="SIMPLE { }\nrelaxationFactors { fields { p 0.1; } equations { U 0.3; } }\n")
    relax = moves(recover.run(case))["under-relaxation"]
    assert relax["status"] == "done"
    assert "not the move" in relax["why"]


def test_simplec_pressure_is_left_unrelaxed(recover, tmp_path):
    case = write_case(
        tmp_path, application="simpleFoam",
        solution="SIMPLE { consistent yes; }\nrelaxationFactors { equations { U 0.9; } }\n")
    relax = moves(recover.run(case))["under-relaxation"]
    assert "SIMPLEC" in relax["read"]
    assert "fields" not in relax["edit"]
    assert "U 0.7;" in relax["edit"]


def test_a_buoyant_case_relaxes_p_rgh_from_its_field_not_its_factors(recover, tmp_path):
    """No p_rgh factor written: the solver relaxes p_rgh (pEqn.H calls p_rgh.relax()),
    so the edit names p_rgh, read off 0/p_rgh. rho is not an equation to relax."""
    case = write_case(
        tmp_path, application="buoyantSimpleFoam",
        solution="SIMPLE { }\nrelaxationFactors { fields { rho 1; } equations { U 0.7; } }\n")
    (case / "0" / "p_rgh").write_text(HEADER % "p_rgh")
    (case / "0" / "T").write_text(HEADER % "T")
    (case / "constant").mkdir()
    (case / "constant" / "thermophysicalProperties").write_text(
        HEADER % "thermophysicalProperties"
        + "thermoType { type heRhoThermo; energy sensibleEnthalpy; }\n")
    found = moves(recover.run(case))
    relax = found["under-relaxation"]
    assert "fields { p_rgh 0.3; }" in relax["edit"]
    assert " p 0.3" not in relax["edit"]
    assert "h 0.7;" in relax["edit"], "the energy equation, unrelaxed, comes down to 0.7"
    assert "rho" not in relax["edit"]
    assert found["potential-flow start"]["status"] == "n/a"


def test_a_boussinesq_case_relaxes_t_not_h(recover, tmp_path):
    """buoyantBoussinesqSimpleFoam's TEqn.H relaxes TEqn: an `h` factor is dead."""
    case = write_case(tmp_path, application="buoyantBoussinesqSimpleFoam",
                      solution="SIMPLE { }\nrelaxationFactors { equations { U 0.7; } }\n")
    (case / "0" / "p_rgh").write_text(HEADER % "p_rgh")
    (case / "0" / "T").write_text(HEADER % "T")
    relax = moves(recover.run(case))["under-relaxation"]
    assert "T 0.7;" in relax["edit"]
    assert " h " not in relax["edit"]


def test_old_layout_regex_never_reaches_the_pressure(recover, tmp_path):
    """solution.C takes the old layout's field factors from literal keys starting
    with p or rho: `".*" 0.2` relaxes equations, and p stays unrelaxed."""
    case = write_case(tmp_path, application="simpleFoam",
                      solution='SIMPLE { }\nrelaxationFactors { ".*" 0.2; U 0.9; }\n')
    relax = moves(recover.run(case))["under-relaxation"]
    assert "p unrelaxed" in relax["read"]
    assert "p 0.3;" in relax["edit"]


def test_v_suffixed_and_limited_schemes_count_as_higher_order(recover, tmp_path):
    case = write_case(
        tmp_path, application="simpleFoam", solution="SIMPLE { }\n",
        schemes="divSchemes { default none; div(phi,U) bounded Gauss linearUpwindV grad(U);"
                " div(phi,k) bounded Gauss limitedVanLeer 0 1; div(phi,omega) bounded Gauss upwind; }")
    conv = moves(recover.run(case))["first-order convection"]
    assert conv["status"] == "try"
    assert "div(phi,U) bounded Gauss upwind;" in conv["edit"]
    assert "div(phi,k) bounded Gauss upwind;" in conv["edit"]
    assert "omega" not in conv["edit"]


def test_a_scheme_macro_is_expanded_before_it_is_judged(recover, tmp_path):
    case = write_case(
        tmp_path, application="rhoPimpleFoam", solution=PIMPLE_ONE,
        schemes="divSchemes { default none; turbulence Gauss limitedLinear 1;"
                " div(phi,U) Gauss upwind; div(phi,k) $turbulence; div(phi,epsilon) $turbulence; }")
    conv = moves(recover.run(case))["first-order convection"]
    assert "div(phi,k) Gauss upwind;" in conv["edit"]
    assert "div(phi,epsilon) Gauss upwind;" in conv["edit"]
    assert "div(phi,U)" not in conv["edit"]


def test_regex_convection_keys_with_wildcard_fluxes_are_read(recover, tmp_path):
    case = write_case(
        tmp_path, application="reactingTwoPhaseEulerFoam", solution=PIMPLE_ONE,
        schemes='divSchemes { "div\\(alphaRhoPhi.*,U.*\\)" Gauss limitedLinearV 1;'
                ' "div\\(alphaRhoPhi.*,(h|e).*\\)" Gauss limitedLinear 1;'
                ' "div\\(phi.*,alpha.*\\)" Gauss vanLeer; }')
    conv = moves(recover.run(case))["first-order convection"]
    assert conv["status"] == "try"
    assert '"div\\(alphaRhoPhi.*,U.*\\)" Gauss upwind;' in conv["edit"]
    assert '(h|e)' in conv["edit"]
    assert "alpha.*" not in conv["edit"].replace("alphaRhoPhi.*", "")


def test_a_second_order_default_gets_explicit_upwind_entries(recover, tmp_path):
    """cylinder2D's shape: `default Gauss linear;` and no div(phi,...) line."""
    case = write_case(tmp_path, application="pimpleFoam", solution=PIMPLE_ONE,
                      schemes="divSchemes { default Gauss linear; }")
    (case / "0" / "k").write_text(HEADER % "k")
    conv = moves(recover.run(case))["first-order convection"]
    assert conv["status"] == "try"
    assert "div(phi,U) Gauss upwind; div(phi,k) Gauss upwind;" in conv["edit"]
    assert "default" not in conv["edit"]


def test_alphat_is_not_a_second_phase(recover, tmp_path):
    """Compressible and buoyant cases carry 0/alphat; the explicit-entry move must
    still fire for them."""
    case = write_case(tmp_path, application="rhoSimpleFoam", solution="SIMPLE { }\n",
                      schemes="divSchemes { default bounded Gauss linearUpwind limited; }")
    (case / "0" / "alphat").write_text(HEADER % "alphat")
    (case / "0" / "T").write_text(HEADER % "T")
    (case / "constant").mkdir()
    (case / "constant" / "thermophysicalProperties").write_text(
        HEADER % "thermophysicalProperties" + "thermoType { energy sensibleInternalEnergy; }\n")
    conv = moves(recover.run(case))["first-order convection"]
    assert conv["status"] == "try"
    assert "div(phi,e) bounded Gauss upwind;" in conv["edit"]


def test_the_trapfpe_banner_does_not_count_as_a_crash(recover, tmp_path):
    log = ("trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
           + transient_log([0.1, 0.2, 0.3], [0.4, 0.4, 0.4], [1e-3, 5e-4, 2e-4]) + "End\n")
    case = write_case(tmp_path, application="pimpleFoam", control="deltaT 0.001;",
                      solution=PIMPLE_ONE, log=log)
    report = recover.run(case)
    assert report["log"]["fpe"] is False
    assert report["diverged"] is False


def test_a_simple_block_makes_the_case_steady_whatever_its_name(recover, tmp_path):
    case = write_case(tmp_path, application="simpleReactingParcelFoam",
                      solution="SIMPLE { }\nrelaxationFactors { equations { U 0.9; } }\n")
    found = moves(recover.run(case))
    assert found["time step"]["status"] == "n/a"
    assert found["under-relaxation"]["status"] == "try"


def test_srf_cases_get_no_potential_start(recover, tmp_path):
    case = write_case(tmp_path, application="SRFSimpleFoam", solution="SIMPLE { }\n")
    assert moves(recover.run(case))["potential-flow start"]["status"] == "n/a"


def test_a_phase_momentum_term_is_convection(recover, tmp_path):
    case = write_case(
        tmp_path, application="multiphaseEulerFoam", solution=PIMPLE_ONE,
        schemes="divSchemes { div(alphaRhoPhi.air,U.air) Gauss linearUpwind limited;"
                " div(phi.air,alpha.air) Gauss vanLeer; }")
    conv = moves(recover.run(case))["first-order convection"]
    assert "div(alphaRhoPhi.air,U.air) Gauss upwind;" in conv["edit"]
    assert "alpha.air)" not in conv["edit"]


def test_the_restart_names_the_directory_as_written(recover, tmp_path):
    log = transient_log([0.1, 0.2, 0.3], [0.3, 0.3, 0.3], [1e-3, 1e-3, 5.0])
    case = write_case(tmp_path, application="pimpleFoam", control="deltaT 0.001;",
                      solution=PIMPLE_ONE, log=log, times=(0,))
    (case / "0.1234567").mkdir()
    restart = moves(recover.run(case))["restart"]
    assert "startTime 0.1234567;" in restart["edit"]


TUTORIALS = Path("/usr/lib/openfoam/openfoam2512/tutorials")


@pytest.mark.skipif(not TUTORIALS.is_dir(), reason="needs the OpenFOAM v2512 tutorials")
def test_real_tutorial_dictionaries(recover):
    """The dictionaries the synthetic cases stand in for, read as they ship."""
    bike = moves(recover.run(TUTORIALS / "incompressible/simpleFoam/motorBike"))
    assert "linearUpwindV" in bike["first-order convection"]["read"]
    assert bike["first-order convection"]["status"] == "try"
    cavity = moves(recover.run(TUTORIALS / "compressible/rhoPimpleFoam/RAS/cavity"))
    assert "div(phi,k)" in cavity["first-order convection"]["edit"]
    room = moves(recover.run(TUTORIALS / "heatTransfer/buoyantBoussinesqSimpleFoam/hotRoom"))
    assert "T " in room["under-relaxation"]["read"]
    assert " h " not in room["under-relaxation"]["edit"]
    for case in sorted(TUTORIALS.rglob("system/controlDict"))[:200]:
        report = recover.run(case.parent.parent)
        for move in report["moves"]:
            assert "default" not in move["edit"].split("divSchemes:")[-1] or \
                move["move"] != "first-order convection"


def test_a_regex_relaxation_key_is_resolved_and_never_raised(recover, tmp_path):
    """`".*" 0.3` relaxes U at 0.3. Proposing `U 0.7` would raise it, because the exact
    key outranks the regex."""
    case = write_case(tmp_path, application="simpleFoam",
                      solution='SIMPLE { }\nrelaxationFactors { equations { ".*" 0.3; } }\n')
    relax = moves(recover.run(case))["under-relaxation"]
    assert 'U 0.3 (from ".*")' in relax["read"]
    assert "U " not in relax["edit"], "0.3 is already far below 0.7, so it is left alone"
    assert "U already far below" in relax["why"]


def test_a_default_relaxation_key_is_resolved(recover, tmp_path):
    case = write_case(tmp_path, application="simpleFoam",
                      solution="SIMPLE { }\nrelaxationFactors { equations { default 0.3; } }\n")
    relax = moves(recover.run(case))["under-relaxation"]
    assert "U 0.3 (from default)" in relax["read"]
    assert "U 0.7" not in relax["edit"]


def test_top_level_factors_are_ignored_when_a_subdict_exists(recover, tmp_path):
    """OpenFOAM reads `fields`/`equations` when either is present and drops the
    block's own entries: U is 0.9 and p is unrelaxed here."""
    case = write_case(tmp_path, application="simpleFoam",
                      solution="SIMPLE { }\nrelaxationFactors { p 0.3; U 0.7; equations { U 0.9; } }\n")
    relax = moves(recover.run(case))["under-relaxation"]
    assert "p unrelaxed" in relax["read"]
    assert "U 0.9" in relax["read"]


def test_old_top_level_factors_are_edited_where_they_are(recover, tmp_path):
    case = write_case(tmp_path, application="simpleFoam",
                      solution="SIMPLE { }\nrelaxationFactors { p 0.3; U 0.7; }\n")
    relax = moves(recover.run(case))["under-relaxation"]
    assert "top level" in relax["edit"]
    assert "fields {" not in relax["edit"], "a subdict would switch the top-level ones off"


def test_no_relaxation_block_gets_the_subdict_spelling(recover, tmp_path):
    case = write_case(tmp_path, application="simpleFoam", solution="SIMPLE { }\n")
    relax = moves(recover.run(case))["under-relaxation"]
    assert "p unrelaxed" in relax["read"]
    assert "fields { p 0.3; } equations { U 0.7; }" in relax["edit"]


def test_the_default_div_scheme_is_never_rewritten(recover, tmp_path):
    """`default` also covers the viscous term, and `Gauss upwind` there stops the run."""
    case = write_case(tmp_path, application="simpleFoam", solution="SIMPLE { }\n",
                      schemes="divSchemes { default Gauss linear; }")
    conv = moves(recover.run(case))["first-order convection"]
    assert "default" not in conv["edit"]
    assert "div(phi,U) bounded Gauss upwind;" in conv["edit"]
    assert "default Gauss linear" in conv["read"]


def test_a_regex_convection_key_is_read(recover, tmp_path):
    case = write_case(
        tmp_path, application="simpleFoam", solution="SIMPLE { }\n",
        schemes='divSchemes { default none; div(phi,U) bounded Gauss upwind;'
                ' "div\\(phi,(k|epsilon)\\)" bounded Gauss limitedLinear 1; }')
    conv = moves(recover.run(case))["first-order convection"]
    assert conv["status"] == "try"
    assert '"div\\(phi,(k|epsilon)\\)" bounded Gauss upwind;' in conv["edit"]


def test_multiphase_convection_skips_interface_compression_and_alpha(recover, tmp_path):
    case = write_case(
        tmp_path, application="interFoam", solution=PIMPLE_ONE,
        schemes="divSchemes { div(rhoPhi,U) Gauss linearUpwind grad(U);"
                " div(phi,alpha) Gauss vanLeer; div(phirb,alpha) Gauss linear; }")
    conv = moves(recover.run(case))["first-order convection"]
    assert "div(rhoPhi,U) Gauss upwind;" in conv["edit"]
    assert "alpha" not in conv["edit"]


def test_a_crash_with_no_onset_does_not_size_the_step_from_the_crash(recover, tmp_path):
    """A sudden FPE: residuals fine to the end, the Courant number 300 on the last
    step. The step is sized from 0.5, not 300."""
    log = transient_log([0.1, 0.2, 0.3, 0.4, 0.5], [0.4, 0.4, 0.4, 0.5, 300.0],
                        [1e-3, 1e-3, 1e-3, 1e-3, 1e-3], FPE_TAIL)
    case = write_case(tmp_path, application="pimpleFoam", control="deltaT 0.001;",
                      solution=PIMPLE_ONE, log=log)
    report = recover.run(case)
    assert report["onset"] is None and report["diverged"] is True
    step = moves(report)["time step"]
    assert "before the last step" in step["read"]
    assert "Courant max 0.5" in step["read"]
    assert step["status"] == "done", "0.5 is already low; 300 was the crash, not the cause"
    assert step["edit"] == ""


def test_interface_courant_lines_are_not_read_as_courant(recover, tmp_path):
    log = ("Courant Number mean: 0.1 max: 0.4\nInterface Courant Number mean: 0 max: 9\n"
           "Time = 0.1\n")
    case = write_case(tmp_path, application="interFoam", solution=PIMPLE_ONE, log=log)
    assert [c[2] for c in recover.run(case)["log"]["courant_tail"]] == [0.4]


def test_a_decomposed_case_restarts_from_its_processor_times(recover, tmp_path):
    log = transient_log([0.1, 0.2, 0.3, 0.4], [0.3, 0.3, 0.3, 0.3], [1e-3, 1e-3, 1e-3, 5.0])
    case = write_case(tmp_path, application="pimpleFoam", control="deltaT 0.001;",
                      solution=PIMPLE_ONE, log=log)
    for t in ("0", "0.1", "0.2"):
        (case / "processor0" / t).mkdir(parents=True)
    restart = moves(recover.run(case))["restart"]
    assert "startTime 0.2;" in restart["edit"]


def test_a_named_log_that_is_missing_is_said(recover, tmp_path):
    case = write_case(tmp_path, application="pimpleFoam", solution=PIMPLE_ONE)
    report = recover.run(case, log="log.nope")
    assert any("no such log" in note for note in report["errors"])


def test_non_orthogonal_correctors_follow_the_measured_mesh(recover, tmp_path):
    check = "    Mesh non-orthogonality Max: 78.4 average: 12.1\n"
    case = write_case(tmp_path, application="pimpleFoam", solution=PIMPLE_ONE,
                      check_mesh=check)
    move = moves(recover.run(case))["non-orthogonal correctors"]
    assert move["status"] == "try"
    assert "78.4" in move["read"]
    assert "nNonOrthogonalCorrectors 2;" in move["edit"]


def test_a_mild_mesh_gets_no_corrector_advice(recover, tmp_path):
    check = "    Mesh non-orthogonality Max: 40 average: 5\n"
    case = write_case(tmp_path, application="pimpleFoam", solution=PIMPLE_ONE,
                      check_mesh=check)
    assert moves(recover.run(case))["non-orthogonal correctors"]["status"] == "n/a"


def test_potential_start_needs_a_uniform_field_and_no_written_time(recover, tmp_path):
    case = write_case(tmp_path, application="simpleFoam", solution="SIMPLE { }\n")
    move = moves(recover.run(case))["potential-flow start"]
    assert move["status"] == "try"
    assert "potentialFoam" in move["edit"]
    assert "Phi solver" in move["edit"], "this fvSolution has neither"


def test_a_log_that_never_diverged_says_so(recover, tmp_path, capsys):
    log = transient_log([0.1, 0.2, 0.3], [0.4, 0.4, 0.4], [1e-3, 5e-4, 2e-4])
    case = write_case(tmp_path, application="pimpleFoam", control="deltaT 0.001;",
                      solution=PIMPLE_ONE, log=log)
    report = recover.run(case)
    assert report["diverged"] is False
    assert recover.main([str(case)]) == 0
    assert "nothing in it says the run diverged" in capsys.readouterr().out


def test_json_and_a_missing_case_both_exit_zero(recover, tmp_path, capsys):
    assert recover.main([str(tmp_path / "nowhere"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] and report["moves"] == []


def test_the_docstring_keeps_it_advisory_and_puts_the_mesh_last(recover):
    doc = recover.__doc__
    assert "edits nothing, refuses nothing, and blocks nothing" in doc
    assert "exit code is 0 whatever" in doc
    assert "python3" in doc
    assert doc.index("**the mesh**") > doc.index("**restart**")


def test_it_is_stdlib_numpy_and_its_own_siblings(recover):
    source = (TOOLBOX / "recover.py").read_text(encoding="utf-8")
    siblings = {script.stem for script in TOOLBOX.glob("*.py")}
    allowed = set(sys.stdlib_module_names) | siblings | {"__future__", "numpy"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        for root in roots:
            assert root in allowed, f"recover.py imports {root}"


def test_the_toolbox_index_carries_it():
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    assert "recover.py" in index
