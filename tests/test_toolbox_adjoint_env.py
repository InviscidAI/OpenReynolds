"""The adjoint probe, against a tutorial tree built from the real v2512 one.

The fixture below is the dictionary set of ESI v2512's own
`tutorials/incompressible/adjointOptimisationFoam/sensitivityMaps/naca0012/laminar/drag`,
read from the release tag on gitlab.com (ref `OpenFOAM-v2512`, tag commit
`87ed40d256d22ea38fcc648dfc82a22162427b18`) rather than from a release note or a
secondary source -- `benchmarks/README.md` is the argument for why that distinction is
worth the trouble. The keys the probe reads out of it are that case's actual keys, so
a key renamed in a later release makes this fail rather than making the probe quietly
report blanks.

Nothing here runs OpenFOAM. The probe's job is to report what is on an instance, and
what is checkable off the instance is that it reads a real case correctly, refuses
what it cannot read, and claims nothing about a sensitivity having been verified.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def adjoint_env():
    return load("adjoint_env")


OPTIMISATION_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      optimisationDict;
}

optimisationManager    singleRun;

primalSolvers
{
    op1
    {
        active                 true;
        type                   incompressible;
        solver                 simple;
        solutionControls
        {
            nIters 3000;
            residualControl
            {
                "p.*"       1.e-7;
                "U.*"       1.e-7;
            }
        }
    }
}

adjointManagers
{
    adjManager1
    {
        primalSolver             op1;
        adjointSolvers
        {
            adjS1
            {
                active                 true;
                type                   incompressible;
                solver                 adjointSimple;
                objectives
                {
                    type               incompressible;
                    objectiveNames
                    {
                        drag
                        {
                            weight          1.;
                            type            force;
                            patches         (pressure suction);
                            direction       (0.9993908270189999 0.034899496703 0);
                            Aref            2.;
                            rhoInf          1.225;
                            UInf            1;
                        }
                    }
                }
                ATCModel
                {
                    ATCModel        standard;
                }
                solutionControls
                {
                    nIters 3000;
                    residualControl
                    {
                        "pa.*"       1.e-7;
                        "Ua.*"       1.e-7;
                    }
                }
            }
        }
    }
}

optimisation
{
    designVariables
    {
        sensitivityType surfacePoints;
        patches         (pressure suction);
    }
}
"""

ADJOINT_RAS = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      RASProperties;
}

adjointRASModel   adjointLaminar;

adjointTurbulence on;

printCoeffs       off;
"""

CONTROL_DICT = """\
application     adjointOptimisationFoam;

startFrom       latestTime;

stopAt          endTime;

endTime         6000;

deltaT          1;

writeControl    timeStep;

writeInterval   1000;
"""

FV_SCHEMES = """\
ddtSchemes
{
    default         steadyState;
}

divSchemes
{
    default                 Gauss linear;
    div(phi,U)      bounded Gauss linearUpwind gradUConv;
    div(-phi,Ua)    bounded Gauss linearUpwind gradUaConv;
}
"""

FV_SOLUTION = """\
solvers
{
    "(p|pa).*"
    {
        solver           PCG;
        preconditioner   DIC;
    };

    "(U|Ua).*"
    {
        solver           PBiCGStab;
        preconditioner   DILU;
    }
}
"""


def write_case(root: Path, where: str, *, objective: str = "drag",
               turbulence: str = "adjointLaminar", manager: str = "singleRun",
               complete: bool = True) -> Path:
    """One tutorial case on disk, in the v2512 layout."""
    case = root / where
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir(parents=True)
    (case / "0.orig").mkdir(parents=True)
    text = (OPTIMISATION_DICT
            .replace("optimisationManager    singleRun;", f"optimisationManager    {manager};")
            .replace("                        drag\n", f"                        {objective}\n"))
    (case / "system" / "optimisationDict").write_text(text, encoding="utf-8")
    (case / "system" / "controlDict").write_text(CONTROL_DICT, encoding="utf-8")
    (case / "system" / "fvSchemes").write_text(FV_SCHEMES, encoding="utf-8")
    if complete:
        (case / "system" / "fvSolution").write_text(FV_SOLUTION, encoding="utf-8")
    (case / "constant" / "adjointRASProperties").write_text(
        ADJOINT_RAS.replace("adjointLaminar", turbulence), encoding="utf-8")
    for field in ("U", "Ua", "p", "pa"):
        (case / "0.orig" / field).write_text("// a field\n", encoding="utf-8")
    (case / "Allrun").write_text("#!/bin/sh\n", encoding="utf-8")
    return case


@pytest.fixture
def tutorials(tmp_path, monkeypatch, adjoint_env):
    """A `$FOAM_TUTORIALS` with the adjoint tree under it, as an instance has."""
    root = tmp_path / "tutorials"
    adjoint = root / adjoint_env.TUTORIALS
    write_case(adjoint, "sensitivityMaps/naca0012/laminar/drag")
    write_case(adjoint, "sensitivityMaps/naca0012/laminar/lift", objective="lift")
    write_case(adjoint, "shapeOptimisation/sbend/turbulent",
               turbulence="adjointSpalartAllmaras", manager="steadyOptimisation")
    (adjoint / "resources" / "meshes").mkdir(parents=True)
    monkeypatch.setenv("FOAM_TUTORIALS", str(root))
    return adjoint


# -- what it says when the solver is not there -------------------------------------


def test_a_solver_not_on_path_is_reported_as_a_fact_about_the_image(adjoint_env,
                                                                   monkeypatch):
    """The distinction this probe exists for: absent HERE is not absent from OpenFOAM,
    and reporting the second would send somebody off to hand-roll an adjoint."""
    monkeypatch.setattr(adjoint_env.shutil, "which", lambda name: None)
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)

    found = adjoint_env.probe()
    text = adjoint_env.report(found)

    assert found["on_path"] is False
    assert found["binary"] == {"checked": "no", "why": "not on PATH"}
    assert "NOT on PATH on this instance" in text
    assert "source tree" in text
    assert "not about OpenFOAM" in text


def test_the_legacy_solver_being_present_is_not_evidence_for_the_new_one(adjoint_env,
                                                                        monkeypatch):
    """`adjointShapeOptimizationFoam` is a porosity-field topology optimisation and
    was the only adjoint name this package knew. It is not a substitute."""
    monkeypatch.setattr(adjoint_env.shutil, "which",
                        lambda name: "/opt/bin/legacy" if name == adjoint_env.LEGACY else None)
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)

    found = adjoint_env.probe()
    text = adjoint_env.report(found)

    assert found["on_path"] is False
    assert found["legacy_solver"]["on_path"] is True
    assert "not a substitute" in text


def test_a_binary_that_cannot_be_interrogated_is_unchecked_not_broken(adjoint_env,
                                                                      monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("Exec format error")

    monkeypatch.setattr(adjoint_env.subprocess, "run", refuse)
    assert adjoint_env.version("/opt/bin/x") == {
        "checked": "no", "why": "OSError: Exec format error"}


# -- reading the real tutorial set --------------------------------------------------


def test_the_installed_cases_are_found_by_their_optimisation_dict(adjoint_env,
                                                                  tutorials):
    """A case is a directory with a `system/optimisationDict`. The tutorial tree
    nests them three deep under a family and a turbulence model, so listing one level
    finds nothing."""
    found = adjoint_env.probe()

    assert len(found["cases"]) == 3
    names = sorted(str(Path(str(row["case"])).name) for row in found["cases"])
    assert names == ["drag", "lift", "turbulent"]


def test_a_sensitivity_map_is_the_single_run_manager_and_is_named_as_such(adjoint_env,
                                                                         tutorials):
    """`optimisationManager singleRun` computes a sensitivity and stops; the other
    managers run an optimisation loop, which is a different and much longer thing."""
    found = adjoint_env.probe()

    assert len(found["sensitivity_only_cases"]) == 2
    assert all(name.endswith(("drag", "lift")) for name in found["sensitivity_only_cases"])
    text = adjoint_env.report(found)
    assert "without an optimisation loop" in text


def test_each_case_reports_its_objective_sensitivity_type_and_adjoint_turbulence(
        adjoint_env, tutorials):
    found = adjoint_env.probe()
    drag = next(row for row in found["cases"] if str(row["case"]).endswith("drag"))

    assert drag["manager"] == "singleRun"
    assert drag["sensitivity_type"] == "surfacePoints"
    assert drag["objectives"] == ["force"]
    assert drag["adjoint_turbulence"] == "adjointLaminar"
    assert drag["initial_fields"] == ["U", "Ua", "p", "pa"]
    assert drag["complete"] is True


def test_the_turbulence_models_listed_are_the_ones_the_cases_use(adjoint_env, tutorials):
    """And the report says that, because it is not the same claim as what the build
    registers -- which nothing off the instance can enumerate."""
    found = adjoint_env.probe()
    text = adjoint_env.report(found)

    assert found["adjoint_turbulence_in_tutorials"] == [
        "adjointLaminar", "adjointSpalartAllmaras"]
    assert "not a list of what the build registers" in text


def test_a_case_missing_a_dictionary_is_reported_incomplete_rather_than_offered(
        adjoint_env, tmp_path, monkeypatch):
    root = tmp_path / "tutorials"
    write_case(root / adjoint_env.TUTORIALS, "half/a/case", complete=False)
    monkeypatch.setenv("FOAM_TUTORIALS", str(root))

    found = adjoint_env.probe()
    text = adjoint_env.report(found)

    assert found["cases"][0]["complete"] is False
    assert found["sensitivity_only_cases"] == []
    assert "INCOMPLETE, missing system/fvSolution" in text


def test_no_tutorials_is_said_rather_than_reported_as_no_cases(adjoint_env, monkeypatch,
                                                               tmp_path):
    monkeypatch.setenv("FOAM_TUTORIALS", str(tmp_path / "nothing-here"))
    found = adjoint_env.probe()
    assert found["tutorials_root"] == ""
    assert "no adjoint tutorials found" in adjoint_env.report(found)


# -- the claim it must never make ---------------------------------------------------


def test_the_probe_never_claims_a_sensitivity_has_been_checked(adjoint_env, tutorials):
    """The whole output is about what is installed. Nothing here has differentiated
    anything, and a reader must not come away thinking otherwise."""
    found = adjoint_env.probe()
    text = adjoint_env.report(found)

    assert found["sensitivity_checked_here"] is False
    assert "no sensitivity from this solver has been checked against a finite difference" \
        in text


def test_the_finite_difference_recipe_names_the_two_traps(adjoint_env, capsys):
    assert adjoint_env.main(["--finite-difference"]) == 0
    # normalised, because where a sentence of the recipe happens to wrap is not the point
    printed = " ".join(capsys.readouterr().out.split())

    assert "central difference" in printed.lower()
    # the perturbation has to move the mesh the primal solved on
    assert "unperturbed mesh" in printed
    # surfacePoints is per point, so the comparison is an inner product
    assert "inner product" in printed
    assert "two values of h" in printed


# -- copying one out ---------------------------------------------------------------


def test_a_case_is_copied_without_the_mesh_it_does_not_own(adjoint_env, tutorials,
                                                           tmp_path, capsys):
    """The tutorial cases keep their mesh in a shared `resources/` that their Allrun
    copies in. A copy that said nothing about that fails later on a missing polyMesh."""
    target = tmp_path / "work"
    target.mkdir()

    code = adjoint_env.main(["--case", "naca0012/laminar/drag", "--copy", str(target)])

    assert code == 0
    assert (target / "drag" / "system" / "optimisationDict").is_file()
    assert not (target / "drag" / "constant" / "polyMesh").exists()
    printed = capsys.readouterr().out
    assert "its mesh is NOT here" in printed


def test_copying_without_a_case_is_refused_because_there_are_several(adjoint_env,
                                                                     tutorials, tmp_path,
                                                                     capsys):
    assert adjoint_env.main(["--copy", str(tmp_path)]) == 2
    assert "--copy needs --case" in capsys.readouterr().out


def test_an_unknown_case_names_what_is_there(adjoint_env, tutorials, capsys):
    assert adjoint_env.main(["--case", "nonesuch"]) == 0
    assert "no tutorial case matching 'nonesuch'" in capsys.readouterr().out


# -- the CLI ------------------------------------------------------------------------


def test_the_json_carries_the_same_facts(adjoint_env, tutorials, capsys):
    assert adjoint_env.main(["--json"]) == 0
    found = json.loads(capsys.readouterr().out)

    assert found["solver"] == "adjointOptimisationFoam"
    assert len(found["cases"]) == 3
    assert found["sensitivity_checked_here"] is False


def test_one_case_prints_its_dictionary_set(adjoint_env, tutorials, capsys):
    assert adjoint_env.main(["--case", "naca0012/laminar/lift"]) == 0
    printed = capsys.readouterr().out

    for name in adjoint_env.DICTIONARIES:
        assert name in printed
    assert "initial fields: U, Ua, p, pa" in printed


# -- the one-word fix this probe found ---------------------------------------------


def test_the_adjoint_solver_counts_iterations_not_seconds():
    """`preflight.STEADY_APPLICATIONS` named the legacy adjoint solver and not this
    one, so a Courant number would have been reported for a run whose time column
    counts SIMPLE iterations.

    The evidence is the v2512 tutorial's own `system/controlDict` -- `application
    adjointOptimisationFoam`, `deltaT 1`, `endTime 6000` -- beside its `fvSchemes`
    `ddtSchemes { default steadyState; }`. Both are in this file's fixture above,
    quoted from the release tag.
    """
    preflight = load("preflight")

    assert "adjointOptimisationFoam" in preflight.STEADY_APPLICATIONS
    assert "adjointShapeOptimizationFoam" in preflight.STEADY_APPLICATIONS
    assert "steadyState" in FV_SCHEMES and "deltaT          1;" in CONTROL_DICT
