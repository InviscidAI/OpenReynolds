"""`amr.py`: the dictionary that drives OpenFOAM's own refinement, and what it requires.

AMR on this image is `dynamicRefineFvMesh` named in `constant/dynamicMeshDict`, so the
deliverable is a dictionary and the thing worth testing is whether it carries the nine
entries the class reads with no default. Every fact asserted here was read from the
OpenFOAM v2512 tag on 2026-11-21 (`src/dynamicFvMesh/dynamicRefineFvMesh/*`,
`applications/solvers/{incompressible/simpleFoam,incompressible/pimpleFoam,multiphase/interFoam}/*.C`,
`tutorials/multiphase/interFoam/laminar/damBreakWithObstacle/constant/dynamicMeshDict`)
and not from a tutorial copy or from memory, which is the difference between this file
and the two wrong reference values `benchmarks/README.md` records.

What is deliberately NOT here: a number. The honest check on refinement is
self-consistency -- a refined run agreeing with a uniformly fine one, for fewer cells --
and that needs two OpenFOAM runs. None has been made, so nothing below asserts an
accuracy or a saving, and `test_the_docstring_claims_no_accuracy_it_did_not_measure`
holds that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def amr():
    return load("amr")


def a_case(root: Path, application: str = "pimpleFoam") -> Path:
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "system" / "controlDict").write_text(
        f"application     {application};\nstartFrom       latestTime;\nendTime  1;\n",
        encoding="utf-8",
    )
    return root


def a_field(case: Path, time: str, name: str, cls: str) -> None:
    (case / time).mkdir(parents=True, exist_ok=True)
    (case / time / name).write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {cls};\n    object {name};\n}}\n\ninternalField uniform 0;\n",
        encoding="utf-8",
    )


# -- the dictionary ------------------------------------------------------------


def test_the_dictionary_carries_every_entry_that_has_no_default(amr):
    """Each of the nine is read with `get<>` or `readEntry`, so a dictionary short of
    one fatals at the first update instead of falling back. `correctFluxes` is the one
    nobody expects to be mandatory: it is read in `readDict()` as
    `get<List<Pair<word>>>("correctFluxes")`."""
    text = amr.dynamic_mesh_dict("alpha.water", 0.001, 0.999, max_cells=200000)

    assert "dynamicFvMesh   dynamicRefineFvMesh;" in text
    assert amr.COEFFS in text
    for entry in amr.REQUIRED:
        assert entry in text, f"{entry} has no default and is not in the dictionary"


def test_unrefine_level_is_the_only_entry_left_out_and_its_absence_is_explained(amr):
    """`getOrDefault<scalar>("unrefineLevel", GREAT)`: left out, nothing is ever merged
    back, which on a moving feature means the mesh only grows."""
    assert "unrefineLevel" not in amr.REQUIRED
    without = amr.dynamic_mesh_dict("alpha.water", 0.001, 0.999, max_cells=1000)
    assert "GREAT" in without and "only grows" in without
    assert "    unrefineLevel" not in without

    with_it = amr.dynamic_mesh_dict("alpha.water", 0.001, 0.999, max_cells=1000, unrefine=10)
    assert "unrefineLevel   10;" in with_it


def test_the_three_fatal_ranges_are_refused_where_they_cost_a_second(amr):
    for kwargs in ({"refine_interval": 0}, {"max_cells": 0}, {"max_refinement": 0}):
        with pytest.raises(amr.DictError) as raised:
            amr.dynamic_mesh_dict("alpha.water", 0.001, 0.999,
                                  **{"max_cells": 1000, **kwargs})
        assert "fatal" in str(raised.value)


def test_the_maxrefinement_message_is_quoted_with_its_own_bug(amr):
    """OpenFOAM's FatalError for an illegal `maxRefinement` says "The maxCells setting
    in the dynamicMeshDict should be > 0." A reader who takes that at face value goes
    looking at the wrong entry, so the quote carries the correction with it."""
    message = amr.POSITIVE["maxRefinement"][2]
    assert "maxCells setting" in message
    assert "the check is on maxRefinement" in message


def test_an_empty_refinement_band_is_refused_although_openfoam_would_accept_it(amr):
    """Not a fatal: lower >= upper marks no cell, so the run goes the whole way, pays
    for the topology machinery and refines nothing. That is the failure that looks like
    a success."""
    with pytest.raises(amr.DictError) as raised:
        amr.dynamic_mesh_dict("alpha.water", 0.999, 0.001, max_cells=1000)
    assert "band is empty" in str(raised.value)


def test_the_default_flux_pair_is_the_one_a_tutorial_writes(amr):
    """`damBreakWithObstacle`'s own dictionary sets its fluxes to `none`, read at the
    v2512 tag. A default here is a tutorial's choice or it is a guess."""
    text = amr.dynamic_mesh_dict("alpha.water", 0.001, 0.999, max_cells=1000)
    assert "correctFluxes   ( (phi none) );" in text

    named = amr.dynamic_mesh_dict("alpha.water", 0.001, 0.999, max_cells=1000,
                                  fluxes=(("phi", "U"), ("rhoPhi", "none")))
    assert "correctFluxes   ( (phi U) (rhoPhi none) );" in named


def test_the_dictionary_says_which_script_wrote_it(amr):
    """The file furniture comes from `case_gen.py`, whose banner names itself."""
    text = amr.dynamic_mesh_dict("alpha.water", 0.001, 0.999, max_cells=1000)
    assert "amr.py" in text
    assert "case_gen.py" not in text


def test_what_is_written_reads_back_as_what_was_asked_for(amr, tmp_path):
    case = a_case(tmp_path / "case")
    assert amr.main(["write", str(case), "--field", "mag(U)", "--lower", "2",
                     "--upper", "8", "--max-cells", "400000", "--max-refinement", "3",
                     "--unrefine", "1"]) == 0

    found = amr.check(case)
    assert [f.status for f in found].count("fail") == 0
    measured = " ".join(f.measured for f in found)
    assert "mag(U) is between 2 and 8" in measured
    assert "maxCells 400000" in measured


def test_it_refuses_to_overwrite_a_dictionary_somebody_wrote(amr, tmp_path):
    case = a_case(tmp_path / "case")
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "dynamicMeshDict").write_text("// hand written\n", encoding="utf-8")

    assert amr.main(["write", str(case), "--field", "x", "--lower", "0", "--upper", "1",
                     "--max-cells", "10"]) == 2
    assert (case / "constant" / "dynamicMeshDict").read_text() == "// hand written\n"


# -- reading a case back -------------------------------------------------------


def test_a_case_with_no_dictionary_is_told_that_nothing_refines(amr, tmp_path):
    case = a_case(tmp_path / "case")
    found = amr.check(case)
    assert [f.status for f in found] == ["fail"]
    assert "no constant/dynamicMeshDict" in found[0].measured
    assert "case_gen.py writes no dynamicMeshDict" in found[0].meaning


def test_a_dictionary_naming_a_mesh_that_does_not_refine_is_not_amr(amr, tmp_path):
    case = a_case(tmp_path / "case")
    (case / "constant").mkdir(parents=True)
    (case / "constant" / "dynamicMeshDict").write_text(
        "dynamicFvMesh   dynamicMotionSolverFvMesh;\nmotionSolver sixDoFRigidBodyMotion;\n",
        encoding="utf-8")

    found = amr.check(case)
    assert found[0].status == "fail"
    assert "dynamicFvMesh is dynamicMotionSolverFvMesh" in found[0].measured


def test_a_coefficients_block_short_of_an_entry_is_named_before_the_run(amr, tmp_path):
    case = a_case(tmp_path / "case")
    (case / "constant").mkdir(parents=True)
    (case / "constant" / "dynamicMeshDict").write_text(
        "dynamicFvMesh   dynamicRefineFvMesh;\n"
        f"{amr.COEFFS}\n{{\n"
        "    refineInterval 1;\n    field alpha.water;\n"
        "    lowerRefineLevel 0.001;\n    upperRefineLevel 0.999;\n"
        "    nBufferLayers 1;\n    maxRefinement 2;\n    maxCells 200000;\n"
        "    dumpLevel true;\n}\n",
        encoding="utf-8")

    found = amr.check(case)
    failures = [f for f in found if f.status == "fail"]
    assert failures and "correctFluxes" in failures[0].measured


def test_a_missing_coefficients_block_is_not_an_io_error_and_is_still_fatal(amr, tmp_path):
    """`optionalSubDict` means the block's absence parses; the entries inside it are
    read with `get<>`, so the run stops at the first `refineInterval` instead."""
    case = a_case(tmp_path / "case")
    (case / "constant").mkdir(parents=True)
    (case / "constant" / "dynamicMeshDict").write_text(
        "dynamicFvMesh   dynamicRefineFvMesh;\n", encoding="utf-8")

    found = amr.check(case)
    assert found[-1].status == "fail"
    assert "optionalSubDict" in found[-1].meaning


def test_a_flux_on_disk_that_the_list_does_not_name_is_a_warning_and_not_an_error(amr, tmp_path):
    """The run continues and exits 0 with "Cannot find surfaceScalarField <name>" in
    the log, so this is read off the dictionary and the time directories rather than
    off an exit code."""
    case = a_case(tmp_path / "case")
    amr.main(["write", str(case), "--field", "alpha.water", "--lower", "0.001",
              "--upper", "0.999", "--max-cells", "1000"])
    a_field(case, "0.5", "phi", "surfaceScalarField")
    a_field(case, "0.5", "rhoPhi", "surfaceScalarField")
    a_field(case, "0.5", "U", "volVectorField")

    warnings = [f for f in amr.check(case) if f.status == "warn"]
    text = " ".join(f.measured for f in warnings)
    assert "rhoPhi" in text
    assert "phi," not in text and "U" not in text.replace("rhoPhi", ""), (
        "phi is named in the list and U is not a flux"
    )


def test_the_fields_read_are_the_ones_whose_own_class_says_surfacescalarfield(amr, tmp_path):
    case = a_case(tmp_path / "case")
    a_field(case, "0", "p", "volScalarField")
    a_field(case, "1", "phi", "surfaceScalarField")
    assert amr.surface_scalar_fields(case) == {"phi": "1"}


def test_the_steady_solver_is_reported_as_one_that_reads_this_dictionary(amr, tmp_path):
    """simpleFoam at v2512 includes createDynamicFvMesh.H and calls
    mesh.controlledUpdate() in its simple.loop(), so refinement on a steady solve needs
    no change of solver -- and `case_gen.py:1012` maps a steady study to simpleFoam."""
    case = a_case(tmp_path / "case", application="simpleFoam")
    amr.main(["write", str(case), "--field", "mag(U)", "--lower", "1", "--upper", "5",
              "--max-cells", "1000"])

    measured = " ".join(f.measured for f in amr.check(case))
    assert "application simpleFoam constructs a dynamicFvMesh" in measured


def test_a_solver_nobody_read_is_a_question_rather_than_a_refusal(amr, tmp_path):
    case = a_case(tmp_path / "case", application="rhoCentralFoam")
    amr.main(["write", str(case), "--field", "T", "--lower", "300", "--upper", "400",
              "--max-cells", "1000"])

    warned = [f for f in amr.check(case) if f.status == "warn"]
    assert any("rhoCentralFoam was not read for this" in f.measured for f in warned)
    assert any("createMesh.H" in f.meaning for f in warned)


def test_maxcells_is_read_against_the_mesh_that_is_there(amr, tmp_path):
    case = a_case(tmp_path / "case")
    mesh = case / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "owner").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class labelList;\n"
        '    note "nPoints:1200 nCells:1000 nFaces:3400 nInternalFaces:3000";\n'
        "    object owner;\n}\n", encoding="utf-8")
    amr.main(["write", str(case), "--field", "mag(U)", "--lower", "1", "--upper", "5",
              "--max-cells", "200000"])

    assert amr.cells_now(case) == 1000.0
    found = [f for f in amr.check(case) if "maxCells" in f.measured]
    assert found and "200.0x" in found[0].measured
    assert found[0].status == "warn", "200 times the mesh it starts from is worth a word"


def test_a_mesh_whose_header_carries_no_count_is_not_guessed_at(amr, tmp_path):
    case = a_case(tmp_path / "case")
    mesh = case / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "owner").write_text("FoamFile\n{\n    class labelList;\n}\n", encoding="utf-8")
    assert amr.cells_now(case) is None


# -- what the file claims ------------------------------------------------------


def test_the_docstring_claims_no_accuracy_it_did_not_measure(amr):
    """The owner's bar: improve things only where physically accurate against
    real-world data. On refinement the check is self-consistency rather than a
    reference, and no pair of runs exists here, so the file says so instead of quoting
    a saving somebody else measured on another code."""
    doc = " ".join(amr.__doc__.split())
    assert "No such pair has been run on this image" in doc
    assert "states no accuracy and no saving" in doc
    assert "grid.py" in doc, "the instrument that would make the measurement is named"


def test_the_docstring_says_where_every_fact_in_it_was_read(amr):
    doc = " ".join(amr.__doc__.split())
    assert "gitlab.com/openfoam/core/openfoam" in doc
    assert "optionalSubDict" in doc and "get<>" in doc
    assert "protectedCell_" in doc, "non-hex cells are not refined, which decides the mesher"
    assert "preflight.py:2841" in doc, "ran=False stays the reading; this file does not move it"


def test_non_hex_cells_are_named_as_the_limit_on_which_mesh_this_works_on(amr):
    doc = " ".join(amr.__doc__.split())
    assert "hexRef8" in doc
    for mesher in ("blockMesh", "snappyHexMesh", "cartesianMesh"):
        assert mesher in doc
