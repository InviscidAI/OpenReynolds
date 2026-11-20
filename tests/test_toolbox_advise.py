"""`advise.py`: which solver, and how many ranks.

Both answers are read out of the case, and the thing worth testing is where the script
refuses to answer. A rank count is the kind of number that gets produced from a
plausible curve nobody measured, and this corpus has no curve: of 103 case directories
in the local studies, exactly one had been run at more than one rank count. So the
guard asserted hardest here is the negative one -- one rank count yields no speedup
figure, and an unestablished Mach number yields no solver pick.

The parser underneath it is the other risk. OpenFOAM appends, so a restarted case
leaves two or three runs in one log file, and reading such a file whole overstates the
iteration count and understates the per-iteration cost. `case_N32` in the corpus has a
`log.simpleFoam` whose last `ExecutionTime` is 376 s and whose `Time =` lines number
3,590, where the 376 s covers the 1,600 iterations after a restart. That shape is
fixtured below.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def advise():
    spec = importlib.util.spec_from_file_location("toolbox_advise", TOOLBOX / "advise.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- fixtures ----------------------------------------------------------------------


def run_text(application: str, ranks: int, stamps: list[float]) -> str:
    """One OpenFOAM run, banner and all, at a given per-iteration cost."""
    head = (
        "/*---------------------------------------------------------------------------*\\\n"
        f"Build  : _87ed40d2-20251219 OPENFOAM=2512 version=2512\n"
        f"Exec   : {application}" + (" -parallel\n" if ranks > 1 else "\n")
        + f"nProcs : {ranks}\n"
    )
    body = "".join(
        f"Time = {index + 1}\nExecutionTime = {value:.2f} s  ClockTime = {value:.0f} s\n"
        for index, value in enumerate(stamps)
    )
    return head + body + "\nEnd\n"


def case_with(root: Path, logs: dict[str, str], application: str = "simpleFoam") -> Path:
    case = root / "case"
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(f"application     {application};\n")
    (case / "system" / "blockMeshDict").write_text(
        "convertToMeters 1;\n"
        "vertices ( (0 0 0) (2 0 0) (2 1 0) (0 1 0) (0 0 1) (2 0 1) (2 1 1) (0 1 1) );\n"
        "blocks ( hex (0 1 2 3 4 5 6 7) (40 20 20) simpleGrading (1 1 1) );\n"
        "boundary ( inlet { type patch; faces ((0 4 7 3)); } "
        "outlet { type patch; faces ((1 2 6 5)); } );\n"
    )
    for name, text in logs.items():
        (case / name).write_text(text)
    return case


def with_mesh(case: Path, cells: int) -> Path:
    """A checkMesh log, which is one of the two places `preflight` reads a real count."""
    (case / "log.checkMesh").write_text(
        f"Mesh stats\n    points:           {cells * 2}\n    cells:            {cells}\n"
        "Checking geometry...\nMesh OK.\n\nEnd\n"
    )
    return case


# -- the restarted log, which is the load-bearing parse ----------------------------


def test_a_restarted_log_is_two_runs_not_one(tmp_path, advise):
    """`>>` appends, so one file holds two banners. Whole-file reading would call this
    40 iterations at the second run's ExecutionTime and get both numbers wrong."""
    first = run_text("simpleFoam", 2, [1.0 * n for n in range(1, 21)])
    second = run_text("simpleFoam", 4, [0.5 * n for n in range(1, 21)])
    case = case_with(tmp_path, {"log.simpleFoam": first + second})

    runs = advise.runs_in(case / "log.simpleFoam")
    assert len(runs) == 2
    assert [run["ranks"] for run in runs] == [2, 4]
    assert [run["runs_in_file"] for run in runs] == [2, 2]
    # The median of the differences, not the last ExecutionTime over the count.
    assert runs[0]["median_s_per_iteration"] == pytest.approx(1.0)
    assert runs[1]["median_s_per_iteration"] == pytest.approx(0.5)


def test_the_median_ignores_a_startup_or_a_write_spike(tmp_path, advise):
    """Total over count carries the mesh read and every field write; the median of the
    steps is the cost of a step. One 50 s write must not move it."""
    stamps = [100.0, 101.0, 102.0, 152.0, 153.0, 154.0, 155.0]
    case = case_with(tmp_path, {"log.simpleFoam": run_text("simpleFoam", 1, stamps)})
    run, = advise.runs_in(case / "log.simpleFoam")
    assert run["median_s_per_iteration"] == pytest.approx(1.0)
    assert run["iterations"] == len(stamps)


def test_a_meshing_log_is_not_a_solver_run(tmp_path, advise):
    """`log.decomposePar` and friends carry ExecutionTime too, and a decomposition
    counted as a solver run at 1 rank would invent a serial baseline."""
    case = case_with(tmp_path, {
        "log.decomposePar": run_text("decomposePar", 1, [1.0, 2.0, 3.0, 4.0]),
        "log.reconstructPar": run_text("reconstructPar", 1, [1.0, 2.0, 3.0, 4.0]),
        "log.simpleFoam": run_text("simpleFoam", 4, [1.0, 2.0, 3.0, 4.0]),
    })
    by_ranks = advise.measured(advise_case(advise, case))["by_ranks"]
    assert [row["ranks"] for row in by_ranks] == [4]
    assert by_ranks[0]["applications"] == ["simpleFoam"]


def advise_case(advise, path: Path):
    """`preflight.Case` as `advise` imported it, so the test does not import it twice."""
    return sys.modules["toolbox_advise"].preflight.Case(path)


# -- the negative guards -----------------------------------------------------------


def test_one_rank_count_reports_no_speedup(tmp_path, advise):
    """The whole point. One point is not a curve and a ratio off it would be invented."""
    case = with_mesh(case_with(tmp_path, {
        "log.simpleFoam": run_text("simpleFoam", 4, [1.0 * n for n in range(1, 31)]),
    }), 400_000)
    data = advise.inspect(case, cores=8)
    assert data["measured"]["comparison"] is None
    assert len(data["measured"]["by_ranks"]) == 1
    report = advise.render(data)
    assert "one rank count only" in report
    assert "no speedup to report" in report


def test_two_rank_counts_report_the_measured_ratio(tmp_path, advise):
    """Four times the ranks for twice the speed is reported as exactly that, and the
    ratio is arithmetic on the two medians rather than an efficiency verdict."""
    case = with_mesh(case_with(tmp_path, {
        "log.simpleFoam": run_text("simpleFoam", 2, [2.0 * n for n in range(1, 21)]),
        "log.simpleFoam_8way": run_text("simpleFoam", 8, [1.0 * n for n in range(1, 21)]),
    }), 400_000)
    comparison = advise.inspect(case, cores=8)["measured"]["comparison"]
    assert comparison["from_ranks"] == 2
    assert comparison["to_ranks"] == 8
    assert comparison["rank_ratio"] == pytest.approx(4.0)
    assert comparison["speedup"] == pytest.approx(2.0)


def test_an_unestablished_mach_number_picks_nothing(tmp_path, advise):
    """A Mach number is not in a case's files, and the solver comparison turns on it.
    Both families are offered and none is marked."""
    case = with_mesh(case_with(tmp_path, {}), 400_000)
    data = advise.inspect(case)
    assert data["mach"] is None
    assert not any(row["mach_pick"] for row in data["solvers"])
    assert "--mach says it" in data["mach_source"]
    assert "not established" in advise.render(data)


def test_a_given_mach_number_agrees_with_compressible(tmp_path, advise):
    """The pick is `compressible.choose_solver`'s, not a second reading of the same
    number -- two readings would be two places to disagree about M 1."""
    compressible = sys.modules["toolbox_advise"].compressible
    case = with_mesh(case_with(tmp_path, {}), 400_000)
    for mach in (0.2, 0.84, 2.0):
        data = advise.inspect(case, mach=mach)
        expected, _why = compressible.choose_solver("auto", mach, "steady")
        marked = [row["name"] for row in data["solvers"] if row["mach_pick"]]
        assert marked == [expected], f"M {mach}"


def test_a_mach_number_that_decides_nothing_says_so(tmp_path, advise):
    """A VOF case does not branch on a Mach number, and a report that printed the
    regime and went quiet would read as the number having been used. It said "both the
    incompressible and the compressible candidates are below" on a case that listed
    neither, which is the bug this pins."""
    case = with_mesh(case_with(tmp_path, {}, application="interFoam"), 400_000)
    (case / "constant").mkdir(exist_ok=True)
    (case / "0").mkdir(exist_ok=True)
    (case / "0" / "alpha.water").write_text("dimensions [0 0 0 0 0 0 0];\n")

    data = advise.inspect(case, mach=0.5, cores=4)
    assert data["class"] == "free-surface-marine"
    families = {row["family"] for row in data["solvers"]}
    assert families == {"free-surface"}
    report = advise.render(data)
    assert "decides nothing here" in report
    assert "undivided" not in report


# -- cells, and where they came from -----------------------------------------------


def test_a_built_mesh_beats_the_estimate(tmp_path, advise):
    case = with_mesh(case_with(tmp_path, {}), 398_336)
    cells, source = advise.cells_in(advise_case(advise, case))
    assert cells == 398_336
    assert source == "the built mesh"


def test_with_no_mesh_the_count_is_the_documented_arithmetic(tmp_path, advise):
    """`cells_estimate.estimate` is reused rather than re-derived, so the number here
    is the one that script's own test pins: background + surface band + layers."""
    cells_estimate = sys.modules["toolbox_advise"].cells_estimate
    case = case_with(tmp_path, {})
    (case / "system" / "snappyHexMeshDict").write_text(
        "castellatedMeshControls\n{\n    refinementSurfaces\n    {\n"
        "        body\n        {\n            level (2 3);\n        }\n    }\n}\n"
        "addLayersControls\n{\n    nSurfaceLayers 4;\n}\n"
    )
    surface = case / "constant" / "triSurface"
    surface.mkdir(parents=True)
    triangle = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    (surface / "body.stl").write_bytes(
        b"\0" * 80 + struct.pack("<I", 1)
        + struct.pack("<12fH", 0, 0, 1, *triangle, 0)
    )

    cells, source = advise.cells_in(advise_case(advise, case))
    expected = cells_estimate.estimate(
        delta0=0.05, volume=2.0, levels={"body": 3}, area=0.5, n_layers=4)
    assert cells == int(expected["total"])
    assert "cells_estimate.py" in source


def test_with_neither_a_mesh_nor_a_block_dict_nothing_is_claimed(tmp_path, advise):
    case = tmp_path / "bare"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text("application     simpleFoam;\n")
    data = advise.inspect(case, cores=8)
    assert data["cells"] is None
    assert data["ranks"] == []
    assert "cells: unknown" in advise.render(data)


def test_the_rank_table_stops_at_the_cores_there_are(tmp_path, advise):
    case = with_mesh(case_with(tmp_path, {}), 1_000_000)
    rows = advise.inspect(case, cores=6)["ranks"]
    assert [row["ranks"] for row in rows] == [1, 2, 4, 6]
    assert rows[1]["cells_per_rank"] == 500_000
    # The ceiling is stated with the study it was measured in, not as a round number.
    assert "20260830-164816-eee0" in advise.render(advise.inspect(case, cores=6))


# -- the free-will contract --------------------------------------------------------


def test_the_exit_code_is_zero_whatever_the_report_says(tmp_path, advise, capsys):
    case = with_mesh(case_with(tmp_path, {}), 400_000)
    assert advise.main([str(case), "--cores", "4"]) == 0
    assert "# the case" in capsys.readouterr().out
    assert advise.main([str(case), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["class"] == "internal-flow"


def test_it_offers_rather_than_instructs(advise):
    """Advisory output informs. A script that told the agent what it had to run would
    be the workflow injection docs/design.md section 1 exists to forbid."""
    text = (TOOLBOX / "advise.py").read_text(encoding="utf-8").lower()
    for imperative in ("you must", "you should", "always run", "do not proceed",
                       "before you run", "step 1"):
        assert imperative not in text, imperative


def test_it_writes_nothing_into_the_case(tmp_path, advise):
    case = with_mesh(case_with(tmp_path, {
        "log.simpleFoam": run_text("simpleFoam", 4, [1.0, 2.0, 3.0, 4.0]),
    }), 400_000)
    before = {path.name for path in case.rglob("*")}
    advise.inspect(case, cores=4)
    assert {path.name for path in case.rglob("*")} == before


def test_the_docstring_shows_how_to_run_it(advise):
    doc = ast.get_docstring(ast.parse((TOOLBOX / "advise.py").read_text(encoding="utf-8")))
    assert "python3 advise.py" in doc
