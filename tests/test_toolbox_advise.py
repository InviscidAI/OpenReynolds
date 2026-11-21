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
import re
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
    # The bug this pins is about the Mach number, so what matters is that no
    # compressible candidate is offered for this class -- not the exact set, which grew
    # when `sph.py` joined the free-surface comparison.
    assert "compressible" not in families
    assert "free-surface" in families
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


# -- the machine, and the budget against it ----------------------------------------
#
# The other half of the same question: a cell count means nothing without the machine,
# and the machine is what the corpus repeatedly got wrong -- `nproc` answering 4 and an
# agent decomposing for 12, and no memory reading taken anywhere at all. What is
# asserted here is mostly the refusals: no memory reading means no memory claim, an
# adjustable time step means no hour count, and the one measured memory pair is reported
# as one case with the study and the PIDs it came from.


def test_the_machine_probe_answers_on_whatever_this_is(advise):
    """Linux, Windows and macOS read memory three different ways and this uses all
    three. CI runs the first two, so the core count has to come back on both and memory
    has to come back as either a number or a sentence saying why not."""
    here = advise.machine()
    assert here["cores_logical"] >= 1
    assert 1 <= here["cores_usable"] <= here["cores_logical"]
    assert "os.cpu_count()" in here["cores_source"]
    if here["memory_bytes"] is None:
        assert "not established" in here["memory_source"]
    else:
        assert here["memory_bytes"] > 64 * 2**20, "a runner with under 64 MiB is a bug"
        assert here["memory_source"]


def test_an_unreadable_cgroup_is_not_a_limit(advise):
    """Both probes read files that are absent off Linux, and cgroup v1 spells "no limit"
    as a number near 2**63 rather than as a word. A ceiling invented from either would
    be a budget against nothing."""
    memory, _note = advise._cgroup_memory()
    assert memory is None or 0 < memory < 2**53
    cores, _note = advise._cgroup_cores()
    assert cores is None or cores > 0


def test_the_cost_band_falls_back_and_says_so(advise):
    """A solver with one run in the corpus has no band: p10 and p90 would be its own two
    numbers. It falls back to every run in the corpus, with the count in the sentence."""
    row, basis = advise.cost_band("buoyantSimpleFoam")
    assert row is advise.COST_PER_CELL_S[""]
    assert "under ten runs" in basis and "101 runs" in basis

    row, basis = advise.cost_band("simpleFoam")
    assert row["runs"] == 14 and "14 simpleFoam run(s)" in basis


def test_every_band_in_the_table_carries_its_sample_size(advise):
    """The table is the measurement, so each row says how many runs it came from, and
    ten is the floor a row needs to be its own row at all."""
    for application, row in advise.COST_PER_CELL_S.items():
        assert row["runs"] >= 10, application
        assert row["cases"] >= 1
        assert row["p10"] <= row["median"] <= row["p90"], application


def test_an_adjustable_time_step_yields_no_iteration_count(tmp_path, advise):
    """The honest None. With adjustTimeStep on, the number of steps depends on a Courant
    number the solver works out per step from a field nobody has yet."""
    case = case_with(tmp_path, {})
    (case / "system" / "controlDict").write_text(
        "application     pimpleFoam;\nendTime 10;\ndeltaT 0.001;\nadjustTimeStep yes;\n")
    count, source = advise.iterations_of(advise_case(advise, case))
    assert count is None
    assert "adjustTimeStep" in source

    data = advise.inspect(case, cores=4)
    assert "the hours below are left out" in advise.render(data)
    assert not any("hours_low" in row for row in data["ranks"])


def test_a_fixed_time_step_gives_the_hours_as_a_band(tmp_path, advise):
    """endTime over deltaT is the step count, and the hours are that count times the
    measured band for the load one rank carries -- a band, never a single number."""
    case = with_mesh(case_with(tmp_path, {}), 400_000)
    (case / "system" / "controlDict").write_text(
        "application     simpleFoam;\nstartTime 0;\nendTime 1000;\ndeltaT 1;\n")
    data = advise.inspect(case, cores=4, memory_gb=8)
    assert data["iterations"] == 1000

    band = advise.COST_PER_CELL_S["simpleFoam"]
    row = next(r for r in data["ranks"] if r["ranks"] == 4)
    assert row["s_low"] == pytest.approx(band["p10"] * 100_000)
    assert row["s_high"] == pytest.approx(band["p90"] * 100_000)
    assert row["hours_low"] == pytest.approx(band["p10"] * 100_000 * 1000 / 3600)
    report = advise.render(data)
    assert "s/iteration" in report
    assert "not a claim that twice the ranks is half the time" in report


def test_the_memory_claim_is_one_case_and_says_so(tmp_path, advise):
    """The rule the repository runs on: a number that reaches a user traces to a
    measurement. This one traces to two processes in one snapshot of one run, so the
    report names the study and the PIDs and promises no limit."""
    case = with_mesh(case_with(tmp_path, {}), 400_000)
    report = advise.render(advise.inspect(case, cores=4, memory_gb=8))
    assert "20260830-164816-eee0" in report
    assert "10382" in report and "10383" in report
    assert "One case, one snapshot" in report
    assert "no cell limit is promised" in report


def test_a_case_that_fits_is_told_so_and_offered_nothing(tmp_path, advise):
    """400,000 cells at 1.7 kB each is 0.7 GiB, and a list of smaller meshes under a
    case that fits in 8 GiB is noise."""
    case = with_mesh(case_with(tmp_path, {}), 400_000)
    data = advise.inspect(case, cores=4, memory_gb=8)
    assert data["fit"]["fits"] is True
    assert "coarser" not in data["fit"]
    assert "base_cell_factor" not in data["fit"]


def test_a_case_that_does_not_fit_is_told_what_would(tmp_path, advise):
    """The laptop case. 20,000,000 cells against 2 GiB does not fit, and the useful
    output is the cell counts that do, as arithmetic off the one measurement."""
    case = with_mesh(case_with(tmp_path, {}), 20_000_000)
    data = advise.inspect(case, cores=4, memory_gb=2)
    fit = data["fit"]
    assert fit["fits"] is False
    assert fit["holds_cells"][0] < 20_000_000
    assert fit["base_cell_factor"] == pytest.approx(
        (20_000_000 / fit["holds_cells"][0]) ** (1 / 3))
    # 2 GiB over 1,730 B is about 1.24M cells, 20M over that is about 16, and the cube
    # root of 16 is about 2.5.
    assert fit["base_cell_factor"] == pytest.approx(2.52, abs=0.05)
    # `case_with`'s block dict is 40x20x20 over 2x1x1 m, so the shortest side is 20
    # cells deep and one cell thick is a twentieth of the count.
    counts = {label: count for label, count in fit["coarser"]}
    assert 20_000_000 // 8 in counts.values()
    assert 20_000_000 // 20 in counts.values()
    assert any("one cell thick" in label for label in counts)
    report = advise.render(data)
    assert "the case is larger than that" in report
    assert "refinement band goes as the square" in report


def test_no_memory_reading_means_no_memory_claim(tmp_path, advise, monkeypatch):
    """A machine whose memory cannot be read gets the sentence, not a default: a budget
    against an assumed 8 GiB is the fabrication this section exists not to be."""
    monkeypatch.setattr(advise, "machine", lambda: {
        "platform": "nowhere", "cores_logical": 4, "cores_usable": 4,
        "cores_source": "os.cpu_count() says 4", "memory_bytes": None,
        "memory_available_bytes": None,
        "memory_source": "not established on nowhere: no /proc/meminfo",
    })
    case = with_mesh(case_with(tmp_path, {}), 400_000)
    data = advise.inspect(case, cores=4)
    assert data["memory_bytes"] is None
    assert "holds_cells" not in data["fit"]
    assert "fits" not in data["fit"]
    report = advise.render(data)
    assert "not established" in report
    assert "GiB at that rate is about" not in report


# -- what advise.py offers is what this repository actually ships ----------------------
#
# The defect these two are for: `advise.py` recommended 17 solvers across 7 families and
# knew none of the multi-physics this repository ships. The templates name eight
# OpenFOAM solvers; mhdFoam, porousSimpleFoam and chemFoam were in no family at all, so
# a session asked about a conducting duct, a packed bed or a reacting mixture got a list
# that omitted the capability -- with a template and a grader sitting behind it.
#
# Both checks DERIVE their expectation. A literal list is what went stale three times in
# one day in this repository: two index guards and this advisor.

TEMPLATES_DIR = TOOLBOX / "templates"

# Not preceded by `/`, because a path is not an invocation: chem0d.py copies thermo
# data out of `$FOAM_TUTORIALS/combustion/reactingFoam/laminar/...`, and the first
# version of this check read that path as "the template runs reactingFoam". And not
# `*ToFoam`, which is OpenFOAM's own naming for a converter: `chemkinToFoam` and
# `gmshToFoam` are utilities a template legitimately runs, not solvers to recommend.
FOAM_SOLVER = re.compile(r"(?<![/\w])([a-z][A-Za-z]*Foam)\b")


def is_a_solver(name: str) -> bool:
    return not name.endswith("ToFoam")


def solvers_the_templates_name() -> set[str]:
    """Every `*Foam` application the shipped templates and chem0d actually invoke.

    Read from the files rather than listed, because a template is added by writing one
    and nobody adds a row to a test while doing it.
    """
    found: set[str] = set()
    for path in sorted(TEMPLATES_DIR.glob("*.py")) + [TOOLBOX / "chem0d.py"]:
        found |= {name for name in FOAM_SOLVER.findall(path.read_text(encoding="utf-8"))
                  if is_a_solver(name)}
    assert len(found) >= 6, f"only {found} -- the regex or the directory is wrong"
    return found


def test_advise_knows_every_solver_a_shipped_template_runs(advise):
    """A solver this repository writes a case for, and grades, that `advise.py` cannot
    offer is a capability the product has and cannot recommend.

    This failed for mhdFoam, porousSimpleFoam and chemFoam when it was written.
    """
    known = {option["name"] for options in advise.FAMILIES.values() for option in options}
    for solver in sorted(solvers_the_templates_name()):
        assert solver in known, (
            f"{solver} is run by a shipped template and advise.py offers no family "
            f"containing it, so nothing will ever recommend it")


def test_every_family_is_reachable_from_some_problem_class(advise):
    """A family in FAMILIES that no class offers is dead code that reads as coverage.

    `advise` chooses by class, so a family absent from every `CLASS_FAMILIES` value can
    never be printed however right it is.
    """
    offered = {family for families in advise.CLASS_FAMILIES.values() for family in families}
    for family in advise.FAMILIES:
        assert family in offered, (
            f"the {family!r} family is in FAMILIES and no problem class offers it")


def test_every_class_offers_only_families_that_exist(advise):
    """The other direction: a class naming a family that is not there would raise at the
    moment somebody asked, which is the worst time to find out."""
    for problem_class, families in advise.CLASS_FAMILIES.items():
        for family in families:
            assert family in advise.FAMILIES, (
                f"class {problem_class!r} offers {family!r}, which is not in FAMILIES")
