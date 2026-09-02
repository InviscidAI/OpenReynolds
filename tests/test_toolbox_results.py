"""The standard results renderer.

Its rendering needs pyvista and so lives in the container, but the half that decides
*what* gets rendered -- the preset table, the postProcessing `.dat` parsing, the
skipped-with-a-reason path and the summary -- is ordinary Python and is tested here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def results():
    return load("results")


@pytest.fixture
def state():
    return load("study_state")


# -- fixtures shaped like the real files ---------------------------------------

FORCE_COEFFS = """\
# Force coefficients
# dragDir           : (1.0000e+00 0.0000e+00 0.0000e+00)
# liftDir           : (0.0000e+00 0.0000e+00 1.0000e+00)
# pitchAxis         : (0.0000e+00 1.0000e+00 0.0000e+00)
# magUInf           : 1.0000e+01
# lRef              : 1.0000e+00
# Aref              : 1.0000e+00
# Time              Cm              Cd              Cl              Cl(f)           Cl(r)
0.1                 1.0e-02         1.20e+00        3.0e-01         1.6e-01         1.4e-01
0.2                 1.1e-02         1.24e+00        3.2e-01         1.7e-01         1.5e-01
0.3                 1.2e-02         1.28e+00        3.4e-01         1.8e-01         1.6e-01
0.4                 1.3e-02         1.32e+00        3.6e-01         1.9e-01         1.7e-01
"""

OLD_FORCES = """\
# Forces
# CofR                : (0 0 0)
# Time forces(pressure viscous porous) moment(pressure viscous porous)
0.1 ((1 2 3) (4 5 6) (0 0 0)) ((7 8 9) (10 11 12) (0 0 0))
0.2 ((2 3 4) (5 6 7) (0 0 0)) ((8 9 10) (11 12 13) (0 0 0))
"""

NEW_FORCES = """\
# Force
# CofR                : (0 0 0)
# Time          (total_x total_y total_z)       (pressure_x pressure_y pressure_z) (viscous_x viscous_y viscous_z)
0.1     (1 2 3) (4 5 6) (7 8 9)
0.2     (2 3 4) (5 6 7) (8 9 10)
"""

SOLVER_LOG = """\
Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1.0e-01, Final residual = 2.0e-03, No Iterations 3
GAMG:  Solving for p, Initial residual = 9.0e-01, Final residual = 1.0e-02, No Iterations 9
time step continuity errors : sum local = 1.5e-06, global = -2.0e-09, cumulative = -3.0e-09
bounding k, min: -0.01 max: 5 average: 1
ExecutionTime = 1.40 s  ClockTime = 1 s

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 5.0e-03, Final residual = 1.0e-04, No Iterations 3
GAMG:  Solving for p, Initial residual = 4.0e-02, Final residual = 5.0e-04, No Iterations 7
time step continuity errors : sum local = 7.5e-07, global = -1.0e-09, cumulative = -4.0e-09
ExecutionTime = 2.80 s  ClockTime = 2 s
"""

CHECK_MESH_LOG = """\
Create polyMesh for time = 0
Mesh stats
    points: 10
Mesh OK.
"""


# -- the preset table ----------------------------------------------------------


def test_the_five_presets_are_all_there(results):
    assert set(results.PRESETS) == {
        "external-flow-2d", "duct-flow", "transient-wake", "vehicle-aero", "mesh-validation",
    }


def test_every_preset_has_outputs_and_a_line_saying_what_it_is_for(results):
    for name, outputs in results.PRESETS.items():
        assert outputs, f"{name} produces nothing"
        assert results.PRESET_ABOUT.get(name), f"{name} has no description"


def test_every_output_declares_a_kind_study_state_knows(results, state):
    for name, outputs in results.PRESETS.items():
        for output in outputs:
            assert output.kind in state.KINDS, f"{name}/{output.name} has kind {output.kind}"


def test_every_output_names_a_producer_that_exists(results):
    for name, outputs in results.PRESETS.items():
        for output in outputs:
            assert output.producer in results.PRODUCERS, f"{name}/{output.name}"


def test_output_names_are_unique_within_a_preset(results):
    """They become filenames in one directory, so a collision is a lost picture."""
    for name, outputs in results.PRESETS.items():
        names = [output.name for output in outputs]
        assert len(names) == len(set(names)), f"{name} repeats an output name"


def test_every_output_says_what_it_is(results):
    for outputs in results.PRESETS.values():
        for output in outputs:
            assert output.about.strip(), f"{output.name} has no description"


def test_the_table_covers_the_whole_advertised_range(results):
    """Between them the presets have to reach every kind the renderer claims."""
    kinds = {output.kind for outputs in results.PRESETS.values() for output in outputs}
    assert kinds >= {
        "pressure", "velocity", "vorticity", "streamlines", "residuals", "forces", "mesh-full",
    }


def test_velocity_appears_as_magnitude_and_as_a_component(results):
    options = [
        output.options
        for outputs in results.PRESETS.values()
        for output in outputs
        if output.options.get("field") == "U"
    ]
    assert any(option.get("component") == "mag" for option in options)
    assert any(option.get("component") in ("x", "y", "z") for option in options)


def test_mesh_validation_asks_nothing_of_a_solution(results):
    """It runs before a solve exists, so nothing in it may need one."""
    producers = {output.producer for output in results.PRESETS["mesh-validation"]}
    assert producers <= {"mesh-cut", "mesh-quality"}


def test_an_unknown_preset_names_the_ones_that_exist(results):
    with pytest.raises(SystemExit) as excinfo:
        results.preset_outputs("no-such-preset")
    assert "external-flow-2d" in str(excinfo.value)


def test_list_prints_every_preset_and_every_output(results, capsys):
    assert results.main(["--list"]) == 0
    printed = capsys.readouterr().out
    for name, outputs in results.PRESETS.items():
        assert name in printed
        for output in outputs:
            assert output.name in printed


def test_what_it_prints_does_not_instruct(results):
    """The toolbox is offered, not imposed: no script tells the model what to do next."""
    text = results.describe_presets().lower()
    for imperative in ("you must", "you should", "always run", "before you", "step 1"):
        assert imperative not in text


# -- header parsing ------------------------------------------------------------


def test_header_names_keep_the_parenthesised_split_columns(results):
    names = results.split_header_names(
        "# Time              Cm              Cd              Cl              Cl(f)           Cl(r)"
    )
    assert names == ["Time", "Cm", "Cd", "Cl", "Cl_f", "Cl_r"]


def test_header_names_expand_a_prefixed_group(results):
    names = results.split_header_names("# Time forces(pressure viscous porous)")
    assert names == ["Time", "forces_pressure", "forces_viscous", "forces_porous"]


def test_header_names_take_a_bare_group_as_it_is(results):
    names = results.split_header_names("# Time (total_x total_y total_z)")
    assert names == ["Time", "total_x", "total_y", "total_z"]


def test_align_columns_expands_vector_groups_to_components(results):
    columns = results.align_columns(["Time", "forces_pressure", "forces_viscous"], 7)
    assert columns == [
        "Time",
        "forces_pressure_x", "forces_pressure_y", "forces_pressure_z",
        "forces_viscous_x", "forces_viscous_y", "forces_viscous_z",
    ]


def test_align_columns_expands_a_lone_vector_column_too(results):
    assert results.align_columns(["Time", "total"], 4) == [
        "Time", "total_x", "total_y", "total_z",
    ]


def test_align_columns_pads_and_trims_rather_than_giving_up(results):
    """A count that divides into nothing sensible still leaves the data readable."""
    assert results.align_columns(["Time", "Cd", "Cl"], 6) == [
        "Time", "Cd", "Cl", "c3", "c4", "c5",
    ]
    assert results.align_columns(["Time", "Cd", "Cl"], 2) == ["Time", "Cd"]
    assert results.align_columns([], 3) == ["Time", "c1", "c2"]


# -- .dat parsing --------------------------------------------------------------


def test_force_coefficients_parse_to_named_columns(results):
    data = results.parse_dat(FORCE_COEFFS)
    assert data["columns"] == ["Time", "Cm", "Cd", "Cl", "Cl_f", "Cl_r"]
    assert len(data["rows"]) == 4
    assert data["rows"][-1][2] == pytest.approx(1.32)


def test_the_header_key_value_lines_become_metadata_not_columns(results):
    data = results.parse_dat(FORCE_COEFFS)
    assert data["meta"]["magUInf"] == "1.0000e+01"
    assert data["meta"]["Aref"] == "1.0000e+00"
    assert "dragDir" not in data["columns"]
    assert "magUInf" not in data["columns"]


def test_the_old_forces_format_with_vector_columns(results):
    data = results.parse_dat(OLD_FORCES)
    assert len(data["columns"]) == 19
    assert data["columns"][1:4] == ["forces_pressure_x", "forces_pressure_y", "forces_pressure_z"]
    assert len(data["rows"][0]) == 19
    assert results.column(data, "forces_viscous_y").tolist() == [5.0, 6.0]


def test_the_newer_forces_format_needs_no_expansion(results):
    data = results.parse_dat(NEW_FORCES)
    assert data["columns"] == [
        "Time", "total_x", "total_y", "total_z",
        "pressure_x", "pressure_y", "pressure_z",
        "viscous_x", "viscous_y", "viscous_z",
    ]
    assert results.column(data, "total_x").tolist() == [1.0, 2.0]


def test_a_truncated_final_row_is_dropped_not_misread(results):
    """A job killed mid-write leaves half a line; misreading it shifts every column."""
    data = results.parse_dat(FORCE_COEFFS + "0.5   1.4e-02   1.3")
    assert len(data["rows"]) == 4
    assert results.column(data, "Cd").tolist() == [1.20, 1.24, 1.28, 1.32]


def test_a_file_with_no_header_still_yields_addressable_columns(results):
    data = results.parse_dat("0.1 1.0 2.0\n0.2 1.1 2.1\n")
    assert data["columns"] == ["Time", "c1", "c2"]
    assert len(data["rows"]) == 2


def test_a_header_with_no_rows_yet_is_not_an_error(results):
    data = results.parse_dat("# Time Cd Cl\n")
    assert data["rows"] == []
    assert data["columns"] == []


# -- finding and merging the force files ---------------------------------------


def write_force_files(case: Path) -> Path:
    first = case / "postProcessing" / "forces" / "0"
    first.mkdir(parents=True)
    (first / "forceCoeffs.dat").write_text(FORCE_COEFFS)
    return case


def test_force_files_are_found_under_post_processing(tmp_path, results):
    case = write_force_files(tmp_path / "case")
    grouped = results.find_force_files(case)
    assert list(grouped) == ["forces/forceCoeffs.dat"]
    assert grouped["forces/forceCoeffs.dat"][0].name == "forceCoeffs.dat"


def test_a_case_with_no_post_processing_yields_nothing(tmp_path, results):
    (tmp_path / "case").mkdir()
    assert results.find_force_files(tmp_path / "case") == {}


def test_a_restart_directory_is_merged_in_time_order(tmp_path, results):
    """Two time directories of one function object are one history, not two."""
    case = write_force_files(tmp_path / "case")
    restart = case / "postProcessing" / "forces" / "0.3"
    restart.mkdir(parents=True)
    (restart / "forceCoeffs.dat").write_text(
        FORCE_COEFFS.split("# Time")[0]
        + "# Time              Cm              Cd              Cl              Cl(f)           Cl(r)\n"
        "0.3                 1.2e-02         9.90e+00        3.4e-01         1.8e-01         1.6e-01\n"
        "0.4                 1.3e-02         9.91e+00        3.6e-01         1.9e-01         1.7e-01\n"
        "0.5                 1.4e-02         9.92e+00        3.8e-01         2.0e-01         1.8e-01\n"
    )
    grouped = results.find_force_files(case)
    history = results.read_history(grouped["forces/forceCoeffs.dat"])

    assert results.column(history, "Time").tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    # The restart wrote 0.3 and 0.4 again; the later file is the run still on disk.
    assert results.column(history, "Cd").tolist() == [1.20, 1.24, 9.90, 9.91, 9.92]
    assert len(history["sources"]) == 2


def test_coefficients_are_preferred_over_raw_forces(results):
    grouped = {"forces/forces.dat": [], "forces/forceCoeffs.dat": [], "forces/moment.dat": []}
    assert results.choose_force_series(grouped) == "forces/forceCoeffs.dat"


def test_choosing_from_nothing_is_a_reason_not_a_crash(results):
    with pytest.raises(results.Unavailable) as excinfo:
        results.choose_force_series({})
    assert "postProcessing" in str(excinfo.value)


def test_plot_columns_picks_the_whole_body_coefficients(results):
    chosen = results.plot_columns(["Time", "Cm", "Cd", "Cl", "Cl_f", "Cl_r"])
    assert chosen == ["Cm", "Cd", "Cl"]


def test_plot_columns_falls_back_to_the_totals_then_to_pressure(results):
    assert results.plot_columns(["Time", "total_x", "total_y", "pressure_x"]) == [
        "total_x", "total_y",
    ]
    assert results.plot_columns(["Time", "forces_pressure_x", "forces_viscous_x"]) == [
        "forces_pressure_x"
    ]


def test_the_headline_reports_the_last_value_and_the_tail_mean(results):
    history = results.parse_dat(FORCE_COEFFS)
    notes = results.force_headline(history, ["Cd"], fraction=0.5)
    assert notes["Cd (last)"] == "1.32"
    assert float(notes["Cd (mean of last 50%)"]) == pytest.approx(1.30)
    assert notes["magUInf"] == "1.0000e+01"


def test_tail_mean_of_a_short_or_empty_series(results):
    assert results.tail_mean(np.array([1.0, 3.0]), 0.5) == pytest.approx(3.0)
    assert np.isnan(results.tail_mean(np.array([])))


# -- the solver log ------------------------------------------------------------


def test_the_solver_log_is_the_one_with_residuals_in_it(tmp_path, results):
    case = tmp_path / "case"
    case.mkdir()
    (case / "log.blockMesh").write_text("Create polyMesh\nEnd\n")
    (case / "log.checkMesh").write_text(CHECK_MESH_LOG)
    (case / "log.simpleFoam").write_text(SOLVER_LOG)
    assert results.find_solver_log(case).name == "log.simpleFoam"


def test_a_case_with_only_a_mesh_log_has_no_solver_log(tmp_path, results):
    case = tmp_path / "case"
    case.mkdir()
    (case / "log.checkMesh").write_text(CHECK_MESH_LOG)
    assert results.find_solver_log(case) is None


def test_a_log_in_a_logs_subdirectory_is_found(tmp_path, results):
    case = tmp_path / "case"
    (case / "logs").mkdir(parents=True)
    (case / "logs" / "log.pimpleFoam").write_text(SOLVER_LOG)
    assert results.find_solver_log(case).name == "log.pimpleFoam"


# -- the numbers behind the pictures -------------------------------------------


def test_resolve_time_defaults_to_the_latest_write(results):
    value, note = results.resolve_time("latest", [0.0, 1.0, 2.5])
    assert value == 2.5 and "latest" in note


def test_resolve_time_takes_first_and_the_nearest_number(results):
    assert results.resolve_time("first", [0.0, 1.0])[0] == 0.0
    value, note = results.resolve_time(1.2, [0.0, 1.0, 2.0])
    assert value == 1.0 and "nearest" in note


def test_resolve_time_on_a_case_that_wrote_nothing(results):
    value, note = results.resolve_time("latest", [])
    assert value is None and "no write times" in note


def test_resolve_time_rejects_a_word_it_does_not_know(results):
    with pytest.raises(SystemExit):
        results.resolve_time("halfway", [0.0, 1.0])


def test_the_colour_range_ignores_a_single_spike(results):
    values = np.concatenate([np.linspace(0.0, 1.0, 999), [1e6]])
    low, high = results.robust_clim(values)
    assert high < 10.0


def test_a_signed_field_gets_a_range_centred_on_zero(results):
    low, high = results.symmetric_clim(np.linspace(-2.0, 8.0, 500))
    assert low == pytest.approx(-high)


def test_a_flat_field_still_gets_a_usable_range(results):
    low, high = results.robust_clim(np.full(50, 3.0))
    assert low < high


# -- the plots that need no case data ------------------------------------------


def test_the_force_plot_is_written(tmp_path, results):
    history = results.parse_dat(FORCE_COEFFS)
    out = results.force_plot(history, ["Cd", "Cl"], tmp_path / "forces.png", title="forceCoeffs")
    assert out.exists() and out.stat().st_size > 1000


def test_the_quality_histograms_are_written(tmp_path, results):
    measures = {
        "cell volume": np.logspace(-9, -3, 500),
        "aspect ratio": np.linspace(1.0, 12.0, 500),
    }
    out = results.quality_histograms(measures, tmp_path / "quality.png", title="mesh")
    assert out.exists() and out.stat().st_size > 1000


def test_quality_histograms_with_nothing_to_draw_is_a_reason(tmp_path, results):
    with pytest.raises(results.Unavailable):
        results.quality_histograms({"aspect ratio": np.array([])}, tmp_path / "q.png")


# -- running outputs -----------------------------------------------------------


class FakeContext:
    """Everything a producer is given, without a case behind it."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.case = Path("case")
        self.case_name = "case"
        self.normal = "z"
        self.time_label = "10"
        self.recorded: list[tuple] = []

    def record(self, kind, path, label=""):
        self.recorded.append((kind, Path(path), label))


def test_a_missing_input_is_a_skipped_row_carrying_its_reason(tmp_path, results):
    spec = results.Output("forces", "forces", "forces", "history")

    def refuse(ctx, spec):
        raise results.Unavailable("no forces were logged")

    outcome = results.run_output(FakeContext(tmp_path), spec, {"forces": refuse})

    assert outcome.status == "skipped"
    assert outcome.reason == "no forces were logged"
    assert outcome.path is None


def test_an_unexpected_error_is_a_failed_row_and_nothing_more(tmp_path, results):
    spec = results.Output("pressure", "pressure", "field", "p")

    def explode(ctx, spec):
        raise ZeroDivisionError("division by zero")

    outcome = results.run_output(FakeContext(tmp_path), spec, {"field": explode})

    assert outcome.status == "failed"
    assert "ZeroDivisionError" in outcome.reason


def test_an_output_naming_a_producer_that_is_gone_is_skipped(tmp_path, results):
    spec = results.Output("odd", "other", "not-a-producer", "?")
    outcome = results.run_output(FakeContext(tmp_path), spec, {})
    assert outcome.status == "skipped" and "not-a-producer" in outcome.reason


def test_a_produced_output_is_registered_under_its_kind(tmp_path, results):
    spec = results.Output("vorticity", "vorticity", "vorticity", "the wake")
    context = FakeContext(tmp_path)

    def draw(ctx, spec):
        path = ctx.out_dir / f"{spec.name}.png"
        path.write_text("png")
        return path, {"range": "-3 to 3"}

    outcome = results.run_output(context, spec, {"vorticity": draw})

    assert outcome.status == "produced"
    assert outcome.notes == {"range": "-3 to 3"}
    assert context.recorded == [("vorticity", tmp_path / "vorticity.png", "the wake")]


def test_one_skipped_output_does_not_cost_the_others(tmp_path, results):
    context = FakeContext(tmp_path)

    def draw(ctx, spec):
        path = ctx.out_dir / f"{spec.name}.png"
        path.write_text("png")
        return path, {}

    def refuse(ctx, spec):
        raise results.Unavailable("the case never wrote p")

    producers = dict.fromkeys(results.PRODUCERS, draw)
    producers["field"] = refuse
    outcomes = results.run_preset(context, "external-flow-2d", producers)

    statuses = {outcome.spec.name: outcome.status for outcome in outcomes}
    assert len(outcomes) == len(results.PRESETS["external-flow-2d"])
    assert statuses["pressure"] == "skipped"
    assert statuses["vorticity"] == "produced"
    assert statuses["forces"] == "produced"


def test_only_narrows_the_run_to_named_outputs(tmp_path, results):
    def draw(ctx, spec):
        path = ctx.out_dir / f"{spec.name}.png"
        path.write_text("png")
        return path, {}

    outcomes = results.run_preset(
        FakeContext(tmp_path), "external-flow-2d",
        dict.fromkeys(results.PRODUCERS, draw), only=["residuals"],
    )
    assert [outcome.spec.name for outcome in outcomes] == ["residuals"]


def test_a_case_that_cannot_be_read_is_opened_only_once(tmp_path, results, monkeypatch):
    """Six outputs asking for a mesh that is not there is one failed read, not six."""
    calls = []

    def refuse(case, requested):
        calls.append(case)
        raise results.Unavailable("no internalMesh in this case")

    monkeypatch.setattr(results, "open_case", refuse)
    context = results.Context(tmp_path, tmp_path / "results")
    for _ in range(3):
        with pytest.raises(results.Unavailable):
            context.mesh()
    assert len(calls) == 1


def test_the_phase_status_reflects_what_actually_happened(results):
    spec = results.Output("a", "other", "field", "x")
    assert results.phase_result([results.Outcome(spec, "produced")])[0] == "done"
    assert results.phase_result([results.Outcome(spec, "skipped")])[0] == "skipped"
    assert results.phase_result([results.Outcome(spec, "failed")])[0] == "failed"
    mixed = [results.Outcome(spec, "produced"), results.Outcome(spec, "failed")]
    assert results.phase_result(mixed)[0] == "done"


# -- the summary ---------------------------------------------------------------


def test_the_summary_lists_what_was_made_and_why_the_rest_was_not(tmp_path, results):
    made = results.Outcome(
        results.Output("pressure", "pressure", "field", "static pressure"),
        "produced", path=tmp_path / "results" / "pressure.png", notes={"p range": "-4 to 9"},
    )
    missed = results.Outcome(
        results.Output("forces", "forces", "forces", "coefficients"),
        "skipped", reason="no force or coefficient .dat under postProcessing/",
    )
    broke = results.Outcome(
        results.Output("streamlines", "streamlines", "streamlines", "lines"),
        "failed", reason="RuntimeError: VTK said no",
    )

    text = results.summary_markdown(
        "external-flow-2d", [made, missed, broke],
        case="cylinder", time_label="200", time_note="latest of 41 write times",
        root=tmp_path, out_dir=tmp_path / "results",
    )

    assert text.startswith("# results -- external-flow-2d")
    assert "cylinder" in text and "200" in text
    assert "results/pressure.png" in text
    assert "no force or coefficient .dat under postProcessing/" in text
    assert "VTK said no" in text
    assert "p range: -4 to 9" in text
    assert "1 produced, 1 skipped, 1 failed" in text


def test_the_summary_of_a_run_that_produced_nothing_still_says_why(tmp_path, results):
    missed = results.Outcome(
        results.Output("residuals", "residuals", "residuals", "residuals"),
        "skipped", reason="no log with 'Solving for' lines",
    )
    text = results.summary_markdown("duct-flow", [missed], case="duct")
    assert "## produced" not in text
    assert "no log with 'Solving for' lines" in text


def test_the_summary_offers_no_verdict(tmp_path, results):
    """The renderer reports numbers; whether they are converged is the model's call."""
    made = results.Outcome(
        results.Output("forces", "forces", "forces", "coefficients"),
        "produced", path=tmp_path / "forces.png", notes={"Cd (last)": "1.32"},
    )
    text = results.summary_markdown("external-flow-2d", [made], case="cylinder").lower()
    for verdict in ("is converged", "looks good", "acceptable", "mesh independent."):
        assert verdict not in text


# -- end to end on a case with logs but no mesh --------------------------------


def test_a_case_with_only_logs_still_gets_the_plots_it_can(tmp_path, results, state):
    """The half of a preset that reads text does not depend on the half that renders.

    A case that has not been reconstructed (or a machine with no pyvista, which is
    this one) can still produce residuals and forces, and the outputs that need the
    mesh come back as rows with reasons rather than taking the run down.
    """
    study = tmp_path / "study"
    (study / ".reynolds").mkdir(parents=True)
    case = study / "cylinder"
    case.mkdir()
    (case / "log.simpleFoam").write_text(SOLVER_LOG)
    write_force_files(case)

    exit_code = results.main([str(case), "--preset", "external-flow-2d"])

    assert exit_code == 0
    out_dir = case / "results"
    assert (out_dir / "residuals.png").exists()
    assert (out_dir / "forces.png").exists()

    summary = (out_dir / "results.md").read_text(encoding="utf-8")
    assert "Cd (last)" in summary
    assert "final residual p" in summary

    rows = [json.loads(line) for line in
            (study / ".reynolds" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {row["kind"] for row in rows}
    assert {"residuals", "forces", "report"} <= kinds
    assert all(row["case"] == "cylinder" for row in rows)

    phases = json.loads((study / ".reynolds" / "phases.json").read_text(encoding="utf-8"))
    render = [row for row in phases["phases"] if row["name"] == "render"][0]
    assert render["status"] == "done"
    assert "external-flow-2d" in render["note"]


def test_no_state_writes_pictures_but_no_manifest(tmp_path, results):
    study = tmp_path / "study"
    (study / ".reynolds").mkdir(parents=True)
    case = study / "cylinder"
    case.mkdir()
    (case / "log.simpleFoam").write_text(SOLVER_LOG)

    results.main([str(case), "--preset", "duct-flow", "--no-state"])

    assert (case / "results" / "residuals.png").exists()
    assert not (study / ".reynolds" / "manifest.jsonl").exists()


def test_a_case_directory_that_is_not_there_is_refused(tmp_path, results):
    with pytest.raises(SystemExit):
        results.main([str(tmp_path / "nope"), "--preset", "duct-flow"])


def test_only_naming_something_the_preset_does_not_produce_is_refused(tmp_path, results):
    """A typo in --only would otherwise leave an empty results directory that reads
    as 'the case had none of it' rather than as a mistyped argument."""
    case = tmp_path / "case"
    case.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        results.main([str(case), "--preset", "duct-flow", "--only", "presure"])
    message = str(excinfo.value)
    assert "presure" in message and "pressure" in message


def test_a_log_path_that_is_not_there_is_refused_rather_than_ignored(tmp_path, results):
    """Falling back to the search would blame the case for a wrong argument."""
    case = tmp_path / "case"
    case.mkdir()
    with pytest.raises(SystemExit):
        results.main([str(case), "--preset", "duct-flow", "--log", str(tmp_path / "nope.log")])


def test_the_camera_is_aimed_down_the_slice_normal(results):
    """`camera_position = 'z'` is not a thing pyvista accepts -- its string form
    names a plane -- so the direction is given as a vector."""

    class FakePlotter:
        def __init__(self):
            self.vectors = []

        def view_vector(self, vector, viewup=None):
            self.vectors.append((tuple(vector), tuple(viewup) if viewup else None))

    for normal, expected in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        plotter = FakePlotter()
        results.aim(plotter, normal)
        assert plotter.vectors[0][0] == expected


def test_the_camera_is_told_which_way_is_up(results):
    """Found by looking at a real render rather than by reading the code: with no up
    vector VTK picks one, and for a z-normal slice it picked +x -- so every picture of
    a left-to-right flow came out a quarter turn round, and a cylinder wake read as
    though the flow went upward. Physics right, orientation wrong, which is the worse
    of the two: a wrong number gets checked and a wrong orientation gets believed."""

    class FakePlotter:
        def __init__(self):
            self.calls = []

        def view_vector(self, vector, viewup=None):
            self.calls.append((tuple(vector), tuple(viewup) if viewup is not None else None))

    plotter = FakePlotter()
    results.aim(plotter, "z")

    # Down z, with y up, so x -- the streamwise axis of every external case -- is across.
    assert plotter.calls == [((0, 0, 1), (0.0, 1.0, 0.0))]

    for normal in ("x", "y"):
        plotter = FakePlotter()
        results.aim(plotter, normal)
        assert plotter.calls[0][1] is not None, f"{normal} was left to VTK to guess"


def test_no_renderer_sets_a_camera_from_a_bare_axis_letter(results):
    source = (TOOLBOX / "results.py").read_text(encoding="utf-8")
    assert "camera_position = normal" not in source
