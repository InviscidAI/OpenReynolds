"""Reading a Felix case's `output/` from the toolbox (goal A5).

Fixtures are real output of the Felix solver (one A100, build a60138c7f780,
2026-10-07), with the scratch path rewritten to `/work/cases/<name>`:

- `tests/data/felix/cavity_small/`: a quasi-2D cavity, 8x8x1 hexes, Re 100, 30
  steps with `output.every: 10`, two probes and the force on the lid. The output
  directory is exactly what the run wrote.
- `tests/data/felix/nonfinite/output/summary.json`: the summary of a run stopped on a
  NaN at step 4.
- `tests/data/felix/cylinder_2d1/`: the Schafer-Turek 2D-1 cylinder (Re 20) to a steady
  state, with its forces and probes CSVs as written. Its `solution.vtu` is cut down
  to the inlet and outlet columns and the cells around the cylinder, to keep the
  repository small.

The half that needs pyvista skips where pyvista is absent, as the other toolbox
rendering tests do.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
DATA = Path(__file__).parent / "data" / "felix"
SMALL = DATA / "cavity_small"
CYLINDER = DATA / "cylinder_2d1"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fr():
    return load("felix_results")


def needs_pyvista():
    pytest.importorskip("pyvista", reason="the drawing half runs in the workspace image",
                        exc_type=ImportError)


# -- where things are ------------------------------------------------------------


def test_the_output_directory_defaults_to_case_output(fr, tmp_path):
    assert fr.output_dir(SMALL) == SMALL / "output"
    case = tmp_path / "c"
    case.mkdir()
    (case / "io.yaml").write_text("output:\n  json: true\n  dir: results/run1  # here\n")
    assert fr.output_dir(case) == case / "results" / "run1"


def test_snapshots_are_listed_by_step(fr):
    assert [step for step, _ in fr.snapshots(SMALL)] == [10, 20, 30]


@pytest.mark.parametrize("which, name", [
    ("solution", "solution.vtu"),
    ("latest", "snapshot_000030.vtu"),
    ("20", "snapshot_000020.vtu"),
    ("snapshot_000010", "snapshot_000010.vtu"),
])
def test_a_result_file_is_chosen_by_name_step_or_latest(fr, which, name):
    assert fr.choose_file(SMALL, which).name == name


def test_a_step_with_no_snapshot_names_the_ones_there_are(fr):
    with pytest.raises(SystemExit, match="10, 20, 30"):
        fr.choose_file(SMALL, "15")


# -- summary.json -----------------------------------------------------------------


def test_the_summary_says_how_far_the_run_got_and_what_it_cost(fr):
    text = "\n".join(fr.summary_lines(fr.read_summary(SMALL)))
    assert "step 30" in text
    assert "t = 1.5" in text
    assert "snapshots: 3" in text
    assert "u_centre" in text and "-0.1772" in text
    assert "health: every counter is 0" in text


def test_a_stopped_run_leads_with_its_error_and_last_finite_step(fr):
    text = "\n".join(fr.summary_lines(fr.read_summary(DATA / "nonfinite")))
    first = text.splitlines()[0]
    assert "non-finite" in first
    assert "last finite step 3" in text


def test_non_zero_health_counters_are_named(fr):
    summary = json.loads((SMALL / "output" / "summary.json").read_text())
    summary["health"]["corrector_nonconverged"] = 8
    summary["health"]["cg_nonconverged"] = 2
    text = "\n".join(fr.summary_lines(summary))
    assert "corrector_nonconverged 8" in text and "cg_nonconverged 2" in text
    assert "every counter is 0" not in text


def test_a_ptc_run_reports_whether_it_converged(fr):
    text = "\n".join(fr.summary_lines(fr.read_summary(CYLINDER)))
    assert "PTC converged" in text


def test_a_missing_summary_says_the_run_never_wrote_one(fr, tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(SystemExit, match="no summary.json"):
        fr.read_summary(tmp_path)


# -- CSVs ----------------------------------------------------------------------------


def test_the_forces_csv_is_read_by_header_name_past_its_comments(fr):
    table = fr.read_csv(SMALL / "output" / "forces.csv")
    assert table["columns"] == ["step", "time", "Fx", "Fy", "Fz", "Fx_pres", "Fx_visc"]
    assert len(table["rows"]) == 30
    assert any("lid" in line for line in table["comments"])
    fx = fr.column(table, "Fx")
    assert fx[-1] == pytest.approx(-0.0082856239197643539)


def test_the_probes_csv_starts_at_the_initial_state(fr):
    table = fr.read_csv(SMALL / "output" / "probes.csv")
    assert table["columns"] == ["step", "time", "u_centre_velocity_x", "p_corner_pressure"]
    assert fr.column(table, "step")[0] == 0
    assert len(table["rows"]) == 31


def test_force_coefficients_need_a_reference_and_use_the_tail_mean(fr):
    table = fr.read_csv(CYLINDER / "output" / "forces.csv")
    # Schafer-Turek 2D-1: mean inflow 0.2, D = 0.1, depth 0.02 -> A = 0.002.
    coeffs = fr.force_coefficients(table, uref=0.2, aref=0.1 * 0.02)
    assert coeffs["Cd"]["last"] == pytest.approx(5.52, abs=0.01)
    assert set(coeffs) == {"Cd", "Cl", "Cz"}


def test_forces_lines_report_last_value_and_tail_mean(fr):
    text = "\n".join(fr.forces_lines(fr.read_csv(SMALL / "output" / "forces.csv")))
    assert "Fx: last -0.00828562" in text
    assert "mean of last 25%" in text


def test_probe_and_force_plots_are_written(fr, tmp_path):
    out = fr.plot_table(fr.read_csv(SMALL / "output" / "probes.csv"), tmp_path / "p.png", "probes")
    assert out.stat().st_size > 1000
    out = fr.plot_table(fr.read_csv(SMALL / "output" / "forces.csv"), tmp_path / "f.png", "forces")
    assert out.stat().st_size > 1000


def test_the_log_subcommand_plots_the_step_lines(fr, tmp_path):
    log = DATA / "logs" / "re1000_ptc.stderr"
    series = fr.read_step_lines(log.read_text())
    assert series["step"][-1] == 623
    assert "res0" in series and len(series["res0"]) == len(series["step"])
    out = fr.plot_steps(series, tmp_path / "r.png")
    assert out.stat().st_size > 1000


# -- fields (pyvista) -------------------------------------------------------------------


def test_fields_lists_arrays_with_their_range_and_the_step(fr):
    needs_pyvista()
    mesh = fr.read_vtu(fr.choose_file(SMALL, "solution"))
    info = fr.field_table(mesh)
    assert info["velocity"]["components"] == 3
    assert info["pressure"]["components"] == 1
    assert info["velocity"]["max"] > 0.9  # the lid moves at 1
    assert fr.vtu_step_time(mesh) == (30, pytest.approx(1.5))


def test_a_line_through_the_cavity_samples_the_velocity(fr):
    needs_pyvista()
    mesh = fr.read_vtu(fr.choose_file(SMALL, "solution"))
    s, values = fr.sample_line(mesh, (0.5, 0.0, 0.06), (0.5, 1.0, 0.06), 41, "velocity", "x")
    assert len(s) == len(values) == 41
    assert values[-1] == pytest.approx(1.0, abs=1e-6)  # the lid
    assert values[0] == pytest.approx(0.0, abs=1e-6)  # the floor
    assert np.min(values) < -0.1  # the return flow under the vortex


def test_a_slice_is_rendered_to_png(fr, tmp_path):
    needs_pyvista()
    mesh = fr.read_vtu(fr.choose_file(SMALL, "latest"))
    out = fr.render_slice(mesh, "velocity", "mag", "z", tmp_path / "u.png", title="|u|")
    assert out.stat().st_size > 5000


def test_an_absent_field_names_the_ones_there_are(fr, tmp_path):
    needs_pyvista()
    mesh = fr.read_vtu(fr.choose_file(SMALL, "solution"))
    with pytest.raises(Exception, match="velocity"):
        fr.render_slice(mesh, "temperature", None, "z", tmp_path / "t.png")


# -- the command line -----------------------------------------------------------------


def test_cli_summary_prints_the_run(fr, capsys):
    assert fr.main(["summary", str(SMALL)]) == 0
    assert "step 30" in capsys.readouterr().out


def test_cli_all_writes_the_standard_set(fr, tmp_path, capsys):
    needs_pyvista()
    out = tmp_path / "results"
    assert fr.main(["all", str(SMALL), "--out", str(out), "--normal", "z"]) == 0
    names = {p.name for p in out.iterdir()}
    assert {"velocity_mag.png", "pressure.png", "probes.png", "forces.png"} <= names
    printed = capsys.readouterr().out
    assert "velocity_mag.png" in printed


def test_cli_line_writes_a_csv(fr, tmp_path):
    needs_pyvista()
    csv = tmp_path / "line.csv"
    assert fr.main(["line", str(SMALL), "--start", "0.5", "0", "0.06", "--end", "0.5", "1", "0.06",
                    "--field", "velocity", "--component", "x", "-n", "11", "--csv", str(csv)]) == 0
    lines = csv.read_text().splitlines()
    assert lines[0] == "s,x,y,z,velocity_x"
    assert len(lines) == 12
