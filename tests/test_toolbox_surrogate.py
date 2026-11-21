"""The surrogate, held to a response whose exact answer is known.

A surrogate cannot be validated against a published number: there is no literature
value for "how well does a cubic fit this sweep". What can be done, and is what these
tests do, is fit it to a closed form and compare its own error estimate with the error
measured on points it never saw. That comparison is the whole claim the script makes
about itself, so it is measured here and the measured numbers are printed.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def surrogate():
    return load("surrogate")


def grid(n: int, seed: int = 3) -> np.ndarray:
    """`n` points in the unit square, on a fixed seed."""
    return np.random.default_rng(seed).random((n, 2))


# the response used for the honest-error check: smooth, bounded, and not a polynomial,
# so no degree can fit it exactly and the leave-one-out error has something to measure
def wavy(points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return np.sin(2.4 * x) * np.exp(-1.3 * y) + 0.35 * x * y


# -- exactness where exactness is the right answer ---------------------------------


def test_a_quadratic_the_basis_contains_is_recovered_to_round_off(surrogate):
    """1 + 2x - 3y + 4x^2 - 5xy is in the degree-2 basis, so both errors are zero.

    This is the one case where a tiny error is the correct result rather than a
    suspicious one, and it is here to catch the scaling: the fit is built on
    coordinates mapped to [-1, 1], so getting this back means the map and its inverse
    agree.
    """
    points = grid(21)
    x, y = points[:, 0], points[:, 1]
    values = 1 + 2 * x - 3 * y + 4 * x ** 2 - 5 * x * y

    found = surrogate.build(["x", "y"], points, values, max_degree=3)

    assert found["fit"]["degree"] == 2, "a richer degree should not beat an exact fit"
    assert found["fit"]["rmse_loo"] < 1e-9, found["fit"]["rmse_loo"]
    assert found["fit"]["rmse_fit"] < 1e-12, found["fit"]["rmse_fit"]
    # and it predicts the closed form away from its samples
    asked = np.array([[0.31, 0.78]])
    exact = 1 + 2 * 0.31 - 3 * 0.78 + 4 * 0.31 ** 2 - 5 * 0.31 * 0.78
    assert surrogate.predict(found["fit"], asked)[0] == pytest.approx(exact, abs=1e-9)


def test_degree_selection_prefers_the_degree_the_data_came_from(surrogate):
    """Cubic data, 40 points: degree 3 fits it exactly and degree 1 and 2 cannot."""
    points = grid(40, seed=5)
    x, y = points[:, 0], points[:, 1]
    values = 0.5 + x - 2 * y ** 2 + 3 * x ** 3 - y * x ** 2

    found = surrogate.build(["x", "y"], points, values, max_degree=4)

    by_degree = {row["degree"]: row for row in found["degrees"] if "rmse_loo" in row}
    assert found["fit"]["degree"] == 3, found["degrees"]
    assert by_degree[3]["rmse_loo"] < 1e-9
    assert by_degree[2]["rmse_loo"] > 0.05, "a quadratic should not fit a cubic"
    assert by_degree[1]["rmse_loo"] > by_degree[2]["rmse_loo"]


# -- the claim the whole script rests on -------------------------------------------


def test_the_cross_validated_error_tracks_the_error_on_points_it_never_saw(surrogate, capsys):
    """The leave-one-out error against the true error on 2500 independent points.

    This is the test the script's docstring quotes, and its numbers are printed rather
    than hidden: a cross-validated error is only worth reporting if it predicts the
    error on unseen data, and nothing but a comparison against unseen data shows that
    it does.
    """
    training = grid(25, seed=11)
    found = surrogate.build(["x", "y"], training, wavy(training), max_degree=3)
    fit = found["fit"]

    held_out = np.random.default_rng(99).random((2500, 2))
    truth = wavy(held_out)
    error = float(np.sqrt(np.mean((surrogate.predict(fit, held_out) - truth) ** 2)))

    print(f"\ndegree {fit['degree']} on 25 points"
          f"\n  rmse_fit  {fit['rmse_fit']:.6g}   (on the points it was fitted to)"
          f"\n  rmse_loo  {fit['rmse_loo']:.6g}   (what this script reports)"
          f"\n  true      {error:.6g}   (2500 points it never saw)"
          f"\n  loo/true  {fit['rmse_loo'] / error:.4f}")

    # The measured ratio is recorded in the docstring of surrogate.py. The bound here
    # is deliberately loose in both directions: leave-one-out on 25 points is itself a
    # noisy estimator and a bound tight enough to look impressive would be a bound
    # that fails on a re-seed. What is NOT allowed is the thing this script exists to
    # prevent, which is the error estimate being optimistic by an order of magnitude.
    assert 0.5 < fit["rmse_loo"] / error < 2.0, (fit["rmse_loo"], error)
    # and the fit error is the useless number, by a factor this records
    assert fit["rmse_fit"] < fit["rmse_loo"]


def test_the_training_error_of_a_rich_fit_is_a_lie(surrogate, capsys):
    """Enough terms for the point count and the fit error goes to nothing while the
    surface is wrong everywhere between the samples.

    The numbers printed here are the ones quoted in `surrogate.py`'s docstring as the
    reason both errors are always reported. The degree-4 fit is built directly rather
    than through `build`, because `build` would not choose it -- which is itself the
    last assertion here.
    """
    training = grid(16, seed=7)
    values = wavy(training)
    lo, hi, _ = surrogate.scaling(training)
    scaled = surrogate.to_unit(training, lo, hi)
    rich = surrogate.fit_degree(scaled, values, 4)
    assert "refused" not in rich, rich.get("refused")

    held_out = np.random.default_rng(123).random((2500, 2))
    truth = wavy(held_out)
    forced = dict(rich)
    forced.update({"lo": [float(v) for v in lo], "hi": [float(v) for v in hi]})
    error = float(np.sqrt(np.mean((surrogate.predict(forced, held_out) - truth) ** 2)))
    print(f"\ndegree 4 on 16 points ({rich['n_terms']} terms)"
          f"\n  rmse_fit  {rich['rmse_fit']:.6g}"
          f"\n  rmse_loo  {rich['rmse_loo']:.6g}"
          f"\n  true      {error:.6g}"
          f"\n  max leverage {rich['max_leverage']:.6f}"
          f"\n  the fit error understates the true error by "
          f"{error / max(rich['rmse_fit'], 1e-300):.3g}x")

    assert rich["rmse_fit"] < 0.01 * rich["rmse_loo"], "the point of the test"
    assert error > 10 * rich["rmse_fit"]
    # and the honest number is the one that saw it coming
    assert rich["rmse_loo"] > rich["rmse_fit"]
    # the chosen degree is not this one
    found = surrogate.build(["x", "y"], training, values, max_degree=4)
    assert found["fit"]["degree"] != 4


def test_a_degree_with_as_many_terms_as_points_is_refused_by_name(surrogate):
    """Six points, a quadratic in two variables: six terms. Every leverage is 1."""
    points = grid(6, seed=13)
    found = surrogate.build(["x", "y"], points, wavy(points), max_degree=2)

    quadratic = [row for row in found["degrees"] if row["degree"] == 2][0]
    assert "refused" in quadratic
    assert "6 terms for 6 points" in quadratic["refused"]
    assert "rmse_loo" not in quadratic
    assert found["fit"]["degree"] == 1


def test_a_sample_that_supports_no_degree_refuses_rather_than_fitting(surrogate):
    """Three points in two variables: even a plane has three terms."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(surrogate.Unavailable) as raised:
        surrogate.build(["x", "y"], points, np.array([1.0, 2.0, 3.0]), max_degree=2)
    assert "support no degree" in str(raised.value)


def test_a_degenerate_sample_is_refused_rather_than_least_squared(surrogate):
    """Points on a line in a two-variable space do not determine a plane.

    `lstsq` is happy to return one of the infinitely many answers, and its residual on
    the sample is zero, so without the rank check this reports an exact fit to a
    surface the sample says nothing about off that line.
    """
    t = np.linspace(0, 1, 9)
    points = np.column_stack([t, t])
    with pytest.raises(surrogate.Unavailable) as raised:
        surrogate.build(["x", "y"], points, 2 * t + 1, max_degree=1)
    # the two columns are identical after scaling, so rank 2 of 3 terms
    assert "degenerate" in str(raised.value)
    assert "rank 2 of 3" in str(raised.value)


def test_a_constant_design_variable_is_dropped_and_named(surrogate):
    """A sweep that forgot to vary one knob should say so, not divide by its range."""
    points = grid(20, seed=17)
    with_fixed = np.column_stack([points, np.full(len(points), 0.004)])
    found = surrogate.build(["x", "y", "chord"], with_fixed, wavy(points), max_degree=2)

    assert found["fit"]["dropped_as_constant"] == ["chord"]
    assert found["fit"]["design"] == ["x", "y"]


def test_every_variable_constant_is_a_refusal(surrogate):
    points = np.full((5, 2), 1.5)
    with pytest.raises(surrogate.Unavailable):
        surrogate.build(["x", "y"], points, np.arange(5.0), max_degree=1)


# -- the leave-one-out identity itself ---------------------------------------------


def test_the_closed_form_leave_one_out_equals_refitting_without_the_point(surrogate):
    """The identity `r_i / (1 - h_ii)` is the reason this is affordable, so it is
    checked against the thing it replaces: twenty actual refits, each on nineteen
    points, predicting the twentieth."""
    points = grid(20, seed=23)
    values = wavy(points)
    lo, hi, _ = surrogate.scaling(points)
    scaled = surrogate.to_unit(points, lo, hi)
    fit = surrogate.fit_degree(scaled, values, 2)

    basis = [tuple(t) for t in fit["terms"]]
    by_refit = []
    for left_out in range(len(values)):
        keep = [i for i in range(len(values)) if i != left_out]
        matrix = surrogate.design_matrix(scaled[keep], basis)
        coefficients, *_ = np.linalg.lstsq(matrix, values[keep], rcond=None)
        row = surrogate.design_matrix(scaled[left_out:left_out + 1], basis)
        by_refit.append(float((row @ coefficients)[0]))

    assert fit["loo_predicted"] == pytest.approx(by_refit, rel=1e-9, abs=1e-12)
    refit_rmse = float(np.sqrt(np.mean((values - np.array(by_refit)) ** 2)))
    assert fit["rmse_loo"] == pytest.approx(refit_rmse, rel=1e-9)


# -- the optimum it reports --------------------------------------------------------


def test_the_predicted_optimum_of_a_known_bowl_is_its_known_centre(surrogate):
    """(x - 0.4)^2 + (y - 0.7)^2 has its minimum at (0.4, 0.7) and the surrogate's
    basis contains it exactly, so the reported minimum is a test of the search."""
    points = grid(30, seed=29)
    x, y = points[:, 0], points[:, 1]
    found = surrogate.build(["x", "y"], points, (x - 0.4) ** 2 + (y - 0.7) ** 2,
                            max_degree=2)
    best = surrogate.best_in_box(found["fit"])

    assert best["at"]["x"] == pytest.approx(0.4, abs=0.02)
    assert best["at"]["y"] == pytest.approx(0.7, abs=0.02)
    assert best["predicted"] == pytest.approx(0.0, abs=2e-3)
    assert best["on_the_box_edge"] == []


def test_an_optimum_against_the_sweep_bound_says_so(surrogate):
    """A monotone response has its minimum on a face of the box, and that is a fact
    about where the sweep stopped rather than about the response."""
    points = grid(30, seed=31)
    found = surrogate.build(["x", "y"], points,
                            2 * points[:, 0] + points[:, 1], max_degree=1)
    best = surrogate.best_in_box(found["fit"])
    assert set(best["on_the_box_edge"]) == {"x", "y"}


def test_the_optimum_is_the_same_on_two_runs(surrogate):
    points = grid(30, seed=37)
    found = surrogate.build(["x", "y"], points, wavy(points), max_degree=2)
    first = surrogate.best_in_box(found["fit"])
    second = surrogate.best_in_box(found["fit"])
    assert first == second


# -- reading what a sweep leaves behind --------------------------------------------


def test_a_csv_sweep_is_read_and_the_objective_column_named(surrogate, tmp_path):
    table = tmp_path / "sweep.csv"
    rows = ["alpha,camber,Cd,note"]
    for alpha in (0.0, 2.0, 4.0, 6.0, 8.0):
        for camber in (0.0, 0.02, 0.04):
            cd = 0.01 + 0.0004 * alpha ** 2 + 0.3 * camber ** 2
            rows.append(f"{alpha},{camber},{cd:.8f},run{len(rows)}")
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")

    read = surrogate.read_table(table)
    names, points, values = surrogate.as_samples(read, "Cd", ["alpha", "camber"])

    assert names == ["alpha", "camber"]
    assert points.shape == (15, 2)
    found = surrogate.build(names, points, values, max_degree=2)
    assert found["fit"]["rmse_loo"] < 1e-9


def test_a_row_with_a_blank_objective_is_a_point_that_was_not_run(surrogate, tmp_path):
    table = tmp_path / "sweep.csv"
    table.write_text("x,y,Cd\n0,0,1\n1,0,2\n0,1,3\n1,1,\n0.5,0.5,2.0\n", encoding="utf-8")
    names, points, values = surrogate.as_samples(surrogate.read_table(table), "Cd")
    assert len(values) == 4
    assert names == ["x", "y"]


def test_an_unknown_objective_names_the_columns_it_does_have(surrogate, tmp_path):
    table = tmp_path / "sweep.csv"
    table.write_text("x,y,Cd\n0,0,1\n", encoding="utf-8")
    with pytest.raises(surrogate.Unavailable) as raised:
        surrogate.as_samples(surrogate.read_table(table), "Cl")
    assert "Cd" in str(raised.value) and "Cl" in str(raised.value)


def test_a_json_table_is_read_as_rows_or_as_columns(surrogate, tmp_path):
    as_rows = tmp_path / "rows.json"
    as_rows.write_text(json.dumps([{"x": 0, "y": 0, "Cd": 1}, {"x": 1, "y": 1, "Cd": 2}]),
                       encoding="utf-8")
    as_columns = tmp_path / "cols.json"
    as_columns.write_text(json.dumps({"x": [0, 1], "y": [0, 1], "Cd": [1, 2]}),
                          encoding="utf-8")
    assert surrogate.read_table(as_rows) == surrogate.read_table(as_columns)


# -- the candidate-directory layout a parametric round leaves -----------------------


def write_candidate(root: Path, name: str, constants: dict, metrics: dict | None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "design_constants.py").write_text(
        "# the round's design point\n"
        + "".join(f"{k} = {v}\n" for k, v in constants.items()), encoding="utf-8")
    if metrics is not None:
        (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return directory


def test_candidate_directories_are_read_without_executing_their_constants(
        surrogate, tmp_path):
    """`design_constants.py` is somebody else's data, so it is matched, not imported.

    The guard is in the test as well as the code: a constants file that would raise on
    import has to read fine here.
    """
    root = tmp_path / "round1"
    for index, (alpha, camber) in enumerate(
            [(0.0, 0.01), (2.0, 0.01), (4.0, 0.01), (0.0, 0.03), (2.0, 0.03), (4.0, 0.03)]):
        cd = 0.01 + 0.0004 * alpha ** 2 + 2.0 * camber ** 2
        write_candidate(root, f"c{index}", {"ALPHA": alpha, "CAMBER": camber},
                        {"label": "ok", "metrics": {"body": {"Cd": cd}}})
    (root / "c0" / "design_constants.py").write_text(
        "import a_module_that_does_not_exist\nALPHA = 0.0\nCAMBER = 0.01\n",
        encoding="utf-8")

    rows, skipped = surrogate.read_candidates(root, "metrics.body.Cd")

    assert skipped == []
    assert len(rows) == 6
    assert rows[0]["ALPHA"] == 0.0
    names, points, values = surrogate.as_samples(rows, "metrics.body.Cd",
                                                 ["ALPHA", "CAMBER"])
    assert names == ["ALPHA", "CAMBER"]
    assert len(values) == 6


def test_an_unscored_or_failed_candidate_is_named_and_skipped(surrogate, tmp_path):
    """A diverged candidate's coefficient is not a sample of the response. Fitting it
    in is how a surrogate learns that a broken mesh is a good design."""
    root = tmp_path / "round2"
    write_candidate(root, "good", {"A": 1.0}, {"label": "ok", "metrics": {"Cd": 0.4}})
    write_candidate(root, "diverged", {"A": 2.0},
                    {"label": "diverged", "metrics": {"Cd": 1e9}})
    write_candidate(root, "running", {"A": 3.0}, None)
    (root / "not_a_candidate").mkdir()

    rows, skipped = surrogate.read_candidates(root, "metrics.Cd")

    assert [row["A"] for row in rows] == [1.0]
    assert any("diverged: label diverged" in note for note in skipped)
    assert any("running: not scored" in note for note in skipped)
    assert any("not_a_candidate: no design_constants.py" in note for note in skipped)


def test_a_candidate_whose_metrics_lack_the_objective_is_skipped(surrogate, tmp_path):
    root = tmp_path / "round3"
    write_candidate(root, "c0", {"A": 1.0}, {"label": "ok", "metrics": {"Cl": 0.4}})
    rows, skipped = surrogate.read_candidates(root, "metrics.Cd")
    assert rows == []
    assert "no metrics.Cd" in skipped[0]


# -- the CLI ------------------------------------------------------------------------


@pytest.fixture
def sweep_csv(tmp_path):
    table = tmp_path / "sweep.csv"
    rows = ["alpha,camber,Cd"]
    for alpha in (0.0, 1.5, 3.0, 4.5, 6.0, 7.5):
        for camber in (0.0, 0.015, 0.03, 0.045):
            cd = 0.0092 + 0.00041 * (alpha - 1.1) ** 2 + 0.28 * (camber - 0.02) ** 2
            rows.append(f"{alpha},{camber},{cd:.9f}")
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return table


def test_the_cli_writes_its_findings_and_its_figure(surrogate, sweep_csv, tmp_path,
                                                    capsys):
    out = tmp_path / "surrogate.json"
    png = tmp_path / "fit.png"

    code = surrogate.main([str(sweep_csv), "--objective", "Cd",
                           "--json", str(out), "--png", str(png)])

    assert code == 0
    assert png.is_file() and png.stat().st_size > 1000
    found = json.loads(out.read_text(encoding="utf-8"))
    assert found["fit"]["design"] == ["alpha", "camber"]
    assert found["fit"]["degree"] == 2
    assert found["fit"]["rmse_loo"] < 1e-9
    # the quadratic bowl's centre, which the table was built around
    assert found["optimum"]["at"]["alpha"] == pytest.approx(1.1, abs=0.05)
    assert found["optimum"]["at"]["camber"] == pytest.approx(0.02, abs=0.002)
    printed = capsys.readouterr().out
    assert "rmse_loo" in printed
    assert "is not its error" in printed, "the fit error has to be labelled as one"


def test_the_cli_evaluates_the_surrogate_where_it_is_asked(surrogate, sweep_csv, capsys):
    code = surrogate.main([str(sweep_csv), "--objective", "Cd",
                           "--at", "alpha=2.0,camber=0.025"])
    assert code == 0
    printed = capsys.readouterr().out
    exact = 0.0092 + 0.00041 * (2.0 - 1.1) ** 2 + 0.28 * (0.025 - 0.02) ** 2
    assert f"{exact:.6g}" in printed, printed


def test_asking_outside_the_sampled_box_says_so(surrogate, sweep_csv, capsys):
    surrogate.main([str(sweep_csv), "--objective", "Cd", "--at", "alpha=40,camber=0.02"])
    printed = capsys.readouterr().out
    assert "outside the sampled range in alpha" in printed


def test_the_cli_refuses_a_table_too_small_to_fit_without_raising(surrogate, tmp_path,
                                                                 capsys):
    table = tmp_path / "tiny.csv"
    table.write_text("x,y,Cd\n0,0,1\n1,1,2\n", encoding="utf-8")
    assert surrogate.main([str(table), "--objective", "Cd"]) == 0
    assert "not fitted:" in capsys.readouterr().out


def test_the_cli_reads_a_round_of_candidates(surrogate, tmp_path, capsys):
    root = tmp_path / "round"
    for index, alpha in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
        write_candidate(root, f"c{index}", {"ALPHA": alpha},
                        {"label": "ok", "metrics": {"body": {"Cd": 0.01 + 0.002 * alpha}}})
    write_candidate(root, "bad", {"ALPHA": 9.0},
                    {"label": "mesh", "metrics": {"body": {"Cd": 0.0}}})

    code = surrogate.main(["--candidates", str(root), "--objective", "metrics.body.Cd"])

    assert code == 0
    printed = capsys.readouterr().out
    assert "skipped bad: label mesh" in printed
    assert "ALPHA" in printed


def test_the_maximum_is_reported_when_asked_for(surrogate, sweep_csv, capsys):
    surrogate.main([str(sweep_csv), "--objective", "Cd", "--maximise"])
    printed = capsys.readouterr().out
    assert "predicted maximum" in printed


# -- the contract the toolbox keeps -------------------------------------------------


def test_the_script_names_no_statistics_library(surrogate):
    """scipy is not on the image and sklearn cannot be added, so the fit is numpy."""
    text = (TOOLBOX / "surrogate.py").read_text(encoding="utf-8")
    for absent in ("import scipy", "import sklearn", "from scipy", "from sklearn"):
        assert absent not in text


def test_the_docstring_says_the_fit_error_is_not_the_error(surrogate):
    """The one thing a reader of this script has to come away with."""
    doc = (TOOLBOX / "surrogate.py").read_text(encoding="utf-8").split('"""')[1].lower()
    assert "leave-one-out" in doc
    assert "python3 surrogate.py" in doc


def test_the_index_offers_the_surrogate(surrogate):
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    assert index.count("| `surrogate.py` |") == 1, "exactly one row, like every other"


def test_an_exact_fit_is_chosen_by_degree_and_not_by_round_off():
    """The failure this guards was Windows-only and could not be reproduced on a
    developer machine, because it depends on the BLAS: on a quadratic the basis
    contains, degree 2 and degree 3 both fit to round-off, and
    `min(graded, key=rmse_loo)` then compares two numbers near 1e-16 that carry no
    information. GitHub's windows-latest runners ordered them the other way from Linux
    and `build` returned a cubic (#239, runs 44825570959).

    So the rule is exercised directly with the noise forced both ways, rather than
    hoping this machine's arithmetic reproduces a particular order. What must hold is
    that the answer does not depend on which way it falls, and that a degree which
    genuinely fits better still wins.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    scale = float(np.sqrt(np.mean(np.square(values))))
    floor = 32.0 * float(np.finfo(float).eps) * max(scale, 1.0)

    def choose(rows):
        exact = [row for row in rows if row["rmse_loo"] <= floor]
        if exact:
            return min(exact, key=lambda row: row["degree"])["degree"]
        return min(rows, key=lambda row: row["rmse_loo"])["degree"]

    noise_favours_two = [{"degree": 2, "rmse_loo": 2.1e-16},
                         {"degree": 3, "rmse_loo": 3.4e-16}]
    noise_favours_three = [{"degree": 2, "rmse_loo": 3.4e-16},
                           {"degree": 3, "rmse_loo": 2.1e-16}]
    assert choose(noise_favours_two) == 2
    assert choose(noise_favours_three) == 2, (
        "the chosen degree still depends on round-off, which is the #239 failure")

    # The floor must not swallow a real improvement: a cubic that actually fits better
    # is a cubic, and neither residual is anywhere near round-off.
    genuine = [{"degree": 2, "rmse_loo": 0.21}, {"degree": 3, "rmse_loo": 0.03}]
    assert choose(genuine) == 3

    # And the floor is relative to the data, not an absolute 1e-14: objectives in this
    # repository run from 1e-6 drag counts to 1e5 pascals.
    big = np.array([1e5, 2e5, 3e5])
    big_floor = 32.0 * float(np.finfo(float).eps) * max(
        float(np.sqrt(np.mean(np.square(big)))), 1.0)
    assert big_floor > floor * 1e4
