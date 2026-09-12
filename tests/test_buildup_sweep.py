"""The sweep: one run per case, and the paired table that compares two sweeps.

What is pinned here is the arithmetic a decision rests on. The runs themselves cost real
money and minutes, so nothing here launches one -- the driver's job of starting three
processes in order is exercised by `test_buildup_cli.py` one run at a time, and what is
left is which rows pair, which differences count, and what the test over them says.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from openreynolds.buildup import record

ROOT = Path(__file__).resolve().parents[1]


def load_sweep_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "cad_sweep_under_test", ROOT / "scripts" / "cad_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_sweep_module()


RAN_AT = iter(f"2026-09-{day:02d}T09:00:00" for day in range(1, 28))


def make_sweep(tmp_path: Path, name: str, runs: dict[str, dict], **manifest) -> Path:
    directory = tmp_path / name
    (directory / "runs").mkdir(parents=True)
    for case, fields in runs.items():
        record.save(directory / "runs" / case,
                    record.Record(case=case, arm="core", **fields))
    (directory / "sweep.json").write_text(json.dumps(
        {"sweep_id": name, "label": name, "model": "claude-opus-5", "effort": "medium",
         "git_sha": "abc123", "noise_steps": 8, "cases": sorted(runs),
         "started_at": next(RAN_AT), **manifest}),
        encoding="utf-8")
    return directory


# -- the sign test ---------------------------------------------------------------


def test_the_sign_test_is_two_sided_and_drops_the_ties():
    """A case that did not move is not evidence either way, so it is not counted --
    dropping ties is what makes the remaining count a fair coin under the null."""
    assert sweep.sign_test(0, 0) == 1.0
    assert sweep.sign_test(3, 3) == 1.0
    assert sweep.sign_test(8, 0) == pytest.approx(2 / 256)
    assert sweep.sign_test(0, 8) == pytest.approx(2 / 256)
    assert sweep.sign_test(5, 1) == pytest.approx(2 * 7 / 64)


def test_eight_of_eight_is_the_significance_the_last_round_reported():
    """The hand-off's headline was eight prompts, n = 1 each, p = 0.008 -- which is this
    function, and is why breadth rather than repeats is what buys confidence."""
    assert sweep.sign_test(8, 0) < 0.01


# -- what pairs, and what counts --------------------------------------------------


def test_a_difference_inside_the_noise_band_is_not_a_difference(tmp_path, capsys):
    """Measured on T1 with Opus at medium: the same prompt lands within three to eight
    cells of itself. So an eight-cell move has not been shown to be a move, and a report
    that reads one as a result is reading the sampling."""
    before = make_sweep(tmp_path, "core-1", {
        "T1": dict(n_steps=20, checkmesh_ok=True), "T2": dict(n_steps=20, checkmesh_ok=True)})
    after = make_sweep(tmp_path, "core+x-2", {
        "T1": dict(n_steps=12, checkmesh_ok=True),   # -8, the band's edge
        "T2": dict(n_steps=11, checkmesh_ok=True)})  # -9, outside it
    sweep.table(str(after), str(before))
    out = capsys.readouterr().out
    assert "-8 (noise)" in out and "-9 (fewer)" in out
    assert "1 cases used fewer cells, 0 more" in out


def test_a_case_only_one_sweep_ran_is_excluded_rather_than_counted(tmp_path, capsys):
    before = make_sweep(tmp_path, "core-1", {"T1": dict(n_steps=20), "T2": dict(n_steps=20)})
    after = make_sweep(tmp_path, "core+x-2", {"T1": dict(n_steps=5), "T9": dict(n_steps=5)})
    sweep.table(str(after), str(before))
    out = capsys.readouterr().out
    assert "unpaired, excluded from the test: T2, T9" in out
    assert "1 cases used fewer cells" in out


def test_a_contaminated_run_is_discarded_from_the_baseline_not_averaged_in(tmp_path, capsys):
    """It is not a measurement of the arm it is filed under, so it does not get to
    contribute to the arm's number -- and it is named rather than dropped quietly."""
    directory = make_sweep(tmp_path, "core-1", {
        "T1": dict(n_steps=20, checkmesh_ok=True),
        "T2": dict(n_steps=4, checkmesh_ok=True, contaminated=True)})
    sweep.table(str(directory))
    out = capsys.readouterr().out
    assert "CONTAMINATED, discarded from the baseline: T2" in out
    assert "total: 1/1 checkMesh ok" in out


def test_a_sweep_whose_model_moved_is_not_a_comparison_and_says_so(tmp_path, capsys):
    """Model capability dominated every tooling effect measured last round. Two sweeps on
    different models differ by the model, whatever else changed."""
    before = make_sweep(tmp_path, "core-1", {"T1": dict(n_steps=20)}, model="claude-sonnet-5")
    after = make_sweep(tmp_path, "core+x-2", {"T1": dict(n_steps=8)})
    sweep.table(str(after), str(before))
    assert "WARNING: the model or the effort moved" in capsys.readouterr().out


def test_the_verdict_line_says_it_is_not_any_single_row(tmp_path, capsys):
    """With one run per case, a single row flipping is sampling. The paired count over
    the whole corpus is the result, and the report has to say so where it is read."""
    before = make_sweep(tmp_path, "core-1", {"T1": dict(n_steps=20)})
    after = make_sweep(tmp_path, "core+x-2", {"T1": dict(n_steps=2)})
    sweep.table(str(after), str(before))
    out = capsys.readouterr().out
    assert "not any single row" in out and "sign test over the cases that moved" in out


# -- the corpus and the manifest --------------------------------------------------


def test_the_cases_are_named_off_the_corpus_and_a_typo_is_refused():
    assert sweep._names("all") == sorted(sweep.corpus())
    assert sweep._names("t1,t2") == ["T1", "T2"]
    with pytest.raises(SystemExit, match="no such case"):
        sweep._names("T1,T99")


def test_a_sweep_records_what_the_core_was_when_it_ran():
    """A re-sweep after an addition compares two sweeps, so 'the same core apart from the
    addition' has to be checkable rather than remembered."""
    assert sweep.git_sha()
    assert sweep.NOISE_STEPS == 8


def test_the_default_is_one_run_per_case():
    """Repeats buy precision on a single case's pass rate, which is not a number anybody
    decides on. The flag stays for calibrating variance and for chasing one intermittent
    failure; it is not the default."""
    args = sweep.build_parser().parse_args(["run", "--label", "core"])
    assert args.repeat == 1 and args.cases == "all"


def test_a_sweep_cannot_be_run_without_a_label():
    """Every comparison names two sweeps, so a sweep nobody can say the point of is a
    sweep nobody can cite."""
    with pytest.raises(SystemExit):
        sweep.build_parser().parse_args(["run"])


# -- the chain -------------------------------------------------------------------


def test_a_sweep_records_what_it_is_to_be_read_against(tmp_path, monkeypatch, capsys):
    """Iteration N's sweep is iteration N+1's baseline, so the chain is a fact about the
    sweep and belongs in its manifest. Reconstructing it from timestamps afterwards works
    right up until two sweeps were run on the same day for different reasons."""
    monkeypatch.setattr(sweep, "SWEEPS", tmp_path)
    first = make_sweep(tmp_path, "core-1", {"T1": dict(n_steps=20, checkmesh_ok=True)})
    second = make_sweep(tmp_path, "core+x-2", {"T1": dict(n_steps=9, checkmesh_ok=True)},
                        baseline=first.name)

    sweep.table(str(second))            # no --against: it already knows
    out = capsys.readouterr().out
    assert f"against {first.name}" in out
    assert "-11 (fewer)" in out


def test_latest_is_the_newest_by_when_it_ran_not_by_its_name(tmp_path, monkeypatch):
    """A sweep directory is `<label>-<id>`, and `+` sorts below `-`, so sorting the names
    puts `core+x-...` before `core-...`. A lexicographic `latest` is right only while every
    sweep carries the same label -- and the whole point of a chain is that they do not."""
    monkeypatch.setattr(sweep, "SWEEPS", tmp_path)
    make_sweep(tmp_path, "core-1", {"T1": dict(n_steps=20)})          # ran first
    make_sweep(tmp_path, "core+x-2", {"T1": dict(n_steps=9)})         # ran second
    assert sorted(["core-1", "core+x-2"])[-1] == "core-1", "the trap this guards"
    assert sweep._resolve_baseline("latest") == "core+x-2"
    assert sweep._resolve_baseline("") == ""


def test_an_explicit_against_beats_the_recorded_one(tmp_path, monkeypatch, capsys):
    """For the comparison nobody planned: two additions back, or a re-run baseline."""
    monkeypatch.setattr(sweep, "SWEEPS", tmp_path)
    old = make_sweep(tmp_path, "core-0", {"T1": dict(n_steps=40)})
    recent = make_sweep(tmp_path, "core+x-1", {"T1": dict(n_steps=20)})
    latest = make_sweep(tmp_path, "core+y-2", {"T1": dict(n_steps=9)},
                        baseline=recent.name)
    sweep.table(str(latest), str(old))
    out = capsys.readouterr().out
    assert f"against {old.name}" in out and "-31 (fewer)" in out


def test_the_listing_shows_the_chain(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "SWEEPS", tmp_path)
    make_sweep(tmp_path, "core-1", {"T1": dict(n_steps=20)})
    make_sweep(tmp_path, "core+x-2", {"T1": dict(n_steps=9)}, baseline="core-1")
    sweep.listing()
    out = capsys.readouterr().out
    assert "core+x-2" in out and "<- core-1" in out
