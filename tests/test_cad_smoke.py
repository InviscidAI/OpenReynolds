"""The artifact check, and the sweep it was written after.

An outcome check cannot see instrumentation that has stopped working. Three smoke runs
were called clean while every one of them was missing `heartbeat.jsonl`, because what was
checked was "did the case pass" and "did the gate produce states" -- neither of which the
bug touched. The runs were short enough to finish inside the watcher's 420 s threshold, so
the harness never had to be right for them to look right.

So this asks the other question: does a run leave behind what a healthy run leaves behind.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location(
        "cad_smoke", ROOT / "scripts" / "cad_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(sweep: Path, name: str, *, files: dict[str, str]) -> Path:
    run = sweep / "runs" / name
    run.mkdir(parents=True)
    for filename, body in files.items():
        (run / filename).write_text(body, encoding="utf-8")
    return run


WHOLE = {"record.json": '{"stopped": "done"}', "cells.log": "# -- cell 1 (1s, exit 0)\n",
         "heartbeat.jsonl": '{"turn": 1}\n', "replies.jsonl": '{"text": "hi"}\n',
         "watch.json": "{}"}


def test_a_whole_run_passes(smoke, tmp_path):
    _run(tmp_path, "T2", files=WHOLE)
    assert smoke.check(tmp_path, smoke.EXPECTED) == []


def test_the_run_that_started_this_is_caught(smoke, tmp_path):
    """Work on disk, no heartbeat: the exact shape of the eight that were killed.

    They reported `wedged` at turn 0, steps 0 and $0.00 with fifteen cells in their
    `cells.log`, and the sweep table would have carried them as desk failures.
    """
    missing = dict(WHOLE)
    del missing["heartbeat.jsonl"], missing["replies.jsonl"]
    missing["record.json"] = '{"stopped": "wedged"}'
    _run(tmp_path, "T11", files=missing)

    problems = smoke.check(tmp_path, smoke.EXPECTED)
    assert len(problems) == 1
    assert "heartbeat.jsonl" in problems[0] and "replies.jsonl" in problems[0]


def test_an_empty_artifact_counts_as_missing(smoke, tmp_path):
    """The file existing is not the claim. `pulse.start()` can create it and nothing
    write to it, which is the same silence with a file in the way."""
    hollow = dict(WHOLE, **{"heartbeat.jsonl": ""})
    _run(tmp_path, "T3", files=hollow)
    assert any("heartbeat.jsonl" in line for line in smoke.check(tmp_path, smoke.EXPECTED))


def test_a_refusal_may_legitimately_have_run_no_cells(smoke, tmp_path):
    """The one tolerated absence, and it is tolerated by the ending's own name."""
    refused = dict(WHOLE, **{"record.json": '{"stopped": "refused"}', "cells.log": ""})
    _run(tmp_path, "T6", files=refused)
    assert smoke.check(tmp_path, smoke.EXPECTED) == []
    # And the tolerance does not spread: a refusal still has to have been watched.
    (tmp_path / "runs" / "T6" / "heartbeat.jsonl").write_text("", encoding="utf-8")
    assert smoke.check(tmp_path, smoke.EXPECTED)


def test_the_expected_set_can_be_learned_from_a_known_good_sweep(smoke, tmp_path):
    """Stricter than a list somebody remembered to update."""
    good = tmp_path / "good"
    _run(good, "T1", files=dict(WHOLE, **{"extra.json": "{}"}))
    _run(good, "T2", files=dict(WHOLE, **{"extra.json": "{}"}))
    common = set.intersection(*[{p.name for p in run.glob("*") if p.is_file()}
                                for run in smoke.runs(good)])
    assert "extra.json" in common, "what every healthy run has is what is expected"

    here = tmp_path / "here"
    _run(here, "T1", files=WHOLE)
    assert any("extra.json" in line for line in smoke.check(here, tuple(sorted(common))))


def test_the_real_baseline_sweep_is_whole(smoke):
    """The check, against the sweep it was calibrated on. Skipped where it is not on disk."""
    sweep = ROOT / "docs" / "cad-buildup" / "sweeps" / "core+cad_export-20260917-022129-dd05"
    # The sweep's reading -- `report.md`, `sweep.json`, `findings.jsonl` -- is committed
    # and its `runs/` are not (`.gitignore`), so the directory being there is not the
    # same as the runs being there. This reads the runs, so that is what it asks for.
    if not smoke.runs(sweep):
        pytest.skip("the baseline sweep's run records are not in this checkout "
                    "(`sweeps/*/runs/` is gitignored)")
    assert smoke.check(sweep, smoke.EXPECTED) == [], (
        "the sweep this was calibrated against no longer passes it")
    assert len(smoke.runs(sweep)) == 26
