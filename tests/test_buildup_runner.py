"""The runner's refusals, which are the enforcement half of §3.

A run started dirty cannot be cleaned up afterwards, so everything here is about what the
runner declines to do. The observation half -- the grep, the probes, the alarms -- is the
supervisor's and is tested in `test_buildup_cli.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from openreynolds.buildup import isolation

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    """The script, imported by path: `scripts/` is not a package and should not become
    one just so a test can reach it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "cad_buildup_under_test", ROOT / "scripts" / "cad_buildup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_a_workspace_inside_the_repository_is_refused(tmp_path):
    """The task message names the case directory, so a workspace under the tree puts the
    repo path into the desk's own thread -- and then every run greps as contaminated,
    correctly and uselessly."""
    with pytest.raises(SystemExit, match="inside"):
        runner._outside_the_repo(ROOT / "scratch")
    assert runner._outside_the_repo(tmp_path) == tmp_path.resolve()


def test_the_default_workspace_parent_is_outside_the_tree():
    assert runner._outside_the_repo(runner.WORK) == Path(runner.WORK).resolve()


def test_a_case_gets_a_fresh_workspace_and_its_fixture_copied_into_it(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "WELL_KNOWN", ())
    workspace, geometry, prompt = runner.prepare("T5", tmp_path, tmp_path / "runs" / "T5-r1")
    assert workspace.parent == tmp_path.resolve() and workspace.name.endswith("T5-r1")
    assert Path(geometry).is_file() and Path(geometry).parent.name == runner.FIXTURE_DIR
    assert prompt["request"]


def test_a_case_with_no_fixture_is_handed_no_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "WELL_KNOWN", ())
    _workspace, geometry, _prompt = runner.prepare("T1", tmp_path, tmp_path / "r" / "T1-r1")
    assert geometry == ""


def test_a_house_path_that_resolves_aborts_before_a_model_is_called(tmp_path, monkeypatch):
    """The sighting that was missed last round: a stale `/work/.toolbox` from an earlier
    arm. The run does not start, rather than starting and being discarded."""
    stale = tmp_path / "stale" / ".toolbox"
    stale.mkdir(parents=True)
    monkeypatch.setattr(isolation, "WELL_KNOWN", (str(stale),))
    with pytest.raises(isolation.Dirty, match="resolves"):
        runner.prepare("T1", tmp_path / "work", tmp_path / "r" / "T1-r1")


def test_the_abort_is_an_exit_code_and_not_a_traceback(tmp_path, monkeypatch):
    """A harness that aborts has to be scriptable: `3` is 'the environment is dirty'."""
    stale = tmp_path / "stale" / ".toolbox"
    stale.mkdir(parents=True)
    monkeypatch.setattr(isolation, "WELL_KNOWN", (str(stale),))
    code = runner.main(["run", "T1", "--work", str(tmp_path / "work"),
                        "--runs", str(tmp_path / "runs")])
    assert code == 3


def test_an_unknown_case_says_which_ones_there_are(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "WELL_KNOWN", ())
    with pytest.raises(SystemExit, match="T1"):
        runner.prepare("T99", tmp_path, tmp_path / "r" / "T99-r1")


def test_the_runner_never_syncs_a_toolbox(tmp_path):
    """Not by a hidden path, not read-only, not at all. Read off the source rather than
    asserted in prose, because this is the one line whose absence is the arm."""
    text = (ROOT / "scripts" / "cad_buildup.py").read_text(encoding="utf-8")
    assert "put_tree" not in text
    assert "sync_toolbox" not in text


def test_the_record_carries_the_accounting_before_the_run_returns(tmp_path, monkeypatch):
    """T5 of the first baseline sweep recorded $0.00 against 22 cells and 535 s.

    Not a free run -- an unrecorded one. The accounting was written once, from `result`,
    after `desk.run` returned, and a killed run never returns: the watcher signals the
    process group, the default SIGTERM disposition terminates without unwinding, so
    neither the `except` nor the `finally` runs and the record keeps its defaults. The
    ending was recorded by the watcher and the price of reaching it was not, so the case
    sorted last in a report that ranks failures by cost.

    So the invariant is not "the totals are right at the end" -- it is that they are on
    disk **while the run is still going**, which is the only state a kill can observe.
    This test reads `record.json` back at each turn boundary, from inside the run.
    """
    module = load_runner()

    seen: list[dict] = []

    class StubDesk:
        on_turn = None
        on_step = None

        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            # Two turns, each spending, and after each one the record as a killer
            # would find it on disk.
            for turn, total in ((1, {"input": 10, "output": 1000}),
                                (2, {"input": 10, "output": 3000})):
                self.on_turn(turn=turn, steps=0, stop_reason="end_turn",
                             output_tokens=total["output"], tokens=dict(total),
                             fenced=True, text_chars=5, thinking_chars=0,
                             text="hi", block_types=["text"])
                seen.append(record.load(run_dir_holder[0]))
            raise AssertionError("the kill lands here; the run never returns")

    run_dir_holder: list[Path] = []
    real_prepare = module.prepare

    def prepare(case, parent, run_dir):
        run_dir_holder.append(Path(run_dir))
        return real_prepare(case, parent, run_dir)

    from openreynolds.buildup import record

    monkeypatch.setattr(module, "prepare", prepare)
    monkeypatch.setattr(module.core, "CoreDesk", StubDesk)
    monkeypatch.setattr(module, "find_bashrc", lambda: "/dev/null", raising=False)

    with pytest.raises(AssertionError, match="the kill lands here"):
        module.drive("T1", tmp_path / "work", tmp_path / "runs", 0, 0.0)

    assert len(seen) == 2, seen
    # Not defaults: the price of the run so far is readable at every turn boundary.
    assert seen[0]["tokens"] == {"input": 10, "output": 1000}
    assert seen[0]["usd"] > 0
    # `seconds` is asserted present rather than positive: this stub returns in under the
    # 0.1 s the field is rounded to, so a `> 0` here would be testing the clock.
    assert "seconds" in seen[0]
    # And it tracks, rather than being written once and left.
    assert seen[1]["tokens"] == {"input": 10, "output": 3000}
    assert seen[1]["usd"] > seen[0]["usd"]
    assert seen[1]["n_turns"] == 2
