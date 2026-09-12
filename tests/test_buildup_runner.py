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

    def prepare(case, parent, run_dir, identifier=""):
        run_dir_holder.append(Path(run_dir))
        return real_prepare(case, parent, run_dir, identifier)

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


def test_two_sweeps_do_not_ask_for_the_same_workspace(tmp_path, monkeypatch):
    """A sweep names its run directories after the case -- `runs/T3` -- so a workspace
    named from the directory is `T3-T3` for every sweep that ever runs.

    `fresh_workspace` refuses a root that is not empty, correctly, so the second sweep on
    a machine aborted on its own leftovers and exited 3 -- which the driver reports as
    "this sweep is void", the same words it uses for a house surface. The standalone path
    hid it: there the run directory is already `T1-<id>`, so the workspace was unique by
    accident rather than by construction.
    """
    module = load_runner()
    parent = tmp_path / "work"
    sweep_run_dir = tmp_path / "sweeps" / "core-1" / "runs" / "T1"

    first = module.prepare("T1", parent, sweep_run_dir, "20260912-090000-aaaa")[0]
    second = module.prepare("T1", parent, sweep_run_dir, "20260912-100000-bbbb")[0]

    assert first != second, "two runs of one case shared a workspace"
    assert first.name == "T1-20260912-090000-aaaa"
    assert second.name == "T1-20260912-100000-bbbb"


def test_a_model_with_no_price_on_record_refuses_before_it_spends(tmp_path, monkeypatch):
    """The runner writes the records a sweep ranks by cost, so an unpriced model does not
    make a run cheaper -- it makes its cost meaningless. Beside the key check, and for the
    same reason: a meaningless number that looks like a real one is how the first baseline
    reported $3.00 for a corpus that cost $7.49."""
    from openreynolds.llm import presets

    assert presets.prices("claude-opus-5") is not None
    assert presets.prices("a-model-nobody-priced") is None
    # Priced at zero rather than crashing, so the guard is what refuses, not the sum.
    assert presets.spend({"output": 1_000_000}, "a-model-nobody-priced") == 0.0
    assert presets.spend({"output": 1_000_000}, "claude-opus-5") == 25.0


def test_opus_and_sonnet_do_not_price_the_same_run_the_same(tmp_path):
    """The bug itself: one untagged table held Sonnet 5's rates because Sonnet 5 is the
    default preset, while the build-up sweep sets `mesher_model` to Opus 5."""
    from openreynolds.llm import presets

    tokens = {"input": 30, "cache_read": 251_953, "cache_write": 89_915, "output": 38_936}
    opus = presets.spend(tokens, "claude-opus-5")
    sonnet = presets.spend(tokens, "claude-sonnet-5")
    assert round(sonnet, 4) == 0.6646
    assert round(opus, 4) == 1.6615
    assert round(opus / sonnet, 2) == 2.50


def test_a_case_whose_pass_is_a_refusal_is_scored_on_that_and_not_on_checkmesh():
    """The scoring half of T6. `checkmesh_ok` and `passed` are opposites here.

    The record used to carry only `checkmesh_ok`, so a case whose correct outcome is not
    finishing was scored by whether it finished -- and T6's guess, which produced a
    perfectly valid mesh of a part whose scale nobody established, counted as the
    corpus's sixth success.
    """
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from cad_accept import load_prompts

    prompts = load_prompts()
    assert prompts["T6"]["expects"] == "refused", "T6 declares its criterion in its file"

    # Every case that is not scored on a mesh says so in its own file, and says it in the
    # one form `_expects` reads. This used to assert the set was exactly {T6}, which was a
    # census rather than a mechanism: §4 grows the corpus rather than freezing it, and T25
    # -- a request that states no dimension at all -- made the assertion fail for the right
    # reason. What has to hold is that `expects` is never invented here, only read.
    refusals = {name for name, p in prompts.items() if p["expects"] != "done"}
    assert "T6" in refusals
    for name in refusals:
        assert prompts[name]["expects"] == "refused", (
            f"{name} expects {prompts[name]['expects']!r}; the only non-`done` criterion the "
            "record can score is `refused`")
        declared = Path(ROOT / prompts[name]["file"]).read_text(encoding="utf-8")
        assert "**Passes as:** `refused`" in declared, (
            f"{name} is scored as a refusal, so its own file must declare it")

    def scored(expects, stopped, checkmesh_ok):
        return (stopped == "refused" if expects == "refused"
                else stopped == "done" and checkmesh_ok)

    # The run that actually happened: finished, meshed, and failed the case.
    assert scored("refused", "done", True) is False
    # The run the case asks for: no mesh at all, and a pass.
    assert scored("refused", "refused", False) is True
    # And an ordinary case is unaffected by any of it.
    assert scored("done", "done", True) is True
    assert scored("done", "done", False) is False
    assert scored("done", "refused", False) is False


def test_refused_is_a_terminal_state_the_record_will_accept():
    """A run that ends outside `TERMINAL` is a bug, not a result -- so the refusal path
    is not usable until the vocabulary has a word for it."""
    from openreynolds.buildup import record

    assert "refused" in record.TERMINAL
    assert record.classify("refused") == "refused"
