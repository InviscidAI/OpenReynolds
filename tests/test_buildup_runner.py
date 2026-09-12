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
