"""Absence, asserted before a run and verified after it.

Every test here exists because the last round assumed isolation and the assumption was
false: the bare arm was given no toolbox, found `/work/.toolbox` through a stale symlink,
read a signature out of our source with `sed` and `grep`, and opened a template the brief
no longer mentions. Every number that arm produced was measuring a desk with tools.
"""

from __future__ import annotations

import pytest

from openreynolds.buildup import isolation


def test_a_reused_workspace_is_refused_before_anything_runs(tmp_path):
    """One root per case per run. A directory another arm worked in carries its files."""
    first = isolation.fresh_workspace(tmp_path, "T1", "r1")
    (first / "build.py").write_text("x = 1", encoding="utf-8")
    with pytest.raises(isolation.Dirty):
        isolation.fresh_workspace(tmp_path, "T1", "r1")


def test_a_house_file_under_the_workspace_is_dirty(tmp_path):
    work = tmp_path / "work"
    (work / "case").mkdir(parents=True)
    (work / "case" / "cad_convert.py").write_text("# ours", encoding="utf-8")
    with pytest.raises(isolation.Dirty, match="cad_convert.py"):
        isolation.preflight(work, well_known=())


def test_a_symlink_that_leaves_the_workspace_for_the_repo_is_dirty(tmp_path):
    """The exact shape of the sighting that was missed: not a copy, a link."""
    work = tmp_path / "work"
    work.mkdir()
    (work / ".toolbox").symlink_to(isolation.TOOLBOX_DIR)
    with pytest.raises(isolation.Dirty):
        isolation.preflight(work, well_known=())


def test_a_well_known_path_that_resolves_anywhere_is_dirty(tmp_path):
    """`/work/.toolbox` existing at all is the environment being dirty, wherever it points."""
    stale = tmp_path / "stale"
    stale.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(isolation.Dirty, match="resolves"):
        isolation.preflight(work, well_known=(str(stale),))


def test_a_clean_workspace_passes_and_says_what_it_looked_at(tmp_path):
    work = tmp_path / "work"
    (work / "case" / "constant").mkdir(parents=True)
    report = isolation.preflight(work, well_known=())
    assert report["workspace"] == str(work.resolve())
    assert report["house_names"] > 10


def test_the_grep_reports_the_line_it_saw_a_house_surface_on():
    found = isolation.scan({"cells.log": "ls ../.toolbox\nsed -n 1,40p cad_convert.py"})
    assert found.contaminated
    assert [hit.name for hit in found.hits] == [".toolbox", "cad_convert.py"]
    assert found.hits[1].line == 2
    assert "sed -n" in found.hits[1].evidence


def test_the_repo_path_itself_counts_as_a_house_surface():
    assert isolation.scan({"log": f"find / -name cells.py | head\n{isolation.REPO}/openreynolds"}).contaminated


def test_a_name_too_ordinary_to_be_evidence_is_not_grepped_for():
    """A case that writes its own `README.md` has not found us, and a good run should
    not be discarded for it."""
    assert "README.md" not in isolation.house_names()
    assert not isolation.scan({"log": "cat README.md"}).contaminated


def test_a_tooled_arm_that_never_reached_its_tool_is_invalid_too():
    """The same discipline in reverse: a run that failed to reach a tool it was given is
    not a measurement of the tooled arm."""
    never = isolation.scan({"log": "print(1)"}, expected=["mesh_look.py"])
    assert never.contaminated and never.missing == ["mesh_look.py"]
    reached = isolation.scan({"log": "python3 mesh_look.py ."}, expected=["mesh_look.py"])
    assert not reached.contaminated and reached.missing == []


def test_the_grep_reads_every_text_file_a_run_left_behind(tmp_path):
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    (run / "replies.jsonl").write_text('{"text": "nothing to see"}\n', encoding="utf-8")
    (run / "outputs" / "step-3.txt").write_text("cat /work/.toolbox/preflight.py",
                                                encoding="utf-8")
    (run / "render.png").write_bytes(b"\x89PNG not text")
    found = isolation.scan_run(run)
    assert found.contaminated
    assert {hit.where for hit in found.hits} == {"outputs/step-3.txt"}


def test_the_supervisors_own_record_is_not_grepped_back_at_it(tmp_path):
    """The graded record quotes every hit it found, so a second pass over the same
    directory would read those quotes back and call a clean run contaminated."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "record.json").write_text(
        '{"contamination": {"hits": [{"name": ".toolbox"}]}}', encoding="utf-8")
    assert not isolation.scan_run(run).contaminated
