"""The study record keeps its earlier states instead of writing over them.

`session.json` is rewritten whole on every save, with a per-field merge so two writers
do not revert each other (`store.Store.save`). What that merge cannot do is remember:
a field's previous value is gone the moment the next save lands, so a study that ran on
two models looked afterwards like a study that had only ever used the second, with one
token total and no way to split it.

The transcript has always been append-only. These tests are about the record being so
too, for the fields that say how the study was set up.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from openreynolds import cli
from openreynolds.catalog import history, lineage
from openreynolds.config import Config
from openreynolds.store import REVISED, Store, new_study_id


def rows(root, study_id):
    path = root / study_id / "revisions.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_study_being_made_is_not_a_revision_of_itself(tmp_path):
    """Otherwise every study opens with a wall of fields changing from nothing."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model = "claude-opus-5"
    store.save()

    assert rows(root, "s1") == []


def test_a_model_swapped_mid_study_is_kept_with_both_models(tmp_path):
    """The case this exists for. `/model` rewrites one field and the old value was the
    only record of what the first half of the study actually ran on."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model, store.session.provider = "claude-opus-5", "anthropic"
    store.save()
    store.session.model, store.session.provider = "claude-sonnet-4", "reynolds"
    store.save()

    changed = rows(root, "s1")
    assert len(changed) == 1
    assert changed[0]["changes"]["model"] == {"from": "claude-opus-5", "to": "claude-sonnet-4"}
    assert changed[0]["changes"]["provider"] == {"from": "anthropic", "to": "reynolds"}
    assert changed[0]["at"]


def test_the_accounting_fields_do_not_fill_the_log(tmp_path):
    """Tokens, the job table and the capture sequence change on nearly every save. A
    log of those would be a second and worse copy of the transcript."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model = "m"
    store.save()
    for turn in range(5):
        store.record_accounting(1.5, {"input": 100, "output": 20})
        store.append_message("user", f"turn {turn}")
        store.record_job(f"job-{turn}", cmd="simpleFoam", name="solve")

    assert rows(root, "s1") == []
    assert store.session.tokens["input"] == 500


def test_a_change_is_recorded_once_by_the_writer_that_made_it(tmp_path):
    """`openreynolds label` and a live session are two processes on one record, and the
    merge in `save()` hands each the other's fields back. Recording `payload` against
    what was on disk would log one change from both sides and attribute it to both."""
    root = tmp_path / "studies"
    session = Store(root, "s1")
    session.session.model = "m"
    session.save()

    labeller = Store(root, "s1")
    labeller.session.project = "ahmed-body"
    labeller.save()

    # the session saves again and adopts the project it never set
    session.append_message("user", "hello")
    session.save()

    changed = rows(root, "s1")
    assert len(changed) == 1, changed
    assert changed[0]["changes"] == {"project": {"from": "", "to": "ahmed-body"}}
    assert session.session.project == "ahmed-body"


def test_tags_added_one_at_a_time_read_back_as_the_two_steps_they_were(tmp_path):
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.title = "cylinder"
    store.save()
    store.session.tags = ["validated"]
    store.save()
    store.session.tags = ["validated", "re-100"]
    store.save()

    changed = rows(root, "s1")
    assert [row["changes"]["tags"]["to"] for row in changed] == [
        ["validated"], ["validated", "re-100"]]


def test_the_solver_being_settled_on_is_a_revision_from_nothing(tmp_path):
    """`solver` is None until the agent chooses, and which choice was made when is
    exactly the kind of thing a rerun needs to know."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.title = "x"
    store.save()
    store.session.solver = "openfoam"
    store.save()

    assert rows(root, "s1")[0]["changes"]["solver"] == {"from": None, "to": "openfoam"}


def test_the_log_survives_a_restart_and_keeps_its_order(tmp_path):
    root = tmp_path / "studies"
    first = Store(root, "s1")
    first.session.model = "a"
    first.save()
    first.session.model = "b"
    first.save()

    reopened = Store(root, "s1")
    reopened.session.model = "c"
    reopened.save()

    assert [row["changes"]["model"]["to"] for row in reopened.revisions()] == ["b", "c"]


def test_a_study_with_no_log_reads_as_no_revisions_rather_than_raising(tmp_path):
    """Every study made before this existed, which is all of them."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.save()
    assert store.revisions() == []


def test_a_half_written_last_line_is_dropped_not_raised_on(tmp_path):
    """It is read while a session appends to it, as the transcript is."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model = "a"
    store.save()
    store.session.model = "b"
    store.save()
    path = root / "s1" / "revisions.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"at": "2026-', encoding="utf-8")

    assert len(store.revisions()) == 1


def test_every_revised_field_is_a_field_the_record_has(tmp_path):
    """The list is written by hand, so a field renamed out of `Session` would make it
    silently stop logging rather than fail."""
    store = Store(tmp_path / "studies", "s1")
    for field in REVISED:
        assert hasattr(store.session, field), field


# -- lineage, which was recorded all along and read by nothing ---------------------


def test_a_rerun_of_a_rerun_can_say_what_it_came_from(tmp_path):
    root = tmp_path / "studies"
    first = Store(root, "s1")
    first.save()
    second = Store(root, "s2")
    second.session.cloned_from = "s1"
    second.save()
    third = Store(root, "s3")
    third.session.cloned_from = "s2"
    third.save()

    assert lineage(root, "s3") == ["s1", "s2", "s3"]
    assert lineage(root, "s1") == ["s1"]


def test_a_parent_this_machine_does_not_have_ends_the_chain(tmp_path):
    """A clone of a study since deleted, or made on another machine. A gap is not an
    origin and is not reported as one."""
    root = tmp_path / "studies"
    child = Store(root, "s2")
    child.session.cloned_from = "s1-elsewhere"
    child.save()

    assert lineage(root, "s2") == ["s2"]


def test_a_cloned_from_cycle_cannot_wedge_the_walk(tmp_path):
    root = tmp_path / "studies"
    a, b = Store(root, "s1"), Store(root, "s2")
    a.session.cloned_from = "s2"
    a.save()
    b.session.cloned_from = "s1"
    b.save()

    assert lineage(root, "s1") == ["s2", "s1"]


# -- what the command prints -------------------------------------------------------


def test_the_history_of_a_study_carries_its_changes_its_chain_and_its_transcript(tmp_path):
    root = tmp_path / "studies"
    parent = Store(root, "s1")
    parent.save()
    store = Store(root, "s2")
    store.session.cloned_from = "s1"
    store.session.model = "claude-opus-5"
    store.save()
    store.append_message("user", "a cylinder at Re 100")
    store.append_message("assistant", "reading the case")
    store.session.model = "claude-sonnet-4"
    store.save()

    found = history(root, "s2")

    assert found["lineage"] == ["s1", "s2"]
    assert found["cloned_from"] == "s1"
    assert found["messages"] == {"user": 1, "assistant": 1}
    # the clone's own `cloned_from` went in on its first save, which is its creation
    assert len(found["revisions"]) == 1
    assert found["now"]["model"] == "claude-sonnet-4"
    assert set(found["now"]) == set(REVISED)


def test_the_history_of_an_untouched_study_is_empty_rather_than_invented(tmp_path):
    root = tmp_path / "studies"
    store = Store(root, new_study_id())
    store.save()
    found = history(root, store.session.study_id)
    assert found["revisions"] == []
    assert found["messages"] == {}
    assert found["lineage"] == [store.session.study_id]


# -- the command, which is the only way anybody sees any of this --------------------


@pytest.fixture
def studies_dir(tmp_path, monkeypatch):
    """A studies directory `Config.load` points at, as every subcommand reads it."""
    root = tmp_path / "studies"
    root.mkdir()
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))
    return root


def test_the_command_prints_each_change_with_both_values(studies_dir):
    store = Store(studies_dir, "20261120-090000-aaaa")
    store.session.model, store.session.provider = "claude-opus-5", "anthropic"
    store.save()
    store.session.model = "claude-sonnet-4"
    store.save()
    store.session.project = "ahmed-body"
    store.save()

    said = CliRunner().invoke(cli.main, ["history", "--study", "20261120-090000-aaaa"])

    assert said.exit_code == 0, said.output
    assert "model: claude-opus-5 -> claude-sonnet-4" in said.output
    assert "project: (unset) -> ahmed-body" in said.output


def test_the_command_says_a_study_has_no_log_rather_than_that_nothing_changed(studies_dir):
    """Every study made before revisions were recorded has none, and the two cases
    are not the same thing."""
    Store(studies_dir, "20261120-090000-bbbb").save()

    said = CliRunner().invoke(cli.main, ["history", "--study", "20261120-090000-bbbb"])

    assert said.exit_code == 0, said.output
    assert "has no log" in said.output


def test_the_command_names_the_study_a_rerun_came_from(studies_dir):
    Store(studies_dir, "20261120-090000-cccc").save()
    child = Store(studies_dir, "20261120-100000-dddd")
    child.session.cloned_from = "20261120-090000-cccc"
    child.save()

    said = CliRunner().invoke(cli.main, ["history"])  # the newest, which is the clone

    assert said.exit_code == 0, said.output
    assert "cloned from 20261120-090000-cccc" in said.output


def test_the_command_hands_a_script_one_json_object(studies_dir):
    store = Store(studies_dir, "20261120-090000-eeee")
    store.session.solver = None
    store.save()
    store.session.solver = "openfoam"
    store.save()

    said = CliRunner().invoke(cli.main, ["history", "--study", "20261120-090000-eeee",
                                         "--json"])

    assert said.exit_code == 0, said.output
    found = json.loads(said.stdout)
    assert found["revisions"][0]["changes"]["solver"] == {"from": None, "to": "openfoam"}
    assert found["lineage"] == ["20261120-090000-eeee"]
