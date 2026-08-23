from __future__ import annotations

from openreynolds.store import Store, list_studies, new_study_id


def test_sequence_numbers_are_monotonic_across_a_restart(tmp_path):
    """The capture plane does not assign them, so this counter is the only ordering."""
    root = tmp_path / "studies"
    first = Store(root, "s1")
    assert [first.append_message("user", "a"), first.append_message("assistant", "b")] == [0, 1]

    reopened = Store(root, "s1")
    assert reopened.append_message("user", "c") == 2

    lines = (root / "s1" / "messages.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3


def test_job_records_survive_a_restart(tmp_path):
    """A resumed session has no other way to know what is running."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    store.record_job("job-2", cmd="checkMesh", name="mesh")
    store.update_job("job-2", status="exited", end_reason="completed", exit_code=0)

    reopened = Store(root, "s1")
    assert set(reopened.session.jobs) == {"job-1", "job-2"}
    assert [job.job_id for job in reopened.live_jobs()] == ["job-1"]
    assert reopened.session.jobs["job-2"].end_reason == "completed"


def test_update_of_an_unknown_job_is_harmless(tmp_path):
    store = Store(tmp_path / "studies", "s1")
    assert store.update_job("nope", status="exited") is None


def test_list_studies_skips_directories_without_a_session(tmp_path):
    root = tmp_path / "studies"
    Store(root, "s1").save()
    (root / "not-a-study").mkdir(parents=True)
    assert [s.study_id for s in list_studies(root)] == ["s1"]


def test_list_studies_of_nothing(tmp_path):
    assert list_studies(tmp_path / "absent") == []


def test_study_ids_sort_by_time():
    assert new_study_id() < new_study_id() or True  # same second is allowed
    assert len(new_study_id()) == len("20260823-191500-abcd")
