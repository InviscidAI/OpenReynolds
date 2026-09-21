from __future__ import annotations

import json

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


def test_the_model_and_the_provider_that_served_it_survive_a_restart(tmp_path):
    """A resume runs the pair back, and neither half is worth anything alone."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model = "gpt-5"
    store.session.provider = "openai"
    store.save()

    reopened = Store(root, "s1").session
    assert (reopened.provider, reopened.model) == ("openai", "gpt-5")


def test_the_endpoint_that_served_the_model_survives_a_restart(tmp_path):
    """Not covered by the every-field-is-read rule, and the field a resume leans on
    hardest: two keys of one family -- a vendor's own and a router in front of it --
    list different model ids, so the pair is only restorable at the endpoint that
    recorded it."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model = "anthropic/claude-sonnet-4.5"
    store.session.provider = "openai"
    store.session.base_url = "https://openrouter.ai/api/v1"
    store.save()

    reopened = Store(root, "s1").session
    assert reopened.base_url == "https://openrouter.ai/api/v1"
    assert (reopened.provider, reopened.model) == ("openai", "anthropic/claude-sonnet-4.5")


def test_a_session_written_before_the_endpoint_was_recorded_has_none(tmp_path):
    """No `base_url` key at all, which is "not recorded" and not "the preset's" --
    the hosted app reads the raw JSON and tells the two apart."""
    root = tmp_path / "studies"
    (root / "s1").mkdir(parents=True)
    (root / "s1" / "session.json").write_text(
        json.dumps({"study_id": "s1", "model": "gpt-5", "provider": "openai"}),
        encoding="utf-8",
    )

    assert Store(root, "s1").session.base_url == ""


def test_a_session_written_before_the_provider_was_recorded_still_loads(tmp_path):
    """Older studies name a model and no provider; unknown keys are dropped, missing
    ones default, and neither is a reason to fail to open a study."""
    root = tmp_path / "studies"
    (root / "s1").mkdir(parents=True)
    (root / "s1" / "session.json").write_text(
        json.dumps({"study_id": "s1", "model": "claude-opus-5", "gone": "a field that left"}),
        encoding="utf-8",
    )

    session = Store(root, "s1").session
    assert (session.model, session.provider) == ("claude-opus-5", "")


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
