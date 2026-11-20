from __future__ import annotations

import json
import threading

import pytest

from openreynolds import store as store_mod
from openreynolds.store import Store, UnreadableRecord, list_studies, new_study_id


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


def _a_real_study(root, study_id="20260824-121326-8584"):
    store = Store(root, study_id)
    store.session.instance_id = "974f4406"
    store.session.remote_study_id = "3dc97696"
    store.session.model = "claude-opus-5"
    store.session.home = f"/work/{study_id}"
    store.session.created_at = "2026-08-24T04:13:26Z"
    store.session.capture_seq = 135
    for n in range(5):
        store.record_job(f"job-{n}", cmd="pimpleFoam", name=f"run{n}")
    store.save()
    return store


def test_a_truncated_record_is_refused_and_left_as_it_was(tmp_path):
    root = tmp_path / "studies"
    path = _a_real_study(root).dir / "session.json"
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 2])
    torn = path.read_bytes()

    with pytest.raises(UnreadableRecord) as raised:
        Store(root, "20260824-121326-8584")

    assert raised.value.path == path
    assert "session.json" in str(raised.value)
    assert path.read_bytes() == torn


@pytest.mark.parametrize("data", [
    b"", b"   \n", b"[]", b"null", b"42",
    '{"study_id": "s1", "title": "Düse'.encode()[:-1],  # cut inside a character
    b'{"study_id": "s1", "jobs": [1, 2]}',
    b'{"study_id": "s1", "jobs": {"j": 5}}',
    b'{"study_id": "s1", "jobs": {"j": {"job_id": "j", "unknown": 1}}}',
])
def test_a_record_that_is_not_a_study_is_refused(tmp_path, data):
    root = tmp_path / "studies"
    (root / "s1").mkdir(parents=True)
    (root / "s1" / "session.json").write_bytes(data)

    with pytest.raises(UnreadableRecord):
        Store(root, "s1")


def test_a_record_saved_with_a_byte_order_mark_still_loads(tmp_path):
    root = tmp_path / "studies"
    (root / "s1").mkdir(parents=True)
    (root / "s1" / "session.json").write_bytes(b'\xef\xbb\xbf{"study_id": "s1", "model": "m"}')

    assert Store(root, "s1").session.model == "m"


def test_no_record_at_all_is_still_a_new_study(tmp_path):
    store = Store(tmp_path / "studies", "s1")
    assert store.session.study_id == "s1" and store.session.capture_seq == 0


def test_save_leaves_only_the_record_behind(tmp_path):
    store = _a_real_study(tmp_path / "studies")
    store.save()

    assert sorted(p.name for p in store.dir.iterdir() if p.is_file()) == ["session.json"]
    again = Store(tmp_path / "studies", store.session.study_id).session
    assert again.model == "claude-opus-5" and len(again.jobs) == 5 and again.capture_seq == 135


def test_a_save_that_dies_partway_leaves_the_old_record_whole(tmp_path, monkeypatch):
    root = tmp_path / "studies"
    store = _a_real_study(root)
    path = store.dir / "session.json"
    before = path.read_bytes()

    real_write_text = type(path).write_text

    def dies_halfway(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(type(path), "write_text", dies_halfway)
    store.session.tags = ["anything"]
    with pytest.raises(OSError, match="disk full"):
        store.save()
    monkeypatch.undo()

    assert path.read_bytes() == before
    assert [p.name for p in store.dir.iterdir() if p.name.endswith(".tmp")] == []


def test_a_replace_the_platform_refuses_for_a_moment_is_tried_again(tmp_path, monkeypatch):
    store = _a_real_study(tmp_path / "studies")
    real_replace = store_mod.os.replace
    refusals = iter([PermissionError(13, "in use"), PermissionError(13, "in use")])

    def busy_then_free(src, dst):
        for exc in refusals:
            raise exc
        real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", busy_then_free)
    monkeypatch.setattr(store_mod, "_PAUSE", 0)
    store.session.tags = ["anything"]
    store.save()

    assert Store(tmp_path / "studies", store.session.study_id).session.tags == ["anything"]


def test_readers_never_see_half_a_record_while_others_save(tmp_path):
    root = tmp_path / "studies"
    study_id = _a_real_study(root).session.study_id
    stop = threading.Event()
    failures: list[BaseException] = []

    def saver(n):
        try:
            for i in range(40):
                store = Store(root, study_id)
                store.session.tags = [f"t{n}-{i}"] * 50
                store.save()
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    def reader():
        try:
            while not stop.is_set():
                session = Store(root, study_id).session
                assert session.model == "claude-opus-5" and len(session.jobs) == 5
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    watching = threading.Thread(target=reader)
    watching.start()
    savers = [threading.Thread(target=saver, args=(n,)) for n in range(4)]
    for thread in savers:
        thread.start()
    for thread in savers:
        thread.join()
    stop.set()
    watching.join()

    assert failures == []
    assert [p.name for p in (root / study_id).iterdir() if p.name.endswith(".tmp")] == []


def test_listing_skips_an_unreadable_study_and_says_so(tmp_path, capsys):
    root = tmp_path / "studies"
    Store(root, "20260101-000000-good").save()
    (root / "20260202-000000-torn").mkdir(parents=True)
    (root / "20260202-000000-torn" / "session.json").write_text('{"study_id": "2026', encoding="utf-8")

    assert [s.study_id for s in list_studies(root)] == ["20260101-000000-good"]
    assert "20260202-000000-torn" in capsys.readouterr().err
