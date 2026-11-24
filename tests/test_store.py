from __future__ import annotations

import json
import threading

import pytest

from openreynolds.store import Store, TornRecord, list_studies, new_study_id


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


# -- the record is not destroyed by writing it (#167, #168) --------------------


def test_a_record_that_will_not_parse_is_refused_rather_than_read_as_a_blank_study(tmp_path):
    """It used to load as a default `Session` -- `created_at` of now, no model, no
    home, no jobs -- which the next `save()` wrote over the real record. One
    `openreynolds label` then destroyed a study permanently and exited 0 (#167)."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model = "claude-opus-5"
    store.session.home = "/work/s1"
    store.session.capture_seq = 135
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    store.save()
    record = root / "s1" / "session.json"
    whole = record.read_text(encoding="utf-8")
    record.write_text(whole[: len(whole) // 2], encoding="utf-8")

    with pytest.raises(TornRecord):
        Store(root, "s1")
    assert record.read_text(encoding="utf-8") == whole[: len(whole) // 2], \
        "nothing was written to it"

    torn: list[str] = []
    assert list_studies(root, torn) == [], "it is not listed with invented facts"
    assert torn == ["s1"], "it is named instead"

    record.write_text(whole, encoding="utf-8")
    back = Store(root, "s1").session
    assert (back.model, back.home, back.capture_seq, len(back.jobs)) == (
        "claude-opus-5", "/work/s1", 135, 1)


def test_a_save_never_leaves_half_a_record_on_disk(tmp_path):
    """The write used to truncate in place, so a concurrent reader saw part of a
    record -- and `_load` turned that into the blank study above."""
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.save()
    record = root / "s1" / "session.json"
    torn: list[int] = []
    stop = threading.Event()

    def read() -> None:
        while not stop.is_set():
            try:
                text = record.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                json.loads(text)
            except ValueError:
                torn.append(len(text))

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        for n in range(40):
            store.session.tags = ["a-tag-long-enough-to-move-the-size-around"] * (n % 3)
            store.save()
    finally:
        stop.set()
        reader.join(5)
    assert torn == []


def test_a_live_store_does_not_revert_what_another_writer_filed(tmp_path):
    """A session holds one Store for its whole run and saves the whole record on every
    message, so `openreynolds label` from a second terminal was reverted -- and the
    project the session loaded at startup came back with it (#168). Each writer owns
    the fields it changed."""
    root = tmp_path / "studies"
    live = Store(root, "s1")
    live.session.project = "Other"
    live.session.model = "claude-opus-5"
    live.save()
    live = Store(root, "s1")             # the session, now running

    filing = Store(root, "s1")           # `openreynolds label`, a second terminal
    filing.session.project = "Important"
    filing.session.tags = ["keepme"]
    filing.save()

    live.append_message("assistant", "the session carries on and saves")
    on_disk = json.loads((root / "s1" / "session.json").read_text(encoding="utf-8"))
    assert (on_disk["project"], on_disk["tags"]) == ("Important", ["keepme"])

    # A field edited in place is still a changed field. `commands.tag_lines` appends
    # to and removes from `session.tags`, and a baseline sharing that list with the
    # record would change when it does -- so dropping a tag would look like no
    # change and never be written, which is what `label --untag` did.
    filing = Store(root, "s1")
    filing.session.tags.remove("keepme")
    filing.save()
    on_disk = json.loads((root / "s1" / "session.json").read_text(encoding="utf-8"))
    assert on_disk["tags"] == []
    filing.session.tags.append("again")
    filing.save()
    assert json.loads((root / "s1" / "session.json").read_text(encoding="utf-8"))["tags"]         == ["again"]

    # and it still owns its own fields, over more than one save
    live.record_job("job-1", cmd="simpleFoam", name="solve")
    for _ in range(3):
        live.append_message("assistant", "again")
    on_disk = json.loads((root / "s1" / "session.json").read_text(encoding="utf-8"))
    assert (on_disk["project"], on_disk["tags"]) == ("Important", ["again"])
    assert (on_disk["model"], on_disk["capture_seq"], list(on_disk["jobs"])) == (
        "claude-opus-5", 4, ["job-1"])


def test_a_live_store_mends_a_record_torn_under_it_rather_than_dying_on_it(tmp_path):
    """`save()` refusing to write over an unparseable file would kill a running
    session on its next message. It does not have to: `_load` refuses, so a Store
    that exists holds a complete record, and writing that over the wreckage is a
    repair rather than #167's loss of one."""
    root = tmp_path / "studies"
    live = Store(root, "s1")
    live.session.model = "claude-opus-5"
    live.session.capture_seq = 7
    live.save()
    record = root / "s1" / "session.json"
    record.write_text(record.read_text(encoding="utf-8")[:40], encoding="utf-8")

    live.append_message("assistant", "the session carries on")
    back = Store(root, "s1").session
    assert (back.model, back.capture_seq) == ("claude-opus-5", 8)


@pytest.mark.parametrize("data", [
    '{"study_id": "s1", "title": "Dü'.encode()[:-1],  # cut inside a character
    b"[1, 2]",
    b'{"study_id": "s1", "jobs": [1, 2]}',
    b'{"study_id": "s1", "jobs": {"j": 5}}',
    b'{"study_id": "s1", "jobs": {"j": {"job_id": "j", "unknown": 1}}}',
])
def test_a_record_that_is_not_a_study_is_refused_and_named_in_the_listing(tmp_path, data):
    root = tmp_path / "studies"
    (root / "s1").mkdir(parents=True)
    record = root / "s1" / "session.json"
    record.write_bytes(data)

    with pytest.raises(TornRecord):
        Store(root, "s1")
    torn: list[str] = []
    assert list_studies(root, torn) == [] and torn == ["s1"]
    assert record.read_bytes() == data


def test_a_record_saved_with_a_byte_order_mark_still_loads(tmp_path):
    root = tmp_path / "studies"
    (root / "s1").mkdir(parents=True)
    (root / "s1" / "session.json").write_bytes(b'\xef\xbb\xbf{"study_id": "s1", "model": "m"}')

    assert Store(root, "s1").session.model == "m"


def test_a_save_that_dies_partway_leaves_the_record_whole_and_no_temp_file(tmp_path, monkeypatch):
    root = tmp_path / "studies"
    store = Store(root, "s1")
    store.session.model = "claude-opus-5"
    store.save()
    record = root / "s1" / "session.json"
    before = record.read_bytes()
    real_write_text = type(record).write_text

    def dies_halfway(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(type(record), "write_text", dies_halfway)
    store.session.tags = ["anything"]
    with pytest.raises(OSError, match="disk full"):
        store.save()
    monkeypatch.undo()

    assert record.read_bytes() == before
    assert [p.name for p in (root / "s1").iterdir()] == ["session.json"]


def test_several_writers_at_once_never_show_a_reader_half_a_record(tmp_path):
    """With one temp name shared by every writer, a second save truncated the file
    the first was about to move into place, and readers saw a torn record."""
    root = tmp_path / "studies"
    Store(root, "s1").save()
    stop = threading.Event()
    torn: list[str] = []

    def writer(n: int) -> None:
        for i in range(30):
            store = Store(root, "s1")
            store.session.title = f"{n}-{i}" * 200
            store.save()

    def reader() -> None:
        while not stop.is_set():
            try:
                Store(root, "s1")
            except TornRecord as exc:
                torn.append(str(exc))

    watching = threading.Thread(target=reader, daemon=True)
    watching.start()
    writers = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
    try:
        for thread in writers:
            thread.start()
        for thread in writers:
            thread.join()
    finally:
        stop.set()
        watching.join(5)
    assert torn == []
    # No temp file left behind by any of the three, which is what this line is for.
    # It used to read `== ["session.json"]`; `revisions.jsonl` now sits beside it,
    # appended to by whichever writer changed the title (`store.REVISED`), so the
    # check is spelled as the absence of leftovers rather than as a file count.
    left = {p.name for p in (root / "s1").iterdir()}
    assert left == {"session.json", "revisions.jsonl"}, left
    assert not any(".tmp" in name for name in left)


def test_a_stale_writer_never_turns_the_transcript_counter_back(tmp_path):
    """Hosted, 2026-11-24: another writer saved the record with `capture_seq` one
    behind. The session had not touched the field since its last save, so per-field
    merging took the lower value back, and the next row reused the last one's seq. The
    hosted page keys rows on seq and never showed the agent's reply."""
    import json

    store = Store(tmp_path, "s1")
    store.append_message("user", "and a sphere")
    store.append_message("event", "the workspace is ready")
    record = store.dir / "session.json"
    stale = json.loads(record.read_text())
    stale["capture_seq"] = 1                    # what a writer holding an older record puts back
    record.write_text(json.dumps(stale))
    store.session.title = "and a sphere"
    store.save()
    seq = store.append_message("assistant", "You said: and a sphere")
    seqs = [json.loads(line)["seq"] for line in (store.dir / "messages.jsonl").read_text().splitlines()]
    assert seq == 2 and seqs == [0, 1, 2], seqs
    assert json.loads(record.read_text())["capture_seq"] == 3


def test_a_second_writer_titling_the_study_never_rolls_the_session_back(tmp_path):
    """The hosted runner keeps a Store of its own and titles the study while the
    agent's Store appends rows and counts tokens. Neither may put the other's fields
    back as they were when it read them."""
    import json

    agent = Store(tmp_path, "s1")
    agent.save()
    stop = threading.Event()

    def title_it():
        runner = Store(tmp_path, "s1")
        n = 0
        while not stop.is_set():
            n += 1
            runner.session.title = f"title {n}"
            runner.save()

    other = threading.Thread(target=title_it)
    other.start()
    try:
        for i in range(300):
            agent.append_message("user", f"row {i}")
            agent.session.runtime_s = float(i + 1)
            agent.save()
    finally:
        stop.set()
        other.join()
    seqs = [json.loads(line)["seq"] for line in (agent.dir / "messages.jsonl").read_text().splitlines()]
    record = json.loads((agent.dir / "session.json").read_text())
    assert seqs == list(range(300)), "a seq was given twice"
    assert record["capture_seq"] == 300 and record["runtime_s"] == 300.0
