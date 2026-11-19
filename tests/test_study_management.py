"""Projects, tags, the dashboard, the accounting, and running a study again.

Five roadmap items turned out to be one piece of work, because each of them wanted the
same missing thing: metadata on the study record and a way to ask about it. These are
the tests for the parts that can go wrong quietly.

Two of them are policy rather than behaviour, and they are the ones to read first.

A STUDY'S STATE IS NOT GUESSED. A job row reads `running` because that is how the
session that started it left the record; the only thing that knows whether the process
is alive is the workspace service, and asking it means starting a workspace. A listing
does not get to spend money, so such a study is reported `unknown` and the test below
asserts that nothing was reserved, acquired or started to answer it.

MONEY IS SHOWN ONLY WHERE THE PERSON'S OWN KEY PAID. The hosted path never names a
cost in money: the service's spending cap is rendered as a multiple of a normal study
(`ceiling.py`), a money clause in a service message is stripped before it is shown
(`backend.base.without_money`), and `/status` reports tokens. Bring-your-own-key is
the other case entirely -- the person is billed by their vendor, `llm.presets` knows
the rates, and withholding the figure there would be withholding their own bill. So
there are two tests, one for each side of that line.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from openreynolds import catalog, cli, commands
from openreynolds.config import Config
from openreynolds.store import JobRecord, Session, Store, list_studies, tag_name
from openreynolds.watch import earlier_conversation

OLD_RECORD = {
    "study_id": "20260101-090000-aaaa",
    "instance_id": "iid-old",
    "model": "claude-opus-5",
    "created_at": "2026-01-01T09:00:00Z",
    "title": "duct pressure drop",
    "capture_seq": 2,
    "jobs": {},
}
"""A `session.json` as a release before any of this wrote one: no project, no tags, no
end time, no runtime, no token totals, no `cloned_from`."""


def write(root, study_id, record):
    (root / study_id).mkdir(parents=True, exist_ok=True)
    (root / study_id / "session.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")


def transcript(root, study_id, rows):
    (root / study_id).mkdir(parents=True, exist_ok=True)
    (root / study_id / "messages.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def studies_dir(tmp_path, monkeypatch):
    """A studies directory `Config.load` points at, as every subcommand reads it."""
    root = tmp_path / "studies"
    root.mkdir()
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))
    return root


@pytest.fixture
def no_workspace(monkeypatch):
    """Every way to a workspace, booby-trapped.

    A dashboard that reached one would be a listing that starts a billed machine, and
    the mistake is invisible from the output: the answer would simply be more
    confident. So the seams raise, and a test that gets its answer anyway has proved
    the listing never asked.
    """
    reached: list[str] = []

    def refuse(name):
        def go(*_args, **_kwargs):
            reached.append(name)
            raise AssertionError(f"a listing reached the service through {name}")
        return go

    for name in ("acquire", "reserve", "start_instance", "FoamdClient"):
        if hasattr(cli.hosted, name):
            monkeypatch.setattr(cli.hosted, name, refuse(name))
    return reached


# -- the record on disk --------------------------------------------------------


def test_a_record_from_before_any_of_this_loads_clean(studies_dir):
    """The format drops keys it does not know (`Store._load`), so the reverse has to
    hold too: a record missing every field added here must load without a word, with
    the defaults, and must not lose what it does carry."""
    write(studies_dir, OLD_RECORD["study_id"], OLD_RECORD)

    session = Store(studies_dir, OLD_RECORD["study_id"]).session

    assert session.title == "duct pressure drop"
    assert session.capture_seq == 2
    assert session.project == ""
    assert session.tags == []
    assert session.cloned_from == ""
    assert session.ended_at == ""
    assert session.runtime_s == 0.0
    assert session.tokens == {}


def test_saving_an_old_record_keeps_what_it_had_and_writes_the_new_fields(studies_dir):
    write(studies_dir, OLD_RECORD["study_id"], OLD_RECORD)
    store = Store(studies_dir, OLD_RECORD["study_id"])

    store.session.project = "ducts"
    store.save()

    raw = json.loads((studies_dir / OLD_RECORD["study_id"] / "session.json").read_text())
    for key, value in OLD_RECORD.items():
        assert raw[key] == value, key
    assert raw["project"] == "ducts"
    assert raw["tags"] == []
    assert raw["tokens"] == {}


def test_a_new_record_round_trips(studies_dir):
    store = Store(studies_dir, "20261101-120000-bbbb")
    store.session.project = "wing"
    store.session.tags = ["validated", "mesh-study"]
    store.session.cloned_from = "20260101-090000-aaaa"
    store.session.runtime_s = 930.5
    store.session.tokens = {"input": 100, "cache_read": 9000, "output": 42}
    store.save()

    again = Store(studies_dir, "20261101-120000-bbbb").session

    assert again.project == "wing"
    assert again.tags == ["validated", "mesh-study"]
    assert again.cloned_from == "20260101-090000-aaaa"
    assert again.runtime_s == 930.5
    assert again.tokens == {"input": 100, "cache_read": 9000, "output": 42}


def test_an_older_version_of_the_loader_reads_a_new_record_without_failing(studies_dir):
    """A record written here, read by the code as it was before these fields existed.

    The loader keeps only the keys its own `Session` declares, so an older build drops
    the project and the tags -- it has nowhere to put them, and that is how this format
    has always handled a field it does not know. What must not happen is the other
    thing: an older build raising on a key it has never seen, which would make a study
    unreadable by the version the user may still be running on another machine.
    """
    store = Store(studies_dir, "20261101-120000-bbbb")
    store.session.project = "wing"
    store.session.tags = ["validated"]
    store.session.tokens = {"input": 100}
    store.save()
    raw = json.loads((studies_dir / "20261101-120000-bbbb" / "session.json").read_text())

    was_known = {"study_id", "instance_id", "remote_study_id", "model", "provider",
                 "base_url", "mode", "solver", "home", "created_at", "title",
                 "capture_seq", "checkpoint"}
    older = Session(**{k: v for k, v in raw.items() if k in was_known})

    assert older.study_id == "20261101-120000-bbbb"
    assert older.title == ""


def test_a_null_on_disk_leaves_the_field_at_its_default(studies_dir):
    """A hand-edited or half-written record. `tags: null` reaching the filters as None
    is a TypeError in the middle of a listing."""
    write(studies_dir, "20261101-120000-cccc", {
        "study_id": "20261101-120000-cccc", "tags": None, "tokens": None, "title": None,
    })

    session = Store(studies_dir, "20261101-120000-cccc").session

    assert session.tags == []
    assert session.tokens == {}
    assert session.title == ""


def test_tags_are_one_tag_however_they_were_typed():
    assert tag_name("Mesh Study") == "mesh-study"
    assert tag_name(" validated ") == "validated"
    assert tag_name("Mesh  Study") == "mesh-study"


# -- the accounting ------------------------------------------------------------


def test_time_and_tokens_add_up_across_the_sessions_of_one_study(studies_dir):
    """A study is resumed, and two sessions of twenty minutes is forty minutes of
    work -- not the twenty of whichever session happened to be last."""
    store = Store(studies_dir, "20261101-120000-dddd")

    store.record_accounting(1200.0, {"input": 100, "cache_read": 9000})
    store.record_accounting(600.0, {"input": 50, "output": 7})

    session = Store(studies_dir, "20261101-120000-dddd").session
    assert session.runtime_s == 1800.0
    assert session.tokens == {"input": 150, "cache_read": 9000, "output": 7}
    assert session.ended_at


def test_a_session_records_on_the_study_what_it_spent(monkeypatch, tmp_path):
    """The wiring, not the arithmetic. A field nothing writes is the whole class of
    bug `tests/test_wiring.py` exists for, and the accounting is only worth having if
    a real session's close-down puts it on the record."""
    from conftest import message, text_block
    from test_refused_start import _session

    run = _session(monkeypatch, tmp_path, [message([text_block("done")])], one_shot="go")

    written = Store(tmp_path / "studies", run.store.session.study_id).session
    assert written.tokens.get("input") == 100
    assert written.tokens.get("output") == 10
    assert written.ended_at
    assert written.runtime_s >= 0.0


def test_a_study_that_recorded_no_time_is_not_reported_as_having_taken_none():
    assert catalog.elapsed(0) == "-"
    assert catalog.elapsed(None) == "-"
    assert catalog.elapsed(45) == "45s"
    assert catalog.elapsed(930) == "15m30s"
    assert catalog.elapsed(3700) == "1h01m"


# -- state ---------------------------------------------------------------------


def hosted_study(root, study_id="20261110-090000-eeee", **fields):
    store = Store(root, study_id)
    store.session.title = "wing at 5 deg"
    store.session.instance_id = "iid-hosted"
    store.session.model = "claude-opus-5"
    store.session.provider = "reynolds"
    store.session.jobs["j1"] = JobRecord(job_id="j1", name="simpleFoam", status="running")
    store.session.runtime_s = 930.0
    store.session.tokens = {"input": 20_000, "cache_read": 400_000, "output": 9_000}
    for name, value in fields.items():
        setattr(store.session, name, value)
    store.save()
    return store


def byok_study(root, study_id="20261115-090000-ffff", **fields):
    store = Store(root, study_id)
    store.session.title = "cylinder Cd"
    store.session.model = "claude-opus-5"
    store.session.provider = "anthropic"
    store.session.jobs["j1"] = JobRecord(
        job_id="j1", name="simpleFoam", status="exited", end_reason="completed")
    store.session.runtime_s = 3700.0
    store.session.tokens = {"input": 20_000, "cache_read": 400_000, "output": 9_000}
    store.session.ended_at = "2026-11-15T10:00:00Z"
    for name, value in fields.items():
        setattr(store.session, name, value)
    store.save()
    return store


def test_a_job_row_still_reading_running_is_unknown_and_nothing_is_started(
    studies_dir, no_workspace
):
    """The whole point of `unknown`. The record says a job was running; the workspace
    is the only thing that knows whether it still is, and a dashboard that asked would
    start a billed machine to draw a table."""
    hosted_study(studies_dir)

    result = CliRunner().invoke(cli.main, ["dashboard", "--json"])

    assert result.exit_code == 0, result.output
    row, = json.loads(result.stdout)["studies"]
    assert row["state"] == catalog.UNKNOWN
    assert "asking would start one" in row["state_detail"]
    assert row["running_jobs"] == 1
    assert no_workspace == [], "the dashboard reached the service"


def test_a_study_whose_jobs_all_ended_is_finished(studies_dir, no_workspace):
    byok_study(studies_dir)

    result = CliRunner().invoke(cli.main, ["dashboard", "--json"])

    row, = json.loads(result.stdout)["studies"]
    assert row["state"] == catalog.FINISHED
    assert "completed" in row["state_detail"]
    assert no_workspace == []


def test_a_study_that_never_ran_anything_is_new(studies_dir):
    write(studies_dir, OLD_RECORD["study_id"], OLD_RECORD)

    row, = catalog.rows(studies_dir)

    assert row["state"] == catalog.NEW


def test_running_is_never_claimed_from_disk_alone():
    """`running` is deliberately not a state this machine can report. If it is ever
    added, it has to come with something that actually asked the workspace."""
    assert "running" not in catalog.STATES


# -- the money line ------------------------------------------------------------


def test_a_hosted_study_is_reported_in_time_and_tokens_and_never_in_money(studies_dir):
    hosted_study(studies_dir)

    text = CliRunner().invoke(cli.main, ["dashboard"])
    data = CliRunner().invoke(cli.main, ["dashboard", "--json"])

    assert "$" not in text.output, text.output
    assert "15m30s" in text.output
    assert "429,000" in text.output
    row, = json.loads(data.stdout)["studies"]
    assert row["cost_usd"] is None
    assert row["own_key"] is False
    assert row["total_tokens"] == 429_000
    assert row["runtime_s"] == 930.0


def test_a_bring_your_own_key_study_is_told_what_its_own_key_cost(studies_dir):
    """The person is billed by their vendor for these tokens at a rate this package
    already knows (`llm.presets.PRICE_PER_MTOK`). Withholding it would be withholding
    their own bill."""
    byok_study(studies_dir)

    text = CliRunner().invoke(cli.main, ["dashboard"])
    data = CliRunner().invoke(cli.main, ["dashboard", "--json"])

    row, = json.loads(data.stdout)["studies"]
    assert row["own_key"] is True
    assert row["cost_usd"] == pytest.approx(20_000 * 5.00 / 1e6
                                            + 400_000 * 0.50 / 1e6
                                            + 9_000 * 25.00 / 1e6)
    assert f"${row['cost_usd']:.2f}" in text.output


def test_a_study_with_no_provider_recorded_is_not_priced(studies_dir):
    """Empty is "not recorded", which is neither the hosted model nor a known key, and
    a guess either way would be a number nobody can check."""
    byok_study(studies_dir, provider="")

    row, = catalog.rows(studies_dir)

    assert row["own_key"] is False
    assert row["cost_usd"] is None


def test_a_model_with_no_known_rate_has_no_figure_rather_than_a_zero(studies_dir):
    """`prices()` answers None rather than an empty dict precisely so that an unpriced
    run cannot be reported as a free one."""
    byok_study(studies_dir, model="some-vendors-new-model")

    row, = catalog.rows(studies_dir)

    assert row["own_key"] is True
    assert row["cost_usd"] is None


def test_a_study_whose_sessions_predate_the_token_record_has_no_figure(studies_dir):
    byok_study(studies_dir, tokens={})

    row, = catalog.rows(studies_dir)

    assert row["cost_usd"] is None


# -- filters -------------------------------------------------------------------


@pytest.fixture
def three_studies(studies_dir):
    write(studies_dir, OLD_RECORD["study_id"], OLD_RECORD)
    hosted_study(studies_dir, project="wing", tags=["validated"])
    byok_study(studies_dir, project="wing")
    return studies_dir


def ids(result):
    return [row["study_id"] for row in json.loads(result.stdout)["studies"]]


def listed(*args):
    return CliRunner().invoke(cli.main, ["studies", "--json", *args])


def test_studies_are_newest_first(three_studies):
    assert ids(listed()) == ["20261115-090000-ffff", "20261110-090000-eeee",
                             "20260101-090000-aaaa"]


def test_filter_by_project(three_studies):
    assert ids(listed("--project", "wing")) == ["20261115-090000-ffff",
                                                "20261110-090000-eeee"]


def test_filter_by_project_ignores_how_it_was_capitalised(three_studies):
    assert len(ids(listed("--project", "WING"))) == 2


def test_filter_by_tag(three_studies):
    assert ids(listed("--tag", "Validated")) == ["20261110-090000-eeee"]


def test_every_tag_given_has_to_be_on_the_study(three_studies):
    """An and, not an or: somebody naming two tags is narrowing."""
    assert ids(listed("--tag", "validated", "--tag", "nope")) == []


def test_filter_by_free_text_over_the_title_and_the_project(three_studies):
    assert ids(listed("--text", "duct")) == ["20260101-090000-aaaa"]
    assert ids(listed("--text", "WING")) == ["20261115-090000-ffff",
                                             "20261110-090000-eeee"]


def test_filter_by_state(three_studies):
    assert ids(listed("--state", "finished")) == ["20261115-090000-ffff"]
    assert ids(listed("--state", "unknown")) == ["20261110-090000-eeee"]
    assert ids(listed("--state", "new")) == ["20260101-090000-aaaa"]


def test_filters_combine(three_studies):
    assert ids(listed("--project", "wing", "--state", "finished")) == [
        "20261115-090000-ffff"]


def test_a_filter_that_matches_nothing_is_an_empty_list_and_not_an_error(three_studies):
    result = listed("--project", "there-is-no-such-project")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["studies"] == []
    assert json.loads(result.stdout)["dir"] == str(three_studies)


def test_a_filter_that_matches_nothing_says_so_rather_than_that_there_are_none(
    three_studies
):
    result = CliRunner().invoke(cli.main, ["studies", "--project", "nope"])

    assert "match those filters" in result.output


def test_an_unknown_state_is_refused_rather_than_matching_nothing(three_studies):
    result = CliRunner().invoke(cli.main, ["studies", "--state", "running"])

    assert result.exit_code == 2
    assert "running" in result.output


# -- the shape nothing may break -----------------------------------------------

WAS_ANSWERED = ("study_id", "title", "instance_id", "model", "created_at", "running_jobs")
"""What `openreynolds studies --json` answered before any of this. Something is
reading these."""


def test_the_studies_json_keys_that_existed_are_all_still_there(three_studies):
    for row in json.loads(listed().stdout)["studies"]:
        for key in WAS_ANSWERED:
            assert key in row, f"studies --json no longer answers {key}"
    assert "studies" in json.loads(listed().stdout)
    assert "dir" in json.loads(listed().stdout)


def test_the_studies_json_still_counts_the_running_jobs_as_it_did(three_studies):
    rows = {row["study_id"]: row for row in json.loads(listed().stdout)["studies"]}

    assert rows["20261110-090000-eeee"]["running_jobs"] == 1
    assert rows["20261115-090000-ffff"]["running_jobs"] == 0


def test_the_group_output_flag_works_for_the_dashboard(three_studies):
    """A subcommand that answers in JSON has to be in `JSON_CAPABLE_COMMANDS`, or the
    group flag in front of it is refused."""
    result = CliRunner().invoke(cli.main, ["--output-format", "stream-json", "dashboard"])

    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["studies"]) == 3


def test_a_listing_and_a_comparison_describe_a_study_the_same_way(three_studies):
    """The rows are built on `compare.identity`, so the two commands cannot come to
    disagree about what a study is."""
    from openreynolds.compare import identity

    row = catalog.row(three_studies, Store(three_studies, "20261115-090000-ffff"))
    said = identity(three_studies, "20261115-090000-ffff")

    for key, value in said.items():
        assert row[key] == value, key


# -- setting a project and tags ------------------------------------------------


def test_label_files_a_study_and_tags_it(three_studies):
    result = CliRunner().invoke(cli.main, [
        "label", "--study", "20260101-090000-aaaa",
        "--project", "ducts", "--tag", "Mesh Study", "--tag", "validated",
    ])

    assert result.exit_code == 0, result.output
    session = Store(three_studies, "20260101-090000-aaaa").session
    assert session.project == "ducts"
    assert session.tags == ["mesh-study", "validated"]


def test_a_tag_with_a_space_in_it_is_one_tag(three_studies):
    """`--tag` is repeatable, so its value is a whole tag -- splitting it on
    whitespace made `--tag "Mesh Study"` into two."""
    CliRunner().invoke(cli.main, ["label", "--study", "20260101-090000-aaaa",
                                  "--tag", "Mesh Study"])

    assert Store(three_studies, "20260101-090000-aaaa").session.tags == ["mesh-study"]


def test_label_drops_a_tag(three_studies):
    CliRunner().invoke(cli.main, ["label", "--study", "20261110-090000-eeee",
                                  "--untag", "validated"])

    assert Store(three_studies, "20261110-090000-eeee").session.tags == []


def test_label_says_when_there_was_no_such_tag_to_drop(three_studies):
    result = CliRunner().invoke(cli.main, ["label", "--study", "20261110-090000-eeee",
                                           "--untag", "never-applied"])

    assert "nothing to drop" in result.output
    assert Store(three_studies, "20261110-090000-eeee").session.tags == ["validated"]


def test_label_with_an_empty_project_clears_it(three_studies):
    CliRunner().invoke(cli.main, ["label", "--study", "20261110-090000-eeee",
                                  "--project", ""])

    assert Store(three_studies, "20261110-090000-eeee").session.project == ""


def test_label_with_nothing_asked_only_reports(three_studies):
    result = CliRunner().invoke(cli.main, ["label", "--study", "20261110-090000-eeee"])

    assert "validated" in result.output
    session = Store(three_studies, "20261110-090000-eeee").session
    assert session.project == "wing" and session.tags == ["validated"]


def test_label_refuses_a_study_it_does_not_have_rather_than_making_one(three_studies):
    result = CliRunner().invoke(cli.main, ["label", "--study", "no-such-study",
                                           "--project", "x"])

    assert result.exit_code == 1
    assert "No study called" in result.output
    assert not (three_studies / "no-such-study").exists()


def test_the_slash_commands_write_the_same_record(studies_dir):
    """`/project` and `/tag` in a session, `openreynolds label` out of one: one record,
    and the slash commands are answered by the harness without a model call."""
    store = Store(studies_dir, "20261101-120000-gggg")

    assert commands.project_lines(store, "ducts") == ["project: ducts"]
    assert commands.tag_lines(store, "Mesh Study validated") == ["tags: mesh, study, validated"]
    assert commands.tag_lines(store, "-study") == ["tags: mesh, validated"]

    session = Store(studies_dir, "20261101-120000-gggg").session
    assert session.project == "ducts"
    assert session.tags == ["mesh", "validated"]


def test_the_slash_commands_report_when_nothing_is_set(studies_dir):
    store = Store(studies_dir, "20261101-120000-hhhh")

    assert "no project" in commands.project_lines(store)[0]
    assert "no tags" in commands.tag_lines(store)[0]


def test_both_new_verbs_are_in_the_one_registry():
    """`/help`, Tab completion and the web composer's list are all read off
    `COMMANDS`, so a verb that is not there exists on one surface only."""
    verbs = {spec.verb for spec in commands.COMMANDS}

    assert {"/project", "/tag"} <= verbs
    assert "/project" in commands.HELP_TEXT and "/tag" in commands.HELP_TEXT
    assert ("/tag ", "tag this study so you can find it again, or drop a tag with -tag") in [
        (line, summary) for line, summary in commands.completions("/ta")]


def test_a_tag_typed_while_a_question_is_open_does_not_decline_it():
    """Anything typed at an open approval that is not an answer declines it, which
    would make filing a study cancel the job the agent asked to run."""
    from openreynolds import approval

    assert commands.PROJECT in approval._LOCAL_KINDS
    assert commands.TAG in approval._LOCAL_KINDS


# -- clone and rerun -----------------------------------------------------------


def with_a_transcript(root, study_id="20261115-090000-ffff"):
    store = byok_study(root, study_id, project="wing", tags=["validated"])
    transcript(root, study_id, [
        {"seq": 0, "role": "user", "at": "2026-11-15T09:00:00Z",
         "content": "Find the drag coefficient of a 2D cylinder at Re = 100."},
        {"seq": 1, "role": "assistant", "at": "2026-11-15T09:01:00Z", "content": "ok"},
        {"seq": 2, "role": "user", "at": "2026-11-15T09:30:00Z", "content": "refine it"},
    ])
    (store.fetch_dir() / "cylinder.step").write_text("an input", encoding="utf-8")
    return store


def test_clone_copies_the_setup_and_the_inputs_and_not_the_transcript(studies_dir):
    with_a_transcript(studies_dir)

    result = CliRunner().invoke(cli.main, ["clone", "--study", "20261115-090000-ffff"])

    assert result.exit_code == 0, result.output
    fresh, = [item for item in list_studies(studies_dir)
              if item.cloned_from == "20261115-090000-ffff"]
    assert fresh.project == "wing"
    assert fresh.tags == ["validated"]
    assert fresh.title == "cylinder Cd"
    assert fresh.model == "claude-opus-5" and fresh.provider == "anthropic"
    here = studies_dir / fresh.study_id
    assert (here / "files" / "cylinder.step").read_text() == "an input"
    assert not (here / "messages.jsonl").exists()


def test_clone_carries_nothing_of_the_run_that_happened(studies_dir):
    with_a_transcript(studies_dir)

    CliRunner().invoke(cli.main, ["clone", "--study", "20261115-090000-ffff"])

    fresh, = [item for item in list_studies(studies_dir)
              if item.cloned_from == "20261115-090000-ffff"]
    assert fresh.instance_id == ""
    assert fresh.jobs == {}
    assert fresh.runtime_s == 0.0 and fresh.tokens == {}
    assert fresh.ended_at == "" and fresh.checkpoint is None
    assert fresh.remote_study_id == "" and fresh.home == ""


def test_rerun_reproduces_the_opening_request_word_for_word(studies_dir):
    with_a_transcript(studies_dir)
    asked = "Find the drag coefficient of a 2D cylinder at Re = 100."

    result = CliRunner().invoke(cli.main, ["rerun", "--study", "20261115-090000-ffff"])

    assert result.exit_code == 0, result.output
    fresh, = [item for item in list_studies(studies_dir)
              if item.cloned_from == "20261115-090000-ffff"]
    rows = [json.loads(line) for line
            in (studies_dir / fresh.study_id / "messages.jsonl").read_text().splitlines()]
    assert [row["content"] for row in rows] == [asked]
    assert rows[0]["role"] == "user"


def test_a_rerun_opens_unanswered_so_an_ordinary_resume_answers_it(studies_dir):
    """Nothing new runs the request. It is written as the person's own first message
    and left without a reply, which is the state `watch.earlier_conversation` already
    marks as awaiting an answer and a resumed session answers first."""
    with_a_transcript(studies_dir)
    CliRunner().invoke(cli.main, ["rerun", "--study", "20261115-090000-ffff"])
    fresh, = [item for item in list_studies(studies_dir)
              if item.cloned_from == "20261115-090000-ffff"]

    rows = [json.loads(line) for line
            in (studies_dir / fresh.study_id / "messages.jsonl").read_text().splitlines()]
    briefing = earlier_conversation(rows)

    assert briefing.unanswered == 1
    assert "Re = 100" in briefing.text


def test_rerun_takes_the_first_thing_said_and_not_the_last(studies_dir):
    """The study's opening request, which is what started it -- not "refine it", which
    only means anything after the first one has been answered."""
    with_a_transcript(studies_dir)

    assert catalog.opening_request(studies_dir, "20261115-090000-ffff").startswith(
        "Find the drag")


def test_rerun_ignores_tool_rows_and_harness_notes_looking_for_the_request(studies_dir):
    byok_study(studies_dir)
    transcript(studies_dir, "20261115-090000-ffff", [
        {"seq": 0, "role": "note", "content": "the workspace is starting"},
        {"seq": 1, "role": "tool", "content": {"tool": "bash"}},
        {"seq": 2, "role": "user", "content": "   "},
        {"seq": 3, "role": "user", "content": "Mesh this duct."},
    ])

    assert catalog.opening_request(studies_dir, "20261115-090000-ffff") == "Mesh this duct."


def test_a_study_whose_request_cannot_be_recovered_is_refused_by_name(studies_dir):
    """Reconstructing something plausible would make a new study claim to be a repeat
    of an old one, which is the one thing a rerun must not be."""
    byok_study(studies_dir, study_id="20261115-090000-ffff")

    result = CliRunner().invoke(cli.main, ["rerun", "--study", "20261115-090000-ffff"])

    assert result.exit_code == 1
    assert "20261115-090000-ffff" in result.output
    assert "nothing to re-issue" in result.output
    assert [item for item in list_studies(studies_dir)] == list_studies(studies_dir)
    assert not [item for item in list_studies(studies_dir) if item.cloned_from]


def test_a_study_with_only_the_agents_words_in_it_is_refused_too(studies_dir):
    byok_study(studies_dir)
    transcript(studies_dir, "20261115-090000-ffff", [
        {"seq": 0, "role": "assistant", "content": "I have meshed the duct."},
    ])

    with pytest.raises(catalog.NoOpeningRequest) as refused:
        catalog.duplicate(studies_dir, "20261115-090000-ffff", rerun=True)

    assert "20261115-090000-ffff" in str(refused.value)


def test_clone_and_rerun_start_nothing(studies_dir, no_workspace):
    with_a_transcript(studies_dir)

    named = ["--study", "20261115-090000-ffff"]
    assert CliRunner().invoke(cli.main, ["clone", *named]).exit_code == 0
    assert CliRunner().invoke(cli.main, ["rerun", *named]).exit_code == 0
    assert no_workspace == []


def test_a_study_that_is_not_here_is_refused_rather_than_cloned_empty(studies_dir):
    with_a_transcript(studies_dir)

    result = CliRunner().invoke(cli.main, ["clone", "--study", "no-such-study"])

    assert result.exit_code == 1
    assert "No study called" in result.output
    assert not (studies_dir / "no-such-study").exists()
