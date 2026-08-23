"""The session wiring: the loop, watch mode, capture and the toolbox, joined up."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from conftest import ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli
from openreynolds.backend.base import BackendError, JobStatus
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.store import Store

pytestmark = pytest.mark.usefixtures("fast_polling")


@pytest.fixture
def loop(ctx, store, console):
    return Loop(Config(anthropic_api_key="k", model="claude-opus-5"), ctx, store, console)


@pytest.fixture
def quiet_console(monkeypatch, console):
    monkeypatch.setattr(cli, "console", console)
    return console


def finishing_jobs(backend, end_reason="completed", exit_code=0):
    """Make every started job report as already finished, so watch mode terminates."""
    original = backend.job_start

    def job_start(cmd, cwd=None, name=None, kill_on=None):
        job_id = original(cmd, cwd=cwd, name=name, kill_on=kill_on)
        backend.logs[job_id] = b"Time = 500\nEnd\n"
        backend.jobs[job_id] = JobStatus(
            job_id=job_id,
            name=name,
            status="exited",
            exit_code=exit_code,
            end_reason=end_reason,
            log_size=len(backend.logs[job_id]),
        )
        return job_id

    backend.job_start = job_start


# -- one-shot ------------------------------------------------------------------


def test_one_shot_waits_for_the_job_it_started(loop, backend, store, quiet_console):
    finishing_jobs(backend)
    fake = install_model(
        loop,
        [
            message(
                [tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})],
                stop_reason="tool_use",
            ),
            message([text_block("launched; I will look when it lands")]),
            message([text_block("it converged")]),
        ],
    )

    cli._run_one_shot(loop, backend, store, "run the elbow")

    assert not store.live_jobs(), "one-shot returns only once no job is left running"
    informed = [m for m in loop.messages if "end_reason=completed" in str(m.get("content"))]
    assert informed, "the model was woken with the job's outcome"
    assert len(fake.calls) >= 3


def test_one_shot_reports_an_expired_sandbox_rather_than_a_clean_exit(
    loop, backend, store, quiet_console
):
    """The volume survives this, so it is the model's call what to do about it."""
    finishing_jobs(backend, end_reason="sandbox_expired", exit_code=None)
    install_model(
        loop,
        [
            message([tool_block("job_start", {"cmd": "simpleFoam"})], stop_reason="tool_use"),
            message([text_block("waiting")]),
            message([text_block("I will restart from latestTime")]),
        ],
    )

    cli._run_one_shot(loop, backend, store, "solve it")

    assert any("sandbox_expired" in str(m.get("content")) for m in loop.messages)


# -- interactive ---------------------------------------------------------------


def test_interactive_runs_a_turn_then_exits(loop, backend, store, quiet_console, monkeypatch):
    monkeypatch.setattr(cli, "LineReader", lambda: ScriptedReader(["how many cells?", "/exit"]))
    install_model(loop, [message([text_block("94,321")])])

    cli._run_interactive(loop, backend, store)

    assert loop.messages[0]["content"] == "how many cells?"


def test_interactive_stops_on_eof(loop, backend, store, quiet_console, monkeypatch):
    monkeypatch.setattr(cli, "LineReader", lambda: ScriptedReader([]))
    install_model(loop, [message([text_block("unused")])])

    cli._run_interactive(loop, backend, store)

    assert loop.messages == []


def test_typing_during_a_job_reaches_the_model(loop, backend, store, quiet_console, monkeypatch):
    backend.job_start("sleep 600", name="solve")
    store.record_job("job-1", cmd="sleep 600", name="solve")
    monkeypatch.setattr(cli, "LineReader", lambda: ScriptedReader(["stop", "/exit"]))
    install_model(loop, [message([text_block("stopping")])])

    cli._run_interactive(loop, backend, store)

    assert loop.messages[0] == {"role": "user", "content": "stop"}


# -- results pickup ------------------------------------------------------------


class RecordingCapture:
    def __init__(self):
        self.results = []
        self.artifacts = []

    def result(self, payload):
        self.results.append(payload)

    def artifact(self, path, kind=None):
        self.artifacts.append(path)


def test_results_are_picked_up_when_the_model_happened_to_leave_some(backend):
    backend.files[cli.RESULTS_PICKUP] = json.dumps({"dp": 29.2, "units": "Pa"}).encode()
    capture = RecordingCapture()

    cli._pickup_results(backend, capture)

    assert capture.results == [{"dp": 29.2, "units": "Pa"}]


def test_no_results_file_is_normal(backend):
    capture = RecordingCapture()
    cli._pickup_results(backend, capture)
    assert capture.results == []


def test_unparseable_results_are_skipped_quietly(backend):
    backend.files[cli.RESULTS_PICKUP] = b"not json at all"
    capture = RecordingCapture()
    cli._pickup_results(backend, capture)
    assert capture.results == []


def test_pickup_without_capture_is_a_no_op(backend):
    backend.files[cli.RESULTS_PICKUP] = b"{}"
    cli._pickup_results(backend, None)  # must not raise


# -- toolbox and fetch ---------------------------------------------------------


def test_toolbox_is_pushed_at_session_start(backend, quiet_console):
    cli._sync_toolbox(backend)
    assert backend.trees, "the toolbox was synced"
    local_dir, remote = backend.trees[0]
    assert remote == cli.TOOLBOX_DEST
    assert (local_dir / "log_digest.py").exists()


def test_a_failed_toolbox_sync_does_not_stop_the_session(backend, quiet_console):
    def refuse(local_dir, remote_dir):
        raise BackendError("disk full", code="bad_request")

    backend.put_tree = refuse
    cli._sync_toolbox(backend)  # must not raise


def test_fetched_files_are_captured_and_offered_to_the_terminal(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(cli.images, "show", lambda path: shown.append(path) or True)
    capture = RecordingCapture()
    png = tmp_path / "u.png"
    png.write_bytes(b"\x89PNG")

    cli._fetch_hook(capture)([png])

    assert capture.artifacts == [png]
    assert shown == [png]


def test_fetch_still_displays_when_capture_is_off(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(cli.images, "show", lambda path: shown.append(path) or True)
    cli._fetch_hook(None)([tmp_path / "u.png"])
    assert shown


# -- commands ------------------------------------------------------------------


def test_missing_configuration_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config()))
    result = CliRunner().invoke(cli.main, [])
    assert result.exit_code == 1
    assert "FOAMD_URL" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_studies_lists_local_studies(tmp_path, monkeypatch):
    root = tmp_path / "studies"
    store = Store(root, "20260823-120000-abcd")
    store.session.title = "elbow"
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))

    result = CliRunner().invoke(cli.main, ["studies"])

    assert result.exit_code == 0
    assert "20260823-120000-abcd" in result.output
    assert "elbow" in result.output
    assert "1 job(s) running" in result.output


def test_studies_with_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Config, "load", classmethod(lambda cls: Config(studies_dir=tmp_path / "none"))
    )
    result = CliRunner().invoke(cli.main, ["studies"])
    assert "No studies under" in result.output


def test_config_writes_credentials_outside_the_repo(tmp_path, monkeypatch):
    target = tmp_path / "config.json"
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(target))
    monkeypatch.delenv("FOAMD_URL", raising=False)
    monkeypatch.delenv("FOAMD_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = CliRunner().invoke(
        cli.main, ["config"], input="https://svc.example/\nof_live_x\nsk-ant-y\nclaude-opus-5\n"
    )

    assert result.exit_code == 0
    saved = json.loads(target.read_text())
    assert saved["foamd_url"] == "https://svc.example"
    assert saved["foamd_api_key"] == "of_live_x"
    assert saved["model"] == "claude-opus-5"


def test_exit_works_while_a_job_is_running(loop, backend, store, quiet_console, monkeypatch):
    """Regression: /exit was only honoured when nothing was running, so a user
    watching a long solve could not quit -- the word went to the model instead."""
    backend.job_start("sleep 600", name="solve")
    store.record_job("job-1", cmd="sleep 600", name="solve")
    monkeypatch.setattr(cli, "LineReader", lambda: ScriptedReader(["/exit"]))
    install_model(loop, [message([text_block("should never be reached")])])

    cli._run_interactive(loop, backend, store)

    assert loop.messages == [], "the exit word never reached the model"
    assert store.live_jobs(), "the job is left running on the instance"
