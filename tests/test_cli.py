"""The session wiring: the loop, watch mode, capture and the toolbox, joined up."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from conftest import ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli
from openreynolds.backend.base import BackendError, ExecResult, JobStatus
from openreynolds.browse import Browser
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.store import Store
from openreynolds.watch import NullReader

pytestmark = pytest.mark.usefixtures("fast_polling")


@pytest.fixture
def loop(ctx, store, view):
    return Loop(Config(anthropic_api_key="k", model="claude-opus-5"), ctx, store, view)


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


def test_one_shot_waits_for_the_job_it_started(loop, backend, store, view, quiet_console):
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

    cli._run_one_shot(loop, backend, store, "run the elbow", view, NullReader())

    assert not store.live_jobs(), "one-shot returns only once no job is left running"
    informed = [m for m in loop.messages if "end_reason=completed" in str(m.get("content"))]
    assert informed, "the model was woken with the job's outcome"
    assert len(fake.calls) >= 3


def test_one_shot_reports_an_expired_sandbox_rather_than_a_clean_exit(
    loop, backend, store, view, quiet_console
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

    cli._run_one_shot(loop, backend, store, "solve it", view, NullReader())

    assert any("sandbox_expired" in str(m.get("content")) for m in loop.messages)


# -- interactive ---------------------------------------------------------------


def test_interactive_runs_a_turn_then_exits(loop, backend, store, view, quiet_console):
    install_model(loop, [message([text_block("94,321")])])

    cli._run_interactive(loop, backend, store, view, Browser(backend, store), ScriptedReader(["how many cells?", "/exit"]))

    assert loop.messages[0]["content"] == "how many cells?"


def test_interactive_stops_on_eof(loop, backend, store, view, quiet_console):
    install_model(loop, [message([text_block("unused")])])

    cli._run_interactive(loop, backend, store, view, Browser(backend, store), ScriptedReader([]))

    assert loop.messages == []


def test_typing_during_a_job_reaches_the_model(loop, backend, store, view, quiet_console):
    backend.job_start("sleep 600", name="solve")
    store.record_job("job-1", cmd="sleep 600", name="solve")
    install_model(loop, [message([text_block("stopping")])])

    cli._run_interactive(loop, backend, store, view, Browser(backend, store), ScriptedReader(["stop", "/exit"]))

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


HOME = "/work/20260824-120000-abcd"


def test_results_are_picked_up_from_the_study_s_own_directory(backend):
    backend.files[f"{HOME}/{cli.RESULTS_FILE}"] = json.dumps({"dp": 29.2, "units": "Pa"}).encode()
    capture = RecordingCapture()

    cli._pickup_results(backend, capture, HOME)

    assert capture.results == [{"dp": 29.2, "units": "Pa"}]


def test_another_study_s_results_are_not_picked_up_as_this_one_s(backend):
    """Every study used to write to the same path, so the last one to leave a file
    got credited with whatever the previous one had left there."""
    backend.files["/work/somebody-else/results.json"] = json.dumps({"dp": 1.0}).encode()
    capture = RecordingCapture()

    cli._pickup_results(backend, capture, HOME)

    assert capture.results == []


def test_no_results_file_is_normal(backend):
    capture = RecordingCapture()
    cli._pickup_results(backend, capture, HOME)
    assert capture.results == []


def test_unparseable_results_are_skipped_quietly(backend):
    backend.files[f"{HOME}/{cli.RESULTS_FILE}"] = b"not json at all"
    capture = RecordingCapture()
    cli._pickup_results(backend, capture, HOME)
    assert capture.results == []


def test_pickup_without_capture_is_a_no_op(backend):
    backend.files[f"{HOME}/{cli.RESULTS_FILE}"] = b"{}"
    cli._pickup_results(backend, None, HOME)  # must not raise


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
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
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


def test_exit_works_while_a_job_is_running(loop, backend, store, view, quiet_console):
    """Regression: /exit was only honoured when nothing was running, so a user
    watching a long solve could not quit -- the word went to the model instead."""
    backend.job_start("sleep 600", name="solve")
    store.record_job("job-1", cmd="sleep 600", name="solve")
    install_model(loop, [message([text_block("should never be reached")])])

    cli._run_interactive(loop, backend, store, view, Browser(backend, store), ScriptedReader(["/exit"]))

    assert loop.messages == [], "the exit word never reached the model"
    assert store.live_jobs(), "the job is left running on the instance"


def _api_error(status=429):
    import anthropic
    from types import SimpleNamespace

    return anthropic.RateLimitError(
        "rate limit exceeded",
        response=SimpleNamespace(status_code=status, headers={}, request=None),
        body=None,
    )


def test_a_rate_limit_does_not_end_the_session(loop, backend, store, view, quiet_console, monkeypatch):
    """A long study meets one eventually; losing the thread to it is a poor trade."""
    calls = []

    def failing_run():
        calls.append(1)
        raise _api_error()

    monkeypatch.setattr(loop, "run", failing_run)
    cli._run_interactive(loop, backend, store, view, Browser(backend, store), ScriptedReader(["go", "/exit"]))

    assert calls, "the turn was attempted"
    assert loop.messages[0]["content"] == "go", "the thread survived"


def test_a_failed_turn_does_not_trigger_a_context_refresh(loop, view, monkeypatch):
    refreshed = []
    monkeypatch.setattr(loop, "run", lambda: (_ for _ in ()).throw(_api_error()))
    monkeypatch.setattr(loop, "refresh", lambda blurb: refreshed.append(blurb))
    monkeypatch.setattr(type(loop), "needs_refresh", property(lambda self: True))

    assert cli._run_turn(loop, view) is False
    assert refreshed == []


def test_one_shot_stops_when_the_model_api_fails(loop, backend, store, view, quiet_console, monkeypatch):
    monkeypatch.setattr(loop, "run", lambda: (_ for _ in ()).throw(_api_error()))
    cli._run_one_shot(loop, backend, store, "do it", view, NullReader())  # returns rather than hanging


# -- doctor --------------------------------------------------------------------


def full_config(**over):
    values = dict(
        foamd_url="https://svc.example",
        foamd_api_key="of_live_abcdefghijklmnop",
        anthropic_api_key="sk-ant-secret-value",
        model="claude-opus-5",
    )
    values.update(over)
    return Config(**values)


def stub_service(monkeypatch, instances=None, error=None, study_error=None):
    class Client:
        def list_instances(self):
            if error:
                raise error
            return instances or []

        def create_study(self, title, instance_id):
            if study_error:
                raise study_error
            return "stud-1234abcd"

        def close(self):
            pass

    monkeypatch.setattr(cli.hosted, "FoamdClient", lambda url, key: Client())


def stub_model(monkeypatch, tokens=8, error=None):
    from types import SimpleNamespace

    class Messages:
        def count_tokens(self, **kwargs):
            if error:
                raise error
            return SimpleNamespace(input_tokens=tokens)

    monkeypatch.setattr(
        cli.anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=Messages())
    )


def labels(results):
    return {label: (ok, detail) for label, ok, detail in results}


def test_doctor_reports_every_check(monkeypatch):
    stub_service(monkeypatch, instances=[{"id": "abcdefgh1234", "status": "running"}])
    stub_model(monkeypatch)

    results = labels(cli.run_checks(full_config()))

    assert all(ok for ok, _ in results.values())
    assert set(results) == {
        "settings",
        "workspace service",
        "model API",
        "capture",
        "toolbox",
        "terminal",
    }
    assert "1 instance(s)" in results["workspace service"][1]
    assert "claude-opus-5 reachable" in results["model API"][1]


def test_doctor_never_prints_a_key_in_full(monkeypatch):
    stub_service(monkeypatch)
    stub_model(monkeypatch)
    cfg = full_config()

    rendered = " ".join(f"{label} {detail}" for label, _, detail in cli.run_checks(cfg))

    assert cfg.foamd_api_key not in rendered
    assert cfg.anthropic_api_key not in rendered
    assert "of_live_abcd" in rendered, "enough of the prefix to identify which key"


def test_doctor_names_a_bad_service_key(monkeypatch):
    stub_service(monkeypatch, error=BackendError("invalid key", code="unauthorized", status=401))
    stub_model(monkeypatch)

    ok, detail = labels(cli.run_checks(full_config()))["workspace service"]

    assert ok is False
    assert "unauthorized" in detail


def test_doctor_names_a_bad_model_key(monkeypatch):
    import anthropic
    from types import SimpleNamespace

    stub_service(monkeypatch)
    stub_model(
        monkeypatch,
        error=anthropic.AuthenticationError(
            "invalid x-api-key",
            response=SimpleNamespace(status_code=401, headers={}, request=None),
            body=None,
        ),
    )

    ok, detail = labels(cli.run_checks(full_config()))["model API"]

    assert ok is False
    assert "401" in detail


def test_doctor_reports_an_empty_but_reachable_service(monkeypatch):
    stub_service(monkeypatch, instances=[])
    stub_model(monkeypatch)
    ok, detail = labels(cli.run_checks(full_config()))["workspace service"]
    assert ok and "no instance yet" in detail


def test_doctor_ignores_deleted_instances(monkeypatch):
    stub_service(monkeypatch, instances=[{"id": "x", "status": "deleted"}])
    stub_model(monkeypatch)
    assert "no instance yet" in labels(cli.run_checks(full_config()))["workspace service"][1]


def test_doctor_flags_missing_settings(monkeypatch):
    stub_service(monkeypatch)
    stub_model(monkeypatch)
    results = cli.run_checks(Config())
    settings = results[0]
    assert settings[1] is False
    assert "FOAMD_URL" in settings[0]


def test_doctor_exits_nonzero_when_something_is_wrong(monkeypatch):
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config()))
    stub_service(monkeypatch)
    stub_model(monkeypatch)

    result = CliRunner().invoke(cli.main, ["doctor"])

    assert result.exit_code == 1
    assert "check(s) failed" in result.output


def test_doctor_exits_zero_when_ready(monkeypatch):
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: full_config()))
    stub_service(monkeypatch, instances=[{"id": "abcdefgh", "status": "stopped"}])
    stub_model(monkeypatch)

    result = CliRunner().invoke(cli.main, ["doctor"])

    assert result.exit_code == 0
    assert "Ready" in result.output


def test_doctor_finds_the_toolbox_it_would_sync(monkeypatch):
    stub_service(monkeypatch)
    stub_model(monkeypatch)
    ok, detail = labels(cli.run_checks(full_config()))["toolbox"]
    assert ok
    scripts = len(list(cli.TOOLBOX_SOURCE.glob("*.py")))
    assert f"{scripts} scripts" in detail and "3 notes" in detail


def test_config_without_a_terminal_explains_instead_of_aborting(monkeypatch, tmp_path):
    """Through a non-interactive channel click just says 'Aborted!', which tells
    the user nothing."""
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    result = CliRunner().invoke(cli.main, ["config"], input="")
    assert result.exit_code == 1
    assert "could not prompt here" in result.output
    assert "--key-file" in result.output


def test_config_key_file_never_echoes_the_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    secret = tmp_path / "key.txt"
    secret.write_text("sk-ant-super-secret-value\n")

    result = CliRunner().invoke(cli.main, ["config", "--key-file", str(secret)])

    assert result.exit_code == 0
    assert "sk-ant-super-secret-value" not in result.output
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["anthropic_api_key"] == "sk-ant-super-secret-value"


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.setenv("FOAMD_URL", "https://svc.example")
    monkeypatch.setenv("FOAMD_API_KEY", "of_live_env")

    result = CliRunner().invoke(cli.main, ["config", "--from-env"])

    assert result.exit_code == 0
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["anthropic_api_key"] == "sk-ant-from-env"
    assert saved["foamd_url"] == "https://svc.example"


def test_config_explains_when_a_shell_claims_a_terminal_then_sends_eof(monkeypatch, tmp_path):
    """Git Bash reports isatty() true for a redirected stdin and then delivers EOF,
    so the pre-check passes and click aborts. The abort has to explain itself."""
    import click as _click

    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
    monkeypatch.setattr(cli.click, "prompt", lambda *a, **k: (_ for _ in ()).throw(_click.Abort()))

    result = CliRunner().invoke(cli.main, ["config"])

    assert result.exit_code == 1
    assert "could not prompt here" in result.output
    assert "--key-file" in result.output
    assert "Aborted" not in result.output.split("could not prompt")[0][-40:]


def test_unattended_waiting_can_be_bounded(loop, backend, store, view, quiet_console):
    """A -p run whose model ends its turn asking a question has nobody to answer it,
    so it waits on the job -- for hours. Seen live: parked over an hour."""
    backend.job_start("sleep 100000", name="sweep")
    store.record_job("job-1", cmd="sleep 100000", name="sweep")
    install_model(loop, [message([text_block("which outlet length do you want?")])])

    started = __import__("time").monotonic()
    cli._run_one_shot(loop, backend, store, "run it", view, NullReader(), max_wait_minutes=0.01)
    elapsed = __import__("time").monotonic() - started

    assert elapsed < 20, "it gave up rather than waiting on the job"
    assert store.live_jobs(), "and left the job running on the instance"
    assert any("stopped waiting" in i for i in view.infos)
    assert any(store.session.study_id in i for i in view.infos), "it says how to resume"


def test_without_a_bound_it_still_waits_for_the_job(loop, backend, store, view, quiet_console):
    """The default is unchanged: closing the laptop on a long solve is the point."""
    finishing_jobs(backend)
    install_model(
        loop,
        [
            message([tool_block("job_start", {"cmd": "simpleFoam"})], stop_reason="tool_use"),
            message([text_block("waiting")]),
            message([text_block("done")]),
        ],
    )

    cli._run_one_shot(loop, backend, store, "solve", view, NullReader())

    assert not store.live_jobs()


# -- looking at the workspace --------------------------------------------------


def test_a_workspace_path_survives_a_posix_emulating_shell():
    """Git Bash rewrites a leading /work before this process sees the argument, so a
    correct command comes back as a 404 naming a path nobody typed."""
    assert cli.workspace_path("C:/Program Files/Git/work/case/log") == "/work/case/log"
    assert cli.workspace_path(r"C:\Program Files\Git\work\case") == "/work/case"
    assert cli.workspace_path("C:/Program Files/Git/work") == "/work"


def test_a_real_workspace_path_is_left_exactly_as_it_is():
    for path in ("/work", "/work/case/log", "", "constant/triSurface"):
        assert cli.workspace_path(path) == path


def test_a_local_windows_path_that_is_not_a_workspace_path_is_untouched():
    assert cli.workspace_path("C:/Users/me/notes.md") == "C:/Users/me/notes.md"


# -- what the session is told about the workspace -------------------------------


def workspace_listing(backend, *paths):
    """Make the workspace answer a listing with these directories in it."""
    lines = "".join(f"d\t4096\t1700000000.0\t{path}\n" for path in paths)
    backend.exec_result = ExecResult(0, lines, False, None)


def test_a_new_study_gets_its_own_directory(backend, store):
    """Opening a new study straight into the shared volume, among every other study's
    cases, is not a clean slate by any reading of the words."""
    home = cli._home_for(store, backend, resuming=False)

    assert home == f"/work/{store.session.study_id}"
    assert backend.last_exec[0] == f"mkdir -p {home}", "and it is made before use"


def test_a_study_keeps_the_directory_it_was_given(backend, store):
    store.session.home = "/work/20260101-000000-aaaa"
    assert cli._home_for(store, backend, resuming=True) == "/work/20260101-000000-aaaa"


def test_a_study_from_before_homes_existed_keeps_the_whole_workspace(backend, store):
    """Moving their files out from under them would be worse than the untidiness."""
    assert cli._home_for(store, backend, resuming=True) == "/work"


def test_a_directory_that_cannot_be_made_falls_back_rather_than_failing(
    backend, store, quiet_console
):
    def refuse(cmd, cwd=None, timeout_s=120):
        raise BackendError("read-only volume", code="bad_request")

    backend.exec = refuse
    assert cli._home_for(store, backend, resuming=False) == "/work"


def test_the_brief_names_the_study_s_own_directory(backend, store):
    workspace_listing(backend, "/work/s1/elbow")
    store.session.home = "/work/s1"

    brief = cli._situation_brief(
        store, backend, resuming=False, interactive=True, browser=Browser(backend, store)
    )

    assert "Your directory is /work/s1" in brief
    assert "elbow" in brief
    assert "not written for this request" in brief


def test_the_brief_says_so_when_the_study_directory_is_empty(backend, store):
    workspace_listing(backend)
    store.session.home = "/work/s1"
    brief = cli._situation_brief(
        store, backend, resuming=False, interactive=True, browser=Browser(backend, store)
    )
    assert "It is empty" in brief


def test_the_rest_of_the_volume_is_mentioned_once_not_listed(backend, store):
    """Other sessions' work is readable if wanted, and not this study's business.

    Named and counted rather than listed: what those directories are is worth one
    sentence, and their contents are somebody else's question.
    """
    workspace_listing(backend)
    store.session.home = "/work/s1"
    brief = cli._situation_brief(
        store, backend, resuming=False, interactive=True, browser=Browser(backend, store)
    )
    assert "this tool's own earlier sessions" in brief
    assert "was written for this request" in brief


def test_a_resumed_session_is_told_the_workspace_is_its_own(backend, store):
    workspace_listing(backend, "/work/elbow")
    brief = cli._situation_brief(
        store, backend, resuming=True, interactive=True, browser=Browser(backend, store)
    )
    assert "as this study left it" in brief
    assert "none of it was made by this study" not in brief


def test_infrastructure_directories_are_not_listed_as_someone_else_s_work(backend, store):
    """`.toolbox` and the service's own directory are described in the prompt already;
    listing them here would bury the two names that matter."""
    workspace_listing(backend, "/work/.toolbox", "/work/.foamd", "/work/elbow")
    brief = cli._situation_brief(
        store, backend, resuming=False, interactive=True, browser=Browser(backend, store)
    )
    assert "toolbox" not in brief and "foamd" not in brief
    assert "elbow" in brief


def test_a_workspace_that_cannot_be_listed_does_not_stop_the_session(backend, store):
    def broken(cmd, cwd=None, timeout_s=120):
        raise BackendError("instance is not up", code="unavailable")

    backend.exec = broken
    brief = cli._situation_brief(
        store, backend, resuming=False, interactive=True, browser=Browser(backend, store)
    )
    assert brief, "the rest of the brief still arrives"


def test_leaving_says_where_the_study_s_files_are(store, backend, monkeypatch):
    """Knowing a study has a directory of its own is no use without being told which
    one it is."""
    import io

    from rich.console import Console

    written = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=written, width=200))
    store.session.home = "/work/20260824-120000-abcd"

    cli._report_on_exit(backend, store)

    assert "/work/20260824-120000-abcd" in written.getvalue()


# -- doctor tells you when capture has quietly stopped --------------------------


def test_doctor_reports_capture_reaching_the_platform(monkeypatch):
    """Capture fails quietly on purpose, so nothing else would ever tell you."""
    stub_service(monkeypatch)
    stub_model(monkeypatch)

    ok, detail = labels(cli.run_checks(full_config()))["capture"]

    assert ok
    assert "transcripts are being kept" in detail


def test_doctor_says_when_capture_is_broken(monkeypatch):
    stub_service(
        monkeypatch,
        study_error=BackendError("study quota reached", code="quota", status=402),
    )
    stub_model(monkeypatch)

    ok, detail = labels(cli.run_checks(full_config()))["capture"]

    assert not ok
    assert "quota reached" in detail


def test_doctor_does_not_call_the_platform_when_capture_is_off(monkeypatch):
    """Checking a thing that is switched off would report it broken for doing what
    it was told."""
    stub_service(
        monkeypatch,
        study_error=AssertionError("capture is off; nothing should have been opened"),
    )
    stub_model(monkeypatch)
    cfg = full_config()
    cfg.capture = False

    ok, detail = labels(cli.run_checks(cfg))["capture"]

    assert ok
    assert "off for this configuration" in detail
