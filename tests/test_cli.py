"""The session wiring: the loop, watch mode, capture and the toolbox, joined up."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from conftest import FakeBackend, ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli
from openreynolds.backend.base import BackendError, ExecResult, JobStatus
from openreynolds.browse import Browser
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.store import Store
from openreynolds.watch import NullReader, Wake

pytestmark = pytest.mark.usefixtures("fast_polling")


@pytest.fixture
def loop(ctx, store, view):
    return Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view)


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
    assert "FOAMD_API_KEY" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "openreynolds login" in result.output


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
        cli.main, ["config"], input="https://svc.example/\nof_live_x\nzai\nsk-zai-y\nglm-4.6\n"
    )

    assert result.exit_code == 0
    saved = json.loads(target.read_text())
    assert saved["foamd_url"] == "https://svc.example"
    assert saved["foamd_api_key"] == "of_live_x"
    assert saved["model"] == "glm-4.6"
    assert saved["provider"] == "zai"
    assert saved["llm_api_key"] == "sk-zai-y"
    assert saved["context_window"] == 200_000


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
    from openreynolds.llm import ProviderError

    return ProviderError("rate limit exceeded", status)


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
    outcome = cli._run_one_shot(loop, backend, store, "do it", view, NullReader())
    assert outcome == "failed"  # and it returned rather than hanging


# -- doctor --------------------------------------------------------------------


def full_config(**over):
    values = dict(
        foamd_url="https://svc.example",
        foamd_api_key="of_live_abcdefghijklmnop",
        llm_api_key="sk-ant-secret-value",
        model="claude-opus-5",
    )
    values.update(over)
    return Config(**values)


def stub_service(monkeypatch, instances=None, error=None):
    class Client:
        def list_instances(self):
            if error:
                raise error
            return instances or []

        def create_study(self, title, instance_id):
            raise AssertionError("doctor opened a study; it is meant to change nothing")

        def post_messages(self, study_id, messages):
            raise AssertionError("doctor posted a message; it is meant to change nothing")

        def close(self):
            pass

    made = []

    def make(url, key):
        made.append(Client())
        return made[-1]

    monkeypatch.setattr(cli.hosted, "FoamdClient", make)
    return made


def stub_model(monkeypatch, tokens=8, error=None):
    from types import SimpleNamespace

    class Messages:
        def count_tokens(self, **kwargs):
            if error:
                raise error
            return SimpleNamespace(input_tokens=tokens)

    from openreynolds.llm import anthropic_api

    monkeypatch.setattr(
        anthropic_api.anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=Messages())
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
        "video assembly",
    }
    assert "1 instance(s)" in results["workspace service"][1]
    assert "claude-opus-5 reachable" in results["model API"][1]


def test_doctor_never_prints_a_key_in_full(monkeypatch):
    stub_service(monkeypatch)
    stub_model(monkeypatch)
    cfg = full_config()

    rendered = " ".join(f"{label} {detail}" for label, _, detail in cli.run_checks(cfg))

    assert cfg.foamd_api_key not in rendered
    assert cfg.llm_api_key not in rendered
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
    assert "FOAMD_API_KEY" in settings[0]


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
    assert saved["llm_api_key"] == "sk-ant-super-secret-value"


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.setenv("FOAMD_URL", "https://svc.example")
    monkeypatch.setenv("FOAMD_API_KEY", "of_live_env")

    result = CliRunner().invoke(cli.main, ["config", "--from-env"])

    assert result.exit_code == 0
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["llm_api_key"] == "sk-ant-from-env"
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
    outcome = cli._run_one_shot(
        loop, backend, store, "run it", view, NullReader(), max_wait_minutes=0.01
    )
    elapsed = __import__("time").monotonic() - started

    assert elapsed < 20, "it gave up rather than waiting on the job"
    assert outcome == "timeout"
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

    outcome = cli._run_one_shot(loop, backend, store, "solve", view, NullReader())

    assert not store.live_jobs()
    assert outcome == "ok"


# -- what a -p run tells the shell that started it -----------------------------


def _headless(monkeypatch, outcome):
    """A `-p` invocation whose session ended the given way, and nothing else."""
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: full_config()))
    seen = {}

    def session(cfg, **kwargs):
        seen.update(kwargs)
        return outcome

    monkeypatch.setattr(cli, "session", session)
    return seen


@pytest.mark.parametrize(
    "outcome,code",
    [("ok", 0), ("failed", 1), ("timeout", 2)],
)
def test_a_headless_run_s_exit_code_says_how_it_ended(monkeypatch, outcome, code):
    """Every ending used to exit 0, so a scheduled run whose every turn was refused
    by a rate limit looked, to whatever started it, like one that had finished."""
    seen = _headless(monkeypatch, outcome)

    result = CliRunner().invoke(cli.main, ["-p", "run the elbow", "--max-wait", "5"])

    assert result.exit_code == code, result.output
    assert seen["one_shot"] == "run the elbow"
    assert seen["max_wait"] == 5.0


def test_an_interactive_session_still_exits_zero(monkeypatch):
    """Whatever happened in it was said on screen to someone."""
    _headless(monkeypatch, None)

    result = CliRunner().invoke(cli.main, ["--plain"])

    assert result.exit_code == 0, result.output


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
    assert "yours is the first" in brief


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
    assert "transcripts will be kept" in detail


def test_doctor_opens_nothing_to_check_capture(monkeypatch):
    """Every `doctor` run used to leave a study called "openreynolds doctor" on the
    platform -- the one command sold as changing nothing, changing something each
    time it ran. The stub raises on any write, so a check that writes fails here."""
    stub_service(monkeypatch, instances=[{"id": "abcdefgh1234", "status": "stopped"}])
    stub_model(monkeypatch)

    ok, detail = labels(cli.run_checks(full_config()))["capture"]

    assert ok, detail  # a write would have raised, and been reported as a failure


def test_doctor_says_when_capture_is_broken(monkeypatch):
    stub_service(
        monkeypatch,
        error=BackendError("key revoked", code="unauthorized", status=401),
    )
    stub_model(monkeypatch)

    ok, detail = labels(cli.run_checks(full_config()))["capture"]

    assert not ok
    assert "key revoked" in detail


def test_doctor_does_not_call_the_platform_when_capture_is_off(monkeypatch):
    """Checking a thing that is switched off would report it broken for doing what
    it was told."""
    made = stub_service(monkeypatch)
    stub_model(monkeypatch)
    cfg = full_config()
    cfg.capture = False

    ok, detail = labels(cli.run_checks(cfg))["capture"]

    assert ok
    assert "off for this configuration" in detail
    assert len(made) == 1, "only the service check spoke to the platform"


# -- the instance lives exactly as long as the session --------------------------


class Stoppable(FakeBackend):
    """A backend that records being put down, the way the hosted one is."""

    def __init__(self, already_running=False):
        super().__init__()
        self.was_already_running = already_running
        self.stopped = 0

    def shutdown(self):
        self.stopped += 1


def test_ending_a_session_stops_the_jobs_and_the_instance(store, quiet_console):
    """A container left running is a container being paid for, and the only thing
    that knows the session is over is the session."""
    backend = Stoppable()
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    cli._close_down(backend, store)

    assert backend.stopped == 1, "the instance was put down"
    assert not store.live_jobs(), "and the work was stopped first"


def test_keep_alive_leaves_it_up_and_says_what_it_is_choosing(store, quiet_console, capsys):
    backend = Stoppable()
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    cli._close_down(backend, store, keep_alive=True)

    assert backend.stopped == 0
    assert store.live_jobs(), "the job is still going, which is the point of the flag"


def test_a_borrowed_instance_is_handed_back(quiet_console):
    """Listing files lazy-starts a container as a side effect of asking a question.
    Leaving it up afterwards is fifteen minutes of somebody's bill per question."""
    backend = Stoppable(already_running=False)
    cli._release(backend)
    assert backend.stopped == 1


def test_an_instance_somebody_else_started_is_left_alone(quiet_console):
    """It belongs to whoever is using it, and a session may well be running on it."""
    backend = Stoppable(already_running=True)
    cli._release(backend)
    assert backend.stopped == 0


def test_a_session_does_not_stop_a_workspace_it_merely_joined(store, monkeypatch):
    """The session path never asked the question `_release` has always asked.

    An account is capped at one instance and `acquire()` joins the existing one, so an
    ordinary `/exit` in a second terminal stopped the container out from under the first
    one's solve. `_release` forty lines away already had the rule."""
    import io as _io

    from rich.console import Console

    said = _io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=said, force_terminal=False, width=100))
    backend = Stoppable(already_running=True)
    cli._close_down(backend, store)

    assert backend.stopped == 0, "it belongs to whoever started it"
    assert "left up" in said.getvalue(), "and the person is told why it is still running"


def test_a_session_that_started_the_workspace_still_puts_it_down(store, quiet_console):
    backend = Stoppable(already_running=False)
    cli._close_down(backend, store)
    assert backend.stopped == 1


def test_the_exit_sweep_is_scoped_to_this_study(store, quiet_console):
    """`_close_down` used to pass force=True, which is the instance-wide pkill by name."""
    import inspect

    source = inspect.getsource(cli._close_down)
    assert "home=home" in source, "the sweep is anchored on this study's directory"
    assert "force=True" not in source, "the ordinary exit path may not force an unscoped kill"


def test_a_backend_that_cannot_be_stopped_does_not_break_the_exit(store, quiet_console):
    class Stubborn(Stoppable):
        def shutdown(self):
            raise BackendError("instance already gone", code="not_found", status=404)

    cli._close_down(Stubborn(), store)  # must not raise


def test_the_files_come_home_before_anything_is_stopped(store):
    """Order matters more than either step: stopping first would mean mirroring from
    a container that is no longer there."""
    import inspect

    source = inspect.getsource(cli.session)
    assert source.index("_final_sync(") < source.index("_close_down("), (
        "the study is mirrored before the instance is put down"
    )


# -- the bar and the reply -----------------------------------------------------


def test_the_turn_end_sync_never_stands_between_the_user_and_the_model(
    loop, backend, store, view, quiet_console
):
    """A sync that blocks the session thread blocks the reading of what was typed.
    The interactive loop asks the mirror to catch up and moves on."""

    class Live:
        def __init__(self):
            self.caught_up = 0

        def catch_up(self):
            self.caught_up += 1

        def sync_now(self):
            raise AssertionError("the session thread must not wait on a sync")

    live = Live()
    install_model(loop, [message([text_block("done")])])

    cli._run_interactive(
        loop, backend, store, view, Browser(backend, store),
        ScriptedReader(["hi", "/exit"]), live=live,
    )

    assert live.caught_up == 1


def test_status_shows_the_same_headline_as_the_bar(loop, backend, store, view, quiet_console):
    from openreynolds import commands
    from openreynolds.progress import Tracker

    tracker = Tracker(view)
    tracker.begin("tool", "bash", cmd="snappyHexMesh", cwd="/work/study-test")

    cli._local(commands.parse("/status"), view, Browser(backend, store), store, loop, tracker)

    joined = "\n".join(view.statuses[-1])
    assert "right now: bash: snappyHexMesh" in joined


def test_the_interactive_loop_tells_the_bar_when_it_is_waiting(
    loop, backend, store, view, quiet_console
):
    class Bar:
        def __init__(self):
            self.kinds = []

        def begin(self, kind, label="", **facts):
            self.kinds.append(kind)

        def idle(self):
            self.kinds.append("idle")

        def snapshot(self):
            from openreynolds.progress import Progress

            return Progress()

        def refresh_jobs(self, force=False):
            return []

        def facts_for_wake(self):
            return []

    bar = Bar()
    install_model(loop, [message([text_block("done")])])

    cli._run_interactive(
        loop, backend, store, view, Browser(backend, store),
        ScriptedReader(["hi", "/exit"]), progress=bar,
    )

    assert bar.kinds[0] == "waiting"


# -- the front desk ------------------------------------------------------------


class FakeDesk:
    def __init__(self):
        self.asked = []
        self.working_states = []

    def ask(self, text):
        self.asked.append(text)

    def working(self, yes=True):
        self.working_states.append(yes)


def test_a_message_typed_mid_turn_goes_to_both_the_model_and_the_desk(
    loop, backend, store, view
):
    """The screenshot bug: mid-turn text reached the model (eventually) but nothing
    answered now. It must now also reach the desk."""
    desk = FakeDesk()
    reader = ScriptedReader(["how long will this take?"])

    handed = cli._typed_while_working(loop, view, Browser(backend, store), store, reader, concierge=desk)

    assert handed == "how long will this take?"  # still goes to the model
    assert desk.asked == ["how long will this take?"]  # and to the desk, now


def test_a_local_command_mid_turn_does_not_reach_the_desk(loop, backend, store, view):
    desk = FakeDesk()
    reader = ScriptedReader(["/status"])

    cli._typed_while_working(loop, view, Browser(backend, store), store, reader, concierge=desk)

    assert desk.asked == []  # /status is answered locally; the desk is for questions


def test_typing_during_a_job_reaches_the_desk_too(loop, backend, store, view, quiet_console):
    backend.job_start("sleep 600", name="solve")
    store.record_job("job-1", cmd="sleep 600", name="solve")
    install_model(loop, [message([text_block("stopping")])])
    desk = FakeDesk()

    cli._run_interactive(
        loop, backend, store, view, Browser(backend, store),
        ScriptedReader(["what is it doing?", "/exit"]), concierge=desk,
    )

    assert "what is it doing?" in desk.asked


# -- push (upload) -------------------------------------------------------------


def _study_for_push(tmp_path):
    studies = tmp_path / "studies"
    st = Store(studies, "study-x")
    st.session.instance_id = "iid-1"
    st.session.home = "/work/study-x"
    st.save()
    return studies


def _wire_push(monkeypatch, studies, backend):
    monkeypatch.setattr(cli.hosted, "acquire", lambda url, key, iid: (backend, None, "iid-1"))
    monkeypatch.setattr(
        cli.Config, "load",
        classmethod(lambda cls: Config(foamd_url="u", foamd_api_key="k", studies_dir=studies)),
    )


def test_push_uploads_a_directory_keeping_its_name(tmp_path, monkeypatch):
    studies = _study_for_push(tmp_path)
    backend = FakeBackend()
    _wire_push(monkeypatch, studies, backend)
    local = tmp_path / "mycase"
    (local / "system").mkdir(parents=True)
    (local / "system" / "controlDict").write_text("x")

    result = CliRunner().invoke(cli.main, ["push", str(local), "--study", "study-x"])

    assert result.exit_code == 0, result.output
    assert backend.trees, "put_tree was called"
    _local_dir, remote = backend.trees[-1]
    assert remote == "/work/study-x/mycase", "a directory keeps its name under the study home"


def test_push_uploads_a_single_file_into_the_study_home(tmp_path, monkeypatch):
    studies = _study_for_push(tmp_path)
    backend = FakeBackend()
    _wire_push(monkeypatch, studies, backend)
    geom = tmp_path / "geom.stl"
    geom.write_text("solid")

    result = CliRunner().invoke(cli.main, ["push", str(geom), "--study", "study-x"])

    assert result.exit_code == 0, result.output
    assert "/work/study-x/geom.stl" in backend.files


def test_push_honours_an_explicit_destination(tmp_path, monkeypatch):
    studies = _study_for_push(tmp_path)
    backend = FakeBackend()
    _wire_push(monkeypatch, studies, backend)
    geom = tmp_path / "geom.stl"
    geom.write_text("solid")

    result = CliRunner().invoke(
        cli.main,
        ["push", str(geom), "--to", "/work/study-x/constant/triSurface", "--study", "study-x"],
    )

    assert result.exit_code == 0, result.output
    assert "/work/study-x/constant/triSurface/geom.stl" in backend.files


def test_push_needs_a_path_that_exists(tmp_path, monkeypatch):
    studies = _study_for_push(tmp_path)
    _wire_push(monkeypatch, studies, FakeBackend())

    result = CliRunner().invoke(cli.main, ["push", str(tmp_path / "nope"), "--study", "study-x"])

    assert result.exit_code != 0


def test_repeated_api_failures_escalate_to_a_plain_explanation(loop, view, monkeypatch):
    """After two failures in a row, stop repeating 'the thread is intact' and say
    what is happening and how to recover -- the frustration a live session hit."""
    import io as _io
    from rich.console import Console

    buf = _io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=200))
    monkeypatch.setattr(loop, "run", lambda: (_ for _ in ()).throw(_api_error()))

    assert cli._run_turn(loop, view) is False
    assert loop.api_failures == 1
    assert cli._run_turn(loop, view) is False
    assert loop.api_failures == 2

    out = buf.getvalue()
    assert "failed 2 times" in out
    assert f"--study {loop.store.session.study_id}" in out


def test_a_completed_turn_resets_the_failure_count(loop, view, monkeypatch):
    loop.api_failures = 3
    monkeypatch.setattr(loop, "run", lambda: None)
    assert cli._run_turn(loop, view) is True
    assert loop.api_failures == 0


def test_a_failed_prompt_turn_reaches_the_desk(loop, backend, store, view, quiet_console, monkeypatch):
    """When a turn typed at the prompt fails, the desk still answers -- the reason
    someone types 'are you still working?' is that the agent went quiet."""
    monkeypatch.setattr(loop, "run", lambda: (_ for _ in ()).throw(_api_error()))
    desk = FakeDesk()

    cli._run_interactive(
        loop, backend, store, view, Browser(backend, store),
        ScriptedReader(["are you still working?", "/exit"]), concierge=desk,
    )

    assert "are you still working?" in desk.asked


def test_an_empty_key_is_not_reported_as_set():
    assert cli._redact("") == ""
    assert cli._redact("short") == "set"
    assert cli._redact("of_live_abcdefghijklmnop") == "of_live_abcd..."


# -- login -------------------------------------------------------------------------


def test_login_waits_for_approval_then_saves_the_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.delenv("FOAMD_API_KEY", raising=False)
    monkeypatch.delenv("FOAMD_URL", raising=False)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    polls = []

    def code(url, name):
        assert url == "https://api.tryreynolds.com"
        return {"device_code": "dc", "user_code": "K7QF-M2ZR", "verification_url": "https://app.tryreynolds.com/cli", "expires_in": 600, "interval": 1}

    def token(url, device):
        polls.append(device)
        if len(polls) < 3:
            return None
        return {"api_key": "of_live_newkey", "name": "laptop", "base_url": "https://api.tryreynolds.com"}

    monkeypatch.setattr(cli.hosted, "device_code", code)
    monkeypatch.setattr(cli.hosted, "device_token", token)

    result = CliRunner().invoke(cli.main, ["login", "--browser", "--no-browser", "--name", "laptop"])

    assert result.exit_code == 0, result.output
    assert "K7QF-M2ZR" in result.output and "app.tryreynolds.com/cli" in result.output
    assert polls == ["dc", "dc", "dc"]
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["foamd_api_key"] == "of_live_newkey"
    assert saved["foamd_url"] == "https://api.tryreynolds.com"
    assert "of_live_newkey" not in result.output


def test_login_gives_up_when_the_code_expires(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.hosted, "device_code", lambda url, name: {"device_code": "dc", "user_code": "X", "expires_in": 0, "interval": 1})
    monkeypatch.setattr(cli.hosted, "device_token", lambda url, device: None)

    result = CliRunner().invoke(cli.main, ["login", "--browser", "--no-browser"])

    assert result.exit_code == 1
    assert "expired" in result.output
    assert not (tmp_path / "c.json").exists()


def test_login_reports_a_service_that_cannot_be_reached(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))

    def down(url, name):
        raise BackendError("cannot reach the service: refused", code="unreachable")

    monkeypatch.setattr(cli.hosted, "device_code", down)
    result = CliRunner().invoke(cli.main, ["login", "--browser", "--no-browser", "--service", "https://nowhere.example"])
    assert result.exit_code == 1
    assert "nowhere.example" in result.output


def _auth_stubs(monkeypatch, *, sessions, signups=None, minted=None, terms=None):
    """The service and its identity provider, scripted."""
    calls = {"terms": [], "mint": [], "signup": []}
    monkeypatch.setattr(cli.hosted, "auth_config", lambda url: {"supabase_url": "https://sb.example", "publishable_key": "pk"})

    def password_session(sb, key, email, password):
        answer = sessions.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def sign_up(sb, key, email, password):
        calls["signup"].append((email, password))
        if isinstance(signups, Exception):
            raise signups
        return signups

    monkeypatch.setattr(cli.hosted, "password_session", password_session)
    monkeypatch.setattr(cli.hosted, "sign_up", sign_up)
    monkeypatch.setattr(cli.hosted, "accept_terms", lambda url, jwt: calls["terms"].append(jwt) or {"tos_accepted_at": "now"})
    monkeypatch.setattr(cli.hosted, "mint_key", lambda url, jwt, name: calls["mint"].append((jwt, name)) or (minted or {"key": "of_live_pw", "name": name}))
    return calls


def test_login_with_a_password_saves_this_machines_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.delenv("FOAMD_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
    calls = _auth_stubs(monkeypatch, sessions=[{"access_token": "jwt-1"}])

    result = CliRunner().invoke(cli.main, ["login", "--name", "laptop"], input="kabir@example.com\nhunter22\n")

    assert result.exit_code == 0, result.output
    assert calls["terms"] == ["jwt-1"] and calls["mint"] == [("jwt-1", "laptop")]
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["foamd_api_key"] == "of_live_pw"
    assert "hunter22" not in result.output and "of_live_pw" not in result.output
    assert "Signed in as kabir@example.com" in result.output


def test_login_offers_to_create_the_account_when_credentials_are_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
    refused = BackendError("Invalid login credentials", code="invalid_credentials", status=400)
    calls = _auth_stubs(monkeypatch, sessions=[refused], signups={"access_token": "jwt-new"})

    result = CliRunner().invoke(cli.main, ["login"], input="new@example.com\npw-pw-pw-pw\ny\ny\n")

    assert result.exit_code == 0, result.output
    assert calls["signup"] == [("new@example.com", "pw-pw-pw-pw")]
    assert calls["terms"] == ["jwt-new"] and calls["mint"][0][0] == "jwt-new"
    assert "/terms" in result.output


def test_login_says_to_confirm_the_address_when_the_provider_wants_that(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
    refused = BackendError("Invalid login credentials", code="invalid_credentials", status=400)
    calls = _auth_stubs(monkeypatch, sessions=[refused], signups=None)

    result = CliRunner().invoke(cli.main, ["login"], input="new@example.com\npw-pw-pw-pw\ny\ny\n")

    assert result.exit_code == 0
    assert "Confirm the address" in result.output
    assert calls["mint"] == [] and not (tmp_path / "c.json").exists()


def test_login_declining_to_create_an_account_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
    refused = BackendError("Invalid login credentials", code="invalid_credentials", status=400)
    _auth_stubs(monkeypatch, sessions=[refused])

    result = CliRunner().invoke(cli.main, ["login"], input="x@example.com\nnope\nn\n")

    assert result.exit_code == 1
    assert not (tmp_path / "c.json").exists()


def test_login_without_a_terminal_needs_the_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: False)
    result = CliRunner().invoke(cli.main, ["login"])
    assert result.exit_code == 1
    assert "--password-stdin" in result.output


def test_login_reads_the_password_from_stdin_for_scripts(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: False)
    calls = _auth_stubs(monkeypatch, sessions=[{"access_token": "jwt-ci"}])
    result = CliRunner().invoke(cli.main, ["login", "--email", "ci@example.com", "--password-stdin"], input="secret\n")
    assert result.exit_code == 0, result.output
    assert calls["mint"][0][0] == "jwt-ci"


def test_login_hands_over_to_the_browser_when_the_provider_wants_a_captcha(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "c.json"))
    monkeypatch.setattr(cli, "_can_prompt", lambda: True)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    refused = BackendError("captcha verification process failed", code="captcha_failed", status=400)
    calls = _auth_stubs(monkeypatch, sessions=[refused])
    monkeypatch.setattr(cli.hosted, "device_code", lambda url, name: {"device_code": "dc", "user_code": "AB12-CD34", "verification_url": "https://app.tryreynolds.com/cli?code=AB12-CD34", "expires_in": 600, "interval": 1})
    monkeypatch.setattr(cli.hosted, "device_token", lambda url, device: {"api_key": "of_live_browser", "name": "laptop"})

    result = CliRunner().invoke(cli.main, ["login", "--no-browser"], input="me@example.com\npw-pw-pw-pw\n")

    assert result.exit_code == 0, result.output
    assert "AB12-CD34" in result.output and "app.tryreynolds.com/cli" in result.output
    assert calls["mint"] == [], "no password mint happened"
    assert json.loads((tmp_path / "c.json").read_text())["foamd_api_key"] == "of_live_browser"

# -- a refusal is not a hiccup -------------------------------------------------


def test_a_refusal_stops_the_harness_asking_again(loop, view, quiet_console):
    """402 means the account cannot pay for the call. Every later call says the same
    thing, and a live session proved it: the budget ran out mid-study and the harness
    made ninety more refused calls over twenty-six minutes, answering nobody."""
    loop.run = lambda: (_ for _ in ()).throw(_api_error(402))

    assert cli._run_turn(loop, view) is False
    assert loop.blocked_reason and "402" in loop.blocked_reason
    assert any("402" in n for n in view.notices), "the page hears it, not just the console"
    assert "refusal, not a hiccup" in quiet_console.file.getvalue() if hasattr(
        quiet_console.file, "getvalue") else True


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413])
def test_every_refusal_status_blocks(loop, view, quiet_console, status):
    loop.run = lambda: (_ for _ in ()).throw(_api_error(status))
    cli._run_turn(loop, view)
    assert loop.blocked_reason is not None


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 503, None])
def test_a_failure_worth_retrying_does_not_block(loop, view, quiet_console, status):
    """A rate limit, a timeout, a bad gateway -- those pass. Blocking on them would
    end sessions that the old behaviour correctly survived."""
    loop.run = lambda: (_ for _ in ()).throw(_api_error(status))
    cli._run_turn(loop, view)
    assert loop.blocked_reason is None


def test_a_completed_turn_lifts_the_block(loop, view, quiet_console):
    loop.blocked_reason = "The model API returned 402: no budget"
    loop.run = lambda: None
    assert cli._run_turn(loop, view) is True
    assert loop.blocked_reason is None


def test_a_blocked_session_does_not_burn_turns_on_progress_wakes(
    loop, backend, store, view, quiet_console, fast_polling, monkeypatch
):
    """The 26-minute case, in miniature: a job is running, the service is refusing,
    and the harness must stop sending. Progress chatter is dropped entirely; a job
    that actually ended is still written down for whenever the session resumes."""
    turns = []
    monkeypatch.setattr(loop, "run", lambda: turns.append(1))
    loop.blocked_reason = "The model API returned 402: no budget"

    wakes = iter([
        Wake("narrate", "run_x: 14m, log 0B"),
        Wake("narrate", "run_x: 15m, log 0B"),
        Wake("job", "job run_x ended exit_code=0"),
        Wake("eof"),
    ])
    monkeypatch.setattr(cli, "watch", lambda *a, **k: next(wakes))
    store.record_job("j1", "pimpleFoam", "run_x", cwd="/work")

    cli._run_interactive(loop, backend, store, view, Browser(backend, store),
                         ScriptedReader([]))

    assert turns == [], "not one refused call was made"
    informed = [m for m in loop.messages if m["role"] in ("system", "user")]
    assert any("ended" in str(m["content"]) for m in informed), "the job's end survived"
    assert not any("log 0B" in str(m["content"]) for m in informed), "chatter did not"


def test_speaking_lifts_the_block_and_the_turn_runs(
    loop, backend, store, view, quiet_console, fast_polling, monkeypatch
):
    """A person is the one thing that can change a refusal -- they have topped the
    account up, or fixed the key, or want to hear the failure again."""
    turns = []
    monkeypatch.setattr(loop, "run", lambda: turns.append(1))
    loop.blocked_reason = "The model API returned 402: no budget"
    wakes = iter([Wake("user", "I raised the budget"), Wake("eof")])
    monkeypatch.setattr(cli, "watch", lambda *a, **k: next(wakes))
    store.record_job("j1", "pimpleFoam", "run_x", cwd="/work")

    cli._run_interactive(loop, backend, store, view, Browser(backend, store),
                         ScriptedReader([]))

    assert loop.blocked_reason is None
    assert turns == [1]


# --- resuming a study this machine has never seen (F-39) ------------------------


def test_a_study_resumed_elsewhere_is_named_by_its_id(backend, store):
    """A study opened in the browser and resumed on a laptop has no local session.

    It used to fall back to the workspace root -- so it opened among every other
    study's files, the mirror tried to bring the whole volume home, and capture
    could not find the row it belonged to. The id already names the directory.
    """
    store.session.home = ""
    assert cli._home_for(store, backend, resuming=True, known_here=False) == \
        f"/work/{store.session.study_id}"


def test_a_recorded_home_still_wins(backend, store):
    store.session.home = "/work/somewhere-else"
    assert cli._home_for(store, backend, resuming=True, known_here=False) == "/work/somewhere-else"


class _StudyClient:
    """A platform that knows about a study this machine does not."""

    def __init__(self, row=None, boom=False):
        self.row = row
        self.boom = boom
        self.asked: list[str] = []

    def get_study(self, study_id):
        self.asked.append(study_id)
        if self.boom:
            raise RuntimeError("no route on this service")
        return self.row


def test_recover_session_fills_in_what_the_laptop_does_not_know(store):
    store.session.home = ""
    store.session.instance_id = ""
    store.session.title = ""
    client = _StudyClient({"id": "20260828-065853-27b6", "home": "/work/20260828-065853-27b6",
                           "instance_id": "inst-9", "title": "lid driven cavity"})
    cli._recover_session(store, client, "20260828-065853-27b6")
    assert store.session.home == "/work/20260828-065853-27b6"
    assert store.session.instance_id == "inst-9"
    assert store.session.title == "lid driven cavity"
    assert store.session.remote_study_id == "20260828-065853-27b6"


def test_recover_session_leaves_local_state_alone(store):
    """What this machine already knows is not overwritten by the platform."""
    store.session.home = "/work/mine"
    store.session.instance_id = "inst-local"
    client = _StudyClient({"id": "s", "home": "/work/theirs", "instance_id": "inst-remote"})
    cli._recover_session(store, client, "s")
    assert store.session.home == "/work/mine"        # a known home short-circuits
    assert client.asked == []                        # and does not even ask


def test_recover_session_survives_a_service_without_the_route(store):
    store.session.home = ""
    cli._recover_session(store, _StudyClient(boom=True), "s")
    assert store.session.home == ""                  # unchanged, no exception
