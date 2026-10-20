"""Felix mode: the second solver (docs/felix-mode-acceptance.md, goals A1 and A3).

A1 -- `--solver felix` makes a Felix study: the study is created `solver: felix`, on a
workspace of kind `felix`; a resume keeps the stored solver and ignores a different
flag; the solver cannot change during a study; OpenFOAM studies are exactly as before.

A3 -- Felix mode does not offer the OpenFOAM-only tools (`cad`, `mesh_review`, the
OpenFOAM toolbox with its case generators), and `job_start` runs `felix run` without
the OpenFOAM restart guard or the trapFpe check.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from openreynolds import cli, commands
from openreynolds.backend import hosted as hosted_mod
from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.capture import Capture
from openreynolds.config import Config
from openreynolds.store import Store
from openreynolds.tools import TOOLS, ToolContext, dispatch, tools_for

from conftest import FakeBackend


# -- helpers ------------------------------------------------------------------


def _said(monkeypatch) -> io.StringIO:
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=200, force_terminal=False))
    return buf


def _no_service(monkeypatch, tmp_path, studies=None):
    """`reserve` as a recorder that then fails, so `session` stops right after it."""
    calls: list[tuple[tuple, dict]] = []

    def reserve(*args, **kwargs):
        calls.append((args, kwargs))
        raise BackendError("no service in this test")

    monkeypatch.setattr(cli.hosted, "reserve", reserve)
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="a",
                 studies_dir=studies or tmp_path / "studies")
    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.delenv("OPENREYNOLDS_LOCAL", raising=False)
    monkeypatch.delenv("OPENREYNOLDS_SOLVER", raising=False)
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    return calls, cfg


def _stored(tmp_path, solver):
    studies = tmp_path / "studies"
    store = Store(studies, "study-x")
    if solver is not None:
        store.session.solver = solver
    store.session.instance_id = "iid-stored"
    store.save()
    if solver is None:
        # A record written before solvers existed has no `solver` key at all.
        import json

        path = studies / "study-x" / "session.json"
        raw = json.loads(path.read_text())
        raw.pop("solver", None)
        path.write_text(json.dumps(raw))
    return studies


def _solver_of(call) -> str:
    """The workspace kind a `reserve` call asked for. Not passed means OpenFOAM."""
    return call[1].get("solver", "openfoam")


# -- A1: the solver of a study --------------------------------------------------


def test_solver_is_a_cli_option_that_defaults_to_the_agent_choosing():
    option = next(p for p in cli.main.params if p.name == "solver")
    assert isinstance(option.type, click.Choice)
    assert sorted(option.type.choices) == ["auto", "felix", "openfoam"]
    assert option.default is None, "unset: then OPENREYNOLDS_SOLVER, then auto"
    assert "auto" in option.help and "default" in option.help


def test_solver_felix_reserves_a_felix_workspace(monkeypatch, tmp_path):
    calls, _ = _no_service(monkeypatch, tmp_path)

    result = CliRunner().invoke(cli.main, ["-p", "go", "--solver", "felix"])

    assert result.exit_code == 1, result.output  # the fake service is down after reserve
    assert calls and _solver_of(calls[0]) == "felix"


def test_solver_openfoam_reserves_exactly_as_before(monkeypatch, tmp_path):
    """No flag, or `--solver openfoam`: the call is the one made before Felix existed,
    three positional arguments and nothing else, so every embedder's stand-in for
    `reserve` keeps working."""
    calls, cfg = _no_service(monkeypatch, tmp_path)

    CliRunner().invoke(cli.main, ["-p", "go", "--solver", "openfoam"])
    # An embedder that names OpenFOAM through the environment gets the same call.
    monkeypatch.setenv("OPENREYNOLDS_SOLVER", "openfoam")
    with pytest.raises(SystemExit):
        cli.session(cfg, study_id=None, instance_id=None, one_shot="go")

    assert len(calls) == 2
    for args, kwargs in calls:
        assert len(args) == 3 and kwargs == {}


def test_solver_resume_keeps_the_stored_solver_and_ignores_a_different_flag(
    monkeypatch, tmp_path
):
    studies = _stored(tmp_path, "felix")
    calls, cfg = _no_service(monkeypatch, tmp_path, studies)
    said = _said(monkeypatch)

    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None,
                    solver="openfoam")

    assert _solver_of(calls[0]) == "felix"
    assert "felix" in said.getvalue().lower() and "ignored" in said.getvalue()
    assert Store(studies, "study-x").session.solver == "felix"


def test_solver_resume_of_an_openfoam_study_ignores_felix(monkeypatch, tmp_path):
    studies = _stored(tmp_path, "openfoam")
    calls, cfg = _no_service(monkeypatch, tmp_path, studies)
    said = _said(monkeypatch)

    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None,
                    solver="felix")

    assert _solver_of(calls[0]) == "openfoam"
    assert "ignored" in said.getvalue()


def test_solver_a_study_from_before_felix_reads_as_openfoam(monkeypatch, tmp_path):
    """A session.json with no `solver` key is an OpenFOAM study."""
    studies = _stored(tmp_path, None)
    raw = (studies / "study-x" / "session.json").read_text()
    assert '"solver"' not in raw
    calls, cfg = _no_service(monkeypatch, tmp_path, studies)

    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None,
                    solver="felix")

    assert _solver_of(calls[0]) == "openfoam"


def test_solver_resume_without_a_flag_says_nothing(monkeypatch, tmp_path):
    studies = _stored(tmp_path, "felix")
    calls, cfg = _no_service(monkeypatch, tmp_path, studies)
    said = _said(monkeypatch)

    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None)

    assert _solver_of(calls[0]) == "felix"
    assert "ignored" not in said.getvalue()


def test_solver_cannot_be_changed_during_a_study():
    """There is a `/mode` and a `/model`; there is no verb for the solver."""
    verbs = {name for spec in commands.COMMANDS for name in (spec.verb, *spec.aliases)}
    assert "/mode" in verbs
    assert not any("solver" in verb for verb in verbs)


def test_solver_felix_needs_the_hosted_workspace(monkeypatch, tmp_path):
    calls, cfg = _no_service(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENREYNOLDS_LOCAL", "1")
    said = _said(monkeypatch)

    with pytest.raises(SystemExit) as raised:
        cli.session(cfg, study_id=None, instance_id=None, one_shot="go", solver="felix")

    assert raised.value.code not in (0, None)
    assert "Felix" in said.getvalue() and "hosted" in said.getvalue()
    assert not calls


def test_solver_round_trips_through_the_study_record(tmp_path):
    import json

    store = Store(tmp_path, "s")
    assert store.session.solver is None and store.recorded_solver is None
    store.save()
    assert json.loads((tmp_path / "s" / "session.json").read_text())["solver"] is None
    assert Store(tmp_path, "s").recorded_solver == "auto", "recorded, still choosing"
    store.session.solver = "felix"
    store.save()
    assert Store(tmp_path, "s").session.solver == "felix"
    assert Store(tmp_path, "s").recorded_solver == "felix"


def test_solver_a_record_without_the_key_is_from_before_solvers(tmp_path):
    import json

    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "session.json").write_text(json.dumps({"study_id": "s"}))
    assert Store(tmp_path, "s").recorded_solver is None


# -- A1: the platform calls ----------------------------------------------------


class _Recording:
    """`FoamdClient.request` as a recorder of what would have been sent."""

    def __init__(self, answer):
        self.sent: list[tuple[str, str, dict | None]] = []
        self.answer = answer

    def __call__(self, method, path, **kwargs):
        self.sent.append((method, path, kwargs.get("json")))
        return SimpleNamespace(status_code=200, json=lambda: self.answer, content=b"{}",
                               text="{}", headers={})


def _client(monkeypatch, answer):
    client = hosted_mod.FoamdClient("https://svc.example", "of_live_test")
    recorder = _Recording(answer)
    monkeypatch.setattr(client, "request", recorder)
    return client, recorder


def test_solver_create_study_sends_felix(monkeypatch):
    client, sent = _client(monkeypatch, {"study_id": "s1"})
    client.create_study("t", "iid", solver="felix")
    assert sent.sent[-1][2]["solver"] == "felix"


def test_solver_create_study_for_openfoam_sends_what_it_always_sent(monkeypatch):
    client, sent = _client(monkeypatch, {"study_id": "s1"})
    client.create_study("t", "iid", study_id="x", home="/work/x")
    assert sent.sent[-1][2] == {"title": "t", "instance_id": "iid", "id": "x", "home": "/work/x"}


def test_solver_create_instance_sends_its_kind(monkeypatch):
    client, sent = _client(monkeypatch, {"instance_id": "i1"})
    client.create_instance(kind="felix")
    assert sent.sent[-1][2] == {"kind": "felix"}
    client.create_instance()
    assert sent.sent[-1][2] == {}


def test_solver_capture_opens_a_felix_study(monkeypatch):
    seen = {}

    class Client:
        def create_study(self, title, instance_id, **extra):
            seen.update(extra)
            return "remote-1"

    capture = Capture.start(Client(), "t", "iid", study_id="s", solver="felix")
    assert capture is not None and seen["solver"] == "felix"
    capture.close(timeout=1)

    seen.clear()
    capture = Capture.start(Client(), "t", "iid", study_id="s")
    assert "solver" not in seen, "an OpenFOAM study opens with the same call as before"
    capture.close(timeout=1)


class _Fleet:
    """A service holding workspaces of both kinds, rows in no useful order."""

    def __init__(self, rows):
        self.rows = rows
        self.created: list[dict] = []

    def list_instances(self):
        return list(self.rows)

    def create_instance(self, **kwargs):
        self.created.append(kwargs)
        return "iid-new"

    def close(self):
        pass


ROWS = [
    {"id": "of-old", "status": "stopped", "kind": "openfoam",
     "last_active_at": "2026-10-01T00:00:00Z"},
    {"id": "fx-1", "status": "stopped", "kind": "felix",
     "last_active_at": "2026-10-05T00:00:00Z"},
    {"id": "of-legacy", "status": "stopped",  # no kind: from before Felix, so OpenFOAM
     "last_active_at": "2026-10-03T00:00:00Z"},
    {"id": "fx-newest", "status": "running", "kind": "felix",
     "last_active_at": "2026-10-06T00:00:00Z"},
]


def _reserve(monkeypatch, rows, **kwargs):
    fleet = _Fleet(rows)
    monkeypatch.setattr(hosted_mod, "FoamdClient", lambda *a, **k: fleet)
    _client, instance_id, _starter = hosted_mod.reserve("https://svc", "key", None, **kwargs)
    return fleet, instance_id


def test_solver_reserve_reuses_only_a_workspace_of_its_kind(monkeypatch):
    _fleet, felix = _reserve(monkeypatch, ROWS, solver="felix")
    assert felix == "fx-newest"

    _fleet, openfoam = _reserve(monkeypatch, ROWS)
    assert openfoam == "of-legacy", "a row with no kind is an OpenFOAM workspace"


def test_solver_reserve_creates_a_felix_workspace_when_there_is_none(monkeypatch):
    rows = [row for row in ROWS if row.get("kind") != "felix"]
    fleet, instance_id = _reserve(monkeypatch, rows, solver="felix")
    assert instance_id == "iid-new"
    assert fleet.created == [{"kind": "felix"}]


def test_solver_reserve_creates_an_openfoam_workspace_as_before(monkeypatch):
    rows = [row for row in ROWS if row.get("kind") == "felix"]
    fleet, instance_id = _reserve(monkeypatch, rows)
    assert instance_id == "iid-new"
    assert fleet.created == [{}], "created exactly as before: no kind sent"


def test_solver_recovered_from_the_platform_for_a_study_new_to_this_machine(tmp_path):
    store = Store(tmp_path, "s")

    class Client:
        def get_study(self, study_id):
            return {"id": study_id, "home": "/work/s", "solver": "felix"}

    cli._recover_session(store, Client(), "s")
    assert store.session.solver == "felix"


# -- A3: the tools in Felix mode ---------------------------------------------


@pytest.fixture
def felix(ctx):
    ctx.solver = "felix"
    ctx.cad = object()
    ctx.reviewer = object()
    return ctx


def _names(tools):
    return [tool["name"] for tool in tools]


def test_tools_felix_withholds_the_openfoam_only_tools(felix):
    names = _names(tools_for(felix))
    assert "cad" not in names and "mesh_review" not in names
    for kept in ("bash", "fetch", "job_check", "job_kill", "job_start", "read_file",
                 "write_file"):
        assert kept in names
    assert names == sorted(names)


def test_tools_felix_list_is_the_same_bytes_every_call(felix):
    assert tools_for(felix) == tools_for(felix)
    assert tools_for(felix) is tools_for(felix)


def test_tools_openfoam_list_is_unchanged(ctx):
    ctx.cad = object()
    ctx.reviewer = object()
    assert tools_for(ctx) is TOOLS


def test_tools_felix_structured_mode_still_has_its_checkpoint(felix):
    felix.mode = "structured"
    assert "checkpoint" in _names(tools_for(felix))


def _tool(ctx, name):
    return next(tool for tool in tools_for(ctx) if tool["name"] == name)


def test_tools_felix_job_start_describes_felix_run(felix):
    job_start = _tool(felix, "job_start")
    text = job_start["description"]
    for fact in ("felix run", "--gpu", "--wall", "felix continue", "stdout", "stderr",
                 "Exit 0", "output.every", "output.previous", "warm_start", "symlink",
                 "402 allowance_exhausted", "403 plan_limit", "409 study_paused"):
        assert fact in text, fact
    for openfoam in ("trapFpe", "decomposePar", "startFrom", "OpenFOAM", "mpirun"):
        assert openfoam not in text, openfoam
    properties = job_start["input_schema"]["properties"]
    assert "overwrite" not in properties
    assert "trapFpe" not in properties["kill_on"]["description"]
    assert "OpenFOAM" not in properties["kill_on"]["description"]


def test_tools_felix_bash_promises_no_openfoam_environment(felix):
    assert "OpenFOAM" not in _tool(felix, "bash")["description"]
    assert "RunFunctions" not in _tool(felix, "bash")["description"]


def test_tools_felix_job_start_runs_felix_without_the_openfoam_guards(felix, backend):
    """A case that looks like a transient with written times, and a kill pattern
    that matches the trapFpe banner: neither matters to `felix run`."""
    backend.files["/work/cav/system/controlDict"] = b"startFrom startTime;\nendTime 1;\n"
    backend.exec_results["ls -d /work/cav/[0-9]* /work/cav/processor*/[0-9]* "
                         "/work/cav/processors*/[0-9]* 2>/dev/null"] = ExecResult(
        0, "/work/cav/0.1\n/work/cav/0.2\n", False, None)

    content, is_error = dispatch(felix, "job_start", {
        "cmd": "felix run /work/cav --gpu H100 --wall 3600",
        "name": "cavity",
        "kill_on": ["Floating point exception"],
    })

    assert not is_error
    assert content == "started job job-1 (cavity)"
    assert backend.started == [{
        "cmd": "felix run /work/cav --gpu H100 --wall 3600", "cwd": felix.home,
        "name": "cavity", "kill_on": ["Floating point exception"],
    }]
    assert backend.execs == [], "no listing, no controlDict read, no mesher probe"
    assert felix.store.session.jobs["job-1"].cmd.startswith("felix run")


def test_tools_felix_job_start_still_refuses_a_redirect(felix, backend):
    """The job captures stdout and stderr; a redirect hides the log from job_check."""
    content, _ = dispatch(felix, "job_start", {"cmd": "felix run cav --gpu H100 2>&1"})
    assert content.startswith("not started:")
    assert not backend.started


def test_tools_openfoam_job_start_still_guards(ctx, backend):
    content, _ = dispatch(ctx, "job_start", {
        "cmd": "simpleFoam", "kill_on": ["Floating point exception"]})
    assert content.startswith("not started:") and "trapFpe" in content


def test_tools_context_carries_the_solver(backend, store):
    assert ToolContext(backend=backend, store=store, max_output=1).solver == "openfoam"


# -- A2/A3: what a Felix study's model is actually sent ----------------------------


def _first_call(ctx, store, view):
    from openreynolds.loop import Loop

    from conftest import install_model, message, text_block

    loop = Loop(Config(llm_api_key="test-key", model="claude-opus-5"), ctx, store, view)
    fake = install_model(loop, [message([text_block("ok")])])
    loop.say("go")
    loop.run()
    return fake.calls[0]


def _system_text(call) -> str:
    system = call["system"]
    if isinstance(system, str):
        return system
    return "".join(block.get("text", "") for block in system)


def test_tools_a_felix_study_is_sent_the_felix_prompt_and_tools(felix, store, view):
    from openreynolds.prompt import FELIX_PROMPT

    call = _first_call(felix, store, view)
    assert _system_text(call) == FELIX_PROMPT
    names = _names(call["tools"])
    assert "cad" not in names and "mesh_review" not in names


def test_tools_an_openfoam_study_is_sent_what_it_always_was(ctx, store, view):
    from openreynolds.prompt import SYSTEM_PROMPT

    call = _first_call(ctx, store, view)
    assert _system_text(call) == SYSTEM_PROMPT


# -- A1: the agent chooses the solver (`--solver auto`, the default) -------------


class _Platform:
    """The service as a session sees it through the client `reserve` hands back:
    studies opened, messages posted. Anything else it is asked is answered None."""

    def __init__(self):
        self.studies: list[dict] = []
        self.posted: list[dict] = []

    def create_study(self, title, instance_id, **extra):
        self.studies.append({"title": title, "instance_id": instance_id, **extra})
        return "remote-1"

    def post_messages(self, study_id, messages):
        self.posted.extend(messages)

    def __getattr__(self, name):
        return lambda *a, **k: None


def _auto_session(monkeypatch, tmp_path, responses, *, solver="auto", study_id=None,
                  lines=("go",), env=None):
    """One session end to end against a scripted model and a fake service, with the
    solver as the CLI hands it over. Returns what the test looks at."""
    from openreynolds.backend import pending as pending_mod
    from openreynolds.loop import Loop

    from conftest import ReadyStarter, RecordingView, ScriptedReader, install_model

    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    monkeypatch.delenv("OPENREYNOLDS_LOCAL", raising=False)
    monkeypatch.delenv("OPENREYNOLDS_SOLVER", raising=False)
    if env is not None:
        monkeypatch.setenv("OPENREYNOLDS_SOLVER", env)
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    _said(monkeypatch)
    backend = FakeBackend()
    studies = tmp_path / "studies"
    platform = _Platform()
    reserves: list[dict] = []
    loops: list[Loop] = []

    def reserve(url, key, iid=None, **kwargs):
        reserves.append({"iid": iid, "model_calls": sum(len(l.fake.calls) for l in loops),
                         "recorded": _recorded_solvers(studies), **kwargs})
        return platform, "iid-9", ReadyStarter(backend, "iid-9")

    class ScriptedLoop(Loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fake = install_model(self, list(responses))
            loops.append(self)

    monkeypatch.setattr(cli.hosted, "reserve", reserve)
    monkeypatch.setattr(cli, "Loop", ScriptedLoop)
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="sk-test", model="m",
                 studies_dir=tmp_path / "studies", capture=True, desk=False,
                 mesh_tool=False, mirror_interval_s=0.0, workspace_eta_s=20.0)
    class SolverView(RecordingView):
        """Records the solver it is told, and what the study's record said of the
        solver at the moment the header went out."""

        def __init__(self):
            super().__init__()
            self.solvers: list[str] = []
            self.recorded_at_header: list = []

        def header(self, *args, **kwargs):
            self.recorded_at_header.append(_recorded_solvers(studies))
            super().header(*args, **kwargs)

        def solver(self, name):
            self.solvers.append(name)

    view = SolverView()

    def interface(drive):
        drive(view, ScriptedReader(list(lines)))
        return False

    kwargs = {} if solver is None else {"solver": solver}
    cli.session(cfg, study_id=study_id, instance_id=None, one_shot=None,
                interface=interface, **kwargs)
    return SimpleNamespace(reserves=reserves, platform=platform, backend=backend,
                           loop=loops[0], cfg=cfg, view=view)


def _recorded_solvers(studies):
    """`solver` in every study's session.json on disk, `<absent>` where there is none."""
    import json

    found = []
    for path in sorted(studies.glob("*/session.json")):
        raw = json.loads(path.read_text())
        found.append(raw.get("solver", "<absent>"))
    return found


def _choose(solver, reason="the flow is incompressible and the mesh is large"):
    from conftest import message, tool_block

    return message([tool_block("choose_solver", {"solver": solver, "reason": reason})],
                   stop_reason="tool_use")


def _done():
    from conftest import message, text_block

    return message([text_block("done")])


def _study_dir(run):
    (only,) = [p for p in run.cfg.studies_dir.iterdir() if p.is_dir()]
    return only


def test_solver_auto_routes_the_first_message_before_the_first_main_request(monkeypatch, tmp_path):
    """The solver is settled by a request of its own, on the person's first message,
    before the main agent's first request -- which already has the chosen solver's
    prompt and tools. Nothing is reserved until the routing request has answered."""
    from openreynolds import desk
    from openreynolds.prompt import FELIX_PROMPT

    run = _auto_session(monkeypatch, tmp_path, [_choose("felix"), _done()])

    calls = run.loop.fake.calls
    assert _names(calls[0]["tools"]) == ["choose_solver"], "the routing request"
    assert _system_text(calls[0]) == desk.ROUTING_SYSTEM
    assert calls[0]["messages"] == [{"role": "user", "content": "go"}], "the person's words"
    assert len(run.reserves) == 1, "one workspace, reserved once"
    assert run.reserves[0]["model_calls"] == 1, "reserved after routing, before the agent"
    assert run.reserves[0].get("solver") == "felix"
    assert run.platform.studies and run.platform.studies[0]["solver"] == "felix"
    assert run.platform.posted, "what was said before the choice still reaches the platform"
    # The main agent's first request is a Felix study's.
    assert _system_text(calls[1]) == FELIX_PROMPT
    names = _names(calls[1]["tools"])
    assert "choose_solver" not in names and "cad" not in names and "job_start" in names
    assert Store(run.cfg.studies_dir, _study_dir(run).name).session.solver == "felix"
    # The reason reaches the person and the agent.
    assert any("Felix, chosen from your first message" in n for n in run.view.notices)
    briefing = calls[1]["messages"][0]["content"]
    assert "chosen from the person's first message" in briefing
    assert "the flow is incompressible and the mesh is large" in briefing


def test_solver_routing_reads_as_the_agent_working_not_the_persons_turn(monkeypatch, tmp_path):
    """The canary (production study 20261019-013223-f805): the first message was routed
    in about 14 s, and for all of it the hosted page still showed the person's turn
    from its idle prompt -- an MCP caller read "Reynolds has finished its turn" with an
    empty transcript and ended the study. During the routing request the agent is the
    one working, and the view says so before the request goes out."""
    import conftest

    seen = []
    views = []
    real_stream = conftest.FakeMessages.stream

    def watching(self, **kwargs):
        view = views[0] if views else None
        seen.append((kwargs["tools"][0]["name"] if len(kwargs["tools"]) == 1 else "main",
                     getattr(view, "whose", None), list(getattr(view, "working_on", []))))
        return real_stream(self, **kwargs)

    monkeypatch.setattr(conftest.FakeMessages, "stream", watching)
    real_header = conftest.RecordingView.header

    def header(self, *args, **kwargs):
        views.append(self)
        return real_header(self, *args, **kwargs)

    monkeypatch.setattr(conftest.RecordingView, "header", header)

    run = _auto_session(monkeypatch, tmp_path, [_choose("felix"), _done()])

    routing = seen[0]
    assert routing[0] == "choose_solver"
    assert routing[1] == "agent", "the agent's move while the solver is chosen"
    assert routing[2] == ["choosing the solver"]
    assert run.view.prompts >= 1, "the idle prompt came first, as it does on the page"


def test_solver_auto_never_changes_the_prompt_or_tools_mid_conversation(monkeypatch, tmp_path):
    """The rule the old design broke: every request of the main conversation carries
    the same system prompt and the same tool list, from the first to the last. The
    swap it made after `choose_solver` was a 400 on Bedrock (`desk.py`)."""
    from conftest import message, tool_block

    run = _auto_session(monkeypatch, tmp_path, [
        _choose("felix"),
        message([tool_block("bash", {"cmd": "ls"})], stop_reason="tool_use"),
        message([tool_block("read_file", {"path": "/work"}, "tu_2")], stop_reason="tool_use"),
        _done(),
    ])

    main = run.loop.fake.calls[1:]
    assert len(main) >= 3
    assert len({_system_text(call) for call in main}) == 1
    assert all(call["tools"] == main[0]["tools"] for call in main)
    assert all("choose_solver" not in _names(call["tools"]) for call in main)


def test_solver_auto_choosing_openfoam_gives_the_openfoam_study(monkeypatch, tmp_path):
    from openreynolds.prompt import SYSTEM_PROMPT

    run = _auto_session(monkeypatch, tmp_path, [_choose("openfoam"), _done()])

    calls = run.loop.fake.calls
    assert _system_text(calls[1]) == SYSTEM_PROMPT
    assert "choose_solver" not in _names(calls[1]["tools"])
    assert run.reserves and "solver" not in run.reserves[0], "reserved exactly as before"
    assert "solver" not in run.platform.studies[0]
    assert Store(run.cfg.studies_dir, _study_dir(run).name).session.solver == "openfoam"


def test_solver_a_routing_request_that_does_not_choose_is_openfoam_said_aloud(monkeypatch, tmp_path):
    """No guess and no silence: the study is OpenFOAM's, and both the person and the
    agent are told why."""
    from conftest import message, text_block
    from openreynolds.prompt import SYSTEM_PROMPT

    run = _auto_session(monkeypatch, tmp_path, [message([text_block("hmm")]), _done()])

    assert run.reserves and "solver" not in run.reserves[0]
    assert any("could not be chosen" in n for n in run.view.notices)
    calls = run.loop.fake.calls
    assert _system_text(calls[1]) == SYSTEM_PROMPT
    assert "OpenFOAM by default" in calls[1]["messages"][0]["content"]


def test_solver_auto_session_nobody_speaks_in_starts_nothing(monkeypatch, tmp_path):
    """A hosted study can be opened with no first message. It waits at the prompt, and
    is routed on the first message typed; when none ever is, nothing was spent."""
    run = _auto_session(monkeypatch, tmp_path, [_done()], lines=())

    assert run.loop.fake.calls == []
    assert run.reserves == []
    assert run.platform.studies == []
    assert run.backend.execs == []
    study = Store(run.cfg.studies_dir, _study_dir(run).name)
    assert study.session.solver is None
    assert study.recorded_solver == "auto", "still to be chosen when it is resumed"


def test_solver_auto_choice_holds_on_resume(monkeypatch, tmp_path):
    first = _auto_session(monkeypatch, tmp_path, [_choose("felix"), _done()])
    study_id = _study_dir(first).name

    again = _auto_session(monkeypatch, tmp_path, [_done()], study_id=study_id,
                          solver="openfoam")

    assert again.reserves[0].get("solver") == "felix", "reserved at the start, as Felix"
    assert again.reserves[0]["model_calls"] == 0
    assert "choose_solver" not in _names(again.loop.fake.calls[0]["tools"])


def test_solver_forced_reserves_at_the_start_with_no_choice_offered(monkeypatch, tmp_path):
    run = _auto_session(monkeypatch, tmp_path, [_done()], solver="felix")

    assert run.reserves[0]["model_calls"] == 0, "no extra wait: reserved before the model"
    assert run.reserves[0].get("solver") == "felix"
    assert "choose_solver" not in _names(run.loop.fake.calls[0]["tools"])


def test_solver_auto_on_a_local_workspace_is_openfoam(monkeypatch, tmp_path):
    """Felix needs the hosted service, so there is nothing to choose between."""
    assert cli._settle_solver(None, "auto", resuming=False, local=True) == "openfoam"
    assert cli._settle_solver(None, "auto", resuming=False, local=False) == "auto"
    assert cli._settle_solver(None, None, resuming=False, local=False) == "auto"


@pytest.mark.parametrize(
    ("stored", "asked", "resuming", "settled"),
    [
        (None, "auto", False, "auto"),        # a new study: the agent chooses
        (None, None, False, "auto"),          # ... by default
        (None, "felix", False, "felix"),      # forced
        (None, "openfoam", False, "openfoam"),
        (None, "auto", True, "openfoam"),     # a study from before solvers: OpenFOAM
        (None, "felix", True, "openfoam"),    # ... and a flag does not change it
        ("felix", "auto", True, "felix"),     # the stored solver wins
        ("felix", "openfoam", True, "felix"),
        ("openfoam", "felix", True, "openfoam"),
        ("auto", "auto", True, "auto"),       # never chosen: still the agent's to choose
        ("auto", "felix", True, "felix"),     # never chosen: a flag is a choice
    ],
)
def test_solver_settles_from_the_record_and_the_flag(stored, asked, resuming, settled):
    assert cli._settle_solver(stored, asked, resuming=resuming, local=False) == settled


# -- A3: one tool list for the whole study ----------------------------------------


def test_tools_no_main_list_offers_the_solver_choice_and_every_one_offers_send_to_agent(ctx):
    """The choice is routing's own tool now (`desk.ROUTE_TOOL`), never the agent's; and
    `send_to_agent` is in every list from the first request, because the side chat
    sends this list as it is (`desk.SideChat`)."""
    for solver in ("openfoam", "felix", "auto"):
        ctx.solver = solver
        for mode in ("auto", "structured"):
            ctx.mode = mode
            names = _names(tools_for(ctx))
            assert "choose_solver" not in names
            assert "send_to_agent" in names


# -- the seam the Felix log parser (A4) plugs into ------------------------------


@pytest.mark.parametrize(
    ("cmd", "felix"),
    [
        ("felix run cav --gpu H100", True),
        ("cd /work/s && felix continue cav --gpu B200 --wall 600", True),
        ("felix gpus", False),
        ("felix-check-mesh --case cav", False),
        ("simpleFoam -parallel", False),
    ],
)
def test_tools_a_felix_solve_is_told_from_its_command(cmd, felix):
    from openreynolds.progress import is_felix_solve

    assert is_felix_solve(cmd) is felix


def test_tools_a_felix_solve_reaches_the_progress_seam(monkeypatch, backend, store):
    from openreynolds import progress

    seen = []
    monkeypatch.setattr(progress, "felix_job_progress",
                        lambda record, tail, size: seen.append((record.cmd, tail)) or None)
    backend.job_start("felix run cav --gpu H100")
    backend.logs["job-1"] = b"step=1 t=0.01 dt=0.01\n"
    import dataclasses

    backend.jobs["job-1"] = dataclasses.replace(backend.jobs["job-1"],
                                                log_size=len(backend.logs["job-1"]))
    store.record_job("job-1", cmd="felix run cav --gpu H100", name="cav", cwd="/work")
    tracker = progress.Tracker(view=None, backend=backend, store=store, home="/work")

    jobs = tracker.refresh_jobs(force=True)

    assert seen and seen[0][0].startswith("felix run") and "step=1" in seen[0][1]
    assert jobs and jobs[0].job_id == "job-1", "with no Felix reading, shown as any job"


# -- the interface an embedder (the web app) uses ------------------------------------


def test_solver_an_embedder_that_names_none_has_it_routed(monkeypatch, tmp_path):
    run = _auto_session(monkeypatch, tmp_path, [_choose("openfoam"), _done()], solver=None)
    assert run.reserves and run.reserves[0]["model_calls"] == 1
    assert _names(run.loop.fake.calls[0]["tools"]) == ["choose_solver"]


@pytest.mark.parametrize("env", ["felix", "openfoam"])
def test_solver_an_embedder_can_name_it_in_the_environment(monkeypatch, tmp_path, env):
    run = _auto_session(monkeypatch, tmp_path, [_done()], solver=None, env=env)
    assert run.reserves and run.reserves[0]["model_calls"] == 0
    assert run.reserves[0].get("solver", "openfoam") == env


def test_solver_the_keyword_wins_over_the_environment(monkeypatch, tmp_path):
    run = _auto_session(monkeypatch, tmp_path, [_done()], solver="felix", env="openfoam")
    assert run.reserves[0].get("solver") == "felix"


def test_solver_a_given_solver_is_on_record_before_the_header(monkeypatch, tmp_path):
    run = _auto_session(monkeypatch, tmp_path, [_done()], solver="felix")
    assert run.view.recorded_at_header == [["felix"]]
    assert run.view.solvers == ["felix"]


def test_solver_a_chosen_solver_is_on_record_the_moment_it_is_chosen(monkeypatch, tmp_path):
    run = _auto_session(monkeypatch, tmp_path, [_choose("felix"), _done()])
    assert run.view.recorded_at_header == [[None]], "null until chosen"
    assert run.reserves[0]["recorded"] == ["felix"], "written before anything is reserved"
    assert run.view.solvers == ["felix"]


def test_solver_the_json_stream_says_the_solver(tmp_path):
    import io as _io
    import json

    from openreynolds.jsonview import JsonView

    out = _io.StringIO()
    JsonView(out).solver("felix")
    event = json.loads(out.getvalue().splitlines()[-1])
    assert event["type"] == "solver" and event["solver"] == "felix"


# -- a study resumed on a machine that never held it ----------------------------------


def _remote_resume(monkeypatch, tmp_path, answer, *, solver=None):
    """Resume `study-r`, which this machine has no record of, with the platform
    answering `answer` for its solver: ("felix", True), (None, True) for no such
    study, or (None, False) for a platform that could not be reached."""
    calls, cfg = _no_service(monkeypatch, tmp_path)
    asked = []

    def study_solver(url, key, study_id):
        asked.append(study_id)
        return answer

    monkeypatch.setattr(cli.hosted, "study_solver", study_solver)
    said = _said(monkeypatch)
    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-r", instance_id=None, one_shot=None,
                    **({} if solver is None else {"solver": solver}))
    return calls, asked, said.getvalue()


def test_solver_a_study_new_here_takes_its_solver_from_the_platform(monkeypatch, tmp_path):
    calls, asked, said = _remote_resume(monkeypatch, tmp_path, ("felix", True))
    assert asked == ["study-r"]
    assert _solver_of(calls[0]) == "felix", "reserved as the Felix study it is"


def test_solver_the_platforms_solver_wins_over_a_flag(monkeypatch, tmp_path):
    calls, _, said = _remote_resume(monkeypatch, tmp_path, ("felix", True), solver="openfoam")
    assert _solver_of(calls[0]) == "felix"
    assert "ignored" in said


def test_solver_an_unreachable_platform_is_said_not_guessed(monkeypatch, tmp_path):
    calls, _, said = _remote_resume(monkeypatch, tmp_path, (None, False))
    assert _solver_of(calls[0]) == "openfoam"
    flat = " ".join(said.split()).lower()
    assert "could not read this study's solver" in flat and "--solver felix" in flat


def test_solver_an_unreachable_platform_takes_a_named_solver(monkeypatch, tmp_path):
    calls, _, said = _remote_resume(monkeypatch, tmp_path, (None, False), solver="felix")
    assert _solver_of(calls[0]) == "felix"


def test_solver_a_study_held_here_is_not_asked_about(monkeypatch, tmp_path):
    studies = _stored(tmp_path, "felix")
    calls, cfg = _no_service(monkeypatch, tmp_path, studies)
    asked = []
    monkeypatch.setattr(cli.hosted, "study_solver",
                        lambda *a: asked.append(a) or ("openfoam", True))
    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None)
    assert asked == [] and _solver_of(calls[0]) == "felix"


class _OneStudy:
    def __init__(self, answer):
        self.answer = answer
        self.closed = False

    def get_study(self, study_id):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"id": "s", "solver": "felix"}, ("felix", True)),
        ({"id": "s"}, ("openfoam", True)),                       # the service's default
        (BackendError("not found", code="http_error", status=404), (None, True)),
        (BackendError("unreachable", code="unreachable"), (None, False)),
    ],
)
def test_solver_read_from_the_platform(monkeypatch, answer, expected):
    client = _OneStudy(answer)
    monkeypatch.setattr(hosted_mod, "FoamdClient", lambda *a, **k: client)
    assert hosted_mod.study_solver("https://svc", "key", "s") == expected
    assert client.closed
