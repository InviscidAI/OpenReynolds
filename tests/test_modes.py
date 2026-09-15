"""Modes: full auto, ask before compute, structured.

The contract (docs/design.md section 1) holds unchanged in auto. The other two gate
exactly what the person chose to have gated, and these tests hold both halves.
"""

from __future__ import annotations

import json
import re

import click
import pytest
from click.testing import CliRunner

from conftest import ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli, commands, modes
from openreynolds.approval import Approver
from openreynolds.backend.base import BackendError
from openreynolds.browse import Browser
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.store import Store
from openreynolds.toolbox import study_run
from test_prompt import IMPERATIVE_PATTERNS


@pytest.fixture
def loop(ctx, store, view):
    return Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view)


def tool_results(loop):
    return [
        block
        for m in loop.messages
        if m["role"] == "user" and isinstance(m["content"], list)
        for block in m["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def texts_in_results(loop):
    return [
        block["text"]
        for m in loop.messages
        if m["role"] == "user" and isinstance(m["content"], list)
        for block in m["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    ]


# -- the vocabulary --------------------------------------------------------------


def test_every_mode_has_a_label_and_a_description():
    assert modes.MODES == ("auto", "partial", "structured")
    assert modes.LABELS == {"auto": "Full auto", "partial": "Ask before compute",
                            "structured": "Structured"}
    assert set(modes.DESCRIPTIONS) == set(modes.MODES)


@pytest.mark.parametrize("name,expected", [
    ("auto", "auto"), ("FULL", "auto"), ("full-auto", "auto"), ("fullauto", "auto"),
    ("free", "auto"), ("ask", "partial"), ("approve", "partial"), ("approvals", "partial"),
    ("partial-auto", "partial"), ("plan", "structured"), ("staged", "structured"),
    ("stages", "structured"), ("guided", "structured"), (" Structured ", "structured"),
    ("nonsense", None), ("", None), (None, None),
])
def test_names_and_aliases_normalise(name, expected):
    assert modes.normalize(name) == expected


def test_the_stages_are_the_guided_pipelines_own_phases():
    """Invented stage names would be a second vocabulary for the same study."""
    assert modes.STAGES == study_run.PHASE_NAMES


def test_auto_adds_nothing_to_the_briefing():
    assert modes.briefing("auto") == ""


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_what_the_model_is_told_about_a_mode_mandates_no_workflow(pattern):
    for mode in modes.MODES:
        for text in (modes.briefing(mode), modes.switched(mode), modes.held("job_start")):
            match = re.search(pattern, text, re.IGNORECASE)
            assert match is None, f"imperative language for {mode}: {match!r}"


def test_the_structured_briefing_names_every_stage():
    said = modes.briefing("structured")
    for stage in modes.STAGES:
        assert stage in said
    assert "checkpoint" in said


# -- configuration and the command line -----------------------------------------


@pytest.fixture
def clean_config(monkeypatch, tmp_path):
    for name in ("OPENREYNOLDS_MODE", "OPENREYNOLDS_PROVIDER", "OPENREYNOLDS_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "config.json"))
    return tmp_path / "config.json"


def test_the_mode_comes_from_the_environment_by_any_name(clean_config, monkeypatch):
    monkeypatch.setenv("OPENREYNOLDS_MODE", "ask")
    assert Config.load().mode == "partial"


def test_the_mode_comes_from_the_config_file(clean_config):
    clean_config.write_text(json.dumps({"mode": "plan"}), encoding="utf-8")
    assert Config.load().mode == "structured"


def test_a_mode_that_is_not_one_loads_as_full_auto(clean_config, monkeypatch):
    """Picked over refusing to load: `Config.load` serves `doctor`, `studies` and every
    other subcommand, and a typo in a mode must not take those away. The session
    prints that the value was ignored."""
    monkeypatch.setenv("OPENREYNOLDS_MODE", "sometimes")
    assert Config.load().mode == "auto"


def test_a_bad_environment_mode_falls_through_to_the_config_file(clean_config, monkeypatch):
    clean_config.write_text(json.dumps({"mode": "plan"}), encoding="utf-8")
    monkeypatch.setenv("OPENREYNOLDS_MODE", "sometimes")
    assert Config.load().mode == "structured"


def test_a_bad_environment_mode_leaves_a_resumed_study_in_its_stored_mode(monkeypatch, tmp_path):
    """What the docs say: an ignored value names no mode, so the stored one stands."""
    monkeypatch.setenv("OPENREYNOLDS_MODE", "sometimes")
    studies = _stored_study(tmp_path, "partial")
    _, cfg = _no_instance(monkeypatch, tmp_path, Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", studies_dir=studies))

    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None)

    assert cfg.mode == "partial"


def test_saving_a_default_config_says_nothing_about_modes(clean_config):
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="a")
    assert "mode" not in json.loads(cfg.save().read_text())
    cfg.mode = "partial"
    assert json.loads(cfg.save().read_text())["mode"] == "partial"


def _no_instance(monkeypatch, tmp_path, cfg=None):
    acquired = []

    def acquire(url, key, iid):
        acquired.append(iid)
        raise BackendError("no service in this test")

    monkeypatch.setattr(cli.hosted, "acquire", acquire)
    config = cfg or Config(foamd_url="u", foamd_api_key="k", llm_api_key="a",
                           studies_dir=tmp_path / "studies")
    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls: config))
    return acquired, config


@pytest.mark.parametrize("mode", ["partial", "structured", "ask"])
def test_a_one_shot_run_cannot_take_a_mode_that_asks(mode, monkeypatch, tmp_path):
    """-p has nobody at the terminal; a question asked there waits forever."""
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    acquired, _ = _no_instance(monkeypatch, tmp_path)

    result = CliRunner().invoke(cli.main, ["-p", "go", "--mode", mode])

    assert result.exit_code == 2, result.output
    assert "needs someone to answer" in result.output
    assert not acquired, "nothing is acquired for a run that could only wait"


def test_the_environment_mode_is_refused_with_a_one_shot_run_too(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREYNOLDS_MODE", "structured")
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="a",
                 studies_dir=tmp_path / "studies", mode="structured")
    acquired, _ = _no_instance(monkeypatch, tmp_path, cfg)

    result = CliRunner().invoke(cli.main, ["-p", "go"])

    assert result.exit_code == 2
    assert not acquired


def test_full_auto_with_a_one_shot_run_goes_ahead(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    acquired, _ = _no_instance(monkeypatch, tmp_path)

    result = CliRunner().invoke(cli.main, ["-p", "go", "--mode", "full"])

    assert result.exit_code == 1  # the fake service is down, which is past the check
    assert acquired


def _stored_study(tmp_path, mode):
    studies = tmp_path / "studies"
    store = Store(studies, "study-x")
    store.session.mode = mode
    store.save()
    return studies


def test_a_resumed_study_keeps_the_mode_it_was_started_in(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    studies = _stored_study(tmp_path, "partial")
    acquired, cfg = _no_instance(monkeypatch, tmp_path, Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", studies_dir=studies))

    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None)

    assert cfg.mode == "partial"


def test_an_explicit_mode_wins_over_the_stored_one(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    studies = _stored_study(tmp_path, "structured")
    _, cfg = _no_instance(monkeypatch, tmp_path, Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", studies_dir=studies))

    with pytest.raises(SystemExit) as raised:
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot="go",
                    mode_explicit=True)

    assert not isinstance(raised.value, click.UsageError)
    assert cfg.mode == "auto"


def test_the_environment_counts_as_explicit_for_an_embedder(monkeypatch, tmp_path):
    """The hosted runner sets OPENREYNOLDS_MODE and calls `session` without the flag."""
    monkeypatch.setenv("OPENREYNOLDS_MODE", "auto")
    studies = _stored_study(tmp_path, "structured")
    _, cfg = _no_instance(monkeypatch, tmp_path, Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", studies_dir=studies))

    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None)

    assert cfg.mode == "auto"


def test_a_resumed_one_shot_run_of_a_structured_study_is_refused(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    studies = _stored_study(tmp_path, "structured")
    acquired, cfg = _no_instance(monkeypatch, tmp_path, Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", studies_dir=studies))

    with pytest.raises(click.UsageError):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot="go")
    assert not acquired


# -- /mode -----------------------------------------------------------------------


def test_mode_with_nothing_after_it_lists_the_modes(loop, view, backend, store):
    cli._local(commands.parse("/mode"), view, Browser(backend, store), store, loop)
    shown = "\n".join(view.statuses[-1])
    for mode in modes.MODES:
        assert mode in shown
    assert "/mode" in shown


def test_switching_mode_is_recorded_and_told_to_the_model(loop, view, backend, store):
    loop.say("hello")
    cli._local(commands.parse("/mode ask"), view, Browser(backend, store), store, loop)

    assert loop.ctx.mode == "partial" == store.session.mode
    assert json.loads((store.dir / "session.json").read_text())["mode"] == "partial"
    assert "ask-before-compute" in loop.messages[-1]["content"]
    assert "next tool call" in "\n".join(view.statuses[-1])


def test_switching_into_structured_starts_the_plan_afresh(loop, view, backend, store):
    loop.ctx.mode, loop.ctx.plan_approved = "structured", True
    loop.set_mode("auto")
    loop.set_mode("structured")
    assert loop.ctx.plan_approved is False


def test_an_unknown_mode_changes_nothing(loop, view, backend, store):
    cli._local(commands.parse("/mode sometimes"), view, Browser(backend, store), store, loop)
    assert loop.ctx.mode == "auto"
    assert "no mode called" in view.statuses[-1][0]
    assert loop.messages == []


def test_an_answer_with_no_question_says_so(loop, view, backend, store):
    for line in ("/yes", "/no", "/all"):
        cli._local(commands.parse(line), view, Browser(backend, store), store, loop)
        assert view.statuses[-1] == ["nothing is waiting for an answer"]


def test_a_switch_typed_mid_turn_rides_with_the_tool_results(loop, backend, store, view):
    """The message after a tool call has to be its results, so a fact said between
    tool calls cannot be a message of its own."""
    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "ls"})], stop_reason="tool_use"),
        message([text_block("done")]),
    ])
    loop.interject = lambda: loop.set_mode("structured") and None
    loop.say("go")
    loop.run()

    results_message = next(m for m in loop.messages
                           if m["role"] == "user" and isinstance(m["content"], list))
    assert results_message["content"][0]["type"] == "tool_result"
    assert any("structured mode" in t for t in texts_in_results(loop))
    names = [t["name"] for t in fake.calls[-1]["tools"]]
    assert "checkpoint" in names, "the tool list follows the switch at the next request"
    assert "checkpoint" not in [t["name"] for t in fake.calls[0]["tools"]]


# -- the gate in the loop --------------------------------------------------------


def a_job_then_done():
    return [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})],
                stop_reason="tool_use"),
        message([text_block("started")]),
    ]


def test_auto_never_asks_and_never_holds(loop, backend, view):
    asked = []
    loop.approver = type("A", (), {"ask": lambda self, *a, **k: asked.append(a)})()
    install_model(loop, a_job_then_done())
    loop.say("go")
    loop.run()
    assert backend.started and not asked


def test_partial_runs_a_job_once_the_person_approves(loop, backend, view):
    loop.ctx.mode = "partial"
    loop.approver = Approver(view, ScriptedReader(["/yes"]))
    install_model(loop, a_job_then_done())
    loop.say("go")
    loop.run()
    assert [s["cmd"] for s in backend.started] == ["simpleFoam"]
    assert not tool_results(loop)[0].get("is_error")


def test_partial_declines_with_the_persons_words(loop, backend, view):
    loop.ctx.mode = "partial"
    loop.approver = Approver(view, ScriptedReader(["use a coarser mesh first"]))
    install_model(loop, a_job_then_done())
    loop.say("go")
    loop.run()

    assert not backend.started
    (result,) = tool_results(loop)
    assert result["is_error"] is True
    assert "declined" in result["content"]
    assert "use a coarser mesh first" in result["content"]


def test_partial_lets_everything_else_run_freely(loop, backend, view):
    loop.ctx.mode = "partial"
    reader = ScriptedReader([])
    loop.approver = Approver(view, reader)
    install_model(loop, [
        message([tool_block("bash", {"cmd": "ls"}),
                 tool_block("write_file", {"path": "/work/a", "content": "x"}, "tu_2")],
                stop_reason="tool_use"),
        message([text_block("ok")]),
    ])
    loop.say("go")
    loop.run()
    assert "ls" in backend.execs and "/work/a" in backend.files
    assert not any(r.get("is_error") for r in tool_results(loop))


def test_approve_all_switches_the_session_to_full_auto(loop, backend, view, store):
    loop.ctx.mode = "partial"
    loop.approver = Approver(view, ScriptedReader(["/all"]))
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "blockMesh"})], stop_reason="tool_use"),
        message([tool_block("job_start", {"cmd": "simpleFoam"}, "tu_2")], stop_reason="tool_use"),
        message([text_block("both")]),
    ])
    loop.say("go")
    loop.run()

    assert [s["cmd"] for s in backend.started] == ["blockMesh", "simpleFoam"]
    assert loop.ctx.mode == "auto" == store.session.mode
    assert any("full auto" in t for t in texts_in_results(loop))


def test_structured_holds_compute_until_a_plan_is_approved(loop, backend, view, store):
    loop.ctx.mode = "structured"
    loop.approver = Approver(view, ScriptedReader(["/yes"]))
    loop.ctx.approver = loop.approver
    fake = install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam"})], stop_reason="tool_use"),
        message([tool_block("checkpoint", {"stage": "plan", "summary": "mesh then solve",
                                           "next": "blockMesh"}, "tu_2")],
                stop_reason="tool_use"),
        message([tool_block("job_start", {"cmd": "blockMesh"}, "tu_3")], stop_reason="tool_use"),
        message([text_block("meshing")]),
    ])
    loop.say("go")
    loop.run()

    held, checkpoint, ran = tool_results(loop)
    assert held["is_error"] and "held" in held["content"]
    assert "approved" in checkpoint["content"] and "blockMesh" in checkpoint["content"]
    assert not ran.get("is_error")
    assert [s["cmd"] for s in backend.started] == ["blockMesh"]
    assert "checkpoint" in [t["name"] for t in fake.calls[0]["tools"]]


def test_a_checkpoint_answered_with_changes_carries_them_back(loop, backend, view):
    loop.ctx.mode = "structured"
    loop.ctx.approver = Approver(view, ScriptedReader(["make the inlet 2 m/s"]))
    install_model(loop, [
        message([tool_block("checkpoint", {"stage": "plan", "summary": "s", "next": "n"})],
                stop_reason="tool_use"),
        message([text_block("revising")]),
    ])
    loop.say("go")
    loop.run()

    (result,) = tool_results(loop)
    assert "make the inlet 2 m/s" in result["content"]
    assert loop.ctx.plan_approved is False


def test_an_interrupted_turn_keeps_a_fact_said_during_it(loop):
    loop.messages.append({"role": "assistant", "content": [tool_block("bash", {"cmd": "x"})]})
    loop._notes.append("The person switched this session to full auto mode.")
    loop.settle()
    content = loop.messages[-1]["content"]
    assert content[0]["type"] == "tool_result"
    assert "full auto" in content[-1]["text"]
