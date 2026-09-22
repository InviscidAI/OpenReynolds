"""The independent reviewer: a fresh model looks at the mesh and the harness decides.

What these pin is the harness around the model, not the model. The verdict is handed
over by a scripted provider, the pictures by a fake workspace, and what is under test is
what the reviewer does with them: that it renders through the staged toolbox and takes
it away again, that every view reaches the model in one message, that the decision rule
is the harness's and not the model's, that every failure of its own machinery is
`skipped` rather than a verdict, and that the words the desk and the main agent are
handed say what happened.

The desk-side loop -- a fail handed back as work, the rounds, the accepted third declare
-- is `cad/agent.py`'s and is tested with it.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass

import pytest

from openreynolds import images
from openreynolds.backend.base import ExecResult
from openreynolds.cad import review as review_module
from openreynolds.cad.review import (
    CONFIDENT,
    MAX_REPLY_TOKENS,
    REVIEW_SYSTEM,
    VERDICT_NAME,
    VERDICT_TOOL,
    VIEW_EDGE,
    Problem,
    Review,
    Reviewer,
    facts_from_check,
    facts_text,
)
from openreynolds.config import Config
from openreynolds.llm import ProviderError, TextBlock, ToolUseBlock, Turn

CASE = "/work/study/valve"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64 + b"IEND\xaeB`\x82"
"""Whole as far as `images.incomplete` is concerned, and too small to carry an IHDR,
so `images.downscale` hands it back untouched and the bytes can be compared."""

TWO_D_VIEWS = ["renders/review/overview.png", "renders/review/cells.png",
               "renders/review/detail.png"]

PAYLOAD = {"polymesh": True, "cells": 3750, "faces": 15000, "points": 7600,
           "two_d": True, "bounds": [0.0, 0.0, 0.0, 0.12, 0.03, 0.001],
           "views": TWO_D_VIEWS, "checkmesh": "",
           "patches": [{"name": "inlet", "type": "patch", "nFaces": 20,
                        "area": 3e-5, "normal": [-1.0, 0.0, 0.0]},
                       {"name": "outlet", "type": "patch", "nFaces": 20}]}

WRONG_LOBES = {"what": "the lobes are hexagons, not arcs", "seen_in": "detail.png, top left",
               "why_it_matters": "a Tesla valve's diodicity comes from smooth turning",
               "severity": "blocking"}
COARSE = {"what": "the mesh is coarse near the wall", "seen_in": "cells.png",
          "why_it_matters": "resolution, not shape", "severity": "note"}


# -- fakes ------------------------------------------------------------------------


@dataclass
class Verdict:
    """A scripted turn that answers through the tool."""

    input: dict
    prose: str = ""


class VerdictProvider:
    """Replays turns; repeats the last one if asked again.

    Each entry is a `Verdict`, a string (prose and no tool call), a `ToolUseBlock` by
    some other name, or an exception to raise. The thread is snapshotted per call, as
    `test_cad_agent.ScriptedProvider` does."""

    name = "anthropic"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []
        self.made = 0

    def stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        turn = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
        if isinstance(turn, Exception):
            raise turn
        self.made += 1
        if isinstance(turn, Verdict):
            content = ([TextBlock(text=turn.prose)] if turn.prose else []) + [
                ToolUseBlock(id=f"v-{self.made}", name=VERDICT_NAME, input=turn.input)]
            stop = "tool_use"
        elif isinstance(turn, ToolUseBlock):
            content, stop = [turn], "tool_use"
        else:
            content, stop = [TextBlock(text=str(turn))], "end_turn"
        return Turn(content=content, provider=self.name, stop_reason=stop,
                    tokens={"input": 100, "output": 7})


def staging(monkeypatch):
    """A `staged` that records what it was asked and yields the gate directory."""
    seen: list[dict] = []

    @contextlib.contextmanager
    def staged(backend, case_dir, scripts):
        seen.append({"case_dir": case_dir, "scripts": tuple(scripts), "open": True})
        try:
            yield f"{case_dir}/.gate"
        finally:
            seen[-1]["open"] = False

    monkeypatch.setattr(review_module, "staged", staged)
    return seen


def rendered(backend, payload=PAYLOAD, case_dir=CASE, report="valve\n  3,750 cells"):
    """Have the workspace answer the render command with this payload, and put the
    views it names on disk."""
    cmd = review_module._render_command(f"{case_dir}/.gate")
    body = f"{report}\n{review_module._JSON_MARK}\n{json.dumps(payload)}\n"
    backend.exec_results[cmd] = ExecResult(0, body, False, None)
    for rel in payload.get("views") or []:
        backend.files[f"{case_dir}/{rel}"] = PNG
    return cmd


def reviewer(backend, turns, on_step=None, **cfg_kwargs):
    cfg = Config(llm_api_key="k", model="claude-opus-5", **cfg_kwargs)
    made = Reviewer(cfg, backend, on_step=on_step)
    made.provider = VerdictProvider(turns)
    return made


def look(made, case_rel="valve", request="a Tesla valve, 4 lobes", said=None,
         summary="built the valve", facts=None, prior=None):
    return made.review(CASE, case_rel, request, said, summary, facts, prior=prior)


def sent(provider, call=0) -> list[dict]:
    """The content blocks of the one user message the reviewer opened with."""
    return provider.calls[call]["messages"][0]["content"]


def words(provider, call=0) -> str:
    return "\n".join(b.get("text", "") for b in sent(provider, call)
                     if b.get("type") == "text")


# -- the decision rule --------------------------------------------------------------


@pytest.mark.parametrize("verdict,confidence,problems,blocks", [
    ("fail", "sure", [WRONG_LOBES], True),
    ("fail", "likely", [WRONG_LOBES], True),
    ("fail", "unsure", [WRONG_LOBES], False),
    ("fail", "", [WRONG_LOBES], False),
    ("fail", "sure", [COARSE], False),
    ("fail", "sure", [], False),
    ("pass", "sure", [WRONG_LOBES], False),
    ("pass", "sure", [], False),
    ("skipped", "sure", [WRONG_LOBES], False),
])
def test_the_harness_decides_what_blocks_and_not_the_model(verdict, confidence, problems,
                                                          blocks):
    """The whole of the rule: a fail, with a blocking problem, the reviewer would bet on.
    Everything else is a pass carrying notes, which is what keeps a second opinion from
    becoming a veto over taste."""
    made = Review(verdict=verdict, confidence=confidence,
                  problems=[Problem.from_dict(p) for p in problems])
    assert made.blocks_finish() is blocks


def test_the_confident_words_are_the_two_the_schema_offers_short_of_unsure():
    assert set(CONFIDENT) == set(VERDICT_TOOL["input_schema"]["properties"]["confidence"]["enum"]) - {"unsure"}


def test_a_severity_that_is_not_blocking_is_a_note_by_construction():
    assert Problem.from_dict({**WRONG_LOBES, "severity": "major"}).severity == "note"
    assert Problem.from_dict({**WRONG_LOBES, "fixed": "maybe"}).fixed == ""
    assert Problem.from_dict({**WRONG_LOBES, "fixed": "not fixed"}).fixed == "not fixed"


# -- one review, end to end ---------------------------------------------------------


def test_a_review_renders_through_the_staged_toolbox_and_shows_every_view(backend,
                                                                          monkeypatch):
    """Staged and taken away again, on the advisory gate's argument: this desk is
    measured on having no toolbox. Three pictures, each named, all in the one message,
    and the verdict read off the tool call."""
    seen = staging(monkeypatch)
    cmd = rendered(backend)
    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure",
                                       "problems": [], "looked_fine": ["4 lobes"]},
                                      prose="Looks right.")],
                    review_model="claude-sonnet-5")
    result = look(made)

    assert seen == [{"case_dir": CASE, "scripts": ("mesh_look.py",), "open": False}]
    assert backend.last_exec == (cmd, CASE, review_module.REVIEW_TIMEOUT_S)
    assert "--no-check" in cmd and "--views renders/review" in cmd
    assert f"{CASE}/.gate/mesh_look.py" in cmd

    assert result.verdict == "pass" and result.confidence == "sure"
    assert result.looked_fine == ["4 lobes"] and result.summary == "Looks right."
    assert result.views == TWO_D_VIEWS
    assert result.png == PNG
    assert result.tokens == {"input": 100, "output": 7}
    assert result.round == 1 and not result.blocks_finish()

    call = made.provider.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["system"] == REVIEW_SYSTEM
    assert call["tools"] == [VERDICT_TOOL]
    assert call["max_tokens"] == MAX_REPLY_TOKENS
    assert len(call["messages"]) == 1, "a fresh thread: one user message and no history"
    blocks = sent(made.provider)
    assert [b["type"] for b in blocks] == ["text", "text", "image", "text", "image",
                                          "text", "image"]
    assert [b["text"] for b in blocks if b["type"] == "text"][1:] == [
        "view 1 of 3: overview.png", "view 2 of 3: cells.png", "view 3 of 3: detail.png"]
    assert all(b["source"]["media_type"] == "image/png" for b in blocks
               if b["type"] == "image")
    assert blocks[2]["source"]["data"] == images.attachment(PNG, "image/png")["source"]["data"]


def test_the_words_the_reviewer_reads_are_the_request_the_person_and_the_claims(
        backend, monkeypatch):
    staging(monkeypatch)
    rendered(backend)
    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure"})])
    look(made, request="a Tesla valve with four lobes",
         said=["build me a tesla valve", "the lobes must be round"],
         summary="Four circular lobes of radius 5 mm.",
         facts={"cells": 3750, "two_d": True, "bounds": [0, 0, 0, 0.12, 0.03, 0.001],
                "patches": [{"name": "inlet", "type": "patch", "nFaces": 20}]})
    text = words(made.provider)
    assert "a Tesla valve with four lobes" in text
    assert "build me a tesla valve" in text and "the lobes must be round" in text
    assert "Four circular lobes of radius 5 mm." in text
    assert "its claims, not measurements" in text
    assert "3,750 cells" in text and "120 x 30 x 1 mm" in text
    assert "inlet patch 20 faces" in text
    assert "re-review" not in text


def test_a_fail_the_reviewer_would_bet_on_blocks_and_reads_as_work(backend, monkeypatch):
    staging(monkeypatch)
    rendered(backend)
    made = reviewer(backend, [Verdict({"verdict": "fail", "confidence": "likely",
                                       "problems": [WRONG_LOBES, COARSE]})])
    result = look(made)
    assert result.verdict == "fail" and result.blocks_finish()
    assert [p.severity for p in result.problems] == ["blocking", "note"]

    work = result.as_work(1, 2)
    assert "looked at the mesh in 3 views and did not pass it (review 1 of 2)" in work
    assert "- the lobes are hexagons, not arcs (seen in detail.png, top left): a Tesla" in work
    assert "Also noted (not blocking):" in work
    assert work.index("hexagons") < work.index("Also noted") < work.index("coarse")
    assert "if the reviewer is mistaken say why in your closing summary" in work
    assert work.rstrip().endswith("then declare again.")


def test_facts_come_off_the_render_when_the_caller_has_none(backend, monkeypatch):
    """The standalone tool reviews a case nobody has checked, so the numbers beside
    the pictures come from the same payload that named the pictures."""
    staging(monkeypatch)
    rendered(backend)
    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure"})])
    look(made, facts=None)
    text = words(made.provider)
    assert "3,750 cells, 15,000 faces, 7,600 points (2D, one cell thick)" in text
    assert "outlet patch 20 faces" in text


# -- every failure of its own is skipped ------------------------------------------------


def test_a_render_that_writes_no_answer_is_skipped_with_the_reason(backend, monkeypatch):
    """Never a pass, never a fail: nothing was looked at."""
    staging(monkeypatch)
    cmd = review_module._render_command(f"{CASE}/.gate")
    backend.exec_results[cmd] = ExecResult(1, "Traceback: no module named pyvista\n",
                                           False, None)
    made = reviewer(backend, [Verdict({"verdict": "fail", "confidence": "sure",
                                       "problems": [WRONG_LOBES]})])
    result = look(made)
    assert result.verdict == "skipped" and not result.blocks_finish()
    assert result.error.startswith("could not render the mesh for review:")
    assert "no module named pyvista" in result.error
    assert made.provider.calls == [], "no picture, no question"
    assert result.lines() == [f"not reviewed: {result.error}"]


def test_a_staging_that_fails_is_skipped_rather_than_raised(backend, monkeypatch):
    @contextlib.contextmanager
    def broken(backend, case_dir, scripts):
        raise RuntimeError("the workspace would not take the script")
        yield  # noqa: unreachable - a generator all the same

    monkeypatch.setattr(review_module, "staged", broken)
    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure"})])
    result = look(made)
    assert result.verdict == "skipped"
    assert "would not take the script" in result.error


def test_views_that_cannot_be_fetched_are_left_out_and_none_at_all_is_skipped(
        backend, monkeypatch):
    staging(monkeypatch)
    rendered(backend)
    del backend.files[f"{CASE}/renders/review/cells.png"]
    backend.files[f"{CASE}/renders/review/detail.png"] = PNG[:-12]   # still being written
    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure"})])
    result = look(made)
    assert result.views == ["renders/review/overview.png"]
    assert sum(b["type"] == "image" for b in sent(made.provider)) == 1

    for rel in TWO_D_VIEWS:
        backend.files.pop(f"{CASE}/{rel}", None)
    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure"})])
    result = look(made)
    assert result.verdict == "skipped" and "could be fetched" in result.error


def test_a_model_failure_is_skipped_and_a_passing_one_is_tried_once_more(backend,
                                                                          monkeypatch):
    monkeypatch.setattr(review_module, "RETRY_PAUSE_S", 0)
    staging(monkeypatch)
    rendered(backend)
    made = reviewer(backend, [ProviderError("529 overloaded", 529),
                              Verdict({"verdict": "pass", "confidence": "sure"})])
    result = look(made)
    assert result.verdict == "pass" and len(made.provider.calls) == 2

    made = reviewer(backend, [ProviderError("400 bad request", 400)])
    result = look(made)
    assert result.verdict == "skipped" and "400 bad request" in result.error
    assert len(made.provider.calls) == 1, "a 400 is a fact about the request"


def test_a_reply_with_no_tool_call_is_asked_once_more_and_then_skipped(backend,
                                                                       monkeypatch):
    staging(monkeypatch)
    rendered(backend)
    made = reviewer(backend, ["I think it passes.", "Yes, it passes."])
    result = look(made)
    assert result.verdict == "skipped"
    assert result.error == "the reviewer gave no verdict"
    assert len(made.provider.calls) == 2
    second = made.provider.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[-1]["content"][0]["text"] == f"Answer through the {VERDICT_NAME} tool."
    assert result.tokens == {"input": 200, "output": 14}, "both turns are paid for"

    made = reviewer(backend, ["Let me answer properly.",
                              Verdict({"verdict": "pass", "confidence": "likely"})])
    assert look(made).verdict == "pass"


def test_a_call_by_another_name_is_answered_as_an_error_before_asking_again(backend,
                                                                            monkeypatch):
    """The API wants every tool_use answered, or the next request is a 400."""
    staging(monkeypatch)
    rendered(backend)
    made = reviewer(backend, [ToolUseBlock(id="odd", name="run_cell", input={"source": "1"}),
                              Verdict({"verdict": "pass", "confidence": "sure"})])
    assert look(made).verdict == "pass"
    answer = made.provider.calls[1]["messages"][-1]["content"][0]
    assert answer["type"] == "tool_result" and answer["tool_use_id"] == "odd"
    assert answer["is_error"] and VERDICT_NAME in answer["content"]


def test_a_verdict_that_is_neither_word_is_skipped(backend, monkeypatch):
    staging(monkeypatch)
    rendered(backend)
    made = reviewer(backend, [Verdict({"verdict": "maybe", "confidence": "sure"})])
    result = look(made)
    assert result.verdict == "skipped" and "'maybe'" in result.error


# -- the re-review ------------------------------------------------------------------------


def test_a_re_review_names_what_was_flagged_last_time(backend, monkeypatch):
    staging(monkeypatch)
    rendered(backend)
    prior = Review(verdict="fail", confidence="sure", round=1, views=TWO_D_VIEWS,
                   problems=[Problem.from_dict(WRONG_LOBES), Problem.from_dict(COARSE)])
    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure",
                                       "problems": [{**WRONG_LOBES, "severity": "note",
                                                     "fixed": "fixed"}]})])
    result = look(made, prior=prior)
    text = words(made.provider)
    assert "This is a re-review (round 2)" in text
    assert "the lobes are hexagons, not arcs" in text
    assert "the mesh is coarse near the wall" in text
    assert "`fixed` or `not fixed`" in text
    assert "negligent to miss" in text
    assert result.round == 2
    assert result.problems[0].fixed == "fixed"
    assert "[fixed]" in result.problems[0].bullet()


# -- what the main agent reads --------------------------------------------------------


def test_the_lines_say_pass_with_its_notes():
    made = Review(verdict="pass", confidence="sure", views=TWO_D_VIEWS,
                  problems=[Problem.from_dict(COARSE)])
    out = made.lines()
    assert out[0] == "independent review: PASS -- looked at 3 views"
    assert out[1].startswith("  notes: the mesh is coarse near the wall")


def test_the_lines_say_an_unsure_fail_did_not_bind_and_still_report_it():
    made = Review(verdict="fail", confidence="unsure", views=TWO_D_VIEWS,
                  problems=[Problem.from_dict(WRONG_LOBES)])
    out = made.lines()
    assert out[0] == "independent review: PASS -- looked at 3 views"
    assert "confidence unsure" in out[1] and "does not bind" in out[1]
    assert "hexagons" in out[2]


def test_the_lines_say_fail_and_unresolved_in_their_own_words():
    made = Review(verdict="fail", confidence="sure", views=TWO_D_VIEWS,
                  problems=[Problem.from_dict(WRONG_LOBES)])
    out = made.lines()
    assert out[0] == "independent review: FAIL -- looked at 3 views:"
    assert "hexagons" in out[1] and "seen in detail.png" in out[1]

    made.unresolved = True
    made.round = 2
    out = made.lines()
    assert out[0] == ("independent review: did not pass after 2 rounds -- treat this "
                      "mesh as suspect:")
    assert "hexagons" in out[1]


def test_the_lines_say_not_reviewed_when_it_was_skipped():
    assert Review(verdict="skipped", error="no pyvista").lines() == ["not reviewed: no pyvista"]


# -- the facts --------------------------------------------------------------------------


def test_facts_come_off_a_check_by_attribute_with_defaults():
    from openreynolds.cad.check import Check

    check = Check(cells=10, two_d=True, bounds=[0, 0, 0, 20.0, 5.0, 1.0],
                  checkmesh="Mesh OK.", regions=["air", "solid"],
                  patches=[{"name": "wall", "type": "wall", "nFaces": 4}])
    facts = facts_from_check(check)
    assert facts["cells"] == 10 and facts["faces"] == 0 and facts["two_d"]
    assert facts["regions"] == ["air", "solid"]
    text = facts_text(facts)
    assert "bounds 20 x 5 x 1 m" in text and "mm" not in text, "over 10 m: metres only"
    assert "regions: air, solid" in text
    assert "checkMesh: Mesh OK." in text and "not yours to re-decide" in text
    assert "wall wall 4 faces" in text

    text = facts_text({"bounds": [0, 0, 0, 0.074, 0.02, 0.001], "patches": [
        {"name": "inlet", "type": "patch", "nFaces": 3, "area": 2e-5,
         "normal": [-1, 0, 0]}]})
    assert "= 74 x 20 x 1 mm" in text
    assert "inlet patch 3 faces area 2e-05 m2 normal (-1.00 +0.00 +0.00)" in text


# -- the progress line ------------------------------------------------------------------


def test_the_hook_is_told_one_step_per_review_and_may_not_end_it(backend, monkeypatch):
    staging(monkeypatch)
    rendered(backend)
    steps: list = []
    made = reviewer(backend, [Verdict({"verdict": "fail", "confidence": "sure",
                                       "problems": [WRONG_LOBES]})], on_step=steps.append)
    result = look(made)
    assert len(steps) == 1
    step = steps[0]
    assert step.cmd == "[review] fail: the lobes are hexagons, not arcs"
    assert step.exit_code == 1 and step.image == "3 pictures"
    assert step.seconds == result.seconds
    assert "independent review: FAIL" in step.output

    def explode(step):
        raise RuntimeError("the renderer fell over")

    made = reviewer(backend, [Verdict({"verdict": "pass", "confidence": "sure"})],
                    on_step=explode)
    assert look(made).verdict == "pass"

    steps.clear()
    made = reviewer(backend, ["no tool call", "still none"], on_step=steps.append)
    look(made)
    assert steps[0].cmd == "[review] skipped: the reviewer gave no verdict"
    assert steps[0].exit_code == 0, "a skipped review is not a failure of the mesh"


# -- the brief ---------------------------------------------------------------------------


def test_the_brief_is_the_fair_critic_and_names_every_view():
    for phrase in ("You did not build it", "Look at every view", "would change the CFD answer",
                   "Tesla valve", "aerofoil", "20%", "one cell thick", "inlet and outlet",
                   "named gap or clearance", "symmetry", "mesh density", "checkMesh",
                   "left unstated", "rendering artefacts", "which view would settle it",
                   "Only fail if you would bet on it", "re-review", "`fixed` or `not fixed`",
                   "negligent to miss", "verdict` tool"):
        assert phrase in REVIEW_SYSTEM, phrase
    for view in ("overview.png", "cells.png", "detail.png", "iso.png", "ortho.png",
                 "cuts.png"):
        assert view in REVIEW_SYSTEM, view
    assert "+x, -x, +y, -y, +z, -z" in REVIEW_SYSTEM


def test_the_tool_schema_is_what_the_brief_promises():
    schema = VERDICT_TOOL["input_schema"]
    assert VERDICT_TOOL["name"] == VERDICT_NAME == "verdict"
    assert schema["required"] == ["verdict", "confidence"]
    assert schema["properties"]["verdict"]["enum"] == ["pass", "fail"]
    assert schema["properties"]["confidence"]["enum"] == ["sure", "likely", "unsure"]
    problem = schema["properties"]["problems"]["items"]
    assert problem["required"] == ["what", "seen_in", "why_it_matters", "severity"]
    assert problem["properties"]["severity"]["enum"] == ["blocking", "note"]
    assert problem["properties"]["fixed"]["enum"] == ["fixed", "not fixed", ""]
    assert "only way to answer" in VERDICT_TOOL["description"]
    assert VIEW_EDGE == 1024 and review_module.MAX_ROUNDS == 2


# -- the model it looks with --------------------------------------------------------------


def test_the_reviewer_looks_with_the_review_model_then_the_desks_then_the_mains(backend):
    assert reviewer(backend, []).model == "claude-opus-5"
    assert reviewer(backend, [], mesher_model="claude-sonnet-5").model == "claude-sonnet-5"
    assert reviewer(backend, [], mesher_model="claude-sonnet-5",
                    review_model="gpt-5").model == "gpt-5"
    made = reviewer(backend, [], mesher_effort="medium")
    assert made.effort == "medium" and made.enabled is True


# -- the configuration --------------------------------------------------------------------


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "config.json"))
    for name in ("OPENREYNOLDS_REVIEW_MODEL", "OPENREYNOLDS_CAD_REVIEW",
                 "OPENREYNOLDS_CAD_MODEL", "OPENREYNOLDS_MESHER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "config.json"


def test_the_review_model_and_the_switch_are_read_from_the_environment(clean_env,
                                                                        monkeypatch):
    cfg = Config.load()
    assert cfg.review_model == "" and cfg.cad_review is True

    monkeypatch.setenv("OPENREYNOLDS_REVIEW_MODEL", "gpt-5")
    monkeypatch.setenv("OPENREYNOLDS_CAD_REVIEW", "0")
    cfg = Config.load()
    assert cfg.review_model == "gpt-5" and cfg.cad_review is False

    monkeypatch.setenv("OPENREYNOLDS_CAD_REVIEW", "off")
    assert Config.load().cad_review is False
    monkeypatch.setenv("OPENREYNOLDS_CAD_REVIEW", "1")
    assert Config.load().cad_review is True


def test_the_json_config_supplies_them_too_and_neither_is_persisted(clean_env):
    clean_env.write_text(json.dumps({"review_model": "claude-sonnet-5", "cad_review": "0"}))
    cfg = Config.load()
    assert cfg.review_model == "claude-sonnet-5" and cfg.cad_review is False
    written = json.loads(cfg.save().read_text())
    assert "review_model" not in written and "cad_review" not in written


def test_the_package_exports_the_reviewer():
    from openreynolds import cad

    assert cad.Reviewer is Reviewer and cad.Review is Review and cad.Problem is Problem
    assert cad.REVIEW_SYSTEM is REVIEW_SYSTEM
    for name in ("Reviewer", "Review", "Problem", "REVIEW_SYSTEM"):
        assert name in cad.__all__


# =======================================================================================
# -- the desk-side loop: a fail handed back as work, the rounds, the accepted third
#    declare. `cad/agent.py`'s half, driven with `test_cad_agent.py`'s fakes and a
#    reviewer that hands over scripted verdicts and records what it was asked.
# =======================================================================================

import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from test_buildup_gate import Declare, DeclaringProvider  # noqa: E402
from test_cad_agent import (  # noqa: E402
    DONE,
    PASSES,
    ScriptedProvider,
    block,
    checking,
    kernelled,
)

from openreynolds.cad import agent as agent_module  # noqa: E402
from openreynolds.cad.agent import STEP_TIMEOUT_S, CadDesk  # noqa: E402
from openreynolds.cad.check import Check  # noqa: E402
from openreynolds.cad.review import MAX_ROUNDS, REVIEW_TIMEOUT_S  # noqa: E402

REVIEW_TOKENS = {"input": 300, "output": 9}

FAIL = dict(verdict="fail", confidence="sure", problems=[WRONG_LOBES, COARSE],
            views=TWO_D_VIEWS, png=PNG, tokens=REVIEW_TOKENS)
"""A fail the harness acts on: blocking, and the reviewer would bet on it."""
PASS = dict(verdict="pass", confidence="sure", problems=[COARSE], views=TWO_D_VIEWS,
            png=PNG, tokens=REVIEW_TOKENS)
UNSURE = dict(verdict="fail", confidence="unsure", problems=[WRONG_LOBES],
              views=TWO_D_VIEWS, png=PNG, tokens=REVIEW_TOKENS)
NOTES_ONLY = dict(verdict="fail", confidence="sure", problems=[COARSE],
                  views=TWO_D_VIEWS, png=PNG, tokens=REVIEW_TOKENS)
SKIPPED = dict(verdict="skipped", error="could not render the mesh for review: no pyvista")

COMPLETE = Declare({"outcome": "complete"})


class FakeReviewer:
    """Hands over scripted verdicts in order (the last repeats) and records each ask.

    A fresh `Review` per call, the way the real one makes one, so two fails in a row
    are two objects and marking the last `unresolved` cannot leak into the first. The
    `round` is set as `Reviewer.review` sets it, off `prior`. `advance` moves a fake
    clock while reviewing, which is how the budget credit is measured without
    sleeping."""

    def __init__(self, verdicts, advance=0.0, clock=None):
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []
        self.advance = advance
        self.clock = clock

    def review(self, case_dir, case_rel, request, said, summary, facts, prior=None):
        self.calls.append(dict(case_dir=case_dir, case_rel=case_rel, request=request,
                               said=said, summary=summary, facts=facts, prior=prior))
        spec = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        if isinstance(spec, Exception):
            raise spec
        if self.clock is not None and self.advance:
            self.clock.now += self.advance
        made = Review(round=(prior.round + 1) if prior else 1)
        for key, value in spec.items():
            if key == "problems":
                value = [Problem.from_dict(p) for p in value]
            setattr(made, key, value)
        return made


def reviewed(backend, store, monkeypatch, turns, verdicts, *, check=PASSES,
             reviewer=None, on_turn=None, **cfg_kwargs):
    """A `CadDesk` with a reviewer wired, its finish check stubbed to `check`, and a
    provider that replays `turns` -- `Declare`s and cell sources through
    `DeclaringProvider`, or `test_cad_agent`'s `block`/`DONE` shorthand through
    `ScriptedProvider` when any turn is one of those."""
    checking(monkeypatch, check)
    kernelled(backend)
    if reviewer is None and verdicts is not None:
        reviewer = FakeReviewer(verdicts)
    made = CadDesk(Config(llm_api_key="k", model="claude-opus-5", **cfg_kwargs),
                   backend, store, "/work/study", on_turn=on_turn, reviewer=reviewer)
    legacy = any(not isinstance(t, (Declare, str)) for t in turns)
    made.provider = ScriptedProvider(turns) if legacy else DeclaringProvider(turns)
    return made


def handed(provider, call: int) -> dict:
    """The tool_result block the desk answered the previous turn's call with, as it
    was in front of the model on the way into `call`."""
    return provider.calls[call]["messages"][-1]["content"][0]


# -- a fail is work, and a pass after it is the finish -------------------------------------


def test_a_blocking_fail_comes_back_as_work_and_the_next_declare_is_reviewed_again(
        backend, store, monkeypatch):
    """The loop the plan describes, on the declare path. `checkMesh` and the gates
    passed the first declare; the reviewer did not, so the desk is handed the problems
    as a failing tool result and declares again; the second review sees the first as
    `prior` and passes, and that pass is what decided the finish."""
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [FAIL, PASS])
    result = made.run("a Tesla valve, 4 lobes")

    assert result.ok and result.stopped == ""
    assert result.review is not None and result.review.verdict == "pass"
    assert len(result.reviews) == 2 and result.review is result.reviews[1]
    assert [r.verdict for r in result.reviews] == ["fail", "pass"]
    assert [r.round for r in result.reviews] == [1, 2]
    assert not result.reviews[0].unresolved, "it was answered, not run out on"
    assert len(made._declares) == 2, "the second declare is what the pass was on"

    reviewer = made.reviewer
    assert len(reviewer.calls) == 2
    assert reviewer.calls[0]["prior"] is None
    assert reviewer.calls[1]["prior"] is result.reviews[0]
    ask = reviewer.calls[0]
    assert ask["case_dir"] == "/work/study/cad" and ask["case_rel"] == "cad"
    assert ask["request"] == "a Tesla valve, 4 lobes"
    assert ask["summary"] == "working", "the desk's own closing words beside the declare"
    assert ask["facts"]["cells"] == 3750 and ask["facts"]["two_d"] is True
    assert ask["facts"]["patches"][0]["name"] == "inlet"
    assert isinstance(ask["said"], list)

    # What the desk was told, exactly where a failing check would have told it.
    answer = handed(made.provider, 1)
    assert answer["type"] == "tool_result" and answer["is_error"] is True
    assert answer["tool_use_id"] == "d-1"
    work = answer["content"]
    assert work == result.reviews[0].as_work(1, MAX_ROUNDS)
    assert "did not pass it (review 1 of 2)" in work
    assert "the lobes are hexagons, not arcs (seen in detail.png, top left)" in work
    assert "Also noted (not blocking):" in work and "coarse near the wall" in work
    assert work.rstrip().endswith("then declare again.")


def test_the_legacy_finish_cell_is_reviewed_the_same_way(backend, store, monkeypatch):
    """The other accept point. `print("CAD_DONE")` still finishes a run, so a reviewer
    that only sat behind `declare_complete` would be one the shipped sentinel walked
    past. Same loop, same words, answered on the cell's own tool call."""
    made = reviewed(backend, store, monkeypatch, [block("body = 1"), DONE], [FAIL, PASS])
    result = made.run("a Tesla valve")

    assert result.ok and result.review.verdict == "pass"
    assert len(result.reviews) == 2 and len(made.reviewer.calls) == 2
    assert made.reviewer.calls[1]["prior"] is result.reviews[0]
    assert len(result.steps) == 1, "the finish cell is not a step, and ran in no kernel"
    assert backend.kernel.cells == ["body = 1"]
    answer = handed(made.provider, 2)
    assert answer["type"] == "tool_result" and answer["is_error"] is True
    assert "review 1 of 2" in answer["content"] and "hexagons" in answer["content"]


# -- the rounds run out, and the loop closes --------------------------------------------


def test_after_two_fails_the_third_declare_is_accepted_with_the_concern_unresolved(
        backend, store, monkeypatch):
    """Bounded by construction. The reviewer is asked `MAX_ROUNDS` times and not a
    third; the declare after that stands, `ok`, and the last failing review travels up
    marked `unresolved` -- the main agent and the person are told, the desk is not
    argued with forever."""
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [FAIL])
    result = made.run("a Tesla valve")

    assert result.ok, "the third declare is accepted"
    assert len(made.reviewer.calls) == MAX_ROUNDS == 2
    assert len(result.reviews) == 2 and len(made._declares) == 3
    assert result.review is result.reviews[-1]
    assert result.review.unresolved is True and result.review.verdict == "fail"
    assert result.review.round == 2
    assert result.reviews[0].unresolved is False, "only the one that counted"
    assert result.review.lines()[0] == (
        "independent review: did not pass after 2 rounds -- treat this mesh as suspect:")

    assert "review 1 of 2" in handed(made.provider, 1)["content"]
    assert "review 2 of 2" in handed(made.provider, 2)["content"]
    # The re-review was a re-review: it was handed the first as `prior`.
    assert made.reviewer.calls[1]["prior"] is result.reviews[0]


# -- what does not block ---------------------------------------------------------------------


def test_a_skipped_review_lets_the_finish_stand_in_one_declare(backend, store, monkeypatch):
    """No pyvista, no render, no verdict: never a pass, never a fail, and never a
    reason to keep the desk working. The result carries the skip so the tool result
    can say "not reviewed"."""
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [SKIPPED])
    result = made.run("a duct")
    assert result.ok and len(made._declares) == 1
    assert result.review.verdict == "skipped" and "no pyvista" in result.review.error
    assert len(result.reviews) == 1 and len(made.reviewer.calls) == 1
    assert result.review.lines() == [f"not reviewed: {result.review.error}"]


@pytest.mark.parametrize("spec", [UNSURE, NOTES_ONLY], ids=["unsure", "notes-only"])
def test_a_fail_the_harness_does_not_act_on_is_a_pass_with_notes(backend, store,
                                                                    monkeypatch, spec):
    """`blocks_finish` is the whole of the rule, and the desk consults nothing else:
    an unsure fail and a fail with only notes are both accepted first time."""
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [spec])
    result = made.run("a duct")
    assert result.ok and len(made._declares) == 1
    assert result.review.verdict == "fail" and not result.review.blocks_finish()
    assert result.review.unresolved is False
    assert result.review.lines()[0] == "independent review: PASS -- looked at 3 views"


def test_a_reviewer_that_raises_is_recorded_as_skipped_and_the_run_finishes(
        backend, store, monkeypatch):
    """`Reviewer.review` never raises by contract. One that does anyway is a fault in
    the second opinion, and a fault there may not end the build it was asked about."""
    made = reviewed(backend, store, monkeypatch, [COMPLETE],
                    [RuntimeError("the renderer fell over")])
    result = made.run("a duct")
    assert result.ok and result.review.verdict == "skipped"
    assert "the renderer fell over" in result.review.error
    assert result.reviews == [result.review]


# -- no reviewer, and the paths nobody reviews ----------------------------------------------


def test_with_no_reviewer_wired_the_desk_is_what_it_was(backend, store, monkeypatch):
    made = reviewed(backend, store, monkeypatch, [COMPLETE], None)
    assert made.reviewer is None
    result = made.run("a duct")
    assert result.ok and len(made._declares) == 1
    assert result.review is None and result.reviews == []

    made = reviewed(backend, store, monkeypatch, [block("body = 1"), DONE], None)
    result = made.run("a duct")
    assert result.ok and result.review is None and result.reviews == []


def test_a_refusal_is_not_reviewed(backend, store, monkeypatch):
    """There is nothing to look at, and the refusal is the work."""
    made = reviewed(backend, store, monkeypatch,
                    [Declare({"outcome": "refuse", "reason": "the STEP declares no unit"})],
                    [PASS])
    result = made.run("mesh this STEP")
    assert result.stopped == "refused" and not result.ok
    assert made.reviewer.calls == [] and result.reviews == [] and result.review is None


def test_a_run_that_stopped_on_its_budget_is_checked_but_not_reviewed(backend, store,
                                                                       monkeypatch):
    """The end-of-run `checkMesh` still runs -- an unexamined mesh is the failure this
    desk exists to end -- but the reviewer judges a shape the desk said was finished,
    and this desk said no such thing. `result.review` stays None and the tool result
    says "not reviewed"."""
    made = reviewed(backend, store, monkeypatch, [block("body = 1")], [PASS],
                    mesher_max_steps=2)
    result = made.run("a duct")
    assert result.stopped == "steps"
    assert result.ok and result.check is PASSES, "checked, on what was on disk"
    assert made.reviewer.calls == [] and result.reviews == [] and result.review is None


# -- the budget --------------------------------------------------------------------------------


def ticking(monkeypatch):
    """A clock the desk reads instead of the real one, so a review that "takes" 100 s
    takes none and the arithmetic is exact."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(agent_module, "time", SimpleNamespace(
        monotonic=lambda: clock.now,
        sleep=lambda s: setattr(clock, "now", clock.now + s)))
    return clock


def test_time_spent_reviewing_is_credited_back_to_the_desks_budget(backend, store,
                                                                     monkeypatch):
    """The reviewer's render and model call are the harness's caution, not the desk's
    work. A 60 s budget and a review that takes 100 s: without the credit the loop
    would break on `time` before the desk could act on the fail. With it the desk is
    handed the work, fixes it, and finishes -- while `result.seconds` still reads the
    wall clock the caller actually waited."""
    clock = ticking(monkeypatch)
    reviewer = FakeReviewer([FAIL, PASS], advance=100.0, clock=clock)
    made = reviewed(backend, store, monkeypatch, [COMPLETE], None, reviewer=reviewer,
                    mesher_max_seconds=60.0)
    result = made.run("a Tesla valve")

    assert result.stopped == "", f"stopped on {result.stopped!r}: the review was charged"
    assert result.ok and result.review.verdict == "pass"
    assert len(reviewer.calls) == 2
    assert result.seconds >= 200.0, "two reviews of 100 s each, on the wall clock"
    assert made._started - 1000.0 == pytest.approx(200.0), "and both credited"


def test_without_the_credit_the_same_run_would_have_stopped_on_time(backend, store,
                                                                      monkeypatch):
    """The control: the test above is only evidence if the clock it moves is the one
    the loop reads. The same 100 s spent on a cell instead of a review is charged."""
    clock = ticking(monkeypatch)
    made = reviewed(backend, store, monkeypatch, [block("body = 1"), DONE], [PASS],
                    mesher_max_seconds=60.0)
    real_run = backend.kernel_run

    def kernel_run(code, timeout_s=STEP_TIMEOUT_S):
        clock.now += 100.0
        return real_run(code, timeout_s)

    backend.kernel_run = kernel_run
    result = made.run("a duct")
    assert result.stopped == "time"
    assert made.reviewer.calls == []


def test_the_watcher_is_told_a_review_is_a_long_quiet_stretch(backend, store, monkeypatch):
    """`_mark("review", ...)` for the reason `_mark("gates", ...)` exists: the render and
    the call are turn-free, and a staleness threshold that knows only turns would call
    the silence a stall. The reviewer reports its own step line, so the desk emits no
    second one."""
    beats: list[dict] = []
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [PASS],
                    on_turn=lambda **kw: beats.append(kw))
    steps: list = []
    made.on_step = steps.append
    made.run("a duct")
    marks = [b for b in beats if b.get("phase") == "review"]
    assert len(marks) == 1 and marks[0]["expect_s"] == float(REVIEW_TIMEOUT_S)
    assert steps == [], "the desk ran no cell and posted no step of its own for the review"


# -- what travels back -----------------------------------------------------------------------


def test_the_reviews_first_view_is_the_picture_when_the_check_drew_none(backend, store,
                                                                         monkeypatch):
    """The shipped desk's finish check draws nothing, so without this the main agent
    got words about a shape and no picture of it."""
    bare = Check(ok=True, cells=3750, two_d=True, checkmesh="Mesh OK.", regions=[""])
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [PASS], check=bare)
    result = made.run("a duct")
    assert result.ok and result.png == PNG

    # And when the check did draw, its picture is the one that goes back.
    drew = Check(ok=True, cells=3750, regions=[""], render="cad/renders/cad_look.png",
                 render_abs="/work/study/cad/renders/cad_look.png")
    backend.files["/work/study/cad/renders/cad_look.png"] = b"the check's own"
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [PASS], check=drew)
    assert made.run("a duct").png == b"the check's own"

    # A skipped review has no picture to offer, and none is invented.
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [SKIPPED], check=bare)
    assert made.run("a duct").png is None


def test_the_reviews_tokens_land_in_the_results(backend, store, monkeypatch):
    """One declare at 10/5 from the desk's provider, one review at 300/9."""
    made = reviewed(backend, store, monkeypatch, [COMPLETE], [PASS])
    result = made.run("a duct")
    assert result.tokens == {"input": 310, "output": 14}

    made = reviewed(backend, store, monkeypatch, [COMPLETE], [FAIL, PASS])
    result = made.run("a duct")
    assert result.tokens == {"input": 620, "output": 28}, "two turns and two reviews"


# -- the wiring and the briefs ---------------------------------------------------------------


def test_the_core_desk_takes_the_reviewer_through_the_same_keyword(backend, store):
    from openreynolds.cad.core import CoreDesk

    seen = FakeReviewer([PASS])
    cfg = Config(llm_api_key="k", model="claude-opus-5")
    made = CoreDesk(cfg, backend, store, "/work/study", reviewer=seen)
    assert made.reviewer is seen and made.toolbox == ""
    assert CoreDesk(cfg, backend, store, "/work/study").reviewer is None


def test_both_briefs_tell_the_desk_a_reviewer_exists_and_what_it_may_do():
    """A desk handed a reviewer's problems as work that was never told a reviewer
    exists reads them as a check it cannot find. The paragraph names the move it has
    -- say why in the closing summary -- and the bound, so it is not a veto."""
    from openreynolds.cad import brief as cad_brief
    from openreynolds.cad import core

    for text in (core.brief(), cad_brief.system_prompt(240)):
        flat = " ".join(text.split())
        for phrase in ("an independent reviewer looks at the mesh", "did not build the shape",
                       "several views", "confident change the answer", "wrong topology",
                       "sharp corners where the request implies curves",
                       "named gap one cell wide", "inlet and outlet on the wrong ends",
                       "exactly as a failing check does, at most twice",
                       "if the reviewer is mistaken say why in your closing summary",
                       "the third declare is accepted", "reported up with the result"):
            assert phrase in flat, phrase
        assert flat.count("independent reviewer") == 1
    # The core brief still names no house surface: the reviewer is described by what it
    # does, not by the script that renders for it.
    assert "toolbox" not in core.CORE_SYSTEM.lower()
    assert "mesh_look" not in core.CORE_SYSTEM
    assert core.CORE_SYSTEM.count("# Finishing") == 1
    assert core.CORE_SYSTEM.index("independent reviewer") > core.CORE_SYSTEM.index("# Finishing")
