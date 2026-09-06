"""The geometry desk: its brief, its laps, its commit, and its seat in the tool surface.

The brief is pinned here and not by `test_prompt.py`: the desk is a tool with a narrow
job and its instructions are allowed to be instructions. What the main prompt says about
it stays under the free-will tests like everything else in the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openreynolds import geometry, tools
from openreynolds.config import Config
from openreynolds.geometry import GEOMETRY_SYSTEM, GeometryAgent, GeometryResult, extract_spec
from openreynolds.llm.base import TextBlock, Turn
from openreynolds.tools import ToolContext

from conftest import FakeMessages, message, text_block

GOOD = {"ops": [{"op": "rect", "name": "body", "origin": [0, 0], "size": [60, 3]}],
        "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"}],
        "scale": 0.001}
BAD = {"ops": [{"op": "rect", "name": "body", "origin": [0, 0], "size": [0, 3]}]}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def cfg(**over):
    base = dict(llm_api_key="k", model="claude-sonnet-5")
    base.update(over)
    return Config(**base)


class FakeProvider:
    """Scripted replies in the provider's own shape: a `Turn` per lap, with usage."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.client = None

    def stream(self, **kwargs):
        # the loop appends to its message list after the call; keep what was sent
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return Turn(content=[TextBlock(reply)], tokens={"input": 100, "output": 10})


class Recording(GeometryAgent):
    """The real loop over a fake builder: a spec with a zero size is refused (as the
    tool would refuse it), anything else draws a picture and measures."""

    def __init__(self, cfg, backend, store, home, replies, available=True):
        super().__init__(cfg, backend, store, home)
        self._provider = FakeProvider(replies)
        self.builds: list = []
        self.written: list = []

    def _build(self, mode, spec, work, scale):
        self.builds.append((mode, spec, scale))
        ops = spec.get("ops", spec) if isinstance(spec, dict) else spec
        if any(0 in op.get("size", [1]) for op in ops):
            return 1, "rect 'body': size must be positive", None
        return 0, f"extent 0.06 x 0.003 m, area 0.00018 m2, 0 islands (scale {scale})", PNG

    def _write_case(self, mode, spec, study, local, scale):
        local.mkdir(parents=True, exist_ok=True)
        (local / "Allmesh").write_text("#!/bin/sh\n")
        self.written.append((mode, spec, study, local, scale))


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(geometry, "unavailable", lambda: None)


# -- the brief ------------------------------------------------------------------


def test_the_brief_says_render_measure_compare_commit_and_the_cap():
    text = GEOMETRY_SYSTEM.lower()
    for word in ("picture", "report", "compare", "commit", "islands", "extent", "revised spec"):
        assert word in text, word
    assert "at most 6 laps" in text
    assert "never run a solver" in text and "never mesh" in text
    assert "disagrees:" in text


def test_the_brief_is_explicit_and_that_is_allowed_here():
    """The desk's brief is imperative on purpose; the main prompt's line about the tool
    is not. Both facts, checked in one place."""
    assert "reply with only a json spec" in GEOMETRY_SYSTEM.lower()
    from openreynolds.prompt import SYSTEM_PROMPT
    line = next(l for l in SYSTEM_PROMPT.splitlines() if l.startswith("- `geometry`"))
    assert "takes a shape described in words" in line


def test_the_main_model_is_the_default_and_a_setting_overrides_it(backend, store, available):
    assert GeometryAgent(cfg(), backend, store, "/work/s").model == "claude-sonnet-5"
    assert GeometryAgent(cfg(geometry_model="claude-opus-5"), backend, store, "/work/s").model == "claude-opus-5"


def test_preferences_ride_along_in_the_users_own_words(backend, store, available):
    agent = GeometryAgent(cfg(preferences="work in millimetres"), backend, store, "/work/s")
    assert agent._system().endswith("work in millimetres")
    assert "in their own words" in agent._system()
    assert GeometryAgent(cfg(), backend, store, "/work/s")._system() == GEOMETRY_SYSTEM


# -- reading a reply --------------------------------------------------------------


@pytest.mark.parametrize("text, spec, commit", [
    ('{"ops": [1]}', {"ops": [1]}, False),
    ('```json\n{"ops": [1]}\n```', {"ops": [1]}, False),
    ('Here it is:\n```\n[{"op": "rect"}]\n```', [{"op": "rect"}], False),
    ("COMMIT", None, True),
    ("commit\ndisagrees: the third loop overlaps", None, True),
    ("I cannot do that.", None, False),
    ('{"ops": [', None, False),
])
def test_a_reply_is_a_spec_or_a_commit_or_neither(text, spec, commit):
    assert extract_spec(text) == (spec, commit)


# -- the laps --------------------------------------------------------------------


def test_bad_spec_then_good_spec_then_commit(backend, store, available):
    agent = Recording(cfg(), backend, store, "/work/s", [json.dumps(BAD), json.dumps(GOOD), "COMMIT"])
    result = agent.run("a straight channel 60 mm by 3 mm", case="valve")

    assert result.error == "" and result.agreed and result.laps == 3
    assert [b[2] for b in agent.builds] == [1.0, 0.001], "scale is read off the spec"
    assert result.png == PNG and "0 islands" in result.report
    # the refusal came back verbatim and the picture went back as an image block
    calls = agent._provider.calls
    assert "size must be positive" in calls[1]["messages"][-1]["content"][0]["text"]
    picture = calls[2]["messages"][-1]["content"]
    assert picture[0]["type"] == "image" and picture[1]["type"] == "text"
    assert "COMPARE" in picture[1]["text"].upper()
    # every lap ran with no tools and the desk's own brief
    assert all(c["tools"] == [] and c["system"] == GEOMETRY_SYSTEM for c in calls)
    assert result.tokens == {"input": 300, "output": 30}
    # committed: the case written locally under the study's fetch dir, then shipped
    local = Path(store.fetch_dir()) / "valve"
    assert (local / "Allmesh").exists()
    assert agent.written[0][1] == GOOD and agent.written[0][2] == "mesh" and agent.written[0][4] == 0.001
    assert backend.trees == [(local, "/work/s/valve")]
    assert result.case_rel == "/work/s/valve"


def test_a_commit_before_any_build_is_not_a_commit(backend, store, available):
    agent = Recording(cfg(), backend, store, "/work/s", ["COMMIT", json.dumps(GOOD), "COMMIT"])
    result = agent.run("anything")
    assert result.laps == 3 and result.agreed
    nudge = agent._provider.calls[1]["messages"][-1]["content"][0]["text"]
    assert "not a spec" in nudge.lower()


def test_the_lap_cap_commits_the_last_good_spec_and_says_so(backend, store, available):
    agent = Recording(cfg(), backend, store, "/work/s", [json.dumps(GOOD)])
    result = agent.run("anything", case="c")
    assert result.laps == geometry.MAX_LAPS
    assert not result.agreed and result.disagrees == "" and result.error == ""
    assert backend.trees and agent.written[-1][1] == GOOD


def test_a_commit_that_still_disagrees_is_recorded(backend, store, available):
    agent = Recording(cfg(), backend, store, "/work/s",
                      [json.dumps(GOOD), "COMMIT\ndisagrees: the fourth loop is open"])
    result = agent.run("anything")
    assert not result.agreed and result.disagrees == "the fourth loop is open"
    assert result.case_rel.endswith("/geometry")


def test_the_time_cap_is_honoured(backend, store, available, monkeypatch):
    monkeypatch.setattr(geometry, "MAX_SECONDS", 0.0)
    agent = Recording(cfg(), backend, store, "/work/s", [json.dumps(GOOD), "COMMIT"])
    result = agent.run("anything")
    assert result.laps == 0 and "no spec built" in result.error and backend.trees == []


def test_a_failed_commit_is_reported_not_raised(backend, store, available):
    class Failing(Recording):
        def _write_case(self, *a):
            raise RuntimeError("gmsh: could not fuse")
    result = Failing(cfg(), backend, store, "/work/s", [json.dumps(GOOD), "COMMIT"]).run("x")
    assert "did not write" in result.error and "could not fuse" in result.error
    assert result.png == PNG and backend.trees == []


def test_a_model_failure_is_reported_not_raised(backend, store, available):
    from openreynolds.llm.base import ProviderError

    class Down(Recording):
        def _ask(self, messages):
            raise ProviderError("overloaded", 529)
    result = Down(cfg(), backend, store, "/work/s", ["x"]).run("x")
    assert "model call failed" in result.error and result.laps == 1


def test_without_gmsh_the_desk_says_why(backend, store, monkeypatch):
    monkeypatch.setattr(geometry.importlib.util, "find_spec",
                        lambda name: None if name == "gmsh" else object())
    assert "gmsh" in geometry.unavailable()
    result = Recording(cfg(), backend, store, "/work/s", ["x"]).run("x")
    assert "gmsh" in result.error and result.laps == 0


def test_the_client_seam_is_the_desks(backend, store, available):
    """The same replacement `test_desk.py` uses: the SDK client under the provider."""
    agent = Recording(cfg(), backend, store, "/work/s", [])
    agent._provider = geometry.make_provider(cfg())
    fake = FakeMessages([message([text_block(json.dumps(GOOD))]), message([text_block("COMMIT")])])
    agent._client = SimpleNamespace(messages=fake)
    result = agent.run("anything")
    assert result.agreed and result.laps == 2
    assert result.tokens["input"] == 200 and result.tokens["output"] == 20
    assert fake.calls[0]["model"] == "claude-sonnet-5" and fake.calls[0]["system"]


# -- the tool ---------------------------------------------------------------------


def test_geometry_is_the_eighth_tool_and_needs_only_words():
    tool = next(t for t in tools.TOOLS if t["name"] == "geometry")
    assert tool["input_schema"]["required"] == ["request"]
    assert set(tool["input_schema"]["properties"]) == {"request", "case", "mode", "study"}
    assert "geometry" in tools._HANDLERS


def test_the_tool_answers_with_the_picture_first_and_the_words_second(ctx):
    class Desk:
        def run(self, request, mode, study, case):
            self.args = (request, mode, study, case)
            return GeometryResult(report="extent 0.06 x 0.012 m, 4 islands", png=PNG,
                                  case_rel="/work/s/valve", laps=2, seconds=7.0,
                                  tokens={"input": 5}, agreed=True)
    spent = []
    ctx.geometry = Desk()
    ctx.on_tokens = spent.append
    out, failed = tools.dispatch(ctx, "geometry", {"request": "a tesla valve", "case": "valve"})
    assert not failed
    assert ctx.geometry.args == ("a tesla valve", "2d", "mesh", "valve")
    assert out[0]["type"] == "image" and out[1]["type"] == "text"
    text = out[1]["text"]
    assert "/work/s/valve" in text and "laps 2" in text and "4 islands" in text
    # the first live run read "case written" as "meshed" and went looking for a
    # checkMesh log that did not exist yet; the words now say what is still to do
    assert "Not yet an OpenFOAM mesh" in text and "sh Allmesh" in text
    assert spent == [{"input": 5}]
    # what survives eviction is the words, and they carry the measurements
    assert "4 islands" in tools.describe(out)


def test_the_tool_without_a_desk_says_where_the_same_thing_is(ctx):
    out, failed = tools.dispatch(ctx, "geometry", {"request": "a tesla valve"})
    assert not failed
    assert isinstance(out, str) and "mesh2d.py --spec" in out


def test_the_tool_reports_a_desk_that_did_not_agree(ctx):
    ctx.geometry = SimpleNamespace(run=lambda *a, **k: GeometryResult(
        report="r", case_rel="/work/s/geometry", laps=6, seconds=80.0, agreed=False,
        disagrees="the third loop overlaps the fourth"))
    out, _ = tools.dispatch(ctx, "geometry", {"request": "x"})
    assert "without agreeing: the third loop overlaps the fourth" in out


def test_the_session_wires_the_desk_only_where_it_can_run():
    source = (Path(geometry.__file__).parent / "cli.py").read_text(encoding="utf-8")
    assert "ctx.on_tokens = loop.add_tokens" in source
    assert "geometry.unavailable() is None" in source
    assert "ctx.geometry = geometry.GeometryAgent(cfg, backend, store, store.session.home)" in source


# -- for real ---------------------------------------------------------------------


TESLA = {"ops": [
    {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
    {"op": "channel", "name": "bypass", "width": 3, "start": [9, 1.5], "heading": 60,
     "path": [{"line": 2}, {"arc": {"radius": 3.5, "angle": 240}}, {"line": {"to": "y:1.5"}}]},
    {"op": "repeat", "name": "bypasses", "target": "bypass", "count": 4, "step": [12, 0]},
    {"op": "fuse", "name": "body", "of": ["main", "bypasses"]}],
    "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"}],
    "scale": 0.001}


def test_end_to_end_a_tesla_valve_lands_on_the_workspace(backend, store):
    """The real builder and the real commit, under a scripted model: the picture is
    drawn, the report counts four islands, the case is written and shipped."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")

    class Scripted(GeometryAgent):
        def __init__(self, *a):
            super().__init__(*a)
            self._provider = FakeProvider([json.dumps(TESLA), "COMMIT"])

    result = Scripted(cfg(), backend, store, "/work/s").run(
        "a Tesla valve, 60 mm long, 3 mm channel, 4 bypasses", case="tesla")
    assert result.error == "", result.error
    assert result.agreed and result.laps == 2
    assert "islands 4" in result.report or "4 islands" in result.report, result.report
    assert result.png and len(result.png) > 10_000
    local = Path(store.fetch_dir()) / "tesla"
    for name in ("Allmesh", "body.msh", "outline.png", "geometry.json",
                 "system/controlDict", "constant/geometry/body.step"):
        assert (local / name).exists(), name
    assert backend.trees == [(local, "/work/s/tesla")]
    assert json.loads((local / "geometry.json").read_text())["scale"] == 0.001
