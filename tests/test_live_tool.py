"""`live_sim`: the dict the model reads, and the handler behind it.

The description is the whole of what the model knows about this tool, and three of the
facts in it are ones no other tool in `TOOLS` has to carry: that the work happens on
the user's own machine and not in the workspace, that the speed depends on hardware the
model cannot see, and that `case=list` is how to find out what can be run. A
description missing any of them produces a specific failure -- a model that calls it
expecting workspace files, or at a resolution that takes an hour on a CPU, or guessing
case names -- so they are asserted rather than trusted to survive an edit.

The schema's own invariants (object type, a non-empty description, sorted position in
the list, a handler for the name) are checked for every tool at once in
`tests/test_tools.py::test_tool_list_is_deterministic` and
`test_every_tool_name_has_a_handler_and_every_handler_a_tool` below; this file only
adds what is particular to this one.
"""

from __future__ import annotations

import sys

import pytest

from openreynolds import live_tool
from openreynolds.tools import TOOLS, _HANDLERS

TOOL = live_tool.TOOL
SCHEMA = TOOL["input_schema"]


@pytest.fixture
def no_warp(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", None)


# -- the dict ------------------------------------------------------------------


def test_the_tool_is_in_the_list_once_and_has_a_handler():
    assert [tool["name"] for tool in TOOLS].count("live_sim") == 1
    assert TOOLS[[t["name"] for t in TOOLS].index("live_sim")] is TOOL
    assert _HANDLERS["live_sim"] is live_tool.handler


def test_every_tool_name_has_a_handler_and_every_handler_a_tool():
    """Both directions. A name in `TOOLS` with no handler is a tool the model is
    offered and cannot call (`dispatch` answers "No such tool"), and a handler with no
    tool is dead code that looks like a feature."""
    assert sorted(tool["name"] for tool in TOOLS) == sorted(_HANDLERS)


def test_the_description_says_it_does_not_run_in_the_workspace():
    """The fact with the most expensive failure. Every other tool here except `fetch`
    and `cad` runs a command in the study's workspace; a model that reads this one the
    same way looks for the figures on the far side of the sandbox and writes a report
    about files that are not there. `DECISIONS_LIVE.md` D6 is why it cannot run there:
    no Warp, no CUDA toolchain, no GPU in the image."""
    said = TOOL["description"]
    assert "user's own machine" in said
    assert "NOT in the study's workspace" in said


def test_the_description_says_what_a_missing_gpu_costs():
    """The model cannot see whether there is a GPU, and the difference is seconds
    against minutes. Stated as a fact about the two cases rather than as advice."""
    said = TOOL["description"]
    assert "GPU" in said and "CPU" in said
    assert "seconds" in said and "minutes" in said


def test_the_description_says_where_the_figures_land():
    said = TOOL["description"]
    assert "study's own directory" in said


def test_the_description_says_how_to_get_the_catalogue():
    """Without this the model has to guess case names, and a guessed name costs a
    whole round trip to learn the real ones."""
    assert "case=list" in TOOL["description"]


def test_the_description_does_not_repeat_the_shell_contract():
    """`tests/test_shell_contract.py` holds this for every tool that is not `bash` or
    `job_start`. Here it is not a style rule but the same fact as the first test: this
    tool starts no shell anywhere."""
    for fragment in ("fresh shell", "RunFunctions", "carries between calls"):
        assert fragment not in TOOL["description"]


def test_the_schema_asks_for_a_case_and_nothing_else():
    assert SCHEMA["type"] == "object"
    assert SCHEMA["required"] == ["case"]
    assert set(SCHEMA["properties"]) == {
        "case", "resolution", "steps", "collision", "fluid", "note"}
    assert SCHEMA["properties"]["resolution"]["type"] == "integer"
    assert SCHEMA["properties"]["steps"]["type"] == "integer"
    assert SCHEMA["properties"]["note"]["type"] == "string"
    for name, description in SCHEMA["properties"].items():
        assert description.get("description"), name


def test_the_enums_are_closed_and_match_the_backends_own():
    """An enum rather than free text, because a misspelled collision operator is a
    refusal the model pays a round trip for. The values are the same two sets the CLI's
    flags offer, which `tests/test_live_cli.py` ties back to `lattice.SCHEMES` and
    `units.FLUIDS`."""
    from openreynolds import cli

    assert SCHEMA["properties"]["collision"]["enum"] == list(cli.LIVE_COLLISIONS)
    assert SCHEMA["properties"]["fluid"]["enum"] == list(cli.LIVE_FLUIDS)


def test_the_note_is_recorded_and_not_passed_on():
    """It is there so a figure found later says what question it was drawn for. It is
    not an override, and `run_case` must not be handed a field no case reads."""
    assert "note" not in live_tool.OVERRIDES
    assert set(live_tool.OVERRIDES) == {"resolution", "steps", "collision", "fluid"}


# -- the handler ---------------------------------------------------------------


def test_the_handler_returns_the_refusal_when_the_extra_is_absent(ctx, no_warp):
    """The tool is withheld from the list on such a machine (`tools_for`), so this
    should be unreachable -- and it exists because the list is not the only way a call
    arrives. A replayed transcript, another harness or an MCP client can dispatch the
    name, and the answer to that is the install line rather than a traceback through a
    lattice kernel."""
    said = live_tool.handler(ctx, {"case": "cavity"})
    assert "pip install 'openreynolds[live]'" in said
    assert "optional extra" in said


def test_the_refusal_comes_back_as_a_result_and_not_as_an_error(ctx, no_warp):
    """Through `dispatch`, the way the loop calls it: `is_error` false, because
    nothing failed -- the tool was asked for something this machine cannot do and said
    so, which is an answer."""
    from openreynolds.tools import dispatch

    content, is_error = dispatch(ctx, "live_sim", {"case": "cavity"})
    assert is_error is False
    assert "pip install 'openreynolds[live]'" in content


def test_an_empty_case_is_answered_without_running_anything(ctx):
    said = live_tool.handler(ctx, {"case": "  "})
    assert "nothing was run" in said and "case=list" in said


def test_offered_is_the_one_question_the_tool_list_asks():
    from openreynolds import live

    assert live_tool.offered() == live.available()


# -- where the figures go ------------------------------------------------------


def test_the_figures_go_to_the_studys_own_renders_directory(ctx):
    """The flat, obvious home for pictures (`store.py`): the mirror already copies
    every workspace render there, so a live figure lands where the person and the
    interface both already look, and needs no fetch."""
    where = live_tool.figures_dir(ctx)
    assert where == ctx.store.renders_dir
    assert where.is_dir(), "made, not just named: matplotlib will not create it"


def test_a_context_with_no_store_writes_where_the_caller_is():
    """A harness, a test or an MCP client with no study behind it. The working
    directory rather than a temporary one: a figure written somewhere the caller cannot
    name is a figure nobody sees."""
    from pathlib import Path

    assert live_tool.figures_dir(object()) == Path.cwd()


def test_each_figure_is_announced_to_the_interface(ctx):
    seen = []
    ctx.on_render = seen.append
    live_tool.rendered(ctx, "figs/cavity.png")
    assert seen == ["figs/cavity.png"]


def test_a_panel_that_raises_does_not_cost_the_answer(ctx):
    """`on_render` is a nudge to whatever is drawing the session. The result was
    already computed, and the picture matters more than the panel (`tools._read_image`
    swallows the same failure for the same reason)."""
    def boom(path):
        raise RuntimeError("no panel")

    ctx.on_render = boom
    live_tool.rendered(ctx, "figs/cavity.png")  # no raise
