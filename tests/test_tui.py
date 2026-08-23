"""The terminal interface.

Textual runs its apps headlessly for tests, so the panes, the reactive state and the
worker thread are all exercised here without a terminal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from openreynolds.tui import JobsPane, OpenReynoldsApp, SessionBar, TuiReader, TuiView, _escape
from openreynolds.view import ConsoleView, View
from openreynolds.watch import NOTHING


def idle_app():
    return OpenReynoldsApp(lambda app: None)


# -- the seam ------------------------------------------------------------------


def test_both_views_satisfy_the_protocol():
    """The loop cannot tell which one it has, which is the point."""
    assert isinstance(ConsoleView(), View)
    assert isinstance(TuiView(idle_app()), View)


def test_the_two_views_implement_the_same_surface():
    surface = {n for n in dir(View) if not n.startswith("_")}
    for implementation in (ConsoleView, TuiView):
        missing = surface - {n for n in dir(implementation) if not n.startswith("_")}
        assert not missing, f"{implementation.__name__} is missing {missing}"


def test_model_output_is_never_treated_as_markup():
    """A model writing `[U]` or `[0]` must not open a style tag."""
    assert _escape("field [U] at [0]") == r"field \[U] at \[0]"


# -- panes ---------------------------------------------------------------------


async def test_the_session_bar_shows_what_the_session_is():
    async with idle_app().run_test() as pilot:
        bar = pilot.app.query_one("#bar", SessionBar)
        bar.study, bar.instance, bar.model = "20260823-x", "974f4406-11da", "claude-opus-5"
        bar.tokens, bar.fraction = 812_000, 0.812
        rendered = bar.render()

    assert "20260823-x" in rendered
    assert "974f4406" in rendered
    assert "claude-opus-5" in rendered
    assert "812,000 tokens" in rendered
    assert "81% of the window" in rendered


async def test_the_jobs_pane_says_jobs_outlive_the_session():
    async with idle_app().run_test() as pilot:
        jobs = pilot.app.query_one("#jobs", JobsPane)
        assert "none yet" in jobs.render()
        jobs.names = ["miter_medium", "mesh"]
        rendered = jobs.render()

    assert "miter_medium" in rendered and "mesh" in rendered
    assert "keep running if you leave" in rendered


async def test_the_panes_exist_and_the_input_has_focus():
    async with idle_app().run_test() as pilot:
        for pane in ("#conversation", "#activity", "#jobs", "#bar", "#prompt"):
            assert pilot.app.query_one(pane) is not None
        assert pilot.app.query_one("#prompt").has_focus


# -- input ---------------------------------------------------------------------


async def test_typing_reaches_the_session_and_is_echoed():
    app = idle_app()
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"that looks wrong")
        await pilot.press("enter")

    assert app.typed.get_nowait() == "that looks wrong"


async def test_an_empty_line_is_not_a_turn():
    app = idle_app()
    async with app.run_test() as pilot:
        await pilot.click("#prompt")
        await pilot.press("enter")
    assert app.typed.empty()


async def test_the_reader_stands_in_for_stdin():
    app = idle_app()
    reader = TuiReader(app)

    assert reader.poll() is NOTHING
    app.typed.put("this value looks off")
    assert reader.poll() == "this value looks off"
    assert reader.get(timeout=0.01) is None

    app.typed.put("another")
    assert reader.get(timeout=0.5) == "another"


async def test_quitting_releases_a_session_waiting_on_input():
    """Otherwise the worker thread blocks forever on a reader that never answers."""
    app = idle_app()
    async with app.run_test() as pilot:
        app.action_quit()
        await pilot.pause()
    assert app.typed.get_nowait() is None


# -- what the view writes ------------------------------------------------------


async def test_streamed_text_reaches_the_conversation_pane():
    app = idle_app()
    async with app.run_test() as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.header, "s1", "i1", "claude-opus-5", Path("/tmp/x"))
        await asyncio.to_thread(view.text_delta, "The pressure drop is 240 Pa.\n")
        await asyncio.to_thread(view.turn_end)
        await pilot.pause()
        bar = app.query_one("#bar", SessionBar)

    assert bar.study == "s1"
    assert bar.model == "claude-opus-5"


async def test_partial_lines_are_held_until_the_turn_ends():
    """Deltas arrive mid-word; a log line per token would be unreadable."""
    app = idle_app()
    async with app.run_test():
        view = TuiView(app)
        await asyncio.to_thread(view.text_delta, "the loss ")
        assert view._pending == "the loss "
        await asyncio.to_thread(view.text_delta, "coefficient is 1.2\n")
        assert view._pending == ""


async def test_usage_updates_the_bar():
    app = idle_app()
    async with app.run_test() as pilot:
        await asyncio.to_thread(TuiView(app).usage, 500_000, 0.5)
        await pilot.pause()
        bar = app.query_one("#bar", SessionBar)
    assert bar.tokens == 500_000
    assert bar.fraction == 0.5


async def test_watching_fills_the_jobs_pane():
    app = idle_app()
    async with app.run_test() as pilot:
        await asyncio.to_thread(TuiView(app).watching, ["miter_medium"])
        await pilot.pause()
        assert app.query_one("#jobs", JobsPane).names == ["miter_medium"]


async def test_tool_activity_is_separated_from_the_conversation():
    """Tool calls in the transcript bury what the agent is actually saying."""
    app = idle_app()
    async with app.run_test() as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.tool, "bash", "blockMesh")
        await asyncio.to_thread(view.tool_error, "not_found (404)")
        await pilot.pause()
        activity = app.query_one("#activity")
        conversation = app.query_one("#conversation")

    assert activity.lines, "tool activity was written"
    assert not conversation.lines, "and did not land in the conversation"


async def test_the_worker_thread_runs_the_session():
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def run(app):
        loop.call_soon_threadsafe(started.set)

    async with OpenReynoldsApp(run).run_test():
        await asyncio.wait_for(started.wait(), timeout=5)


async def test_a_session_that_raises_is_shown_not_swallowed():
    def explode(app):
        raise RuntimeError("backend went away")

    app = OpenReynoldsApp(explode)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert app.query_one("#conversation").lines, "the failure was written somewhere visible"


async def test_the_jobs_pane_shows_status_not_just_names():
    """It used to update only while watching, so a job started mid-turn was invisible."""
    app = idle_app()
    async with app.run_test() as pilot:
        await asyncio.to_thread(
            TuiView(app).jobs,
            [
                {"name": "solve", "status": "running", "end_reason": None},
                {"name": "mesh", "status": "killed", "end_reason": "sandbox_expired"},
            ],
        )
        await pilot.pause()
        rendered = app.query_one("#jobs", JobsPane).render()

    assert "1 running" in rendered
    assert "solve" in rendered and "mesh" in rendered
    assert "sandbox_expired" in rendered


async def test_thinking_stays_out_of_the_transcript_by_default():
    """Streamed in full it is hundreds of grey lines that bury the answer."""
    app = idle_app()
    async with app.run_test() as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.thinking_begin)
        await asyncio.to_thread(view.thinking_delta, "weighing snappyHexMesh against gmsh\n")
        await pilot.pause()
        conversation = app.query_one("#conversation")
        stage = app.query_one("#stage")

    assert not conversation.lines, "reasoning did not land in the conversation"
    assert "thinking" in stage.text


async def test_thinking_can_be_shown_on_request():
    app = idle_app()
    async with app.run_test() as pilot:
        app.action_toggle_thinking()
        view = TuiView(app)
        await asyncio.to_thread(view.thinking_begin)
        await asyncio.to_thread(view.thinking_delta, "weighing the options\n")
        await pilot.pause()
        assert app.query_one("#conversation").lines, "ctrl+t puts it in the log"


async def test_the_stage_line_says_what_is_happening():
    app = idle_app()
    async with app.run_test() as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.tool, "bash", "blockMesh")
        await pilot.pause()
        assert "blockMesh" in app.query_one("#stage").text
        await asyncio.to_thread(view.turn_end)
        await pilot.pause()
        assert app.query_one("#stage").text == "", "and clears when the turn ends"


async def test_quitting_is_flagged_so_the_caller_can_force_the_exit():
    """A worker thread inside a network call cannot be interrupted, and being unable
    to close the program is worse than an untidy shutdown."""
    app = idle_app()
    async with app.run_test() as pilot:
        assert app.quitting is False
        app.action_quit()
        await pilot.pause()
    assert app.quitting is True
    assert app.typed.get_nowait() is None
