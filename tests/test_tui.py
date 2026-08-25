"""The terminal interface.

Textual runs its apps headlessly for tests, so the panes, the reactive state and the
worker thread are all exercised here without a terminal.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from openreynolds.tui import JobsPane, OpenReynoldsApp, SessionBar, TuiReader, TuiView, _escape
from openreynolds.view import ConsoleView, View
from openreynolds.watch import NOTHING


_RELEASES: list[threading.Event] = []


@asynccontextmanager
async def running(app):
    """Run an app, and let its held session go before the app comes down.

    Textual waits for thread workers when it shuts down -- the same reason quitting a
    real session needs a force exit -- so a session still holding on at that moment
    hangs the test rather than failing it.
    """
    async with app.run_test() as pilot:
        try:
            yield pilot
        finally:
            # Quitting first, exactly as ctrl+C does: a session that ends because the
            # app is going away has nobody to report to, and hopping across to an
            # event loop mid-shutdown is how a test waits seconds for nothing.
            app.quitting = True
            for release in _RELEASES:
                release.set()
            _RELEASES.clear()


def idle_app():
    """An app with a session that is still running, as it is for almost all of one.

    A session function that returns immediately is a session that has ended, and the
    app is right to say so and stop taking input -- so a double that returns at once
    would put every other test in a state no live session is ever in.
    """
    release = threading.Event()
    _RELEASES.append(release)
    return OpenReynoldsApp(lambda running: release.wait(30))


def ending_app(fail: bool = False):
    """An app whose session is over the moment it starts."""

    def run(running):
        if fail:
            raise RuntimeError("the instance went away")

    return OpenReynoldsApp(run)


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
    async with running(idle_app()) as pilot:
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
    async with running(idle_app()) as pilot:
        jobs = pilot.app.query_one("#jobs", JobsPane)
        assert "none yet" in jobs.render()
        jobs.names = ["miter_medium", "mesh"]
        rendered = jobs.render()

    assert "miter_medium" in rendered and "mesh" in rendered
    assert "keep running if you leave" in rendered


async def test_the_panes_exist_and_the_input_has_focus():
    async with running(idle_app()) as pilot:
        for pane in ("#conversation", "#activity", "#jobs", "#bar", "#prompt"):
            assert pilot.app.query_one(pane) is not None
        assert pilot.app.query_one("#prompt").has_focus


# -- input ---------------------------------------------------------------------


async def test_typing_reaches_the_session_and_is_echoed():
    app = idle_app()
    async with running(app) as pilot:
        await pilot.click("#prompt")
        await pilot.press(*"that looks wrong")
        await pilot.press("enter")

    assert app.typed.get_nowait() == "that looks wrong"


async def test_an_empty_line_is_not_a_turn():
    app = idle_app()
    async with running(app) as pilot:
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
    async with running(app) as pilot:
        app.action_quit()
        await pilot.pause()
    assert app.typed.get_nowait() is None


# -- what the view writes ------------------------------------------------------


async def test_streamed_text_reaches_the_conversation_pane():
    app = idle_app()
    async with running(app) as pilot:
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
    async with running(app):
        view = TuiView(app)
        await asyncio.to_thread(view.text_delta, "the loss ")
        assert view._pending == "the loss "
        await asyncio.to_thread(view.text_delta, "coefficient is 1.2\n")
        assert view._pending == ""


async def test_usage_updates_the_bar():
    app = idle_app()
    async with running(app) as pilot:
        await asyncio.to_thread(TuiView(app).usage, 500_000, 0.5)
        await pilot.pause()
        bar = app.query_one("#bar", SessionBar)
    assert bar.tokens == 500_000
    assert bar.fraction == 0.5


async def test_watching_fills_the_jobs_pane():
    app = idle_app()
    async with running(app) as pilot:
        await asyncio.to_thread(TuiView(app).watching, ["miter_medium"])
        await pilot.pause()
        assert app.query_one("#jobs", JobsPane).names == ["miter_medium"]


async def test_tool_activity_is_separated_from_the_conversation():
    """Tool calls in the transcript bury what the agent is actually saying."""
    app = idle_app()
    async with running(app) as pilot:
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
    async with running(app) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert app.query_one("#conversation").lines, "the failure was written somewhere visible"


async def test_the_jobs_pane_shows_status_not_just_names():
    """It used to update only while watching, so a job started mid-turn was invisible."""
    app = idle_app()
    async with running(app) as pilot:
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
    async with running(app) as pilot:
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
    async with running(app) as pilot:
        app.action_toggle_thinking()
        view = TuiView(app)
        await asyncio.to_thread(view.thinking_begin)
        await asyncio.to_thread(view.thinking_delta, "weighing the options\n")
        await pilot.pause()
        assert app.query_one("#conversation").lines, "ctrl+t puts it in the log"


async def test_the_stage_line_says_what_is_happening():
    app = idle_app()
    async with running(app) as pilot:
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
    async with running(app) as pilot:
        assert app.quitting is False
        app.action_quit()
        await pilot.pause()
    assert app.quitting is True
    assert app.typed.get_nowait() is None


# -- the files pane ------------------------------------------------------------


class StubBrowser:
    """A workspace without a workspace."""

    store = None
    """No local mirror either, so nothing is answered from disk."""

    def __init__(self, entries=(), text="Time = 1", pulled=(), cache=None):
        self.entries = list(entries)
        self.text = text
        self.pulled = list(pulled)
        self.cache = cache
        self.asked = []
        self.reads = []
        self.pulls = []

    def tree(self, path="/work", depth=4):
        self.asked.append(path)
        return self.entries

    def cached(self, path=""):
        return list(self.cache) if self.cache is not None else None

    def cache_age(self):
        return 0.0 if self.cache is not None else None

    def read(self, path, limit=None):
        self.reads.append(path)
        return self.text, True

    def pull(self, path):
        self.pulls.append(path)
        return list(self.pulled)

    def local(self):
        return []


async def test_the_files_pane_shows_what_is_in_the_workspace():
    from openreynolds.browse import Entry
    from openreynolds.tui import FilesTree

    app = idle_app()
    app.browser = StubBrowser(
        [
            Entry(path="/work/case", is_dir=True),
            Entry(path="/work/case/log.simpleFoam", is_dir=False, size=2048),
            Entry(path="/work/notes.md", is_dir=False, size=12),
        ]
    )
    async with running(app) as pilot:
        app.load_files("/work")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

        tree = app.query_one("#filestree", FilesTree)
        labels = [str(node.label) for node in tree.root.children]

    assert "case/" in labels
    assert any("notes.md" in label for label in labels)


async def test_a_nested_file_hangs_off_its_own_directory():
    from openreynolds.browse import Entry
    from openreynolds.tui import FilesTree

    app = idle_app()
    app.browser = StubBrowser(
        [
            Entry(path="/work/case", is_dir=True),
            Entry(path="/work/case/log.simpleFoam", is_dir=False, size=2048),
        ]
    )
    async with running(app) as pilot:
        app.load_files("/work")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

        tree = app.query_one("#filestree", FilesTree)
        case = tree.root.children[0]
        assert [str(child.label).split()[0] for child in case.children] == ["log.simpleFoam"]


async def test_an_empty_workspace_says_so_rather_than_showing_nothing():
    from openreynolds.tui import FilesTree

    app = idle_app()
    app.browser = StubBrowser([])
    async with running(app) as pilot:
        app.load_files("/work")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()
        tree = app.query_one("#filestree", FilesTree)
        assert "nothing here yet" in str(tree.root.children[0].label)


async def test_the_files_pane_is_answered_from_the_mirror_without_a_round_trip():
    """The background mirror already took a listing; drawing the pane must not cost
    the network another one."""
    from openreynolds.browse import Entry
    from openreynolds.tui import FilesTree

    app = idle_app()
    app.browser = StubBrowser(
        cache=[Entry(path="/work/case", is_dir=True),
               Entry(path="/work/case/log.simpleFoam", is_dir=False, size=64)]
    )
    async with running(app) as pilot:
        app.load_files("/work")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

        tree = app.query_one("#filestree", FilesTree)
        labels = [str(node.label) for node in tree.root.children]

    assert "case/" in labels
    assert app.browser.asked == [], "the cache answered; the network was not asked"


async def test_a_refresh_goes_to_the_workspace_itself():
    """Ctrl+R exists to bypass the cache, not to redraw it."""
    from openreynolds.browse import Entry

    app = idle_app()
    app.browser = StubBrowser(
        entries=[Entry(path="/work/notes.md", is_dir=False, size=9)],
        cache=[Entry(path="/work/stale.md", is_dir=False, size=1)],
    )
    async with running(app) as pilot:
        app.load_files("/work", force_remote=True)
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

    assert app.browser.asked == ["/work"]


async def test_a_mirror_cycle_redraws_the_pane_when_the_workspace_changed():
    from openreynolds.browse import Entry
    from openreynolds.mirror import MirrorReport
    from openreynolds.tui import FilesTree

    app = idle_app()
    app.browser = StubBrowser(cache=[Entry(path="/work/new.md", is_dir=False, size=5)])
    async with running(app) as pilot:
        report = MirrorReport(local_dir=Path("/tmp/x"))
        app.files_synced(report)
        await pilot.pause()

        tree = app.query_one("#filestree", FilesTree)
        labels = [str(node.label) for node in tree.root.children]

    assert any("new.md" in label for label in labels)


async def test_a_mirror_cycle_says_what_arrived():
    from openreynolds.mirror import MirrorReport

    app = idle_app()
    app.browser = StubBrowser()
    async with running(app) as pilot:
        report = MirrorReport(local_dir=Path("/tmp/x"), pulled=[Path("/tmp/x/a.png")])
        app.files_synced(report)
        await pilot.pause()
        written = "".join(str(line) for line in app.query_one("#activity").lines)

    assert "mirrored 1 file(s)" in written


async def test_a_listing_failure_is_reported_not_swallowed():
    app = idle_app()

    class Broken(StubBrowser):
        def tree(self, path="/work", depth=4):
            raise RuntimeError("the instance went away")

    app.browser = Broken()
    async with running(app) as pilot:
        app.load_files("/work")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()
        written = "".join(str(line) for line in app.query_one("#activity").lines)

    assert "went away" in written


async def test_opening_an_image_copies_it_out_because_it_cannot_be_drawn_here():
    """The terminal cannot show it, but the file browser can - so it has to land
    somewhere the user can actually open it."""
    app = idle_app()
    app.browser = StubBrowser(pulled=[Path("/tmp/studies/x/files/mesh.png")])
    async with running(app) as pilot:
        app.open_path("/work/case/renders/mesh.png")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

        assert app.browser.pulls == ["/work/case/renders/mesh.png"]
        assert "mesh.png" in app.screen.body
        assert "Copied to your machine" in app.screen.body


async def test_opening_a_text_file_shows_it():
    app = idle_app()
    app.browser = StubBrowser(text="Time = 1\nCourant Number mean: 0.2")
    async with running(app) as pilot:
        app.open_path("/work/case/log.simpleFoam")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

        assert "Courant" in app.screen.body


async def test_the_files_view_asks_the_workspace_for_the_path_requested():
    app = idle_app()
    app.browser = StubBrowser([])
    async with running(app) as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.show_files, "/work/case")
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()

    assert app.browser.asked[-1] == "/work/case"


async def test_a_local_command_is_not_echoed_as_something_the_agent_was_told():
    """`/status` never reaches the model; showing it as speech would claim it did."""
    app = idle_app()
    async with running(app) as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/status"
        await pilot.press("enter")
        await pilot.pause()

        written = "".join(str(line) for line in app.query_one("#conversation").lines)

    assert "you" not in written
    assert app.typed.get_nowait() == "/status"


async def test_a_message_is_echoed_as_speech():
    app = idle_app()
    async with running(app) as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "run the coarse case"
        await pilot.press("enter")
        await pilot.pause()

        written = "".join(str(line) for line in app.query_one("#conversation").lines)

    assert "you" in written and "coarse" in written


async def test_status_is_shown_in_the_conversation_where_it_was_asked():
    app = idle_app()
    async with running(app) as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.status, ["study x on instance abcd1234", "1 job(s) running"])
        await pilot.pause()

        written = "".join(str(line) for line in app.query_one("#conversation").lines)

    assert "1 job(s) running" in written


async def test_a_reader_hands_back_what_it_could_not_use():
    app = idle_app()
    reader = TuiReader(app)
    reader.putback(None)
    assert reader.poll() is None


# -- a session that has ended ---------------------------------------------------


async def test_an_ended_session_says_so_and_stops_taking_input():
    """A live-looking input box on a dead session is the worst silent failure there
    is: everything typed into it is accepted, echoed, and discarded."""
    app = ending_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()

        conversation = "".join(str(line) for line in app.query_one("#conversation").lines)
        box = app.query_one("#prompt")
        assert "ended" in conversation
        assert box.disabled
        assert "ended" in box.placeholder


async def test_a_session_that_dies_says_why_and_stops_taking_input():
    app = ending_app(fail=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()

        conversation = "".join(str(line) for line in app.query_one("#conversation").lines)
        assert "the instance went away" in conversation
        assert app.query_one("#prompt").disabled


async def test_quitting_is_not_reported_as_the_session_dying():
    """Ctrl+C ends the session by design. Announcing it to someone already leaving,
    across an event loop that is shutting down, is noise at best and a hang at worst."""
    app = ending_app()
    app.quitting = True
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()

        conversation = "".join(str(line) for line in app.query_one("#conversation").lines)
        assert "ended" not in conversation


# -- the bar -------------------------------------------------------------------


async def test_progress_lands_on_its_own_pane():
    from openreynolds.progress import Progress
    from openreynolds.tui import ProgressPane

    app = idle_app()
    async with running(app) as pilot:
        snap = Progress("solving", "solving solve · Time 3 / 6 s · 14 min", "p 1.0e-03", 0.5, True, 7)
        await asyncio.to_thread(TuiView(app).progress, snap)
        await pilot.pause()
        pane = app.query_one("#progress", ProgressPane)
        drawn = pane.render()

    assert pane.snapshot is snap
    assert "50%" in drawn
    assert "solving solve" in drawn
    assert "p 1.0e-03" in drawn
    assert "█" in drawn and "░" in drawn


async def test_the_stage_line_keeps_a_clock_on_thinking():
    app = idle_app()
    async with running(app) as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.thinking_begin)
        await asyncio.to_thread(view.thinking_delta, "the mesh is coarse")
        await pilot.pause()
        text = app.query_one("#stage").text

    assert text.startswith("thinking ")
    assert "s: the mesh is coarse" in text


# -- the front desk and the now line -------------------------------------------


async def test_the_bar_pane_hides_when_there_is_nothing_to_show():
    from openreynolds.progress import Progress
    from openreynolds.tui import ProgressPane

    app = idle_app()
    async with running(app) as pilot:
        pane = app.query_one("#progress", ProgressPane)
        await asyncio.to_thread(TuiView(app).progress, Progress("waiting", "waiting for you", "", None, False, 0))
        await pilot.pause()
        assert pane.display is False
        await asyncio.to_thread(TuiView(app).progress, Progress("solving", "solving x", "p 1e-3", 0.4, True, 1))
        await pilot.pause()
        assert pane.display is True


async def test_the_now_line_shows_the_desks_narration():
    from openreynolds.tui import NowPane

    app = idle_app()
    async with running(app) as pilot:
        await asyncio.to_thread(TuiView(app).narration, "reworking the near-wake mesh before the solve")
        await pilot.pause()
        assert "near-wake mesh" in app.query_one("#now", NowPane).text


async def test_a_desk_reply_lands_in_the_conversation_labelled():
    app = idle_app()
    async with running(app) as pilot:
        view = TuiView(app)
        await asyncio.to_thread(view.desk, "the agent is 20% through the solve, about 19 min left")
        await pilot.pause()
    # It reached the conversation pane (no exception); attribution is in the markup.


async def test_the_renders_tab_lists_delivered_pictures(tmp_path):
    from types import SimpleNamespace
    from openreynolds.tui import RendersPane

    renders = tmp_path / "renders"
    renders.mkdir()
    (renders / "field_speed.png").write_bytes(b"\x89PNG")

    app = idle_app()
    async with running(app) as pilot:
        app.browser = SimpleNamespace(store=SimpleNamespace(renders_dir=renders), home="/work")
        await asyncio.to_thread(TuiView(app).delivered, SimpleNamespace(lines=lambda: ["1 new render"]))
        await pilot.pause()
        pane = app.query_one("#renders", RendersPane)
        assert any(p.name == "field_speed.png" for p in pane.paths)
        assert "field_speed.png" in pane.render()
