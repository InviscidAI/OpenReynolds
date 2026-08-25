"""The terminal interface.

A session has four things worth seeing at once: what the agent is saying, what it is
doing to the workspace, what is still running out on the instance, and what is in the
workspace at all. A scrolling log shows the first and buries the rest, so they get
their own panes.

The agent loop is synchronous and blocking, so it runs on a worker thread and reports
through `TuiView`, which is the same `View` the plain terminal implements. Nothing in
here can influence the model -- it is presentation, and the loop cannot tell which view
it has. The files pane reads the workspace directly for the same reason: looking at
something must not require asking the agent to fetch it.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)

from . import commands, images
from .browse import Entry, human
from .mirror import local_for
from .progress import BAR_WIDTH, Progress
from .progress import bar as draw_bar
from .view import View

TOOL_STYLE = {
    "bash": "cyan",
    "write_file": "green",
    "read_file": "blue",
    "job_start": "magenta",
    "job_check": "magenta",
    "job_kill": "red",
    "fetch": "yellow",
}


class SessionBar(Static):
    """Study, instance, model, and how full the thread is."""

    study = reactive("")
    instance = reactive("")
    model = reactive("")
    tokens = reactive(0)
    fraction = reactive(0.0)

    def render(self) -> str:
        used = f"{self.tokens:,} tokens"
        if self.fraction:
            used += f"  ({self.fraction * 100:.0f}% of the window)"
        return (
            f"[b]study[/b] {self.study}   [b]instance[/b] {self.instance[:8]}   "
            f"[b]model[/b] {self.model}\n[dim]{used}[/dim]"
        )


def _field(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


class JobsPane(Static):
    """What is running, updated the moment it changes rather than only while watching."""

    names: reactive[list[str]] = reactive(list)
    records: reactive[list[Any]] = reactive(list)

    MARKS = {
        "running": "[green]*[/green]",
        "exited": "[dim]done[/dim]",
        "killed": "[red]x[/red]",
        "unknown": "[yellow]?[/yellow]",
    }

    def render(self) -> str:
        rows: list[Any] = list(self.records) or [
            {"name": n, "status": "running"} for n in self.names
        ]
        if not rows:
            return "[b]jobs[/b]\n[dim]none yet[/dim]"
        live = [r for r in rows if _field(r, "status") == "running"]
        out = [f"[b]jobs[/b]  [dim]{len(live)} running[/dim]", ""]
        for row in rows[-8:]:
            status = _field(row, "status") or "running"
            name = _field(row, "name") or str(_field(row, "job_id") or "")[:8]
            reason = _field(row, "end_reason")
            tail = f"\n     [dim]{reason}[/dim]" if reason and reason != "completed" else ""
            out.append(f" {self.MARKS.get(status, '-')} {name}{tail}")
        if live:
            out.append("\n[dim]these keep running if you leave[/dim]")
        return "\n".join(out)


def _bar_worth_showing(snap: Progress) -> bool:
    """Whether the bar earns its two lines right now.

    The complaint was a bar that is always on: a pulsing strip that says "busy"
    tells you nothing you did not know. It earns its place when it carries a
    number or names a real job -- a solve, a mesh, a sync -- and not merely because
    a turn is in progress. Thinking and writing are the front desk's to narrate and
    the stream's to show; the bar stays out of it."""
    return snap.phase in ("solving", "meshing", "decomposing", "reconstructing",
                          "checking the mesh", "post-processing", "syncing")


class ProgressPane(Static):
    """The bar and its line: what is running and how far along it is.

    Kept separate from the stage line on purpose. The stage line is whatever
    happened last -- a thought, a tool name -- and it is overwritten by the next
    thing. This pane is the picture: a solve stays on it while the model thinks,
    with its percentage, its clock and its residuals, until the solve is over. When
    nothing is running it takes no room at all rather than pulsing at nothing.
    """

    snapshot: reactive[Progress] = reactive(Progress())

    def watch_snapshot(self, snap: Progress) -> None:
        show = _bar_worth_showing(snap)
        self.display = show
        self.styles.height = 2 if show else 0

    def render(self) -> str:
        snap = self.snapshot
        colour = "green" if snap.fraction is not None else "cyan"
        head = _escape(snap.headline)
        first = f"[{colour}]{draw_bar(snap)}[/{colour}] {snap.percent()}  [b]{head}[/b]"
        if not snap.detail:
            return first
        # Two lines is the pane; a detail that wraps into a third is clipped, so it
        # is cut where the pane ends instead of where the terminal does.
        room = max(20, (self.size.width or 120) - BAR_WIDTH - 9)
        detail = snap.detail if len(snap.detail) <= room else snap.detail[: room - 1] + "…"
        return f"{first}\n{'':{BAR_WIDTH + 7}}[dim]{_escape(detail)}[/dim]"


class NowPane(Static):
    """One plain-language line on what the agent is doing right now.

    This is the "what is happening?" the user kept asking for. It is written by the
    front desk (`desk.py`) from the transcript and the live job facts -- prose, not
    the mechanical phase labels the bar carries -- and it stays put until the desk
    replaces it, so it reads as a status and not as a flicker."""

    text = reactive("")

    def render(self) -> str:
        return f"[cyan]›[/cyan] [italic]{_escape(self.text)}[/italic]" if self.text else ""


class RendersPane(Static):
    """The pictures, newest first, in one place.

    The mirror copies every render and assembled gif into a flat folder; this lists
    it so "where is the image?" is answered by a tab that is always current, and
    `enter` opens the newest in the machine's own viewer. No hunting through the
    workspace tree, and nothing the agent had to remember to hand over."""

    paths: reactive[list] = reactive(list)
    folder: reactive[str] = reactive("")

    def render(self) -> str:
        if not self.paths:
            return "[b]renders[/b]\n[dim]none yet - they appear here as they are made[/dim]"
        out = [f"[b]renders[/b]  [dim]{len(self.paths)}, newest first[/dim]", ""]
        for path in self.paths[:16]:
            out.append(f" {_escape(path.name)}")
        out.append("\n[dim]ctrl+G opens the newest[/dim]")
        return "\n".join(out)


class StagePane(Static):
    """One line on what is happening right now, so the screen is never silent."""

    text = reactive("")

    def render(self) -> str:
        return f"[dim italic]{self.text}[/dim italic]" if self.text else ""


def _files_signature(entries: list[Entry]) -> tuple:
    """What the pane actually draws. mtimes are left out on purpose: a growing log
    changes its mtime every cycle, and redrawing the tree over a difference nobody
    can see costs the user their cursor position for nothing."""
    return tuple((entry.path, entry.size, entry.is_dir) for entry in entries)


class FilesTree(Tree):
    """The workspace as it actually is, not as it was described.

    Built from one listing rather than a call per directory: a case directory has
    hundreds of them, and a pane that takes a minute to fill is not a pane.
    """

    def __init__(self, root_path: str, **kwargs: Any) -> None:
        super().__init__(root_path, data=root_path, **kwargs)
        self.root_path = root_path
        self.show_root = True
        self.guide_depth = 2

    def load(self, entries: list[Entry]) -> None:
        """Replace the tree. Entries arrive parents-first, so one pass is enough."""
        self.reset(self.root_path, data=self.root_path)
        nodes = {self.root_path: self.root}
        for entry in entries:
            parent = nodes.get(entry.path.rpartition("/")[0])
            if parent is None:
                continue
            if entry.is_dir:
                nodes[entry.path] = parent.add(
                    Text(entry.name + "/", style="bold"), data=entry.path
                )
            else:
                label = Text(entry.name)
                label.append(f"  {human(entry.size)}", style="dim")
                parent.add_leaf(label, data=entry.path)
        self.root.expand()
        if not entries:
            self.root.add_leaf(Text("(nothing here yet)", style="dim"))


class FileScreen(ModalScreen):
    """One file, full screen, read-only."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("ctrl+p", "pull", "Copy out"),
    ]

    CSS = """
    FileScreen { align: center middle; }
    #sheet { width: 90%; height: 90%; border: round $primary; background: $surface; }
    #sheet-title { height: 1; padding: 0 1; background: $panel; }
    """

    def __init__(self, path: str, body: str):
        super().__init__()
        self.path = path
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="sheet"):
            yield Static(f"[b]{self.path}[/b]  [dim]esc to close[/dim]", id="sheet-title")
            yield TextArea(self.body, read_only=True, soft_wrap=False, show_line_numbers=True)

    def action_dismiss(self) -> None:
        self.app.pop_screen()

    def action_pull(self) -> None:
        self.app.pull_path(self.path)


class OpenReynoldsApp(App):
    """The session, as a product."""

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #left { width: 3fr; }
    #right { width: 42; }
    SessionBar { height: 2; padding: 0 1; background: $panel; }
    ProgressPane { height: 2; padding: 0 1; }
    NowPane { height: 1; padding: 0 1; }
    StagePane { height: 1; padding: 0 1; }
    JobsPane { padding: 0 1; height: 1fr; }
    #conversation { height: 3fr; border: round $primary; padding: 0 1; }
    #activity { height: 1fr; border: round $secondary; padding: 0 1; }
    Input { dock: bottom; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+t", "toggle_thinking", "Thinking"),
        ("ctrl+f", "files", "Files"),
        ("ctrl+g", "renders", "Renders"),
        ("ctrl+r", "refresh_files", "Refresh"),
    ]

    show_thinking = reactive(False)

    def __init__(self, run_session: Callable[[OpenReynoldsApp], None]):
        super().__init__()
        self._run_session = run_session
        self.typed: queue.Queue[str | None] = queue.Queue()
        self._streaming = False
        self.quitting = False
        self.browser: Any = None
        self._files_sig: tuple | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SessionBar(id="bar")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield ProgressPane(id="progress")
                yield NowPane(id="now")
                yield StagePane(id="stage")
                yield RichLog(id="conversation", wrap=True, markup=True, highlight=False)
                yield RichLog(id="activity", wrap=True, markup=True, highlight=False)
            with Vertical(id="right"):
                with TabbedContent(id="panes"):
                    with TabPane("jobs", id="tab-jobs"):
                        yield JobsPane(id="jobs")
                    with TabPane("renders", id="tab-renders"):
                        yield RendersPane(id="renders")
                    with TabPane("files", id="tab-files"):
                        yield FilesTree("/work", id="filestree")
        yield Input(placeholder="Ask for something, or /btw to speak without interrupting", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "OpenReynolds"
        self.sub_title = "CFD agent"
        self.query_one("#activity", RichLog).write(
            "[dim]/help for what you can type - /btw says something without asking it "
            "to stop, /status says what is happening[/dim]"
        )
        self.query_one("#prompt", Input).focus()
        self.start_session()

    @work(thread=True, exclusive=True, group="session")
    def start_session(self) -> None:
        """The agent loop is blocking, so it lives off the UI thread."""
        try:
            self._run_session(self)
        except Exception as exc:  # surfaced, never swallowed
            self._announce(f"[red]session ended: {type(exc).__name__}: {exc}[/red]")
        else:
            self._announce("[yellow]the session has ended[/yellow]")

    def _announce(self, markup: str) -> None:
        """Report the end from the worker thread, if there is still an app to report to.

        Quitting tears the app down while this thread is still unwinding, so the hop
        across can arrive too late. Saying nothing is fine there -- the user is already
        leaving -- but raising out of the worker is not, and neither is waiting on an
        event loop that is busy shutting down.
        """
        if self.quitting:
            return
        try:
            self.call_from_thread(self._session_ended, markup)
        except RuntimeError:
            pass

    def _session_ended(self, markup: str) -> None:
        """Say the session is over, and stop accepting things nobody will read.

        A live-looking input box on a dead session is the worst kind of silent
        failure: everything typed into it is accepted, echoed, and discarded.

        The screen may already be coming down around this -- a session ending and an
        app quitting are the same moment seen from two threads -- so a widget that is
        no longer there means the message has no one left to reach, not an error.
        """
        if self.quitting:
            return
        try:
            self.query_one("#conversation", RichLog).write(f"\n{markup}")
            box = self.query_one("#prompt", Input)
        except NoMatches:
            return
        box.disabled = True
        box.placeholder = "the session has ended - ctrl+C to close"

    # -- input ----------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        log = self.query_one("#conversation", RichLog)
        command = commands.parse(text)
        if command.kind in (commands.SAY, commands.ASIDE):
            log.write(f"\n[bold green]you[/bold green]  {_escape(text)}")
        else:
            # It is answered here and never reaches the model; echoing it as speech
            # would claim otherwise.
            log.write(f"[dim]{_escape(text)}[/dim]")
        self.typed.put(text)

    def action_clear(self) -> None:
        self.query_one("#conversation", RichLog).clear()

    def action_toggle_thinking(self) -> None:
        """Reasoning is long and grey and buries the answer under itself, so it is one
        summary line by default and full text only when asked for."""
        self.show_thinking = not self.show_thinking
        self.query_one("#activity", RichLog).write(
            f"[dim]thinking {'shown in full' if self.show_thinking else 'collapsed'}[/dim]"
        )

    def action_quit(self) -> None:
        """Leave, and mean it.

        The session runs on a worker thread that is usually blocked inside a network
        call, and a thread cannot be interrupted from outside. Releasing the reader
        lets it finish if it is between calls; the caller force-exits if it is not.
        Being unable to quit is worse than skipping a tidy shutdown.
        """
        self.quitting = True
        self.typed.put(None)
        self.exit()

    # -- files ----------------------------------------------------------------

    def action_files(self) -> None:
        self.show_files_tab()

    def action_renders(self) -> None:
        """Open the renders tab and the newest picture in the machine's viewer."""
        self.show_renders_tab()
        pics = self._render_paths()
        if pics:
            _open_path(pics[0])

    def _render_paths(self) -> list[Path]:
        store = getattr(self.browser, "store", None) if self.browser else None
        folder = getattr(store, "renders_dir", None)
        if folder is None or not Path(folder).is_dir():
            return []
        pics = [p for p in Path(folder).iterdir() if p.is_file()]
        return sorted(pics, key=lambda p: p.stat().st_mtime, reverse=True)

    def refresh_renders(self) -> None:
        """Redraw the renders list from the flat folder. Runs on the UI thread."""
        try:
            pane = self.query_one("#renders", RendersPane)
        except NoMatches:
            return
        pane.paths = self._render_paths()

    def show_renders_tab(self) -> None:
        self.query_one("#panes", TabbedContent).active = "tab-renders"
        self.refresh_renders()

    def action_refresh_files(self) -> None:
        """An explicit refresh means "go and look", not "show me the cache"."""
        self.load_files(
            self.query_one("#filestree", FilesTree).root_path, force_remote=True
        )

    def show_files_tab(self, path: str = "") -> None:
        tree = self.query_one("#filestree", FilesTree)
        if path:
            tree.root_path = path
        self.query_one("#panes", TabbedContent).active = "tab-files"
        self.load_files(tree.root_path)

    @work(thread=True, group="files")
    def load_files(self, path: str, force_remote: bool = False) -> None:
        """Fill the files pane, from the mirror's listing when it has one.

        The background mirror lists the workspace every cycle anyway, so most loads
        are answered instantly from that; the network round trip is only paid when
        nothing has looked yet, when the path is outside what the mirror watches,
        or when a refresh explicitly asks for the workspace itself.
        """
        if self.browser is None:
            return
        entries = None if force_remote else self.browser.cached(path)
        if entries is None:
            self.call_from_thread(self._set_stage, f"listing {path}")
            try:
                entries = self.browser.tree(path)
            except Exception as exc:
                self.call_from_thread(
                    self._note, f"[red]could not list {path}: {exc}[/red]"
                )
                return
        self._files_sig = _files_signature(entries)
        tree = self.query_one("#filestree", FilesTree)
        self.call_from_thread(tree.load, entries)
        self.call_from_thread(self._set_stage, "")

    def files_synced(self, report: Any) -> None:
        """A mirror cycle finished: say what arrived, and redraw the pane if the
        workspace actually changed. Runs on the UI thread."""
        pulled = getattr(report, "pulled", None)
        if pulled:
            self._note(f"[dim]mirrored {len(pulled)} file(s)[/dim]")
        if self.browser is None:
            return
        try:
            tree = self.query_one("#filestree", FilesTree)
        except NoMatches:
            return
        entries = self.browser.cached(tree.root_path)
        if entries is None:
            return
        sig = _files_signature(entries)
        if sig == self._files_sig:
            return
        self._files_sig = sig
        tree.load(entries)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """A leaf is a file. Directories expand themselves; opening one shows nothing."""
        path = event.node.data
        if path and not event.node.allow_expand:
            self.open_path(str(path))

    @work(thread=True, group="files")
    def open_path(self, path: str) -> None:
        """Read one file for viewing. The mirrored local copy answers when it is
        current -- opening a file should not cost a network round trip when the
        file is already on this machine. An image is handed over as a local path
        either way: the terminal cannot draw it here, but the file browser can."""
        if self.browser is None:
            return
        body = self._open_local(path)
        if body is None:
            self.call_from_thread(self._set_stage, f"opening {path}")
            try:
                if images.media_type(path):
                    written = self.browser.pull(path)
                    where = "\n".join(str(p) for p in written) or "(nothing was copied)"
                    body = f"{path}\n\nAn image. Copied to your machine, open it from:\n\n{where}"
                else:
                    body, _is_text = self.browser.read(path)
            except Exception as exc:
                body = f"{path}\n\ncould not be read: {exc}"
        self.call_from_thread(self.push_screen, FileScreen(path, body))
        self.call_from_thread(self._set_stage, "")

    def _open_local(self, path: str) -> str | None:
        """The file, from the mirror, or None when the instance has to answer.

        Current means the local size matches what the last listing saw. A file the
        listing says has moved on is read from the instance -- showing a stale copy
        without saying so would be the pane lying about the workspace."""
        local = self._local_copy(path)
        if local is None:
            return None
        if images.media_type(path):
            return (
                f"{path}\n\nAn image, already on your machine:\n\n{local}"
            )
        try:
            raw = local.read_bytes()
        except OSError:
            return None
        if b"\x00" in raw[:8_000]:
            return None  # binary: the remote path describes it rather than dumping it
        return raw.decode("utf-8", errors="replace")

    def _local_copy(self, path: str) -> Path | None:
        store = getattr(self.browser, "store", None)
        if store is None:
            return None
        target = local_for(store.fetch_dir(), path)
        try:
            size = target.stat().st_size
        except OSError:
            return None
        entries = self.browser.cached(path)
        if entries:
            entry = next((e for e in entries if e.path == path.rstrip("/")), None)
            if entry is not None and (entry.is_dir or entry.size != size):
                return None
        return target

    @work(thread=True, group="files")
    def pull_path(self, path: str) -> None:
        if self.browser is None:
            return
        try:
            written = self.browser.pull(path)
        except Exception as exc:
            self.call_from_thread(self._note, f"[red]could not copy {path}: {exc}[/red]")
            return
        for local in written:
            self.call_from_thread(self._note, f"[green]copied to[/green] {local}")

    def _note(self, markup: str) -> None:
        self.query_one("#activity", RichLog).write(markup)

    def _set_stage(self, text: str) -> None:
        self.query_one("#stage", StagePane).text = text


class TuiView(View):
    """Reports session events into the app, from the worker thread."""

    def __init__(self, app: OpenReynoldsApp):
        self.app = app
        self._pending = ""
        self._thought = ""
        self._thinking_since = 0.0

    # Every call arrives off the UI thread, so all of them hop across.
    def _to(self, widget_id: str, markup: str) -> None:
        def write() -> None:
            self.app.query_one(f"#{widget_id}", RichLog).write(markup)

        self.app.call_from_thread(write)

    def _set(self, widget_id: str, **fields: Any) -> None:
        def apply() -> None:
            widget = self.app.query_one(f"#{widget_id}")
            for key, value in fields.items():
                setattr(widget, key, value)

        self.app.call_from_thread(apply)

    def header(self, study_id: str, instance_id: str, model: str, mirror: Path) -> None:
        self._set("bar", study=study_id, instance=instance_id, model=model)
        self._to("activity", f"[dim]fetched files land in {mirror}[/dim]")

    def thinking_begin(self) -> None:
        self._thought = ""
        self._thinking_since = time.monotonic()
        self._set("stage", text="thinking...")

    def thinking_delta(self, text: str) -> None:
        """Reasoning goes to the one-line stage indicator, not the transcript.

        Streamed in full it is hundreds of grey lines that bury the answer underneath
        them. Ctrl+T puts it in the log for anyone who wants it.
        """
        self._thought += text
        while "\n" in self._thought:
            line, self._thought = self._thought.split("\n", 1)
            if line.strip() and self.app.show_thinking:
                self._to("conversation", f"[dim italic]{_escape(line)}[/dim italic]")
        latest = " ".join(self._thought.split())
        if latest:
            # With a clock on it: two minutes of reasoning and a stalled connection
            # look the same without one.
            elapsed = time.monotonic() - self._thinking_since
            self._set("stage", text=f"thinking {elapsed:.0f}s: {latest[-100:]}")

    def text_delta(self, text: str) -> None:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._to("conversation", _escape(line) or " ")

    def turn_end(self) -> None:
        if self._pending.strip():
            self._to("conversation", _escape(self._pending))
        self._pending = ""
        self._thought = ""
        self._set("stage", text="")

    def tool(self, name: str, summary: str) -> None:
        colour = TOOL_STYLE.get(name, "white")
        self._to("activity", f"[{colour}]{name:<11}[/{colour}] {_escape(summary)}")
        self._set("stage", text=f"{name}: {summary[:90]}")

    def tool_error(self, message: str) -> None:
        self._to("activity", f"[red]{'':<11} {_escape(message)}[/red]")

    def notice(self, message: str) -> None:
        self._to("conversation", f"\n[yellow]{_escape(message)}[/yellow]")

    def warn(self, message: str) -> None:
        self._to("activity", f"[yellow]{_escape(message)}[/yellow]")

    def info(self, message: str) -> None:
        self._to("activity", f"[dim]{_escape(message)}[/dim]")

    def usage(self, tokens: int, fraction: float) -> None:
        self._set("bar", tokens=tokens, fraction=fraction)

    def prompt(self) -> None:
        """The input box is always there, so nothing to announce."""

    def jobs(self, records: list[Any]) -> None:
        self._set("jobs", records=list(records))

    def stage(self, text: str) -> None:
        self._set("stage", text=text)

    def step(self, number: int, seconds: float, tool_calls: int) -> None:
        """A rule across the activity pane, so the loop has visible joints."""
        calls = f"{tool_calls} call{'' if tool_calls == 1 else 's'}"
        self._to("activity", f"[dim]{'-' * 8} step {number}  {calls}  {seconds:.0f}s[/dim]")

    def interjection(self, text: str) -> None:
        """The input box already showed what was typed; this confirms it was carried."""
        self._to("conversation", "[dim](sent - it reads this at its next step)[/dim]")

    def workspace(self, browser: Any) -> None:
        """Hand the interface a way to look at the workspace, and fill the pane once."""
        self.app.browser = browser
        self.app.call_from_thread(self.app.show_files_tab, browser.home)

    def show_files(self, path: str = "", depth: int = 0) -> None:
        """Depth is the flat listing's concern; the tree loads what it needs."""
        self.app.call_from_thread(self.app.show_files_tab, path)

    def status(self, lines: list[str]) -> None:
        body = "\n".join(f"[cyan]{_escape(line)}[/cyan]" for line in lines)
        self._to("conversation", f"{body}\n")

    def mirrored(self, report: Any) -> None:
        """Arrivals land in the activity pane and the files pane redraws itself.

        The mirror outlives moments the interface does not: a teardown race is
        answered with silence rather than a crash, because the files are already
        home either way and only the telling would be lost."""
        try:
            self.app.call_from_thread(self.app.files_synced, report)
        except RuntimeError:
            pass

    def watching(self, names: list[str]) -> None:
        self._set("jobs", names=list(names))
        self._set("stage", text=f"watching {len(names)} job(s) - type any time")
        self._to("activity", f"[dim]watching {len(names)} job(s)[/dim]")

    def progress(self, snapshot: Any) -> None:
        """Once a second from the tracker's thread; the pane compares and redraws."""
        try:
            self._set("progress", snapshot=snapshot)
        except RuntimeError:
            pass  # the app is coming down; the next tick has nowhere to go either

    def narration(self, text: str) -> None:
        """The front desk's 'what's happening now' line."""
        try:
            self._set("now", text=text)
        except RuntimeError:
            pass

    def desk(self, text: str) -> None:
        """A front-desk reply, in the conversation pane, clearly the desk and not
        the agent -- so nobody reads it as the agent having answered."""
        self._to("conversation", f"\n[bold cyan]desk[/bold cyan]  {_escape(text)}")

    def delivered(self, event: Any) -> None:
        """New renders arrived from the mirror: say so and refresh the renders tab.
        No auto-open -- a viewer window opening itself mid-solve is startling; the
        line says a picture is ready and the tab holds it."""
        for line in event.lines():
            self._to("activity", f"[green]{_escape(line)}[/green]")
        self.app.call_from_thread(self.app.refresh_renders)

    def show_renders(self, renders_dir: Any) -> None:
        self.app.call_from_thread(self.app.show_renders_tab)


class TuiReader:
    """Stands in for stdin: lines come from the input box instead."""

    def __init__(self, app: OpenReynoldsApp):
        self.app = app

    def get(self, timeout: float | None = None) -> str | None:
        try:
            return self.app.typed.get(timeout=timeout)
        except queue.Empty:
            return None

    def poll(self):
        from .watch import NOTHING

        try:
            return self.app.typed.get_nowait()
        except queue.Empty:
            return NOTHING

    def putback(self, line: str | None) -> None:
        self.app.typed.put(line)

    def pending(self) -> bool:
        """Whether something is waiting, without taking it."""
        return not self.app.typed.empty()


def _open_path(path: Path) -> None:
    """Open a file with the machine's own viewer. Best-effort: a picture that will
    not open is not worth an exception into the UI thread."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606 - the platform's own opener
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])
    except (OSError, AttributeError):
        pass


def _escape(text: str) -> str:
    """Model output is not markup; square brackets in it must not become tags."""
    return text.replace("[", r"\[")
