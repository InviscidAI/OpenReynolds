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

import queue
from pathlib import Path
from typing import Any, Callable

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
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


class StagePane(Static):
    """One line on what is happening right now, so the screen is never silent."""

    text = reactive("")

    def render(self) -> str:
        return f"[dim italic]{self.text}[/dim italic]" if self.text else ""


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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SessionBar(id="bar")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield StagePane(id="stage")
                yield RichLog(id="conversation", wrap=True, markup=True, highlight=False)
                yield RichLog(id="activity", wrap=True, markup=True, highlight=False)
            with Vertical(id="right"):
                with TabbedContent(id="panes"):
                    with TabPane("jobs", id="tab-jobs"):
                        yield JobsPane(id="jobs")
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
            self.call_from_thread(
                self.query_one("#conversation", RichLog).write,
                f"[red]session ended: {type(exc).__name__}: {exc}[/red]",
            )

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

    def action_refresh_files(self) -> None:
        self.load_files(self.query_one("#filestree", FilesTree).root_path)

    def show_files_tab(self, path: str = "") -> None:
        tree = self.query_one("#filestree", FilesTree)
        if path:
            tree.root_path = path
        self.query_one("#panes", TabbedContent).active = "tab-files"
        self.load_files(tree.root_path)

    @work(thread=True, group="files")
    def load_files(self, path: str) -> None:
        """Listing the workspace is a network call, so it never runs on the UI thread."""
        if self.browser is None:
            return
        self.call_from_thread(self._set_stage, f"listing {path}")
        try:
            entries = self.browser.tree(path)
        except Exception as exc:
            self.call_from_thread(self._note, f"[red]could not list {path}: {exc}[/red]")
            return
        tree = self.query_one("#filestree", FilesTree)
        self.call_from_thread(tree.load, entries)
        self.call_from_thread(self._set_stage, "")

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """A leaf is a file. Directories expand themselves; opening one shows nothing."""
        path = event.node.data
        if path and not event.node.allow_expand:
            self.open_path(str(path))

    @work(thread=True, group="files")
    def open_path(self, path: str) -> None:
        """Read one file for viewing. An image is copied out instead: the terminal
        cannot draw it here, but the file browser can."""
        if self.browser is None:
            return
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
            self._set("stage", text=f"thinking: {latest[-110:]}")

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

    def interjection(self, text: str) -> None:
        """The input box already showed what was typed; this confirms it was carried."""
        self._to("conversation", "[dim](sent - it reads this at its next step)[/dim]")

    def workspace(self, browser: Any) -> None:
        """Hand the interface a way to look at the workspace, and fill the pane once."""
        self.app.browser = browser
        self.app.call_from_thread(self.app.load_files, "/work")

    def show_files(self, path: str = "") -> None:
        self.app.call_from_thread(self.app.show_files_tab, path)

    def status(self, lines: list[str]) -> None:
        body = "\n".join(f"[cyan]{_escape(line)}[/cyan]" for line in lines)
        self._to("conversation", f"{body}\n")

    def watching(self, names: list[str]) -> None:
        self._set("jobs", names=list(names))
        self._set("stage", text=f"watching {len(names)} job(s) - type any time")
        self._to("activity", f"[dim]watching {len(names)} job(s)[/dim]")


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


def _escape(text: str) -> str:
    """Model output is not markup; square brackets in it must not become tags."""
    return text.replace("[", r"\[")
