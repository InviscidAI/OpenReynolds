"""The terminal interface.

A session has three things worth seeing at once: what the agent is saying, what it is
doing to the workspace, and what is still running out on the instance. A scrolling log
shows the first and buries the other two, so they get their own panes.

The agent loop is synchronous and blocking, so it runs on a worker thread and reports
through `TuiView`, which is the same `View` the plain terminal implements. Nothing in
here can influence the model -- it is presentation, and the loop cannot tell which view
it has.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any, Callable

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

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


class OpenReynoldsApp(App):
    """The session, as a product."""

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #left { width: 3fr; }
    #right { width: 1fr; }
    SessionBar { height: 2; padding: 0 1; background: $panel; }
    StagePane { height: 1; padding: 0 1; }
    JobsPane { padding: 0 1; background: $panel; height: 1fr; }
    #conversation { height: 3fr; border: round $primary; padding: 0 1; }
    #activity { height: 1fr; border: round $secondary; padding: 0 1; }
    Input { dock: bottom; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+t", "toggle_thinking", "Thinking"),
    ]

    show_thinking = reactive(False)

    def __init__(self, run_session: Callable[[OpenReynoldsApp], None]):
        super().__init__()
        self._run_session = run_session
        self.typed: queue.Queue[str | None] = queue.Queue()
        self._streaming = False
        self.quitting = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SessionBar(id="bar")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield StagePane(id="stage")
                yield RichLog(id="conversation", wrap=True, markup=True, highlight=False)
                yield RichLog(id="activity", wrap=True, markup=True, highlight=False)
            with Vertical(id="right"):
                yield JobsPane(id="jobs")
        yield Input(placeholder="Ask for something, or say what looks wrong...", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "OpenReynolds"
        self.sub_title = "CFD agent"
        self.query_one("#prompt", Input).focus()
        self.start_session()

    @work(thread=True, exclusive=True)
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
        self.query_one("#conversation", RichLog).write(f"\n[bold green]you[/bold green]  {text}")
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
        self._to("conversation", f"[green]you[/green]  {_escape(text)} [dim](sent)[/dim]")

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


def _escape(text: str) -> str:
    """Model output is not markup; square brackets in it must not become tags."""
    return text.replace("[", r"\[")
