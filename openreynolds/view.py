"""Where session output goes.

The loop used to print straight to a `rich` Console, which meant the only possible
interface was a scrolling terminal. This is the seam: the loop reports what happened,
and something else decides how it looks. `ConsoleView` keeps the plain streaming
behaviour; the TUI supplies its own.

A view is presentation only. It never sees a decision and never makes one -- adding a
view can change what the user reads, never what the model does.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rich.console import Console

from . import images


@runtime_checkable
class View(Protocol):
    """Everything a session needs to show."""

    def header(self, study_id: str, instance_id: str, model: str, mirror: Path) -> None: ...

    def thinking_begin(self) -> None: ...

    def thinking_delta(self, text: str) -> None: ...

    def text_delta(self, text: str) -> None: ...

    def turn_end(self) -> None: ...

    def tool(self, name: str, summary: str) -> None: ...

    def tool_error(self, message: str) -> None: ...

    def notice(self, message: str) -> None:
        """Something the user should register -- a refusal, a truncated turn."""

    def warn(self, message: str) -> None: ...

    def info(self, message: str) -> None:
        """Quiet background detail."""

    def usage(self, tokens: int, fraction: float) -> None: ...

    def watching(self, names: list[str]) -> None: ...

    def jobs(self, records: list[Any]) -> None:
        """Current job state, whenever it changes -- not only while watching."""

    def stage(self, text: str) -> None:
        """What is happening right now, in a few words."""

    def step(self, number: int, seconds: float, tool_calls: int) -> None:
        """One round of think-then-act finished.

        Without a mark between them the activity pane is an undivided column of
        tool calls, and a turn that took three rounds looks like one that took
        thirty. Facts only: which round, how long, how many calls."""

    def interjection(self, text: str) -> None:
        """Something the user said that will reach the model without stopping it."""

    def workspace(self, browser: Any) -> None:
        """A read-only way to look at the workspace, for views that can show one."""

    def show_files(self, path: str = "", depth: int = 0) -> None:
        """Show what is in the workspace. Answered locally; the model is not told."""

    def show_renders(self, renders_dir: Any) -> None:
        """Show the flat pictures folder, newest first, on request (`/renders`)."""

    def status(self, lines: list[str]) -> None:
        """Answer 'what is going on' from what the harness knows. Costs no turn."""

    def mirrored(self, report: Any) -> None:
        """A background sync brought the study's files home.

        Facts about what arrived; a view may refresh whatever it shows of the
        workspace from it. Called from the mirror's own thread."""

    def prompt(self) -> None:
        """Signal that it is the user's turn. An always-present input box need not."""

    def progress(self, snapshot: Any) -> None:
        """The bar and its line: what is running, how far along, what the harness
        is doing meanwhile. Pushed about once a second from the tracker's own
        thread, whether or not a turn is in flight. Shown only when there is
        something to show -- a job, a slow tool, a sync -- hidden otherwise."""

    def narration(self, text: str) -> None:
        """One plain-language line on what the agent is doing right now, from the
        front desk. This is the 'what is happening' line, distinct from the model's
        own reasoning peek."""

    def desk(self, text: str) -> None:
        """The front desk answering the user while the main agent is mid-turn.
        Shown as speech, clearly attributed to the desk and not the agent."""

    def delivered(self, event: Any) -> None:
        """The mirror surfaced new renders (and maybe assembled a gif). Announce
        them and, where a view can, show or offer to open them. The agent is not
        involved -- this is the harness delivering what it already has."""


MAX_LISTED = 300

PROGRESS_REPEAT_S = 30.0
"""How often the plain terminal repeats an unchanged progress line. It has no pane to
hold one, so the line is printed; every second would be a scroll of nothing."""
"""A scrolling terminal cannot show more than this usefully; the pane can."""


class ConsoleView(View):
    """The plain streaming terminal."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._thinking = False
        self._browser: Any = None
        self._watching: list[str] = []
        self._progress_key: tuple = ()
        self._progress_at = 0.0
        self._narration = ""

    def header(self, study_id: str, instance_id: str, model: str, mirror: Path) -> None:
        self.console.print(
            f"[bold]study[/] {study_id}   [bold]instance[/] {instance_id}   "
            f"[bold]model[/] {model}"
        )
        self.console.print(f"[dim]fetched files land in {mirror}[/]\n")

    def thinking_begin(self) -> None:
        self._thinking = True
        self.console.print("\n[dim]thinking...[/]")

    def thinking_delta(self, text: str) -> None:
        self.console.print(f"[dim]{text}[/]", end="")

    def text_delta(self, text: str) -> None:
        if self._thinking:
            self._thinking = False
            self.console.print()
        self.console.print(text, end="", highlight=False)

    def turn_end(self) -> None:
        self._thinking = False
        self.console.print()

    def tool(self, name: str, summary: str) -> None:
        self.console.print(f"[cyan]{name}[/] {summary}")

    def tool_error(self, message: str) -> None:
        self.console.print(f"  [red]{message}[/]")

    def notice(self, message: str) -> None:
        self.console.print(f"\n[yellow]{message}[/]")

    def warn(self, message: str) -> None:
        self.console.print(f"[yellow]{message}[/]")

    def info(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/]")

    def usage(self, tokens: int, fraction: float) -> None:
        """The plain view has nowhere to keep this, so it stays quiet."""

    def stage(self, text: str) -> None:
        """No pane to hold one line, so it becomes another line.

        Worth the noise: a five-minute command that prints nothing after the line
        announcing it is indistinguishable from a hang, and the honest fix is to keep
        saying so rather than to hope nobody minds.
        """
        if text:
            self.console.print(f"[dim]  {text}[/]")

    def step(self, number: int, seconds: float, tool_calls: int) -> None:
        calls = f"{tool_calls} tool call{'' if tool_calls == 1 else 's'}"
        self.console.print(f"[dim]  -- step {number}: {calls}, {seconds:.0f}s --[/]")

    def watching(self, names: list[str]) -> None:
        """Said once per set of jobs. Watch mode is re-entered after every local
        command, and repeating the same line each time is how a screen fills up
        with nothing."""
        if names == self._watching:
            return
        self._watching = list(names)
        self.console.print(
            f"[dim]watching {len(names)} job(s): {', '.join(names)} - type any time[/]"
        )

    def interjection(self, text: str) -> None:
        self.console.print(f"[green]sent:[/] {text}")

    def workspace(self, browser: Any) -> None:
        self._browser = browser

    def show_files(self, path: str = "", depth: int = 0) -> None:
        if self._browser is None:
            self.console.print("[yellow]no workspace to look at[/]")
            return
        target = path or self._browser.home
        try:
            entries = self._browser.tree(target, depth) if depth else self._browser.tree(target)
        except Exception as exc:
            self.console.print(f"[red]could not list {target}:[/] {exc}")
            return
        if not entries:
            self.console.print(f"[dim]{target} is empty[/]")
            return
        for entry in entries[:MAX_LISTED]:
            style = "bold blue" if entry.is_dir else ""
            self.console.print(f"[{style}]{entry.line()}[/]" if style else entry.line())
        if len(entries) > MAX_LISTED:
            self.console.print(f"[dim]... {len(entries) - MAX_LISTED} more[/]")

    def show_renders(self, renders_dir: Any) -> None:
        renders_dir = Path(renders_dir)
        pics = sorted(
            (p for p in renders_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ) if renders_dir.is_dir() else []
        if not pics:
            self.console.print("[dim]no renders yet[/]")
            return
        self.console.print(f"[cyan]{len(pics)} render(s) in {renders_dir}[/]")
        for path in pics[:20]:
            self.console.print(f"[dim]  {path.name}[/]")
        if not images.show(pics[0]):
            self.console.print(f"[dim]newest: {pics[0]}[/]")

    def status(self, lines: list[str]) -> None:
        for index, line in enumerate(lines):
            self.console.print(f"[cyan]{line}[/]" if index == 0 else f"[dim]{line}[/]")

    def mirrored(self, report: Any) -> None:
        """Only arrivals rate a line here -- this fires every cycle for the whole
        session, and the full account (skips, warnings) still prints when the
        session closes down."""
        pulled = getattr(report, "pulled", None)
        if pulled:
            self.console.print(
                f"[dim]mirrored {len(pulled)} file(s) -> {report.local_dir}[/]"
            )

    def prompt(self) -> None:
        self.console.print("\n[bold green]>[/] ", end="")

    def progress(self, snapshot: Any) -> None:
        """A line when the picture changes, and now and then when it does not.

        The elapsed counters change every second, so "changed" means the phase or
        the fraction moved, and a same-looking line is repeated at most every
        `PROGRESS_REPEAT_S` -- enough to show a solve is alive without drowning the
        conversation. The model's own thinking and writing are already visible in
        this terminal, so those are left to the stream."""
        if snapshot.phase in ("thinking", "writing", "waiting"):
            return
        percent = None if snapshot.fraction is None else int(snapshot.fraction * 20)
        key = (snapshot.phase, percent)
        now = time.monotonic()
        if key == self._progress_key and now - self._progress_at < PROGRESS_REPEAT_S:
            return
        self._progress_key, self._progress_at = key, now
        line = f"{snapshot.percent()} {snapshot.headline}".strip()
        if snapshot.detail:
            line += f"\n       {snapshot.detail}"
        self.console.print(f"[dim]  {line}[/]", highlight=False)

    def narration(self, text: str) -> None:
        """The 'what's happening now' line. Printed when it changes, since the plain
        terminal has no pane to hold it steady."""
        text = text.strip()
        if text and text != self._narration:
            self._narration = text
            self.console.print(f"[dim italic]  - {text}[/]", highlight=False)

    def desk(self, text: str) -> None:
        self.console.print(f"\n[bold cyan]desk[/] [cyan]{text}[/]", highlight=False)

    def delivered(self, event: Any) -> None:
        """Say what arrived and, on a graphics terminal, draw it. Elsewhere the path
        is the delivery: it is one flat folder, which is the whole point."""
        for line in event.lines():
            self.console.print(f"[green]{line}[/]")
        for path in list(event.videos) + list(event.images):
            if not images.show(Path(path)):
                self.console.print(f"[dim]  {path}[/]")
