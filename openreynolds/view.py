"""Where session output goes.

The loop used to print straight to a `rich` Console, which meant the only possible
interface was a scrolling terminal. This is the seam: the loop reports what happened,
and something else decides how it looks. `ConsoleView` keeps the plain streaming
behaviour; the TUI supplies its own.

A view is presentation only. It never sees a decision and never makes one -- adding a
view can change what the user reads, never what the model does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rich.console import Console


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

    def interjection(self, text: str) -> None:
        """Something the user said that will reach the model without stopping it."""

    def workspace(self, browser: Any) -> None:
        """A read-only way to look at the workspace, for views that can show one."""

    def show_files(self, path: str = "") -> None:
        """Show what is in the workspace. Answered locally; the model is not told."""

    def status(self, lines: list[str]) -> None:
        """Answer 'what is going on' from what the harness knows. Costs no turn."""

    def prompt(self) -> None:
        """Signal that it is the user's turn. An always-present input box need not."""


MAX_LISTED = 300
"""A scrolling terminal cannot show more than this usefully; the pane can."""


class ConsoleView(View):
    """The plain streaming terminal."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._thinking = False
        self._browser: Any = None
        self._watching: list[str] = []

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

    def show_files(self, path: str = "") -> None:
        if self._browser is None:
            self.console.print("[yellow]no workspace to look at[/]")
            return
        target = path or self._browser.backend.workspace_root
        try:
            entries = self._browser.tree(target)
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

    def status(self, lines: list[str]) -> None:
        for index, line in enumerate(lines):
            self.console.print(f"[cyan]{line}[/]" if index == 0 else f"[dim]{line}[/]")

    def prompt(self) -> None:
        self.console.print("\n[bold green]>[/] ", end="")
