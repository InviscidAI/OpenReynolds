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

    def prompt(self) -> None:
        """Signal that it is the user's turn. An always-present input box need not."""


class ConsoleView(View):
    """The plain streaming terminal."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._thinking = False

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

    def watching(self, names: list[str]) -> None:
        self.console.print(
            f"[dim]watching {len(names)} job(s): {', '.join(names)} - type to interrupt[/]"
        )

    def prompt(self) -> None:
        self.console.print("\n[bold green]>[/] ", end="")
