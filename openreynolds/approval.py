"""Putting one question to the person and waiting for their answer.

Only ever used because the person asked to be consulted (`modes.py`): in full auto
nothing here runs. The question goes to the view, and the answer comes back through the
same reader the session already reads typed lines from, so a terminal, the interface,
a JSON stream and the web page all answer it the same way.

Only what is typed after the question is shown answers it: the loop drains the reader
before every tool call (`Loop._gather`), so a line typed while the model was still
streaming reaches the model as an interjection instead of silently declining a call
nobody saw. Anything typed while a question is open is read as an answer to it, with two
exceptions: commands answered locally (`/status`, `/files`, `/help`, `/mode`...) are
handled and the question stays open, and `/exit` declines and is put back for the
prompt to act on. Words that are not a yes, a no or an "all" are steering, so they
decline the call and travel back to the model verbatim.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from . import commands

KINDS = ("job", "checkpoint")

APPROVED = "approved"
DECLINED = "declined"
APPROVED_ALL = "approved_all"

NOBODY = "nobody answered"
LEAVING = "the person is leaving the session"

_YES = {"y", "yes", "ok", "okay", "go", "approve", "approved"}
_NO = {"n", "no"}
_ALL = {"a", "all"}

_LOCAL_KINDS = {
    commands.STATUS,
    commands.FILES,
    commands.RENDERS,
    commands.OPEN,
    commands.HELP,
    commands.MODEL,
    commands.EFFORT,
    commands.MODE,
}


@dataclass
class Decision:
    approved: bool
    note: str = ""
    """What the person said alongside the answer, verbatim. Goes back to the model."""
    all: bool = False
    """Approve this and stop asking: the session switches to full auto."""

    @property
    def outcome(self) -> str:
        if not self.approved:
            return DECLINED
        return APPROVED_ALL if self.all else APPROVED


def choices_for(kind: str) -> list[str]:
    if kind == "checkpoint":
        return ["approve", "ask for changes", "approve all"]
    return ["approve", "decline", "approve all"]


class Approver:
    """Asks the person, reads their answer."""

    def __init__(self, view: Any, reader: Any, local: Callable[[commands.Command], None] | None = None):
        self.view = view
        self.reader = reader
        self.local = local
        """Answers a local command typed while a question is open (cli wires `_local`)."""
        self.open: str | None = None
        """The id of the question waiting for an answer, if any."""

    def ask(self, kind: str, title: str, detail: str, choices: list[str] | None = None) -> Decision:
        request_id = uuid.uuid4().hex[:8]
        self.open = request_id
        self.view.approval(request_id, kind, title, detail, list(choices or choices_for(kind)))
        try:
            decision = self._wait()
        finally:
            self.open = None
        self.view.approval_done(request_id, decision.outcome, decision.note)
        return decision

    def _wait(self) -> Decision:
        while True:
            line = self.reader.get()
            if line is None:
                # EOF belongs to whoever waits at the prompt; swallowing it here would
                # leave the session unendable.
                self.reader.putback(None)
                return Decision(False, NOBODY)
            decision = self._read(line)
            if decision is not None:
                return decision

    def _read(self, line: str) -> Decision | None:
        command = commands.parse(line)
        kind = command.kind
        if kind == commands.YES:
            # "/yes all" is the same answer as "/all".
            return Decision(True, all=command.text.strip().lower() == "all")
        if kind == commands.NO:
            return Decision(False, command.text)
        if kind == commands.ALL:
            return Decision(True, all=True)
        if kind == commands.EXIT:
            self.reader.putback(line)
            return Decision(False, LEAVING)
        if kind in _LOCAL_KINDS:
            if self.local is not None:
                self.local(command)
            return None
        text = command.text.strip()
        if kind == commands.ASIDE:
            # The aside wording is for the model's thread; the note carries what was typed.
            text = line.strip().partition(" ")[2].strip()
        if not text:
            return None
        if kind == commands.SAY:
            word = text.lower().rstrip(".!")
            if word in _YES:
                return Decision(True)
            if word in _NO:
                return Decision(False)
            if word in _ALL or word in ("yes all", "approve all"):
                return Decision(True, all=True)
        # Anything else is the person steering: the call does not run, and their words
        # are what the model gets back.
        return Decision(False, text)
