"""`import openreynolds`: the library face of the same capabilities the tools offer.

A program that wants a CFD study has three ways in, and they are one piece of work. The
MCP server (`openreynolds mcp`) is for a calling agent, `openreynolds serve` is for
anything that speaks HTTP, and this is for Python: the nine capabilities in
`openreynolds/mcp/core.py` -- start a study, watch it, say something to it, answer an
approval, end it, list studies, read results, fetch a file -- as objects and functions,
over the same `core.Host`. Nothing here re-derives a state name, a transcript row or a
refusal; drift between the faces is not possible because there is only one of each.

    import openreynolds

    study = openreynolds.start("drag on a sphere at Re=1000, SI units")
    for event in study.events():
        print(event["state"], event["stage"])
    print(study.results()["final_message"])

The default host runs each study in this process on a thread of its own, as
`openreynolds mcp` does, with the configuration and the workspace the terminal would
use and the same `studies/` directory. A study holds a workspace that costs money, so
`shutdown()` ends every live one -- jobs stopped, workspace stopped, files kept -- and
it is registered to run at interpreter exit; `start(study_id=...)` carries one on.
"""

from __future__ import annotations

import atexit
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from .mcp import core

DONE = ("waiting_for_input", "ended", "error")
"""The states at which the study has handed the turn back: `Study.events` stops there.
`core.STATES` has them all."""

_lock = threading.Lock()
_host: Any = None


def _default_host() -> Any:
    """This machine, made once: `mcp.local.LocalHost`, lazily, because importing it
    pulls in the session and everything a session needs."""
    global _host
    with _lock:
        if _host is None:
            from . import cli
            from .mcp.local import LocalHost

            # A library caller's stdout belongs to the library caller: no line of a
            # session's terminal reporting in the middle of their own output. This is
            # the seam `mcp/local.py` used to reach into `cli.console` for.
            cli.quiet_stdout()
            _host = LocalHost()
            atexit.register(shutdown)
        return _host


def _host_for(host: Any) -> Any:
    return _default_host() if host is None else host


class Study:
    """One study and the session running it. `start()` returns one.

    `status` and `events` read it, `send` and `answer` speak to it, `end` closes its
    session, and `results` and `file` read what it produced. Every answer is the object
    the MCP tool of the same name returns.
    """

    def __init__(self, host: Any, session_id: str, study_id: str | None = None,
                 *, after_seq: int = 0) -> None:
        self.host = host
        self.session_id = session_id
        self.study_id = study_id
        self.after_seq = after_seq
        """The transcript cursor: `status` reads from here and moves it on, so each
        call answers with what is new and nothing twice."""
        self.state = "starting"
        self._deliveries = core.deliveries_for(host)

    def __repr__(self) -> str:
        return f"<Study {self.study_id or '(unnamed)'} {self.state}>"

    def status(self, *, wait_s: float = 0.0) -> dict[str, Any]:
        """What the study is doing now, and what has been said since the last read.

        `wait_s` holds the call open (up to `core.MAX_WAIT_S`) and returns early the
        moment the study wants something or ends. Each call also tells the session
        somebody is still there; a study nobody reads for half an hour ends itself.
        """
        self.host.check(self.session_id)
        self.host.heartbeat(self.session_id)
        row, picture, state = core.wait_for_change(self.host, self._deliveries,
                                                   self.session_id, wait_s)
        answer = core.status_answer(self.host, self.session_id, row, picture, state,
                                    self.after_seq)
        self.after_seq = int(answer["next_after_seq"])
        self.study_id = answer["study_id"] or self.study_id
        self.state = state
        return answer

    def events(self, *, wait_s: float = core.MAX_WAIT_S) -> Iterator[dict[str, Any]]:
        """Yield what is new as it happens, until the study hands the turn back.

        Each item is a `status` answer carrying only the transcript rows you have not
        seen, so the whole conversation arrives once across the iteration. It stops
        after the item whose state is in `DONE` -- the study is waiting for you, over,
        or broken -- and iterating again carries on from there.
        """
        while True:
            answer = self.status(wait_s=wait_s)
            yield answer
            if answer["state"] in DONE:
                return
            if answer["state"] not in ("starting", "working"):
                # `wait_for_change` holds the call open only while the study is working;
                # a session closing down answers at once, and an iteration must wait a
                # beat for it rather than spin through its whole close-down.
                time.sleep(core.POLL_S)

    def wait(self, *, wait_s: float = core.MAX_WAIT_S) -> dict[str, Any]:
        """Run `events` to its end and answer with the last of them: the common case,
        where a script wants the study's answer and not its narration."""
        answer = {}
        for answer in self.events(wait_s=wait_s):
            pass
        return answer or self.status()

    def send(self, text: str) -> None:
        """Say something to the running study: an answer, a requirement, a change of
        course, or a slash command (`/status`, `/btw <question>`, `/mode ...`). It is
        read at the session's next step."""
        self.host.say(self.session_id, core.said_text(text))

    def answer(self, decision: str, note: str = "") -> str:
        """Answer the open approval: `approve`, `decline` (the note says why) or
        `approve_all`. Returns the line that was sent."""
        line = core.approval_line(decision, note)
        self.host.say(self.session_id, line)
        return line

    def end(self) -> str:
        """End the session: jobs stopped, workspace stopped, every file kept. The study
        itself carries on from `start(study_id=...)`."""
        self.state = self.host.end(self.session_id)
        return self.state

    def results(self) -> dict[str, Any]:
        """What the study came to. Works while it is still running."""
        return results(self._named(), host=self.host)

    def file(self, path: str) -> Path:
        """One of the study's files, here, by its workspace path."""
        return fetch(self._named(), path, host=self.host)

    def _named(self) -> str:
        if not self.study_id:
            raise core.McpError("this study has no id yet; read status until one appears")
        return self.study_id


def start(task: str, *, upload_ids: list[str] | None = None, study_id: str | None = None,
          depth: str = "quick", mode: str = "auto", model: str | None = None,
          effort: str | None = None, notes: str | None = None, solver: str = "auto",
          host: Any = None) -> Study:
    """Start a study, or continue one by `study_id`, and return at once with a `Study`.

    `task` says what to simulate and what you want to know -- geometry, units, flow
    conditions, quantities of interest. `depth` is `quick` or `full`, `mode` is `auto`,
    `partial` or `structured`, `solver` is `auto`, `openfoam` or `felix`. `upload_ids`
    are files staged with `upload(...)`; they are placed in the study's `uploads/` and
    named to the agent before it starts. `notes` is a standing instruction of your own.
    """
    host = _host_for(host)
    started = core.begin_study(host, core.deliveries_for(host), task, upload_ids=upload_ids,
                               study_id=study_id, depth=depth, mode=mode, model=model,
                               effort=effort, notes=notes, solver=solver)
    return Study(host, started["session_id"], started["study_id"])


def upload(path: str, *, host: Any = None) -> dict[str, Any]:
    """Stage one input file -- a geometry, a drawing, a data file -- by its path here.
    Pass the `upload_id` from the answer to `start`."""
    return _host_for(host).stage_upload(path)


def studies(*, limit: int = 20, host: Any = None) -> dict[str, Any]:
    """The studies on this machine, newest first, and the sessions still live."""
    return core.studies_list(_host_for(host), limit)


def results(study_id: str, *, host: Any = None) -> dict[str, Any]:
    """A study's final message, its `results.json`, its reports, its pictures and its
    file list. A path from `files` is what `fetch` takes."""
    return core.study_results(_host_for(host), study_id)


def fetch(study_id: str, path: str, *, host: Any = None) -> Path:
    """One of a study's files, here: copied out of a live workspace first. A path
    outside the study's workspace is refused."""
    return Path(_host_for(host).download(study_id, path)["path"])


def shutdown(*, wait_s: float | None = None, host: Any = None) -> None:
    """End every live study and wait for them to finish ending, as `end` does.

    Registered to run at interpreter exit, because a study left running holds a
    workspace that is being paid for. Call it earlier, or with your own `wait_s`, if
    you would rather not wait at exit.
    """
    global _host
    with _lock:
        target = host if host is not None else _host
        if host is None:
            _host = None
    if target is None:
        return
    target.close(**({} if wait_s is None else {"wait_s": wait_s}))
