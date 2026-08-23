"""What the user can type that is not a message.

These are the user's own words about how they want to be heard, not the harness's
opinion about how the model should work. `/btw` marks a message as an aside because
the user chose to mark it; `/status` is answered here and never reaches the model at
all, which is the only way to ask "what is going on" without becoming a turn.

Nothing here inspects, rewrites or withholds an ordinary message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SAY = "say"
ASIDE = "aside"
STATUS = "status"
FILES = "files"
OPEN = "open"
HELP = "help"
EXIT = "exit"

HELP_TEXT = """\
  /btw <something>   say it without asking the agent to stop what it is doing
  /btw               what is happening right now, answered here - the agent is not told
  /status            the same thing
  /files [path]      look at the workspace
  /open              open this study's folder in the file browser
  /help              this
  /exit              leave (jobs keep running on the instance)"""


@dataclass(frozen=True)
class Command:
    kind: str
    text: str = ""
    """For `say` and `aside`, what goes to the model. For `files`, the path asked for."""


_VERBS = {
    "/btw": ASIDE,
    "/bytheway": ASIDE,
    "/aside": ASIDE,
    "/status": STATUS,
    "/what": STATUS,
    "/files": FILES,
    "/ls": FILES,
    "/open": OPEN,
    "/help": HELP,
    "/?": HELP,
    "/exit": EXIT,
    "/quit": EXIT,
}


def parse(line: str) -> Command:
    """Classify one typed line. Anything unrecognised is a message, not an error."""
    text = line.strip()
    if not text.startswith("/"):
        return Command(SAY, text)

    verb, _, rest = text.partition(" ")
    kind = _VERBS.get(verb.lower())
    if kind is None:
        # A path, a formula, a sentence that happens to start with a slash: the user
        # meant to say it. Guessing "unknown command" at them would be worse.
        return Command(SAY, text)

    rest = rest.strip()
    if kind is ASIDE and not rest:
        # "/btw" on its own is someone asking what is going on, not an empty aside.
        return Command(STATUS)
    if kind is ASIDE:
        return Command(ASIDE, aside(rest))
    return Command(kind, rest)


def aside(text: str) -> str:
    """The user's framing, in the user's voice.

    They typed `/btw`; this is what that meant. It is a statement of what they want,
    not an instruction about how to work -- what to do about it stays the model's call.
    """
    return f"By the way, no need to stop what you are doing: {text}"


def status_lines(store: Any, stage: str = "", tokens: int = 0, local_files: int = 0) -> list[str]:
    """A picture of the session assembled from what the harness already knows.

    No model call: asking what is happening should not cost a turn, and a question
    that costs a turn is a question people stop asking.
    """
    session = store.session
    lines = [f"study {session.study_id} on instance {session.instance_id[:8] or '?'}"]
    if stage:
        lines.append(f"right now: {stage}")
    if tokens:
        lines.append(f"thread: {tokens:,} tokens")

    jobs = list(session.jobs.values())
    live = [job for job in jobs if job.status == "running"]
    if live:
        lines.append(f"{len(live)} job(s) running:")
        for job in live:
            lines.append(f"  {job.name or job.job_id[:8]}  {job.cmd[:70]}")
    elif jobs:
        last = jobs[-1]
        ended = last.end_reason or last.status
        lines.append(f"no jobs running (last: {last.name or last.job_id[:8]} {ended})")
    else:
        lines.append("no jobs started yet")

    lines.append(f"{local_files} file(s) pulled to {store.dir}")
    return lines
