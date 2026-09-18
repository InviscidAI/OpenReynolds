"""What the user can type that is not a message.

These are the user's own words about how they want to be heard, not the harness's
opinion about how the model should work. `/btw` marks a message as an aside because
the user chose to mark it; `/status` is answered here and never reaches the model at
all, which is the only way to ask "what is going on" without becoming a turn.

Nothing here inspects, rewrites or withholds an ordinary message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SAY = "say"
ASIDE = "aside"
MESH = "mesh"
STATUS = "status"
FILES = "files"
RENDERS = "renders"
OPEN = "open"
HELP = "help"
EXIT = "exit"

HELP_TEXT = """\
  /mesh <request>    build a geometry and its mesh, said straight to the CAD desk
                     name a file with @: /mesh prepare @/work/uploads/part.step
  /btw <something>   say it without asking the agent to stop what it is doing
  /btw               what is happening right now, answered here - the agent is not told
  /status            the same thing
  /files [path]      look at the workspace
  /renders           open the pictures folder and show the newest
  /open              open this study's folder in the file browser
  /help              this
  /exit              leave (jobs keep running on the instance)"""


@dataclass(frozen=True)
class Command:
    kind: str
    text: str = ""
    """For `say` and `aside`, what goes to the model. For `files`, the path asked for."""
    inputs: tuple[str, ...] = ()
    """The files the line handed over with `@`, in the order they were typed.

    A path the user marked, never one inferred from the prose: `@/work/plan.png` is a
    handover and `/work/plan.png` is a sentence mentioning a path. The difference is the
    whole reason for the sigil -- "it is like the duct in /work/old/duct.step" names a
    file nobody is asking to have opened, and no amount of care in a heuristic tells
    that apart from a request to open it."""


_VERBS = {
    "/mesh": MESH,
    "/geometry": MESH,
    "/btw": ASIDE,
    "/bytheway": ASIDE,
    "/aside": ASIDE,
    "/status": STATUS,
    "/what": STATUS,
    "/files": FILES,
    "/ls": FILES,
    "/renders": RENDERS,
    "/pics": RENDERS,
    "/images": RENDERS,
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
        spoken, handed = handovers(text)
        return Command(SAY, spoken, handed)

    verb, _, rest = text.partition(" ")
    kind = _VERBS.get(verb.lower())
    if kind is None:
        # A path, a formula, a sentence that happens to start with a slash: the user
        # meant to say it. Guessing "unknown command" at them would be worse.
        spoken, handed = handovers(text)
        return Command(SAY, spoken, handed)

    rest = rest.strip()
    if kind is ASIDE and not rest:
        # "/btw" on its own is someone asking what is going on, not an empty aside.
        return Command(STATUS)
    if kind in (ASIDE, SAY, MESH):
        spoken, handed = handovers(rest)
        if kind is ASIDE:
            return Command(ASIDE, aside(spoken), handed)
        return Command(kind, spoken, handed)
    # `/files`, `/open` and the rest take a path as their whole argument, not a
    # sentence with a file named inside it, so the sigil means nothing there.
    return Command(kind, rest)


_HANDOVER = re.compile(r"""(?:(?<=\s)|^)@(?:"([^"]+)"|'([^']+)'|(\S+))""")
r"""A file handed over. The `@` has to start a token, so an address
like `name@example.com` and a decorator pasted into a sentence are left alone.

Quoted forms exist because a path with a space in it is not hypothetical here: the
tests already carry `/work/study/uploads/chassis v2.step`."""

_TRAILING = ".,;:!?)]}\'\""
"""Punctuation that ends a sentence rather than a filename. `@/work/plan.png,` is a
path followed by a comma every time, and a file whose name really ends in a comma is
not worth the ambiguity -- it can be quoted."""


def handovers(text: str) -> tuple[str, tuple[str, ...]]:
    """Split a typed line into what was said and what was handed over.

    The `@` is terminal syntax, so it is taken back out: the desk is given an ordinary
    sentence naming an ordinary path, and never has to know how the person typed it.

    Order is the order they were typed, and a file named twice is handed over once --
    somebody writing "compare @a.step against @a.step" means one file, and staging it
    twice would put the same path into the record twice for no reason.
    """
    handed: list[str] = []

    def take(match: re.Match) -> str:
        quoted = match.group(1) or match.group(2)
        path = quoted if quoted else match.group(3).rstrip(_TRAILING)
        if not path:
            return match.group(0)
        if path not in handed:
            handed.append(path)
        # What is left behind is the path as prose, with whatever punctuation the
        # stripping took off put back, so the sentence still reads as written.
        return path + (match.group(3)[len(path):] if not quoted else "")

    spoken = _HANDOVER.sub(take, text).strip()
    return spoken, tuple(handed)


def aside(text: str) -> str:
    """The user's framing, in the user's voice.

    They typed `/btw`; this is what that meant. It is a statement of what they want,
    not an instruction about how to work -- what to do about it stays the model's call.
    """
    return f"By the way, no need to stop what you are doing: {text}"


def status_lines(
    store: Any,
    stage: str = "",
    tokens: int = 0,
    local_files: int = 0,
    sync_age: float | None = None,
    token_totals: dict | None = None,
) -> list[str]:
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
    totals = token_totals or {}
    billed = sum(totals.values())
    if billed:
        # Cache reads at a tenth of input, cache writes at 1.25x: the split is the
        # difference between a study that costs $1 and the same study costing $7.50.
        read = totals.get("cache_read", 0)
        lines.append(
            f"sent so far: {billed:,} tokens - {read:,} read from cache "
            f"({read / billed:.0%}), {totals.get('cache_write', 0):,} written to it, "
            f"{totals.get('output', 0):,} out"
        )

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

    pulled = f"{local_files} file(s) pulled to {store.dir}"
    if sync_age is not None:
        # "Are my files here?" deserves a when, not just a count.
        pulled += f" (workspace synced {int(sync_age)}s ago)"
    lines.append(pulled)
    return lines
