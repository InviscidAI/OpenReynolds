"""What the user can type that is not a message.

These are the user's own words about how they want to be heard, not the harness's
opinion about how the model should work. `/btw` marks a message as an aside because
the user chose to mark it; `/status` is answered here and never reaches the model at
all, which is the only way to ask "what is going on" without becoming a turn. `/mode`,
`/model` and `/effort` are the same kind of thing: the person saying how they want the
session run, answered by the harness.

Nothing here inspects, rewrites or withholds an ordinary message.

`/mesh` is the one verb that is not answered here and is not an ordinary message
either: it is said straight to the CAD desk, past the main agent.

`COMMANDS` is the one list of verbs. The parser, `/help`, the terminal's Tab completion
and the web composer's suggestion list (`as_json`) are all read off it, so a verb cannot
exist in one place and be missing from another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .llm.presets import EFFORTS

SAY = "say"
ASIDE = "aside"
MESH = "mesh"
STATUS = "status"
FILES = "files"
RENDERS = "renders"
OPEN = "open"
HELP = "help"
EXIT = "exit"
MODE = "mode"
MODEL = "model"
EFFORT = "effort"
YES = "yes"
NO = "no"
ALL = "all"

SURFACES = ("both", "terminal", "web")

PLAIN = "plain"
"""The plain streaming terminal (`--plain`, `ConsoleView`): it reads whole lines, so it
has the terminal's commands but none of the interface's completion or keys. Not a
`Spec.where`: a command typed there is a terminal command."""


@dataclass(frozen=True)
class Spec:
    """One verb: how it is typed, what it does, and where it can be typed."""

    verb: str
    aliases: tuple[str, ...]
    kind: str
    args: str
    summary: str
    detail: str
    where: str = "both"
    """`both`, `terminal` or `web`: `/open` opens a folder on this machine, which a
    web page has no way to do."""
    choices: tuple[str, ...] = ()
    """The fixed words the argument takes, when it takes fixed words."""


COMMANDS: tuple[Spec, ...] = (
    Spec("/mesh", ("/geometry",), MESH, "<request>",
         "build a geometry and its mesh, said straight to the CAD desk",
         "Your words go to the CAD desk as the request, past the main agent, which "
         "hears about it when the desk reports. Name a file with @ to hand it over: "
         "/mesh prepare @/work/uploads/part.step."),
    Spec("/btw", ("/bytheway", "/aside"), ASIDE, "<something>",
         "say it without asking the agent to stop what it is doing",
         "Your words reach the agent at its next step, marked as an aside, so it can "
         "take them into account without abandoning the command it is running. On its "
         "own, /btw is the same as /status."),
    Spec("/status", ("/what",), STATUS, "",
         "what is happening right now, answered here; the agent is not told",
         "Jobs running, the thread size, tokens sent and how much came from the cache, "
         "and when the workspace last synced. Built from what the harness already "
         "knows, so it costs no model call."),
    Spec("/files", ("/ls",), FILES, "[path]",
         "look at the workspace",
         "Lists the study's directory on the workspace, or the path you give. Answered "
         "here; the agent is not told."),
    Spec("/renders", ("/pics", "/images"), RENDERS, "",
         "show the pictures folder, newest first",
         "Every render the study has produced is copied into one flat folder. This "
         "lists it and shows the newest picture where the screen can."),
    Spec("/open", (), OPEN, "",
         "open this study's folder in the file browser",
         "Opens the study's local folder (the mirror of the workspace) in your "
         "platform's file browser.",
         where="terminal"),
    Spec("/mode", (), MODE, "[name]",
         "how much the agent does before asking you: auto, partial or structured",
         "With no name, says which mode the session is in. With one, switches from the "
         "next tool call: auto runs everything, partial asks before each job_start, "
         "structured works in stages you approve. See /help modes."),
    Spec("/model", (), MODEL, "[model]",
         "which model the agent runs on, or switch it",
         "With nothing after it, shows the provider, model, effort and mode, and the "
         "models this provider is known to have. /model <model> switches model on the "
         "same provider; /model <provider>:<model> or /model <provider> switches "
         "provider, if a key for it is available. See /help model."),
    Spec("/effort", (), EFFORT, "[level]",
         "how hard the model thinks: " + ", ".join(EFFORTS),
         "Read on every request, so a change applies from the next one. Higher effort "
         "spends more tokens thinking.",
         choices=EFFORTS),
    Spec("/yes", ("/approve", "/y"), YES, "",
         "approve what the agent is asking to run",
         "Answers an open question: in partial mode a job_start call, in "
         "structured mode a checkpoint. Typing y, yes or ok on its own does the same."),
    Spec("/no", ("/deny", "/n"), NO, "[reason]",
         "decline it; anything after /no goes back to the agent as your reason",
         "The call does not run and the agent is told you declined it, with your reason "
         "in your words. While a question is open, anything you type that is not a yes "
         "(y, yes, ok, go), a no (n, no), an all (a, all) or a command answered here "
         "(/status, /files, /renders, /open, /help, /mode, /model, /effort) also "
         "declines, and what you typed is the reason. /exit declines and leaves."),
    Spec("/all", (), ALL, "",
         "approve this and everything after it (switches the session to auto)",
         "Approves the open question and switches the session to full auto for the rest "
         "of the session, so nothing more is held. /yes all does the same."),
    Spec("/help", ("/?",), HELP, "[topic]",
         "this list, or a topic: commands, modes, model, tools, keys",
         "/help on its own lists every command. /help <topic> goes deeper."),
    Spec("/exit", ("/quit",), EXIT, "",
         "leave (jobs keep running on the instance)",
         "Ends this session. Jobs already started keep running on the workspace, and "
         "openreynolds --study <id> picks the study up again."),
)

_VERBS: dict[str, str] = {}
for _spec in COMMANDS:
    for _name in (_spec.verb, *_spec.aliases):
        _VERBS[_name] = _spec.kind

TOPICS: tuple[tuple[str, str], ...] = (
    ("commands", "every command, in more detail"),
    ("modes", "full auto, ask before compute and structured"),
    ("model", "changing the model or effort mid-study"),
    ("tools", "what each of the agent's tools does"),
    ("keys", "keys, and how completion works"),
)


@dataclass(frozen=True)
class Command:
    kind: str
    text: str = ""
    """For `say` and `aside`, what goes to the model. For `files`, the path asked for;
    for `mode`, `model`, `effort` and `help`, the argument; for `no`, the reason."""
    inputs: tuple[str, ...] = ()
    """The files the line handed over with `@`, in the order they were typed.

    A path the user marked, never one inferred from the prose: `@/work/plan.png` is a
    handover and `/work/plan.png` is a sentence mentioning a path. The difference is the
    whole reason for the sigil -- "it is like the duct in /work/old/duct.step" names a
    file nobody is asking to have opened, and no amount of care in a heuristic tells
    that apart from a request to open it."""


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
    if kind is YES and rest.lower() == "all":
        return Command(ALL)
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


# -- help ----------------------------------------------------------------------


def visible(where: str = "terminal") -> list[Spec]:
    """The commands that can be typed on this surface."""
    if where == PLAIN:
        where = "terminal"
    return [spec for spec in COMMANDS if spec.where in ("both", where)]


def _usage(spec: Spec) -> str:
    return f"{spec.verb} {spec.args}".strip()


def _modes() -> list[tuple[str, str, str]]:
    """(value, label, description) for each mode, read from `modes.py`."""
    from . import modes

    return [(name, modes.LABELS[name], modes.DESCRIPTIONS[name]) for name in modes.MODES]


def _overview(where: str) -> list[str]:
    specs = visible(where)
    width = max(len(_usage(spec)) for spec in specs) + 3
    lines = ["commands:"]
    lines += [f"  {_usage(spec).ljust(width)}{spec.summary}" for spec in specs]
    lines.append("")
    lines.append("modes (/mode <name> to switch):")
    lines += [f"  {value.ljust(12)}{label}: {description}" for value, label, description in _modes()]
    lines.append("")
    if where == "web":
        lines.append("type / to see matching commands above the message box; arrow keys "
                     "move, Tab or Enter picks one, Esc closes the list")
    elif where == PLAIN:
        lines.append("this plain terminal reads whole lines and has no completion; "
                     "without --plain, the interface completes commands as you type")
    else:
        lines.append("type / to see matching commands; Tab completes, Up and Down pick, "
                     "Esc closes the list")
    lines.append("more: " + ", ".join(f"/help {name}" for name, _ in TOPICS))
    return lines


def _commands_topic(where: str) -> list[str]:
    lines = []
    for spec in visible(where):
        also = f"  (also {', '.join(spec.aliases)})" if spec.aliases else ""
        lines.append(f"{_usage(spec)}{also}")
        lines.append(f"  {spec.detail}")
    return lines


def _modes_topic() -> list[str]:
    from . import modes

    lines = []
    for value, label, description in _modes():
        lines.append(f"{value}: {label}")
        lines.append(f"  {description}")
    stages = getattr(modes, "STAGES", None)
    lines += [
        "",
        "auto holds nothing: the agent runs every tool as it judges best, which is how "
        "OpenReynolds has always worked.",
        "partial puts each job_start call to you before it runs, since that is what "
        "spends compute. bash, read_file, write_file, fetch, job_check, job_kill and "
        "cad run freely. Answer /yes, /no <reason> or /all (approve and switch to "
        "auto for the rest of the session).",
        "structured gives the agent a checkpoint tool. Each checkpoint shows you a "
        "summary and what comes next, and waits for your answer: /yes carries on, "
        "anything else goes back as the changes you want. job_start is held until you "
        "have approved a first checkpoint, the plan.",
    ]
    if stages:
        lines.append("stages: " + ", ".join(str(stage) for stage in stages))
    lines += [
        "",
        "switch any time with /mode <name>; it applies from the next tool call.",
        "start in a mode with --mode <name> or OPENREYNOLDS_MODE; a resumed study keeps "
        "the mode it had unless you give one.",
        "partial and structured need someone to answer, so they cannot run with -p.",
    ]
    return lines


def _model_topic() -> list[str]:
    return [
        "/model                      provider, model, effort, mode and the known models",
        "/model <model>              another model from the same provider",
        "/model <provider>:<model>   another provider, if a key for it is available",
        "/model <provider>           that provider's default model",
        "/effort <level>             " + ", ".join(EFFORTS),
        "",
        "A new model is checked before it is accepted: the key, the endpoint and the "
        "model id are probed, with an image, and a model that cannot see images is "
        "refused.",
        "The switch happens between turns: typed while the agent is idle, it answers "
        "your next message; typed mid-turn, it takes over when the current turn ends.",
        "Earlier reasoning blocks are dropped from the thread when the model changes, "
        "and the prompt cache is written once more for the new model, so the first "
        "request after a switch costs more than the ones after it.",
        "If the thread is too large for the new model's context window, it is "
        "refreshed on the current model first. The window is the one known for that "
        "model where there is one (claude-haiku-4-5: 200,000 tokens), otherwise the "
        "provider's.",
        "A provider needs its key in the environment (for example ANTHROPIC_API_KEY); "
        "openreynolds config --provider <name> sets one up.",
        "A resumed study carries on on the model it was last running, together with the "
        "provider that served it, unless --model or OPENREYNOLDS_MODEL names one, that "
        "provider has no key here, or it was served through an endpoint this run does "
        "not point at. A model id belongs to the endpoint that served it: a key and a "
        "gateway in front of the same vendor list different ids.",
        "Effort is read on every request, so /effort applies from the next one.",
    ]


TOOL_HELP: tuple[tuple[str, str], ...] = (
    ("bash", "runs a shell command on the workspace and waits for it, up to a few minutes"),
    ("write_file", "writes a text file on the workspace, creating directories as needed"),
    ("read_file", "reads part of a file or lists a directory; a picture comes back as a picture"),
    ("job_start", "starts a long command, a solve for instance, detached; it keeps running "
                  "after the turn and after the session"),
    ("job_check", "reports a job's status and new log lines, and can wait for it to finish"),
    ("job_kill", "stops a running job"),
    ("fetch", "copies files from the workspace to this machine"),
    ("cad", "hands a shape described in words, or a CAD file already on the workspace, "
            "to the CAD desk: a second agent that builds it, meshes it and checks it"),
    ("checkpoint", "structured mode only: shows you a stage summary and what comes next, "
                   "and waits for your answer"),
)


def _tools_topic() -> list[str]:
    lines = ["the tools the agent works with:"]
    lines += [f"  {name.ljust(12)}{text}" for name, text in TOOL_HELP]
    lines.append("Everything else, OpenFOAM included, is done through bash on the workspace.")
    return lines


def _keys_topic(where: str) -> list[str]:
    if where == "web":
        return [
            "in the message box:",
            "  /            opens the list of matching commands",
            "  Up, Down     move through the list",
            "  Tab, Enter   pick the highlighted suggestion (Enter sends a complete line)",
            "  Esc          closes the list",
            "  click        picks a suggestion",
        ]
    if where == PLAIN:
        return [
            "this plain terminal reads a whole line at a time:",
            "  Enter        sends the line",
            "  ctrl+c       quit (jobs keep running)",
            "There is no completion and no other key here. Without --plain, the "
            "interface lists matching commands as you type / and completes them with Tab.",
        ]
    return [
        "in the prompt:",
        "  /            opens the list of matching commands above the prompt",
        "  Up, Down     move through the list",
        "  Tab          takes the highlighted suggestion, or the grey completion",
        "  Right        takes the grey completion",
        "  Enter        sends the line (on a half-typed command, takes the suggestion)",
        "  Esc          closes the list",
        "anywhere:",
        "  ctrl+t       show or hide the agent's thinking",
        "  ctrl+f       the workspace files",
        "  ctrl+g       the renders",
        "  ctrl+r       refresh the file list",
        "  ctrl+l       clear the activity pane",
        "  ctrl+c       quit (jobs keep running)",
    ]


def help_lines(topic: str = "", where: str = "terminal") -> list[str]:
    """What `/help` answers, for a topic and a surface (`terminal`, `plain` or `web`)."""
    name = (topic or "").strip().lower()
    if name in ("", "overview"):
        return _overview(where)
    if name in ("commands", "command"):
        return _commands_topic(where)
    if name in ("modes", "mode"):
        return _modes_topic()
    if name in ("model", "models", "effort"):
        return _model_topic()
    if name in ("tools", "tool"):
        return _tools_topic()
    if name in ("keys", "key", "keyboard"):
        return _keys_topic(where)
    return [
        f"no help topic called {topic.strip()!r}; the topics are "
        + ", ".join(name for name, _ in TOPICS),
        "",
        *_overview(where),
    ]


HELP_TEXT = "\n".join(help_lines())
"""The overview, as one string, for anything that wants `/help` without a surface."""


# -- completion ----------------------------------------------------------------


def _pairs(items: Iterable[Any]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if isinstance(item, (tuple, list)):
            out.append((str(item[0]), str(item[1]) if len(item) > 1 else ""))
        else:
            out.append((str(item), ""))
    return out


def completions(
    text: str,
    where: str = "terminal",
    models: Iterable[Any] = (),
    efforts: Iterable[Any] = (),
) -> list[tuple[str, str]]:
    """Suggestions for a half-typed line: (the whole line it would become, a summary).

    A verb that takes an argument completes with a space after it, so accepting it goes
    straight on to the argument's own suggestions. Aliases are offered only when typed
    in full. `models` and `efforts` are what the caller knows is available; either may
    be plain strings or (value, description) pairs.
    """
    if not text.startswith("/"):
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(line: str, summary: str) -> None:
        if line.lower() not in seen:
            seen.add(line.lower())
            out.append((line, summary))

    specs = visible(where)
    if " " not in text:
        typed = text.lower()
        for spec in specs:
            if spec.verb.startswith(typed):
                add(spec.verb + (" " if spec.args else ""), spec.summary)
        for spec in specs:
            if typed in spec.aliases:
                add(typed + (" " if spec.args else ""), spec.summary)
        return out

    verb, _, arg = text.partition(" ")
    verb = verb.lower()
    spec = next((s for s in specs if verb == s.verb or verb in s.aliases), None)
    if spec is None or " " in arg.strip():
        return []
    typed = arg.strip().lower()
    if spec.kind == MODE:
        choices = [(value, f"{label}: {description}") for value, label, description in _modes()]
    elif spec.kind == MODEL:
        choices = _pairs(models)
    elif spec.kind == EFFORT:
        choices = _pairs(efforts) or [(level, "") for level in EFFORTS]
    elif spec.kind == HELP:
        choices = list(TOPICS)
    elif spec.kind == YES:
        choices = [("all", "approve this and everything after it")]
    else:
        return []
    for value, summary in choices:
        if value.lower().startswith(typed):
            add(f"{verb} {value}", summary)
    return out


def as_json(where: str = "web") -> dict[str, Any]:
    """The command list for a client that draws its own completion (the web composer)."""
    modes = _modes()
    return {
        "commands": [
            {
                "verb": spec.verb,
                "aliases": list(spec.aliases),
                "args": spec.args,
                "summary": spec.summary,
                "detail": spec.detail,
                "choices": list(spec.choices or ([m[0] for m in modes] if spec.kind == MODE else ())),
            }
            for spec in visible(where)
        ],
        "topics": [{"name": name, "summary": summary} for name, summary in TOPICS],
        "modes": [
            {"value": value, "label": label, "description": description}
            for value, label, description in modes
        ],
        "efforts": list(EFFORTS),
    }


# -- status --------------------------------------------------------------------


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
