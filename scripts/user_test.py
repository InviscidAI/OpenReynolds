#!/usr/bin/env python3
"""End-to-end testing with a simulated user who cannot code.

Everything else in this repository tests the agent from the inside, by someone who
knows what a `tool_result` is. This drives it from the outside instead: a persona that
only writes prose, only sees what appears in the terminal, and reacts the way a client
would -- "that number looks wrong", "you said something different earlier", "would that
really separate there?". Diagnosing any of it is the agent's job, not the user's.

    python scripts/user_test.py --persona engineer
    python scripts/user_test.py --persona all --budget 12
    python scripts/user_test.py --goal "..." --turns 8 --user-model claude-sonnet-5

Personas live in `scripts/personas.py`; `--goal` overrides whichever one is chosen.
Nothing here inspects the workspace, reads a log, or touches a file. If the persona
needs to know something, it has to ask the agent, exactly like a real user.

A run is bounded three ways -- turns, seconds per reply, and total minutes -- because a
test nobody will sit through is a test nobody runs. Whatever is still going on the
instance when a persona finishes gets stopped, so the next one starts on a quiet
machine and nothing is left burning.
"""

from __future__ import annotations

import argparse
import codecs
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anthropic  # noqa: E402

from openreynolds.config import Config  # noqa: E402
from openreynolds.terminal import tolerant_stdout  # noqa: E402
from personas import ALL, SATISFIED, STUCK, Persona  # noqa: E402

CODE_LIKE = re.compile(
    r"```|\bsudo\b|\bpython3?\s+\S+\.py\b|\bblockMesh\b|\bsimpleFoam\b|\bcheckMesh\b"
    r"|\bsnappyHexMesh\b|\bfoamDictionary\b|\bcd\s+/|\bls\s+-|/work/",
    re.IGNORECASE,
)

INTERFACE_COMMAND = re.compile(r"^/(btw|status|files|help|open|ls)\b", re.IGNORECASE)
"""Typing `/status` is using the product, not writing code, so it survives the filter."""

# The agent's output passes through here on its way to the screen, so the harness needs
# the same protection the product has. Without it one sigma in one reply ends the whole
# run, which is exactly what happened the first time this was pointed at a live session.
tolerant_stdout()

STUDY_ID = re.compile(r"\bstudy\s+(\d{8}-\d{6}-[0-9a-f]{4})\b")


PROMPT = re.compile(r"\n> ?$")
"""The product saying it is the user's turn.

It carries no trailing newline -- nothing follows it until somebody answers -- which
is exactly why the reader has to work in bytes rather than lines.
"""

HEARTBEAT_S = 30.0
"""How often a long wait says it is still a wait and not a dead run."""


@dataclass
class Reply:
    """One stretch of agent output, and how it came to an end."""

    text: str
    alive: bool = True
    timed_out: bool = False
    by_prompt: bool = False
    """True when the agent asked for input, rather than merely going quiet."""
    mid_turn: bool = False
    """True when the user got tired of waiting and spoke over work in progress."""

    def note(self) -> str:
        """What to write down about how this ended, when it did not end normally."""
        marks = []
        if self.timed_out:
            marks.append("TIMED OUT -- it was still producing output when the cap ran out")
        if not self.alive:
            marks.append("THE AGENT EXITED")
        if not self.text.strip():
            marks.append("NOTHING WAS SAID")
        elif self.mid_turn:
            marks.append("STILL WORKING -- the user spoke over it")
        elif not self.by_prompt and not self.timed_out and self.alive:
            marks.append("went quiet without asking for input")
        return "  [" + "; ".join(marks) + "]" if marks else ""


class AgentSession:
    """The product under test, as a subprocess, seen only through its terminal."""

    def __init__(self, argv: list[str], cwd: Path):
        self.process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._chunks: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        """Read bytes, not lines.

        The prompt that says "your turn" is written without a trailing newline, so a
        line-oriented reader cannot see it until something else arrives -- which is
        never, because nothing else will until somebody answers it. Reading bytes is
        what makes waiting for the prompt possible at all.
        """
        assert self.process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        fileno = self.process.stdout.fileno()
        while True:
            try:
                data = os.read(fileno, 65536)
            except OSError:
                break
            if not data:
                break
            text = decoder.decode(data)
            if text:
                self._chunks.put(text)
        self._chunks.put(None)

    def read_until_turn(
        self, idle_s: float, hard_cap_s: float, speak_after: float = 0.0
    ) -> Reply:
        """Collect output until the agent asks for input, goes quiet, or runs out of cap.

        Waiting for the prompt rather than for silence is the whole difference between
        a conversation and two monologues. Silence is a guess, and it guesses wrong
        every time a command takes longer than the threshold: the reply gets cut mid
        sentence, the next message lands while the agent is still working, and two
        turns are spent talking past each other.

        Silence is still the fallback, because while a job is being watched there is
        no prompt to wait for.

        `speak_after` is the impatient user. A turn can legitimately run for ten
        minutes -- the agent polls its own solve with a blocking sleep, so there is no
        prompt and no silence -- and a person watching that would not sit through it.
        They would say something, and it would be heard, because anything typed while
        the agent works reaches it between tool calls. A harness that only ever speaks
        at a prompt never tests that path at all.
        """
        collected: list[str] = []
        tail = ""
        started = time.monotonic()
        deadline = started + hard_cap_s
        last = started
        announced = started

        while time.monotonic() < deadline:
            try:
                chunk = self._chunks.get(timeout=0.5)
            except queue.Empty:
                now = time.monotonic()
                if now - last >= self._idle_for(collected, idle_s):
                    return Reply("".join(collected), alive=True)
                if speak_after and now - started >= speak_after and collected:
                    return Reply("".join(collected), alive=True, mid_turn=True)
                if now - announced >= HEARTBEAT_S:
                    announced = now
                    # Otherwise a seven-minute wait and a dead run look identical from
                    # outside, and whoever is watching has no way to tell.
                    print(
                        f"    [waiting {now - started:.0f}s, "
                        f"quiet for {now - last:.0f}s]",
                        flush=True,
                    )
                continue
            if chunk is None:
                return Reply("".join(collected), alive=False)
            collected.append(chunk)
            tail = (tail + chunk)[-32:]
            last = time.monotonic()
            if PROMPT.search(tail):
                return Reply("".join(collected), alive=True, by_prompt=True)
            if speak_after and last - started >= speak_after:
                return Reply("".join(collected), alive=True, mid_turn=True)

        return Reply("".join(collected), alive=True, timed_out=True)

    @staticmethod
    def _idle_for(collected: list[str], idle_s: float) -> float:
        """Wait longer while a job is being watched -- silence there means work."""
        tail = "".join(collected[-30:])
        return idle_s * 6 if "watching" in tail else idle_s

    def send(self, text: str) -> None:
        """The pipe is bytes now, because the reader had to be."""
        assert self.process.stdin is not None
        self.process.stdin.write((text.rstrip("\n") + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def close(self, timeout: float = 60.0) -> int | None:
        try:
            if self.process.poll() is None:
                self.send("/exit")
            return self.process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            # kill() only signals; without the wait the process is not yet reaped and
            # the exit code comes back as None.
            self.process.kill()
            try:
                return self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                return self.process.poll()


def sanitize(message: str) -> str:
    """Keep the persona in character.

    A user who cannot code cannot paste a command, so if the model slips into being an
    engineer the line is dropped rather than sent. Better a shorter message than a test
    that quietly stops testing what it claims to.
    """
    kept = [
        line
        for line in message.splitlines()
        if INTERFACE_COMMAND.match(line.strip()) or not CODE_LIKE.search(line)
    ]
    cleaned = "\n".join(kept).strip()
    return cleaned or "Sorry, could you explain that in plain terms?"


def next_user_message(
    client, model: str, persona: Persona, exchanges: list[tuple[str, str, bool]], goal: str
) -> str:
    """Ask the persona what it says next, given everything it has been shown."""
    conversation: list[dict] = [
        {
            "role": "user",
            "content": f"Your goal, in your own words: {goal}\n\nOpen the conversation.",
        }
    ]
    for said, seen, interrupted in exchanges:
        conversation.append({"role": "assistant", "content": said})
        # A persona that thinks it was answered writes as though it was. Saying which
        # of the two happened is the difference between "that number looks wrong" and
        # "are you still going?", and only one of those is in character here.
        heading = (
            "The consultant is still working. This is what you can see so far"
            if interrupted
            else "The consultant replied"
        )
        conversation.append({"role": "user", "content": f"{heading}:\n\n{seen[-6000:]}"})

    reply = client.messages.create(
        model=model,
        max_tokens=1000,
        system=persona.system,
        messages=conversation,
    )
    return "".join(b.text for b in reply.content if b.type == "text").strip()


ATTIC = "/work/.attic"


def clear_workspace() -> str:
    """Move earlier work aside so a persona opens on an empty workspace.

    The volume outlives every session, so without this the second persona finds the
    first one's case and answers from it. That happened: a run inherited a velocity
    from an abandoned case and carried it for several turns.

    Moved, not deleted. A test that destroys work nobody asked it to destroy is a
    worse problem than the one it set out to fix.
    """
    from openreynolds.backend import hosted

    cfg = Config.load()
    backend, _client, _iid = hosted.acquire(cfg.foamd_url, cfg.foamd_api_key, None)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    attic = f"{ATTIC}/{stamp}"
    try:
        result = backend.exec(
            f"mkdir -p {attic}; "
            f"find /work -maxdepth 1 -mindepth 1 ! -name '.*' -exec mv -t {attic} {{}} + "
            f"2>/dev/null; "
            f"ls -A {attic} | wc -l; rmdir {attic} 2>/dev/null; true",
            timeout_s=120,
        )
    except Exception as exc:
        return f"could not clear the workspace: {exc}"
    finally:
        backend.close()

    counted = (result.output or "0").strip().splitlines()
    moved = counted[-1].strip() if counted else "0"
    return f"moved {moved} earlier item(s) aside to {attic}" if moved != "0" else "workspace was already empty"


def quieten(repo: Path, study: str | None) -> str:
    """Stop whatever this run left running, so the next one starts on a quiet machine."""
    if not study:
        return "no study id was seen, so nothing could be stopped"
    result = subprocess.run(
        [sys.executable, "-m", "openreynolds.cli", "stop", "--study", study, "--force"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return (result.stdout or result.stderr or "").strip() or "nothing to stop"


def run_one(client, persona: Persona, args, repo: Path) -> dict:
    """One persona, one session, bounded three ways."""
    argv = [sys.executable, "-u", "-m", "openreynolds.cli"]
    if args.model:
        argv += ["--model", args.model]

    log = args.log or repo / f"user-test-{persona.name}.log"
    goal = args.goal or persona.goal
    transcript: list[str] = []

    def record(who: str, text: str) -> None:
        block = f"\n{'=' * 70}\n{who}\n{'=' * 70}\n{text.rstrip()}\n"
        transcript.append(block)
        print(block, flush=True)
        log.write_text("".join(transcript), encoding="utf-8")

    record("PERSONA", f"{persona.name}\n\ngoal: {goal}")
    if args.fresh:
        record("WORKSPACE", clear_workspace())

    session = AgentSession(argv, repo)
    exchanges: list[tuple[str, str, bool]] = []
    verdict = "ran out of turns"
    study: str | None = None
    started = time.monotonic()
    budget = started + args.budget * 60 if args.budget else None

    try:
        opening = session.read_until_turn(args.idle, args.cap, args.speak_after)
        record("AGENT (startup)" + opening.note(), opening.text)
        found = STUDY_ID.search(opening.text)
        study = found.group(1) if found else None
        reply = opening

        # An explicit --turns is an instruction, not a ceiling to be clipped by the
        # persona's own default: asking for more and silently getting fewer is how a
        # conversation gets cut off and reported as having run its course.
        wanted = persona.turns if args.turns is None else args.turns
        for turn in range(wanted):
            if not reply.alive:
                verdict = "the agent exited"
                break
            if reply.timed_out:
                # Carrying on here means talking to something that has stopped
                # listening, and calling the result a conversation.
                verdict = f"stalled: still producing output after {args.cap:g}s"
                break
            if budget and time.monotonic() > budget:
                verdict = f"out of time after {args.budget:g} min"
                break

            raw = next_user_message(client, args.user_model, persona, exchanges, goal)
            done = SATISFIED in raw or STUCK in raw
            verdict = "satisfied" if SATISFIED in raw else "stuck" if STUCK in raw else verdict
            message = sanitize(raw.replace(SATISFIED, "").replace(STUCK, "").strip())
            record(f"USER (turn {turn + 1})", message)
            if done:
                break

            session.send(message)
            reply = session.read_until_turn(args.idle, args.cap, args.speak_after)
            record(f"AGENT (turn {turn + 1})" + reply.note(), reply.text)
            if not reply.text.strip():
                verdict = "the agent said nothing"
                break
            exchanges.append((message, reply.text, reply.mid_turn))
    finally:
        code = session.close()

    elapsed = time.monotonic() - started
    record("CLEANUP", quieten(repo, study))
    record(
        "RESULT",
        f"persona: {persona.name}\nverdict: {verdict}\nturns used: {len(exchanges)}\n"
        f"wall clock: {elapsed / 60:.1f} min\nstudy: {study}\n"
        f"agent exit code: {code}\ntranscript: {log}",
    )
    return {
        "persona": persona.name,
        "verdict": verdict,
        "turns": len(exchanges),
        "minutes": elapsed / 60,
        "study": study,
        "log": str(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona",
        default="engineer",
        help=f"One of {', '.join(sorted(ALL))}, or 'all' to run each in turn.",
    )
    parser.add_argument("--goal", default=None, help="Override the persona's own goal.")
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Maximum user messages. Defaults to whatever the persona asks for.",
    )
    parser.add_argument(
        "--idle",
        type=float,
        default=45.0,
        help="Fallback: seconds of silence taken as its turn when no prompt appears.",
    )
    parser.add_argument("--cap", type=float, default=600.0, help="Seconds to wait per reply.")
    parser.add_argument(
        "--speak-after",
        type=float,
        default=180.0,
        help="Seconds of work before the user speaks over it anyway. 0 = wait forever.",
    )
    parser.add_argument("--budget", type=float, default=15.0, help="Minutes per persona, 0 = none.")
    parser.add_argument("--model", default=None, help="Model for the agent under test.")
    parser.add_argument("--user-model", default=None, help="Model playing the user.")
    parser.add_argument("--log", type=Path, default=None, help="Where to write the transcript.")
    parser.add_argument(
        "--fresh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Move every earlier study aside first. Rarely needed: a study now gets "
            "its own directory, so personas no longer land in each other's work."
        ),
    )
    args = parser.parse_args()

    cfg = Config.load()
    missing = cfg.missing()
    if missing:
        print(f"missing configuration: {', '.join(missing)}")
        return 2

    chosen = list(ALL.values()) if args.persona == "all" else [ALL.get(args.persona)]
    if chosen == [None]:
        print(f"no such persona: {args.persona}. Try one of {', '.join(sorted(ALL))}, or all.")
        return 2
    if args.persona == "all" and args.goal:
        print("--goal with --persona all would give every persona the same words; pick one.")
        return 2

    client = anthropic.Anthropic(
        api_key=cfg.llm_api_key or None, base_url=cfg.llm_base_url or None
    )
    args.user_model = args.user_model or cfg.model
    repo = Path(__file__).resolve().parents[1]

    results = []
    for persona in chosen:
        # One persona falling over must not take the other three with it. The first
        # live run of this suite died on persona one, turn one, and the other three
        # never ran -- so the crash cost four runs' worth of information, not one.
        try:
            results.append(run_one(client, persona, args, repo))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "persona": persona.name,
                    "verdict": f"harness failed: {type(exc).__name__}",
                    "turns": 0,
                    "minutes": 0.0,
                    "study": None,
                    "log": "-",
                }
            )

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for result in results:
        print(
            f"  {result['persona']:<12} {result['verdict']:<28} "
            f"{result['turns']} turns  {result['minutes']:.1f} min  {result['log']}"
        )
    return 0 if all(r["verdict"] == "satisfied" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
