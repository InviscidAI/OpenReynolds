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
import queue
import re
import subprocess
import sys
import threading
import time
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


class AgentSession:
    """The product under test, as a subprocess, seen only through its terminal."""

    def __init__(self, argv: list[str], cwd: Path):
        self.process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._chunks: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._chunks.put(line)
        self._chunks.put(None)

    def read_until_quiet(self, idle_s: float, hard_cap_s: float) -> tuple[str, bool]:
        """Collect output until it stops for `idle_s`. Returns (text, still_running)."""
        collected: list[str] = []
        deadline = time.monotonic() + hard_cap_s
        last = time.monotonic()

        while time.monotonic() < deadline:
            try:
                chunk = self._chunks.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - last >= self._idle_for(collected, idle_s):
                    return "".join(collected), True
                continue
            if chunk is None:
                return "".join(collected), False
            collected.append(chunk)
            last = time.monotonic()

        return "".join(collected), True

    @staticmethod
    def _idle_for(collected: list[str], idle_s: float) -> float:
        """Wait longer while a job is being watched -- silence there means work."""
        tail = "".join(collected[-30:])
        return idle_s * 6 if "watching" in tail else idle_s

    def send(self, text: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(text.rstrip("\n") + "\n")
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
    client, model: str, persona: Persona, exchanges: list[tuple[str, str]], goal: str
) -> str:
    """Ask the persona what it says next, given everything it has been shown."""
    conversation: list[dict] = [
        {
            "role": "user",
            "content": f"Your goal, in your own words: {goal}\n\nOpen the conversation.",
        }
    ]
    for said, seen in exchanges:
        conversation.append({"role": "assistant", "content": said})
        conversation.append(
            {"role": "user", "content": f"The consultant replied:\n\n{seen[-6000:]}"}
        )

    reply = client.messages.create(
        model=model,
        max_tokens=1000,
        system=persona.system,
        messages=conversation,
    )
    return "".join(b.text for b in reply.content if b.type == "text").strip()


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

    session = AgentSession(argv, repo)
    exchanges: list[tuple[str, str]] = []
    verdict = "ran out of turns"
    study: str | None = None
    started = time.monotonic()
    budget = started + args.budget * 60 if args.budget else None

    try:
        opening, alive = session.read_until_quiet(args.idle, args.cap)
        record("AGENT (startup)", opening)
        found = STUDY_ID.search(opening)
        study = found.group(1) if found else None

        for turn in range(min(args.turns, persona.turns)):
            if not alive:
                verdict = "the agent exited"
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
            reply, alive = session.read_until_quiet(args.idle, args.cap)
            record(f"AGENT (turn {turn + 1})", reply)
            exchanges.append((message, reply))
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
    parser.add_argument("--turns", type=int, default=8, help="Maximum user messages.")
    parser.add_argument("--idle", type=float, default=15.0, help="Seconds of silence = its turn.")
    parser.add_argument("--cap", type=float, default=600.0, help="Seconds to wait per reply.")
    parser.add_argument("--budget", type=float, default=15.0, help="Minutes per persona, 0 = none.")
    parser.add_argument("--model", default=None, help="Model for the agent under test.")
    parser.add_argument("--user-model", default=None, help="Model playing the user.")
    parser.add_argument("--log", type=Path, default=None, help="Where to write the transcript.")
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
        api_key=cfg.anthropic_api_key or None, base_url=cfg.llm_base_url or None
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
