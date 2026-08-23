#!/usr/bin/env python3
"""End-to-end testing with a simulated user who cannot code.

Everything else in this repository tests the agent from the inside, by someone who
knows what a `tool_result` is. This drives it from the outside instead: a persona that
only writes prose, only sees what appears in the terminal, and reacts the way a client
would -- "that number looks wrong", "you said something different earlier", "would that
really separate there?". Diagnosing any of it is the agent's job, not the user's.

    python scripts/user_test.py --goal "I need the pressure drop through a 90 degree
                                        elbow in a 100mm square duct"
    python scripts/user_test.py --goal "..." --turns 8 --user-model claude-sonnet-5
    python scripts/user_test.py --study 20260823-213712-babc --goal "check this for me"

Nothing here inspects the workspace, reads a log, or touches a file. If the persona
needs to know something, it has to ask the agent, exactly like a real user.
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

import anthropic  # noqa: E402

from openreynolds.config import Config  # noqa: E402

SATISFIED = "[SATISFIED]"
STUCK = "[STUCK]"

PERSONA_SYSTEM = f"""\
You are playing a person who has hired an engineering consultant. You are testing the \
consultant by talking to them. Stay in character at all times.

Who you are: you work in a building-services or mechanical role. You have good physical \
intuition about air and water -- you know roughly how fast air moves in a duct, that a \
sharp bend costs more than a gentle one, that a pressure drop of several kilopascals \
across one fitting would be absurd, that a fan curve has to meet a system curve. You \
have opinions about whether a number smells right.

What you cannot do: you do not code. You have never written a script. You do not know \
what OpenFOAM is, what a mesh is in any technical sense, what a boundary condition is \
called, or what any command does. You would not know a residual from a radiator. You \
never type commands, never paste code, never name a piece of software, and never \
suggest a technical fix. If the consultant shows you something technical, you react to \
what it means, not to how it was done.

How you behave:
- Write one short message at a time, the way someone types in a chat. One to three \
sentences, occasionally a single line. Plain language.
- React to what you are actually told. If a number seems too big, too small, or does \
not square with something said earlier, say so and ask about it. Quote the bit that \
bothers you.
- Hold the consultant to their own words. If they said one thing and later say \
another, point at the difference. If they gave you a number without saying how sure \
they are, ask how confident they are.
- Be suspicious of confidence without evidence, and of a result that arrives suspiciously \
fast.
- If they ask you a question, answer it as this person would -- with judgement about the \
application, never with technical settings. It is fine to say "I don't know, you're the \
expert, use your judgement."
- If they are clearly working -- something is running, they said it will take a while -- \
it is fine to say so briefly and let them get on with it.
- Do not be a pushover and do not be a jerk. You are a client who wants a defensible \
answer.

Ending: when you have an answer you would actually accept -- a number, with some sense \
of how much to trust it -- reply with your closing remark and then, on its own final \
line, exactly {SATISFIED}. If the consultant is stuck in a loop, has given up, or the \
conversation has clearly stopped going anywhere, reply and then put {STUCK} on its own \
final line. Do not use either marker before then.

Write only your message. No preamble, no stage directions, no quotation marks around it.
"""

CODE_LIKE = re.compile(
    r"```|\bsudo\b|\bpython3?\s+\S+\.py\b|\bblockMesh\b|\bsimpleFoam\b|\bcheckMesh\b"
    r"|\bsnappyHexMesh\b|\bfoamDictionary\b|\bcd\s+/|\bls\s+-|/work/",
    re.IGNORECASE,
)


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
    kept = [line for line in message.splitlines() if not CODE_LIKE.search(line)]
    cleaned = "\n".join(kept).strip()
    return cleaned or "Sorry, could you explain that in plain terms?"


def next_user_message(client, model: str, exchanges: list[tuple[str, str]], goal: str) -> str:
    """Ask the persona what it says next, given everything it has been shown."""
    conversation: list[dict] = [
        {"role": "user", "content": f"Your goal, in your own words: {goal}\n\nOpen the conversation."}
    ]
    for said, seen in exchanges:
        conversation.append({"role": "assistant", "content": said})
        conversation.append(
            {"role": "user", "content": f"The consultant replied:\n\n{seen[-6000:]}"}
        )

    reply = client.messages.create(
        model=model,
        max_tokens=1000,
        system=PERSONA_SYSTEM,
        messages=conversation,
    )
    return "".join(b.text for b in reply.content if b.type == "text").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True, help="What the simulated user wants.")
    parser.add_argument("--turns", type=int, default=10, help="Maximum user messages.")
    parser.add_argument("--idle", type=float, default=20.0, help="Seconds of silence = its turn.")
    parser.add_argument("--cap", type=float, default=1800.0, help="Seconds to wait per reply.")
    parser.add_argument("--study", default=None, help="Resume an existing study.")
    parser.add_argument("--model", default=None, help="Model for the agent under test.")
    parser.add_argument("--user-model", default=None, help="Model playing the user.")
    parser.add_argument("--log", type=Path, default=None, help="Where to write the transcript.")
    args = parser.parse_args()

    cfg = Config.load()
    missing = cfg.missing()
    if missing:
        print(f"missing configuration: {', '.join(missing)}")
        return 2

    client = anthropic.Anthropic(
        api_key=cfg.anthropic_api_key or None, base_url=cfg.llm_base_url or None
    )
    user_model = args.user_model or cfg.model

    argv = [sys.executable, "-u", "-m", "openreynolds.cli"]
    if args.study:
        argv += ["--study", args.study]
    if args.model:
        argv += ["--model", args.model]

    repo = Path(__file__).resolve().parents[1]
    log = args.log or repo / "user-test.log"
    transcript: list[str] = []

    def record(who: str, text: str) -> None:
        block = f"\n{'=' * 70}\n{who}\n{'=' * 70}\n{text.rstrip()}\n"
        transcript.append(block)
        print(block, flush=True)
        log.write_text("".join(transcript), encoding="utf-8")

    session = AgentSession(argv, repo)
    exchanges: list[tuple[str, str]] = []
    verdict = "ran out of turns"
    started = time.monotonic()

    try:
        opening, alive = session.read_until_quiet(args.idle, args.cap)
        record("AGENT (startup)", opening)

        for turn in range(args.turns):
            if not alive:
                verdict = "the agent exited"
                break

            raw = next_user_message(client, user_model, exchanges, args.goal)
            done = SATISFIED in raw or STUCK in raw
            verdict = (
                "satisfied" if SATISFIED in raw else "stuck" if STUCK in raw else verdict
            )
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
    record(
        "RESULT",
        f"verdict: {verdict}\nturns used: {len(exchanges)}\n"
        f"wall clock: {elapsed / 60:.1f} min\nagent exit code: {code}\ntranscript: {log}",
    )
    return 0 if verdict == "satisfied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
