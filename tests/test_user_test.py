"""The simulated-user harness.

The point of this harness is to test the product the way someone who cannot code would
use it, so the thing most worth testing is that the persona stays that person.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "user_test.py"


@pytest.fixture(scope="module")
def ut():
    spec = importlib.util.spec_from_file_location("user_test_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_test_under_test"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("user_test_under_test", None)


# -- staying in character ------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "run `blockMesh` first",
        "try python3 gen_case.py",
        "just cd /work/elbow and look",
        "```\nsimpleFoam -parallel\n```",
        "check /work/elbow/log.simpleFoam",
        "you should run checkMesh on that",
        "use snappyHexMesh instead",
        "ls -la the case directory",
    ],
)
def test_technical_lines_never_reach_the_agent(ut, line):
    """A user who cannot code cannot paste a command. If the model slips into being an
    engineer, the line is dropped rather than sent -- otherwise the test quietly stops
    testing what it claims to."""
    assert ut.CODE_LIKE.search(line), "this line should be recognised as technical"
    assert line not in ut.sanitize(line)


@pytest.mark.parametrize(
    "line",
    [
        "That pressure drop seems way too high for a duct that size.",
        "You said 1.2 earlier and now you're saying 0.8 - which is it?",
        "How confident are you in that number?",
        "Would the air really separate right after the bend?",
        "I don't know, you're the expert - use your judgement.",
        "Fine, take your time.",
    ],
)
def test_ordinary_user_language_passes_through(ut, line):
    assert ut.sanitize(line) == line


def test_a_wholly_technical_message_degrades_to_a_plain_question(ut):
    assert ut.sanitize("```\nblockMesh\n```") == "Sorry, could you explain that in plain terms?"


def test_mixed_message_keeps_only_the_human_part(ut):
    cleaned = ut.sanitize(
        "That number looks off to me.\nrun checkMesh and see\nCan you double check it?"
    )
    assert "That number looks off to me." in cleaned
    assert "Can you double check it?" in cleaned
    assert "checkMesh" not in cleaned


def test_the_persona_is_told_what_it_cannot_do(ut):
    for constraint in ("you do not code", "never type commands", "never paste code"):
        assert constraint in ut.PERSONA_SYSTEM.lower()


def test_the_persona_is_told_to_push_back(ut):
    lowered = ut.PERSONA_SYSTEM.lower()
    assert "does not square with something said earlier" in lowered
    assert "how confident" in lowered
    assert "suspicious of confidence without evidence" in lowered


def test_the_persona_can_end_the_conversation(ut):
    assert ut.SATISFIED in ut.PERSONA_SYSTEM
    assert ut.STUCK in ut.PERSONA_SYSTEM


# -- driving the subprocess ----------------------------------------------------


def spawn(ut, python_code: str):
    return ut.AgentSession([sys.executable, "-u", "-c", python_code], Path.cwd())


def test_it_reads_until_the_agent_goes_quiet(ut):
    session = spawn(ut, "import time,sys; print('hello'); sys.stdout.flush(); time.sleep(30)")
    try:
        text, alive = session.read_until_quiet(idle_s=0.5, hard_cap_s=10)
        assert "hello" in text
        assert alive is True
    finally:
        session.process.kill()


def test_it_notices_when_the_agent_exits(ut):
    session = spawn(ut, "print('done')")
    text, alive = session.read_until_quiet(idle_s=0.5, hard_cap_s=10)
    assert "done" in text
    assert alive is False


def test_it_round_trips_a_message(ut):
    session = spawn(
        ut,
        "import sys\n"
        "print('ready'); sys.stdout.flush()\n"
        "line = sys.stdin.readline().strip()\n"
        "print(f'you said: {line}'); sys.stdout.flush()\n",
    )
    try:
        session.read_until_quiet(idle_s=0.5, hard_cap_s=10)
        session.send("that looks wrong to me")
        reply, _alive = session.read_until_quiet(idle_s=0.5, hard_cap_s=10)
        assert "you said: that looks wrong to me" in reply
    finally:
        session.process.kill()


def test_silence_while_a_job_is_watched_is_not_mistaken_for_a_turn(ut):
    """A solve prints nothing for minutes. Interrupting it would be a bad user."""
    quick = ut.AgentSession._idle_for(["some output\n"], idle_s=10)
    watching = ut.AgentSession._idle_for(["watching 1 job(s): solve\n"], idle_s=10)
    assert watching > quick
    assert watching == 60


def test_close_is_bounded_even_if_the_agent_ignores_exit(ut):
    session = spawn(ut, "import time; time.sleep(120)")
    started = time.monotonic()
    session.close(timeout=2)
    assert time.monotonic() - started < 20
    assert session.process.poll() is not None


# -- the conversation it builds ------------------------------------------------


def test_the_persona_sees_only_what_the_terminal_showed(ut):
    """It has no file access, no logs, no internals -- only the reply text."""
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Why is it that high?")]
            )

    from types import SimpleNamespace

    client = SimpleNamespace(messages=FakeMessages())
    exchanges = [("what's the pressure drop?", "About 240 Pa.")]

    reply = ut.next_user_message(client, "claude-opus-5", exchanges, "I need a pressure drop")

    assert reply == "Why is it that high?"
    assert captured["system"] is ut.PERSONA_SYSTEM
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert "I need a pressure drop" in captured["messages"][0]["content"]
    assert "About 240 Pa." in captured["messages"][2]["content"]


def test_a_very_long_agent_reply_is_trimmed_not_dropped(ut):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    from types import SimpleNamespace

    huge = "x" * 20_000 + "THE-ENDING"
    ut.next_user_message(
        SimpleNamespace(messages=FakeMessages()), "m", [("hi", huge)], "goal"
    )
    shown = captured["messages"][2]["content"]
    assert "THE-ENDING" in shown, "the most recent part is what a user would have on screen"
    assert len(shown) < 7000
