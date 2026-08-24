"""The simulated-user harness.

The point of this harness is to test the product the way someone who cannot code would
use it, so the thing most worth testing is that the persona stays that person.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
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


def test_every_persona_is_told_what_it_cannot_do(ut):
    for persona in ut.ALL.values():
        lowered = persona.system.lower()
        for constraint in ("you do not code", "never type commands", "never paste code"):
            assert constraint in lowered, f"{persona.name} was not told: {constraint}"


def test_every_persona_can_end_the_conversation(ut):
    for persona in ut.ALL.values():
        assert ut.SATISFIED in persona.system
        assert ut.STUCK in persona.system


def test_every_persona_has_a_goal_of_its_own(ut):
    for persona in ut.ALL.values():
        assert len(persona.goal.split()) > 8, f"{persona.name} has no real goal"


def test_the_personas_push_on_different_things(ut):
    """One persona finds one class of problem. A suite of identical ones finds one
    class of problem four times."""
    characters = {persona.name: persona.character for persona in ut.ALL.values()}
    assert len(set(characters.values())) == len(characters)
    assert "intuition" in characters["engineer"]
    assert "no physical intuition at all" in characters["novice"]
    assert "visibility" in characters["controller"]
    assert "no longer holds" in characters["shifting"]


def test_only_the_persona_who_would_use_them_is_told_about_the_commands(ut):
    """Someone who has read the help knows about /status. Telling every persona would
    test the feature and nothing else."""
    assert ut.ALL["controller"].knows_interface
    assert not ut.ALL["novice"].knows_interface
    assert "/status" in ut.ALL["controller"].system
    assert "/status" not in ut.ALL["engineer"].system


# -- driving the subprocess ----------------------------------------------------


def spawn(ut, python_code: str):
    return ut.AgentSession([sys.executable, "-u", "-c", python_code], Path.cwd())


def test_it_reads_until_the_agent_goes_quiet(ut):
    session = spawn(ut, "import time,sys; print('hello'); sys.stdout.flush(); time.sleep(30)")
    try:
        reply = session.read_until_turn(idle_s=0.5, hard_cap_s=10)
        assert "hello" in reply.text
        assert reply.alive is True
        assert reply.timed_out is False
        assert reply.by_prompt is False
        assert "went quiet without asking for input" in reply.note()
    finally:
        session.process.kill()


def test_it_notices_when_the_agent_exits(ut):
    session = spawn(ut, "print('done')")
    reply = session.read_until_turn(idle_s=0.5, hard_cap_s=10)
    assert "done" in reply.text
    assert reply.alive is False
    assert "THE AGENT EXITED" in reply.note()


def test_it_round_trips_a_message(ut):
    session = spawn(
        ut,
        "import sys\n"
        "print('ready'); sys.stdout.flush()\n"
        "line = sys.stdin.readline().strip()\n"
        "print(f'you said: {line}'); sys.stdout.flush()\n",
    )
    try:
        session.read_until_turn(idle_s=0.5, hard_cap_s=10)
        session.send("that looks wrong to me")
        reply = session.read_until_turn(idle_s=0.5, hard_cap_s=10)
        assert "you said: that looks wrong to me" in reply.text
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
    exchanges = [("what's the pressure drop?", "About 240 Pa.", False)]

    persona = ut.ALL["engineer"]
    reply = ut.next_user_message(
        client, "claude-opus-5", persona, exchanges, "I need a pressure drop"
    )

    assert reply == "Why is it that high?"
    assert captured["system"] == persona.system
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
        SimpleNamespace(messages=FakeMessages()), "m", ut.ALL["engineer"], [("hi", huge, False)], "goal"
    )
    shown = captured["messages"][2]["content"]
    assert "THE-ENDING" in shown, "the most recent part is what a user would have on screen"
    assert len(shown) < 7000


# -- breaking loudly -----------------------------------------------------------


def test_running_out_of_cap_is_not_reported_as_a_finished_turn(ut):
    """Going quiet means the agent finished. Running out of cap means it never did,
    and a run that confuses the two keeps talking to something that stopped listening."""
    session = spawn(
        ut,
        "import sys, time\n"
        "for _ in range(200):\n"
        "    print('still working'); sys.stdout.flush(); time.sleep(0.05)\n",
    )
    try:
        reply = session.read_until_turn(idle_s=5.0, hard_cap_s=1.0)
        assert reply.timed_out is True
        assert "TIMED OUT" in reply.note()
    finally:
        session.process.kill()


def test_an_agent_that_says_nothing_at_all_is_marked_as_such(ut):
    assert "NOTHING WAS SAID" in ut.Reply("").note()
    assert "NOTHING WAS SAID" in ut.Reply("   \n ").note()
    assert "NOTHING WAS SAID" not in ut.Reply("a number").note()


def test_a_long_wait_says_it_is_still_waiting(ut, capsys):
    """A seven-minute wait and a dead run look identical from outside unless one of
    them says something."""
    session = spawn(ut, "import time; time.sleep(30)")
    try:
        ut.HEARTBEAT_S = 0.2
        session.read_until_turn(idle_s=1.0, hard_cap_s=1.2)
        assert "waiting" in capsys.readouterr().out
    finally:
        ut.HEARTBEAT_S = 30.0
        session.process.kill()


# -- waiting for the turn, not for silence -------------------------------------

PROMPTS = "import sys, time\nsys.stdout.write('\\n> ')\nsys.stdout.flush()\ntime.sleep(30)\n"


def test_the_prompt_ends_a_turn_immediately(ut):
    """The whole point: an agent that asks for input is done, whatever the clock says.

    Waiting out an idle period instead wastes it, and guessing from silence cuts the
    reply mid sentence and sends the next message while the agent is still working.
    """
    session = spawn(ut, "print('here is your number')\n" + PROMPTS)
    try:
        started = time.monotonic()
        reply = session.read_until_turn(idle_s=20.0, hard_cap_s=25.0)
        elapsed = time.monotonic() - started

        assert reply.by_prompt is True
        assert "here is your number" in reply.text
        assert elapsed < 10, "it returned on the prompt, not after waiting out the idle"
        assert reply.note() == "", "asking for input is a turn ending normally"
    finally:
        session.process.kill()


def test_a_prompt_with_no_trailing_space_is_still_a_prompt(ut):
    """Rich strips trailing whitespace on some streams, so both forms arrive."""
    assert ut.PROMPT.search("all done\n>")
    assert ut.PROMPT.search("all done\n> ")
    assert not ut.PROMPT.search("the ratio is 3 > 2 here")
    assert not ut.PROMPT.search("still working\n")


def test_a_slow_command_does_not_end_the_turn(ut):
    """The failure this replaced: output pausing while a command ran was read as the
    agent finishing, so the next message landed mid-work and two turns were wasted."""
    session = spawn(
        ut,
        "import sys, time\n"
        "print('starting the mesh'); sys.stdout.flush()\n"
        "time.sleep(1.2)\n"
        "print('done'); sys.stdout.flush()\n" + PROMPTS,
    )
    try:
        reply = session.read_until_turn(idle_s=3.0, hard_cap_s=20.0)
        assert reply.by_prompt is True
        assert "starting the mesh" in reply.text and "done" in reply.text
    finally:
        session.process.kill()


def test_bytes_are_decoded_even_when_split_across_reads(ut):
    """Reading raw bytes means a multi-byte character can straddle two reads."""
    session = spawn(
        ut,
        "import sys, time\n"
        "raw = 'sigma \\u03c3 and mu \\u00b5\\n'.encode('utf-8')\n"
        "for byte in raw:\n"
        "    sys.stdout.buffer.write(bytes([byte]))\n"
        "    sys.stdout.buffer.flush()\n"
        "    time.sleep(0.002)\n" + PROMPTS,
    )
    try:
        reply = session.read_until_turn(idle_s=5.0, hard_cap_s=20.0)
        assert "sigma σ and mu µ" in reply.text
    finally:
        session.process.kill()


# -- the impatient user --------------------------------------------------------


def test_a_long_turn_gets_spoken_over_rather_than_waited_out(ut):
    """A turn can legitimately run ten minutes: the agent polls its own solve with a
    blocking sleep, so there is no prompt and no silence. A person would not sit
    through that, and anything they typed would be heard between tool calls -- a
    harness that only speaks at a prompt never exercises that path."""
    session = spawn(
        ut,
        "import sys, time\n"
        "print('meshing'); sys.stdout.flush()\n"
        "for _ in range(100):\n"
        "    print('  bash still running'); sys.stdout.flush(); time.sleep(0.1)\n",
    )
    try:
        reply = session.read_until_turn(idle_s=30.0, hard_cap_s=30.0, speak_after=1.0)

        assert reply.mid_turn is True
        assert reply.by_prompt is False
        assert reply.timed_out is False
        assert "meshing" in reply.text
        assert "STILL WORKING" in reply.note()
    finally:
        session.process.kill()


def test_a_prompt_still_wins_over_impatience(ut):
    """If the agent finishes first, that is a finished turn, not an interruption."""
    session = spawn(ut, "print('done, over to you')\n" + PROMPTS)
    try:
        reply = session.read_until_turn(idle_s=30.0, hard_cap_s=30.0, speak_after=10.0)
        assert reply.by_prompt is True
        assert reply.mid_turn is False
    finally:
        session.process.kill()


def test_waiting_forever_is_still_available(ut):
    """speak_after=0 keeps the old behaviour, for anyone who wants it."""
    session = spawn(ut, "print('working')\n" + PROMPTS)
    try:
        reply = session.read_until_turn(idle_s=30.0, hard_cap_s=20.0, speak_after=0.0)
        assert reply.by_prompt is True
    finally:
        session.process.kill()


def test_the_persona_is_told_whether_it_was_answered_or_is_interrupting(ut):
    """A persona that thinks it was answered writes as though it was, and asks about
    a number nobody gave it."""
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    from types import SimpleNamespace

    client = SimpleNamespace(messages=FakeMessages())
    persona = ut.ALL["controller"]

    ut.next_user_message(client, "m", persona, [("go", "meshing now", True)], "goal")
    assert "still working" in captured["messages"][2]["content"]

    ut.next_user_message(client, "m", persona, [("go", "it is 22 Pa", False)], "goal")
    assert "consultant replied" in captured["messages"][2]["content"]


def test_a_fresh_run_actually_clears_the_workspace(ut, monkeypatch):
    """The flag existed, was documented, appeared in --help, and was never read. Four
    persona runs shared a workspace because of it, and one of them opened by finding
    another study's cases and having to verify them before it could start."""
    from types import SimpleNamespace

    cleared = []
    monkeypatch.setattr(ut, "clear_workspace", lambda: cleared.append(True) or "cleared")
    monkeypatch.setattr(ut, "AgentSession", _NoSession)
    monkeypatch.setattr(ut, "quieten", lambda repo, study: "nothing to stop")

    args = SimpleNamespace(
        fresh=True, goal=None, log=None, model=None, turns=0, idle=1.0, cap=1.0,
        speak_after=0.0, budget=0.0, user_model="m",
    )
    ut.run_one(None, ut.ALL["engineer"], args, Path("."))

    assert cleared, "the workspace was cleared before the persona started"


def test_no_fresh_leaves_the_workspace_alone(ut, monkeypatch):
    from types import SimpleNamespace

    cleared = []
    monkeypatch.setattr(ut, "clear_workspace", lambda: cleared.append(True) or "cleared")
    monkeypatch.setattr(ut, "AgentSession", _NoSession)
    monkeypatch.setattr(ut, "quieten", lambda repo, study: "nothing to stop")

    args = SimpleNamespace(
        fresh=False, goal=None, log=None, model=None, turns=0, idle=1.0, cap=1.0,
        speak_after=0.0, budget=0.0, user_model="m",
    )
    ut.run_one(None, ut.ALL["engineer"], args, Path("."))

    assert not cleared


class _NoSession:
    """An agent that starts, says nothing, and closes."""

    def __init__(self, argv, cwd):
        self.process = SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0)

    def read_until_turn(self, *a, **k):
        import user_test_under_test as module

        return module.Reply("study 20260824-000000-abcd on instance x\n> ", by_prompt=True)

    def send(self, text):
        raise AssertionError("no turns were requested")

    def close(self, timeout=60.0):
        return 0


def test_every_persona_goal_names_a_geometry_and_asks_for_one_thing(ut):
    """A goal has to be answerable inside the run's budget. The first controller goal
    asked how the air divides between two branches of a symmetric wye -- which is set
    by what breaks the symmetry, none of it in the geometry -- and it burned
    twenty-eight minutes being right about that instead of producing a number."""
    for persona in ut.ALL.values():
        goal = persona.goal.lower()
        assert "mm" in goal, f"{persona.name} does not say how big anything is"
        wants = ("pressure drop", "pressure we lose", "how much pressure", "problem")
        assert any(w in goal for w in wants), f"{persona.name} does not ask for a number"
