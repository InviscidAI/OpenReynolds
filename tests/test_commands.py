"""What the user types, and what it does. `/status` must never reach the model."""

from __future__ import annotations

import pytest

from openreynolds import commands
from openreynolds.commands import ASIDE, EXIT, FILES, HELP, OPEN, SAY, STATUS, parse


@pytest.mark.parametrize(
    "line,kind",
    [
        ("run the coarse case", SAY),
        ("/btw is it converging?", ASIDE),
        ("/BTW is it converging?", ASIDE),
        ("/bytheway hello", ASIDE),
        ("/btw", STATUS),
        ("/status", STATUS),
        ("/files", FILES),
        ("/ls /work/case", FILES),
        ("/open", OPEN),
        ("/help", HELP),
        ("/exit", EXIT),
        ("/quit", EXIT),
        ("/mode partial", commands.MODE),
        ("/model claude-sonnet-5", commands.MODEL),
        ("/effort low", commands.EFFORT),
        ("/yes", commands.YES),
        ("/approve", commands.YES),
        ("/y", commands.YES),
        ("/no too many cells", commands.NO),
        ("/deny", commands.NO),
        ("/n", commands.NO),
        ("/all", commands.ALL),
        ("/yes all", commands.ALL),
        ("/help modes", HELP),
    ],
)
def test_lines_are_classified(line, kind):
    assert parse(line).kind == kind


@pytest.mark.parametrize(
    "line",
    [
        "/work/case/log.simpleFoam looks wrong",
        "/dev/null",
        "/ is a path separator, not a command",
    ],
)
def test_something_that_merely_starts_with_a_slash_is_a_message(line):
    """Guessing 'unknown command' at someone who meant to say a path is worse than
    passing it along."""
    assert parse(line).kind == SAY
    assert parse(line).text == line


def test_an_aside_keeps_what_was_said():
    command = parse("/btw the inlet velocity looks like mm/s")
    assert "the inlet velocity looks like mm/s" in command.text


def test_an_aside_is_framed_as_the_user_not_the_harness():
    """The user typed `/btw`; this says what they meant by it. It is not an
    instruction about how to work."""
    text = parse("/btw carry on").text
    assert text.startswith("By the way")
    assert "harness" not in text.lower()


def test_files_carries_the_path_asked_for():
    assert parse("/files /work/case").text == "/work/case"
    assert parse("/files").text == ""


def test_status_reports_running_jobs(store):
    store.record_job("job-1", cmd="simpleFoam -parallel", name="solve")
    lines = commands.status_lines(store, stage="bash: blockMesh", tokens=12_345)
    joined = "\n".join(lines)

    assert "1 job(s) running" in joined
    assert "solve" in joined
    assert "simpleFoam -parallel" in joined
    assert "bash: blockMesh" in joined
    assert "12,345" in joined


def test_status_says_so_when_nothing_is_running(store):
    assert "no jobs started yet" in "\n".join(commands.status_lines(store))


def test_status_names_the_last_job_once_it_has_finished(store):
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    store.update_job("job-1", status="exited", end_reason="completed")
    joined = "\n".join(commands.status_lines(store))
    assert "no jobs running" in joined and "completed" in joined


def test_the_help_lists_every_verb_it_accepts():
    """A command nobody can discover is a command nobody uses."""
    for verb in ("/btw", "/status", "/files", "/open", "/help", "/exit",
                 "/mode", "/model", "/effort", "/yes", "/no", "/all"):
        assert verb in commands.HELP_TEXT


def test_arguments_are_carried():
    assert parse("/no the mesh is too coarse").text == "the mesh is too coarse"
    assert parse("/mode Structured").text == "Structured"
    assert parse("/help keys").text == "keys"


def test_every_verb_in_the_registry_parses_to_its_kind():
    for spec in commands.COMMANDS:
        for name in (spec.verb, *spec.aliases):
            expected = STATUS if spec.kind == ASIDE else spec.kind  # bare /btw asks
            assert parse(name).kind == expected, name


def test_renders_verbs_parse():
    from openreynolds import commands

    for verb in ("/renders", "/pics", "/images"):
        assert commands.parse(verb).kind == commands.RENDERS


# -- files handed over with @ --------------------------------------------------


def test_an_at_path_is_handed_over_and_the_sigil_leaves_the_sentence():
    command = commands.parse("/mesh mesh the air inside @/work/uploads/plan.png")
    assert command.kind == commands.MESH
    assert command.inputs == ("/work/uploads/plan.png",)
    assert command.text == "mesh the air inside /work/uploads/plan.png"


def test_several_files_come_back_in_the_order_they_were_typed():
    command = commands.parse(
        "/mesh trace @/work/a/plan.png against @/work/a/rooms.csv and @/work/a/part.step")
    assert command.inputs == ("/work/a/plan.png", "/work/a/rooms.csv", "/work/a/part.step")
    assert "@" not in command.text


def test_a_file_named_twice_is_handed_over_once():
    command = commands.parse("/mesh compare @/work/a.step against @/work/a.step")
    assert command.inputs == ("/work/a.step",)


def test_a_path_with_a_space_in_it_can_be_quoted():
    """`/work/study/uploads/chassis v2.step` is already a fixture elsewhere in here."""
    command = commands.parse('/mesh prepare @"/work/study/uploads/chassis v2.step"')
    assert command.inputs == ("/work/study/uploads/chassis v2.step",)
    assert command.text == "prepare /work/study/uploads/chassis v2.step"


def test_a_bare_path_is_prose_and_is_not_handed_over():
    """The whole reason for the sigil: naming a file is not offering it."""
    command = commands.parse("/mesh build it like the duct in /work/old/duct.step")
    assert command.inputs == ()
    assert command.text == "build it like the duct in /work/old/duct.step"


def test_an_address_is_not_a_handover():
    """`@` only counts where it starts a token, or every email is a file."""
    command = commands.parse("tell ziming@inviscidai.com it is done")
    assert command.inputs == ()
    assert command.text == "tell ziming@inviscidai.com it is done"


def test_sentence_punctuation_is_not_part_of_the_filename():
    command = commands.parse("/mesh mesh @/work/plan.png, 2.7 m ceilings.")
    assert command.inputs == ("/work/plan.png",)
    assert command.text == "mesh /work/plan.png, 2.7 m ceilings."


def test_a_handover_works_in_an_ordinary_message_too():
    """The sigil is the session's, not one command's, so it means the same everywhere."""
    command = commands.parse("have a look at @/work/uploads/plan.png")
    assert command.kind == commands.SAY
    assert command.inputs == ("/work/uploads/plan.png",)
    assert command.text == "have a look at /work/uploads/plan.png"


def test_a_path_argument_command_is_left_alone():
    """`/files` takes a path as its whole argument; there is no sentence to mark up."""
    command = commands.parse("/files /work/uploads")
    assert command.text == "/work/uploads"
    assert command.inputs == ()


def test_mesh_with_nothing_after_it_is_still_a_mesh_command():
    """So the caller can say what it needs rather than the parser guessing."""
    command = commands.parse("/mesh")
    assert command.kind == commands.MESH
    assert command.text == ""
