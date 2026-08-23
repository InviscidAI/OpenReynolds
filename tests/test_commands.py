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
    for verb in ("/btw", "/status", "/files", "/open", "/help", "/exit"):
        assert verb in commands.HELP_TEXT
