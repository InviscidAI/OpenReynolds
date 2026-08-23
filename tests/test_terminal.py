"""Terminal output must not be able to kill the session.

A CFD conversation is full of Greek letters, superscripts and arrows, and stdout is not
always UTF-8: a redirect to a file on Windows lands on cp1252, where a single mu raises
mid-stream. Two defences, both tested here -- our own output stays ASCII, and the
streams are made tolerant so the model's output cannot raise either.
"""

from __future__ import annotations

import io
import pathlib
import sys

import pytest
from rich.console import Console

from openreynolds import cli

PACKAGE = pathlib.Path(cli.__file__).parent


def console_lines():
    for path in sorted(PACKAGE.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "console.print" in line:
                yield path.name, number, line


def test_our_own_console_output_is_ascii():
    offenders = [
        f"{name}:{number}: {line.strip()}"
        for name, number, line in console_lines()
        if any(ord(char) > 127 for char in line)
    ]
    assert not offenders, "non-ASCII on a console path:\n" + "\n".join(offenders)


def test_streams_are_made_tolerant_of_undecodable_output():
    class Recorder:
        def __init__(self):
            self.errors = None

        def reconfigure(self, **kwargs):
            self.errors = kwargs.get("errors")

    out, err = Recorder(), Recorder()
    original = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        cli._tolerant_stdout()
    finally:
        sys.stdout, sys.stderr = original

    assert out.errors == "replace"
    assert err.errors == "replace"


def test_a_stream_that_cannot_be_reconfigured_is_left_alone():
    """Under pytest's capture, and on some platforms, reconfigure is unavailable."""
    original = sys.stdout
    sys.stdout = io.StringIO()  # no reconfigure attribute
    try:
        cli._tolerant_stdout()  # must not raise
    finally:
        sys.stdout = original


@pytest.mark.parametrize("text", ["mu = 1.8e-5 Pa.s", "y+ ~ 30", "dp = 29.2 Pa"])
def test_ascii_physics_still_reads_well(text):
    """The ASCII rule applies to our strings, not the model's -- but ours should
    still say what they mean."""
    buffer = io.StringIO()
    Console(file=buffer, force_terminal=False).print(text)
    assert text.split()[0] in buffer.getvalue()


def test_the_model_may_still_emit_anything():
    """Nothing in the harness restricts what the model prints; the streams absorb it."""
    buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="replace")
    Console(file=buffer, force_terminal=False).print("mu = 1.8x10^-5, y+ = 30, dp -> 29.2")
    buffer.flush()
