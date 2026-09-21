"""The words for a solve whose residuals stopped falling.

Five studies in three days told the person "did not converge" about runs that had not
failed (`openreynolds/convergence.py` has each one, with its numbers). These tests pin
the guidance that draws the line -- a stall on an unsteady flow or a plateau fit for a
first look is not a failure; a climb, an exception, a bounded field or a rejected mesh
is -- and keep it in the harness's voice rather than a checklist's.
"""

from __future__ import annotations

import re

import pytest

from openreynolds import convergence
from test_prompt import IMPERATIVE_PATTERNS


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_the_guidance_is_not_a_workflow(pattern):
    for text in (convergence.SOLVE_NOTE, convergence.STEADY_LAUNCH_NOTE):
        assert re.search(pattern, text, re.IGNORECASE) is None, f"imperative language: {pattern!r}"


def test_the_failure_words_appear_once_each_as_the_words_ruled_out():
    """"could not converge", "did not converge", "failed to converge" are what the
    person heard four times in five studies. The note names each exactly once, in the
    clause that says they do not describe a stall or a plateau, and nowhere else."""
    note = convergence.SOLVE_NOTE
    for words in convergence.FAILURE_WORDS:
        assert note.count(words) == 1, words
    ruled_out = note[note.index("the words for one"):note.index("do not describe them")]
    for words in convergence.FAILURE_WORDS:
        assert words in ruled_out, f"{words!r} is used somewhere other than as the words ruled out"
    assert "not converge" not in convergence.STEADY_LAUNCH_NOTE


def test_the_line_is_drawn_on_physics_not_on_a_tolerance():
    """A steady solver stalling on an unsteady flow is the flow, not the numerics; a
    plateau at the 1e-4..1e-3 level is fit for a first look; the failures are the
    things that cannot work. No residual number is set as the bar."""
    note = convergence.SOLVE_NOTE
    assert "steady solver on a flow that is unsteady" in note
    assert "reporting the flow rather than the numerics" in note
    assert "1e-4..1e-3" in note and "fit to show" in note
    for failure in ("residual that climbs", "floating point exception", "keeps bounding", "checkMesh"):
        assert failure in note
    assert "what failed and what would fix it" in note
    assert "tolerance" not in note and "residualControl" not in note


def test_the_note_says_what_the_report_leads_with():
    note = convergence.SOLVE_NOTE
    assert "what the flow is doing" in note
    assert "in one clause and in neutral terms" in note


def test_each_kind_has_an_example_sentence_with_its_number_in_it():
    """Two ways of saying a stall, one of saying a failure, each with the number in
    the sentence so the person learns what the flow did and the residual is a clause."""
    note = convergence.SOLVE_NOTE
    stalls = [
        "levelled off at 3e-3 from iteration 800 and this is the field at iteration 2000",
        "sat at 2e-4 from iteration 1500 on, which is fine for this look",
    ]
    failure = "diverged at iteration 37: Uy climbed from 1e-3 to 40"
    for example in (*stalls, failure):
        assert example in note
    assert "remeshing that corner is the fix" in note, "a failure sentence names the fix"
    assert note.index(stalls[0]) < note.index(stalls[1]) < note.index(failure)


def test_a_first_look_is_told_what_it_is_and_what_it_is_not():
    """"not a mesh independence study" -- the owner's words. A transient window or a
    mesh study is offered afterwards, not started because the residuals stalled."""
    note = convergence.SOLVE_NOTE
    assert "one mesh, one run and the picture" in note
    assert "a next step to offer, not to start unasked" in note


def test_the_steady_launch_note_names_the_shape_and_its_worth():
    note = convergence.STEADY_LAUNCH_NOTE
    assert note.startswith("steady solver:")
    assert "level off rather than fall" in note
    assert "snapshot worth showing, not a failed run" in note
    assert len(note) < 250, "one clause on a launch line, not a paragraph"


@pytest.mark.parametrize("executable,steady", [
    ("simpleFoam", True), ("rhoSimpleFoam", True), ("buoyantSimpleFoam", True),
    ("SRFSimpleFoam", True), ("porousSimpleFoam", True),
    ("pimpleFoam", False), ("pisoFoam", False), ("icoFoam", False), ("interFoam", False),
    ("rhoPimpleFoam", False), ("foamRun", False), ("", False),
])
def test_a_steady_solver_is_told_by_its_name(executable, steady):
    assert convergence.is_steady_solver(executable) is steady


def test_the_note_stays_a_paragraph():
    """It rides in every briefing. Past a screenful it stops being read."""
    assert len(convergence.SOLVE_NOTE) < 1700
