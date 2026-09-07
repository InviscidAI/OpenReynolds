"""The two axes, and the words that name them.

The whole point of naming these rather than leaving them to `preferences.md` is that a
named thing can be checked, reported and changed. So the tests are about the menu
holding its shape -- distinct names, no imperative prose, a safe answer for a word
nobody recognises -- rather than about what a level makes the model do, which is the
model's business and always was.
"""

from __future__ import annotations

import re

import pytest

from openreynolds import levels
from test_prompt import IMPERATIVE_PATTERNS


def test_the_two_axes_have_three_levels_each():
    """Two is a switch and the middle position is what most studies want; a longer
    list is a menu nobody reads."""
    assert len(levels.AMBITION) == 3
    assert len(levels.CONSENT) == 3


def test_no_level_name_belongs_to_both_axes():
    """`/level thorough never` routes each word by name alone. A name on both axes
    would make that ambiguous, and the ambiguity would be silent."""
    assert not set(levels.AMBITION) & set(levels.CONSENT)


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
@pytest.mark.parametrize(
    "text", [*levels.AMBITION.values(), *levels.CONSENT.values()], ids=lambda t: t[:24]
)
def test_no_level_tells_the_model_what_to_do(pattern, text):
    """Every one of these is relayed into the briefing, so it lives under the same
    rule as the briefing does: it says what the person wants, not what to do."""
    match = re.search(pattern, text, re.IGNORECASE)
    assert match is None, f"imperative language in a level: {match!r}"


def test_the_defaults_are_on_the_menu():
    assert levels.DEFAULT_AMBITION in levels.AMBITION
    assert levels.DEFAULT_CONSENT in levels.CONSENT


# -- reading a value nobody checked --------------------------------------------


@pytest.mark.parametrize("value", ["thorogh", "", "   ", "yes", "high"])
def test_an_unrecognised_value_becomes_the_default(value):
    assert levels.normalise(value, "ambition") == levels.DEFAULT_AMBITION
    assert levels.normalise(value, "consent") == levels.DEFAULT_CONSENT


@pytest.mark.parametrize("value", ["Thorough", " THOROUGH ", "thorough"])
def test_a_level_is_recognised_however_it_was_typed(value):
    assert levels.normalise(value, "ambition") == "thorough"


def test_axis_of_names_the_axis_or_nothing():
    assert levels.axis_of("sketch") == "ambition"
    assert levels.axis_of("never") == "consent"
    assert levels.axis_of("sideways") is None


# -- what `/level <words>` means -----------------------------------------------


def test_one_word_changes_one_axis():
    assert levels.chosen("thorough") == ("thorough", "", [])
    assert levels.chosen("never") == ("", "never", [])


def test_two_words_change_both_in_either_order():
    assert levels.chosen("thorough never") == ("thorough", "never", [])
    assert levels.chosen("never thorough") == ("thorough", "never", [])


def test_nothing_typed_changes_nothing():
    assert levels.chosen("") == ("", "", [])


def test_a_word_nobody_recognises_changes_neither_axis():
    """`/level thorogh never` half-applied would set consent, drop the word they
    cared about, and say nothing about it. Both stay put and the menu comes back."""
    assert levels.chosen("thorogh never") == ("", "", ["thorogh"])


# -- the prose each of them produces -------------------------------------------


def test_the_briefing_names_both_levels_and_says_what_they_mean():
    lines = levels.briefing_lines("sketch", "never")

    assert "Ambition, `sketch`:" in lines[1]
    assert levels.AMBITION["sketch"] in lines[1]
    assert "Consent, `never`:" in lines[2]
    assert levels.CONSENT["never"] in lines[2]


def test_the_briefing_lines_survive_a_value_from_a_stale_config():
    lines = "\n".join(levels.briefing_lines("aggressive", "sometimes"))

    assert "aggressive" not in lines and "sometimes" not in lines
    assert "`standard`" in lines and "`costly`" in lines


def test_a_mid_study_change_is_a_sentence_in_the_first_person():
    """It reaches the model the way anything else the user types does, because that
    is what it is: they typed it."""
    said = levels.spoken(ambition="thorough")

    assert said.startswith("Changing what I picked")
    assert "`thorough`" in said
    assert levels.AMBITION["thorough"] in said


def test_a_change_to_one_axis_says_nothing_about_the_other():
    said = levels.spoken(consent="never").lower()

    assert "consent" in said
    assert "ambition" not in said


def test_the_menu_shows_where_the_session_currently_stands():
    lines = "\n".join(levels.menu_lines("sketch", "early"))

    assert "ambition is sketch, consent is early" in lines
    for name in (*levels.AMBITION, *levels.CONSENT):
        assert name in lines


def test_capitalising_a_line_leaves_the_level_name_alone():
    """`str.capitalize()` lowercases the rest of the string, which would rewrite the
    level names inside the sentence into words nobody typed."""
    line = levels.briefing_lines("thorough", "never")[1]

    assert "`thorough`" in line
    assert line.startswith("Ambition")


def test_two_words_on_one_axis_apply_neither():
    """`/level standard thorough` is somebody correcting themselves without clearing
    the line. Taking the second quietly would show them a confirmation naming one word
    out of the two they typed, which is the half-application this refuses everywhere
    else."""
    assert levels.chosen("standard thorough") == ("", "", ["thorough"])
    assert levels.chosen("never early") == ("", "", ["early"])


def test_one_word_per_axis_is_still_fine():
    assert levels.chosen("sketch never") == ("sketch", "never", [])
