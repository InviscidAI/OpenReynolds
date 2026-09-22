"""Completion: what a half-typed command becomes, in the terminal and on the web."""

from __future__ import annotations

from openreynolds import modes
from openreynolds.commands import completions


def lines(text, **kw):
    return [line for line, _ in completions(text, **kw)]


def test_a_slash_offers_every_command_in_registry_order():
    got = lines("/")
    assert got[0] == "/mesh "  # the registry leads with it; the rest follow in order
    assert got[1] == "/btw "
    assert "/status" in got and "/help " in got and "/exit" in got
    assert "/quit" not in got and "/y" not in got  # aliases only when typed exactly


def test_a_prefix_narrows_and_a_verb_with_arguments_ends_in_a_space():
    assert lines("/mo") == ["/mode ", "/model "]
    assert lines("/sta") == ["/status"]
    assert lines("/MO") == ["/mode ", "/model "]


def test_an_alias_typed_in_full_is_offered():
    assert "/quit" in lines("/quit")
    assert lines("/y") == ["/yes", "/y"]


def test_nothing_for_a_message_or_an_unknown_verb():
    assert completions("hello") == []
    assert completions("/nonsense ") == []
    assert completions("/work/case") == []


def test_mode_offers_the_modes_with_labels():
    got = completions("/mode ")
    assert [line for line, _ in got] == [f"/mode {m}" for m in modes.MODES]
    assert modes.LABELS["partial"] in got[1][1]
    assert lines("/mode st") == ["/mode structured"]


def test_model_offers_what_the_caller_knows():
    got = completions("/model ", models=[("claude-opus-5", "deepest"), "claude-sonnet-5"])
    assert got == [("/model claude-opus-5", "deepest"), ("/model claude-sonnet-5", "")]
    assert lines("/model claude-s", models=["claude-opus-5", "claude-sonnet-5"]) == [
        "/model claude-sonnet-5"
    ]


def test_effort_and_help_topics():
    assert lines("/effort ") == ["/effort low", "/effort medium", "/effort high"]
    assert lines("/effort ", efforts=["high"]) == ["/effort high"]
    assert "/help modes" in lines("/help ")
    assert lines("/help k") == ["/help keys"]
    assert lines("/? t") == ["/? tools"]


def test_terminal_only_commands_are_not_offered_on_the_web():
    assert "/open" in lines("/o")
    assert "/open" not in lines("/o", where="web")


def test_no_suggestions_once_the_argument_is_finished():
    assert completions("/mode partial now") == []


def test_results_are_deduplicated():
    got = lines("/model ", models=["a", "a", "A"])
    assert got == ["/model a"]
