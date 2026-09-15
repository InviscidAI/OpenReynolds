"""`/help`: every command, every mode and every tool is explained somewhere a person can find it."""

from __future__ import annotations

from openreynolds import commands, modes
from openreynolds.tools import TOOLS


def joined(topic: str = "", where: str = "terminal") -> str:
    return "\n".join(commands.help_lines(topic, where))


def test_the_overview_lists_every_command_with_its_arguments():
    text = joined()
    for spec in commands.visible("terminal"):
        assert f"{spec.verb} {spec.args}".strip() in text
        assert spec.summary in text


def test_the_overview_names_the_modes_the_topics_and_completion():
    text = joined()
    for mode in modes.MODES:
        assert mode in text and modes.LABELS[mode] in text
    for name, _ in commands.TOPICS:
        assert f"/help {name}" in text
    assert "Tab" in text


def test_the_web_overview_leaves_out_terminal_only_commands_and_explains_its_own_list():
    text = joined(where="web")
    assert "/open" not in text
    assert "above the message box" in text
    assert "/open" in joined(where="terminal")


def test_the_commands_topic_carries_each_detail_and_alias():
    text = joined("commands")
    for spec in commands.visible("terminal"):
        assert spec.detail in text
        for alias in spec.aliases:
            assert alias in text


def test_the_modes_topic_says_what_each_gates_and_how_to_switch():
    text = joined("modes")
    for mode in modes.MODES:
        assert modes.DESCRIPTIONS[mode] in text
    assert "job_start" in text and "mesh" in text and "checkpoint" in text
    assert "/mode <name>" in text and "--mode" in text and "OPENREYNOLDS_MODE" in text
    for stage in modes.STAGES:
        assert stage in text


def test_the_model_topic_covers_the_spec_forms_timing_and_cost():
    text = joined("model")
    assert "/model <provider>:<model>" in text
    assert "/effort" in text
    assert "turn ends" in text
    assert "cache" in text


def test_every_tool_is_explained_in_plain_words():
    text = joined("tools")
    for name in [tool["name"] for tool in TOOLS] + ["checkpoint"]:
        assert f"  {name}" in text, name


def test_the_keys_topic_covers_completion_and_the_bindings():
    text = joined("keys")
    for key in ("Tab", "Up", "Down", "Esc", "Right", "ctrl+t", "ctrl+f", "ctrl+g",
                "ctrl+r", "ctrl+l", "ctrl+c"):
        assert key in text, key
    assert "ctrl+t" not in joined("keys", "web")


def test_the_plain_terminal_does_not_promise_the_interfaces_completion_or_keys():
    """`--plain` reads whole lines: no Tab, no suggestion list, no ctrl bindings."""
    from openreynolds.view import ConsoleView

    assert ConsoleView.surface == commands.PLAIN
    text = joined(where="plain")
    assert "Tab" not in text and "no completion" in text
    assert "/open" in text, "the plain terminal still has the terminal's commands"
    keys = joined("keys", "plain")
    assert "ctrl+t" not in keys and "Up, Down" not in keys and "ctrl+c" in keys


def test_the_no_detail_names_what_does_not_decline():
    detail = next(spec.detail for spec in commands.COMMANDS if spec.verb == "/no")
    for word in ("ok", "/status", "/help", "/mode"):
        assert word in detail, word


def test_an_unknown_topic_says_so_and_still_helps():
    lines = commands.help_lines("nonsense")
    assert "nonsense" in lines[0]
    for name, _ in commands.TOPICS:
        assert name in lines[0]
    assert "\n".join(commands.help_lines()) in "\n".join(lines)


def test_help_text_is_the_overview():
    assert commands.HELP_TEXT == joined()


def test_as_json_is_what_the_web_composer_draws_from():
    data = commands.as_json("web")
    verbs = [row["verb"] for row in data["commands"]]
    assert "/open" not in verbs and "/mode" in verbs and "/help" in verbs
    row = next(r for r in data["commands"] if r["verb"] == "/mode")
    assert row["choices"] == list(modes.MODES)
    assert set(row) == {"verb", "aliases", "args", "summary", "detail", "choices"}
    assert [m["value"] for m in data["modes"]] == list(modes.MODES)
    assert data["modes"][1]["label"] == "Ask before compute"
    assert data["efforts"] == ["low", "medium", "high"]
    assert [t["name"] for t in data["topics"]] == [name for name, _ in commands.TOPICS]


def test_help_answers_by_topic_and_surface(store, view):
    from openreynolds.cli import _local

    view.surface = "web"
    _local(commands.parse("/help keys"), view, browser=None, store=store)
    assert view.statuses[-1] == commands.help_lines("keys", "web")


def test_the_plain_terminal_prints_the_argument_forms():
    """`[path]` and `[name]` are rich markup tags unless escaped, and vanished."""
    import io

    from rich.console import Console

    from openreynolds.view import ConsoleView

    out = io.StringIO()
    ConsoleView(Console(file=out, force_terminal=False, width=200)).status(commands.help_lines())
    printed = out.getvalue()
    for spec in commands.visible("terminal"):
        assert f"{spec.verb} {spec.args}".strip() in printed, spec.verb
