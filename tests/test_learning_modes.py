"""Learn mode and beginner mode: a note the person chose, and nothing else.

These are the roadmap items most likely to be built the wrong way. A teaching mode that
injects a checklist into the prompt, demands an explanation before each tool call or
holds the solve until the person has answered a question is the workflow injection
`docs/design.md` section 1 forbids, and `tests/test_prompt.py` and
`tests/test_briefing.py` already reject it in the two places it would appear. What is
left to hold is the thing those two cannot see: that the two new styles went in as text
in the person's voice and changed nothing else -- not the gated tools, not the modes,
not the system prompt, not the billing vocabulary.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from openreynolds import ceiling, cli, modes, study_style
from openreynolds.backend.base import BackendError
from openreynolds.browse import Browser
from openreynolds.config import Config
from openreynolds.prompt import SYSTEM_PROMPT
from openreynolds.study_style import STYLES, TEACHING, normalize_style, preferences_for
from test_briefing import a_workspace, brief_for


def test_the_two_teaching_styles_are_the_ones_the_roadmap_asked_for():
    assert TEACHING == ("learn", "beginner")
    for name in TEACHING:
        assert name in STYLES
        assert normalize_style(name) == name


def test_a_teaching_style_keeps_quick_s_scope():
    """A run somebody is reading along with is a run they want to see the whole of, so
    both are built on `quick` rather than `full`: one coarse mesh and the headline
    numbers, with the telling added on top."""
    for name in TEACHING:
        assert STYLES[name].startswith(study_style.QUICK)
        assert "one coarse mesh" in STYLES[name]
        assert STYLES[name].endswith(study_style.ALWAYS)


def test_learn_mode_asks_for_the_reasoning_and_not_for_a_pause():
    """The measured failure this sentence is for: a full-auto session opened with a
    numbered list of questions and then meshed for seven minutes while the list sat
    there (study 20260920-161908-c7ef, quoted in `modes.AUTO_NO_QUESTIONS`). A note
    asking to be taught reads as an invitation to stop and check understanding."""
    text = STYLES["learn"]
    assert "learn CFD" in text
    assert "keep working" in text, "a teaching note that waits for a reply ends the work"


def test_beginner_mode_asks_for_plain_language_and_the_limits_of_the_answer():
    text = STYLES["beginner"]
    assert "plain language" in text
    assert "y+" in text, "the terms worth spelling out are named rather than implied"
    assert "does not support" in text
    assert "do not need to approve" in text


def test_neither_teaching_style_gates_a_tool_call():
    """`modes.py` stays the only thing in the repository that holds a tool call, and
    it holds `job_start` and the `checkpoint` question. A style is text."""
    assert modes.COMPUTE_TOOLS == frozenset({"job_start"})
    assert modes.MODES == ("auto", "partial", "structured")
    for name in TEACHING:
        assert modes.normalize(name) is None, f"{name} is a style, not a mode"
        assert modes.gated("auto", "job_start") is False


def test_the_system_prompt_says_nothing_about_a_style():
    """It sits at the front of the cached prefix and must not vary per session
    (`test_prompt.py`); a teaching mode baked in there would be the injection."""
    for name in TEACHING:
        assert name not in SYSTEM_PROMPT.lower()
    assert "beginner" not in SYSTEM_PROMPT.lower()


def test_a_teaching_style_reaches_the_model_as_the_persons_own_words(backend, store):
    a_workspace(backend)
    store.session.home = "/work/mine"

    brief = brief_for(backend, store, preferences=preferences_for("learn"))

    assert "In their own words:" in brief
    assert "I am using this study to learn CFD" in brief
    # In the person's voice and the harness's alongside it, with nothing added between.
    assert "the harness" not in brief.split("In their own words:")[1]


def test_the_persons_own_note_still_goes_after_the_style():
    text = preferences_for("beginner", "I care about the pressure drop.")
    assert text.index("plain language") < text.index("I care about the pressure drop.")


def test_the_two_new_styles_are_not_a_new_billing_vocabulary():
    """The service sets a study's ceiling from a style of its own and reports it back
    (`backend/base.py:267` reads `extra["style"]`); `ceiling.style_of` never sees the
    depth chosen here. So the tables stay at the service's two words, and a learn study
    is billed by whatever the service called it rather than by a third name this
    package invented."""
    assert sorted(ceiling.NORMAL_CENTS) == ["full", "quick"]
    assert sorted(ceiling.DEFAULT_CAP_CENTS) == ["full", "quick"]


# -- the terminal can choose one -------------------------------------------------


def _no_instance(monkeypatch, tmp_path, cfg=None):
    """A config whose workspace reservation fails: the run gets past the options and
    dies at the (absent) service, which is where `cfg` can be read back."""
    def reserve(url, key, iid, **_kw):
        raise BackendError("no service in this test")

    monkeypatch.setattr(cli.hosted, "reserve", reserve)
    config = cfg or Config(foamd_url="u", foamd_api_key="k", llm_api_key="a",
                           studies_dir=tmp_path / "studies")
    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls: config))
    return config


@pytest.mark.parametrize("depth", ["quick", "full", "learn", "beginner"])
def test_the_depth_flag_opens_the_session_with_that_standing_note(depth, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    cfg = _no_instance(monkeypatch, tmp_path)

    CliRunner().invoke(cli.main, ["-p", "go", "--solver", "openfoam", "--depth", depth])

    assert cfg.preferences == STYLES[depth]


def test_the_depth_flag_puts_the_style_ahead_of_the_persons_own_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    cfg = _no_instance(monkeypatch, tmp_path, Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a",
        studies_dir=tmp_path / "studies", preferences="Metric units, please."))

    CliRunner().invoke(cli.main, ["-p", "go", "--solver", "openfoam", "--depth", "learn"])

    assert cfg.preferences.startswith(STYLES["learn"])
    assert cfg.preferences.endswith("Metric units, please.")


def test_no_depth_flag_leaves_the_standing_note_exactly_as_it_was(monkeypatch, tmp_path):
    """A session started without the flag carries the briefing it always did."""
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    cfg = _no_instance(monkeypatch, tmp_path, Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a",
        studies_dir=tmp_path / "studies", preferences="Metric units, please."))

    CliRunner().invoke(cli.main, ["-p", "go", "--solver", "openfoam"])

    assert cfg.preferences == "Metric units, please."


def test_a_depth_that_is_not_one_is_refused_by_name(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    _no_instance(monkeypatch, tmp_path)

    result = CliRunner().invoke(cli.main, ["-p", "go", "--depth", "expert"])

    assert result.exit_code == 2
    assert "learn" in result.output and "beginner" in result.output


def test_the_depth_is_not_written_into_the_config_file(monkeypatch, tmp_path):
    """It is one session's note, not a stored setting: the file records credentials and
    the mode, and `preferences.md` is the person's own file."""
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "config.json"))
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="a",
                 preferences=STYLES["learn"])
    assert "preferences" not in json.loads(cfg.save().read_text())


def test_the_mcp_servers_offer_the_same_four_words():
    """`depth` is one vocabulary across the faces over the `Host` protocol, so the MCP
    description and `STYLES` cannot drift apart on what a caller may ask for."""
    from openreynolds.mcp.core import DESCRIPTIONS

    text = DESCRIPTIONS["start_study"]
    for name in STYLES:
        assert f'"{name}"' in text, f"start_study does not offer depth {name}"


def test_a_browser_is_not_needed_for_any_of_this(backend, store):
    """Guards the shape of the change: a style is relayed text, so nothing about it
    needs the workspace. If this ever needs a browser, something has become a step."""
    a_workspace(backend)
    store.session.home = "/work/mine"
    brief = cli._situation_brief(store, backend, resuming=False, interactive=True,
                                 browser=Browser(backend, store),
                                 preferences=preferences_for("beginner"))
    assert "plain language" in brief
