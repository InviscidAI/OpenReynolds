"""The other place workflow could leak in.

`tests/test_prompt.py` guards the system prompt, which is frozen and reviewed. The
briefing is neither: it is assembled fresh every session, it has grown every time
something turned out to be worth saying, and it is the natural place for "and then you
should..." to appear one day without anyone noticing.

It carries facts -- which directory is yours, whether anyone is at the terminal, what is
still running. Every one of those is something the model cannot find out for itself and
would otherwise get wrong. None of them tells it what to do about any of it, and these
tests are what keeps that true.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import ScriptedReader  # noqa: F401  (keeps conftest importable)
from openreynolds import cli
from openreynolds.backend.base import ExecResult, JobStatus
from openreynolds.browse import Browser
from openreynolds import levels
from openreynolds.store import JobRecord
from test_prompt import IMPERATIVE_PATTERNS


def brief_for(backend, store, **kwargs) -> str:
    defaults = dict(resuming=False, interactive=True, browser=Browser(backend, store))
    defaults.update(kwargs)
    return cli._situation_brief(store, backend, **defaults)


def a_workspace(backend, *paths):
    lines = "".join(f"d\t4096\t1700000000.0\t{path}\n" for path in paths)
    backend.exec_result = ExecResult(0, lines, False, None)


def every_shape_of_briefing(backend, store):
    """The briefing is assembled from parts; each combination is a thing a user sees."""
    store.session.home = "/work/20260824-120000-abcd"
    store.session.instance_id = "inst-1"

    a_workspace(backend)
    yield "new, empty", brief_for(backend, store)
    yield "new, empty, nobody watching", brief_for(backend, store, interactive=False)

    a_workspace(backend, "/work/20260824-120000-abcd/elbow")
    yield "new, inherited files", brief_for(backend, store)
    yield "resumed", brief_for(backend, store, resuming=True)

    store.session.jobs["job-1"] = JobRecord(
        job_id="job-1", name="solve", cmd="simpleFoam", status="running"
    )
    # A resume re-reads every running job's status, so the backend has to know it too.
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")
    yield "resumed with a job running", brief_for(backend, store, resuming=True)

    # Nine combinations of the two levels, and each one is prose the model reads. A
    # level is the harness saying what the person wants, which is a hair away from the
    # harness saying what to do, so every one of them goes through the same sweep --
    # including with a standing note beside them, which is where the sentence about
    # which of the two wins appears.
    store.session.jobs.clear()
    a_workspace(backend)
    for ambition in levels.AMBITION:
        for consent in levels.CONSENT:
            yield (
                f"{ambition}/{consent}",
                brief_for(backend, store, ambition=ambition, consent=consent),
            )
    yield "levels beside a note", brief_for(
        backend, store, ambition="thorough", consent="never",
        preferences="Render the mesh and look at it.",
    )


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_no_shape_of_the_briefing_tells_the_model_what_to_do(pattern, backend, store):
    """Same rule as the system prompt, applied to the text that actually varies."""
    for shape, brief in every_shape_of_briefing(backend, store):
        match = re.search(pattern, brief, re.IGNORECASE)
        assert match is None, f"imperative language in the {shape} briefing: {match!r}"


def test_the_briefing_says_the_things_only_the_harness_knows(backend, store):
    """Each of these is something the model cannot find out for itself, and gets
    wrong when it is left unsaid. That is the whole test for whether a fact belongs."""
    a_workspace(backend)
    store.session.home = "/work/20260824-120000-abcd"
    store.session.instance_id = "inst-1"

    brief = brief_for(backend, store)

    assert "20260824-120000-abcd" in brief, "which study this is"
    assert "Your directory is" in brief, "and which directory is its own"
    assert "person is at the terminal" in brief, "whether an answer can arrive"


def test_a_run_with_nobody_watching_is_told_so(backend, store):
    """A question asked into a one-shot run is a turn ending on something nobody will
    ever read, and the model has no other way to tell which kind of session it is in."""
    a_workspace(backend)
    brief = brief_for(backend, store, interactive=False)

    assert "non-interactive" in brief
    assert "will not be seen" in brief


def test_the_briefing_stays_short(backend, store):
    """It is prepended to every session. Things that turned out to be worth saying
    accumulate, and the point at which nobody reads it is a real point."""
    a_workspace(backend, *[f"/work/s/case{n}" for n in range(60)])
    store.session.home = "/work/s"

    brief = brief_for(backend, store)

    assert len(brief) < 4000, "the briefing has grown past a screenful"
    assert "and 20 more" in brief, "a long listing is summarised rather than dumped"


def test_the_design_document_describes_the_workspace_the_code_builds():
    """A plan that contradicts the code is worse than no plan: it is the document
    someone reads first, and it will be believed."""
    plan = (Path(__file__).resolve().parents[1] / "docs" / "design.md").read_text(encoding="utf-8")
    assert "/work/<study-id>" in plan, "the plan still describes one shared workspace"
    assert "whether anyone is at the terminal" in plan


def test_the_design_document_keeps_up_with_the_transports_that_ship():
    """The same rule as above, applied to the one section that actually went stale.

    The transport section spent months stating there was no metered proxy and that
    a hosted one was future work, while the metered provider was live and shipping
    in `presets.py`. Nothing caught it, because the claim was prose and the truth
    was a dict. So tie the two together: if the code carries the preset, the
    document has to describe it.
    """
    from openreynolds.llm.presets import PRESETS

    plan = (Path(__file__).resolve().parents[1] / "docs" / "design.md").read_text(encoding="utf-8")
    if "reynolds" in PRESETS:
        assert "/v1/llm" in plan, (
            "the metered provider ships in presets.py and the design document "
            "does not describe it"
        )
        assert "there is no `/v1/llm`" not in plan, (
            "the design document still denies a transport the code ships"
        )


# -- the user's standing note ----------------------------------------------------


def test_a_standing_note_is_relayed_verbatim_in_the_users_voice(backend, store):
    """The user wrote it; the harness passes it on and adds nothing. What to do
    about it stays the model's call, which is what keeps this inside the contract."""
    a_workspace(backend)
    store.session.home = "/work/mine"

    brief = brief_for(
        backend, store, preferences="When meshing, render the mesh and look at it."
    )

    assert "In their own words:" in brief
    assert "When meshing, render the mesh and look at it." in brief


def test_no_note_means_no_mention_of_one(backend, store):
    a_workspace(backend)
    brief = brief_for(backend, store)
    assert "standing note" not in brief


# -- the two levels ---------------------------------------------------------------


def test_the_briefing_says_which_levels_the_user_picked(backend, store):
    """Before this the only control over how ambitious a run is was a blank page, and
    an empty one meant nothing was said -- which is not neutral. It is the ambitious
    end, chosen by default and found out about afterwards."""
    a_workspace(backend)

    brief = brief_for(backend, store, ambition="sketch", consent="never")

    assert "Ambition, `sketch`:" in brief
    assert "Consent, `never`:" in brief
    assert levels.AMBITION["sketch"] in brief
    assert levels.CONSENT["never"] in brief


def test_the_levels_are_there_even_when_nobody_set_them(backend, store):
    """A default that is written down is a default someone can disagree with."""
    a_workspace(backend)

    brief = brief_for(backend, store)

    assert "Ambition, `standard`:" in brief
    assert "Consent, `costly`:" in brief


def test_a_note_beside_the_levels_is_said_to_be_the_later_word(backend, store):
    """Which of the two wins when they disagree is a decision, and an undecided one
    would be discovered by whoever hit it first. The levels come off a menu; the note
    is the person's own sentences, so the note is what they meant."""
    a_workspace(backend)

    brief = brief_for(
        backend, store, ambition="sketch", preferences="Always mesh independence."
    )

    assert "the note is the one they wrote themselves" in brief
    assert "Always mesh independence." in brief


def test_an_unknown_level_is_shown_as_the_default_rather_than_relayed(backend, store):
    """A typo in `OPENREYNOLDS_AMBITION` reaching the briefing would put a word in the
    user's mouth that they never picked and the tool does not understand."""
    a_workspace(backend)

    brief = brief_for(backend, store, ambition="thorogh")

    assert "thorogh" not in brief
    assert "Ambition, `standard`:" in brief


# -- what the other directories on the volume are ------------------------------


def volume_with(backend, *paths):
    """`find` answers for whichever path it is asked about."""

    def looking(cmd, cwd=None, timeout_s=120, *, background=False):
        target = next(
            word.strip("'\"") for word in cmd.split()[1:] if not word.startswith("-")
        )
        rows = [p for p in paths if p.rsplit("/", 1)[0] == target.rstrip("/")]
        return ExecResult(0, "".join(f"d\t0\t1700000000.0\t{p}\n" for p in rows), False, None)

    backend.exec = looking


def test_the_briefing_says_what_the_other_directories_are(backend, store):
    """A live run found several near-identical studies made minutes apart by a user
    who did not remember commissioning them, and spent turns working through whether
    that meant an intruder. Saying whose the work is without saying what those
    sessions are leaves exactly that question open."""
    store.session.home = "/work/mine"
    volume_with(backend, "/work/mine", "/work/s1", "/work/s2")

    brief = brief_for(backend, store)

    assert "2 other directories from this tool's own earlier sessions" in brief
    assert "one per study id" in brief


def test_one_neighbour_is_described_in_the_singular(backend, store):
    store.session.home = "/work/mine"
    volume_with(backend, "/work/mine", "/work/s1")

    brief = brief_for(backend, store)

    assert "one other directory from this tool's own earlier sessions" in brief


def test_an_empty_volume_says_there_are_none(backend, store):
    store.session.home = "/work/mine"
    volume_with(backend, "/work/mine")

    assert "yours is the first" in brief_for(backend, store)


def test_a_study_that_owns_the_whole_workspace_is_told_nothing_about_neighbours(
    backend, store
):
    """There are none to describe, and inventing a sentence about it would be noise."""
    volume_with(backend)
    brief = brief_for(backend, store)
    assert "earlier sessions" not in brief


# -- what survives a context refresh ----------------------------------------------


def refreshed(backend, store, **settings):
    from openreynolds.config import Config

    return cli._fresh_thread_brief(store, backend, Config(**settings))


def test_a_refreshed_thread_is_told_the_levels_again(backend, store):
    """A refresh empties the thread, so the second half of a long study used to run on
    defaults nobody chose: the levels were said once at session start and thrown away
    at 80% of the window, while `/status` went on reporting them from the config."""
    a_workspace(backend)

    brief = refreshed(backend, store, ambition="sketch", consent="never")

    assert "Ambition, `sketch`:" in brief
    assert "Consent, `never`:" in brief


def test_a_refreshed_thread_is_told_the_standing_note_again(backend, store):
    """Same loss, and this one predates the levels: the note was relayed at session
    start and never again. The workspace survives a refresh on disk; what the user
    asked for only survives by being said."""
    a_workspace(backend)

    brief = refreshed(backend, store, preferences="Render the mesh and look at it.")

    assert "Render the mesh and look at it." in brief


def test_a_refreshed_thread_still_says_what_the_workspace_is(backend, store):
    """The facts `situation()` carried are not displaced by the ones added to it."""
    a_workspace(backend)

    brief = refreshed(backend, store)

    assert "fresh conversation thread" in brief


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_the_refreshed_briefing_tells_the_model_nothing_to_do(pattern, backend, store):
    a_workspace(backend)
    brief = refreshed(
        backend, store, ambition="thorough", consent="early",
        preferences="Check the layer report.",
    )

    match = re.search(pattern, brief, re.IGNORECASE)
    assert match is None, f"imperative language in the refreshed briefing: {match!r}"
