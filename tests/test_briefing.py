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


# -- what the other directories on the volume are ------------------------------


def volume_with(backend, *paths):
    """`find` answers for whichever path it is asked about."""

    def looking(cmd, cwd=None, timeout_s=120):
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
