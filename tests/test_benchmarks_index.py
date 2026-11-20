"""`benchmarks/README.md` is an index, and an index that runs ahead of what it indexes
fails nothing unless something checks it.

The precedent is `test_toolbox.py::test_the_toolbox_index_names_nothing_that_is_not_there`:
the toolbox index once carried finished descriptions of two scripts that existed nowhere,
in the present tense, and a session reading it was promised instruments it could not run.
The same failure is available here and is worse, because what this index carries is
published reference values. A reference value that drifts away from the one the grader
actually applies turns the documentation into a second, quieter opinion about what the
right answer is.

So: every case the index names has a prompt or a directory behind it, and every reference
number it quotes is the number the grading code holds.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
INDEX = BENCHMARKS / "README.md"
TUTORIALS = BENCHMARKS / "tutorials"


def _load(name: str, directory: Path):
    spec = importlib.util.spec_from_file_location(name, directory / "grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


felix = _load("felix_grade", BENCHMARKS / "felix")
moving_mesh = _load("moving_mesh_grade", BENCHMARKS / "moving_mesh")


# -- the index points at things that are there -----------------------------------------


def test_every_directory_the_index_links_exists():
    """Relative links in the index, which are the only navigation it offers."""
    links = re.findall(r"\]\(([^)]+)\)", INDEX.read_text(encoding="utf-8"))
    relative = [a for a in links if not a.startswith(("http://", "https://", "#"))]
    assert relative, "the index has no links, so it is no longer an index"
    for target in relative:
        assert (BENCHMARKS / target).exists(), f"the index links {target}, which is not there"


def test_every_prompt_the_tutorials_index_names_exists():
    index = (TUTORIALS / "README.md").read_text(encoding="utf-8")
    named = re.findall(r"`([A-Za-z0-9_]+\.txt)`", index)
    assert named, "the tutorials index names no prompt files"
    for name in sorted(set(named)):
        assert (TUTORIALS / name).is_file(), f"the index names {name}, which is not there"


def test_every_prompt_in_the_directory_is_named_in_its_index():
    """The other direction: a prompt nobody can find is a prompt nobody runs."""
    index = (TUTORIALS / "README.md").read_text(encoding="utf-8")
    for prompt in sorted(TUTORIALS.glob("*.txt")):
        assert prompt.name in index, f"{prompt.name} is in tutorials/ and not in its index"


def test_each_prompt_names_the_reference_it_is_measured_against():
    """A prompt with no reference in it produces a number with nothing to check it
    against, which is the one thing these cases exist to avoid."""
    for prompt in sorted(TUTORIALS.glob("*.txt")):
        text = prompt.read_text(encoding="utf-8")
        assert re.search(r"\d", text), f"{prompt.name} quotes no number at all"
        named = re.search(r"Graetz|de Vahl Davis|Gartling|Armaly", text)
        assert named, f"{prompt.name} names no published source"


# -- the numbers in the prose are the numbers in the code ------------------------------


def test_the_index_quotes_the_drag_reference_the_grader_applies():
    index = INDEX.read_text(encoding="utf-8")
    assert f"{felix.ST_2D1['cd']}" in index
    low, high = felix.ST_2D1["cd_interval"]
    assert f"{low}-{high}" in index, "the index's Schafer-Turek interval is not the grader's"


def test_the_index_quotes_the_lift_interval_the_grader_reports():
    index = INDEX.read_text(encoding="utf-8")
    low, high = felix.ST_2D1["cl_interval"]
    assert f"{low:.4f}-{high:.4f}" in index


def test_the_index_quotes_the_strouhal_reference_the_grader_applies():
    index = INDEX.read_text(encoding="utf-8")
    assert f"{moving_mesh.THRESHOLDS['strouhal_reference']}" in index


def test_the_index_cites_ghia_with_the_tables_the_grader_reads():
    """The cavity grader samples Ghia's Table I and Table II, so the index naming one
    table, or the wrong journal volume, is a reader sent to the wrong page."""
    index = INDEX.read_text(encoding="utf-8")
    assert "Ghia, Ghia & Shin (1982)" in index
    assert "48" in index and "387-411" in index
    assert "Table I" in index and "Table II" in index


def test_the_graetz_value_is_quoted_to_the_digits_it_is_exact_to():
    """3.66 is the textbook rounding, 0.09% above the eigenvalue. The tutorial claims
    +0.55% on a measured 3.677, which is only meaningful against 3.6568."""
    for text in (INDEX.read_text(encoding="utf-8"),
                 (TUTORIALS / "README.md").read_text(encoding="utf-8")):
        assert "3.6568" in text


# -- the two reference values that were wrong before this directory existed ------------
#
# Both are recorded as tests rather than only as prose, because both are the kind of
# error that flatters the result and so survives review: they make the agreement look
# better, and nothing downstream complains.


def test_the_hot_wall_nusselt_reference_is_not_the_cavity_average():
    """de Vahl Davis (1983) Table V gives Nu_0 = 4.509 on the hot wall and Nu = 4.519
    through the cavity. Secondary sources almost universally cite 4.519 as the hot-wall
    value; it is 0.22% out, against a paper whose own accuracy at Ra 1e5 is 0.3%. A
    wall-integrated Nusselt number belongs against 4.509."""
    for text in (INDEX.read_text(encoding="utf-8"),
                 (TUTORIALS / "README.md").read_text(encoding="utf-8")):
        assert "4.509" in text, "the hot-wall reference is missing"
        assert "4.519" in text, "the value it is distinguished from is missing"
    # Somewhere the text has to say what 4.519 actually is, or quoting it is the same
    # misquote with an extra number beside it. Any mention with the explanation within
    # reach of it counts; which mention does the work is not the point.
    tutorial = " ".join((TUTORIALS / "README.md").read_text(encoding="utf-8").split())
    explained = [
        tutorial[max(0, m) - 200:m + 400].lower()
        for m in (i for i in range(len(tutorial)) if tutorial.startswith("4.519", i))
    ]
    assert any("cavity" in w or "mid-plane" in w for w in explained), (
        "4.519 is quoted without saying it is the cavity average or the mid-plane value")


def test_the_step_reattachment_reference_is_not_quoted_in_step_heights():
    """Gartling's 6.1 is in units of the downstream channel height, which is two step
    heights. Reading it as step heights is a factor of two, and it was read that way
    here: `reattach.py`, `claims.py` and this repository's own records all say 6.32 or
    6.42 "matches Gartling's 6.1". Whatever else this directory says about that case, it
    does not repeat the claim."""
    text = " ".join((TUTORIALS / "README.md").read_text(encoding="utf-8").split())
    assert "12.2" in text, "the step-height equivalent of Gartling's 6.1 is not stated"
    assert "channel height" in text
    assert not re.search(r"6\.1\s*step heights", text), (
        "6.1 is quoted as step heights, which is the factor-of-two error")


def test_an_unread_primary_source_is_flagged_as_unread():
    """Gartling (1990) is paywalled and was not read. The brief this directory was built
    to is explicit that an honest "not established" beats a confident number, so the
    absence is stated rather than smoothed over."""
    text = " ".join((TUTORIALS / "README.md").read_text(encoding="utf-8").split()).lower()
    assert "no primary access" in text or "was not read" in text or "paywall" in text


@pytest.mark.parametrize("case", ["felix", "moving_mesh", "tutorials", "lbm", "sph"])
def test_the_index_names_every_family_beside_it(case):
    """A family added under benchmarks/ and not added to the index is invisible."""
    assert f"{case}/" in INDEX.read_text(encoding="utf-8")


def test_the_index_states_an_agreement_for_every_case_it_lists():
    """The rule the whole directory exists for: a case with a reference and no measured
    agreement beside it teaches a reader to trust an unvalidated number. Rows that
    deliberately have no verdict say so in words instead of leaving the cell empty."""
    rows = [line for line in INDEX.read_text(encoding="utf-8").splitlines()
            if line.startswith("|") and line.count("|") >= 5]
    body = [r for r in rows if not re.match(r"^\|[\s|:-]+\|$", r) and "measures" not in r]
    assert len(body) >= 8, f"the index lists only {len(body)} cases"
    for row in body:
        measured = row.split("|")[4]
        assert re.search(r"\d", measured) or "not graded" in measured or "no " in measured, (
            f"this row states no agreement achieved: {row.strip()}")
