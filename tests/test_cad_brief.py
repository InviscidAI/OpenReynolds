"""What the CAD desk is told, and the build123d index it is pointed at.

Every item here is a gate item from C7 of `docs/cad-implementation-plan.md`, and each is
written against a failure that has already happened rather than against a style. The two
that carry the most weight:

* **An instrument nobody is told about is unused.** `geometry_view.py` was run zero times
  across every persona run while `read_file` on a PNG was run constantly. So each script
  is asserted present in the brief *with a sentence saying what it answers*, not merely
  named.
* **An entry that names nothing is worse than no entry**, now that the index's job is
  discovery rather than reference. So the index is regenerated here and every curated
  name is resolved against the installed build123d.

Two measurements are recorded rather than only bounded, because the three-tier split is
an argument about size and an argument about size that nobody measured is a guess:

    the brief         11,782 chars, ~2,945 tokens at 4 chars/token
    b123d_api.md      17,683 bytes, 17.3 KB

The plan estimated the brief at ~1.5k tokens before the full requirement list was
written; the assertions below bound what was actually built, generously, so that a brief
that doubles is caught and one that gains a sentence is not.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from openreynolds.cad.brief import CAD_DONE, remark_message, system_prompt, task_message
from openreynolds.prompt import SYSTEM_PROMPT

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"

build123d = pytest.importorskip("build123d")


@pytest.fixture(scope="module")
def api():
    """`b123d_api.py` loaded by path: the toolbox is data, not a package."""
    spec = importlib.util.spec_from_file_location("toolbox_b123d_api",
                                                  TOOLBOX / "b123d_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def index(api):
    """The index regenerated here, not the committed copy -- entries are resolved."""
    return api.render()


@pytest.fixture(scope="module")
def brief():
    return system_prompt(240)


def normalise(text: str) -> str:
    """Where a sentence happens to wrap is not the point."""
    return " ".join(text.split())


# -- the instruments, by name and by the question each answers -----------------

INSTRUMENTS = {
    "mesh_look.py": "what does this mesh actually look like",
    "cad_convert.py": "what do these faces become as triangles",
    "cad_audit.py": "is the exported surface sound",
    "domain_probe.py": "is this point inside the fluid",
    "patch_entries.py": "do the three snappy dictionary sections agree",
    "cfmesh.py": "is cfMesh available here",
    "layer_report.py": "how much of the wall actually carries a prism layer",
    "preflight.py": "which of the cheap questions fail before an expensive run",
    "templates/snappy/README.md": "how is a patch set taken to a snappy mesh",
    "templates/prep/README.md": "how is an imported STEP taken to a tagged",
}


@pytest.mark.parametrize("script,question", sorted(INSTRUMENTS.items()))
def test_every_instrument_is_named_with_the_question_it_answers(brief, script, question):
    """`geometry_view.py` was used zero times across every persona run. A name on its
    own is not what gets an instrument reached for; what it answers is."""
    flat = normalise(brief)
    assert script in flat, f"{script} is not named in the brief"
    assert question in flat, f"{script} is named without saying what it answers"


def test_the_instruments_are_offered_rather_than_demanded(brief):
    """The desk is allowed to be told what to do; the toolbox is still not imposed."""
    assert "offered, not imposed" in normalise(brief)


# -- the paid-for inheritance, one assertion per item --------------------------
#
# The list is the build plan's §2, "What the replacement must inherit", in its order.
# Each item gets its own distinct phrase, so a brief that dropped one cannot pass on
# another's wording.

@pytest.mark.parametrize("phrase", [
    "comes back to you attached",                         # PNGs come back attached
    "it does not take your word for anything",            # the finish token is checked
    "an unexamined mesh is the failure this desk exists to end",   # budget-exhausted
    "unreachable is not failure",                         # unreachable is not a verdict
    "these are what they want",                           # verbatim outranks paraphrase
    "takes precedence over what you were told at the start",       # a mid-run remark
    "Never assign a patch by asking where a face sits in the bounding box",
    "**Metres, always.**",
    "**Coarse first.**",
    "**Say what you did not check.**",
])
def test_the_inherited_list_is_complete(brief, phrase):
    """Each of these was paid for in a real run. A brief that quietly drops one loses
    what the run bought."""
    whole = normalise(brief + task_message("r", "/c", "c", said=["w"])
                      + remark_message("wider"))
    assert normalise(phrase) in whole, f"the brief no longer carries: {phrase}"


def test_the_person_s_own_words_arrive_verbatim_and_outrank_the_paraphrase():
    message = task_message("a duct", "/work/s/c", "c", said=["make it 74 mm across"])
    assert "make it 74 mm across" in message
    assert "another agent's reading of this" in message


def test_a_remark_is_acted_on_now_rather_than_after():
    message = remark_message("  2 mm wider  ")
    assert '"2 mm wider"' in message
    assert "rather than finishing what you were doing first" in normalise(message)


def test_a_supplied_cad_file_arrives_with_the_job_and_with_the_read_only_rule():
    message = task_message("clean this up", "/work/s/c", "c",
                           geometry="/work/in/chassis.step")
    assert "/work/in/chassis.step" in message
    assert "do not redesign it" in message


# -- the read-only rule is here and not in the frozen prompt -------------------

def test_the_read_only_rule_is_in_the_brief(brief):
    """§1's non-goal: prep removes and cuts, it does not redesign. It is a distinction
    of intent, so nothing in the type system can carry it."""
    flat = normalise(brief)
    assert "Do not redesign what you were handed." in flat
    assert "you are preparing it, not improving it" in flat


def test_the_read_only_rule_is_absent_from_the_system_prompt():
    """`tests/test_prompt.py` holds the frozen prompt free of mandated workflow, and
    this rule is mandated workflow. It belongs to one desk, not to the machine."""
    for phrase in ("redesign", "preparing it, not improving it", "algebra mode"):
        assert phrase not in SYSTEM_PROMPT.lower(), \
            f"{phrase!r} leaked into the frozen system prompt"


# -- the new rules the kernel and the plan add ---------------------------------

@pytest.mark.parametrize("phrase", [
    "Write build123d in algebra mode, directly",           # #13, no facade
    "no house wrapper over it",                            # no facade, said outright
    "Do not use builder mode",                             # the rejected surface
    "named constants",                                     # §3 cell authoring
    "depend on nothing that is not bound by a cell that was accepted",
    "re-running the concatenation must give the same answer",
    "Your reasoning is carried with the cell",
    "snappyHexMesh is the default; gmsh-OCC is the quick shape check",
    "Cheap checks before expensive operations.",
    "There is no phase order here",
])
def test_the_new_requirements_are_in_the_brief(brief, phrase):
    assert normalise(phrase) in normalise(brief), f"the brief does not say: {phrase}"


def test_a_cell_that_might_outrun_the_window_checkpoints_to_disk(brief):
    """The one habit the kernel asks for that a shell did not: `nohup ... &` was free
    under bash because everything communicated through files, and a `Part` computed in
    a subprocess does not come back into the session."""
    flat = normalise(brief)
    assert "A cell that might outrun the window checkpoints to disk." in flat
    assert "loses the live binding" in flat
    assert "export_step" in flat


def test_the_window_expiring_does_not_kill_the_cell(brief):
    """I9: on expiry the cell is reported still running with what it streamed, and the
    desk's next step polls or interrupts on purpose. Killing would lose every binding."""
    flat = normalise(brief)
    assert "That window expiring does not kill your cell." in flat
    assert "still running" in flat
    assert "polls it or interrupts it on purpose" in flat


def test_a_mesher_still_belongs_in_the_background(brief):
    """A mesher communicates through files and loses nothing by being detached; the
    checkpoint rule must not be read as banning that."""
    assert "because a mesher communicates through files" in normalise(brief)


# -- the finish token ----------------------------------------------------------

def test_the_finish_token_is_a_request_that_is_checked(brief):
    assert CAD_DONE == "CAD_DONE"
    flat = normalise(brief)
    assert f'print("{CAD_DONE}")' in flat
    assert "The harness then runs its own check" in flat
    assert ("it is not a formality and it does not take your word for anything") in flat


def test_running_out_of_budget_still_gets_checked(brief):
    assert "the same check still runs on whatever is in the directory" in normalise(brief)


# -- tier 1: one line, and the index is not inlined ----------------------------

def instrument_bullets(brief: str) -> list[str]:
    """The instrument section's bullets, unwrapped."""
    section = brief.split("# The instruments", 1)[1].split("# Finishing", 1)[0]
    return [normalise(b) for b in re.split(r"\n- ", "\n" + section) if b.strip()][1:]


def test_tier_one_is_one_entry_and_no_larger_than_any_other_instrument(brief):
    """The brief is paid at every step of a thirty-step run. An index inside it would
    be several times the brief; a pointer to it is one bullet like the others."""
    bullets = instrument_bullets(brief)
    ours = [b for b in bullets if "b123d_api" in b]
    assert len(ours) == 1, f"b123d_api is mentioned in {len(ours)} bullets, not one"
    others = [len(b) for b in bullets if "b123d_api" not in b]
    assert len(ours[0]) <= max(others), \
        "the build123d pointer takes more room in the brief than any other instrument"


def test_the_brief_points_at_the_file_and_at_the_kernel(brief):
    """Tier 2 is the file; tier 3 is `help()` and `inspect.signature()` live. The second
    is named because a model used to bash will not reach for it by default -- and it is
    also what makes the curated set's omissions a decision rather than an oversight."""
    flat = normalise(brief)
    assert "/work/.toolbox/b123d_api.md" in flat
    assert "help()" in flat
    assert "inspect.signature()" in flat
    assert "grep" in flat, "the index's search is grep; the brief says so"


def test_the_index_is_not_inlined_into_the_brief(brief, index):
    """Recorded: the brief is 11,782 chars (~2,945 tokens at 4 chars/token) and the
    index is 17,683 bytes. The whole point of the split is that the second is not paid
    for at every step."""
    chars = len(brief)
    assert chars < 14_000, f"the brief has grown to {chars} chars (~{chars // 4} tokens)"
    assert len(index) > chars, "the index is meant to be the bigger of the two"
    headings = [line for line in index.splitlines() if line.startswith("## ")]
    for heading in headings:
        assert heading[3:] not in brief, f"index section {heading!r} inlined in the brief"
    # The entries themselves, not only the headings.
    assert "filter_by_position" not in brief
    assert brief.count("b123d_api.md") == 1


# -- the index: every entry resolves -------------------------------------------

def test_every_curated_entry_names_a_live_attribute(api):
    """The verification half of "curated selection, introspected content, verified
    live". An entry that names nothing is the failure mode of a discovery aid."""
    assert api.unresolved() == []


def test_the_generator_refuses_rather_than_writing_a_stale_index(api, tmp_path, capsys):
    """A name that has gone is an error, not a line quietly dropped from the output."""
    api.GROUPS.append(("Invented", "nothing", [("no_such_thing", "", "not real")]))
    try:
        out = tmp_path / "b123d_api.md"
        assert api.main(["--out", str(out)]) == 2
        assert not out.exists()
        assert "no_such_thing" in capsys.readouterr().err
    finally:
        api.GROUPS.pop()


def test_the_shipped_index_is_what_the_generator_produces(index):
    """It is committed so it reaches `/work/.toolbox/` with the rest of the toolbox;
    committing it is also how it can drift, so it is compared rather than trusted."""
    shipped = (TOOLBOX / "b123d_api.md").read_text(encoding="utf-8")
    assert shipped == index, "b123d_api.md is stale; re-run `python3 b123d_api.py`"


# -- the index: what is in it --------------------------------------------------

@pytest.mark.parametrize("operator,meaning", [
    ("`a + b`", "fuse"),
    ("`a - b`", "cut"),
    ("`a & b`", "intersect"),
])
def test_the_operators_are_present_and_explicit(index, operator, meaning):
    """No introspection pass produces a dunder, and these three *are* algebra mode."""
    line = [b for b in index.split("\n- ") if b.startswith(operator)]
    assert line, f"{operator} is not in the index"
    assert meaning in line[0], f"{operator} is listed without saying what it does"


def test_the_operators_carry_the_warning_that_order_matters(index):
    """Difference does not commute and mixed compositions do not associate. A surface
    sold on locality of reference has to say where the ordering still lives."""
    assert "difference does not commute" in index


@pytest.mark.parametrize("method", [
    "filter_by", "filter_by_position", "sort_by", "sort_by_distance", "group_by",
    "first", "last", "center", "faces", "edges", "vertices", "solids",
    "bounding_box", "moved", "located",
])
def test_the_selector_group_is_present_and_non_empty(index, method):
    """C5's tag-stability gate is written in these; undocumented, they are the one
    correctness-critical idiom nobody is told about."""
    assert method in index, f"{method} is missing from the index"


def test_the_selector_group_says_selectors_are_re_derived(index):
    assert "re-derived from geometry on each run" in index
    assert "a face index from a previous session is not a selector" in index


@pytest.mark.parametrize("group", [
    "operations", "primitives", "Curves", "Placement", "Selecting", "Measuring",
    "Reading and writing files",
])
def test_the_index_is_grouped_rather_than_flat(index, group):
    """A flat alphabetical dump is where the twenty things that matter drown."""
    headings = " ".join(line for line in index.splitlines() if line.startswith("## "))
    assert group in headings, f"no section about {group}"


def test_the_index_has_enough_sections_to_be_a_map(index):
    headings = [line for line in index.splitlines() if line.startswith("## ")]
    assert len(headings) >= 8, f"only {len(headings)} sections; that is close to flat"


def test_the_index_is_under_twenty_kilobytes(index):
    """Recorded: 17,683 bytes, 17.3 KB, against the plan's 15-20 KB band. Tier 2 is
    read on demand, so it can be this big; it cannot be the 179 KB full-method dump."""
    size = len(index.encode("utf-8"))
    assert 12_000 <= size <= 20 * 1024, f"the index is {size / 1024:.1f} KB"


# -- the index: what is deliberately out ---------------------------------------

@pytest.mark.parametrize("pattern", [
    r"\bBuildPart\b", r"\bBuildSketch\b", r"\bBuildLine\b", r"\bBuildSolid\b",
])
def test_builder_mode_is_absent_from_the_index(index, pattern):
    """Algebra mode was chosen because there is no ambient context to replay. Offering
    the builder surface beside it invites the failure class the choice avoided."""
    for line in index.splitlines():
        if line.startswith("- ") or line.startswith("  "):
            assert not re.search(pattern, line), f"builder mode offered: {line!r}"


def test_the_mode_parameter_is_stripped_from_every_signature(index):
    """`mode=Mode.ADD` is on the end of every operation's real signature and is builder
    machinery: it names the pending context to add to, and there is no context here."""
    assert not re.search(r"\bMode\b", index), "Mode leaked into the index"
    for line in index.splitlines():
        if line.startswith("- `"):
            assert not re.search(r"\bmode=", line), \
                f"builder-mode parameter in a signature: {line!r}"


@pytest.mark.parametrize("pattern", [r"\bBRep\w*", r"\bBnd_\w*", r"\bTopoDS\w*", r"\bgp_\w+"])
def test_raw_occt_bindings_are_absent_from_the_index(index, pattern):
    assert not re.search(pattern, index), f"OCCT binding matching {pattern} in the index"


@pytest.mark.parametrize("leak", ["dataclass", "Callable", "sqrt", "radians"])
def test_standard_library_leakage_is_absent_from_the_index(index, leak):
    """`dir(build123d)` re-exports these; they are not build123d's API."""
    assert not re.search(rf"\b{leak}\b", index), f"{leak} leaked into the index"


def test_methods_beyond_the_curated_set_are_deliberately_absent(index, api):
    """Every public method on the core classes is 9,998 entries and ~179k tokens. What
    is here is a reading list a couple of hundred long, and the omission is the point --
    the brief names `help()` and `inspect.signature()` as how to get the rest, asserted
    in the same test so it reads as a decision rather than an oversight."""
    entries = [line for line in index.splitlines() if line.startswith("- ")]
    assert len(entries) < 200, f"{len(entries)} entries; that is drifting into a dump"
    curated = {name for _, target, _ in api.OPERATORS for name in [target]}
    for _, group_entries in [(t, e) for t, _, e in api.GROUPS]:
        curated.update((target or display) for display, target, _ in group_entries)
    for uncurated in ("wrapped", "geom_adaptor", "project_to_viewport", "relocate",
                      "combined_center", "cast", "to_splines"):
        assert uncurated not in curated
        assert not re.search(rf"\b{uncurated}\b", index), \
            f"{uncurated} is in the index and is not part of the curated selection"
    flat = normalise(system_prompt(240))
    assert "help()" in flat and "inspect.signature()" in flat


def test_the_index_says_what_it_left_out_and_why(api):
    """The provenance claim is curated selection, introspected content, verified live.
    A reader who cannot see the cut cannot tell an omission from an oversight."""
    source = (TOOLBOX / "b123d_api.py").read_text(encoding="utf-8")
    assert "curated selection, introspected content, verified live" in source
    for cut in ("Builder mode is absent", "Raw OCCT is absent",
                "Standard-library leakage is absent"):
        assert cut in source, f"the file does not record the cut: {cut}"


# -- the toolbox index ---------------------------------------------------------

def test_the_toolbox_index_carries_the_script(api):
    """`tests/test_toolbox.py` fails on a script missing from the index; this says what
    the row has to be about, not only that a row exists."""
    row = [line for line in (TOOLBOX / "README.md").read_text(encoding="utf-8").splitlines()
           if "b123d_api.py" in line]
    assert len(row) == 1, "exactly one row, appended at the end of the table"
    assert "b123d_api.md" in row[0], "the row does not name what the script writes"
