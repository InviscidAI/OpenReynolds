"""`claims.py`: the figure carries the number, and the numbers get put side by side.

The incident is F-36. The backward-facing-step study measured its reattachment
length twice -- 6.32 from one of its own scripts, 7.5667 from the other, the one
that drew the picture -- wrote 6.4 in the answer, handed over a figure annotated
"reattachment x/h=7.57", and reconciled none of it. Every number was in the same
session's tool output minutes apart.

So the tests here are about the two comparisons that were never made: a claim
against another claim of the same quantity, and a claim against the prose that is
supposed to be about it. The fixtures use the actual numbers, because a check that
would not have caught this one is not the check that was needed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def claims():
    return _load("claims")


@pytest.fixture
def png(tmp_path):
    """A real PNG, drawn the way the toolbox draws them, so the chunk surgery is
    tested against matplotlib's own output rather than against a byte string this
    file made up."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = tmp_path / "figure.png"
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    fig.savefig(out)
    plt.close(fig)
    return out


# -- the stamp -----------------------------------------------------------------


def test_a_stamped_png_is_still_a_png_that_decodes(claims, png):
    """The stamp is worthless if it costs the picture. A `tEXt` chunk is inserted
    before IEND and every other chunk is copied byte for byte, so the image an
    ordinary reader gets back is the one matplotlib wrote."""
    import matplotlib.image as mpimg

    before = mpimg.imread(str(png)).shape
    claims.stamp_png(png, [claims.claim("reattachment-length", 6.32, symbol="x_r/h")])
    assert png.read_bytes().startswith(claims.PNG_SIGNATURE)
    assert mpimg.imread(str(png)).shape == before


def test_the_number_travels_with_the_file(claims, png, tmp_path):
    """Not a sidecar. The figure in F-36 was fetched to a laptop and looked at
    there; a number kept beside it in the study directory would not have been in
    the room."""
    row = claims.claim("reattachment-length", 6.32, symbol="x_r/h", source="reattach.py")
    claims.stamp_png(png, [row])
    moved = tmp_path / "somewhere-else" / "renamed.png"
    moved.parent.mkdir()
    moved.write_bytes(png.read_bytes())
    (back,) = claims.read_png(moved)
    assert back["value"] == 6.32
    assert back["source"] == "reattach.py"


def test_redrawing_a_figure_replaces_its_claim_rather_than_adding_a_second(claims, png):
    """A figure carrying both 7.5667 and 6.32 is the original confusion with extra
    steps: the reader would still have to pick."""
    claims.stamp_png(png, [claims.claim("reattachment-length", 7.5667, symbol="x_r/h")])
    claims.stamp_png(png, [claims.claim("reattachment-length", 6.32, symbol="x_r/h")])
    rows = claims.read_png(png)
    assert [row["value"] for row in rows] == [6.32]


def test_a_figure_nobody_stamped_says_nothing_rather_than_failing(claims, png):
    """Most PNGs carry no claim, including every one drawn before this existed."""
    assert claims.read_png(png) == []


def test_something_that_is_not_a_png_is_a_fact_not_a_traceback(claims, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("x_r/h = 6.32")
    assert claims.read_png(junk) == []
    with pytest.raises(claims.StampError):
        claims.stamp_png(junk, [])


def test_the_caption_drawn_on_a_figure_is_the_value_stamped_into_it(claims):
    """The structural point of the whole file. `analyze.py` annotated 7.57 and
    printed 7.5667 from a different variable; here the annotation and the record
    are one call, so the two cannot drift."""
    row = claims.claim("reattachment-length", 6.3204, symbol="x_r/h")
    drawn = claims.caption(row)
    (number,) = re.findall(r"[-+]?\d*\.?\d+", drawn)
    assert float(number) == pytest.approx(row["value"])


# -- claim against claim -------------------------------------------------------


def study(claims, tmp_path, entries):
    """A study directory holding one stamped figure per entry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (tmp_path / ".reynolds").mkdir(exist_ok=True)
    for name, value, source in entries:
        out = tmp_path / name
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        fig.savefig(out)
        plt.close(fig)
        row = claims.claim(
            "reattachment-length", value, symbol="x_r/h", source=source,
            aliases=["reattachment length", "x/h"],
        )
        claims.attach(out, [row], root=tmp_path)
    return tmp_path


def test_two_scripts_that_answered_one_question_differently_are_caught(claims, tmp_path):
    """F-36 itself, with its own numbers: `analyze.py` drew 7.5667 and `reattach.py`
    measured 6.32, both in the same session, and nothing put them together."""
    root = study(claims, tmp_path, [
        ("bfs_result.png", 7.5667, "analyze.py"),
        ("wall_shear.png", 6.32, "reattach.py"),
    ])
    (clash,) = claims.contradictions(claims.collect(root))
    assert clash["name"] == "reattachment-length"
    assert {clash["left"]["value"], clash["right"]["value"]} == {7.5667, 6.32}
    assert clash["apart"] > 0.15


def test_two_scripts_that_agree_are_not_a_finding(claims, tmp_path):
    root = study(claims, tmp_path, [
        ("a.png", 6.32, "reattach.py"),
        ("b.png", 6.30, "another.py"),
    ])
    assert claims.contradictions(claims.collect(root)) == []


def test_the_same_length_in_two_units_is_not_converted_and_not_dropped_either(claims):
    """6.32 step heights and 0.0594 m may be one measurement. Guessing at the
    conversion is how a reconciliation tool starts manufacturing its own errors --
    but dropping the pair is worse, because the tool then goes on to say nothing
    disagrees. It comes back marked not-comparable, with no `apart` invented for
    it."""
    rows = [
        claims.claim("reattachment-length", 6.32, units=""),
        claims.claim("reattachment-length", 0.0594, units="m"),
    ]
    (pair,) = claims.contradictions(rows)
    assert pair["comparable"] is False
    assert pair["apart"] is None


def test_a_pair_that_could_not_be_compared_is_never_covered_by_the_all_clear(claims, tmp_path):
    """The false pass that made the first version of this file worthless on the
    incident it was written for. `reattach.py` labelled its own 6.32 "m", the
    hand-attached 7.5667 carried the default empty label, `contradictions` skipped
    the pair on the mismatch, and the report printed "no two claims of the same name
    disagree" over the top of F-36 itself."""
    root = study(claims, tmp_path, [("bfs_result.png", 7.5667, "analyze.py")])
    hand = claims.claim("reattachment-length", 6.32, units="m", source="reattach.py")
    rows = claims.collect(root) + [hand]
    text = claims.report(rows, claims.contradictions(rows), [], None, claims.DEFAULT_TOL)
    assert "no two claims of the same name disagree" not in text
    assert "NOT compared" in text
    assert "7.5667" in text and "6.32" in text


def test_a_pair_that_cannot_be_compared_is_reported_even_when_it_looks_close(claims):
    """The tolerance gate used to run BEFORE the comparability test, so a pair whose
    raw numbers happened to sit within tol under two different labels was dropped
    silently and the all-clear printed over it. 6.32 h and 6.30 m are 6 cm and about
    a hundred times that; nothing compared them, and the report's last line said no
    two claims of the same name disagree. Numbers under two labels being numerically
    close is a coincidence of two scales, not agreement, so there is no tolerance to
    apply to them."""
    rows = [
        claims.claim("reattachment-length", 6.32, units="h", source="reattach.py"),
        claims.claim("reattachment-length", 6.30, units="m", source="analyze.py"),
    ]
    (pair,) = claims.contradictions(rows)
    assert pair["comparable"] is False
    assert pair["apart"] is None

    text = claims.report(rows, claims.contradictions(rows), [], None, claims.DEFAULT_TOL)
    assert "no two claims of the same name disagree" not in text
    assert "NOT compared" in text

    # The gate itself is intact for the pairs it applies to: one label, close values.
    agreeing = [
        claims.claim("reattachment-length", 6.32, units="h"),
        claims.claim("reattachment-length", 6.30, units="h"),
    ]
    assert claims.contradictions(agreeing) == []


def test_a_unit_spelled_two_ways_on_two_command_lines_is_one_unit(claims):
    """`--units h` and `--units " H"` are one person being inconsistent, not two
    quantities. Every difference in this string costs a comparison, so the cheap
    ones are folded away."""
    rows = [
        claims.claim("reattachment-length", 6.32, units="h"),
        claims.claim("reattachment-length", 7.5667, units=" H "),
    ]
    (clash,) = claims.contradictions(rows)
    assert clash["comparable"] is True
    assert clash["apart"] > 0.15


def test_a_claim_stamped_before_the_study_had_state_is_still_found(claims, tmp_path, png):
    """`collect` reads the manifest and the images. The manifest misses a figure
    drawn before `.reynolds/` existed; the images miss nothing."""
    claims.stamp_png(png, [claims.claim("reattachment-length", 7.5667, symbol="x_r/h")])
    found = claims.collect(png.parent)
    assert [row["value"] for row in found] == [7.5667]


# -- claim against the written answer ------------------------------------------


ANSWER_AS_DELIVERED = """# Backward-facing step at Re = 800

The recirculation closes on the lower wall: the reattachment length is
x_r/h = 6.4, within 5% of Gartling's benchmark 6.1. See bfs_result.png for the
wall-shear distribution the number comes from.
"""


def test_the_delivered_answer_and_the_delivered_figure_are_put_side_by_side(claims, tmp_path):
    """The finding, exactly. The answer says 6.4 and points the reader at a figure
    annotated 7.57, and a reader who trusts the picture -- which is what a picture
    is for -- takes away a number 24% above the benchmark."""
    root = study(claims, tmp_path, [("bfs_result.png", 7.5667, "analyze.py")])
    (verdict,) = claims.against_answer(claims.collect(root), ANSWER_AS_DELIVERED)
    assert verdict["verdict"] == "disagrees"
    assert 6.4 in verdict["answer_numbers"]


def test_a_figure_that_says_what_the_answer_says_passes(claims, tmp_path):
    root = study(claims, tmp_path, [("wall_shear.png", 6.32, "reattach.py")])
    (verdict,) = claims.against_answer(claims.collect(root), ANSWER_AS_DELIVERED)
    assert verdict["verdict"] == "agrees"


def test_an_answer_that_never_quotes_the_number_is_unmentioned_not_agreed(claims, tmp_path):
    """A third verdict on purpose: "the answer does not contain this number" and
    "the answer contains it correctly" are different states, and a tool that scored
    them the same would report a silent deliverable as a clean one."""
    root = study(claims, tmp_path, [("wall_shear.png", 6.32, "reattach.py")])
    (verdict,) = claims.against_answer(claims.collect(root), "The solve converged.\n")
    assert verdict["verdict"] == "unmentioned"


def test_a_number_from_the_previous_sentence_cannot_rescue_a_disagreement(claims, tmp_path):
    """The window back from a mention stops at its own sentence. `Re = 800` two
    lines above must not be swept in as a candidate, because a stray number that
    happens to land within tolerance turns a real disagreement into an agreement --
    which is the direction this check must never fail in."""
    root = study(claims, tmp_path, [("bfs_result.png", 7.5667, "analyze.py")])
    text = "The Reynolds number is 7.6e0 here.\n\nThe reattachment length is x_r/h = 6.4.\n"
    (verdict,) = claims.against_answer(claims.collect(root), text)
    assert verdict["answer_numbers"] == [6.4]
    assert verdict["verdict"] == "disagrees"


def test_a_number_from_the_next_sentence_cannot_rescue_a_disagreement_either(claims, tmp_path):
    """The forward half of the window used to be a flat 60 characters, so the
    sentence after the mention could hand a claim an alibi. Both of these are
    ordinary ways to write the sentence that points a reader at a picture, and both
    turned the delivered 7.5667 into `agrees`: once on a cell count, once on a
    benchmark quoted in a sentence that was not about the claim."""
    root = study(claims, tmp_path, [("bfs_result.png", 7.5667, "analyze.py")])
    rows = claims.collect(root)

    (cells,) = claims.against_answer(
        rows, "The reattachment length is x_r/h = 6.4. The mesh has 7.5 M cells.\n")
    assert cells["answer_numbers"] == [6.4]
    assert cells["verdict"] == "disagrees"

    (figure,) = claims.against_answer(
        rows,
        "The reattachment length is plotted in the figure. Gartling reports 7.6 for "
        "this case.\n")
    assert figure["verdict"] == "unmentioned"


def test_a_sentence_with_no_boundary_behind_it_does_not_get_the_whole_document(
    claims, tmp_path
):
    """The backward half of the window had a second door to the same failure. `back`
    was initialised to 0 and only ever written inside the boundary search, so a
    mention with no sentence boundary in the 200 characters behind it got a window
    running to character 0 of the file, and every number before it became a
    candidate. One 256-character sentence is enough -- ordinary prose, no
    contrivance -- and the delivered 7.5667 came back `agrees` on a benchmark 250
    characters upstream."""
    root = study(claims, tmp_path, [("bfs_result.png", 7.5667, "analyze.py")])
    rows = claims.collect(root)

    text = (
        "Gartling reports 7.6 for this configuration, and the mesh used here was "
        "refined until the wall spacing stopped moving the answer at all, which "
        "took rather more cells than the first attempt used and rather longer to "
        "run than the schedule allowed for, so the reattachment "
        "length is plotted in the figure below.\n"
    )
    assert text.index("reattachment length") - (text.index("7.6") + 3) > claims.LOOK_BACK
    (verdict,) = claims.against_answer(rows, text)
    assert 7.6 not in verdict["answer_numbers"]
    assert verdict["verdict"] == "unmentioned"

    # And the budget is a budget, not a hole: a number inside it, in the same
    # unpunctuated stretch, is still found.
    near = "the reattachment length is 6.4 in this run"
    (close,) = claims.against_answer(rows, near)
    assert close["answer_numbers"] == [6.4]


def test_a_percentage_of_more_than_one_digit_is_not_a_length_either(claims, tmp_path):
    """The first guard was `(?!\\s*%)`, which only held for single-digit integer
    percentages: `\\d+` backtracks, so "within 15%" was rejected as 15 and then
    ACCEPTED as 1, "some 24% above" gave 2, and "12.5%" gave 12. Those truncations
    are candidates that did not exist before the guard, and 1.0-2.0 x/h is an
    ordinary reattachment length, so the guard manufactured the failure it was added
    to stop."""
    for text, expected in (
        ("within 5% of the benchmark", []),
        ("within 15% of the benchmark", []),
        ("some 24% above Gartling", []),
        ("agrees to 15.5%", []),
        ("within 6.4% of it", []),
        ("a spread of 100 %", []),
        # and the numbers that are not percentages still come through
        ("x_r/h = 6.4, within 5% of the benchmark", [6.4]),
        ("x_r/h = 6.4, within 15% of the benchmark", [6.4]),
        ("6.1 and 6.4", [6.1, 6.4]),
    ):
        assert [float(m.group()) for m in claims._NUMBER.finditer(text)] == expected, text

    root = study(claims, tmp_path, [("wall_shear.png", 1.02, "analyze.py")])
    (verdict,) = claims.against_answer(
        claims.collect(root),
        "The reattachment length is x_r/h = 6.4, within 15% of the benchmark.\n")
    assert verdict["answer_numbers"] == [6.4]
    assert verdict["verdict"] == "disagrees"


def test_a_percentage_is_how_far_apart_two_things_are_not_a_length(claims, tmp_path):
    """"within 5% of Gartling's benchmark 6.1" offered 5.0 as a candidate value for
    the reattachment length, so a figure stamped 5.1 -- an entirely ordinary answer
    for this quantity -- agreed with an answer that said 6.4. On the fixture that is
    the delivered text, no less."""
    root = study(claims, tmp_path, [("wall_shear.png", 5.1, "analyze.py")])
    (verdict,) = claims.against_answer(claims.collect(root), ANSWER_AS_DELIVERED)
    assert 5.0 not in verdict["answer_numbers"]
    assert verdict["verdict"] == "disagrees"


def test_the_benchmark_quoted_beside_the_result_is_allowed_to_be_the_match(claims, tmp_path):
    """Deliberately the weak form. A check that fires because the sentence also
    names Gartling's 6.1 is a check that gets switched off; the failure worth
    catching is an answer and a figure that share no number at all."""
    root = study(claims, tmp_path, [("wall_shear.png", 6.1, "reattach.py")])
    (verdict,) = claims.against_answer(claims.collect(root), ANSWER_AS_DELIVERED)
    assert verdict["verdict"] == "agrees"


# -- the command line ----------------------------------------------------------


def test_check_returns_non_zero_when_the_deliverable_contradicts_itself(claims, tmp_path, capsys):
    """It reports an exit code -- as `preflight.py` does, and unlike the scripts
    here that only ever report -- because what it reports is arithmetic: two numbers
    either differ by more than the tolerance or they do not. That is what makes it
    usable from a job rather than only by eye."""
    root = study(claims, tmp_path, [
        ("bfs_result.png", 7.5667, "analyze.py"),
        ("wall_shear.png", 6.32, "reattach.py"),
    ])
    answer = root / "answer.md"
    answer.write_text(ANSWER_AS_DELIVERED)
    assert claims.main(["check", "--root", str(root), "--answer", str(answer)]) == 1
    printed = capsys.readouterr().out
    assert "7.5667" in printed and "6.32" in printed
    assert "One of the two is wrong" in printed


def test_check_returns_zero_when_the_deliverable_hangs_together(claims, tmp_path, capsys):
    root = study(claims, tmp_path, [("wall_shear.png", 6.32, "reattach.py")])
    answer = root / "answer.md"
    answer.write_text(ANSWER_AS_DELIVERED)
    assert claims.main(["check", "--root", str(root), "--answer", str(answer)]) == 0


def test_exit_zero_gives_the_report_without_the_status(claims, tmp_path, capsys):
    root = study(claims, tmp_path, [
        ("a.png", 7.5667, "analyze.py"), ("b.png", 6.32, "reattach.py"),
    ])
    assert claims.main(["check", "--root", str(root), "--exit-zero"]) == 0
    assert "contradict" in capsys.readouterr().out


def test_attach_stamps_a_figure_somebody_else_drew(claims, png, capsys):
    """The figure in F-36 was drawn by a hand-written script, which is the normal
    case: a claim has to be attachable to a picture this toolbox did not make."""
    assert claims.main([
        "attach", str(png), "--name", "reattachment-length",
        "--symbol", "x_r/h", "--value", "6.32", "--source", "analyze.py",
    ]) == 0
    (row,) = claims.read_png(png)
    assert row["value"] == 6.32 and row["source"] == "analyze.py"


def test_json_is_the_whole_comparison_and_parses(claims, tmp_path, capsys):
    root = study(claims, tmp_path, [
        ("a.png", 7.5667, "analyze.py"), ("b.png", 6.32, "reattach.py"),
    ])
    answer = root / "answer.md"
    answer.write_text(ANSWER_AS_DELIVERED)
    claims.main(["check", "--root", str(root), "--answer", str(answer), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["claims"]) == 2
    assert len(payload["contradictions"]) == 1
    assert {entry["verdict"] for entry in payload["answer"]} == {"agrees", "disagrees"}


def test_the_stamp_is_read_and_written_with_the_standard_library_alone(tmp_path, png):
    """Stated as a source grep once, on the premise that this image has no Pillow.
    It has -- 12.2.0, under matplotlib -- so the grep was checking a false claim
    with a weak instrument. The property actually worth having is that a stamp can
    be written and read where nothing scientific is importable at all, which is what
    makes `claims.py read` usable on a laptop; so it is run that way, with every
    such module barred from the import system.

    `struct` and `zlib` also copy every chunk the encoder did not write, which no
    decode-and-re-encode library will do for you.
    """
    barred = ("PIL", "matplotlib", "numpy", "scipy", "imageio", "pyvista")

    class Barrier:
        def find_module(self, name, path=None):  # pragma: no cover - py2-era hook
            return self.find_spec(name, path)

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in barred:
                raise ImportError(f"{name} is barred for this test")
            return None

    before = {name: module for name, module in sys.modules.items()
              if name.split(".")[0] in barred or name == "claims"}
    for name in list(before):
        del sys.modules[name]
    sys.meta_path.insert(0, Barrier())
    try:
        bare = _load("claims")
        row = bare.claim("reattachment-length", 6.32, symbol="x_r", units="h")
        bare.stamp_png(png, [row])
        (back,) = bare.read_png(png)
        assert back["value"] == 6.32 and back["units"] == "h"
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(before)
