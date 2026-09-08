"""`reattach.py`: one reattachment length, and the reason it is not the other three.

In P02 the agent wrote two scripts to answer one question. `reattach.py` found the
wall-shear sign changes at 0.21, 6.32, 7.86 and 9.97 and read x_r/h = 6.32 -- right,
and 4% above Gartling's 6.1. `analyze.py`, the one that drew the delivered figure,
read 7.5667 -- 24% above it. The written answer said 6.4 and pointed the reader at
the figure. Nothing reconciled any of it, and nothing could have, because there was
no blessed implementation for either script to be checked against.

So the series 0.21 / 6.32 / 7.86 / 9.97 is the fixture almost everything here runs
on: it is the one case where the right answer, the two wrong ones a careless script
produces, and the benchmark are all known in advance. The rest is a hand-written
ASCII polyMesh with a wall shear written onto it, so what comes back out is what
was put in.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def reattach():
    spec = importlib.util.spec_from_file_location("reattach", TOOLBOX / "reattach.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- the step's own wall-shear series ------------------------------------------
#
# Sign changes exactly at the four stations P02 measured. The magnitude does not
# matter to anything here -- only where the curve crosses zero -- so it is squashed
# through tanh to keep the far field from running away.

STEP_CROSSINGS = (0.21, 6.32, 7.86, 9.97)
PRIMARY = 6.32
GARTLING = 6.1


def step_profile(dx: float = 0.01, end: float = 30.0):
    x = np.arange(0.0, end + dx / 2, dx)
    product = np.ones_like(x)
    for zero in STEP_CROSSINGS:
        product = product * (x - zero)
    return x, np.tanh(product / 50.0)


def bare_profile(x, tau, **extra):
    """The dict `report`/`plot` want, without going through a case."""
    row = {
        "x": np.asarray(x, float), "tau": np.asarray(tau, float),
        "faces": len(x), "stations": len(x), "mixed_sign_stations": 0,
        "patch": "lowerWall", "field": "wallShearStress", "time": "1000",
        "direction": "x", "case": "", "source": "",
    }
    row.update(extra)
    return row


# -- the arithmetic that was got wrong -----------------------------------------


def test_the_primary_reattachment_is_the_one_the_benchmark_matches(reattach):
    """6.32, not 7.5667. The rule is the downstream end of the LONGEST reversed-flow
    run, which is what separates the primary bubble from everything else on the
    wall without needing to be told where the step is."""
    x, tau = step_profile()
    result = reattach.measure(x, tau)
    assert result["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.02)
    assert abs(result["reattachment_scaled"] - GARTLING) / GARTLING < 0.05


def test_the_first_sign_change_is_the_corner_eddy_and_is_not_the_answer(reattach):
    """0.21 is where the counter-rotating eddy in the step corner gives the floor
    back to the recirculation. A script that stops at "the shear changes sign here"
    reports it as the reattachment; this one names it a separation and names the
    structure it closes."""
    x, tau = step_profile()
    result = reattach.measure(x, tau)
    first = result["crossings"][0]
    assert first["x"] == pytest.approx(0.21, abs=0.02)
    assert first["kind"] == "separation"
    assert result["structures"][0]["what"] == "corner eddy"
    assert result["first_crossing_scaled"] != result["reattachment_scaled"]


def test_the_last_sign_change_belongs_to_a_secondary_bubble(reattach):
    """9.97 closes a second, shorter separation downstream. Taking the last
    crossing is the other way to get a number 24% too high."""
    x, tau = step_profile()
    result = reattach.measure(x, tau)
    assert result["last_crossing_scaled"] == pytest.approx(9.97, abs=0.02)
    named = {span["what"] for span in result["structures"]}
    assert "secondary bubble" in named
    assert result["last_crossing_scaled"] != result["reattachment_scaled"]


def test_the_report_prints_the_two_answers_a_careless_script_would_give(reattach):
    """Not merely avoided -- printed. The value of this file to a study that has
    already computed the number some other way is that the other number is on the
    page with a reason beside it."""
    x, tau = step_profile()
    result = reattach.measure(x, tau)
    text = reattach.report(bare_profile(x, tau), result)
    assert "FIRST sign change would say 0.21" in text
    assert "LAST would say 9.97" in text
    assert "x_r = 6.32" in text


def test_the_report_does_not_tell_a_wall_with_one_bubble_that_its_answer_is_wrong(reattach):
    """The line above is guarded on FIRST or LAST differing from the answer, and the
    sentence it prints says "neither is the primary reattachment". On a wall with a
    single bubble the crossings are [separation, reattachment], so the LAST one IS
    the answer -- and the first version printed `x_r = 7.995` and then, two lines
    down, that a script taking the last sign change would say 7.995 and that it is
    not the primary reattachment. An agent checking its own 7.995 against that is
    steered off the right number by the tool built to stop exactly that."""
    x = np.linspace(0.0, 20.0, 2001)
    tau = np.tanh((x - 2.005) * (x - 7.995) / 10.0)
    result = reattach.measure(x, tau)
    text = reattach.report(bare_profile(x, tau), result)
    assert result["reattachment_scaled"] == pytest.approx(7.995, abs=0.02)
    assert "neither is the primary" not in text
    assert "FIRST sign change would say 2.005" in text
    assert "the LAST sign change on this wall IS the primary one" in text


def test_choosing_the_longest_run_is_stated_wherever_it_chose_something(reattach):
    """"The longest reversed run" is a rule, not a law: on a diffuser or a stalled
    aerofoil the downstream separation can be the longer one and the rule names it.
    On the step's four crossings it decided between two bubbles, so it says so;
    on a wall with one bubble there was nothing to decide and it stays quiet."""
    x, tau = step_profile()
    two = reattach.measure(x, tau)
    assert any("LONGEST" in note for note in two["notes"])

    x = np.linspace(0.0, 20.0, 2001)
    one = reattach.measure(x, np.tanh((x - 2.005) * (x - 7.995) / 10.0))
    assert not any("LONGEST" in note for note in one["notes"])


def test_a_wall_with_one_sign_on_it_says_it_cannot_tell_rather_than_no_separation(reattach):
    """A wall lying entirely inside a separation is byte-for-byte identical to a wall
    the flow never left: one sign everywhere. The length vote picks the majority, the
    majority is the only sign there is, and the first version returned "nothing
    separates" about a wall that is 100% reversed -- with no note, because both the
    open-start note and the tail-sign note are downstream of that return."""
    x = np.linspace(0.0, 10.0, 101)
    result = reattach.measure(x, np.full_like(x, -0.4))
    assert result["separated"] is False
    assert any("cannot tell attached from reversed" in note for note in result["notes"])
    # Given the sign, there is nothing to hedge about: it is attached, and quietly.
    told = reattach.measure(x, np.full_like(x, -0.4), forward=-1)
    assert not any("cannot tell" in note for note in told["notes"])


def test_all_four_sign_changes_are_reported_not_only_the_chosen_one(reattach):
    """The list is the evidence. Reporting 6.32 alone would make this script one
    more thing to be believed rather than checked."""
    x, tau = step_profile()
    result = reattach.measure(x, tau)
    found = [point["x"] for point in result["crossings"]]
    assert len(found) == 4
    for expected, got in zip(STEP_CROSSINGS, found):
        assert got == pytest.approx(expected, abs=0.02)


def test_the_answer_does_not_depend_on_the_wall_shear_sign_convention(reattach):
    """OpenFOAM's `wallShearStress` sign for forward flow is a convention of the
    function object, and a hand-computed `nu dU/dy` has the other one. Attached is
    taken to be whichever sign covers more of the wall, so flipping the whole
    profile has to leave the number alone -- otherwise every study would have to
    get the convention right to get the physics right."""
    x, tau = step_profile()
    forward = reattach.measure(x, tau)
    flipped = reattach.measure(x, -tau)
    assert flipped["reattachment_scaled"] == pytest.approx(forward["reattachment_scaled"])
    assert flipped["forward_sign"] == -forward["forward_sign"]


def test_a_forward_sign_given_by_hand_overrides_the_vote(reattach):
    x, tau = step_profile()
    result = reattach.measure(x, tau, forward=-1)
    assert result["forward_basis"] == "given on the command line"
    assert result["reattachment_scaled"] != pytest.approx(PRIMARY, abs=0.02)


def test_a_bubble_still_reversed_at_the_last_face_reports_no_length(reattach):
    """A wall too short to hold the reattachment does not hold the answer either.
    Reporting the end of the patch as the reattachment is how a domain-length
    mistake becomes a physics result."""
    x, tau = step_profile(end=4.0)
    result = reattach.measure(x, tau, forward=1)
    assert result["separated"] is True
    assert result["reattachment_scaled"] is None
    assert any("does not close" in note for note in result["notes"])


def test_a_wall_that_is_mostly_reversed_says_the_sign_vote_may_be_backwards(reattach):
    """The one case the length vote gets wrong, and it must not get it wrong
    quietly. On a wall cut off at 4 h the bubble IS the majority, so "attached is
    the sign covering more of the wall" calls the bubble attached and hands back
    the corner eddy at 0.21 -- the exact 0.21-for-6.32 substitution F-36 turned on.
    With no velocity field to settle it, the note is what stands between that and a
    published number."""
    x, tau = step_profile(end=4.0)
    result = reattach.measure(x, tau)
    assert result["reattachment_scaled"] == pytest.approx(0.21, abs=0.02)
    assert any("wrong way round" in note for note in result["notes"])


def test_a_wall_with_no_reversed_flow_says_so_rather_than_returning_zero(reattach):
    x = np.linspace(0.0, 30.0, 300)
    result = reattach.measure(x, np.full_like(x, 0.4))
    assert result["separated"] is False
    assert result["reattachment_scaled"] is None
    assert any("nothing separates" in note for note in result["notes"])


def test_the_number_is_quoted_in_step_heights_when_a_height_is_given(reattach):
    """x_r/h, which is the form every benchmark for this case is written in. The
    origin moves with it, because the step face is not usually at x = 0."""
    x, tau = step_profile()
    result = reattach.measure(x * 0.0094 + 0.5, tau, origin=0.5, height=0.0094)
    assert result["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.02)


def test_the_reported_resolution_is_half_the_spacing_of_the_faces_it_sits_between(reattach):
    """The crossing was interpolated between two faces and could have been anywhere
    between them. A wall sampled every 0.5 h does not support a four-figure answer,
    and `analyze.py` published one."""
    x, tau = step_profile(dx=0.5)
    result = reattach.measure(x, tau)
    assert result["resolution"] == pytest.approx(0.25, rel=1e-6)


def test_the_error_bar_and_the_spacing_in_the_table_are_the_same_quantity(reattach):
    """One run, one spacing, two numbers -- inside the report that argues against
    exactly that. `crossings` records `resolution` in the mesh's own coordinate and
    `measure` divides only its own copy by --height, so the `## sign changes` table
    printed the raw spacing beside an `x` column already in step heights: a headline
    of `+/- 0.025 (half the local face spacing)` over a row reading 0.00047, which
    differ by the height and said nothing about it. A height of 1 cannot see this,
    which is why the assertion is made at 0.0094."""
    x, tau = step_profile(dx=0.5)
    result = reattach.measure(x * 0.0094, tau, height=0.0094)
    profile = bare_profile(x * 0.0094, tau)
    text = reattach.report(profile, result)

    lines = text.splitlines()
    start = lines.index(next(line for line in lines if line.startswith("## sign changes")))
    rows = []
    for line in lines[start + 2:]:
        if not line.strip():
            break
        rows.append(line)
    assert rows, text
    for line in rows:
        assert float(line.split()[-1]) == pytest.approx(2.0 * result["resolution"], rel=1e-3)
    assert f"+/- {result['resolution']:.3g}" in text
    assert "(both columns in h)" in text
    # The raw, unscaled spacing (0.0047 mesh units) must not appear anywhere in it.
    assert "0.0047" not in text


def test_the_claim_is_rounded_to_what_the_wall_can_actually_see(reattach):
    """7.5667 asserts the reattachment is known to a ten-thousandth of a step
    height. On a wall whose faces are half a step height apart it is known to
    about a quarter of one, and the claim says so by not writing the digits."""
    x, tau = step_profile(dx=0.5)
    result = reattach.measure(x, tau)
    row = reattach.as_claim(bare_profile(x, tau), result)
    assert row["value"] == round(row["value"], 1)
    assert reattach.digits_for(0.005) == 3
    assert reattach.digits_for(0.25) == 1


def test_an_unknown_face_spacing_buys_no_decimals_at_all(reattach):
    """The first version returned six decimals when the spacing was zero or not
    finite -- a millionth of a step height asserted in precisely the case where
    nothing had established the spacing. That is this file's own rule inverted, so
    the answer is None and the value goes out unrounded and says it has no bound."""
    assert reattach.digits_for(0.0) is None
    assert reattach.digits_for(float("nan")) is None
    assert reattach.digits_for(None) is None

    x, tau = step_profile(dx=0.5)
    result = reattach.measure(x, tau)
    result["resolution"] = 0.0
    row = reattach.as_claim(bare_profile(x, tau), result)
    assert row["value"] == result["reattachment_scaled"]
    assert "neither rounded nor bounded" in row["note"]
    assert "no error bar" in reattach.report(bare_profile(x, tau), result)


def test_the_report_and_the_stamp_quote_the_same_number(reattach):
    """They were computed apart once: the report printed `:.4g` of the unrounded
    value and the figure carried the rounded claim, so one run said 6.324 in text
    and 6.32 on the picture. 0.06% and inside every tolerance, and still two numbers
    for one quantity in one session, which is the sentence this file was written
    under."""
    x, tau = step_profile()
    result = reattach.measure(x, tau)
    profile = bare_profile(x, tau)
    row = reattach.as_claim(profile, result)
    printed = [line for line in reattach.report(profile, result).splitlines()
               if line.startswith("x_r =")]
    assert len(printed) == 1
    (number,) = re.findall(r"[-+]?\d*\.?\d+", printed[0].split("+/-")[0])
    assert float(number) == row["value"]


def test_a_number_this_script_cannot_name_a_unit_for_is_not_called_metres(reattach):
    """Nothing here reads the mesh's length unit, so nothing here may assert one.
    Two shipped mistakes, both of them F-36's own shape. `--from-file` on a table
    already in x/h, with no --height, was stamped `units="m"` -- which then made
    `claims.py` skip the comparison against a hand-attached claim as a unit
    mismatch, and print its all-clear over the F-36 pair. And `--origin` with no
    --height stamped the caption `x_r/h` on a metres-valued number while the report
    line beside it read `x_r = 0.05941 m`: one number, two units, one run."""
    x, tau = step_profile()
    profile = bare_profile(x, tau)

    plain = reattach.as_claim(profile, reattach.measure(x, tau))
    assert plain["units"] == "" and plain["symbol"] == "x_r"

    shifted = reattach.measure(x * 0.0094, tau, origin=0.0047)
    row = reattach.as_claim(profile, shifted)
    assert row["units"] == "" and row["symbol"] == "x_r"
    assert " m" not in reattach.report(profile, shifted)

    scaled = reattach.measure(x * 0.0094 + 0.5, tau, origin=0.5, height=0.0094)
    high = reattach.as_claim(profile, scaled)
    assert high["units"] == "h" and high["symbol"] == "x_r"
    assert "x_r = 6.32 h" in reattach.report(profile, scaled)

    # And when the operator knows what this cannot read, they can say it, once, and
    # the report, the caption and the stamp all say it.
    told = reattach.as_claim(profile, reattach.measure(x, tau), "h")
    assert told["units"] == "h" and told["symbol"] == "x_r"
    assert "x_r = 6.32 h" in reattach.report(profile, reattach.measure(x, tau), "h")


def test_the_report_and_the_stamp_use_one_symbol_and_do_not_divide_by_h_twice(reattach):
    """A third version of the same drift, and it shipped: `report` wrote the symbol
    out as a literal `x_r` while `as_claim` wrote `x_r/h` whenever the unit was "h",
    so one run printed `x_r = 6.32 h` and stamped `x_r/h = 6.32 h` into the figure
    beside it -- two symbols for one number, and the stamped one a ratio that has
    already divided by h with h re-attached as its unit. The symbol names the
    quantity; the unit label carries the step height, and it is the label `claims.py`
    compares on."""
    import claims

    x, tau = step_profile()
    profile = bare_profile(x, tau)
    scaled = reattach.measure(x * 0.0094 + 0.5, tau, origin=0.5, height=0.0094)
    row = reattach.as_claim(profile, scaled)

    assert claims.caption(row) == "x_r = 6.32 h"
    assert row["caption"] == claims.caption(row)
    assert "x_r/h" not in reattach.report(profile, scaled)
    # The report's headline is the caption, word for word.
    assert any(line.startswith(claims.caption(row))
               for line in reattach.report(profile, scaled).splitlines())

    # And the ratio spelling an answer is likely to use is still searched for, which
    # is what the symbol was carrying it for.
    assert "x_r/h" in claims.searchable(row)
    (verdict,) = claims.against_answer(
        [row], "The reattachment length is x_r/h = 6.32 in this run.\n")
    assert verdict["verdict"] == "agrees"


def test_a_claim_this_script_made_is_actually_compared_with_a_hand_attached_one(reattach):
    """The F-36 replay, across both files, which is the pair that matters and the
    one nothing tested. `reattach.py` measures 6.32; a person stamps the figure that
    `analyze.py` drew with `claims.py attach --value 7.5667`, which takes no --units
    and defaults to the empty label, exactly as this module's own documented example
    writes it. The first version labelled its own number "m", `contradictions`
    dropped the pair, and `check` printed "no two claims of the same name disagree"
    about F-36 itself."""
    import claims  # loaded as a sibling by reattach's own sys.path insert

    x, tau = step_profile()
    mine = reattach.as_claim(bare_profile(x, tau), reattach.measure(x, tau))
    theirs = claims.claim("reattachment-length", 7.5667, symbol="x_r/h",
                          source="analyze.py")
    (clash,) = claims.contradictions([mine, theirs])
    assert clash["comparable"] is True
    assert clash["apart"] > 0.15
    text = claims.report([mine, theirs], [clash], [], None, claims.DEFAULT_TOL)
    assert "no two claims of the same name disagree" not in text
    assert "16% apart" in text


# -- stations, and a reattachment line that is not a point ---------------------


def test_faces_at_one_streamwise_station_become_one_reading(reattach):
    """A 2D case is one cell thick and has two floor faces per station; a 3D floor
    has a whole span of them. Both have to come out as one curve."""
    x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    tau = np.array([1.0, 3.0, -1.0, -3.0, 2.0, 4.0])
    grouped = reattach.stations(x, tau)
    assert grouped["stations"] == 3
    assert list(grouped["tau"]) == [2.0, -2.0, 3.0]
    assert grouped["faces"] == 6


def test_a_reattachment_line_that_wanders_across_the_span_is_flagged(reattach):
    """On a genuinely three-dimensional floor the reattachment is a line, and a
    span-averaged scalar quietly stands in for it. Counting the stations that carry
    both signs is what stops that from being invisible -- the ONERA M6 failure was
    a picture read as showing something it did not, and this is the same class of
    thing seen from the numeric side."""
    x = np.array([0.0, 0.0, 1.0, 1.0])
    tau = np.array([1.0, -1.0, 2.0, 2.0])
    grouped = reattach.stations(x, tau)
    assert grouped["mixed_sign_stations"] == 1
    text = reattach.report(bare_profile(grouped["x"], grouped["tau"], **{
        k: grouped[k] for k in ("faces", "stations", "mixed_sign_stations")
    }), reattach.measure(grouped["x"], grouped["tau"]))
    assert "BOTH" in text and "a line, not a point" in text


def test_a_station_that_reads_exactly_zero_is_not_a_fifth_sign_change(reattach):
    """A face whose shear is zero to machine precision is between two signs, not a
    structure of its own, and counting it would put an extra crossing in a list the
    whole method is built on."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    tau = np.array([1.0, 0.0, -1.0, -1.0])
    assert len(reattach.crossings(x, tau)) == 1


# -- reading it off a case -----------------------------------------------------


def foam(cls: str, obj: str, body: str, location: str = "constant/polyMesh") -> str:
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    location    \"{location}\";\n"
        f"    object      {obj};\n}}\n\n" + body + "\n"
    )


def write_channel(
    case: Path, xs, tau_of_x, *, ny: int = 3, convention: int = 1, bulk: float = 1.0,
    near_wall=None, ys=(0.0, 1.0), z=(0.0, 0.1),
) -> Path:
    """A structured `nx x ny` channel with a `lowerWall`, a wall shear on it, and a
    velocity field consistent with both.

    The faces on `lowerWall` are written in the order `startFace ..
    startFace + nFaces`, which is the order the field's `value` list is in, so what
    the script pairs up is what this function paired up, and `tau_of_x` is evaluated
    at each floor face centre -- the crossings are put in by hand and the script has
    to find them again.

    `ny` is at least two rows because the velocity that settles the sign convention
    is not read off the wall: the near-wall row carries `tau_of_x` as the physical
    near-wall velocity and the rows above it carry `bulk`, which is what a domain
    mean is supposed to point with. `convention` multiplies the WRITTEN WALL SHEAR
    and nothing else, which is what a sign convention is -- the flow is the same
    flow, and one function object writes it one way round and another the other.

    `near_wall` writes that bottom row from a different function than the shear.
    It defaults to `tau_of_x`, and while that was the ONLY way this fixture could be
    written the near-wall row was `tau` exactly, so `sum(near * tau)` -- the vote
    `forward_sign_from_velocity` takes -- was `convention * sum(tau**2)` and could
    not have come out any other way. Every test of that vote was therefore a test of
    an identity. Real near-wall velocity only tracks tau; it does not equal it.
    """
    near_wall = tau_of_x if near_wall is None else near_wall
    xs = np.asarray(xs, float)
    nx = xs.size - 1
    ys = np.linspace(ys[0], ys[1], ny + 1)
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)

    def pt(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    def cell(i, j):
        return i + nx * j

    points = [
        f"({xs[i]:.10g} {ys[j]:.10g} {z[k]:.10g})"
        for k in (0, 1) for j in range(ny + 1) for i in range(nx + 1)
    ]
    (poly / "points").write_text(
        foam("vectorField", "points", f"{len(points)}\n(\n" + "\n".join(points) + "\n)\n")
    )

    faces: list[str] = []
    owner: list[int] = []
    neighbour: list[int] = []
    for j in range(ny):  # internal, normal +x
        for i in range(1, nx):
            faces.append(f"4({pt(i,j,0)} {pt(i,j+1,0)} {pt(i,j+1,1)} {pt(i,j,1)})")
            owner.append(cell(i - 1, j))
            neighbour.append(cell(i, j))
    for j in range(1, ny):  # internal, normal +y
        for i in range(nx):
            faces.append(f"4({pt(i,j,0)} {pt(i,j,1)} {pt(i+1,j,1)} {pt(i+1,j,0)})")
            owner.append(cell(i, j - 1))
            neighbour.append(cell(i, j))

    patches: list[tuple[str, str, int, int]] = []

    start = len(faces)
    for i in range(nx):  # lowerWall, normal -y
        faces.append(f"4({pt(i,0,0)} {pt(i+1,0,0)} {pt(i+1,0,1)} {pt(i,0,1)})")
        owner.append(cell(i, 0))
    patches.append(("lowerWall", "wall", start, nx))

    start = len(faces)
    for i in range(nx):  # upperWall, normal +y
        faces.append(f"4({pt(i,ny,0)} {pt(i,ny,1)} {pt(i+1,ny,1)} {pt(i+1,ny,0)})")
        owner.append(cell(i, ny - 1))
    patches.append(("upperWall", "wall", start, nx))

    start = len(faces)
    for j in range(ny):
        faces.append(f"4({pt(0,j,0)} {pt(0,j,1)} {pt(0,j+1,1)} {pt(0,j+1,0)})")
        owner.append(cell(0, j))
    patches.append(("inlet", "patch", start, ny))

    start = len(faces)
    for j in range(ny):
        faces.append(f"4({pt(nx,j,0)} {pt(nx,j+1,0)} {pt(nx,j+1,1)} {pt(nx,j,1)})")
        owner.append(cell(nx - 1, j))
    patches.append(("outlet", "patch", start, ny))

    start = len(faces)
    for j in range(ny):
        for i in range(nx):
            faces.append(f"4({pt(i,j,0)} {pt(i,j+1,0)} {pt(i+1,j+1,0)} {pt(i+1,j,0)})")
            owner.append(cell(i, j))
    for j in range(ny):
        for i in range(nx):
            faces.append(f"4({pt(i,j,1)} {pt(i+1,j,1)} {pt(i+1,j+1,1)} {pt(i,j+1,1)})")
            owner.append(cell(i, j))
    patches.append(("frontAndBack", "empty", start, 2 * nx * ny))

    (poly / "faces").write_text(
        foam("faceList", "faces", f"{len(faces)}\n(\n" + "\n".join(faces) + "\n)\n")
    )
    (poly / "owner").write_text(
        foam("labelList", "owner",
             f"{len(owner)}\n(\n" + "\n".join(str(v) for v in owner) + "\n)\n")
    )
    (poly / "neighbour").write_text(
        foam("labelList", "neighbour",
             f"{len(neighbour)}\n(\n" + "\n".join(str(v) for v in neighbour) + "\n)\n")
    )
    blocks = "\n".join(
        f"    {name}\n    {{\n        type            {kind};\n"
        f"        nFaces          {count};\n        startFace       {first};\n    }}"
        for name, kind, first, count in patches
    )
    (poly / "boundary").write_text(
        foam("polyBoundaryMesh", "boundary", f"{len(patches)}\n(\n{blocks}\n)\n")
    )

    time = case / "1000"
    time.mkdir(parents=True, exist_ok=True)

    centres = 0.5 * (xs[:-1] + xs[1:])
    shear = "\n".join(f"({convention * tau_of_x(c):.10g} 0 0)" for c in centres)
    (time / "wallShearStress").write_text(foam(
        "volVectorField", "wallShearStress",
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    lowerWall\n    {\n        type            calculated;\n"
        f"        value           nonuniform List<vector>\n{nx}\n(\n{shear}\n)\n;\n    }}\n"
        "    upperWall\n    {\n        type            calculated;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    inlet\n    {\n        type            calculated;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    outlet\n    {\n        type            calculated;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n}\n",
        location="1000",
    ))

    speeds = []
    for j in range(ny):
        for value in centres:
            speeds.append(near_wall(value) if j == 0 else bulk)
    velocity = "\n".join(f"({value:.10g} 0 0)" for value in speeds)
    (time / "U").write_text(foam(
        "volVectorField", "U",
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        f"internalField   nonuniform List<vector>\n{len(speeds)}\n(\n{velocity}\n)\n;\n\n"
        "boundaryField\n{\n"
        "    lowerWall\n    {\n        type            noSlip;\n    }\n"
        "    upperWall\n    {\n        type            noSlip;\n    }\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n    }\n}\n",
        location="1000",
    ))
    return case


def step_shear(x: float) -> float:
    product = 1.0
    for zero in STEP_CROSSINGS:
        product *= x - zero
    return float(np.tanh(product / 50.0))


@pytest.fixture
def bfs_case(tmp_path):
    return write_channel(tmp_path / "bfs", np.linspace(0.0, 30.0, 601), step_shear)


def test_it_reads_the_wall_shear_a_solver_wrote_and_finds_the_step_answer(reattach, bfs_case):
    """End to end off a case: polyMesh face centres paired face-for-face with the
    patch's `value` list, which is the pairing that makes the x coordinate mean
    anything."""
    profile = reattach.wall_profile(bfs_case, "lowerWall")
    result = reattach.measure(profile["x"], profile["tau"])
    assert profile["stations"] == 600
    assert profile["time"] == "1000"
    assert result["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.05)


def test_the_sign_convention_is_measured_off_the_velocity_not_assumed(reattach, tmp_path):
    """The same wall shear with the sign relation inverted -- which is the
    difference between `wallShearStress` and a hand-computed `nu dU/dy`, and the
    reason wrong-signed wall plots keep appearing -- has to give the same
    reattachment. Nothing here knows which way round OpenFOAM writes it, and
    nothing should have to."""
    xs = np.linspace(0.0, 30.0, 601)
    same = reattach.wall_profile(
        write_channel(tmp_path / "same", xs, step_shear, convention=1), "lowerWall")
    flipped = reattach.wall_profile(
        write_channel(tmp_path / "flip", xs, step_shear, convention=-1), "lowerWall")
    assert same["forward"] == 1 and flipped["forward"] == -1
    for profile in (same, flipped):
        result = reattach.measure(profile["x"], profile["tau"], forward=profile["forward"])
        assert result["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.05)


def test_the_velocity_settles_a_wall_the_length_vote_reads_backwards(reattach, tmp_path):
    """A wall cut off at 4 h is mostly bubble, so the length vote calls the bubble
    attached and returns the corner eddy. The velocity field answers it properly
    because it asks two separate questions: the near-wall velocity along this wall
    gives the sign relation, and the mean over the WHOLE domain -- which the
    truncation does not touch, since the rows above the wall are still going
    downstream -- gives which way downstream is."""
    case = write_channel(tmp_path / "short", np.linspace(0.0, 4.0, 81), step_shear)
    profile = reattach.wall_profile(case, "lowerWall")
    assert profile["forward"] == 1
    voted = reattach.measure(profile["x"], profile["tau"])
    measured = reattach.measure(profile["x"], profile["tau"], forward=profile["forward"])
    assert voted["reattachment_scaled"] == pytest.approx(0.21, abs=0.05)
    assert measured["reattachment_scaled"] is None
    assert any("does not close" in note for note in measured["notes"])


def test_the_sign_relation_holds_when_tau_and_the_velocity_only_track_each_other(
    reattach, tmp_path
):
    """The vote is `sum(near_wall_U * tau)` over the patch, and until this fixture
    could write the two from different functions the near-wall row WAS tau, so that
    sum was `sum(tau**2)` and every test of the vote was a test of an identity.

    Here the magnitudes are unrelated to tau's and one face in seven disagrees in
    sign outright -- which is what a real near-wall row looks like, since tau tracks
    U up to a coefficient that varies along the wall and both are noisy near a
    crossing. The vote is a sum over every face precisely so that a minority of
    dissenters cannot carry it, and the answer has to be the same 6.32.
    """
    def tracks_it(value: float) -> float:
        with_it = 1.0 if step_shear(value) >= 0 else -1.0
        dissent = -1.0 if int(value * 20.0) % 7 == 0 else 1.0
        return dissent * with_it * (0.2 + abs(float(np.sin(value))))

    case = write_channel(tmp_path / "tracking", np.linspace(0.0, 30.0, 601),
                         step_shear, near_wall=tracks_it)
    profile = reattach.wall_profile(case, "lowerWall")
    assert profile["forward"] == 1
    assert "weak" not in profile["forward_basis"]
    result = reattach.measure(profile["x"], profile["tau"], forward=profile["forward"])
    assert result["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.05)


def test_a_vote_that_is_nearly_a_coin_toss_is_reported_as_weak(reattach, tmp_path):
    """A sum can be carried by a few large faces while most faces disagree, and then
    the "measured off the velocity" basis is a coin toss wearing a measurement's
    clothes -- a `U` from a different time, or a wall the flow crosses rather than
    runs along. Here 45% of the faces point the other way and are small enough that
    the sum still lands on the right sign, so the number is right and the basis is
    not trustworthy, which the report has to distinguish from the ordinary case
    where the two agree at nearly every face.
    """
    def barely(value: float) -> float:
        with_it = 1.0 if step_shear(value) >= 0 else -1.0
        return -0.01 * with_it if int(value * 20.0) % 20 < 9 else with_it

    case = write_channel(tmp_path / "cointoss", np.linspace(0.0, 30.0, 601),
                         step_shear, near_wall=barely)
    profile = reattach.wall_profile(case, "lowerWall")
    assert profile["forward"] == 1
    assert "weak" in profile["forward_basis"]
    assert "--forward-sign" in profile["forward_basis"]
    text = reattach.report(bare_profile(profile["x"], profile["tau"]),
                           reattach.measure(profile["x"], profile["tau"],
                                            forward=profile["forward"],
                                            basis=profile["forward_basis"]))
    assert "weak" in text


def test_a_case_with_no_velocity_written_falls_back_and_says_so(reattach, bfs_case):
    """`--from-file`, or a `U` that was never written. The fallback is the length
    vote, and the basis it reports has to say that is what happened."""
    (bfs_case / "1000" / "U").unlink()
    profile = reattach.wall_profile(bfs_case, "lowerWall")
    assert profile["forward"] is None
    result = reattach.measure(profile["x"], profile["tau"])
    assert "covering" in result["forward_basis"]
    assert result["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.05)


def test_a_domain_with_no_net_streamwise_flow_declines_to_point(reattach, tmp_path):
    """A closed cavity has no downstream. Reading a direction out of that mean
    would be reading it out of noise, so the question is handed back to the length
    vote rather than answered."""
    def circulating(value: float) -> float:
        return float(np.sin(2.0 * np.pi * value / 30.0))

    case = write_channel(tmp_path / "cavity", np.linspace(0.0, 30.0, 121),
                         circulating, bulk=0.0)
    profile = reattach.wall_profile(case, "lowerWall")
    assert profile["forward"] is None


def test_a_patch_that_is_not_in_the_mesh_is_a_sentence_and_the_others_are_named(
    reattach, bfs_case
):
    with pytest.raises(reattach.Unmeasurable) as raised:
        reattach.wall_profile(bfs_case, "floor")
    assert "lowerWall" in str(raised.value)


def test_a_patch_carrying_a_uniform_value_is_no_data_rather_than_no_separation(
    reattach, bfs_case
):
    """`upperWall` in the fixture has `value uniform (0 0 0)` -- the field before
    anything was computed on it. Reading that as "this wall never separates" would
    be a confident answer about a wall nobody measured."""
    with pytest.raises(reattach.Unmeasurable) as raised:
        reattach.wall_profile(bfs_case, "upperWall")
    assert "uniform" in str(raised.value)


def test_a_case_with_no_wall_shear_written_says_how_to_get_some(reattach, tmp_path):
    case = write_channel(tmp_path / "bare", np.linspace(0.0, 30.0, 61), step_shear)
    (case / "1000" / "wallShearStress").unlink()
    with pytest.raises(reattach.Unmeasurable) as raised:
        reattach.wall_profile(case, "lowerWall")
    assert "wallShearStress function object" in str(raised.value)


def test_an_unmeasurable_case_exits_zero_and_says_what_is_missing(reattach, tmp_path, capsys):
    """Same contract as `layer_report.py` and `locate.py`: not being able to measure
    something is a reading, and a reading is what this offers."""
    assert reattach.main([str(tmp_path / "nothing"), "--patch", "lowerWall"]) == 0
    assert "not measured:" in capsys.readouterr().out


# -- reading it out of a table somebody else wrote -----------------------------


def test_a_sampled_raw_surface_goes_through_the_same_arithmetic(reattach, tmp_path):
    """`--from-file` exists so a number produced another way can be run through
    this rather than argued with. The default columns are the layout of an
    OpenFOAM `.raw` sampled surface, `x y z v_x v_y v_z`."""
    raw = tmp_path / "wallShearStress_lowerWall.raw"
    lines = ["# x  y  z  tau_x  tau_y  tau_z"]
    for value in np.linspace(0.0, 30.0, 601):
        lines.append(f"{value:.6g} 0 0 {step_shear(value):.6g} 0 0")
    raw.write_text("\n".join(lines) + "\n")
    profile = reattach.profile_from_file(raw, (0, 3))
    result = reattach.measure(profile["x"], profile["tau"])
    assert result["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.02)


def test_a_table_with_nothing_usable_in_it_is_reported(reattach, tmp_path):
    empty = tmp_path / "nothing.raw"
    empty.write_text("# header only\n")
    with pytest.raises(reattach.Unmeasurable):
        reattach.profile_from_file(empty, (0, 3))


# -- the figure carries the number ---------------------------------------------


def test_the_figure_carries_the_number_it_is_annotated_with(reattach, tmp_path):
    """The whole point of F-36. `analyze.py` drew a green line labelled 7.57 and
    printed 7.5667; the answer said 6.4; nothing tied any of them together. Here
    the annotation and the stamp are one call, and the stamp is machine-readable,
    so `claims.py check` can put the figure next to the answer."""
    import claims  # loaded as a sibling by reattach's own sys.path insert

    x, tau = step_profile()
    result = reattach.measure(x, tau)
    profile = bare_profile(x, tau)
    row = reattach.as_claim(profile, result)
    out = reattach.plot(profile, result, tmp_path / "bfs_result.png", row)
    claims.stamp_png(out, [row])

    (carried,) = claims.read_png(out)
    assert carried["value"] == pytest.approx(result["reattachment_scaled"], abs=0.01)
    drawn = claims.caption(carried)
    (number,) = re.findall(r"[-+]?\d*\.?\d+", drawn)
    assert float(number) == carried["value"]


def test_running_it_over_a_case_leaves_a_stamped_figure_and_a_manifest_line(
    reattach, bfs_case, capsys
):
    """The default path, so that reconciling costs nothing later: draw the figure
    and the number is in it and in the study's manifest without being asked for."""
    import claims

    (bfs_case / ".reynolds").mkdir()
    out = bfs_case / "reattachment.png"
    assert reattach.main([
        str(bfs_case), "--patch", "lowerWall", "--plot", str(out), "--height", "1",
    ]) == 0
    printed = capsys.readouterr().out
    assert "x_r = 6.3" in printed
    (carried,) = claims.read_png(out)
    assert carried["value"] == pytest.approx(PRIMARY, abs=0.05)
    assert [row["value"] for row in claims.collect(bfs_case)] == [carried["value"]]


def test_the_source_of_the_number_is_part_of_the_claim(reattach):
    """Two scripts producing one quantity is the whole incident. A claim that
    cannot say which one produced it cannot be adjudicated."""
    x, tau = step_profile()
    row = reattach.as_claim(bare_profile(x, tau), reattach.measure(x, tau))
    assert "reattach.py" in row["source"]
    assert "lowerWall" in row["source"]


def test_no_figure_is_stamped_when_there_is_no_number_to_stamp(reattach, tmp_path):
    """A wall that does not reattach gets a plot and no claim, rather than a claim
    of None that something downstream would have to guess at."""
    x, tau = step_profile(end=4.0)
    result = reattach.measure(x, tau, forward=1)
    assert reattach.as_claim(bare_profile(x, tau), result) is None


def test_json_is_the_whole_measurement_and_parses(reattach, bfs_case, capsys):
    assert reattach.main([str(bfs_case), "--patch", "lowerWall", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["result"]["crossings"]) == 4
    assert payload["claim"]["name"] == "reattachment-length"
    assert payload["result"]["reattachment_scaled"] == pytest.approx(PRIMARY, abs=0.05)


def test_the_json_result_carries_the_same_keys_whether_or_not_it_found_a_length(reattach):
    """`first_crossing_scaled` and `last_crossing_scaled` were written only on the
    path that found a reattachment, so the `result` object in `--json` grew and lost
    keys between runs and a consumer reading them back got a KeyError instead of a
    null. A schema that varies with the answer is a schema nobody can read twice."""
    x, tau = step_profile()
    found = reattach.measure(x, tau)
    short, short_tau = step_profile(end=4.0)
    open_bubble = reattach.measure(short, short_tau, forward=1)
    flat = reattach.measure(x, np.full_like(x, 0.4))

    assert open_bubble["reattachment_scaled"] is None and flat["separated"] is False
    assert set(found) == set(open_bubble) == set(flat)
    for absent in (open_bubble, flat):
        assert absent["first_crossing_scaled"] is None
        assert absent["last_crossing_scaled"] is None


# -- the contract this script keeps with the rest of the toolbox ---------------


def test_it_reuses_the_mesh_readers_rather_than_writing_a_third_set(reattach):
    """`layer_report.py` computes face centres by OpenFOAM's own decomposition and
    `locate.py` reads a polyMesh in either format. A second reader here would be a
    second place for a mesh to be read slightly differently, which is the shape of
    the bug this file exists to close."""
    source = (TOOLBOX / "reattach.py").read_text()
    assert "import layer_report" in source and "import locate" in source
    assert "def read_points" not in source and "def read_faces" not in source


def test_it_needs_nothing_the_image_does_not_have(reattach):
    source = (TOOLBOX / "reattach.py").read_text()
    for absent in ("import scipy", "import shapely", "from scipy", "import fitz"):
        assert absent not in source


def test_it_edits_nothing_in_the_case(reattach, bfs_case, capsys):
    """Asserted as a source grep once -- `"subprocess" not in source` and no
    `write_text` -- which would have passed just as happily for `open(p, "w")`, and
    was over-broad besides: with `--plot` this script does write, the PNG and the
    manifest line. So the property is checked where it lives. Every file the case
    had before the run is byte-for-byte the same after it, and the only new paths
    are the figure that was asked for and the study state under `.reynolds/`.
    """
    (bfs_case / ".reynolds").mkdir()
    before = {path: path.read_bytes() for path in sorted(bfs_case.rglob("*"))
              if path.is_file()}

    out = bfs_case / "reattachment.png"
    assert reattach.main([
        str(bfs_case), "--patch", "lowerWall", "--plot", str(out), "--height", "1",
    ]) == 0
    capsys.readouterr()

    after = {path: path.read_bytes() for path in sorted(bfs_case.rglob("*"))
             if path.is_file()}
    for path, content in before.items():
        assert after[path] == content, f"{path} was edited"
    new = set(after) - set(before)
    assert out in new
    assert all(path == out or ".reynolds" in path.parts for path in new), sorted(new)
