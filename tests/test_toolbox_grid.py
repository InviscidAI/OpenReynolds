"""`grid.py` -- grid convergence, Richardson extrapolation and the GCI.

Pure arithmetic on numbers, no OpenFOAM and no solver, which is the point: the whole
reason this file exists is that the four lines of Richardson extrapolation are easy to
write and easy to write wrongly, and the ways they mislead are silent. So the first
test checks the arithmetic against a published worked example from outside this
repository, and the rest check each refusal the script is supposed to make rather than
each number it is supposed to produce.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grid():
    return load("grid")


def ladder(grid, *pairs):
    """`(h, value)` pairs as the grid list `analyse` takes."""
    return [{"h": h, "value": value, "label": f"h={h:g}"} for h, value in pairs]


# -- the published example -----------------------------------------------------


def test_the_nasa_glenn_worked_example(grid):
    """NASA Glenn Research Center, "Examining Spatial (Grid) Convergence",
    https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html, which works
    Roache's GCI through on the pressure recovery of an inlet over three grids at a
    constant refinement ratio of 2.

    Its published numbers, which this reproduces:

        f1 = 0.97050 (fine), f2 = 0.96854, f3 = 0.96178 (coarse), r = 2
        p       = 1.786170   (its VERIFY program prints 1.78618479)
        f_exact = 0.97130    (VERIFY: 0.971300304)
        GCI_12  = 0.103083%  (VERIFY: 0.103080), Fs = 1.25
        GCI_23  = 0.356249%  (VERIFY: 0.356244)
        asymptotic range check = 1.002019

    The page's hand-worked figures and its VERIFY output differ in their last digits,
    because the hand arithmetic rounds p to six places first. This is checked against
    the hand-worked column at the precision that column gives, which is also the tighter
    of the two tests: agreeing with the page's own printed digits is the thing a reader
    of the page can verify.

    The ladder is given in the page's own normalised spacings (1, 2, 4) rather than in
    cell counts, which the page does not publish. That is not a shortcut: every formula
    in `grid.py` uses only ratios of h, so a ladder given as spacings and the same
    ladder given as cell counts are the same input, which `test_cell_counts_and_spacings_are_the_same_ladder`
    checks directly.
    """
    analysis = grid.analyse(ladder(grid, (1.0, 0.97050), (2.0, 0.96854), (4.0, 0.96178)))
    primary = analysis["primary"]

    assert primary["mode"] == "three-grid"
    assert primary["convergence"] == "monotone"
    assert primary["ratio_21"] == pytest.approx(2.0)
    assert primary["ratio_32"] == pytest.approx(2.0)

    assert primary["order_observed"] == pytest.approx(1.786170, abs=5e-7)
    assert primary["order_usable"] is True
    assert primary["order_used"] == primary["order_observed"]

    assert primary["extrapolated"] == pytest.approx(0.97130, abs=5e-6)
    assert primary["gci_fine_percent"] == pytest.approx(0.103083, abs=5e-6)
    assert primary["gci_coarse_percent"] == pytest.approx(0.356249, abs=5e-6)
    assert primary["asymptotic_ratio"] == pytest.approx(1.002019, abs=5e-6)

    # Fs = 1.25 is the page's factor and this script's default for three grids, and the
    # report says both, because a band whose factor is unstated is a band of unstated
    # size.
    assert analysis["safety_factor"] == 1.25
    assert analysis["safety_factor_is_default"] is True
    assert "1.25 (the default for this many grids)" in grid.report(analysis, [])

    # The quoted answer: 0.97050 +/- 0.103083% of itself.
    assert primary["band"] == pytest.approx(0.97050 * 0.103083 / 100.0, abs=5e-9)
    assert primary["band_source"] == "GCI"


def test_the_nasa_example_the_other_way_up(grid):
    """The same three values handed over coarsest-first. Which grid is fine is a fact
    about h and not about argument order, and a ladder typed in the order the meshes
    were built is the order a person will type."""
    analysis = grid.analyse(ladder(grid, (4.0, 0.96178), (2.0, 0.96854), (1.0, 0.97050)))
    assert analysis["primary"]["order_observed"] == pytest.approx(1.786170, abs=5e-7)
    assert analysis["primary"]["grids"][0] == "h=1"


# -- the arithmetic on data whose answer is known by construction --------------


def test_exact_second_order_data_recovers_p_equals_two(grid):
    """f(h) = 1 + h**2 sampled at h = 1, 2, 4. The observed order has to come out 2 and
    the extrapolation has to come out 1, exactly, because that is what the data is."""
    analysis = grid.analyse(ladder(grid, (1.0, 1.5), (2.0, 3.0), (4.0, 9.0)))
    primary = analysis["primary"]
    assert primary["order_observed"] == pytest.approx(2.0, abs=1e-9)
    assert primary["order_usable"] is True
    assert primary["extrapolated"] == pytest.approx(1.0, abs=1e-12)
    assert primary["convergence"] == "monotone"


def test_a_non_integer_refinement_ratio(grid):
    """The ratios a mesher actually delivers.

    `p = ln(eps32/eps21) / ln(r)` has one r in it and there are two here: 1.7 and
    1.8235. Exact second-order data at h = 1, 1.7, 3.1, so the answer is still 2, and
    the textbook single-ratio formula gets 2.39 -- a 20% error in the exponent that
    the extrapolation and the whole band hang off.
    """
    hs = (1.0, 1.7, 3.1)
    values = tuple(1.0 + 0.5 * h**2 for h in hs)
    analysis = grid.analyse(ladder(grid, *zip(hs, values)))
    primary = analysis["primary"]

    assert primary["ratio_21"] == pytest.approx(1.7)
    assert primary["ratio_32"] == pytest.approx(3.1 / 1.7)
    assert primary["order_observed"] == pytest.approx(2.0, abs=1e-8)
    assert primary["extrapolated"] == pytest.approx(1.0, abs=1e-10)

    # What the single-ratio formula would have said, for the size of the error it makes.
    naive = math.log(abs((values[2] - values[1]) / (values[1] - values[0]))) / math.log(1.7)
    assert naive == pytest.approx(2.3906, abs=0.0001)


def test_cell_counts_and_spacings_are_the_same_ladder(grid):
    """h is a proportionality, so a ladder of cell counts and the same ladder of
    spacings have to give the same p, the same extrapolation and the same GCI. If they
    did not, the published example could not be checked against spacings."""
    counts = (640000, 160000, 40000)
    hs = [grid.representative_spacing(n, 2) for n in counts]
    values = (0.97050, 0.96854, 0.96178)
    by_count = grid.analyse(ladder(grid, *zip(hs, values)))["primary"]
    by_spacing = grid.analyse(ladder(grid, *zip((1.0, 2.0, 4.0), values)))["primary"]
    assert by_count["order_observed"] == pytest.approx(by_spacing["order_observed"])
    assert by_count["extrapolated"] == pytest.approx(by_spacing["extrapolated"])
    assert by_count["gci_fine_percent"] == pytest.approx(by_spacing["gci_fine_percent"])


# -- each way a naive implementation misleads ----------------------------------


def test_oscillatory_data_is_refused_rather_than_extrapolated(grid):
    """The middle value not between the other two. Richardson's derivation presumes a
    single error term approached from one side; this sequence has already contradicted
    that, so the extrapolation would be arithmetic on a false premise -- and it would
    still print a number. It is reported as oscillatory, with the interval the ladder
    does support, and with no GCI percentage."""
    analysis = grid.analyse(ladder(grid, (1.0, 1.0), (2.0, 1.2), (4.0, 1.1)))
    primary = analysis["primary"]

    assert primary["convergence"] == "oscillatory"
    assert primary["extrapolated"] is None
    assert primary["gci_fine_percent"] is None
    assert primary["gci_coarse_percent"] is None
    assert primary["order_observed"] is None
    assert primary["order_usable"] is False
    assert primary["value_range"] == [1.0, 1.2]
    assert "oscillatory" in grid.report(analysis, []).lower()
    assert "no value with an uncertainty" in grid.report(analysis, [])


def test_an_absurd_observed_order_is_reported_and_flagged(grid):
    """f(h) = 1 + h**8 gives p = 8 against a formal order of 2. The number is real and
    is the most diagnostic thing the triplet has to say, so it is printed; it is not a
    usable order of convergence, so it is not used, and the GCI falls back to the
    formal order with the clamp named.

    Changed by issue #142, and this is the one existing assertion that moved. It used to
    assert that this ladder still produced a GCI percentage. It no longer does, because
    extrapolating 2.0, 257.0, 65537.0 on the clamped p = 2 gives -83, a limit on the far
    side of zero from all three values -- the same self-disqualifying shape as the
    `-0.80441086` drag coefficient #142 reported, which is now withheld wherever it
    appears. What the test was for is unchanged and still checked here: the absurd p is
    printed, is said to be unusable, and is not used. The second ladder below keeps the
    original claim that a clamped order still yields a band labelled an estimate, on a
    triplet whose extrapolation stays on the ladder's own side of zero.
    """
    analysis = grid.analyse(ladder(grid, (1.0, 2.0), (2.0, 257.0), (4.0, 65537.0)))
    primary = analysis["primary"]

    assert primary["order_observed"] == pytest.approx(8.0, abs=1e-8)
    assert primary["order_usable"] is False
    assert primary["order_used"] == 2.0
    assert "clamped to the formal order" in primary["order_source"]
    assert "not a usable order" in primary["note"]
    assert "8.000000" in grid.report(analysis, [])

    # The extrapolation on the clamped order lands at -83, outside the range of every
    # input, so the band goes with it.
    assert primary["gci_fine_percent"] is None
    assert primary["band"] is None
    assert any("differs from every value" in reason
               for reason in primary["disqualified"])

    # Same p = 8, same clamp, an extrapolation that does not change sign: the band is
    # still produced and still labelled an estimate.
    scaled = grid.analyse(ladder(grid, (1.0, 1.001), (2.0, 1.256), (4.0, 66.537)))
    assert scaled["primary"]["order_observed"] == pytest.approx(8.0, abs=1e-4)
    assert scaled["primary"]["order_used"] == 2.0
    assert scaled["primary"]["gci_fine_percent"] is not None
    assert "the band is an estimate" in grid.report(scaled, [])


def test_a_negative_observed_order_is_reported_and_flagged(grid):
    """Differences that shrink as the mesh coarsens: the sequence is moving away from
    its limit under refinement. p comes out negative and the solve has to look at
    negative p to find it -- reporting "no order found" here would hide the diagnosis.
    """
    analysis = grid.analyse(ladder(grid, (1.0, 1.0), (2.0, 1.5), (4.0, 1.6)))
    primary = analysis["primary"]

    assert primary["convergence"] == "monotone"
    assert primary["order_observed"] == pytest.approx(math.log(0.2) / math.log(2.0), abs=1e-8)
    assert primary["order_observed"] < 0.0
    assert primary["order_usable"] is False
    assert primary["order_used"] == 2.0
    assert "diverging as the mesh refines" in primary["note"]


def test_identical_values_do_not_divide_by_zero(grid):
    """Two grids giving the same number is mesh independence reached, which is the good
    outcome, and it makes `eps21` zero. Nothing is divided by it and no order is
    claimed off a difference that is not there."""
    analysis = grid.analyse(ladder(grid, (1.0, 0.412), (2.0, 0.412), (4.0, 0.430)))
    primary = analysis["primary"]

    assert primary["convergence"] == "no change"
    assert primary["order_observed"] is None
    assert primary["order_usable"] is False
    assert primary["extrapolated"] is None
    assert "same value" in primary["note"]
    assert "mesh independence" in primary["note"]


def test_values_equal_to_rounding_do_not_divide_by_near_zero(grid):
    """Not only exactly equal: a difference at the level of floating-point noise is not
    a difference, and an order of convergence computed from noise is noise raised to a
    logarithm."""
    analysis = grid.analyse(ladder(grid, (1.0, 1.0), (2.0, 1.0 + 1e-15), (4.0, 1.4)))
    primary = analysis["primary"]
    assert primary["convergence"] == "no change"
    assert primary["order_observed"] is None
    assert str(grid.EQUAL_RELATIVE) in primary["note"] or "1e-12" in primary["note"]


def test_a_near_zero_fine_value_gives_an_absolute_band_not_a_relative_one(grid):
    """The GCI is a relative error. A fine value of zero -- a lift coefficient at zero
    incidence, a net flux that ought to cancel -- leaves it nothing to be relative to.
    `reattach.py` returns None for its error bar rather than a generous default in the
    same situation; here the absolute difference stands in, labelled as the weaker
    thing it is."""
    analysis = grid.analyse(ladder(grid, (1.0, 0.0), (2.0, 0.01), (4.0, 0.04)))
    primary = analysis["primary"]
    assert primary["gci_fine_percent"] is None
    assert primary["band"] == pytest.approx(0.01)
    assert "nothing to be relative to" in primary["band_source"]


# -- two grids -----------------------------------------------------------------


def test_two_grids_assume_an_order_and_take_the_larger_safety_factor(grid):
    """Two grids observe no order, so the formal order is assumed and the band has to
    cover the assumption too: Fs = 3.0, not 1.25. Both the factor and the fact that it
    was assumed are in the output, since a two-grid band quoted as though it were a
    three-grid one is the main way this arithmetic gets oversold."""
    analysis = grid.analyse(ladder(grid, (1.0, 0.97050), (2.0, 0.96854)))
    primary = analysis["primary"]

    assert analysis["safety_factor"] == 3.0
    assert analysis["safety_factor_is_default"] is True
    assert primary["mode"] == "two-grid"
    assert primary["order_observed"] is None
    assert primary["order_used"] == 2.0
    assert primary["order_source"].startswith("assumed")

    # Fs * |eps/f1| / (r**p - 1) * 100 with p = 2, r = 2: the denominator is 3.
    expected = 3.0 * abs(-0.00196 / 0.97050) / 3.0 * 100.0
    assert primary["gci_fine_percent"] == pytest.approx(expected, rel=1e-9)
    assert primary["extrapolated"] == pytest.approx(0.97050 + 0.00196 / 3.0, abs=5e-9)

    text = grid.report(analysis, [])
    assert "3 (the default for this many grids)" in text
    assert "two-grid estimate" in text


def test_an_explicit_safety_factor_is_not_reported_as_the_default(grid):
    given = grid.analyse(ladder(grid, (1.0, 1.5), (2.0, 3.0), (4.0, 9.0)), safety=1.5)
    assert given["safety_factor"] == 1.5
    assert given["safety_factor_is_default"] is False
    assert "1.5 (the default" not in grid.report(given, [])


def test_one_grid_is_not_a_convergence_study(grid):
    with pytest.raises(ValueError):
        grid.analyse(ladder(grid, (1.0, 0.412)))


# -- cell counts to h ----------------------------------------------------------


def test_cell_counts_become_h_in_two_and_three_dimensions(grid):
    """h ~ N**(-1/3) in 3D and N**(-1/2) in 2D. A plane case in OpenFOAM is a
    one-cell-thick 3D mesh, so the cell count alone cannot say which, and reading it as
    3D when it is 2D moves the exponent by a factor of 1.5."""
    assert grid.representative_spacing(10_000, 2) == pytest.approx(0.01)
    assert grid.representative_spacing(8_000, 3) == pytest.approx(0.05)

    # Four times the cells in 2D halves h; eight times the cells in 3D halves h.
    assert grid.representative_spacing(40_000, 2) == pytest.approx(0.005)
    assert grid.representative_spacing(64_000, 3) == pytest.approx(0.025)

    # The same count read the two ways is a different ladder, which is why --dim is a
    # flag and not a guess.
    assert grid.representative_spacing(10_000, 2) != pytest.approx(
        grid.representative_spacing(10_000, 3))

    with pytest.raises(ValueError):
        grid.representative_spacing(0, 3)


# -- reading it off a case -----------------------------------------------------

CHECK_MESH = """\
Create time

Create polyMesh for time = constant

Time = constant

Mesh stats
    points:           79173
    internal faces:   151504
    faces:            159184
    cells:            76800
    faces per cell:   6
    boundary patches: 5
    point zones:      0
    face zones:       0
    cell zones:       0

Overall number of cells of each type:
    hexahedra:     76800
    prisms:        0
    wedges:        0
    pyramids:      0
    tet wedges:    0
    tetrahedra:    0
    polyhedra:     0

Checking topology...
    Boundary definition OK.
    Cell to face addressing OK.
    Point usage OK.
    Upper triangular ordering OK.
    Face vertices OK.
    Number of regions: 1 (OK).

Checking geometry...
    Overall domain bounding box (-0.0206 -0.0254 -0.0005) (0.29 0.0254 0.0005)
    Mesh has 2 geometric (non-empty/wedge) directions (1 1 0)
    Mesh has 2 solution (non-empty) directions (1 1 0)
    Boundary openness (1.2e-17 3.1e-18 -2.4e-15) OK.
    Max cell openness = 2.1e-16 OK.
    Max aspect ratio = 12.4 OK.
    Minimum face area = 1.1e-08. Maximum face area = 7.9e-06.  Face area magnitudes OK.
    Min volume = 5.5e-12. Max volume = 7.9e-09.  Total volume = 0.000376.  Cell volumes OK.
    Mesh non-orthogonality Max: 42.1 average: 6.2
    Non-orthogonality check OK.
    Face pyramids OK.
    Max skewness = 1.9 OK.
    Coupled point location match (average 0) OK.

Mesh OK.

End
"""

BOUNDARY_2D = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}

5
(
    inlet
    {
        type            patch;
        nFaces          80;
        startFace       151504;
    }
    outlet
    {
        type            patch;
        nFaces          80;
        startFace       151584;
    }
    walls
    {
        type            wall;
        nFaces          1920;
        startFace       151664;
    }
    frontAndBack
    {
        type            empty;
        nFaces          5600;
        startFace       153584;
    }
)
"""


def test_a_real_checkmesh_log_is_read_for_its_cell_count(grid, tmp_path):
    """The cell count is the x axis of the whole study and it comes from `checkMesh`
    output through `mesh_digest.parse`, which is the sibling that already knows how to
    read it. A log already in the case is preferred over running anything, because the
    mesh a ladder is about may be on a machine with no OpenFOAM on it and the log
    written when the mesh was built is the same number."""
    case = tmp_path / "medium"
    case.mkdir()
    (case / "log.checkMesh").write_text(CHECK_MESH, encoding="utf-8")

    cells, where = grid.cells_from_case(case)
    assert cells == 76800
    assert where.endswith("log.checkMesh")

    # And the h it becomes, taking the plane case the log describes as 2D.
    assert grid.representative_spacing(cells, 2) == pytest.approx(76800 ** -0.5)


def test_an_empty_patch_is_read_as_two_dimensional(grid, tmp_path):
    """`--dim auto`'s convenience route. An `empty` patch is OpenFOAM's own statement
    that a direction is not solved in, which is what `case_gen.py` reads it as too. The
    route is returned alongside the answer, because a detected 2D and a declared 2D are
    not equally strong claims and the report says which it had."""
    case = tmp_path / "plane"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "constant" / "polyMesh" / "boundary").write_text(BOUNDARY_2D, encoding="utf-8")

    dimension, route = grid.dimension_from_case(case)
    assert dimension == 2
    assert "frontAndBack" in route and "empty" in route

    solid = tmp_path / "solid"
    (solid / "constant" / "polyMesh").mkdir(parents=True)
    (solid / "constant" / "polyMesh" / "boundary").write_text(
        BOUNDARY_2D.replace("empty", "wall"), encoding="utf-8")
    dimension, route = grid.dimension_from_case(solid)
    assert dimension == 3
    assert "no patch" in route

    missing, route = grid.dimension_from_case(tmp_path / "nothing")
    assert missing is None
    assert "no " in route


# -- the report ----------------------------------------------------------------


def test_the_report_names_every_number_and_its_provenance(grid):
    """A grid ladder is the one place where reading the wrong column on one of three
    cases produces a perfectly plausible answer, so the report says which case, which
    file and which column each number came from rather than only what it was."""
    analysis = grid.analyse(ladder(grid, (1.0, 0.97050), (2.0, 0.96854), (4.0, 0.96178)))
    analysis["quantity"] = "Cd"
    text = grid.report(analysis, [
        "fine: 640,000 cells from /work/fine/log.checkMesh; Cd = 0.9705 from column "
        "'Cd' (index 1) of /work/fine/postProcessing/forces/0/coefficient.dat",
    ])
    assert "where the numbers came from" in text
    assert "coefficient.dat" in text
    assert "index 1" in text
    assert "Cd = 0.9705 +/-" in text
    assert "+/-" in text
    assert "1.786170" in text
    assert "Richardson" in text


def test_the_coarser_triplets_are_reported_beside_the_fine_one(grid):
    """Four grids give two triplets. A p that collapses on the coarse triplet says the
    coarse end of the ladder is not in the asymptotic range, which is worth seeing
    rather than inferring from a single number."""
    analysis = grid.analyse(ladder(
        grid, (1.0, 1.5), (2.0, 3.0), (4.0, 9.0), (8.0, 33.0)))
    assert len(analysis["triplets"]) == 2
    assert analysis["primary"]["grids"] == ["h=1", "h=2", "h=4"]
    assert len(analysis["pairs"]) == 3
    assert [round(p["ratio"], 6) for p in analysis["pairs"]] == [2.0, 2.0, 2.0]
    assert "the coarser triplets" in grid.report(analysis, [])


# -- the guards on the inputs, from issue #142 ---------------------------------
#
# Every test above hands `analyse` numbers. These hand it the numbers a real study
# handed it, which reported `Cd = 0.27556258 +/- 3.240` on `case2` with an extrapolated
# value of `-0.80441086` and a GCI of 1175.75% -- in the report's "the answer" block,
# which is the part that gets quoted -- off a refinement ratio of r = 1.1302 and a
# quantity that was the mean of the last 1 of 4 rows. The arithmetic was right. Nothing
# was checking that the inputs were in the domain the arithmetic is defined on.

CASE_CELLS = 262_099
CASE2_CELLS = 378_396
"""The two meshes of the #142 ladder, whose 3D cell counts give r = 1.1302."""


def real_142_ladder(grid, **overrides):
    """The #142 pair as `analyse` takes it: h off the real cell counts, the real values,
    and the row counts of the real histories -- 4 rows on `case2`, and 3,000 on `case`,
    which is what `results.read_history` gives there now that #146 is fixed too (the
    report in the issue printed 3,006, six of them superseded by a restart)."""
    fine = {"h": grid.representative_spacing(CASE2_CELLS, 3), "value": 0.27556258,
            "label": "case2", "rows": 4, "rows_averaged": 1}
    coarse = {"h": grid.representative_spacing(CASE_CELLS, 3), "value": 0.57512774,
              "label": "case", "rows": 3000, "rows_averaged": 1}
    fine.update(overrides)
    return [fine, coarse]


def test_a_refinement_ratio_below_one_point_three_is_reported_with_no_gci(grid):
    """Issue #142, the first of its three disqualifying facts.

    262,099 and 378,396 cells in 3D are r = 1.1302, below the 1.3 Celik's order relation
    and Roache's factors are calibrated on. At that r with p = 2 the denominator
    `r**p - 1` is 0.277, which is what turned a difference of 0.2996 in Cd into a GCI of
    1175.75% and a band of `+/- 3.240` on a value of 0.2756. A band 1,176% of its own
    value is not an uncertainty, it is a vanishing denominator being printed.

    What this protects: the ratio is still reported, with its number, and no GCI, no
    extrapolation and no band are. The refusal has to be checkable, so the values and
    the ratio stay in the report; the thing that looked like an answer goes.
    """
    # Only this guard: enough rows, and an extrapolation that keeps its sign.
    analysis = grid.analyse(real_142_ladder(grid, rows=4000, value=0.56))
    analysis["quantity"] = "Cd"
    primary = analysis["primary"]

    assert primary["ratio_21"] == pytest.approx(1.1302, abs=5e-5)
    assert len(primary["disqualified"]) == 1
    assert "1.1302" in primary["disqualified"][0]
    assert "1.3" in primary["disqualified"][0]
    assert primary["gci_fine_percent"] is None
    assert primary["extrapolated"] is None
    assert primary["band"] is None

    text = grid.report(analysis, [])
    assert "r = 1.1302" in text          # the input is still there to be checked
    assert "no band on Cd" in text
    assert "GCI on the fine grid" not in text
    assert "1175" not in text
    assert "the values themselves stand" in text


def test_a_ratio_at_the_threshold_is_still_a_ladder(grid):
    """The guard is a threshold and a threshold has to not fire on the thing it was
    calibrated to admit. r = 1.3 is Celik's own lower bound, so it passes and gets the
    full band; #142's 1.1302 does not. Without this the fix could be a refusal of
    everything, which is also not an answer."""
    hs = (1.0, 1.3, 1.69)
    values = tuple(1.0 + 0.5 * h**2 for h in hs)
    primary = grid.analyse(ladder(grid, *zip(hs, values)))["primary"]
    assert primary["ratio_21"] == pytest.approx(1.3)
    assert "disqualified" not in primary
    assert primary["order_observed"] == pytest.approx(2.0, abs=1e-7)
    assert primary["gci_fine_percent"] is not None


def test_a_value_from_a_four_row_history_is_refused_by_name(grid):
    """Issue #142, the second fact. `case2`'s Cd was reported as the `mean of the last 1
    of 4 rows` -- iteration 4 of a run that aborted in its startup transient, with one
    time directory and no written result. A single row of a four-row history is not a
    converged value and cannot carry a grid-convergence claim, and the script had the
    row count in its own provenance line while quoting the number anyway.

    The refusal names the case, because on a three-rung ladder the reader needs to know
    which rung to re-run, and states both numbers -- 1 of 4 -- because `mean of the last
    1 of 3006 rows` is the documented default on a converged steady case and is fine.
    """
    primary = grid.analyse(real_142_ladder(grid))["primary"]

    rows_reason = [r for r in primary["disqualified"] if "rows of its history" in r]
    assert len(rows_reason) == 1
    assert rows_reason[0].startswith("case2's value is the mean of the last 1 of 4 rows")
    assert primary["band"] is None

    # The 3,000-row rung is not what is being complained about.
    assert "case's value" not in " ".join(primary["disqualified"])

    # And a ladder of two settled histories at the same values is not refused for rows.
    settled = grid.analyse(real_142_ladder(grid, rows=3000, value=0.56))
    assert not [r for r in settled["primary"]["disqualified"]
                if "rows of its history" in r]


def test_an_extrapolation_whose_sign_no_input_shares_disqualifies_itself(grid):
    """Issue #142, the third fact: the Richardson limit it printed was `-0.80441086`
    from two positive drag coefficients, 0.57512774 and 0.27556258. A negative drag
    coefficient is not a limit. More generally a limit on the far side of zero from
    every value it was formed from is a result that disqualifies itself -- the sequence
    cannot be approaching something outside its own range from one side, which is the
    premise the extrapolation rests on.

    Reported with the number and the inputs rather than as a bare refusal, because
    `-0.80441086 against 0.2756 and 0.5751` is the evidence and "the extrapolation
    failed" is not.
    """
    primary = grid.analyse(real_142_ladder(grid, rows=4000))["primary"]
    sign = [r for r in primary["disqualified"] if "far side of zero" in r]
    assert len(sign) == 1
    assert "-0.80441086" in sign[0]
    assert "0.275563" in sign[0] and "0.575128" in sign[0]
    assert primary["extrapolated"] is None
    assert primary["band"] is None

    # The mirror image: an all-negative ladder whose limit comes out positive.
    flipped = grid.analyse(ladder(grid, (1.0, -0.27556258), (1.1302, -0.57512774)))
    assert any("far side of zero" in r for r in flipped["primary"]["disqualified"])

    # A limit outside the ladder's range but on its own side of zero is ordinary
    # Richardson extrapolation, which is most of what this script does.
    fine = grid.analyse(ladder(grid, (1.0, 1.5), (2.0, 3.0), (4.0, 9.0)))["primary"]
    assert fine["extrapolated"] == pytest.approx(1.0)
    assert "disqualified" not in fine


def test_the_whole_142_ladder_reports_all_three_facts_and_no_answer(grid):
    """The reproduction from the issue, end to end. All three guards fire on it at once,
    and the output has to contain the three measurements and none of `+/- 3.240`,
    `-0.80441086` or `1175.75%`."""
    analysis = grid.analyse(real_142_ladder(grid))
    analysis["quantity"] = "Cd"
    text = grid.report(analysis, [])

    assert len(analysis["primary"]["disqualified"]) == 3
    assert "1.1302" in text and "1 of 4 rows" in text and "far side of zero" in text
    assert "Cd = 0.27556258 +/-" not in text
    assert "3.240" not in text
    assert "1175" not in text
    # -0.80441086 stays in the report, as the evidence inside the reason it is being
    # refused. What is gone is the line that offered it as the value.
    assert "extrapolated value" not in text
    assert "-0.80441086" in " ".join(analysis["primary"]["disqualified"])
    # The values are not in doubt and are still printed, which is what makes the
    # refusal reviewable rather than only a refusal.
    assert "0.27556258" in text and "0.57512774" in text


# -- reading a quantity off a real case, from issue #146 -----------------------

FORCE_DAT = """\
# Force
# CofR         : (0.0000000e+00 0.0000000e+00 0.0000000e+00)
#
# Time         \ttotal_x total_y total_z\tpressure_x pressure_y pressure_z\tviscous_x viscous_y viscous_z
{rows}
"""

COEFF_DAT = """\
# Force and moment coefficients
# magUInf       : 1.00000000e+00
#
# Time         \tCd\tCl
{rows}
"""


def write_dat(path: Path, template: str, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.format(rows=rows), encoding="utf-8")


def forces_case(root: Path) -> Path:
    """A case shaped like the `ami_L1` of issue #146: a `forces` function object, which
    writes `force.dat` and `moment.dat` with the same column names, and a second
    function object on another patch, which writes two more. The nine columns of a
    forces object appear in all four files, so a bare `--quantity viscous_z` matches
    four series on a case that is only logging forces the ordinary way."""
    case = root / "ami_L1"
    for function, scale in (("forces", 1.0), ("torqueOuter", 10.0)):
        for name in ("force.dat", "moment.dat"):
            factor = scale * (2.0 if name == "moment.dat" else 1.0)
            rows = "\n".join(
                f"{t:<16g}" + " ".join(f"{factor * (t + part):.7e}" for part in range(1, 10))
                for t in range(1, 21))
            write_dat(case / "postProcessing" / function / "0" / name, FORCE_DAT, rows)
    return case


def test_a_series_qualified_quantity_resolves_a_repeated_column(grid, tmp_path):
    """Issue #146, part one: `--quantity` could only name a column, and every column of
    a `forces` function object appears in both its `force.dat` and its `moment.dat`. So
    on the real `ami_L1/L2/L3` ladder -- and on every case that logs forces at all --
    `--quantity viscous_z` answered "'viscous_z' is a column in more than one series ...
    name the one you mean with a more specific --quantity", and no such syntax existed.
    The case-reading path of the script was unreachable on that shape of study.

    This is the syntax it was asking for. `series:column` is matched against the series
    keys the function already builds, so `torqueOuter/moment.dat:viscous_z` -- the
    inner-wall torque that ladder is about -- is sayable.
    """
    case = forces_case(tmp_path)

    # The series name narrows it to one file, and the value is that file's.
    picked = grid.quantity_from_case(case, "torqueOuter/moment.dat:viscous_z")
    assert picked["value"] == pytest.approx(20.0 * (20 + 9))
    assert picked["series"] == "torqueOuter/moment.dat"
    assert picked["column"] == "viscous_z"

    # A different series, the same column name, a different number: which is the whole
    # reason naming the series cannot be optional.
    other = grid.quantity_from_case(case, "forces/force.dat:viscous_z")
    assert other["value"] == pytest.approx(1.0 * (20 + 9))
    assert other["value"] != picked["value"]

    # A bare column name still works where it is unambiguous, which is every existing
    # call: nothing has to be requalified.
    write_dat(tmp_path / "plain" / "postProcessing" / "forceCoeffs" / "0" /
              "coefficient.dat", COEFF_DAT,
              "\n".join(f"{t:<16g}{0.4 + t / 1000:.7e} {0.1:.7e}" for t in range(1, 21)))
    assert grid.quantity_from_case(tmp_path / "plain", "Cd")["value"] == pytest.approx(0.42)


def test_an_ambiguous_quantity_names_the_syntax_and_lists_the_series(grid, tmp_path):
    """Also #146 part one, and the half that made it a dead end rather than a refusal:
    the message asked for "a more specific --quantity" without saying what one looks
    like. It now names the form, shows a complete flag the reader can paste, and lists
    every candidate, since the choice between `forces/moment.dat:viscous_z` and
    `torqueOuter/moment.dat:viscous_z` is a choice about which number the study is about
    and is not the script's to make."""
    case = forces_case(tmp_path)
    note = grid.quantity_from_case(case, "viscous_z")["note"]

    assert note.startswith("'viscous_z' is a column in 4 series here")
    assert "SERIES:COLUMN" in note
    assert "--quantity forces/force.dat:viscous_z" in note
    for series in ("forces/force.dat", "forces/moment.dat",
                   "torqueOuter/force.dat", "torqueOuter/moment.dat"):
        assert f"{series}:viscous_z" in note

    # A partial series name that still fits two series refuses the same way, with the
    # two it fits, rather than picking one.
    narrowed = grid.quantity_from_case(case, "forces/:viscous_z")["note"]
    assert "is a column in 2 series here" in narrowed
    assert "forces/force.dat:viscous_z" in narrowed
    assert "torqueOuter" not in narrowed

    # A series name that fits nothing says so, and lists what is there.
    missing = grid.quantity_from_case(case, "nosuch/moment.dat:viscous_z")["note"]
    assert "no series" in missing and "nosuch/moment.dat" in missing
    assert "torqueOuter/moment.dat" in missing


def test_a_restart_superseded_history_is_stitched_the_way_read_history_does_it(
        grid, tmp_path):
    """Issue #146, part two, and the one that produced wrong numbers rather than a dead
    end. `quantity_from_case`'s docstring claimed it stitched a restarted series "the
    way `results.py` does it"; what it did was concatenate every file of the series and
    take the tail. `results.read_history` additionally drops the rows of a file that lie
    past the point where a later run restarted -- the rows of the run that was
    abandoned, which is the whole reason that function exists.

    Measured on `20260829-051829-530e/cyl`: 9369 rows against `read_history`'s 8586,
    783 rows of abandoned runs merged in, giving 1.21093 at `--tail 1000` where the
    right answer is 1.20803, 0.24% out; and 1.5% out at `--tail 4000` on
    `20260830-164816-eee0/case`. This script exists to produce a band of 0.003% to
    0.36% on that corpus, so a 1.5% error in the value is an order larger than the
    uncertainty it reports. The row count it printed as provenance -- the line a reader
    checks -- was wrong by the same 783.

    The case here is that shape in miniature: a run to t = 10 restarted from the write
    at t = 5, so the first file's rows past 5 are a run that was thrown away.
    """
    results = load("results")
    case = tmp_path / "cyl"
    first = case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    second = case / "postProcessing" / "forceCoeffs" / "5" / "coefficient.dat"
    write_dat(first, COEFF_DAT,
              "\n".join(f"{t:<16g}{1.0 + t / 100:.7e} {0.1:.7e}" for t in range(1, 11)))
    write_dat(second, COEFF_DAT,
              "\n".join(f"{t:<16g}{2.0 + t / 100:.7e} {0.1:.7e}" for t in range(6, 9)))
    # `written_order` reads mtimes, and the restart is the file written last.
    os.utime(first, (1_600_000_000, 1_600_000_000))
    os.utime(second, (1_600_000_100, 1_600_000_100))

    history = results.read_history([first, second])
    assert len(history["rows"]) == 8     # 1..5 from the first run, 6..8 from the restart
    assert history["superseded"]         # and it names what it dropped

    measured = grid.quantity_from_case(case, "Cd", tail=20)
    assert measured["rows"] == len(history["rows"])
    assert measured["rows"] == 8         # not the 13 that concatenating the files gives
    assert measured["rows_averaged"] == 8

    expected = sum(results.column(history, "Cd")) / len(history["rows"])
    assert measured["value"] == pytest.approx(expected)

    # The superseded blocks are carried out so the provenance line can say rows were
    # dropped: a reader who sees 8 rows where the files hold 13 needs to be told why.
    assert measured["superseded"] == history["superseded"]

    # The abandoned rows are the ones with the wrong number in them -- the discarded run
    # wrote Cd near 1.1 at t = 6..10 where the restart wrote near 2.06 -- so the two
    # row counts are two different answers and not only two different counts.
    assert measured["value"] > 1.3


# -- the rung that was dropped (#171) ------------------------------------------


OWNER_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       labelList;
    note        "nPoints:{points} nCells:{cells} nFaces:{faces} nInternalFaces:{internal}";
    location    "constant/polyMesh";
    object      owner;
}}

{faces}
(
)
"""


def rung(root: Path, name: str, cells: int, value: float) -> Path:
    """A ladder rung with no checkMesh log: a mesh whose own header carries `nCells`,
    an `empty` patch for `--dim auto`, and a one-column force history."""
    case = root / name
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "owner").write_text(
        OWNER_HEADER.format(points=2 * cells, cells=cells, faces=3 * cells,
                            internal=2 * cells), encoding="utf-8")
    (poly / "boundary").write_text(BOUNDARY_2D, encoding="utf-8")
    write_dat(case / "postProcessing" / "forces" / "0" / "moment.dat", FORCE_DAT,
              " ".join([f"{1.0:<16g}"] + [f"{value:.8e}"] * 9))
    return case


def test_a_rung_with_no_checkmesh_log_is_read_from_the_mesh_header(grid, tmp_path):
    """#171. Only checkMesh logs were looked at, so the finest rung of a real
    three-grid ladder was dropped for want of a cell count that was on disk one file
    away -- the `nCells` note OpenFOAM writes into `constant/polyMesh/owner`, in the
    same directory as the `boundary` that `--dim auto` read successfully in that very
    case. The ladder became a two-grid estimate on the wrong grid with a band 2.1x too
    wide, and a second real ladder, with no log on any of its three rungs, was refused
    outright: `0 was readable`.

    `study_state.owner_ncells` is the read, which already existed for `digest`. A
    checkMesh log still wins where there is one -- it is the fuller reading -- and
    running `checkMesh` is still the last resort.
    """
    case = rung(tmp_path, "L2", 11520, 1.5)
    cells, where = grid.cells_from_case(case)
    assert cells == 11520
    assert where.endswith("owner"), where

    # A log, where there is one, is still preferred over the header.
    (case / "log.checkMesh").write_text(CHECK_MESH, encoding="utf-8")
    cells, where = grid.cells_from_case(case)
    assert (cells, where.endswith("log.checkMesh")) == (76800, True)


def test_a_dropped_rung_is_named_in_the_report_and_the_estimate_called_degraded(
        grid, tmp_path, capsys):
    """The other half of #171: the only notice that a rung had been dropped was a line
    on stderr, and the report is the artifact that gets saved. A three-grid ladder
    silently became a two-grid estimate that named the wrong grid, gave the wrong value
    and widened its band, with nothing in the document saying so.

    The first call is the ladder as given, all three rungs read from their mesh
    headers. The second is the same ladder with the finest rung's mesh removed, which
    is the shape the issue reported.
    """
    for name, cells, value in (("L1", 2880, 1.0), ("L2", 11520, 1.5), ("L3", 46080, 1.75)):
        rung(tmp_path, name, cells, value)
    cases = [str(tmp_path / name) for name in ("L1", "L2", "L3")]
    argv = cases + ["--quantity", "forces/moment.dat:total_z", "--dim", "auto"]

    assert grid.main(argv) == 0
    whole = capsys.readouterr().out
    assert "three-grid estimate on L3, L2, L1" in whole
    assert "DEGRADED" not in whole
    assert "are not in this estimate" not in whole

    (tmp_path / "L3" / "constant" / "polyMesh" / "owner").unlink()
    assert grid.main(argv) == 0
    degraded = capsys.readouterr().out
    assert "two-grid estimate on L2, L1" in degraded
    assert "1 grid(s) given on the command line are not in this estimate" in degraded
    assert "L3: no cell count" in degraded
    assert "DEGRADED" in degraded
