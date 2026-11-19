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
    formal order with the clamp named."""
    analysis = grid.analyse(ladder(grid, (1.0, 2.0), (2.0, 257.0), (4.0, 65537.0)))
    primary = analysis["primary"]

    assert primary["order_observed"] == pytest.approx(8.0, abs=1e-8)
    assert primary["order_usable"] is False
    assert primary["order_used"] == 2.0
    assert "clamped to the formal order" in primary["order_source"]
    assert "not a usable order" in primary["note"]
    assert primary["gci_fine_percent"] is not None

    text = grid.report(analysis, [])
    assert "8.000000" in text
    assert "the band is an estimate" in text


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
