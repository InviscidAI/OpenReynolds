"""The tutorial index names every case the product can run, and quotes no bare number.

`tests/test_benchmarks_index.py` already holds `benchmarks/README.md` to its graders. This
file holds the tier structure beside it, which is a different claim: that a case somebody
can run is *findable*, and that every reference quoted at a reader carries the scale it is
normalised on.

Both halves are failures this repository has already paid for. The index ran three cases
while thirteen more were reachable only by finding the directory they live in, so the
toolbox's two numpy solvers -- the only cases that need no OpenFOAM and no account at all,
and the cheapest way to find out whether any of this works -- were invisible to anyone
reading the tutorials. And of the reference values in the index, re-checking them against
their sources changed two: Gartling's reattachment length, read as step heights when the
paper states channel heights, which is a factor of two and reached a customer-facing page,
and de Vahl Davis's cavity average quoted as his hot-wall value. Neither was arithmetic
and neither was a solver. Both were a number read without its scale, so the scale is what
is pinned here.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
TUTORIALS = BENCHMARKS / "tutorials"
INDEX = TUTORIALS / "README.md"
TEMPLATES_PAGE = TUTORIALS / "templates.md"
TEMPLATES = ROOT / "openreynolds" / "toolbox" / "templates"


def index() -> str:
    return INDEX.read_text(encoding="utf-8")


HEADER_CELLS = {"case", "prompt", "template"}


def case_rows() -> list[str]:
    """The body rows of the three tier tables: a leading pipe, at least five cells, not a
    separator, and not the header. Keyed on the header's first cell rather than on a word
    appearing anywhere in the row, because the rows are prose and any word in a header is
    also a word in some row."""
    rows = []
    for line in index().splitlines():
        if not line.startswith("|") or line.count("|") < 5:
            continue
        if re.match(r"^\|[\s|:-]+\|$", line):
            continue
        first = line.split("|")[1].strip().strip("*").lower()
        if first in HEADER_CELLS:
            continue
        rows.append(line)
    return rows


def flat(text: str) -> str:
    """Where a sentence happens to wrap is not the point."""
    return " ".join(text.split())


# -- every runnable case is findable ---------------------------------------------------


@pytest.mark.parametrize("family", ["lbm", "sph", "felix", "moving_mesh", "thermal_stress"])
def test_the_tutorial_index_names_every_benchmark_family(family):
    """A case family under `benchmarks/` that the tutorial index does not name is a case
    nobody reading the tutorials can find. `benchmarks/README.md` has its own version of
    this check; this one is about the index a reader is sent to first."""
    assert f"{family}/" in index(), (
        f"benchmarks/{family}/ holds runnable cases and the tutorial index does not name it")


@pytest.mark.parametrize("template", sorted(p.name for p in TEMPLATES.glob("*.py")))
def test_the_tier_two_page_names_every_template(template):
    """Six templates ship; the index named none of them until this page existed."""
    assert template in TEMPLATES_PAGE.read_text(encoding="utf-8"), (
        f"{template} is a case somebody can run and the tier 2 page does not name it")


def test_the_tier_two_page_names_nothing_that_is_not_a_template():
    """The other direction, which is the failure `test_toolbox.py` records: an index
    carried finished descriptions of two scripts that existed nowhere, in the present
    tense, and a session reading it was promised instruments it could not run."""
    named = set(re.findall(r"`([a-z0-9_]+\.py)`", TEMPLATES_PAGE.read_text(encoding="utf-8")))
    toolbox = {p.name for p in (TEMPLATES.parent).glob("*.py")}
    shipped = {p.name for p in TEMPLATES.glob("*.py")} | toolbox
    for name in sorted(named):
        assert name in shipped, f"the tier 2 page describes {name}, which is nowhere here"


def test_the_numpy_solvers_are_findable_without_an_openfoam():
    """The two cheapest cases in the repository and the only ones that need no OpenFOAM,
    no workspace, no model and no account. They are also the two whose graders *solve*
    rather than reading a finished run, so they are the only cases a reader can re-derive
    from nothing but this package."""
    text = flat(index())
    for command in ("python benchmarks/lbm/grade.py poiseuille",
                    "python benchmarks/sph/grade.py sloshing"):
        assert command in text, f"the index does not show how to run {command}"
    assert "numpy only" in text.lower(), (
        "the index does not say that a tier of these cases needs nothing installed")


def test_every_tier_says_what_running_it_costs():
    """The one thing a reader has to know before the physics: tier 3 bills a real
    account. An index that lists a hosted study beside a numpy script without saying so
    has invited somebody to spend money by accident."""
    text = flat(index()).lower()
    assert "nothing billed" in text or "nothing is billed" in text
    assert "cost a workspace and a model turn" in text or "cost money" in text
    assert "run by nothing automatically" in text or (
        "nothing runs tier 3 automatically" in text)


# -- no number without its scale -------------------------------------------------------


def test_the_index_states_a_normalisation_for_every_case_row():
    """Each tier table's reference column has to name the scale, not just the value.

    The check is deliberately crude -- it looks for a scale word in the column that
    quotes the reference -- because the precise phrasing is prose and the thing being
    prevented is a bare number. A row that says "6.1" and not what 6.1 is in is the
    exact shape of the error that cost this repository a factor of two.
    """
    scales = ("exact", "channel height", "step height", "diameter", "cavity side",
              "lattice units", "chord", "superficial", "depth", "tank length",
              "hydraulic diameter", "lid speed", "mixed mean", "pascals",
              "thickness", "closed form", "could not be read", "no reference")
    rows = case_rows()
    assert len(rows) >= 18, f"the index lists only {len(rows)} cases across its tiers"
    for row in rows:
        reference = row.split("|")[3].lower()
        assert any(scale in reference for scale in scales), (
            f"this row quotes a reference with no scale on it: {row.strip()[:160]}")


def test_every_case_row_states_an_agreement_or_says_it_has_none():
    """`benchmarks/README.md` is held to this and the tutorial index is the page a reader
    reaches first, so it is held to it too. A reference with no measured agreement beside
    it teaches a reader to trust an unvalidated number."""
    for row in case_rows():
        measured = row.split("|")[4].lower()
        stated = bool(re.search(r"\d", measured)) or any(
            phrase in measured for phrase in
            ("no verdict", "not graded", "none", "no agreement", "the mesh only",
             "not solved", "no reference", "ungraded"))
        assert stated, f"this row states no agreement and does not say it has none: {row.strip()[:160]}"


def test_an_unread_source_is_named_as_unread_wherever_it_is_quoted():
    """Two papers in this index could not be read: Gartling (1990) behind the Wiley
    paywall, and Martin & Moyce (1952), whose publisher returns 403. Both are quoted --
    there is no way to write these cases without naming them -- so both have to be flagged
    at every page that quotes them, or a reader takes the figure as established."""
    for page in (INDEX, TEMPLATES_PAGE):
        text = flat(page.read_text(encoding="utf-8")).lower()
        if "martin & moyce" in text:
            assert "could not be read" in text, f"{page.name} quotes Martin & Moyce as read"
            # and the reason it matters: two non-dimensionalisations differing by sqrt(2)
            assert "1.414" in text, (
                f"{page.name} does not say what the unverified normalisation is worth")
        if "gartling" in text:
            assert ("no primary access" in text or "was not read" in text
                    or "paywall" in text), f"{page.name} quotes Gartling as read"


def test_the_exact_cases_are_labelled_exact_rather_than_merely_close():
    """A tutorial that ends in an exact comparison teaches more and cannot rot, so the
    index says which ones those are rather than leaving a reader to work out whether 6.1
    is a measurement with scatter behind it or an identity."""
    text = flat(index())
    for exact in ("`u(y) = G/(2 mu) y (H - y)`", "`p = rho g (h - y)`",
                  "`omega^2 = g k tanh(k h)`", "`u_max = 1.5 u_mean`", "3.6568"):
        assert exact in text, f"the index does not carry the exact form {exact}"


def test_the_links_the_index_offers_exist():
    """Relative links are the only navigation the index has, and it now points at five
    sibling directories and a second page rather than at nothing."""
    for page in (INDEX, TEMPLATES_PAGE):
        links = re.findall(r"\]\(([^)]+)\)", page.read_text(encoding="utf-8"))
        relative = [a for a in links if not a.startswith(("http://", "https://", "#"))]
        assert len(relative) >= 5, f"{page.name} offers {len(relative)} links"
        for target in relative:
            assert (page.parent / target).exists(), (
                f"{page.name} links {target}, which is not there")


# -- the numbers on the tier 2 page are the ones its code computes ---------------------


def _template(name: str):
    spec = importlib.util.spec_from_file_location(f"tutorial_{name}", TEMPLATES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_porous_numbers_on_the_page_are_the_ones_the_template_prints():
    """Both of them, and in both units, because the factor between them is rho and that
    is the error the case exists to prevent: a 503 Pa measurement compared against a
    solved field reading 418 and called a 17% error.
    """
    porous = _template("porous")
    speed = porous.design_velocity_m_s()
    darcy, forchheimer, _ = porous.coefficients_from_curve()
    pascals = porous.predicted_drop_pa(speed, darcy, forchheimer)
    kinematic = pascals / porous.RHO_KG_M3

    page = TEMPLATES_PAGE.read_text(encoding="utf-8")
    for printed, computed in (("503.34", pascals), ("418.057", kinematic)):
        assert f"{computed:.6g}".startswith(printed.rstrip("0").rstrip(".")[:6]) or (
            printed in f"{computed:.6g}"), (
            f"the page quotes {printed} and the template computes {computed:.6g}")
        assert printed in page, f"{printed} is not on the tier 2 page"


def test_the_thermal_stress_peaks_on_the_page_are_the_ones_the_template_computes():
    """-220.0000 MPa and -314.2857 MPa appear in three documents and are computed in one
    place. `tests/test_benchmarks_index.py` pins the other two."""
    bar = _template("thermal_stress")
    peak = bar.E_PA * bar.ALPHA_PER_K * bar.DELTA_T_K
    page = TEMPLATES_PAGE.read_text(encoding="utf-8")
    assert f"{peak / 1e6:.4f}" == "220.0000"
    assert f"{peak / (1.0 - bar.NU) / 1e6:.4f}" == "314.2857"
    assert "-220.0000 MPa" in page and "-314.2857 MPa" in page
    # and the e-folding time, which is why END_TIME_S is not a round number
    assert f"{bar.expected()['diffusion_e_folding_s']:.4f}" == "2.2834"
    assert "2.2834" in page


def test_the_rotation_rate_on_the_page_is_the_one_constant_mrfproperties_carries():
    """rpm against rad/s is a factor of 9.5493 and `MRFProperties` takes the second one.
    A torque quoted from a case whose omega was entered in rpm is out by that factor and
    looks like a modelling problem."""
    rotating = _template("rotating")
    omega = rotating.ROTOR_RPM * rotating.RAD_S_PER_RPM
    page = TEMPLATES_PAGE.read_text(encoding="utf-8")
    assert f"{omega:.6f}" == "104.719755", f"omega is {omega:.6f} rad/s"
    assert "104.719755 rad/s" in page
    assert "9.5493" in page


def test_the_duct_expressions_on_the_page_are_the_ones_the_journey_asserts():
    """The exact answer lives in `tests/test_journey_duct.py` as algebra and on the tier 2
    page as prose, and the page is what a reader copies."""
    page = TEMPLATES_PAGE.read_text(encoding="utf-8")
    journey = (ROOT / "tests" / "test_journey_duct.py").read_text(encoding="utf-8")
    for form in ("1.5 u_mean", "-12 nu u_mean / H^2", "6 nu u_mean / H"):
        assert form in page, f"the page does not carry {form}"
        assert form in journey, f"{form} is on the page and not in the test that asserts it"
