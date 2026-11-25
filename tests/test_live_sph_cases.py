"""The SPH cases, tested at everything that does not need the solver.

`openreynolds/live/sph_cases.py` was written against `docs/sph-contract.md` **before**
`openreynolds/live/sph.py` existed, which is what that contract is for: the interface is
fixed up front so the solver and its grading can be built at the same time. That leaves a
real question about this file -- what can be tested with no solver -- and it has a precise
answer in three parts.

**The analytic references, which are the whole of what a case grades against.** Every
graded number in `sph_cases.py` comes from one of eight closed forms: Ritter's profile and
front, the linear dispersion relation, the heave period, Archimedes' draft, the metacentric
height, the wall-sided heel equilibrium, and the three-point Richardson estimate. All eight
can be computed on paper, and that is how they are checked here -- a square-section prism
at half water's density heels to **exactly 45 degrees**, at a quarter to **exactly
`atan(1/2)`**, and the Richardson estimate reproduces `PROGRESS_LIVE.md`'s own recorded
worked numbers (order 1.232, limit 1.1382 from the lattice dam break's three grids). A
helper that agrees with a hand calculation at a value where the answer is a round number is
a helper whose algebra is right, and these are the only numbers a case can be wrong about
without the solver being wrong.

**The house rules, which are enforced and not described.** Every case carries a milestone
and a sentence; every reference says where it came from, what was actually read, what is
wrong with it and what its tolerance is on; a reference that was **not read** carries no
tolerance and grades nothing; a reference shared with the lattice is **one copy** and not a
second one that can be widened; and every case writes a figure and takes its tolerance from
`REFERENCES` rather than from a literal. `tests/test_live_cases.py` has the same guards for
the lattice cases and this file is their twin.

**That nothing pretends to run.** With no `sph.py` every runner must refuse by name, and
the refusal must say it is an unwritten backend rather than a broken install. And
`_timestep` must refuse rather than guess, because a guessed timestep makes every period,
front speed and decay rate in that module wrong by one unknown factor and leaves all of
them looking plausible.

What is **not** here is any physics: not one number in this file came out of a solver. When
`sph.py` lands, the graded runs go to `benchmarks/live/` on a GPU, the way the lattice
cases' do.
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

GRAVITY = 9.81
MILESTONES = ("M0", "M1", "M2", "M3", "M5", "M6", "M7")


# -- the cases, and whether their grading is honest --------------------------------------


def test_every_case_carries_a_milestone_and_a_sentence():
    """The same guard `tests/test_live_cases.py` puts on the lattice catalogue.

    A case with no milestone cannot be scheduled and a case with no description cannot be
    chosen, and `describe_cases` is what the CLI and the agent tool print.
    """
    from openreynolds.live import sph_cases

    for name, case in sph_cases.SPH_CASES.items():
        assert case.name == name
        assert case.about and len(case.about) > 30, f"{name} has no useful description"
        assert case.milestone in MILESTONES, f"{name} has milestone {case.milestone!r}"
        assert callable(case.runner)


def test_the_milestones_are_the_ones_the_brief_assigns():
    """Free surface M5, floating bodies M6, containers M7.

    Asserted rather than left to the catalogue, because the three buckets are how the work
    is sequenced: the floating cases cannot be attempted before the free-surface ones pass,
    and `PROGRESS_LIVE.md`'s M7 is the carried cup, of which the sloshing tank is the
    static half.
    """
    from openreynolds.live import sph_cases

    by_milestone: dict[str, set[str]] = {}
    for name, case in sph_cases.SPH_CASES.items():
        by_milestone.setdefault(case.milestone, set()).add(name)
    assert by_milestone["M5"] == {"sph_hydrostatic", "sph_dam_break"}
    assert by_milestone["M7"] == {"sph_sloshing"}
    assert "sph_static_float" in by_milestone["M6"]
    assert "sph_heel_stability" in by_milestone["M6"]


def test_the_catalogue_lists_every_case():
    from openreynolds.live import sph_cases

    listed = "\n".join(sph_cases.describe_sph_cases())
    for name in sph_cases.SPH_CASES:
        assert name in listed, f"{name} is a case and the catalogue does not name it"


def test_the_catalogue_is_the_shape_the_lattice_one_is():
    """`CASES.update(SPH_CASES)` has to be all the wiring there is.

    The brief for this module is that registering these cases later is one import, which
    is only true if the entries are `cases.Case` instances with the same fields -- so that
    `cases.run_case`, `describe_cases`, the CLI and the agent tool work unchanged.
    """
    from openreynolds.live import cases, sph_cases

    # `sph_cases` now registers itself into `cases.CASES` as it imports -- that is what
    # broke the cycle, since `cases` cannot import it at module level while it imports
    # `Case` from `cases`. So by the time this test runs the merge has already happened,
    # and the question is no longer "are the two sets disjoint" but "did registering
    # replace anything". Identity answers it: a name present in both must be the **same
    # object**, which is only true if the lattice did not already own that name.
    shadowed = sorted(n for n, c in sph_cases.SPH_CASES.items()
                      if cases.CASES.get(n) is not c)
    assert not shadowed, (
        f"registering the SPH cases did not take for {shadowed}, which means a lattice "
        f"case already owned those names. An SPH case sharing a name with a lattice one "
        f"would mean merging the catalogues silently replaced it"
    )
    lattice_only = set(cases.CASES) - set(sph_cases.SPH_CASES)
    assert not lattice_only & set(sph_cases.SPH_CASES), (
        "an SPH case shares a name with a lattice case, so merging the catalogues would "
        "silently replace one"
    )
    for case in sph_cases.SPH_CASES.values():
        assert isinstance(case, cases.Case)
    merged = {**cases.CASES, **sph_cases.SPH_CASES}
    assert len(merged) == len(cases.CASES), (
        "CASES already holds every SPH case after registration, so re-merging must be a "
        "no-op; a longer result would mean some SPH case is registered under a different "
        "name than its key in SPH_CASES"
    )


def test_an_unknown_sph_case_is_refused_by_name():
    from openreynolds.live import sph_cases
    from openreynolds.live.units import UnphysicalScaling

    with pytest.raises(UnphysicalScaling) as raised:
        sph_cases.run_sph_case("not_a_case")
    assert "not_a_case" in str(raised.value)
    for name in sph_cases.SPH_CASES:
        assert name in str(raised.value), "the refusal does not list the cases"


# -- the references ----------------------------------------------------------------------


def test_every_reference_says_where_it_was_read_and_what_is_wrong_with_it():
    """`benchmarks/README.md`'s rule, applied to this file's own table.

    A reference with no provenance is a number somebody remembered, and this repository has
    four studies on record that did exactly that. The lattice twin of this test is
    `tests/test_live_cases.py::test_every_reference_says_where_it_was_read_and_what_is_wrong_with_it`,
    with one addition: an entry here may be **ungraded**, and then it must carry no
    tolerance at all rather than a number nothing uses.
    """
    from openreynolds.live.sph_cases import REFERENCES

    for name, entry in REFERENCES.items():
        assert entry.get("source"), f"{name} names no source"
        assert entry.get("read"), f"{name} does not say what was actually read"
        assert entry.get("caveat"), f"{name} states no caveat, which no reference has"
        assert entry.get("tolerance_is"), f"{name} does not say what its tolerance is on"
        assert "graded" in entry, (
            f"{name} does not say whether it is graded. Every entry has to, because the "
            f"honest thing to do with a reference that cannot be read is to report it and "
            f"grade nothing against it, and that has to be machine-readable"
        )
        if entry["graded"]:
            tolerance = entry["tolerance"]
            assert isinstance(tolerance, (int, float)) and tolerance > 0, (
                f"{name} is graded and its tolerance is {tolerance!r}"
            )
        else:
            assert entry["tolerance"] is None, (
                f"{name} is not graded and still carries a tolerance of "
                f"{entry['tolerance']!r}. A tolerance nothing grades against is a number "
                f"a reader will assume something does"
            )
            assert "not graded" in entry["tolerance_is"], (
                f"{name} is ungraded and does not say so in tolerance_is"
            )


def test_an_unread_reference_grades_nothing():
    """The house rule: a case that cannot be graded honestly is reported and not graded.

    Both of the references in this module that could not be read are named here rather than
    discovered, so that a later edit which quietly starts grading against one of them fails
    this test instead of passing review.
    """
    from openreynolds.live.sph_cases import REFERENCES

    for name in ("martin_moyce", "bogner_rude"):
        entry = REFERENCES[name]
        assert entry["graded"] is False, f"{name} was not read and is being graded"
        assert entry["tolerance"] is None
        assert "not read" in entry["read"].lower(), (
            f"{name}'s `read` field does not say it was not read, which is the one thing "
            f"it has to say"
        )


def test_the_martin_moyce_caveat_keeps_both_conventions_and_the_factor():
    """The specific thing that makes this reference unusable, asserted as text.

    `t sqrt(g/a)` and `t sqrt(2g/a)` differ by 1.414 at every point, and a front position
    quoted in the wrong one agrees with a solver that is 41% wrong. `DECISIONS_LIVE.md` D14
    is the decision taken on it and `benchmarks/README.md` records the same refusal; the
    number is in the caveat so that nobody has to go and find out again.
    """
    from openreynolds.live.sph_cases import REFERENCES

    caveat = REFERENCES["martin_moyce"]["caveat"]
    assert "sqrt(g/a)" in caveat and "sqrt(2g/a)" in caveat
    assert "1.414" in caveat


def test_a_reference_shared_with_the_lattice_is_one_copy_and_not_two():
    """A duplicated reference is a tolerance that can be widened in one copy only.

    Four of these entries are the same references the lattice free-surface cases use, and
    `_shared` builds them from `cases.REFERENCES` so that the provenance and the tolerance
    have one home. This asserts the sharing is real: if somebody retypes an entry here, the
    fields will drift and this fails.
    """
    from openreynolds.live import cases
    from openreynolds.live.sph_cases import REFERENCES

    for name in ("hydrostatic", "ritter", "linear_sloshing", "archimedes"):
        mine, theirs = REFERENCES[name], cases.REFERENCES[name]
        # `hydrostatic` deliberately overrides the tolerance, because it grades a
        # different quantity -- the next test pins both numbers.
        fields = ("source", "read", "caveat")
        if name != "hydrostatic":
            fields = fields + ("tolerance",)
        for field in fields:
            assert mine[field] == theirs[field], (
                f"{name}.{field} differs from cases.REFERENCES'. These are deliberately "
                f"one copy: a second hand-typed caveat drifts, and a second tolerance can "
                f"be widened without the first one changing"
            )
        assert mine["tolerance_is"] != theirs["tolerance_is"], (
            f"{name} did not override tolerance_is, and what a tolerance is *on* does "
            f"differ between a lattice and a particle method"
        )


def test_the_shared_tolerances_are_the_numbers_the_lattice_grades_at():
    """No widened tolerances, stated as numbers so the test fails if one moves.

    The two cases this backend shares with the lattice are graded at the same bounds, which
    is what makes a comparison between the backends mean anything -- `benchmarks/README.md`
    makes the same point about the cavity, graded identically by `felix/grade.py` and by
    the live backend on the one case they share.
    """
    from openreynolds.live import cases
    from openreynolds.live.sph_cases import REFERENCES

    assert REFERENCES["ritter"]["tolerance"] == 0.15
    assert REFERENCES["linear_sloshing"]["tolerance"] == 0.05
    assert REFERENCES["archimedes"]["tolerance"] == 0.02
    # The one deliberate difference, and this is where it is justified rather than
    # noticed. The lattice grades `p = rho g h` by proxy -- surface height and flatness,
    # at 2% -- because `d(rho)/dz` is its measurable. The contract promises `pressures()`,
    # so the SPH case grades the pressure itself, and the number this repository has
    # measured for *that* quantity is 2.50% RMS (`benchmarks/README.md`'s `sph/` row). A
    # 2% bound would fail a solver matching the best this repository has achieved, which
    # is not a tolerance but a trap; 5% is twice the measurement and the entry says it
    # should be tightened to the measurement once this backend has one.
    assert cases.REFERENCES["hydrostatic"]["tolerance"] == 0.02
    assert REFERENCES["hydrostatic"]["tolerance"] == 0.05
    assert "2.50%" in REFERENCES["hydrostatic"]["tolerance_is"], (
        "the SPH hydrostatic tolerance is looser than the lattice's and must say which "
        "measurement it was set from"
    )


# -- Ritter, which is the dam break's whole reference ------------------------------------


def test_ritters_depth_at_the_dam_is_four_ninths_at_every_time():
    """The one number in a dam break that needs neither a threshold nor a fit.

    Ritter fixes the depth at the dam at exactly `4 h0 / 9`, independent of time, for as
    long as the solution holds -- which is why `DECISIONS_LIVE.md` D14 grades that and not
    the front position.
    """
    from openreynolds.live.sph_cases import ritter_depth

    h0 = 0.05
    for elapsed in (0.01, 0.05, 0.1, 0.5, 2.0):
        assert float(ritter_depth(0.0, elapsed, h0)) == pytest.approx(4.0 * h0 / 9.0,
                                                                      rel=1e-12)


def test_ritters_front_runs_at_twice_the_celerity():
    from openreynolds.live.sph_cases import ritter_front

    h0, elapsed = 0.05, 0.1
    assert ritter_front(elapsed, h0) == pytest.approx(
        2.0 * elapsed * math.sqrt(GRAVITY * h0), rel=1e-12)
    # Worked by hand: sqrt(9.81 * 0.05) = 0.700357..., twice that times 0.1 s.
    assert ritter_front(0.1, 0.05) == pytest.approx(0.1400714, abs=1e-7)
    assert ritter_front(0.0, h0) == 0.0


def test_ritters_profile_is_undisturbed_upstream_and_dry_beyond_the_front():
    """The three regions of the characteristics solution, checked at their boundaries.

    Upstream of `-c0 t` nothing has happened yet, beyond `2 c0 t` the bed is dry, and in
    between the depth falls monotonically. The profile is also continuous at both joins,
    which is the part an off-by-one in the `where` would break while leaving the middle
    right.
    """
    from openreynolds.live.sph_cases import ritter_depth

    h0, elapsed = 0.05, 0.1
    celerity = math.sqrt(GRAVITY * h0)
    x = np.linspace(-4.0 * celerity * elapsed, 4.0 * celerity * elapsed, 801)
    depth = np.asarray(ritter_depth(x, elapsed, h0))

    assert np.allclose(depth[x < -celerity * elapsed], h0)
    assert np.allclose(depth[x > 2.0 * celerity * elapsed], 0.0)
    inside = (x > -celerity * elapsed) & (x < 2.0 * celerity * elapsed)
    assert np.all(np.diff(depth[inside]) <= 1e-12), "the profile is not monotone"
    # Continuous at the upstream join, where the wave meets still water.
    assert float(ritter_depth(-celerity * elapsed, elapsed, h0)) == pytest.approx(h0,
                                                                                  rel=1e-12)
    # And at the front, where it reaches zero rather than jumping to it.
    assert float(ritter_depth(2.0 * celerity * elapsed, elapsed, h0)) == pytest.approx(
        0.0, abs=1e-15)


def test_ritters_tongue_is_why_the_front_is_not_graded():
    """D14's measured reason, reproduced from the formula rather than quoted.

    "Where the front is" depends entirely on how little water still counts as water: the
    place where Ritter's own profile reaches half a particle spacing of depth is far behind
    the true front, and the gap is the reference's and not the solver's. This checks the
    effect is large, which is what makes grading the front dishonest.
    """
    from openreynolds.live.sph_cases import ritter_depth, ritter_front

    h0, elapsed, spacing = 0.05, 0.1, 0.005
    true_front = ritter_front(elapsed, h0)
    x = np.linspace(0.0, true_front, 20001)
    depth = np.asarray(ritter_depth(x, elapsed, h0))
    apparent = float(x[depth > 0.5 * spacing][-1])
    assert apparent < true_front
    assert (true_front - apparent) / true_front > 0.1, (
        "if the threshold moved the front by less than 10% there would be no reason not "
        "to grade it"
    )


# -- the linear dispersion relation, which is the sloshing tank's reference ---------------


def test_the_dispersion_relation_at_the_tank_the_lattice_case_uses():
    """`omega^2 = g k tanh(k h)` at L = 100 mm, h = 30 mm, worked to five decimals.

    The same tank `cases.py::run_sloshing` defaults to, so the two backends are graded
    against the same number: `PROGRESS_LIVE.md` and `benchmarks/README.md` both record
    theory as 0.417086 s there.
    """
    from openreynolds.live.sph_cases import linear_sloshing

    mode = linear_sloshing(0.1, 0.03)
    assert mode["wavenumber"] == pytest.approx(math.pi / 0.1, rel=1e-12)
    assert mode["kh"] == pytest.approx(0.9424778, abs=1e-7)
    assert mode["period"] == pytest.approx(0.417086, abs=2e-6)
    assert mode["omega"] == pytest.approx(2.0 * math.pi / mode["period"], rel=1e-12)


def test_the_dispersion_relation_has_the_right_two_limits():
    """Deep water and shallow water, both of which are closed forms of their own.

    `kh` large gives `T = 2 pi / sqrt(g k)` and `kh` small gives the shallow-water
    `T = 2 L / sqrt(g h)`. A sign or a factor misplaced in the `tanh` passes at one depth
    and fails at least one of these.
    """
    from openreynolds.live.sph_cases import linear_sloshing

    deep = linear_sloshing(0.1, 5.0)
    k = math.pi / 0.1
    assert deep["period"] == pytest.approx(2.0 * math.pi / math.sqrt(GRAVITY * k),
                                           rel=1e-6)
    shallow = linear_sloshing(1.0, 1.0e-4)
    assert shallow["period"] == pytest.approx(
        2.0 * 1.0 / math.sqrt(GRAVITY * 1.0e-4), rel=1e-6)


def test_the_release_velocity_is_the_modes_own_and_the_depth_average_is_bounded():
    """What `run_sph_sloshing` releases, since the contract cannot displace a surface.

    At the instant a standing wave's surface is flat, all its energy is kinetic and the
    surface velocity amplitude is `a omega / tanh(kh)` -- equivalently `a g k / omega`,
    and the two forms agreeing is the check that neither was mistyped. The depth-average
    factor `tanh(kh)/(kh)` is a projection of the mode's `cosh` profile and must lie in
    (0, 1]: the case pays for the difference in higher modes, and a factor above 1 would
    mean it was adding energy.
    """
    from openreynolds.live.sph_cases import linear_sloshing

    mode = linear_sloshing(0.1, 0.03)
    assert mode["velocity_per_elevation"] == pytest.approx(
        GRAVITY * mode["wavenumber"] / mode["omega"], rel=1e-12)
    assert 0.0 < mode["depth_average"] <= 1.0
    # Shallow water is nearly depth-uniform, so the projection is nearly perfect there.
    assert linear_sloshing(1.0, 1.0e-4)["depth_average"] == pytest.approx(1.0, abs=1e-6)


# -- the heave period and Archimedes' draft ----------------------------------------------


def test_the_heave_period_is_two_pi_root_draft_over_g():
    """Worked by hand at a 20 mm draft: 2 pi sqrt(0.02/9.81) = 0.283701 s.

    And the scaling, which is the content of the formula: the waterplane area cancels, so
    the period depends on the draft alone and goes as its square root. A box twice as deep
    in the water heaves `sqrt(2)` times more slowly whatever its beam.
    """
    from openreynolds.live.sph_cases import heave_period

    assert heave_period(0.02) == pytest.approx(0.2837007, abs=1e-7)
    assert heave_period(0.04) / heave_period(0.02) == pytest.approx(math.sqrt(2.0),
                                                                    rel=1e-12)
    assert heave_period(0.02, gravity=4.0 * GRAVITY) == pytest.approx(
        heave_period(0.02) / 2.0, rel=1e-12)


def test_archimedes_draft_is_the_density_ratio_times_the_height():
    from openreynolds.live.sph_cases import archimedes_draft

    assert archimedes_draft(0.5, 0.032) == pytest.approx(0.016, rel=1e-12)
    assert archimedes_draft(1.0, 0.032) == pytest.approx(0.032, rel=1e-12)


# -- the metacentric criterion and the heel equilibrium ----------------------------------


def test_a_square_section_prism_at_half_waters_density_cannot_float_upright():
    """The arithmetic that forced the static float's geometry, at numbers that are exact.

    `KB = d/2 = H/4`, `BM = b^2/(12 d) = H/6`, `KG = H/2`, so `GM = -H/12`: negative. A
    cube at half water's density **capsizes**, which is why `cases.py::run_archimedes`
    holds its cube square to the water and why `run_sph_static_float` uses a beam twice its
    height instead.
    """
    from openreynolds.live.sph_cases import metacentric_height

    hydro = metacentric_height(1.0, 1.0, 0.5)
    assert hydro["KB"] == pytest.approx(0.25, rel=1e-12)
    assert hydro["BM"] == pytest.approx(1.0 / 6.0, rel=1e-12)
    assert hydro["KG"] == pytest.approx(0.5, rel=1e-12)
    assert hydro["GM"] == pytest.approx(-1.0 / 12.0, rel=1e-12)
    assert hydro["stable"] is False

    raft = metacentric_height(2.0, 1.0, 0.5)
    assert raft["BM"] == pytest.approx(4.0 / 6.0, rel=1e-12)
    assert raft["GM"] == pytest.approx(0.25 + 4.0 / 6.0 - 0.5, rel=1e-12)
    assert raft["stable"] is True


def test_the_stability_criterion_is_the_sign_of_gm_everywhere_it_is_defined():
    """`(b/H)^2 > 6 r (1-r)` is `GM > 0` rearranged, so they must never disagree.

    Checked over a grid rather than at a point, because the rearrangement divides by the
    draft and a dropped factor would survive at one aspect ratio. This is the identity the
    heel case's graded sign check rests on.
    """
    from openreynolds.live.sph_cases import metacentric_height

    for slenderness in (0.5, 0.8165, 0.9, 1.0, 1.2247, 2.0, 4.0):
        for ratio in (0.05, 0.25, 0.4, 0.5, 0.6, 0.75, 0.95):
            hydro = metacentric_height(slenderness, 1.0, ratio)
            predicted = hydro["slenderness"] > hydro["criterion"]
            assert hydro["stable"] == predicted, (
                f"GM {hydro['GM']:+.6f} disagrees with the criterion at b/H "
                f"{slenderness} and r {ratio}"
            )


def test_the_heel_equilibrium_comes_out_at_the_angles_that_are_exact():
    """Two geometries whose answer is a round number, and one that is the default.

    A square-section prism at half water's density: `tan^2 a = 12 (0.25) (1) - 2 = 1`, so
    **exactly 45 degrees**. At a quarter (or three quarters, by the symmetry of `r(1-r)`):
    `tan^2 a = 12 (0.1875) - 2 = 0.25`, so **exactly `atan(1/2)`**, 26.565 degrees. The
    heel case's default hull is `H/b = 0.9` at half density, which gives `tan^2 a = 0.43`.
    """
    from openreynolds.live.sph_cases import heel_equilibrium

    square = heel_equilibrium(1.0, 1.0, 0.5)
    assert math.degrees(square["angle"]) == pytest.approx(45.0, abs=1e-9)
    assert square["tan"] == pytest.approx(1.0, rel=1e-12)

    quarter = heel_equilibrium(1.0, 1.0, 0.25)
    assert quarter["tan"] == pytest.approx(0.5, rel=1e-12)
    assert math.degrees(quarter["angle"]) == pytest.approx(26.5650512, abs=1e-6)
    assert heel_equilibrium(1.0, 1.0, 0.75)["tan"] == pytest.approx(0.5, rel=1e-12)

    default = heel_equilibrium(0.045, 0.0405, 0.5)
    assert default["tan"] == pytest.approx(math.sqrt(0.43), rel=1e-9)
    assert math.degrees(default["angle"]) == pytest.approx(33.2546129, abs=1e-6)


def test_a_heeled_equilibrium_exists_exactly_when_upright_is_unstable():
    """The identity the heel case's two graded signs rest on.

    `tan^2(a_eq) = 12 r (1-r) (H/b)^2 - 2` is positive precisely when
    `(b/H)^2 < 6 r (1-r)`, which is the metacentric criterion reversed. If these two ever
    disagreed, the case would be grading a hull as unstable while computing no angle for
    it to reach.
    """
    from openreynolds.live.sph_cases import heel_equilibrium

    for slenderness in (0.5, 0.8, 0.9, 1.0, 1.1, 1.5, 3.0):
        for ratio in (0.1, 0.25, 0.5, 0.75, 0.9):
            heeled = heel_equilibrium(slenderness, 1.0, ratio)
            assert math.isnan(heeled["angle"]) == bool(heeled["stable"]), (
                f"at b/H {slenderness} and r {ratio}, stable={heeled['stable']} and the "
                f"heel angle is {heeled['angle']}"
            )
            assert heeled["upright_stable"] == heeled["stable"]


def test_the_wall_sided_window_is_reported_and_the_defaults_sit_inside_it():
    """The closed form holds only while the waterline stays on the two vertical sides.

    `tan a <= 2 min(d, H-d)/b`, and at beam = height the equilibrium sits exactly **on**
    that limit -- where the formula is not a statement about anything, because the keel
    corner is just emerging. The heel case's default geometry was chosen to have a margin,
    and this is the assertion that it does.
    """
    from openreynolds.live.sph_cases import heel_equilibrium

    square = heel_equilibrium(1.0, 1.0, 0.5)
    assert square["margin"] == pytest.approx(0.0, abs=1e-12), (
        "a square section at half density heels exactly to its wall-sided limit, which is "
        "why it is not the geometry the case uses"
    )
    default = heel_equilibrium(0.045, 0.0405, 0.5)
    assert default["wall_sided_limit"] == pytest.approx(0.9, rel=1e-12)
    assert default["margin"] > 0.2, "the default hull has no wall-sided margin"


def test_the_two_hulls_the_heel_case_builds_are_on_opposite_sides_of_the_criterion():
    """The case needs one hull that cannot float upright and one that can.

    A solver that capsizes everything passes the first check and fails the second, which is
    the only reason there are two hulls. Read from the runner's own defaults so that the
    test cannot drift from the geometry the case actually runs.
    """
    from openreynolds.live.sph_cases import (heel_equilibrium, metacentric_height,
                                             run_sph_heel_stability)

    defaults = {name: parameter.default for name, parameter
                in inspect.signature(run_sph_heel_stability).parameters.items()}
    beam, ratio = defaults["beam"], defaults["density_ratio"]
    height = defaults["slenderness"] * beam
    assert not heel_equilibrium(beam, height, ratio)["stable"]
    assert metacentric_height(defaults["stable_beam"], height, ratio)["stable"]


def test_the_static_floats_default_hull_can_actually_float_upright():
    """And the case refuses a geometry that cannot, rather than measuring a capsize.

    A draft is only a draft if the hull stays upright, so `run_sph_static_float` checks the
    criterion before it builds anything. This asserts both halves: the default passes, and
    a cube at half density is refused with the arithmetic in the message.
    """
    from openreynolds.live.sph_cases import metacentric_height, run_sph_static_float
    from openreynolds.live.units import UnphysicalScaling

    defaults = {name: parameter.default for name, parameter
                in inspect.signature(run_sph_static_float).parameters.items()}
    assert metacentric_height(defaults["beam"], defaults["height"],
                              defaults["density_ratio"])["stable"]

    with pytest.raises(UnphysicalScaling) as raised:
        run_sph_static_float(beam=0.04, height=0.04, density_ratio=0.5)
    said = str(raised.value)
    assert "cannot float upright" in said and "6 r (1-r)" in said


def test_the_heave_case_refuses_the_same_geometry_for_the_same_reason():
    """A hull that capsizes has no heave equilibrium to decay about.

    The two cases share a geometry, so a mistake in it has to fail in both -- a heave
    period measured off a capsizing box is a number with nothing behind it.
    """
    from openreynolds.live.sph_cases import run_sph_heave_decay
    from openreynolds.live.units import UnphysicalScaling

    with pytest.raises(UnphysicalScaling) as raised:
        run_sph_heave_decay(beam=0.04, height=0.04, density_ratio=0.5)
    assert "no heave equilibrium" in str(raised.value)


# -- the three-point Richardson estimate -------------------------------------------------


def test_richardson_recovers_a_synthetic_second_order_triple_exactly():
    """`f(h) = 10 + h^2` at h = 1, 2, 4 gives 11, 14, 26: order 2 and limit 10, exactly.

    The cleanest possible check of the algebra, with no reference and no solver:
    `ln((26-14)/(14-11))/ln 2 = ln 4 / ln 2 = 2`, and
    `11 + (11-14)/(2^2 - 1) = 11 - 1 = 10`.
    """
    from openreynolds.live.sph_cases import richardson

    fit = richardson([11.0, 14.0, 26.0], 2.0)
    assert fit["monotone"] is True
    assert fit["order"] == pytest.approx(2.0, rel=1e-12)
    assert fit["limit"] == pytest.approx(10.0, rel=1e-12)


def test_richardson_reproduces_the_worked_estimate_this_repository_recorded():
    """`PROGRESS_LIVE.md`'s lattice dam break: order 1.232 and limit 1.1382.

    The three grids are 325, 650 and 1300 cells -- a constant refinement ratio of 2 -- and
    the depth at the dam reads 1.2450, 1.1837 and 1.1576 of Ritter's `4 h0/9`. The document
    records an observed order of **1.232** and a Richardson limit of **1.1382** from them,
    and this is the test that `richardson` is the same estimator that produced those, so a
    number in the prose and a number in the code cannot drift apart.
    """
    from openreynolds.live.sph_cases import richardson

    fit = richardson([1.1576, 1.1837, 1.2450], 2.0)
    assert fit["order"] == pytest.approx(1.232, abs=5e-4)
    assert fit["limit"] == pytest.approx(1.1382, abs=5e-5)


def test_richardson_refuses_two_values_because_the_two_grid_form_is_circular():
    """The mistake `PROGRESS_LIVE.md` records, made impossible rather than documented.

    A two-grid estimate needs the exact answer supplied, so feeding it the reference and
    reading back "the limit is the reference" proves nothing -- it returned 1.0000 for
    three different cases, which is what gave it away. The refusal says that, so that
    whoever reaches for the two-point form finds out why not.
    """
    from openreynolds.live.sph_cases import richardson
    from openreynolds.live.units import UnphysicalScaling

    with pytest.raises(UnphysicalScaling) as raised:
        richardson([1.1576, 1.1837], 2.0)
    said = str(raised.value)
    assert "three" in said
    assert "exact answer" in said or "circular" in said or "reference" in said

    with pytest.raises(UnphysicalScaling):
        richardson([1.0, 1.1, 1.2], 1.0)


def test_richardson_reports_a_non_monotone_triple_instead_of_fitting_it():
    """A triple that turns around has no order that fits it, and the formula is silent.

    `(f3-f2)/(f2-f1)` is negative, its logarithm is not a real number, and numpy hands back
    `nan` without comment -- so a reader would see a missing number rather than a
    non-convergent run. `monotone` is returned for the case to grade, which it does.
    """
    from openreynolds.live.sph_cases import richardson

    fit = richardson([1.05, 1.20, 1.10], 2.0)
    assert fit["monotone"] is False
    assert math.isnan(fit["order"])
    assert math.isnan(fit["limit"])


def test_richardson_handles_convergence_from_below_as_well_as_above():
    """Monotone means monotone, not decreasing: an error that approaches from below counts.

    `f(h) = 10 - h^2` at 1, 2, 4 gives 9, 6, -6, and the same estimate must return order 2
    and limit 10. An implementation that tested `f1 < f2 < f3` rather than the signs of the
    two differences would reject this and quietly drop half the cases that converge.
    """
    from openreynolds.live.sph_cases import richardson

    fit = richardson([9.0, 6.0, -6.0], 2.0)
    assert fit["monotone"] is True
    assert fit["order"] == pytest.approx(2.0, rel=1e-12)
    assert fit["limit"] == pytest.approx(10.0, rel=1e-12)


# -- the force-noise arithmetic ----------------------------------------------------------


def test_the_noise_fraction_is_the_rms_about_the_mean_over_the_mean():
    """A signal whose answer can be written down: mean 1, deviations +/- 0.1 alternating.

    RMS is then exactly 0.1 and the fraction exactly 10%, which is the bound the case
    grades at -- so this also pins what "10% noise" means before anything is graded on it.
    """
    from openreynolds.live.sph_cases import noise_fraction

    alternating = [1.1, 0.9] * 50
    measured = noise_fraction(alternating)
    assert measured["mean"] == pytest.approx(1.0, rel=1e-12)
    assert measured["rms"] == pytest.approx(0.1, rel=1e-12)
    assert measured["fraction"] == pytest.approx(0.1, rel=1e-12)


def test_averaging_reduces_white_noise_by_about_the_root_of_the_window():
    """What the case reports as *measured* rather than assuming, checked where it is true.

    White noise averaged over `n` samples falls as `sqrt(n)`; the case prints the measured
    reduction instead of `sqrt(n)` precisely because correlated noise does not, and this is
    the control that says the measurement would see the white case correctly.
    """
    from openreynolds.live.sph_cases import noise_fraction

    rng = np.random.default_rng(20261123)
    white = 1.0 + 0.1 * rng.standard_normal(20000)
    raw = noise_fraction(white, window=1)
    smoothed = noise_fraction(white, window=10)
    assert raw["fraction"] == pytest.approx(0.1, rel=0.05)
    assert raw["fraction"] / smoothed["fraction"] == pytest.approx(math.sqrt(10.0),
                                                                   rel=0.1)

    # And correlated noise does not fall that fast, which is the whole reason the case
    # measures the ratio: a constant offset is perfectly correlated and averaging it
    # changes nothing at all.
    correlated = 1.0 + 0.1 * np.sign(np.sin(np.arange(20000) / 500.0))
    slow = noise_fraction(correlated, window=10)
    assert noise_fraction(correlated)["fraction"] / slow["fraction"] < 1.5


def test_the_noise_fraction_survives_too_few_samples_and_a_zero_mean():
    """Both degenerate inputs return a reported `nan` rather than raising mid-run.

    A case that has sampled two steps when something went wrong should print a missing
    number and the rest of its report, not lose the report to a ZeroDivisionError.
    """
    from openreynolds.live.sph_cases import noise_fraction

    assert math.isnan(noise_fraction([1.0])["fraction"])
    assert math.isinf(noise_fraction([1.0, -1.0, 1.0, -1.0])["fraction"])


# -- the house rules, enforced ----------------------------------------------------------


def test_every_case_writes_a_figure_and_reads_its_tolerance_from_the_references():
    """Two of this repository's rules, checked against each runner's own source.

    Every case needs a figure: `benchmarks/README.md` and the project's own site tooling
    both turn on it, and a case with no figure is a number with nothing to look at. And
    every graded tolerance comes out of `REFERENCES`, never a literal -- which is what
    makes `test_the_shared_tolerances_are_the_numbers_the_lattice_grades_at` above able to
    pin them.
    """
    from openreynolds.live import sph_cases

    for name, case in sph_cases.SPH_CASES.items():
        source = inspect.getsource(case.runner)
        assert "figures." in source, f"{name} writes no figure"
        assert "ensure_rgb" in source, (
            f"{name} does not flatten its figure to RGB, and an RGBA PNG renders as a "
            f"black panel wherever a thumbnailer composites it"
        )
        assert "REFERENCES[" in source, (
            f"{name} does not read a tolerance from REFERENCES, so the number it grades "
            f"at is a literal that no test can pin"
        )


def test_every_decision_this_module_cites_is_written_down():
    """`DECISIONS_LIVE.md D14` in a docstring is a promise that D14 exists.

    The same guard `tests/test_live_cases.py` applies repository-wide, narrowed to this
    module so that a failure here names this file. It did not hold once: three decisions
    were cited from five files while the document stopped at D7, so a reader following the
    pointer found nothing and had to take the surrounding sentence on trust.
    """
    document = ROOT / "DECISIONS_LIVE.md"
    assert document.is_file(), "DECISIONS_LIVE.md is gone"
    written = set(re.findall(r"^## (D\d+)\.", document.read_text(encoding="utf-8"), re.M))
    assert written, "DECISIONS_LIVE.md has no numbered decisions"

    module = ROOT / "openreynolds" / "live" / "sph_cases.py"
    cited = set(re.findall(r"DECISIONS_LIVE\.md\W{0,3}(D\d+)",
                           module.read_text(encoding="utf-8")))
    assert cited, "this module cites no decisions at all, which would be a surprise"
    assert not (cited - written), (
        f"openreynolds/live/sph_cases.py cites {', '.join(sorted(cited - written))}, "
        f"which DECISIONS_LIVE.md does not contain"
    )


def test_what_the_cases_need_from_the_solver_is_recorded_as_data():
    """The contract gaps are a list the report can print, not a paragraph in a docstring.

    The brief asks what these cases need that `docs/sph-contract.md` does not promise, and
    the answer belongs where the solver's author will see it: in the module, in the result
    of every run, and in a shape that can lose an entry when the contract grows one.
    """
    from openreynolds.live.sph_cases import NEEDS_FROM_THE_SOLVER

    assert len(NEEDS_FROM_THE_SOLVER) >= 5
    for entry in NEEDS_FROM_THE_SOLVER:
        assert len(entry) > 80, f"a one-line need is not a need: {entry!r}"
    joined = " ".join(NEEDS_FROM_THE_SOLVER)
    # The two that stop a case rather than costing it accuracy.
    assert "timestep" in joined and "surface_height" in joined


# -- that nothing pretends to run --------------------------------------------------------


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec(
        "openreynolds.live.sph") is not None,
    reason="the SPH backend has landed, so these cases can be run for real")
@pytest.mark.parametrize("name", sorted({"sph_hydrostatic", "sph_dam_break",
                                         "sph_sloshing"}))
def test_a_case_refuses_by_name_while_the_backend_is_unwritten(name, tmp_path):
    """With no `sph.py`, a case must say so rather than fail somewhere inside numpy.

    These cases were written against the contract before the solver, which is the point of
    fixing an interface up front -- so "the backend is not here" is an expected state and
    the message distinguishes it from a broken install.
    """
    from openreynolds.live import sph_cases
    from openreynolds.live.units import UnphysicalScaling

    with pytest.raises(UnphysicalScaling) as raised:
        sph_cases.run_sph_case(name, out_dir=tmp_path)
    said = str(raised.value)
    assert "sph.py" in said and "sph-contract" in said


def test_the_timestep_is_refused_rather_than_guessed():
    """The one contract gap that stops every case, and it fails loudly.

    `step()` returns `seconds`, which is wall clock, and the contract promises no simulated
    time. A guessed timestep would make every period, every front speed and every decay
    rate in this module wrong by one unknown factor and leave all of them looking
    plausible, so `_timestep` refuses and names what would fix it.
    """
    from openreynolds.live import sph_cases
    from openreynolds.live.units import UnphysicalScaling

    class _Mute:
        pass

    with pytest.raises(UnphysicalScaling) as raised:
        sph_cases._timestep(_Mute())
    said = str(raised.value)
    assert "dt" in said and "wall clock" in said

    class _Told:
        dt = 2.5e-5

    assert sph_cases._timestep(_Told()) == pytest.approx(2.5e-5)


def test_the_period_and_damping_estimators_are_the_lattices_and_not_a_second_copy():
    """`cases.py::_period_of` and `_damping_of` carry the runs that forced them.

    A second copy here would be a second pair of estimators with the measurements that
    justify them left behind in the other file: the crossings winning on one run and the
    FFT on the next, and a damped period being longer by `1/sqrt(1 - zeta^2)`. So the
    sloshing and heave cases import them, and this is the assertion that they do.
    """
    from openreynolds.live import cases, sph_cases

    for name in ("run_sph_sloshing", "run_sph_heave_decay"):
        source = inspect.getsource(getattr(sph_cases, name))
        assert "from .cases import _damping_of, _period_of" in source, (
            f"{name} does not import the lattice estimators"
        )
    # And they still exist to be imported, which a rename in cases.py would break here
    # rather than at the first run on a GPU.
    assert callable(cases._period_of) and callable(cases._damping_of)


def test_the_two_estimators_agree_on_a_clean_sine_of_a_known_period():
    """A sanity check on the imported pair, at a signal whose period is written down.

    Not a test of `cases.py` -- that file has its own -- but of the one thing this module
    assumes about them: that on a clean four-cycle oscillation they both return the period
    it was built with, so that the sloshing case's graded *spread* between them means what
    it says.
    """
    from openreynolds.live.cases import _period_of

    period = 0.4171
    times = np.linspace(0.0, 4.0 * period, 241)
    signal = 0.002 * np.cos(2.0 * math.pi * times / period)
    by_fft, _ = _period_of(times, signal)
    by_crossings, _ = _period_of(times, signal, method="crossings")
    assert by_fft == pytest.approx(period, rel=0.05)
    assert by_crossings == pytest.approx(period, rel=0.05)
