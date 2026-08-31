"""preflight's `method` check: can this discretisation produce the answer asked for?

Every other preflight check asks whether a case will run. This one asks whether,
having run, it can show the feature it was commissioned to show, and it exists
because of one study that got every other question right. The ONERA M6 replication of
2026-08-30 built a 1.79M-cell mesh with a refinement box through the supersonic
pocket, put the farfield sixteen chords out, and converged to six significant figures
in drag -- and showed no shock at any of seven span stations, because a pressure-based
steady solver with fully limited gradients and an upwinded pressure flux cannot hold
one. The whole failure was visible in `system/fvSchemes` before anything was meshed.

So the case built here is that exact setup, and the test asserts all three of its
defects are named separately: the solver class, the pressure flux, the limiter. Kept
apart because the repairs are different, and a single "transonic is hard" verdict
would not be actionable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def preflight():
    spec = importlib.util.spec_from_file_location("preflight_method", TOOLBOX / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(root: Path, application: str, schemes: str) -> Path:
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "system" / "controlDict").write_text(
        f"application     {application};\nstartTime       0;\nendTime         100;\n"
    )
    (root / "system" / "fvSchemes").write_text(schemes)
    return root


SMEARING = """\
gradSchemes { default cellLimited Gauss linear 1; }
divSchemes
{
    div(phi,U)      bounded Gauss linearUpwind limited;
    div(phid,p)     Gauss upwind;
}
"""

SHARP = """\
gradSchemes { default cellLimited Gauss linear 0.33; }
divSchemes
{
    div(phi,U)      Gauss vanLeer;
    div(phid,p)     Gauss vanLeer;
}
"""


def statuses(findings):
    return [f.status for f in findings]


def measured(findings):
    return " | ".join(f.measured for f in findings)


def test_the_onera_setup_is_told_it_cannot_show_a_shock(preflight, tmp_path):
    case = write(tmp_path / "m6", "rhoSimpleFoam", SMEARING)
    found = preflight.run_checks(case, ["method"], preflight.Intent(resolve="shock"))

    assert "fail" in statuses(found)
    text = measured(found)
    assert "rhoSimpleFoam" in text and "pressure-based" in text
    assert "div(phid,p)" in text, "the pressure flux must be named, not just the solver"
    assert "cellLimited" in text, "the fully limited gradient must be named too"
    assert len(found) == 3, "three defects, three findings, three different repairs"


def test_the_pressure_flux_is_reported_even_though_momentum_looks_modern(preflight, tmp_path):
    """The trap this check was written for.

    Upgrading `div(phi,U)` to linearUpwind while `div(phid,p)` stays upwind reads in a
    report as "less dissipative schemes" and is not, at the only term that carries the
    compression. That is what the real study did between its two rounds.
    """
    case = write(tmp_path / "m6", "rhoCentralFoam", SMEARING)
    found = preflight.run_checks(case, ["method"], preflight.Intent(resolve="shock"))

    solver = [f for f in found if "density-based" in f.measured]
    assert solver and solver[0].status == "ok", "the solver alone is fine here"
    flux = [f for f in found if "pressure flux is first order" in f.measured]
    assert flux and flux[0].status == "fail", "and the scheme is still wrong"


def test_a_density_based_solver_with_sharp_schemes_passes(preflight, tmp_path):
    case = write(tmp_path / "m6", "rhoCentralFoam", SHARP)
    found = preflight.run_checks(case, ["method"], preflight.Intent(resolve="shock"))
    assert set(statuses(found)) == {"ok"}


def test_nothing_declared_is_skipped_not_passed(preflight, tmp_path):
    """Silence from a checker reads as approval, so an unstated deliverable says so."""
    case = write(tmp_path / "m6", "rhoSimpleFoam", SMEARING)
    found = preflight.run_checks(case, ["method"], preflight.Intent())
    assert statuses(found) == ["skipped"]
    assert "no feature named" in measured(found)


def test_a_feature_with_no_rule_is_skipped_rather_than_guessed(preflight, tmp_path):
    case = write(tmp_path / "m6", "rhoSimpleFoam", SMEARING)
    found = preflight.run_checks(case, ["method"], preflight.Intent(resolve="cavitation"))
    assert statuses(found) == ["skipped"]
    # `measured` is evidence and `meaning` is interpretation, and preflight keeps the
    # two apart on purpose -- so "there is no rule for this" belongs in the second.
    assert "cavitation" in measured(found)
    assert "no rule" in found[0].meaning
    assert "not evidence that the method is adequate" in found[0].meaning


def test_a_single_phase_solver_cannot_resolve_a_free_surface(preflight, tmp_path):
    case = write(tmp_path / "hull", "simpleFoam", SHARP)
    found = preflight.run_checks(case, ["method"], preflight.Intent(resolve="interface"))
    assert statuses(found) == ["fail"]
    assert "double-body" in found[0].repair, "the honest alternative is worth naming"


def test_a_loose_alpha_courant_number_is_a_warning(preflight, tmp_path):
    """maxAlphaCo 5 came from a tutorial chasing throughput; it smears the wave."""
    case = tmp_path / "hull"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text(
        "application     interFoam;\nmaxCo           5;\nmaxAlphaCo      5;\n"
    )
    (case / "system" / "fvSchemes").write_text(SHARP)
    found = preflight.run_checks(case, ["method"], preflight.Intent(resolve="interface"))
    assert "warn" in statuses(found)
    assert "maxAlphaCo = 5" in measured(found)


def test_scheme_entries_keeps_keys_that_hold_brackets(preflight):
    """The parse bug that made this check silently pass while it was being written.

    `entry_values` matches keys with `\\w+`, so on `div(phid,p) Gauss upwind;` it
    returns `{"Gauss": "upwind"}` -- the entry being looked for is gone and the check
    reports nothing wrong.
    """
    body = "div(phi,U) bounded Gauss linearUpwind limited; div((phi|interpolate(rho)),p) Gauss upwind;"
    entries = dict(preflight._scheme_entries(body))
    assert "div(phi,U)" in entries
    assert "div((phi|interpolate(rho)),p)" in entries
    assert entries["div((phi|interpolate(rho)),p)"] == "Gauss upwind"


def test_method_is_in_the_check_table_and_runs_first(preflight):
    assert "method" in preflight.CHECKS
    assert preflight.CHECK_ORDER[0] == "method", (
        "it is the cheapest question and the earliest one: it needs no mesh"
    )


def test_the_docstring_keeps_the_check_advisory(preflight):
    """The free-will contract reaches here too: preflight suggests, never blocks."""
    text = preflight.check_method.__doc__
    assert "edits nothing and blocks nothing" in text
    assert "the reading can be wrong" in text
