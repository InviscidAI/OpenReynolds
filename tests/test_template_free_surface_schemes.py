"""The free-surface template must write a compression scheme `interFoam` can find.

Why this exists. The site said of this template that nothing had ever solved it. The
reason turned out not to be that nobody had bothered: the case it wrote could not start.
`interFoam` looks up a scheme for the interface-compression flux, which is named
`div(phirb,alpha)` in OpenFOAM ESI -- the line the workspace image runs -- and
`div(phir,alpha,alpha)` in the Foundation line. The template wrote a regex for the
Foundation name only, so on ESI the lookup found nothing and the solver stopped at the
first step.

Study 20261122-142104-b7fe was the first run to try it, hit exactly that, added the
entry by hand and went on to solve the case. This pins the fix so the template ships
able to run.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox" / "templates" / "free_surface.py"


def template():
    spec = importlib.util.spec_from_file_location("free_surface_template", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def div_patterns() -> list[str]:
    """The quoted regexes in divSchemes, as OpenFOAM would read them."""
    text = template().fv_schemes()
    block = text[text.index("divSchemes"):text.index("laplacianSchemes")]
    return re.findall(r'"([^"]+)"', block)


def test_a_scheme_matches_the_name_esi_actually_asks_for():
    """`div(phirb,alpha)` is the ESI name, and the workspace image is ESI."""
    assert any(re.fullmatch(p, "div(phirb,alpha)") for p in div_patterns()), div_patterns()


def test_a_scheme_still_matches_the_foundation_name():
    """So the template keeps working on the other line rather than trading one for the
    other, which would be the same bug facing the other way."""
    assert any(re.fullmatch(p, "div(phir,alpha,alpha)") for p in div_patterns()), div_patterns()


def test_exactly_one_pattern_matches_each_so_the_lookup_is_not_ambiguous():
    """Two patterns that both match is a dictionary ambiguity, not extra safety."""
    for name in ("div(phirb,alpha)", "div(phir,alpha,alpha)"):
        hits = [p for p in div_patterns() if re.fullmatch(p, name)]
        assert len(hits) == 1, (name, hits)


def test_the_plain_alpha_flux_is_still_there():
    """The compression term is the counter-diffusion; this is the transport it opposes.
    Losing it while fixing the other would make the interface diffuse away."""
    assert any(re.fullmatch(p, "div(phi,alpha)") for p in div_patterns()), div_patterns()
