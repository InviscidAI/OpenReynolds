"""The published reference data the toolbox ships, and the single source it must match.

Why this file exists. Asked to compare a lid-driven cavity against Ghia, Ghia & Shin
(1982), a study on 2026-11-22 (20261122-110157-2985) wrote the table into its own
analysis script from the model's memory, saying so in its reasoning: "I'm recalling the
standard Ghia et al. Re=1000 reference values". It had no alternative. The repository's
copy lives in `benchmarks/felix/grade.py`, which is not part of the toolbox the agent is
given, so there was nothing in `/work/.toolbox` to read.

That run's recalled table turned out to be correct, point for point to five decimals.
That is luck, not a property of the system, and the whole claim this project makes is
that a number was measured against a reference rather than remembered alongside one. So
the table ships in the toolbox, and this test keeps the shipped copy honest: it is
generated from `benchmarks/felix/grade.py` and must stay equal to it, so there is still
one source of truth and not two that can drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "openreynolds" / "toolbox" / "references" / "ghia_re1000.json"


def published():
    sys.path.insert(0, str(ROOT / "benchmarks" / "felix"))
    from grade import GHIA_RE1000_U, GHIA_RE1000_V  # noqa: E402
    return GHIA_RE1000_U, GHIA_RE1000_V


def test_the_shipped_table_is_the_repositorys_own_table():
    """Equal to `benchmarks/felix/grade.py`, which the graders already use. If this
    fails, one of the two was edited and the other was not."""
    shipped = json.loads(DATA.read_text(encoding="utf-8"))
    table_u, table_v = published()
    for key, source in (("table_1_u_on_vertical_centreline", table_u),
                        ("table_2_v_on_horizontal_centreline", table_v)):
        mine = [(round(a, 4), b) for a, b in shipped[key]]
        theirs = [(round(a, 4), b) for a, b in sorted(source)]
        assert mine == theirs, (key, mine[:3], theirs[:3])


def test_it_carries_its_citation_and_its_convention():
    """A table of numbers with no paper and no convention attached is not a reference:
    the ordinate direction is exactly what a reader would otherwise have to guess, and
    guessing it inverts the profile."""
    shipped = json.loads(DATA.read_text(encoding="utf-8"))
    assert "Ghia" in shipped["source"] and "1982" in shipped["source"]
    assert "48:387" in shipped["source"], "the volume and page make it findable"
    assert "lid" in shipped["note"]
    # The convention has to be checkable from the data itself, not only asserted.
    u = dict((round(a, 4), b) for a, b in shipped["table_1_u_on_vertical_centreline"])
    assert u[1.0] == 1.0, "u = 1 at the lid"
    assert u[0.0] == 0.0, "u = 0 at the stationary wall"


def test_every_point_is_inside_the_lid_speed():
    """Both profiles are normalised by the lid speed, so nothing may exceed it."""
    shipped = json.loads(DATA.read_text(encoding="utf-8"))
    for key in ("table_1_u_on_vertical_centreline", "table_2_v_on_horizontal_centreline"):
        for abscissa, value in shipped[key]:
            assert 0.0 <= abscissa <= 1.0, (key, abscissa)
            assert -1.0 <= value <= 1.0, (key, abscissa, value)
