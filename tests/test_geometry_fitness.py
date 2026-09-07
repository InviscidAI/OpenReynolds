"""The fitness table (DESIGN.md 3.11, D14, D34): measured from a checkMesh log and the
sizes the case was written with, never asserted. A missing number is `Unmeasured(reason)`,
a stated absence; None and NaN are refused at construction; the 3D branch carries an
Unmeasured passage whose reason names Phase 1. The log is the one `test_geometry_agent.py`
feeds the finish, parsed by the toolbox's own `mesh_digest`, so the numbers here are the
numbers the desk sees."""
from __future__ import annotations

import json
import math

import pytest

from openreynolds.geometry import _toolbox, fitness
from openreynolds.geometry.fitness import FitnessTable, Unmeasured, from_digest, schemes_for
from openreynolds.geometry.strategy import Sizes

from test_geometry_agent import CHECKMESH_LOG

SIZES = Sizes(cell=4.75e-4, wall_cell=4.75e-4, thickness=4.75e-4)
PASSAGE_M = 2.98e-3
"""A valve's narrowest passage as `measure.passage_widths` would report it, in metres."""


def digest(log: str = CHECKMESH_LOG) -> dict:
    return _toolbox.load("mesh_digest").parse(log)


def table(**over) -> FitnessTable:
    fields = dict(cells=2611, hex_fraction=1.0, smallest_passage_m=PASSAGE_M, cell_m=SIZES.cell,
                  cells_across_smallest_passage=PASSAGE_M / SIZES.cell, wall_cell_m=SIZES.wall_cell,
                  first_cell_height_m=SIZES.wall_cell / 2, y_plus=Unmeasured(fitness.NO_FLOW),
                  y_plus_verdict=Unmeasured(fitness.NO_FLOW), non_orth_max=42.32, non_orth_mean=7.26,
                  skew_max=1.14, aspect_max=Unmeasured(fitness.NO_ASPECT), schemes="standard",
                  measured_from="log.checkMesh + mesh_sizes2d cell 4.75e-4 m + passage.min 2.98 mm")
    fields.update(over)
    return FitnessTable(**fields)


# ----------------------------------------------------------------------------- the constructor's guard

def test_cells_none_is_refused():
    with pytest.raises(ValueError, match=r"FitnessTable\.cells is required; a fitness table is measured, not asserted"):
        table(cells=None)
    for name in ("hex_fraction", "cell_m", "wall_cell_m", "first_cell_height_m", "non_orth_max", "non_orth_mean",
                 "skew_max"):
        with pytest.raises(ValueError, match=rf"FitnessTable\.{name} is required"):
            table(**{name: None})


def test_nan_is_refused():
    with pytest.raises(ValueError, match=r"FitnessTable\.non_orth_max is required"):
        table(non_orth_max=math.nan)
    with pytest.raises(ValueError, match=r"FitnessTable\.y_plus is required"):
        table(y_plus=math.nan)
    with pytest.raises(ValueError, match="measured_from is required"):
        table(measured_from="")


def test_unmeasured_is_an_explicit_absence_and_prints_its_reason():
    """None in an optional field is refused: `Unmeasured(reason)` is the only way to say a
    number is absent, and it prints its reason; a measured passage with an Unmeasured
    cells-across (or the reverse) is refused too, since one implies the other."""
    for name in ("smallest_passage_m", "cells_across_smallest_passage", "y_plus", "y_plus_verdict", "aspect_max"):
        with pytest.raises(ValueError, match=rf"FitnessTable\.{name} is required.*Unmeasured\(reason\)"):
            table(**{name: None})
    with pytest.raises(ValueError, match="cells_across_smallest_passage is Unmeasured exactly when"):
        table(cells_across_smallest_passage=Unmeasured("lost"))
    with pytest.raises(ValueError, match="cells_across_smallest_passage is Unmeasured exactly when"):
        table(smallest_passage_m=Unmeasured("lost"))
    t = table()
    lines = t.lines()
    assert "y+ not estimated: no flow speed in a mesh-only study" in lines
    assert any(line.endswith("aspect ratio not in the log (checkMesh's log has no aspect ratio line)") for line in lines)
    assert str(Unmeasured("no flow speed in a mesh-only study")) == "not measured (no flow speed in a mesh-only study)"
    d = json.loads(json.dumps(t.as_dict()))
    assert d["y_plus"] == {"unmeasured": "no flow speed in a mesh-only study"}
    assert d["aspect_max"] == {"unmeasured": "checkMesh's log has no aspect ratio line"}
    back = FitnessTable.from_dict(d)
    assert back == t and isinstance(back.y_plus, Unmeasured)


# ----------------------------------------------------------------------------- from_digest

def test_from_digest_on_the_test_log():
    """The CHECKMESH_LOG of test_geometry_agent.py: 3750 cells, all hexahedra, max
    non-orthogonality 42.32, max skewness 1.14, so the standard schemes; the cell size is
    the writer's `Sizes`, the passage a measurement in metres."""
    t = from_digest(digest(), SIZES, PASSAGE_M, study="mesh", flow=None, scale=0.001)
    assert t.cells == 3750 and t.hex_fraction == 1.0
    assert t.non_orth_max == pytest.approx(42.32, abs=0.01) and t.non_orth_mean == pytest.approx(7.256, abs=0.01)
    assert t.skew_max == pytest.approx(1.14, abs=0.01)
    assert t.schemes == "standard" and t.quality_gate == "not applied"
    assert t.cell_m == SIZES.cell and t.wall_cell_m == SIZES.wall_cell and t.first_cell_height_m == SIZES.wall_cell / 2
    assert t.smallest_passage_m == PASSAGE_M
    assert t.cells_across_smallest_passage == pytest.approx(PASSAGE_M / SIZES.cell)
    assert isinstance(t.y_plus, Unmeasured) and t.y_plus.reason == "no flow speed in a mesh-only study"
    assert isinstance(t.aspect_max, Unmeasured)
    lines = t.lines()
    assert lines[0] == "cells 3750 (hex 100 %)"
    assert lines[1] == "across the smallest passage 6.3 cells (2.98 mm / 0.475 mm)"
    assert "y+ not estimated: no flow speed in a mesh-only study" in lines
    assert any(line.startswith("non-orthogonality max 42.32, mean 7.26") for line in lines)
    assert "schemes standard" in lines and "quality gate not applied" in lines
    assert t.measured_from == "log.checkMesh + mesh_sizes2d cell 0.000475 m + passage.min 2.98 mm"
    assert FitnessTable.from_dict(json.loads(json.dumps(t.as_dict()))) == t


def test_a_3d_digest_gives_an_unmeasured_passage_not_a_placeholder():
    """D34: the 3D branch has no 2D passage measure; the table says so in both fields with
    the same reason, and the line reads 'not measured (...)', never a number."""
    absent = Unmeasured("no 2D passage measure for a 3D body in Phase 1")
    t = from_digest(digest(), SIZES, absent, study="mesh", flow=None, scale=1.0)
    assert t.smallest_passage_m == absent
    assert isinstance(t.cells_across_smallest_passage, Unmeasured)
    assert t.cells_across_smallest_passage.reason == absent.reason
    assert t.lines()[1] == "across the smallest passage: not measured (no 2D passage measure for a 3D body in Phase 1)"
    assert "passage not measured (no 2D passage measure for a 3D body in Phase 1)" in t.measured_from
    assert t.as_dict()["smallest_passage_m"] == {"unmeasured": absent.reason}


def test_y_plus_is_estimated_when_the_study_has_a_flow():
    """With a flow the y+ comes from the case writer's own correlation
    (`case_gen.estimate_y_plus`) at the first cell's height, and the verdict names the
    layer the first cell sits in."""
    case_gen = _toolbox.load("case_gen")
    flow = case_gen.Flow(speed=1.0, length=0.06, nu=1e-6, reynolds=60000.0, derived="reynolds")
    t = from_digest(digest(), SIZES, PASSAGE_M, study="flow", flow=flow, scale=0.001)
    assert t.y_plus == pytest.approx(case_gen.estimate_y_plus(SIZES.wall_cell / 2, flow))
    assert t.y_plus_verdict in ("resolved", "wall function", "buffer layer")
    assert t.y_plus_verdict == fitness.y_plus_verdict(t.y_plus)
    assert any(line.startswith(f"y+ {t.y_plus:.3g} ({t.y_plus_verdict})") for line in t.lines())
    assert fitness.y_plus_verdict(1.0) == "resolved"
    assert fitness.y_plus_verdict(12.0) == "buffer layer"
    assert fitness.y_plus_verdict(45.0) == "wall function"


def test_a_log_without_the_numbers_is_refused():
    """A fitness table is measured, not asserted: a log with no cell count or no
    non-orthogonality line cannot become one."""
    with pytest.raises(ValueError, match="no cells count"):
        from_digest(digest(""), SIZES, PASSAGE_M, study="mesh", flow=None, scale=0.001)
    with pytest.raises(ValueError, match="non-orthogonality"):
        from_digest(digest("Mesh stats\n    cells:            3750\nMesh OK.\n"), SIZES, PASSAGE_M,
                    study="mesh", flow=None, scale=0.001)


# ----------------------------------------------------------------------------- schemes and the gate

def test_schemes_refused_above_70():
    assert schemes_for(42.32) == "standard"
    assert schemes_for(64.99) == "standard"
    assert schemes_for(65.0) == "limited"
    assert schemes_for(70.0) == "limited"
    assert schemes_for(70.01) == "refused"
    log = CHECKMESH_LOG.replace("Max: 42.3202", "Max: 72.5")
    t = from_digest(digest(log), SIZES, PASSAGE_M, study="mesh", flow=None, scale=0.001)
    assert t.non_orth_max == pytest.approx(72.5) and t.schemes == "refused"
    assert "schemes refused" in t.lines()
    with pytest.raises(ValueError, match="one of standard, limited, refused"):
        table(schemes="fast")


def test_quality_gate_is_not_applied_in_phase_1():
    """The gate is a Phase 2 decision; until then every table says so rather than
    claiming a pass it did not judge."""
    t = from_digest(digest(), SIZES, PASSAGE_M, study="mesh", flow=None, scale=0.001)
    assert t.quality_gate == "not applied"
    assert FitnessTable.__dataclass_fields__["quality_gate"].default == "not applied"
    assert t.lines()[-1] == "quality gate not applied"
