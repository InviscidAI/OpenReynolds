"""The annotated picture (DESIGN.md 3.9): what is drawn is read back from the Figure
through `preview._figure` -- inset titles, marker positions in data coordinates, the
caption lines, the reference shift -- and the PNG is judged by its size, never by its
pixels (D17: a picture is regenerated, not diffed).

Until U2 lands the measurements come through mesh2d by path (`cli.legacy_analysis`), so
the junction insets are placed from the leg tables; once `measure.measure` records
junctions the same windows come from there, and every assertion here holds either way.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from openreynolds.geometry import cli, preview
from openreynolds.geometry.lint import Finding

from _geometry_fakes import T01_SPEC, T05_SPEC, plan_of, partial_plan, row_refusal, FIXTURES

pytest.importorskip("gmsh")
pytest.importorskip("matplotlib")


@pytest.fixture
def session():
    with cli._gmsh_session("preview-test") as gmsh:
        yield gmsh


def analysis(gmsh, spec: dict, claims=None):
    plan = plan_of(spec)
    return plan, cli.analyse(gmsh, plan, claims)


def _line_with_gid(fig, gid: str):
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_gid() == gid:
                return line
    return None


def test_draw_writes_a_png_over_10kb_for_t01(session, tmp_path):
    plan, a = analysis(session, T01_SPEC)
    out = preview.draw(session, a.face, plan, a.m, a.findings, a.table, tmp_path / "preview.png")
    assert out == tmp_path / "preview.png" and out.stat().st_size > 10_000
    fig, meta = preview._figure(session, a.face, plan, a.m, a.findings, a.table)
    try:
        # the caption is the LINT block: T01's one info line, the gap between neighbours
        assert any("I-GAP" in line and "2.4" in line for line in meta["caption"]), meta["caption"]
        # every junction of the four loops is a window: leave and landing per instance,
        # named `loops[k]` 0-based (D29), capped at eight
        assert meta["insets"][0].startswith("loops[0] leave (7.24, 1.5)")
        assert any(t.startswith("loops[0] landing (3.024, 1.5)") for t in meta["insets"])
        assert len([t for t in meta["insets"] if t != "..."]) == preview.MAX_INSETS
        assert meta["bounds"][0] == pytest.approx(0.0, abs=1e-3) and meta["bounds"][2] == pytest.approx(60.0, abs=1e-3)
        # the legs are annotated: a heading label in mesh2d's longest-axis sense, an arc's radius
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert "260 deg (-x)" in texts and "20 deg (+x)" in texts, texts
        assert any(t.startswith("r 4.5 (outer 6) 240 deg") for t in texts), texts
        assert any(t.startswith("pitch 14.66") for t in texts), texts
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_insets_are_capped_at_eight(session):
    """A 6-loop row has 12 junctions; the top strip shows eight and says '...'."""
    spec = copy.deepcopy(T01_SPEC)
    spec["ops"][0]["path"] = [{"line": 90}]
    spec["ops"][4]["count"] = 6
    spec["patches"][1]["at"] = "near:90,0"
    plan, a = analysis(session, spec)
    fig, meta = preview._figure(session, a.face, plan, a.m, a.findings, a.table)
    try:
        assert meta["ellipsis"] is True
        assert meta["insets"][-1] == "..." and len(meta["insets"]) == preview.MAX_INSETS + 1
        assert sum(1 for ax in fig.axes if ax.get_title().startswith("loops[")) == preview.MAX_INSETS
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_findings_are_drawn_where_they_are(session):
    plan, a = analysis(session, T01_SPEC)
    findings = a.findings + [
        Finding(level="error", code="E-TEST", subject="fluid", what="a cross at (20, 5)", where=(20.0, 5.0)),
        Finding(level="warn", code="W-TEST", subject="fluid", what="an orange cross", where=(40.0, 2.0),
                draw=[{"segment": [(38.0, 2.0), (42.0, 2.0)]}]),
    ]
    fig, meta = preview._figure(session, a.face, plan, a.m, findings, a.table)
    try:
        assert (20.0, 5.0, "error", "E-TEST") in meta["marks"] and (40.0, 2.0, "warn", "W-TEST") in meta["marks"]
        cross = _line_with_gid(fig, "finding:E-TEST")
        assert cross is not None and list(cross.get_xdata()) == [20.0] and list(cross.get_ydata()) == [5.0]
        assert cross.get_color() == preview.LEVEL_COLOURS["error"]
        warn = _line_with_gid(fig, "finding:W-TEST")
        assert warn is not None and warn.get_color() == preview.LEVEL_COLOURS["warn"]
        # the code sits in a label box beside the cross, and the caption carries the finding
        labels = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert "E-TEST" in labels and "W-TEST" in labels
        assert any("E-TEST" in line and "(20, 5)" in line for line in meta["caption"])
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_a_fixture_with_findings_draws_them_and_captions_them(session, tmp_path):
    """Attempt 3 (fixture 1): three overlapping pairs read through mesh2d's checks are
    red crosses inside the loops and lead the caption; the PNG is a real picture."""
    import json
    spec = json.loads((FIXTURES / "tesla_real_attempt3.json").read_text(encoding="utf-8"))
    plan, a = analysis(session, spec)
    overlaps = [f for f in a.findings if f.code == "E-OVERLAP"]
    assert len(overlaps) == 3 and all(f.where is not None for f in overlaps)
    out = preview.draw(session, a.face, plan, a.m, a.findings, a.table, tmp_path / "attempt3.png")
    assert out.stat().st_size > 10_000
    fig, meta = preview._figure(session, a.face, plan, a.m, a.findings, a.table)
    try:
        assert len([m for m in meta["marks"] if m[3] == "E-OVERLAP"]) == 3
        assert meta["caption"][0].startswith("!! ERROR  E-OVERLAP")
        assert all(m[1] > 1.5 for m in meta["marks"] if m[3] == "E-OVERLAP"), "inside the loops, above the wall"
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_a_row_refusal_draws_the_footprint_as_a_dashed_box(session, tmp_path):
    """The rc-5 partial path: the single instance with its solved footprint outlined."""
    from matplotlib.patches import Polygon
    plan = partial_plan()
    a = cli.analyse(session, plan, None)
    exc = row_refusal()
    findings = [Finding(level="error", code=exc.code, subject=exc.feature, what=exc.what, fix="\n".join(exc.lines))]
    fig, meta = preview._figure(session, a.face, plan, a.m, findings, None)
    try:
        boxes = [p for p in fig.axes[0].patches if isinstance(p, Polygon) and p.get_linestyle() == "--"]
        assert len(boxes) == 1
        xs = boxes[0].get_xy()[:, 0]
        assert min(xs) == pytest.approx(-12.46, abs=1e-6) and max(xs) == pytest.approx(7.28, abs=1e-6)
        assert any(t.get_text().startswith("footprint 19.74") for t in fig.axes[0].texts)
        assert meta["caption"][0].startswith("!! ERROR  E-ROW-FIT  Row 'loops'")
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    out = preview.draw(session, a.face, plan, a.m, findings, None, tmp_path / "partial.png")
    assert out.stat().st_size > 10_000


def test_reference_panel_translates_the_golden_so_inlets_coincide(session):
    """A golden of the same shape drawn 5 right and 2 up is shifted back onto the
    candidate by the inlet centres, and the Hausdorff distance is then zero."""
    from openreynolds.geometry.library import ReferenceMatch
    plan, a = analysis(session, T01_SPEC)
    span = max(a.m.extent)
    mine = preview.outline_polyline(session, a.face, span / 500)
    shifted = [[(x + 5.0, y + 2.0) for x, y in loop] for loop in mine]
    ref = ReferenceMatch(entry="tesla_valve", preset="t01", approved_at="2026-09-08", outline=shifted,
                         measurements={"extent": [60, 14.25], "patches": {"inlet": {"midpoints": [[5.0, 2.0]]}}},
                         hausdorff=None)
    fig, meta = preview._figure(session, a.face, plan, a.m, a.findings, a.table, ref)
    try:
        assert meta["reference_shift"] == pytest.approx((-5.0, -2.0), abs=1e-6)
        assert meta["hausdorff"] == pytest.approx(0.0, abs=1e-6)
        line = _line_with_gid(fig, "reference-outline")
        assert line is not None and line.get_color() == preview.GOLDEN
        xs, ys = list(line.get_xdata()), list(line.get_ydata())
        assert min(xs) == pytest.approx(min(p[0] for p in mine[0]), abs=1e-6)
        assert max(ys) == pytest.approx(max(p[1] for p in mine[0]), abs=1e-6)
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        assert "library: tesla_valve (t01) -- approved 2026-09-08" in titles
        assert any(line.startswith("REFERENCE  tesla_valve (t01): Hausdorff") for line in meta["caption"])
        # the panel sits to the right of the main axes at the same scale
        main, panel = fig.axes[0], next(ax for ax in fig.axes if ax.get_title(loc="left").startswith("library:"))
        assert panel.get_position().x0 > main.get_position().x1
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_outline_polyline_round_trips_the_extent(session):
    plan, a = analysis(session, T01_SPEC)
    loops = preview.outline_polyline(session, a.face, max(a.m.extent) / 500)
    assert len(loops) == 1 + a.m.islands
    outer = loops[0]
    xs, ys = [p[0] for p in outer], [p[1] for p in outer]
    assert max(xs) - min(xs) == pytest.approx(a.m.extent[0], abs=1e-6)
    assert max(ys) - min(ys) == pytest.approx(a.m.extent[1], abs=1e-3)   # an arc apex between two samples
    # the outer loop runs counter-clockwise, every hole clockwise, and the points are
    # spaced at the asked arc length so two goldens of one shape agree point for point
    assert preview._signed_area(outer) > 0 and all(preview._signed_area(h) < 0 for h in loops[1:])
    steps = [math.dist(outer[i], outer[i + 1]) for i in range(len(outer) - 1)]
    assert max(steps) <= max(a.m.extent) / 500 + 1e-9
    assert preview.hausdorff(loops, loops) == 0.0
    assert preview.hausdorff(loops, [[(x + 1.0, y) for x, y in loop] for loop in loops]) == pytest.approx(1.0, abs=1e-6)


def test_the_t05_picture_colours_the_cylinder_as_its_own_patch(session, tmp_path):
    plan, a = analysis(session, T05_SPEC)
    fig, meta = preview._figure(session, a.face, plan, a.m, a.findings, a.table)
    try:
        legend = fig.axes[0].get_legend()
        assert [t.get_text() for t in legend.get_texts()] == ["cylinder", "inlet", "outlet", "walls"]
        assert any(t.get_text() == "d 10" for t in fig.axes[0].texts)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    out = preview.draw(session, a.face, plan, a.m, a.findings, a.table, tmp_path / "t05.png")
    assert out.stat().st_size > 10_000


def test_draw_plain_is_mesh2d_preview_by_path(session, tmp_path):
    from openreynolds.geometry import _toolbox
    mesh2d = _toolbox.load("mesh2d")
    ops, rules = mesh2d.parse_spec(copy.deepcopy(T05_SPEC))
    face = mesh2d.build_face(session, ops, scale=1.0)
    built = mesh2d.measure2d(session, face)
    curve_patch = mesh2d.classify_curves(session, built, mesh2d.longest_axis2d(built.extent), rules)
    out = preview.draw_plain(session, built, curve_patch, tmp_path / "plain.png", "t05", ["a caption"], [])
    assert out == Path(tmp_path / "plain.png") and out.stat().st_size > 10_000
