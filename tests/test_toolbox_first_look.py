"""first_look.py -- the parts of it that do not need a graphics stack.

pyvista is not installed on the dev machine, so the script keeps its rendering behind
function-local imports and everything decided before a pixel is drawn -- which region
to frame, what the stats table says, how the sheet is laid out, what happens when a
panel raises -- is plain Python and is tested here.
"""

from __future__ import annotations

import ast
import gzip
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def first_look():
    return load("first_look")


def fake_png(path: Path, colour: str = "steelblue") -> Path:
    """A panel-shaped image, so the sheet composer gets something real to place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(2, 1.5), dpi=50)
    figure.add_subplot(111).set_facecolor(colour)
    figure.savefig(str(path))
    plt.close(figure)
    return path


# -- it must stay importable without a graphics stack ------------------------------


def test_pyvista_is_not_imported_at_module_scope():
    """The whole test file depends on this: a top-level `import pyvista` would make
    every check below unrunnable on a machine without OSMesa."""
    tree = ast.parse((TOOLBOX / "first_look.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "pyvista" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "pyvista"


# -- region of interest ------------------------------------------------------------

DOMAIN = (0.0, 2.0, 0.0, 1.0, 0.0, 0.1)

CYLINDER = {"name": "cylinder", "n_cells": 400, "area": 0.03,
            "bounds": (0.4, 0.6, 0.4, 0.6, 0.0, 0.1)}
INLET = {"name": "inlet", "n_cells": 200, "area": 0.1,
         "bounds": (0.0, 0.0, 0.0, 1.0, 0.0, 0.1)}
OUTLET = {"name": "outlet", "n_cells": 200, "area": 0.1,
          "bounds": (2.0, 2.0, 0.0, 1.0, 0.0, 0.1)}
FRONT_AND_BACK = {"name": "frontAndBack", "n_cells": 4000, "area": 4.0,
                  "bounds": (0.0, 2.0, 0.0, 1.0, 0.0, 0.1)}
LOWER_WALL = {"name": "lowerWall", "n_cells": 900, "area": 0.2,
              "bounds": (0.0, 2.0, 0.0, 0.3, 0.0, 0.1)}


def test_a_patch_flat_on_a_domain_face_is_part_of_the_box(first_look):
    assert first_look.classify_patch(INLET["bounds"], DOMAIN)["outer"] is True
    assert first_look.classify_patch(OUTLET["bounds"], DOMAIN)["outer"] is True


def test_a_patch_filling_the_box_in_every_direction_is_part_of_the_box(first_look):
    """`frontAndBack`, and any `walls` patch that wraps several sides at once: its
    bounding box is the domain's, which is what gives it away."""
    assert first_look.classify_patch(FRONT_AND_BACK["bounds"], DOMAIN)["outer"] is True


def test_an_obstacle_inside_the_flow_is_not_part_of_the_box(first_look):
    verdict = first_look.classify_patch(CYLINDER["bounds"], DOMAIN)
    assert verdict["outer"] is False
    assert verdict["touches"] == 0


def test_a_wall_running_along_the_floor_touches_the_box_once(first_look):
    verdict = first_look.classify_patch(LOWER_WALL["bounds"], DOMAIN)
    assert verdict["outer"] is False
    assert verdict["touches"] == 1


def test_the_obstacle_is_chosen_over_the_domain_patches(first_look):
    chosen, why = first_look.choose_region(
        [INLET, OUTLET, FRONT_AND_BACK, CYLINDER], DOMAIN
    )
    assert chosen["name"] == "cylinder"
    assert "cylinder" in why


def test_a_patch_clear_of_the_box_beats_a_smaller_one_that_touches_it(first_look):
    """Smallness alone would pick the lower wall here; being detached wins."""
    smaller_wall = dict(LOWER_WALL, area=0.001)
    chosen, _why = first_look.choose_region([smaller_wall, CYLINDER], DOMAIN)
    assert chosen["name"] == "cylinder"


def test_the_smaller_of_two_equal_candidates_wins(first_look):
    big = {"name": "sphere", "n_cells": 900, "area": 0.5,
           "bounds": (0.8, 1.2, 0.3, 0.7, 0.0, 0.1)}
    chosen, _why = first_look.choose_region([big, CYLINDER], DOMAIN)
    assert chosen["name"] == "cylinder"


def test_face_count_orders_the_patches_when_no_area_is_known(first_look):
    small = {"name": "small", "n_cells": 10, "area": None,
             "bounds": (0.4, 0.5, 0.4, 0.5, 0.0, 0.1)}
    large = {"name": "large", "n_cells": 900, "area": None,
             "bounds": (1.0, 1.5, 0.2, 0.8, 0.0, 0.1)}
    chosen, _why = first_look.choose_region([large, small], DOMAIN)
    assert chosen["name"] == "small"


def test_a_cavity_leaves_no_candidate_and_says_so(first_look):
    """Every patch of a lid-driven cavity is part of the box; the honest answer is
    none, and the caller falls through to the densest cells."""
    chosen, why = first_look.choose_region([INLET, OUTLET, FRONT_AND_BACK], DOMAIN)
    assert chosen is None
    assert "domain box" in why


def test_no_patches_at_all_is_not_an_exception(first_look):
    chosen, why = first_look.choose_region([], DOMAIN)
    assert chosen is None and why


def test_a_named_region_is_taken_as_given(first_look):
    chosen, why = first_look.choose_region([INLET, CYLINDER], DOMAIN, prefer="INLET")
    assert chosen["name"] == "inlet"
    assert "command line" in why


def test_a_name_that_is_not_a_patch_lists_the_ones_that_are(first_look):
    with pytest.raises(first_look.Missing) as raised:
        first_look.choose_region([INLET, CYLINDER], DOMAIN, prefer="wing")
    assert "inlet" in str(raised.value) and "cylinder" in str(raised.value)


def test_a_patch_with_no_bounds_is_skipped_rather_than_crashing(first_look):
    broken = {"name": "broken", "n_cells": 1, "area": 0.0, "bounds": None}
    chosen, _why = first_look.choose_region([broken, CYLINDER], DOMAIN)
    assert chosen["name"] == "cylinder"


# -- the densest-cells fallback ----------------------------------------------------


def test_the_dense_region_boxes_the_smallest_cells(first_look):
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [5.2, 5.1, 5.3]])
    volumes = np.array([1.0, 1.0, 1e-6, 2e-6])
    box = first_look.dense_region(centers, volumes, fraction=0.5, min_cells=2)
    assert box == pytest.approx((5.0, 5.2, 5.0, 5.1, 5.0, 5.3))


def test_the_dense_region_keeps_at_least_a_floor_of_cells(first_look):
    centers = np.arange(300, dtype=float).reshape(100, 3)
    volumes = np.linspace(1.0, 2.0, 100)
    box = first_look.dense_region(centers, volumes, fraction=0.0, min_cells=10)
    # ten smallest volumes are the first ten rows, whose x runs 0, 3, ... 27.
    assert box[0] == pytest.approx(0.0)
    assert box[1] == pytest.approx(27.0)


def test_an_empty_mesh_has_no_dense_region(first_look):
    assert first_look.dense_region(np.zeros((0, 3)), np.zeros(0)) is None


def test_non_finite_volumes_do_not_poison_the_choice(first_look):
    centers = np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]])
    volumes = np.array([np.nan, 3.0])
    box = first_look.dense_region(centers, volumes, fraction=1.0, min_cells=1)
    assert box == pytest.approx((9.0, 9.0, 9.0, 9.0, 9.0, 9.0))


# -- framing -----------------------------------------------------------------------


def test_the_close_up_box_is_padded_around_its_own_centre(first_look):
    box = first_look.focus_bounds((0.9, 1.1, 0.4, 0.6, 0.0, 0.1), (0.0, 4.0, 0.0, 2.0, 0.0, 0.1),
                                  pad=2.0)
    assert box[0] == pytest.approx(0.8)
    assert box[1] == pytest.approx(1.2)


def test_the_close_up_box_is_clipped_to_the_domain(first_look):
    box = first_look.focus_bounds((0.0, 0.4, 0.0, 0.4, 0.0, 0.1), DOMAIN, pad=10.0)
    assert box[0] == pytest.approx(DOMAIN[0])
    assert box[3] <= DOMAIN[3] + 1e-12


def test_a_flat_region_is_given_a_thickness(first_look):
    """A planar patch has no extent normal to itself, and scaling zero by 2.5 leaves
    a box with no inside for the clip to keep."""
    box = first_look.focus_bounds((1.0, 1.0, 0.4, 0.6, 0.0, 0.1), (0.0, 4.0, 0.0, 2.0, 0.0, 0.1))
    assert box[1] > box[0]


def test_a_two_dimensional_case_is_viewed_down_its_thin_axis(first_look):
    assert first_look.camera_for((0.0, 2.0, 0.0, 1.0, 0.0, 0.01)) == "xy"
    assert first_look.camera_for((0.0, 0.01, 0.0, 1.0, 0.0, 2.0)) == "yz"


def test_a_three_dimensional_case_is_viewed_isometrically(first_look):
    assert first_look.camera_for((0.0, 2.0, 0.0, 1.0, 0.0, 1.5)) == "iso"
    assert first_look.camera_for((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == "iso"


def test_patch_colours_are_stable_and_cycle(first_look):
    names = [f"p{index}" for index in range(len(first_look.COLOURS) + 2)]
    colours = first_look.patch_colours(names)
    assert colours["p0"] == first_look.COLOURS[0]
    assert colours[names[-2]] == first_look.COLOURS[0]
    assert first_look.patch_colours(names) == colours


# -- reading the case off disk -----------------------------------------------------

OWNER_HEAD = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    note        "nPoints:12057 nCells:9000 nFaces:36200 nInternalFaces:17400";
    location    "constant/polyMesh";
    object      owner;
}
"""

BOUNDARY_FILE = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    location    "constant/polyMesh";
    object      boundary;
}

4
(
    inlet
    {
        type            patch;
        nFaces          30;
        startFace       17400;
    }
    outlet
    {
        type            patch;
        nFaces          57;
        startFace       17430;
    }
    cylinder
    {
        type            wall;
        inGroups        1(wall);
        nFaces          400;
        startFace       17487;
    }
    frontAndBack
    {
        type            empty;
        nFaces          18000;
        startFace       17887;
    }
)
"""


def test_the_owner_header_gives_the_counts(first_look):
    counts = first_look.parse_owner_note(OWNER_HEAD)
    assert counts["nCells"] == 9000
    assert counts["nPoints"] == 12057
    assert counts["nInternalFaces"] == 17400


def test_an_owner_with_no_note_yields_no_counts(first_look):
    assert first_look.parse_owner_note("FoamFile\n{\n    object owner;\n}\n") == {}


def test_the_boundary_file_gives_names_types_and_face_counts(first_look):
    patches = first_look.parse_boundary(BOUNDARY_FILE)
    assert [patch["name"] for patch in patches] == ["inlet", "outlet", "cylinder", "frontAndBack"]
    assert patches[2]["type"] == "wall"
    assert patches[3]["nFaces"] == 18000


def test_the_foamfile_header_is_not_read_as_a_patch(first_look):
    """Regression bait: the header is a bare word followed by a brace block, which
    is exactly the shape a patch entry has."""
    names = [patch["name"] for patch in first_look.parse_boundary(BOUNDARY_FILE)]
    assert "FoamFile" not in names


def test_junk_is_not_a_boundary_file(first_look):
    assert first_look.parse_boundary("this is not a mesh") == []


def write_case(root: Path, *, compressed: bool = False, processor: bool = False) -> Path:
    case = root / "cavity"
    relative = "processor0/constant/polyMesh" if processor else "constant/polyMesh"
    mesh = case / relative
    mesh.mkdir(parents=True)
    if compressed:
        for name, text in (("owner.gz", OWNER_HEAD), ("boundary.gz", BOUNDARY_FILE)):
            with gzip.open(str(mesh / name), "wt", encoding="utf-8") as handle:
                handle.write(text)
    else:
        (mesh / "owner").write_text(OWNER_HEAD, encoding="utf-8")
        (mesh / "boundary").write_text(BOUNDARY_FILE, encoding="utf-8")
    return case


def test_disk_stats_reads_a_case_without_opening_the_mesh(tmp_path, first_look):
    case = write_case(tmp_path)
    stats = first_look.disk_stats(case)
    assert stats["case"] == "cavity"
    assert stats["counts"]["nCells"] == 9000
    assert len(stats["patches"]) == 4
    assert stats["mesh_dir"] == "constant/polyMesh"


def test_disk_stats_reads_a_compressed_poly_mesh(tmp_path, first_look):
    """writeCompression is on by default in plenty of tutorials, and a stats panel
    that goes blank because of it would be reporting a mesh that is right there."""
    case = write_case(tmp_path, compressed=True)
    assert first_look.disk_stats(case)["counts"]["nCells"] == 9000


def test_disk_stats_falls_back_to_a_processor_directory_and_says_so(tmp_path, first_look):
    case = write_case(tmp_path, processor=True)
    stats = first_look.disk_stats(case)
    assert "processor0" in stats["mesh_dir"]
    assert any("decomposed" in note for note in stats["notes"])


def test_an_unmeshed_case_reports_that_rather_than_raising(tmp_path, first_look):
    case = tmp_path / "fresh"
    (case / "system").mkdir(parents=True)
    stats = first_look.disk_stats(case)
    assert stats["counts"] == {}
    assert any("not been meshed" in note for note in stats["notes"])


def test_surfaces_are_found_in_the_usual_places(tmp_path, first_look):
    case = tmp_path / "case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "constant" / "triSurface" / "wing.stl").write_text("solid wing\nendsolid wing\n")
    (case / "constant" / "triSurface" / "notes.txt").write_text("ignore me")
    found = first_look.find_surfaces(case)
    assert [path.name for path in found] == ["wing.stl"]


def test_a_case_with_no_surfaces_returns_an_empty_list(tmp_path, first_look):
    case = tmp_path / "blockmesh-only"
    (case / "system").mkdir(parents=True)
    assert first_look.find_surfaces(case) == []


# -- the stats text ----------------------------------------------------------------


def full_stats(first_look, tmp_path):
    stats = first_look.disk_stats(write_case(tmp_path))
    stats["bounds"] = (0.0, 2.0, 0.0, 1.0, 0.0, 0.01)
    stats["cell_volume"] = (1.1e-9, 3.4e-7)
    return stats


def test_the_stats_text_carries_the_counts_the_extents_and_the_patches(tmp_path, first_look):
    text = first_look.format_stats(full_stats(first_look, tmp_path))
    assert "9,000" in text
    assert "cylinder" in text and "wall" in text
    assert "cell volume" in text
    assert "metres" in text


def test_the_stats_text_stays_narrow_enough_to_read_in_a_panel(tmp_path, first_look):
    text = first_look.format_stats(full_stats(first_look, tmp_path))
    for line in text.splitlines():
        assert len(line) <= first_look.STATS_WIDTH + 12


def test_the_stats_text_offers_no_verdict(tmp_path, first_look):
    """Whether nine thousand cells is right depends on what is about to run on it."""
    text = first_look.format_stats(full_stats(first_look, tmp_path)).lower()
    for verdict in ("too coarse", "too fine", "acceptable", "should refine", "bad mesh"):
        assert verdict not in text


def test_the_stats_text_still_names_the_case_when_nothing_is_known(first_look):
    text = first_look.format_stats({"case": "empty", "counts": {}, "patches": [], "notes": []})
    assert "empty" in text


def test_a_long_patch_list_is_truncated_with_a_count(first_look):
    patches = [{"name": f"patch{index}", "type": "wall", "nFaces": index} for index in range(20)]
    text = first_look.format_stats({"case": "many", "counts": {}, "patches": patches, "notes": []})
    assert "patch0" in text
    assert "and 8 more" in text


# -- sheet layout ------------------------------------------------------------------


@pytest.mark.parametrize(
    "count, shape",
    [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (5, (2, 3)), (6, (2, 3)),
     (7, (3, 3)), (9, (3, 3)), (10, (4, 3))],
)
def test_the_grid_is_as_square_as_three_columns_allow(first_look, count, shape):
    assert first_look.grid_shape(count) == shape


def test_no_panels_is_an_empty_grid(first_look):
    assert first_look.grid_shape(0) == (0, 0)


def make_panels(first_look, tmp_path):
    return [
        first_look.Panel("geometry", "geometry-preview", "geometry",
                         image=fake_png(tmp_path / "geometry.png"), ok=True),
        first_look.Panel("mesh", "mesh-full", "the whole mesh",
                         image=fake_png(tmp_path / "mesh.png", "seagreen"), ok=True),
        first_look.Panel("closeup", "mesh-closeup", "close-up",
                         image=fake_png(tmp_path / "closeup.png", "indianred"), ok=True),
        first_look.Panel("patches", "mesh-patches", "patches",
                         image=fake_png(tmp_path / "patches.png", "goldenrod"), ok=True),
        first_look.Panel("stats", "other", "counts", text="cells 9,000", ok=True),
    ]


def test_the_contact_sheet_is_one_png_holding_every_panel(tmp_path, first_look):
    out = tmp_path / "first_look.png"
    written = first_look.compose_sheet(make_panels(first_look, tmp_path), out, title="first look")
    assert written == out
    assert out.stat().st_size > 5000


def test_a_lost_panel_does_not_lose_the_sheet(tmp_path, first_look):
    """The point of the degrade rule: four panels rendered and one broken still gets
    composed, and the broken square says why instead of being blank."""
    panels = make_panels(first_look, tmp_path)
    panels[0] = first_look.Panel("geometry", "geometry-preview", "geometry",
                                 note="no surface file under constant/triSurface")
    out = tmp_path / "sheet.png"
    first_look.compose_sheet(panels, out)
    assert out.exists() and out.stat().st_size > 5000


def test_a_panel_whose_image_vanished_is_still_composed(tmp_path, first_look):
    panels = make_panels(first_look, tmp_path)
    panels[1].image.unlink()
    out = tmp_path / "sheet.png"
    first_look.compose_sheet(panels, out)
    assert out.exists()


def test_composing_nothing_is_a_missing_not_a_crash(tmp_path, first_look):
    with pytest.raises(first_look.Missing):
        first_look.compose_sheet([], tmp_path / "sheet.png")


def test_the_caption_carries_the_reason_a_panel_is_empty(first_look):
    panel = first_look.Panel("geometry", "geometry-preview", "geometry",
                             note="no surface file under constant/triSurface")
    caption = first_look.panel_caption(panel)
    assert caption.startswith("geometry")
    assert "no surface file" in caption


def test_a_caption_with_nothing_to_explain_is_just_the_title(first_look):
    panel = first_look.Panel("mesh", "mesh-full", "the whole mesh", ok=True)
    assert first_look.panel_caption(panel) == "the whole mesh"


# -- degrading -------------------------------------------------------------------


def test_attempt_returns_the_value_when_nothing_goes_wrong(first_look):
    value, note = first_look.attempt(lambda: "fine")
    assert value == "fine" and note == ""


def test_attempt_turns_an_exception_into_a_note(first_look):
    value, note = first_look.attempt(lambda: 1 / 0)
    assert value is None
    assert note.startswith("ZeroDivisionError")


def test_a_missing_input_reads_as_a_sentence_not_a_traceback(first_look):
    def absent():
        raise first_look.Missing("no surface file under constant/triSurface")

    _value, note = first_look.attempt(absent)
    assert note == "no surface file under constant/triSurface"


def test_one_failing_builder_costs_exactly_one_panel(tmp_path, first_look):
    def boom():
        raise RuntimeError("OSMesa said no")

    specs = [
        ("geometry", "geometry-preview", "geometry", lambda: fake_png(tmp_path / "a.png")),
        ("mesh", "mesh-full", "mesh", boom),
        ("closeup", "mesh-closeup", "close-up", lambda: fake_png(tmp_path / "b.png")),
        ("patches", "mesh-patches", "patches", lambda: fake_png(tmp_path / "c.png")),
        ("stats", "other", "counts", lambda: "cells 9,000"),
    ]
    panels = first_look.build_panels(specs)

    assert len(panels) == 5
    assert [panel.ok for panel in panels] == [True, False, True, True, True]
    assert "OSMesa said no" in panels[1].note
    assert panels[4].text == "cells 9,000"
    assert panels[0].image is not None


def test_a_builder_that_returns_nothing_is_recorded_as_such(first_look):
    panels = first_look.build_panels([("mesh", "mesh-full", "mesh", lambda: None)])
    assert panels[0].ok is False
    assert panels[0].note


def test_every_panel_name_has_a_manifest_kind(first_look):
    """The manifest rows are how the next session finds these pictures again, so a
    panel with no kind would be a panel that is written and then lost."""
    import sys as _sys

    _sys.path.insert(0, str(TOOLBOX))
    study_state = load("study_state")
    for name in first_look.PANEL_ORDER:
        assert first_look.PANEL_KINDS[name] in study_state.KINDS


# -- the one call, end to end ------------------------------------------------------
#
# Everything above tests a piece. What the spec actually asks for is one call that
# writes a sheet and leaves the manifest and the phase table saying so, and none of
# that is exercised by testing the pieces: a `first_look` that never called
# `study_state.record` would pass every check above.


def study_home(tmp_path: Path) -> Path:
    """A study root, so `find_root` settles on tmp_path and not on whatever
    ancestor of the system temp directory happens to have state in it."""
    (tmp_path / ".reynolds").mkdir()
    return tmp_path


def manifest_rows(first_look, root: Path) -> list[dict]:
    import json

    state = load("study_state")
    path = root / state.STATE_DIR / state.MANIFEST
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_meshed_case_with_no_renderer_still_gets_a_sheet(tmp_path, first_look):
    """pyvista is absent here, which is the harshest degrade there is: four panels
    lost at once. The sheet is still written, the stats panel still has the counts,
    and the phase says how much of it worked."""
    root = study_home(tmp_path)
    case = write_case(root)
    result = first_look.first_look(case)

    assert result["sheet"] is not None and result["sheet"].exists()
    named = {panel.name: panel for panel in result["panels"]}
    assert named["stats"].ok is True
    assert "9,000" in named["stats"].text
    for name in ("mesh", "closeup", "patches"):
        assert named[name].ok is False
        assert named[name].note, f"{name} lost its panel without saying why"

    state = load("study_state")
    preview = [row for row in state.load_phases(root)["phases"] if row["name"] == "preview"]
    assert preview and preview[0]["status"] == "done"
    assert "panels" in preview[0]["note"]
    kinds = [row["kind"] for row in manifest_rows(first_look, root)]
    assert "contact-sheet" in kinds


def test_a_case_directory_that_is_not_there_is_said_so_before_anything_is_written(
    tmp_path, first_look
):
    """A mistyped path used to make the directories, write a sheet and exit 0."""
    root = study_home(tmp_path)
    with pytest.raises(first_look.Missing):
        first_look.first_look(root / "typo")
    assert not (root / "typo").exists()
    assert first_look.main([str(root / "typo")]) == 1


def test_asking_for_one_panel_writes_only_that_panel(tmp_path, first_look):
    root = study_home(tmp_path)
    case = write_case(root)
    result = first_look.first_look(case, wanted=["geometry"])
    assert [panel.name for panel in result["panels"]] == ["geometry"]
    assert result["stats"] is None
    assert not (result["out"] / "stats.txt").exists()


class FakeMesh:
    def __init__(self, bounds, n_cells=100, area=1.0):
        self.bounds = bounds
        self.n_cells = n_cells
        self.n_points = n_cells * 2
        self.area = area


def stub_renderers(first_look, monkeypatch):
    """Stand in for pyvista so the registration path can be walked at all: without
    this no image panel ever succeeds on this machine and the four `record` calls
    the spec asks for are never reached."""
    scene = first_look.Scene(
        internal=FakeMesh(DOMAIN, n_cells=9000),
        patches={"inlet": FakeMesh(INLET["bounds"], 200, 0.1),
                 "cylinder": FakeMesh(CYLINDER["bounds"], 400, 0.03)},
    )
    monkeypatch.setattr(first_look, "open_case", lambda case: scene)
    monkeypatch.setattr(first_look, "render_geometry",
                        lambda case, out: fake_png(out))
    monkeypatch.setattr(first_look, "render_mesh_full",
                        lambda scene, out: fake_png(out))
    monkeypatch.setattr(first_look, "render_closeup",
                        lambda scene, region, out, label="": fake_png(out))
    monkeypatch.setattr(first_look, "render_patches",
                        lambda scene, out: fake_png(out))
    return scene


def test_every_rendered_panel_lands_in_the_manifest_under_its_own_kind(
    tmp_path, monkeypatch, first_look
):
    root = study_home(tmp_path)
    case = write_case(root)
    stub_renderers(first_look, monkeypatch)

    result = first_look.first_look(case, label="run one")
    assert all(panel.ok for panel in result["panels"]), \
        [(p.name, p.note) for p in result["panels"] if not p.ok]

    kinds = [row["kind"] for row in manifest_rows(first_look, root)]
    for expected in ("geometry-preview", "mesh-full", "mesh-closeup",
                     "mesh-patches", "contact-sheet"):
        assert expected in kinds, f"{expected} was rendered and never registered"

    state = load("study_state")
    preview = [row for row in state.load_phases(root)["phases"] if row["name"] == "preview"][0]
    assert preview["status"] == "done"
    assert preview["note"] == "5/5 panels"


def test_the_close_up_caption_says_the_region_was_a_guess(tmp_path, monkeypatch, first_look):
    """The whole point of printing the reason: the reader can see it picked the
    cylinder, and see that picking it was a heuristic."""
    root = study_home(tmp_path)
    case = write_case(root)
    stub_renderers(first_look, monkeypatch)
    result = first_look.first_look(case)
    closeup = [panel for panel in result["panels"] if panel.name == "closeup"][0]
    assert "cylinder" in closeup.note
    assert "cylinder" in first_look.panel_caption(closeup)


def test_one_renderer_failing_still_registers_the_other_four(
    tmp_path, monkeypatch, first_look
):
    root = study_home(tmp_path)
    case = write_case(root)
    stub_renderers(first_look, monkeypatch)

    def broken(case, out):
        raise RuntimeError("OSMesa said no")

    monkeypatch.setattr(first_look, "render_geometry", broken)
    result = first_look.first_look(case)

    named = {panel.name: panel for panel in result["panels"]}
    assert named["geometry"].ok is False
    assert "OSMesa said no" in named["geometry"].note
    assert result["sheet"] is not None and result["sheet"].exists()

    kinds = [row["kind"] for row in manifest_rows(first_look, root)]
    assert "geometry-preview" not in kinds
    for expected in ("mesh-full", "mesh-closeup", "mesh-patches", "contact-sheet"):
        assert expected in kinds


def test_the_command_line_reports_the_sheet_and_exits_zero(
    tmp_path, monkeypatch, first_look, capsys
):
    root = study_home(tmp_path)
    case = write_case(root)
    stub_renderers(first_look, monkeypatch)
    assert first_look.main([str(case)]) == 0
    assert "contact sheet (5/5 panels)" in capsys.readouterr().out


def test_no_sheet_at_all_is_a_failed_phase(tmp_path, first_look):
    """Nothing composed is the one case the phase table must not call done, so a
    resumed study does not skip a preview that never happened."""
    root = study_home(tmp_path)
    case = write_case(root)
    result = first_look.first_look(case, wanted=[])
    assert result["sheet"] is None and result["sheet_note"]
    state = load("study_state")
    preview = [row for row in state.load_phases(root)["phases"] if row["name"] == "preview"][0]
    assert preview["status"] == "failed"


def test_a_sheet_of_one_empty_panel_is_still_a_sheet(tmp_path, first_look, capsys):
    """`--panels geometry` on a blockMesh case renders nothing, and that sheet --
    one bordered square reading 'no surface file' -- is the answer, not a failure.
    Exit 0 tracks whether the sheet was written, not whether it was interesting."""
    root = study_home(tmp_path)
    case = write_case(root)
    assert first_look.main([str(case), "--panels", "geometry"]) == 0
    out = capsys.readouterr().out
    assert "no surface file" in out
    assert "contact sheet (0/1 panels)" in out


# -- the patches panel on a 2D case -------------------------------------------------


class _FakePatch:
    def __init__(self, n_cells):
        self.n_cells = n_cells


def _scene_with(first_look, patches):
    scene = first_look.Scene()
    scene.patches = {name: _FakePatch(n) for name, n in patches.items()}
    return scene


def test_the_flat_faces_of_a_2d_case_are_recognised(first_look):
    """Found by looking at a live contact sheet: 8,476 `frontAndBack` faces against
    384 for everything else, drawn from an iso camera, covered all six patches. The
    panel was a coloured slab with a correct legend beside it."""
    scene = _scene_with(first_look, {
        "inlet": 54, "outlet": 54, "body": 128, "bottomWall": 69, "topWall": 69,
        "frontAndBack": 8476,
    })

    assert first_look.is_empty_patch("frontAndBack", scene)
    for other in ("inlet", "outlet", "body", "topWall", "bottomWall"):
        assert not first_look.is_empty_patch(other, scene)


def test_a_patch_named_like_an_empty_one_but_tiny_is_not_treated_as_one(first_look):
    """Name and size together: `defaultFaces` carrying four faces is a leftover, not
    the face you are looking through."""
    scene = _scene_with(first_look, {"walls": 9000, "defaultFaces": 4})
    assert not first_look.is_empty_patch("defaultFaces", scene)


def test_a_3d_case_hides_nothing(first_look):
    """No patch dominates, so the iso view shows the case and every patch is drawn."""
    scene = _scene_with(first_look, {"inlet": 200, "outlet": 200, "car": 900, "ground": 400})
    assert not any(first_look.is_empty_patch(name, scene) for name in scene.patches)
