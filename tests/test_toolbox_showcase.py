"""showcase.py: the two things that go wrong silently.

Both bugs guarded here produced a picture that looked entirely reasonable and was
wrong, which is the only kind worth a test on a file whose output is judged by eye.
Neither can be exercised without pyvista -- showcase imports it at module level and
every path here ends in a render -- so they are guarded by reading the source, the way
`test_toolbox_animate.py` guards `render_frame`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
SOURCE = (TOOLBOX / "showcase.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"showcase.py has no function {name!r}")


def test_a_scalar_bar_title_is_never_a_shared_literal():
    """VTK keys scalar bars by title, so two bars with one title are one bar.

    A scene drawing two quantities then shows a single bar carrying the first one's
    range underneath both labels -- correct colours, confident numbers, wrong for one
    of them. The title must come from `_bar_key`, which hands out a different amount
    of whitespace per quantity."""
    coloured = function("coloured")
    titles = [kw.value for node in ast.walk(coloured)
              if isinstance(node, ast.Call)
              for kw in node.keywords if kw.arg == "title"]
    assert titles, "coloured() no longer sets a scalar bar title at all"
    for value in titles:
        assert not isinstance(value, ast.Constant), (
            "a literal scalar bar title is shared between every bar in the scene; "
            "use _bar_key(scalars)"
        )
        assert isinstance(value, ast.Call) and getattr(value.func, "id", "") == "_bar_key"


def test_bar_keys_are_not_hashed():
    """`hash` on a string is salted per process, so hashing the quantity name would
    make a bar's title depend on which process drew the frame. Frames of one sequence
    are rendered by whatever the harness starts; they must not disagree."""
    key = function("_bar_key")
    called = {getattr(node.func, "id", "") for node in ast.walk(key)
              if isinstance(node, ast.Call)}
    assert "hash" not in called, "_bar_key must not depend on a per-process hash seed"


def test_the_body_is_merged_rather_than_the_smallest_patch():
    """A body from CAD arrives as many patches: the OpenFOAM motorBike is 67 of them,
    and the smallest is a six-cell instrument dial. Taking the smallest drew the dial
    under its own pressure field and framed the camera on it -- silently, because a
    six-cell patch is a perfectly valid surface."""
    body = function("body_patch")
    calls = [getattr(node.func, "id", "") for node in ast.walk(body)
             if isinstance(node, ast.Call)]
    assert "_merge_patches" in calls, (
        "body_patch must merge its candidate patches; a single-patch case is a no-op "
        "and a sixty-seven-patch case is the whole point"
    )
    assert "min" not in calls, (
        "body_patch is choosing one patch by size again -- that is the dial bug"
    )


@pytest.mark.parametrize("name", ["scene_section", "_around", "_trimmed", "_merge_patches"])
def test_the_section_scene_and_its_helpers_exist(name):
    function(name)


def test_merged_patches_are_welded():
    """Each patch carries its own copy of the points along its border with the next, so
    an unwelded merge of 67 patches reports 146,099 open edges on a closed surface --
    75 once they are fused. Smooth shading has no shared point to average a normal
    across and shades each patch as an island, so the body renders faceted."""
    merge = function("_merge_patches")
    attrs = {node.func.attr for node in ast.walk(merge)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "clean" in attrs, "_merge_patches must weld the seams between its patches"


def test_section_is_offered_on_the_command_line():
    """A scene nothing can select is a scene nobody runs."""
    assert '"section"' in SOURCE
    choices = [node for node in ast.walk(TREE)
               if isinstance(node, ast.Call)
               for kw in node.keywords
               if kw.arg == "choices" and isinstance(kw.value, ast.List)
               and any(isinstance(e, ast.Constant) and e.value == "section"
                       for e in kw.value.elts)]
    assert choices, "--scene does not offer 'section'"


def test_the_cut_is_clipped_to_the_body():
    """An external-aero domain is tens of body lengths across. An unclipped cut plane
    fills the frame with freestream and stops being the flow around the object."""
    section = function("scene_section")
    attrs = {node.func.attr for node in ast.walk(section)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "clip_box" in attrs, "scene_section must clip its cut plane to the body"


# -- the flythrough: the camera path, and the frame that never got written -------------


@pytest.fixture(scope="module")
def showcase():
    """The module itself, for the parts of it that are arithmetic.

    `flight_path` is a list of camera stations and touches nothing that draws, but
    showcase imports pyvista at the top of the file, so loading it needs the library
    present -- not a renderer. Same gate as `test_toolbox_views.py:39-43`.
    """
    import importlib.util

    pytest.importorskip("pyvista", reason="showcase imports pyvista at module level",
                        exc_type=ImportError)
    spec = importlib.util.spec_from_file_location("toolbox_showcase",
                                                  TOOLBOX / "showcase.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_frame_is_staged_under_a_name_that_still_ends_in_png():
    """The bug that made `--orbit` and `--sweep` write nothing at all.

    Each frame goes to a dotted temporary and is renamed into place, so a frame that
    exists is a frame that is whole. The temporary was `.frame_0000.png.part`, and
    pyvista validates the extension it is handed:

        ValueError: Unsupported extension .part

    so every frame raised, the scene's own `except Exception` swallowed it, and the
    command printed one line and exited with an empty directory. Measured against
    `origin/main` on `studies/20260906-091113-f9cf/.../cylRe100`: `--scene slice
    --orbit 3` wrote 0 of 3 frames; with the writer fixed it writes 3 of 3.
    `animate.py:761` has always had the right form. Checked by reading the source
    because the failure is inside a render."""
    writer = function("_write_frame")
    joined = ast.dump(writer)
    assert "suffix" in joined, (
        "the staged name must end in the target's own suffix: pyvista refuses .part"
    )
    assert joined.index("'.part'") < joined.index("suffix"), (
        "the suffix has to come after the .part marker: a name ending in .part is the "
        "one pyvista will not write"
    )


def test_every_frame_loop_goes_through_the_one_writer():
    """Three loops wrote frames and all three had the same wrong temporary name. One
    of them fixed is one of them fixed; the point of a shared writer is that the next
    frame loop cannot reintroduce it."""
    for name in ("_orbit", "_sweep", "_flythrough"):
        body = ast.dump(function(name))
        assert "_write_frame" in body, f"{name} writes its own frames again"
        assert "screenshot" not in body, f"{name} screenshots directly again"


def test_the_flight_starts_and_ends_at_rest(showcase):
    """A constant-rate move begins and ends at full speed, which is the single thing
    that makes a rendered sequence look mechanical. The eased time makes the first and
    last steps the smallest ones in the path."""
    path = showcase.flight_path(60)
    steps = [abs(path[i + 1][0] - path[i][0]) for i in range(len(path) - 1)]
    middle = steps[len(steps) // 2]
    assert steps[0] < 0.25 * middle, "the camera starts at speed"
    assert steps[-1] < 0.25 * middle, "the camera stops dead"


def test_the_flight_never_doubles_back_or_jumps(showcase):
    """Azimuth is monotone -- a camera that reverses reads as a mistake, not a move --
    and no single step is more than a little above the average, which is what a seam
    between two keyframes would show up as."""
    path = showcase.flight_path(120)
    angles = [row[0] for row in path]
    steps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    assert all(step >= 0 for step in steps), "the camera doubles back"
    assert max(steps) < 2.2 * (sum(steps) / len(steps)), "there is a jump in the path"


def test_the_flight_opens_and_closes_wide_and_passes_close(showcase):
    """The shape of the shot, as numbers: wide at both ends, closest in the middle,
    and the descent and the climb are the same move run backwards."""
    path = showcase.flight_path(81)
    zooms = [row[2] for row in path]
    elevations = [row[1] for row in path]
    assert zooms[0] == pytest.approx(showcase.FLIGHT_WIDE)
    assert zooms[-1] == pytest.approx(showcase.FLIGHT_WIDE)
    assert zooms[len(zooms) // 2] == pytest.approx(showcase.FLIGHT_CLOSE)
    assert elevations[0] == pytest.approx(showcase.FLIGHT_HIGH)
    assert elevations[len(elevations) // 2] == pytest.approx(showcase.FLIGHT_LOW)
    assert elevations[-1] == pytest.approx(showcase.FLIGHT_HIGH)


def test_the_swing_is_not_a_whole_turn(showcase):
    """360 degrees ends the shot on the frame it opened with, which is what makes an
    orbit read as a turntable. The flythrough exists because that is not a move."""
    path = showcase.flight_path(40)
    assert 180.0 < path[-1][0] - path[0][0] < 355.0


def test_a_path_of_one_frame_is_a_still_rather_than_a_division_by_zero(showcase):
    assert showcase.flight_path(1) == [(235.0, showcase.FLIGHT_HIGH, showcase.FLIGHT_WIDE)]
    assert showcase.flight_path(0) == []


def test_the_flythrough_is_offered_on_the_command_line():
    """A scene nothing can select is a scene nobody runs -- and so is a motion."""
    assert '"--flythrough"' in SOURCE


def test_the_flythrough_promises_no_soundtrack():
    """The image carries no audio beyond the encoder imageio bundles, and a synthesised
    one would be invented content in a file whose whole point is that the picture
    carries the data. Said in the docstring rather than discovered by a user."""
    doc = ast.get_docstring(TREE) or ""
    assert "no audio" in doc.lower() or "no soundtrack" in doc.lower()
