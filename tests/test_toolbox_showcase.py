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
