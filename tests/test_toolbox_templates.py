"""The templates: working geo -> msh -> gmshToFoam -> checkMesh scripts, not snippets.

They need gmsh, gmshToFoam and checkMesh to actually run, none of which this test
environment is promised to have (the toolbox's own convention -- see
`test_render_is_valid_python` in `test_toolbox.py` -- is a parse check plus the same
import-allowlist and docstring rules the rest of the toolbox follows, not an import).
What can be checked without the image is checked here: that each file parses, is
named in the index, imports nothing the instance does not have, and states the
gotchas the brief asked for in comments rather than leaving them to be rediscovered.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
TEMPLATES = TOOLBOX / "templates"

TEMPLATE_FILES = sorted(TEMPLATES.glob("*.py"))


def test_there_is_a_2d_and_a_3d_template():
    names = {p.name for p in TEMPLATE_FILES}
    assert "duct2d.py" in names
    assert "body_in_box.py" in names


def test_every_template_parses():
    for script in TEMPLATE_FILES:
        ast.parse(script.read_text(encoding="utf-8"))


def test_every_template_has_a_usage_docstring():
    for script in TEMPLATE_FILES:
        doc = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))
        assert doc, f"{script.name} has no docstring"
        assert "python3" in doc, f"{script.name} does not show how to run it"


def test_every_template_edits_two_numbers_and_says_so():
    """"Copy it in, edit two numbers, run it" is the whole point of a template."""
    for script in TEMPLATE_FILES:
        text = script.read_text(encoding="utf-8")
        assert "edit these two" in text.lower(), f"{script.name} does not mark its two numbers"


def test_templates_stick_to_what_the_image_provides():
    """Same rule as the rest of the toolbox (see test_toolbox.py): the standard
    library and gmsh are on the image; nothing else may be imported at module level."""
    allowed = set(sys.stdlib_module_names) | {"gmsh", "__future__"}
    for script in TEMPLATE_FILES:
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in allowed, f"{script.name} imports {root}, absent from the image"


def test_templates_are_named_in_the_toolbox_index():
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    for script in TEMPLATE_FILES:
        assert f"templates/{script.name}" in index, (
            f"{script.name} is in templates/ and not named in README.md"
        )


def test_environment_manifest_names_the_templates():
    env = (TOOLBOX / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "templates/" in env
    for script in TEMPLATE_FILES:
        assert script.name in env, f"{script.name} is not named in ENVIRONMENT.md"


def test_duct2d_bakes_in_the_2d_gotchas():
    text = (TEMPLATES / "duct2d.py").read_text(encoding="utf-8")
    for gotcha in (
        "Mesh.Algorithm", "8",                       # Frontal-Delaunay for quads
        "RecombinationAlgorithm", "3",                # Blossom, full-quad
        "setRecombine",
        "numElements=[1]", "recombine=True",           # one layer, swept to hex
        "MshFileVersion", "2.2",                        # gmshToFoam reads only this
        "foamDictionary",
        "empty",
        "metres",
        "controlDict",                                  # Foam::Time, even mesh-only
        "fvSchemes",                                     # checkMesh builds an fvMesh
        "entry0",                                        # foamDictionary's list wrapper
    ):
        assert gotcha in text, f"duct2d.py does not mention {gotcha!r}"
    # the physical group per edge is named where the line is created, not worked out
    # afterwards from where a face sits
    assert "addPhysicalGroup" in text
    assert "bounding box" in text.lower()


def test_body_in_box_bakes_in_the_3d_gotchas():
    text = (TEMPLATES / "body_in_box.py").read_text(encoding="utf-8")
    for gotcha in (
        "occ.cut",
        "getEntitiesInBoundingBox",
        "MshFileVersion", "2.2",
        "foamDictionary",
        "wall",
        "metres",
        "UPSTREAM_R", "DOWNSTREAM_R", "SIDE_R",
        "controlDict",
        "fvSchemes",
        "entry0",
    ):
        assert gotcha in text, f"body_in_box.py does not mention {gotcha!r}"


def test_templates_end_with_a_mesh_look_call():
    for script in TEMPLATE_FILES:
        text = script.read_text(encoding="utf-8")
        assert "mesh_look.py" in text, f"{script.name} does not call mesh_look.py"
        assert "checkMesh" in text, f"{script.name} does not run checkMesh"


def test_templates_fall_back_to_their_own_toolbox_when_not_deployed():
    """`/work/.toolbox` is where the instance keeps mesh_look.py; a template read or
    run outside that image (this test run included) has to find its sibling instead."""
    for script in TEMPLATE_FILES:
        text = script.read_text(encoding="utf-8")
        assert "/work/.toolbox" in text
        assert "parent.parent" in text
