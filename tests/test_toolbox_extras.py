"""What `pip install 'openreynolds[toolbox]'` promises, against what the toolbox imports.

`tests/test_toolbox.py` already checks the toolbox's imports against an allowlist of
what the *workspace image* provides. That direction is the one CI and the hosted path
exercise, and it is why #204 survived: inside the image every module is present, so four
scripts that no extra declared -- `cad_convert.py` (gmsh), `cad_export.py` (build123d,
OCP) and `encode.py` (imageio, declared under `[video]` only) -- imported cleanly in
every place anybody looked, and raised `ImportError` only on a user's own machine,
following the documented install.

So this file checks the other direction: every third-party module the toolbox imports is
declared by an extra, and the two CAD scripts are the only ones that need the large one.

It is a static check rather than a real import of each script, because the import it
would have to perform is the thing being tested: CI installs `[dev]`, not `[toolbox]`
and not `[cad]`, and installing `[cad]` to prove it imports would add several hundred MB
of OCCT to eight matrix jobs. The import names are read with `ast`, the declarations are
read out of `pyproject.toml`, and the mapping between the two is written down once
below -- `OCP` is the only one whose module name is not its distribution name.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip(
    "tomllib",
    reason="stdlib tomllib is 3.11+; the other six matrix jobs run this check",
)

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"

# Module name -> the distribution that provides it. Only the toolbox's third-party set;
# the standard library and the scripts' own siblings are what test_toolbox.py sorts out.
DISTRIBUTION = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "pyvista": "pyvista",
    "imageio": "imageio",
    "gmsh": "gmsh",
    "build123d": "build123d",
    # build123d's own OCCT binding. `import build123d` imports it, and `cad_export.py`
    # reaches through it for the triangulation already attached to a shape.
    "OCP": "cadquery-ocp-novtk",
}

CAD_SCRIPTS = {"cad_convert.py", "cad_export.py"}
"""The two that need `[cad]`. Named here so that a third one joining the toolbox has to
be a deliberate edit to this line rather than a silent widening of the big extra."""


def requirements() -> dict[str, list[str]]:
    """Extra name -> the requirement strings it declares, as written."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def extras() -> dict[str, set[str]]:
    """Extra name -> the distribution names it declares, lower-cased and unpinned."""
    # "imageio[ffmpeg]>=2.31" -> "imageio"; "cadquery-ocp-novtk==7.9.3.1.1" -> itself.
    return {
        name: {re.split(r"[\[<>=!;~ ]", requirement, 1)[0].strip().lower()
               for requirement in declared}
        for name, declared in requirements().items()
    }


def imports_of(script: Path) -> set[str]:
    """The third-party modules this script imports, by module name."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    return roots & set(DISTRIBUTION)


def test_every_toolbox_import_is_declared_by_some_extra():
    """The bug in #204, as one assertion.

    Four scripts imported modules that no extra named, so the documented install could
    not run them. Nothing is allowed to be left to a comment: a module the toolbox
    imports is either in `[toolbox]` or in `[cad]`.
    """
    declared = extras()
    promised = declared["toolbox"] | declared["cad"]
    for script in sorted(TOOLBOX.glob("*.py")):
        for module in sorted(imports_of(script)):
            distribution = DISTRIBUTION[module]
            assert distribution.lower() in promised, (
                f"{script.name} imports {module}, which neither [toolbox] nor [cad] "
                f"installs -- a documented install cannot run it"
            )


def test_only_the_two_cad_scripts_need_the_large_extra():
    """`[toolbox]` is the one a laptop installs for post-processing, so what belongs in
    it is everything that is not hundreds of MB of OCCT. Each script outside the CAD
    pair must run on `[toolbox]` alone."""
    toolbox_only = extras()["toolbox"]
    for script in sorted(TOOLBOX.glob("*.py")):
        if script.name in CAD_SCRIPTS:
            continue
        for module in sorted(imports_of(script)):
            assert DISTRIBUTION[module].lower() in toolbox_only, (
                f"{script.name} imports {module}, which only [cad] installs. Either it "
                f"belongs beside cad_convert.py and cad_export.py in CAD_SCRIPTS, or "
                f"{module} belongs in [toolbox]"
            )


def test_the_cad_scripts_are_the_reason_the_cad_extra_exists():
    """The other direction: `[cad]` carries nothing the toolbox does not import. An
    extra that installs OCCT for nobody is a download charged for nothing."""
    needed = {
        DISTRIBUTION[module].lower()
        for name in CAD_SCRIPTS
        for module in imports_of(TOOLBOX / name)
    }
    for distribution in extras()["cad"]:
        assert distribution in needed, (
            f"[cad] installs {distribution} and neither CAD script imports it"
        )


def test_encode_can_write_an_mp4_from_what_is_declared():
    """`imageio` without the ffmpeg plugin writes a GIF and raises on an mp4, and
    `encode.py` offers both. The image installs `imageio[ffmpeg]`, so the extras do."""
    declared = requirements()
    for extra in ("toolbox", "video"):
        assert any(requirement.startswith("imageio[ffmpeg]")
                   for requirement in declared[extra]), (
            f"[{extra}] declares imageio without the ffmpeg plugin"
        )


# -- the pins, which are the lesson of #132 ------------------------------------
#
# `build123d>=0.9` resolved to 0.9.x when it was written and to 0.11 later, and the OCP
# that came with it broke production for four days. The image pins all three exactly;
# the extra pins the same three to the same versions, so that a local post-processing
# run and a hosted one are the same software. Two numbers in two files agreeing by
# intention is how they come to disagree, hence this.

DOCKERFILE_ARG = {
    "gmsh": "GMSH_VERSION",
    "build123d": "BUILD123D_VERSION",
    "cadquery-ocp-novtk": "OCP_VERSION",
}


def test_the_cad_extra_pins_what_the_image_pins():
    pinned = dict(
        requirement.split("==", 1)  # type: ignore[misc]
        for requirement in requirements()["cad"] if "==" in requirement
    )
    assert set(pinned) == set(DOCKERFILE_ARG), (
        f"[cad] should pin exactly {sorted(DOCKERFILE_ARG)}, and pins {sorted(pinned)}"
    )

    dockerfile = (ROOT / "docker" / "workspace" / "Dockerfile").read_text(encoding="utf-8")
    for distribution, arg in DOCKERFILE_ARG.items():
        found = re.search(rf"^ARG\s+{arg}=(\S+)", dockerfile, re.M)
        assert found, (
            f"the Dockerfile no longer declares {arg}, so [cad]'s "
            f"{distribution}=={pinned[distribution]} pin tracks nothing. Point this "
            f"test at whatever replaced it, or drop the pin and say why."
        )
        assert pinned[distribution] == found.group(1), (
            f"[cad] pins {distribution}=={pinned[distribution]} and the image installs "
            f"{found.group(1)}; a local run and a hosted run would be different software"
        )
