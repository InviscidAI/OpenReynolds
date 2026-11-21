"""The wall-shear map, graded against the one wall shear that has an exact answer.

`showcase.py --scene surface --surface-field wallShearStress` draws the skin friction
on the geometry. A picture of a wall map is judged by eye, and a wall map drawn from
the wrong component of a vector, or from the cell values rather than the point values,
or scaled by a factor somebody introduced in a colour range, looks exactly as
convincing as a right one. So the map is graded here against a number that is not
anybody's measurement.

Developed plane Poiseuille flow between two walls a distance H apart:

    u(y)  = 6 u_mean (y/H) (1 - y/H)
    tau_wall / rho = nu |du/dy|_wall = 6 nu u_mean / H

which is exact, not a correlation and not a published value -- it falls out of
differentiating the profile, and `tests/test_journey_duct.py:19-35` derives the same
identity for the same reason, citing White, *Viscous Fluid Flow*, 3rd ed., for the
profile itself. The case here is hand-written at those numbers, so the value the
picture is drawn from is known before the reader is asked for it.

What this does NOT claim is that OpenFOAM would produce that field. It asserts that
the field a solver writes is the field the map reads: nothing between
`<time>/wallShearStress` on disk and the array the renderer colours by alters the
number. `tests/test_journey_duct.py` is where the solved value meets the exact one, in
CI, with a real `simpleFoam`.

    python3 -m pytest tests/test_toolbox_wall_shear_map.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[0] / "openreynolds" / "toolbox"

sys.path.insert(0, str(HERE))
from test_toolbox_layer_report import write_box  # noqa: E402

pytest.importorskip("pyvista", reason="the case is read through pyvista's OpenFOAM reader",
                    exc_type=ImportError)


# -- the channel, at numbers whose answer is arithmetic -------------------------

HEIGHT_M = 0.02
LENGTH_M = 0.2
NU = 1.0e-05
U_MEAN = 0.1
TAU_WALL = 6.0 * NU * U_MEAN / HEIGHT_M
"""3.0e-04 m2/s2, kinematic -- OpenFOAM's incompressible `wallShearStress` is tau/rho,
which is what `showcase.BARS` says on the bar under the picture."""


@pytest.fixture(scope="module")
def showcase():
    spec = importlib.util.spec_from_file_location("toolbox_showcase",
                                                  TOOLBOX / "showcase.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _field(path: Path, cls: str, dimensions: str, internal: str,
           boundary: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = "".join(f"    {name}\n    {{\n{body}\n    }}\n"
                     for name, body in boundary.items())
    path.write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    location    \"{path.parent.name}\";\n"
        f"    object      {path.name};\n}}\n\n"
        f"dimensions      {dimensions};\n\ninternalField   uniform {internal};\n\n"
        f"boundaryField\n{{\n{blocks}}}\n", encoding="utf-8")


@pytest.fixture(scope="module")
def channel(tmp_path_factory) -> Path:
    """A plane channel of height H, walled top and bottom, carrying the exact shear.

    The mesh is `test_toolbox_layer_report.write_box`, which writes a real ASCII
    `constant/polyMesh` -- the same fixture `test_toolbox_wall.py` derives y+ on, so
    one hand-written mesh serves both readers.
    """
    case = tmp_path_factory.mktemp("poiseuille") / "channel"
    case.mkdir()
    write_box(case,
              np.linspace(0.0, LENGTH_M, 9),
              np.linspace(0.0, 0.01, 3),
              np.linspace(0.0, HEIGHT_M, 11),
              wall=("zmin", "zmax"))
    calculated = "        type            calculated;\n        value           uniform "
    _field(case / "0.5" / "wallShearStress", "volVectorField", "[0 2 -2 0 0 0 0]",
           "(0 0 0)",
           # Negative x: the function object writes the stress the wall exerts on the
           # fluid, which opposes the flow. The map takes the magnitude, and a map that
           # took a component instead would come back signed.
           {"wall": calculated + f"(-{TAU_WALL:.12g} 0 0);",
            "outer": calculated + "(0 0 0);"})
    _field(case / "0.5" / "p", "volScalarField", "[0 2 -2 0 0 0 0]", "0",
           {"wall": "        type            zeroGradient;",
            "outer": calculated + "0;"})
    (case / "system").mkdir(exist_ok=True)
    (case / "system" / "controlDict").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
        "    object controlDict;\n}\n\napplication     simpleFoam;\n"
        "startTime       0;\nendTime         0.5;\ndeltaT          0.1;\n"
        "writeInterval   1;\n", encoding="utf-8")
    return case


def test_the_wall_map_reads_back_the_exact_poiseuille_wall_shear(showcase, channel):
    """6 nu u_mean / H, through the whole reading path and out as the array the
    renderer colours by. Nothing here was measured; it was differentiated."""
    _reader, block, chosen, times = showcase.open_case(channel, None)
    assert times == [0.5] and chosen == pytest.approx(0.5)

    patch, name = showcase.body_patch(block, "wall")
    assert patch is not None, "the wall patch did not come back off the case"
    assert name == "wall"

    surface, scalars = showcase.surface_scalars(patch, "wallShearStress")
    assert scalars == "|wallShearStress|", (
        "a vector wall field has to be reduced to its magnitude by name, so the "
        "picture says which number it is"
    )
    values = np.asarray(surface.point_data[scalars])
    assert values.size == patch.n_points
    # Single precision, because that is what the reader carries: VTK stores an
    # OpenFOAM field as float32 whatever the file held. Measured here, the map reads
    # 3.000000142492354e-04 against an exact 3.0e-04, a relative error of 4.7e-08 --
    # float32 eps is 1.2e-07, so the whole of the discrepancy is the storage and none
    # of it is the reading. A tighter tolerance here would be a test asserting a
    # precision the picture never had.
    assert float(values.min()) == pytest.approx(TAU_WALL, rel=1e-6)
    assert float(values.max()) == pytest.approx(TAU_WALL, rel=1e-6)
    assert abs(float(values.max()) - TAU_WALL) / TAU_WALL < np.finfo(np.float32).eps


def test_the_bar_under_the_picture_says_the_shear_is_divided_by_density(showcase):
    """Incompressible `wallShearStress` is tau/rho and a reader cannot see that in the
    colours. A map labelled in Pa that is in m2/s2 is wrong by the density -- 1000 on
    water -- and looks identical."""
    assert "/rho" in showcase.BARS["wallShearStress"]
    assert "m2/s2" in showcase.BARS["wallShearStress"]


def test_the_map_for_a_shear_is_sequential_rather_than_diverging(showcase):
    """|tau| cannot be negative, so a diverging map would centre the picture on a value
    the quantity never takes and invent a midline in it."""
    assert showcase.MAPS["wallShearStress"] == "magma"


def test_a_wall_with_no_shear_written_is_drawn_plain_rather_than_at_zero(showcase,
                                                                        channel):
    """The refusal that matters: a patch carrying nothing is not a patch carrying
    zeroes. `wall.py:508-526` had the same bug the other way round and reported a
    distribution over twenty faces manufactured from one placeholder."""
    _reader, block, _chosen, _times = showcase.open_case(channel, None)
    patch, _name = showcase.body_patch(block, "wall")
    _surface, scalars = showcase.surface_scalars(patch, "nowhere")
    assert scalars == ""
