"""The views that compare: iso-contours, a difference between two runs, a wall map.

What is tested here is the arithmetic and the refusals, not the pictures. All three of
these additions can produce a figure that looks entirely reasonable and means nothing --
a difference drawn from two meshes that do not correspond, two panels each scaled to
their own range, a vector field colour-mapped by whichever component VTK reached for --
and none of those is visible by eye in the output. The draws themselves are in
`test_toolbox_render_smoke.py`, behind the off-screen OpenGL gate.

    python3 -m pytest tests/test_toolbox_views.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"

np = pytest.importorskip("numpy")


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def render():
    """`render.py` imports pyvista inside the functions that draw, so it loads here."""
    return load("render")


@pytest.fixture(scope="module")
def showcase():
    """`showcase.py` imports pyvista at module level, so this needs it present -- not a
    renderer, only the library: nothing in these tests draws."""
    pytest.importorskip("pyvista", reason="showcase imports pyvista at module level",
                        exc_type=ImportError)
    return load("showcase")


class Mesh:
    """Just enough of a mesh for `correspondence`: a point count and a bounding box.

    The rule it checks is entirely about those two things, and a stub says so more
    plainly than a real dataset would -- and it runs on a machine with no pyvista.
    """

    def __init__(self, n_points, bounds):
        self.n_points = n_points
        self.bounds = bounds


BOX = (0.0, 2.0, 0.0, 1.0, 0.0, 0.1)


# -- iso-contour lines ---------------------------------------------------------------


def test_contour_levels_come_from_the_field_and_the_count_is_honoured(render):
    """Ten lines means ten lines. `linspace(min, max, 10)` is eight lines and two that
    sit on the extremes, where an iso-line is a single point or the domain edge."""
    field = np.linspace(-3.0, 7.0, 400)

    levels = render.contour_levels(field, 10)

    assert len(levels) == 10
    assert min(levels) > -3.0 and max(levels) < 7.0
    assert levels == sorted(levels)
    # Evenly spaced across the range the field actually has.
    gaps = np.diff(levels)
    assert np.allclose(gaps, gaps[0])


def test_a_uniform_field_has_no_contour_levels(render):
    """There is no interior level between a value and itself, and `contour` on one
    returns an empty surface -- so this says so rather than drawing nothing."""
    assert render.contour_levels(np.full(50, 4.0), 8) == []
    assert render.contour_levels(np.array([]), 8) == []
    assert render.contour_levels(np.linspace(0, 1, 10), 0) == []


def test_contour_levels_ignore_the_values_that_are_not_numbers(render):
    """A field with a NaN in it still has a range; taking min/max over the NaN gives
    NaN levels, and every contour then draws nothing at all."""
    field = np.array([0.0, 1.0, np.nan, 2.0, np.inf])

    levels = render.contour_levels(field, 3)

    assert len(levels) == 3
    assert all(np.isfinite(levels))
    assert max(levels) < 2.0


def test_a_vector_field_is_contoured_on_its_magnitude(render):
    """`U` is three numbers per point and a contour level is one, so the magnitude is
    what the lines are levels of."""
    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])

    assert np.allclose(render.as_scalar(vectors), [5.0, 1.0])
    assert np.allclose(render.as_scalar(np.array([1.0, 2.0])), [1.0, 2.0])


# -- the difference between two runs -------------------------------------------------


def test_the_difference_of_a_field_with_itself_is_exactly_zero(render):
    """The one case with a known answer. A difference view that cannot get this right
    is reporting its own interpolation error as physics."""
    field = np.linspace(-5.0, 11.0, 997)

    record = render.diff_record(field, field, field="p", normal="z",
                                names=("run_A", "run_A2"), how="same-mesh")

    assert record["max_abs_difference"] == 0.0
    assert record["identical"] is True
    assert np.all(np.asarray(field) - np.asarray(field) == 0.0)
    # Zero everywhere still has to produce a usable scale rather than [0, 0], which VTK
    # draws as one flat colour with no bar worth reading.
    assert record["diff_clim"] == [-1.0, 1.0]


def test_both_runs_are_recorded_on_one_colour_scale_and_one_camera(render):
    """Asserted on the recorded numbers, not on pixels. Two panels each scaled to their
    own range are two pictures that cannot be read against each other, and the camera is
    the other half of the same claim."""
    first = np.linspace(0.0, 10.0, 100)
    second = np.linspace(-2.0, 8.0, 100)

    record = render.diff_record(first, second, field="p", normal="z",
                                names=("coarse", "fine"), how="resampled")

    assert record["clim"] == [-2.0, 10.0], "the scale must span both runs, not one"
    assert record["cmap"] == "viridis"
    # The difference gets its own scale, and that one is symmetric about zero on a
    # diverging map -- the same thing render.py already does for signed vorticity.
    assert record["diff_clim"] == [-2.0, 2.0]
    assert record["diff_cmap"] == "coolwarm"
    assert record["view_vector"] == list(render.NORMALS["z"])
    assert record["viewup"] == list(render.VIEWUP["z"])
    assert record["runs"] == ["coarse", "fine"]
    assert record["correspondence"] == "resampled"


def test_the_same_mesh_twice_is_differenced_point_by_point(render):
    """Same point count over the same box: no resampling, so no interpolation error in
    the answer at all."""
    assert render.correspondence(Mesh(400, BOX), Mesh(400, BOX)) == "same-mesh"


def test_two_meshes_of_one_domain_are_resampled(render):
    """A mesh study is exactly this: one domain, two discretisations. Refusing it would
    refuse the comparison most worth drawing."""
    assert render.correspondence(Mesh(400, BOX), Mesh(1600, BOX)) == "resampled"


def test_mismatched_meshes_are_refused_by_name(render):
    """A different domain is a different problem, and a difference between two problems
    is a plausible picture of nothing. Both runs are named so the refusal says which
    pair was asked for rather than that something was wrong."""
    other = (0.0, 2.4, 0.0, 1.0, 0.0, 0.1)

    with pytest.raises(SystemExit) as refused:
        render.correspondence(Mesh(400, BOX), Mesh(400, other), "step_h1", "step_h2")

    message = str(refused.value)
    assert "step_h1" in message and "step_h2" in message
    assert "refusing" in message.lower()
    assert "same domain" in message


def test_a_mesh_shifted_within_tolerance_is_still_the_same_case(render):
    """The tolerance is there because two meshers of one geometry disagree in the last
    digits of a bound, and a difference view refused over a micron is a view nobody can
    use."""
    nudged = (0.0, 2.0 + 1e-5, 0.0, 1.0, 0.0, 0.1)

    assert render.correspondence(Mesh(400, BOX), Mesh(400, nudged)) == "resampled"


def test_the_refusal_names_the_axis_the_two_runs_disagree_on(render):
    with pytest.raises(SystemExit) as refused:
        render.correspondence(Mesh(400, BOX), Mesh(400, (0.0, 2.0, 0.0, 1.9, 0.0, 0.1)))

    assert " on y" in str(refused.value)


def grid(cells: int, origin=(0.0, 0.0, 0.0)):
    """A flat uniform grid carrying `p = x`, which has an exact value everywhere so a
    resampled difference has a known answer rather than an interpolated one."""
    pv = pytest.importorskip("pyvista", exc_type=ImportError)
    step = 2.0 / cells
    mesh = pv.ImageData(dimensions=(cells + 1, cells + 1, 1), spacing=(step, step, step),
                        origin=origin)
    mesh.point_data["p"] = np.asarray(mesh.points)[:, 0].copy()
    return mesh


def test_one_run_differenced_against_itself_is_zero_at_every_point(render):
    """The same thing again on real datasets, through the sampling path and all: the
    two arrays come back equal, so the picture is flat zero."""
    first, second = grid(8), grid(8)

    _, a, b, how = render.difference_on(first, second, "p", "first", "second")

    assert how == "same-mesh"
    assert np.all(a - b == 0.0), "a run differenced against itself is not zero"


def test_a_finer_mesh_of_the_same_domain_is_resampled_onto_the_first(render):
    """`p = x` on both, so the difference is zero to interpolation error and the test is
    about the plumbing: the second run's array arriving on the first run's points, and
    not being quietly mixed with the first run's own."""
    first, second = grid(8), grid(13)

    cut, a, b, how = render.difference_on(first, second, "p", "coarse", "fine")

    assert how == "resampled"
    assert a.shape == b.shape == (cut.n_points,)
    assert np.max(np.abs(a - b)) < 1e-9


def test_a_second_run_covering_only_part_of_the_cut_is_refused(render):
    """The other half of the correspondence rule. The boxes agree, so this gets past
    `correspondence`, and the sampling is what finds out: half of the first run's points
    land outside the second's cells, and the picture would be the difference between a
    field and nothing over most of its area."""
    first = grid(8)
    # Two opposite corner cells of the same box: the bounds are unchanged, so this is
    # the same domain by every measure `correspondence` can take, and there is nothing
    # under the middle of the cut.
    second = grid(8).extract_cells([0, grid(8).n_cells - 1])

    with pytest.raises(SystemExit) as refused:
        render.difference_on(first, second, "p", "whole", "gappy")

    message = str(refused.value)
    assert "whole" in message and "gappy" in message
    assert "refusing" in message.lower()
    assert "%" in message, "the refusal should say how much of the cut was covered"


@pytest.mark.parametrize("name", ["difference_on", "render_difference", "render_panel"])
def test_the_difference_view_is_assembled_from_functions_that_can_be_called(render, name):
    assert callable(getattr(render, name))


def test_the_difference_is_offered_on_the_command_line(render):
    """A view nothing can select is a view nobody runs."""
    source = (TOOLBOX / "render.py").read_text(encoding="utf-8")
    assert '"--diff"' in source
    assert '"--contours"' in source


# -- the surface field ----------------------------------------------------------------


def test_the_surface_field_still_defaults_to_pressure(showcase):
    """The point of the flag is that the field stopped being hard-coded, not that the
    picture changed: every existing `--scene surface` call draws what it drew."""
    import inspect

    assert inspect.signature(showcase.scene_surface).parameters["surface_field"].default == "p"
    assert inspect.signature(showcase.scene_section).parameters["surface_field"].default == "p"
    assert showcase.MAPS["p"] == "coolwarm"
    assert showcase.BARS["p"] == "pressure  p/rho  m2/s2"


def test_a_scalar_surface_field_is_coloured_by_itself(showcase):
    pv = pytest.importorskip("pyvista", exc_type=ImportError)
    patch = pv.Plane(i_resolution=4, j_resolution=4)
    patch.point_data["p"] = np.linspace(-1.0, 1.0, patch.n_points)

    patch, scalars = showcase.surface_scalars(patch, "p")

    assert scalars == "p"


def test_wall_shear_is_accepted_and_drawn_as_its_magnitude(showcase):
    """`wallShearStress` is a vector, and a wall map of it is |tau|. Handed to the
    renderer as three components it is coloured by whichever one VTK picks -- a picture
    of the x-component captioned as the wall shear."""
    pv = pytest.importorskip("pyvista", exc_type=ImportError)
    patch = pv.Plane(i_resolution=4, j_resolution=4)
    tau = np.zeros((patch.n_points, 3))
    tau[:, 0] = 3.0
    tau[:, 1] = 4.0
    patch.point_data["wallShearStress"] = tau

    patch, scalars = showcase.surface_scalars(patch, "wallShearStress")

    assert scalars == "|wallShearStress|"
    assert np.allclose(patch.point_data[scalars], 5.0)
    assert showcase.MAPS["wallShearStress"] == "magma", (
        "a shear magnitude is non-negative, so a diverging map centres it on nothing"
    )
    assert "tau" in showcase.BARS["wallShearStress"]


def test_a_surface_field_on_the_faces_is_moved_to_the_points(showcase):
    """A patch carries `wallShearStress` per face, which shades flat per facet."""
    pv = pytest.importorskip("pyvista", exc_type=ImportError)
    patch = pv.Plane(i_resolution=4, j_resolution=4)
    patch.cell_data["p"] = np.linspace(0.0, 1.0, patch.n_cells)

    patch, scalars = showcase.surface_scalars(patch, "p")

    assert scalars == "p"
    assert "p" in patch.point_data


def test_a_patch_without_the_field_says_so_rather_than_inventing_one(showcase):
    pv = pytest.importorskip("pyvista", exc_type=ImportError)
    patch = pv.Plane(i_resolution=2, j_resolution=2)

    _, scalars = showcase.surface_scalars(patch, "wallShearStress")

    assert scalars == ""
