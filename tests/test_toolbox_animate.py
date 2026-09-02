"""animate.py: everything about an animation that is decided before anything is drawn.

Which write times are in it, which frames a second run has to redraw, what the
colour limits are and where they came from, and what the sidecar tells the machine
that does the encoding. None of that needs VTK, and all of it is the kind of
arithmetic that rots quietly, so it is tested here -- on a machine with no pyvista
on it, which is also why the script must not import pyvista at module level.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

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
def animate():
    return load("animate")


def frames_on_disk(directory: Path, indices, size: int = 32) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in indices:
        (directory / f"frame_{index:04d}.png").write_bytes(b"x" * size)
    return directory


# -- which write times ---------------------------------------------------------


def test_times_are_filtered_before_the_stride(animate):
    """`--every 2` means every second time in the window asked for, not every
    second time in the case: otherwise `--from` silently shifts the phase."""
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert animate.select_times(times, every=2, start=1.0) == [1.0, 2.0]
    assert animate.select_times(times, every=2) == [0.0, 1.0, 2.0]


def test_the_window_is_inclusive_at_both_ends(animate):
    times = [0.0, 1.0, 2.0, 3.0]
    assert animate.select_times(times, start=1.0, end=2.0) == [1.0, 2.0]


def test_a_stride_below_one_does_not_empty_the_sequence(animate):
    assert animate.select_times([0.0, 1.0], every=0) == [0.0, 1.0]


def test_times_come_back_as_floats(animate):
    """The reader hands back numpy scalars; the sidecar has to be JSON."""
    chosen = animate.select_times(np.array([0.0, 1.0]))
    assert [type(value) for value in chosen] == [float, float]
    json.dumps(chosen)


# -- frames, and picking up where the last run stopped -------------------------


def test_frame_names_sort_in_time_order(animate):
    """Zero padding, because frame_10 sorting before frame_2 is how a finished
    animation ends up playing its middle backwards."""
    names = [animate.frame_name(index) for index in (0, 2, 10)]
    assert names == ["frame_0000.png", "frame_0002.png", "frame_0010.png"]
    assert sorted(names) == names


def test_the_plan_marks_frames_already_on_disk(tmp_path, animate):
    frames_on_disk(tmp_path / "wake_frames", [0, 1])
    plan = animate.plan_frames(tmp_path / "wake_frames", [0.0, 0.5, 1.0])

    assert [row["ready"] for row in plan] == [True, True, False]
    assert [row["time"] for row in plan] == [0.0, 0.5, 1.0]
    assert [row["name"] for row in plan] == ["frame_0000.png", "frame_0001.png", "frame_0002.png"]


def test_an_empty_png_is_not_a_finished_frame(tmp_path, animate):
    """A zero-length file is what an interrupted screenshot leaves behind; treating
    it as done would put a black hole in the middle of the animation."""
    directory = tmp_path / "frames"
    directory.mkdir()
    (directory / "frame_0000.png").write_bytes(b"")

    assert animate.frame_ready(directory / "frame_0000.png") is False
    assert animate.plan_frames(directory, [0.0])[0]["ready"] is False


def test_a_missing_directory_is_simply_all_pending(tmp_path, animate):
    plan = animate.plan_frames(tmp_path / "never_made", [0.0, 1.0])
    assert animate.pending_frames(plan) == plan


def test_resume_renders_only_what_is_new(tmp_path, animate):
    frames_on_disk(tmp_path / "frames", [0, 1, 2])
    plan = animate.plan_frames(tmp_path / "frames", [0.0, 1.0, 2.0, 3.0, 4.0])

    todo = animate.pending_frames(plan)

    assert [row["index"] for row in todo] == [3, 4]


def test_force_redraws_everything(tmp_path, animate):
    frames_on_disk(tmp_path / "frames", [0, 1])
    plan = animate.plan_frames(tmp_path / "frames", [0.0, 1.0])

    assert animate.pending_frames(plan, force=True) == plan


def test_new_write_times_keep_the_old_frames_named_as_they_were(tmp_path, animate):
    """The index is the position in the selected window, so a solve that has
    written three more times since the last run appends three frames rather than
    renumbering -- which is what makes re-running safe mid-solve."""
    first = animate.plan_frames(tmp_path / "frames", [0.0, 1.0])
    second = animate.plan_frames(tmp_path / "frames", [0.0, 1.0, 2.0])

    assert [row["name"] for row in second][:2] == [row["name"] for row in first]


# -- colour limits -------------------------------------------------------------


def test_percentile_limits_trim_the_extremes(animate):
    """Raw min/max is set by a handful of cells at a wall, and the flow then draws
    in the middle two colours of the map."""
    values = np.concatenate([np.zeros(98), np.array([1000.0, -1000.0])])
    low, high = animate.percentile_limits(values, 2.0, 98.0)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_percentile_limits_ignore_what_is_not_finite(animate):
    values = np.array([0.0, 10.0, np.nan, np.inf])
    low, high = animate.percentile_limits(values, 0.0, 100.0)
    assert (low, high) == (0.0, 10.0)


def test_percentile_limits_of_nothing_is_none(animate):
    assert animate.percentile_limits(np.array([np.nan, np.nan])) is None


def test_merged_limits_cover_every_frame_sampled(animate):
    assert animate.merge_limits([(-1.0, 4.0), (-6.0, 2.0), None]) == (-6.0, 4.0)


def test_a_flat_field_is_widened_rather_than_passed_on(animate):
    """VTK draws a zero-width colour range as one flat colour, which reads as a
    broken render instead of a uniform field."""
    low, high = animate.merge_limits([(3.0, 3.0)])
    assert low < 3.0 < high


def test_merged_limits_of_nothing_is_none(animate):
    assert animate.merge_limits([None, (np.nan, 1.0)]) is None


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("first", [0.0]),
        ("last", [2.0]),
        ("all", [0.0, 1.0, 2.0]),
        ("explicit", []),
    ],
)
def test_the_limit_source_decides_which_times_get_read(animate, mode, expected):
    assert animate.limit_times([0.0, 1.0, 2.0], mode) == expected


def test_an_explicit_range_beats_the_recorded_one(animate):
    assert animate.carried_limits({"clim": [-1.0, 1.0]}, explicit=[0.0, 5.0]) == (0.0, 5.0)


def test_limits_are_carried_from_the_sidecar(animate):
    """The whole point of recording them: a second run must not resample and give
    the frames it adds a different scale from the frames already on disk."""
    assert animate.carried_limits({"clim": [-2.0, 7.0]}) == (-2.0, 7.0)


def test_reclim_throws_the_recorded_limits_away(animate):
    assert animate.carried_limits({"clim": [-2.0, 7.0]}, reclim=True) is None


def test_nothing_recorded_means_sample_them(animate):
    assert animate.carried_limits({}) is None
    assert animate.carried_limits({"clim": "junk"}) is None
    assert animate.carried_limits(None) is None


# -- is this the same animation as last time? ----------------------------------


def SETTINGS(**overrides):
    base = {"field": "vorticity", "component": "z", "normal": "z", "cmap": "RdBu_r",
            "streamlines": 0, "size": [1000, 750]}
    base.update(overrides)
    return base


def test_the_same_settings_continue_the_sequence(animate):
    assert animate.same_render(SETTINGS(), SETTINGS()) is True


@pytest.mark.parametrize(
    "changed",
    [{"field": "p"}, {"component": "mag"}, {"normal": "y"}, {"cmap": "viridis"},
     {"streamlines": 200}, {"size": [640, 480]}],
)
def test_a_setting_that_changes_the_picture_ends_the_sequence(animate, changed):
    """The frames directory is named after the case and nothing else, so
    `--field pressure` re-run over a vorticity sequence lands in the same
    directory. Its four finished PNGs are pictures of vorticity and must not be
    kept and relabelled."""
    assert animate.same_render(SETTINGS(), SETTINGS(**changed)) is False


def test_a_sidecar_missing_a_setting_is_not_a_disagreement(animate):
    """No previous run, or one written before a setting was recorded: an existing
    sequence is not redrawn just because the sidecar is older than the check."""
    assert animate.same_render({}, SETTINGS()) is True
    assert animate.same_render(None, SETTINGS()) is True
    assert animate.same_render({"field": "vorticity"}, SETTINGS()) is True


def test_streamline_counts_compare_as_numbers(animate):
    assert animate.same_render(SETTINGS(streamlines=None), SETTINGS(streamlines=0)) is True
    assert animate.same_render(SETTINGS(streamlines=200), SETTINGS(streamlines=50)) is False


# -- the sidecar ---------------------------------------------------------------


def test_the_sidecar_names_the_container_the_rate_and_the_order(tmp_path, animate):
    frames_on_disk(tmp_path / "wake_frames", [0, 1, 2])
    plan = animate.plan_frames(tmp_path / "wake_frames", [0.0, 0.5, 1.0])

    data = animate.build_sidecar(
        plan, frames_dir_name="wake_frames", container="mp4", fps=24.0,
        field="vorticity", scalar="vorticity_z", clim=(-5.0, 5.0), case="cylinder",
    )

    assert data["container"] == "mp4"
    assert data["output"] == "wake.mp4"
    assert data["fps"] == 24.0
    assert data["frames"] == ["frame_0000.png", "frame_0001.png", "frame_0002.png"]
    assert data["frame_times"] == [0.0, 0.5, 1.0]
    assert data["clim"] == [-5.0, 5.0]
    assert data["complete"] == data["expected"] == 3
    assert data["partial"] is False
    json.dumps(data)


def test_the_sidecar_counts_a_run_that_is_still_going(tmp_path, animate):
    """A frames directory that arrives home half-finished still says what it was
    going to be, and how much of it is there."""
    frames_on_disk(tmp_path / "frames", [0, 1])
    plan = animate.plan_frames(tmp_path / "frames", [0.0, 1.0, 2.0, 3.0])

    data = animate.build_sidecar(plan, frames_dir_name="frames")

    assert (data["complete"], data["expected"]) == (2, 4)
    assert data["partial"] is True
    assert data["frames"] == ["frame_0000.png", "frame_0001.png"]


def test_a_gap_in_the_frames_withholds_the_printf_pattern(tmp_path, animate):
    """Encoders that take `frame_%04d.png` stop at the first missing number, so
    offering the pattern over a hole would truncate the animation silently."""
    frames_on_disk(tmp_path / "frames", [0, 2])
    plan = animate.plan_frames(tmp_path / "frames", [0.0, 1.0, 2.0])

    data = animate.build_sidecar(plan, frames_dir_name="frames")

    assert data["pattern"] is None
    assert data["frames"] == ["frame_0000.png", "frame_0002.png"]


def test_frames_left_by_a_wider_window_are_not_part_of_this_sequence(tmp_path, animate):
    """The frames directory is named after the case, so a second run with `--to 2`
    lands on top of a longer sequence. `frame_%04d.png` walks the numbers on disk,
    not the sidecar's list, so the leftovers would be spliced onto the end."""
    frames_on_disk(tmp_path / "frames", [0, 1, 2, 3, 4])
    plan = animate.plan_frames(tmp_path / "frames", [0.0, 1.0, 2.0])

    stray = animate.stray_frames(tmp_path / "frames", plan)
    data = animate.build_sidecar(plan, frames_dir_name="frames", stray=stray)

    assert stray == ["frame_0003.png", "frame_0004.png"]
    assert data["stray"] == stray
    assert data["pattern"] is None
    assert data["frames"] == ["frame_0000.png", "frame_0001.png", "frame_0002.png"]


def test_nothing_is_stray_when_the_window_did_not_shrink(tmp_path, animate):
    frames_on_disk(tmp_path / "frames", [0, 1])
    plan = animate.plan_frames(tmp_path / "frames", [0.0, 1.0, 2.0])

    assert animate.stray_frames(tmp_path / "frames", plan) == []
    assert animate.stray_frames(tmp_path / "never_made", plan) == []


def test_a_half_written_frame_is_not_counted_as_stray(tmp_path, animate):
    """The dotted temporary a screenshot is written under sits in the same
    directory and must not be reported as somebody else's frame."""
    directory = frames_on_disk(tmp_path / "frames", [0])
    (directory / ".frame_0000.part.png").write_bytes(b"half")

    assert animate.stray_frames(directory, animate.plan_frames(directory, [0.0])) == []


def test_a_full_run_offers_the_pattern(animate):
    assert animate.contiguous_pattern(["frame_0000.png", "frame_0001.png"]) == "frame_%04d.png"
    assert animate.contiguous_pattern([]) is None


def test_a_gif_records_its_loop_and_an_mp4_does_not(tmp_path, animate):
    plan = animate.plan_frames(tmp_path / "frames", [0.0])
    assert animate.build_sidecar(plan, container="gif")["loop"] == 0
    assert "loop" not in animate.build_sidecar(plan, container="mp4")


def test_the_sidecar_says_where_the_encoding_happens(tmp_path, animate):
    plan = animate.plan_frames(tmp_path / "frames", [0.0])
    assert "no encoder" in animate.build_sidecar(plan)["encoded_by"]


@pytest.mark.parametrize(
    "directory, container, expected",
    [
        ("wake_frames", "gif", "wake.gif"),
        ("wake_frames", "webp", "wake.webp"),
        ("stills", "mp4", "stills.mp4"),
    ],
)
def test_the_output_name_comes_from_the_frames_directory(animate, directory, container, expected):
    assert animate.output_name(directory, container) == expected


def test_the_sidecar_round_trips_and_leaves_no_temporary_behind(tmp_path, animate):
    """It is rewritten after every frame while the mirror is copying the directory,
    so it is moved into place rather than written in place."""
    plan = animate.plan_frames(tmp_path / "frames", [0.0, 1.0])
    animate.write_sidecar(tmp_path / "frames", animate.build_sidecar(plan))

    assert [p.name for p in (tmp_path / "frames").iterdir()] == ["frames.json"]
    assert animate.read_sidecar(tmp_path / "frames")["expected"] == 2


def test_a_sidecar_that_will_not_parse_is_not_fatal(tmp_path, animate):
    directory = tmp_path / "frames"
    directory.mkdir()
    (directory / "frames.json").write_text("{ half a fi")

    assert animate.read_sidecar(directory) == {}
    assert animate.read_sidecar(tmp_path / "absent") == {}


# -- labels --------------------------------------------------------------------


def test_the_label_carries_the_time_the_field_and_the_reynolds_number(animate):
    label = animate.frame_label(2.5, "vorticity_z", 16500.0, case="cylinder")
    assert "cylinder" in label
    assert "vorticity_z" in label
    assert "t = 2.5 s" in label
    assert "Re = 1.65e+04" in label


def test_an_unknown_reynolds_number_is_left_off_rather_than_guessed(animate):
    label = animate.frame_label(1.0, "p")
    assert "Re" not in label
    assert label.splitlines() == ["p", "t = 1 s"]


# -- the camera ----------------------------------------------------------------


def test_the_camera_fits_the_slice_it_was_given(animate):
    """parallel_scale is half the visible height, so a slice wider than the window
    is fitted by its width divided by the window aspect."""
    camera = animate.camera_setup((0, 2, 0, 1, 0, 0), "z", window=(1000, 750), padding=0.0)
    assert camera["parallel_scale"] == pytest.approx(1.0 / (1000 / 750))

    tall = animate.camera_setup((0, 1, 0, 4, 0, 0), "z", window=(1000, 750), padding=0.0)
    assert tall["parallel_scale"] == pytest.approx(2.0)


def test_the_padding_only_widens_the_view(animate):
    tight = animate.camera_setup((0, 2, 0, 1, 0, 0), "z", padding=0.0)
    padded = animate.camera_setup((0, 2, 0, 1, 0, 0), "z", padding=0.05)
    assert padded["parallel_scale"] == pytest.approx(tight["parallel_scale"] * 1.05)


def test_the_camera_stands_off_along_the_normal_and_looks_at_the_centre(animate):
    camera = animate.camera_setup((0, 2, 0, 1, 0, 0), "z")
    assert camera["focal_point"] == pytest.approx((1.0, 0.5, 0.0))
    assert camera["position"][:2] == pytest.approx((1.0, 0.5))
    assert camera["position"][2] > 0
    assert camera["up"] == (0, 1, 0)


def test_an_x_normal_slice_is_drawn_the_right_way_up(animate):
    camera = animate.camera_setup((0, 0.1, 0, 2, 0, 1), "x")
    assert camera["up"] == (0, 0, 1)
    assert camera["position"][0] > camera["focal_point"][0]


def test_a_flat_slice_does_not_divide_by_zero(animate):
    camera = animate.camera_setup((0, 0, 0, 0, 0, 0), "z")
    assert camera["parallel_scale"] > 0


# -- fields --------------------------------------------------------------------


class FakeMesh:
    """Stands in for a pyvista mesh: a point_data dict and a derivative filter."""

    def __init__(self, point_data, derivative=None):
        self.point_data = dict(point_data)
        self._derivative = derivative

    @property
    def center(self):
        return (0.0, 0.0, 0.0)

    def compute_derivative(self, scalars=None, vorticity=False):
        assert self._derivative is not None, "compute_derivative should not have been called"
        assert (scalars, vorticity) == ("U", True)
        return FakeMesh({**self.point_data, "vorticity": self._derivative})


def velocity(rows=4):
    return np.tile(np.array([3.0, 4.0, 0.0]), (rows, 1))


@pytest.mark.parametrize(
    "component, name, expected",
    [("mag", "U_mag", 5.0), ("x", "U_x", 3.0), ("y", "U_y", 4.0)],
)
def test_a_vector_field_becomes_the_component_asked_for(animate, component, name, expected):
    mesh, scalar = animate.scalar_of(FakeMesh({"U": velocity()}), "U", component)
    assert scalar == name
    assert mesh.point_data[name][0] == pytest.approx(expected)


def test_a_scalar_field_is_used_as_it_is(animate):
    mesh, scalar = animate.scalar_of(FakeMesh({"k": np.ones(4)}), "k")
    assert scalar == "k"


def test_vorticity_is_computed_when_the_case_did_not_write_it(animate):
    spin = np.tile(np.array([0.0, 0.0, -2.0]), (4, 1))
    mesh, scalar = animate.scalar_of(FakeMesh({"U": velocity()}, derivative=spin), "vorticity")

    assert scalar == "vorticity_z"
    assert mesh.point_data["vorticity_z"][0] == pytest.approx(-2.0)


def test_vorticity_the_case_wrote_is_not_recomputed(animate):
    """`compute_derivative` on the fake raises if it is called; a case that ran the
    vorticity function object already has the better answer."""
    written = np.tile(np.array([0.0, 0.0, 1.5]), (4, 1))
    _mesh, scalar = animate.scalar_of(FakeMesh({"U": velocity(), "vorticity": written}), "vorticity")
    assert scalar == "vorticity_z"


def test_vorticity_without_u_says_so(animate):
    with pytest.raises(SystemExit) as caught:
        animate.scalar_of(FakeMesh({"p": np.ones(4)}), "vorticity")
    assert "U" in str(caught.value)


def test_a_missing_field_lists_what_the_case_does_have(animate):
    with pytest.raises(SystemExit) as caught:
        animate.scalar_of(FakeMesh({"p": np.ones(4), "k": np.ones(4)}), "T")
    message = str(caught.value)
    assert "T" in message and "p" in message and "k" in message


@pytest.mark.parametrize(
    "asked, resolved",
    [("velocity", "U"), ("pressure", "p"), ("vorticity", "vorticity"),
     ("k", "k"), ("omega", "omega"), ("nut", "nut"), ("alphaWater", "alphaWater")],
)
def test_friendly_names_resolve_and_unknown_ones_pass_through(animate, asked, resolved):
    assert animate.resolve_field(asked) == resolved


def test_vorticity_keeps_its_sign_by_default(animate):
    """A magnitude would throw away exactly what separates the two rows of a
    von Karman street."""
    assert animate.default_component("vorticity") == "z"
    assert animate.default_component("U") == "mag"


def test_the_predicted_scalar_name_matches_what_the_render_produces(animate):
    """The sidecar is written before any mesh is read, including on a resumed run
    that draws nothing at all."""
    for field, component in (("U", None), ("U", "x"), ("vorticity", None), ("k", None)):
        mesh = FakeMesh({"U": velocity(), "vorticity": velocity(), "k": np.ones(4)})
        _mesh, scalar = animate.scalar_of(mesh, field, component)
        assert animate.scalar_name(field, component) == scalar


def test_fields_around_zero_get_a_diverging_map(animate):
    assert animate.default_cmap("vorticity") == "RdBu_r"
    assert animate.default_cmap("p") == "coolwarm"
    assert animate.default_cmap("U") == "viridis"
    assert animate.default_cmap("k") == "inferno"


def test_seeds_are_spread_over_the_slice(animate):
    assert animate.seed_stride(1000, 200) == 5
    assert animate.seed_stride(10, 200) == 1
    assert animate.seed_stride(0, 0) == 1


# -- the Reynolds number on the label ------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "nu              nu [ 0 2 -1 0 0 0 0 ] 1e-05;",
        "nu              [0 2 -1 0 0 0 0] 1e-05;",
        "nu              1e-05;",
    ],
)
def test_viscosity_is_read_from_either_spelling(animate, entry):
    assert animate.parse_nu(entry) == pytest.approx(1e-05)


def test_no_viscosity_entry_is_not_an_error(animate):
    assert animate.parse_nu("transportModel Newtonian;") is None


def test_the_reference_velocity_is_the_largest_uniform_vector(animate):
    """The internal field and the walls are zero and the inlet is not."""
    field = """
    internalField   uniform (0 0 0);
    boundaryField
    {
        inlet { type fixedValue; value uniform (1.5 0 0); }
        walls { type noSlip; }
    }
    """
    assert animate.parse_reference_velocity(field) == pytest.approx(1.5)
    assert animate.parse_reference_velocity("internalField uniform (0 0 0);") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("// Re = 1.65e4 turbulent", 16500.0),
        ("Reynolds number: 200", 200.0),
        ("// Region = 4", None),
        ("nothing here", None),
    ],
)
def test_a_reynolds_number_written_in_the_case_is_found(animate, text, expected):
    assert animate.find_declared_reynolds(text) == expected


def test_a_declared_reynolds_number_wins_over_arithmetic(tmp_path, animate):
    """A case set up from a benchmark carries the number it was set up to match,
    and that is the number the picture should say -- not one recomputed from a
    length scale this script guessed at."""
    case = tmp_path / "cylinder"
    (case / "constant").mkdir(parents=True)
    (case / "constant" / "transportProperties").write_text(
        "// Re = 200\nnu              [0 2 -1 0 0 0 0] 1e-05;\n"
    )
    (case / "0").mkdir()
    (case / "0" / "U").write_text("internalField uniform (10 0 0);")

    assert animate.reynolds_from_case(case, length=1.0) == 200.0


def test_reynolds_is_computed_from_nu_and_the_inlet_when_a_length_is_given(tmp_path, animate):
    case = tmp_path / "cylinder"
    (case / "constant").mkdir(parents=True)
    (case / "constant" / "transportProperties").write_text("nu  [0 2 -1 0 0 0 0] 1e-05;\n")
    (case / "0").mkdir()
    (case / "0" / "U").write_text(
        "internalField uniform (0 0 0);\ninlet { value uniform (1 0 0); }\n"
    )

    assert animate.reynolds_from_case(case, length=0.1) == pytest.approx(10000.0)


def test_without_a_length_scale_nothing_is_invented(tmp_path, animate):
    case = tmp_path / "cylinder"
    (case / "constant").mkdir(parents=True)
    (case / "constant" / "transportProperties").write_text("nu  1e-05;\n")

    assert animate.reynolds_from_case(case) is None
    assert animate.reynolds_from_case(tmp_path / "no_such_case", length=1.0) is None


# -- the whole run, with the drawing stubbed out -------------------------------


class FakeReader:
    """Stands in for pyvista's OpenFOAMReader: a list of times and a mesh per time."""

    def __init__(self, times):
        self.time_values = list(times)
        self.time = self.time_values[0]

    def set_active_time_value(self, value):
        self.time = float(value)

    def read(self):
        return FakeMesh({"U": velocity(), "p": np.linspace(-1.0, 1.0, 4)})


class FakeSlice:
    bounds = (0.0, 2.0, 0.0, 1.0, 0.0, 0.0)


@pytest.fixture
def run(tmp_path, animate, monkeypatch):
    """`main` with everything that needs VTK replaced, and a record of what it drew."""
    case = tmp_path / "cylinder"
    (case / "constant").mkdir(parents=True)
    drawn: list[str] = []

    reader = FakeReader([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(animate, "open_reader", lambda _case: reader)
    monkeypatch.setattr(animate, "internal_mesh", lambda block: block)
    monkeypatch.setattr(animate, "slice_at", lambda mesh, normal: FakeSlice())

    def draw(mesh, scalar, out, **kwargs):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"png-bytes")
        drawn.append(Path(out).name)
        return Path(out)

    monkeypatch.setattr(animate, "render_frame", draw)

    def go(*flags):
        drawn.clear()
        animate.main([str(case), *flags])
        return list(drawn)

    go.case = case
    go.reader = reader
    go.frames = tmp_path / "cylinder_frames"
    return go


def test_a_first_run_draws_every_frame_and_leaves_a_sidecar(run):
    assert run("--field", "pressure") == [f"frame_{i:04d}.png" for i in range(4)]

    data = json.loads((run.frames / "frames.json").read_text())
    assert (data["complete"], data["expected"], data["partial"]) == (4, 4, False)
    assert data["scalar"] == "p" and data["pattern"] == "frame_%04d.png"


def test_a_second_run_draws_nothing_and_keeps_the_recorded_limits(run):
    run("--field", "pressure")
    first = json.loads((run.frames / "frames.json").read_text())["clim"]

    assert run("--field", "pressure") == []
    assert json.loads((run.frames / "frames.json").read_text())["clim"] == first


def test_write_times_that_appeared_since_the_last_run_are_the_only_new_frames(run):
    run("--field", "pressure")
    run.reader.time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    assert run("--field", "pressure") == ["frame_0004.png", "frame_0005.png"]


def test_changing_the_field_redraws_instead_of_relabelling_what_is_there(run):
    """The regression this guards: the frames directory is named after the case,
    so a second run with a different field found four finished PNGs, drew nothing,
    and rewrote the sidecar to claim a field the pictures were not of."""
    run("--field", "pressure")

    assert run("--field", "velocity") == [f"frame_{i:04d}.png" for i in range(4)]

    data = json.loads((run.frames / "frames.json").read_text())
    assert (data["field"], data["scalar"], data["cmap"]) == ("U", "U_mag", "viridis")
    # and the pressure run's limits are not carried onto a speed of 5 m/s
    assert data["clim"][0] > 1.0


def test_a_narrower_window_does_not_encode_the_frames_it_dropped(run):
    """`frame_%04d.png` walks the numbers on disk, so the two frames the first run
    left beyond the new window would be spliced onto the end of the animation."""
    run("--field", "pressure")

    run("--field", "pressure", "--to", "1.0")

    data = json.loads((run.frames / "frames.json").read_text())
    assert data["frames"] == ["frame_0000.png", "frame_0001.png"]
    assert data["stray"] == ["frame_0002.png", "frame_0003.png"]
    assert data["pattern"] is None


def test_force_redraws_a_finished_sequence(run):
    run("--field", "pressure")
    assert run("--field", "pressure", "--force") == [f"frame_{i:04d}.png" for i in range(4)]


def test_a_sequence_too_short_to_animate_says_so(run):
    run.reader.time_values = [0.0]
    with pytest.raises(SystemExit) as caught:
        run("--field", "pressure")
    assert "2 write times" in str(caught.value)


def test_explicit_limits_without_a_range_is_refused(run):
    with pytest.raises(SystemExit) as caught:
        run("--clim-from", "explicit")
    assert "--clim" in str(caught.value)


def test_the_finished_run_registers_the_frames_directory(run):
    import importlib.util

    run("--field", "pressure")
    spec = importlib.util.spec_from_file_location("state_check", TOOLBOX / "study_state.py")
    state = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state)

    rows = state.artifacts(root=run.case, kind="animation")
    assert rows and rows[-1]["meta"]["frames"] == 4
    assert state.phase_status("animate", root=run.case) == "done"


# -- the file itself -----------------------------------------------------------


def test_pyvista_is_not_imported_until_something_is_drawn(animate):
    """This test file runs on a machine with no VTK on it. A top-level `import
    pyvista` would put every function above out of reach of a test."""
    tree = ast.parse((TOOLBOX / "animate.py").read_text(encoding="utf-8"))
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        assert not any(name.split(".")[0] == "pyvista" for name in names)
    assert "import pyvista" in (TOOLBOX / "animate.py").read_text(encoding="utf-8")


def test_the_docstring_says_where_the_encoding_happens(animate):
    """The container has no encoder at all, and a frames directory that does not
    explain itself gets treated as a failed animation."""
    doc = " ".join(animate.__doc__.lower().split())
    assert "user's machine" in doc
    assert "python3 animate.py" in doc


def test_every_flag_the_old_command_line_had_still_parses(animate):
    """The scripts are edited in place by whoever is using them; a flag that
    quietly stopped existing would break a command someone already had."""
    args = animate.build_parser().parse_args(
        ["/work/case", "--field", "U", "--component", "mag", "--normal", "y",
         "--every", "2", "--from", "2.0", "--clim", "-5", "5", "--out", "/work/f"]
    )
    assert args.field == "U"
    assert args.component == "mag"
    assert args.every == 2
    assert args.start == 2.0
    assert args.clim == [-5.0, 5.0]
    assert str(args.out).endswith("f")


def test_the_new_flags_have_the_documented_defaults(animate):
    args = animate.build_parser().parse_args(["/work/case"])
    assert args.container == "gif"
    assert args.fps == 10.0
    assert args.clim_from is None  # meaning "last", or "explicit" when --clim is given
    assert args.streamlines == 0
    assert args.force is False

    with_streamlines = animate.build_parser().parse_args(["/work/case", "--streamlines"])
    assert with_streamlines.streamlines == animate.DEFAULT_SEEDS
