"""The animation guards in `openreynolds/live/figures.py`.

Six assertions about two bugs, both of which turned a real run into a picture of
nothing. The first was worth a test file of its own because it put a false claim into
this repository's own progress document and kept it there.

`ensure_rgb` flattens an image to RGB so that an RGBA PNG does not render as a black
thumbnail. Every GIF is mode `"P"`, so it matched too: the conversion kept frame 0 and
wrote the file back as a one-frame GIF. `cases.py` calls `ensure_rgb` on everything in
`drawn`, and `drawn` holds the GIFs -- so the cylinder, the falling sphere and the
quadrotor each animated correctly inside `figures.animate` and were destroyed one line
later. `PROGRESS_LIVE.md` claimed they animated, every tool agreed the files were GIFs,
and only a viewer disagreed.

So: `ensure_rgb` must leave an animation alone, and `animate` must refuse to write a
picture that shows nothing moving rather than return a path for one.

The second bug is the colour scale, and it is the same failure one layer along: a file
that animates and still shows nothing. `animate` took its scale from the first five
frames, which is right for a decaying flow -- the peak is in frame one -- and wrong for
every flow that starts from rest. The ground-effect jet takes 60% of its run to reach
the floor, so a scale set from its first five frames was set from almost nothing, and
the rest of the sequence clipped: 87% of the last frame one saturated colour, with the
clipping boundary as the only thing that moved. The scale now comes from every frame's
peak, which for a decaying flow is the same number it was before.
"""

from __future__ import annotations

import numpy as np
import pytest

from openreynolds.live import figures


def _frames(count: int, *, moving: bool = True) -> list[np.ndarray]:
    """A stack of distinguishable frames: a bright blob that walks across the field."""
    rng = np.random.default_rng(0)
    base = rng.random((48, 32))
    out = []
    for index in range(count):
        frame = base.copy()
        if moving:
            frame[index % 40: index % 40 + 6, 10:20] += 4.0
        out.append(frame)
    return out


def test_animate_writes_every_frame(tmp_path):
    path = figures.animate(_frames(20), tmp_path / "moving.gif", fps=20)
    if path is None:
        pytest.skip("no GIF encoder installed")
    assert figures.frame_count(path) == 20


def test_ensure_rgb_leaves_an_animation_alone(tmp_path):
    """The regression. Before the guard this came back as 1."""
    path = figures.animate(_frames(20), tmp_path / "moving.gif", fps=20)
    if path is None:
        pytest.skip("no GIF encoder installed")
    figures.ensure_rgb(path)
    assert figures.frame_count(path) == 20


def test_ensure_rgb_still_flattens_an_rgba_still(tmp_path):
    """The behaviour the guard must not have removed."""
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "still.png"
    Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8), mode="RGBA").save(path)
    figures.ensure_rgb(path)
    with Image.open(path) as got:
        assert got.mode == "RGB"


def test_animate_refuses_one_frame(tmp_path):
    with pytest.raises(ValueError, match="at least 2 frames"):
        figures.animate(_frames(1), tmp_path / "one.gif")


def test_animate_refuses_identical_frames(tmp_path):
    with pytest.raises(ValueError, match="identical"):
        figures.animate(_frames(20, moving=False), tmp_path / "flat.gif")


def test_animate_scales_from_every_frame_and_not_the_first_few(tmp_path):
    """A flow that starts from rest must not clip, which is a flat frame by another name.

    The ramp below is what an establishing jet looks like to the colour scale: the last
    frame peaks forty times higher than the fifth. Scaled from the first five frames,
    seven eighths of it lands on the top colour of the map; scaled from the sequence, it
    lands across the map and the structure is visible. The assertion is on the fraction
    of the last frame that is a single saturated colour, because that is what a reader
    of the GIF actually sees -- 0.875 before the fix.
    """
    Image = pytest.importorskip("PIL.Image")
    from PIL import ImageSequence

    gradient = np.linspace(0.0, 1.0, 32)[None, :] * np.ones((48, 1))
    frames = [gradient * (1.0 + index) for index in range(40)]
    path = figures.animate(frames, tmp_path / "ramp.gif", fps=10)
    if path is None:
        pytest.skip("no GIF encoder installed")
    with Image.open(path) as image:
        last = np.asarray(
            list(ImageSequence.Iterator(image))[-1].convert("RGB"), dtype=np.int16)
    brightest = last.reshape(-1, 3)[int(np.argmax(last.sum(axis=2)))]
    saturated = float(np.all(last == brightest, axis=2).mean())
    assert saturated < 0.1, (
        f"{saturated:.1%} of the last frame is one colour, so the sequence is clipping; "
        f"the scale is being taken from the quiet start of the run"
    )


def test_animate_is_not_washed_out_by_a_few_extreme_cells(tmp_path):
    """The other end of the same choice: one hot cell must not set the scale.

    Scaled from the largest value anywhere in the sequence, the quadrotor's downwash
    came back as pale pastel -- a few cells at the rim of an actuator disk carry several
    times the velocity of the flow the picture is of. The spike below is 50x the
    structure, which is the same shape of problem; the frame has to keep using most of
    the colour map in spite of it.
    """
    Image = pytest.importorskip("PIL.Image")
    from PIL import ImageSequence

    gradient = np.linspace(0.0, 1.0, 32)[None, :] * np.ones((48, 1))
    frames = [gradient * (1.0 + 0.02 * index) for index in range(40)]
    frames[10] = frames[10].copy()
    frames[10][0, 0] = 50.0
    path = figures.animate(frames, tmp_path / "spike.gif", fps=10)
    if path is None:
        pytest.skip("no GIF encoder installed")
    with Image.open(path) as image:
        last = np.asarray(
            list(ImageSequence.Iterator(image))[-1].convert("RGB"), dtype=np.int16)
    # The gradient has 32 steps, so 32 is every one of them resolved and a handful is
    # the whole field squeezed onto the dark end of the map.
    colours = len(np.unique(last.reshape(-1, 3), axis=0))
    assert colours > 16, (
        f"the last frame uses {colours} colours, so one outlying cell has flattened the "
        f"whole sequence onto the bottom of the colour map"
    )


def test_animate_refuses_frames_that_collapse_on_encoding(tmp_path):
    """Distinct arrays, one palette: the encoder writes a single frame and must say so."""
    base = np.random.default_rng(1).random((48, 32))
    nudged = [base + 1.0e-12 * index for index in range(20)]
    with pytest.raises(ValueError, match="collapsed"):
        figures.animate(nudged, tmp_path / "collapsed.gif")
