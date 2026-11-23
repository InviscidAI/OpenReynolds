"""The animation guards in `openreynolds/live/figures.py`.

Five assertions about one bug, which was worth a test file of its own because it put a
false claim into this repository's own progress document and kept it there.

`ensure_rgb` flattens an image to RGB so that an RGBA PNG does not render as a black
thumbnail. Every GIF is mode `"P"`, so it matched too: the conversion kept frame 0 and
wrote the file back as a one-frame GIF. `cases.py` calls `ensure_rgb` on everything in
`drawn`, and `drawn` holds the GIFs -- so the cylinder, the falling sphere and the
quadrotor each animated correctly inside `figures.animate` and were destroyed one line
later. `PROGRESS_LIVE.md` claimed they animated, every tool agreed the files were GIFs,
and only a viewer disagreed.

So: `ensure_rgb` must leave an animation alone, and `animate` must refuse to write a
picture that shows nothing moving rather than return a path for one.
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


def test_animate_refuses_frames_that_collapse_on_encoding(tmp_path):
    """Distinct arrays, one palette: the encoder writes a single frame and must say so."""
    base = np.random.default_rng(1).random((48, 32))
    nudged = [base + 1.0e-12 * index for index in range(20)]
    with pytest.raises(ValueError, match="collapsed"):
        figures.animate(nudged, tmp_path / "collapsed.gif")
