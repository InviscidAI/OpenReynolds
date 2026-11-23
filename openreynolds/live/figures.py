"""Pictures of a lattice: a velocity slice, a profile against its reference, an animation.

Every case in `cases.py` draws one, because this repository's rule is that a study
without a figure is a study nobody can check. `openreynolds/toolbox/claims.py` and
`check_copy.py` enforce that for the OpenFOAM path; there is nothing to enforce it here
yet, so `cases.py` simply always calls one of these.

## Why this does not reuse `toolbox/render.py` or `toolbox/encode.py`

Those are the right tools for an OpenFOAM case and the wrong shape for this one. They
are **standalone scripts**, copied verbatim into a workspace and run as
`python3 /work/.toolbox/encode.py`; `tests/test_toolbox.py` asserts that each imports
only the standard library, a closed set of third-party modules and its siblings,
precisely so that they keep working there. Importing one from this package would make
the dependency run the wrong way (a package importing a script that must not import the
package), and `render.py` reads VTK through pyvista, which is a mesh reader looking for
a mesh. A lattice is a numpy array on a uniform grid; `imshow` is the whole renderer.

What is borrowed is the convention rather than the code: one figure, two panels, the
field on the left and the thing being graded on the right, which is the layout
`toolbox/lbm.py::figure` already uses for the same reason -- the picture shows you the
flow is the flow it should be, and the panel beside it is the number.

## Both themes, and why the colour map is not `jet`

`viridis` for speed and `RdBu_r` for a signed component, matching `toolbox/lbm.py`. Not
a style preference: `jet` has luminance reversals that invent edges in a smooth field
and read as structure that is not there, which on a velocity slice is exactly the kind
of mistake a figure exists to prevent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["slice_figure", "profile_figure", "animate", "frame_count", "AXES"]

AXES = ("x", "y", "z")


def _matplotlib():
    """matplotlib with the Agg backend, or a refusal that names the install.

    Agg before `pyplot`, and not negotiable: a case run from the agent tool or from a
    headless CI job has no display, and importing `pyplot` first binds whatever backend
    the machine's rc file names. The failure that produces is a hang rather than an
    error, which is the worst kind.
    """
    try:
        import matplotlib
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "drawing a figure needs matplotlib, which the [live] extra declares:\n"
            f"    pip install 'openreynolds[live]'\n(the import said: {exc})"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plane(field: np.ndarray, axis: int, index: int | None) -> tuple[np.ndarray, int]:
    """One slice of a 3-D array, and which index it came from (the middle by default)."""
    if index is None:
        index = field.shape[axis] // 2
    index = int(np.clip(index, 0, field.shape[axis] - 1))
    return np.take(field, index, axis=axis), index


def slice_figure(
    world: Any,
    path: str | Path,
    *,
    axis: int = 2,
    index: int | None = None,
    title: str = "",
    component: int | None = None,
    bodies: bool = True,
    streamlines: bool = True,
) -> Path:
    """A velocity slice through the lattice, with the solid masked and the bodies on it.

    `axis` is the axis the slice is normal to, so the default 2 is the x-y plane -- the
    one every quasi-two-dimensional case here is set up in. `component` draws one signed
    velocity component on a diverging map instead of the speed, which is what a
    downwash or a wake deficit wants.

    The solid is drawn from `phi < 0` rather than from the geometry, which is the point:
    what this figure shows is the body **the solver sees**, staircases and all. A figure
    drawn from the geometry instead would hide exactly the resolution problem the
    picture is there to reveal.
    """
    plt = _matplotlib()
    got = world.fields()
    u = got["u"]
    solid = got["solid"]
    speed = (np.linalg.norm(u, axis=-1) if component is None else u[..., component])

    plane, index = _plane(speed, axis, index)
    solid_plane, _ = _plane(solid, axis, index)
    kept = [a for a in range(3) if a != axis]
    coords = [world.node_coordinates(a) for a in kept]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    finite = plane[np.isfinite(plane)]
    if component is None:
        limits = {"cmap": "viridis", "vmin": 0.0,
                  "vmax": float(np.nanmax(finite)) if finite.size else 1.0}
    else:
        span = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        limits = {"cmap": "RdBu_r", "vmin": -span, "vmax": span}
    image = ax.imshow(
        plane.T, origin="lower", aspect="equal",
        extent=[coords[0][0], coords[0][-1], coords[1][0], coords[1][-1]],
        **limits,
    )
    if bodies and solid_plane.any():
        # One contour at the surface, from the same field the boundary condition reads.
        ax.contour(coords[0], coords[1], solid_plane.T.astype(float), levels=[0.5],
                   colors="black", linewidths=1.1)
    if streamlines and not component:
        try:
            a, b = (u[..., kept[0]], u[..., kept[1]])
            pa, _ = _plane(a, axis, index)
            pb, _ = _plane(b, axis, index)
            ax.streamplot(coords[0], coords[1], np.nan_to_num(pa).T,
                          np.nan_to_num(pb).T, color="white", linewidth=0.5,
                          density=1.1, arrowsize=0.6)
        except Exception:
            # A streamplot wants a regular grid and a field with no all-nan row; a case
            # whose body spans the domain can produce one. The slice is the figure and
            # the streamlines are a garnish, so a failure here does not lose the picture.
            pass

    label = "|u| (m/s)" if component is None else f"u_{AXES[component]} (m/s)"
    fig.colorbar(image, ax=ax, label=label, shrink=0.85)
    ax.set_xlabel(f"{AXES[kept[0]]} (m)")
    ax.set_ylabel(f"{AXES[kept[1]]} (m)")
    ax.set_title(title or f"{AXES[axis]} = {world.node_coordinates(axis)[index]:.4g} m",
                 fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def profile_figure(
    path: str | Path,
    series: Sequence[dict[str, Any]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    panels: int = 1,
) -> Path:
    """The graded quantity against its reference, one or two panels.

    `series` is a list of `{"x", "y", "label", "style", "panel"}`. A reference goes in
    with `style="o"` so that a published table reads as points and a solved curve as a
    line -- the distinction matters, because a reader needs to see at a glance which of
    the two is the measurement.
    """
    plt = _matplotlib()
    fig, axes = plt.subplots(1, max(1, panels), figsize=(5.4 * max(1, panels), 4.4),
                             squeeze=False)
    for entry in series:
        ax = axes[0][min(int(entry.get("panel", 0)), panels - 1)]
        style = entry.get("style", "-")
        ax.plot(entry["x"], entry["y"], style, label=entry.get("label", ""),
                ms=4, lw=1.4)
    for ax in axes[0]:
        ax.grid(alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if any(line.get_label() and not line.get_label().startswith("_")
               for line in ax.lines):
            ax.legend(fontsize=8)
    axes[0][0].set_title(title, fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def animate(frames: Sequence[np.ndarray], path: str | Path, *, fps: int = 20,
            extent: Sequence[float] | None = None, component: bool = False,
            title: str = "") -> Path | None:
    """A GIF of a sequence of slices. Returns the path, or None if nothing could write one.

    `imageio` is what writes it, and it is in the `[live]` extra for this. It returns
    None rather than raising when the encoder is missing, because a demo that ran is
    still a demo that ran: the measurement is the result and the animation is how it is
    shown. `cases.py` reports which of the two happened.

    The colour scale is fixed across the whole sequence from the first few frames rather
    than per frame. A per-frame scale makes a decaying flow look like it is not decaying,
    which is the one thing an animation of a transient must not do.

    ## Why this raises instead of writing whatever it was handed

    A one-frame GIF looks like a GIF to every tool except a viewer, and this repository
    has already paid for that: `PROGRESS_LIVE.md` claimed three demos animated while the
    files on disk held exactly one frame each (the cause was `ensure_rgb`, which flattens
    mode `"P"`). So the three ways an animation can come out dead are checked rather
    than hoped for -- too few frames in, every frame identical, and the written file
    collapsing anyway, which is what Pillow does to identical consecutive frames and to
    frames that quantise onto the same 256-colour palette. A missing encoder is still
    `None`, because that is an absent tool and not a dead picture.
    """
    frames = [np.asarray(frame, dtype=np.float64) for frame in frames]
    if len(frames) < 2:
        raise ValueError(
            f"an animation needs at least 2 frames and was given {len(frames)}; "
            f"a one-frame GIF is a still with the wrong extension. Check that the "
            f"case's sampling callback is appending frames at all."
        )
    if all(np.array_equal(frame, frames[0], equal_nan=True) for frame in frames[1:]):
        raise ValueError(
            f"all {len(frames)} frames are identical, so the GIF would show nothing "
            f"moving; the sample callback is probably reading a cached or aliased field."
        )
    try:
        import imageio.v3 as iio
    except Exception:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except Exception:
            return None

    plt = _matplotlib()
    from matplotlib import cm, colors

    sample = np.concatenate([f[np.isfinite(f)].ravel() for f in frames[:5]])
    if component:
        span = float(np.nanmax(np.abs(sample))) if sample.size else 1.0
        norm = colors.Normalize(-span, span)
        table = cm.get_cmap("RdBu_r") if hasattr(cm, "get_cmap") else plt.get_cmap("RdBu_r")
    else:
        high = float(np.nanmax(sample)) if sample.size else 1.0
        norm = colors.Normalize(0.0, high if high > 0 else 1.0)
        table = cm.get_cmap("viridis") if hasattr(cm, "get_cmap") else plt.get_cmap("viridis")

    images = []
    for frame in frames:
        rgba = table(norm(np.nan_to_num(frame.T[::-1])))
        images.append((rgba[..., :3] * 255).astype(np.uint8))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(out, images, duration=max(1, int(1000 / max(1, fps))), loop=0)
    except Exception:
        try:
            iio.mimwrite(out, images, fps=fps)  # older imageio
        except Exception:
            return None
    written = frame_count(out)
    if written is not None and written < 2:
        raise ValueError(
            f"{out.name}: {len(frames)} frames went in and the written file has "
            f"{written}. The encoder collapsed them -- after palettisation the frames "
            f"are indistinguishable, so the run has nothing to animate."
        )
    del extent, title
    return out


def frame_count(path: str | Path) -> int | None:
    """How many frames a written image actually has, or None if PIL cannot say.

    The only measurement that settles whether a GIF animates. `animate` calls it on its
    own output and `benchmarks/live/gallery.py` calls it on everything in the gallery,
    because "the writer was handed 200 frames" and "the file has 200 frames" turned out
    to be different statements.
    """
    try:
        from PIL import Image

        with Image.open(Path(path)) as image:
            return int(getattr(image, "n_frames", 1))
    except Exception:
        return None


def slice_frame(world: Any, axis: int = 2, index: int | None = None,
                component: int | None = None) -> np.ndarray:
    """One frame for `animate`: the same field `slice_figure` draws, as a bare array.

    Separate from `slice_figure` because an animation wants two hundred of these and
    cannot afford a matplotlib figure for each. A `fields()` call is a host read of the
    whole lattice, so a caller sampling every step of a long run is paying for it; every
    case here samples every tenth coupling step or sparser.
    """
    got = world.fields()
    u = got["u"]
    field = np.linalg.norm(u, axis=-1) if component is None else u[..., component]
    plane, _ = _plane(field, axis, index)
    return plane


def ensure_rgb(path: str | Path) -> Path:
    """Flatten a figure to RGB, because an RGBA thumbnail renders as a black panel.

    Learned on this project's own site rather than from a library's documentation: a
    saved PNG with an alpha channel comes back as a black rectangle wherever the
    thumbnailer composites it against nothing. matplotlib's `savefig` writes RGBA by
    default. Called by `cases.py` on every figure it writes.

    **It must not touch an animation, and that is not a detail.** Every GIF is mode
    `"P"`, so the conversion below matched it, kept frame 0, and wrote the file back as a
    one-frame GIF. `cases.py` calls this on everything in `drawn`, and `drawn` contains
    the GIFs -- so the cylinder, the falling sphere and the quadrotor each animated
    correctly inside `figures.animate`, were destroyed one line later, and
    `PROGRESS_LIVE.md` claimed they animated for weeks. The dam break survived only
    because its caller happens to guard with `endswith(".png")`. The guard belongs here,
    where every caller routes through it, not in each caller.
    """
    out = Path(path)
    try:
        from PIL import Image
    except Exception:
        return out
    try:
        with Image.open(out) as image:
            if getattr(image, "n_frames", 1) > 1:
                return out
            if image.mode in ("RGBA", "LA", "P"):
                image.convert("RGB").save(out)
    except Exception:
        pass
    return out


def clock(seconds: float) -> str:
    """`1 h 04 m` or `3.2 s`, for a progress line that has to be read at a glance."""
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{int(seconds // 60)} m {int(seconds % 60):02d} s"
    return f"{int(seconds // 3600)} h {int((seconds % 3600) // 60):02d} m"


def strouhal(times: Sequence[float], signal: Sequence[float], length: float,
             speed: float) -> dict[str, Any]:
    """The Strouhal number of an oscillating signal, by zero crossings and by FFT.

    Both, and reported together, because they fail differently and a case is only
    trustworthy when they agree. Zero crossings are exact for a clean limit cycle and
    are fooled by a signal that has not finished its transient; an FFT peak is robust to
    the transient and is quantised by the record length, so a short record gives a
    confident wrong digit. `benchmarks/moving_mesh/grade.py` counts cycles for the same
    quantity and the same reason, and reports how many it had.

    `signal` should be the lift history with its mean removed; the mean is removed here
    anyway, because a lift signal with a small offset crosses zero at the wrong places
    and that is the commonest way this measurement goes wrong.
    """
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    if t.size < 8 or not np.isfinite(y).all():
        return {"strouhal": float("nan"), "cycles": 0.0, "method": "too short"}
    y = y - y.mean()

    crossings = np.flatnonzero(np.sign(y[:-1]) != np.sign(y[1:]))
    by_crossings = float("nan")
    cycles = 0.0
    if crossings.size >= 3:
        # Linear interpolation of each crossing time, then the mean half-period.
        exact = []
        for index in crossings:
            y0, y1 = y[index], y[index + 1]
            if y1 == y0:
                continue
            exact.append(t[index] + (t[index + 1] - t[index]) * (-y0 / (y1 - y0)))
        if len(exact) >= 3:
            half = np.diff(exact)
            period = 2.0 * float(half.mean())
            cycles = (exact[-1] - exact[0]) / period if period > 0 else 0.0
            by_crossings = length / (period * speed) if period > 0 else float("nan")

    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=float(np.mean(np.diff(t))))
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    by_fft = float(freqs[peak] * length / speed) if speed else float("nan")

    chosen = by_crossings if math.isfinite(by_crossings) else by_fft
    return {
        "strouhal": float(chosen),
        "strouhal_crossings": float(by_crossings),
        "strouhal_fft": float(by_fft),
        "cycles": float(cycles),
        "method": "zero crossings" if math.isfinite(by_crossings) else "fft peak",
        "frequency": float(chosen * speed / length) if length else float("nan"),
    }
