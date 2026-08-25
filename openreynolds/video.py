"""Assembling rendered frames into a video, on this machine.

Stills render on the instance, next to the data: a field render reads hundreds of
megabytes of mesh and fields to make a 100 KB picture, so moving the data to the
renderer is the wrong direction. Encoding is the opposite case -- a video needs no
case data at all, only the frames, and those are already mirrored home. So the
instance image ships no encoder, and this module uses whatever the user's machine
has: real ffmpeg when it is on PATH, imageio when only the library is around, and
an error that says what to install when it is neither.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FRAME_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

DEFAULT_FPS = 10.0


class VideoError(Exception):
    """Anything that stops a video being made. The message is the whole story."""


def frames_in(directory: Path) -> list[Path]:
    """Image files directly inside one directory, in name order.

    Name order is the order a render loop writes -- frame_0001, frame_0002 -- and
    is the only ordering the frames themselves can testify to."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES
    )


def best_frame_dir(root: Path) -> Path | None:
    """The directory under `root` holding the most frames, or None below two.

    One image is a picture, not a film; asking for a video of it deserves the
    honest answer that there is nothing to assemble."""
    best: Path | None = None
    most = 1
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for directory in candidates:
        count = len(frames_in(directory))
        if count > most:
            best, most = directory, count
    return best


def encoder() -> str | None:
    """Which encoder this machine offers: 'ffmpeg', 'imageio', or None."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio  # noqa: F401
    except ImportError:
        return None
    return "imageio"


def assemble(
    frames: list[Path],
    out: Path,
    fps: float = DEFAULT_FPS,
    run=subprocess.run,
) -> str:
    """Encode `frames` into `out`. Returns the name of the tool that did it."""
    if len(frames) < 2:
        raise VideoError(f"need at least 2 frames to make a video; found {len(frames)}")
    tool = encoder()
    if tool is None:
        raise VideoError(
            "no encoder on this machine: install ffmpeg (winget install ffmpeg), "
            "or `pip install imageio imageio-ffmpeg`"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    if tool == "ffmpeg":
        _ffmpeg(frames, out, fps, run)
    else:
        _imageio(frames, out, fps)
    return tool


def _ffmpeg(frames: list[Path], out: Path, fps: float, run) -> None:
    """Encode via the concat demuxer, which takes any file names.

    A `-i frame_%04d.png` pattern would silently drop frames the moment a render
    loop numbered them differently; a listing names each file, so what goes into
    the film is exactly what was found. The pad filter keeps yuv420p happy with
    odd-sized renders, and yuv420p is what every player actually plays."""
    listing = out.with_suffix(out.suffix + ".frames.txt")
    step = 1.0 / max(fps, 0.001)
    lines = []
    for frame in frames:
        lines.append(f"file '{frame.as_posix()}'")
        lines.append(f"duration {step:.6f}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    result = run(cmd, capture_output=True, text=True)
    listing.unlink(missing_ok=True)
    if getattr(result, "returncode", 1) != 0:
        tail = (getattr(result, "stderr", "") or "")[-400:]
        raise VideoError(f"ffmpeg failed (rc={result.returncode}): {tail}")


def _imageio(frames: list[Path], out: Path, fps: float) -> None:
    import imageio

    try:
        with imageio.get_writer(str(out), fps=fps) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(str(frame)))
    except Exception as exc:  # noqa: BLE001 - the message is the interface
        raise VideoError(f"imageio could not encode: {exc}") from exc
