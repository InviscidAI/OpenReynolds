#!/usr/bin/env python3
"""Turn a `*_frames/` directory into a gif, mp4 or webp -- on the instance.

`animate.py` and `showcase.py` render the frames and write a `frames.json` sidecar, and
leave the encoding for the user's machine. That was right when the image carried no
encoder; it carries one now (imageio with the ffmpeg plugin), and in the hosted app there
is no user machine to encode on, so a session that wants the finished animation there has
had to hand-write imageio itself, 413 times over. This reads the sidecar and does exactly
that, once.

    python3 encode.py wake_frames                  # -> wake.gif at the sidecar's fps
    python3 encode.py wake_frames --format mp4 --fps 24
    python3 encode.py wake_frames --out /work/case/wake.gif

The frame order and fps come from `frames.json` when it is there; without one, the frames
are taken in numeric order (`frame_2` before `frame_10`) and a gif at 10 fps is written.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONTAINERS = {"gif": ".gif", "mp4": ".mp4", "webp": ".webp"}


def _frame_index(name: str) -> tuple:
    """Numeric-aware sort key so `frame_10.png` sorts after `frame_2.png`."""
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name))


def plan_encode(frames_dir: Path, fmt: str | None = None, fps: float | None = None,
                out: Path | None = None) -> dict:
    """Everything the encode needs, resolved from the sidecar and the arguments.

    Pure: no imageio, no writing -- so the resolution (which frames, in what order, to
    what file, at what rate) is testable off the container. The sidecar wins for order
    and defaults; explicit arguments win over the sidecar.
    """
    sidecar = frames_dir / "frames.json"
    meta: dict = {}
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            meta = {}

    container = (fmt or meta.get("container") or "gif").lower()
    if container not in CONTAINERS:
        raise SystemExit(f"unknown format {container!r}; one of {', '.join(sorted(CONTAINERS))}")
    ext = CONTAINERS[container]

    names = meta.get("frames") or [p.name for p in frames_dir.glob("frame_*.png")]
    names = sorted(names, key=_frame_index) if not meta.get("frames") else list(names)
    frames = [frames_dir / n for n in names]

    rate = float(fps or meta.get("fps") or 10.0)
    if out is not None:
        out_path = out
    else:
        stem = meta.get("output") or (frames_dir.name.replace("_frames", "") + ext)
        # A sidecar `output` already carries an extension; a derived stem carries ext.
        out_path = frames_dir.parent / stem
        if out_path.suffix != ext:
            out_path = out_path.with_suffix(ext)
    return {
        "container": container,
        "fps": rate,
        "frames": frames,
        "out": out_path,
        "loop": int(meta.get("loop", 0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames_dir", type=Path, help="a *_frames/ directory")
    parser.add_argument("--format", dest="fmt", default=None, choices=sorted(CONTAINERS))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None, help="output path (default beside the frames)")
    args = parser.parse_args(argv)

    if not args.frames_dir.is_dir():
        raise SystemExit(f"not a directory: {args.frames_dir}")
    plan = plan_encode(args.frames_dir, args.fmt, args.fps, args.out)
    if not plan["frames"]:
        raise SystemExit(f"no frames in {args.frames_dir} (expected frame_*.png)")

    import imageio.v2 as iio

    images = [iio.imread(f) for f in plan["frames"] if f.exists()]
    if not images:
        raise SystemExit(f"the frames named in the sidecar are not on disk under {args.frames_dir}")
    plan["out"].parent.mkdir(parents=True, exist_ok=True)
    if plan["container"] == "gif":
        iio.mimwrite(plan["out"], images, fps=plan["fps"], loop=plan["loop"])
    else:
        iio.mimwrite(plan["out"], images, fps=plan["fps"])
    print(f"{plan['out']}  ({len(images)} frames, {plan['container']}, {plan['fps']:g} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
