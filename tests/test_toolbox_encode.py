"""encode.py resolves what to encode off the sidecar without touching imageio."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(dirpath: Path, names):
    dirpath.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dirpath / n).write_bytes(b"png")
    return dirpath


def test_plan_reads_frame_order_and_rate_from_the_sidecar(tmp_path):
    encode = load("encode")
    d = _frames(tmp_path / "wake_frames", ["frame_1.png", "frame_2.png"])
    (d / "frames.json").write_text(json.dumps({
        "container": "gif", "fps": 24.0, "output": "wake.gif",
        "frames": ["frame_1.png", "frame_2.png"], "loop": 0,
    }))
    plan = encode.plan_encode(d)
    assert plan["container"] == "gif"
    assert plan["fps"] == 24.0
    assert [p.name for p in plan["frames"]] == ["frame_1.png", "frame_2.png"]
    assert plan["out"].name == "wake.gif"


def test_plan_falls_back_to_numeric_order_without_a_sidecar(tmp_path):
    encode = load("encode")
    # deliberately out of lexical order: frame_10 must come after frame_2, not before.
    d = _frames(tmp_path / "wake_frames", ["frame_10.png", "frame_2.png", "frame_1.png"])
    plan = encode.plan_encode(d)
    assert [p.name for p in plan["frames"]] == ["frame_1.png", "frame_2.png", "frame_10.png"]
    assert plan["out"].name == "wake.gif"  # derived from the dir name
    assert plan["fps"] == 10.0


def test_explicit_format_and_fps_win_over_the_sidecar(tmp_path):
    encode = load("encode")
    d = _frames(tmp_path / "wake_frames", ["frame_1.png"])
    (d / "frames.json").write_text(json.dumps({"container": "gif", "fps": 10.0}))
    plan = encode.plan_encode(d, fmt="mp4", fps=30.0)
    assert plan["container"] == "mp4"
    assert plan["out"].suffix == ".mp4"
    assert plan["fps"] == 30.0


def test_an_unknown_format_is_refused(tmp_path):
    encode = load("encode")
    d = _frames(tmp_path / "f_frames", ["frame_1.png"])
    try:
        encode.plan_encode(d, fmt="avi")
    except SystemExit:
        return
    raise AssertionError("unknown format should be refused")
