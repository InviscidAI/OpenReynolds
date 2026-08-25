"""Video assembly is local: frames come from the mirror, encoders from this machine."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openreynolds import video
from openreynolds.video import VideoError, assemble, best_frame_dir, encoder, frames_in

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 20


def frames(directory: Path, count: int, stem: str = "frame") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    for index in range(count):
        p = directory / f"{stem}_{index:04d}.png"
        p.write_bytes(PNG)
        made.append(p)
    return made


# -- finding the frames ----------------------------------------------------------


def test_frames_come_back_in_name_order(tmp_path):
    made = frames(tmp_path, 3)
    (tmp_path / "notes.md").write_text("not a frame")
    assert frames_in(tmp_path) == made


def test_a_missing_directory_is_no_frames(tmp_path):
    assert frames_in(tmp_path / "absent") == []


def test_the_biggest_frame_set_wins(tmp_path):
    frames(tmp_path / "renders" / "small", 2)
    frames(tmp_path / "renders" / "sweep", 5)
    assert best_frame_dir(tmp_path) == tmp_path / "renders" / "sweep"


def test_one_image_is_a_picture_not_a_film(tmp_path):
    """A singleton must not be offered as a video; there is nothing to assemble."""
    frames(tmp_path / "renders", 1)
    assert best_frame_dir(tmp_path) is None


# -- assembling ------------------------------------------------------------------


def test_too_few_frames_is_a_clear_error(tmp_path):
    with pytest.raises(VideoError, match="at least 2"):
        assemble(frames(tmp_path, 1), tmp_path / "out.mp4")


def test_no_encoder_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "encoder", lambda: None)
    with pytest.raises(VideoError, match="ffmpeg"):
        assemble(frames(tmp_path, 3), tmp_path / "out.mp4")


def test_ffmpeg_gets_every_frame_by_name_and_a_playable_pixel_format(tmp_path, monkeypatch):
    """A %04d input pattern silently drops frames named differently; the concat
    listing names each file, so the film holds exactly what was found."""
    monkeypatch.setattr(video, "encoder", lambda: "ffmpeg")
    made = frames(tmp_path, 3)
    seen = {}

    def fake_run(cmd, capture_output, text):
        seen["cmd"] = cmd
        listing = Path(cmd[cmd.index("-i") + 1])
        seen["listing"] = listing.read_text(encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    tool = assemble(made, tmp_path / "out.mp4", fps=20, run=fake_run)

    assert tool == "ffmpeg"
    assert "yuv420p" in seen["cmd"]
    for frame in made:
        assert frame.as_posix() in seen["listing"]
    assert "duration 0.050000" in seen["listing"]
    leftovers = list(tmp_path.glob("*.frames.txt"))
    assert leftovers == [], "the listing file is cleaned up"


def test_a_failed_encode_reports_ffmpegs_own_words(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "encoder", lambda: "ffmpeg")

    def failing(cmd, capture_output, text):
        return SimpleNamespace(returncode=1, stderr="Unknown decoder 'png'")

    with pytest.raises(VideoError, match="Unknown decoder"):
        assemble(frames(tmp_path, 2), tmp_path / "out.mp4", run=failing)


def test_encoder_prefers_the_real_ffmpeg(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: "C:/tools/ffmpeg.exe")
    assert encoder() == "ffmpeg"


# -- the command -----------------------------------------------------------------


@pytest.fixture
def as_the_only_study(monkeypatch, store):
    from openreynolds import cli
    from openreynolds.config import Config

    monkeypatch.setattr(
        Config, "load", classmethod(lambda cls: Config(studies_dir=store.dir.parent))
    )
    store.save()  # list_studies only counts a directory with a session.json in it
    return cli


def test_the_command_finds_frames_and_says_what_it_made(as_the_only_study, store, monkeypatch):
    from click.testing import CliRunner

    cli = as_the_only_study
    made = frames(store.fetch_dir() / "study-test" / "renders" / "sweep", 3)

    def fake_assemble(sequence, out, fps=10.0, run=None):
        assert sequence == made
        out.write_bytes(b"mp4")
        return "ffmpeg"

    monkeypatch.setattr(cli.video_mod, "assemble", fake_assemble)

    result = CliRunner().invoke(cli.main, ["video"])

    assert result.exit_code == 0, result.output
    assert "sweep.mp4" in result.output
    assert "3 frames" in result.output


def test_the_command_maps_a_workspace_path_into_the_mirror(as_the_only_study, store, monkeypatch):
    from click.testing import CliRunner

    cli = as_the_only_study
    frames(store.fetch_dir() / "s" / "case" / "renders", 2)
    asked = {}

    def fake_assemble(sequence, out, fps=10.0, run=None):
        asked["dirs"] = {p.parent for p in sequence}
        out.write_bytes(b"mp4")
        return "imageio"

    monkeypatch.setattr(cli.video_mod, "assemble", fake_assemble)

    result = CliRunner().invoke(cli.main, ["video", "/work/s/case/renders"])

    assert result.exit_code == 0, result.output
    assert asked["dirs"] == {store.fetch_dir() / "s" / "case" / "renders"}


def test_no_frames_anywhere_is_a_pointer_not_a_stack_trace(as_the_only_study, store):
    from click.testing import CliRunner

    result = CliRunner().invoke(as_the_only_study.main, ["video"])

    assert result.exit_code == 1
    assert "two or more" in result.output
