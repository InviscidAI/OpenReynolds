"""Inline images are a convenience; the fallback is the path, so nothing may raise."""

from __future__ import annotations

import base64
import io

import pytest

from openreynolds import images
from openreynolds.images import MAX_INLINE_BYTES, protocol, show

PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    b"YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"TERM": "xterm-kitty"}, "kitty"),
        ({"KITTY_WINDOW_ID": "1"}, "kitty"),
        ({"TERM_PROGRAM": "iTerm.app"}, "iterm2"),
        ({"TERM_PROGRAM": "WezTerm"}, "iterm2"),
        ({"LC_TERMINAL": "iTerm2"}, "iterm2"),
        ({"TERM": "xterm-256color"}, None),
        ({"WT_SESSION": "abc"}, None),
        ({}, None),
    ],
)
def test_protocol_detection(env, expected):
    assert protocol(env) == expected


def test_detection_can_be_turned_off():
    assert protocol({"TERM": "xterm-kitty", "OPENREYNOLDS_INLINE_IMAGES": "off"}) is None


def png(tmp_path, name="plot.png", data=PNG):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_iterm2_sequence(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv("TERM", raising=False)
    out = io.StringIO()

    assert show(png(tmp_path), stream=out) is True

    written = out.getvalue()
    assert written.startswith("\033]1337;File=")
    assert "inline=1" in written
    assert base64.b64encode(PNG).decode() in written
    assert written.endswith("\a\n")


def test_kitty_chunks_the_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    big = PNG + b"\x00" * 20_000
    out = io.StringIO()

    assert show(png(tmp_path, data=big), stream=out) is True

    written = out.getvalue()
    assert written.count("\033_G") > 1, "a large image is sent in chunks"
    assert "a=T,f=100,m=1;" in written, "the first chunk opens the transfer"
    assert "m=0;" in written, "the last chunk closes it"


def test_a_plain_terminal_draws_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
    monkeypatch.delenv("LC_TERMINAL", raising=False)
    out = io.StringIO()
    assert show(png(tmp_path), stream=out) is False
    assert out.getvalue() == ""


def test_non_images_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-kitty")
    report = tmp_path / "report.md"
    report.write_text("# results")
    assert show(report, stream=io.StringIO()) is False


def test_a_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-kitty")
    assert show(tmp_path / "absent.png", stream=io.StringIO()) is False


def test_an_oversized_image_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-kitty")
    huge = png(tmp_path, data=b"\x89PNG" + b"\x00" * MAX_INLINE_BYTES)
    assert show(huge, stream=io.StringIO()) is False


def test_a_broken_stream_does_not_propagate(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-kitty")

    class Broken(io.StringIO):
        def write(self, _data):
            raise OSError("pipe closed")

    assert show(png(tmp_path), stream=Broken()) is False


# -- what travels is not always what is on disk --------------------------------


def test_a_large_render_is_scaled_for_transport_only():
    """Renders come out at 900x700 to 1600x500 and cost 840-1,333 tokens each, and a
    study reads the same path several times while it fixes the framing. A mesh at
    1024 px still shows near-wall spacing and a refinement band."""
    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2048, 1536), "white").save(buffer, format="PNG")
    original = buffer.getvalue()

    smaller = images.downscale(original, "image/png")
    assert max(images.dimensions(smaller)) == images.ATTACH_MAX_EDGE
    assert len(smaller) < len(original)


def test_a_picture_already_small_enough_is_left_exactly_as_it_is():
    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(buffer, format="PNG")
    original = buffer.getvalue()
    assert images.downscale(original, "image/png") is original


def test_an_animation_is_never_flattened():
    """Scaling a gif through a still encoder loses the thing it is of."""
    assert images.downscale(b"GIF89a-not-really", "image/gif") == b"GIF89a-not-really"


def test_an_unreadable_picture_travels_whole_rather_than_not_at_all():
    """A picture that arrives larger than necessary is a cost; one that arrives damaged
    is a wrong answer."""
    assert images.downscale(b"not an image at all", "image/png") == b"not an image at all"
