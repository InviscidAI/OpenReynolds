"""The two halves of the 2026-09-12 incident: a half-written figure ended two sessions.

Both runs died on `400 invalid_request_error: Could not process image`, immediately
after the agent redrew a matplotlib figure and read it back, 27 minutes into one study
and 2 h 23 m into the other. Two separate defects had to line up:

  1. `_read_image` attached bytes nobody had checked were a whole picture. The existing
     guard compares the read length against the stat size, which catches a transport
     that cut the read short and cannot catch a file that was genuinely that size
     because something was still writing it.
  2. One refused picture ended the session. A 400 is correctly not retryable by
     waiting, but the bad bytes live in a thread this process owns, so it is repairable
     by editing rather than by waiting.

These tests fail against the code as it stood that morning: the first one attached the
truncated PNG, the second one left the session blocked.
"""
from __future__ import annotations

import pytest

from openreynolds import images


PNG_HEAD = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"


def whole_png() -> bytes:
    return PNG_HEAD + b"\x00\x00\x00\rIHDR" + b"\x00" * 30 + PNG_IEND


# -- half one: the bytes are checked before they are attached ---------------------


def test_a_png_still_being_written_is_not_a_picture():
    """The exact shape of the incident: a valid header, real content, no IEND yet."""
    half = PNG_HEAD + b"\x00\x00\x00\rIHDR" + b"\x00" * 30
    why = images.incomplete(half, "image/png")
    assert why is not None
    assert "IEND" in why


def test_a_finished_png_passes():
    assert images.incomplete(whole_png(), "image/png") is None


def test_an_empty_file_is_not_a_picture():
    assert images.incomplete(b"", "image/png") == "the file is empty"


@pytest.mark.parametrize(
    "data, media",
    [
        (b"\xff\xd8" + b"\x00" * 40, "image/jpeg"),
        (b"GIF89a" + b"\x00" * 40, "image/gif"),
        (b"RIFF" + (900).to_bytes(4, "little") + b"WEBP" + b"\x00" * 20, "image/webp"),
    ],
)
def test_every_format_notices_a_missing_tail(data, media):
    assert images.incomplete(data, media) is not None


@pytest.mark.parametrize(
    "data, media",
    [
        (b"\xff\xd8" + b"\x00" * 40 + b"\xff\xd9", "image/jpeg"),
        (b"GIF89a" + b"\x00" * 40 + b"\x3b", "image/gif"),
        (b"RIFF" + (28).to_bytes(4, "little") + b"WEBP" + b"\x00" * 24, "image/webp"),
    ],
)
def test_every_format_accepts_a_finished_file(data, media):
    assert images.incomplete(data, media) is None


def test_the_reader_reports_a_partial_figure_instead_of_attaching_it(ctx):
    """The whole point: a half-written render comes back as words, not as a refusal.

    Words are the better answer even ignoring the API: the model is told the file is
    not ready and can simply look again, which is what a person would do.
    """
    from openreynolds.tools import dispatch

    ctx.backend.files["/work/fig.png"] = PNG_HEAD + b"\x00\x00\x00\rIHDR" + b"\x00" * 30
    content, _ = dispatch(ctx, "read_file", {"path": "/work/fig.png"})

    assert isinstance(content, str), "a partial image must not be attached"
    assert "not a whole image" in content
    assert "IEND" in content


def test_a_finished_figure_is_still_attached(ctx):
    from openreynolds.tools import dispatch

    ctx.backend.files["/work/fig.png"] = whole_png()
    content, _ = dispatch(ctx, "read_file", {"path": "/work/fig.png"})

    assert isinstance(content, list), "a complete image must still come back as a picture"
    assert any(b["type"] == "image" for b in content)


# -- half two: a refused picture costs the picture, not the run -------------------


class _Loop:
    """Just enough Loop to exercise drop_images."""

    def __init__(self, messages):
        self.messages = messages

    drop_images = None  # bound below


def _loop_with(messages):
    from openreynolds.loop import Loop

    loop = _Loop(messages)
    loop.drop_images = Loop.drop_images.__get__(loop, _Loop)
    return loop


def test_dropping_images_leaves_the_words_and_the_shape_of_the_thread():
    messages = [
        {"role": "user", "content": "look at the mesh"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "AAAA"}},
                {"type": "text", "text": "/work/fig.png - 1024x768 image/png, 40 bytes"},
            ]},
        ]},
    ]
    loop = _loop_with(messages)
    assert loop.drop_images() == 1

    blocks = messages[1]["content"][0]["content"]
    assert all(b["type"] == "text" for b in blocks), "no image blocks may survive"
    assert "refused it" in blocks[0]["text"]
    assert "/work/fig.png" in blocks[1]["text"], "the path must survive so it can be re-read"
    assert messages[0]["content"] == "look at the mesh", "plain text turns are untouched"


def test_dropping_images_when_there_are_none_says_so():
    loop = _loop_with([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert loop.drop_images() == 0, "the caller needs to tell 'nothing to fix' from 'fixed'"


def test_an_image_refusal_is_matched_by_its_words_not_its_code():
    from openreynolds.cli import _IMAGE_REFUSAL

    seen_in_production = (
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Could not process image'}, 'request_id': 'req_011CeyfVq63FKAoyrGfz2Sr6'}"
    )
    assert _IMAGE_REFUSAL.search(seen_in_production)
    assert not _IMAGE_REFUSAL.search("no budget: the account cannot pay for this call")
    assert not _IMAGE_REFUSAL.search("the key was not accepted")
