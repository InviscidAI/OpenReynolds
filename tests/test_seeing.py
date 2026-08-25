"""A render the model cannot look at is a file, not a picture.

`read_file` on an image has to come back as an image block, or the agent can only ever
read its own description of what it meant to draw.
"""

from __future__ import annotations

import base64
import json

from conftest import install_model, message, text_block, tool_block
from openreynolds.config import Config
from openreynolds.images import MAX_ATTACH_BYTES
from openreynolds.loop import Loop
from openreynolds.tools import describe, dispatch

PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    b"YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_a_png_comes_back_as_a_picture(ctx):
    ctx.backend.files["/work/case/renders/mesh.png"] = PNG

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/case/renders/mesh.png"})

    assert not is_error
    assert isinstance(content, list)
    image = next(b for b in content if b["type"] == "image")
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == PNG


def test_the_picture_arrives_with_a_line_saying_what_it_is(ctx):
    ctx.backend.files["/work/a.png"] = PNG
    content, _ = dispatch(ctx, "read_file", {"path": "/work/a.png"})

    caption = next(b for b in content if b["type"] == "text")["text"]
    assert "/work/a.png" in caption
    assert "1x1" in caption, "the shape of a render is worth knowing without decoding it"


def test_looking_at_a_picture_nudges_the_mirror(ctx):
    """A render the model just examined is exactly the file the user wants on their
    machine now, not at the next cycle."""
    ctx.backend.files["/work/case/renders/mesh.png"] = PNG
    poked = []
    ctx.on_render = poked.append

    dispatch(ctx, "read_file", {"path": "/work/case/renders/mesh.png"})

    assert poked == ["/work/case/renders/mesh.png"]


def test_a_nudge_that_fails_does_not_cost_the_model_its_picture(ctx):
    ctx.backend.files["/work/case/renders/mesh.png"] = PNG

    def explode(_path):
        raise RuntimeError("the mirror is mid-teardown")

    ctx.on_render = explode

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/case/renders/mesh.png"})

    assert not is_error
    assert any(block.get("type") == "image" for block in content)


def test_the_suffix_is_matched_whatever_its_case(ctx):
    ctx.backend.files["/work/A.PNG"] = PNG
    content, _ = dispatch(ctx, "read_file", {"path": "/work/A.PNG"})
    assert isinstance(content, list)


def test_an_explicit_byte_window_still_reads_bytes(ctx):
    """Asking for an offset means asking for the file, not for a look at it."""
    ctx.backend.files["/work/a.png"] = PNG
    content, _ = dispatch(ctx, "read_file", {"path": "/work/a.png", "offset": 0, "limit": 8})
    assert isinstance(content, str)
    assert "bytes 0" in content


def test_an_oversized_image_reports_its_size_rather_than_vanishing(ctx):
    """A picture that never arrives and a picture of nothing look the same from
    the inside, so the reason has to be said out loud."""
    ctx.backend.files["/work/huge.png"] = b"\x89PNG" + b"\x00" * MAX_ATTACH_BYTES

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/huge.png"})

    assert not is_error
    assert isinstance(content, str)
    assert str(MAX_ATTACH_BYTES) in content


def test_an_ordinary_file_is_untouched(ctx):
    ctx.backend.files["/work/log"] = b"Time = 1\n"
    content, _ = dispatch(ctx, "read_file", {"path": "/work/log"})
    assert isinstance(content, str) and "Time = 1" in content


def test_a_directory_is_still_a_listing(ctx):
    ctx.backend.dirs["/work/renders"] = ["mesh.png"]
    content, _ = dispatch(ctx, "read_file", {"path": "/work/renders"})
    assert isinstance(content, str) and "mesh.png" in content


def test_describe_keeps_base64_out_of_anything_a_human_reads():
    payload = base64.b64encode(b"x" * 900).decode()
    written = describe(
        [
            {"type": "image", "source": {"media_type": "image/png", "data": payload}},
            {"type": "text", "text": "/work/a.png"},
        ]
    )
    assert payload not in written
    assert "image/png" in written and "/work/a.png" in written


def test_the_model_is_sent_the_image_and_the_log_keeps_the_description(ctx, view):
    """Both halves matter: the picture has to reach the request, and a megabyte of
    base64 must not end up in the message log, which is what people read afterwards."""
    ctx.backend.files["/work/a.png"] = PNG
    cfg = Config(anthropic_api_key='test-key', model='claude-opus-5')
    loop = Loop(cfg, ctx, ctx.store, view)
    fake = install_model(
        loop,
        [
            message([tool_block("read_file", {"path": "/work/a.png"})], stop_reason="tool_use"),
            message([text_block("the mesh looks uniform")]),
        ],
    )
    loop.say("look at the render")
    loop.run()

    sent = fake.calls[-1]["messages"]
    blocks = [b for turn in sent if isinstance(turn["content"], list) for b in turn["content"]]
    results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert any(
        isinstance(part, dict) and part.get("type") == "image"
        for result in results
        for part in (result["content"] if isinstance(result["content"], list) else [])
    ), "the picture reached the request"

    logged = (ctx.store.dir / "messages.jsonl").read_text(encoding="utf-8")
    assert base64.b64encode(PNG).decode() not in logged
    assert "image/png" in logged
    for line in logged.splitlines():
        json.loads(line)  # the log stays parseable


# -- a part of a picture is not a picture ---------------------------------------


class Paged(dict):
    """A backend that answers an unbounded read with a page, as the hosted one does."""


def paging_backend(ctx, size=2_200_000, page=1_000_000):
    ctx.backend.files["/work/big.png"] = b"\x89PNG\r\n\x1a\n" + b"\x00" * (size - 8)
    original = ctx.backend.get_file

    def get_file(path, offset=0, limit=None):
        return original(path, offset=offset, limit=limit if limit is not None else page)

    ctx.backend.get_file = get_file
    return ctx


def test_a_whole_image_is_asked_for_by_name(ctx):
    """Asked for a file with no limit, the hosted service returns its page size and
    says nothing about the rest. A render between that page and the attachment
    ceiling reached the model truncated, and passed every check on the way."""
    paging_backend(ctx)

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/big.png"})

    assert not is_error
    assert isinstance(content, list), "the whole picture came back and was attached"
    image = next(b for b in content if b["type"] == "image")
    assert len(base64.b64decode(image["source"]["data"])) == 2_200_000


def test_a_short_read_is_refused_rather_than_attached(ctx):
    """If it still comes back short, saying so beats handing over a broken PNG."""
    ctx.backend.files["/work/short.png"] = b"\x89PNG\r\n\x1a\n" + b"\x00" * 500_000

    def stingy(path, offset=0, limit=None):
        return ctx.backend.files[path][:1000]

    ctx.backend.get_file = stingy

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/short.png"})

    assert not is_error
    assert isinstance(content, str)
    assert "only 1000 came back" in content
    assert "not a smaller image" in content
