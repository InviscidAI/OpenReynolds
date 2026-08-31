"""Pictures: for the user's terminal, and for the model.

Two unrelated jobs happen to share a file-type question. `show` draws a fetched render
inline where the terminal speaks a graphics protocol -- pure convenience, and terminals
that speak neither get the local path instead.

`attachment` is the other one, and it is not convenience. A render nobody looks at is
a file; the model can generate a picture of a mesh, and unless the picture comes back
as a picture it can only ever read its own description of what it meant to draw. This
turns image bytes into a content block, so `read_file` on a PNG returns the thing
itself and looking is possible.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

INLINE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})

MAX_INLINE_BYTES = 4_000_000
"""Past this an escape sequence is more nuisance than picture."""

_KITTY_CHUNK = 4096


def protocol(env: dict[str, str] | None = None) -> str | None:
    """Which graphics protocol this terminal speaks, if any."""
    env = os.environ if env is None else env
    if env.get("OPENREYNOLDS_INLINE_IMAGES", "").lower() in ("0", "off", "no"):
        return None
    if env.get("TERM") == "xterm-kitty" or env.get("KITTY_WINDOW_ID"):
        return "kitty"
    if env.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm") or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm2"
    return None


def show(path: Path, stream=None) -> bool:
    """Draw the image inline. Returns whether anything was drawn."""
    stream = stream or sys.stdout
    if path.suffix.lower() not in INLINE_SUFFIXES:
        return False
    kind = protocol()
    if kind is None:
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if not data or len(data) > MAX_INLINE_BYTES:
        return False

    payload = base64.b64encode(data)
    try:
        if kind == "iterm2":
            _iterm2(payload, path.name, len(data), stream)
        else:
            _kitty(payload, stream)
        stream.flush()
    except (OSError, ValueError):
        return False
    return True


def _iterm2(payload: bytes, name: str, size: int, stream) -> None:
    name_b64 = base64.b64encode(name.encode("utf-8")).decode("ascii")
    stream.write(
        f"\033]1337;File=name={name_b64};size={size};inline=1;"
        f"width=auto;preserveAspectRatio=1:{payload.decode('ascii')}\a\n"
    )


def _kitty(payload: bytes, stream) -> None:
    """Kitty takes the payload in 4 KB chunks, flagged until the last one."""
    chunks = [payload[i : i + _KITTY_CHUNK] for i in range(0, len(payload), _KITTY_CHUNK)]
    for index, chunk in enumerate(chunks):
        last = index == len(chunks) - 1
        control = "a=T,f=100," if index == 0 else ""
        stream.write(f"\033_G{control}m={0 if last else 1};{chunk.decode('ascii')}\033\\")
    stream.write("\n")


# -- pictures for the model ----------------------------------------------------

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
"""The four formats the model API accepts. Anything else is bytes to it."""

MAX_ATTACH_BYTES = 3_500_000
"""Under the API's per-image ceiling with room for base64 expansion."""


def media_type(name: str) -> str | None:
    """The media type for a path, or None if this is not a picture."""
    suffix = Path(name).suffix.lower()
    return MEDIA_TYPES.get(suffix)


ATTACH_MAX_EDGE = 1024
"""Longest edge an attached render is scaled down to.

Renders arrive at 900x700 to 1600x500 and cost 840-1,333 tokens each, and a study reads
the same path several times over while it fixes the framing. A mesh at 1024 px shows
everything a mesh check is looking for -- cell spacing near a wall, a refinement band,
whether the geometry is the right way up -- at roughly half the tokens. Transport only:
the model still sees every picture it asks for, and `read_file` still hands back the
whole file to anything that wants the bytes."""


def downscale(data: bytes, media: str, max_edge: int = ATTACH_MAX_EDGE) -> bytes:
    """`data` with its longest edge brought down to `max_edge`, or unchanged.

    Unchanged is the honest fallback for every reason it might not work -- no Pillow on
    this machine, an animation whose frames must not be flattened, a format that will
    not round-trip. A picture that arrives larger than necessary is a cost; one that
    arrives damaged, or does not arrive, is a wrong answer."""
    if media == "image/gif":
        return data  # flattening an animation loses the thing it is of
    shape = dimensions(data)
    if not shape or max(shape) <= max_edge:
        return data
    try:
        import io as _io

        from PIL import Image
    except ImportError:
        return data
    try:
        with Image.open(_io.BytesIO(data)) as image:
            image.thumbnail((max_edge, max_edge))
            buffer = _io.BytesIO()
            image.save(buffer, format=image.format or "PNG")
    except Exception:  # noqa: BLE001 - a picture that arrives is worth more than a small one
        return data
    smaller = buffer.getvalue()
    return smaller if 0 < len(smaller) < len(data) else data


def attachment(data: bytes, media: str) -> dict:
    """One image content block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def dimensions(data: bytes) -> tuple[int, int] | None:
    """Width and height straight out of a PNG header, or None for anything else.

    Enough to report the shape of a render without a decoder: no image library is a
    runtime dependency here, and the answer only ever appears next to the picture.
    """
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return (width, height) if width and height else None
