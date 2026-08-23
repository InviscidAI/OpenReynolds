"""Show a fetched render in the terminal, where the terminal can do that.

Pure convenience. Terminals that speak neither protocol get the local path, which is
what the model already prints, so nothing depends on this working.
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
