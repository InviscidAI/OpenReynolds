"""A file someone hands a study, put where its agent looks for it.

Every door a file comes through -- an MCP caller's upload, a page's upload on the
hosted service -- lands it here, under `<home>/uploads/`, so what the agent finds is
the same whichever door it came through.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .backend.base import BackendError

Derive = Callable[[str, bytes], "tuple[list[tuple[str, bytes]], str]"]
"""(the placed path, its bytes) -> (derived files as (path under the original's stem,
bytes), a line saying what was derived). How an embedder unpacks a file the agent's
tools cannot read (a PDF into page images and text, say)."""


def safe_filename(name: str) -> str:
    """The basename, in the charset a workspace path can carry anywhere: no
    directories, no shell-significant characters, never empty."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "upload"
    return base[:120]


def place_upload(backend: Any, home: str, name: str, data: bytes,
                 derive: Derive | None = None) -> tuple[str, list[str], str]:
    """Put a file under `<home>/uploads/` on the workspace. Returns the workspace path,
    the derived files and a line saying what was derived. Raises `BackendError` when
    the workspace refuses the file."""
    base = safe_filename(name)
    target = f"{home}/uploads/{base}"
    # A second upload of the same name sits beside the first, not over it: the first
    # may already be what a running job is reading.
    stem, dot, ext = base.partition(".")
    for n in range(2, 21):
        try:
            backend.stat(target)
        except BackendError:
            break
        target = f"{home}/uploads/{stem}-{n}{dot}{ext}" if dot else f"{home}/uploads/{stem}-{n}"
    backend.put_file(target, data)
    extras: list[str] = []
    made = ""
    if derive is not None:
        derived, made = derive(target, data)
        root = target.rsplit(".", 1)[0] if "." in Path(target).name else target
        for rel, blob in derived:
            try:
                backend.put_file(f"{root}/{rel}", blob)
                extras.append(f"{root}/{rel}")
            except BackendError:
                break          # the original is up; partial derivatives are fine
    return target, extras, made
