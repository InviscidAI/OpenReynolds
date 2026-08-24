"""Looking at the workspace, without having to ask the model for it.

The model decides what it copies out; the user should not have to negotiate with it to
find out what is there. This is a read-only window onto the workspace and onto the
local mirror. It never writes to the workspace, never appears in the conversation, and
nothing the model does depends on whether anyone is looking.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from .backend.base import WORKSPACE_ROOT, Backend, BackendError
from .store import Store

MAX_ENTRIES = 4_000
"""A listing past this is scrolling, not information."""

DEFAULT_DEPTH = 4

TEXT_PREVIEW_BYTES = 200_000

BINARY_SNIFF_BYTES = 8_000

FIND_FORMAT = r"%y\t%s\t%T@\t%p\n"
"""Type, size, mtime, path. The escapes are for `find`, so they must survive as text."""


@dataclass(frozen=True)
class Entry:
    """One path in the workspace."""

    path: str
    is_dir: bool
    size: int = 0
    mtime: float = 0.0

    @property
    def name(self) -> str:
        return self.path.rstrip("/").rsplit("/", 1)[-1] or self.path

    @property
    def depth(self) -> int:
        return self.path.count("/")

    def line(self) -> str:
        return f"{'d' if self.is_dir else '-'} {human(self.size):>8}  {self.path}"


def human(size: int) -> str:
    """Bytes at a glance."""
    step = float(size)
    for unit in ("B", "K", "M", "G"):
        if step < 1024 or unit == "G":
            return f"{step:.0f}{unit}" if unit == "B" or step >= 10 else f"{step:.1f}{unit}"
        step /= 1024
    return f"{size}B"


class Browser:
    """Read-only access to a workspace and to what has been pulled out of it."""

    def __init__(self, backend: Backend, store: Store | None = None, home: str = WORKSPACE_ROOT):
        self.backend = backend
        self.store = store
        self.home = home or WORKSPACE_ROOT
        """Where looking starts: this study's own directory, not everyone else's."""

    # -- listing ---------------------------------------------------------------

    def tree(self, path: str = "", depth: int = DEFAULT_DEPTH) -> list[Entry]:
        """Everything under `path`, to a depth, in one round trip.

        One command beats one call per directory: a workspace has hundreds of
        directories, and a listing that takes a minute to draw is not a listing.
        """
        path = path or self.home
        cmd = (
            f"find {shlex.quote(path)} -maxdepth {int(depth)} -mindepth 1 "
            f"-printf '{FIND_FORMAT}' 2>/dev/null | head -n {MAX_ENTRIES}"
        )
        result = self.backend.exec(cmd, timeout_s=60)
        entries = [entry for line in result.output.splitlines() if (entry := _parse(line))]
        if entries or result.exit_code == 0:
            return sorted(entries, key=_order)
        return sorted(self.list_dir(path), key=_order)

    def list_dir(self, path: str = "") -> list[Entry]:
        """One directory's immediate children. Sizes would need a stat each, so are 0."""
        info = self.backend.stat(path or self.home)
        if not info.is_dir:
            return [Entry(path=info.path, is_dir=False, size=info.size, mtime=info.mtime)]
        base = path.rstrip("/")
        return [Entry(path=f"{base}/{name}", is_dir=False) for name in info.entries]

    # -- one file --------------------------------------------------------------

    def read(self, path: str, limit: int = TEXT_PREVIEW_BYTES) -> tuple[str, bool]:
        """Text of a file, and whether it was text at all.

        A binary file gets a description instead of its bytes: a screenful of mojibake
        tells the reader less than one line saying what it is.
        """
        info = self.backend.stat(path)
        if info.is_dir:
            listing = "\n".join(sorted(info.entries)) or "(empty)"
            return f"{path}  directory, {len(info.entries)} entries\n\n{listing}", True

        raw = self.backend.get_file(path, offset=0, limit=limit)
        if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
            return f"{path}\n\n{human(info.size)} of binary data. Pull it out to open it.", False

        text = raw.decode("utf-8", errors="replace")
        if info.size > len(raw):
            text += f"\n\n[showing the first {human(len(raw))} of {human(info.size)}]"
        return text, True

    def pull(self, path: str) -> list[Path]:
        """Copy something out to the local mirror, and say where it landed."""
        if self.store is None:
            raise BackendError("no study directory to pull into", code="no_store")
        return self.backend.get_tree([path], self.store.fetch_dir())

    # -- the local side --------------------------------------------------------

    def local(self) -> list[Path]:
        """What has already been copied out, oldest first."""
        if self.store is None or not self.store.files_dir.is_dir():
            return []
        found = [p for p in self.store.files_dir.rglob("*") if p.is_file()]
        return sorted(found, key=lambda p: p.stat().st_mtime)


def _parse(line: str) -> Entry | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        return None
    kind, size, mtime, path = parts
    try:
        return Entry(path=path, is_dir=kind == "d", size=int(size), mtime=float(mtime))
    except ValueError:
        return None


def _order(entry: Entry) -> tuple:
    """Directories before files, dotted things last, alphabetical within that."""
    parent, _, name = entry.path.rpartition("/")
    return (parent, not entry.is_dir, name.startswith("."), name)
