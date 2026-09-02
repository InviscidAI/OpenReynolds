"""Looking at the workspace, without having to ask the model for it.

The model decides what it copies out; the user should not have to negotiate with it to
find out what is there. This is a read-only window onto the workspace and onto the
local mirror. It never writes to the workspace, never appears in the conversation, and
nothing the model does depends on whether anyone is looking.
"""

from __future__ import annotations

import shlex
import time
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
        self._cached_entries: list[Entry] | None = None
        self._cached_root: str = ""
        self._cached_at: float = 0.0

    # -- listing ---------------------------------------------------------------

    def tree(self, path: str = "", depth: int = DEFAULT_DEPTH) -> list[Entry]:
        """Everything under `path`, to a depth, in one round trip.

        One command beats one call per directory: a workspace has hundreds of
        directories, and a listing that takes a minute to draw is not a listing.
        """
        path = path or self.home
        # `-H` follows a symlink named on the command line, and only that one. The
        # workspace root is a symlink to the volume, so without this, listing it
        # returns nothing at all -- not an error, just an empty workspace, which is
        # the most convincing wrong answer available. Deeper symlinks are still left
        # alone, so no loop can be walked into.
        cmd = (
            f"find -H {shlex.quote(path)} -maxdepth {int(depth)} -mindepth 1 "
            f"-printf '{FIND_FORMAT}' 2>/dev/null | head -n {MAX_ENTRIES}"
        )
        result = self.backend.exec(cmd, timeout_s=60)
        entries = [entry for line in result.output.splitlines() if (entry := _parse(line))]
        if entries or result.exit_code == 0:
            return sorted(entries, key=_order)
        return sorted(self.list_dir(path), key=_order)

    def remember(self, root: str, entries: list[Entry]) -> None:
        """Keep the last full listing, and when it was taken.

        The background mirror lists the workspace every cycle anyway; keeping the
        answer means anything that wants to *show* the workspace can do so without
        paying a network round trip for a listing somebody just took."""
        self._cached_root = root.rstrip("/") or WORKSPACE_ROOT
        self._cached_entries = list(entries)
        self._cached_at = time.time()

    def cached(self, path: str = "") -> list[Entry] | None:
        """The remembered listing under `path`, or None when it does not cover it.

        None means "go and look", never "there is nothing there" -- a cache miss and
        an empty directory are different answers and only one of them is this one's
        to give."""
        if self._cached_entries is None:
            return None
        target = (path or self.home).rstrip("/") or WORKSPACE_ROOT
        if target == self._cached_root:
            return list(self._cached_entries)
        if not target.startswith(self._cached_root + "/"):
            return None
        prefix = target + "/"
        return [
            entry
            for entry in self._cached_entries
            if entry.path == target or entry.path.startswith(prefix)
        ]

    def cache_age(self) -> float | None:
        """Seconds since the remembered listing was taken, or None if there is none."""
        if self._cached_entries is None:
            return None
        return max(0.0, time.time() - self._cached_at)

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
        """Copy something out to the local mirror, and say where it landed.

        Bounded, unlike the single `get_tree` this used to be. That asked the service to
        build one in-memory tar of the whole subtree, consulting none of the mirror's
        caps -- aimed at a case root during a decomposed solve it meant every
        `processorN/` directory in one archive. The mirror already records what that
        costs: a 38 MB batch whose connection closed at 6 MB, taking every file in it
        down with it. So this goes through the same sync the background cycles use,
        which batches by count and by bytes and reports what it left behind.
        """
        if self.store is None:
            raise BackendError("no study directory to pull into", code="no_store")
        from .mirror import sync

        # One named file is already bounded -- it is one file -- and asking for a
        # listing of it first would be a round trip to learn what the caller said.
        try:
            if not self.backend.stat(path).is_dir:
                return self.backend.get_tree([path], self.store.fetch_dir())
        except BackendError:
            pass  # cannot tell what it is; let the walk below decide

        report = sync(self, path=path, live=True)
        if report.warnings and not report.pulled:
            # Nothing came back and something went wrong: that is a failed pull, and
            # the caller should hear the reason rather than an empty list.
            raise BackendError("; ".join(report.warnings), code="pull_failed")
        return list(report.pulled)

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
