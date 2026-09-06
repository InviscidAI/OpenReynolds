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


class Listing(list):
    """The entries, and whether the walk that produced them was cut short.

    A `list`, so every caller that only wants the entries is unchanged and nothing
    downstream has to learn a new type. The flag rides along for the callers that
    *draw* the listing, because a cap nobody is told about is indistinguishable from
    a workspace that ends there -- which is the most convincing wrong answer a file
    browser can give.

    Not a filter. When the walk is capped the answer is to say so and let the person
    narrow `path` or `depth`; a browser that quietly omits a directory that exists on
    the filesystem is worse than one that admits it ran out of room, and the person
    most likely to be looking at the dotted bookkeeping directory is somebody
    debugging the thing that fills it.
    """

    def __init__(
        self,
        entries=(),
        *,
        truncated: bool = False,
        limit: int = MAX_ENTRIES,
        root: str = "",
        depth: int = DEFAULT_DEPTH,
    ):
        super().__init__(entries)
        self.truncated = truncated
        self.limit = limit
        self.root = root
        self.depth = depth

    @property
    def notice(self) -> str:
        """One line saying the listing was cut short, or "" when it was not.

        Empty when nothing was cut, so a caller can print it unconditionally.
        """
        if not self.truncated:
            return ""
        where = self.root or "this path"
        return (
            f"listing capped at {self.limit:,} entries: {where} holds more than that "
            f"within depth {self.depth}, and the rest was not looked at. Nothing is "
            f"hidden -- narrow the path, or lower the depth, to see it."
        )

    def lines(self) -> list[str]:
        """What to print: every entry, and the cap notice when there was one."""
        drawn = [entry.line() for entry in self]
        if self.truncated:
            drawn.append(self.notice)
        return drawn


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
        self._cached_truncated: bool = False
        self._cached_depth: int = DEFAULT_DEPTH

    # -- listing ---------------------------------------------------------------

    def tree(self, path: str = "", depth: int = DEFAULT_DEPTH) -> Listing:
        """Everything under `path`, to a depth, in one round trip.

        One command beats one call per directory: a workspace has hundreds of
        directories, and a listing that takes a minute to draw is not a listing.

        Capped at `MAX_ENTRIES`, and the result says when the cap was reached. It
        used to say nothing: 2,993 command logs under one dotted directory took
        three quarters of the budget and the study tree was cut off behind them,
        with a listing that looked complete. `head` is asked for one line more than
        the cap so "there was more" is measured rather than inferred from a listing
        that happens to be exactly `MAX_ENTRIES` long.
        """
        path = path or self.home
        # `-H` follows a symlink named on the command line, and only that one. The
        # workspace root is a symlink to the volume, so without this, listing it
        # returns nothing at all -- not an error, just an empty workspace, which is
        # the most convincing wrong answer available. Deeper symlinks are still left
        # alone, so no loop can be walked into.
        cmd = (
            f"find -H {shlex.quote(path)} -maxdepth {int(depth)} -mindepth 1 "
            f"-printf '{FIND_FORMAT}' 2>/dev/null | head -n {MAX_ENTRIES + 1}"
        )
        result = self.backend.exec(cmd, timeout_s=60)
        entries = [entry for line in result.output.splitlines() if (entry := _parse(line))]
        truncated = len(entries) > MAX_ENTRIES
        del entries[MAX_ENTRIES:]
        if entries or result.exit_code == 0:
            return Listing(
                sorted(entries, key=_order),
                truncated=truncated,
                limit=MAX_ENTRIES,
                root=path,
                depth=int(depth),
            )
        return Listing(sorted(self.list_dir(path), key=_order), root=path, depth=int(depth))

    def remember(self, root: str, entries: list[Entry]) -> None:
        """Keep the last full listing, and when it was taken.

        The background mirror lists the workspace every cycle anyway; keeping the
        answer means anything that wants to *show* the workspace can do so without
        paying a network round trip for a listing somebody just took."""
        self._cached_root = root.rstrip("/") or WORKSPACE_ROOT
        self._cached_entries = list(entries)
        # Whether the listing was cut short is part of the listing. A pane drawn from
        # the remembered copy must not look more complete than the one drawn from the
        # call that took it.
        self._cached_truncated = bool(getattr(entries, "truncated", False))
        self._cached_depth = int(getattr(entries, "depth", DEFAULT_DEPTH))
        self._cached_at = time.time()

    def cached(self, path: str = "") -> Listing | None:
        """The remembered listing under `path`, or None when it does not cover it.

        None means "go and look", never "there is nothing there" -- a cache miss and
        an empty directory are different answers and only one of them is this one's
        to give."""
        if self._cached_entries is None:
            return None
        target = (path or self.home).rstrip("/") or WORKSPACE_ROOT
        if target == self._cached_root:
            return self._remembered(self._cached_entries, self._cached_root)
        if not target.startswith(self._cached_root + "/"):
            return None
        prefix = target + "/"
        return self._remembered(
            [
                entry
                for entry in self._cached_entries
                if entry.path == target or entry.path.startswith(prefix)
            ],
            target,
        )

    def _remembered(self, entries: list[Entry], root: str) -> Listing:
        """A slice of the remembered listing, still carrying whether it was capped.

        A subtree of a capped walk is capped too: `find` stopped part-way through the
        whole listing, so what is missing could be anywhere under it."""
        return Listing(
            entries,
            truncated=self._cached_truncated,
            limit=MAX_ENTRIES,
            root=root,
            depth=self._cached_depth,
        )

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
