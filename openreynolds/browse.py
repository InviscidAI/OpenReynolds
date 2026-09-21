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

from .backend.base import WORKSPACE_ROOT, Backend, BackendError, ExecResult, StoredListing
from .store import Store

MAX_ENTRIES = 4_000
"""A listing past this is scrolling, not information."""

DEFAULT_DEPTH = 4

TEXT_PREVIEW_BYTES = 200_000

BINARY_SNIFF_BYTES = 8_000

FIND_FORMAT = r"%y\t%s\t%T@\t%p\n"
"""Type, size, mtime, path. The escapes are for `find`, so they must survive as text."""

LIST_PIPELINE = f"| sort -n -s | cut -f2- | head -n {MAX_ENTRIES + 1}"
"""Breadth-first, then the cap. `find` prints its depth (`%d`) first and walks in
directory order -- depth-first, siblings as the filesystem happens to hold them -- so
the cap used to fall wherever the walk was at entry 4,000. In a transient study that
was inside `run/processors4/`: the walk went down that directory and streamed its
thousands of per-time field files, and the study's `renders/`, `README.md` and 201
animation frames -- written, looked at, described to the person -- were past the cap
and so never in any listing, and so never mirrored. The page for the finished study
showed `run/` and nothing else. Sorting numerically on the depth (stably, so within a
depth it is still `find`'s order) puts every shallow entry before any deep one: the
files a person is watching for sit in the first two levels, the solver's bulk in the
fourth, and the cut -- when there is one -- lands in the bulk. `cut` strips the depth
again so the parsed shape is unchanged. This is `sort` over one listing, milliseconds.

This walk is the FALLBACK now, not the listing. On a backend that can list the
workspace without running a command on it (`Backend.list_stored` -- the hosted one,
through the service's own files route) `Browser.tree` takes that answer first, running
or stopped, and only walks when there is no such answer: a local backend, a workspace
still coming up, a service without the route, a network failure. The reason is a cap
this pipeline cannot see. The hosted workspace's daemon caps a command's output at
64 KB and cuts it mid-line (`OpenFoam_Instance/daemon/settings.py`
`EXEC_OUTPUT_CAP_BYTES`, `daemon/runner.py` `out[:cap]`); a study of 1,532 entries
lists at about 83 bytes a row, ~127 KB, so `head -n 4001` never bound anything -- the
byte cap did, at ~850 rows, and the last row it left was partial. On 2026-09-21
(study 20260921-033356-076b) that partial row was `f\\t1528155\\t<mtime>\\t/work/
20260921-033356-076b`: the row for `mesh/zoom.png`, 1,528,155 bytes, cut right after
the study id, and it parsed as a 1.5 MB *file* at the study root. The page's tree
collapsed to that one leaf, and the mirror asked the service to archive the "file"
every twenty seconds (502/504 each time, 04:07 to 04:16). The service route has no
byte cap: it answers the tree to 5,000 entries, breadth-first, from the live daemon
when the workspace is up and from the copy when it is stopped. Where the walk is
still what runs, `Browser.tree` now drops the partial tail of a capped output and
says the listing was cut (`Listing.output_capped`), and `_parse` refuses a row that
is not under the listed path -- so a cut can still cost entries, but never invent one."""


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
        output_capped: bool = False,
    ):
        super().__init__(entries)
        self.truncated = truncated
        self.limit = limit
        self.root = root
        self.depth = depth
        self.output_capped = output_capped
        """The walk's output was cut by the backend's cap on what a command may print
        -- by bytes, part-way through a line -- and not by `limit`. Only ever set
        alongside `truncated`; it changes what the notice says was the cause, because a
        listing of 850 rows told it was "capped at 4,000 entries" is a notice that
        contradicts what is in front of the reader."""

    @property
    def notice(self) -> str:
        """One line saying the listing was cut short, or "" when it was not.

        Empty when nothing was cut, so a caller can print it unconditionally.
        """
        if not self.truncated:
            return ""
        where = self.root or "this path"
        if self.output_capped:
            return (
                f"listing cut short: the workspace capped the command's output part-way "
                f"through {where} within depth {self.depth}, and the rest was not looked "
                f"at. Nothing is hidden -- narrow the path, or lower the depth, to see it."
            )
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
        self._cached_output_capped: bool = False
        self._cached_depth: int = DEFAULT_DEPTH

    # -- listing ---------------------------------------------------------------

    def tree(self, path: str = "", depth: int = DEFAULT_DEPTH,
             *, background: bool = False) -> Listing:
        """Everything under `path`, to a depth, in one round trip.

        Asked of the service first, where there is one to ask. A backend that can
        list the workspace without running a command on it says so by answering
        `Backend.list_stored`; the hosted one does, through the service's own files
        route, which lists the live machine when the workspace is up and the copy
        the service keeps when it is stopped -- and starts nothing either way. That
        answer is the listing, running or stopped, cut to `depth` and `MAX_ENTRIES`
        breadth-first (`_from_store`). One path for both states, no command on the
        workspace, and none of the walk's caps: see `LIST_PIPELINE` for the 64 KB
        one that cut a study's listing mid-line and grew a 1.5 MB file at its root.

        The walk -- one `find` over the exec channel, `LIST_PIPELINE` -- is what runs
        when there is no such answer: a backend with nothing but the machine to ask
        (the local one), a workspace still coming up (`PendingBackend.list_stored`
        answers None until the machine is there, and the walk then waits for it as
        every call did), a service without the route, a network failure. One command
        beats one call per directory: a workspace has hundreds of directories, and a
        listing that takes a minute to draw is not a listing. Capped at
        `MAX_ENTRIES`, and the result says when the cap was reached; `head` is asked
        for one line more than the cap so "there was more" is measured rather than
        inferred from a listing that happens to be exactly `MAX_ENTRIES` long. And
        when the backend cut the output itself (`ExecResult.truncated`), the tail
        after the last newline is dropped before parsing and the listing says the
        output was capped -- a partial row is not an entry, whatever it parses as.

        `background=True` marks the listing a poll: it must never start a machine,
        and it does not keep one alive. Through the service that is so by
        construction -- the route starts nothing and leaves the workspace's
        last-activity clock alone. Where the walk is what runs, the poll runs only on
        a workspace already up, and raises BackendError("workspace_idle") when there
        is nothing running -- which is a thing to wait out, not an empty workspace.

        A foreground walk is asked as a poll first, too, and sent as work only when
        the poll finds nothing running: somebody asking is somebody working, and
        starting the machine is then the right answer, as it always was. That order
        is what stopped the close-down of a session whose workspace the service had
        already stopped from starting a fresh machine for one `find` (2026-09-21,
        02:24:32: a c7i.2xlarge adopted from the pool for an idle-timed-out
        session's final sync, then left to sit until the reaper took it down at
        02:42). The service's listing now answers that case before any command is
        considered; the poll-first order is kept for the backends that still walk.
        """
        path = path or self.home
        depth = int(depth)
        # `hasattr` rather than the protocol's word: every backend in the package has
        # `list_stored` (the protocol gives it a default), and a stand-in that predates
        # it -- a test's, or an embedder's -- lists as it always did.
        asks_the_service = hasattr(self.backend, "list_stored")
        if asks_the_service:
            # Before any command, whatever `background` says. Taking the listing this
            # way does not count as use of the workspace, and a listing never did: a
            # foreground one on a running workspace went out as a poll since the
            # close-down fix, and a poll leaves the workspace's last-activity clock
            # alone -- as does this, by `Backend.list_stored`'s contract. What keeps a
            # workspace alive is the work done on it, and a look at the files is not
            # work; the commands and copies that follow a listing somebody asked for
            # still count exactly as they did.
            listed = self.backend.list_stored(path, depth)
            if listed is not None:
                return self._from_store(listed, path, depth)
        # `-H` follows a symlink named on the command line, and only that one. The
        # workspace root is a symlink to the volume, so without this, listing it
        # returns nothing at all -- not an error, just an empty workspace, which is
        # the most convincing wrong answer available. Deeper symlinks are still left
        # alone, so no loop can be walked into.
        # Depth first on each line, for `sort`; see LIST_PIPELINE for why the order of
        # the walk is not the order of the listing.
        cmd = (
            f"find -H {shlex.quote(path)} -maxdepth {depth} -mindepth 1 "
            f"-printf '%d\\t{FIND_FORMAT}' 2>/dev/null {LIST_PIPELINE}"
        )
        if background or not asks_the_service:
            result = self.backend.exec(cmd, timeout_s=60, background=background)
        else:
            # The service had no listing to give (a workspace still coming up, a
            # service without the route, a failed request), so the walk it is -- as a
            # poll first, so a workspace that is up answers without being claimed,
            # and as work only when nothing is running. The service is not asked a
            # second time here: it just answered None, and the one fallback left is
            # the machine.
            result = self.backend.exec(cmd, timeout_s=60, background=True)
            if result.idle:
                result = self.backend.exec(cmd, timeout_s=60, background=False)
        if result.idle:
            # Nothing ran, so there is nothing to say about what is on disk. Falling
            # through would answer "no files", and the list_dir fallback below would
            # go start the very workspace this listing declined to start.
            raise BackendError("the workspace is not running", code="workspace_idle")
        return self._from_walk(result, path, depth)

    def _from_walk(self, result: ExecResult, root: str, depth: int) -> Listing:
        """A `Listing` from the walk's output -- what `find | sort | cut | head`
        printed, as far as the backend let it through.

        `result.truncated` is the backend saying the output is only part of what the
        command printed, and both backends cut by bytes, mid-line (the hosted daemon at
        64 KB, the local one at `MAX_OUTPUT_BYTES`). The tail after the last newline
        is therefore a row that was cut off, and it is dropped unparsed: on 2026-09-21
        the cut fell inside a path and left `f\\t1528155\\t<mtime>\\t/work/<study>`,
        four well-formed fields naming a 1.5 MB file at the study root that did not
        exist. A cut that happens to land on a line end drops nothing. Either way the
        listing is marked truncated, and its notice names the output cap rather than
        the entry cap, because the entry cap was not what was reached.
        """
        output = result.output
        output_capped = bool(result.truncated)
        if output_capped:
            output = output[: output.rfind("\n") + 1]
        entries = [entry for line in output.splitlines() if (entry := _parse(line, root))]
        over_the_cap = len(entries) > MAX_ENTRIES
        del entries[MAX_ENTRIES:]
        if entries or result.exit_code == 0:
            return Listing(
                sorted(entries, key=_order),
                truncated=over_the_cap or output_capped,
                limit=MAX_ENTRIES,
                root=root,
                depth=depth,
                # The entry cap, when it was reached, is the fuller explanation: 4,001
                # rows came through whole, and the notice that counts them is right.
                output_capped=output_capped and not over_the_cap,
            )
        return Listing(sorted(self.list_dir(root), key=_order), root=root, depth=depth)

    def _from_store(self, stored: StoredListing, root: str, depth: int) -> Listing:
        """A `Listing` from the service's answer, held to the same contract as one
        from the walk.

        "Stored" names where the answer comes from once the machine is down -- the
        copy the service writes at every checkpoint and stop; while the machine is
        up the same route reads the live tree. Either way the route answers a whole
        tree and its own cap (`stored.truncated`); this listing keeps `MAX_ENTRIES`,
        so the cut is applied here as well, and it is applied breadth-first for the
        reason `LIST_PIPELINE` gives: the route lists in its own order, and a cut at
        entry 4,000 of that order can fall on the pictures as surely as `find`'s did.
        Shallow entries first, then the cut lands in the solver's bulk. Either cap
        makes the listing truncated -- an answer the service cut short is not
        complete because this side had room for it.

        The route's `mtime` are whole seconds where `find`'s `%T@` carried a
        fraction; `Entry.mtime` stays a float and nothing that reads it is finer than
        that -- the mirror's "already here" test compares it with this machine's
        clock against the workspace's, a gap larger than any fraction of a second.
        """
        entries = [
            Entry(path=item.path, is_dir=item.is_dir, size=item.size, mtime=item.mtime)
            for item in stored.entries
        ]
        entries.sort(key=lambda entry: entry.depth)  # stable, like `sort -n -s`
        truncated = bool(stored.truncated) or len(entries) > MAX_ENTRIES
        del entries[MAX_ENTRIES:]
        return Listing(
            sorted(entries, key=_order),
            truncated=truncated,
            limit=MAX_ENTRIES,
            root=root,
            depth=depth,
        )

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
        self._cached_output_capped = bool(getattr(entries, "output_capped", False))
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
            output_capped=self._cached_output_capped,
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


def _parse(line: str, root: str) -> Entry | None:
    """One row of the walk as an `Entry`, or None for a row that is not one.

    Four tab-separated fields that read as type, size, mtime and path -- and a path
    that is strictly under `root`. The walk is `-mindepth 1`, so a row naming the root
    itself, or anything not below it, cannot be a row the walk printed whole: it is a
    row cut off inside its path (the 2026-09-21 listing's `.../<study>` was the first
    part of `.../<study>/mesh/zoom.png`), or noise on the channel. Well-formed is not
    the same as true, and a phantom entry at the root is the one wrong answer this
    listing's readers act on -- the page drew it as the whole tree, and the mirror
    asked the service to archive it every cycle.
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        return None
    kind, size, mtime, path = parts
    if not path.startswith(root.rstrip("/") + "/"):
        return None
    try:
        return Entry(path=path, is_dir=kind == "d", size=int(size), mtime=float(mtime))
    except ValueError:
        return None


def _order(entry: Entry) -> tuple:
    """Directories before files, dotted things last, alphabetical within that."""
    parent, _, name = entry.path.rpartition("/")
    return (parent, not entry.is_dir, name.startswith("."), name)
