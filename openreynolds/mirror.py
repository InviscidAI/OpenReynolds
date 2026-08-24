"""Bringing the study home.

The work happens on the instance and the agent decides what it copies out, which meant
a study that ran for twenty-seven minutes left exactly two files on the machine of the
person who commissioned it: the message log and the session record. The case, the
solver logs, the renders -- everything anyone would actually want to look at -- stayed
out on the volume, reachable only by asking for it a path at a time.

So the session mirrors, without being asked. Two things make that safe to do
automatically:

**Everything comes over.** That is the instruction, and it is the right one: a
selective mirror is a filter deciding on somebody's behalf what they wanted, and the
complaint that produced this file was exactly that -- work done and not delivered. So
the default is all of it, and the only limits left are two caps that exist so an
automatic background copy can never be the reason a laptop runs out of disk.
`--readable-only` is there for anyone who wants the small version on purpose.

**Nothing is skipped quietly.** A silent filter and an empty workspace look identical
from here, and the whole complaint that produced this file was somebody not being able
to tell those apart. Every file left behind carries the reason it was left, and the
reason is reported.

It also never gets in the way: failures are recorded and the sync returns, because a
mirror that ends a session is worse than one that misses a file.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .backend.base import WORKSPACE_ROOT, BackendError
from .browse import MAX_ENTRIES, Browser, Entry, human

MAX_FILE_BYTES = 500 * 1024 * 1024
"""No single file may be bigger than this.

Not a filter -- a stop. Everything comes home by default now, so the only job left
for a cap is to make sure an automatic background copy can never be the reason a
laptop runs out of disk. Anything it catches is named and counted, never dropped in
silence, and `openreynolds pull` will still fetch it on request."""

MAX_TOTAL_BYTES = 5 * 1024 * 1024 * 1024
"""And no single sync may pull more than this in total.

The per-file cap alone bounds nothing: ten thousand files under the limit still fill
a disk. Deliberately generous, because the instruction is that everything comes
over -- this is the line past which "everything" would start costing somebody their
machine, not a judgement about what is worth having."""

DEPTH = 12
"""How deep to look. A case sits a few directories down inside a study, and
`postProcessing/forceCoeffs/0/coefficient.dat` is another four below that."""

BATCH = 200
"""Files per round trip. One archive for everything would be fewer calls, and one
failure would then cost every file rather than two hundred of them."""

DICTIONARY_BYTES = 2 * 1024 * 1024
"""Above this, something kept for sitting in a case directory is field data.

A real dictionary is kilobytes. `0/U` on a half-million-cell mesh is nine
megabytes and is the same field data as `500/U`, sitting in the directory the
solver started from. Keeping it by location alone turned a 43-file mirror into
42 MB of it."""

BATCH_BYTES = 8_000_000
"""And a byte budget per round trip, which is the one that actually bites.

Two hundred small dictionaries are nothing; two hundred field files are forty
megabytes, and the service builds the archive in memory and streams it. A live run
asked for thirty-eight files at once and the connection closed at 6 MB of an expected
38 MB -- so the batch failed, and every keepable file in it failed with it. Counting
files bounds the blast radius of a failure; counting bytes stops it happening."""

KEEP_SUFFIXES = frozenset(
    {
        # Renders and plots -- the whole point of running the thing.
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
        # What was written down about it.
        ".md", ".txt", ".csv", ".json", ".yaml", ".yml",
        ".log",
        # What the agent wrote to do the work. Small, and the record of how a number
        # was arrived at.
        ".py", ".sh",
    }
)

CASE_DIRS = frozenset({"system", "constant", "postProcessing"})
"""Directories whose contents are worth keeping whatever the file is called.

An OpenFOAM case is defined by its dictionaries, and they have no extensions:
`fvSchemes`, `controlDict`, `transportProperties`, `U`, `p`. They are the setup -- the
thing you would need to re-run the study or to argue with its result -- and together
they are smaller than one render. `postProcessing/` is here because it holds the
answers: forces, residuals, probe values, as plain columns."""

SKIP_SUFFIXES = frozenset({".vtk", ".vtu", ".vtp", ".vtm", ".pvd", ".foam", ".pyc"})
"""Written for a viewer that is not here. A `.foam` file is an empty marker ParaView
opens; the rest are the mesh and the fields again, in another format."""

_TIME_DIR = re.compile(r"^\d+(\.\d+)?([eE][-+]?\d+)?$")
_PROCESSOR_DIR = re.compile(r"^processor\d+$")


@dataclass(frozen=True)
class Skip:
    """One file that was left on the instance, and why."""

    path: str
    reason: str
    size: int = 0

    def line(self) -> str:
        return f"  {human(self.size):>8}  {self.path}   {self.reason}"


@dataclass
class MirrorReport:
    """What one sync did. Everything it did not do is in here too."""

    local_dir: Path
    study_id: str = ""
    pulled: list[Path] = field(default_factory=list)
    bytes_pulled: int = 0
    unchanged: int = 0
    """Already here, byte for byte, and not asked for again."""
    skipped: list[Skip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    """Things that went wrong, or listings that ran out of room. Never raised."""

    def brief(self) -> list[str]:
        """One or two lines for the end of a turn. Empty when nothing happened."""
        lines = []
        if self.pulled:
            lines.append(
                f"mirrored {len(self.pulled)} file(s), {human(self.bytes_pulled)}"
                f" -> {self.local_dir}"
            )
        if self.skipped:
            lines.append(f"left on the instance: {grouped(self.skipped)}{self._hint()}")
        lines.extend(self.warnings)
        return lines

    def lines(self) -> list[str]:
        """The full account, for someone who typed `openreynolds pull` and is waiting."""
        lines = list(self.brief())
        if not lines:
            lines.append(
                f"nothing new: {self.unchanged} file(s) already here"
                if self.unchanged
                else "nothing to mirror"
            )
        elif self.unchanged:
            lines.append(f"{self.unchanged} file(s) were already here and unchanged")
        for skip in self.skipped[:LISTED_SKIPS]:
            lines.append(skip.line())
        if len(self.skipped) > LISTED_SKIPS:
            lines.append(f"  ... and {len(self.skipped) - LISTED_SKIPS} more")
        return lines

    def _hint(self) -> str:
        if not self.study_id:
            return ""
        return f"  (openreynolds pull --study {self.study_id} tries them again)"


LISTED_SKIPS = 40
"""Past this a list of refusals is scrolling, not information. The count and the
grouped reasons above it still cover everything."""


def grouped(skips: list[Skip]) -> str:
    """Skips as counts per reason, commonest first.

    A hundred lines saying `processor decomposition data` is the same fact a hundred
    times; one line saying so about a hundred files is the fact."""
    counts = Counter(skip.reason for skip in skips)
    top = counts.most_common(3)
    parts = [f"{count} {reason}" for reason, count in top]
    rest = len(skips) - sum(count for _, count in top)
    if rest:
        parts.append(f"{rest} other")
    return f"{len(skips)} file(s) - " + ", ".join(parts)


def reason_to_skip(relative: str) -> str | None:
    """Why this file stays on the instance, or None to bring it down.

    Checked in order, and the order carries meaning: the always-too-big things are
    ruled out first, so `constant/polyMesh/points` is mesh data before it is anything
    under `constant/`.
    """
    parts = relative.split("/")
    name = parts[-1]

    for part in parts[:-1]:
        if _PROCESSOR_DIR.match(part):
            return "processor decomposition data"
        if part == "polyMesh":
            return "mesh data"
        if part == "__pycache__":
            return "compiled python"
        if _is_time_directory(part):
            return f"a written time directory ({part}/)"

    suffix = _suffix(name)
    if suffix in SKIP_SUFFIXES:
        return "written for a viewer that is not on this machine"

    if any(part in CASE_DIRS or _is_initial_fields(part) for part in parts[:-1]):
        return None
    if name.startswith("log.") or suffix in KEEP_SUFFIXES:
        return None
    return "not an image, a report, a log or a case dictionary"


def reason_by_size(relative: str, size: int) -> str | None:
    """A second look, once the size is known.

    A case dictionary is kilobytes: `controlDict`, `fvSchemes`, `U` on a tutorial
    mesh. The same `0/U` on a half-million-cell mesh is nine megabytes, and it is
    field data whatever directory it is sitting in. Keeping it by location alone
    turned a 43-file mirror into 42 MB of it.
    """
    if size <= DICTIONARY_BYTES:
        return None
    parts = [part for part in relative.split("/") if part]
    kept_by_place = any(
        part in CASE_DIRS or _is_initial_fields(part) for part in parts[:-1]
    )
    if not kept_by_place:
        return None
    if _suffix(parts[-1]) in KEEP_SUFFIXES or parts[-1].startswith("log."):
        return None
    return f"field data ({human(size)}); a case dictionary is not this big"


def sync(
    browser: Browser,
    *,
    path: str = "",
    everything: bool = False,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> MirrorReport:
    """Copy down what has changed under the study's directory, and say what it did not.

    Returns a report whatever happens. Callers are sessions with work in flight, and
    none of them can afford an exception from a convenience.
    """
    store = browser.store
    if store is None:
        return MirrorReport(local_dir=Path.cwd(), warnings=["no study directory to mirror into"])

    report = MirrorReport(local_dir=store.fetch_dir(), study_id=store.session.study_id)
    root = path or browser.home or WORKSPACE_ROOT

    try:
        entries = browser.tree(root, depth=DEPTH)
    except (BackendError, OSError) as exc:
        report.warnings.append(f"could not look at {root}: {exc}")
        return report

    if len(entries) >= MAX_ENTRIES:
        # `find` output is capped, so the tail of a very large workspace was never
        # examined at all. That is a different thing from there being nothing there.
        report.warnings.append(
            f"the listing stopped at {MAX_ENTRIES} entries; part of {root} was not looked at"
        )

    candidates = _wanted(entries, root, report, everything, max_file_bytes)
    _pull(browser, _within_budget(candidates, report, max_total_bytes), report)
    return report


def local_for(local_dir: Path, remote: str) -> Path:
    """Where a workspace path lands once it has been copied out.

    `get_tree` preserves the shape a file had relative to the workspace root, so this
    is a prediction rather than a guess -- which is what makes it possible to ask
    whether a file is already here without pulling it to find out.
    """
    if remote.startswith(WORKSPACE_ROOT + "/"):
        relative = remote[len(WORKSPACE_ROOT) + 1 :]
    else:
        relative = Path(remote).name
    return local_dir / relative


# -- the parts of one sync -----------------------------------------------------


def _wanted(
    entries: list[Entry],
    root: str,
    report: MirrorReport,
    everything: bool,
    max_file_bytes: int,
) -> list[Entry]:
    """Files worth asking for: not filtered out, not too big, not already here."""
    wanted = []
    for entry in entries:
        if entry.is_dir:
            continue
        relative = _relative(entry.path, root)
        reason = None if everything else (
            reason_to_skip(relative) or reason_by_size(relative, entry.size)
        )
        if reason:
            report.skipped.append(Skip(entry.path, reason, entry.size))
            continue
        if entry.size > max_file_bytes:
            # The cap holds even under `--all`: "everything" is a statement about
            # which files are interesting, not about how much disk to use.
            report.skipped.append(
                Skip(
                    entry.path,
                    f"{human(entry.size)}, over the {human(max_file_bytes)} limit for one file",
                    entry.size,
                )
            )
            continue
        if _already_here(local_for(report.local_dir, entry.path), entry):
            report.unchanged += 1
            continue
        wanted.append(entry)
    return wanted


def _within_budget(
    candidates: list[Entry], report: MirrorReport, max_total_bytes: int
) -> list[Entry]:
    """As much as fits, smallest first.

    Smallest first because of what running out should cost: one enormous log left
    behind is a sentence in the report, and three hundred dictionaries left behind
    because that log went first is the study missing.
    """
    budget = max_total_bytes
    wanted = []
    for entry in sorted(candidates, key=lambda item: (item.size, item.path)):
        if entry.size > budget:
            report.skipped.append(
                Skip(
                    entry.path,
                    f"past the {human(max_total_bytes)} budget for one sync",
                    entry.size,
                )
            )
            continue
        budget -= entry.size
        wanted.append(entry)
    return wanted


def _batches(wanted: list[Entry]) -> list[list[Entry]]:
    """Split into round trips bounded by both count and bytes.

    A single file over the byte budget still goes on its own rather than being dropped
    -- it is the caller's job to decide what is too big, and it already has.
    """
    batches: list[list[Entry]] = []
    current: list[Entry] = []
    carried = 0
    for entry in wanted:
        if current and (len(current) >= BATCH or carried + entry.size > BATCH_BYTES):
            batches.append(current)
            current, carried = [], 0
        current.append(entry)
        carried += entry.size
    if current:
        batches.append(current)
    return batches


def _pull(browser: Browser, wanted: list[Entry], report: MirrorReport) -> None:
    """Fetch in batches, and treat a failed batch as a fact rather than an end.

    A batch that fails is retried one file at a time before being given up on. The
    usual cause is one awkward file in otherwise fine company, and losing the company
    with it is how a mirror comes back empty from a study that had plenty worth
    keeping -- which is exactly what a live run did.
    """
    for batch in _batches(wanted):
        try:
            written = browser.backend.get_tree([entry.path for entry in batch], report.local_dir)
        except (BackendError, OSError) as exc:
            if len(batch) > 1:
                report.warnings.append(
                    f"a batch of {len(batch)} failed ({exc}); trying them one at a time"
                )
                for entry in batch:
                    _pull(browser, [entry], report)
                continue
            report.warnings.append(f"could not copy {len(batch)} file(s): {exc}")
            report.skipped.extend(
                Skip(entry.path, "the copy failed", entry.size) for entry in batch
            )
            continue
        report.pulled.extend(written)
        report.bytes_pulled += sum(entry.size for entry in batch)


def _already_here(target: Path, entry: Entry) -> bool:
    """Whether the local copy is still the remote file.

    The local copy is written fresh on extraction, so its mtime is when it was pulled,
    not when it was written on the instance. That makes "the instance's copy is newer
    than the moment we copied it" the question worth asking, and it is the one that
    keeps a per-turn sync down to the handful of files that actually changed.
    """
    try:
        stat = target.stat()
    except OSError:
        return False
    return stat.st_size == entry.size and stat.st_mtime >= entry.mtime


def _is_initial_fields(part: str) -> bool:
    """The starting conditions: `0`, and the copies people keep beside it.

    `0.orig` is the convention the tutorials use, but it is only a convention -- a
    live run wrote `0.initial` and every field in it was skipped as "not a case
    dictionary", because the rule listed names instead of describing them. So: `0`,
    or `0.` followed by anything that is not a number, since `0.5` is a time the
    solver wrote at and `0.initial` is the setup it started from.
    """
    if part == "0":
        return True
    head, dot, tail = part.partition(".")
    return bool(head == "0" and dot and tail and not tail.isdigit())


def _is_time_directory(part: str) -> bool:
    """A directory named after a time the solver wrote at -- `0.5`, `250`, `1e-05`.

    The initial conditions are excluded on purpose: they are part of the setup, and
    every later one is the field data the setup produces."""
    if part in CASE_DIRS or _is_initial_fields(part):
        return False
    if not _TIME_DIR.match(part):
        return False
    return float(part) != 0.0


def _suffix(name: str) -> str:
    dot = name.rfind(".")
    return name[dot:].lower() if dot > 0 else ""


def _relative(path: str, root: str) -> str:
    base = root.rstrip("/") + "/"
    return path[len(base) :] if path.startswith(base) else path.lstrip("/")
