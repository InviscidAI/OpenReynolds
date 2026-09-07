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

There are three policies, not two, and the third is the one the background cycles use
(`live=True`, `reason_to_skip_live`). `everything` asks a decomposed solve for every
per-processor field file it is writing, which is why a cycle never converged during one:
thirty-one minutes after the answer was written, moving 233 files of which 156 were
`processorN/` data nothing in either repo opens. `--readable-only` is the opposite
mistake for a live cycle -- it drops `constant/polyMesh`, every written time and every
STL, which is exactly what the hosted 3D viewer reads out of the mirror. The live policy
keeps the mesh, the newest time, logs, postProcessing, dictionaries, images and surfaces,
and drops only what is redundant. Pictures are also asked for in the first round trip
rather than smallest-first, because smallest-first during a solve means "after the
processor files".

**Nothing is skipped quietly.** A silent filter and an empty workspace look identical
from here, and the whole complaint that produced this file was somebody not being able
to tell those apart. Every file left behind carries the reason it was left, and the
reason is reported.

It also never gets in the way: failures are recorded and the sync returns, because a
mirror that ends a session is worse than one that misses a file.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import trace
from .backend.base import WORKSPACE_ROOT, BackendError
from .browse import MAX_ENTRIES, Browser, Entry, human

GATE_WAIT_S = 120.0
"""How long a background cycle will stand aside for a tool call before going anyway.

The service used to run the mirror's transfers and the model's commands on the same
container, and a transfer that stalls there stalls the command behind it: a finish
step timed at 27 s on the instance took 120, 236 and 323 s inside the tool call, each
time with a cycle in flight. That is still true of the listing (`browser.tree`, a
`find` over the exec channel a tool call also uses), so a cycle still waits for the
tool call to end before it starts one -- bounded, because a turn of back-to-back tool
calls would otherwise never even list at all, and a bounded wait is a delay, which the
cycle can afford, where an unbounded one is the twenty-five minutes the turn-end sync
once cost. It is no longer true of a pull: `get_tree(..., via="volume")` reads the
persistent volume directly, off the container a tool call uses, so a live cycle's
round trips no longer wait on this at all (see `_pull_batch`) -- there is nothing left
to stand aside for."""

LIVE_PULL_TIMEOUT_S = 25.0
"""How long a background cycle's own archive request may run before it gives up on it.

Measured against the mesher acceptance runs (`qa-runs/LATENCY.md`): every ordinary
round trip in that corpus, the smallest and the largest, finished in 18 s or less; the
ones that did not finished in two to eleven *minutes* -- one path per cycle that the
service takes about 120 s to answer 400 for (a jail probe timing out on a file this
sandbox has never touched, `qa-runs/FINDINGS.md` F-55) and the transfer stalling in
its company. 25 s clears every real transfer seen and cuts every one of those off
early, at one attempt (see `max_attempts` on `Backend.get_tree`) rather than the
five the transport retries by default -- five attempts at the ordinary 300 s timeout
is how one awkward path turns a cycle into ten or twenty-five minutes of it, all spent
on the same container a tool call is waiting to use. A cut-off file is not lost: it stays
skipped, reported, and tried again next cycle, at the cost of this same 25 s again
until it is either fetchable or the service answers its 400 fast enough to be
remembered (`refused`) -- worse than F-55 being fixed, much better than not bounding
this at all."""

SLOW_CYCLE_S = 30.0
"""A cycle that pulled nothing and took longer than this still gets a line. Silence
is right for the ordinary empty cycle -- it runs every twenty seconds for the whole
session -- and wrong for one that spent eight minutes finding that out."""

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
PROCESSOR_DATA = "processor decomposition data"
"""Why a decomposed field is left behind. Named so the report can recognise it."""

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
    round_trips: int = 0
    """Archive requests made, retries included. What a cycle cost in calls."""
    seconds: float = 0.0
    """What it cost in time, listing included. Together these are the line that
    shows whether a cycle is a convenience or the reason a session is slow."""
    gate_wait_seconds: float = 0.0
    """Of `seconds`, how much was spent standing aside for a tool call in flight
    (`Gate.clear`), as opposed to actually listing or transferring. Not shown in the
    text report -- it is diagnostic, for telling "the cycle waited" from "the cycle
    was slow" apart (`qa-runs/LATENCY.md` asked this question of a live cycle and had
    no way to answer it directly; this is that way). Carried in the `mirror` trace
    event as `gate_wait_seconds`."""

    def cost(self) -> str:
        trips = f"{self.round_trips} round trip{'s' if self.round_trips != 1 else ''}"
        return f"{trips}, {took(self.seconds)}"

    def brief(self) -> list[str]:
        """One or two lines for the end of a turn. Empty when nothing happened.

        A cycle that pulled nothing but took minutes did happen, and says so: a
        session's wall clock was going somewhere and the log had no line for it."""
        lines = []
        if self.pulled:
            lines.append(
                f"mirrored {len(self.pulled)} file(s), {human(self.bytes_pulled)}"
                f" -> {self.local_dir}  ({self.cost()})"
            )
        elif self.seconds >= SLOW_CYCLE_S:
            lines.append(f"mirror: nothing new after {self.cost()}")
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
        parts = []
        if self.study_id:
            parts.append(f"openreynolds pull --study {self.study_id} tries them again")
        # Decomposed fields are the one skip a retry cannot fix: the filter drops them
        # every time, by design, so the line above is a promise it does not keep for
        # them. What brings them home is reconstructing them into a case. Said only
        # when there are some, and alongside the retry rather than instead of it --
        # a mixed set has files the retry does work for.
        if any(skip.reason == PROCESSOR_DATA for skip in self.skipped):
            parts.append("reconstructPar makes the decomposed fields a case")
        return f"  ({'; '.join(parts)})" if parts else ""


LISTED_SKIPS = 40
"""Past this a list of refusals is scrolling, not information. The count and the
grouped reasons above it still cover everything."""


def took(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def is_a_fact_about_the_path(exc: BaseException) -> bool:
    """Whether a failed copy will fail the same way if asked again.

    A 400 is the service saying the request itself is wrong -- and for an archive
    request the only thing in it is the paths. It answers that way for a path outside
    the jail, and it answers the same way when its own jail probe (`realpath` and
    `test -e` on the instance) runs past its two-minute timeout, which is what an
    uploaded file nothing on the instance has opened yet does to it. From here the
    two are the same fact: this path costs two minutes and yields nothing, and it
    will tomorrow too. A connection that dropped or a service that blinked is not
    that; those are tried again next cycle as they always were."""
    return isinstance(exc, BackendError) and (
        exc.status == 400 or exc.code == "bad_request"
    )


def refusal(exc: BaseException) -> str:
    """The reason a refused path is left with, so the report can group it and a
    reader can see the service's own words."""
    said = getattr(exc, "message", None) or str(exc)
    return f"the service refused it this session ({said})"


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


LIVE_DROP_DIRS = frozenset({"VTK", "__pycache__"})
"""Directories a background cycle never brings home. `processorN/` is handled
separately because it is matched by shape rather than by name."""


def newest_times(entries, root: str) -> dict[str, str]:
    """For each case in the listing, the name of its newest written time directory.

    A running solve writes a new one every few seconds and every earlier one is a
    complete copy of the fields. Keeping them all is what made a cycle never converge;
    keeping the newest is what the viewer and the user actually look at."""
    newest: dict[str, tuple[float, str]] = {}
    for entry in entries:
        parts = _relative(entry.path, root).split("/")
        for index, part in enumerate(parts[:-1]):
            if not _is_time_directory(part):
                continue
            try:
                value = float(part)
            except ValueError:
                continue
            case = "/".join(parts[:index])
            if value > newest.get(case, (float("-inf"), ""))[0]:
                newest[case] = (value, part)
    return {case: name for case, (_value, name) in newest.items()}


def reason_to_skip_live(relative: str, newest: dict[str, str]) -> str | None:
    """Why a file stays on the instance during a *background* cycle.

    A third policy, between the two that existed. `--readable-only` is far too narrow
    for this -- it drops `constant/polyMesh`, every written time and every STL, which is
    exactly the data the hosted 3D viewer reads out of the mirror, so a cycle using it
    would blind the page. And `everything=True`, which is what the cycles used, asks a
    decomposed solve for every per-processor field file it is writing: one live run
    spent thirty-one minutes after the answer was already written moving 233 files, 156
    of them `processorN/` data that nothing in either repo will ever open, while a
    200 KB render the user was waiting for sat behind them.

    So: the mesh, the newest time, logs, postProcessing, dictionaries, images and
    surfaces all come over. What is dropped is only what is genuinely redundant --
    per-processor copies of fields that get reconstructed, superseded time directories,
    and formats written for a viewer that is not on this machine.
    """
    parts = relative.split("/")
    for index, part in enumerate(parts[:-1]):
        if _PROCESSOR_DIR.match(part):
            return "processor decomposition data"
        if part in LIVE_DROP_DIRS:
            return "written for a viewer that is not on this machine"
        if _is_time_directory(part):
            keep = newest.get("/".join(parts[:index]))
            if keep is not None and part != keep:
                return f"a superseded time directory ({part}/)"
    if _suffix(parts[-1]) in SKIP_SUFFIXES:
        return "written for a viewer that is not on this machine"
    return None


PRIORITY_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")


def _priority(entry) -> int:
    """What to ask for first. Lower goes sooner.

    Smallest-first was chosen so that one enormous log left behind is a sentence in the
    report rather than three hundred dictionaries missing -- sound, and it inverts under
    a live solve, because the smallest files in a decomposed case are precisely the
    per-processor ones nobody is waiting for. A render the model just made and looked at
    sat behind thousands of them: nine minutes for one picture, and a second that never
    arrived at all. Pictures and reports are what somebody is watching for, so they go
    in the first round trip and the bulk follows."""
    name = entry.path.rsplit("/", 1)[-1]
    if _suffix(name) in PRIORITY_SUFFIXES:
        return 0
    if name.startswith("log.") or _suffix(name) in (".md", ".txt", ".csv", ".json"):
        return 1
    return 2


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
            return PROCESSOR_DATA
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
    live: bool = False,
    background: bool = False,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    refused: dict[str, str] | None = None,
    gate: "Gate | None" = None,
) -> MirrorReport:
    """Copy down what has changed under the study's directory, and say what it did not.

    Returns a report whatever happens. Callers are sessions with work in flight, and
    none of them can afford an exception from a convenience.

    `background=True` is the unattended cycle on the clock below, as opposed to a sync
    somebody asked for. It tells the backend this listing is a poll, so that a session
    nobody is using stops looking busy -- see LiveMirror.

    `refused` is the caller's memory of paths the service has refused, path to reason.
    Read here so they are not asked for again, written here when one is refused. The
    background cycles share one across a session; `openreynolds pull` passes none and
    so asks again, which is what "tries them again" in the report promises.

    `gate` is where a session says a tool call is in flight; a cycle given one stands
    aside for it before each round trip (see GATE_WAIT_S).
    """
    started = time.monotonic()
    report = _sync(browser, path, everything, live, background, max_file_bytes,
                   max_total_bytes, refused, gate)
    report.seconds = time.monotonic() - started
    return report


def _sync(
    browser: Browser,
    path: str,
    everything: bool,
    live: bool,
    background: bool,
    max_file_bytes: int,
    max_total_bytes: int,
    refused: dict[str, str] | None,
    gate: "Gate | None",
) -> MirrorReport:
    store = browser.store
    if store is None:
        return MirrorReport(local_dir=Path.cwd(), warnings=["no study directory to mirror into"])

    report = MirrorReport(local_dir=store.fetch_dir(), study_id=store.session.study_id)
    root = path or browser.home or WORKSPACE_ROOT

    # The gate was already asked once, in `LiveMirror._run`, before this cycle was
    # let start -- but the listing below is itself a round trip (`browser.tree`,
    # `find` over the exec channel), and a tool call can arrive in the gap between
    # that first check and this one. Asked again here for the same reason `_pull_batch`
    # asks again before every archive request: the wait is cheap, and skipping it
    # is how a cycle that was clear to start ends up listing right through a call
    # that started a moment later.
    if gate is not None:
        report.gate_wait_seconds += _timed_clear(gate, GATE_WAIT_S)

    try:
        entries = browser.tree(root, depth=DEPTH, background=background)
    except (BackendError, OSError) as exc:
        if getattr(exc, "code", "") == "workspace_idle":
            # The expected steady state of an idle session, not a fault: there is no
            # workspace up because nothing has been asked of it, and this cycle
            # declined to start one. Saying so every twenty seconds for a night would
            # bury the warnings that mean something.
            return report
        report.warnings.append(f"could not look at {root}: {exc}")
        return report

    # The listing is the expensive part of a sync, and it is also exactly what a
    # files pane wants to draw. Remember it so showing the workspace is free.
    browser.remember(root, entries)

    # `find` output is capped, so the tail of a very large workspace was never examined
    # at all. That is a different thing from there being nothing there.
    #
    # Asked of the listing rather than counted here. `len(entries) >= MAX_ENTRIES` was a
    # guess from the outside and it was wrong at the boundary: a workspace holding
    # exactly MAX_ENTRIES entries is complete, and this called it truncated. `browse`
    # now asks `find` for one more line than it will keep, so the flag is measured. The
    # wording lives there too -- one sentence, in one place, rather than two that drift.
    if getattr(entries, "truncated", False):
        report.warnings.append(entries.notice)

    candidates = _wanted(
        entries, root, report, everything, max_file_bytes, live=live, refused=refused
    )
    _pull(
        browser, _within_budget(candidates, report, max_total_bytes), report,
        refused=refused, gate=gate,
    )
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
    live: bool = False,
    refused: dict[str, str] | None = None,
) -> list[Entry]:
    """Files worth asking for: not filtered out, not too big, not already here, and
    not refused by the service already this session."""
    wanted = []
    newest = newest_times(entries, root) if live else {}
    for entry in entries:
        if entry.is_dir:
            continue
        if refused and entry.path in refused:
            # Named and counted like every other file left behind. This is the
            # opposite of dropping it quietly: the path failed once, the reason is
            # here, and the retry is a command away.
            report.skipped.append(Skip(entry.path, refused[entry.path], entry.size))
            continue
        relative = _relative(entry.path, root)
        if live:
            reason = reason_to_skip_live(relative, newest)
        elif everything:
            reason = None
        else:
            reason = reason_to_skip(relative) or reason_by_size(relative, entry.size)
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
    # Which files come home is decided by the budget, smallest-first, so running out
    # still costs one enormous log rather than three hundred dictionaries. Which round
    # trip they come home *in* is decided here, and pictures go first: during a
    # decomposed solve the smallest files are the per-processor ones, so a 200 KB render
    # the model had just looked at sat behind thousands of them -- nine minutes for one
    # picture, and a second that never arrived before the session ended.
    for entry in sorted(wanted, key=lambda item: (_priority(item), item.size, item.path)):
        if current and (len(current) >= BATCH or carried + entry.size > BATCH_BYTES):
            batches.append(current)
            current, carried = [], 0
        current.append(entry)
        carried += entry.size
    if current:
        batches.append(current)
    return batches


def _pull(
    browser: Browser,
    wanted: list[Entry],
    report: MirrorReport,
    refused: dict[str, str] | None = None,
    gate: "Gate | None" = None,
) -> None:
    """Fetch in batches, and treat a failed batch as a fact rather than an end.

    A batch that fails is taken apart to find the file it failed on, and the rest
    come down without it. The usual cause is one awkward file in otherwise fine
    company, and losing the company with it is how a mirror comes back empty from a
    study that had plenty worth keeping -- which is exactly what a live run did.
    """
    for batch in _batches(wanted):
        _pull_batch(browser, batch, report, refused, gate)


def _pull_batch(
    browser: Browser,
    batch: list[Entry],
    report: MirrorReport,
    refused: dict[str, str] | None,
    gate: "Gate | None",
) -> bool:
    """One archive request, and what to do when it fails. True when it came down.

    The search for the file a batch failed on used to be every file on its own: a
    round trip each, and on a service where the awkward one costs two minutes and the
    ordinary ones four seconds, thirty files was six minutes -- then the same six
    minutes on the next cycle, because the awkward file never arrived and so was
    never "already here". Now it is: one at a time only until one fails, then the
    remainder as a batch again (it was fine without the file it failed on), and a
    file whose failure is a fact about the path is remembered and not asked for again
    this session. Which file it was is named, because the log never said.
    """
    # `gate is not None` is exactly "this cycle shares a container with a tool call
    # that may be running right now" (see `LiveMirror._cycle`). It used to mean
    # waiting for the gate before every round trip, on the reasoning that a slow
    # archive request here was a `bash` call stalled behind it -- true while building
    # the archive meant running inside that same container. `via="volume"` asks the
    # backend for a copy that does not (see `Backend.get_tree`, and `backend/hosted.py`
    # for what it actually does): there is nothing left to stand aside for, and
    # waiting anyway was pure delay -- up to GATE_WAIT_S of it, every batch, for a
    # wait that protected nothing. So a live cycle now goes straight to the round
    # trip. The bound stays regardless (`LIVE_PULL_TIMEOUT_S`, one attempt): the
    # backend does not promise this alternate path is always fast
    # (`qa-runs/LATENCY.md`), and this cycle still must not sit on one file for
    # minutes even when nothing else is waiting on it.
    report.round_trips += 1
    try:
        if gate is not None:
            written = browser.backend.get_tree(
                [entry.path for entry in batch], report.local_dir,
                timeout=LIVE_PULL_TIMEOUT_S, max_attempts=1, via="volume",
            )
        else:
            written = browser.backend.get_tree([entry.path for entry in batch], report.local_dir)
    except (BackendError, OSError) as exc:
        if len(batch) > 1:
            report.warnings.append(
                f"a batch of {len(batch)} failed ({exc}); looking for the file it failed on"
            )
            for index, entry in enumerate(batch):
                if _pull_batch(browser, [entry], report, refused, gate):
                    continue
                rest = batch[index + 1 :]
                if rest:
                    _pull_batch(browser, rest, report, refused, gate)
                return False
            return True
        entry = batch[0]
        report.warnings.append(f"could not copy {entry.path}: {exc}")
        report.skipped.append(Skip(entry.path, "the copy failed", entry.size))
        if refused is not None and is_a_fact_about_the_path(exc):
            refused[entry.path] = refusal(exc)
        return False
    report.pulled.extend(written)
    report.bytes_pulled += sum(entry.size for entry in batch)
    return True


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


# -- keeping it home while it happens ------------------------------------------


class Gate:
    """Where a session says a tool call is in flight, so the mirror stays out of its way.

    The session's thread holds it around each tool call; the mirror's thread asks
    `clear` before it starts a cycle and before its listing, and waits -- bounded by
    GATE_WAIT_S -- for the call to end (a live pull no longer asks at all; see
    `GATE_WAIT_S`'s docstring and `_pull_batch`). Nothing here ever blocks the
    session: the session only counts, and the counting is a lock held for
    nanoseconds.
    """

    def __init__(self) -> None:
        self._held = 0
        self._cond = threading.Condition()

    @contextmanager
    def held(self) -> Iterator[None]:
        self.enter()
        try:
            yield
        finally:
            self.exit()

    def enter(self) -> None:
        with self._cond:
            self._held += 1

    def exit(self) -> None:
        with self._cond:
            self._held = max(0, self._held - 1)
            self._cond.notify_all()

    @property
    def busy(self) -> bool:
        with self._cond:
            return self._held > 0

    def clear(self, timeout: float, unless: Callable[[], bool] | None = None) -> bool:
        """Wait until nothing is in flight. True if it cleared, False if `timeout`
        ran out first or `unless()` came true -- the mirror stopping, say, which a
        wait must not outlive."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while self._held:
                if unless is not None and unless():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                # In slices, so `unless` is looked at even when nobody notifies.
                self._cond.wait(min(remaining, 0.5))
            return True


def _timed_clear(gate: Gate, timeout: float) -> float:
    """`gate.clear`, and how long it took -- so a cycle can say how much of its own
    wall time was standing aside versus doing anything (`MirrorReport.gate_wait_seconds`,
    asked for directly in `qa-runs/LATENCY.md`: whether the Gate was actually the
    bottleneck, or just where the clock happened to be ticking, was not answerable
    from the report before this).

    `timeout` has no default on purpose: `GATE_WAIT_S` is read at the call site, not
    captured here, so a test (or a future caller) that changes it after this module
    loads is still honoured -- a default parameter value is bound once, at `def`."""
    started = time.monotonic()
    gate.clear(timeout)
    return time.monotonic() - started


class LiveMirror:
    """A background thread that runs `sync` for as long as the session lives.

    The turn-end sync arrives exactly when nothing is happening. A two-hour solve
    writes its fields, its logs and its renders while the model's turn is over and
    the session is just watching -- and none of it reached the user's machine until
    the session wound down. So the mirror runs on its own clock: every `interval_s`
    the study's directory is listed, whatever changed comes down, and whatever is
    showing the workspace is told. The user's copy is at most one interval behind
    the instance, all session long.

    One sync at a time, wherever it is asked from: the turn-end path and the timer
    share a lock, so two syncs can never race each other over the same files.
    Nothing here may end a session -- a cycle that fails becomes a report with a
    warning in it, and the next cycle tries again.
    """

    def __init__(self, browser: Browser, interval_s: float = 20.0):
        self.browser = browser
        self.interval_s = float(interval_s)
        self.view = None
        """Told about every cycle via `mirrored(report)`. Set once the session
        knows which interface it is running."""
        self.progress = None
        """Told when a cycle begins and ends, so the bar can say files are moving."""
        self.gallery = None
        """Surfaces and assembles renders from each cycle's arrivals (`delivery.py`).
        Set by the session; may be None. Runs on this thread, never raises."""
        self.last_report: MirrorReport | None = None
        self.refused: dict[str, str] = {}
        """Paths the service has refused this session, and its reason for each. Read
        and written by every cycle (see `sync`), so a path that costs two minutes and
        yields nothing costs it once. `openreynolds pull` does not consult it."""
        self.gate = Gate()
        """Held by the session around each tool call. Cycles stand aside for it."""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin the background cycles. An interval of zero or less means the
        feature is off, and only the explicit `sync_now` calls run."""
        if self.interval_s <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="mirror", daemon=True)
        self._thread.start()

    def poke(self) -> None:
        """Ask for a cycle now, without waiting for one and without blocking.

        For the moments the interval is too slow for: the model just rendered
        something and looked at it, and the user should not be twenty seconds
        behind a picture that already exists. A poke with no thread running is
        quietly nothing -- the turn-end syncs still cover that configuration."""
        self._wake.set()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the cycles and wait for any sync in flight to finish.

        Bounded, because an exit that hangs on a convenience is worse than a sync
        that gets cut off -- the session's own final sync still runs after this."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def sync_now(self) -> MirrorReport:
        """One sync, immediately, on the caller's thread. Serialized with the
        background cycles by the same lock.

        Foreground by definition: something asked for it, so the workspace is in use
        and may be started if it is not up."""
        return self._cycle()

    def catch_up(self) -> MirrorReport | None:
        """A sync soon, without holding up whoever asked.

        The turn-end sync used to run on the session thread, and the session thread
        is the one that reads what the user types. During a six-rank transient solve
        the cycle was pulling every per-processor field file the solver wrote -- an
        unbounded amount of work -- and for twenty-five minutes two typed messages
        sat in the queue behind it, unread, while the screen showed a finished turn.
        The user's word for it was that the model "just does not respond".

        So when the background thread is running, this only pokes it: the files come
        home on its clock, and the reply comes now. Without a thread (interval 0)
        the caller's sync is the only one, and it runs here as before."""
        if self._thread is not None and self._thread.is_alive():
            self.poke()
            return None
        return self.sync_now()

    def _run(self) -> None:
        while True:
            self._wake.wait(self.interval_s)
            self._wake.clear()
            if self._stop.is_set():
                return
            # Every cycle on this thread is a poll, including a poked one: whatever
            # provoked the poke was itself a command a moment ago, and that is what
            # keeps the workspace alive. This loop outlives the person -- it ran ~150
            # listings an hour for as long as a session process stayed open -- so if
            # it counted as use, a hosted workspace could never be reclaimed and
            # billed until its 24-hour ceiling. It does not count.
            #
            # And not while the model is in a tool call: the cycle's listing and
            # transfers run on the same container as the call, and the call waits
            # behind them. Bounded, so a busy turn still gets mirrored eventually.
            self.gate.clear(GATE_WAIT_S, unless=self._stop.is_set)
            if self._stop.is_set():
                return
            self._cycle(background=True)

    def _cycle(self, *, background: bool = False) -> MirrorReport:
        progress = self.progress
        with self._lock:
            if progress is not None:
                progress.sync_begin()
            try:
                report = sync(
                    self.browser, live=True, background=background,
                    refused=self.refused,
                    # Only an unattended cycle stands aside for tool calls. A sync
                    # somebody asked for -- the close-down one -- is the thing being
                    # waited on, and nothing is in flight to wait for.
                    gate=self.gate if background else None,
                )
            except Exception as exc:  # noqa: BLE001 - a convenience may not end a session
                report = MirrorReport(
                    local_dir=Path.cwd(), warnings=[f"could not mirror: {exc}"]
                )
            if progress is not None:
                progress.sync_end(report)
            if trace.on:
                trace.event(
                    "mirror", background=background, pulled=len(report.pulled),
                    bytes=report.bytes_pulled, skipped=len(report.skipped),
                    unchanged=report.unchanged, round_trips=report.round_trips,
                    seconds=round(report.seconds, 3),
                    gate_wait_seconds=round(report.gate_wait_seconds, 3),
                    warnings=list(report.warnings),
                )
            # Surface and assemble the pictures this cycle brought home. Inside the
            # lock is deliberate: it reads the same files the sync just wrote, and a
            # second sync must not move them mid-assembly.
            event = None
            if self.gallery is not None:
                event = self.gallery.ingest(report)
        self.last_report = report
        view = self.view
        if view is not None:
            try:
                view.mirrored(report)
                if event:
                    view.delivered(event)
            except Exception:  # noqa: BLE001 - presentation may be tearing down
                pass  # the files are already home; only the telling failed
        return report
