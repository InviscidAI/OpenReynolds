"""The case itself, captured at session end, so the study survives its workspace.

## Why this exists

The capture plane has always carried the *conversation*: 21,996 messages across 215
studies, essentially complete. It has never carried the *work*. Measured on production
2026-09-08: of 215 studies, 87 have any artifact at all (40%), 26 have a results
payload (12%), and **zero** have their case definition, their mesh, or their solver
logs. Every deliverable this product has made -- the mesh, the `Allrun` you edit to
re-run it, the log that says what happened -- exists in exactly one place, on a Volume
shared with every other study on the account, with no retention policy and no backup.

That is not a hypothetical exposure. On 2026-09-08 that Volume stood at 30.8 GB against
a 20 GB quota and a live study could not write; 24.5 GB of it was regenerable scratch
that nothing was responsible for clearing (F-56). The disk filling up is an outage. The
work product having no second copy is a single point of loss.

So this uploads the case at the end of a session, as one artifact against the study,
and it is worth doing whether or not the workspace ever becomes per-session -- it is
also the prerequisite for that, because a per-session workspace can only be thrown away
if `openreynolds --study <id>` can fetch the study back from somewhere.

## What goes in, and in what order

The bundle is built in tiers, cheapest and most irreplaceable first, and stops when the
next tier would not fit. Measured shares of a 6.4 GB post-prune Volume, and whether
each could be produced again:

| tier | share | rebuildable? |
|---|---|---|
| `definition` -- `system/`, `constant/` (not polyMesh, not geometry), `0/`, `0.orig/`, `Allrun`, `Allmesh`, `build.py` | 6.7% | **no** |
| `notes` -- `*.md`, `.reynolds/` | small | **no** |
| `record` -- `log.*`, and the tables in `postProcessing/` (forces, probes, residuals) | 6.4% | **no** |
| `geometry` -- `*.stl`, `*.obj`, `*.step`, wherever they sit | in `other` | only if the source is kept |
| `mesh` -- `constant/polyMesh` | 47.3% | yes, from `Allmesh` + geometry, at minutes of cost |
| `fields` -- the latest time directory | 16.8% | only by re-solving |
| `samples` -- the rest of `postProcessing/`: surfaces, planes and sets, one per write time | | only by re-solving |

`record` and `samples` may travel in part (#94). A study's record used to go in or out
whole, and one 190 MB solver log was enough to leave out every log while the mesh,
which `Allmesh` rebuilds, went in. Now, when `record` does not fit whole, a long file
keeps its first `CUT_HEAD` and last `CUT_TAIL` bytes -- the setup, and the final
residuals and forces -- with a line in the middle saying what was cut; tables go
before logs, the answer before the diagnosis. `notes` is ahead of `record` because a
cut record fills the room it is given, and notes are irreplaceable too. `samples` are
per-write-time snapshots, gigabytes of them on a long transient run, so they go last
and newest first, for as many as the room left allows. The manifest lists every cut
file and the report says how many samples were left out.

`definition` + `record` is 835 MB across all 215 studies -- **about 4 MB a study** -- so
the part that genuinely cannot be reproduced always fits. Everything above `mesh` is a
convenience that makes a resume instant instead of a rebuild.

## What it will not do

It will not fail or delay a session. Every path here returns a report; nothing raises
out of `build`, and the caller uploads through `Capture`, which is fire-and-forget by
design. A bundle that cannot be built is a warning and a study that captured what it
always captured.

It will not truncate silently. If a tier does not fit, the manifest says which and why,
the warning says it out loud, and the tiers that did fit are still uploaded. The one
thing worse than an incomplete backup is an incomplete backup that claims to be whole.
"""

from __future__ import annotations

import functools
import io
import json
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

CEILING_MB = 100
"""The most a bundle is ever compressed to, even for the tier that goes regardless.

Twice the service's artifact cap today. Past this the upload is refused anyway, so the
rest is compression nobody receives -- and a GB-scale `0/` of nonuniform fields would
otherwise be compressed in full, twice, at End (#82)."""

BUDGET_MB = 40
"""How large a bundle may get, compressed.

The service caps an artifact at `FOAMD_MAX_ARTIFACT_MB` (50 today) and there is no
route that reports the number, so this is deliberately under it rather than equal to
it. A client that guessed exactly would start failing the day an operator lowered the
cap, and the failure would be a 413 at the end of a session -- the worst possible
moment to discover a configuration mismatch."""

_TIME = re.compile(r"^\d+(\.\d+)?(e[-+]?\d+)?$", re.I)
_PROCESSOR = re.compile(r"^processor\d+$")

MEDIA_SUFFIXES = {".png", ".gif", ".mp4", ".webm", ".svg", ".pdf"}
GEOMETRY_SUFFIXES = {".stl", ".obj", ".step", ".stp", ".igs", ".iges"}
DEFINITION_NAMES = {
    "Allrun", "Allmesh", "Allclean", "Allpost", "Allbuild", "Allprep",
    "build.py", "build.sh", "cad.py", "make.py", "model.py", "prep.py",
    "case.foam",
}
"""The names a rebuild script may travel under.

Wider than `build.py` on purpose. A desk that writes its geometry as a script chooses
the filename, and a name nobody listed used to fall through every branch of
`classify()` and return `None` -- the artifact dropped silently, which is the one
outcome this module exists to prevent.

The known risk of a permissive set is the desk inventing a name that is still not in
it. That is closed on the other side rather than here: `cad/check.py` imports this set
and refuses to finish a run whose rebuild script is not captured by it, so an
unlisted name fails the run loudly instead of losing the artifact quietly."""
DEFINITION_DIRS = {"system", "0", "0.orig"}

TIERS = ("definition", "notes", "record", "geometry", "mesh", "fields", "samples")
"""Priority order. Everything before `mesh` is irreplaceable; `mesh`, `fields` and
`samples` are convenience -- a resume can rebuild the first and re-solve the others."""

PARTIAL = ("record", "samples")
"""The tiers that may travel in part. The others are whole or useless: half a mesh
or half a time directory restores nothing."""

SAMPLE_SUFFIXES = {".vtp", ".vtk", ".vtu", ".vtm", ".vtkhdf", ".ply", ".obj", ".stl",
                   ".x3d", ".case", ".geo", ".nas", ".png", ".jpg"}
"""Surface and image formats a function object writes into `postProcessing/`, also when
gzipped (`clip.vtp.gz`). Everything else there is a table -- forces, probes, residuals,
one row per time step -- unless it is written once per write time (`PER_TIME`).

A suffix list rather than a table list, because a probe's file is named after its
field: `p`, `U`, and on a multiphase case `alpha.water`, whose suffix is `.water`."""

PER_TIME = 20
"""A function object with more time directories than this writes once per write time
(sets, raw surfaces), so its files are samples. Tables are written into the start time's
directory, one more per restart: the 3.4 GB study in #94 has five under `forces/`, and
they hold its whole force history. A short run's sets written fewer times than this
travel as record, which costs room, not the answer."""

UNCUT_SUFFIXES = {".csv", ".json"}
"""Formats with no comment syntax, which a cut would leave unparseable. They travel
whole or not at all."""

CUT_HEAD = 64 * 1024
CUT_TAIL = 1024 * 1024
"""What a long log or table keeps when the record does not fit whole: its start (the
header, the case setup, the first iterations) and its end (the final residuals, the
converged forces, `End` or the error that stopped it). On the 3.4 GB study in #94 the
record came to 409 MB of logs, two of them 190 MB each."""


@dataclass
class Report:
    """What went in, what did not, and why -- said plainly enough to act on."""
    included: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    partial: dict[str, str] = field(default_factory=dict)
    """Tiers that went in part, and what was left out of each."""
    cut: dict[str, int] = field(default_factory=dict)
    """Files that travel as their head and tail, by name, with their full size."""
    files: int = 0
    bytes: int = 0
    oversized: bool = False
    """The irreplaceable tier alone went past the budget, so it was sent regardless
    and the service may refuse it."""
    error: str | None = None

    def brief(self) -> list[str]:
        if self.error:
            return [f"could not bundle this study's case: {self.error}"]
        if not self.included:
            return ["nothing to bundle: no case files reached this machine"]
        kb = self.bytes / 1024
        size = f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.0f} KB"
        lines = [f"case captured: {', '.join(self.included)} "
                 f"({self.files} file(s), {size})"]
        if self.oversized:
            lines.append("  this case is larger than an artifact may be; the upload "
                         "may be refused, and nothing else would have fitted either")
        for tier, what in self.partial.items():
            lines.append(f"  {tier} in part: {what}")
        for tier, why in self.skipped.items():
            lines.append(f"  {tier} left out: {why}")
        return lines


def _latest_time_dirs(root: Path) -> set[Path]:
    """The newest time directory of every case under `root`.

    A case is any directory holding a `system/`. `0` is excluded because it is an
    input, not a result, and it travels in `definition` anyway.
    """
    out: set[Path] = set()
    for system in root.rglob("system"):
        if not system.is_dir():
            continue
        case = system.parent
        best: tuple[float, Path] | None = None
        for child in case.iterdir():
            if not child.is_dir() or child.name in ("0", "0.orig"):
                continue
            if not _TIME.match(child.name):
                continue
            try:
                value = float(child.name)
            except ValueError:
                continue
            if best is None or value > best[0]:
                best = (value, child)
        if best is not None:
            out.add(best[1])
    return out


def _per_time(path: Path, counts: dict[Path, int]) -> bool:
    """Whether `path` sits in one of many time directories of a function object."""
    for parent in path.parents:
        if parent.name == "postProcessing":
            return False
        if _TIME.match(parent.name):
            holder = parent.parent
            if holder not in counts:
                try:
                    counts[holder] = sum(1 for c in holder.iterdir()
                                         if c.is_dir() and _TIME.match(c.name))
                except OSError:
                    counts[holder] = 0
            return counts[holder] > PER_TIME
    return False


def classify(path: Path, root: Path, latest: set[Path],
             counts: dict[Path, int] | None = None) -> str | None:
    """Which tier `path` belongs to, or None if it does not travel.

    Nothing regenerable-and-large goes: `processor*` is a decomposed copy of fields
    the reconstructed case already holds, and a superseded time directory is a state
    the run passed through. Both are what `toolbox/disk.py` prunes, and shipping them
    here would be re-uploading exactly the bytes that filled the disk.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    name = path.name
    suffix = path.suffix.lower()

    if any(_PROCESSOR.match(p) for p in parts):
        return None
    if "polyMesh" in parts:
        return "mesh"
    if name.startswith("log."):
        return "record"
    if "postProcessing" in parts:
        base = name[:-3] if name.lower().endswith(".gz") else name
        if Path(base).suffix.lower() in SAMPLE_SUFFIXES:
            return "samples"
        return "samples" if _per_time(path, {} if counts is None else counts) else "record"
    if suffix == ".md" or ".reynolds" in parts:
        return "notes"
    if name in DEFINITION_NAMES or any(p in DEFINITION_DIRS for p in parts[:-1]):
        return "definition"
    # Before `constant/`, and that order is the whole point. A CAD desk exports one
    # STL per patch into `constant/triSurface/`, and a chassis is twenty of them; on
    # the other order they were `definition` -- tier 0, which travels even when it is
    # oversized -- so a geometry set would inflate exactly the tier the design
    # guarantees will survive. Here they are `geometry`, where they are budgeted and
    # can be reported as skipped.
    if suffix in GEOMETRY_SUFFIXES:
        return "geometry"
    if "constant" in parts:
        return "definition"
    if suffix in MEDIA_SUFFIXES:
        # Renders already travel one by one through `Capture.artifact`, so they are
        # not repeated here; a figure that was never rendered through the tool is
        # picked up by `notes`/`definition` only if it sits with them.
        return None
    for parent in path.parents:
        if parent in latest:
            return "fields"
    if any(_TIME.match(p) for p in parts[:-1]):
        return None  # a superseded time directory
    return None


COMPRESSLEVEL = 6
"""gzip's own default rather than tarfile's 9. Measured 2026-11-10 on 364 MB of a real
study's logs and fields: level 9 took 13.8 s for 57 MB, level 6 took 7.1 s for 59 MB.
Half the time for 3% more bytes, at a moment somebody is waiting on it."""


class _TooBig(Exception):
    """The archive being written has already passed its limit."""


class _CappedBuffer(io.BytesIO):
    """A BytesIO that refuses to grow past `limit` bytes.

    gzip writes its output as it goes, so the moment the compressed archive crosses the
    budget is the moment there is no point compressing the rest of it. Without this a
    tier that could never fit was compressed in full before being thrown away: measured
    2026-11-10 on a 3.4 GB study, 133 of the bundle's 134 seconds went to a tier that
    was then left out, and on prod `close.capture_case` ran to ten minutes."""

    def __init__(self, limit: int | None) -> None:
        super().__init__()
        self.limit = limit

    def write(self, data) -> int:  # type: ignore[override]
        n = super().write(data)
        if self.limit is not None and self.tell() > self.limit:
            raise _TooBig()
        return n


@dataclass(frozen=True)
class _Cut:
    """A file that travels as its first `CUT_HEAD` and last `CUT_TAIL` bytes."""
    path: Path
    size: int

    @functools.cached_property
    def body(self) -> bytes:
        """Read once: it is packed in the fit and again in the bundle, and both have to
        be the same bytes."""
        with open(self.path, "rb") as f:
            head = f.read(CUT_HEAD)
            f.seek(max(self.size - CUT_TAIL, 0))
            tail = f.read(CUT_TAIL)
        # Whole lines either side of the cut, so a table still parses row by row.
        end = head.rfind(b"\n")
        head = head[:end + 1] if end >= 0 else head
        start = tail.find(b"\n")
        tail = tail[start + 1:] if start >= 0 else tail
        left_out = self.size - len(head) - len(tail)
        # `#` opens a comment in OpenFOAM's own tables, so the note does not become a row.
        note = (f"# [case bundle] {left_out} bytes of this file were left out here; "
                "the whole file is in the study's own copy\n").encode()
        return head + note + tail


def _cuttable(path: Path) -> _Cut | None:
    if path.suffix.lower() in UNCUT_SUFFIXES and not path.name.startswith("log."):
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return _Cut(path, size) if size > CUT_HEAD + CUT_TAIL + CUT_HEAD else None


Member = tuple["Path | _Cut", str]


def _add(tar: tarfile.TarFile, source: "Path | _Cut", arcname: str) -> None:
    try:
        if isinstance(source, _Cut):
            body = source.body
            info = tarfile.TarInfo(arcname)
            info.size = len(body)
            info.mtime = int(source.path.stat().st_mtime)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(body))
        else:
            tar.add(source, arcname=arcname, recursive=False)
    except OSError:
        pass  # vanished under us; a backup is not worth failing a session


def _fit(members: list[Member], room: int) -> tuple[int, int]:
    """How many of `members`, in order, fit in `room` compressed bytes, and about how
    many bytes those take.

    One streaming pass that stops at the first member taking the compressed size past
    `room` less a margin: the margin is what gzip may still hold unwritten, so the
    size returned is the bytes written plus that margin, an overestimate."""
    margin = min(1024 * 1024, room // 8)
    buf = _CappedBuffer(room)
    tar = tarfile.open(fileobj=buf, mode="w:gz", compresslevel=COMPRESSLEVEL)
    count, size = 0, 0
    try:
        for source, arcname in members:
            _add(tar, source, arcname)
            if buf.tell() > room - margin:
                break
            count, size = count + 1, buf.tell() + margin
    except _TooBig:
        pass
    finally:
        buf.limit = None
        tar.close()
    return count, size


def _write_time(arcname: str) -> float:
    """The simulation time a sample was written at: its last time-named directory
    (`postProcessing/planes/2.36/mid.vtp`), or -1 for none.

    Not the file's mtime, which on the local mirror is when the file was pulled, not
    when the solver wrote it, so one end-of-session sync gives every sample the same
    age in whatever order they arrived."""
    for part in reversed(arcname.split("/")[:-1]):
        if _TIME.match(part):
            try:
                return float(part)
            except ValueError:
                break
    return -1.0


def _size(member: Member) -> int:
    source = member[0]
    if isinstance(source, _Cut):
        return source.size
    try:
        return source.stat().st_size
    except OSError:
        return 0


def _mtime(member: Member) -> float:
    source = member[0]
    try:
        return (source.path if isinstance(source, _Cut) else source).stat().st_mtime
    except OSError:
        return 0.0


def _pack(members: list[Member], manifest_bytes: bytes | None = None,
          limit: int | None = None) -> bytes:
    """Pack `members` into a tar.gz; raise `_TooBig` as soon as it passes `limit`."""
    buf = _CappedBuffer(limit)
    tar = tarfile.open(fileobj=buf, mode="w:gz", compresslevel=COMPRESSLEVEL)
    try:
        if manifest_bytes is not None:
            # Inside the archive rather than in the artifact's `meta`, on purpose. It
            # then needs no API change to carry (the upload route takes a `meta` form
            # field the client does not send), it survives any transport that moves the
            # bytes, and whoever unpacks this on another machine reads it first without
            # having to ask the service anything.
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(manifest_bytes)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(manifest_bytes))
        for source, arcname in members:
            _add(tar, source, arcname)
    finally:
        # Closed on every way out, with the limit lifted first: closing only flushes
        # what gzip already holds and writes the trailer, and a stream left open --
        # abandoned at the limit, or interrupted by Ctrl+C -- is closed by the garbage
        # collector instead, which prints a traceback on the way out.
        buf.limit = None
        tar.close()
    out = buf.getvalue()
    if limit is not None and len(out) > limit:
        raise _TooBig()  # the trailer, written with the limit lifted, crossed it
    return out


def build(root: Path, budget_mb: int = BUDGET_MB,
          study_id: str = "") -> tuple[bytes | None, Report]:
    """Pack what is under `root`, tier by tier, until the next tier will not fit.

    `root` is the local mirror of the study (`store.fetch_dir()`), not the instance:
    the close-down sync has just brought it down, so this costs no network at all and
    cannot stall behind a workspace that is being torn down.

    Never raises. A failure is a Report with `error` set.
    """
    report = Report()
    try:
        if not root.is_dir():
            report.error = f"{root} is not a directory"
            return None, report

        latest = _latest_time_dirs(root)
        counts: dict[Path, int] = {}
        by_tier: dict[str, list[Member]] = {t: [] for t in TIERS}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            tier = classify(path, root, latest, counts)
            if tier is None:
                continue
            by_tier[tier].append((path, str(path.relative_to(root)).replace("\\", "/")))

        budget = budget_mb * 1024 * 1024
        ceiling = max(CEILING_MB, budget_mb) * 1024 * 1024
        taken: list[Member] = []
        used = 0
        # Each tier is packed on its own against the room the earlier ones left, and
        # the bundle is packed once at the end. Packing `taken + tier` for every tier
        # re-compressed everything taken so far once per tier: with the record in, that
        # was 40 MB again for each of six tiers. gzip's window is 32 KB, so an archive
        # of the whole compresses to the sum of its parts, give or take that much.
        for tier in TIERS:
            members = by_tier[tier]
            if not members:
                continue
            room = budget - used
            # Only a first tier that cannot be cut may go past the budget (below), and
            # not past the ceiling: beyond that the service refuses it anyway.
            first = not taken and tier not in PARTIAL
            size: int | None = None
            if tier != "samples" and (first or room > 0):  # samples: newest first, always
                try:
                    size = len(_pack(members, limit=ceiling if first else room))
                except _TooBig:
                    pass
            if size is None:
                part = _in_part(tier, members, room, report)
                if part is None:
                    report.skipped[tier] = (
                        f"{len(members)} file(s) come to more than {CEILING_MB} MB on "
                        "their own, more than an upload may be" if first else
                        f"{len(members)} file(s) would take the bundle past {budget_mb} MB"
                    )
                    continue
                members, size = part
            if size > budget and first:
                # The first tier is the irreplaceable one and it does not fit. It goes
                # anyway, and the report says the upload may be refused.
                #
                # The alternative was tried and is worse: skipping it ships nothing at
                # all, so a study whose definition and logs are too big to bundle is
                # exactly the study that gets no second copy -- silently, since every
                # later tier is larger and also skipped. Sending it means the service
                # either accepts it or answers 413, and either way somebody learns
                # something. `definition` + `record` measured about 4 MB a study across
                # 215 production studies, so this is the rare case by construction.
                report.oversized = True
            taken += members
            used += size
            report.included.append(tier)

        if not taken:
            return None, report
        report.files = len(taken)
        # Packed once more with the manifest in it, because what the manifest says --
        # which tiers made it and which were dropped -- is not known until every tier
        # has been tried. `report.bytes` is the size that actually ships.
        blob = _pack(taken, manifest(report, study_id).encode("utf-8"))
        report.bytes = len(blob)
        return blob, report
    except Exception as exc:  # noqa: BLE001 - a backup may never end a session
        report.error = f"{type(exc).__name__}: {exc}"
        return None, report


def _in_part(tier: str, members: list[Member], room: int,
             report: Report) -> tuple[list[Member], int] | None:
    """As much of a tier that does not fit whole as fits in `room`, with about how
    many compressed bytes it takes, or None for nothing.

    `record` first cuts every long file to its head and tail, tables before logs;
    `samples` go newest first. Then as many as fit are taken, in that order."""
    if tier not in PARTIAL or room <= 0:
        return None
    if tier == "record":
        ordered = [(_cuttable(p) or p, a) for p, a in members]
        # Tables first: the forces and residuals are the answer, and small. Then logs,
        # largest first, which puts the solver's ahead of blockMesh's. Not by mtime,
        # which on the mirror is when a file was pulled.
        ordered.sort(key=lambda m: (m[1].rsplit("/", 1)[-1].startswith("log."),
                                    -_size(m)))
    else:
        ordered = sorted(members, key=lambda m: (-_write_time(m[1]), -_mtime(m)))
    count, size = _fit(ordered, room)
    if count == 0:
        return None
    part = ordered[:count]
    cut = {a: s.size for s, a in part if isinstance(s, _Cut)}
    report.cut.update(cut)
    said = []
    if cut:
        said.append(f"{len(cut)} long file(s) kept as their first {CUT_HEAD // 1024} KB "
                    f"and last {CUT_TAIL // 1024 // 1024} MB")
    if count < len(ordered):
        older = " (the oldest)" if tier == "samples" else ""
        said.append(f"{len(ordered) - count} of {len(ordered)} file(s) left out{older}")
    if said:
        report.partial[tier] = "; ".join(said)
    return part, size


def manifest(report: Report, study_id: str) -> str:
    """What the artifact's `meta` carries, so the bundle describes itself.

    A reader on another machine has to be able to tell a complete capture from one
    that dropped its mesh, without unpacking it and guessing.
    """
    # No size field, deliberately: the manifest travels *inside* the archive, so a
    # byte count written here would be a count of something that does not exist yet.
    # Whoever holds the file can measure it; nobody can measure it from in here.
    return json.dumps({
        "study_id": study_id,
        "tiers": report.included,
        "skipped": report.skipped,
        "partial": report.partial,
        "cut": report.cut,
        "files": report.files,
        "complete": not report.skipped and not report.partial,
        "over_budget": report.oversized,
    })
