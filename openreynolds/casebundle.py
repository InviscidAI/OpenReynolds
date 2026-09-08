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
| `definition` -- `system/`, `constant/` (not polyMesh), `0/`, `0.orig/`, `Allrun`, `Allmesh`, `build.py` | 6.7% | **no** |
| `record` -- `log.*`, `postProcessing/` | 6.4% | **no** |
| `notes` -- `*.md`, `.reynolds/` | small | **no** |
| `geometry` -- `*.stl`, `*.obj`, `*.step` | in `other` | only if the source is kept |
| `mesh` -- `constant/polyMesh` | 47.3% | yes, from `Allmesh` + geometry, at minutes of cost |
| `fields` -- the latest time directory | 16.8% | only by re-solving |

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

import io
import json
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

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
DEFINITION_NAMES = {"Allrun", "Allmesh", "Allclean", "Allpost", "build.py", "case.foam"}
DEFINITION_DIRS = {"system", "0", "0.orig"}

TIERS = ("definition", "record", "notes", "geometry", "mesh", "fields")
"""Priority order. Everything before `mesh` is irreplaceable; `mesh` and `fields` are
convenience -- a resume can rebuild the first and re-solve the second."""


@dataclass
class Report:
    """What went in, what did not, and why -- said plainly enough to act on."""
    included: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
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


def classify(path: Path, root: Path, latest: set[Path]) -> str | None:
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
    if name.startswith("log.") or "postProcessing" in parts:
        return "record"
    if suffix == ".md" or ".reynolds" in parts:
        return "notes"
    if name in DEFINITION_NAMES or any(p in DEFINITION_DIRS for p in parts[:-1]):
        return "definition"
    if "constant" in parts:
        return "definition"
    if suffix in GEOMETRY_SUFFIXES:
        return "geometry"
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


def _pack(members: list[tuple[Path, str]], manifest_bytes: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
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
        for path, arcname in members:
            try:
                tar.add(path, arcname=arcname, recursive=False)
            except OSError:
                continue  # vanished under us; a backup is not worth failing a session
    return buf.getvalue()


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
        by_tier: dict[str, list[tuple[Path, str]]] = {t: [] for t in TIERS}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            tier = classify(path, root, latest)
            if tier is None:
                continue
            by_tier[tier].append((path, str(path.relative_to(root)).replace("\\", "/")))

        budget = budget_mb * 1024 * 1024
        taken: list[tuple[Path, str]] = []
        blob = b""
        for tier in TIERS:
            members = by_tier[tier]
            if not members:
                continue
            candidate = _pack(taken + members)
            if len(candidate) > budget and taken:
                report.skipped[tier] = (
                    f"{len(members)} file(s) would take the bundle past {budget_mb} MB"
                )
                continue
            if len(candidate) > budget and not taken:
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
                blob = candidate
                report.included.append(tier)
                continue
            taken += members
            blob = candidate
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
        "files": report.files,
        "complete": not report.skipped,
        "over_budget": report.oversized,
    })
