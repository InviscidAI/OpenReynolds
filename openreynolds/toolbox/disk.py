#!/usr/bin/env python3
"""What is filling the workspace, and what of it is safe to delete.

There is one disk. An account is capped at one instance, `acquire()` joins the
existing one rather than making a second, and one instance is one Modal Volume
mounted at `/work` -- so every study the account has ever run is a sibling
directory in it. The quota (`FOAMD_VOLUME_QUOTA_GB`, 20 GB) is measured over the
whole Volume by `du -sm -H /work`, not per study. So a study that is 24 MB can be
refused because an animation run three weeks ago left 3 GB of frames next door,
and nothing in the product ever cleaned any of it up.

That is what this exists for. It happened on 2026-09-08: a session with a 24 MB
case could not write solver output, `/work` was at 31.5 GB against the 20 GB
quota, and the six biggest directories were old studies of the same account. The
agent stopped and asked rather than delete what it could not prove was safe --
the right call, and an expensive one, because there was nothing it could read to
find out.

It got that bad quietly. `du -sm -- /work` measures the *symlink*: Modal mounts
the Volume elsewhere and leaves `/work` as a link to `/__modal/volumes/vo-<id>`.
So the quota check answered 1 MB for every instance on every call from the day it
was written until `-H` was added on 2026-09-06, `volume_used_mb` was persisted as
1, and the 507 it exists to raise was never once raised. Measured on the live
service: `du -sm /work` -> 1 MB and `du -sLm /work` -> 28,633 MB, the same
Sandbox, the same moment.

## What it will and will not delete

The distinction this script is built around is **regenerable** against
**irreplaceable**, and it is deliberately conservative about which is which.

Regenerable -- it can be produced again by re-running, from inputs that survive:

    processor*/          decomposed copies of the same fields
    <time>/              solution time directories, except the latest; on one that
                         carries a <time>/polyMesh of its own the fields go one by
                         one and the mesh subtree beside them stays
    VTK/, *.vtk, *.vtu   post-processing dumps
    frames/, *.ppm       animation frames (the encoded video is not touched)
    .foam scratch        `.foamd/`, `.reynolds/tmp/`

Irreplaceable -- never touched, whatever flag you pass:

    0/, 0.orig/          the initial and boundary conditions
    constant/, system/   the case definition, including polyMesh
    <time>/polyMesh/     a moving mesh's deformed mesh at that instant: the only
                         record the run keeps of its own motion
    log.*                the record of what happened; often the only evidence left
    postProcessing/      measured results, usually kilobytes
    *.png *.gif *.mp4    figures and finished animations
    *.md *.py *.stl      reports, scripts, geometry
    Allrun Allmesh       the "change a number and re-run" files the desk leaves

`report` is always safe and changes nothing. `prune` prints what it would remove
and removes nothing unless `--apply` is passed, and it refuses to touch a study
other than the one you name unless you name that one too. Nothing here deletes a
whole study directory: removing a study is a person's decision, and this tool's
job is to make it unnecessary.

    python3 disk.py report                     # every study on the volume, biggest first
    python3 disk.py report --study <id>        # one study, broken down
    python3 disk.py prune --study <id>         # what could go, and how much it would free
    python3 disk.py prune --study <id> --apply # actually remove it
    python3 disk.py prune --all --keep-latest  # every study, regenerable output only
    python3 disk.py report --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

WORK = os.environ.get("OPENREYNOLDS_WORK", "/work")

# Directory names that are never regenerable, checked as whole path components.
# `polyMesh` is in here for the moving-mesh case: on a case whose mesh deforms, the
# deformed mesh is written to `<time>/polyMesh/points`, and that is the only record the
# run keeps of its own motion -- there is nothing to regenerate it from short of
# re-running the solve. `constant/polyMesh` was already covered by `constant`; this
# covers the time-directory copies as well.
KEEP_DIRS = {"0", "0.orig", "constant", "system", "postProcessing", "notes", "references",
             "polyMesh"}
# Suffixes that are always kept, whatever directory they are in.
KEEP_SUFFIXES = {".png", ".gif", ".mp4", ".webm", ".md", ".py", ".sh", ".stl", ".obj",
                 ".step", ".stp", ".csv", ".json", ".pdf", ".svg"}
# Filenames kept whatever their suffix -- the editable record the desk leaves behind.
KEEP_NAMES = {"Allrun", "Allmesh", "Allclean", "build.py", "case.foam"}

REGENERABLE_DIRS = {"VTK", "frames", ".foamd"}
REGENERABLE_SUFFIXES = {".vtk", ".vtu", ".vtp", ".ppm", ".pvd"}

_TIME_DIR = re.compile(r"^\d+(\.\d+)?(e[-+]?\d+)?$", re.I)
_PROCESSOR_DIR = re.compile(r"^processor\d+$")


def _is_time_dir(name: str) -> bool:
    """A solution time directory: `0.5`, `1200`, `1e-05`.

    `0` and `0.orig` match the pattern and are excluded by KEEP_DIRS before this is
    ever consulted -- the initial condition is an input, not an output, and deleting
    it is how a case that could have been re-run becomes one that cannot.
    """
    return bool(_TIME_DIR.match(name))


def _tree_bytes(path: Path) -> int:
    """Bytes under `path`, following nothing.

    `os.walk(followlinks=False)` on purpose, and it is the same trap the quota check
    fell into from the other side: an OpenFOAM case tree is full of internal links
    (`0` to `0.orig`, `constant/polyMesh` into a mesh directory, `processor*`
    scratch), and following them counts the same blocks once per link. Here that
    would report a directory as bigger than it is and invite deleting the wrong one.
    """
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = Path(root) / name
            try:
                if fp.is_symlink():
                    continue
                total += fp.stat().st_size
            except OSError:
                continue  # vanished under us, or unreadable; not worth failing a report
    return total


def _human(n: int) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


@dataclass
class Candidate:
    """One thing that could be removed, and what removing it would free."""
    path: Path
    bytes: int
    why: str


@dataclass
class StudyUsage:
    name: str
    path: Path
    bytes: int
    regenerable: int = 0
    candidates: list[Candidate] = field(default_factory=list)


def _keep(rel_parts: tuple[str, ...], name: str) -> bool:
    """Whether a path is in the never-touch set."""
    if name in KEEP_NAMES or Path(name).suffix.lower() in KEEP_SUFFIXES:
        return True
    return any(part in KEEP_DIRS for part in rel_parts)


def candidates_in(case: Path, keep_latest: bool = True) -> list[Candidate]:
    """What could be removed from one case directory, and why.

    `keep_latest` keeps the newest time directory, which is what a restart resumes
    from (`scratch.py` reads `latestTime`) and what a render reads. Passing it False
    is for a case whose results are already extracted and whose fields nobody will
    look at again; it is not the default because "I already have the numbers" is a
    belief, and a time directory is the only thing that can prove it wrong.

    A time directory that holds a `polyMesh` of its own keeps that *subtree* whatever
    `keep_latest` says, because on a moving mesh it is the mesh at that instant and there
    is nothing left to regenerate it from. The fields beside it are offered one by one
    instead. Skipping the whole directory was the first shape of this rule, and the
    review of 2026-09-12 measured what it costs: OpenFOAM writes `<time>/polyMesh/points`
    at every write of a morphing-mesh run, so every written time qualified and a case
    shaped like the Wigley free phase reported 2.7 GB used and 0 B regenerable, in both
    modes -- "nothing regenerable found" to the one situation (F-56, /work at 31.5 GB
    against a 20 GB quota) this module exists to answer.
    """
    out: list[Candidate] = []
    if not case.is_dir():
        return out

    times: list[tuple[float, Path]] = []
    for child in sorted(case.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        name = child.name
        if name in KEEP_DIRS:
            continue
        if _PROCESSOR_DIR.match(name):
            out.append(Candidate(child, _tree_bytes(child),
                                 "a decomposed copy of fields the reconstructed case holds"))
            continue
        if name in REGENERABLE_DIRS:
            out.append(Candidate(child, _tree_bytes(child),
                                 "post-processing output, rebuilt by re-running the writer"))
            continue
        if _is_time_dir(name):
            try:
                times.append((float(name), child))
            except ValueError:
                pass

    times.sort()
    droppable = times[:-1] if (keep_latest and times) else times
    for value, path in droppable:
        # A time directory that carries its own polyMesh keeps that subtree and gives up
        # everything else. On a moving mesh -- a hull released in heave and pitch, a
        # rotor on a sliding interface -- `<time>/polyMesh/points` IS the mesh at that
        # instant and nothing regenerates it; the fields written beside it (U, p_rgh,
        # alpha.water, k, omega, nut, phi) are the ordinary regenerable output this
        # module was written to reclaim, and on such a run they are the bulk of the
        # volume. Keeping the whole directory instead, as this did until 2026-09-12,
        # freed nothing at all on the case that most needs it. The reason line names the
        # test each time passed, so a prune that frees less than expected says why
        # without anyone reading this.
        if (path / "polyMesh").is_dir():
            for child in sorted(path.iterdir()):
                if child.is_symlink() or _keep((child.name,), child.name):
                    continue  # `polyMesh` is in KEEP_DIRS, so this is where it survives
                try:
                    size = _tree_bytes(child) if child.is_dir() else child.stat().st_size
                except OSError:
                    continue  # vanished under us; not worth failing a prune over
                out.append(Candidate(child, size,
                                     f"solution time {value:g}, not the latest; the "
                                     "deformed mesh beside it is kept"))
            continue
        out.append(Candidate(path, _tree_bytes(path),
                             f"solution time {value:g}, not the latest, and carries no "
                             "mesh of its own"))

    # Anything already covered by a directory above must not be listed again. Two
    # candidates for the same bytes double-counts what a prune would free, and the
    # second removal then fails because the first took its parent with it -- which is
    # exactly what `VTK/` plus `VTK/case_100.vtu` did the first time this ran.
    claimed = [c.path for c in out]

    def already_going(path: Path) -> bool:
        return any(path == c or c in path.parents for c in claimed)

    for root, dirs, files in os.walk(case, followlinks=False):
        here = Path(root)
        rel = here.relative_to(case).parts
        if any(part in KEEP_DIRS for part in rel) or already_going(here):
            dirs[:] = []
            continue
        for name in files:
            if _keep(rel, name):
                continue
            if Path(name).suffix.lower() in REGENERABLE_SUFFIXES:
                fp = here / name
                if fp.is_symlink() or already_going(fp):
                    continue
                try:
                    out.append(Candidate(fp, fp.stat().st_size, "a post-processing dump"))
                except OSError:
                    pass
    return out


def scan(work: Path, study: str | None = None, keep_latest: bool = True) -> list[StudyUsage]:
    """Every study directory under `work`, or just one, with what could be freed."""
    if not work.is_dir():
        return []
    roots = [work / study] if study else [
        p for p in sorted(work.iterdir()) if p.is_dir() and not p.is_symlink()
    ]
    out: list[StudyUsage] = []
    for root in roots:
        if not root.is_dir():
            continue
        usage = StudyUsage(name=root.name, path=root, bytes=_tree_bytes(root))
        # A study holds cases; a case is any directory with a `system/`, and a study
        # that IS a case (the flat layout the older studies use) counts as its own.
        cases = [c for c in sorted(root.iterdir()) if c.is_dir() and (c / "system").is_dir()]
        if (root / "system").is_dir():
            cases.append(root)
        for case in cases:
            usage.candidates.extend(candidates_in(case, keep_latest=keep_latest))
        usage.regenerable = sum(c.bytes for c in usage.candidates)
        out.append(usage)
    out.sort(key=lambda u: u.bytes, reverse=True)
    return out


def render_report(usages: list[StudyUsage], work: Path, quota_gb: int | None) -> str:
    total = sum(u.bytes for u in usages)
    free = sum(u.regenerable for u in usages)
    lines = [f"{work}: {_human(total)} in {len(usages)} director(ies)"]
    if quota_gb:
        pct = 100.0 * total / (quota_gb * (1 << 30))
        lines.append(f"quota {quota_gb} GB -- {pct:.0f}% used")
    lines.append(f"regenerable, and removable with `prune`: {_human(free)}")
    lines.append("")
    lines.append(f"{'study':<32} {'total':>10} {'regenerable':>12}")
    for u in usages:
        lines.append(f"{u.name[:32]:<32} {_human(u.bytes):>10} {_human(u.regenerable):>12}")
    if not usages:
        lines.append("(nothing here)")
    return "\n".join(lines)


def render_prune(usages: list[StudyUsage], apply: bool) -> str:
    lines = []
    freed = 0
    for u in usages:
        if not u.candidates:
            continue
        lines.append(f"{u.name}:")
        for c in sorted(u.candidates, key=lambda c: c.bytes, reverse=True):
            lines.append(f"  {_human(c.bytes):>10}  {c.path}  -- {c.why}")
            freed += c.bytes
    if not lines:
        return "nothing regenerable found; everything here is input, log or result"
    head = (f"removing {_human(freed)}" if apply
            else f"would remove {_human(freed)} -- pass --apply to do it")
    return head + "\n" + "\n".join(lines)


def prune(usages: list[StudyUsage]) -> tuple[int, list[str]]:
    """Remove every candidate. Returns (bytes freed, failures)."""
    freed, failed = 0, []
    for u in usages:
        for c in u.candidates:
            try:
                if c.path.is_dir() and not c.path.is_symlink():
                    shutil.rmtree(c.path)
                else:
                    c.path.unlink()
                freed += c.bytes
            except OSError as error:
                failed.append(f"{c.path}: {error}")
    return freed, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="what is on the volume, biggest first")
    rep.add_argument("--work", default=WORK)
    rep.add_argument("--study", default=None)
    rep.add_argument("--quota-gb", type=int, default=20)
    rep.add_argument("--json", action="store_true")

    pru = sub.add_parser("prune", help="remove regenerable output (dry run by default)")
    pru.add_argument("--work", default=WORK)
    pru.add_argument("--study", default=None,
                     help="the study to prune; required unless --all is given")
    pru.add_argument("--all", action="store_true",
                     help="every study on the volume, not just one")
    pru.add_argument("--keep-latest", dest="keep_latest", action="store_true", default=True)
    pru.add_argument("--drop-latest", dest="keep_latest", action="store_false",
                     help="also remove the newest time directory; a restart cannot resume after this")
    pru.add_argument("--apply", action="store_true", help="actually remove it")
    pru.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    work = Path(args.work)

    if args.cmd == "report":
        usages = scan(work, study=args.study)
        if args.json:
            print(json.dumps({
                "work": str(work),
                "total_bytes": sum(u.bytes for u in usages),
                "regenerable_bytes": sum(u.regenerable for u in usages),
                "studies": [{"name": u.name, "bytes": u.bytes,
                             "regenerable_bytes": u.regenerable} for u in usages],
            }, indent=2))
        else:
            print(render_report(usages, work, args.quota_gb))
        return 0

    # prune
    if not args.study and not args.all:
        # Naming the study is the guard. One volume holds every study the account has,
        # and a prune that defaults to all of them is one keystroke from removing the
        # time directories of a solve running next door.
        print("say which study to prune (--study <id>), or --all for every study",
              file=sys.stderr)
        return 2
    usages = scan(work, study=args.study, keep_latest=args.keep_latest)
    if args.apply:
        freed, failed = prune(usages)
        print(f"freed {_human(freed)}")
        for line in failed:
            print("could not remove", line, file=sys.stderr)
        return 1 if failed else 0
    if args.json:
        print(json.dumps([{"path": str(c.path), "bytes": c.bytes, "why": c.why}
                          for u in usages for c in u.candidates], indent=2))
    else:
        print(render_prune(usages, apply=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
