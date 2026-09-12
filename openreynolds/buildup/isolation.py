"""Absence, enforced before a run and verified after it.

The hand-off's §3, and it exists because not naming a tool turned out not to withhold
it. The bare arm of the last round was given no toolbox and found one anyway: it
located `/work/.toolbox` through a symlink left by an earlier arm, read `cad_convert.py`
with `sed` and `grep` to recover a function signature, and `cat`-ed a copy-modify-run
template the brief no longer mentions. A `find /` for a house filename turned up
leftovers from unrelated runs. Every number that arm produced was measuring a desk with
tools against a desk with tools.

So there are two halves here and neither is sufficient alone:

* **before** -- a workspace root that no other run has used, with nothing of ours
  reachable beneath it and no well-known house path resolving anywhere. A failure here
  aborts rather than runs, because a run started dirty cannot be cleaned up afterwards;
* **after** -- the cell log and every captured output grepped for house filenames, the
  toolbox directory name and the repo path. A run that touched one is contaminated,
  recorded as such, and **discarded from the baseline** rather than averaged in or
  quietly retried.

The same discipline runs in reverse for an arm that is supposed to have the tools: a run
that failed to reach one it was given is invalid for the same reason, which is what
`missing` in `Contamination` is for.

**A basename is a blunt instrument, deliberately.** `render.py` is a house filename and
also a plausible thing for a desk to write; a hit on it may be the desk's own file. The
verdict stays mechanical anyway -- contaminated is contaminated -- because the
alternative is a judgement call inside the thing doing the measuring, and the evidence
line travels with every hit so a reader can see which it was.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
TOOLBOX_DIR = REPO / "openreynolds" / "toolbox"

TOOLBOX_NAME = ".toolbox"
"""What `cli.py` calls the directory it syncs. The name is itself a house surface: a
desk that greps for it has found us whether or not the directory is there."""

WELL_KNOWN = (
    "/work/.toolbox",
    "/work/.probes",
    "/work/.accept",
    "/work/.wheels",
    "/work/.pydeps",
)
"""Paths the brief, the finish check or a past harness has named out loud.

`/work` itself is not on the list and that is a deviation from the hand-off's letter,
taken on its reason: the hazard measured was a *reachable house surface*, not a
directory called `/work` -- the hosted image has one by definition, and the local arm
was rooted there on purpose. Each entry above is checked for existence and for where it
resolves, which is what caught the stale symlink."""

GENERIC = frozenset({"README.md", "ENVIRONMENT.md", "__init__.py"})
"""House filenames too ordinary to be evidence of anything.

A run that writes `README.md` into its case directory has not found us, and a grep that
says it has costs a good run for nothing. Everything else in the toolbox is named
distinctly enough that a sighting means what it looks like."""

WALK_LIMIT = 40_000
"""Entries the pre-run walk will look at before it gives up and says so. A workspace
with more than this in it is not a fresh workspace, and the check says that instead of
spending a minute proving it."""


class Dirty(Exception):
    """The environment is not clean enough to measure in. Raised, never returned.

    A warning here would be read past. The last round's isolation was assumed rather
    than asserted, and the assumption held for exactly as long as nobody looked."""


def house_names(toolbox: Path | None = None) -> tuple[str, ...]:
    """Every filename a run must not be seen touching, read off the toolbox as it is.

    Off disk rather than written down, so a tool added next week is covered by the grep
    without anyone remembering to add it here."""
    directory = Path(toolbox or TOOLBOX_DIR)
    names = {TOOLBOX_NAME}
    if directory.is_dir():
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md"):
                if path.name not in GENERIC:
                    names.add(path.name)
    return tuple(sorted(names))


@dataclass(frozen=True)
class Hit:
    """One sighting: what was seen, where, and the line it was seen on."""

    name: str
    where: str
    line: int
    evidence: str


@dataclass
class Contamination:
    """The after-the-run verdict. `contaminated` is the only field with authority."""

    contaminated: bool = False
    hits: list[Hit] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    """For a tooled arm: house names it was given and never reached. Empty otherwise."""

    def as_dict(self) -> dict[str, object]:
        return {
            "contaminated": bool(self.contaminated),
            "hits": [vars(hit) for hit in self.hits],
            "missing": list(self.missing),
        }

    def lines(self) -> list[str]:
        if not self.contaminated:
            return ["no house surface appears in the run's own output"]
        return [f"{hit.where}:{hit.line}: {hit.name} -- {hit.evidence}" for hit in self.hits]


# -- before the run ------------------------------------------------------------


def fresh_workspace(parent: Path, case: str, run_id: str) -> Path:
    """A workspace root nothing else has used, made empty, and refused if it is not.

    One per case, never reused. A directory another arm has worked in carries that
    arm's exports, its checkpoints and its `.toolbox`, and the second run reads them as
    its own starting state."""
    root = Path(parent).expanduser().resolve() / f"{case}-{run_id}"
    if root.exists() and any(root.iterdir()):
        raise Dirty(
            f"{root} already exists and is not empty. A workspace is per case and per "
            "run: reusing one hands the next arm the last arm's files.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def preflight(workspace: Path, *, toolbox: Path | None = None,
              well_known: Iterable[str] | None = None) -> dict[str, object]:
    """Assert the run can start clean, or raise `Dirty` saying what is reachable.

    Three questions, in the order that makes the cheapest one first: does a well-known
    house path resolve; is anything of ours beneath the workspace; does a symlink under
    the workspace leave it for the repo."""
    root = Path(workspace).expanduser().resolve()
    # Read off the module at call time, not bound as a default: the list is what this
    # environment counts as a house path, and a test -- or an operator who has cleaned one
    # up -- has to be able to say so without editing the signature.
    well_known = WELL_KNOWN if well_known is None else well_known
    names = set(house_names(toolbox))
    repo = REPO

    reachable: list[str] = []
    for candidate in well_known:
        path = Path(candidate)
        if path.exists() or path.is_symlink():
            target = _resolve(path)
            reachable.append(f"{candidate} -> {target}")

    found: list[str] = []
    escapes: list[str] = []
    seen = 0
    truncated = False
    for current, directories, files in os.walk(root, followlinks=False):
        for entry in list(directories) + list(files):
            seen += 1
            if seen > WALK_LIMIT:
                truncated = True
                break
            path = Path(current) / entry
            if entry in names:
                found.append(str(path))
            if path.is_symlink():
                target = _resolve(path)
                if _under(target, repo) or target.name in names:
                    escapes.append(f"{path} -> {target}")
        if truncated:
            break

    if truncated:
        raise Dirty(
            f"{root} holds more than {WALK_LIMIT:,} entries; a fresh workspace does "
            "not, so this one is not fresh and was not searched to the end.")
    if reachable or found or escapes:
        raise Dirty("the environment is not clean: " + "; ".join(
            [*(f"{item} resolves" for item in reachable),
             *(f"{item} is a house file under the workspace" for item in found),
             *(f"{item} leaves the workspace" for item in escapes)]))
    return {"workspace": str(root), "entries": seen, "house_names": len(names),
            "well_known_checked": list(well_known)}


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


# -- after the run -------------------------------------------------------------


def scan(texts: Mapping[str, str], *, toolbox: Path | None = None,
         expected: Iterable[str] = ()) -> Contamination:
    """Grep the run's own words for house surfaces. `texts` is name -> whole text.

    The cell log and every captured output, which between them are everything the run
    saw or said. Searched as text rather than as paths because the sighting that
    mattered last round was a `grep` for a filename, not an import of it.
    """
    names = set(house_names(toolbox)) | {str(REPO)}
    hits: list[Hit] = []
    for where, text in texts.items():
        for number, line in enumerate((text or "").splitlines(), start=1):
            for name in names:
                if name in line:
                    hits.append(Hit(name=name, where=where, line=number,
                                    evidence=line.strip()[:200]))
    wanted = set(expected)
    reached = {hit.name for hit in hits}
    missing = sorted(wanted - reached)
    # Two arms, one rule: the bare arm is invalid if it touched a house surface, and the
    # tooled arm is invalid if it never reached one it was handed. Both are runs that did
    # not have the tooling the number will be attributed to.
    return Contamination(contaminated=bool(missing) if wanted else bool(hits),
                         hits=hits, missing=missing)


_TEXT_SUFFIXES = (".txt", ".log", ".json", ".jsonl", ".md", ".py", ".out")


OBSERVER_FILES = ("record.json", "runner.log")
"""What the harness itself writes into the run directory, and skips when it greps it.

Found immediately: the graded record quotes every contamination hit, so a second pass over
the same directory reads the supervisor's own words back and finds the house names it had
just written down. The observer must not be able to contaminate the thing it observes,
including retrospectively.

`runner.log` is the same mistake wearing the driver's clothes, and it cost a sweep to see.
`cad_sweep.py` captures the runner's stdout and writes it here, and that stdout is
operator-facing: it prints the record directory and the `observe` command to run next, both
of which spell out the repo path. So every swept run graded contaminated on the harness's
own chatter while the standalone path -- which prints to a terminal and writes no log --
graded clean, and the difference was the measuring apparatus rather than the desk.

The exclusion is safe because `runner.log` holds no desk output at all: the desk's cells,
its replies and its script go to `cells.log`, `replies.jsonl` and `build.py`, which are
still read. A file the desk cannot write to is not evidence about the desk."""


def scan_run(run_dir: Path, *, toolbox: Path | None = None,
             expected: Iterable[str] = (),
             exclude: Iterable[str] = OBSERVER_FILES) -> Contamination:
    """The same grep over everything a run directory holds that reads as text."""
    directory = Path(run_dir)
    skip = set(exclude)
    texts: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        if path.name in skip:
            continue
        try:
            texts[str(path.relative_to(directory))] = path.read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            continue
    return scan(texts, toolbox=toolbox, expected=expected)


_STEP = re.compile(r"```python\n(.*?)```", re.S)


def cells_of(script: str) -> list[str]:
    """The fenced cells of a transcript, for grepping a log that is not yet a file."""
    return [block.strip() for block in _STEP.findall(script or "") if block.strip()]
