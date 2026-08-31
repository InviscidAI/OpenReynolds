#!/usr/bin/env python3
"""Drive a study through its phases, and pick up where the last session stopped.

A study is the same ten phases every time -- geometry, preview, mesh, checkMesh,
probe, solve, reconstruct, render, animate, report -- and the expensive ones are
long enough that a session regularly ends in the middle of one. What that used to
cost was not the compute: it was the next session re-deriving the whole case from
the transcript, listing directories, opening logs, working out whether the mesh
that is on disk is the one the dictionaries describe. This runs the phases and
leaves the answer in `study_state`, so the question "where is this study" is a
lookup rather than an investigation.

what it decides from
--------------------

Two sources, and the disk wins. `phases.json` says what the last run *recorded*;
the case directory says what is actually there (`constant/polyMesh/owner` for a
mesh, a reconstructed time directory for a reconstruct, an `End` line for a
finished solve). A study gets advanced by hand between sessions all the time --
someone runs `snappyHexMesh` themselves, or deletes a mesh they did not like --
and a recorded status that disagrees with the evidence is out of date, not
authoritative. So a phase whose evidence is present counts as done however it is
recorded, and a phase recorded done whose evidence has since vanished is reported
as stale and planned again. `skipped` is the exception: it is a decision someone
made, not an outcome, so it stays put unless the evidence turns up anyway.

what it does not do
-------------------

**A failing phase stops the pipeline and nothing is repaired.** The failure is
recorded with the tail of its log as the note, and the exit code is non-zero.
There is no retry, no fallback mesh, no automatic coarsening -- what to do about
a `snappyHexMesh` that produced 40 million cells or a solve that diverged at
iteration 300 is the sort of judgement this script has no business making, and a
workflow engine that made it would be quietly deciding the study. It runs things
in an order and writes down what happened. Everything else is yours: edit the
phase table below, run the phases by hand, or ignore the file entirely.

running it long
---------------

Meshes and solves take longer than a synchronous command may, so this is built to
be started as a job. It writes its own progress to `<study>/.reynolds/study_run.log`
and flushes every line, and each command's own output goes to `log.<command>` next
to the case in the usual OpenFOAM place -- so `job_check`, the progress reporter,
`log_digest.py` and a plain `tail -f` all have something to read while it runs.

The phases that other scripts own (`first_look.py` for preview, `preflight.py` for
the probe, `results.py` for render, `animate.py` for animate) are called by path
through the interpreter. A script that is not in the toolbox is not an error: that
phase is skipped with a note saying which script was missing.

    python3 study_run.py /work/case --status          # the table, and what is next
    python3 study_run.py /work/case --dry-run         # the plan and its commands
    python3 study_run.py /work/case --solver simpleFoam
    python3 study_run.py /work/case --from mesh       # redo the mesh and continue
    python3 study_run.py /work/case --only checkMesh
    python3 study_run.py /work/case --skip animate --skip report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state

PYTHON = sys.executable or "python3"

SURFACE_SUFFIXES = (".stl", ".stlb", ".obj", ".ply", ".vtk", ".vtp", ".vtu")

TAIL_LINES = 14
"""How much of a failed command's log goes into the phase note. Enough to show a
FOAM FATAL ERROR and the lines that led to it, short enough that the phase table
still reads as a table."""

TAIL_BYTES = 64 * 1024
"""How much of a log is read to answer a question about its end. A solver log runs
to hundreds of megabytes and every `--status` asks whether it reached `End`;
reading the last page of it answers that without pulling the whole run into
memory."""


# -- what is on disk ---------------------------------------------------------------


def surfaces(case: Path) -> list[Path]:
    """Surface files under `constant/triSurface`, the place snappy looks for them."""
    directory = case / "constant" / "triSurface"
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in SURFACE_SUFFIXES)


def numeric_dirs(directory: Path) -> list[tuple[float, Path]]:
    """Time directories, as (value, path), oldest first.

    A name that is not a number is not a time: `0.orig`, `constant` and `system`
    all live in the same directory and all of them would otherwise have to be
    excluded by name.
    """
    if not directory.is_dir():
        return []
    found: list[tuple[float, Path]] = []
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        found.append((value, child))
    return sorted(found, key=lambda pair: pair[0])


def written_times(directory: Path) -> list[tuple[float, Path]]:
    """Time directories past 0 -- results, rather than initial conditions."""
    return [pair for pair in numeric_dirs(directory) if pair[0] > 0.0]


def latest_time(directory: Path) -> float | None:
    times = written_times(directory)
    return times[-1][0] if times else None


def processor_dirs(case: Path) -> list[Path]:
    if not case.is_dir():
        return []
    found = []
    for child in case.iterdir():
        if child.is_dir() and child.name.startswith("processor") and child.name[9:].isdigit():
            found.append(child)
    return sorted(found, key=lambda path: int(path.name[9:]))


def has_mesh(case: Path) -> bool:
    """`owner` and not the directory: `constant/polyMesh` is created early and an
    interrupted `snappyHexMesh` leaves it behind holding nothing usable."""
    return (case / "constant" / "polyMesh" / "owner").exists()


def tail_text(path: Path, max_bytes: int = TAIL_BYTES) -> str:
    """The last `max_bytes` of a file, decoded. Empty when there is nothing to read.

    Seeking rather than reading the whole file matters here and nowhere else in
    this script: the files these questions are asked about are solver logs, they
    are asked on every `--status`, and a `read_text` of a long run costs several
    times the log in memory. Cutting into the middle of a multi-byte character is
    what `errors="replace"` is for.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            raw = handle.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def log_finished(path: Path) -> bool:
    """Whether an OpenFOAM log reached its own `End`.

    A solver that was killed -- the instance reaped, the wall clock hit, a
    `SIGKILL` from the job runner -- leaves a log full of perfectly healthy
    iterations and no `End`, which is the only thing that separates "ran out of
    time" from "converged and stopped".
    """
    if not path.exists():
        return False
    for line in reversed(tail_text(path).splitlines()[-40:]):
        if line.strip() == "End":
            return True
    return False


def log_has_output(path: Path) -> bool:
    """Whether a log holds anything beyond the `$ command` line this script echoes
    into it before starting a command.

    "The file is not empty" is not evidence that a phase ran, because this script
    creates the file itself: a binary that is not on the image leaves the echo and
    nothing else, and a phase that reads its own failure back as evidence would
    resume past the thing that broke.
    """
    if not path.exists():
        return False
    for line in tail_text(path).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("$ "):
            return True
    return False


def log_tail(path: Path | None, lines: int = TAIL_LINES, limit: int = 1400) -> str:
    if path is None or not Path(path).exists():
        return ""
    tail = "\n".join(tail_text(Path(path)).splitlines()[-lines:]).strip()
    return tail[-limit:]


def frame_dirs(*roots: Path) -> list[Path]:
    """`*_frames/` directories holding at least two PNGs -- the animate convention."""
    found: list[Path] = []
    for root in roots:
        if not root or not root.is_dir():
            continue
        for child in sorted(root.glob("*_frames")):
            if child.is_dir() and len(list(child.glob("*.png"))) >= 2:
                found.append(child)
    return found


def registered(ctx: "Context", kinds: Iterable[str]) -> str:
    """The path of the newest still-present artifact of any of these kinds, or ""."""
    for kind in kinds:
        row = study_state.latest(kind, root=ctx.root, exists=True)
        if row:
            return str(row.get("path") or "")
    return ""


def registered_matching(ctx: "Context", kind: str, needle: str) -> str:
    """The newest still-present artifact of a kind whose path or label mentions
    `needle`, or "".

    Two scripts legitimately register the same kind -- `first_look.py` and
    `gallery.py` both produce a `contact-sheet` -- and a phase that treats either as
    its own evidence marks itself done on the strength of the other's work. The
    producer is not a field in the manifest, so it is read off the path and the label
    it wrote, which is enough to tell those two apart.
    """
    for row in reversed(study_state.artifacts(root=ctx.root, kind=kind, exists=True)):
        haystack = f"{row.get('path', '')} {row.get('label', '')}".lower()
        if needle.lower() in haystack:
            return str(row.get("path") or "")
    return ""


def read_text(path: Path) -> str:
    """A file's text, or "" -- every caller here is asking a question about a file
    that may not have been written yet, and that is an answer, not an error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# -- the run context ---------------------------------------------------------------


class Context(NamedTuple):
    case: Path
    root: Path
    toolbox: Path
    solver: str
    field: str
    parallel: int
    preset: str = ""
    """Which `results.py` preset the render phase asks for. Empty means the phase
    reads one off the case (`results_preset`)."""


def context(case: Path, *, toolbox: Path | None = None, solver: str = "simpleFoam",
            field: str = "vorticity", parallel: int = 0, preset: str = "") -> Context:
    case = Path(case).resolve()
    return Context(
        case=case,
        root=Path(study_state.find_root(case)),
        toolbox=Path(toolbox) if toolbox else Path(__file__).resolve().parent,
        solver=solver,
        field=field,
        parallel=max(0, int(parallel)),
        preset=preset,
    )


class Command(NamedTuple):
    name: str
    argv: list[str]
    cwd: Path
    log: Path

    def describe(self) -> str:
        return f"{' '.join(self.argv)}   > {self.log.name}"


def foam(ctx: Context, name: str, *args: str, log: str = "") -> Command:
    """One OpenFOAM binary, run in the case directory, logging where OpenFOAM users
    look for it (`log.blockMesh` and friends)."""
    return Command(name, [name, *args], ctx.case, ctx.case / f"log.{log or name}")


def script(ctx: Context, path: Path, *args: str) -> Command:
    return Command(path.name, [PYTHON, str(path), *args], ctx.case, ctx.case / f"log.{path.stem}")


def find_script(ctx: Context, *names: str) -> Path | None:
    """The first of these toolbox scripts that is actually present.

    Ordered by preference, so a phase can name the script that owns it and fall
    back to the older one that does part of the same job -- and get neither
    without failing.
    """
    for name in names:
        candidate = ctx.toolbox / name
        if candidate.exists():
            return candidate
    return None


# -- the phases --------------------------------------------------------------------

Evidence = Callable[[Context], "tuple[bool, str]"]
Builder = Callable[[Context], "tuple[list[Command], str]"]


class PhaseSpec(NamedTuple):
    name: str
    required: bool
    evidence: Evidence
    build: Builder
    about: str


def geometry_evidence(ctx: Context) -> tuple[bool, str]:
    found = surfaces(ctx.case)
    if found:
        return True, f"{len(found)} surface(s) in constant/triSurface"
    if (ctx.case / "system" / "blockMeshDict").exists():
        return True, "system/blockMeshDict"
    if (ctx.case / "system" / "blockMeshDict.m4").exists():
        return True, "system/blockMeshDict.m4"
    return False, "no surfaces in constant/triSurface and no blockMeshDict"


def geometry_build(ctx: Context) -> tuple[list[Command], str]:
    return [], "geometry is authored, not generated; this phase only looks for it"


def preview_evidence(ctx: Context) -> tuple[bool, str]:
    # A contact sheet counts as well as a geometry panel: `first_look.py` writes the
    # sheet whether or not every panel on it rendered, and a sheet with four good
    # panels is a preview that happened.
    row = registered(ctx, ("geometry-preview",))
    if row:
        return True, f"registered geometry-preview: {row}"
    # A contact sheet counts only when it is `first_look.py`'s. `gallery.py` writes
    # one too, from the finished study, and letting that stand as the preview meant
    # a study that had reached the gallery could never be previewed again -- the
    # evidence for "we looked before spending" was satisfied by looking afterwards.
    sheet = registered_matching(ctx, "contact-sheet", "first_look")
    if sheet:
        return True, f"first_look contact sheet: {sheet}"
    for directory in (ctx.case / "renders", ctx.root / "renders"):
        if directory.is_dir() and list(directory.rglob("geometry*.png")):
            return True, f"{directory.name}/geometry*.png"
    return False, "no geometry-preview registered and no renders/geometry*.png"


def preview_build(ctx: Context) -> tuple[list[Command], str]:
    owner = find_script(ctx, "first_look.py")
    if owner:
        return [script(ctx, owner, str(ctx.case))], ""
    fallback = find_script(ctx, "geometry_view.py")
    if fallback and surfaces(ctx.case):
        return [script(ctx, fallback, str(ctx.case / "constant" / "triSurface"),
                       "--out", str(ctx.case / "renders"))], ""
    if fallback:
        return [], "geometry_view.py is here but there are no surfaces to draw"
    return [], "neither first_look.py nor geometry_view.py is in the toolbox"


def mesh_evidence(ctx: Context) -> tuple[bool, str]:
    if has_mesh(ctx.case):
        return True, "constant/polyMesh/owner"
    return False, "no constant/polyMesh/owner"


def mesh_build(ctx: Context) -> tuple[list[Command], str]:
    system = ctx.case / "system"
    commands: list[Command] = []
    if (system / "blockMeshDict").exists() or (system / "blockMeshDict.m4").exists():
        commands.append(foam(ctx, "blockMesh"))
    if (system / "surfaceFeatureExtractDict").exists():
        commands.append(foam(ctx, "surfaceFeatureExtract"))
    elif (system / "surfaceFeaturesDict").exists():
        commands.append(foam(ctx, "surfaceFeatures"))
    if (system / "snappyHexMeshDict").exists():
        commands.append(foam(ctx, "snappyHexMesh", "-overwrite"))
    if not commands:
        return [], "no blockMeshDict and no snappyHexMeshDict in system/"
    return commands, ""


def check_mesh_evidence(ctx: Context) -> tuple[bool, str]:
    """`End`, and not merely a log that exists.

    The log is created by this script before `checkMesh` is started, so a
    `checkMesh` that is not on the image or that died on the mesh leaves a file
    behind either way -- and since the evidence outranks the record, treating that
    file as proof would turn the failure this run just recorded into a `done` on
    the next one and resume past an unchecked mesh.
    """
    log = ctx.case / "log.checkMesh"
    if log_finished(log):
        return True, "log.checkMesh reached End"
    if log.exists():
        return False, "log.checkMesh has no End line: checkMesh did not finish"
    return False, "no log.checkMesh"


def check_mesh_build(ctx: Context) -> tuple[list[Command], str]:
    commands = [foam(ctx, "checkMesh")]
    digest = find_script(ctx, "mesh_digest.py")
    if digest:
        commands.append(Command(
            digest.name,
            [PYTHON, str(digest), str(ctx.case / "log.checkMesh")],
            ctx.case,
            ctx.case / "log.mesh_digest",
        ))
    return commands, ""


def probe_evidence(ctx: Context) -> tuple[bool, str]:
    """A probe log with something in it other than the command that was attempted.
    `preflight.py` has no `End` line to look for, so the bar is lower than
    `checkMesh`'s -- but an empty log is still not a probe that happened."""
    attempted = ""
    for name in ("log.preflight", "log.probe"):
        log = ctx.case / name
        if log_has_output(log):
            return True, name
        if log.exists():
            attempted = name
    if attempted:
        return False, f"{attempted} holds only the command that was attempted"
    return False, "no log.preflight"


def probe_build(ctx: Context) -> tuple[list[Command], str]:
    owner = find_script(ctx, "preflight.py")
    if not owner:
        return [], "preflight.py is not in the toolbox"
    # preflight takes no --solver: it reads the application out of controlDict
    # itself. Passing one made argparse refuse the call, so the phase that exists to
    # be the cheap check before the expensive run failed before checking anything.
    return [script(ctx, owner, str(ctx.case))], ""


def solve_evidence(ctx: Context) -> tuple[bool, str]:
    log = ctx.case / f"log.{ctx.solver}"
    if log_finished(log):
        return True, f"log.{ctx.solver} reached End"
    processors = processor_dirs(ctx.case)
    reached = latest_time(processors[0]) if processors else latest_time(ctx.case)
    where = f"latest written time {reached:g}" if reached is not None else "no written times"
    if log.exists():
        return False, f"log.{ctx.solver} has no End line ({where})"
    return False, f"no log.{ctx.solver} ({where})"


def solve_build(ctx: Context) -> tuple[list[Command], str]:
    commands: list[Command] = []
    if ctx.parallel > 1:
        if not (ctx.case / "system" / "decomposeParDict").exists():
            return [], f"--parallel {ctx.parallel} was asked for and there is no system/decomposeParDict"
        if not processor_dirs(ctx.case):
            commands.append(foam(ctx, "decomposePar"))
        commands.append(Command(
            f"mpirun {ctx.solver}",
            ["mpirun", "-np", str(ctx.parallel), ctx.solver, "-parallel"],
            ctx.case,
            ctx.case / f"log.{ctx.solver}",
        ))
        return commands, ""
    commands.append(foam(ctx, ctx.solver))
    return commands, ""


def reconstruct_evidence(ctx: Context) -> tuple[bool, str]:
    if not processor_dirs(ctx.case):
        return True, "serial run: nothing was decomposed"
    reached = latest_time(ctx.case)
    if reached is not None:
        return True, f"reconstructed time {reached:g} in the case directory"
    return False, f"{len(processor_dirs(ctx.case))} processor directories and no reconstructed time"


def reconstruct_build(ctx: Context) -> tuple[list[Command], str]:
    if not processor_dirs(ctx.case):
        return [], "no processor* directories: the run was serial"
    return [foam(ctx, "reconstructPar")], ""


def render_evidence(ctx: Context) -> tuple[bool, str]:
    row = registered(ctx, ("velocity", "pressure", "vorticity", "streamlines"))
    if row:
        return True, f"registered field picture: {row}"
    results = ctx.case / "results"
    if results.is_dir() and list(results.rglob("*.png")):
        return True, f"PNGs in {results}"
    for directory in (ctx.case / "renders", ctx.root / "renders"):
        pictures = list(directory.glob("*.png")) if directory.is_dir() else []
        # Geometry pictures are the preview's, not the run's.
        if [path for path in pictures if not path.name.startswith("geometry")]:
            return True, f"PNGs in {directory}"
    return False, "no field picture registered and none in results/ or renders/"


BODY_PATCHES = ("body", "wheel", "car", "vehicle", "obstacle", "cylinder", "aerofoil",
                "airfoil", "foil", "wing", "hull", "building")
"""Patch names that mean "something is in the flow". Matched as substrings and case
sensitively enough to be useful without being clever -- `wheelFront` and `bodySurface`
both count, and a `topWall` does not."""


def is_two_dimensional(patches: str) -> bool:
    """Whether the boundary list has an `empty` patch, i.e. this is a 2D case."""
    return "empty" in patches


def results_preset(ctx: Context) -> str:
    """Which set of pictures this case wants, read off the case itself.

    `results.py` has no default preset and refuses without one, so something has to
    choose. The case knows: a transient run with a wake is not the same picture set
    as a duct, and a case with no solution at all wants the mesh looked at rather
    than fields that do not exist yet.
    """
    if ctx.preset:
        return ctx.preset
    if not written_times(ctx.case):
        return "mesh-validation"
    control = read_text(ctx.case / "system" / "controlDict")
    transient = "pimpleFoam" in control or "pisoFoam" in control
    patches = read_text(ctx.case / "constant" / "polyMesh" / "boundary")
    if not patches:
        patches = read_text(ctx.case / "system" / "blockMeshDict")

    # Is there a body in the flow? That is what separates external flow from a duct,
    # and it is a positive signal rather than the absence of two patch names: reading
    # "no topWall and no bottomWall" as "duct" made every snappyHexMesh case -- whose
    # patches are whatever the STL was called -- render as a duct, with no vorticity,
    # no forces and no mesh cut.
    obstacle = any(name in patches for name in BODY_PATCHES)
    if not obstacle:
        return "duct-flow"
    # A case that is not one cell thick is a 3D external case: the vehicle preset is
    # the one with a mesh cut in it.
    if not is_two_dimensional(patches):
        return "vehicle-aero"
    return "transient-wake" if transient else "external-flow-2d"


def render_build(ctx: Context) -> tuple[list[Command], str]:
    owner = find_script(ctx, "results.py")
    if owner:
        return [script(ctx, owner, str(ctx.case), "--preset", results_preset(ctx))], ""
    fallback = find_script(ctx, "render.py")
    if fallback:
        return [script(ctx, fallback, str(ctx.case))], ""
    return [], "neither results.py nor render.py is in the toolbox"


def animate_evidence(ctx: Context) -> tuple[bool, str]:
    row = registered(ctx, ("animation",))
    if row:
        return True, f"registered animation: {row}"
    directories = frame_dirs(ctx.case, ctx.case.parent, ctx.root)
    if directories:
        return True, f"frames in {directories[0].name}"
    return False, "no *_frames directory with frames in it"


def animate_build(ctx: Context) -> tuple[list[Command], str]:
    owner = find_script(ctx, "animate.py")
    if not owner:
        return [], "animate.py is not in the toolbox"
    times = written_times(ctx.case)
    if len(times) < 2:
        return [], f"only {len(times)} written time(s): there is nothing to animate yet"
    return [script(ctx, owner, str(ctx.case), "--field", ctx.field)], ""


def report_evidence(ctx: Context) -> tuple[bool, str]:
    # Not `report`: `results.py` registers its own results summary under that kind
    # from inside the render phase, which marked the report done before anything had
    # been written up. The gallery page is the write-up this phase produces.
    row = registered(ctx, ("gallery",))
    if row:
        return True, f"registered gallery: {row}"
    for candidate in (ctx.root / "report.md", ctx.case / "report.md"):
        if candidate.exists():
            return True, str(candidate.name)
    return False, "no report registered and no report.md"


def report_build(ctx: Context) -> tuple[list[Command], str]:
    """The write-up is `gallery.py`: the page and the contact sheet that hand the
    study over. This used to look for a `report.py` that does not exist and never
    did, so the last phase of the pipeline was permanently unbuildable."""
    owner = find_script(ctx, "gallery.py")
    if owner:
        return [script(ctx, owner, str(ctx.root))], ""
    fallback = find_script(ctx, "report.py", "study_report.py")
    if fallback:
        return [script(ctx, fallback, str(ctx.case))], ""
    return [], "neither gallery.py nor report.py is in the toolbox"


SPECS: tuple[PhaseSpec, ...] = (
    PhaseSpec("geometry", True, geometry_evidence, geometry_build,
              "a surface or a blockMeshDict to build from"),
    PhaseSpec("preview", False, preview_evidence, preview_build,
              "pictures of the geometry before anything is meshed"),
    PhaseSpec("mesh", True, mesh_evidence, mesh_build,
              "blockMesh, feature extraction and snappyHexMesh, as the dictionaries ask"),
    PhaseSpec("checkMesh", True, check_mesh_evidence, check_mesh_build,
              "checkMesh and its digest"),
    PhaseSpec("probe", False, probe_evidence, probe_build,
              "the cheap solver check before the expensive solve"),
    PhaseSpec("solve", True, solve_evidence, solve_build,
              "the solver, serial or under mpirun"),
    PhaseSpec("reconstruct", False, reconstruct_evidence, reconstruct_build,
              "reconstructPar, when the run was decomposed"),
    PhaseSpec("render", False, render_evidence, render_build,
              "field pictures from the finished run"),
    PhaseSpec("animate", False, animate_evidence, animate_build,
              "one PNG per write time into a *_frames directory"),
    PhaseSpec("report", False, report_evidence, report_build,
              "the write-up, if a script owns it"),
)

PHASE_NAMES: tuple[str, ...] = tuple(spec.name for spec in SPECS)


def spec_for(name: str) -> PhaseSpec:
    for spec in SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


# -- reconciling the record with the evidence --------------------------------------


class PhaseState(NamedTuple):
    name: str
    recorded: str
    evident: bool
    why: str
    status: str
    stale: bool

    def line(self) -> str:
        mark = "  (recorded done, evidence gone)" if self.stale else ""
        if self.recorded != self.status and not self.stale:
            mark = f"  (recorded {self.recorded})"
        return f"  {self.status:<8} {self.name:<12} {self.why}{mark}"


def reconcile(ctx: Context, table: dict | None = None) -> list[PhaseState]:
    """Every phase's real status, from the record and the case directory together.

    The rules, in order: evidence on disk means done, whatever was written down;
    `skipped` is a decision and survives the absence of evidence; a phase recorded
    done with nothing to show for it is stale and goes back to pending; anything
    else keeps the status it was left with, so a `failed` from the last session is
    still visible as a failure rather than flattened into "not done yet".
    """
    if table is None:
        table = study_state.load_phases(ctx.root)
    recorded = {
        str(row.get("name")): str(row.get("status") or "pending")
        for row in table.get("phases", [])
        if isinstance(row, dict)
    }
    states: list[PhaseState] = []
    for spec in SPECS:
        was = recorded.get(spec.name, "pending")
        evident, why = spec.evidence(ctx)
        stale = False
        if evident:
            status = "done"
        elif was == "skipped":
            status = "skipped"
        elif was == "done":
            status, stale = "pending", True
        elif was in ("failed", "running"):
            status = was
        else:
            status = "pending"
        states.append(PhaseState(spec.name, was, evident, why, status, stale))
    return states


def next_to_run(states: Sequence[PhaseState]) -> str:
    """The first phase that is neither done nor skipped -- where a resumed study
    picks up. Empty when the pipeline is finished."""
    for state in states:
        if state.status not in ("done", "skipped"):
            return state.name
    return ""


def build_plan(states: Sequence[PhaseState], *, start: str = "", only: str = "",
               skip: Iterable[str] = ()) -> list[str]:
    """Which phases this invocation would run.

    Three shapes, and they mean different things on purpose. Plain: everything
    that is not already done, starting at the first incomplete phase -- the resume
    case, and the one that must never re-run a finished solve. `--from PHASE`:
    that phase and everything after it, *including* what is already done, because
    the reason to ask for it is that something upstream changed and the downstream
    results are now suspect. `--only PHASE`: exactly that one.
    """
    names = [state.name for state in states]
    unwanted = set(skip)
    for name in (*unwanted, start, only):
        if name and name not in names:
            raise ValueError(f"{name!r} is not a phase; the phases are {', '.join(names)}")
    if only:
        return [] if only in unwanted else [only]
    if start:
        window = names[names.index(start):]
        return [name for name in window if name not in unwanted]
    plan = []
    for state in states:
        if state.name in unwanted or state.status in ("done", "skipped"):
            continue
        plan.append(state.name)
    return plan


def plan_lines(ctx: Context, plan: Sequence[str]) -> list[str]:
    """The plan with each phase's commands resolved, for `--dry-run`."""
    lines: list[str] = []
    for name in plan:
        spec = spec_for(name)
        commands, note = spec.build(ctx)
        lines.append(f"  {name}" + ("" if spec.required else "   (optional)"))
        if not commands:
            evident, why = spec.evidence(ctx)
            if evident:
                lines.append(f"      nothing to run: already satisfied by {why}")
            else:
                lines.append(f"      would be skipped: {note or 'nothing to run'}")
            continue
        for command in commands:
            lines.append(f"      {command.describe()}")
    return lines


# -- running -----------------------------------------------------------------------


class Journal:
    """One line at a time to stdout and, when there is a study to hold it, to
    `.reynolds/study_run.log`. Both flushed on every line: a solve is long enough
    that a progress file nobody can read until the process exits is no progress
    file at all."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        if self.path is None:
            return
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
                handle.flush()
        except OSError:
            self.path = None


def shell_runner(command: Command) -> tuple[int, str]:
    """Run one command with its output in its log; return the code and the tail.

    A binary that is not on PATH comes back as a failure with a readable note
    rather than a traceback: on an image without a given solver that is a fact
    about the phase, and the phase table is where it belongs.
    """
    command.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with command.log.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write(f"$ {' '.join(command.argv)}\n")
            handle.flush()
            finished = subprocess.run(command.argv, cwd=str(command.cwd),
                                      stdout=handle, stderr=subprocess.STDOUT)
        return finished.returncode, log_tail(command.log)
    except FileNotFoundError:
        return 127, f"{command.argv[0]}: not found on PATH"
    except OSError as exc:
        return 1, f"{command.argv[0]}: {exc}"


class PhaseResult(NamedTuple):
    name: str
    status: str
    note: str
    log: Path | None = None


def run_phase(ctx: Context, name: str, *, runner: Callable[[Command], tuple[int, str]],
              journal: Journal, record: bool = True) -> PhaseResult:
    """Run one phase's commands in order and return what became of it.

    The runner is injected so this can be exercised without an OpenFOAM
    installation anywhere near it.
    """
    spec = spec_for(name)
    commands, note = spec.build(ctx)
    if not commands:
        # No commands and the evidence is already there is a different thing from
        # no commands and nothing to show: geometry is authored by hand and a
        # serial run has nothing to reconstruct, and neither of those is a phase
        # that was passed over.
        evident, why = spec.evidence(ctx)
        if evident:
            journal.say(f"{name}: done -- {why}")
            study_state.set_phase(name, "done", root=ctx.root, note=why, case=ctx.case.name)
            return PhaseResult(name, "done", why)
        journal.say(f"{name}: skipped -- {note or 'nothing to run'}")
        study_state.set_phase(name, "skipped", root=ctx.root, note=note or "nothing to run",
                              case=ctx.case.name)
        return PhaseResult(name, "skipped", note or "nothing to run")

    study_state.set_phase(name, "running", root=ctx.root, case=ctx.case.name)
    for command in commands:
        journal.say(f"{name}: {command.describe()}")
        code, tail = runner(command)
        if record:
            _record_log(ctx, name, command)
        if code != 0:
            failure = f"{command.name} exited {code}\n{tail}".strip()
            journal.say(f"{name}: FAILED ({command.name} exited {code})")
            if tail:
                journal.say(tail)
            study_state.set_phase(name, "failed", root=ctx.root, note=failure, case=ctx.case.name)
            return PhaseResult(name, "failed", failure, command.log)

    evident, why = spec.evidence(ctx)
    if evident:
        journal.say(f"{name}: done -- {why}")
        study_state.set_phase(name, "done", root=ctx.root, note=why, case=ctx.case.name)
        return PhaseResult(name, "done", why, commands[-1].log)
    # The commands returned 0 and the thing they were supposed to leave behind is
    # not there. That is worth saying out loud and is not worth calling a failure:
    # a preview phase whose renderer wrote somewhere else is fine, and only the
    # agent can tell that from a mesher that silently produced nothing.
    unproven = f"commands succeeded but {why}"
    journal.say(f"{name}: done -- {unproven}")
    study_state.set_phase(name, "done", root=ctx.root, note=unproven, case=ctx.case.name)
    return PhaseResult(name, "done", unproven, commands[-1].log)


def _record_log(ctx: Context, phase: str, command: Command) -> None:
    """Put a command's log in the manifest, so "where is the snappy log" is a
    lookup. Never allowed to take a run down with it."""
    try:
        if command.log.exists() and command.log.stat().st_size > 0:
            study_state.record("other", command.log, root=ctx.root, case=ctx.case.name,
                               label=f"{phase} log: {command.name}")
    except (OSError, ValueError):
        pass


def execute(ctx: Context, plan: Sequence[str], *, runner: Callable[[Command], tuple[int, str]],
            journal: Journal, skip: Iterable[str] = (), record: bool = True) -> tuple[int, list[PhaseResult]]:
    """Run the plan, stopping at the first failure. Returns (exit code, results)."""
    for name in skip:
        study_state.set_phase(name, "skipped", root=ctx.root, note="skipped on the command line",
                              case=ctx.case.name)
    results: list[PhaseResult] = []
    for name in plan:
        result = run_phase(ctx, name, runner=runner, journal=journal, record=record)
        results.append(result)
        if result.status == "failed":
            journal.say("")
            journal.say(f"stopped at {name}. Nothing was repaired and nothing else was run.")
            if result.log is not None:
                journal.say(f"the log is {result.log}")
            return 1, results
    journal.say("")
    journal.say(f"ran {len(results)} phase(s): " + ", ".join(f"{r.name}={r.status}" for r in results))
    return 0, results


# -- the command line --------------------------------------------------------------


def status_lines(ctx: Context, states: Sequence[PhaseState], plan: Sequence[str]) -> list[str]:
    lines = [f"case  {ctx.case}", f"study {ctx.root}", ""]
    lines += [state.line() for state in states]
    lines.append("")
    upcoming = next_to_run(states)
    lines.append(f"next incomplete phase: {upcoming or 'none, the pipeline is finished'}")
    lines.append("would run: " + (", ".join(plan) if plan else "nothing"))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, help="The case directory.")
    parser.add_argument("--solver", default="simpleFoam", help="Solver binary (default simpleFoam).")
    parser.add_argument("--field", default="vorticity", help="Field for the animate phase.")
    parser.add_argument("--preset", default="", metavar="NAME",
                        help="results.py preset for the render phase. Default: read off "
                             "the case (`results.py --list` names them all).")
    parser.add_argument("--parallel", type=int, default=0, metavar="N",
                        help="Decompose and run the solve under mpirun on N ranks.")
    parser.add_argument("--toolbox", type=Path, default=None,
                        help="Where the other scripts are (default: beside this one).")
    parser.add_argument("--from", dest="start", default="", choices=("", *PHASE_NAMES),
                        metavar="PHASE", help="Run this phase and everything after it, done or not.")
    parser.add_argument("--only", default="", choices=("", *PHASE_NAMES), metavar="PHASE",
                        help="Run just this phase.")
    parser.add_argument("--skip", action="append", default=[], choices=PHASE_NAMES,
                        metavar="PHASE", help="Leave this phase out (repeatable).")
    parser.add_argument("--status", action="store_true",
                        help="Print the phase table and what is next; run nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; run nothing.")
    args = parser.parse_args(argv)

    if args.start and args.only:
        parser.error("--from and --only ask for different things; pick one")
    if not args.case.exists():
        parser.error(f"{args.case} does not exist")

    ctx = context(args.case, toolbox=args.toolbox, solver=args.solver, preset=args.preset,
                  field=args.field, parallel=args.parallel)
    states = reconcile(ctx)
    try:
        plan = build_plan(states, start=args.start, only=args.only, skip=args.skip)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if args.status:
        for line in status_lines(ctx, states, plan):
            print(line)
        return 0

    if args.dry_run:
        print(f"case  {ctx.case}")
        print(f"study {ctx.root}")
        print("plan:" if plan else "plan: nothing to do")
        for line in plan_lines(ctx, plan):
            print(line)
        if args.skip:
            print("skipped on request: " + ", ".join(args.skip))
        return 0

    if not plan:
        print(f"nothing to run: {next_to_run(states) or 'every phase is done or skipped'}")
        return 0

    journal = Journal(Path(study_state.state_dir(ctx.root)) / "study_run.log")
    journal.say("")
    journal.say(f"study_run {ctx.case}  solver={ctx.solver}  plan: {', '.join(plan)}")
    try:
        study_state.record("other", journal.path, root=ctx.root, case=ctx.case.name,
                           label="study_run progress log")
    except (OSError, ValueError):
        pass
    code, _results = execute(ctx, plan, runner=shell_runner, journal=journal, skip=args.skip)
    return code


if __name__ == "__main__":
    sys.exit(main())
