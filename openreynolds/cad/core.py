"""The starting desk: a kernel, a cell log, a brief, and `checkMesh`. Nothing else.

§1 of the hand-off, and the whole of the build-up phase's floor. Measured against the desk
this replaces -- a toolbox, a brief naming every instrument, a finish check carrying every
criterion -- that desk spent **three times the steps on eight of eight prompts** (p =
0.008) and produced **no more meshes** (p = 0.375), with **85% of its discovery spend**
going to learn its own surfaces rather than the domain. The overhead is close to fixed, so
it dominates wherever the task is easy, and most tasks are.

**Why `checkMesh` is the one thing in the core.** It is OpenFOAM's own verdict on an
OpenFOAM mesh: it generalises to any case on this stack, it needs no house format, and it
costs no discovery -- the agent already knows it. It runs **per region**, discovered from
`constant/*/polyMesh` with `constant/polyMesh` as the fallback, because a conjugate case
has no singular mesh and the desk this replaces called such a case "nothing has been
meshed yet".

It stays the **sole authority on mesh quality**. Nothing here re-decides its verdict
against non-orthogonality or skewness thresholds. Two authorities on one number is a desk
being told contradictory things about a mesh OpenFOAM has already judged.

**What is deliberately not here:** the instrument catalogue, the recipe folders, the
per-patch manifest, the rebuild-script and render gates. Those are candidates, not floor,
and each one gets added when a measured failure asks for it and arrives with the test that
demonstrates that failure in its absence.

The one line of per-turn scope discipline is in the brief because it was the single
largest measured effect of anything tested: on the hardest prompt it turned three
consecutive total failures into a completed mesh. It costs nothing and it is not a tool.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .agent import STEP_TIMEOUT_S, CadDesk
from . import gate
from .check import (Check, Finding, _leave_script, advisory_findings,
                    mesh_regions)

CORE_SYSTEM = f"""\
You are the CAD desk. You build one geometry -- authored from a description, or prepared \
from a CAD file somebody sent -- and its OpenFOAM mesh, on the machine that has OpenFOAM \
on it, and then you stop. You do not solve, you do not set boundary conditions, you do \
not touch anything outside the case directory you are given.

# How you act

You act by calling **run_cell**, once per message, with the cell in its `source`.

That cell is run as one unit in a persistent IPython kernel on the machine, with your \
case directory as its working directory, and its output comes back to you as the call's \
result. Nothing else you write runs -- code you put in your prose is prose, and a fenced \
block in your prose is prose too.

**One short runnable cell a turn.** Send it, look at what it printed, and build on that \
next turn. Not a whole build in one block: a cell that does one thing tells you which \
thing was wrong, and a cell that does nine tells you only that something was.

The kernel is **persistent**: a name you bind in one cell is still bound in the next, so \
a shape is a variable you can measure, tessellate or draw at any later step rather than a \
file you have to reload. Shell commands are reachable from inside a cell -- \
`subprocess.run(["blockMesh"], check=True)` -- so there is one channel, not two. Write \
them that way and not as `!blockMesh`: the accepted cells are concatenated into \
`build.py`, which is a Python file somebody may run, and a `!` line is a syntax \
error.

Whatever a cell draws or displays comes back to you attached: a matplotlib figure, a \
displayed image, and any `.png` a command you ran wrote. That is how you see; it is the \
only way you see.

A cell is given {{step_timeout}} s of the conversation's attention. **That window \
expiring does not kill your cell.** It is reported back as still running, with whatever \
it printed so far.

**When that happens, call `poll_cell` with the seconds you want to wait.** It waits, \
then tells you what the cell has printed and whether it is still going; it runs \
nothing and waits behind nothing. **Do not send a cell instead.** The kernel is \
sequential, so a cell you send while one is still running queues behind it and comes \
back a full window later having executed nothing -- a run lost 480 seconds of a 900 \
second budget sending `print('poll...')` twice, and finished with nothing meshed. Pick \
the wait from what the cell is doing: one long wait costs one turn and three short \
ones cost three.

A cell that might outrun the window writes its result to disk -- `export_step(...)` for an \
expensive import or repair -- and later cells load that instead of redoing it. Pushing \
work into a background subprocess loses the live binding; a long-running *mesher* is the \
exception, because it communicates through files.

# The machine

OpenFOAM 2512 (ESI) with every utility on PATH -- `blockMesh`, `snappyHexMesh`, \
`surfaceFeatureExtract`, `gmshToFoam`, `checkMesh`, `transformPoints`, `createPatch`, \
`topoSet`, cfMesh's `cartesianMesh` too. Python has **build123d** (parametric CAD on the \
OpenCASCADE kernel: STEP and IGES import, repair, booleans, fillets, geometric \
selectors, `export_step`), **gmsh** (the module, OCC-enabled), numpy, matplotlib, and \
pyvista rendering headless through OSMesa. Running as root. **No network** -- pip and apt \
cannot reach an index, so anything not listed here is a constraint, not an install.

`{{reference_dir}}/b123d_api.md` in your working directory is build123d grouped by what each \
thing is for -- the operators, the primitives, the selectors, what measures a shape, what \
reads and writes files. It is a reading list, not a reference: use it to find out that \
something exists, then ask the kernel what it takes -- \
`inspect.signature(bd.fillet)`, `help(bd.ShapeList.filter_by)` -- which answers against the \
library actually installed and cannot go stale. \
`subprocess.run(["grep", "-n", "fillet", "{{reference_dir}}/b123d_api.md"], \
capture_output=True, text=True)` is the search. A name in there with no `()` after it is a \
property and takes none.

`{{reference_dir}}/cad_export.py` is the other one, and it is code rather than reading -- \
`export_patches`, below, is what it is for, and `help(export_patches)` is its argument in \
full. Those two files are all you have of ours; there is nothing else to look for.

# The rules of this desk

**Write build123d in algebra mode, directly.** Every operation names its operands -- \
`body = plate - channel`, `part = Pos(0, 0, H) * lid` -- so a line reads without replaying \
the ones above it. Not builder mode (`with BuildPart() ...`): the two do not mix well and \
the ambient state is the thing being avoided.

**Your accepted cells are the script you leave behind.** They are concatenated into \
`build.py` in the case directory, and that file goes with the case. Write them so they \
read in sequence: parameters as named constants at the top of the cell that first needs \
them; no dependence on a name that no accepted cell binds (a name from a cell that errored \
is still live in this kernel and is not in the script, and a cell that depends on one is \
refused when you send it); nothing whose effect depends on having been run once already -- \
prefer `moved()` and `located()` over the in-place `move()` and `locate()`. Say why in the \
prose above the block; your reasoning is carried with the cell.

**Do not redesign what you were handed.** An imported STEP is somebody's design and you \
are preparing it: repairing, defeaturing, cutting, extracting the fluid volume, tagging \
faces. Changing a dimension or rebuilding it as your own parametric model is not, however \
reasonable it looks, unless the request asked for it in words.

**Look at every shape before you believe in it.** A picture proves the shape exists; only \
a number proves the branch is at 20 degrees. Measure the properties the request actually \
named -- a 20 degree branch, a 6 mm radius, a 3:1 contraction -- from your own geometry, \
and print each next to the number that was asked for. A property you did not measure is a \
property you did not build.

**Name the patches where you create the surfaces.** A patch name is decided on the face as \
it is made or selected. Never assign a patch by asking where a face sits in the bounding \
box -- on any bend or elbow that labels the wrong end and says nothing while it does it.

**Export the named patches with `export_patches`, in one call.**

    import sys; sys.path.insert(0, "{{reference_dir}}")
    from cad_export import export_patches
    export_patches(fluid, {{{{"inlet": ins, "outlet": outs, "walls": ...}}}}, tolerance=2.5e-4)

One shape, its own faces, one patch may be `...` for the rest, `tolerance` in metres. It \
refuses a face that is in two patches, in none, or not a face of that shape, and it prints \
what it wrote -- triangles and area per patch, the extent in metres, and the open-edge and \
winding counts of the union. **Do not export the patches one at a time with `export_stl`.** \
It meshes whatever you hand it, so patch-by-patch each one discretises the shared edges \
separately and the union has a torn seam along every patch boundary -- holes snappyHexMesh \
reads as leaks, under a `checkMesh` that passes anyway. `export_patches` tessellates the \
shape once and cuts the patches out of that, so the seams weld by construction. \
`export_stl` is still the right call for a preview of one shape you are looking at.

**Metres, always.** OpenFOAM has no units: it reads the mesh's numbers as metres, and a \
74 mm duct built in millimetres becomes a 74 m duct, at a thousandth of the Reynolds \
number, with `checkMesh` and the picture both perfectly happy. An imported STEP usually \
declares millimetres; scale once, early, and say which way.

**Coarse first.** Get the shape right at a few thousand cells, look at it, and only then \
refine. A wrong shape at 2 million cells is 20 wasted minutes.

**Say what you did not check.** If a property could not be measured, or something came \
back warning and you decided to live with it, write it in your closing lines. A gap you \
named is worth more than a verdict you implied.

# Finishing

When the geometry is built and the mesh exists, call the `declare_complete` tool with \
`outcome: "complete"`, and put your closing summary in the prose beside it: what you \
built, the numbers you measured against the request, what you could not check.

That call runs the checks. **`checkMesh` is the binding one** -- the bare form, run per \
region -- and a run that declares complete over a mesh it refuses is handed the refusal \
and keeps working. The others are advisory in one specific sense: being right about your \
geometry is enough to get past them.

**`checkMesh -allGeometry` is a reference reading, not the bar.** It adds checks the bare \
form does not run at all -- cell determinant, face interpolation weight, concave cells, \
face tets -- so it will fail meshes that are entirely usable: on this corpus it failed 14 \
of the 22 meshes the gate passed, at 1.7-6% of cells, and nine of those were put through \
`simpleFoam`, where eight converged or were still converging and none diverged. \
snappyHexMesh and gmsh both routinely produce meshes that fail it. Run it if you want the \
extra information and say what it told you, but **do not treat it as a defect to chase, \
and do not rebuild a working mesh because of it** -- a corpus run lost its whole step \
budget doing exactly that. What you must not do is the converse: run the stricter form, \
see it fail, and re-run the barer one as though that repaired something. The others are advisory in one specific sense: being right about your geometry \
is enough to get past them. **A warning you neither fix nor waive means the declare is \
not accepted, exactly as a failing `checkMesh` is not accepted, and it comes back to \
you.**

**If you already know an advisory check is going to flag something that is correct, say so \
on the same call.** Put it in `waive` with your reason -- an open surface because the part \
is a zero-thickness baffle, a meshing point outside the exported surface because the flow \
is external. Said before you see the result that is a prediction about your own geometry, \
it is recorded as one, and it finishes in a single call. Said after the check has flagged \
it is still accepted and recorded differently. Naming a check that then does not flag is \
recorded too, and means you expected something about your own geometry that was not there.

Two of them are worth knowing precisely, because it is easy to be right about the geometry \
and wrong about the check. **`union_closure` welds every STL in the directory into a \
single surface and counts the free edges of that union.** Individual patch files are open \
surfaces by construction and that is not what it measures, so "each patch is a separate \
sheet" does not explain a non-zero count. **`normals` counts edges walked twice in the \
same direction on that same union, which is winding consistency and not orientation.** \
Whether your faces point into the fluid or out of it is not what it measures and does not \
waive it; two triangles wound opposite ways is. A surface can be correctly outward-facing \
throughout and still fail it, and a count of zero says nothing about which way the \
normals point.

**After `checkMesh` and the gates accept a declare, an independent reviewer looks at the \
mesh.** It did not build the shape and has not read this thread: it is shown the request, \
your closing summary and the mesh drawn from several views, and it judges whether what is \
there is what was asked for. It may hand back problems it is confident change the answer \
-- the wrong topology, sharp corners where the request implies curves, proportions off, a \
named gap one cell wide, inlet and outlet on the wrong ends. Those come back to you as \
work, exactly as a failing check does, at most twice. Fix what is wrong, or if the \
reviewer is mistaken say why in your closing summary, and declare again; the third declare \
is accepted, and any concern still standing is reported up with the result rather than \
argued about.

**When the request cannot be answered correctly** -- a file that declares no length unit, a \
request that states no dimension at all -- call `declare_complete` with \
`outcome: "refuse"` and a one-line `reason`, and build nothing. Reporting up is the work in \
that case; guessing is not.

**Every number you chose rather than were given is an assumption, and it goes in your \
closing summary as one.** Short of a refusal, a request will still leave things open -- a \
wall thickness nobody stated, a domain extent, where "near the floor" is -- and choosing \
is your job. Recording the choice is also your job: say which numbers came from the \
request and which came from you, in the same place you report what you measured. A \
dimension written down as though it had been given is the one kind of wrong answer nobody \
downstream can see, because it is self-consistent everywhere it appears and the geometry \
built from it checks out perfectly.

If you run out of steps or seconds before you get there, `checkMesh` still runs on \
whatever is in the directory and you are told what it found -- an unexamined mesh is the \
failure this desk exists to end. And if the machine cannot be reached at all, that is \
reported as unreachable rather than as a verdict: unreachable is not failure.
"""

CORE_NUDGE = (
    "That is 12 cells and nothing in the case directory has been meshed yet. Whatever is "
    "left to work out about the shape, work it out in the mesh rather than before it: "
    "build the coarsest version that exists at all, run the mesher, and look at it. A "
    "shape that is 80% right and on disk is worth more than one that is exact and is not.")
"""The nudge with the sentence naming a recipe folder taken out.

The recipe folders are not in the core, so the text that pointed at them cannot be either
-- and a brief that names a path the workspace does not have is how a desk spends its
opening steps looking for what it was told it had. Measured on a local run of the desk
this replaces: seven steps of twenty-seven."""

CHECKMESH_TIMEOUT_S = 600

GATE_TIMEOUT_S = 300
"""How long the advisory gates may take without the watcher calling the run stale.

Declared for the same reason `checkMesh` is: the 420 s staleness threshold counts silence,
and a gate run is silent. T23 welded 544,306 triangles, which is the shape of the case
that makes this more than a formality."""
"""`checkMesh` on a few million cells is a minute; ten is a mesh that is not coming back."""

_STATS = {
    "points": re.compile(r"^\s*points:\s*(\d+)", re.M),
    "faces": re.compile(r"^\s*faces:\s*(\d+)", re.M),
    "cells": re.compile(r"^\s*cells:\s*(\d+)", re.M),
}
_BOX = re.compile(
    r"Overall domain bounding box \(([-\d.eE+ ]+)\) \(([-\d.eE+ ]+)\)")
_VERDICT = re.compile(r"^\s*(Mesh OK\.|Failed \d+ mesh checks?\.)", re.M)


def check_command(region: str) -> str:
    """`checkMesh` for one region, and nothing else run beside it.

    `-region` rather than a region loop in the shell, because a conjugate case's regions
    are separate meshes and OpenFOAM's own verdict is per mesh. The output is taken whole:
    it already carries the cell, face and point counts and the bounding box, so there is
    nothing to ask a second tool for.

    **Why the bare form and not `-allGeometry`.** `-allGeometry` was binding here briefly
    on 2026-09-16 and was reverted the same day on solver evidence. It is not a stricter
    setting of the same checks -- it adds checks the bare form does not run at all (cell
    determinant, face interpolation weight, concave cells, face tets), so a mesh the bare
    form calls `Mesh OK.` has not passed them, it has not been asked. On the sol corpus it
    failed 14 of the 22 meshes the bare form passed, at 1.7-6% of cells.

    Those meshes solve. `simpleFoam`, laminar, on the nine of eleven newly-failing cases
    that have an inlet and an outlet: four converged to 1e-5 on p and U (T2, T4, T14,
    T18), four more were still descending at the 300-iteration cap (T23 275x over the run,
    T12 88x, T17 45x, T11 18x), and none diverged, hit a floating-point exception, or
    errored. Only T16 stalled, at 2.2e-3 -- and T16 was already failing on other grounds.

    OpenFOAM's own meshers produce meshes that fail it: snappyHexMesh made 5 of those 14
    and gmsh 6. There is also no single vendor verdict to defer to -- bare, `-meshQuality`
    with the shipped `meshQualityDict`, and `-allGeometry` score this corpus 21, 13 and 10
    of 26, and they disagree in both directions (T19 passes `-allGeometry` and fails
    `-meshQuality`; T2, T14, T15, T16 and T24 do the reverse).

    So `-allGeometry` is **a reference reading, not a gate**. Run it, record what it says,
    rank failures with it -- and do not fail a run on it. It measures how much concavity a
    cut-cell mesher leaves near curved surfaces, which is a property of snappyHexMesh and
    gmsh rather than of the desk's work.
    """
    flag = f" -region {shlex.quote(region)}" if region else ""
    return f"checkMesh{flag} 2>&1 | tail -n 200"


def read_checkmesh(output: str) -> dict[str, Any]:
    """What `checkMesh` said, as numbers. Its own words are kept beside them."""
    found: dict[str, Any] = {"ok": False, "verdict": "", "cells": 0, "faces": 0,
                             "points": 0, "bounds": [], "output": (output or "")[-2000:]}
    verdict = _VERDICT.search(output or "")
    if verdict:
        found["verdict"] = verdict.group(1)
        found["ok"] = verdict.group(1) == "Mesh OK."
    for name, pattern in _STATS.items():
        match = pattern.search(output or "")
        if match:
            found[name] = int(match.group(1))
    box = _BOX.search(output or "")
    if box:
        try:
            low = [float(v) for v in box.group(1).split()]
            high = [float(v) for v in box.group(2).split()]
            if len(low) == 3 and len(high) == 3:
                found["bounds"] = low + high
        except ValueError:
            pass
    return found


def verify(backend: Any, case_dir: str, case_rel: str = "",
           mark: Any = None) -> Check:
    """The whole of the core finish check: is there a mesh, and does `checkMesh` pass it.

    A workspace that cannot answer is `unreachable` -- not a pass, not a verdict about the
    mesh, and never an exception into the middle of a run. Three real runs had a finished
    mesh reported as missing because a container recycled inside the one attempt."""
    try:
        regions = mesh_regions(backend, case_dir)
    except Exception as exc:  # noqa: BLE001 - the workspace, not the mesh
        return _unreachable(exc)

    if not regions:
        return Check(ok=False, regions=[],
                     missing=["nothing in the case directory has been meshed yet"],
                     findings=[Finding("checkMesh", "fail",
                                       "no constant/polyMesh and no constant/*/polyMesh",
                                       "there is no mesh here to judge",
                                       "mesh it, however coarsely, and look at it")])

    findings: list[Finding] = []
    missing: list[str] = []
    raw: dict[str, Any] = {"regions": {}}
    cells = faces = points = 0
    bounds: list[float] = []
    verdicts: list[str] = []
    for region in regions:
        if mark:
            # Per region rather than once: a conjugate case is two `checkMesh` runs and
            # one declaration up front would have under-stated the silence by half.
            mark("check", CHECKMESH_TIMEOUT_S)
        try:
            outcome = backend.exec(check_command(region), cwd=case_dir,
                                   timeout_s=CHECKMESH_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            return _unreachable(exc)
        read = read_checkmesh(outcome.output or "")
        raw["regions"][region or "(single)"] = read
        named = f"{region}: " if region else ""
        cells += read["cells"]
        faces += read["faces"]
        points += read["points"]
        bounds = _union(bounds, read["bounds"])
        verdicts.append(f"{named}{read['verdict'] or 'checkMesh said nothing readable'}")
        if read["ok"]:
            findings.append(Finding("checkMesh", "pass",
                                    f"{named}{read['verdict']} ({read['cells']:,} cells)"))
        else:
            reason = (f"{named}{read['verdict']}" if read["verdict"] else
                      f"{named}checkMesh wrote no verdict this run could read")
            findings.append(Finding(
                "checkMesh", "fail", reason,
                "OpenFOAM's own verdict on this mesh, and the only one this desk keeps",
                "read what it failed on above and mesh it again"))
            missing.append(reason)

    return Check(ok=not missing, regions=regions, cells=cells, faces=faces, points=points,
                 bounds=bounds, checkmesh="; ".join(verdicts), findings=findings,
                 missing=missing, raw=raw)


def _unreachable(exc: Exception) -> Check:
    return Check(
        ok=False, unreachable=True, error=str(exc), regions=[],
        missing=[f"the workspace did not answer, so the mesh could not be checked -- it "
                 f"may well be there ({exc})"],
        findings=[Finding("workspace", "skipped",
                          f"the workspace did not answer ({exc})",
                          "nothing is known about the mesh either way")])


def _union(current: list[float], other: list[float]) -> list[float]:
    """The bounding box over regions -- a conjugate case has one domain, not two."""
    if len(other) != 6:
        return current
    if len(current) != 6:
        return list(other)
    return [min(current[i], other[i]) for i in range(3)] + \
           [max(current[i + 3], other[i + 3]) for i in range(3)]


TOOLBOX_DIR = Path(__file__).resolve().parents[1] / "toolbox"
"""Where the reference files are read from, in this process.

The desk never learns this path and cannot render it: `self.toolbox` is
empty, and what reaches the workspace is two files under a name the brief
gives. Reading them here is the same legitimacy `buildup/probes.py` has for
importing the toolbox -- it happens on our side of the backend."""

REFERENCE_DIR = ".reference"
"""Where the files this arm is handed live inside its workspace.

Not `.toolbox`: that name is itself a house surface (`isolation.TOOLBOX_NAME`), and a
desk that writes it has found us whether or not the directory exists. This one is the
arm's own, holds only what the arm was given, and is named in the brief."""

REFERENCE_FILES = ("b123d_api.md", "cad_export.py")
"""What the core desk is handed, and the whole of it. Two files, each on its own evidence.

`b123d_api.md`, the reading: in `core+declare_gate-20260913-124524-aa21`, 27 of the 36
cells that raised a named exception were the desk calling something that does not exist,
across 15 of 26 cases, and 17 of those 27 were build123d -- exactly that file's subject.
Handing it over took `AttributeError` 11 -> 2, `NameError` 4 -> 0 and
`ModuleNotFoundError` 1 -> 0 in `core+reference-20260914-093903-3472`.

`cad_export.py`, the code: **111 of the 139 runs that delivered a mesh, across twelve
sweeps, wrote their own per-patch STL export**, and 30 of those 111 shipped a patch set
with open or flipped edges -- 15 of the 26 cases hit at least once, worst at 94,803
flipped edges of 100,385 triangles. The cause is one tessellation per patch, and it is
not even reliably wrong: OpenCASCADE caches a triangulation on the shape, so a loop of
`export_stl` calls at one tolerance welds by luck and the same loop at two tolerances,
or per face, or against a face built beside the solid, does not. Closes
`exported_surface_winding_inconsistent`, `exported_surface_has_open_edges` /
`no_closure_assertion_between_export_and_meshing`, `named_patches_land_empty_under_a_
passing_checkmesh` and `exported_surface_duplicated_in_trisurface`.
`tests/test_buildup_cad_export.py` demonstrates each in its absence.

Nothing else from the toolbox comes with them: `house_names` still covers every other
name in there, so a run that reaches for `cad_convert.py` still grades contaminated."""


def brief(step_timeout: int | float = STEP_TIMEOUT_S) -> str:
    """The core brief as the desk receives it.

    A renderer rather than a raw `.format` at every call site: `CORE_SYSTEM` carries
    placeholders, and a caller that forgets one gets a `KeyError` at the moment it can
    least afford one. Anything that wants to read the brief -- a test, a report, the desk
    itself -- goes through here."""
    return CORE_SYSTEM.format(step_timeout=step_timeout, reference_dir=REFERENCE_DIR)


class CoreDesk(CadDesk):
    """The same loop, told less, judged by `checkMesh`, and given no tools at all.

    What it is, after 2026-09-18: the desk that ships. The brief, the nudge and the
    absent toolbox are the three things the corpus measured and they are what is left
    here. Everything else -- the loop, the kernel, the cell log, the finish check and the
    advisory gates -- is `CadDesk`'s, inherited rather than overridden, because a desk
    that overrides them is a second implementation of the thing being measured.

    `self.toolbox` is emptied so that nothing can render a path to a directory this run's
    workspace does not have -- and `isolation.scan_run` is what proves it afterwards,
    because a desk that is merely not told about a toolbox has been known to find one.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.toolbox = ""
        """Still empty, and the reference below does not change that.

        `self.toolbox` is what renders a path to the *toolbox*, and this desk has none.
        What it has is `REFERENCE_FILES` copied into its own workspace under
        `REFERENCE_DIR`, which the brief names directly. The distinction is the
        measurement: a desk that can render `/work/.toolbox` can go looking in it, and
        two files it was handed by name are not a catalogue it went looking for.

        The finish check is the exception and always was: `check.verify` runs
        `mesh_look.py`, `cad_audit.py` and `domain_probe.py` out of the real toolbox over
        the backend, in the harness's own process, where the desk cannot reach them and
        never sees their path."""

    def _prepare(self, case_dir: str) -> None:
        """Put the two reference files in the case directory, over the backend.

        Into the *case* directory, because that is what the brief calls
        `.reference/b123d_api.md`. Putting them at the workspace root is what the first
        attempt at this addition did: the files existed, the brief named them, and no
        desk could reach them -- T16 ran the grep the brief suggests and got an empty
        string back.

        Over the backend rather than with `shutil`, which is what
        `scripts/cad_buildup.py` does and could do because its workspace is a local
        directory. A session's case is on the volume, so a file copied with `shutil` here
        would land on the machine holding the conversation and nowhere the desk can read.
        That is the same mistake as the first one, made one layer down.

        A missing source file raises, and `run` turns that into a refusal before the
        kernel starts: `b123d_api.md` is generated and can legitimately be absent from a
        checkout, and finding that out on cell nine costs a model call for every cell
        before it.
        """
        for name in REFERENCE_FILES:
            source = TOOLBOX_DIR / name
            if not source.is_file():
                how = (f"run `python3 {TOOLBOX_DIR / 'b123d_api.py'}` to write it"
                       if name.endswith(".md") else "it is source and should be in the repo")
                raise FileNotFoundError(
                    f"this desk is given {name} and it is not on disk at {source}; {how}")
            self.backend.put_file(f"{case_dir}/{REFERENCE_DIR}/{name}",
                                  source.read_bytes())

    def _system(self) -> str:
        return brief(STEP_TIMEOUT_S)

    def _nudge(self) -> str:
        return CORE_NUDGE

    def _verify(self, case_rel: str, request: str, script: str) -> Check:
        """`checkMesh` per region, and nothing else. The floor §1 set, and still it.

        This desk was briefly given the whole of `check.verify` -- the render, the patch
        naming, the request-scale reading, the rebuild script, the replay -- on the
        argument that the shipped desk's finish was the better of the two. Two of those
        five were then removed for failing the desk over things it had no move against,
        and a third (`render`) turned out to fail this desk for an artifact its brief
        never asks for, in words that name the toolbox it is briefed as not having.

        Which is the argument for coming back here rather than auditing the rest one at a
        time. **Additions arrive when a measured failure asks for them**, carrying that
        failure and a test. None of those five arrived that way: they came across in a
        port, as a set, without a sweep between them and the corpus.

        `request` and `script` are the wider desk's inputs. The script is written into the
        case as the artifact it is -- that is not a check and nothing fails on it.
        """
        _leave_script(self.backend, self.case_dir, script)
        return verify(self.backend, self.case_dir, case_rel, mark=self._mark)

    # -- the declared finish, and the advisory gates that run at it -------------

    def _declare(self, payload: dict[str, Any], case_rel: str,
                 request: str) -> tuple[Check, str, list[str]]:
        """Bind on `checkMesh` alone, and report the surface findings.

        The gate arrived with its own measured failure and is the one part of the port
        that stays: `union_closure` read 259 free edges on T26, the record kept them, the
        run scored `passed: true`, and nothing told the desk. It is advisory because four
        of the six checks behind it have been wrong at least once, so a gate built on them
        blocks correct work while a warning costs a waiver and a line in the record.

        What changed in the port and is kept: the numbers come from `cad_audit.py` and
        `domain_probe.py` run **over the backend** rather than from `buildup/probes.py`
        read off a local `Path`. The probe version worked only where the case is a local
        directory, so on a hosted workspace every state came back `n/a` and the gate
        passed everything silently.
        """
        check = self._verify(case_rel, request, self.log.script())
        try:
            self._mark("gates", GATE_TIMEOUT_S)
            states = gate.evaluate(
                advisory_findings(self.backend, self.case_dir),
                payload.get("waive") or (), self._warned)
        except Exception as exc:  # noqa: BLE001 - an advisory check may not end a run
            states = [gate.GateState("gates", gate.NOT_RUN,
                                     f"{type(exc).__name__}: {exc}")]
        self._declares.append(gate.Declaration(
            outcome="complete", reason=str(payload.get("reason") or ""),
            states=states, checkmesh_ok=bool(check.ok)).as_dict())
        self._warned |= {s.check for s in states
                         if s.state in (gate.WARNED, gate.XFAIL, gate.WAIVED)}
        unresolved = [s.check for s in states if s.state == gate.WARNED]
        return check, gate.render(states, self.case_dir), unresolved
