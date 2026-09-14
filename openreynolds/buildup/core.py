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

from ..cad.agent import CELL_TOOL, DECLARE_TOOL, STEP_TIMEOUT_S, CadDesk
from . import gate, probes
from ..cad.brief import CAD_DONE
from ..cad.check import Check, Finding, mesh_regions

CORE_SYSTEM = f"""\
You are the CAD desk. You build one geometry -- authored from a description, or prepared \
from a CAD file somebody sent -- and its OpenFOAM mesh, on the machine that has OpenFOAM \
on it, and then you stop. You do not solve, you do not set boundary conditions, you do \
not touch anything outside the case directory you are given.

# How you act

Every message you send contains exactly one fenced python block:

```python
your code here
```

That block is run as one cell in a persistent IPython kernel on the machine, with your \
case directory as its working directory, and its output comes back to you as the next \
message. Nothing else you write runs.

**One short runnable cell a turn.** Send it, look at what it printed, and build on that \
next turn. Not a whole build in one block: a cell that does one thing tells you which \
thing was wrong, and a cell that does nine tells you only that something was.

The kernel is **persistent**: a name you bind in one cell is still bound in the next, so \
a shape is a variable you can measure, tessellate or draw at any later step rather than a \
file you have to reload. Shell commands are reachable from inside a cell -- \
`subprocess.run(["blockMesh"], check=True)` -- so there is one channel, not two. Write \
them that way and not as `!blockMesh`: the accepted cells are concatenated into \
`build.py` and that file is re-run as `python3 build.py`, where a `!` line is a syntax \
error.

Whatever a cell draws or displays comes back to you attached: a matplotlib figure, a \
displayed image, and any `.png` a command you ran wrote. That is how you see; it is the \
only way you see.

A cell is given {{step_timeout}} s of the conversation's attention. **That window \
expiring does not kill your cell.** It is reported back as still running, with whatever \
it printed so far, and your next step either polls it or interrupts it on purpose. A cell \
that might outrun the window writes its result to disk -- `export_step(...)` for an \
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
property and takes none. It is the only file of ours you have; there is nothing else to \
look for.

# The rules of this desk

**Write build123d in algebra mode, directly.** Every operation names its operands -- \
`body = plate - channel`, `part = Pos(0, 0, H) * lid` -- so a line reads without replaying \
the ones above it. Not builder mode (`with BuildPart() ...`): the two do not mix well and \
the ambient state is the thing being avoided.

**Your accepted cells are the script you leave behind.** They are concatenated into \
`build.py` in the case directory, and that file has to reproduce what you built when it is \
run from empty. So a cell is correct only if it still works in sequence: parameters as \
named constants at the top of the cell that first needs them; no dependence on a name that \
no accepted cell binds (a name from a cell that errored is still live in this kernel and \
will not exist on replay); nothing whose effect depends on having been run once already -- \
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
it is made or selected, and each named group is exported to its own STL. Never assign a \
patch by asking where a face sits in the bounding box -- on any bend or elbow that labels \
the wrong end and says nothing while it does it.

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

That call runs the checks. **`checkMesh` is the binding one** -- it runs per region, and \
a run that declares complete over a mesh it refuses is handed the refusal and keeps \
working. The others are advisory in one specific sense: being right about your geometry \
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
    nothing to ask a second tool for."""
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


REFERENCE_DIR = ".reference"
"""Where the files this arm is handed live inside its workspace.

Not `.toolbox`: that name is itself a house surface (`isolation.TOOLBOX_NAME`), and a
desk that writes it has found us whether or not the directory exists. This one is the
arm's own, holds only what the arm was given, and is named in the brief."""

REFERENCE_FILES = ("b123d_api.md",)
"""What the core desk is handed, and the whole of it.

One file, added on measured evidence rather than because it seemed useful: in
`core+declare_gate-20260913-124524-aa21`, 27 of the 36 cells that raised a named
exception were the desk calling something that does not exist, across 15 of 26 cases,
and 17 of those 27 were build123d -- exactly this file's subject. Nothing else from the
toolbox comes with it: `house_names` still covers every other name in there, so a run
that reaches for `cad_convert.py` still grades contaminated."""


def brief(step_timeout: int | float = STEP_TIMEOUT_S) -> str:
    """The core brief as the desk receives it.

    A renderer rather than a raw `.format` at every call site: `CORE_SYSTEM` carries
    placeholders, and a caller that forgets one gets a `KeyError` at the moment it can
    least afford one. Anything that wants to read the brief -- a test, a report, the desk
    itself -- goes through here."""
    return CORE_SYSTEM.format(step_timeout=step_timeout, reference_dir=REFERENCE_DIR)


class CoreDesk(CadDesk):
    """The same loop, told less, judged by `checkMesh`, and given no tools at all.

    It differs from `CadDesk` in exactly three places, which is why those three are seams
    in the loop rather than branches: the brief, the nudge, and what verifies the finish.
    `self.toolbox` is emptied so that nothing can render a path to a directory this run's
    workspace does not have -- and `isolation.scan_run` is what proves it afterwards,
    because a desk that is merely not told about a toolbox has been known to find one.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.toolbox = ""
        """Still empty, and the reference below does not change that.

        `self.toolbox` is what renders a path to the *toolbox*, and this arm has none.
        What it has is one file copied into its own workspace under `REFERENCE_DIR`,
        which the brief names directly. The distinction is the measurement: a desk that
        can render `/work/.toolbox` can go looking in it."""
        self._declares: list[dict[str, Any]] = []
        self._warned: set[str] = set()
        """Which advisory checks have raised a concern on some earlier declare. A waiver
        naming a check already in here is a reaction; one naming a check that is not is a
        prediction. `gate.evaluate` does the labelling and the desk cannot reach it."""

    def _system(self) -> str:
        return brief(STEP_TIMEOUT_S)

    def _nudge(self) -> str:
        return CORE_NUDGE

    def _verify(self, case_rel: str, request: str, script: str) -> Check:
        # `request` and `script` are the wider desk's inputs: the property criteria it
        # graded against the request, and the replay of the accepted log. Neither is in
        # the core -- §7 leaves the replay criterion open, and grading the request is the
        # supervisor's -- so they are deliberately unused here.
        return verify(self.backend, self.case_dir, case_rel, mark=self._mark)

    # -- the declared finish, and the advisory gates that run at it -------------

    def _tools(self) -> list[dict[str, Any]]:
        return [CELL_TOOL, DECLARE_TOOL]

    def _declare(self, payload: dict[str, Any], case_rel: str,
                 request: str) -> tuple[Check, str, list[str]]:
        """Run every check, bind on `checkMesh` alone, and report the rest.

        The advisory half is why this exists. `union_closure` measured 259 free edges on
        T26 -- the tread-column tangency the case was written around -- wrote it into the
        record, and the run scored `passed: true`, because the probes observe from the
        supervisor and have no channel into the run. Nothing told the desk. This is the
        channel, and it stays advisory: four of the six probes have been wrong at least
        once, so a gate built on them would block correct work, while a warning built on
        them costs nothing when wrong and is the only thing that will ever produce the
        evidence to repair them.
        """
        check = self._verify(case_rel, request, self.log.script())
        states: list[gate.GateState] = []
        try:
            self._mark("gates", GATE_TIMEOUT_S)
            found = [r.as_dict() if hasattr(r, "as_dict") else dict(r)
                     for r in probes.run_all(Path(self.case_dir), {})]
            states = gate.evaluate(found, payload.get("waive") or (), self._warned)
        except Exception as exc:  # noqa: BLE001 - an advisory check may not end a run
            states = [gate.GateState("gates", gate.NOT_RUN, f"{type(exc).__name__}: {exc}")]
        # What has already fired is what separates a prediction from a reaction on the
        # next declare, and the desk does not get a say in it.
        self._declares.append(gate.Declaration(
            outcome="complete", reason=str(payload.get("reason") or ""),
            states=states, checkmesh_ok=bool(check.ok)).as_dict())
        # An unresolved warning is treated exactly as a failing `checkMesh` is: the
        # declare is not accepted and comes back. There is no once-only allowance and no
        # separate bound -- `turns >= max_steps` and the `no-progress` alarm already
        # bound a desk that keeps declaring over a refusal, and this is that.
        #
        # The consequence is the point: **a waiver is the only way past a warning**, so
        # the desk must either fix it or say why it is correct. A pass rate that falls
        # because a desk would do neither is the measurement, not a regression.
        self._warned |= {s.check for s in states
                         if s.state in (gate.WARNED, gate.XFAIL, gate.WAIVED)}
        unresolved = [s.check for s in states if s.state == gate.WARNED]
        return check, gate.render(states, self.case_dir), unresolved
