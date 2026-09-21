"""The finish line, which is not the agent's to declare.

`echo MESH_DONE` is a request to be checked, not a result. This module runs the check
on the machine that has the mesh -- `mesh_look.py --json`, which draws the mesh and
reads it back out of the files -- and turns the answer into either a finish or a piece
of work handed back in words.

What it insists on is deliberately short, and every item is something that has actually
shipped broken:

* a `constant/polyMesh` that exists, because "case written" was once read as "meshed";
* `checkMesh` clean, because a mesh with 40,000 illegal faces used to reach a solver;
* patches with names somebody chose, because `patch0`/`patch1` out of `gmshToFoam` is
  what a mesh looks like when nobody said which end was the inlet;
* an `Allmesh` that rebuilds it, because a mesh that cannot be rebuilt cannot be edited.

It does not check that the shape is the shape that was asked for. Nothing here can:
that is what the measurements the agent prints, and the picture a person looks at, are
for. Saying so is better than a green tick that means less than it appears to.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

LOOK = "/work/.toolbox/mesh_look.py"
RENDER_REL = "renders/mesh_look.png"
JSON_REL = "renders/mesh_look.json"
TIMEOUT_S = 280
"""checkMesh on a few million cells is a minute; the render is seconds."""

RETRY_PAUSE_S = 10.0
"""How long to wait before asking the workspace a second time. A recycled container is
back in seconds and the Volume under it never went anywhere."""

_UNNAMED = re.compile(r"^(patch|region|surface|volume)\d+$|^defaultFaces$")

_SIZED = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm|millimetres?|millimeters?|centimetres?|"
                    r"centimeters?|metres?|meters?|m)\b", re.I)
_IN_METRES = {"mm": 0.001, "millimetre": 0.001, "millimetres": 0.001,
              "millimeter": 0.001, "millimeters": 0.001,
              "cm": 0.01, "centimetre": 0.01, "centimetres": 0.01,
              "centimeter": 0.01, "centimeters": 0.01,
              "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0}

SCALE_SLACK = 100.0
"""How far the mesh may be from the size the request named before it is called wrong.

Generous on purpose: a flow box round a 20 mm sphere is legitimately twenty times the
body, and a request may name a small feature on a large part. What this is looking for
is the thousandfold -- a geometry built in millimetres and left there, which OpenFOAM
reads as metres and solves as a duct 74 m across. That one is not a judgement call.
"""


@dataclass
class Check:
    """What the machine says about the mesh, and whether that is a finish."""

    ok: bool = False
    unreachable: bool = False
    """The check could not be run at all -- the workspace was gone or would not answer.

    Different in kind from every other failure here, and the difference is what three
    runs got wrong: a Sandbox recycled mid-mesh, the check could not reach it, and the
    tool said "NOT a usable mesh" about a mesh that was built, checked and rendered and
    was sitting on the Volume the whole time. Nothing is known either way in this case,
    and saying so is the only honest answer."""
    missing: list[str] = field(default_factory=list)
    """The reasons it is not, in the words the agent is handed."""
    cells: int = 0
    points: int = 0
    faces: int = 0
    bounds: list[float] = field(default_factory=list)
    """xmin ymin zmin xmax ymax zmax, in metres."""
    patches: list[dict[str, Any]] = field(default_factory=list)
    checkmesh: str = ""
    """The verdict line, e.g. 'Mesh OK.' or 'Failed 2 mesh checks: ...'."""
    metrics: dict[str, float] = field(default_factory=dict)
    """max non-orthogonality, max skewness, aspect ratio -- what a solver cares about."""
    two_d: bool = False
    cell: float = 0.0
    """Representative cell size, m: the n-th root of (domain volume / cells)."""
    across: float = 0.0
    """How many cells span the SMALLEST extent of the domain. The one resolution
    number that means the same thing on a channel, a cavity and a pipe."""
    unchanged: bool = False
    """This mesh is indistinguishable from the one that was here before the run."""
    build: list[str] = field(default_factory=list)
    """The scripts left behind that rebuild this: Allmesh, build.py, blockMeshDict..."""
    render: str = ""
    """Where the picture is, written the way the calling session refers to it."""
    render_abs: str = ""
    """The same picture as a workspace path, which is what fetches it."""
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_refusal(self) -> str:
        """What the agent is told when it said done and it is not."""
        why = "\n".join(f"- {item}" for item in self.missing)
        return ("The check did not pass, so the run is not finished:\n" + why +
                "\n\nFix it and say done again when it is right.")

    def lines(self) -> list[str]:
        """The report a person and the calling agent both read."""
        out: list[str] = []
        if self.cells:
            shape = "2D (one cell thick)" if self.two_d else "3D"
            out.append(f"{self.cells:,} cells, {self.faces:,} faces, {self.points:,} points, {shape}")
        if self.bounds and len(self.bounds) == 6:
            x0, y0, z0, x1, y1, z1 = self.bounds
            out.append(f"bounds {x1 - x0:.4g} x {y1 - y0:.4g} x {z1 - z0:.4g} m "
                       f"(x {x0:.4g}..{x1:.4g}, y {y0:.4g}..{y1:.4g}, z {z0:.4g}..{z1:.4g})")
        if self.checkmesh:
            out.append(f"checkMesh: {self.checkmesh}")
        if self.metrics:
            out.append("  " + ", ".join(f"{k} {v:g}" for k, v in sorted(self.metrics.items())))
        if self.patches:
            out.append("patches:")
            for patch in self.patches:
                bits = [f"  {patch.get('name', '?')}", f"{patch.get('type', '?')}",
                        f"{patch.get('nFaces', 0):,} faces"]
                if patch.get("area"):
                    bits.append(f"area {patch['area']:.4g} m2")
                if patch.get("normal"):
                    n = patch["normal"]
                    bits.append(f"normal ({n[0]:+.2f} {n[1]:+.2f} {n[2]:+.2f})")
                if patch.get("center"):
                    c = patch["center"]
                    bits.append(f"at ({c[0]:.4g} {c[1]:.4g} {c[2]:.4g})")
                out.append(" ".join(bits))
        if self.across:
            out.append(f"resolution: ~{self.cell:.4g} m cells, {self.across:.0f} across "
                       "the narrowest extent of the domain")
        if self.build:
            out.append("rebuilds with: " + ", ".join(self.build))
        return out


def verify(backend: Any, case_dir: str, case_rel: str, request: str = "",
           before: dict[str, Any] | None = None) -> Check:
    """Run the check on the workspace and read the verdict.

    One command: draw the mesh, measure it, run checkMesh, write the JSON, print it.
    A workspace that cannot answer at all is a failed check with the reason in it --
    never a pass, and never an exception into the middle of a tool call.
    """
    cmd = (f"python3 {LOOK} . --out {RENDER_REL} --json {JSON_REL} >/dev/null 2>&1; "
           f"cat {JSON_REL}")
    outcome = None
    for attempt in (1, 2):
        try:
            outcome = backend.exec(cmd, cwd=case_dir, timeout_s=TIMEOUT_S)
            break
        except Exception as exc:  # noqa: BLE001 - the workspace, not the mesh
            if attempt == 2:
                # A container that recycles mid-mesh is a Modal preemption and comes
                # back in seconds; the mesh it was building is on the Volume and
                # outlives it. One retry recovers the check; failing it outright
                # reported three finished meshes as missing.
                return Check(
                    ok=False, unreachable=True, error=str(exc),
                    missing=[f"the workspace did not answer, so the mesh could not be "
                             f"checked -- it may well be there ({exc})"])
            time.sleep(RETRY_PAUSE_S)
    payload = _json_in(outcome.output or "")
    if payload is None:
        tail = (outcome.output or "").strip()[-800:]
        return Check(
            ok=False,
            missing=["the check wrote no readable answer -- run "
                     f"`python3 {LOOK} . --out {RENDER_REL} --json {JSON_REL}` yourself "
                     "and fix what it says" + (f"\n  {tail}" if tail else "")],
            error="no json",
        )
    return read(payload, case_rel, case_dir, request, before)


def read(payload: dict[str, Any], case_rel: str = "", case_dir: str = "",
         request: str = "", before: dict[str, Any] | None = None) -> Check:
    """The JSON `mesh_look.py --json` writes, turned into a verdict.

    Split from `verify` so the rules can be tested without a workspace: this function
    is the whole of what "done" means.
    """
    check = Check(raw=payload)
    check.cells = int(payload.get("cells") or 0)
    check.points = int(payload.get("points") or 0)
    check.faces = int(payload.get("faces") or 0)
    check.bounds = [float(v) for v in (payload.get("bounds") or [])]
    check.patches = list(payload.get("patches") or [])
    check.two_d = bool(payload.get("two_d"))
    check.cell, check.across = resolution(payload)
    check.unchanged = same_mesh(before, payload)
    check.build = list(payload.get("build") or [])
    check.metrics = {str(k): float(v) for k, v in (payload.get("metrics") or {}).items()}
    mesh_ok = bool(payload.get("checkmesh_ok"))
    check.checkmesh = str(payload.get("checkmesh") or "")
    render = str(payload.get("render") or "")
    if render.startswith("/"):
        check.render = check.render_abs = render
    elif render:
        # `mesh_look.py` reports the picture relative to the case, which is what a
        # person reads; a fetch needs the workspace path, and the two are different
        # strings often enough that keeping only one of them cost a missing picture.
        check.render_abs = f"{case_dir}/{render}" if case_dir else render
        check.render = f"{case_rel}/{render}" if case_rel else render

    missing: list[str] = []
    if not payload.get("polymesh"):
        missing.append(f"there is no constant/polyMesh in {case_rel or 'the case'} -- "
                       "nothing has been meshed yet")
    elif not check.cells:
        missing.append("constant/polyMesh is there but has no cells in it")
    if not mesh_ok:
        missing.append("checkMesh does not pass: " + (check.checkmesh or "no verdict was read")
                       + " -- read log.checkMesh for the failing checks")
    named = [p for p in check.patches if int(p.get("nFaces") or 0) > 0]
    if len(named) < 2:
        missing.append(f"the mesh has {len(named)} boundary patch(es) with faces on them; a "
                       "flow case needs at least an inlet, an outlet and walls")
    unnamed = [p.get("name", "?") for p in named if _UNNAMED.match(str(p.get("name", "")))]
    if unnamed:
        missing.append("these patches carry the mesher's default names rather than names "
                       f"somebody chose: {', '.join(unnamed)} -- name them where you build "
                       "the surfaces, or retype them in Allmesh")
    if not check.build:
        missing.append("there is no Allmesh (or build script) in the case, so nothing here "
                       "can be rebuilt or edited -- leave the script that made this mesh")
    if not check.render:
        missing.append("no picture of the mesh was drawn")
    off = scale_mismatch(request, check.bounds)
    if off:
        missing.append(off)
    wanted = asked_resolution(request)
    if wanted and check.across and check.across + 0.5 < wanted:
        missing.append(
            f"the request asks for at least {wanted} cells across and this mesh has "
            f"{check.across:.0f} (cells about {check.cell:.4g} m). Refine it -- a mesh "
            "that cannot resolve what the study is measuring passes every other check "
            "on this list")
    if check.unchanged:
        missing.append(
            "this mesh is IDENTICAL to the one that was in the case before this run -- "
            "same cell count, same bounds, same patch areas. Whatever was asked for, "
            "nothing about the mesh changed, so either the build script was not "
            "actually re-run or the change never reached it. Make the change, rebuild, "
            "and look at the numbers before saying done")
    if payload.get("error"):
        missing.append(str(payload["error"]))

    check.missing = missing
    check.ok = not missing
    return check


def resolution(payload: dict[str, Any]) -> tuple[float, float]:
    """Representative cell size, and how many cells span the domain's smallest extent.

    Deliberately crude and deliberately reported rather than judged. `cells` and
    `bounds` are the only two things every mesh here has, so `h = (volume/cells)**(1/n)`
    is the one resolution figure that can be computed for a channel, a cavity, a pipe
    and a flow box alike. A 2D mesh is one cell thick, so its third extent is excluded
    from both the volume and the span.
    """
    bounds = [float(v) for v in (payload.get("bounds") or [])]
    cells = int(payload.get("cells") or 0)
    if len(bounds) != 6 or cells <= 0:
        return 0.0, 0.0
    spans = [bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]]
    spans = [s for s in spans if s > 0]
    if payload.get("two_d") and len(spans) == 3:
        spans = sorted(spans)[1:]   # drop the one-cell thickness
    if not spans:
        return 0.0, 0.0
    volume = 1.0
    for s in spans:
        volume *= s
    cell = (volume / cells) ** (1.0 / len(spans))
    return cell, (min(spans) / cell if cell > 0 else 0.0)


_ASKED_CELLS = re.compile(
    r"(?:at least\s+)?(\d+)\s*cells?\s+(?:across|through|over|spanning)", re.I)


def asked_resolution(request: str) -> int:
    """A resolution the REQUEST named, e.g. "at least 40 cells across the gap".

    Why a caller has to say it, rather than the check having a threshold of its own:
    the measurements in this repository show a threshold cannot work. A plane channel
    at TEN cells across its height reproduced f*Re to within 2%, because the profile
    it has to represent is a smooth parabola. A lid-driven cavity at TWENTY cells
    across -- twice the resolution -- missed Ghia's centreline extrema by 18.7% and
    21.2%, because what it has to represent is a corner-driven recirculation. The same
    number is comfortably enough for one case and badly short for another, and nothing
    the mesh desk can see tells the two apart. The physics does, and the caller knows
    the physics.

    So: a stated requirement is enforced, an unstated one is REPORTED and never
    guessed at. The alternative -- a number picked here -- would fail the channel to
    catch the cavity.
    """
    best = 0
    for found in _ASKED_CELLS.findall(request or ""):
        try:
            best = max(best, int(found))
        except ValueError:
            continue
    return best


def same_mesh(before: dict[str, Any] | None, after: dict[str, Any]) -> bool:
    """Whether nothing measurable changed between two meshes of the same case.

    The failure this exists for, measured: asked to "make the channel 30 mm tall
    instead of 20", the desk rebuilt the case, ran its own look, reported "Rebuild is
    clean: bounds 0.200 x 0.020 x 0.002 m, checkMesh reports Mesh OK" -- and the
    channel was still 20 mm. Identical extent, identical 1,000 cells, identical
    answer. Every clause of the finish check passed, because after a no-op there IS a
    valid mesh of the right rough size. Nothing in the check compared what was ASKED
    against what CHANGED.

    A silent no-op is worse than a failure: the person gets a clean rebuild, a green
    checkMesh, a confident summary and the old shape.
    """
    if not before or not before.get("cells"):
        return False
    if int(before.get("cells") or 0) != int(after.get("cells") or 0):
        return False
    a = [float(v) for v in (before.get("bounds") or [])]
    b = [float(v) for v in (after.get("bounds") or [])]
    if len(a) != 6 or len(b) != 6:
        return False
    scale = max(abs(v) for v in b) or 1.0
    if any(abs(x - y) > 1e-6 * scale for x, y in zip(a, b)):
        return False
    # Bounds and cell count can both survive a real change -- widening a cylinder
    # inside a fixed box moves neither -- so the patch areas are the third witness.
    area = lambda rows: {str(r.get("name")): round(float(r.get("area") or 0.0), 12)
                         for r in (rows or [])}
    return area(before.get("patches")) == area(after.get("patches"))


def largest_length(request: str) -> float:
    """The biggest length the request names, in metres, or 0 when it names none.

    `"a duct 100 mm long and 20 mm tall"` is 0.1. Read off the words rather than
    assumed, so the comparison below is between two measurements.
    """
    best = 0.0
    for number, unit in _SIZED.findall(request or ""):
        try:
            metres = float(number) * _IN_METRES[unit.lower()]
        except (ValueError, KeyError):
            continue
        best = max(best, metres)
    return best


def scale_mismatch(request: str, bounds: list[float]) -> str:
    """Whether the mesh is the size the request asked for, to within a factor of 100.

    The failure this exists for: a geometry authored in millimetres and meshed
    without a scale. OpenFOAM has no units -- it reads the numbers as metres -- so a
    74 mm duct becomes a 74 m duct, every velocity is the wrong Reynolds number, and
    nothing in checkMesh or in the picture says a word about it. The only place the
    intended size is written down is the request, so that is what it is measured
    against.
    """
    asked = largest_length(request)
    if not asked or not bounds or len(bounds) != 6:
        return ""
    built = max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    if built <= 0:
        return ""
    ratio = built / asked
    if 1.0 / SCALE_SLACK <= ratio <= SCALE_SLACK:
        return ""
    return (f"the mesh is {built:.4g} m across and the request's largest dimension is "
            f"{asked:.4g} m -- {ratio:.0f}x out. OpenFOAM reads the mesh in metres, so a "
            "geometry built in millimetres has to be scaled (gmsh: multiply the "
            "coordinates or set Mesh.ScalingFactor; blockMesh: `scale 0.001;`; an "
            "existing mesh: `transformPoints -scale '(0.001 0.001 0.001)'`)")


def _json_in(text: str) -> dict[str, Any] | None:
    """The JSON object in a command's output, whatever the shell printed around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def look_command(case_dir: str) -> str:
    """The command a person or an agent runs by hand to see the same thing."""
    return f"python3 {LOOK} {shlex.quote(case_dir)} --out {RENDER_REL} --json {JSON_REL}"
