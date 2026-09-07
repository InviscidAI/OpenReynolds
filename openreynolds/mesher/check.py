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
from dataclasses import dataclass, field
from typing import Any

LOOK = "/work/.toolbox/mesh_look.py"
RENDER_REL = "renders/mesh_look.png"
JSON_REL = "renders/mesh_look.json"
TIMEOUT_S = 280
"""checkMesh on a few million cells is a minute; the render is seconds."""

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
        if self.build:
            out.append("rebuilds with: " + ", ".join(self.build))
        return out


def verify(backend: Any, case_dir: str, case_rel: str, request: str = "") -> Check:
    """Run the check on the workspace and read the verdict.

    One command: draw the mesh, measure it, run checkMesh, write the JSON, print it.
    A workspace that cannot answer at all is a failed check with the reason in it --
    never a pass, and never an exception into the middle of a tool call.
    """
    cmd = (f"python3 {LOOK} . --out {RENDER_REL} --json {JSON_REL} >/dev/null 2>&1; "
           f"cat {JSON_REL}")
    try:
        outcome = backend.exec(cmd, cwd=case_dir, timeout_s=TIMEOUT_S)
        payload = _json_in(outcome.output or "")
    except Exception as exc:  # noqa: BLE001 - the workspace, not the mesh
        return Check(ok=False, missing=[f"the check could not be run: {exc}"],
                     error=str(exc))
    if payload is None:
        tail = (outcome.output or "").strip()[-800:]
        return Check(
            ok=False,
            missing=["the check wrote no readable answer -- run "
                     f"`python3 {LOOK} . --out {RENDER_REL} --json {JSON_REL}` yourself "
                     "and fix what it says" + (f"\n  {tail}" if tail else "")],
            error="no json",
        )
    return read(payload, case_rel, case_dir, request)


def read(payload: dict[str, Any], case_rel: str = "", case_dir: str = "",
         request: str = "") -> Check:
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
    if payload.get("error"):
        missing.append(str(payload["error"]))

    check.missing = missing
    check.ok = not missing
    return check


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
