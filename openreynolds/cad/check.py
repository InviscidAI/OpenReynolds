"""The finish line for CAD and mesh as one job, which is not the desk's to declare.

`CAD_DONE` is a request to be checked, not a result. This module runs the check on the
machine that has the geometry and the mesh -- `mesh_look.py --json` per region,
`cad_audit.py` and `domain_probe.py` on the exported patch set -- and turns the answers
into either a finish or a piece of work handed back in words.

Three things it is, stated because each one is a rule somewhere else in the plan:

* **A coordinator, not a second geometry library.** The triangles are measured by the
  toolbox on the instance and arrive here as JSON. Nothing under `openreynolds/` outside
  `toolbox/` opens a CAD kernel, and this module reaches the toolbox the way everything
  above the backend does -- by running it over the backend and reading its `--json`.
  That is also why the `Finding` record below is re-hydrated rather than imported: the
  toolbox is data to the loop, copied to the instance, and importing it here would be a
  transport nobody above the protocol is allowed to know about.
* **`checkMesh` is the sole authority on mesh quality.** The thresholds copied below
  phrase a finding -- they put a number and its meaning in front of a reader -- and they
  never overrule the verdict. Two authorities on one number is a desk being told
  contradictory things about a mesh OpenFOAM has already judged.
* **Region-aware.** `constant/*/polyMesh` first, `constant/polyMesh` as the fallback,
  `checkMesh -region` per region, all of them required to pass. A conjugate case is a
  mesh that exists, and reporting "nothing has been meshed yet" about it is a gate
  rejecting a valid result.

What it insists on is deliberately short, and every item is something that has actually
shipped broken:

* a mesh that exists and has cells in it, because "case written" was once read as
  "meshed";
* `checkMesh` clean, because a mesh with 40,000 illegal faces used to reach a solver;
* patches with names somebody chose, because `patch0`/`patch1` out of `gmshToFoam` is
  what a mesh looks like when nobody said which end was the inlet -- **named, not
  counted**: a sealed cavity driven by sources on cell zones legitimately has one patch,
  so one patch is a warning with its assumption stated, not a refusal;
* a rebuild script that the case bundle will actually take, because a script under a
  name nobody captures is an artifact lost quietly;
* a picture, because nobody reads a mesh out of a table alone;
* the size the request asked for, because OpenFOAM reads the mesh in metres and a
  geometry left in millimetres passes every other check;
* the concatenated cell log re-running clean *from empty* and producing the same
  geometry, because a script that only runs where it was written is not the artifact it
  claims to be;
* unreachable reported as unreachable, because nothing is known in that case and saying
  so is the only honest answer.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from openreynolds.backend.base import WORKSPACE_ROOT

from . import gate

TOOLBOX = f"{WORKSPACE_ROOT}/.toolbox"
"""The hosted workspace's toolbox directory, and the default everywhere below.

These three paths are handed to a shell, and a shell gets the literal string: nothing
translates `/work` for it. On the hosted backend that is correct and this is the whole
story. On `LocalBackend` -- which is what the stack is developed against -- the root is
somewhere else, `python3 /work/.toolbox/mesh_look.py` finds nothing, and the check
reports **"the check wrote no readable answer"** about a mesh it never opened: a silent
wrong verdict rather than an error. So the directory is a parameter with the hosted
value as its default, and `verify` passes the one its backend reports."""

LOOK = f"{TOOLBOX}/mesh_look.py"
CAD_AUDIT = f"{TOOLBOX}/cad_audit.py"
DOMAIN_PROBE = f"{TOOLBOX}/domain_probe.py"
SURFACE_REL = "constant/triSurface"
MANIFEST_REL = f"{SURFACE_REL}/patches.json"
RENDER_REL = "renders/mesh_look.png"
JSON_REL = "renders/mesh_look.json"
BUILD_SCRIPT = "build.py"
"""What the concatenated cell log is left in the case as."""

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

NON_ORTHO_WARN = 70.0
NON_ORTHO_FAIL = 85.0
SKEWNESS_WARN = 4.0
SKEWNESS_FAIL = 10.0
ASPECT_WARN = 1000.0
"""Copies of `toolbox/preflight.py`'s thresholds, and the only copies in the tree.

This module cannot import the toolbox, so these are duplicated rather than shared, and
the duplication is pinned: `tests/test_cad_check.py` reads `preflight.py`'s source and
fails if a number here and a number there disagree.

They **phrase** findings. A number past one of them goes into `measured` and into
`meaning` so a reader sees it; it does not decide anything. `checkMesh`'s own verdict
decides, because it already judged this mesh and a second judge only ever disagrees.
"""

class Finding(NamedTuple):
    """One check's answer. `toolbox/preflight.py:93`, re-hydrated.

    Identical by contract, duplicated because the toolbox is not importable from here
    (it is copied to the instance and run there). `measured` is evidence and should
    survive being quoted on its own; `meaning` is the interpretation and is allowed to
    be wrong; `repair` is a suggestion and is allowed to be ignored.
    """

    check: str
    status: str
    measured: str
    meaning: str = ""
    repair: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "check": self.check,
            "status": self.status,
            "measured": self.measured,
            "means": self.meaning,
            "repair": self.repair,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        """A finding as a toolbox script serialised it -- note `means`, not `meaning`."""
        return cls(
            check=str(data.get("check") or "?"),
            status=str(data.get("status") or "ok"),
            measured=str(data.get("measured") or ""),
            meaning=str(data.get("means") or data.get("meaning") or ""),
            repair=str(data.get("repair") or ""),
        )


STATUSES = ("fail", "warn", "ok", "skipped")
_SEVERITY = {"fail": 3, "warn": 2, "ok": 1, "skipped": 0}


def worst_status(findings) -> str:
    return max((f.status for f in findings), key=lambda s: _SEVERITY.get(s, 0), default="ok")


def summarise(findings) -> dict[str, int]:
    return {status: sum(1 for f in findings if f.status == status) for status in STATUSES}


@dataclass
class Check:
    """What the machine says about the geometry and the mesh, and whether that is a finish."""

    ok: bool = False
    unreachable: bool = False
    """The check could not be run at all -- the workspace was gone or would not answer.

    Different in kind from every other failure here, and the difference is what three
    runs got wrong: a Sandbox recycled mid-mesh, the check could not reach it, and the
    tool said "NOT a usable mesh" about a mesh that was built, checked and rendered and
    was sitting on the Volume the whole time. Nothing is known either way in this case,
    and saying so is the only honest answer."""
    findings: list[Finding] = field(default_factory=list)
    """Every answer, in the register the toolbox already shares."""
    missing: list[str] = field(default_factory=list)
    """The reasons it is not a finish, in the words the desk is handed -- derived, one
    per failing finding."""
    regions: list[str] = field(default_factory=list)
    """The mesh regions found: `[""]` for a single-region case, `[]` for nothing meshed."""
    cells: int = 0
    points: int = 0
    faces: int = 0
    bounds: list[float] = field(default_factory=list)
    """xmin ymin zmin xmax ymax zmax, in metres; the union over regions."""
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
        """What the desk is told when it said done and it is not."""
        bullets: list[str] = []
        for finding in self.findings:
            if finding.status != "fail":
                continue
            text = f"- {finding.measured}"
            if finding.repair:
                text += f"\n  try: {finding.repair}"
            bullets.append(text)
        if not bullets:
            bullets = [f"- {item}" for item in self.missing]
        return ("The check did not pass, so the run is not finished:\n" +
                "\n".join(bullets) +
                "\n\nFix it and say done again when it is right.")

    def lines(self) -> list[str]:
        """The report a person and the calling agent both read."""
        out: list[str] = []
        if self.cells:
            shape = "2D (one cell thick)" if self.two_d else "3D"
            out.append(f"{self.cells:,} cells, {self.faces:,} faces, {self.points:,} points, {shape}")
        if len(self.regions) > 1:
            out.append("regions: " + ", ".join(self.regions))
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
                if patch.get("region"):
                    bits.append(f"in {patch['region']}")
                out.append(" ".join(bits))
        if self.build:
            out.append("rebuilds with: " + ", ".join(self.build))
        if self.findings:
            counts = summarise(self.findings)
            out.append("checks: " + ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s]))
            for finding in self.findings:
                if finding.status in ("fail", "warn"):
                    out.append(f"  {finding.status}: {finding.check} -- {finding.measured}")
        return out


# -- reaching the workspace ----------------------------------------------------


class _Unreachable(Exception):
    """The workspace did not answer, twice. Never a verdict about the mesh."""


def _ask(backend: Any, cmd: str, cwd: str, timeout_s: int = TIMEOUT_S):
    """One command, asked twice before it is given up on.

    A container that recycles mid-mesh is a preemption: it comes back in seconds and
    the Volume under it never went anywhere. Three real runs had a finished mesh
    reported as missing because the one attempt landed in that window.
    """
    for attempt in (1, 2):
        try:
            return backend.exec(cmd, cwd=cwd, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001 - the workspace, not the mesh
            if attempt == 2:
                raise _Unreachable(str(exc)) from exc
            time.sleep(RETRY_PAUSE_S)
    raise _Unreachable("no answer")  # pragma: no cover - the loop returns or raises


_REGIONS_CMD = (
    'for d in constant/*/polyMesh; do [ -f "$d/points" ] && '
    'echo "REGION:$(basename "$(dirname "$d")")"; done; '
    '[ -f constant/polyMesh/points ] && echo "SINGLE:"; true'
)


def mesh_regions(backend: Any, case_dir: str) -> list[str]:
    """The regions that have a polyMesh.

    Names for `constant/<name>/polyMesh`, `[""]` for a single-region
    `constant/polyMesh`, `[]` for nothing meshed yet.

    One answer to "is there a mesh here", shared with the desk's nudge on purpose. Two
    copies of that question is how a finish check and a nudge come to disagree about
    whether a conjugate case has been meshed, which is a bug already in the repo twice.
    """
    outcome = _ask(backend, _REGIONS_CMD, case_dir, timeout_s=60)
    named: list[str] = []
    single = False
    for line in (outcome.output or "").splitlines():
        line = line.strip()
        if line.startswith("REGION:") and line[7:]:
            named.append(line[7:])
        elif line.startswith("SINGLE:"):
            single = True
    if named:
        # A split case keeps its base `constant/polyMesh` around; the regions are the
        # mesh once it exists, so they win the fallback rather than sharing with it.
        return sorted(set(named))
    return [""] if single else []


_ZONES_SNIPPET = r"""
import json, pathlib, re, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(errors="replace") if path.exists() else ""
found = re.finditer(r"(\w+)\s*\{[^{}]*?cellLabels\s*(?:List<label>)?\s*(\d+)", text, re.S)
print(json.dumps([[m.group(1), int(m.group(2))] for m in found]))
"""


def toolbox_for(backend: Any) -> str:
    """This backend's toolbox directory, or the hosted one if it will not say."""
    root = str(getattr(backend, "workspace_root", "") or WORKSPACE_ROOT).rstrip("/")
    return f"{root}/.toolbox"


def _region_command(region: str, toolbox: str = TOOLBOX) -> str:
    """Draw the mesh, measure it, run checkMesh, read the cell zones, print both.

    One command per region rather than one per question: a workspace round trip is the
    expensive part, and a region that answers half-way is worse than one that does not.
    """
    suffix = f"_{region}" if region else ""
    render = RENDER_REL.replace(".png", f"{suffix}.png")
    payload = JSON_REL.replace(".json", f"{suffix}.json")
    zones = f"constant/{region}/polyMesh/cellZones" if region else "constant/polyMesh/cellZones"
    flag = f" --region {shlex.quote(region)}" if region else ""
    return (
        f"python3 {toolbox}/mesh_look.py . --out {render} --json {payload}{flag}"
        f" >/dev/null 2>&1\n"
        f"echo '@@CELLZONES@@'\n"
        f"python3 - {shlex.quote(zones)} <<'ORZONES'\n{_ZONES_SNIPPET}ORZONES\n"
        f"echo '@@JSON@@'\n"
        f"cat {payload} 2>/dev/null\n"
    )


def _cad_command(script_path: str) -> str:
    """One toolbox audit, run only where there is a patch set for it to read."""
    return (f"if [ -f {MANIFEST_REL} ]; then python3 {script_path} {SURFACE_REL} --json; "
            f"else echo 'no patch set'; fi")


def verify(backend: Any, case_dir: str, case_rel: str, request: str = "",
           script: str = "", mark: Any = None) -> Check:
    """Run the check on the workspace and read the verdict.

    A workspace that cannot answer at all is a failed check with the reason in it and
    `unreachable` set -- never a pass, never a verdict about the mesh, and never an
    exception into the middle of a tool call.

    `script` is the concatenated cell log, left in the case as `build.py`. It is an
    artifact, not a claim this checks: see `_leave_script`.

    `mark` declares the long turn-free stretches to whatever is watching from outside the
    run, with how long each may take. It arrived with `CoreDesk`, which is watched: a
    check is minutes of silence, the watcher's staleness threshold is 420 s, and without
    this it would kill a run for being checked -- twice over on a conjugate case, which is
    two `checkMesh` runs. Per region rather than once up front, for that reason. A mark is
    liveness and not progress: it does not advance the turn and does not count toward the
    `no-progress` alarm.
    """
    toolbox = toolbox_for(backend)

    def declare(phase: str, seconds: float) -> None:
        if mark:
            mark(phase, seconds)

    try:
        regions = mesh_regions(backend, case_dir)
        composite: dict[str, Any] = {"regions": {}, "meshed": bool(regions),
                                     "cad": []}
        for region in regions or [""]:
            declare("check", TIMEOUT_S)
            outcome = _ask(backend, _region_command(region, toolbox), case_dir)
            composite["regions"][region] = _region_entry(outcome.output or "")
        for name in ("cad_audit", "domain_probe"):
            declare("gates", TIMEOUT_S)
            composite["cad"].append(
                _cad_entry(backend, case_dir, name, f"{toolbox}/{name}.py"))
        if script:
            _leave_script(backend, case_dir, script)
    except _Unreachable as gone:
        return Check(
            ok=False, unreachable=True, error=str(gone), regions=[],
            missing=[f"the workspace did not answer, so the mesh could not be "
                     f"checked -- it may well be there ({gone})"],
            findings=[Finding(
                "workspace", "skipped",
                f"the workspace did not answer twice in a row ({gone})",
                "nothing is known about the mesh either way; a container that recycles "
                "mid-run comes back and the Volume under it keeps the files",
                f"look for yourself: {look_command(case_dir, toolbox)}")])
    return read(composite, case_rel, case_dir, request, toolbox)


def _region_entry(output: str) -> dict[str, Any]:
    """One region's command output, split into the payload and the cell zones."""
    head, _, tail = output.rpartition("@@JSON@@")
    if not _:
        head, tail = output, ""
    zones_text = head.rpartition("@@CELLZONES@@")[2]
    entry: dict[str, Any] = {"look": _json_in(tail), "output": (tail or output).strip()[-800:]}
    zones = _json_in_list(zones_text)
    if zones is not None:
        entry["cellzones"] = [[str(name), int(count)] for name, count in zones]
    return entry


def _cad_entry(backend: Any, case_dir: str, name: str, script_path: str) -> dict[str, Any]:
    """One toolbox audit's I2 envelope, or a note saying why there is none.

    A toolbox script that will not run is not a mesh failure: it comes back `skipped`
    with what happened in it, on the same rule as everything else here.
    """
    try:
        outcome = backend.exec(_cad_command(script_path), cwd=case_dir, timeout_s=TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - the toolbox, not the geometry
        return {"script": name, "unavailable": str(exc)}
    payload = _json_in(outcome.output or "")
    if payload is None:
        return {"script": name, "unavailable": (outcome.output or "").strip()[-300:]}
    payload.setdefault("script", name)
    return payload


def _leave_script(backend: Any, case_dir: str, script: str) -> None:
    """Write the concatenated cell log into the case, as `build.py`.

    **The brief has always said this happens** -- "your accepted cells are the script you
    leave behind; they are concatenated into `build.py` in the case directory" -- and for
    a long time nothing did it. Two gates stood on the claim instead, and both are gone:
    one failed any case without a rebuild script under a name `casebundle` packs, the
    other re-ran the concatenation from empty and compared the geometry.

    **The desk had no move against either.** The cell log is append-only -- `CellLog`
    offers `propose`, `accept`, `cells`, `bound` and `script`, and nothing that edits or
    drops a cell already in -- so a replay that breaks at cell 3 cannot be fixed by
    writing cell 9. A gate whose failure has no answer is not work handed back; it is a
    run that ends.

    Nothing else in the product is held to it, either. The session agent works through
    `bash`, and no one concatenates its commands and re-runs them to decide whether it
    was really finished. The desk's cells are the same kind of thing.

    So the script is an artifact rather than a claim. `build.py` is in
    `casebundle.DEFINITION_NAMES`, so the bundle carries it; `build_files` reads the case
    root, so the check reports it; and the sentence in the brief is true. Whether it
    re-runs is the reader's business, and `CellLog.accept` -- which refuses a cell
    referencing a name no accepted cell binds -- is what keeps it worth reading.

    Failing to write it is not a verdict about the mesh.
    """
    try:
        backend.put_file(f"{case_dir.rstrip('/')}/{BUILD_SCRIPT}", script.encode("utf-8"))
    except Exception:  # noqa: BLE001 - not a verdict about the mesh
        pass


def read(payload: dict[str, Any], case_rel: str = "", case_dir: str = "",
         request: str = "", toolbox: str = TOOLBOX) -> Check:
    """The answers `verify` gathered, turned into a verdict.

    Takes either the composite `verify` builds -- `{"regions": {...}, "cad": [...]}` --
    or a single `mesh_look.py --json` payload, which is the one-region case written the
    short way. Split from `verify` so the rules can be
    tested without a workspace: this function is the whole of what "done" means.
    """
    composite = _as_composite(payload)
    regions: dict[str, Any] = composite["regions"]
    check = Check(raw=payload)
    check.regions = [] if not composite.get("meshed", True) else list(regions)
    multi = len(regions) > 1

    findings: list[Finding] = []
    bounds_lo = [math.inf] * 3
    bounds_hi = [-math.inf] * 3
    verdicts: list[str] = []
    for region, entry in regions.items():
        look = entry.get("look")
        where = f"region {region}: " if multi and region else ""
        if not isinstance(look, dict):
            tail = str(entry.get("output") or "")
            findings.append(Finding(
                "look", "fail",
                f"{where}the check wrote no readable answer" + (f" -- {tail}" if tail else ""),
                "mesh_look.py did not print its JSON, so nothing about this mesh was read",
                f"run `python3 {toolbox}/mesh_look.py . --out {RENDER_REL} "
                f"--json {JSON_REL}"
                + (f" --region {region}" if region else "") + "` yourself and fix what it says"))
            continue
        findings.extend(_region_findings(look, entry, region, where, case_rel))
        check.cells += int(look.get("cells") or 0)
        check.points += int(look.get("points") or 0)
        check.faces += int(look.get("faces") or 0)
        check.two_d = check.two_d or bool(look.get("two_d"))
        for patch in look.get("patches") or []:
            entry_patch = dict(patch)
            if multi and region:
                entry_patch["region"] = region
            check.patches.append(entry_patch)
        for key, value in (look.get("metrics") or {}).items():
            try:
                check.metrics[f"{region}.{key}" if multi and region else str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        verdict = str(look.get("checkmesh") or "")
        if verdict:
            verdicts.append(f"{region}: {verdict}" if multi and region else verdict)
        box = [float(v) for v in (look.get("bounds") or [])]
        if len(box) == 6:
            for i in range(3):
                bounds_lo[i] = min(bounds_lo[i], box[i])
                bounds_hi[i] = max(bounds_hi[i], box[i + 3])
        if not check.build:
            check.build = list(look.get("build") or [])
        if not check.render:
            check.render, check.render_abs = _render_paths(look, case_rel, case_dir)

    if all(v < math.inf for v in bounds_lo):
        check.bounds = bounds_lo + bounds_hi
    check.checkmesh = "; ".join(verdicts)

    if not regions or not composite.get("meshed", True):
        if not any(f.check == "mesh" and f.status == "fail" for f in findings):
            findings.insert(0, _nothing_meshed(case_rel))
    findings.extend(_case_findings(check, request, toolbox))
    findings.extend(_cad_findings(composite.get("cad") or []))

    check.findings = findings
    # **Advisory findings are reported and do not bind.** They used to: `ok` was
    # `worst_status(findings) != "fail"` over this whole list, `cad_audit`'s answers are
    # in it, and so a single open edge on a deliberate zero-thickness baffle failed the
    # finish with nothing the desk could do about it -- there was no waiver on this desk,
    # and declaring again unchanged got the same answer.
    #
    # The core desk has always had these advisory, which is why two sweeps' worth of
    # evidence about them exists and none of it came from here: three of the six were
    # measured defective, and `coverage` warned on 19 of 20 correct partitions. They
    # reach the desk through `gate.evaluate` instead, where a warning costs a waiver
    # and a line in the record rather than a mesh.
    binding = [f for f in findings if not gate.is_advisory(f.check)]
    check.missing = [f.measured for f in binding if f.status == "fail"]
    check.ok = worst_status(binding) != "fail"
    return check


def _as_composite(payload: dict[str, Any]) -> dict[str, Any]:
    """A single `mesh_look` payload read as the one-region composite it is."""
    regions = payload.get("regions")
    if isinstance(regions, dict):
        composite = dict(payload)
        composite["regions"] = {str(k): dict(v) for k, v in regions.items()}
        return composite
    return {"regions": {"": {"look": payload}},
            "meshed": bool(payload.get("polymesh")),
            "cad": payload.get("cad") or []}


def _nothing_meshed(case_rel: str) -> Finding:
    return Finding(
        "mesh", "fail",
        f"there is no constant/polyMesh in {case_rel or 'the case'} -- "
        "nothing has been meshed yet",
        "neither a single-region mesh nor any constant/<region>/polyMesh is on disk",
        "mesh it and leave the script that did: blockMesh, snappyHexMesh or "
        "cartesianMesh, then checkMesh")


def _region_findings(look: dict[str, Any], entry: dict[str, Any], region: str,
                     where: str, case_rel: str) -> list[Finding]:
    """Everything that is true of one region's mesh on its own."""
    findings: list[Finding] = []
    cells = int(look.get("cells") or 0)
    if not look.get("polymesh"):
        findings.append(_nothing_meshed(case_rel) if not region else Finding(
            "mesh", "fail",
            f"{where}there is no constant/{region}/polyMesh, though the region is named",
            "a region without a mesh is a region nothing can be solved on",
            f"mesh {region} too, or drop it from the region set"))
    elif not cells:
        findings.append(Finding(
            "mesh", "fail", f"{where}constant/polyMesh is there but has no cells in it",
            "an empty mesh directory is what a mesher that stopped early leaves behind",
            "read the mesher's log for where it stopped, and re-run it"))
    else:
        findings.append(Finding(
            "mesh", "ok",
            f"{where}{cells:,} cells, {int(look.get('faces') or 0):,} faces, "
            f"{int(look.get('points') or 0):,} points",
            "2D (one cell thick)" if look.get("two_d") else "3D"))
    findings.append(_checkmesh_finding(look, where))
    findings.extend(_patch_findings(look, where))
    findings.append(_cellzone_finding(entry, where))
    if look.get("error"):
        findings.append(Finding(
            "look", "fail", f"{where}{look['error']}",
            "mesh_look.py reached the mesh and could not finish looking at it",
            "run the look command by hand and fix what it says"))
    return findings


def _checkmesh_finding(look: dict[str, Any], where: str) -> Finding:
    """checkMesh's verdict, reported. Never re-decided.

    The thresholds put the numbers in context so a reader knows what 78 degrees means.
    They do not overrule the verdict: `checkMesh` already judged this mesh, and a
    second judge on the same number only ever produces a disagreement.
    """
    verdict = str(look.get("checkmesh") or "")
    metrics = look.get("metrics") or {}
    numbers = _metric_phrases(metrics)
    if not look.get("checkmesh_ok"):
        return Finding(
            "checkmesh", "fail",
            f"{where}checkMesh does not pass: {verdict or 'no verdict was read'} -- "
            "read log.checkMesh for the failing checks",
            "the mesh OpenFOAM was given is one OpenFOAM refuses",
            "fix the checks checkMesh names -- it lists them by name with cell counts "
            "in log.checkMesh -- and run it again")
    measured = f"{where}{verdict or 'Mesh OK.'}"
    if numbers:
        measured += " -- " + ", ".join(numbers)
    return Finding(
        "checkmesh", "ok", measured,
        "checkMesh accepts this mesh; the thresholds beside the numbers are there to "
        "read them by and do not overrule it, because it has already judged them")


def _metric_phrases(metrics: dict[str, Any]) -> list[str]:
    """The three numbers a solver minds, each said next to the threshold it is read by."""
    out: list[str] = []
    for key, warn, fail in (("max_non_orthogonality", NON_ORTHO_WARN, NON_ORTHO_FAIL),
                            ("max_skewness", SKEWNESS_WARN, SKEWNESS_FAIL),
                            ("max_aspect_ratio", ASPECT_WARN, None)):
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            continue
        name = key.replace("max_", "max ").replace("_", "-")
        if fail is not None and value > fail:
            out.append(f"{name} {value:g} (past {fail:g}, which checkMesh accepted anyway)")
        elif value > warn:
            out.append(f"{name} {value:g} (over the usual {warn:g}, which checkMesh "
                       "accepted anyway)")
        else:
            out.append(f"{name} {value:g} (under {warn:g})")
    return out


PATCH_COUNT_MEANING = (
    "one boundary patch is consistent with a closed domain driven from inside it -- "
    "volumetric sources on cell zones, or a body force -- where every boundary carries "
    "the same condition. It is not consistent with buoyancy driven by wall temperature, "
    "which needs a hot patch and a cold patch, nor with through-flow, which needs an "
    "inlet and an outlet")
PATCH_COUNT_REPAIR = (
    "if the case needs two, split the patch with topoSet + createPatch, or name the "
    "surfaces separately where they are built")


def _patch_findings(look: dict[str, Any], where: str) -> list[Finding]:
    """Patches carrying names somebody chose -- named, not counted.

    The count was always a proxy: what matters is that a person said which surface was
    which, not how many surfaces there are. A sealed cavity has one patch and is right.
    """
    patches = list(look.get("patches") or [])
    named = [p for p in patches if int(p.get("nFaces") or 0) > 0]
    if not named:
        return [Finding(
            "patches", "fail",
            f"{where}the mesh has no boundary patch with any faces on it",
            "a mesh with no boundary is a mesh nothing can be solved on",
            "name the surfaces where they are built, and check the mesher wrote a "
            "boundary file with faces in it")]
    unnamed = [str(p.get("name", "?")) for p in named if _UNNAMED.match(str(p.get("name", "")))]
    out: list[Finding] = []
    if unnamed:
        out.append(Finding(
            "patches", "fail",
            f"{where}these patches carry the mesher's default names rather than names "
            f"somebody chose: {', '.join(unnamed)}",
            "patch0/defaultFaces out of gmshToFoam is what a mesh looks like when "
            "nobody said which end was the inlet",
            "name them where you build the surfaces, or retype them in Allmesh with "
            "createPatch"))
        return out
    out.append(Finding(
        "patches", "ok",
        f"{where}{len(named)} boundary patch(es) with faces, all named: " +
        ", ".join(str(p.get("name", "?")) for p in named),
        "the names are somebody's choice rather than a mesher's default"))
    if len(named) < 2:
        out.append(Finding(
            "patch_count", "warn",
            f"{where}one boundary patch with faces on it: {named[0].get('name', '?')} "
            f"({int(named[0].get('nFaces') or 0):,} faces)",
            PATCH_COUNT_MEANING, PATCH_COUNT_REPAIR))
    return out


def _cellzone_finding(entry: dict[str, Any], where: str) -> Finding:
    """The cell zones, as a measurement.

    Where a mesh's zones are the mechanism -- a heat source and a sink driving
    convection -- a verdict that cannot see them is describing half the mesh. Reported,
    never asserted on: no number here decides anything.
    """
    zones = entry.get("cellzones")
    if zones is None:
        return Finding(
            "cellzones", "skipped", f"{where}constant/polyMesh/cellZones was not read",
            "nothing is known about this mesh's cell zones either way")
    if not zones:
        return Finding(
            "cellzones", "ok", f"{where}no cellZones in this mesh",
            "nothing is driven from inside the domain: every source this case has is "
            "on a boundary")
    listed = ", ".join(f"{name} {count:,} cells" for name, count in zones)
    return Finding(
        "cellzones", "ok", f"{where}{len(zones)} cellZone(s): {listed}",
        "a zone is where a volumetric source, a porous region or a solid region acts")


def _case_findings(check: Check, request: str, toolbox: str = TOOLBOX) -> list[Finding]:
    """What is true of the case as a whole rather than of one region's mesh."""
    findings: list[Finding] = []
    if check.render:
        findings.append(Finding(
            "render", "ok", f"a picture of the mesh is at {check.render}",
            "somebody can look at this mesh without opening ParaView"))
    else:
        findings.append(Finding(
            "render", "fail", "no picture of the mesh was drawn",
            "a mesh nobody has seen is a mesh nobody has checked the shape of",
            f"run `python3 {toolbox}/mesh_look.py . --out {RENDER_REL} "
            f"--json {JSON_REL}` and keep the PNG"))
    off = _scale_measured(request, check.bounds)
    if off:
        findings.append(Finding("scale", "fail", off,
                                "OpenFOAM has no units: it reads the mesh in metres, so a "
                                "geometry authored in millimetres solves a body a thousand "
                                "times too big and nothing else complains",
                                SCALE_REPAIR))
    elif largest_length(request) and len(check.bounds) == 6:
        built = max(check.bounds[3] - check.bounds[0], check.bounds[4] - check.bounds[1],
                    check.bounds[5] - check.bounds[2])
        findings.append(Finding(
            "scale", "ok",
            f"the mesh is {built:.4g} m across and the request's largest dimension is "
            f"{largest_length(request):.4g} m",
            f"within the factor of {SCALE_SLACK:g} a flow box round a small body needs"))
    return findings


# Two checks stood here and neither does now: `_build_finding`, which required a rebuild
# script under a name `casebundle` packs, and `_replay_findings`, which re-ran the
# concatenated cell log from empty and compared the geometry it produced.
#
# **Both failed the desk for things it could not answer.** The brief tells the desk its
# cells "are concatenated into `build.py` in the case directory" and nothing wrote that
# file, so the first gate failed runs for the harness's omission. The second asked the
# cell log to be a re-runnable script, and the log is append-only -- `CellLog` has no way
# to edit or drop a cell that is already in -- so a replay that broke at cell 3 had no
# repair available at cell 9. A gate whose failure has no answer ends the run instead of
# handing work back.
#
# The first also passed for the wrong reason more often than it failed: `build_files`
# counts `system/blockMeshDict`, present whenever blockMesh ran and not a thing that
# rebuilds the geometry.
#
# Nothing else in the product is held to this. The session agent works through `bash` and
# nobody concatenates its commands to re-run them as proof it finished. What remains is
# `_leave_script`, which writes the file the brief promised, and `CellLog.accept`, which
# refuses a cell depending on a name no accepted cell binds -- the preventive half, at the
# moment the desk can still act on it.


def _cad_findings(envelopes: list[Any]) -> list[Finding]:
    """C2's and C3's findings, consumed.

    Carried through with their own `measured` and `repair`, untouched. Nothing here
    recomputes a triangle: the audit ran where the triangles are, and a second opinion
    computed from a different code path is how two answers about one surface start.
    """
    out: list[Finding] = []
    for envelope in envelopes:
        if not isinstance(envelope, dict):
            continue
        name = str(envelope.get("script") or "cad")
        if envelope.get("unavailable") is not None:
            out.append(Finding(
                f"{name}", "skipped",
                f"{name} did not report: {envelope['unavailable'] or 'no output'}",
                "either no patch set has been exported yet, or the script did not run"))
            continue
        for raw in envelope.get("findings") or []:
            if not isinstance(raw, dict):
                continue
            finding = Finding.from_dict(raw)
            out.append(finding._replace(check=f"{name}.{finding.check}"))
    return out


def _render_paths(look: dict[str, Any], case_rel: str, case_dir: str) -> tuple[str, str]:
    render = str(look.get("render") or "")
    if render.startswith("/"):
        return render, render
    if not render:
        return "", ""
    # `mesh_look.py` reports the picture relative to the case, which is what a person
    # reads; a fetch needs the workspace path, and the two are different strings often
    # enough that keeping only one of them cost a missing picture.
    return (f"{case_rel}/{render}" if case_rel else render,
            f"{case_dir}/{render}" if case_dir else render)


# -- the size the request asked for --------------------------------------------


SCALE_REPAIR = ("scale it: gmsh -- multiply the coordinates or set Mesh.ScalingFactor; "
                "blockMesh -- `scale 0.001;`; an existing mesh -- "
                "`transformPoints -scale '(0.001 0.001 0.001)'`")


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


def _scale_measured(request: str, bounds: list[float]) -> str:
    """The evidence half of the scale finding: two numbers and their ratio."""
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
            f"{asked:.4g} m -- {ratio:.0f}x out")


def scale_mismatch(request: str, bounds: list[float]) -> str:
    """Whether the mesh is the size the request asked for, to within a factor of 100.

    The failure this exists for: a geometry authored in millimetres and meshed
    without a scale. OpenFOAM has no units -- it reads the numbers as metres -- so a
    74 mm duct becomes a 74 m duct, every velocity is the wrong Reynolds number, and
    nothing in checkMesh or in the picture says a word about it. The only place the
    intended size is written down is the request, so that is what it is measured
    against.
    """
    measured = _scale_measured(request, bounds)
    return f"{measured}. {SCALE_REPAIR}" if measured else ""


# -- reading what came back ----------------------------------------------------


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


def _json_in_list(text: str) -> list | None:
    """The JSON array in a command's output, same rule as `_json_in`."""
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, list) else None


def look_command(case_dir: str, toolbox: str = TOOLBOX) -> str:
    """The command a person or an agent runs by hand to see the same thing."""
    return (f"python3 {toolbox}/mesh_look.py {shlex.quote(case_dir)} "
            f"--out {RENDER_REL} --json {JSON_REL}")
