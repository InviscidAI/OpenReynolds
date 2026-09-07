"""The case on disk and the finish on the instance.

`write_case` writes the record, the script, the claims and the compliance table at the
case root (D24) and then runs `mesh2d.main` in a child so body.msh, the STEP, 0/, system/
and Allmesh are the proven writers' output; `finish_on_instance` is today's `_finish`
exec plus the boundary read-back that closes the T05 class (the patch lost in the
extrusion) in the same tool call.

Why the mesh sizes are recomputed here rather than parsed from the writer's output:
`mesh2d.main` decides the cell from the built extent and its options, and the same
function with the same inputs gives the same number in this process. The fitness table
then carries the writer's cell, not a guess, and the seam stays one function call wide.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import _toolbox
from .compile import Plan  # noqa: F401
from .fitness import FitnessTable  # noqa: F401

if TYPE_CHECKING:
    from .claims import ComplianceTable
    from .strategy import Sizes

TOOLBOX_DEST = "/work/.toolbox"
"""Where the toolbox is refreshed to on the instance (cli.py), for the finish step."""

IMPLICIT_PATCHES = ("walls", "frontAndBack")
"""The two patches every 2D case carries whether or not the record names them: mesh2d's
`generate` puts every unnamed curve on `walls` and the two z-flat faces on `frontAndBack`."""

_EXTENT_LINE = re.compile(r"extent\s+([0-9.eE+-]+)\s*x\s*([0-9.eE+-]+)")
"""mesh2d's summary line, in metres, read only when the record carries no measurements."""


@dataclass
class CasePaths:
    case_rel: str = ""
    """/work/<study>/<case>."""
    record: str = "geometry.json"
    """The section 4.2 record, at the case root; constant/geometry/body.json is mesh2d's copy of it."""
    script: str = "geometry.py"
    """The script alone."""
    claims: str = "claims.json"
    compliance: str = "compliance.json"
    fitness: str = "fitness.json"
    """Written locally after the finish."""
    outline_png: str = "outline.png"
    mesh_png: str = "renders/mesh_z.png"
    checkmesh_log: str = "log.checkMesh"

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CasePaths":
        return CasePaths(**{k: v for k, v in d.items() if k in CasePaths.__dataclass_fields__})


def case_args(local: Path, study: str, scale: float, paths: CasePaths) -> list[str]:
    """The commit: today's `mesh2d.py <case> --spec geometry.json --study ... --preview
    outline.png --force [--scale s]`, in a child interpreter so a gmsh crash cannot take
    the desk down."""
    args = [sys.executable, str(_toolbox.TOOLBOX / "mesh2d.py"), str(local), "--spec", str(local / paths.record),
            "--study", study, "--preview", str(local / paths.outline_png), "--force"]
    if scale and scale != 1.0:
        args += ["--scale", str(scale)]
    return args


def extent_m(record: dict, tool_output: str, scale: float) -> tuple[float, float]:
    """The built extent in metres: the record's measurement scaled, or, for a record
    with no measurements (a hand-written spec), the extent the writer printed."""
    m = record.get("measurements") if isinstance(record, dict) else None
    extent = (m or {}).get("extent") if isinstance(m, dict) else None
    if extent and len(extent) == 2:
        return float(extent[0]) * scale, float(extent[1]) * scale
    found = _EXTENT_LINE.search(tool_output)
    if found:
        return float(found.group(1)), float(found.group(2))
    raise RuntimeError("the record carries no measurements and the writer printed no extent line, so "
                       "the cell size it meshed with cannot be reproduced; a record from `cli build` "
                       "carries `measurements.extent`")


def write_case(local: Path, record: dict, script: str, claims: dict | None, table: "ComplianceTable | None",
               study: str, scale: float) -> tuple[CasePaths, "Sizes"]:
    """Writes geometry.json (the record), geometry.py, claims.json, compliance.json into
    `local`, then runs mesh2d.main([str(local), "--spec", geometry.json, "--scale", scale,
    "--study", study, "--preview", outline.png, "--force"]) in a child interpreter (today's
    case_args, unchanged) so body.msh, constant/geometry/{body.step, body.json}, 0/, system/,
    constant/, Allmesh, Allrun are the proven writers' output. Raises RuntimeError with the
    tool's last 1500 characters on a non-zero exit. Returns the paths and the `Sizes` the
    mesh was built with, computed in-process by `_toolbox.load("mesh2d").mesh_sizes2d(extent_m,
    opts)` from the record's measurements -- the same function `mesh2d.main` calls with the
    same inputs, so `fitness.from_digest` gets `cell_m` from the writer, not from a guess."""
    from .strategy import Sizes  # a return type only; the module graph keeps strategy beside case, not under it

    paths = CasePaths()
    # mesh2d copies the spec it reads into constant/geometry/body.json as it is, so the
    # claims and the table ride on the record and the instance's copy carries them too
    record = dict(record)
    if claims is not None and "claims" not in record:
        record["claims"] = claims
    if table is not None and "compliance" not in record:
        record["compliance"] = table.as_dict()
    local.mkdir(parents=True, exist_ok=True)
    (local / paths.record).write_text(json.dumps(record, indent=1), encoding="utf-8")
    (local / paths.script).write_text(script or "", encoding="utf-8")
    if claims is not None:
        (local / paths.claims).write_text(json.dumps(claims, indent=1), encoding="utf-8")
    if table is not None:
        (local / paths.compliance).write_text(json.dumps(table.as_dict(), indent=1), encoding="utf-8")
    proc = subprocess.run(case_args(local, study, scale, paths), capture_output=True, text=True, timeout=300)
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        raise RuntimeError(output[-1500:] or f"exit {proc.returncode}")
    mesh2d = _toolbox.load("mesh2d")
    cell, wall_cell, thickness = mesh2d.mesh_sizes2d(extent_m(record, output, scale), {})
    return paths, Sizes(cell=float(cell), wall_cell=float(wall_cell), thickness=float(thickness))


@dataclass
class FinishReport:
    rc: int
    output: str
    digest: str
    digest_data: dict | None
    mesh_png: bytes | None
    boundary_patches: list[str]
    missing: list[str]
    extra: list[str]

    def as_dict(self) -> dict:
        return {"rc": self.rc, "output": self.output, "digest": self.digest, "digest_data": self.digest_data,
                "mesh_png": None if self.mesh_png is None else self.mesh_png.hex(),
                "boundary_patches": self.boundary_patches, "missing": self.missing, "extra": self.extra}

    @staticmethod
    def from_dict(d: dict) -> "FinishReport":
        png = d.get("mesh_png")
        return FinishReport(rc=d["rc"], output=d.get("output", ""), digest=d.get("digest", ""),
                            digest_data=d.get("digest_data"), mesh_png=None if png is None else bytes.fromhex(png),
                            boundary_patches=list(d.get("boundary_patches", [])), missing=list(d.get("missing", [])),
                            extra=list(d.get("extra", [])))


FINISH_CMD = ("sh Allmesh > log.Allmesh 2>&1; rc=$?; tail -12 log.Allmesh; "
              # the case by its absolute path, not `.`: render.py names its ParaView
              # marker after the directory, and `.` gave it the name `.foam`
              f"python3 {TOOLBOX_DEST}/render.py \"$PWD\" --scene mesh --out renders > log.render 2>&1 "
              "|| tail -5 log.render; exit $rc")


def finish_on_instance(backend, remote: str, expected_patches: list[str], timeout_s: int) -> FinishReport:
    """Today's _finish exec, verbatim: `sh Allmesh > log.Allmesh 2>&1; rc=$?; tail -12
    log.Allmesh; python3 /work/.toolbox/render.py "$PWD" --scene mesh --out renders >
    log.render 2>&1 || tail -5 log.render; exit $rc`; then get_file(log.checkMesh) ->
    mesh_digest.parse/report; get_file(renders/mesh_z.png); get_file(constant/polyMesh/boundary)
    -> patches_agree."""
    exec_ = getattr(backend, "exec", None)
    get_file = getattr(backend, "get_file", None)
    if exec_ is None or get_file is None:
        return FinishReport(rc=1, output="the backend cannot run commands or read files, so Allmesh was not run",
                            digest="", digest_data=None, mesh_png=None, boundary_patches=[], missing=[], extra=[])
    try:
        run = exec_(FINISH_CMD, cwd=remote, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - the words say what happened
        return FinishReport(rc=1, output=f"the mesh was not built on the instance: {exc}", digest="",
                            digest_data=None, mesh_png=None, boundary_patches=[], missing=[], extra=[])
    output = (getattr(run, "output", "") or "").strip()
    rc = getattr(run, "exit_code", 1)
    digest, data = "", None
    try:
        log = get_file(f"{remote}/log.checkMesh", limit=400_000).decode("utf-8", "replace")
        mesh_digest = _toolbox.load("mesh_digest")
        data = mesh_digest.parse(log)
        digest = mesh_digest.report(data).strip()
    except Exception:  # noqa: BLE001 - no log means the mesh step did not get that far
        digest, data = "", None
    try:
        mesh_png = get_file(f"{remote}/renders/mesh_z.png", limit=20_000_000)
    except Exception:  # noqa: BLE001
        mesh_png = None
    found: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    try:
        boundary = get_file(f"{remote}/constant/polyMesh/boundary", limit=400_000).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - no boundary file: gmshToFoam did not run, and rc says so
        boundary = None
    if boundary is not None:
        found = boundary_patch_names(boundary)
        missing, extra = patches_agree(boundary, expected_patches)
    return FinishReport(rc=rc, output=output, digest=digest, digest_data=data, mesh_png=mesh_png,
                        boundary_patches=found, missing=missing, extra=extra)


_NAME_LINE = re.compile(r"^\s*(\w+)\s*$")


def boundary_patch_names(boundary_text: str) -> list[str]:
    """The patch names in constant/polyMesh/boundary, in file order: a line that is one
    word followed by a line that is `{`. The `FoamFile` header has the same shape and is
    not a patch."""
    lines = boundary_text.splitlines()
    names: list[str] = []
    for i, line in enumerate(lines[:-1]):
        match = _NAME_LINE.match(line)
        if match and lines[i + 1].strip() == "{" and match.group(1) != "FoamFile":
            names.append(match.group(1))
    return names


def patches_agree(boundary_text: str, expected: list[str]) -> tuple[list[str], list[str]]:
    """(missing, extra): patch names in constant/polyMesh/boundary (regex `^\\s*(\\w+)\\s*$`
    followed by a line that is `{`) against the record's patch names plus `walls` and
    `frontAndBack`. The T05 class (the cylinder lost in the extrusion) closes here, in the
    same tool call."""
    found = boundary_patch_names(boundary_text)
    wanted: list[str] = []
    for name in [*expected, *IMPLICIT_PATCHES]:
        if name not in wanted:
            wanted.append(name)
    missing = [name for name in wanted if name not in found]
    extra = [name for name in found if name not in wanted]
    return missing, extra
