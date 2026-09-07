"""The case on disk and the finish on the instance.

`write_case` writes the record, the script, the claims and the compliance table at the
case root (D24) and then runs `mesh2d.main` in a child so body.msh, the STEP, 0/, system/
and Allmesh are the proven writers' output; `finish_on_instance` is today's `_finish`
exec plus the boundary read-back that closes the T05 class (the patch lost in the
extrusion) in the same tool call. Skeleton (U0): the types; the functions raise
NotImplementedError until U5 lands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import _toolbox  # noqa: F401  (mesh2d.main by path writes the case)
from .compile import Plan  # noqa: F401
from .fitness import FitnessTable  # noqa: F401

if TYPE_CHECKING:
    from .claims import ComplianceTable
    from .strategy import Sizes

_U5 = ("not built in the U0 skeleton: U5 (desk v2 + result + case + tools) implements it "
       "against DESIGN.md section 3.13")


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
    raise NotImplementedError(_U5)


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


def finish_on_instance(backend, remote: str, expected_patches: list[str], timeout_s: int) -> FinishReport:
    """Today's _finish exec, verbatim: `sh Allmesh > log.Allmesh 2>&1; rc=$?; tail -12
    log.Allmesh; python3 /work/.toolbox/render.py "$PWD" --scene mesh --out renders >
    log.render 2>&1 || tail -5 log.render; exit $rc`; then get_file(log.checkMesh) ->
    mesh_digest.parse/report; get_file(renders/mesh_z.png); get_file(constant/polyMesh/boundary)
    -> patches_agree."""
    raise NotImplementedError(_U5)


def patches_agree(boundary_text: str, expected: list[str]) -> tuple[list[str], list[str]]:
    """(missing, extra): patch names in constant/polyMesh/boundary (regex `^\\s*(\\w+)\\s*$`
    followed by a line that is `{`) against the record's patch names plus `walls` and
    `frontAndBack`. The T05 class (the cylinder lost in the extrusion) closes here, in the
    same tool call."""
    raise NotImplementedError(_U5)
