"""The judge: predicates over the walk and the measurements, each a Finding with a
stable code from DESIGN.md section 5.

Every threshold is a multiple of the reference width or the span (never an absolute
length), every message names the feature, the numbers and what changes it, and every
instance reference is `<row>[k]`, 0-based (D29). `from_legacy` turns mesh2d's check dicts
into Findings so the toolbox's own text survives only inside `mesh2d.check_lines`.
Skeleton (U0): the Finding and LintConfig types with their round trips; the predicates
raise NotImplementedError until U2 lands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from .measure import Measurements, Walk  # noqa: F401

if TYPE_CHECKING:
    from .claims import ClaimSet
    from .compile import Plan

_U2 = ("not built in the U0 skeleton: U2 (measure + lint) implements it against "
       "DESIGN.md section 3.6")

Level = Literal["error", "warn", "info"]

_LEAD = {"error": "!! ERROR ", "warn": "!!       ", "info": "check    "}


@dataclass
class Finding:
    level: Level
    code: str
    """Section 5's codes: E-OVERLAP, W-SHORT-EDGE, I-GAP, ..."""
    subject: str
    """"Row 'loops'", "Bypass 'loop[2]'", "inlet", "fluid"."""
    what: str
    """One sentence with the number, the threshold and the coordinates."""
    fix: str = ""
    """What changes it (may be empty for info)."""
    where: tuple[float, float] | None = None
    """Sketch units; drawn as a red (error) / orange (warn) cross."""
    numbers: dict[str, float] = field(default_factory=dict)
    draw: list[dict] = field(default_factory=list)
    """{"segment": [(x,y),(x,y)]} | {"circle": (cx, cy, r)} | {"ray": ...}."""

    def text(self) -> str:
        """"!! ERROR  CODE  subject: what\\n          fix" (section 5 format); warnings lead
        with `!!        W-CODE`, info with `check     I-CODE`."""
        head = f"{_LEAD[self.level]} {self.code}  {self.subject}: {self.what}"
        return head if not self.fix else f"{head}\n          {self.fix}"

    def as_legacy(self) -> dict:
        """{"level", "where", "what"} for today's regexes."""
        return {"level": self.level, "where": self.subject, "what": self.what}

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Finding":
        where = d.get("where")
        return Finding(level=d["level"], code=d["code"], subject=d["subject"], what=d["what"],
                       fix=d.get("fix", ""), where=None if where is None else (float(where[0]), float(where[1])),
                       numbers=dict(d.get("numbers", {})), draw=list(d.get("draw", [])))


TEXTS: dict[str, str] = {}
"""code -> the message template of section 5 (every Finding's text is rendered from it;
invariant 1's readable surface scans this dict, and a test pins that every template names
a number placeholder and never "looks"/"seems"/"probably"). Filled by U2."""


@dataclass
class LintConfig:
    overlap_area_rel: float = 1e-6
    """Of the smaller instance's area (OCC fuzzy ~1e-7 of size; 3 orders above noise, 4
    below any real overlap); applied in from_legacy and overlap_between."""
    touch_rel: float = 1e-6
    """Of span (60 nm on a 60 mm valve)."""
    thin_wall_rel: float = 0.1
    """Of w_min: undeclared gap below this warns (the accepted T01 sits at 0.8 on 3)."""
    landing_probe_rel: float = 0.1
    """Of w_min, into the parent."""
    landing_samples: int = 5
    notch_edge_rel: float = 1.0
    """Of w_min: candidate edge length (the square cap is exactly w/2)."""
    notch_solid_rel: float = 1.0 / 3
    """Of w_min: SOLID behind the edge along the outward normal before the ray re-enters the fluid (D9)."""
    short_warn_rel: float = 1.0 / 3
    """Phase 0's rule, kept."""
    short_error_rel: float = 1.0 / 30
    """Below the finest cell Phase 2 makes (w/20, w/30 with layers)."""
    cusp_info_deg: float = 30.0
    """FLUID angle below this: I-CUSP; SOLID angle below this: I-LIP (D31)."""
    cusp_warn_deg: float = 5.0
    """FLUID angle below this: W-CUSP."""
    passage_warn_rel: float = 0.5
    open_end_angle_tol_deg: float = 30.0
    open_end_depth_rel: float = 1.0
    units_warn: tuple[float, float] = (0.5, 3.0)
    units_error: tuple[float, float] = (1 / 8, 8.0)

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LintConfig":
        cfg = LintConfig(**{k: v for k, v in d.items() if k not in ("units_warn", "units_error")})
        if "units_warn" in d:
            cfg.units_warn = (d["units_warn"][0], d["units_warn"][1])
        if "units_error" in d:
            cfg.units_error = (d["units_error"][0], d["units_error"][1])
        return cfg


def from_legacy(checks: list[dict], row: str, cfg: LintConfig) -> list[Finding]:
    """mesh2d.py's check dicts (from build_face's repeat branch: overlap, touch, the gap
    info) -> Findings with the codes E-OVERLAP, E-TOUCH, I-GAP. The numeric keys `pair`,
    `gap`, `overlap`, `instance_area` (section 3.17 edit 2) are read; the subject and the
    text are rewritten to `<row>[k]` 0-based from `pair` (D29: mesh2d's "copies 1 and 2"
    becomes `loops[0]` and `loops[1]`); an overlap below `cfg.overlap_area_rel *
    instance_area` is demoted to I-OVERLAP-NOISE (OCC fuzz on a tangent pair; measured
    empty on tangent disks, so the branch is a guard, not a path the fixtures take).
    Checks from an rc-2 build with a missing `pair` (a spec older than edit 2) keep the
    legacy text verbatim."""
    raise NotImplementedError(_U2)


def judge(gmsh, face: int, plan: "Plan", wk: Walk, m: Measurements, legacy_checks: list[dict],
          claims: "ClaimSet | None", cfg: LintConfig = LintConfig()) -> list[Finding]:
    """Every predicate of DESIGN.md 3.6's table, in this order, errors first in the
    result. Returns after E-DISJOINT (no walk of a multi-face fluid)."""
    raise NotImplementedError(_U2)


def has_errors(findings: list[Finding]) -> bool:
    return any(f.level == "error" for f in findings)


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "error"]


def warnings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "warn"]
