"""Claims: what the request states, written before any geometry, and the compliance
table the built shape is judged against.

The schema is DESIGN.md 4.1 (`id`, `says`, `of`, top-level `unit`/`kind`/`flow`, an
explicit `kind` per claim, D12); an unknown predicate or measure is `not_measurable`,
never an error; a claim naming an absent feature is a `fail` row listing the script's
features. `can_commit` is pure and is where D10 (warnings do not block in Phase 1), D11
(disagrees must equal the failing set) and D33 (a table needs a checkable row) live.
Skeleton (U0): the types with their round trips and the counting properties; the parser,
the predicates and `comply` raise NotImplementedError until U3 lands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from .lint import Finding  # noqa: F401
from .measure import Measurements  # noqa: F401

if TYPE_CHECKING:
    from .compile import Plan
    from .desk import Reply

_U3 = ("not built in the U0 skeleton: U3 (claims + report + fitness) implements it "
       "against DESIGN.md section 3.7")

KINDS = ("measure", "range", "count", "predicate", "patch", "report", "not_measurable")

MEASURES: dict[str, str] = {}
"""Measure key -> one sentence (the keys of section 3.5's table). Filled by U3."""

LENGTH_MEASURES: frozenset[str] = frozenset({
    "width", "height", "length", "extent_x", "extent_y", "outer_radius", "inner_radius", "radius",
    "diameter", "pitch", "gap", "gap_measured", "spacing", "pass_length", "bend_radius", "pass_pitch",
    "wall_between", "leave_length", "lands_upstream_by", "centre_x", "centre_y", "footprint",
    "circumference",
})

WARNINGS_BLOCK_COMMIT: bool = False
"""D10: revisit after the benchmark measures how often a warning fires on a correct shape."""


class ClaimsError(ValueError):
    """A malformed claims file: the message names the claim id and the field."""


@dataclass
class Verdict:
    ok: bool | None
    """None = not measurable."""
    measured: str
    """"heading 260 deg, 100 deg from the flow (+x): component -0.17"."""
    where: tuple[float, float] | None = None
    numbers: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Verdict":
        where = d.get("where")
        return Verdict(ok=d.get("ok"), measured=d.get("measured", ""),
                       where=None if where is None else (float(where[0]), float(where[1])),
                       numbers=dict(d.get("numbers", {})))


Predicate = Callable[[Measurements, str, dict], Verdict]
"""(m: Measurements, of: str, args: dict) -> Verdict."""

PREDICATES: dict[str, Predicate] = {}
"""name -> callable. The Phase 1 predicates of DESIGN.md 3.7's table; filled by U3."""


@dataclass
class Claim:
    id: str
    says: str
    kind: str
    measure: str = ""
    of: str = ""
    value: float | None = None
    tol: float | None = None
    tol_rel: float | None = None
    min: float | None = None
    max: float | None = None
    predicate: str = ""
    args: dict = field(default_factory=dict)
    patch: str = ""
    at: str | list | None = None
    side: str | None = None
    patch_kind: str = ""
    not_measurable: str = ""
    contradictory: bool = False
    """The benchmark's own claims files use it; the desk's never set it."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(**{k: v for k, v in d.items() if k in Claim.__dataclass_fields__})


@dataclass
class ClaimSet:
    unit: str
    kind: str
    flow: str | None
    claims: list[Claim]
    largest_length: float | None = None
    """Derived: the largest value/max among LENGTH_MEASURES claims."""

    def as_dict(self) -> dict:
        return {"unit": self.unit, "kind": self.kind, "flow": self.flow,
                "claims": [c.as_dict() for c in self.claims], "largest_length": self.largest_length}

    @staticmethod
    def from_dict(d: dict) -> "ClaimSet":
        return ClaimSet(unit=d["unit"], kind=d["kind"], flow=d.get("flow"),
                        claims=[Claim.from_dict(c) for c in d.get("claims", [])],
                        largest_length=d.get("largest_length"))


def parse(payload: dict) -> tuple[ClaimSet, list[str]]:
    """Validate the schema of section 4.1. Returns the set and one line per claim for the
    claims-lap answer: 'c4 outer_radius of loops[*]: measurable' / 'c10: not measurable
    here (the finish reports it)' / 'c6: no predicate named sweeps_round; known: ...'. An
    unknown predicate or measure is NOT an error: the claim becomes not_measurable with the
    reason. Malformed JSON, a duplicate id, a missing `says`, a `kind` outside KINDS, or a
    `kind` whose required fields are absent raise ClaimsError naming the claim id and field.
    `kind: body_in_flow` with no BodyInBox in the script (known at the first script lap) is
    a warning line in the next print-back, not a failure: 'c0: kind body_in_flow but the
    script builds no BodyInBox; judged as a passage'."""
    raise NotImplementedError(_U3)


@dataclass
class ComplianceRow:
    id: str
    says: str
    kind: str
    expected: str
    """"3 +/- 0.03", "4", "<= 30 deg", "inlet at (0, 0) +/- 0.5"."""
    measured: str
    """"3.000 (main.width)" | "no feature named 'loops'; features: main, loop, row, fluid"."""
    verdict: Literal["pass", "fail", "not_measurable", "disagreed"]
    where: tuple[float, float] | None = None
    numbers: dict = field(default_factory=dict)
    detail: str = ""
    """The explanation under a FAIL (T02's c8 sentence)."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ComplianceRow":
        where = d.get("where")
        return ComplianceRow(id=d["id"], says=d["says"], kind=d["kind"], expected=d.get("expected", ""),
                             measured=d.get("measured", ""), verdict=d["verdict"],
                             where=None if where is None else (float(where[0]), float(where[1])),
                             numbers=dict(d.get("numbers", {})), detail=d.get("detail", ""))


CHECKABLE_KINDS = frozenset({"measure", "range", "count", "predicate", "patch"})


@dataclass
class ComplianceTable:
    rows: list[ComplianceRow]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "fail")

    @property
    def unmeasurable(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "not_measurable")

    @property
    def disagreed(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "disagreed")

    @property
    def checkable(self) -> int:
        """Rows of kind measure / range / count / predicate / patch (report and
        not_measurable do not count)."""
        return sum(1 for r in self.rows if r.kind in CHECKABLE_KINDS)

    def ok(self) -> bool:
        """failed == 0 AND checkable >= 1 (D33: an empty or report-only table never passes)."""
        return self.failed == 0 and self.checkable >= 1

    def failing(self) -> list[ComplianceRow]:
        return [r for r in self.rows if r.verdict == "fail"]

    def failing_ids(self) -> list[str]:
        return [r.id for r in self.failing()]

    def lines(self) -> list[str]:
        """The CLAIMS block of section 6."""
        raise NotImplementedError(_U3)

    def as_dict(self) -> dict:
        return {"rows": [r.as_dict() for r in self.rows]}

    @staticmethod
    def from_dict(d: dict) -> "ComplianceTable":
        return ComplianceTable(rows=[ComplianceRow.from_dict(r) for r in d.get("rows", [])])


def comply(claims: ClaimSet, m: Measurements, findings: list[Finding], plan: "Plan") -> ComplianceTable:
    """One row per claim. `of` joins on the script's `name=`: `name[*]` = every instance of
    a Row (all must pass; the measured column says `3.0 (x4)`), `name[k]` = one instance;
    `fluid`, a patch name, or `inlet`/`outlet`. A claim naming an absent feature is a
    `fail` row whose `measured` lists the script's feature names (D12)."""
    raise NotImplementedError(_U3)


def can_commit(findings: list[Finding], table: ComplianceTable | None, reply: "Reply") -> tuple[bool, str]:
    """Pure. (ok, reason). No table -> refused. A table with `checkable == 0` -> refused:
    'COMMIT refused: no checkable claim was recorded; the claims lap failed twice: <parse
    error>' (D33). Lint errors -> refused naming codes and coordinates. reply.disagrees must
    equal table.failing_ids() as sets (D11). Warnings: refused when WARNINGS_BLOCK_COMMIT and
    any warning is not in reply.accepts (matched by code and `where` within 0.1 mm);
    otherwise never blocking."""
    raise NotImplementedError(_U3)
