"""The print-back: the one place text is assembled (DESIGN.md section 6).

Fixed section order (SCRIPT, LINT, FEATURES, LEGS, MEASURED, PATCHES, CLAIMS, REFERENCE,
VERDICT), all lengths in the sketch's units with the metre equivalent once on the extent
line; `verdict` is computed from the tables and the desk reads its bool, never the text.
Skeleton (U0): signatures and docstrings; U3 implements them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import _toolbox  # noqa: F401  (mesh2d.leg_lines by path)
from .claims import ComplianceTable  # noqa: F401
from .lint import Finding  # noqa: F401
from .measure import Measurements  # noqa: F401

if TYPE_CHECKING:
    from .compile import Plan
    from .library import ReferenceMatch

_U3 = ("not built in the U0 skeleton: U3 (claims + report + fitness) implements it "
       "against DESIGN.md section 3.8")


def text(printed: str, notes: list[str], seconds: float, findings: list[Finding], plan: "Plan | None",
         m: Measurements | None, legs: dict, table: ComplianceTable | None,
         reference: "ReferenceMatch | None", refusal: str | None = None) -> str:
    """The print-back of section 6, in the fixed section order, all lengths in the
    sketch's units with the metre equivalent once on the extent line. Renders only the
    sections it has: at rc 5 with a `partial` (3.15) `plan`/`m` are the partial's, CLAIMS
    reads 'not evaluated: the sketch did not build' and VERDICT 'not ready: LINT <code>';
    at rc 5 without a partial and at rc 3 the whole text is `refusal` under one SCRIPT line
    (the one rule for rc 3/5 text; 3.10 and 6.1 say the same)."""
    raise NotImplementedError(_U3)


def verdict(findings: list[Finding], table: ComplianceTable | None) -> tuple[str, bool]:
    """('ready to COMMIT', True) | ('not ready: LINT E-ROW-FIT; claims c6 FAIL   (COMMIT
    disagrees: c6 records it)', False). Computed, never parsed; the desk uses the bool."""
    raise NotImplementedError(_U3)


def leg_lines(legs: dict, wall_frame: dict | None = None) -> list[str]:
    """mesh2d.leg_lines by path (the regex-pinned format), plus for a Row the first instance
    then 'x4 at pitch 14.66 (13.02 + 1.64 solved)'; every instance reference is `<row>[k]`
    0-based (D29)."""
    raise NotImplementedError(_U3)
