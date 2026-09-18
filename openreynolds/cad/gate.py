"""The declared finish, and what binds at it.

`buildup/gate.py` answered this for the core desk and the answer generalises, so the
protocol lives here and both desks use it. What differs between them is only where the
advisory numbers come from: the core desk reads `buildup/probes.py` against a local case
directory, the shipped desk reads the findings `check.py` already gathered over the
backend. The labelling, the waivers and the prose are the same thing twice, and were.

## What is binding, and what this changes

**The shipped desk used to gate on every one of these.** `check.ok` was
`worst_status(findings) != "fail"`, and `cad_audit`'s findings are in that list, so one
open edge on a deliberate baffle failed the finish with no way past it: there was no
waiver, and declaring again unchanged got the same answer. A correct geometry could not
be delivered. That is strictly worse than either of the two defensible positions, and it
was invisible because the corpus measures the core desk, where those same checks have
always been advisory.

The evidence for which way to resolve it is in `buildup/gate.py`: three of the six probes
were measured defective and a fourth was wrongly accused, and `coverage` warned on 19 of
20 correct partitions in `core+cad_export-20260917-022129-dd05` §3.1. A gate built on
that blocks correct work. So:

* **binding** -- `checkMesh`, that a mesh exists at all, the size the request named, and
  the render. OpenFOAM's own verdict, and three facts about the case that are true or
  not. None of them has ever been wrong about a correct case.
* **advisory** -- everything `cad_audit.py` and `domain_probe.py` say. Reported to the
  desk, waivable with a reason, recorded either way, and never a silent pass: an
  unresolved warning bounces the declare exactly as a failing `checkMesh` does.

`check.py`'s own `scale` finding stays **binding**, and the first draft of this had it
advisory until a test said otherwise. It is not the same measurement as `cad_audit.scale`:
that one reads the union's extent, this one compares the mesh bounds to the largest
dimension the request names, and it only fires past a factor of 100. Nothing legitimate
lives out there -- an external-flow box around a small body is a factor of ten -- so what
it catches is the millimetres-for-metres error, which is self-consistent everywhere it
appears and is the one wrong answer nobody downstream can see. A waiver on that is a
waiver on the most expensive silent failure in the product.

The asymmetry is deliberate. A binding check that is wrong blocks a correct mesh and the
desk cannot say so. An advisory one that is wrong costs a waiver and a line in the
record, and the record is what eventually repairs the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

CLEAN = "clean"
WARNED = "warned"
XFAIL = "xfail"
WAIVED = "waived"
XPASS = "xpass"
NOT_RUN = "n/a"

STATES = (CLEAN, WARNED, XFAIL, WAIVED, XPASS, NOT_RUN)

ADVISORY_SCRIPTS = ("cad_audit", "domain_probe")
"""The two scripts `check.verify` runs on the backend whose answers are advisory.

Named by script rather than by check so a new finding from either arrives advisory. The
alternative -- a whitelist of check names -- makes the *absence* of a name mean binding,
which is the wrong default for a script whose whole job is to have opinions."""

BINDING = ("mesh", "look", "checkMesh", "render", "scale", "patches", "patch_count")
"""Kept as prose rather than used as a test: `is_advisory` decides, and this is the list
a reader wants when asking what survived the split.

`replay` and `build` were here until 2026-09-18 and are not checks any more -- see
`check._leave_script` for why. The cell log is still the artifact; it is simply no longer
something the desk can fail."""


def is_advisory(check: str) -> bool:
    """Whether this finding is advisory rather than binding.

    Takes the finding's own name as `check.py` writes it -- `_cad_findings` prefixes a
    script's findings with the script name, so `cad_audit.closure` and
    `domain_probe.location_in_mesh` are the forms that arrive here.
    """
    name = str(check or "")
    return any(name.startswith(f"{script}.") for script in ADVISORY_SCRIPTS)


RENAMED = {"cad_audit.scale": "surface_scale"}
"""Advisory checks whose bare name would collide with a binding one.

`cad_audit.scale` measures the exported union's extent and is advisory.  `check.py` also
raises a `scale`, comparing the mesh bounds to the largest dimension the request names,
and that one binds -- it fires only past a factor of 100, where the millimetres-for-metres
error lives and nothing legitimate does.  Two different measurements, and stripping the
prefix gave them one name, so a desk reading `scale` in the enum could not tell which it
was being offered.  It is offered the one it can have."""


def gate_id(check: str) -> str:
    """The name the desk waives this check by: the finding's name without its script.

    `cad_audit.closure` is `closure` to the desk. The script prefix says which of our
    files computed it, which is our bookkeeping and not something the desk should have to
    type. Where stripping it would collide with a binding check, `RENAMED` says so.
    """
    name = str(check or "")
    if name in RENAMED:
        return RENAMED[name]
    for script in ADVISORY_SCRIPTS:
        if name.startswith(f"{script}."):
            return name[len(script) + 1:]
    return name


WAIVABLE = ("closure", "manifold", "normals", "degenerate", "coverage",
            "self_intersection", "surface_scale", "manifest", "surface_check",
            "location_in_mesh", "min_width", "min_wall_thickness", "domain")
"""Every advisory check either script can raise, and the enum the tool offers.

`checkMesh` and the other binding checks are deliberately absent: they cannot be waived,
and a schema that will not form the call is better than a handler that rejects it after
the fact. Held against the tool's own copy by a test rather than by an import, because
`agent.py` defines the tool this module's caller uses."""

ALIASES = {"union_closure": "closure", "scale": "surface_scale"}
"""The core desk's vocabulary, accepted here.

`buildup/probes.py` calls the welded-union check `union_closure` and `cad_audit.py` calls
it `closure`; the same probe's `scale` is `cad_audit.scale`, renamed here. Both
vocabularies are in the corpus, in the sweep reports and in two briefs, so a desk that has
read either forms a call this accepts rather than one it has to be corrected on.

Aliasing `scale` is safe in the direction that matters. It is not in the enum, so a desk
following the schema never types it; one that types it anyway has the core desk's
vocabulary in mind, where `scale` *is* the surface measurement. And the binding
request-scale check is not waivable by any name, so nothing here can reach it."""


@dataclass
class GateState:
    """One check, after the waivers have been read against it."""

    check: str
    state: str
    concern: str = ""
    """The one line the check gives when it has something to say; empty when it does not."""
    because: str = ""
    """The desk's reason, when it named this check."""

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "state": self.state,
                "concern": self.concern, "because": self.because}


@dataclass
class Declaration:
    """One `declare_complete` call and what the gates made of it."""

    outcome: str = "complete"
    reason: str = ""
    states: list[GateState] = field(default_factory=list)
    checkmesh_ok: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "reason": self.reason,
                "checkmesh_ok": self.checkmesh_ok,
                "states": [s.as_dict() for s in self.states]}

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.states:
            out[s.state] = out.get(s.state, 0) + 1
        return out


def concern_of(finding: Any) -> str:
    """The one line this finding has to say, or "" when it is content.

    The finding's own `status` decides. `cad_audit.py` and `domain_probe.py` already
    grade what they measure -- that is the whole of what a `Finding` is -- so re-deriving
    a verdict from their numbers here would be the second opinion `_cad_findings` exists
    to avoid: *"nothing here recomputes a triangle"*.

    `skipped` is not a concern and is not a pass. A check that could not read the surface
    has said nothing, and `evaluate` records that as `n/a`.
    """
    status = str(getattr(finding, "status", "") or "")
    if status not in ("fail", "warn"):
        return ""
    measured = str(getattr(finding, "measured", "") or "")
    # The interpretation with the number, because the number alone is what the desk
    # already has. `9 open edges in the union of 2 patch files` states a fact; `the
    # exported surface has a hole in it, so snappyHexMesh cannot tell inside from
    # outside` is why that fact ends the run, and it is the half a desk acts on. The
    # probe version of this carried both in one sentence and reading only `measured`
    # quietly dropped the second -- a test noticed, on the wording rather than on the
    # number, which is the only way it could have.
    meaning = str(getattr(finding, "meaning", "") or "")
    return f"{measured} -- {meaning}" if meaning else measured


def evaluate(findings: Iterable[Any], waivers: Iterable[dict[str, Any]],
             already_warned: Iterable[str] = ()) -> list[GateState]:
    """Read the waivers against the advisory findings and label each check.

    `already_warned` is the set of checks that have raised a concern on some *earlier*
    declare in this run. It is what separates a prediction from a reaction, and the desk
    has no say in it: a waiver named before that check has ever fired is falsifiable and
    made without the result in hand; one named after is a desk with an interest in
    dismissing it. Both are allowed and both are recorded, apart.

    `XPASS` is what pays for the scheme. A desk that predicts a warning it does not get
    has misread its own geometry or misread the check, and both are findings -- which is
    also what makes blanket-waiving self-punishing.
    """
    named: dict[str, str] = {}
    for waiver in waivers or ():
        check = str((waiver or {}).get("check") or "")
        check = ALIASES.get(check, check)
        if check in WAIVABLE:
            named[check] = str((waiver or {}).get("because") or "").strip()
    seen = {ALIASES.get(c, c) for c in (already_warned or ())}

    out: list[GateState] = []
    for finding in findings or ():
        if not is_advisory(getattr(finding, "check", "")):
            continue
        pid = gate_id(getattr(finding, "check", ""))
        if str(getattr(finding, "status", "") or "") == "skipped":
            out.append(GateState(pid, NOT_RUN,
                                 str(getattr(finding, "measured", "") or "")[:200]))
            continue
        concern = concern_of(finding)
        because = named.get(pid, "")
        if concern and pid in named:
            out.append(GateState(pid, WAIVED if pid in seen else XFAIL, concern, because))
        elif concern:
            out.append(GateState(pid, WARNED, concern))
        elif pid in named:
            out.append(GateState(pid, XPASS, "", because))
        else:
            out.append(GateState(pid, CLEAN))
    return out


_ABSOLUTE = re.compile(r"/(?:[^\s/\"']+/){2,}([^\s/\"']+/[^\s/\"']+)")


def scrub(text: str, case_dir: str) -> str:
    """Strip the absolute case path out of anything the desk is going to read.

    Inherited from the core desk's gate, where getting it wrong voids a sweep: probe
    prose names the workspace by absolute path, and `isolation.scan_run` greps the whole
    thread for exactly that. It matters less here -- a shipped session is allowed to know
    where its own workspace is -- and it is kept because a path in a warning is noise the
    desk has to read past either way.
    """
    if not text:
        return ""
    out = text
    if case_dir:
        for form in (case_dir.rstrip("/") + "/", case_dir.rstrip("/")):
            out = out.replace(form, "")
    out = _ABSOLUTE.sub(r"\1", out)
    return out.strip()


def render(states: Iterable[GateState], case_dir: str = "") -> str:
    """What the desk is handed back. Empty when there is nothing to say."""
    rows = [s for s in states if s.state in (WARNED, XFAIL, WAIVED, XPASS)]
    if not rows:
        return ""
    lines = ["The gates ran. checkMesh is the binding one; these are the others:"]
    for state in sorted(rows, key=lambda r: (r.state != WARNED, r.check)):
        if state.state == WARNED:
            lines.append(f"  [warned] {state.check}: {scrub(state.concern, case_dir)}")
        elif state.state in (XFAIL, WAIVED):
            word = "expected" if state.state == XFAIL else "waived after the fact"
            lines.append(f"  [{word}] {state.check}: {scrub(state.concern, case_dir)}"
                         f"  -- you said: {state.because}")
        else:
            lines.append(f"  [xpass] {state.check}: you expected this to flag and it did "
                         f"not -- you said: {state.because}")
    warned = [s.check for s in rows if s.state == WARNED]
    if warned:
        lines.append("")
        lines.append("Fix these and declare again, or declare again naming them in "
                     "`waive` with the reason each one is correct here. Until every "
                     "warning is fixed or waived the declare is not accepted -- the same "
                     "way a failing checkMesh is not accepted. A reason is what gets you "
                     "past one; declaring again unchanged does not.")
        # Both names, because the two desks measure this with different code and call it
        # different things: `cad_audit.py` says `closure` and `buildup/probes.py` says
        # `union_closure`. Keying on one of them silently drops the paragraph for the
        # other desk, which is how the misconception this paragraph exists to correct
        # went three-for-three in the first sweep.
        if {"closure", "union_closure"} & set(warned):
            lines.append("")
            lines.append("On closure: it welds every STL in the directory into one "
                         "surface and counts the free edges of that union. Individual "
                         "patch files are open by construction and that is not what it "
                         "measures, so 'each patch is a separate sheet' does not explain "
                         "a non-zero count.")
    return "\n".join(lines)
