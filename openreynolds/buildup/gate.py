"""The gates that run when the desk declares itself complete, and what the desk is told.

`checkMesh` is binding and is not in here. Everything in here is **advisory**: it is
reported to the desk, recorded, and never blocks a finish. That split is the answer §7
left open -- "whether an activated check is a gate or a reported finding" -- and the
reason it can be answered now is that three of the six probes were measured defective and
a fourth was wrongly accused. A gate built on any of them would block correct work; a
warning built on all of them costs nothing when wrong and *generates the evidence to
repair them*, which two sweeps of firing-with-no-consequence did not.

## xfail, and why the order of declaration is the whole point

A waiver named in a declare **before** that check has ever warned is a prediction: *the
surface is open because I built a baffle, and `union_closure` is about to say so*. It is
falsifiable before the fact and it is made without the result in hand. A waiver named
**after** the warning is a reaction, by a desk that now has an interest in dismissing it.
Both are allowed, because the first gate run can legitimately surface something nobody
could predict -- but they are recorded apart, and they weigh differently.

The desk does not choose which it gets. It states its waivers; `evaluate` labels each one
from whether that check had already fired, so the stronger claim cannot be claimed.

`XPASS` is the third outcome and the one that pays for the scheme. A desk that predicts a
warning it does not get has misread its own geometry or misread the check, and both are
findings. It also makes blanket-waiving self-punishing: name all six and five come back
`xpass`, in the record, every run.
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

WAIVABLE = ("union_closure", "normals", "self_intersection",
            "location_in_mesh", "coverage", "scale")
"""Every advisory gate, and the enum the tool offers.

`checkmesh` is deliberately absent. It is binding, so it is not waivable, and saying so in
the schema is better than rejecting it in the handler: the desk cannot form the call."""


@dataclass
class GateState:
    """One check, after the waivers have been read against it."""

    check: str
    state: str
    concern: str = ""
    """The one line the probe gives when it has something to say; empty when it does not."""
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


def concern_of(probe: dict[str, Any]) -> str:
    """The one line this probe has to say, or "" when it is content.

    Reads the probe's own `measured` numbers rather than its prose, because the prose
    carries workspace paths and the numbers do not. What counts as a concern is per
    probe and is deliberately generous: an advisory channel that under-reports is the
    silent failure it exists to replace.
    """
    if probe.get("state") != "measured":
        return ""
    m = probe.get("measured") or {}
    pid = probe.get("id", "")
    if pid == "union_closure":
        n = int(m.get("open_edges") or 0)
        return (f"the welded surface has {n:,} free edges, so it does not close"
                if n else "")
    if pid == "normals":
        n = int(m.get("flipped_edges") or 0)
        return (f"{n:,} edges are walked twice the same way, so the winding is "
                "inconsistent across the surface" if n else "")
    if pid == "self_intersection":
        n = int(m.get("pairs") or 0)
        tri = int(m.get("triangles") or 0)
        return (f"{n:,} triangle pairs cross, over {tri:,} triangles" if n else "")
    if pid == "location_in_mesh":
        if str(m.get("classification") or "") == "outside":
            return ("the meshing point is outside the exported surface by "
                    f"{float(m.get('clearance_m') or 0.0):.4g} m")
        return ""
    if pid in ("coverage", "scale"):
        # Neither has ever returned a verdict; when one does, its own words are the
        # concern, because there is no measured shape to read yet.
        return str(probe.get("why") or "")
    return ""


def evaluate(probes: Iterable[dict[str, Any]],
             waivers: Iterable[dict[str, Any]],
             already_warned: Iterable[str] = ()) -> list[GateState]:
    """Read the waivers against the probes and label each check.

    `already_warned` is the set of checks that have raised a concern on some *earlier*
    declare in this run. It is what separates a prediction from a reaction, and the desk
    has no say in it.
    """
    named = {}
    for w in waivers or ():
        check = str((w or {}).get("check") or "")
        if check in WAIVABLE:
            named[check] = str((w or {}).get("because") or "").strip()
    seen = set(already_warned or ())

    out: list[GateState] = []
    for probe in probes or ():
        pid = str(probe.get("id") or "")
        if probe.get("state") != "measured":
            out.append(GateState(pid, NOT_RUN, str(probe.get("why") or "")[:200]))
            continue
        concern = concern_of(probe)
        because = named.get(pid, "")
        if concern and pid in named:
            # Named after it had already fired is a reaction; named before is a
            # prediction. The desk states the waiver; this line picks the label.
            out.append(GateState(pid, WAIVED if pid in seen else XFAIL, concern, because))
        elif concern:
            out.append(GateState(pid, WARNED, concern))
        elif pid in named:
            out.append(GateState(pid, XPASS, "", because))
        else:
            out.append(GateState(pid, CLEAN))
    return out


_HOUSE = re.compile(r"(/[^\s\"']*?)(?=/[^/\s\"']+/?(?:constant|system|0)\b|$)")


def scrub(text: str, case_dir: str) -> str:
    """Strip the absolute case path out of anything the desk is going to read.

    **This is the one thing that voids a sweep if it is got wrong.** Probe prose names the
    workspace by absolute path -- `no readable patch set under /home/.../work/T26-.../t26/
    constant/triSurface` -- and until now that string only ever reached the record.
    Showing it to the desk puts a house path into the conversation, where `scan_run` greps
    the whole thread for exactly that, and every run of the sweep grades contaminated.
    """
    if not text:
        return ""
    out = text
    if case_dir:
        for form in (case_dir.rstrip("/") + "/", case_dir.rstrip("/")):
            out = out.replace(form, "")
    # Anything still absolute is reduced to its last two segments, which is enough to say
    # which file is meant and not enough to say where this machine keeps it.
    out = re.sub(r"/(?:[^\s/\"']+/){2,}([^\s/\"']+/[^\s/\"']+)", r"\1", out)
    return out.strip()


def render(states: Iterable[GateState], case_dir: str = "") -> str:
    """What the desk is handed back. Empty when there is nothing to say."""
    rows = [s for s in states if s.state in (WARNED, XFAIL, WAIVED, XPASS)]
    if not rows:
        return ""
    lines = ["The gates ran. checkMesh is the binding one; these are the others:"]
    for s in sorted(rows, key=lambda r: (r.state != WARNED, r.check)):
        if s.state == WARNED:
            lines.append(f"  [warned] {s.check}: {scrub(s.concern, case_dir)}")
        elif s.state in (XFAIL, WAIVED):
            word = "expected" if s.state == XFAIL else "waived after the fact"
            lines.append(f"  [{word}] {s.check}: {scrub(s.concern, case_dir)}"
                         f"  -- you said: {s.because}")
        else:
            lines.append(f"  [xpass] {s.check}: you expected this to flag and it did "
                         f"not -- you said: {s.because}")
    warned = [s.check for s in rows if s.state == WARNED]
    if warned:
        lines.append("")
        lines.append("Fix these and declare again, or declare again naming them in "
                     "`waive` with the reason each one is correct here. Until every "
                     "warning is fixed or waived the declare is not accepted -- the same "
                     "way a failing checkMesh is not accepted. A reason is what gets you "
                     "past one; declaring again unchanged does not.")
        if "union_closure" in warned:
            lines.append("")
            lines.append("On union_closure: it welds every STL in the directory into one "
                         "surface and counts the free edges of that union. Individual "
                         "patch files are open by construction and that is not what it "
                         "measures, so 'each patch is a separate sheet' does not explain "
                         "a non-zero count.")
    return "\n".join(lines)
