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

from typing import Any, Iterable

from ..cad.gate import (CLEAN, NOT_RUN, STATES, WAIVED, WARNED, XFAIL, XPASS,
                        Declaration, GateState, render, scrub)

__all__ = ["CLEAN", "WARNED", "XFAIL", "WAIVED", "XPASS", "NOT_RUN", "STATES",
           "WAIVABLE", "GateState", "Declaration", "concern_of", "evaluate",
           "render", "scrub"]

# The labelling, the states, the prose and the scrub are one protocol and they moved to
# `cad/gate.py` when the shipped desk started using them too. What stays here is the only
# half that differs: where the advisory numbers come from. This desk reads
# `buildup/probes.py` against a local case directory; the shipped desk reads the findings
# `check.py` already gathered over the backend, because a shipped session's case is on a
# volume and `Path(case_dir)` does not resolve on the machine holding the conversation.

WAIVABLE = ("union_closure", "normals", "self_intersection",
            "location_in_mesh", "coverage", "scale")
"""Every advisory gate, and the enum the tool offers.

`checkmesh` is deliberately absent. It is binding, so it is not waivable, and saying so in
the schema is better than rejecting it in the handler: the desk cannot form the call."""


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
    if pid == "coverage":
        # `cad_audit`'s own verdict, not the fact that it had one. This branch used to
        # return `why` unconditionally, written when `coverage` could never read
        # anything at all -- it refused without a `patches.json` the core desk never
        # wrote, so the branch was dead and its unconditional return was invisible.
        #
        # `cad_export.export_patches` writes that manifest, and the first sweep in which
        # `coverage` could measure is the sweep in which this fired on **19 of 20 runs**,
        # every one of them on an exhaustive, disjoint, entirely correct partition.
        # Eighteen desks spent a declare turn waiving it. T12's only declare landed on
        # turn 30 of 30, was bounced for want of a waiver it had no turn left to give,
        # and a correct 29,831-cell mesh at the requested volume scored `passed: false`.
        return "" if str(m.get("status") or "") == "ok" else str(probe.get("why") or "")
    if pid == "scale":
        # The probe fires outside a factor of `probes.SCALE_FACTOR`; inside it, a ratio
        # near 1 is the answer and not a concern. Read as a band rather than as a
        # verdict because `_scale` returns no status field -- it returns the ratio, and
        # the ratio is the finding.
        try:
            ratio = float(m.get("ratio") or 0.0)
        except (TypeError, ValueError):
            return str(probe.get("why") or "")
        if ratio and 0.01 <= ratio <= 100.0:
            return ""
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
