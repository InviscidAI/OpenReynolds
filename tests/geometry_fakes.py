"""Fakes for the desk's tests (DESIGN.md section 9: fakes and fixtures let each unit's
tests pass alone).

Two kinds. `outcome()` builds a canned `RunOutcome` in the 3.10 / 4.3 shape, which is
what the desk reads: the child never runs in these tests. `install_u3_fallbacks` stands
in for the pieces of U3 (claims.parse, claims.can_commit, the two `lines()` methods,
fitness.from_digest, fitness.schemes_for) the desk calls at runtime -- but only while
the real one still raises NotImplementedError, so the day U3 lands the same tests run
over the real code and nothing here is used.
"""
from __future__ import annotations

import copy

from openreynolds.geometry import claims as claims_mod
from openreynolds.geometry import fitness as fitness_mod
from openreynolds.geometry.claims import (
    CHECKABLE_KINDS, KINDS, LENGTH_MEASURES, Claim, ClaimSet, ClaimsError, ComplianceRow, ComplianceTable,
)
from openreynolds.geometry.fitness import FitnessTable, Unmeasured
from openreynolds.geometry.lint import Finding
from openreynolds.geometry.runner import RunOutcome

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

CLAIMS = {
    "schema": "openreynolds.geometry/claims-1",
    "unit": "mm", "kind": "passage", "flow": "+x",
    "claims": [
        {"id": "c1", "kind": "measure", "says": "a straight channel 3 mm wide", "measure": "width", "of": "main",
         "value": 3},
        {"id": "c2", "kind": "measure", "says": "60 mm long", "measure": "length", "of": "main", "value": 60,
         "tol": 0.1},
        {"id": "c3", "kind": "count", "says": "4 bypass loops", "of": "loops", "value": 4},
        {"id": "c4", "kind": "measure", "says": "loop channel 3 mm wide", "measure": "width", "of": "loops[*]",
         "value": 3},
        {"id": "c5", "kind": "measure", "says": "outer radius 6 mm", "measure": "outer_radius", "of": "loops[*]",
         "value": 6},
        {"id": "c6", "kind": "predicate", "says": "leave the main channel at a shallow angle",
         "predicate": "shallow_angle", "of": "loops[*]", "args": {"max": 30}},
        {"id": "c7", "kind": "predicate", "says": "return into the main channel against the forward direction",
         "predicate": "returns_against_flow", "of": "loops[*]"},
        {"id": "c8", "kind": "patch", "says": "inlet at x=0", "patch": "inlet", "at": [0, 0], "tol": 0.5},
        {"id": "c9", "kind": "patch", "says": "outlet at x=60 mm", "patch": "outlet", "at": [60, 0], "tol": 0.5},
        {"id": "c10", "kind": "report", "says": "sweep round", "measure": "sweep", "of": "loops[*]"},
        {"id": "c11", "kind": "report", "says": "(how steeply it returns, and how far apart the loops are)",
         "measure": "return_angle", "of": "loops[*]"},
        {"id": "c12", "kind": "not_measurable", "says": "produce the mesh, run checkMesh, render the mesh",
         "not_measurable": "the finish and the main agent do these; the fitness table reports the mesh"},
    ],
}
"""DESIGN.md 4.1's claims file: 9 measurable, 2 reported, 1 not measurable."""

SCRIPT = '''s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=0, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6, return_angle=80, name="loop")
loops = Row(loop, count=4, name="loops")
s.fluid = main | loops
s.inlet(main.start)
s.outlet(main.end)
'''

CHECKMESH_LOG = """\
Mesh stats
    points:           8206
    cells:            3750
    hexahedra:     3750
    prisms:        0
Checking geometry...
    Mesh non-orthogonality Max: 42.3202 average: 7.25575
    Max skewness = 1.14032 OK.
Mesh OK.
"""

BOUNDARY = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}

4
(
    inlet
    {
        type            patch;
        nFaces          10;
        startFace       7300;
    }
    outlet
    {
        type            patch;
        nFaces          10;
        startFace       7310;
    }
    walls
    {
        type            wall;
        nFaces          260;
        startFace       7320;
    }
    frontAndBack
    {
        type            empty;
        nFaces          7500;
        startFace       7580;
    }
)
"""


def measurements_dict(extent=(60.0, 14.25), passage_min: float | None = 2.98, units="mm", scale=0.001) -> dict:
    """A Measurements.as_dict() with every key the round trip needs."""
    return {
        "units": units, "scale": scale, "bounds": [0.0, -1.5, extent[0], extent[1] - 1.5],
        "extent": list(extent), "area": 513.7, "islands": 4, "n_curves": 36,
        "shortest_edge": [1.5, [0.75, 1.5]],
        "patches": {"inlet": {"curves": [1], "n": 1, "length": 3.0, "midpoints": [[0, 0]]},
                    "outlet": {"curves": [2], "n": 1, "length": 3.0, "midpoints": [[60, 0]]}},
        "legs": {}, "features": {"main": {"width": 3.0, "length": 60.0}}, "rows": {},
        "open_ends": [], "junctions": [], "vertices": [],
        "passage": None if passage_min is None else {"min": passage_min, "where_min": [30.0, 0.0], "p10": 3.0,
                                                    "median": 3.0, "n": 200},
        "reference_width": 3.0, "reference_width_from": "declared by 'main'", "holes": [], "flow": [1.0, 0.0],
    }


def rows_for(claims: dict, fails=(), unmeasurable=()) -> list[dict]:
    rows = []
    for c in claims["claims"]:
        kind = c["kind"]
        if c["id"] in fails:
            verdict = "fail"
        elif kind == "not_measurable" or c["id"] in unmeasurable:
            verdict = "not_measurable"
        else:
            verdict = "pass"
        rows.append(ComplianceRow(id=c["id"], says=c["says"], kind=kind, expected=str(c.get("value", "")),
                                  measured=f"{c.get('of', '')}.{c.get('measure', c.get('patch', ''))} measured",
                                  verdict=verdict,
                                  detail="the request cannot be met as stated" if verdict == "fail" else "").as_dict())
    return rows


def outcome(rc: int = 0, *, errors=(), warns=(), fails=(), claims: dict | None = CLAIMS, units="mm",
            passage_min: float | None = 2.98, extent=(60.0, 14.25), png: bytes | None = PNG, text: str | None = None,
            script: str = SCRIPT, seconds: float = 1.2) -> RunOutcome:
    """A canned RunOutcome. `errors` / `warns` are (code, (x, y)) pairs that become
    Findings; `fails` are claim ids whose rows FAIL; `claims=None` is a lap with no
    claims file (compliance None)."""
    findings = [Finding(level="error", code=code, subject="Row 'loops'", what=f"{code} at {where}", where=where).as_dict()
                for code, where in errors]
    findings += [Finding(level="warn", code=code, subject="fluid", what=f"{code} at {where}", where=where).as_dict()
                 for code, where in warns]
    m = measurements_dict(extent=extent, passage_min=passage_min, units=units)
    if rc in (3, 4, 5):
        body = text or f"!! ERROR  E-SCRIPT  line 1: refused (rc {rc})"
        return RunOutcome(rc=rc, text=body, png=png if rc == 5 else None, seconds=seconds,
                          result={"ok": False, "rc": rc, "seconds": seconds, "record": None, "measurements": None,
                                  "lint": [], "compliance": None, "report": body, "verdict": None, "preview": None,
                                  "reference": None, "error": body, "traceback": None, "code": "E-SCRIPT"})
    table = None if claims is None else {"rows": rows_for(claims, fails)}
    verdict_text = ("ready to COMMIT" if rc == 0 and table is not None and not fails
                    else "not ready: " + ("LINT " + ", ".join(c for c, _ in errors) if errors else
                                         ("claims " + ", ".join(fails) + " FAIL" if fails else "no claims")))
    body = text or "\n".join([
        f"SCRIPT     ran in {seconds} s; no prints",
        "LINT       " + ("clean" if not errors else f"{len(errors)} error(s) (errors block)"),
        *[f"           !! ERROR  {code}  Row 'loops': at {where}" for code, where in errors],
        "FEATURES   main      Passage    width 3, one leg 60 along +x",
        f"MEASURED   extent {extent[0]:g} x {extent[1]:g} {units} ({extent[0] * 0.001:g} x {extent[1] * 0.001:g} m)"
        "   area 513.7 mm2   islands 4",
        "PATCHES    inlet (1 edge, 3.0) at (0, 0) = main.start     outlet (1 edge, 3.0) at (60, 0) = main.end",
        "CLAIMS     " + ("not evaluated: no claims" if table is None else f"{len(table['rows'])} claims"),
        f"VERDICT    {verdict_text}",
    ])
    record = {
        "format": "openreynolds.geometry/1", "units": units, "scale": 0.001,
        "ops": [{"op": "rect", "name": "fluid", "origin": [0, -1.5], "size": [extent[0], 3]}],
        "patches": [{"name": "inlet", "at": "near:0,0"}, {"name": "outlet", "at": f"near:{extent[0]:g},0"}],
        "ports": [{"name": "inlet", "kind": "inlet", "edges": ["main.start"]},
                  {"name": "outlet", "kind": "outlet", "edges": ["main.end"]}],
        "script": script, "features": {}, "instances": {}, "apart": [],
        "claims": claims, "compliance": table, "lint": findings, "measurements": m,
        "built_with": {"gmsh": "4.15.2", "at": "2026-09-07T10:12:00Z"},
    }
    result = {"ok": rc == 0, "rc": rc, "seconds": seconds, "record": record, "measurements": m, "lint": findings,
              "compliance": table, "report": body, "verdict": {"text": verdict_text, "ready": rc == 0 and not fails},
              "preview": "preview.png", "reference": None, "error": None, "traceback": None, "code": None}
    return RunOutcome(rc=rc, text=body, png=png, result=result, seconds=seconds)


def fresh(o: RunOutcome) -> RunOutcome:
    """A copy the desk may annotate (E-UNITS-CLAIMS appends to the lint) without touching
    the canned original."""
    return copy.deepcopy(o)


# -- U3 stand-ins ----------------------------------------------------------------------

_REQUIRED = {"measure": ("measure", "of", "value"), "range": ("measure", "of", "min", "max"),
             "count": ("of", "value"), "predicate": ("predicate", "of"), "patch": ("patch",),
             "report": ("measure", "of"), "not_measurable": ("not_measurable",)}


def fake_parse(payload: dict) -> tuple[ClaimSet, list[str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        raise ClaimsError("claims: a list of claim objects is required")
    unit = payload.get("unit")
    if unit not in ("mm", "cm", "m", "in"):
        raise ClaimsError(f"unit: one of mm, cm, m, in, not {unit!r}")
    seen: set[str] = set()
    out: list[Claim] = []
    lines: list[str] = []
    for i, raw in enumerate(payload["claims"]):
        if not isinstance(raw, dict) or not raw.get("id"):
            raise ClaimsError(f"claims[{i}]: id is required")
        cid = raw["id"]
        if cid in seen:
            raise ClaimsError(f"{cid}: duplicate id")
        seen.add(cid)
        if not raw.get("says"):
            raise ClaimsError(f"{cid}: says is required (the request's own words)")
        kind = raw.get("kind")
        if kind not in KINDS:
            raise ClaimsError(f"{cid}: kind {kind!r} is not one of {', '.join(KINDS)}")
        for name in _REQUIRED[kind]:
            if name not in raw:
                raise ClaimsError(f"{cid}: {kind} claims need {name}")
        if kind == "patch" and "at" not in raw and "side" not in raw:
            raise ClaimsError(f"{cid}: patch claims need at or side")
        claim = Claim.from_dict(raw)
        if kind == "predicate" and claims_mod.PREDICATES and claim.predicate not in claims_mod.PREDICATES:
            known = ", ".join(claims_mod.PREDICATES)
            claim.kind, claim.not_measurable = "not_measurable", f"no predicate named {claim.predicate}"
            lines.append(f"{cid}: no predicate named {claim.predicate}; known: {known}")
        elif kind == "report":
            lines.append(f"{cid} {claim.measure} of {claim.of}: reported (measured and printed, not judged)")
        elif kind == "not_measurable":
            lines.append(f"{cid} not measurable here: {claim.not_measurable}")
        elif kind == "predicate":
            lines.append(f"{cid} {claim.predicate} of {claim.of}: measurable")
        elif kind == "patch":
            lines.append(f"{cid} {claim.patch} patch: measurable")
        else:
            lines.append(f"{cid} {claim.measure or 'count'} of {claim.of}: measurable")
        out.append(claim)
    lengths = [v for c in out if c.measure in LENGTH_MEASURES for v in (c.value, c.max) if v is not None]
    return ClaimSet(unit=unit, kind=payload.get("kind", "passage"), flow=payload.get("flow"), claims=out,
                    largest_length=max(lengths) if lengths else None), lines


def fake_can_commit(findings, table, reply) -> tuple[bool, str]:
    if table is None:
        return False, "COMMIT refused: no compliance table was recorded; a script must build against the claims first"
    if table.checkable == 0:
        return False, "COMMIT refused: no checkable claim was recorded; the claims lap failed twice"
    errors = [f for f in findings if f.level == "error"]
    if errors:
        named = "; ".join(f"{f.code} at ({f.where[0]:g}, {f.where[1]:g})" if f.where else f.code for f in errors)
        return False, f"COMMIT refused: lint errors block a commit and cannot be disagreed with: {named}"
    failing = set(table.failing_ids())
    named_ids = set(reply.disagrees)
    for cid in sorted(named_ids - failing):
        return False, f"COMMIT refused: nothing to disagree with on {cid}; it passes"
    unnamed = sorted(failing - named_ids)
    if unnamed:
        return False, (f"COMMIT refused: claims {', '.join(unnamed)} FAIL and are not named; reply "
                       f"`COMMIT disagrees: {', '.join(sorted(failing))}` to record them")
    if claims_mod.WARNINGS_BLOCK_COMMIT:
        accepted = {code.upper() for code, _where, _why in reply.accepts}
        left = [f.code for f in findings if f.level == "warn" and f.code.upper() not in accepted]
        if left:
            return False, f"COMMIT refused: warnings not accepted: {', '.join(left)}"
    return True, "ok"


def fake_table_lines(self: ComplianceTable) -> list[str]:
    head = (f"{len(self.rows)} claims: {self.passed} pass, {self.failed} FAIL, {self.unmeasurable} not measurable, "
            f"{sum(1 for r in self.rows if r.kind == 'report')} reported"
            + (f", {self.disagreed} disagreed" if self.disagreed else ""))
    out = [head]
    for r in self.rows:
        verdict = {"fail": "FAIL", "not_measurable": "not measurable"}.get(r.verdict, r.verdict)
        out.append(f"{r.id:<5}{r.says[:34]:<36}{r.measured[:40]:<42}{r.expected[:14]:<16}{verdict}")
        if r.detail:
            out.append(f"     {r.detail}")
    return out


def fake_fitness_lines(self: FitnessTable) -> list[str]:
    out = [f"cells {self.cells} (hex {self.hex_fraction * 100:.0f} %)"]
    if isinstance(self.smallest_passage_m, Unmeasured):
        out.append(f"across the smallest passage: not measured ({self.smallest_passage_m.reason})")
    else:
        out.append(f"across the smallest passage {self.cells_across_smallest_passage:.1f} cells "
                   f"({self.smallest_passage_m * 1e3:.3g} mm / {self.cell_m * 1e3:.3g} mm)")
    if isinstance(self.y_plus, Unmeasured):
        out.append(f"y+ not estimated: {self.y_plus.reason}")
    else:
        out.append(f"y+ {self.y_plus:.3g} ({self.y_plus_verdict})")
    out.append(f"non-orthogonality max {self.non_orth_max:.2f}, mean {self.non_orth_mean:.2f}; "
               f"skewness max {self.skew_max:.2f}")
    out.append(f"schemes {self.schemes}; quality gate {self.quality_gate}")
    return out


def fake_schemes_for(non_orth_max: float) -> str:
    return "standard" if non_orth_max < 65 else ("limited" if non_orth_max <= 70 else "refused")


def fake_from_digest(digest: dict, sizes, smallest_passage_m, study: str, flow, scale: float) -> FitnessTable:
    counts = digest.get("counts", {})
    cells = int(counts.get("cells", 0))
    hexa = int(counts.get("hexahedra", 0))
    non_ortho = digest.get("non_ortho") or ("0", "0")
    non_orth_max, non_orth_mean = float(non_ortho[0]), float(non_ortho[1])
    aspect = digest.get("aspect_ratio")
    cell_m = float(sizes.cell)
    if isinstance(smallest_passage_m, Unmeasured):
        across = Unmeasured(smallest_passage_m.reason)
    else:
        across = float(smallest_passage_m) / cell_m
    return FitnessTable(
        cells=cells, hex_fraction=(hexa / cells) if cells else 0.0, smallest_passage_m=smallest_passage_m,
        cell_m=cell_m, cells_across_smallest_passage=across, wall_cell_m=float(sizes.wall_cell),
        first_cell_height_m=float(sizes.wall_cell) / 2.0,
        y_plus=Unmeasured("no flow speed in a mesh-only study") if flow is None else 1.0,
        y_plus_verdict=Unmeasured("no flow speed in a mesh-only study") if flow is None else "resolved",
        non_orth_max=non_orth_max, non_orth_mean=non_orth_mean, skew_max=float(digest.get("skewness", 0.0)),
        aspect_max=float(aspect) if aspect else Unmeasured("checkMesh's log has no aspect ratio line"),
        schemes=fake_schemes_for(non_orth_max),
        measured_from=f"log.checkMesh + mesh_sizes2d cell {cell_m:.3g} m + passage.min "
                      f"{'unmeasured' if isinstance(smallest_passage_m, Unmeasured) else f'{smallest_passage_m * 1e3:.3g} mm'}")


def _unbuilt(fn, *args) -> bool:
    """True when `fn` is still U0's stub: it raises NotImplementedError. Any other
    outcome (a value, a ClaimsError, a KeyError on a thin argument) means it is built."""
    try:
        fn(*args)
    except NotImplementedError:
        return True
    except Exception:  # noqa: BLE001 - built, and unhappy with the probe's argument
        return False
    return False


def _sample_fitness() -> FitnessTable:
    return FitnessTable(cells=1, hex_fraction=1.0, smallest_passage_m=1e-3, cell_m=1e-4, cells_across_smallest_passage=10.0,
                        wall_cell_m=1e-4, first_cell_height_m=5e-5, y_plus=Unmeasured("probe"), y_plus_verdict=Unmeasured("probe"),
                        non_orth_max=1.0, non_orth_mean=1.0, skew_max=1.0, aspect_max=Unmeasured("probe"), schemes="standard",
                        measured_from="probe")


def install_u3_fallbacks(monkeypatch) -> dict[str, bool]:
    """Patch each U3 function the desk calls with its stand-in while the real one is a
    stub. Returns which were patched, so a test can see what it ran over."""
    from openreynolds.geometry import desk
    patched = {}
    probe_reply = desk.Reply(kind="commit")
    if _unbuilt(claims_mod.parse, {"unit": "mm", "kind": "passage", "flow": "+x", "claims": []}):
        monkeypatch.setattr(claims_mod, "parse", fake_parse)
        patched["parse"] = True
    if _unbuilt(claims_mod.can_commit, [], None, probe_reply):
        monkeypatch.setattr(claims_mod, "can_commit", fake_can_commit)
        patched["can_commit"] = True
    if _unbuilt(ComplianceTable([]).lines):
        monkeypatch.setattr(ComplianceTable, "lines", fake_table_lines)
        patched["table_lines"] = True
    if _unbuilt(_sample_fitness().lines):
        monkeypatch.setattr(FitnessTable, "lines", fake_fitness_lines)
        patched["fitness_lines"] = True
    if _unbuilt(fitness_mod.schemes_for, 10.0):
        monkeypatch.setattr(fitness_mod, "schemes_for", fake_schemes_for)
        patched["schemes_for"] = True
    from openreynolds.geometry.strategy import Sizes
    if _unbuilt(fitness_mod.from_digest, {"counts": {"cells": 1, "hexahedra": 1}, "non_ortho": ("1", "1"),
                                          "skewness": "1"}, Sizes(1e-4, 1e-4, 1e-4), 1e-3, "mesh", None, 1e-3):
        monkeypatch.setattr(fitness_mod, "from_digest", fake_from_digest)
        patched["from_digest"] = True
    return patched

