"""Shared by the measure and lint tests: an ops-grammar fixture (DESIGN.md 8.1, D21)
built through `mesh2d.build_face` by path, walked, measured and judged, with a sidecar
plan hand-written where the fixture needs what `compile.plan` will later produce
(a feature's `lands_on`, a Row's declared gap, `apart` pairs, `expect`).

Not a test module: pytest collects `test_*.py` only; this sits beside `conftest.py`
so `from geometry_specs import ...` resolves from the tests directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from openreynolds.geometry import _toolbox, lint, measure
from openreynolds.geometry.compile import Plan, Solved

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "geometry"


def load_spec(name: str) -> dict:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"ops": payload}


def spec_plan(spec: dict, *, features: dict[str, Solved] | None = None, instances: dict | None = None,
              apart: list | None = None, outlines: dict | None = None, expected=(1, 1), rules=None,
              declared_widths=None, units: str | None = None, ports=None, edge_targets=None) -> Plan:
    """A Plan for an ops fixture: units from the scale, declared widths from the channels,
    the repeat ops as rows, the sidecar's features / instances / apart / outlines, and,
    for the port-intent tests, `ports` (PortIntents whose edges are strings) with the
    `edge_targets` compile.plan will produce for them."""
    scale = float(spec.get("scale", 1.0))
    ops = list(spec["ops"])
    if declared_widths is None:
        declared_widths = [float(op["width"]) for op in ops if op.get("op") == "channel"]
    rows = dict(instances or {})
    for op in ops:
        if op.get("op") == "repeat" and op.get("name") not in rows:
            step = op.get("step", [0, 0])
            rows[op["name"]] = {"count": int(op["count"]), "step": tuple(step), "gap": None, "gap_declared": False,
                                "footprint": (0.0, 0.0)}
    return Plan(units=units or measure.UNITS_BY_SCALE.get(scale, "units"), scale=scale, ops=ops,
                rules=list(spec.get("patches", []) if rules is None else rules), ports=list(ports or []),
                features=dict(features or {}), instances=rows, outlines=dict(outlines or {}),
                edge_targets=dict(edge_targets or {}), apart=list(apart or []), declared_widths=declared_widths,
                expected=tuple(expected), notes=[])


def build(gmsh, spec: dict, model: str = "fixture"):
    """(face, legs, checks) through mesh2d.build_face at scale 1 in a fresh model."""
    mesh2d = _toolbox.load("mesh2d")
    gmsh.model.add(model)
    ops, _ = mesh2d.parse_spec(spec)
    legs: dict = {}
    checks: list = []
    face = mesh2d.build_face(gmsh, ops, scale=1.0, legs=legs, checks=checks)
    return face, legs, checks


def outline_of(gmsh, spec: dict, model: str = "outline") -> list[list[tuple[float, float]]]:
    """The sampled outline loops of a spec built alone (what compile.plan's `outlines`
    will carry in closed form): the outer loop first, holes after."""
    face, _, _ = build(gmsh, spec, model)
    wk = measure.walk(gmsh, face, 1.0)
    return [wk.loop_points(k) for k in range(len(wk.loops))]


def sub_spec(spec: dict, name: str) -> dict:
    """The ops an op named `name` depends on, as a spec whose body is that op."""
    ops = {op["name"]: op for op in spec["ops"]}
    keep: list[str] = []

    def visit(n):
        op = ops[n]
        for key in ("of", "take"):
            for dep in op.get(key, []) or []:
                visit(dep)
        for key in ("from", "target"):
            dep = op.get(key)
            if isinstance(dep, str) and op.get("op") != "channel" and dep in ops:
                visit(dep)
        if n not in keep:
            keep.append(n)

    visit(name)
    out = [dict(ops[n]) for n in keep]
    return {"scale": spec.get("scale", 1.0), "ops": _rename_body(out, keep[-1])}


def _rename_body(ops: list[dict], last: str) -> list[dict]:
    """The last op becomes 'body' and references to it follow."""
    renamed = []
    for op in ops:
        op = dict(op)
        if op["name"] == last:
            op["name"] = "body"
        renamed.append(op)
    return renamed


def run(gmsh, spec: dict, plan: Plan | None = None, claims=None, cfg: lint.LintConfig | None = None,
        model: str = "fixture"):
    """Build, walk, resolve ports, measure and judge one fixture. Returns
    (face, wk, m, findings, legs, checks)."""
    plan = plan if plan is not None else spec_plan(spec)
    face, legs, checks = build(gmsh, spec, model)
    w_hint = min(plan.declared_widths) if plan.declared_widths else 1.0
    wk = measure.walk(gmsh, face, w_hint)
    ends = measure.open_ends(gmsh, wk, w_hint, None)
    curve_patch, _, _ = measure.resolve_ports(gmsh, wk, plan, ends)
    m = measure.measure(gmsh, face, plan, wk, legs, curve_patch, claims)
    findings = lint.judge(gmsh, face, plan, wk, m, checks, claims, cfg or lint.LintConfig())
    return face, wk, m, findings, legs, checks


def bypass_plan(spec: dict, channel: str, name: str | None = None, kind: str = "Bypass", value: float = 1.5,
                **extra) -> Plan:
    """A sidecar plan whose feature `name` (default: the channel op) records `lands_on`
    main.top with the wall line y = value flowing +x, outward +y -- what compile.plan
    writes for a Bypass or an `end_on` Passage (DESIGN.md 3.4's Solved keys)."""
    wall = {"axis": 1, "value": value, "flow": (1.0, 0.0), "outward": (0.0, 1.0), "span": 60.0}
    ops = [op["name"] for op in spec["ops"] if op["name"].startswith(channel)]
    feature = Solved(kind=kind, params={}, solved={"lands_on": "main.top", "wall_line": wall}, ops=ops or [channel])
    return spec_plan(spec, features={name or channel: feature}, **extra)


def codes(findings) -> list[str]:
    return [f.code for f in findings]


def with_code(findings, code: str) -> list:
    return [f for f in findings if f.code == code]
