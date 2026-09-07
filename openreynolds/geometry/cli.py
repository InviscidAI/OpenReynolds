"""The geometry command line: build | preview | check | reference | golden.

    python -I openreynolds/geometry/cli.py build   --script s.py [--claims c.json] --out DIR [--preview p.png] [--reference NAME]
                                                    (the child of runner.run_script; writes result.json, report.txt, record.json, preview.png)
    python -m openreynolds.geometry.cli preview    (--script | --record | --library NAME [--param k=v] | --spec spec.json [--scale]) --out p.png [--plain]
    python -m openreynolds.geometry.cli check      (--record geometry.json | --script s.py | --spec spec.json) [--claims c.json]   (lint + measure + compliance; exit 2 on errors)
    python -m openreynolds.geometry.cli reference  (prints api_summary(); `--write` rewrites reference.md)
    python -m openreynolds.geometry.cli golden     (--regenerate [NAME] | --check | --approve NAME --by "Kabir")

Exit codes: 0 built; 2 lint errors; 3 the script raised (and E-KERNEL-STATE); 4 timeout
(set by the parent); 5 refused (SketchError / guard / E-REFERENCE-UNAPPROVED). `build`
never writes a case: the case is written by `case.write_case` through `mesh2d.main`.

Run by path under `-I` (D3): the `__main__` shim puts the checkout root on `sys.path` and
imports `openreynolds.geometry.cli`, so the kernel has one module name in the child and
`openreynolds/trace.py` never shadows the stdlib `trace`. The shim sits above the
package-relative imports because, run by path, this file is `__main__` with no parent
package and a relative import would fail before any shim at the bottom ran.

The script runs in a namespace where the API is already bound (`sketch.API`), under an
import guard that refuses every import but math and json from the script's own frame and
nothing else: the kernel's lazy imports (gmsh inside compile, matplotlib inside preview)
happen after the guard is gone or from other frames. A refusal (SketchError) is rc 5 with
the text the API wrote; a Row refusal carries a `partial`, and the lap still gets the
single instance built, measured and drawn (3.15); anything else the script raises is rc 3
with the traceback trimmed to the script's own lines.

Where a kernel instrument is not built yet (a unit still lands in parallel and its
function raises NotImplementedError), the build and the measure run through
`toolbox/mesh2d.py` by path -- the same ops, the same OCC build, mesh2d's own checks turned
into Findings -- so `--spec` works today and each instrument takes over the moment it
exists. Those fallbacks are marked `transitional` and go with Phase 1's integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout root: `openreynolds` importable
    from openreynolds.geometry import cli as _cli                    # one module name for the kernel in the child
    sys.exit(_cli.main())

import argparse  # noqa: E402
import builtins  # noqa: E402
import contextlib  # noqa: E402
import datetime as _dt  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback as _traceback  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from . import _toolbox, claims, compile, library, lint, measure, preview, report, runner, sketch  # noqa: E402
from .compile import Plan, Solved  # noqa: E402
from .lint import Finding  # noqa: E402
from .measure import Measurements, RowMeasure  # noqa: E402
from .sketch import PortIntent, SketchError  # noqa: E402

EXIT_OK, EXIT_LINT, EXIT_RAISED, EXIT_TIMEOUT, EXIT_REFUSED = 0, 2, 3, 4, 5

RLIMIT_AS_BYTES = 2 * 1024 ** 3
"""The child's address-space limit on Linux (3.10); never applied on Windows."""

UNITS_BY_SCALE = {0.001: "mm", 0.01: "cm", 0.0254: "in", 1.0: "m"}
"""A `--spec` fixture's units, read from its scale (3.6, the units predicate)."""

REFERENCE_MD = Path(__file__).resolve().with_name("reference.md")

KWARG_GLOSS = {
    "outer_radius": "of the outer wall", "leave_angle": "from the flow along the wall",
    "return_angle": "from the upstream direction", "leave_length": "the straight leg before the arc",
    "width": "of the channel", "count": "of instances", "gap": "between neighbours",
    "pitch": "anchor to anchor", "radius": "of the centreline", "turn": "degrees, + left / - right",
    "centre": "the centre point", "diameter": "of the disk", "size": "(size_x, size_y)",
    "origin": "the lower-left corner", "heading": "degrees counter-clockwise from +x",
}
"""One phrase per constructor parameter for the E-SCRIPT hint ('Bypass takes outer_radius
(of the outer wall)'); a parameter without one is named alone."""


# -- the import guard ----------------------------------------------------------------


class ImportRefused(Exception):
    """An import from the script's own frame that the guard refused (E-IMPORT)."""

    def __init__(self, name: str, fromlist: tuple, level: int):
        self.name = name
        self.fromlist = fromlist
        self.level = level
        super().__init__(name)


def _guarded_import(real):
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        scope = globals
        if scope is None:
            # `__import__("os")` called by hand passes no globals; the caller's frame
            # still says whose import it is
            scope = sys._getframe(1).f_globals
        if scope.get("__name__") == "__sketch__":
            top = name.split(".")[0] if name else ""
            if level or top not in runner.ALLOWED_IMPORTS:
                raise ImportRefused(name, tuple(fromlist or ()), level)
        return real(name, globals, locals, fromlist, level)
    return guarded


@contextlib.contextmanager
def import_guard():
    """Installed on `builtins.__import__` around the `exec` alone and removed in the
    finally: it refuses only when the calling frame's globals carry `__name__ ==
    "__sketch__"`, so the kernel's own imports are never touched and everything after the
    `with` runs with the real `__import__`."""
    real = builtins.__import__
    builtins.__import__ = _guarded_import(real)
    try:
        yield
    finally:
        builtins.__import__ = real


def apply_rlimits(timeout_s: float) -> None:
    """On Linux RLIMIT_AS 2 GB and RLIMIT_CPU = timeout + 5; a limit is only ever lowered,
    never raised, and a platform without `resource` (Windows) is left alone."""
    if sys.platform == "win32":
        return
    try:
        import resource
    except ImportError:
        return
    for name, value in (("RLIMIT_AS", RLIMIT_AS_BYTES), ("RLIMIT_CPU", int(timeout_s) + 5)):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            soft, hard = resource.getrlimit(limit)
            new = value if hard == resource.RLIM_INFINITY else min(value, hard)
            if soft == resource.RLIM_INFINITY or new < soft:
                resource.setrlimit(limit, (new, hard))
        except (ValueError, OSError):
            pass


# -- the texts of a failed script -------------------------------------------------------


def _bound_names() -> str:
    names = [n for n in sketch.API if n not in ("math", "EdgeRef")]
    return ", ".join(names) + " and math are already bound"


def _source_line(src: str, lineno: int | None) -> str:
    lines = src.splitlines()
    if lineno is None or lineno < 1 or lineno > len(lines):
        return ""
    return lines[lineno - 1].strip()


def _script_frames(tb) -> list[tuple[int, str]]:
    """(line, function) for every frame whose file is the script, outermost first."""
    frames = []
    for frame in _traceback.extract_tb(tb):
        if frame.filename == "script.py":
            frames.append((frame.lineno, frame.name))
    return frames


def _refusal_text(code: str, head: str, *lines: str) -> str:
    return "\n".join([f"!! ERROR  {code}  {head}", *(f"          {line}" for line in lines)])


def import_refusal(exc: ImportRefused, src: str) -> tuple[str, str]:
    """(code, text) for E-IMPORT: the statement quoted from the script's line, and the fix
    -- the API needs no import for its own module under either spelling, and the
    arithmetic the other imports were for is in the print-back."""
    frames = _script_frames(exc.__traceback__)
    lineno = frames[-1][0] if frames else 1
    statement = _source_line(src, lineno) or (f"from {'.' * exc.level}{exc.name} import ..." if exc.fromlist
                                              else f"import {exc.name}")
    parts = exc.name.split(".") if exc.name else []
    api_spelling = "sketch" in parts or (parts and parts[0] in ("openreynolds", "geometry"))
    if api_spelling:
        return "E-IMPORT", _refusal_text("E-IMPORT", f"line {lineno}: `{statement}` -- the API needs no import",
                                         _bound_names())
    # section 5's order ("math and json"), not the set's: the text is pinned
    allowed = " and ".join(n for n in ("math", "json") if n in runner.ALLOWED_IMPORTS)
    return "E-IMPORT", _refusal_text("E-IMPORT", f"line {lineno}: `{statement}` -- the script may import {allowed} only",
                                     "the API does the arithmetic: footprints, landings and pitches are in the print-back")


def _constructor_hint(message: str) -> str | None:
    """A TypeError on a known constructor gets the signature hint: 'Bypass takes
    outer_radius (of the outer wall); see the reference card'."""
    m = re.match(r"(\w+)(?:\.__init__)?\(\) got an unexpected keyword argument '(\w+)'", message)
    if m and m.group(1) in sketch.API:
        cls, wrong = sketch.API[m.group(1)], m.group(2)
        try:
            params = [p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "alias")]
        except (TypeError, ValueError):
            params = []
        close = difflib.get_close_matches(wrong, params, n=1, cutoff=0.4)
        if close:
            gloss = KWARG_GLOSS.get(close[0])
            return f"{m.group(1)} takes {close[0]}" + (f" ({gloss})" if gloss else "") + "; see the reference card"
        if params:
            return f"{m.group(1)} takes {', '.join(params)}; see the reference card"
    m = re.match(r"(\w+)(?:\.__init__)?\(\) missing \d+ required keyword-only arguments?: (.+)", message)
    if m and m.group(1) in sketch.API:
        return f"{m.group(1)} needs {m.group(2)}; see the reference card"
    m = re.match(r"(\w+)(?:\.__init__)?\(\) takes \d+ positional arguments? but \d+ were given", message)
    if m and m.group(1) in sketch.API:
        return f"{m.group(1)} takes keyword arguments only ({m.group(1)}(width=..., ...)); see the reference card"
    return None


def kernel_refusal(exc: BaseException) -> str:
    """rc 3 for a kernel instrument that is not there (a unit still landing raises
    NotImplementedError): nothing in the script to fix, reported to the desk."""
    return _refusal_text("E-KERNEL", f"the kernel refused: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}",
                         "(nothing in the script to fix: an instrument is not built; reported to the desk)")


def kernel_failure(lap: "Lap", exc: BaseException) -> int:
    """rc 3 with a result.json for an exception the kernel raised outside the script (a
    gmsh error, a bug in an instrument): the desk still gets a result and a text that says
    nothing in the script is to be fixed. `traceback` carries the kernel's own frames here
    -- the one case they are the useful ones -- trimmed to the last 4000 characters."""
    first = str(exc).splitlines()[0] if str(exc) else ""
    text = _refusal_text("E-KERNEL", f"the kernel failed: {type(exc).__name__}{': ' + first if first else ''}",
                         "(nothing in the script to fix: an internal error; reported to the desk)")
    tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
    return lap.refuse("E-KERNEL", text, tb=tb, rc=EXIT_RAISED)


MAX_PRINT_LINES = 100
"""Script prints echoed under SCRIPT before '... N more lines': the print-back is for
numbers, not for a loop's log, and every line of it reaches the model."""

MAX_PRINT_CHARS = 8000


def clip_prints(printed: str) -> str:
    """The script's stdout as SCRIPT shows it: the first MAX_PRINT_LINES lines (and at most
    MAX_PRINT_CHARS characters), then one line saying how much was cut."""
    lines = printed.splitlines()
    kept: list[str] = []
    size = 0
    for line in lines:
        if len(kept) >= MAX_PRINT_LINES or size + len(line) > MAX_PRINT_CHARS:
            break
        kept.append(line)
        size += len(line) + 1
    if len(kept) == len(lines):
        return printed
    cut = len(lines) - len(kept)
    kept.append(f"... {cut} more line{'s' if cut != 1 else ''} not shown (the print-back is for numbers, not logs)")
    return "\n".join(kept) + "\n"


def script_failure(exc: BaseException, src: str) -> tuple[str, str, str]:
    """(code, text, traceback) for a script that raised: E-SCRIPT with the failing line
    quoted, the constructor hint when one applies, and the traceback trimmed to the
    script's own frames (never a kernel frame)."""
    if isinstance(exc, SyntaxError):
        lineno = exc.lineno or 1
        statement = (exc.text or _source_line(src, lineno)).strip()
        text = _refusal_text("E-SCRIPT", f"line {lineno}: SyntaxError: {exc.msg}", f"> {statement}",
                             "the script is Python: see the reference card for the API's spelling")
        return "E-SCRIPT", text, f"  line {lineno}\n    {statement}\nSyntaxError: {exc.msg}"
    frames = _script_frames(exc.__traceback__)
    lineno = frames[-1][0] if frames else None
    message = str(exc).strip() or type(exc).__name__
    message = re.sub(r"^(\w+)\.__init__\(\)", r"\1()", message)
    head = f"line {lineno}: " if lineno else ""
    if isinstance(exc, TypeError) and _constructor_hint(message):
        head += message
        hint = _constructor_hint(message)
    else:
        head += f"{type(exc).__name__}: {message}"
        if isinstance(exc, (ZeroDivisionError, OverflowError, ArithmeticError)):
            hint = "(the API does the arithmetic: footprints, landings and pitches are in the print-back)"
        elif isinstance(exc, NameError):
            hint = _bound_names()
        elif isinstance(exc, NotImplementedError):
            hint = "(nothing in the script to fix: the kernel refused; reported to the desk)"
        else:
            hint = "see the reference card"
    lines = []
    statement = _source_line(src, lineno)
    if statement:
        lines.append(f"> {statement}")
    lines.append(hint)
    for outer_line, name in reversed(frames[:-1]):
        lines.append(f"called from line {outer_line} in {name}: > {_source_line(src, outer_line)}")
    text = _refusal_text("E-SCRIPT", head, *lines)
    tb = "\n".join(f"  line {ln}, in {name}\n    {_source_line(src, ln)}" for ln, name in frames)
    tb += ("\n" if tb else "") + f"{type(exc).__name__}: {message}"
    return "E-SCRIPT", text, tb


# -- specs (the ops grammar) as plans -------------------------------------------------------


def _record_rules(rules: list[dict]) -> list[dict]:
    """Rules as the record writes them: `{"name", "at": "x:min" | "near:x,y"}` or
    `{"name", "box": [..]}`, whichever form they came in."""
    out = []
    for r in rules or []:
        item = {"name": r["name"]}
        if r.get("kind") and r["kind"] not in ("inlet", "outlet") or (r.get("kind") and r["kind"] != r["name"]
                                                                      and r["name"] not in ("inlet", "outlet")):
            item["kind"] = r["kind"]
        at = r.get("at")
        if isinstance(at, str):
            item["at"] = at
        elif isinstance(at, (list, tuple)) and len(at) == 3:
            axis, how, value = at
            if how == "near":
                item["at"] = f"near:{value[0]:g},{value[1]:g}"
            elif how in ("min", "max"):
                item["at"] = f"{'xy'[axis]}:{how}"
            else:
                item["at"] = f"{'xy'[axis]}:{value:g}"
        elif r.get("box") is not None:
            item["box"] = [float(v) for v in r["box"]]
        out.append(item)
    return out


def _mesh2d_rules(rules: list[dict]) -> list[dict]:
    """Rules in mesh2d's parsed form, whichever form the plan carries."""
    mesh2d = _toolbox.load("mesh2d")
    parsed = [r for r in rules or [] if isinstance(r.get("at"), tuple) or isinstance(r.get("box"), tuple)]
    if len(parsed) == len(rules or []):
        return list(rules or [])
    return mesh2d.parse_rules(_record_rules(rules))


def mirror_baked(ops: list[dict]) -> bool:
    """True when the fluid's last op is a `mirror` with no fuse after it (D30): the
    classifier is wrong on such a face, and `keep: true` fuses the mirror image with the
    original, which is the fuse that clears it."""
    if not ops:
        return False
    mesh2d = _toolbox.load("mesh2d")
    by_name = {op.get("name"): op for op in ops}
    op = by_name.get(mesh2d.body_name(ops))
    seen = set()
    while op is not None and op.get("name") not in seen:
        seen.add(op.get("name"))
        kind = op.get("op")
        if kind == "mirror":
            return not bool(op.get("keep", False))
        if kind in ("translate", "rotate", "copy"):
            op = by_name.get(op.get("target"))
            continue
        return False
    return False


def plan_from_spec(payload, scale: float | None = None) -> Plan:
    """A Plan for a spec in the ops grammar (a fixture, a record's ops): mesh2d validates
    it, the ops stay as written (the record must rebuild through `mesh2d.py --spec`), the
    patch rules become port intents by string, and every channel's width is a declared
    width. Units come from the scale (3.6)."""
    mesh2d = _toolbox.load("mesh2d")
    mesh2d.parse_spec(payload)
    raw_ops = payload["ops"] if isinstance(payload, dict) else payload
    raw_rules = (payload.get("patches") or []) if isinstance(payload, dict) else []
    if scale is None:
        scale = float(payload.get("scale", 1.0)) if isinstance(payload, dict) else 1.0
    units = UNITS_BY_SCALE.get(scale, "mm" if scale < 1 else "m")
    if mirror_baked(raw_ops):
        raise SketchError("E-MIRROR-BAKE", "spec", "the fluid's last op is `mirror`; the classifier is wrong on a mirrored face",
                          "fuse it with something after the mirror, or mirror the primitives' coordinates (the API does this itself)")
    features: dict[str, Solved] = {}
    instances: dict[str, dict] = {}
    widths: list[float] = []
    for op in raw_ops:
        kind, name = op.get("op"), op.get("name")
        params = {("centre" if k == "center" else k): v for k, v in op.items() if k not in ("op", "name")}   # D39
        if kind == "channel":
            widths.append(float(op["width"]))
            features[name] = Solved(kind="Passage", params=params, solved={}, ops=[name])
        elif kind == "repeat":
            step = tuple(op.get("step") or (0.0, 0.0))
            instances[name] = {"count": int(op["count"]), "step": list(step), "gap": None,
                               "gap_declared": False, "footprint": None}
            features[name] = Solved(kind="Row", params=params, solved={"pitch": math.hypot(*step)}, ops=[name])
        elif kind in ("rect", "disk", "annulus", "band", "polygon", "outline"):
            features[name] = Solved(kind=kind.capitalize(), params=params, solved=dict(params), ops=[name])
    ports = []
    for r in raw_rules:
        kind = r.get("kind") or (r["name"] if r["name"] in ("inlet", "outlet") else "wall")
        where = r["at"] if "at" in r else f"box:{','.join(f'{v:g}' for v in r['box'])}"
        ports.append(PortIntent(name=r["name"], kind=kind, edges=(where,)))
    return Plan(units=units, scale=scale, ops=list(raw_ops), rules=list(raw_rules), ports=ports,
                features=features, instances=instances, outlines={}, edge_targets={}, apart=[],
                declared_widths=widths, expected=(1, 1), notes=[], clip_boxes=[])


# -- the pipeline: build, measure, judge, comply -------------------------------------------------


@dataclass
class Analysis:
    face: int
    legs: dict
    checks: list
    m: Measurements
    findings: list[Finding]
    table: object | None
    curve_patch: dict[int, str]
    resolved_rules: list[dict]
    transitional: list[str] = field(default_factory=list)
    """Which instruments ran through mesh2d instead of the kernel (empty once every unit landed)."""


def build_face(gmsh, plan: Plan) -> tuple[int, dict, list[dict], bool]:
    """compile.build, or (transitional) mesh2d.build_face on the plan's ops at scale 1
    with mesh2d's own landing check; returns (face, legs, checks, transitional)."""
    try:
        face, legs, checks = compile.build(gmsh, plan)
        return face, legs, list(checks), False
    except NotImplementedError:
        pass
    mesh2d = _toolbox.load("mesh2d")
    ops, _ = mesh2d.parse_spec({"ops": plan.ops, "patches": []})
    legs: dict = {}
    checks: list = []
    try:
        face = mesh2d.build_face(gmsh, ops, scale=1.0, legs=legs, checks=checks)
    except SystemExit as exc:
        text = str(exc)
        code = "E-DISJOINT" if "separate faces" in text else "E-BUILD"
        raise SketchError(code, "fluid", text) from None
    mesh2d.landing_checks(gmsh, face, legs, 1.0, checks)
    return face, legs, checks, True


def _reference_w_min(gmsh, face: int, plan: Plan) -> float:
    if plan.declared_widths:
        return float(min(plan.declared_widths))
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, face)
    return max(min(x1 - x0, y1 - y0) / 10.0, 1e-9)


def _findings_from_checks(checks: list[dict], plan: Plan) -> list[Finding]:
    """mesh2d's check dicts as Findings (transitional: lint.from_legacy is U2's): the
    codes of section 5 by the sentence each check writes, every instance reference
    rewritten to `<row>[k]`, 0-based (D29)."""
    out = []
    for c in checks:
        what, level, where = c["what"], c["level"], c.get("where")
        m = re.match(r"(repeat|channel) '([^']+)': (.*)", what, re.S)
        subject_kind, subject_name, rest = (m.group(1), m.group(2), m.group(3)) if m else ("", "", what)
        subject = {"repeat": f"Row '{subject_name}'", "channel": f"Passage '{subject_name}'"}.get(subject_kind, "fluid")
        numbers: dict = {}
        if "overlap by" in what:
            code = "E-OVERLAP"
            area = re.search(r"overlap by ([\d.eE+-]+)", what)
            numbers["area"] = float(area.group(1)) if area else 0.0
        elif "touch (gap 0)" in what:
            code = "E-TOUCH"
            numbers["gap"] = 0.0
        elif "gap between copies" in what:
            code = "I-GAP"
            gap = re.search(r"gap between copies ([\d.eE+-]+)", what)
            numbers["gap"] = float(gap.group(1)) if gap else 0.0
        elif "lands at" in what:
            code = "E-LAND"
        elif "was read as a wall" in what:
            code = "E-PORT-OPEN"
        elif re.search(r"\d+ (inlet|outlet) edges", what):
            code = "E-PORT-COUNT"
        else:
            code = {"error": "E-CHECK", "warn": "W-CHECK", "info": "I-CHECK"}[level]
        if subject_kind == "repeat":
            rest = re.sub(r"copies (\d+) and (\d+)", lambda mm: f"{subject_name}[{int(mm.group(1)) - 1}] and "
                                                              f"{subject_name}[{int(mm.group(2)) - 1}]", rest)
        out.append(Finding(level=level, code=code, subject=subject, what=rest,
                           where=None if where is None else (float(where[0]), float(where[1])), numbers=numbers))
    return out


def legacy_analysis(gmsh, face: int, plan: Plan, legs: dict, checks: list[dict]) -> Analysis:
    """The measure and the judge through mesh2d by path (transitional: the kernel's
    `measure`/`lint` are U2's): mesh2d's classification for the patches, its port and
    count checks, its short edges as W-/E-SHORT-EDGE against a thirtieth and a third of
    the reference width, and a Measurements record of what it reads."""
    mesh2d = _toolbox.load("mesh2d")
    rules = _mesh2d_rules(plan.rules)
    w_min = _reference_w_min(gmsh, face, plan)
    built = mesh2d.measure2d(gmsh, face, short_below=w_min / 3.0)
    axis = mesh2d.longest_axis2d(built.extent)
    curve_patch = mesh2d.classify_curves(gmsh, built, axis, rules)
    checks = list(checks)
    mesh2d.port_checks(gmsh, built, curve_patch, legs, 1.0, checks)
    mesh2d.count_checks(gmsh, built, curve_patch, rules, checks)
    findings = _findings_from_checks(checks, plan)
    for c, length in built.short:
        mx, my = mesh2d.curve_midpoint(gmsh, c)
        error = length < w_min / 30.0
        findings.append(Finding(
            level="error" if error else "warn", code="E-SHORT-EDGE" if error else "W-SHORT-EDGE", subject="fluid",
            what=(f"edge of {length:.2g} at ({mx:.4g}, {my:.4g}): shorter than a "
                  f"{'thirtieth' if error else 'third'} of the {w_min:g} passage ({w_min / (30 if error else 3):.2g})"),
            fix="a sliver no cell can sit on" if error else
                "a corner poking through a wall, or a leg shorter than the channel is wide; the cells there will be poor",
            where=(mx, my), numbers={"length": float(length)}))
    order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: order.get(f.level, 3))
    patches: dict[str, dict] = {}
    for c in built.curves:
        name = curve_patch.get(c, "walls")
        entry = patches.setdefault(name, {"curves": [], "n": 0, "length": 0.0, "midpoints": []})
        entry["curves"].append(int(c))
        entry["n"] += 1
        entry["length"] += float(built.lengths[c])
        entry["midpoints"].append(list(mesh2d.curve_midpoint(gmsh, c)))
    shortest_tag = min(built.curves, key=lambda c: built.lengths[c]) if built.curves else None
    shortest = ((float(built.lengths[shortest_tag]), tuple(mesh2d.curve_midpoint(gmsh, shortest_tag)))
                if shortest_tag is not None else (0.0, (0.0, 0.0)))
    features = {name: {"kind": s.kind, **{k: v for k, v in (s.solved or {}).items()}} for name, s in plan.features.items()}
    features["fluid"] = {"extent_x": built.extent[0], "extent_y": built.extent[1], "area": built.area,
                         "islands": built.islands, "edges": len(built.curves), "narrowest_passage": None}
    rows: dict[str, RowMeasure] = {}
    gaps = {f.subject: f.numbers.get("gap") for f in findings if f.code == "I-GAP"}
    for name, info in plan.instances.items():
        step = info.get("step") or (0.0, 0.0)
        fp = info.get("footprint") or (0.0, 0.0)
        rows[name] = RowMeasure(name=name, count=int(info.get("count", 0)), pitch=math.hypot(*step),
                                footprint=(float(fp[0]), float(fp[1])), gap=info.get("gap"),
                                gap_measured=gaps.get(f"Row '{name}'"), anchors=[],
                                span=(float(built.bounds[0]), float(built.bounds[2])))
    channels = [op["name"] for op in plan.ops if op.get("op") == "channel"]
    m = Measurements(
        units=plan.units, scale=plan.scale, bounds=tuple(float(v) for v in built.bounds),
        extent=(float(built.extent[0]), float(built.extent[1])), area=float(built.area), islands=int(built.islands),
        n_curves=len(built.curves), shortest_edge=shortest, patches=patches, legs=legs, features=features,
        rows=rows, open_ends=[], junctions=[], vertices=[], passage=None, reference_width=w_min,
        reference_width_from=(f"declared by '{channels[0]}'" if plan.declared_widths and channels else "extent / 10"),
        holes=[], flow=None)
    return Analysis(face=face, legs=legs, checks=checks, m=m, findings=findings, table=None, curve_patch=curve_patch,
                    resolved_rules=_record_rules(plan.rules), transitional=["measure", "lint"])


def analyse(gmsh, plan: Plan, claim_set, face: int | None = None, legs: dict | None = None,
            checks: list | None = None) -> Analysis:
    """Build (unless a face is given), walk, resolve the ports, measure, judge, comply."""
    transitional: list[str] = []
    if face is None:
        face, legs, checks, fallback = build_face(gmsh, plan)
        if fallback:
            transitional.append("build")
    legs = legs or {}
    checks = list(checks or [])
    try:
        w_min = _reference_w_min(gmsh, face, plan)
        wk = measure.walk(gmsh, face, w_min)
        ends = measure.open_ends(gmsh, wk, w_min, None)
        curve_patch, resolved_rules, port_findings = measure.resolve_ports(gmsh, wk, plan, ends)
        m = measure.measure(gmsh, face, plan, wk, legs, curve_patch, claim_set)
        findings = list(lint.judge(gmsh, face, plan, wk, m, checks, claim_set))
        seen = {(f.code, f.subject, f.what) for f in findings}
        findings += [f for f in port_findings if (f.code, f.subject, f.what) not in seen]
        analysis = Analysis(face=face, legs=legs, checks=checks, m=m, findings=findings, table=None,
                            curve_patch=curve_patch, resolved_rules=resolved_rules, transitional=transitional)
    except NotImplementedError:
        analysis = legacy_analysis(gmsh, face, plan, legs, checks)
        analysis.transitional = transitional + analysis.transitional
    if claim_set is not None:
        try:
            analysis.table = claims.comply(claim_set, analysis.m, analysis.findings, plan)
        except NotImplementedError:
            analysis.table = None
            analysis.transitional.append("claims")
    return analysis


# -- the print-back, the verdict, the record -------------------------------------------------


def _table_lines(table) -> list[str]:
    try:
        return list(table.lines())
    except NotImplementedError:
        pass
    n = len(table.rows)
    lines = [f"{n} claims: {table.passed} pass, {table.failed} FAIL, {table.unmeasurable} not measurable"]
    for r in table.rows:
        verdict = "FAIL" if r.verdict == "fail" else r.verdict.replace("_", " ")
        lines.append(f"{r.id:<4} {r.says[:34]:<34} {r.measured[:44]:<44} {r.expected[:16]:<16} {verdict}")
        if r.detail:
            lines.append(f"     {r.detail}")
    return lines


def verdict(findings: list[Finding], table) -> tuple[str, bool]:
    """report.verdict, or (transitional) the same rule: ready only with no lint error and
    a table that is ok; 'not ready: no claims' without a table."""
    try:
        return report.verdict(findings, table)
    except NotImplementedError:
        pass
    parts = []
    codes = [f.code for f in findings if f.level == "error"]
    if codes:
        parts.append("LINT " + ", ".join(dict.fromkeys(codes)))
    if table is None:
        parts.append("no claims")
    else:
        failing = table.failing_ids()
        if failing:
            ids = ", ".join(failing)
            parts.append(f"claims {ids} FAIL   (COMMIT disagrees: {ids} records it)")
        elif not table.ok():
            parts.append("no checkable claim was recorded")
    if not parts:
        return "ready to COMMIT", True
    return "not ready: " + "; ".join(parts), False


def _feature_summary(s: Solved) -> str:
    source = s.solved or s.params or {}
    bits = []
    for k, v in source.items():
        if isinstance(v, (dict, list, tuple)) and len(str(v)) > 40:
            continue
        if isinstance(v, float):
            bits.append(f"{k} {v:.4g}")
        else:
            bits.append(f"{k} {v}")
    text = ", ".join(bits)
    return text if len(text) <= 110 else text[:107] + "..."


def render_report(printed: str, notes: list[str], seconds: float, findings: list[Finding], plan: Plan | None,
                  m: Measurements | None, legs: dict, table, reference, refusal: str | None = None) -> str:
    """report.text, or (transitional) the section-6.1 layout for what this unit has."""
    try:
        return report.text(printed, notes, seconds, findings, plan, m, legs, table, reference, refusal=refusal)
    except NotImplementedError:
        pass
    printed_lines = [ln for ln in (printed or "").splitlines() if ln.strip()]
    counts = []
    counts.append(f"{len(printed_lines)} print{'s' if len(printed_lines) != 1 else ''}" if printed_lines else "no prints")
    if notes:
        counts.append(f"{len(notes)} note{'s' if len(notes) != 1 else ''}")
    out = [f"SCRIPT     ran in {seconds:.1f} s; {', '.join(counts)}"]
    out += [f"           > {ln}" for ln in printed_lines]
    out += [f"           note: {n}" for n in notes]
    if m is None:
        if refusal:
            out += refusal.splitlines()
        return "\n".join(out)
    errors = [f for f in findings if f.level == "error"]
    warns = [f for f in findings if f.level == "warn"]
    if errors or warns:
        head = ", ".join(x for x in (f"{len(errors)} error{'s' if len(errors) != 1 else ''}" if errors else "",
                                     f"{len(warns)} warning{'s' if len(warns) != 1 else ''}" if warns else "") if x)
        out.append(f"LINT       {head}" + (" (errors block)" if errors else ""))
    else:
        out.append("LINT       clean")
    order = {"error": 0, "warn": 1, "info": 2}
    for f in sorted(findings, key=lambda f: order.get(f.level, 3)):
        out += [f"           {ln}" for ln in f.text().splitlines()]
    if plan is not None and plan.features:
        first = True
        for name, s in plan.features.items():
            lead = "FEATURES   " if first else "           "
            out.append(f"{lead}{name:<9} {s.kind:<10} {_feature_summary(s)}")
            first = False
    if legs:
        mesh2d = _toolbox.load("mesh2d")
        axis = 0 if m.extent[0] >= m.extent[1] else 1
        first = True
        for ln in mesh2d.leg_lines(legs, axis):
            out.append(("LEGS       " if first else "           ") + ln)
            first = False
    w, h = m.extent
    scale = m.scale or 1.0
    unit = m.units
    passage = f"narrowest passage {m.passage.min:.4g}   " if m.passage else ""
    out.append(f"MEASURED   extent {w:.4g} x {h:.4g} {unit} ({w * scale:.4g} x {h * scale:.4g} m)   area {m.area:.4g} {unit}2   "
               f"islands {m.islands}   {passage}{m.n_curves} edges, shortest {m.shortest_edge[0]:.4g} at "
               f"({m.shortest_edge[1][0]:.4g}, {m.shortest_edge[1][1]:.4g})")
    cells = []
    for name, info in m.patches.items():
        n = info.get("n", len(info.get("curves", [])))
        mids = info.get("midpoints") or []
        where = f" at ({mids[0][0]:.4g}, {mids[0][1]:.4g})" if len(mids) == 1 else ""
        cells.append(f"{name} ({n} edge{'s' if n != 1 else ''}, {info.get('length', 0.0):.4g}){where}")
    out.append("PATCHES    " + "     ".join(cells))
    if table is not None:
        first = True
        for ln in _table_lines(table):
            out.append(("CLAIMS     " if first else "           ") + ln)
            first = False
    elif refusal:
        out.append("CLAIMS     not evaluated: the sketch did not build")
    else:
        out.append("CLAIMS     none (no claims file)")
    if reference is not None:
        hd = f"Hausdorff {reference.hausdorff:.4g}" if reference.hausdorff is not None else "Hausdorff not measured"
        out.append(f"REFERENCE  library {reference.entry} (preset {reference.preset}, approved {reference.approved_at}): {hd}")
    else:
        out.append("REFERENCE  none (no approved library entry matches)")
    if refusal:
        # the sketch did not build: the claims were never evaluated, so the verdict names
        # the refusal alone (section 6.1), never "no claims"
        codes = ", ".join(dict.fromkeys(f.code for f in findings if f.level == "error"))
        out.append(f"VERDICT    not ready: LINT {codes}")
    else:
        out.append(f"VERDICT    {verdict(findings, table)[0]}")
    return "\n".join(out)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def build_record(sk, plan: Plan, script: str, m: Measurements, findings: list[Finding], table,
                 resolved_rules: list[dict], claims_payload: dict | None, gmsh=None) -> dict:
    """compile.record, or (transitional, and for a spec with no Sketch) the record of
    section 4.2 assembled here."""
    if sk is not None:
        try:
            return compile.record(sk, plan, script, m, findings, table, resolved_rules)
        except NotImplementedError:
            pass
    version = ""
    if gmsh is not None:
        try:
            version = gmsh.option.getString("General.Version")
        except Exception:  # noqa: BLE001 - the version is decoration
            version = ""
    return _json_safe({
        "format": "openreynolds.geometry/1", "units": plan.units, "scale": plan.scale,
        "ops": plan.ops, "patches": resolved_rules or _record_rules(plan.rules),
        "ports": [p.as_dict() for p in plan.ports], "script": script,
        "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest() if script else "",
        "features": {k: v.as_dict() for k, v in plan.features.items()}, "instances": plan.instances,
        "apart": [list(a) for a in plan.apart], "claims": claims_payload,
        "compliance": None if table is None else table.as_dict(),
        "lint": [f.as_dict() for f in findings], "measurements": m.as_dict(),
        "built_with": {"gmsh": version, "at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
    })


# -- the reference gate ---------------------------------------------------------------------


def reference_gate(name: str | None):
    """None, or a ReferenceMatch for an APPROVED library entry (D32); refuses with
    E-REFERENCE-UNAPPROVED otherwise. A library that is not built yet approves nothing."""
    if not name:
        return None
    refusal = _refusal_text("E-REFERENCE-UNAPPROVED",
                            f"library entry '{name}' is not approved; nothing is shown as a reference",
                            f"until a person approves it (cli golden --approve {name} --by <name>)")
    try:
        entry = library.get(name)
        approval = library.approval(entry)
        if approval is None or approval.get("source_sha") != library.source_sha(entry):
            raise Refusal("E-REFERENCE-UNAPPROVED", refusal)
        golden = json.loads((library.GOLDEN / f"{entry.name}.json").read_text(encoding="utf-8"))
    except (NotImplementedError, KeyError, FileNotFoundError, OSError, ValueError):
        raise Refusal("E-REFERENCE-UNAPPROVED", refusal) from None
    return library.ReferenceMatch(entry=entry.name, preset=str(golden.get("preset") or approval.get("preset") or ""),
                                  approved_at=str(approval.get("at", "")),
                                  outline=[[(float(p[0]), float(p[1])) for p in loop] for loop in golden.get("outline", [])],
                                  measurements=dict(golden.get("measurements") or {}), hausdorff=None)


class Refusal(Exception):
    """A refusal that is not the API's (the reference gate, a malformed claims file):
    rc 5 with the text as written."""

    def __init__(self, code: str, text: str):
        self.code = code
        self.text = text
        super().__init__(text)


def load_claims(path: Path | None) -> tuple[object | None, dict | None, list[str]]:
    """(ClaimSet, the payload, the parse notes); a claims lap that the parser (U3) does
    not yet judge leaves the set None with a note, so the lap still builds."""
    if path is None:
        return None, None, []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal("E-CLAIMS", _refusal_text("E-CLAIMS", f"claims file {path}: {exc}",
                                                "the claims lap writes a JSON object in the claims schema")) from None
    try:
        claim_set, lines = claims.parse(payload)
    except NotImplementedError:
        return None, payload, ["claims not evaluated: the compliance judge is not built yet"]
    except claims.ClaimsError as exc:
        raise Refusal("E-CLAIMS", _refusal_text("E-CLAIMS", f"claims file: {exc}",
                                                "every claim needs id, kind and says; see the claims schema")) from None
    return claim_set, payload, list(lines)


# -- the lap ------------------------------------------------------------------------------


@dataclass
class Lap:
    """What one child run writes: result.json (4.3), report.txt, record.json, preview.png."""

    work: Path
    preview_path: Path | None
    started: float
    reference: object | None = None
    printed: str = ""
    notes: list[str] = field(default_factory=list)
    claims_payload: dict | None = None
    script: str = ""

    def seconds(self) -> float:
        return time.monotonic() - self.started

    def write(self, *, rc: int, record: dict | None, m: Measurements | None, findings: list[Finding],
              table, report_text: str, verdict_text: str, ready: bool, drawn: bool, error: str | None,
              tb: str | None, code: str | None, hausdorff: float | None = None) -> int:
        result = {
            "ok": rc == 0, "rc": rc, "seconds": round(self.seconds(), 3),
            "record": record, "measurements": None if m is None else _json_safe(m.as_dict()),
            "lint": [f.as_dict() for f in findings], "compliance": None if table is None else table.as_dict(),
            "report": report_text, "verdict": {"text": verdict_text, "ready": ready},
            "preview": (self.preview_path.name if drawn and self.preview_path is not None else None),
            "reference": (None if self.reference is None else
                          {"entry": self.reference.entry, "preset": self.reference.preset, "hausdorff": hausdorff}),
            "error": error, "traceback": tb, "code": code,
        }
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "result.json").write_text(json.dumps(_json_safe(result), indent=1), encoding="utf-8")
        (self.work / "report.txt").write_text(report_text, encoding="utf-8")
        if record is not None:
            (self.work / "record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(report_text)
        return rc

    def refuse(self, code: str, text: str, tb: str | None = None, rc: int = EXIT_REFUSED) -> int:
        report_text = render_report(self.printed, self.notes, self.seconds(), [], None, None, {}, None,
                                    self.reference, refusal=text)
        return self.write(rc=rc, record=None, m=None, findings=[], table=None, report_text=report_text,
                          verdict_text=f"not ready: {code}", ready=False, drawn=False,
                          error=text.splitlines()[0] if text else code, tb=tb, code=code)


def run_plan(gmsh, lap: Lap, plan: Plan, sk, claim_set, *, refusal: SketchError | None = None) -> int:
    """Build, analyse, draw, record and report one plan; with `refusal` (the rc-5 partial
    path) the findings are the refusal alone, CLAIMS are not evaluated and the exit is 5."""
    analysis = analyse(gmsh, plan, None if refusal else claim_set)
    m, table = analysis.m, analysis.table
    if refusal is not None:
        findings = [Finding(level="error", code=refusal.code, subject=refusal.feature, what=refusal.what,
                            fix="\n          ".join(refusal.lines), numbers=_json_safe(refusal.numbers))]
        table = None
    else:
        findings = analysis.findings
    hausdorff = None
    drawn = False
    if lap.preview_path is not None:
        try:
            if lap.reference is not None:
                hausdorff = preview.reference_distance(gmsh, analysis.face, m, lap.reference)
                lap.reference.hausdorff = hausdorff
            preview.draw(gmsh, analysis.face, plan, m, findings, table, lap.preview_path, lap.reference)
            drawn = lap.preview_path.exists()
        except Exception as exc:  # noqa: BLE001 - gmsh raises a plain Exception; the picture is never worth the lap
            # the build, the measurements and the print-back stand on their own: a
            # triangulation the mesher refuses or a missing matplotlib costs the picture only
            lap.notes.append(f"no picture: {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}".rstrip(": "))
            with contextlib.suppress(OSError):
                lap.preview_path.unlink(missing_ok=True)
    for note in analysis.transitional:
        lap.notes.append(f"{note}: through mesh2d until its unit lands")
    record = build_record(sk, plan, lap.script, m, findings, table, analysis.resolved_rules, lap.claims_payload, gmsh)
    report_text = render_report(lap.printed, lap.notes, lap.seconds(), findings, plan, m, analysis.legs, table,
                                lap.reference, refusal=None if refusal is None else str(refusal))
    if refusal is not None:
        verdict_text, ready = f"not ready: LINT {refusal.code}", False
        return lap.write(rc=EXIT_REFUSED, record=record, m=m, findings=findings, table=None, report_text=report_text,
                         verdict_text=verdict_text, ready=False, drawn=drawn, error=str(refusal).splitlines()[0],
                         tb=None, code=refusal.code, hausdorff=hausdorff)
    verdict_text, ready = verdict(findings, table)
    rc = EXIT_LINT if lint.has_errors(findings) else EXIT_OK
    return lap.write(rc=rc, record=record, m=m, findings=findings, table=table, report_text=report_text,
                     verdict_text=verdict_text, ready=ready, drawn=drawn, error=None, tb=None, code=None,
                     hausdorff=hausdorff)


def refused_with_partial(gmsh, lap: Lap, exc: SketchError) -> int:
    """The rc-5 path (3.15): with a `partial`, the single instance is planned by
    `compile.plan_feature`, built, measured and drawn with the refusal under LINT; without
    one, or when the partial cannot be built either, the refusal alone."""
    if exc.partial is not None:
        try:
            plan_p = compile.plan_feature(exc.partial, gmsh)
            return run_plan(gmsh, lap, plan_p, None, None, refusal=exc)
        except (NotImplementedError, SketchError, SystemExit, ValueError, KeyError, TypeError) as inner:
            lap.notes.append(f"the refused item could not be drawn alone: {str(inner).splitlines()[0][:120]}")
    return lap.refuse(exc.code, str(exc))


@contextlib.contextmanager
def _gmsh_session(name: str):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def exec_script(src: str, work: Path, claims_path: Path | None = None, reference: str | None = None,
                preview_path: Path | None = None, timeout_s: float = runner.RUN_TIMEOUT_S) -> int:
    """`build --script`: the namespace is {"__name__": "__sketch__", **sketch.API} (the API
    is pre-bound: no import of the sketch module is ever needed, and both `from
    geometry.sketch import ...` and `from openreynolds.geometry.sketch import ...` are
    refused with the E-IMPORT text "the API needs no import"). The import guard is
    installed on `builtins.__import__` immediately before `exec(compile(src, "script.py",
    "exec"), ns)` and removed in a `finally`; it refuses only when the calling frame's
    globals carry `__name__ == "__sketch__"` and the name is not in
    `runner.ALLOWED_IMPORTS`. On Linux RLIMIT_AS 2 GB and RLIMIT_CPU = timeout + 5.
    Exactly one `Sketch` must exist (E-NO-SKETCH / E-MANY-SKETCHES); a traceback is
    trimmed to the frames whose filename is `script.py`, the failing line quoted, and a
    `TypeError` on a known constructor gets the signature hint (E-SCRIPT text). The rc-5
    path with a `partial` builds, measures and draws the single instance (3.15). Returns
    the exit code and writes result.json, report.txt, record.json, preview.png into `work`.
    The rlimits are set by `cmd_build`, the child's entry point, not here: a test that
    calls this in-process must not cap its own interpreter."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lap = Lap(work=work, preview_path=Path(preview_path) if preview_path else work / "preview.png",
              started=time.monotonic(), script=src)
    try:
        lap.reference = reference_gate(reference)
        claim_set, lap.claims_payload, claim_notes = load_claims(claims_path)
    except Refusal as exc:
        return lap.refuse(exc.code, exc.text)
    lap.notes += claim_notes

    with contextlib.suppress(NotImplementedError):
        sketch.Sketch._reset()
    captured = io.StringIO()
    try:
        code = builtins.compile(src, "script.py", "exec")
        ns = {"__name__": "__sketch__", "__builtins__": builtins, **sketch.API}
        with contextlib.redirect_stdout(captured), import_guard():
            exec(code, ns)   # noqa: S102 - the script is the model's reply, run in its own namespace
    except ImportRefused as exc:
        lap.printed = clip_prints(captured.getvalue())
        code_, text = import_refusal(exc, src)
        return lap.refuse(code_, text)
    except SketchError as exc:
        lap.printed = clip_prints(captured.getvalue())
        try:
            with _gmsh_session("geometry") as gmsh:
                return refused_with_partial(gmsh, lap, exc)
        except Exception as inner:  # noqa: BLE001 - the kernel, not the script: rc 3 with a result.json
            return kernel_failure(lap, inner)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - every failure of the script is rc 3 with its own frames
        lap.printed = clip_prints(captured.getvalue())
        code_, text, tb = script_failure(exc, src)
        return lap.refuse(code_, text, tb=tb, rc=EXIT_RAISED)
    lap.printed = clip_prints(captured.getvalue())

    try:
        sk = sketch.Sketch.current()
        lap.notes += list(getattr(sk, "notes", []) or [])
    except SketchError as exc:
        return lap.refuse(exc.code, str(exc))
    except NotImplementedError as exc:
        return lap.refuse("E-KERNEL", kernel_refusal(exc), rc=EXIT_RAISED)
    try:
        with _gmsh_session("geometry") as gmsh:
            try:
                plan = compile.plan(sk, claim_set, gmsh)
            except SketchError as exc:
                return refused_with_partial(gmsh, lap, exc)
            except NotImplementedError as exc:
                return lap.refuse("E-KERNEL", kernel_refusal(exc), rc=EXIT_RAISED)
            try:
                return run_plan(gmsh, lap, plan, sk, claim_set)
            except SketchError as exc:
                if exc.code == "E-KERNEL-STATE":
                    return lap.refuse(exc.code, str(exc), rc=EXIT_RAISED)
                return lap.refuse(exc.code, str(exc))
            except NotImplementedError as exc:
                return lap.refuse("E-KERNEL", kernel_refusal(exc), rc=EXIT_RAISED)
    except Exception as exc:  # noqa: BLE001 - whatever else the kernel raised: the lap still answers, rc 3
        return kernel_failure(lap, exc)


def exec_spec(payload, work: Path, scale: float | None = None, claims_path: Path | None = None,
              preview_path: Path | None = None, reference: str | None = None, script: str = "") -> int:
    """`build --spec` / `check --spec` / `preview --spec`: a spec in the ops grammar through
    the same lap as a script, with the record's ops as written."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lap = Lap(work=work, preview_path=Path(preview_path) if preview_path else None, started=time.monotonic(),
              script=script)
    try:
        lap.reference = reference_gate(reference)
        claim_set, lap.claims_payload, claim_notes = load_claims(claims_path)
        plan = plan_from_spec(payload, scale)
    except Refusal as exc:
        return lap.refuse(exc.code, exc.text)
    except SketchError as exc:
        return lap.refuse(exc.code, str(exc))
    except SystemExit as exc:
        return lap.refuse("E-SPEC", _refusal_text("E-SPEC", f"spec: {exc}", "see the ops grammar in mesh2d.py"))
    lap.notes += claim_notes
    try:
        with _gmsh_session("geometry") as gmsh:
            try:
                return run_plan(gmsh, lap, plan, None, claim_set)
            except SketchError as exc:
                return lap.refuse(exc.code, str(exc), rc=EXIT_RAISED if exc.code == "E-KERNEL-STATE" else EXIT_REFUSED)
    except Exception as exc:  # noqa: BLE001 - the kernel, not the spec: rc 3 with a result.json
        return kernel_failure(lap, exc)


# -- the subcommands ------------------------------------------------------------------------


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record_to_spec(record: dict, work: Path) -> tuple[dict, float, Path | None]:
    """A record (4.2) as the spec mesh2d reads plus its claims as a file for the lap,
    written into `work` (a temporary directory the caller removes) so nothing is left
    behind in the system temp."""
    spec = {"ops": record["ops"], "patches": record.get("patches") or [], "scale": record.get("scale", 1.0)}
    claims_file = None
    if record.get("claims"):
        claims_file = Path(work) / "claims.json"
        claims_file.write_text(json.dumps(record["claims"]), encoding="utf-8")
    return spec, float(record.get("scale", 1.0)), claims_file


def cmd_build(args) -> int:
    out = Path(args.out)
    preview_path = Path(args.preview) if args.preview else out / "preview.png"
    if args.script:
        apply_rlimits(args.timeout)   # this process is the child; a test calling exec_script in-process is not
        src = Path(args.script).read_text(encoding="utf-8")
        return exec_script(src, out, Path(args.claims) if args.claims else None, args.reference, preview_path,
                           timeout_s=args.timeout)
    payload = _read_json(args.spec)
    return exec_spec(payload, out, args.scale, Path(args.claims) if args.claims else None, preview_path,
                     args.reference)


def _plain_preview(payload, scale: float, out: Path) -> int:
    """`preview --plain`: mesh2d's own picture and report lines, by path."""
    mesh2d = _toolbox.load("mesh2d")
    ops, rules = mesh2d.parse_spec(payload)
    with _gmsh_session("plain") as gmsh:
        legs: dict = {}
        checks: list = []
        face = mesh2d.build_face(gmsh, ops, scale=1.0, legs=legs, checks=checks)
        narrow = mesh2d.narrowest_channel(ops)
        built = mesh2d.measure2d(gmsh, face, short_below=narrow / 3.0 if narrow else None)
        axis = mesh2d.longest_axis2d(built.extent)
        curve_patch = mesh2d.classify_curves(gmsh, built, axis, rules)
        mesh2d.landing_checks(gmsh, face, legs, 1.0, checks)
        mesh2d.port_checks(gmsh, built, curve_patch, legs, 1.0, checks)
        mesh2d.count_checks(gmsh, built, curve_patch, rules, checks)
        caption = mesh2d.report_lines(built, curve_patch, "spec", legs, checks, axis, gmsh)
        marks = [(*c["where"], c["level"]) for c in checks if c.get("where") and c["level"] != "info"]
        marks += [(*mesh2d.curve_midpoint(gmsh, c), "warn") for c, _ in built.short]
        preview.draw_plain(gmsh, built, curve_patch, out, "spec", caption, marks)
    for line in caption:
        print(line)
    return EXIT_LINT if mesh2d.has_errors(checks) else EXIT_OK


def cmd_preview(args) -> int:
    out = Path(args.out)
    with tempfile.TemporaryDirectory(prefix="geometry-preview-") as tmp:
        work = Path(tmp)
        if args.script:
            src = Path(args.script).read_text(encoding="utf-8")
            return exec_script(src, work, Path(args.claims) if args.claims else None, None, out)
        if args.library:
            entry = library.get(args.library)
            params = {}
            for item in args.param or []:
                key, _, value = item.partition("=")
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
            sk = entry.build(args.preset, **params)
            lap = Lap(work=work, preview_path=out, started=time.monotonic(), script="")
            with _gmsh_session("library") as gmsh:
                try:
                    plan = compile.plan(sk, None, gmsh)
                    return run_plan(gmsh, lap, plan, sk, None)
                except SketchError as exc:
                    return refused_with_partial(gmsh, lap, exc)
        if args.record:
            payload, scale, claims_file = _record_to_spec(_read_json(args.record), work)
        else:
            payload = _read_json(args.spec)
            scale = args.scale if args.scale is not None else (payload.get("scale", 1.0) if isinstance(payload, dict) else 1.0)
            claims_file = Path(args.claims) if args.claims else None
        if args.plain:
            return _plain_preview(payload, scale, out)
        return exec_spec(payload, work, scale, claims_file, out)


def cmd_check(args) -> int:
    with tempfile.TemporaryDirectory(prefix="geometry-check-") as tmp:
        work = Path(tmp)
        if args.script:
            src = Path(args.script).read_text(encoding="utf-8")
            return exec_script(src, work, Path(args.claims) if args.claims else None, None, work / "preview.png")
        if args.record:
            payload, scale, claims_file = _record_to_spec(_read_json(args.record), work)
            if args.claims:
                claims_file = Path(args.claims)
        else:
            payload = _read_json(args.spec)
            scale = args.scale
            claims_file = Path(args.claims) if args.claims else None
        return exec_spec(payload, work, scale, claims_file, None)


def cmd_reference(args) -> int:
    card = sketch.api_summary()
    if args.write:
        REFERENCE_MD.write_text(card if card.endswith("\n") else card + "\n", encoding="utf-8")
        print(f"wrote {REFERENCE_MD}")
        return EXIT_OK
    print(card)
    return EXIT_OK


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _span(golden: dict) -> float:
    extent = (golden.get("measurements") or {}).get("extent") or [1.0, 1.0]
    return float(max(extent))


def cmd_golden(args) -> int:
    approvals_path = library.GOLDEN / "APPROVALS.json"
    approvals = _read_json(approvals_path) if approvals_path.exists() else {}
    if args.approve:
        entry = library.get(args.approve)
        golden_path = library.GOLDEN / f"{entry.name}.json"
        if not golden_path.exists():
            print(f"no golden for {entry.name}: cli golden --regenerate {entry.name} first")
            return 1
        golden = _read_json(golden_path)
        if golden.get("source_sha") != library.source_sha(entry):
            print(f"{entry.name}: the golden was made from another source; regenerate it before approving")
            return 1
        approvals[entry.name] = {"preset": golden.get("preset"), "by": args.by,
                                 "at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "source_sha": golden["source_sha"],
                                 "measurements_sha": _sha(golden.get("measurements")),
                                 "outline_sha": _sha(golden.get("outline"))}
        approvals_path.parent.mkdir(parents=True, exist_ok=True)
        approvals_path.write_text(json.dumps(approvals, indent=1), encoding="utf-8")
        print(f"approved {entry.name} ({golden.get('preset')}) by {args.by}")
        return EXIT_OK
    if args.regenerate is not None:
        targets = [library.get(args.regenerate)] if args.regenerate else library.entries()
        for entry in targets:
            golden = library.regenerate(entry, into=library.GOLDEN)
            print(f"regenerated {entry.name} ({golden.get('preset')}): {library.GOLDEN / (entry.name + '.json')}")
        return EXIT_OK
    if args.check:
        failed = 0
        for entry in library.entries():
            golden_path = library.GOLDEN / f"{entry.name}.json"
            if not golden_path.exists():
                print(f"{entry.name}: no golden")
                failed += 1
                continue
            golden = _read_json(golden_path)
            approval = approvals.get(entry.name)
            sha = library.source_sha(entry)
            if approval is None:
                print(f"{entry.name}: not approved")
                failed += 1
                continue
            if approval.get("source_sha") != sha:
                print(f"{entry.name}: the source changed since its approval (sha {sha[:12]} vs approved {approval.get('source_sha', '')[:12]})")
                failed += 1
                continue
            with tempfile.TemporaryDirectory(prefix="golden-") as tmp:
                rebuilt = library.regenerate(entry, preset=golden.get("preset"), into=Path(tmp))
            distance = library.hausdorff(rebuilt["outline"], golden["outline"])
            agree = library.measurements_agree(rebuilt["measurements"], golden["measurements"], 1e-6)
            if distance >= 1e-6 * _span(golden) or not agree:
                print(f"{entry.name}: differs from its golden (Hausdorff {distance:.3g}, measurements {'agree' if agree else 'differ'})")
                failed += 1
            else:
                print(f"{entry.name}: matches its golden")
        return 1 if failed else EXIT_OK
    print("golden: one of --regenerate [NAME], --check, --approve NAME --by NAME")
    return 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="geometry", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    b = sub.add_parser("build", help="the child of runner.run_script")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", type=Path)
    src.add_argument("--spec", type=Path)
    b.add_argument("--claims", type=Path, default=None)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--preview", type=Path, default=None)
    b.add_argument("--reference", default=None)
    b.add_argument("--scale", type=float, default=None, help="--spec: the spec's units per metre (0.001 for mm)")
    b.add_argument("--timeout", type=float, default=runner.RUN_TIMEOUT_S)

    p = sub.add_parser("preview", help="draw a script, a record, a library entry or a spec")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", type=Path)
    src.add_argument("--record", type=Path)
    src.add_argument("--library", default=None)
    src.add_argument("--spec", type=Path)
    p.add_argument("--param", action="append", default=[], help="--library: k=v")
    p.add_argument("--preset", default=None)
    p.add_argument("--claims", type=Path, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plain", action="store_true", help="mesh2d's own picture")

    c = sub.add_parser("check", help="lint + measure + compliance; exit 2 on errors")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", type=Path)
    src.add_argument("--script", type=Path)
    src.add_argument("--spec", type=Path)
    c.add_argument("--claims", type=Path, default=None)
    c.add_argument("--scale", type=float, default=None)

    r = sub.add_parser("reference", help="print the API card")
    r.add_argument("--write", action="store_true")

    g = sub.add_parser("golden", help="regenerate, check or approve the library goldens")
    g.add_argument("--regenerate", nargs="?", const="", default=None, metavar="NAME")
    g.add_argument("--check", action="store_true")
    g.add_argument("--approve", default=None, metavar="NAME")
    g.add_argument("--by", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    """The subcommands above; returns the exit code."""
    ap = parser()
    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help()
        return 1
    if args.command == "golden" and args.approve and not args.by:
        ap.error("--approve NAME needs --by <who>")
    handler = {"build": cmd_build, "preview": cmd_preview, "check": cmd_check,
               "reference": cmd_reference, "golden": cmd_golden}[args.command]
    try:
        return handler(args)
    except NotImplementedError as exc:
        print(f"not available yet: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())   # `python -m openreynolds.geometry.cli`: the package is known, the shim above did not run
