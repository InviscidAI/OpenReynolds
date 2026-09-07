"""The geometry desk: a second, specialist loop that authors a shape, draws it, measures
it and commits it -- run where the model is, and handed back to the main agent as one
tool result: the case on the workspace, the picture, the report, the compliance table
against the request's own claims, and the mesh fitness table.

Why a separate loop. The same model one-shots a Tesla-valve mesh in a chat app and, in
this harness, took 66 turns for the wrong shape: every step of authoring a geometry was
a remote round trip (write a script, run it, read the error), and there was no way to
see the shape before meshing. Measured on the fix (`qa-runs/RESULTS-mesh2d.md`): with
`mesh2d.py` in the toolbox the main agent still spends four preview laps of its own
turns getting a shape right, because a spec written first shot is valid five times in
six and correct one time in six -- the model does not reason about where an arc ends or
whether neighbours overlap until it sees the picture and the numbers. Those laps are
the work; this loop does them locally (the kernel in a child interpreter, ~1-2 s a lap,
no sandbox in the path), so from the main agent's point of view a shape is one call.

Why claims come first (Phase 1, DESIGN.md section 1). Phase 0 committed a valve whose
return legs ran WITH the flow because the only judge was the model reading its own
picture. Now the model writes the request's checkable claims before any geometry, the
tool fills a compliance table against them each lap, and a COMMIT is accepted only when
the table has no failing row that the model has not named as a disagreement (D11) and
the lint has no error. Nothing here trusts a verdict the model asserts: the table is
computed by the child, the desk reads its rows.

Why this does not break the free-will contract: as with the front desk (`desk.py`),
the contract governs what may influence the *main model's* decisions. This loop is a
tool with a narrow job; its brief is explicit about its steps, lives here beside it,
and is pinned by `tests/test_geometry_agent.py` rather than by the prompt tests. The
main prompt gains one descriptive sentence naming the tool and nothing that tells the
main model when to use it.

What it never does: run a solver, touch the main agent's thread or files. It writes a
case directory, runs Allmesh + checkMesh + one render on the instance, and returns.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .. import images, trace
from ..llm import ProviderError, make_provider
from ..llm.base import Listener
from . import case as case_mod
from . import claims as claims_mod
from . import fitness as fitness_mod
from . import library, runner
from .claims import ClaimSet, ComplianceRow, ComplianceTable
from .fitness import FitnessTable, Unmeasured
from .lint import Finding
from .measure import Measurements
from .runner import RUN_TIMEOUT_S, RunOutcome
from .strategy import Sizes

TOOLBOX = Path(__file__).resolve().parents[1] / "toolbox"

TOOLBOX_DEST = "/work/.toolbox"
"""Where the toolbox is refreshed to on the instance (cli.py), for the finish step."""

FINISH_TIMEOUT_S = 240
"""gmshToFoam, checkMesh and one mesh render on a one-cell 2D case take seconds; a 3D
case with a million tets takes about a minute. Past this the finish reports it did not
complete and the main agent has Allmesh to run itself."""

MAX_LAPS = 8
"""Script laps (a reply and its run) before the last clean script is committed. The
claims lap is lap 0 and is not counted (D15): it is a separate model call whose cost
the benchmark records on its own."""

MAX_SECONDS = 600.0
"""Wall-clock budget over the claims lap and the script laps; the commit and the finish
are not counted.

Measured on the serpentine run (qa-runs/RESULTS-mesh2d.md): a lap is 30-40 s of model
time at medium effort and 1-2 s of build, and a 90 s budget cut one call at three laps
with every edge still classified `inlet`. At high effort, which is what the desk now
reasons at, the first re-measured valve (2026-09-07, study 20260907-003655-6aab) ran
into a 240 s budget at 246 s: the cap, not agreement, ended it. Ten minutes is the
ceiling the revamp plan sets; the laps end on agreement well before it when the
checks pass, and the main agent sees "still running" meanwhile."""

MAX_REPLY_TOKENS = 16_000
"""The premise gate saw a 2k reply spent entirely on thinking, with no spec in it."""

COMMIT = "COMMIT"

REFERENCE_CARD = Path(__file__).resolve().parent / "reference.md"
"""The API card the claims lap sends; generated from `sketch.api_summary()` and pinned
equal to it by a test, so reading the file is reading the code."""

_BRIEF_TEMPLATE = """\
You are the geometry desk: an engineer with instruments who authors a 2D shape for a CFD case
from a request in words. You work in laps with a tool that builds what you write, lints it,
measures it, draws it, and fills a compliance table against the claims you wrote first.

First, claims. Reply with ONLY a JSON object in the claims schema given below: the unit the
request uses; every length, angle, radius and count as a measure or count claim with the
request's own words in "says"; every qualitative clause as a predicate from the list, or a
not_measurable claim when no predicate fits (that is the honest answer, not a failure); the
inlet and the outlet as patch claims saying where they are; and a report claim for any
number the request leaves open that a person will want to see (how steeply a loop returns,
how far apart the loops are). Write the claims before any geometry so the shape is judged
against the words. Join every claim to a feature by the name you will give it in the script.

Then, each lap, a script. Reply with ONLY a Python script in a ```python fence against the
reference card. The API solves what follows from what is stated: a Bypass takes the wall it
leaves, the leave angle, the outer radius and the return angle, and the tool works out the
arc, the landing and the footprint; a Row takes a count, and the tool solves the largest gap
that fits and prints it -- give gap= only when the request states one; when the row does not
fit, the tool refuses with the footprint at five values of the free parameter and the
largest gap that fits. Do not compute footprints, landings or pitches yourself: the tool
prints them, and a refusal carries the numbers to choose from. Use the request's own units
in Sketch(units=...). Name the inlet and outlet by intent (main.start, duct.left, cyl.edge),
never by coordinates. A branch that leaves a wall starts on it (start=(main.top, 30)); a leg
that must rejoin a wall ends on it (end_on(main.bottom)); a leg that just runs to a
coordinate (line_to) is an open end. Give every feature a claim names a name=.

Every lap you see the annotated picture (headings in degrees, radii, the pitch, red crosses
where a check failed, insets at every junction) and the print-back: LINT, FEATURES, LEGS,
MEASURED, PATCHES, CLAIMS and VERDICT. The tables are the judge; the picture is for a person.
Reply with a revised script (the whole script) while any LINT line is an ERROR or any claim
FAILS. When VERDICT says ready, reply COMMIT. When a claim cannot be met because the request
contradicts itself or the geometry cannot satisfy it, reply COMMIT disagrees: <claim id>
naming every failing row, so the person sees exactly which clause was not met; a lint error
can never be disagreed with. A warning may be accepted by naming it: COMMIT accepts: W-CUSP
at (x, y) -- why. At most {MAX_LAPS} script laps in {MAX_SECONDS:.0f} s.

When a library reference is shown beside your candidate it is a shape a person approved;
its parameters are the library's, and the request's numbers override them.

Never run a solver, never mesh, never reply with anything but claims, a script, or COMMIT."""

BRIEF = _BRIEF_TEMPLATE.format(MAX_LAPS=MAX_LAPS, MAX_SECONDS=MAX_SECONDS)
"""Section 6.3 of DESIGN.md; imperative on purpose (the desk is a tool with a narrow job)
and pinned by test_geometry_agent.py, never by the prompt tests."""

GEOMETRY_SYSTEM = BRIEF
"""The name the tests and the session import."""

CLAIMS_SCHEMA = """\
{"schema": "openreynolds.geometry/claims-1",
 "unit": "mm" | "cm" | "m" | "in",
 "kind": "passage" | "body_in_flow"   (body_in_flow only when the script builds a BodyInBox),
 "flow": "+x" | "-x" | "+y" | "-y" | null   (the main flow direction; null: the inlet's inward normal),
 "claims": [ one object per clause of the request, each with
   "id" (unique, c1, c2, ...), "kind" (below), "says" (the request's own words), then per kind:
   measure:        "measure", "of", "value"  [+ "tol" | "tol_rel"]   judged |measured - value| <= tol
   range:          "measure", "of", "min", "max"                        judged min <= measured <= max
   count:          "of" (a Row name, or islands / open_ends / inlets / outlets / bends / passes / holes / edges), "value"
   predicate:      "predicate" (from the list below), "of"  [+ "args"]  the predicate's own rule and margin
   patch:          "patch", and one of "at": [x, y] | a rule string, "side": left | right | top | bottom  [+ "tol", "patch_kind"]
   report:         "measure", "of"                                       measured and printed, never judged
   not_measurable: "not_measurable" (why)                                carried into the table and the tool text
 ]}
"of" names a feature by the name= the script will give it: "main"; "loops[*]" every instance of a Row;
"loops[2]" one instance, 0-based; "main.legs[1]" one leg of a Passage; "fluid"; a patch name.
Example, for a request of a straight channel 3 mm wide and 60 mm long with an inlet at x=0:
{"schema": "openreynolds.geometry/claims-1", "unit": "mm", "kind": "passage", "flow": "+x", "claims": [
 {"id": "c1", "kind": "measure", "says": "a straight channel 3 mm wide", "measure": "width", "of": "main", "value": 3},
 {"id": "c2", "kind": "measure", "says": "60 mm long", "measure": "length", "of": "main", "value": 60, "tol": 0.1},
 {"id": "c3", "kind": "patch", "says": "inlet at x=0", "patch": "inlet", "at": [0, 0], "tol": 0.5},
 {"id": "c4", "kind": "not_measurable", "says": "mesh it and run checkMesh", "not_measurable": "the finish and the main agent do these; the fitness table reports the mesh"}]}"""
"""The claims schema of DESIGN.md 4.1, as the first user message states it. The example
names no shape (invariant 1): a shape word here would need an approved golden."""

COMMIT_RE = re.compile(r"^\s*COMMIT(?P<rest>.*)$", re.I | re.S)
DISAGREES_RE = re.compile(r"disagrees:\s*(?P<ids>[\w\-]+(?:\s*,\s*[\w\-]+)*)(?:\s*--\s*(?P<note>[^\n]*))?", re.I)
ACCEPTS_RE = re.compile(r"accepts:\s*(?P<code>[EWI]-[A-Z\-]+)(?:\s+at\s+\((?P<x>-?[\d.]+),\s*(?P<y>-?[\d.]+)\))?"
                        r"\s*(?:--\s*(?P<why>[^\n;]*))?", re.I)
_PYTHON_FENCE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)
_ANY_FENCE = re.compile(r"```(?:json)?[ \t]*\n?(.*?)```", re.S)


def grammar(mode: str) -> str:
    """The spec grammar as the tool's own docstring states it (the 3D branch's brief)."""
    script = TOOLBOX / ("mesh2d.py" if mode == "2d" else "cad_gen.py")
    parts = script.read_text(encoding="utf-8").split('"""', 2)
    return parts[1].strip() if len(parts) >= 2 else ""


def unavailable() -> str | None:
    """Why the desk cannot run in this process, or None when it can."""
    missing = [name for name in ("gmsh", "matplotlib") if importlib.util.find_spec(name) is None]
    if missing:
        return (f"{' and '.join(missing)} not importable in this process (`pip install "
                "openreynolds[geometry]`); on the instance, `mesh2d.py --spec` builds the "
                "same case from a spec written by hand")
    return None


def extract_spec(text: str) -> tuple[Any, bool]:
    """(the JSON spec if the reply carries one, whether the reply is a COMMIT). The 3D
    branch's reader, kept until Phase 3 replaces that branch."""
    stripped = text.strip()
    commit = stripped.upper().startswith(COMMIT)
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S)
    body = fence.group(1).strip() if fence else stripped
    starts = [i for i in (body.find("{"), body.find("[")) if i >= 0]
    if not starts:
        return None, commit
    try:
        value, _ = json.JSONDecoder().raw_decode(body[min(starts):])
    except json.JSONDecodeError:
        return None, commit
    return value, commit


def build_args(mode: str, spec_path: Path, out_dir: Path, scale: float) -> list[str]:
    """The dry-run + preview call, in this interpreter (the 3D branch's lap)."""
    script = TOOLBOX / ("mesh2d.py" if mode == "2d" else "cad_gen.py")
    args = [sys.executable, str(script), "--spec", str(spec_path), "--dry-run",
            "--preview", str(out_dir / "preview.png")]
    if scale and scale != 1.0:
        args += ["--scale", str(scale)]
    return args


def case_args(mode: str, spec_path: Path, target: Path, study: str, scale: float) -> list[str]:
    """The commit: the case files, the STEP, the msh (cad_gen) -- no Allmesh, no solver."""
    script = TOOLBOX / ("mesh2d.py" if mode == "2d" else "cad_gen.py")
    args = [sys.executable, str(script), str(target), "--spec", str(spec_path), "--study", study,
            "--preview", str(target / "outline.png"), "--force"]
    if scale and scale != 1.0:
        args += ["--scale", str(scale)]
    return args


# -- reading a reply ---------------------------------------------------------------


@dataclass
class Reply:
    kind: Literal["claims", "script", "commit", "none"]
    claims: dict | None = None
    script: str = ""
    disagrees: list[str] = field(default_factory=list)
    accepts: list[tuple[str, tuple[float, float] | None, str]] = field(default_factory=list)
    """(code, where, why)."""
    note: str = ""
    """Free text after the ids."""


def _first_json_with_claims(text: str) -> dict | None:
    """The first JSON object (fenced or bare) carrying a "claims" key."""
    candidates = [m.group(1) for m in _ANY_FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for body in candidates:
        start = body.find("{")
        while start >= 0:
            try:
                value, _ = decoder.raw_decode(body[start:])
            except json.JSONDecodeError:
                start = body.find("{", start + 1)
                continue
            if isinstance(value, dict) and "claims" in value:
                return value
            start = body.find("{", start + 1)
    return None


def extract_reply(text: str, expecting: Literal["claims", "script"]) -> Reply:
    """A reply whose first non-blank line starts with COMMIT is a commit (disagrees: and
    accepts: parsed from the rest, ids comma-separated). Otherwise, expecting "claims": the
    first JSON object (fenced or bare) with a "claims" key; expecting "script": the first
    ```python fence, else the whole text if ast.parse accepts it. A bare JSON object with
    "claims" during script laps is a claims revision. Anything else: kind "none"."""
    stripped = text.strip()
    match = COMMIT_RE.match(stripped)
    if match and stripped[:len(COMMIT)].upper() == COMMIT and not match.group("rest")[:1].isalnum():
        rest = match.group("rest")
        disagrees: list[str] = []
        note = ""
        found = DISAGREES_RE.search(rest)
        if found:
            disagrees = [i.strip() for i in found.group("ids").split(",") if i.strip()]
            note = (found.group("note") or "").strip()
        accepts = []
        for acc in ACCEPTS_RE.finditer(rest):
            where = None
            if acc.group("x") is not None:
                where = (float(acc.group("x")), float(acc.group("y")))
            accepts.append((acc.group("code").upper(), where, (acc.group("why") or "").strip()))
        return Reply(kind="commit", disagrees=disagrees, accepts=accepts, note=note)
    if expecting == "claims":
        payload = _first_json_with_claims(text)
        return Reply(kind="claims", claims=payload) if payload is not None else Reply(kind="none")
    fence = _PYTHON_FENCE.search(text)
    if fence:
        return Reply(kind="script", script=fence.group(1).strip("\n"))
    payload = _first_json_with_claims(text)
    if payload is not None:
        return Reply(kind="claims", claims=payload)
    if stripped:
        try:
            ast.parse(stripped)
        except SyntaxError:
            return Reply(kind="none")
        return Reply(kind="script", script=stripped)
    return Reply(kind="none")


# -- the result --------------------------------------------------------------------


@dataclass
class GeometryResult:
    """What the main agent reads first, then the loop's accounting. Today's names
    (`png`, `case_rel`, `mesh_report`) are accepted by the constructor and readable as
    properties so `tools._geometry` and the older tests keep working across the move."""

    outline_png: bytes | None = None
    mesh_png: bytes | None = None
    report: str = ""
    """The last print-back."""
    compliance: ComplianceTable | None = None
    lint: list[Finding] = field(default_factory=list)
    """The last lap's findings plus E-MESH-PATCH from the finish."""
    warnings_unaccepted: list[Finding] = field(default_factory=list)
    accepted: list[tuple] = field(default_factory=list)
    fitness: FitnessTable | None = None
    checkmesh: str = ""
    """The digest, or why there is none."""
    meshed: bool = False
    """Set only by mark_meshed (guarded)."""
    paths: case_mod.CasePaths = field(default_factory=case_mod.CasePaths)
    # the loop's accounting
    claims: ClaimSet | None = None
    claims_revisions: int = 0
    source: str = ""
    """The committed script."""
    record: dict | None = None
    measurements: Measurements | None = None
    laps: int = 0
    claims_seconds: float = 0.0
    seconds: float = 0.0
    tokens: dict = field(default_factory=dict)
    agreed: bool = False
    disagrees: list[str] = field(default_factory=list)
    """Claim ids (D18)."""
    disagreement_text: str = ""
    """The failing rows' text, verbatim."""
    capped: str = ""
    """"laps" | "time" | "claims" | ""."""
    error: str = ""
    mode: str = "2d"
    scale: float = 1.0
    script: str = "mesh2d.py"
    """Today's name: the toolbox script that rebuilds the record on the instance."""
    study: str = "mesh"
    """The study the case was written for: the finish's fitness table needs it to say
    whether a y+ can be estimated."""
    sizes: Sizes | None = None
    """The mesh sizes the case writer built with (write_case returns them); the fitness
    table's cell_m comes from here, never from a guess."""
    # today's names, accepted by the constructor and mapped in __post_init__
    png: InitVar[bytes | None] = None
    case_rel: InitVar[str | None] = None
    mesh_report: InitVar[str | None] = None

    def __post_init__(self, png: bytes | None, case_rel: str | None, mesh_report: str | None) -> None:
        if png is not None:
            self.outline_png = png
        if case_rel is not None:
            self.paths.case_rel = case_rel
        if mesh_report is not None:
            self.checkmesh = mesh_report

    def mark_meshed(self, digest: str, fitness: FitnessTable) -> None:
        """The only way meshed becomes True. Refuses a missing table (TypeError) -- a
        FitnessTable cannot be a placeholder (its constructor refuses one)."""
        if not isinstance(fitness, FitnessTable):
            raise TypeError("mark_meshed needs a FitnessTable; a mesh with no fitness table is not "
                            "a mesh the result can vouch for (Unmeasured says what is absent, "
                            "None says nothing)")
        object.__setattr__(self, "_marking", True)
        try:
            self.meshed = True
        finally:
            object.__setattr__(self, "_marking", False)
        self.fitness = fitness
        self.checkmesh = digest

    def __setattr__(self, name, value):
        """Raises ValueError only when name == "meshed" and value is True and the private
        flag mark_meshed sets is unset; False is always allowed (the generated __init__
        assigns the default through this method, so a blanket guard would refuse
        construction)."""
        if name == "meshed" and value and not getattr(self, "_marking", False):
            raise ValueError("meshed is set by mark_meshed(digest, fitness) only: a result is meshed "
                             "when it carries the checkMesh digest and a fitness table, not by assertion")
        object.__setattr__(self, name, value)


def _png_get(self: GeometryResult) -> bytes | None:
    return self.outline_png


def _png_set(self: GeometryResult, value: bytes | None) -> None:
    self.outline_png = value


def _case_rel_get(self: GeometryResult) -> str:
    return self.paths.case_rel


def _case_rel_set(self: GeometryResult, value: str) -> None:
    self.paths.case_rel = value


def _mesh_report_get(self: GeometryResult) -> str:
    return self.checkmesh


def _mesh_report_set(self: GeometryResult, value: str) -> None:
    self.checkmesh = value


# The dataclass keeps the InitVar in its generated __init__; the class attribute of the
# same name is then free to be the property that reads the mapped field back.
GeometryResult.png = property(_png_get, _png_set)
GeometryResult.case_rel = property(_case_rel_get, _case_rel_set)
GeometryResult.mesh_report = property(_mesh_report_get, _mesh_report_set)


# -- the first user message --------------------------------------------------------


def reference_card() -> str:
    """reference.md beside the package; `sketch.api_summary()` when the file is not there."""
    try:
        return REFERENCE_CARD.read_text(encoding="utf-8").strip()
    except OSError:
        from . import sketch
        return sketch.api_summary()


def _predicate_lines() -> list[str]:
    lines = []
    for name, fn in claims_mod.PREDICATES.items():
        doc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
        lines.append(f"  {name}: {doc[0] if doc else ''}".rstrip())
    return lines


def claims_message(request: str, reference: tuple[Any, str] | None) -> str:
    """The lap-0 user message: the request, the reference card, the claims schema, the
    measures and predicates the table can judge, the approved library match when there
    is one (D32), and the one-line instruction."""
    parts = [f"The request:\n{request.strip()}",
             f"The reference card (the API the script is written against):\n{reference_card()}",
             f"The claims schema:\n{CLAIMS_SCHEMA}"]
    measures = [f"  {key}: {sentence}" for key, sentence in claims_mod.MEASURES.items()]
    parts.append("Measures a claim may name:\n" + ("\n".join(measures) if measures else "  (none registered)"))
    predicates = _predicate_lines()
    parts.append("Predicates a claim may name:\n" + ("\n".join(predicates) if predicates else "  (none registered)"))
    if reference is not None:
        entry, preset = reference
        approval = None
        try:
            approval = library.approval(entry)
        except Exception:  # noqa: BLE001 - the sentence below then names no date
            approval = None
        when = (approval or {}).get("at", "")
        parts.append(f"A library reference: {entry.name} (preset {preset}). A person approved this shape on "
                     f"{when or 'record'} from {entry.source}; its parameters are the library's, the "
                     "request's numbers override them.")
    parts.append("Reply with the claims JSON only.")
    return "\n\n".join(parts)


def claims_answer(claims: ClaimSet, lines: list[str]) -> str:
    """Section 6.2: the counts, the per-claim lines the parser returned, and the turn."""
    checkable = sum(1 for c in claims.claims if c.kind in claims_mod.CHECKABLE_KINDS)
    reported = sum(1 for c in claims.claims if c.kind == "report")
    unmeasurable = sum(1 for c in claims.claims if c.kind == "not_measurable")
    head = (f"{len(claims.claims)} claims: {checkable} measurable, {reported} reported, "
            f"{unmeasurable} not measurable")
    body = "\n".join(f"  {line}" for line in lines)
    return (head + ("\n" + body if body else "") +
            "\nNow the script. Reply with ONLY a ```python block against the reference card.")


def _user(text: str, png: bytes | None = None) -> dict:
    content: list[dict] = []
    if png:
        content.append(images.attachment(images.downscale(png, "image/png"), "image/png"))
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def findings_of(outcome: RunOutcome) -> list[Finding]:
    return [Finding.from_dict(f) for f in (outcome.result or {}).get("lint") or []]


def table_of(outcome: RunOutcome) -> ComplianceTable | None:
    table = (outcome.result or {}).get("compliance")
    return None if table is None else ComplianceTable.from_dict(table)


def units_disagree(outcome: RunOutcome, claims: ClaimSet | None) -> Finding | None:
    """E-UNITS-CLAIMS (DESIGN.md 4.1): the sketch's `units=` differs from the claims'
    `unit` and the lint's converted ratio test passed, so both are right by the numbers
    and the record would carry two units for one shape. Raised by the desk, not the lint."""
    if claims is None or not outcome.result:
        return None
    record = outcome.result.get("record") or {}
    sketch_units = record.get("units")
    if not sketch_units or not claims.unit or sketch_units == claims.unit:
        return None
    if any(f.get("code") in ("E-UNITS", "W-UNITS") for f in outcome.result.get("lint") or []):
        return None
    extent = ((outcome.result.get("measurements") or {}).get("extent")) or (record.get("measurements") or {}).get("extent")
    span = max(extent) if extent else None
    largest = claims.largest_length
    numbers = (f"({span:g} vs {largest:g})" if span is not None and largest is not None
               else "(by the lint's ratio test)")
    return Finding(level="error", code="E-UNITS-CLAIMS", subject="claims",
                   what=(f"the claims say unit \"{claims.unit}\" and the sketch says units=\"{sketch_units}\", "
                         f"yet the numbers agree {numbers}"),
                   fix=("the unit changed between the claims and the script; one of them is wrong "
                        "(raised by the desk, not the lint)"))


class GeometryAgent:
    """The loop. One provider of its own, the same key and (by default) the same
    model as the main agent; `_client` is the seam a test replaces, as the desk's."""

    def __init__(self, cfg: Any, backend: Any, store: Any, home: str):
        self.backend = backend
        self.store = store
        self.home = str(home or "").rstrip("/")
        self.model = cfg.geometry_model or cfg.model
        # The desk's own effort, not the main loop's: the hosted app runs the loop at
        # medium, and at medium the model places an arc's end right one time in six.
        self.effort = cfg.geometry_effort or "high"
        self.preferences = getattr(cfg, "preferences", "") or ""
        self._provider = make_provider(cfg, timeout=min(120.0, cfg.llm_timeout_s or 120.0))

    @property
    def _client(self):
        return self._provider.client

    @_client.setter
    def _client(self, value) -> None:
        self._provider.client = value

    # -- the parts a test replaces -------------------------------------------

    def _system(self) -> str:
        if not self.preferences.strip():
            return BRIEF
        return (BRIEF + "\n\nThe person keeps a standing note, in their own words:\n"
                + self.preferences.strip())

    def _ask(self, messages: list[dict]) -> Any:
        return self._provider.stream(model=self.model, system=self._system(), messages=messages,
                                     tools=[], effort=self.effort, max_tokens=MAX_REPLY_TOKENS,
                                     listener=Listener())

    def _run_script(self, script: str, work: Path, claims_path: Path | None, reference: str | None) -> RunOutcome:
        """One script lap in the child interpreter (runner.run_script): the kernel loads
        there, never in the process that hosts the desk."""
        return runner.run_script(script, work, claims_path, reference, timeout_s=RUN_TIMEOUT_S)

    def _write_case(self, local: Path, record: dict, script: str, claims: dict | None,
                    table: ComplianceTable | None, study: str, scale: float) -> tuple[case_mod.CasePaths, Sizes]:
        return case_mod.write_case(local, record, script, claims, table, study, scale)

    def _build(self, mode: str, spec: Any, work: Path, scale: float) -> tuple[int, str, bytes | None]:
        """The 3D branch's lap: dry-run + preview in a child interpreter (cad_gen), so a
        gmsh crash cannot take the runner down, and the tool's own report text is the
        feedback. Phase 3 replaces it."""
        work.mkdir(parents=True, exist_ok=True)
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        try:
            proc = subprocess.run(build_args(mode, spec_path, work, scale), capture_output=True,
                                  text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return 1, "the build ran past 120 s and was stopped", None
        report_text = (proc.stdout + proc.stderr).strip()
        png = work / "preview.png"
        return proc.returncode, report_text, (png.read_bytes() if png.exists() else None)

    def _write_case_3d(self, spec: Any, study: str, local: Path, scale: float) -> tuple[case_mod.CasePaths, Sizes | None]:
        """The 3D commit as today (cad_gen writes the case). The sizes come from the
        writer's own `cells ... m on the body` line; None when it printed none, and the
        finish then says the fitness table could not be built."""
        local.mkdir(parents=True, exist_ok=True)
        spec_path = local / "geometry.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        proc = subprocess.run(case_args("3d", spec_path, local, study, scale), capture_output=True,
                              text=True, timeout=300)
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            raise RuntimeError(output[-1500:] or f"exit {proc.returncode}")
        found = re.search(r"cells\s+([0-9.eE+-]+) m on the body", output)
        sizes = None
        if found:
            near = float(found.group(1))
            sizes = Sizes(cell=near, wall_cell=near, thickness=near)
        return case_mod.CasePaths(), sizes

    def _reference(self, request: str) -> tuple[Any, str] | None:
        """The approved library match for the request, or None: a library that cannot
        answer is no reference, never a failed lap."""
        try:
            return library.match(request)
        except Exception as exc:  # noqa: BLE001 - the desk runs without a library
            trace.event("geometry_library", error=f"{type(exc).__name__}: {exc}"[:200])
            return None

    # -- the loop --------------------------------------------------------------

    def run(self, request: str, mode: str = "2d", study: str = "mesh",
            case: str | None = None) -> GeometryResult:
        reason = unavailable()
        if reason:
            return GeometryResult(error=reason)
        mode = "3d" if str(mode).lower() == "3d" else "2d"
        study = study if study in ("mesh", "steady", "transient") else "mesh"
        case = (case or "geometry").strip("/ ") or "geometry"
        if mode == "3d":
            return self._run_3d(request, study, case)
        return self._run_2d(request, study, case)

    def _turn(self, messages: list[dict], tokens: dict[str, int]) -> Any:
        turn = self._ask(messages)
        for name, count in (getattr(turn, "tokens", None) or {}).items():
            tokens[name] = tokens.get(name, 0) + int(count)
        messages.append(turn.as_message())
        return turn

    def _run_2d(self, request: str, study: str, case: str) -> GeometryResult:
        work = Path(tempfile.mkdtemp(prefix="geometry-"))
        tokens: dict[str, int] = {}
        started = time.monotonic()
        result = GeometryResult(mode="2d", study=study, script="mesh2d.py")
        reference = self._reference(request)
        reference_name = reference[0].name if reference else None

        # lap 0: the claims, a separate call not counted toward MAX_LAPS (D15)
        messages: list[dict] = [_user(claims_message(request, reference), self._golden_png(reference))]
        claims_set: ClaimSet | None = None
        claims_payload: dict | None = None
        claims_error = ""
        answer = ""
        for attempt in range(2):
            try:
                turn = self._turn(messages, tokens)
            except ProviderError as exc:
                return GeometryResult(error=f"the model call failed: {exc}", laps=0,
                                      seconds=time.monotonic() - started, tokens=tokens)
            reply = extract_reply(turn.text, "claims")
            if reply.kind != "claims" or reply.claims is None:
                claims_error = "the reply carried no JSON object with a \"claims\" key"
            else:
                try:
                    claims_set, lines = claims_mod.parse(reply.claims)
                    claims_payload = reply.claims
                    answer = claims_answer(claims_set, lines)
                    break
                except Exception as exc:  # noqa: BLE001 - a malformed file is a fact the model sees
                    claims_error = str(exc) or type(exc).__name__
            if attempt == 0:
                messages.append(_user(f"The claims did not parse: {claims_error}\nReply with the claims "
                                      "JSON only, in the schema given."))
        result.claims_seconds = time.monotonic() - started
        claims_path: Path | None = None
        if claims_set is not None:
            claims_path = work / "claims.json"
            claims_path.write_text(json.dumps(claims_payload, indent=1), encoding="utf-8")
            messages.append(_user(answer))
        else:
            messages.append(_user(f"The claims did not parse twice ({claims_error}). No claims are recorded, "
                                  "so nothing can be committed; the shape is still worth drawing. Reply with "
                                  "ONLY a ```python block against the reference card."))
        result.claims = claims_set

        laps = 0
        last_built: RunOutcome | None = None
        # the last outcome that built (rc 0 or 2): what a COMMIT is judged on, so a COMMIT
        # over a lint error is refused naming the error rather than "nothing has built"
        last_script = ""
        last_clean: RunOutcome | None = None
        # the last rc-0 outcome with no error the desk added: what a cap commits (3.14)
        last_clean_script = ""
        last_png: bytes | None = None
        last_text = ""
        commit_reply: Reply | None = None
        no_claims_end = False

        def absorb(outcome: RunOutcome, script: str) -> dict:
            """Keep what the outcome carries and build the next user message from it."""
            nonlocal last_built, last_script, last_clean, last_clean_script, last_png, last_text
            if outcome.png:
                last_png = outcome.png
            trace.event("geometry_lap", lap=laps, rc=outcome.rc, seconds=round(time.monotonic() - started, 1))
            units_finding = units_disagree(outcome, claims_set)
            if units_finding is not None and outcome.result is not None:
                outcome.result.setdefault("lint", []).append(units_finding.as_dict())
                outcome.text = units_finding.text() + "\n" + outcome.text
            if outcome.rc in (0, 2):
                last_text = outcome.text
            if outcome.rc in (3, 4, 5):
                return _user(f"The tool refused it:\n{outcome.text[-6000:]}", outcome.png)
            last_built, last_script = outcome, script
            if outcome.rc == 0 and units_finding is None:
                last_clean, last_clean_script = outcome, script
            if outcome.rc == 2 or units_finding is not None:
                return _user(outcome.text[-6000:] + "\n\nFix the errors; reply with the whole script.", outcome.png)
            return _user(outcome.text[-6000:] + "\n\nCompare the CLAIMS table with the request. Reply with a "
                         "revised script, COMMIT when VERDICT says ready, or COMMIT disagrees: <ids>.",
                         outcome.png)

        while laps < MAX_LAPS and time.monotonic() - started < MAX_SECONDS:
            laps += 1
            try:
                turn = self._turn(messages, tokens)
            except ProviderError as exc:
                # the laps so far are not lost with the call: the last picture and
                # print-back go back with the words
                result.error = f"the model call failed: {exc}"
                result.laps, result.seconds, result.tokens = laps, time.monotonic() - started, tokens
                result.outline_png, result.report = last_png, last_text
                return result
            reply = extract_reply(turn.text, "script")
            if reply.kind == "script":
                outcome = self._run_script(reply.script, work / f"lap{laps}", claims_path, reference_name)
                messages.append(absorb(outcome, reply.script))
            elif reply.kind == "claims":
                try:
                    claims_set, lines = claims_mod.parse(reply.claims or {})
                except Exception as exc:  # noqa: BLE001
                    messages.append(_user(f"The claims revision did not parse: {exc}. The earlier claims stand; "
                                          "reply with a ```python block, a corrected claims JSON, or COMMIT."))
                    continue
                claims_payload = reply.claims
                result.claims = claims_set
                result.claims_revisions += 1
                claims_path = work / f"claims{result.claims_revisions}.json"
                claims_path.write_text(json.dumps(claims_payload, indent=1), encoding="utf-8")
                answer = claims_answer(claims_set, lines)
                if last_built is not None:
                    # the last built script is judged again against the new claims: a run, not a model call
                    outcome = self._run_script(last_script, work / f"lap{laps}", claims_path, reference_name)
                    message = absorb(outcome, last_script)
                    message["content"][-1]["text"] = (f"Claims revised ({result.claims_revisions} revision(s)); the "
                                                      "last built script judged against them:\n" +
                                                      message["content"][-1]["text"])
                    messages.append(message)
                else:
                    messages.append(_user(f"Claims revised ({result.claims_revisions} revision(s)).\n{answer}"))
            elif reply.kind == "commit":
                if last_built is None:
                    messages.append(_user("No script has built yet; nothing can be committed. Reply with a "
                                          "```python block against the reference card."))
                    continue
                if claims_set is None:
                    no_claims_end = True
                    break
                ok, reason = claims_mod.can_commit(findings_of(last_built), table_of(last_built), reply)
                if ok:
                    commit_reply = reply
                    break
                table = table_of(last_built)
                failing = table.failing_ids() if table is not None else []
                head = reason if reason.lower().startswith("commit refused") else f"COMMIT refused: {reason}"
                # the disagreement is offered only when there is a failing row to name: an
                # invented pair of ids would be refused next lap as "nothing to disagree with"
                offer = (f" Fix it, or reply `COMMIT disagrees: {', '.join(failing)}` naming every failing claim."
                         if failing else " Fix it; reply with the whole script.")
                messages.append(_user(f"{head}.{offer}"))
            else:
                messages.append(_user("That was not a script. Reply with ONLY a ```python block, a claims "
                                      "JSON, or COMMIT."))

        seconds = time.monotonic() - started
        result.laps, result.seconds, result.tokens = laps, seconds, tokens
        result.outline_png, result.report = last_png, last_text
        if commit_reply is None and not no_claims_end:
            result.capped = "laps" if laps >= MAX_LAPS else "time"
        if claims_set is None:
            result.capped = "claims"
            result.error = f"no claims: {claims_error}"
            return result
        if last_built is None:
            result.error = (f"no clean geometry within the {result.capped} allowed "
                            f"({laps} lap(s), {seconds:.0f} s)")
            return result
        chosen, chosen_script = last_built, last_script
        if commit_reply is None:
            # the cap: the last rc-0 outcome is committed, and only when its lint is clean
            # (integration 8.1); a run that never got there ends with the last picture
            if last_clean is None:
                codes = ", ".join(sorted({f.code for f in findings_of(last_built) if f.level == "error"}))
                result.error = (f"no clean geometry within the {result.capped} allowed ({laps} lap(s), "
                                f"{seconds:.0f} s): the last script that built has lint errors ({codes})")
                return result
            chosen, chosen_script = last_clean, last_clean_script
            table = table_of(chosen)
            if table is None or table.checkable == 0:
                result.error = (f"no clean geometry within the {result.capped} allowed ({laps} lap(s), "
                                f"{seconds:.0f} s): the last script that built clean has no checkable claim "
                                "to judge it by")
                return result
        # the picture and the print-back handed back are the committed script's: a later
        # refusal (rc 5 with a partial) drew its own picture, and that is not the shape
        if chosen.png:
            result.outline_png = chosen.png
        result.report = chosen.text
        return self._commit_2d(result, chosen, chosen_script, findings_of(chosen), table_of(chosen),
                               claims_payload, commit_reply, study, case)

    def _golden_png(self, reference: tuple[Any, str] | None) -> bytes | None:
        if reference is None:
            return None
        try:
            png = library.GOLDEN / f"{reference[0].name}.png"
            return png.read_bytes() if png.exists() else None
        except Exception:  # noqa: BLE001 - a picture that cannot be read is no picture
            return None

    def _commit_2d(self, result: GeometryResult, outcome: RunOutcome, script: str, findings: list[Finding],
                   table: ComplianceTable | None, claims_payload: dict | None, reply: Reply | None,
                   study: str, case: str) -> GeometryResult:
        """write_case (local) -> put_tree -> the finish -> the fitness table."""
        disagrees = list(reply.disagrees) if reply else []
        accepts = list(reply.accepts) if reply else []
        if table is not None:
            for row in table.rows:
                if row.id in disagrees:
                    row.verdict = "disagreed"
        disagreed_rows = [r for r in (table.rows if table else []) if r.id in disagrees]
        result.disagrees = disagrees
        result.disagreement_text = "\n".join(_row_text(r) for r in disagreed_rows)
        result.agreed = reply is not None and not disagrees
        result.accepted = accepts
        result.compliance = table
        result.lint = findings
        result.warnings_unaccepted = [f for f in findings if f.level == "warn" and not _accepted(f, accepts)]
        result.source = script
        record = dict((outcome.result or {}).get("record") or {})
        # the record carries the script it was compiled from and its digest (4.2), so a
        # rebuild can tell an edited geometry.py from the one that was committed
        record.setdefault("script", script)
        record.setdefault("script_sha256", hashlib.sha256(str(record["script"]).encode("utf-8")).hexdigest())
        if claims_payload is not None:
            record["claims"] = claims_payload
        if table is not None:
            record["compliance"] = table.as_dict()
        record["lint"] = [f.as_dict() for f in findings]
        if accepts:
            record["accepted"] = [[code, list(where) if where else None, why] for code, where, why in accepts]
        result.record = record
        measurements_dict = (outcome.result or {}).get("measurements")
        try:
            result.measurements = Measurements.from_dict(measurements_dict) if measurements_dict else None
        except (KeyError, TypeError, ValueError):
            result.measurements = None
        scale = record.get("scale") or (result.measurements.scale if result.measurements else None) or 1.0
        result.scale = float(scale)
        local = Path(self.store.fetch_dir()) / case
        try:
            paths, sizes = self._write_case(local, record, script, claims_payload, table, study, result.scale)
        except Exception as exc:  # noqa: BLE001 - the report says what happened
            result.error = f"the case did not write: {exc}"
            return result
        remote = f"{self.home}/{case}" if self.home else case
        self.backend.put_tree(local, remote)
        paths.case_rel = remote
        result.paths, result.sizes = paths, sizes
        expected = [p.get("name") for p in record.get("patches") or [] if isinstance(p, dict) and p.get("name")]
        self._finish(remote, result, expected)
        if result.fitness is not None:
            try:
                (local / paths.fitness).write_text(json.dumps(result.fitness.as_dict(), indent=1), encoding="utf-8")
            except OSError:
                pass
        return result

    # -- the 3D branch: today's loop, unchanged, until Phase 3 ---------------------

    def _run_3d(self, request: str, study: str, case: str) -> GeometryResult:
        mode = "3d"
        work = Path(tempfile.mkdtemp(prefix="geometry-"))
        messages: list[dict] = [_user(f"The grammar:\n\n{grammar(mode)}\n\nThe request:\n{request.strip()}\n\n"
                                      "Reply with the JSON spec only.")]
        tokens: dict[str, int] = {}
        laps = 0
        started = time.monotonic()
        last_spec: Any = None
        last_scale = 1.0
        last_report = ""
        last_png: bytes | None = None
        agreed = False
        disagreement = ""

        while laps < MAX_LAPS and time.monotonic() - started < MAX_SECONDS:
            laps += 1
            try:
                turn = self._turn(messages, tokens)
            except ProviderError as exc:
                return GeometryResult(error=f"the model call failed: {exc}", laps=laps,
                                      seconds=time.monotonic() - started, tokens=tokens, mode=mode)
            text = turn.text
            spec, commit = extract_spec(text)
            if commit and last_spec is not None:
                lower = text.lower()
                if "disagrees:" in lower:
                    disagreement = text[lower.index("disagrees:") + len("disagrees:"):].strip().splitlines()[0]
                agreed = not disagreement
                break
            if spec is None:
                messages.append(_user("That was not a spec. Reply with ONLY the JSON spec in the grammar, or the "
                                      "word COMMIT once a built spec is the shape that was asked for."))
                continue
            scale = float(spec.get("scale", 1.0)) if isinstance(spec, dict) else 1.0
            rc, report_text, png = self._build(mode, spec, work / f"lap{laps}", scale)
            trace.event("geometry_lap", lap=laps, rc=rc, seconds=round(time.monotonic() - started, 1))
            if rc != 0:
                messages.append(_user(f"The tool refused it:\n{report_text[-3000:]}\n\nReply with a corrected spec.",
                                      png))
                continue
            last_spec, last_scale, last_report, last_png = spec, scale, report_text, png
            messages.append(_user("The tool drew it (the picture above) and measured:\n" + report_text[-3000:] +
                                  "\n\nCompare with the request. Reply with a revised spec, or COMMIT if the "
                                  "picture and the numbers are the shape that was asked for.", png))

        seconds = time.monotonic() - started
        capped = "" if (agreed or disagreement) else ("laps" if laps >= MAX_LAPS else "time")
        if last_spec is None:
            return GeometryResult(error=f"no spec built within the {capped} allowed "
                                  f"({laps} lap(s), {seconds:.0f} s)", laps=laps,
                                  seconds=seconds, tokens=tokens, capped=capped, mode=mode)
        local = Path(self.store.fetch_dir()) / case
        try:
            paths, sizes = self._write_case_3d(last_spec, study, local, last_scale)
        except Exception as exc:  # noqa: BLE001 - the report says what happened
            return GeometryResult(report=last_report, png=last_png, laps=laps, seconds=seconds,
                                  tokens=tokens, error=f"the case did not write: {exc}", mode=mode)
        remote = f"{self.home}/{case}" if self.home else case
        self.backend.put_tree(local, remote)
        paths.case_rel = remote
        # D34: a 3D commit carries a table that says what it does not know, never None
        table = ComplianceTable(rows=[ComplianceRow(id="c0", says=request.strip(), kind="not_measurable", expected="",
                                                    measured="3D claims arrive with Phase 3",
                                                    verdict="not_measurable")])
        result = GeometryResult(report=last_report, png=last_png, laps=laps, seconds=seconds, tokens=tokens,
                                agreed=agreed, disagreement_text=disagreement, script="cad_gen.py",
                                scale=last_scale, capped=capped, mode=mode, study=study, compliance=table,
                                record=last_spec if isinstance(last_spec, dict) else {"ops": last_spec},
                                paths=paths, sizes=sizes)
        self._finish(remote, result, [])
        return result

    # -- the finish: the OpenFOAM mesh, its check and its picture, on the instance ----

    def _finish(self, remote: str, result: GeometryResult, expected_patches: list[str]) -> None:
        """Run Allmesh, checkMesh and the mesh render on the instance in one exec, read
        the boundary file back, and carry the digest, the picture and the fitness table
        in the result. Measured before this existed: the main agent spent four to six
        turns after "case written" finding out that nothing was meshed yet, that ./Allmesh
        had no exec bit, what the render tool's flags were. None of that is geometry."""
        # a 2D case carries walls and frontAndBack unnamed; a cad_gen case has no z-flat faces
        implicit = case_mod.IMPLICIT_PATCHES if result.mode == "2d" else ()
        finish = case_mod.finish_on_instance(self.backend, remote, expected_patches, FINISH_TIMEOUT_S,
                                             implicit=implicit)
        result.mesh_png = finish.mesh_png
        for name in finish.missing:
            listed = ", ".join(finish.boundary_patches) or "nothing"
            result.lint.append(Finding(level="error", code="E-MESH-PATCH", subject="mesh",
                                       what=f"the OpenFOAM mesh has no patch '{name}'; constant/polyMesh/boundary "
                                            f"lists {listed}",
                                       fix="(found by the finish; the record names it -- the extrusion lost it)"))
        if finish.rc != 0 or not finish.digest:
            result.checkmesh = ("Allmesh did not finish (exit %s):\n%s" % (finish.rc, finish.output[-1500:])
                                if finish.rc != 0 else (finish.output[-1500:] or "checkMesh wrote no log"))
            return
        if result.sizes is None:
            result.checkmesh = finish.digest + "\n(meshed, but no fitness table: the case writer reported no cell size)"
            return
        try:
            table = fitness_mod.from_digest(finish.digest_data or {}, result.sizes, self._smallest_passage_m(result),
                                            result.study, None, result.scale)
            result.mark_meshed(finish.digest, table)
        except Exception as exc:  # noqa: BLE001 - a digest without a table is reported, not asserted
            result.checkmesh = finish.digest + f"\n(meshed, but no fitness table: {type(exc).__name__}: {exc})"

    @staticmethod
    def _smallest_passage_m(result: GeometryResult) -> float | Unmeasured:
        """The measured narrowest passage in metres (never a declared width); the 3D
        branch and a record without the measure say so."""
        if result.mode == "3d":
            return Unmeasured("no 2D passage measure for a 3D body in Phase 1")
        m = result.measurements
        if m is None or m.passage is None:
            return Unmeasured("the record carries no passage measure")
        return float(m.passage.min) * float(result.scale)


def _row_text(row: ComplianceRow) -> str:
    text = f"{row.id} '{row.says}': {row.measured}"
    return f"{text}; {row.detail}" if row.detail else text


def _accepted(finding: Finding, accepts: list[tuple]) -> bool:
    """A warning is accepted when a COMMIT accepts: names its code and, when both give a
    place, the same place within 0.1 (sketch units)."""
    for code, where, _why in accepts:
        if code.upper() != finding.code.upper():
            continue
        if where is None or finding.where is None:
            return True
        if abs(where[0] - finding.where[0]) <= 0.1 and abs(where[1] - finding.where[1]) <= 0.1:
            return True
    return False
