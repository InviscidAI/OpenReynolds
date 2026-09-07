"""The library: parameterised shapes a person approved, with their goldens.

An entry is a module beside this one (`tesla_valve.py`, `serpentine.py`, `t_junction.py`)
that carries `SOURCE` (where its numbers come from), `KEYWORDS` (the words a request uses
for the shape), `PRESETS` (named parameter sets; the first is the default), `CLAIMS` (the
claims file its golden is judged against, or None) and `build(**params) -> Sketch` with no
default geometric number (D16: no number in the repo comes from nowhere; the presets are
the numbers, and each preset says where it was read from).

`match` returns only entries whose approval exists, is flagged `approved`, and whose
`source_sha` matches the entry's file now (D32), so nothing unapproved is ever shown to the
model as a reference: while the goldens wait for a person, the desk's REFERENCE block reads
"none" and `cli build --reference NAME` is refused. Goldens (`golden/<name>.json`) pin the
outline polyline and the measurements; the PNG beside each is regenerated and never diffed
(D17). `golden/APPROVALS.json` is keyed by entry name and carries the source sha the person
looked at; `cli golden --regenerate` writes a record with `approved: false`, and only
`cli golden --approve NAME --by <person>` (or a hand edit) flips it.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import importlib
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .. import claims, compile, lint, measure, preview  # noqa: F401
from ..sketch import Sketch  # noqa: F401

GOLDEN: Path = Path(__file__).resolve().parent / "golden"
"""<name>.json, <name>.png, APPROVALS.json."""

APPROVALS = "APPROVALS.json"

OUTLINE_POINTS = 500
"""The golden outline is sampled at span / OUTLINE_POINTS arc length (4.5)."""

GOLDEN_MEASUREMENT_KEYS = ("extent", "area", "islands", "features", "patches")
"""What a golden pins of Measurements.as_dict() (4.5's extent, area, islands, features;
`patches` beside them because preview.reference_distance translates the golden so the
inlet centres coincide, and it reads the inlet's midpoints from here)."""


class LibraryError(KeyError):
    """A name the library does not know, or an entry that cannot make a golden; the
    message names the fix. A KeyError so the cli's reference gate (which catches
    KeyError) reads it as 'not approved', which is what an unknown entry is."""

    def __str__(self) -> str:   # KeyError quotes its argument; the message is prose
        return self.args[0] if self.args else ""


@dataclass
class Entry:
    name: str
    module: ModuleType
    source: str
    keywords: tuple[str, ...]
    presets: dict[str, dict]
    claims: dict | None

    def build(self, preset: str | None = None, **params) -> Sketch:
        """The entry's sketch from a preset (the first one when none is named) with
        `params` overriding the preset's numbers: 'its parameters are the library's, the
        request's numbers override them'."""
        name = preset or self.default_preset
        if name not in self.presets:
            raise LibraryError(f"library entry '{self.name}' has no preset '{name}'; "
                               f"its presets are {', '.join(self.presets)}")
        merged = {**self.presets[name], **params}
        return self.module.build(**merged)

    @property
    def default_preset(self) -> str:
        return next(iter(self.presets))

    def as_dict(self) -> dict:
        """The module by its name: a module object does not round-trip through JSON."""
        return {"name": self.name, "module": getattr(self.module, "__name__", str(self.module)),
                "source": self.source, "keywords": list(self.keywords), "presets": self.presets,
                "claims": self.claims}

    @staticmethod
    def from_dict(d: dict) -> "Entry":
        """Needs the entry's module imported: `get(name)` is the way back from a record."""
        return get(d["name"])


@dataclass
class ReferenceMatch:
    """What preview.draw and report.text receive."""

    entry: str
    preset: str
    approved_at: str
    outline: list[list[tuple[float, float]]]
    measurements: dict
    hausdorff: float | None

    def as_dict(self) -> dict:
        return {"entry": self.entry, "preset": self.preset, "approved_at": self.approved_at,
                "outline": [[list(p) for p in loop] for loop in self.outline],
                "measurements": self.measurements, "hausdorff": self.hausdorff}

    @staticmethod
    def from_dict(d: dict) -> "ReferenceMatch":
        return ReferenceMatch(entry=d["entry"], preset=d["preset"], approved_at=d.get("approved_at", ""),
                              outline=[[(float(p[0]), float(p[1])) for p in loop] for loop in d.get("outline", [])],
                              measurements=dict(d.get("measurements", {})), hausdorff=d.get("hausdorff"))


# -- the entries -------------------------------------------------------------------------


def _entry_names() -> list[str]:
    """Every module beside this one that is not private: the directory is the registry,
    so adding an entry is adding a file."""
    here = Path(__file__).resolve().parent
    return sorted(p.stem for p in here.glob("*.py") if not p.stem.startswith("_"))


def _load(name: str) -> Entry:
    module = importlib.import_module(f"{__name__}.{name}")
    missing = [attr for attr in ("SOURCE", "KEYWORDS", "PRESETS", "build") if not hasattr(module, attr)]
    if missing:
        raise LibraryError(f"library entry '{name}' lacks {', '.join(missing)}; an entry module carries SOURCE, "
                           "KEYWORDS, PRESETS, CLAIMS (or None) and build(**params) -> Sketch")
    if not module.PRESETS:
        raise LibraryError(f"library entry '{name}' has no preset; PRESETS needs at least one named parameter set "
                           "(the first is the default)")
    return Entry(name=name, module=module, source=str(module.SOURCE), keywords=tuple(module.KEYWORDS),
                 presets={k: dict(v) for k, v in module.PRESETS.items()}, claims=getattr(module, "CLAIMS", None))


def entries() -> list[Entry]:
    return [_load(name) for name in _entry_names()]


def get(name: str) -> Entry:
    names = _entry_names()
    if name not in names:
        raise LibraryError(f"no library entry '{name}'; the entries are {', '.join(names) or 'none'}")
    return _load(name)


def source_sha(entry: Entry) -> str:
    """sha256 of the entry module's text with line endings normalised to LF, so a
    checkout with CRLF endings approves the same source as one without."""
    text = Path(entry.module.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- approvals ----------------------------------------------------------------------------


def approvals() -> dict:
    """golden/APPROVALS.json as a dict keyed by entry name ({} when absent)."""
    path = GOLDEN / APPROVALS
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise LibraryError(f"{path} is not JSON ({exc}); regenerate the goldens (cli golden --regenerate) "
                           "or fix the file by hand") from None
    return data if isinstance(data, dict) else {}


def approval(entry: Entry) -> dict | None:
    """golden/APPROVALS.json[name] when a person flipped its `approved` flag; None for a
    record a regeneration wrote (approved: false) or no record at all. The sha check
    against the entry's file now is the caller's (D32: `match` and the cli's gate do it)."""
    record = approvals().get(entry.name)
    if not isinstance(record, dict) or record.get("approved") is not True:
        return None
    return record


def is_approved(entry: Entry) -> bool:
    """The whole D32 gate: an approval flagged by a person whose source_sha is the entry's now."""
    record = approval(entry)
    return record is not None and record.get("source_sha") == source_sha(entry)


# -- matching a request ----------------------------------------------------------------------


_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")


def _numbers(text: str) -> list[float]:
    return [float(t) for t in _NUMBER.findall(text)]


def _nearest_preset(entry: Entry, request: str) -> str:
    """The preset with the most numeric parameters that appear in the request (a tie
    keeps the earlier preset, so the default wins when the request gives no numbers)."""
    numbers = _numbers(request)
    best, best_hits = entry.default_preset, -1
    for name, params in entry.presets.items():
        hits = sum(1 for v in params.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and any(math.isclose(float(v), n, rel_tol=1e-9, abs_tol=1e-9) for n in numbers))
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def match(request: str) -> tuple[Entry, str] | None:
    """Lower-cased keyword hit -> (entry, the preset nearest the request's numbers), for
    APPROVED entries only: `approval(entry) is not None and approval["source_sha"] ==
    source_sha(entry)` (D32); an unapproved entry is never matched, whatever the request
    says. Shown to the model as: 'a person approved this shape on <date> from <SOURCE>;
    its parameters are the library's, the request's numbers override them'. Several
    entries hit: the longest keyword wins (a more specific word is a better reading)."""
    low = request.lower()
    best: tuple[int, Entry] | None = None
    for entry in entries():
        if not is_approved(entry):
            continue
        hits = [k for k in entry.keywords if k.lower() in low]
        if not hits:
            continue
        score = max(len(k) for k in hits)
        if best is None or score > best[0]:
            best = (score, entry)
    if best is None:
        return None
    entry = best[1]
    return entry, _nearest_preset(entry, request)


# -- goldens ----------------------------------------------------------------------------------


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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _reference_w_min(gmsh, face: int, plan) -> float:
    if plan.declared_widths:
        return float(min(plan.declared_widths))
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, face)
    return max(min(x1 - x0, y1 - y0) / 10.0, 1e-9)


def _analyse(gmsh, plan, claim_set):
    """The kernel's pipeline as the cli runs it (build, walk, ports, measure, judge,
    comply), reproduced here because the library sits below the cli in the dependency
    graph (section 2) and a golden must be made by the same instruments a lap is."""
    face, legs, checks = compile.build(gmsh, plan)
    w_min = _reference_w_min(gmsh, face, plan)
    wk = measure.walk(gmsh, face, w_min)
    ends = measure.open_ends(gmsh, wk, w_min, None)
    curve_patch, resolved_rules, port_findings = measure.resolve_ports(gmsh, wk, plan, ends)
    m = measure.measure(gmsh, face, plan, wk, legs, curve_patch, claim_set)
    findings = list(lint.judge(gmsh, face, plan, wk, m, list(checks or []), claim_set))
    seen = {(f.code, f.subject, f.what) for f in findings}
    findings += [f for f in port_findings if (f.code, f.subject, f.what) not in seen]
    table = None if claim_set is None else claims.comply(claim_set, m, findings, plan)
    return face, m, findings, table


def regenerate(entry: Entry, preset: str | None = None, into: Path | None = None) -> dict:
    """Builds, measures, draws; returns the golden dict of 4.5 (and writes <name>.json and
    <name>.png into `into` when it is given). A golden is refused when the build carries a
    lint error or its compliance table has a FAIL row: a golden is a shape a person is asked
    to approve, and an entry that does not pass its own claims has nothing to approve."""
    preset = preset or entry.default_preset
    params = dict(entry.presets[preset]) if preset in entry.presets else None
    if params is None:
        raise LibraryError(f"library entry '{entry.name}' has no preset '{preset}'; its presets are "
                           f"{', '.join(entry.presets)}")
    claim_set = None
    if entry.claims is not None:
        claim_set, _ = claims.parse(entry.claims)
    sk = entry.build(preset)
    with _gmsh_session(f"golden-{entry.name}") as gmsh:
        plan = compile.plan(sk, claim_set, gmsh)
        face, m, findings, table = _analyse(gmsh, plan, claim_set)
        errors = sorted({f.code for f in findings if f.level == "error"})
        if errors:
            raise LibraryError(f"library entry '{entry.name}' (preset {preset}) builds with lint errors "
                               f"{', '.join(errors)}; fix the entry's numbers before regenerating its golden")
        if table is not None and table.failing():
            failing = ", ".join(r.id for r in table.failing())
            raise LibraryError(f"library entry '{entry.name}' (preset {preset}) fails its own claims {failing}; "
                               "the entry's CLAIMS describe the shape it builds, so one of the two is wrong")
        span = max(max(m.extent), 1e-9)
        outline = preview.outline_polyline(gmsh, face, span / OUTLINE_POINTS)
        if into is not None:
            into = Path(into)
            into.mkdir(parents=True, exist_ok=True)
            preview.draw(gmsh, face, plan, m, findings, table, into / f"{entry.name}.png", None)
        gmsh_version = gmsh.option.getString("General.Version")
    measurements = {k: v for k, v in m.as_dict().items() if k in GOLDEN_MEASUREMENT_KEYS}
    measurements["params"] = params   # report.reference_lines prints the entry's numbers from here
    golden = {
        "entry": entry.name, "preset": preset, "params": params, "source_sha": source_sha(entry),
        "measurements": _json_safe(measurements),
        "outline": [[[float(x), float(y)] for x, y in loop] for loop in outline],
        "compliance": None if table is None else _json_safe(table.as_dict()),
        "lint": [f.as_dict() for f in findings],
        "gmsh": gmsh_version,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if into is not None:
        (into / f"{entry.name}.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")
    return golden


def hausdorff(a: list, b: list) -> float:
    """Symmetric, numpy, chunked at 500 rows above 5000 points: preview's, which the
    caption prints, so a golden check and a lap's REFERENCE line measure the same thing."""
    return preview.hausdorff(a, b)


def measurements_agree(a: dict, b: dict, rel: float) -> bool:
    """The same keys, every number within `rel` (relative, with `rel` as the floor near
    zero), counts and words exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(measurements_agree(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(measurements_agree(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=rel)
    return a == b


def shas(golden: dict) -> dict[str, str]:
    """The three shas an approval records (4.6): the entry's source, and the golden's
    measurements and outline as JSON with sorted keys."""
    def sha(value) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
    return {"source_sha": str(golden.get("source_sha", "")), "measurements_sha": sha(golden.get("measurements")),
            "outline_sha": sha(golden.get("outline"))}


def span_of(golden: dict) -> float:
    """The larger extent of a golden: what its Hausdorff tolerance is relative to."""
    extent = (golden.get("measurements") or {}).get("extent") or [1.0, 1.0]
    return float(max(extent))


def signature_defaults(entry: Entry) -> dict[str, object]:
    """`build`'s parameters that carry a default, for the test that no geometric number
    is one (D16): `units` and a direction word may default, a length or an angle may not."""
    return {name: p.default for name, p in inspect.signature(entry.module.build).parameters.items()
            if p.default is not inspect.Parameter.empty}
