"""The library: parameterised shapes a person approved, with their goldens.

An entry is code with no default geometric numbers (D16: no number in the repo comes
from nowhere); `match` returns only entries whose approval exists and whose `source_sha`
matches the file now (D32), so nothing unapproved is ever shown to the model as a
reference. Goldens pin the outline polyline and the measurements; the PNG is
regenerated and never diffed (D17). Skeleton (U0): the types and the signatures; U6
implements them and adds tesla_valve.py, serpentine.py, t_junction.py and golden/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .. import claims, compile, lint, measure, preview  # noqa: F401
from ..sketch import Sketch  # noqa: F401

_U6 = ("not built in the U0 skeleton: U6 (library + goldens) implements it against "
       "DESIGN.md section 3.16")

GOLDEN: Path = Path(__file__).resolve().parent / "golden"
"""<name>.json, <name>.png, APPROVALS.json."""


@dataclass
class Entry:
    name: str
    module: ModuleType
    source: str
    keywords: tuple[str, ...]
    presets: dict[str, dict]
    claims: dict | None

    def build(self, preset: str | None = None, **params) -> Sketch:
        raise NotImplementedError(_U6)

    def as_dict(self) -> dict:
        """The module by its name: a module object does not round-trip through JSON."""
        return {"name": self.name, "module": getattr(self.module, "__name__", str(self.module)),
                "source": self.source, "keywords": list(self.keywords), "presets": self.presets,
                "claims": self.claims}

    @staticmethod
    def from_dict(d: dict) -> "Entry":
        """Needs the entry's module imported: `get(name)` is the way back from a record."""
        raise NotImplementedError(_U6)


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


def entries() -> list[Entry]:
    raise NotImplementedError(_U6)


def get(name: str) -> Entry:
    raise NotImplementedError(_U6)


def match(request: str) -> tuple[Entry, str] | None:
    """Lower-cased keyword hit -> (entry, the preset nearest the request's numbers), for
    APPROVED entries only: `approval(entry) is not None and approval["source_sha"] ==
    source_sha(entry)` (D32); an unapproved entry is never matched, whatever the request
    says. Shown to the model as: 'a person approved this shape on <date> from <SOURCE>;
    its parameters are the library's, the request's numbers override them'."""
    raise NotImplementedError(_U6)


def approval(entry: Entry) -> dict | None:
    """golden/APPROVALS.json[name]."""
    raise NotImplementedError(_U6)


def source_sha(entry: Entry) -> str:
    raise NotImplementedError(_U6)


def regenerate(entry: Entry, preset: str | None = None, into: Path | None = None) -> dict:
    """Builds, measures, draws; returns the golden dict of 4.5 (and writes it when `into` is given)."""
    raise NotImplementedError(_U6)


def hausdorff(a: list, b: list) -> float:
    """Symmetric, numpy, chunked at 500 rows above 5000 points."""
    raise NotImplementedError(_U6)


def measurements_agree(a: dict, b: dict, rel: float) -> bool:
    raise NotImplementedError(_U6)
