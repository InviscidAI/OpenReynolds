"""The silent failures, measured from outside the run and told to nobody inside it.

Some defects never present as failures. Mesh the outside of the part and `checkMesh`
passes a perfectly valid mesh of the wrong volume. A leak in the exported surface gives
the wrong fluid volume and still checks clean. Neither can be discovered by waiting for
a failure, so neither can wait for the build-up loop to surface it.

So detection is separated from delivery. **These run in the supervisor**, out of band,
on the case as it stands on disk, and the result is graded into the run record only. The
agent is not told, the checks are not in its brief, and nothing is mounted in its
workspace -- `isolation.scan_run` is what proves the last part rather than a comment
claiming it.

**A probe is `dormant` until it fires.** On the first run that produces a silently-wrong
result of a kind, and not before, that criterion joins the finish check and its script
becomes reachable by the agent. A probe that never fires across the whole corpus stays
dormant forever, and that is the point: it is the tool rule made mechanical -- the agent
pays for a check only after the failure it catches has been observed to happen.

Nothing here reimplements geometry. `cad_audit.py`, `domain_probe.py`, `surfaces.py` and
`preflight.py` already do it and are tested; this module imports them, decides what
counts as fired, and stays out of their way. They are imported by path because they are
sibling scripts rather than a package, and they are imported **here**, in the
supervisor's process, which is the whole argument of §2.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLBOX = Path(__file__).resolve().parents[1] / "toolbox"

DORMANT = "dormant"
ACTIVE = "active"

MEASURED = "measured"
"""The probe got its numbers. Whether they mean the mesh is wrong is not its call."""

NOT_APPLICABLE = "n/a"
"""The probe's inputs are not present in this case -- no triSurface, no meshing point,
no stated dimension. Not a pass, not a failure, and no longer called `skipped`, because
`skipped` in a report reads as a check that was run and found nothing."""

ERROR = "error"

FIRED = "fired"
PASS = "pass"
SKIPPED = "skipped"
"""The verdicts, kept only so an old record still reads. Nothing produces them now.

**The first baseline is why.** Three probes fired across eight cases and the supervisor
overturned all three: the point 0.312 m outside T2's surface was a correct external-flow
case, T4's 132 flipped edges were a directory holding one surface twice while meshDict
named one file, and neither mesh was the one being complained about. Meanwhile the two
runs that really did go wrong -- T6 guessing a unit and shipping a mesh checkMesh passed,
T3 burning 28 cells to a mesh that never got a verdict -- produced no probe signal at all,
because five of eight cases never write a triSurface for a probe to read.

Every measurement was right and every verdict was wrong, which is the whole finding. A
probe can count edges; it cannot know whether the surface it counted is the one that was
meshed, or whether outside is where the seed point belonged. That is the case's intent,
and the supervisor reads the case."""


@dataclass(frozen=True)
class Probe:
    """A row of the registry in `docs/cad-silent-failures.md`, in code."""

    id: str
    catches: str
    detect: str
    state: str = DORMANT


@dataclass
class ProbeResult:
    """What one probe measured on one case. `measured` is the part with authority.

    `why` says what was measured, in the measurement's own terms. It does not say what
    the measurement implies -- the probe has no way to know."""

    id: str
    state: str = NOT_APPLICABLE
    why: str = ""
    measured: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "state": self.state, "why": self.why,
                "measured": self.measured}


REGISTRY: tuple[Probe, ...] = (
    Probe("location_in_mesh",
          "the meshing point sits outside the fluid, so the mesh is of the volume "
          "around the part; snappy succeeds and checkMesh passes a valid mesh of the "
          "wrong volume",
          "cast a ray from the point the case actually used against the exported "
          "surface after the run, and count parity"),
    Probe("union_closure",
          "the exported patch set has free edges, so the meshed volume is not the one "
          "intended; nothing errors because an open surface is a legal STL",
          "open edges on the welded union, never per file -- individual patch files "
          "are open surfaces by construction and a per-file check passes nothing real"),
    Probe("normals",
          "inconsistent outward orientation turns a solid into a void for snappy, "
          "which meshes the complement without complaint",
          "edges walked twice in the same direction on the welded union"),
    Probe("coverage",
          "a face assigned to two patches, or to none; an unassigned face lands "
          "silently in a default patch and takes whatever boundary condition it carries",
          "cad_audit's coverage finding, which needs the patch manifest to know which "
          "face was meant to be whose"),
    Probe("scale",
          "the case is a factor of a thousand from the dimensions the request stated -- "
          "millimetres read as metres -- and every number downstream is self-consistent",
          "the union's largest extent against the extent the case states, which means "
          "the case has to state one"),
    Probe("self_intersection",
          "the exported surface crosses itself, so the volume it bounds is not "
          "well defined and the mesher resolves it silently either way",
          "triangle-triangle intersection between non-neighbours on the union"),
)

BY_ID = {probe.id: probe for probe in REGISTRY}

SCALE_FACTOR = 100.0
"""How far off the stated extent counts as fired.

The failure is the millimetre one, which is a thousand. A hundred is the band around it
that cannot be anything else: a mesh the caller asked for in centimetres is not off by
two orders of magnitude, and a surface that is is not in the units anybody meant."""

TRIANGLE_LIMIT = 200_000
"""Above this the crossing sweep is skipped and says so. A probe that takes ten minutes
on a fine surface would be a supervisor that stops supervising."""


def _toolbox():
    """The three scripts, imported into the supervisor's process and no other."""
    if str(TOOLBOX) not in sys.path:
        sys.path.insert(0, str(TOOLBOX))
    import cad_audit  # noqa: E402  (sibling script, not a package)
    import domain_probe  # noqa: E402
    import preflight  # noqa: E402
    import surfaces  # noqa: E402
    return cad_audit, domain_probe, preflight, surfaces


_POINT = re.compile(
    r"\b(?:locationInMesh|insidePoint)\s*\(?\s*"
    r"(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)?")


def meshing_point(case: Path) -> tuple[list[float] | None, str]:
    """The point the case actually used, off its own dictionary or its manifest.

    Off the case rather than off the run's conversation: what was meshed is what the
    dictionary said, and a desk that talked about one point and wrote another is
    exactly the run this probe is for."""
    for rel in ("system/snappyHexMeshDict", "system/meshDict",
                "constant/triSurface/patches.json"):
        path = Path(case) / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix == ".json":
            try:
                point = json.loads(text).get("location_in_mesh")
            except ValueError:
                point = None
            if isinstance(point, (list, tuple)) and len(point) == 3:
                return [float(v) for v in point], rel
            continue
        match = _POINT.search(text)
        if match:
            return [float(v) for v in match.groups()], rel
    return None, ""


def tri_surface(case: Path) -> Path:
    return Path(case) / "constant" / "triSurface"


def run_all(case: Path, spec: dict[str, Any] | None = None) -> list[ProbeResult]:
    """Every probe against one case directory, as instruments rather than judges.

    Never raises: a probe whose inputs are absent records `n/a` with the reason, which
    is not a pass and is not a finding. What the numbers mean is the supervisor's to
    say, with the case in front of it."""
    spec = dict(spec or {})
    results: list[ProbeResult] = []
    for probe in REGISTRY:
        try:
            results.append(_ONE[probe.id](Path(case), spec))
        except Exception as exc:  # noqa: BLE001 - a probe failing is a probe result
            results.append(ProbeResult(probe.id, ERROR, f"{type(exc).__name__}: {exc}"))
    return results


def measured(results: list[ProbeResult]) -> list[str]:
    """The ids that came back with numbers, which is what there is to read."""
    return [result.id for result in results if result.state == MEASURED]


def fired(results: list[ProbeResult]) -> list[str]:
    """Always empty. Kept so callers that ask do not break while they are updated.

    Nothing fires any more: a probe reports what it measured and the supervisor decides
    what it means. See the note on `FIRED` for what the first baseline showed."""
    return [result.id for result in results if result.state == FIRED]


# -- one probe each ----------------------------------------------------------------


def _union(case: Path):
    """The exported surface as one welded union, manifest or no manifest.

    `domain_probe.read_union` reads a triSurface directory without requiring
    `patches.json`, which matters here: the core desk of §1 has no per-patch manifest,
    so a probe that insisted on one would be dormant for the wrong reason."""
    _audit, domain_probe, _preflight, _surfaces = _toolbox()
    return domain_probe.read_union(tri_surface(case))


def _no_surface(probe: str, case: Path, exc: Exception) -> ProbeResult:
    return ProbeResult(probe, NOT_APPLICABLE,
                       f"no readable patch set under {tri_surface(case)}: {exc}")


def _topology(case: Path) -> tuple[dict[str, Any], Any]:
    _audit, _domain, preflight, _surfaces = _toolbox()
    union = _union(case)
    return preflight.surface_topology(union.triangles), union


def _location_in_mesh(case: Path, spec: dict[str, Any]) -> ProbeResult:
    _audit, domain_probe, _preflight, _surfaces = _toolbox()
    point, source = meshing_point(case)
    if point is None:
        return ProbeResult("location_in_mesh", NOT_APPLICABLE,
                           "the case names no locationInMesh, so nothing was meshed by "
                           "a point and there is no point to test")
    try:
        union = _union(case)
    except Exception as exc:  # noqa: BLE001
        return _no_surface("location_in_mesh", case, exc)
    labels, distances = domain_probe.classify(union, [point])
    label = labels[0]
    clearance = abs(float(distances[0]))
    measured = {"point": point, "source": source, "classification": label,
                "clearance_m": clearance}
    # Where the point is, and nothing about whether that is right. Outside is correct for
    # an external-flow case and wrong for an internal one, and the surface alone cannot
    # tell those apart -- T2 of the first baseline was called a failure for being right.
    return ProbeResult("location_in_mesh", MEASURED,
                       f"the meshing point from {source} is {label} the exported "
                       f"surface, {clearance:.4g} m clear", measured)


def _union_closure(case: Path, spec: dict[str, Any]) -> ProbeResult:
    try:
        topology, union = _topology(case)
    except Exception as exc:  # noqa: BLE001
        return _no_surface("union_closure", case, exc)
    if not topology.get("computed"):
        return ProbeResult("union_closure", NOT_APPLICABLE, str(topology.get("note", "")),
                           dict(topology))
    measured = {"open_edges": topology["open_edges"], "files": list(union.files),
                "triangles": topology["triangles"]}
    # Open is not the same as wrong: a baffle is a zero-thickness wall on purpose, and
    # snappy meshes one deliberately. The corpus happens to hold only closed solids,
    # which is why this looked like an invariant.
    return ProbeResult("union_closure", MEASURED,
                       f"{topology['open_edges']:,} free edges on the welded union of "
                       f"{topology['triangles']:,} triangles across "
                       f"{len(union.files)} file(s)", measured)


def _normals(case: Path, spec: dict[str, Any]) -> ProbeResult:
    try:
        topology, _union_obj = _topology(case)
    except Exception as exc:  # noqa: BLE001
        return _no_surface("normals", case, exc)
    if not topology.get("computed"):
        return ProbeResult("normals", NOT_APPLICABLE, str(topology.get("note", "")),
                           dict(topology))
    measured = {"flipped_edges": topology["flipped_edges"],
                "triangles": topology.get("triangles")}
    # Counted over the whole triSurface directory, which is not necessarily the surface
    # meshDict named: T4's 132 flipped edges were one surface present twice, and the mesh
    # came from blockMesh regardless. The count is true; what it implies is not ours.
    return ProbeResult("normals", MEASURED,
                       f"{topology['flipped_edges']:,} edges walked twice the same way "
                       "on the welded union of the triSurface directory", measured)


def _coverage(case: Path, spec: dict[str, Any]) -> ProbeResult:
    cad_audit, _domain, _preflight, _surfaces = _toolbox()
    directory = tri_surface(case)
    if not (directory / "patches.json").is_file():
        return ProbeResult("coverage", NOT_APPLICABLE,
                           "no patch manifest, so nothing declares which face was meant "
                           "to be whose and double-assignment is unmeasurable")
    try:
        report = cad_audit.audit(directory)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("coverage", NOT_APPLICABLE, f"{type(exc).__name__}: {exc}")
    for row in report.get("findings", []):
        if row.get("check") == "coverage":
            return ProbeResult("coverage", MEASURED, str(row.get("measured", "")),
                               dict(row))
    return ProbeResult("coverage", NOT_APPLICABLE,
                       "the audit reported no coverage finding")


def _scale(case: Path, spec: dict[str, Any]) -> ProbeResult:
    stated = spec.get("extent_m")
    if not stated:
        return ProbeResult("scale", NOT_APPLICABLE,
                           "the case states no dimension, and this probe is measured "
                           "against the request -- which means the request must state one")
    try:
        union = _union(case)
    except Exception as exc:  # noqa: BLE001
        return _no_surface("scale", case, exc)
    lower, upper = union.lower, union.upper
    extent = max(float(upper[i] - lower[i]) for i in range(3))
    ratio = extent / float(stated) if stated else 0.0
    measured = {"extent_m": extent, "stated_m": float(stated), "ratio": ratio}
    return ProbeResult("scale", MEASURED,
                       f"the surface spans {extent:.4g} m against {float(stated):.4g} m "
                       f"stated -- a factor of {ratio:.4g}", measured)


def _self_intersection(case: Path, spec: dict[str, Any]) -> ProbeResult:
    cad_audit, _domain, _preflight, surfaces = _toolbox()
    try:
        union = _union(case)
    except Exception as exc:  # noqa: BLE001
        return _no_surface("self_intersection", case, exc)
    triangles = union.triangles
    if len(triangles) > TRIANGLE_LIMIT:
        return ProbeResult("self_intersection", NOT_APPLICABLE,
                           f"{len(triangles):,} triangles is over the sweep's limit of "
                           f"{TRIANGLE_LIMIT:,}", {"triangles": int(len(triangles))})
    corners, _vertices = surfaces.weld(triangles)
    crossings = cad_audit.crossing_pairs(triangles, corners)
    measured = {"pairs": len(crossings["pairs"]),
                "triangles": len(crossings["triangles"]),
                "pairs_tested": crossings["tested"]}
    # Coincident faces where two parts touch cross too, so a non-zero count is a number
    # to read against the geometry rather than a verdict on it.
    return ProbeResult("self_intersection", MEASURED,
                       f"{len(crossings['pairs']):,} of {crossings['tested']:,} tested "
                       "triangle pairs cross", measured)


_ONE = {
    "location_in_mesh": _location_in_mesh,
    "union_closure": _union_closure,
    "normals": _normals,
    "coverage": _coverage,
    "scale": _scale,
    "self_intersection": _self_intersection,
}
