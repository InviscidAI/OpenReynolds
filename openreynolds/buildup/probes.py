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

OVER_LIMIT = "over-limit"
"""The inputs are present and too large to count. Distinct from `n/a` on purpose.

The sweep that found this reported a 695,308-triangle surface in the same word, and with
an indistinguishable reason string, as a case that exported no surface at all. "Too big
to check" and "nothing to check" are opposite facts about a run and neither the gate nor
a reader could tell them apart."""

UNTESTED = "untested"
"""The probe ran and its own denominator came back zero, so it measured nothing.

`self_intersection` reported `measured` with `pairs_tested: 0` against a 20,480-triangle
sphere, in the same state and the same shape as a genuine clean reading over 1,448 tested
pairs. A count of zero out of zero is not a clean surface; it is no test."""

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
          "surface after the run, and count parity",
          state=ACTIVE),
    Probe("union_closure",
          "the exported patch set has free edges, so the meshed volume is not the one "
          "intended; nothing errors because an open surface is a legal STL",
          "open edges on the welded union, never per file -- individual patch files "
          "are open surfaces by construction and a per-file check passes nothing real",
          state=ACTIVE),
    Probe("normals",
          "inconsistent outward orientation turns a solid into a void for snappy, "
          "which meshes the complement without complaint",
          "edges walked twice in the same direction on the welded union",
          state=ACTIVE),
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
          "triangle-triangle intersection between non-neighbours on the union",
          state=ACTIVE),
)

BY_ID = {probe.id: probe for probe in REGISTRY}

SCALE_FACTOR = 100.0
"""How far off the stated extent counts as fired.

The failure is the millimetre one, which is a thousand. A hundred is the band around it
that cannot be anything else: a mesh the caller asked for in centimetres is not off by
two orders of magnitude, and a surface that is is not in the units anybody meant."""

TRIANGLE_LIMIT = 200_000
"""Above this the crossing sweep is skipped and says so. A probe that takes ten minutes
on a fine surface would be a supervisor that stops supervising.

The edge probes carry their own, larger ceiling -- `preflight.TOPOLOGY_TRIANGLE_LIMIT`,
400,000 -- so the two are not the same number and a surface can be over one and under the
other. Both are now reported as `over-limit` with the ceiling that stopped them, because
a reader has no way to guess either."""

PATCH_SET_CANDIDATES = ("constant/triSurface", "constant/geometry", ".")
"""Where a patch set is looked for, in order.

`constant/triSurface` is where snappyHexMesh wants it and where most of the corpus puts
it. Three cases exported perfectly good STLs to the case root instead -- their route was
gmsh to gmshToFoam, which never needs a triSurface directory -- and every surface probe
reported `n/a: does not exist` with eight files sitting on disk. A probe that reads one
path and calls every miss `n/a` is not measuring the corpus it was pointed at."""

_SURFACE_FILE = re.compile(r"([A-Za-z0-9_.+-]+\.stl)\b", re.IGNORECASE)

MESH_DICTS = ("system/snappyHexMeshDict", "system/meshDict", "system/surfaceFeatureDict")


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
    """The conventional location, kept as the name callers already use."""
    return Path(case) / "constant" / "triSurface"


def patch_set(case: Path) -> tuple[Path | None, str]:
    """Where this case's STLs actually are, and how that was decided.

    Returns the first candidate directory holding at least one `.stl`, so a desk that
    meshed through gmsh and left its patches in the case root is read rather than
    reported absent. `""` for the note when it is the conventional path, so the ordinary
    case says nothing extra."""
    case = Path(case)
    for rel in PATCH_SET_CANDIDATES:
        directory = case / rel if rel != "." else case
        try:
            if not directory.is_dir():
                continue
            if not any(child.suffix.lower() == ".stl" for child in directory.iterdir()):
                continue
        except OSError:
            continue
        note = "" if rel == "constant/triSurface" else f"patch set read from {rel}"
        return directory, note
    return None, ""


def named_surfaces(case: Path) -> tuple[set[str], str]:
    """The STL filenames the case's own mesh dictionary references.

    This is the difference between counting the surface that was meshed and counting
    whatever is lying in the directory. T22 declared with `brakeDisc.stl` -- a
    byte-for-byte duplicate of its five patch files, left behind by an abandoned cfMesh
    attempt -- still on disk; `snappyHexMeshDict` named only the five. Welding all six
    put every triangle in twice, which leaves the open-edge count at zero (each edge's
    reverse is always present) and makes every one of 62,208 edges look walked twice the
    same way. The mesh that shipped was built from the five."""
    case = Path(case)
    for rel in MESH_DICTS:
        path = case / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = {m.group(1) for m in _SURFACE_FILE.finditer(text)}
        if found:
            return found, rel
    return set(), ""


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
    so a probe that insisted on one would be dormant for the wrong reason.

    Two things happen before the read. The patch set is *located* rather than assumed,
    and it is *restricted* to the files the case's own mesh dictionary names when it
    names any. Both are repairs to readings this sweep proved wrong rather than
    improvements: see `patch_set` and `named_surfaces`."""
    union, _note = _union_read(case)
    return union


def _union_read(case: Path):
    """The union and one line about how it was assembled, for the probe to report.

    Restricting to the files the mesh dictionary names is a repair to a false reading
    (see `named_surfaces`), and it has to not become one. **A duplicate can only ever
    leave the open-edge count where it was** -- every edge of a doubled surface still has
    its reverse -- so dropping one never opens a surface that was closed. A patch the
    dictionary happens not to name, because snappy took the domain box from `blockMesh`
    instead, is load-bearing, and dropping that does open it.

    So the restriction proves itself: it is used unless it leaves *more* free edges than
    reading the directory whole, and T14 is why the check exists rather than the argument
    for it -- its `snappyHexMeshDict` names three of six genuinely closed patches, and
    restricting to them turned a true 0 into a false 158.
    """
    _audit, domain_probe, preflight, _surfaces = _toolbox()
    directory, where = patch_set(case)
    if directory is None:
        directory = tri_surface(case)  # let read_union raise its own refusal
    notes = [where] if where else []

    whole = domain_probe.read_union(directory)
    wanted, dict_rel = named_surfaces(case)
    used = {n for n in wanted if n in set(whole.files)}
    ignored = sorted(set(whole.files) - used) if used else []
    if not ignored:
        return whole, "; ".join(n for n in notes if n)

    restricted = domain_probe.read_union(directory, used)
    loose = lambda u: int(preflight.surface_topology(u.triangles).get("open_edges") or 0)
    if loose(restricted) > loose(whole):
        notes.append(f"{dict_rel} names only {len(used)} of {len(whole.files)} patch "
                     f"files and dropping the rest opens the surface, so the directory "
                     f"was read whole")
        return whole, "; ".join(n for n in notes if n)
    notes.append(f"{len(ignored)} file(s) in the directory that {dict_rel} does not "
                 f"name were not welded in: {', '.join(ignored)}")
    return restricted, "; ".join(n for n in notes if n)


def _no_surface(probe: str, case: Path, exc: Exception) -> ProbeResult:
    directory, _where = patch_set(case)
    where = directory if directory is not None else tri_surface(case)
    return ProbeResult(probe, NOT_APPLICABLE,
                       f"no readable patch set under {where}: {exc}")


def _topology(case: Path) -> tuple[dict[str, Any], Any, str]:
    _audit, _domain, preflight, _surfaces = _toolbox()
    union, note = _union_read(case)
    return preflight.surface_topology(union.triangles), union, note


def _over_limit(probe: str, topology: dict[str, Any], note: str) -> ProbeResult:
    """A surface too large to count, said in a word that is not `n/a`."""
    _audit, _domain, preflight, _surfaces = _toolbox()
    count = int(topology.get("triangles") or 0)
    why = (f"{count:,} triangles is over this probe's ceiling of "
           f"{preflight.TOPOLOGY_TRIANGLE_LIMIT:,}, so its edges were not counted -- "
           "the surface is present and unmeasured, not absent")
    return ProbeResult(probe, OVER_LIMIT, _joined(why, note),
                       {"triangles": count,
                        "limit": int(preflight.TOPOLOGY_TRIANGLE_LIMIT)})


def _joined(why: str, note: str) -> str:
    return f"{why} ({note})" if note else why


def _open_edge_detail(union, triangles) -> dict[str, Any]:
    """Which files the open edges sit on, and the box they occupy.

    A count is not a lead. T12 was told its union had 2,177 free edges and spent nine
    cells -- STL byte inspection, two re-tessellations, feature-edge extraction, a z
    histogram, an OCCT adjacency walk -- working out where they were, settled on a cause
    a tolerance sweep refutes, and ran out of budget with the defect untouched. The
    numbers below are what those nine cells were reconstructing, and they come free:
    the edge walk has already computed everything this needs.
    """
    import numpy as np

    flat = np.asarray(triangles, dtype=float).reshape(-1, 3)
    span = float(np.max(flat.max(axis=0) - flat.min(axis=0)))
    if span <= 0:
        return {}
    welded = np.round(flat / span, 9)
    unique, index = np.unique(welded, axis=0, return_inverse=True)
    corners = np.asarray(index, dtype=np.int64).reshape(-1, 3)
    vertices = int(corners.max()) + 1

    triangle_of = np.tile(np.arange(len(corners)), 3)
    starts = np.concatenate([corners[:, 0], corners[:, 1], corners[:, 2]])
    ends = np.concatenate([corners[:, 1], corners[:, 2], corners[:, 0]])
    keep = starts != ends
    starts, ends, triangle_of = starts[keep], ends[keep], triangle_of[keep]
    low, high = np.minimum(starts, ends), np.maximum(starts, ends)
    _edges, inverse, counts = np.unique(low * vertices + high, return_inverse=True,
                                        return_counts=True)
    loose = counts[inverse] == 1
    if not loose.any():
        return {}

    detail: dict[str, Any] = {}
    if union.counts and len(union.counts) == len(union.files):
        owner = np.concatenate([np.full(n, i) for i, n in enumerate(union.counts)])
        hit = owner[triangle_of[loose]]
        per_file = {}
        for i, name in enumerate(union.files):
            n = int((hit == i).sum())
            if n:
                per_file[name] = {"open_edges": n, "triangles": int(union.counts[i])}
        detail["by_file"] = dict(sorted(per_file.items(),
                                        key=lambda kv: -kv[1]["open_edges"]))
    points = unique[np.concatenate([low[loose], high[loose]])] * span
    detail["bounds_m"] = [[round(float(v), 6) for v in points.min(axis=0)],
                          [round(float(v), 6) for v in points.max(axis=0)]]
    return detail


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
        topology, union, note = _topology(case)
    except Exception as exc:  # noqa: BLE001
        return _no_surface("union_closure", case, exc)
    if not topology.get("computed"):
        if int(topology.get("triangles") or 0):
            return _over_limit("union_closure", topology, note)
        return ProbeResult("union_closure", NOT_APPLICABLE,
                           _joined(str(topology.get("note", "")), note), dict(topology))
    measured = {"open_edges": topology["open_edges"], "files": list(union.files),
                "triangles": topology["triangles"]}
    where = ""
    if topology["open_edges"]:
        try:
            detail = _open_edge_detail(union, union.triangles)
        except Exception:  # noqa: BLE001 - a missing lead is not a failed probe
            detail = {}
        if detail:
            measured.update(detail)
            worst = list(detail.get("by_file", {}).items())[:3]
            if worst:
                where = " -- on " + ", ".join(
                    f"{name} ({row['open_edges']:,} of {row['triangles']:,} triangles)"
                    for name, row in worst)
    # Open is not the same as wrong: a baffle is a zero-thickness wall on purpose, and
    # snappy meshes one deliberately. The corpus happens to hold only closed solids,
    # which is why this looked like an invariant.
    return ProbeResult("union_closure", MEASURED,
                       _joined(f"{topology['open_edges']:,} free edges on the welded "
                               f"union of {topology['triangles']:,} triangles across "
                               f"{len(union.files)} file(s){where}", note), measured)


def _normals(case: Path, spec: dict[str, Any]) -> ProbeResult:
    try:
        topology, _union_obj, note = _topology(case)
    except Exception as exc:  # noqa: BLE001
        return _no_surface("normals", case, exc)
    if not topology.get("computed"):
        if int(topology.get("triangles") or 0):
            return _over_limit("normals", topology, note)
        return ProbeResult("normals", NOT_APPLICABLE,
                           _joined(str(topology.get("note", "")), note), dict(topology))
    measured = {"flipped_edges": topology["flipped_edges"],
                "triangles": topology.get("triangles")}
    # Counted over the whole triSurface directory, which is not necessarily the surface
    # meshDict named: T4's 132 flipped edges were one surface present twice, and the mesh
    # came from blockMesh regardless. The count is true; what it implies is not ours.
    return ProbeResult("normals", MEASURED,
                       _joined(f"{topology['flipped_edges']:,} edges walked twice the "
                               "same way on the welded union of the patch set", note),
                       measured)


def _coverage(case: Path, spec: dict[str, Any]) -> ProbeResult:
    """Is every face of the exported surface in exactly one patch?

    Was gated on a `patches.json` the core desk does not write, so it returned `n/a`
    on 26 of 26 cases in this corpus and on every case of every sweep before it -- a
    column of probe ids that read as coverage and screened nothing. The manifest is
    not what makes the question askable: one STL per patch is already an assignment,
    and `union_closure` and `normals` have always read the directory on exactly that
    basis. So a directory with no manifest is audited against a manifest derived from
    it, and the reading says which kind it was, because the two answer different
    questions -- a derived one cannot know that a file on disk was never meant to be
    a patch.
    """
    cad_audit, _domain, _preflight, _surfaces = _toolbox()
    directory = tri_surface(case)
    declared = (directory / "patches.json").is_file()
    try:
        manifest = None if declared else cad_audit.derived_manifest(directory)
        if manifest is not None and not manifest.get("patches"):
            return ProbeResult("coverage", NOT_APPLICABLE,
                               f"no patch files under {directory}, so there is no "
                               "partition to check")
        report = cad_audit.audit(directory, manifest=manifest)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("coverage", NOT_APPLICABLE, f"{type(exc).__name__}: {exc}")
    source = ("the patch set declared in patches.json" if declared
              else "the patch set the directory itself declares, one STL per patch; "
                   "whether every file there was meant to be a patch is not read here")
    for row in report.get("findings", []):
        if row.get("check") == "coverage":
            measured = dict(row)
            measured["manifest"] = "declared" if declared else cad_audit.DERIVED_SOURCE
            return ProbeResult("coverage", MEASURED,
                               _joined(str(row.get("measured", "")), source), measured)
    return ProbeResult("coverage", NOT_APPLICABLE,
                       "the audit reported no coverage finding")


def _scale(case: Path, spec: dict[str, Any]) -> ProbeResult:
    stated = spec.get("extent_m")
    if not stated:
        # Was "the case states no dimension", which is false on every case that states
        # one in prose -- T8, T9, T14 and T24 all do. What is missing is the spec field.
        return ProbeResult("scale", NOT_APPLICABLE,
                           "no extent_m was passed in the spec for this case, so there "
                           "is no stated dimension to measure the surface against; this "
                           "probe compares against the request and something has to "
                           "carry the request's number to it")
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
        return ProbeResult("self_intersection", OVER_LIMIT,
                           f"{len(triangles):,} triangles is over this probe's ceiling "
                           f"of {TRIANGLE_LIMIT:,}, so no pair was tested -- the surface "
                           "is present and unmeasured, not absent",
                           {"triangles": int(len(triangles)),
                            "limit": int(TRIANGLE_LIMIT)})
    corners, _vertices = surfaces.weld(triangles)
    crossings = cad_audit.crossing_pairs(triangles, corners)
    measured = {"pairs": len(crossings["pairs"]),
                "triangles": len(crossings["triangles"]),
                "pairs_tested": crossings["tested"]}
    if not crossings["tested"]:
        # Zero of zero is not a clean surface, and reporting it as one put a 20,480
        # triangle sphere in the same state and shape as a real 1,448-pair clean read.
        return ProbeResult("self_intersection", UNTESTED,
                           f"no triangle pair was tested on {len(triangles):,} "
                           "triangles, so this is not a reading either way", measured)
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
