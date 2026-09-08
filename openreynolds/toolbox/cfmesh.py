#!/usr/bin/env python3
"""The other mesher on this image, and the four traps between you and it.

snappyHexMesh shrinks the mesh, inserts prisms into the void it left, and deletes
them again wherever a quality metric fails -- a loop, run `nLayerIter` times. On the
Wigley hull on 2026-08-31 that loop reached 88.6% layer coverage on its FIRST growth
iteration and eroded to 66.7% by its fiftieth, and it printed 66.7% and exited 0.
Four rounds read that as a dictionary problem and re-tuned. It is not one: cfMesh's
`cartesianMesh` extrudes a boundary-layer sheet over the patch and optimises quality
AFTERWARDS, so there is no loop that can un-extrude a face, and on the same hull it
covered 100.0% of the wall. The measurement behind both numbers is `layer_report.py`
next to this file -- one metric applied to every mesh, which is the only reason the
two are comparable.

cfMesh is already here. It is built into the OpenFOAM ESI v2512 image as part of the
`openfoam2512:amd64` package (`dpkg -S $(which cartesianMesh)`, run on the instance on
2026-08-31), so it needs no build, no volume and no exports; `where` below
says whether that is still true of the image you are on. It was available and unused
for four rounds, which is the whole reason this script exists.

What it will not do is stated where it is chosen rather than discovered afterwards:
cfMesh has **no first-layer target and so no y+ request**. The stack is sized from
the local cell -- the first cell comes out as `edge / (1 + r + ... + r^(n-1))`
(`refineBoundaryLayersFunctions.C:688-696` in the v2512 source) -- and
`maxFirstLayerThickness` enters only as `min(max(cap, SMALL), that)` at `:709-714`,
a cap that can thin the first cell but can never place it at a stated height. The 1.21 mm first cell measured
on the Wigley hull is a consequence of the cell size, not a target that was hit. If
the study needs a wall spacing chosen by y+, snappy's layer spec is the tool that
takes one, and its coverage is the thing to measure rather than assume.

The four traps each cost a round of the comparison, all four are silent, and three of
the four end in exit code 0:

1. `surfaceFeatureEdges` renames every patch: it splits each solid at the feature
   angle and names the pieces `<solid>_<index>`, where the index is the GLOBAL new-patch
   counter and not a per-solid one (`triSurfacePatchManipulator.C:120-122`). A hull
   written first into a hull-plus-box STL therefore comes out `hull_0`, `hull_1`, ...
   and the box faces after it as `xMin_3`, `xMax_4`, ... -- which is why a key has to
   be the REGEX `"hull_.*"` rather than a guessed index, and why `renameBoundary` puts
   the name back. cfMesh matches with `findMatchingStrings(regExp(name), ...)`;
   a mesh-side miss prints `Cannot find any patch names matching hull` as a *Warning*
   (`polyMeshGenFaces.C:281`) and a surface-side miss prints nothing at all, because
   the identical warning (`triSurfFacets.C:103`) sits under the `#ifdef DEBUGtriSurf`
   at `triSurfFacets.C:100`, which nothing in the tree defines. Either way it meshes at
   `maxCellSize` with no refinement and no layers and exits 0. The tell was a
   10,576-face mesh where 300,000 were expected.

   The coupling runs the other way too: keys and surface format decide each other. A
   raw `.stl` or `.fms` handed straight to `cartesianMesh` keeps its solid names
   verbatim, so THERE a literal key is the right one and `"hull_.*"` is the miss.
   `check` reads `surfaceFile` before it judges a key, because cfMesh's own tutorials
   mesh raw STLs with literal keys and they are not wrong.
2. `surfaceGenerateBoundingBox` writes an FMS the FMS reader cannot read (empty
   `geometricType` on the six generated patches; the reader then takes the next
   patch's name as this patch's type and dies inside the *point* list). `box` below
   writes the geometry and the domain box as one closed multi-solid STL instead.
3. cfMesh cannot read a snappyHexMesh mesh as written: snappy writes
   `constant/polyMesh` binary with `class faceCompactList` and cfMesh's reader wants
   ASCII `faceList`. `writeFormat ascii;` in `controlDict` **and then**
   `foamFormatConvert -constant`; converting first converts the mesh to what it
   already was and looks like it did nothing.
4. `generateBoundaryLayers` moves symmetry-plane points by ~1e-9 rad, and
   `symmetryPlanePolyPatch` demands planarity to machine precision -- after which
   `checkMesh`, `foamFormatConvert` and `postProcess` all refuse to *open* the mesh.
   Type `symmetry`, never `symmetryPlane`, on anything a layer will touch.

And one consequence that is not a trap so much as a default nobody reads:
`renameBoundary` with a `defaultName` merges every patch it was not told about into
ONE patch, added only if something is in fact left over (`renameBoundaryPatches.C:149-168`,
the `addPatch` test at `:151-161`; omitting `defaultName` takes the `else` branch at
`:169-180` and keeps every original name instead). So a
`case` written with `--wall hull` and nothing else gives a TWO-patch mesh -- `hull`
and `farField` -- with all six box faces merged into the second, and no inlet to
point a velocity at. Name the faces the case needs with `--patch xMin:patch`
(repeatable) or `--symmetry yMin`; whatever is left unnamed still merges, and
`case` prints which faces those are.

This is offered, not imposed. `where` and `check` read and print. `box` and `case`
write the two files they are asked for and refuse to overwrite. `prepare` prints its
repairs and applies them only with `--write`.

    python3 cfmesh.py where                     # is cfMesh on this instance
    python3 cfmesh.py box hull.stl --out hull_box.stl
    python3 cfmesh.py box hull.stl --out hull_box.stl --box -4.5 9 -3 3 -1.2 0.4
    python3 cfmesh.py case . --surface hull_box.stl --wall hull \\
                            --max-cell 0.15 --wall-cell 0.009375 --layers 3 \\
                            --patch xMin:patch --patch xMax:patch --symmetry yMin
    python3 cfmesh.py prepare <case>            # a snappy mesh made readable
    python3 cfmesh.py check <case> --wall hull --log log.cartesianMesh
    python3 cfmesh.py check <case> --wall hull --ref <the same case, no layers>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_report  # noqa: E402  (sibling script, not a package)
import preflight  # noqa: E402

Finding = preflight.Finding

# The four meshers and the one standalone layer tool, as the image ships them.
MESHERS = ("cartesianMesh", "cartesian2DMesh", "pMesh", "tetMesh")
LAYER_TOOL = "generateBoundaryLayers"
# The three the routes here actually run, not a category. The record groups the 25 by
# role (qa-runs/comp/cfmesh/README.md:20-26) and these do not fall in one group:
# `surfaceFeatureEdges` and `surfaceToFMS` are in its surface-prep row,
# `improveSymmetryPlanes` in its mesh-tools row. `surfaceGenerateBoundingBox` is in
# the surface-prep row too and is deliberately NOT probed -- it is trap 2, and the
# route replaced it with `box` rather than depending on it.
SURFACE_TOOLS = ("surfaceFeatureEdges", "surfaceToFMS", "improveSymmetryPlanes")

# Looking for these on $PATH and not finding them is not a broken install: in cfMesh
# as shipped with v2512 they are C++ classes inside `libmeshLibrary`, under
# `meshLibrary/utilities/boundaryLayers/`, called by the meshers and by
# `generateBoundaryLayers`. A session that goes looking for `refineBoundaryLayers` as
# a command and concludes cfMesh is half-installed has lost the round for nothing.
NOT_EXECUTABLES = ("detectBoundaryLayers", "extrudeLayer", "refineBoundaryLayers")

# The one line cfMesh prints when a dictionary key matched no patch. It is a Warning,
# the run continues, and the exit code is 0 -- so it has to be read out of the log on
# purpose or not at all.
UNMATCHED = re.compile(r"Cannot find any patch names matching\s+(\S+)")

# `surfaceFeatureEdges` appends an index to every solid name, so the solid `hull`
# never survives as `hull` once it has run. This is the selector that was measured to
# work; OpenFOAM's `regExp` is a WHOLE-string match, so `hull` as a pattern does not
# match `hull_0` -- and, in the other direction, `hull_.*` does not match a bare `hull`
# either, which is why the surface format decides which of the two is right.
SELECTOR = '"{name}_.*"'

# cfMesh's own reading of these two, from the v2512 source, both of which it accepts
# with a Warning and then does not carry out. Both citations are the PER-PATCH
# setters, which is what a `patchBoundaryLayers` entry goes through, and which is what
# this script writes:
#   nLayers < 2       "boundary layers disabled for this patch"
#                     setNumberOfLayersForPatch, refineBoundaryLayers.C:193-199
#   thicknessRatio<1  "thickness ratio for patch ... is less than 1.0", then `return`
#                     setThicknessRatioForPatch, refineBoundaryLayers.C:223-229
#
# The GLOBAL forms one screen up are the same for the ratio and NOT the same for the
# layer count, which is why `read_meshdict` reads the two apart and `check_meshdict`
# grades them differently (`warn` global, `fail` per-patch):
#   setGlobalThicknessRatio  :134 warns and `return`s at :150 -- same as per-patch,
#                            which also `return`s at :229.
#   setGlobalNumberOfLayers  :110 warns and `return`s at :127, leaving the global at
#                            its no-layers default; setNumberOfLayersForPatch warns at
#                            :193-199 and does NOT return, storing the value at :206.
# So a global `nLayers 1` means "no layers", which is what three of cfMesh's own
# tutorials mean by it, while a per-patch `nLayers 1` is a patch that asked for layers
# and gets none. Do not merge these two readings back together.
MIN_LAYERS = 2
MIN_RATIO = 1.0

# The box around a body, in multiples of its longest extent. The same rule of thumb as
# `templates/body_in_box.py` and the field notes: a wake needs more room than an
# approach. Overridden wholesale by `--box`, which is what a half model needs.
UPSTREAM, DOWNSTREAM, SIDE = 5.0, 10.0, 5.0

BOX_FACES = ("xMin", "xMax", "yMin", "yMax", "zMin", "zMax")


class OverwriteError(Exception):
    """An output file is already there and `--force` was not given.

    Its own class because it used to be a bare `SystemExit`, which exits 1 with no
    `refused:` prefix -- so the script had two shapes of failure after all, and the
    test that says otherwise was reading only one of them.
    """


class SurfaceError(Exception):
    """The surface could not be read, or is not one cfMesh can mesh a volume from."""


class DictError(Exception):
    """A meshDict was asked for that cfMesh would accept and quietly not carry out."""


# ------------------------------------------------------------------- where --


def _run(cmd: list[str]) -> str:
    """A command's stdout, or "" -- this is a probe and a missing tool is an answer."""
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def unresolved_libraries(binary: str) -> list[str] | None:
    """The shared objects `ldd` could not find, or None where it could not be asked.

    None is not zero. A binary that could not be analysed is unchecked, and saying so
    is the difference between "cfMesh works" and "cfMesh is here and I did not look".
    """
    out = _run(["ldd", binary])
    if not out:
        return None
    return [line.split("=>")[0].strip() for line in out.splitlines() if "not found" in line]


def where() -> dict[str, Any]:
    """What of cfMesh is on this instance, measured with `which`, `ldd` and `dpkg`."""
    found: dict[str, Any] = {"meshers": {}, "layers": {}, "surface": {}}
    for group, names in (
        ("meshers", MESHERS),
        ("layers", (LAYER_TOOL,)),
        ("surface", SURFACE_TOOLS),
    ):
        for name in names:
            path = shutil.which(name)
            found[group][name] = (
                None if path is None else {"path": path, "unresolved": unresolved_libraries(path)}
            )
    primary = found["meshers"]["cartesianMesh"]
    found["present"] = primary is not None
    found["package"] = ""
    if primary:
        # Which package owns it says whether it came with the image (and so survives
        # instance deletion) or was built onto the volume (and so does not).
        owner = _run(["dpkg", "-S", primary["path"]])
        found["package"] = owner.split(":")[0].strip() if owner else ""
    found["not_executables"] = list(NOT_EXECUTABLES)
    return found


def report_where(found: dict[str, Any]) -> str:
    lines: list[str] = []
    if not found["present"]:
        lines.append("cartesianMesh is NOT on this instance's PATH.")
        lines.append(
            "  It shipped inside the openfoam2512 package on the image used through"
        )
        lines.append(
            "  2026-08-31, so an image that lacks it is a change worth recording."
        )
        lines.append(
            "  A source build is the way back: qa-runs/comp/cfmesh/ carries the v2512"
        )
        lines.append("  plugin tree and the build plan.")
    else:
        pkg = found["package"] or "unknown package"
        lines.append(f"cartesianMesh: {found['meshers']['cartesianMesh']['path']}")
        lines.append(f"  owned by {pkg}")
        if pkg.startswith("openfoam"):
            lines.append(
                "  -- part of the image, so no exports, no volume, and it survives"
            )
            lines.append("     instance deletion.")
    for group, title in (
        ("meshers", "meshers"),
        ("layers", "standalone layer tool"),
        ("surface", "the surface and mesh tools these routes run"),
    ):
        lines.append(f"\n{title}:")
        for name, entry in found[group].items():
            if entry is None:
                lines.append(f"  {name:26s} absent")
                continue
            unresolved = entry["unresolved"]
            if unresolved is None:
                note = "present (libraries unchecked)"
            elif unresolved:
                note = "present, UNRESOLVED: " + ", ".join(unresolved)
            else:
                note = "present, libraries resolve"
            lines.append(f"  {name:26s} {note}")
    lines.append(
        "\nnot executables in this release, and never were -- they are classes inside"
    )
    lines.append("libmeshLibrary, called by the meshers:")
    lines.append("  " + ", ".join(found["not_executables"]))
    return "\n".join(lines)


# --------------------------------------------------------------- surfaces ---


class Solid(NamedTuple):
    """One named solid of an STL. The name becomes a cfMesh patch name, indexed."""

    name: str
    facets: list[tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]]


def read_stl(path: Path, name: str | None = None) -> list[Solid]:
    """An STL, ASCII or binary, as named solids.

    A BINARY STL carries no solid names at all -- the format has nowhere to put them.
    cfMesh takes its patch names from those names, so a binary body meshes into a wall
    you cannot address in `localRefinement` or `patchBoundaryLayers`, which is trap 1
    arriving from the geometry rather than from the dictionary. Hence `name`, and
    hence the refusal when it is not given.
    """
    raw = path.read_bytes()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:2048].lower():
        return _read_ascii_stl(raw.decode("utf-8", errors="replace"), path)
    if name is None:
        raise SurfaceError(
            f"{path.name} is a binary STL, which has no solid names, and cfMesh names"
            " its patches after them -- pass --name to say what this body is called"
        )
    return [Solid(name, _read_binary_stl(raw, path))]


def _read_binary_stl(raw: bytes, path: Path) -> list:
    if len(raw) < 84:
        raise SurfaceError(f"{path.name}: {len(raw)} bytes, too short to be an STL")
    (count,) = struct.unpack("<I", raw[80:84])
    if len(raw) < 84 + 50 * count:
        raise SurfaceError(
            f"{path.name}: header claims {count} facets, file holds {(len(raw) - 84) // 50}"
        )
    facets = []
    for i in range(count):
        v = struct.unpack("<12f", raw[84 + 50 * i : 84 + 50 * i + 48])
        facets.append((v[0:3], (v[3:6], v[6:9], v[9:12])))
    return facets


def _read_ascii_stl(text: str, path: Path) -> list[Solid]:
    solids: list[Solid] = []
    name, facets = "", []
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    verts: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        word = line.split()
        if not word:
            continue
        head = word[0].lower()
        if head == "solid":
            name = word[1] if len(word) > 1 else path.stem
            facets = []
        elif head == "facet":
            normal = tuple(float(x) for x in word[2:5]) if len(word) >= 5 else (0.0, 0.0, 0.0)
            verts = []
        elif head == "vertex":
            verts.append(tuple(float(x) for x in word[1:4]))
        elif head == "endfacet":
            if len(verts) == 3:
                facets.append((normal, tuple(verts)))
        elif head == "endsolid":
            solids.append(Solid(name or path.stem, facets))
            name, facets = "", []
    if name and facets:  # a file that never closed its last solid
        solids.append(Solid(name, facets))
    if not solids:
        raise SurfaceError(f"{path.name}: no solids read -- is it an STL?")
    return solids


def bounds(solids: list[Solid]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The extent of a set of solids. Metres, like everything else on the instance."""
    verts = [v for s in solids for _, tri in s.facets for v in tri]
    if not verts:
        raise SurfaceError("no vertices")
    lo = tuple(min(v[i] for v in verts) for i in range(3))
    hi = tuple(max(v[i] for v in verts) for i in range(3))
    return lo, hi


def domain_box(lo, hi, upstream=UPSTREAM, downstream=DOWNSTREAM, side=SIDE):
    """A flow box around a body, in multiples of the body's longest extent."""
    length = max(hi[i] - lo[i] for i in range(3))
    if length <= 0:
        raise SurfaceError("the body has no extent")
    return (
        (lo[0] - upstream * length, lo[1] - side * length, lo[2] - side * length),
        (hi[0] + downstream * length, hi[1] + side * length, hi[2] + side * length),
    )


def box_solids(lo, hi) -> list[Solid]:
    """The six faces of a box, each its own named solid, normals pointing outward."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    corners = {
        "xMin": ((-1.0, 0, 0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
        "xMax": ((1.0, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        "yMin": ((0, -1.0, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        "yMax": ((0, 1.0, 0), [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
        "zMin": ((0, 0, -1.0), [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]),
        "zMax": ((0, 0, 1.0), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
    }
    out = []
    for name in BOX_FACES:
        normal, p = corners[name]
        out.append(Solid(name, [(normal, (p[0], p[1], p[2])), (normal, (p[0], p[2], p[3]))]))
    return out


def enclose(body: list[Solid], lo, hi, tol_frac: float = 1e-6) -> list[str]:
    """Which box faces the body touches, refusing outright when it pokes through.

    cfMesh meshes the volume a closed surface encloses, so a body outside the box does
    not make a hole in it -- it makes a different problem, silently. The usual cause is
    an STL still in millimetres inside a box written in metres, which is the same
    mistake `preflight.py`'s STL-scale check exists for. A body that TOUCHES a face is
    the ordinary half model, and the face it touches is a cut plane: it needs
    `symmetry` on it, never `symmetryPlane` (trap 4).
    """
    blo, bhi = bounds(body)
    span = max(hi[i] - lo[i] for i in range(3))
    tol = tol_frac * span
    outside, touching = [], []
    for i, axis in enumerate("xyz"):
        if blo[i] < lo[i] - tol:
            outside.append(f"{axis} reaches {blo[i]:.6g}, below the box's {lo[i]:.6g}")
        elif blo[i] < lo[i] + tol:
            touching.append(f"{axis}Min")
        if bhi[i] > hi[i] + tol:
            outside.append(f"{axis} reaches {bhi[i]:.6g}, above the box's {hi[i]:.6g}")
        elif bhi[i] > hi[i] - tol:
            touching.append(f"{axis}Max")
    if outside:
        raise SurfaceError(
            "the body is not inside the box: " + "; ".join(outside)
            + " -- check the units before the box (an STL in millimetres inside a box"
            " in metres is the usual cause)"
        )
    return touching


def write_stl(solids: list[Solid], out: Path) -> None:
    """One closed multi-solid ASCII STL, LF-terminated whatever wrote it.

    The newline is not cosmetic. Every CRLF that broke a run in the cfMesh comparison
    arrived from editing a file with Windows Python -- `write_text` writes `\\r\\n`,
    and bash on the instance then answers `$'\\r': command not found`. Normalising at
    the point a file is written is the only place it stays fixed.
    """
    parts: list[str] = []
    for solid in solids:
        parts.append(f"solid {solid.name}\n")
        for normal, tri in solid.facets:
            parts.append("facet normal %.9g %.9g %.9g\n outer loop\n" % tuple(normal))
            for v in tri:
                parts.append("  vertex %.9g %.9g %.9g\n" % tuple(v))
            parts.append(" endloop\nendfacet\n")
        parts.append(f"endsolid {solid.name}\n")
    out.write_text("".join(parts), encoding="utf-8", newline="\n")


# --------------------------------------------------------------- meshDict ---


def selector(name: str) -> str:
    """The dictionary key that will actually match this solid AFTER feature edges.

    This is only the right key for a `.ftr`. Pointed at a raw STL it matches nothing,
    because the index it is written for does not exist yet -- see `feature_file`, and
    `check_meshdict`, which reads `surfaceFile` before it judges a key.

    A name already carrying a regex metacharacter is passed through as written: the
    caller has said what they mean and this is not the place to second-guess it.
    """
    if any(c in name for c in ".*[]()?+|^$\\"):
        return f'"{name}"'
    return SELECTOR.format(name=name)


def feature_file(surface: str) -> str:
    """What `surfaceFile` should name, given what the geometry is written as.

    An STL can be handed to `cartesianMesh` directly, and then its patch names are the
    solid names exactly as written -- but the route that was measured to work runs
    `surfaceFeatureEdges` first, because without feature edges the octree rediscovers
    every hard edge by refinement and rounds the ones it misses. The two decisions are
    coupled: feature edges index the names, so the dictionary keys must be regexes,
    and a dictionary whose keys are indexed regexes pointed at a raw STL matches
    nothing. This is where they are kept in step.

    `.ftr` rather than `.fms` for two reasons, and the second is the one that bites:
    an FMS written by `surfaceGenerateBoundingBox` on this release cannot be read back
    (trap 2), and `surfaceFeatureEdges` itself only runs its `triSurfacePatchManipulator`
    -- the pass that splits and indexes the patches -- when the output is NOT an FMS
    (the `else` branch of the format test at `surfaceFeatureEdges.C:86`, manipulator
    at `:93`). Writing FMS leaves the names exactly as they were, so
    the two output formats need opposite dictionary keys.
    """
    stem = Path(surface)
    if stem.suffix.lower() in (".stl", ".obj", ".ftr", ".fms", ".vtk", ".stlb"):
        return stem.with_suffix(".ftr").as_posix()
    return surface


HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      meshDict;
}}
// Written by cfmesh.py. cfMesh meshes the volume enclosed by a closed surface, so
// `surfaceFile` is the geometry AND the domain box: there is no locationInMesh and no
// blockMesh. It names the .ftr rather than the STL because the route runs
// `surfaceFeatureEdges` first -- and that is why every patch key below is a regex:
// feature edges split each solid at the feature angle and name the pieces
// `<solid>_<index>`, the index being the global new-patch counter and not a per-solid
// one (triSurfacePatchManipulator.C:120-122), so `{wall}` becomes `{wall}_0`,
// `{wall}_1`, ... and the box faces take the numbers after those. A literal name
// matches none of them, which cfMesh reports as a Warning before meshing at
// maxCellSize with no refinement and no layers, and exiting 0.
"""


def meshdict(
    surface: str,
    max_cell: float,
    wall: str | None = None,
    wall_cell: float | None = None,
    layers: int = 0,
    ratio: float = 1.2,
    wall_type: str = "wall",
    default_name: str = "farField",
    default_type: str = "patch",
    symmetry: tuple[str, ...] = (),
    patches: tuple[tuple[str, str], ...] = (),
) -> str:
    """A `system/meshDict` that carries out what it says it does.

    The refusals are the traps, encoded: a request cfMesh would accept, warn about and
    then not perform is refused here instead, where it costs a second rather than a
    meshing run and a study built on a mesh that has no layers on it.

    `patches` is `(solid name, patch type)` pairs and `symmetry` is the shorthand for
    the same thing with type `symmetry`. Everything NOT named in one of the three --
    wall, patches, symmetry -- is merged by `renameBoundary` into a single patch called
    `default_name`, because a `defaultName` that is present makes cfMesh map every
    unmatched patch into one patch (renameBoundaryPatches.C:149-168); it adds that
    patch only where something is actually left unmatched (the `addPatch` test at
    :151-161), so a dictionary that names every solid gets no `default_name` patch at
    all. That is the difference between a mesh with an inlet and an outlet and a mesh
    with one six-sided `farField`, and it is not visible anywhere in the run's own
    output.

    A solid may be named once. Naming it twice is refused rather than written, because
    the two entries carry the same key and OpenFOAM keeps the later one.
    """
    if max_cell <= 0:
        raise DictError("maxCellSize must be positive; cfMesh exits fatal otherwise")
    if wall_cell is not None and wall_cell <= 0:
        raise DictError("the wall cell size must be positive")
    if wall_cell is not None and wall_cell > max_cell:
        raise DictError(
            f"the wall cell ({wall_cell:g}) is coarser than maxCellSize ({max_cell:g}),"
            " so localRefinement would coarsen the wall rather than refine it"
        )
    if layers:
        if wall is None:
            raise DictError("layers were asked for with no wall patch to put them on")
        if layers < MIN_LAYERS:
            raise DictError(
                f"nLayers {layers} is below {MIN_LAYERS}: cfMesh warns 'boundary layers"
                " disabled for this patch' (refineBoundaryLayers.C:193-199) and meshes"
                " without them -- ask for 2 or more, or for none"
            )
        if ratio < MIN_RATIO:
            raise DictError(
                f"thicknessRatio {ratio:g} is below {MIN_RATIO:g}: cfMesh warns and"
                " returns, keeping its own value (refineBoundaryLayers.C:223-229), so"
                " the request would be silently dropped -- 1.0 or more, 1.2 being the"
                " value the Wigley comparison ran"
            )
    # Trap 4, refused at every place a type can enter. This was a check of the one
    # NAME `symmetryPlane` until 2026-09-08: `symmetry` holds face names whose type is
    # hardcoded to `symmetry` below, so it refused a correct dictionary whenever a user
    # called a face `symmetryPlane`, and it could not see the type trap it stood for,
    # because until `--patch NAME:TYPE` and `--default-type` existed no CLI flag set a
    # type at all. Naming a face `symmetryPlane` is not the mistake; typing one is.
    #
    # `wall_type` is named for the ARGUMENT and not for a flag, because there is no
    # `--wall` flag that sets it: `--wall` takes a name and the type stays `wall`. A
    # refusal that quotes a flag the CLI does not have is the same overclaim in
    # miniature, and only a Python caller can reach this one.
    typed = [("wall_type", wall_type), ("--default-type", default_type)]
    typed += [(f"--patch {name}", ptype) for name, ptype in patches]
    for flag, ptype in typed:
        if ptype == "symmetryPlane":
            raise DictError(
                f"{flag} asks for type symmetryPlane, which is the type that breaks:"
                " the layer pass moves its points by ~1e-9 rad and"
                " symmetryPlanePolyPatch demands machine-precision planarity, after"
                " which checkMesh, foamFormatConvert and postProcess all refuse to open"
                " the mesh and improveSymmetryPlanes declines the repair. Use type"
                " symmetry"
            )

    body = [HEADER.format(wall=wall or "the wall")]
    body.append(f'\nsurfaceFile "{feature_file(surface)}";\n')
    body.append(f"\nmaxCellSize {max_cell:g};\n")

    if wall and wall_cell:
        body.append(
            "\n// Refinement is matched on the SURFACE's patch names. A miss here is\n"
            "// completely silent: the identical warning is at triSurfFacets.C:103 but\n"
            "// sits under the #ifdef DEBUGtriSurf at :100, which nothing defines.\n"
            "localRefinement\n{\n"
            f"    {selector(wall)}\n    {{\n        cellSize {wall_cell:g};\n    }}\n}}\n"
        )

    if wall and layers:
        body.append(
            "\n// The sheet is extruded over the patch and the quality optimisation\n"
            "// runs afterwards, so nothing here can un-extrude a face. There is no\n"
            "// first-layer thickness to set: the stack is sized from the local cell.\n"
            "boundaryLayers\n{\n"
            "    patchBoundaryLayers\n    {\n"
            f"        {selector(wall)}\n        {{\n"
            f"            nLayers         {layers};\n"
            f"            thicknessRatio  {ratio:g};\n"
            "        }\n    }\n\n"
            "    optimiseLayer 1;\n}\n"
        )

    named = [(wall, wall_type)] if wall else []
    named += [(face, "symmetry") for face in symmetry]
    named += list(patches)
    # A solid named twice -- `--wall hull --patch hull:patch`, or `--symmetry yMin
    # --patch yMin:patch` -- writes the same `"hull_.*"` sub-dictionary into
    # `newPatchNames` twice. OpenFOAM warns on the duplicate keyword and the LATER
    # entry wins, so the wall silently comes out typed `patch` and the layer request
    # goes with it: a request accepted and quietly not carried out, which is the one
    # thing this function exists to refuse.
    seen: dict[str, str] = {}
    for name, ptype in named:
        if name in seen:
            raise DictError(
                f"{name} is named twice, as {seen[name]} and as {ptype}: both write the"
                f" same {selector(name)} key into newPatchNames, OpenFOAM warns on the"
                f" duplicate and the later one wins, so it would come out {ptype}"
                " -- name it once, with the type it should have"
            )
        seen[name] = ptype
    body.append(
        "\n// And the names put back. Without this the mesh's wall keeps the indexed\n"
        f"// name `{wall}_<n>` that surfaceFeatureEdges gave it -- `{wall}_0` if this\n"
        "// solid is written first, and the counter is global, so not otherwise --\n"
        "// and every later dictionary (fields, forces, y+) has to know it. Note what\n"
        "// defaultName does: because it is PRESENT, cfMesh maps every solid not named\n"
        "// below into ONE patch, which it adds only if some solid is in fact left\n"
        "// over (renameBoundaryPatches.C:149-168, the addPatch test at :151-161). So\n"
        "// the mesh gets exactly the patches named here, plus"
        f" `{default_name}` if anything\n// is unnamed and nothing extra if all of them"
        " are named. Named here: "
        + (", ".join(name for name, _ in named) or "nothing, so ALL of them merge")
        + ".\n"
        "renameBoundary\n{\n"
        f"    defaultName     {default_name};\n"
        f"    defaultType     {default_type};\n\n"
        "    newPatchNames\n    {\n"
    )
    for name, ptype in named:
        body.append(
            f"        {selector(name)}\n        {{\n"
            f"            type    {ptype};\n"
            f"            newName {name};\n        }}\n"
        )
    body.append("    }\n}\n")
    return "".join(body)


def route(surface: str, wall: str | None, feature_angle: float = 45.0) -> list[str]:
    """The commands, in the order that works, with the reason each one is there.

    A surface that is ALREADY a `.ftr` gets no `surfaceFeatureEdges` line. It used to
    get one, and it read `surfaceFeatureEdges -angle 45 domain.ftr domain.ftr`, which
    cannot run: `surfaceFeatureEdges.C:67-73` fatals with "Output file ... would
    overwrite the input file". Nothing was destroyed, but the first line of a printed
    route that fatals is worse than no line -- a route is copied, not read.
    """
    lines = []
    if Path(surface).suffix.lower() == ".ftr":
        lines.append(
            f"# {surface} is already a .ftr, so the feature-edge pass has run and the"
            " patches are\n#   indexed; running it again on itself is a fatal error"
            " (surfaceFeatureEdges.C:67-73)."
        )
    else:
        # `-angle` is real: `addOption("angle")` at surfaceFeatureEdges.C:61, and the
        # default of 45 is applied at :75-79, where it also prints "Using 45 deg as
        # default angle!". It is written out rather than left implicit because the
        # angle decides which edges the octree is told about, and a default that is
        # never seen is a default never revisited.
        lines.append(
            f"surfaceFeatureEdges -angle {feature_angle:g} {surface}"
            f" {Path(surface).stem}.ftr"
            "   # .ftr, not .fms -- trap 2, and only .ftr indexes the patches"
        )
    lines += [
        "cartesianMesh                                # reads system/meshDict",
        "checkMesh -allGeometry",
    ]
    if wall:
        here = Path(__file__).resolve().parent
        lines.append(
            f"python3 {(here / 'cfmesh.py').as_posix()} check . --wall {wall}"
            " --log log.cartesianMesh"
        )
        lines.append(
            f"python3 {(here / 'layer_report.py').as_posix()} . --patch {wall}"
            "   # coverage, measured"
        )
    return lines


COSTS = """\
What this mesher will not do, said here rather than found later:
  * No first-layer target, so no y+ request. The first cell comes out as
    edge / (1 + r + ... + r^(n-1)) from the local cell
    (refineBoundaryLayersFunctions.C:688-696); `maxFirstLayerThickness` is a cap on
    that, min(max(cap, SMALL), computed) at :709-714, never a target.
    On the Wigley hull the first cell landed at 1.21 mm as a consequence of the cell
    size. Measure it afterwards with layer_report.py; do not assume it.
  * `renameBoundary` merges. Because `defaultName` is present, cfMesh maps every solid
    the dictionary did not name into ONE patch, which it adds only where some solid is
    in fact unnamed (renameBoundaryPatches.C:149-168, the addPatch test at :151-161),
    so `--wall hull` alone yields a TWO-patch mesh:
    `hull` and a `farField` carrying all six box faces, pointing six ways. That is a
    mesh you cannot give an inlet and an outlet, and so not one a grid-convergence
    study can be run on. Name the faces the case needs -- `--patch xMin:patch`,
    `--patch xMax:patch`, `--symmetry yMin` -- and check them with
    `case_gen.py`, which infers the inlet from a patch's OWN normal and cannot do
    that for a merged patch.
  * The domain box is geometry. A half model needs its cut plane in the STL, and that
    patch wants type `symmetry`.
  * The octree is isotropic, so a wall cell is a cell everywhere. On the Wigley hull
    that was 40,244 wall faces at full width against snappy's 16,578 on a half model,
    with a 7.1 mm wall cell against snappy's 27 mm.
  * Both meshes failed 6 of `checkMesh -allGeometry`'s checks. cfMesh's were milder
    (max non-orthogonality 72.1 against 85.3, max aspect ratio 34.9 against 72.3),
    which is a reading of one hull and not a general claim.
"""


# ------------------------------------------------- reading a meshDict back ---


def _block(text: str, key: str) -> str | None:
    """The text inside `key { ... }`, brace-balanced, or None."""
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*(?:\n\s*)?\{{", text)
    if not m:
        return None
    depth, start = 0, text.index("{", m.start())
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    return None


_SUBDICT = re.compile(r'(?m)^[ \t]*("[^"]+"|[\w.\-]+)[ \t]*(?:\n[ \t]*)?\{')


def _keys(block: str) -> list[str]:
    """The sub-dictionary keys of a block, quoted or bare, in order."""
    return [key for key, _ in _entries(block)]


def _entries(block: str) -> list[tuple[str, str]]:
    """A block's sub-dictionaries as `(key, body)`, brace-balanced, in order.

    The pairing is the point. `_keys` plus a flat `findall` over the whole block
    returns two lists that are not the same length as soon as one entry omits the
    value being looked for -- cfMesh's own `socketOctree` has three `localRefinement`
    entries and two `cellSize`s, because `patch15` asks with
    `additionalRefinementLevels` instead -- and reading the Nth size as belonging to
    the Nth key is then wrong for every entry after the gap. Anything that has to know
    WHICH key carries a value reads it from here.
    """
    out: list[tuple[str, str]] = []
    at = 0
    while True:
        m = _SUBDICT.search(block, at)
        if not m:
            return out
        depth, start = 0, block.index("{", m.end() - 1)
        end = len(block)
        for j in range(start, len(block)):
            if block[j] == "{":
                depth += 1
            elif block[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        out.append((m.group(1), block[start + 1 : end]))
        at = end + 1


def _scalar(text: str, key: str) -> float | None:
    """`key <number>;` in this text, or None."""
    m = re.search(rf"\b{re.escape(key)}\s+([-\deE.+]+)\s*;", text)
    return float(m.group(1)) if m else None


def _word(text: str, key: str) -> str | None:
    """`key <word>;` in this text, or None."""
    m = re.search(rf"\b{re.escape(key)}\s+([\w.\-]+)\s*;", text)
    return m.group(1) if m else None


def read_meshdict(text: str) -> dict[str, Any]:
    """The handful of facts `check` needs out of a meshDict, without a dict parser."""
    text = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))
    out: dict[str, Any] = {}
    m = re.search(r"\bsurfaceFile\s+\"?([^\";]+)\"?\s*;", text)
    out["surface"] = m.group(1).strip() if m else None
    out["max_cell"] = _scalar(text, "maxCellSize")
    # `boundaryCellSize` is not another local refinement: it is stamped onto EVERY
    # surface triangle before any patch-wise level is appended
    # (meshOctreeCreatorCreateOctreeBoxes.C:130-186, the `forAll(surfRefLevel_, triI)`
    # at :182), so wherever it is set it -- and not `maxCellSize` -- is the size a wall
    # comes out at when nothing else asks for anything. 9 of the 14 meshDicts cfMesh
    # ships set it. `check_refinement` reads the wall's floor from here.
    out["boundary_cell"] = _scalar(text, "boundaryCellSize")
    # Read only for its presence. `patchCellSize` appends a patch-wise level the same
    # way `localRefinement` does (meshOctreeCreatorCreateOctreeBoxes.C:190-278, the
    # append at :274) but in either a dictionary or a `(patch size)` list form, and no
    # shipped tutorial uses it, so this does not pretend to resolve it -- it is enough
    # to know that something other than `localRefinement` may also have sized this wall.
    out["patch_cell_size"] = re.search(r"\bpatchCellSize\b", text) is not None
    local = _block(text, "localRefinement")
    # (key, cellSize or None) -- paired, because an entry may ask with
    # `additionalRefinementLevels` instead and leave the sizes list short.
    out["refinement_pairs"] = [
        (key, _scalar(body, "cellSize")) for key, body in _entries(local or "")
    ]
    out["refinement"] = [key for key, _ in out["refinement_pairs"]]
    # There is deliberately no flat `refinement_sizes` here any more. There was one,
    # and `check_refinement` took `min()` of it and attributed the finest size in the
    # whole block to whatever wall it had been given -- so a dictionary that refines
    # the bow and the propeller called a correctly refined bow a failure. Ask
    # `refinement_asked_of` which entries reach a patch instead.
    layers_block = _block(text, "boundaryLayers")
    patch_layers = _block(layers_block, "patchBoundaryLayers") if layers_block else None
    out["layer_patches"] = _keys(patch_layers) if patch_layers is not None else []
    # The two are read apart because cfMesh treats them apart, and reading them
    # together made this call `nLayers 1` a failure on three of cfMesh's own tutorials
    # (sBendOctree, socketOctree, multipleOrifices/pMesh). A GLOBAL `nLayers 1` goes to
    # setGlobalNumberOfLayers, which warns and RETURNS (refineBoundaryLayers.C:121-128)
    # leaving the global at its no-layers default -- which is exactly what those
    # dictionaries mean by it, and sBendOctree then asks for 3 on `walls` anyway. A
    # PER-PATCH `nLayers 1` is the different thing: setNumberOfLayersForPatch warns and
    # stores it (:193-207), so a patch that was asked for layers ends up with none.
    globals_ = (layers_block or "").replace(patch_layers or "\0", "")
    out["n_layers"] = [int(v) for v in re.findall(r"\bnLayers\s+(\d+)\s*;", patch_layers or "")]
    out["global_layers"] = [int(v) for v in re.findall(r"\bnLayers\s+(\d+)\s*;", globals_)]
    out["ratios"] = [
        float(v) for v in re.findall(r"\bthicknessRatio\s+([-\deE.+]+)\s*;", patch_layers or "")
    ]
    out["ratios"] += [
        float(v) for v in re.findall(r"\bthicknessRatio\s+([-\deE.+]+)\s*;", globals_)
    ]
    rename = _block(text, "renameBoundary")
    new_names = _block(rename, "newPatchNames") if rename else None
    # (key, newName or None) -- paired for the same reason: the key is the SURFACE-side
    # selector and `newName` is what the patch is called in the mesh afterwards, and
    # `check_refinement` has to get from one to the other.
    out["rename_pairs"] = [
        (key, _word(body, "newName")) for key, body in _entries(new_names or "")
    ]
    out["renamed"] = [key for key, _ in out["rename_pairs"]]
    out["rename_types"] = re.findall(r"\btype\s+(\w+)\s*;", new_names or "")
    return out


# The surface formats whose solid names reach cfMesh VERBATIM. Everything the octree
# is handed is one of these or a `.ftr`, and which it is decides what a dictionary key
# has to look like -- see `surface_kind`.
VERBATIM_SURFACES = (".stl", ".stlb", ".obj", ".fms", ".vtk")

INDEXED = re.compile(r"_\d+$")
METACHARS = ".*[]()?+|^$"


def _literal(key: str) -> bool:
    """Is this dictionary key a plain name rather than a regex?"""
    return not any(c in key.strip('"') for c in METACHARS)


def surface_kind(surface: str | None) -> str:
    """`indexed`, `verbatim` or `unknown`: what the patch names will be at match time.

    This is the fact that decides whether a literal key or an indexed regex is the
    right one, and reading a key without it was the defect this function exists to
    stop. `check_meshdict` used to call any literal key a `fail` outright, which is a
    `fail` on 6 of the 14 meshDicts cfMesh itself ships -- `asmoOctree`,
    `intakePortOctree`, `cutCubeOctree`, `socketOctree`, `sBendOctree` and
    `ship5415Octree`, whose Allruns hand a raw surface straight to `cartesianMesh`
    with no feature-edge pass at all. A fail-severity check that is wrong about a
    working dictionary is the same error F-53 exists to stop, one level up.
    """
    if not surface:
        return "unknown"
    suffix = Path(surface).suffix.lower()
    if suffix == ".ftr":
        return "indexed"
    if suffix in VERBATIM_SURFACES:
        return "verbatim"
    return "unknown"


# The shape this script writes and the shape the record documents: a plain solid
# name, a MANDATORY underscore, and a wildcard tail. That underscore is the index
# separator `surfaceFeatureEdges` introduces, so on a surface that has not been
# through it the key can only match a solid whose own name already carries one.
#
# `_` is in the name class as well as being the separator. It was not until
# 2026-09-08, and the omission meant this did not recognise the very key `selector()`
# writes for a solid whose own name has an underscore in it: `"hull_box_.*"` and
# `"HULL_AND_BOX_.*"` both read as needing no index and a raw-STL dictionary carrying
# one came back `ok`. The tail decides the shape, not the name, so widening the name
# class re-flags nothing: `"inlet.*"` and `"orifice0[3-6].*"` -- the two cfMesh's own
# tutorials write against raw STLs -- have no trailing `_` and still do not match.
INDEX_KEY = re.compile(r"^[A-Za-z0-9_\-]+_(\.\*|\.\+|\[0-9\][*+]?|\\d[*+]?)$")


def wants_an_index(key: str) -> bool:
    """Is this key written for a name `surfaceFeatureEdges` has indexed?

    Deliberately narrow, and narrower than the question. OpenFOAM's `regExp` is a
    WHOLE-string match, so `"inlet.*"` matches a raw STL's `inlet` -- `.*` matches
    empty -- and `"orifice0[3-6].*"` matches `orifice03`. A mandatory `_` before the
    wildcard is the one shape that provably makes a key miss an un-indexed name, and
    this recognises that shape only: `<name>_` followed by `.*`, `.+`, `[0-9]` or
    `\\d`, with an optional `*`/`+`. The name may itself contain underscores, so
    `"hull_box_.*"` is recognised -- the LAST `_` is the separator.

    Other shapes that need an index are missed on purpose rather than by oversight:
    `"hull_[0-9][0-9]"`, `"hull_(0|1)"` and anything with an alternation are all keys
    that cannot match a bare `hull`, and none of them match here. The general question
    ("can this regex match some name with no index?") is not decidable from the
    dictionary, and the previous attempt at it -- probe the leading literal prefix --
    called cfMesh's own `"orifice0[3-6].*"` a miss, which is the false-positive this
    whole check was rewritten to stop making. So a False from this function means "not
    the one shape we can prove", never "this key is fine".

    Even this shape is a warning and not a failure on a raw surface: the dictionary
    cannot show what the solids are called, and a surface really carrying a solid
    named `hull_port` would match. Against a `.ftr` the direction that IS provable --
    every patch has been renamed, so a bare literal matches nothing -- is the failure.
    """
    return INDEX_KEY.match(key.strip('"')) is not None


def check_meshdict(text: str) -> list[Finding]:
    """A meshDict read for the requests cfMesh accepts and then does not carry out.

    Static, so it costs nothing and can be run on a dictionary written by hand, by a
    tutorial, or by this script and then edited.
    """
    d = read_meshdict(text)
    out: list[Finding] = []

    surface = d["surface"] or ""
    kind = surface_kind(surface)
    all_keys = d["refinement"] + d["layer_patches"] + d["renamed"]

    if all_keys and kind == "unknown":
        out.append(
            Finding(
                "cfmesh-selectors",
                "warn",
                f"{len(all_keys)} patch keys and surfaceFile "
                + (surface or "is not set"),
                "which key is right is decided by the surface format -- a `.ftr` has"
                " been through surfaceFeatureEdges and carries indexed names, anything"
                " else reaches cfMesh with its solid names verbatim -- and this"
                " dictionary does not say which it is",
                "name a surfaceFile this can read the extension of",
            )
        )
    elif kind == "indexed":
        # A `.ftr`: names are `<solid>_<index>`, so a bare literal matches nothing.
        # A literal that already CARRIES an index does match -- ship5415Octree, the
        # tutorial the measured route was copied from, keys on the literal
        # `HULL_AND_BOX_1` -- so that is a warning about brittleness, not a failure.
        stale = [k for k in all_keys if _literal(k) and not INDEXED.search(k.strip('"'))]
        pinned = [k for k in all_keys if _literal(k) and INDEXED.search(k.strip('"'))]
        if stale:
            out.append(
                Finding(
                    "cfmesh-selectors",
                    "fail",
                    f"surfaceFile {surface} with literal patch keys: "
                    + ", ".join(stale),
                    "a .ftr has been through surfaceFeatureEdges, which names every"
                    " piece `<solid>_<index>`, so a literal key matches nothing;"
                    " cfMesh warns on the mesh side, says nothing at all on the"
                    " surface side, meshes at maxCellSize with no refinement and no"
                    " layers, and exits 0",
                    'quote them as regexes -- "hull_.*" -- and put the name back with'
                    " renameBoundary",
                )
            )
        elif pinned:
            out.append(
                Finding(
                    "cfmesh-selectors",
                    "warn",
                    "patch keys pinned to an index: " + ", ".join(pinned),
                    "these match, and cfMesh's own ship5415Octree tutorial writes them"
                    " -- but the suffix is the GLOBAL new-patch counter"
                    " (triSurfacePatchManipulator.C:120-122), so adding one solid to"
                    " the surface, or a feature angle that splits a solid differently,"
                    " renumbers them and the key silently stops matching",
                    'a regex on the name instead -- "hull_.*"',
                )
            )
        elif all_keys:
            out.append(
                Finding(
                    "cfmesh-selectors",
                    "ok",
                    f"{len(all_keys)} patch keys against a .ftr, all regexes",
                    "they can match the indexed names surfaceFeatureEdges produced",
                )
            )
    elif kind == "verbatim" and all_keys:
        # A raw surface: the solid names arrive as written, so a literal key is the
        # RIGHT key here and an index-requiring regex is the miss.
        missing = [k for k in all_keys if wants_an_index(k)]
        if missing:
            out.append(
                Finding(
                    "cfmesh-selectors",
                    "warn",
                    f"surfaceFile {surface} with keys that require an index: "
                    + ", ".join(missing),
                    "no surfaceFeatureEdges pass has run on this surface, so its solid"
                    " names arrive exactly as written -- the mandatory `_` in these"
                    " keys then matches only a solid whose own name carries one, which"
                    " the dictionary cannot show; if it does not, the miss is SILENT on"
                    " the surface side, because triSurfFacets.C:103's warning is under"
                    " #ifdef DEBUGtriSurf, and cfMesh meshes on and exits 0",
                    "run `surfaceFeatureEdges <surface> <name>.ftr` and point"
                    " surfaceFile at the .ftr -- which is also what puts the hard edges"
                    " in the octree -- or key on the solid names as written",
                )
            )
        else:
            out.append(
                Finding(
                    "cfmesh-selectors",
                    "ok",
                    f"{len(all_keys)} patch keys against a raw {Path(surface).suffix}",
                    "the surface has not been through surfaceFeatureEdges, so its solid"
                    " names reach cfMesh verbatim and these keys can match them",
                )
            )

    thin = [n for n in d["global_layers"] if n < MIN_LAYERS]
    if thin and not d["layer_patches"]:
        out.append(
            Finding(
                "cfmesh-layers",
                "warn",
                f"boundaryLayers asks for a global nLayers {thin[0]} and names no"
                " patches",
                "cfMesh warns 'The specified global number of boundary layers is less"
                " than 2' and RETURNS (refineBoundaryLayers.C:121-128), so the global"
                " keeps its no-layers default and this mesh gets none -- which is what"
                " three of cfMesh's own tutorials mean by it, and is a surprise only if"
                " `nLayers 1` was read as one layer",
                "2 or more, or a patchBoundaryLayers entry naming the wall",
            )
        )

    bad_layers = [n for n in d["n_layers"] if n < MIN_LAYERS]
    if bad_layers:
        out.append(
            Finding(
                "cfmesh-layers",
                "fail",
                f"a patchBoundaryLayers entry asks for nLayers"
                f" {', '.join(str(n) for n in bad_layers)}",
                "below 2 cfMesh warns 'boundary layers disabled for this patch'"
                " (refineBoundaryLayers.C:193-199) and meshes without them",
                "ask for 2 or more layers, or remove the boundaryLayers block so the"
                " mesh is honestly layerless",
            )
        )
    bad_ratio = [r for r in d["ratios"] if r < MIN_RATIO]
    if bad_ratio:
        out.append(
            Finding(
                "cfmesh-ratio",
                "fail",
                f"thicknessRatio {', '.join(f'{r:g}' for r in bad_ratio)}",
                "below 1.0 cfMesh warns and returns, keeping its own value"
                " (setThicknessRatioForPatch, refineBoundaryLayers.C:223-229), so the"
                " request is dropped",
                "a ratio of 1.0 or more, 1.2 being the value the Wigley comparison ran",
            )
        )

    if "symmetryPlane" in d["rename_types"]:
        out.append(
            Finding(
                "cfmesh-symmetry",
                "fail",
                "renameBoundary sets a patch to type symmetryPlane",
                "the layer pass moves its points by ~1e-9 rad and"
                " symmetryPlanePolyPatch demands planarity to machine precision, after"
                " which checkMesh, foamFormatConvert and postProcess all refuse to open"
                " the mesh -- and improveSymmetryPlanes declines the repair",
                "type symmetry",
            )
        )

    if d["layer_patches"] and not d["renamed"]:
        out.append(
            Finding(
                "cfmesh-rename",
                "warn",
                "layers are requested and renameBoundary names nothing",
                "the mesh's wall keeps the indexed name surfaceFeatureEdges gave it,"
                " `<name>_<n>` -- and n is the GLOBAL new-patch counter"
                " (triSurfacePatchManipulator.C:120-122), so it is `_0` only if this"
                " solid was written into the surface first, and not knowable from the"
                " dictionary otherwise -- and every dictionary after it (fields,"
                " forces, y+) has to carry whichever name that turns out to be",
                "a renameBoundary block putting the name back",
            )
        )

    # `.FMS` as well as `.fms`: this check is advertised as runnable on a hand-written
    # dictionary, and a hand-written one is exactly where the capitalisation varies.
    # Case-sensitivity here disarmed trap 2 on the only input it was written for.
    if surface.lower().endswith(".fms"):
        out.append(
            Finding(
                "cfmesh-surface",
                "warn",
                f"surfaceFile {surface}",
                "an FMS written by `surfaceGenerateBoundingBox` on this release is"
                " unreadable -- its six generated patches carry an empty geometricType,"
                " and the reader then takes the next patch's name as this one's type and"
                " dies inside the point list, naming the point list and not the patches",
                "`surfaceFeatureEdges <stl> <name>.ftr` and point surfaceFile at the"
                " .ftr, or write the closed multi-solid STL with `cfmesh.py box`",
            )
        )
    return out


# ------------------------------------------- an existing mesh, made readable --


def mesh_written_as(case: Path) -> tuple[str, str]:
    """`(format, class)` off the head of `constant/polyMesh/faces`."""
    faces = case / "constant" / "polyMesh" / "faces"
    if not faces.is_file():
        raise SurfaceError(f"{case.as_posix()}: no constant/polyMesh/faces")
    head = faces.read_bytes()[:2048].decode("utf-8", errors="replace")
    fmt = re.search(r"\bformat\s+(\w+)\s*;", head)
    cls = re.search(r"\bclass\s+(\w+)\s*;", head)
    return (fmt.group(1) if fmt else "unknown"), (cls.group(1) if cls else "unknown")


def set_write_format_ascii(text: str) -> tuple[str, bool]:
    """`writeFormat ascii;` in a controlDict, and whether it had to be changed.

    This has to happen BEFORE `foamFormatConvert`, not after. `foamFormatConvert`
    writes in whatever `writeFormat` says, so running it first converts the mesh to
    exactly what it already was and looks like it did nothing -- which is what happened
    on 2026-08-31 and cost the round.
    """
    m = re.search(r"(?m)^(\s*writeFormat\s+)(\w+)(\s*;.*)$", text)
    if m:
        if m.group(2) == "ascii":
            return text, False
        return text[: m.start()] + m.group(1) + "ascii" + m.group(3) + text[m.end() :], True
    return text.rstrip("\n") + "\nwriteFormat     ascii;\n", True


def retype_symmetry_planes(text: str) -> tuple[str, list[str]]:
    """`type symmetryPlane;` -> `type symmetry;` in a `boundary` file, and which moved.

    Only the type token. This used to run `.replace("symmetryPlane", "symmetry")` over
    the whole match, which also rewrote the patch's own NAME: a patch a mesher had
    called `symmetryPlane` came back called `symmetry`, and since this is the one path
    that writes to `constant/polyMesh/boundary` under `--write`, every `0/` field's
    `boundaryField` entry for that patch then stopped matching -- a mesh broken more
    quietly than the one being repaired.

    `inGroups` is deliberately left alone. A snappy boundary file carries
    `inGroups 1(symmetryPlane);` beside the type, and that word is a patch GROUP name
    that a `boundaryField` may be keyed on; rewriting it would silently drop those
    entries, which is the same class of harm in the other direction. The group name is
    now stale rather than wrong, and the patch type is what OpenFOAM constructs from.
    """
    moved: list[str] = []

    def repl(m: re.Match) -> str:
        moved.append(m.group(1))
        return m.group(1) + m.group(2) + "symmetry" + m.group(3)

    out = re.sub(
        r"([\w.\-]+)(\s*\{[^{}]*?\btype\s+)symmetryPlane(\s*;[^{}]*?\})",
        repl,
        text,
        flags=re.S,
    )
    return out, moved


def prepare(case: Path) -> list[Finding]:
    """What stands between an existing mesh and cfMesh's `generateBoundaryLayers`."""
    out: list[Finding] = []
    try:
        fmt, cls = mesh_written_as(case)
    except SurfaceError as exc:
        return [Finding("cfmesh-readable", "skipped", str(exc))]

    if fmt != "ascii" or cls != "faceList":
        out.append(
            Finding(
                "cfmesh-readable",
                "fail",
                f"constant/polyMesh/faces is {fmt}, class {cls}",
                "cfMesh's polyMeshGen reader wants a plain ASCII faceList and fails"
                " with `unexpected class name faceCompactList expected faceList`;"
                " snappyHexMesh writes binary faceCompactList",
                "set `writeFormat ascii;` in system/controlDict FIRST, then"
                " `foamFormatConvert -constant` -- the other order converts the mesh to"
                " what it already was and looks like it did nothing",
            )
        )
    else:
        out.append(
            Finding(
                "cfmesh-readable",
                "ok",
                f"constant/polyMesh/faces is {fmt}, class {cls}",
                "cfMesh's reader can open this mesh",
            )
        )

    boundary = case / "constant" / "polyMesh" / "boundary"
    if boundary.is_file():
        _, planes = retype_symmetry_planes(boundary.read_text(errors="replace"))
        if planes:
            out.append(
                Finding(
                    "cfmesh-symmetry",
                    "fail",
                    "symmetryPlane patches: " + ", ".join(planes),
                    "`generateBoundaryLayers` moves symmetry-plane points by ~1e-9 rad"
                    " and symmetryPlanePolyPatch demands machine-precision planarity,"
                    " after which checkMesh, foamFormatConvert and postProcess all"
                    " refuse to open the mesh; improveSymmetryPlanes declines the"
                    " repair and says to use type symmetry instead",
                    "type symmetry on those patches in constant/polyMesh/boundary"
                    " (`cfmesh.py prepare <case> --write` does both edits)",
                )
            )
        else:
            out.append(
                Finding(
                    "cfmesh-symmetry",
                    "ok",
                    "no symmetryPlane patches",
                    "nothing here for the layer pass to un-planarise",
                )
            )

    out.append(
        Finding(
            "cfmesh-layer-cost",
            "warn",
            "the hybrid route's layer pass took 21 minutes on a 246k-cell mesh",
            "snappy geometry with cfMesh layers reached 86.9% coverage on the Wigley"
            " hull against snappy's own 66.7%, at a 10.68 mm first cell -- better"
            " coverage, a thicker wall cell, and a mesh that has to be repaired"
            " afterwards. `cartesianMesh` from the surface reached 100.0%",
            "budget for it, or mesh from the surface with cartesianMesh instead",
        )
    )
    return out


def apply_preparation(case: Path) -> list[str]:
    """The two edits `prepare` describes, made. Returns what changed."""
    changed: list[str] = []
    control = case / "system" / "controlDict"
    if control.is_file():
        text, did = set_write_format_ascii(control.read_text(errors="replace"))
        if did:
            control.write_text(text, encoding="utf-8", newline="\n")
            changed.append("system/controlDict: writeFormat ascii")
    boundary = case / "constant" / "polyMesh" / "boundary"
    if boundary.is_file():
        text, planes = retype_symmetry_planes(boundary.read_text(errors="replace"))
        if planes:
            boundary.write_text(text, encoding="utf-8", newline="\n")
            changed.append(
                "constant/polyMesh/boundary: type symmetry on patch(es) "
                + ", ".join(planes)
            )
    return changed


# ------------------------------------------------- the mesh, after the fact --


def scan_log(text: str) -> list[str]:
    """The patch names cfMesh said it could not match. One line, then it carries on."""
    seen: list[str] = []
    for name in UNMATCHED.findall(text):
        if name not in seen:
            seen.append(name)
    return seen


def selects(pattern: str, name: str) -> bool:
    """Would cfMesh's own matching select the patch `name` with this dictionary key?

    Whole-string, quotes stripped. cfMesh resolves every key this way --
    `findMatchingStrings(regExp(patchName), patchNames())`, `triSurfFacets.C:92-104`
    on the surface side and `polyMeshGenFaces.C:271-285` on the mesh side -- and a
    whole-string match is what makes the two halves of trap 1 opposite: `hull` does
    not select `hull_0`, and `hull_.*` does not select a bare `hull`.

    A key OpenFOAM would accept but Python's `re` rejects returns False rather than
    raising; a key that cannot be compiled here is one this script cannot reason
    about, and the callers all treat "cannot say" as "do not claim anything".
    """
    try:
        return re.fullmatch(pattern.strip('"'), name) is not None
    except re.error:
        return False


def refinement_asked_of(spec: dict[str, Any], wall: str) -> tuple[list[str], float | None]:
    """The `localRefinement` entries that reach `wall`, and the finest size among them.

    `wall` is the name the patch has IN THE MESH, which is the far end of a rename:
    `localRefinement`'s keys select SURFACE patch names, and `renameBoundary` is what
    turns those into the mesh's names. So the wall is resolved back to its surface-side
    selectors first -- its own name, which is right whenever no rename happened, plus
    the key of any `newPatchNames` entry whose `newName` is this wall -- and a
    refinement key reaches the wall when it is that same selector or when it selects
    one of those names outright.

    Two regexes cannot in general be compared, and this does not try: the match is
    string equality between the two keys, plus a whole-string regex match against the
    literal names. So a refinement key written differently from the rename key that
    covers the same solids (`"hull_[0-9]"` against `"hull_.*"`) is not paired here and
    comes back as nothing asked -- the conservative direction, and the caller then
    reports `skipped` rather than a verdict. It leans the other way only where a
    refinement key happens to match the TEXT of a rename key, which for a key like
    `".*"` is also the right answer and for a contrived one like `"hull_\\..*"` is not;
    the cost there is a comparison against a neighbouring size, not an all-clear.

    Taking the FINEST of the matching sizes is cfMesh's own arithmetic: every matching
    entry APPENDS a level to the triangle (`meshOctreeCreatorCreateOctreeBoxes.C:443-457`)
    and the octree keeps splitting while any appended request is deeper than the cube's
    current level (`meshOctreeCreatorAdjustOctreeToSurface.C:95-127`), so the deepest
    request wins.
    """
    selectors = [wall]
    selectors += [
        key.strip('"') for key, new in spec.get("rename_pairs", []) if new == wall
    ]
    hit = [
        (key, size)
        for key, size in spec.get("refinement_pairs", [])
        if any(key.strip('"') == s or selects(key, s) for s in selectors)
    ]
    sizes = [size for _, size in hit if size is not None]
    return [key for key, _ in hit], (min(sizes) if sizes else None)


def check_refinement(mesh: dict, spec: dict[str, Any], wall: str | None) -> list[Finding]:
    """Did `localRefinement` reach the wall, measured on the wall's own faces?

    This is the SURFACE-side miss, the half of trap 1 that prints nothing at all
    because `triSurfFacets.C:103`'s warning is under `#ifdef DEBUGtriSurf`. The mesh is
    the only witness.

    It used to be a cell count against `volume / maxCellSize**3` with a `warn` below
    1.3x. The estimate was fine -- cfMesh resizes the root cube so the cell at
    `globalRefLevel` is exactly `maxCellSize`
    (`meshOctreeCreatorCreateOctreeBoxes.C:83-106`) -- but the 1.3 was invented. No run
    in `qa-runs/comp/cfmesh/` calibrates it, its two tests bracketed it at 0.86x and
    32x, and on this tool's own default box a correctly refined external case can land
    near it. The workspace rule is that a brief carries measurements and not constants,
    and a threshold nobody measured is that rule broken inside the check.

    What the QA record actually recorded as the tell is a SIZE -- "a 10,576-face mesh
    where 300,000 were expected" -- so this measures one. cfMesh's octree only halves,
    so a wall the key reached carries faces within one octree level of the `cellSize`
    that key asked for, and a wall the key missed comes out at the floor the rest of
    the dictionary sets for it. That floor is `maxCellSize` only when nothing else
    sizes the surface: `boundaryCellSize`, where it is set, is stamped onto EVERY
    surface triangle before any patch-wise level is appended
    (`meshOctreeCreatorCreateOctreeBoxes.C:130-186`), and it is set in 9 of the 14
    meshDicts cfMesh ships. `read_meshdict` reads it and this compares against
    whichever of the two is the real floor; reading `maxCellSize` as the floor over a
    `boundaryCellSize 0.35` dictionary printed a confident `ok` over exactly the silent
    miss this check exists to catch.

    On the Wigley numbers request and floor are 16x apart in edge length; the question
    is only which of them the measured spacing is nearer, in log2, and there is no
    constant to choose. Where they are less than one level apart, the two answers are
    not distinguishable and this says so rather than guessing.

    Three things it does NOT claim. The request it compares against is the one keyed to
    THIS wall (`refinement_asked_of`), so a dictionary that also refines something else
    more finely is not read as a demand on the wall -- that bug printed
    `fail ... against 10 mm requested` on a wall that had been given exactly the 0.35 m
    it asked for. Where no key resolves to the wall, or where `patchCellSize` may also
    have sized it, the answer is `skipped` and says which. And a `fail` here is a
    statement about SIZE, not about the mechanism: a wall sitting at the floor is what
    a missed key looks like, and it is also what a key that matched but asked for the
    floor looks like -- which is why the `levels < 1.0` guard above refuses the reading
    rather than picking one.
    """
    max_cell = spec.get("max_cell")
    if not spec.get("refinement") or not max_cell:
        return []
    if wall is None or wall not in mesh["boundary"]:
        return [
            Finding(
                "cfmesh-refinement",
                "skipped",
                "localRefinement is asked for and there is no named wall to measure it"
                " on" if wall is None else f"no patch `{wall}` in this mesh",
                "the surface-side miss shows up as a wall meshed at the dictionary's"
                " floor size, which is a reading of the wall's own faces and needs the"
                " wall",
                "pass --wall <the refined patch>",
            )
        ]

    keys, asked = refinement_asked_of(spec, wall)
    if not keys:
        return [
            Finding(
                "cfmesh-refinement",
                "skipped",
                f"localRefinement asks nothing of `{wall}`: its keys are "
                + ", ".join(spec["refinement"]),
                "none of those keys resolves to this wall -- not as its own name, not"
                " as the newPatchNames key that renamed it -- so there is no request"
                " about this wall to check the mesh against; another patch's cellSize"
                " is not one",
                "name the wall in localRefinement, or --wall the patch that is refined",
            )
        ]
    if asked is None:
        return [
            Finding(
                "cfmesh-refinement",
                "skipped",
                f"localRefinement names `{wall}` via " + ", ".join(keys)
                + " but with no cellSize this could read",
                "an entry can ask with additionalRefinementLevels instead, which is a"
                " level count and not a size, and there is then no length to compare"
                " the wall's faces against",
            )
        ]

    # The floor: what this wall comes out at if localRefinement reaches nothing.
    floor, floor_name = max_cell, "maxCellSize"
    boundary_cell = spec.get("boundary_cell")
    if boundary_cell and 0 < boundary_cell < max_cell:
        floor, floor_name = boundary_cell, "boundaryCellSize"
    if spec.get("patch_cell_size"):
        return [
            Finding(
                "cfmesh-refinement",
                "skipped",
                f"localRefinement asks {asked * 1e3:.3g} mm of `{wall}` and the"
                " dictionary also has a patchCellSize block",
                "patchCellSize appends a patch-wise level the same way localRefinement"
                " does, so a wall measuring fine cannot be attributed to either of them"
                " -- and this does not resolve patchCellSize's two accepted forms",
                "read the two blocks together by hand, or fold the request into"
                " localRefinement",
            )
        ]

    spacing = layer_report.face_spacing(mesh, wall)
    faces = mesh["boundary"][wall]["nFaces"]
    if spacing <= 0 or asked <= 0:
        return [
            Finding("cfmesh-refinement", "skipped", f"{wall}: no measurable face area")
        ]

    levels = math.log2(floor / asked)
    measured_ = (
        f"{wall}: {faces:,} faces at sqrt(median area) {spacing * 1e3:.3g} mm,"
        f" against {asked * 1e3:.3g} mm asked of it by {', '.join(keys)}"
        f" and {floor * 1e3:.3g} mm {floor_name}"
        f" ({levels:.1f} octree levels apart)"
    )
    if levels < 1.0:
        return [
            Finding(
                "cfmesh-refinement",
                "skipped",
                measured_,
                "the request and the floor this wall falls back to are less than one"
                " octree halving apart, so a wall that got the refinement and a wall"
                " that missed it are the same size and this cannot tell them apart",
                f"ask for a cellSize at least half of {floor_name} before reading this",
            )
        ]

    # `face_spacing` is sqrt(median face area). On a cfMesh octree wall the faces are
    # squares, so that IS the cell edge, and the comparison below is between two edge
    # lengths at least one octree halving apart -- see the note on that function.
    to_asked = abs(math.log2(spacing / asked))
    to_floor = abs(math.log2(spacing / floor))
    if to_floor < to_asked:
        return [
            Finding(
                "cfmesh-refinement",
                "fail",
                measured_,
                f"the wall is nearer the {floor_name} it falls back to than the size it"
                " asked for, so localRefinement matched nothing there -- and on the"
                " surface side that miss prints NOTHING, because triSurfFacets.C:103's"
                " warning sits under #ifdef DEBUGtriSurf; cfMesh meshed on and exited 0",
                "check the localRefinement key against the surfaceFile format:"
                ' "hull_.*" for a .ftr, the bare solid name for a raw STL',
            )
        ]
    return [
        Finding(
            "cfmesh-refinement",
            "ok",
            measured_,
            "the wall is nearer the size localRefinement asked for than the"
            f" {floor_name} it would have fallen back to, so the key matched",
        )
    ]


def check(
    case: Path,
    wall: str | None = None,
    log: Path | None = None,
    ref: Path | None = None,
    wall_spacing: float | None = None,
) -> list[Finding]:
    """Did the mesh get what the dictionary asked for? Read off the mesh, not the log.

    The log is read too where there is one, because trap 1's warning is the only place
    a mesh-side miss is ever stated -- but the mesh is the evidence, and a run whose
    log has been rotated away is still measurable.
    """
    out: list[Finding] = []

    if log is not None and not log.is_file():
        # Not silence. The log scan is the only place trap 1's mesh-side warning is
        # ever stated, so a typo'd or rotated path that produced no finding at all read
        # exactly like a clean run -- `prepare` already returns a `skipped` Finding for
        # a missing file one function up, and this is the same convention.
        out.append(
            Finding(
                "cfmesh-log",
                "skipped",
                f"{log.as_posix()}: no such file",
                "the `Cannot find any patch names matching` warning is the only thing"
                " that says a dictionary key matched nothing on the mesh side, and it"
                " was not read -- this is not the same as reading it and finding none",
                "point --log at the cartesianMesh log, or drop it and rely on the mesh",
            )
        )
    if log is not None and log.is_file():
        missed = scan_log(log.read_text(errors="replace"))
        if missed:
            out.append(
                Finding(
                    "cfmesh-log",
                    "fail",
                    "`Cannot find any patch names matching` " + ", ".join(missed),
                    "cfMesh matched no patch for those keys, warned, meshed without"
                    " them and exited 0 -- so the mesh has no refinement and no layers"
                    " where those keys were meant to put them",
                    "quote the keys as regexes to match the indexed names"
                    ' surfaceFeatureEdges produced -- "hull_.*"',
                )
            )
        else:
            out.append(
                Finding(
                    "cfmesh-log",
                    "ok",
                    f"{log.name}: no unmatched patch warnings",
                    "every dictionary key found a patch",
                )
            )

    dict_path = case / "system" / "meshDict"
    spec: dict[str, Any] = {}
    if dict_path.is_file():
        text = dict_path.read_text(errors="replace")
        spec = read_meshdict(text)
        out.extend(check_meshdict(text))

    try:
        mesh = layer_report.load(case)
    except layer_report.MeshError as exc:
        out.append(Finding("cfmesh-mesh", "skipped", str(exc)))
        return out

    patches = mesh["boundary"]
    if wall:
        if wall in patches:
            out.append(
                Finding(
                    "cfmesh-wall",
                    "ok",
                    f"patch {wall}: {patches[wall]['nFaces']:,} faces,"
                    f" type {patches[wall]['type']}",
                    "renameBoundary put the name back",
                )
            )
        else:
            indexed = sorted(p for p in patches if p.startswith(f"{wall}_"))
            out.append(
                Finding(
                    "cfmesh-wall",
                    "fail",
                    f"no patch `{wall}`; the mesh has: " + ", ".join(sorted(patches)),
                    (
                        "the indexed names survived, so renameBoundary matched nothing"
                        f" ({', '.join(indexed)})"
                        if indexed
                        else "nothing in this mesh carries that name"
                    ),
                    "a renameBoundary block keyed on a regex",
                )
            )

    out.extend(check_refinement(mesh, spec, wall))

    walls = [wall] if wall else layer_report.wall_patches(mesh)
    if walls:
        found = layer_report.measure(case, walls, ref, wall_spacing)
        for problem in found["problems"]:
            out.append(Finding("cfmesh-coverage", "skipped", problem))
        for patch in found["patches"]:
            coverage = patch.get("coverage_pct")
            if coverage is None:
                out.append(
                    Finding(
                        "cfmesh-coverage",
                        "skipped",
                        f"{patch['patch']}: {patch.get('note', 'not measured')}",
                    )
                )
                continue
            out.append(
                Finding(
                    "cfmesh-coverage",
                    "ok" if coverage >= 95 else "warn",
                    f"{patch['patch']}: {coverage:.1f}% of faces carry a first cell"
                    f" thinner than the threshold ({patch.get('basis', 'unstated')})",
                    "with no --ref mesh and no known wall spacing the threshold is this"
                    " mesh's own median, which only means something near 50% -- rerun"
                    " layer_report.py --ref against the same wall meshed with no layer"
                    " request for a number that stands on its own"
                    if "median" in str(patch.get("basis", ""))
                    else "measured against a reference wall spacing",
                    f"python3 {(Path(__file__).resolve().parent / 'layer_report.py').as_posix()}"
                    f" {case.as_posix()} --patch {patch['patch']}"
                    " --ref <the same case meshed with no layers>",
                )
            )
    return out


# ------------------------------------------------------------------- report --


def report(findings: list[Finding]) -> str:
    lines = []
    for f in findings:
        lines.append(f"[{f.status:7s}] {f.check}")
        lines.append(f"    measured: {f.measured}")
        if f.meaning:
            lines.append(f"    means:    {f.meaning}")
        if f.repair:
            lines.append(f"    repair:   {f.repair}")
    return "\n".join(lines) if lines else "nothing to report"


# --------------------------------------------------------------------- CLI ---


def _cmd_where(args) -> int:
    found = where()
    print(json.dumps(found, indent=2) if args.json else report_where(found))
    return 0


def _cmd_box(args) -> int:
    body = read_stl(args.body, args.name)
    if args.box:
        lo = (args.box[0], args.box[2], args.box[4])
        hi = (args.box[1], args.box[3], args.box[5])
    else:
        lo, hi = domain_box(*bounds(body), args.upstream, args.downstream, args.side)
    touching = enclose(body, lo, hi)
    if args.out.exists() and not args.force:
        raise OverwriteError(f"{args.out} exists; --force to overwrite")
    write_stl(body + box_solids(lo, hi), args.out)
    facets = sum(len(s.facets) for s in body)
    print(f"wrote {args.out}: {len(body)} body solid(s), {facets} facets, "
          f"+ 6 box faces (12 facets)")
    print(f"  body   ({bounds(body)[0][0]:.4g} {bounds(body)[0][1]:.4g} "
          f"{bounds(body)[0][2]:.4g}) .. ({bounds(body)[1][0]:.4g} "
          f"{bounds(body)[1][1]:.4g} {bounds(body)[1][2]:.4g})")
    print(f"  box    ({lo[0]:.4g} {lo[1]:.4g} {lo[2]:.4g}) .. "
          f"({hi[0]:.4g} {hi[1]:.4g} {hi[2]:.4g})")
    print("  solids " + ", ".join(s.name for s in body) + ", " + ", ".join(BOX_FACES))
    if touching:
        print("  the body meets " + ", ".join(touching) + " -- those are cut planes:")
        print("  give them type `symmetry`, never `symmetryPlane` (trap 4).")
    print("\n" + "\n".join(
        route(args.out.name, body[0].name if body else None, args.angle)
    ))
    return 0


def _parse_patch(spec: str) -> tuple[str, str]:
    """`name` or `name:type` -> `(name, type)`, defaulting to type `patch`."""
    name, _, ptype = spec.partition(":")
    if not name:
        raise DictError(f"--patch {spec!r} names no face")
    return name, ptype or "patch"


def _cmd_case(args) -> int:
    patches = tuple(_parse_patch(s) for s in (args.patch or ()))
    text = meshdict(
        surface=args.surface,
        max_cell=args.max_cell,
        wall=args.wall,
        wall_cell=args.wall_cell,
        layers=args.layers,
        ratio=args.ratio,
        default_name=args.default_name,
        default_type=args.default_type,
        symmetry=tuple(args.symmetry or ()),
        patches=patches,
    )
    out = args.case / "system" / "meshDict"
    if out.exists() and not args.force:
        raise OverwriteError(f"{out} exists; --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    named = feature_file(args.surface)
    if named != args.surface:
        print(f"  surfaceFile names {named}, not {args.surface}: the keys below are"
              " regexes, and only surfaceFeatureEdges produces the names they match")
    # The merge, said at the moment the dictionary is written rather than discovered
    # when case_gen.py asks a six-sided `farField` which way the inlet points. Whether
    # it happens at all is conditional -- cfMesh adds the defaultName patch only where
    # some solid is left unmatched (the addPatch test, renameBoundaryPatches.C:151-161)
    # -- and this said it unconditionally until 2026-09-08, so a run that named all
    # seven solids still printed a merge with nothing to merge. The names of the
    # surface's OWN solids are not known here (the .stl is not read), so what can be
    # checked is `box`'s six faces; anything else in the surface is called out as the
    # open question it is.
    kept = [args.wall] if args.wall else []
    kept += list(args.symmetry or ()) + [name for name, _ in patches]
    merged = [f for f in BOX_FACES if f not in kept]
    print(f"  named patches: {', '.join(kept) or 'none'}")
    if merged:
        print(f"  everything else merges into one `{args.default_name}` patch of type"
              f" {args.default_type} (renameBoundaryPatches.C:149-168)")
        print("  -- of `box`'s six faces that is: " + ", ".join(merged) + ".")
        print("     A merged patch has faces pointing six ways, so it cannot be given"
              " an inlet")
        print("     velocity; name the ones this case needs with --patch xMin:patch.")
    else:
        print(f"  no `box` face is left over, and cfMesh adds the `{args.default_name}`"
              " patch only if something is (renameBoundaryPatches.C:151-161), so on a"
              " surface")
        print("     carrying just these solids the mesh gets exactly the patches named"
              " above. Any OTHER solid in the surface still merges into"
              f" `{args.default_name}`.")
    print("\n" + "\n".join(route(args.surface, args.wall, args.angle)))
    print("\n" + COSTS)
    return 0


def _cmd_prepare(args) -> int:
    findings = prepare(args.case)
    print(report(findings))
    if args.write:
        changed = apply_preparation(args.case)
        print("\napplied:")
        for line in changed or ["nothing to change"]:
            print(f"  {line}")
        if changed:
            print("  then: foamFormatConvert -constant")
    return 0


def _cmd_check(args) -> int:
    findings = check(args.case, args.wall, args.log, args.ref, args.wall_spacing)
    if args.json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        print(report(findings))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("where", help="is cfMesh on this instance, and all of it")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_where)

    p = sub.add_parser("box", help="a body and its domain box as one closed STL")
    p.add_argument("body", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default=None, help="solid name for a binary STL, which has none")
    p.add_argument("--box", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    p.add_argument("--upstream", type=float, default=UPSTREAM)
    p.add_argument("--downstream", type=float, default=DOWNSTREAM)
    p.add_argument("--side", type=float, default=SIDE)
    p.add_argument("--angle", type=float, default=45.0,
                   help="feature angle for surfaceFeatureEdges, in the printed route")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_box)

    p = sub.add_parser("case", help="write system/meshDict for a cartesianMesh run")
    p.add_argument("case", type=Path)
    p.add_argument("--surface", required=True, help="the closed surface, .stl or .ftr")
    p.add_argument("--max-cell", type=float, required=True)
    p.add_argument("--wall", default=None, help="the solid name the layers go on")
    p.add_argument("--wall-cell", type=float, default=None)
    p.add_argument("--layers", type=int, default=0)
    p.add_argument("--ratio", type=float, default=1.2)
    p.add_argument("--symmetry", action="append", default=None,
                   help="a box face that is a cut plane, typed `symmetry`; repeatable")
    p.add_argument("--patch", action="append", default=None, metavar="NAME[:TYPE]",
                   help="a solid to keep as its own patch, TYPE defaulting to `patch`."
                        " Anything not named here, by --wall or by --symmetry is MERGED"
                        " into one --default-name patch; repeatable")
    p.add_argument("--default-name", default="farField",
                   help="the one patch every unnamed solid is merged into")
    p.add_argument("--default-type", default="patch",
                   help="its type")
    p.add_argument("--angle", type=float, default=45.0,
                   help="feature angle for surfaceFeatureEdges, in the printed route")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_case)

    p = sub.add_parser("prepare", help="an existing mesh made readable by cfMesh")
    p.add_argument("case", type=Path)
    p.add_argument("--write", action="store_true", help="make the two edits")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("check", help="what the mesh got, against what was asked for")
    p.add_argument("case", type=Path)
    p.add_argument("--wall", default=None)
    p.add_argument("--log", type=Path, default=None, help="the cartesianMesh log, for trap 1")
    p.add_argument("--ref", type=Path, default=None,
                   help="the same wall meshed with no layer request; makes the"
                        " coverage number stand on its own")
    p.add_argument("--wall-spacing", type=float, default=None,
                   help="a reference wall spacing in metres, when no such mesh is around")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OSError as exc:
        # A missing or unreadable input used to come out as a bare traceback while
        # every other failure came out as one sentence; two shapes of failure in one
        # script is one more than a session should have to recognise.
        #
        # This is wider than it reads: it also swallows an OSError raised by a bug in
        # this script -- a write to a path it built wrongly, say -- into the same
        # "refused" line, with no traceback to find it by. An errno here that does not
        # obviously name one of the paths the command was given is a bug in this file
        # and not a problem with the input; rerun with the body of `main` inlined, or
        # under `python -X dev`, before believing the sentence.
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (SurfaceError, DictError, OverwriteError) as exc:
        # A refusal is this script's most useful output: every one of them stands for a
        # round of the Wigley comparison that ended in a mesh with no layers and exit 0.
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
