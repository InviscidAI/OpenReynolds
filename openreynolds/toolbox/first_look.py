#!/usr/bin/env python3
"""Everything worth seeing about a case that has been meshed but not yet run, as
one picture.

The panels -- geometry, the whole mesh, a close-up on whatever the mesh was refined
around, the patches with their names, and the counts as text -- are each useful on
their own, and each was already renderable one command at a time. What was missing
was doing it in one go: a session spent five round-trips rendering one view after
another, and only after the fifth did it turn out the STL had been read in
millimetres, which the first picture had shown all along. Five renders is five
chances to look at the wrong one first. So the last thing this writes is a contact
sheet -- every panel captioned, side by side, in a single PNG -- and one `read_file`
on it is the whole state of the case before a solver has burned a core-hour.

It degrades rather than fails. No STL is a sentence on the geometry panel, not an
exception; a panel that raises loses that panel and nothing else, and its error is
printed on the sheet where the picture would have been, because a blank square that
does not say why is worse than no square.

The close-up picks its own region: the smallest boundary patch that is not part of
the domain box (the obstacle, the aerofoil, the step), and failing that the box
around the smallest cells in the mesh, which is where the refinement went and so
where somebody meant the interesting thing to be. `--roi` overrides it by name.

    python3 first_look.py /work/case
    python3 first_look.py /work/case --out /work/case/renders/first_look
    python3 first_look.py /work/case --panels mesh patches stats
    python3 first_look.py /work/case --roi cylinder

The mesh is read at the earliest available time, so this costs the same on a case
with 400 write directories as on one with none.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state  # noqa: E402  (sibling script, not a package)


SURFACE_SUFFIXES = (".stl", ".obj", ".ply", ".vtk", ".vtp", ".vtu", ".stlb")

SURFACE_DIRS = ("constant/triSurface", "constant/geometry", "constant/trisurface")

PANEL_KINDS = {
    "geometry": "geometry-preview",
    "mesh": "mesh-full",
    "closeup": "mesh-closeup",
    "patches": "mesh-patches",
    "stats": "other",
}

PANEL_ORDER = ("geometry", "mesh", "closeup", "patches", "stats")

COLOURS = (
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
    "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd",
    "#7f9dcf", "#e8a87c", "#8fbf8f", "#d98b8e", "#a99ac9",
)

VIEW_SIZE = (1000, 750)
"""One panel, rendered by pyvista. 4:3 so it drops into the sheet cell unpadded."""

PANEL_INCHES = (6.0, 4.6)
SHEET_DPI = 120

EDGE_BUDGET = 60_000
"""Above this many faces in one view, drawing every edge fills the panel with black
and the mesh reads as a solid. Below it the edges are the whole point."""

STATS_WIDTH = 64
"""The stats panel is a sixth of a contact sheet, not a terminal. Lines wider than
this shrink the font until nothing is legible, so they are kept from happening."""


class Missing(Exception):
    """Something a panel needs is not in this case.

    Separate from every other exception because it is not a fault: a case with no
    STL genuinely has no geometry to draw, and the panel should say that in plain
    words rather than print a traceback class at the reader.
    """


class Panel:
    """One cell of the contact sheet: a rendered image, or text, or the reason for
    neither.

    Written out by hand rather than as a dataclass because the toolbox scripts are
    loaded by path -- `importlib.util.spec_from_file_location` without a `sys.modules`
    entry -- and `@dataclass` resolves string annotations through `sys.modules`,
    which under that idiom is not there yet.
    """

    def __init__(self, name, kind, title, image=None, text="", note="", ok=False):
        self.name = name
        self.kind = kind
        self.title = title
        self.image = image
        self.text = text
        self.note = note
        self.ok = ok

    def __repr__(self) -> str:
        return f"Panel({self.name!r}, ok={self.ok!r}, note={self.note!r})"


class Scene:
    """What one read of the case gives us: the volume mesh and its patches."""

    def __init__(self, internal=None, patches=None, time=0.0):
        self.internal = internal
        self.patches = patches if patches is not None else {}
        self.time = time


# -- panel plumbing ----------------------------------------------------------------


def attempt(func):
    """Run one panel's work. Returns `(value, note)`; the note is empty on success.

    Every panel goes through here so that a failure is worth exactly one panel. The
    alternative -- letting the first broken renderer take the process down -- is how
    a case with an unreadable STL used to produce no pictures at all, including the
    four that would have rendered fine.
    """
    try:
        return func(), ""
    except Missing as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a lost panel, not a lost sheet
        return None, f"{type(exc).__name__}: {exc}"


def build_panels(specs) -> list[Panel]:
    """`specs` is `(name, kind, title, builder)`; a builder returns a Path to a PNG
    it wrote, or a string to be set as text."""
    panels: list[Panel] = []
    for name, kind, title, builder in specs:
        value, note = attempt(builder)
        panel = Panel(name=name, kind=kind, title=title, note=note)
        if isinstance(value, Path):
            panel.image = value
            panel.ok = True
        elif isinstance(value, str) and value.strip():
            panel.text = value
            panel.ok = True
        elif not note:
            panel.note = "produced nothing"
        panels.append(panel)
    return panels


CAPTION_WIDTH = 78
"""Characters that fit across one panel at the caption's font size. Wrapped rather
than left to run: an unwrapped caption is drawn straight over the neighbouring panel,
and matplotlib will not tell you it did it."""


def panel_caption(panel: Panel) -> str:
    """Title, and the reason underneath it when there is one."""
    if not panel.note:
        return panel.title
    note = textwrap.shorten(panel.note, width=3 * CAPTION_WIDTH, placeholder=" ...")
    return f"{panel.title}\n" + textwrap.fill(note, width=CAPTION_WIDTH)


# -- the contact sheet -------------------------------------------------------------


def grid_shape(count: int, max_cols: int = 3) -> tuple[int, int]:
    """Rows and columns for `count` panels, as square as it can be.

    Capped at three columns: the sheet is read by opening one image, and past three
    across the captions are too small to read at the size an image viewer picks.
    """
    if count <= 0:
        return (0, 0)
    cols = min(max_cols, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    return (rows, cols)


def draw_panel(axes, panel: Panel) -> None:
    """One cell: the caption above it, and a bordered square below.

    Every cell keeps its border whatever went into it, so a panel that produced
    nothing is still visibly a panel with a caption saying why, rather than white
    space that reads as the end of the sheet.
    """
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color("#b0bec5")
        spine.set_linewidth(0.8)
    axes.set_title(panel_caption(panel), fontsize=8.5, loc="left", color="black")

    body = ""
    centred = False
    if panel.image is not None and Path(panel.image).exists():
        try:
            # aspect="auto" fills the cell: the renders are 4:3 and so is the cell,
            # so nothing is stretched noticeably and no letterbox appears.
            axes.imshow(mpimg.imread(str(panel.image)), aspect="auto")
            return
        except Exception as exc:  # noqa: BLE001
            body = f"image written but not readable back\n{type(exc).__name__}: {exc}"
    elif panel.text:
        body = panel.text
    else:
        body = "(no picture -- see the caption above)"
        centred = True

    axes.set_facecolor("#fafafa")
    if centred:
        axes.text(0.5, 0.5, body, transform=axes.transAxes, family="monospace",
                  fontsize=7.5, va="center", ha="center", color="#78909c")
    else:
        axes.text(0.02, 0.97, body, transform=axes.transAxes, family="monospace",
                  fontsize=7.0, va="top", ha="left", color="black")


def compose_sheet(panels: list[Panel], out: Path, title: str = "") -> Path:
    """The panels, captioned, in one PNG.

    matplotlib and not pyvista: this is a page layout, and the panels arriving here
    are already-rendered PNGs plus a block of text. Nothing on this image can encode
    a video, but a single wide PNG is the one composite format that both a person
    and a `read_file` can take in at once.
    """
    if not panels:
        raise Missing("no panels to compose")
    rows, cols = grid_shape(len(panels))
    figure, grid = plt.subplots(
        rows, cols,
        figsize=(PANEL_INCHES[0] * cols, PANEL_INCHES[1] * rows + 0.5),
        dpi=SHEET_DPI, squeeze=False,
    )
    cells = [cell for row in grid for cell in row]
    for cell in cells[len(panels):]:
        cell.set_axis_off()
    for cell, panel in zip(cells, panels):
        draw_panel(cell, panel)
    if title:
        figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(out), facecolor="white")
    plt.close(figure)
    return out


# -- what is on disk, without opening the mesh -------------------------------------


OWNER_NOTE = re.compile(r"note\s+\"(.*?)\"", re.DOTALL)
NOTE_FIELD = re.compile(r"(nPoints|nCells|nFaces|nInternalFaces)\s*:\s*(\d+)")
BOUNDARY_ENTRY = re.compile(
    r"^[ \t]*(\w[\w.\-]*)[ \t]*\r?\n[ \t]*\{(.*?)\}", re.MULTILINE | re.DOTALL
)
"""A patch is a bare name on one line and a brace block under it. The count and the
opening parenthesis above them are not `name` followed by `{`, so they fall out."""
KEY_VALUE = re.compile(r"(\w+)\s+([^;]+);")


def read_maybe_gz(path: Path) -> str:
    """polyMesh files are routinely gzipped by writeCompression, and a stats panel
    that goes blank because of that is a stats panel that lies."""
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def find_file(directory: Path, stem: str) -> Path | None:
    for candidate in (directory / stem, directory / f"{stem}.gz"):
        if candidate.is_file():
            return candidate
    return None


def polymesh_dir(case: Path) -> Path | None:
    """`constant/polyMesh`, or a processor's copy of it when the mesh was decomposed
    and never reconstructed -- the counts are per-processor then, which the stats
    panel says out loud rather than quietly reporting a fifth of the mesh."""
    direct = case / "constant" / "polyMesh"
    if find_file(direct, "boundary") or find_file(direct, "owner"):
        return direct
    for processor in sorted(case.glob("processor*")):
        candidate = processor / "constant" / "polyMesh"
        if find_file(candidate, "boundary") or find_file(candidate, "owner"):
            return candidate
    return None


def parse_owner_note(text: str) -> dict:
    """The counts OpenFOAM writes into the header note of `owner`.

    They are already there, so the cheapest cell count in the world is a regex over
    a few hundred bytes rather than a full mesh read.
    """
    match = OWNER_NOTE.search(text)
    if not match:
        return {}
    return {key: int(value) for key, value in NOTE_FIELD.findall(match.group(1))}


def parse_boundary(text: str) -> list[dict]:
    """Patch name, type and face count out of `constant/polyMesh/boundary`."""
    body = text
    header = body.find("FoamFile")
    if header >= 0:
        closing = body.find("}", header)
        if closing >= 0:
            body = body[closing + 1:]
    patches: list[dict] = []
    for name, block in BOUNDARY_ENTRY.findall(body):
        entry = {"name": name, "type": "", "nFaces": 0}
        for key, value in KEY_VALUE.findall(block):
            value = value.strip()
            if key == "type":
                entry["type"] = value
            elif key == "nFaces":
                entry["nFaces"] = int(value) if value.isdigit() else 0
            elif key == "inGroups":
                entry["inGroups"] = value
        patches.append(entry)
    return patches


def find_surfaces(case: Path) -> list[Path]:
    """Surface files a snappy case would have been built from.

    De-duplicated by resolved path: `triSurface` and `trisurface` are both spelled
    in the wild and are the same directory on a case-insensitive filesystem, which
    otherwise lists every STL twice and draws it twice.
    """
    found: list[Path] = []
    seen: set[str] = set()

    def take(paths) -> None:
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in SURFACE_SUFFIXES:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(path)

    for relative in SURFACE_DIRS:
        directory = case / relative
        if directory.is_dir():
            take(sorted(directory.iterdir()))
    if not found:
        take(sorted(case.glob("*")))
    return found


def disk_stats(case: Path) -> dict:
    """Counts and patches read straight off the files, so the stats panel still has
    something to say when the mesh itself will not open."""
    stats: dict = {
        "case": case.name,
        "mesh_dir": "",
        "counts": {},
        "patches": [],
        "notes": [],
        "bounds": None,
        "cell_volume": None,
    }
    directory = polymesh_dir(case)
    if directory is None:
        stats["notes"].append("no constant/polyMesh -- this case has not been meshed")
        return stats
    try:
        stats["mesh_dir"] = str(directory.relative_to(case)).replace("\\", "/")
    except ValueError:
        stats["mesh_dir"] = str(directory)
    if "processor" in stats["mesh_dir"]:
        stats["notes"].append("counts are for one processor: the mesh is decomposed")

    owner = find_file(directory, "owner")
    if owner is not None:
        stats["counts"] = parse_owner_note(read_maybe_gz(owner))
        if not stats["counts"]:
            stats["notes"].append("owner has no header note, so no counts from it")

    boundary = find_file(directory, "boundary")
    if boundary is not None:
        stats["patches"] = parse_boundary(read_maybe_gz(boundary))
    return stats


def human(value) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def format_stats(stats: dict) -> str:
    """The counts, the extents and the patch table as fixed-width text.

    No thresholds and no verdicts: a 12 million cell mesh is either exactly right or
    a mistake depending on what is about to be run on it, and that is not this
    script's call to make.
    """
    lines = [f"case      {stats.get('case') or '?'}"]
    if stats.get("mesh_dir"):
        lines.append(f"mesh      {stats['mesh_dir']}")

    counts = stats.get("counts") or {}
    labels = (
        ("nCells", "cells"),
        ("nPoints", "points"),
        ("nFaces", "faces"),
        ("nInternalFaces", "internal faces"),
    )
    present = [(label, counts[key]) for key, label in labels if key in counts]
    if present:
        lines.append("")
        for label, value in present:
            lines.append(f"  {label:<16}{human(value):>14}")

    bounds = stats.get("bounds")
    if bounds is not None and len(bounds) == 6:
        low = np.asarray(bounds[0::2], dtype=float)
        high = np.asarray(bounds[1::2], dtype=float)
        extent = high - low
        lines.append("")
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            lines.append(
                f"  {axis} {low[index]:>12.5g} .. {high[index]:<12.5g}"
                f" span {extent[index]:.4g}"
            )
        lines.append("  (OpenFOAM reads these as metres)")

    volume = stats.get("cell_volume")
    if volume is not None and len(volume) == 2:
        smallest, largest = float(volume[0]), float(volume[1])
        lines.append("")
        lines.append(f"  cell volume     {smallest:.3g} .. {largest:.3g}")
        if smallest > 0 and largest > 0:
            lines.append(
                f"  cube-root size  {smallest ** (1 / 3):.3g} .. {largest ** (1 / 3):.3g}"
            )

    patches = stats.get("patches") or []
    if patches:
        lines.append("")
        lines.append(f"patches ({len(patches)})")
        lines.append(f"  {'name':<22}{'type':<14}{'faces':>10}")
        for patch in patches[:12]:
            name = str(patch.get("name", ""))[:21]
            kind = str(patch.get("type", ""))[:13]
            lines.append(f"  {name:<22}{kind:<14}{human(patch.get('nFaces', 0)):>10}")
        if len(patches) > 12:
            lines.append(f"  ... and {len(patches) - 12} more")

    for note in stats.get("notes") or []:
        lines.append("")
        lines.append(textwrap.fill(note, width=STATS_WIDTH))

    return "\n".join(line[:STATS_WIDTH + 12] for line in lines)


# -- choosing what to look at closely ----------------------------------------------


def _tolerance(domain_bounds) -> float:
    bounds = np.asarray(domain_bounds, dtype=float)
    span = float(np.linalg.norm(bounds[1::2] - bounds[0::2]))
    return max(span * 1e-6, 1e-12)


def classify_patch(bounds, domain_bounds) -> dict:
    """Whether a patch is part of the box around the case, and how much of the box
    it leans on.

    Three shapes turn up and only the third is worth a close-up:

    - flat against one face of the domain (an inlet, an outlet, a moving lid);
    - filling the domain box in all three directions (a `walls` patch wrapping
      several sides at once, or the `frontAndBack` pair of a 2D case);
    - anything else -- an obstacle sitting inside the flow.

    `touches` counts the directions in which a patch reaches a domain face without
    spanning it, which separates a cylinder floating in a channel (0) from a lower
    wall that runs the length of the domain along its floor (1).
    """
    patch = np.asarray(bounds, dtype=float)
    domain = np.asarray(domain_bounds, dtype=float)
    tol = _tolerance(domain)

    flat = False
    spans = 0
    touches = 0
    for axis in range(3):
        low, high = patch[2 * axis], patch[2 * axis + 1]
        dlow, dhigh = domain[2 * axis], domain[2 * axis + 1]
        at_low = abs(low - dlow) <= tol
        at_high = abs(high - dhigh) <= tol
        thin = abs(high - low) <= tol
        if thin and (at_low or at_high):
            flat = True
        if at_low and at_high and not thin:
            spans += 1
        elif at_low or at_high:
            touches += 1
    return {"outer": bool(flat or spans == 3), "touches": touches, "spans": spans}


def patch_size(description: dict) -> float:
    """Area when it is known, face count when it is not. Either orders the patches
    the same way for the purpose here, which is only ever 'which is smallest'."""
    area = description.get("area")
    if area is not None and float(area) > 0:
        return float(area)
    return float(description.get("n_cells") or 0)


def choose_region(descriptions, domain_bounds, prefer: str = ""):
    """Pick the patch the close-up should frame. Returns `(description, why)`.

    Smallest-first among the patches that are not part of the domain box, because
    the small non-box patch is the thing the mesh was refined around in nearly every
    external-flow case there is. It is a heuristic and it is named as one on the
    panel; `--roi` exists for when it guesses wrong.
    """
    rows = list(descriptions or [])
    if not rows:
        return None, "the case exposes no boundary patches"

    if prefer:
        for row in rows:
            if str(row.get("name", "")).lower() == prefer.lower():
                return row, f"named on the command line: {row.get('name')}"
        available = ", ".join(str(row.get("name", "")) for row in rows)
        raise Missing(f"no patch called '{prefer}'; this case has: {available}")

    candidates = []
    for row in rows:
        bounds = row.get("bounds")
        if bounds is None or len(bounds) != 6:
            continue
        verdict = classify_patch(bounds, domain_bounds)
        if verdict["outer"]:
            continue
        candidates.append((verdict["touches"], patch_size(row), str(row.get("name", "")), row))

    if not candidates:
        return None, "every patch lies on the domain box"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    touches, _size, name, row = candidates[0]
    detached = "clear of the domain box" if touches == 0 else f"touching it on {touches} side(s)"
    return row, f"smallest patch that is not the domain box: {name}, {detached}"


def dense_region(centers, volumes, fraction: float = 0.02, min_cells: int = 50):
    """The box around the smallest cells in the mesh.

    The fallback when every patch belongs to the domain box -- a lid-driven cavity
    has nothing else. Wherever the cells are smallest is wherever somebody spent
    refinement, which is the best available guess at what the case is about.
    """
    centers = np.asarray(centers, dtype=float)
    volumes = np.asarray(volumes, dtype=float).ravel()
    if centers.size == 0 or volumes.size == 0 or centers.shape[0] != volumes.shape[0]:
        return None
    finite = np.isfinite(volumes)
    if not finite.any():
        return None
    centers = centers[finite]
    volumes = volumes[finite]
    wanted = max(min_cells, int(round(fraction * volumes.size)))
    wanted = min(wanted, volumes.size)
    order = np.argsort(volumes, kind="stable")[:wanted]
    picked = centers[order]
    return (
        float(picked[:, 0].min()), float(picked[:, 0].max()),
        float(picked[:, 1].min()), float(picked[:, 1].max()),
        float(picked[:, 2].min()), float(picked[:, 2].max()),
    )


def focus_bounds(bounds, domain_bounds, pad: float = 2.5):
    """Grow a region about its own centre and clip it to the domain.

    Padded well past the object on purpose: a close-up cropped to the obstacle shows
    the obstacle and not the refinement transition around it, and the transition is
    where the cell-size jumps that wreck a solve are visible. Directions with no
    extent -- the span axis of a one-cell-thick 2D case, or a planar patch -- get a
    floor off the domain size instead of being scaled from zero.
    """
    region = np.asarray(bounds, dtype=float)
    domain = np.asarray(domain_bounds, dtype=float)
    domain_extent = domain[1::2] - domain[0::2]
    floor = 0.02 * float(np.max(np.abs(domain_extent))) if domain_extent.size else 0.0

    out = np.empty(6, dtype=float)
    for axis in range(3):
        low, high = region[2 * axis], region[2 * axis + 1]
        middle = 0.5 * (low + high)
        half = max(0.5 * (high - low) * pad, floor)
        out[2 * axis] = max(middle - half, domain[2 * axis])
        out[2 * axis + 1] = min(middle + half, domain[2 * axis + 1])
        if out[2 * axis] >= out[2 * axis + 1]:
            out[2 * axis] = domain[2 * axis]
            out[2 * axis + 1] = domain[2 * axis + 1]
    return tuple(float(value) for value in out)


def camera_for(bounds) -> str:
    """A view direction from the shape of the domain.

    A 2D OpenFOAM case is one cell thick and an isometric view of it is a picture of
    its `frontAndBack` patch, edge-on, showing nothing. Thin in one direction means
    look down that direction.
    """
    extent = np.asarray(bounds, dtype=float)[1::2] - np.asarray(bounds, dtype=float)[0::2]
    if extent.size != 3 or not np.all(np.isfinite(extent)):
        return "iso"
    largest = float(np.max(extent))
    if largest <= 0:
        return "iso"
    thin = [axis for axis in range(3) if extent[axis] <= 0.05 * largest]
    if len(thin) == 1:
        return {0: "yz", 1: "xz", 2: "xy"}[thin[0]]
    return "iso"


def patch_colours(names) -> dict:
    """One colour per patch, by position, so the same case renders the same twice."""
    return {str(name): COLOURS[index % len(COLOURS)] for index, name in enumerate(names)}


# -- rendering (pyvista lives inside these) ----------------------------------------


def _pyvista():
    """Imported here and not at module scope so that everything above can be tested
    on a machine with no OSMesa in it."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


def open_case(case: Path) -> Scene:
    """The volume mesh and its patches, at the earliest time on disk."""
    pv = _pyvista()
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    reader.cell_to_point_creation = True
    try:
        reader.enable_all_patch_arrays()
    except Exception:  # noqa: BLE001 - older readers expose every patch anyway
        pass
    times = list(reader.time_values)
    chosen = float(times[0]) if times else 0.0
    try:
        reader.set_active_time_value(chosen)
    except Exception:  # noqa: BLE001
        pass
    block = reader.read()
    scene = Scene(internal=_internal_mesh(block, pv), patches=_patch_blocks(block, pv), time=chosen)
    if scene.internal is None or scene.internal.n_cells == 0:
        raise Missing("the case opens but has no internalMesh -- it has not been meshed")
    return scene


def _internal_mesh(block, pv):
    if isinstance(block, pv.MultiBlock):
        keys = list(block.keys() or [])
        if "internalMesh" in keys:
            return block["internalMesh"]
        for item in block:
            if item is None:
                continue
            found = _internal_mesh(item, pv)
            if found is not None:
                return found
        return None
    return block


def _patch_blocks(block, pv) -> dict:
    """Every named block that is not the internal mesh, flattened to name -> surface."""
    found: dict = {}
    if not isinstance(block, pv.MultiBlock):
        return found
    for name in list(block.keys() or []):
        item = block[name]
        if item is None or name == "internalMesh":
            continue
        if isinstance(item, pv.MultiBlock):
            found.update(_patch_blocks(item, pv))
        elif getattr(item, "n_cells", 0):
            found[str(name)] = item
    return found


def describe_patches(patches: dict) -> list[dict]:
    """Patch blocks reduced to the plain numbers `choose_region` decides on."""
    rows: list[dict] = []
    for name, mesh in patches.items():
        row = {"name": name, "n_cells": int(getattr(mesh, "n_cells", 0) or 0)}
        try:
            row["bounds"] = tuple(float(value) for value in mesh.bounds)
        except Exception:  # noqa: BLE001
            row["bounds"] = None
        try:
            row["area"] = float(mesh.area)
        except Exception:  # noqa: BLE001
            row["area"] = None
        rows.append(row)
    return rows


def _finish(plotter, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    return out


def render_geometry(case: Path, out: Path) -> Path:
    """The input surfaces, drawn with their own scale bars.

    Whatever encloses everything else goes translucent and the caption says so,
    because an unexplained transparent box reads as a property of the geometry.
    """
    # Looked for before pyvista is imported, so that a case with no geometry says
    # "no geometry" whatever the state of the graphics stack.
    files = find_surfaces(case)
    if not files:
        raise Missing(
            "no surface file under constant/triSurface -- either this is a blockMesh "
            "case or the geometry has not been copied in yet"
        )
    pv = _pyvista()
    parts = []
    skipped = []
    for path in files[:10]:
        try:
            mesh = pv.read(str(path))
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            parts.append((path.name, mesh))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{path.name}: {type(exc).__name__}")
    if not parts:
        raise Missing("surface files present but none could be read: " + "; ".join(skipped))

    spans = [float(np.linalg.norm(np.asarray(mesh.bounds)[1::2] - np.asarray(mesh.bounds)[0::2]))
             for _name, mesh in parts]
    biggest = int(np.argmax(spans)) if len(parts) > 1 else -1

    plotter = pv.Plotter(off_screen=True, window_size=VIEW_SIZE)
    try:
        plotter.enable_depth_peeling(10)
    except Exception:  # noqa: BLE001
        pass
    order = [i for i in range(len(parts)) if i != biggest] + ([biggest] if biggest >= 0 else [])
    for slot in order:
        name, mesh = parts[slot]
        plotter.add_mesh(
            mesh,
            color=COLOURS[slot % len(COLOURS)],
            show_edges=mesh.n_cells <= EDGE_BUDGET,
            edge_color="black",
            line_width=0.3,
            opacity=0.25 if slot == biggest else 1.0,
            label=f"{name} ({mesh.n_cells:,} faces)",
        )
    plotter.show_bounds(grid="back", location="outer", ticks="outside", fmt="%.3g", font_size=8)
    if len(parts) > 1:
        plotter.add_legend(bcolor="white", size=(0.35, min(0.35, 0.06 * len(parts))), loc="lower right")
        plotter.add_text(f"{parts[biggest][0]} translucent (it encloses the rest)",
                         font_size=7, position="upper_left", color="dimgray")
    plotter.camera_position = "iso"
    return _finish(plotter, out)


def render_mesh_full(scene: Scene, out: Path) -> Path:
    """The outside of the mesh, whole, with the domain ticks on it."""
    pv = _pyvista()
    surface = scene.internal.extract_surface()
    plotter = pv.Plotter(off_screen=True, window_size=VIEW_SIZE)
    plotter.add_mesh(
        surface,
        color="#d9dee6",
        show_edges=surface.n_cells <= EDGE_BUDGET,
        edge_color="#37474f",
        line_width=0.3,
    )
    if surface.n_cells > EDGE_BUDGET:
        plotter.add_text(
            f"edges hidden: {surface.n_cells:,} surface faces", font_size=7,
            position="upper_left", color="dimgray",
        )
    plotter.show_bounds(grid="back", location="outer", ticks="outside", fmt="%.3g", font_size=8)
    plotter.camera_position = camera_for(scene.internal.bounds)
    return _finish(plotter, out)


def render_closeup(scene: Scene, region, out: Path, label: str = "") -> Path:
    """The mesh inside one box, with every cell edge drawn.

    `clip_box(..., invert=False)` keeps what is inside the box; the default keeps
    what is outside it, which produces a picture of the whole domain with a bite out
    of exactly the part you wanted to look at.
    """
    pv = _pyvista()
    box = focus_bounds(region, scene.internal.bounds)
    try:
        inside = scene.internal.clip_box(list(box), invert=False)
    except Exception:  # noqa: BLE001
        inside = None
    if inside is None or inside.n_cells == 0:
        inside = scene.internal
        box = tuple(float(value) for value in scene.internal.bounds)

    plotter = pv.Plotter(off_screen=True, window_size=VIEW_SIZE)
    plotter.add_mesh(
        inside,
        color="white",
        show_edges=inside.n_cells <= EDGE_BUDGET,
        edge_color="#263238",
        line_width=0.35,
    )
    plotter.show_bounds(grid="back", location="outer", ticks="outside", fmt="%.3g", font_size=8)
    caption = label or "region of interest"
    plotter.add_text(f"{caption}  ({inside.n_cells:,} cells shown)", font_size=8,
                     position="upper_left", color="dimgray")
    plotter.camera_position = camera_for(box)
    return _finish(plotter, out)


def render_patches(scene: Scene, out: Path) -> Path:
    """Every boundary patch in its own colour, with the names in a legend.

    The names are the point. A boundary condition is written against a patch name
    and there is no other picture in the pipeline that says which surface carries
    which one, which is how `inlet` ends up on the outflow face.
    """
    pv = _pyvista()
    if not scene.patches:
        raise Missing(
            "the reader exposed no boundary patches -- constant/polyMesh/boundary "
            "still lists them, see the stats panel"
        )
    names = list(scene.patches.keys())
    colours = patch_colours(names)

    # On a 2D case the `empty` patches are the two faces you are looking at, and they
    # are most of the boundary by area: drawn from an iso camera they cover every
    # other patch and the panel is a coloured slab with a legend beside it. Seen in a
    # live run -- 8,476 frontAndBack faces hiding all six patches. So they are left
    # out and the case is looked at square on, which is how a 2D case is read anyway.
    flat = [name for name in names if is_empty_patch(name, scene)]
    drawn = [name for name in names if name not in flat] or names
    plotter = pv.Plotter(off_screen=True, window_size=VIEW_SIZE)
    entries = []
    for name in names:
        mesh = scene.patches[name]
        if name in drawn:
            plotter.add_mesh(mesh, color=colours[name], show_edges=False, opacity=1.0)
        label = f"{name} ({int(mesh.n_cells):,})"
        entries.append((label + ("  [hidden]" if name not in drawn else ""), colours[name]))
    plotter.add_legend(labels=entries, bcolor="white", size=(0.38, min(0.6, 0.05 * len(entries) + 0.05)),
                       loc="lower right")
    plotter.show_bounds(grid="back", location="outer", ticks="outside", fmt="%.3g", font_size=8)
    # Iso either way. Looking straight down z was the obvious answer for a 2D case
    # and the wrong one: with the flat faces hidden, what is left are the *side*
    # patches, which are edge-on from there -- the panel came back all but empty.
    # From an iso camera the same patches read as the rim of the domain: inlet on
    # one edge, outlet on the other, the body in the middle.
    plotter.camera_position = "iso"
    return _finish(plotter, out)


EMPTY_PATCH_HINTS = ("frontandback", "front_and_back", "empty", "defaultfaces")
"""Patch names that mean "the two faces of a 2D case". The reader does not carry the
patch *type* through, so the name is what there is; `constant/polyMesh/boundary` is
the authority and the stats panel prints it."""


def is_empty_patch(name: str, scene: "Scene") -> bool:
    """Whether this patch is one of the flat faces of a one-cell-thick case.

    By name, and then confirmed by size: a patch carrying most of the boundary is
    the flat face whatever it is called, and a patch called `empty` that is a sliver
    is not. Both together are harder to fool than either.
    """
    if not any(hint in name.lower() for hint in EMPTY_PATCH_HINTS):
        return False
    mesh = scene.patches.get(name)
    total = sum(int(getattr(m, "n_cells", 0) or 0) for m in scene.patches.values())
    here = int(getattr(mesh, "n_cells", 0) or 0)
    return bool(total) and here >= 0.4 * total


def scene_stats(scene: Scene) -> dict:
    """The numbers only a mesh read can give: bounds and the cell volume range."""
    extra: dict = {
        "bounds": tuple(float(value) for value in scene.internal.bounds),
        "counts": {"nCells": int(scene.internal.n_cells), "nPoints": int(scene.internal.n_points)},
    }
    try:
        sized = scene.internal.compute_cell_sizes(length=False, area=False, volume=True)
        volumes = np.asarray(sized.cell_data["Volume"], dtype=float)
        # A zero or negative cell volume is a broken cell, not a dense one. Blanked
        # to NaN so the close-up is not dragged to a single inverted cell, and
        # `dense_region` drops what is not finite.
        usable = np.where(np.isfinite(volumes) & (volumes > 0), volumes, np.nan)
        if np.isfinite(usable).any():
            finite = usable[np.isfinite(usable)]
            extra["cell_volume"] = (float(finite.min()), float(finite.max()))
            extra["centers"] = np.asarray(sized.cell_centers().points, dtype=float)
            extra["volumes"] = usable
    except Exception:  # noqa: BLE001
        pass
    return extra


# -- the one call ------------------------------------------------------------------


def first_look(case: Path, out: Path | None = None, wanted=PANEL_ORDER,
               roi: str = "", label: str = "") -> dict:
    """Render the panels that were asked for and compose them into one sheet."""
    case = Path(case).resolve()
    if not case.is_dir():
        # Checked before anything is created. A mistyped path used to produce a
        # renders/ tree, a .reynolds/ state directory and a confident sheet saying
        # the case had not been meshed, which is a different problem from the one
        # the reader actually has.
        raise Missing(f"no such case directory: {case}")
    out = Path(out) if out else case / "renders" / "first_look"
    out.mkdir(parents=True, exist_ok=True)
    root = study_state.find_root(case)
    study_state.set_phase("preview", "running", root=root, case=case.name)

    wanted = [name for name in PANEL_ORDER if name in set(wanted)]
    stats = disk_stats(case)

    scene, scene_note = (None, "")
    if any(name in wanted for name in ("mesh", "closeup", "patches", "stats")):
        scene, scene_note = attempt(lambda: open_case(case))
        if scene is None and scene_note:
            # Said on the stats panel too, so that a sheet of empty squares is not
            # the only clue that the mesh itself would not open.
            stats["notes"].append(f"the mesh could not be opened: {scene_note}")

    region, why = (None, "")
    if scene is not None:
        extra = scene_stats(scene)
        stats["bounds"] = extra.get("bounds")
        stats["cell_volume"] = extra.get("cell_volume")
        if not stats.get("counts"):
            stats["counts"] = extra.get("counts", {})
        if "closeup" in wanted:
            # choose_region returns a pair; attempt wraps it in another one, and a
            # bad --roi name arrives here as the note rather than the pair.
            picked, note = attempt(
                lambda: choose_region(describe_patches(scene.patches), scene.internal.bounds, roi)
            )
            region, why = picked if picked is not None else (None, note)
            if region is None and extra.get("centers") is not None:
                fallback = dense_region(extra["centers"], extra["volumes"])
                if fallback is not None:
                    region = fallback
                    why = f"{why}; framing the smallest cells instead"

    def needs_scene(builder):
        def run():
            if scene is None:
                raise Missing(scene_note or "the case could not be opened")
            return builder()
        return run

    def closeup():
        if region is None:
            raise Missing(why or "no region of interest could be chosen")
        target = region["bounds"] if isinstance(region, dict) else region
        name = region.get("name", "") if isinstance(region, dict) else ""
        return render_closeup(scene, target, out / "closeup.png", name)

    builders = {
        "geometry": ("geometry: input surfaces",
                     lambda: render_geometry(case, out / "geometry.png")),
        "mesh": ("mesh: the whole domain",
                 needs_scene(lambda: render_mesh_full(scene, out / "mesh_full.png"))),
        "closeup": ("mesh: close-up on the region of interest",
                    needs_scene(closeup)),
        "patches": ("patches: one colour each, named in the legend",
                    needs_scene(lambda: render_patches(scene, out / "patches.png"))),
        "stats": ("counts and extents", lambda: format_stats(stats)),
    }

    specs = [(name, PANEL_KINDS[name], builders[name][0], builders[name][1]) for name in wanted]
    panels = build_panels(specs)

    if region is not None and why:
        for panel in panels:
            if panel.name == "closeup" and panel.ok:
                panel.note = why

    for panel in panels:
        if panel.ok and panel.image is not None:
            study_state.record(panel.kind, panel.image, root=root, case=case.name,
                               label=label or panel.title)

    # Only when the stats panel was asked for: `--panels geometry` writing a stats
    # file and putting a row in the manifest for it is a side effect nobody ordered.
    stats_path = None
    if "stats" in wanted:
        stats_path = out / "stats.txt"
        stats_path.write_text(format_stats(stats) + "\n", encoding="utf-8")
        study_state.record("other", stats_path, root=root, case=case.name,
                           label="first look: mesh counts as text")

    sheet, sheet_note = attempt(
        lambda: compose_sheet(panels, out / "first_look.png",
                              title=f"first look -- {case.name}")
    )
    if sheet is not None:
        study_state.record("contact-sheet", sheet, root=root, case=case.name,
                           label=label or f"first look: {case.name}")
        study_state.set_phase("preview", "done", root=root, case=case.name,
                              note=f"{sum(1 for p in panels if p.ok)}/{len(panels)} panels")
    else:
        study_state.set_phase("preview", "failed", root=root, case=case.name, note=sheet_note)

    return {"panels": panels, "sheet": sheet, "sheet_note": sheet_note,
            "stats": stats_path, "out": out, "region": why}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, help="an OpenFOAM case directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="where the PNGs go (default <case>/renders/first_look)")
    parser.add_argument("--panels", nargs="+", default=list(PANEL_ORDER),
                        choices=list(PANEL_ORDER),
                        help="which panels to render (default: all five)")
    parser.add_argument("--roi", default="",
                        help="frame the close-up on this patch instead of guessing")
    parser.add_argument("--label", default="", help="label recorded in the manifest")
    args = parser.parse_args(argv)

    try:
        result = first_look(args.case, args.out, args.panels, args.roi, args.label)
    except Missing as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for panel in result["panels"]:
        where = panel.image or ("(text)" if panel.text else "-")
        state = "ok  " if panel.ok else "none"
        print(f"{state} {panel.name:<9} {where}")
        if panel.note:
            print(f"       {panel.note}")
    if result["stats"] is not None:
        print(f"     stats     {result['stats']}")
    if result["sheet"] is not None:
        print(f"\ncontact sheet ({sum(1 for p in result['panels'] if p.ok)}"
              f"/{len(result['panels'])} panels): {result['sheet']}")
        return 0
    print(f"\nno contact sheet: {result['sheet_note']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
