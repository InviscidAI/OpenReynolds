"""The case opened in ParaView on the user's machine, on this study's camera and colours.

    python3 paraview_export.py /work/case --out /work/case/paraview

ParaView is **not in this image** and is not going to be: `docker/workspace` builds
pyvista on `vtk-osmesa` and an image change is a 1.2 GB rebuild. So this script runs no
ParaView and converts nothing. It writes one small Python file that ParaView *there*
runs:

    paraview --script=case_paraview.py        # or Tools > Python Script Editor > Run
    pvpython case_paraview.py                 # headless, writes a PNG

Everything it needs is already in the study directory the mirror brought home. The
mirror copies the case whole -- `system/`, `constant/polyMesh`, every written time
(`mirror.py:10-29`) -- and drops only the formats that are the same data again:
`.vtk`, `.vtu`, `.vtp`, `.vtm`, `.pvd` and the empty `.foam` marker, "written for a
viewer that is not here" (`mirror.py:195-197`). That list is right and is left alone.
The marker is one empty file, so the generated script creates its own rather than
asking the mirror to carry it, and ParaView's `OpenFOAMReader` then reads the case
directly -- which is also how this repo reads it (`render.py:92-99`).

## What makes this an export rather than a file path

A picture is a comparison only if the camera and the colour limits are the same ones.
Both are already pinned here, into this repo's own sidecars, with the measurement
written down beside them:

- `render.py:487-498` -- `diff_record` carries `clim` over both runs, `cmap`, the
  diverging `diff_clim` centred on zero, and the camera as `normal`/`view_vector`/
  `viewup`, because "two panels each scaled to their own range are not a comparison".
- `animate.py:405-444` -- `build_sidecar` writes `frames.json` with the sequence's one
  `clim`, `cmap`, `normal` and `field`; a `--bare` sequence "carries its colour range
  only here".
- `showcase.py:717-729` -- the 3D camera: `235` degrees round the subject, `22` degrees
  up, at `1.15` times the bounding box diagonal, looking at its centre, up `+z`.

Those numbers went into PNGs and nowhere a user's ParaView could read them. This script
reads whichever of those sidecars exist and writes them into the state script, so the
window that opens is the figure, with its scale bar on the same two numbers. With no
sidecar it says so in `sources` and leaves ParaView to rescale to the data range --
which is the honest answer, not a made-up limit.

It decides nothing about the case: no field is computed, no file in the case is
modified, and the exit code is 0 unless the case directory is not one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PRESETS = {
    "viridis": "Viridis (matplotlib)",
    "plasma": "Plasma (matplotlib)",
    "inferno": "Inferno (matplotlib)",
    "magma": "Magma (matplotlib)",
    "coolwarm": "Cool to Warm",
    "bwr": "Cool to Warm",
    "turbo": "Turbo",
    "jet": "Jet",
    "gray": "X Ray",
    "grey": "X Ray",
}
"""matplotlib and pyvista colour map names against the ParaView presets that are the
same ramp. ParaView ships the matplotlib four under those exact long names, so a figure
drawn here on `viridis` and reopened there is the same ramp and not an approximation of
it. A name not in here is left to ParaView's default rather than guessed at, and the
manifest says the preset was not set."""

CASE_MARKERS = ("system/controlDict", "system", "constant")
"""What makes a directory an OpenFOAM case. `system/controlDict` is the one file every
case has and no other kind of directory does; the two directory names are the fallback
for a case whose dictionaries are still being written."""


def is_case(case: Path) -> bool:
    return (case / "system" / "controlDict").exists() or (
        (case / "system").is_dir() and (case / "constant").is_dir())


def read_sidecars(case: Path) -> list[dict]:
    """Every render/animation sidecar under the case, newest first.

    `render.py:578` writes `<stem>_difference.json` beside the PNGs and `animate.py`
    writes `frames.json` into the frames directory; `showcase.py:738` writes a thinner
    `frames.json` of its own. All three are searched for rather than any one being
    assumed, because which of them a study produced depends entirely on what the agent
    chose to run.
    """
    found = []
    for path in sorted(case.glob("**/*.json")):
        if path.name != "frames.json" and not path.name.endswith("_difference.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data["_from"] = str(path)
            found.append(data)
    return sorted(found, key=lambda d: Path(d["_from"]).stat().st_mtime, reverse=True)


def pick_view(sidecars: list[dict], field: str = "", clim=None) -> dict:
    """The field, colour limits, ramp and camera to pin, and where each came from.

    A later sidecar wins over an earlier one and an explicit flag wins over both, so
    `--field p --clim 0 120` is always exactly what you get. Each value records its
    own source: a limit a user cannot trace back to a run is a limit they are right
    not to trust.
    """
    view: dict = {"field": "", "clim": None, "cmap": "", "normal": "",
                  "view_vector": None, "viewup": None, "sources": {}}
    for data in reversed(sidecars):  # oldest first, so the newest overwrites
        origin = data.get("_from", "")
        for key in ("field", "cmap", "normal", "view_vector", "viewup"):
            if data.get(key):
                view[key] = data[key]
                view["sources"][key] = origin
        if data.get("clim"):
            view["clim"] = [float(data["clim"][0]), float(data["clim"][1])]
            view["sources"]["clim"] = origin
        # `render.py`'s difference sidecar names its field under `field` and the runs
        # under `runs`; its `scalar` sibling in `animate.py` is the array as VTK names
        # it, which is what ParaView wants when the two differ.
        if data.get("scalar"):
            view["field"] = data["scalar"]
            view["sources"]["field"] = origin
    if field:
        view["field"] = field
        view["sources"]["field"] = "--field"
    if clim is not None:
        view["clim"] = [float(clim[0]), float(clim[1])]
        view["sources"]["clim"] = "--clim"
    view["preset"] = PRESETS.get(str(view["cmap"]).lower(), "")
    return view


SCRIPT = '''"""Open {case_name} in ParaView, on the camera and colours its figures used.

Written by openreynolds `paraview_export.py` for the case at {case!r}, quoted
rather than written plain: a Windows path's separators are string escapes in a
docstring and made the generated file unparseable. ParaView 5.9 or newer.

    paraview --script={script_name}      # the GUI, with this already set up
    pvpython {script_name}               # headless; writes {png_name}

Nothing here reads the internet and nothing here edits the case: the one file it
creates is the empty `.foam` marker ParaView's OpenFOAM reader needs to be pointed at,
whose contents the reader ignores (the same marker `render.py:92-99` makes).
"""

import os

from paraview.simple import *  # noqa: F403 - this is how ParaView scripts are written

CASE = os.path.dirname(os.path.abspath(__file__)) if {in_case} else {case!r}
FIELD = {field!r}
CLIM = {clim!r}
PRESET = {preset!r}
VIEW_VECTOR = {view_vector!r}
VIEWUP = {viewup!r}
ORBIT_ANGLE = {orbit_angle!r}
ORBIT_ELEVATION = {orbit_elevation!r}
ORBIT_RADIUS_FACTOR = {orbit_radius!r}

marker = os.path.join(CASE, os.path.basename(CASE.rstrip(os.sep)) + ".foam")
if not os.path.exists(marker):
    open(marker, "w").close()

case = OpenFOAMReader(registrationName="case", FileName=marker)
case.MeshRegions = ["internalMesh"]
case.CaseType = "Reconstructed Case"
case.Createcelltopointfiltereddata = 1
case.UpdatePipeline()

# The last write time, which is what every figure in this study was drawn at unless it
# said otherwise. `animate.py` and `render.py` both default to the latest time.
times = list(case.TimestepValues) or [0.0]
GetAnimationScene().UpdateAnimationUsingDataTimingInformation()
view = GetActiveViewOrCreate("RenderView")
view.ViewTime = times[-1]

display = Show(case, view)
display.SetRepresentationType("Surface")
if FIELD:
    ColorBy(display, ("POINTS", FIELD))
    bar = GetScalarBar(GetColorTransferFunction(FIELD), view)
    bar.Title, bar.ComponentTitle = FIELD, ""
    display.SetScalarBarVisibility(view, True)
    transfer = GetColorTransferFunction(FIELD)
    if PRESET:
        transfer.ApplyPreset(PRESET, True)
    if CLIM:
        # Pinned, not rescaled: these are the two numbers the study's own figures were
        # drawn on, so the scale bar here reads the same as the scale bar there.
        transfer.RescaleTransferFunction(CLIM[0], CLIM[1])
        GetOpacityTransferFunction(FIELD).RescaleTransferFunction(CLIM[0], CLIM[1])
    else:
        display.RescaleTransferFunctionToDataRange(True, False)

bounds = case.GetDataInformation().GetBounds()
centre = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0,
          (bounds[4] + bounds[5]) / 2.0]
span = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
diagonal = sum(s * s for s in span) ** 0.5 or 1.0

view.CameraFocalPoint = centre
if VIEW_VECTOR:
    # A slice figure's camera: straight down the cut normal, as `render.py`'s `aim()`
    # does, under parallel projection because a flat thing seen in perspective is a
    # parallelogram with most of its area foreshortened (`showcase.py:742-746`).
    view.CameraPosition = [c + v * diagonal for c, v in zip(centre, VIEW_VECTOR)]
    view.CameraViewUp = list(VIEWUP or [0.0, 0.0, 1.0])
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.55 * max(
        [s for s, v in zip(span, VIEW_VECTOR) if abs(v) < 0.5] or [diagonal])
else:
    # The 3D camera `showcase.py:717-729` uses for every still it draws.
    import math

    theta, phi = math.radians(ORBIT_ANGLE), math.radians(ORBIT_ELEVATION)
    radius = diagonal * ORBIT_RADIUS_FACTOR
    view.CameraPosition = [centre[0] + radius * math.cos(theta) * math.cos(phi),
                           centre[1] + radius * math.sin(theta) * math.cos(phi),
                           centre[2] + radius * math.sin(phi)]
    view.CameraViewUp = [0.0, 0.0, 1.0]

Render()

# Saved whichever way this ran, rather than on a guess about whether there is a window.
# Under `pvpython` it is the only output there is; in the GUI it is a file beside the
# script and the window is still there. Asking the view whether it has an interactor is
# the other option and it is a ParaView version question nobody here can test.
SaveScreenshot(os.path.join(os.path.dirname(os.path.abspath(__file__)), {png_name!r}),
               view, ImageResolution=[1600, 1200])
'''

ORBIT = (235.0, 22.0, 1.15)
"""`showcase.py:717-729`'s camera, as the three numbers it is: the angle round the
subject, the elevation, and the multiple of the bounding-box diagonal it stands off at.
Repeated here rather than imported because this script must parse and run with nothing
installed -- it is written to be read by a ParaView on another machine."""


def write_script(case: Path, out_dir: Path, view: dict) -> Path:
    """The state script, in `out_dir`, named after the case."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{case.name}_paraview.py"
    in_case = out_dir.resolve() == case.resolve()
    text = SCRIPT.format(
        case=str(case), case_name=case.name, script_name=name, in_case=in_case,
        png_name=f"{case.name}_paraview.png",
        field=str(view.get("field") or ""),
        clim=view.get("clim"), preset=str(view.get("preset") or ""),
        view_vector=view.get("view_vector"), viewup=view.get("viewup"),
        orbit_angle=ORBIT[0], orbit_elevation=ORBIT[1], orbit_radius=ORBIT[2],
    )
    path = out_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def export(case: Path, out_dir: Path, *, field: str = "", clim=None) -> dict:
    """Write the state script and report what was pinned into it and from where."""
    sidecars = read_sidecars(case)
    view = pick_view(sidecars, field=field, clim=clim)
    script = write_script(case, out_dir, view)
    return {
        "case": str(case),
        "script": str(script),
        "open_with": [f"paraview --script={script.name}", f"pvpython {script.name}"],
        "field": view["field"] or None,
        "clim": view["clim"],
        "cmap": view["cmap"] or None,
        "preset": view["preset"] or None,
        "camera": "slice normal" if view["view_vector"] else "orbit",
        "view_vector": view["view_vector"],
        "viewup": view["viewup"],
        "sources": view["sources"],
        "sidecars_read": [d["_from"] for d in sidecars],
        "note": ("the colour limits and camera are this study's own, read from the "
                 "sidecars listed" if view["sources"] else
                 "no render sidecar found: ParaView will rescale to the data range"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path, help="the OpenFOAM case directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the script (default: <case>/paraview)")
    parser.add_argument("--field", default="",
                        help="the field to colour by; default is whatever the study's "
                             "own sidecars last drew")
    parser.add_argument("--clim", nargs=2, type=float, default=None,
                        metavar=("LOW", "HIGH"),
                        help="colour limits, overriding the sidecars'")
    args = parser.parse_args(argv)
    case = args.case
    if not is_case(case):
        print(f"{case} is not an OpenFOAM case: no system/controlDict and no "
              "system/ beside a constant/")
        return 2
    report = export(case, args.out or (case / "paraview"), field=args.field, clim=args.clim)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
