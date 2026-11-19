#!/usr/bin/env python3
"""A few fixed pyvista scenes: mesh cuts and field slices, saved as PNGs.

Headless via OSMesa. Fixed cameras so two runs of the same case are a visual diff
rather than two unrelated pictures.

`--fields` takes solver outputs (`U`, `p`, `k`, ...) and also the quantities a solver
does not write but a study usually wants to see: `vorticity` (the out-of-plane
component, drawn signed so the two rows of a von Karman street read as opposite colours)
and `Q` (the Q-criterion). Those are functions of the velocity gradient, computed here
from `U`, so they no longer need a hand-written pyvista script.

`--contours N` puts N iso-contour lines over a field slice, at levels taken from that
field's own range. A colour-mapped slice says where a quantity is high; the lines say
where it is *equal*, which is what makes a gradient legible and what every textbook
figure of a pressure field is made of.

`--diff OTHER` is the fixed cameras cashed in: one field, two runs, drawn as two panels
on one colour scale plus their difference on a diverging map centred on zero. It refuses
rather than drawing when the two runs are not of the same domain.

    python3 render.py /work/case                      # latest time, all default scenes
    python3 render.py /work/case --fields U p --normal z --time 500
    python3 render.py /work/case --fields vorticity Q  # derived from U
    python3 render.py /work/case --fields p --contours 12
    python3 render.py /work/case_B --fields p --diff /work/case_A
    python3 render.py /work/case --scene mesh --out /work/case/renders
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# pyvista is imported inside the functions that draw rather than at the top of the
# file, so the camera geometry below -- NORMALS, VIEWUP, aim, frame -- can be imported
# and tested anywhere. `results.py` has been written this way since it had the camera
# fix; this file kept the top-level import, which is why the tests that cover the fix
# could not load it outside the container. Nothing above the drawing functions touches pv.


def _pyvista():
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


NORMALS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def open_case(case: Path, time: float | None):
    """The case as pyvista blocks, at the chosen (default latest) write time.

    The fast path is pyvista's `OpenFOAMReader`. It fails on a field file the VTK
    reader cannot parse -- a `0/` value that is a `$variable`, a `flowVelocity` or
    `Tinf` symbol, or an `#includeEtc` directive -- with `Error reading line N of
    ./0/U`, and 73 studies met that and hand-wrote a pyvista script to get around
    it. `foamToVTK` does not have that limitation: it reads through OpenFOAM's own
    libraries, which expand includes and resolve variables, so when the reader
    raises we fall back to it. It is slower (it writes files), which is why it is
    the fallback and not the primary path.
    """
    pv = _pyvista()
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    try:
        reader = pv.OpenFOAMReader(str(foam))
        times = list(reader.time_values)
        chosen = times[-1] if times else 0.0
        if time is not None and times:
            chosen = min(times, key=lambda t: abs(t - time))
        reader.set_active_time_value(chosen)
        reader.cell_to_point_creation = True
        block = reader.read()
        if internal_mesh(block) is None:
            raise ValueError("no internalMesh from OpenFOAMReader")
        return block, chosen, times
    except Exception as exc:  # the reader choked -- go through foamToVTK
        print(f"OpenFOAMReader failed ({type(exc).__name__}: {exc}); "
              f"falling back to foamToVTK", flush=True)
        return open_case_via_foamtovtk(case, time)


def open_case_via_foamtovtk(case: Path, time: float | None):
    """Read the case through `foamToVTK`, the include/variable-safe path.

    Runs `foamToVTK` (which needs the OpenFOAM environment the toolbox is invoked
    under) to write `<case>/VTK/`, then reads the newest internal-mesh output pyvista
    understands. foamToVTK's layout has changed across releases -- modern builds write
    `VTK/<name>_<index>/internal.vtu`, older ones `VTK/<name>_<index>.vtk` -- so we
    take whichever internal-mesh file is newest rather than assume one shape.
    """
    import subprocess

    pv = _pyvista()
    argv = ["foamToVTK", "-case", str(case)]
    if time is None:
        argv.append("-latestTime")
    else:
        argv += ["-time", f"{time:g}"]
    proc = subprocess.run(argv, cwd=str(case), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "could not read this case: OpenFOAMReader failed and foamToVTK also failed "
            f"(exit {proc.returncode}). Last lines:\n" + "\n".join(proc.stderr.splitlines()[-8:])
        )

    newest = newest_internal_vtk(case / "VTK")
    if newest is None:
        raise SystemExit(f"foamToVTK wrote nothing readable under {case / 'VTK'}")
    block = pv.read(str(newest))
    stem = newest.parent.name if newest.name == "internal.vtu" else newest.stem
    chosen = time_from_stem(stem)
    return block, chosen, [chosen]


def newest_internal_vtk(vtk_dir: Path) -> Path | None:
    """The most recently written internal-mesh file foamToVTK left, or None.

    Modern foamToVTK writes `<name>_<index>/internal.vtu`; older builds write
    `<name>_<index>.vtk` directly in `VTK/`. We accept either and take the newest by
    mtime -- the `-latestTime`/`-time` write we just made -- rather than assuming a
    layout or that the highest index is the wanted time.
    """
    candidates = list(vtk_dir.glob("**/internal.vtu")) + list(vtk_dir.glob("*.vtk"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def time_from_stem(stem: str) -> float:
    """The write time read out of a foamToVTK output name, else 0.0.

    `motorBike_300` -> 300.0, `cavity_12` -> 12.0. The trailing numeric token is the
    write index/time; anything unparseable leaves 0.0 rather than failing a render.
    """
    for token in reversed(stem.replace("_", " ").split()):
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


def internal_mesh(block):
    """The internalMesh block, whatever nesting this case produced."""
    pv = _pyvista()
    if isinstance(block, pv.MultiBlock):
        if "internalMesh" in (block.keys() or []):
            return block["internalMesh"]
        for item in block:
            if item is None:
                continue
            found = internal_mesh(item)
            if found is not None:
                return found
        return None
    return block


def slice_at(mesh, normal: str, origin=None):
    """The cut through `origin`, the mesh's own centre by default.

    `origin` is passed in for a comparison of two runs: cutting each through its own
    centre is two different planes wherever the two meshes' extents differ by a cell,
    and a difference between two planes is not a difference between two runs.
    """
    if origin is None:
        origin = np.array(mesh.center)
    return mesh.slice(normal=NORMALS[normal], origin=origin)


VIEWUP = {"z": (0.0, 1.0, 0.0), "y": (0.0, 0.0, 1.0), "x": (0.0, 0.0, 1.0)}
"""Which way is up, per slice normal.

Without it VTK picks, and for a z-normal slice it picks something collinear with the
view direction and falls back to an arbitrary up vector: a 28x16 domain came out drawn
as a portrait column with the flow running bottom to top and 60% of the canvas empty.
`results.py` and `animate.py` have carried this dict since they were written; this file
was the one that never got it, and every study that called `render.py --scene mesh`
paid for the omission by hand-writing its own pyvista script instead."""


def aim(plotter, normal: str) -> None:
    """Point the camera down the slice normal, with x across the page.

    `plotter.camera_position = "z"` reads as if it would do this and does not: the
    string form takes a view *plane* ("xy", "xz", "yz"), and a single axis letter is
    not one of them. A live study hit this, worked out the fix and applied it to its
    own copy under `/work/.toolbox/` -- which is the copy that gets overwritten from
    the distribution at the start of the next session, so the fix has been rediscovered
    since. Naming the direction as a vector says what is meant and survives the sync;
    naming the up vector with it is what keeps the streamwise axis horizontal.

    Parallel projection because these are slices: perspective on a flat cut makes the
    far side of the domain smaller than the near side, so a uniform mesh reads as graded.
    """
    plotter.view_vector(NORMALS[normal], viewup=VIEWUP.get(normal, (0.0, 1.0, 0.0)))
    plotter.enable_parallel_projection()


def frame(plotter, cut, zoom: float | None, bounds: tuple | None) -> None:
    """Fill the canvas with the part worth looking at.

    A close-up was impossible without this: asked for the mesh around a step, the tool
    drew the whole domain as a horizontal sliver with the step invisible, so the study
    wrote its own script twice. `bounds` crops to a region, `zoom` magnifies about the
    centre; neither is applied unless asked for, so the default picture is unchanged.
    """
    if bounds is not None:
        plotter.reset_camera(bounds=list(bounds))
    if zoom:
        plotter.camera.zoom(zoom)


def save(plotter, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    return out


def canonical(name: str) -> str:
    """A field name reduced so `vorticity`, `Vorticity`, `vort` and `q-criterion` all
    reach the same entry. Case, underscores, hyphens and the |mag| bars are noise here."""
    return name.lower().replace("_", "").replace("-", "").replace("|", "")


DERIVED_FROM_U = {
    "vorticity": "vorticity", "vort": "vorticity", "curl": "vorticity",
    "q": "qcriterion", "qcriterion": "qcriterion", "qcrit": "qcriterion",
}
"""Field names the solver never writes, mapped to the VTK quantity that makes them.

`foamToVTK` and the OpenFOAMReader carry only what is on disk -- `U`, `p`, the turbulence
fields -- and vorticity and the Q-criterion are neither: they are contractions of grad(U).
So `render.py --fields vorticity` used to print "no field named vorticity" and 73 studies
answered it by hand-writing a pyvista script that ran `compute_derivative`. That is the
one call this needs; it is done once, on the volume mesh, so the ordinary slice-and-colour
path draws the result like any written field."""

SIGNED_FIELDS = frozenset({"vorticity_z"})
"""Fields drawn about zero with a diverging map. Vorticity in a wake is signed -- the two
rows of a shed street turn opposite ways -- and a sequential map (viridis) hides the sign
that is the whole reason to plot it, so those get `coolwarm` centred on zero instead."""


def add_derived(mesh, field: str):
    """Compute `field` from `U` onto `mesh` when the solver did not write it.

    Returns the name of the scalar array to colour by, or None when `field` is neither a
    written field nor one we know how to derive. The gradient needs the cells, so this
    runs on the volume mesh before any slice is taken; the slice then inherits the array.
    """
    quantity = DERIVED_FROM_U.get(canonical(field))
    if quantity is None:
        return None
    if "U" not in mesh.point_data and "U" not in mesh.cell_data:
        return None  # nothing to take a gradient of
    source = mesh if "U" in mesh.point_data else mesh.cell_data_to_point_data()
    kwargs = {"scalars": "U", quantity: True}
    derived = source.compute_derivative(**kwargs)
    values = np.asarray(derived.point_data[quantity])
    if quantity == "vorticity":
        # The out-of-plane component is the one a 2D wake is read by; keep the signed z
        # component as the default and the magnitude alongside it for a 3D case.
        mesh.point_data["vorticity_z"] = values[:, 2]
        mesh.point_data["vorticity_mag"] = np.linalg.norm(values, axis=1)
        return "vorticity_z"
    mesh.point_data[quantity] = values
    return quantity


def resolve(mesh, field: str) -> str | None:
    """The array name `field` is drawn from: itself if written, else derived from `U`."""
    if field in mesh.point_data or field in mesh.cell_data:
        return field
    return add_derived(mesh, field)


def as_scalar(values):
    """One number per point: a vector field becomes its magnitude.

    Both a contour level and a difference are statements about a single number, and a
    vector field has three. The magnitude is the one a picture of `U` or of
    `wallShearStress` is already drawn on, so it is taken here rather than in each
    caller deciding differently.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim > 1 and data.shape[-1] > 1:
        return np.linalg.norm(data, axis=-1)
    return data.reshape(-1)


def point_values(mesh, name: str):
    """`(mesh, values)` for `name` as point data, converting from cells if need be.

    Contouring and differencing both work point by point -- a contour line passes
    between points, and a difference is taken at one -- and a cell array has neither.
    Returns the converted mesh alongside the values because the conversion makes a new
    object and the caller has to draw the one the numbers came off.
    """
    if name not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    data = mesh.point_data.get(name)
    if data is None:
        return mesh, None
    return mesh, as_scalar(data)


def contour_levels(values, count: int) -> list[float]:
    """`count` iso-contour levels across the field's own range.

    The ends are deliberately left off. `linspace(min, max, count)` puts the first and
    last level exactly on the extremes, where the iso-line is a single point or runs
    along the domain edge and draws nothing -- so a request for 10 lines comes back as
    8 lines and 2 invisible ones. Taking the interior of `count + 2` steps is `count`
    lines that all have somewhere to go, which is why the count is honoured rather than
    merely asked for.
    """
    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if count < 1 or data.size == 0:
        return []
    low, high = float(data.min()), float(data.max())
    if not high > low:
        return []  # a uniform field has no interior level
    return [float(level) for level in np.linspace(low, high, count + 2)[1:-1]]


CONTOUR_COLOUR = "black"
CONTOUR_WIDTH = 1.6


# -- two runs, one picture -------------------------------------------------------------

BOX_TOLERANCE = 0.01
"""How far two runs' bounding boxes may disagree, as a fraction of the first's extent,
before they are not the same domain. A millimetre of difference between two meshes of
one case is a mesher's rounding; a tenth of the domain is a different problem."""

MIN_COVERAGE = 0.999
"""How much of the first run's cut must find a value in the second when the two meshes
are not the same mesh. Below this the picture is mostly the difference between a field
and nothing, drawn in the same colours as a real difference."""


def correspondence(first, second, name_first: str = "A", name_second: str = "B",
                   *, box_tolerance: float = BOX_TOLERANCE) -> str:
    """Whether -- and how -- a field on `second` can be subtracted from one on `first`.

    `"same-mesh"` when the two carry the same number of points over the same box: the
    subtraction is then point by point and exact. `"resampled"` when the domain is the
    same and the discretisation is not, which is the ordinary case of a mesh study and
    is handled by sampling one onto the other. Otherwise this raises, naming both runs:
    a difference between two different domains is a plausible picture of nothing, and a
    plausible picture of nothing is worse than no picture.
    """
    before = np.asarray(first.bounds, dtype=float)
    after = np.asarray(second.bounds, dtype=float)
    span = np.maximum(before[1::2] - before[::2], 1e-12)
    apart = np.abs(after - before).reshape(3, 2).max(axis=1) / span
    worst = float(apart.max())
    if worst > box_tolerance:
        axis = "xyz"[int(np.argmax(apart))]
        raise SystemExit(
            f"refusing to draw a difference: {name_first} and {name_second} are not "
            f"over the same domain -- their bounds are {100 * worst:.1f}% of the extent "
            f"apart on {axis} (tolerance {100 * box_tolerance:g}%). "
            f"{name_first} spans {tuple(round(float(v), 6) for v in before)}, "
            f"{name_second} spans {tuple(round(float(v), 6) for v in after)}. "
            "A difference between two different domains is a picture of nothing."
        )
    if first.n_points == second.n_points and worst <= 1e-9:
        # The same count over the same box is not yet the same mesh. A mesh whose points
        # arrive in another order passes both of those and then subtracts one run's
        # value at a point from the other run's value somewhere else -- the shapes
        # agree, nothing raises, and the picture is noise drawn in the right colours.
        # The coordinates settle it, and a mesh that fails it is not refused: resampling
        # is exactly the right answer for one.
        here = getattr(first, "points", None)
        there = getattr(second, "points", None)
        if here is None or there is None or np.allclose(here, there):
            return "same-mesh"
    return "resampled"


def difference_on(first, second, scalars: str, name_first: str, name_second: str):
    """`(first, a, b, how)`: the two runs' values for `scalars` on `first`'s points.

    The resampled path samples onto a *stripped copy* of the first cut rather than onto
    the cut itself, because `sample` overwrites an array of the same name: sampling onto
    the cut as it stands would leave `p` meaning the first run wherever the second had
    no value and the second run everywhere else, which is the one failure this function
    exists to prevent.
    """
    how = correspondence(first, second, name_first, name_second)
    first, a = point_values(first, scalars)
    if a is None:
        raise SystemExit(f"{name_first} carries no {scalars!r} on this cut")
    if how == "same-mesh":
        _, b = point_values(second, scalars)
    else:
        empty = first.copy()
        empty.clear_data()
        sampled = empty.sample(second)
        mask = np.asarray(sampled.point_data.get("vtkValidPointMask",
                                                 np.ones(sampled.n_points)))
        covered = float(np.mean(mask > 0)) if mask.size else 0.0
        if covered < MIN_COVERAGE:
            raise SystemExit(
                f"refusing to draw a difference: resampling {name_second} onto "
                f"{name_first}'s cut found a value at {100 * covered:.1f}% of its points, "
                f"under the {100 * MIN_COVERAGE:g}% these two meshes would have to "
                "correspond to for the difference to mean anything."
            )
        _, b = point_values(sampled, scalars)
    if b is None:
        raise SystemExit(f"{name_second} carries no {scalars!r} to difference against")
    if a.shape != b.shape:
        raise SystemExit(
            f"refusing to draw a difference: {name_first} gives {a.shape[0]} values for "
            f"{scalars!r} on this cut and {name_second} gives {b.shape[0]}"
        )
    return first, a, b, how


def diff_record(a, b, *, field: str, normal: str, names, how: str) -> dict:
    """The numbers both runs and their difference are drawn on, as a dict.

    Written beside the PNGs as a sidecar, the way `animate.py` records the `clim` it
    used. A difference view is only a comparison if both panels are on one colour scale
    and one camera, and that is a claim about numbers -- so the numbers are recorded
    and can be checked, rather than asserted in a docstring and trusted. The difference
    itself goes on `coolwarm` centred on zero, which is what `SIGNED_FIELDS` already
    does for vorticity: zero is the answer "these two runs agree here", and it has to
    be a colour you can find.
    """
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    finite = delta[np.isfinite(delta)]
    reach = float(np.max(np.abs(finite))) if finite.size else 0.0
    both = np.concatenate([np.asarray(a)[np.isfinite(a)], np.asarray(b)[np.isfinite(b)]])
    return {
        "field": field,
        "runs": [str(name) for name in names],
        "correspondence": how,
        "points": int(delta.size),
        # One scale over both runs. Each panel scaled to its own range is two pictures
        # that cannot be read against each other, which is the whole failure mode.
        "clim": [float(both.min()), float(both.max())] if both.size else [0.0, 1.0],
        "cmap": "viridis",
        "diff_clim": [-reach, reach] if reach > 0 else [-1.0, 1.0],
        "diff_cmap": "coolwarm",
        "max_abs_difference": reach,
        "identical": bool(reach == 0.0),
        # One camera over both panels: this is `aim()`'s pair of vectors, written down.
        "normal": normal,
        "view_vector": [float(v) for v in NORMALS[normal]],
        "viewup": [float(v) for v in VIEWUP.get(normal, (0.0, 1.0, 0.0))],
    }


def render_panel(cut, scalars: str, out: Path, *, normal: str, clim, cmap: str,
                 label: str, zoom: float | None = None,
                 bounds: tuple | None = None) -> Path:
    """One panel of a comparison: a slice drawn on limits handed to it.

    `render_field` works out its own limits from its own data, which is right for a
    single picture and wrong for a pair. Here the limits arrive as an argument so that
    one caller decides them once for every panel.
    """
    plotter = _pyvista().Plotter(off_screen=True, window_size=(1100, 800))
    plotter.add_mesh(cut, scalars=scalars, cmap=cmap, clim=list(clim), show_edges=False)
    plotter.add_scalar_bar(title=scalars, n_labels=5)
    aim(plotter, normal)
    frame(plotter, cut, zoom, bounds)
    plotter.add_text(label, font_size=10)
    return save(plotter, out)


def render_difference(first, second, field: str, normal: str, out_dir: Path, names,
                      zoom: float | None = None, bounds: tuple | None = None):
    """Two runs of one field as two panels and their difference, plus the record of it.

    Returns `(record, paths)`, or `(None, [])` when the field is in neither run. Both
    panels share the cut plane, the camera and the colour limits; only the difference
    gets its own scale, and that one is symmetric about zero.
    """
    name_first, name_second = (str(name) for name in names)
    scalars = resolve(first, field)
    if scalars is None or resolve(second, field) != scalars:
        return None, []
    origin = np.array(first.center)
    cut_first = slice_at(first, normal, origin)
    cut_second = slice_at(second, normal, origin)
    if cut_first.n_points == 0 or cut_second.n_points == 0:
        raise SystemExit(f"the {normal}-normal cut is empty in one of these two runs")

    cut_first, a, b, how = difference_on(cut_first, cut_second, scalars,
                                         name_first, name_second)
    cut_second, own = point_values(cut_second, scalars)
    record = diff_record(a, b, field=scalars, normal=normal,
                         names=(name_first, name_second), how=how)

    # The magnitude is written back under the array's own name so that what is drawn is
    # the number that was differenced, not a second reduction of the same vector.
    cut_first[scalars] = a
    cut_second[scalars] = own
    delta = cut_first.copy()
    delta_name = f"delta {scalars}"
    delta[delta_name] = a - b

    stem = f"{field}_{normal}"
    panels = dict(normal=normal, zoom=zoom, bounds=bounds)
    paths = [
        render_panel(cut_first, scalars, out_dir / f"{stem}_{name_first}.png",
                     clim=record["clim"], cmap=record["cmap"],
                     label=f"{name_first} -- {scalars}", **panels),
        render_panel(cut_second, scalars, out_dir / f"{stem}_{name_second}.png",
                     clim=record["clim"], cmap=record["cmap"],
                     label=f"{name_second} -- {scalars}", **panels),
        render_panel(delta, delta_name, out_dir / f"{stem}_difference.png",
                     clim=record["diff_clim"], cmap=record["diff_cmap"],
                     label=f"{name_first} minus {name_second} -- {scalars}", **panels),
    ]
    sidecar = out_dir / f"{stem}_difference.json"
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    paths.append(sidecar)
    return record, paths


def render_field(mesh, field: str, normal: str, out: Path, zoom: float | None = None,
                 bounds: tuple | None = None, contours: int = 0) -> Path | None:
    name = resolve(mesh, field)
    if name is None:
        return None
    cut = slice_at(mesh, normal)
    data = cut.point_data.get(name, cut.cell_data.get(name))
    if data is None:
        return None
    scalars = name
    if data.ndim > 1 and data.shape[1] > 1:
        cut[f"|{name}|"] = np.linalg.norm(data, axis=1)
        scalars = f"|{name}|"

    # A signed field reads about zero on a diverging map; everything else on viridis.
    cmap = "coolwarm" if scalars in SIGNED_FIELDS else "viridis"
    clim = None
    if scalars in SIGNED_FIELDS:
        extent = float(np.max(np.abs(data))) or 1.0
        clim = (-extent, extent)

    plotter = _pyvista().Plotter(off_screen=True, window_size=(1100, 800))
    plotter.add_mesh(cut, scalars=scalars, cmap=cmap, clim=clim, show_edges=False)
    plotter.add_scalar_bar(title=scalars, n_labels=5)

    # Iso-lines over the colour, not instead of it: the map says where the quantity is
    # high and the lines say where it is equal, and the pair of them is how a gradient
    # is read. Only drawn when asked for, so the default picture is the one it was.
    source, at_points = point_values(cut, scalars) if contours else (cut, None)
    levels = contour_levels(at_points, contours) if contours else []
    if levels:
        lines = source.contour(isosurfaces=levels, scalars=scalars)
        if lines.n_points:
            plotter.add_mesh(lines, color=CONTOUR_COLOUR, line_width=CONTOUR_WIDTH,
                             show_scalar_bar=False)

    aim(plotter, normal)
    frame(plotter, cut, zoom, bounds)
    label = f"{scalars} — {normal}-normal slice"
    if levels:
        label += f" -- {len(levels)} iso-contours"
    plotter.add_text(label, font_size=10)
    return save(plotter, out)


def render_mesh(mesh, normal: str, out: Path, zoom: float | None = None,
                bounds: tuple | None = None) -> Path:
    cut = slice_at(mesh, normal)
    plotter = _pyvista().Plotter(off_screen=True, window_size=(1100, 800))
    plotter.add_mesh(cut, color="white", show_edges=True, edge_color="black", line_width=0.4)
    aim(plotter, normal)
    frame(plotter, cut, zoom, bounds)
    label = f"mesh — {normal}-normal cut"
    if bounds is not None or zoom:
        label += " (close-up)"
    plotter.add_text(label, font_size=10)
    return save(plotter, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="defaults to <case>/renders")
    parser.add_argument("--fields", nargs="*", default=["U", "p"],
                        help="solver outputs (U, p, k, ...) or derived: vorticity, Q")
    parser.add_argument("--normal", default="z", choices=sorted(NORMALS))
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--scene", default="all", choices=["all", "mesh", "fields"])
    parser.add_argument("--zoom", type=float, default=None,
                        help="magnify about the centre, e.g. 4 for a close-up")
    parser.add_argument("--bounds", nargs=6, type=float, default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="frame on this region instead of the whole domain")
    parser.add_argument("--contours", type=int, default=0, metavar="N",
                        help="N iso-contour lines over each field slice, at levels "
                             "taken from that field's own range")
    parser.add_argument("--diff", type=Path, default=None, metavar="OTHER_CASE",
                        help="another run of the same case: each field comes out as two "
                             "panels on one colour scale plus their difference. Refused "
                             "if the two are not over the same domain.")
    args = parser.parse_args()
    bounds = tuple(args.bounds) if args.bounds else None

    block, chosen, times = open_case(args.case, args.time)
    mesh = internal_mesh(block)
    if mesh is None:
        raise SystemExit("no internalMesh in this case — has it been meshed?")

    out_dir = args.out or args.case / "renders"
    print(f"time {chosen:g}  (available: {', '.join(f'{t:g}' for t in times) or 'none'})")
    print(f"cells {mesh.n_cells:,}  points {mesh.n_points:,}")
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.diff is not None:
        other, other_time, _ = open_case(args.diff, args.time)
        second = internal_mesh(other)
        if second is None:
            raise SystemExit(f"no internalMesh in {args.diff} -- has it been meshed?")
        print(f"against {args.diff} at time {other_time:g} "
              f"({second.n_cells:,} cells, {second.n_points:,} points)")
        for field in args.fields:
            record, paths = render_difference(
                mesh, second, field, args.normal, out_dir,
                (args.case.name, args.diff.name), zoom=args.zoom, bounds=bounds,
            )
            if record is None:
                print(f"  (no field named {field} in both runs)")
                continue
            low, high = record["clim"]
            print(f"  {field}: {record['correspondence']}, "
                  f"max |difference| {record['max_abs_difference']:.4g}"
                  + ("  (the two runs are identical here)" if record["identical"] else "")
                  + f", both panels on clim {low:.4g} .. {high:.4g} "
                  f"and the {args.normal}-normal camera")
            written += paths
        for path in written:
            print(path)
        return

    suffix = "_closeup" if (bounds or args.zoom) else ""
    if args.scene in ("all", "mesh"):
        written.append(render_mesh(mesh, args.normal,
                                   out_dir / f"mesh_{args.normal}{suffix}.png",
                                   zoom=args.zoom, bounds=bounds))
    if args.scene in ("all", "fields"):
        for field in args.fields:
            path = render_field(
                mesh, field, args.normal,
                out_dir / f"{field}_{args.normal}_{chosen:g}{suffix}.png",
                zoom=args.zoom, bounds=bounds, contours=args.contours,
            )
            if path:
                written.append(path)
            else:
                print(f"  (no field named {field} at this time)")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
