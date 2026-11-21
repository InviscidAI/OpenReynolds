#!/usr/bin/env python3
"""Heat transfer read back off a finished case: wall flux, Nusselt, hot spots.

Nothing in the toolbox computed any of this. `ladder.py` names the answers a
conjugate case is supposed to be checked against -- Fourier's law through a slab,
the Graetz eigenvalues, Dittus-Boelter -- and its `classify` reads a conjugate case
off `constant/regionProperties`, so the ladder could tell a study what to measure
and there was no instrument to measure it with. A heated run therefore
ended with a temperature picture and no number: no heat rate, no coefficient, and
no answer to "where is the hottest part" beyond the maximum of a colour bar.

what it reads, and why that is three things rather than one

1. `postProcessing/wallHeatFlux/<time>/wallHeatFlux.dat`, which is OpenFOAM's own
   `wallHeatFlux` function object reporting min, max and the integral per patch per
   write. Already computed, so it is the cheapest and the most direct.
2. a written `wallHeatFlux` field, whose `boundaryField` carries one value per face.
   The integral is then formed here as `sum(q_i * |A_i|)` with the face areas out of
   `constant/polyMesh`, which is also what makes the area-weighted average a real
   area average: the plain mean of the face values weights a 1 mm2 face like a
   1 m2 one and on a graded wall the two differ by tens of percent.
3. a `T` field, for the wall and reference temperatures, the hot spots, and whether
   the temperature was still moving at the last write.

The sign is OpenFOAM's: a positive `wallHeatFlux` is heat leaving the domain through
that patch. It is reported as it was written rather than flipped to whichever
convention a reader might prefer, since a silently flipped sign turns a cooler into
a heater.

the reference temperature is the whole answer

Nu = h L / k with h = q_wall / (T_wall - T_ref), and the three of q, T_wall and T_ref
are read off the case while L and k come from it or from a flag. None is defaulted:
`benchmarks/moving_mesh/grade.py`'s rule is that a missing constant stops and names
the flag that would supply it, because a Nusselt number built on an assumed
conductivity looks measured and is not.

T_ref is the part that is usually left unsaid, and it changes the answer. Three
conventions are offered and the one used is printed beside every number:

- `freestream` -- a temperature stated outright (`--t-ref`), which is what an
  external flow's T_inf is.
- `inlet` -- the area-weighted mean of T over a patch.
- `bulk` -- the mixed-mean (flux-weighted) temperature over a patch,
  `sum(phi T) / sum(phi)` with `phi = U . A` per face, which is `probe.py`'s
  `--weighted flux` arithmetic. This is the one an internal flow's Nu is defined
  against, and it is the default where a patch is named: in a duct the fluid near
  the wall is the hot fluid, so the area average over a section sits above the
  mixed mean and a Nu taken on it comes out low.

what hot spots are, as opposed to a maximum

A maximum is a number; a hot spot is a place. `locate.py` already reports an
extremum's cell, coordinates, owning patch and distance to the boundary for a
diverging field, and its readers are reused here rather than written again, so the
hottest cell is placed the same way the worst pressure cell is. On top of the peak:
how large the region above a threshold is, in cells, as a fraction of the mesh, and
as the box its cell centres span -- a 400 K peak in four cells and the same peak
over a tenth of the solid are different objects.

a temperature can be converged by a control that was not watching it

A passive temperature added with a `scalarTransport` function object is not a field
the SIMPLE loop solves, so naming `T` in `residualControl` does nothing at all: the
control never sees a function object's field. A heated run here stopped at iteration
87 on p and U with T's residual still at 6e-3 and the quantity being optimised still
rising 3% an iteration. So the control is read and the two last written T fields are
differenced, and both are reported next to the Nusselt number they underwrite.

Conjugate cases are per region: `constant/regionProperties` names them, their meshes
sit in `constant/<region>/polyMesh` and their fields in `<time>/<region>/`. A case
with one region says so rather than saying nothing, since "one region" and "the
regions could not be read" are different events.

Findings are in preflight's measured/means/repair register. This edits nothing, runs
no solver, and exits 0 whatever it finds.

    python3 heat.py /work/case
    python3 heat.py /work/case --length 0.05 --conductivity 0.6 --ref-patch inlet
    python3 heat.py /work/case --t-ref 300 --reynolds 50000 --prandtl 0.71
    python3 heat.py /work/case --json heat.json --png heat.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_report  # noqa: E402  (sibling scripts, not a package)
import locate  # noqa: E402
import preflight  # noqa: E402
import results  # noqa: E402
import wall  # noqa: E402

Finding = preflight.Finding


class HeatError(Exception):
    """A measurement that could not be made. Not the same as a measurement of zero."""


def _need(value: Any, what: str, where: str, flag: str) -> Any:
    """`benchmarks/moving_mesh/grade.py:125`'s rule, which `wall.py` keeps too: a
    missing constant stops and names the flag that would supply it. A default here is
    a Nusselt number with the right format and the wrong physics."""
    if value is None:
        raise HeatError(f"could not read {what} from {where}; pass {flag}")
    return value


# There was a `_NUMBER` regex here, this file's own copy of the one #138 was filed
# about, and it is gone: every `boundaryField` value heat.py reads now comes back from
# `wall.boundary_scalars`, which knows whether the file is ascii or binary.


# ------------------------------------------------------------------- regions --


class Region(NamedTuple):
    """One mesh region of a case, and where its three kinds of file live.

    `sub` is the subdirectory inside a time directory, which is the region name on a
    multi-region case and empty on an ordinary one -- so a single-region case runs
    through exactly the same code path rather than a special one.
    """

    name: str
    poly: Path
    case: Path
    sub: str


def region_names(text: str) -> list[str]:
    """The mesh regions `constant/regionProperties` names.

    OpenFOAM groups them by the physics each is solved with --
    `regions (fluid (water) solid (metal));` -- and it is the INNER names that are the
    mesh regions. Reading the outer word as a region is how a case comes back with a
    region called `fluid` that has no polyMesh, so the innermost parentheses are what
    is read: the pattern matches no nested group, which makes `(water)` a match and
    the whole list not one.
    """
    body = preflight.strip_comments(text)
    match = re.search(r"\bregions\b", body)
    if not match:
        return []
    names: list[str] = []
    for group in re.finditer(r"\(([^()]*)\)", body[match.end():]):
        for token in group.group(1).split():
            if token not in names:
                names.append(token)
    return names


def regions_of(case: Path) -> tuple[list[Region], list[str]]:
    """The regions to report on, and the names `regionProperties` declared.

    Two layouts are read because both exist: `splitMeshRegions` writes the meshes
    under `constant/<region>/polyMesh` with the fields under `<time>/<region>/`, and
    a case assembled by hand sometimes gives each region its own case directory.
    A declared region with no polyMesh in either place is left out of the list and
    stays in the declared names, which is what makes the gap reportable.
    """
    case = Path(case)
    declared = region_names(preflight.read_text(case / "constant" / "regionProperties"))
    found: list[Region] = []
    for name in declared:
        if (case / "constant" / name / "polyMesh").is_dir():
            found.append(Region(name, case / "constant" / name / "polyMesh", case, name))
        elif (case / name / "constant" / "polyMesh").is_dir():
            found.append(Region(name, case / name / "constant" / "polyMesh", case / name, ""))
    if found:
        return found, declared
    return [Region("", case / "constant" / "polyMesh", case, "")], declared


def mesh_of(region: Region) -> dict[str, Any] | str:
    """The region's mesh, or a sentence saying why not. `locate.read_mesh` is the
    reader because it takes a polyMesh directory outright (so a region's mesh needs no
    special case) and reads binary as well as ascii, and what it returns is the dict
    `layer_report`'s geometry already works on."""
    return locate.read_mesh(region.poly)


def field_file(region: Region, name: str, requested: str | None) -> tuple[Path, float | None] | None:
    """The latest written `name` in this region, or the one `--time` asked for."""
    times = locate.time_dirs(region.case)
    if requested is not None:
        times = [
            (value, folder) for value, folder in times
            if folder.name == requested
            or (locate.is_time_name(requested) and value == float(requested))
        ]
    for value, folder in reversed(times):
        for candidate in (folder / region.sub / name, folder / region.sub / f"{name}.gz"):
            if candidate.is_file():
                return candidate, value
    return None


# -------------------------------------------------------- the wall heat flux --


_FLUX_ALIASES = {
    "time": ("time",),
    "patch": ("patch", "patchname"),
    "min": ("min", "minimum"),
    "max": ("max", "maximum"),
    "integral": ("integral", "total", "sum"),
    "average": ("average", "mean", "avg"),
}
_REQUIRED = ("time", "patch", "min", "max")
_POSITIONAL = {"time": 0, "patch": 1, "min": 2, "max": 3, "integral": 4}
"""`# Time patch min max integral` is what the function object writes, and the
fallback when a header cannot be read. The names are looked up first all the same:
assuming positions is how an integral in watts gets reported as a maximum flux."""


def dat_columns(text: str) -> dict[str, int]:
    """Which column is which, read off the `# Time patch min max integral` header.

    The header is split by `results.split_header_names`, so a `.dat` header is read
    one way in the toolbox rather than several. `results.parse_dat` is the function
    that would normally read the whole file and it cannot read this one: its row
    reader returns nothing for any line carrying a non-numeric token -- a guard
    against a job killed mid-write shifting every column -- and every row here
    carries a patch NAME in its second column. `wall.py` met the same wall on
    `yPlus.dat` and this is its two-step shape with the heat flux's own columns.
    """
    found: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            if found:
                break
            continue
        names = [name.lower() for name in results.split_header_names(stripped)]
        if "patch" not in names:
            continue
        found = {}
        for key, aliases in _FLUX_ALIASES.items():
            for index, name in enumerate(names):
                if name in aliases:
                    found[key] = index
                    break
    return found if set(found) >= set(_REQUIRED) else dict(_POSITIONAL)


def dat_rows(path: Path) -> list[dict[str, Any]]:
    """One row per patch per write of a `wallHeatFlux.dat`."""
    text = preflight.read_text(path)
    columns = dat_columns(text)
    width = max(columns.values()) + 1
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < width:
            continue
        try:
            row = {
                key: float(tokens[index])
                for key, index in columns.items()
                if key != "patch"
            }
        except ValueError:
            continue
        row["patch"] = tokens[columns["patch"]]
        rows.append(row)
    return rows


def dat_files(case: Path, region: str) -> dict[str, list[Path]]:
    """Every `wallHeatFlux*.dat` under `postProcessing`, grouped into one series each.

    A restart writes a second time directory beside the first, and a restart into a
    directory that already holds the file writes `wallHeatFlux_0.2.dat` beside it.
    Both are one history, as `results.py` has it. On a multi-region case OpenFOAM
    writes under `postProcessing/<region>/`, so the paths are narrowed to the region
    wherever any of them names it; where none does, this region has no `.dat` source
    rather than the wrong one.
    """
    base = Path(case) / "postProcessing"
    grouped: dict[str, list[Path]] = {}
    for dat in sorted(base.rglob("*.dat")) if base.is_dir() else []:
        if not dat.is_file():
            continue
        name, _ = results.series_name(dat)
        if not name.lower().startswith("wallheatflux"):
            continue
        parts = dat.relative_to(base).parts
        if region and region not in parts:
            continue
        function = "/".join(parts[:-2]) if len(parts) >= 3 else dat.parent.name
        grouped.setdefault(f"{function}/{name}", []).append(dat)
    for paths in grouped.values():
        paths.sort(key=results.written_order)
    return grouped


def patch_faces(mesh: dict[str, Any], patch: str) -> np.ndarray:
    entry = mesh["boundary"].get(patch)
    if entry is None:
        raise HeatError(f"no patch '{patch}' in this mesh")
    start, count = entry["startFace"], entry["nFaces"]
    if count and start + count > mesh["offsets"].size - 1:
        raise HeatError(f"patch '{patch}' names faces the face list does not have")
    return np.arange(start, start + count)


def patch_area_vectors(mesh: dict[str, Any], patch: str) -> np.ndarray:
    """One outward area vector per face of the patch, by OpenFOAM's own fan
    decomposition -- `layer_report.face_geometry`, not a second copy of it."""
    _, area = layer_report.face_geometry(mesh, patch_faces(mesh, patch))
    return area


def flux_row(patch: str, values: np.ndarray, areas: np.ndarray) -> dict[str, Any]:
    """One patch's heat rate, area-weighted mean flux and range, off per-face values.

    A `uniform` entry arrives as a single value and multiplies the patch's total area,
    which is both the right arithmetic and the honest record of what the file said.
    """
    values = np.asarray(values, dtype=float)
    total_area = float(np.sum(areas))
    if values.size == 1:
        total = float(values[0]) * total_area
    elif values.size == areas.size:
        total = float(np.sum(values * areas))
    else:
        raise HeatError(
            f"patch '{patch}': {values.size} flux values against {areas.size} faces"
        )
    return {
        "patch": patch,
        "faces": int(areas.size),
        "area": total_area,
        "heat_rate": total,
        "mean_flux": total / total_area if total_area else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def from_field(region: Region, mesh: dict[str, Any] | str,
               requested: str | None) -> dict[str, Any] | None:
    """A written `wallHeatFlux` field: the distribution, integrated against the mesh."""
    found = field_file(region, "wallHeatFlux", requested)
    if found is None:
        return None
    path, at = found
    if isinstance(mesh, str):
        raise HeatError(
            f"{path.parent.name}/wallHeatFlux is written and the mesh is not readable "
            f"({mesh}), so there are no face areas to integrate it over"
        )
    values = patch_values(path)
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for name, array in sorted(values.items()):
        if array is None or array.size == 0:
            continue
        if name not in mesh["boundary"]:
            problems.append(f"the field has a patch '{name}' the mesh does not")
            continue
        try:
            areas = np.linalg.norm(patch_area_vectors(mesh, name), axis=1)
            rows.append(flux_row(name, array, areas))
        except HeatError as error:
            problems.append(str(error))
    if not rows:
        if problems:
            raise HeatError("; ".join(problems))
        return None
    return {
        "source": "written wallHeatFlux field",
        "detail": f"{path.parent.name}/{path.name}, areas from constant/polyMesh",
        "time": at,
        "patches": rows,
        "problems": problems,
    }


def from_dat(region: Region, mesh: dict[str, Any] | str,
             requested: str | None) -> list[dict[str, Any]]:
    """One source record per `wallHeatFlux` series the case logged, stitched.

    The function object writes the integral, so the heat rate is read rather than
    formed. The area-weighted mean flux is that integral over the patch area, which
    needs the mesh: where the mesh cannot be read the `average` column is used if the
    header has one, and otherwise the mean is reported as unavailable rather than as
    the mean of a min and a max.
    """
    sources: list[dict[str, Any]] = []
    for key, paths in sorted(dat_files(region.case, region.sub).items()):
        rows: list[dict[str, Any]] = []
        for path in paths:
            start = results.restart_point(path)
            if start is not None:
                edge = start - results.time_tolerance(start)
                rows = [row for row in rows if row["time"] < edge]
            rows.extend(dat_rows(path))
        if not rows:
            continue
        times = sorted({row["time"] for row in rows})
        if requested is not None:
            wanted = float(requested)
            times = [t for t in times if abs(t - wanted) <= results.time_tolerance(wanted)]
            if not times:
                continue
        at = times[-1]
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            if abs(row["time"] - at) <= results.time_tolerance(at):
                latest[row["patch"]] = row
        patches: list[dict[str, Any]] = []
        for name, row in sorted(latest.items()):
            area = None
            if not isinstance(mesh, str) and name in mesh["boundary"]:
                area = float(np.sum(
                    np.linalg.norm(patch_area_vectors(mesh, name), axis=1)
                ))
            total = row.get("integral")
            mean = row.get("average")
            if mean is None and total is not None and area:
                mean = total / area
            if total is None and mean is not None and area:
                total = mean * area
            patches.append({
                "patch": name,
                "faces": (
                    int(mesh["boundary"][name]["nFaces"])
                    if not isinstance(mesh, str) and name in mesh["boundary"] else None
                ),
                "area": area,
                "heat_rate": total,
                "mean_flux": mean,
                "min": row["min"],
                "max": row["max"],
            })
        sources.append({
            "source": "wallHeatFlux function object",
            "detail": ", ".join(results.post_relative(path) for path in paths),
            "series": key,
            "time": at,
            "times": sorted({row["time"] for row in rows}),
            "patches": patches,
            "problems": [],
        })
    return sources


# ------------------------------------------------------------ temperature --


def patch_values(path: Path) -> dict[str, np.ndarray | None]:
    """A scalar field's `boundaryField`, per patch, ascii or binary.

    `wall.read_field` decodes a binary payload or refuses the file by name; handing
    `preflight.read_text` straight to a value parser is #138, and doing it here is
    #163 -- the same defect, in the file that feeds every Nusselt number, for two
    months after `wall.py` was fixed. There is one decoder and both tools call it.
    """
    return wall.boundary_values(*wall.read_field(path))


def boundary_vectors(path_or_text: Path | str) -> dict[str, np.ndarray | None]:
    """`{patch: (n, 3) values}` out of a vector field's `boundaryField`.

    `wall.boundary_vectors` is the reader -- the same `boundaryField` walk the scalars
    go through, with the components kept, because the sign of `U . A` is what makes a
    flux a flux and a magnitude cannot tell an inlet from an outlet. A `Path` is read
    through `wall.read_field`, so a binary `U` is decoded rather than scraped; a `str`
    is text already read, which is what the tests hand it.
    """
    if isinstance(path_or_text, str):
        return wall.boundary_vectors(path_or_text)
    return wall.boundary_vectors(*wall.read_field(path_or_text))


def patch_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """A weighted mean, with a `uniform` entry broadcast over the patch.

    One value is a `uniform` entry and is broadcast. That is only safe because
    `wall.boundary_scalars` refuses a `nonuniform List<...> N(` whose body does not
    hold N items: the way a wrong answer got through this exemption silently was a
    binary payload from which a number regex scraped exactly one number (#163), and
    that read no longer reaches here at all.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return float(values[0])
    if values.size != weights.size:
        raise HeatError(f"{values.size} values against {weights.size} faces")
    total = float(np.sum(weights))
    if total == 0.0:
        raise HeatError("the weights over this patch sum to zero")
    return float(np.sum(values * weights) / total)


CONVENTIONS = ("bulk", "inlet", "freestream")
"""The reference temperatures a Nusselt number can be taken against. `bulk` is the
mixed mean an internal flow's Nu is defined on, `inlet` the area-weighted mean over
the same patch, `freestream` a value stated outright. Which was used is printed with
every number, because the three give three different Nusselt numbers off one case."""


class Reference(NamedTuple):
    convention: str
    value: float
    patch: str
    detail: str


def reference_temperature(region: Region, mesh: dict[str, Any] | str,
                          temperatures: dict[str, np.ndarray | None] | None,
                          *, stated: float | None, patch: str | None,
                          convention: str | None, time: str | None) -> Reference:
    """T_ref and the convention it is, refusing rather than choosing silently.

    A stated value is a free-stream temperature and is taken as one. Otherwise a
    patch is needed, and over that patch the default is the mixed mean: in a duct the
    fluid at the wall is the hot fluid, so the area average sits above the mixed mean
    and a Nu taken on it comes out low.
    """
    if stated is not None and (convention is None or convention == "freestream"):
        return Reference("freestream", float(stated), "", "--t-ref")
    if convention == "freestream":
        raise HeatError("--t-ref-from freestream needs the value; pass --t-ref")
    if patch is None:
        raise HeatError(
            "no reference temperature: pass --t-ref for a free-stream value, or "
            "--ref-patch to take one off a patch of this case"
        )
    if isinstance(mesh, str):
        raise HeatError(f"a reference temperature over '{patch}' needs the mesh ({mesh})")
    if temperatures is None:
        raise HeatError(
            f"no T field written in this region, so '{patch}' has no temperature on "
            "it; pass --t-ref"
        )
    values = temperatures.get(patch)
    if values is None or values.size == 0:
        have = ", ".join(sorted(name for name, v in temperatures.items() if v is not None))
        raise HeatError(
            f"T has no value on patch '{patch}' (it has one on: {have or 'nothing'}); "
            "pass --t-ref"
        )
    area_vectors = patch_area_vectors(mesh, patch)
    areas = np.linalg.norm(area_vectors, axis=1)
    wanted = convention or "bulk"
    if wanted == "inlet":
        return Reference(
            "inlet", patch_mean(values, areas), patch,
            f"area-weighted mean of T over '{patch}' ({areas.size} faces)",
        )
    found = field_file(region, "U", time)
    if found is None:
        raise HeatError(
            "a bulk mean is flux-weighted and this case wrote no U; pass "
            "--t-ref-from inlet for the area-weighted mean over the patch, or --t-ref"
        )
    path, _ = found
    velocities = boundary_vectors(path).get(patch)
    if velocities is None or velocities.size == 0:
        raise HeatError(
            f"U has no value on patch '{patch}', so there is no flux to weight by; "
            "pass --t-ref-from inlet or --t-ref"
        )
    if velocities.shape[0] not in (1, areas.size):
        raise HeatError(
            f"patch '{patch}': {velocities.shape[0]} velocities against "
            f"{areas.size} faces"
        )
    # `phi = U . A` per face, with a `uniform` U broadcast over the patch's own area
    # vectors rather than over one summed normal: on a curved patch the normals do
    # not all point the same way, and summing them first turns an inflow into a
    # smaller inflow.
    phi = np.sum(velocities * area_vectors, axis=1)
    if abs(float(np.sum(phi))) == 0.0:
        raise HeatError(
            f"the flux through '{patch}' sums to zero, so a mixed mean would divide "
            "by it; pass --t-ref-from inlet or --t-ref"
        )
    return Reference(
        "bulk", patch_mean(values, phi), patch,
        f"flux-weighted (mixed mean) over '{patch}': sum(phi T) / sum(phi) with "
        f"phi = U . A per face, net flux {float(np.sum(phi)):.4g} m3/s",
    )


# ------------------------------------------------------------ conductivity --


def _scalar(text: str, name: str) -> float | None:
    """A `key value;` entry, with the dimension set OpenFOAM may put in front of the
    number skipped over. `preflight.parse_nu` is the same shape for nu alone."""
    match = re.search(
        r"(?<![\w.])" + re.escape(name)
        + r"\s+(?:" + re.escape(name) + r"\s+)?(?:\[[^\]]*\]\s*)?"
        r"([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*;",
        preflight.strip_comments(text),
    )
    return preflight.as_float(match.group(1)) if match else None


CONSTANT_FILES = (
    "thermophysicalProperties", "physicalProperties", "transportProperties",
    "momentumTransport", "radiationProperties",
)
"""Where a material property might be written, newest spelling first."""


def conductivity(region: Region, given: float | None) -> tuple[float | None, str]:
    """Thermal conductivity in W/m K, and where it came from.

    Three routes, because a case states it in whichever of three ways its solver
    wanted. `kappa` outright is a solid's entry; `rho * Cp * nu / Pr` is the same
    number for a fluid whose dictionary gives those four, and it is spelled out in
    the report so the arithmetic can be checked rather than trusted. A case stating
    none of it is refused by name: a conductivity guessed at water's or air's gives a
    Nusselt number that is wrong by the ratio of the two.
    """
    if given is not None:
        return given, "--conductivity"
    # The region's own dictionary directory first (`constant/<region>/` on a split
    # mesh), then the case's, since a conjugate case puts the fluid's transport
    # properties beside the fluid's mesh and only the shared ones at the top.
    roots = [region.poly.parent, region.case / "constant"]
    seen: set[Path] = set()
    for root in roots:
        for name in CONSTANT_FILES:
            path = root / name
            if path in seen:
                continue
            seen.add(path)
            text = preflight.read_text(path)
            if not text:
                continue
            kappa = _scalar(text, "kappa")
            if kappa is not None:
                return kappa, f"{path.parent.name}/{name}: kappa"
            rho = _scalar(text, "rho")
            cp = _scalar(text, "Cp")
            nu = preflight.parse_nu(text)
            pr = _scalar(text, "Pr") or _scalar(text, "Prt")
            if None not in (rho, cp, nu, pr) and pr:
                return (
                    rho * cp * nu / pr,
                    f"{path.parent.name}/{name}: rho {rho:g} * Cp {cp:g} * nu {nu:g} "
                    f"/ Pr {pr:g}",
                )
    return None, ""


# ----------------------------------------------------------------- Nusselt --


class Nusselt(NamedTuple):
    patch: str
    nu: float
    h: float
    flux: float
    t_wall: float
    reference: Reference
    length: float
    conductivity: float


def nusselt(patch: str, flux: float, t_wall: float, reference: Reference,
            length: float, k: float) -> Nusselt:
    """Nu = h L / k with h = q / (T_wall - T_ref). Every term is an argument: there
    is no reading inside here, so the arithmetic is checkable on its own."""
    delta = t_wall - reference.value
    if delta == 0.0:
        raise HeatError(
            f"patch '{patch}': T_wall and the {reference.convention} reference are "
            f"both {t_wall:g}, so h = q / 0 and there is no Nusselt number"
        )
    if not length > 0:
        raise HeatError(f"the length scale is {length:g}; pass --length")
    if not k > 0:
        raise HeatError(f"the conductivity is {k:g}; pass --conductivity")
    h = flux / delta
    return Nusselt(patch, h * length / k, h, flux, t_wall, reference, length, k)


LAMINAR_LIMIT = 2300.0
"""Where the ladder's internal-flow catalogue stops offering Graetz and starts
offering Dittus-Boelter. The transition is a band and not a line; which side a case
was compared on is printed with the comparison."""


DITTUS_BOELTER_RE_MIN = 1e4
DITTUS_BOELTER_PR_MIN = 0.7
DITTUS_BOELTER_PR_MAX = 160.0
"""Where Dittus-Boelter is evaluated, and outside which it is refused rather than
evaluated anyway.

The upper Pr bound and the Re bound are the ones `ladder.py`'s catalogue and this
module's own output string have always stated. The lower Pr bound is the one number
here that could not be read from a source: **0.6 and 0.7 both circulate**, the original
paper is paywalled (F. W. Dittus and L. M. K. Boelter, University of California
Publications in Engineering 2 (1930) 443, reprinted Int. Commun. Heat Mass Transfer 12
(1985) 3-22), and the open-access alternative -- Lienhard & Lienhard, *A Heat Transfer
Textbook*, 6th ed., section 7.3, free at ahtt.mit.edu -- is a single 32 MB PDF that was
not retrievable here. So the STRICTER of the two is what is checked: between 0.6 and 0.7
the range itself is unverified, and a verdict there would rest on which textbook was to
hand rather than on the correlation. `benchmarks/README.md` records what quoting a
reference from a secondary source has cost this repository twice, and `benchmarks/sph/`
does the same thing with Martin & Moyce (1952): report the quantity, name the
alternative, withhold the verdict.

Separately, and not checkable from a case: Dittus-Boelter is a fit for fully developed
flow in a smooth circular pipe. The tube bank in `20260830-000854-31c3` is not one, and
nothing here can tell. That is said in the output rather than guarded against (#190)."""


class Correlation(NamedTuple):
    name: str
    nu: float | None
    band: float
    source: str
    outside: tuple[str, ...] = ()


def dittus_boelter_outside(reynolds: float, prandtl: float) -> tuple[str, ...]:
    """Which of the correlation's own bounds this Re and Pr are outside of.

    Checked before the formula is evaluated, because `0.023 Re^0.8 Pr^n` returns a
    plausible number for any positive inputs: at Re 3000 and Pr 0.7 it gives 12.0632,
    and that was reported as the CFD being "370.3% away" in the same sentence that
    stated the correlation holds over Re > 1e4 (#190). A percentage against a
    correlation outside its range is a measurement of the range violation, not of the
    solve.
    """
    out = []
    if reynolds < DITTUS_BOELTER_RE_MIN:
        out.append(f"Re {reynolds:g} is below the correlation's "
                   f"Re > {DITTUS_BOELTER_RE_MIN:g}")
    if prandtl < DITTUS_BOELTER_PR_MIN:
        out.append(f"Pr {prandtl:g} is below the correlation's "
                   f"Pr > {DITTUS_BOELTER_PR_MIN:g} (0.6 is also in circulation as that "
                   "bound and could not be verified against a source, so the stricter "
                   "one is checked)")
    if prandtl > DITTUS_BOELTER_PR_MAX:
        out.append(f"Pr {prandtl:g} is above the correlation's "
                   f"Pr < {DITTUS_BOELTER_PR_MAX:g}")
    return tuple(out)


def correlation_for(reynolds: float, prandtl: float, heating: bool) -> Correlation:
    """The published Nusselt number for a fully developed pipe flow at this Re and Pr.

    Both come out of `ladder.py`'s conjugate catalogue, with the accuracy each one
    states for itself rather than a tolerance invented here: the Graetz eigenvalues
    are exact for the problem they solve (constant wall temperature, fully developed
    laminar flow), and Dittus-Boelter is a fit to measured data whose own scatter is
    20-25%. A Nusselt number inside 25% of Dittus-Boelter has agreed with it; one
    inside 25% of Graetz has not.

    Outside Dittus-Boelter's own range there is no published number to compare with,
    so none is returned: `nu` comes back None and `outside` names every bound that was
    broken. The transitional band between `LAMINAR_LIMIT` and `Re = 1e4` is where
    neither entry in the catalogue applies, and saying so is the answer there.
    """
    if reynolds < LAMINAR_LIMIT:
        return Correlation(
            "Graetz, constant wall temperature", 3.66, 0.02,
            "the Graetz problem's first eigenvalue for fully developed laminar pipe "
            "flow at a constant wall temperature (4.36 is the constant-flux case); "
            "exact, so the 2% is the solver's and not the correlation's",
        )
    exponent = 0.4 if heating else 0.3
    outside = dittus_boelter_outside(reynolds, prandtl)
    source = (
        "Dittus-Boelter, Nu = 0.023 Re^0.8 Pr^n with n = 0.4 heating and 0.3 "
        f"cooling, a fit to measured heat-transfer data over Re > "
        f"{DITTUS_BOELTER_RE_MIN:g} and {DITTUS_BOELTER_PR_MIN:g} < Pr < "
        f"{DITTUS_BOELTER_PR_MAX:g} in a smooth circular pipe, whose own stated "
        "scatter is 20-25%"
    )
    return Correlation(
        f"Dittus-Boelter, n = {exponent:g} ({'heating' if heating else 'cooling'})",
        None if outside else 0.023 * reynolds ** 0.8 * prandtl ** exponent,
        0.25, source, outside,
    )


# --------------------------------------------------------------- hot spots --


HOT_FRACTION = 0.95
"""Where the hot region starts, as a fraction of the field's own min-to-max span. A
fraction rather than a temperature, because 400 K is a hot spot in a water jacket
and ambient in a combustor; `--threshold` states one outright."""


def nearest_patch(mesh: dict[str, Any], point: np.ndarray) -> tuple[str, float]:
    """The patch whose nearest face centre is closest to the point, and how far.

    For an interior hot cell, which is the case `locate.cell_patches` returns nothing
    for: a peak in the middle of a solid still sits near something, and which
    something is the part of the answer a coordinate triple does not carry.
    """
    best, distance = "", math.inf
    for name in mesh["boundary"]:
        try:
            centres, _ = layer_report.face_geometry(mesh, patch_faces(mesh, name))
        except HeatError:
            continue
        if centres.size == 0:
            continue
        gap = float(np.linalg.norm(centres - point, axis=1).min())
        if gap < distance:
            best, distance = name, gap
    return best, distance


def hot_spots(region: Region, mesh: dict[str, Any] | str, requested: str | None,
              *, threshold: float | None = None,
              fraction: float = HOT_FRACTION) -> dict[str, Any] | None:
    """The hottest place in this region, and how big the region above a threshold is.

    The peak's cell, coordinates, owning or nearest patch and distance to the domain
    bounding box are `locate.py`'s measurements, made with `locate.py`'s readers, so
    the hottest cell here is placed exactly the way the worst pressure cell is there.
    """
    found = field_file(region, "T", requested)
    if found is None:
        return None
    path, at = found
    data = locate.read_internal_field(path)
    if data is None:
        raise HeatError(f"{path.parent.name}/{path.name}: internalField unreadable")
    if data["kind"] == "uniform":
        return {
            "time": at, "detail": f"{path.parent.name}/{path.name}",
            "uniform": float(np.atleast_1d(data["values"])[0]),
        }
    values = np.asarray(data["values"], dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise HeatError(f"{path.parent.name}/{path.name}: every T value is non-finite")
    low, high = float(finite.min()), float(finite.max())
    edge = threshold if threshold is not None else low + fraction * (high - low)
    hot = np.flatnonzero(np.isfinite(values) & (values >= edge))
    peak = int(np.nanargmax(np.where(np.isfinite(values), values, -np.inf)))
    record: dict[str, Any] = {
        "time": at,
        "detail": f"{path.parent.name}/{path.name}",
        "min": low,
        "max": high,
        "mean": float(finite.mean()),
        "threshold": float(edge),
        "threshold_from": (
            "--threshold" if threshold is not None
            else f"{fraction:g} of the min-to-max span"
        ),
        "cells": int(values.size),
        "hot_cells": int(hot.size),
        "hot_fraction": round(100.0 * hot.size / max(values.size, 1), 3),
        "peak": {"value": float(values[peak]), "cell": peak},
    }
    if isinstance(mesh, str):
        record["peak"]["mesh"] = mesh
        return record
    if peak >= mesh["n_cells"]:
        record["peak"]["mesh"] = (
            f"cell {peak} is outside a mesh of {mesh['n_cells']} cells, so the field "
            "and the mesh are not of the same region"
        )
        return record
    record["cells_in_mesh"] = int(mesh["n_cells"])
    if values.size != mesh["n_cells"]:
        # The coordinates below are still this cell's, but a field shorter or longer
        # than the mesh means the two are not of the same region -- or that only one
        # processor's slice was read -- and a hot spot located in the wrong mesh is
        # worse than none.
        record["mismatch"] = (
            f"the field has {values.size} values and the mesh {mesh['n_cells']} cells"
        )
    centre = layer_report.cell_centres(mesh, np.array([peak]))[0]
    record["peak"]["centre"] = [round(float(part), 6) for part in centre]
    on = locate.cell_patches(mesh, peak)
    record["peak"]["patches"] = on
    if on:
        record["peak"]["nearest_patch"] = on[0]
        record["peak"]["patch_distance"] = 0.0
    else:
        name, gap = nearest_patch(mesh, centre)
        record["peak"]["nearest_patch"] = name
        record["peak"]["patch_distance"] = gap
    record["peak"].update(locate.bbox_facts(mesh, centre))
    if hot.size and hot.max() < mesh["n_cells"]:
        centres = layer_report.cell_centres(mesh, hot)
        lo, hi = centres.min(axis=0), centres.max(axis=0)
        record["hot_extent"] = {
            "min": [round(float(part), 6) for part in lo],
            "max": [round(float(part), 6) for part in hi],
            "diagonal": float(np.linalg.norm(hi - lo)),
        }
    return record


# ------------------------------- was anything watching the temperature? --


def system_text(region: Region, name: str) -> str:
    """One `system/` dictionary, the region's own copy before the case's. A split
    conjugate case gives each region a `system/<region>/fvSolution` and leaves the
    shared `controlDict` at the top, so both places are read in that order."""
    if region.sub:
        text = preflight.read_text(region.case / "system" / region.sub / name)
        if text:
            return text
    return preflight.read_text(region.case / "system" / name)


def temperature_control(region: Region, requested: str | None) -> dict[str, Any]:
    """Whether the run's own stopping criterion was looking at T, and whether T had
    stopped moving when it stopped.

    A passive temperature added with a `scalarTransport` function object is not a
    field the SIMPLE loop solves, so naming `T` in `residualControl` does nothing:
    that control never sees a function object's field. A heated run here stopped at
    iteration 87 on p and U with T's residual at 6e-3 and the quantity being
    optimised still rising 3% an iteration -- which is why this is measured beside
    the Nusselt number rather than assumed away.
    """
    text = system_text(region, "fvSolution")
    controlled: list[str] = []
    loop = ""
    for keyword in ("SIMPLE", "PIMPLE", "PISO"):
        body = preflight.block_body(text, keyword)
        if not body:
            continue
        inner = preflight.block_body(body, "residualControl")
        if inner:
            loop = keyword
            controlled = [name for name, _ in preflight.block_entries(inner) if name]
            controlled += list(preflight.entry_values(inner))
            break
    functions = system_text(region, "controlDict") + system_text(region, "functions")
    transported: list[str] = []
    for name, inner in preflight.block_entries(preflight.block_body(functions, "functions")):
        values = preflight.entry_values(inner)
        if values.get("type") == "scalarTransport":
            transported.append(values.get("field", "T"))
    record: dict[str, Any] = {
        "loop": loop,
        "residual_control": sorted(set(controlled)),
        "transported": transported,
        "change": None,
        "change_between": None,
    }
    times = locate.time_dirs(region.case)
    if requested is not None:
        times = [pair for pair in times if pair[1].name == requested]
    written = [
        (value, folder) for value, folder in times
        if (folder / region.sub / "T").is_file() or (folder / region.sub / "T.gz").is_file()
    ]
    if len(written) >= 2:
        pair = []
        for _, folder in written[-2:]:
            data = locate.read_internal_field(folder / region.sub / "T")
            if data is None or data["kind"] != "nonuniform":
                pair = []
                break
            pair.append(np.asarray(data["values"], dtype=float).reshape(-1))
        if len(pair) == 2 and pair[0].size == pair[1].size:
            gap = np.abs(pair[1] - pair[0])
            span = float(np.nanmax(pair[1]) - np.nanmin(pair[1]))
            record["change"] = float(np.nanmax(gap))
            record["change_span"] = span
            record["change_between"] = [folder.name for _, folder in written[-2:]]
    return record


# ---------------------------------------------------------------- measuring --


def measure(case: Path, *, time: str | None = None, region: str | None = None,
            length: float | None = None, k: float | None = None,
            t_wall: float | None = None, t_ref: float | None = None,
            ref_patch: str | None = None, ref_from: str | None = None,
            reynolds: float | None = None, prandtl: float | None = None,
            threshold: float | None = None,
            fraction: float = HOT_FRACTION) -> dict[str, Any]:
    """Everything this case says about its own heat transfer, as plain data."""
    case = Path(case)
    entries, declared = regions_of(case)
    if region is not None:
        entries = [entry for entry in entries if entry.name == region]
    found: dict[str, Any] = {
        "case": case.as_posix(),
        "time": time,
        "declared_regions": declared,
        "multi_region": bool(declared) and len(entries) > 0 and bool(entries[0].name),
        "regions": [],
        "problems": [],
    }
    if region is not None and not entries:
        found["problems"].append(
            f"no region '{region}' in this case; it names "
            f"{', '.join(declared) or 'none'}"
        )
        return found
    for entry in entries:
        found["regions"].append(measure_region(
            entry, time=time, length=length, k=k, t_wall=t_wall, t_ref=t_ref,
            ref_patch=ref_patch, ref_from=ref_from, reynolds=reynolds,
            prandtl=prandtl, threshold=threshold, fraction=fraction,
        ))
    return found


def measure_region(region: Region, *, time: str | None, length: float | None,
                   k: float | None, t_wall: float | None, t_ref: float | None,
                   ref_patch: str | None, ref_from: str | None,
                   reynolds: float | None, prandtl: float | None,
                   threshold: float | None, fraction: float) -> dict[str, Any]:
    """One region's flux sources, Nusselt numbers, hot spot and control reading."""
    mesh = mesh_of(region)
    out: dict[str, Any] = {
        "region": region.name,
        "mesh": mesh if isinstance(mesh, str) else f"{mesh['n_cells']} cells",
        "sources": [],
        "nusselt": [],
        "hot": None,
        "control": {},
        "conductivity": None,
        "conductivity_from": "",
        "problems": [],
    }
    read_k, k_from = conductivity(region, k)
    out["conductivity"], out["conductivity_from"] = read_k, k_from

    out["sources"].extend(from_dat(region, mesh, time))
    try:
        source = from_field(region, mesh, time)
    except HeatError as error:
        out["problems"].append(str(error))
        source = None
    if source is not None:
        out["sources"].append(source)
    for source in out["sources"]:
        out["problems"].extend(source.pop("problems", []))
    if not out["sources"]:
        out["problems"].append(
            "no wall heat flux anywhere: no postProcessing/**/wallHeatFlux*.dat and "
            "no written wallHeatFlux field"
        )

    temperatures: dict[str, np.ndarray | None] | None = None
    found = field_file(region, "T", time)
    if found is not None:
        temperatures = patch_values(found[0])

    try:
        out["hot"] = hot_spots(region, mesh, time, threshold=threshold, fraction=fraction)
    except HeatError as error:
        out["problems"].append(str(error))
    if out["hot"] is None and found is None:
        out["problems"].append("no T field written in this region, so no hot spot")

    try:
        out["control"] = temperature_control(region, time)
    except (OSError, ValueError) as error:
        out["problems"].append(f"the convergence control could not be read: {error}")

    lead = primary(out)
    if lead is None:
        return out
    try:
        reference = reference_temperature(
            region, mesh, temperatures, stated=t_ref, patch=ref_patch,
            convention=ref_from, time=time,
        )
    except HeatError as error:
        out["problems"].append(str(error))
        return out
    out["reference"] = {
        "convention": reference.convention, "value": reference.value,
        "patch": reference.patch, "detail": reference.detail,
    }
    for row in lead["patches"]:
        if row["mean_flux"] is None:
            out["problems"].append(
                f"patch '{row['patch']}': no area-weighted flux, so no Nusselt number"
            )
            continue
        try:
            wall_t = t_wall
            if wall_t is None:
                values = (temperatures or {}).get(row["patch"])
                if values is None or values.size == 0:
                    raise HeatError(
                        f"T has no value on patch '{row['patch']}'; pass --t-wall"
                    )
                if isinstance(mesh, str):
                    raise HeatError(
                        f"patch '{row['patch']}': T_wall is an area-weighted mean and "
                        f"the mesh is not readable ({mesh}); pass --t-wall"
                    )
                areas = np.linalg.norm(patch_area_vectors(mesh, row["patch"]), axis=1)
                wall_t = patch_mean(values, areas)
            answer = nusselt(
                row["patch"], row["mean_flux"], float(wall_t), reference,
                float(_need(length, "the length scale Nu is on", "the case", "--length")),
                float(_need(read_k, "the thermal conductivity",
                            "constant/thermophysicalProperties", "--conductivity")),
            )
        except HeatError as error:
            out["problems"].append(str(error))
            continue
        record: dict[str, Any] = {
            "patch": answer.patch, "Nu": answer.nu, "h": answer.h,
            "flux": answer.flux, "t_wall": answer.t_wall,
            "t_ref": answer.reference.value,
            "convention": answer.reference.convention,
            "length": answer.length, "k": answer.conductivity,
        }
        if reynolds is not None and prandtl is not None:
            published = correlation_for(
                float(reynolds), float(prandtl), answer.t_wall > answer.reference.value
            )
            relative = (
                abs(answer.nu - published.nu) / published.nu if published.nu else None
            )
            record["correlation"] = {
                "name": published.name, "Nu": published.nu, "band": published.band,
                "source": published.source, "relative": relative,
                # No verdict where the correlation was not evaluated: `inside` is None
                # rather than False, because the measured Nu did not disagree with
                # anything -- there is nothing here to disagree with (#190).
                "inside": (None if published.outside else
                           bool(relative is not None and relative <= published.band)),
                "outside_range": list(published.outside),
                "reynolds": float(reynolds), "prandtl": float(prandtl),
            }
        elif reynolds is not None or prandtl is not None:
            out["problems"].append(
                "a correlation needs both the Reynolds and the Prandtl number; pass "
                "--reynolds and --prandtl"
            )
        out["nusselt"].append(record)
    return out


def primary(found: dict[str, Any]) -> dict[str, Any] | None:
    """The source to lead with and to draw: the one the run computed itself, before
    one computed here from a field and a mesh."""
    for name in ("wallHeatFlux function object", "written wallHeatFlux field"):
        for source in found["sources"]:
            if source["source"] == name:
                return source
    return found["sources"][0] if found["sources"] else None


def total_heat_rate(source: dict[str, Any]) -> float | None:
    """The patches' heat rates added. None where any patch has none, since a total
    over some of a boundary is not a total."""
    rates = [row["heat_rate"] for row in source["patches"]]
    if not rates or any(rate is None for rate in rates):
        return None
    return float(sum(rates))


# ---------------------------------------------------------------- findings --


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    return f"{value:.6g}"


def region_findings(found: dict[str, Any]) -> list[Finding]:
    out: list[Finding] = []
    where = f"region '{found['region']}': " if found["region"] else ""

    for source in found["sources"]:
        measured = "; ".join(
            f"{row['patch']}: {_fmt(row['heat_rate'])} W over {_fmt(row['area'])} m2, "
            f"mean {_fmt(row['mean_flux'])} W/m2, {_fmt(row['min'])} to "
            f"{_fmt(row['max'])} W/m2"
            for row in source["patches"]
        )
        total = total_heat_rate(source)
        detail = source["detail"]
        if source.get("time") is not None:
            detail += f" at time {source['time']:g}"
        out.append(Finding(
            "wall-heat-flux", "ok",
            f"{where}{source['source']}, {detail} -- {measured}"
            + (f"; total {total:.6g} W over the patches reported" if total is not None else ""),
            "a positive flux is heat leaving the domain through that patch, which is "
            "OpenFOAM's sign and is reported as written; the mean is area-weighted, "
            "so a graded wall's small faces do not count like its large ones",
        ))

    for row in found["nusselt"]:
        measured = (
            f"{where}{row['patch']}: Nu {row['Nu']:.6g} = h {row['h']:.6g} W/m2K * L "
            f"{row['length']:g} m / k {row['k']:.6g} W/mK, with q {row['flux']:.6g} "
            f"W/m2, T_wall {row['t_wall']:.6g} and T_ref {row['t_ref']:.6g}"
        )
        meaning = (
            f"the reference temperature is the {row['convention']} one. Nu is not "
            "defined without saying which: the same wall against a free-stream, an "
            "inlet and a mixed-mean reference gives three different numbers"
        )
        out.append(Finding("nusselt", "ok", measured, meaning))
        published = row.get("correlation")
        if published is None:
            continue
        if published.get("outside_range"):
            # The range violation IS the finding. A percentage here would be a number
            # about the correlation's range and would read as one about the solve.
            out.append(Finding(
                "correlation", "skipped",
                f"{where}{row['patch']}: Nu {row['Nu']:.6g} measured, and not compared: "
                + "; ".join(published["outside_range"]),
                published["source"]
                + ". Evaluated outside its own range it still returns a plausible "
                  "number, so no Nusselt number is quoted from it here and no "
                  "percentage is taken against one",
            ))
            continue
        gap = (
            f"{100.0 * published['relative']:.1f}% away"
            if published["relative"] is not None else "no comparison"
        )
        out.append(Finding(
            "correlation", "ok" if published["inside"] else "warn",
            f"{where}{row['patch']}: Nu {row['Nu']:.6g} measured against "
            f"{published['Nu']:.6g} from {published['name']} at Re "
            f"{published['reynolds']:g}, Pr {published['prandtl']:g} -- {gap}, "
            f"against a band of {100.0 * published['band']:.0f}%",
            published["source"]
            + (". Inside its own stated accuracy" if published["inside"]
               else ". Outside its own stated accuracy, which is a disagreement with "
                    "the published number rather than with a tolerance set here"),
        ))

    hot = found.get("hot")
    if hot and "uniform" in hot:
        out.append(Finding(
            "hot-spot", "ok",
            f"{where}T is uniform {hot['uniform']:g} in {hot['detail']}",
            "a uniform field has no hot spot to place",
        ))
    elif hot:
        peak = hot["peak"]
        pieces = [
            f"{where}T {hot['min']:.6g} to {hot['max']:.6g} (mean {hot['mean']:.6g}) "
            f"in {hot['detail']}",
            f"peak {peak['value']:.6g} at cell {peak['cell']}",
        ]
        if "centre" in peak:
            pieces.append(
                "centre (" + ", ".join(f"{part:.6g}" for part in peak["centre"]) + ")"
            )
            on = peak.get("patches") or []
            if on:
                pieces.append(f"on {', '.join(on)}")
            else:
                pieces.append(
                    f"interior, nearest patch {peak['nearest_patch'] or 'none'} at "
                    f"{peak['patch_distance']:.4g}"
                )
            pieces.append(
                f"{peak['distance_to_bbox']:.4g} from the domain bounding box"
            )
        elif "mesh" in peak:
            pieces.append(f"no coordinates: {peak['mesh']}")
        pieces.append(
            f"{hot['hot_cells']} of {hot['cells']} cells above "
            f"{hot['threshold']:.6g} ({hot['hot_fraction']:g}%), threshold from "
            f"{hot['threshold_from']}"
        )
        if hot.get("mismatch"):
            pieces.append(hot["mismatch"])
        if "hot_extent" in hot:
            pieces.append(
                "that region spans ("
                + ", ".join(f"{part:.4g}" for part in hot["hot_extent"]["min"])
                + ") to ("
                + ", ".join(f"{part:.4g}" for part in hot["hot_extent"]["max"])
                + f"), diagonal {hot['hot_extent']['diagonal']:.4g}"
            )
        out.append(Finding(
            "hot-spot", "ok", "; ".join(pieces),
            "where the peak is and how much is near it are measured; whether either "
            "matters is not, and depends on what the material is rated for",
        ))

    control = found.get("control") or {}
    if control.get("transported") or control.get("residual_control"):
        watched = set(control.get("residual_control") or [])
        transported = control.get("transported") or []
        unwatched = [name for name in transported if name not in watched]
        measured = (
            f"{where}{control.get('loop') or 'no loop'} residualControl names "
            f"{', '.join(sorted(watched)) or 'nothing'}"
            + (f"; a scalarTransport function object solves "
               f"{', '.join(transported)}" if transported else "")
        )
        if control.get("change") is not None:
            measured += (
                f"; T moved by at most {control['change']:.4g} between "
                f"{' and '.join(control['change_between'])}, on a span of "
                f"{control.get('change_span', float('nan')):.4g}"
            )
        meaning = (
            "a scalarTransport field is not a field the outer loop solves, so naming "
            "it in residualControl changes nothing: the control never sees a function "
            "object's field. One heated run here stopped at iteration 87 on p and U "
            "with T's residual at 6e-3 and the quantity being optimised still rising "
            "3% an iteration"
            if unwatched else
            "the fields the control was watching are listed; a temperature outside "
            "that list was not part of what stopped the run"
        )
        out.append(Finding(
            "temperature-control", "warn" if unwatched else "ok", measured, meaning,
            "the two last written T fields differenced above is the measurement that "
            "settles it, whatever the control was named" if unwatched else "",
        ))

    if found["conductivity"] is not None:
        out.append(Finding(
            "conductivity", "ok",
            f"{where}k {found['conductivity']:.6g} W/mK ({found['conductivity_from']})",
            "every term of Nu = h L / k is read or given, and this is the term a "
            "wrong guess scales the answer by",
        ))

    for problem in found["problems"]:
        out.append(Finding("heat", "skipped", f"{where}{problem}"))
    return out


def findings(found: dict[str, Any]) -> list[Finding]:
    out: list[Finding] = []
    regions = found["regions"]
    if regions:
        names = [entry["region"] for entry in regions if entry["region"]]
        if names:
            out.append(Finding(
                "regions", "ok",
                f"{len(names)} mesh regions: {', '.join(names)}"
                + (f" (constant/regionProperties declares "
                   f"{', '.join(found['declared_regions'])})"
                   if found["declared_regions"] else ""),
                "each region is reported on its own; the interface flux appears once "
                "per side and the two agreeing is a conservation check",
            ))
        else:
            out.append(Finding(
                "regions", "ok", "one region: constant/polyMesh, no regions declared"
                if not found["declared_regions"] else
                "one region read, though constant/regionProperties declares "
                + ", ".join(found["declared_regions"]),
                "a single-region case says so rather than saying nothing: one region "
                "and unreadable regions are different events",
            ))
    for entry in regions:
        out.extend(region_findings(entry))
    for problem in found["problems"]:
        out.append(Finding("heat", "skipped", problem))
    return out


# --------------------------------------------------------------- reporting --


def report(found: dict[str, Any]) -> str:
    lines = [f"heat transfer: {found['case']}"]
    for entry in found["regions"]:
        lines.append("")
        head = f"  region {entry['region']}" if entry["region"] else "  one region"
        lines.append(f"{head}   {entry['mesh']}")
        for source in entry["sources"]:
            at = source.get("time")
            line = f"    {source['source']} -- {source['detail']}"
            if at is not None:
                line += f"   time {at:g}"
            lines.append(line)
            if source.get("times") and len(source["times"]) > 1:
                lines.append(
                    f"      stitched over {source['times'][0]:g} .. "
                    f"{source['times'][-1]:g} ({len(source['times'])} writes)"
                )
            lines.append(
                f"      {'patch':<16}{'faces':>8}{'area m2':>12}{'Q W':>14}"
                f"{'mean W/m2':>13}{'min':>12}{'max':>12}"
            )
            for row in source["patches"]:
                faces = str(row["faces"]) if row["faces"] else "-"
                lines.append(
                    f"      {row['patch']:<16}{faces:>8}{_fmt(row['area']):>12}"
                    f"{_fmt(row['heat_rate']):>14}{_fmt(row['mean_flux']):>13}"
                    f"{_fmt(row['min']):>12}{_fmt(row['max']):>12}"
                )
            total = total_heat_rate(source)
            if total is not None:
                lines.append(f"      {'total':<16}{'':>8}{'':>12}{total:>14.6g}")
        if entry.get("reference"):
            reference = entry["reference"]
            lines.append(
                f"    T_ref {reference['value']:.6g} -- {reference['convention']}: "
                f"{reference['detail']}"
            )
        for row in entry["nusselt"]:
            lines.append(
                f"    Nu {row['Nu']:.6g} on {row['patch']}   h {row['h']:.6g} W/m2K, "
                f"L {row['length']:g} m, k {row['k']:.6g} W/mK, "
                f"T_wall {row['t_wall']:.6g}, T_ref {row['t_ref']:.6g} "
                f"({row['convention']})"
            )
            published = row.get("correlation")
            if published and published.get("outside_range"):
                lines.append(f"      {published['name']} not evaluated: "
                             + "; ".join(published["outside_range"]))
            elif published:
                lines.append(
                    f"      against {published['name']}: {published['Nu']:.6g}, "
                    + (f"{100.0 * published['relative']:.1f}% away"
                       if published["relative"] is not None else "not compared")
                    + f", band {100.0 * published['band']:.0f}%"
                )
        if entry["conductivity"] is not None:
            lines.append(
                f"    k {entry['conductivity']:.6g} W/mK   "
                f"({entry['conductivity_from']})"
            )

    lines.append("")
    lines.append("findings")
    for finding in sorted(findings(found),
                          key=lambda f: -preflight._SEVERITY.get(f.status, 0)):
        lines.append(f"  [{finding.status}] {finding.check}")
        lines.append(f"      measured: {finding.measured}")
        if finding.meaning:
            lines.append(f"      means:    {finding.meaning}")
        if finding.repair:
            lines.append(f"      repair:   {finding.repair}")
    lines.append("")
    lines.append(
        "nothing here decides whether this heat transfer is good enough, and a "
        "Nusselt number carries its reference temperature or it carries nothing."
    )
    return "\n".join(lines)


def plot(found: dict[str, Any], out: Path) -> Path | None:
    """Two panels: the heat rate per patch, and where the temperature sits.

    A bar per patch because the question a heated run is asked first is which wall
    the energy went through, and a histogram beside it because a mean temperature
    over a solid says nothing about the tail that melts.
    """
    bars: list[tuple[str, float]] = []
    temperatures: list[np.ndarray] = []
    marks: list[tuple[str, float]] = []
    for entry in found["regions"]:
        label = f"{entry['region']}/" if entry["region"] else ""
        source = primary(entry)
        for row in (source or {}).get("patches", []):
            if row["heat_rate"] is not None:
                bars.append((f"{label}{row['patch']}", float(row["heat_rate"])))
        hot = entry.get("hot")
        if hot and "uniform" not in hot and "min" in hot:
            marks.append((f"{label}threshold", float(hot["threshold"])))
        values = region_temperatures(entry, found)
        if values is not None and values.size:
            temperatures.append(values)
    if not bars and not temperatures:
        return None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if bars:
        names = [name for name, _ in bars]
        rates = [rate for _, rate in bars]
        axes[0].barh(range(len(bars)), rates, color="#b4503c")
        axes[0].set_yticks(range(len(bars)))
        axes[0].set_yticklabels(names)
        axes[0].invert_yaxis()
        axes[0].axvline(0.0, color="#555555", lw=1)
        axes[0].set_xlabel("heat rate (W, positive out of the domain)")
    else:
        axes[0].text(0.5, 0.5, "no per-patch heat rate", ha="center", va="center")
        axes[0].set_axis_off()
    if temperatures:
        merged = np.concatenate(temperatures)
        axes[1].hist(merged[np.isfinite(merged)], bins=40, color="#3b6ea5")
        for name, value in marks:
            axes[1].axvline(value, color="#b4503c", ls="--", lw=1, label=name)
        if marks:
            axes[1].legend(fontsize="x-small")
        axes[1].set_xlabel("T")
        axes[1].set_ylabel("cells")
    else:
        axes[1].text(0.5, 0.5, "no written T field", ha="center", va="center")
        axes[1].set_axis_off()
    figure.suptitle(Path(found["case"]).name)
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    plt.close(figure)
    return out


def region_temperatures(entry: dict[str, Any], found: dict[str, Any]) -> np.ndarray | None:
    """The region's internal T values, for the histogram. Read again rather than
    carried in the measurement, which keeps a million cells out of the `--json`."""
    entries, _ = regions_of(Path(found["case"]))
    for region in entries:
        if region.name != entry["region"]:
            continue
        got = field_file(region, "T", found.get("time"))
        if got is None:
            return None
        data = locate.read_internal_field(got[0])
        if data is None or data["kind"] != "nonuniform":
            return None
        return np.asarray(data["values"], dtype=float).reshape(-1)
    return None


def as_json(found: dict[str, Any]) -> str:
    payload = dict(found)
    found_findings = findings(found)
    payload["findings"] = [finding.as_dict() for finding in found_findings]
    payload["worst"] = preflight.worst_status(found_findings)
    return json.dumps(payload, indent=2, sort_keys=False, default=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, help="the case directory")
    parser.add_argument("--time", default=None,
                        help="read this time, not the latest written")
    parser.add_argument("--region", default=None,
                        help="report only this mesh region")
    parser.add_argument("--length", type=float, default=None,
                        help="the length scale Nu is taken on, in metres")
    parser.add_argument("--conductivity", type=float, default=None, metavar="K",
                        help="thermal conductivity in W/m K, where the case states none")
    parser.add_argument("--t-wall", type=float, default=None,
                        help="wall temperature, instead of the T field's own")
    parser.add_argument("--t-ref", type=float, default=None,
                        help="a free-stream or ambient reference temperature")
    parser.add_argument("--ref-patch", default=None, metavar="NAME",
                        help="take the reference temperature over this patch")
    parser.add_argument("--t-ref-from", choices=CONVENTIONS, default=None,
                        help="which reference temperature convention to use "
                             "(bulk is the mixed mean and the default over a patch)")
    parser.add_argument("--reynolds", type=float, default=None,
                        help="with --prandtl, compare Nu against the published "
                             "correlation for a pipe flow at this Re")
    parser.add_argument("--prandtl", type=float, default=None,
                        help="with --reynolds, the Prandtl number the correlation is on")
    parser.add_argument("--threshold", type=float, default=None,
                        help="the temperature a hot spot starts at, outright")
    parser.add_argument("--hot-fraction", type=float, default=HOT_FRACTION,
                        help="where the hot region starts as a fraction of T's own "
                             f"min-to-max span (default {HOT_FRACTION:g})")
    parser.add_argument("--json", dest="json_out", nargs="?", const="-", default=None,
                        metavar="PATH",
                        help="the same facts as JSON, on stdout or to a file")
    parser.add_argument("--png", default=None, metavar="PATH",
                        help="a per-patch heat-rate chart and a temperature histogram")
    args = parser.parse_args(argv)

    if not Path(args.case).is_dir():
        # #194, the same as wall.py: a missing directory read as a case with nothing in
        # it printed "no wall heat flux anywhere" and exited 0, so a typo came back as
        # a finding about a run.
        print(f"there is no directory at {Path(args.case).as_posix()}; this reads the "
              "heat a run left behind, so it wants the case directory that holds the "
              "run", file=sys.stderr)
        return 2

    try:
        found = measure(
            args.case, time=args.time, region=args.region, length=args.length,
            k=args.conductivity, t_wall=args.t_wall, t_ref=args.t_ref,
            ref_patch=args.ref_patch, ref_from=args.t_ref_from,
            reynolds=args.reynolds, prandtl=args.prandtl,
            threshold=args.threshold, fraction=args.hot_fraction,
        )
    except Exception as error:  # pragma: no cover - the last net under the contract
        found = {
            "case": Path(args.case).as_posix(), "time": args.time,
            "declared_regions": [], "multi_region": False, "regions": [],
            "problems": [f"heat.py raised {type(error).__name__}: {error}"],
        }

    if args.png:
        try:
            written = plot(found, Path(args.png))
        except Exception as error:  # pragma: no cover - a picture is never the point
            found["problems"].append(f"the figure could not be drawn: {error}")
            written = None
        found["picture"] = written.resolve().as_posix() if written else None

    payload = as_json(found)
    if args.json_out and args.json_out != "-":
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(report(found))
        print(f"json:    {path.resolve().as_posix()}")
    elif args.json_out:
        print(payload)
    else:
        print(report(found))
    if args.png and found.get("picture"):
        print(f"picture: {found['picture']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
