#!/usr/bin/env python3
r"""Complete, runnable OpenFOAM cases around a surface you uploaded.

`case_gen.py` builds a body-fitted blockMesh for shapes it knows how to draw. This
builds the other half: a background box, snappyHexMesh cut to an STL, boundary
layers sized to a y+ you name, and every dictionary the solve needs. Between them
they cover external flow over real geometry, which is what most studies are.

Three things here are deliberate, because each was a wrong answer that shipped:

* **The reference area is measured off the surface, not typed in.** Wetted area is
  the triangle sum; frontal area is the silhouette, rasterised, so a body with a
  hollow or a second part behind the first does not get its shadow counted twice.

* **A symmetry plane halves the reference area, in the same place it halves the
  mesh.** A half model whose force is divided by the whole body's area reads
  exactly half, and it reads plausible. The two decisions live in one function so
  they cannot drift apart.

* **The first layer follows from y+, not from a guess.** Ask for `--y-plus 1` or
  `--y-plus 50` and the stack is sized to land there, with the correlation that
  produced the number printed beside it.

Everything is written as ordinary dictionaries you can open and edit. If a choice
is wrong for your case, the summary names the file it lives in.

    python3 snappy_gen.py CASE --stl hull.stl --speed 3.046 --nu 1e-6             --symmetry y --area wetted --surface-cell 0.02 --y-plus 50
    python3 snappy_gen.py CASE --stl prop.stl --speed 0.1 --mrf --mrf-rpm 4014             --mrf-axis x --study steady
    python3 snappy_gen.py CASE --stl bundle.stl --speed 6 --study thermal             --wall-temperature 333.15 --inlet-temperature 293.15

`--dry-run` lists the files without writing them. Then `sh CASE/Allmesh` is the
whole mesh: blockMesh, feature extraction, snappyHexMesh, checkMesh and a digest,
in one call rather than five round trips.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import case_gen
import study_state


# -- reading a surface -------------------------------------------------------------

class Surface:
    """Triangles off an STL, and the handful of numbers a case needs from them."""

    def __init__(self, name: str, triangles: list, path: Path):
        self.name = name
        self.path = path
        self.triangles = triangles
        xs = [p[0] for tri in triangles for p in tri]
        ys = [p[1] for tri in triangles for p in tri]
        zs = [p[2] for tri in triangles for p in tri]
        self.bounds = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    @property
    def extent(self):
        x0, x1, y0, y1, z0, z1 = self.bounds
        return (x1 - x0, y1 - y0, z1 - z0)

    @property
    def centre(self):
        x0, x1, y0, y1, z0, z1 = self.bounds
        return ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)

    def wetted_area(self) -> float:
        """Sum of the triangle areas: the area a shear stress acts on."""
        total = 0.0
        for a, b, c in self.triangles:
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            cx = uy * vz - uz * vy
            cy = uz * vx - ux * vz
            cz = ux * vy - uy * vx
            total += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        return total

    def frontal_area(self, axis: int = 0, grid: int = 500) -> float:
        """Projected area normal to `axis`, by rasterising the silhouette.

        The cheap form is 0.5 * sum(A_i |n_i . d|), which is exact for a convex
        closed body and an overestimate for anything with a hollow or a second
        body behind the first -- a duct, a bundle of tubes, a rotor behind a hub.
        Rasterising costs a second and is right for all of them.
        """
        u, v = [i for i in (0, 1, 2) if i != axis]
        lo_u, hi_u = self.bounds[2 * u], self.bounds[2 * u + 1]
        lo_v, hi_v = self.bounds[2 * v], self.bounds[2 * v + 1]
        span_u, span_v = hi_u - lo_u, hi_v - lo_v
        if span_u <= 0 or span_v <= 0:
            return 0.0
        # Square pixels, so one cell's area is exact rather than a mean.
        step = max(span_u, span_v) / grid
        nu = max(1, int(math.ceil(span_u / step)))
        nv = max(1, int(math.ceil(span_v / step)))
        covered = bytearray(nu * nv)

        for tri in self.triangles:
            pu = [(p[u] - lo_u) / step for p in tri]
            pv = [(p[v] - lo_v) / step for p in tri]
            v0 = max(0, int(math.floor(min(pv))))
            v1 = min(nv - 1, int(math.ceil(max(pv))))
            for row in range(v0, v1 + 1):
                yc = row + 0.5
                spans = []
                for i in range(3):
                    j = (i + 1) % 3
                    ya, yb = pv[i], pv[j]
                    if (ya <= yc < yb) or (yb <= yc < ya):
                        t = (yc - ya) / (yb - ya)
                        spans.append(pu[i] + t * (pu[j] - pu[i]))
                if len(spans) < 2:
                    continue
                a, b = min(spans), max(spans)
                c0 = max(0, int(math.floor(a + 0.5)))
                c1 = min(nu - 1, int(math.ceil(b - 0.5)))
                base = row * nu
                for col in range(c0, c1 + 1):
                    covered[base + col] = 1
        return sum(covered) * step * step


def write_binary_stl(path: Path, triangles) -> None:
    """A binary STL of exactly the triangles that were meshed."""
    with path.open("wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            fh.write(struct.pack("<3f", nx / length, ny / length, nz / length))
            for point in (a, b, c):
                fh.write(struct.pack("<3f", *point))
            fh.write(struct.pack("<H", 0))


def read_stl(path: Path) -> Surface:
    """ASCII or binary STL, whichever it turns out to be.

    The test is the content, not the extension: plenty of binary STLs carry a
    header beginning "solid", which is the classic way a reader decides a 40 MB
    binary file is an empty ASCII one and returns a surface with no triangles.
    Sizing off the byte count settles it.
    """
    raw = path.read_bytes()
    if len(raw) < 15:
        raise SystemExit(f"{path} is too short to be an STL")
    triangles = []
    binary = True
    if raw[:5].lower() == b"solid":
        count = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else 0
        binary = len(raw) == 84 + count * 50
    if binary:
        count = struct.unpack("<I", raw[80:84])[0]
        expected = 84 + count * 50
        if len(raw) < expected:
            raise SystemExit(
                f"{path}: the header says {count} triangles, which needs {expected} "
                f"bytes, but the file is {len(raw)}. A truncated upload?"
            )
        off = 84
        for _ in range(count):
            vals = struct.unpack_from("<12f", raw, off)
            triangles.append((vals[3:6], vals[6:9], vals[9:12]))
            off += 50
    else:
        pts = []
        for line in raw.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "vertex":
                pts.append(tuple(float(x) for x in parts[1:4]))
                if len(pts) == 3:
                    triangles.append(tuple(pts))
                    pts = []
    if not triangles:
        raise SystemExit(f"{path} parsed to no triangles")
    return Surface(path.stem, triangles, path)


MILLIMETRE_HINT = 50.0
"""A body more than this many 'metres' across is almost certainly millimetres.

Reported, not corrected: a silent factor of a thousand is precisely the failure
this is meant to catch, so the generator says so and lets you decide."""


# nu and rho are not independent: they are two properties of one fluid, and the
# pairs people actually run are few. Air at 20 C is (1.5e-5, 1.2); fresh water is
# (1.0e-6, 998); sea water is (1.1e-6, 1025).
FLUIDS = (("air", 1.5e-5, 1.205), ("water", 1.0e-6, 998.2),
          ("sea water", 1.19e-6, 1025.0))


def fluid_notes(nu: float, density: float) -> list[str]:
    """Catch a viscosity from one fluid with a density from another.

    This costs nothing to check and it is a factor of 830 when it happens: the
    velocity field an incompressible solver produces depends only on nu, so a hull
    run in water with air's density gives a perfectly correct Cd beside a force
    three orders of magnitude too small, and the coefficient's correctness is
    exactly what stops anyone looking at the force."""
    best = min(FLUIDS, key=lambda f: abs(math.log(nu / f[1])))
    name, fluid_nu, fluid_rho = best
    if abs(math.log(density / fluid_rho)) > math.log(2.0):
        return [
            f"!! nu = {nu:.3g} m2/s is {name} ({fluid_nu:.3g}), but --density is "
            f"{density:g} kg/m3 and {name} is {fluid_rho:g}. Coefficients will still "
            f"be right -- they do not use rho -- but every FORCE in newtons will be "
            f"out by {fluid_rho / density:.0f}x. --density {fluid_rho:g} if that is "
            f"the fluid you meant."
        ]
    return []


def axis_notes(surface: Surface, mrf_axis: str | None) -> list[str]:
    """Say which way the body faces, and object if a rotor spins about the wrong one.

    Looking down a propeller's own shaft you see its blade planform, which is much
    the largest of the three projections. So the axis of maximum projected area IS
    the shaft, and an --mrf-axis that disagrees is spinning the disc about a line
    lying in it. The symptom is a thrust that looks plausible beside a torque an
    order of magnitude too big, which reads as a mesh problem and is not one."""
    areas = {name: surface.frontal_area(axis=i) for name, i in AXES.items()}
    biggest = max(areas, key=areas.get)
    notes = [
        "projected area  " + ",  ".join(
            f"down {name} {areas[name]:.6g} m2" for name in ("x", "y", "z"))
    ]
    if mrf_axis and mrf_axis != biggest:
        notes.append(
            f"!! --mrf-axis {mrf_axis}, but the body's largest projection is down "
            f"{biggest} ({areas[biggest]:.6g} m2 against {areas[mrf_axis]:.6g}). For a "
            f"rotor that projection IS the disc, so the shaft is {biggest}, not "
            f"{mrf_axis}. Either --mrf-axis {biggest}, or --rotate to bring the shaft "
            f"onto x where the flow is."
        )
    return notes


def scale_notes(surface: Surface, scale: float) -> list[str]:
    notes = []
    span = max(surface.extent)
    if scale == 1.0 and span > MILLIMETRE_HINT:
        notes.append(
            f"!! {surface.name} reads {span:.4g} m across. That is big enough that the "
            f"file is probably in millimetres -- --stl-scale 0.001 if so."
        )
    if scale == 1.0 and span < 1e-3:
        notes.append(
            f"!! {surface.name} reads {span:.4g} m across, under a millimetre. "
            f"Check the units before meshing it."
        )
    return notes


def parse_rotation(spec) -> list[tuple[int, float]]:
    """`--rotate y:90,z:-30` -> [(1, 90.0), (2, -30.0)], applied in order."""
    if not spec:
        return []
    out = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        axis, _, angle = item.partition(":")
        if axis not in AXES or not angle:
            raise SystemExit(
                f"--rotate {item}: expected axis:degrees, e.g. y:90")
        try:
            out.append((AXES[axis], float(angle)))
        except ValueError:
            raise SystemExit(f"--rotate {item}: '{angle}' is not a number")
    return out


def rotated(surface: Surface, turns: list[tuple[int, float]]) -> Surface:
    """Turn the surface before meshing.

    Uploaded geometry does not arrive aligned to anybody's wind tunnel, and this
    generator's inlet is always -x. A propeller exported with its shaft along z is
    not a broken file, it is a normal one, and meshing it as-is spins it about an
    axis lying in its own disc."""
    if not turns:
        return surface
    tris = surface.triangles
    for axis, degrees in turns:
        c, sn = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        u, v = [i for i in (0, 1, 2) if i != axis]
        # Right-handed about `axis`, so y:90 carries +z onto +x.
        def turn(pt, u=u, v=v, c=c, sn=sn):
            out = list(pt)
            out[u] = pt[u] * c + pt[v] * sn
            out[v] = -pt[u] * sn + pt[v] * c
            return tuple(out)
        tris = [tuple(turn(pt) for pt in tri) for tri in tris]
    return Surface(surface.name, tris, surface.path)


def rescaled(surface: Surface, scale: float) -> Surface:
    if scale == 1.0:
        return surface
    tris = [tuple(tuple(c * scale for c in p) for p in tri) for tri in surface.triangles]
    return Surface(surface.name, tris, surface.path)


# -- the boundary layer ------------------------------------------------------------

def flat_plate_cf(reynolds: float) -> float:
    """Local skin friction on a smooth flat plate: Cf = 0.058 Re^-0.2.

    Used only to size the first cell, where being 20% out moves y+ by 10% and
    changes nothing that matters."""
    return 0.058 * max(reynolds, 1.0) ** -0.2


def ittc57(reynolds: float) -> float:
    """The ITTC-57 model-ship correlation line, Cf = 0.075/(log10(Re)-2)^2.

    Printed in the summary as a floor: a *total* drag coefficient below the
    flat-plate friction of the same body at the same Reynolds number is not a
    tight result, it is an impossible one, and saying so up front is cheaper than
    discovering it in the write-up."""
    return 0.075 / (math.log10(max(reynolds, 11.0)) - 2.0) ** 2


def first_layer_thickness(y_plus: float, flow, length: float) -> tuple[float, float]:
    """(cell thickness, u_tau) for a wanted y+.

    y+ is defined on the distance from the wall to the first cell's *centroid*, so
    the cell itself is twice that. Skipping the factor of two is the commonest
    reason a mesh built 'for y+ 1' reports 2."""
    reynolds = flow.speed * length / flow.nu
    cf = flat_plate_cf(reynolds)
    u_tau = flow.speed * math.sqrt(cf / 2.0)
    centroid = y_plus * flow.nu / u_tau
    return 2.0 * centroid, u_tau


BUFFER_LAYER = (5.0, 30.0)
"""The y+ band where neither wall treatment applies.

Below 5 the sublayer wants resolving and a wall function has nothing to bridge;
above 30 the first cell is in the log region and a wall function is exact there.
Between them the answer is a blend of two approximations, neither of which holds,
and it moves with the mesh -- four meshes of one hull at y+ 4.3, 9.6, 16.9 and 33.5
gave form factors of 0.786, 0.928, 0.977 and 1.018. The first three look like a
convergence trend and are three invalid points."""


def y_plus_notes(y_plus: float, layers: int) -> list[str]:
    low, high = BUFFER_LAYER
    if low <= y_plus <= high:
        return [
            f"!! y+ {y_plus:g} is in the buffer layer ({low:g}-{high:g}), where neither "
            f"treatment holds: too coarse to resolve the sublayer, too fine for a wall "
            f"function to bridge. Aim under {low:g} with enough layers to resolve, or "
            f"over {high:g} and let the wall function do its job. A number from in "
            f"here moves with the mesh and does not converge."
        ]
    if y_plus < low and layers < 5:
        return [
            f"!! y+ {y_plus:g} asks for a resolved sublayer but there are only "
            f"{layers} layers. Resolving means carrying the profile out through the "
            f"buffer region, which takes 10 or more; with this few, the second cell "
            f"lands in the buffer layer and undoes the first."
        ]
    return []


def stack_thickness(first: float, layers: int, ratio: float) -> float:
    if layers <= 0:
        return 0.0
    if abs(ratio - 1.0) < 1e-9:
        return first * layers
    return first * (ratio ** layers - 1.0) / (ratio - 1.0)


# -- the background box ------------------------------------------------------------

AXES = {"x": 0, "y": 1, "z": 2}

FACE_VERTICES = {
    "xmin": (0, 4, 7, 3), "xmax": (1, 2, 6, 5),
    "ymin": (0, 1, 5, 4), "ymax": (3, 7, 6, 2),
    "zmin": (0, 3, 2, 1), "zmax": (4, 5, 6, 7),
}


class Domain:
    """The blockMesh box snappy carves the body out of."""

    def __init__(self, bounds, cells, patches: dict, planes=()):
        self.bounds = bounds          # (x0, x1, y0, y1, z0, z1)
        self.cells = cells            # (nx, ny, nz)
        self.patches = patches        # name -> [face key, ...]
        self.planes = list(planes)    # the symmetry cuts, in order

    @property
    def symmetry(self) -> str:
        return ",".join(p["name"] for p in self.planes) or "none"

    @property
    def halvings(self) -> int:
        """How many cuts actually bisect the body.

        A plane tangent to it -- a waterline closing a hull, a floor under a car --
        bounds the flow without removing any of the body, and halving the reference
        area for one of those is the same error as failing to halve for one that
        does."""
        return sum(1 for p in self.planes if p["bisects"])

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.cells
        return nx * ny * nz

    @property
    def cell_size(self):
        x0, x1, y0, y1, z0, z1 = self.bounds
        nx, ny, nz = self.cells
        return ((x1 - x0) / nx, (y1 - y0) / ny, (z1 - z0) / nz)


def build_domain(surface: Surface, opts) -> tuple[Domain, list[str]]:
    """A box around the body, sized in body lengths, cut on a symmetry plane."""
    notes = []
    x0, x1, y0, y1, z0, z1 = surface.bounds
    # Margins scale on the body's LARGEST span, not its streamwise one. A propeller
    # lying in the y-z plane has an x extent of a few millimetres -- its blade
    # thickness -- and sizing a wind tunnel in multiples of that gives a domain
    # smaller than the disc it is meant to contain.
    length = max(x1 - x0, y1 - y0, z1 - z0)
    bounds = [x0 - opts["ahead"] * length, x1 + opts["behind"] * length,
              y0 - opts["side"] * length, y1 + opts["side"] * length,
              z0 - opts["below"] * length, z1 + opts["above"] * length]

    planes = parse_symmetry(opts.get("symmetry"), surface)
    for spec in planes:
        if spec["keep"] == "low":
            bounds[2 * spec["axis"] + 1] = spec["plane"]
        else:
            bounds[2 * spec["axis"]] = spec["plane"]
        if spec["bisects"]:
            notes.append(
                f"{spec['name']} symmetry at {spec['plane']:.6g} m cuts the body in "
                f"half; the reference area is halved with it"
            )
        else:
            notes.append(
                f"{spec['name']} symmetry at {spec['plane']:.6g} m sits at the body's "
                f"own edge, so it bounds the flow without removing any of the body "
                f"-- the reference area is NOT halved for it"
            )
    domain_bounds = tuple(bounds)

    span = max(domain_bounds[1] - domain_bounds[0],
               domain_bounds[3] - domain_bounds[2],
               domain_bounds[5] - domain_bounds[4])
    base = opts.get("base_cell") or span / opts["background_cells"]
    # A background cell wider than the body is not merely coarse, it is silent:
    # snappy marks cells whose EDGES the surface crosses, so a slender hull lying
    # inside one row of cells intersects nothing, refines nothing, and finishes
    # "without any errors" having meshed an empty box. Requiring the body to span
    # a few cells across its thinnest direction is what makes the cut happen at all.
    # The test is the MEDIAN span, not the smallest. A thin blade is thinner than
    # any sensible cell and still crosses thousands of edges, because it is wide in
    # the other two directions; what refines nothing is a body smaller than a cell
    # in two directions at once, which is the slender hull above.
    median = sorted(surface.extent)[1]
    across = max(1, int(opts.get("cells_across_body") or 2))
    cap = median / across
    if median > 0 and base > cap:
        notes.append(
            f"background cell {base:.4g} m would be wider than the body's "
            f"{median:.4g} m cross-section; using {cap:.4g} m so the surface crosses "
            f"cell edges at all. --cells-across-body sets the margin."
        )
        base = cap
    cells = tuple(
        max(1, int(round((domain_bounds[2 * i + 1] - domain_bounds[2 * i]) / base)))
        for i in range(3)
    )

    patches = {"inlet": ["xmin"], "outlet": ["xmax"], "top": ["zmax"],
               "bottom": ["zmin"], "side": ["ymin", "ymax"]}
    # One patch per plane, never one patch holding several. `symmetryPlane` is a
    # constraint that requires its faces to be coplanar, so merging a centreline and
    # a waterline into a single patch is rejected outright:
    #   "Symmetry plane 'symmetry' is not planar ... the normal (0 -1 0) differs
    #    from the average normal by 0.57"
    for spec in planes:
        cut = ({0: "xmax", 1: "ymax", 2: "zmax"} if spec["keep"] == "low"
               else {0: "xmin", 1: "ymin", 2: "zmin"})[spec["axis"]]
        for name in list(patches):
            patches[name] = [f for f in patches[name] if f != cut]
            if not patches[name]:
                del patches[name]
        label = "symmetry" if len(planes) == 1 else f"symmetry{spec['name'].upper()}"
        patches[label] = [cut]
    return Domain(domain_bounds, cells, patches, planes), notes


OFF_FACE = 0.4137
"""How far across a span to put a point that must not land on a cell face.

Any fraction with a short binary expansion -- 1/2, 1/4, 3/8 -- coincides with a face
for some cell count, and the failure is a confusing one because the rejected point is
visibly inside the printed bounding box."""


def parse_symmetry(spec, surface: Surface) -> list[dict]:
    """`--symmetry y,z:max` -> the planes to cut on, and whether each halves the body.

    A bare axis means the body's own mid-plane: a hull exported with its origin at
    the bow is still symmetric about its own centreline, and cutting at zero would
    slice it off-centre without saying so. `:max` and `:min` put the plane at the
    body's own extreme, which is what a double-body model wants at the waterline --
    there the flat lid that closes the hull IS the plane, rather than a deck towed
    through the water and charged for its friction.
    """
    if not spec or spec == "none":
        return []
    planes = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        axis_name, _, where = item.partition(":")
        if axis_name not in AXES:
            raise SystemExit(f"--symmetry: '{axis_name}' is not one of x, y, z")
        axis = AXES[axis_name]
        lo, hi = surface.bounds[2 * axis], surface.bounds[2 * axis + 1]
        if where in ("", "mid"):
            centre = surface.centre[axis]
            plane = 0.0 if abs(centre) <= 1e-6 * max(hi - lo, 1e-12) else centre
        elif where == "max":
            plane = hi
        elif where == "min":
            plane = lo
        else:
            try:
                plane = float(where)
            except ValueError:
                raise SystemExit(
                    f"--symmetry {item}: expected an axis, or axis:min, axis:max, "
                    f"or axis:<coordinate>")
        margin = 1e-6 * max(hi - lo, 1e-12)
        # Which half to keep is not a preference, it is where the body is. A hull
        # closed at its waterline lies BELOW that plane, so cutting the domain's
        # floor there would leave the hull outside the mesh entirely. A plane that
        # bisects keeps the upper half by convention; one at the body's own top
        # keeps everything under it.
        keep = "low" if plane >= hi - margin else "high"
        planes.append({"axis": axis, "name": axis_name, "plane": plane,
                       "bisects": lo + margin < plane < hi - margin,
                       "keep": keep})
    return planes


def inside_point(domain: Domain, surface: Surface) -> tuple[float, float, float]:
    """A point in the fluid, for snappy's locationInMesh.

    A quarter of the way from the domain's own corner toward the body's bounding
    box is outside any body that fits in that box, which is all of them. Picking
    the domain centre instead lands inside the geometry for anything hollow."""
    point = []
    for i in range(3):
        d_lo, d_hi = domain.bounds[2 * i], domain.bounds[2 * i + 1]
        b_lo = surface.bounds[2 * i]
        # The body is not always inside the box. A domain deliberately cut INSIDE
        # the geometry -- tubes trimmed by their own end planes so they span the
        # bank the way the correlation assumes -- leaves the body's lower bound
        # BELOW the domain's, and stepping a fraction of the way towards it walks
        # out through the floor. snappy then rejects the point while printing a
        # bounding box that does not contain it, which reads as a domain problem.
        if b_lo <= d_lo + 1e-9 * (d_hi - d_lo):
            point.append(d_lo + OFF_FACE * (d_hi - d_lo))
            continue
        if any(spec["axis"] == i for spec in domain.planes):
            # The cut face is a boundary, not somewhere to sit: come off it. The
            # fraction is deliberately not a round one -- half of an even number of
            # cells is exactly a cell face, and snappy refuses a locationInMesh that
            # sits on a face or an edge:
            #   "Point (...) is not inside the mesh or on a face or edge"
            # while printing a bounding box that plainly contains it, which sends
            # you looking at the domain instead of at the arithmetic.
            point.append(d_lo + OFF_FACE * (d_hi - d_lo))
        else:
            point.append(d_lo + 0.25 * OFF_FACE * 4 * (b_lo - d_lo))
    return tuple(point)


# -- the patch roles ---------------------------------------------------------------

class PatchList:
    """Stands in for `case_gen.Mesh2D` where the 0/ writers want only patch names."""

    def __init__(self, names):
        self.patch_faces = {name: 1 for name in names}


def build_roles(domain: Domain, body_patch: str, opts) -> dict:
    far = opts.get("far") or "slip"
    roles = {}
    for name in domain.patches:
        if name == "inlet":
            roles[name] = {"kind": "inlet", "direction": (1.0, 0.0, 0.0)}
        elif name == "outlet":
            roles[name] = {"kind": "outlet"}
        elif name.startswith("symmetry"):
            roles[name] = {"kind": "symmetry"}
        elif name == "bottom" and opts.get("ground"):
            roles[name] = {"kind": "belt"} if opts.get("moving_ground") else {"kind": "wall"}
        else:
            roles[name] = {"kind": far}
    roles[body_patch] = {"kind": "wall"}
    return roles


# -- dictionaries ------------------------------------------------------------------

def block_mesh_dict(domain: Domain, roles: dict | None = None) -> str:
    x0, x1, y0, y1, z0, z1 = domain.bounds
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    lines = ["scale   1;", "", "vertices", "("]
    lines += [f"    ({p[0]:.8g} {p[1]:.8g} {p[2]:.8g})" for p in corners]
    lines += [");", "", "blocks", "(",
              f"    hex (0 1 2 3 4 5 6 7) "
              f"({domain.cells[0]} {domain.cells[1]} {domain.cells[2]}) "
              f"simpleGrading (1 1 1)",
              ");", "", "edges", "(", ");", "", "boundary", "("]
    for name in case_gen.patch_order(domain.patches):
        role = (roles or {}).get(name, {}).get("kind", "")
        kind = "symmetry" if (role == "symmetry" or name.startswith("symmetry")) \
            else "patch"
        lines += [f"    {name}", "    {", f"        type            {kind};",
                  "        faces", "        ("]
        for face in domain.patches[name]:
            v = FACE_VERTICES[face]
            lines.append(f"            ({v[0]} {v[1]} {v[2]} {v[3]})")
        lines += ["        );", "    }"]
    lines += [");", "", "mergePatchPairs", "(", ");"]
    return case_gen.foam_file("dictionary", "blockMeshDict", "\n".join(lines), "system")


def surface_feature_dict(names: list[str], angle: float) -> str:
    lines = []
    for name in names:
        lines += [f"{name}.stl", "{", "    extractionMethod    extractFromSurface;",
                  "    extractFromSurfaceCoeffs", "    {",
                  f"        includedAngle   {angle:g};", "    }",
                  "    writeObj            no;", "}", ""]
    return case_gen.foam_file("dictionary", "surfaceFeatureExtractDict",
                              "\n".join(lines), "system")


def snappy_dict(surfaces, opts, layer: dict, point) -> str:
    level = opts["refine"]
    near = max(0, level - 1)
    lines = ["castellatedMesh true;", "snap            true;",
             f"addLayers       {'true' if layer['layers'] > 0 else 'false'};",
             "", "geometry", "{"]
    for s in surfaces:
        lines += [f"    {s.name}.stl", "    {", "        type            triSurfaceMesh;",
                  f"        name            {s.name};", "    }"]
    box = opts.get("refinement_box")
    if box:
        lines += ["    wake", "    {", "        type            searchableBox;",
                  f"        min             ({box[0]:.8g} {box[2]:.8g} {box[4]:.8g});",
                  f"        max             ({box[1]:.8g} {box[3]:.8g} {box[5]:.8g});",
                  "    }"]
    lines += ["}", "", "castellatedMeshControls", "{",
              "    maxLocalCells       2000000;",
              f"    maxGlobalCells      {opts['max_cells']};",
              "    minRefinementCells  10;", "    maxLoadUnbalance    0.10;",
              "    nCellsBetweenLevels 3;", "", "    features", "    ("]
    for s in surfaces:
        lines.append(f'        {{ file "{s.name}.eMesh"; level {level}; }}')
    lines += ["    );", "", "    refinementSurfaces", "    {"]
    for s in surfaces:
        lines += [f"        {s.name}", "        {",
                  f"            level           ({near} {level});",
                  "            patchInfo       { type wall; }", "        }"]
    lines += ["    }", "", "    resolveFeatureAngle 30;", "",
              "    refinementRegions", "    {"]
    if box:
        lines += ["        wake", "        {", "            mode            inside;",
                  f"            levels          ((1e15 {max(0, level - 2)}));", "        }"]
    lines += ["    }", "",
              f"    locationInMesh  ({point[0]:.8g} {point[1]:.8g} {point[2]:.8g});",
              "    allowFreeStandingZoneFaces true;", "}", "", "snapControls", "{",
              "    nSmoothPatch    3;", "    tolerance       2.0;",
              "    nSolveIter      50;", "    nRelaxIter      5;",
              "    nFeatureSnapIter 15;", "    implicitFeatureSnap false;",
              "    explicitFeatureSnap true;", "    multiRegionFeatureSnap false;",
              "}", "", "addLayersControls", "{",
              "    relativeSizes   false;   // firstLayerThickness is in metres",
              "    layers", "    {"]
    for s in surfaces:
        lines += [f"        {s.name}", "        {",
                  f"            nSurfaceLayers  {layer['layers']};", "        }"]
    lines += ["    }", "",
              f"    firstLayerThickness {layer['first']:.6g};",
              f"    expansionRatio  {layer['ratio']:g};",
              f"    minThickness    {layer['first'] * 0.1:.6g};",
              "    nGrow           0;", "    featureAngle    130;",
              "    slipFeatureAngle 30;", "    nRelaxIter      5;",
              "    nSmoothSurfaceNormals 1;", "    nSmoothNormals  3;",
              "    nSmoothThickness 10;", "    maxFaceThicknessRatio 0.5;",
              # These three are what decide whether a layer stack gets BUILT, and the
              # tutorial defaults assume a chunky body. On a thin one they quietly
              # refuse: `maxThicknessToMedialRatio` limits the stack by the distance
              # to the body's medial axis, and a hull 0.3 m across whose bow and
              # stern come to a knife edge has a medial distance going to zero along
              # both ends. Asking such a hull for ten layers produced 1.01 of 10 --
              # snappy declined, said nothing, and the case came back at y+ 113
              # having been asked for y+ 1.
              f"    maxThicknessToMedialRatio {opts['medial_ratio']:g};",
              f"    minMedialAxisAngle {opts['medial_angle']:g};",
              "    nBufferCellsNoExtrude 0;",
              f"    nLayerIter      {opts['layer_iter']};",
              f"    nRelaxedIter    {max(20, opts['layer_iter'] // 2)};",
              "}", "", "meshQualityControls", "{",
              '    #include "meshQualityDict"', "}", "", "writeFlags", "(",
              "    scalarLevels", "    layerSets", "    layerFields", ");", "",
              "mergeTolerance  1e-6;"]
    return case_gen.foam_file("dictionary", "snappyHexMeshDict", "\n".join(lines), "system")


MESH_QUALITY = """maxNonOrtho         65;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave          80;
minVol              1e-13;
minTetQuality       1e-15;
minArea             -1;
minTwist            0.02;
minDeterminant      0.001;
minFaceWeight       0.02;
minVolRatio         0.01;
minTriangleTwist    -1;
nSmoothScale        4;
errorReduction      0.75;

relaxed
{
    maxNonOrtho     75;
}
"""


def mesh_quality_dict() -> str:
    return case_gen.foam_file("dictionary", "meshQualityDict", MESH_QUALITY, "system")


def decompose_dict(cores: int) -> str:
    body = f"numberOfSubdomains  {cores};\n\nmethod              scotch;\n"
    return case_gen.foam_file("dictionary", "decomposeParDict", body, "system")


def topo_set_dict(opts) -> str:
    """A cylinder of cells for the rotating zone."""
    p1, p2, r = opts["mrf_p1"], opts["mrf_p2"], opts["mrf_radius"]
    body = "\n".join([
        "actions", "(", "    {", "        name        rotor;",
        "        type        cellSet;", "        action      new;",
        "        source      cylinderToCell;",
        f"        point1      ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});",
        f"        point2      ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});",
        f"        radius      {r:.6g};", "    }", "    {",
        "        name        rotor;", "        type        cellZoneSet;",
        "        action      new;", "        source      setToCellZone;",
        "        set         rotor;", "    }", ");",
    ])
    return case_gen.foam_file("dictionary", "topoSetDict", body, "system")


def mrf_properties(opts) -> str:
    p1, p2 = opts["mrf_p1"], opts["mrf_p2"]
    axis = [p2[i] - p1[i] for i in range(3)]
    norm = math.sqrt(sum(a * a for a in axis)) or 1.0
    axis = [a / norm for a in axis]
    omega = opts["mrf_rpm"] * 2.0 * math.pi / 60.0
    body = "\n".join([
        "rotor", "{", "    cellZone        rotor;", "    active          yes;", "",
        f"    origin          ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});",
        f"    axis            ({axis[0]:.6g} {axis[1]:.6g} {axis[2]:.6g});",
        f"    omega           {omega:.6g};   // {opts['mrf_rpm']:g} rpm, "
        f"{omega / (2 * math.pi):.6g} rev/s", "",
        "    nonRotatingPatches ();", "}",
    ])
    return case_gen.foam_file("dictionary", "MRFProperties", body, "constant")


def gravity_file() -> str:
    body = "dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 0 -9.81);\n"
    return case_gen.foam_file("uniformDimensionedVectorField", "g", body, "constant")


def thermo_properties(opts, flow) -> str:
    """Air as a perfect gas, with mu set from the nu the case reports.

    Letting the thermo pick its own viscosity means the Reynolds number in the
    summary is not the one the solver runs at, which is a silent disagreement
    between the write-up and the result."""
    rho = opts["density"]
    mu = rho * flow.nu
    body = "\n".join([
        "thermoType", "{", "    type            heRhoThermo;",
        "    mixture         pureMixture;", "    transport       const;",
        "    thermo          hConst;", "    equationOfState perfectGas;",
        "    specie          specie;", "    energy          sensibleEnthalpy;", "}", "",
        "mixture", "{", "    specie", "    {", "        molWeight       28.96;", "    }",
        "    thermodynamics", "    {", f"        Cp              {opts['cp']:g};",
        "        Hf              0;", "    }", "    transport", "    {",
        f"        mu              {mu:.6g};   // rho {rho:g} * nu {flow.nu:.6g}",
        f"        Pr              {opts['prandtl']:g};", "    }", "}",
    ])
    return case_gen.foam_file("dictionary", "thermophysicalProperties", body, "constant")


# -- the 0/ fields the thermal solver adds -----------------------------------------

def field_T(plan, opts) -> str:
    inlet = opts["inlet_temperature"]
    wall = opts["wall_temperature"]
    body_patch = opts["_body_patch"]

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "inlet":
            return ["type            fixedValue;", f"value           uniform {inlet:g};"]
        if kind == "outlet":
            return ["type            inletOutlet;",
                    f"inletValue      uniform {inlet:g};",
                    f"value           uniform {inlet:g};"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if name == body_patch:
            return ["type            fixedValue;", f"value           uniform {wall:g};"]
        return ["type            zeroGradient;"]

    body = (f"dimensions      [0 0 0 1 0 0 0];\n\ninternalField   uniform {inlet:g};\n\n"
            + case_gen.boundary_field(plan, entry))
    return case_gen.foam_file("volScalarField", "T", body, "0")


def field_alphat(plan, opts) -> str:
    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind in ("inlet", "outlet"):
            return ["type            calculated;", "value           uniform 0;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        return ["type            compressible::alphatWallFunction;",
                f"Prt             {opts['turbulent_prandtl']:g};",
                "value           uniform 0;"]

    body = ("dimensions      [1 -1 -1 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
            + case_gen.boundary_field(plan, entry))
    return case_gen.foam_file("volScalarField", "alphat", body, "0")


def field_p_thermal(plan, opts, name: str) -> str:
    """p and p_rgh for the compressible solver: absolute pascals, not p/rho."""
    ref = opts["pressure"]
    is_rgh = name == "p_rgh"

    def entry(patch: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "outlet":
            if is_rgh:
                return ["type            fixedValue;", f"value           uniform {ref:g};"]
            return ["type            calculated;", f"value           uniform {ref:g};"]
        if is_rgh:
            return ["type            fixedFluxPressure;", f"value           uniform {ref:g};"]
        return ["type            calculated;", f"value           uniform {ref:g};"]

    body = (f"dimensions      [1 -1 -2 0 0 0 0];\n\ninternalField   uniform {ref:g};\n\n"
            + case_gen.boundary_field(plan, entry))
    return case_gen.foam_file("volScalarField", name, body, "0")


# -- controlDict and the function objects ------------------------------------------

def force_block(opts, flow, body_patch: str) -> list[str]:
    """forceCoeffs on the body, with the reference area the mesh actually has.

    `Aref` is the halved area when the mesh is halved. That pairing is the whole
    point of this function: a half model's force over the whole body's area is
    exactly half the right answer and looks entirely reasonable."""
    lines = ["    forceCoeffs", "    {", "        type            forceCoeffs;",
             "        libs            (forces);",
             f"        patches         ({body_patch});",
             "        writeControl    timeStep;", "        writeInterval   1;",
             "        log             yes;", ""]
    # `rho` says where the density comes from; `rhoInf` is the reference density the
    # COEFFICIENTS are normalised by, and forceCoeffs wants it either way -- a
    # compressible run without it aborts on the first write with
    # "Entry 'rhoInf' not found", after the mesh is built and the case decomposed.
    lines += ["        rho             " + ("rho;" if opts["thermal"] else "rhoInf;"),
              f"        rhoInf          {opts['density']:g};"]
    lines += [f"        magUInf         {flow.speed:.6g};",
              f"        lRef            {opts['_l_ref']:.6g};",
              f"        Aref            {opts['_a_ref']:.6g};   // {opts['_a_ref_why']}",
              "        CofR            (0 0 0);",
              "        liftDir         (0 0 1);", "        dragDir         (1 0 0);",
              "        pitchAxis       (0 1 0);", "    }", ""]
    lines += ["    forces", "    {", "        type            forces;",
              "        libs            (forces);",
              f"        patches         ({body_patch});",
              "        writeControl    timeStep;", "        writeInterval   1;",
              "        log             yes;"]
    lines += ["        rho             " + ("rho;" if opts["thermal"] else "rhoInf;"),
              f"        rhoInf          {opts['density']:g};",
              "        CofR            (0 0 0);", "    }", ""]
    return lines


def function_objects(opts, flow, body_patch: str) -> str:
    lines = ["functions", "{", ""]
    lines += force_block(opts, flow, body_patch)
    lines += ["    yPlus", "    {", "        type            yPlus;",
              "        libs            (fieldFunctionObjects);",
              "        writeControl    writeTime;", "        log             yes;",
              "    }", "",
              "    solverInfo", "    {", "        type            solverInfo;",
              "        libs            (utilityFunctionObjects);",
              "        fields          (U p);", "        writeResidualFields no;",
              "        writeControl    timeStep;", "    }", ""]
    if opts["thermal"]:
        lines += ["    wallHeatFlux", "    {", "        type            wallHeatFlux;",
                  "        libs            (fieldFunctionObjects);",
                  f"        patches         ({body_patch});",
                  "        writeControl    writeTime;", "        log             yes;",
                  "    }", "",
                  "    wallHeatFluxMean", "    {",
                  "        type            surfaceFieldValue;",
                  "        libs            (fieldFunctionObjects);",
                  f"        regionType      patch;", f"        name            {body_patch};",
                  "        operation       areaAverage;",
                  "        fields          (wallHeatFlux);"]
        if opts["study"] in TRANSIENT:
            # In time the coefficient oscillates with the shedding, and the answer
            # is its mean over whole periods -- so it has to be sampled densely
            # enough to average, not once per write.
            lines += ["        writeControl    timeStep;",
                      "        writeInterval   5;"]
        else:
            lines += ["        writeControl    writeTime;"]
        lines += ["        log             yes;",
                  "        writeFields     no;", "    }", "",
                  # Every heat-transfer coefficient worth comparing with a
                  # correlation is defined on the LOG-MEAN temperature difference,
                  # which needs the outlet bulk temperature as well as the inlet
                  # and the wall. Using (T_wall - T_inlet) instead is not a small
                  # error -- on this bundle it is 10% of the answer, in the
                  # pessimistic direction, and nothing about the number looks wrong.
                  "    outletTemperature", "    {",
                  "        type            surfaceFieldValue;",
                  "        libs            (fieldFunctionObjects);",
                  "        regionType      patch;", "        name            outlet;",
                  "        operation       areaAverage;",
                  "        fields          (T);",
                  "        writeControl    writeTime;", "        log             yes;",
                  "        writeFields     no;", "    }", ""]
    lines += ["}"]
    return "\n".join(lines)


def control_dict(opts, flow, body_patch: str) -> str:
    study = opts["study"]
    solver = SOLVERS[study]
    if study in TRANSIENT:
        end, write = opts["end_time"], opts["end_time"] / opts["writes"]
        control, delta = "adjustableRunTime", opts.get("delta_t") or 1e-5
        extra = ["adjustTimeStep  yes;", f"maxCo           {opts['courant']:g};"]
    else:
        end, write = float(opts["iterations"]), max(1.0, opts["iterations"] / opts["writes"])
        control, delta = "timeStep", 1.0
        extra = []
    lines = [f"application     {solver};", "", "startFrom       latestTime;",
             "startTime       0;", "stopAt          endTime;",
             f"endTime         {end:g};", f"deltaT          {delta:g};", "",
             f"writeControl    {control};", f"writeInterval   {write:g};",
             "purgeWrite      0;", "writeFormat     binary;",
             "writePrecision  8;", "writeCompression off;",
             "timeFormat      general;", "timePrecision   6;",
             "runTimeModifiable true;"]
    lines += extra
    body = "\n".join(lines) + "\n\n" + function_objects(opts, flow, body_patch) + "\n"
    return case_gen.foam_file("dictionary", "controlDict", body, "system")


SOLVERS = {"mesh": "", "steady": "simpleFoam", "transient": "pimpleFoam",
           "thermal": "buoyantSimpleFoam",
           "thermal-transient": "buoyantPimpleFoam"}

THERMAL = ("thermal", "thermal-transient")
TRANSIENT = ("transient", "thermal-transient")
"""Steady RANS suppresses the very thing that carries heat off a bluff body -- the
unsteady wake -- so a bank of cylinders comes out 15-30% low however good the mesh
is. `thermal-transient` is the way out: run it in time and average the coefficient
over several shedding periods."""


# -- schemes and solution ----------------------------------------------------------

def fv_schemes(opts) -> str:
    steady = opts["study"] not in TRANSIENT
    ddt = "steadyState" if steady else "Euler"
    # Second order upwind on the divergence terms: snappy meshes are not orthogonal
    # enough for pure linear, and first-order upwind smears exactly the shear layer
    # the drag depends on.
    div = ["div(phi,U)      bounded Gauss linearUpwind grad(U);" if steady
           else "div(phi,U)      Gauss linearUpwind grad(U);"]
    if opts["thermal"]:
        div = ["div(phi,U)      bounded Gauss linearUpwind grad(U);",
               "div(phi,h)      bounded Gauss limitedLinear 1;",
               "div(phi,K)      bounded Gauss limitedLinear 1;",
               "div(phi,e)      bounded Gauss limitedLinear 1;",
               "div(phi,Ekp)    bounded Gauss limitedLinear 1;"]
    div += ["div(phi,k)      bounded Gauss limitedLinear 1;",
            "div(phi,omega)  bounded Gauss limitedLinear 1;",
            "div(phi,epsilon) bounded Gauss limitedLinear 1;",
            "div(phi,nuTilda) bounded Gauss limitedLinear 1;",
            # The transition model's two transported fields. A field written into 0/
            # with no scheme for its convection term is not caught until the solver
            # reads fvSchemes and stops -- after the mesh is built and decomposed.
            "div(phi,gammaInt) bounded Gauss limitedLinear 1;",
            "div(phi,ReThetat) bounded Gauss limitedLinear 1;"]
    if opts["thermal"]:
        div.append("div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;")
    else:
        div.append("div((nuEff*dev2(T(grad(U))))) Gauss linear;")
    lines = ["ddtSchemes", "{", f"    default         {ddt};", "}", "",
             "gradSchemes", "{", "    default         Gauss linear;",
             "    limited         cellLimited Gauss linear 1;",
             "    grad(U)         $limited;", "    grad(k)         $limited;",
             "    grad(omega)     $limited;", "}", "", "divSchemes", "{",
             "    default         none;"]
    lines += [f"    {d}" for d in div]
    lines += ["}", "", "laplacianSchemes", "{",
              "    default         Gauss linear limited corrected 0.33;", "}", "",
              "interpolationSchemes", "{", "    default         linear;", "}", "",
              "snGradSchemes", "{", "    default         limited corrected 0.33;", "}", "",
              "wallDist", "{", "    method          meshWave;", "}"]
    return case_gen.foam_file("dictionary", "fvSchemes", "\n".join(lines), "system")


def fv_solution(opts, model: str) -> str:
    thermal = opts["thermal"]
    transient = opts["study"] in TRANSIENT
    turb = [f for f in case_gen.turbulence_fields(model)]
    p_name = "p_rgh" if thermal else "p"
    lines = ["solvers", "{"]
    if thermal and transient:
        # buoyantPimpleFoam carries a density equation that the steady solver does
        # not, and without an entry for it the run dies on its first step with
        # "Entry 'rho' not found in dictionary system/fvSolution/solvers".
        lines += ['    "rho.*"', "    {", "        solver          diagonal;",
                  "    }", ""]
    lines += [f"    {p_name}", "    {",
             "        solver          GAMG;", "        tolerance       1e-7;",
             "        relTol          0.01;", "        smoother        GaussSeidel;", "    }",
             ""]
    if transient:
        lines += [f"    {p_name}Final", "    {",
                  "        solver          GAMG;", "        tolerance       1e-7;",
                  "        relTol          0;", "        smoother        GaussSeidel;",
                  "    }", ""]
    fields = ["U"] + turb + (["h", "e"] if thermal else [])
    lines += [f'    "({"|".join(fields)})"', "    {",
              "        solver          smoothSolver;",
              "        smoother        symGaussSeidel;",
              "        tolerance       1e-8;", "        relTol          0.1;", "    }", ""]
    if transient:
        lines += [f'    "({"|".join(fields)})Final"', "    {",
                  "        solver          smoothSolver;",
                  "        smoother        symGaussSeidel;",
                  "        tolerance       1e-8;", "        relTol          0;", "    }", ""]
    lines += ["}", ""]

    if transient:
        lines += ["PIMPLE", "{", "    nOuterCorrectors 2;", "    nCorrectors     2;",
                  "    nNonOrthogonalCorrectors 1;", "    consistent      no;"]
        if thermal:
            lines += ["    pRefCell        0;",
                      f"    pRefValue       {opts['pressure']:g};",
                      "    rhoMin          0.2;", "    rhoMax          2.0;"]
        lines += ["}", ""]
    else:
        # SIMPLEC (consistent yes) carries a much higher pressure factor and
        # converges faster on an incompressible case. The compressible solver is
        # the exception: buoyantSimpleFoam is well behaved with plain SIMPLE and a
        # 0.3 pressure factor, and pairing SIMPLEC with 0.3 -- conservative on both
        # counts -- converges so slowly that a fixed iteration budget runs out
        # before the wall heat flux has settled, which is the whole answer here.
        lines += ["SIMPLE", "{", "    nNonOrthogonalCorrectors 1;",
                  f"    consistent      {'no' if thermal else 'yes'};"]
        if thermal:
            lines += ["    pRefCell        0;",
                      f"    pRefValue       {opts['pressure']:g};",
                      "    rhoMin          0.2;", "    rhoMax          2.0;"]
        lines += ["", "    residualControl", "    {"]
        controls = [(p_name, 1e-4), ("U", 1e-5)] + [(f, 1e-5) for f in turb]
        if thermal:
            controls.append(("h", 1e-5))
        for name, value in controls:
            lines.append(f"        {name:<12s} {value:g};")
        lines += ["    }", "}", ""]

    lines += ["relaxationFactors", "{", "    fields", "    {"]
    if transient:
        lines += ["        \".*\"            1;"]
    else:
        lines += [f"        {p_name}           {0.3 if thermal else 0.7:g};"]
    lines += ["    }", "    equations", "    {"]
    if transient:
        lines += ['        ".*"            1;']
    else:
        lines += [f'        U               {0.7 if thermal else 0.9:g};',
                  '        ".*"            0.7;']
        if thermal:
            lines += ['        h               0.7;']
    lines += ["    }", "}"]
    return case_gen.foam_file("dictionary", "fvSolution", "\n".join(lines), "system")


# -- the one command that meshes it ------------------------------------------------

def allmesh(opts, surfaces, body_patch: str) -> str:
    """blockMesh, features, snappy, checkMesh and a report, in one call.

    Each of those is a round trip to whoever is driving the case, and a study that
    spends fifty of them on setup has spent more wall clock on dictionaries than on
    the solve. One script, one log per stage, one digest at the end."""
    stages = ["blockMesh", "surfaceFeatureExtract", "snappyHexMesh -overwrite"]
    if opts["mrf"]:
        stages.append("topoSet")
    lines = ["#!/bin/sh",
             "# Written by snappy_gen.py. Edit freely -- nothing reads it back.",
             "set -e",
             'cd "$(dirname "$0")"', "",
             "run() {",
             '    stage=$(echo "$1" | cut -d" " -f1)',
             '    printf "%-24s" "$stage ..."',
             '    if $1 > "log.$stage" 2>&1; then',
             '        echo " ok"',
             "    else",
             '        echo " FAILED -- last 20 lines of log.$stage:"',
             '        tail -20 "log.$stage"',
             "        exit 1",
             "    fi",
             "}", ""]
    for stage in stages:
        lines.append(f'run "{stage}"')
    # checkMesh is a diagnostic, not a build step. It exits non-zero on findings
    # that are routine on a layered snappy mesh -- concave cells where a layer
    # meets a curved surface, a handful of low-weight faces -- and treating that
    # as a build failure throws away a mesh that is fine. Its verdict is printed
    # below; the judgement is the reader's.
    lines += ['printf "%-24s" "checkMesh ..."',
              'checkMesh -allGeometry -allTopology -constant > log.checkMesh 2>&1',
              'echo " done (findings below are for you to weigh, not errors)"']
    lines += ["", "echo",
              '# A surface that never intersected the background mesh refines nothing',
              '# and snappy still exits 0 saying "Finished meshing without any errors".',
              '# The mesh is then an empty box, and the solve on it looks fine.',
              'if grep -q "Cells per refinement level" log.snappyHexMesh; then',
              '    levels=$(sed -n "/Cells per refinement level/,/^$/p" log.snappyHexMesh '
              '| tail -n +2 | wc -l)',
              '    if [ "$levels" -le 1 ]; then',
              '        echo "!! snappyHexMesh refined NOTHING -- every cell is still at"',
              '        echo "!! level 0, so the surface never intersected the background"',
              '        echo "!! mesh and this is an empty box. Usually the background"',
              '        echo "!! cell is wider than the body: check the domain and"',
              '        echo "!! --cells-across-body before solving anything on it."',
              '        exit 2',
              '    fi',
              'fi',
              'echo "---- mesh ----"',
              'grep -E "^ *cells:" log.checkMesh | head -1',
              'grep -E "cells:|faces:|points:" log.snappyHexMesh | tail -3',
              'echo',
              'echo "---- layers: got of asked, and thickness percent ----"',
              # The table ROWS, not its headers. Layers that do not fit are dropped
              # in silence, so the count that matters is the achieved one -- a mesh
              # described by its request rather than its result is the specific way
              # a boundary layer goes missing.
              'grep -A40 "patch  *faces  *layers" log.snappyHexMesh 2>/dev/null'
              ' | grep -E "^[A-Za-z_][A-Za-z0-9_.]*  *[0-9]" | tail -6 || true',
              'echo', 'echo "---- checkMesh verdict ----"',
              'grep -E "\\*\\*\\*|Failed|failed|Mesh OK" log.checkMesh | head -20',
              'echo',
              'echo "next: run the solver, or first_look.py to see what you built"']
    return "\n".join(lines) + "\n"


# -- assembly ----------------------------------------------------------------------

def case_files(surfaces, domain: Domain, plan, flow, opts, model: str,
               layer: dict, point) -> dict[str, str]:
    body_patch = opts["_body_patch"]
    files = {
        "system/blockMeshDict": block_mesh_dict(domain, plan.roles),
        "system/surfaceFeatureExtractDict": surface_feature_dict(
            [s.name for s in surfaces], opts["feature_angle"]),
        "system/snappyHexMeshDict": snappy_dict(surfaces, opts, layer, point),
        "system/meshQualityDict": mesh_quality_dict(),
        "system/decomposeParDict": decompose_dict(opts["cores"]),
        "system/controlDict": control_dict(opts, flow, body_patch),
        "system/fvSchemes": fv_schemes(opts),
        "system/fvSolution": fv_solution(opts, model),
        "constant/turbulenceProperties": case_gen.turbulence_properties(model),
        "Allmesh": allmesh(opts, surfaces, body_patch),
    }
    if opts["thermal"]:
        files["constant/thermophysicalProperties"] = thermo_properties(opts, flow)
        files["constant/g"] = gravity_file()
        files["0/p"] = field_p_thermal(plan, opts, "p")
        files["0/p_rgh"] = field_p_thermal(plan, opts, "p_rgh")
        files["0/T"] = field_T(plan, opts)
        files["0/alphat"] = field_alphat(plan, opts)
    else:
        files["constant/transportProperties"] = case_gen.transport_properties(flow)
        files["0/p"] = case_gen.field_p(plan)
    if opts["mrf"]:
        files["constant/MRFProperties"] = mrf_properties(opts)
        files["system/topoSetDict"] = topo_set_dict(opts)

    files["0/U"] = case_gen.field_U(plan, flow)
    fields = case_gen.turbulence_fields(model)
    if fields:
        intensity = opts["turbulent_intensity"]
        if intensity is None:
            # 0.1% is a clean wind tunnel. It is the wrong number for anything the
            # flow is confined by -- a tube bank, a duct, a heat exchanger -- where
            # the turbulence arriving anywhere past the first row is generated by
            # the geometry itself. Using the tunnel value there starves the first
            # rows and the answer comes out low with nothing to point at.
            intensity = (case_gen.DUCT_INTENSITY if opts.get("far") == "symmetry"
                         else case_gen.FREE_STREAM_INTENSITY)
        mixing = opts.get("mixing_length") or 0.07 * plan.length
        ratio = None if opts.get("mixing_length") else (
            opts.get("viscosity_ratio") or case_gen.FREE_STREAM_VISCOSITY_RATIO)
        writers = {
            "k": lambda: case_gen.field_k(plan, flow, intensity),
            "omega": lambda: case_gen.field_omega(plan, flow, intensity, mixing,
                                                  nu=flow.nu, ratio=ratio),
            "epsilon": lambda: case_gen.field_epsilon(plan, flow, intensity, mixing),
            "nuTilda": lambda: case_gen.field_nu_tilda(plan, flow),
            "gammaInt": lambda: case_gen.field_gamma_int(plan),
            "ReThetat": lambda: case_gen.field_re_theta(plan, intensity),
        }
        for field in fields:
            files[f"0/{field}"] = writers[field]()
        files["0/nut"] = case_gen.field_nut(plan, model)
    return files


def reference_area(surface: Surface, domain: Domain, opts) -> tuple[float, str]:
    """The area the coefficients are divided by, and why it is that number."""
    cuts = domain.halvings
    factor = 0.5 ** cuts
    if opts.get("ref_area"):
        area = float(opts["ref_area"])
        why = "given with --ref-area"
        if cuts:
            area *= factor
            why = (f"{factor:g} of the {opts['ref_area']:.6g} m2 given with "
                   f"--ref-area, because the mesh cuts the body on {cuts} "
                   f"symmetry plane(s)")
        return area, why
    if opts["area"] == "wetted":
        area = surface.wetted_area()
        why = "wetted area, measured off the STL"
    else:
        area = surface.frontal_area(axis=0)
        why = "frontal area, rasterised off the STL"
    if cuts:
        area *= factor
        bisecting = ", ".join(p["name"] for p in domain.planes if p["bisects"])
        why += (f"; scaled by {factor:g}, because the mesh cuts the body on the "
                f"{bisecting} symmetry plane(s)")
    return area, why


def summary(surfaces, domain: Domain, flow, opts, model: str, why: str,
            layer: dict, notes: list[str]) -> list[str]:
    s = surfaces[0]
    ex = s.extent
    lines = [
        f"surface    {', '.join(x.name for x in surfaces)}  "
        f"{sum(len(x.triangles) for x in surfaces):,} triangles",
        f"           {ex[0]:.6g} x {ex[1]:.6g} x {ex[2]:.6g} m  "
        f"(wetted {s.wetted_area():.6g} m2, frontal {s.frontal_area(0):.6g} m2)",
        f"flow       U {flow.speed:.6g} m/s   nu {flow.nu:.6g} m2/s   "
        f"Re {flow.reynolds:,.0f} on {opts['_l_ref']:.6g} m",
        f"turbulence {model}   ({why})",
        f"domain     {domain.bounds[0]:.4g}..{domain.bounds[1]:.4g} x "
        f"{domain.bounds[2]:.4g}..{domain.bounds[3]:.4g} x "
        f"{domain.bounds[4]:.4g}..{domain.bounds[5]:.4g} m",
        f"           background {domain.cells[0]}x{domain.cells[1]}x{domain.cells[2]} "
        f"= {domain.cell_count:,} cells, {domain.cell_size[0]:.4g} m each",
        f"refinement level {opts['refine']} at the surface -> "
        f"{domain.cell_size[0] / 2 ** opts['refine']:.4g} m",
        f"           that is {max(ex) / (domain.cell_size[0] / 2 ** opts['refine']):.0f} "
        f"cells along the body and "
        f"{min(e for e in ex if e > 0) / (domain.cell_size[0] / 2 ** opts['refine']):.0f} "
        f"across its thinnest span -- the number to judge, not the level",
    ]
    if layer["layers"]:
        lines += [
            f"layers     {layer['layers']} at ratio {layer['ratio']:g}, first "
            f"{layer['first']:.4g} m -> y+ {opts['y_plus']:g} "
            f"(u_tau {layer['u_tau']:.4g} m/s)",
            f"           stack {layer['total']:.4g} m, which is "
            f"{layer['total'] / (domain.cell_size[0] / 2 ** opts['refine']) * 100:.0f}% "
            f"of the surface cell",
        ]
    else:
        lines.append("layers     none -- wall functions on the snapped surface alone")
    fine = domain.cell_size[0] / 2 ** opts["refine"]
    faces = s.wetted_area() / (fine * fine)
    if domain.symmetry != "none":
        faces *= 0.5
    lines += [
        f"size       ~{faces:,.0f} faces on the body at that spacing; external-flow "
        f"meshes",
        f"           land at roughly 5-15x that in total, so expect "
        f"{faces * 5 / 1e6:.1f}-{faces * 15 / 1e6:.1f} M cells",
        f"           plus {faces * layer['layers'] / 1e6:.1f} M in the layers. "
        f"cells_estimate.py is the closer look.",
        f"forces     Aref {opts['_a_ref']:.6g} m2, lRef {opts['_l_ref']:.6g} m",
        f"           {opts['_a_ref_why']}",
    ]
    floor = ittc57(flow.reynolds)
    lines.append(
        f"           sanity floor: flat-plate friction alone at this Re is "
        f"Cf = {floor:.5f} (ITTC-57)."
    )
    lines.append(
        "           a *total* drag coefficient below that, on the same area, is not "
        "a tight"
    )
    lines.append(
        "           result -- it is an impossible one. Check the area before the mesh."
    )
    if opts["mrf"]:
        lines.append(
            f"MRF        zone 'rotor', r {opts['mrf_radius']:.4g} m, "
            f"{opts['mrf_rpm']:g} rpm = {opts['mrf_rpm'] / 60:.6g} rev/s"
        )
        lines.append(
            "           n in C_T = T/(rho n^2 D^4) is rev/s, not rad/s -- the number "
            "above is the one to use."
        )
    if opts["thermal"]:
        lines.append(
            f"heat       inlet {opts['inlet_temperature']:g} K, wall "
            f"{opts['wall_temperature']:g} K, Pr {opts['prandtl']:g}, "
            f"solver {SOLVERS['thermal']}"
        )
    lines += [f"           {note}" for note in notes]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, nargs="?", help="Directory to write the case into.")
    ap.add_argument("--stl", action="append", default=[], required=False,
                    help="Surface to mesh around. Repeat for more than one.")
    ap.add_argument("--stl-scale", type=float, default=1.0, dest="stl_scale",
                    help="Multiply the surface by this (0.001 for a file in mm).")
    ap.add_argument("--rotate", default="",
                    help="Turn the surface before meshing: 'y:90' or 'y:90,z:-30', "
                         "degrees, right-handed, applied in order. The inlet is "
                         "always -x, so this is how a body that was not exported "
                         "facing the flow is pointed at it.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--force", action="store_true")

    flow = ap.add_argument_group("the flow")
    flow.add_argument("--speed", type=float, default=1.0)
    flow.add_argument("--nu", type=float, default=None,
                      help="Kinematic viscosity, m2/s (default 1.5e-5, air at 20 C).")
    flow.add_argument("--reynolds", type=float, default=None)
    flow.add_argument("--length", type=float, default=None,
                      help="Reference length for Re (default: the body's x extent).")
    flow.add_argument("--density", type=float, default=1.205,
                      help="rho, kg/m3, for forces and the thermo (default 1.205).")
    flow.add_argument("--turbulence", default="auto",
                      choices=["auto", "laminar", "kOmegaSST", "kOmegaSSTLM",
                               "kEpsilon", "SpalartAllmaras"],
                      help="kOmegaSSTLM adds the Langtry-Menter transition "
                           "equations. Below Re ~ 5e5 a blade carries a laminar "
                           "separation bubble over much of its chord, and a fully "
                           "turbulent model answers a different question -- the lift "
                           "comes out low however fine the mesh.")
    flow.add_argument("--turbulent-intensity", type=float, default=None,
                      dest="turbulent_intensity")
    flow.add_argument("--viscosity-ratio", type=float, default=None, dest="viscosity_ratio")
    flow.add_argument("--mixing-length", type=float, default=None, dest="mixing_length")

    run = ap.add_argument_group("the run")
    run.add_argument("--study", default="steady",
                     choices=["mesh", "steady", "transient", "thermal",
                              "thermal-transient"],
                     help="thermal-transient is buoyantPimpleFoam: the way to get a "
                          "heat-transfer coefficient off a bluff body, where steady "
                          "RANS suppresses the wake that does the mixing and comes "
                          "out 15-30%% low however good the mesh is.")
    run.add_argument("--iterations", type=int, default=1000)
    run.add_argument("--end-time", type=float, default=1.0, dest="end_time")
    run.add_argument("--delta-t", type=float, default=None, dest="delta_t")
    run.add_argument("--courant", type=float, default=5.0)
    run.add_argument("--writes", type=int, default=10)
    run.add_argument("--cores", type=int, default=4)

    box = ap.add_argument_group("the domain, in body lengths")
    box.add_argument("--ahead", type=float, default=2.0)
    box.add_argument("--behind", type=float, default=5.0)
    box.add_argument("--side", type=float, default=2.0)
    box.add_argument("--above", type=float, default=2.0)
    box.add_argument("--below", type=float, default=2.0)
    box.add_argument("--symmetry", default="none",
                     help="Cut the domain on one or more planes, comma separated: "
                          "'y' (the body's own mid-plane), 'z:max' or 'z:min' (its "
                          "own extreme -- a waterline closing a hull), or "
                          "'z:0.125' (a coordinate). A plane that bisects the body "
                          "halves the reference area with it; one that only bounds "
                          "the flow does not.")
    box.add_argument("--far", default="slip", choices=["slip", "symmetry"],
                     help="Treatment of the far boundaries (default slip).")
    box.add_argument("--ground", action="store_true",
                     help="Make the floor a wall rather than free stream.")
    box.add_argument("--moving-ground", action="store_true", dest="moving_ground")

    mesh = ap.add_argument_group("the mesh")
    mesh.add_argument("--cells-across-body", type=int, default=2,
                      dest="cells_across_body",
                      help="Background cells across the body's cross-section "
                           "(default 2). The floor under --background-cells.")
    mesh.add_argument("--background-cells", type=int, default=40,
                      dest="background_cells",
                      help="Cells across the domain's longest side (default 40).")
    mesh.add_argument("--base-cell", type=float, default=None, dest="base_cell",
                      help="Background cell size in metres; overrides the count.")
    mesh.add_argument("--refine", type=int, default=3,
                      help="snappy refinement level at the surface (default 3).")
    mesh.add_argument("--surface-cell", type=float, default=None, dest="surface_cell",
                      help="Wanted cell size ON the body, in metres. The refinement "
                           "level follows from it, which is the way round you "
                           "usually want: resolution is the thing you can judge, "
                           "the level is an implementation detail.")
    mesh.add_argument("--max-cells", type=int, default=8_000_000, dest="max_cells")
    mesh.add_argument("--feature-angle", type=float, default=150.0, dest="feature_angle")
    mesh.add_argument("--layers", type=int, default=5)
    mesh.add_argument("--layer-ratio", type=float, default=1.2, dest="layer_ratio")
    mesh.add_argument("--y-plus", type=float, default=50.0, dest="y_plus",
                      help="Target y+ for the first cell (default 50: wall functions).")
    mesh.add_argument("--medial-ratio", type=float, default=0.6, dest="medial_ratio",
                      help="snappy's maxThicknessToMedialRatio (default 0.6). The "
                           "tutorial value of 0.3 assumes a chunky body and refuses "
                           "to build layers on a thin one; raise it further for a "
                           "hull or a blade, lower it if layers are self-intersecting.")
    mesh.add_argument("--medial-angle", type=float, default=90.0, dest="medial_angle",
                      help="snappy's minMedialAxisAngle (default 90).")
    mesh.add_argument("--layer-iter", type=int, default=50, dest="layer_iter",
                      help="snappy's nLayerIter (default 50). Layer addition is "
                           "iterative and gives up quietly; more iterations is the "
                           "cheapest thing to try when the achieved count is short.")
    mesh.add_argument("--wall-speed", type=float, default=None, dest="wall_speed",
                      help="The velocity that sets the wall shear, if it is not the "
                           "free stream. On a rotor it is the blade speed -- a "
                           "propeller tip at 4000 rpm sees 50 m/s while the tunnel "
                           "sees 7, and sizing the layer on 7 puts y+ out by an "
                           "order of magnitude.")
    mesh.add_argument("--wall-length", type=float, default=None, dest="wall_length",
                      help="The length that goes with --wall-speed (a blade chord, "
                           "not the whole body). Defaults to the reference length.")

    forces = ap.add_argument_group("the forces")
    forces.add_argument("--area", default="frontal", choices=["frontal", "wetted"],
                        help="Which measured area the coefficients use (default frontal).")
    forces.add_argument("--ref-area", type=float, default=None, dest="ref_area",
                        help="Reference area of the WHOLE body; --symmetry halves it.")

    rot = ap.add_argument_group("a rotating zone (MRF)")
    rot.add_argument("--mrf", action="store_true")
    rot.add_argument("--mrf-rpm", type=float, default=0.0, dest="mrf_rpm")
    rot.add_argument("--mrf-axis", default="x", choices=["x", "y", "z"], dest="mrf_axis")
    rot.add_argument("--mrf-radius", type=float, default=None, dest="mrf_radius")
    rot.add_argument("--mrf-thickness", type=float, default=None, dest="mrf_thickness")

    heat = ap.add_argument_group("heat")
    heat.add_argument("--wall-temperature", type=float, default=333.15,
                      dest="wall_temperature")
    heat.add_argument("--inlet-temperature", type=float, default=293.15,
                      dest="inlet_temperature")
    heat.add_argument("--prandtl", type=float, default=0.71)
    heat.add_argument("--turbulent-prandtl", type=float, default=0.85,
                      dest="turbulent_prandtl")
    heat.add_argument("--cp", type=float, default=1005.0)
    heat.add_argument("--pressure", type=float, default=101325.0)

    args = ap.parse_args(argv)
    if not args.stl or args.case is None:
        ap.error("a --stl and a directory to write the case into")

    opts = vars(args)
    opts["thermal"] = args.study in THERMAL
    if opts["nu"] is None and opts["reynolds"] is None:
        opts["nu"] = 1.5e-5

    surfaces = []
    notes = []
    if opts["nu"]:
        notes += fluid_notes(float(opts["nu"]), float(opts["density"]))
    for item in args.stl:
        path = Path(item)
        if not path.exists():
            raise SystemExit(f"no such surface: {path}")
        surface = read_stl(path)
        notes += scale_notes(surface, args.stl_scale)
        surfaces.append(rotated(rescaled(surface, args.stl_scale),
                                parse_rotation(args.rotate)))
    primary = surfaces[0]
    if args.rotate:
        notes.append(f"the surface was turned {args.rotate} before meshing")
    notes += axis_notes(primary, args.mrf_axis if args.mrf else None)

    opts["_l_ref"] = args.length or primary.extent[0]
    flow_state = case_gen.derive_flow(opts, opts["_l_ref"])
    model, why = case_gen.turbulence_model(opts, flow_state)

    domain, domain_notes = build_domain(primary, opts)
    notes += domain_notes
    body_patch = primary.name
    opts["_body_patch"] = body_patch
    opts["_a_ref"], opts["_a_ref_why"] = reference_area(primary, domain, opts)

    if args.surface_cell:
        base = domain.cell_size[0]
        level = max(0, int(math.ceil(math.log2(base / args.surface_cell))))
        notes.append(
            f"--surface-cell {args.surface_cell:.4g} m on a {base:.4g} m background "
            f"cell needs refinement level {level}; using it instead of --refine "
            f"{args.refine}."
        )
        opts["refine"] = args.refine = level

    wall_flow, wall_length = flow_state, opts["_l_ref"]
    if args.wall_speed or args.wall_length:
        wall_length = args.wall_length or opts["_l_ref"]
        speed = args.wall_speed or flow_state.speed
        wall_flow = case_gen.Flow(speed, wall_length, flow_state.nu,
                                  speed * wall_length / flow_state.nu, "reynolds")
        notes.append(
            f"the layer is sized on {speed:g} m/s over {wall_length:.4g} m "
            f"(Re {wall_flow.reynolds:,.0f}), not on the free stream -- that is what "
            f"sets the wall shear here."
        )
    notes += y_plus_notes(args.y_plus, args.layers)
    first, u_tau = first_layer_thickness(args.y_plus, wall_flow, wall_length)
    layer = {"layers": args.layers, "ratio": args.layer_ratio, "first": first,
             "u_tau": u_tau,
             "total": stack_thickness(first, args.layers, args.layer_ratio)}
    surface_cell = domain.cell_size[0] / 2 ** args.refine
    if layer["layers"] and layer["total"] > 0.5 * surface_cell:
        notes.append(
            f"!! the layer stack ({layer['total']:.4g} m) is more than half the surface "
            f"cell ({surface_cell:.4g} m); snappy will drop layers. Refine less, or "
            f"use fewer layers, or accept a higher y+."
        )

    if args.mrf:
        axis = AXES[args.mrf_axis]
        centre = primary.centre
        # The zone has to CONTAIN the rotor. A cylinder sized off the disc radius
        # but only a quarter of it thick is thinner than the blades are long in the
        # axial direction, so the tips sit outside the rotating frame, feel no
        # rotation, and the thrust comes out wrong in a way nothing reports.
        span = max(primary.extent[i] for i in (0, 1, 2) if i != axis)
        radius = args.mrf_radius or 0.6 * span
        thickness = args.mrf_thickness or 1.5 * primary.extent[axis]
        if thickness < primary.extent[axis]:
            notes.append(
                f"!! the MRF zone is {thickness:.4g} m thick along {args.mrf_axis} but "
                f"the rotor spans {primary.extent[axis]:.4g} m there; part of it would "
                f"sit outside the rotating frame. --mrf-thickness."
            )
        if 2 * radius < span:
            notes.append(
                f"!! the MRF zone is {2 * radius:.4g} m across but the rotor spans "
                f"{span:.4g} m; --mrf-radius."
            )
        half = thickness / 2.0
        p1, p2 = list(centre), list(centre)
        p1[axis] -= half
        p2[axis] += half
        opts["mrf_p1"], opts["mrf_p2"], opts["mrf_radius"] = tuple(p1), tuple(p2), radius

    plan = case_gen.Plan(PatchList(list(domain.patches) + [body_patch]),
                         build_roles(domain, body_patch, opts),
                         opts["_l_ref"], {}, notes)
    files = case_files(surfaces, domain, plan, flow_state, opts, model, layer,
                       inside_point(domain, primary))

    for line in summary(surfaces, domain, flow_state, opts, model, why, layer, notes):
        print(line)
    print()

    if args.dry_run:
        print(f"would write {len(files)} files into {args.case}:")
        for relative in sorted(files):
            print(f"  {relative}   ({len(files[relative].splitlines())} lines)")
        return 0

    target = Path(args.case)
    if (target / "system" / "snappyHexMeshDict").exists() and not args.force:
        raise SystemExit(f"{target} already holds a case; --force to write over it")
    written = case_gen.write_case(target, files)
    tri = target / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    for s in surfaces:
        target_stl = tri / f"{s.name}.stl"
        if args.rotate or args.stl_scale != 1.0:
            # Write what was actually meshed. Copying the original file here would
            # hand snappy a surface in a different place from the domain built
            # around it, which it reports as an empty mesh.
            write_binary_stl(target_stl, s.triangles)
        else:
            target_stl.write_bytes(s.path.read_bytes())
        written.append(target_stl)
    (target / "Allmesh").chmod(0o755)
    print(f"wrote {len(written)} files into {target}")
    print(f"next: sh {target}/Allmesh     "
          f"(blockMesh, features, snappy, checkMesh -- one call)")

    home = target.parent
    if home.name in ("work", "") or home == home.parent:
        home = target
    try:
        study_state.record("other", target, root=home, case=target.name,
                           label=f"snappy case around {body_patch}, "
                                 f"{domain.cell_count:,} background cells",
                           template="surface", study=args.study,
                           reynolds=flow_state.reynolds, nu=flow_state.nu)
        study_state.set_phase("geometry", "done", root=home, case=target.name,
                              note=f"surface {body_patch}, refine {args.refine}")
    except OSError as exc:
        print(f"(the study manifest could not be updated: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
