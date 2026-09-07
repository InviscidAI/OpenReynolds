#!/usr/bin/env python3
r"""The case files for a mesh that already exists: 0/, system/ and constant/.

This used to build the mesh too, from a template name -- a dozen parameterised
shapes and a blockMeshDict writer. That half is gone, and with it the habit it
encouraged of adding one more shape every time a new case turned up. A mesh now
arrives from wherever it came from (the `mesh` tool, blockMesh by hand,
snappyHexMesh, cfMesh, a converted .msh) and this dresses it.

What it writes is the dull, error-prone part: a `boundaryField` entry for every
patch in the mesh, in every field the turbulence model needs; the schemes and the
solution controls for a steady, transient or mesh-only study; the viscosity that
makes the Reynolds number you asked for. The mistakes it exists to stop all cost a
solver run to find -- a patch called `inlet` in the mesh and `Inlet` in the fields;
a `frontAndBack` declared `empty` in the mesh and `zeroGradient` in 0/U, which
stops the solver on the first time step; a viscosity typed in for a Reynolds number
somebody meant to run last week.

what it reads off the mesh, rather than being told

    The patch names and types come out of `constant/polyMesh/boundary`. The cell
    count, the bounding box and the smallest cell come from `checkMesh` -- OpenFOAM's
    own answer, not arithmetic here. Each patch's area, centre and mean normal come
    from `mesh_look.py`. From those: the inlet's direction (its own normal, pointed
    at the middle of the domain -- which is why a U-duct with both ends on the left
    comes out right, where a bounding-box rule gets it exactly backwards); the
    characteristic length (the inlet's hydraulic diameter, 2w for a plane passage);
    and whether this is a body in open flow or a passage, which sets the free-stream
    turbulence -- a wall patch that is not part of the domain's own bounding box is a
    body in the flow.

what it derives, and says it derived

    Give it a speed, a Reynolds number and a length and it solves nu = U*L/Re, writes
    that into transportProperties, and prints the three numbers. Give it --nu instead
    and it prints the Reynolds number that implies. The one that is not stated is the
    one that gets misremembered.

roles

    Every patch gets one: inlet (with a direction), outlet, wall, slip, symmetry,
    belt (a ground that moves with the stream), spinning (a wheel), or empty. What
    you name on the command line wins; then the mesh's own types (`empty` and the
    symmetry constraints are what the mesh says they are); then the name; and a name
    that says nothing is a wall, which is the safe reading.

    python3 case_gen.py /work/study/mesh --study steady --speed 10 --reynolds 5e5
    python3 case_gen.py /work/study/duct --inlet inlet --outlet outlet,outlet2 --length 0.05
    python3 case_gen.py /work/study/car --external --body car --belt road --spin wheel:0.15:z
    python3 case_gen.py /work/study/mesh --study transient --end-time 2 --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state

TOLERANCE = 1e-9
"""Coordinates closer than this are the same point. blockMesh matches faces by
vertex index, not by position, so two points that ought to be one and are not
produce a mesh with a crack down it that checkMesh reports as an open cell."""


def patch_order(patches) -> list[str]:
    """Inlets first, then outlets, then walls, then the empty one.

    Only cosmetic -- but a boundary list you can read top to bottom is a boundary
    list whose omissions you notice.
    """
    rank = {"inlet": 0, "outlet": 1, "walls": 3, "frontAndBack": 9}

    def key(name: str) -> tuple[int, str]:
        if name in rank:
            return (rank[name], name)
        if name.startswith("inlet"):
            return (0, name)
        if name.startswith("outlet"):
            return (1, name)
        if name == "frontAndBack":
            return (9, name)
        return (2, name)

    return sorted(patches, key=key)


# -- OpenFOAM file furniture -------------------------------------------------------

BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Website:  www.openfoam.com                      |
|   \\  /    A nd           | Written:  openreynolds case_gen.py              |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/"""

FOAM_FOOTER = "// ************************************************************************* //"

RULE = "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"


def foam_header(cls: str, obj: str, location: str = "") -> str:
    """The FoamFile block every OpenFOAM dictionary needs.

    `location` matters more than it looks: a field file without it still reads,
    but several utilities report the wrong path in their errors when it is
    missing, which turns a one-line typo into a hunt.
    """
    lines = [BANNER, "FoamFile", "{", "    version     2.0;", "    format      ascii;",
             f"    class       {cls};"]
    if location:
        lines.append(f'    location    "{location}";')
    lines.append(f"    object      {obj};")
    lines.append("}")
    lines.append(RULE)
    lines.append("")
    return "\n".join(lines)


def foam_file(cls: str, obj: str, body: str, location: str = "") -> str:
    return foam_header(cls, obj, location) + "\n" + body.rstrip("\n") + "\n\n" + FOAM_FOOTER + "\n"


def vector(value) -> str:
    # `+ 0.0` turns -0.0 into 0.0: a direction measured off a patch normal produces
    # negative zeros, and `uniform (-0 -5 -0)` reads like a bug to whoever opens the
    # file even though OpenFOAM is perfectly happy with it.
    return "(" + " ".join(f"{float(v) + 0.0:g}" for v in value) + ")"


# -- the mesh that is already there ------------------------------------------------
#
# This file used to build the mesh as well, from a template name and a few numbers:
# a dozen parameterised shapes, an O-grid tiler, a blockMeshDict writer. All of it is
# gone. A mesh now arrives from the mesh desk (or from anywhere else: blockMesh by
# hand, snappyHexMesh, cfMesh, a converted .msh), and what this script does is dress
# it -- the 0/ directory, the system dictionaries and constant/, sized to the mesh in
# front of it. The division is the one that stopped the shapes multiplying: one thing
# makes geometry, one thing makes physics, and neither has an opinion about the other.


class MeshFacts:
    """What `constant/polyMesh` is, read off the mesh rather than assumed.

    Every number here has a source. The patch names and types come out of
    `constant/polyMesh/boundary` as text; the cell count, the bounding box and the
    smallest cell come out of `checkMesh`, which is OpenFOAM's own answer rather than
    this script's arithmetic; the areas, centres and normals come from
    `mesh_look.py`, which reads the faces. Where a number could not be measured it is
    zero and the caller says so, because a case sized off a guessed length is the kind
    of wrong that runs.
    """

    def __init__(self, patches: list[dict], bounds: list[float], cells: int,
                 min_volume: float, two_d: bool, enclosure: set[str] | None = None,
                 checkmesh: str = ""):
        self.patches = list(patches)
        self.bounds = list(bounds)
        self.cells = int(cells)
        self.min_volume = float(min_volume or 0.0)
        self.two_d = bool(two_d)
        self.enclosure = set(enclosure or ())
        self.checkmesh = checkmesh

    # -- what the writers ask for ---------------------------------------------

    @property
    def patch_faces(self) -> dict:
        return {p["name"]: int(p.get("nFaces") or 0) for p in self.patches}

    def patch_face_counts(self) -> dict:
        return self.patch_faces

    def patch(self, name: str) -> dict:
        for entry in self.patches:
            if entry.get("name") == name:
                return entry
        return {}

    @property
    def thin_axis(self) -> int:
        """Which axis a plane case is one cell thick in, measured not assumed.

        `mesh_look.py` finds it by span ratio and so does this: nothing anywhere says
        a 2D case must be extruded in z, and reading `bounds[5] - bounds[2]` on a case
        extruded in y gives a real metre-scale extent where a cell width was meant,
        which then divides the hydraulic diameter, the cell size, the time step and
        the y+ estimate.
        """
        span = self.span
        widest = max(span) if span else 0.0
        if widest <= 0:
            return 2
        thinnest = min(range(3), key=lambda i: span[i])
        return thinnest if span[thinnest] <= 0.05 * widest else 2

    @property
    def thickness(self) -> float:
        """The extent across the thin direction -- the span a 2D case is per."""
        span = self.span
        return span[self.thin_axis] if span else 0.0

    @property
    def span(self) -> tuple[float, float, float]:
        if len(self.bounds) != 6:
            return (0.0, 0.0, 0.0)
        return (self.bounds[3] - self.bounds[0], self.bounds[4] - self.bounds[1],
                self.bounds[5] - self.bounds[2])

    @property
    def centre(self) -> tuple[float, float, float]:
        if len(self.bounds) != 6:
            return (0.0, 0.0, 0.0)
        return tuple((self.bounds[i] + self.bounds[i + 3]) / 2.0 for i in range(3))

    def _cell_from(self, volume: float) -> float:
        if volume <= 0:
            return 0.0
        if self.two_d and self.thickness > 0:
            return math.sqrt(volume / self.thickness)
        return volume ** (1.0 / 3.0)

    @property
    def smallest_cell(self) -> float:
        """The edge of the smallest cell, from checkMesh's minimum cell volume.

        A 2D case is one cell thick, so the in-plane size is the volume over the
        thickness, square-rooted; a 3D one is the cube root. It is the *smallest*
        rather than the average because what it is for -- the y+ estimate and the
        time step a Courant number implies -- are both set by the smallest cell, and
        an average on a graded mesh is optimistic by whatever the grading is.

        Where checkMesh could not be run, the average cell stands in and
        `cell_is_estimate` says so, because a time step from an average cell is
        optimistic and a time step of zero is a broken dictionary.
        """
        measured = self._cell_from(self.min_volume)
        return measured if measured > 0 else self.average_cell

    @property
    def average_cell(self) -> float:
        """The domain's volume over its cell count, as a length."""
        span = self.span
        volume = span[0] * span[1] * span[2]
        if volume <= 0 or self.cells <= 0:
            return 0.0
        return self._cell_from(volume / self.cells)

    @property
    def cell_is_estimate(self) -> bool:
        return self._cell_from(self.min_volume) <= 0 and self.average_cell > 0

    def bodies(self) -> list[str]:
        """Wall patches that are things *in* the flow rather than the flow's boundary.

        Measured, not guessed from the name: a wall patch is a body when it is not
        part of the enclosure and reaches the end of at most one of the domain's axes
        (`mesh_look.box_contact`). A cylinder mid-channel reaches none; a square body
        on the floor reaches one; the wall of an L-duct runs into the ends and the
        sides and reaches two. It matters because a body in open flow carries a
        hundredth of the free-stream turbulence a passage does, and because the drag
        coefficient is only worth computing for the first kind.
        """
        out = []
        for entry in self.patches:
            if entry.get("type") != "wall" or entry["name"] in self.enclosure:
                continue
            touches = entry.get("touches")
            if touches is None or touches <= 1:
                out.append(entry["name"])
        return out


def read_mesh(case: Path) -> MeshFacts:
    """Measure the case's mesh with the toolbox's own instrument.

    `mesh_look.py` is the one place a mesh is read and measured, so this asks it
    rather than parsing polyMesh a second way. Its drawing half needs pyvista and its
    checkMesh half needs OpenFOAM; both are on the instance, and where either is
    missing what comes back is smaller rather than wrong.
    """
    import mesh_look

    payload = mesh_look.look(case, None, check=True)
    if not payload.get("polymesh"):
        raise SystemExit(
            f"there is no constant/polyMesh in {case}. This writes the case files for a "
            "mesh that already exists -- mesh it first (the `mesh` tool, or blockMesh, "
            "snappyHexMesh, cfMesh, gmshToFoam by hand)"
        )
    patches = payload.get("patches") or []
    return MeshFacts(
        patches=patches,
        bounds=payload.get("bounds") or [],
        cells=int(payload.get("cells") or 0),
        min_volume=float((payload.get("metrics") or {}).get("min_volume") or 0.0),
        two_d=bool(payload.get("two_d")),
        enclosure=set(payload.get("enclosure") or ()),
        checkmesh=str(payload.get("checkmesh") or ""),
    )


# -- what each patch is for ---------------------------------------------------------

INLET_NAMES = ("inlet", "in", "intake", "supply")
OUTLET_NAMES = ("outlet", "out", "exit", "outflow", "exhaust")

ROLE_FLAGS = (("inlet", "inlet"), ("outlet", "outlet"), ("wall", "wall"),
              ("slip", "slip"), ("symmetry", "symmetry"), ("belt", "belt"))
"""`--inlet`, `--outlet`, `--wall`, `--slip`, `--symmetry`, `--belt`, each taking
patch names. What is named wins over what is guessed, always."""


def named(opts, key: str) -> list[str]:
    value = opts.get(key) or []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value:
        out.extend(part.strip() for part in str(item).split(",") if part.strip())
    return out


def roles_for(mesh: MeshFacts, opts) -> dict:
    """Every patch's role: what it is named, else what its type says, else its name.

    The order matters and is the whole of the policy. A patch named on the command
    line is that thing. Otherwise `empty` and the symmetry types are what the mesh
    says they are -- those are constraints, and getting them wrong stops the solver
    rather than misleading it. Only then does the name decide, and a name that says
    nothing is a wall, which is the safe reading: a wall that should have been an
    outlet shows up in the first residual plot, an outlet that should have been a
    wall quietly drains the domain.
    """
    roles: dict = {}
    stated: dict = {}
    for flag, kind in ROLE_FLAGS:
        for name in named(opts, flag):
            stated[name] = kind
    for entry in named(opts, "spin"):
        parts = entry.split(":")
        if len(parts) < 2:
            raise SystemExit(f"--spin wants name:radius[:axis], not {entry!r}")
        stated[parts[0]] = "spinning"

    unknown = [n for n in stated if n not in mesh.patch_faces]
    if unknown:
        raise SystemExit(
            f"the mesh has no patch called {', '.join(sorted(unknown))}; it has "
            + ", ".join(sorted(mesh.patch_faces))
        )

    for entry in mesh.patches:
        name = entry["name"]
        kind = stated.get(name)
        if kind is None:
            if entry.get("type") == "empty":
                kind = "empty"
            elif str(entry.get("type", "")).startswith("symmetry"):
                kind = "symmetry"
            elif _looks_like(name, INLET_NAMES):
                kind = "inlet"
            elif _looks_like(name, OUTLET_NAMES):
                kind = "outlet"
            else:
                kind = "wall"
        role: dict = {"kind": kind}
        if kind == "inlet":
            role["direction"] = inlet_direction(entry, mesh, opts)
        if kind == "spinning":
            role.update(_spin(name, opts))
        roles[name] = role
    return roles


def _looks_like(name: str, words: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(lowered == w or lowered.startswith(w) for w in words)


def _spin(name: str, opts) -> dict:
    for entry in named(opts, "spin"):
        parts = entry.split(":")
        if parts[0] != name:
            continue
        radius = float(parts[1])
        axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[
            parts[2] if len(parts) > 2 else "z"]
        return {"radius": radius, "axis": axis, "origin": (0.0, 0.0, 0.0)}
    return {"radius": 1.0, "axis": (0.0, 0.0, 1.0), "origin": (0.0, 0.0, 0.0)}


def inlet_direction(entry: dict, mesh: MeshFacts, opts) -> tuple[float, float, float]:
    """Which way the flow enters, measured off the patch rather than assumed.

    `--direction` wins. Otherwise: the patch's own mean normal gives the axis it
    faces, and the sign is fixed by pointing it at the middle of the domain, which
    is inside by construction. That is why a U-duct works -- both of its ends are on
    the left, and the one that is the inlet still gets +x rather than whatever the
    bounding box would have said. A patch with no measurable normal (a curved inlet,
    or no pyvista to measure with) falls back to +x and the summary says so.
    """
    stated = opts.get("direction")
    if stated:
        axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        sign = -1.0 if str(stated).startswith("-") else 1.0
        axis = axes.get(str(stated).lstrip("+-").lower())
        if axis is None:
            raise SystemExit(f"--direction takes x, y, z, -x, -y or -z, not {stated!r}")
        return tuple(sign * c for c in axis)
    normal = entry.get("normal") or []
    centre = entry.get("center") or []
    if len(normal) == 3 and any(normal):
        length = math.sqrt(sum(c * c for c in normal))
        unit = tuple(c / length for c in normal) if length > TOLERANCE else None
        if unit:
            # Which side of the patch the fluid is on, asked of the mesh itself
            # (`mesh_look.inward_sign` steps off the face both ways and sees which
            # point lands in a cell). Where that could not be answered, the middle of
            # the domain stands in -- right for a straight duct and for a U-bend,
            # wrong for a passage concave enough that the middle is not in the fluid.
            sign = float(entry.get("inward") or 0)
            if not sign and len(centre) == 3:
                toward = tuple(mesh.centre[i] - centre[i] for i in range(3))
                sign = 1.0 if sum(unit[i] * toward[i] for i in range(3)) > 0 else -1.0
            if sign:
                return tuple(sign * c for c in unit)
    return (1.0, 0.0, 0.0)


# -- the case's own length ----------------------------------------------------------


def hydraulic_diameter(area: float, thickness: float, two_d: bool) -> float:
    """The length a passage's Reynolds number is actually defined on.

    For a 2D channel between parallel plates D_h = 4A/P -> 2w, and the width is the
    inlet's area over the one cell's thickness. In 3D, 4A/P for a circle is the
    diameter, so the equivalent circular diameter is the honest reading of an inlet
    whose perimeter this script cannot see. Passing the width instead of D_h is what
    once ran every duct at twice the Reynolds number it reported.
    """
    if area <= 0:
        return 0.0
    if two_d and thickness > 0:
        return 2.0 * (area / thickness)
    return math.sqrt(4.0 * area / math.pi)


def characteristic_length(mesh: MeshFacts, roles: dict, opts) -> tuple[float, str]:
    """(length, where it came from). `--length` wins; else the inlet's D_h."""
    stated = opts.get("length")
    if stated:
        return float(stated), "--length"
    for name, role in roles.items():
        if role.get("kind") != "inlet":
            continue
        area = float(mesh.patch(name).get("area") or 0.0)
        length = hydraulic_diameter(area, mesh.thickness, mesh.two_d)
        if length > 0:
            # In 3D this is the diameter of a circle of the same area, not 4A/P: the
            # perimeter of an arbitrary inlet is not something this script can see.
            # For a round pipe the two agree; for a 100 x 5 mm slot they differ by
            # 2.6x, and calling it the hydraulic diameter would put that error into
            # the Reynolds number under a name that says it is exact.
            what = ("the hydraulic diameter of" if mesh.two_d
                    else "the equivalent circular diameter of")
            return length, f"{what} {name}"
    raise SystemExit(
        "no characteristic length: this case has no inlet whose area could be "
        "measured, so give --length (the body's size across, or the passage's "
        "hydraulic diameter) -- the Reynolds number is meaningless without it"
    )


class Plan:
    """A mesh that exists, plus what its patches are for.

    `roles` is the whole of what the 0/ writer needs: a patch is an inlet with a
    direction, an outlet, a no-slip wall, a wall that translates, or a wall that
    spins. Keeping it here rather than deriving it from the patch names inside every
    writer means a renamed patch does not silently become a wall in one file and an
    inlet in another.
    """

    def __init__(self, mesh: MeshFacts, roles: dict, length: float, info: dict, notes=None):
        self.mesh = mesh
        self.roles = roles
        self.length = float(length)
        self.info = dict(info)
        self.notes = list(notes or [])

    @property
    def external(self) -> bool:
        """Whether this is a body in open flow rather than a passage."""
        return bool(self.info.get("external"))


def build_plan(case: Path, opts) -> Plan:
    """Read the mesh, give every patch a role, and size the case to it."""
    mesh = read_mesh(case)
    roles = roles_for(mesh, opts)
    length, source = characteristic_length(mesh, roles, opts)
    bodies = mesh.bodies()
    external = bool(opts.get("external")) or bool(bodies)
    notes = [f"length {length:.4g} m from {source}"]
    for name, role in roles.items():
        if role.get("kind") != "inlet":
            continue
        entry = mesh.patch(name)
        if opts.get("direction"):
            continue
        if not (entry.get("normal") and any(entry.get("normal"))):
            notes.append(f"{name}'s normal could not be measured, so the flow is set "
                         "along +x -- `--direction` says otherwise")
        elif not entry.get("inward"):
            notes.append(f"which side of {name} the fluid is on could not be probed, so "
                         "its direction was taken from the middle of the domain")
    if bodies:
        notes.append("body in the flow: " + ", ".join(bodies))
    if mesh.cell_is_estimate:
        notes.append(f"checkMesh reported no minimum cell volume, so the average cell "
                     f"({mesh.average_cell:.3g} m) stands in for the smallest one in the "
                     "y+ estimate and the Courant time step -- both are optimistic on a "
                     "graded mesh")
    elif not mesh.smallest_cell:
        notes.append("the smallest cell could not be measured, so the y+ estimate and "
                     "the Courant time step are left out")
    info = {"external": external, "bodies": bodies, "length_from": source}
    return Plan(mesh, roles, length, info, notes)


# -- the numbers, and which of them was derived ------------------------------------


class Flow:
    """Speed, length and viscosity, with the one that was calculated named.

    Three numbers with one relation between them, so two of them are given and the
    third is arithmetic. Which one that was is worth carrying around: a case whose
    viscosity was solved for a Reynolds number and a case whose Reynolds number came
    out of a viscosity are the same files and different intentions, and the printed
    line is the only place the difference survives.
    """

    def __init__(self, speed: float, length: float, nu: float, reynolds: float, derived: str):
        self.speed = float(speed)
        self.length = float(length)
        self.nu = float(nu)
        self.reynolds = float(reynolds)
        self.derived = derived

    def line(self) -> str:
        what = {"nu": "nu = U*L/Re", "reynolds": "Re = U*L/nu"}[self.derived]
        return (f"U = {self.speed:g} m/s, L = {self.length:g} m, nu = {self.nu:g} m2/s, "
                f"Re = {self.reynolds:.0f}   ({what})")


def derive_flow(opts, length: float) -> Flow:
    """Solve for whichever of nu and Re was not given."""
    speed = float(opts["speed"])
    length = float(opts.get("length") or length)
    if speed <= 0 or length <= 0:
        raise SystemExit("--speed and the characteristic length must both be positive")
    nu = opts.get("nu")
    reynolds = opts.get("reynolds")
    if nu is not None and reynolds is not None:
        raise SystemExit("give --nu or --reynolds, not both: the other one follows from it")
    if nu is not None:
        nu = float(nu)
        if nu <= 0:
            raise SystemExit("--nu must be positive")
        return Flow(speed, length, nu, speed * length / nu, "reynolds")
    reynolds = float(reynolds if reynolds is not None else 100.0)
    if reynolds <= 0:
        raise SystemExit("--reynolds must be positive")
    return Flow(speed, length, speed * length / reynolds, reynolds, "nu")


FREE_STREAM_INTENSITY = 0.001
"""Turbulence intensity outside a body in open flow: 0.1%, a wind tunnel's own.

Inside a passage it is percent-level and the default below says 5%. Using the duct
number for a body in open air is what once set a free-stream eddy viscosity of 9,000
times the molecular one, and the answer came back 41% high with a plausible
explanation attached."""

FREE_STREAM_VISCOSITY_RATIO = 10.0
"""nu_t/nu in the free stream of an external case; omega follows from it."""


def free_stream_intensity(plan, opts) -> float:
    """What was asked for, else 0.1% around a body and 5% inside a passage.

    Which of the two it is comes from the mesh -- a wall patch that is not part of
    the domain's own bounding box is a body in the flow -- not from a template name,
    because there are no templates any more."""
    asked = opts.get("turbulent_intensity")
    if asked is not None:
        return float(asked)
    return FREE_STREAM_INTENSITY if plan.external else 0.05


def viscosity_ratio(plan, opts) -> float | None:
    """The nu_t/nu the free stream is set from, or None to use the mixing length."""
    asked = opts.get("viscosity_ratio")
    if asked is not None:
        return float(asked)
    return FREE_STREAM_VISCOSITY_RATIO if plan.external else None


LAMINAR_BELOW = 2300.0
"""Where the automatic choice of turbulence model changes. Not a law -- transition
depends on the geometry and on what is upstream -- but a case run laminar at Re=10^6
and a case run kOmegaSST at Re=40 are both wrong in ways that take a run to notice,
and `--turbulence` overrides it."""


def turbulence_model(opts, flow: Flow) -> tuple[str, str]:
    """The model and the sentence saying why it is that one."""
    asked = (opts.get("turbulence") or "auto").strip()
    if asked != "auto":
        return asked, f"--turbulence {asked}"
    if flow.reynolds < LAMINAR_BELOW:
        return "laminar", f"Re = {flow.reynolds:.0f} is below {LAMINAR_BELOW:g}"
    return "kOmegaSST", f"Re = {flow.reynolds:.0f} is above {LAMINAR_BELOW:g}"


TURBULENCE_FIELDS = {
    "kOmegaSST": ("k", "omega"),
    "kOmegaSSTLM": ("k", "omega", "gammaInt", "ReThetat"),
    "kEpsilon": ("k", "epsilon"),
    "SpalartAllmaras": ("nuTilda",),
}
"""`kOmegaSSTLM` is kOmegaSST with the Langtry-Menter transition equations bolted on,
and it exists here because below Re ~ 5e5 a fully turbulent model is not merely less
accurate, it is answering a different question. A blade section at Re 6e4 carries a
laminar separation bubble over much of its chord; assume it turbulent from the leading
edge and the lift comes out low no matter how fine the mesh. It costs two extra
transported fields."""


TRANSITION_MODELS = ("kOmegaSSTLM",)


def free_stream_re_theta(intensity: float) -> float:
    """Inlet ReThetat from turbulence intensity, Langtry & Menter's own correlation.

    Tu is in PERCENT here, which is the convention the correlation is written in and
    a factor of 100 waiting to happen -- the 1/Tu^2 term makes getting it wrong
    spectacular rather than subtle."""
    tu = max(intensity * 100.0, 0.027)
    if tu <= 1.3:
        return 1173.51 - 589.428 * tu + 0.2196 / (tu * tu)
    return 331.50 * (tu - 0.5658) ** -0.671
"""The fields each model transports, under the names it looks them up by.

Not interchangeable and not optional. kEpsilon reads `epsilon` and never reads
`omega`; SpalartAllmaras reads neither and reads `nuTilda`. Writing the wrong pair
is not a run that converges badly -- the solver stops before the first iteration
saying it cannot find a file, and the same list has to drive 0/, the divSchemes and
the linear solvers or one of the three is left describing a different case."""


def turbulence_fields(model: str) -> tuple[str, ...]:
    if model == "laminar":
        return ()
    return TURBULENCE_FIELDS.get(model, ("k", "omega"))


def nut_wall_function(model: str) -> str:
    """Spalding's law, for every model.

    `nutkWallFunction` is the plain high-Re Launder-Spalding form and is only valid for
    a first cell landing around y+ 30-300. On a generated aerofoil O-grid the first cell
    centre sits at y+ ~1200 and the boundary layer is thinner than one cell, so skin
    friction -- and therefore Cd -- came from an extrapolation the model was never valid
    for. A live study measured y+ 40-1713 on exactly this and said so itself.

    `nutUSpaldingWallFunction` blends through the viscous sublayer, the buffer layer and
    the log layer, so it is right across the whole range rather than only inside a band
    the generator cannot guarantee. It also needs no k, which is why Spalart-Allmaras
    already had it. Nothing is lost by using it everywhere: on a mesh that *is* in the
    log layer the two agree.
    """
    return "nutUSpaldingWallFunction"


# -- the 0/ directory --------------------------------------------------------------


def boundary_field(plan: Plan, entry) -> str:
    """A `boundaryField` block with one entry per patch in the mesh.

    Driven by the mesh's own patch list rather than by the roles, so a patch the
    template made and forgot to describe is a loud KeyError here instead of a
    missing entry the solver finds on the first time step.
    """
    lines = ["boundaryField", "{"]
    for name in patch_order(plan.mesh.patch_faces):
        role = plan.roles.get(name)
        if role is None:
            raise SystemExit(
                f"the mesh has a patch '{name}' that the template did not give a role; "
                "every patch needs one, or its 0/ entries are guesses"
            )
        body = entry(name, role)
        lines.append(f"    {name}")
        lines.append("    {")
        for item in body:
            lines.append(f"        {item}")
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def stream_direction(plan: Plan) -> tuple[float, float, float]:
    """The direction the inlet points, or +x when a template has no inlet.

    The internal field and the outlet's fall-back value are seeded with this. A
    duct whose inlet is on the floor (`duct-f`, `duct-m`) runs up the y axis, and
    starting every cell in it at (U 0 0) points the whole domain across the duct
    instead of along it -- which converges, eventually, from further away.
    """
    for name in patch_order(plan.roles):
        role = plan.roles[name]
        if role.get("kind") == "inlet":
            direction = tuple(float(c) for c in role.get("direction", (1.0, 0.0, 0.0)))
            length = math.sqrt(sum(c * c for c in direction))
            if length > TOLERANCE:
                return (direction[0] / length, direction[1] / length, direction[2] / length)
    return (1.0, 0.0, 0.0)


def field_U(plan: Plan, flow: Flow) -> str:
    stream = tuple(component * flow.speed for component in stream_direction(plan))
    inlet = f"uniform {vector(stream)}"

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            direction = role.get("direction", (1.0, 0.0, 0.0))
            value = tuple(component * flow.speed for component in direction)
            return ["type            fixedValue;", f"value           uniform {vector(value)};"]
        if kind == "outlet":
            return ["type            inletOutlet;",
                    "inletValue      uniform (0 0 0);",
                    f"value           {inlet};"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "belt":
            return ["type            fixedValue;", f"value           {inlet};"]
        if kind == "spinning":
            origin, axis = role["origin"], role["axis"]
            # omega is the wheel's, and a rolling wheel's is the road speed over its
            # radius: a number that follows from --speed rather than one to type in.
            omega = flow.speed / float(role["radius"])
            return ["type            rotatingWallVelocity;",
                    f"origin          {vector(origin)};",
                    f"axis            {vector(axis)};",
                    f"omega           {omega:.6g};",
                    "value           uniform (0 0 0);"]
        return ["type            noSlip;"]

    body = ("dimensions      [0 1 -1 0 0 0 0];\n\n"
            f"internalField   {inlet};\n\n" + boundary_field(plan, entry))
    return foam_file("volVectorField", "U", body, "0")


def field_p(plan: Plan) -> str:
    """Kinematic pressure: incompressible OpenFOAM solves p/rho, in m2/s2. Naming
    the units here is not decoration -- a force computed from this as if it were
    pascals is out by a factor of rho and looks plausible."""

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            # `symmetry`, not `symmetryPlane`: the latter is a constraint that
            # requires its faces to be coplanar, so it cannot carry the two
            # opposite walls of a tunnel as one patch, and blockMesh rejects it
            # outright. The physics is identical.
            # A symmetry patch takes the constraint on every field it
            # carries, p included: OpenFOAM refuses a zeroGradient there
            # rather than quietly accepting it.
            return ["type            symmetry;"]
        if kind == "outlet":
            return ["type            fixedValue;", "value           uniform 0;"]
        if kind == "inlet":
            return ["type            zeroGradient;"]
        return ["type            zeroGradient;"]

    body = ("dimensions      [0 2 -2 0 0 0 0];   // kinematic: p/rho, m2/s2\n\n"
            "internalField   uniform 0;\n\n" + boundary_field(plan, entry))
    return foam_file("volScalarField", "p", body, "0")


CMU = 0.09
"""The k-epsilon family's model constant. The omega and epsilon estimates are the same
mixing-length argument written twice."""


def wall_function(name: str) -> list[str]:
    return [f"type            {name};", "value           $internalField;"]


def field_k(plan: Plan, flow: Flow, intensity: float) -> str:
    value = 1.5 * (intensity * flow.speed) ** 2

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function("kqRWallFunction")

    body = (f"dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform {value:.6g};\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "k", body, "0")


def omega_for(k: float, nu: float, mixing: float, ratio: float | None) -> float:
    """omega for the free stream.

    `ratio` is a target eddy-viscosity ratio nu_t/nu, and when it is given omega comes
    straight from it: nu_t = k/omega, so omega = k/(ratio*nu). That is the external-aero
    way to state free-stream turbulence, and it is stated because the other way was
    silently wrong here: omega = sqrt(k)/(Cmu^0.25 * l) with l = 0.07*L is the
    fully-developed **pipe** mixing-length correlation, applied to a body length in
    unbounded flow. With the old 5% intensity default it put nu_t/nu at ~7,000 on every
    turbulent external case -- against a recommended free-stream band of 0.1 to 10.

    Two live studies confirmed it and neither noticed: a NACA 0012 came back with Cd
    seven times the published value, and an Ahmed body 41% high with its 25-degree slant
    separation diffused away. The mixing-length form is kept for the internal flows it
    is actually about, and for anyone who asks for it with --mixing-length.
    """
    if ratio and nu > 0:
        return k / (ratio * nu)
    return math.sqrt(k) / (CMU ** 0.25 * mixing)


def field_omega(plan: Plan, flow: Flow, intensity: float, mixing: float,
                nu: float = 0.0, ratio: float | None = None) -> str:
    k = 1.5 * (intensity * flow.speed) ** 2
    value = omega_for(k, nu, mixing, ratio)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function("omegaWallFunction")

    body = (f"dimensions      [0 0 -1 0 0 0 0];\n\ninternalField   uniform {value:.6g};\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "omega", body, "0")


def field_epsilon(plan: Plan, flow: Flow, intensity: float, mixing: float) -> str:
    """epsilon for kEpsilon, from the same k and mixing length omega comes from:
    epsilon = Cmu^0.75 k^1.5 / L."""
    k = 1.5 * (intensity * flow.speed) ** 2
    value = CMU ** 0.75 * k ** 1.5 / mixing

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function("epsilonWallFunction")

    body = (f"dimensions      [0 2 -3 0 0 0 0];\n\ninternalField   uniform {value:.6g};\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "epsilon", body, "0")


SA_NU_TILDA_RATIO = 4.0
"""Free-stream nuTilda as a multiple of nu. Spalart's own recommendation is between
3 and 5 times; the middle of it is a starting value, not a measurement."""


def field_nu_tilda(plan: Plan, flow: Flow) -> str:
    value = SA_NU_TILDA_RATIO * flow.nu

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        # Spalart-Allmaras carries nuTilda to zero at a wall; there is no wall
        # function for it, the wall function is on nut.
        return ["type            fixedValue;", "value           uniform 0;"]

    body = (f"dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform {value:.6g};   "
            f"// {SA_NU_TILDA_RATIO:g} * nu\n\n" + boundary_field(plan, entry))
    return foam_file("volScalarField", "nuTilda", body, "0")


def field_gamma_int(plan: Plan) -> str:
    """Intermittency: 1 in the free stream, and the model decides where it falls."""

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "inlet":
            return ["type            inletOutlet;", "inletValue      uniform 1;",
                    "value           uniform 1;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      uniform 1;",
                    "value           uniform 1;"]
        return ["type            zeroGradient;"]

    body = ("dimensions      [0 0 0 0 0 0 0];\n\ninternalField   uniform 1;\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "gammaInt", body, "0")


def field_re_theta(plan: Plan, intensity: float) -> str:
    value = free_stream_re_theta(intensity)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind in ("inlet", "outlet"):
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        return ["type            zeroGradient;"]

    body = (f"dimensions      [0 0 0 0 0 0 0];\n\ninternalField   uniform "
            f"{value:.6g};   // Langtry-Menter, Tu = {intensity * 100:g}%\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "ReThetat", body, "0")


def field_nut(plan: Plan, model: str) -> str:
    wall = nut_wall_function(model)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind in ("inlet", "outlet"):
            return ["type            calculated;", "value           uniform 0;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function(wall)

    body = ("dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "nut", body, "0")


# -- system and constant -----------------------------------------------------------

SOLVERS = {"steady": "simpleFoam", "transient": "pimpleFoam", "mesh": ""}


def function_objects(plan: Plan, flow: Flow, opts, study: str) -> str:
    """`forceCoeffs` and a residual log, written into the case that is meant to answer
    for them.

    controlDict had no `functions` block at all, while `preflight.py` warns when a
    forceCoeffs object is missing and `results.py` reads `forceCoeffs.dat` -- both ends
    of the pipeline assumed one existed and only the generator omitted it. So every
    study hand-wrote its own, including `Aref = size * thickness`, where the thickness
    defaults to size/10: an easy factor of ten that silently rescales every coefficient.

    `lRef` and `Aref` are written from the geometry this template actually built, so the
    coefficients mean what their names say without anybody re-deriving them. Only the
    external templates get forces -- a duct has no body to take them on.
    """
    body = str(opts.get("body_patch") or "body")
    if body not in plan.mesh.patch_face_counts():
        return ""
    area = plan.length * plan.mesh.thickness
    magnitude = flow.speed
    entries = [
        "functions", "{",
        "    forceCoeffs", "    {",
        "        type            forceCoeffs;",
        "        libs            (forces);",
        f"        patches         ({body});",
        "        rho             rhoInf;",
        "        rhoInf          1;",
        "        liftDir         (0 1 0);",
        "        dragDir         (1 0 0);",
        "        CofR            (0 0 0);",
        "        pitchAxis       (0 0 1);",
        f"        magUInf         {magnitude:g};",
        f"        lRef            {plan.length:.6g};",
        f"        Aref            {area:.6g};   // {plan.length:g} m span-chord x "
        f"{plan.mesh.thickness:g} m thickness",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "    }",
        "    residuals", "    {",
        "        type            solverInfo;",
        "        libs            (utilityFunctionObjects);",
        "        fields          (U p);",
        "        writeResidualFields false;",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "    }",
        "}",
    ]
    return "\n" + "\n".join(entries)


def purge_write(study: str, opts) -> int:
    """How many recent time directories to keep on disk (0 keeps all).

    A steady run's intermediate writes are only convergence snapshots -- the answer is
    `latestTime` and the convergence history is in the log, not the fields -- so keeping
    a few and discarding the rest is saved I/O at no cost. A transient run's time series
    is often the deliverable itself (a shedding animation is its frames), so it keeps
    everything by default; `--purge N` sets a bound for a transient run whose history is
    not wanted. The write path is the network Volume, so every retained time directory is
    a set of files written across the wire and read back again by reconstruct.
    """
    requested = opts.get("purge") if opts else None
    if requested is not None:
        return max(0, int(requested))
    return 0 if study == "transient" else 3


def io_optimisation() -> str:
    """The collated file handler, set on the case so decomposePar, the solver and
    reconstructPar all agree without a `-fileHandler` flag on each command.

    Uncollated writes one file per field per processor per write -- 29 files a write on a
    4-rank cylinder case; collated writes one file per processor region, 7 a write. Fewer
    files is the whole cost on the 9p Volume: measured, collated reconstruct ran in 16.6s
    against uncollated's 39.3s on the Volume, and it is 4x fewer inodes to checkpoint and
    to reconstruct. Set in the case's `OptimisationSwitches` rather than globally so the
    case is self-describing and a person who opens it sees how it was written.
    """
    return "\nOptimisationSwitches\n{\n    fileHandler     collated;\n}\n"


def control_dict(plan: Plan, flow: Flow, opts, study: str) -> str:
    """The run's clock.

    A steady run counts iterations; a transient one counts seconds, and its deltaT
    is the Courant number asked for times the mesh's shortest cell over the free
    stream -- the shortest cell as the mesh actually has it, grading included,
    because that is the one the Courant number is set by.

    `--delta-t` means a fixed step, so it turns `adjustTimeStep` off. Writing the
    step the user asked for and then leaving the solver free to change it is the
    kind of quiet disagreement between what was said and what was written that
    only shows up in the time directory names.
    """
    solver = SOLVERS.get(study, "")
    if study == "transient":
        end = float(opts["end_time"])
        cell = plan.mesh.smallest_cell
        fixed = opts.get("delta_t") is not None
        delta = float(opts["delta_t"]) if fixed else float(opts["courant"]) * cell / flow.speed
        interval = end / max(1, int(opts["writes"]))
        entries = [f"application     {solver};", "startFrom       latestTime;", "startTime       0;",
                   "stopAt          endTime;", f"endTime         {end:g};",
                   f"deltaT          {delta:.6g};",
                   f"writeControl    {'runTime' if fixed else 'adjustableRunTime'};",
                   f"writeInterval   {interval:.6g};", f"purgeWrite      {purge_write(study, opts)};",
                   "writeFormat     binary;",
                   "writePrecision  6;", "writeCompression off;", "timeFormat      general;",
                   "timePrecision   6;", "runTimeModifiable true;"]
        which = "the average cell" if plan.mesh.cell_is_estimate else "the shortest cell"
        if fixed:
            entries += ["adjustTimeStep  no;",
                        f"// --delta-t {delta:.6g} s; {which} is {cell:.4g} m, so "
                        f"Co = {delta * flow.speed / cell:.3g} at {flow.speed:g} m/s"]
        else:
            entries += ["adjustTimeStep  yes;",
                        f"maxCo           {float(opts['courant']):g};",
                        # 5x, not 100x. The solver grows into maxDeltaT whenever the
                        # flow lets it, and 100x the initial step is a Courant number
                        # around 90 -- a bound that bounds nothing.
                        f"maxDeltaT       {delta * 5:.6g};",
                        f"// deltaT = maxCo * {cell:.4g} m ({which}) / {flow.speed:g} m/s"]
    else:
        end = float(opts["iterations"])
        entries = [f"application     {solver or 'simpleFoam'};", "startFrom       latestTime;",
                   "startTime       0;", "stopAt          endTime;", f"endTime         {end:g};",
                   "deltaT          1;", "writeControl    timeStep;",
                   f"writeInterval   {max(1, int(end / max(1, int(opts['writes'])))):d};",
                   f"purgeWrite      {purge_write(study, opts)};", "writeFormat     binary;",
                   "writePrecision  6;",
                   "writeCompression off;", "timeFormat      general;", "timePrecision   6;",
                   "runTimeModifiable true;"]
    body = "\n".join(entries) + function_objects(plan, flow, opts, study) + io_optimisation()
    return foam_file("dictionary", "controlDict", body, "system")


def fv_schemes(study: str, model: str) -> str:
    time_scheme = "steadyState" if study != "transient" else "backward"
    # Steady runs keep linearUpwind: bounded and stable is what gets a SIMPLE run to a
    # converged answer. A transient one gets `Gauss linear`, because upwind dissipation
    # damps exactly the instability a transient study is usually run to observe -- a
    # Re=100 shedding case can be smoothed into a steady-looking wake by the scheme
    # rather than by the physics, which is the one error that looks like a result.
    divergence = ("bounded Gauss linearUpwind grad(U)" if study != "transient"
                  else "Gauss linear")
    prefix = "bounded " if study != "transient" else ""
    lines = ["ddtSchemes", "{", f"    default         {time_scheme};", "}", "",
             "gradSchemes", "{", "    default         Gauss linear;", "}", "",
             "divSchemes", "{", "    default         none;", f"    {'div(phi,U)':<16}{divergence};"]
    # `default none` means every convected field has to be named here. Naming the
    # ones this model does not transport is harmless; leaving out one it does
    # transport stops the run on the first iteration.
    for field in turbulence_fields(model):
        lines.append(f"    {f'div(phi,{field})':<16}{prefix}Gauss limitedLinear 1;")
    lines += ["    div((nuEff*dev2(T(grad(U))))) Gauss linear;", "}", "",
              "laplacianSchemes", "{", "    default         Gauss linear corrected;", "}", "",
              "interpolationSchemes", "{", "    default         linear;", "}", "",
              "snGradSchemes", "{", "    default         corrected;", "}", "",
              "wallDist", "{", "    method          meshWave;", "}"]
    return foam_file("dictionary", "fvSchemes", "\n".join(lines), "system")


P_SOLVER = ["solver          GAMG;", "tolerance       1e-07;", "relTol          0.01;",
            "smoother        GaussSeidel;"]
SMOOTH_SOLVER = ["solver          smoothSolver;", "smoother        symGaussSeidel;",
                 "tolerance       1e-08;", "relTol          0.1;"]
PHI_SOLVER = ["solver          GAMG;", "smoother        DIC;", "tolerance       1e-06;",
              "relTol          0.01;"]
"""The linear solver `potentialFoam` needs for the velocity-potential Laplace equation.

A steady SIMPLE run started from a uniform field spends its first hundred-odd iterations
just filling the domain with a flow field before the residuals mean anything, and a
coarse case can diverge in that startup before it ever converges. `potentialFoam`
computes a potential-flow field in seconds and writes it back as the initial condition,
which the field notes record as routinely turning a diverging start into a converging
one -- but it stops at once with `keyword Phi is undefined` unless fvSolution names a
solver for `Phi` and a `potentialFlow` block sets its corrector count. The generator
never wrote either, so an agent reaching for the warm-start it is told to reach for had
to hand-edit fvSolution first. Both are inert for simpleFoam itself (it reads neither),
so a steady case now carries the warm-start as a capability rather than a chore."""


def solver_entry(name: str, settings: list[str]) -> list[str]:
    return [f"    {name}", "    {"] + [f"        {line}" for line in settings] + ["    }", ""]


def converged(settings: list[str]) -> list[str]:
    """The same solver run to its absolute tolerance instead of a relative one."""
    return [line for line in settings if not line.startswith("relTol")] + ["relTol          0;"]


RESIDUAL_CONTROL = (("p", 1e-4), ("U", 1e-5))
"""What "converged" means for a steady run, so the solver can say it reached it.

There was no residualControl anywhere in the toolbox, so `endTime = --iterations` was
the only stopping rule: every steady case ran exactly its iteration count and stopped,
converged or not, and never printed "SIMPLE solution converged". In one run of ten
studies, four would have shipped an unconverged answer if the agent had trusted the
residual tables it was shown -- it caught them instead by computing conservation checks
the product does not provide.

Ordinary values, and deliberately loose enough not to stop a run early. They decide
nothing: a solver that converges says so and stops, one that plateaus still runs to
endTime, and whether a plateau is physics or a bad mesh is a reading nothing here makes.
"""


def residual_control(fields: tuple[str, ...]) -> list[tuple[str, float]]:
    """The residualControl entries, including whichever turbulence fields this model has."""
    entries = list(RESIDUAL_CONTROL)
    if fields:
        # One field is named plainly; several become one regex. Same convention the
        # linear-solver block uses, so the two read as describing the same case.
        entries.append((f'"({"|".join(fields)})"' if len(fields) > 1 else fields[0], 1e-5))
    return entries


def fv_solution(study: str, model: str) -> str:
    """The linear solvers.

    PIMPLE asks for `pFinal` and `UFinal` by those exact literal names on the last
    inner iteration of every time step, and `p` does not stand in for `pFinal`:
    without the Final entries pimpleFoam stops on the first step with `keyword
    pFinal is undefined in dictionary solvers`. A steady SIMPLE run never asks for
    them, which is why the omission survives a steady case and kills a transient one.
    """
    fields = turbulence_fields(model)
    entries = [("p", P_SOLVER), ("U", SMOOTH_SOLVER)]
    if fields:
        entries.append((f'"({"|".join(fields)})"' if len(fields) > 1 else fields[0], SMOOTH_SOLVER))

    lines = ["solvers", "{"]
    for name, settings in entries:
        lines += solver_entry(name, settings)
        if study == "transient":
            final = name[:-1] + 'Final"' if name.endswith('"') else name + "Final"
            lines += solver_entry(final, converged(settings))
    if study != "transient":
        # Only the steady case carries it: `potentialFoam` is the warm-start for a
        # segregated steady solve, and a transient solvers block asks every entry for a
        # matching `Final` -- `Phi` has none, and does not want one.
        lines += solver_entry("Phi", PHI_SOLVER)
    lines += ["}", ""]
    if study == "transient":
        lines += ["PIMPLE", "{", "    nOuterCorrectors 2;", "    nCorrectors     2;",
                  "    nNonOrthogonalCorrectors 1;", "}"]
    else:
        lines += ["potentialFlow", "{", "    nNonOrthogonalCorrectors 10;", "}", "",
                  "SIMPLE", "{", "    nNonOrthogonalCorrectors 1;", "    consistent      yes;",
                  "    residualControl", "    {"]
        lines += [f"        {field:<15} {tol:g};" for field, tol in residual_control(fields)]
        lines += ["    }", "}", "",
                  "relaxationFactors", "{", "    equations", "    {",
                  '        U               0.9;', '        ".*"            0.9;', "    }", "}"]
    return foam_file("dictionary", "fvSolution", "\n".join(lines), "system")


def transport_properties(flow: Flow) -> str:
    body = ("transportModel  Newtonian;\n\n"
            f"nu              {flow.nu:.6g};\n\n"
            f"// nu = U*L/Re with U = {flow.speed:g} m/s, L = {flow.length:g} m, "
            f"Re = {flow.reynolds:.0f}")
    return foam_file("dictionary", "transportProperties", body, "constant")


def turbulence_properties(model: str) -> str:
    if model == "laminar":
        body = "simulationType  laminar;"
    else:
        body = ("simulationType  RAS;\n\nRAS\n{\n"
                f"    RASModel        {model};\n"
                "    turbulence      on;\n"
                "    printCoeffs     on;\n}")
    return foam_file("dictionary", "turbulenceProperties", body, "constant")


# -- the case ----------------------------------------------------------------------


def case_files(plan: Plan, flow: Flow, opts, study: str, model: str) -> dict[str, str]:
    """Every file the case is made of, path relative to the case directory.

    Returned rather than written so `--dry-run` and the writer share one answer:
    a dry run that lists different files from the ones a real run writes is worse
    than no dry run.
    """
    files = {
        "system/controlDict": control_dict(plan, flow, opts, study),
        "system/fvSchemes": fv_schemes(study, model),
        "system/fvSolution": fv_solution(study, model),
        "constant/transportProperties": transport_properties(flow),
        "constant/turbulenceProperties": turbulence_properties(model),
    }
    files["0/U"] = field_U(plan, flow)
    files["0/p"] = field_p(plan)
    fields = turbulence_fields(model)
    if fields:
        intensity = free_stream_intensity(plan, opts)
        mixing = float(opts.get("mixing_length") or 0.07 * plan.length)
        # A stated mixing length is a deliberate choice and wins; otherwise a body in
        # open flow sets omega from a viscosity ratio, which is what free-stream
        # turbulence is actually specified by (see omega_for).
        ratio = None if opts.get("mixing_length") else viscosity_ratio(plan, opts)
        writers = {
            "k": lambda: field_k(plan, flow, intensity),
            "omega": lambda: field_omega(plan, flow, intensity, mixing,
                                         nu=flow.nu, ratio=ratio),
            "epsilon": lambda: field_epsilon(plan, flow, intensity, mixing),
            "nuTilda": lambda: field_nu_tilda(plan, flow),
        }
        for field in fields:
            files[f"0/{field}"] = writers[field]()
        files["0/nut"] = field_nut(plan, model)
    return files


def write_case(target: Path, files: dict[str, str]) -> list[Path]:
    written = []
    for relative, text in sorted(files.items()):
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def turbulence_notes(plan: Plan, flow: Flow, model: str, opts) -> list[str]:
    """The two numbers a turbulent case is most often quietly wrong about.

    Both are printed because of an asymmetry found in live runs: the near-wall problem
    (F-15) is measurable from the case afterwards -- a y+ function object -- and a study
    duly found it and said the mesh was unfit. The free-stream eddy viscosity is not
    measurable from anything the product reported, so a second study inherited a
    nu_t/nu of 9,000 and never mentioned it, and the answer came back 41% high with a
    plausible-looking explanation attached. A bad default that is reported gets caught;
    a bad default that is invisible does not. So these are printed whether or not they
    look wrong, and the reading is left to whoever is holding the case.
    """
    if not turbulence_fields(model):
        return []
    intensity = free_stream_intensity(plan, opts)
    mixing = float(opts.get("mixing_length") or 0.07 * plan.length)
    ratio = None if opts.get("mixing_length") else viscosity_ratio(plan, opts)
    k = 1.5 * (intensity * flow.speed) ** 2
    omega = omega_for(k, flow.nu, mixing, ratio)
    nut = k / omega if omega > 0 else 0.0
    notes = [
        f"free stream I = {intensity:g}, k = {k:.4g}, omega = {omega:.4g}"
        f"   -> nu_t/nu = {nut / flow.nu:,.1f}" if flow.nu > 0 else "",
        "           (external aerodynamics wants roughly 0.1-10; far above it the "
        "boundary layer sees an ambient viscosity that is not there)",
    ]
    # The shortest edge in the mesh, which on a body wrapped in a graded O-grid is the
    # first cell off the wall -- the one the wall treatment is applied at.
    first = plan.mesh.smallest_cell
    if first and math.isfinite(first):
        y_plus = estimate_y_plus(first, flow)
        notes.append(f"near wall  first cell {first:.3g} m -> y+ ~ {y_plus:.0f}"
                     f"   ({nut_wall_function(model)})")
    return [line for line in notes if line]


def estimate_y_plus(first_cell: float, flow: Flow) -> float:
    """y+ at the centre of the first cell, from a flat-plate skin-friction correlation.

    Order of magnitude, and that is enough: the question it answers is whether the wall
    treatment is being asked for something a thousand times outside its range, not what
    the third digit is. cf = 0.0576 Re_x^-0.2 at x = L/2.
    """
    re_x = max(flow.reynolds / 2.0, 1.0)
    cf = 0.0576 * re_x ** -0.2
    u_tau = flow.speed * math.sqrt(max(cf, 1e-12) / 2.0)
    return (first_cell / 2.0) * u_tau / flow.nu if flow.nu > 0 else 0.0


def summary(case: Path, plan: Plan, flow: Flow, study: str, model: str, why: str,
            opts=None) -> list[str]:
    """What was decided, in the order somebody checking it would ask."""
    mesh = plan.mesh
    counts = mesh.patch_face_counts()
    span = mesh.span
    lines = [
        f"case       {case}",
        f"study      {study}" + (f"   solver {SOLVERS[study]}" if SOLVERS.get(study)
                                 else "   mesh only, no solver"),
        f"flow       {flow.line()}",
        f"turbulence {model}   ({why})",
        f"mesh       {mesh.cells:,} cells, {span[0]:.4g} x {span[1]:.4g} x {span[2]:.4g} m"
        + ("   (2D, one cell thick)" if mesh.two_d else "")
        + (f"   {mesh.checkmesh}" if mesh.checkmesh else ""),
        "patches    " + ", ".join(
            f"{patch} ({counts[patch]}, {plan.roles[patch]['kind']})"
            for patch in patch_order(counts)),
    ]
    if opts is not None:
        lines += turbulence_notes(plan, flow, model, opts)
    lines += [f"           {note}" for note in plan.notes]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("case", nargs="?", type=Path,
                    help="A case directory that already holds constant/polyMesh.")
    ap.add_argument("--dry-run", action="store_true", help="Say what would be written; write nothing.")
    ap.add_argument("--force", action="store_true", help="Write over 0/ and the dictionaries if they are there.")

    flow = ap.add_argument_group("the flow")
    flow.add_argument("--speed", type=float, default=1.0, help="Free-stream or inlet speed, m/s (default 1).")
    flow.add_argument("--reynolds", type=float, default=None, help="Reynolds number; nu follows (default 100).")
    flow.add_argument("--nu", type=float, default=None, help="Kinematic viscosity m2/s; Re follows.")
    flow.add_argument("--length", type=float, default=None,
                      help="Characteristic length for Re. Default: the hydraulic diameter "
                           "of the inlet, measured off the mesh.")
    flow.add_argument("--direction", default="",
                      help="Which way the inlet flows: x, y, z, -x, -y, -z. Default: the "
                           "inlet patch's own normal, pointed into the domain.")
    flow.add_argument("--turbulence", default="auto",
                      choices=["auto", "laminar", "kOmegaSST", "kOmegaSSTLM",
                               "kEpsilon", "SpalartAllmaras"],
                      help="auto: laminar below Re 2300, kOmegaSST above. "
                           "kOmegaSSTLM adds the Langtry-Menter transition "
                           "equations, which is what a blade or an aerofoil below "
                           "Re ~ 5e5 needs -- assume it turbulent from the leading "
                           "edge and the lift comes out low however fine the mesh.")
    flow.add_argument("--turbulent-intensity", type=float, default=None,
                      dest="turbulent_intensity",
                      help="Free-stream turbulence intensity. Default 0.001 for a body in "
                           "open flow, 0.05 inside a passage.")
    flow.add_argument("--viscosity-ratio", type=float, default=None, dest="viscosity_ratio",
                      help="Target free-stream nu_t/nu; omega follows from it (a body in "
                           "open flow, default 10). --mixing-length overrides.")
    flow.add_argument("--mixing-length", type=float, default=None, dest="mixing_length")
    flow.add_argument("--external", action="store_true",
                      help="Treat this as a body in open flow even when the mesh does not "
                           "say so (it says so when a wall patch is not part of the "
                           "domain's own bounding box).")

    run = ap.add_argument_group("the run")
    run.add_argument("--study", default="steady", choices=["mesh", "steady", "transient"])
    run.add_argument("--iterations", type=int, default=1000, help="Steady: how many (default 1000).")
    run.add_argument("--end-time", type=float, default=1.0, dest="end_time", help="Transient: seconds.")
    run.add_argument("--delta-t", type=float, default=None, dest="delta_t",
                     help="Transient: a fixed step, which turns adjustTimeStep off.")
    run.add_argument("--courant", type=float, default=0.9, help="Transient: target max Courant (default 0.9).")
    run.add_argument("--writes", type=int, default=50, help="How many times to write (default 50).")
    run.add_argument("--purge", type=int, default=None, dest="purge",
                     help="Keep only the last N time directories (0 keeps all). Default: 3 for a "
                          "steady run (the answer is latestTime), 0 for a transient one (its time "
                          "series may be the deliverable).")

    patches = ap.add_argument_group("what each patch is for (names, comma-separated)")
    patches.add_argument("--inlet", action="append", default=[],
                         help="Patches the flow enters through. Default: a patch named inlet.")
    patches.add_argument("--outlet", action="append", default=[],
                         help="Patches the flow leaves through. Default: a patch named outlet.")
    patches.add_argument("--wall", action="append", default=[], help="No-slip walls.")
    patches.add_argument("--slip", action="append", default=[], help="Free-slip walls.")
    patches.add_argument("--symmetry", action="append", default=[], help="Symmetry planes.")
    patches.add_argument("--belt", action="append", default=[],
                         help="Walls that translate with the stream -- a moving ground.")
    patches.add_argument("--spin", action="append", default=[],
                         help="A wall that rotates: name:radius[:axis], e.g. wheel:0.15:z. "
                              "omega is the road speed over the radius.")
    patches.add_argument("--body", default="body", dest="body_patch",
                         help="The patch forces are taken on, when there is one (default 'body'). "
                              "A patch by that name gets a forceCoeffs function object.")

    args = ap.parse_args(argv)

    if args.case is None:
        ap.error("a case directory that already holds constant/polyMesh")

    opts = vars(args)
    plan = build_plan(Path(args.case), opts)
    flow = derive_flow(opts, plan.length)
    model, why = turbulence_model(opts, flow)
    files = case_files(plan, flow, opts, args.study, model)

    for line in summary(Path(args.case), plan, flow, args.study, model, why, opts):
        print(line)
    print()

    if args.dry_run:
        print(f"would write {len(files)} files into {args.case}:")
        for relative in sorted(files):
            print(f"  {relative}   ({len(files[relative].splitlines())} lines)")
        return 0

    target = Path(args.case)
    if (target / "0" / "U").exists() and not args.force:
        raise SystemExit(f"{target} already has a 0/ directory; --force to write over it")
    written = write_case(target, files)
    print(f"wrote {len(written)} files into {target}")
    if SOLVERS.get(args.study):
        print(f"next: {SOLVERS[args.study]} -case {target}   (the mesh is already there)")

    # The state belongs to the STUDY, not to the case: a study with two cases in it
    # (a mesh study and the solve, or Re=100 beside Re=200) has one manifest and one
    # phase table, and `gallery.py <study>` is meant to see both. Recording against
    # the case put a `.reynolds` inside each case directory and left the study home
    # with none, so the gallery of the study came back empty.
    #
    # `study_state.find_root` walks up and prefers a `.reynolds` that already exists,
    # so passing the parent is only a starting point and an established study home
    # further up still wins. The parent is not used when it is the workspace root
    # itself -- every study under /work would then share one manifest.
    home = target.parent
    if home.name in ("work", "") or home == home.parent:
        home = target
    try:
        study_state.record("other", target, root=home, case=target.name,
                           label=f"case files on a {plan.mesh.cells}-cell mesh",
                           study=args.study, reynolds=flow.reynolds,
                           nu=flow.nu, cells=plan.mesh.cells)
        study_state.set_phase("case", "done", root=home, case=target.name,
                              note=f"{args.study}, {plan.mesh.cells} cells")
    except OSError as exc:
        # The case is on disk either way; a manifest that could not be written is
        # worth a line and not worth losing the case over.
        print(f"(the study manifest could not be updated: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
