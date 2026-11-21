"""A conjugate wall: two regions, one interface, and the exact answer it owes.

    python3 cht_wall.py            # write the case and build the two-region mesh
    python3 cht_wall.py --no-mesh  # write the files only, no OpenFOAM needed

`advise.py` offers a `conjugate-heat-transfer` family -- `chtMultiRegionFoam` and its
steady twin -- and this repository had no case for either: no template, no tutorial row,
nothing graded. `openreynolds/toolbox/conjugate.py` supplied the reference half. This
builds **the two-region mesh with its interface**, which is what such a case cannot
start without, and prints the exact answer the eventual solve owes.

**It does not invent the solver dictionaries: it copies them off the image.** The
per-region thermophysical files and the coupled thermal boundary condition are spelled
differently in the ESI and Foundation builds -- `thermophysicalProperties` against
`physicalProperties`, `turbulenceProperties` against `momentumTransport` -- and nothing
here had run `chtMultiRegionSimpleFoam` to establish which this image takes. Guessing
them would produce a case nobody has run that reads like one that works.

So they are taken from the image's own tutorial, which is `chem0d.py`'s pattern and for
`chem0d.py`'s reason: it copies `constant/reactions` and `constant/thermo` out of
`$FOAM_TUTORIALS` so the data is the image's own file byte for byte, and off the image it
says which path it wanted instead of inventing coefficients. Here:

* every file in the tutorial's fluid-region and solid-region `constant/` directories is
  copied -- **without enumerating their names**, because the names are the thing not
  known; whatever this image's tutorial carries, the case gets;
* the **coupled thermal boundary condition is read out of the tutorial's own `T` field**,
  by finding the entry on its region-to-region patch and keeping that type block verbatim.
  That is the one entry a guess would most likely get wrong, and it is now the image's
  answer rather than mine;
* the fluid's `Pr` and the solid's `kappa` are then patched to this case's values with
  asserted replacements, so a tutorial that does not carry those entries is **refused by
  name** rather than silently solved with the tutorial's properties.

If the tutorial has moved, the refusal says which path it wanted and lists what is beside
it, so one run on the image is enough to find the new one.

**What is verified and what is not.** The mesh recipe here is not invented: it follows
`docs/cad-acceptance/artifacts/desk/T8/build.py`, a recorded run against this image that
built a two-region mesh and checked it. From that artifact --

* `blockMesh` takes the cell-zone name in the block definition, so two blocks named
  `fluid` and `solid` need no `topoSet` at all;
* `splitMeshRegions -cellZones -overwrite` then writes one mesh per region **and makes
  the interface a mappedWall pair by itself**, which is why the interface faces are left
  out of `boundary` entirely -- they merge as internal first;
* the unsplit `constant/polyMesh`, `constant/cellToRegion` and the `0/` directory have to
  be removed afterwards, because `cellToRegion` is a field of neither region and anything
  reading the base mesh afterwards is reading the wrong mesh;
* `checkMesh -allRegions` is the verdict.

**No solved result exists in this repository yet**, which is the state `rotating.py` and
`free_surface.py` are also in and say so. What is claimed here is the mesh, checked by
`checkMesh -allRegions`, the arithmetic, and that the properties are the image's own --
nothing about an answer.

The answer it owes, when something does solve it
------------------------------------------------
With gravity off the fluid cannot convect, so both regions conduct and the steady
solution is series resistance -- exact, derived and graded in `conjugate.py`:

    q = (T_hot - T_cold) / (L_f/k_f + L_s/k_s)
    T_interface = T_hot - q L_f/k_f

Gravity is set to zero deliberately and that is a modelling choice, not a convenience: a
fluid layer with a side-to-side temperature difference convects at any Rayleigh number,
so a case left under gravity has no closed form to be graded against. `g` is in
`constant/g` where a reader can see it.

**The trap in the fluid's conductivity.** OpenFOAM's `thermophysicalProperties` for a
fluid has no `kappa` entry: conductivity comes out of the Prandtl number, as
`kappa = mu Cp / Pr`. So asking for a fluid conductivity means SETTING Pr, and a template
that wrote `kappa` somewhere would be setting nothing. The Prandtl number below is
computed from the conductivity this case wants and is printed with the arithmetic beside
it, because `Pr` read as a property of the fluid rather than as the way `kappa` is
specified is how the wrong conductivity gets solved with nobody noticing.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# -- the wall ----------------------------------------------------------------------
#
# EDIT THESE TWO: the two layer thicknesses. Everything else follows -- the mesh is
# sized from them, and `expected()` recomputes the flux and the interface temperature,
# so a copy of this template with different thicknesses still prints the answer its own
# case owes. Changing a conductivity works too, and moves Pr with it.

FLUID_THICK_M = 0.02      # the conducting fluid layer, x
SOLID_THICK_M = 0.01      # the solid layer, x
SPAN_M = 0.004            # y and z: two cells each, so this is 1D with a real 3D mesh
CELLS_ACROSS_FLUID = 20
CELLS_ACROSS_SOLID = 10
CELLS_SPAN = 2

HOT_K = 400.0             # the fluid's outer face
COLD_K = 300.0            # the solid's outer face

# Fluid, water-like but not water: the numbers are round so the arithmetic is readable.
FLUID_RHO = 1000.0
FLUID_CP = 4181.0
FLUID_MU = 1.0e-3
FLUID_K = 0.6             # what we want kappa to be; Pr is derived from it below
FLUID_MOLWEIGHT = 18.0

# Solid, steel-like.
SOLID_RHO = 7854.0
SOLID_CP = 434.0
SOLID_K = 16.0


def prandtl() -> float:
    """`Pr = mu Cp / kappa`, which is how a fluid's conductivity is set in OpenFOAM.

    Inverted from the conductivity this case wants rather than quoted from a table: the
    point of the case is that `kappa` is what the answer turns on, and `Pr` is only the
    place it is written down.
    """
    return FLUID_MU * FLUID_CP / FLUID_K


def expected() -> dict[str, float]:
    """The exact steady answer, by series resistance. `conjugate.py` derives and grades
    this; it is recomputed here so the template prints what it owes without importing a
    sibling, which a template copied into an empty directory could not do."""
    r_fluid = FLUID_THICK_M / FLUID_K
    r_solid = SOLID_THICK_M / SOLID_K
    flux = (HOT_K - COLD_K) / (r_fluid + r_solid)
    return {
        "R_fluid": r_fluid,
        "R_solid": r_solid,
        "flux_W_per_m2": flux,
        "T_interface_K": HOT_K - flux * r_fluid,
        "drop_in_fluid_K": flux * r_fluid,
        "drop_in_solid_K": flux * r_solid,
        "Pr": prandtl(),
    }


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """A command, printed, run, and refused by name when it fails or is not there.

    The second half is the one that was missing. A command that FAILS was already
    reported with its binary named and its stderr shown; a command that does not
    EXIST raised out of `subprocess.run` before the returncode check, so running this
    template on a machine with no OpenFOAM ended in `FileNotFoundError: [WinError 2]
    The system cannot find the file specified` (POSIX says `[Errno 2] No such file or
    directory`), naming no binary and not saying OpenFOAM was the thing missing.

    That is the laptop case, and this file is the worst place for it: a template is
    what a person copies. So the refusal names the binary, says where it comes from,
    and says what has already been written, because the mesh is usually the expensive
    part and it is not lost.
    """
    print("+", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    except FileNotFoundError:
        here = sorted(p.name for p in Path.cwd().iterdir() if p.is_file())
        raise SystemExit(
            f"refused: `{cmd[0]}` is not on PATH. It comes with OpenFOAM, which this "
            "machine does not appear to have, so the rest of this template cannot "
            "run here. What it has already written: "
            + (", ".join(here[:10]) + ("" if len(here) <= 10
                                       else f" and {len(here) - 10} more")
               if here else "nothing yet")
            + ". Copy this directory to a machine with OpenFOAM and run the template "
            "there, or use `openreynolds --local` for the Docker workspace."
        ) from None
    if proc.stdout:
        print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{cmd[0]} failed (exit {proc.returncode})")
    return proc

TUTORIAL_SOURCE = "heatTransfer/chtMultiRegionFoam/multiRegionHeater"
"""Where the per-region properties are copied from, under `$FOAM_TUTORIALS`.

A stock OpenFOAM tutorial, chosen because it is the long-standing conjugate case and
carries both a fluid region and a solid one. Its region names are the tutorial's own --
`bottomAir` and `heater` below -- and nothing here depends on them beyond this constant,
so a differently named tutorial is one edit.
"""

TUTORIAL_FLUID = "bottomAir"
TUTORIAL_SOLID = "heater"
"""The tutorial's own region names, which are not this case's."""

COUPLED_PATCH = re.compile(r"^\s*(\w+_to_\w+)$", re.M)
"""A region-to-region patch name in a tutorial field file, which is what `splitMeshRegions`
produces and therefore what carries the coupled thermal condition."""


def tutorials_root() -> Path:
    """`$FOAM_TUTORIALS`, or a refusal that says what to set."""
    where = os.environ.get("FOAM_TUTORIALS")
    if not where:
        raise SystemExit(
            "refused: FOAM_TUTORIALS is not set, so the per-region properties cannot be "
            "copied off the image. This template does not invent them -- see its "
            "docstring. Source an OpenFOAM environment (OPENREYNOLDS_FOAM_BASHRC names "
            "one) or run with --no-properties to stop after the mesh.")
    root = Path(where)
    if not root.is_dir():
        raise SystemExit(f"refused: FOAM_TUTORIALS is {where}, which is not a directory")
    return root


def copy_region_properties(case: Path) -> dict[str, list[str]]:
    """Copy both regions' `constant/` files out of the tutorial, names unenumerated.

    The names are the thing not known, so they are not listed: whatever this image's
    tutorial carries is what the case gets. A missing tutorial is refused with the path
    and a listing beside it, which is `chem0d.py`'s refusal and for the same reason.
    """
    source = tutorials_root() / TUTORIAL_SOURCE / "constant"
    copied: dict[str, list[str]] = {}
    for ours, theirs in (("fluid", TUTORIAL_FLUID), ("solid", TUTORIAL_SOLID)):
        directory = source / theirs
        if not directory.is_dir():
            beside = sorted(p.name for p in source.iterdir()) if source.is_dir() else []
            raise SystemExit(
                f"refused: {directory} is not there. It is a stock OpenFOAM tutorial "
                f"directory; what is beside it: {', '.join(beside) or 'nothing'}. "
                f"`ls $FOAM_TUTORIALS/{TUTORIAL_SOURCE}/constant` will say where it went.")
        names = []
        # `constant/` is the properties and `system/` is the per-region schemes and
        # solvers. Both are the solver's vocabulary rather than this case's geometry, so
        # both come off the image. `system/` is optional only because a tutorial that
        # keeps its schemes at the case root is still a usable source for `constant/`.
        for where in ("constant", "system"):
            source_dir = tutorials_root() / TUTORIAL_SOURCE / where / theirs
            if not source_dir.is_dir():
                continue
            into = case / where / ours
            into.mkdir(parents=True, exist_ok=True)
            for item in sorted(source_dir.iterdir()):
                if item.is_file():
                    shutil.copyfile(item, into / item.name)
                    names.append(f"{where}/{item.name}")
        if not names:
            raise SystemExit(f"refused: {directory} holds no files to copy")
        copied[ours] = names
    return copied


def coupled_condition() -> tuple[str, str]:
    """The coupled thermal condition's entry, lifted verbatim from the tutorial's `T`.

    Returns (the patch name it was found on, the body between its braces). This is the
    one entry a guess would most likely get wrong -- the type name differs across builds
    and releases -- so it is read out of the image instead of written here.
    """
    source = tutorials_root() / TUTORIAL_SOURCE
    for where in (source / "0" / TUTORIAL_FLUID / "T",
                  source / "0.orig" / TUTORIAL_FLUID / "T"):
        if where.is_file():
            text = where.read_text(encoding="utf-8")
            break
    else:
        raise SystemExit(
            f"refused: no T field for the {TUTORIAL_FLUID} region under {source}/0 or "
            f"{source}/0.orig. `find {source} -name T` will say where it is.")
    found = COUPLED_PATCH.search(text)
    if not found:
        raise SystemExit(
            f"refused: {where} names no `<region>_to_<region>` patch, so the coupled "
            "thermal condition cannot be read off it. The patch naming that "
            "splitMeshRegions produces may have changed.")
    patch = found.group(1)
    start = text.index("{", found.end())
    depth, index = 0, start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    return patch, text[start + 1:index].strip(chr(10))


HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield       | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration   |                                                 |
|   \\\\  /    A nd         |                                                 |
|    \\\\/     M anipulation|                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    %sobject      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def foam(path: Path, cls: str, obj: str, body: str, location: str = "") -> None:
    """One OpenFOAM file, banner and all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    where = f'location    "{location}";\n    ' if location else ""
    path.write_text(HEADER % (cls, where, obj) + body, encoding="utf-8", newline="\n")


def block_mesh_dict() -> str:
    """Two blocks, each carrying its cell-zone name, sharing the interface face.

    The zone name is the third token of a `hex` entry and it is what
    `splitMeshRegions -cellZones` reads. The shared face is named in NO patch, so
    blockMesh merges it as internal and the split turns it into the mappedWall pair --
    this is the T8 artifact's finding and it is the whole reason no `topoSet` is needed.
    """
    x0, x1, x2 = 0.0, FLUID_THICK_M, FLUID_THICK_M + SOLID_THICK_M
    y0, y1 = 0.0, SPAN_M
    z0, z1 = 0.0, SPAN_M

    def corners(xlo: float, xhi: float, first: int) -> str:
        return " ".join(str(first + n) for n in range(8))

    vertices = []
    for x in (x0, x1, x2):
        for z in (z0, z1):
            for y in (y0, y1):
                vertices.append((x, y, z))
    # Vertex order per block: the standard hex ordering, built from the index of each
    # (x, y, z) above rather than written out, so a changed thickness cannot desync it.
    def vid(x: float, y: float, z: float) -> int:
        return vertices.index((x, y, z))

    def hex_for(xlo: float, xhi: float, zone: str, nx: int) -> str:
        ids = [vid(xlo, y0, z0), vid(xhi, y0, z0), vid(xhi, y1, z0), vid(xlo, y1, z0),
               vid(xlo, y0, z1), vid(xhi, y0, z1), vid(xhi, y1, z1), vid(xlo, y1, z1)]
        return (f"    hex ({' '.join(str(i) for i in ids)}) {zone} "
                f"({nx} {CELLS_SPAN} {CELLS_SPAN}) simpleGrading (1 1 1)")

    vertex_text = "\n".join(f"    ({x:.6g} {y:.6g} {z:.6g})" for x, y, z in vertices)

    def face(xlo: float, xhi: float, side: str) -> str:
        x = xlo if side == "min" else xhi
        return (f"            ({vid(x, y0, z0)} {vid(x, y1, z0)} "
                f"{vid(x, y1, z1)} {vid(x, y0, z1)})")

    def side_faces() -> str:
        """The four faces along the span, for BOTH blocks.

        These are not decoration. The reference this case owes is one-dimensional
        series conduction, and that is only the answer if no heat leaves through the
        sides -- so they are adiabatic walls, and naming them is what makes them so.
        Left unnamed they fall into `defaultFaces`, whose type is a setting rather than
        a statement, and a side that conducts makes the series answer wrong while the
        run still converges.
        """
        out = []
        for xlo, xhi in ((x0, x1), (x1, x2)):
            for y in (y0, y1):
                out.append(f"            ({vid(xlo, y, z0)} {vid(xhi, y, z0)} "
                           f"{vid(xhi, y, z1)} {vid(xlo, y, z1)})")
            for z in (z0, z1):
                out.append(f"            ({vid(xlo, y0, z)} {vid(xhi, y0, z)} "
                           f"{vid(xhi, y1, z)} {vid(xlo, y1, z)})")
        return chr(10).join(out)

    return f"""scale   1;

vertices
(
{vertex_text}
);

blocks
(
{hex_for(x0, x1, "fluid", CELLS_ACROSS_FLUID)}
{hex_for(x1, x2, "solid", CELLS_ACROSS_SOLID)}
);

edges ();

boundary
(
    hot
    {{
        type            wall;
        faces
        (
{face(x0, x1, "min")}
        );
    }}
    cold
    {{
        type            wall;
        faces
        (
{face(x1, x2, "max")}
        );
    }}
    sides
    {{
        type            wall;
        faces
        (
{side_faces()}
        );
    }}
);

mergePatchPairs ();
"""


def write_case(case: Path) -> dict[str, float]:
    """Every file, both regions. Returns what the case owes, for printing."""
    answer = expected()

    foam(case / "system" / "blockMeshDict", "dictionary", "blockMeshDict",
         block_mesh_dict(), location="system")

    foam(case / "system" / "controlDict", "dictionary", "controlDict", """\
application     chtMultiRegionSimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         500;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
timeFormat      general;
runTimeModifiable true;
""", location="system")

    # `splitMeshRegions` reads these before it will touch the mesh, which is how CI
    # found them missing: blockMesh built both zones (80 fluid cells, 40 solid) and the
    # split then stopped with `cannot find file "system/fvSchemes"`. The T8 artifact
    # wrote them too, with an empty `solvers {}`, and that part of its recipe was the
    # part not copied. They are the mesh utilities' requirement rather than the
    # solver's: the per-region schemes the solver uses come off the image beside the
    # properties.
    foam(case / "system" / "fvSchemes", "dictionary", "fvSchemes", """\
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
""", location="system")

    foam(case / "system" / "fvSolution", "dictionary", "fvSolution", """\
solvers {}
""", location="system")

    # Gravity off, and visible. A fluid layer with a side-to-side temperature
    # difference convects at any Rayleigh number, so this is what makes the series
    # solution the right reference at all.
    foam(case / "constant" / "g", "uniformDimensionedVectorField", "g", """\
dimensions      [0 1 -2 0 0 0 0];
value           (0 0 0);
""", location="constant")

    foam(case / "constant" / "regionProperties", "dictionary", "regionProperties", """\
regions
(
    fluid       (fluid)
    solid       (solid)
);
""", location="constant")

    return answer


def set_entry(path: Path, key: str, value: str) -> None:
    """Replace a single `key  value;` entry, or refuse by name.

    Asserted rather than best-effort: a property that silently keeps the tutorial's value
    is a case solved for somebody else's fluid, and the answer would be wrong while the
    run converged. `t.replace()` without a count check is a mistake this repository has
    already paid for.
    """
    text = path.read_text(encoding="utf-8")
    found = re.findall(rf"^(\s*){re.escape(key)}(\s+)([^;]+);", text, re.M)
    if len(found) != 1:
        raise SystemExit(
            f"refused: {path} has {len(found)} `{key}` entries, expected exactly one. "
            "This file came from the image's own tutorial, so its shape is the image's; "
            f"open it and see what carries {key} instead of guessing.")
    text = re.sub(rf"^(\s*){re.escape(key)}(\s+)([^;]+);",
                  lambda m: f"{m.group(1)}{key}{m.group(2)}{value};", text, count=1,
                  flags=re.M)
    path.write_text(text, encoding="utf-8", newline=chr(10))


def find_file(directory: Path, *candidates: str) -> Path:
    """The first of `candidates` that is actually there, or a refusal listing the rest.

    The ESI and Foundation builds disagree about these names, which is the whole reason
    the files are copied rather than written, so the lookup disagrees politely too.
    """
    for name in candidates:
        if (directory / name).is_file():
            return directory / name
    present = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    raise SystemExit(
        f"refused: none of {', '.join(candidates)} is in {directory}. What is: "
        f"{', '.join(present) or 'nothing'}. The tutorial this was copied from names it "
        "something else, and this template does not guess property file names.")


def find_mesh_look() -> Path:
    """The deployed toolbox first, then this file's own siblings for local testing."""
    deployed = Path("/work/.toolbox/mesh_look.py")
    if deployed.exists():
        return deployed
    return Path(__file__).resolve().parent.parent / "mesh_look.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the two-region mesh a conjugate case needs, and say what it owes.")
    parser.add_argument("--no-mesh", action="store_true",
                        help="write the files and stop: no OpenFOAM needed")
    parser.add_argument("--no-properties", action="store_true",
                        help="build the mesh but do not copy the per-region properties "
                             "off the image")
    args = parser.parse_args()

    case = Path.cwd()
    answer = write_case(case)

    print("# a conjugate wall: fluid then solid, heat along x, gravity off")
    print(f"  fluid  L {FLUID_THICK_M:g} m  k {FLUID_K:g} W/m/K  "
          f"R {answer['R_fluid']:.6f} m2K/W")
    print(f"  solid  L {SOLID_THICK_M:g} m  k {SOLID_K:g} W/m/K  "
          f"R {answer['R_solid']:.6f} m2K/W")
    print(f"  faces  hot {HOT_K:g} K, cold {COLD_K:g} K")
    print()
    print("  what a solve of this mesh owes, exactly (conjugate.py derives and grades it):")
    print(f"    flux             {answer['flux_W_per_m2']:.8f} W/m2, one value through both")
    print(f"    T at the interface {answer['T_interface_K']:.8f} K")
    print(f"    drop in the fluid  {answer['drop_in_fluid_K']:.6f} K")
    print(f"    drop in the solid  {answer['drop_in_solid_K']:.6f} K")
    print()
    print(f"  Pr = mu Cp / kappa = {FLUID_MU:g} * {FLUID_CP:g} / {FLUID_K:g} = "
          f"{answer['Pr']:.6f}")
    print("    -- `thermophysicalProperties` has no kappa entry: this is where the")
    print("       fluid's conductivity is actually set, and the answer turns on it")

    if args.no_mesh:
        print()
        print("--no-mesh: stopping before blockMesh. The files are written.")
        return

    run(["blockMesh"])
    # The T8 artifact's sequence, in its order and for its reasons.
    run(["splitMeshRegions", "-cellZones", "-overwrite"])
    for leftover in (case / "constant" / "polyMesh", case / "0"):
        if leftover.is_dir():
            shutil.rmtree(leftover)
            print(f"  removed {leftover.name}/: not the mesh of record once the regions exist")
    stale = case / "constant" / "cellToRegion"
    if stale.exists():
        stale.unlink()
        print("  removed constant/cellToRegion: a field of neither region")
    run(["checkMesh", "-allRegions"])
    # One picture per region: a two-region mesh is the one kind where "it meshed" and
    # "it meshed as two things that touch" look identical in a log.
    for region in ("fluid", "solid"):
        run([sys.executable, str(find_mesh_look()), ".", "--region", region,
             "--out", f"look-{region}.png"])

    print()
    print("The two-region mesh is built and checked, and `constant/regionProperties` "
          "names both.")

    if args.no_properties:
        print("--no-properties: stopping before the per-region properties.")
        return

    copied = copy_region_properties(case)
    for region, names in copied.items():
        print(f"  {region}/: copied {len(names)} file(s) from "
              f"$FOAM_TUTORIALS/{TUTORIAL_SOURCE}/constant -- {', '.join(names)}")

    # Only what this case must change, and each refused by name if it is not there.
    fluid_thermo = find_file(case / "constant" / "fluid",
                             "thermophysicalProperties", "physicalProperties")
    set_entry(fluid_thermo, "Pr", f"{prandtl():.6f}")
    print(f"  fluid: Pr set to {prandtl():.6f} in {fluid_thermo.name}, which is where "
          "kappa is specified")

    solid_thermo = find_file(case / "constant" / "solid",
                             "thermophysicalProperties", "physicalProperties")
    set_entry(solid_thermo, "kappa", f"{SOLID_K:g}")
    print(f"  solid: kappa set to {SOLID_K:g} in {solid_thermo.name}")

    patch, body = coupled_condition()
    interface = case / "constant" / "coupled-condition.from-tutorial"
    note = (
        "// Read verbatim from $FOAM_TUTORIALS/" + TUTORIAL_SOURCE
        + ", patch " + patch + "." + chr(10)
        + "// This is the entry a guess would most likely get wrong, so it is the"
        + " image's own" + chr(10)
        + "// rather than this template's. The journey test applies it to"
        + " fluid_to_solid." + chr(10) + body + chr(10))
    interface.write_text(note, encoding="utf-8", newline=chr(10))
    print(f"  the coupled thermal condition was read off the tutorial's {patch} patch "
          f"and kept in {interface.name}")

    print()
    print("No solver has been run on this case in this repository, so no agreement is")
    print("quoted. What the solve owes is printed above and graded by conjugate.py.")


if __name__ == "__main__":
    main()
