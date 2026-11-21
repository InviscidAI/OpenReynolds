"""A conjugate wall: two regions, one interface, and the exact answer it owes.

    python3 cht_wall.py            # write the case and build the two-region mesh
    python3 cht_wall.py --no-mesh  # write the files only, no OpenFOAM needed

`advise.py` offers a `conjugate-heat-transfer` family -- `chtMultiRegionFoam` and its
steady twin -- and this repository had no case for either: no template, no tutorial row,
nothing graded. `openreynolds/toolbox/conjugate.py` supplied the reference half. This
builds **the two-region mesh with its interface**, which is what such a case cannot
start without, and prints the exact answer the eventual solve owes.

**It deliberately writes no solver dictionaries.** The per-region thermophysical files
and the coupled thermal boundary condition differ between the ESI and Foundation builds
of OpenFOAM -- `thermophysicalProperties` against `physicalProperties`,
`turbulenceProperties` against `momentumTransport` -- and nothing in this repository has
run `chtMultiRegionSimpleFoam` to establish which spelling this image takes. Writing
three hundred lines of guessed dictionaries would produce a case nobody has run while
looking like one that works, which is the failure this repository spends its time
catching. The journey test that solves this mesh is what settles them, and it is the next
item.

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

**No solved result exists in this repository**, which is the state `rotating.py` and
`free_surface.py` are also in and say so. What is claimed here is the mesh, checked by
`checkMesh -allRegions`, and the arithmetic -- nothing about an answer.

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
import shutil
import subprocess
import sys
from pathlib import Path

# -- the wall ----------------------------------------------------------------------

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the two-region mesh a conjugate case needs, and say what it owes.")
    parser.add_argument("--no-mesh", action="store_true",
                        help="write the files and stop: no OpenFOAM needed")
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

    print()
    print("The two-region mesh is built and checked, and `constant/regionProperties` "
          "names both.")
    print("No solver dictionaries are written: see this file's docstring for why, and "
          "for what")
    print("the solve owes when something runs it.")


if __name__ == "__main__":
    main()
