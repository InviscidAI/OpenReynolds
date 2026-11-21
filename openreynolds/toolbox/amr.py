#!/usr/bin/env python3
"""Adaptive mesh refinement on this image: the one dictionary that drives it, the
nine entries it requires, and the two things it will quietly not do.

OpenFOAM refines a running mesh through `dynamicRefineFvMesh`, which is a
`dynamicFvMesh` named in `constant/dynamicMeshDict` -- so AMR here is a dictionary
written into a case, not an algorithm. There is nothing to install: the type is in the
v2512 tree and `preflight.py`'s attestation table carries it as `ran=False` with the
provenance "it was read out of src/dynamicFvMesh/ at the v2512 tag and no run here has
used it" (`preflight.py:2841`). This script does not change that reading. It writes the
dictionary and it reads one back; whether the type is registered in the build is one
step of `preflight.py --only probe`, which constructs the real mesh and is the
authority.

Everything below was read on 2026-11-21 from the OpenFOAM v2512 tag at
`gitlab.com/openfoam/core/openfoam`, file by file, because the entries this dictionary
requires are not the entries a tutorial copy suggests.

**The coefficients block is optional and its contents are not.** Both places that read
it use `optionalSubDict("dynamicRefineFvMeshCoeffs")`, so leaving the block out is not
an IO error; every entry inside it is then read with `get<>`, which is. `refineInterval`,
`maxCells`, `maxRefinement`, `field`, `lowerRefineLevel`, `upperRefineLevel` and
`nBufferLayers` come out of `updateTopology()`; `correctFluxes` and `dumpLevel` out of
`readDict()`. Only `unrefineLevel` has a default, `getOrDefault<scalar>("unrefineLevel",
GREAT)`, which means "never unrefine" unless it is set. A dictionary missing any of the
other nine fails the run rather than falling back, and `correctFluxes` is the one
nobody expects to be mandatory.

**Three entries have ranges, and one of the three error messages is wrong.**
`refineInterval` must be >= 1 ("The refineInterval setting in the dynamicMeshDict
should be >= 1."), `maxCells` must be > 0, and `maxRefinement` must be > 0 -- whose
FatalError says "The maxCells setting in the dynamicMeshDict should be > 0." while
checking `maxRefinement`. A run that stops on that message with a plausible `maxCells`
in the dictionary is reading about a different entry.

**`correctFluxes` is how a flux survives a cell being cut**, and an omission is a
warning rather than an error. It is a list of `(flux velocity)` word pairs; the fluxes
on faces the refinement created are recomputed by interpolating the named velocity. The
value `none` skips a field, `NaN` fills it with NaN to make a use of the mapped variable
show up, and a `surfaceScalarField` not in the list gets `WarningInFunction` "Cannot
find surfaceScalarField <name>" and keeps whatever was on those faces. The run
continues. `$FOAM_TUTORIALS/multiphase/interFoam/laminar/damBreakWithObstacle`'s own
`constant/dynamicMeshDict` -- read at the v2512 tag, not run here -- sets its fluxes to
`none`, so `none` is a tutorial's choice and not a mistake.

**Non-hex cells are not refined at all.** The mesh cutter is `hexRef8` and the class
holds `bitSet protectedCell_`, commented "Protected cells (usually since not hexes)".
So AMR on a tet or polyhedral mesh refines the hexes in it and leaves the rest: the
cell count goes up, the region you cared about may not. `blockMesh`, `snappyHexMesh`
and cfMesh's `cartesianMesh` all produce meshes this can cut.

**The steady solver reads this dictionary too.** `simpleFoam` at v2512 includes
`createDynamicFvMesh.H` and calls `mesh.controlledUpdate()` once per `simple.loop()`,
exactly as `pimpleFoam` does -- so refinement driven by a converging steady solve needs
no change of solver, and `case_gen.py`'s `{"steady": "simpleFoam"}` case can carry it.
`interFoam`'s own description is "with optional mesh motion and mesh topology changes
including adaptive re-meshing", and `pimpleFoam`'s notes that `updateControl` and
`updateInterval` in the `dynamicMeshDict` influence how often it updates. Those three
are the solvers read; a fourth is an open question, not a no.

**What is NOT here: a measurement.** The honest check on refinement is not a published
number, it is self-consistency -- a refined run and a uniformly fine run agreeing on
the quantity asked for, and the refined one reaching that agreement with fewer cells.
No such pair has been run on this image, so this file states no accuracy and no saving.
Both halves of that measurement are ordinary instruments already here: `grid.py` for
the convergence of a quantity across meshes, `cells_estimate.py` and `mesh_digest.py`
for the cell counts, `probe.py` for the quantity itself. Until a pair exists,
`maxCells` is the only thing standing between a refinement criterion and a bill.

    python3 amr.py write <case> --field alpha.water --lower 0.001 --upper 0.999 \\
                               --max-cells 200000
    python3 amr.py write <case> --field mag(U) --lower 2 --upper 8 --max-cells 400000 \\
                               --max-refinement 3 --unrefine 1 --flux phi:U
    python3 amr.py check <case>
    python3 amr.py check <case> --json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_gen  # noqa: E402  (sibling script, not a package -- for the file furniture)
import preflight  # noqa: E402

Finding = preflight.Finding

COEFFS = "dynamicRefineFvMeshCoeffs"

REQUIRED = (
    "refineInterval",
    "field",
    "lowerRefineLevel",
    "upperRefineLevel",
    "nBufferLayers",
    "maxRefinement",
    "maxCells",
    "correctFluxes",
    "dumpLevel",
)
"""Every entry `dynamicRefineFvMesh` reads with `get<>`/`readEntry`, in the order the
dictionary reads best, so a missing one is named here rather than at the first update.
`unrefineLevel` is deliberately absent: it is the only entry with a default
(`getOrDefault<scalar>("unrefineLevel", GREAT)`, which never unrefines)."""

POSITIVE = {
    "refineInterval": ("refineInterval", 1,
                       "The refineInterval setting in the dynamicMeshDict should be >= 1."),
    "maxCells": ("maxCells", 1, "The maxCells setting in the dynamicMeshDict should be > 0."),
    "maxRefinement": ("maxRefinement", 1,
                      "The maxCells setting in the dynamicMeshDict should be > 0. "
                      "(the message names maxCells; the check is on maxRefinement)"),
}
"""The three entries with a fatal range, and the message each one fatals with."""

DYNAMIC_MESH_SOLVERS = {
    "simplefoam": "it includes createDynamicFvMesh.H and calls mesh.controlledUpdate() "
                  "in its simple.loop(), read at the v2512 tag",
    "pimplefoam": "it includes createDynamicFvMesh.H and createDyMControls.H and calls "
                  "mesh.controlledUpdate(), read at the v2512 tag",
    "interfoam": "its own description is \"with optional mesh motion and mesh topology "
                 "changes including adaptive re-meshing\", and it calls mesh.update(), "
                 "read at the v2512 tag",
}
"""Solvers read in the v2512 source and found to construct a `dynamicFvMesh`. Same rule
as `preflight.MESH_TYPES_ATTESTED`: the entry carries how the knowledge was come by, and
a name absent from here is an open question rather than a refusal."""

SURFACE_SCALAR = "surfaceScalarField"


# ------------------------------------------------------- writing the dict ---


class DictError(Exception):
    """A request that would be accepted and then not carried out."""


def dynamic_mesh_dict(
    field: str, lower: float, upper: float, max_cells: int, max_refinement: int = 2,
    refine_interval: int = 1, n_buffer_layers: int = 1, unrefine: float | None = None,
    fluxes: tuple[tuple[str, str], ...] = (), dump_level: bool = True,
) -> str:
    """The `constant/dynamicMeshDict` for a refining mesh, every required entry present.

    The refusals are the three fatal ranges, checked here where they cost a second
    rather than at the first `refineInterval` of a solve that has already meshed and
    started. `lower >= upper` is refused as well, which is NOT a fatal in OpenFOAM: it
    is an empty band, so nothing is ever marked, and the run goes the whole way having
    paid for the topology machinery and refined nothing.
    """
    if refine_interval < 1:
        raise DictError(f"refineInterval {refine_interval} is fatal: {POSITIVE['refineInterval'][2]}")
    if max_cells < 1:
        raise DictError(f"maxCells {max_cells} is fatal: {POSITIVE['maxCells'][2]}")
    if max_refinement < 1:
        raise DictError(f"maxRefinement {max_refinement} is fatal: {POSITIVE['maxRefinement'][2]}")
    if n_buffer_layers < 0:
        raise DictError("nBufferLayers is a count of layers and cannot be negative")
    if lower >= upper:
        raise DictError(
            f"lowerRefineLevel {lower:g} is not below upperRefineLevel {upper:g}, so the "
            "band is empty and no cell is ever marked -- the run would carry the cost of "
            "a refining mesh and refine nothing"
        )
    pairs = fluxes or (("phi", "none"),)
    flux_text = " ".join(f"({name} {velocity})" for name, velocity in pairs)
    body = "\n".join([
        "// Written by amr.py. Every entry in the block below is read with get<> and has",
        "// no default except unrefineLevel; the block itself is read with optionalSubDict,",
        "// so losing the block is not an error and losing an entry inside it is.",
        "",
        "dynamicFvMesh   dynamicRefineFvMesh;",
        "",
        COEFFS,
        "{",
        "    // How often the criterion is applied, in time steps (fatal below 1).",
        f"    refineInterval  {refine_interval};",
        "",
        "    // The volScalarField the criterion reads. It has to be in the registry when",
        "    // it is first read: a field the solver makes, or one on disk.",
        f"    field           {field};",
        "",
        f"    // Cells with {field} between these two are marked for refinement.",
        f"    lowerRefineLevel {lower:g};",
        f"    upperRefineLevel {upper:g};",
        "",
        *unrefine_lines(unrefine),
        "",
        "    // Refinement spreads this many cells past what was marked, which is what",
        "    // keeps the level change gradual rather than 2:1 at the band's edge.",
        f"    nBufferLayers   {n_buffer_layers};",
        "",
        "    // At most this many levels on any one cell, and at most this many cells in",
        "    // the mesh. maxCells is the only bound on what refinement costs.",
        f"    maxRefinement   {max_refinement};",
        f"    maxCells        {max_cells};",
        "",
        "    // (flux velocity) pairs: the flux on a face refinement created is recomputed",
        "    // by interpolating that velocity. `none` leaves the field alone, `NaN` fills",
        "    // it with NaN so a use of the mapped value shows up. A surfaceScalarField",
        "    // missing from this list gets a warning -- \"Cannot find surfaceScalarField",
        "    // <name>\" -- and the run carries on with whatever was on those faces.",
        f"    correctFluxes   ( {flux_text} );",
        "",
        "    // Writes cellLevel as a volScalarField, so the refinement can be looked at.",
        f"    dumpLevel       {'true' if dump_level else 'false'};",
        "}",
    ])
    # `case_gen.py` owns the file furniture (banner, FoamFile block, footer) and its
    # banner names itself as the writer, which this file is not: the name is swapped at
    # the same width so the dictionary says who wrote it.
    return case_gen.foam_file("dictionary", "dynamicMeshDict", body, "constant").replace(
        "openreynolds case_gen.py", "openreynolds amr.py     ", 1)


def unrefine_lines(unrefine: float | None) -> list[str]:
    """The `unrefineLevel` entry, or the comment saying what its absence means."""
    if unrefine is None:
        return ["    // unrefineLevel is the one entry with a default -- GREAT, which",
                "    // never unrefines -- so leaving it out means the mesh only grows."]
    return ["    // Cells below this are merged back, down to their original level.",
            f"    unrefineLevel   {unrefine:g};"]


# ------------------------------------------------------- reading one back ---


def surface_scalar_fields(case: Path) -> dict[str, str]:
    """`{field name: the time directory it was found in}` for every
    `surfaceScalarField` on disk.

    Read off each file's own `class` entry rather than from a list of names, because
    which fluxes a case carries is the solver's business: `phi` is everywhere, `rhoPhi`
    and `alphaPhi` belong to the multiphase solvers, and a function object can add one.
    Later times win, so the report names the newest directory a flux appears in."""
    found: dict[str, str] = {}
    times = sorted(
        (p for p in case.iterdir() if p.is_dir() and _is_time(p.name)),
        key=lambda p: float(p.name),
    ) if case.is_dir() else []
    for time in times:
        for path in sorted(time.iterdir()):
            if not path.is_file():
                continue
            try:
                head = path.read_bytes()[:600].decode("utf-8", "replace")
            except OSError:
                continue
            header = preflight.block_body(head, "FoamFile")
            if preflight.entry_values(header).get("class") == SURFACE_SCALAR:
                found[path.name] = time.name
    return found


def _is_time(name: str) -> bool:
    if name in ("constant", "system"):
        return False
    try:
        float(name)
    except ValueError:
        return False
    return True


_FLUX_LIST = re.compile(r"correctFluxes\s*\((.*?)\)\s*;", re.S)
_FLUX_PAIR = re.compile(r"\(\s*([^\s()]+)\s+([^\s()]+)\s*\)")


def flux_pairs(text: str) -> list[tuple[str, str]]:
    """The `(flux velocity)` pairs of a `correctFluxes` entry, as written.

    Read off the raw text rather than through `entry_values`, which reads `key value;`
    pairs: this entry's value is a list of lists and may be written across lines. The
    outer match is lazy up to `) ;`, which is where the list ends and nowhere inside
    it."""
    found = _FLUX_LIST.search(preflight.strip_comments(text))
    return _FLUX_PAIR.findall(found.group(1)) if found else []


def check(case: Path) -> list[Finding]:
    """What this case's refinement setup says, and what it leaves to the first update.

    Narrow on purpose: the dictionary's own entries, the solver that will read it, and
    the fluxes on disk that the list does or does not name. Whether
    `dynamicRefineFvMesh` is registered in the build is `preflight.py --only probe`'s
    question and is not answered twice."""
    out: list[Finding] = []
    text = _read(case / "constant" / "dynamicMeshDict") or _read(case / "system" / "dynamicMeshDict")
    if not text.strip():
        return [Finding(
            "amr", "fail", "no constant/dynamicMeshDict",
            "nothing refines without one: a dynamicFvMesh is named in that file and "
            "nowhere else, and case_gen.py writes no dynamicMeshDict at all",
            "python3 amr.py write <case> --field <volScalarField> --lower <a> "
            "--upper <b> --max-cells <n>",
        )]

    top = preflight.entry_values(preflight.outer_text(preflight.strip_comments(text)))
    mesh_type = (top.get("dynamicFvMesh") or "").strip()
    if mesh_type.lower() != "dynamicrefinefvmesh":
        out.append(Finding(
            "amr", "fail", f"dynamicFvMesh is {mesh_type or 'unnamed'}",
            "dynamicRefineFvMesh is the only refining mesh in this build's "
            "src/dynamicFvMesh/; the other attested types move points and change no "
            "topology (preflight.py:2841)",
            "set `dynamicFvMesh dynamicRefineFvMesh;` at the top level, or stop calling "
            "this case refined",
        ))
        return out

    body = preflight.block_body(text, COEFFS)
    if not body.strip():
        out.append(Finding(
            "amr", "fail", f"no {COEFFS} block",
            "the block is read with optionalSubDict, so its absence is not an IO error; "
            "every entry inside it is read with get<>, so the run fatals at the first "
            "refineInterval instead",
            f"add a {COEFFS} block with: " + ", ".join(REQUIRED),
        ))
        return out

    entries = preflight.entry_values(body)
    missing = [name for name in REQUIRED
               if name not in entries and not (name == "correctFluxes" and flux_pairs(body))]
    if missing:
        out.append(Finding(
            "amr", "fail", "missing required entries: " + ", ".join(missing),
            "each one is read with get<> or readEntry and has no default, so the run "
            "stops at the first update rather than carrying on unrefined",
            "add them; only unrefineLevel has a default (GREAT, which never unrefines)",
        ))
    else:
        out.append(Finding(
            "amr", "ok", "every required entry is present",
            "refineInterval, field, lowerRefineLevel, upperRefineLevel, nBufferLayers, "
            "maxRefinement, maxCells, correctFluxes and dumpLevel are the nine with no "
            "default",
        ))

    for key, (name, floor, message) in POSITIVE.items():
        raw = entries.get(key)
        if raw is None:
            continue
        try:
            value = int(float(raw))
        except ValueError:
            continue
        if value < floor:
            out.append(Finding(
                "amr", "fail", f"{name} is {raw}",
                f"fatal at start-up: \"{message}\"",
                f"{name} of at least {floor}",
            ))

    lower, upper = _number(entries.get("lowerRefineLevel")), _number(entries.get("upperRefineLevel"))
    if lower is not None and upper is not None:
        if lower >= upper:
            out.append(Finding(
                "amr", "fail",
                f"lowerRefineLevel {lower:g} is not below upperRefineLevel {upper:g}",
                "the band is empty, so no cell is ever marked. Not a fatal: the run goes "
                "the whole way and refines nothing, having paid for the topology machinery",
                "put the band around the value the feature sits at -- 0.001 to 0.999 on a "
                "phase fraction for an interface",
            ))
        else:
            out.append(Finding(
                "amr", "ok",
                f"refining where {entries.get('field', '?')} is between {lower:g} and {upper:g}",
                "cells inside that band are marked, and nBufferLayers of cells around "
                "them with it",
            ))
    if "unrefineLevel" not in entries:
        out.append(Finding(
            "amr", "warn", "no unrefineLevel",
            "its default is GREAT, so nothing is ever merged back and the mesh only "
            "grows: on a transient whose feature moves, the refinement is left behind "
            "it and maxCells is reached by history rather than by need",
            "set unrefineLevel for a moving feature; leave it out for one that stays put",
        ))

    out.extend(_cost(entries, case))
    out.extend(_solver(case))
    out.extend(_fluxes(body, case))
    return out


def cells_now(case: Path) -> float | None:
    """The mesh's cell count, off the `note` OpenFOAM writes into a polyMesh header.

    `nCells:` is in that note when OpenFOAM wrote the file and absent when the mesh
    came from a converter that did not, so this returns None rather than a guess:
    `checkMesh` and `mesh_digest.py` are the instruments that always know."""
    for name in ("owner", "faces"):
        path = case / "constant" / "polyMesh" / name
        if not path.is_file():
            continue
        try:
            head = path.read_bytes()[:900].decode("utf-8", "replace")
        except OSError:
            continue
        found = re.search(r"nCells:\s*(\d+)", head)
        if found:
            return float(found.group(1))
    return None


def _cost(entries: dict[str, str], case: Path) -> list[Finding]:
    """What `maxCells` means against the mesh that is actually there."""
    cells = _number(entries.get("maxCells"))
    if cells is None:
        return []
    now = cells_now(case)
    if now is None:
        return [Finding(
            "amr", "ok", f"maxCells {cells:g}",
            "the bound on what this refinement can cost. The mesh's own cell count was "
            "not read here (constant/polyMesh/owner carries it in a note only when "
            "OpenFOAM wrote one), so the ratio is unmeasured",
            "checkMesh or mesh_digest.py reports the count this starts from",
        )]
    return [Finding(
        "amr", "ok" if cells / max(now, 1) <= 20 else "warn",
        f"maxCells {cells:g} against {now:g} cells now ({cells / max(now, 1):.1f}x)",
        "the bound on what refinement can cost, as a multiple of the mesh it starts "
        "from. Each level of refinement is a cut of the cells inside the band, not of "
        "the mesh, so the multiple reached depends on how much of the domain the band "
        "covers",
        "cells_estimate.py puts a cell count against a runtime" if cells / max(now, 1) > 20 else "",
    )]


def _solver(case: Path) -> list[Finding]:
    """Whether the application in `controlDict` constructs a `dynamicFvMesh` at all."""
    control = _read(case / "system" / "controlDict")
    if not control.strip():
        return []
    application = (preflight.parse_control(control).get("application") or "").strip()
    if not application:
        return []
    known = DYNAMIC_MESH_SOLVERS.get(application.lower())
    if known:
        return [Finding(
            "amr", "ok", f"application {application} constructs a dynamicFvMesh",
            f"{known}, so it reads constant/dynamicMeshDict and updates the mesh as it runs",
        )]
    return [Finding(
        "amr", "warn", f"application {application} was not read for this",
        "a solver built on createMesh.H instead of createDynamicFvMesh.H never opens "
        "constant/dynamicMeshDict, and says nothing about ignoring it: the run looks "
        "ordinary and the cell count never changes. simpleFoam, pimpleFoam and "
        "interFoam were read at the v2512 tag and all three do construct one",
        "grep createDynamicFvMesh in the solver's source, or watch the cell count in "
        "the log: a refining run prints its refinement each interval",
    )]


def _fluxes(body: str, case: Path) -> list[Finding]:
    """The `correctFluxes` list against the `surfaceScalarField`s on disk."""
    named = {name for name, _ in flux_pairs(body)}
    on_disk = surface_scalar_fields(case)
    if not on_disk:
        return [Finding(
            "amr", "ok", "correctFluxes names " + (", ".join(sorted(named)) or "nothing"),
            "no surfaceScalarField is on disk yet, which is the usual state of a case "
            "that has not run: phi is written by the solver. So the list cannot be "
            "checked against anything here, and an omission shows up as "
            "\"Cannot find surfaceScalarField\" in the log",
        )]
    unlisted = sorted(set(on_disk) - named)
    if unlisted:
        return [Finding(
            "amr", "warn", "surfaceScalarField not in correctFluxes: " + ", ".join(unlisted),
            "each one gets WarningInFunction \"Cannot find surfaceScalarField <name>\" "
            "and keeps whatever was on the faces refinement created. The run continues "
            "and exits 0, which is why this is read off the dictionary rather than the log",
            "add `(" + unlisted[0] + " <velocity>)` to correctFluxes, or `(" + unlisted[0]
            + " none)` if it genuinely needs no reinterpolation",
        )]
    return [Finding(
        "amr", "ok", "correctFluxes names every surfaceScalarField on disk",
        "found " + ", ".join(f"{name} (in {time}/)" for name, time in sorted(on_disk.items())),
    )]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().rstrip(";"))
    except ValueError:
        return None


# --------------------------------------------------------------------- CLI ---


def _parse_flux(spec: str) -> tuple[str, str]:
    name, _, velocity = spec.partition(":")
    if not name.strip() or not velocity.strip():
        raise DictError(f"--flux takes FLUX:VELOCITY (or FLUX:none), not {spec!r}")
    return name.strip(), velocity.strip()


def _cmd_write(args) -> int:
    text = dynamic_mesh_dict(
        field=args.field, lower=args.lower, upper=args.upper, max_cells=args.max_cells,
        max_refinement=args.max_refinement, refine_interval=args.refine_interval,
        n_buffer_layers=args.n_buffer_layers, unrefine=args.unrefine,
        fluxes=tuple(_parse_flux(s) for s in (args.flux or ())),
        dump_level=not args.no_dump_level,
    )
    out = args.case / "constant" / "dynamicMeshDict"
    if out.exists() and not args.force:
        raise DictError(f"{out} exists; --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    if not args.flux:
        print("  correctFluxes names only `(phi none)`, which is what "
              "damBreakWithObstacle's own dictionary does with its fluxes. Any other "
              "surfaceScalarField this case carries needs its own pair, or it gets a "
              "warning and an unmapped flux: `check` reads the ones on disk.")
    if args.unrefine is None:
        print("  no unrefineLevel, so the mesh only grows; maxCells is then reached by "
              "history on a case whose feature moves.")
    print("  non-hex cells are protected from refinement (hexRef8's protectedCell_), so "
          "this refines the hexes in the mesh and leaves the rest.")
    print("  nothing has been measured about this: a refined run is worth the cells only "
          "against a uniformly fine one that agrees with it (grid.py), and no such pair "
          "has been run here.")
    print("\nthen: " + " ".join([
        "python3", Path(__file__).name, "check", str(args.case)]))
    return 0


def _cmd_check(args) -> int:
    findings = check(args.case)
    if args.json:
        print(preflight.as_json(findings, f"{args.case} (amr)"))
    else:
        print(preflight.render(findings, f"{args.case} (amr)"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="write constant/dynamicMeshDict for dynamicRefineFvMesh")
    w.add_argument("case", type=Path)
    w.add_argument("--field", required=True,
                   help="the volScalarField the criterion reads (alpha.water, mag(U), ...)")
    w.add_argument("--lower", type=float, required=True, help="lowerRefineLevel")
    w.add_argument("--upper", type=float, required=True, help="upperRefineLevel")
    w.add_argument("--max-cells", type=int, required=True,
                   help="maxCells: the only bound on what refinement costs")
    w.add_argument("--max-refinement", type=int, default=2, help="levels on any one cell")
    w.add_argument("--refine-interval", type=int, default=1, help="time steps between updates")
    w.add_argument("--n-buffer-layers", type=int, default=1,
                   help="cells of refinement past the marked band")
    w.add_argument("--unrefine", type=float, default=None,
                   help="unrefineLevel; omitted, the mesh only ever grows")
    w.add_argument("--flux", action="append", default=None, metavar="FLUX:VELOCITY",
                   help="a correctFluxes pair, repeatable; VELOCITY may be none or NaN")
    w.add_argument("--no-dump-level", action="store_true",
                   help="do not write cellLevel as a field")
    w.add_argument("--force", action="store_true")

    c = sub.add_parser("check", help="read a case's refinement setup back")
    c.add_argument("case", type=Path)
    c.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        return _cmd_write(args) if args.cmd == "write" else _cmd_check(args)
    except DictError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
