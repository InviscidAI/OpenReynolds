#!/usr/bin/env python3
"""Which solver and how many ranks, both read out of the case rather than assumed.

Two questions that get answered from memory and are cheap to answer from the case:
which of this image's solvers suits the problem, and what `-np` is worth asking for.
Neither has an instrument. `ladder.py` already classifies a case and `compressible.py`
already picks between the three density- and pressure-based compressible solvers from
a Mach number, but nothing puts the families side by side, and nothing anywhere turns
a cell count into a rank count.

the measurement this exists because of

    In study 20260830-164816-eee0 (the ONERA M6 replication) an agent reconstructed a
    1,793,894-cell case off its 4-way decomposition, re-decomposed it 12 ways and
    relaunched it, saying in the same breath "leave the other 4 cores free". `nproc` on
    that container had answered `4`, twice, in that study's own transcript. `mpirun -np
    12` did not refuse -- Open MPI read the host's slots rather than the container's
    cgroup -- so nothing stopped it, and the run went through 2,153 iterations at a
    median 1.45 s each. The four 4-rank stages on the same mesh ran at a median of
    1.45, 1.62, 1.63 and 1.77 s. Three times the ranks bought between nothing and about
    1.2x, which is inside the spread of the numerics changes between those stages.

    That is one case, and one case cannot locate a cells-per-rank floor. What it does
    show is the thing that cost the hour: the rank count was chosen against a believed
    core count of 16 and the measured one was 4. So `nproc` is reported here beside the
    arithmetic, and the one claim made is the one that datum supports.

what is not claimed

    No speedup curve. The corpus behind this script -- 128 local study directories,
    137 solver-log run segments -- has exactly one case that was run at more than one
    rank count, and that one was oversubscribed. Cells per rank is printed as
    arithmetic, not as a verdict against a threshold, because no threshold in here
    would trace to a measurement. Where this case has been run before, its own logs are
    read and what they measured is reported with the number of runs behind it.

Nothing here edits, refuses or blocks. It writes no case, runs no solver, and the exit
code is 0 whatever the report says. A solver named below and not found on `$PATH` is
reported as not found rather than left out, because "this image does not have it" and
"nobody looked" are different answers.

    python3 advise.py /work/case
    python3 advise.py /work/case --mach 0.84 --cores 8
    python3 advise.py /work/case --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells_estimate  # noqa: E402  (sibling scripts, not a package)
import compressible  # noqa: E402
import ladder  # noqa: E402
import preflight  # noqa: E402


# -- which solvers this image has for a class --------------------------------------
#
# The names are written out and then each is looked for on `$PATH`, rather than a list
# being derived from the image, because this script runs on a laptop as often as in the
# workspace and `$FOAM_APPBIN` is not there to glob. A name that is wrong therefore
# reports itself as "not found here" instead of becoming a false claim about the image.
# `compressible.SOLVERS` supplies its own three verbatim so the two cannot drift.


def Option(name: str, what: str, when: str) -> dict[str, str]:
    """One candidate solver: what it is, and the condition that makes it the one."""
    return {"name": name, "what": what, "when": when}


COMPRESSIBLE_SWITCH = (
    "M < 0.3 is where density varies by under 5% and the incompressible case answers "
    "the same question more cheaply (compressible.regime); at M >= 1 a pressure "
    "equation has no characteristic running upstream to carry it, which is where the "
    "density-based solvers are the only ones left"
)

FAMILIES: dict[str, tuple[Option, ...]] = {
    "incompressible": (
        Option("simpleFoam", "steady, incompressible, pressure-based (SIMPLE)",
               "a steady answer is the question and M < 0.3 -- case_gen.SOLVERS writes it"),
        Option("pimpleFoam", "transient, incompressible, pressure-based (PIMPLE)",
               "the answer is in the unsteadiness: shedding, a spectrum, a transient"),
        Option("potentialFoam", "one potential-flow solve, no viscosity, no turbulence",
               "not an answer -- an initial field, which is what it was used for on the "
               "cylinder case in this corpus (study 20260906-091113-f9cf, 3 s against "
               "173 s for the simpleFoam run it seeded)"),
    ),
    "compressible": tuple(
        Option(name, what, COMPRESSIBLE_SWITCH)
        for name, what in compressible.SOLVERS.items()
    ) + (
        Option("rhoPimpleFoam", "transient, compressible, pressure-based (PIMPLE)",
               "named in the image by compressible.py and deliberately without a "
               "dictionary writer there: at the transonic job it is worse than either "
               "density-based solver"),
        Option("sonicFoam", "transient, compressible, pressure-based, trans/supersonic",
               "the same -- in the image, no dictionary writer, superseded by the two "
               "density-based ones for shock capturing"),
    ),
    "free-surface": (
        Option("interFoam", "transient VOF, two incompressible phases, one momentum equation",
               "a free surface is in the answer -- gravity on and an alpha field present"),
        Option("interIsoFoam", "interFoam with the isoAdvector geometric interface scheme",
               "the interface is smearing across cells and sharpness is the thing being "
               "measured"),
        Option("compressibleInterFoam", "the same with both phases compressible",
               "a gas phase whose density is part of the answer -- a slam, a cavity "
               "collapse"),
    ),
    "conjugate-heat-transfer": (
        Option("chtMultiRegionFoam", "transient, solid and fluid regions solved together",
               "constant/regionProperties names more than one region and the transient "
               "conduction is in the answer"),
        Option("chtMultiRegionSimpleFoam", "the steady version of the same",
               "the steady temperature field is the question"),
    ),
    "stress-analysis": (
        Option("solidDisplacementFoam",
               "transient segregated linear-elastic small-strain deformation, with "
               "optional thermal diffusion and thermal stresses",
               "the answer is a stress or a deflection in a solid rather than "
               "anything about a fluid: a thermal stress, a pressure vessel, a "
               "bracket. `thermalStress yes` in constant/thermalProperties is what "
               "turns the temperature field on, and T there is a rise above the "
               "stress-free state because the solver subtracts no reference "
               "temperature. templates/thermal_stress.py writes a whole case and "
               "benchmarks/thermal_stress grades it against the closed form"),
        Option("solidEquilibriumDisplacementFoam",
               "the same constitutive law solved for steady equilibrium, with a "
               "steady-state displacement rather than a time loop",
               "the loading does not change and only the final deflection is "
               "wanted; small-strain linear elasticity either way"),
    ),
}
"""The candidates per family. `heat.py` measures what comes out of the conjugate ones
and `ladder.CHT_APPLICATIONS` is what recognises them on the way in; the names here are
the same two.

`stress-analysis` is the family nothing in this repository used to name, and the two
solvers in it have been in the image since the day it was first built -- the base is a
full ESI distribution and they are standard OpenFOAM, at
`applications/solvers/stressAnalysis/`. A capability this script cannot see is
invisible to whoever reads it, which is the only reason they were missing."""


# Which families a ladder class can reach. A class does not pick a solver on its own:
# an external aerodynamics case is incompressible or compressible depending on a Mach
# number that is not in the case files, so both are offered and the switch is stated.
CLASS_FAMILIES: dict[str, tuple[str, ...]] = {
    "external-aerodynamics": ("incompressible", "compressible"),
    "internal-flow": ("incompressible", "compressible"),
    "rotating-machinery": ("incompressible", "compressible"),
    "free-surface-marine": ("free-surface",),
    "conjugate-heat-transfer": ("conjugate-heat-transfer",),
    # `stress-analysis` reaches a case only through `generic`, and deliberately.
    # `ladder.classify` recognises a case from its flow signals -- a free stream, an
    # inlet, a region list -- and a solid body being heated has none of them, so it
    # comes back `generic`/"unrecognised" (ladder.py:539). That is the right place for
    # it: the family is offered where nothing else was recognised, which is exactly
    # the case a solid stress question presents as, and it is not offered beside
    # `interFoam` on a dam break where it would be noise.
    "generic": ("incompressible", "compressible", "free-surface",
                "conjugate-heat-transfer", "stress-analysis"),
}


def present(name: str) -> dict[str, Any]:
    """Whether this solver is on this machine, as a fact rather than an assumption.

    `compressible.available` for everything it knows -- it is the one that link-checks
    HiSA rather than trusting a binary that is there and whose `libhisa*.so` does not
    resolve -- and a `$PATH` lookup for the rest.
    """
    if name in compressible.SOLVERS:
        return compressible.available(name)
    found = shutil.which(name)
    return {
        "present": found is not None,
        "where": found or "not on $PATH here (the workspace image has its solvers at "
                          "$FOAM_APPBIN)",
    }


def options_for(class_key: str, mach: float | None) -> list[dict[str, Any]]:
    """The candidates for this class, each with whether it is actually here.

    When a Mach number is given, `compressible.choose_solver` is asked which of the
    compressible three it points at and that one is marked -- the same answer the
    dictionary writer would act on, rather than a second reading of the same number.
    """
    picked = ""
    if mach is not None:
        picked, _why = compressible.choose_solver("auto", mach, "steady")
    rows: list[dict[str, Any]] = []
    for family in CLASS_FAMILIES.get(class_key, CLASS_FAMILIES["generic"]):
        for option in FAMILIES[family]:
            row = dict(option)
            row["family"] = family
            row.update(present(option["name"]))
            row["mach_pick"] = bool(picked) and option["name"] == picked
            rows.append(row)
    return rows


def mach_of(case: preflight.Case, given: float | None) -> tuple[float | None, str]:
    """(Mach number, where it came from). None is an answer and is printed as one.

    A Mach number is not in a case's files. The velocity is, and the speed of sound is
    a property of the thermophysics, so a compressible case that was written by
    `compressible.py` has both and the number is recoverable; an incompressible case
    has no speed of sound at all and the honest answer is that nobody said. Inventing
    one here would invent the single input the solver comparison turns on.
    """
    if given is not None:
        return given, "--mach"
    thermo = case.read("constant/thermophysicalProperties")
    if not thermo.strip():
        return None, ("not in the case: no constant/thermophysicalProperties, so there "
                      "is no speed of sound to divide by (--mach says it)")
    return None, ("constant/thermophysicalProperties is here, and the Mach number still "
                  "needs the velocity it is taken against (--mach says it)")


# -- how many ranks ----------------------------------------------------------------


def cells_in(case: preflight.Case) -> tuple[int | None, str]:
    """(cells, where that came from). The built mesh if there is one, else the estimate.

    `preflight.Case.cell_count` reads the `owner` header note or a checkMesh log, which
    is the measured count. With no mesh yet the prediction comes from
    `cells_estimate.estimate` -- the documented arithmetic, order of magnitude, within
    about a factor of two for ordinary geometries -- and is labelled as a prediction,
    because a rank count chosen off a prediction and a rank count chosen off a mesh are
    not the same confidence.
    """
    measured = case.cell_count
    if measured:
        return measured, "the built mesh"

    block = case.read("system/blockMeshDict")
    delta0, volume, _extent = cells_estimate.block_mesh_delta(block)
    if not delta0:
        return None, ("no mesh and no readable system/blockMeshDict, so there is "
                      "nothing to size a decomposition against")
    snappy = case.read("system/snappyHexMeshDict")
    levels = cells_estimate.surface_levels(snappy)
    layers_match = cells_estimate.N_LAYERS.search(snappy)
    area = 0.0
    for relative in ("constant/triSurface", "constant/geometry"):
        directory = case.path / relative
        if directory.is_dir():
            for entry in sorted(directory.iterdir()):
                if entry.is_file() and entry.suffix.lower() in (".stl", ".stlb"):
                    area += cells_estimate.stl_area(entry)[0]
    result = cells_estimate.estimate(
        delta0, volume, levels, area, int(layers_match.group(1)) if layers_match else 0)
    total = int(result["total"])
    return (total or None), "cells_estimate.py, before the build (order of magnitude)"


def rank_rows(cells: int | None, cores: int) -> list[dict[str, Any]]:
    """Cells per rank at each rank count up to the cores there are.

    The powers of two plus the core count itself, because those are what a
    `decomposeParDict` actually gets set to and a row per integer would be a table
    nobody reads. No row is marked good or bad: the quantity this decision turns on is
    cells per rank, and printing it is the whole job -- `tools._load_per_rank` says the
    same thing at launch for the same reason.
    """
    if not cells or cores < 1:
        return []
    counts = sorted({1, *(2**n for n in range(1, 8) if 2**n <= cores), cores})
    return [{"ranks": n, "cells_per_rank": cells // n} for n in counts if n <= cores]


# -- what this case has already measured -------------------------------------------

BANNER = re.compile(r"^Build\s*:\s*", re.M)
EXEC_LINE = re.compile(r"ExecutionTime = ([0-9.eE+-]+) s\s+ClockTime = ([0-9.]+) s")
N_PROCS = re.compile(r"^nProcs\s*:\s*(\d+)", re.M)
EXEC_NAME = re.compile(r"^Exec\s*:\s*(\S+)", re.M)

MESH_TOOLS = frozenset({
    "blockMesh", "checkMesh", "snappyHexMesh", "surfaceFeatureExtract", "decomposePar",
    "reconstructPar", "reconstructParMesh", "extrudeMesh", "topoSet", "mapFields",
    "mapFieldsPar", "foamToVTK", "postProcess", "setFields", "transformPoints",
})


def runs_in(log: Path) -> list[dict[str, Any]]:
    """Every solver run in one log file, one record each.

    Split on OpenFOAM's `Build :` banner rather than taken whole, because `>` and `>>`
    both happen and a restarted case leaves two or three runs in one file: `case_N32`
    in this corpus has a `log.simpleFoam` whose last `ExecutionTime` is 376 s and whose
    `Time =` lines number 3,590, and the 376 s covers the 1,600 iterations after the
    restart. Reading that as one run overstates the iteration count by more than twice
    and understates the per-iteration cost by the same. (`log_digest.digest` reads it
    whole; this does not use it, and the two disagree on such a log.)

    Per-iteration cost is the median of the differences between successive
    `ExecutionTime` values, not total over count. The total carries the mesh read, the
    decomposition read and every field write, so on a short run it is mostly startup;
    the median of the steps is the cost of a step.
    """
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return []
    if "ExecutionTime" not in text:
        return []
    starts = [match.start() for match in BANNER.finditer(text)] or [0]
    bounds = starts + [len(text)]
    found: list[dict[str, Any]] = []
    for index in range(len(starts)):
        segment = text[bounds[index]:bounds[index + 1]]
        stamps = [float(execution) for execution, _clock in EXEC_LINE.findall(segment)]
        if len(stamps) < 3:
            # Two stamps give one difference and no median worth the name.
            continue
        name = EXEC_NAME.search(segment)
        application = (name.group(1) if name else "").split("/")[-1]
        if application in MESH_TOOLS:
            continue
        steps = [later - earlier
                 for earlier, later in zip(stamps, stamps[1:]) if later > earlier]
        if not steps:
            continue
        ranks = N_PROCS.search(segment)
        found.append({
            "log": log.name,
            "run": index + 1,
            "runs_in_file": len(starts),
            "application": application or None,
            "ranks": int(ranks.group(1)) if ranks else 1,
            "iterations": len(stamps),
            "median_s_per_iteration": round(statistics.median(steps), 4),
        })
    return found


def measured(case: preflight.Case) -> dict[str, Any]:
    """What this case's own logs measured, grouped by rank count.

    The case's own history is the only scaling evidence that is about this mesh and
    these schemes. Where it has been run at one rank count, that is said and no
    comparison is drawn from it -- one point is not a curve, and a ratio invented off
    it would be the fabrication this script exists to avoid.
    """
    runs = [run for log in sorted(case.path.glob("log.*")) for run in runs_in(log)]
    groups: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(run["ranks"], []).append(run)
    by_ranks = [
        {
            "ranks": ranks,
            "runs": len(rows),
            "iterations": sum(row["iterations"] for row in rows),
            "median_s_per_iteration": round(
                statistics.median([row["median_s_per_iteration"] for row in rows]), 4),
            "applications": sorted({row["application"] for row in rows if row["application"]}),
        }
        for ranks, rows in sorted(groups.items())
    ]
    out: dict[str, Any] = {"runs": runs, "by_ranks": by_ranks, "comparison": None}
    if len(by_ranks) >= 2:
        slowest, fastest = by_ranks[0], by_ranks[-1]
        if fastest["median_s_per_iteration"] > 0:
            out["comparison"] = {
                "from_ranks": slowest["ranks"],
                "to_ranks": fastest["ranks"],
                "rank_ratio": round(fastest["ranks"] / max(slowest["ranks"], 1), 2),
                "speedup": round(slowest["median_s_per_iteration"]
                                 / fastest["median_s_per_iteration"], 2),
                "runs": slowest["runs"] + fastest["runs"],
            }
    return out


# -- the report --------------------------------------------------------------------


def inspect(case_path: Path | str, mach: float | None = None,
            cores: int | None = None) -> dict[str, Any]:
    """Everything this script knows about one case, as plain data. `render` formats it."""
    case = preflight.Case(Path(case_path))
    signals = ladder.read_signals(case)
    detection = ladder.classify(case, signals)
    mach_value, mach_source = mach_of(case, mach)
    cells, cells_source = cells_in(case)
    core_count = cores if cores is not None else (os.cpu_count() or 0)
    return {
        "case": str(Path(case_path)),
        "class": detection.key,
        "class_name": detection.title,
        "generic": detection.generic,
        "reason": detection.reason,
        "application": case.application or None,
        "mach": mach_value,
        "mach_source": mach_source,
        "solvers": options_for(detection.key, mach_value),
        "cells": cells,
        "cells_source": cells_source,
        "cores": core_count,
        "cores_source": ("--cores" if cores is not None
                         else "os.cpu_count() on the machine this ran on, which is not "
                              "necessarily the machine the solver runs on -- `nproc` in "
                              "the workspace is the one that counts"),
        "ranks": rank_rows(cells, core_count),
        "measured": measured(case),
    }


def render(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# the case")
    lines.append(f"  {data['case']}")
    lines.append(f"  reads as: {data['class_name']}"
                 + (" (unrecognised)" if data["generic"] else ""))
    lines.append(f"  because: {data['reason']}")
    lines.append(f"  application in controlDict: {data['application'] or 'none named'}")

    lines.append("")
    lines.append("# solvers in this image for that")
    families = sorted({row["family"] for row in data["solvers"]})
    if data["mach"] is None:
        lines.append(f"  Mach number: not established -- {data['mach_source']}")
        if {"incompressible", "compressible"} <= set(families):
            lines.append("  so the incompressible and the compressible candidates are "
                         "both below, undivided")
    elif "compressible" not in families:
        # Said rather than left out: a Mach number was given and this class does not
        # branch on it, and silence there reads as the number having been used.
        label, _meaning = compressible.regime(data["mach"])
        lines.append(f"  Mach number: {data['mach']:.3f} ({data['mach_source']}) -- "
                     f"{label}, and it decides nothing here: the candidates for a "
                     f"{data['class_name']} case do not divide on it")
    else:
        label, meaning = compressible.regime(data["mach"])
        lines.append(f"  Mach number: {data['mach']:.3f} ({data['mach_source']}) -- "
                     f"{label}, {meaning}")
    for row in data["solvers"]:
        mark = " <- choose_solver points here" if row["mach_pick"] else ""
        here = "on PATH" if row["present"] else "not found here"
        lines.append(f"  {row['name']:<24} [{here}]{mark}")
        lines.append(f"      is:   {row['what']}")
        lines.append(f"      when: {row['when']}")

    lines.append("")
    lines.append("# ranks")
    if not data["cells"]:
        lines.append(f"  cells: unknown -- {data['cells_source']}")
    else:
        lines.append(f"  cells: {data['cells']:,} ({data['cells_source']})")
    lines.append(f"  cores: {data['cores'] or 'unknown'} ({data['cores_source']})")
    for row in data["ranks"]:
        lines.append(f"    -np {row['ranks']:<4} {row['cells_per_rank']:>12,} cells each")
    if data["ranks"]:
        lines.append(f"  past {data['cores']} ranks the ranks share cores. In study "
                     "20260830-164816-eee0 a 1,793,894-cell case was run 12 ways on a "
                     "container whose `nproc` answered 4; mpirun did not refuse, and the "
                     "12-rank run's median 1.45 s/iteration sat at the bottom of the "
                     "1.45-1.77 s band its four 4-rank stages had already measured.")

    lines.append("")
    lines.append("# what this case has measured")
    evidence = data["measured"]
    if not evidence["by_ranks"]:
        lines.append("  no solver run with three or more ExecutionTime lines in this "
                     "directory, so there is nothing measured here to report")
    for row in evidence["by_ranks"]:
        apps = ", ".join(row["applications"]) or "unnamed"
        lines.append(f"  {row['ranks']:>3} rank(s): median {row['median_s_per_iteration']:g} "
                     f"s/iteration over {row['runs']} run(s), {row['iterations']:,} "
                     f"iterations ({apps})")
    comparison = evidence["comparison"]
    if comparison is None and evidence["by_ranks"]:
        lines.append("  one rank count only, so there is no speedup to report from this "
                     "case -- a ratio off one point would be invented")
    elif comparison:
        lines.append(f"  {comparison['from_ranks']} -> {comparison['to_ranks']} ranks is "
                     f"{comparison['rank_ratio']}x the ranks for {comparison['speedup']}x "
                     f"the speed, over {comparison['runs']} run(s). Whether the mesh and "
                     "the schemes were the same across those runs is not read here, and "
                     "on the one case in the corpus that had two rank counts neither was: "
                     "its 2-rank runs were on a 1,735,460-cell mesh and its 4- and "
                     "12-rank runs on a 1,793,894-cell one, through four stages of "
                     "scheme changes.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path)
    parser.add_argument("--mach", type=float, default=None,
                        help="the Mach number, which is not in the case files")
    parser.add_argument("--cores", type=int, default=None,
                        help="cores on the machine the solver will run on (`nproc` there)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    data = inspect(args.case, mach=args.mach, cores=args.cores)
    print(json.dumps(data, indent=2, default=str) if args.json else render(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
