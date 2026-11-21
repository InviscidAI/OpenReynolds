#!/usr/bin/env python3
"""Which solver, how many ranks, and whether this machine can finish it.

Three questions that get answered from memory and are cheap to answer from the case:
which of the solvers available here suits the problem, what `-np` is worth asking for,
and whether the mesh about to be built fits in the memory and the hours there are.
"Available here" is the image's applications and the two solvers that are scripts in
this toolbox (`lbm.py` and `sph.py`), which are listed beside the rest because the
question is what can solve the case and not where the solver is installed.
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

the machine, which is the other half of the same question

    A cell count means nothing without the machine it runs on, and the machine is the
    one thing a session reliably gets wrong: `nproc` counts the host's processors, so a
    four-core container is told sixteen, and nothing anywhere was reading memory at all.
    `machine()` reads cores three ways (`os.cpu_count`, `os.sched_getaffinity`, the
    cgroup CPU quota) and memory on Linux, macOS and Windows with the standard library
    and no new dependency. Every probe is allowed to answer "no", and a "no" prints as
    not established rather than as a default.

    What a cell costs is then two measurements out of the corpus, both with their sample
    size in the output: time per cell per iteration from 101 solver run segments
    (`COST_PER_CELL_S`), and memory per cell from the single run in the whole corpus that
    anybody measured the memory of (`MEMORY_PER_CELL_B`).

what is not claimed

    No speedup curve. The corpus behind this script -- 128 local study directories, 137
    solver-log run segments when that was written, and 130 directories and 138 segments
    by the same function on 2026-11-20 -- has exactly one case that was run at more than
    one rank count, and that one was oversubscribed. Cells per rank is printed as
    arithmetic, not as a verdict against a threshold, because no threshold in here would
    trace to a measurement. Where this case has been run before, its own logs are read
    and what they measured is reported with the number of runs behind it.

    No memory limit. One case in the corpus has a solver's resident memory beside a cell
    count, so the arithmetic off it is printed as arithmetic off one case and no ceiling
    is promised. Within that, no per-rank correction either: one rank count cannot
    separate the per-process constant from the per-cell slope.

    No laptop calibration. Every timing in the corpus was measured on a hosted
    container, 4 to 16 cores. A slower core than those is outside the printed band, not
    inside it, and that is said where the band is printed rather than left to be assumed.

Nothing here edits, refuses or blocks. It writes no case, runs no solver, and the exit
code is 0 whatever the report says -- a case that will not fit is reported as not
fitting, beside the cell counts that would, and the decision stays where it was. A
solver named below and not found -- on `$PATH` for an application, beside this file
for a toolbox script -- is reported as not found rather than left out, because "this
image does not have it" and "nobody looked" are different answers.

    python3 advise.py /work/case
    python3 advise.py /work/case --mach 0.84 --cores 8
    python3 advise.py /work/case --cores 4 --memory-gb 8
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


def Option(name: str, what: str, when: str, script: str = "") -> dict[str, str]:
    """One candidate solver: what it is, and the condition that makes it the one.

    `script` names a solver that is a toolbox script beside this file rather than a
    binary on `$PATH`. Both kinds belong in the same comparison -- the question a session
    asks is "what can solve this", and "it is a file in /work/.toolbox rather than a
    compiled application in $FOAM_APPBIN" is an answer to a different question.
    """
    return {"name": name, "what": what, "when": when, "script": script}


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
    # The two solvers that are not in the image. They are numpy scripts in this
    # directory, so they travel with the package rather than with the workspace, and
    # they are in this table for the reason the table exists: a solver a session cannot
    # see is a solver the product does not have.
    "lattice-boltzmann": (
        Option("lbm.py", "D2Q9 two-relaxation-time lattice Boltzmann in numpy, a script "
               "in this toolbox rather than an application in the image",
               "the geometry is a solid mask rather than a mesh -- an awkward passage, a "
               "porous sample, anything voxelised -- and the meshing step is the "
               "expensive part of the question. It is explicit, weakly compressible, "
               "second order, and carries no turbulence model at all, so a transonic "
               "case or a high-Reynolds one belongs to the families above; "
               "`benchmarks/lbm/` has what it measured against Poiseuille and Ghia",
               script="lbm.py"),
    ),
    "particle": (
        Option("sph.py", "2D weakly compressible smoothed-particle hydrodynamics in "
               "numpy, a script in this toolbox rather than an application in the image",
               "a free surface that breaks, folds or separates, with no mesh to deform "
               "through it -- a collapsing column, a slosh, a wave impact. interFoam "
               "above solves the same physics and resolves a thin interface better once "
               "a mesh exists; this needs none. The time step is acoustic, so a second "
               "of motion is thousands of steps however few particles there are, and "
               "`benchmarks/sph/` has both what it was graded on and what it was not",
               script="sph.py"),
    ),
    "mhd": (
        Option("mhdFoam", "transient, incompressible, with the Lorentz force on the "
               "momentum equation and B transported beside U",
               "a conducting liquid in an imposed magnetic field, where the field is "
               "part of the answer rather than a boundary condition: a liquid-metal "
               "duct, an electromagnetic pump, a Hartmann layer. The Hartmann number "
               "Ha = B L sqrt(sigma/(rho nu)) sets the wall layer at L/Ha, and that "
               "layer is what fixes the cost -- `benchmarks/mhd/` has the closed form, "
               "what a run measured against it, and the run that diverged first and "
               "why. `mhd_hartmann.py` writes the case"),
    ),
    "porous-media": (
        Option("porousSimpleFoam", "steady, incompressible, with a Darcy-Forchheimer "
               "sink over a cell zone",
               "a pressure drop through a region you do not want to mesh -- a packed "
               "bed, a filter, a heat-exchanger core, a screen. The sink is "
               "dp = (mu d U + 0.5 rho f U^2) L on the SUPERFICIAL (face) velocity and "
               "the thickness along the flow, so a drop quoted without its thickness is "
               "half a measurement. `porous.py` writes the case and prints the "
               "coefficients; its arithmetic is graded in "
               "tests/test_toolbox_rotating_porous.py, and the solve is not graded "
               "anywhere in this repository yet"),
    ),
    "combustion": (
        Option("chemFoam", "one cell, no transport: chemistry integrated at constant "
               "pressure or constant volume",
               "the question is about the chemistry and not about a flow -- an ignition "
               "delay, an equilibrium temperature, whether a mechanism behaves. "
               "`createFields.H` refuses a mesh with more than one cell, so there is no "
               "geometry, no boundary condition and no turbulence-chemistry interaction "
               "in it at all. It is NOT a flame: what it conserves is the mixture's "
               "absolute enthalpy, so the temperature it reaches is the "
               "complete-combustion one and a real flame is cooler. `chem0d.py` writes "
               "the case and grades it; `benchmarks/combustion/` says what is graded "
               "and what is deliberately not"),
        Option("reactingFoam", "transient, compressible, species transport with "
               "combustion and a turbulence-chemistry model",
               "an actual flame, where the question is also where the heat goes and not "
               "only how much there is: a burner, a furnace, a lifted flame. In the "
               "image and with NO case in this repository -- no template writes one, "
               "nothing grades one, and chem0d.py deliberately copies its thermo data "
               "out of this solver's tutorial rather than running it. Offered because "
               "it is the honest answer to a flame question, and its absence here is "
               "said plainly so nobody reads it as a validated route"),
    ),
    "potential-flow": (
        Option("panel.py", "a linear-strength vortex panel method for aerofoils and a "
               "lifting-line solve for wings, in numpy, a script in this toolbox "
               "rather than an application in the image",
               "lift, moment or induced drag on a thin lifting surface at small angle, "
               "wanted in seconds and with no mesh -- a section comparison, an angle "
               "sweep, a first wing. It is inviscid with no Reynolds number in it, so "
               "it CANNOT answer a drag question: d'Alembert makes the pressure drag of "
               "a closed body exactly zero and the solver reproduces that to 2.19e-04 "
               "rather than estimating a real drag. Stall, separation and thick "
               "sections belong to the families above; `openreynolds/toolbox/panel.py "
               "--check` grades the statements that are exact",
               script="panel.py"),
    ),
    "particle-tracking": (
        Option("tracks.py", "particles integrated through a flow field that is already "
               "solved, with Stokes or Schiller-Naumann drag, in numpy",
               "where the carried thing goes, once the carrying flow is known -- "
               "deposition, capture on a body, residence time, a tracer. It is one-way: "
               "the particles do not act back on the flow, so a loaded jet or a dense "
               "suspension is a different question. `benchmarks/particles/` grades it "
               "against two exact closed forms and one published capture threshold, and "
               "the three cases are in the tutorial index's tier 1",
               script="tracks.py"),
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
    "external-aerodynamics": ("incompressible", "compressible", "lattice-boltzmann",
                              "potential-flow", "particle-tracking"),
    "internal-flow": ("incompressible", "compressible", "lattice-boltzmann",
                      "mhd", "porous-media", "particle-tracking"),
    # Not rotating machinery: both toolbox solvers have static boundaries, and neither
    # an MRF zone nor a sliding interface exists in either of them.
    "rotating-machinery": ("incompressible", "compressible"),
    "free-surface-marine": ("free-surface", "particle"),
    "conjugate-heat-transfer": ("conjugate-heat-transfer",),
    # `stress-analysis` reaches a case only through `generic`, and deliberately.
    # `ladder.classify` recognises a case from its flow signals -- a free stream, an
    # inlet, a region list -- and a solid body being heated has none of them, so it
    # comes back `generic`/"unrecognised" (ladder.py:539). That is the right place for
    # it: the family is offered where nothing else was recognised, which is exactly
    # the case a solid stress question presents as, and it is not offered beside
    # `interFoam` on a dam break where it would be noise.
    # `combustion` reaches a case only through `generic`, for the reason
    # `stress-analysis` does: a one-cell reactor has no flow signal for
    # `ladder.classify` to recognise, so it arrives here, and offering chemFoam beside
    # interFoam on a dam break would be noise.
    "generic": ("incompressible", "compressible", "free-surface",
                "conjugate-heat-transfer", "stress-analysis",
                "lattice-boltzmann", "particle",
                "mhd", "porous-media", "combustion", "potential-flow",
                "particle-tracking"),
}


def present(name: str, script: str = "") -> dict[str, Any]:
    """Whether this solver is on this machine, as a fact rather than an assumption.

    `compressible.available` for everything it knows -- it is the one that link-checks
    HiSA rather than trusting a binary that is there and whose `libhisa*.so` does not
    resolve -- a file check for the toolbox's own solvers, and a `$PATH` lookup for the
    rest. A toolbox solver travels with this file, so it is here whenever this is.
    """
    if script:
        path = Path(__file__).resolve().parent / script
        return {
            "present": path.is_file(),
            "label": "in this toolbox",
            "where": str(path) if path.is_file()
                     else f"{script} is not beside advise.py, so this toolbox is partial",
        }
    if name in compressible.SOLVERS:
        return {"label": "on PATH", **compressible.available(name)}
    found = shutil.which(name)
    return {
        "present": found is not None,
        "label": "on PATH",
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
            row.update(present(option["name"], option.get("script", "")))
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


# -- this machine ------------------------------------------------------------------
#
# Standard library only, and every probe may answer "no". A laptop is the case this
# section exists for, and the one thing worse than not knowing its memory is guessing.


def _text(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def _kb(meminfo: str, key: str) -> int | None:
    """A `/proc/meminfo` row, in bytes. Its unit is kB and its name says so."""
    found = re.search(rf"^{key}:\s+(\d+)\s*kB", meminfo, re.M)
    return int(found.group(1)) * 1024 if found else None


def _cgroup_memory() -> tuple[int | None, str]:
    """A container's own memory ceiling, which `/proc/meminfo` does not show.

    `/proc/meminfo` inside a container is the host's, so a 2 GiB container on a 64 GiB
    host reads 64 GiB there and is killed at 2. cgroup v2 spells no limit as the word
    `max`; v1 spells it as a number near 2**63, which is why the upper bound is checked
    rather than only the word.
    """
    for path, note in (
        ("/sys/fs/cgroup/memory.max", "cgroup v2 memory.max"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "cgroup v1 memory.limit_in_bytes"),
    ):
        raw = _text(path).strip()
        if raw.isdigit() and 0 < int(raw) < 2**53:
            return int(raw), note
    return None, ""


def _cgroup_cores() -> tuple[float | None, str]:
    """The CPU quota as a number of cores, where a cgroup sets one.

    This is the gap the corpus walked into: `nproc` counts the host's processors, and a
    container with a quota of four is still told sixteen. v2 puts `quota period` in one
    file; v1 puts them in two, with -1 for no quota.
    """
    v2 = _text("/sys/fs/cgroup/cpu.max").split()
    if len(v2) == 2 and v2[0].isdigit() and v2[1].isdigit() and int(v2[1]) > 0:
        return int(v2[0]) / int(v2[1]), "cgroup v2 cpu.max"
    quota = _text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").strip()
    period = _text("/sys/fs/cgroup/cpu/cpu.cfs_period_us").strip()
    if quota.isdigit() and period.isdigit() and int(period) > 0:
        return int(quota) / int(period), "cgroup v1 cpu.cfs_quota_us/cpu.cfs_period_us"
    return None, ""


def _windows_memory() -> tuple[int | None, int | None]:
    """(total, available) from GlobalMemoryStatusEx. `ctypes` is in the standard library."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except Exception:  # noqa: BLE001 - not knowing is an answer this prints
        return None, None


def machine() -> dict[str, Any]:
    """Cores and memory on the machine this is running on, each with its source.

    Three platforms, three readings, no dependency: `/proc/meminfo` on Linux,
    `GlobalMemoryStatusEx` through `ctypes` on Windows, and `os.sysconf` for the BSDs
    and macOS, which have `SC_PHYS_PAGES` where Windows has no `sysconf` at all. Any of
    them may fail, and a `None` here prints as "not established" rather than as a
    default -- a budget against an assumed memory size is the fabrication this whole
    section is trying not to be.

    Cores are reported twice over, because the two numbers differ on exactly the machine
    that matters. `os.cpu_count()` is the processors the OS can see;
    `os.sched_getaffinity` is the ones this process may use, and a cgroup CPU quota is a
    third answer that neither of them reflects.
    """
    logical = os.cpu_count() or 0
    sources = [f"os.cpu_count() says {logical}"]
    usable = float(logical)
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
        sources.append(f"os.sched_getaffinity says {affinity}")
        usable = min(usable, affinity) if affinity else usable
    quota, quota_note = _cgroup_cores()
    if quota:
        sources.append(f"{quota_note} works out at {quota:g}")
        usable = min(usable, quota)

    total = available = None
    memory_source = ""
    meminfo = _text("/proc/meminfo")
    if meminfo:
        total, available = _kb(meminfo, "MemTotal"), _kb(meminfo, "MemAvailable")
        memory_source = "/proc/meminfo"
    elif sys.platform == "win32":
        total, available = _windows_memory()
        memory_source = "GlobalMemoryStatusEx" if total else ""
    elif hasattr(os, "sysconf") and "SC_PHYS_PAGES" in getattr(os, "sysconf_names", {}):
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            memory_source = "os.sysconf(SC_PAGE_SIZE x SC_PHYS_PAGES)"
        except (OSError, ValueError):
            total = None

    ceiling, ceiling_note = _cgroup_memory()
    if ceiling and (total is None or ceiling < total):
        memory_source = (f"{ceiling_note} -- a container ceiling under the "
                         f"{total / 2**30:.1f} GiB the host reports" if total
                         else ceiling_note)
        total = ceiling
    if available is not None and total is not None:
        available = min(available, total)

    return {
        "platform": sys.platform,
        "cores_logical": logical,
        # A CPU quota can be fractional -- half a core is a real container shape -- and
        # one rank is still the floor of what runs, so it rounds up to 1 rather than to 0.
        "cores_usable": max(1, int(usable)) if usable > 0 else 0,
        "cores_source": "; ".join(sources),
        "memory_bytes": total,
        "memory_available_bytes": available,
        "memory_source": memory_source or (
            f"not established on {sys.platform}: no /proc/meminfo, no sysconf "
            "SC_PHYS_PAGES, and no GlobalMemoryStatusEx (--memory-gb says it)"),
    }


# -- what a cell costs, measured ---------------------------------------------------
#
# Mined on 2026-11-20 out of the 425 `log.*` files under the 130 study directories in
# `data/qa-runs/studies` and `OpenReynolds/studies`, with `runs_in` below doing the
# splitting and `preflight.Case.cell_count` supplying the mesh size: 101 solver run
# segments in 50 of the 104 case directories that hold a `system/controlDict`. The other
# 54 directories have no solver log with three or more `ExecutionTime` lines in them.
#
# 101 rather than the 138 segments the whole corpus holds, because a segment is only
# usable here if a cell count attaches to it, and that means its log has to sit in a case
# directory that still has its mesh or its checkMesh log. The 37 left out are in
# `*.log` files and `.foamd/exec` spool copies away from any case.

COST_SOURCE = ("101 solver run segments in 50 case directories under studies/, at 1, 2, "
               "4, 8 and 12 ranks and 400 to 1,793,894 cells")

COST_PER_CELL_S: dict[str, dict[str, Any]] = {
    "simpleFoam":     {"p10": 2.512e-06, "median": 3.767e-06, "p90": 5.913e-06,
                       "runs": 14, "cases": 12},
    "rhoSimpleFoam":  {"p10": 2.083e-06, "median": 3.635e-06, "p90": 5.162e-06,
                       "runs": 15, "cases": 4},
    "rhoCentralFoam": {"p10": 2.747e-06, "median": 5.036e-06, "p90": 1.213e-05,
                       "runs": 11, "cases": 3},
    "interFoam":      {"p10": 9.470e-06, "median": 1.333e-05, "p90": 5.707e-05,
                       "runs": 20, "cases": 5},
    "pimpleFoam":     {"p10": 8.180e-06, "median": 1.476e-05, "p90": 2.250e-05,
                       "runs": 36, "cases": 22},
    "": {"p10": 3.126e-06, "median": 9.700e-06, "p90": 2.375e-05,
         "runs": 101, "cases": 50},
}
"""Seconds per iteration per cell *held by one rank*, as p10, median and p90.

Two things in this table are measurements and the third is a choice.

The measurement that picked the denominator: over the same 101 runs, dividing the median
s/iteration by the total cell count spreads p10 to p90 over 21.6x, and dividing it by
cells-per-rank spreads it over 7.6x. Cells per rank is the better predictor of the two by
a factor of nearly three, so it is the one indexed on. That is not a claim that twice the
ranks is half the time -- the corpus cannot support one (see "what is not claimed") --
it is the measured cost of one rank carrying a given load.

The measurement per solver: a steady pressure-based iteration costs about a quarter of a
transient one per cell, which is the difference between the simpleFoam/rhoSimpleFoam rows
near 3.7e-06 and the pimpleFoam/interFoam rows near 1.4e-05.

The choice: only solvers with ten or more run segments get a row, because below that the
p10 and p90 are the smallest and largest of a handful rather than a band. The six other
applications in the corpus have one or two runs each and fall back to the `""` row, which
is every run in the corpus and is wide for exactly that reason.

Every number here was measured on hosted containers -- Modal and EC2, 4 to 16 cores --
so it is the cost of a core there, not of a core in the laptop this may be running on.
"""

MEMORY_PER_CELL_B = (1700.0, 1730.0)
"""Bytes of resident memory per cell held by a rank. ONE case, which is why it is a pair.

    Study 20260830-164816-eee0 (the ONERA M6 replication), 2026-08-30. `rhoSimpleFoam
    -parallel` started 10:14:53 on a 1,735,460-cell mesh at nProcs 2
    (`log.rhoSimpleFoam_resume3`, and `log.checkMesh` for the count). At 10:16:31 a `ps
    aux` in the same container caught PIDs 10382 and 10383 -- the two PIDs that log's own
    banner names -- at 1,501,420 kB and 1,475,596 kB resident. Against 867,730 cells per
    rank that is 1,730 and 1,700 bytes per cell.

That is the whole of it. The corpus has 20 other memory readings, in 11 studies, and
every one is a `free` at idle with between 1.9 MiB and 354 MiB in use, which measures
nothing about a solver. Twelve further solver ranks have a readable RSS (study
20260830-032052-6fe9: three `mpirun -np 4 simpleFoam` groups at 275 to 442 MB a rank)
and belong to no launch in that study's transcript -- its own runs were rhoCentralFoam,
it eventually `pkill`ed them, and the container was shared -- so no cell count attaches
to them and they are not used here.

Two things are therefore refused below rather than predicted. A limit: one case cannot
locate a per-cell slope, so the arithmetic off this pair is printed as arithmetic off one
case and no cell ceiling is promised. And a rank correction: the one case with a memory
reading was run at one rank count, so the per-process constant (libraries, halo layers)
cannot be separated from the per-cell slope, and what a different rank count would cost
in total is not stated.
"""


def cost_band(application: str) -> tuple[dict[str, Any], str]:
    """The band for this application, or the whole corpus with a note saying so."""
    row = COST_PER_CELL_S.get(application.split("/")[-1])
    if row:
        return row, f"{row['runs']} {application} run(s) in {row['cases']} case(s)"
    corpus = COST_PER_CELL_S[""]
    named = f"{application} " if application else "no application named in controlDict, so "
    return corpus, (f"{named}has under ten runs in the corpus; this is every solver in "
                    f"it, {corpus['runs']} runs in {corpus['cases']} cases")


def iterations_of(case: preflight.Case) -> tuple[int | None, str]:
    """(iterations the controlDict asks for, where that came from). None is an answer.

    An adjustable time step is the honest None: the count then depends on a Courant
    number the solver works out per step from a field nobody has yet.
    """
    control = case.control
    if control.get("adjustTimeStep", "no").strip().startswith(("y", "t", "on")):
        return None, ("adjustTimeStep is on, so deltaT is the solver's to choose and the "
                      "number of steps is not in the case")
    end = preflight.as_float(control.get("endTime"))
    step = preflight.as_float(control.get("deltaT"))
    start = preflight.as_float(control.get("startTime")) or 0.0
    if not end or not step or step <= 0 or end <= start:
        return None, "endTime and deltaT do not give a step count in system/controlDict"
    source = f"endTime {end:g}, deltaT {step:g}"
    if start:
        source += f", startTime {start:g}"
    if control.get("startFrom", "").strip() == "latestTime":
        source += " (startFrom latestTime, so a restart has fewer left than this)"
    return int((end - start) / step), source


def budget(cells: int | None, ranks: int, application: str,
           iterations: int | None) -> dict[str, Any]:
    """Seconds an iteration and hours in total, as the measured band for this load."""
    row, basis = cost_band(application)
    if not cells or ranks < 1:
        return {"basis": basis}
    per_rank = cells / ranks
    low, high = row["p10"] * per_rank, row["p90"] * per_rank
    out = {"basis": basis, "s_low": low, "s_high": high,
           "s_median": row["median"] * per_rank}
    if iterations:
        out["hours_low"] = low * iterations / 3600.0
        out["hours_high"] = high * iterations / 3600.0
    return out


def what_would_fit(cells: int | None, memory_bytes: int | None,
                   case: preflight.Case) -> dict[str, Any]:
    """The same arithmetic run backwards: the cell counts the numbers above allow.

    Nothing here is a new model. The memory row inverts the one measured pair. The
    coarser-mesh rows are `cells_estimate`'s own background arithmetic -- the background
    count is volume / delta0**3, so a base cell f times larger is f**3 times fewer cells,
    and a one-cell-thick domain is the depth in cells times fewer. Applied to a snappy
    mesh they are an over-estimate of the saving in one known direction: a refinement
    band goes as area / delta**2, the square rather than the cube, so a mesh whose count
    is mostly refinement band shrinks by less than these rows say. The direction is
    stated rather than corrected, because correcting it needs a second mesh to calibrate
    against and the corpus has none at two base cell sizes.
    """
    out: dict[str, Any] = {}
    if memory_bytes:
        out["holds_cells"] = (int(memory_bytes / MEMORY_PER_CELL_B[1]),
                              int(memory_bytes / MEMORY_PER_CELL_B[0]))
    if cells:
        out["wants_bytes"] = (int(cells * MEMORY_PER_CELL_B[0]),
                              int(cells * MEMORY_PER_CELL_B[1]))
    if not (cells and out.get("holds_cells")):
        return out
    # The alternatives are printed when the case is larger than the memory that was
    # actually read, and not otherwise: a list of smaller meshes under a case that fits
    # is noise, and "it fits" is the answer there.
    holds = out["holds_cells"][0]
    out["fits"] = cells <= holds
    if out["fits"]:
        return out
    out["base_cell_factor"] = (cells / holds) ** (1 / 3)
    delta0, _volume, extent = cells_estimate.block_mesh_delta(
        case.read("system/blockMeshDict"))
    if delta0 and extent and min(extent) > 0:
        depth = min(extent) / delta0
        out["coarser"] = [("a base cell twice the size, in three dimensions", cells // 8)]
        if depth >= 2:
            out["coarser"].append(
                (f"one cell thick instead of the {depth:.0f} the shortest side has now",
                 int(cells / depth)))
    return out


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
            cores: int | None = None, memory_gb: float | None = None) -> dict[str, Any]:
    """Everything this script knows about one case, as plain data. `render` formats it."""
    case = preflight.Case(Path(case_path))
    signals = ladder.read_signals(case)
    detection = ladder.classify(case, signals)
    mach_value, mach_source = mach_of(case, mach)
    cells, cells_source = cells_in(case)
    here = machine()
    core_count = cores if cores is not None else (
        here["cores_usable"] or here["cores_logical"])
    memory_bytes = int(memory_gb * 2**30) if memory_gb else here["memory_bytes"]
    iterations, iterations_source = iterations_of(case)
    application = case.application or ""
    rows = rank_rows(cells, core_count)
    for row in rows:
        row.update(budget(cells, row["ranks"], application, iterations))
    return {
        "machine": here,
        "memory_bytes": memory_bytes,
        "memory_source": ("--memory-gb" if memory_gb else here["memory_source"]),
        "iterations": iterations,
        "iterations_source": iterations_source,
        "cost_basis": cost_band(application)[1],
        "cost_source": COST_SOURCE,
        "fit": what_would_fit(cells, memory_bytes, case),
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
                         else "the usable count from the machine above, which is not "
                              "necessarily the machine the solver runs on -- `nproc` in "
                              "the workspace is the one that counts"),
        "ranks": rows,
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
        here = row.get("label", "on PATH") if row["present"] else "not found here"
        lines.append(f"  {row['name']:<24} [{here}]{mark}")
        lines.append(f"      is:   {row['what']}")
        lines.append(f"      when: {row['when']}")

    lines.append("")
    lines.append("# this machine")
    here = data["machine"]
    lines.append(f"  cores: {here['cores_usable'] or 'unknown'} usable "
                 f"({here['cores_source']})")
    if data["memory_bytes"]:
        gib = data["memory_bytes"] / 2**30
        spare = here["memory_available_bytes"]
        free = f", {spare / 2**30:.1f} GiB of it free right now" if spare else ""
        lines.append(f"  memory: {gib:.1f} GiB ({data['memory_source']}){free}")
    else:
        lines.append(f"  memory: {data['memory_source']}")

    lines.append("")
    lines.append("# ranks, and what an iteration costs at that load")
    if not data["cells"]:
        lines.append(f"  cells: unknown -- {data['cells_source']}")
    else:
        lines.append(f"  cells: {data['cells']:,} ({data['cells_source']})")
    lines.append(f"  cores: {data['cores'] or 'unknown'} ({data['cores_source']})")
    if data["iterations"]:
        lines.append(f"  iterations asked for: {data['iterations']:,} "
                     f"({data['iterations_source']})")
    else:
        lines.append(f"  iterations asked for: not established -- "
                     f"{data['iterations_source']}, so the hours below are left out")
    for row in data["ranks"]:
        line = f"    -np {row['ranks']:<4} {row['cells_per_rank']:>12,} cells each"
        if "s_low" in row:
            line += f"   {row['s_low']:>8.3g} to {row['s_high']:<8.3g} s/iteration"
        if "hours_low" in row:
            line += f"   {row['hours_low']:>7.3g} to {row['hours_high']:<7.3g} h"
        lines.append(line)
    if data["ranks"] and data["cells"]:
        lines.append(f"  the band is p10 to p90 of {data['cost_basis']}, taken over "
                     f"{data['cost_source']}, as seconds per iteration per cell held by "
                     "one rank. Dividing by cells-per-rank rather than by total cells is "
                     "what the corpus supports: it narrows p10-to-p90 from 21.6x to 7.6x. "
                     "It is not a claim that twice the ranks is half the time -- it is "
                     "the measured cost of a rank carrying that many cells. Every run in "
                     "it was on a hosted container, 4 to 16 cores, so a slower core here "
                     "is outside the band rather than inside it.")
    if data["ranks"]:
        lines.append(f"  past {data['cores']} ranks the ranks share cores. In study "
                     "20260830-164816-eee0 a 1,793,894-cell case was run 12 ways on a "
                     "container whose `nproc` answered 4; mpirun did not refuse, and the "
                     "12-rank run's median 1.45 s/iteration sat at the bottom of the "
                     "1.45-1.77 s band its four 4-rank stages had already measured.")

    lines.append("")
    lines.append("# memory, from the one case in the corpus that measured any")
    fit = data["fit"]
    low, high = MEMORY_PER_CELL_B
    lines.append(f"  the measurement: {low:.0f} and {high:.0f} bytes per cell, from the "
                 "two ranks of one rhoSimpleFoam run on a 1,735,460-cell mesh (study "
                 "20260830-164816-eee0, PIDs 10382 and 10383 at 1,501,420 and 1,475,596 "
                 "kB resident). One case, one snapshot. It locates no slope, so no cell "
                 "limit is promised here and what follows is arithmetic off that pair, "
                 "labelled as such.")
    if "wants_bytes" in fit:
        lines.append(f"  this case at {data['cells']:,} cells would want "
                     f"{fit['wants_bytes'][0] / 2**30:.2f} to "
                     f"{fit['wants_bytes'][1] / 2**30:.2f} GiB at that rate")
    if "holds_cells" in fit:
        lines.append(f"  {data['memory_bytes'] / 2**30:.1f} GiB at that rate is about "
                     f"{fit['holds_cells'][0]:,} to {fit['holds_cells'][1]:,} cells")
    if fit.get("fits"):
        lines.append("  so this case is the smaller of the two, by that arithmetic")
    if "base_cell_factor" in fit:
        lines.append(f"  the case is larger than that. Cells go as the cube of the base "
                     f"cell size in three dimensions, so a base cell about "
                     f"{fit['base_cell_factor']:.2f}x larger is about the count that "
                     "would fit, with the refinement left as it is")
    for label, count in fit.get("coarser", []):
        lines.append(f"  {label}: about {count:,} cells")
    if fit.get("coarser"):
        lines.append("  those two are the background mesh's own arithmetic (volume / "
                     "delta0^3) carried across to the whole count. A refinement band goes "
                     "as the square of the cell size rather than the cube, so a mesh that "
                     "is mostly band shrinks by less than they say.")
    lines.append("  the per-process part of that memory -- the libraries, the halo layer "
                 "-- is not separated from the per-cell part, because the one case with a "
                 "memory reading was run at one rank count. So the total at a different "
                 "rank count is not stated here. Ranks and memory were not measured "
                 "together anywhere in this corpus.")

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
    parser.add_argument("--memory-gb", type=float, default=None, dest="memory_gb",
                        help="memory on that machine, in GiB, where this cannot read it")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    data = inspect(args.case, mach=args.mach, cores=args.cores,
                   memory_gb=args.memory_gb)
    print(json.dumps(data, indent=2, default=str) if args.json else render(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
