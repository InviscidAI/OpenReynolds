#!/usr/bin/env python3
"""Is the adjoint optimisation solver on this instance, and what does a case need?

    python3 adjoint_env.py                  # present or absent, and the tutorials
    python3 adjoint_env.py --json           # the same facts as JSON
    python3 adjoint_env.py --case sbend     # the dictionary set of one tutorial case
    python3 adjoint_env.py --copy /work/sens --case naca0012/laminar/drag

An adjoint solve gives dJ/d(every surface point) for the cost of about one extra
primal: the whole point of it is that a thousand design variables cost what one costs.
OpenFOAM ESI v2512 ships that as `adjointOptimisationFoam`, and nothing in this
toolbox knew it existed -- the only adjoint name anywhere in the package was the
LEGACY `adjointShapeOptimizationFoam`, in a list of solvers whose time column counts
iterations (`preflight.py`).

WHETHER IT IS ON THIS INSTANCE IS A FACT ABOUT THIS INSTANCE, which is why this is a
probe and not a case writer. It is in the v2512 source tree at
`applications/solvers/incompressible/adjointOptimisationFoam` -- read from the release
tag itself, not from a release note -- but what is on PATH here depends on what the
base image's package actually built, and that could not be established from outside
the image. Guessing wrong is expensive in both directions: assuming it is absent means
a hand-rolled adjoint, which is a research project, and assuming it is present means
finding out from the middle of a job log.

on why this prints a path instead of writing a case

An adjoint sensitivity case is four dictionaries that have to agree with each other --
`system/optimisationDict` (the primal solver, the adjoint manager, the objective and
the design variables), `constant/adjointRASProperties` (the adjoint turbulence model),
the adjoint fields `Ua` and `pa` in the time directory, and `fvSolution`/`fvSchemes`
entries for them -- and the installed build ships a complete working set of them, per
objective, under `$FOAM_TUTORIALS`. Retyping those here from a reading of the source
would produce a second copy that nothing on this instance can check, that drifts from
the installed version on the next image bump, and that no test in this repository can
run, there being no OpenFOAM in CI's unit jobs. So this finds the tutorial that
already works, prints what is in it, and will copy it: the same argument
`hisa_env.py` makes about the ONERA M6 case it points at, and for the same reason --
`foamToC` and `foamInfo` are absent from this image, so the tutorials are the only
machine-readable statement of what the build supports.

What it reports about the tutorials is read off the disk in front of it. The adjoint
turbulence models named are the ones the installed cases actually use, which is not
the same claim as the ones the build registers, and the output says so.

on the number that would make a sensitivity trustworthy, which is not computed here

A sensitivity field is a derivative, and the only check for a derivative is a finite
difference: perturb one design variable, re-solve the primal, and compare
`(J(x+h) - J(x-h)) / 2h` against the adjoint's own projection onto that same
perturbation. Nothing here does that, and this says so rather than implying the
solver's output has been checked. It needs three more primal solves and a geometry
perturbation that matches the design parameterisation exactly, which is a workspace
run and not a probe. `--finite-difference` prints the recipe and the two traps
(the perturbation has to move the MESH the primal saw, not a surface file beside it;
and `sensitivityType surfacePoints` is per surface POINT, so the comparison is the
inner product of the point sensitivities with the point displacements, not a single
number read off anywhere).

This reports and changes nothing -- it writes only where `--copy` says to, installs
nothing, and runs no solver. A binary it cannot interrogate is reported as unchecked
rather than as working.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOLVER = "adjointOptimisationFoam"
"""The ESI v2512 framework: `optimisationManager`, primal and adjoint solvers,
objectives, design variables, and an optimisation loop around them."""

LEGACY = "adjointShapeOptimizationFoam"
"""The older single-purpose solver, which is a topology optimisation of a porosity
field and not the same thing. Named here because it is the only adjoint solver this
package previously mentioned, and because its presence is not evidence of the other's."""

TUTORIALS = "incompressible/adjointOptimisationFoam"
"""Where the installed cases sit under `$FOAM_TUTORIALS`, when they are installed."""

DICTIONARIES = (
    "system/optimisationDict",
    "constant/adjointRASProperties",
    "system/fvSolution",
    "system/fvSchemes",
    "system/controlDict",
)
"""The files a sensitivity case is made of. Listed so a case found on disk can be
reported as complete or as missing one of them, rather than assumed to have them."""

SENSITIVITY_OUTPUT = """\
Where the sensitivity lands, from the v2512 source rather than from a run:
`ShapeSensitivitiesBase::write()` writes ORDINARY OpenFOAM FIELDS into the time
directory -- the IOobject instance is `mesh_.time().timeName()` -- so `foamToVTK` and
every reader in this toolbox can already open them. Four names, each suffixed with the
adjoint solver's own name plus `ESI` or `SI`, so `adjS1` in the tutorial gives
`faceSensNormaladjS1ESI`:

  faceSensNormal<suffix>      volScalarField    always written
  pointSensNormal<suffix>     pointScalarField  always written
  faceSensVec, faceSensNormalVec, pointSensVec, pointSensNormalVec
                              vector versions, only under `writeAllSurfaceFiles true`
                              in the sensitivity dictionary (DEFAULT FALSE)
  fieldSens<suffix>           only under `writeFieldSens true` (DEFAULT FALSE)

AND THE TWO HALVES KEEP THEIR NUMBERS IN DIFFERENT PLACES, which is the trap:
`constructAndWriteSensitivityField` leaves the FACE field's internalField zero and puts
the values in its boundaryField on the sensitivity patches, while
`constructAndWriteSensitivtyPointField` does the opposite -- it pushes the patch values
into the INTERNAL field, with the source's own reason beside it ("Paraview does not
visualise values on fixedValuePointPatchFields. Only option is to store in
internalField"). A reader that looks in the wrong one of those gets a field of zeros
and no error, which is the same silent-wrong-answer shape `claims.py` exists for.

Read from gitlab.com/openfoam/core/openfoam at ref OpenFOAM-v2512:
src/optimisation/adjointOptimisation/adjoint/optimisation/adjointSensitivity/
  adjointSensitivity/shape/shapeSensitivityBase/ShapeSensitivitiesBase{.C,Templates.C}
  sensitivity/sensitivity.C
"""

SENSITIVITY_FIELDS = ("faceSensNormal", "pointSensNormal")
"""The two written whatever the dictionary says. The vector forms and the volume field
are behind `writeAllSurfaceFiles` and `writeFieldSens`, both of which default false."""

ADJOINT_FIELDS = ("Ua", "pa", "nuaTilda", "ma")
"""Adjoint field names to look for in a case's initial-condition directory. Which of
them a case has depends on its adjoint turbulence model, so the report lists what is
there instead of requiring a set."""


def which(name: str) -> str:
    found = shutil.which(name)
    return found or ""


def version(binary: str) -> dict[str, str]:
    """What the binary says about itself, or why that could not be asked.

    An OpenFOAM application with no arguments prints its banner and then complains
    about a missing case, so `-help` is the only safe way to ask -- and even that is
    run with a timeout, because a binary whose shared libraries are half-present can
    hang rather than fail.
    """
    if not binary:
        return {"checked": "no", "why": "not on PATH"}
    try:
        done = subprocess.run([binary, "-help"], capture_output=True, text=True,
                              timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"checked": "no", "why": f"{type(exc).__name__}: {exc}"}
    text = (done.stdout or "") + (done.stderr or "")
    build = re.search(r"Build\s*:\s*(\S+)", text)
    usage = re.search(r"^Usage:.*$", text, re.M)
    return {
        "checked": "yes",
        "exit_code": str(done.returncode),
        "build": build.group(1) if build else "",
        "usage": usage.group(0).strip() if usage else "",
    }


def tutorial_root() -> Path | None:
    """`$FOAM_TUTORIALS/incompressible/adjointOptimisationFoam`, if it is there."""
    base = os.environ.get("FOAM_TUTORIALS", "")
    if not base:
        return None
    root = Path(base) / TUTORIALS
    return root if root.is_dir() else None


def case_dirs(root: Path) -> list[Path]:
    """Every directory under `root` that has a `system/optimisationDict`.

    That file is what makes a directory an adjoint case; the tutorial tree nests
    cases two and three deep under a family name and a turbulence model, so the
    families cannot be listed by reading one level.
    """
    found = [path.parent.parent for path in sorted(root.rglob("system/optimisationDict"))]
    return [path for path in found if path.is_dir()]


def read_entry(text: str, key: str) -> str:
    """One `key value;` out of a dictionary, by regex.

    No OpenFOAM parser is written here, for the reason `spectra.foam_entry` gives: a
    wrong answer from this is a blank line in a report rather than a wrong case.
    """
    found = re.search(rf"^\s*{re.escape(key)}\s+([^;\s]+)\s*;", text, re.M)
    return found.group(1) if found else ""


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def objectives(text: str) -> list[str]:
    """The objective `type`s named inside an `optimisationDict`'s objectives blocks.

    Read by matching `type <word>;` and keeping the ones that are not the solver's own
    `type incompressible` -- a block-structure parser for a nested OpenFOAM dictionary
    is not worth writing to answer "what is being minimised", and a name this misses
    is a shorter list in a report.
    """
    inside = re.search(r"objectiveNames\s*\{(.*)", text, re.S)
    body = inside.group(1) if inside else text
    found = [m.group(1) for m in re.finditer(r"^\s*type\s+(\w+)\s*;", body, re.M)]
    return sorted({name for name in found if name != "incompressible"})


def describe(case: Path) -> dict[str, object]:
    """One tutorial case, as what it is set up to do and what it is made of."""
    optimisation = read_text(case / "system" / "optimisationDict")
    initial = next((case / name for name in ("0", "0.orig")
                    if (case / name).is_dir()), None)
    fields = []
    if initial is not None:
        fields = sorted(p.name for p in initial.iterdir()
                        if p.name in ADJOINT_FIELDS or p.name in ("U", "p"))
    return {
        "case": str(case),
        "manager": read_entry(optimisation, "optimisationManager"),
        "sensitivity_type": read_entry(optimisation, "sensitivityType"),
        "adjoint_solver": read_entry(optimisation, "solver"),
        "objectives": objectives(optimisation),
        "adjoint_turbulence": read_entry(
            read_text(case / "constant" / "adjointRASProperties"), "adjointRASModel"),
        "initial_fields": fields,
        "dictionaries": {name: (case / name).is_file() for name in DICTIONARIES},
        "complete": all((case / name).is_file() for name in DICTIONARIES),
        "has_allrun": (case / "Allrun").is_file(),
    }


def probe() -> dict[str, object]:
    """Everything this can establish about the adjoint capability of this instance."""
    binary = which(SOLVER)
    root = tutorial_root()
    cases = [describe(case) for case in case_dirs(root)] if root else []
    single_run = [row for row in cases
                  if row["manager"] == "singleRun" and row["complete"]]
    return {
        "solver": SOLVER,
        "on_path": bool(binary),
        "path": binary,
        "binary": version(binary),
        "legacy_solver": {"name": LEGACY, "on_path": bool(which(LEGACY)),
                          "path": which(LEGACY)},
        "openfoam_version": os.environ.get("WM_PROJECT_VERSION", ""),
        "tutorials_root": str(root) if root else "",
        "cases": cases,
        "sensitivity_only_cases": [row["case"] for row in single_run],
        "adjoint_turbulence_in_tutorials": sorted(
            {str(row["adjoint_turbulence"]) for row in cases if row["adjoint_turbulence"]}),
        "sensitivity_fields": list(SENSITIVITY_FIELDS),
        "sensitivity_checked_here": False,
    }


FINITE_DIFFERENCE = """\
A sensitivity is a derivative, and the check for a derivative is a finite difference.
Nothing in this toolbox has run one against this solver, so its output here is
unverified -- the recipe, and what makes it go wrong:

  1. Pick ONE design variable the geometry is actually built from, not a mesh point:
     an angle, a radius, a camber. Call it a.
  2. Solve the primal at a and record the objective J(a) the same way the
     optimisationDict defines it -- the same patches, the same direction, the same
     Aref and rhoInf. A force coefficient read from forceCoeffs.dat with a different
     reference area is not the J the adjoint differentiated.
  3. Re-mesh at a+h and at a-h and solve the primal on each, to the SAME residual
     controls. Central difference: (J(a+h) - J(a-h)) / 2h.
  4. Project the adjoint's own answer onto that same perturbation. With
     `sensitivityType surfacePoints` the sensitivity is per surface POINT, so the
     comparable number is the inner product of the point sensitivities with the point
     displacements d(x_p)/da taken between the a-h and a+h meshes. There is no single
     number anywhere in the output to read instead.

Two traps, both of which make an honest adjoint look wrong:

  - h has to be large enough that the two primals differ by more than their own
    convergence tolerance, and small enough to stay linear. Report the result at two
    values of h: if the difference between them is the size of the disagreement, the
    finite difference is the thing that is wrong.
  - the perturbation has to move the mesh the primal actually solved on. A perturbed
    STL beside an unperturbed mesh gives a clean, plausible, meaningless number.

Agreement to a few percent on a converged case is the usual outcome, and a sign error
or a factor of the reference area is the usual cause of anything worse.
"""


def report(found: dict[str, object]) -> str:
    lines: list[str] = []
    if found["on_path"]:
        build = found["binary"].get("build") or "build not reported"
        lines.append(f"{SOLVER}: present at {found['path']} ({build})")
    else:
        lines.append(f"{SOLVER}: NOT on PATH on this instance")
        lines.append("  it is in the ESI v2512 source tree "
                     "(applications/solvers/incompressible/adjointOptimisationFoam), "
                     "so this is a fact about what this image built, not about OpenFOAM")
    legacy = found["legacy_solver"]
    lines.append(f"{legacy['name']}: {'present' if legacy['on_path'] else 'absent'}"
                 + ("  (topology optimisation of a porosity field -- a different solver, "
                    "and not a substitute)" if legacy["on_path"] else ""))
    if found["openfoam_version"]:
        lines.append(f"OpenFOAM: {found['openfoam_version']}")

    if not found["tutorials_root"]:
        lines.append("")
        lines.append("no adjoint tutorials found: $FOAM_TUTORIALS is unset or has no "
                     f"{TUTORIALS}")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"tutorials: {found['tutorials_root']}")
    lines.append(f"  {len(found['cases'])} case(s) with a system/optimisationDict")
    for row in found["cases"]:
        name = Path(str(row["case"])).relative_to(Path(str(found["tutorials_root"])))
        parts = [f"manager={row['manager'] or '?'}"]
        if row["sensitivity_type"]:
            parts.append(f"sensitivityType={row['sensitivity_type']}")
        if row["objectives"]:
            parts.append("objectives=" + ",".join(str(o) for o in row["objectives"]))
        if row["adjoint_turbulence"]:
            parts.append(f"adjointRASModel={row['adjoint_turbulence']}")
        if not row["complete"]:
            missing = [k for k, there in row["dictionaries"].items() if not there]
            parts.append("INCOMPLETE, missing " + ",".join(missing))
        lines.append(f"    {name}  " + "  ".join(parts))
    if found["sensitivity_only_cases"]:
        lines.append("")
        lines.append("a sensitivity map without an optimisation loop is "
                     "`optimisationManager singleRun`, and these cases are already that:")
        for case in found["sensitivity_only_cases"]:
            lines.append(f"    {case}")
        lines.append("  `--copy DIR --case <name>` copies one here to edit")
    if found["adjoint_turbulence_in_tutorials"]:
        lines.append("")
        lines.append("adjointRASModel values the INSTALLED CASES use: "
                     + ", ".join(found["adjoint_turbulence_in_tutorials"]))
        lines.append("  which is what the tutorials chose, not a list of what the build "
                     "registers -- a wrong one makes the solver print its own list and stop")
    lines.append("")
    lines.append(SENSITIVITY_OUTPUT.rstrip())
    lines.append("")
    lines.append("no sensitivity from this solver has been checked against a finite "
                 "difference by anything in this toolbox: `--finite-difference` says how")
    return "\n".join(lines)


def matching(found: dict[str, object], wanted: str) -> list[dict[str, object]]:
    """The cases whose path ends with `wanted`, separator-agnostic.

    A case is named by the tail of its path (`naca0012/laminar/drag`), and the tail is
    compared as a posix path on both sides: the instance is Linux, but the tests that
    read a tutorial tree run on the Windows leg of the matrix too, where
    `str(path).endswith("a/b")` is false for every path there is.
    """
    tail = wanted.replace("\\", "/").strip("/")
    return [row for row in found["cases"]
            if Path(str(row["case"])).as_posix().endswith(tail)]


def copy_case(found: dict[str, object], wanted: str, target: Path) -> Path:
    """Copy one tutorial case to a writable directory, mesh excluded.

    The tutorial cases keep their mesh in a shared `resources/` directory that their
    own `Allrun` copies in, so a copy made here has a `constant/` with no `polyMesh`.
    That is said in the caller rather than papered over: a case copied without its
    mesh and run anyway fails on a missing `polyMesh` three steps later.
    """
    matches = matching(found, wanted)
    if not matches:
        raise FileNotFoundError(
            f"no tutorial case matching {wanted!r}; "
            + ", ".join(str(row["case"]) for row in found["cases"]))
    source = Path(str(matches[0]["case"]))
    target = target / source.name if target.is_dir() else target
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Whether the adjoint optimisation solver is on this instance, and "
                    "what a sensitivity case made of.")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="the same facts as JSON")
    parser.add_argument("--case", default="",
                        help="one tutorial case, by the tail of its path "
                             "(e.g. naca0012/laminar/drag): its dictionary set in full")
    parser.add_argument("--copy", type=Path, default=None,
                        help="copy the --case here to edit (its mesh lives in the "
                             "tutorials' shared resources/ and is not copied)")
    parser.add_argument("--finite-difference", action="store_true",
                        help="print how to check a sensitivity against a finite "
                             "difference, which nothing here has done")
    args = parser.parse_args(argv)

    if args.finite_difference:
        print(FINITE_DIFFERENCE)
        return 0

    found = probe()
    if args.case:
        matches = matching(found, args.case)
        found["cases"] = matches
        if not matches:
            print(f"no tutorial case matching {args.case!r} on this instance")
            return 0
    if args.copy is not None:
        if not args.case:
            print("--copy needs --case: there are several, and which one is the point")
            return 2
        try:
            where = copy_case(found, args.case, args.copy)
        except (FileNotFoundError, OSError) as exc:
            print(f"not copied: {exc}")
            return 1
        print(f"copied to {where}")
        print("  its mesh is NOT here: the tutorial's Allrun copies one in from "
              "$FOAM_TUTORIALS/incompressible/adjointOptimisationFoam/resources")
    if args.as_json:
        sys.stdout.write(json.dumps(found, indent=2, default=str) + "\n")
        return 0
    print(report(found))
    if args.case and found["cases"]:
        row = found["cases"][0]
        print("")
        print(f"{row['case']}")
        for name, there in row["dictionaries"].items():
            print(f"  {'yes' if there else 'NO '}  {name}")
        print(f"  initial fields: {', '.join(str(f) for f in row['initial_fields']) or 'none found'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
