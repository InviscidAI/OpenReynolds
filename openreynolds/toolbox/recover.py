#!/usr/bin/env python3
"""What to change on a diverged run before its mesh, read out of the case.

A buoyant case with thin baffles diverged, and was remeshed. The blow-up moved, and it
was remeshed again; over several days every change was to the mesh, each one moved the
blow-up somewhere else, and the relaxation factors, the convection schemes and the time
step it had started with were never touched. `locate.py` says where and which field;
`log_digest.py` says that the residuals climbed. Neither says what to change, and the
reading that comes most easily -- it blew up where the cells are, so fix the cells --
is the most expensive move there is and, when the cause is numerical, one that moves
the blow-up rather than removing it.

So this reads the case and its solver log and lists the numerical moves, cheapest
first, each against what this case already has:

1. **restart** from the last time written before the trouble started, so nothing that
   already ran is thrown away;
2. **time step**, for a transient solver: the Courant number the log recorded, and a
   `deltaT` or `maxCo` that brings it down;
3. **under-relaxation**, for a steady solver or PIMPLE with outer correctors: the
   factors in `fvSolution` against the classic `p 0.3 / U 0.7`;
4. **first-order convection**: the `div(phi,...)` terms not already upwind, to run on
   upwind first and continue on the second-order schemes from that field;
5. **non-orthogonal correctors**, when checkMesh measured a non-orthogonality where
   one correction pass is not enough;
6. **a potential-flow start**, for an incompressible single-phase case starting from a
   uniform field;
7. **the mesh**, last, with `locate.py` for where.

Each move says `try`, `done` (the case already has it) or `n/a` (it does not apply to
this solver), with what was read and the exact edit. A move that is `try` carries the
dictionary lines to write; nothing here writes them. Under-relaxation that is already
far below the classic values is reported as `done` and not lowered further: a solve
that only holds together there is telling you about its mesh, its boundary conditions
or a flow that is not steady (`notes/openfoam-field-notes.md`, "Under-relaxation").
The same holds for the list as a whole -- if three of these have each left it
diverging at the same place and time, it is not the numerics, and `locate.py
--compare` against a differently meshed sibling is what tells the mesh from the
boundary specification.

This script edits nothing, refuses nothing, and blocks nothing. It writes no file,
runs no solver, and the exit code is 0 whatever it finds, including a case it could
not read. Which move to make, and whether to make any, is yours.

    python3 recover.py /work/case
    python3 recover.py /work/case --log log.buoyantPimpleFoam
    python3 recover.py /work/case --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locate  # noqa: E402  (sibling script, not a package)
import mesh_digest  # noqa: E402
import preflight  # noqa: E402

STATUSES = ("try", "done", "n/a")

CLASSIC_RELAXATION = {"p": 0.3, "p_rgh": 0.3, "U": 0.7}
"""The stable pair, and `p_rgh` with it. Every other transported field (k, epsilon,
omega, nuTilda, h, T, e, alpha) is relaxed like U."""

OTHER_FIELDS_RELAXATION = 0.7

FAR_BELOW = 0.5
"""A factor at or below this fraction of its classic value is already far below it;
lowering it further is not offered."""

TARGET_COURANT = 0.5
"""The Courant number a fixed `deltaT` is scaled to, and the ceiling for `maxCo`."""

NON_ORTHO_CORRECT = preflight.NON_ORTHO_WARN
"""checkMesh's maximum non-orthogonality above which one correction pass is short."""

INCOMPRESSIBLE_SINGLE_PHASE = frozenset({
    "simpleFoam", "pimpleFoam", "pisoFoam", "icoFoam",
    "porousSimpleFoam",
})
"""The solvers `potentialFoam`'s field is a start for: one incompressible phase, no
energy equation. A buoyant, compressible or multiphase solver is not on it, and nor
are the SRF solvers: they hold `0/Urel`, and potentialFoam reads `U`."""

FIRST_ORDER = re.compile(r"^(bounded\s+)?Gauss\s+upwind(\s+\S+)?$")
"""A convection scheme that is already upwind. Everything else -- linearUpwind(V),
limitedLinear(V), LUST, MUSCL(V), vanLeer, the limited and blended families -- is a
candidate: there are too many higher-order names to list, and one upwind."""
FATAL = re.compile(r"FOAM FATAL (?:IO )?ERROR[^\n]*\n?([^\n]*)")
FPE = re.compile(r"(?m)^(?!.*trapFpe).*(sigFpe::sigHandler|Floating point exception)")
"""A floating point exception that happened, not the `trapFpe: Floating point exception
trapping enabled` banner every OpenFOAM log opens with."""


@dataclass
class Move:
    move: str
    status: str
    read: str
    edit: str = ""
    why: str = ""


# -- reading the case ------------------------------------------------------------


def _number(value: Any) -> float | None:
    return preflight.as_float(value)


def _block(text: str, *path: str) -> str:
    for name in path:
        text = preflight.block_body(text, name)
        if not text:
            return ""
    return text


def _algorithm(fv_solution: str) -> tuple[str, dict[str, str]]:
    for name in ("PIMPLE", "SIMPLE", "PISO"):
        body = preflight.block_body(fv_solution, name)
        if body:
            return name, preflight.entry_values(body)
    return "", {}


_KEYED = re.compile(r'("(?:[^"\\]|\\.)*"|[^\s;{}"]+)\s+([^;{}]+?)\s*;')
"""One `key value;` entry, the key either a word or a quoted regex such as `".*"`,
`"(U|k|epsilon)"` or `"div\\(phi,(k|epsilon)\\)"`."""


def keyed_entries(body: str) -> list[tuple[str, str]]:
    """The `key value;` entries directly inside a block, in file order, quoted keys
    kept quoted. Directives (`#include`, `#inputMode`) are dropped first: they end
    without a `;` and would otherwise swallow the entry after them."""
    text = "\n".join(line for line in preflight.strip_comments(body).splitlines()
                     if not line.lstrip().startswith("#"))
    return [(m.group(1), " ".join(m.group(2).split()))
            for m in _KEYED.finditer(preflight.outer_text(text))]


def lookup(entries: list[tuple[str, str]], name: str) -> tuple[str, str] | None:
    """`name` resolved the way OpenFOAM's dictionary resolves it: the exact key (the
    last one written), then the regex keys from the last written up, then `default`."""
    for key, value in reversed(entries):
        if key == name:
            return key, value
    for key, value in reversed(entries):
        if key.startswith('"') and key.endswith('"') and len(key) > 1:
            try:
                if re.fullmatch(key[1:-1], name):
                    return key, value
            except re.error:
                continue
    for key, value in reversed(entries):
        if key == "default":
            return key, value
    return None


def _has_block(body: str, name: str) -> bool:
    return bool(re.search(r"(?<![\w.])" + name + r"\s*\{", preflight.strip_comments(body)))


def relaxation_sections(fv_solution: str) -> tuple[dict[str, list[tuple[str, str]]], bool]:
    """`relaxationFactors` as OpenFOAM reads it: `fields {}` and `equations {}` when
    either is present -- and then the block's own top-level entries are ignored --
    else the top-level entries, which the old spelling applied to both. The flag says
    which: an edit has to go where the solver will read it."""
    body = preflight.block_body(fv_solution, "relaxationFactors")
    if _has_block(body, "fields") or _has_block(body, "equations"):
        return {"fields": keyed_entries(preflight.block_body(body, "fields")),
                "equations": keyed_entries(preflight.block_body(body, "equations"))}, False
    top = keyed_entries(body)
    # solution.C reads the old layout's field factors from the keys that start with p
    # or rho, literally; a top-level regex or default never reaches a field.
    fields = [(k, v) for k, v in top if not k.startswith('"') and k != "default"
              and (k.startswith("p") or k.startswith("rho"))]
    return {"fields": fields, "equations": top}, bool(top)


def factor(entries: list[tuple[str, str]], name: str) -> tuple[float | None, str]:
    """The factor `name` gets, and the key it came from. No match is no relaxation,
    which is a factor of 1; a value that is not a number is None."""
    found = lookup(entries, name)
    if found is None:
        return 1.0, ""
    return _number(found[1]), found[0]


TURBULENCE_FIELDS = ("k", "epsilon", "omega", "nuTilda", "kl", "kt", "v2", "f", "R",
                     "ReThetat", "gammaInt")


def _is_convection(key: str) -> bool:
    """A divergence term carried by a flux: `div(phi,U)`, `div(rhoPhi,U)`,
    `div(phid,p)`, or a regex key over them. Not `div(phir,alpha)` or
    `div(phirb,alpha)`, which are interface compression, and not alpha itself,
    whose convection upwinding smears the interface it exists to keep sharp. A phase's
momentum, `div(alphaRhoPhi.air,U.air)`, is convection like any other."""
    text = key.strip('"').replace("\\(", "(").replace("\\)", ")")
    if not re.search(r"div\(\s*[^,]*[Pp]hi[^,]*,", text):
        return False
    return not re.search(r"div\(\s*phir", text) and not re.search(r",\s*\(?alpha", text)


def convection_entries(fv_schemes: str) -> tuple[list[tuple[str, str]], str]:
    """The convection entries of `divSchemes` as written, and the `default` scheme.
    `default` is never one of them: it also covers the viscous terms, which take no
    upwind scheme (`Gauss upwind` there wants a flux name and stops the run)."""
    entries = keyed_entries(preflight.block_body(fv_schemes, "divSchemes"))
    default = next((v for k, v in reversed(entries) if k == "default"), "")
    return [(k, expand(entries, v)) for k, v in entries if _is_convection(k)], default


def expand(entries: list[tuple[str, str]], value: str) -> str:
    """`$turbulence` read as what `turbulence` holds in the same block, the last one
    written; a macro that cannot be resolved here is left as written."""
    match = re.fullmatch(r"\$:?(\w+)", value.strip())
    if not match:
        return value
    for key, held in reversed(entries):
        if key == match.group(1) and not held.startswith("$"):
            return held
    return value


def _non_ortho(case: preflight.Case) -> float | None:
    value = mesh_digest.parse(case.check_mesh_log_text).get("non_ortho")
    if isinstance(value, tuple):
        value = value[0]
    return _number(value)


def _times(case_path: Path) -> list[tuple[float, str]]:
    """Times written, serial or the first processor's, each with its directory name:
    a restart names the directory as written, because `0.1234567` printed to six
    figures is a directory the solver will not find."""
    roots = [case_path] + sorted(case_path.glob("processor0"))
    found: dict[float, str] = {}
    for root in roots:
        for t, path in locate.time_dirs(root):
            found.setdefault(t, path.name)
    return sorted(found.items())


def _log_facts(log_path: Path | None) -> dict[str, Any]:
    if log_path is None:
        return {"path": None, "first_degrading": None, "courant": [], "fatal": "",
                "fpe": False, "last_time": None}
    parsed = locate.parse_log(log_path)
    text = Path(log_path).read_text(errors="replace")
    fatal = FATAL.search(text)
    times = [t for series in parsed["residuals"].values() for t, _ in series]
    return {
        "path": str(log_path),
        "first_degrading": locate.first_degrading(parsed),
        "courant": parsed["courant"],
        "fatal": " ".join(fatal.group(1).split()) if fatal else "",
        "fpe": bool(FPE.search(text)),
        "last_time": max(times) if times else None,
    }


def _onset(log: dict[str, Any]) -> float | None:
    first = log.get("first_degrading")
    return first["time"] if first else None


# -- the moves -------------------------------------------------------------------


def restart_move(times: list[tuple[float, str]], onset: float | None) -> Move:
    names = dict(times)
    written = [t for t, _ in times if t > 0]
    if not written:
        return Move("restart", "n/a", "no time after 0 was written",
                    why="there is nothing to restart from but the initial condition")
    if onset is None:
        last = written[-1]
        return Move(
            "restart", "try",
            f"latest write {last:g}; the log gave no onset to place it against",
            f"controlDict: startFrom startTime; startTime {names[last]};",
            "the last write may already carry the blow-up -- locate.py on it says "
            "whether its fields are sane before you start from it",
        )
    good = [t for t in written if t < onset]
    if not good:
        return Move("restart", "n/a",
                    f"trouble starts at {onset:g}, before the first write ({written[0]:g})",
                    why="no write predates the onset; this run restarts from 0")
    return Move(
        "restart", "try",
        f"trouble starts at {onset:g}; the last write before it is {good[-1]:g}",
        f"controlDict: startFrom startTime; startTime {names[good[-1]]};",
        "the run up to there is kept, so each move below costs only the stretch "
        "that failed",
    )


def time_step_move(case: preflight.Case, log: dict[str, Any], onset: float | None) -> Move:
    if steady(case):
        return Move("time step", "n/a", f"{case.application} is steady",
                    why="its deltaT counts iterations; there is no Courant number")
    control = case.control
    delta_t = _number(control.get("deltaT"))
    adjusting = str(control.get("adjustTimeStep", "")).strip() in ("yes", "on", "true", "1")
    max_co = _number(control.get("maxCo"))
    courant = log.get("courant") or []
    crashed = bool(log.get("fatal") or log.get("fpe"))
    # OpenFOAM prints a step's Courant number above its `Time =` line, so the parser
    # files it under the step before: the last entry before the onset is the onset's
    # own, and in a run that crashed with no onset the last entry is the crash step's.
    if onset is not None:
        before, where = [mx for t, _m, mx in courant if t < onset][:-1], "before the onset"
    elif crashed:
        before, where = [mx for _t, _m, mx in courant][:-1], "before the last step"
    else:
        before, where = [mx for _t, _m, mx in courant], "over the run"
    peak = max((mx for _t, _mean, mx in courant), default=None)
    healthy = max(before) if before else None
    if healthy is not None:
        seen = f"Courant max {healthy:.3g} {where}, peak {peak:.3g}"
    elif peak is not None:
        seen = f"no Courant number {where}; peak {peak:.3g}"
    else:
        seen = "no Courant number in the log"
    if adjusting:
        if max_co is not None and max_co <= TARGET_COURANT:
            return Move("time step", "done",
                        f"adjustTimeStep on with maxCo {max_co:g}; {seen}",
                        why="the step already adapts to a Courant number this low")
        target = TARGET_COURANT if max_co is None else min(TARGET_COURANT, max_co / 2)
        return Move("time step", "try",
                    f"adjustTimeStep on with maxCo {max_co if max_co is not None else '?'}; "
                    f"{seen}",
                    f"controlDict: maxCo {target:g};",
                    "a smaller Courant target is the cheapest move on a transient run")
    if delta_t is None:
        return Move("time step", "try", f"no deltaT read from controlDict; {seen}",
                    "controlDict: adjustTimeStep yes; maxCo 0.5;")
    if healthy is not None and healthy > TARGET_COURANT:
        proposed = delta_t * TARGET_COURANT / healthy
        return Move("time step", "try", f"deltaT {delta_t:g} fixed; {seen}",
                    f"controlDict: deltaT {proposed:.3g};  (or adjustTimeStep yes; maxCo 0.5;)",
                    f"scales the step to Courant {TARGET_COURANT:g} at the speeds this run saw")
    if healthy is not None:
        return Move("time step", "done", f"deltaT {delta_t:g} fixed; {seen}",
                    why="the Courant number was already low before the trouble; a smaller "
                        "step is unlikely to be the fix")
    return Move("time step", "try", f"deltaT {delta_t:g} fixed; {seen}",
                f"controlDict: deltaT {delta_t / 2:.3g};")


def _energy_name(case: preflight.Case) -> str:
    """The key the energy equation is relaxed under: `T` for a Boussinesq or VoF solver
    (their TEqn.H relaxes TEqn) or a case with no thermo energy entry, else `e` or `h`
    from `thermoType { energy ...; }`."""
    application = case.application
    if "Boussinesq" in application or application.endswith("InterFoam"):
        return "T"
    thermo = preflight.strip_comments(case.read("constant/thermophysicalProperties"))
    match = re.search(r"\benergy\s+(\w+)\s*;", thermo)
    if not match:
        return "T"
    return "e" if "InternalEnergy" in match.group(1) else "h"


def steady(case: preflight.Case) -> bool:
    """A steady run: a steady solver by name, a `SIMPLE` block in fvSolution, or a
    `steadyState` ddt default. The name alone misses `simpleReactingParcelFoam`."""
    if preflight.is_steady(case.application):
        return True
    fv_solution = case.read("system/fvSolution")
    if preflight.block_body(fv_solution, "SIMPLE") and not \
            preflight.block_body(fv_solution, "PIMPLE"):
        return True
    ddt = preflight.entry_values(preflight.block_body(case.read("system/fvSchemes"),
                                                      "ddtSchemes"))
    return ddt.get("default", "").strip() == "steadyState"


def _equation_names(case: preflight.Case, sections) -> list[str]:
    """U, the turbulence fields this case has, the energy equation when it solves one,
    and any other equation the dictionary names exactly."""
    field_dir = case.field_dir or case.path / "0"
    names = ["U"] + [n for n in TURBULENCE_FIELDS if (field_dir / n).is_file()]
    if (field_dir / "T").is_file() and case.application not in INCOMPRESSIBLE_SINGLE_PHASE:
        names.append(_energy_name(case))
    for key, _value in sections["equations"]:
        if (not key.startswith('"') and key not in ("default", "rho", "p", "p_rgh")
                and not key.endswith("Final") and key not in names):
            names.append(key)
    return names


def _shown(name: str, value: float | None, key: str) -> str:
    shown = "?" if value is None else f"{value:g}"
    if not key:
        return f"{name} unrelaxed"
    return f"{name} {shown}" + ("" if key == name else f" (from {key})")


def relaxation_move(case: preflight.Case) -> Move:
    fv_solution = case.read("system/fvSolution")
    algorithm, entries = _algorithm(fv_solution)
    sections, top_level = relaxation_sections(fv_solution)
    is_steady = steady(case)
    outer = _number(entries.get("nOuterCorrectors")) or 1
    if not is_steady and outer <= 1:
        value, key = factor(sections["equations"], "UFinal")
        return Move("under-relaxation", "n/a",
                    f"{algorithm or 'PIMPLE'} with nOuterCorrectors {outer:g}; "
                    + _shown("UFinal", value, key),
                    why="with one outer corrector only the Final factors apply (UFinal, or a "
                        "regex or default entry that matches it); on a transient the time "
                        "step is the lever, and relaxation proper needs nOuterCorrectors "
                        "above 1")
    consistent = str(entries.get("consistent", "")).strip() in ("yes", "on", "true", "1")
    field_dir = case.field_dir or case.path / "0"
    pressure = "p_rgh" if (field_dir / "p_rgh").is_file() else "p"
    wanted = [] if consistent else [("fields", pressure)]
    wanted += [("equations", n) for n in _equation_names(case, sections)]
    read_parts, proposals, far, unknown = [], [], [], []
    for section, name in wanted:
        value, key = factor(sections[section], name)
        read_parts.append(_shown(name, value, key))
        if value is None:
            unknown.append(name)
            continue
        classic = CLASSIC_RELAXATION.get(name, OTHER_FIELDS_RELAXATION)
        if value <= classic * FAR_BELOW:
            far.append(name)
        elif value > classic:
            proposals.append((section, name, classic))
        else:
            proposals.append((section, name, max(round(value * 2 / 3, 2), classic * FAR_BELOW)))
    read = (f"{algorithm or '?'}{' (SIMPLEC: pressure left unrelaxed)' if consistent else ''}: "
            + ", ".join(read_parts))
    if not proposals:
        return Move("under-relaxation", "done" if far else "n/a", read,
                    why="already far below the classic values; going lower is not the "
                        "move -- see the note on what that says about the case" if far
                        else "no factor could be read as a number")
    if top_level:
        edit = "fvSolution relaxationFactors (top level, as written): " + " ".join(
            f"{n} {v:g};" for _s, n, v in proposals)
    else:
        edit = "fvSolution relaxationFactors: " + " ".join(
            f"{section} {{ " + " ".join(f"{n} {v:g};" for s, n, v in proposals if s == section)
            + " }" for section in ("fields", "equations")
            if any(s == section for s, _n, _v in proposals))
    why = ("lower factors trade iterations for stability, roughly linearly; an exact key "
           "outranks a regex or default entry, so these can be added beside them")
    if far:
        why += f"; {', '.join(far)} already far below classic and left alone"
    if unknown:
        why += f"; {', '.join(unknown)} not read as a number and left alone"
    return Move("under-relaxation", "try", read, edit, why)


def convection_move(case: preflight.Case) -> Move:
    terms, default = convection_entries(case.read("system/fvSchemes"))
    default_note = (f"; default {default} covers any term not listed"
                    if default and default != "none" else "")
    if not terms:
        field_dir = case.field_dir or case.path / "0"
        single_phase = not any(field_dir.glob("alpha.*"))  # alphat is not a phase
        solved_any = (field_dir / "U").is_file()
        if default_note and not FIRST_ORDER.match(default) and single_phase and solved_any:
            sections, _ = relaxation_sections(case.read("system/fvSolution"))
            solved = [n for n in _equation_names(case, sections)
                      if (field_dir / n).is_file() or n in ("h", "e")]
            bounded = "bounded " if steady(case) else ""
            return Move(
                "first-order convection", "try",
                "no convection entry in divSchemes" + default_note,
                "fvSchemes divSchemes, added as explicit entries: " + " ".join(
                    f"div(phi,{n}) {bounded}Gauss upwind;" for n in solved),
                "an explicit entry overrides default for that term alone; default is left "
                "as it is, because it also covers the viscous terms, which take no upwind")
        return Move("first-order convection", "n/a",
                    "no convection entry in divSchemes" + default_note,
                    why="an explicit div(phi,X) entry overrides the default for that term")
    unresolved = [k for k, v in terms if v.startswith("$")]
    higher = [(k, v) for k, v in terms if not v.startswith("$") and not FIRST_ORDER.match(v)]
    read = "; ".join(f"{k} {v}" for k, v in terms) + default_note
    if unresolved and not higher:
        return Move("first-order convection", "n/a", read,
                    why=f"{', '.join(unresolved)} name a macro this block does not define; "
                        "read what it expands to before deciding")
    if not higher:
        return Move("first-order convection", "done", read,
                    why="every listed convection term is already upwind")
    bounded = "bounded " if steady(case) else ""
    edit = "fvSchemes divSchemes: " + " ".join(
        f"{k} {bounded}Gauss upwind;" for k, _v in higher)
    return Move(
        "first-order convection", "try", read, edit,
        "upwind is diffusive and forgiving; once it runs, startFrom latestTime on the "
        "second-order schemes continues from a field that is already close. default "
        "is left alone: it also covers the viscous terms, which take no upwind scheme",
    )


def non_ortho_move(case: preflight.Case) -> Move:
    measured = _non_ortho(case)
    _, entries = _algorithm(case.read("system/fvSolution"))
    correctors = _number(entries.get("nNonOrthogonalCorrectors")) or 0
    if measured is None:
        return Move("non-orthogonal correctors", "n/a",
                    "no checkMesh log to read the non-orthogonality from",
                    why="run checkMesh to know whether this applies")
    read = f"checkMesh max non-orthogonality {measured:g}; nNonOrthogonalCorrectors {correctors:g}"
    if measured <= NON_ORTHO_CORRECT:
        return Move("non-orthogonal correctors", "n/a", read,
                    why=f"below {NON_ORTHO_CORRECT:g} degrees one pass is enough")
    if correctors >= 2:
        return Move("non-orthogonal correctors", "done", read)
    laplacian = keyed_entries(preflight.block_body(case.read("system/fvSchemes"),
                                                   "laplacianSchemes"))
    explicit = any(k == "default" and v == "none" for k, v in laplacian)
    schemes = ("each laplacianSchemes entry (default is none here) to Gauss linear limited "
               "corrected 0.33" if explicit else
               "laplacianSchemes { default Gauss linear limited corrected 0.33; }")
    return Move(
        "non-orthogonal correctors", "try", read,
        "fvSolution SIMPLE/PIMPLE: nNonOrthogonalCorrectors 2;  fvSchemes: "
        f"{schemes}; snGradSchemes {{ default limited corrected 0.33; }}",
        "non-orthogonality is corrected in these two places; a cellLimited gradient "
        "is neither of them",
    )


def potential_move(case: preflight.Case, times: list[tuple[float, str]]) -> Move:
    application = case.application
    if application not in INCOMPRESSIBLE_SINGLE_PHASE:
        return Move("potential-flow start", "n/a", f"solver {application or '?'}",
                    why="potentialFoam's field is a start for one incompressible phase only")
    if any(t > 0 for t, _ in times):
        return Move("potential-flow start", "n/a", "the run has written past 0",
                    why="a restart from a written time is the better start")
    u_text = case.read(f"{(case.field_dir or case.path / '0').name}/U")
    uniform = bool(re.search(r"internalField\s+uniform", preflight.strip_comments(u_text)))
    if not uniform:
        return Move("potential-flow start", "done", "0/U internalField is not uniform",
                    why="the initial field is already shaped")
    fv_solution = case.read("system/fvSolution")
    ready = bool(preflight.block_body(fv_solution, "potentialFlow")) and \
        bool(re.search(r"\bPhi\b", preflight.block_body(fv_solution, "solvers")))
    needs = "" if ready else "  (fvSolution needs a Phi solver and a potentialFlow block)"
    return Move("potential-flow start", "try", "0/U internalField is uniform",
                f"run potentialFoam -writep before {application}{needs}",
                "a uniform start shocks the first iterations; the potential field does not")


def mesh_move(log: dict[str, Any]) -> Move:
    first = log.get("first_degrading")
    read = (f"{first['field']} degrades first, at {first['time']:g} ({first['how']})"
            if first else "no first-degrading field in the log")
    return Move(
        "mesh", "try", read,
        "python3 locate.py <case> -- where that field's extremum sits; then "
        "--compare a differently meshed sibling",
        "last, because it costs a remesh and a rerun, and when the cause was numerical "
        "it moves the blow-up rather than removing it",
    )


def run(case_path: Path, *, log: str | None = None) -> dict[str, Any]:
    case_path = Path(case_path)
    if not (case_path / "system" / "controlDict").is_file():
        return {"case": str(case_path), "errors": [f"no system/controlDict under {case_path}"],
                "diverged": None, "log": _log_facts(None), "moves": []}
    case = preflight.Case(case_path)
    notes: list[str] = []
    log_path = case.solver_log(log)
    if log and log_path is None:
        notes.append(f"no such log: {log}; the moves are read from the dictionaries alone")
    if log_path is not None and log_path.suffix == ".gz":
        notes.append(f"{log_path.name} is compressed; gunzip it and pass --log to read it")
        log_path = None
    if (case_path / "constant" / "regionProperties").is_file():
        notes.append("a multi-region case: these moves read the top-level system/ "
                     "dictionaries, and each region's own are under system/<region>/")
    log_facts = _log_facts(log_path)
    onset = _onset(log_facts)
    times = _times(case_path)
    moves = [
        restart_move(times, onset),
        time_step_move(case, log_facts, onset),
        relaxation_move(case),
        convection_move(case),
        non_ortho_move(case),
        potential_move(case, times),
        mesh_move(log_facts),
    ]
    diverged = bool(log_facts["fatal"] or log_facts["fpe"] or onset is not None)
    return {"case": str(case_path), "errors": notes, "application": case.application,
            "diverged": diverged if log_facts["path"] else None, "onset": onset,
            "log": {k: v for k, v in log_facts.items() if k != "courant"}
            | {"courant_tail": log_facts["courant"][-5:]},
            "moves": [asdict(m) for m in moves]}


def render(report: dict[str, Any]) -> str:
    lines = [f"# recover {report['case']}"]
    for error in report["errors"]:
        lines.append(f"note: {error}")
    log = report["log"]
    if log["path"] is None:
        lines.append("no solver log found; the moves below are read from the dictionaries alone")
    else:
        how = []
        if log["fatal"]:
            how.append(f"FOAM FATAL: {log['fatal']}")
        if log["fpe"]:
            how.append("a floating point exception")
        first = log["first_degrading"]
        if first:
            how.append(f"{first['field']} degrading first at {first['time']:g} ({first['how']})")
        lines.append(f"{Path(log['path']).name}: " + ("; ".join(how) if how else
                     "nothing in it says the run diverged -- these moves may not be needed"))
    for move in report["moves"]:
        lines.append("")
        lines.append(f"{move['status'].upper():<6}{move['move']}")
        lines.append(f"  read  {move['read']}")
        if move["edit"]:
            lines.append(f"  edit  {move['edit']}")
        if move["why"]:
            lines.append(f"  why   {move['why']}")
    if report["moves"]:
        lines.append("")
        lines.append("Cheapest first. Three that each leave it diverging at the same place "
                     "and time say it is not the numerics. Nothing here has changed the case.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, help="the case directory")
    parser.add_argument("--log", default=None, help="the solver log, if not the newest log.*")
    parser.add_argument("--json", action="store_true", help="the report as JSON")
    args = parser.parse_args(argv)
    try:
        report = run(args.case, log=args.log)
    except Exception as error:  # pragma: no cover - the last net under the contract
        report = {"case": str(args.case), "diverged": None, "moves": [],
                  "log": {"path": None, "fatal": "", "fpe": False, "first_degrading": None},
                  "errors": [f"recover raised {type(error).__name__}: {error} "
                             "(a bug in recover.py, not necessarily the case)"]}
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
