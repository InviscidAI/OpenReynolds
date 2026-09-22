#!/usr/bin/env python3
"""One candidate of a parametric study: build, mesh, dress, solve, score -- one point per call.

    python3 parametric.py CANDIDATE_DIR --round ROUND_DIR --lock goal.lock.json --mode mesh
    python3 parametric.py CANDIDATE_DIR --round ROUND_DIR --lock goal.lock.json --mode solve

A design study is the same case built many times with different numbers, and the
numbers have to arrive somewhere a rebuild script can read without being edited. The
convention is one file: `CANDIDATE_DIR/design_constants.py`, lines of `NAME = <number>`,
written by whoever is choosing the points (an optimizer, a person, a shell loop). The
round's `build.py` begins with `from design_constants import *` and takes its parameters
from there -- this refuses to start on a `build.py` that does not, because a build that
ignores the constants would score every candidate on the same geometry and report it as
a landscape.

what happens, in order

    1. `build.py` (and `hooks.py`, if the round has one) are copied from ROUND_DIR into
       CANDIDATE_DIR beside the constants, and both are read before anything runs:
       `build.py` must import the constants, and `hooks.py` may define only
       `case_args()` -- any other top-level definition is refused by name.
    2. `python3 build.py` runs in CANDIDATE_DIR and has to leave `constant/polyMesh`.
    3. `mesh_look.py` measures it and runs `checkMesh`; the payload is `look.json`.
       `--mode mesh` stops here: the score is `mesh` or `ok` (or `cells`, when the count
       is outside the lock's band) and there are no coefficients.
    4. `case_gen.py CANDIDATE_DIR <lock.case_gen_args> <hooks.case_args()> --iterations
       <fidelity.iters> --yplus --force [--body <lock.body_patch>]` dresses the mesh. A
       hook may add flags; it may not repeat one the lock already sets -- the lock is the
       harness's physics and a hook that overrode `--reynolds` would be scoring a
       different problem under the same name.
    5. The lock's solver (`simpleFoam` unless it says otherwise) runs to `log.<solver>`,
       serial at `fidelity.ranks` 1 and `decomposePar` / `mpirun -np N` / `reconstructPar
       -latestTime` above it.
    6. `score.py` writes `CANDIDATE_DIR/metrics.json`. Nothing else does.

the trim

    When the lock carries `trim` -- `{"variable": "ALPHA_DEG", "metric": "Cl", "target":
    0.8, "tol": 0.01, "max_solves": 3}` -- the candidate is solved at fixed lift rather
    than fixed incidence: the variable is one of the design constants, so every trim
    iteration is a rebuild, a mesh and a solve in its own `trim_k/` under CANDIDATE_DIR.
    The secant starts from the caller's value and the value plus 1.0 and stops inside
    `tol` or at `max_solves`; the accepted case is the one nearest the target, its
    metrics are the candidate's, and `metrics.json` records `trim: {variable, target,
    achieved, value, solves}` so a miss is visible as one.

what this leaves behind

    `run.json` in CANDIDATE_DIR, updated as it goes: the `case_gen.py` arguments used,
    the wall time so far, the trim record, and which directory the accepted case is. A
    caller that kills a run past its time budget scores it with `score.py CANDIDATE_DIR
    --lock ... --label timeout`, and that file is what tells the score what had been
    decided before the kill.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLBOX = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLBOX))
import score  # noqa: E402

CONSTANTS_IMPORT = "from design_constants import *"
"""The line a round's `build.py` has to carry, verbatim."""

CONSTANT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(#.*)?$")
"""One design constant: `NAME = <number>`, an optional comment after it."""

HOOK = "case_args"
"""The one function `hooks.py` may define. Anything else it defines is refused: a hook
that computes a metric, writes a file or redefines a constant is a second scorer, and
`score.py` is the only one."""

OWNED_FLAGS = ("--iterations", "--end-time", "--force", "--yplus", "--dry-run")
"""`case_gen.py` flags this script sets itself; a hook may not."""

TRIM_STEP = 1.0
"""The second trim point is the caller's value plus this: one degree of incidence, the
scale the secant needs a slope over."""


# -- refusals, all of them before anything runs -------------------------------------


def read_constants(path: Path) -> dict[str, float]:
    """`{NAME: value}` off a `design_constants.py`, the lines this convention allows."""
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CONSTANT.match(line)
        if match:
            out[match.group(1)] = float(match.group(2))
    return out


def set_constant(text: str, name: str, value: float) -> str:
    """The same file with one constant's value replaced; refuses a name it does not have."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = CONSTANT.match(line)
        if match and match.group(1) == name:
            comment = f"  {match.group(3)}" if match.group(3) else ""
            lines[index] = f"{name} = {value!r}{comment}"
            return "\n".join(lines) + "\n"
    raise SystemExit(f"design_constants.py has no constant named {name}, which the trim varies")


def hooks_definitions(path: Path) -> list[str]:
    """Every top-level name `hooks.py` defines other than `case_args`, so the refusal
    can say what was there. Imports and a docstring are allowed; nothing else is."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offending: list[str] = []
    seen_hook = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue  # the module docstring
        if isinstance(node, ast.FunctionDef) and node.name == HOOK and not seen_hook:
            seen_hook = True
            continue
        offending.append(_node_name(node))
    if not seen_hook:
        offending.append(f"(no {HOOK}() defined)")
    return offending


def _node_name(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id if isinstance(t, ast.Name) else ast.dump(t)[:40] for t in targets]
        return ", ".join(names)
    return type(node).__name__.lower()


def case_args_from(path: Path) -> list[str]:
    """`hooks.case_args()`, after the file has passed `hooks_definitions`."""
    offending = hooks_definitions(path)
    if offending:
        raise SystemExit(
            f"{path.name} may define only {HOOK}() and imports; it also defines: "
            + ", ".join(offending)
        )
    spec = importlib.util.spec_from_file_location("hooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = getattr(module, HOOK)()
    if not isinstance(args, (list, tuple)) or not all(isinstance(a, str) for a in args):
        raise SystemExit(f"{path.name}: {HOOK}() must return a list of strings, not {args!r}")
    return [str(a) for a in args]


def flag_names(args: list[str]) -> set[str]:
    """The `--flag` names in an argument list, `--flag=value` included."""
    return {a.split("=", 1)[0] for a in args if a.startswith("-") and not _is_number(a)}


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def check_no_override(lock_args: list[str], hook_args: list[str]) -> None:
    """A hook may add to the lock's `case_gen.py` arguments; it may not repeat one."""
    repeated = sorted(flag_names(lock_args) & flag_names(hook_args))
    owned = sorted(set(OWNED_FLAGS) & flag_names(hook_args))
    if repeated:
        raise SystemExit(
            "hooks.case_args() sets " + ", ".join(repeated) + ", which the lock's "
            "case_gen_args already sets; the lock is the harness's physics and a hook "
            "does not override it"
        )
    if owned:
        raise SystemExit(
            "hooks.case_args() sets " + ", ".join(owned) + ", which parametric.py sets "
            "itself from the lock's fidelity"
        )


def prepare(candidate: Path, round_dir: Path) -> list[str]:
    """Copy the round's scripts in beside the constants and read both. Returns the
    hook's `case_args()` (empty without a `hooks.py`)."""
    if not (candidate / "design_constants.py").is_file():
        raise SystemExit(f"{candidate} has no design_constants.py; the caller writes one first")
    build = round_dir / "build.py"
    if not build.is_file():
        raise SystemExit(f"{round_dir} has no build.py")
    text = build.read_text(encoding="utf-8")
    if CONSTANTS_IMPORT not in text:
        raise SystemExit(
            f"{build} does not contain `{CONSTANTS_IMPORT}`; a build that ignores the "
            "constants would score every candidate on the same geometry"
        )
    shutil.copyfile(build, candidate / "build.py")
    hooks = round_dir / "hooks.py"
    if not hooks.is_file():
        return []
    shutil.copyfile(hooks, candidate / "hooks.py")
    return case_args_from(candidate / "hooks.py")


# -- the steps, each a subprocess with its own log ---------------------------------


def run(cmd: list[str], cwd: Path, log: Path) -> subprocess.CompletedProcess:
    """A command run in the case, stdout and stderr into `log`."""
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("+ " + " ".join(cmd) + "\n")
        handle.flush()
        return subprocess.run(cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
                              text=True)


def tail(log: Path, lines: int = 3) -> str:
    try:
        return " | ".join(log.read_text(errors="replace").strip().splitlines()[-lines:])[:400]
    except OSError:
        return ""


def build(case: Path) -> str:
    """`python3 build.py` in the case; the reason it failed, or empty."""
    proc = run([sys.executable, "build.py"], case, case / "log.build")
    if proc.returncode != 0:
        return f"build.py exited {proc.returncode}: {tail(case / 'log.build')}"
    if not (case / "constant" / "polyMesh" / "points").exists():
        return "build.py exited 0 and left no constant/polyMesh"
    return ""


def measure_mesh(case: Path) -> dict:
    """`mesh_look.py`'s payload, with `checkMesh`, written to `look.json` for `score.py`."""
    import mesh_look

    payload = mesh_look.look(case, case / "look.png", check=True)
    (case / "look.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def case_gen_args(lock: dict, hook_args: list[str]) -> list[str]:
    """Everything after the case path on the `case_gen.py` command line."""
    lock_args = [str(a) for a in (lock.get("case_gen_args") or [])]
    fidelity = score.fidelity_of(lock)
    iters = fidelity.get("iters")
    args = [*lock_args, *hook_args]
    body = lock.get("body_patch")
    if body and "--body" not in flag_names(lock_args):
        args += ["--body", str(body)]
    if iters is not None:
        # `fidelity.iters` is a count of iterations, which is `--iterations` on the
        # steady study `case_gen.py` defaults to; only a lock that asks for a transient
        # study means seconds by it, and that flag is `--end-time`.
        args += ["--end-time" if study_of(lock_args) == "transient" else "--iterations",
                 str(iters)]
    args += ["--yplus", "--force"]
    return args


def study_of(args: list[str]) -> str:
    """The `--study` value in a `case_gen.py` argument list, `steady` when unsaid."""
    for index, item in enumerate(args):
        if item.startswith("--study="):
            return item.split("=", 1)[1]
        if item == "--study" and index + 1 < len(args):
            return args[index + 1]
    return "steady"


def dress(case: Path, args: list[str]) -> str:
    proc = run([sys.executable, str(TOOLBOX / "case_gen.py"), str(case), *args], case,
               case / "log.case_gen")
    if proc.returncode != 0:
        return f"case_gen.py exited {proc.returncode}: {tail(case / 'log.case_gen')}"
    return ""


DECOMPOSE = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
numberOfSubdomains {ranks};
method          scotch;
"""


def solve(case: Path, solver: str, ranks: int) -> str:
    """The solve, serial or decomposed; the reason it failed, or empty.

    A non-zero exit is reported and not decided on: the log is what says whether the
    solver diverged or was killed, and `score.py` reads it either way.
    """
    if ranks > 1:
        (case / "system").mkdir(exist_ok=True)
        (case / "system" / "decomposeParDict").write_text(DECOMPOSE.format(ranks=ranks),
                                                          encoding="utf-8")
        proc = run(["decomposePar", "-force"], case, case / "log.decomposePar")
        if proc.returncode != 0:
            return f"decomposePar exited {proc.returncode}: {tail(case / 'log.decomposePar')}"
        proc = run(["mpirun", "-np", str(ranks), solver, "-parallel"], case,
                   case / f"log.{solver}")
        why = "" if proc.returncode == 0 else f"{solver} exited {proc.returncode}"
        recon = run(["reconstructPar", "-latestTime"], case, case / "log.reconstructPar")
        if recon.returncode != 0 and not why:
            why = f"reconstructPar exited {recon.returncode}: {tail(case / 'log.reconstructPar')}"
        return why
    proc = run([solver], case, case / f"log.{solver}")
    return "" if proc.returncode == 0 else f"{solver} exited {proc.returncode}"


# -- one case, start to finish -------------------------------------------------------


class Outcome:
    """Which case holds the answer, and the verdict where this script decided one.

    `label` empty means "nothing here failed; `score.py` decides from the case" -- the
    ordinary end of a solve. A label set here is a failure this script saw directly
    (the build, the mesh, the dressing) with the reason beside it.
    """

    __slots__ = ("case", "label", "detail")

    def __init__(self, case: Path, label: str = "", detail: str = ""):
        self.case = Path(case)
        self.label = label
        self.detail = detail


def evaluate(case: Path, lock: dict, args: list[str], mode: str) -> Outcome:
    why = build(case)
    if why:
        return Outcome(case, "mesh", why)
    payload = measure_mesh(case)
    if not payload.get("polymesh"):
        return Outcome(case, "mesh", "build.py left no constant/polyMesh")
    if not payload.get("checkmesh_ok"):
        return Outcome(case, "mesh", f"checkMesh: {payload.get('checkmesh') or 'no verdict'}")
    if mode == "mesh":
        return Outcome(case)
    why = dress(case, args)
    if why:
        return Outcome(case, "mesh", why)
    solver = str(lock.get("solver") or "simpleFoam")
    ranks = int(score.fidelity_of(lock).get("ranks") or 1)
    why = solve(case, solver, ranks)
    if why:
        # The solver stopped itself or was stopped. What that means is in the log;
        # score.py reads the log, so the detail travels and the label is measured.
        print(why, file=sys.stderr)
    return Outcome(case)


# -- fixed lift: the secant over a design constant ---------------------------------


def trim_case(candidate: Path, index: int, variable: str, value: float) -> Path:
    """`trim_<index>/` under the candidate: the constants with one value changed, and
    the same `build.py` and `hooks.py`."""
    case = candidate / f"trim_{index}"
    case.mkdir(exist_ok=True)
    text = (candidate / "design_constants.py").read_text(encoding="utf-8")
    (case / "design_constants.py").write_text(set_constant(text, variable, value),
                                              encoding="utf-8")
    for name in ("build.py", "hooks.py"):
        if (candidate / name).is_file():
            shutil.copyfile(candidate / name, case / name)
    return case


def next_point(history: list[tuple[float, float]], target: float) -> float:
    """The secant's next abscissa from the last two (x, metric) pairs; the first step
    is the fixed one, and a flat pair steps again rather than dividing by zero."""
    if len(history) == 1:
        return history[0][0] + TRIM_STEP
    (xa, ma), (xb, mb) = history[-2], history[-1]
    if mb == ma:
        return xb + TRIM_STEP
    return xb - (mb - target) * (xb - xa) / (mb - ma)


def trim(candidate: Path, lock: dict, args: list[str], spec: dict,
         record: dict, checkpoint=None) -> Outcome:
    """Solve at fixed `spec["metric"]` by varying `spec["variable"]`, one rebuilt case per
    iteration. `record` is filled in as it goes (it is what `run.json` carries) and
    `checkpoint()` is called after every solve, so a run killed mid-trim leaves the
    record of the solves it did finish."""
    variable = str(spec["variable"])
    metric = str(spec.get("metric") or "Cl")
    target = float(spec["target"])
    tol = float(spec.get("tol") or 0.01)
    max_solves = int(spec.get("max_solves") or 3)
    fraction = float(score.fidelity_of(lock).get("window") or score.DEFAULT_WINDOW)
    solver = str(lock.get("solver") or "simpleFoam")

    constants = read_constants(candidate / "design_constants.py")
    if variable not in constants:
        raise SystemExit(f"design_constants.py has no {variable}, which the trim varies")
    record.update({"variable": variable, "target": target, "achieved": None,
                   "value": constants[variable], "solves": 0})

    history: list[tuple[float, float]] = []
    cases: list[Path] = []
    x = constants[variable]
    for index in range(max_solves):
        case = trim_case(candidate, index, variable, x)
        record["value"] = x
        record["solves"] = index + 1
        outcome = evaluate(case, lock, args, "solve")
        if checkpoint is not None:
            checkpoint()
        if outcome.label:
            return outcome
        # A diverged iteration has no coefficient worth taking a slope over; the
        # candidate is what it is at that angle and the trim stops on it.
        log = score.read_log(score.solver_log(case, solver))
        if log["diverged"]:
            return Outcome(case, "diverged", f"{case.name}: {log['diverged']}")
        metrics, _ = score.force_metrics(case, fraction)
        value = metrics.get(metric)
        if value is None or value != value:
            return Outcome(case, "unconverged", f"no {metric} in the force record of {case.name}")
        history.append((x, value))
        cases.append(case)
        if abs(value - target) <= tol:
            break
        x = next_point(history, target)

    best = min(range(len(history)), key=lambda i: abs(history[i][1] - target))
    record.update({"value": history[best][0], "achieved": history[best][1],
                   "solves": len(history)})
    return Outcome(cases[best])


# -- the command line ----------------------------------------------------------------


def write_run(candidate: Path, info: dict) -> None:
    info["wall_seconds"] = max(0.0, time.time() - float(info["started"]))
    (candidate / "run.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", type=Path,
                    help="the candidate directory; holds design_constants.py already")
    ap.add_argument("--round", type=Path, required=True, dest="round_dir",
                    help="the round directory: build.py and, optionally, hooks.py")
    ap.add_argument("--lock", type=Path, required=True, help="goal.lock.json")
    ap.add_argument("--mode", choices=["mesh", "solve"], required=True,
                    help="mesh: build, mesh, checkMesh, score; solve: all the way")
    args = ap.parse_args(argv)

    candidate = args.candidate.resolve()
    candidate.mkdir(parents=True, exist_ok=True)
    lock = score.load_lock(args.lock)
    hook_args = prepare(candidate, args.round_dir.resolve())
    lock_args = [str(a) for a in (lock.get("case_gen_args") or [])]
    check_no_override(lock_args, hook_args)
    gen_args = case_gen_args(lock, hook_args) if args.mode == "solve" else []

    info: dict = {"started": time.time(), "mode": args.mode, "case_args": gen_args,
                  "trim": None, "case": "."}
    write_run(candidate, info)

    spec = lock.get("trim")
    if args.mode == "solve" and isinstance(spec, dict) and spec:
        record: dict = {}
        info["trim"] = record
        outcome = trim(candidate, lock, gen_args, spec, record,
                       checkpoint=lambda: write_run(candidate, info))
    else:
        outcome = evaluate(candidate, lock, gen_args, args.mode)
    info["case"] = str(outcome.case.relative_to(candidate)) if outcome.case != candidate else "."
    write_run(candidate, info)

    metrics = score.score(outcome.case, lock, info, label=outcome.label,
                          detail=outcome.detail, mesh_only=args.mode == "mesh")
    out = score.write(metrics, candidate / "metrics.json")
    print(f"{metrics['label']}" + (f"  ({metrics['detail']})" if metrics["detail"] else ""))
    for name, value in sorted(metrics["metrics"].items()):
        print(f"  {name} = {value:.5g}")
    if metrics.get("trim"):
        t = metrics["trim"]
        print(f"  trim {t['variable']} = {t['value']:.4g} -> {t.get('achieved')} "
              f"(target {t['target']}, {t['solves']} solves)")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
