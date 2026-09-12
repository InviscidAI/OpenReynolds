#!/usr/bin/env python3
"""A corpus through the core desk, once each, and the paired table that compares two.

    python3 scripts/cad_sweep.py run --label core --cases T1,T2,T3 --parallel 2
    python3 scripts/cad_sweep.py run --label core+audit --cases all
    python3 scripts/cad_sweep.py table <sweep-id>
    python3 scripts/cad_sweep.py table <sweep-id> --against <baseline-sweep-id>
    python3 scripts/cad_sweep.py list

**A launcher, not an observer.** For each case it starts `cad_buildup.py run` and
`cad_supervise.py watch` as two processes and then calls `cad_supervise.py observe` -- the
same three commands a person runs by hand, in the same order, in the same processes. It
reads records afterwards; it never looks at a live run, and nothing here imports a probe.
One out-of-band observer is the rule for the whole phase and a sweep driver is not a
second one.

**One run per case.** Repeats buy precision on a single case's pass rate, which is not a
number anybody decides on; the decision is "did this addition move the corpus", and that
comparison is **paired within case**, so the unit of evidence is the case and breadth is
worth more than depth. `--repeat` exists for the two narrow jobs that do want it:
calibrating variance once, and asking whether an addition closed one specific intermittent
failure.

**A dirty environment voids the sweep, not the run.** `cad_buildup.py` exits 3 when a
house path resolves; the first time that happens the sweep stops. One contaminated run can
be discarded, but an environment that was dirty at run 6 was dirty at run 1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cad_buildup import WORK, cases as corpus  # noqa: E402
from openreynolds.buildup import record  # noqa: E402

SWEEPS = ROOT / "docs" / "cad-buildup" / "sweeps"

NOISE_STEPS = 8
"""How big a per-case step difference has to be before it is a difference at all.

Measured on T1 with Opus at medium: the same prompt, unchanged, lands within a band of
three to eight steps. So a case that moves by eight or fewer has not been shown to move,
and a report that reads such a flip as a result is reading the sampling. It is one
number from one case and it is the threshold the whole comparison rests on -- re-measure
it when the model or the brief changes, and say so here when you do."""

DEADLINE_S = 1500.0
"""What the supervisor is given before it calls a run wedged on its own authority.

Longer than the desk's own 900 s budget plus the finish check, so a healthy run always
ends on its own terms and this only fires on one that cannot."""


def sweep_id() -> str:
    from openreynolds.store import new_study_id
    return new_study_id()


def git_sha() -> str:
    """What the core was when this sweep ran.

    A re-sweep after an addition compares two sweeps, and "the same core apart from the
    addition" has to be checkable rather than remembered."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip() + ("+dirty" if (dirty.stdout or "").strip() else "")
    except Exception:  # noqa: BLE001 - a sweep is not a git client
        return "unknown"


# -- running -------------------------------------------------------------------


def one_run(case: str, run_dir: Path, work: str, steps: int, seconds: float) -> dict[str, Any]:
    """The three commands, in their own processes, in the order a person runs them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts" / "cad_buildup.py"), "run", case,
         "--run-dir", str(run_dir), "--work", work, "--own-group",
         *(["--steps", str(steps)] if steps else []),
         *(["--seconds", str(seconds)] if seconds else [])],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    case_dir = _case_dir(run_dir)
    watcher = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts" / "cad_supervise.py"), "watch",
         str(run_dir), *(["--case", case_dir] if case_dir else []),
         "--deadline", str(DEADLINE_S)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    runner_out, _ = runner.communicate()
    watcher.communicate()
    (run_dir / "runner.log").write_text(runner_out or "", encoding="utf-8")

    if runner.returncode == 3:
        return {"case": case, "run_dir": str(run_dir), "aborted": True,
                "why": (runner_out or "").strip().splitlines()[0] if runner_out else ""}

    observed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cad_supervise.py"), "observe",
         str(run_dir), *(["--case", _case_dir(run_dir)] if _case_dir(run_dir) else []),
         *(["--spec", str(run_dir / "spec.json")] if (run_dir / "spec.json").is_file() else [])],
        cwd=ROOT, capture_output=True, text=True)
    try:
        graded = json.loads(observed.stdout or "{}")
    except ValueError:
        graded = {}
    return {"case": case, "run_dir": str(run_dir), "aborted": False, **graded}


def _case_dir(run_dir: Path) -> str:
    """Where the desk is working, off the record the runner wrote as it started.

    Polled rather than assumed: the record exists within a second of the run starting and
    carries the path, and guessing it here would be a second place that knows how a case
    directory is named."""
    for _ in range(60):
        data = record.load(run_dir)
        if data.get("case_dir"):
            return str(data["case_dir"])
        time.sleep(0.5)
    return ""


def drive(label: str, names: list[str], repeat: int, parallel: int, work: str,
          steps: int, seconds: float, baseline: str = "") -> int:
    identifier = sweep_id()
    directory = SWEEPS / f"{label}-{identifier}"
    (directory / "runs").mkdir(parents=True, exist_ok=True)

    from openreynolds.config import Config
    cfg = Config.load()
    manifest = {
        "sweep_id": identifier, "label": label, "cases": names, "repeat": repeat,
        "model": cfg.mesher_model or cfg.model, "effort": cfg.mesher_effort,
        "git_sha": git_sha(), "noise_steps": NOISE_STEPS,
        "baseline": _resolve_baseline(baseline),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": [],
    }
    _save(directory, manifest)
    print(f"sweep {directory.name}: {len(names)} cases x {repeat}, "
          f"{manifest['model']} at {manifest['effort']}, core {manifest['git_sha']}")
    if manifest["baseline"]:
        print(f"  to be read against {manifest['baseline']}")

    queue = [(case, index) for case in names for index in range(1, repeat + 1)]
    results: list[dict[str, Any]] = []
    aborted = False

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {}
        for case, index in queue:
            name = case if repeat == 1 else f"{case}-{index}"
            run_dir = directory / "runs" / name
            futures[pool.submit(one_run, case, run_dir, work, steps, seconds)] = name
        for future, name in futures.items():
            outcome = future.result()
            results.append(outcome)
            manifest["runs"] = results
            _save(directory, manifest)
            if outcome.get("aborted"):
                aborted = True
                print(f"  {name}: ABORTED -- {outcome.get('why', '')}")
                continue
            print(f"  {name}: {outcome.get('stopped', '?')}, "
                  f"{outcome.get('n_steps', 0)} cells, "
                  f"checkMesh {'ok' if outcome.get('checkmesh_ok') else 'not ok'}, "
                  f"${outcome.get('usd', 0):.2f}"
                  + (" CONTAMINATED" if outcome.get("contaminated") else "")
                  + (f" fired={outcome['fired']}" if outcome.get("fired") else ""))

    manifest["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save(directory, manifest)
    print(f"\n{directory}")
    if aborted:
        print("a run aborted on a dirty environment: this sweep is void, not noisy")
        return 3
    return 0


def _save(directory: Path, manifest: dict[str, Any]) -> None:
    (directory / "sweep.json").write_text(json.dumps(manifest, indent=2, default=str),
                                          encoding="utf-8")


# -- reading -------------------------------------------------------------------


def _resolve_baseline(name: str) -> str:
    """The sweep this one is to be compared with, named at the time rather than later.

    **Iteration N's sweep is iteration N+1's baseline** -- the "after" of an addition is
    the "before" of the next one -- so the chain is a fact about the sweep and belongs in
    its manifest. Reconstructing it afterwards from timestamps works right up until two
    sweeps were run on the same day for different reasons.

    `latest` means the newest sweep on disk, which is what the chain almost always wants.
    """
    if not name:
        return ""
    if name.strip().lower() == "latest":
        found = all_sweeps()
        return found[-1][0].name if found else ""
    directory, _manifest = load_sweep(name)
    return directory.name


def all_sweeps() -> list[tuple[Path, dict[str, Any]]]:
    """Every sweep on disk, oldest first -- **by when it ran**, not by its name.

    A sweep directory is `<label>-<id>`, so sorting the names puts `core+audit-...` before
    `core-...`: `+` sorts below `-`. Which is to say that a lexicographic `latest` is right
    only while every sweep carries the same label, and the whole point of a chain is that
    they do not.
    """
    found: list[tuple[Path, dict[str, Any]]] = []
    for directory in SWEEPS.glob("*/"):
        try:
            manifest = json.loads((directory / "sweep.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found.append((directory, manifest))
    return sorted(found, key=lambda row: (str(row[1].get("started_at") or ""),
                                          row[0].stat().st_mtime))


def load_sweep(name: str) -> tuple[Path, dict[str, Any]]:
    directory = Path(name) if Path(name).is_dir() else SWEEPS / name
    if not (directory / "sweep.json").is_file():
        matches = sorted(SWEEPS.glob(f"*{name}*"))
        if not matches:
            raise SystemExit(f"no sweep matching {name} under {SWEEPS}")
        directory = matches[-1]
    return directory, json.loads((directory / "sweep.json").read_text(encoding="utf-8"))


def rows(directory: Path) -> dict[str, dict[str, Any]]:
    """Every run in a sweep, keyed by the name its directory carries."""
    found: dict[str, dict[str, Any]] = {}
    for run_dir in sorted((directory / "runs").iterdir()):
        if run_dir.is_dir() and (data := record.load(run_dir)):
            found[run_dir.name] = data
    return found


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact binomial on the cases that moved. Ties are dropped, as they must
    be: a case that did not move is not evidence either way."""
    total = better + worse
    if not total:
        return 1.0
    fewer = min(better, worse)
    tail = sum(comb(total, k) for k in range(fewer + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def table(name: str, against: str = "") -> int:
    directory, manifest = load_sweep(name)
    here = rows(directory)
    if against.strip().lower() == "latest":
        against = _resolve_baseline("latest")
    if not against and manifest.get("baseline"):
        # A sweep that was told what it follows does not have to be told again at every
        # reading. `--against` still wins, for the comparison nobody planned.
        against = str(manifest["baseline"])
    print(f"# {directory.name}")
    print(f"{manifest['model']} at {manifest['effort']}, core {manifest['git_sha']}, "
          f"{len(here)} runs")

    dirty = [key for key, data in here.items() if data.get("contaminated")]
    if dirty:
        print(f"\nCONTAMINATED, discarded from the baseline: {', '.join(dirty)}")
    clean = {key: data for key, data in here.items() if not data.get("contaminated")}

    if not against:
        print("\n| case | ended | cells | first mesh | checkMesh | $ | probes fired |")
        print("|---|---|---|---|---|---|---|")
        for key, data in clean.items():
            fired = [row["id"] for row in data.get("probes", [])
                     if row.get("state") == "fired"]
            print(f"| {key} | {data.get('stopped', '?')} | {data.get('n_steps', 0)} | "
                  f"{data.get('first_mesh_step') or '-'} | "
                  f"{'ok' if data.get('checkmesh_ok') else 'no'} | "
                  f"{data.get('usd', 0):.2f} | {', '.join(fired) or '-'} |")
        _totals(clean)
        return 0

    baseline_dir, baseline_manifest = load_sweep(against)
    before = {key: data for key, data in rows(baseline_dir).items()
              if not data.get("contaminated")}
    print(f"against {baseline_dir.name} ({baseline_manifest['model']} at "
          f"{baseline_manifest['effort']}, core {baseline_manifest['git_sha']})")
    if baseline_manifest.get("model") != manifest.get("model") or \
            baseline_manifest.get("effort") != manifest.get("effort"):
        print("\nWARNING: the model or the effort moved between these sweeps, so the "
              "difference below is not the addition's.")

    noise = int(manifest.get("noise_steps") or NOISE_STEPS)
    print(f"\n| case | cells before | after | delta | checkMesh before -> after |")
    print("|---|---|---|---|---|")
    better = worse = 0
    for key in sorted(set(before) & set(clean)):
        was, now = before[key], clean[key]
        delta = int(now.get("n_steps", 0)) - int(was.get("n_steps", 0))
        moved = "noise" if abs(delta) <= noise else ("fewer" if delta < 0 else "more")
        if moved == "fewer":
            better += 1
        elif moved == "more":
            worse += 1
        print(f"| {key} | {was.get('n_steps', 0)} | {now.get('n_steps', 0)} | "
              f"{delta:+d} ({moved}) | "
              f"{'ok' if was.get('checkmesh_ok') else 'no'} -> "
              f"{'ok' if now.get('checkmesh_ok') else 'no'} |")

    only_here = sorted(set(clean) - set(before))
    only_there = sorted(set(before) - set(clean))
    if only_here or only_there:
        print(f"\nunpaired, excluded from the test: "
              f"{', '.join(sorted(only_here + only_there))}")

    print(f"\n{better} cases used fewer cells, {worse} more, "
          f"the rest within the {noise}-cell noise band.")
    print(f"sign test over the cases that moved: p = {sign_test(better, worse):.3f}")
    print("The verdict is this line, not any single row: one case's flip is sampling.")
    _totals(clean, before)
    return 0


def _totals(now: dict[str, dict[str, Any]], before: dict[str, dict[str, Any]] | None = None
            ) -> None:
    def summarise(data: dict[str, dict[str, Any]]) -> str:
        meshed = sum(1 for row in data.values() if row.get("checkmesh_ok"))
        cells = sum(int(row.get("n_steps", 0)) for row in data.values())
        spend = sum(float(row.get("usd", 0)) for row in data.values())
        return f"{meshed}/{len(data)} checkMesh ok, {cells} cells, ${spend:.2f}"

    print(f"\ntotal: {summarise(now)}")
    if before:
        print(f"before: {summarise(before)}")


def listing() -> int:
    for directory, manifest in all_sweeps():
        chained = manifest.get("baseline")
        print(f"{directory.name}  {manifest.get('model', '?')} "
              f"{manifest.get('effort', '?')}  core {manifest.get('git_sha', '?')}  "
              f"{len(manifest.get('runs', []))} runs"
              + (f"  <- {chained}" if chained else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The commands, built separately from being run, so a test can read a default.

    A default that only exists inside `main` is a default nothing can check without
    launching a sweep, and `--repeat 1` is the one decision in this file that a reader
    most needs to be able to confirm."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)

    one = subs.add_parser("run", help="a corpus through the core desk, once each")
    one.add_argument("--label", required=True,
                     help="what is under test: `core`, `core+cad_convert`, ...")
    one.add_argument("--cases", default="all", help="T1,T2,... or `all`")
    one.add_argument("--repeat", type=int, default=1,
                     help="runs per case; 1 unless you are calibrating variance")
    one.add_argument("--baseline", default="",
                     help="the sweep this one is to be read against: a sweep id, or "
                          "`latest` for the previous iteration's")
    one.add_argument("--parallel", type=int, default=2)
    one.add_argument("--work", default="", help="where workspace roots are made")
    one.add_argument("--steps", type=int, default=0)
    one.add_argument("--seconds", type=float, default=0.0)
    one.set_defaults(run_command=lambda args: drive(
        args.label, _names(args.cases), args.repeat, args.parallel,
        args.work or str(WORK), args.steps, args.seconds, args.baseline))

    two = subs.add_parser("table", help="one sweep, or two compared")
    two.add_argument("sweep")
    two.add_argument("--against", default="",
                     help="the baseline to pair against: a sweep id, `latest`, or "
                          "omitted to use the one the sweep recorded")
    two.set_defaults(run_command=lambda args: table(args.sweep, args.against))

    three = subs.add_parser("list", help="the sweeps on disk")
    three.set_defaults(run_command=lambda args: listing())

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run_command(args)


def _names(text: str) -> list[str]:
    if text.strip().lower() in ("all", "*"):
        return sorted(corpus())
    chosen = [name.strip().upper() for name in text.split(",") if name.strip()]
    known = corpus()
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise SystemExit(f"no such case: {', '.join(unknown)}; "
                         f"there are {', '.join(sorted(known))}")
    return chosen


if __name__ == "__main__":
    raise SystemExit(main())
