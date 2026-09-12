#!/usr/bin/env python3
"""One build-up run: the core desk, in a workspace with nothing of ours in it.

    python3 scripts/cad_buildup.py cases                      # the corpus, as it stands
    python3 scripts/cad_buildup.py run T1                     # one case, isolated
    python3 scripts/cad_buildup.py run T1 --steps 20 --seconds 600
    python3 scripts/cad_buildup.py run T7 --work /var/tmp/bu  # a different workspace parent

**This is the enforcement half of §3, and it is not the observation half.** Before a model
is called this makes a workspace root no other run has used, asserts that nothing of ours
is reachable beneath it and that no well-known house path resolves anywhere, and **aborts
rather than runs** if either is false -- a run started dirty cannot be cleaned up
afterwards. It does not sync the toolbox, to a hidden path or read-only or at all, and
the desk it drives (`buildup/core.py`) is briefed without one.

What it does **not** do is look at the run. The contamination grep, the silent-failure
probes, the alarms and the grading are the supervisor's, in its own process
(`scripts/cad_supervise.py`, and the `cad-supervisor` subagent that drives it). This
writes `run.pid` and a heartbeat line per model turn for it to read, and stops there. Two
processes, on purpose: a check that lives in the harness is one refactor away from living
in the desk's path, and then the thing being measured is doing the measuring.

So a supervised run is two commands, the second in another terminal:

    python3 scripts/cad_buildup.py run T1                     # writes <run-dir> on start
    python3 scripts/cad_supervise.py watch <run-dir> --case <case-dir> --deadline 1200

and then `observe` on the same run directory once it ends.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cad_accept import LOCAL_FIXTURES, load_prompts, spend  # noqa: E402
from openreynolds.buildup import core, heartbeat, isolation, record  # noqa: E402

WORK = Path(os.environ.get("OPENREYNOLDS_BUILDUP_WORK")
            or Path.home() / ".openreynolds-buildup" / "work")
"""Where the per-case workspace roots are made, and **outside the repository**.

Not a preference. The task message names the case directory, so a workspace under the tree
would put the repo path into the desk's own thread, and the contamination grep would then
find it in every run's log -- correctly, and uselessly. `_outside_the_repo` refuses it
rather than producing a corpus of runs that are all contaminated for the same reason."""

RUNS = ROOT / "docs" / "cad-buildup" / "runs"
"""Where the records go. In the tree, because a record is the measurement and the
measurement is the point of the phase; the workspace is output and stays out."""

FIXTURE_DIR = "geometry"
"""Where a case's CAD file is put inside its own workspace, after the preflight.

Named for what it is rather than hidden: the desk is told the path, so hiding it would
only cost steps. It is copied in per run, so no two runs share a file."""


def run_id() -> str:
    from openreynolds.store import new_study_id
    return new_study_id()


def _outside_the_repo(parent: Path) -> Path:
    root = Path(parent).expanduser().resolve()
    try:
        root.relative_to(ROOT)
    except ValueError:
        return root
    raise SystemExit(
        f"refusing to work in {root}: it is inside {ROOT}, so the repository path would "
        "appear in the desk's own thread and every run would grep as contaminated. Pass "
        "--work, or set OPENREYNOLDS_BUILDUP_WORK, to somewhere outside the tree.")


def cases() -> dict[str, dict[str, Any]]:
    """The corpus, off the committed prompt files. §4 grows it; nothing here fixes it."""
    return load_prompts()


def prepare(case: str, parent: Path, run_dir: Path) -> tuple[Path, str, dict[str, Any]]:
    """A fresh workspace, asserted clean, with this case's fixture in it -- or an abort.

    The order matters: the assertion runs on an empty directory, and the fixture is copied
    in after it. A fixture copied first would be something the preflight had to be taught
    to ignore, and a check with exceptions in it is a check that grows more."""
    prompts = cases()
    if case not in prompts:
        raise SystemExit(f"no case {case}; there are {', '.join(sorted(prompts))}")
    prompt = prompts[case]
    workspace = isolation.fresh_workspace(_outside_the_repo(parent), case, run_dir.name)
    report = isolation.preflight(workspace)
    print(f"  workspace {workspace} clean ({report['entries']} entries, "
          f"{report['house_names']} house names, no toolbox)")

    geometry = ""
    if prompt["geometry"]:
        name = Path(prompt["geometry"]).name
        source = LOCAL_FIXTURES.get(name)
        if not source or not Path(source).is_file():
            raise SystemExit(f"{case} needs the fixture {name} and it is not on disk")
        target = workspace / FIXTURE_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        geometry = str(target)
        print(f"  fixture {name} -> {geometry}")
    return workspace, geometry, prompt


def own_group() -> int:
    """Become the leader of a new process group, so the run can be killed as a tree.

    A mesher is a grandchild: the desk runs cells in a kernel process, and a cell starts
    `snappyHexMesh` from there. Killing the runner alone leaves both behind -- a
    misconfigured snappy that was going to run for an hour goes on running for an hour,
    on the machine the next case is about to use. With its own group the supervisor can
    signal the whole tree.

    Off by default, because it also detaches the run from the terminal's job control: a
    person running one case by hand should keep their Ctrl-C. A sweep passes
    `--own-group`, and nobody is typing at those.
    """
    try:
        os.setpgrp()
    except OSError:
        pass
    return os.getpgid(0)


def drive(case: str, parent: Path, runs: Path, steps: int, seconds: float,
          run_dir: Path | None = None, group: bool = False) -> int:
    """One case, start to finish, with the record written as it goes.

    `run_dir` lets an orchestrator name the directory instead of reading it back off
    stdout: a sweep has to start the supervisor on a run the moment the run starts, and
    parsing the launcher's own output for the path is a race with the first model turn."""
    from openreynolds.backend.local import LocalBackend, find_bashrc
    from openreynolds.config import Config
    from openreynolds.store import Store

    identifier = run_id()
    run_dir = Path(run_dir) if run_dir else Path(runs) / f"{case}-{identifier}"
    print(f"{case}: run {identifier}")

    # The workspace is asserted clean *before* the run directory exists, so an aborted
    # attempt leaves no empty record behind to be counted as a run that happened.
    workspace, geometry, prompt = prepare(case, parent, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  record {run_dir}")

    cfg = Config.load()
    missing = cfg.model_key_missing()
    if missing:
        raise SystemExit(f"missing configuration: {missing}")
    bashrc = find_bashrc()
    if not bashrc:
        raise SystemExit("no OpenFOAM installation found; set OPENREYNOLDS_FOAM_BASHRC")
    if steps:
        cfg = replace(cfg, mesher_max_steps=steps)
    if seconds:
        cfg = replace(cfg, mesher_max_seconds=seconds)

    backend = LocalBackend(root=workspace, bashrc=bashrc)
    store = Store(cfg.studies_dir, f"buildup-{identifier}")
    desk = core.CoreDesk(cfg, backend, store, str(workspace))

    entry = record.Record(
        run_id=identifier, case=case, arm="core", model=cfg.mesher_model or cfg.model,
        workspace=str(workspace), case_dir=str(workspace / case.lower()),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    record.save(run_dir, entry)

    pulse = heartbeat.Heartbeat(run_dir)
    pulse.start()
    if group:
        (run_dir / "run.pgid").write_text(f"{own_group()}\n", encoding="utf-8")
    log = run_dir / "cells.log"

    def on_turn(**fields: Any) -> None:
        pulse.beat(**{name: value for name, value in fields.items()
                      if name in heartbeat.Beat.__annotations__})
        # Every raw reply, as it arrives. Written per turn rather than at the end because
        # the runs that most need explaining are the ones that get killed -- and without
        # the full text, the thinking length and the block types, the `starved` failure
        # was undiagnosable across three runs.
        record.reply(run_dir, **fields)

    def on_step(step: Any) -> None:
        entry.trace.append(asdict(step))
        entry.n_steps = len(entry.trace)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"# -- cell {entry.n_steps} ({step.seconds:.1f}s, "
                         f"exit {step.exit_code})\n{step.cmd}\n# -- output\n"
                         f"{step.output}\n\n")
        record.save(run_dir, entry)

    desk.on_turn = on_turn
    desk.on_step = on_step

    started = time.time()
    try:
        result = desk.run(prompt["request"], case=case.lower(), geometry=geometry)
    except BaseException as exc:  # noqa: BLE001 - a crash is a result too
        entry.stopped = "provider" if "provider" in str(type(exc)).lower() else "time"
        entry.why = f"{type(exc).__name__}: {exc}"
        entry.seconds = round(time.time() - started, 1)
        (run_dir / "crash.txt").write_text(traceback.format_exc()[-8000:], encoding="utf-8")
        record.save(run_dir, entry)
        pulse.done()
        raise
    finally:
        pulse.done()

    entry.seconds = round(time.time() - started, 1)
    entry.case_dir = result.case_dir
    entry.n_steps = len(result.steps)
    entry.tokens = dict(result.tokens)
    entry.usd = round(spend(result.tokens), 4)
    entry.stopped = result.stopped or "done"
    entry.mesh_exists = bool(result.check and result.check.regions)
    entry.checkmesh_ok = bool(result.check and result.check.ok)
    entry.why = "; ".join(result.check.missing) if result.check else ""
    entry.properties = [{"property": text, "measured": None} for text in prompt["properties"]]
    record.save(run_dir, entry)
    if result.script:
        (run_dir / "build.py").write_text(result.script, encoding="utf-8")

    print(f"  {entry.stopped} in {entry.seconds}s, {entry.n_steps} cells, "
          f"${entry.usd:.2f}, checkMesh {'ok' if entry.checkmesh_ok else 'not ok'}")
    print(f"  now: python3 scripts/cad_supervise.py observe {run_dir} "
          f"--case {entry.case_dir}")
    return 0 if entry.checkmesh_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)

    listing = subs.add_parser("cases", help="the corpus this phase runs against")
    listing.set_defaults(run_command=lambda args: _cases())

    one = subs.add_parser("run", help="one case, in a workspace with nothing of ours in it")
    one.add_argument("case")
    one.add_argument("--work", default=str(WORK), help="where workspace roots are made")
    one.add_argument("--runs", default=str(RUNS), help="where the record goes")
    one.add_argument("--own-group", action="store_true",
                     help="take a process group, so a kill reaches the mesher too")
    one.add_argument("--run-dir", default="",
                     help="name the record directory, for an orchestrator")
    one.add_argument("--steps", type=int, default=0)
    one.add_argument("--seconds", type=float, default=0.0)
    one.set_defaults(run_command=lambda args: drive(
        args.case, Path(args.work), Path(args.runs), args.steps, args.seconds,
        Path(args.run_dir) if args.run_dir else None, args.own_group))

    args = parser.parse_args(argv)
    try:
        return args.run_command(args)
    except isolation.Dirty as exc:
        print(f"aborting rather than running: {exc}", file=sys.stderr)
        return 3


def _cases() -> int:
    for name, prompt in sorted(cases().items()):
        print(f"{name}  {prompt['title']}"
              + (f"  [{Path(prompt['geometry']).name}]" if prompt["geometry"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
