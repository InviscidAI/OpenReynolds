"""Run a case on fast local disk instead of the networked Volume, and reconstruct
the same way.

The case directory lives on the Modal Volume (`/work`), which is a 9p network
filesystem: every file open crosses the network, so the stages that touch many
small files -- decomposePar, the write side of a solve, and above all
reconstructPar -- pay a large per-file tax. Measured on one 160-write case,
`reconstructPar` took 5m23s on the Volume and 27s on container-local `/tmp`; a
40-write case, 39s versus 4s. The solver itself is ~20% slower on the Volume.

`scratch.py` stages a case to `/tmp`, runs a command there, and copies results
back to the Volume as they are written -- so the durable copy on the Volume is
never more than one checkpoint behind, and a preempted run resumes from
`latestTime` with nothing lost. Reconstruction is a separate command against the
decomposed data already on the Volume, so a lost reconstruct never costs the
solve.

    # solve on local disk, checkpointing back every 60s; resumes from latestTime
    python3 scratch.py run /work/<study>/case -- mpirun -np 16 pimpleFoam -parallel

    # reconstruct on local disk; scoped and idempotent (only what is missing)
    python3 scratch.py reconstruct /work/<study>/case --latest --fields "U p"
    python3 scratch.py reconstruct /work/<study>/case --time 120:135
    python3 scratch.py reconstruct /work/<study>/case            # all new times

    # what an interrupted run left on disk, and which time can be restarted from
    python3 scratch.py resume /work/<study>/case

`resume` reads and prints; it edits nothing. It is here because the promise above --
"a preempted run resumes from `latestTime` with nothing lost" -- has one hole, and the
hole is silent. `checkpoint` says a time directory caught mid-write is completed by the
next checkpoint, which is true of every checkpoint except the one that never comes:
when the Sandbox is preempted, the newest time directory on the Volume is whatever
`rsync` had reached, and `startFrom latestTime` then sends the solver into it. So
`resume` compares the newest time's files with the time before it, checks each one for
the end divider OpenFOAM writes last, and names the newest time that is whole. A short
newest time is a directory to move aside, not a solve to run again.

It uses only `rsync`, `cp` and the OpenFOAM tools already on the instance, and
sources nothing: the command it runs inherits the environment it was given, which
is the OpenFOAM one when launched from a job or a sourced shell.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRATCH_ROOT = "/tmp/reynolds-scratch"
"""Container-local disk (overlay fs), not the Volume. Wiped when the Sandbox stops,
which is exactly why the Volume is checkpointed alongside it."""

DEFAULT_EVERY_S = 60.0
"""How often the run loop copies new results back to the Volume. The bound on what a
preemption costs is one of these intervals (plus whatever the solver had not yet
written), so it is deliberately short relative to a solve."""


# --- pure helpers (unit-tested; no container needed) ------------------------

def scratch_dir(case: str, root: str = SCRATCH_ROOT) -> str:
    """A stable local directory for a Volume case path.

    Keyed by the full path's hash so two cases with the same basename
    (`case`, `mesh/case`) never collide on local disk, and the same case always
    maps to the same scratch dir so a resumed run re-stages onto its own files."""
    name = Path(case.rstrip("/")).name or "case"
    digest = hashlib.sha1(case.rstrip("/").encode("utf-8")).hexdigest()[:10]
    return f"{root.rstrip('/')}/{name}-{digest}"


def is_time_name(name: str) -> bool:
    """Whether a directory name is an OpenFOAM time directory (a bare number).

    `0`, `0.005`, `1.5e-05` and `100.00510749` are times; `constant`, `system`,
    `0.orig`, `processors16` and `processor0` are not."""
    if name in ("constant", "system"):
        return False
    try:
        float(name)
    except ValueError:
        return False
    return True


def time_dirs(names) -> list[str]:
    """The time-directory names among `names`, ordered by their numeric value."""
    times = [n for n in names if is_time_name(n)]
    return sorted(times, key=float)


def processor_container(names) -> str | None:
    """Where decomposed times live: `processorsNN` for the collated file handler,
    else the first `processorN`. None if the case is not decomposed."""
    collated = sorted(n for n in names if n.startswith("processors") and n[len("processors"):].isdigit())
    if collated:
        return collated[0]
    procs = sorted(n for n in names if n.startswith("processor") and n[len("processor"):].isdigit())
    return procs[0] if procs else None


def pending_times(reconstructed, decomposed) -> list[str]:
    """Decomposed times not yet present in the reconstructed (case-root) set.

    This is what `reconstruct` would produce; an empty list means the case is
    already reconstructed and the whole stage-and-run can be skipped. `0` and
    `constant` are never counted -- decomposePar writes a `0` per processor that
    reconstruct does not add to the root."""
    have = set(time_dirs(reconstructed))
    want = [t for t in time_dirs(decomposed) if t != "0"]
    return [t for t in want if t not in have]


def reconstruct_argv(latest: bool = False, time_range: str | None = None,
                     new_times: bool = True, fields: str | None = None) -> list[str]:
    """The `reconstructPar` command for a scope.

    `--latest` and `--time A:B` are exclusive ways to bound which times; with
    neither, `-newTimes` reconstructs only times missing from the case root, which
    is both the cheap default and what makes a re-run idempotent. `fields` is a
    space- or comma-separated list, passed as OpenFOAM's `-fields '(U p)'`."""
    argv = ["reconstructPar"]
    if latest:
        argv.append("-latestTime")
    elif time_range:
        lo, _, hi = time_range.partition(":")
        argv += ["-time", f"{lo},{hi}" if hi else lo]
    elif new_times:
        argv.append("-newTimes")
    if fields:
        names = " ".join(f.strip() for f in fields.replace(",", " ").split() if f.strip())
        argv += ["-fields", f"({names})"]
    return argv


FOAM_END = b"// *****"
"""The start of the end divider OpenFOAM writes as the last line of every file it
writes, ascii or binary (`case_gen.py:109` writes the same line into the dictionaries
it generates). A field file without it was not finished: either the solver was killed
between opening it and closing it, or `rsync` copied it while it was still being
written. Checked on the tail of the file, so it costs one seek per field."""

TAIL_BYTES = 256
"""How much of a file's tail to read looking for `FOAM_END`. The divider plus the
newline after it is 76 bytes; 256 covers a trailing blank line or two and is still one
read."""


def ends_whole(tail: bytes) -> bool:
    """Whether a file's last bytes carry OpenFOAM's end divider."""
    return FOAM_END in tail


def short_of(previous: set[str], newest: set[str]) -> list[str]:
    """What the time before the newest one holds that the newest one does not.

    Only the newest time is at risk, so only the newest time is compared, and it is
    compared with its own predecessor rather than with a list of expected fields: which
    fields a case writes is the case's business (`writeControl`, `functions`, a
    `purgeWrite`), and the one thing known for certain about a finished write is that it
    wrote what the write before it wrote. `uniform` is a directory of its own and is
    counted like a field, because `uniform/time` is what a restart reads the index and
    the deltaT from."""
    return sorted(previous - newest)


def rsync_argv(src: str, dst: str, delete: bool = False) -> list[str]:
    """`rsync` to copy `src/` into `dst`, creating `dst`. `-a` preserves times so a
    later `--update`-style pass copies only what changed; `--delete` mirrors
    removals (a purgeWrite on the scratch side), off by default so a checkpoint
    only ever adds to the durable copy."""
    argv = ["rsync", "-a"]
    if delete:
        argv.append("--delete")
    return argv + [f"{src.rstrip('/')}/", dst.rstrip("/")]


# --- execution (needs the instance) -----------------------------------------

def _run(argv, cwd=None) -> int:
    """Run a command, streaming its output, and return its exit code."""
    print(f"+ {' '.join(argv)}", flush=True)
    return subprocess.run(argv, cwd=cwd).returncode


def stage(case: str, local: str) -> None:
    """Copy the Volume case onto local disk (creating it or refreshing a resume)."""
    os.makedirs(local, exist_ok=True)
    rc = _run(rsync_argv(case, local))
    if rc != 0:
        raise SystemExit(f"scratch: could not stage {case} -> {local} (rsync rc={rc})")


def checkpoint(local: str, case: str, delete: bool = False) -> int:
    """Copy new/changed results from local disk back to the Volume case.

    rsync sends only what changed, so a checkpoint mid-solve is cheap; a
    time directory caught mid-write is completed by the next checkpoint, and the
    final sync after the command exits is authoritative."""
    return _run(rsync_argv(local, case, delete=delete))


class _Checkpointer(threading.Thread):
    """Copies results back to the Volume every `every` seconds while a run is going."""

    def __init__(self, local: str, case: str, every: float):
        super().__init__(daemon=True)
        # NB: not `self._stop` -- threading.Thread has its own private `_stop()` method
        # that its `join()` calls, and shadowing it with an Event makes join() try to
        # call the Event and raise "'Event' object is not callable".
        self.local, self.case, self.every = local, case, every
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.wait(self.every):
            try:
                checkpoint(self.local, self.case)
            except Exception as exc:  # noqa: BLE001 - a failed checkpoint must not kill the run
                print(f"scratch: checkpoint failed (will retry): {exc}", flush=True)

    def stop(self) -> None:
        self._done.set()


def run_on_scratch(case: str, command: list[str], every: float = DEFAULT_EVERY_S,
                   root: str = SCRATCH_ROOT, keep: bool = False) -> int:
    """Stage `case`, run `command` on local disk with periodic checkpoints, sync
    back, and return the command's exit code. The Volume copy is never more than
    `every` seconds behind, so a preemption resumes from `latestTime`."""
    local = scratch_dir(case, root)
    t0 = time.time()
    stage(case, local)
    print(f"scratch: staged {case} -> {local} in {time.time() - t0:.1f}s", flush=True)

    ticker = _Checkpointer(local, case, every)
    ticker.start()
    try:
        rc = _run(list(command), cwd=local)
    finally:
        ticker.stop()
        ticker.join(timeout=every + 30)
        t1 = time.time()
        checkpoint(local, case)  # authoritative final sync
        print(f"scratch: synced results back in {time.time() - t1:.1f}s", flush=True)
        if not keep:
            _run(["rm", "-rf", local])
    return rc


def reconstruct(case: str, latest: bool = False, time_range: str | None = None,
                new_times: bool = True, fields: str | None = None,
                every: float = DEFAULT_EVERY_S, root: str = SCRATCH_ROOT,
                keep: bool = False) -> int:
    """Reconstruct `case` on local disk and sync the reconstructed times back.

    Skips the whole stage-and-run when nothing is pending (the default `-newTimes`
    scope is already satisfied), so calling it again is nearly free."""
    names = os.listdir(case) if os.path.isdir(case) else []
    container = processor_container(names)
    if container is None:
        raise SystemExit(f"scratch: {case} has no decomposed (processor*) data to reconstruct")
    if new_times and not latest and not time_range:
        proc_times = os.listdir(os.path.join(case, container))
        if not pending_times(names, proc_times):
            print("scratch: nothing to reconstruct (case is up to date)", flush=True)
            return 0
    return run_on_scratch(
        case, reconstruct_argv(latest, time_range, new_times, fields),
        every=every, root=root, keep=keep,
    )


def _listing(path: str) -> set[str]:
    try:
        return set(os.listdir(path))
    except OSError:
        return set()


def _tail(path: str) -> bytes:
    """The last `TAIL_BYTES` of a file, or b"" if it cannot be read."""
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, os.path.getsize(path) - TAIL_BYTES))
            return handle.read()
    except OSError:
        return b""


def time_homes(case: str) -> list[str]:
    """Every directory under `case` that holds time directories of its own.

    One entry for an undecomposed case (the case root), one for the collated layout
    (`processorsN`), and one per rank for the default decomposed layout -- which is
    why this is a list. `rsync` can be interrupted part-way through any one of them, so
    a time that is whole in `processor0` says nothing about the same time in
    `processor7`."""
    names = _listing(case)
    ranks = sorted(n for n in names if n.startswith("processor") and n[len("processor"):].isdigit())
    collated = sorted(n for n in names
                      if n.startswith("processors") and n[len("processors"):].isdigit())
    if collated:
        return [os.path.join(case, collated[0])]
    if ranks:
        return [os.path.join(case, n) for n in ranks]
    return [case]


def whole_times(home: str) -> tuple[list[str], list[str], list[str]]:
    """`(times, missing, unfinished)` for one directory of time directories.

    `missing` is what the time before the newest one holds and the newest does not;
    `unfinished` names the files in the newest time with no end divider. Both are empty
    for a run that stopped between writes, which is the usual case and the whole point
    of saying so."""
    times = time_dirs(_listing(home))
    if len(times) < 1:
        return times, [], []
    newest = os.path.join(home, times[-1])
    missing = (
        short_of(_listing(os.path.join(home, times[-2])), _listing(newest))
        if len(times) > 1 else []
    )
    unfinished = [
        name for name in sorted(_listing(newest))
        if os.path.isfile(os.path.join(newest, name))
        and not ends_whole(_tail(os.path.join(newest, name)))
    ]
    return times, missing, unfinished


def resume(case: str) -> int:
    """Say what an interrupted run left on disk and which time can be restarted from.

    Reads and prints. The two edits it names -- moving a short time directory aside and
    setting `startFrom latestTime` -- are left to whoever is reading, because which of
    them is right depends on what the run was for, and a script that moves a solver's
    output on its own guess is a worse failure than a restart that stops with an
    error."""
    if not os.path.isdir(case):
        raise SystemExit(f"scratch: no case at {case}")
    homes = time_homes(case)
    layout = ("undecomposed" if homes == [case]
              else f"{len(homes)} rank director{'y' if len(homes) == 1 else 'ies'}")
    print(f"case:   {case} ({layout})", flush=True)

    flawed: dict[str, list[str]] = {}
    seen: list[set[str]] = []
    for home in homes:
        times, missing, unfinished = whole_times(home)
        where = os.path.relpath(home, case) if home != case else "."
        seen.append(set(times))
        if not times:
            print(f"{where}: no time directories", flush=True)
            continue
        print(f"{where}: {len(times)} time(s), {times[0]} .. {times[-1]}", flush=True)
        if missing:
            flawed.setdefault(times[-1], []).append(where)
            print(f"  {times[-1]}/ is short of {times[-2]}/ by: "
                  + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""), flush=True)
        if unfinished:
            flawed.setdefault(times[-1], []).append(where)
            print("  no end divider (so written, not finished): "
                  + ", ".join(unfinished[:8]) + ("..." if len(unfinished) > 8 else ""),
                  flush=True)

    # A time counts only where every one of those directories has it: a parallel
    # restart and `reconstructPar` both read all of them, so a time written in
    # processor0 and not in processor7 is not a time this case can be restarted from.
    everywhere = time_dirs(set.intersection(*seen) if seen else set())
    for home, times in zip(homes, seen):
        where = os.path.relpath(home, case) if home != case else "."
        for extra in sorted(set(times) - set(everywhere), key=float):
            flawed.setdefault(extra, []).append(where)
            print(f"  {extra}/ is in {where} and not in all the others", flush=True)

    whole = [t for t in everywhere if t not in flawed]
    if not whole:
        print("\nnothing whole to resume from: " + (
            "this case holds no time directory present in every one of those "
            "directories." if not everywhere else
            f"the newest time present everywhere is {everywhere[-1]}, and it is one of "
            "the times named above."), flush=True)
        return 1
    print(f"\nresume from {whole[-1]}: set `startFrom latestTime;` in "
          "system/controlDict, raise endTime, and start the solver again -- every write "
          "so far is kept.", flush=True)
    # `latestTime` resolves to the newest directory on disk by name, so a partial one
    # left where it was is what the solver would read. Naming the move is the whole
    # value of this command.
    later = sorted(flawed, key=float)
    later = [t for t in later if float(t) > float(whole[-1])]
    if later:
        print(f"first move {', '.join(later)} aside wherever they exist, because "
              "`latestTime` resolves to the newest directory on disk and not to the "
              "newest whole one.", flush=True)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch", default=SCRATCH_ROOT, help="local scratch root (default %(default)s)")
    p.add_argument("--every", type=float, default=DEFAULT_EVERY_S,
                   help="seconds between checkpoints back to the Volume (default %(default)s)")
    p.add_argument("--keep", action="store_true", help="leave the local scratch copy in place")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="stage a case, run a command on local disk, checkpoint back")
    r.add_argument("case")
    r.add_argument("command", nargs=argparse.REMAINDER,
                   help="the command to run, after a `--` (e.g. -- mpirun -np 16 pimpleFoam -parallel)")

    rc = sub.add_parser("reconstruct", help="reconstruct on local disk, scoped and idempotent")
    rc.add_argument("case")
    g = rc.add_mutually_exclusive_group()
    g.add_argument("--latest", action="store_true", help="only the latest time")
    g.add_argument("--time", dest="time_range", help="a single time or a A:B range")
    rc.add_argument("--fields", help='space/comma list, e.g. "U p" (default: all fields)')

    rs = sub.add_parser("resume", help="what an interrupted run left, and which time is whole")
    rs.add_argument("case")

    a = p.parse_args(argv)
    if a.cmd == "resume":
        return resume(a.case)
    if a.cmd == "run":
        command = a.command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            p.error("run needs a command after `--`")
        return run_on_scratch(a.case, command, every=a.every, root=a.scratch, keep=a.keep)
    return reconstruct(a.case, latest=a.latest, time_range=a.time_range,
                       fields=a.fields, every=a.every, root=a.scratch, keep=a.keep)


if __name__ == "__main__":
    sys.exit(main())
