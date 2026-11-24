#!/usr/bin/env python
"""The suite, in parallel shards, minus two enumerated and explained lists.

    python -X utf8 scripts/precommit_tests.py

What `.githooks/pre-commit` runs. Takes no arguments. `--all` includes the slow tier,
`--self-check` exercises the output parsing without running the suite, and
`PRECOMMIT_JOBS=n` sets the shard count (`1` runs it serially, which is what to do when
the hook itself is the thing being debugged). Exits non-zero when a test fails twice.

Why this exists. The hook used to be `pytest -q --no-header -x`. Two test files fail for
environment reasons and they sort 13th of 185, so `-x` stopped there every single time
and the hook never once reported anything further down the suite. It hid a real failure
through several commits, and it taught everyone to type `--no-verify`, which is a hook
that protects nothing.

Three things fix that, and they pull against each other:

  - **Report everything, not the first thing.** No `-x`, and the known failures are
    deselected by node id rather than by file, because `test_buildup_cad_export.py` has
    tests that pass and skipping the file would hide those *and* any new failure in it.
  - **Be fast enough that nobody reaches for `--no-verify`.** ~25 minutes serially.
    Little of it is CPU-bound -- subprocesses, sleeps, kernels and Docker -- so the test
    files are dealt into `jobs` shards and each shard is one pytest process. No plugin,
    no new dependency. Four shards, not one per core: the ceiling is a shard's ~1 GB of
    scientific-stack imports, not the CPU.

    Shards and not a process-per-file pool, which was measured and was worse on both
    counts. `conftest.py` imports anthropic, pyvista, matplotlib and textual, so a
    pytest process costs about 5 seconds before it runs anything; 185 of them spend
    ~900s on startup alone where 4 spend 20s. A pool also had four real-kernel files
    starting kernels at once, which made three tests fail that pass under shards.
    Measured: shards 370s green at 8 wide, pool 795-957s with load flakes.

    The shard count is deliberately *below* the core count -- see `main` -- because
    the limit here is RAM per process, not CPU.
  - **Do not go red because the box was busy.** Some tests here are load-sensitive --
    a kernel that "didn't respond in 90 seconds", a timing assertion -- and twice a
    whole shard has died of an access violation with VTK and OpenGL loaded in eight
    processes at once. So a shard that fails is retried once, alone: its named failures
    with pytest's own `--lf`, or the whole shard again if it crashed without naming
    one. A test that then passes is reported as FLAKY rather than blocking the commit;
    a test that fails again is a real failure. A genuine regression fails both times,
    so nothing is hidden, and a shard whose tests did not all run never passes.

Two lists sit beside the hook, both of node ids it deselects, both explaining
themselves:

  - `.githooks/known-failures.txt` -- fails for a reason that is not this repo (a
    drifted local install). Re-run on its own every time, with no deselect, so a listed
    test that has started passing is *reported*. A baseline that only grows is how 17
    failures became invisible in the first place.
  - `.githooks/slow-tests.txt` -- costs a measured fraction of the suite's wall clock.
    Deferred to CI, which runs the whole suite on every push and pull request. `--all`
    runs them here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".githooks" / "known-failures.txt"
SLOW = ROOT / ".githooks" / "slow-tests.txt"


def read_list(path: Path) -> list[str]:
    """The node ids in one of the two lists beside the hook, in file order.

    Blank lines and `#` are comments. Both files hold the same thing -- node ids the
    hook deselects -- and differ only in why, and in what is then checked:
    `known-failures.txt` is re-run to catch one that has started passing,
    `slow-tests.txt` is simply left out unless `--all`.
    """
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def pytest_argv(*args: str) -> list[str]:
    return [sys.executable, "-X", "utf8", "-m", "pytest", "--no-header", *args]


def shards(files: list[Path], jobs: int) -> list[list[Path]]:
    """Deal the files round-robin, biggest first.

    Biggest-first so the shards finish at roughly the same time: dealing in name order
    puts the whole of `test_journey_*` in one shard. File size is a poor stand-in for
    runtime, but the shards only have to be roughly even -- what actually bounds the run
    is the slowest *file*, and no arrangement of shards changes that.
    ponytail: size as a proxy for duration. If one shard starts running away, the fix is
    to make the slow file faster or defer it in slow-tests.txt, not to weigh the shards.
    """
    out: list[list[Path]] = [[] for _ in range(jobs)]
    for i, path in enumerate(sorted(files, key=lambda p: -p.stat().st_size)):
        out[i % jobs].append(path)
    return [s for s in out if s]


def failed_in(log: Path) -> list[str]:
    """The node ids a shard reported, off its `-rf` summary lines.

    `FAILED tests/x.py::test_y - AssertionError: ...`. Splitting on the dash is wrong,
    because a parametrised id can hold spaces *and* dashes --
    `test_the_refusals[foreign-not a face of the shape]` -- so the id is matched by its
    shape instead: no whitespace outside the brackets, and the brackets taken whole.

    Only what gets printed and classified rides on this. Which tests are retried does
    not: that is `--lf` against the shard's own pytest cache.
    """
    text = log.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"^(?:FAILED|ERROR) (tests[/\\]\S+?\.py::[^\s\[]+(?:\[.*?\])?)",
                      text, re.M)


def stale(baseline_output: str, known: list[str]) -> tuple[list[str], list[str]]:
    """Which baselined node ids passed, and which no longer exist.

    Read out of the baseline shard's `-v` output. The two cases are not symmetric: a
    node id pytest cannot find is a *usage* error, so it exits 4 having collected
    nothing at all, and one stale line stops the whole baseline from being checked
    that run. The not-found message names an absolute path rather than the id as
    written, hence the suffix match.
    """
    # `.+?` not `\S+?`: a parametrised id can hold spaces, e.g.
    # `test_the_refusals[foreign-not a face of the shape]`.
    now_passing = re.findall(r"^(tests[/\\]\S*?\.py::.+?) (?:PASSED|XPASS)\b",
                             baseline_output, re.M)
    gone = [line.split("not found: ", 1)[1].replace("\\", "/")
            for line in baseline_output.splitlines() if "ERROR: not found: " in line]
    missing = [n for n in known if any(g.endswith(n.split("/")[-1]) for g in gone)]
    return now_passing, missing


def self_check() -> int:
    """`python scripts/precommit_tests.py --self-check`: the output parsing, which is
    the one part of this that can be wrong without anything noticing."""
    passing, missing = stale(
        "tests/test_a.py::test_one PASSED                     [ 50%]\n"
        "tests/test_a.py::test_two[foreign-not a face of it] PASSED   [ 75%]\n"
        "tests/test_a.py::test_three FAILED                   [100%]\n",
        ["tests/test_a.py::test_one", "tests/test_a.py::test_three"])
    assert passing == ["tests/test_a.py::test_one",
                       "tests/test_a.py::test_two[foreign-not a face of it]"], passing
    assert missing == [], missing

    passing, missing = stale(
        "collecting ... ERROR: not found: C:\\r\\tests\\test_a.py::test_gone\n"
        "(no match in any of [<Module test_a.py>])\n",
        ["tests/test_a.py::test_gone", "tests/test_a.py::test_here"])
    assert passing == [], passing
    assert missing == ["tests/test_a.py::test_gone"], missing
    assert stale("", []) == ([], [])

    log = Path(os.environ.get("TEMP") or "/tmp") / "precommit-selfcheck.log"
    log.write_text(
        "FAILED tests/test_a.py::test_one - AssertionError: assert False is True\n"
        "FAILED tests/test_b.py::test_two[a-b - c] - TypeError: nope\n"
        "ERROR tests/test_c.py::test_three\n"
        "1 failed, 2 passed in 1.00s\n", encoding="utf-8")
    assert failed_in(log) == ["tests/test_a.py::test_one",
                              "tests/test_b.py::test_two[a-b - c]",
                              "tests/test_c.py::test_three"], failed_in(log)
    log.unlink()
    print("precommit: self-check ok")
    return 0


def main() -> int:
    if "--self-check" in sys.argv:
        return self_check()

    # One shard per core, which measured 370s green on this box (8 cores).
    #
    # The honest caveat, because it is the thing most likely to bite someone: what runs
    # out here first is *memory*, not CPU. `conftest.py` and the toolbox tests pull in
    # VTK, scipy, sklearn, pandas and matplotlib, so a shard is a ~1 GB process, and at
    # eight of them this box has twice produced a `MemoryError` during collection and
    # once an access violation with VTK and OpenGL loaded. The retry below is the net
    # for that -- it re-ran the crashed shards alone and they passed -- but the net
    # costs a few minutes. Halving this is the fix on a machine with less RAM:
    #
    #     PRECOMMIT_JOBS=4 git commit ...
    #
    # Fewer shards is slower rather than wrong: at four, the shard holding
    # test_kernel_channel, test_serve and test_cad_agent ran 526s on its own.
    jobs = int(os.environ.get("PRECOMMIT_JOBS") or max(2, os.cpu_count() or 2))
    logs = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
    logs = logs / "openreynolds-precommit"
    logs.mkdir(parents=True, exist_ok=True)
    for old in logs.glob("*.log"):
        old.unlink()

    known = read_list(BASELINE)
    # The slow tier. Deferred unless `--all`, because one 4.5-minute flaky test is no
    # reason for everybody to stop committing; CI runs the whole suite on every push and
    # pull request across eight OS/Python combinations, so nothing here goes unrun.
    slow = [] if "--all" in sys.argv else read_list(SLOW)
    drop = known + slow
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if not files:
        print("precommit: no tests found", file=sys.stderr)
        return 1

    started = time.monotonic()
    running = []

    def spawn(name, argv, log):
        handle = log.open("w", encoding="utf-8")
        running.append((name, subprocess.Popen(argv, cwd=ROOT, stdout=handle,
                                               stderr=subprocess.STDOUT), log, handle))

    # The baseline's own run: the known failures, with no deselect, so one that has
    # started passing is seen. Its failures are the expected ones, so neither its exit
    # code nor its counts belong in what the hook reports -- the `PASSED` lines are the
    # only thing read out of it. It runs alongside the shards; it is seconds of work.
    base_log = logs / "baseline.log"
    if known:
        spawn("baseline", pytest_argv("-v", *known), base_log)

    # Each shard gets its own pytest cache so that `--lf` on a retry means *this
    # shard's* failures. One shared cache would have the shards overwriting each
    # other's `lastfailed`.
    units = shards(files, jobs)
    for n, shard in enumerate(units):
        names = {p.name for p in shard}
        # Only the deselects that name a file in this shard: pytest errors on a
        # `--deselect` whose file it was not asked to collect.
        mine = [f"--deselect={node}" for node in drop
                if node.split("::")[0].split("/")[-1] in names]
        spawn(f"shard{n}",
              pytest_argv("-q", "-rf", "--tb=short", f"-o=cache_dir={logs}/cache{n}",
                          *mine, *(str(p) for p in shard)),
              logs / f"shard{n}.log")

    results = []
    for name, proc, log, handle in running:
        code = proc.wait()
        handle.close()
        results.append((name, code, log))

    # The log each shard's totals are read from. A shard that crashed and was re-run
    # whole is counted from the re-run, because the crashed one counted nothing and
    # reporting "4577 passed" for a green suite of 6032 is a lie of the useful kind.
    counted = {name: log for name, _, log in results if name != "baseline"}

    # Retry, one shard at a time so that load is not a variable, and only the tests that
    # failed (`--lf` against that shard's own cache). A test that passes alone was a
    # flake; a test that fails again is real.
    red: list[str] = []
    flaky: list[str] = []
    for n, shard in enumerate(units):
        name, code, log = next(r for r in results if r[0] == f"shard{n}")
        if not code:
            continue
        before = failed_in(log)
        # A shard that names no test did not fail a test: it crashed. Eight pytest
        # processes each loading VTK and OpenGL have taken an access violation on this
        # box -- pytest's faulthandler dump, not a traceback -- and that is a symptom of
        # a busy machine in the same way a kernel timeout is. So it gets the same
        # treatment, except that there are no node ids to retry and the whole shard runs
        # again. Twice in a row is real, and either way the commit is never let through
        # on a shard whose tests did not all run.
        whole = not before
        if whole:
            print(f"precommit: {name} exited {code} without naming a test -- it "
                  f"crashed. Running the whole shard again, alone...", file=sys.stderr)
        else:
            print(f"precommit: {name} is red; retrying its {len(before)} failure(s) "
                  f"alone to see whether the box was just busy...", file=sys.stderr)
        again = logs / f"retry{n}.log"
        names = {p.name for p in shard}
        mine = [f"--deselect={node}" for node in drop
                if node.split("::")[0].split("/")[-1] in names]
        handle = again.open("w", encoding="utf-8")
        code_again = subprocess.call(
            pytest_argv("-q", "-rf", "--tb=short",
                        *([] if whole else ["--lf"]),
                        f"-o=cache_dir={logs}/cache{n}", *mine,
                        *(str(p) for p in shard)),
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        if whole:
            after = failed_in(again)
            counted[name] = again  # the crashed run counted nothing; this one did
            if code_again and not after:
                red.append(f"{name} crashed twice without naming a test; see {again}")
            elif after:
                red += after
            else:
                flaky.append(f"{name} crashed under load and passed when it had the "
                             f"box to itself")
            continue
        still = set(failed_in(again))
        red += [f for f in before if f in still]
        flaky += [f"{f} failed in the parallel run and passed on its own"
                  for f in before if f not in still]

    counts = dict.fromkeys(("failed", "passed", "skipped"), 0)
    for log in counted.values():
        # Totals off each shard's own summary line, which is the last non-empty line
        # pytest writes ("1 failed, 946 passed, 7 skipped in 180.00s").
        tail = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if l.strip()]
        for number, word in re.findall(r"(\d+) (failed|passed|skipped)",
                                       tail[-1] if tail else ""):
            counts[word] += int(number)
    # After the retries, so the number is the wall clock the developer actually waited.
    elapsed = time.monotonic() - started

    now_passing, missing = stale(
        base_log.read_text(encoding="utf-8", errors="replace")
        if known and base_log.exists() else "", known)
    unchecked = bool(missing) and not now_passing

    print(f"precommit: {counts['failed']} failed, {counts['passed']} passed, "
          f"{counts['skipped']} skipped "
          f"in {elapsed:.0f}s over {len(files)} files in {len(units)} shards "
          f"({len(known)} baselined"
          + (f", {len(slow)} slow deferred to CI -- `--all` runs them" if slow else "")
          + ", see .githooks/)")

    if now_passing or missing:
        print(file=sys.stderr)
        for node in now_passing:
            print(f"precommit: BASELINE STALE -- {node} passes now.", file=sys.stderr)
        for node in missing:
            print(f"precommit: BASELINE STALE -- {node} no longer exists.",
                  file=sys.stderr)
        print(f"precommit: delete those lines from {BASELINE.relative_to(ROOT)} "
              f"(the commit is not blocked on it).", file=sys.stderr)
        if unchecked:
            print("precommit: and the rest of the baseline went unchecked this run -- "
                  "pytest stops at a node id it cannot find.", file=sys.stderr)

    if flaky:
        print(file=sys.stderr)
        for note in flaky:
            print(f"precommit: FLAKY -- {note}.", file=sys.stderr)
        print("precommit: not blocking the commit on those, but they are load-"
              "sensitive and worth a look.", file=sys.stderr)

    if red:
        print(file=sys.stderr)
        print("precommit: the suite is red, so this commit is not going out.",
              file=sys.stderr)
        print("precommit: each of these failed twice, the second time on its own:",
              file=sys.stderr)
        print(file=sys.stderr)
        for node in red:
            print(f"  {node}", file=sys.stderr)
        print(file=sys.stderr)
        print(f"(full output under {logs}; fix it, or --no-verify if you truly mean to)",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
