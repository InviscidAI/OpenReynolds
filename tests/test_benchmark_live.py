"""The live backend's graded runs, asserted from what the GPU actually wrote.

The brief for this extension asks that the M1 to M3 validations "pass at the stated
tolerances and live in the test suite". Half of that is easy and half of it is not.

**Why they cannot simply run here.** The cylinder sweep is three resolutions times
170,000 to 310,000 lattice steps; the Re 1000 cavity spins up over `cells^2/nu_lattice`
steps, about 14 million at 192 cells; the sphere at Re 200 is 63 million nodes. On an
H100 those are seconds to minutes. On Warp's single-threaded CPU backend -- which is what
CI has, and what the machine this was written on has -- they are hours each, and this
repository's whole suite is budgeted at about 24 minutes.

**So they live here the way this repository's other expensive validations live here.**
`tests/data/moving_mesh/` holds a finished OpenFOAM run, trimmed to what its grader
reads, and `tests/test_benchmark_moving_mesh.py` asserts the numbers from it;
`tests/data/felix/` does the same for the Felix solver. `tests/data/live/` is the same
device: each recorded run's `result.json` exactly as `benchmarks/live/modal_run.py`
wrote it, and this file asserts that every graded check in it passed, at the tolerance
the check itself states, on a device it names.

What that buys and what it does not. It buys a regression test on the *recorded* result:
a grader whose tolerance is quietly loosened, a check that stops being graded, a case
that disappears, or a number that drifts from the one `benchmarks/README.md` quotes will
all fail here. It does not re-derive the physics -- only a GPU can do that, and
`benchmarks/live/modal_run.py` is how. `PROVENANCE.md` beside the data says how to
regenerate it, and the small CPU-runnable cases at the bottom of this file are the part
that does re-derive, every run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "live"
INDEX = ROOT / "benchmarks" / "README.md"


def recorded() -> list[tuple[str, dict]]:
    """Every recorded run, by the label its directory carries."""
    if not DATA.is_dir():
        return []
    found = []
    for path in sorted(DATA.glob("*/result.json")):
        found.append((path.parent.name,
                      json.loads(path.read_text(encoding="utf-8"))))
    return found


def labels() -> list[str]:
    return [name for name, _ in recorded()]


def test_there_are_recorded_runs_at_all():
    """A directory that quietly empties turns every parametrised test below into zero
    tests, which passes. `tests/test_benchmarks_index.py` guards its own globs the same
    way and for the same reason."""
    assert DATA.is_dir(), f"{DATA} is missing: the recorded GPU runs are the data here"
    assert len(labels()) >= 10, (
        f"only {labels()} are recorded; M1 to M3 are twelve runs and the suite cannot "
        f"assert what is not there"
    )
    assert (DATA / "PROVENANCE.md").is_file(), (
        "recorded output with no provenance is a number nobody can regenerate"
    )


@pytest.mark.parametrize("label", labels())
def test_every_graded_check_in_a_recorded_run_passed(label):
    """The validations themselves. One test a run, failing with the check that failed."""
    payload = dict(recorded())[label]
    assert not payload.get("error"), (
        f"{label} did not complete: {str(payload.get('error'))[:400]}"
    )
    graded = [c for c in payload["checks"] if c.get("passed") is not None]
    assert graded, f"{label} recorded no graded check at all, so it asserts nothing"
    failed = [c for c in graded if not c["passed"]]
    assert not failed, (
        f"{label} has {len(failed)} failing graded check(s): "
        + "; ".join(f"{c['name']} measured {c['measured']} against {c['target']}"
                    for c in failed)
    )


@pytest.mark.parametrize("label", labels())
def test_every_recorded_run_names_the_device_that_produced_it(label):
    """A rate without a device is not a measurement, and an agreement without one cannot
    be reproduced. Every row in `PROGRESS_LIVE.md` carries this and so must the data."""
    payload = dict(recorded())[label]
    name = str(payload.get("gpu_name") or "")
    device = str(payload.get("device") or "")
    assert name and name != "no nvidia-smi", (
        f"{label} does not say what hardware produced it"
    )
    assert device, f"{label} does not say which Warp device it ran on"


# -- the numbers in the index are the numbers in the data ------------------------------
#
# The rule `tests/test_benchmarks_index.py` exists for, applied to this family. What the
# index carries is measured agreements, and an index that drifts from the data becomes a
# second, quieter opinion about what the run found.


QUOTED = {
    # label -> the figures benchmarks/README.md quotes from it, as they appear there
    "translating-sphere": ["1.0315"],
    "falling-sphere": ["108.06", "-3.80%"],
    "momentum": ["1.9e-5", "9.8e-4"],
    "underwater": ["1.1035", "1.0917"],
    "ground-effect": ["0.8975", "0.9906", "1.0472", "1.0562", "1.0308"],
}
"""Which measured figures the index quotes for each run.

Deliberately a literal map and not derived: what the index chooses to quote is an
editorial decision, and a derived list would quietly stop checking the number that was
dropped from the prose. Adding a row here is how a new quoted figure gets guarded.
"""


@pytest.mark.parametrize("label", sorted(QUOTED))
def test_the_index_quotes_the_numbers_the_recorded_run_measured(label):
    index = INDEX.read_text(encoding="utf-8")
    runs = dict(recorded())
    if label not in runs:
        pytest.skip(f"{label} is not recorded yet")
    for figure in QUOTED[label]:
        assert figure in index, (
            f"benchmarks/README.md does not quote {figure} for {label}, which the "
            f"recorded run measured. Either the index is stale or this guard is"
        )


def test_the_index_names_the_live_family_and_its_rows_are_graded():
    """`live/` has rows and every one of them states an agreement. The shared guard in
    `tests/test_benchmarks_index.py` checks that for the whole index; this checks that
    the live rows in particular did not arrive as placeholders."""
    index = INDEX.read_text(encoding="utf-8")
    rows = [line for line in index.splitlines()
            if line.startswith("| [") and "](live/)" in line]
    assert len(rows) >= 7, f"only {len(rows)} live rows in the index"
    for row in rows:
        measured = row.split("|")[4]
        assert re.search(r"\d", measured), f"this live row states no agreement: {row[:120]}"


# -- what does re-derive, every run ----------------------------------------------------


def _warp_is_here() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("warp") is not None
    except (ImportError, ValueError):
        return False


needs_warp = pytest.mark.skipif(not _warp_is_here(),
                                reason="the [live] extra is not installed here")


@needs_warp
def test_the_grader_runs_and_grades_the_cheapest_case(tmp_path):
    """`benchmarks/live/grade.py smoke` end to end, exit status and all.

    The part of the family that is re-derived rather than read. It exercises the grader's
    own argument handling, its refusal paths and its exit codes, which the recorded data
    above cannot: a grader that could not run at all would still let every assertion in
    this file pass.
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "live" / "grade.py"), "smoke",
         "--device", "cpu", "--quiet", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert done.returncode == 0, (
        f"grade.py smoke exited {done.returncode}\n{done.stdout[-1500:]}\n"
        f"{done.stderr[-1500:]}"
    )
    assert "graded checks passed" in done.stdout


@needs_warp
def test_the_grader_refuses_an_unknown_case_with_status_two():
    """Exit 2 is "the case could not be run", which is the contract
    `benchmarks/felix/grade.py` and `benchmarks/moving_mesh/grade.py` both follow. A
    grader that returned 1 for a typo would look like a failing validation."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "live" / "grade.py"), "not_a_case",
         "--quiet"],
        capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert done.returncode == 2, f"exited {done.returncode}, not 2"
    assert "not_a_case" in done.stderr


@needs_warp
def test_the_grader_lists_every_case_and_every_reference():
    import subprocess
    import sys

    from openreynolds.live import cases

    done = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "live" / "grade.py"), "list"],
        capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert done.returncode == 0
    for name in cases.CASES:
        assert name in done.stdout, f"`grade.py list` does not name {name}"
    for name in cases.REFERENCES:
        assert name in done.stdout, f"`grade.py list` does not name the {name} reference"
