"""`benchmarks/tutorials/run_tier1.py`: the case table, and that it names real things.

The runner exists so eight tier 1 cases can be checked with one command instead of eight
copied out of a table. Its own failure mode is the table: a case whose script was renamed,
a flag that no longer exists, or a `graded` marker that disagrees with what the case
actually reports. None of that shows up until somebody runs it, and the point of the
runner is that somebody does not have to.

So these tests check the table rather than the runs. They do not solve anything: the
cases themselves take about fourteen minutes together and are graded by their own suites
(`tests/test_benchmark_mhd.py`, `tests/test_toolbox_acoustics_tone.py`,
`tests/test_toolbox_wind_formula.py` and the `benchmarks` job). What is checked here is
that the runner would reach them.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "tutorials" / "run_tier1.py"
INDEX = ROOT / "benchmarks" / "tutorials" / "README.md"


def runner():
    """The runner as a module, without running it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_run_tier1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_runner_exists_and_the_index_offers_it():
    """A runner nobody is told about is a file. The index is where a reader looks."""
    assert RUNNER.is_file()
    text = INDEX.read_text(encoding="utf-8")
    assert "run_tier1.py" in text, "the tutorial index does not mention the runner"
    assert "--quick" in text, "the index should say the slow ones can be skipped"


def test_every_case_names_a_script_that_exists():
    """The failure this is for: a script renamed and the table left behind. The runner
    would report `NOT AVAILABLE` or a traceback per case, and only when run."""
    module = runner()
    assert len(module.CASES) >= 6, f"only {len(module.CASES)} cases"
    for name, cmd, _against, _slow, _graded in module.CASES:
        script = ROOT / cmd[0]
        assert script.is_file(), f"{name}: {cmd[0]} does not exist"
        assert script.suffix == ".py", f"{name}: {cmd[0]} is not a script"


def test_every_flag_the_table_passes_is_one_the_script_accepts():
    """`--tone`, `--check` and `--naca` are the table's own flags. A flag that has been
    renamed makes a case silently NOT AVAILABLE, which reads as the case being absent
    rather than the table being stale."""
    module = runner()
    for name, cmd, _against, _slow, _graded in module.CASES:
        flags = [part for part in cmd[1:] if part.startswith("--")]
        if not flags:
            continue
        text = (ROOT / cmd[0]).read_text(encoding="utf-8")
        for flag in flags:
            assert f'"{flag}"' in text or f"'{flag}'" in text, (
                f"{name}: {cmd[0]} does not define {flag}")


def test_the_graded_marker_is_a_property_of_the_case_not_of_its_output():
    """Exactly one case is ungraded, and it is the dam break, whose reference could not
    be read. If a second appears, either a grader was lost or the marker is being used
    to paper over a failure."""
    module = runner()
    ungraded = [name for name, _c, _a, _s, graded in module.CASES if not graded]
    assert ungraded == ["dam break, SPH"], (
        f"ungraded cases are {ungraded}; only the dam break should be, and "
        "benchmarks/README.md says why")


def test_the_slow_cases_are_the_ones_quick_skips_and_they_are_named():
    """--quick has to say what it left out, or a green run means less than it looks."""
    module = runner()
    slow = [name for name, _c, _a, is_slow, _g in module.CASES if is_slow]
    assert len(slow) == 3, f"slow cases are {slow}"
    done = subprocess.run([sys.executable, "-X", "utf8", str(RUNNER), "--list", "--quick"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    for name in slow:
        assert name in done.stdout, f"--list --quick does not mention {name}"
    assert "skipped by --quick" in done.stdout


def test_list_runs_nothing_and_says_what_it_would():
    """`--list` is the cheap way to see the table, so it must not solve anything."""
    module = runner()
    done = subprocess.run([sys.executable, "-X", "utf8", str(RUNNER), "--list"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0
    for name, cmd, _a, _s, _g in module.CASES:
        assert name in done.stdout, f"--list omits {name}"
        assert " ".join(cmd) in done.stdout, f"--list omits the command for {name}"
    # Nothing graded, because nothing ran.
    assert "PASS" not in done.stdout and "agrees" not in done.stdout


def test_every_case_says_what_it_is_measured_against():
    """A case with no reference is a demonstration. The index's whole rule is that a row
    states one, and the runner prints it beside what the case reported."""
    module = runner()
    for name, _cmd, against, _slow, _graded in module.CASES:
        assert against and len(against) > 20, f"{name}: against is {against!r}"
        assert not against.lower().startswith("todo"), name


def test_a_missing_flag_is_reported_as_absent_rather_than_as_no_verdict():
    """The distinction that cost a wrong reading once: a flag this checkout does not have
    is NOT the same as a case that reports no verdict, and conflating them reads as the
    case being ungraded."""
    module = runner()
    found = module.run_one(("openreynolds/toolbox/panel.py", "--definitely-not-a-flag"), 120.0)
    assert found["verdict"] == "NOT AVAILABLE", found
    assert "--definitely-not-a-flag" in found["headline"]


@pytest.mark.parametrize("case", ["lbm/", "sph/"])
def test_the_graders_the_table_calls_are_the_ones_the_index_lists(case):
    """The runner and the index have to agree about which families are tier 1."""
    module = runner()
    called = {cmd[0] for _n, cmd, _a, _s, _g in module.CASES}
    assert any(case in path for path in called), f"no tier 1 case calls {case}"
    assert case in INDEX.read_text(encoding="utf-8")
