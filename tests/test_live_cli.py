"""`openreynolds live`: the group is there, its help prints, and the flags arrive.

The group exists so that the live backend is reachable without a session, a key or a
workspace -- `openreynolds live crosscheck` in CI is the case that decided its shape --
and so every subcommand answers `--json`, like every other command here that a script
would read.

## Why `openreynolds.live.cases` is a stub in most of this file

`cases.py` is written separately and its contract is fixed: `CASES`,
`describe_cases()`, `run_case()`, `run_case_file()`, `benchmark()`, `crosscheck()` and
`report(result)`, with every run returning a dict carrying at least `case`, `passed`,
`checks`, `report` and `figures`. What this file tests is the half the CLI owns -- that
the flags the person typed become those calls, that `--json` prints the dict and
nothing else, and that the exit code is the verdict -- and it tests that against the
contract rather than against a real run.

That is not only expedience. A real cavity, even a small one, is seconds to minutes of
lattice stepping on a CPU, and a suite that ran one would either be slow in CI or
quietly stop running it. The physics is checked in `tests/test_live_*` and
`benchmarks/live/`, on grids chosen for it; this is the wiring, and the wiring is worth
testing on a machine with no GPU at all.

`require` is stubbed with it, so this file passes with or without the extra installed.
"""

from __future__ import annotations

import json
import types

import pytest
from click.testing import CliRunner

from openreynolds import cli

RESULT = {
    "case": "cavity",
    "passed": True,
    "checks": [{"name": "u at y=0.5", "measured": -0.2057, "target": -0.2058,
                "passed": True}],
    "report": ["cavity  Re 100  128^3  trt", "  u at y=0.5  -0.2057 vs -0.2058  ok"],
    "figures": ["figs/cavity.png"],
}
"""One result in the shape the contract promises. The numbers are Ghia's shape, not
Ghia's numbers: nothing here is a measurement, and `benchmarks/live/` is where a number
has to be real."""


class Case:
    def __init__(self, about: str):
        self.about = about


@pytest.fixture
def cases(monkeypatch):
    """A stand-in for `openreynolds.live.cases` that records how it was called."""
    from openreynolds import live

    stub = types.ModuleType("openreynolds.live.cases")
    stub.calls = []
    stub.result = dict(RESULT)
    stub.CASES = {"cavity": Case("lid-driven cavity at Re 100, against Ghia"),
                  "cylinder": Case("fixed cylinder at Re 100, against the OpenFOAM run")}
    stub.describe_cases = lambda: [f"{name}  {case.about}"
                                   for name, case in stub.CASES.items()]

    def record(what):
        def call(*args, **kwargs):
            stub.calls.append((what, args, kwargs))
            return stub.result
        return call

    for name in ("run_case", "run_case_file", "benchmark", "crosscheck"):
        setattr(stub, name, record(name))
    stub.report = lambda result: "\n".join(result["report"])
    monkeypatch.setattr(live, "cases", stub, raising=False)
    monkeypatch.setattr(live, "require", lambda: None)
    return stub


def invoke(*args):
    return CliRunner().invoke(cli.main, ["live", *args])


# -- the group is registered and every page prints -----------------------------


def test_the_group_is_on_the_main_command():
    assert "live" in cli.main.commands
    assert set(cli.main.commands["live"].commands) == {
        "cases", "run", "bench", "crosscheck"}


def test_the_group_help_says_where_it_runs_and_what_installs_it():
    """Read by somebody who has just seen `live` in `openreynolds --help` and does not
    know that this one command does not use the workspace."""
    result = invoke("--help")
    assert result.exit_code == 0, result.output
    assert "this machine" in result.output
    assert "openreynolds[live]" in result.output


@pytest.mark.parametrize("command", ["cases", "run", "bench", "crosscheck"])
def test_every_subcommand_has_a_help_page_and_a_json_flag(command):
    """`--help` has to work without the extra and without `cases.py`, which is the
    whole reason every import in the group's bodies is deferred: click builds the
    page from the decorators alone."""
    result = invoke(command, "--help")
    assert result.exit_code == 0, result.output
    assert "--json" in result.output


def test_the_choices_are_the_backends_own():
    """The flag choices are written out in `cli.py` instead of imported, so that
    building the page does not pull Warp. Copied constants with no check between them
    is how two files come to hold two different opinions."""
    from openreynolds.live import units

    assert set(cli.LIVE_FLUIDS) == set(units.FLUIDS)
    pytest.importorskip("warp")
    from openreynolds.live import lattice

    assert set(cli.LIVE_COLLISIONS) == set(lattice.SCHEMES)


# -- cases ---------------------------------------------------------------------


def test_cases_lists_one_line_each(cases):
    result = invoke("cases")
    assert result.exit_code == 0, result.output
    assert "cavity" in result.output and "Ghia" in result.output


def test_cases_json_names_each_case_and_what_it_is(cases):
    result = invoke("cases", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [row["name"] for row in payload["cases"]] == ["cavity", "cylinder"]
    assert payload["cases"][0]["about"] == cases.CASES["cavity"].about


# -- run -----------------------------------------------------------------------


def test_run_passes_only_the_flags_that_were_given(cases):
    """A case's defaults are part of what it validates against -- the resolution
    Ghia's cavity needs to agree with Ghia is not a taste -- so an unset flag must
    leave the case's own number alone rather than substitute one of the CLI's."""
    assert invoke("run", "cavity").exit_code == 0
    what, args, kwargs = cases.calls[-1]
    assert what == "run_case" and args == ("cavity",)
    # `out_dir`, which is the keyword `cases.run_case` and every runner actually
    # takes. It was `out` here and in `cli.py`, and `out` is not a parameter of any of
    # them: it travelled through `**overrides` into the runner and was swallowed by its
    # trailing `**_`, so `--out` was accepted, ignored, and every figure landed in the
    # working directory. This assertion is the reason to spell the keyword out rather
    # than check that *some* path was passed.
    assert kwargs == {"out_dir": "."}


def test_run_passes_every_flag_that_was(cases):
    result = invoke("run", "cavity", "--resolution", "64", "--steps", "2000",
                    "--collision", "regularized", "--fluid", "air",
                    "--out", "figs", "--device", "cuda:0")
    assert result.exit_code == 0, result.output
    what, args, kwargs = cases.calls[-1]
    assert what == "run_case" and args == ("cavity",)
    assert kwargs == {"out_dir": "figs", "resolution": 64, "steps": 2000,
                      "collision": "regularized", "fluid": "air", "device": "cuda:0"}


@pytest.mark.parametrize("path", ["drone.yaml", "drone.yml", "cases/Drone.YAML"])
def test_a_case_file_goes_to_the_file_runner(cases, path):
    """The same argument carries both, because to the person typing it there is one
    thing being named: the case to run."""
    assert invoke("run", path).exit_code == 0
    what, args, _ = cases.calls[-1]
    assert what == "run_case_file" and args == (path,)


def test_run_prints_the_report(cases):
    result = invoke("run", "cavity")
    assert "u at y=0.5" in result.output


def test_run_json_prints_the_result_and_nothing_else(cases):
    result = invoke("run", "cavity", "--json")
    assert json.loads(result.stdout) == RESULT


def test_the_group_output_flag_is_the_same_request_as_live_s_own_json(cases):
    """`--output-format` is a group option, so an agent puts it in front of
    everything; `live` is in `JSON_CAPABLE_COMMANDS` because every command under it
    answers in JSON."""
    result = CliRunner().invoke(
        cli.main, ["--output-format", "stream-json", "live", "run", "cavity"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["case"] == "cavity"


# -- bench and crosscheck ------------------------------------------------------


def test_bench_takes_two_resolutions_in_one_go(cases):
    """Repeatable, because the scaling of a rate measured at one resolution is an
    extrapolation and at two it is a measurement."""
    result = invoke("bench", "--resolution", "64", "--resolution", "128")
    assert result.exit_code == 0, result.output
    what, _, kwargs = cases.calls[-1]
    assert what == "benchmark" and kwargs == {"resolutions": [64, 128]}


def test_bench_with_nothing_asked_for_asks_for_nothing(cases):
    assert invoke("bench").exit_code == 0
    assert cases.calls[-1] == ("benchmark", (), {})


def test_crosscheck_needs_no_options(cases):
    assert invoke("crosscheck").exit_code == 0
    what, _, kwargs = cases.calls[-1]
    assert what == "crosscheck" and kwargs == {"out_dir": "."}


# -- the verdict is the exit code ----------------------------------------------


@pytest.mark.parametrize("command", [("run", "cavity"), ("bench",), ("crosscheck",)])
def test_a_failed_check_exits_one(cases, command):
    """So that `openreynolds live crosscheck` in CI needs to parse nothing. Said once
    per subcommand because an exit code nobody sets is an exit code of 0."""
    cases.result = {**RESULT, "passed": False}
    assert invoke(*command).exit_code == 1


@pytest.mark.parametrize("command", [("run", "cavity"), ("bench",), ("crosscheck",)])
def test_a_failed_check_still_prints_what_it_measured(cases, command):
    """A non-zero exit with nothing said is the one unhelpful failure: the point of a
    validation is the number it got."""
    cases.result = {**RESULT, "passed": False}
    result = invoke(*command)
    assert "u at y=0.5" in result.output
    assert json.loads(invoke(*command, "--json").stdout)["passed"] is False
