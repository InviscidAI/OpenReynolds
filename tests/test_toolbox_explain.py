"""`explain.py`: the caveat has to arrive before the number, not after it.

Everything this script prints was produced by another file, so what is worth testing
is not the readings -- they have their own tests -- but the three things only this
file decides: which reading costs the numbers what, what order they come out in, and
whether the verdict is ever stronger than the evidence.

The last of those is the one that matters. An "explain this result" feature that says
a diverging run's coefficients are fine is worse than no feature, because a person who
asked what the numbers mean has been told. So the tests below are mostly about the
verdict being refused.

    python3 -m pytest tests/test_toolbox_explain.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def explain():
    spec = importlib.util.spec_from_file_location("toolbox_explain",
                                                  TOOLBOX / "explain.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- a case, written by hand so the right verdict is known before it is asked ----


HEADER = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
          "    object controlDict;\n}\n\napplication     simpleFoam;\n"
          "startTime       0;\nendTime         100;\ndeltaT          1;\n")


def case_with(tmp_path: Path, name: str, residuals: list[tuple[int, float, float]],
              ended: bool = True, times: tuple[str, ...] = ("0", "100")) -> Path:
    """A case directory with a solver log whose residual series is dictated here.

    The log is in the shape `log_digest` parses -- `Time = n`, then one `Solving for`
    line per field -- because the reading under test is the one `log_digest` makes of
    a real log, and a stubbed digest would test this file against its own idea of one.
    """
    case = tmp_path / name
    for time in times:
        (case / time).mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(HEADER, encoding="utf-8")
    lines = ["/*---- OpenFOAM ----*/", "Create mesh for time = 0", ""]
    for step, p_res, u_res in residuals:
        lines.append(f"Time = {step}")
        lines.append(f"smoothSolver:  Solving for Ux, Initial residual = {u_res:g}, "
                     "Final residual = 1e-09, No Iterations 3")
        lines.append(f"GAMG:  Solving for p, Initial residual = {p_res:g}, "
                     "Final residual = 1e-08, No Iterations 5")
        lines.append("ExecutionTime = 1 s  ClockTime = 1 s")
        lines.append("")
    if ended:
        lines.append("End")
    (case / "log.simpleFoam").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return case


def falling(count: int = 60) -> list[tuple[int, float, float]]:
    return [(i + 1, 1e-2 * 0.85 ** i, 1e-3 * 0.85 ** i) for i in range(count)]


def climbing(count: int = 60) -> list[tuple[int, float, float]]:
    return [(i + 1, 1e-6 * 1.4 ** i, 1e-7 * 1.4 ** i) for i in range(count)]


def test_a_diverging_run_is_told_not_to_be_quoted(explain, tmp_path):
    """The reading that has to survive every later change to this file. A run whose
    residuals are climbing has produced numbers and they are not an answer, and a
    reader who is shown the coefficients without being shown that has been misled by
    a tool built to prevent exactly that."""
    case = case_with(tmp_path, "diverged", climbing())
    found = explain.explain(case)
    assert found["verdict"] == explain.STOP
    assert found["says"] == "do not quote these numbers"
    said = " ".join(row["said"] for row in found["readings"]
                    if row["level"] == explain.STOP)
    assert "diverging" in said


def test_a_case_that_wrote_no_time_but_zero_is_refused_by_name(explain, tmp_path):
    """Every field in it is the initial condition. It draws, plots and reports
    perfectly well, which is the whole problem: `showcase.open_case` refuses this for
    a decomposed case and nothing refused it for a reconstructed one. Six of the 34
    real case directories under `studies/` are in exactly this state."""
    case = case_with(tmp_path, "initial", falling(), times=("0",))
    found = explain.explain(case)
    assert found["verdict"] == explain.STOP
    said = " ".join(row["said"] for row in found["readings"])
    assert "initial condition" in said


def test_a_run_still_falling_is_a_reading_rather_than_an_answer(explain, tmp_path):
    """Not a failure and not a result: more iterations would tighten it, and nothing
    about it says it is wrong. That is `log_digest`'s own distinction and it has to
    survive the trip through here as a caveat and not as a stop."""
    case = case_with(tmp_path, "falling", falling())
    found = explain.explain(case)
    assert found["verdict"] == explain.CAVEAT
    assert found["says"] == "a reading, not an answer"


def test_how_the_run_stopped_is_not_marked_with_how_its_residuals_were_doing(
        explain, tmp_path):
    """Two facts, two costs. A run that reached the end of its controlDict with
    climbing residuals should not have "ran to the end" marked as the alarming half;
    marking every line at the worst of them is what makes a reader stop reading the
    markers at all."""
    case = case_with(tmp_path, "both", climbing())
    found = explain.explain(case)
    ended = [row for row in found["readings"] if row["said"].startswith("ended:")]
    assert ended and ended[0]["level"] == explain.CLEAR
    climb = [row for row in found["readings"] if row["said"].startswith("residuals:")]
    assert climb and climb[0]["level"] == explain.STOP


def test_the_verdict_is_never_the_word_converged(explain, tmp_path):
    """None of the three verdicts claims convergence or validation, because this file
    reads four of the case's own files and not one of them is a benchmark. The
    strongest thing true here is that nothing it read contradicts the numbers."""
    for word in ("converged", "validated", "correct", "verified"):
        assert not any(word in said.lower() for said in explain.VERDICTS.values())
    assert explain.VERDICTS[explain.CLEAR] == "nothing here contradicts them"


def test_a_reading_that_could_not_be_made_is_printed_as_one(explain, tmp_path):
    """The gap where a mesh grade should be reads as a mesh that was checked and found
    fine. `claims.py` and `report.py` have each paid for that once: #192 took the
    `verified` column out of the report precisely because a column that always reads
    zero reads as nothing having checked out."""
    case = case_with(tmp_path, "nochecks", falling())
    found = explain.explain(case)
    said = " ".join(row["said"] for row in found["readings"])
    assert "ungraded" in said or "not graded" in said
    text = explain.report(found)
    assert "not read:" in text


def test_a_case_with_no_solver_log_says_so_rather_than_assuming(explain, tmp_path):
    case = tmp_path / "bare"
    (case / "0").mkdir(parents=True)
    (case / "1").mkdir(parents=True)
    found = explain.explain(case)
    said = " ".join(row["said"] for row in found["readings"])
    assert "no solver log" in said
    assert found["verdict"] == explain.CAVEAT


def test_the_report_puts_the_caveats_above_the_numbers(explain, tmp_path):
    """The order is the feature. A document that leads with Cd and mentions on page
    two that the run was still falling has already told the reader the number."""
    case = case_with(tmp_path, "ordered", falling())
    text = explain.report(explain.explain(case))
    assert text.index("read this first") < text.index("so: ")


def test_it_exits_zero_whatever_the_verdict(explain, tmp_path, capsys):
    """This reports; it does not gate. `preflight.py` exits non-zero because it is a
    check somebody asked to be stopped by, and a reading of a finished run is not one:
    a non-zero exit would make "explain the result" look like a failed command in
    every log that ran it."""
    case = case_with(tmp_path, "exits", climbing())
    assert explain.main([str(case)]) == 0
    assert "do not quote these numbers" in capsys.readouterr().out


def test_a_directory_that_is_not_there_is_refused_rather_than_explained(explain,
                                                                       tmp_path):
    with pytest.raises(SystemExit):
        explain.main([str(tmp_path / "nothing")])


def test_nothing_in_here_measures_anything_of_its_own(explain):
    """The property that makes this safe to add. Every number it prints came from an
    instrument that already produced it, so there is no second opinion to reconcile --
    which is the shape of F-36, where two scripts measured one quantity and nothing
    put the two side by side."""
    import ast

    tree = ast.parse((TOOLBOX / "explain.py").read_text(encoding="utf-8"))
    imported = {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    assert "numpy" not in imported, (
        "explain.py has started doing arithmetic of its own; the instruments it "
        "collects from are where a number is allowed to be computed"
    )
