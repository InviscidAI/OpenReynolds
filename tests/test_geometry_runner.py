"""The child interpreter (DESIGN.md 3.10): `cli.py build --script` run by path under
`-I` with `child_env`, and what comes back as a RunOutcome -- the exit codes 0/2/3/4/5,
the guard that refuses every import but math and json from the script's own frame and
nothing else, the trimmed traceback, the timeout.

Every test here goes through the real subprocess. The ones that need a script to build
(rc 0 on T05, two Sketches, the Row refusal with a partial) need U1's `Sketch`; until it
lands they skip, and their in-process twins over a canned Plan live in
`test_geometry_cli.py`. The draw test uses `build --spec` through the same command shape
and the same `child_env`, because that is what proves the child can draw at all on this
platform (finding 27: three keys alone cannot on Windows).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from openreynolds.geometry import runner

from _geometry_fakes import T01_CLAIMS, T01_LAP1_SCRIPT, T01_LAP2_SCRIPT, T05_SCRIPT, T05_SPEC, landed

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "openreynolds" / "geometry" / "cli.py"

needs_u1 = pytest.mark.skipif(not landed("U1"), reason="needs U1's Sketch (the script must construct one)")
needs_u3 = pytest.mark.skipif(not landed("U3"), reason="needs U3's claims judge (the table decides `ready`)")


def run(script: str, work: Path, **kw) -> runner.RunOutcome:
    return runner.run_script(script, work, kw.pop("claims_path", None), kw.pop("reference", None), **kw)


# -- the guard ----------------------------------------------------------------------------


def test_import_os_is_refused_with_rc_5(tmp_path):
    out = run("import os\ns = Sketch(units='mm')\n", tmp_path)
    assert out.rc == 5 and out.png is None
    assert out.result["code"] == "E-IMPORT" and out.result["ok"] is False
    assert "line 1: `import os` -- the script may import math and json only" in out.text   # section 5's order
    assert "the API does the arithmetic" in out.text
    assert out.text.startswith("SCRIPT")


@pytest.mark.parametrize("statement", ["from openreynolds.geometry.sketch import Sketch",
                                       "from geometry.sketch import Sketch",
                                       "import openreynolds.geometry.sketch"])
def test_from_openreynolds_geometry_sketch_import_is_refused_with_rc_5(tmp_path, statement):
    out = run(statement + "\n", tmp_path)
    assert out.rc == 5 and out.result["code"] == "E-IMPORT"
    assert f"`{statement}` -- the API needs no import" in out.text
    assert "Sketch, Rect, Disk, Polygon, Band, Outline, Passage, Bypass, Row, Serpentine, BodyInBox and math are already bound" in out.text


def test_a_bare_dunder_import_is_refused_too(tmp_path):
    out = run("os = __import__('os')\n", tmp_path)
    assert out.rc == 5 and out.result["code"] == "E-IMPORT"


def test_math_and_json_are_allowed(tmp_path):
    out = run("import math, json\nprint(json.dumps({'r': math.hypot(3, 4)}))\n", tmp_path)
    # the script ran to its end: what stops it is the missing Sketch (E-NO-SKETCH once
    # U1's registry is there; the kernel's own refusal before), never the import
    assert out.result["code"] != "E-IMPORT" and out.rc in (3, 5), out.text
    assert '> {"r": 5.0}' in out.text


@needs_u1
def test_the_guard_is_gone_after_the_script(tmp_path):
    """rc 0 on the T05 script and preview.png exists: the kernel imported matplotlib
    (and gmsh) after the script ran, under the real `__import__`."""
    out = run(T05_SCRIPT, tmp_path)
    assert out.rc == 0, out.text
    assert (tmp_path / "preview.png").exists() and out.png and out.png[:4] == b"\x89PNG"


# -- the environment ------------------------------------------------------------------------


def test_the_child_draws_under_the_runner_env(tmp_path):
    """The real command shape (`python -I <cli.py> build ... --out --preview`) with
    `child_env` on the T05 record; `preview.png` exists. This is the test that fails on
    any platform where the child cannot draw."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    spec = tmp_path / "t05.json"
    spec.write_text(json.dumps(T05_SPEC), encoding="utf-8")
    argv = runner.command(tmp_path / "script.py", tmp_path, None, None)
    argv[argv.index("--script")] = "--spec"
    argv[argv.index(str(tmp_path / "script.py"))] = str(spec)
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=runner.RUN_TIMEOUT_S,
                          env=runner.child_env(tmp_path), cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "preview.png").stat().st_size > 10_000
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["preview"] == "preview.png" and result["rc"] == 0
    assert "cylinder" in result["measurements"]["patches"]


def test_child_env_is_an_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "C:/somewhere/else")
    monkeypatch.setenv("USERPROFILE", "C:/Users/someone")
    monkeypatch.setenv("SOME_SECRET", "x")
    env = runner.child_env(tmp_path)
    assert "PYTHONPATH" not in env and "SOME_SECRET" not in env
    assert env["MPLCONFIGDIR"] == str(tmp_path / ".mpl") and (tmp_path / ".mpl").is_dir()
    assert env["PYTHONUTF8"] == "1" and env["MPLBACKEND"] == "Agg" and env["PATH"] == os.environ["PATH"]
    assert env["USERPROFILE"] == "C:/Users/someone"
    assert set(env) <= {"PATH", "PYTHONUTF8", "MPLBACKEND", "MPLCONFIGDIR", *runner.PASS_THROUGH}


def test_the_child_runs_by_path_under_dash_i(tmp_path, monkeypatch):
    """The command has `-I` and the cli path; PYTHONPATH in the parent env does not reach
    the child (twice: `-I` ignores it, and the allowlist never passes it)."""
    argv = runner.command(tmp_path / "script.py", tmp_path, tmp_path / "c.json", "tesla_valve")
    assert argv[:3] == [sys.executable, "-I", str(CLI)] and argv[3] == "build"
    assert argv[argv.index("--out") + 1] == str(tmp_path)
    assert argv[argv.index("--preview") + 1] == str(tmp_path / "preview.png")
    assert argv[argv.index("--claims") + 1] == str(tmp_path / "c.json")
    assert argv[argv.index("--reference") + 1] == "tesla_valve"
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "poison"))
    (tmp_path / "poison").mkdir()
    (tmp_path / "poison" / "json.py").write_text("raise RuntimeError('PYTHONPATH reached the child')\n", encoding="utf-8")
    out = run("import json\nprint(json.dumps([1]))\n", tmp_path)
    assert "PYTHONPATH reached the child" not in out.text and "> [1]" in out.text


# -- exit codes -----------------------------------------------------------------------------


def test_a_while_true_script_is_stopped_with_rc_4_within_timeout_plus_5(tmp_path):
    t0 = time.monotonic()
    out = run("while True:\n    pass\n", tmp_path, timeout_s=3)
    assert out.rc == 4 and out.result is None and out.png is None
    assert out.text == "the script ran past 3 s and was stopped; a loop in the script is not terminating"
    assert time.monotonic() - t0 < 3 + 5


@needs_u1
def test_a_missing_sketch_is_rc_5(tmp_path):
    out = run("x = 1\n", tmp_path)
    assert out.rc == 5 and out.result["code"] == "E-NO-SKETCH"
    assert "the script made no Sketch" in out.text


@needs_u1
def test_two_sketches_is_rc_5(tmp_path):
    out = run("a = Sketch(units='mm')\nb = Sketch(units='mm')\n", tmp_path)
    assert out.rc == 5 and out.result["code"] == "E-MANY-SKETCHES"
    assert "2 Sketch objects; exactly one" in out.text


def test_a_raising_script_is_rc_3_and_the_traceback_has_no_kernel_frames(tmp_path):
    script = "count = 4\ndef pitch(n):\n    return 60 / (n - 4)\n\np = pitch(count)\n"
    out = run(script, tmp_path)
    assert out.rc == 3 and out.result["ok"] is False and out.result["code"] == "E-SCRIPT"
    assert out.result["record"] is None and out.result["measurements"] is None and out.png is None
    assert "!! ERROR  E-SCRIPT  line 3: ZeroDivisionError: division by zero" in out.text
    assert "> return 60 / (n - 4)" in out.text
    assert "(the API does the arithmetic: footprints, landings and pitches are in the print-back)" in out.text
    assert "called from line 5" in out.text
    tb = out.result["traceback"]
    assert "cli.py" not in tb and "openreynolds" not in tb and "site-packages" not in tb
    assert "line 3" in tb and "line 5" in tb and tb.endswith("ZeroDivisionError: division by zero")
    # the one rule for rc 3: SCRIPT, then the refusal, nothing else
    assert out.text.splitlines()[0].startswith("SCRIPT     ran in") and "LINT" not in out.text


def test_a_wrong_kwarg_quotes_the_line_and_names_the_right_kwarg(tmp_path):
    script = ("s = 1\n"
              "loop = Bypass(wall=None, width=3, leave_angle=20, radius=6, return_angle=80)\n")
    out = run(script, tmp_path)
    assert out.rc == 3 and out.result["code"] == "E-SCRIPT"
    assert "line 2: Bypass() got an unexpected keyword argument 'radius'" in out.text
    assert "> loop = Bypass(wall=None, width=3, leave_angle=20, radius=6, return_angle=80)" in out.text
    assert "Bypass takes outer_radius (of the outer wall); see the reference card" in out.text


def test_a_syntax_error_is_rc_3_with_the_line(tmp_path):
    out = run("s = Sketch(units='mm'\nx = 1\n", tmp_path)
    assert out.rc == 3 and out.result["code"] == "E-SCRIPT"
    assert "SyntaxError" in out.text and "line 1" in out.text


def test_a_script_that_prints_is_shown_its_prints(tmp_path):
    out = run("print('hello', 1.5)\nprint('again')\nraise ValueError('stop')\n", tmp_path)
    assert out.rc == 3
    assert "SCRIPT     ran in" in out.text and "2 prints" in out.text
    assert "           > hello 1.5" in out.text and "           > again" in out.text


def test_a_lap_never_reads_the_previous_laps_files(tmp_path):
    (tmp_path / "result.json").write_text('{"rc": 0, "report": "stale"}', encoding="utf-8")
    (tmp_path / "preview.png").write_bytes(b"\x89PNG stale")
    out = run("while True:\n    pass\n", tmp_path, timeout_s=2)
    assert out.rc == 4 and out.result is None and out.png is None


def test_a_child_that_dies_without_a_result_is_rc_3_whatever_its_exit_code(tmp_path, monkeypatch):
    """Every path the cli owns writes result.json; a child that left none died in the
    kernel. Its exit code is not a lap code: an exit 2 must not read as 'lint errors'."""
    monkeypatch.setattr(runner, "command", lambda *a, **k: [sys.executable, "-I", "-c", "import sys; sys.exit(2)"])
    out = run("s = 1\n", tmp_path)
    assert out.rc == 3 and out.result is None and out.png is None
    assert out.text.startswith("the child interpreter exited 2 without a result.json; nothing in the script to fix")


def test_a_print_flood_is_clipped_in_the_print_back(tmp_path):
    """The print-back reaches the model line for line: a loop's log is cut after
    MAX_PRINT_LINES with one line saying how much was left out."""
    from openreynolds.geometry import cli
    out = run("for i in range(5000):\n    print('line', i)\n", tmp_path)
    echoed = [ln for ln in out.text.splitlines() if ln.startswith("           > ")]
    assert len(echoed) == cli.MAX_PRINT_LINES + 1, len(echoed)
    assert echoed[0] == "           > line 0" and echoed[-2] == f"           > line {cli.MAX_PRINT_LINES - 1}"
    assert echoed[-1] == f"           > ... {5000 - cli.MAX_PRINT_LINES} more lines not shown (the print-back is for numbers, not logs)"
    assert len(out.text) < 20_000
    # and a short print is shown whole
    out = run("print('one')\nprint('two')\n", tmp_path)
    assert "           > one\n           > two" in out.text and "not shown" not in out.text


def test_run_outcome_reads_the_result_back(tmp_path):
    out = run("raise RuntimeError('x')\n", tmp_path)
    assert out.result == json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert out.text == (tmp_path / "report.txt").read_text(encoding="utf-8") == out.result["report"]
    assert out.seconds > 0 and not out.ready and out.failing_claims == []
    assert out.result["verdict"] == {"text": "not ready: E-SCRIPT", "ready": False}
    assert (tmp_path / "script.py").read_text(encoding="utf-8") == "raise RuntimeError('x')\n"


# -- the worked T01 through the real child (after U1 / U3) ------------------------------------


@needs_u1
@needs_u3
def test_t01_lap2_is_rc_0_and_ready(tmp_path):
    pytest.importorskip("gmsh")
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(T01_CLAIMS), encoding="utf-8")
    out = run(T01_LAP2_SCRIPT, tmp_path / "lap2", claims_path=claims_path)
    assert out.rc == 0, out.text
    assert out.ready and out.lint_clean and out.failing_claims == []
    assert out.result["compliance"] is not None and out.result["verdict"]["ready"] is True
    assert out.png and (tmp_path / "lap2" / "preview.png").stat().st_size > 10_000
    assert "VERDICT    ready to COMMIT" in out.text
    assert out.result["record"]["features"]["loops"]["kind"] == "Row"


@needs_u1
def test_a_row_refusal_is_rc_5_with_a_picture_and_a_partial_record(tmp_path):
    pytest.importorskip("gmsh")
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(T01_CLAIMS), encoding="utf-8")
    out = run(T01_LAP1_SCRIPT, tmp_path / "lap1", claims_path=claims_path)
    assert out.rc == 5, out.text
    assert out.png is not None and out.result["code"] == "E-ROW-FIT"
    assert "loop" in out.result["record"]["features"]
    assert out.result["compliance"] is None and out.result["measurements"] is not None
    assert "!! ERROR  E-ROW-FIT  Row 'loops': 4 x Bypass 'loop' do not fit on main.top (60 long)" in out.text
    assert "CLAIMS     not evaluated: the sketch did not build" in out.text
    assert "VERDICT    not ready: LINT E-ROW-FIT" in out.text
