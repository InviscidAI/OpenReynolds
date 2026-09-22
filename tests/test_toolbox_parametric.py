"""`parametric.py`: one candidate of a parametric study, without an OpenFOAM.

The build step is real -- `python3 build.py` runs in the candidate directory, and the
fake `build.py` here writes a `constant/polyMesh` the way the real one would -- while
`mesh_look.py`'s checkMesh, `case_gen.py` and the solver are stood in for by functions
that leave the files those steps leave. What is pinned is the contract: what is refused
before anything runs, what the `case_gen.py` command line is, where the trim's cases
go and which one is scored, and that `metrics.json` comes out of `score.py`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def parametric():
    spec = importlib.util.spec_from_file_location("toolbox_parametric", TOOLBOX / "parametric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the round and the candidate -------------------------------------------------------

BUILD_PY = """\
from design_constants import *
from pathlib import Path

# A stand-in for gmsh -> gmshToFoam: leaves what a mesh leaves, sized off a constant.
poly = Path("constant") / "polyMesh"
poly.mkdir(parents=True, exist_ok=True)
(poly / "points").write_text("0()\\n")
cells = int(1000 * THICKNESS * 100)
(poly / "owner").write_text(
    'FoamFile\\n{\\n    note        "nPoints:1  nCells:%d  nFaces:1  nInternalFaces:0";\\n}\\n' % cells)
Path("built.txt").write_text(f"ALPHA_DEG={ALPHA_DEG} THICKNESS={THICKNESS}\\n")
"""

HOOKS_PY = '''\
"""The round's own reading of its patches."""
from __future__ import annotations


def case_args() -> list[str]:
    return ["--slip", "top"]
'''

CONSTANTS_PY = """\
# the caller's numbers
THICKNESS = 0.12      # fraction of chord
ALPHA_DEG = 2.0       # nose-up positive
RIDE_HEIGHT_M = 0.2
"""

LOCK = {
    "template_version": "t-1",
    "solver": "simpleFoam",
    "case_gen_args": ["--speed", "10", "--reynolds", "1e6", "--length", "1.0", "--belt", "ground"],
    "body_patch": "body",
    "fidelity": {"ranks": 1, "iters": 400, "window": 0.2,
                 "cells": {"min": 1000, "max": 50000}, "yplus": {"max": 300}, "reference": True},
}


@pytest.fixture
def study(tmp_path):
    round_dir = tmp_path / "rounds" / "000"
    round_dir.mkdir(parents=True)
    (round_dir / "build.py").write_text(BUILD_PY, encoding="utf-8")
    (round_dir / "hooks.py").write_text(HOOKS_PY, encoding="utf-8")
    candidate = tmp_path / "cand" / "007"
    candidate.mkdir(parents=True)
    (candidate / "design_constants.py").write_text(CONSTANTS_PY, encoding="utf-8")
    lock = tmp_path / "goal.lock.json"
    lock.write_text(json.dumps(LOCK), encoding="utf-8")
    return {"round": round_dir, "candidate": candidate, "lock": lock, "root": tmp_path}


def argv(study, mode: str) -> list[str]:
    return [str(study["candidate"]), "--round", str(study["round"]),
            "--lock", str(study["lock"]), "--mode", mode]


# -- stand-ins for the steps that need the image -------------------------------------


def looked(case: Path) -> dict:
    """What `mesh_look.py` would say about the fake polyMesh, checkMesh passing."""
    text = (case / "constant" / "polyMesh" / "owner").read_text()
    cells = int(text.split("nCells:")[1].split()[0])
    payload = {"polymesh": True, "cells": cells, "checkmesh_ok": True, "checkmesh": "Mesh OK.",
               "patches": [{"name": "body", "type": "wall"}]}
    (case / "look.json").write_text(json.dumps(payload))
    return payload


def levelled_log(case: Path, solver: str = "simpleFoam") -> None:
    import math

    lines = []
    for step in range(1, 301):
        value = max(0.015 * (1 + 0.1 * math.sin(step / 7.0)), 0.95 ** step)
        lines.append(f"Time = {step}\n\nsmoothSolver:  Solving for Ux, Initial residual = "
                     f"{value:.6e}, Final residual = {value / 10:.6e}, No Iterations 3\n\n")
    lines.append("End\n")
    (case / f"log.{solver}").write_text("".join(lines), encoding="utf-8")


def force_record(case: Path, cl: float, cd: float = 0.03) -> None:
    lines = ["# Force coefficients", "# Time  Cd  Cl  CmPitch"]
    for step in range(1, 101):
        settle = 0.3 * (1 - min(step, 80) / 80)
        wobble = 0.02 * (1 if step % 2 else -1) if step > 80 else 0.0
        lines.append(f"{step} {cd + settle / 10:.6e} {cl + settle + wobble:.6e} -0.05")
    target = case / "postProcessing" / "forces" / "0"
    target.mkdir(parents=True, exist_ok=True)
    (target / "forceCoeffs.dat").write_text("\n".join(lines) + "\n")


def alpha_of(case: Path) -> float:
    for line in (case / "design_constants.py").read_text().splitlines():
        if line.startswith("ALPHA_DEG"):
            return float(line.split("=")[1].split("#")[0])
    raise AssertionError("no ALPHA_DEG")


@pytest.fixture
def fake_image(parametric, monkeypatch):
    """checkMesh passes, case_gen records its argv, the solver leaves a levelled log
    and a force record whose Cl is linear in the candidate's ALPHA_DEG."""
    seen: dict = {"dress": [], "solve": []}
    monkeypatch.setattr(parametric, "measure_mesh", looked)

    def dress(case, args):
        seen["dress"].append((case, list(args)))
        (case / "0").mkdir(exist_ok=True)
        return ""

    def solve(case, solver, ranks):
        seen["solve"].append((case, solver, ranks))
        levelled_log(case, solver)
        force_record(case, cl=0.1 * alpha_of(case) + 0.2)
        return ""

    monkeypatch.setattr(parametric, "dress", dress)
    monkeypatch.setattr(parametric, "solve", solve)
    return seen


# -- refused before anything runs ----------------------------------------------------


def test_a_build_that_ignores_the_constants_is_refused_by_the_missing_line(parametric, study):
    (study["round"] / "build.py").write_text("print('same shape every time')\n")
    with pytest.raises(SystemExit) as raised:
        parametric.main(argv(study, "mesh"))
    assert "from design_constants import *" in str(raised.value)
    assert not (study["candidate"] / "build.py").exists(), "nothing was copied in"
    assert not (study["candidate"] / "constant").exists(), "nothing ran"


def test_a_candidate_without_constants_is_refused(parametric, study):
    (study["candidate"] / "design_constants.py").unlink()
    with pytest.raises(SystemExit, match="design_constants.py"):
        parametric.main(argv(study, "mesh"))


def test_a_round_without_a_build_script_is_refused(parametric, study):
    (study["round"] / "build.py").unlink()
    with pytest.raises(SystemExit, match="no build.py"):
        parametric.main(argv(study, "mesh"))


@pytest.mark.parametrize("extra,named", [
    ("REYNOLDS = 5e5\n", "REYNOLDS"),
    ("def report_extras():\n    return {'Cl': 9}\n", "report_extras"),
    ("class Scorer:\n    pass\n", "Scorer"),
    ("import os\nprint(os.name)\n", "expr"),
])
def test_a_hook_that_defines_anything_but_case_args_is_refused_by_name(parametric, study, extra, named):
    (study["round"] / "hooks.py").write_text(HOOKS_PY + "\n" + extra, encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        parametric.main(argv(study, "mesh"))
    message = str(raised.value)
    assert "may define only case_args()" in message
    assert named in message
    assert not (study["candidate"] / "constant").exists(), "refused before the build"


def test_a_hook_without_case_args_is_refused(parametric, study):
    (study["round"] / "hooks.py").write_text("import os\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no case_args"):
        parametric.main(argv(study, "mesh"))


def test_a_hook_may_carry_a_docstring_and_a_future_import(parametric, tmp_path):
    hooks = tmp_path / "hooks.py"
    hooks.write_text(HOOKS_PY, encoding="utf-8")
    assert parametric.hooks_definitions(hooks) == []
    assert parametric.case_args_from(hooks) == ["--slip", "top"]


def test_a_hook_that_returns_something_other_than_strings_is_refused(parametric, tmp_path):
    hooks = tmp_path / "hooks.py"
    hooks.write_text("def case_args():\n    return {'--slip': 'top'}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="list of strings"):
        parametric.case_args_from(hooks)


def test_a_hook_flag_the_lock_already_sets_is_refused(parametric, study):
    (study["round"] / "hooks.py").write_text(
        "def case_args():\n    return ['--reynolds', '1e5', '--slip', 'top']\n", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        parametric.main(argv(study, "mesh"))
    assert "--reynolds" in str(raised.value)
    assert "does not override" in str(raised.value)


def test_the_override_check_sees_the_equals_form_too(parametric):
    with pytest.raises(SystemExit, match="--speed"):
        parametric.check_no_override(["--speed", "10"], ["--speed=12"])
    parametric.check_no_override(["--speed", "10"], ["--slip", "top", "-1.5"])  # a negative number is a value


def test_a_hook_may_not_set_what_parametric_sets_from_the_fidelity(parametric):
    with pytest.raises(SystemExit, match="--iterations"):
        parametric.check_no_override([], ["--iterations", "5"])


# -- mesh mode -------------------------------------------------------------------------


def test_mesh_mode_builds_measures_and_scores_without_a_solve(parametric, study, fake_image, capsys):
    assert parametric.main(argv(study, "mesh")) == 0
    candidate = study["candidate"]

    assert (candidate / "build.py").read_text(encoding="utf-8") == BUILD_PY, "copied in"
    assert (candidate / "hooks.py").exists()
    assert (candidate / "built.txt").read_text().startswith("ALPHA_DEG=2.0"), "ran with the constants"
    assert (candidate / "log.build").exists()
    assert (candidate / "look.json").exists()
    assert fake_image["dress"] == [] and fake_image["solve"] == [], "a probe does not dress or solve"

    metrics = json.loads((candidate / "metrics.json").read_text())
    assert metrics["label"] == "ok"
    assert metrics["metrics"] == {} and metrics["window"] == 0
    assert metrics["cells"] == 12000
    assert metrics["case_args"] == [], "no case_gen line in a probe"
    assert metrics["build_sha256"]
    run = json.loads((candidate / "run.json").read_text())
    assert run["mode"] == "mesh" and run["case"] == "."
    assert "ok" in capsys.readouterr().out


def test_a_build_that_fails_is_mesh_with_the_scripts_last_words(parametric, study, fake_image):
    (study["round"] / "build.py").write_text(
        "from design_constants import *\nraise SystemExit('gmsh: the section self-intersects')\n")
    parametric.main(argv(study, "mesh"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    assert metrics["label"] == "mesh"
    assert "exited 1" in metrics["detail"]
    assert "self-intersects" in metrics["detail"]


def test_a_build_that_leaves_no_mesh_is_mesh(parametric, study, fake_image):
    (study["round"] / "build.py").write_text("from design_constants import *\nprint('ok')\n")
    parametric.main(argv(study, "mesh"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    assert metrics["label"] == "mesh"
    assert "no constant/polyMesh" in metrics["detail"]


def test_a_failed_checkmesh_is_mesh_with_the_verdict(parametric, study, fake_image, monkeypatch):
    def failing(case):
        payload = looked(case)
        payload.update({"checkmesh_ok": False, "checkmesh": "failed 1 mesh checks"})
        (case / "look.json").write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(parametric, "measure_mesh", failing)
    parametric.main(argv(study, "mesh"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    assert metrics["label"] == "mesh"
    assert "failed 1 mesh checks" in metrics["detail"]


# -- solve mode ----------------------------------------------------------------------


def test_solve_mode_dresses_with_the_lock_then_the_hook_then_the_fidelity(parametric, study, fake_image):
    assert parametric.main(argv(study, "solve")) == 0
    candidate = study["candidate"]

    [(case, args)] = fake_image["dress"]
    assert case == candidate
    assert args == ["--speed", "10", "--reynolds", "1e6", "--length", "1.0", "--belt", "ground",
                    "--slip", "top",
                    "--body", "body", "--iterations", "400", "--yplus", "--force"]
    assert fake_image["solve"] == [(candidate, "simpleFoam", 1)]

    metrics = json.loads((candidate / "metrics.json").read_text())
    assert metrics["label"] == "ok", metrics["detail"]
    assert metrics["metrics"]["Cl"] == pytest.approx(0.4, abs=1e-6), "0.1 * 2 deg + 0.2, the tail mean"
    assert metrics["window"] == 20
    assert metrics["converged"] is True
    assert metrics["case_args"] == args
    assert metrics["trim"] is None
    assert metrics["wall_seconds"] >= 0


def test_a_lock_that_names_its_body_flag_is_not_given_a_second_one(parametric):
    lock = dict(LOCK, case_gen_args=["--body", "wing", "--speed", "1"])
    args = parametric.case_gen_args(lock, [])
    assert args.count("--body") == 1 and "wing" in args


def test_a_transient_lock_means_seconds_by_iters(parametric):
    lock = dict(LOCK, case_gen_args=["--study", "transient", "--speed", "1"])
    assert "--end-time" in parametric.case_gen_args(lock, [])
    assert "--iterations" not in parametric.case_gen_args(lock, [])
    lock = dict(LOCK, case_gen_args=["--study=transient"])
    assert "--end-time" in parametric.case_gen_args(lock, [])


def test_the_solver_and_ranks_come_from_the_lock(parametric, study, fake_image):
    lock = dict(LOCK, solver="hisa", fidelity=dict(LOCK["fidelity"], ranks=4))
    study["lock"].write_text(json.dumps(lock))
    parametric.main(argv(study, "solve"))
    assert fake_image["solve"] == [(study["candidate"], "hisa", 4)]


def test_a_solver_that_diverged_is_scored_off_its_log(parametric, study, fake_image, monkeypatch):
    def blew_up(case, solver, ranks):
        (case / f"log.{solver}").write_text(
            "Time = 1\nsmoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 1e30, No Iterations 1000\n"
            "--> FOAM FATAL ERROR: Floating point exception\n")
        return f"{solver} exited 1"

    monkeypatch.setattr(parametric, "solve", blew_up)
    parametric.main(argv(study, "solve"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    assert metrics["label"] == "diverged"
    assert "Floating point exception" in metrics["detail"]


# -- the trim --------------------------------------------------------------------------


def test_the_trim_is_a_secant_over_the_constant_in_its_own_rebuilt_cases(parametric, study, fake_image):
    """Cl = 0.1 alpha + 0.2 in the stand-in solver; the target 0.8 is at alpha = 6. From
    2 and 3 the secant lands on 6 exactly, so three solves: trim_0, trim_1, trim_2."""
    lock = dict(LOCK, trim={"variable": "ALPHA_DEG", "metric": "Cl", "target": 0.8,
                            "tol": 0.01, "max_solves": 3})
    study["lock"].write_text(json.dumps(lock))
    candidate = study["candidate"]

    assert parametric.main(argv(study, "solve")) == 0

    cases = [case for case, _ in fake_image["dress"]]
    assert [c.name for c in cases] == ["trim_0", "trim_1", "trim_2"]
    assert [alpha_of(c) for c in cases] == pytest.approx([2.0, 3.0, 6.0])
    for case in cases:
        assert (case / "build.py").read_text(encoding="utf-8") == BUILD_PY
        assert (case / "hooks.py").exists()
        assert (case / "built.txt").exists(), "each iteration is a rebuild"
    # the comment on the constant's line survives the rewrite
    assert "# nose-up positive" in (cases[2] / "design_constants.py").read_text()
    assert alpha_of(candidate) == 2.0, "the caller's file is left as written"

    metrics = json.loads((candidate / "metrics.json").read_text())
    assert metrics["label"] == "ok", metrics["detail"]
    assert metrics["trim"] == {"variable": "ALPHA_DEG", "target": 0.8,
                               "achieved": pytest.approx(0.8, abs=1e-6),
                               "value": pytest.approx(6.0), "solves": 3}
    assert metrics["metrics"]["Cl"] == pytest.approx(0.8, abs=1e-6), "from the accepted case"
    run = json.loads((candidate / "run.json").read_text())
    assert run["case"] == "trim_2"
    assert not (candidate / "constant").exists(), "the candidate root itself was never built"


def test_a_trim_that_starts_inside_tolerance_solves_once(parametric, study, fake_image):
    (study["candidate"] / "design_constants.py").write_text(
        CONSTANTS_PY.replace("ALPHA_DEG = 2.0", "ALPHA_DEG = 6.0"))
    lock = dict(LOCK, trim={"variable": "ALPHA_DEG", "metric": "Cl", "target": 0.8, "tol": 0.01})
    study["lock"].write_text(json.dumps(lock))
    parametric.main(argv(study, "solve"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    assert metrics["trim"]["solves"] == 1
    assert metrics["trim"]["value"] == pytest.approx(6.0)


def test_a_trim_that_runs_out_of_solves_reports_the_nearest_and_the_miss(parametric, study, fake_image, monkeypatch):
    """A metric that is not linear: the secant does not land in three, and what is
    recorded is the nearest point, its achieved value, and that it took every solve --
    the miss is in `achieved` against `target`, for the caller to read."""
    def solve(case, solver, ranks):
        levelled_log(case, solver)
        a = alpha_of(case)
        force_record(case, cl=0.2 + 0.1 * a + 0.05 * a * a)
        return ""

    monkeypatch.setattr(parametric, "solve", solve)
    lock = dict(LOCK, trim={"variable": "ALPHA_DEG", "metric": "Cl", "target": 0.8,
                            "tol": 1e-6, "max_solves": 3})
    study["lock"].write_text(json.dumps(lock))
    parametric.main(argv(study, "solve"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    trim = metrics["trim"]
    assert trim["solves"] == 3
    assert abs(trim["achieved"] - 0.8) > 1e-6
    assert metrics["metrics"]["Cl"] == pytest.approx(trim["achieved"], abs=1e-9)


def test_a_trim_iteration_that_fails_to_mesh_ends_the_trim_with_that_label(parametric, study, fake_image):
    (study["round"] / "build.py").write_text(BUILD_PY + "\nif ALPHA_DEG > 2.5:\n    raise SystemExit('no mesh at this angle')\n")
    lock = dict(LOCK, trim={"variable": "ALPHA_DEG", "metric": "Cl", "target": 0.8})
    study["lock"].write_text(json.dumps(lock))
    parametric.main(argv(study, "solve"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    assert metrics["label"] == "mesh"
    assert "no mesh at this angle" in metrics["detail"]
    assert metrics["trim"]["solves"] == 2 and metrics["trim"]["achieved"] is None
    assert metrics["trim"]["value"] == pytest.approx(3.0)


def test_a_trim_iteration_that_diverges_ends_the_trim_as_diverged(parametric, study, fake_image, monkeypatch):
    """A coefficient off a diverged run is not a point to take a slope over."""
    def solve(case, solver, ranks):
        if alpha_of(case) > 2.5:
            (case / f"log.{solver}").write_text(
                "Time = 1\nsmoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 1e30, No Iterations 1000\n"
                "--> FOAM FATAL ERROR: Floating point exception\n")
            force_record(case, cl=99.0)
            return f"{solver} exited 1"
        levelled_log(case, solver)
        force_record(case, cl=0.1 * alpha_of(case) + 0.2)
        return ""

    monkeypatch.setattr(parametric, "solve", solve)
    lock = dict(LOCK, trim={"variable": "ALPHA_DEG", "metric": "Cl", "target": 0.8})
    study["lock"].write_text(json.dumps(lock))
    parametric.main(argv(study, "solve"))
    metrics = json.loads((study["candidate"] / "metrics.json").read_text())
    assert metrics["label"] == "diverged"
    assert "trim_1" in metrics["detail"] and "Floating point" in metrics["detail"]
    assert metrics["trim"]["solves"] == 2 and metrics["trim"]["achieved"] is None
    run = json.loads((study["candidate"] / "run.json").read_text())
    assert run["trim"]["solves"] == 2, "checkpointed after every solve"


def test_a_trim_on_a_constant_the_candidate_does_not_have_is_refused(parametric, study, fake_image):
    lock = dict(LOCK, trim={"variable": "FLAP_DEG", "metric": "Cl", "target": 0.8})
    study["lock"].write_text(json.dumps(lock))
    with pytest.raises(SystemExit, match="FLAP_DEG"):
        parametric.main(argv(study, "solve"))


def test_the_secant_step(parametric):
    assert parametric.next_point([(2.0, 0.4)], 0.8) == 3.0
    assert parametric.next_point([(2.0, 0.4), (3.0, 0.5)], 0.8) == pytest.approx(6.0)
    assert parametric.next_point([(2.0, 0.4), (3.0, 0.4)], 0.8) == 4.0, "a flat pair steps again"


def test_set_constant_rewrites_one_line_and_keeps_the_rest(parametric):
    text = parametric.set_constant(CONSTANTS_PY, "ALPHA_DEG", 6.0)
    assert "ALPHA_DEG = 6.0  # nose-up positive" in text
    assert "THICKNESS = 0.12" in text and "# the caller's numbers" in text
    with pytest.raises(SystemExit, match="FLAP_DEG"):
        parametric.set_constant(CONSTANTS_PY, "FLAP_DEG", 1.0)


def test_read_constants_takes_only_number_lines(parametric, tmp_path):
    path = tmp_path / "design_constants.py"
    path.write_text(CONSTANTS_PY + "NAME = 'wing'\nSCALE = 1e-3\nNEG = -2.5e+1\n")
    assert parametric.read_constants(path) == {"THICKNESS": 0.12, "ALPHA_DEG": 2.0,
                                               "RIDE_HEIGHT_M": 0.2, "SCALE": 1e-3, "NEG": -25.0}


# -- the shipped round -----------------------------------------------------------------


def test_the_template_round_passes_every_check_parametric_makes(parametric, tmp_path):
    """`templates/evaluate_steady.py --write-round` is the documented example; if it
    would be refused by the script it documents, the example is wrong."""
    round_dir = tmp_path / "round"
    proc = subprocess.run([sys.executable, str(TOOLBOX / "templates" / "evaluate_steady.py"),
                           "--write-round", str(round_dir)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    for name in ("build.py", "hooks.py", "goal.lock.json", "design_constants.py"):
        assert (round_dir / name).is_file(), name

    candidate = tmp_path / "cand"
    candidate.mkdir()
    (candidate / "design_constants.py").write_bytes((round_dir / "design_constants.py").read_bytes())
    hook_args = parametric.prepare(candidate, round_dir)
    assert hook_args == ["--slip", "top"]

    lock = json.loads((round_dir / "goal.lock.json").read_text())
    parametric.check_no_override(lock["case_gen_args"], hook_args)
    for key in ("solver", "case_gen_args", "body_patch", "fidelity", "trim", "template_version"):
        assert key in lock, key
    for key in ("ranks", "iters", "window", "cells", "yplus", "reference"):
        assert key in lock["fidelity"], key
    assert set(lock["trim"]) == {"variable", "metric", "target", "tol", "max_solves"}
    constants = parametric.read_constants(candidate / "design_constants.py")
    assert lock["trim"]["variable"] in constants
    for name in ("CHORD_M", "CAMBER_1", "CAMBER_POS", "THICKNESS", "ALPHA_DEG", "RIDE_HEIGHT_M"):
        assert name in constants, name
    args = parametric.case_gen_args(lock, hook_args)
    assert args[-2:] == ["--yplus", "--force"] and "--iterations" in args
