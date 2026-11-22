"""Solve the porous slab, and compare the drop in the unit the solver actually reports.

`porous.py` has shipped a complete `porousSimpleFoam` case for a long time, and the
tutorial index has said of it, in these words, that the solve is ungraded: "nothing in
this repository has solved this case and quoted a drop against the 503.34 Pa the template
prints, and until something has, that is the state". This is that something.

**The whole case turns on a unit**, which is why it is the one worth grading. The
template's own docstring says it plainly: `porousSimpleFoam` is incompressible and solves
`p/rho` in m2/s2, so a solved field reading about 418 and a datasheet reading 503 Pa
agree exactly, and calling that a 17% error is the mistake. This test measures the drop
the way the solver reports it, converts once, and asserts the conversion in both
directions: the Pa comparison must pass AND the uncorrected comparison must fail by the
factor rho. A test that only did the first would pass just as happily if somebody later
made the template print pascals into a field that is not in pascals.

The reference is exact by construction rather than published: `dp = (mu d U +
0.5 rho f U^2) L` is OpenFOAM's own momentum sink integrated across the slab, so what is
being checked is that the solver applies the sink the template wrote, on the mesh the
template built, and nothing about whether Ergun was right about packed beds.

Run by the `journey` job, in the workspace image, on every pull request.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "openreynolds" / "toolbox" / "templates" / "porous.py"

pytestmark = pytest.mark.skipif(
    not os.environ.get("FOAM_TUTORIALS"),
    reason="needs an OpenFOAM environment: this one solves the case rather than writing it")


def load():
    spec = importlib.util.spec_from_file_location("template_porous", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shell(cmd: list[str], cwd: Path, timeout: float = 1800) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if done.stdout:
        print(done.stdout[-2500:], flush=True)
    assert done.returncode == 0, (
        f"{cmd[0]} exited {done.returncode}\n--- stderr ---\n{done.stderr[-2500:]}")
    return done


def patch_pressure(case: Path, patch: str) -> float:
    """The area-averaged pressure on a patch, from the function object the case itself ran.

    No `postProcess` call: the template already declares `inletPressure` and
    `outletPressure` surfaceFieldValue objects (see its `functions` block), so the solve
    writes both files as it goes and the numbers are in the solver log too. Reading those
    is reading the case's own instrument.

    Two earlier versions of this helper measured nothing, and both failures are worth
    keeping in mind. The first globbed `*{patch}*/**/*.dat` and took the last match, so
    it read a file the test never wrote. The second ran its own `postProcess` and
    addressed that directory exactly, but took the last column of the last row, which in
    that file is the time -- both ends came back as 103.0, the final iteration. A drop
    that comes out identical for inlet and outlet is the signature of reading the wrong
    column or the wrong file, which is why this one asserts its header.
    """
    root = case / "postProcessing" / f"{patch}Pressure"
    assert root.is_dir(), (
        f"the solve wrote no {patch}Pressure output. Under postProcessing: "
        + ", ".join(sorted(str(q.relative_to(case))
                           for q in (case / "postProcessing").rglob("*"))))
    found = sorted(root.rglob("surfaceFieldValue.dat"))
    assert found, f"{root} holds no surfaceFieldValue.dat"

    text = found[-1].read_text(encoding="utf-8")
    # The header names the columns, and it has to say which patch and which operation,
    # so that a file for the other end or for a different average cannot be read by
    # mistake -- the exact confusion that produced inlet == outlet twice.
    header = [l for l in text.splitlines() if l.startswith("#")]
    assert any(patch in l for l in header), (
        f"{found[-1]} does not mention {patch} in its header: {header}")
    assert any("areaAverage(p)" in l for l in header), (
        f"{found[-1]} is not an areaAverage of p: {header}")

    rows = [l.split() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    assert rows, f"{found[-1]} holds no rows"
    assert len(rows[-1]) == 2, (
        f"expected 'time value' in {found[-1]}, got {rows[-1]}")
    return float(rows[-1][1])


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """Write the case, mesh it, solve it."""
    porous = load()
    case = tmp_path_factory.mktemp("porous-journey")
    shell([sys.executable, str(TEMPLATE)], cwd=case)
    assert (case / "constant" / "polyMesh").is_dir(), "the template built no mesh"
    assert (case / "constant" / "porosityProperties").is_file(), (
        "no porosityProperties: the sink this case exists to check was never written")
    shell(["porousSimpleFoam"], cwd=case)
    return case, porous


def test_the_sink_was_written_with_both_coefficients(solved):
    """`d` and `f` are vectors, and the components nobody thinks about are the ones that
    matter: a slab that resists along its thickness and not across it is a different
    physical object from the one the coefficients describe."""
    case, _ = solved
    text = (case / "constant" / "porosityProperties").read_text(encoding="utf-8")
    assert "DarcyForchheimer" in text, text[:400]
    for key in ("d", "f"):
        assert re.search(rf"^\s*{key}\s", text, re.M), f"no {key} entry in porosityProperties"


def test_the_drop_matches_the_coefficients_that_were_written(solved):
    """The reference is exact by construction: the sink integrated across the slab.

    What it checks is that the solver applied the sink the template wrote, on the mesh
    the template built. It says nothing about whether Ergun was right, and the tutorial
    page says so too.
    """
    case, porous = solved
    inlet = patch_pressure(case, "inlet")
    outlet = patch_pressure(case, "outlet")
    drop_kinematic = inlet - outlet
    assert drop_kinematic > 1e-6, (
        f"the slab reported no drop at all: inlet {inlet}, outlet {outlet}. A porous "
        "zone that resists nothing means the sink was not applied or the wrong file was "
        "read -- either way it is a broken measurement, not a small answer")
    drop_pa = drop_kinematic * porous.RHO_KG_M3

    speed = porous.design_velocity_m_s()
    darcy, forchheimer = porous.coefficients_from_curve()[:2]
    expected_pa = porous.predicted_drop_pa(speed, darcy, forchheimer)

    print(f"  solved drop {drop_kinematic:.4f} m2/s2 = {drop_pa:.4f} Pa", flush=True)
    print(f"  the coefficients predict {expected_pa:.4f} Pa at {speed:.4f} m/s", flush=True)
    print(f"  difference {(drop_pa - expected_pa) / expected_pa:+.3%}", flush=True)

    # 12% of the drop. The sink is exact, but the slab sits in a duct: the solved drop
    # between the two patches also carries the duct's own wall friction over 0.55 m, and
    # the development length upstream of the slab. Those are real and small, not error.
    assert abs(drop_pa - expected_pa) / expected_pa < 0.12, (drop_pa, expected_pa)


def test_the_unit_is_the_point_and_the_wrong_one_fails(solved):
    """The failure this case exists to prevent, asserted in both directions.

    `porousSimpleFoam` solves p/rho. Comparing the raw field against a datasheet in
    pascals understates the drop by exactly rho, and the template's docstring records
    somebody calling that a 17% error when the two agreed exactly. If a later change ever
    made the field pascals, the first assertion here would start passing and this one
    would fail, which is the pair that keeps the question honest.
    """
    case, porous = solved
    drop_kinematic = patch_pressure(case, "inlet") - patch_pressure(case, "outlet")
    assert drop_kinematic > 1e-6, f"no drop to take a ratio of: {drop_kinematic}"
    speed = porous.design_velocity_m_s()
    darcy, forchheimer = porous.coefficients_from_curve()[:2]
    expected_pa = porous.predicted_drop_pa(speed, darcy, forchheimer)

    ratio = expected_pa / (drop_kinematic * porous.RHO_KG_M3)
    print(f"  pascals over converted: {ratio:.4f}", flush=True)
    assert 0.88 < ratio < 1.12, "the converted comparison should be the one that agrees"

    naive = expected_pa / drop_kinematic
    print(f"  pascals over RAW field: {naive:.4f}, and rho is {porous.RHO_KG_M3:g}",
          flush=True)
    # It should land on rho itself, because that IS the error: 503.34 / 418.06 = 1.204.
    # Written against rho rather than against a number, so air becoming water moves it.
    assert abs(naive - porous.RHO_KG_M3) / porous.RHO_KG_M3 < 0.12, (
        "comparing the raw p/rho field against a drop in pascals should be wrong by "
        f"about rho = {porous.RHO_KG_M3:g}; it came out {naive:.4f}, so either the "
        "field is no longer p/rho or the density changed")
    assert naive > 1.05, "rho near 1 would make this check vacuous; it is not a bound"
