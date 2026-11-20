"""The journey a user takes, end to end, with the physics asserted at every seam.

Every other test in this repository is either a unit test against a fixture or a
tool run against a study somebody already finished. Nothing walked the chain: start
from an empty directory, build a mesh, dress it into a case, solve it, read the
field back with the real readers, post-process it, and check the answer. A seam
between two of those stages -- a factor of ten, a unit mismatch, a field the solver
wrote under a name the post-processor does not look for -- was invisible to the
whole suite.

So this is one case that genuinely solves, and the assertions are on the physics
rather than on the exit codes.

the case, and why its answer is known exactly

`openreynolds/toolbox/templates/duct2d.py` as it ships: a 0.10 x 0.02 m plane duct,
one cell thick, about ten cells across the height. `case_gen.py` dresses it for a
steady laminar solve at Re = 10 on the hydraulic diameter. Far enough from the inlet
the flow is fully developed plane Poiseuille flow, whose answer is not a published
measurement with an uncertainty on it but the exact solution of the steady
Navier-Stokes equations -- which is what makes it the right first journey:

    developed, 2D, steady:   nu d2u/dy2 = (1/rho) dp/dx = const,  u(0) = u(H) = 0

    =>  u(y) = -(1/(2 nu)) (dp/dx)_kin y (H - y)        (a parabola)
        u_mean = -(H^2 / (12 nu)) (dp/dx)_kin           (integrate over H)
        u_max  = u(H/2) = 1.5 u_mean                    (put y = H/2 in the first)
        (dp/dx)_kin = -12 nu u_mean / H^2               (rearrange the second)
        tau_wall / rho = nu |du/dy|_0 = 6 nu u_mean / H (differentiate the first)

Three lines of algebra, no correlation, no constant anybody has to look up. The
plane-channel case of Hagen-Poiseuille flow; White, *Viscous Fluid Flow*, 3rd ed.,
section 3-3.2 derives the same expressions. u_mean is 1 m/s exactly because the
inlet is a uniform 1 m/s fixedValue and mass is conserved, so every number below is
fixed by the case's own settings and nothing is fitted.

The one thing that is a correlation rather than an identity is where the entrance
region ends, and it is used only to choose where to measure, never as an answer: at
Re_Dh = 10 the development length is well under 0.04 m (Durst et al. 2005 give
L/Dh = [0.619^1.6 + (0.0567 Re)^1.6]^(1/1.6) for a pipe, and the plane-channel form
is of the same order -- under one Dh here), so the measurements are taken between
x = 0.06 and x = 0.09, clear of both the entrance and the outlet.

what is asserted at each seam, and not at the end

* the mesh exists, is one cell thick, and its z faces are `empty` -- a "2D" case that
  is really a thin 3D one is the standard way this goes quietly wrong;
* `case_gen.py` wrote nu = U L / Re into `constant/transportProperties`, so the
  viscosity the solver used is the one the Reynolds number asked for;
* the solver wrote U and p at a time the readers can find, with p in m^2 s^-2
  (kinematic) -- the unit a force computed from it would be out by rho on;
* the mass balance closes, and the throughflow equals U * H * thickness, which is
  the assertion a factor of ten in the mesh's units could not survive;
* the developed profile is a parabola peaking at 1.5 u_mean;
* the developed pressure gradient is -12 nu u_mean / H^2;
* y+ is measured from the mesh and the field, with nu read off the case's own
  dictionary rather than defaulted;
* `report.py`'s key numbers trace back to the residuals in the solver's own log.

Skipped where there is no OpenFOAM. The place this runs is the `journey` job in
`.github/workflows/tests.yml`, inside the workspace image.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[0] / "openreynolds" / "toolbox"
TEMPLATE = TOOLBOX / "templates" / "duct2d.py"

# gmshToFoam and checkMesh are duct2d.py's; simpleFoam is the solve; foamDictionary
# is how the template retypes the patches.
BINARIES = ("gmshToFoam", "checkMesh", "simpleFoam", "foamDictionary")


def _absent() -> list[str]:
    missing = [name for name in BINARIES if shutil.which(name) is None]
    if importlib.util.find_spec("gmsh") is None:
        missing.append("the gmsh module")
    return missing


pytestmark = pytest.mark.skipif(
    bool(_absent()),
    reason="not an OpenFOAM machine, so no case can be solved here: "
           + ", ".join(_absent() or ["-"]) + " missing",
)


# -- the case, exactly as the template and the generator define it -------------------

LENGTH_M = 0.10               # duct2d.py:64  LENGTH_M
HEIGHT_M = 0.02               # duct2d.py:65  HEIGHT_M
THICK_M = HEIGHT_M / 10.0     # duct2d.py:68-69  CELL_SIZE_M = HEIGHT_M/10; THICK_M = that
SPEED = 1.0                   # --speed, the uniform inlet
REYNOLDS = 10.0               # --reynolds
RE_LENGTH = 2.0 * HEIGHT_M    # --length: the hydraulic diameter of a plane passage
NU = SPEED * RE_LENGTH / REYNOLDS   # case_gen.derive_flow:592  nu = U*L/Re  -> 4e-3

# Where the measurements are taken: developed, and clear of the outlet.
X0, X1 = 0.06, 0.09
X_PROFILE = 0.075

# -- the exact answer, from the algebra in the docstring -----------------------------

U_MEAN = SPEED                                        # uniform inlet + continuity
U_MAX = 1.5 * U_MEAN                                  # = 1.5 m/s
DPDX = -12.0 * NU * U_MEAN / HEIGHT_M**2              # = -120 m/s^2, kinematic
DROP = DPDX * (X1 - X0)                               # = -3.6 m^2/s^2 over 30 mm
TAU_WALL = 6.0 * NU * U_MEAN / HEIGHT_M               # = 1.2 m^2/s^2, kinematic
U_TAU = math.sqrt(TAU_WALL)                           # = 1.0954 m/s
Y_PLUS = U_TAU * (THICK_M / 2.0) / NU                 # = 0.274 at a 2 mm first cell
FLUX = U_MEAN * HEIGHT_M * THICK_M                    # = 4e-5 m^3/s

# Tolerances, each with the reason it is the number it is. None of them is a
# published band: the reference is exact, so every one of these is the discretisation
# this mesh is allowed, and a tighter one would be a test of ten cells rather than of
# the chain.
TOL = {
    # Boundary fluxes over a converged SIMPLE solve; this is continuity, not accuracy.
    "closure": 1e-3,
    # A fixedValue inlet on planar faces: the only error is the face-area sum. A
    # factor of ten anywhere in the geometry or the units fails this by 900%.
    "flux": 0.02,
    # Ten cells across the height, sampled at the nodes (where VTK averages the two
    # neighbouring cells, which reads about half a percent under the true peak).
    "u_max": 0.04,
    # Second-order on ten cells, over a 30 mm window of a 100 mm duct.
    "dpdx": 0.08,
    # y+ is first-cell-height-sensitive and gmsh's quads are not a perfectly uniform
    # 50 x 10 grid, so this allows the spacing to wander. What it still catches is a
    # defaulted viscosity or a y+ off by a factor.
    "y_plus": 0.25,
    # The two walls of a symmetric duct see the same shear. Any gap is numerical.
    "symmetry": 0.05,
}


def run(label: str, cmd: list[str], cwd: Path | None = None, check: bool = True):
    """A stage of the journey, printed with its wall-clock time so the CI log reads
    as the walk it is -- and so the job's cost is visible per stage, not just in
    total."""
    started = time.monotonic()
    done = subprocess.run([str(part) for part in cmd], cwd=None if cwd is None else str(cwd),
                          capture_output=True, text=True)
    took = time.monotonic() - started
    print(f"\n=== {label}  [{took:.1f}s, exit {done.returncode}]")
    print("$ " + " ".join(str(part) for part in cmd))
    print(done.stdout[-8000:])
    if done.returncode != 0:
        print(done.stderr[-4000:], file=sys.stderr)
        if check:
            pytest.fail(f"the journey stopped at: {label} (exit {done.returncode})\n"
                        f"--- stdout tail ---\n{done.stdout[-3000:]}\n"
                        f"--- stderr tail ---\n{done.stderr[-3000:]}")
    return done


def script(name: str, *args) -> list[str]:
    return [sys.executable, str(TOOLBOX / f"{name}.py"), *args]


def load(name: str):
    """Import a toolbox script by path, the way tests/test_toolbox.py does."""
    spec = importlib.util.spec_from_file_location(f"journey_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def journey(tmp_path_factory) -> dict:
    """The whole walk, once: empty directory -> mesh -> case -> solve -> readings.

    One fixture rather than a test per stage because the stages are not independent
    -- there is nothing to read before there is something solved -- and because one
    solve is the whole runner budget this is allowed. Each seam is then its own
    named assertion below, so a failure says which seam.
    """
    study = tmp_path_factory.mktemp("journey")
    case = study / "duct"
    case.mkdir()
    readings = study / "readings"
    readings.mkdir()
    started = time.monotonic()

    # 1. nothing -> a mesh. The template is run where it is, with the case as the
    #    working directory, which is how its `find_mesh_look` reaches the toolbox.
    run("mesh: duct2d.py (gmsh -> gmshToFoam -> checkMesh -> mesh_look)",
        [sys.executable, TEMPLATE], cwd=case)

    # 2. the mesh -> a case. --force because the template left system/ stubs behind.
    case_gen = run("case: case_gen.py reads the mesh and writes 0/, system/, constant/",
                   script("case_gen", case, "--study", "steady", "--speed", SPEED,
                          "--reynolds", REYNOLDS, "--length", RE_LENGTH,
                          "--iterations", 2000, "--force"))

    # 3. the solve. The log is kept where `results.find_solver_log` looks for it,
    #    which is what a person's `tee log.simpleFoam` does.
    solve = run("solve: simpleFoam", ["simpleFoam", "-case", case])
    log = case / "log.simpleFoam"
    log.write_text(solve.stdout, encoding="utf-8")

    # 4. the two wall fields the post-processors look for. `case_gen.py` writes no
    #    functions block for a case with no body patch, so these are asked for
    #    afterwards -- through the solver, because the generic `postProcess` builds no
    #    momentum-transport model and these two function objects need one.
    for func in ("yPlus", "wallShearStress"):
        run(f"post: simpleFoam -postProcess -func {func}",
            ["simpleFoam", "-case", case, "-postProcess", "-func", func, "-latestTime"],
            check=False)

    # 5. read it back with the real readers.
    def probe(label: str, *args) -> dict:
        out = readings / f"{label}.json"
        run(f"read: probe.py {label}", script("probe", case, "--out", out, *args))
        return json.loads(out.read_text(encoding="utf-8"))

    fields = probe("fields", "--fields")
    flux_in = probe("flux-in", "--flux", "--at-x", 0.001)
    flux_out = probe("flux-out", "--flux", "--at-x", LENGTH_M - 0.001)
    profile = probe("profile", "--line", X_PROFILE, 0.0, X_PROFILE, HEIGHT_M,
                    "--field", "U", "--n", 201)
    drop = probe("drop", "--between", X0, X1, "--field", "p", "--weighted", "flux")

    balance_json = readings / "balance.json"
    run("post: balance.py (the mass balance)",
        script("balance", case, "--json", balance_json, "--png", readings / "balance.png"))

    wall_json = readings / "wall.json"
    run("post: wall.py (y+ as achieved)", script("wall", case, "--json", wall_json))

    digest_out = run("post: log_digest.py (convergence)",
                     script("log_digest", log, "-o", readings / "residuals.png"))

    run("post: results.py --preset duct-flow (the figures)",
        script("results", case, "--preset", "duct-flow", "--out", case / "results"))

    report_json = readings / "report.json"
    run("post: report.py (the PDF)",
        script("report", study, "--json", report_json, "--out", study / "journey.pdf"))

    elapsed = time.monotonic() - started
    print(f"\n=== the whole journey took {elapsed:.1f}s")

    return {
        "study": study, "case": case, "log": log,
        "case_gen_stdout": case_gen.stdout,
        "solve_stdout": solve.stdout,
        "fields": fields,
        "flux_in": flux_in, "flux_out": flux_out,
        "profile": profile, "drop": drop,
        "balance": json.loads(balance_json.read_text(encoding="utf-8")),
        "wall": json.loads(wall_json.read_text(encoding="utf-8")),
        "digest_stdout": digest_out.stdout,
        "report": json.loads(report_json.read_text(encoding="utf-8")),
        "pdf": study / "journey.pdf",
        "seconds": elapsed,
    }


# -- seam 1: the mesh is the mesh that was asked for ---------------------------------


def test_the_mesh_is_one_cell_thick_and_its_z_faces_are_empty(journey):
    """A "2D" OpenFOAM case is a 3D mesh one cell deep with the z faces typed
    `empty`. The two ways this goes wrong silently -- several cells in z, or a side
    patch left `patch` -- both produce a case that runs and answers a different
    question, so they are checked here and not inferred from the answer."""
    boundary = (journey["case"] / "constant" / "polyMesh" / "boundary").read_text(
        encoding="utf-8")
    assert "frontAndBack" in boundary, boundary
    front = boundary.split("frontAndBack", 1)[1][:400]
    assert "empty" in front, f"frontAndBack is not empty:\n{front}"
    for wall in ("wall_bottom", "wall_top"):
        assert wall in boundary, f"{wall} is not in the mesh's boundary file"

    checked = journey["case"] / "look.json"
    assert checked.is_file(), "duct2d.py did not finish: mesh_look wrote no look.json"
    look = json.loads(checked.read_text(encoding="utf-8"))
    print(json.dumps({k: look[k] for k in list(look)[:12]}, indent=2, default=str)[:2000])


# -- seam 2: the Reynolds number the user asked for is the viscosity the solver used --


def test_case_gen_turned_the_reynolds_number_into_the_viscosity_on_disk(journey):
    """`--reynolds 10 --length 0.04 --speed 1` has exactly one answer, nu = U L / Re,
    and the file the solver reads is where it has to appear. This is the seam a
    remembered viscosity from last week used to cross."""
    transport = (journey["case"] / "constant" / "transportProperties").read_text(
        encoding="utf-8")
    print(transport)
    written = [line for line in transport.splitlines() if line.strip().startswith("nu ")]
    assert written, f"no nu in transportProperties:\n{transport}"
    value = float(written[0].split()[-1].rstrip(";"))
    assert value == pytest.approx(NU, rel=1e-6), (
        f"transportProperties says nu = {value:g}; U L / Re = {NU:g}")
    assert "laminar" in journey["case_gen_stdout"], (
        "Re = 10 is laminar and case_gen did not say so:\n" + journey["case_gen_stdout"])


# -- seam 3: the solver wrote what the post-processors look for -----------------------


def test_the_solve_converged_and_log_digest_can_read_it(journey):
    """`log_digest.py` is the convergence reader. What matters at this seam is that
    the log it is handed is a log it can parse: residual lines it recognises, a
    verdict, and a continuity summary. A run that stopped at the iteration ceiling
    with residuals still falling would read differently, and does not here."""
    digest = load("log_digest").digest(journey["log"])
    shape = load("log_digest").residual_shape(digest["residuals"])
    print(journey["digest_stdout"])
    assert digest["residuals"], "log_digest found no residual lines in the solver log"
    for field in ("Ux", "Uy", "p"):
        assert field in digest["residuals"], (
            f"no residual series for {field}; got {sorted(digest['residuals'])}")
    last = shape["last"]
    assert last["p"] < 1e-3, f"p never came down: final initial residual {last['p']:g}"
    assert last["Ux"] < 1e-3, f"Ux never came down: {last['Ux']:g}"
    assert digest["fatal"] is None, digest["fatal"]
    assert digest["ended"], "the solver log has no `End`, so the run did not finish"
    assert digest["converged_at"], (
        "simpleFoam did not report `solution converged in N iterations`, so the "
        "residualControl that case_gen's fv_solution writes was never met inside 2000")
    print(f"converged at iteration {digest['converged_at']}, "
          f"ExecutionTime {digest['exec_time']} s")


def test_the_solver_wrote_U_and_p_and_p_is_kinematic(journey):
    """The field names and the units the whole post-processing chain assumes. An
    incompressible OpenFOAM `p` is p/rho in m^2 s^-2; a number taken off it as if it
    were pascals is out by rho and looks entirely plausible, which is why the unit is
    asserted and not just the presence of the field."""
    fields = journey["fields"]["data"]
    print(json.dumps(fields, indent=2))
    assert "U" in fields and "p" in fields, f"the case wrote {sorted(fields)}"
    assert fields["U"]["units"] == "m s^-1", fields["U"]["units"]
    assert fields["p"]["units"] == "m^2 s^-2", (
        f"p is in {fields['p']['units']!r}, not the kinematic m^2 s^-2 every reader "
        "downstream of this assumes")


# -- seam 4: the mass balance closes, in the right units -----------------------------


def test_the_mass_balance_closes(journey):
    """`balance.py` tells inflow from outflow by the sign of U.A and never by a
    patch's name. In a steady incompressible duct the two are equal; the residue is
    numerical, and it is the single most useful diagnostic an internal-flow case
    has."""
    balance = journey["balance"]["balance"]
    patches = journey["balance"]["patches"]
    print(json.dumps({"balance": balance,
                      "patches": [{k: row[k] for k in ("patch", "direction", "flux")
                                   if k in row} for row in patches]}, indent=2))
    relative = balance["closure_error_relative"]
    assert relative is not None, "balance.py found no throughflow to close over"
    assert abs(relative) < TOL["closure"], (
        f"the mass balance does not close: {100 * relative:.4g}% of the throughflow")
    # Read off the sign of U.A, never off the name -- so this also checks the mesh's
    # `inlet` really is where the flow goes in.
    directions = {row["patch"]: row.get("direction") for row in patches}
    assert directions.get("inlet") == "inflow", directions
    assert directions.get("outlet") == "outflow", directions
    assert directions.get("wall_top") == "closed", directions
    assert journey["balance"]["balance"]["flux_units"] == "m^3 s^-1", (
        "an incompressible case has no density, so the balance is volumetric")


def test_the_throughflow_is_the_inlet_speed_times_the_inlet_area(journey):
    """U * H * thickness, with every length read off the mesh the template built.
    This is the assertion a factor of ten -- a duct authored in millimetres and never
    scaled, the mistake duct2d.py's own notes warn about -- cannot survive."""
    inflow = journey["flux_in"]["data"]["value"]
    outflow = journey["flux_out"]["data"]["value"]
    print(f"inlet flux {inflow:.6g} m3/s, outlet flux {outflow:.6g} m3/s, "
          f"U*H*t = {FLUX:.6g} m3/s")
    assert inflow == pytest.approx(FLUX, rel=TOL["flux"]), (
        f"{inflow:.6g} m3/s through the inlet section; U*H*t = {FLUX:.6g}")
    assert outflow == pytest.approx(FLUX, rel=TOL["flux"])


# -- seam 5: the physics, against the exact solution ---------------------------------


def test_the_developed_profile_is_a_parabola_peaking_at_one_and_a_half_u_mean(journey):
    """u_max = 1.5 u_mean, exactly, for developed plane Poiseuille flow -- the
    docstring's algebra, no constant from anywhere else. The symmetry of the profile
    is checked beside it, because a parabola with the right peak in the wrong place
    is a different flow."""
    data = journey["profile"]["data"]
    ys = [y for y, u in zip(data["y"], data["U_x"]) if u is not None]
    us = [u for u in data["U_x"] if u is not None]
    assert len(us) > 50, f"only {len(us)} of {len(data['U_x'])} line samples hit the mesh"

    peak = max(us)
    at = ys[us.index(peak)]
    # The mean over the section, from the same samples. Three assertions follow from
    # it and none of them depends on whether VTK handed back the interpolated node
    # values or the containing cell's -- which is a property of the reader, not of the
    # flow, and not what this is testing.
    mean = sum(us) / len(us)
    print(f"u_max = {peak:.5g} m/s at y = {at * 1000:.3g} mm, mean {mean:.5g} m/s; "
          f"exact: 1.5 u_mean = {U_MAX:.5g} m/s at y = {HEIGHT_M * 500:.3g} mm")

    assert mean == pytest.approx(U_MEAN, rel=TOL["u_max"]), (
        f"the section mean is {mean:.5g} m/s, not the {U_MEAN:g} m/s the inlet put in")
    assert peak == pytest.approx(U_MAX, rel=TOL["u_max"]), (
        f"u_max is {peak:.5g} m/s; developed plane Poiseuille flow at u_mean = "
        f"{U_MEAN:g} m/s gives {U_MAX:g} m/s")
    assert peak / mean == pytest.approx(1.5, rel=TOL["u_max"]), (
        f"u_max / u_mean is {peak / mean:.4g}; the exact developed value is 1.5 "
        "(2.0 would be a pipe, 1.0 a profile that never developed)")
    assert at == pytest.approx(HEIGHT_M / 2.0, abs=HEIGHT_M / 10.0), (
        f"the peak is at y = {at:.5g} m, not mid-channel")

    # Symmetric about the centreline, which a parabola is and a developing or a
    # one-sided profile is not. Paired on the line's own index, not on the filtered
    # list: the samples are evenly spaced from y = 0 to y = H, so sample i and sample
    # n-1-i are the same distance from opposite walls.
    raw, raw_y = data["U_x"], data["y"]
    pairs = [(u, raw[len(raw) - 1 - i])
             for i, (u, y) in enumerate(zip(raw, raw_y))
             if u is not None and raw[len(raw) - 1 - i] is not None
             and HEIGHT_M * 0.05 < y < HEIGHT_M / 2]
    assert pairs, "no symmetric pair of samples survived"
    worst = max(abs(u - v) for u, v in pairs) / U_MAX
    print(f"worst top-to-bottom asymmetry: {100 * worst:.3g}% of u_max")
    assert worst < TOL["symmetry"], (
        f"the profile is not symmetric about the centreline: {100 * worst:.3g}% of u_max")


def test_the_developed_pressure_gradient_is_the_poiseuille_gradient(journey):
    """(dp/dx)_kin = -12 nu u_mean / H^2, measured as the flux-weighted mixed mean
    between two sections 30 mm apart in the developed region -- which is the honest
    average for a duct and what `probe.py --between` computes."""
    difference = journey["drop"]["data"]["difference"]
    if isinstance(difference, dict):
        difference = next(iter(difference.values()))
    gradient = difference / (X1 - X0)
    print(f"dp between x={X0} and x={X1}: {difference:.5g} m2/s2 "
          f"-> dp/dx = {gradient:.5g} m/s2; -12 nu u_mean / H^2 = {DPDX:.5g} m/s2")
    assert journey["drop"]["units"] == "m^2 s^-2", journey["drop"]["units"]
    assert gradient == pytest.approx(DPDX, rel=TOL["dpdx"]), (
        f"dp/dx is {gradient:.5g} m/s2; the exact developed value is {DPDX:.5g}")


# -- seam 6: y+ is measured, from this mesh and this field ---------------------------


def test_y_plus_is_measured_off_the_case_and_not_defaulted(journey):
    """`wall.py` refuses to assume a viscosity, because a y+ off a guessed nu looks
    like a measurement and is not one. So the seam here is twofold: that nu came out
    of this case's own dictionary, and that the y+ it then reports is the one the
    exact wall shear implies -- u_tau = sqrt(6 nu u_mean / H), y the half-height of
    the first cell."""
    found = journey["wall"]
    print(json.dumps({"nu": found["nu"], "nu_from": found["nu_from"],
                      "sources": [{"source": s["source"],
                                   "patches": s["patches"]} for s in found["sources"]],
                      "problems": found.get("problems")}, indent=2, default=str))

    assert found["nu"] == pytest.approx(NU, rel=1e-6), (
        f"wall.py read nu = {found['nu']}; the case says {NU:g}")
    assert "transportProperties" in (found["nu_from"] or ""), (
        f"nu came from {found['nu_from']!r}, which is not the case's own dictionary")
    assert found["sources"], (
        "no y+ source at all: neither a yPlus function object, a written yPlus field, "
        "nor a wallShearStress field reached wall.py")

    for source in found["sources"]:
        rows = {row["patch"]: row for row in source["patches"]}
        walls = [rows[name] for name in ("wall_bottom", "wall_top") if name in rows]
        assert len(walls) == 2, (
            f"{source['source']} reported {sorted(rows)}, not both duct walls")
        for row in walls:
            assert row["mean"] == pytest.approx(Y_PLUS, rel=TOL["y_plus"]), (
                f"{source['source']} gives y+ = {row['mean']:.4g} on {row['patch']}; "
                f"u_tau = sqrt(6 nu u_mean / H) = {U_TAU:.4g} m/s over a "
                f"{THICK_M * 500:.3g} mm first-cell half-height gives {Y_PLUS:.4g}")
        bottom, top = (row["mean"] for row in walls)
        assert bottom == pytest.approx(top, rel=TOL["symmetry"]), (
            f"{source['source']}: the two walls of a symmetric duct disagree, "
            f"{bottom:.4g} against {top:.4g}")


# -- seam 7: what the document says is what the run produced -------------------------


def test_the_figures_the_duct_preset_promises_are_drawn(journey):
    """`results.py --preset duct-flow` is the standard set. A preset that silently
    produced nothing used to read as "the case had none of it"."""
    out = journey["case"] / "results"
    pngs = [path for path in out.rglob("*.png") if path.stat().st_size > 1000]
    print("\n".join(f"{p.relative_to(out)}  {p.stat().st_size} B" for p in sorted(pngs)))
    assert len(pngs) >= 4, f"duct-flow drew {len(pngs)} figures with bytes in them"
    summary = out / "results.md"
    assert summary.is_file(), "results.py wrote no results.md"


def test_the_report_carries_the_numbers_the_solve_produced(journey):
    """The last seam, and the one that can be wrong while every stage above is
    right: the PDF's key numbers are supposed to come from this run's files. So the
    final residual the document prints is checked against the final residual
    `log_digest` reads out of the solver's own log -- the same number, not a
    plausible one."""
    record = journey["report"]
    assert journey["pdf"].is_file() and journey["pdf"].stat().st_size > 10_000, (
        f"no PDF, or an empty one: {journey['pdf']}")
    assert record["pages"] >= 3, record["page_sections"]
    assert record["is_study"], (
        "report.py did not recognise a study, so case_gen/results registered nothing")

    digest = load("log_digest").digest(journey["log"])
    shape = load("log_digest").residual_shape(digest["residuals"])
    rows = {row["label"]: row for row in record["numbers"]}
    print(json.dumps([{k: row[k] for k in ("label", "text", "state", "source")
                       if k in row} for row in record["numbers"]], indent=2))

    for field, last in shape["last"].items():
        label = f"final residual {field}"
        assert label in rows, f"the document has no {label!r}; it has {sorted(rows)}"
        printed = float(rows[label]["text"].split()[0])
        assert printed == pytest.approx(last, rel=0.02), (
            f"the document prints {label} = {printed:g}; the solver log says {last:g}")

    assert str(journey["case"].name) in record["case"], record["case"]
