"""`case_gen.py` writing an LES or a DNS, which is not a different solver.

OpenFOAM runs LES with the same `pimpleFoam` a URANS case runs on; the only file that
says which is `constant/turbulenceProperties`. So nothing here tests a solver. What it
tests is the set of things that make an LES case wrong while leaving it entirely
runnable, which is the whole difficulty: an upwinded LES converges and is a RAS, a
steady LES converges and is nonsense, an instantaneous field reported as the answer is
a noise sample, and a subgrid model the installation does not have is a dictionary that
parses.

The mesh is the same `tests/data/cavity` the case_gen tests use, through the same two
helpers, because a second fixture is a second thing to keep in step. Its geometry is
real, so the first-cell heights and the cell size are measured rather than mocked --
the cavity is of course not a mesh anyone should run an LES on, and the point of the
y+ lines under test is that the script says so with numbers instead of assuming.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import pytest

from test_toolbox_case_gen import mesh_only, rename_patches

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def case_gen():
    spec = importlib.util.spec_from_file_location("case_gen", TOOLBOX / "case_gen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def box(tmp_path):
    """The cavity as a passage with a wall to take a boundary layer on."""
    case = mesh_only(tmp_path, "les")
    rename_patches(case, {
        "movingWall": ("inlet", "patch"),
        "fixedWalls": ("walls", "wall"),
        "frontAndBack": ("frontAndBack", "empty"),
    })
    return case


def les(case_gen, case: Path, *args: str) -> int:
    """A transient LES run of the kind the generator is meant to produce."""
    return case_gen.main([str(case), "--length", "0.1", "--speed", "2",
                          "--reynolds", "50000", "--study", "transient",
                          "--end-time", "4", *args])


def read(case: Path, relative: str) -> str:
    return (case / relative).read_text(encoding="utf-8")


# -- the dictionary that makes it an LES -------------------------------------------


def test_the_named_subgrid_model_and_delta_are_what_gets_written(case_gen, box):
    les(case_gen, box, "--turbulence", "WALE", "--les-delta", "vanDriest")
    text = read(box, "constant/turbulenceProperties")
    assert "simulationType  LES;" in text
    assert "LESModel        WALE;" in text
    assert "delta           vanDriest;" in text
    # vanDriest damps a geometric delta and reads that delta out of its own Coeffs
    # dictionary with `subDict`, not `optionalSubDict`: leaving the block out is a
    # missing-dictionary error at start-up rather than a default.
    assert "vanDriestCoeffs" in text
    assert "Aplus           26;" in text and "Cdelta          0.158;" in text
    assert re.search(r"vanDriestCoeffs\s*\n\s*\{[^}]*delta\s+cubeRootVol;", text)


def test_the_default_delta_is_stated_and_is_the_cheap_geometric_one(case_gen, box):
    les(case_gen, box, "--turbulence", "kEqn")
    assert "delta           cubeRootVol;" in read(box, "constant/turbulenceProperties")
    assert "deltaCoeff      1;" in read(box, "constant/turbulenceProperties")


def test_a_subgrid_model_that_does_not_exist_is_refused_rather_than_written(case_gen, box):
    """`Vreman`, `OneEqEddy`, `homogeneousDynSmagorinsky` are all real names of real
    models in some code or some other OpenFOAM lineage, and none of them is in ESI
    v2512. A dictionary naming one parses perfectly and the solver stops at start-up
    after the job has been queued and the mesh decomposed."""
    with pytest.raises(SystemExit) as raised:
        les(case_gen, box, "--turbulence", "Vreman")
    message = str(raised.value)
    assert "Vreman" in message
    assert "WALE" in message and "kEqn" in message
    assert not (box / "constant" / "turbulenceProperties").exists()


def test_a_delta_that_does_not_exist_is_refused_with_the_ones_that_do(case_gen, box):
    with pytest.raises(SystemExit) as raised:
        les(case_gen, box, "--turbulence", "WALE", "--les-delta", "vanDriestDelta")
    assert "vanDriestDelta" in str(raised.value)
    assert "cubeRootVol" in str(raised.value)


def test_an_iddes_model_is_refused_on_a_delta_it_cannot_cast(case_gen, box):
    """ESI's `SpalartAllmarasIDDES::setDelta()` tests `isA<IDDESDelta>` and exits with
    "The delta function must be set to a IDDESDelta -based model". The generator can
    see that pairing before the queue does."""
    with pytest.raises(SystemExit) as raised:
        les(case_gen, box, "--turbulence", "SpalartAllmarasIDDES",
            "--les-delta", "cubeRootVol")
    assert "IDDESDelta" in str(raised.value)
    # and without being told, it picks the one that model can actually run on
    assert les(case_gen, box, "--turbulence", "SpalartAllmarasIDDES") == 0
    assert "delta           IDDESDelta;" in read(box, "constant/turbulenceProperties")


def test_a_dns_says_laminar_and_says_why(case_gen, box):
    """OpenFOAM has no `simulationType DNS`. A direct simulation is the momentum
    equation with nothing added, which it spells `laminar` -- and a case file that says
    only `laminar` is indistinguishable from a Re 100 duct run on the wrong mesh."""
    les(case_gen, box, "--turbulence", "DNS")
    text = read(box, "constant/turbulenceProperties")
    assert "simulationType  laminar;" in text
    assert "DNS" in text and "no subgrid model" in text
    assert "LESModel" not in text


# -- LES is transient, always -------------------------------------------------------


def test_a_steady_les_is_refused_by_name_rather_than_written(case_gen, box):
    """It would run. `simpleFoam` drops the ddt term, so the solver itself averages
    away every eddy the subgrid model exists to be a subgrid to, and what lands on
    disk is a converged steady field with an LES dictionary beside it."""
    with pytest.raises(SystemExit) as raised:
        case_gen.main([str(box), "--length", "0.1", "--study", "steady",
                       "--turbulence", "WALE"])
    message = str(raised.value)
    assert "steady" in message and "WALE" in message
    assert "no steady LES" in message
    assert "--study transient" in message
    assert not (box / "system").exists(), "nothing was written"


# -- the numerics, which are the part usually got wrong ----------------------------


def ddt_and_div(text: str) -> tuple[str, str]:
    ddt = re.search(r"ddtSchemes\s*\n\{\s*\n\s*default\s+([^;]+);", text)
    div = re.search(r"div\(phi,U\)\s+([^;]+);", text)
    assert ddt and div, f"no schemes in\n{text}"
    return ddt.group(1).strip(), div.group(1).strip()


def test_the_les_convection_and_ddt_differ_from_the_ras_transient_case(case_gen, tmp_path):
    """The dissipation an upwind-biased scheme adds acts at the grid scale, which is
    exactly where the subgrid model acts, and it is larger -- so an LES on the RAS
    transient numerics is a slower RAS with a worse answer and a log that says nothing.
    Both schemes have to move, and `backward` is second order too, so moving only the
    ddt "to second order" would have moved nothing."""
    ras_ddt, ras_div = ddt_and_div(case_gen.fv_schemes("transient", "kOmegaSST"))
    les_ddt, les_div = ddt_and_div(case_gen.fv_schemes("transient", "WALE"))
    assert (ras_ddt, ras_div) == ("backward", "Gauss linear")
    assert les_ddt != ras_ddt and les_div != ras_div
    assert les_ddt == case_gen.LES_DDT and les_div == case_gen.LES_DIV_U
    # near-central, and nothing bounded or upwinded on the momentum equation
    assert "LUST" in les_div or "linear" == les_div.split()[-1]
    assert "linearUpwind" not in les_div.replace("LUST", "")
    assert "bounded" not in les_div


def test_the_scheme_change_is_stated_and_not_only_made(case_gen, box, capsys):
    les(case_gen, box, "--turbulence", "WALE", "--dry-run")
    printed = capsys.readouterr().out
    assert "schemes changed for LES" in printed
    assert case_gen.LES_DDT in printed and case_gen.LES_DIV_U in printed
    assert "backward" in printed and "Gauss linear" in printed, (
        "what it changed FROM is half of saying what it changed"
    )
    # the cost of a central scheme, as something to measure rather than obey
    assert "non-orthogonality" in printed and "skewness" in printed


# -- the fields the chosen model actually transports --------------------------------


@pytest.mark.parametrize("model,extra", [
    ("WALE", ()),               # algebraic: k from grad(U), nothing on disk but nut
    ("Smagorinsky", ()),
    ("kEqn", ("k",)),           # owns a k_ it reads MUST_READ
    ("dynamicKEqn", ("k",)),
    ("SpalartAllmarasDDES", ("nuTilda",)),
    ("kOmegaSSTDDES", ("k", "omega")),
])
def test_the_zero_directory_is_what_the_subgrid_model_reads(case_gen, tmp_path,
                                                            model, extra):
    """`nut` is always on disk -- every eddy-viscosity model reads it MUST_READ -- and
    the rest is per model. An algebraic model with no transported field at all was the
    one combination the generator could not write: `0/nut` lived inside the branch that
    needed a transported field to exist first, so `WALE` came out with no nut and
    stopped before the first step."""
    case = mesh_only(tmp_path, model)
    rename_patches(case, {"movingWall": ("inlet", "patch"),
                          "fixedWalls": ("walls", "wall"),
                          "frontAndBack": ("frontAndBack", "empty")})
    les(case_gen, case, "--turbulence", model)
    written = {path.name for path in (case / "0").iterdir()}
    assert written == {"U", "p", "nut", *extra}, f"0/ for {model}"
    # and the same table drives the schemes and the linear solvers, or one of the
    # three is describing a different case
    schemes = read(case, "system/fvSchemes")
    solution = read(case, "system/fvSolution")
    for field in extra:
        assert f"div(phi,{field})" in schemes
    if extra:
        # One field is named plainly, several become one regex -- fv_solution's own
        # convention. PIMPLE then asks for the matching `Final` by literal name and
        # stops on the first step without it.
        group = f'"({"|".join(extra)})"' if len(extra) > 1 else extra[0]
        final = group[:-1] + 'Final"' if group.endswith('"') else group + "Final"
        assert group in solution and final in solution
    for absent in {"k", "omega", "epsilon", "nuTilda"} - set(extra):
        assert f"div(phi,{absent})" not in schemes


def test_a_dns_transports_nothing_and_needs_no_nut(case_gen, box):
    les(case_gen, box, "--turbulence", "DNS")
    assert {path.name for path in (box / "0").iterdir()} == {"U", "p"}


# -- the answer is a mean -----------------------------------------------------------


def test_field_average_is_written_with_a_start_time(case_gen, box):
    les(case_gen, box, "--turbulence", "WALE", "--average-from", "1.5")
    text = read(box, "system/controlDict")
    assert "type            fieldAverage;" in text
    assert "timeStart       1.5;" in text
    assert "prime2Mean      on;" in text, (
        "the fluctuation is what an LES was run for and a mean field cannot give it back"
    )


def test_without_a_start_time_the_default_is_half_the_run_and_says_it_is_a_guess(
        case_gen, box, capsys):
    les(case_gen, box, "--turbulence", "WALE")
    assert "timeStart       2;" in read(box, "system/controlDict")
    printed = capsys.readouterr().out
    assert "mean from t = 2 s" in printed
    assert "spectra.py" in printed, "what decides when the flow settled"


def test_a_ras_case_gets_the_wall_fields_and_no_field_average(case_gen, tmp_path):
    """A mean is for a run whose answer is one. The forces and residual objects are
    unchanged by any of this.

    The claim this test makes about the rest of the block narrowed when
    `case_gen.wall_fields` landed. It used to assert `"functions" not in text` -- that
    this case gets no block at all -- and that was true by accident rather than by
    intent: `force_coeffs` returns nothing because no patch here is called `body`, and
    `solverInfo` happens to be bundled inside its entry list, so a bodyless case lost
    its residual logging too and the block came out empty. Nothing decided that a
    turbulent wall-bounded run should be bare.

    This fixture is the case the wall fields are most clearly *for*: a transient RAS
    run at Re 50,000 over a patch of type `wall`. y+ is the number that decides whether
    the wall treatment is valid at all, and `wall.py` exists because a generated O-grid
    landed at y+ about 1200 under a form written for 300. The `empty` pair is untouched
    -- both function objects evaluate on wall patches only -- so the case being one cell
    thick costs nothing here.

    The empty-block rule itself is still worth keeping and now lives where it is still
    exercised: `test_toolbox_case_gen.py`'s no-wall case, which genuinely has nothing to
    put in a block.
    """
    plain = mesh_only(tmp_path, "plain")
    rename_patches(plain, {"movingWall": ("inlet", "patch"), "fixedWalls": ("walls", "wall")})
    case_gen.main([str(plain), "--length", "0.1", "--study", "transient",
                   "--reynolds", "50000"])
    text = read(plain, "system/controlDict")
    assert "fieldAverage" not in text, "a mean is for a run whose answer is one"
    assert "wallShearStress" in text and "yPlus" in text
    assert "forceCoeffs" not in text, "no patch here is a body, so there is no force"

    body = mesh_only(tmp_path, "body")
    rename_patches(body, {"movingWall": ("inlet", "patch"), "fixedWalls": ("body", "wall")})
    les(case_gen, body, "--turbulence", "WALE", "--body", "body")
    text = read(body, "system/controlDict")
    assert "forceCoeffs" in text and "solverInfo" in text and "fieldAverage" in text


# -- the resolution requirement, stated rather than assumed met --------------------


def test_the_courant_limited_step_is_computed_from_the_mesh_and_printed(case_gen, box,
                                                                        capsys):
    les(case_gen, box, "--turbulence", "WALE", "--courant", "0.5")
    printed = capsys.readouterr().out
    cell = case_gen.read_mesh(box).smallest_cell
    expected = 0.5 * cell / 2.0
    assert re.search(rf"Co 0\.5 on the \w+ cell \({cell:.3g} m\) at 2 m/s", printed)
    assert f"{expected:.3g} s" in printed
    written = float(re.search(r"deltaT\s+([\d.eE+-]+);",
                              read(box, "system/controlDict")).group(1))
    assert written == pytest.approx(expected, rel=1e-3), (
        "the number printed is the number written"
    )
    assert "maxCo           0.5;" in read(box, "system/controlDict")
    assert "adjustTimeStep  yes;" in read(box, "system/controlDict")


def test_a_courant_number_above_one_is_said_out_loud(case_gen, box, capsys):
    """Not refused: `--courant 1.2` is a choice somebody may have reasons for. But
    above 1 the resolved scales are advected more than a cell a step, and those are
    precisely the scales the subgrid model is not modelling."""
    les(case_gen, box, "--turbulence", "WALE", "--courant", "1.2", "--dry-run")
    assert "is above 1" in capsys.readouterr().out


def test_the_y_plus_implication_is_measured_per_wall_and_points_at_wall_py(case_gen, box,
                                                                           capsys):
    """The generator cannot know the mesh is adequate. It can say what the mesh it was
    handed implies: the first-cell height off `constant/polyMesh` (`layer_report.py`,
    no solver, nothing written) through the flat-plate correlation `case_gen` already
    has. That is an a priori number on a measured height, and it is labelled as one --
    `wall.py` is the measurement, once there is a solution to measure."""
    les(case_gen, box, "--turbulence", "WALE", "--dry-run")
    printed = capsys.readouterr().out

    import sys
    sys.path.insert(0, str(TOOLBOX))
    import layer_report

    heights, _ = layer_report.first_cell_heights(layer_report.load(box), "walls")
    assert f"walls: first cell {float(heights.min()):.3g}" in printed
    assert "a priori" in printed and "flat plate" in printed
    assert "wall-resolved LES wants that y+ of order 1" in printed
    assert "wall-modelled LES wants the first cell in the log region" in printed
    assert "near-isotropic" in printed
    assert "wall.py" in printed, "the way to check the claim once there is a solution"


def test_a_mesh_whose_first_cells_cannot_be_read_says_so_instead(case_gen, box, capsys,
                                                                 monkeypatch):
    """Reporting, not solving: a polyMesh this cannot read costs a line of output. What
    it must not do is print a y+ off a height nobody measured."""
    monkeypatch.setattr(case_gen, "measured_first_cells", lambda case, plan: {})
    les(case_gen, box, "--turbulence", "WALE", "--dry-run")
    printed = capsys.readouterr().out
    assert "could not be read off constant/polyMesh" in printed
    assert "layer_report.py" in printed and "wall.py" in printed
    assert "a priori" not in printed


# -- the rest of the case is unchanged ---------------------------------------------


def test_an_les_still_gets_every_patch_in_every_field_and_the_empty_one_empty(case_gen,
                                                                             box):
    les(case_gen, box, "--turbulence", "kEqn")
    for name in ("U", "p", "k", "nut"):
        text = read(box, f"0/{name}")
        for patch in ("inlet", "walls", "frontAndBack"):
            assert f"    {patch}\n" in text, f"{patch} missing from 0/{name}"
        assert "type            empty;" in text.split("frontAndBack")[1][:200]


def test_auto_never_chooses_an_les(case_gen):
    """Which eddies a mesh resolves is not something a Reynolds number knows."""
    high = case_gen.derive_flow({"speed": 1.0, "reynolds": 1e8, "nu": None,
                                 "length": None}, 0.1)
    assert case_gen.turbulence_model({"turbulence": "auto"}, high)[0] == "kOmegaSST"


def test_every_model_the_flag_offers_is_one_the_tables_know(case_gen):
    """The flag's list is built from the tables rather than typed beside them: a name
    in one and not the other writes a dictionary that parses and a case that stops."""
    for model in case_gen.MODELS:
        if model in ("auto", "laminar", "DNS"):
            continue
        assert model in case_gen.TURBULENCE_FIELDS or model in case_gen.LES_FIELDS
    assert set(case_gen.LES_DELTAS) == set(case_gen.DELTA_COEFFS)


# -- what the mesh it was handed can actually resolve --------------------------------


def test_an_les_on_a_one_cell_thick_mesh_says_it_is_not_one(case_gen, box, capsys):
    """#182. The cavity is 2D -- one cell in z between two `empty` patches -- so
    d/dz = 0 by construction and the three-dimensional cascade a subgrid model stands
    for cannot occur in it. WALE still returns a nu_t off the in-plane strain rate, so
    the case computes a laminar 2D run with an extra viscosity in it, and the summary
    said nothing while printing "(2D, one cell thick)" four lines above. Said rather
    than refused: the mesh is the user's, and the dictionary is the one they asked
    for."""
    les(case_gen, box, "--turbulence", "WALE", "--dry-run")
    printed = capsys.readouterr().out
    assert "(2D, one cell thick)" in printed, "the fixture is the 2D mesh this is about"
    assert "ON A 2D MESH" in printed
    assert "d/dz = 0" in printed
    assert "not an LES" in printed
    assert "kOmegaSST" in printed, "and what the honest choice on this mesh would be"


def test_a_dns_on_a_one_cell_thick_mesh_says_so_in_its_own_words(case_gen, box, capsys):
    """The same mesh with no subgrid model: there is nothing to resolve in z either,
    and a DNS has no model to blame it on."""
    les(case_gen, box, "--turbulence", "DNS", "--dry-run")
    printed = capsys.readouterr().out
    assert "ON A 2D MESH" in printed
    assert "not a direct simulation" in printed
    assert "not an LES" not in printed, "this case has no subgrid model to talk about"


def test_a_three_dimensional_mesh_gets_no_such_note(case_gen, box, capsys, monkeypatch):
    """The note is about the mesh in hand, so it is keyed on the measurement and not on
    LES: the same mesh read as a volume must not carry it. (`tests/data/cavity` is the
    only mesh here and it is one cell thick, so what is swapped is the reading.)"""
    real = case_gen.read_mesh

    def as_a_volume(case):
        mesh = real(case)
        mesh.two_d = False
        return mesh

    monkeypatch.setattr(case_gen, "read_mesh", as_a_volume)
    les(case_gen, box, "--turbulence", "WALE", "--dry-run")
    printed = capsys.readouterr().out
    assert "(2D, one cell thick)" not in printed, "the reading was swapped"
    assert "ON A 2D MESH" not in printed
    assert "LES        WALE" in printed, "and the rest of the notes are still printed"


def test_the_dns_notes_are_about_a_dns_and_not_about_a_subgrid_model(case_gen, box,
                                                                    capsys):
    """#184. The header was right -- `DNS  no subgrid model, simulationType laminar` --
    and every line under it talked about LES on a case that has none: "schemes changed
    for LES", "exactly where the subgrid model acts", "an upwinded LES is a slower
    RAS", "wall-resolved LES wants that y+ of order 1"."""
    les(case_gen, box, "--turbulence", "DNS", "--dry-run")
    printed = capsys.readouterr().out
    assert "DNS        no subgrid model" in printed
    assert "schemes changed for DNS" in printed
    assert "schemes changed for LES" not in printed
    assert "the subgrid model acts" not in printed
    assert "an upwinded LES is a slower RAS" not in printed
    assert "upwinded DNS resolves less than its mesh does" in printed
    assert "wall-resolved LES wants" not in printed
    assert "a DNS resolves the wall or it is not a DNS" in printed
    # and an LES still says all of it, because for an LES all of it is true
    les(case_gen, box, "--turbulence", "WALE", "--dry-run")
    for_les = capsys.readouterr().out
    assert "schemes changed for LES" in for_les
    assert "the subgrid model acts" in for_les
    assert "wall-resolved LES wants that y+ of order 1" in for_les


def test_a_dns_says_how_far_short_of_resolving_itself_the_mesh_is(case_gen, box, capsys):
    """#184. The switch printed the cell count and the deltaT and nothing about whether
    this mesh resolves anything, which is the one statement that separates DNS the
    dictionary setting from DNS the claim. The scaling is the one the case file's own
    comment states: eta = L Re^(-3/4), so a resolved isotropic box of side L is
    (L/eta)^3 = Re^(9/4) cells. At Re 50000 that is 7.5e10 against this mesh's 400."""
    les(case_gen, box, "--turbulence", "DNS", "--dry-run")
    printed = capsys.readouterr().out
    assert "eta = L Re^(-3/4)" in printed
    assert "Re^(9/4)" in printed
    assert "short" in printed
    assert f"{50000 ** 2.25:.2g}" in printed, "the number is computed, not asserted"
    # The estimate is for three dimensions, which this mesh has one cell of.
    assert "that count is for three dimensions" in printed
    # And an LES, which has a model for what it does not resolve, does not claim it.
    les(case_gen, box, "--turbulence", "WALE", "--dry-run")
    assert "Re^(9/4)" not in capsys.readouterr().out


def test_the_step_count_the_run_is_comes_out_with_the_step(case_gen, box, capsys):
    """#184. deltaT and the cell count were both printed; the product of them with the
    end time -- the number of cell-updates this run is -- was not, and it is the one
    that says whether it finishes."""
    les(case_gen, box, "--turbulence", "WALE", "--end-time", "4", "--dry-run")
    printed = capsys.readouterr().out
    line, = [l for l in printed.splitlines() if "steps on" in l]
    step = float(re.search(r"implies deltaT ([\d.e+-]+) s", printed).group(1))
    steps = int(re.search(r"is ([\d,]+) steps", line).group(1).replace(",", ""))
    # The printed step is rounded to three figures, so this is the arithmetic and not
    # the rounding: 4 s of it, and the mesh it is run on.
    assert steps == pytest.approx(4 / step, rel=0.01), line
    assert f"on {case_gen.read_mesh(box).cells:,} cells" in line, line
    assert "advise.py" in line, "where a wall-clock comes from, which needs the machine"


def test_the_a_priori_y_plus_is_printed_to_two_significant_figures(case_gen, box, capsys):
    """#183. `:.0f` rounded away the whole quantity: a real wall at y+ 0.0325 printed
    as `0` and walls at 0.82 printed as `1`, three lines above "wall-resolved LES wants
    that y+ of order 1". a421575 corrected the per-wall lines in `les_notes` and missed
    this one in `turbulence_notes`, which is the line every RAS case prints.

    On this mesh at Re 120 the correlation gives y+ 0.85 -- 15% below the order-1 wall
    it would be read against, and `:.0f` printed it as exactly 1.
    """
    case_gen.main([str(box), "--length", "0.1", "--speed", "2", "--reynolds", "120",
                   "--turbulence", "kOmegaSST", "--dry-run"])
    printed = capsys.readouterr().out
    line, = [l for l in printed.splitlines() if "near wall" in l]
    first = float(re.search(r"first cell ([\d.e+-]+) m", line).group(1))
    flow = case_gen.derive_flow({"speed": 2.0, "reynolds": 120.0, "nu": None,
                                 "length": None}, 0.1)
    y_plus = case_gen.estimate_y_plus(first, flow)
    assert y_plus < 1 and f"{y_plus:.0f}" == "1", f"the defect's own shape: {y_plus}"
    assert f"y+ ~ {y_plus:.2g}" in line, line
