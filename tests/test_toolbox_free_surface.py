"""`templates/free_surface.py`: the first template that writes a case meant to solve.

`test_toolbox_templates.py` covers what every template owes -- it parses, it marks its
two numbers, it is in the index, it imports nothing the instance lacks -- by reading the
source with `ast` rather than importing it, because gmsh, `gmshToFoam` and `checkMesh`
are not promised here. This file asks the harder question that only applies to this one:
whether the case it writes is a *correct* two-phase case. That cannot be read off the
source, so the module is loaded by path and its dictionary writers are called against a
`tmp_path`. Loading it is safe for the same reason the template works at all: `import
gmsh` lives inside `build_mesh`, so nothing at module scope needs the mesher.

The failures these guard against are the ones that leave `interFoam` running happily to
the end: `maxCo` with no `maxAlphaCo`, a `fixedValue` on the open boundary, a uniform
`alpha` start, a `0/p` fighting the `p_rgh` the solver derives it from, air entered at
rho = 1. None of them is visible in a log.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"
TEMPLATE = TOOLBOX / "templates" / "free_surface.py"
NOTE = TOOLBOX / "notes" / "free-surface-field-notes.md"

HAS_GMSH = importlib.util.find_spec("gmsh") is not None
needs_gmsh = pytest.mark.skipif(not HAS_GMSH, reason="gmsh module not on this machine")


def load(tmp_path: Path, name: str = "fs0", **edits: float):
    """The template as an imported module, optionally with its two numbers edited.

    "Edit the two numbers and re-run" means re-running the file, not reassigning a
    global: `CELL_SIZE_M` and `THICK_M` are derived from `TANK_WIDTH_M` at module
    scope, so a test that only rebound `TANK_WIDTH_M` would be testing something no
    user does. The source is rewritten and loaded afresh instead.
    """
    source = TEMPLATE.read_text(encoding="utf-8")
    for const, value in edits.items():
        source, n = re.subn(rf"^{const} = \S+", f"{const} = {value!r}", source, flags=re.M)
        assert n == 1, f"{const} is not a single editable module constant"
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def case(tmp_path: Path, name: str = "fs0", **edits: float) -> tuple[Path, object]:
    """The case written to disk, and the module that wrote it."""
    module = load(tmp_path, name=name, **edits)
    out = tmp_path / name
    out.mkdir()
    module.write_case(out)
    return out, module


def entry(text: str, key: str) -> str:
    """The value of a top-level `key value;` entry, with comments stripped."""
    match = re.search(rf"^\s*{re.escape(key)}\s+([^;/]+?)\s*(?://.*)?;", text, re.M)
    assert match, f"no entry {key!r} in:\n{text}"
    return match.group(1).strip()


def patch_block(text: str, patch: str) -> str:
    """One `boundaryField` entry's braces, which is where a wrong condition hides."""
    match = re.search(rf"^    {patch}\n    \{{\n(.*?)^    \}}", text, re.M | re.S)
    assert match, f"no boundary entry for {patch!r} in:\n{text}"
    return match.group(1)


# -- every file interFoam reads, and nothing it does not -----------------------


def test_the_case_carries_every_file_interfoam_needs(tmp_path):
    """A missing dictionary is the one VOF failure that *is* loud, and it costs a
    hosted launch to find out. The list is the solver's, not a preference."""
    out, _ = case(tmp_path)
    written = {str(p.relative_to(out)).replace("\\", "/")
               for p in out.rglob("*") if p.is_file()}
    for required in (
        "0/alpha.water", "0/U", "0/p_rgh",
        "constant/g", "constant/transportProperties", "constant/turbulenceProperties",
        "system/controlDict", "system/fvSchemes", "system/fvSolution",
        "system/setFieldsDict",
    ):
        assert required in written, f"{required} is missing from the case"
    # and the pristine copy setFields would otherwise eat
    assert "0/alpha.water.orig" in written


def test_p_rgh_is_the_solved_pressure_and_there_is_no_bare_p(tmp_path):
    """`interFoam` solves p - rho*g*h and derives `p` from it. A `0/p` is an initial
    condition for a field about to be overwritten, and it reads as if `p` were what the
    case controls."""
    out, _ = case(tmp_path)
    assert not (out / "0" / "p").exists(), "a bare 0/p fights the p that p_rgh derives"
    p_rgh = (out / "0" / "p_rgh").read_text(encoding="utf-8")
    assert "object      p_rgh" in p_rgh
    assert entry(p_rgh, "dimensions") == "[1 -1 -2 0 0 0 0]", "p_rgh is a pressure"
    # a hydrostatic column at rest has p_rgh flat, not increasing with depth
    assert entry(p_rgh, "internalField") == "uniform 0"

    solution = (out / "system" / "fvSolution").read_text(encoding="utf-8")
    assert re.search(r"^    p_rgh\n", solution, re.M), "no p_rgh solver"
    assert re.search(r"^    p_rghFinal\n", solution, re.M), "no final-corrector solver"
    assert not re.search(r"^    p\n", solution, re.M), "p is derived, not solved"


# -- the interface itself ------------------------------------------------------


def test_alpha_starts_as_geometry_and_setfields_puts_it_where_the_numbers_say(tmp_path):
    """`internalField uniform 0.5` is not a free surface, it is a fog of mean density,
    and it solves. The initial field has to come from `setFields` and a region stated
    in metres, so it follows the two numbers and survives a change of mesh."""
    out, module = case(tmp_path)
    alpha = (out / "0" / "alpha.water").read_text(encoding="utf-8")
    # the uniform value is the *background* phase, and the only safe one is air
    assert entry(alpha, "internalField") == "uniform 0", (
        "a uniform alpha other than the background phase is a fog, not an interface"
    )
    assert entry(alpha, "dimensions") == "[0 0 0 0 0 0 0]"

    fields = (out / "system" / "setFieldsDict").read_text(encoding="utf-8")
    assert "volScalarFieldValue alpha.water 0" in fields, "no air background"
    assert "boxToCell" in fields, "the water region is not stated as geometry"
    assert "volScalarFieldValue alpha.water 1" in fields, "no water in the region"
    assert "cellSet" not in fields and "setToCell" not in fields, (
        "a cell list is a statement about one mesh, not about the water column"
    )

    box = re.search(r"box \(([^)]+)\) \(([^)]+)\)", fields)
    assert box, "no box corners in setFieldsDict"
    lo = [float(v) for v in box.group(1).split()]
    hi = [float(v) for v in box.group(2).split()]
    expect_lo, expect_hi = module.water_box()
    assert lo == pytest.approx(list(expect_lo))
    assert hi == pytest.approx(list(expect_hi))
    # the column is where the parameters put it: the corner at the origin, as wide as
    # COLUMN_WIDTH_M and COLUMN_ASPECT times as tall
    assert (lo[0], lo[1]) == (0.0, 0.0)
    assert hi[0] == pytest.approx(module.COLUMN_WIDTH_M)
    assert hi[1] == pytest.approx(module.COLUMN_ASPECT * module.COLUMN_WIDTH_M)
    # and it is inside the tank rather than overflowing it
    (_, _), (tank_x, tank_y) = module.tank_box()
    assert hi[0] < tank_x and hi[1] < tank_y
    # the z span is padded past the single cell, because boxToCell selects on centres
    assert lo[2] < 0.0 < module.THICK_M < hi[2]


def test_the_alpha_equation_carries_mules_and_interface_compression(tmp_path):
    """Bounded is not the same as sharp. Without the compression term alpha stays in
    [0, 1] to eight figures while the interface widens monotonically into a gradient."""
    out, _ = case(tmp_path)
    solution = (out / "system" / "fvSolution").read_text(encoding="utf-8")
    assert int(entry(solution, "nAlphaCorr")) >= 1
    assert int(entry(solution, "nAlphaSubCycles")) >= 1
    assert float(entry(solution, "cAlpha")) > 0, "cAlpha 0 is compression switched off"
    assert entry(solution, "MULESCorr") == "yes", (
        "maxAlphaCo 0.5 is outside explicit MULES' range"
    )

    schemes = (out / "system" / "fvSchemes").read_text(encoding="utf-8")
    assert "interfaceCompression" in schemes
    assert "vanLeer" in schemes
    advection = [line for line in schemes.splitlines() if "phi,alpha" in line]
    assert advection, "alpha has no advection scheme of its own"
    for line in advection:
        assert "interfaceCompression" in line and "vanLeer" in line, (
            f"alpha advected on {line.strip()!r} -- the interface will diffuse away"
        )


# -- the two phases, with their own numbers ------------------------------------


def test_two_phases_with_distinct_rho_and_nu_and_a_sigma(tmp_path):
    """A VOF answer is a density-ratio answer. Air left at the tutorials' round rho = 1
    is 20 percent off against air at 20 C and the solve never mentions it."""
    out, _ = case(tmp_path)
    transport = (out / "constant" / "transportProperties").read_text(encoding="utf-8")
    assert entry(transport, "phases") == "(water air)"

    numbers = {}
    for phase in ("water", "air"):
        block = re.search(rf"^{phase}\n\{{\n(.*?)^\}}", transport, re.M | re.S)
        assert block, f"no {phase} sub-dictionary"
        numbers[phase] = (float(entry(block.group(1), "rho")),
                          float(entry(block.group(1), "nu")))

    (rho_w, nu_w), (rho_a, nu_a) = numbers["water"], numbers["air"]
    assert rho_w != rho_a and nu_w != nu_a, "two phases sharing one set of properties"
    assert rho_w / rho_a > 100, "a water/air density ratio is of order 1000, not 1"
    # water and air at a named temperature, not magic constants: 20 C, 1 atm
    assert rho_w == pytest.approx(998.2, rel=1e-3)
    assert nu_w == pytest.approx(1.004e-06, rel=1e-2)
    assert rho_a == pytest.approx(1.204, rel=1e-3), "air at 20 C, not the round 1"
    assert nu_a == pytest.approx(1.516e-05, rel=1e-2)
    assert float(entry(transport, "sigma")) == pytest.approx(0.0728, rel=1e-2)
    # the source of every one of them is stated where the number is
    assert "20 C" in transport and "kg/m^3" in transport and "m^2/s" in transport
    assert "N/m" in transport


def test_gravity_is_present_and_points_at_the_floor(tmp_path):
    """A `g` pointing +y runs, drains the column up through the atmosphere patch, and
    reports nothing wrong."""
    out, module = case(tmp_path)
    g = (out / "constant" / "g").read_text(encoding="utf-8")
    assert entry(g, "dimensions") == "[0 1 -2 0 0 0 0]"
    vector = [float(v) for v in entry(g, "value").strip("()").split()]
    assert vector[0] == 0.0 and vector[2] == 0.0
    assert vector[1] == pytest.approx(-9.81, rel=1e-3), "gravity is -y here"
    # -y is the floor: the tank is cornered at the origin and open at the top
    (_, y0), (_, y1) = module.tank_box()
    assert y0 < y1


# -- the time step, and the writes ---------------------------------------------


def test_the_interface_courant_limit_is_set_as_well_as_the_bulk_one(tmp_path):
    """The quiet failure this template exists for. `maxCo` is set by the fastest cell
    anywhere -- thin, fast air -- while the surface moves at the speed of the water, so
    a step inside `maxCo` can still smear the interface over ten cells. Both limits, or
    the case looks fine and resolves nothing."""
    out, _ = case(tmp_path)
    control = (out / "system" / "controlDict").read_text(encoding="utf-8")
    assert entry(control, "adjustTimeStep") in ("yes", "on", "true")
    max_co = float(entry(control, "maxCo"))
    max_alpha_co = float(entry(control, "maxAlphaCo"))
    assert 0 < max_co <= 1 and 0 < max_alpha_co <= 1
    assert float(entry(control, "maxDeltaT")) > 0, "nothing bounds the step at rest"
    assert entry(control, "application") == "interFoam"


def test_writes_land_on_round_times_and_an_interrupted_solve_resumes(tmp_path):
    """`adjustTimeStep` without `adjustableRunTime` writes wherever the step landed,
    and a front position at t = 0.0937 s cannot be put beside a published one at
    t = 0.1 s. `startFrom latestTime` is the 24 h container lifetime -- see "Long
    solves" in the field notes."""
    out, _ = case(tmp_path)
    control = (out / "system" / "controlDict").read_text(encoding="utf-8")
    assert entry(control, "writeControl") == "adjustableRunTime"
    assert entry(control, "startFrom") == "latestTime"
    interval = float(entry(control, "writeInterval"))
    end_time = float(entry(control, "endTime"))
    assert 0 < interval < end_time, "the writes have to fall inside the run"


# -- the open boundary ---------------------------------------------------------


def test_the_open_boundary_is_an_inlet_outlet_pair_not_a_fixed_value(tmp_path):
    """Behind a collapsing column the air has to come back in. A `fixedValue` lid is a
    sealed box with a pressure condition painted on it; a `fixedValue` p_rgh holds the
    surface flat as it crosses, which is the measurement the case exists to make."""
    out, _ = case(tmp_path)
    p_rgh = patch_block((out / "0" / "p_rgh").read_text(encoding="utf-8"), "atmosphere")
    assert "totalPressure" in p_rgh
    assert "fixedValue" not in p_rgh, "a fixedValue lid will not let the surface move"
    assert "p0" in p_rgh, "totalPressure needs its reference"

    u = patch_block((out / "0" / "U").read_text(encoding="utf-8"), "atmosphere")
    assert "pressureInletOutletVelocity" in u
    assert "fixedValue" not in u

    alpha = patch_block((out / "0" / "alpha.water").read_text(encoding="utf-8"),
                        "atmosphere")
    assert "inletOutlet" in alpha, "zeroGradient alpha feeds water back in through the lid"
    assert "inletValue      uniform 0" in alpha, "the returning fluid is air"


def test_the_walls_hold_flux_rather_than_a_zero_gradient(tmp_path):
    """Under a buoyancy-driven field the wall-normal pressure gradient is not zero, so
    a `zeroGradient` wall generates or absorbs a little flow all run."""
    out, _ = case(tmp_path)
    p_rgh = patch_block((out / "0" / "p_rgh").read_text(encoding="utf-8"), "walls")
    assert "fixedFluxPressure" in p_rgh
    u = patch_block((out / "0" / "U").read_text(encoding="utf-8"), "walls")
    assert "noSlip" in u
    # and the case stays 2D: the z faces are empty, not two more open sides
    for field in ("alpha.water", "U", "p_rgh"):
        block = patch_block((out / "0" / field).read_text(encoding="utf-8"),
                            "frontAndBack")
        assert "empty" in block, f"{field} leaves the z faces open"


# -- the two marked numbers actually do something ------------------------------


def test_editing_the_two_numbers_moves_the_geometry(tmp_path):
    """"Copy it in, edit the two numbers, run it" is the contract. A number that the
    geometry does not follow is decoration."""
    small, mod_small = case(tmp_path, name="small")
    wide, mod_wide = case(tmp_path, name="wide", TANK_WIDTH_M=2.0)
    fat, mod_fat = case(tmp_path, name="fat", COLUMN_WIDTH_M=0.3)

    # TANK_WIDTH_M moves the tank, and the cell size and extrude depth with it
    assert mod_wide.tank_box() != mod_small.tank_box()
    assert mod_wide.tank_box()[1] == (2.0, 2.0)
    assert mod_wide.CELL_SIZE_M > mod_small.CELL_SIZE_M
    assert mod_wide.THICK_M == mod_wide.CELL_SIZE_M, "still one cell thick"

    # COLUMN_WIDTH_M moves the water, and the dict on disk follows
    assert mod_fat.water_box() != mod_small.water_box()
    assert mod_fat.water_box()[1][0] == pytest.approx(0.3)
    assert mod_fat.water_box()[1][1] == pytest.approx(0.6), "twice as tall as it is wide"
    assert ((small / "system" / "setFieldsDict").read_text(encoding="utf-8")
            != (fat / "system" / "setFieldsDict").read_text(encoding="utf-8")), (
        "the water column is written into setFieldsDict, so editing it changes the dict"
    )
    # the tank number reaches the dict too, through the one-cell z padding
    assert ((small / "system" / "setFieldsDict").read_text(encoding="utf-8")
            != (wide / "system" / "setFieldsDict").read_text(encoding="utf-8"))


@needs_gmsh
def test_the_mesh_is_the_tank_the_numbers_asked_for(tmp_path):
    """The only test here that needs the mesher, so it is guarded like
    `test_cad_wiring.py` does. Everything above reads dictionaries; this one checks the
    geometry is not merely written down but built -- a 2D tank, one cell thick, with the
    three physical groups the retyping afterwards addresses by name."""
    import gmsh

    module = load(tmp_path, name="meshed", TANK_WIDTH_M=1.0)
    msh = tmp_path / "meshed.msh"
    module.build_mesh(msh)
    assert msh.exists() and msh.stat().st_size > 0

    gmsh.initialize()
    try:
        gmsh.open(str(msh))
        names = {gmsh.model.getPhysicalName(dim, tag)
                 for dim, tag in gmsh.model.getPhysicalGroups()}
        assert {"walls", "atmosphere", "frontAndBack", "internal"} <= names
        nodes = gmsh.model.mesh.getNodes()[1]
        xs, ys, zs = nodes[0::3], nodes[1::3], nodes[2::3]
        assert min(xs) == pytest.approx(0.0) and max(xs) == pytest.approx(1.0)
        assert min(ys) == pytest.approx(0.0) and max(ys) == pytest.approx(1.0)
        # one cell in z, which is what makes the empty z faces legitimate
        assert sorted(set(round(z, 9) for z in zs)) == [
            pytest.approx(0.0), pytest.approx(module.THICK_M)]
    finally:
        gmsh.finalize()


# -- the template stays inside the image it runs on ----------------------------


def test_the_template_imports_only_the_standard_library_and_gmsh(tmp_path):
    """The same allowlist `test_toolbox_templates.py` applies to every template, asserted
    here as well because this is the file most tempted to reach outside it: a VOF case
    wants numpy for an interface position and pyvista for the picture, and both are on
    the instance -- just not inside a template, which is copied into a case directory
    and has to run as a single file."""
    allowed = set(sys.stdlib_module_names) | {"gmsh", "__future__"}
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        for root in roots:
            assert root in allowed, f"free_surface.py imports {root}"
    # gmsh is imported inside build_mesh, not at module scope, which is what lets this
    # file be loaded and its dictionaries checked on a machine with no mesher
    assert "import gmsh" not in TEMPLATE.read_text(encoding="utf-8").split(
        "def build_mesh", 1)[0]


def test_the_template_says_its_case_is_meant_to_solve(tmp_path):
    """The other two templates write stubs they expect to be overwritten. This one does
    not, and a reader who assumes otherwise will overwrite a working case."""
    doc = ast.get_docstring(ast.parse(TEMPLATE.read_text(encoding="utf-8")))
    assert "stub" in doc, "the docstring does not distinguish itself from the stub ones"
    assert "solved" in doc or "solve" in doc
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    row = next(line for line in index.splitlines()
               if "templates/free_surface.py" in line)
    assert "stub" in row and "solve" in row, (
        "the index row does not say this case is meant to be solved"
    )


# -- the field note ------------------------------------------------------------


def test_the_note_carries_what_a_two_phase_run_costs_to_learn():
    """Each of these is a failure that leaves a finished run and a plausible video:
    the interface Courant limit, p_rgh against p, an open boundary that cannot admit
    flow, a uniform alpha start, and a smeared interface in a picture."""
    note = NOTE.read_text(encoding="utf-8")
    for fact in ("maxAlphaCo", "maxCo", "MULESCorr",
                 "p_rgh", "fixedFluxPressure",
                 "totalPressure", "pressureInletOutletVelocity", "inletOutlet",
                 "setFields", "boxToCell", "uniform 0.5",
                 "cAlpha", "Martin & Moyce"):
        assert fact in note, f"the free-surface note does not mention {fact}"
    # how to tell a smeared interface from a resolved one, in the picture
    picture = note.split("## Telling a smeared interface", 1)
    assert len(picture) == 2, "the note does not say how to read an alpha picture"
    section = picture[1].split("\n## ", 1)[0]
    assert "one to three cells" in section, "no count that makes the picture readable"
    assert "0.5" in section and "mesh" in section.lower()


def test_the_note_states_facts_rather_than_a_procedure():
    """The notes are offered like everything else in here. A section that told a session
    the order to do things in would be the workflow injection the design exists to
    avoid. Same assertion as the moving-mesh note's in `test_toolbox.py`."""
    note = " ".join(NOTE.read_text(encoding="utf-8").lower().split())
    for imperative in ("you must", "always run", "before you", "step 1", "first,",
                       "you should", "make sure"):
        assert imperative not in note
