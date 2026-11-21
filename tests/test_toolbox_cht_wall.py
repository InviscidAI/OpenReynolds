"""`cht_wall.py`: the two-region mesh, and that it claims only what it has.

The template builds the mesh a conjugate case cannot start without, for the family that
had no case at all. These tests check the written text -- which is the half that can be
checked without OpenFOAM, and the half the Slurm plan in `docs/slurm-backend-plan.md`
argues is worth checking on its own ("the script writer, as text").

Two things get special attention. The **span faces must be named**, because the reference
is 1D series conduction and a side that conducts makes it wrong while the run still
converges. And the template must say in words that **no solved result exists**, because a
case nobody has run that reads like one that works is the failure this directory exists
to prevent.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "openreynolds" / "toolbox" / "templates" / "cht_wall.py"


@pytest.fixture(scope="module")
def cht():
    spec = importlib.util.spec_from_file_location("template_cht_wall", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def written(tmp_path_factory, cht):
    """The case, written once into an empty directory."""
    case = tmp_path_factory.mktemp("cht")
    cht.write_case(case)
    return case


def test_the_arithmetic_agrees_with_the_grader_that_owns_it(cht):
    """`conjugate.py` derives and grades series conduction; this template recomputes it
    so a copy in an empty directory can print what it owes. Two implementations, so the
    one check worth having is that they agree -- a template quoting a different interface
    temperature from the grader is the drift this repository keeps finding."""
    spec = importlib.util.spec_from_file_location(
        "toolbox_conjugate", ROOT / "openreynolds" / "toolbox" / "conjugate.py")
    conjugate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = conjugate
    spec.loader.exec_module(conjugate)

    layers = ((cht.FLUID_THICK_M, cht.FLUID_K), (cht.SOLID_THICK_M, cht.SOLID_K))
    mine = cht.expected()
    theirs_flux = conjugate.flux(layers, cht.HOT_K, cht.COLD_K)
    theirs_interface = conjugate.interface_temperatures(
        layers, cht.HOT_K, cht.COLD_K)[1]
    assert abs(mine["flux_W_per_m2"] - theirs_flux) < 1e-9, (mine, theirs_flux)
    assert abs(mine["T_interface_K"] - theirs_interface) < 1e-9


def test_the_prandtl_number_is_inverted_from_the_conductivity(cht):
    """`thermophysicalProperties` has no kappa entry, so Pr is where the fluid's
    conductivity is set. If somebody edits FLUID_K, Pr has to move with it."""
    assert abs(cht.prandtl() - cht.FLUID_MU * cht.FLUID_CP / cht.FLUID_K) < 1e-12
    assert abs(cht.expected()["Pr"] - cht.prandtl()) < 1e-12
    assert 1.0 < cht.prandtl() < 100.0, "a liquid-like Pr, not a typo"


def test_both_cell_zones_are_named_in_the_blocks(written):
    """`splitMeshRegions -cellZones` reads the zone name out of the `hex` entry. Without
    both names there is one region and no interface, which is not a conjugate case."""
    text = (written / "system" / "blockMeshDict").read_text(encoding="utf-8")
    assert text.count(") fluid (") == 1, text
    assert text.count(") solid (") == 1, text


def test_the_interface_faces_are_in_no_patch(written):
    """The T8 artifact's finding: the shared face is named nowhere, so blockMesh merges
    it as internal and `splitMeshRegions` makes the mappedWall pair itself. A patch there
    would prevent the interface from existing at all."""
    text = (written / "system" / "blockMeshDict").read_text(encoding="utf-8")
    for forbidden in ("interface", "fluid_to_solid", "solid_to_fluid", "mergePatchPairs\n(\n    ("):
        assert forbidden not in text, f"{forbidden!r} must not be in the blockMeshDict"
    assert "mergePatchPairs ();" in text


def test_the_span_faces_are_named_walls_and_all_of_them(written, cht):
    """Physics, not cosmetics. 1D series conduction is the answer only if nothing leaves
    through the span, and an unnamed face falls into `defaultFaces`, whose type is a
    setting rather than a statement. Four faces per block, two blocks."""
    text = (written / "system" / "blockMeshDict").read_text(encoding="utf-8")
    start = text.index("    sides")
    block = text[start:text.index("}", start)]
    assert "type            wall;" in block, block
    faces = [line for line in block.splitlines() if line.strip().startswith("(")
             and line.strip() != "("]
    assert len(faces) == 8, f"expected 4 faces per block for 2 blocks, got {len(faces)}"


def test_gravity_is_written_and_is_zero(written):
    """The modelling choice the exact reference depends on. A fluid layer with a
    side-to-side temperature difference convects at any Rayleigh number, so under gravity
    there is no closed form to grade against -- and `constant/g` is where a reader sees
    that decision rather than having to infer it."""
    text = (written / "constant" / "g").read_text(encoding="utf-8")
    assert "value           (0 0 0);" in text, text


def test_both_regions_are_declared(written):
    """`regionProperties` is what tells chtMultiRegion* which region is which."""
    text = (written / "constant" / "regionProperties").read_text(encoding="utf-8")
    assert "fluid       (fluid)" in text
    assert "solid       (solid)" in text


def test_write_case_invents_no_per_region_properties(written):
    """`write_case` writes the mesh and the shared files only. The per-region properties
    are the image's own, copied in `main`, so they cannot appear here."""
    assert not (written / "constant" / "fluid").exists()
    assert not (written / "constant" / "solid").exists()


def test_the_properties_are_copied_off_the_image_rather_than_invented(cht):
    """The design that replaced guessing them. The ESI and Foundation builds spell these
    files differently and nothing here had run the solver to settle which, so they come
    from `$FOAM_TUTORIALS` -- `chem0d.py`'s pattern, for `chem0d.py`'s reason."""
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "does not invent the solver dictionaries" in text
    assert "No solved result exists in this repository yet" in text
    assert cht.TUTORIAL_SOURCE.startswith("heatTransfer/chtMultiRegionFoam/")
    # The file names are deliberately NOT enumerated: that is the thing not known.
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "without enumerating their names" in source


def test_a_missing_image_is_refused_with_the_path_it_wanted(cht, monkeypatch):
    """Off the image this cannot work, and it must say so rather than write a case with
    no properties that looks complete."""
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)
    with pytest.raises(SystemExit) as refused:
        cht.tutorials_root()
    assert "FOAM_TUTORIALS is not set" in str(refused.value)
    assert "does not invent them" in str(refused.value)
    monkeypatch.setenv("FOAM_TUTORIALS", "/nowhere/at/all")
    with pytest.raises(SystemExit) as missing:
        cht.tutorials_root()
    assert "not a directory" in str(missing.value)


def test_setting_a_property_that_is_not_there_is_refused(cht, tmp_path):
    """The failure this repository has already paid for: `t.replace()` with no count
    check. A property silently keeping the tutorial's value is a case solved for
    somebody else's fluid, with the run converging and the answer wrong."""
    good = tmp_path / "props"
    good.write_text("thermoType { type heRhoThermo; }" + chr(10)
                    + "Pr              0.7;" + chr(10),
                    encoding="utf-8")
    cht.set_entry(good, "Pr", "6.968333")
    assert "Pr              6.968333;" in good.read_text(encoding="utf-8")

    absent = tmp_path / "nopr"
    absent.write_text("thermoType { type heRhoThermo; }" + chr(10),
                      encoding="utf-8")
    with pytest.raises(SystemExit) as refused:
        cht.set_entry(absent, "Pr", "6.968333")
    assert "0 `Pr` entries" in str(refused.value)

    twice = tmp_path / "twopr"
    twice.write_text("Pr 0.7;" + chr(10) + "Pr 0.9;" + chr(10),
                     encoding="utf-8")
    with pytest.raises(SystemExit) as doubled:
        cht.set_entry(twice, "Pr", "6.968333")
    assert "2 `Pr` entries" in str(doubled.value)


def test_a_property_file_under_either_name_is_found_and_neither_is_guessed(cht, tmp_path):
    """ESI says `thermophysicalProperties`, the Foundation build says
    `physicalProperties`. The lookup takes whichever is there and refuses with a listing
    when neither is, rather than picking one and writing it."""
    where = tmp_path / "fluid"
    where.mkdir()
    (where / "physicalProperties").write_text("Pr 0.7;" + chr(10),
                                              encoding="utf-8")
    found = cht.find_file(where, "thermophysicalProperties", "physicalProperties")
    assert found.name == "physicalProperties"

    empty = tmp_path / "solid"
    empty.mkdir()
    with pytest.raises(SystemExit) as refused:
        cht.find_file(empty, "thermophysicalProperties", "physicalProperties")
    assert "does not guess property file names" in str(refused.value)


def test_no_mesh_writes_the_files_and_runs_no_openfoam():
    """The path a laptop takes. It must exit 0 and write the case, so the arithmetic and
    the dictionaries are reviewable with nothing installed."""
    import tempfile
    with tempfile.TemporaryDirectory() as where:
        done = subprocess.run([sys.executable, "-I", "-X", "utf8", str(TEMPLATE),
                               "--no-mesh"],
                              cwd=where, capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "stopping before blockMesh" in done.stdout
        assert "2944.78527607" in done.stdout, done.stdout
        assert "301.84049080" in done.stdout, done.stdout
        assert (Path(where) / "system" / "blockMeshDict").is_file()
