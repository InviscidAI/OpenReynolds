"""site_abl.py -- held to the solver's own arithmetic, and then to the solver.

The profiles here are not this script's invention: OpenFOAM computes them, in
`src/atmosphericModels/derivedFvPatchFields/atmBoundaryLayer/atmBoundaryLayer.C`,
and all this script writes is the dictionary those conditions read. So the tests
come in three layers:

  1. **the arithmetic**, against expressions written out longhand in this file from
     the C++ rather than by calling the thing under test;
  2. **the dictionary**, against the shape OpenFOAM's own `turbineSiting` tutorial
     uses -- an `include/ABLConditions` with six mandatory entries and a patch entry
     that names the type and includes it;
  3. **the solver**, where there is one: a graded column meshed with `blockMesh`,
     one `simpleFoam` iteration, and the inlet values the solver wrote read back out
     of `1/U`, `1/k` and `1/epsilon` and compared with this script's table. That
     last layer is skipped on a machine with no OpenFOAM and runs in the workspace
     image, which is the only place the real v2512 is.

The one thing layer 3 caught that layers 1 and 2 could not: `z` is measured from the
minimum of the inlet patch's **points** (`boundBox(pp.localPoints()).min()`), not
from its lowest face centre. Comparing against the lowest face centre made the
bottom face disagree by a factor of infinity and every face above it by a few
percent, which looks like a tolerance problem and is not.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def abl():
    return load("site_abl")


# -- layer 1: the arithmetic -------------------------------------------------------
#
# Transcribed from atmBoundaryLayer.C by hand, so a test passing means two
# independent transcriptions agree and not that one function equals itself.


KAPPA, CMU = 0.41, 0.09


def reference(z: float, uref: float, zref: float, z0: float, d: float = 0.0,
              c1: float = 0.0, c2: float = 1.0) -> dict[str, float]:
    us = KAPPA * uref / math.log((zref + z0) / z0)          # Ustar(), line 179
    ze = z - d + z0                                          # the argument, lines 223-266
    shape = math.sqrt(c1 * math.log(ze / z0) + c2)
    return {
        "U": (us / KAPPA) * math.log(ze / z0),               # U(), line 223
        "k": us ** 2 / math.sqrt(CMU) * shape,               # k(), lines 239-240
        "epsilon": us ** 3 / (KAPPA * ze) * shape,           # epsilon(), lines 253-254
        "omega": us / (KAPPA * math.sqrt(CMU) * ze),         # omega(), line 266
    }


@pytest.mark.parametrize("z0", [0.0002, 0.03, 0.5, 2.0])
@pytest.mark.parametrize("z", [0.05, 1.0, 10.0, 97.0, 300.0])
def test_the_profile_is_the_solver_expression(abl, z, z0):
    want = reference(z, 10.0, 10.0, z0)
    got = abl.profile(z, 10.0, 10.0, z0)
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=1e-12), key


def test_ustar_is_the_documented_closed_form(abl):
    assert abl.ustar(10.0, 10.0, 0.5) == pytest.approx(
        0.41 * 10.0 / math.log(10.5 / 0.5), rel=1e-12)


def test_the_speed_at_the_reference_height_is_the_reference_speed(abl):
    """The one number the whole thing is anchored to. With d = 0 it is exact."""
    for uref, zref, z0 in ((10.0, 10.0, 0.5), (4.851, 10.0, 2.0), (7.2, 80.0, 0.03)):
        assert abl.profile(zref, uref, zref, z0)["U"] == pytest.approx(uref, rel=1e-12)


def test_with_a_displacement_height_uref_lands_at_zref_plus_d(abl):
    """`Ustar()` has no `d` in it, so `Uref` is the speed where z - d + z0 = Zref + z0,
    which is z = Zref + d and not z = Zref. A case that assumed otherwise would be
    running a different wind speed than it said it was."""
    uref, zref, z0, d = 10.0, 10.0, 0.5, 4.0
    assert abl.profile(zref + d, uref, zref, z0, d)["U"] == pytest.approx(uref, rel=1e-12)
    assert abl.profile(zref, uref, zref, z0, d)["U"] < uref


def test_k_does_not_vary_with_height_in_the_richards_hoxey_case(abl):
    """C1 = 0 and C2 = 1 is what the defaults are, and what it means is a constant k."""
    values = [abl.profile(z, 10.0, 10.0, 0.5)["k"] for z in (0.1, 1.0, 10.0, 200.0)]
    assert max(values) == pytest.approx(min(values), rel=1e-12)
    assert values[0] == pytest.approx(abl.ustar(10.0, 10.0, 0.5) ** 2 / math.sqrt(CMU),
                                      rel=1e-12)


def test_k_varies_once_c1_is_not_zero(abl):
    low = abl.profile(1.0, 10.0, 10.0, 0.5, c1=-0.11, c2=1.21)["k"]
    high = abl.profile(100.0, 10.0, 10.0, 0.5, c1=-0.11, c2=1.21)["k"]
    assert high < low  # a negative C1 makes k fall with height, as YGCJ fit it


def test_a_height_the_log_cannot_take_is_refused_rather_than_nan(abl):
    with pytest.raises(abl.Refused, match="not defined"):
        abl.profile(1.0, 10.0, 10.0, 0.5, d=5.0)


def test_the_height_ladder_contains_the_reference_height(abl):
    """A table that does not sample Zref cannot show the one value that was set."""
    ladder = abl.heights(10.0, 200.0)
    assert 10.0 in ladder
    assert ladder == sorted(ladder)
    assert ladder[0] < 0.1 and ladder[-1] == 200.0


def test_the_ladder_starts_above_the_displacement_height(abl):
    """It used to start at 0.05 m whatever `d` was, so the first realistic
    displacement height it met -- 15 m for a city centre -- put every low sample
    under the profile's floor and the whole table refused instead of printing."""
    ladder = abl.heights(10.0, 300.0, d=15.0)
    assert min(ladder) > 15.0
    assert 25.0 in ladder  # Zref + d, where Uref actually occurs
    rows = abl.table(inputs_for(abl, z0=2.0, d=15.0, uref=4.851), 300.0)
    assert rows and rows[0]["U"] > 0
    at_reference = [r for r in rows if r["z"] == 25.0]
    assert at_reference and at_reference[0]["U"] == pytest.approx(4.851, rel=1e-12)


# -- the roughness table and the refusals ------------------------------------------


def test_there_is_no_default_roughness(abl):
    result = abl.main(["--uref", "10", "--zref", "10"])
    assert result == 2


def test_the_refusal_prints_the_whole_davenport_table(abl, capsys):
    abl.main(["--uref", "10", "--zref", "10"])
    text = capsys.readouterr().err
    for key in abl.ROUGHNESS:
        assert key in text
    assert "sea" in text and "0.0002" in text and "chaotic" in text


def test_a_class_name_sets_z0_and_says_which_class_it_was(abl, capsys):
    assert abl.main(["--uref", "10", "--zref", "10", "--roughness", "open"]) == 0
    out = capsys.readouterr().out
    assert "0.03" in out and "--roughness open" in out


def test_an_unknown_class_lists_the_ones_there_are(abl, capsys):
    assert abl.main(["--uref", "10", "--zref", "10", "--roughness", "urban"]) == 2
    assert "roughly-open" in capsys.readouterr().err


def test_zero_roughness_is_refused_because_the_log_divides_by_it(abl, capsys):
    assert abl.main(["--uref", "10", "--zref", "10", "--z0", "0"]) == 2
    assert "undefined at zero roughness" in capsys.readouterr().err


def test_a_negative_reference_height_is_refused_before_the_solver_aborts(abl, capsys):
    """atmBoundaryLayer::Ustar calls FatalErrorInFunction on a negative Zref. Better
    to say so here than to find out from a solver that has already been queued."""
    assert abl.main(["--uref", "10", "--zref", "-1", "--z0", "0.1"]) == 2
    assert "Ustar" in capsys.readouterr().err


def test_a_zero_reference_height_is_refused_rather_than_dividing_by_zero(abl, capsys):
    """`atmBoundaryLayer.H`'s own Usage example shows `Zref 0.0;` and the class only
    guards `Zref < 0`, but at zero the denominator is ln(1) and u* divides by it.
    Reaching that as a ZeroDivisionError, or handing the solver an infinity, are both
    worse than saying so."""
    assert abl.main(["--uref", "10", "--zref", "0", "--z0", "0.1"]) == 2
    assert "ln(1) = 0" in capsys.readouterr().err


def test_no_reference_wind_at_all_names_both_ways_to_supply_one(abl, capsys):
    assert abl.main(["--z0", "0.1"]) == 2
    text = capsys.readouterr().err
    assert "--site" in text and "--uref" in text


# -- layer 2: the dictionary -------------------------------------------------------


def inputs_for(abl, **kwargs):
    import argparse
    defaults = dict(site="", statistic="mean", sector=None, uref=10.0, zref=10.0,
                    z0=0.5, roughness="", d=0.0, flow_dir="", z_dir="",
                    kappa=abl.KAPPA, cmu=abl.CMU, c1=0.0, c2=1.0)
    defaults.update(kwargs)
    return abl.resolve(argparse.Namespace(**defaults))


def test_the_conditions_file_carries_the_six_mandatory_entries(abl):
    """The six `atmBoundaryLayer.H` marks "yes" in its Usage table. A missing one is
    a FatalIOError at read time, which costs a queued run."""
    text = abl.conditions_text(inputs_for(abl))
    for entry in ("Uref", "Zref", "zDir", "flowDir", "z0", "d"):
        assert re.search(rf"^{entry}\s+", text, re.M), entry
    assert "uniform 0.5;" in text  # z0 is a PatchFunction1, so it takes `uniform`


def test_an_optional_entry_at_its_default_is_left_out(abl):
    """`kappa 0.41;` written out reads as a decision somebody made. It was not.

    Asserted against a dictionary entry -- a key at the start of a line -- and not
    against the word, because the header comment states the `u*` formula and the
    word `kappa` is in it.
    """
    text = abl.conditions_text(inputs_for(abl))
    for entry in ("kappa", "Cmu", "C1", "C2"):
        assert not re.search(rf"^{entry}\s", text, re.M), entry


def test_an_optional_entry_moved_off_its_default_is_written(abl):
    text = abl.conditions_text(inputs_for(abl, kappa=0.4, c1=-0.11, c2=1.21))
    assert "kappa                0.4;" in text
    assert "C1                   -0.11;" in text
    assert "C2                   1.21;" in text
    assert "Cmu" not in text


def test_the_conditions_file_states_the_friction_velocity_it_implies(abl):
    text = abl.conditions_text(inputs_for(abl))
    assert f"{abl.ustar(10.0, 10.0, 0.5):.6g}" in text


def test_the_inlet_entry_names_the_type_and_includes_the_conditions(abl):
    for field, kind in abl.INLET_TYPES.items():
        block = abl.inlet_block(field)
        assert f"type            {kind};" in block
        assert '#include        "include/ABLConditions"' in block


def test_the_inlet_entry_carries_a_value_because_a_real_build_demands_one(abl):
    """Measured, on OpenFOAM v2412: the tutorial-shaped two-line entry fails to read
    with "Required entry 'value' : missing for patch inlet", and it fails the same way
    with `initABL true` added. The source reads `value` only `if (!initABL_)`, so the
    build and the source disagree; the entry that reads everywhere is the one with it.
    """
    block = abl.inlet_block("U")
    assert "value           $internalField;" in block
    doc = abl.inlet_block.__doc__ or ""
    assert "Required entry 'value'" in doc and "v2412" in doc


def test_the_ground_wall_functions_take_the_roughness_they_need(abl):
    """`atmNutkWallFunction.H`, `atmEpsilonWallFunction.H` and `atmOmegaWallFunction.H`
    all mark `z0` required; `kqRWallFunction` takes none."""
    assert "z0              uniform 0.7;" in abl.ground_block("nut", 0.7)
    assert "z0              uniform 0.7;" in abl.ground_block("epsilon", 0.7)
    assert "z0              uniform 0.7;" in abl.ground_block("omega", 0.7)
    assert "z0" not in abl.ground_block("k", 0.7)
    assert "kqRWallFunction" in abl.ground_block("k", 0.7)


def test_a_direction_is_three_numbers_however_it_is_spelled(abl):
    assert abl.parse_vector("1 0 0", "--flow-dir") == [1.0, 0.0, 0.0]
    assert abl.parse_vector("(0.866 0.5 0)", "--flow-dir") == [0.866, 0.5, 0.0]
    assert abl.parse_vector("0,0,1", "--z-dir") == [0.0, 0.0, 1.0]
    with pytest.raises(abl.Refused, match="three numbers"):
        abl.parse_vector("1 0", "--flow-dir")


def test_minus_zero_is_written_as_zero(abl):
    """A direction taken off a patch normal produces negative zeros, and `(-0 -0 1)`
    reads like a bug to whoever opens the file."""
    assert abl.vector_text([-0.0, -0.0, 1.0]) == "(0 0 1)"


# -- editing a field file ----------------------------------------------------------


FIELD = """\
FoamFile { version 2.0; format ascii; class volScalarField; object epsilon; }

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform 0.01;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            inletOutlet;
        inletValue      uniform 0.01;
        value           $internalField;
    }

    ground
    {
        type            zeroGradient;
    }
}
"""


def test_only_the_named_patch_changes(abl):
    new = abl.replace_patch(FIELD, "inlet", lambda indent: abl.inlet_block("epsilon", indent))
    assert "atmBoundaryLayerInletEpsilon" in new
    # The other two patches survive byte for byte.
    for block in ("inletValue      uniform 0.01;", "    ground\n    {\n        type            zeroGradient;"):
        assert block in new
    assert new.count("zeroGradient") == 1


def test_replacing_twice_is_replacing_once(abl):
    once = abl.replace_patch(FIELD, "inlet", lambda i: abl.inlet_block("epsilon", i))
    twice = abl.replace_patch(once, "inlet", lambda i: abl.inlet_block("epsilon", i))
    assert once == twice


def test_a_quoted_patch_name_is_found(abl):
    text = FIELD.replace("    inlet\n", '    "inlet"\n')
    assert "atmBoundaryLayerInletEpsilon" in abl.replace_patch(
        text, "inlet", lambda i: abl.inlet_block("epsilon", i))


def test_a_patch_that_is_not_there_is_a_key_error_not_a_guess(abl):
    with pytest.raises(KeyError):
        abl.find_patch_block(FIELD, "nowhere")


def test_an_ambiguous_name_is_refused_rather_than_guessed_at(abl):
    """Rewriting the wrong patch produces a case that runs and lies."""
    text = FIELD + "\ninlet\n{\n    type zeroGradient;\n}\n"
    with pytest.raises(abl.Refused, match="appears 2 times"):
        abl.find_patch_block(text, "inlet")


def test_an_unclosed_block_is_refused(abl):
    with pytest.raises(abl.Refused, match="never closed"):
        abl.find_patch_block("inlet\n{\n    type zeroGradient;\n", "inlet")


def test_nested_braces_inside_a_patch_do_not_end_it_early(abl):
    text = FIELD.replace(
        "        type            zeroGradient;\n",
        "        type            zeroGradient;\n        extra { a 1; b 2; }\n", 1)
    opening, closing, _ = abl.find_patch_block(text, "inlet")
    assert text[opening:closing].count("{") == text[opening:closing].count("}")
    assert "extra" in text[opening:closing]


def test_the_conditions_include_goes_in_once(abl):
    text, added = abl.ensure_include(FIELD)
    assert added and '#include "include/ABLConditions"' in text
    again, added_again = abl.ensure_include(text)
    assert not added_again and again == text


# -- writing a whole case ----------------------------------------------------------


def make_case(root: Path, fields=("U", "k", "epsilon", "omega", "nut")) -> Path:
    (root / "0").mkdir(parents=True)
    for name in fields:
        (root / "0" / name).write_text(FIELD.replace("object epsilon", f"object {name}"),
                                       encoding="utf-8")
    return root


def test_the_case_gets_the_include_and_four_retyped_patches(abl, tmp_path):
    case = make_case(tmp_path / "case")
    report = abl.write_case(case, inputs_for(abl), "inlet")
    assert (case / "0" / "include" / "ABLConditions").is_file()
    for field, kind in abl.INLET_TYPES.items():
        assert kind in (case / "0" / field).read_text(encoding="utf-8")
    assert any("0/nut" in line and "#include" in line for line in report)


def test_writing_twice_produces_the_same_bytes(abl, tmp_path):
    case = make_case(tmp_path / "case")
    abl.write_case(case, inputs_for(abl), "inlet")
    before = {p.name: p.read_bytes() for p in sorted((case / "0").rglob("*")) if p.is_file()}
    abl.write_case(case, inputs_for(abl), "inlet")
    after = {p.name: p.read_bytes() for p in sorted((case / "0").rglob("*")) if p.is_file()}
    assert before == after


def test_a_field_that_is_not_there_is_reported_and_not_invented(abl, tmp_path):
    case = make_case(tmp_path / "case", fields=("U", "k"))
    report = abl.write_case(case, inputs_for(abl), "inlet")
    assert any("no 0/epsilon" in line for line in report)
    assert not (case / "0" / "epsilon").exists()


def test_a_case_with_no_time_zero_is_refused(abl, tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(abl.Refused, match="no 0/ or 0.orig/"):
        abl.write_case(tmp_path / "bare", inputs_for(abl), "inlet")


def test_the_ground_patch_is_only_touched_when_it_is_named(abl, tmp_path):
    case = make_case(tmp_path / "case")
    abl.write_case(case, inputs_for(abl), "inlet")
    assert "atmNutkWallFunction" not in (case / "0" / "nut").read_text(encoding="utf-8")

    case2 = make_case(tmp_path / "case2")
    abl.write_case(case2, inputs_for(abl), "inlet", ground="ground")
    assert "atmNutkWallFunction" in (case2 / "0" / "nut").read_text(encoding="utf-8")
    assert "atmEpsilonWallFunction" in (case2 / "0" / "epsilon").read_text(encoding="utf-8")
    assert "atmOmegaWallFunction" in (case2 / "0" / "omega").read_text(encoding="utf-8")
    assert "kqRWallFunction" in (case2 / "0" / "k").read_text(encoding="utf-8")


# -- the site file -----------------------------------------------------------------


SITE = {
    "schema": "openreynolds.site/1",
    "site": {"name": "somewhere", "latitude": 51.5, "longitude": -0.1},
    "weather": {
        "prevailing": {"centre_deg": 240.0, "frequency": 0.25},
        "sectors": [
            {"centre_deg": 240.0, "hours": 2147, "frequency": 0.25,
             "speed_10m": {"hours": 2147, "mean": 4.851, "p50": 4.5, "p90": 7.9}},
            {"centre_deg": 60.0, "hours": 500, "frequency": 0.06,
             "speed_10m": {"hours": 500, "mean": 3.2, "p50": 3.0, "p90": 5.5}},
        ],
        "shear": {"fitted_z0_m": 0.28364},
    },
}


def write_site(tmp_path: Path, data=None) -> Path:
    path = tmp_path / "site.json"
    path.write_text(json.dumps(data if data is not None else SITE), encoding="utf-8")
    return path


def test_a_missing_site_file_names_the_command_that_makes_one(abl, tmp_path):
    with pytest.raises(abl.Refused) as caught:
        abl.load_site(tmp_path / "nope.json")
    message = str(caught.value)
    assert "openreynolds site --lat" in message
    assert "no outbound network" in message
    assert "--uref" in message  # and the way round it


def test_a_file_of_the_wrong_schema_is_refused(abl, tmp_path):
    path = write_site(tmp_path, {"schema": "something/else"})
    with pytest.raises(abl.Refused, match="openreynolds.site/1"):
        abl.load_site(path)


def test_unreadable_json_is_refused_by_name(abl, tmp_path):
    path = tmp_path / "site.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(abl.Refused, match="not readable JSON"):
        abl.load_site(path)


def test_the_prevailing_sector_supplies_uref_and_the_flow_direction(abl, tmp_path):
    inputs = inputs_for(abl, site=str(write_site(tmp_path)), uref=None, zref=None)
    assert inputs["Uref"] == 4.851
    assert inputs["Zref"] == 10.0
    # From 240 degrees means travelling towards 060: east-north-east.
    assert inputs["flowDir"][0] == pytest.approx(math.sin(math.radians(60)), abs=1e-5)
    assert inputs["flowDir"][1] == pytest.approx(math.cos(math.radians(60)), abs=1e-5)
    assert "prevailing sector" in inputs["from"]["Uref"]
    assert "2147 hours" in inputs["from"]["Uref"]


def test_a_statistic_other_than_the_mean_can_be_asked_for(abl, tmp_path):
    inputs = inputs_for(abl, site=str(write_site(tmp_path)), uref=None, zref=None,
                        statistic="p90")
    assert inputs["Uref"] == 7.9


def test_a_statistic_the_file_does_not_have_lists_what_it_does(abl, tmp_path):
    with pytest.raises(abl.Refused, match="p98"):
        inputs_for(abl, site=str(write_site(tmp_path)), uref=None, zref=None,
                   statistic="p98")


def test_a_named_sector_overrides_the_prevailing_one(abl, tmp_path):
    inputs = inputs_for(abl, site=str(write_site(tmp_path)), uref=None, zref=None,
                        sector=70.0)
    assert inputs["Uref"] == 3.2


def test_an_explicit_uref_beats_the_file(abl, tmp_path):
    inputs = inputs_for(abl, site=str(write_site(tmp_path)), uref=12.0, zref=80.0)
    assert (inputs["Uref"], inputs["Zref"]) == (12.0, 80.0)
    assert inputs["from"]["Uref"] == "--uref"


def test_a_weather_block_that_records_a_failure_is_not_read_as_data(abl, tmp_path):
    data = dict(SITE, weather={"failed": "Open-Meteo answered 503"})
    with pytest.raises(abl.Refused, match="503"):
        inputs_for(abl, site=str(write_site(tmp_path, data)), uref=None, zref=None)


def test_the_files_fitted_roughness_is_offered_and_labelled_as_not_the_sites(abl, tmp_path):
    """The reanalysis fit is the grid cell's roughness. Handing it over as the site's
    would be the brief's own mistake, so the refusal says what it is."""
    import argparse
    with pytest.raises(abl.Refused) as caught:
        abl.resolve(argparse.Namespace(
            site=str(write_site(tmp_path)), statistic="mean", sector=None, uref=None,
            zref=None, z0=None, roughness="", d=0.0, flow_dir="", z_dir="",
            kappa=abl.KAPPA, cmu=abl.CMU, c1=0.0, c2=1.0))
    message = str(caught.value)
    assert "0.28364" in message
    assert "grid cell's roughness and not the" in message


# -- the notes, the table and the figure -------------------------------------------


def test_the_notes_say_where_z_is_measured_from(abl):
    text = " ".join(abl.notes(inputs_for(abl)))
    assert "lowest point" in text
    assert "not above z = 0" in text


def test_the_notes_carry_the_sigma_epsilon_caveat(abl):
    """The profiles satisfy the k equation and only satisfy the epsilon equation at
    sigmaEpsilon 1.11, which is not the kEpsilon default. Stated, not enforced."""
    text = " ".join(abl.notes(inputs_for(abl)))
    assert "1.11" in text and "1.3" in text and "Hargreaves" in text


def test_the_notes_survive_a_displacement_height(abl):
    """`notes` sampled the profile at 2 z0, which with d = 2 m is below the profile's
    own floor, so `profile` refused it and a report became a traceback. It samples at
    d + 2 z0 now, and states where the floor is."""
    text = " ".join(abl.notes(inputs_for(abl, d=2.0)))
    assert "U is zero at z = d = 2 m" in text
    assert "d - z0 = 1.5 m" in text
    assert "Zref + d = 12 m" in text


def test_the_notes_offer_no_verdict(abl):
    """The toolbox reports; it does not grade. `docs/design.md` section 1."""
    text = " ".join(abl.notes(inputs_for(abl, d=2.0))).lower()
    for word in ("you must", "you should", "too coarse", "unacceptable", "wrong",
                 "fix the", "increase the"):
        assert word not in text


def test_the_table_and_the_figure_are_written_where_asked(abl, tmp_path):
    table = tmp_path / "profile.csv"
    figure = tmp_path / "abl.png"
    assert abl.main(["--uref", "10", "--zref", "10", "--z0", "0.5",
                     "--table", str(table), "--figure", str(figure), "--json"]) == 0
    rows = table.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == ["z", "z_eff", "U", "k", "epsilon", "omega"]
    assert len(rows) > 20
    assert figure.stat().st_size > 5000


def test_the_json_answer_carries_the_inputs_their_provenance_and_the_profile(abl, capsys):
    assert abl.main(["--uref", "10", "--zref", "10", "--z0", "0.5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ustar"] == pytest.approx(abl.ustar(10.0, 10.0, 0.5))
    assert payload["inputs"]["from"]["z0"] == "--z0"
    assert payload["inlet_types"]["U"] == "atmBoundaryLayerInletVelocity"
    assert len(payload["profile"]) > 20


# -- layer 3: against the solver ---------------------------------------------------
#
# Skipped without OpenFOAM, which is every developer laptop and every runner in the
# matrix; it runs in the workspace image, where v2512 is. The bashrc search is the
# one `tests/test_toolbox_snappy_path.py` uses, for the same reason.


def foam_bashrc() -> Path | None:
    etc = os.environ.get("FOAM_ETC")
    if etc and (Path(etc) / "bashrc").is_file():
        return Path(etc) / "bashrc"
    if shutil.which("blockMesh") and shutil.which("simpleFoam"):
        return Path("")  # already on PATH; no sourcing needed
    for root in (Path("/usr/lib/openfoam"), Path("/opt"), Path("/usr/lib")):
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob("openfoam*/etc/bashrc"), reverse=True):
            return candidate
    return None


BASHRC = foam_bashrc()
needs_openfoam = pytest.mark.skipif(
    BASHRC is None,
    reason="OpenFOAM is not installed here; the profile read-back needs a real solver",
)

HEAD = "FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"

BLOCK_MESH = HEAD.format(cls="dictionary", obj="blockMeshDict") + """
scale 1;
vertices
(
    (0 0 0) (100 0 0) (100 20 0) (0 20 0)
    (0 0 200) (100 0 200) (100 20 200) (0 20 200)
);
blocks ( hex (0 1 2 3 4 5 6 7) (6 2 40) simpleGrading (1 1 10) );
edges ();
boundary
(
    inlet  { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    ground { type wall;  faces ((0 1 2 3)); }
    top    { type patch; faces ((4 5 6 7)); }
    sides  { type patch; faces ((0 1 5 4) (3 7 6 2)); }
);
mergePatchPairs ();
"""

CONTROL_DICT = HEAD.format(cls="dictionary", obj="controlDict") + """
application simpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0;
writeFormat ascii; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable false;
"""

FV_SCHEMES = HEAD.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss limitedLinear 1;
    div(phi,epsilon) bounded Gauss limitedLinear 1;
    div(phi,omega) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
"""

FV_SOLUTION = HEAD.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p { solver GAMG; tolerance 1e-6; relTol 0.1; smoother GaussSeidel; }
    "(U|k|epsilon|omega)" { solver smoothSolver; smoother symGaussSeidel;
                            tolerance 1e-8; relTol 0.1; }
}
SIMPLE { nNonOrthogonalCorrectors 0; consistent no; pRefCell 0; pRefValue 0;
         residualControl { p 1e-20; U 1e-20; } }
relaxationFactors { fields { p 0.3; } equations { U 0.7; "(k|epsilon|omega)" 0.7; } }
"""

ZERO_FIELDS = {
    "U": ("volVectorField", "[0 1 -1 0 0 0 0]", "uniform (1 0 0)"),
    "p": ("volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0"),
    "k": ("volScalarField", "[0 2 -2 0 0 0 0]", "uniform 6"),
    "epsilon": ("volScalarField", "[0 2 -3 0 0 0 0]", "uniform 0.5"),
    "omega": ("volScalarField", "[0 0 -1 0 0 0 0]", "uniform 1"),
    "nut": ("volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0.1"),
}

PLACEHOLDER = """
dimensions      {dims};

internalField   {internal};

boundaryField
{{
    inlet   {{ type zeroGradient; }}
    outlet  {{ type zeroGradient; }}
    ground  {{ type zeroGradient; }}
    top     {{ type slip; }}
    sides   {{ type slip; }}
}}
"""


def build_column(root: Path) -> Path:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "0").mkdir()
    (root / "system" / "blockMeshDict").write_text(BLOCK_MESH, encoding="utf-8")
    (root / "system" / "controlDict").write_text(CONTROL_DICT, encoding="utf-8")
    (root / "system" / "fvSchemes").write_text(FV_SCHEMES, encoding="utf-8")
    (root / "system" / "fvSolution").write_text(FV_SOLUTION, encoding="utf-8")
    (root / "constant" / "transportProperties").write_text(
        HEAD.format(cls="dictionary", obj="transportProperties")
        + "\ntransportModel Newtonian;\nnu 1.5e-05;\n", encoding="utf-8")
    (root / "constant" / "turbulenceProperties").write_text(
        HEAD.format(cls="dictionary", obj="turbulenceProperties")
        + "\nsimulationType RAS;\nRAS { model kEpsilon; turbulence on; printCoeffs no; }\n",
        encoding="utf-8")
    for name, (cls, dims, internal) in ZERO_FIELDS.items():
        (root / "0" / name).write_text(
            HEAD.format(cls=cls, obj=name) + PLACEHOLDER.format(dims=dims, internal=internal),
            encoding="utf-8")
    return root


def foam(command: str, cwd: Path) -> subprocess.CompletedProcess:
    prefix = f"source {BASHRC} && " if BASHRC and str(BASHRC) else ""
    return subprocess.run(["bash", "-lc", f"{prefix}cd {cwd} && {command}"],
                          capture_output=True, text=True, timeout=600)


def patch_values(path: Path, patch: str, module, count: int | None = None) -> list:
    """The `value` OpenFOAM wrote on one patch, as one entry a face.

    Both forms, because which one appears is itself a result: `k` comes back as
    `value uniform 6.04516418298;` -- one number for the whole patch -- and that is
    the Richards-Hoxey profile being genuinely constant with height, not the
    condition failing to vary. U and epsilon come back as nonuniform lists.
    """
    text = path.read_text(encoding="utf-8")
    opening, closing, _ = module.find_patch_block(text, patch)
    block = text[opening:closing]

    uniform = re.search(r"value\s+uniform\s+(\([^)]*\)|[-+0-9.eE]+)\s*;", block)
    if uniform:
        assert count, f"{path.name}: a uniform {patch} value needs a face count"
        raw = uniform.group(1)
        one = (tuple(float(v) for v in raw.strip("()").split())
               if raw.startswith("(") else float(raw))
        return [one] * count

    match = re.search(r"value\s+nonuniform\s+List<\w+>\s*\n?\s*\d+\s*\(", block)
    assert match, f"{path.name} wrote no readable value on {patch}:\n{block[:500]}"
    start = block.index("(", match.end() - 1)
    depth, index = 0, start
    while index < len(block):
        if block[index] == "(":
            depth += 1
        elif block[index] == ")":
            depth -= 1
            if depth == 0:
                break
        index += 1
    body = block[start + 1:index]
    if "(" in body:
        return [tuple(float(v) for v in triple.split())
                for triple in re.findall(r"\(([^)]*)\)", body)]
    return [float(v) for v in body.split()]


def read_back(abl, case: Path, fields: tuple[str, ...], uref: float, zref: float,
              z0: float) -> dict[str, float]:
    """One iteration, then the inlet values the solver wrote against this script's.

    Returns the worst relative difference per field. `z` is measured from the inlet
    patch's lowest point, which is the bottom of the block at z = 0, and not from the
    lowest face centre -- `boundBox(pp.localPoints()).min()` is over points.
    """
    run = foam("simpleFoam", case)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    centres = foam("postProcess -func writeCellCentres -time 1", case)
    assert centres.returncode == 0, centres.stdout[-3000:]

    out = case / "1"
    face = patch_values(out / "C", "inlet", abl)
    faces = len(face)
    assert faces > 20

    solver: dict[str, list] = {}
    for field in fields:
        values = patch_values(out / field, "inlet", abl, faces)
        assert len(values) == faces, field
        solver[field] = ([math.sqrt(sum(c * c for c in v)) for v in values]
                         if field == "U" else values)

    worst = {field: 0.0 for field in fields}
    for index, point in enumerate(face):
        want = abl.profile(point[2] - 0.0, uref, zref, z0)
        for field in fields:
            got = solver[field][index]
            worst[field] = max(worst[field],
                               abs(got - want[field]) / max(abs(want[field]), 1e-30))
    return worst


@needs_openfoam
def test_the_solver_reads_back_the_profile_this_script_wrote(abl, tmp_path):
    """The whole point, measured: `simpleFoam` evaluates the conditions this script
    pointed the inlet at, writes the patch values it arrived at, and they are read
    back here and compared with this script's own table at the same face centres.

    Two runs, because no single turbulence model evaluates all four conditions:
    kEpsilon never touches `omega` and kOmegaSST never touches `epsilon`, and a
    condition the model does not use is never asked for its value, so it keeps the
    placeholder and a comparison against it would be comparing nothing.

    Measured on v2412 at the floor of ASCII `writePrecision 12`: U to 3.7e-12, k to
    4.7e-13, epsilon to 4.4e-12, omega to 4.4e-12. The tolerance is between two and
    three orders looser than that, which still leaves no room for a wrong formula.
    """
    uref, zref, z0 = 10.0, 10.0, 0.5
    case = build_column(tmp_path / "column")
    abl.write_case(case, inputs_for(abl, uref=uref, zref=zref, z0=z0), "inlet",
                   ground="ground")
    mesh = foam("blockMesh", case)
    assert mesh.returncode == 0, mesh.stdout[-3000:] + mesh.stderr[-3000:]

    worst = read_back(abl, case, ("U", "k", "epsilon"), uref, zref, z0)

    turbulence = case / "constant" / "turbulenceProperties"
    turbulence.write_text(turbulence.read_text(encoding="utf-8")
                          .replace("model kEpsilon;", "model kOmegaSST;"), encoding="utf-8")
    shutil.rmtree(case / "1")
    worst.update(read_back(abl, case, ("U", "omega"), uref, zref, z0))

    assert set(worst) == {"U", "k", "epsilon", "omega"}
    for field, difference in worst.items():
        assert difference < 1e-9, (
            f"{field}: the solver and this script disagree by {difference:.3e} at "
            f"worst; the profile written is not the profile read back"
        )


@needs_openfoam
def test_the_case_the_script_writes_reads_without_a_fatal_io_error(abl, tmp_path):
    """The `value` entry's reason for existing, as a test rather than a comment: a
    field file this script wrote is read by a real OpenFOAM without complaint."""
    case = build_column(tmp_path / "column")
    abl.write_case(case, inputs_for(abl), "inlet", ground="ground")
    assert foam("blockMesh", case).returncode == 0
    read = foam('postProcess -time 0 -fields "(U k epsilon omega)"', case)
    assert "FATAL" not in read.stdout.upper(), read.stdout[-3000:]
    assert read.returncode == 0, read.stdout[-3000:]
