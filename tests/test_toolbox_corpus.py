"""`corpus.py`: reading an OpenFOAM dictionary well enough to index one.

The reader is a keyword scraper, not a parser, and the interesting tests are all
hazards taken from the real v2512 tutorial tree rather than from the format's
documentation. Two of them would each have produced an index that silently held
nothing: turbulence keys live one block down, and at least one shipped file carries
a comment inside the value it is asked for.

What the reader must never do is guess. A key it cannot find and a file it cannot
open both come back as `None`, because a corpus with no benchmark tier under it
cannot afford invented values that look measured.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def corpus():
    spec = importlib.util.spec_from_file_location("toolbox_corpus", TOOLBOX / "corpus.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2512                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


# -- comments ------------------------------------------------------------------


def test_a_semicolon_inside_a_trailing_comment_does_not_extend_the_value(corpus):
    """`lagrangian/kinematicParcelFoam/drippingChair` reads

        simulationType      laminar; // RAS;

    and 125 entries across the shipped tree have that shape: a settled value, then
    a comment that itself contains a semicolon. Read greedily -- to the last `;` on
    the line -- the value comes back as `laminar; // RAS`, which matches no known
    simulation type, so the case indexes with its turbulence nulled out.
    """
    text = BANNER + "simulationType      laminar; // RAS;\n"
    found = corpus.entries(corpus.strip_comments(text))
    assert found["simulationType"] == "laminar"
    assert "RAS" not in found


def test_a_url_in_the_banner_does_not_unterminate_the_block_comment(corpus):
    """Four files in the tree carry `https://www.openfoam.com` in the banner, and
    the rest carry it without the scheme. Those four are why comments are stripped
    blocks-first: a `//` pass run first cuts that line at the scheme, leaving the
    banner unterminated, and the block pass then eats the file down to whatever
    `*/` comes next -- which is the closing banner of the *following* section, so
    the real entries in between vanish.
    """
    with_scheme = BANNER.replace("www.openfoam.com", "https://www.openfoam.com")
    text = with_scheme + "application     simpleFoam;\n"
    assert corpus.entries(corpus.strip_comments(text))["application"] == "simpleFoam"


def test_a_block_comment_does_not_reach_the_value(corpus):
    """Defensive, not observed: the shipped tree has no entry with a block comment
    between its value and its semicolon. It is cheap to be right about anyway, and
    a hand-edited case is not held to the tree's habits."""
    text = BANNER + "application  /* was: icoFoam */ simpleFoam;\n"
    assert corpus.entries(corpus.strip_comments(text))["application"] == "simpleFoam"


def test_the_banner_contributes_nothing_but_the_foamfile_block(corpus):
    """Every dictionary in the tree opens with that block comment, and it is full of
    words followed by other words. If the banner parses, every case in the corpus
    carries the same handful of junk fields."""
    found = corpus.entries(corpus.strip_comments(BANNER))
    assert "Website" not in found
    assert "Version" not in found
    assert set(found) <= {"version", "format", "class", "object", "arch", "note", "location"}


# -- whitespace ----------------------------------------------------------------


@pytest.mark.parametrize(
    "spacing",
    [
        "application interFoam;",
        "application     interFoam;",
        "application\tinterFoam;",
        "  application       interFoam;",
    ],
)
def test_spacing_does_not_change_the_value(corpus, spacing):
    """Both `application     interFoam;` and `application       adjointOptimisationFoam;`
    appear in the shipped tree; nothing normalises them upstream."""
    assert corpus.entries(corpus.strip_comments(spacing))["application"] == "interFoam"


# -- depth ---------------------------------------------------------------------


TURBULENCE_PROPERTIES = BANNER + """
simulationType      RAS;

RAS
{
    // Tested with kEpsilon, realizableKE, kOmega, kOmegaSST,
    // ShihQuadraticKE, LienCubicKE.
    RASModel        kEpsilon;

    turbulence      on;

    printCoeffs     on;
}


// ************************************************************************* //
"""
"""`incompressible/simpleFoam/pitzDaily/constant/turbulenceProperties`, verbatim."""


def test_a_key_one_block_down_is_still_found(corpus):
    """ESI keeps the model inside the block that names the simulation type, so a
    reader that only looks at the top level of the file finds `simulationType` and
    never the model -- which is the field a seed query is actually ranked on."""
    found = corpus.entries(corpus.strip_comments(TURBULENCE_PROPERTIES))
    assert found["simulationType"] == "RAS"
    assert found["RASModel"] == "kEpsilon"


def test_the_models_listed_in_a_comment_are_not_read_as_the_model(corpus):
    """That file recommends five alternatives to `kEpsilon` in a comment directly
    above the entry that chooses it. A reader that indexed the file's text rather
    than its entries would have six models for one case, and the ranking in
    `search.py` would then be scoring a suggestion as though it were a decision."""
    found = corpus.entries(corpus.strip_comments(TURBULENCE_PROPERTIES))
    assert found["RASModel"] == "kEpsilon"
    for suggestion in ("realizableKE", "kOmega", "kOmegaSST", "ShihQuadraticKE"):
        assert suggestion not in found.values()


def test_the_first_occurrence_of_a_repeated_key_wins(corpus):
    text = "startFrom  latestTime;\nstartFrom  startTime;\n"
    assert corpus.entries(corpus.strip_comments(text))["startFrom"] == "latestTime"


# -- absence -------------------------------------------------------------------


def test_an_absent_key_reads_as_none(corpus, tmp_path):
    path = tmp_path / "controlDict"
    path.write_text(BANNER + "application  simpleFoam;\n", encoding="utf-8")
    assert corpus.entry(path, "application") == "simpleFoam"
    assert corpus.entry(path, "RASModel") is None


def test_a_file_that_will_not_read_is_none_and_not_a_traceback(corpus, tmp_path):
    """556 cases indexed in one pass; one unreadable file must cost that case and
    not the corpus."""
    assert corpus.entry(tmp_path / "does-not-exist", "application") is None
    assert corpus.read(tmp_path / "does-not-exist") == ""


def test_a_directory_in_place_of_a_file_is_survived(corpus, tmp_path):
    (tmp_path / "controlDict").mkdir()
    assert corpus.entry(tmp_path / "controlDict", "application") is None


def test_a_binary_file_is_survived(corpus, tmp_path):
    path = tmp_path / "points"
    path.write_bytes(b"\x00\x81\xfe binary contents \x00")
    assert corpus.entry(path, "application") is None


# -- boundary conditions -------------------------------------------------------


FIELD_U = BANNER + """dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (10 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    "wall.*"
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""


def test_boundary_types_maps_each_patch_to_its_type(corpus):
    assert corpus.boundary_types(corpus.strip_comments(FIELD_U)) == {
        "inlet": "fixedValue",
        "outlet": "zeroGradient",
        "wall.*": "noSlip",
        "frontAndBack": "empty",
    }


def test_nothing_outside_boundaryfield_becomes_a_patch(corpus):
    """`FoamFile` is a word followed by a brace at the top of every field file, and
    `dimensions` and `internalField` are entries beside `boundaryField`. A scan that
    takes any `name {` as a patch reports `FoamFile` as a boundary on every case in
    the corpus."""
    patches = corpus.boundary_types(corpus.strip_comments(FIELD_U))
    assert "FoamFile" not in patches
    assert "dimensions" not in patches
    assert "internalField" not in patches


def test_a_value_entry_is_not_mistaken_for_the_type(corpus):
    """`value` sits beside `type` in most patch entries and would otherwise overwrite
    it, giving every fixedValue patch the type `uniform (10 0 0)`."""
    assert corpus.boundary_types(corpus.strip_comments(FIELD_U))["inlet"] == "fixedValue"


def test_a_patch_entry_with_its_own_block_keeps_its_type(corpus):
    """Depth bookkeeping: a nested block inside one patch must not swallow the
    patches that follow it."""
    text = BANNER + """boundaryField
{
    inlet
    {
        type            codedFixedValue;
        value           uniform (0 0 0);
        codeOptions
        {
            type        ignored;
        }
    }
    outlet
    {
        type            inletOutlet;
    }
}
"""
    assert corpus.boundary_types(corpus.strip_comments(text)) == {
        "inlet": "codedFixedValue",
        "outlet": "inletOutlet",
    }


def test_a_field_with_no_boundaryfield_is_empty_and_not_an_error(corpus):
    assert corpus.boundary_types(corpus.strip_comments(BANNER)) == {}


@pytest.mark.parametrize(
    "name,expected",
    [
        ('".*"', ".*"),
        ('"outlet.*"', "outlet.*"),
        ('"(left|right)"', "(left|right)"),
        ('"(?i).*walls"', "(?i).*walls"),
        ('"wall|fixedWalls|topOPatch"  ', "wall|fixedWalls|topOPatch"),
    ],
)
def test_a_regular_expression_patch_name_survives_whole(corpus, name, expected):
    """The five most common quoted patch names in the tree's `U` files, one of them
    with the trailing whitespace it really carries. These are patch *groups*, and a
    name broken at the `|` or the `(` would index one case's walls as several
    patches that do not exist."""
    text = BANNER + "boundaryField\n{\n    " + name + "\n    {\n        type noSlip;\n    }\n}\n"
    assert corpus.boundary_types(corpus.strip_comments(text)) == {expected: "noSlip"}


# =============================================================================
# The vendor harvest
# =============================================================================
#
# Fixtures are miniature case trees, a handful of files each, not copies of real
# tutorials. What each one encodes is a fact measured off `opencfd/openfoam-default:2512`,
# and where a count is quoted it was counted.


def dictionary(path: Path, body: str) -> Path:
    """Write a dictionary file with the banner every real one carries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BANNER + body, encoding="utf-8")
    return path


def make_case(
    root: Path,
    *,
    application: str | None = "simpleFoam",
    turbulence: tuple[str, str] | None = ("RAS", "kEpsilon"),
    fork: str = "esi",
    mesh: tuple[str, ...] = ("blockMeshDict",),
    geometry: bool = False,
    ddt: str | None = "steadyState",
    schemes_before_ddt: bool = False,
    properties: str | None = "transportProperties",
    time_dir: str | None = "0.orig",
    regions: dict[str, tuple[str, str]] | None = None,
) -> Path:
    """A miniature case: a handful of files, each one carrying a measured fact."""
    root.mkdir(parents=True, exist_ok=True)
    control = "startTime  0;\nendTime  100;\n"
    if application is not None:
        control = f"application     {application};\n" + control
    dictionary(root / "system" / "controlDict", control)

    if turbulence is not None:
        simulation, model = turbulence
        key = "model" if fork == "foundation" else f"{simulation}Model"
        name = "momentumTransport" if fork == "foundation" else "turbulenceProperties"
        dictionary(
            root / "constant" / name,
            f"simulationType      {simulation};\n\n{simulation}\n{{\n"
            f"    {key}        {model};\n    turbulence      on;\n}}\n",
        )

    for name in mesh:
        dictionary(root / "system" / name, "scale   0.001;\n")
    if geometry:
        (root / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
        (root / "constant" / "triSurface" / "body.stl").write_text("solid body\nendsolid\n")

    if ddt is not None:
        blocks = ""
        if schemes_before_ddt:
            # Real fvSchemes files open with several blocks that each carry a
            # `default`, and gradSchemes is often one of them.
            blocks += "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        blocks += f"ddtSchemes\n{{\n    default         {ddt};\n}}\n\n"
        blocks += "divSchemes\n{\n    default         none;\n}\n"
        dictionary(root / "system" / "fvSchemes", blocks)

    if properties is not None:
        dictionary(root / "constant" / properties, "nu  [0 2 -1 0 0 0 0] 1e-05;\n")

    if time_dir is not None:
        dictionary(
            root / time_dir / "U",
            "dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   uniform (0 0 0);\n\n"
            "boundaryField\n{\n    inlet\n    {\n        type            fixedValue;\n"
            "        value           uniform (10 0 0);\n    }\n"
            "    outlet\n    {\n        type            zeroGradient;\n    }\n}\n",
        )
        dictionary(
            root / time_dir / "p",
            "boundaryField\n{\n    inlet\n    {\n        type            zeroGradient;\n    }\n}\n",
        )

    if regions:
        dictionary(
            root / "constant" / "regionProperties",
            "regions\n(\n    fluid       (" + " ".join(regions) + ")\n);\n",
        )
        for region, (simulation, model) in regions.items():
            dictionary(
                root / "constant" / region / "turbulenceProperties",
                f"simulationType      {simulation};\n\n{simulation}\n{{\n"
                f"    {simulation}Model        {model};\n}}\n",
            )
    return root


@pytest.fixture
def one_case(corpus, tmp_path):
    """Harvest a single case built with `make_case` kwargs, and hand back its row.

    Each call gets its own tree. Sharing one would let a second call's files land
    beside the first's -- a case carrying both `thermophysicalProperties` and
    `transportProperties`, which is not a case the corpus contains.
    """
    trees = iter(range(1000))

    def build(**kwargs):
        tree = tmp_path / f"tutorials-{next(trees)}"
        make_case(tree / "family" / "aCase", **kwargs)
        rows, counts = corpus.harvest_tutorials(tree)
        assert counts["skipped"] == 0, counts
        assert len(rows) == 1, rows
        return rows[0]

    return build


# -- what a row says -----------------------------------------------------------


def test_a_minimal_case_indexes_its_solver_and_turbulence(corpus, one_case):
    row = one_case()
    assert row["tier"] == "vendor"
    assert row["solver"] == {"executable": "simpleFoam", "module": None}
    assert row["runs"] is True
    assert row["turbulence"] == {"simulation_type": "RAS", "model": "kEpsilon"}
    assert row["mesh_type"] == "blockMesh"
    assert row["bc_map"]["U"] == {"inlet": "fixedValue", "outlet": "zeroGradient"}
    assert row["provenance"]["schema_version"] == corpus.SCHEMA_VERSION
    assert row["verdict"] is None


def test_the_path_recorded_is_the_case_directory(one_case):
    row = one_case()
    assert row["path"].endswith("aCase")
    assert Path(row["path"]).is_dir()


# -- runs: false has two causes, not one ---------------------------------------


def test_a_mesher_in_the_application_field_does_not_run(one_case):
    """18 cases in the tree name `snappyHexMesh` as their application. Taking that as
    the solver would put a mesher in the solver field and make those cases retrievable
    as seeds for a solve they cannot perform."""
    row = one_case(application="snappyHexMesh", mesh=("blockMeshDict", "snappyHexMeshDict"))
    assert row["solver"]["executable"] == "snappyHexMesh"
    assert row["runs"] is False


def test_a_case_with_no_application_at_all_does_not_run(one_case):
    """The other four: `mesh/foamyHexMesh/blob`, `foamyHexMesh/simpleShapes`,
    `mesh/foamyQuadMesh/jaggedBoundary` and `foamyQuadMesh/square` carry no
    `application` entry. The spec anticipated one cause for `runs: false`; there are
    two, and this one cannot be derived from the solver name because there is none."""
    row = one_case(application=None)
    assert row["solver"]["executable"] is None
    assert row["runs"] is False


# -- mesh_type, in the order the corpus actually requires ----------------------


def test_a_snappy_case_with_a_background_blockmesh_is_a_snappy_case(one_case):
    """62 of the 64 snappyHexMesh cases also carry a `blockMeshDict`, because snappy
    cuts its mesh out of a background block. The spec's table checks `blockMeshDict`
    first, which labels 97% of the snappy tier `blockMesh` -- so a query for a snappy
    precedent would find two cases instead of sixty-four."""
    row = one_case(mesh=("blockMeshDict", "snappyHexMeshDict"))
    assert row["mesh_type"] == "snappyHexMesh"


def test_a_case_with_only_a_blockmeshdict_is_blockmesh(one_case):
    assert one_case(mesh=("blockMeshDict",))["mesh_type"] == "blockMesh"


def test_a_case_with_geometry_and_no_mesh_dictionary_is_a_surface_case(one_case):
    assert one_case(mesh=(), geometry=True)["mesh_type"] == "surface"


def test_a_case_with_nothing_to_mesh_from_is_unknown(one_case):
    assert one_case(mesh=(), geometry=False)["mesh_type"] == "unknown"


# -- steady or transient, measured rather than inferred from the solver name ---


def test_steady_is_read_from_the_ddtschemes_block(one_case):
    """165 of the tree's `fvSchemes` set `ddtSchemes { default steadyState; }`. This
    is measurable, unlike the internal/external half of the spec's regime class,
    which is why `regime.class` carries only what was read."""
    row = one_case(ddt="steadyState")
    assert row["regime"]["steady"] is True
    assert row["regime"]["class"] == "incompressible-steady"


@pytest.mark.parametrize("scheme", ["Euler", "backward", "CrankNicolson 0.9", "localEuler"])
def test_transient_schemes_read_as_transient(one_case, scheme):
    """294 Euler, 18 backward, 9 localEuler, 3 CrankNicolson -- and CrankNicolson
    carries a coefficient, so only the first word is the scheme."""
    row = one_case(ddt=scheme)
    assert row["regime"]["steady"] is False
    assert row["regime"]["class"] == "incompressible-transient"


def test_the_ddt_scheme_is_not_taken_from_another_schemes_block(one_case):
    """`fvSchemes` has a `default` in gradSchemes, divSchemes, laplacianSchemes and
    more. A flat read of the file returns whichever comes first in the text, so a
    case whose gradSchemes precedes its ddtSchemes would be classified by
    `Gauss linear`. This is what `block()` exists for."""
    row = one_case(ddt="steadyState", schemes_before_ddt=True)
    assert row["regime"]["steady"] is True


def test_a_case_with_no_ddtschemes_is_null_and_not_guessed(one_case):
    """Five `fvSchemes` in the tree have no ddtSchemes block at all."""
    row = one_case(ddt=None)
    assert row["regime"]["steady"] is None
    assert row["regime"]["class"] == "incompressible"


def test_an_unrecognised_ddt_scheme_is_null_rather_than_assumed_transient(one_case):
    row = one_case(ddt="someSchemeShippedLater")
    assert row["regime"]["steady"] is None


def test_compressible_is_read_from_which_properties_file_is_present(one_case):
    """118 cases carry `thermophysicalProperties`, 333 `transportProperties`, 1
    `physicalProperties`."""
    assert one_case(properties="thermophysicalProperties")["regime"]["compressible"] is True
    assert one_case(properties="transportProperties")["regime"]["compressible"] is False
    assert one_case(properties=None)["regime"]["compressible"] is None


def test_the_regime_numbers_stay_null(one_case):
    """Deriving Re or Ma needs a velocity scale, a length scale and a viscosity that a
    tutorial does not reliably state. Inventing them is the closed loop starting on
    day one, so they are null and documented as null."""
    regime = one_case()["regime"]
    assert regime["Re"] is None
    assert regime["Ma"] is None
    assert regime["shedding_risk"] is None


# -- both forks ----------------------------------------------------------------


def test_the_foundation_fork_turbulence_file_is_read(one_case):
    """ESI v2512 has zero `momentumTransport` files at any depth and keys the model
    `RASModel`; the Foundation fork uses `momentumTransport` and keys it `model`. A
    harvester written against either name alone indexes nothing on the other."""
    row = one_case(fork="foundation", turbulence=("RAS", "kEpsilon"))
    assert row["turbulence"] == {"simulation_type": "RAS", "model": "kEpsilon"}


def test_a_case_with_no_turbulence_file_is_null_on_both_members(one_case):
    row = one_case(turbulence=None)
    assert row["turbulence"] == {"simulation_type": None, "model": None}


def test_a_turbulence_file_with_no_model_keeps_the_simulation_type(corpus, tmp_path):
    """179 cases are `laminar`, and a laminar file names no model at all."""
    tree = tmp_path / "tutorials"
    case = make_case(tree / "family" / "laminarCase", turbulence=None)
    dictionary(case / "constant" / "turbulenceProperties", "simulationType      laminar;\n")
    rows, _ = corpus.harvest_tutorials(tree)
    assert rows[0]["turbulence"] == {"simulation_type": "laminar", "model": None}


# -- multi-region --------------------------------------------------------------


def test_multi_region_turbulence_is_recorded_per_region(one_case):
    """None of the 18 multi-region cases carries a case-level `turbulenceProperties`;
    16 of them keep one per region under `constant/<region>/`. Read only at the case
    level, as the spec prescribes, every one of them indexes with turbulence nulled
    out -- which is a different claim from "the tutorial does not say"."""
    row = one_case(
        turbulence=None,
        regions={"topAir": ("RAS", "kEpsilon"), "bottomWater": ("RAS", "kOmegaSST")},
    )
    assert row["turbulence"] == {"simulation_type": None, "model": None}
    assert row["regions"] == {
        "topAir": {"simulation_type": "RAS", "model": "kEpsilon"},
        "bottomWater": {"simulation_type": "RAS", "model": "kOmegaSST"},
    }


def test_a_single_region_case_has_no_regions_field_to_read(one_case):
    assert one_case()["regions"] == {}


# -- boundary conditions -------------------------------------------------------


def test_bc_map_prefers_the_orig_time_directory(corpus, tmp_path):
    """369 cases keep initial conditions in `0.orig/` and 122 in `0/`; no case has
    both, so this preference never actually has to choose in the shipped tree. It is
    asserted anyway because a case copied out of one and run once will have both, and
    then `0/` is the solver output rather than the case specification."""
    tree = tmp_path / "tutorials"
    case = make_case(tree / "family" / "aCase", time_dir="0.orig")
    dictionary(
        case / "0" / "U",
        "boundaryField\n{\n    inlet\n    {\n        type            calculated;\n    }\n}\n",
    )
    rows, _ = corpus.harvest_tutorials(tree)
    assert rows[0]["bc_map"]["U"]["inlet"] == "fixedValue"


def test_bc_map_falls_back_to_the_zero_directory(one_case):
    assert one_case(time_dir="0")["bc_map"]["U"]["inlet"] == "fixedValue"


def test_bc_map_covers_every_field_in_the_directory(one_case):
    assert set(one_case()["bc_map"]) == {"U", "p"}


def test_a_case_with_no_time_directory_has_an_empty_bc_map(one_case):
    """65 cases have neither `0/` nor `0.orig/`, because their Allrun generates the
    fields."""
    assert one_case(time_dir=None)["bc_map"] == {}


# -- the walk ------------------------------------------------------------------


def test_a_symlinked_controldict_is_still_indexed(corpus, tmp_path):
    """7 of the 556 are relative symlinks into a shared `common/` directory
    (`basic/laplacianFoam/multiWorld2/*/system/controlDict`). A walk that skips links
    loses those cases and reports nothing wrong."""
    tree = tmp_path / "tutorials"
    shared = make_case(tree / "family" / "common", application="laplacianFoam")
    linked = tree / "family" / "slab1"
    (linked / "system").mkdir(parents=True)
    try:
        (linked / "system" / "controlDict").symlink_to(shared / "system" / "controlDict")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable here: {error}")
    rows, counts = corpus.harvest_tutorials(tree)
    assert {Path(row["path"]).name for row in rows} == {"common", "slab1"}
    assert counts["skipped"] == 0


def test_a_case_whose_controldict_will_not_read_is_skipped_and_counted(corpus, tmp_path):
    """A case that cannot be read is skipped and counted, never raised on. The stamp
    records `skipped` beside `indexed`, so a corpus that silently halved is visible."""
    tree = tmp_path / "tutorials"
    make_case(tree / "family" / "good")
    broken = tree / "family" / "broken" / "system"
    broken.mkdir(parents=True)
    (broken / "controlDict").write_bytes(b"\x00\xff not a dictionary \x00")
    rows, counts = corpus.harvest_tutorials(tree)
    assert [Path(row["path"]).name for row in rows] == ["good"]
    assert counts == {"indexed": 1, "skipped": 1}


def test_an_empty_tree_is_an_empty_index_and_not_an_error(corpus, tmp_path):
    tree = tmp_path / "nothing"
    tree.mkdir()
    rows, counts = corpus.harvest_tutorials(tree)
    assert rows == []
    assert counts == {"indexed": 0, "skipped": 0}


# -- R1: the rule the missing benchmark tier forces ----------------------------


def test_no_vendor_row_carries_a_reference_value(corpus, tmp_path):
    """Asserted over the whole built index rather than one row. Tutorials demonstrate
    features, not accuracy; a vendor row with a `reference_value` would be a measured
    answer that nobody measured, and the corpus has no benchmark tier to contradict
    it. The field is absent by construction, not by omission."""
    tree = tmp_path / "tutorials"
    for name in ("a", "b", "c"):
        make_case(tree / "family" / name)
    rows, _ = corpus.harvest_tutorials(tree)
    assert rows
    for row in rows:
        assert "reference_value" not in row
        assert "reference_value" not in row["regime"]
        assert row["verdict"] is None


# -- what gets written ---------------------------------------------------------


def test_build_writes_one_json_object_per_line(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tree = tmp_path / "tutorials"
    for name in ("a", "b"):
        make_case(tree / "family" / name)
    out = tmp_path / "corpus"

    stamp = corpus.build(tree, out)

    index = (out / "tutorials.index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index) == 2
    for line in index:
        row = json.loads(line)
        assert row["tier"] == "vendor"
        assert row["of_version"] == "v2512"
        assert row["of_fork"] == "esi"
    assert stamp["counts"]["tutorials"] == {"indexed": 2, "skipped": 0}
    assert stamp["schema_version"] == corpus.SCHEMA_VERSION
    assert stamp["of_version"] == "v2512"
    assert stamp["built_at"]


def test_the_stamp_is_written_beside_the_index(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tree = tmp_path / "tutorials"
    make_case(tree / "family" / "a")
    out = tmp_path / "corpus"

    corpus.build(tree, out)

    written = json.loads((out / "corpus.stamp.json").read_text(encoding="utf-8"))
    assert written["of_version"] == "v2512"
    assert written["counts"]["tutorials"]["indexed"] == 1


def test_an_unset_version_is_recorded_as_unknown_and_not_guessed(corpus, tmp_path, monkeypatch):
    """`of_version` and `of_fork` are recorded as `unknown`, never guessed. A stale
    index is worse than no index because it looks authoritative, and `search.py`
    rebuilds on a version mismatch -- which needs the recorded version to be honest."""
    monkeypatch.delenv("WM_PROJECT_VERSION", raising=False)
    tree = tmp_path / "tutorials"
    make_case(tree / "family" / "a")
    out = tmp_path / "corpus"

    stamp = corpus.build(tree, out)

    assert stamp["of_version"] == "unknown"
    row = json.loads((out / "tutorials.index.jsonl").read_text().splitlines()[0])
    assert row["of_version"] == "unknown"


def test_a_rebuild_replaces_the_index_rather_than_appending(corpus, tmp_path):
    """The index is a rebuild artifact, not a log. Appending would leave a case that
    was deleted upstream in the index forever, and a query would then hand back a
    path that is not there."""
    tree = tmp_path / "tutorials"
    make_case(tree / "family" / "a")
    out = tmp_path / "corpus"

    corpus.build(tree, out)
    corpus.build(tree, out)

    assert len((out / "tutorials.index.jsonl").read_text().splitlines()) == 1


def test_one_case_reached_by_two_paths_is_indexed_once(corpus, tmp_path):
    """`mesh/foamyHexMesh/straightDuctImplicit` is a symlinked *directory* pointing at
    `incompressible/porousSimpleFoam/straightDuctImplicit`, so one case answers to two
    paths. It is the reason the tree counts 556 cases and not 557: `find -L` and
    `find` disagree by exactly this one, and Python 3.12 happens to side with 556.

    That is a default rather than a decision -- 3.13 moved it behind
    `recurse_symlinks` -- so the row is de-duplicated by resolved path instead. Indexed
    twice, one precedent sits in the corpus twice, and the value distribution across
    hits counts it twice too.
    """
    tree = tmp_path / "tutorials"
    real = make_case(tree / "incompressible" / "straightDuct", application="porousSimpleFoam")
    alias = tree / "mesh" / "straightDuct"
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable here: {error}")

    rows, counts = corpus.harvest_tutorials(tree)

    assert counts["indexed"] == 1, [row["path"] for row in rows]
    assert len(corpus.case_roots(tree)) == 1


def test_case_roots_never_returns_the_same_directory_twice(corpus, tmp_path):
    """The de-duplication holds with no symlinks involved, which is the part that can
    be checked on any platform."""
    tree = tmp_path / "tutorials"
    for name in ("a", "b", "c"):
        make_case(tree / "family" / name)
    roots = corpus.case_roots(tree)
    assert len(roots) == 3
    assert len({root.resolve() for root in roots}) == 3


# =============================================================================
# The earned harvest
# =============================================================================
#
# There is no real study bundle to test against: `studies/` is empty, no `.reynolds`
# exists anywhere in the repository, and the hosted path cannot be reached from here.
# The design document says so plainly and so does this file.
#
# What can be done instead of pretending otherwise: build every fixture with
# `study_state.py`'s **own** writers -- `record`, `set_phase`, `record_rung` -- rather
# than by hand-writing JSON that matches what they are believed to produce. A
# hand-rolled fixture agrees with the harvester and with nothing else, and both can be
# wrong together. Going through the writer means the bundle in the test is the bundle
# the system writes, and a change to either shape breaks these tests rather than
# silently parting company from them.


@pytest.fixture(scope="module")
def study_state():
    """The module that defines the bundle, loaded the way the toolbox tests load one."""
    spec = importlib.util.spec_from_file_location("study_state", TOOLBOX / "study_state.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_study(
    root: Path,
    study_state,
    *,
    phases: dict[str, str] | None = None,
    case_name: str | None = "case",
    cases: tuple[str, ...] = ("case",),
    artifacts: tuple[tuple[str, str], ...] = (("mesh-full", "renders/mesh.png"),),
    rungs: tuple[dict, ...] = (),
    **case_kwargs,
) -> Path:
    """A study bundle, written through `study_state.py` rather than by hand."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".reynolds").mkdir(exist_ok=True)
    for name in cases:
        make_case(root / name, **case_kwargs)

    for kind, relative in artifacts:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not really a png")
        study_state.record(kind, target, root=root, case=case_name or "")

    for rung in rungs:
        study_state.record_rung(root=root, case=case_name or "", **rung)

    if phases is not None:
        for name, status in phases.items():
            study_state.set_phase(name, status, root=root, case=case_name or "")
    return root


@pytest.fixture
def one_study(corpus, study_state, tmp_path):
    """Harvest a single study and hand back its row."""
    counter = iter(range(1000))

    def build(**kwargs):
        work = tmp_path / f"work-{next(counter)}"
        make_study(work / "a-study", study_state, **kwargs)
        rows, counts = corpus.harvest_studies(work)
        assert counts["skipped"] == 0, counts
        assert len(rows) == 1, rows
        return rows[0]

    return build


ALL_DONE = {name: "done" for name in
            ("geometry", "preview", "mesh", "checkMesh", "probe", "solve",
             "reconstruct", "render", "animate", "report")}


# -- R2: what keeps the tiers apart --------------------------------------------


def test_every_earned_row_is_labelled_earned(one_study):
    """R2. An earned row is never promoted, merged into, or presented as equivalent to
    another tier. The label is how `search.py` can report the tier of every hit, which
    is the only thing standing between a mixed result set and a corpus that quietly
    treats its own output as evidence."""
    assert one_study(phases=ALL_DONE)["tier"] == "earned"


def test_an_earned_row_carries_its_study_id(one_study):
    row = one_study(phases=ALL_DONE)
    assert row["study_id"] == "a-study"
    assert row["path"].endswith("a-study")


def test_the_two_tiers_do_not_share_a_row_shape_by_accident(corpus, study_state, tmp_path):
    """Both tiers answer the same queries, so the fields a query ranks on have to mean
    the same thing in both. What must differ is `tier`, and `verdict` -- which is null
    for every vendor row by construction and is the whole point of an earned one."""
    tutorials = tmp_path / "tutorials"
    make_case(tutorials / "family" / "aCase")
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE)

    vendor, _ = corpus.harvest_tutorials(tutorials)
    earned, _ = corpus.harvest_studies(work)

    shared = set(vendor[0]) & set(earned[0])
    for field in ("path", "tier", "solver", "runs", "turbulence", "regime", "mesh_type",
                  "bc_map", "verdict", "provenance"):
        assert field in shared, f"{field} is not in both tiers"
    assert vendor[0]["verdict"] is None
    assert earned[0]["verdict"] is not None
    assert set(earned[0]) - set(vendor[0]) == {
        "study_id", "case", "cases", "artifacts", "rungs", "notes",
    }
    assert not set(vendor[0]) - set(earned[0])


# -- the verdict, derived and not invented -------------------------------------


def test_a_study_whose_phases_are_all_done_reads_completed(one_study):
    assert one_study(phases=ALL_DONE)["verdict"] == "completed"


def test_a_skipped_phase_does_not_stop_a_study_being_complete(one_study):
    """`study_state.py`: a mesh-only study stops after `checkMesh` and a case with no
    moving parts never animates, and skipping is recorded as `skipped` rather than
    left `pending` so that "not done" and "not wanted" do not read the same. A
    verdict that ignored that distinction would report every mesh-only study as
    unfinished."""
    phases = dict(ALL_DONE, animate="skipped", report="skipped")
    assert one_study(phases=phases)["verdict"] == "completed"


def test_a_study_with_a_failed_phase_says_failed(one_study):
    """`failed` outranks everything else in the table. A study that meshed, solved and
    rendered but failed its report is not a study to seed from without knowing that."""
    phases = dict(ALL_DONE, report="failed")
    assert one_study(phases=phases)["verdict"] == "failed"


def test_a_failure_outranks_a_phase_still_running(one_study):
    phases = dict(ALL_DONE, solve="failed", render="running")
    assert one_study(phases=phases)["verdict"] == "failed"


def test_a_study_still_running_says_running(one_study):
    phases = dict(ALL_DONE, render="running")
    assert one_study(phases=phases)["verdict"] == "running"


def test_a_study_that_stopped_partway_says_incomplete(one_study):
    """The common shape: a session ended mid-study, so the later phases are still
    `pending`. Not a failure, and not something finished."""
    phases = dict(ALL_DONE, animate="pending", report="pending")
    assert one_study(phases=phases)["verdict"] == "incomplete"


def test_a_study_with_nothing_but_skipped_phases_is_not_completed(one_study):
    """Every phase skipped is not a completed study, it is a study that did nothing.
    `completed` requires at least one phase to have actually been done."""
    phases = {name: "skipped" for name in ALL_DONE}
    assert one_study(phases=phases)["verdict"] == "incomplete"


def test_a_study_with_no_phases_file_has_a_null_verdict(corpus, study_state, tmp_path):
    """Null, not `unknown` and not `incomplete`. A study that recorded no phase table
    has said nothing about how far it got, and that is a different fact from having
    said it got nowhere -- which is what `load_phases` returning a blank all-pending
    table would otherwise turn it into."""
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=None)
    assert not (work / "a-study" / ".reynolds" / "phases.json").exists()
    rows, _ = corpus.harvest_studies(work)
    assert rows[0]["verdict"] is None


def test_an_unrecognised_phase_status_is_not_swallowed(corpus, study_state, tmp_path):
    """A status this was not written against must not read as `completed`. The verdict
    is a reading of the table, and a table saying something unrecognised is a reason
    to withhold a verdict rather than to pick the nearest one."""
    work = tmp_path / "work"
    study = make_study(work / "a-study", study_state, phases=ALL_DONE)
    table = json.loads((study / ".reynolds" / "phases.json").read_text(encoding="utf-8"))
    table["phases"][3]["status"] = "somethingLater"
    (study / ".reynolds" / "phases.json").write_text(json.dumps(table), encoding="utf-8")
    rows, _ = corpus.harvest_studies(work)
    assert rows[0]["verdict"] == "unrecognised"


# -- the artifacts the manifest records ----------------------------------------


def test_the_artifacts_are_counted_by_kind(one_study):
    row = one_study(
        phases=ALL_DONE,
        artifacts=(
            ("mesh-full", "renders/mesh.png"),
            ("mesh-closeup", "renders/close.png"),
            ("residuals", "renders/res.png"),
            ("mesh-full", "renders/mesh2.png"),
        ),
    )
    assert row["artifacts"] == {"mesh-closeup": 1, "mesh-full": 2, "residuals": 1}


def test_a_study_with_no_manifest_has_no_artifacts_and_is_still_a_row(corpus, study_state, tmp_path):
    """A study that produced nothing is still a precedent: its case, its solver and
    its verdict are all readable, and "this was tried and got this far" is exactly
    what the earned tier exists to carry."""
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE, artifacts=())
    rows, _ = corpus.harvest_studies(work)
    assert rows[0]["artifacts"] == {}
    assert rows[0]["verdict"] == "completed"


def test_a_manifest_line_that_will_not_parse_does_not_take_the_study_down(
    corpus, study_state, tmp_path
):
    """`study_state.artifacts` skips a line that will not parse, because a manifest is
    written by several scripts and a job killed mid-write leaves half a line. Reading
    it through that function inherits the behaviour rather than re-deciding it."""
    work = tmp_path / "work"
    study = make_study(work / "a-study", study_state, phases=ALL_DONE)
    manifest = study / ".reynolds" / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "residuals", "path": "renders/hal\n')
    rows, counts = corpus.harvest_studies(work)
    assert counts["skipped"] == 0
    assert rows[0]["artifacts"] == {"mesh-full": 1}


# -- rung evidence -------------------------------------------------------------


def test_rung_evidence_is_carried_with_its_known_answer(one_study):
    """The one earned signal that is not purely self-referential. A ladder rung's
    answer never comes from another solve -- it comes from Archimedes, the ITTC-57
    line, Hagen-Poiseuille -- so a recorded rung carries the measured value *and* the
    known answer it was set against.

    That does not make the earned tier a benchmark tier, and it is not a substitute
    for one. It is the only thing in the corpus that was checked against something
    outside the corpus, which is worth keeping legible.
    """
    row = one_study(
        phases=ALL_DONE,
        rungs=(
            {"number": 1, "status": "pass", "class_key": "external-2d",
             "name": "buoyancy", "value": 0.998, "known": "Archimedes"},
            {"number": 2, "status": "fail", "class_key": "external-2d",
             "name": "flat-plate drag", "value": 0.41, "known": "ITTC-57"},
        ),
    )
    assert row["rungs"] == [
        {"class": "external-2d", "rung": 1, "name": "buoyancy", "status": "pass",
         "value": 0.998, "known": "Archimedes"},
        {"class": "external-2d", "rung": 2, "name": "flat-plate drag", "status": "fail",
         "value": 0.41, "known": "ITTC-57"},
    ]


def test_a_re_recorded_rung_appears_once(one_study):
    """`record_rung` replaces an earlier line for the same class and rung rather than
    sitting beside it, because re-recording is a correction. The index has to show one
    outcome per rung for the same reason."""
    row = one_study(
        phases=ALL_DONE,
        rungs=(
            {"number": 1, "status": "fail", "class_key": "external-2d", "name": "buoyancy"},
            {"number": 1, "status": "pass", "class_key": "external-2d", "name": "buoyancy"},
        ),
    )
    assert [(r["rung"], r["status"]) for r in row["rungs"]] == [(1, "pass")]


def test_a_study_with_no_rungs_carries_an_empty_list(one_study):
    """Climbing none of the rungs is a legitimate choice, per `ladder.py`. An empty
    list says that; it does not say the rungs failed."""
    assert one_study(phases=ALL_DONE)["rungs"] == []


# -- which case represents the study -------------------------------------------


def test_the_case_named_in_the_phase_table_is_the_one_read(corpus, study_state, tmp_path):
    """A study holds a primary case and a copy of it per attempt (`runs/01-coarse`,
    `runs/02-medium`), so "which case is this study" has several answers on disk and
    exactly one on record: `phases.json` carries the case, because `set_phase` is
    given it. Read from the record rather than guessed from the directory listing."""
    work = tmp_path / "work"
    study = work / "a-study"
    make_study(
        study, study_state, phases=ALL_DONE, case_name="runs/02-medium",
        cases=("case", "runs/01-coarse", "runs/02-medium"), application="pimpleFoam",
    )
    dictionary(study / "runs" / "02-medium" / "system" / "controlDict",
               "application     interFoam;\n")
    rows, _ = corpus.harvest_studies(work)
    assert rows[0]["case"] == "runs/02-medium"
    assert rows[0]["solver"]["executable"] == "interFoam"


def test_a_single_case_needs_no_record_to_be_unambiguous(corpus, study_state, tmp_path):
    """The one-case study is the common shape, and there the answer is not a guess:
    there is only one case to be."""
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=None, case_name=None,
               cases=("case",), application="simpleFoam")
    rows, _ = corpus.harvest_studies(work)
    assert rows[0]["case"] == "case"
    assert rows[0]["solver"]["executable"] == "simpleFoam"


def test_several_cases_and_no_record_leaves_the_case_null(corpus, study_state, tmp_path):
    """Four cases and nothing saying which is the study's is not a reason to pick the
    first alphabetically. The row still lists what was found, so nothing is lost and a
    reader can see why the case fields are null."""
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=None, case_name=None,
               cases=("case", "runs/01", "runs/02"))
    rows, _ = corpus.harvest_studies(work)
    row = rows[0]
    assert row["case"] is None
    assert row["solver"] == {"executable": None, "module": None}
    assert sorted(row["cases"]) == ["case", "runs/01", "runs/02"]


def test_the_cases_found_are_always_listed(one_study):
    assert one_study(phases=ALL_DONE)["cases"] == ["case"]


def test_a_named_case_that_is_not_there_falls_back_rather_than_inventing(
    corpus, study_state, tmp_path
):
    """A phase table can name a case that has since been deleted or renamed. That is a
    stale record, not a case, and it must not become a path in the index that is not
    there."""
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE, case_name="runs/gone",
               cases=("case",))
    rows, _ = corpus.harvest_studies(work)
    assert rows[0]["case"] == "case"


# -- the walk over /work -------------------------------------------------------


def test_a_directory_with_no_reynolds_state_is_not_a_study(corpus, study_state, tmp_path):
    """`/work` holds whatever the agent put there -- geometry, scratch directories, a
    downloaded archive. A study is a directory carrying `.reynolds/`, and nothing else
    is, so nothing else is indexed."""
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE)
    (work / "geometry").mkdir(parents=True)
    (work / "geometry" / "wing.stl").write_text("solid\nendsolid\n")
    make_case(work / "a-loose-case")

    rows, counts = corpus.harvest_studies(work)

    assert [row["study_id"] for row in rows] == ["a-study"]
    assert counts == {"indexed": 1, "skipped": 0}


def test_several_studies_each_get_a_row(corpus, study_state, tmp_path):
    work = tmp_path / "work"
    for name in ("m6-transonic", "cylinder-shedding", "heat-sink"):
        make_study(work / name, study_state, phases=ALL_DONE)
    rows, counts = corpus.harvest_studies(work)
    assert sorted(row["study_id"] for row in rows) == [
        "cylinder-shedding", "heat-sink", "m6-transonic",
    ]
    assert counts["indexed"] == 3


def test_an_empty_work_directory_is_an_empty_index(corpus, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    rows, counts = corpus.harvest_studies(work)
    assert rows == []
    assert counts == {"indexed": 0, "skipped": 0}


def test_a_missing_work_directory_is_not_an_error(corpus, tmp_path):
    """The first session on a fresh instance has no `/work/<study>` yet, and asking
    for the earned tier then is a reasonable thing to do."""
    rows, counts = corpus.harvest_studies(tmp_path / "not-created-yet")
    assert rows == []
    assert counts == {"indexed": 0, "skipped": 0}


# -- what gets written ---------------------------------------------------------


def test_build_writes_both_tiers_and_stamps_both(corpus, study_state, tmp_path, monkeypatch):
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tutorials = tmp_path / "tutorials"
    make_case(tutorials / "family" / "aCase")
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE)
    out = tmp_path / "corpus"

    stamp = corpus.build(tutorials, out, work=work)

    vendor = [json.loads(l) for l in (out / corpus.TUTORIALS_INDEX).read_text().splitlines()]
    earned = [json.loads(l) for l in (out / corpus.STUDIES_INDEX).read_text().splitlines()]
    assert [row["tier"] for row in vendor] == ["vendor"]
    assert [row["tier"] for row in earned] == ["earned"]
    assert stamp["counts"]["tutorials"] == {"indexed": 1, "skipped": 0}
    assert stamp["counts"]["studies"] == {"indexed": 1, "skipped": 0}


def test_the_two_indexes_are_separate_files(corpus, study_state, tmp_path):
    """R2 again, at the level of the filesystem. Two tiers in one file would make
    "never merged" a property of the reader rather than of the corpus."""
    tutorials = tmp_path / "tutorials"
    make_case(tutorials / "family" / "aCase")
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE)
    out = tmp_path / "corpus"

    corpus.build(tutorials, out, work=work)

    assert corpus.TUTORIALS_INDEX != corpus.STUDIES_INDEX
    assert (out / corpus.TUTORIALS_INDEX).exists()
    assert (out / corpus.STUDIES_INDEX).exists()


def test_building_with_no_work_directory_still_writes_the_vendor_tier(
    corpus, tmp_path, monkeypatch
):
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tutorials = tmp_path / "tutorials"
    make_case(tutorials / "family" / "aCase")
    out = tmp_path / "corpus"

    stamp = corpus.build(tutorials, out, work=tmp_path / "no-work")

    assert stamp["counts"]["tutorials"]["indexed"] == 1
    assert stamp["counts"]["studies"] == {"indexed": 0, "skipped": 0}
    assert (out / corpus.STUDIES_INDEX).read_text() == ""


def test_no_earned_row_carries_a_reference_value(one_study):
    """R1 is a vendor rule, and the earned tier does not get a reference value either:
    a study's own result is not a reference, it is the thing a reference would check.
    The verdict is what an earned row carries instead, and it says how far the study
    got rather than whether it was right."""
    row = one_study(phases=ALL_DONE)
    assert "reference_value" not in json.dumps(row)


# -- keys that are not identifiers ---------------------------------------------


FV_SCHEMES = BANNER + """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}
"""


@pytest.mark.parametrize(
    "key,expected",
    [
        ("div(phi,U)", "bounded Gauss linearUpwind grad(U)"),
        ("div(phi,k)", "bounded Gauss limitedLinear 1"),
        ("div(((rho*nuEff)*dev2(T(grad(U)))))", "Gauss linear"),
        ("grad(p)", "Gauss linear"),
    ],
)
def test_a_key_that_is_an_expression_is_read(corpus, key, expected):
    """`fvSchemes` is keyed by the term being discretised, not by an identifier:
    `div(phi,U)` in **299 cases**, `div(phi,k)` in 209,
    `div(((rho*nuEff)*dev2(T(grad(U)))))` in 186. A key pattern restricted to
    identifiers reads none of them -- and `div(phi,U)` is the design document's own
    example of a keyword query, so the reader could not have answered the query the
    document asks for."""
    assert corpus.entries(corpus.strip_comments(FV_SCHEMES))[key] == expected


def test_a_quoted_regular_expression_key_is_read(corpus):
    """`fvSolution` keys its solver blocks by regular expressions over field names:
    `".*"` 130 times, `"pa.*"` 72, `"Ua.*"` 70."""
    text = BANNER + 'relaxationFactors\n{\n    ".*"        0.9;\n    "pa.*"      0.3;\n}\n'
    found = corpus.entries(corpus.strip_comments(text))
    assert found['".*"'] == "0.9"
    assert found['"pa.*"'] == "0.3"


def test_broadening_the_key_did_not_start_matching_list_bodies(corpus):
    """The risk of a permissive key: `blockMeshDict` is mostly list bodies, and a
    pattern that matched inside one would invent entries on 441 cases. Nothing in a
    list carries a semicolon of its own, and the `);` that closes one has no value."""
    text = BANNER + """convertToMeters 0.001;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)
);

edges
(
);

mergePatchPairs
(
);
"""
    found = corpus.entries(corpus.strip_comments(text))
    assert found == {"convertToMeters": "0.001", "version": "2.0", "format": "ascii",
                     "class": "dictionary", "object": "controlDict"}


def test_an_include_directive_is_not_an_entry(corpus):
    """`#include` and `#includeEtc` carry no semicolon and so would not match anyway.
    The leading `#` is excluded explicitly so that the reason is stated rather than
    incidental."""
    text = BANNER + '#include "initialConditions"\n#includeEtc "caseDicts/setConstraintTypes"\n'
    found = corpus.entries(corpus.strip_comments(text))
    assert "#include" not in found
    assert "#includeEtc" not in found


def test_the_block_scoped_read_still_picks_the_right_default(corpus):
    """Four blocks in that file carry a `default`. The ddt one is the one that says
    whether the case is steady, and `block()` is what keeps them apart -- broadening
    the key pattern must not have changed which `default` is found where."""
    ddt = corpus.block(corpus.strip_comments(FV_SCHEMES), "ddtSchemes")
    grad = corpus.block(corpus.strip_comments(FV_SCHEMES), "gradSchemes")
    div = corpus.block(corpus.strip_comments(FV_SCHEMES), "divSchemes")
    assert corpus.entries(ddt)["default"] == "steadyState"
    assert corpus.entries(grad)["default"] == "Gauss linear"
    assert corpus.entries(div)["default"] == "none"


# =============================================================================
# From the code review
# =============================================================================


def test_an_unset_tutorials_variable_does_not_index_the_working_directory(
    corpus, tmp_path, monkeypatch, capsys
):
    """`Path(os.environ.get("FOAM_TUTORIALS", ""))` is `Path(".")`, not an empty path.
    `Path("")` normalises to the current directory, which is truthy and is a directory,
    so the guard `if not args.tutorials or not args.tutorials.is_dir()` could never
    fire and its message was unreachable.

    With the variable unset, `build` therefore walked the working directory, found
    whatever `system/controlDict` happened to be under it, and reported success. An
    index of the wrong tree that says `indexed 1` is worse than a refusal.
    """
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)
    monkeypatch.chdir(tmp_path)
    make_case(tmp_path / "a-stray-case")

    code = corpus.main(["build", "--out", str(tmp_path / "corpus")])

    assert code == 1
    assert "FOAM_TUTORIALS" in capsys.readouterr().err
    assert not (tmp_path / "corpus" / corpus.TUTORIALS_INDEX).exists()


def test_a_tutorials_path_that_is_not_there_is_refused(corpus, tmp_path, capsys):
    code = corpus.main(["build", "--tutorials", str(tmp_path / "nope"),
                        "--out", str(tmp_path / "corpus")])
    assert code == 1
    assert "nope" in capsys.readouterr().err


def test_the_stamp_records_which_tree_it_was_built_from(corpus, tmp_path, monkeypatch):
    """Without it, switching `--tutorials` or fixing `$FOAM_TUTORIALS` mid-session is
    invisible to the staleness check: the stamp matches on version and schema, so the
    index built from the wrong tree keeps answering. The tree is the one thing the
    stamp described nothing about."""
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tree = tmp_path / "tutorials"
    make_case(tree / "family" / "a")

    stamp = corpus.build(tree, tmp_path / "corpus", work=tmp_path / "work")

    assert Path(stamp["tutorials"]) == tree.resolve()
    assert Path(stamp["work"]) == (tmp_path / "work").resolve()
    written = json.loads((tmp_path / "corpus" / corpus.STAMP).read_text(encoding="utf-8"))
    assert Path(written["tutorials"]) == tree.resolve()


# -- physicalProperties is not proof of incompressibility ----------------------


def test_physical_properties_is_read_rather_than_taken_as_incompressible(
    corpus, tmp_path
):
    """The Foundation fork's `physicalProperties` replaced *both*
    `transportProperties` and `thermophysicalProperties`, so its mere presence says
    nothing about compressibility. Treating it as proof of incompressible stated a
    fact that had not been read -- the one thing this module says it never does.

    Read instead: a `thermoType` in it is a thermophysical model and therefore
    compressible; a `nu` or a `transportModel` is the incompressible form.
    """
    tree = tmp_path / "tutorials"
    compressible = make_case(tree / "family" / "compressible", properties=None)
    dictionary(compressible / "constant" / "physicalProperties",
               "thermoType\n{\n    type            hePsiThermo;\n"
               "    equationOfState perfectGas;\n}\n")
    incompressible = make_case(tree / "family" / "incompressible", properties=None)
    dictionary(incompressible / "constant" / "physicalProperties",
               "viscosityModel  constant;\nnu  [0 2 -1 0 0 0 0] 1e-05;\n")

    rows, _ = corpus.harvest_tutorials(tree)
    by_name = {Path(row["path"]).name: row for row in rows}

    assert by_name["compressible"]["regime"]["compressible"] is True
    assert by_name["incompressible"]["regime"]["compressible"] is False


def test_a_physical_properties_that_says_neither_is_null(corpus, tmp_path):
    """Null rather than a coin toss. A file this reader cannot place is a file it has
    not read, and the index says so.

    This is not hypothetical, and it is not only a Foundation-fork concern. The single
    `physicalProperties` in the whole of v2512 belongs to
    `electromagnetics/electrostaticFoam/chargedWire`, and it holds `epsilon0` and `k` --
    a vacuum permittivity and a mobility. It is **electrostatics, with no fluid
    transport in it at all**, and the old reading called it incompressible purely
    because a file of that name existed. So the guess was already wrong on the tree
    being indexed, on the one case that could expose it.
    """
    tree = tmp_path / "tutorials"
    case = make_case(tree / "family" / "chargedWire", properties=None)
    dictionary(case / "constant" / "physicalProperties",
               "epsilon0  epsilon0 [-1 -3 4 0 0 2 0] 8.85419e-12;\n"
               "k         k        [ 0  0 0 0 0 0 0] 1.6e-4;\n")
    rows, _ = corpus.harvest_tutorials(tree)
    assert rows[0]["regime"]["compressible"] is None
    assert rows[0]["regime"]["class"] == "steady"


def test_thermophysical_and_transport_properties_still_decide_directly(corpus, tmp_path):
    """118 cases carry `thermophysicalProperties` and 333 `transportProperties`, and
    those two names are unambiguous on both forks."""
    tree = tmp_path / "tutorials"
    make_case(tree / "family" / "thermo", properties="thermophysicalProperties")
    make_case(tree / "family" / "transport", properties="transportProperties")
    rows, _ = corpus.harvest_tutorials(tree)
    by_name = {Path(row["path"]).name: row for row in rows}
    assert by_name["thermo"]["regime"]["compressible"] is True
    assert by_name["transport"]["regime"]["compressible"] is False
