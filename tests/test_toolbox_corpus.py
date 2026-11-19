"""`corpus.py`: the dictionary reader, the vendor harvest and the earned harvest.

Fixtures are miniature case trees; where a count is quoted it was counted off
`opencfd/openfoam-default:2512`. Study fixtures are written through `study_state.py`'s
own writers, so a change to the bundle's shape breaks these tests rather than parting
company with them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str, as_name: str):
    spec = importlib.util.spec_from_file_location(as_name, TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus():
    return load("corpus", "toolbox_corpus")


@pytest.fixture(scope="module")
def study_state():
    return load("study_state", "study_state")


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


def read_text(corpus, text: str) -> dict:
    return corpus.entries(corpus.strip_comments(text))


# -- the reader ----------------------------------------------------------------


def test_comments_do_not_leak_into_values(corpus):
    """`drippingChair` writes `simulationType laminar; // RAS;`, and four banners carry
    `https://`, which a `//` pass run first would leave unterminated."""
    found = read_text(corpus, BANNER + "simulationType      laminar; // RAS;\n")
    assert found["simulationType"] == "laminar" and "RAS" not in found
    with_scheme = BANNER.replace("www.openfoam.com", "https://www.openfoam.com")
    assert read_text(corpus, with_scheme + "application simpleFoam;\n")["application"] == \
        "simpleFoam"


@pytest.mark.parametrize("spacing", ["application interFoam;", "application\tinterFoam;",
                                     "  application       interFoam;"])
def test_spacing_does_not_change_the_value(corpus, spacing):
    assert read_text(corpus, spacing)["application"] == "interFoam"


def test_a_key_one_block_down_is_found_and_the_first_occurrence_wins(corpus):
    """ESI keeps the model inside the block that names the simulation type."""
    text = BANNER + "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n}\n" \
                    "startFrom latestTime;\nstartFrom startTime;\n"
    found = read_text(corpus, text)
    assert (found["RASModel"], found["startFrom"]) == ("kEpsilon", "latestTime")


@pytest.mark.parametrize("key,expected", [
    ("div(phi,U)", "bounded Gauss linearUpwind grad(U)"),
    ("div(((rho*nuEff)*dev2(T(grad(U)))))", "Gauss linear"),
    ('".*"', "0.9"),
])
def test_keys_that_are_not_identifiers_are_read(corpus, key, expected):
    """`fvSchemes` is keyed by terms, `fvSolution` by quoted regexes (`".*"` 130 times)."""
    text = (BANNER + "divSchemes\n{\n    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
            "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}\n"
            'relaxationFactors\n{\n    ".*"        0.9;\n}\n')
    assert read_text(corpus, text)[key] == expected


def test_absent_and_unreadable_both_read_as_none(corpus, tmp_path):
    path = tmp_path / "controlDict"
    path.write_text(BANNER + "application  simpleFoam;\n", encoding="utf-8")
    assert corpus.entry(path, "application") == "simpleFoam"
    assert corpus.entry(path, "RASModel") is None
    assert corpus.entry(tmp_path / "missing", "application") is None
    assert corpus.read(tmp_path / "missing") == ""


FIELD_U = BANNER + """dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet
    {
        type            codedFixedValue;
        value           uniform (10 0 0);
        codeOptions
        {
            type        ignored;
        }
    }
    outlet
    {
        type            zeroGradient;
    }
    "(left|right)"
    {
        type            noSlip;
    }
}
"""


def test_boundary_types_maps_each_patch_and_nothing_else(corpus):
    """Scoped to `boundaryField`, so `FoamFile` is not a patch; a nested block does not
    swallow the patches after it; a regex patch group stays whole."""
    assert corpus.boundary_types(corpus.strip_comments(FIELD_U)) == {
        "inlet": "codedFixedValue", "outlet": "zeroGradient", "(left|right)": "noSlip",
    }


# -- the vendor harvest --------------------------------------------------------


def dictionary(path: Path, body: str) -> Path:
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
    properties: str | None = "transportProperties",
    time_dir: str | None = "0.orig",
    regions: dict[str, tuple[str, str]] | None = None,
) -> Path:
    """A miniature case: a handful of files, each one carrying a measured fact."""
    root.mkdir(parents=True, exist_ok=True)
    control = "startTime  0;\n"
    if application is not None:
        control = f"application     {application};\n" + control
    dictionary(root / "system" / "controlDict", control)
    if turbulence is not None:
        simulation, model = turbulence
        key = "model" if fork == "foundation" else f"{simulation}Model"
        name = "momentumTransport" if fork == "foundation" else "turbulenceProperties"
        dictionary(root / "constant" / name,
                   f"simulationType {simulation};\n{simulation}\n{{\n    {key} {model};\n}}\n")
    for name in mesh:
        dictionary(root / "system" / name, "scale   0.001;\n")
    if geometry:
        (root / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
        (root / "constant" / "triSurface" / "body.stl").write_text("solid body\nendsolid\n")
    if ddt is not None:
        # gradSchemes first: every schemes block carries a `default`.
        dictionary(root / "system" / "fvSchemes",
                   "gradSchemes\n{\n    default Gauss linear;\n}\n"
                   f"ddtSchemes\n{{\n    default {ddt};\n}}\n")
    if properties is not None:
        dictionary(root / "constant" / properties, "nu  [0 2 -1 0 0 0 0] 1e-05;\n")
    if time_dir is not None:
        dictionary(root / time_dir / "U",
                   "boundaryField\n{\n    inlet\n    {\n        type fixedValue;\n"
                   "        value uniform (10 0 0);\n    }\n}\n")
    for region, (simulation, model) in (regions or {}).items():
        dictionary(root / "constant" / "regionProperties", "regions ();\n")
        dictionary(root / "constant" / region / "turbulenceProperties",
                   f"simulationType {simulation};\n{simulation}\n{{\n"
                   f"    {simulation}Model {model};\n}}\n")
    return root


@pytest.fixture
def one_case(corpus, tmp_path):
    """Harvest a single case built with `make_case` kwargs, and hand back its row."""
    trees = iter(range(1000))

    def build(**kwargs):
        tree = tmp_path / f"tutorials-{next(trees)}"
        make_case(tree / "family" / "aCase", **kwargs)
        rows, counts = corpus.harvest_tutorials(tree)
        assert counts["skipped"] == 0 and len(rows) == 1, (counts, rows)
        return rows[0]

    return build


def test_a_minimal_case_indexes_its_fields(corpus, one_case):
    row = one_case()
    assert row["tier"] == "vendor"
    assert row["solver"] == {"executable": "simpleFoam", "module": None}
    assert row["runs"] is True
    assert row["turbulence"] == {"simulation_type": "RAS", "model": "kEpsilon"}
    assert row["mesh_type"] == "blockMesh"
    assert row["regime"]["class"] == "incompressible-steady"
    assert row["bc_map"] == {"U": {"inlet": "fixedValue"}}
    assert row["provenance"]["schema_version"] == corpus.SCHEMA_VERSION


def test_no_vendor_row_carries_a_reference_value_or_a_number_it_did_not_read(one_case):
    """R1. `Re`, `Ma` and `shedding_risk` need scales a tutorial does not reliably state."""
    row = one_case()
    assert "reference_value" not in json.dumps(row)
    assert row["verdict"] is None
    assert [row["regime"][k] for k in ("Re", "Ma", "shedding_risk")] == [None, None, None]


@pytest.mark.parametrize("application", ["snappyHexMesh", None])
def test_a_mesher_or_no_application_does_not_run(one_case, application):
    """18 cases name `snappyHexMesh` as their application; four name nothing."""
    assert one_case(application=application)["runs"] is False


@pytest.mark.parametrize("kwargs,expected", [
    ({"mesh": ("blockMeshDict", "snappyHexMeshDict")}, "snappyHexMesh"),
    ({"mesh": ("blockMeshDict",)}, "blockMesh"),
    ({"mesh": (), "geometry": True}, "surface"),
])
def test_mesh_type(one_case, kwargs, expected):
    """snappy first: 62 of the 64 snappy cases also carry a background `blockMeshDict`."""
    assert one_case(**kwargs)["mesh_type"] == expected


@pytest.mark.parametrize("ddt,steady", [
    ("steadyState", True), ("none", True), ("Euler", False), ("CrankNicolson 0.9", False),
    (None, None), ("someSchemeShippedLater", None),
])
def test_steadiness_is_read_off_ddtschemes(one_case, ddt, steady):
    """From the `ddtSchemes` block, not the `default` of the block before it. A scheme
    this does not know is null, not transient."""
    assert one_case(ddt=ddt)["regime"]["steady"] is steady


@pytest.mark.parametrize("properties,compressible", [
    ("thermophysicalProperties", True), ("transportProperties", False), (None, None),
])
def test_compressibility_is_read_off_the_properties_file(one_case, properties, compressible):
    assert one_case(properties=properties)["regime"]["compressible"] is compressible


def test_physical_properties_is_opened_rather_than_taken_as_incompressible(corpus, tmp_path):
    """The Foundation fork's `physicalProperties` replaced both other files."""
    tree = tmp_path / "tutorials"
    hot = make_case(tree / "family" / "compressible", properties=None)
    dictionary(hot / "constant" / "physicalProperties",
               "thermoType\n{\n    type hePsiThermo;\n}\n")
    cold = make_case(tree / "family" / "incompressible", properties=None)
    dictionary(cold / "constant" / "physicalProperties", "nu  1e-05;\n")
    rows, _ = corpus.harvest_tutorials(tree)
    by_name = {Path(row["path"]).name: row["regime"]["compressible"] for row in rows}
    assert by_name == {"compressible": True, "incompressible": False}


def test_both_forks_and_a_laminar_file_are_read(corpus, one_case, tmp_path):
    """ESI keys the model `RASModel`, the Foundation fork `model` in `momentumTransport`;
    179 cases are `laminar` and name no model."""
    assert one_case(fork="foundation")["turbulence"] == \
        {"simulation_type": "RAS", "model": "kEpsilon"}
    tree = tmp_path / "laminar"
    case = make_case(tree / "family" / "aCase", turbulence=None)
    dictionary(case / "constant" / "turbulenceProperties", "simulationType laminar;\n")
    rows, _ = corpus.harvest_tutorials(tree)
    assert rows[0]["turbulence"] == {"simulation_type": "laminar", "model": None}


def test_multi_region_turbulence_is_recorded_per_region(one_case):
    """16 of the 18 multi-region cases keep turbulence only under `constant/<region>/`."""
    row = one_case(turbulence=None,
                   regions={"topAir": ("RAS", "kEpsilon"), "bottomWater": ("LES", "WALE")})
    assert row["turbulence"] == {"simulation_type": None, "model": None}
    assert row["regions"] == {"topAir": {"simulation_type": "RAS", "model": "kEpsilon"},
                              "bottomWater": {"simulation_type": "LES", "model": "WALE"}}


def test_bc_map_prefers_0_orig_then_0_and_is_empty_without_either(corpus, one_case, tmp_path):
    tree = tmp_path / "tutorials"
    case = make_case(tree / "family" / "aCase", time_dir="0.orig")
    dictionary(case / "0" / "U", "boundaryField\n{\n    inlet\n    {\n        type calculated;\n"
                                 "    }\n}\n")
    rows, _ = corpus.harvest_tutorials(tree)
    assert rows[0]["bc_map"]["U"]["inlet"] == "fixedValue"
    assert one_case(time_dir="0")["bc_map"]["U"]["inlet"] == "fixedValue"
    assert one_case(time_dir=None)["bc_map"] == {}


def symlink(link: Path, target: Path, **kwargs) -> None:
    try:
        link.symlink_to(target, **kwargs)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable here: {error}")


def test_a_symlinked_controldict_is_a_case_and_a_symlinked_directory_is_not(corpus, tmp_path):
    """7 cases are a symlinked `controlDict`; one directory is a symlink to another case."""
    tree = tmp_path / "tutorials"
    shared = make_case(tree / "family" / "common", application="laplacianFoam")
    (tree / "family" / "slab1" / "system").mkdir(parents=True)
    symlink(tree / "family" / "slab1" / "system" / "controlDict",
            shared / "system" / "controlDict")
    (tree / "mesh").mkdir()
    symlink(tree / "mesh" / "alias", shared, target_is_directory=True)
    rows, counts = corpus.harvest_tutorials(tree)
    assert sorted(Path(row["path"]).name for row in rows) == ["common", "slab1"]
    assert counts == {"indexed": 2, "skipped": 0}


def test_a_case_whose_controldict_will_not_read_is_skipped_and_counted(corpus, tmp_path):
    tree = tmp_path / "tutorials"
    make_case(tree / "family" / "good")
    broken = tree / "family" / "broken" / "system"
    broken.mkdir(parents=True)
    (broken / "controlDict").write_bytes(b"\x00\xff not a dictionary \x00")
    rows, counts = corpus.harvest_tutorials(tree)
    assert [Path(row["path"]).name for row in rows] == ["good"]
    assert counts == {"indexed": 1, "skipped": 1}


def test_build_writes_the_index_and_the_stamp_and_replaces_rather_than_appends(
    corpus, tmp_path, monkeypatch
):
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tree = tmp_path / "tutorials"
    for name in ("a", "b"):
        make_case(tree / "family" / name)
    out = tmp_path / "corpus"
    corpus.build(tree, out)
    stamp = corpus.build(tree, out)
    rows = [json.loads(line) for line in (out / corpus.TUTORIALS_INDEX).read_text().splitlines()]
    assert [(row["tier"], row["of_version"], row["of_fork"]) for row in rows] == \
        [("vendor", "v2512", "esi")] * 2
    assert json.loads((out / corpus.STAMP).read_text()) == stamp
    assert stamp["counts"]["tutorials"] == {"indexed": 2, "skipped": 0}


def test_an_unset_tutorials_variable_does_not_index_the_working_directory(
    corpus, tmp_path, monkeypatch, capsys
):
    """`Path("")` is `.` and truthy, so as a default it indexed the working directory."""
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)
    monkeypatch.chdir(tmp_path)
    make_case(tmp_path / "a-stray-case")
    assert corpus.main(["build", "--out", str(tmp_path / "corpus")]) == 1
    assert "FOAM_TUTORIALS" in capsys.readouterr().err
    assert not (tmp_path / "corpus" / corpus.TUTORIALS_INDEX).exists()


# -- the earned harvest --------------------------------------------------------


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
    (root / ".reynolds").mkdir(parents=True, exist_ok=True)
    for name in cases:
        make_case(root / name, **case_kwargs)
    for kind, relative in artifacts:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not really a png")
        study_state.record(kind, target, root=root, case=case_name or "")
    for rung in rungs:
        study_state.record_rung(root=root, case=case_name or "", **rung)
    for name, status in (phases or {}).items():
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
        assert counts["skipped"] == 0 and len(rows) == 1, (counts, rows)
        return rows[0]

    return build


ALL_DONE = {name: "done" for name in
            ("geometry", "preview", "mesh", "checkMesh", "probe", "solve",
             "reconstruct", "render", "animate", "report")}


def test_an_earned_row_is_labelled_earned_and_carries_no_reference_value(one_study):
    """R2: never promoted or merged. A study's own result is not a reference."""
    row = one_study(phases=ALL_DONE)
    assert (row["tier"], row["study_id"]) == ("earned", "a-study")
    assert "reference_value" not in json.dumps(row)


@pytest.mark.parametrize("changes,verdict", [
    ({}, "completed"),
    ({"animate": "skipped", "report": "skipped"}, "completed"),
    ({"report": "failed"}, "failed"),
    ({"solve": "failed", "render": "running"}, "failed"),
    ({"report": "running"}, "running"),
    ({"animate": "pending", "report": "pending"}, "incomplete"),
    ({name: "skipped" for name in ALL_DONE}, "incomplete"),
])
def test_the_verdict_is_read_off_the_phase_table(one_study, changes, verdict):
    """How far a study got, not whether it was right. `skipped` is not `pending`, but a
    table of nothing but `skipped` is not a completed study."""
    assert one_study(phases=dict(ALL_DONE, **changes))["verdict"] == verdict


def test_no_phase_table_is_a_null_verdict_and_an_unknown_status_is_not_swallowed(
    corpus, study_state, tmp_path
):
    work = tmp_path / "work"
    make_study(work / "none", study_state, phases=None)
    study = make_study(work / "odd", study_state, phases=ALL_DONE)
    table = json.loads((study / ".reynolds" / "phases.json").read_text(encoding="utf-8"))
    table["phases"][3]["status"] = "somethingLater"
    (study / ".reynolds" / "phases.json").write_text(json.dumps(table), encoding="utf-8")
    rows, _ = corpus.harvest_studies(work)
    assert {row["study_id"]: row["verdict"] for row in rows} == \
        {"none": None, "odd": "unrecognised"}


def test_artifacts_are_counted_by_kind_and_rungs_carry_their_known_answer(one_study):
    """A rung is the one earned signal checked against something outside the corpus."""
    row = one_study(
        phases=ALL_DONE,
        artifacts=(("mesh-full", "renders/a.png"), ("residuals", "renders/r.png"),
                   ("mesh-full", "renders/b.png")),
        rungs=({"number": 1, "status": "pass", "class_key": "external-2d",
                "name": "buoyancy", "value": 0.998, "known": "Archimedes"},),
    )
    assert row["artifacts"] == {"mesh-full": 2, "residuals": 1}
    assert row["rungs"] == [{"class": "external-2d", "rung": 1, "name": "buoyancy",
                             "status": "pass", "value": 0.998, "known": "Archimedes"}]


@pytest.mark.parametrize("recorded,cases,expected", [
    ("runs/02", ("case", "runs/01", "runs/02"), "runs/02"),
    (None, ("case",), "case"),
    (None, ("case", "runs/01", "runs/02"), None),
    ("runs/gone", ("case",), "case"),
])
def test_which_case_represents_the_study(corpus, study_state, tmp_path, recorded, cases,
                                         expected):
    """The recorded case when it is there, else the only one, else null -- never
    whichever sorted first."""
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE if recorded else None,
               case_name=recorded, cases=cases)
    rows, _ = corpus.harvest_studies(work)
    assert rows[0]["case"] == expected
    assert rows[0]["cases"] == sorted(cases)


def test_only_directories_carrying_reynolds_state_are_studies(corpus, study_state, tmp_path):
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE)
    (work / "geometry").mkdir()
    make_case(work / "a-loose-case")
    rows, counts = corpus.harvest_studies(work)
    assert [row["study_id"] for row in rows] == ["a-study"]
    assert counts == {"indexed": 1, "skipped": 0}
    assert corpus.harvest_studies(tmp_path / "not-created-yet") == \
        ([], {"indexed": 0, "skipped": 0})


def test_build_writes_the_two_tiers_to_two_files(corpus, study_state, tmp_path):
    """R2 at the level of the filesystem."""
    tutorials = tmp_path / "tutorials"
    make_case(tutorials / "family" / "aCase")
    work = tmp_path / "work"
    make_study(work / "a-study", study_state, phases=ALL_DONE)
    out = tmp_path / "corpus"
    stamp = corpus.build(tutorials, out, work=work)
    for name, tier in ((corpus.TUTORIALS_INDEX, "vendor"), (corpus.STUDIES_INDEX, "earned")):
        rows = [json.loads(line) for line in (out / name).read_text().splitlines()]
        assert [row["tier"] for row in rows] == [tier]
    assert stamp["counts"]["studies"] == {"indexed": 1, "skipped": 0}
