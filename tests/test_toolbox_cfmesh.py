"""`cfmesh.py`: the other mesher, and the four traps that stand in front of it.

The finding this closes (F-53) is not that cfMesh is missing. It is on the image,
prebuilt, and it was on the image for the whole four-round Wigley replication that
spent its budget re-tuning snappyHexMesh layer parameters against an architecture
that erodes layers by design -- 88.6% coverage on snappy's first growth iteration,
66.7% by its fiftieth, printed as 66.7% with exit code 0. cfMesh reached 100.0% on
the same hull. The remedy was available and untaken, so what is tested here is the
part that makes taking it cheap: the dictionary that comes out, the requests that
are refused because cfMesh would accept them and then quietly not carry them out,
and the reading-back of a mesh that was meshed without the thing it was asked for.

None of this can run OpenFOAM, so nothing here meshes anything. What is checked is
what is checkable off the image: the text of the meshDict, the refusals, the STL,
the two text edits `prepare` makes, and the findings `check` reads out of a
hand-written polyMesh -- the same hand-written box `test_toolbox_layer_report.py`
measures coverage on, reused rather than rebuilt.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

from test_toolbox_layer_report import XS, YS, LAYERED, write_box

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def cfmesh():
    spec = importlib.util.spec_from_file_location("cfmesh", TOOLBOX / "cfmesh.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def statuses(findings, check: str) -> list[str]:
    return [f.status for f in findings if f.check == check]


def measured(findings, check: str) -> str:
    return " ".join(f.measured for f in findings if f.check == check)


def meaning(findings, check: str) -> str:
    return " ".join(f.meaning for f in findings if f.check == check)


# -- trap 1: the patch renaming that meshes with no layers and exits 0 ---------


def test_every_patch_key_is_a_regex_because_feature_edges_index_the_solid_name(cfmesh):
    """`surfaceFeatureEdges` turns the solid `hull` into `hull_0`, `hull_1`, ... and
    cfMesh matches dictionary keys with `findMatchingStrings(regExp(name), ...)`, so
    the literal name matches nothing. The measured consequence was a 10,576-face mesh
    where 300,000 were expected, printed as a Warning, exit code 0."""
    text = cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", wall_cell=0.009375, layers=3)

    assert '"hull_.*"' in text
    assert "\n    hull\n" not in text  # never the bare name as a key
    assert "newName hull;" in text  # and the name is put back at the end


def test_a_dictionary_that_names_the_patch_literally_reads_as_a_silent_miss(cfmesh):
    """The half of this trap nobody sees: a mesh-side miss at least prints
    `Cannot find any patch names matching hull` (polyMeshGenFaces.C:277), but the
    identical warning on the surface side sits under `#ifdef DEBUGtriSurf`, which
    nothing in the v2512 tree defines -- so a `localRefinement` key that matches
    nothing produces no output at all. This is the only place it gets said."""
    text = cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", wall_cell=0.009375)
    literal = text.replace('"hull_.*"', "hull")

    found = cfmesh.check_meshdict(literal)

    assert statuses(found, "cfmesh-selectors") == ["fail"]
    assert "hull" in measured(found, "cfmesh-selectors")
    assert "exits 0" in meaning(found, "cfmesh-selectors")


def test_indexed_keys_pointed_at_a_raw_stl_match_nothing_either(cfmesh):
    """The other end of the same coupling. An STL handed straight to `cartesianMesh`
    keeps its solid names exactly as written -- the index only appears once
    `surfaceFeatureEdges` has run -- so a dictionary carrying `"hull_.*"` and a
    `surfaceFile` of `hull_box.stl` is the trap arriving from the opposite side.

    A warning and not a failure, and the difference is what the dictionary can show:
    against a `.ftr` every patch provably carries an index, so a bare literal provably
    misses; against a raw STL the solid names are not in the dictionary at all, and a
    surface really carrying a solid called `hull_port` would match."""
    text = cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", wall_cell=0.009375)
    raw = text.replace('surfaceFile "hull_box.ftr";', 'surfaceFile "hull_box.stl";')

    found = cfmesh.check_meshdict(raw)

    assert statuses(found, "cfmesh-selectors") == ["warn"]
    assert "surfaceFeatureEdges" in " ".join(f.repair for f in found)


def test_cfmeshs_own_tutorial_dictionaries_are_not_called_broken(cfmesh):
    """The check that stops an agent re-tuning a dictionary that was never the problem
    has to be right about a dictionary that was never the problem. This one was not:
    it read every literal key as a failure without looking at `surfaceFile`, which
    fails 6 of the 14 meshDicts cfMesh itself ships -- `asmoOctree`,
    `intakePortOctree`, `cutCubeOctree`, `socketOctree`, `sBendOctree` and
    `ship5415Octree`, the tutorial the measured route was copied from -- and the naive
    reading of the other direction failed 3 more, because `"inlet.*"` DOES match
    `inlet`: OpenFOAM's regExp is a whole-string match and `.*` matches empty.

    These are the four shapes, taken from those dictionaries verbatim."""
    raw = 'surfaceFile "geom.stl";\nmaxCellSize 5;\n'
    ftr = 'surfaceFile "ship5415-hull-and-box.ftr";\nmaxCellSize 150;\n'

    def keys(head, *ks):
        block = "".join(f"    {k}\n    {{\n        cellSize 1;\n    }}\n" for k in ks)
        return cfmesh.check_meshdict(head + "localRefinement\n{\n" + block + "}\n")

    # asmoOctree and friends: a raw STL, a literal key, no feature-edge pass at all.
    assert statuses(keys(raw, "inlet", "outlet"), "cfmesh-selectors") == ["ok"]
    # elbow_90degree, singleOrifice, multipleOrifices: `name.*` against a raw STL.
    assert statuses(keys(raw, '"inlet.*"', '"orifice0[3-6].*"'), "cfmesh-selectors") == [
        "ok"
    ]
    # ship5415Octree: a literal that already carries its index, against a .ftr. It
    # matches -- but the suffix is the global new-patch counter, so it is brittle.
    assert statuses(keys(ftr, "HULL_AND_BOX_1"), "cfmesh-selectors") == ["warn"]
    # And the trap itself, which still has to fire.
    assert statuses(keys(ftr, "hull"), "cfmesh-selectors") == ["fail"]


def test_an_index_key_is_recognised_when_the_solid_name_carries_an_underscore(cfmesh):
    """`selector()` writes `"<name>_.*"` for whatever name it is given, and the name
    `box` writes for a hull-plus-domain surface is `hull_box`. The shape test excluded
    `_` from the name part until 2026-09-08, so it did not recognise the very key this
    script writes: `"hull_box_.*"` against a raw STL came back `ok | ... these keys can
    match them`, and `hull_box_.*` is a whole-string miss against the solid `hull_box`.

    The tail is what decides the shape, so widening the name part re-flags nothing --
    the two raw-STL shapes cfMesh's own tutorials write are asserted here as well."""
    assert cfmesh.wants_an_index('"hull_box_.*"')
    assert cfmesh.wants_an_index('"HULL_AND_BOX_.*"')
    assert cfmesh.wants_an_index('"hull_.*"')
    assert not cfmesh.wants_an_index('"inlet.*"')
    assert not cfmesh.wants_an_index('"orifice0[3-6].*"')

    raw = 'surfaceFile "geom.stl";\nmaxCellSize 5;\n'
    text = raw + 'localRefinement\n{\n    "hull_box_.*"\n    {\n        cellSize 0.1;\n    }\n}\n'

    found = cfmesh.check_meshdict(text)

    assert statuses(found, "cfmesh-selectors") == ["warn"]
    assert "hull_box_.*" in measured(found, "cfmesh-selectors")


def test_naming_a_solid_twice_is_refused_rather_than_written_twice(cfmesh):
    """`--wall hull --patch hull:patch` used to emit the same `"hull_.*"`
    sub-dictionary twice inside `newPatchNames`. OpenFOAM warns on the duplicate
    keyword and the LATER one wins, so the wall came out typed `patch` and the layer
    request went with it -- a request accepted and quietly not carried out, in a
    function whose whole job is refusing those."""
    with pytest.raises(cfmesh.DictError) as exc:
        cfmesh.meshdict("h.ftr", 1.0, wall="hull", patches=(("hull", "patch"),))

    assert "named twice" in str(exc.value)
    assert "wall" in str(exc.value) and "patch" in str(exc.value)

    with pytest.raises(cfmesh.DictError):
        cfmesh.meshdict("h.ftr", 1.0, wall="hull", symmetry=("yMin",),
                        patches=(("yMin", "patch"),))


def test_a_global_one_layer_is_not_the_same_request_as_a_per_patch_one(cfmesh):
    """cfMesh reads the two through different setters and only one of them is a
    request quietly dropped. A GLOBAL `nLayers 1` goes to `setGlobalNumberOfLayers`,
    which warns and RETURNS (refineBoundaryLayers.C:121-128), leaving the no-layers
    default -- which is what sBendOctree, socketOctree and multipleOrifices mean by it,
    and sBendOctree then asks for 3 on `walls` anyway. A PER-PATCH `nLayers 1` goes to
    `setNumberOfLayersForPatch`, which warns and STORES it (:193-207): a patch that was
    asked for layers gets none. Reading them together called three shipped tutorials
    broken."""
    per_patch = (
        'surfaceFile "x.ftr";\nmaxCellSize 1;\nboundaryLayers\n{\n'
        '    patchBoundaryLayers\n    {\n        "walls.*"\n'
        "        {\n            nLayers 1;\n        }\n    }\n}\n"
    )
    global_off = (
        'surfaceFile "x.ftr";\nmaxCellSize 1;\nboundaryLayers\n{\n'
        "    nLayers 1;\n}\n"
    )

    assert statuses(cfmesh.check_meshdict(per_patch), "cfmesh-layers") == ["fail"]
    assert statuses(cfmesh.check_meshdict(global_off), "cfmesh-layers") == ["warn"]


def test_the_meshdict_names_the_ftr_even_when_the_stl_is_what_was_handed_over(cfmesh):
    """Because the two decisions are coupled and the coupling is what goes wrong: the
    feature-edge pass is what indexes the names the regex keys are written for."""
    text = cfmesh.meshdict("hull_box.stl", 0.15, wall="hull", wall_cell=0.009375)

    assert 'surfaceFile "hull_box.ftr";' in text
    assert cfmesh.read_meshdict(text)["surface"] == "hull_box.ftr"


def test_the_unmatched_warning_is_read_out_of_a_log_that_exited_zero(cfmesh, tmp_path):
    """cfMesh says this once and carries on. A session reading the exit code learns
    nothing; a session reading the last lines of the log learns nothing either,
    because the warning is printed early and the mesh looks finished."""
    log = tmp_path / "log.cartesianMesh"
    log.write_text(
        "Creating octree\n"
        "--> FOAM Warning : Cannot find any patch names matching hull\n"
        "Refining boundary\n"
        "--> FOAM Warning : Cannot find any patch names matching hull\n"
        "Finished meshing in 41 s\nEnd\n"
    )

    assert cfmesh.scan_log(log.read_text()) == ["hull"]


# -- the requests cfMesh accepts, warns about, and does not carry out ----------


def test_one_layer_is_refused_because_cfmesh_would_disable_layers_instead(cfmesh):
    """`refineBoundaryLayers.C:196` warns "boundary layers disabled for this patch"
    for any `nLayers` under 2 and meshes on. A study that asked for one layer and got
    none has no error to find, so the refusal has to happen before the run."""
    with pytest.raises(cfmesh.DictError) as exc:
        cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", layers=1)

    assert "disabled" in str(exc.value)


def test_a_thickness_ratio_below_one_is_refused_because_cfmesh_keeps_its_own(cfmesh):
    """Same shape of failure at `refineBoundaryLayers.C:145`: warned, ignored, meshed."""
    with pytest.raises(cfmesh.DictError) as exc:
        cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", layers=3, ratio=0.8)

    assert "1.2" in str(exc.value)  # the value the Wigley comparison actually ran


def test_a_wall_cell_coarser_than_the_background_is_refused(cfmesh):
    """`localRefinement` with a cellSize above `maxCellSize` is a coarsening written
    where a refinement was meant, and it is accepted."""
    with pytest.raises(cfmesh.DictError) as exc:
        cfmesh.meshdict("hull_box.ftr", 0.01, wall="hull", wall_cell=0.15)

    assert "coarser" in str(exc.value)


def test_symmetry_plane_is_refused_wherever_a_layer_will_touch_it(cfmesh):
    """Trap 4. The layer pass moves symmetry-plane points by about 1e-9 rad and
    `symmetryPlanePolyPatch` demands planarity to machine precision, after which
    `checkMesh`, `foamFormatConvert` and `postProcess` all refuse to OPEN the mesh --
    and cfMesh's own `improveSymmetryPlanes` declines to repair it.

    A check of every place a TYPE can enter, which is what it always claimed to be.
    Until 2026-09-08 it read `"symmetryPlane" in symmetry` -- a list of face NAMES,
    whose type this hardcodes to `symmetry` -- so it refused a dictionary that was
    correct while the trap it stood for could not be reached from the CLI at all,
    because no flag set a type. Naming a face `symmetryPlane` is not the mistake."""
    for kwargs in (
        {"wall_type": "symmetryPlane"},
        {"default_type": "symmetryPlane"},
        {"patches": (("yMin", "symmetryPlane"),)},
    ):
        with pytest.raises(cfmesh.DictError) as exc:
            cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", layers=3, **kwargs)
        assert "type symmetry" in str(exc.value)

    # And the face a half model actually needs, whatever it is called.
    ok = cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", symmetry=("symmetryPlane",))
    assert "type    symmetry;\n            newName symmetryPlane;" in ok


def test_a_symmetry_plane_type_in_a_hand_edited_dict_is_still_caught(cfmesh):
    """The generator cannot write one, but a dictionary that arrived from a tutorial
    or a text editor can, and this is what reads it back."""
    text = cfmesh.meshdict("hull_box.ftr", 0.15, wall="hull", symmetry=("yMin",))
    edited = text.replace("type    symmetry;", "type    symmetryPlane;")

    assert statuses(cfmesh.check_meshdict(edited), "cfmesh-symmetry") == ["fail"]


def test_a_clean_dictionary_reports_nothing_to_fix(cfmesh):
    """The refusals are only worth having if the working dictionary passes them: this
    is the shape that meshed the Wigley hull to 100.0% coverage."""
    text = cfmesh.meshdict(
        "hull_box.ftr", 0.15, wall="hull", wall_cell=0.009375, layers=3, ratio=1.2
    )

    found = cfmesh.check_meshdict(text)

    assert [f.status for f in found] == ["ok"]


# -- trap 2: the domain box, written rather than generated ---------------------


def hull_stl(path: Path) -> Path:
    """Two triangles named `hull`, which is all cfMesh needs to name a patch."""
    path.write_text(
        "solid hull\n"
        "facet normal 0 0 1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
        " endloop\nendfacet\n"
        "facet normal 0 0 -1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 0 1 0\n  vertex 1 0 0\n"
        " endloop\nendfacet\n"
        "endsolid hull\n"
    )
    return path


def test_the_body_and_its_box_come_out_as_one_closed_multi_solid_stl(cfmesh, tmp_path):
    """cfMesh meshes the volume a closed surface encloses, so the domain boundary is
    geometry: there is no `locationInMesh` and no `blockMesh`. Writing this by hand is
    what sidesteps `surfaceGenerateBoundingBox`, whose FMS the FMS reader cannot read."""
    body = cfmesh.read_stl(hull_stl(tmp_path / "hull.stl"))
    lo, hi = cfmesh.domain_box(*cfmesh.bounds(body))
    out = tmp_path / "hull_box.stl"

    cfmesh.write_stl(body + cfmesh.box_solids(lo, hi), out)
    back = cfmesh.read_stl(out)

    assert [s.name for s in back] == ["hull", *cfmesh.BOX_FACES]
    assert sum(len(s.facets) for s in back) == 2 + 12
    assert cfmesh.bounds(back) == (lo, hi)


def test_a_body_outside_the_box_is_refused_rather_than_meshed_around(cfmesh, tmp_path):
    """A body that pokes through the box does not make a hole in it; it makes a
    different closed volume, silently. The usual cause is an STL still in millimetres
    inside a box written in metres, which is the mistake `preflight.py`'s STL-scale
    check exists for at the other end of the pipeline."""
    body = cfmesh.read_stl(hull_stl(tmp_path / "hull.stl"))

    with pytest.raises(cfmesh.SurfaceError) as exc:
        cfmesh.enclose(body, (0.0, 0.0, -1.0), (0.5, 1.0, 1.0))

    assert "not inside the box" in str(exc.value)
    assert "millimetres" in str(exc.value)


def test_a_body_that_meets_a_box_face_is_named_as_a_cut_plane(cfmesh, tmp_path):
    """The ordinary half model. The face it meets is a symmetry plane, and saying so
    here is what stops it being typed `symmetryPlane` two steps later."""
    body = cfmesh.read_stl(hull_stl(tmp_path / "hull.stl"))

    touching = cfmesh.enclose(body, (-1.0, 0.0, -1.0), (2.0, 2.0, 1.0))

    assert touching == ["yMin"]


def test_a_binary_stl_with_no_name_is_refused_because_patches_are_named_after_solids(
    cfmesh, tmp_path
):
    """The binary STL format has nowhere to put a solid name, and cfMesh takes its
    patch names from those names. A binary body therefore meshes into a wall that no
    `localRefinement` or `patchBoundaryLayers` key can address -- trap 1 arriving from
    the geometry instead of the dictionary."""
    binary = tmp_path / "hull.stl"
    binary.write_bytes(
        b"\0" * 80
        + struct.pack("<I", 1)
        + struct.pack("<12f", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)
        + b"\0\0"
    )

    with pytest.raises(cfmesh.SurfaceError) as exc:
        cfmesh.read_stl(binary)
    assert "--name" in str(exc.value)

    named = cfmesh.read_stl(binary, "hull")
    assert [s.name for s in named] == ["hull"]
    assert len(named[0].facets) == 1


# -- trap 5: the CRLF that arrives from the tooling, not from git --------------


def test_nothing_this_writes_carries_a_carriage_return(cfmesh, tmp_path):
    """`pathlib.write_text` on Windows Python writes `\\r\\n`, and bash on the instance
    answers `$'\\r': command not found` -- or, on a shebang, `cannot execute: required
    file not found`. Normalising where the file is written is the only place it stays
    fixed, so both writers are pinned here."""
    body = cfmesh.read_stl(hull_stl(tmp_path / "hull.stl"))
    stl = tmp_path / "hull_box.stl"
    cfmesh.write_stl(body + cfmesh.box_solids((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)), stl)

    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "system" / "meshDict").write_text(
        cfmesh.meshdict("hull_box.stl", 0.15, wall="hull"), encoding="utf-8", newline="\n"
    )

    assert b"\r" not in stl.read_bytes()
    assert b"\r" not in (case / "system" / "meshDict").read_bytes()


# -- trap 3 and trap 4 on a mesh that already exists ---------------------------


def test_write_format_is_changed_before_the_conversion_is_suggested(cfmesh):
    """`foamFormatConvert` writes in whatever `writeFormat` says, so running it first
    converts the mesh to exactly what it already was and looks like it did nothing.
    That happened on 2026-08-31 and cost the round; the order is the whole fix."""
    text = "application     simpleFoam;\nwriteFormat     binary;\nwritePrecision  6;\n"

    out, changed = cfmesh.set_write_format_ascii(text)

    assert changed and "writeFormat     ascii;" in out
    assert "binary" not in out
    assert cfmesh.set_write_format_ascii(out) == (out, False)


def test_a_controldict_with_no_write_format_gets_one(cfmesh):
    out, changed = cfmesh.set_write_format_ascii("application     simpleFoam;\n")
    assert changed and "writeFormat     ascii;" in out


def test_a_binary_facecompactlist_mesh_reads_as_unreadable_by_cfmesh(cfmesh, tmp_path):
    """snappy writes `constant/polyMesh` binary with `class faceCompactList`; cfMesh's
    `polyMeshGen` reader wants a plain ASCII `faceList` and fails with
    `unexpected class name faceCompactList expected faceList`, which names a class and
    not a remedy."""
    case = tmp_path / "case"
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    (poly / "faces").write_text(
        "FoamFile\n{\n    format      binary;\n    class       faceCompactList;\n"
        "    object      faces;\n}\n"
    )

    found = cfmesh.prepare(case)

    assert statuses(found, "cfmesh-readable") == ["fail"]
    assert "faceCompactList" in meaning(found, "cfmesh-readable")
    assert "FIRST" in " ".join(f.repair for f in found)


def test_symmetry_planes_are_retyped_and_named_rather_than_silently_edited(cfmesh):
    text = (
        "2\n(\n"
        "    ySymmetry\n    {\n        type            symmetryPlane;\n"
        "        nFaces          100;\n    }\n"
        "    inlet\n    {\n        type            patch;\n"
        "        nFaces          40;\n    }\n)\n"
    )

    out, moved = cfmesh.retype_symmetry_planes(text)

    assert moved == ["ySymmetry"]
    assert "type            symmetry;" in out
    assert "symmetryPlane" not in out
    assert "type            patch;" in out  # and nothing else was touched


def test_a_patch_named_symmetryplane_keeps_its_name_when_its_type_is_repaired(cfmesh):
    """The one path in this script that writes to a mesh, and it used to corrupt one
    quietly. The replacement ran `.replace("symmetryPlane", "symmetry")` over the WHOLE
    match, which begins at the patch NAME -- so a patch a mesher had called
    `symmetryPlane` came back called `symmetry`, `--write` put that in
    `constant/polyMesh/boundary`, and every `0/` field's `boundaryField` entry for that
    patch then matched nothing. A mesh broken more quietly than the one being repaired,
    and the existing test used a patch called `ySymmetry`, so it never saw it.

    `inGroups` is left alone on purpose: that word is a patch GROUP a `boundaryField`
    may be keyed on, and rewriting it would drop those entries -- the same harm in the
    other direction. Stale, not wrong; OpenFOAM constructs the patch from `type`."""
    text = (
        "1\n(\n    symmetryPlane\n    {\n"
        "        type            symmetryPlane;\n"
        "        inGroups        1(symmetryPlane);\n"
        "        nFaces          100;\n        startFace       0;\n    }\n)\n"
    )

    out, moved = cfmesh.retype_symmetry_planes(text)

    assert moved == ["symmetryPlane"]
    assert "    symmetryPlane\n" in out, "the patch keeps the name the fields know"
    assert "type            symmetry;" in out
    assert "inGroups        1(symmetryPlane);" in out


def test_prepare_applies_both_edits_and_says_which(cfmesh, tmp_path):
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text("writeFormat     binary;\n")
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    (poly / "boundary").write_text(
        "1\n(\n    ySymmetry\n    {\n        type            symmetryPlane;\n"
        "        nFaces          100;\n        startFace       0;\n    }\n)\n"
    )

    changed = cfmesh.apply_preparation(case)

    assert any("writeFormat ascii" in line for line in changed)
    assert any("type symmetry on patch(es) ySymmetry" in line for line in changed)
    assert "symmetryPlane" not in (poly / "boundary").read_text()


# -- the mesh, read back -------------------------------------------------------


def test_a_wall_that_kept_its_index_is_the_rename_that_did_not_happen(cfmesh, tmp_path):
    """What trap 1 leaves behind when nobody kept the log: a mesh whose wall is called
    `hull_0`, and a case whose every later dictionary was written for `hull`."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)
    boundary = case / "constant" / "polyMesh" / "boundary"
    boundary.write_text(boundary.read_text().replace("    wall\n", "    hull_0\n"))

    found = cfmesh.check(case, wall="hull")

    assert statuses(found, "cfmesh-wall") == ["fail"]
    assert "hull_0" in meaning(found, "cfmesh-wall")


def test_a_wall_that_was_renamed_reads_as_ok(cfmesh, tmp_path):
    case = write_box(tmp_path / "case", XS, YS, LAYERED)

    found = cfmesh.check(case, wall="wall")

    assert statuses(found, "cfmesh-wall") == ["ok"]
    assert "8 faces" in measured(found, "cfmesh-wall")


def test_coverage_comes_back_from_the_same_measurement_as_layer_report(cfmesh, tmp_path):
    """The number that settled the argument, offered where the mesh is being checked.
    It is `layer_report.py`'s measurement, called rather than re-implemented: one
    metric applied to every mesh is the only reason snappy's 66.7% and cfMesh's 100.0%
    could be put in the same table."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)

    found = cfmesh.check(case, wall="wall", wall_spacing=0.2)

    assert statuses(found, "cfmesh-coverage") == ["ok"]
    assert "100.0%" in measured(found, "cfmesh-coverage")


def test_a_coverage_number_with_no_reference_says_where_its_threshold_came_from(
    cfmesh, tmp_path
):
    """Without a reference wall the threshold is this mesh's own median, which can
    only see a split near a half -- a wholly layered wall and a wholly bare one look
    alike to it. A number that cannot stand alone has to say so where it is printed."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)

    found = cfmesh.check(case, wall="wall")

    assert "median" in measured(found, "cfmesh-coverage")
    assert "--ref" in meaning(found, "cfmesh-coverage")


# The wall of `write_box(XS, YS, ...)` is 8 faces of area 0.25x0.4 and 0.25x0.6, so
# `sqrt(median area)` -- the same in-plane length scale `layer_report.face_spacing`
# uses -- is sqrt(0.125) = 353.6 mm. Every number below is placed against that ONE
# measured spacing rather than against a chosen threshold, which is the whole point of
# the rewrite: the check that used to live here turned on a `1.3` that no run in
# qa-runs/comp/cfmesh calibrates, bracketed by tests at 0.86x and 32x.
WALL_SPACING_MM = 353.6  # printed to 3 significant figures as "354 mm"


def test_refinement_that_did_not_take_shows_as_a_wall_at_the_background_size(
    cfmesh, tmp_path
):
    """The signature of the SURFACE-side miss, which prints nothing at all because
    `triSurfFacets.C:103`'s warning sits under `#ifdef DEBUGtriSurf`. A
    `localRefinement` that matched no patch leaves the wall meshed at `maxCellSize`,
    and the QA record's own tell was exactly that size -- "a 10,576-face mesh where
    300,000 were expected". Here the wall measures 354 mm, 20 mm was asked for and
    350 mm is the background, so the wall is sitting on the background."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "meshDict").write_text(
        cfmesh.meshdict("hull_box.ftr", 0.35, wall="wall", wall_cell=0.02)
    )

    found = cfmesh.check(case, wall="wall")

    assert statuses(found, "cfmesh-refinement") == ["fail"]
    assert f"{WALL_SPACING_MM:.3g} mm" in measured(found, "cfmesh-refinement")
    assert "DEBUGtriSurf" in meaning(found, "cfmesh-refinement")


def test_a_refined_wall_is_not_flagged(cfmesh, tmp_path):
    """The same reading on a wall that got what it asked for. A check that fires on the
    working case is worse than no check -- and unlike the cell-count version this is
    calibrated on the mesh in front of it: 354 mm measured against 350 mm requested and
    a 5.6 m background, which is the same four octree levels apart as the Wigley run's
    9.375 mm against 150 mm."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "meshDict").write_text(
        cfmesh.meshdict("hull_box.ftr", 5.6, wall="wall", wall_cell=0.35)
    )

    assert statuses(cfmesh.check(case, wall="wall"), "cfmesh-refinement") == ["ok"]


def test_a_refinement_within_one_octree_level_says_it_cannot_tell(cfmesh, tmp_path):
    """cfMesh's octree only halves, so a request less than one halving below
    `maxCellSize` produces a wall the same size either way. The honest answer is that
    there is no reading here, not a verdict picked off a threshold -- which is what the
    `1.3` this replaced would have given, and what F-53 is about."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "meshDict").write_text(
        cfmesh.meshdict("hull_box.ftr", 0.5, wall="wall", wall_cell=0.35)
    )

    found = cfmesh.check(case, wall="wall")

    assert statuses(found, "cfmesh-refinement") == ["skipped"]
    assert "octree halving" in meaning(found, "cfmesh-refinement")


def test_another_regions_cellsize_is_not_read_as_the_walls_own_request(cfmesh, tmp_path):
    """Refine the bow, refine the propeller: the most ordinary hand-edit there is, and
    5 of the 14 meshDicts cfMesh ships already carry more than one `cellSize`. This
    used to take `min` of every size in the block, so the identical mesh that passes
    above came back `fail :: ... against 10 mm requested`, blaming the wall for a
    number asked of a different solid. A fail-severity check that is wrong about a
    correct dictionary is the error F-53 exists to stop."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)
    (case / "system").mkdir(parents=True, exist_ok=True)
    one = cfmesh.meshdict("hull_box.ftr", 5.6, wall="wall", wall_cell=0.35)
    two = one.replace(
        '    "wall_.*"\n    {\n        cellSize 0.35;\n    }\n',
        '    "wall_.*"\n    {\n        cellSize 0.35;\n    }\n'
        '    "prop_.*"\n    {\n        cellSize 0.01;\n    }\n',
    )
    assert two != one, "the second region has to actually be in the dictionary"
    (case / "system" / "meshDict").write_text(two)

    found = cfmesh.check(case, wall="wall")

    assert statuses(found, "cfmesh-refinement") == ["ok"]
    assert "350 mm asked of it by \"wall_.*\"" in measured(found, "cfmesh-refinement")
    assert "10 mm" not in measured(found, "cfmesh-refinement")


def test_the_keys_and_the_sizes_are_paired_not_two_flat_lists(cfmesh):
    """cfMesh's own `socketOctree` is the shape that breaks the flat reading: three
    `localRefinement` entries and two `cellSize`s, because `patch15` asks with
    `additionalRefinementLevels` instead. Read as parallel lists the sizes shift up by
    one and every entry after the gap is attributed to the wrong key."""
    text = (
        'surfaceFile "socket.fms";\nmaxCellSize 9;\n'
        "localRefinement\n{\n"
        "    patch15\n    {\n        additionalRefinementLevels 1;\n    }\n"
        "    subset1\n    {\n        cellSize 4.5;\n    }\n"
        "    subset2\n    {\n        cellSize 2.25;\n    }\n}\n"
    )

    assert cfmesh.read_meshdict(text)["refinement_pairs"] == [
        ("patch15", None),
        ("subset1", 4.5),
        ("subset2", 2.25),
    ]


def test_a_localrefinement_that_names_no_wall_is_skipped_rather_than_compared(
    cfmesh, tmp_path
):
    """There is no request about this wall to check the mesh against, and another
    patch's `cellSize` is not one. The honest answer is that nothing was asked."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "meshDict").write_text(
        cfmesh.meshdict("hull_box.ftr", 5.6, wall="prop", wall_cell=0.35)
    )

    found = cfmesh.check(case, wall="wall")

    assert statuses(found, "cfmesh-refinement") == ["skipped"]
    assert "asks nothing of `wall`" in measured(found, "cfmesh-refinement")


def test_the_floor_a_wall_falls_back_to_is_boundarycellsize_where_one_is_set(
    cfmesh, tmp_path
):
    """`boundaryCellSize` is stamped onto EVERY surface triangle before any patch-wise
    level is appended (meshOctreeCreatorCreateOctreeBoxes.C:130-186), so where it is
    set it -- and not `maxCellSize` -- is the size a wall comes out at when the key
    misses. 9 of the 14 shipped meshDicts set it.

    The wall here measures 354 mm because `boundaryCellSize 0.35` put it there and the
    175 mm the key asked for never arrived: exactly the silent surface-side miss. Read
    against `maxCellSize 5.6` instead, 354 mm looks far nearer 175 mm than 5600 mm and
    this printed a confident `ok` over it."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)
    (case / "system").mkdir(parents=True, exist_ok=True)
    text = cfmesh.meshdict("hull_box.ftr", 5.6, wall="wall", wall_cell=0.175)
    (case / "system" / "meshDict").write_text(
        text.replace("\nmaxCellSize 5.6;\n", "\nmaxCellSize 5.6;\n\nboundaryCellSize 0.35;\n")
    )

    found = cfmesh.check(case, wall="wall")

    assert statuses(found, "cfmesh-refinement") == ["fail"]
    assert "350 mm boundaryCellSize" in measured(found, "cfmesh-refinement")
    assert "DEBUGtriSurf" in meaning(found, "cfmesh-refinement")


# -- the CLI surface -----------------------------------------------------------


def test_where_reports_an_absent_cfmesh_without_calling_it_broken(
    cfmesh, capsys, monkeypatch
):
    """Three of cfMesh's boundary-layer names -- `detectBoundaryLayers`,
    `extrudeLayer`, `refineBoundaryLayers` -- are classes inside `libmeshLibrary` and
    never were executables in this release. A session that goes looking for them on
    PATH and concludes the install is broken has lost a round for nothing.

    `which` is stubbed rather than trusted. The version of this test that passed on
    whatever the machine happened to have installed asserted only two lines
    `report_where` prints unconditionally, so it passed identically on an instance
    where cfMesh was present -- a test named for absence that never saw one."""
    monkeypatch.setattr(cfmesh.shutil, "which", lambda name: None)

    assert cfmesh.main(["where"]) == 0
    out = capsys.readouterr().out

    assert "cartesianMesh is NOT on this instance's PATH" in out
    assert out.count("absent") == 8, "the eight binaries `where` actually probes"
    assert "qa-runs/comp/cfmesh/" in out, "the source tarball is the way back"
    assert "not executables in this release" in out
    assert "refineBoundaryLayers" in out


def test_where_reports_a_present_cfmesh_as_present(cfmesh, capsys, monkeypatch):
    """The other branch, which nothing exercised: on this Windows workspace `which`
    finds nothing and `ldd` cannot be asked, so `unresolved_libraries` only ever
    returned None. A binary that could not be analysed is UNCHECKED, and saying
    "present (libraries unchecked)" rather than "present" is the difference between
    "cfMesh works" and "cfMesh is here and I did not look"."""
    monkeypatch.setattr(cfmesh.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cfmesh, "_run", lambda cmd: "" if cmd[0] == "ldd" else "")

    assert cfmesh.main(["where"]) == 0
    out = capsys.readouterr().out

    assert "cartesianMesh: /usr/bin/cartesianMesh" in out
    assert out.count("present (libraries unchecked)") == 8
    assert "NOT on this instance" not in out


def test_a_present_cfmesh_is_read_out_of_dpkg_and_ldd_rather_than_assumed(
    cfmesh, capsys, monkeypatch
):
    """The two branches nothing had ever run, because this workspace has neither
    `dpkg` nor `ldd`: the package owner, and a library that does not resolve.

    Which package owns `cartesianMesh` is the whole reason `where` exists. `openfoam*`
    means it came with the IMAGE -- no build, no volume, no exports, and it survives
    instance deletion -- which is the fact that was available and unread for four
    rounds of the Wigley comparison. Anything else means somebody built it onto the
    volume, and that answer expires. An unresolved library is the third state: on
    PATH, and it will not start."""
    monkeypatch.setattr(cfmesh.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd):
        if cmd[0] == "dpkg":
            return "openfoam2512: /usr/bin/cartesianMesh\n"
        if cmd[0] == "ldd" and cmd[1].endswith("tetMesh"):
            return "\tlibmeshLibrary.so => not found\n"
        return "\tlibc.so.6 => /lib/libc.so.6 (0x00007f)\n"

    monkeypatch.setattr(cfmesh, "_run", fake_run)

    assert cfmesh.main(["where"]) == 0
    out = capsys.readouterr().out

    assert "owned by openfoam2512" in out
    assert "part of the image" in out and "survives" in out
    assert "tetMesh" in out and "UNRESOLVED: libmeshLibrary.so" in out
    assert out.count("present, libraries resolve") == 7, "the other seven"


def test_a_missing_input_file_refuses_in_the_same_shape_as_every_other_failure(
    cfmesh, tmp_path, capsys
):
    """`read_stl` raised a bare `FileNotFoundError` traceback while the refusals came
    out as `refused: <sentence>` and exit 2. Two shapes of failure in one script is one
    more than a session should have to recognise, and a traceback is the shape that
    reads as "the tool is broken" rather than "the input is not there".

    This docstring claimed "every other failure" and a third shape was live until
    2026-09-08: the `--force` refusals raised a bare `SystemExit`, printing
    `<path> exists; --force to overwrite` with no prefix and exit 1. They raise
    `OverwriteError` now, and `test_case_refuses_to_overwrite_a_meshdict_it_did_not_write`
    reads the code and the prefix rather than only that something was raised."""
    code = cfmesh.main(
        ["box", str(tmp_path / "not_here.stl"), "--out", str(tmp_path / "out.stl")]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("refused: ")
    assert "not_here.stl" in err
    assert "Traceback" not in err


def test_a_refusal_exits_two_with_a_sentence_rather_than_a_traceback(cfmesh, tmp_path, capsys):
    """Every refusal in here stands for a round of the comparison that ended in a mesh
    with no layers on it and an exit code of 0, so the refusal has to be legible."""
    code = cfmesh.main(
        ["case", str(tmp_path), "--surface", "hull_box.stl", "--max-cell", "0.15",
         "--wall", "hull", "--layers", "1"]
    )

    assert code == 2
    assert "refused:" in capsys.readouterr().err


def test_case_refuses_to_overwrite_a_meshdict_it_did_not_write(cfmesh, tmp_path, capsys):
    """And refuses it in the ONE shape this script has for a refusal. This used to
    raise a bare `SystemExit`, which prints the sentence with no `refused:` prefix and
    exits 1 -- a third shape of failure alongside the two the test above names, and one
    that test could not see because it only asserted that something was raised."""
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "meshDict").write_text("// hand written\n")

    code = cfmesh.main(
        ["case", str(tmp_path), "--surface", "a.stl", "--max-cell", "0.1"]
    )

    assert code == 2
    assert capsys.readouterr().err.startswith("refused: ")
    assert (tmp_path / "system" / "meshDict").read_text() == "// hand written\n"


def test_the_route_it_prints_is_the_route_that_was_measured_to_work(cfmesh):
    """`.ftr`, not `.fms` -- trap 2 -- and the coverage measurement at the end of it,
    because a mesh that was not measured is a mesh whose layers are a claim."""
    lines = cfmesh.route("hull_box.stl", "hull")

    assert lines[0].startswith("surfaceFeatureEdges -angle 45 hull_box.stl hull_box.ftr")
    assert any("cartesianMesh" in line for line in lines)
    assert any("layer_report.py" in line for line in lines)


def test_the_feature_angle_asked_for_is_the_one_the_route_prints(cfmesh, tmp_path, capsys):
    """`--angle` was parsed on both `box` and `case` and never reached `route()`, so
    `--angle 20` -- the ordinary thing to ask for on a hull with soft chines -- printed
    `-angle 45` and handed back a command indexing a different set of patches than was
    asked for. A request accepted and quietly not carried out is the exact defect class
    every refusal in this script exists to stop cfMesh committing, so the script
    committing it is worse than a missing flag. Both subcommands, through main()."""
    body = hull_stl(tmp_path / "hull.stl")

    assert cfmesh.main(["box", str(body), "--out", str(tmp_path / "b.stl"),
                        "--angle", "20"]) == 0
    assert "surfaceFeatureEdges -angle 20 " in capsys.readouterr().out

    assert cfmesh.main(["case", str(tmp_path), "--surface", "b.stl",
                        "--max-cell", "0.15", "--wall", "hull", "--angle", "12"]) == 0
    assert "surfaceFeatureEdges -angle 12 " in capsys.readouterr().out


def test_a_surface_that_is_already_a_ftr_gets_no_feature_edge_pass(cfmesh):
    """`route()` derived the feature file as `<stem>.ftr`, so a `.ftr` input printed
    `surfaceFeatureEdges -angle 45 domain.ftr domain.ftr` -- which cannot run:
    `surfaceFeatureEdges.C:67-73` fatals with "Output file ... would overwrite the input
    file". Nothing was destroyed, but a route is copied rather than read, and its first
    line was a guaranteed error. `case --help` advertises `.ftr` as an input."""
    lines = cfmesh.route("domain.ftr", "hull")

    assert not any("surfaceFeatureEdges -angle" in line for line in lines)
    assert "already a .ftr" in lines[0] and lines[0].startswith("#")
    assert lines[1].startswith("cartesianMesh")


def test_an_fms_surface_is_read_whatever_its_capitalisation(cfmesh):
    """`check_meshdict` advertises itself as runnable on a hand-written dictionary, and
    a hand-written one is where the capitalisation varies. The `.fms` test was
    case-sensitive while its two neighbours were not, which disarmed trap 2 on the one
    input the function was written for."""
    for suffix in (".fms", ".FMS", ".Fms"):
        text = f'surfaceFile "geom{suffix}";\nmaxCellSize 1;\n'
        assert statuses(cfmesh.check_meshdict(text), "cfmesh-surface") == ["warn"], suffix


def test_a_log_that_is_not_there_is_reported_rather_than_passed_over(cfmesh, tmp_path):
    """The log scan is, by this function's own docstring, the only place trap 1's
    mesh-side warning is ever stated. An explicitly-passed `--log` that did not exist
    produced no finding at all, so a typo'd or rotated path read exactly like a clean
    run -- and a clean run is the answer a session is hoping for. `prepare` already
    returns a `skipped` Finding for a missing file, so the convention existed."""
    case = write_box(tmp_path / "case", XS, YS, LAYERED)

    found = cfmesh.check(case, wall="wall", log=tmp_path / "log.rotated-away")

    assert statuses(found, "cfmesh-log") == ["skipped"]
    assert "no such file" in measured(found, "cfmesh-log")


def test_every_face_not_named_is_merged_into_one_patch_and_that_is_said(
    cfmesh, tmp_path, capsys
):
    """`renameBoundary` carries a `defaultName`, and because it is PRESENT cfMesh adds
    ONE patch and maps every solid the dictionary did not name into it
    (renameBoundaryPatches.C:149-168; the `else` at :169-180 is what would have kept
    the original names). So `--wall hull` alone yields a two-patch mesh -- `hull` and a
    `farField` carrying all six box faces, pointing six ways -- which cannot be given an
    inlet and an outlet, and so cannot carry the grid-convergence study this whole
    finding is asking for. `case_gen.py` infers the inlet from a patch's own normal
    aimed at the domain centre, and a merged patch has no such normal.

    Nothing said so: not COSTS, not the notes, not the CLI. Now the merge is named at
    the moment the dictionary is written, and `--patch NAME[:TYPE]` is the way out."""
    assert cfmesh.main(["case", str(tmp_path), "--surface", "hull_box.stl",
                        "--max-cell", "0.15", "--wall", "hull"]) == 0
    out = capsys.readouterr().out

    assert "merges into one `farField` patch" in out
    for face in cfmesh.BOX_FACES:
        assert face in out.split("of `box`'s six faces that is:")[1]
    assert "renameBoundary merges" in cfmesh.COSTS.replace("`", "")


def test_a_run_with_nothing_left_over_does_not_claim_a_merge(cfmesh, tmp_path, capsys):
    """The merge is conditional and this printed it unconditionally. cfMesh adds the
    `defaultName` patch only where some solid is actually unmatched -- the `addPatch`
    test at renameBoundaryPatches.C:151-161 -- so a run that names all seven solids
    gets no `farField` at all, and a line saying otherwise is the same shape of
    overclaim this script exists to catch, told about itself."""
    assert cfmesh.main(
        ["case", str(tmp_path), "--surface", "hull_box.stl", "--max-cell", "0.15",
         "--wall", "hull", "--patch", "xMin", "--patch", "xMax", "--patch", "yMax",
         "--patch", "zMin", "--patch", "zMax", "--symmetry", "yMin"]
    ) == 0
    out = capsys.readouterr().out

    assert "merges into one `farField` patch" not in out
    assert "no `box` face is left over" in out
    # And it still says what it cannot see: the surface's own solid list is not read
    # here, so any OTHER solid in it would still merge.
    assert "Any OTHER solid in the surface still merges" in out


def test_a_named_face_survives_the_merge_with_the_type_it_was_given(cfmesh):
    """The way out, in the dictionary rather than the message. `--patch xMin:patch` and
    `--symmetry yMin` write their own `newPatchNames` entries, so those faces come out
    as themselves and only what is left merges."""
    text = cfmesh.meshdict(
        "hull_box.ftr", 0.15, wall="hull", wall_cell=0.009375, layers=3,
        symmetry=("yMin",), patches=(("xMin", "patch"), ("xMax", "patch")),
    )

    for name, ptype in (("hull", "wall"), ("yMin", "symmetry"),
                        ("xMin", "patch"), ("xMax", "patch")):
        assert f'"{name}_.*"' in text
        assert f"type    {ptype};\n            newName {name};" in text
    assert "Named here: hull, yMin, xMin, xMax." in text
    assert cfmesh.check_meshdict(text) == [
        f for f in cfmesh.check_meshdict(text) if f.status == "ok"
    ], "the dictionary it writes still passes its own checks"


# -- what it will not do, said where the choice is made ------------------------


def test_the_absence_of_y_plus_control_is_stated_with_the_mechanism(cfmesh):
    """The honest cost of taking this route, and the one most likely to be discovered
    late. cfMesh has no first-layer target: the stack is sized from the local cell,
    `edge / (1 + r + ... + r^(n-1))` at refineBoundaryLayersFunctions.C:688-696, and
    `maxFirstLayerThickness` enters only as `min(max(cap, SMALL), that)` at :709-714 --
    a cap that can thin the first cell but never place it at a stated height. Both are
    quoted here as `COSTS` quotes them: a docstring in this codebase is a published
    citation, and it has no licence to be looser than the code it guards. The 1.21 mm
    first cell measured on the Wigley hull is a consequence of the cell size, not a
    target that was hit."""
    assert "no y+ request" in cfmesh.COSTS.lower()
    assert "maxFirstLayerThickness" in cfmesh.COSTS
    assert "cap" in cfmesh.COSTS
    assert "1.21 mm" in cfmesh.COSTS


def test_the_field_notes_say_which_mesher_and_what_it_costs():
    """The toolbox is offered, not imposed, so the script only helps a session that
    already knows there is a choice to make. The notes are where that gets said."""
    notes = (TOOLBOX / "notes" / "openfoam-field-notes.md").read_text(encoding="utf-8")

    assert "cartesianMesh" in notes
    assert "erode" in notes.lower()
    assert "no y⁺ control at all" in notes  # the cost, not only the remedy
    assert "cfmesh.py" in notes
