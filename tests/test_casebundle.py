"""The work product gets a second copy, and says honestly what is in it.

Step 1 of `qa-runs/PROPOSAL-per-session-workspace.md`. Measured on production
2026-09-08 over 215 studies: the transcript is captured for essentially all of them, an
artifact for 40%, a results payload for 12%, and the case definition, mesh and solver
logs for **none**. So every mesh and every `Allrun` the product has made lives in one
place, on a Volume shared with every other study, with no retention and no backup --
which on that same day was 30.8 GB against a 20 GB quota with a live study unable to
write (F-56).

Two properties carry the whole design and both are tested here: the irreplaceable tiers
go first so they always fit, and nothing is ever dropped silently.
"""

from __future__ import annotations

import io
import json
import os
import tarfile

import pytest

from openreynolds import casebundle


def _study(root, case="cyl", times=("0", "0.5", "1"), mesh_kb=600, procs=2):
    """A mirrored study, shaped the way the toolbox actually writes one."""
    c = root / case
    for d in ("system", "constant/polyMesh", "0", "0.orig", "postProcessing/forces"):
        (c / d).mkdir(parents=True, exist_ok=True)
    (c / "system" / "controlDict").write_text("application simpleFoam;\n")
    (c / "system" / "fvSchemes").write_text("ddtSchemes { default steadyState; }\n")
    (c / "constant" / "transportProperties").write_text("nu 1e-05;\n")
    # Random bytes, not a repeated character: a mesh of 600 KB of "p" gzips to
    # nothing, and a budget test against a bundle that compresses to 2 KB proves
    # nothing at all. A real polyMesh is coordinates and does not compress like that.
    (c / "constant" / "polyMesh" / "points").write_bytes(os.urandom(mesh_kb * 1024))
    (c / "constant" / "polyMesh" / "faces").write_bytes(os.urandom(mesh_kb * 1024))
    (c / "0" / "U").write_text("internalField uniform (1 0 0);\n")
    (c / "0.orig" / "U").write_text("internalField uniform (1 0 0);\n")
    (c / "postProcessing" / "forces" / "forces.dat").write_text("t Cd Cl\n1 0.4 0.0\n")
    (c / "log.simpleFoam").write_text("Time = 1\nExecutionTime = 3 s\n" * 50)
    (c / "log.blockMesh").write_text("mesh ok\n")
    (c / "Allrun").write_text("#!/bin/sh\nsimpleFoam\n")
    (c / "Allmesh").write_text("#!/bin/sh\nblockMesh\n")
    (c / "build.py").write_text("# the geometry, as a script\n")
    (c / "notes.md").write_text("# what this study is\n")
    (c / "hull.stl").write_text("solid hull\nendsolid\n")
    (c / "mesh_z.png").write_bytes(b"not-really-a-png")
    for t in times:
        (c / t).mkdir(exist_ok=True)
        (c / t / "U").write_text("u" * 512)
        (c / t / "p").write_text("p" * 512)
    for i in range(procs):
        p = c / f"processor{i}" / "0"
        p.mkdir(parents=True, exist_ok=True)
        (p / "U").write_text("d" * 8192)
    return c


@pytest.fixture
def mirror(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    _study(root)
    return root


def _names(blob):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        return set(tar.getnames())


def _manifest(blob):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        return json.loads(tar.extractfile("MANIFEST.json").read())


# -- what travels ---------------------------------------------------------------

def test_the_irreplaceable_case_travels(mirror):
    """None of this exists anywhere else. `Allrun` and `build.py` especially: F-55 was
    about exactly those two not reaching the user's machine, because they are what a
    person edits to change a number and re-run."""
    blob, report = casebundle.build(mirror, study_id="s1")
    names = _names(blob)
    for wanted in ("cyl/system/controlDict", "cyl/constant/transportProperties",
                   "cyl/0/U", "cyl/log.simpleFoam",
                   "cyl/postProcessing/forces/forces.dat",
                   "cyl/Allrun", "cyl/Allmesh", "cyl/build.py", "cyl/notes.md"):
        assert wanted in names, f"{wanted} was not captured"
    assert "definition" in report.included and "record" in report.included


def test_the_regenerable_bulk_does_not(mirror):
    """`processor*` is a decomposed copy of fields the case already holds, and a
    superseded time directory is a state the run passed through. Both are what
    `toolbox/disk.py` prunes off the volume; shipping them here would be uploading
    exactly the bytes that filled the disk in the first place."""
    blob, _ = casebundle.build(mirror, study_id="s1")
    names = _names(blob)
    assert not any(n.startswith("cyl/processor") for n in names)
    assert "cyl/0.5/U" not in names, "a superseded time directory travelled"
    assert "cyl/1/U" in names, "the latest time is the one worth keeping"


def test_renders_are_not_sent_twice(mirror):
    """They already travel one at a time through `Capture.artifact`."""
    assert "cyl/mesh_z.png" not in _names(casebundle.build(mirror, study_id="s1")[0])


# -- the priority order, which is the whole design ------------------------------

def test_the_irreplaceable_tiers_survive_a_budget_that_excludes_the_mesh(mirror):
    """835 MB of 6.4 GB across 215 studies is definition + record -- about 4 MB a
    study -- so the part that cannot be reproduced always fits, and the mesh (47%)
    is the thing that gets dropped. A resume rebuilds a mesh from `Allmesh`; nothing
    rebuilds a log.

    And a tier that does not fit does not close the door on the ones after it:
    `fields` is small here and still travels once `mesh` has been refused. Filling
    the remaining room is strictly better than stopping at the first thing too big
    for it, and costs only that the bundle's contents are no longer a prefix of the
    priority order -- which is why the manifest lists what is in it rather than a
    high-water mark."""
    blob, report = casebundle.build(mirror, budget_mb=1, study_id="s1")
    assert blob is not None
    assert "definition" in report.included and "record" in report.included
    assert "mesh" in report.skipped, "the 47% that a resume can rebuild is what goes"
    assert "fields" in report.included, "a small later tier still fits"
    names = _names(blob)
    assert "cyl/log.simpleFoam" in names
    assert not any("polyMesh" in n for n in names)


def test_a_dropped_tier_is_named_in_the_report_and_in_the_manifest(mirror):
    """The one thing worse than an incomplete backup is one that claims to be whole."""
    blob, report = casebundle.build(mirror, budget_mb=1, study_id="s1")
    said = " ".join(report.brief())
    assert "left out" in said and "mesh" in said
    body = _manifest(blob)
    assert body["complete"] is False
    assert "mesh" in body["skipped"] and "definition" in body["tiers"]


def test_a_complete_bundle_says_so(mirror):
    blob, report = casebundle.build(mirror, study_id="s1")
    assert report.skipped == {}
    body = _manifest(blob)
    assert body["complete"] is True and body["study_id"] == "s1"
    assert body["files"] == report.files and report.bytes > 0
    assert body["over_budget"] is False


def test_the_manifest_travels_inside_the_archive(mirror):
    """Not in the artifact's `meta`: the upload route takes a `meta` form field the
    client does not send, so putting it here needs no API change, survives any
    transport that moves the bytes, and is the first thing an unpacker reads."""
    assert "MANIFEST.json" in _names(casebundle.build(mirror, study_id="s1")[0])


# -- it may never end a session -------------------------------------------------

def test_a_missing_directory_is_a_report_not_an_exception(tmp_path):
    blob, report = casebundle.build(tmp_path / "nope", study_id="s1")
    assert blob is None and report.error and "not a directory" in report.error
    assert "could not bundle" in report.brief()[0]


def test_an_empty_mirror_bundles_nothing_and_says_so(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    blob, report = casebundle.build(root, study_id="s1")
    assert blob is None and report.included == []
    assert "nothing to bundle" in report.brief()[0]


def test_a_file_that_vanishes_mid_pack_does_not_raise(mirror, monkeypatch):
    """The mirror is still settling and a temp file can go between the walk and the
    add. A backup is not worth failing a session over."""
    real_add = tarfile.TarFile.add

    def flaky(self, name, arcname=None, recursive=True, **kw):
        if str(name).endswith("log.blockMesh"):
            raise OSError("vanished")
        return real_add(self, name, arcname=arcname, recursive=recursive, **kw)

    monkeypatch.setattr(tarfile.TarFile, "add", flaky)
    blob, report = casebundle.build(mirror, study_id="s1")
    assert blob is not None and report.error is None
    assert "cyl/log.simpleFoam" in _names(blob)


def test_the_budget_is_under_the_services_cap(mirror):
    """`FOAMD_MAX_ARTIFACT_MB` is 50 and no route reports it. Guessing exactly would
    start failing the day an operator lowered it, at the end of a session."""
    assert casebundle.BUDGET_MB < 50


def test_a_bundle_stays_within_its_budget(mirror):
    blob, report = casebundle.build(mirror, budget_mb=1, study_id="s1")
    assert blob is not None and len(blob) <= 1 * 1024 * 1024
    assert report.bytes == len(blob)


def test_an_irreplaceable_tier_too_big_for_the_budget_is_sent_anyway(tmp_path):
    """Skipping it ships nothing at all -- every later tier is larger and also
    skipped -- so the study whose definition is too big to bundle would be exactly
    the one that silently gets no second copy. It goes, and the report says the
    service may refuse it."""
    root = tmp_path / "files"
    root.mkdir()
    case = root / "big"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text("x" * 400_000)
    blob, report = casebundle.build(root, budget_mb=0, study_id="s1")
    assert blob is not None and report.oversized is True
    assert "definition" in report.included
    assert "may be refused" in " ".join(report.brief())
    assert _manifest(blob)["over_budget"] is True
