"""The volume fills up and nothing ever cleaned it: what may go, and what may not.

F-56, 2026-09-08. A session with a 24 MB case could not write solver output because
`/work` stood at 31.5 GB against a 20 GB quota, and the six biggest directories were
that same account's own older studies. The agent stopped and asked rather than delete
what it could not prove was safe -- the right call, and one it should not have to make
blind, because there was nothing it could read to find out.

These tests are almost entirely about the *refusals*. A tool that frees space is only
worth having if it is trustworthy about what it will not touch, and every entry in the
keep set below is something whose loss would turn a case that could be re-run into one
that cannot.
"""

from __future__ import annotations

import json

import pytest

from openreynolds.toolbox import disk


def _case(root, name="case", times=("0", "0.5", "1"), procs=2):
    """A case tree with the shape the toolbox actually writes."""
    c = root / name
    for d in ("system", "constant/polyMesh", "0.orig", "postProcessing/forces"):
        (c / d).mkdir(parents=True, exist_ok=True)
    (c / "system" / "controlDict").write_text("x" * 100)
    (c / "constant" / "polyMesh" / "points").write_text("p" * 5000)
    (c / "postProcessing" / "forces" / "forces.dat").write_text("f" * 400)
    (c / "log.simpleFoam").write_text("L" * 3000)
    (c / "Allrun").write_text("#!/bin/sh\n")
    (c / "build.py").write_text("# geometry\n")
    (c / "mesh_z.png").write_text("PNG")
    for t in times:
        (c / t).mkdir(exist_ok=True)
        (c / t / "U").write_text("u" * 2000)
    for i in range(procs):
        p = c / f"processor{i}" / "0"
        p.mkdir(parents=True, exist_ok=True)
        (p / "U").write_text("d" * 4000)
    vtk = c / "VTK"
    vtk.mkdir(exist_ok=True)
    (vtk / "case_100.vtu").write_text("v" * 9000)
    return c


@pytest.fixture
def work(tmp_path):
    root = tmp_path / "work"
    (root / "study-a").mkdir(parents=True)
    _case(root / "study-a", "cyl")
    (root / "study-b").mkdir(parents=True)
    _case(root / "study-b", "duct", times=("0", "2"), procs=0)
    return root


# -- what it will never touch --------------------------------------------------

@pytest.mark.parametrize("survivor", [
    "cyl/system/controlDict",
    "cyl/constant/polyMesh/points",
    "cyl/postProcessing/forces/forces.dat",
    "cyl/log.simpleFoam",
    "cyl/Allrun",
    "cyl/build.py",
    "cyl/mesh_z.png",
    "cyl/0.orig",
    "cyl/1/U",
])
def test_the_irreplaceable_is_never_a_candidate(work, survivor):
    """Each of these is either an input, the only record of what happened, or the
    latest state a restart resumes from. `0.orig` matters twice over: it matches the
    time-directory pattern, and deleting it is how a case that could be re-run
    becomes one that cannot."""
    usages = disk.scan(work, study="study-a")
    named = {str(c.path) for c in usages[0].candidates}
    assert not any(str(work / "study-a" / survivor) in n or n.endswith(survivor)
                   for n in named), f"{survivor} was offered up for deletion"


def test_the_initial_condition_is_not_a_time_directory(work):
    """`0` matches the numeric pattern and is an input. It is excluded before the
    pattern is ever consulted, and this is the test that says so."""
    usages = disk.scan(work, study="study-a")
    assert (work / "study-a" / "cyl" / "0" / "U").exists()
    assert not any(c.path.name == "0" for c in usages[0].candidates)


# -- what it will offer --------------------------------------------------------

def test_processor_directories_and_old_times_and_dumps_are_offered(work):
    usages = disk.scan(work, study="study-a")
    offered = {c.path.name for c in usages[0].candidates}
    assert "processor0" in offered and "processor1" in offered
    assert "0.5" in offered, "an intermediate time is regenerable"
    assert "1" not in offered, "the latest time is what a restart resumes from"
    assert "VTK" in offered


def test_drop_latest_is_available_and_is_not_the_default(work):
    kept = disk.scan(work, study="study-a")
    dropped = disk.scan(work, study="study-a", keep_latest=False)
    assert not any(c.path.name == "1" for c in kept[0].candidates)
    assert any(c.path.name == "1" for c in dropped[0].candidates)


# -- the report ----------------------------------------------------------------

def test_the_report_ranks_by_size_and_says_what_could_be_freed(work):
    usages = disk.scan(work)
    assert [u.name for u in usages] == ["study-a", "study-b"], "biggest first"
    text = disk.render_report(usages, work, quota_gb=20)
    assert "regenerable" in text and "study-a" in text and "study-b" in text


def test_the_report_changes_nothing(work):
    before = sorted(str(p) for p in work.rglob("*"))
    disk.render_report(disk.scan(work), work, quota_gb=20)
    assert sorted(str(p) for p in work.rglob("*")) == before


def test_report_json_carries_the_two_numbers_a_caller_acts_on(work, capsys):
    disk.main(["report", "--work", str(work), "--json"])
    body = json.loads(capsys.readouterr().out)
    assert body["total_bytes"] > 0 and body["regenerable_bytes"] > 0
    assert body["regenerable_bytes"] < body["total_bytes"]


# -- the deletion itself -------------------------------------------------------

def test_nothing_is_offered_twice(work):
    """`VTK/` and `VTK/case_100.vtu` were both offered the first time this ran, which
    double-counts the bytes a prune would free and then fails the second removal
    because the first took its parent with it."""
    usages = disk.scan(work, study="study-a")
    paths = [c.path for c in usages[0].candidates]
    assert len(paths) == len(set(paths))
    for p in paths:
        assert not any(other in p.parents for other in paths), f"{p} is inside another candidate"


def test_prune_removes_nothing_without_apply(work, capsys):
    rc = disk.main(["prune", "--work", str(work), "--study", "study-a"])
    said = capsys.readouterr().out
    assert rc == 0 and "would remove" in said
    assert (work / "study-a" / "cyl" / "processor0").exists()


def test_prune_refuses_to_guess_which_study(work, capsys):
    """One volume holds every study the account has. A prune that defaulted to all
    of them is one keystroke from removing the time directories of a solve running
    next door."""
    rc = disk.main(["prune", "--work", str(work)])
    assert rc == 2
    assert "say which study" in capsys.readouterr().err


def test_prune_with_apply_frees_the_space_and_keeps_the_case_runnable(work, capsys):
    rc = disk.main(["prune", "--work", str(work), "--study", "study-a", "--apply"])
    assert rc == 0 and "freed" in capsys.readouterr().out
    c = work / "study-a" / "cyl"
    assert not (c / "processor0").exists() and not (c / "0.5").exists()
    for survivor in ("system/controlDict", "constant/polyMesh/points", "0/U", "1/U",
                     "log.simpleFoam", "Allrun", "mesh_z.png",
                     "postProcessing/forces/forces.dat"):
        assert (c / survivor).exists(), f"{survivor} did not survive the prune"


def test_pruning_one_study_leaves_the_other_alone(work):
    disk.main(["prune", "--work", str(work), "--study", "study-a", "--apply"])
    assert (work / "study-b" / "duct" / "0" / "U").exists()
    assert (work / "study-b" / "duct" / "2" / "U").exists()


# -- the trap the quota check itself fell into ---------------------------------

def test_a_symlink_is_not_followed_and_not_counted_twice(work):
    """`du -sm -- /work` measured the symlink, not the tree, and answered 1 MB for
    every instance on every call until `-H` was added on 2026-09-06 -- so the quota
    was blind for the whole life of the module. This is the same trap from the other
    side: an OpenFOAM tree is full of internal links, and following them reports a
    directory as bigger than it is and invites deleting the wrong one."""
    c = work / "study-a" / "cyl"
    try:
        (c / "0.link").symlink_to(c / "0", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not allow symlinks without elevation")
    usages = disk.scan(work, study="study-a")
    plain = disk._tree_bytes(c / "0")
    assert usages[0].bytes < 10 * plain, "the link's target was counted again"
    assert not any(c.path.name == "0.link" for c in usages[0].candidates)


def test_a_directory_that_is_not_a_case_is_reported_but_offers_nothing(tmp_path):
    """An upload directory, a notes folder: it takes space and none of it is
    regenerable, so it is counted and nothing is proposed."""
    work = tmp_path / "work"
    (work / "uploads").mkdir(parents=True)
    (work / "uploads" / "hull.stl").write_text("s" * 2000)
    usages = disk.scan(work)
    assert usages[0].bytes > 0 and usages[0].candidates == []


# -- the mesh that moves -------------------------------------------------------


def test_a_time_directory_that_carries_its_own_mesh_is_kept_whole(work):
    """On a moving mesh `<time>/polyMesh/points` IS the mesh at that instant.

    Nothing regenerates it short of re-running the solve, so a prune that takes the
    time directory destroys the only record the run keeps of its own motion -- and
    pruning is routine here, because /work is one shared 20 GB quota. The reason line on
    the times that do go names the test they passed.
    """
    case = work / "study-a" / "cyl"
    (case / "0.5" / "polyMesh").mkdir(parents=True)
    (case / "0.5" / "polyMesh" / "points").write_text("p" * 400)

    usages = disk.scan(work, study="study-a")
    offered = {c.path.name for c in usages[0].candidates}
    assert "0.5" not in offered, "the deformed mesh at t = 0.5 has nothing to rebuild it"
    assert not any("polyMesh" in str(c.path) for c in usages[0].candidates)


def test_an_ordinary_time_directory_still_goes_and_says_why(work):
    usages = disk.scan(work, study="study-a")
    dropped = [c for c in usages[0].candidates if c.path.name == "0.5"]
    assert dropped, "a plain intermediate time is still regenerable"
    assert "carries no mesh of its own" in dropped[0].why


FIELDS = ("U", "p_rgh", "alpha.water", "k", "omega", "nut", "phi")


def _moving_case(root, name="hull", times=(3.6, 3.8, 4.0, 4.2, 4.4), mesh_bytes=600,
                 field_bytes=4000):
    """The Wigley free phase: every written time carries its own deformed mesh.

    A morphing mesh writes `<time>/polyMesh/points` at every write, so this shape --
    not some corner of it -- is what a moving-mesh study looks like on the volume, and
    the field data beside those meshes is the bulk of it.
    """
    case = _case(root, name, times=(), procs=0)
    for t in times:
        d = case / f"{t:g}"
        (d / "polyMesh").mkdir(parents=True, exist_ok=True)
        (d / "polyMesh" / "points").write_text("p" * mesh_bytes)
        (d / "uniform").mkdir(exist_ok=True)
        (d / "uniform" / "time").write_text("t" * 50)
        for f in FIELDS:
            (d / f).write_text("f" * field_bytes)
    return case


def test_the_fields_beside_a_deformed_mesh_are_still_reclaimed(tmp_path):
    """Keeping the whole time directory answered F-56 with "nothing regenerable found".

    Measured on this shape before the fix: 0 candidates and 0 B regenerable, in both
    modes, on a case where the fields are most of the volume. Only the `polyMesh`
    subtree is irreplaceable; `U`, `p_rgh`, `alpha.water`, `k`, `omega`, `nut` and `phi`
    at an intermediate time are exactly what this module's own header classifies as
    regenerable, and a session that cannot write because /work is over quota is the
    situation the module exists for.
    """
    work = tmp_path / "work"
    (work / "study-m").mkdir(parents=True)
    _moving_case(work / "study-m")

    usages = disk.scan(work, study="study-m")
    offered = {str(c.path) for c in usages[0].candidates}
    assert usages[0].regenerable > 0, "a moving-mesh case must not report 0 B reclaimable"
    for t in ("3.6", "3.8", "4"):  # 4.0 is written as `4`
        for f in FIELDS:
            assert str(work / "study-m" / "hull" / t / f) in offered, (
                f"{t}/{f} is regenerable and was not offered"
            )
    assert not any("polyMesh" in str(c.path) for c in usages[0].candidates), (
        "the deformed mesh at that instant is the only record of the run's own motion"
    )
    latest = work / "study-m" / "hull" / "4.4"
    assert not any(c.path == latest or latest in c.path.parents
                   for c in usages[0].candidates), (
        "the latest time is what a restart resumes from"
    )


def test_a_moving_mesh_case_frees_the_bulk_of_its_volume_rather_than_nothing(tmp_path):
    """F-56 in miniature: /work over quota, and the answer has to be a number that
    helps. The fields are roughly seven eighths of this tree and all of them come back
    from a re-run; the meshes are the small remainder that does not."""
    work = tmp_path / "work"
    (work / "study-m").mkdir(parents=True)
    _moving_case(work / "study-m", times=tuple(3.6 + 0.1 * i for i in range(45)))

    usage = disk.scan(work, study="study-m")[0]
    assert usage.regenerable > usage.bytes // 2, (
        f"only {usage.regenerable} of {usage.bytes} offered; the fields are most of it"
    )


def test_drop_latest_on_a_moving_mesh_reaches_the_last_times_fields_but_not_its_mesh(tmp_path):
    """`--drop-latest` is the mode a person reaches for when the volume is full. Before
    the fix it freed nothing at all here, which is the worst possible answer to give
    someone who has already accepted that a restart will not resume."""
    work = tmp_path / "work"
    (work / "study-m").mkdir(parents=True)
    _moving_case(work / "study-m", times=(1.0, 2.0))

    usage = disk.scan(work, study="study-m", keep_latest=False)[0]
    offered = {str(c.path) for c in usage.candidates}
    assert str(work / "study-m" / "hull" / "2" / "U") in offered
    assert not any("polyMesh" in str(c.path) for c in usage.candidates)


def test_pruning_a_moving_mesh_case_keeps_every_deformed_mesh_and_takes_the_fields(tmp_path):
    """The prune itself, end to end: what the report promised is what the disk shows."""
    work = tmp_path / "work"
    (work / "study-m").mkdir(parents=True)
    case = _moving_case(work / "study-m", times=(1.0, 2.0, 3.0))

    rc = disk.main(["prune", "--work", str(work), "--study", "study-m", "--apply"])
    assert rc == 0
    for t in ("1", "2", "3"):
        assert (case / t / "polyMesh" / "points").exists(), (
            f"the deformed mesh at t = {t} was destroyed"
        )
    for t in ("1", "2"):
        for f in FIELDS:
            assert not (case / t / f).exists(), f"{t}/{f} is regenerable and should be gone"
    for f in FIELDS:
        assert (case / "3" / f).exists(), "the latest time is what a restart resumes from"


def test_a_moving_mesh_candidate_says_the_mesh_beside_it_is_kept(tmp_path):
    """A prune that frees less than the directory listing suggests has to say why on the
    line itself, because nobody reads the module to find out."""
    work = tmp_path / "work"
    (work / "study-m").mkdir(parents=True)
    _moving_case(work / "study-m", times=(1.0, 2.0))

    usage = disk.scan(work, study="study-m")[0]
    whys = {c.why for c in usage.candidates}
    assert any("the deformed mesh beside it is kept" in w for w in whys), whys


def test_drop_latest_does_not_reach_a_moving_meshs_last_mesh_either(work):
    """`--keep-latest False` is for a case whose numbers are already out; it is not
    consent to delete a mesh that only exists inside a time directory."""
    case = work / "study-b" / "duct"
    (case / "2" / "polyMesh").mkdir(parents=True)
    (case / "2" / "polyMesh" / "points").write_text("p" * 400)

    usages = disk.scan(work, study="study-b", keep_latest=False)
    assert not any(c.path.name == "2" for c in usages[0].candidates)
