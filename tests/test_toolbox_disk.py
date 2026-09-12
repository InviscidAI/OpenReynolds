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


def test_drop_latest_does_not_reach_a_moving_meshs_last_mesh_either(work):
    """`--keep-latest False` is for a case whose numbers are already out; it is not
    consent to delete a mesh that only exists inside a time directory."""
    case = work / "study-b" / "duct"
    (case / "2" / "polyMesh").mkdir(parents=True)
    (case / "2" / "polyMesh" / "points").write_text("p" * 400)

    usages = disk.scan(work, study="study-b", keep_latest=False)
    assert not any(c.path.name == "2" for c in usages[0].candidates)
