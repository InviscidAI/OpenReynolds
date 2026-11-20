"""The workspace's index of finished meshes (#35).

Two studies twenty-four minutes apart meshed the same 2D text, 728 s and then 402 s,
while a finished, checkMesh-clean copy of the first sat in `/work/<first study>/mesh`.
Nothing said it was there. These pin that a finished mesh is written down, that what is
written down is offered back to the next study and to the desk, and that the index never
costs a mesh or a session when the workspace will not answer.

Local workspaces only, both ways: a hosted study gets a workspace of its own (#92), so
there the index could only ever hold the current study's own meshes.

None of it runs a command, which is what Windows CI caught: `exec` is `bash -lc`, and on
a Windows runner `bash` is WSL's stub (`conftest.a_kernel_can_come_up`). These tests had
never run there -- every earlier CI run on the branch was `action_required` -- and the
first one that did failed six of them.
"""

from __future__ import annotations

import json

from openreynolds.backend.local import LocalBackend
from openreynolds.cad import mesh_index
from openreynolds.cli import _meshes_note

from test_cad_agent import DONE, PASSES, REFUSES, block, checking, desk, kernelled


def _polymesh(root, case, region=""):
    """A meshed case directory, named the way the index names it: forward slashes, which
    is what a native Windows workspace's own backslash root becomes (`_posix`)."""
    faces = root / case / "constant" / region / "polyMesh" / "faces"
    faces.parent.mkdir(parents=True, exist_ok=True)
    faces.write_text("0()\n")
    return (root / case).as_posix()


def _line(case_dir, request="the text Inviscid AI in a 2D channel", cells=3750,
          regions=("",)):
    check = type("C", (), {"cells": cells, "two_d": True, "bounds": [0, 0, 0, 1, 1, 0.01],
                           "patches": [{"name": "inlet"}, {"name": "walls"}],
                           "regions": list(regions)})()
    return mesh_index.entry(check, case_dir, request, "20260920-155504-4379")


def _shared(root, **attrs):
    """A workspace studies share, which is what a local one is, with no shell behind it
    at all -- `exec` raises, so anything reaching for one fails the test."""
    made = LocalBackend(root=root, bashrc="")

    def no_shell(*a, **k):
        raise AssertionError("the mesh index must not need a shell")

    made.exec = no_shell
    for name, value in attrs.items():
        setattr(made, name, value)
    return made


def test_a_recorded_mesh_is_read_back_newest_first_one_per_case(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    root = backend.workspace_root
    first = _polymesh(tmp_path, "a/mesh")
    second = _polymesh(tmp_path, "b/mesh")
    assert mesh_index.record(backend, _line(first, cells=1), root)
    assert mesh_index.record(backend, _line(second, cells=2), root)
    assert mesh_index.record(backend, _line(first, cells=3), root)  # the same case, re-meshed

    got = mesh_index.read(backend, root)
    assert [(g["case_dir"], g["cells"]) for g in got] == [(first, 3), (second, 2)]


def test_a_mesh_whose_directory_is_gone_is_not_offered(tmp_path):
    """What is offered is what can be copied: a deleted study drops out."""
    backend = LocalBackend(root=tmp_path, bashrc="")
    kept = _polymesh(tmp_path, "kept/mesh")
    root = backend.workspace_root
    mesh_index.record(backend, _line(kept), root)
    mesh_index.record(backend, _line((tmp_path / "deleted/mesh").as_posix()), root)
    assert [g["case_dir"] for g in mesh_index.read(backend, root)] == [kept]


def test_a_multi_region_mesh_is_offered_and_names_its_regions(tmp_path):
    """A CHT mesh is `constant/<region>/polyMesh`, the layout `cad.check.mesh_regions`
    accepts. Testing only `constant/polyMesh/faces` dropped every one of them on every
    read, so they were recorded and never offered."""
    backend = LocalBackend(root=tmp_path, bashrc="")
    case = _polymesh(tmp_path, "cht/mesh", region="air")
    _polymesh(tmp_path, "cht/mesh", region="solid")
    mesh_index.record(backend, _line(case, regions=("air", "solid")), backend.workspace_root)
    (got,) = mesh_index.read(backend, backend.workspace_root)
    assert got["case_dir"] == case
    assert mesh_index.mesh_paths(got) == [f"{case}/constant/air/polyMesh",
                                          f"{case}/constant/solid/polyMesh"]


def test_the_offer_names_the_mesh_to_copy_not_the_case_directory(tmp_path):
    """`cp -r A/mesh B/mesh` lands the mesh at `B/mesh/mesh/constant/polyMesh`, where
    the finish check cannot see it, and brings any solved times and `processor*` with
    it: the desk's case directory already exists before the first model call."""
    note = mesh_index.offer([_line("/work/20260920-155504-4379/mesh")])
    assert "/work/20260920-155504-4379/mesh/constant/polyMesh" in note
    assert "not the case directory" in note and "processor*" in note


def test_a_retired_entry_is_not_offered_and_hides_the_one_before_it(tmp_path):
    """A study that re-meshes a case it already recorded overwrites `constant/polyMesh`.
    If that run then fails its check, nothing new is recorded and the old clean entry
    still points at the mesh that is now half-written: the directory is there, so the
    offer passed it and the next study copied it."""
    backend = LocalBackend(root=tmp_path, bashrc="")
    root = backend.workspace_root
    mine = _polymesh(tmp_path, "a/mesh")
    other = _polymesh(tmp_path, "b/mesh")
    mesh_index.record(backend, _line(mine), root)
    mesh_index.record(backend, _line(other), root)
    assert mesh_index.retire(backend, mine, root)
    assert [g["case_dir"] for g in mesh_index.read(backend, root)] == [other]

    # And a mesh that finishes after the retire is on the index again.
    mesh_index.record(backend, _line(mine, cells=99), root)
    assert [(g["case_dir"], g["cells"]) for g in mesh_index.read(backend, root)] \
        == [(mine, 99), (other, 3750)]


def test_retiring_with_no_index_yet_creates_nothing(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    mesh_index.retire(backend, str(tmp_path / "a/mesh"), backend.workspace_root)
    assert mesh_index.read(backend, backend.workspace_root) == []
    assert not (tmp_path / mesh_index.INDEX_REL).exists()


def test_a_study_is_not_offered_its_own_meshes(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    mine = _polymesh(tmp_path, "this-study/mesh")
    other = _polymesh(tmp_path, "other-study/mesh")
    for case in (mine, other):
        mesh_index.record(backend, _line(case), backend.workspace_root)
    got = mesh_index.read(backend, backend.workspace_root, exclude=str(tmp_path / "this-study"))
    assert [g["case_dir"] for g in got] == [other]
    # An exclude that is a prefix of a name, not a directory, does not drop it.
    got = mesh_index.read(backend, backend.workspace_root, exclude=str(tmp_path / "other"))
    assert {g["case_dir"] for g in got} == {mine, other}


def test_a_line_that_does_not_parse_is_skipped_not_fatal(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    good = _polymesh(tmp_path, "good/mesh")
    mesh_index.record(backend, _line(good), backend.workspace_root)
    with open(mesh_index.index_path(backend.workspace_root), "a") as f:
        f.write("{not json\n[1, 2]\n\n")
    assert [g["case_dir"] for g in mesh_index.read(backend, backend.workspace_root)] == [good]


def test_no_index_is_no_offer(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    assert mesh_index.read(backend, backend.workspace_root) == []
    assert mesh_index.offer([]) == ""


def test_the_index_never_runs_a_command(tmp_path):
    """The whole of it works with no shell, which is the bug Windows CI caught.

    `exec` runs `bash -lc`, and on a Windows runner `bash` is WSL's stub: every command
    fails whatever its text says, which is what `conftest.a_kernel_can_come_up` measures
    before skipping the tests that need a shell. Six of these tests failed there as
    "expected a path, got nothing" -- one of them a `FileNotFoundError` on the index the
    shell had just reported creating -- and no amount of path shaping could have fixed
    it. A workspace studies share is a directory on this machine, so the file is opened
    in Python and `exec` is never reached.
    """
    backend = _shared(tmp_path)
    root = backend.workspace_root
    case = _polymesh(tmp_path, "20260920-155504-4379/mesh")
    assert mesh_index.record(backend, _line(case), root)
    assert [g["case_dir"] for g in mesh_index.read(backend, root)] == [case]
    assert mesh_index.retire(backend, case, root)
    assert mesh_index.read(backend, root) == []


def test_a_write_that_cannot_land_is_not_reported_as_recorded(tmp_path):
    """A full or read-only volume, staged here as a file where the index directory has
    to go. Reporting success would put a line in the index that nobody could read back.
    """
    backend = _shared(tmp_path)
    (tmp_path / mesh_index.INDEX_DIR).write_text("not a directory\n")
    assert mesh_index.record(backend, _line("/work/s/mesh"), backend.workspace_root) is False
    assert mesh_index.read(backend, backend.workspace_root) == []


def test_a_workspace_studies_do_not_share_is_never_recorded_to_or_read(tmp_path):
    """Every hosted study gets its own workspace (#92), so a hosted index can only ever
    hold the current study's meshes: it cannot answer #35, and offering a study its own
    earlier meshes as though somebody else built them is worse than offering nothing."""
    backend = _shared(tmp_path, shared_workspace=False)
    case = _polymesh(tmp_path, "some-study/mesh")
    root = backend.workspace_root
    assert mesh_index.record(backend, _line(case), root) is False
    assert mesh_index.retire(backend, case, root) is False
    assert mesh_index.read(backend, root) == []
    assert not (tmp_path / mesh_index.INDEX_DIR).exists()
    assert _meshes_note(backend, f"{root}/some-study") == ""


def test_a_workspace_that_cannot_be_located_costs_nothing():
    """A mesh that cannot be recorded is still a finished mesh; a session that cannot
    read the index builds, which is what it did before there was one. A backend that
    declares a shared workspace but cannot say where it is on this machine is the one
    case `_real_dir` returns None for."""
    class Nowhere:
        workspace_root = "/work"
        shared_workspace = True

    assert mesh_index.record(Nowhere(), _line("/work/s/mesh")) is False
    assert mesh_index.retire(Nowhere(), "/work/s/mesh") is False
    assert mesh_index.read(Nowhere()) == []


def test_the_offer_says_what_it_is_and_where(tmp_path):
    item = _line("/work/20260920-155504-4379/mesh")
    note = mesh_index.offer([item])
    assert "/work/20260920-155504-4379/mesh" in note
    assert "Inviscid AI" in note and "3,750 cells" in note and "2D" in note
    assert "inlet, walls" in note and mesh_index.INDEX_REL in note


# -- the desk -------------------------------------------------------------------
#
# The CAD desk is what builds meshes (`cad/agent.py`); the finish check is handed its
# verdict, as in `test_cad_agent.py`, because what is under test is what the desk does
# with a passing or failing one.


def _entries(tmp_path):
    """The index as it was actually written. The desk tests used to read the command the
    module sent; it sends none now, so they read the file."""
    index = tmp_path / mesh_index.INDEX_REL
    if not index.is_file():
        return []
    return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]


def _appended(tmp_path):
    return [e for e in _entries(tmp_path) if not e.get("retired")]


def _retired(tmp_path):
    return [e for e in _entries(tmp_path) if e.get("retired")]


def _first_message(made):
    return made.provider.calls[0]["messages"][0]["content"][0]["text"]


def _local(backend, tmp_path):
    """The fake workspace, read as a local one: studies share it, and `/work` is a real
    directory on this machine.

    `FakeBackend` is the hosted protocol by default (`Backend.shared_workspace` is
    False) and keeps its files in a dict, so the index -- a real file under the
    workspace root, opened in Python rather than through a shell -- needs a real
    directory to live in. `real_path` is what `LocalBackend` offers for this.
    """
    backend.shared_workspace = True
    backend.real_path = lambda path: _mapped(path, tmp_path)
    return backend


def _mapped(path, tmp_path):
    text = str(path)
    rel = text[len("/work"):].lstrip("/") if text.startswith("/work") else text
    return tmp_path / rel


def test_a_finished_mesh_is_written_to_the_index(backend, store, monkeypatch, tmp_path):
    checking(monkeypatch, PASSES)
    kernelled(_local(backend, tmp_path))
    made = desk(backend, store, [DONE])
    result = made.run("the text Inviscid AI in a 2D channel", case="mesh")
    assert result.ok
    (written,) = _appended(tmp_path)
    assert written["case_dir"] == "study/mesh" and written["cells"] == 3750
    assert written["study_id"] == "study" and written["two_d"] is True
    assert written["patches"] == ["inlet"] and written["regions"] == [""]


def test_the_study_is_named_as_the_platform_names_it(backend, store, monkeypatch, tmp_path):
    """`Backend.study_id` is the real value; the home's last segment is `work` for a
    study old enough to own the workspace root."""
    checking(monkeypatch, PASSES)
    kernelled(_local(backend, tmp_path))
    backend.study_id = "stu_2bK9"
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    (written,) = _appended(tmp_path)
    assert written["study_id"] == "stu_2bK9"


def test_a_mesh_that_never_passed_is_not_written(backend, store, monkeypatch, tmp_path):
    checking(monkeypatch, REFUSES)
    kernelled(_local(backend, tmp_path))
    made = desk(backend, store, [DONE], mesher_max_steps=2)
    result = made.run("a duct", case="mesh")
    assert not result.ok
    assert _appended(tmp_path) == []


def test_a_mesh_nobody_reviewed_is_not_written(backend, store, monkeypatch, tmp_path):
    """A run that stopped on steps, time or a provider error is re-verified so a built
    mesh is not reported as nothing, and marked "not reviewed" because no reviewer
    judged the shape against the request. Such a mesh can pass `checkMesh` and still be
    partial or the wrong shape, and the index would offer it under that request's words.
    """
    checking(monkeypatch, PASSES)
    kernelled(_local(backend, tmp_path))
    made = desk(backend, store, [block('print("still working")')], mesher_max_steps=1)
    result = made.run("the text Inviscid AI in a 2D channel", case="mesh")
    assert result.ok and result.check is not None and result.review is None
    assert _appended(tmp_path) == []


def test_the_case_the_desk_starts_on_is_retired_from_the_index(backend, store, monkeypatch,
                                                              tmp_path):
    checking(monkeypatch, REFUSES)
    kernelled(_local(backend, tmp_path))
    _polymesh(tmp_path, "study/mesh")
    mesh_index.record(backend, _line("/work/study/mesh"), "/work")
    made = desk(backend, store, [DONE], mesher_max_steps=2)
    made.run("a duct", case="mesh")
    (gone,) = _retired(tmp_path)
    assert gone["case_dir"] == "study/mesh"
    # And what it retired is no longer offered, though the mesh is still on disk.
    assert mesh_index.read(backend, "/work") == []


def test_the_desk_is_told_what_is_already_built(backend, store, monkeypatch, tmp_path):
    other = "/work/20260920-155504-4379/mesh"
    checking(monkeypatch, PASSES)
    kernelled(_local(backend, tmp_path))
    _polymesh(tmp_path, "20260920-155504-4379/mesh")
    mesh_index.record(backend, _line(other), "/work")
    made = desk(backend, store, [DONE])
    made.run("the text Inviscid AI in a 2D channel", case="mesh")
    first = _first_message(made)
    assert "Meshes already built on this workspace" in first and other in first


def test_the_desk_is_not_offered_the_case_it_is_rebuilding(backend, store, monkeypatch,
                                                           tmp_path):
    checking(monkeypatch, PASSES)
    kernelled(_local(backend, tmp_path))
    _polymesh(tmp_path, "study/mesh")
    mesh_index.record(backend, _line("/work/study/mesh"), "/work")
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    assert "already built" not in _first_message(made)


def test_the_desk_is_told_nothing_when_nothing_is_built(backend, store, monkeypatch, tmp_path):
    checking(monkeypatch, PASSES)
    kernelled(_local(backend, tmp_path))
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    assert "already built" not in _first_message(made)


def test_a_hosted_desk_neither_offers_nor_records(backend, store, monkeypatch, tmp_path):
    checking(monkeypatch, PASSES)
    # Left as the hosted protocol: studies do not share the workspace. The mesh and an
    # index entry are both staged, so only the gate can be what withholds the offer.
    kernelled(backend)
    backend.real_path = lambda path: _mapped(path, tmp_path)
    _polymesh(tmp_path, "20260920-155504-4379/mesh")
    (tmp_path / mesh_index.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    staged = {**_line("/work/20260920-155504-4379/mesh"),
              "case_dir": "20260920-155504-4379/mesh"}
    (tmp_path / mesh_index.INDEX_REL).write_text(json.dumps(staged) + "\n", encoding="utf-8")
    made = desk(backend, store, [DONE])
    assert made.run("a duct", case="mesh").ok
    assert "already built" not in _first_message(made)
    assert _retired(tmp_path) == [] and _appended(tmp_path) == [staged]


# -- the session-start note ------------------------------------------------------


def test_the_workspace_ready_note_offers_other_studies_meshes(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    other = _polymesh(tmp_path, "20260920-155504-4379/mesh")
    mine = _polymesh(tmp_path, "20260920-161908-c7ef/mesh")
    for case in (other, mine):
        mesh_index.record(backend, _line(case), backend.workspace_root)
    note = _meshes_note(backend, str(tmp_path / "20260920-161908-c7ef"))
    assert other in note and mine not in note
