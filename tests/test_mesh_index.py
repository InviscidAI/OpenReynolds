"""The workspace's index of finished meshes (#35).

Two studies twenty-four minutes apart meshed the same 2D text, 728 s and then 402 s,
while a finished, checkMesh-clean copy of the first sat in `/work/<first study>/mesh`.
Nothing said it was there. These pin that a finished mesh is written down, that what is
written down is offered back to the next study and to the desk, and that the index never
costs a mesh or a session when the workspace will not answer.

Local workspaces only, both ways: a hosted study gets a workspace of its own (#92), so
there the index could only ever hold the current study's own meshes.
"""

from __future__ import annotations

import json

from openreynolds.backend.base import ExecResult
from openreynolds.backend.local import LocalBackend
from openreynolds.cad import mesh_index
from openreynolds.cli import _meshes_note

from test_cad_agent import DONE, PASSES, REFUSES, answers, block, checking, desk, kernelled


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


def _shared(**attrs):
    """A fake workspace studies share, which is what a local one is."""
    class Fake:
        workspace_root = "/work"
        shared_workspace = True

        def __init__(self):
            self.execs: list[str] = []
            self.cwds: list[str | None] = []
            self.result = ExecResult(0, "", False, None)

        def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
            self.execs.append(cmd)
            self.cwds.append(cwd)
            return self.result

    made = Fake()
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


def test_the_index_is_read_in_one_round_trip(tmp_path):
    """Both the tail and the per-entry existence test are charged to the desk's budget
    before its first model call, so they are one command."""
    backend = _shared()
    mesh_index.read(backend)
    assert len(backend.execs) == 1


def test_no_command_carries_an_absolute_path(tmp_path):
    r"""Whether an absolute Windows path in command text resolves depends on which bash
    is on PATH. `mkdir -p C:/Users/.../.reynolds` made the directory on the developer's
    Git bash and not on the Windows CI runner, where six of these tests failed as
    "expected a path, got nothing" -- including one that could not open the index the
    shell had just reported creating. Every path in a command is relative now, and the
    root goes through `cwd`, which the backend resolves in Python.
    """
    backend = _shared(workspace_root=str(tmp_path))
    case = (tmp_path / "20260920-155504-4379/mesh").as_posix()
    mesh_index.record(backend, _line(case), str(tmp_path))
    mesh_index.retire(backend, case, str(tmp_path))
    mesh_index.read(backend, str(tmp_path))
    assert backend.execs, "nothing ran, so nothing was pinned"
    for cmd in backend.execs:
        assert str(tmp_path) not in cmd and tmp_path.as_posix() not in cmd
        assert ":" not in cmd.split(".reynolds")[0]
    # The root reached the backend as `cwd`, which is resolved in Python, not by a shell.
    assert all(c == str(tmp_path) for c in backend.cwds)


def test_an_append_that_failed_is_not_reported_as_recorded():
    """A `>>` into a read-only or full volume fails quietly: `exec` does not raise and
    the exit code is the only thing that says so."""
    backend = _shared()
    backend.result = ExecResult(1, "Read-only file system", False, None)
    assert mesh_index.record(backend, _line("/work/s/mesh")) is False
    backend.result = ExecResult(0, "", False, None)
    assert mesh_index.record(backend, _line("/work/s/mesh")) is True


def test_a_workspace_studies_do_not_share_is_never_recorded_to_or_read(tmp_path):
    """Every hosted study gets its own workspace (#92), so a hosted index can only ever
    hold the current study's meshes: it cannot answer #35, and offering a study its own
    earlier meshes as though somebody else built them is worse than offering nothing."""
    backend = _shared()
    backend.shared_workspace = False
    assert mesh_index.record(backend, _line("/work/s/mesh")) is False
    assert mesh_index.retire(backend, "/work/s/mesh") is False
    assert mesh_index.read(backend) == []
    assert backend.execs == []
    assert _meshes_note(backend, "/work/s") == ""


def test_a_workspace_that_will_not_answer_costs_nothing():
    """A mesh that cannot be recorded is still a finished mesh; a session that cannot
    read the index builds, which is what it did before there was one."""
    class Gone:
        workspace_root = "/work"
        shared_workspace = True

        def exec(self, *a, **k):
            raise ConnectionError("workspace recycled")

    assert mesh_index.record(Gone(), _line("/work/s/mesh")) is False
    assert mesh_index.read(Gone()) == []


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


def _appended(backend):
    return [c for c in backend.execs
            if ">>" in c and mesh_index.INDEX_REL in c and '"retired"' not in c]


def _retired(backend):
    return [c for c in backend.execs if mesh_index.INDEX_REL in c and '"retired"' in c]


def _first_message(made):
    return made.provider.calls[0]["messages"][0]["content"][0]["text"]


def _local(backend):
    """The fake workspace, read as a local one. `FakeBackend` is the hosted protocol by
    default (`Backend.shared_workspace` is False), and the index is local-only."""
    backend.shared_workspace = True
    return backend


def test_a_finished_mesh_is_written_to_the_index(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(_local(backend))
    made = desk(backend, store, [DONE])
    result = made.run("the text Inviscid AI in a 2D channel", case="mesh")
    assert result.ok
    (cmd,) = _appended(backend)
    assert '"case_dir":"study/mesh"' in cmd and '"cells":3750' in cmd
    assert '"study_id":"study"' in cmd
    assert '"two_d":true' in cmd and '"patches":["inlet"]' in cmd and '"regions":[""]' in cmd


def test_the_study_is_named_as_the_platform_names_it(backend, store, monkeypatch):
    """`Backend.study_id` is the real value; the home's last segment is `work` for a
    study old enough to own the workspace root."""
    checking(monkeypatch, PASSES)
    kernelled(_local(backend))
    backend.study_id = "stu_2bK9"
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    (cmd,) = _appended(backend)
    assert '"study_id":"stu_2bK9"' in cmd


def test_a_mesh_that_never_passed_is_not_written(backend, store, monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(_local(backend))
    made = desk(backend, store, [DONE], mesher_max_steps=2)
    result = made.run("a duct", case="mesh")
    assert not result.ok
    assert _appended(backend) == []


def test_a_mesh_nobody_reviewed_is_not_written(backend, store, monkeypatch):
    """A run that stopped on steps, time or a provider error is re-verified so a built
    mesh is not reported as nothing, and marked "not reviewed" because no reviewer
    judged the shape against the request. Such a mesh can pass `checkMesh` and still be
    partial or the wrong shape, and the index would offer it under that request's words.
    """
    checking(monkeypatch, PASSES)
    kernelled(_local(backend))
    made = desk(backend, store, [block('print("still working")')], mesher_max_steps=1)
    result = made.run("the text Inviscid AI in a 2D channel", case="mesh")
    assert result.ok and result.check is not None and result.review is None
    assert _appended(backend) == []


def test_the_case_the_desk_starts_on_is_retired_from_the_index(backend, store, monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(_local(backend))
    made = desk(backend, store, [DONE], mesher_max_steps=2)
    made.run("a duct", case="mesh")
    (cmd,) = _retired(backend)
    assert '"case_dir":"study/mesh"' in cmd


def test_the_desk_is_told_what_is_already_built(backend, store, monkeypatch):
    other = "/work/20260920-155504-4379/mesh"
    checking(monkeypatch, PASSES)
    kernelled(_local(backend))
    answers(backend, {"tail -n": ExecResult(0, json.dumps(_line(other)) + "\n", False, None)})
    made = desk(backend, store, [DONE])
    made.run("the text Inviscid AI in a 2D channel", case="mesh")
    first = _first_message(made)
    assert "Meshes already built on this workspace" in first and other in first


def test_the_desk_is_not_offered_the_case_it_is_rebuilding(backend, store, monkeypatch):
    mine = "/work/study/mesh"
    checking(monkeypatch, PASSES)
    kernelled(_local(backend))
    answers(backend, {"tail -n": ExecResult(0, json.dumps(_line(mine)) + "\n", False, None)})
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    assert "already built" not in _first_message(made)


def test_the_desk_is_told_nothing_when_nothing_is_built(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(_local(backend))
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    assert "already built" not in _first_message(made)


def test_a_hosted_desk_neither_offers_nor_records(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend)  # left as the hosted protocol: studies do not share the workspace
    answers(backend, {"tail -n": ExecResult(
        0, json.dumps(_line("/work/20260920-155504-4379/mesh")) + "\n", False, None)})
    made = desk(backend, store, [DONE])
    assert made.run("a duct", case="mesh").ok
    assert "already built" not in _first_message(made)
    assert _appended(backend) == [] and _retired(backend) == []


# -- the session-start note ------------------------------------------------------


def test_the_workspace_ready_note_offers_other_studies_meshes(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    other = _polymesh(tmp_path, "20260920-155504-4379/mesh")
    mine = _polymesh(tmp_path, "20260920-161908-c7ef/mesh")
    for case in (other, mine):
        mesh_index.record(backend, _line(case), backend.workspace_root)
    note = _meshes_note(backend, str(tmp_path / "20260920-161908-c7ef"))
    assert other in note and mine not in note
