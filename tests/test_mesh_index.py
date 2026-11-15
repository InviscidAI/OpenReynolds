"""The workspace's index of finished meshes (#35).

Two studies twenty-four minutes apart meshed the same 2D text, 728 s and then 402 s,
while a finished, checkMesh-clean copy of the first sat in `/work/<first study>/mesh`.
Nothing said it was there. These pin that a finished mesh is written down, that what is
written down is offered back to the next study and to the desk, and that the index never
costs a mesh or a session when the workspace will not answer.
"""

from __future__ import annotations

import json

from openreynolds.backend.base import ExecResult
from openreynolds.backend.local import LocalBackend
from openreynolds.cad import mesh_index
from openreynolds.cli import _meshes_note

from test_cad_agent import DONE, PASSES, REFUSES, answers, checking, desk, kernelled


def _polymesh(root, case):
    faces = root / case / "constant" / "polyMesh" / "faces"
    faces.parent.mkdir(parents=True, exist_ok=True)
    faces.write_text("0()\n")
    return str(root / case)


def _line(case_dir, request="the text Inviscid AI in a 2D channel", cells=3750):
    check = type("C", (), {"cells": cells, "two_d": True, "bounds": [0, 0, 0, 1, 1, 0.01],
                           "patches": [{"name": "inlet"}, {"name": "walls"}]})()
    return mesh_index.entry(check, case_dir, request, "20260920-155504-4379")


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
    """What is offered is what `cp -r` can copy: a deleted study drops out."""
    backend = LocalBackend(root=tmp_path, bashrc="")
    kept = _polymesh(tmp_path, "kept/mesh")
    mesh_index.record(backend, _line(kept), backend.workspace_root)
    mesh_index.record(backend, _line(str(tmp_path / "deleted/mesh")), backend.workspace_root)
    assert [g["case_dir"] for g in mesh_index.read(backend, backend.workspace_root)] == [kept]


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


def test_a_workspace_that_will_not_answer_costs_nothing():
    """A mesh that cannot be recorded is still a finished mesh; a session that cannot
    read the index builds, which is what it did before there was one."""
    class Gone:
        workspace_root = "/work"

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
    return [c for c in backend.execs if ">>" in c and mesh_index.INDEX_REL in c]


def _first_message(made):
    return made.provider.calls[0]["messages"][0]["content"][0]["text"]


def test_a_finished_mesh_is_written_to_the_index(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [DONE])
    result = made.run("the text Inviscid AI in a 2D channel", case="mesh")
    assert result.ok
    (cmd,) = _appended(backend)
    assert "/work/study/mesh" in cmd and '"cells":3750' in cmd and '"study_id":"study"' in cmd
    assert '"two_d":true' in cmd and '"patches":["inlet"]' in cmd


def test_a_mesh_that_never_passed_is_not_written(backend, store, monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    made = desk(backend, store, [DONE], mesher_max_steps=2)
    result = made.run("a duct", case="mesh")
    assert not result.ok
    assert _appended(backend) == []


def test_the_desk_is_told_what_is_already_built(backend, store, monkeypatch):
    other = "/work/20260920-155504-4379/mesh"
    checking(monkeypatch, PASSES)
    kernelled(backend)
    answers(backend, {
        "tail -n": ExecResult(0, json.dumps(_line(other)) + "\n", False, None),
        "for d in": ExecResult(0, other + "\n", False, None),
    })
    made = desk(backend, store, [DONE])
    made.run("the text Inviscid AI in a 2D channel", case="mesh")
    first = _first_message(made)
    assert "Meshes already built on this workspace" in first and other in first


def test_the_desk_is_not_offered_the_case_it_is_rebuilding(backend, store, monkeypatch):
    mine = "/work/study/mesh"
    checking(monkeypatch, PASSES)
    kernelled(backend)
    answers(backend, {
        "tail -n": ExecResult(0, json.dumps(_line(mine)) + "\n", False, None),
        "for d in": ExecResult(0, mine + "\n", False, None),
    })
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    assert "already built" not in _first_message(made)


def test_the_desk_is_told_nothing_when_nothing_is_built(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [DONE])
    made.run("a duct", case="mesh")
    assert "already built" not in _first_message(made)


# -- the session-start note ------------------------------------------------------


def test_the_workspace_ready_note_offers_other_studies_meshes(tmp_path):
    backend = LocalBackend(root=tmp_path, bashrc="")
    other = _polymesh(tmp_path, "20260920-155504-4379/mesh")
    mine = _polymesh(tmp_path, "20260920-161908-c7ef/mesh")
    for case in (other, mine):
        mesh_index.record(backend, _line(case), backend.workspace_root)
    note = _meshes_note(backend, str(tmp_path / "20260920-161908-c7ef"))
    assert other in note and mine not in note
