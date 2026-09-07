"""What "done" means, as a function of the facts, tested without a workspace.

Every rule here is one that shipped broken once: a case written and read as meshed, a
mesh with failing checks handed to a solver, `patch0`/`patch1` out of gmshToFoam with
nobody able to say which end was the inlet, and a mesh nobody could rebuild.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openreynolds.backend.base import ExecResult
from openreynolds.mesher.check import (
    LOOK, Check, largest_length, look_command, read, scale_mismatch, verify,
)

GOOD = {
    "polymesh": True, "cells": 3750, "faces": 15000, "points": 7600,
    "bounds": [0, 0, 0, 0.1, 0.05, 0.001], "two_d": True,
    "patches": [{"name": "inlet", "type": "patch", "nFaces": 20},
                {"name": "outlet", "type": "patch", "nFaces": 20},
                {"name": "walls", "type": "wall", "nFaces": 400}],
    "build": ["Allmesh"], "checkmesh": "Mesh OK.", "checkmesh_ok": True,
    "render": "renders/mesh_look.png",
}


def facts(**changes):
    payload = {k: (list(v) if isinstance(v, list) else v) for k, v in GOOD.items()}
    payload.update(changes)
    return payload


def test_the_good_case_passes():
    check = read(facts(), "mesh", "/work/s/mesh")
    assert check.ok and check.missing == []
    assert check.render == "mesh/renders/mesh_look.png"
    assert check.render_abs == "/work/s/mesh/renders/mesh_look.png"


def test_no_polymesh_is_the_first_thing_said():
    check = read(facts(polymesh=False), "mesh")
    assert not check.ok
    assert "nothing has been meshed yet" in check.missing[0]


def test_a_mesh_that_checkmesh_refuses_is_not_done():
    check = read(facts(checkmesh_ok=False, checkmesh="failed 2 mesh checks"), "mesh")
    assert not check.ok
    assert any("failed 2 mesh checks" in m and "log.checkMesh" in m for m in check.missing)


def test_default_mesher_names_are_refused_by_name():
    """gmshToFoam types and names every patch itself; a case that reaches here with
    `patch0` on it is one where nobody said which surface was the inlet."""
    check = read(facts(patches=[{"name": "patch0", "nFaces": 20},
                                {"name": "defaultFaces", "nFaces": 400}]), "mesh")
    assert not check.ok
    named = next(m for m in check.missing if "default names" in m)
    assert "patch0" in named and "defaultFaces" in named


@pytest.mark.parametrize("name", ["inlet", "outlet", "walls", "frontAndBack", "wall2", "duct3"])
def test_a_name_somebody_chose_is_accepted_even_with_a_digit_in_it(name):
    check = read(facts(patches=[{"name": name, "nFaces": 4},
                                {"name": "rest", "nFaces": 4}]), "mesh")
    assert "default names" not in " ".join(check.missing)


def test_one_patch_is_not_a_flow_case():
    check = read(facts(patches=[{"name": "walls", "nFaces": 400}]), "mesh")
    assert any("at least an inlet" in m for m in check.missing)


def test_patches_with_no_faces_do_not_count():
    check = read(facts(patches=[{"name": "inlet", "nFaces": 0},
                                {"name": "walls", "nFaces": 400}]), "mesh")
    assert any("1 boundary patch" in m for m in check.missing)


def test_a_mesh_nobody_can_rebuild_is_not_finished():
    check = read(facts(build=[]), "mesh")
    assert any("nothing here can be rebuilt or edited" in m for m in check.missing)


def test_a_mesh_nobody_drew_is_not_finished():
    check = read(facts(render=""), "mesh")
    assert any("no picture" in m for m in check.missing)


def test_the_refusal_reads_as_work_not_as_a_verdict():
    check = read(facts(polymesh=False), "mesh")
    text = check.as_refusal()
    assert text.startswith("The check did not pass")
    assert "Fix it and say done again" in text


# -- the size the request asked for --------------------------------------------


@pytest.mark.parametrize("request_text,expected", [
    ("a passage 8 mm wide, two legs 60 mm long, 20 mm apart", 0.06),
    ("a channel 300 mm long and 60 mm high", 0.3),
    ("a box 1 m long and 0.6 m high", 1.0),
    ("a pipe of 20 mm bore with a 4 cm bend radius", 0.04),
    ("something with no sizes in it at all", 0.0),
])
def test_the_largest_length_is_read_off_the_words(request_text, expected):
    assert largest_length(request_text) == pytest.approx(expected)


def test_a_mesh_left_in_millimetres_is_caught():
    """The T04 U-duct: gmsh built it in millimetres and nothing scaled it, so the
    mesh was 74 m across. checkMesh passed, the picture was right, and the case
    would have solved a duct a thousand times too big."""
    off = scale_mismatch("a passage 8 mm wide, two legs 60 mm long",
                         [0, -0.004, 0, 0.074, 0.024, 0.001])
    assert off == ""
    off = scale_mismatch("a passage 8 mm wide, two legs 60 mm long", [0, -4, 0, 74, 24, 1])
    assert "74 m across" in off and "1233x out" in off
    assert "transformPoints -scale" in off


def test_a_flow_box_round_a_small_body_is_not_a_scale_error():
    """A sphere of 20 mm in a box twenty times its size is the normal case, and a
    check that calls it wrong is a check nobody can leave switched on."""
    assert scale_mismatch("a sphere of 20 mm diameter in a flow box",
                          [-0.1, -0.1, -0.1, 0.3, 0.1, 0.1]) == ""


def test_no_sizes_and_no_bounds_mean_no_opinion():
    assert scale_mismatch("mesh me something nice", [0, 0, 0, 74, 24, 1]) == ""
    assert scale_mismatch("a duct 60 mm long", []) == ""
    assert scale_mismatch("a duct 60 mm long", [0, 0, 0, 0, 0, 0]) == ""


def test_the_scale_check_reaches_the_verdict():
    check = read(facts(bounds=[0, 0, 0, 74, 24, 1]), "mesh", "/work/s/mesh",
                 "a passage 8 mm wide, two legs 60 mm long")
    assert not check.ok
    assert any("1233x out" in m for m in check.missing)


# -- reaching the workspace ----------------------------------------------------


class Answering:
    workspace_root = "/work"

    def __init__(self, output, raises=None, raises_once=False):
        self.output = output
        self.raises = raises
        self.raises_once = raises_once
        self.calls: list[tuple] = []

    def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
        self.calls.append((cmd, cwd, timeout_s))
        if self.raises:
            raise self.raises
        if self.raises_once and len(self.calls) == 1:
            raise RuntimeError("sandbox recycled")
        return ExecResult(0, self.output, False, None)


def test_the_check_runs_one_command_in_the_case_and_reads_its_json():
    backend = Answering('shell noise\n{"polymesh": true, "cells": 5, "checkmesh_ok": true, '
                        '"patches": [{"name": "a", "nFaces": 1}, {"name": "b", "nFaces": 1}], '
                        '"build": ["Allmesh"], "render": "r.png"}\n')
    check = verify(backend, "/work/s/mesh", "mesh")
    cmd, cwd, _ = backend.calls[0]
    assert cwd == "/work/s/mesh"
    assert LOOK in cmd and "--json" in cmd
    assert check.ok and check.cells == 5


def test_a_workspace_that_cannot_answer_is_a_failed_check_not_an_exception(monkeypatch):
    monkeypatch.setattr("openreynolds.mesher.check.RETRY_PAUSE_S", 0)
    check = verify(Answering("", raises=RuntimeError("sandbox gone")), "/work/s/mesh", "mesh")
    assert not check.ok
    assert check.unreachable
    assert "sandbox gone" in check.missing[0]
    assert "may well be there" in check.missing[0]


def test_the_check_is_asked_twice_before_it_gives_up(monkeypatch):
    """A container that recycles mid-mesh is a Modal preemption: it comes back in
    seconds and the Volume under it never went anywhere. Three real runs had a
    finished mesh reported as missing because the one attempt landed in that window."""
    monkeypatch.setattr("openreynolds.mesher.check.RETRY_PAUSE_S", 0)
    backend = Answering('{"polymesh": true, "cells": 5, "checkmesh_ok": true, '
                        '"patches": [{"name": "a", "nFaces": 1}, {"name": "b", "nFaces": 1}], '
                        '"build": ["Allmesh"], "render": "r.png"}', raises_once=True)
    check = verify(backend, "/work/s/mesh", "mesh")
    assert check.ok and not check.unreachable
    assert len(backend.calls) == 2


def test_a_mesh_that_could_not_be_checked_is_not_reported_as_missing(monkeypatch):
    """The tool result led with "NOT a usable mesh yet" when nothing at all was known,
    and the calling agent believed it and rebuilt a mesh that was already there."""
    from openreynolds.mesher.agent import MeshResult
    from openreynolds.tools import mesh_text

    monkeypatch.setattr("openreynolds.mesher.check.RETRY_PAUSE_S", 0)
    check = verify(Answering("", raises=RuntimeError("sandbox recycled")), "/work/s/mesh", "mesh")
    text = mesh_text(MeshResult(case_rel="mesh", check=check, seconds=60.0))
    assert "could NOT BE CHECKED" in text
    assert "not a statement about the mesh" in text.lower()
    assert "NOT a usable mesh" not in text


def test_output_with_no_json_says_what_to_run_by_hand():
    check = verify(Answering("Traceback: pyvista exploded"), "/work/s/mesh", "mesh")
    assert not check.ok
    assert "no readable answer" in check.missing[0]
    assert "pyvista exploded" in check.missing[0]


def test_the_look_command_is_the_one_a_person_would_type():
    assert look_command("/work/s/mesh").startswith(f"python3 {LOOK} /work/s/mesh")


# -- what the caller is shown --------------------------------------------------


def test_the_report_lines_carry_the_numbers_a_reader_needs():
    check = read(facts(patches=[{"name": "inlet", "type": "patch", "nFaces": 20,
                                 "area": 0.0001, "center": [0, 0.025, 0.0005],
                                 "normal": [-1, 0, 0]}],
                       metrics={"max_non_orthogonality": 12.4}), "mesh")
    text = "\n".join(check.lines())
    assert "3,750 cells" in text and "2D (one cell thick)" in text
    assert "bounds 0.1 x 0.05 x 0.001 m" in text
    assert "inlet" in text and "area 0.0001 m2" in text and "normal (-1.00 +0.00 +0.00)" in text
    assert "max_non_orthogonality 12.4" in text
    assert "rebuilds with: Allmesh" in text


def test_an_empty_check_reports_nothing_rather_than_zeroes():
    assert Check().lines() == []
