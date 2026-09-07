"""What "done" means, as a function of the facts, tested without a workspace.

Every rule here is one that shipped broken once: a case written and read as meshed, a
mesh with failing checks handed to a solver, `patch0`/`patch1` out of gmshToFoam with
nobody able to say which end was the inlet, and a mesh nobody could rebuild.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openreynolds.backend.base import ExecResult
from openreynolds.mesher.check import LOOK, Check, look_command, read, verify

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


# -- reaching the workspace ----------------------------------------------------


class Answering:
    workspace_root = "/work"

    def __init__(self, output, raises=None):
        self.output = output
        self.raises = raises
        self.calls: list[tuple] = []

    def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
        self.calls.append((cmd, cwd, timeout_s))
        if self.raises:
            raise self.raises
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


def test_a_workspace_that_cannot_answer_is_a_failed_check_not_an_exception():
    check = verify(Answering("", raises=RuntimeError("sandbox gone")), "/work/s/mesh", "mesh")
    assert not check.ok
    assert "sandbox gone" in check.missing[0]


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
