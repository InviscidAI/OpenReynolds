"""The starting desk: a kernel, a cell log, a brief, and `checkMesh`. Nothing else.

The desk this replaces spent three times the steps on eight of eight prompts and produced
no more meshes, with 85% of its discovery spend going to learn its own surfaces. So what
is pinned here is mostly what is *absent*: a brief that names no instrument, a nudge that
names no recipe folder, a finish check that asks OpenFOAM and nobody else, and a desk
whose toolbox path is empty so that nothing can render one.
"""

from __future__ import annotations

import pytest

from openreynolds.backend.base import ExecResult
from openreynolds.buildup import core, isolation
from openreynolds.cad.brief import CAD_DONE
from openreynolds.config import Config
from test_cad_agent import ScriptedProvider, block, kernelled, said

MESH_OK = """
Mesh stats
    points:           1000
    faces:            2700
    cells:            729
Overall domain bounding box (0 0 0) (0.1 0.1 0.1)
Checking geometry...
Mesh OK.
End
"""

MESH_BAD = """
Mesh stats
    points:           40
    faces:            60
    cells:            9
Overall domain bounding box (0 0 0) (1 1 1)
 ***Number of severely non-orthogonal faces: 12.
Failed 2 mesh checks.
End
"""


def core_desk(backend, store, texts):
    made = core.CoreDesk(Config(llm_api_key="k", model="claude-opus-5"), backend, store,
                         "/work/study")
    made.provider = ScriptedProvider(texts)
    return made


# -- what the core does not say --------------------------------------------------


def test_the_core_brief_names_no_house_surface_at_all():
    """Not a style point: the brief is the one thing the desk certainly reads, so a
    toolbox named there is a toolbox found there, and the arm is not a bare arm."""
    found = isolation.scan({"brief": core.CORE_SYSTEM, "nudge": core.CORE_NUDGE})
    assert not found.contaminated, found.lines()
    assert "toolbox" not in (core.CORE_SYSTEM + core.CORE_NUDGE).lower()


def test_the_nudge_stops_pointing_at_recipes_that_are_not_there():
    """The wider desk's nudge offers `templates/`. A brief that names a path the
    workspace does not have is how a desk spends its opening steps looking for what it
    was told it had -- seven of twenty-seven, measured."""
    from openreynolds.cad.agent import NUDGE

    assert "templates" in NUDGE and "templates" not in core.CORE_NUDGE
    assert "coarsest version" in core.CORE_NUDGE


def test_the_brief_carries_the_line_that_was_the_largest_measured_effect():
    """One short runnable cell a turn. On the hardest prompt it turned three consecutive
    total failures into a completed mesh, and it is not a tool."""
    assert "One short runnable cell a turn." in core.CORE_SYSTEM
    assert CAD_DONE in core.CORE_SYSTEM


def test_the_core_desk_renders_the_core_brief_and_holds_no_toolbox(backend, store):
    made = core_desk(backend, store, [block("x = 1")])
    assert made.toolbox == ""
    assert made._system().startswith("You are the CAD desk.")
    assert "The instruments" not in made._system()
    assert made._nudge() == core.CORE_NUDGE


# -- checkMesh, and nothing beside it --------------------------------------------


def test_nothing_meshed_is_a_failure_that_says_which_directories_were_looked_at(backend):
    verdict = core.verify(backend, "/work/case")
    assert not verdict.ok and verdict.regions == []
    assert "has been meshed" in verdict.missing[0]


def test_a_clean_single_region_mesh_passes_with_the_numbers_checkmesh_printed(backend):
    from test_cad_agent import answers

    answers(backend, {"constant/*/polyMesh": ExecResult(0, "SINGLE:\n", False, None),
                      "checkMesh": ExecResult(0, MESH_OK, False, None)})
    verdict = core.verify(backend, "/work/case")
    assert verdict.ok and verdict.regions == [""]
    assert (verdict.cells, verdict.faces, verdict.points) == (729, 2700, 1000)
    assert verdict.bounds == [0.0, 0.0, 0.0, 0.1, 0.1, 0.1]
    assert verdict.checkmesh == "Mesh OK."


def test_a_conjugate_case_is_checked_per_region_and_one_bad_region_fails_it(backend):
    """The desk this replaces called a conjugate case 'nothing has been meshed yet',
    because it looked for a singular mesh and a two-region case has none."""
    asked: list[str] = []

    def routed(cmd, cwd=None, timeout_s=120, *, background=False):
        asked.append(cmd)
        if "constant/*/polyMesh" in cmd:
            return ExecResult(0, "REGION:air\nREGION:solid\n", False, None)
        return ExecResult(0, MESH_OK if "air" in cmd else MESH_BAD, False, None)

    backend.exec = routed
    verdict = core.verify(backend, "/work/case")
    assert verdict.regions == ["air", "solid"]
    assert not verdict.ok
    assert verdict.cells == 738  # both regions, summed
    assert "air: Mesh OK." in verdict.checkmesh and "solid: Failed 2" in verdict.checkmesh
    assert [cmd for cmd in asked if "-region air" in cmd]


def test_checkmesh_is_the_only_authority_on_quality(backend):
    """`Mesh OK.` with a severely non-orthogonal face in the body is still a pass: two
    authorities on one number is a desk being told contradictory things about a mesh
    OpenFOAM has already judged."""
    from test_cad_agent import answers

    grumbling = MESH_OK.replace("Checking geometry...",
                                " ***Number of severely non-orthogonal faces: 3.")
    answers(backend, {"constant/*/polyMesh": ExecResult(0, "SINGLE:\n", False, None),
                      "checkMesh": ExecResult(0, grumbling, False, None)})
    assert core.verify(backend, "/work/case").ok


def test_a_workspace_that_will_not_answer_is_unreachable_and_not_a_verdict(backend):
    def dead(cmd, cwd=None, timeout_s=120, *, background=False):
        raise RuntimeError("sandbox_unavailable")

    backend.exec = dead
    verdict = core.verify(backend, "/work/case")
    assert verdict.unreachable and not verdict.ok
    assert "may well be there" in verdict.missing[0]


# -- the loop, driven by the core ------------------------------------------------


def test_the_core_desk_finishes_on_checkmesh_and_on_nothing_else(backend, store):
    from test_cad_agent import answers

    answers(backend, {"constant/*/polyMesh": ExecResult(0, "SINGLE:\n", False, None),
                      "checkMesh": ExecResult(0, MESH_OK, False, None)})
    kernelled(backend)
    made = core_desk(backend, store, [block("x = 1"), block(f'print("{CAD_DONE}")')])
    result = made.run("a duct")
    assert result.ok and result.check.checkmesh == "Mesh OK."
    assert result.check.cells == 729


def test_a_desk_that_says_done_over_a_refused_mesh_is_handed_the_refusal(backend, store):
    from test_cad_agent import answers

    answers(backend, {"constant/*/polyMesh": ExecResult(0, "SINGLE:\n", False, None),
                      "checkMesh": ExecResult(0, MESH_BAD, False, None)})
    kernelled(backend)
    made = core_desk(backend, store, [block(f'print("{CAD_DONE}")')])
    result = made.run("a duct")
    assert not result.ok
    sent = said(made.provider, -1)
    assert "Failed 2 mesh checks." in sent and "not finished" in sent


def test_the_whole_conversation_of_a_core_run_stays_clean(backend, store):
    """The end-to-end version of the first test: what the desk was sent, over a whole
    run, greps clean. This is the claim the bare arm made last round and could not
    support."""
    from test_cad_agent import answers

    answers(backend, {"constant/*/polyMesh": ExecResult(0, "SINGLE:\n", False, None),
                      "checkMesh": ExecResult(0, MESH_OK, False, None)})
    kernelled(backend)
    made = core_desk(backend, store, [block("x = 1"), block(f'print("{CAD_DONE}")')])
    made.run("a duct")
    said = []
    for call in made.provider.calls:
        said.append(call["system"])
        for message in call["messages"]:
            for chunk in message["content"] if isinstance(message["content"], list) else []:
                said.append(str(chunk.get("text", "") if isinstance(chunk, dict) else chunk))
    assert not isolation.scan({"thread": "\n".join(said)}).contaminated
