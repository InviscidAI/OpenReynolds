"""The starting desk: a kernel, a cell log, a brief, and `checkMesh`. Nothing else.

The desk this replaces spent three times the steps on eight of eight prompts and produced
no more meshes, with 85% of its discovery spend going to learn its own surfaces. So what
is pinned here is mostly what is *absent*: a brief that names no instrument, a nudge that
names no recipe folder, a finish check that asks OpenFOAM and nobody else, and a desk
whose toolbox path is empty so that nothing can render one.
"""

from __future__ import annotations

import json

from pathlib import Path

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


def looks_like(checkmesh=MESH_OK, **changes):
    """A `mesh_look.py --json` payload, the way the check reads one off the workspace.

    `CoreDesk` has run the full `check.verify` since 2026-09-18 -- `mesh_look.py`,
    `cad_audit.py`, `domain_probe.py`, and the replay -- so a fake backend that answers
    only `checkMesh` no longer describes a finished case. It answers all of them here,
    because what these tests are about (the marks, the region walk) needs the real check
    to run rather than a stubbed verdict.
    """
    payload = {"polymesh": True, "cells": 729, "faces": 2000, "points": 1000,
               "bounds": [0, 0, 0, 1, 1, 1], "build": ["Allmesh"],
               "patches": [{"name": "inlet", "type": "patch", "nFaces": 20},
                           {"name": "walls", "type": "wall", "nFaces": 400}],
               "checkmesh": checkmesh.strip().splitlines()[-2],
               "checkmesh_ok": "Mesh OK." in checkmesh,
               "render": "renders/mesh_look.png"}
    payload.update(changes)
    return f"@@CELLZONES@@[]\n@@JSON@@{json.dumps(payload)}"


def checked(backend, checkmesh=MESH_OK, regions="SINGLE:\n"):
    from test_cad_agent import answers

    return answers(backend, {
        "constant/*/polyMesh": ExecResult(0, regions, False, None),
        "mesh_look.py": ExecResult(0, looks_like(checkmesh), False, None),
        "cad_audit.py": ExecResult(0, '@@JSON@@{"findings": []}', False, None),
        "domain_probe.py": ExecResult(0, '@@JSON@@{"findings": []}', False, None),
        "checkMesh": ExecResult(0, checkmesh, False, None),
    })


def core_desk(backend, store, texts):
    made = core.CoreDesk(Config(llm_api_key="k", model="claude-opus-5"), backend, store,
                         "/work/study")
    made.provider = ScriptedProvider(texts)
    return made


# -- what the core does not say --------------------------------------------------


def test_the_core_brief_names_exactly_what_it_hands_over_and_nothing_else():
    """Not a style point: the brief is the one thing the desk certainly reads, so a
    toolbox named there is a toolbox found there.

    The arm is no longer bare -- it is handed `b123d_api.md`, on the measured evidence
    that 17 of 27 API-surface failures in `core+declare_gate-20260913-124524-aa21` were
    build123d, across 12 of 26 cases. So the invariant tightens rather than relaxes: the
    brief may name the file it hands over, and may still name nothing else. The word
    `toolbox` stays out entirely, because the directory name is a house surface on its
    own and this arm has no toolbox to point at -- what it has is one file in a
    `.reference` directory of its own.
    """
    text = {"brief": core.CORE_SYSTEM, "nudge": core.CORE_NUDGE}
    assert not isolation.scan(text, given=core.REFERENCE_FILES).contaminated

    leaked = isolation.scan(text).hits
    assert {hit.name for hit in leaked} == set(core.REFERENCE_FILES), (
        "the brief names a house surface that was not handed over: "
        f"{sorted({hit.name for hit in leaked} - set(core.REFERENCE_FILES))}")
    assert "toolbox" not in (core.CORE_SYSTEM + core.CORE_NUDGE).lower()
    assert core.REFERENCE_DIR != isolation.TOOLBOX_NAME


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


def test_the_core_brief_teaches_exactly_one_way_to_finish():
    """It used to be `print("CAD_DONE")`; since 2026-09-13 it is `declare_complete`.

    Both still *work* -- the sentinel path is untouched and is how the shipped desk
    finishes -- but the core brief teaches one of them, because a brief that teaches two
    is a sweep that cannot say which one the desk used. This assertion used to live in
    the test above, bundled beside the cell-a-turn line it has nothing to do with."""
    assert "declare_complete" in core.CORE_SYSTEM
    assert CAD_DONE not in core.CORE_SYSTEM
    assert core.CORE_SYSTEM.count("# Finishing") == 1


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
    checked(backend)
    kernelled(backend)
    made = core_desk(backend, store, [block("x = 1"), block(f'print("{CAD_DONE}")')])
    result = made.run("a duct")
    assert result.ok and result.check.checkmesh == "Mesh OK."
    assert result.check.cells == 729


def test_a_desk_that_says_done_over_a_refused_mesh_is_handed_the_refusal(backend, store):
    checked(backend, MESH_BAD)
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
    thread = {"thread": "\n".join(said)}
    assert not isolation.scan(thread, given=core.REFERENCE_FILES).contaminated
    # And the half that still has to hold: nothing beyond what was handed over.
    assert {hit.name for hit in isolation.scan(thread).hits} <= set(core.REFERENCE_FILES)


# -- what this addition closes, and the failures it is answering -------------------
#
# Each test names the runs behind it. An addition with no run behind it is the top-down
# catalogue this phase exists to undo.


def brief() -> str:
    return core.brief(900)


def test_the_desk_is_told_where_the_build123d_reference_is_and_how_to_use_it():
    """`cell_increase_is_desk_side_retry_churn_not_the_gate`, 15 of 26 cases.

    In `core+declare_gate-20260913-124524-aa21`, 27 of the 36 cells that raised a named
    exception were the desk calling something that does not exist, and 17 of those were
    build123d. The desk had no reference because `CoreDesk` is the arm given no tools;
    `b123d_api.md` was opened by 0 of 26 runs and `help()`/`inspect.signature()` by 1.
    """
    text = brief()
    assert f"{core.REFERENCE_DIR}/b123d_api.md" in text
    assert "inspect.signature" in text and "help(" in text, (
        "the file's whole design is discovery-then-introspect; the brief has to say the "
        "second half or the desk stops at a reading list")
    assert "grep" in text
    assert "reading list" in text


def test_the_desk_is_told_that_a_name_without_parens_is_a_property():
    """Six cases raised `TypeError: 'bool' object is not callable` -- T5, T6, T10, T11,
    T23, T24 -- which is what calling a property like a method looks like. T24's was
    `fluid.is_valid()`, and `Shape.is_valid` is listed in the reference without parens.
    The notation only helps if the desk is told it is notation."""
    assert "property and takes none" in " ".join(brief().split())


def test_the_desk_is_told_what_normals_actually_measures():
    """`check_misunderstood_while_the_geometry_is_understood`, T21.

    T21 pre-waived `normals` because "STLs are the boundary faces of the fluid solid, so
    their normals point out of the fluid" -- correct about the geometry, and not what the
    check measures, so it read back `xpass` on a check that was already clean. The
    `union_closure` precedent is the argument: one explanatory line took its `xpass`
    count from 2 to 0.
    """
    text = " ".join(brief().split())
    assert "winding consistency and not orientation" in text
    assert "does not waive it" in text
    normals = text.index("`normals` counts edges walked twice")
    closure = text.index("`union_closure` welds every STL")
    assert abs(normals - closure) < 1200, "the two explanations belong together"


def test_the_desk_is_told_to_record_the_numbers_it_chose():
    """`assumption_recorded_nowhere_when_the_request_carries_no_number`, T25.

    T25 wrote a 2.10 m fan and a 3.00 m length into its first cell, before any question,
    for a request that states no dimension at all -- and across 2,386 characters of reply
    text used not one word marking those numbers as its own. The repo's own convention is
    `openreynolds/cad/check.py`: a legitimate-but-ambiguous result is "a warning with its
    assumption stated, not a refusal".
    """
    text = " ".join(brief().split())
    assert "is an assumption" in text
    assert "which numbers came from the request and which came from you" in text
    # The refusal route still exists; this is the case short of it, not a replacement.
    assert 'outcome: "refuse"' in text


def test_the_reference_the_brief_names_is_reachable_from_the_desks_own_directory(tmp_path):
    """The first attempt at this addition failed exactly here, and silently.

    `prepare` copied the reference to the workspace root while the desk's working
    directory is the *case* directory one level down, so `.reference/b123d_api.md` --
    the path the brief gives -- resolved to nothing. In
    `core+reference-20260914-025525-6a2b` one run of ten took the brief at its word, ran
    the grep it suggests, and got an empty string back; its own `os.listdir('.')` printed
    `[]`. The file existed, the brief named it, and no desk could reach it.

    A brief naming a path the workspace does not have is how a desk spends its opening
    steps looking for what it was told it had -- seven of twenty-seven, measured, which
    is why `test_the_nudge_stops_pointing_at_recipes_that_are_not_there` exists. So this
    test resolves the path the brief gives *against the directory the desk is given*,
    rather than checking the source file exists somewhere.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cad_buildup
    from openreynolds.buildup import isolation

    for name in core.REFERENCE_FILES:
        assert name in brief(), f"{name} is handed over and never mentioned"
        assert (isolation.TOOLBOX_DIR / name).is_file(), (
            f"{name} is named in the brief and is not on disk to copy")

    workspace, _geometry, _prompt = cad_buildup.prepare(
        "T20", tmp_path, tmp_path / "run", "reachable")
    case_dir = workspace / "t20"
    for name in core.REFERENCE_FILES:
        as_the_brief_says = case_dir / core.REFERENCE_DIR / name
        assert as_the_brief_says.is_file(), (
            f"the brief tells the desk to read {core.REFERENCE_DIR}/{name} from its "
            f"working directory and it is not there: {as_the_brief_says}")
        assert as_the_brief_says.stat().st_size > 1000


# -- the floor, and what is deliberately not standing on it ---------------------


def test_the_binding_finish_is_checkmesh_and_nothing_else():
    """§1's floor, restated as a test after a port briefly widened it.

    Between ef637d7 and this, `CoreDesk` inherited the whole of `check.verify`: the
    render, the patch naming, the request-scale reading, the rebuild script and the
    replay. Two of those five were then removed for failing the desk over things it had
    no move against, and `render` for failing this desk over an artifact its brief never
    asks for.

    None of the five arrived the way an addition is supposed to -- carrying a measured
    failure and a test that shows it in the addition's absence. They arrived as a set, in
    a port, with no sweep between them and the corpus. So the floor is back, and each of
    them is a candidate again rather than a fact.
    """
    from openreynolds.cad import check as cadcheck

    verdict = core.verify(_backend_saying(MESH_OK), "/work/case")
    assert verdict.ok
    assert {f.check for f in verdict.findings} == {"checkMesh"}

    # The wider check still exists and still raises them -- for `CadDesk`, which
    # `scripts/cad_accept.py` drives. What changed is which desk stands on them.
    wider = cadcheck.read({"polymesh": True, "cells": 10, "faces": 10, "points": 10,
                           "bounds": [0, 0, 0, 1, 1, 1], "checkmesh": "Mesh OK.",
                           "checkmesh_ok": True, "render": "", "build": [],
                           "patches": [{"name": "inlet", "nFaces": 4}]}, "c")
    assert "render" in {f.check for f in wider.findings}
    assert not wider.ok, "the wider check still binds on it; this desk just is not it"


def test_the_refusal_this_desk_reads_names_no_house_path():
    """The half of the port that would have voided the next sweep rather than moved it.

    `check.verify`'s binding findings carry repair text with the toolbox in it -- "run
    `python3 /work/.toolbox/mesh_look.py ...`" -- and `isolation.scan_run` greps the whole
    thread for exactly that. A run drawing one of those findings grades contaminated and
    is discarded from the baseline, which is a failure that reads as a missing number
    rather than as a wrong one.

    `gate.scrub` covers the advisory side. Nothing covered the binding side, because until
    the port nothing binding was ever shown to a desk that is watched.
    """
    from openreynolds.buildup import isolation

    bad = MESH_OK.replace("Mesh OK.", "Failed 2 mesh checks.")
    verdict = core.verify(_backend_saying(bad), "/work/case")
    assert not verdict.ok

    scan = isolation.scan({"refusal": verdict.as_refusal()},
                          given=core.REFERENCE_FILES)
    assert not scan.contaminated, sorted({hit.name for hit in scan.hits})


def _backend_saying(checkmesh: str):
    from test_cad_agent import answers

    class Bare:
        workspace_root = "/work"

        def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
            return ExecResult(0, "", False, None)

    return answers(Bare(), {
        "constant/*/polyMesh": ExecResult(0, "SINGLE:\n", False, None),
        "checkMesh": ExecResult(0, checkmesh, False, None),
    })


def test_a_session_puts_the_reference_where_the_brief_says(tmp_path):
    """The same reachability, through the route a person actually has.

    The test above proves `scripts/cad_buildup.py` puts the files there. That is the
    corpus's route, and for a while it was the only one: `CoreDesk` was constructed by
    the runner, which had already staged them with `shutil` because its workspace is a
    local directory. A session's case is on the volume, so the same `shutil` copy would
    land on the machine holding the conversation and nowhere the desk can read -- which
    is the first attempt's failure again, one layer down.

    `CoreDesk._prepare` is that route, and this drives it against a real backend rather
    than a stub, because what is being checked is that bytes cross the transport.
    """
    from openreynolds.backend.local import LocalBackend
    from openreynolds.cad.core import CoreDesk
    from openreynolds.config import Config

    backend = LocalBackend(root=tmp_path)
    desk = CoreDesk(Config(llm_api_key="k", model="claude-opus-5"), backend, None,
                    f"{backend.workspace_root}/study")
    case_dir = f"{backend.workspace_root}/study/cad"
    desk._prepare(case_dir)

    for name in core.REFERENCE_FILES:
        as_the_brief_says = Path(case_dir) / core.REFERENCE_DIR / name
        assert as_the_brief_says.is_file(), (
            f"the brief tells the desk to read {core.REFERENCE_DIR}/{name} from its "
            f"working directory and a session does not put it there")
        assert as_the_brief_says.stat().st_size > 1000
    # And `export_patches` imports from where the brief says it will.
    assert f'sys.path.insert(0, "{core.REFERENCE_DIR}")' in brief()
    assert "from cad_export import export_patches" in brief()


def test_a_missing_reference_stops_the_run_before_any_model_call(tmp_path, monkeypatch):
    """`b123d_api.md` is generated, so a checkout can legitimately be without it.

    Finding that out on cell nine costs a model call for every cell before it, and the
    desk cannot fix it: the file is ours to put there. So `_prepare` raises and `run`
    turns it into an error with the generator named in it.
    """
    from openreynolds.backend.local import LocalBackend
    from openreynolds.cad import core as cadcore
    from openreynolds.cad.core import CoreDesk
    from openreynolds.config import Config

    monkeypatch.setattr(cadcore, "TOOLBOX_DIR", tmp_path / "empty")
    backend = LocalBackend(root=tmp_path)
    desk = CoreDesk(Config(llm_api_key="k", model="claude-opus-5"), backend, None,
                    f"{backend.workspace_root}/study")
    desk.provider = None  # a model call here would raise something else entirely

    result = desk.run("a duct")
    assert not result.ok
    assert "b123d_api.md" in result.error
    assert "b123d_api.py" in result.error, "say how to make it, not just that it is gone"
