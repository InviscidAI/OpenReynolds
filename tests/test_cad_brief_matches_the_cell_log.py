"""What the brief tells the desk to write, the cell log has to be willing to keep.

These two were written by different chunks against the same plan, and the plan says
both of these things: that OpenFOAM utilities stay reachable inside a cell so there is
one channel rather than two, and that the accepted cells concatenated *are* a script
that is re-run as `python3 build.py`. `!blockMesh` satisfies the first and breaks the
second -- it is IPython, not Python, and a file containing one is a syntax error.

The brief shipped advertising `!blockMesh`, `!checkMesh`, `!nohup ... &` and `!grep`
while the cell log refused every one of them at accept time. Nothing failed: the cell
*ran*, so the mesh appeared, and the step was simply missing from `build.py` -- which is
the quiet kind of wrong this desk exists to end. A desk following its own brief would
have collected a refusal every time it did as it was told.

So the rule is pinned from both ends here rather than inside either chunk's own file.
"""

from __future__ import annotations

import pathlib
import re

from openreynolds.cad import brief
from openreynolds.cad.cells import CellLog

# A bare `!command` in backticks, which is what the brief used to offer.
BANG = re.compile(r"`!\w")


def _brief_text() -> str:
    return brief.system_prompt(240)


def test_the_brief_offers_no_shell_escape_the_script_cannot_carry():
    """Every occurrence must be the sentence that forbids it, not one that offers it."""
    text = _brief_text()
    found = list(BANG.finditer(text))
    assert found, "no `!` at all; the prohibition sentence should still name one"
    for match in found:
        # Precise, not a neighbourhood: the token must be the object of "not as",
        # which is the one sentence allowed to say it. A window wide enough to catch
        # the nearby prohibition would pass a brief that offered the idiom again a
        # line later -- measured, after a first version of this test did exactly that.
        before = text[max(0, match.start() - 8): match.start()]
        assert before.endswith("not as "), (
            f"the brief offers {match.group()!r} as an idiom, and the cell log refuses "
            "it: the cell would run and the step would be missing from build.py"
        )


def test_the_brief_says_why_the_bang_is_refused():
    """Naming the mechanism, because a bare prohibition invites working around it."""
    text = _brief_text()
    assert "build.py" in text
    assert "python3 build.py" in text


def test_the_shell_out_the_brief_recommends_is_one_the_log_accepts():
    """The positive half: what it offers instead has to actually survive accept()."""
    log = CellLog()
    cell = log.propose(
        'import subprocess\nsubprocess.run(["blockMesh"], check=True)\n'
    )
    assert log.accept(cell) == "", "the brief's recommended shell-out was refused"
    assert "subprocess.run" in log.script()


def test_the_background_mesher_the_brief_recommends_is_one_the_log_accepts():
    log = CellLog()
    cell = log.propose(
        "import subprocess\n"
        'handle = subprocess.Popen(["snappyHexMesh", "-overwrite"],\n'
        '                          stdout=open("log.snappy", "w"),\n'
        "                          stderr=subprocess.STDOUT)\n"
    )
    assert log.accept(cell) == "", "the brief's background mesher was refused"


def test_the_bang_really_is_refused_so_this_file_is_not_guarding_nothing():
    """If the log ever starts accepting `!`, these assertions stop meaning anything."""
    log = CellLog()
    why = log.accept(log.propose("!blockMesh"))
    assert why, "the cell log accepts `!blockMesh`; this whole file is now vacuous"
    assert "build.py" in why or "python" in why.lower()


# -- the toolbox path the brief names has to be the one the backend has -------------
#
# Same shape of defect as the one above and pinned in the same place, because it also
# spans two chunks that were each right on their own. `cad/brief.py` told the desk its
# instruments "are in `/work/.toolbox/`" as literal text, and `cad/check.py` ran
# `python3 /work/.toolbox/mesh_look.py` through a shell. Both are true on the hosted
# image. Neither is true on `LocalBackend`, which is rooted wherever it was told to be
# and is what the whole stack is developed against -- and nothing translates `/work`
# for a shell or for a `subprocess.run` inside a cell.
#
# What that cost, measured rather than supposed: in a local acceptance run of T1, seven
# of twenty-seven steps went on `grep`, `ls`, `find /` and `ls ../.toolbox` before the
# desk located the tools it had been handed, and the finish check reported "the check
# wrote no readable answer" about a mesh it had never opened. Neither is an error; both
# look like ordinary bad luck.
#
# So both halves are pinned: the hosted rendering must not move by a byte, and a
# backend that says it is rooted elsewhere must see its own path in both places.

from openreynolds.backend.base import WORKSPACE_ROOT  # noqa: E402
from openreynolds.cad import check as cadcheck  # noqa: E402
from openreynolds.cad.brief import system_prompt  # noqa: E402


class _Rooted:
    """The only thing either of these reads off a backend."""

    def __init__(self, root: str) -> None:
        self.workspace_root = root


def test_the_hosted_brief_still_names_the_hosted_toolbox():
    """The default has to reproduce what the image has always been told."""
    text = system_prompt(280)
    assert f"{WORKSPACE_ROOT}/.toolbox/" in text
    assert "{toolbox}" not in text, "the placeholder was left unfilled"


def test_a_brief_for_another_root_names_that_root():
    text = system_prompt(280, toolbox="/srv/box/.toolbox")
    assert "/srv/box/.toolbox/" in text
    assert f"{WORKSPACE_ROOT}/.toolbox" not in text, (
        "the brief still names the hosted toolbox on a backend rooted elsewhere; a desk "
        "reading this goes looking for its own tools and pays steps for it")


def test_the_check_runs_the_toolbox_where_the_backend_actually_is():
    """`mesh_look.py` is handed to a shell, so the string has to be the real path."""
    assert cadcheck.toolbox_for(_Rooted("/srv/box")) == "/srv/box/.toolbox"
    cmd = cadcheck._region_command("", cadcheck.toolbox_for(_Rooted("/srv/box")))
    assert "/srv/box/.toolbox/mesh_look.py" in cmd
    assert f"{WORKSPACE_ROOT}/.toolbox" not in cmd, (
        "the check would run a path that does not exist on this backend and then report "
        "that the mesh could not be read")


def test_the_hosted_check_command_has_not_moved():
    assert f"python3 {WORKSPACE_ROOT}/.toolbox/mesh_look.py" in cadcheck._region_command("")
    assert cadcheck.look_command("/work/s/mesh").startswith(
        f"python3 {WORKSPACE_ROOT}/.toolbox/mesh_look.py /work/s/mesh")


def test_the_desk_takes_the_path_from_its_backend_rather_than_from_the_constant(
        monkeypatch):
    """The wiring, not just the two ends: `CadDesk` must read it off its backend."""
    from openreynolds.cad import agent as cadagent

    monkeypatch.setattr(cadagent, "make_provider", lambda cfg: object())

    class _Cfg:
        model = "claude-sonnet-5"
        mesher_model = ""
        mesher_effort = "high"
        mesher_max_steps = 0
        mesher_max_seconds = 0.0

    desk = cadagent.CadDesk(_Cfg(), _Rooted("/srv/box"), None, "")
    assert desk.toolbox == "/srv/box/.toolbox"
    assert "/srv/box/.toolbox/" in system_prompt(280, toolbox=desk.toolbox)
    assert "/srv/box/.toolbox/templates/" in cadagent.NUDGE.format(toolbox=desk.toolbox)


def test_the_binding_gate_is_a_bare_checkmesh_and_stays_one():
    """`-allGeometry` was binding for a few hours on 2026-09-16 and was reverted on solver
    evidence. It is not a stricter setting of the same checks -- it runs checks the bare
    form does not run at all -- and it failed 14 of the 22 meshes the gate passed. Nine of
    the newly-failing cases went through `simpleFoam`: four converged to 1e-5 on p and U,
    four were still descending at the 300-iteration cap, none diverged. snappyHexMesh made
    5 of those 14 meshes and gmsh 6, so it is measuring the mesher, not the desk.

    Pinned as a test because the argument for adding it is genuinely appealing -- the
    numbers look like defects -- and because the same reasoning will come back."""
    from openreynolds.buildup import core as buildup_core

    assert buildup_core.check_command("") == "checkMesh 2>&1 | tail -n 200"
    assert buildup_core.check_command("fluid").startswith("checkMesh -region fluid")
    assert "-allGeometry" not in buildup_core.check_command("")
    assert "-meshQuality" not in buildup_core.check_command("")


def test_every_reader_of_the_verdict_is_told_allgeometry_is_reference_only():
    """The desk, the production desk, its declare tool, the supervisor and both skills all
    make decisions off `checkMesh`. A corpus run lost most of its step budget chasing an
    advisory warning, and two desks re-ran the barer form after the stricter one failed as
    though that repaired something. Each reader has to carry the caveat, so this asserts
    per file rather than once."""
    from openreynolds.buildup import core as buildup_core
    from openreynolds.cad import brief as cad_brief

    root = pathlib.Path(__file__).resolve().parents[1]
    assert "reference reading, not the bar" in buildup_core.CORE_SYSTEM
    assert "reference reading" in cad_brief.CAD_SYSTEM
    for name in (".claude/agents/cad-supervisor.md",
                 ".claude/skills/cad-sweep/SKILL.md",
                 ".claude/skills/cad-addition/SKILL.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "-allGeometry" in text and "reference reading" in text, name


def test_every_meshing_case_states_the_extent_the_scale_probe_needs():
    """`_scale` reads `spec["extent_m"]`, `cad_sweep.py` passes `--spec` only when
    `<run-dir>/spec.json` exists, and nothing wrote that file -- so the probe returned
    `n/a` on every case of every sweep on disk, six reports deep, while appearing in the
    probe column as though it had screened something. It is the one probe aimed at a
    failure this corpus has actually seen: a STEP whose `LENGTH_UNIT` is empty, where the
    desk chose millimetres and shipped a mesh `checkMesh` passed.

    The number lives in the case file, not the run, because a desk that stated its own
    extent would state one agreeing with the unit it had just guessed."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from cad_buildup import cases

    corpus = cases()
    stated = {name: case.get("extent_m") for name, case in corpus.items()}
    refusals = {name for name, case in corpus.items() if case.get("expects") == "refused"}

    for name, case in corpus.items():
        if name in refusals:
            # Nothing is ever exported, so there is no union to measure and no number to
            # state. `n/a` here is the correct reading, not a gap.
            assert not stated[name], f"{name} is a refusal case and states an extent"
        else:
            assert stated[name] and stated[name] > 0, f"{name} states no extent_m"
