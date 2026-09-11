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
