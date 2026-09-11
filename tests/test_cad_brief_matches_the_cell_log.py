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
