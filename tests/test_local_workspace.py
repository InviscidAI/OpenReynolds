"""A workspace on the machine the person is sitting at, with nothing installed.

These are about the two things that stood between "OpenReynolds is installed" and "a
simulation ran", on a laptop with no OpenFOAM, no Docker and no `python3`:

* **`python3` is not a name every machine has.** Measured on Windows 11 with Python
  3.13 installed and working: `python -c "import numpy"` prints 2.3.2 while `python3`
  is the Microsoft Store's execution alias, which exits 49 printing "Python was not
  found". Every toolbox script is documented as `python3 x.py` and
  `tests/test_toolbox.py` requires that literal in each docstring, so the whole
  toolbox was unreachable there -- including `lbm.py` and `sph.py`, the two solvers
  that need no OpenFOAM and are therefore the only reason such a machine can solve
  anything at all.
* **`/work` was honoured for file calls and not for commands.** `_resolve` mapped it
  to the real root, so `put_tree` and `get_file` landed correctly; nothing mapped it
  in the command text, so `python3 /work/.toolbox/lbm.py` -- the form `advise.py` and
  every docstring give -- looked under Git Bash's own root on Windows and under a
  literal `/work` that does not exist on Linux.

Both are measured here rather than asserted about a platform, because the platform
answers wrongly in both directions: a Windows runner may have no usable shell at all,
and a Windows developer box has one.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from openreynolds.backend import local as local_backend
from openreynolds.backend.base import BackendError
from openreynolds.backend.local import LocalBackend, what_is_missing


@pytest.fixture
def workspace(tmp_path: Path) -> LocalBackend:
    return LocalBackend(root=tmp_path / "work", bashrc=None)


# -- python3 ------------------------------------------------------------------


def test_a_machine_without_a_working_python3_gets_one_pointing_at_this_interpreter(
        workspace, monkeypatch):
    monkeypatch.setattr(local_backend, "_python3_works", lambda: False)
    workspace.python3_line = local_backend._python3_line()

    wrapped = workspace._wrap("python3 /x/y.py --n 32")

    assert wrapped.startswith("python3() {")
    assert "export -f python3" in wrapped
    assert sys.executable.replace(chr(92), "/") in wrapped
    assert wrapped.rstrip().endswith("python3 /x/y.py --n 32")


def test_a_machine_whose_python3_already_runs_is_left_exactly_alone(
        workspace, monkeypatch):
    """So a Linux or macOS session runs the string the agent wrote, unmodified. A
    rewrite nobody needs is a rewrite that can go wrong."""
    monkeypatch.setattr(local_backend, "_python3_works", lambda: True)
    workspace.python3_line = local_backend._python3_line()

    assert workspace._wrap("python3 -c 'x'") == "python3 -c 'x'"


def test_the_python3_probe_runs_an_interpreter_rather_than_looking_for_the_name(
        monkeypatch):
    """`shutil.which` cannot answer this: on Windows the Store alias is a real file on
    `PATH` that is not an interpreter, which is exactly the case that broke."""
    monkeypatch.setattr(local_backend, "_PYTHON3_OK", None)
    asked: list[list[str]] = []

    def refuse(cmd, **kwargs):
        asked.append(cmd)
        raise OSError("no such thing")

    monkeypatch.setattr(local_backend.subprocess, "run", refuse)
    assert local_backend._python3_works() is False
    assert asked and asked[0][:2] == ["python3", "-c"]


def test_an_interpreter_path_that_cannot_be_quoted_is_left_undefined(monkeypatch):
    """A definition built around an unquotable path would fail in a new and stranger
    way than the command failing as it already was."""
    monkeypatch.setattr(local_backend, "_python3_works", lambda: False)
    monkeypatch.setattr(local_backend.sys, "executable", 'C:/a"b/python.exe')

    assert local_backend._python3_line() == ""


# -- /work --------------------------------------------------------------------


def test_work_in_a_command_means_the_workspace_and_not_the_machines_root(workspace):
    wrapped = workspace._wrap("ls /work/.toolbox")

    assert wrapped.endswith(workspace.shell_root + "/.toolbox")
    assert not wrapped.endswith("ls /work/.toolbox")


def test_a_longer_word_beginning_with_work_is_left_alone(workspace):
    """`/workspace` and `/working` are not the workspace, and a substitution that
    could not tell the difference would corrupt a path it was never asked about."""
    for text in ("echo /workspace", "cd /working/dir", "echo work"):
        assert workspace._wrap(text).endswith(text)


def test_the_root_is_named_in_a_form_a_posix_shell_takes_as_one_word(workspace):
    assert chr(92) not in workspace.shell_root
    assert " " not in workspace.shell_root


def test_a_root_with_a_space_it_cannot_say_is_refused_with_the_way_out(tmp_path,
                                                                      monkeypatch):
    """Rather than writing a command that the shell reads as two words. The message
    has to name the switch, because the person did not choose this root -- the
    default is under their home directory, and `C:/Users/John Smith` is ordinary."""
    monkeypatch.setattr(local_backend, "shell_root", lambda root: "C:/Users/A B/work")

    with pytest.raises(BackendError) as refused:
        LocalBackend(root=tmp_path / "work", bashrc=None)

    assert "OPENREYNOLDS_LOCAL_WORK" in str(refused.value)
    assert refused.value.code == "unsayable_root"


def test_a_windows_root_with_a_space_is_said_by_its_short_name_where_there_is_one():
    """The 8.3 name is what Windows keeps for this. Skipped off Windows, where the
    question does not arise, rather than asserted about a path that does not exist."""
    if os.name != "nt":
        pytest.skip("8.3 short names are a Windows thing")
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    said = local_backend.shell_root(home)
    assert said and chr(92) not in said
    if " " in str(home):
        assert " " not in said, "a space survived where a short name should have been used"


# -- what the person is told --------------------------------------------------


def test_no_openfoam_is_told_what_does_work_and_not_only_what_does_not(workspace,
                                                                      monkeypatch):
    """The old sentence was "the agent's commands run, but meshing and solving will
    fail", which was true of OpenFOAM and false of the workspace: `lbm.py poiseuille
    --n 32` converges on a laptop with neither OpenFOAM nor Docker, in 4.1 s, to
    9.7e-07 of u_max against the exact parabola."""
    monkeypatch.setattr(local_backend.shutil, "which",
                        lambda name: None if name == "blockMesh" else "/usr/bin/bash")
    monkeypatch.setattr(local_backend, "_numpy_here", lambda backend: True)

    notes = " ".join(what_is_missing(workspace))

    assert "no OpenFOAM here" in notes
    assert "lbm.py" in notes and "sph.py" in notes
    assert "advise.py" in notes
    assert "OPENREYNOLDS_FOAM_BASHRC" in notes


def test_no_shell_is_said_first_because_nothing_at_all_will_run(workspace,
                                                               monkeypatch):
    monkeypatch.setattr(local_backend, "shell_found", lambda: False)
    monkeypatch.setattr(local_backend.shutil, "which", lambda name: None)
    monkeypatch.setattr(local_backend, "_numpy_here", lambda backend: True)

    notes = what_is_missing(workspace)

    assert notes and "nothing will run" in notes[0]
    assert "Git for Windows" in notes[0]


def test_no_numpy_is_said_too_because_it_takes_the_two_solvers_with_it(workspace,
                                                                      monkeypatch):
    monkeypatch.setattr(local_backend.shutil, "which",
                        lambda name: None if name == "blockMesh" else "/usr/bin/bash")
    monkeypatch.setattr(local_backend, "_numpy_here", lambda backend: False)

    notes = " ".join(what_is_missing(workspace))

    assert "no numpy" in notes


def test_a_machine_with_openfoam_is_told_nothing_about_it(tmp_path, monkeypatch):
    """A note that fires when there is nothing wrong is a note nobody reads."""
    monkeypatch.setattr(local_backend.shutil, "which", lambda name: "/usr/bin/" + name)
    backend = LocalBackend(root=tmp_path / "work", bashrc="/opt/openfoam/etc/bashrc")

    assert what_is_missing(backend) == []


def test_the_numpy_question_is_asked_of_the_interpreter_python3_resolves_to(workspace,
                                                                           monkeypatch):
    """Not of the one running the tests. On Windows they are different interpreters,
    which is the whole reason `_python3_line` exists."""
    asked: list[str] = []

    def record(cmd, **kwargs):
        asked.append(cmd)
        return local_backend.ExecResult(0, "", False, None)

    monkeypatch.setattr(workspace, "exec", record)
    assert local_backend._numpy_here(workspace) is True
    assert asked == ["python3 -c 'import numpy'"]


# -- the whole way through ----------------------------------------------------


def _a_shell_runs_here() -> bool:
    try:
        return subprocess.run([local_backend.SHELL, "-lc", "echo 1"],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _a_shell_runs_here(),
                    reason="no POSIX shell here, so no command can run at all")
def test_a_toolbox_script_runs_in_a_local_workspace_by_the_name_the_docs_give(tmp_path):
    """The end of it: the toolbox is pushed in as a session does it, and then invoked
    exactly as `advise.py` and every docstring say to -- `python3 /work/.toolbox/x.py`
    -- on whatever machine this is. Before the two fixes above, this failed on Windows
    twice over: `python3` was not an interpreter and `/work` was not the workspace."""
    from openreynolds import cli

    backend = LocalBackend(root=tmp_path / "work", bashrc=None)
    cli._sync_toolbox(backend)

    done = backend.exec("python3 /work/.toolbox/lbm.py --help", timeout_s=120)

    assert done.exit_code == 0, done.output
    assert "poiseuille" in done.output

# -- the option that chooses it ----------------------------------------------


def test_native_is_a_flag_of_its_own_and_local_still_means_docker(monkeypatch):
    """`--local` stays a boolean flag. Making it take `docker` or `native` read
    better and broke `openreynolds --local doctor`: Click consumes `doctor` as the
    value, fails the Choice, and the group callback never runs. That spelling is not
    hypothetical -- `cli._local_flag()` exists to append ` --local` to commands shown
    to the person."""
    from click.testing import CliRunner

    from openreynolds import cli

    for argv, want in (
        (["--local", "doctor"], "docker"),
        (["--native", "doctor"], "native"),
        # Asked for both, the person gets the sandboxed one.
        (["--local", "--native", "doctor"], "docker"),
        (["doctor"], None),
    ):
        monkeypatch.delenv("OPENREYNOLDS_LOCAL", raising=False)
        CliRunner().invoke(cli.main, argv, catch_exceptions=True)
        assert os.environ.get("OPENREYNOLDS_LOCAL") == want, argv


def test_the_native_flag_says_a_machine_with_no_openfoam_can_still_solve():
    """Because that is the whole reason it is a documented flag and not a variable
    mentioned in one clause of another flag's help."""
    from openreynolds import cli

    option, = [p for p in cli.main.params if p.name == "native"]

    assert "no Docker" in option.help and "no download" in option.help
    assert "numpy solvers" in option.help

# -- what doctor says --------------------------------------------------------


def test_doctor_does_not_fail_a_machine_that_can_still_solve(tmp_path, monkeypatch):
    """It used to print `FAIL  local OpenFOAM (native)` and count itself in "2
    check(s) failed", on a laptop where `lbm.py poiseuille --n 32` converges in 4.1 s
    to 9.7e-07 of u_max against the exact parabola. A red FAIL and a failure count is
    what a person reads as "this does not work here"."""
    from openreynolds import cli

    monkeypatch.setenv("OPENREYNOLDS_LOCAL_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(local_backend, "find_bashrc", lambda: None)
    monkeypatch.setattr(local_backend, "_numpy_here", lambda backend: True)

    name, ok, detail = cli._check_native_foam()

    assert ok is True, "a machine with numpy and no OpenFOAM is not a failed machine"
    assert "lbm.py" in detail and "sph.py" in detail
    assert "no OpenFOAM here" in detail, "and it still says what is missing"
    assert "OPENREYNOLDS_FOAM_BASHRC" in detail


def test_doctor_fails_a_machine_where_nothing_can_solve(tmp_path, monkeypatch):
    """Which is the answer a FAIL was always for."""
    from openreynolds import cli

    monkeypatch.setenv("OPENREYNOLDS_LOCAL_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(local_backend, "find_bashrc", lambda: None)
    monkeypatch.setattr(local_backend, "_numpy_here", lambda backend: False)

    name, ok, detail = cli._check_native_foam()

    assert ok is False
    assert "nothing here that can solve" in detail
    assert "pip install numpy" in detail
