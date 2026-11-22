"""Core OpenReynolds installs, imports and runs with no `[live]` extra anywhere.

The extension's one acceptance criterion, and the only one that is about something not
happening. `openreynolds/live/` is a CUDA toolchain and a robotics simulator behind one
`pip install 'openreynolds[live]'`, and almost nobody will have it: if any of the
import chain that `openreynolds --help` walks reaches Warp, the product fails on the
laptop it is most often typed on. Three things are checked here, each where it can
actually fail:

1. the four imports, in a subprocess, with Warp hidden before anything is read;
2. `available()` and `require()`, which are what every entry point asks first;
3. `tools_for`, because the tool list is built once per request and a `live_sim` in it
   on a machine that cannot run one costs the model a call to find that out.

## Hiding Warp: `sys.modules["warp"] = None` rather than a `meta_path` finder

Both were tried; this one is used because it blocks the two different questions the
code asks, and a finder only blocks one of them cleanly.

`None` in `sys.modules` is the interpreter's own "this module is known to be absent"
marker. `import warp` then raises `ImportError: import of warp halted; None in
sys.modules`, which `units.require` turns into `LiveUnavailable`; and
`importlib.util.find_spec("warp")` raises `ValueError("warp.__spec__ is None")`, which
`live.available()` catches. A `meta_path` finder that returns `None` from `find_spec`
only makes the *search* fail -- and when Warp is genuinely installed, as it is on the
machine this was written on, a finder appended to `meta_path` never gets asked, while
one inserted at the front has to be careful not to answer for every other module in
the suite. One dict entry, restored by `monkeypatch`, is the whole fixture.

The subprocess is not decoration either: by the time pytest has collected this file
`openreynolds.cli` is already in `sys.modules`, so re-asserting that it imports proves
only that it imported once, earlier, on a machine that has Warp. A fresh interpreter
with the entry in place before the first import is the only honest form of the claim.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from openreynolds import live, live_tool
from openreynolds.tools import ToolContext, tools_for

HIDE = "import sys; sys.modules['warp'] = None\n"
"""Prepended to every subprocess script, before openreynolds is read."""


def without_warp(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter where Warp cannot be imported."""
    return subprocess.run([sys.executable, "-c", HIDE + code],
                          capture_output=True, text=True, timeout=300)


@pytest.fixture
def no_warp(monkeypatch):
    """Warp absent, for this test only."""
    monkeypatch.setitem(sys.modules, "warp", None)


# -- 1. the imports ------------------------------------------------------------


@pytest.mark.parametrize("module", [
    "openreynolds",
    "openreynolds.cli",
    "openreynolds.tools",
    "openreynolds.live",
])
def test_the_core_modules_import_with_no_warp_on_the_machine(module):
    """Each one on its own, so a failure names the module that reached for Warp.

    `openreynolds.live` is in the list on purpose: `tools.py` imports `live_tool`,
    which imports it to ask `available()`, so the package's own import has to be free
    of Warp too. That is what the lazy `__getattr__` in `live/__init__.py` is for.
    """
    done = without_warp(f"import {module}")
    assert done.returncode == 0, done.stderr


def test_the_help_still_prints_with_no_warp_on_the_machine():
    """The product claim, end to end: the group builds, `live` is in it, and the one
    thing printed is a help page rather than a traceback."""
    done = without_warp(
        "from click.testing import CliRunner\n"
        "from openreynolds import cli\n"
        "result = CliRunner().invoke(cli.main, ['--help'])\n"
        "assert result.exit_code == 0, result.output\n"
        "assert 'live' in result.output, result.output\n"
        "print('ok')\n"
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_a_live_command_without_the_extra_says_what_to_install():
    """And exits 1, so a script can tell "not installed" from "the checks failed"."""
    done = without_warp(
        "from click.testing import CliRunner\n"
        "from openreynolds import cli\n"
        "result = CliRunner().invoke(cli.main, ['live', 'cases'])\n"
        "print(result.exit_code)\n"
        "print(result.output)\n"
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == "1"
    assert "pip install 'openreynolds[live]'" in done.stdout


# -- 2. available() and require() ----------------------------------------------


def test_available_is_false_when_warp_cannot_be_imported(no_warp):
    assert live.available() is False
    assert live_tool.offered() is False


def test_require_names_the_install_command(no_warp):
    """The message is the whole value of the refusal. Checked as a string and not only
    as an exception type, because what reaches the person -- and the model, through
    `live_tool.handler` -- is the text."""
    with pytest.raises(ImportError) as refused:
        live.require()
    said = str(refused.value)
    assert "pip install 'openreynolds[live]'" in said
    assert "optional extra" in said
    assert type(refused.value).__name__ == "LiveUnavailable"


def test_the_refusal_is_reachable_without_importing_the_solver(no_warp):
    """`LiveUnavailable` and `require` were `world.py`'s, and reading `world.py`
    imports Warp through `lattice.py` -- so on the one machine the message is written
    for, `live.require()` raised `lattice.py`'s bare import error and the sentence
    naming the install command never ran. They are `units.py`'s now, which imports
    nothing but the standard library, and `world` re-exports them."""
    assert live.LiveUnavailable.__module__ == "openreynolds.live.units"
    assert issubclass(live.LiveUnavailable, ImportError)


# -- 3. the tool list ----------------------------------------------------------


def test_live_sim_is_not_offered_when_the_extra_is_absent(ctx, no_warp):
    """A tool whose only possible answer is "pip install this" is a round trip spent
    learning what the harness already knew (`tools_for`)."""
    assert "live_sim" not in [tool["name"] for tool in tools_for(ctx)]


def test_the_rest_of_the_tool_list_is_unchanged_by_its_absence(ctx, no_warp):
    ctx.cad = object()
    ctx.reviewer = object()
    names = [tool["name"] for tool in tools_for(ctx)]
    assert names == sorted(names)
    for kept in ("bash", "cad", "checkpoint", "fetch", "job_check", "job_kill",
                 "job_start", "mesh_review", "read_file", "send_to_agent",
                 "write_file"):
        assert kept in names


def test_the_list_is_still_one_object_per_answer(ctx, no_warp):
    """Withheld or offered, the list handed back is the same object every request: the
    prompt cache is keyed on the prefix the tool list is the start of, and a tool list
    rebuilt per request is the bug `tools_for`'s docstring is mostly about."""
    assert tools_for(ctx) is tools_for(ctx)
    # What the handler answers a call that arrives anyway is
    # `tests/test_live_tool.py`'s.
