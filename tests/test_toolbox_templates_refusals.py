"""A template on a machine with no OpenFOAM names what is missing.

Found by following a documented tutorial command exactly as `benchmarks/tutorials/`
gives it, on a laptop with no OpenFOAM:

    cp openreynolds/toolbox/templates/porous.py /tmp/slab/ && cd /tmp/slab && python3 porous.py

gmsh meshed it -- 18,122 nodes, 28,224 elements, written to `porous.msh` -- and then:

    + gmshToFoam porous.msh
    Traceback (most recent call last):
      ...
    FileNotFoundError: [WinError 2] The system cannot find the file specified

`run()` already handled a command that FAILS, naming the binary and showing its
stderr. It did not handle a command that is not there, because `subprocess.run` raises
before the returncode check. All six templates carry a byte-identical copy of that
helper, so all six ended the same way, and a template is the worst place for it: a
template is what a person copies.

These tests call `run()` directly rather than running a template end to end, so they
say the same thing on this laptop and inside the workspace image, where OpenFOAM is
present and the end-to-end path would never reach the refusal.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox" / "templates"
NAMES = sorted(p.name for p in TEMPLATES.glob("*.py"))


def _module(name: str):
    """A template imported as a module, without running its `main`."""
    spec = importlib.util.spec_from_file_location(f"_template_{name[:-3]}",
                                                  TEMPLATES / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_there_are_templates_to_check():
    """So a glob that matches nothing cannot pass this file silently."""
    assert len(NAMES) >= 6, NAMES


@pytest.mark.parametrize("name", NAMES)
def test_a_missing_binary_is_refused_by_name_and_blamed_on_openfoam(name, tmp_path,
                                                                    monkeypatch):
    module = _module(name)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "porous.msh").write_text("mesh\n", encoding="utf-8")

    with pytest.raises(SystemExit) as refused:
        module.run(["definitely-not-a-binary-abc123", "porous.msh"])

    said = str(refused.value)
    assert "definitely-not-a-binary-abc123" in said, "the binary is named"
    assert "not on PATH" in said
    assert "OpenFOAM" in said, "and where it was supposed to come from"
    # What has already been written, because the mesh is usually the expensive part
    # and a person needs to know it is not lost.
    assert "porous.msh" in said
    assert "--local" in said, "and one way to get a machine that has it"


@pytest.mark.parametrize("name", NAMES)
def test_an_empty_directory_says_nothing_yet_rather_than_listing_nothing(name, tmp_path,
                                                                        monkeypatch):
    module = _module(name)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as refused:
        module.run(["definitely-not-a-binary-abc123"])

    assert "nothing yet" in str(refused.value)


@pytest.mark.parametrize("name", NAMES)
def test_a_command_that_runs_and_fails_is_still_reported_as_before(name, tmp_path,
                                                                  monkeypatch):
    """The half that already worked, pinned so the new branch cannot swallow it."""
    module = _module(name)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as failed:
        module.run([sys.executable, "-c", "import sys; sys.exit(3)"])

    said = str(failed.value)
    assert "exit 3" in said
    assert "not on PATH" not in said, "a failure is not an absence"


@pytest.mark.parametrize("name", NAMES)
def test_a_command_that_succeeds_still_returns_its_result(name, tmp_path, monkeypatch):
    module = _module(name)
    monkeypatch.chdir(tmp_path)

    done = module.run([sys.executable, "-c", "print('hello from the template')"])

    assert done.returncode == 0
    assert "hello from the template" in done.stdout
