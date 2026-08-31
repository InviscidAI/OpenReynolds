"""`hisa_env.py`: does this instance have the shock-capturing solver, and how is it run?

HiSA was compiled onto the workspace volume on 2026-08-30, in the middle of the ONERA M6
replication where a pressure-based solver converged and showed no shock. It is on the
volume, not in the image, so it is a property of an instance rather than of OpenReynolds
-- an instance rebuilt from scratch has no HiSA, and the failure mode this script exists
to prevent is a session that assumes either answer and finds out from a job log.

So the two cases tested hardest are the two answers: absent has to be a clean report and
not a traceback, and present has to be recognised without any of it being installed here.
The rest is shape -- the export lines and the example path are the payload, and JSON is
how another script reads them.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def hisa_env():
    spec = importlib.util.spec_from_file_location("hisa_env", TOOLBOX / "hisa_env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_inherited_foam_env(monkeypatch):
    """The build's platform name is read from `$WM_OPTIONS` when it is set, which is
    right in the container and would make these assertions depend on the host."""
    for name in ("WM_OPTIONS", "WM_PROJECT_USER_DIR", "FOAM_USER_LIBBIN", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(name, raising=False)


PLATFORM = "linux64GccDPInt32Opt"


def build(root: Path, *, binary: bool = True, example: bool = True) -> tuple[Path, Path]:
    """A fake of the 2026-08-30 layout: a user directory and a source tree."""
    user_dir = root / "OpenFOAM" / "user-v2512"
    appbin = user_dir / "platforms" / PLATFORM / "bin"
    libbin = user_dir / "platforms" / PLATFORM / "lib"
    appbin.mkdir(parents=True)
    libbin.mkdir(parents=True)
    if binary:
        target = appbin / "hisa"
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o755)

    source = root / "hisa" / "hisa"
    if example:
        (source / "examples" / "oneraM6" / "simulation" / "system").mkdir(parents=True)
    else:
        source.mkdir(parents=True)
    return user_dir, source


def test_a_fresh_instance_is_reported_absent_rather_than_crashing(hisa_env, tmp_path):
    """Nothing is there at all -- no user directory, no source tree, no binary."""
    found = hisa_env.probe(tmp_path / "OpenFOAM" / "user-v2512", tmp_path / "hisa")

    assert found["available"] is False
    assert found["binary_present"] is False
    assert "does not have it" in found["reason"]
    assert found["missing_libraries"] == []

    text = hisa_env.report(found)
    assert "not available" in text
    # Absence is only useful with a way out of it, and the cheap alternative is named.
    assert "rhoCentralFoam" in text
    assert "smallest cell" in text


def test_a_binary_on_the_volume_is_recognised(hisa_env, tmp_path):
    user_dir, source = build(tmp_path)
    found = hisa_env.probe(user_dir, source)

    assert found["available"] is True
    assert found["binary_present"] is True
    assert found["binary"].endswith(f"platforms/{PLATFORM}/bin/hisa")
    # A stub shell script is not an ELF binary, so ldd either declines or is not here.
    assert found["linkage"] in ("ok", "unchecked")
    assert found["example_present"] is True
    assert found["example"].endswith("examples/oneraM6/simulation")


def test_an_unresolved_shared_object_is_not_the_same_as_absent(hisa_env, tmp_path, monkeypatch):
    """The build that runs interactively and dies under mpirun. The binary is there;
    the ranks start without `FOAM_USER_LIBBIN` and cannot find `libhisa*.so`. Reporting
    that as "not installed" would send the next session off to rebuild it."""
    user_dir, source = build(tmp_path)
    monkeypatch.setattr(
        hisa_env,
        "link_check",
        lambda binary, libbin: ("missing", ["libhisa.so"]),
    )
    found = hisa_env.probe(user_dir, source)

    assert found["binary_present"] is True
    assert found["available"] is False
    assert found["missing_libraries"] == ["libhisa.so"]
    assert "do not resolve" in found["reason"]
    assert "libhisa.so" in hisa_env.report(found)


def test_the_exports_are_printed_in_dependency_order(hisa_env, tmp_path):
    user_dir, source = build(tmp_path)
    found = hisa_env.probe(user_dir, source)
    exports = found["exports"]

    assert exports[0].startswith("export WM_PROJECT_USER_DIR=")
    assert any("FOAM_USER_APPBIN=" in line for line in exports)
    assert any("FOAM_USER_LIBBIN=" in line for line in exports)
    # The library path appends rather than replaces: the image's own libraries are on it.
    assert "export LD_LIBRARY_PATH=$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH" in exports
    # MPI ranks inherit nothing, so the forwarding is spelled out rather than implied.
    assert found["mpirun"].count("-x ") == 4


def test_the_example_case_is_named_because_the_dictionaries_are_not_guessable(
    hisa_env, tmp_path
):
    """`foamToC` and `foamInfo` are absent from the image, so a HiSA dictionary set is
    not discoverable by introspection. The bundled case is the shortest way to one."""
    user_dir, source = build(tmp_path)
    text = hisa_env.report(hisa_env.probe(user_dir, source))
    assert "examples/oneraM6/simulation" in text

    user_dir, source = build(tmp_path / "second", example=False)
    found = hisa_env.probe(user_dir, source)
    assert found["example_present"] is False
    assert "not there" in hisa_env.report(found)


def test_json_is_the_whole_report_and_parses(hisa_env, tmp_path, capsys):
    user_dir, source = build(tmp_path)
    assert hisa_env.main(["--user-dir", str(user_dir), "--source", str(source), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {
        "available",
        "binary",
        "binary_present",
        "linkage",
        "missing_libraries",
        "reason",
        "user_dir",
        "appbin",
        "libbin",
        "source",
        "example",
        "example_present",
        "exports",
        "mpirun",
    }
    assert payload["available"] is True
    assert isinstance(payload["exports"], list)
    assert all(line.startswith("export ") for line in payload["exports"])
    assert isinstance(payload["missing_libraries"], list)


def test_exports_alone_are_evaluable(hisa_env, tmp_path, capsys):
    """`eval "$(python3 hisa_env.py --exports)"` has to yield shell and nothing else."""
    user_dir, source = build(tmp_path)
    hisa_env.main(["--user-dir", str(user_dir), "--source", str(source), "--exports"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines and all(line.startswith("export ") for line in lines)


def test_absence_exits_zero(hisa_env, tmp_path, capsys):
    """A question answered is not a command failed. A non-zero exit would make an
    honest "no" indistinguishable from a broken script inside a shell pipeline."""
    assert hisa_env.main(["--user-dir", str(tmp_path / "none"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["available"] is False


def test_the_platform_directory_is_read_rather_than_assumed(hisa_env, tmp_path, monkeypatch):
    """A build with different precision or label size lands somewhere else entirely."""
    user_dir = tmp_path / "user-v2512"
    (user_dir / "platforms" / "linux64GccDPInt64Opt" / "bin").mkdir(parents=True)
    assert hisa_env.platform_dir(user_dir) == "linux64GccDPInt64Opt"

    monkeypatch.setenv("WM_OPTIONS", "linux64GccSPInt32Opt")
    assert hisa_env.platform_dir(user_dir) == "linux64GccSPInt32Opt", (
        "the sourced environment knows better than the directory listing"
    )


def test_a_missing_ldd_is_unchecked_and_not_a_failure(hisa_env, tmp_path, monkeypatch):
    """This runs where there is no `ldd` at all, and an unverifiable build is not a
    broken one -- claiming otherwise would be the script inventing a problem.

    This assertion used to be reached by handing `link_check` a path with no binary on
    it, which enshrined the bug instead of testing the behaviour: absent and
    unanalysable are exactly the two things the three outcomes exist to keep apart, and
    routing both to `unchecked` collapsed them. Here the binary is real and it is `ldd`
    that is gone, which is the case the word `unchecked` was coined for.
    """
    user_dir, _ = build(tmp_path)
    binary = user_dir / "platforms" / PLATFORM / "bin" / "hisa"

    def no_ldd(*args, **kwargs):
        raise FileNotFoundError("ldd")

    monkeypatch.setattr(hisa_env.subprocess, "run", no_ldd)
    status, missing = hisa_env.link_check(binary, user_dir)
    assert status == "unchecked"
    assert missing == []


def test_an_absent_binary_is_missing_rather_than_unchecked(hisa_env, tmp_path):
    """Nothing to analyse is not the same as nothing to analyse it with. A session
    reading `unchecked` off an empty prefix would go looking for a broken `ldd`."""
    user_dir, _ = build(tmp_path, binary=False)
    status, missing = hisa_env.link_check(user_dir / "nothing" / "hisa", user_dir)
    assert status == "missing"
    assert missing == [], "no binary means no named library either"


def test_the_probe_is_general_and_looks_only_where_it_is_told(hisa_env, tmp_path, monkeypatch):
    """`find_binary` and `probe_binary` are the part of this file that is not about
    HiSA, and the next tool built onto the volume is meant to import them."""
    user_dir, _ = build(tmp_path)
    appbin = user_dir / "platforms" / PLATFORM / "bin"

    assert hisa_env.find_binary("hisa", appbin) == appbin / "hisa"
    assert hisa_env.find_binary("nosuchtool", appbin) is None

    # `$PATH` is a different question and is only asked when it is asked for.
    monkeypatch.setattr(hisa_env.shutil, "which", lambda name: "/usr/bin/" + name)
    assert hisa_env.find_binary("elsewhere", tmp_path / "empty") is None
    assert hisa_env.find_binary("elsewhere", tmp_path / "empty", search_path=True)

    found = hisa_env.probe_binary("nosuchtool", appbin)
    assert found["present"] is False
    assert found["linkage"] == "missing"
    # Not found is still reported somewhere, so the report can say where it looked.
    assert found["path"].as_posix().endswith("bin/nosuchtool")


def test_the_docstring_keeps_the_script_advisory(hisa_env):
    """The free-will contract reaches the toolbox too: this reports, and that is all."""
    # Normalised, because where a sentence happens to wrap is not the point.
    text = " ".join((hisa_env.__doc__ or "").split())
    assert "exports nothing, writes nothing, and installs nothing" in text
    assert "the reading can be wrong" in text
    for imperative in ("you must", "always ", "make sure", "you should"):
        assert imperative not in text.lower()


def test_it_needs_nothing_the_image_does_not_have(hisa_env):
    source = (TOOLBOX / "hisa_env.py").read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names) | {"__future__"}
    source.encode("ascii")  # raises if a non-ASCII character crept in
