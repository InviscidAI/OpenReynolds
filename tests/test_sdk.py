"""`import openreynolds`: the library face of the capabilities the MCP tools offer.

The surface is `openreynolds/sdk.py`, declared in the package's `__all__` and fetched
lazily, and underneath it is `openreynolds/mcp/core.py` over a `core.Host` -- the same
one the MCP server and `openreynolds serve` are built on. So the fakes here are the
MCP face's own: `LocalHost` with the stand-in session `tests/test_mcp_local.py` drives,
which means a shape that drifted between the two faces would fail in one of them.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import openreynolds
from openreynolds import cli, images, sdk
from openreynolds.mcp import core
from openreynolds.mcp.local import LocalHost
from test_mcp_local import Cfg, Studies

SURFACE = {"Study", "__version__", "fetch", "follow", "results", "show", "shutdown",
           "start", "studies", "upload"}
"""What `import openreynolds` promises. Adding to this is a decision; it is a published
surface, and the rest of the package is wiring that moves.

`show` and `follow` come from `notebook.py` rather than `sdk.py` -- the notebook's two
conveniences, on the same surface because a notebook imports the package once."""


@pytest.fixture
def world(tmp_path: Path):
    root = tmp_path / "studies"
    root.mkdir()
    studies = Studies(root)
    host = LocalHost(load_config=lambda: Cfg(root), run_session=studies,
                     local_workspace=lambda: None, staging_dir=tmp_path / "staging",
                     cwd=tmp_path)
    studies.ready.set()
    yield SimpleNamespace(host=host, studies=studies, root=root, tmp=tmp_path)
    host.close(wait_s=10)


# -- the surface itself ------------------------------------------------------------


def test_the_package_exposes_exactly_the_documented_names():
    assert set(openreynolds.__all__) == SURFACE
    assert dir(openreynolds) == sorted(openreynolds.__all__)
    for name in SURFACE:
        assert getattr(openreynolds, name) is not None
    assert openreynolds.start is sdk.start
    assert openreynolds.Study is sdk.Study
    with pytest.raises(AttributeError):
        openreynolds.no_such_thing  # noqa: B018 - the point is that it raises


def test_importing_the_package_does_not_drag_in_the_tui_or_a_model_client():
    """`import openreynolds` is what a script does to read a version or hold a Study.
    A session needs textual, a model client and the workspace plumbing; none of that
    may be on the import path to the name, so the names are fetched on first use."""
    script = (
        "import json, sys\n"
        "import openreynolds\n"
        "openreynolds.__version__\n"
        "openreynolds.Study\n"  # touching the surface must not cost a session either
        "heavy = ('textual', 'anthropic', 'openai', 'httpx2', 'rich', 'click', 'mcp')\n"
        "print(json.dumps(sorted(m for m in heavy if m in sys.modules)))\n"
    )
    root = Path(openreynolds.__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=300, cwd=str(root))
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def test_a_library_caller_does_not_have_to_monkeypatch_the_console(monkeypatch):
    """`mcp/local.py` kept stdout clean by assigning `cli.console` from another module
    (`local.py:715`). A library caller must not have to reach into a module global, so
    `cli.quiet_stdout()` is the seam the SDK uses instead."""
    monkeypatch.setattr(cli, "console", cli.console)
    try:
        cli.quiet_stdout()
        assert cli.console.file is sys.stderr
    finally:
        images.allow_drawing()  # `quiet_stdout` also stops inline images, for this process
    assert "cli.console =" not in Path(sdk.__file__).read_text(encoding="utf-8")
    assert "cli.quiet_stdout()" in Path(sdk.__file__).read_text(encoding="utf-8")


# -- a study, through the SDK ------------------------------------------------------


def test_a_study_is_started_driven_and_read_through_the_sdk(world):
    study = sdk.start("flow past a cylinder at Re=100", host=world.host)
    assert study.state == "starting"

    states = [event["state"] for event in study.events(wait_s=5)]
    assert states[-1] == "waiting_for_input", states
    assert all(state in core.STATES for state in states)

    answered = study.status()
    assert answered["last_message"] == "done: flow past a cylinder at Re=100"
    assert answered["next"] == core.NEXT["waiting_for_input"]
    assert answered["transcript"] == [], "the cursor moved: nothing is answered twice"

    study.send("now at Re=200")
    said: list[str] = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(said) < 2:
        said += [row.get("text") for row in study.status(wait_s=0.2)["transcript"]]
    assert said == ["now at Re=200", "done: now at Re=200"]

    results = study.results()
    assert results["final_message"] == "done: now at Re=200"
    assert results["results_json"] == {"drag_N": 1.5}
    assert sdk.results(study.study_id, host=world.host) == results

    path = f"/work/{study.study_id}/results.json"
    assert study.file(path).read_bytes() == b'{"drag_N": 1.5}'
    with pytest.raises(core.McpError, match="not a workspace path"):
        sdk.fetch(study.study_id, "/etc/passwd", host=world.host)

    listed = sdk.studies(host=world.host)
    assert [card["study_id"] for card in listed["studies"]] == [study.study_id]
    assert [live["session_id"] for live in listed["live_sessions"]] == [study.session_id]

    assert study.end() == "ending"
    assert study.wait(wait_s=5)["state"] == "ended"
    with pytest.raises(core.McpError, match="has ended"):
        study.send("more")


def test_an_approval_is_asked_and_answered_through_the_sdk(world):
    study = sdk.start("ask me", mode="partial", host=world.host)
    answered = False
    for event in study.events(wait_s=5):
        if event["state"] == "waiting_for_approval" and not answered:
            assert event["approvals"][0]["title"] == "Run the solve?"
            assert study.answer("approve") == "/yes"
            answered = True
    assert answered, "the study never asked"
    assert study.status()["last_message"] == "approval: /yes"
    with pytest.raises(core.McpError, match="decision is"):
        study.answer("maybe")


def test_an_uploaded_file_reaches_the_study_and_a_study_is_continued(world):
    geometry = world.tmp / "wing section.stl"
    geometry.write_bytes(b"solid wing\nendsolid wing\n")
    staged = sdk.upload("wing section.stl", host=world.host)
    assert staged["name"] == "wing_section.stl"

    study = sdk.start("drag on this wing", upload_ids=[staged["upload_id"]], host=world.host)
    last = study.wait(wait_s=5)
    placed = f"/work/{study.study_id}/uploads/wing_section.stl"
    assert world.studies.browsers[study.study_id].backend.files[placed] == geometry.read_bytes()
    assert last["state"] == "waiting_for_input"
    study.end()

    again = sdk.start("carry on", study_id=study.study_id, host=world.host)
    assert again.wait(wait_s=5)["study_id"] == study.study_id
    assert world.studies.calls[-1]["study_id"] == study.study_id
    with pytest.raises(core.McpError, match="no study"):
        sdk.start("nowhere", study_id="../etc", host=world.host)
