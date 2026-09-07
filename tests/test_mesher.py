"""The mesh desk: one bash block a step, a picture back, and a finish it cannot declare.

What these pin is the harness, not the model: that a command is taken out of a message
and run, that a render the command wrote comes back attached, that older pictures fall
out of the thread and their words do not, that `echo MESH_DONE` is checked rather than
believed, and that a budget ends a run that is going nowhere.
"""

from __future__ import annotations

import time

import pytest

from openreynolds import images
from openreynolds.backend.base import ExecResult
from openreynolds.config import Config
from openreynolds.llm import ProviderError, TextBlock, Turn
from openreynolds.mesher import MESH_DONE, Mesher, parse_action, system_prompt, task_message
from openreynolds.mesher.agent import KEEP_IMAGES, _evict

PNG = images.attachment(b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png")["source"]["data"]
RAW_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64

OK_JSON = (
    '{"polymesh": true, "cells": 3750, "faces": 15000, "points": 7600, '
    '"bounds": [0, 0, 0, 0.1, 0.05, 0.001], "two_d": true, '
    '"patches": [{"name": "inlet", "type": "patch", "nFaces": 20, "area": 0.0001, '
    '"center": [0, 0.025, 0.0005], "normal": [-1, 0, 0]}, '
    '{"name": "outlet", "type": "patch", "nFaces": 20}, '
    '{"name": "walls", "type": "wall", "nFaces": 400}], '
    '"build": ["Allmesh", "build.py"], "checkmesh": "Mesh OK.", "checkmesh_ok": true, '
    '"metrics": {"max_non_orthogonality": 12.4}, "render": "renders/mesh_look.png"}'
)

NOT_YET_JSON = '{"polymesh": false, "patches": [], "build": [], "checkmesh_ok": false}'


class ScriptedProvider:
    """Replays turns; repeats the last one if the loop keeps asking.

    The thread it is handed is one mutable list the loop keeps appending to, so each
    call is snapshotted -- a test that reads `calls[1]` wants what was sent then.
    """

    name = "anthropic"

    def __init__(self, texts, delay: float = 0.0):
        self.texts = list(texts)
        self.delay = delay
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(text, Exception):
            raise text
        return Turn(content=[TextBlock(text=text)], provider=self.name,
                    tokens={"input": 10, "output": 5})


def block(cmd: str, prose: str = "") -> str:
    return f"{prose}\n```bash\n{cmd}\n```"


def mesher(backend, store, texts, delay: float = 0.0, **cfg_kwargs):
    cfg = Config(llm_api_key="k", model="claude-opus-5", **cfg_kwargs)
    desk = Mesher(cfg, backend, store, "/work/study")
    desk.provider = ScriptedProvider(texts, delay=delay)
    return desk


def answers(backend, mapping, default=ExecResult(0, "", False, None)):
    """Route exec by a substring of the command, so tests read as intent."""
    real = backend.exec

    def routed(cmd, cwd=None, timeout_s=120, *, background=False):
        real(cmd, cwd, timeout_s, background=background)
        for needle, result in mapping.items():
            if needle in cmd:
                return result
        return default

    backend.exec = routed
    return backend


# -- reading a message ---------------------------------------------------------


def test_the_one_bash_block_is_the_action():
    assert parse_action("thinking\n```bash\nls -la\n```")[0] == "ls -la"
    assert parse_action("```sh\npwd\n```")[0] == "pwd"


def test_no_block_and_two_blocks_are_both_told_what_happened():
    cmd, complaint = parse_action("I will now consider the geometry at length.")
    assert cmd == "" and "no bash block" in complaint
    cmd, complaint = parse_action("```bash\na\n```\nand\n```bash\nb\n```")
    assert cmd == "" and "2 bash blocks" in complaint and "none of them ran" in complaint


def test_a_message_with_no_block_costs_a_turn_and_not_a_step(backend, store):
    """The complaint goes back and the run continues -- it is a nudge, not a failure."""
    desk = mesher(backend, store, ["no command here", block(f"echo {MESH_DONE}")])
    answers(backend, {"mesh_look.py": ExecResult(0, OK_JSON, False, None)})
    result = desk.run("a duct")
    assert result.ok
    assert result.steps == []
    assert "no bash block" in desk.provider.calls[1]["messages"][-1]["content"][0]["text"]


# -- running one --------------------------------------------------------------


def test_a_step_runs_in_the_case_directory_and_reports_the_exit_code(backend, store):
    desk = mesher(backend, store, [block("python3 build.py"), block(f"echo {MESH_DONE}")])
    answers(backend, {"build.py": ExecResult(2, "Traceback: boom", False, None),
                      "mesh_look.py": ExecResult(0, OK_JSON, False, None)})
    result = desk.run("a duct", case="valve")
    assert result.case_rel == "valve" and result.case_dir == "/work/study/valve"
    assert backend.execs[0] == "mkdir -p /work/study/valve"
    step = result.steps[0]
    assert step.cmd == "python3 build.py" and step.exit_code == 2
    observation = desk.provider.calls[1]["messages"][-1]["content"][0]["text"]
    assert observation.startswith("exit 2")
    assert "Traceback: boom" in observation


def test_a_png_the_command_named_comes_back_as_a_picture(backend, store):
    desk = mesher(backend, store, [block("python3 mesh_look.py . --out look.png"),
                                   block(f"echo {MESH_DONE}")])
    answers(backend, {"mesh_look.py . --out look.png": ExecResult(0, "drawn", False, None),
                      "--json": ExecResult(0, OK_JSON, False, None)})
    backend.files["/work/study/mesh/look.png"] = RAW_PNG
    result = desk.run("a duct")
    assert result.steps[0].image == "/work/study/mesh/look.png"
    blocks = desk.provider.calls[1]["messages"][-1]["content"]
    assert [b["type"] for b in blocks] == ["text", "image"]
    assert blocks[1]["source"]["media_type"] == "image/png"


def test_a_png_that_was_never_written_is_simply_not_attached(backend, store):
    desk = mesher(backend, store, [block("echo would draw look.png"),
                                   block(f"echo {MESH_DONE}")])
    answers(backend, {"mesh_look.py": ExecResult(0, OK_JSON, False, None)})
    result = desk.run("a duct")
    assert result.steps[0].image == ""
    assert [b["type"] for b in desk.provider.calls[1]["messages"][-1]["content"]] == ["text"]


def test_a_command_promoted_to_a_job_says_so_rather_than_looking_finished(backend, store):
    """The hosted workspace moves a long command to a detached job and answers 0. Read
    as "it finished" that is a mesh nobody waited for."""
    desk = mesher(backend, store, [block("sh Allmesh"), block(f"echo {MESH_DONE}")])
    answers(backend, {"Allmesh": ExecResult(0, "", False, None, job_id="job-7"),
                      "mesh_look.py": ExecResult(0, OK_JSON, False, None)})
    desk.run("a duct")
    observation = desk.provider.calls[1]["messages"][-1]["content"][0]["text"]
    assert "job-7" in observation and "do not start it again" in observation


def test_a_workspace_that_refuses_a_command_does_not_end_the_run(backend, store):
    def boom(cmd, cwd=None, timeout_s=120, *, background=False):
        if "build.py" in cmd:
            raise RuntimeError("sandbox gone")
        return ExecResult(0, OK_JSON, False, None)

    desk = mesher(backend, store, [block("python3 build.py"), block(f"echo {MESH_DONE}")])
    backend.exec = boom
    result = desk.run("a duct")
    assert result.steps[0].exit_code == -1
    assert result.ok


# -- the finish ---------------------------------------------------------------


def test_done_is_checked_not_believed(backend, store):
    """The whole point of the gate: the desk says done, the machine says otherwise,
    and the run continues with the reason in front of it."""
    desk = mesher(backend, store, [block(f"echo {MESH_DONE}", "built it"),
                                   block("python3 build.py"),
                                   block(f"echo {MESH_DONE}", "now it is real")])
    calls = {"n": 0}

    def routed(cmd, cwd=None, timeout_s=120, *, background=False):
        if "mesh_look.py" in cmd:
            calls["n"] += 1
            return ExecResult(0, NOT_YET_JSON if calls["n"] == 1 else OK_JSON, False, None)
        return ExecResult(0, "", False, None)

    backend.exec = routed
    result = desk.run("a duct")
    assert result.ok
    assert calls["n"] == 2
    refusal = desk.provider.calls[1]["messages"][-1]["content"][0]["text"]
    assert "did not pass" in refusal and "nothing has been meshed yet" in refusal
    assert result.summary == "now it is real"


def test_the_result_carries_the_mesh_facts_and_the_picture(backend, store):
    desk = mesher(backend, store, [block(f"echo {MESH_DONE}", "a 2D valve")])
    answers(backend, {"mesh_look.py": ExecResult(0, OK_JSON, False, None)})
    backend.files["/work/study/mesh/renders/mesh_look.png"] = RAW_PNG
    result = desk.run("a valve")
    assert result.ok and result.check.cells == 3750 and result.check.two_d
    assert result.check.render == "mesh/renders/mesh_look.png"
    assert result.png == RAW_PNG
    assert result.tokens == {"input": 10, "output": 5}
    joined = "\n".join(result.check.lines())
    assert "inlet" in joined and "normal (-1.00 +0.00 +0.00)" in joined
    assert "Mesh OK." in joined


# -- budgets ------------------------------------------------------------------


def test_running_out_of_steps_still_checks_what_is_on_disk(backend, store):
    """A run that never said done may still have left a mesh; it is looked at either
    way, because the one thing this desk exists to end is an unexamined mesh."""
    desk = mesher(backend, store, [block("echo working")], mesher_max_steps=3)
    answers(backend, {"mesh_look.py": ExecResult(0, OK_JSON, False, None)})
    result = desk.run("a duct")
    assert len(result.steps) == 3
    assert result.stopped == "steps"
    assert result.ok and result.check.cells == 3750


def test_running_out_of_time_stops_the_run(backend, store):
    desk = mesher(backend, store, [block("echo working")], delay=0.02,
                  mesher_max_seconds=0.01)
    answers(backend, {"mesh_look.py": ExecResult(0, NOT_YET_JSON, False, None)})
    result = desk.run("a duct")
    assert result.stopped == "time" and not result.ok


def test_a_model_failure_is_reported_rather_than_raised(backend, store):
    desk = mesher(backend, store, [ProviderError("429 overloaded", 429)])
    result = desk.run("a duct")
    assert not result.ok and result.stopped == "provider"
    assert "429 overloaded" in result.error


# -- the thread ---------------------------------------------------------------


def test_old_pictures_leave_the_thread_and_their_words_stay():
    messages = []
    for i in range(4):
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"exit 0 (step {i})"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
        ]})
        _evict(messages)
    with_images = [m for m in messages
                   if any(b.get("type") == "image" for b in m["content"])]
    assert len(with_images) == KEEP_IMAGES
    assert all(any("exit 0 (step" in b.get("text", "") for b in m["content"]) for m in messages)
    assert any("no longer shown" in b.get("text", "") for b in messages[0]["content"])


# -- the brief ----------------------------------------------------------------


def test_the_brief_says_how_to_act_what_is_installed_and_how_it_ends():
    brief = system_prompt(240)
    assert "exactly one fenced bash block" in brief
    assert "240 s" in brief
    assert MESH_DONE in brief and "does not take your word" in brief
    assert "mesh_look.py" in brief
    for fact in ("gmsh", "build123d", "cfMesh", "No network"):
        assert fact in brief


def test_the_brief_forbids_the_two_mistakes_that_were_paid_for():
    brief = system_prompt(240)
    assert "Never assign a patch by asking where a face sits in the bounding box" in brief
    assert "Look at every shape before you believe in it." in brief


def test_the_task_message_names_the_directory_and_the_request():
    text = task_message("a 20 degree branch", "/work/s/mesh", "mesh")
    assert "a 20 degree branch" in text and "/work/s/mesh" in text


@pytest.mark.parametrize("given,expected", [
    (None, "mesh"), ("", "mesh"), ("valve", "valve"), ("../escape", "escape"),
    ("/work/etc", "work_etc"), ("a b", "a_b"),
])
def test_the_case_is_a_directory_name_under_the_study(given, expected, backend, store):
    desk = mesher(backend, store, [block(f"echo {MESH_DONE}")], mesher_max_steps=2)
    answers(backend, {"mesh_look.py": ExecResult(0, NOT_YET_JSON, False, None)})
    assert desk.run("x", case=given).case_rel == expected
