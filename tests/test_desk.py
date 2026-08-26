"""The front desk: it answers the user while the main agent is busy, from the log."""

from __future__ import annotations

import time
from types import SimpleNamespace

import anthropic

from openreynolds.config import Config
from openreynolds.desk import Concierge, _render


def cfg():
    return Config(llm_api_key="k", desk_model="claude-haiku-4-5")


class FakeModel:
    """Stands in for the Haiku client: records prompts, returns scripted text."""

    def __init__(self, reply="the agent is 20% through the steady solve, ~19 min left"):
        self.reply = reply
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            stop_reason="end_turn",
        )


def desk_with(store, view, tracker=None, reply="ok"):
    desk = Concierge(cfg(), store, view, tracker)
    desk._client = FakeModel(reply)
    return desk


def seed(store, rows):
    for role, content in rows:
        store.append_message(role, content)


# -- rendering the shared log --------------------------------------------------


def test_a_tool_row_renders_as_a_call_and_its_output():
    row = {"tool": "bash", "input": {"cmd": "blockMesh > log 2>&1"}, "output": "exit_code: 0\nMesh OK"}
    rendered = _render(row)
    assert "bash(blockMesh" in rendered
    assert "exit_code: 0" in rendered


def test_plain_text_and_lists_render():
    assert _render("  hello   world ") == "hello world"
    assert "a" in _render([{"type": "text", "text": "a"}])


def test_the_transcript_is_built_from_the_store(store, view):
    seed(store, [
        ("assistant", "Launching the steady solve on 8 cores."),
        ("user", "how long will this take?"),
    ])
    desk = desk_with(store, view)
    text = desk._transcript()
    assert "[agent] Launching the steady solve" in text
    assert "[person] how long will this take?" in text


# -- answering -----------------------------------------------------------------


def test_a_reply_reaches_the_view_and_uses_the_transcript(store, view):
    seed(store, [("assistant", "Solving now, t=1.2 of 6 s.")])
    desk = desk_with(store, view, reply="the agent is about a fifth of the way through the solve")

    desk._reply("how far along?")

    assert view.desk_replies == ["the agent is about a fifth of the way through the solve"]
    prompt = desk._client.calls[-1]["messages"][0]["content"]
    assert "Solving now, t=1.2 of 6 s." in prompt
    assert "how far along?" in prompt
    assert desk._client.calls[-1]["model"] == "claude-haiku-4-5"


def test_a_reply_folds_in_live_job_facts(store, view):
    class FakeTracker:
        def snapshot(self):
            from openreynolds.progress import Progress

            return Progress("solving", "solving solve · Time 1.2 / 6 s", "p 8e-3", 0.2, True, 1)

        def facts_for_wake(self):
            return ["solve: solver time 1.2 of endTime 6, ClockTime 300 s"]

    desk = desk_with(store, view, FakeTracker())
    desk._reply("status?")

    prompt = desk._client.calls[-1]["messages"][0]["content"]
    assert "solver time 1.2 of endTime 6" in prompt
    assert "solving solve" in prompt


def test_the_now_line_reaches_the_view_and_is_one_line(store, view):
    seed(store, [("assistant", "Retuning the size fields near the tailgate.")])
    desk = desk_with(store, view, reply="reworking the near-wake mesh before spending solver time")
    desk._busy.set()

    desk._now()

    assert view.narrations == ["reworking the near-wake mesh before spending solver time"]


def test_the_now_line_is_rate_limited(store, view):
    desk = desk_with(store, view, reply="meshing")
    desk._busy.set()
    desk._now()
    desk._now()  # immediately again
    assert len(view.narrations) == 1, "two now-calls inside the min gap collapse to one"


def test_an_api_error_is_swallowed(store, view):
    desk = desk_with(store, view)

    def boom(**kwargs):
        raise anthropic.APIError("down", request=None, body=None)

    desk._client.messages.create = boom
    desk._reply("hi")  # must not raise
    assert view.desk_replies == []


# -- the queue and the thread --------------------------------------------------


def test_ask_answers_off_the_calling_thread(store, view):
    seed(store, [("assistant", "meshing")])
    desk = desk_with(store, view, reply="the agent is meshing the body")
    desk.start()
    try:
        desk.ask("what now?")
        deadline = time.time() + 5.0
        while not view.desk_replies and time.time() < deadline:
            time.sleep(0.01)
    finally:
        desk.stop()
    assert view.desk_replies == ["the agent is meshing the body"]


def test_blank_input_is_ignored(store, view):
    desk = desk_with(store, view)
    desk.ask("   ")
    assert desk._q.empty()


def test_working_false_stops_the_periodic_narration(store, view):
    desk = desk_with(store, view, reply="x")
    desk.working(False)
    # A 'now' item with the agent idle produces nothing.
    desk._q.put(("now", ""))
    desk.start()
    try:
        time.sleep(0.1)
    finally:
        desk.stop()
    assert view.narrations == []
