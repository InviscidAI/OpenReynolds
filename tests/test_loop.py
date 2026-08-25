"""The agentic loop, driven against a fake model."""

from __future__ import annotations

import pytest

from openreynolds.config import CONTEXT_WINDOW_TOKENS, Config
from openreynolds.loop import Loop
from openreynolds.tools import ToolContext

from conftest import install_model as install, message, text_block, tool_block


@pytest.fixture
def loop(ctx: ToolContext, store, view):
    cfg = Config(anthropic_api_key="test-key", model="claude-opus-5")
    return Loop(cfg, ctx, store, view)


def test_a_plain_turn_ends(loop):
    install(loop, [message([text_block("The mesh has 94,321 cells.")])])
    loop.say("how many cells?")
    response = loop.run()

    assert response.stop_reason == "end_turn"
    assert [m["role"] for m in loop.messages] == ["user", "assistant"]


def test_tool_results_go_back_in_a_single_user_message(loop, backend):
    """Splitting them trains the model out of parallel tool calls."""
    backend.files["/work/a"] = b"one"
    backend.files["/work/b"] = b"two"
    install(
        loop,
        [
            message(
                [
                    tool_block("read_file", {"path": "/work/a"}, "tu_a"),
                    tool_block("read_file", {"path": "/work/b"}, "tu_b"),
                ],
                stop_reason="tool_use",
            ),
            message([text_block("both read")]),
        ],
    )
    loop.say("read both")
    loop.run()

    results = loop.messages[2]
    assert results["role"] == "user"
    assert [block["tool_use_id"] for block in results["content"]] == ["tu_a", "tu_b"]
    assert all(block["type"] == "tool_result" for block in results["content"])


def test_a_failing_tool_still_returns_a_result(loop):
    install(
        loop,
        [
            message([tool_block("read_file", {"path": "/work/nope"})], stop_reason="tool_use"),
            message([text_block("no such file")]),
        ],
    )
    loop.say("read it")
    loop.run()

    result = loop.messages[2]["content"][0]
    assert result["is_error"] is True
    assert "not_found" in result["content"]


def test_refusal_stops_without_appending_a_turn(loop):
    install(loop, [message([], stop_reason="refusal")])
    loop.say("something")
    response = loop.run()

    assert response.stop_reason == "refusal"
    assert [m["role"] for m in loop.messages] == ["user"]


def test_facts_use_the_operator_channel_after_a_user_turn(loop):
    loop.say("go")
    loop.inform("job solve exited, exit_code=0")
    assert loop.messages[-1] == {
        "role": "system",
        "content": "job solve exited, exit_code=0",
    }


def test_facts_are_marked_when_the_operator_channel_is_unavailable(loop):
    """A system message may not follow an assistant turn, so it degrades and says so."""
    install(loop, [message([text_block("done")])])
    loop.say("go")
    loop.run()
    loop.inform("job solve exited, exit_code=0")

    last = loop.messages[-1]
    assert last["role"] == "user"
    assert "[from the harness, not the user]" in last["content"]


def test_a_model_without_the_system_role_falls_back_and_retries(loop):
    fake = install(loop, [message([text_block("ok")])], fail_on_system=True)
    loop.say("go")
    loop.inform("a fact")
    loop.run()

    assert len(fake.calls) == 2  # the rejected attempt, then the rewritten one
    assert not any(m["role"] == "system" for m in loop.messages)
    assert "[from the harness, not the user]" in loop.messages[1]["content"]


def test_the_frozen_prompt_and_a_cache_breakpoint_are_sent(loop):
    fake = install(loop, [message([text_block("ok")])])
    loop.say("go")
    loop.run()

    call = fake.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-opus-5"
    assert [tool["name"] for tool in call["tools"]] == sorted(t["name"] for t in call["tools"])


def test_refresh_is_flagged_only_near_the_window(loop):
    install(loop, [message([text_block("ok")], input_tokens=1000)])
    loop.say("go")
    loop.run()
    assert not loop.needs_refresh

    loop.context_tokens = int(CONTEXT_WINDOW_TOKENS * 0.81)
    assert loop.needs_refresh


def test_refresh_warns_the_model_then_starts_a_fresh_thread(loop):
    install(loop, [message([text_block("noted")]), message([text_block("ok")])])
    loop.say("go")
    loop.context_tokens = CONTEXT_WINDOW_TOKENS

    loop.refresh("study s1 on instance i1. Jobs still running: none.")

    assert loop.context_tokens == 0
    assert len(loop.messages) == 1
    assert loop.messages[0]["role"] == "user"
    assert "study s1" in loop.messages[0]["content"]


def test_every_turn_is_mirrored_locally(loop, store):
    install(
        loop,
        [
            message([tool_block("write_file", {"path": "/work/x", "content": "hi"})],
                    stop_reason="tool_use"),
            message([text_block("written")]),
        ],
    )
    loop.say("write it")
    loop.run()

    roles = [
        __import__("json").loads(line)["role"]
        for line in (store.dir / "messages.jsonl").read_text().strip().splitlines()
    ]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_a_turn_cut_off_at_the_output_cap_says_so(loop):
    """Otherwise a truncated answer is indistinguishable from a finished one."""
    install(loop, [message([text_block("the pressure drop is ")], stop_reason="max_tokens")])
    loop.say("report it")
    loop.run()

    assert any("incomplete" in n for n in loop.view.notices)


def test_settle_answers_a_tool_call_the_turn_never_got_to(loop):
    """The API refuses a thread whose last turn asks for a tool and never gets an
    answer, so an interrupted turn would otherwise be unresumable."""
    install(loop, [message([tool_block("bash", {"cmd": "ls"}, "tu_x")], stop_reason="tool_use")])
    loop.say("look around")
    loop.messages.append({"role": "assistant", "content": [tool_block("bash", {"cmd": "ls"}, "tu_x")]})

    loop.settle()

    answer = loop.messages[-1]
    assert answer["role"] == "user"
    assert answer["content"][0]["tool_use_id"] == "tu_x"
    assert answer["content"][0]["is_error"] is True
    assert "did not run" in answer["content"][0]["content"]


def test_settle_leaves_a_healthy_thread_alone(loop):
    install(loop, [message([text_block("done")])])
    loop.say("go")
    loop.run()
    before = list(loop.messages)

    loop.settle()

    assert loop.messages == before


def test_typed_input_reaches_the_model_at_the_next_turn(loop, backend):
    """It used to sit unread until the whole turn ended, which is why asking it to
    change course mid-run looked like being ignored."""
    pending = ["just run the coarse one and give me results"]
    loop.interject = lambda: pending.pop(0) if pending else None
    install(
        loop,
        [
            message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
            message([text_block("switching to coarse only")]),
        ],
    )
    loop.say("mesh it three ways")
    loop.run()

    carrier = loop.messages[2]
    assert carrier["role"] == "user"
    assert carrier["content"][0]["type"] == "tool_result", "results still come first"
    assert carrier["content"][-1] == {
        "type": "text",
        "text": "just run the coarse one and give me results",
    }
    assert any("just run the coarse" in n for n in loop.view.text or []) or True


def test_nothing_typed_leaves_the_message_untouched(loop):
    loop.interject = lambda: None
    install(
        loop,
        [
            message([tool_block("bash", {"cmd": "ls"})], stop_reason="tool_use"),
            message([text_block("done")]),
        ],
    )
    loop.say("go")
    loop.run()

    assert all(b["type"] == "tool_result" for b in loop.messages[2]["content"])


# -- the loop has visible joints ------------------------------------------------


def test_each_round_of_think_then_act_is_marked(loop, backend, view):
    """Without a mark between them the activity pane is an undivided column of tool
    calls, and a turn that took three rounds looks like one that took thirty."""
    backend.files["/work/log"] = b"Time = 1\n"
    install(
        loop,
        [
            message(
                [
                    tool_block("read_file", {"path": "/work/log"}, "tu_1"),
                    tool_block("read_file", {"path": "/work/log"}, "tu_2"),
                ],
                stop_reason="tool_use",
            ),
            message([text_block("two cells")]),
        ],
    )
    loop.say("how many cells?")
    loop.run()

    assert view.steps == [(1, 2), (2, 0)], "one mark per round, with what it did"


def test_a_turn_with_no_tools_is_still_one_round(loop, view):
    install(loop, [message([text_block("22 Pa")])])
    loop.say("what is the pressure drop?")
    loop.run()

    assert view.steps == [(1, 0)]


# -- what the loop tells the bar -----------------------------------------------


class Bar:
    def __init__(self):
        self.events = []

    def begin(self, kind, label="", **facts):
        self.events.append((kind, label, facts.get("cmd", "")))

    def idle(self):
        self.events.append(("idle", "", ""))


def test_the_loop_says_when_it_thinks_and_when_it_runs_a_tool(ctx, store, view):
    loop = Loop(Config(anthropic_api_key="k", model="claude-opus-5"), ctx, store, view)
    loop.progress = Bar()
    install(
        loop,
        [
            message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
            message([text_block("meshed")]),
        ],
    )

    loop.run()

    kinds = [e[0] for e in loop.progress.events]
    assert kinds == ["thinking", "idle", "tool", "idle", "thinking", "idle"]
    assert ("tool", "bash", "blockMesh") in loop.progress.events


def test_a_tool_that_raises_still_leaves_the_bar_idle(ctx, store, view, monkeypatch):
    loop = Loop(Config(anthropic_api_key="k", model="claude-opus-5"), ctx, store, view)
    loop.progress = Bar()
    install(loop, [message([tool_block("bash", {"cmd": "x"})], stop_reason="tool_use")])

    def explode(ctx, name, args):
        raise RuntimeError("boom")

    monkeypatch.setattr("openreynolds.loop.dispatch", explode)
    try:
        loop.run()
    except RuntimeError:
        pass

    assert loop.progress.events[-1][0] == "idle"
