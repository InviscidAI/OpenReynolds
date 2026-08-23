"""The agentic loop, driven against a fake model."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import pytest
from rich.console import Console

from openreynolds.config import CONTEXT_WINDOW_TOKENS, Config
from openreynolds.loop import Loop
from openreynolds.tools import ToolContext


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, tool_input: dict, block_id: str = "tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def message(content, stop_reason="end_turn", input_tokens=100):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=10,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self._response


class FakeMessages:
    def __init__(self, responses, fail_on_system=False):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.fail_on_system = fail_on_system

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_system and any(m["role"] == "system" for m in kwargs["messages"]):
            raise anthropic.BadRequestError(
                "role 'system' is not supported on this model",
                response=SimpleNamespace(status_code=400, headers={}, request=None),
                body=None,
            )
        return FakeStream(self._responses.pop(0))


@pytest.fixture
def loop(ctx: ToolContext, store, monkeypatch):
    cfg = Config(anthropic_api_key="test-key", model="claude-opus-5")
    console = Console(file=open(__import__("os").devnull, "w"), force_terminal=False)
    return Loop(cfg, ctx, store, console)


def install(loop, responses, fail_on_system=False):
    fake = FakeMessages(responses, fail_on_system=fail_on_system)
    loop.client = SimpleNamespace(messages=fake)
    return fake


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
