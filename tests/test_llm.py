"""Bring-your-own-model: the two API families behind one contract."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import openai
import pytest

from openreynolds import llm
from openreynolds.llm.anthropic_api import AnthropicProvider
from openreynolds.llm.base import Listener, TextBlock, ToolUseBlock, Turn, neutral_blocks, split_result
from openreynolds.llm.openai_api import OpenAIProvider

from conftest import FakeMessages, message, text_block, tool_block

# -- the neutral turn ------------------------------------------------------------


def test_a_turn_reads_its_own_blocks_whatever_shape_they_are():
    turn = Turn(
        content=[text_block("hi "), tool_block("bash", {"cmd": "ls"}, "tu_9"), TextBlock("there")],
        stop_reason="tool_use",
    )
    assert turn.text == "hi there"
    assert [(c.id, c.name, c.input) for c in turn.tool_calls] == [("tu_9", "bash", {"cmd": "ls"})]
    assert turn.as_message()["role"] == "assistant"


def test_neutral_blocks_drop_thinking():
    blocks = neutral_blocks(
        [SimpleNamespace(type="thinking", thinking="hmm"), text_block("a"), tool_block("bash", {}, "t")]
    )
    assert [type(b).__name__ for b in blocks] == ["TextBlock", "ToolUseBlock"]


def test_split_result_separates_text_and_images():
    text, images = split_result(
        [{"type": "text", "text": "a.png 1x1"}, {"type": "image", "source": {"data": "AA=="}}]
    )
    assert text == "a.png 1x1"
    assert len(images) == 1
    assert split_result("plain") == ("plain", [])


# -- presets and the factory --------------------------------------------------------


def test_every_preset_names_a_known_family():
    for preset in llm.PRESETS.values():
        assert preset.family in llm.FAMILIES
        assert preset.model and preset.desk_model and preset.context_window > 0


def test_family_of_accepts_presets_and_families_and_nothing_else():
    assert llm.family_of("zai") == "anthropic"
    assert llm.family_of("openrouter") == "openai"
    assert llm.family_of("openai") == "openai"
    assert llm.family_of("") == "anthropic"
    with pytest.raises(ValueError):
        llm.family_of("gemini-magic")


def cfg(**overrides):
    base = dict(provider="anthropic", llm_api_key="k", llm_base_url=None, llm_timeout_s=30.0)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_make_provider_picks_the_family_and_the_preset_endpoint():
    a = llm.make_provider(cfg())
    assert isinstance(a, AnthropicProvider)

    z = llm.make_provider(cfg(provider="zai"))
    assert isinstance(z, AnthropicProvider)
    assert "z.ai" in str(z.client.base_url)

    o = llm.make_provider(cfg(provider="ollama"))
    assert isinstance(o, OpenAIProvider)
    assert "11434" in str(o.client.base_url)

    explicit = llm.make_provider(cfg(provider="openai", llm_base_url="https://gateway.example/v1"))
    assert "gateway.example" in str(explicit.client.base_url)


# -- the Messages API ---------------------------------------------------------------


def anthropic_provider(responses, **kw):
    provider = AnthropicProvider("k")
    provider.client = SimpleNamespace(messages=FakeMessages(responses, **kw))
    return provider


def thread():
    return [
        {"role": "user", "content": "mesh it"},
        {
            "role": "assistant",
            "provider": "anthropic",
            "content": [tool_block("bash", {"cmd": "blockMesh"}, "tu_1")],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "done"}],
        },
    ]


def test_anthropic_sends_its_own_blocks_back_untouched():
    provider = anthropic_provider([message([text_block("ok")])])
    messages = thread()
    turn = provider.stream(
        model="m", system="sys", messages=messages, tools=[], effort="high",
        max_tokens=10, listener=Listener(),
    )
    sent = provider.client.messages.calls[0]
    assert sent["messages"][1]["content"] is messages[1]["content"]
    assert sent["thinking"]["type"] == "adaptive"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert turn.provider == "anthropic" and turn.text == "ok"
    assert turn.context_tokens == 110


def test_anthropic_rebuilds_another_providers_turn_from_neutral_blocks():
    provider = anthropic_provider([message([text_block("ok")])])
    messages = thread()
    messages[1]["provider"] = "openai"
    messages[1]["content"] = [TextBlock("running"), ToolUseBlock("call_1", "bash", {"cmd": "ls"})]
    provider.stream(
        model="m", system="s", messages=messages, tools=[], effort="high",
        max_tokens=10, listener=Listener(),
    )
    sent = provider.client.messages.calls[0]["messages"][1]["content"]
    assert sent == [
        {"type": "text", "text": "running"},
        {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"cmd": "ls"}},
    ]


def test_anthropic_goes_lean_when_a_compatible_endpoint_refuses_the_extras():
    class Picky(FakeMessages):
        def stream(self, **kwargs):
            if "thinking" in kwargs:
                self.calls.append(kwargs)
                raise anthropic.BadRequestError(
                    "unknown parameter: thinking",
                    response=SimpleNamespace(status_code=400, headers={}, request=None),
                    body=None,
                )
            return super().stream(**kwargs)

    provider = AnthropicProvider("k", "https://vendor.example/anthropic")
    provider.client = SimpleNamespace(messages=Picky([message([text_block("fine")])]))
    turn = provider.stream(
        model="m", system="s", messages=[{"role": "user", "content": "hi"}], tools=[],
        effort="high", max_tokens=10, listener=Listener(),
    )
    assert turn.text == "fine"
    assert provider.lean is True
    second = provider.client.messages.calls[-1]
    assert "thinking" not in second and "cache_control" not in second["system"][0]


def test_anthropic_reports_the_system_role_refusal_for_the_loop_to_fold():
    provider = anthropic_provider([message([text_block("x")])], fail_on_system=True)
    with pytest.raises(llm.BadRequest) as caught:
        provider.stream(
            model="m", system="s",
            messages=[{"role": "user", "content": "a"}, {"role": "system", "content": "fact"}],
            tools=[], effort="high", max_tokens=10, listener=Listener(),
        )
    assert "system" in str(caught.value)
    assert provider.lean is False


def test_anthropic_wraps_status_errors():
    class Down:
        def stream(self, **kwargs):
            raise anthropic.RateLimitError(
                "slow down",
                response=SimpleNamespace(status_code=429, headers={}, request=None),
                body=None,
            )

    provider = AnthropicProvider("k")
    provider.client = SimpleNamespace(messages=Down())
    with pytest.raises(llm.ProviderError) as caught:
        provider.stream(
            model="m", system="s", messages=[{"role": "user", "content": "a"}], tools=[],
            effort="high", max_tokens=10, listener=Listener(),
        )
    assert caught.value.status_code == 429


# -- Chat Completions ---------------------------------------------------------------


def chunk(*, content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, role=None)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)] if delta else [],
        usage=usage,
    )


def tc(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class FakeCompletions:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        script = self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]
        if isinstance(script, Exception):
            raise script
        if kwargs.get("stream"):
            return iter(script)
        return script


def openai_provider(scripts):
    provider = OpenAIProvider("k")
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(scripts)))
    return provider


def test_openai_renders_tool_results_as_tool_messages_and_images_as_a_user_turn():
    provider = OpenAIProvider("k")
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "provider": "anthropic",
            "content": [text_block("reading"), tool_block("read_file", {"path": "/work/a.png"}, "tu_1")],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [
                        {"type": "text", "text": "a.png 2x2"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
                    ],
                },
                {"type": "text", "text": "and hurry"},
            ],
        },
        {"role": "system", "content": "a job ended"},
    ]
    rendered = provider.render("SYS", messages)
    assert rendered[0] == {"role": "system", "content": "SYS"}
    assert rendered[1] == {"role": "user", "content": "look"}
    assert rendered[2]["role"] == "assistant" and rendered[2]["content"] == "reading"
    assert rendered[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert rendered[3] == {"role": "tool", "tool_call_id": "tu_1", "content": "a.png 2x2"}
    trailing = rendered[4]
    assert trailing["role"] == "user"
    assert trailing["content"][1]["image_url"]["url"].startswith("data:image/png;base64,QUJD")
    assert trailing["content"][2] == {"type": "text", "text": "and hurry"}
    assert rendered[5] == {"role": "system", "content": "a job ended"}


def test_openai_tools_are_function_tools():
    tools = OpenAIProvider.tools([{"name": "bash", "description": "run", "input_schema": {"type": "object"}}])
    assert tools == [
        {"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}}}
    ]


def test_openai_assembles_a_streamed_tool_call_and_reports_it_as_tool_use():
    provider = openai_provider(
        [
            [
                chunk(reasoning="let me "),
                chunk(reasoning="see"),
                chunk(content="Running "),
                chunk(content="it."),
                chunk(tool_calls=[tc(0, "call_a", "bash", '{"cmd": ')]),
                chunk(tool_calls=[tc(0, None, None, '"ls"}')]),
                chunk(finish="tool_calls"),
                chunk(usage=SimpleNamespace(prompt_tokens=40, completion_tokens=8)),
            ]
        ]
    )
    heard = {"thinking": [], "text": [], "began": 0}
    listener = Listener(
        thinking_begin=lambda: heard.__setitem__("began", heard["began"] + 1),
        thinking=heard["thinking"].append,
        text=heard["text"].append,
    )
    turn = provider.stream(
        model="m", system="s", messages=[{"role": "user", "content": "go"}],
        tools=[{"name": "bash", "input_schema": {}}], effort="high", max_tokens=10, listener=listener,
    )
    assert heard["began"] == 1
    assert "".join(heard["thinking"]) == "let me see"
    assert "".join(heard["text"]) == "Running it."
    assert turn.stop_reason == "tool_use"
    assert turn.text == "Running it."
    assert [(c.id, c.name, c.input) for c in turn.tool_calls] == [("call_a", "bash", {"cmd": "ls"})]
    assert turn.context_tokens == 48
    assert turn.raw["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'
    assert turn.raw["reasoning_content"] == "let me see"
    sent = provider.client.chat.completions.calls[0]
    assert sent["reasoning_effort"] == "high"
    assert sent["max_completion_tokens"] == 10


def test_openai_sends_its_own_raw_message_back():
    provider = OpenAIProvider("k")
    raw = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}
    rendered = provider.render("s", [{"role": "assistant", "provider": "openai", "raw": raw, "content": []}])
    assert rendered[1] is raw


def test_openai_falls_back_on_unknown_parameters_and_remembers():
    def bad(text):
        return openai.BadRequestError(
            text, response=SimpleNamespace(status_code=400, headers={}, request=None),
            body={"error": {"message": text}},
        )

    provider = openai_provider(
        [
            bad("Unsupported parameter: 'max_completion_tokens'"),
            bad("Unrecognized request argument supplied: reasoning_effort"),
            [chunk(content="hi"), chunk(finish="stop")],
        ]
    )
    turn = provider.stream(
        model="m", system="s", messages=[{"role": "user", "content": "go"}], tools=[],
        effort="high", max_tokens=10, listener=Listener(),
    )
    assert turn.text == "hi" and turn.stop_reason == "end_turn"
    assert provider.legacy_max_tokens is True and provider.lean is True
    last = provider.client.chat.completions.calls[-1]
    assert "max_tokens" in last and "reasoning_effort" not in last


def test_openai_length_and_filter_map_onto_the_neutral_stop_reasons():
    cut = openai_provider([[chunk(content="the pressure"), chunk(finish="length")]])
    assert cut.stream(model="m", system="s", messages=[{"role": "user", "content": "x"}], tools=[], effort="", max_tokens=1, listener=Listener()).stop_reason == "max_tokens"
    filtered = openai_provider([[chunk(finish="content_filter")]])
    assert filtered.stream(model="m", system="s", messages=[{"role": "user", "content": "x"}], tools=[], effort="", max_tokens=1, listener=Listener()).stop_reason == "refusal"


def test_openai_invalid_tool_arguments_reach_the_handler_as_a_marker():
    provider = openai_provider([[chunk(tool_calls=[tc(0, "c1", "bash", "{not json")]), chunk(finish="tool_calls")]])
    turn = provider.stream(model="m", system="s", messages=[{"role": "user", "content": "x"}], tools=[], effort="", max_tokens=1, listener=Listener())
    assert turn.tool_calls[0].input == {"__invalid_json__": "{not json"}


def test_openai_complete_returns_the_text():
    provider = openai_provider(
        [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  the agent is meshing "))])]
    )
    assert provider.complete(model="m", system="s", prompt="p", max_tokens=5) == "the agent is meshing"


def test_openai_wraps_status_errors():
    provider = openai_provider(
        [openai.AuthenticationError("bad key", response=SimpleNamespace(status_code=401, headers={}, request=None), body=None)]
    )
    with pytest.raises(llm.ProviderError) as caught:
        provider.complete(model="m", system="s", prompt="p", max_tokens=5)
    assert caught.value.status_code == 401


# -- sight, and Reynolds' own model ----------------------------------------------------------------


def test_the_reynolds_preset_borrows_the_services_address_and_key():
    p = llm.make_provider(cfg(provider="reynolds", llm_api_key="", foamd_url="https://api.example/", foamd_api_key="of_live_k"))
    assert isinstance(p, AnthropicProvider)
    assert str(p.client.base_url).rstrip("/") == "https://api.example/v1/llm"
    assert p.client.api_key == "of_live_k"
    assert llm.PRESETS["reynolds"].needs_key is False


def test_anthropic_probe_with_vision_sends_an_image_and_reads_the_refusal():
    class Counting:
        def __init__(self, reject):
            self.reject = reject
            self.calls = []

        def count_tokens(self, **kwargs):
            self.calls.append(kwargs)
            if self.reject:
                raise anthropic.BadRequestError(
                    "This model does not support image input",
                    response=SimpleNamespace(status_code=400, headers={}, request=None), body=None,
                )
            return SimpleNamespace(input_tokens=9)

    provider = AnthropicProvider("k")
    provider.client = SimpleNamespace(messages=Counting(reject=False))
    assert "can see images" in provider.probe("m", vision=True)
    sent = provider.client.messages.calls[0]["messages"][0]["content"]
    assert sent[0]["type"] == "image" and sent[0]["source"]["media_type"] == "image/png"

    provider.client = SimpleNamespace(messages=Counting(reject=True))
    with pytest.raises(llm.ProviderError) as refused:
        provider.probe("text-only", vision=True)
    assert "cannot see images" in str(refused.value)
    assert "text-only" in str(refused.value)


def test_openai_probe_with_vision_uses_a_data_uri_and_reads_the_refusal():
    seeing = openai_provider([SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])])
    assert "can see images" in seeing.probe("gpt-5", vision=True)
    content = seeing.client.chat.completions.calls[0]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    blind = openai_provider([
        openai.BadRequestError("Invalid content type. image_url is only supported by certain models.",
                               response=SimpleNamespace(status_code=400, headers={}, request=None),
                               body={"error": {"message": "Invalid content type. image_url is only supported by certain models."}})
    ])
    with pytest.raises(llm.ProviderError) as refused:
        blind.probe("text-only", vision=True)
    assert "cannot see images" in str(refused.value)


def test_a_stray_model_key_never_reaches_the_service_under_reynolds():
    p = llm.make_provider(cfg(provider="reynolds", llm_api_key="sk-ant-left-over", llm_base_url="https://elsewhere.example",
                             foamd_url="https://api.example", foamd_api_key="of_live_k"))
    assert p.client.api_key == "of_live_k"
    assert str(p.client.base_url).rstrip("/") == "https://api.example/v1/llm"

