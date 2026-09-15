"""Changing model, provider and effort mid-study (`switch.py`), with fake providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openreynolds import commands, switch
from openreynolds.config import Config
from openreynolds.llm import ProviderError, TextBlock, Turn
from openreynolds.loop import Loop
from openreynolds.tools import ToolContext


class FakeProvider:
    """Probes as told and answers every request with one short turn."""

    def __init__(self, refuse: str = "", context_tokens: int = 100):
        self.refuse = refuse
        self.context_tokens = context_tokens
        self.probed: list[str] = []
        self.calls: list[dict] = []
        self.client = None

    def probe(self, model, vision=False):
        self.probed.append(model)
        if self.refuse:
            raise ProviderError(self.refuse, 400)
        return f"{model} reachable"

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return Turn(content=[TextBlock("ok")], stop_reason="end_turn", stop_explanation="",
                    context_tokens=self.context_tokens, tokens={}, provider="fake")


class Models:
    """Stands in for `make_provider`, recording what it was asked to build."""

    def __init__(self, refuse: str = ""):
        self.refuse = refuse
        self.built: list[tuple] = []
        self.last: FakeProvider | None = None

    def __call__(self, cfg, **kwargs):
        self.built.append((cfg.provider, cfg.model, cfg.llm_api_key))
        self.last = FakeProvider(self.refuse)
        return self.last


@pytest.fixture
def models(monkeypatch):
    fake = Models()
    monkeypatch.setattr("openreynolds.switch.make_provider", fake)
    return fake


@pytest.fixture
def seen(view):
    view.models = []
    view.model = lambda model, effort, provider: view.models.append((model, effort, provider))
    return view


@pytest.fixture
def loop(store, seen):
    cfg = Config(llm_api_key="k", model="claude-opus-5")
    ctx = ToolContext(backend=None, store=store, max_output=1000)
    made = Loop(cfg, ctx, store, seen)
    made.provider = FakeProvider()
    return made


# -- choosing --------------------------------------------------------------------


def test_bare_model_reports_the_session(loop):
    text = "\n".join(switch.request(loop, ""))
    assert "anthropic" in text and "claude-opus-5" in text and "high" in text
    assert "mode auto" in text
    assert "claude-sonnet-5" in text  # a known model for the provider


def test_what_was_typed_is_read_as_provider_and_model():
    cfg = Config(llm_api_key="k")
    assert switch.parse("claude-sonnet-5", cfg) == ("anthropic", "claude-sonnet-5")
    assert switch.parse("openai:gpt-5-mini", cfg) == ("openai", "gpt-5-mini")
    assert switch.parse("openai", cfg) == ("openai", "gpt-5")
    # A colon inside a model id is not a provider.
    assert switch.parse("qwen3:8b", cfg) == ("anthropic", "qwen3:8b")


def test_a_same_provider_switch_is_checked_and_waits_for_the_next_run(loop, models):
    said = "\n".join(switch.request(loop, "claude-sonnet-5", env={}))
    assert models.last.probed == ["claude-sonnet-5"]
    assert loop.pending_model.model == "claude-sonnet-5"
    assert loop.cfg.model == "claude-opus-5"  # not yet
    assert "next message" in said


def test_mid_turn_the_switch_says_it_waits_for_the_turn(loop, models):
    loop.running = True
    assert "current turn ends" in "\n".join(switch.request(loop, "claude-sonnet-5", env={}))


def test_another_provider_without_a_key_is_refused_with_how_to_add_one(loop, models):
    said = "\n".join(switch.request(loop, "openai:gpt-5", env={}))
    assert "no key for openai" in said
    assert "OPENAI_API_KEY" in said and "openreynolds config --provider openai" in said
    assert loop.pending_model is None
    assert models.built == []  # nothing was probed


def test_another_provider_with_a_key_in_the_environment(loop, models):
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    pending = loop.pending_model
    assert (pending.provider, pending.model, pending.llm_api_key) == ("openai", "gpt-5", "sk-o")
    assert pending.context_window == 400_000
    assert pending.desk_model == "gpt-5-mini"


def test_reynolds_needs_only_the_service_key(loop, models):
    loop.cfg.foamd_api_key = ""
    assert "no key" in "\n".join(switch.request(loop, "reynolds", env={}))
    loop.cfg.foamd_api_key = "svc"
    switch.request(loop, "reynolds:claude-opus-5", env={})
    # The preset's default must not replace the model that was asked for.
    assert (loop.pending_model.provider, loop.pending_model.model) == ("reynolds", "claude-opus-5")


def test_a_refused_probe_refuses_the_switch(loop, monkeypatch):
    monkeypatch.setattr("openreynolds.switch.make_provider",
                        Models(refuse="glm-4.6 cannot see images"))
    said = "\n".join(switch.request(loop, "glm-4.6", env={}))
    assert "not switching" in said and "cannot see images" in said
    assert loop.pending_model is None


# -- applying --------------------------------------------------------------------


def test_the_pending_model_is_applied_at_the_next_run(loop, models, store, seen):
    switch.request(loop, "claude-sonnet-5", env={})
    before = loop.provider
    loop.say("carry on")
    loop.run()
    assert loop.cfg.model == "claude-sonnet-5"
    assert before.calls[-1]["model"] == "claude-sonnet-5"  # same provider: not rebuilt
    assert loop.pending_model is None
    assert store.session.model == "claude-sonnet-5"
    assert seen.models[-1] == ("claude-sonnet-5", "high", "anthropic")


def test_a_provider_switch_rebuilds_the_client_and_recomputes_the_window(loop, models):
    loop._no_system_role = True
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    loop.say("carry on")
    loop.run()
    assert loop.provider is models.last
    assert models.last.calls[-1]["model"] == "gpt-5"
    assert loop.window == 400_000
    assert loop._no_system_role is False
    assert loop.cfg.desk_model == "gpt-5-mini"


def test_earlier_thinking_is_stripped_when_the_model_changes(loop, models):
    loop.messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "s"},
            SimpleNamespace(type="redacted_thinking", data="x"),
            TextBlock("done"),
        ]},
    ]
    switch.request(loop, "claude-sonnet-5", env={})
    loop.say("next")
    loop.run()
    kinds = [getattr(b, "type", None) or b.get("type") for b in loop.messages[1]["content"]]
    assert kinds == ["text"]


def test_a_turn_that_was_only_reasoning_keeps_some_content():
    """The APIs refuse an assistant turn with no content at all."""
    messages = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "hmm", "signature": "s"}]}]
    assert switch.strip_thinking(messages) == 1
    (block,) = messages[0]["content"]
    assert block["type"] == "text" and block["text"]


def test_a_thread_too_big_for_the_new_window_is_refreshed_first(loop, models):
    loop.context_tokens = 500_000
    said = "\n".join(switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"}))
    assert loop.refresh_due and "refreshed" in said
    old = loop.provider
    loop.refresh("fresh situation")
    # The refresh ran on the model that built the thread, and then made way.
    assert old.calls and old.calls[-1]["model"] == "claude-opus-5"
    assert loop.refresh_due is False and loop.pending_model is not None
    loop.say("go on")
    loop.run()
    assert loop.cfg.model == "gpt-5" and loop.window == 400_000


def test_cancelling_an_oversize_switch_cancels_its_refresh(loop, models):
    """A refresh left behind by a cancelled switch threw the whole thread away."""
    loop.context_tokens = 500_000
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    assert loop.refresh_due
    said = "\n".join(switch.request(loop, "claude-opus-5", env={}))
    assert "already on" in said
    assert loop.pending_model is None and loop.refresh_due is False


def test_a_replacing_switch_that_fits_does_not_inherit_the_refresh(loop, models):
    loop.context_tokens = 500_000
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    switch.request(loop, "claude-sonnet-5", env={})
    assert loop.pending_model.model == "claude-sonnet-5" and loop.refresh_due is False


def test_a_refused_second_switch_leaves_the_first_and_its_refresh(loop, models, monkeypatch):
    loop.context_tokens = 500_000
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    monkeypatch.setattr("openreynolds.switch.make_provider", Models(refuse="no such model"))
    switch.request(loop, "claude-nonsense", env={})
    assert loop.pending_model.model == "gpt-5" and loop.refresh_due


def test_a_model_change_on_the_same_client_clears_what_it_latched(loop, models):
    """`lean` (no thinking, effort or cache) was learned about the old model."""
    client = loop.provider
    client.lean = True
    client.legacy_max_tokens = True
    switch.request(loop, "claude-sonnet-5", env={})
    loop.say("go")
    loop.run()
    assert loop.provider is client
    assert client.lean is False and client.legacy_max_tokens is False


def test_a_same_provider_switch_takes_the_new_models_own_window(loop, models):
    loop.context_tokens = 600_000  # fits 1M, not Haiku's 200k
    said = "\n".join(switch.request(loop, "claude-haiku-4-5", env={}))
    assert loop.pending_model.context_window == 200_000
    assert loop.refresh_due and "200,000" in said


def test_leaving_a_model_with_its_own_window_goes_back_to_the_providers(loop, models):
    loop.cfg.model, loop.cfg.context_window = "claude-haiku-4-5", 200_000
    switch.request(loop, "claude-opus-5", env={})
    assert loop.pending_model.context_window == 1_000_000
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    assert loop.pending_model.context_window == 400_000


def test_reynolds_offers_only_the_models_the_service_meters():
    from openreynolds.llm.presets import models_for

    assert models_for("reynolds") == ("claude-sonnet-5", "claude-opus-5")
    assert "claude-haiku-4-5" in models_for("anthropic")
    assert models_for("openai") == ("gpt-5", "gpt-5-mini")


def test_the_existing_refresh_check_sees_the_new_window(loop, models):
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    loop.say("go")
    loop.run()
    loop.context_tokens = 350_000  # fine at 1M, over 80% of 400k
    assert loop.needs_refresh


# -- effort ----------------------------------------------------------------------


def test_effort_applies_from_the_next_request(loop, seen):
    assert "low" in "\n".join(switch.effort(loop, "LOW"))
    assert loop.cfg.effort == "low"
    assert seen.models[-1] == ("claude-opus-5", "low", "anthropic")
    loop.say("go")
    loop.run()
    assert loop.provider.calls[-1]["effort"] == "low"


def test_an_unknown_effort_is_refused(loop):
    said = "\n".join(switch.effort(loop, "ludicrous"))
    assert "one of low, medium, high" in said
    assert loop.cfg.effort == "high"


# -- the command line and the other agents -----------------------------------------


def test_model_and_effort_are_answered_locally(loop, store, seen, models):
    from openreynolds.cli import _local

    _local(commands.parse("/effort medium"), seen, None, store, loop)
    assert loop.cfg.effort == "medium"
    _local(commands.parse("/model"), seen, None, store, loop)
    assert "claude-opus-5" in seen.statuses[-1][0]
    _local(commands.parse("/model"), seen, None, store, None)
    assert "outside a session" in seen.statuses[-1][0]


def test_the_desk_follows_a_provider_change(store, view, monkeypatch):
    from openreynolds import desk as desk_mod

    cfg = Config(llm_api_key="k", desk_model="claude-haiku-4-5")
    desk = desk_mod.Concierge(cfg, store, view)
    built = []

    def make(c, **kw):
        built.append(c.provider)
        return SimpleNamespace(complete=lambda **k: f"answer from {k['model']}", client=None)

    monkeypatch.setattr(desk_mod, "make_provider", make)
    desk._call("s", "p", 10)
    assert built == []  # nothing moved, nothing rebuilt
    cfg.provider, cfg.llm_api_key, cfg.desk_model = "openai", "sk-o", "gpt-5-mini"
    assert desk._call("s", "p", 10) == "answer from gpt-5-mini"
    assert built == ["openai"]


def test_the_mesher_reads_the_model_at_each_run(store, monkeypatch):
    from openreynolds.mesher import agent

    cfg = Config(llm_api_key="k", model="claude-opus-5")
    desk = agent.Mesher(cfg, SimpleNamespace(workspace_root="/work"), store, "/work")
    kept = desk.provider
    cfg.model = "claude-sonnet-5"
    desk._follow()
    assert desk.model == "claude-sonnet-5" and desk.provider is kept

    monkeypatch.setattr(agent, "make_provider", lambda c: "rebuilt")
    cfg.provider, cfg.llm_api_key, cfg.model = "openai", "sk-o", "gpt-5"
    desk._follow()
    assert desk.provider == "rebuilt" and desk.model == "gpt-5"
    cfg.mesher_model = "gpt-5-mini"
    desk._follow()
    assert desk.model == "gpt-5-mini"
