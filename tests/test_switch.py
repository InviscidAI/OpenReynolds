"""Changing model, provider and effort mid-study (`switch.py`), with fake providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openreynolds import cli, commands, switch
from openreynolds.backend.base import BackendError
from openreynolds.config import Config
from openreynolds.llm import ProviderError, TextBlock, Turn
from openreynolds.loop import Loop
from openreynolds.store import Store
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


def test_the_model_asked_for_survives_the_presets_default(loop, models):
    """`candidate` rebuilds the configuration with `replace`, which re-runs
    `__post_init__` and its preset swap. On `reynolds` that turned the Opus a person
    typed -- one of the two models that service meters -- straight back into Sonnet,
    on `/model` and on every resume that went through here."""
    cfg = Config(provider="reynolds", foamd_api_key="svc")
    assert cfg.model == "claude-sonnet-5"  # nobody named one, so the preset's
    assert switch.candidate(cfg, "reynolds", "claude-opus-5", "").model == "claude-opus-5"


def test_a_same_provider_switch_leaves_the_desk_model_alone(loop, models):
    """Nothing about the desk was asked for. The swap would have moved it anyway, on a
    gateway configured as `openai` and serving Anthropic ids."""
    cfg = Config(provider="openai", llm_api_key="k", llm_base_url="https://gateway.example/v1")
    # The pair `Config.load` leaves on a gateway that serves Anthropic ids.
    cfg.model, cfg.desk_model = "claude-opus-5", "claude-haiku-4-5"
    new = switch.candidate(cfg, "openai", "claude-sonnet-5", "k")
    assert (new.model, new.desk_model) == ("claude-sonnet-5", "claude-haiku-4-5")


def test_an_endpoint_stands_in_for_a_key_on_the_provider_in_use():
    """`missing()` calls an endpoint with no key a legal configuration -- a gateway
    that authenticates some other way -- and the session is running on one. Answering
    None here refused a `/model`, or a resume, onto the provider already in use."""
    cfg = Config(provider="openai", llm_api_key="", llm_base_url="https://gateway.example/v1")
    assert switch.key_for("openai", cfg, env={}) == ""
    # Another provider is another matter: this endpoint says nothing about that one.
    assert switch.key_for("anthropic", cfg, env={}) is None


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


def test_a_switch_records_the_provider_beside_the_model(loop, models, store):
    """A resume reads the two together, and a cross-provider switch is exactly the
    case where the model alone would be read on the wrong provider."""
    switch.request(loop, "openai:gpt-5", env={"OPENAI_API_KEY": "sk-o"})
    loop.say("carry on")
    loop.run()
    assert (store.session.provider, store.session.model) == ("openai", "gpt-5")


def test_a_switch_records_the_endpoint_the_pair_was_served_from(loop, models, store):
    """A provider name is not an endpoint: a gateway or router in front of one family
    lists other vendors' ids, so a resume has to know which of the two served this
    pair before it restores it onto a key."""
    loop.cfg.llm_base_url = "https://gateway.example/v1"
    switch.request(loop, "claude-sonnet-5", env={})
    loop.say("carry on")
    loop.run()
    assert store.session.base_url == "https://gateway.example/v1"


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


# -- resuming --------------------------------------------------------------------


@pytest.fixture
def nobody_named_a_model(monkeypatch):
    """This run was told no model: no `--model`, and nothing in the environment.

    `OPENAI_API_KEY` goes too, because whether a stored provider can be served here is
    read from the real environment and a developer who has one exported is not a
    different test."""
    for name in ("OPENREYNOLDS_MODEL", "OPENREYNOLDS_MODE", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _stored_study(tmp_path, model, provider, base_url=""):
    studies = tmp_path / "studies"
    store = Store(studies, "study-x")
    store.session.model = model
    store.session.provider = provider
    store.session.base_url = base_url
    store.save()
    return studies


def _resume(monkeypatch, studies, config=None, **kwargs):
    """Resume `study-x` against a service that is not there.

    The model is settled before an instance is acquired, so the configuration the
    session gave up on is the whole of the answer. `config` is for the cases that turn
    on where this run points, which is a fact about the configuration."""
    def acquire(url, key, iid):
        raise BackendError("no service in this test")

    monkeypatch.setattr(cli.hosted, "acquire", acquire)
    cfg = config or Config(foamd_url="u", foamd_api_key="k", llm_api_key="a",
                           studies_dir=studies)
    with pytest.raises(SystemExit):
        cli.session(cfg, study_id="study-x", instance_id=None, one_shot=None, **kwargs)
    return cfg


def said_on_the_console(capsys):
    """One sentence, however the console decided to wrap it."""
    return " ".join(capsys.readouterr().out.split())


def test_a_resumed_study_carries_on_on_the_model_it_was_last_running(
    nobody_named_a_model, monkeypatch, tmp_path
):
    """`/model` left it on Sonnet; the configured default is still Opus."""
    studies = _stored_study(tmp_path, "claude-sonnet-5", "anthropic")
    cfg = _resume(monkeypatch, studies)
    assert (cfg.provider, cfg.model) == ("anthropic", "claude-sonnet-5")


def test_an_explicit_model_wins_over_the_stored_one(nobody_named_a_model, monkeypatch, tmp_path):
    studies = _stored_study(tmp_path, "claude-sonnet-5", "anthropic")
    cfg = _resume(monkeypatch, studies, model_explicit=True)
    assert cfg.model == "claude-opus-5"  # what the configuration says, not the study


def test_the_environment_model_counts_as_explicit_for_an_embedder(
    nobody_named_a_model, monkeypatch, tmp_path
):
    """The hosted runner sets OPENREYNOLDS_MODEL on every session it starts and calls
    `session` without the flag. That model is the one the app chose and the one its
    ledger row and model chooser show, so the agent must not quietly run another."""
    monkeypatch.setenv("OPENREYNOLDS_MODEL", "claude-opus-5")
    studies = _stored_study(tmp_path, "claude-sonnet-5", "anthropic")
    cfg = _resume(monkeypatch, studies)
    assert cfg.model == "claude-opus-5"


def test_a_stored_provider_with_no_key_here_falls_back_and_says_why(
    nobody_named_a_model, monkeypatch, tmp_path, capsys
):
    studies = _stored_study(tmp_path, "gpt-5", "openai")
    cfg = _resume(monkeypatch, studies)
    assert (cfg.provider, cfg.model) == ("anthropic", "claude-opus-5")
    said = said_on_the_console(capsys)
    assert "gpt-5" in said and "no key for openai" in said


def test_a_cross_provider_restore_brings_the_key_endpoint_and_window_with_it(
    nobody_named_a_model, monkeypatch, tmp_path
):
    """Never `cfg.model` alone: that would be Anthropic being asked for gpt-5."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    studies = _stored_study(tmp_path, "gpt-5", "openai")
    cfg = _resume(monkeypatch, studies)
    assert (cfg.provider, cfg.model, cfg.llm_api_key) == ("openai", "gpt-5", "sk-o")
    assert cfg.llm_base_url is None  # the preset's endpoint, not Anthropic's
    assert cfg.context_window == 400_000
    assert cfg.desk_model == "gpt-5-mini"


def test_a_study_recorded_before_the_provider_was_kept_starts_as_it_always_did(
    nobody_named_a_model, monkeypatch, tmp_path, capsys
):
    """A model id does not name a provider, so there is nothing to restore -- and
    nothing to explain either."""
    studies = _stored_study(tmp_path, "claude-sonnet-5", "")
    cfg = _resume(monkeypatch, studies)
    assert cfg.model == "claude-opus-5"
    assert "claude-sonnet-5" not in said_on_the_console(capsys)


def test_a_new_study_records_the_pair_it_started_on(nobody_named_a_model, monkeypatch, tmp_path):
    """There is nothing to restore on a first run, and something to write down: the
    pair the next resume reads. The model is taken out (`FakeLoop`), because what is
    under test is what the session put in `session.json` before it handed over."""
    from conftest import FakeBackend
    from test_jsonview import FakeLoop

    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (FakeBackend(), None, "iid-1"))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    studies = tmp_path / "studies"
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="a",
                 model="claude-sonnet-5", studies_dir=studies, capture=False,
                 desk=False, mesh_tool=False, mirror_interval_s=0.0)

    cli.session(cfg, study_id=None, instance_id=None, one_shot="go", plain=True)

    (session,) = [Store(studies, d.name).session for d in studies.iterdir()]
    assert (session.provider, session.model) == ("anthropic", "claude-sonnet-5")
    assert session.base_url == ""  # this provider's own endpoint


def test_a_pair_from_another_endpoint_on_the_same_provider_is_refused(
    nobody_named_a_model, monkeypatch, tmp_path, capsys
):
    """A provider name is not an endpoint. Two bring-your-own keys of one family -- a
    vendor's own and a router in front of it -- list different model ids, so a router's
    `anthropic/claude-sonnet-4.5` restored onto a direct key is accepted here and
    refused by the vendor mid-turn with a 400 about a model that does not exist."""
    studies = _stored_study(tmp_path, "anthropic/claude-sonnet-4.5", "openai",
                            base_url="https://openrouter.ai/api/v1")
    cfg = _resume(monkeypatch, studies, config=Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", provider="openai",
        studies_dir=studies))
    assert (cfg.provider, cfg.model) == ("openai", "gpt-5")
    said = said_on_the_console(capsys)
    assert "https://openrouter.ai/api/v1" in said and "falls back to gpt-5" in said


def test_a_pair_recorded_at_this_endpoint_is_restored_onto_it(
    nobody_named_a_model, monkeypatch, tmp_path
):
    """Like for like: the same gateway it was last on, so it carries on there -- on the
    id that was recorded, which the preset swap used to put back to `gpt-5`. The
    configured URL is what the record is compared against, never the preset's: a
    provider left on its preset's endpoint records no URL, and comparing that with the
    preset's would refuse every one of its own resumes."""
    studies = _stored_study(tmp_path, "claude-opus-5", "openai",
                            base_url="https://gateway.example/v1")
    cfg = _resume(monkeypatch, studies, config=Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", provider="openai",
        llm_base_url="https://gateway.example/v1", studies_dir=studies))
    assert (cfg.provider, cfg.model) == ("openai", "claude-opus-5")


def test_a_study_that_recorded_no_endpoint_is_refused_rather_than_guessed_at(
    nobody_named_a_model, monkeypatch, tmp_path, capsys
):
    """Recorded before the endpoint was kept and resumed against a gateway: nothing
    says the id came from there."""
    studies = _stored_study(tmp_path, "claude-opus-5", "openai")
    cfg = _resume(monkeypatch, studies, config=Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="a", provider="openai",
        llm_base_url="https://gateway.example/v1", studies_dir=studies))
    assert cfg.model == "gpt-5"
    assert "did not record" in said_on_the_console(capsys)


def test_a_refused_restore_leaves_the_study_the_pair_it_recorded(
    nobody_named_a_model, monkeypatch, tmp_path
):
    """The run that cannot serve what a study remembers is the last one that should be
    allowed to forget it. This run wrote its own pair over the study's a few lines
    later, so the resume on the machine that does have the key found nothing left to
    carry on with."""
    from conftest import FakeBackend
    from test_jsonview import FakeLoop

    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (FakeBackend(), None, "iid-1"))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    studies = _stored_study(tmp_path, "gpt-5", "openai")
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="a", studies_dir=studies,
                 capture=False, desk=False, mesh_tool=False, mirror_interval_s=0.0)

    cli.session(cfg, study_id="study-x", instance_id=None, one_shot="go", plain=True)

    assert cfg.model == "claude-opus-5"  # this run fell back, having no key for openai
    session = Store(studies, "study-x").session
    assert (session.provider, session.model) == ("openai", "gpt-5")


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
