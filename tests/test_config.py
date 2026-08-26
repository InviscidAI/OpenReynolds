"""Configuration: environment first, then a user file, and never the repository."""

from __future__ import annotations

import json
import os

import pytest

from openreynolds.config import DEFAULT_MAX_TOOL_OUTPUT, DEFAULT_MODEL, Config, config_path

ENV_KEYS = (
    "FOAMD_URL",
    "FOAMD_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "ZAI_API_KEY",
    "OPENREYNOLDS_PROVIDER",
    "OPENREYNOLDS_LLM_API_KEY",
    "OPENREYNOLDS_CONTEXT_WINDOW",
    "OPENREYNOLDS_MODEL",
    "OPENREYNOLDS_EFFORT",
    "OPENREYNOLDS_LLM_BASE_URL",
    "OPENREYNOLDS_MAX_TOOL_OUTPUT",
    "OPENREYNOLDS_MIRROR_INTERVAL_S",
    "OPENREYNOLDS_NARRATE_EVERY_S",
    "OPENREYNOLDS_CAPTURE",
    "OPENREYNOLDS_DESK",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "config.json"))
    return tmp_path / "config.json"


def write_config(path, **values):
    path.write_text(json.dumps(values), encoding="utf-8")


def test_defaults_when_nothing_is_set(clean_env):
    cfg = Config.load()
    assert cfg.model == DEFAULT_MODEL == "claude-opus-5"
    assert cfg.max_tool_output == DEFAULT_MAX_TOOL_OUTPUT
    assert cfg.llm_base_url is None
    assert cfg.capture is True


def test_the_file_supplies_values(clean_env, monkeypatch):
    write_config(clean_env, foamd_url="https://svc/", foamd_api_key="of_live_x", model="m")
    cfg = Config.load()
    assert cfg.foamd_url == "https://svc"
    assert cfg.foamd_api_key == "of_live_x"
    assert cfg.model == "m"


def test_the_environment_wins_over_the_file(clean_env, monkeypatch):
    write_config(clean_env, foamd_url="https://from-file", model="from-file")
    monkeypatch.setenv("FOAMD_URL", "https://from-env")
    monkeypatch.setenv("OPENREYNOLDS_MODEL", "from-env")

    cfg = Config.load()

    assert cfg.foamd_url == "https://from-env"
    assert cfg.model == "from-env"


def test_a_corrupt_config_file_is_ignored_rather_than_fatal(clean_env):
    clean_env.write_text("{ not json", encoding="utf-8")
    assert Config.load().model == DEFAULT_MODEL


def test_trailing_slash_is_trimmed_from_the_service_url(clean_env, monkeypatch):
    monkeypatch.setenv("FOAMD_URL", "https://svc.example/")
    assert Config.load().foamd_url == "https://svc.example"


def test_tool_output_budget_is_tunable(clean_env, monkeypatch):
    monkeypatch.setenv("OPENREYNOLDS_MAX_TOOL_OUTPUT", "1234")
    assert Config.load().max_tool_output == 1234


def test_narration_cadence_is_tunable(clean_env, monkeypatch):
    monkeypatch.setenv("OPENREYNOLDS_NARRATE_EVERY_S", "30")
    assert Config.load().narrate_every_s == 30.0


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " No "])
def test_capture_can_be_switched_off_from_the_environment(clean_env, monkeypatch, value):
    """`--no-capture` has to be remembered every time; an environment is set once,
    which is what the scheduled run nobody is watching needs."""
    monkeypatch.setenv("OPENREYNOLDS_CAPTURE", value)
    assert Config.load().capture is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "", "anything else"])
def test_anything_but_a_no_leaves_capture_on(clean_env, monkeypatch, value):
    monkeypatch.setenv("OPENREYNOLDS_CAPTURE", value)
    assert Config.load().capture is True


def test_a_preferences_file_beside_the_config_is_loaded(clean_env):
    (clean_env.parent / "preferences.md").write_text("render the mesh", encoding="utf-8")
    assert Config.load().preferences == "render the mesh"


def test_no_preferences_file_means_an_empty_note(clean_env):
    assert Config.load().preferences == ""


# -- what counts as configured -------------------------------------------------


def test_a_key_is_needed_for_each_of_compute_and_the_model(clean_env):
    assert Config.load().missing() == ["FOAMD_URL", "FOAMD_API_KEY", "ANTHROPIC_API_KEY"]


def test_fully_configured(clean_env):
    cfg = Config(foamd_url="https://svc", foamd_api_key="k", llm_api_key="a")
    assert cfg.missing() == []


def test_a_proxy_stands_in_for_the_model_key(clean_env):
    """If the service ever grows an LLM proxy, its own key covers the model too."""
    cfg = Config(foamd_url="https://svc", foamd_api_key="k", llm_base_url="https://svc/v1/llm")
    assert cfg.missing() == []


# -- saving --------------------------------------------------------------------


def test_save_writes_only_the_settings_that_have_values(clean_env):
    cfg = Config(foamd_url="https://svc", foamd_api_key="k", llm_api_key="a")
    path = cfg.save()

    saved = json.loads(path.read_text())
    assert saved == {
        "foamd_url": "https://svc",
        "foamd_api_key": "k",
        "provider": "anthropic",
        "llm_api_key": "a",
        "model": "claude-opus-5",
        "desk_model": "claude-haiku-4-5",
        "effort": "high",
        "context_window": 1_000_000,
    }
    assert "studies_dir" not in saved
    assert "capture" not in saved


def test_save_round_trips(clean_env):
    Config(foamd_url="https://svc", foamd_api_key="k", llm_api_key="a").save()
    reloaded = Config.load()
    assert (reloaded.foamd_url, reloaded.foamd_api_key) == ("https://svc", "k")


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_credentials_are_not_world_readable(clean_env):
    path = Config(foamd_api_key="secret").save()
    assert path.stat().st_mode & 0o077 == 0


def test_credentials_live_outside_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENREYNOLDS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    target = config_path()

    assert tmp_path not in target.parents, "config must not land in the repository"
    assert target.name == "config.json"
    assert target.parent.name == "openreynolds"


# -- bring your own model -------------------------------------------------------


def test_a_preset_fills_in_endpoint_model_and_window(clean_env):
    cfg = Config(provider="zai", llm_api_key="k")
    assert cfg.model == "glm-4.6"
    assert cfg.desk_model == "glm-4.5-air"
    assert cfg.context_window == 200_000
    assert cfg.llm_base_url is None, "the preset's endpoint is applied by the provider, not stored"


def test_explicit_choices_survive_a_preset(clean_env):
    cfg = Config(provider="openai", llm_api_key="k", model="gpt-5-pro", context_window=123)
    assert (cfg.model, cfg.context_window) == ("gpt-5-pro", 123)


def test_an_unknown_endpoint_gets_the_conservative_window(clean_env):
    cfg = Config(provider="openai", llm_api_key="k", llm_base_url="https://gateway.example/v1")
    assert cfg.context_window == 200_000


def test_the_provider_and_its_vendor_key_come_from_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv("OPENREYNOLDS_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "zk")
    cfg = Config.load()
    assert (cfg.provider, cfg.llm_api_key, cfg.model) == ("zai", "zk", "glm-4.6")


def test_the_general_key_name_beats_the_vendor_one(clean_env, monkeypatch):
    monkeypatch.setenv("OPENREYNOLDS_LLM_API_KEY", "general")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "vendor")
    assert Config.load().llm_api_key == "general"


def test_an_older_config_file_still_supplies_the_key(clean_env):
    config_path().write_text(json.dumps({"anthropic_api_key": "old"}), encoding="utf-8")
    cfg = Config.load()
    assert cfg.llm_api_key == "old" and cfg.provider == "anthropic"


def test_a_local_model_needs_no_key(clean_env):
    cfg = Config(foamd_url="https://svc", foamd_api_key="k", provider="ollama")
    assert cfg.missing() == []


def test_the_missing_key_is_named_in_the_vendors_words(clean_env):
    assert Config(foamd_url="u", foamd_api_key="k", provider="openai").missing() == ["OPENAI_API_KEY"]
    assert Config(foamd_url="u", foamd_api_key="k", provider="deepseek").missing() == ["DEEPSEEK_API_KEY"]
