"""Every model client a session builds names the study in `X-Study-Id`.

The service bills a model call to the study the header names and judges it against the
study's ceiling; a trial account's call with no study of its own is refused. Before this,
the main loop sent the header and the CAD desk, its reviewer and the `/model` probe did
not, so their calls landed on the account untagged: on the hosted service, most of the
untagged calls inside studies in the week to 2026-11-13.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from openreynolds.cad import CoreDesk, Reviewer
from openreynolds.config import Config
from openreynolds.mesher.agent import Mesher

PKG = Path(__file__).resolve().parent.parent / "openreynolds"
STUDY = "20261114-073000-ab12"
HEADERS = {"X-Study-Id": STUDY}


def _cfg(model: str) -> Config:
    return Config(provider="reynolds", foamd_url="https://foamd.test", foamd_api_key="of_live_k",
                  model=model)


def _sent(provider) -> dict:
    return dict(provider.client.default_headers)


BACKEND = SimpleNamespace(workspace_root="/work")
STORE = SimpleNamespace(session=SimpleNamespace(study_id=STUDY))


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "claude-opus-5-5"])
def test_the_desk_the_reviewer_and_the_mesher_name_the_study(model):
    cfg = _cfg(model)
    for agent in (Reviewer(cfg, BACKEND, headers=HEADERS),
                  CoreDesk(cfg, BACKEND, STORE, "/work", headers=HEADERS),
                  Mesher(cfg, BACKEND, STORE, "/work", headers=HEADERS)):
        assert _sent(agent.provider).get("X-Study-Id") == STUDY, type(agent).__name__


def test_a_client_rebuilt_for_another_model_still_names_the_study():
    cfg = _cfg("gpt-6.1-sol")
    desk = CoreDesk(cfg, BACKEND, STORE, "/work", headers=HEADERS)
    reviewer = Reviewer(cfg, BACKEND, headers=HEADERS)
    cfg.model = "claude-opus-5-5"      # a mid-study `/model` to the other family
    desk._follow()
    reviewer._follow()
    assert "anthropic" in type(desk.provider).__module__
    assert _sent(desk.provider).get("X-Study-Id") == STUDY
    assert _sent(reviewer.provider).get("X-Study-Id") == STUDY


# The one client built outside a study: `openreynolds doctor`'s key check.
OUTSIDE_A_STUDY = {("cli.py", "_check_model")}


def _calls_without_headers():
    found = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                if name != "make_provider" or fn.name == "make_provider":
                    continue
                if any(k.arg == "default_headers" for k in call.keywords):
                    continue
                found.append((str(path.relative_to(PKG)), fn.name))
    return sorted(set(found))


def test_no_model_client_is_built_without_the_study_header():
    """A new desk or probe that builds its own client must pass the session's headers,
    or its calls are untagged and, on a trial, refused."""
    assert [c for c in _calls_without_headers() if c not in OUTSIDE_A_STUDY] == []
