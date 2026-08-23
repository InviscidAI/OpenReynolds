"""The free-will contract, made executable.

The harness may not inject step-by-step instructions, checklists, or mandated workflows.
The system prompt is where that would leak in first, so it gets checked rather than
reviewed by memory.
"""

from __future__ import annotations

import re

import pytest

from openreynolds.prompt import SYSTEM_PROMPT, system_prompt

IMPERATIVE_PATTERNS = [
    r"\balways\b",
    r"\bnever forget\b",
    r"\byou must\b",
    r"\byou should\b",
    r"\bmake sure\b",
    r"\bbe sure to\b",
    r"\bbefore you\b",
    r"\bstep \d",
    r"\bfirst,",
    r"\bthen,",
    r"\bdo not proceed\b",
    r"\brequired to\b",
    r"\byou are expected to\b",
    r"\bworkflow\b",
    r"\bchecklist:",
    r"\bphase \d",
]


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_prompt_mandates_no_workflow(pattern):
    match = re.search(pattern, SYSTEM_PROMPT, re.IGNORECASE)
    assert match is None, f"imperative workflow language in the system prompt: {match!r}"


def test_prompt_is_short():
    """Roughly one page. A long prompt is where procedure accumulates."""
    assert len(SYSTEM_PROMPT) < 6000


def test_prompt_is_frozen():
    """It sits at the front of the cached prefix, so it must not vary per session."""
    assert system_prompt() == system_prompt() == SYSTEM_PROMPT
    for volatile in ("study", "instance_id", "{", "}"):
        assert volatile not in SYSTEM_PROMPT.replace("kOmegaSST", "")


def test_prompt_states_the_environment_facts_the_model_needs():
    for fact in ("v2512", "pyvista", "24 hours", "sandbox_expired", "latestTime", "/work"):
        assert fact in SYSTEM_PROMPT


def test_prompt_says_where_honesty_is_expected():
    assert "did not verify" in SYSTEM_PROMPT
