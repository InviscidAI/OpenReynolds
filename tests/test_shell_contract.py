"""What the tool descriptions say about the shell, and whether the shell does it.

`openreynolds/tools.py` tells the model that every call gets a fresh shell, that `cwd`
is the one thing that carries, and that the OpenFOAM environment and the tutorials'
`RunFunctions` helpers are loaded again on every call in both `bash` and `job_start`.

The first two clauses are properties of this repo. The third is a claim about the
service, and this file refuses to take it on trust: it reads the service's own wrappers
and fails if they do not do what the description promises. That coupling is the point.
A description asserting something the environment does not do is the same class of bug
as a listing that does not say it was cut short -- the harness stating a fact that is
not one, and the model paying for it downstream. When issue 11 was live, the cost of
this exact gap was a `127` that read as a missing binary and sent diagnosis at the
image.

The service lives in its own repository. Where it is checked out is configurable
(`FOAMD_REPO`), defaults to a sibling of this one, and when it is nowhere to be found
these checks skip -- a missing checkout is not evidence either way. What they will not
do is pass because the file was not read.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from openreynolds.prompt import SYSTEM_PROMPT
from openreynolds.tools import FRESH_SHELL, TOOLS
from test_prompt import IMPERATIVE_PATTERNS

RUNS_A_SHELL = ("bash", "job_start")
"""The two tools that start one, and so the two that owe the contract."""


def description(name: str) -> str:
    return next(tool["description"] for tool in TOOLS if tool["name"] == name)


# -- the sentence is where it belongs, and it is one sentence in two places ----


@pytest.mark.parametrize("name", RUNS_A_SHELL)
def test_both_tools_that_start_a_shell_carry_the_contract(name):
    """They are read independently. A model looking only at `job_start` must still
    learn that its shell is fresh."""
    assert FRESH_SHELL in description(name)


@pytest.mark.parametrize("tool", [t for t in TOOLS if t["name"] not in RUNS_A_SHELL])
def test_no_other_tool_repeats_it(tool):
    """Said once per shell-starting tool and nowhere else. A fact stated in several
    wordings is a fact that will drift into several different facts."""
    for fragment in ("fresh shell", "RunFunctions", "carries between calls"):
        assert fragment not in tool["description"]


def test_it_is_the_same_words_in_both():
    """Not two paraphrases. One constant, so a correction lands in both at once."""
    first, second = (description(name) for name in RUNS_A_SHELL)
    assert FRESH_SHELL in first and FRESH_SHELL in second


# -- the register: a fact about the environment, and nothing else -------------


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_it_issues_no_instruction(pattern):
    """The same guard the system prompt gets. `docs/design.md` §1 forbids the harness
    telling the model how to work; stating what the environment does is allowed and is
    all this does."""
    match = re.search(pattern, FRESH_SHELL, re.IGNORECASE)
    assert match is None, f"instruction in a tool description: {match!r}"


@pytest.mark.parametrize(
    "verb",
    ["remember to", "so source", "you will need to", "re-source", "keep in mind"],
)
def test_it_does_not_tell_the_model_what_to_do_about_it(verb):
    """The whole point of §5's fix is that the model is not made responsible for a
    quirk of our plumbing. Stating the property and then prescribing a habit for it
    would put the responsibility straight back."""
    assert verb not in FRESH_SHELL.lower()


def test_it_names_no_toolbox_script():
    """`RunFunctions` is OpenFOAM's own, shipped with the tutorials. Nothing of ours
    belongs in a tool description -- the toolbox is offered, not assumed."""
    toolbox = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
    for script in toolbox.glob("*.py"):
        assert script.name not in FRESH_SHELL
        assert script.stem not in FRESH_SHELL
    assert ".toolbox" not in FRESH_SHELL


def test_it_is_short():
    """Two sentences. It rides in the cached prefix of every request, forever."""
    assert len(FRESH_SHELL) < 400


# -- it must not contradict, or duplicate, anything already said --------------


def test_it_does_not_restate_what_the_system_prompt_already_says():
    """The prompt says the environment is sourced, under "What is installed". What is
    new here is the per-call part, and the helpers. Saying the rest again in different
    words is how two descriptions of one fact start disagreeing."""
    assert "The environment is sourced for you" in SYSTEM_PROMPT
    assert "sourced" not in FRESH_SHELL
    assert "on `PATH`" not in FRESH_SHELL


def test_the_bash_description_no_longer_says_it_twice():
    """`bash` used to carry "The OpenFOAM environment is sourced." on its own. The new
    sentence subsumes it, and both together would be the duplication this guards."""
    assert "The OpenFOAM environment is sourced." not in description("bash")
    assert description("bash").count("OpenFOAM environment") == 1


def test_nothing_elsewhere_promises_a_shell_that_remembers():
    """The contradiction that would matter most: something telling the model state
    carries. If one ever appears, this is where it is caught."""
    said = [SYSTEM_PROMPT] + [tool["description"] for tool in TOOLS]
    for text in said:
        for claim in ("same shell", "shell persists", "environment persists"):
            assert claim not in text.lower()


def test_the_one_thing_that_carries_is_the_one_the_tools_actually_take():
    """`cwd` is singled out as the exception. It is the exception precisely because
    both tools take it as an explicit parameter rather than inheriting it."""
    assert "`cwd` carries between calls" in FRESH_SHELL
    for name in RUNS_A_SHELL:
        schema = next(t for t in TOOLS if t["name"] == name)["input_schema"]
        assert "cwd" in schema["properties"]


# -- and the claim about the environment is checked, not trusted --------------


def foamd_source() -> str:
    """The service's sandbox wrappers, or a skip when the repo is not checked out."""
    root = os.environ.get("FOAMD_REPO")
    candidates = [Path(root)] if root else []
    candidates.append(Path(__file__).resolve().parents[2] / "OpenFoam_Instance")
    for candidate in candidates:
        path = candidate / "app" / "sandboxes.py"
        if path.is_file():
            return path.read_text(encoding="utf-8")
    pytest.skip(
        "the service repo is not checked out here; set FOAMD_REPO to check the claim "
        f"in the tool descriptions against it (looked in {[str(c) for c in candidates]})"
    )


def test_the_service_really_does_load_runfunctions():
    """The sentence promises the tutorials' helpers on every call. This fails if the
    wrapper that has to provide them does not."""
    source = foamd_source()

    assert "RunFunctions" in FRESH_SHELL
    assert "bin/tools/RunFunctions" in source, (
        "the tool descriptions tell the model that `RunFunctions` is loaded for it on "
        "every call. The service does not source it. The wording and the wrapper "
        "change land together or not at all (triage-decisions-2026-09-05 §5, §5a)."
    )


def test_it_loads_it_on_both_paths_not_one():
    """`bash` and `job_start` alike, says the sentence. Both wrappers must therefore
    build their script from the one constant that sources it."""
    source = foamd_source()

    shared = re.search(r"^_SOURCE_OF = \(", source, re.MULTILINE)
    assert shared, "the service no longer has one constant for what both paths source"

    exec_body = _body_of(source, "def run_exec(")
    job_body = _body_of(source, "def _wrapped_job_command(")
    assert "_SOURCE_OF" in exec_body, "the exec path stopped sourcing the shared block"
    assert "_SOURCE_OF" in job_body, (
        "the job path stopped sourcing the shared block, so `job_start` no longer has "
        "the helpers the description promises it has"
    )


def test_both_paths_are_the_same_kind_of_shell():
    """`bash -lc` on both. A login shell on one path and not the other is how the
    helpers were missing from jobs in the first place, and the description makes no
    distinction between the two tools."""
    source = foamd_source()

    assert '"bash", "-lc"' in source, "the exec path is no longer a login shell"
    assert "bash -lc" in _body_of(source, "def _wrapped_job_command("), (
        "the job path is not a login shell, so the two shells the description "
        "describes as one are two"
    )


def test_the_environment_the_prompt_promises_is_also_real():
    """The other half of the same claim, and the older one: solver names on `PATH`."""
    source = foamd_source()

    assert "openfoam" in source.lower()
    assert "bashrc" in source


def _body_of(source: str, signature: str) -> str:
    """One function's text, from its `def` to the next top-level one."""
    start = source.index("\n" + signature) + 1
    rest = source[start + len(signature):]
    end = re.search(r"\n(?:def |class |@)", rest)
    return rest[: end.start()] if end else rest
