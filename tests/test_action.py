"""The GitHub Action: packaging, and the free-will contract held across it.

`action.yml` is read as text rather than parsed as YAML, on purpose twice over. PyYAML
is not a dependency of this package and a test that imports it would pass here and skip
in CI; and what these assertions are about is the exact command line the workflow runs,
which is a string in a block scalar either way.

The contract being held is `docs/design.md` section 1: the harness may not "inject
step-by-step instructions, checklists, or mandated workflows -- in the system prompt, in
wake messages, or in any file the agent is required to obey". A CI wrapper is the
obvious place for that to leak in, one helpful line at a time, and nothing else in the
suite is looking at this file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ACTION = Path(__file__).resolve().parents[1] / "action.yml"


@pytest.fixture(scope="module")
def text() -> str:
    return ACTION.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def run_block(text: str) -> str:
    """The step that runs the study, from its `run: |` to the end of the file. What the
    agent is actually invoked with lives here and nowhere else."""
    start = text.index("    - name: Run the study")
    return text[start:]


@pytest.fixture(scope="module")
def commands(run_block: str) -> list[str]:
    """The run step's lines with the comments taken out.

    The comments in this file quote the contract they are keeping, so they contain the
    very words a test for a leaked workflow looks for. What a leak would change is the
    command, so that is what is read."""
    return [line.strip() for line in run_block.splitlines()
            if line.strip() and not line.strip().startswith("#")]


# -- the free-will contract --------------------------------------------------------


def test_the_task_reaches_the_agent_as_itself(run_block: str):
    command = re.search(r"openreynolds -p [^\n]*", run_block)
    assert command, "no `openreynolds -p` line in the run step"
    assert command.group(0).startswith('openreynolds -p "$OR_TASK"')


def test_nothing_is_wrapped_around_the_task(commands: list[str]):
    """A preamble, a suffix, a "remember to" -- each is one line of YAML away and each
    would be the harness telling the agent how to work."""
    used = [line for line in commands if "$OR_TASK" in line and not line.startswith("OR_TASK:")]
    assert used, "the run step never passes the task"
    for line in used:
        assert '-p "$OR_TASK"' in line, f"the task is not passed whole: {line}"
        assert line.split('-p "$OR_TASK"')[0].strip() == "openreynolds",             f"something precedes the task: {line}"


FORBIDDEN = ["checklist", "step by step", "step-by-step", "you must", "make sure",
             "remember to", "first validate", "before you", "do not forget"]
"""The words a well-meaning CI wrapper reaches for. If one of these ever belongs in a
command here, it belongs there with the contract tests changed alongside it rather than
worked around (`docs/design.md` section 1)."""


def test_the_action_mandates_no_workflow(commands: list[str]):
    lowered = " ".join(commands).lower()
    for phrase in FORBIDDEN:
        assert phrase not in lowered, f"the run step says {phrase!r}"


def test_no_input_default_carries_a_workflow_either(text: str):
    """The other way prose reaches the agent: a `default:` on an input that is then
    appended to the command line."""
    defaults = " ".join(re.findall(r"^    default: (.*)$", text, re.M)).lower()
    for phrase in FORBIDDEN:
        assert phrase not in defaults, f"an input default says {phrase!r}"


def test_no_mode_is_imposed_on_the_run(run_block: str):
    """`auto` is the default mode and the one in which no tool call is ever held
    (`docs/design.md` section 1, `openreynolds/modes.py`). Choosing `structured` on the
    user's behalf in CI would gate compute nobody asked to have gated -- so the mode
    travels through `args` if the workflow wants one, and is not set here."""
    assert "--mode" not in run_block


def test_no_approval_is_required_by_the_action(run_block: str):
    assert "--approve" not in run_block and "approval" not in run_block.lower()


# -- injection ---------------------------------------------------------------------


def test_the_task_is_not_interpolated_into_the_shell(text: str):
    """A `with:` value is attacker-controlled on a `pull_request_target` workflow, so
    `-p "${{ inputs.task }}"` is a shell injection. It travels in the environment."""
    assert "OR_TASK: ${{ inputs.task }}" in text
    for template in ("inputs.task", "inputs.args", "inputs.webhook-url"):
        uses = [line for line in text.splitlines()
                if f"${{{{ {template} }}}}" in line]
        assert uses, f"{template} is declared and never used"
        for line in uses:
            assert re.match(r"\s+[A-Z_]+: ", line), f"{template} is used in a command: {line}"


def test_every_secret_input_is_passed_as_an_environment_variable(text: str):
    for name, env in (("api-key", "OPENREYNOLDS_LLM_API_KEY"),
                      ("service-key", "FOAMD_API_KEY"),
                      ("webhook-secret", "OPENREYNOLDS_WEBHOOK_SECRET")):
        assert f"{env}: ${{{{ inputs.{name} }}}}" in text


# -- what it runs ------------------------------------------------------------------


def test_it_installs_the_version_the_workflow_pinned(text: str):
    """`$GITHUB_ACTION_PATH` is this repository at the ref the `uses:` line names.
    Installing a version from PyPI instead lets the agent and the action drift apart
    silently."""
    assert 'pip install --quiet "$GITHUB_ACTION_PATH"' in text
    assert "pip install openreynolds==" not in text


def test_it_reads_the_outcome_off_the_documented_stream(run_block: str):
    """`--output-format stream-json` already emits one object per line and a single
    `session_end` carrying the outcome (`openreynolds/jsonview.py:160-169`). The action
    reads that rather than scraping the terminal."""
    assert "--output-format stream-json" in run_block
    assert 'event.get("type") == "session_end"' in run_block
    assert "study-id=" in run_block and "outcome=" in run_block


def test_it_exits_with_the_study_s_own_status(run_block: str):
    """`ONE_SHOT_EXIT_CODES` maps the outcome to an exit code
    (`openreynolds/cli.py:2985`). Through a pipe that status is `PIPESTATUS[0]`, and
    taking the pipeline's last status instead would report the webhook relay's health
    as the study's."""
    assert "status=${PIPESTATUS[0]}" in run_block
    assert 'exit "$status"' in run_block


def test_the_webhook_is_sent_by_the_runner_and_not_by_the_workspace(run_block: str):
    """The workspace has no outbound network at all: `pip` and `apt` inside the image
    are symlinked to a refusal (`docker/workspace/install-guard.sh`). So a notification
    is this repository's own `webhook` subcommand, in this job."""
    assert "openreynolds webhook" in run_block
    assert "curl" not in run_block


def test_without_a_webhook_url_the_stream_still_goes_through(run_block: str):
    """`cat` as the relay rather than a second branch of the pipeline: one code path
    for the run means the no-webhook case cannot be the untested one."""
    assert "relay=(cat)" in run_block


def test_the_outputs_it_documents_are_the_outputs_it_sets(text: str):
    declared = set(re.findall(r"^  ([a-z-]+):\n    description:", text, re.M))
    head = text[:text.index("runs:")]
    outputs = head[head.index("outputs:"):]
    assert {"study-id", "outcome", "events-file"} <= set(
        re.findall(r"^  ([a-z-]+):", outputs, re.M))
    assert declared, "the inputs block did not parse the way this test reads it"


def test_the_action_declares_what_github_needs_to_show_it(text: str):
    for key in ("name:", "description:", "runs:", "using: composite"):
        assert key in text
