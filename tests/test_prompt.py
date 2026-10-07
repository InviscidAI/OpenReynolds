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
    for volatile in ("instance_id", "{", "}"):
        assert volatile not in SYSTEM_PROMPT.replace("kOmegaSST", "")


@pytest.mark.parametrize(
    "shape",
    [
        r"\d{8}-\d{6}-[0-9a-f]{4}",  # a study id
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}",  # an instance id
        r"\b20\d\d-\d\d-\d\d\b",  # a date
    ],
)
def test_nothing_that_changes_per_session_is_baked_in(shape):
    """The word "study" is fine; a study id is not. Anything that varies between
    sessions invalidates the whole cached prefix, every session, for everyone."""
    assert re.search(shape, SYSTEM_PROMPT) is None


def test_prompt_states_the_environment_facts_the_model_needs():
    for fact in ("v2512", "pyvista", "24 hours", "sandbox_expired", "latestTime", "/work"):
        assert fact in SYSTEM_PROMPT


def test_prompt_says_where_honesty_is_expected():
    assert "did not verify" in SYSTEM_PROMPT


def test_honesty_is_a_fact_about_the_number_not_a_verdict_on_the_run():
    """The sentence used to lead with "an unconverged solve" as the first thing to
    confess to, and four studies in five answered it with a bold "Honesty: the run did
    not converge" about a residual levelled off on a shedding wake -- the physics, not
    a defect (`openreynolds/convergence.py` has them). The expectation stands; the
    example that framed a stall as a confession is gone."""
    assert "unconverged" not in SYSTEM_PROMPT
    assert "a fact about it and not a verdict on the run" in SYSTEM_PROMPT
    assert "a run still moving when its number was read" in SYSTEM_PROMPT


def test_prompt_does_not_promise_tools_the_image_lacks():
    """The A4 run wasted a detour on foamToC, which the prompt claimed was there."""
    assert "`foamToC` is available" not in SYSTEM_PROMPT
    assert "not** in this image" in SYSTEM_PROMPT or "not in this image" in SYSTEM_PROMPT


def test_prompt_says_mpi_is_already_arranged():
    """The container runs as root and has no outbound network, and both of those
    stop OpenMPI from launching unless the environment says otherwise. foamd now
    sets all three (config.SANDBOX_ENV_DEFAULTS), so the fact the model needs is
    that it does not have to arrange anything -- the prompt used to say the
    opposite, and a study spent five minutes proving the prompt wrong."""
    assert "OMPI_ALLOW_RUN_AS_ROOT" in SYSTEM_PROMPT
    assert "PMIX_MCA_gds" in SYSTEM_PROMPT
    assert "fails immediately" not in SYSTEM_PROMPT


def test_prompt_does_not_promise_a_core_count():
    """It said "8 cores" while the default shape was four. A number that changes
    with the instance does not belong in a prompt that is the same for every one."""
    assert "8 cores" not in SYSTEM_PROMPT
    assert "nproc" in SYSTEM_PROMPT


def test_the_prompt_says_a_study_has_a_directory_of_its_own():
    """A new project starting clean is an environmental fact, and the only way the
    model learns which directory is its own is by being told one exists."""
    assert "its own directory" in SYSTEM_PROMPT
    assert "briefing names" in SYSTEM_PROMPT
    assert "empty one" in SYSTEM_PROMPT


# -- Felix mode (docs/felix-mode-acceptance.md, A2) ------------------------------

OPENFOAM_PROMPT_SHA256 = "4c3406a11756d284c19b2c2708eed9578fe2286ef6826a5890eb120136d7b50b"
"""The OpenFOAM prompt as it was before Felix mode. It is the cache prefix of every
OpenFOAM study, and a second solver is no reason for it to move by one byte."""


def test_the_openfoam_prompt_is_byte_identical_to_before_felix():
    import hashlib

    assert hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() == OPENFOAM_PROMPT_SHA256
    assert system_prompt() == system_prompt("openfoam") == SYSTEM_PROMPT


def _felix():
    from openreynolds.prompt import FELIX_PROMPT

    return FELIX_PROMPT


def test_a_felix_study_gets_the_felix_prompt():
    assert system_prompt("felix") == _felix()
    assert _felix() != SYSTEM_PROMPT


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_the_felix_prompt_mandates_no_workflow(pattern):
    match = re.search(pattern, _felix(), re.IGNORECASE)
    assert match is None, f"imperative workflow language in the Felix prompt: {match!r}"


def test_the_felix_prompt_is_short():
    assert len(_felix()) < 6000


def test_the_felix_prompt_is_frozen():
    assert system_prompt("felix") is system_prompt("felix")
    for volatile in ("instance_id", "{", "}"):
        assert volatile not in _felix()


@pytest.mark.parametrize(
    "shape",
    [r"\d{8}-\d{6}-[0-9a-f]{4}", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", r"\b20\d\d-\d\d-\d\d\b"],
)
def test_nothing_per_session_is_baked_into_the_felix_prompt(shape):
    assert re.search(shape, _felix()) is None


@pytest.mark.parametrize(
    "fact",
    [
        "/opt/felix/docs",          # the reference, pointed at rather than copied
        "case.yaml",                # a case is a directory of YAML files
        ".vtu",                     # plus a tagged mesh
        "felix-tag-mesh",
        "felix-check-mesh",
        "felix gpus",               # GPU types and prices come from the service
        "gpu_sizing.md",
        "felix run",
        "--gpu",
        "--wall",
        "felix continue",
        "job_start",
        "output.every",             # every solve snapshots
        "ERROR:",
        "WARNING:",
        "NOTE:",
        "step=",
        "stdout",
        "stderr",
        "summary.json",
        "out of GPU memory",
        "/work",
        "did not verify",
    ],
)
def test_the_felix_prompt_states_what_the_model_needs(fact):
    assert fact in _felix()


def test_the_felix_prompt_states_the_exit_codes():
    text = _felix()
    for code in ("0", "1", "2"):
        assert re.search(rf"\b{code}\b", text)
    assert "refused" in text


def test_the_felix_prompt_carries_no_prices():
    """Prices change and live in the service (`felix gpus`); a number here is stale."""
    assert "$" not in _felix()
    assert not re.search(r"\d+(\.\d+)?\s*/\s*h(ou)?r", _felix())


def test_the_felix_prompt_says_nothing_of_openfoam_tooling():
    for openfoam in ("blockMesh", "controlDict", "decomposePar", "mpirun", "v2512",
                     "latestTime", "`cad`", "mesh_review", ".toolbox"):
        assert openfoam not in _felix(), openfoam
    assert "honesty" in _felix().lower() or "did not verify" in _felix()


# -- before the solver is chosen (`--solver auto`) ------------------------------


def _choosing():
    from openreynolds.prompt import SOLVER_CHOICE_PROMPT

    return SOLVER_CHOICE_PROMPT


def test_a_study_with_no_solver_yet_gets_the_choosing_prompt():
    assert system_prompt("auto") == _choosing()
    assert _choosing() not in (SYSTEM_PROMPT, _felix())


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_the_choosing_prompt_mandates_no_workflow(pattern):
    match = re.search(pattern, _choosing(), re.IGNORECASE)
    assert match is None, f"imperative workflow language in the choosing prompt: {match!r}"


def test_the_choosing_prompt_is_short_and_frozen():
    assert len(_choosing()) < 6000
    assert system_prompt("auto") is system_prompt("auto")
    for volatile in ("instance_id", "{", "}"):
        assert volatile not in _choosing()


@pytest.mark.parametrize(
    "fact",
    [
        "choose_solver",
        "OpenFOAM",
        "Felix",
        "incompressible",       # what Felix solves
        "Spalart-Allmaras",     # its one turbulence model
        "Boussinesq",           # its energy equation
        "compressible",         # what it does not
        "multiphase",
        ".vtu",                 # its mesh
        "GPU",
        "billed",
        "does not change",      # the choice holds for the study
    ],
)
def test_the_choosing_prompt_says_what_each_solver_can_do(fact):
    assert fact in _choosing()


def test_the_choosing_prompt_carries_no_prices():
    assert "$" not in _choosing()
