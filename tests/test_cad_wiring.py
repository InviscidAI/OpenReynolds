"""The desk is wired in, and nothing it makes stops travelling.

C9 is small edits in many files, which is exactly the shape of change that passes every
test it owns and breaks something nobody was looking at. The blast radius was costed
before it was cut (`docs/cad-plan-decisions-2026-09-10.md`, #18/#18a) and this file is
that list made executable: the mirror, the bundle, the tool schema, the environment
variables, the system prompt, the image manifest.

The one property under all of it: **a run's artifacts reach the user's machine and the
study's second copy**. A mesh nobody can get at is a mesh that was not built, and every
way of losing one here is silent -- a suffix missing from a set, a branch tested in the
wrong order, a filename nobody listed.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from openreynolds import casebundle, mirror
from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.browse import Browser
from openreynolds.cad import check as cad_check
from openreynolds.config import Config
from openreynolds.prompt import SYSTEM_PROMPT
from openreynolds.tools import TOOLS, dispatch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "openreynolds"
TOOLBOX = PACKAGE / "toolbox"
HOME = "/work/study-test"

HAS_GMSH = importlib.util.find_spec("gmsh") is not None
needs_gmsh = pytest.mark.skipif(not HAS_GMSH, reason="gmsh module not on this machine")


# -- the workspace, and what the mirror makes of it -----------------------------


def workspace(backend, rows: list[tuple[str, bytes]]) -> None:
    """A listing plus the bytes themselves, so sizes downstream are real sizes.

    The sibling helper in `test_mirror.py` truncates file contents to 64 bytes, which
    is right for tests about *which* files travel and useless for tests about how many
    bytes of them do. Every assertion below is a byte count.
    """
    lines = []
    for path, data in rows:
        backend.files[path] = data
        lines.append(f"-\t{len(data)}\t1700000000.0\t{path}")
    backend.exec_result = ExecResult(0, "\n".join(lines) + "\n", False, None)


def cad_case(patches: int = 20, patch_bytes: int = 110_000) -> list[tuple[str, bytes]]:
    """A case as the CAD desk leaves it: a rebuild script, a patch set, the source CAD.

    Random bytes on purpose. A megabyte of one repeated character gzips to nothing and
    a budget assertion against it would prove nothing; an STL is coordinates.
    """
    rows: list[tuple[str, bytes]] = [
        (f"{HOME}/cad/build.py", b"# the geometry, as a script\n" * 40),
        (f"{HOME}/cad/system/controlDict", b"application simpleFoam;\n"),
        (f"{HOME}/cad/system/snappyHexMeshDict", b"castellatedMesh true;\n" * 20),
        (f"{HOME}/cad/constant/triSurface/patches.json", b'{"patches": []}\n'),
        (f"{HOME}/cad/log.snappyHexMesh", b"Finished meshing\n" * 100),
        (f"{HOME}/chassis.step", os.urandom(300_000)),
    ]
    for index in range(patches):
        rows.append((f"{HOME}/cad/constant/triSurface/patch_{index:02d}.stl",
                     os.urandom(patch_bytes)))
    return rows


def tier_bytes(root: Path) -> dict[str, int]:
    """What each tier weighs on disk, by the bundle's own classifier."""
    latest = casebundle._latest_time_dirs(root)
    totals = {tier: 0 for tier in casebundle.TIERS}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        tier = casebundle.classify(path, root, latest)
        if tier is not None:
            totals[tier] += path.stat().st_size
    return totals


def names_in(blob: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        return set(tar.getnames())


def definition_assumption(tmp_path) -> int:
    """The size `tests/test_casebundle.py` actually assumes, measured rather than quoted.

    Its budget test builds a study with no CAD in it at all and asserts that
    `definition` and `record` both fit inside a **1 MB** bundle while the mesh is
    refused. The decisions doc says "about 4 MB a study" from production; the number
    this repo's tests hold itself to is the smaller one, so that is the one pinned
    here -- and it is read off the fixture rather than typed, so a fixture that grows
    moves this with it.
    """
    from test_casebundle import _study

    root = tmp_path / "plain"
    root.mkdir()
    _study(root)
    blob, report = casebundle.build(root, budget_mb=1, study_id="s0")
    assert blob is not None and "definition" in report.included
    assert "record" in report.included, (
        "test_casebundle's budget assumption has moved; re-read it before trusting "
        "the number this file compares against"
    )
    return tier_bytes(root)["definition"]


def test_the_cad_desks_artifacts_reach_the_bundle_and_land_in_the_right_tier(
    backend, store, tmp_path
):
    """End to end, in bytes: mirror pass, then `casebundle.build()`.

    Three separate ways this used to lose the work, all of them quiet:
    `mirror.KEEP_SUFFIXES` had no `.step` or `.stl`, so nothing arrived to bundle;
    `classify()` tested `constant/` before the geometry suffix, so a patch set landed
    in `definition` -- tier 0, which travels even when it is oversized -- and inflated
    exactly the tier the design guarantees will survive; and a rebuild script under an
    unlisted name fell through every branch and returned None.
    """
    rows = cad_case()
    workspace(backend, rows)
    report = mirror.sync(Browser(backend, store, home=HOME))

    arrived = {Path(path).name for path in report.pulled}
    assert "chassis.step" in arrived, f"the CAD source was skipped: {report.skipped}"
    assert "patch_00.stl" in arrived, f"the patch set was skipped: {report.skipped}"
    assert "build.py" in arrived

    root = store.files_dir
    totals = tier_bytes(root)
    stl_bytes = sum(len(data) for path, data in rows if path.endswith(".stl"))
    step_bytes = sum(len(data) for path, data in rows if path.endswith(".step"))
    assert stl_bytes > 2 * 1024 * 1024, "the fixture is not the case this test is about"

    assert totals["geometry"] == stl_bytes + step_bytes, (
        "the exported patch set and the source CAD belong to the tier that budgets "
        f"them; tier totals were {totals}"
    )
    assert totals["definition"] < definition_assumption(tmp_path) + 100_000, (
        "the definition tier has taken on the geometry it is not budgeted for"
    )

    blob, bundle = casebundle.build(root, study_id="s1")
    assert blob is not None, bundle.error
    names = names_in(blob)
    assert "study-test/chassis.step" in names
    assert "study-test/cad/build.py" in names
    assert "study-test/cad/constant/triSurface/patch_00.stl" in names
    assert "geometry" in bundle.included and "definition" in bundle.included


def test_the_definition_tier_still_fits_the_budget_test_casebundle_pins(backend, store):
    """The property the tier order exists for, asserted on a case with CAD in it.

    `test_casebundle.py` builds at `budget_mb=1` and requires `definition` and
    `record` to survive it. Twenty STLs are more than a megabyte on their own, so
    before the reorder this case could not have passed that assertion at all.
    """
    workspace(backend, cad_case())
    mirror.sync(Browser(backend, store, home=HOME))
    blob, report = casebundle.build(store.files_dir, budget_mb=1, study_id="s1")
    assert blob is not None and len(blob) <= 1024 * 1024
    assert report.included[:1] == ["definition"]
    assert "record" in report.included
    assert report.oversized is False, (
        "the irreplaceable tier went past the budget, which is what putting geometry "
        "in it looks like"
    )
    assert "geometry" in report.skipped
    assert "left out" in " ".join(report.brief()) and "geometry" in " ".join(report.brief())


# -- the size rule, which is the half a suffix fix does not cover ----------------


def test_a_five_megabyte_stl_under_constant_survives_the_mirror(backend, store):
    """`reason_by_size`, not `reason_to_skip`.

    Anything over 2 MB sitting in `system/`, `constant/` or a `0*` directory is field
    data by the dictionary rule -- `0/U` on a half-million-cell mesh is nine megabytes
    and turned a 43-file mirror into 42 MB of it. A per-patch STL is routinely bigger
    than that and is geometry, and it is kept for its suffix rather than for where it
    sits. A fix that only touched the first rule passes every other item in this file.
    """
    relative = "cad/constant/triSurface/hull.stl"
    size = 5 * 1024 * 1024
    assert size > mirror.DICTIONARY_BYTES
    assert mirror.reason_to_skip(relative) is None
    assert mirror.reason_by_size(relative, size) is None

    # The control: the same size, the same directory, and genuinely field data.
    assert mirror.reason_by_size("cad/constant/octree/points", size) is not None

    workspace(backend, [(f"{HOME}/{relative}", os.urandom(size))])
    report = mirror.sync(Browser(backend, store, home=HOME))
    assert not report.skipped, report.skipped
    landed = store.files_dir / "study-test" / relative
    assert landed.is_file() and landed.stat().st_size == size


# -- the rebuild script the bundle will take ------------------------------------


def test_the_script_we_leave_is_a_name_the_bundle_takes(backend, store):
    """The risk the old import guarded, closed at the other end.

    It used to be: a desk invents a name nobody listed, and the rebuild script silently
    does not travel with the case. The finish check closed it by refusing any run whose
    script the bundle would not take -- which meant the two name sets had to be the same
    object, and meant a desk could be failed for the harness's own omission, since
    nothing ever wrote the file the brief said it would.

    Now the harness writes it, under one name, and that name is the bundle's. There is no
    set to agree on because the desk no longer chooses.
    """
    assert cad_check.REPLAY_SCRIPT in casebundle.DEFINITION_NAMES
    assert not hasattr(cad_check, "DEFINITION_NAMES"), (
        "the check does not gate on the bundle's names any more")


def test_the_harness_written_replay_script_lands_under_a_captured_name():
    """The one filename the harness itself chooses, rather than the desk.

    `check.py` writes the accepted cell log out as a script and runs it from empty.
    If that name were not in the captured set, the check would refuse the very
    artifact the harness had just written.
    """
    assert cad_check.REPLAY_SCRIPT in casebundle.DEFINITION_NAMES


@pytest.mark.parametrize("name", sorted(casebundle.DEFINITION_NAMES))
def test_every_captured_name_is_captured_wherever_it_sits(tmp_path, name):
    case = tmp_path / "study" / "cad"
    case.mkdir(parents=True)
    (case / name).write_text("x\n")
    path = case / name
    assert casebundle.classify(path, tmp_path, set()) == "definition"


# -- the tool ------------------------------------------------------------------


def test_the_tool_names_are_exactly_the_sorted_eight_with_cad_second():
    names = [tool["name"] for tool in TOOLS]
    assert names == sorted(names)
    assert len(names) == 8
    assert names[:2] == ["bash", "cad"]
    assert "mesh" not in names


def test_the_cad_tool_takes_a_geometry_path():
    schema = next(t for t in TOOLS if t["name"] == "cad")["input_schema"]
    assert set(schema["properties"]) == {"request", "case", "geometry", "inputs"}
    assert schema["required"] == ["request"]
    described = schema["properties"]["geometry"]["description"]
    for suffix in (".step", ".stp", ".iges", ".igs"):
        assert suffix in described
    assert "/work" in described


def test_the_cad_tool_takes_any_number_of_other_files_to_work_from():
    """`geometry` is "this is the part"; `inputs` is "here is something to work from".

    The difference is what is being asked for, not what the file is -- which is why
    `inputs` carries no suffix list. A whitelist there is the harness deciding which
    kinds of work are possible, and it is how a floorplan drawing could not be handed
    over at all."""
    schema = next(t for t in TOOLS if t["name"] == "cad")["input_schema"]
    inputs = schema["properties"]["inputs"]
    assert inputs["type"] == "array" and inputs["items"]["type"] == "string"
    for suffix in (".step", ".stp", ".iges", ".igs"):
        assert suffix not in inputs["description"]


def test_an_input_is_not_held_to_the_cad_suffix_list(ctx, monkeypatch):
    """A PNG is refused as `geometry` and taken as an `input`, in the same session."""
    path = "/work/study-test/plan.png"
    ctx.backend.files[path] = b"\x89PNG\r\n"
    seen = {}

    class Desk:
        def run(self, request, case=None, geometry="", inputs=()):
            seen.update(geometry=geometry, inputs=list(inputs))
            return type("R", (), {"tokens": {}, "png": None, "check": None, "ok": False,
                                  "error": "", "case_rel": "cad", "summary": "",
                                  "steps": [], "seconds": 1.0, "stopped": "",
                                  "remarks": []})()

    ctx.cad = Desk()
    refused, _ = dispatch(ctx, "cad", {"request": "mesh this", "geometry": path})
    assert "not a CAD file this reads" in refused and "`inputs`" in refused
    assert seen == {}

    dispatch(ctx, "cad", {"request": "mesh this", "inputs": [path]})
    assert seen == {"geometry": "", "inputs": [path]}


def test_every_input_is_checked_before_anything_starts(ctx, desk):
    """The second of three being wrong still costs a stat, not a build."""
    good = "/work/study-test/plan.png"
    ctx.backend.files[good] = b"\x89PNG\r\n"
    answer, _ = dispatch(ctx, "cad", {
        "request": "mesh this",
        "inputs": [good, "/work/study-test/absent.csv", good]})
    assert answer.startswith("nothing was run:")
    assert "absent.csv" in answer
    assert desk.calls == 0 and ctx.backend.execs == []


def test_a_file_passed_both_ways_reaches_the_desk_once(ctx):
    path = "/work/study-test/chassis.step"
    ctx.backend.files[path] = b"ISO-10303-21;\n"
    seen = {}

    class Desk:
        def run(self, request, case=None, geometry="", inputs=()):
            seen.update(geometry=geometry, inputs=list(inputs))
            return type("R", (), {"tokens": {}, "png": None, "check": None, "ok": False,
                                  "error": "", "case_rel": "cad", "summary": "",
                                  "steps": [], "seconds": 1.0, "stopped": "",
                                  "remarks": []})()

    ctx.cad = Desk()
    dispatch(ctx, "cad", {"request": "prepare it", "geometry": path, "inputs": [path]})
    assert seen == {"geometry": path, "inputs": []}


class Recorder:
    """A model client that fails the test if anything asks it a question."""

    def __init__(self):
        self.calls = 0

    def __getattr__(self, name):
        def answer(*args, **kwargs):
            self.calls += 1
            raise AssertionError(f"the provider was called: {name}")
        return answer


@pytest.fixture
def desk(ctx, monkeypatch):
    """A real `CadDesk`, wired to a model client that must never be reached."""
    from openreynolds.cad import agent

    recorder = Recorder()
    monkeypatch.setattr(agent, "make_provider", lambda cfg: recorder)
    ctx.cad = agent.CadDesk(Config(), ctx.backend, ctx.store, HOME)
    return recorder


@pytest.mark.parametrize("path,wrong", [
    ("/work/study-test/absent.step", "could not be read"),
    ("/work/study-test/hull.stl", "not a CAD file this reads"),
    ("chassis.step", "not an absolute path"),
    ("/home/someone/chassis.step", "not under /work"),
    ("/work/study-test/cad", "not a CAD file this reads"),
    ("/work/study-test/empty.step", "empty"),
])
def test_a_geometry_path_that_is_wrong_is_refused_before_anything_starts(
    ctx, desk, path, wrong
):
    """"Refused early when unreadable" (#12) means not discovered on step nine.

    A run that starts costs a kernel, a thread and a model bill; a path with a typo in
    it is worth one sentence. So the assertion is not only that the answer says what
    is wrong -- it is that nothing was spent finding out.
    """
    ctx.backend.files["/work/study-test/hull.stl"] = b"solid\n"
    ctx.backend.files["/work/study-test/empty.step"] = b""
    ctx.backend.dirs["/work/study-test/cad"] = []

    answer, is_error = dispatch(ctx, "cad", {"request": "mesh it", "geometry": path})

    assert is_error is False
    assert path in answer and wrong in answer
    assert answer.startswith("nothing was run:")
    assert desk.calls == 0, "a refusal that cost a model call is not a refusal"
    assert ctx.backend.execs == [], "the workspace was touched by a refused run"


def test_an_unreadable_file_that_exists_is_refused_too(ctx, desk):
    """Existence is not readability: a file on a volume that has gone away stats and
    then will not open, which is the case this property is actually about."""
    path = "/work/study-test/chassis.step"
    ctx.backend.files[path] = b"ISO-10303-21;\n"

    def refuse(*args, **kwargs):
        raise BackendError("permission denied", code="denied", status=403)

    ctx.backend.get_file = refuse
    answer, _ = dispatch(ctx, "cad", {"request": "mesh it", "geometry": path})
    assert path in answer and "could not be opened" in answer
    assert desk.calls == 0 and ctx.backend.execs == []


def test_a_good_geometry_path_reaches_the_desk(ctx, monkeypatch):
    """The other half: the check refuses what is wrong and stays out of the way of
    what is right, geometry and all."""
    path = "/work/study-test/chassis.step"
    ctx.backend.files[path] = b"ISO-10303-21;\n"
    seen = {}

    class Desk:
        def run(self, request, case=None, geometry="", inputs=()):
            seen.update(request=request, case=case, geometry=geometry)
            return type("R", (), {"tokens": {}, "png": None, "check": None, "ok": False,
                                  "error": "", "case_rel": "cad", "summary": "",
                                  "steps": [], "seconds": 1.0, "stopped": "",
                                  "remarks": []})()

    ctx.cad = Desk()
    dispatch(ctx, "cad", {"request": "the fluid volume", "geometry": path})
    assert seen == {"request": "the fluid volume", "case": None, "geometry": path}


def test_without_a_desk_the_tool_says_so_rather_than_pretending(ctx):
    ctx.cad = None
    answer, _ = dispatch(ctx, "cad", {"request": "a duct"})
    assert "not available" in answer and "bash" in answer


# -- the environment variables --------------------------------------------------


ALIASES = [
    ("OPENREYNOLDS_CAD_MODEL", "OPENREYNOLDS_MESHER_MODEL", "mesher_model",
     "claude-opus-4", "claude-opus-4", "claude-sonnet-4", "claude-sonnet-4"),
    ("OPENREYNOLDS_CAD_EFFORT", "OPENREYNOLDS_MESHER_EFFORT", "mesher_effort",
     "medium", "medium", "low", "low"),
    ("OPENREYNOLDS_CAD_MAX_STEPS", "OPENREYNOLDS_MESHER_MAX_STEPS", "mesher_max_steps",
     "12", 12, "9", 9),
    ("OPENREYNOLDS_CAD_MAX_SECONDS", "OPENREYNOLDS_MESHER_MAX_SECONDS",
     "mesher_max_seconds", "120", 120.0, "90", 90.0),
]
"""Each row: the new variable, the old one, the field they both land in, and the two
values with what the field should hold for each."""


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No config file and none of these variables set, whatever the machine has."""
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", str(tmp_path / "config.json"))
    for new, old, *_rest in ALIASES:
        monkeypatch.delenv(new, raising=False)
        monkeypatch.delenv(old, raising=False)
    for name in ("OPENREYNOLDS_CAD_TOOL", "OPENREYNOLDS_MESH_TOOL"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


PARAMS = "new,old,field,new_value,new_expected,old_value,old_expected"


@pytest.mark.parametrize(PARAMS, ALIASES)
def test_the_old_variable_names_still_work(clean_env, monkeypatch, new, old, field,
                                          new_value, new_expected, old_value,
                                          old_expected):
    """None of these are in `_CONFIG_KEYS`, so nothing persists them and nothing
    migrates them: an environment that sets one and is not read reverts to a default
    with no message anywhere."""
    monkeypatch.setenv(old, old_value)
    assert getattr(Config.load(), field) == old_expected


@pytest.mark.parametrize(PARAMS, ALIASES)
def test_the_new_name_wins_where_both_are_set(clean_env, monkeypatch, new, old, field,
                                              new_value, new_expected, old_value,
                                              old_expected):
    monkeypatch.setenv(old, old_value)
    monkeypatch.setenv(new, new_value)
    assert getattr(Config.load(), field) == new_expected


def test_the_tool_switch_answers_to_both_names(clean_env, monkeypatch):
    """`OPENREYNOLDS_MESH_TOOL` carries `5b588c3`'s unresolved experiment -- the desk
    against the bash the caller already has -- and C10 runs that A/B. A rename that
    silently turned the tool back on would corrupt the arm it was meant to switch
    off, and the run would look like it had worked."""
    assert Config.load().mesh_tool is True
    monkeypatch.setenv("OPENREYNOLDS_MESH_TOOL", "0")
    assert Config.load().mesh_tool is False
    monkeypatch.setenv("OPENREYNOLDS_CAD_TOOL", "1")
    assert Config.load().mesh_tool is True, "the new name does not win"
    monkeypatch.delenv("OPENREYNOLDS_CAD_TOOL")
    monkeypatch.setenv("OPENREYNOLDS_CAD_TOOL", "off")
    assert Config.load().mesh_tool is False


def test_the_json_config_answers_to_both_names_too(clean_env, monkeypatch):
    path = clean_env / "config.json"
    path.write_text(json.dumps({"mesher_effort": "low"}))
    assert Config.load().mesher_effort == "low"
    path.write_text(json.dumps({"mesher_effort": "low", "cad_effort": "medium"}))
    assert Config.load().mesher_effort == "medium"


def test_none_of_these_are_persisted_and_that_has_not_changed(clean_env, monkeypatch):
    """`Config.save()` writes `_CONFIG_KEYS` and nothing else. These were never in it,
    and adding an alias must not have quietly added them: a model name written into
    the credentials file would outlive the session that set it."""
    monkeypatch.setenv("OPENREYNOLDS_CAD_MODEL", "claude-sonnet-4")
    cfg = Config.load()
    written = json.loads(cfg.save().read_text())
    assert "mesher_model" not in written and "cad_model" not in written


# -- the prompt ----------------------------------------------------------------


PROMPT_CAP = 6000
"""Unchanged. The new bullet states the CAD-file fact in 339 characters against the
297 it replaced, so the cap held and did not have to be argued with."""


def test_the_prompt_is_still_inside_its_cap():
    assert len(SYSTEM_PROMPT) < PROMPT_CAP


def test_the_bullet_says_what_the_tool_takes_and_what_stays_with_the_caller():
    """A fact about what exists, which is the only thing this prompt carries. The
    part that had to be bought space for is the CAD file: `openreynolds push` is the
    only way a `.step` reaches the volume and nothing told the model the desk could
    be pointed at one."""
    bullet = next(line for line in SYSTEM_PROMPT.splitlines()
                  if line.startswith("- `cad`") or "- `cad`" in line)
    body = SYSTEM_PROMPT[SYSTEM_PROMPT.index("- `cad`"):]
    body = body[:body.index("\n\n")]
    assert ".step" in body and "on the volume" in body
    assert "mesh" in body and "patch table" in body
    assert "stay with you" in body
    assert bullet


def test_the_prompt_still_states_every_pinned_fact():
    for fact in ("v2512", "pyvista", "24 hours", "sandbox_expired", "latestTime", "/work"):
        assert fact in SYSTEM_PROMPT


@pytest.mark.parametrize("pattern", [
    r"\balways\b", r"\byou must\b", r"\byou should\b", r"\bmake sure\b",
    r"\bworkflow\b", r"\bstep \d", r"\bphase \d", r"\bfirst,",
])
def test_the_new_bullet_did_not_bring_a_procedure_with_it(pattern):
    assert re.search(pattern, SYSTEM_PROMPT, re.IGNORECASE) is None


def test_nothing_volatile_was_interpolated():
    for volatile in ("{", "}"):
        assert volatile not in SYSTEM_PROMPT.replace("kOmegaSST", "")


# -- the image manifest ---------------------------------------------------------


def driver_source() -> str:
    """The kernel driver as it is written into the workspace."""
    from openreynolds.backend import kernel

    return kernel._DRIVER


def test_the_manifest_names_what_the_kernel_channel_needs():
    """The `#2` decision put a kernel on the far side of the protocol, and neither
    `ipykernel` nor `jupyter_client` was declared anywhere. The image has no outbound
    network, so an undeclared import is not a slow failure, it is the channel not
    existing."""
    manifest = (TOOLBOX / "ENVIRONMENT.md").read_text(encoding="utf-8").lower()
    assert "ipykernel" in manifest
    assert "jupyter_client" in manifest


def test_every_import_the_kernel_driver_makes_is_in_the_manifest():
    """The discipline `test_toolbox.py` applies to toolbox scripts, applied to the one
    other file this repo writes onto the image. Standard library is on the image by
    definition; everything else has to be named where it can be read."""
    manifest = (TOOLBOX / "ENVIRONMENT.md").read_text(encoding="utf-8").lower()
    roots = set()
    for node in ast.walk(ast.parse(driver_source())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.add((node.module or "").split(".")[0])
    third_party = sorted(root for root in roots
                         if root and root not in sys.stdlib_module_names)
    assert third_party, "the driver imports nothing third-party; has it moved?"
    for root in third_party:
        assert root.lower() in manifest, (
            f"the kernel driver imports {root} and ENVIRONMENT.md does not name it"
        )


def test_the_kernel_the_driver_asks_for_is_declared_as_a_dependency():
    """`KernelManager(kernel_name="python3")` needs `ipykernel` on the image; nothing
    in the driver's imports says so, which is exactly why it was missed."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "ipykernel" in pyproject and "jupyter_client" in pyproject


# -- the two copy-modify-run templates, disposed of deliberately ----------------


def test_the_gmsh_templates_are_kept_and_the_index_says_who_they_are_for():
    """C9's choice, recorded as a test rather than as a sentence in a report.

    They are the pattern the kernel decision removes *for the desk*, and the desk no
    longer copies and edits them. They are not the pattern removed for a person or for
    the main agent, both of which work in bash and have no kernel: a script that runs
    is still the shortest route to a plane case. So they stay, and the index says why,
    which is the half that stops them reading as the desk's abandoned scaffolding.
    """
    for name in ("duct2d.py", "body_in_box.py"):
        assert (TOOLBOX / "templates" / name).is_file()
    index = " ".join((TOOLBOX / "README.md").read_text(encoding="utf-8").split())
    assert "duct2d.py" in index and "body_in_box.py" in index
    assert "kept for whoever is working in bash" in index


def test_the_desks_nudge_still_names_a_path_that_exists():
    """C8's step-12 nudge points the desk at `templates/` by path. Whichever way the
    templates went, the sentence has to stay true.

    The path is no longer written into the sentence. It is filled from the backend's
    own workspace root, because the literal `/work` was true on the image and false
    under `LocalBackend`, where it sent a desk hunting for a directory that was not
    there -- seven steps of one run, including a filesystem-wide `find`. So the
    assertion is made twice: the sentence carries the placeholder, and the placeholder
    resolves to the old literal on the workspace the old literal described."""
    from openreynolds.backend.base import WORKSPACE_ROOT
    from openreynolds.cad.agent import NUDGE

    assert "{toolbox}/templates/" in NUDGE
    assert "/work/.toolbox/templates/" in NUDGE.format(
        toolbox=f"{WORKSPACE_ROOT}/.toolbox"
    )
    assert (TOOLBOX / "templates" / "prep" / "README.md").is_file()
    assert (TOOLBOX / "templates" / "snappy" / "snappyHexMeshDict").is_file()


# -- the desk being replaced ----------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="`openreynolds/mesher/` is still here on purpose: C10 measures its baseline "
           "against it and runs T7/T8 against it to show they are new, and its recorded "
           "pass rate is not in the tree. This flips to a plain assertion -- delete the "
           "marker, keep the test -- when the package is deleted after C10.",
)
def test_nothing_imports_the_desk_being_replaced():
    """"Gone, not extended." The import graph is where that is decided."""
    offenders = []
    for path in sorted([*PACKAGE.rglob("*.py"), *(ROOT / "tests").rglob("*.py"),
                        *(ROOT / "scripts").rglob("*.py")]):
        if "mesher" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bfrom \.?mesher\b|\bopenreynolds\.mesher\b|"
                     r"\bfrom \. import .*\bmesher\b", text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"still importing the old desk: {', '.join(offenders)}"


def test_the_harness_no_longer_constructs_the_old_desk():
    """The half of the deletion that can be finished now: whatever still imports
    `mesher/` for its tests, the session does not build one."""
    cli = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    assert "mesher.Mesher(" not in cli
    # `CoreDesk` since 2026-09-18, and it is a `CadDesk` -- the same loop, the same
    # finish check, briefed without the instrument catalogue that cost three times the
    # steps on eight of eight prompts. What ships is now the configuration the corpus
    # measured, by identity rather than by our asserting the two are equivalent.
    assert "cad.CoreDesk(" in cli
    assert "cad.CadDesk(" not in cli
    assert "from . import cad," in cli


def test_the_session_does_not_import_the_measurement_harness():
    """Shipping the corpus's desk must not ship the corpus.

    `buildup` is the observer: `probes` `sys.path`-imports four toolbox scripts into the
    calling process at import time, and `isolation` and `supervise` exist only to watch a
    run from outside it. None of that belongs in a user's session, which is why the desk
    moved to `cad/core.py` rather than being constructed where it used to live.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import openreynolds.cli, sys; "
         "print([m for m in sys.modules if 'buildup' in m])"],
        capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "[]", f"the session pulled in {out.stdout.strip()}"


# -- the unit leak, cured in the other place it happens -------------------------


LEAK_PROBE = r'''
import json, re, sys
from pathlib import Path
sys.path.insert(0, {toolbox!r})
import gmsh
import cad_convert

directory = Path({directory!r})

def write_box(name):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("leak")
    gmsh.model.occ.addBox(0, 0, 0, 100, 40, 20)
    gmsh.model.occ.synchronize()
    gmsh.write(str(directory / name))
    gmsh.finalize()
    text = (directory / name).read_text(errors="replace")
    xs = [float(m) for m in re.findall(r"CARTESIAN_POINT\('',\(([-0-9.E+]+),", text)]
    return max(xs)

before = write_box("before.step")
if {control!r}:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.add("q")
    gmsh.model.occ.importShapes({source!r})
    gmsh.model.occ.synchronize()
    gmsh.finalize()
else:
    cad_convert.render_tessellation(Path({source!r}), directory / "render")
after = write_box("after.step")
print(json.dumps({{"before": before, "after": after, "ratio": after / before}}))
'''


@needs_gmsh
@pytest.mark.parametrize("control", [True, False])
def test_render_tessellation_puts_the_occ_unit_static_back(tmp_path, control):
    """C5 cured this inside `export_patches()` and could not reach here.

    `Geometry.OCCTargetUnit` is a gmsh option that writes through to an OpenCASCADE
    static; `finalize()` resets the option and not the static, and the static is read
    by the STEP **writer**. So drawing a picture of a STEP file that declares
    millimetres leaves every later STEP write in the process a thousand times too
    large, still declaring millimetres -- a factor of 1000 on every length, arriving
    through the most innocent call in the file, which four scripts import.

    Parametrised against the bare sequence so "no leak" cannot mean "no instrument".
    """
    source = ROOT / "tests" / "data" / "cad" / "synthetic" / "box_with_duct.step"
    directory = tmp_path / ("control" if control else "cured")
    directory.mkdir()
    script = tmp_path / "probe.py"
    script.write_text(LEAK_PROBE.format(
        toolbox=str(TOOLBOX), directory=str(directory), control=control,
        source=str(source),
    ))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    if control:
        assert measured["ratio"] == pytest.approx(1000.0), (
            "the leak is gone from gmsh itself, so this proves nothing about the cure"
        )
    else:
        assert measured["ratio"] == pytest.approx(1.0), measured
