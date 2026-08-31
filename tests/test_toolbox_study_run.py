"""The resumable study orchestrator.

Everything worth testing here is a decision made from files on disk -- is this
phase already done, which phases would this invocation run, what happens to the
phase table when a command fails -- so none of it needs OpenFOAM and none of it is
allowed to run any. The command runner is injected; the fake one records what it
was asked to run and returns whatever the test wants.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study_run():
    return load("study_run")


@pytest.fixture
def state():
    return load("study_state")


# -- fixtures for a case on disk ---------------------------------------------------


def make_study(tmp_path: Path) -> Path:
    """A study directory with a case in it, the way `find_root` expects to find one."""
    root = tmp_path / "study"
    (root / ".reynolds").mkdir(parents=True)
    case = root / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir(parents=True)
    return case


def make_toolbox(tmp_path: Path, *names: str) -> Path:
    """A toolbox holding only the scripts a test wants to exist.

    Deliberately not the real one: the sibling scripts land at different times and
    a test that depended on which of them exists today would fail tomorrow for a
    reason that has nothing to do with it.
    """
    directory = tmp_path / "tools"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("# stand-in\n", encoding="utf-8")
    return directory


def give_mesh(case: Path) -> None:
    polymesh = case / "constant" / "polyMesh"
    polymesh.mkdir(parents=True, exist_ok=True)
    (polymesh / "owner").write_text("owner\n", encoding="utf-8")


def write_phases(state, root: Path, **statuses: str) -> None:
    for name, status in statuses.items():
        state.set_phase(name, status, root=root)


# -- what is on disk ---------------------------------------------------------------


def test_a_mesh_is_owner_and_not_the_directory(tmp_path, study_run):
    """An interrupted snappyHexMesh leaves constant/polyMesh behind holding nothing."""
    case = make_study(tmp_path)
    (case / "constant" / "polyMesh").mkdir()
    assert study_run.has_mesh(case) is False

    give_mesh(case)
    assert study_run.has_mesh(case) is True


def test_time_directories_are_the_numeric_ones(tmp_path, study_run):
    case = make_study(tmp_path)
    for name in ("0", "0.orig", "0.5", "10", "constant", "system", "notes"):
        (case / name).mkdir(exist_ok=True)

    values = [value for value, _path in study_run.numeric_dirs(case)]
    assert values == [0.0, 0.5, 10.0]
    assert [value for value, _ in study_run.written_times(case)] == [0.5, 10.0]
    assert study_run.latest_time(case) == 10.0


def test_no_written_times_is_none_not_zero(tmp_path, study_run):
    case = make_study(tmp_path)
    (case / "0").mkdir()
    assert study_run.latest_time(case) is None


def test_processor_directories_sort_numerically(tmp_path, study_run):
    case = make_study(tmp_path)
    for index in (0, 1, 2, 10):
        (case / f"processor{index}").mkdir()
    (case / "processorSomething").mkdir()

    names = [path.name for path in study_run.processor_dirs(case)]
    assert names == ["processor0", "processor1", "processor2", "processor10"]


def test_a_log_without_end_did_not_finish(tmp_path, study_run):
    """A reaped solver leaves a log full of healthy iterations and no `End`."""
    killed = tmp_path / "log.killed"
    killed.write_text("Time = 300\nExecutionTime = 91 s\n", encoding="utf-8")
    finished = tmp_path / "log.finished"
    finished.write_text("Time = 300\nExecutionTime = 91 s\nEnd\n", encoding="utf-8")

    assert study_run.log_finished(killed) is False
    assert study_run.log_finished(finished) is True
    assert study_run.log_finished(tmp_path / "log.absent") is False


def test_the_tail_is_the_last_lines(tmp_path, study_run):
    log = tmp_path / "log"
    log.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")

    tail = study_run.log_tail(log, lines=3)

    assert tail.splitlines() == ["line 97", "line 98", "line 99"]
    assert study_run.log_tail(tmp_path / "nothing") == ""


def test_a_frames_directory_needs_frames_in_it(tmp_path, study_run):
    empty = tmp_path / "wake_frames"
    empty.mkdir()
    assert study_run.frame_dirs(tmp_path) == []

    (empty / "frame_0000.png").write_bytes(b"x")
    assert study_run.frame_dirs(tmp_path) == []

    (empty / "frame_0001.png").write_bytes(b"x")
    assert [path.name for path in study_run.frame_dirs(tmp_path)] == ["wake_frames"]


# -- evidence per phase ------------------------------------------------------------


def test_geometry_is_a_surface_or_a_block(tmp_path, study_run):
    case = make_study(tmp_path)
    ctx = study_run.context(case)
    assert study_run.geometry_evidence(ctx)[0] is False

    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    assert study_run.geometry_evidence(ctx)[0] is True


def test_a_surface_counts_as_geometry(tmp_path, study_run):
    case = make_study(tmp_path)
    (case / "constant" / "triSurface").mkdir()
    (case / "constant" / "triSurface" / "wing.stl").write_text("solid\n", encoding="utf-8")
    (case / "constant" / "triSurface" / "notes.txt").write_text("hi\n", encoding="utf-8")

    ctx = study_run.context(case)
    evident, why = study_run.geometry_evidence(ctx)

    assert evident is True
    assert "1 surface" in why
    assert [path.name for path in study_run.surfaces(case)] == ["wing.stl"]


def test_the_solve_is_done_when_its_log_ends(tmp_path, study_run):
    case = make_study(tmp_path)
    ctx = study_run.context(case, solver="simpleFoam")
    assert study_run.solve_evidence(ctx)[0] is False

    (case / "log.simpleFoam").write_text("Time = 5\n", encoding="utf-8")
    (case / "5").mkdir()
    evident, why = study_run.solve_evidence(ctx)
    assert evident is False
    assert "no End line" in why and "5" in why

    (case / "log.simpleFoam").write_text("Time = 5\nEnd\n", encoding="utf-8")
    assert study_run.solve_evidence(ctx)[0] is True


def test_a_checkmesh_that_never_ran_is_not_evidence_that_it_did(tmp_path, study_run):
    """The runner writes `$ checkMesh` into the log before starting checkMesh, so a
    checkMesh that is not on the image leaves a file behind and nothing else."""
    case = make_study(tmp_path)
    ctx = study_run.context(case)
    assert study_run.check_mesh_evidence(ctx)[0] is False

    (case / "log.checkMesh").write_text("$ checkMesh\n", encoding="utf-8")
    evident, why = study_run.check_mesh_evidence(ctx)
    assert evident is False
    assert "no End line" in why

    (case / "log.checkMesh").write_text("$ checkMesh\nMesh OK.\nEnd\n", encoding="utf-8")
    assert study_run.check_mesh_evidence(ctx)[0] is True


def test_a_failed_checkmesh_does_not_come_back_done_next_session(tmp_path, study_run, state):
    """The whole point of resuming: a phase that failed must not be walked past
    because the failure itself left a file on disk."""
    case = make_study(tmp_path)
    give_mesh(case)
    ctx = study_run.context(case)
    runner = Runner(code=127, tail="checkMesh: not found on PATH")

    study_run.run_phase(ctx, "checkMesh", runner=runner, journal=study_run.Journal())
    assert state.phase_status("checkMesh", case.parent) == "failed"

    states = study_run.reconcile(study_run.context(case))
    result = {s.name: s for s in states}
    assert result["checkMesh"].status == "failed"
    assert "checkMesh" in study_run.build_plan(states)


def test_a_probe_log_holding_only_the_command_is_not_a_probe(tmp_path, study_run):
    case = make_study(tmp_path)
    ctx = study_run.context(case)

    (case / "log.preflight").write_text("$ python preflight.py /work/case\n", encoding="utf-8")
    evident, why = study_run.probe_evidence(ctx)
    assert evident is False
    assert "only the command" in why

    (case / "log.preflight").write_text("$ python preflight.py\nboundary conditions ok\n",
                                        encoding="utf-8")
    assert study_run.probe_evidence(ctx)[0] is True


def test_the_end_of_a_long_log_is_read_without_reading_the_log(tmp_path, study_run):
    """`--status` asks whether the solve finished, and a solve log is not small."""
    log = tmp_path / "log.simpleFoam"
    with log.open("wb") as handle:
        handle.write(b"Time = 1\n" * 20000)
        handle.write(b"End\n")

    assert study_run.log_finished(log) is True
    assert len(study_run.tail_text(log, max_bytes=512)) <= 512
    assert study_run.tail_text(log).endswith("End\n")
    assert study_run.tail_text(tmp_path / "absent") == ""


def test_a_serial_run_has_nothing_to_reconstruct(tmp_path, study_run):
    case = make_study(tmp_path)
    ctx = study_run.context(case)

    evident, why = study_run.reconstruct_evidence(ctx)
    assert evident is True
    assert "serial" in why
    assert study_run.reconstruct_build(ctx)[0] == []


def test_a_decomposed_run_needs_a_reconstructed_time(tmp_path, study_run):
    case = make_study(tmp_path)
    (case / "processor0").mkdir()
    ctx = study_run.context(case)

    assert study_run.reconstruct_evidence(ctx)[0] is False
    commands, _note = study_run.reconstruct_build(ctx)
    assert [command.argv for command in commands] == [["reconstructPar"]]

    (case / "100").mkdir()
    assert study_run.reconstruct_evidence(ctx)[0] is True


def test_a_geometry_picture_is_not_evidence_that_fields_were_rendered(tmp_path, study_run):
    case = make_study(tmp_path)
    renders = case / "renders"
    renders.mkdir()
    (renders / "geometry.png").write_bytes(b"png")
    ctx = study_run.context(case)

    assert study_run.render_evidence(ctx)[0] is False
    assert study_run.preview_evidence(ctx)[0] is True

    (renders / "U_slice.png").write_bytes(b"png")
    assert study_run.render_evidence(ctx)[0] is True


def test_the_results_directory_counts_as_a_render(tmp_path, study_run):
    case = make_study(tmp_path)
    pictures = case / "results" / "wake"
    pictures.mkdir(parents=True)
    (pictures / "vorticity.png").write_bytes(b"png")

    assert study_run.render_evidence(study_run.context(case))[0] is True


def test_a_registered_artifact_is_evidence(tmp_path, study_run, state):
    case = make_study(tmp_path)
    root = case.parent
    picture = root / "renders" / "geometry.png"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"png")
    state.record("geometry-preview", picture, root=root)

    evident, why = study_run.preview_evidence(study_run.context(case))

    assert evident is True
    assert "geometry-preview" in why


# -- building the commands ---------------------------------------------------------


def test_the_mesh_phase_runs_the_dictionaries_that_are_there(tmp_path, study_run):
    case = make_study(tmp_path)
    system = case / "system"
    system.joinpath("blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    system.joinpath("surfaceFeatureExtractDict").write_text("wing.stl {}\n", encoding="utf-8")
    system.joinpath("snappyHexMeshDict").write_text("castellatedMesh true;\n", encoding="utf-8")

    commands, note = study_run.mesh_build(study_run.context(case))

    assert [command.argv for command in commands] == [
        ["blockMesh"],
        ["surfaceFeatureExtract"],
        ["snappyHexMesh", "-overwrite"],
    ]
    assert note == ""
    assert commands[-1].log.name == "log.snappyHexMesh"
    assert commands[0].cwd == case


def test_no_dictionaries_means_nothing_to_mesh(tmp_path, study_run):
    case = make_study(tmp_path)
    commands, note = study_run.mesh_build(study_run.context(case))
    assert commands == []
    assert "blockMeshDict" in note


def test_check_mesh_digests_its_own_log_when_the_digest_is_there(tmp_path, study_run):
    case = make_study(tmp_path)
    ctx = study_run.context(case, toolbox=make_toolbox(tmp_path, "mesh_digest.py"))

    commands, _note = study_run.check_mesh_build(ctx)

    assert commands[0].argv == ["checkMesh"]
    assert commands[0].log == case / "log.checkMesh"
    assert commands[1].argv[-1] == str(case / "log.checkMesh")
    assert len(commands) == 2


def test_a_missing_script_is_a_skip_and_not_a_crash(tmp_path, study_run):
    """Another agent's script may not have landed yet."""
    case = make_study(tmp_path)
    ctx = study_run.context(case, toolbox=make_toolbox(tmp_path))

    for build in (study_run.preview_build, study_run.probe_build,
                  study_run.render_build, study_run.report_build):
        commands, note = build(ctx)
        assert commands == []
        assert ".py" in note


def test_preview_falls_back_to_geometry_view(tmp_path, study_run):
    case = make_study(tmp_path)
    (case / "constant" / "triSurface").mkdir()
    (case / "constant" / "triSurface" / "wing.stl").write_text("solid\n", encoding="utf-8")
    ctx = study_run.context(case, toolbox=make_toolbox(tmp_path, "geometry_view.py"))

    commands, note = study_run.preview_build(ctx)

    assert note == ""
    assert commands[0].argv[1].endswith("geometry_view.py")
    assert str(case / "constant" / "triSurface") in commands[0].argv


def test_preview_prefers_the_script_that_owns_the_phase(tmp_path, study_run):
    case = make_study(tmp_path)
    ctx = study_run.context(case, toolbox=make_toolbox(tmp_path, "first_look.py", "geometry_view.py"))

    commands, _note = study_run.preview_build(ctx)

    assert commands[0].argv[1].endswith("first_look.py")


def test_render_prefers_results_over_render(tmp_path, study_run):
    case = make_study(tmp_path)
    both = make_toolbox(tmp_path, "results.py", "render.py")

    commands, _note = study_run.render_build(study_run.context(case, toolbox=both))
    assert commands[0].argv[1].endswith("results.py")

    only_old = make_toolbox(tmp_path / "old", "render.py")
    commands, _note = study_run.render_build(study_run.context(case, toolbox=only_old))
    assert commands[0].argv[1].endswith("render.py")


def test_animate_waits_until_there_is_something_to_animate(tmp_path, study_run):
    case = make_study(tmp_path)
    ctx = study_run.context(case, toolbox=make_toolbox(tmp_path, "animate.py"), field="vorticity")

    commands, note = study_run.animate_build(ctx)
    assert commands == []
    assert "nothing to animate" in note

    for name in ("1", "2"):
        (case / name).mkdir()
    commands, note = study_run.animate_build(ctx)
    assert note == ""
    assert commands[0].argv[-2:] == ["--field", "vorticity"]


def test_parallel_decomposes_then_runs_mpirun(tmp_path, study_run):
    case = make_study(tmp_path)
    (case / "system" / "decomposeParDict").write_text("numberOfSubdomains 4;\n", encoding="utf-8")
    ctx = study_run.context(case, solver="pimpleFoam", parallel=4)

    commands, note = study_run.solve_build(ctx)

    assert note == ""
    assert [command.argv for command in commands] == [
        ["decomposePar"],
        ["mpirun", "-np", "4", "pimpleFoam", "-parallel"],
    ]
    assert commands[-1].log.name == "log.pimpleFoam"


def test_an_already_decomposed_case_is_not_decomposed_twice(tmp_path, study_run):
    case = make_study(tmp_path)
    (case / "system" / "decomposeParDict").write_text("numberOfSubdomains 4;\n", encoding="utf-8")
    (case / "processor0").mkdir()
    ctx = study_run.context(case, parallel=4)

    commands, _note = study_run.solve_build(ctx)

    assert commands[0].argv[0] == "mpirun"


def test_parallel_without_a_decompose_dict_says_so(tmp_path, study_run):
    case = make_study(tmp_path)
    commands, note = study_run.solve_build(study_run.context(case, parallel=4))
    assert commands == []
    assert "decomposeParDict" in note


def test_a_serial_solve_is_just_the_solver(tmp_path, study_run):
    case = make_study(tmp_path)
    commands, _note = study_run.solve_build(study_run.context(case, solver="icoFoam"))
    assert [command.argv for command in commands] == [["icoFoam"]]
    assert commands[0].log.name == "log.icoFoam"


# -- reconciling the record with the evidence --------------------------------------


def statuses(states) -> dict:
    return {state.name: state.status for state in states}


def test_evidence_on_disk_beats_a_pending_record(tmp_path, study_run, state):
    """Someone ran snappyHexMesh by hand between sessions."""
    case = make_study(tmp_path)
    give_mesh(case)

    result = study_run.reconcile(study_run.context(case))

    assert statuses(result)["mesh"] == "done"
    assert dict((s.name, s.recorded) for s in result)["mesh"] == "pending"


def test_a_recorded_done_with_no_evidence_is_stale(tmp_path, study_run, state):
    """The mesh was deleted after being recorded, so it is not done any more."""
    case = make_study(tmp_path)
    write_phases(state, case.parent, mesh="done")

    result = {s.name: s for s in study_run.reconcile(study_run.context(case))}

    assert result["mesh"].status == "pending"
    assert result["mesh"].stale is True
    assert result["mesh"].recorded == "done"


def test_skipped_is_a_decision_and_survives(tmp_path, study_run, state):
    case = make_study(tmp_path)
    write_phases(state, case.parent, animate="skipped")

    result = {s.name: s for s in study_run.reconcile(study_run.context(case))}

    assert result["animate"].status == "skipped"
    assert result["animate"].stale is False


def test_skipped_yields_to_evidence(tmp_path, study_run, state):
    case = make_study(tmp_path)
    write_phases(state, case.parent, mesh="skipped")
    give_mesh(case)

    result = {s.name: s for s in study_run.reconcile(study_run.context(case))}

    assert result["mesh"].status == "done"


def test_a_failure_stays_visible_as_a_failure(tmp_path, study_run, state):
    case = make_study(tmp_path)
    write_phases(state, case.parent, mesh="failed")

    result = {s.name: s for s in study_run.reconcile(study_run.context(case))}

    assert result["mesh"].status == "failed"


def test_the_next_phase_is_the_first_unsettled_one(tmp_path, study_run, state):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    write_phases(state, case.parent, preview="skipped")
    give_mesh(case)
    (case / "log.checkMesh").write_text("Mesh OK.\nEnd\n", encoding="utf-8")

    states = study_run.reconcile(study_run.context(case))

    assert statuses(states)["geometry"] == "done"
    assert study_run.next_to_run(states) == "probe"


def test_a_finished_pipeline_has_no_next_phase(tmp_path, study_run, state):
    case = make_study(tmp_path)
    write_phases(state, case.parent, **{name: "skipped" for name in study_run.PHASE_NAMES})

    assert study_run.next_to_run(study_run.reconcile(study_run.context(case))) == ""


# -- the plan ----------------------------------------------------------------------


def fake_states(study_run, **overrides):
    """Phase states without touching a disk, for plan-shape tests."""
    return [
        study_run.PhaseState(name, overrides.get(name, "pending"), False, "why",
                             overrides.get(name, "pending"), False)
        for name in study_run.PHASE_NAMES
    ]


def test_the_plain_plan_resumes_and_leaves_finished_phases_alone(study_run):
    states = fake_states(study_run, geometry="done", preview="skipped", mesh="done")

    plan = study_run.build_plan(states)

    assert plan[0] == "checkMesh"
    assert "mesh" not in plan and "geometry" not in plan and "preview" not in plan


def test_a_done_phase_in_the_middle_is_not_redone(study_run):
    states = fake_states(study_run, render="done")
    assert "render" not in study_run.build_plan(states)


def test_from_includes_phases_already_done(study_run):
    """The reason to ask for --from is that something upstream changed."""
    states = fake_states(study_run, mesh="done", checkMesh="done", solve="done")

    plan = study_run.build_plan(states, start="mesh")

    assert plan == list(study_run.PHASE_NAMES[study_run.PHASE_NAMES.index("mesh"):])


def test_only_runs_exactly_one_phase(study_run):
    assert study_run.build_plan(fake_states(study_run), only="checkMesh") == ["checkMesh"]


def test_skip_removes_phases_from_any_plan(study_run):
    states = fake_states(study_run)

    assert "animate" not in study_run.build_plan(states, skip=["animate"])
    assert "animate" not in study_run.build_plan(states, start="mesh", skip=["animate", "report"])
    assert study_run.build_plan(states, only="animate", skip=["animate"]) == []


def test_a_phase_name_that_does_not_exist_is_refused(study_run):
    with pytest.raises(ValueError):
        study_run.build_plan(fake_states(study_run), start="meshing")
    with pytest.raises(ValueError):
        study_run.build_plan(fake_states(study_run), skip=["solveit"])


def test_the_dry_run_plan_shows_the_commands(tmp_path, study_run):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    ctx = study_run.context(case, toolbox=make_toolbox(tmp_path))

    text = "\n".join(study_run.plan_lines(ctx, ["mesh", "preview"]))

    assert "blockMesh" in text
    assert "would be skipped" in text


# -- running, with a fake runner ---------------------------------------------------


class Runner:
    """Stands in for the shell. Records what it was asked to run, returns what it
    was told to, and can leave a file behind so a phase's evidence appears."""

    def __init__(self, code=0, tail="", writes=()):
        self.code = code
        self.tail = tail
        self.writes = list(writes)
        self.seen = []

    def __call__(self, command):
        self.seen.append(command)
        command.log.parent.mkdir(parents=True, exist_ok=True)
        command.log.write_text(f"$ {' '.join(command.argv)}\n{self.tail}\n", encoding="utf-8")
        for path in self.writes:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("written\n", encoding="utf-8")
        return self.code, self.tail

    @property
    def argvs(self):
        return [command.argv for command in self.seen]


def test_a_successful_phase_is_recorded_done(tmp_path, study_run, state, capsys):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    ctx = study_run.context(case)
    runner = Runner(writes=[case / "constant" / "polyMesh" / "owner"])

    result = study_run.run_phase(ctx, "mesh", runner=runner, journal=study_run.Journal())

    assert result.status == "done"
    assert runner.argvs == [["blockMesh"]]
    assert state.phase_status("mesh", case.parent) == "done"


def test_a_phase_that_produced_no_evidence_says_so_without_failing(tmp_path, study_run, state):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    ctx = study_run.context(case)

    result = study_run.run_phase(ctx, "mesh", runner=Runner(), journal=study_run.Journal())

    assert result.status == "done"
    assert "commands succeeded but" in result.note
    assert state.phase_status("mesh", case.parent) == "done"


def test_a_failing_phase_is_recorded_with_the_tail_of_its_log(tmp_path, study_run, state):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    ctx = study_run.context(case)
    runner = Runner(code=1, tail="--> FOAM FATAL ERROR: face 12 has negative volume")

    result = study_run.run_phase(ctx, "mesh", runner=runner, journal=study_run.Journal())

    assert result.status == "failed"
    assert "FOAM FATAL ERROR" in result.note
    assert "exited 1" in result.note

    table = json.loads((case.parent / ".reynolds" / "phases.json").read_text(encoding="utf-8"))
    row = [entry for entry in table["phases"] if entry["name"] == "mesh"][0]
    assert row["status"] == "failed"
    assert "negative volume" in row["note"]


def test_a_failure_stops_the_pipeline_and_exits_non_zero(tmp_path, study_run, state):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    ctx = study_run.context(case)
    runner = Runner(code=1, tail="boom")

    code, results = study_run.execute(ctx, ["mesh", "checkMesh", "solve"],
                                      runner=runner, journal=study_run.Journal())

    assert code == 1
    assert [result.name for result in results] == ["mesh"]
    assert runner.argvs == [["blockMesh"]]
    assert state.phase_status("checkMesh", case.parent) == "pending"


def test_nothing_is_repaired_after_a_failure(tmp_path, study_run):
    """No retry, no fallback: the phase failed and the run stopped."""
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    runner = Runner(code=1, tail="boom")

    study_run.execute(study_run.context(case), ["mesh"], runner=runner,
                      journal=study_run.Journal())

    assert len(runner.seen) == 1


def test_a_phase_whose_script_is_missing_is_skipped_not_run(tmp_path, study_run, state):
    case = make_study(tmp_path)
    ctx = study_run.context(case, toolbox=make_toolbox(tmp_path))
    runner = Runner()

    result = study_run.run_phase(ctx, "render", runner=runner, journal=study_run.Journal())

    assert result.status == "skipped"
    assert runner.seen == []
    assert state.phase_status("render", case.parent) == "skipped"


def test_a_phase_with_no_commands_but_evidence_is_done(tmp_path, study_run, state):
    """Geometry is authored by hand: there is nothing to run and it is still done."""
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    ctx = study_run.context(case)

    result = study_run.run_phase(ctx, "geometry", runner=Runner(), journal=study_run.Journal())

    assert result.status == "done"
    assert state.phase_status("geometry", case.parent) == "done"


def test_logs_are_registered_so_they_can_be_found_later(tmp_path, study_run, state):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")
    ctx = study_run.context(case)

    study_run.run_phase(ctx, "mesh", runner=Runner(), journal=study_run.Journal())

    rows = state.artifacts(root=case.parent)
    assert any(row["path"].endswith("log.blockMesh") for row in rows)
    assert all(row["kind"] == "other" for row in rows)


def test_skipping_on_the_command_line_is_recorded(tmp_path, study_run, state):
    case = make_study(tmp_path)
    ctx = study_run.context(case)

    study_run.execute(ctx, [], runner=Runner(), journal=study_run.Journal(),
                      skip=["animate", "report"])

    assert state.phase_status("animate", case.parent) == "skipped"
    assert state.phase_status("report", case.parent) == "skipped"


def test_the_journal_writes_a_line_at_a_time(tmp_path, study_run, capsys):
    log = tmp_path / ".reynolds" / "study_run.log"
    journal = study_run.Journal(log)

    journal.say("mesh: blockMesh")
    assert log.read_text(encoding="utf-8") == "mesh: blockMesh\n"

    journal.say("mesh: done")
    assert log.read_text(encoding="utf-8").splitlines() == ["mesh: blockMesh", "mesh: done"]
    assert "mesh: done" in capsys.readouterr().out


# -- the command line --------------------------------------------------------------


def test_status_prints_the_table_and_runs_nothing(tmp_path, study_run, capsys):
    case = make_study(tmp_path)
    give_mesh(case)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")

    code = study_run.main([str(case), "--status"])
    out = capsys.readouterr().out

    assert code == 0
    assert "done     mesh" in out
    assert "next incomplete phase: preview" in out
    # reconstruct settles itself: a serial run has nothing to reconstruct.
    assert "reconstruct" not in out.split("would run: ")[1]
    assert not (case.parent / ".reynolds" / "study_run.log").exists()


def test_dry_run_prints_the_plan_and_runs_nothing(tmp_path, study_run, capsys):
    case = make_study(tmp_path)
    (case / "system" / "blockMeshDict").write_text("blocks ();\n", encoding="utf-8")

    code = study_run.main([str(case), "--dry-run", "--skip", "animate"])
    out = capsys.readouterr().out

    assert code == 0
    assert "blockMesh" in out
    assert "skipped on request: animate" in out
    assert not (case.parent / ".reynolds" / "study_run.log").exists()


def test_from_and_only_together_are_refused(tmp_path, study_run):
    case = make_study(tmp_path)
    with pytest.raises(SystemExit):
        study_run.main([str(case), "--from", "mesh", "--only", "solve"])


def test_a_case_that_is_not_there_is_refused(tmp_path, study_run):
    with pytest.raises(SystemExit):
        study_run.main([str(tmp_path / "nope"), "--status"])


def test_an_unknown_phase_name_is_refused_by_the_parser(tmp_path, study_run):
    case = make_study(tmp_path)
    with pytest.raises(SystemExit):
        study_run.main([str(case), "--only", "meshing"])


# -- which pictures the render phase asks for ---------------------------------------


def _case_with(tmp_path, patches: str, application: str = "simpleFoam"):
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "constant" / "polyMesh" / "boundary").write_text(patches, encoding="utf-8")
    (case / "system" / "controlDict").write_text(
        f"application     {application};\nendTime 1;\n", encoding="utf-8"
    )
    (case / "0.5").mkdir()  # a written time, so it is not read as unmeshed
    return case


DUCT_PATCHES = """4
(
    inlet   { type patch; nFaces 20; }
    outlet  { type patch; nFaces 20; }
    walls   { type wall; nFaces 200; }
    frontAndBack { type empty; nFaces 800; }
)
"""

EXTERNAL_2D_PATCHES = DUCT_PATCHES.replace("walls   { type wall;", "body    { type wall;")

SNAPPY_3D_PATCHES = """4
(
    inlet  { type patch; nFaces 20; }
    outlet { type patch; nFaces 20; }
    car    { type wall; nFaces 900; }
    ground { type wall; nFaces 200; }
)
"""


def test_a_named_preset_beats_whatever_the_case_looks_like(study_run, tmp_path):
    case = _case_with(tmp_path, DUCT_PATCHES)
    ctx = study_run.context(case, toolbox=TOOLBOX, preset="vehicle-aero")
    assert study_run.results_preset(ctx) == "vehicle-aero"


def test_an_unmeshed_case_wants_the_mesh_looked_at(study_run, tmp_path):
    case = tmp_path / "fresh"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
    ctx = study_run.context(case, toolbox=TOOLBOX)
    assert study_run.results_preset(ctx) == "mesh-validation"


def test_a_duct_is_recognised_by_having_nothing_in_the_flow(study_run, tmp_path):
    case = _case_with(tmp_path, DUCT_PATCHES)
    ctx = study_run.context(case, toolbox=TOOLBOX)
    assert study_run.results_preset(ctx) == "duct-flow"


def test_a_2d_case_with_a_body_gets_the_external_presets(study_run, tmp_path):
    steady = _case_with(tmp_path / "a", EXTERNAL_2D_PATCHES, "simpleFoam")
    assert study_run.results_preset(study_run.context(steady, toolbox=TOOLBOX)) == "external-flow-2d"

    unsteady = _case_with(tmp_path / "b", EXTERNAL_2D_PATCHES, "pimpleFoam")
    assert study_run.results_preset(study_run.context(unsteady, toolbox=TOOLBOX)) == "transient-wake"


def test_a_3d_case_with_a_body_is_not_read_as_a_duct(study_run, tmp_path):
    """`no topWall and no bottomWall` meant duct, so every snappyHexMesh case --
    whose patches are whatever the STL was called -- rendered as a duct: no
    vorticity, no forces, no mesh cut. And `vehicle-aero` was unreachable."""
    case = _case_with(tmp_path, SNAPPY_3D_PATCHES)
    ctx = study_run.context(case, toolbox=TOOLBOX)
    assert study_run.results_preset(ctx) == "vehicle-aero"
