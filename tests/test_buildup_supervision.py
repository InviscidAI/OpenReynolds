"""The observer outside the run: a pidfile, a beat per turn, and three named alarms.

The supervision this replaces was wrong twice, and both tests below are named for what
went wrong. `until ! pgrep -f '<pattern>'` matched the watcher's own command line, so the
condition was never true and every watcher spun to its timeout. And the failure that
actually mattered -- a run producing replies and executing zero cells -- went unnoticed
for three consecutive runs, each burning its clock and reporting an ordinary budget
exhaustion.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from openreynolds.buildup import alarms, heartbeat, record, supervise
from openreynolds.buildup.heartbeat import Beat


def beats(*rows, at: float = 0.0):
    """Turns as `(turn, steps, stop_reason, text_chars)`, all at one moment."""
    return [Beat(turn=t, steps=s, stop_reason=r, text_chars=c, at=at)
            for t, s, r, c in rows]


# -- the heartbeat --------------------------------------------------------------


def test_a_beat_is_on_disk_before_the_call_returns(tmp_path):
    """A run that is killed keeps every beat it wrote, and the beats say why it was."""
    pulse = heartbeat.Heartbeat(tmp_path)
    pulse.start()
    pulse.beat(steps=0, stop_reason="end_turn", text_chars=120, fenced=True)
    assert [b.turn for b in heartbeat.read(tmp_path)] == [1]
    assert heartbeat.pid_of(tmp_path) == os.getpid()


def test_a_half_written_last_line_is_dropped_rather_than_raised_on(tmp_path):
    pulse = heartbeat.Heartbeat(tmp_path)
    pulse.beat(steps=1)
    (tmp_path / heartbeat.BEATS).open("a", encoding="utf-8").write('{"turn": 2, "st')
    assert [b.turn for b in heartbeat.read(tmp_path)] == [1]


def test_liveness_is_a_pid_and_never_a_command_line():
    """`pgrep -f` found the watcher itself every time. `kill -0` cannot."""
    assert heartbeat.alive(os.getpid())
    assert not heartbeat.alive(None)
    assert not heartbeat.alive(4_000_000)


def test_a_desk_hook_beats_every_turn_whether_or_not_a_cell_ran(tmp_path):
    class Desk:
        on_turn = None

    desk = Desk()
    pulse = heartbeat.attach(desk, tmp_path)
    desk.on_turn(turn=1, steps=0, stop_reason="max_tokens", text_chars=0)
    desk.on_turn(turn=2, steps=0, stop_reason="end_turn", text_chars=40, fenced=True)
    assert [(b.turn, b.steps) for b in heartbeat.read(tmp_path)] == [(1, 0), (2, 0)]
    assert pulse.turns == 2


# -- the alarms -----------------------------------------------------------------


def test_replying_and_running_nothing_is_no_progress_not_budget_exhaustion():
    """The one that went unseen for three runs. Turns climb; the step count does not."""
    now = time.time()
    raised = alarms.evaluate(beats((4, 2, "end_turn", 300), (5, 2, "end_turn", 280),
                                   (6, 2, "end_turn", 310), at=now),
                             now=now, started_at=now - 60)
    assert raised.name == "no-progress"
    assert "flat at 2" in raised.why


def test_reasoning_that_eats_the_reply_budget_is_starved_and_says_so():
    """Narrower than `no-progress` and true at the same time, so it is asked first: it
    was the `max_tokens`-with-no-text pair that made the failure obvious in one look."""
    now = time.time()
    raised = alarms.evaluate(beats((1, 0, "max_tokens", 0), (2, 0, "max_tokens", 0),
                                   (3, 0, "max_tokens", 0), at=now),
                             now=now, started_at=now - 60)
    assert raised.name == "starved"


def test_a_run_that_is_meshing_raises_nothing():
    now = time.time()
    assert alarms.evaluate(beats((1, 1, "end_turn", 200), (2, 2, "end_turn", 200),
                                 (3, 3, "end_turn", 200), at=now),
                           now=now, started_at=now - 60) is None


def test_a_stale_heartbeat_is_wedged_and_a_finished_run_is_not():
    """Wedged is a live process that has stopped saying anything. A process that is gone
    is a run that ended, and its record is where the ending is."""
    now = time.time()
    old = beats((1, 0, "end_turn", 10), at=now - alarms.HEARTBEAT_STALE_S - 1)
    assert alarms.evaluate(old, now=now, started_at=now - 999).name == "wedged"
    assert alarms.evaluate(old, now=now, started_at=now - 999, running=False) is None


def test_a_run_that_has_never_beaten_is_wedged_once_it_is_overdue():
    now = time.time()
    assert alarms.evaluate([], now=now, started_at=now - 10) is None
    assert alarms.evaluate([], now=now, started_at=now - 9_999).name == "wedged"


# -- the watch ------------------------------------------------------------------


def living(monkeypatch):
    """Make the pidfile's pid look alive, without a process to be alive.

    For the tests about *what a live watch records*, as opposed to the ones about killing
    something -- those use a real child. Staging liveness with a real process and then
    killing it leaves a zombie, which `kill -0` still answers for, and a test resting on
    that is a test resting on when the kernel reaps."""
    monkeypatch.setattr(heartbeat, "alive", lambda pid: bool(pid))


class Clock:
    """A fake clock, so a test of a poll loop takes no wall time."""

    def __init__(self, start: float = 1_000.0):
        self.at = start

    def now(self) -> float:
        return self.at

    def sleep(self, seconds: float) -> None:
        self.at += seconds


def test_the_watch_aborts_on_an_alarm_rather_than_letting_the_clock_run_out(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pulse = heartbeat.Heartbeat(tmp_path)
        pulse.start(child.pid)
        for turn in (1, 2, 3):
            pulse.beat(turn=turn, steps=0, stop_reason="max_tokens", text_chars=0)
        seen = supervise.Supervisor(tmp_path, poll_s=0.01).watch()
        assert seen.alarm.name == "starved" and seen.killed
        assert child.wait(timeout=20) != 0
    finally:
        if child.poll() is None:
            child.kill()


def test_the_watch_returns_quietly_when_the_run_ends_on_its_own_terms(tmp_path):
    clock = Clock()
    pulse = heartbeat.Heartbeat(tmp_path)
    pulse.start(4_000_000)
    pulse.beat(steps=1, stop_reason="end_turn", text_chars=100)
    pulse.done()
    seen = supervise.Supervisor(tmp_path, now=clock.now, sleep=clock.sleep).watch()
    assert seen.alarm is None and not seen.killed and seen.steps == 1


def test_the_supervisors_own_deadline_is_an_alarm_not_a_silence(tmp_path):
    """A run that is alive, beating and going nowhere still ends named rather than
    quietly: the supervisor's own deadline is reported as `wedged`, with its reason."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        clock = Clock()
        pulse = heartbeat.Heartbeat(tmp_path)
        pulse.start(child.pid)
        pulse.beat(steps=1, stop_reason="end_turn", text_chars=10)
        watcher = supervise.Supervisor(tmp_path, now=clock.now, sleep=clock.sleep,
                                       poll_s=30.0, stale_s=1e9)
        seen = watcher.watch(deadline_s=120.0)
        assert seen.alarm.name == "wedged" and "deadline" in seen.alarm.why
        assert child.wait(timeout=20) != 0
    finally:
        if child.poll() is None:
            child.kill()


# -- the record -----------------------------------------------------------------


def test_a_run_that_ends_outside_the_set_is_a_bug_here_not_a_result():
    with pytest.raises(ValueError, match="terminal state"):
        record.classify("finished")
    assert record.classify("no-progress") == "no-progress"


def test_the_scalars_are_flat_and_the_trace_is_not(tmp_path):
    """A trace stored under `steps` gets dumped whole by any naive reader, which is
    what happened and what cost the time."""
    record.save(tmp_path, record.Record(run_id="r1", case="T1", n_steps=4,
                                        trace=[{"cell": "print(1)"}]))
    data = record.load(tmp_path)
    assert data["n_steps"] == 4
    assert isinstance(data["trace"], list)
    flat = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
    assert {"n_steps", "n_turns", "first_mesh_step", "mesh_exists", "checkmesh_ok",
            "seconds", "usd", "stopped", "contaminated"} <= set(flat)


def test_every_reply_is_kept_as_it_arrives(tmp_path):
    """Without these the `starved` failure was undiagnosable; with them, one look."""
    record.reply(tmp_path, turn=1, stop_reason="max_tokens", text="", thinking_chars=23_000)
    record.reply(tmp_path, turn=2, stop_reason="end_turn", text="ok")
    kept = record.replies(tmp_path)
    assert [row["turn"] for row in kept] == [1, 2]
    assert kept[0]["thinking_chars"] == 23_000


def test_a_contaminated_run_is_classified_contaminated_whatever_it_thought(tmp_path):
    """It is not a measurement of anything, so its own ending is beside the point."""
    record.save(tmp_path, record.Record(run_id="r2", case="T1", stopped="done"))
    (tmp_path / "cells.log").write_text("ls /work/.toolbox\n", encoding="utf-8")
    data = supervise.Supervisor(tmp_path).observe()
    assert data["contaminated"] and data["stopped"] == "contaminated"
    assert ".toolbox" in data["why"]


def test_an_alarm_outranks_whatever_the_run_had_got_to_recording(tmp_path):
    record.save(tmp_path, record.Record(run_id="r3", case="T1", stopped="time"))
    watch = supervise.Watch(alarm=alarms.Alarm("no-progress", "3 turns, no cells", 6))
    data = supervise.Supervisor(tmp_path).observe(watch=watch)
    assert data["stopped"] == "no-progress" and data["why"] == "3 turns, no cells"
    assert data["watch"]["alarm"]["turn"] == 6


def test_the_probes_are_graded_into_the_record_and_nowhere_the_run_can_read(tmp_path):
    """§2's whole argument, as a test: the probe result goes to the record, and the
    workspace is left exactly as the run left it."""
    workspace = tmp_path / "work"
    case = workspace / "case"
    (case / "constant").mkdir(parents=True)
    before = sorted(p.name for p in case.rglob("*"))
    record.save(tmp_path / "run", record.Record(run_id="r4", case="T1"))
    data = supervise.Supervisor(tmp_path / "run").observe(case_dir=case)
    assert [row["id"] for row in data["probes"]] == [p.id for p in
                                                     __import__("openreynolds.buildup.probes",
                                                                fromlist=["REGISTRY"]).REGISTRY]
    assert sorted(p.name for p in case.rglob("*")) == before


def test_when_the_mesh_first_appeared_is_seen_from_outside_while_the_run_goes(
    tmp_path, monkeypatch
):
    """It cannot be recovered afterwards -- the finished case says only that a mesh
    exists -- and it is the number the arms are compared on. So the supervisor looks at
    the directory on the same lap it checks the pidfile, and the run is never asked."""
    clock = Clock()
    case = tmp_path / "case"
    (case / "constant").mkdir(parents=True)
    run = tmp_path / "run"
    pulse = heartbeat.Heartbeat(run)
    pulse.start(4_000_000)
    pulse.beat(turn=5, steps=4, stop_reason="end_turn", text_chars=100)
    living(monkeypatch)

    watcher = supervise.Supervisor(run, case_dir=case, now=clock.now, sleep=clock.sleep,
                                   stale_s=1e9)
    assert watcher.watch(deadline_s=1.0).first_mesh_step == 0

    (case / "constant" / "polyMesh").mkdir()
    (case / "constant" / "polyMesh" / "points").write_text("4", encoding="utf-8")
    seen = watcher.watch(deadline_s=1.0)
    assert (seen.first_mesh_step, seen.first_mesh_turn) == (4, 5)

    record.save(run, record.Record(run_id="r5", case="T1"))
    assert supervise.Supervisor(run).observe(watch=seen)["first_mesh_step"] == 4


def test_a_conjugate_case_counts_as_meshed_too(tmp_path):
    """The desk this replaces looked for a singular polyMesh and called a two-region case
    'nothing has been meshed yet'."""
    case = tmp_path / "case"
    (case / "constant" / "air" / "polyMesh").mkdir(parents=True)
    assert not supervise.meshed(case)
    (case / "constant" / "air" / "polyMesh" / "points").write_text("4", encoding="utf-8")
    assert supervise.meshed(case)
    assert not supervise.meshed(None)


def test_what_the_watch_saw_reaches_the_grading_without_a_human_carrying_it(
    tmp_path, monkeypatch
):
    """Found by running the two for real: `watch` and `observe` are separate commands in
    separate invocations, and `first_mesh_step` is knowable only to the first. The T1 run
    graded it as 0 while the watch had seen it at 13."""
    clock = Clock()
    case = tmp_path / "case"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "constant" / "polyMesh" / "points").write_text("8", encoding="utf-8")
    run = tmp_path / "run"
    pulse = heartbeat.Heartbeat(run)
    pulse.start(4_000_000)
    pulse.beat(turn=18, steps=13, stop_reason="end_turn", text_chars=100)
    living(monkeypatch)
    supervise.Supervisor(run, case_dir=case, now=clock.now, sleep=clock.sleep,
                         stale_s=1e9).watch(deadline_s=1.0)

    record.save(run, record.Record(run_id="r6", case="T1"))
    graded = supervise.Supervisor(run).observe()          # a second process, told nothing
    assert graded["first_mesh_step"] == 13
    assert graded["watch"]["first_mesh_turn"] == 18
    assert graded["ended_at"]


def test_an_alarm_survives_into_the_grading_run_too(tmp_path):
    """The killed run's ending is in the watch's file, not in the record the run wrote --
    a killed run does not get to write its own ending."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        run = tmp_path / "run"
        pulse = heartbeat.Heartbeat(run)
        pulse.start(child.pid)
        for turn in (1, 2, 3):
            pulse.beat(turn=turn, steps=0, stop_reason="max_tokens", text_chars=0)
        supervise.Supervisor(run, poll_s=0.01).watch()
        record.save(run, record.Record(run_id="r7", case="T1", stopped="time"))
        graded = supervise.Supervisor(run).observe()
        assert graded["stopped"] == "starved"
    finally:
        if child.poll() is None:
            child.kill()


def test_a_watch_started_after_the_run_ended_invents_no_first_mesh_step(tmp_path):
    """It would see a finished case on its first lap and record the mesh as appearing at
    whatever step the run stopped on -- a number nobody measured."""
    case = tmp_path / "case"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "constant" / "polyMesh" / "points").write_text("8", encoding="utf-8")
    run = tmp_path / "run"
    pulse = heartbeat.Heartbeat(run)
    pulse.start(4_000_000)
    pulse.beat(turn=20, steps=18, stop_reason="end_turn", text_chars=10)
    pulse.done()
    assert supervise.Supervisor(run, case_dir=case).watch().first_mesh_step == 0


def test_a_run_cannot_be_watched_twice(tmp_path, monkeypatch):
    """Two watchers evaluate the same alarms, may signal the same child, and both write
    `watch.json` -- so how a run ended would depend on which finished last. Reachable as
    soon as a sweep driver launches watches, since the subagent that drives a single run
    types the same command."""
    run = tmp_path / "run"
    run.mkdir()
    holder = supervise.Supervisor(run)
    holder.claim()
    monkeypatch.setattr(heartbeat, "alive", lambda pid: True)
    monkeypatch.setattr(supervise.os, "getpid", lambda: 999_999)
    with pytest.raises(supervise.Watched, match="already being watched"):
        supervise.Supervisor(run).claim()


def test_the_watch_is_released_when_it_ends_so_the_next_one_can_take_it(tmp_path):
    run = tmp_path / "run"
    pulse = heartbeat.Heartbeat(run)
    pulse.start(4_000_000)
    pulse.beat(turn=1, steps=1, stop_reason="end_turn", text_chars=10)
    supervise.Supervisor(run).watch()
    assert not (run / supervise.WATCHER).exists()
    supervise.Supervisor(run).watch()  # a second, afterwards, is fine


def test_a_stale_watcher_pidfile_does_not_block_the_next_watch(tmp_path):
    """A watcher that was killed leaves its pidfile. The next one takes the run rather
    than refusing forever -- the pid is checked, not the file's existence."""
    run = tmp_path / "run"
    run.mkdir()
    (run / supervise.WATCHER).write_text("4000000\n", encoding="utf-8")
    supervise.Supervisor(run).claim()
    assert (run / supervise.WATCHER).read_text().strip() == str(os.getpid())


# -- the long silences that are not wedged ---------------------------------------


def test_a_run_being_checked_is_not_a_run_that_is_wedged(tmp_path):
    """`checkMesh` per region is up to 600 s with no model turn in it, and the stale
    threshold is 420. Without the loop declaring the silence, the supervisor would kill a
    run for being checked -- and on a conjugate case it would do it twice."""
    now = time.time()
    checking = Beat(turn=12, steps=9, phase="check", expect_s=600.0, at=now - 500)
    assert alarms.evaluate([checking], now=now, started_at=now - 900) is None

    # and it is bounded: silence past what it asked for is still wedged
    stale = Beat(turn=12, steps=9, phase="check", expect_s=600.0, at=now - 800)
    raised = alarms.evaluate([stale], now=now, started_at=now - 900)
    assert raised.name == "wedged" and "during check" in raised.why


def test_a_declared_silence_does_not_excuse_an_undeclared_one():
    """The allowance belongs to the last beat. A turn beat that goes quiet is wedged on
    the ordinary threshold however long the check before it was allowed."""
    now = time.time()
    beats = [Beat(turn=1, steps=1, phase="check", expect_s=900.0, at=now - 900),
             Beat(turn=2, steps=1, phase="turn", at=now - 500)]
    assert alarms.evaluate(beats, now=now, started_at=now - 999).name == "wedged"


def test_liveness_marks_are_not_counted_as_turns_that_ran_nothing(tmp_path):
    """Three `checkMesh` marks in a row are one turn being patient, not three turns
    replying and executing nothing."""
    now = time.time()
    marks = [Beat(turn=7, steps=5, phase="turn", text_chars=300, at=now),
             *(Beat(turn=7, steps=5, phase="check", expect_s=600.0, at=now)
               for _ in range(3))]
    assert alarms.evaluate(marks, now=now, started_at=now - 60) is None


def test_the_desk_declares_the_finish_check_it_is_about_to_run(backend, store):
    """End to end through the loop: the marks reach the watcher, carrying how long the
    check may take, and they do not advance the turn."""
    from openreynolds.backend.base import ExecResult
    from openreynolds.buildup import core
    from test_buildup_core import MESH_OK, core_desk
    from test_cad_agent import answers, block, kernelled
    from openreynolds.cad.brief import CAD_DONE

    answers(backend, {"constant/*/polyMesh": ExecResult(0, "REGION:air\nREGION:solid\n",
                                                        False, None),
                      "checkMesh": ExecResult(0, MESH_OK, False, None)})
    kernelled(backend)
    seen: list[dict] = []
    made = core_desk(backend, store, [block("x = 1"), block(f'print("{CAD_DONE}")')])
    made.on_turn = lambda **fields: seen.append(fields)
    assert made.run("a duct").ok

    marks = [row for row in seen if row.get("phase") == "check"]
    assert len(marks) == 2, "one per region, so a conjugate case refreshes its allowance"
    assert marks[0]["expect_s"] == core.CHECKMESH_TIMEOUT_S
    assert {row["turn"] for row in marks} == {2}, "a mark is liveness, not a turn"


# -- killing what the run started -------------------------------------------------


def test_a_kill_reaches_the_mesher_and_not_the_supervisor(tmp_path):
    """A mesher is a grandchild: the desk runs cells in a kernel and a cell starts
    snappyHexMesh from there. Signalling the runner alone leaves a runaway mesh on the
    machine the next case is about to use."""
    script = ("import os, subprocess, sys, time\n"
              "os.setpgrp()\n"
              f"open({str(tmp_path / 'run.pgid')!r}, 'w').write(str(os.getpgid(0)))\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              f"open({str(tmp_path / 'child.pid')!r}, 'w').write(str(child.pid))\n"
              "time.sleep(60)\n")
    runner = subprocess.Popen([sys.executable, "-c", script])
    try:
        for _ in range(100):
            if (tmp_path / "child.pid").exists():
                break
            time.sleep(0.1)
        grandchild = int((tmp_path / "child.pid").read_text())
        pulse = heartbeat.Heartbeat(tmp_path)
        pulse.start(runner.pid)
        watcher = supervise.Supervisor(tmp_path, poll_s=0.01)
        assert watcher.group() == int((tmp_path / "run.pgid").read_text())
        assert watcher.stop(runner.pid)
        runner.wait(timeout=20)
        for _ in range(100):
            if not heartbeat.alive(grandchild):
                break
            time.sleep(0.1)
        assert not heartbeat.alive(grandchild), "the mesher outlived the run"
        assert heartbeat.alive(os.getpid()), "and the supervisor did not kill itself"
    finally:
        for pid in (runner.pid,):
            if heartbeat.alive(pid):
                os.kill(pid, 9)


def test_a_run_that_took_no_group_is_killed_on_its_own_and_nothing_elses(tmp_path):
    """Deriving a group with `getpgid` would sooner or later name the supervisor's own --
    under a sweep both are children of one driver -- and killing that kills the sweep."""
    run = tmp_path / "run"
    run.mkdir()
    assert supervise.Supervisor(run).group() == 0
    (run / "run.pgid").write_text(f"{os.getpgid(0)}\n", encoding="utf-8")
    assert supervise.Supervisor(run).group() == 0, "never this process's own group"
