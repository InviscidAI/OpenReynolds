"""`study_state.py`: concurrent manifest writes lose no records (#120)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"

WRITERS = 6
PER_WRITER = 150

# Every tenth write is a rung record, which rewrites the manifest.
WRITER = """
import sys
sys.path.insert(0, sys.argv[1])
import study_state
root, who, count = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for i in range(count):
    if i % 10 == 0:
        study_state.record_rung(i, "pass", root=root, class_key=f"w{who}", name="r")
    else:
        study_state.record("other", f"w{who}-{i}.png", root=root, label=f"{who}:{i}")
"""


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state():
    return load("study_state")


@pytest.fixture
def study(tmp_path):
    (tmp_path / ".reynolds").mkdir()
    return tmp_path


def manifest_lines(study: Path) -> list[str]:
    return (study / ".reynolds" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()


def test_concurrent_writers_lose_no_records_and_cut_no_lines(state, study):
    procs = [
        subprocess.Popen([sys.executable, "-c", WRITER, str(TOOLBOX), str(study),
                          str(who), str(PER_WRITER)])
        for who in range(WRITERS)
    ]
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    lines = manifest_lines(study)
    for line in lines:
        json.loads(line)
    rows = state.artifacts(root=study)
    plain = [r for r in rows if r["kind"] == "other"]
    rungs = [r for r in rows if r["kind"] == state.RUNG_KIND]
    assert len(plain) == WRITERS * (PER_WRITER - PER_WRITER // 10)
    assert len({r["label"] for r in plain}) == len(plain)
    assert len(rungs) == WRITERS * (PER_WRITER // 10)


def test_a_lock_file_left_on_disk_costs_no_wait(state, study):
    # The lock is the OS's, not the file's existence: a leftover file is not held.
    lock = study / ".reynolds" / "manifest.jsonl.lock"
    lock.write_text("left by a writer that crashed", encoding="utf-8")
    started = time.perf_counter()
    state.record("other", "a.png", root=study)
    state.record_rung(1, "pass", root=study, class_key="c")
    state.record_rung(1, "fail", root=study, class_key="c")
    assert time.perf_counter() - started < 1.0
    kinds = [r["kind"] for r in state.artifacts(root=study)]
    assert kinds == ["other", state.RUNG_KIND]
    assert state.artifacts(root=study)[-1]["meta"]["status"] == "fail"


# Takes the manifest lock, says so, then writes a row after HOLD_S unless killed first.
HOLDER = """
import sys, time
sys.path.insert(0, sys.argv[1])
import study_state
from pathlib import Path
study = Path(sys.argv[2])
with study_state._manifest_lock(study):
    print("held", flush=True)
    time.sleep(float(sys.argv[3]))
    study_state._append_row(study, {"kind": "holder", "path": "h.png"})
"""


def hold_lock(study: Path, seconds: float) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(TOOLBOX), str(study), str(seconds)],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_a_writer_waits_for_the_lock_holder(state, study):
    holder = hold_lock(study, 0.5)
    state.record("other", "a.png", root=study)
    assert holder.wait(timeout=30) == 0
    assert [r["kind"] for r in state.artifacts(root=study)] == ["holder", "other"]


def test_a_lock_held_by_a_killed_writer_costs_no_wait(state, study):
    holder = hold_lock(study, 60)
    holder.kill()
    holder.wait(timeout=30)
    started = time.perf_counter()
    state.record("other", "a.png", root=study)
    state.record_rung(1, "pass", root=study, class_key="c")
    assert time.perf_counter() - started < 1.0
    assert [r["kind"] for r in state.artifacts(root=study)] == ["other", state.RUNG_KIND]
