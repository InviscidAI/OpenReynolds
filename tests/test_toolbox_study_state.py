"""`study_state.py`: concurrent manifest writes lose no records (#120)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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
    assert not (study / ".reynolds" / "manifest.jsonl.lock").exists()


def test_a_stale_lock_does_not_silence_the_manifest_forever(state, study, monkeypatch):
    monkeypatch.setattr(state, "LOCK_ATTEMPTS", 2)
    monkeypatch.setattr(state, "LOCK_WAIT_S", 0)
    (study / ".reynolds" / "manifest.jsonl.lock").write_text("held by nobody", encoding="utf-8")
    state.record("other", "a.png", root=study)
    state.record_rung(1, "pass", root=study, class_key="c")
    state.record_rung(1, "fail", root=study, class_key="c")
    kinds = [r["kind"] for r in state.artifacts(root=study)]
    assert kinds == ["other", state.RUNG_KIND]
    assert state.artifacts(root=study)[-1]["meta"]["status"] == "fail"


def test_a_lock_this_writer_took_is_removed_afterwards(state, study):
    state.record("other", "a.png", root=study)
    state.record_rung(2, "pass", root=study, class_key="c")
    assert not (study / ".reynolds" / "manifest.jsonl.lock").exists()
