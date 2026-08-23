from __future__ import annotations

from pathlib import Path

import pytest

from openreynolds.backend.base import Backend, BackendError, ExecResult, JobStatus, Stat
from openreynolds.store import Store
from openreynolds.tools import ToolContext


class FakeBackend(Backend):
    """An in-memory workspace. No network, no service."""

    workspace_root = "/work"

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.dirs: dict[str, list[str]] = {}
        self.exec_result = ExecResult(0, "ok", False, None)
        self.jobs: dict[str, JobStatus] = {}
        self.logs: dict[str, bytes] = {}
        self.started: list[dict] = []
        self.trees: list[tuple[Path, str]] = []
        self.fetched: list[str] = []

    def exec(self, cmd, cwd=None, timeout_s=120):
        self.last_exec = (cmd, cwd, timeout_s)
        return self.exec_result

    def put_file(self, path, data):
        self.files[path] = data

    def get_file(self, path, offset=0, limit=None):
        if path not in self.files:
            raise BackendError(f"no such path: {path}", code="not_found", status=404)
        data = self.files[path][offset:]
        return data[:limit] if limit is not None else data

    def stat(self, path):
        if path in self.dirs:
            return Stat(path, "directory", 0, 0, self.dirs[path])
        if path not in self.files:
            raise BackendError(f"no such path: {path}", code="not_found", status=404)
        return Stat(path, "regular file", len(self.files[path]), 0, [])

    def put_tree(self, local_dir, remote_dir):
        self.trees.append((local_dir, remote_dir))

    def get_tree(self, remote_paths, local_dir):
        self.fetched.extend(remote_paths)
        written = []
        for remote in remote_paths:
            target = local_dir / Path(remote).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.files.get(remote, b"x"))
            written.append(target)
        return written

    def job_start(self, cmd, cwd=None, name=None, kill_on=None):
        job_id = f"job-{len(self.jobs) + 1}"
        self.started.append({"cmd": cmd, "cwd": cwd, "name": name, "kill_on": kill_on})
        self.jobs[job_id] = JobStatus(job_id=job_id, status="running", name=name)
        self.logs[job_id] = b""
        return job_id

    def job_status(self, job_id):
        return self.jobs[job_id]

    def job_tail(self, job_id, offset=0):
        data = self.logs.get(job_id, b"")
        chunk = data[offset:]
        return chunk.decode("utf-8", "replace"), offset + len(chunk), True

    def job_kill(self, job_id):
        current = self.jobs[job_id]
        self.jobs[job_id] = JobStatus(
            job_id=job_id, status="killed", name=current.name, end_reason="killed_by_client"
        )
        return self.jobs[job_id]

    def close(self):
        pass


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "studies", "study-test")


@pytest.fixture
def ctx(backend, store):
    return ToolContext(backend=backend, store=store, max_output=1000)
