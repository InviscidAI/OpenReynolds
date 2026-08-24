from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest
from rich.console import Console

from openreynolds.backend.base import (
    WORKSPACE_ROOT,
    Backend,
    BackendError,
    ExecResult,
    JobStatus,
    Stat,
)
from openreynolds.store import Store
from openreynolds.tools import ToolContext
from openreynolds.view import View


class FakeBackend(Backend):
    """An in-memory workspace. No network, no service."""

    workspace_root = "/work"

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.dirs: dict[str, list[str]] = {}
        self.exec_result = ExecResult(0, "ok", False, None)
        self.exec_results: dict[str, ExecResult] = {}
        self.execs: list[str] = []
        self.jobs: dict[str, JobStatus] = {}
        self.logs: dict[str, bytes] = {}
        self.started: list[dict] = []
        self.trees: list[tuple[Path, str]] = []
        self.fetched: list[str] = []

    def exec(self, cmd, cwd=None, timeout_s=120):
        self.last_exec = (cmd, cwd, timeout_s)
        self.execs.append(cmd)
        return self.exec_results.get(cmd, self.exec_result)

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
        """Mirrors the service: archive members are relative to the workspace root,
        so a fetched file keeps the shape it had in the workspace."""
        self.fetched.extend(remote_paths)
        written = []
        for remote in remote_paths:
            relative = remote[len(WORKSPACE_ROOT) + 1 :] if remote.startswith(
                WORKSPACE_ROOT + "/"
            ) else Path(remote).name
            target = local_dir / relative
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

    def job_kill(self, job_id, signal="TERM"):
        self.kill_signals = getattr(self, "kill_signals", [])
        self.kill_signals.append(signal)
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


# -- a fake model --------------------------------------------------------------


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, tool_input: dict, block_id: str = "tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def message(content, stop_reason="end_turn", input_tokens=100):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=10,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self._response


class FakeMessages:
    """Replays scripted turns; repeats the last one if the loop asks for more."""

    def __init__(self, responses, fail_on_system=False):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.fail_on_system = fail_on_system

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_system and any(m["role"] == "system" for m in kwargs["messages"]):
            raise anthropic.BadRequestError(
                "role 'system' is not supported on this model",
                response=SimpleNamespace(status_code=400, headers={}, request=None),
                body=None,
            )
        if len(self._responses) > 1:
            return FakeStream(self._responses.pop(0))
        return FakeStream(self._responses[0])


def install_model(loop, responses, fail_on_system=False):
    fake = FakeMessages(responses, fail_on_system=fail_on_system)
    loop.client = SimpleNamespace(messages=fake)
    return fake


@pytest.fixture
def console():
    return Console(file=open(os.devnull, "w"), force_terminal=False)


class ScriptedReader:
    """Stands in for stdin: hands back queued lines, then EOF."""

    def __init__(self, lines):
        self._lines = list(lines)

    def poll(self):
        from openreynolds.watch import NOTHING

        return self._lines.pop(0) if self._lines else NOTHING

    def get(self, timeout=None):
        return self._lines.pop(0) if self._lines else None

    def putback(self, line):
        self._lines.insert(0, line)


@pytest.fixture
def fast_polling(monkeypatch):
    monkeypatch.setattr("openreynolds.watch.POLL_MIN_S", 0.001)
    monkeypatch.setattr("openreynolds.watch.POLL_MAX_S", 0.002)
    monkeypatch.setattr("openreynolds.watch.TICK_S", 0.001)


class RecordingView(View):
    """A view that keeps what it was told, so tests can assert on it."""

    def __init__(self):
        self.headers = []
        self.text = []
        self.thinking = []
        self.tools = []
        self.tool_errors = []
        self.notices = []
        self.warnings = []
        self.infos = []
        self.usages = []
        self.watched = []
        self.prompts = 0
        self.job_reports = []
        self.stages = []
        self.steps = []
        self.interjections = []
        self.statuses = []
        self.listings = []
        self.browser = None

    def header(self, study_id, instance_id, model, mirror):
        self.headers.append((study_id, instance_id, model, mirror))

    def thinking_begin(self):
        self.thinking.append("<begin>")

    def thinking_delta(self, text):
        self.thinking.append(text)

    def text_delta(self, text):
        self.text.append(text)

    def turn_end(self):
        self.text.append("<end>")

    def tool(self, name, summary):
        self.tools.append((name, summary))

    def tool_error(self, message):
        self.tool_errors.append(message)

    def notice(self, message):
        self.notices.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)

    def usage(self, tokens, fraction):
        self.usages.append((tokens, fraction))

    def watching(self, names):
        self.watched.append(list(names))

    def jobs(self, records):
        self.job_reports.append(list(records))

    def stage(self, text):
        self.stages.append(text)

    def step(self, number, seconds, tool_calls):
        self.steps.append((number, tool_calls))

    def interjection(self, text):
        self.interjections.append(text)

    def workspace(self, browser):
        self.browser = browser

    def show_files(self, path=""):
        self.listings.append(path)

    def status(self, lines):
        self.statuses.append(list(lines))

    def prompt(self):
        self.prompts += 1

    @property
    def said(self) -> str:
        return "".join(t for t in self.text if not t.startswith("<"))


@pytest.fixture
def view():
    return RecordingView()
