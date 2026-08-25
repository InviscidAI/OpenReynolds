"""The local mirror: `./studies/<id>/`.

Holds session metadata, the message log, and everything fetched back. Job records are
not bookkeeping niceties — the service has no list-jobs endpoint, so without them a
resumed session cannot tell the model what is still running.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def new_study_id() -> str:
    """A short, sortable, human-typable id."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


@dataclass
class JobRecord:
    job_id: str
    name: str | None = None
    cmd: str = ""
    launched_at: str = ""
    status: str = "running"
    end_reason: str | None = None
    exit_code: int | None = None
    log_offset: int = 0
    cwd: str = ""
    """Where the command ran. The progress line finds the case's controlDict from it;
    records written before it existed load with it empty and are shown without an end."""
    """How far the model has already read this job's log."""


@dataclass
class Session:
    study_id: str
    instance_id: str = ""
    remote_study_id: str = ""
    """Id assigned by the capture plane, when capture is on."""
    model: str = ""
    home: str = ""
    """This study's own directory in the workspace.

    Empty on studies made before studies had one, which means the whole workspace.
    """
    created_at: str = ""
    title: str = ""
    capture_seq: int = 0
    jobs: dict[str, JobRecord] = field(default_factory=dict)


class Store:
    """One study directory on the user's machine."""

    def __init__(self, root: Path, study_id: str):
        self.dir = root / study_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.dir / "files"
        self.renders_dir = self.dir / "renders"
        """The flat, obvious home for pictures. The mirror copies every render and
        assembled animation here, newest-first, so "where is the image?" has a
        one-word answer that does not involve the nested `files/<id>/.../renders`
        path the workspace happens to use."""
        self.session = Session(study_id=study_id, created_at=_now())
        self._load()

    # -- persistence -----------------------------------------------------------

    @property
    def _session_path(self) -> Path:
        return self.dir / "session.json"

    @property
    def _messages_path(self) -> Path:
        return self.dir / "messages.jsonl"

    def _load(self) -> None:
        if not self._session_path.exists():
            return
        try:
            raw = json.loads(self._session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        jobs = {jid: JobRecord(**rec) for jid, rec in (raw.pop("jobs", {}) or {}).items()}
        known = {f.name for f in Session.__dataclass_fields__.values()}
        self.session = Session(**{k: v for k, v in raw.items() if k in known}, jobs=jobs)

    def save(self) -> None:
        payload = asdict(self.session)
        self._session_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # -- messages --------------------------------------------------------------

    def append_message(self, role: str, content: Any) -> int:
        """Mirror one message locally and hand back its sequence number.

        The capture plane does not assign sequence numbers, so this counter is the
        single source of ordering for both the local log and the upload.
        """
        seq = self.session.capture_seq
        self.session.capture_seq += 1
        row = {"seq": seq, "role": role, "at": _now(), "content": content}
        with self._messages_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self.save()
        return seq

    # -- jobs ------------------------------------------------------------------

    def recent_messages(self, limit: int = 30) -> list[dict[str, Any]]:
        """The last few transcript rows, oldest-first.

        The concierge reads these to answer the user without a turn from the main
        thread. Append-only, so reading the tail never races an appending writer:
        a half-written final line is dropped rather than raised on."""
        path = self._messages_path
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows = []
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def record_job(self, job_id: str, cmd: str, name: str | None, cwd: str = "") -> JobRecord:
        record = JobRecord(job_id=job_id, name=name, cmd=cmd, launched_at=_now(), cwd=cwd)
        self.session.jobs[job_id] = record
        self.save()
        return record

    def update_job(self, job_id: str, **changes: Any) -> JobRecord | None:
        record = self.session.jobs.get(job_id)
        if record is None:
            return None
        for key, value in changes.items():
            if hasattr(record, key):
                setattr(record, key, value)
        self.save()
        return record

    def live_jobs(self) -> list[JobRecord]:
        return [job for job in self.session.jobs.values() if job.status == "running"]

    # -- fetched files ---------------------------------------------------------

    def fetch_dir(self) -> Path:
        self.files_dir.mkdir(parents=True, exist_ok=True)
        return self.files_dir


def list_studies(root: Path) -> list[Session]:
    """Every local study, newest first."""
    if not root.exists():
        return []
    sessions = []
    for child in sorted(root.iterdir(), reverse=True):
        if (child / "session.json").is_file():
            sessions.append(Store(root, child.name).session)
    return sessions


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
