"""The Backend protocol — the whole independence story.

Nothing above this interface may know whether it is talking to a container over the
network or to a local OpenFOAM install. Concretely: no module that imports this one may
name a transport, a URL, or a particular hosting service. Only the implementation
modules in this package may. `tests/test_negative_obligation.py` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

WORKSPACE_ROOT = "/work"
"""The persistent directory every backend presents to the model."""

EXEC_MAX_TIMEOUT_S = 300
"""Ceiling the protocol advertises for a synchronous command. Backends may enforce it;
work that outlives it belongs in a job."""


@dataclass(frozen=True)
class ExecResult:
    """Outcome of a synchronous command."""

    exit_code: int
    output: str
    """Combined stdout+stderr, already capped by the backend."""
    truncated: bool
    """True when `output` is only part of what the command produced."""
    log_path: str | None
    """Where the complete output lives in the workspace, when the backend keeps one."""


@dataclass(frozen=True)
class Stat:
    """Metadata for one workspace path."""

    path: str
    type: str
    """e.g. "regular file", "directory", "symbolic link"."""
    size: int
    mtime: int
    entries: list[str] = field(default_factory=list)
    """Immediate child names; populated only for directories."""

    @property
    def is_dir(self) -> bool:
        return "directory" in self.type


@dataclass(frozen=True)
class JobStatus:
    """State of a detached job."""

    job_id: str
    status: str
    """running | exited | killed"""
    name: str | None = None
    exit_code: int | None = None
    end_reason: str | None = None
    """completed | failed | killed_externally | kill_on_match | killed_by_client | sandbox_expired

    Only `completed` means the work finished on its own terms. `failed` is the solver
    stopping (a FOAM FATAL, a bad dictionary); `killed_externally` is a 137 or a 143 --
    something outside the job acted, and nothing the job did caused it. They used to all
    read `completed`, which left the exit code as the only honest signal there was."""
    started_at: str | None = None
    ended_at: str | None = None
    log_size: int | None = None
    killed_by: str | None = None
    """The log line that matched a `kill_on` pattern, when one did."""

    @property
    def running(self) -> bool:
        return self.status == "running"


@dataclass(frozen=True)
class ResizeResult:
    """Outcome of a workspace resize request."""
    success: bool
    """True if resize succeeded."""
    new_cost_per_hour: int | None = None
    """New cost in cents/hour, when success=True."""
    error: str | None = None
    """Error message, when success=False."""

class BackendError(Exception):
    """Any failure reaching or acting on the workspace.

    `code` is a short machine-readable token; `message` is human text. Both are shown
    to the model verbatim in a tool_result, so they should read as facts.
    """

    def __init__(self, message: str, code: str = "backend_error", status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def __str__(self) -> str:
        prefix = f"{self.code}"
        if self.status is not None:
            prefix += f" ({self.status})"
        return f"{prefix}: {self.message}"


@runtime_checkable
class Backend(Protocol):
    """A Linux workspace with OpenFOAM in it."""

    workspace_root: str
    """Absolute path of the persistent directory, e.g. "/work"."""

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120) -> ExecResult: ...

    def put_file(self, path: str, data: bytes) -> None: ...

    def get_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        """Bytes from `path`, starting at `offset`.

        `limit` is not optional in the way it looks. A backend is free to answer with
        less than everything, and the hosted one does: asked for a file with no limit
        it returns its own page size and says nothing about the rest. A caller that
        wants a whole file has to `stat` it and ask for that many bytes. Leaving this
        unwritten cost a live bug -- renders between the page size and the attachment
        ceiling reached the model truncated, and a truncated PNG is not a smaller
        picture, it is a broken one.
        """
        ...

    def stat(self, path: str) -> Stat: ...

    def put_tree(self, local_dir: Path, remote_dir: str) -> None: ...

    def get_tree(self, remote_paths: list[str], local_dir: Path) -> list[Path]: ...

    def job_start(
        self,
        cmd: str,
        cwd: str | None = None,
        name: str | None = None,
        kill_on: list[str] | None = None,
    ) -> str: ...

    def job_status(self, job_id: str) -> JobStatus: ...

    def job_tail(self, job_id: str, offset: int = 0) -> tuple[str, int, bool]:
        """Return (data, next_offset, eof). `next_offset` is a byte offset, safe to feed back."""
        ...

    def job_kill(self, job_id: str, signal: str = "TERM") -> JobStatus: ...

    
    def current_workspace_size(self) -> tuple[float, int]:
        """Return (cpu, mem_gb) of the current instance."""
        ...

    def estimate_resize_cost(
        self, from_cpu: float, from_mem_gb: int, to_cpu: float, to_mem_gb: int
    ) -> int:
        """Estimate cost delta in cents/hour for a resize.

        Returns the difference in cost between current and requested size.
        """
        ...

    def can_afford(self, cost_delta_cents: int) -> bool:
        """Check if a cost delta fits within monthly budget."""
        ...

    def resize_workspace(self, cpu: float, mem_gb: int, reason: str | None) -> ResizeResult:
        """Request a workspace resize.

        Returns ResizeResult with success status and cost or error message.
        """
        ...    

    def close(self) -> None: ...
