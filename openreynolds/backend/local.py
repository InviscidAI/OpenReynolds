"""LocalBackend — deferred.

The same protocol over `subprocess` against a sourced OpenFOAM environment, with jobs
run through the same detached-wrapper shape. Not implemented yet; the only obligation
v1 carries is negative, and it belongs to the *other* modules: nothing above
`backend/base.py` may assume HTTP or a particular service.
"""

from __future__ import annotations

from pathlib import Path

from .base import Backend, ExecResult, JobStatus, Stat


class LocalBackend(Backend):
    """Placeholder for a local OpenFOAM install."""

    workspace_root = "/work"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "LocalBackend is not implemented yet — run against the hosted service."
        )

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120) -> ExecResult: ...

    def put_file(self, path: str, data: bytes) -> None: ...

    def get_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes: ...

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

    def job_tail(self, job_id: str, offset: int = 0) -> tuple[str, int, bool]: ...

    def job_kill(self, job_id: str, signal: str = "TERM") -> JobStatus: ...

    def close(self) -> None: ...
