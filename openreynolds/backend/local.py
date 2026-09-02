"""LocalBackend — the same protocol against an OpenFOAM install on this machine.

The hosted workspace is the product; this is the one for working on the product. A
study against the service costs a container-hour and a share of a monthly budget, and
when that budget runs out nothing can be tried at all — which is a poor position to
debug an agent from, because the questions worth asking about an agent are answered by
running it, repeatedly, and watching what it chooses.

So: `subprocess` where the hosted backend has HTTP, a directory where it has a Volume,
and a process table where it has the service's job records. Same protocol, same
semantics, no network and no bill.

Two things it deliberately does not reproduce. There is **no sandbox** — commands run
as the user who started the session, against that user's files, so this is for a
machine whose owner is the one asking. And the workspace root is a real directory
rather than `/work`, since making `/work` needs root: `workspace_root` reports where it
actually is, and `/work` is accepted as an alias for it so that a prompt, a toolbox
destination, or a path the model repeats back all land in the same place either way.

`OPENREYNOLDS_LOCAL_WORK` chooses the root; `OPENREYNOLDS_FOAM_BASHRC` names the
OpenFOAM environment to source, and one is looked for in the usual places when it does
not say.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

from .base import (
    WORKSPACE_ROOT,
    Backend,
    BackendError,
    EXEC_MAX_TIMEOUT_S,
    ExecResult,
    JobStatus,
    Stat,
)

DEFAULT_ROOT = Path.home() / ".openreynolds" / "work"
"""Where the workspace lives when nothing says otherwise."""

MAX_OUTPUT_BYTES = 200_000
"""How much of a command's output comes back inline. The rest stays in its log, which
is the same bargain the hosted backend strikes."""

LOGS_DIR = ".logs"
"""Under the root: where command and job output is kept, so it is on the same volume
the model can read rather than somewhere only this process knows about."""

_BASHRC_GLOBS = (
    "/usr/lib/openfoam/openfoam*/etc/bashrc",
    "/opt/openfoam*/etc/bashrc",
    "/usr/share/openfoam/etc/bashrc",
    str(Path.home() / "OpenFOAM/OpenFOAM-*/etc/bashrc"),
)


def find_bashrc() -> str | None:
    """The OpenFOAM environment to source, if one can be found."""
    named = os.environ.get("OPENREYNOLDS_FOAM_BASHRC")
    if named:
        return named if Path(named).is_file() else None
    for pattern in _BASHRC_GLOBS:
        found = sorted(glob(pattern))
        if found:
            return found[-1]
    return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _Job:
    """One detached command, and what became of it."""

    job_id: str
    name: str | None
    log: Path
    process: subprocess.Popen
    started_at: str
    kill_on: list[re.Pattern] = field(default_factory=list)
    ended_at: str | None = None
    end_reason: str | None = None
    killed_by: str | None = None
    watcher: threading.Thread | None = None


class LocalBackend(Backend):
    """A workspace on this machine."""

    def __init__(self, root: str | Path | None = None, bashrc: str | None = None):
        chosen = root or os.environ.get("OPENREYNOLDS_LOCAL_WORK") or DEFAULT_ROOT
        self.root = Path(chosen).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / LOGS_DIR).mkdir(exist_ok=True)
        self.workspace_root = str(self.root)
        self.bashrc = bashrc if bashrc is not None else find_bashrc()
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self.was_already_running = True
        """Nothing was started to get here, so nothing should be stopped on the way out."""

    # -- shell -----------------------------------------------------------------

    def _wrap(self, cmd: str) -> str:
        """The command with OpenFOAM sourced, when there is an installation to source.

        Sourcing is quiet and non-fatal: a machine without OpenFOAM should still be
        able to run `ls`, and a broken environment should show up as the solver saying
        so rather than as every command failing for a reason nobody can see."""
        if not self.bashrc:
            return cmd
        return f"source {self.bashrc} >/dev/null 2>&1 || true\n{cmd}"

    def _resolve(self, path: str | None) -> Path:
        """A workspace path as a real one, refusing anything outside the root.

        The model is given absolute paths and hands them back; a `..` that climbs out
        of the workspace would be reaching into the machine, which is not what any of
        this is for."""
        if not path:
            return self.root
        # `/work` is what the protocol advertises and what the frozen prompt says, and
        # this machine's root is somewhere else entirely. Treating the advertised root
        # as an alias for the real one means everything above the protocol -- the
        # prompt, the toolbox destination, a path the model repeats back -- keeps
        # working without knowing which backend it has, which is the whole contract.
        text = str(path)
        if text == WORKSPACE_ROOT or text.startswith(WORKSPACE_ROOT + "/"):
            text = str(self.root) + text[len(WORKSPACE_ROOT):]
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = Path(os.path.normpath(str(candidate)))
        if resolved != self.root and self.root not in resolved.parents:
            raise BackendError(
                f"{path} is outside the workspace ({self.root})", "outside_workspace"
            )
        return resolved

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120) -> ExecResult:
        limit = min(int(timeout_s or 120), EXEC_MAX_TIMEOUT_S)
        where = self._resolve(cwd)
        where.mkdir(parents=True, exist_ok=True)
        log = self.root / LOGS_DIR / f"exec-{uuid.uuid4().hex[:12]}.log"
        try:
            done = subprocess.run(
                ["bash", "-lc", self._wrap(cmd)],
                cwd=str(where), capture_output=True, timeout=limit,
            )
            blob = done.stdout + done.stderr
            code = done.returncode
        except subprocess.TimeoutExpired as expired:
            blob = (expired.stdout or b"") + (expired.stderr or b"")
            # The hosted backend reports a timeout as a fact in the output rather than
            # as a raised error, because the command did run and what it printed first
            # is often the whole answer.
            blob += f"\n[timed out after {limit}s]".encode()
            code = 124
        except OSError as exc:
            raise BackendError(str(exc), "exec_failed") from exc
        log.write_bytes(blob)
        text = blob[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")
        return ExecResult(code, text, len(blob) > MAX_OUTPUT_BYTES, str(log))

    # -- files -----------------------------------------------------------------

    def put_file(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def get_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        target = self._resolve(path)
        try:
            with open(target, "rb") as handle:
                handle.seek(max(0, int(offset)))
                return handle.read() if limit is None else handle.read(int(limit))
        except FileNotFoundError as exc:
            raise BackendError(f"no such file: {path}", "not_found") from exc
        except IsADirectoryError as exc:
            raise BackendError(f"{path} is a directory", "is_a_directory") from exc

    def stat(self, path: str) -> Stat:
        target = self._resolve(path)
        try:
            info = target.stat()
        except FileNotFoundError as exc:
            raise BackendError(f"no such file: {path}", "not_found") from exc
        if target.is_dir():
            entries = sorted(child.name for child in target.iterdir())
            return Stat(str(target), "directory", info.st_size, int(info.st_mtime), entries)
        kind = "symbolic link" if target.is_symlink() else "regular file"
        return Stat(str(target), kind, info.st_size, int(info.st_mtime))

    def put_tree(self, local_dir: Path, remote_dir: str) -> None:
        target = self._resolve(remote_dir)
        shutil.copytree(Path(local_dir), target, dirs_exist_ok=True)

    def get_tree(self, remote_paths: list[str], local_dir: Path) -> list[Path]:
        out: list[Path] = []
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for path in remote_paths:
            source = self._resolve(path)
            if not source.exists():
                continue
            landing = destination / source.name
            if source.is_dir():
                shutil.copytree(source, landing, dirs_exist_ok=True)
                out.extend(p for p in landing.rglob("*") if p.is_file())
            else:
                landing.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, landing)
                out.append(landing)
        return out

    # -- jobs ------------------------------------------------------------------

    def job_start(
        self,
        cmd: str,
        cwd: str | None = None,
        name: str | None = None,
        kill_on: list[str] | None = None,
    ) -> str:
        where = self._resolve(cwd)
        where.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        log = self.root / LOGS_DIR / f"job-{job_id[:12]}.log"
        log.touch()
        try:
            handle = open(log, "ab")
            process = subprocess.Popen(
                ["bash", "-lc", self._wrap(cmd)],
                cwd=str(where), stdout=handle, stderr=subprocess.STDOUT,
                # Its own process group, so killing the job reaches the whole tree.
                # An `mpirun` puts its ranks below it and a signal to the shell alone
                # leaves them running, which is how a solve outlives the thing that
                # was supposed to have stopped it.
                start_new_session=True,
            )
        except OSError as exc:
            raise BackendError(str(exc), "job_failed") from exc
        job = _Job(
            job_id=job_id, name=name, log=log, process=process, started_at=_now(),
            kill_on=[re.compile(p) for p in (kill_on or [])],
        )
        with self._lock:
            self._jobs[job_id] = job
        if job.kill_on:
            job.watcher = threading.Thread(target=self._watch, args=(job,), daemon=True)
            job.watcher.start()
        return job_id

    def _watch(self, job: _Job) -> None:
        """Kill the job when its log says the thing it was told to watch for."""
        seen = 0
        while job.process.poll() is None:
            try:
                with open(job.log, "rb") as handle:
                    handle.seek(seen)
                    fresh = handle.read()
                    seen = handle.tell()
            except OSError:
                return
            for line in fresh.decode("utf-8", "replace").splitlines():
                if any(pattern.search(line) for pattern in job.kill_on):
                    job.killed_by = line[:500]
                    job.end_reason = "kill_on_match"
                    self._terminate(job, "TERM")
                    return
            time.sleep(1.0)

    def _terminate(self, job: _Job, sig: str) -> None:
        number = signal.SIGKILL if sig.upper() in ("KILL", "9") else signal.SIGTERM
        try:
            os.killpg(os.getpgid(job.process.pid), number)
        except (ProcessLookupError, PermissionError):
            pass

    def _job(self, job_id: str) -> _Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise BackendError(f"no such job: {job_id}", "not_found")
        return job

    def job_status(self, job_id: str) -> JobStatus:
        job = self._job(job_id)
        code = job.process.poll()
        if code is None:
            status, reason = "running", None
        else:
            if job.ended_at is None:
                job.ended_at = _now()
                job.end_reason = job.end_reason or (
                    "completed" if code == 0 else "killed_by_client" if code < 0 else "completed"
                )
            status = "exited" if code >= 0 and job.end_reason == "completed" else "killed"
            reason = job.end_reason
        return JobStatus(
            job_id=job_id, status=status, name=job.name,
            exit_code=code, end_reason=reason,
            started_at=job.started_at, ended_at=job.ended_at,
            log_size=job.log.stat().st_size if job.log.exists() else 0,
            killed_by=job.killed_by,
        )

    def job_tail(self, job_id: str, offset: int = 0) -> tuple[str, int, bool]:
        job = self._job(job_id)
        with open(job.log, "rb") as handle:
            handle.seek(max(0, int(offset)))
            blob = handle.read(MAX_OUTPUT_BYTES)
            where = handle.tell()
            eof = handle.read(1) == b""
        return blob.decode("utf-8", "replace"), where, eof

    def job_kill(self, job_id: str, signal: str = "TERM") -> JobStatus:
        job = self._job(job_id)
        if job.process.poll() is None:
            job.end_reason = "killed_by_client"
            self._terminate(job, signal)
            try:
                job.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._terminate(job, "KILL")
        return self.job_status(job_id)

    # -- lifecycle -------------------------------------------------------------

    def shutdown(self) -> None:
        """Nothing to stop: the workspace is a directory, and it stays."""

    def close(self) -> None:
        """Jobs outlive the session by design, so this only lets go of the handles."""
        with self._lock:
            self._jobs.clear()
