"""LocalBackend: the same protocol against this machine, OpenFOAM or not.

The hosted workspace is the product; this is the one for working on the product. A
study against the service costs a container-hour and a share of a monthly budget, and
when that budget runs out nothing can be tried at all, which is a poor position to
debug an agent from, because the questions worth asking about an agent are answered by
running it, repeatedly, and watching what it chooses.

So: `subprocess` where the hosted backend has HTTP, a directory where it has a Volume,
and a process table where it has the service's job records. Same protocol, same
semantics, no network and no bill.

Two things it deliberately does not reproduce. There is **no sandbox** -- commands run
as the user who started the session, against that user's files, so this is for a
machine whose owner is the one asking. And the workspace root is a real directory
rather than `/work`, since making `/work` needs root: `workspace_root` reports where it
actually is, and `/work` is an alias for it everywhere, in file calls through
`_resolve` and in command text through `_wrap`, so that a prompt, a toolbox
destination, or a path the model repeats back all land in the same place either way.

Three things the machine may be missing, and this module answers all three rather than
letting the agent discover them one failed command at a time (`what_is_missing`):

* **A POSIX shell.** Every command the agent writes is POSIX, and so is every `Allrun`
  in `$FOAM_TUTORIALS`. Without one, nothing runs at all.
* **OpenFOAM.** Then its meshers and solvers fail, and nothing else does: `lbm.py` and
  `sph.py` are numpy and solve without it, which is what makes a laptop with nothing
  installed useful rather than merely startable.
* **`python3` under that name.** On Windows it is usually the Microsoft Store's
  execution alias, a real file on `PATH` that exits 49 saying "Python was not found",
  on machines whose `python` runs fine. Every toolbox script is documented as `python3
  x.py`, so `_wrap` defines the name where it does not already work.

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
import sys
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
from .kernel import KernelHost

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


SHELL = "bash"
"""What a command is run by. POSIX, because every command the agent writes is POSIX
and so is every `Allrun` in `$FOAM_TUTORIALS`: `cd case && blockMesh && icoFoam`,
pipes, `&&`, `source`. On Windows that means Git for Windows, which ships one."""


def shell_found() -> bool:
    """Whether there is a shell to run anything in.

    Separate from "is OpenFOAM here", because the two absences want different
    sentences. No OpenFOAM leaves the numpy solvers and every analysis script
    working; no shell leaves nothing working at all, and on Windows that is the
    ordinary case rather than a broken install."""
    return shutil.which(SHELL) is not None


_WORK = re.compile(re.escape(WORKSPACE_ROOT) + r"(?![\w-])")
"""`/work` where it is a path and not the start of a longer word, so `/workspace`
and `/working` are left alone."""


def shell_root(root: Path) -> str:
    """The workspace root in a form a POSIX shell will accept as one word.

    Forward slashes because Git Bash takes `C:/Users/...` and hands it to Windows
    unchanged, where a backslash would be an escape. No quoting, because the
    substitution lands mid-token -- `/work/case/log` becomes `<root>/case/log` --
    and a quote opened there would have to be closed somewhere this function cannot
    see.

    That makes a space in the root unrepresentable, which is not hypothetical on
    Windows: `C:\\Users\\John Smith` is an ordinary home directory. Windows keeps an
    8.3 name for exactly this, so the short name is used when there is a space, and
    `__init__` refuses a root that still has one rather than writing a command that
    will be split into two words."""
    text = str(root).replace(chr(92), "/")
    if " " in text and os.name == "nt":
        short = _short_path(root)
        if short:
            text = short.replace(chr(92), "/")
    return text


def _short_path(path: Path) -> str | None:
    """A Windows 8.3 path for `path`, or None where the volume has them disabled.

    `GetShortPathNameW` answers 0 on failure and, when the buffer is too small, the
    length it wants; both are handled rather than assumed away."""
    try:
        import ctypes
        from ctypes import wintypes
    except (ImportError, ValueError):   # pragma: no cover -- not Windows
        return None
    try:
        call = ctypes.windll.kernel32.GetShortPathNameW
    except (AttributeError, OSError):   # pragma: no cover -- not Windows
        return None
    call.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    call.restype = wintypes.DWORD
    wanted = call(str(path), None, 0)
    if not wanted:
        return None
    buffer = ctypes.create_unicode_buffer(wanted)
    if not call(str(path), buffer, wanted):
        return None
    return buffer.value or None


def _python3_works() -> bool:
    """Whether `python3` is a working interpreter here, established by running one.

    `shutil.which` cannot answer this. On Windows `python3` is usually the Microsoft
    Store's execution alias: a real file, on `PATH`, which exits 49 printing "Python
    was not found" even on a machine whose `python` runs and imports numpy. So the
    question is answered by asking an interpreter to start.

    Cached for the process -- it costs a subprocess and cannot change under a
    session. False on any failure, including a timeout, because the fallback works
    either way."""
    global _PYTHON3_OK
    if _PYTHON3_OK is None:
        try:
            _PYTHON3_OK = subprocess.run(
                ["python3", "-c", "import sys"], capture_output=True, timeout=20,
            ).returncode == 0
        except (OSError, subprocess.SubprocessError):
            _PYTHON3_OK = False
    return _PYTHON3_OK


_PYTHON3_OK: bool | None = None


def _python3_line() -> str:
    """A `python3` for a machine that has Python under another name, or not at all.

    This is not cosmetic. Every toolbox script is documented as `python3 x.py` and
    `tests/test_toolbox.py:338,342` requires that literal in each docstring, so
    `advise.py`, every template and every recipe the agent reads all say `python3`.
    Without this line the whole toolbox is unreachable on a Windows laptop --
    including `lbm.py` and `sph.py`, the two solvers that need no OpenFOAM and are
    therefore the only reason a machine with nothing installed can solve anything.

    A shell function and not a shim directory on `PATH`: a Windows path contains the
    colon that POSIX `PATH` uses as its separator, so prepending one is a trap.
    `export -f` carries the definition into child shells, which is as far as an
    `Allrun` or a `Makefile` reaches.

    Empty where `python3` already works, so a Linux or macOS machine runs exactly
    the string the agent wrote and this function is a no-op off Windows."""
    if _python3_works():
        return ""
    exe = (sys.executable or "").replace("\\", "/")
    if not exe or '"' in exe:
        # Nothing to point at, or a path that cannot be quoted into a shell
        # function. The honest outcome is the command failing as it would have,
        # rather than a mangled definition failing in a new way.
        return ""
    return 'python3() { "' + exe + '" "$@"; }\nexport -f python3\n'


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


def what_is_missing(backend: "LocalBackend") -> list[str]:
    """What this machine cannot do, and what it can do anyway, said once at the start.

    Three separate absences, and the old single sentence conflated two of them.

    **No shell.** Nothing runs at all, so this is the one worth leading with. It is
    the ordinary case on Windows rather than a broken install, and Git for Windows
    is the usual way to get one.

    **No OpenFOAM.** Every OpenFOAM mesh and solve will fail, which is most of the
    product -- but not all of it, and the old sentence said "meshing and solving
    will fail" as though it were. `lbm.py` and `sph.py` are numpy, need no OpenFOAM,
    and reach every workspace with the toolbox; `advise.py` already lists them as
    `[in this toolbox]` beside the applications it reports as `[not found here]`.
    Measured on a Windows laptop with neither OpenFOAM nor Docker: `lbm.py
    poiseuille --n 32` converged in 4.1 s to 9.7e-07 of u_max against the exact
    parabola. Telling someone their machine can do nothing, when it can solve that,
    is the kind of wrong sentence that makes a person stop.

    **No numpy.** Then even those two are gone, and the toolbox is down to the
    scripts that only read files. Checked here rather than left to the first solve,
    because it is one import and the answer shapes everything above."""
    notes: list[str] = []
    if not shell_found():
        notes.append(
            f"[red]no `{SHELL}` on this machine:[/] every command needs a POSIX shell, so "
            "nothing will run. On Windows, Git for Windows ships one; or use --local for "
            "the Docker workspace, which brings its own."
        )
    if not backend.bashrc and shutil.which("blockMesh") is None:
        notes.append(
            "[yellow]no OpenFOAM here:[/] its meshers and solvers will fail. What does "
            "work is the toolbox: `lbm.py` and `sph.py` solve in numpy with no OpenFOAM "
            "at all, and the analysis and plotting scripts read any case you already "
            "have. `advise.py` says which of them suits a case. For the rest, set "
            "OPENREYNOLDS_FOAM_BASHRC to an installation's etc/bashrc, or use --local "
            "for the Docker workspace."
        )
        if not _numpy_here(backend):
            notes.append(
                "[yellow]and no numpy for the interpreter `python3` resolves to:[/] that "
                "leaves the toolbox scripts that only read files. `pip install numpy "
                "matplotlib` into that interpreter brings the solvers back."
            )
    return notes


def _numpy_here(backend: "LocalBackend") -> bool:
    """Whether the `python3` a toolbox script will run has numpy.

    Asked through `exec` rather than by importing numpy in this process, because the
    two can be different interpreters -- and on Windows usually are, since
    `_python3_line` points `python3` at `sys.executable` only when the name does not
    already work. A failure to even run counts as no numpy; the shell note above has
    already covered the case where nothing runs."""
    try:
        return backend.exec("python3 -c 'import numpy'", timeout_s=60).exit_code == 0
    except BackendError:
        return False


class LocalBackend(KernelHost, Backend):
    """A workspace on this machine."""

    def __init__(self, root: str | Path | None = None, bashrc: str | None = None):
        chosen = root or os.environ.get("OPENREYNOLDS_LOCAL_WORK") or DEFAULT_ROOT
        self.root = Path(chosen).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / LOGS_DIR).mkdir(exist_ok=True)
        self.workspace_root = str(self.root)
        self.bashrc = bashrc if bashrc is not None else find_bashrc()
        self.shell_root = shell_root(self.root)
        """The root as a command may name it. `/work` in any command is rewritten to
        this (`_wrap`), because that is the only root the frozen prompt and the
        toolbox know about."""
        if " " in self.shell_root:
            raise BackendError(
                f"the workspace root {self.root} has a space in it and this machine has no "
                "8.3 short name for it, so a command naming it would be read as two "
                "words. Set OPENREYNOLDS_LOCAL_WORK to a path without one.",
                "unsayable_root")
        self.python3_line = _python3_line()
        """Prepended to every command, and empty on a machine where `python3`
        already runs."""
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self.was_already_running = True
        """Nothing was started to get here, so nothing should be stopped on the way out."""

    # -- shell -----------------------------------------------------------------

    def _wrap(self, cmd: str) -> str:
        """The command as this machine can run it: `/work` made real, `python3`
        defined where it is not, and OpenFOAM sourced where there is one.

        The rewrite is textual and deliberately so. The model is told `/work` by a
        frozen prompt and reads it again in every toolbox docstring, so a command it
        writes will say `/work` however the backend is built; the alternative to
        rewriting here is a different prompt per backend, which is the thing the
        protocol exists to avoid.

        Sourcing is quiet and non-fatal: a machine without OpenFOAM should still be
        able to run `ls`, and a broken environment should show up as the solver saying
        so rather than as every command failing for a reason nobody can see."""
        cmd = _WORK.sub(self.shell_root, cmd)
        if not self.bashrc:
            return f"{self.python3_line}{cmd}"
        return (f"{self.python3_line}"
                f"source {self.bashrc} >/dev/null 2>&1 || true\n{cmd}")

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

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120,
             *, background: bool = False) -> ExecResult:
        # `background` is accepted and ignored: it exists to stop a hosted backend
        # billing for a workspace a poll is only looking at, and this machine is not
        # rented by the second. A local poll runs like any other command.
        limit = min(int(timeout_s or 120), EXEC_MAX_TIMEOUT_S)
        where = self._resolve(cwd)
        where.mkdir(parents=True, exist_ok=True)
        log = self.root / LOGS_DIR / f"exec-{uuid.uuid4().hex[:12]}.log"
        try:
            done = subprocess.run(
                [SHELL, "-lc", self._wrap(cmd)],
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

    def get_file(
        self, path: str, offset: int = 0, limit: int | None = None,
        *, timeout: float = 300.0, max_attempts: int | None = None,
    ) -> bytes:
        del timeout, max_attempts  # no network here to bound; see HostedBackend.get_file
        target = self._resolve(path)
        try:
            with open(target, "rb") as handle:
                handle.seek(max(0, int(offset)))
                return handle.read() if limit is None else handle.read(int(limit))
        except FileNotFoundError as exc:
            raise BackendError(f"no such file: {path}", "not_found") from exc
        except IsADirectoryError as exc:
            raise BackendError(f"{path} is a directory", "is_a_directory") from exc

    def stat(self, path: str, *, timeout: float = 300.0, max_attempts: int | None = None) -> Stat:
        del timeout, max_attempts
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

    def get_tree(
        self,
        remote_paths: list[str],
        local_dir: Path,
        *,
        timeout: float = 300.0,
        max_attempts: int | None = None,
        via: str | None = None,
    ) -> list[Path]:
        # A local copy has no network to time out or retry on, and nothing to route
        # around, so all three are accepted and ignored -- the mirror can pass them
        # unconditionally regardless of which backend it is talking to.
        del timeout, max_attempts, via
        out: list[Path] = []
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for path in remote_paths:
            source = self._resolve(path)
            if not source.exists():
                continue
            # Where the mirror expects it (`mirror.local_for`): the path's shape under the
            # workspace root, as the hosted backend's archive keeps it. By its bare name
            # it landed where `local_for` never looked, and two directories' `faces`
            # were one file.
            try:
                landing = destination / source.relative_to(self.root)
            except ValueError:
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
                [SHELL, "-lc", self._wrap(cmd)],
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
                # As the service words it: a KILL or TERM this side did not send (-9
                # or -15, or 137/143 from the `bash -lc` wrapper around the solver) is
                # killed_externally; any other non-zero exit, a segfault included, is
                # the job's own failure.
                job.end_reason = job.end_reason or (
                    "completed" if code == 0
                    else "killed_externally" if code in (-9, -15, 137, 143)
                    else "failed"
                )
            # `exited` for an exit the job reached, `killed` for a stop or a kill_on.
            status = ("exited" if job.end_reason in ("completed", "failed", "killed_externally")
                      else "killed")
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
        """Jobs outlive the session by design, so this only lets go of the handles.

        The kernel is the exception: it is this session's, holds this session's
        bindings, and nothing else can use it, so it goes down here."""
        self.kernel_stop()
        with self._lock:
            self._jobs.clear()
