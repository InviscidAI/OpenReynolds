"""The Backend protocol — the whole independence story.

Nothing above this interface may know whether it is talking to a container over the
network or to a local OpenFOAM install. Concretely: no module that imports this one may
name a transport, a URL, or a particular hosting service. Only the implementation
modules in this package may. `tests/test_negative_obligation.py` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # the cell channel's machinery, and only its names, live next door
    from .kernel import CellResult

WORKSPACE_ROOT = "/work"
"""The persistent directory every backend presents to the model."""

EXEC_MAX_TIMEOUT_S = 300
"""Ceiling the protocol advertises for a synchronous command. Backends may enforce it;
work that outlives it belongs in a job."""

EXEC_SYNC_WINDOW_S = 120
"""How long a backend holds a synchronous command before it may stop waiting and move
the command to a detached job instead (`ExecResult.promoted`).

The hosted workspace does exactly that, at this many seconds, for a command whose
caller asked for longer: it starts the command again from scratch as a job and answers
with the job's id, because a connection held past about 150 s is cut by the edge in
front of it. The local backend never does -- it runs a command to its `timeout_s` --
so a caller that asks for more than this has to be ready for either answer. Here
rather than in the hosted module because the tool that runs commands states the
number to the model, and a fact a description states has to have one home."""


@dataclass(frozen=True)
class ExecResult:
    """Outcome of a synchronous command."""

    exit_code: int | None
    """What the command exited with. -1 is a backend's sentinel for "no exit status was
    reported" -- a command killed at its timeout. None only with `promoted`: the command
    has no exit code HERE because it belongs to a job now, and the job's status is where
    the code will appear. It used to be 0 on that shape, which is the one number every
    reader takes for "it finished, and it worked"."""
    output: str
    """Combined stdout+stderr, already capped by the backend."""
    truncated: bool
    """True when `output` is only part of what the command produced."""
    log_path: str | None
    """Where the complete output lives in the workspace, when the backend keeps one."""
    stderr: str = ""
    """The *wrapper's* own stderr -- why the command could not be run at all (a working
    directory that is gone, a capture-sync that failed), not what the command printed
    (that is in `output`). Empty in the normal case; when it is not, it is usually the
    platform rather than the command, which is the difference a caller needs in order
    not to retry a failure that will fail again."""
    job_id: str = ""
    """Set when a synchronous command outran the exec window and the backend moved it to
    a detached job: this is that job's id. The command did not fail and must not be
    re-run -- it is running now, to be followed with job_check."""
    idle: bool = False
    """A `background` command found no workspace running, so nothing was run.

    Not a failure and not an empty workspace -- the difference matters, because an
    empty listing is the most convincing wrong answer a file mirror can be given.
    `output` is empty and `exit_code` says nothing; the only correct reading is
    "ask again later"."""
    promoted: bool = False
    """The command was still running at the backend's synchronous window
    (`EXEC_SYNC_WINDOW_S`) and the backend started it again from scratch as a detached
    job: `job_id` names it, `exit_code` is None, and `output` is whatever the
    synchronous run had produced by then (the hosted service sends none of it today).

    Neither a failure nor a result. The hosted backend used to hand this shape back as
    `exit_code=0` with the service's note for output, and the `bash` tool printed
    `exit_code: 0` as its first line: measured in production on 2026-09-21 (study
    20260921-033019-e1b4, workspace 35c9f018), `sleep 240`, `sleep 200` and `sleep 180`
    each came back that way while still running (jobs 808edf10, 5432459d, e839ac57).
    The model read three commands that had finished with no output, and never polled
    the jobs. A flag of its own, so that no reader has to infer the state from a job id
    beside a zero."""


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
class StoredEntry:
    """One path as a backend's copy of the workspace holds it.

    The same four facts `find` prints for a live listing (`browse.FIND_FORMAT`), so a
    listing read from the copy is the same shape as one walked on the machine, and
    nothing downstream can tell which it was given. Plain data rather than
    `browse.Entry` because `browse` imports this module, and the protocol may not
    import it back."""

    path: str
    is_dir: bool
    size: int = 0
    mtime: float = 0.0


@dataclass(frozen=True)
class StoredListing:
    """What a backend's copy of the workspace says is under a path.

    Not depth-limited by the backend: the copy is asked for everything under the path
    and cut to the caller's depth on this side, because the copy answers as a whole
    tree. `truncated` is the copy's own word that it held more than it listed, so the
    entries are the head of the answer and not the whole of it."""

    entries: list[StoredEntry]
    truncated: bool = False


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

    study_id: str | None = None
    """Which study this workspace is serving, once the session has said (`cli.session`
    sets it when it opens or resumes the study's row). Only a backend that keeps a copy
    of the workspace somewhere other than the machine has a use for it -- the copy is
    kept per study, and `list_stored` reads it under this name. Every other backend
    carries None and nothing asks."""

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120,
             *, background: bool = False) -> ExecResult:
        """Run one command and wait for it.

        `background=True` says this command is a poll rather than work: run it only on
        a workspace that is already up, and do not let it keep that workspace alive. A
        hosted backend bills by the second for as long as something claims to be using
        it, so a listing taken on a timer must not be that claim. Backends with no
        lifecycle to protect may ignore it. When nothing is running, the result comes
        back with `idle` set and no output."""
        ...

    def put_file(self, path: str, data: bytes) -> None: ...

    def get_file(
        self, path: str, offset: int = 0, limit: int | None = None,
        *, timeout: float = 300.0, max_attempts: int | None = None,
    ) -> bytes:
        """Bytes from `path`, starting at `offset`.

        `limit` is not optional in the way it looks. A backend is free to answer with
        less than everything, and the hosted one does: asked for a file with no limit
        it returns its own page size and says nothing about the rest. A caller that
        wants a whole file has to `stat` it and ask for that many bytes. Leaving this
        unwritten cost a live bug -- renders between the page size and the attachment
        ceiling reached the model truncated, and a truncated PNG is not a smaller
        picture, it is a broken one.

        `timeout`/`max_attempts` bound one request, the same contract as on `get_tree`
        -- see there. A backend with no network to bound may accept and ignore them.
        """
        ...

    def stat(self, path: str, *, timeout: float = 300.0, max_attempts: int | None = None) -> Stat: ...

    def put_tree(self, local_dir: Path, remote_dir: str) -> None: ...

    def get_tree(
        self,
        remote_paths: list[str],
        local_dir: Path,
        *,
        timeout: float = 300.0,
        max_attempts: int | None = None,
        via: str | None = None,
    ) -> list[Path]:
        """Pack and download the given paths.

        `timeout` and `max_attempts` bound how long a single request may run and how
        many times it may be retried; a backend with nothing to time out (the local
        one) may accept and ignore them. Callers with work in flight alongside this
        one -- the mirror's background cycles -- pass both; a caller with nothing
        else running, like `openreynolds pull`, passes neither.

        `via` names an alternate path a backend may offer for building the archive --
        the hosted one accepts `"volume"`, which reads the persistent volume directly
        instead of a container that a tool call might also be using, at the cost of
        not preserving file mode. A backend with nothing to route around (the local
        one) may accept and ignore it."""
        ...

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

    def active_jobs(self) -> list[dict[str, Any]]:
        """Every job still running on this workspace, whoever started it.

        Not the same question as "which of MY jobs are running", which the local
        study record already answers. A job started outside any session -- a
        detached render launched from a shell, a solve left by a session that has
        since exited -- appears in no session's record, and shutting the workspace
        down under it kills it. That is F-46: a rendering job lost two animation
        passes to a study session that started after it, ran, and stopped the
        instance on the way out.

        A backend with no shared lifecycle to protect answers with an empty list,
        and so does one whose service cannot be asked: a shutdown must not be
        blocked by a listing that failed."""
        return []

    # -- the cell channel ------------------------------------------------------
    #
    # A long-lived kernel in the workspace, beside `exec` rather than instead of it.
    # `exec` runs a command and forgets it; a desk building geometry needs the
    # bindings to survive the step, and needs what a cell drew handed back as bytes
    # rather than as a filename someone had to guess. The kernel is another thing
    # running in the workspace: nothing here names where it is or how it is reached,
    # and `CellResult` carries no handle to it.

    def kernel_start(self, cwd: str) -> str:
        """Bring a kernel up in `cwd`; return the session id.

        One kernel per run, started in the case directory and put down with the run
        (`close`). Asking twice returns the same id rather than a second kernel."""
        ...

    def kernel_run(self, code: str, timeout_s: int) -> CellResult:
        """Run one cell and wait up to `timeout_s` seconds for it.

        **The window expires; the cell is not killed.** `timeout_s` bounds the wait,
        never the work: on expiry the result comes back with `still_running` set, the
        elapsed time, and whatever streamed so far, and the caller polls or interrupts
        deliberately. Under bash a timeout lost nothing because state lived on disk; in
        a kernel it would lose every binding.

        `timeout_s` is the caller's, from configuration. Nothing in `code` can widen
        it, and there is no request field that would let it."""
        ...

    def kernel_poll(self) -> CellResult:
        """Whether the current cell is still going, and what it printed since the
        last look."""
        ...

    def kernel_interrupt(self) -> None:
        """Interrupt the running cell. An agent action, not a reflex -- the kernel and
        every binding in it survive."""
        ...

    def kernel_restart(self) -> None:
        """A fresh kernel. Every binding is gone, which is what makes the cell log the
        source of truth rather than the kernel."""
        ...

    def list_stored(self, path: str, depth: int) -> StoredListing | None:
        """What is under `path`, to `depth`, listed without running a command on the
        workspace -- from the machine while it is up, from a copy of the workspace
        that outlives the machine once it is down -- or None when this backend has
        no such listing to give, or cannot get it just now.

        The listing a session takes on its way out is what this began as. It went
        through `exec`, and a foreground `exec` on a hosted workspace the service had
        already stopped starts a new machine to run it: on 2026-09-21 the close-down
        of an idle-timed-out session started a c7i.2xlarge at 02:24:32 for one `find`,
        and the machine then sat until the reaper took it down again at 02:42 --
        eighteen minutes of instance for a listing the service could have answered
        from the copy it writes at every stop. The same morning the `find` on a
        *running* workspace was found cut at the exec channel's 64 KB output cap,
        mid-line, and the partial last row read as a 1.5 MB file at the study root
        (`browse.LIST_PIPELINE`). So `Browser.tree` asks this first, running or
        stopped, background or not; the walk over `exec` is what runs when the answer
        is None, and it starts a machine only for a foreground listing that finds
        nothing running, as it always did.

        Same shape as the walk (`StoredEntry` is what `find` prints), relative to
        the same root, so a caller cannot tell which it was given. `depth` counts as
        `find -maxdepth` does: the path's own children are 1.

        Never raises, never starts a machine, never counts as use of one, and None is
        never "empty" -- it means "ask the machine", and the caller does. The default
        is a backend with nothing but the machine to ask."""
        return None

    def close(self) -> None: ...
