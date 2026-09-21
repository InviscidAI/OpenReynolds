"""A workspace that is still coming up, standing in for the one that will be there.

A hosted workspace takes seconds to a minute or two to start, and until this existed
the whole session waited for it: no header, no model, nothing on screen but a
spinner. The person had nothing to do for that minute and the model had nothing to
do either, although the two of them could have been settling what the study is
about -- none of which needs a machine.

`PendingBackend` is what makes the wait somewhere else. It implements `Backend`, so
every tool, the mirror, the tracker and the close-down hold it as if it were the
workspace itself; each call blocks until the real backend arrives (`resolve`) and
is then handed straight to it. A call made while the workspace is still starting
therefore takes as long as the start plus itself, which is exactly what it would
have taken with the old blocking start moved to the front -- only now the
conversation has been running the whole time.

What is NOT delegated: a `background=True` exec answers `idle` at once, because
the protocol's meaning for that flag is "only if the workspace is already up" and a
poll must never be what everyone waits behind.

Transport-free by construction: this module knows nothing about how the workspace
is started, only that somebody will call `resolve` or `fail`. `tests/
test_negative_obligation.py` holds it to that.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from .base import (
    WORKSPACE_ROOT,
    Backend,
    BackendError,
    ExecResult,
    JobStatus,
    Stat,
    StoredListing,
)

WAIT_CEILING_S = 600.0
"""The longest any one call waits for the workspace before giving up on it.

In practice the wait ends when the start does: whoever is starting the workspace
resolves or fails this backend, and the start itself is bounded by its own request
timeout and retries, the same bound `acquire` has always had. This is the backstop
for a starter that never answers at all, so a tool call cannot hang for ever on a
thread that died."""

WAIT_TICK_S = 5.0
"""How often a blocked call says it is still waiting (`on_wait`), so whatever is
showing the session can say why a tool is taking long."""


class PendingBackend(Backend):
    """A `Backend` whose every call waits for the real one.

    `instance_id`, `instances_held` and `was_already_running` are known from the
    moment the instance row is chosen and are answered without waiting; the last
    two are re-read from the live backend once it is here, because the start call
    is what settles whether this session brought the workspace up."""

    workspace_root = WORKSPACE_ROOT

    def __init__(
        self,
        instance_id: str,
        *,
        instances_held: int = 0,
        was_already_running: bool = False,
        on_wait: Callable[[float], None] | None = None,
        closer: Callable[[], None] | None = None,
    ):
        self.instance_id = instance_id
        self._instances_held = instances_held
        self._listed_as_running = was_already_running
        self.on_wait = on_wait
        """Called with the seconds waited so far, every `WAIT_TICK_S`, by a call that
        is blocked on the workspace. Presentation only; a failure in it is swallowed."""
        self._closer = closer
        self.ready_event = threading.Event()
        """Set once the workspace is usable -- or once it is known that it never will
        be. `ready()` tells the two apart."""
        self._live: Backend | None = None
        self._error: BackendError | None = None
        self._since = time.monotonic()
        self._settled_at: float | None = None
        self._study_id: str | None = None

    # -- becoming the real thing -----------------------------------------------

    def resolve(self, backend: Backend) -> None:
        """The workspace is up and set up: from here every call goes to `backend`.

        What the session told this stand-in while it waited is told on: the study
        id arrives before the workspace does (the study's row is opened while the
        machine is still starting), and the live backend is the one that reads the
        study's copy of the workspace under it."""
        if self._study_id is not None:
            backend.study_id = self._study_id
        self._live = backend
        self._settled_at = time.monotonic()
        self.ready_event.set()

    @property
    def study_id(self) -> str | None:
        """Which study this workspace is serving. Held here until the workspace is
        up, then the live backend's -- set on it at `resolve`, or at once if the
        workspace is already here."""
        return self._study_id

    @study_id.setter
    def study_id(self, value: str | None) -> None:
        self._study_id = value
        if self._live is not None:
            self._live.study_id = value

    def fail(self, error: BackendError) -> None:
        """The workspace is not coming. Every waiting call, and every later one,
        raises `error` -- the same error the old blocking start would have raised."""
        self._error = error
        self._settled_at = time.monotonic()
        self.ready_event.set()

    def ready(self) -> bool:
        return self._live is not None

    @property
    def failed(self) -> bool:
        return self._error is not None

    @property
    def elapsed_s(self) -> float:
        """How long the workspace took to come up, or has been coming up so far."""
        end = self._settled_at if self._settled_at is not None else time.monotonic()
        return end - self._since

    @property
    def was_already_running(self) -> bool:
        if self._live is not None:
            return bool(getattr(self._live, "was_already_running", self._listed_as_running))
        return self._listed_as_running

    @property
    def instances_held(self) -> int:
        if self._live is not None:
            return int(getattr(self._live, "instances_held", self._instances_held))
        return self._instances_held

    def wait(self, timeout: float | None = None) -> Backend:
        """The live backend, waiting for it if it is not here yet.

        Raises the start's own `BackendError` if the workspace failed to come up, and
        a `workspace_not_ready` one if `timeout` (default `WAIT_CEILING_S`) ran out
        first."""
        if self._live is not None:
            return self._live
        if self._error is not None:
            raise self._error
        ceiling = WAIT_CEILING_S if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + ceiling
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackendError(
                    f"the workspace did not come up within {ceiling:.0f} s",
                    code="workspace_not_ready",
                )
            if self.ready_event.wait(min(WAIT_TICK_S, remaining)):
                break
            if self.on_wait is not None:
                try:
                    self.on_wait(self.elapsed_s)
                except Exception:  # noqa: BLE001 - presentation may not fail a tool call
                    pass
        if self._error is not None:
            raise self._error
        assert self._live is not None
        return self._live

    # -- Backend ---------------------------------------------------------------
    #
    # Each call is handed to the live backend exactly as it was made -- the same
    # positional and keyword arguments, no defaults filled in here -- so a backend
    # with an older or narrower signature (a test's fake, a local one that takes no
    # `timeout`) sees the call it would have seen without this in between.

    def exec(self, cmd: str, *args: Any, **kwargs: Any) -> ExecResult:
        if kwargs.get("background") and self._live is None:
            # A poll, by the protocol's own definition: run only on a workspace that
            # is already up. Nothing ran, and the result says so rather than looking
            # like an empty workspace.
            return ExecResult(exit_code=-1, output="", truncated=False, log_path=None,
                              stderr="", idle=True)
        return self.wait().exec(cmd, *args, **kwargs)

    def put_file(self, *args: Any, **kwargs: Any) -> None:
        self.wait().put_file(*args, **kwargs)

    def get_file(self, *args: Any, **kwargs: Any) -> bytes:
        return self.wait().get_file(*args, **kwargs)

    def stat(self, *args: Any, **kwargs: Any) -> Stat:
        return self.wait().stat(*args, **kwargs)

    def put_tree(self, *args: Any, **kwargs: Any) -> None:
        self.wait().put_tree(*args, **kwargs)

    def get_tree(self, *args: Any, **kwargs: Any) -> list[Path]:
        return self.wait().get_tree(*args, **kwargs)

    def job_start(self, *args: Any, **kwargs: Any) -> str:
        return self.wait().job_start(*args, **kwargs)

    def job_status(self, *args: Any, **kwargs: Any) -> JobStatus:
        return self.wait().job_status(*args, **kwargs)

    def job_tail(self, *args: Any, **kwargs: Any) -> tuple[str, int, bool]:
        return self.wait().job_tail(*args, **kwargs)

    def job_kill(self, *args: Any, **kwargs: Any) -> JobStatus:
        return self.wait().job_kill(*args, **kwargs)

    def active_jobs(self) -> list[dict[str, Any]]:
        live = self.wait()
        active = getattr(live, "active_jobs", None)
        return active() if active is not None else []

    def list_stored(self, path: str, depth: int) -> StoredListing | None:
        """The live backend's listing of the workspace, or None while there is no
        live backend yet.

        Not waited for, unlike everything else here: a listing that needs no
        machine must not be the call that waits for one. None sends the caller to
        its fallback, the walk over `exec` -- a poll, which this stand-in answers
        `idle` at once, then the foreground `exec`, which waits for the workspace
        exactly as it always did -- so a listing asked for during the start costs
        what it cost before, and nothing is read from a copy that the machine now
        coming up is about to overtake. Once the live backend is here, its own
        answer is handed back, and `Browser.tree` takes it before any command."""
        live = self._live
        if live is None:
            return None
        stored = getattr(live, "list_stored", None)
        return stored(path, depth) if stored is not None else None

    def shutdown(self) -> None:
        """Put the workspace down -- once it is up.

        A start that is already under way cannot be called back: the service has the
        request and the container is being built. So an End pressed during the wait
        waits it out and then stops what it started, rather than leaving a workspace
        up for nobody. A workspace that failed to come, or was never asked to, has
        nothing to stop."""
        if self._error is not None:
            return
        live = self.wait()
        shutdown = getattr(live, "shutdown", None)
        if shutdown is not None:
            shutdown()

    def close(self) -> None:
        if self._live is not None:
            self._live.close()
        elif self._closer is not None:
            self._closer()
