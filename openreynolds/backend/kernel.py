"""The cell channel: a long-lived IPython kernel on the workspace, one cell a step.

`exec` runs a command and forgets everything about it; the desk that builds geometry
needs the opposite. A B-rep costs a minute to construct and is then asked five
questions, and under `exec` each of those questions rebuilds it. So the workspace also
carries a kernel: a process that stays up for the length of a run, holds the bindings
between steps, and hands back what a cell drew as bytes rather than as a filename the
caller had to guess.

**It lives on the far side of the protocol, and that is the point.** On one machine an
in-process kernel and a remote one are indistinguishable, which is exactly how the
deleted stack acquired an in-process kernel and the X/GL layer that came with it. Here
the kernel is another thing running in the workspace, reached the way everything else
in the workspace is reached, and `CellResult` carries no handle to it --- no socket, no
port, no connection file, nothing a caller could learn the transport from.

**One implementation, both backends.** Everything below is written against the `Backend`
protocol itself -- `put_file`, `get_file`, `stat`, `job_start` -- so `LocalBackend` and
`HostedBackend` run the same channel code against the same kernel driver. The two
agreeing is then a property of there being one of them, rather than a thing to keep
testing for.

**The window expires; the cell is not killed.** `timeout_s` bounds how long the caller
waits, never how long the cell may run. With bash a timeout lost nothing, because state
lived on disk; in a kernel it would lose every binding, so on expiry the result says
`still_running` and carries the elapsed time and whatever streamed, and the next step
polls or interrupts deliberately. The run-level budget (`MAX_SECONDS` in the loop) is
what bites on a genuine runaway -- one stuck cell cannot be allowed to eat the run, but
a slow one must not be shot for being slow.

The wire between here and the kernel is files in the workspace, which is the only
vocabulary every backend already has: a request file per cell, the streams appended to
their own files as they arrive, display data written out as PNGs, and a small JSON when
the cell is done. Nothing above this module sees any of it.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from .. import images
from .base import WORKSPACE_ROOT, BackendError

CELL_TIMEOUT_S = 240
"""How long one cell is waited for, in seconds.

Harness configuration, beside the loop's `STEP_TIMEOUT_S`, and deliberately **not
agent-configurable**: the point of a step budget is to keep the conversation moving
rather than to cap compute, and a knob for raising it only lets one stuck cell spend
the whole run's budget. Expiring costs nothing -- the cell keeps running -- so there is
nothing for a cell to gain by reaching for it, and `kernel_run` gives it no way to.
"""

MAX_OUTPUT_BYTES = 200_000
"""How much of one stream comes back inline; the same bargain `exec` strikes."""

MAX_IMAGES = 40
"""Per cell. A loop that draws in every iteration is a mistake to report, not to ship."""

KERNEL_START_TIMEOUT_S = 120.0
"""How long a kernel is given to come up before the workspace is declared to have none."""

CONTROL_TIMEOUT_S = 60.0
"""How long an interrupt or a restart is waited on before it is called a failure."""

POLL_INTERVAL_S = 0.05
"""How often a waiting call looks for its answer."""

_HEALTH_INTERVAL_S = 5.0
"""How often a wait re-asks whether the kernel is still there at all. Rarely, because
the answer is a round trip on a hosted workspace and almost always yes."""


@dataclass(frozen=True)
class CellResult:
    """What one cell did.

    A raising cell is one of these too, with `ok=False` and the exception in `error`
    and `traceback`: the desk is handed the failure as work, exactly as a non-zero
    exit code is today. No exception escapes into the middle of a tool call.

    Every field here is about the cell. There is deliberately no field naming a
    session, a socket, a port or a file descriptor -- `log_path` is a workspace path,
    the same kind of thing `ExecResult.log_path` is.
    """

    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    """Exception type and message, `""` when the cell ran clean."""
    traceback: str = ""
    images: list[bytes] = field(default_factory=list)
    """PNG display data, in emission order -- what the cell drew, not what it named."""
    seconds: float = 0.0
    still_running: bool = False
    """The wait expired and the cell was **not** killed. It is still going."""
    truncated: bool = False
    log_path: str = ""
    """Where the whole of a clipped output lives, in the workspace."""


class KernelChannel:
    """One kernel, and the cells sent to it.

    Owned by a backend and reached through the four protocol methods; nothing above
    the protocol holds one of these.
    """

    def __init__(self, backend):
        self._backend = backend
        self._session = ""
        self._dir = ""
        self._job = ""
        self._seq = 0
        self._started: float = 0.0
        self._stdout_seen = 0
        self._stderr_seen = 0
        self._images_seen = 0
        self._control = 0

    # -- lifecycle -------------------------------------------------------------

    @property
    def session(self) -> str:
        return self._session

    def start(self, cwd: str) -> str:
        """Bring a kernel up in `cwd` and return the session id.

        A second call on a live channel is a no-op that returns the same id: a desk
        that asks twice wants a kernel, not two of them.
        """
        if self._session and self._alive():
            return self._session
        session = uuid.uuid4().hex[:12]
        root = getattr(self._backend, "workspace_root", WORKSPACE_ROOT)
        directory = f"{root}/.kernel/{session}"
        self._backend.put_file(f"{directory}/driver.py", _DRIVER.encode("utf-8"))
        where = _relative(cwd, root)
        self._job = self._backend.job_start(
            f"python3 .kernel/{session}/driver.py {where}",
            cwd=root,
            name=f"kernel-{session}",
        )
        self._session = session
        self._dir = directory
        self._seq = 0
        self._stdout_seen = self._stderr_seen = self._images_seen = 0
        self._await_ready()
        return session

    def _await_ready(self) -> None:
        deadline = time.monotonic() + KERNEL_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._size(f"{self._dir}/ready") >= 0:
                return
            failed = self._read_json("failed.json")
            if failed:
                self.stop()
                raise BackendError(str(failed.get("message", "the kernel did not start")),
                                   "kernel_unavailable")
            if not self._alive():
                break
            time.sleep(POLL_INTERVAL_S)
        note = self._driver_log()
        self.stop()
        raise BackendError(
            "the kernel did not come up on this workspace" + (f": {note}" if note else ""),
            "kernel_unavailable",
        )

    def stop(self) -> None:
        """Put the kernel down. The workspace, and everything the cells wrote, stays."""
        job, self._job = self._job, ""
        self._session = ""
        self._dir = ""
        if not job:
            return
        try:
            self._backend.job_kill(job, "TERM")
        except Exception:  # noqa: BLE001 - a kernel that is already gone is gone
            pass

    def _alive(self) -> bool:
        if not self._job:
            return False
        try:
            return self._backend.job_status(self._job).running
        except Exception:  # noqa: BLE001 - unknown is not proof of death; the wait decides
            return True

    def _driver_log(self) -> str:
        if not self._job:
            return ""
        try:
            text, _offset, _eof = self._backend.job_tail(self._job)
        except Exception:  # noqa: BLE001
            return ""
        return _clip(text.strip(), 800)

    def _require(self) -> None:
        if not self._session:
            raise BackendError("no kernel has been started on this workspace",
                               "no_kernel")

    # -- cells -----------------------------------------------------------------

    def run(self, code: str, timeout_s: int = CELL_TIMEOUT_S) -> CellResult:
        """Send one cell and wait up to `timeout_s` for it.

        `timeout_s` comes from the caller and nowhere else. `code` is written to a
        file and read back by the kernel driver as the string to execute; it is never
        parsed here, and there is no field in the request a cell could set to widen
        its own window.
        """
        self._require()
        window = max(1, int(timeout_s))
        self._seq += 1
        cell = self._seq
        self._stdout_seen = self._stderr_seen = self._images_seen = 0
        self._started = time.monotonic()
        self._backend.put_file(
            f"{self._dir}/req-{cell}.json",
            json.dumps({"code": code}).encode("utf-8"),
        )
        return self._wait(cell, self._started + window)

    def _wait(self, cell: int, deadline: float) -> CellResult:
        checked = time.monotonic()
        while True:
            done = self._read_json(f"done-{cell}.json")
            if done is not None:
                return self._gather(cell, done, whole=True)
            now = time.monotonic()
            if now >= deadline:
                # The cell is not killed, and nothing here asks for it to be. This is
                # the whole difference between a slow computation and a lost one.
                return self._gather(cell, None, whole=True)
            if now - checked > _HEALTH_INTERVAL_S:
                checked = now
                if not self._alive():
                    raise BackendError(
                        "the kernel stopped while this cell was running; its state is "
                        "gone and the cell log is what replays it",
                        "kernel_gone",
                    )
            time.sleep(POLL_INTERVAL_S)

    def poll(self) -> CellResult:
        """Where the current cell is, and what it has printed since the last look.

        **A poll that catches the cell finishing returns the whole of its output**, not
        the delta, which is what `run()` returns on the same event and for the same
        reason: that result is the cell's result, and it is the only one anybody gets.
        `done-N.json` and `out-N.txt` are separate files written by the driver, so a poll
        can see the first before the last flush of the second is visible -- and with a
        delta the tail is then lost for good, because the caller takes the completed
        result and stops asking. It showed up as a 1-in-3 flake in
        `test_a_cell_that_outruns_the_window_is_not_killed`, where the poll that saw the
        cell finish came back without the `print('done')` that finished it.

        The cost is that output already shown in an earlier still-running poll is shown
        again in the final one. That is the right way round: repetition is visible and
        loss is not, and the desk reading this has just been told the cell is done.
        """
        self._require()
        if self._seq == 0:
            return CellResult(ok=True)
        done = self._read_json(f"done-{self._seq}.json")
        return self._gather(self._seq, done, whole=done is not None)

    def interrupt(self) -> None:
        """Interrupt whatever is running. An agent action, not a reflex.

        Waits for the cell to actually stop, so the next cell does not queue behind a
        loop that ignored the signal.
        """
        self._require()
        self._signal("interrupt")
        if self._seq:
            deadline = time.monotonic() + CONTROL_TIMEOUT_S
            while time.monotonic() < deadline:
                if self._read_json(f"done-{self._seq}.json") is not None:
                    return
                time.sleep(POLL_INTERVAL_S)

    def restart(self) -> None:
        """A fresh kernel in the same session. Every binding is gone; the cell log
        replays them."""
        self._require()
        self._signal("restart")
        self._stdout_seen = self._stderr_seen = self._images_seen = 0

    def _signal(self, kind: str) -> None:
        self._control += 1
        token = self._control
        self._backend.put_file(
            f"{self._dir}/{kind}.json", json.dumps({"token": token}).encode("utf-8")
        )
        deadline = time.monotonic() + CONTROL_TIMEOUT_S
        while time.monotonic() < deadline:
            ack = self._read_json(f"ack-{kind}.json")
            if ack and int(ack.get("token", 0)) >= token:
                return
            if not self._alive():
                raise BackendError(f"the kernel is gone; it cannot {kind}", "kernel_gone")
            time.sleep(POLL_INTERVAL_S)
        raise BackendError(f"the kernel did not answer a {kind}", "kernel_stuck")

    # -- reading back ----------------------------------------------------------

    def _gather(self, cell: int, done: dict | None, *, whole: bool) -> CellResult:
        """One result from what is on disk. `done is None` means it is still going."""
        out_path = f"{self._dir}/out-{cell}.txt"
        err_path = f"{self._dir}/err-{cell}.txt"
        stdout, cut_out, seen_out = self._text(out_path, 0 if whole else self._stdout_seen)
        stderr, cut_err, seen_err = self._text(err_path, 0 if whole else self._stderr_seen)
        self._stdout_seen, self._stderr_seen = seen_out, seen_err
        pictures, notes = self._pictures(cell, 0 if whole else self._images_seen)
        if notes:
            stderr = (stderr + "\n" + "\n".join(notes)).strip()
        seconds = float(done.get("seconds", 0.0)) if done else time.monotonic() - self._started
        return CellResult(
            ok=bool(done.get("ok", False)) if done else True,
            stdout=stdout,
            stderr=stderr,
            error=str(done.get("error", "")) if done else "",
            traceback=str(done.get("traceback", "")) if done else "",
            images=pictures,
            seconds=round(seconds, 3),
            still_running=done is None,
            truncated=cut_out or cut_err,
            log_path=f"{self._dir}/cell-{cell}.log",
        )

    def _text(self, path: str, offset: int) -> tuple[str, bool, int]:
        """Text from `offset`, clipped in the middle when there is too much of it.

        The news is at both ends -- the command that failed and the summary after it --
        so a long stream is cut in its middle, and the whole of it stays at `log_path`.
        """
        size = self._size(path)
        if size <= 0 or offset >= size:
            return "", False, max(offset, 0 if size < 0 else size)
        span = size - offset
        if span <= MAX_OUTPUT_BYTES:
            return self._read(path, offset, span).decode("utf-8", "replace"), False, size
        head = MAX_OUTPUT_BYTES // 3
        tail = MAX_OUTPUT_BYTES - head
        first = self._read(path, offset, head).decode("utf-8", "replace")
        last = self._read(path, size - tail, tail).decode("utf-8", "replace")
        cut = span - head - tail
        return f"{first}\n... {cut} bytes cut ...\n{last}", True, size

    def _pictures(self, cell: int, start: int) -> tuple[list[bytes], list[str]]:
        found: list[bytes] = []
        notes: list[str] = []
        index = start
        while index < MAX_IMAGES:
            path = f"{self._dir}/img-{cell}-{index}.png"
            size = self._size(path)
            if size < 0:
                break
            index += 1
            data = self._read(path, 0, size)
            if not data:
                continue
            # The existing transport rules for a picture, not new ones: the same
            # downscale a fetched render gets, and the same ceiling.
            data = images.downscale(data, "image/png")
            if len(data) > images.MAX_ATTACH_BYTES:
                notes.append(
                    f"[a {len(data)} byte image was left at {path}: over the "
                    f"{images.MAX_ATTACH_BYTES} byte ceiling]"
                )
                continue
            found.append(data)
        self._images_seen = index
        return found, notes

    def _read(self, path: str, offset: int, limit: int) -> bytes:
        try:
            return self._backend.get_file(path, offset, limit)
        except BackendError:
            return b""

    def _size(self, path: str) -> int:
        """Bytes at `path`, or -1 when there is nothing there yet."""
        try:
            return int(self._backend.stat(path).size)
        except BackendError:
            return -1

    def _read_json(self, name: str) -> dict | None:
        path = name if name.startswith("/") else f"{self._dir}/{name}"
        size = self._size(path)
        if size <= 0:
            return None
        try:
            return json.loads(self._read(path, 0, size).decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            return None


def _relative(cwd: str | None, root: str) -> str:
    """`cwd` as a path the workspace's own shell can use.

    The command string is the one thing a backend does not resolve for us -- `cwd` it
    does, paths inside the command it does not -- so what travels in the command is
    relative to the workspace root, which is the same place on every backend.
    """
    text = (cwd or "").rstrip("/")
    for prefix in (root.rstrip("/"), WORKSPACE_ROOT):
        if text == prefix:
            return "."
        if prefix and text.startswith(prefix + "/"):
            return text[len(prefix) + 1:]
    return text or "."


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n... {len(text) - limit} characters cut ...\n{text[-tail:]}"


_DRIVER = r'''"""The kernel driver: written into the workspace, run there, never imported.

It holds one IPython kernel and mediates between it and a directory of files, which
is the only vocabulary shared by every backend. A request file is a cell; the streams
are appended to files as they arrive so a caller that times out still sees what was
printed; display data is written out as PNGs; a small JSON says the cell is done.

It is a string in `kernel.py` rather than a module because it runs on the far side of
the backend, where this package does not exist.
"""

import base64
import json
import os
import queue
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLL = 0.02
MAX_IMAGES = 40


def put(name, payload):
    """Atomically, because a reader is polling for exactly this file."""
    tmp = HERE / (name + ".part")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(str(tmp), str(HERE / name))


def take(name):
    path = HERE / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


class Driver:
    def __init__(self):
        self.km = None
        self.kc = None
        self.cell = None
        self.done = 0
        self.seen = {"interrupt": 0, "restart": 0}

    # -- the kernel --------------------------------------------------------

    def open(self):
        from jupyter_client.manager import KernelManager

        self.km = KernelManager(kernel_name="python3")
        # History off, because the history is a liability here and never an asset. The
        # desk never recalls a previous cell through IPython -- the cell log is what it
        # reads -- but the sqlite history db is shared, and when it is missing or locked
        # the thread that writes it reports the failure on *stdout*:
        #
        #   The history saving thread hit an unexpected error (OperationalError('no
        #   such table: history')).History will not be written to the database.
        #
        # which lands inside the cell output the desk is charged to read, and inside the
        # string a test compares against. It reached one run's `cells.log` in each of the
        # last two sweeps, and it is what makes `test_kernel_channel` flaky.
        self.km.start_kernel(extra_arguments=["--HistoryManager.enabled=False"])
        self.connect()

    def connect(self):
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=90)
        # The picture is the reason this channel exists, so the inline backend is set
        # up before the first cell rather than left for the cell to remember. Failing
        # is fine and quiet: a workspace without matplotlib still runs cells.
        self.silent("%matplotlib inline")

    def silent(self, code):
        msg_id = self.kc.execute(code, silent=True, store_history=False)
        limit = time.time() + 60
        while time.time() < limit:
            try:
                msg = self.kc.get_iopub_msg(timeout=0.2)
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                return

    # -- one cell ----------------------------------------------------------

    def begin(self, seq, code):
        self.cell = {
            "seq": seq,
            "msg_id": self.kc.execute(code),
            "t0": time.time(),
            "images": 0,
            "error": "",
            "traceback": "",
            "ok": True,
        }

    def append(self, which, text):
        blob = text.encode("utf-8", "replace")
        for name in ("%s-%d.txt" % (which, self.cell["seq"]),
                     "cell-%d.log" % self.cell["seq"]):
            with open(str(HERE / name), "ab") as handle:
                handle.write(blob)

    def picture(self, payload):
        if self.cell["images"] >= MAX_IMAGES:
            return
        try:
            raw = base64.b64decode(payload)
        except Exception:
            return
        name = "img-%d-%d.png" % (self.cell["seq"], self.cell["images"])
        tmp = HERE / (name + ".part")
        tmp.write_bytes(raw)
        os.replace(str(tmp), str(HERE / name))
        self.cell["images"] += 1

    def finish(self, **extra):
        cell = self.cell
        self.cell = None
        self.done = cell["seq"]
        payload = {
            "ok": bool(cell["ok"]) and not cell["error"],
            "error": cell["error"],
            "traceback": cell["traceback"],
            "seconds": round(time.time() - cell["t0"], 3),
            "images": cell["images"],
        }
        payload.update(extra)
        if payload["error"]:
            payload["ok"] = False
        put("done-%d.json" % cell["seq"], payload)

    def pump(self):
        try:
            msg = self.kc.get_iopub_msg(timeout=POLL)
        except queue.Empty:
            return
        except Exception:
            return
        if self.cell is None:
            return
        if msg.get("parent_header", {}).get("msg_id") != self.cell["msg_id"]:
            return
        kind = msg["msg_type"]
        body = msg["content"]
        if kind == "stream":
            self.append("err" if body.get("name") == "stderr" else "out",
                        body.get("text", ""))
        elif kind in ("display_data", "execute_result", "update_display_data"):
            data = body.get("data", {}) or {}
            if "image/png" in data:
                self.picture(data["image/png"])
            elif kind == "execute_result" and "text/plain" in data:
                # What the cell evaluated to, where a notebook would show it.
                self.append("out", data["text/plain"] + "\n")
        elif kind == "error":
            self.cell["error"] = "%s: %s" % (body.get("ename", "Error"),
                                             body.get("evalue", ""))
            self.cell["traceback"] = "\n".join(body.get("traceback", []) or [])
            self.cell["ok"] = False
        elif kind == "status" and body.get("execution_state") == "idle":
            self.finish()

    # -- control -----------------------------------------------------------

    def controls(self):
        asked = take("interrupt.json")
        if asked and int(asked.get("token", 0)) > self.seen["interrupt"]:
            self.seen["interrupt"] = int(asked["token"])
            if self.cell is not None:
                self.km.interrupt_kernel()
            put("ack-interrupt.json", {"token": self.seen["interrupt"]})
        asked = take("restart.json")
        if asked and int(asked.get("token", 0)) > self.seen["restart"]:
            self.seen["restart"] = int(asked["token"])
            self.restart()
            put("ack-restart.json", {"token": self.seen["restart"]})

    def restart(self):
        try:
            self.kc.stop_channels()
        except Exception:
            pass
        self.km.restart_kernel(now=True)
        self.connect()
        if self.cell is not None:
            self.cell["error"] = ("KernelRestart: the kernel was restarted while this "
                                  "cell was running")
            self.cell["ok"] = False
            self.finish()

    # -- the loop ----------------------------------------------------------

    def serve(self):
        (HERE / "ready").write_text("1", encoding="utf-8")
        while True:
            self.controls()
            if self.cell is None:
                asked = take("req-%d.json" % (self.done + 1))
                if asked is None:
                    time.sleep(POLL)
                    continue
                self.begin(self.done + 1, asked.get("code", ""))
            self.pump()


def main():
    where = sys.argv[1] if len(sys.argv) > 1 else "."
    if where not in ("", "."):
        os.makedirs(where, exist_ok=True)
        os.chdir(where)
    os.environ.setdefault("MPLBACKEND", "Agg")
    driver = Driver()
    try:
        driver.open()
    except Exception as exc:
        put("failed.json", {"message": "no kernel on this workspace: %s" % exc})
        traceback.print_exc()
        raise SystemExit(1)
    driver.serve()


if __name__ == "__main__":
    main()
'''


class KernelHost:
    """The protocol's five kernel methods, for a backend that has its file and job
    primitives.

    Both backends mix this in, so "`LocalBackend` and `HostedBackend` agree" is a
    property of there being one implementation rather than a thing to keep checking.
    The kernel is created on first use and put down by the backend's `close`, which
    is what "a kernel per desk run, torn down with the run" means in code.
    """

    _channel: KernelChannel | None = None

    def _kernel(self) -> KernelChannel:
        if self._channel is None:
            self._channel = KernelChannel(self)
        return self._channel

    def kernel_start(self, cwd: str) -> str:
        return self._kernel().start(cwd)

    def kernel_run(self, code: str, timeout_s: int = CELL_TIMEOUT_S) -> CellResult:
        return self._kernel().run(code, timeout_s)

    def kernel_poll(self) -> CellResult:
        return self._kernel().poll()

    def kernel_interrupt(self) -> None:
        self._kernel().interrupt()

    def kernel_restart(self) -> None:
        self._kernel().restart()

    def kernel_stop(self) -> None:
        """Put this workspace's kernel down, if it ever brought one up."""
        if self._channel is not None:
            self._channel.stop()
