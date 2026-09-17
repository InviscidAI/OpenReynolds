"""The cell channel, against a kernel that is really running.

Nothing here is a mock of the kernel. The point of the channel is the handful of
behaviours a fake would have to assume -- that a binding survives a step, that a slow
cell is not killed for being slow, that an interrupt leaves the session usable, that a
picture comes back without anyone naming a file -- so every test below starts a real
kernel in a real workspace and asks it.

`HostedBackend` is exercised the way `test_hosted.py` exercises it: a stand-in for the
client, answering the service's endpoints. Here that stand-in serves them out of a
`LocalBackend`, so the hosted code path -- its own `put_file`, `get_file`, `stat` and
job calls, and the JSON bodies it parses -- runs for real against a real kernel.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx2 as httpx
import pytest

from openreynolds import images
from openreynolds.backend.base import BackendError
from openreynolds.backend.hosted import HostedBackend
from openreynolds.backend.kernel import CELL_TIMEOUT_S, CellResult, KernelChannel
from openreynolds.backend.local import LocalBackend

pytest.importorskip("ipykernel", reason="the cell channel needs a kernel to talk to")
pytest.importorskip("jupyter_client", reason="the cell channel needs a kernel to talk to")


# -- the two workspaces --------------------------------------------------------


class Loopback:
    """The service's endpoints, answered out of a local workspace.

    Only the ones the cell channel uses, and each one shaped the way the real service
    shapes it, because what is under test is `HostedBackend`'s own reading of those
    bodies. A missing file raises `BackendError` exactly where `FoamdClient.request`
    would raise it on a 404.
    """

    def __init__(self, local: LocalBackend):
        self.local = local
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        params = kwargs.get("params") or {}
        body = kwargs.get("json") or {}
        parts = [p for p in path.split("/") if p]
        try:
            return self._route(method, parts, params, body, kwargs.get("content"))
        except BackendError:
            raise
        except FileNotFoundError as exc:  # pragma: no cover - the local backend raises above
            raise BackendError(str(exc), "not_found", 404) from exc

    def _route(self, method, parts, params, body, content):
        if parts[:2] == ["v1", "instances"] and parts[3:] == ["files"]:
            return self._files(method, params, content)
        if parts[:2] == ["v1", "instances"] and parts[3:] == ["exec"]:
            ran = self.local.exec(body["cmd"], body.get("cwd"), body.get("timeout_s", 120))
            return httpx.Response(200, json={
                "exit_code": ran.exit_code, "output": ran.output,
                "truncated": ran.truncated, "log_path": ran.log_path, "stderr": "",
            })
        if parts[:2] == ["v1", "instances"] and parts[3:] == ["jobs"]:
            job = self.local.job_start(body["cmd"], body.get("cwd"), body.get("name"))
            return httpx.Response(200, json={"job_id": job})
        if parts[:2] == ["v1", "jobs"]:
            return self._jobs(parts[2], parts[3:], params)
        raise AssertionError(f"the channel asked for an endpoint nobody serves: {parts}")

    def _files(self, method, params, content):
        path = params["path"]
        if method == "PUT":
            self.local.put_file(path, content)
            return httpx.Response(200, json={"ok": True})
        if params.get("stat"):
            found = self.local.stat(path)
            return httpx.Response(200, json={
                "path": found.path, "type": found.type, "size": found.size,
                "mtime": found.mtime, "entries": found.entries,
            })
        data = self.local.get_file(path, int(params.get("offset", 0)), params.get("limit"))
        return httpx.Response(200, content=data)

    def _jobs(self, job_id, tail, params):
        if tail == ["log"]:
            data, offset, eof = self.local.job_tail(job_id, int(params.get("offset", 0)))
            return httpx.Response(200, json={"data": data, "next_offset": offset, "eof": eof})
        status = (self.local.job_kill(job_id) if tail == ["kill"]
                  else self.local.job_status(job_id))
        return httpx.Response(200, json={
            "job_id": status.job_id, "status": status.status, "name": status.name,
            "exit_code": status.exit_code, "end_reason": status.end_reason,
            "started_at": status.started_at, "ended_at": status.ended_at,
            "log_size": status.log_size,
        })


@pytest.fixture
def local(tmp_path):
    backend = LocalBackend(root=tmp_path / "work", bashrc="")
    yield backend
    backend.close()


@pytest.fixture
def hosted(tmp_path):
    """`HostedBackend`, talking to a workspace that happens to be this machine."""
    under = LocalBackend(root=tmp_path / "work", bashrc="")
    backend = HostedBackend(Loopback(under), "instance-under-test")
    yield backend
    backend.close()
    under.close()


@pytest.fixture
def desk(local):
    """A workspace with a kernel up in a case directory, as a run would have it."""
    case = f"{local.workspace_root}/case"
    local.exec("mkdir -p case")
    local.kernel_start(case)
    return local


# -- state ---------------------------------------------------------------------


def test_a_binding_survives_the_step_and_does_not_survive_a_restart(desk):
    """Both halves. The first is the feature; the second is what makes the cell log
    the source of truth rather than the kernel."""
    assert desk.kernel_run("x = 1", 60).ok
    carried = desk.kernel_run("print(x + 1)", 60)
    assert carried.ok and carried.stdout.strip() == "2"

    desk.kernel_restart()

    gone = desk.kernel_run("print(x)", 60)
    assert gone.ok is False
    assert "NameError" in gone.error
    assert desk.kernel_run("y = 2\nprint(y)", 60).stdout.strip() == "2", (
        "a restarted kernel is a working kernel, not a dead one"
    )


def test_the_kernel_starts_in_the_case_directory(desk):
    result = desk.kernel_run("import os\nprint(os.getcwd())", 60)
    assert result.stdout.strip().endswith("/case")


# -- failure -------------------------------------------------------------------


def test_a_raising_cell_is_a_result_and_not_an_exception(desk):
    """A failure is work handed to the desk, exactly as a non-zero exit code is."""
    result = desk.kernel_run("def inner():\n    raise ValueError('no such face')\ninner()", 60)
    assert isinstance(result, CellResult)
    assert result.ok is False
    assert result.error.startswith("ValueError")
    assert "no such face" in result.error
    assert "inner" in result.traceback and "ValueError" in result.traceback


def test_a_failed_cell_leaves_the_session_usable(desk):
    desk.kernel_run("kept = 3", 60)
    assert desk.kernel_run("raise RuntimeError('boom')", 60).ok is False
    after = desk.kernel_run("print(kept)", 60)
    assert after.ok and after.stdout.strip() == "3"


# -- pictures ------------------------------------------------------------------


def test_a_cell_that_draws_returns_png_bytes_without_naming_a_file(desk):
    """The whole reason the channel exists in preference to bash: nobody scrapes a
    filename out of the cell, and the cell never writes one."""
    pytest.importorskip("matplotlib")
    result = desk.kernel_run(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([0, 1, 2], [0, 1, 4])\n"
        "plt.show()\n",
        120,
    )
    assert result.ok, result.error
    assert len(result.images) == 1
    picture = result.images[0]
    assert picture[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(picture) <= images.MAX_ATTACH_BYTES
    listing = desk.exec("ls case").output
    assert ".png" not in listing, "the cell wrote no file, and none appeared"


def test_a_picture_is_downscaled_by_the_existing_transport_rule(desk, monkeypatch):
    """`images.downscale`, not a second copy of it living in the channel."""
    pytest.importorskip("matplotlib")
    seen: list[str] = []
    real = images.downscale

    def watched(data: bytes, media: str, *args, **kwargs):
        seen.append(media)
        return real(data, media, *args, **kwargs)

    monkeypatch.setattr(images, "downscale", watched)
    result = desk.kernel_run(
        "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()\n", 120
    )
    assert result.images
    assert seen == ["image/png"]


# -- the window ----------------------------------------------------------------


def test_a_cell_that_outruns_the_window_is_not_killed(desk):
    """On expiry the result says so and carries what streamed; the cell keeps going,
    which is asserted by watching it finish with its binding in place."""
    started = time.monotonic()
    expired = desk.kernel_run(
        "import time\n"
        "print('begun', flush=True)\n"
        "time.sleep(8)\n"
        "slowly = 'arrived'\n"
        "print('done')\n",
        1,
    )
    waited = time.monotonic() - started
    assert expired.still_running is True
    assert expired.ok is True, "nothing has failed; it is merely still going"
    # Bounded generously against the cell's eight seconds, because what is under test
    # is "the wait ended early and the cell did not", not the precision of a clock on
    # a loaded machine.
    assert 0.9 <= expired.seconds < 5.0
    assert waited < 5.0, "the caller was not made to wait for the cell"
    assert "begun" in expired.stdout, "whatever streamed so far comes back"

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        look = desk.kernel_poll()
        if not look.still_running:
            break
        time.sleep(0.1)
    else:  # pragma: no cover - the cell sleeps eight seconds
        pytest.fail("the cell never finished")
    assert look.ok
    assert "done" in look.stdout, "a poll brings back what was printed since the last look"

    kept = desk.kernel_run("print(slowly)", 60)
    assert kept.stdout.strip() == "arrived", "it really did run to the end"


def test_a_hung_cell_is_recoverable_by_an_interrupt(desk):
    desk.kernel_run("before = 'here'", 60)
    hung = desk.kernel_run("while True:\n    pass\n", 1)
    assert hung.still_running is True

    desk.kernel_interrupt()

    after = desk.kernel_poll()
    assert after.still_running is False
    assert after.ok is False and "KeyboardInterrupt" in after.error

    usable = desk.kernel_run("print(before)", 60)
    assert usable.ok and usable.stdout.strip() == "here", (
        "the same session, with the bindings from before the bad cell"
    )


def test_no_cell_can_change_its_own_window(desk):
    """The window is the caller's, from configuration, the way `STEP_TIMEOUT_S` is.

    Three ways in are closed here: the kernel's own namespace, the request the channel
    writes (which carries the code and nothing else), and the signature, whose default
    is a module constant rather than anything the cell can reach.
    """
    desk.kernel_run("timeout_s = 9999\nCELL_TIMEOUT_S = 9999\nwindow = 9999", 60)
    started = time.monotonic()
    result = desk.kernel_run(
        "# timeout_s: 600\ntimeout_s = 600\nimport time\ntime.sleep(30)\n", 1
    )
    assert result.still_running is True
    assert time.monotonic() - started < 5.0, "the cell's opinion did not widen the window"

    channel = desk._kernel()
    request = json.loads(
        desk.get_file(f"{channel._dir}/req-2.json").decode("utf-8")
    )
    assert set(request) == {"code"}, (
        "the request carries the code and nothing a cell could set to buy itself time"
    )
    assert isinstance(CELL_TIMEOUT_S, int)
    desk.kernel_interrupt()


# -- background work -----------------------------------------------------------


def test_background_work_outlives_the_cell_that_started_it(desk):
    """A mesher communicates through files and loses nothing by going to the
    background -- the pattern the brief's `nohup ... &` rule already assumes."""
    started = desk.kernel_run(
        "import subprocess\n"
        "runner = subprocess.Popen(\n"
        "    ['bash', '-c', 'for i in 1 2 3 4 5 6 7 8; do echo tick $i >> mesh.log; sleep 0.4; done'])\n"
        "print('launched')\n",
        60,
    )
    assert started.ok, started.error
    time.sleep(1.0)
    later = desk.kernel_run(
        "print('running' if runner.poll() is None else 'finished')\n"
        "print(open('mesh.log').read().strip().splitlines()[-1])\n",
        60,
    )
    assert later.ok, later.error
    lines = later.stdout.strip().splitlines()
    assert lines[0] == "running", "the subprocess outlived the cell that started it"
    assert lines[1].startswith("tick"), "and its log is readable from a later cell"
    desk.kernel_run("runner.kill()", 60)


# -- bounded output ------------------------------------------------------------


def test_a_long_output_is_cut_in_the_middle_and_kept_whole_at_the_log_path(desk, monkeypatch):
    """The news is at both ends -- the command that failed and the summary after it."""
    monkeypatch.setattr("openreynolds.backend.kernel.MAX_OUTPUT_BYTES", 900)
    result = desk.kernel_run(
        "for i in range(400):\n    print('line %04d padded out a little' % i)\n", 60
    )
    assert result.ok
    assert result.truncated is True
    assert "bytes cut" in result.stdout
    assert "line 0000" in result.stdout and "line 0399" in result.stdout
    assert len(result.stdout) < 1400

    whole = desk.get_file(result.log_path).decode("utf-8")
    assert whole.count("\n") == 400, "all of it is still there"


# -- the transport stays invisible ---------------------------------------------


def test_the_result_names_no_socket_no_port_and_no_session():
    """`CellResult` is about the cell. A caller cannot learn from it that there is a
    kernel somewhere, let alone reach one."""
    fields = set(CellResult.__dataclass_fields__)
    assert fields == {
        "ok", "stdout", "stderr", "error", "traceback", "images",
        "seconds", "still_running", "truncated", "log_path",
    }
    forbidden = ("socket", "port", "session", "fd", "descriptor", "connection",
                 "host", "url", "address", "channel", "kernel", "pid", "handle")
    for name in fields:
        assert not any(word in name.lower() for word in forbidden), name


def test_the_channel_talks_through_the_protocol_and_nothing_else():
    """It is written against `put_file`/`get_file`/`stat`/the job calls, which is what
    lets one implementation serve both backends."""
    import sys
    from pathlib import Path

    source = Path(sys.modules[KernelChannel.__module__].__file__)
    text = source.read_text(encoding="utf-8")
    for token in ("httpx", "http://", "https://", "Bearer", "modal", "foamd"):
        assert token.lower() not in text.lower(), f"the channel names {token!r}"


def test_a_workspace_with_no_kernel_says_so(local):
    with pytest.raises(BackendError) as raised:
        local.kernel_run("x = 1", 5)
    assert "no kernel" in str(raised.value).lower()


# -- the two backends agree ----------------------------------------------------

SEQUENCE = [
    "x = 6",
    "print(x * 7)",
    "raise ValueError('the same on both')",
    "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()",
    "print(x)",
]


def _walk(backend) -> list[tuple]:
    backend.exec("mkdir -p case")
    backend.kernel_start(f"{backend.workspace_root}/case")
    seen = []
    for cell in SEQUENCE:
        got = backend.kernel_run(cell, 120)
        seen.append((got.ok, got.error, got.stdout, len(got.images), got.still_running))
    return seen


def test_the_local_and_hosted_backends_agree(local, hosted):
    """Divergence here is the class of bug that only appears in production."""
    pytest.importorskip("matplotlib")
    assert _walk(local) == _walk(hosted)


def test_the_hosted_backend_really_did_go_through_its_own_client(hosted):
    """Guard on the test above: a loopback that quietly fell through to the local
    backend would make the agreement meaningless."""
    hosted.exec("mkdir -p case")
    session = hosted.kernel_start("/work/case")
    assert session
    result = hosted.kernel_run("print('over the wire')", 60)
    assert result.ok and result.stdout.strip() == "over the wire"
    assert hosted._kernel()._dir.startswith("/work/.kernel/"), (
        "the hosted backend addressed the workspace the way the service does"
    )


def test_a_poll_that_catches_the_cell_finishing_carries_the_whole_of_its_output(desk):
    """The completed result is the only one anybody gets, so it has to be complete.

    `done-N.json` and `out-N.txt` are separate files. A poll can see the first before the
    last flush of the second is visible, and with a delta the tail is then lost for good,
    because the caller takes the completed result and stops asking. It showed up as a
    1-in-3 flake in `test_a_cell_that_outruns_the_window_is_not_killed`, where the poll
    that caught the cell finishing came back without the `print('done')` that finished it.

    The desk now has a `poll_cell` tool hanging off this, so the failure would have been
    a cell reported finished with nothing in it -- which is the confusion the tool exists
    to end.
    """
    expired = desk.kernel_run(
        "import time\n"
        "print('first', flush=True)\n"
        "time.sleep(4)\n"
        "print('last')\n",
        1,
    )
    assert expired.still_running is True and "first" in expired.stdout

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        look = desk.kernel_poll()
        if not look.still_running:
            break
        time.sleep(0.1)
    else:  # pragma: no cover - the cell sleeps four seconds
        pytest.fail("the cell never finished")

    assert look.ok
    # Both halves, including the one printed before the last still-running poll: the
    # completed result repeats rather than resumes, because repetition is visible and
    # loss is not.
    assert "last" in look.stdout
    assert "first" in look.stdout
