"""DockerBackend -- the hosted workspace's image, run on this machine.

`LocalBackend` runs against whatever OpenFOAM the machine has, and everything above the
backend assumes the hosted one: `/work`, OpenFOAM ESI v2512, the toolbox's Python
packages, no outbound network. On a native install a shell command that names `/work`
reaches nothing, or something else (docs/issues.md #31), and the version and the
packages are whatever they happen to be.

So this runs the same image the hosted workspace runs, in a container on this machine,
with the workspace directory mounted at `/work`. The prompt, the toolbox and every path
the model writes are then true as written, the container is a sandbox the model's
commands cannot climb out of, and Docker is the one thing the person installs, on
Linux, macOS or Windows alike.

Files are read and written on the host side of the mount, which is the same directory
and needs no round trip; commands and jobs run inside the container. A job is a few
files under `/work/.logs/jobs/<id>/` (its pid, its log, its exit code once it has one),
so a job outlives the session that started it and a resumed session can still look at
it, which an in-memory process table could not (docs/issues.md #24).

The container is left running when a session ends, because jobs may still be running in
it. It holds no state of its own: removing it loses nothing but the jobs in flight.

`OPENREYNOLDS_IMAGE` chooses the image; `OPENREYNOLDS_LOCAL_WORK` the directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

from .base import WORKSPACE_ROOT, BackendError, EXEC_MAX_TIMEOUT_S, ExecResult, JobStatus, Stat
from .local import LOGS_DIR, MAX_OUTPUT_BYTES, LocalBackend, _now

DEFAULT_IMAGE = "ghcr.io/inviscidai/reynolds-workspace:2512"
"""Built from docker/workspace by .github/workflows/workspace-image.yml. The hosted
workspace runs the same image with its job supervisor added."""

ROOT_SHIM = "/usr/local/lib/libfoamnotadmin.so"
"""Lets OpenFOAM compile code at run time as root; built into the image."""

JOBS_DIR = f"{LOGS_DIR}/jobs"
"""Under the workspace: one directory per job."""

SOURCE_FOAM = (
    '[ -n "${WM_PROJECT_DIR:-}" ] || '
    "for rc in /usr/lib/openfoam/openfoam*/etc/bashrc /openfoam/bashrc; do "
    '[ -f "$rc" ] && { source "$rc" >/dev/null 2>&1; break; }; done; true'
)
"""OpenFOAM's environment, for an image that does not source it for every shell. The
hosted image does (`/etc/profile.d`), and then this does nothing."""

_START_WAIT_S = 10.0
"""How long a job is given to write its pid before starting it is called a failure."""

_MAX_LINKS = 40
"""Symbolic links followed in one path before it is called a loop, as Linux does."""

_NO_DAEMON = ("Cannot connect to the Docker daemon", "permission denied while trying to connect",
              "error during connect", "docker daemon is not running")
"""What the docker client says when it has no daemon to talk to, or may not."""

_GONE = ("is not running", "No such container")
"""What `docker exec` says when the container went away under a session."""


class DockerBackend(LocalBackend):
    """A workspace in a container on this machine, with `/work` mounted from a directory."""

    def __init__(self, root: str | Path | None = None, image: str | None = None,
                 docker: str | None = None):
        super().__init__(root, bashrc="")
        self.workspace_root = WORKSPACE_ROOT
        self.image = image or os.environ.get("OPENREYNOLDS_IMAGE") or DEFAULT_IMAGE
        self.docker = docker or shutil.which("docker") or "docker"
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:10]
        self.container = f"openreynolds-{digest}"
        self._up = False
        self._watchers: dict[str, threading.Thread] = {}
        self._host(JOBS_DIR).mkdir(parents=True, exist_ok=True)
        self._host(".home").mkdir(exist_ok=True)

    # -- the container ---------------------------------------------------------

    def _docker(self, *args: str, timeout: float = 60, check: bool = True,
                ) -> subprocess.CompletedProcess:
        try:
            done = subprocess.run([self.docker, *args], capture_output=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise BackendError(
                "docker is not installed, or not on PATH; a local workspace runs in it "
                "(https://docs.docker.com/get-docker/)", "docker_missing",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"docker {args[0]} took over {timeout:.0f}s", "docker_timeout") from exc
        if check and done.returncode != 0:
            message = " ".join((done.stderr or done.stdout).decode("utf-8", "replace").split())
            if any(sign in message for sign in _NO_DAEMON):
                raise BackendError(
                    f"Docker is installed but this user cannot reach it ({message}). Is "
                    "Docker running? On Linux, is this user in the docker group "
                    "(https://docs.docker.com/engine/install/linux-postinstall/)?",
                    "docker_unavailable",
                )
            raise BackendError(f"docker {args[0]}: {message}", "docker_failed")
        return done

    def _state(self) -> tuple[str, str] | None:
        """(running?, image) of this workspace's container, or None when there is none."""
        done = self._docker(
            "inspect", "-f", "{{.State.Running}} {{.Config.Image}}", self.container,
            check=False,
        )
        if done.returncode != 0:
            return None
        running, _, image = done.stdout.decode().strip().partition(" ")
        return running, image

    def run_args(self) -> list[str]:
        """`docker run` for this workspace's container, after the word `run`."""
        args = [
            "-d", "--name", self.container, "--init",
            # The hosted workspace has no outbound network, the prompt says so, and a
            # model on someone's own machine should not have it either.
            "--network", "none",
            # Its own processes start outside the workspace, so the probe for another
            # session's work (`cli._neighbour_work`) does not find them in it. Every
            # command names its own directory.
            "-v", f"{self.root}:{WORKSPACE_ROOT}", "-w", "/",
            "-e", f"HOME={WORKSPACE_ROOT}/.home",
            "--label", "openreynolds.workspace=" + str(self.root),
        ]
        if hasattr(os, "getuid"):
            # Files the solver writes land in the person's own directory, owned by
            # them rather than by root. Docker Desktop maps ownership itself.
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        else:
            # Root in the container, where OpenFOAM refuses to compile #codeStream
            # and coded boundary conditions; the image's shim answers its "am I
            # root?" with no (docker/workspace/Dockerfile).
            args += ["-e", f"LD_PRELOAD={ROOT_SHIM}"]
        return args + ["--entrypoint", "sleep", self.image, "infinity"]

    @staticmethod
    def _gone(done: subprocess.CompletedProcess) -> bool:
        """Did `docker exec` fail because the container is not there to run in? Docker
        restarting, the machine sleeping, or someone's `docker stop`."""
        if done.returncode == 0:
            return False
        said = (done.stderr or b"").decode("utf-8", "replace")
        return said.startswith("Error response from daemon") and any(g in said for g in _GONE)

    def ensure_up(self) -> None:
        """Start this workspace's container if it is not running."""
        if self._up:
            return
        state = self._state()
        if state is not None and state[0] == "true":
            self._up = True
            return
        if state is not None:
            # Stopped: made fresh, so a changed image or mount takes effect. Nothing is
            # lost; the workspace is the mounted directory.
            self._docker("rm", "-f", self.container, check=False)
        if self._docker("image", "inspect", self.image, check=False).returncode != 0:
            try:
                self._docker("pull", self.image, timeout=3600)
            except BackendError as exc:
                if exc.code != "docker_failed":
                    raise
                raise BackendError(
                    f"could not pull the workspace image {self.image} ({exc.message}). "
                    "OPENREYNOLDS_IMAGE names another one.", "image_missing",
                ) from exc
        self._docker("run", *self.run_args(), timeout=120)
        self._up = True

    def _resolve(self, path: str | None) -> Path:
        """A workspace path as a host path, with its symbolic links followed the way
        the container would follow them, and refused if they lead out.

        Files are read and written here on the host, and a link the model made in the
        container means something else here: `ln -s / /work/s1/root` points at the
        container's root there and at this machine's here, so writing through it
        would put the model's bytes anywhere the person can write. A link to an
        absolute `/work/...` path is the workspace in both places and is followed."""
        lexical = super()._resolve(path)
        parts = list(lexical.relative_to(self.root).parts)
        current, followed = self.root, 0
        while parts:
            part = parts.pop(0)
            if part == "..":
                if current == self.root:
                    raise BackendError(f"{path} leads outside the workspace", "outside_workspace")
                current = current.parent
                continue
            step = current / part
            if not step.is_symlink():
                current = step
                continue
            followed += 1
            if followed > _MAX_LINKS:
                raise BackendError(f"{path}: too many symbolic links", "outside_workspace")
            target = os.readlink(step)
            if target.startswith("/"):
                if target != WORKSPACE_ROOT and not target.startswith(WORKSPACE_ROOT + "/"):
                    raise BackendError(
                        f"{path} leads outside the workspace (a link to {target})",
                        "outside_workspace",
                    )
                current = self.root
                target = target[len(WORKSPACE_ROOT):].lstrip("/")
            parts = [p for p in PurePosixPath(target).parts if p not in ("", ".")] + parts
        return current

    def _host(self, *relative: str) -> Path:
        """A path under the workspace, as `_resolve` finds it."""
        return self._resolve("/".join((WORKSPACE_ROOT, *relative)))

    def put_tree(self, local_dir: Path, remote_dir: str) -> None:
        # File by file, so a link already in the workspace is followed only as far as
        # `_resolve` allows; copytree would follow it anywhere.
        source = Path(local_dir)
        for path in sorted(source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                target = self._resolve(f"{remote_dir.rstrip('/')}/{relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)  # with its mode: the toolbox has scripts

    def get_tree(self, remote_paths: list[str], local_dir: Path, *, timeout: float = 300.0,
                 max_attempts: int | None = None, via: str | None = None) -> list[Path]:
        del timeout, max_attempts, via
        out: list[Path] = []
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for path in remote_paths:
            try:
                source = self._resolve(path)
            except BackendError as exc:
                # A link out of the workspace has nothing here to copy, as a missing
                # file has not; failing on it made the mirror retry it for ever.
                if exc.code == "outside_workspace":
                    continue
                raise
            if not source.exists():
                continue
            landing = destination / source.name
            if source.is_dir():
                # Links inside are copied as links, not followed: one that points out
                # of the workspace would otherwise copy this machine into the mirror.
                shutil.copytree(source, landing, symlinks=True, dirs_exist_ok=True)
                out.extend(p for p in landing.rglob("*") if p.is_file() and not p.is_symlink())
            else:
                landing.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, landing)
                out.append(landing)
        return out

    def _cpath(self, host: Path) -> str:
        """A host path under the workspace as the container sees it."""
        relative = host.relative_to(self.root).as_posix()
        return WORKSPACE_ROOT if relative == "." else f"{WORKSPACE_ROOT}/{relative}"

    # -- shell -----------------------------------------------------------------

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120,
             *, background: bool = False) -> ExecResult:
        del background  # nothing billed here; see LocalBackend.exec
        limit = min(int(timeout_s or 120), EXEC_MAX_TIMEOUT_S)
        where = self._resolve(cwd)
        where.mkdir(parents=True, exist_ok=True)
        self.ensure_up()
        log = self._host(LOGS_DIR, f"exec-{uuid.uuid4().hex[:12]}.log")
        # Timed out inside the container as well as here: killing the docker client
        # leaves the command running in the container.
        inner = f"{SOURCE_FOAM}\n{cmd}"
        argv = [
            self.docker, "exec", "-w", self._cpath(where), self.container,
            "timeout", "-k", "5", str(limit), "bash", "-lc", inner,
        ]
        try:
            done = subprocess.run(argv, capture_output=True, timeout=limit + 30)
            if self._gone(done):
                self._up = False
                self.ensure_up()
                done = subprocess.run(argv, capture_output=True, timeout=limit + 30)
            blob, code = done.stdout + done.stderr, done.returncode
        except subprocess.TimeoutExpired as expired:
            blob, code = (expired.stdout or b"") + (expired.stderr or b""), 124
        except OSError as exc:
            raise BackendError(str(exc), "exec_failed") from exc
        if code == 124:
            blob += f"\n[timed out after {limit}s]".encode()
        log.write_bytes(blob)
        text = blob[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")
        return ExecResult(code, text, len(blob) > MAX_OUTPUT_BYTES, self._cpath(log))

    def stat(self, path: str, *, timeout: float = 300.0, max_attempts: int | None = None) -> Stat:
        found = super().stat(path, timeout=timeout, max_attempts=max_attempts)
        return dataclasses.replace(found, path=self._cpath(Path(found.path)))

    # -- jobs ------------------------------------------------------------------
    #
    # A job is a directory: `cmd`, `meta.json` (name, start, kill_on), `pid` once it
    # runs, `log`, `rc` once it ends, and `end_reason` / `killed_by` when this side
    # stopped it. `run.sh` runs as its own session leader, so its pid is the process
    # group a kill reaches, mpirun's ranks included.

    def _jobdir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
            raise BackendError(f"no such job: {job_id}", "not_found")
        found = self._host(JOBS_DIR, job_id)
        if not found.is_dir():
            raise BackendError(f"no such job: {job_id}", "not_found")
        # This side reads and writes these files on the host, so a link put in place
        # of one would be followed out of the workspace; nothing here makes links.
        if any(entry.is_symlink() for entry in found.iterdir()):
            raise BackendError(f"job {job_id}'s files were replaced by links", "outside_workspace")
        return found

    def job_start(self, cmd: str, cwd: str | None = None, name: str | None = None,
                  kill_on: list[str] | None = None) -> str:
        where = self._resolve(cwd)
        where.mkdir(parents=True, exist_ok=True)
        self.ensure_up()
        job_id = uuid.uuid4().hex
        folder = self._host(JOBS_DIR, job_id)
        folder.mkdir(parents=True)
        inside = f"{WORKSPACE_ROOT}/{JOBS_DIR}/{job_id}"
        (folder / "cmd").write_text(cmd)
        (folder / "log").touch()
        (folder / "meta.json").write_text(json.dumps(
            {"name": name, "started_at": _now(), "kill_on": kill_on or []}
        ))
        (folder / "run.sh").write_text(
            "#!/bin/bash\n"
            f"echo $$ > {inside}/pid\n"
            f"{SOURCE_FOAM}\n"
            f'bash -lc "$(cat {inside}/cmd)" >> {inside}/log 2>&1\n'
            f"echo $? > {inside}/rc.tmp && mv {inside}/rc.tmp {inside}/rc\n"
        )
        start = ("exec", "-d", "-w", self._cpath(where), self.container,
                 "setsid", "bash", f"{inside}/run.sh")
        first = self._docker(*start, check=False)
        if self._gone(first):
            self._up = False
            self.ensure_up()
            self._docker(*start)
        elif first.returncode != 0:
            said = (first.stderr or first.stdout).decode("utf-8", "replace").strip()
            raise BackendError(f"docker exec: {said}", "docker_failed")
        deadline = time.monotonic() + _START_WAIT_S
        while not (folder / "pid").is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not (folder / "pid").is_file():
            raise BackendError(f"job {job_id} did not start within {_START_WAIT_S:.0f}s",
                               "job_failed")
        if kill_on:
            self._watch_job(job_id)
        return job_id

    def _pid(self, folder: Path) -> int | None:
        try:
            return int((folder / "pid").read_text().strip())
        except (OSError, ValueError):
            return None

    def _alive(self, pid: int) -> bool:
        done = self._docker("exec", self.container, "kill", "-0", str(pid), check=False)
        return done.returncode == 0

    def _signal(self, folder: Path, sig: str) -> None:
        pid = self._pid(folder)
        if pid is None:
            return
        number = "KILL" if sig.upper() in ("KILL", "9") else "TERM"
        self._docker("exec", self.container, "kill", f"-{number}", "--", f"-{pid}", check=False)

    def _watch_job(self, job_id: str) -> None:
        folder = self._jobdir(job_id)
        patterns = [re.compile(p) for p in json.loads((folder / "meta.json").read_text())["kill_on"]]

        def watch() -> None:
            seen = 0
            while not (folder / "rc").exists():
                try:
                    with open(folder / "log", "rb") as handle:
                        handle.seek(seen)
                        fresh = handle.read()
                        seen = handle.tell()
                except OSError:
                    return
                for line in fresh.decode("utf-8", "replace").splitlines():
                    if any(p.search(line) for p in patterns):
                        (folder / "killed_by").write_text(line[:500])
                        (folder / "end_reason").write_text("kill_on_match")
                        self._signal(folder, "TERM")
                        return
                time.sleep(1.0)

        thread = threading.Thread(target=watch, daemon=True)
        self._watchers[job_id] = thread
        thread.start()

    def job_status(self, job_id: str) -> JobStatus:
        folder = self._jobdir(job_id)
        meta = json.loads((folder / "meta.json").read_text())
        read = lambda name: (folder / name).read_text().strip() if (folder / name).is_file() else None  # noqa: E731
        rc, reason, killed_by = read("rc"), read("end_reason"), read("killed_by")
        log = folder / "log"
        size = log.stat().st_size if log.exists() else 0
        if rc is None:
            pid = self._pid(folder)
            if pid is not None and self._state() and self._alive(pid):
                return JobStatus(job_id=job_id, status="running", name=meta.get("name"),
                                 started_at=meta.get("started_at"), log_size=size)
            # Gone with no exit code: the container went away under it.
            reason = reason or "killed_externally"
            code = None
        else:
            code = int(rc)
            reason = reason or ("completed" if code == 0
                                else "killed_externally" if code in (137, 143)
                                else "failed")
        ended = read("ended_at")
        if ended is None:
            ended = _now()
            (folder / "ended_at").write_text(ended)
        status = "exited" if reason in ("completed", "failed", "killed_externally") else "killed"
        return JobStatus(
            job_id=job_id, status=status, name=meta.get("name"), exit_code=code,
            end_reason=reason, started_at=meta.get("started_at"), ended_at=ended,
            log_size=size, killed_by=killed_by,
        )

    def job_tail(self, job_id: str, offset: int = 0) -> tuple[str, int, bool]:
        folder = self._jobdir(job_id)
        with open(folder / "log", "rb") as handle:
            handle.seek(max(0, int(offset)))
            blob = handle.read(MAX_OUTPUT_BYTES)
            where = handle.tell()
            eof = handle.read(1) == b""
        return blob.decode("utf-8", "replace"), where, eof

    def job_kill(self, job_id: str, signal: str = "TERM") -> JobStatus:
        folder = self._jobdir(job_id)
        if not (folder / "rc").exists():
            if not (folder / "end_reason").exists():
                (folder / "end_reason").write_text("killed_by_client")
            self._signal(folder, signal)
            deadline = time.monotonic() + 10
            while not (folder / "rc").exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            if not (folder / "rc").exists():
                self._signal(folder, "KILL")
        return self.job_status(job_id)

    def shutdown(self) -> None:
        """Put the container down: the session found nothing running in it. It holds
        no state of its own, so the next session starts a fresh one on the same
        directory."""
        self._docker("rm", "-f", self.container, check=False)
        self._up = False

    def shell_command(self) -> list[str]:
        """What a person runs to get a shell in this workspace, for the README and doctor."""
        return [self.docker, "exec", "-it", self.container, "bash", "-l"]

    def close(self) -> None:
        """Jobs outlive the session, and so does the container they run in."""
        self.kernel_stop()


def describe(backend: DockerBackend) -> str:
    """One line for the session header."""
    return (f"local workspace: {backend.root} as {WORKSPACE_ROOT} in {backend.image} "
            f"(shell: {shlex.join(backend.shell_command())})")
