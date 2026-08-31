"""HTTP client for the hosted OpenFOAM service.

Maps `Backend` 1:1 onto the service's published `/v1` contract. This is the only module
in the package that knows the contract exists.
"""

from __future__ import annotations

import io
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx2 as httpx

from .base import (
    EXEC_MAX_TIMEOUT_S,
    WORKSPACE_ROOT,
    Backend,
    BackendError,
    ExecResult,
    JobStatus,
    ResizeResult,
    Stat,
)

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
"""Statuses worth trying again rather than handing to the model as a failed tool call.

500 is here because of what it costs when it is not. Measured across two live studies,
18% of one study's tool calls came back `http_error (500)` -- clustered in the window
where the sandbox was being restarted underneath it -- while the other study, on the
same instance, saw none. Reproduced by hand: a plain read-only
`GET .../files?stat=1` answered 500 and then succeeded four seconds later, unchanged.

Every one of those reached the model as a tool error, and every tool error costs a full
turn: the whole conversation re-read, to learn that the backend blinked. The service's
own `supa.run()` already retries once on a dropped database socket for the same reason;
this is the same courtesy on the client side. A 500 that survives the retries still
reaches the model, so nothing is hidden -- only the flapping is absorbed."""

_MAX_ATTEMPTS = 5
_DEFAULT_RETRY_AFTER_S = 10.0
_SERVER_ERROR_RETRY_S = 1.0
"""A 500 is a hiccup, not a queue, so it is retried quickly rather than backed off from
the way a 429 or a cold-start 503 is."""


def _decode_error(response: httpx.Response) -> BackendError:
    """Decode either error shape the service produces.

    Application errors use `{"error", "message"}`; request-validation failures come from
    the web framework as a 422 `{"detail": [...]}`.
    """
    status = response.status_code
    try:
        body: Any = response.json()
    except Exception:
        text = (response.text or "").strip()
        return BackendError(text or f"HTTP {status}", code="http_error", status=status)

    if isinstance(body, dict) and "error" in body:
        return BackendError(
            str(body.get("message") or body["error"]),
            code=str(body["error"]),
            status=status,
        )

    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, dict):
                    loc = ".".join(str(p) for p in item.get("loc", []) if p != "body")
                    msg = item.get("msg", "invalid")
                    parts.append(f"{loc}: {msg}" if loc else str(msg))
                else:
                    parts.append(str(item))
            return BackendError("; ".join(parts), code="invalid_request", status=status)
        return BackendError(str(detail), code="invalid_request", status=status)

    return BackendError(str(body), code="http_error", status=status)


def _json(response: httpx.Response) -> Any:
    """Decode a success body, which is not always JSON.

    A long synchronous exec can come back as a bodyless 200 from something between
    here and the service. Without this the JSONDecodeError escapes BackendError
    handling entirely.
    """
    try:
        return response.json()
    except ValueError:
        snippet = (response.text or "").strip()[:200]
        detail = f": {snippet}" if snippet else " (the body was empty)"
        raise BackendError(
            f"the service answered {response.status_code} with a body that is not JSON"
            f"{detail}",
            code="bad_response",
            status=response.status_code,
        ) from None


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        raw = response.headers.get("retry-after")
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        if response.status_code == 503:
            return _DEFAULT_RETRY_AFTER_S
        if response.status_code == 500:
            # Nothing is being asked to queue: the same call succeeded four seconds
            # later, unchanged. Waiting the exponential backoff here would turn an
            # absorbed hiccup into a visible stall.
            return _SERVER_ERROR_RETRY_S
    return min(2.0**attempt, 30.0)


# -- signing in from a terminal --------------------------------------------------------


def _post_json(base_url: str, path: str, body: dict[str, Any], *, headers: dict[str, str] | None = None,
               transport: Any = None) -> httpx.Response:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport) as client:
        try:
            return client.post(path, json=body, headers=headers or {})
        except httpx.HTTPError as exc:
            raise BackendError(f"cannot reach {base_url}: {exc}", code="unreachable") from exc


def auth_config(base_url: str, *, transport: Any = None) -> dict[str, Any]:
    """Where the service's identity provider is. Public by design: it is what the
    service's own sign-in page fetches, so a terminal can sign in the same way."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport) as client:
        try:
            response = client.get("/dashboard/config.json")
        except httpx.HTTPError as exc:
            raise BackendError(f"cannot reach the service: {exc}", code="unreachable") from exc
    if response.status_code >= 400:
        raise _decode_error(response)
    body = _json(response)
    if not body.get("supabase_url") or not body.get("publishable_key"):
        raise BackendError("the service did not say where to sign in", code="no_auth_config")
    return body


def _auth_error(response: httpx.Response) -> BackendError:
    """The identity provider's two error shapes, reduced to a code and a sentence."""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {}
    code = str(body.get("error_code") or body.get("error") or "auth_error")
    message = str(body.get("msg") or body.get("error_description") or body.get("message") or response.text or code)
    if "invalid login credentials" in message.lower() or code == "invalid_grant":
        code = "invalid_credentials"
    return BackendError(message, code=code, status=response.status_code)


def password_session(supabase_url: str, publishable_key: str, email: str, password: str,
                     *, transport: Any = None) -> dict[str, Any]:
    """Sign in with email and password; the answer carries `access_token`."""
    response = _post_json(
        supabase_url, "/auth/v1/token?grant_type=password",
        {"email": email, "password": password},
        headers={"apikey": publishable_key}, transport=transport,
    )
    if response.status_code >= 400:
        raise _auth_error(response)
    return _json(response)


def sign_up(supabase_url: str, publishable_key: str, email: str, password: str,
            *, transport: Any = None) -> dict[str, Any] | None:
    """Create the account. Returns the session when the provider signs the new
    user straight in, `None` when it wants the address confirmed by email first."""
    response = _post_json(
        supabase_url, "/auth/v1/signup", {"email": email, "password": password},
        headers={"apikey": publishable_key}, transport=transport,
    )
    if response.status_code >= 400:
        raise _auth_error(response)
    body = _json(response)
    return body if body.get("access_token") else None


def accept_terms(base_url: str, jwt: str, *, transport: Any = None) -> dict[str, Any]:
    response = _post_json(base_url, "/v1/account/accept-terms", {},
                          headers={"Authorization": f"Bearer {jwt}"}, transport=transport)
    if response.status_code >= 400:
        raise _decode_error(response)
    return _json(response)


def mint_key(base_url: str, jwt: str, name: str, *, transport: Any = None) -> dict[str, Any]:
    """A service key for this machine, in exchange for a signed-in session. The
    plaintext comes back exactly once."""
    response = _post_json(base_url, "/v1/keys", {"name": name},
                          headers={"Authorization": f"Bearer {jwt}"}, transport=transport)
    if response.status_code >= 400:
        raise _decode_error(response)
    return _json(response)


def device_code(base_url: str, name: str | None = None, *, transport: Any = None) -> dict[str, Any]:
    """Ask the service for a code a person can approve in a browser.

    The one request made with no key at all: the answer carries the code to show, the
    address to approve it at, and how patiently to poll. `transport` is for tests."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport) as client:
        try:
            response = client.post("/v1/device/code", json={"name": name} if name else {})
        except httpx.HTTPError as exc:
            raise BackendError(f"cannot reach the service: {exc}", code="unreachable") from exc
    if response.status_code >= 400:
        raise _decode_error(response)
    return _json(response)


def device_token(base_url: str, code: str, *, transport: Any = None) -> dict[str, Any] | None:
    """Collect the key once the code has been approved; `None` while it has not.

    The service hands the plaintext over exactly once, so a caller that gets a dict
    must save it then and there."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport) as client:
        try:
            response = client.post("/v1/device/token", json={"device_code": code})
        except httpx.HTTPError as exc:
            raise BackendError(f"cannot reach the service: {exc}", code="unreachable") from exc
    if response.status_code == 428:
        return None
    if response.status_code >= 400:
        raise _decode_error(response)
    return _json(response)


class FoamdClient:
    """Low-level transport: auth, retries, error decoding, instance lifecycle."""

    def __init__(self, base_url: str, api_key: str, *, connect_timeout: float = 10.0):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0, connect=connect_timeout),
            # Redirects are followed. The service itself issues none, but the edge in
            # front of it does: a long `POST .../exec` comes back as a bare `303` with an
            # empty body, which this client turned into `bad_response (303): the body was
            # empty`. The command has usually *run* by then, so the caller is left unable
            # to tell a failure from a success -- a render that had already been written
            # looked like a render that had not, and the same picture came back three
            # times while the code that drew it was being changed underneath it.
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Issue one request, retrying cold starts and transient failures."""
        last_error: BackendError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            response = None
            try:
                response = self._client.request(method, path, timeout=timeout, **kwargs)
            except httpx.TimeoutException as exc:
                last_error = BackendError(f"request timed out: {exc}", code="timeout")
            except httpx.HTTPError as exc:
                last_error = BackendError(f"cannot reach the service: {exc}", code="unreachable")
            else:
                if response.status_code < 400:
                    return response
                last_error = _decode_error(response)
                if response.status_code not in _RETRY_STATUSES:
                    raise last_error

            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_retry_delay(response, attempt))

        raise last_error or BackendError("request failed", code="unreachable")

    # -- instances -------------------------------------------------------------

    def list_instances(self) -> list[dict[str, Any]]:
        return _json(self.request("GET", "/v1/instances"))

    def create_instance(self, cpu: float = 4.0, mem_gb: int = 8) -> str:
        body = _json(
            self.request("POST", "/v1/instances", json={"cpu": cpu, "mem_gb": mem_gb})
        )
        return body["instance_id"]

    def stop_instance(self, instance_id: str) -> dict[str, Any]:
        """Put the container down. The volume, and everything on it, stays."""
        return _json(
            self.request("POST", f"/v1/instances/{instance_id}/stop", timeout=120.0)
        )

    def start_instance(self, instance_id: str) -> dict[str, Any]:
        return _json(self.request("POST", f"/v1/instances/{instance_id}/start", timeout=180.0))

    # -- capture plane ---------------------------------------------------------

    def create_study(self, title: str | None, instance_id: str | None,
                     study_id: str | None = None, home: str | None = None) -> str:
        """Open the study on the platform, under this study's own id when given.

        The id used to be the service's to choose, so a study was named twice --
        `20260829-061843-9483` here and a uuid there -- and nothing could join the
        row to the directory it described. `home` is recorded for the same reason:
        a resume on a machine with no local state has to be able to find out which
        directory on the volume belongs to this study.
        """
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        if instance_id:
            payload["instance_id"] = instance_id
        if study_id:
            payload["id"] = study_id
        if home:
            payload["home"] = home
        return _json(self.request("POST", "/v1/studies", json=payload))["study_id"]

    def get_study(self, study_id: str) -> dict[str, Any]:
        """One study as the platform holds it: title, instance_id, home, created_at."""
        return _json(self.request("GET", f"/v1/studies/{study_id}"))

    def list_studies(self) -> list[dict[str, Any]]:
        return _json(self.request("GET", "/v1/studies"))

    def post_messages(self, study_id: str, messages: list[dict[str, Any]]) -> None:
        self.request("POST", f"/v1/studies/{study_id}/messages", json=messages)

    def post_result(self, study_id: str, payload: Any) -> None:
        self.request("POST", f"/v1/studies/{study_id}/results", json={"payload": payload})

    def post_artifact(
        self, study_id: str, filename: str, data: bytes, kind: str | None = None
    ) -> None:
        self.request(
            "POST",
            f"/v1/studies/{study_id}/artifacts",
            files={"file": (filename, data, "application/octet-stream")},
            data={"kind": kind} if kind else None,
            timeout=180.0,
        )


class HostedBackend(Backend):
    """One instance of the hosted service, addressed as a workspace."""

    workspace_root = WORKSPACE_ROOT

    def __init__(self, client: FoamdClient, instance_id: str):
        self._client = client
        self.instance_id = instance_id
        self.was_already_running = False
        """Whether somebody else's session had it up before this one asked."""

    def shutdown(self) -> None:
        """Put the container down. The volume is untouched, so nothing is lost."""
        self._client.stop_instance(self.instance_id)

    def close(self) -> None:
        self._client.close()

    def _instance_path(self, suffix: str) -> str:
        return f"/v1/instances/{self.instance_id}{suffix}"

    # -- commands --------------------------------------------------------------

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120) -> ExecResult:
        timeout_s = max(1, min(int(timeout_s), EXEC_MAX_TIMEOUT_S))
        payload: dict[str, Any] = {"cmd": cmd, "timeout_s": timeout_s}
        if cwd:
            payload["cwd"] = cwd
        body = _json(
            self._client.request(
                "POST", self._instance_path("/exec"), json=payload, timeout=timeout_s + 60.0
            )
        )
        return ExecResult(
            exit_code=body.get("exit_code", -1),
            output=body.get("output", ""),
            truncated=bool(body.get("truncated")),
            log_path=body.get("log_path"),
        )

    # -- files -----------------------------------------------------------------

    def put_file(self, path: str, data: bytes) -> None:
        self._client.request(
            "PUT",
            self._instance_path("/files"),
            params={"path": path},
            content=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=300.0,
        )

    def get_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        params: dict[str, Any] = {"path": path, "offset": offset}
        if limit is not None:
            params["limit"] = limit
        return self._client.request(
            "GET", self._instance_path("/files"), params=params, timeout=300.0
        ).content

    def stat(self, path: str) -> Stat:
        body = _json(
            self._client.request(
                "GET", self._instance_path("/files"), params={"path": path, "stat": 1}
            )
        )
        return Stat(
            path=body.get("path", path),
            type=body.get("type", ""),
            size=int(body.get("size", 0)),
            mtime=int(body.get("mtime", 0)),
            entries=list(body.get("entries") or []),
        )

    # -- trees -----------------------------------------------------------------

    def put_tree(self, local_dir: Path, remote_dir: str) -> None:
        archive = _tar_gz_of(local_dir)
        self._client.request(
            "POST",
            self._instance_path("/tar"),
            params={"mode": "extract", "dest": remote_dir},
            content=archive,
            headers={"Content-Type": "application/gzip"},
            timeout=300.0,
        )

    def get_tree(self, remote_paths: list[str], local_dir: Path) -> list[Path]:
        if not remote_paths:
            return []
        response = self._client.request(
            "POST",
            self._instance_path("/tar"),
            params={"mode": "pack", "paths": remote_paths},
            timeout=300.0,
        )
        return _extract_tar_gz(response.content, local_dir)

    # -- jobs ------------------------------------------------------------------

    def job_start(
        self,
        cmd: str,
        cwd: str | None = None,
        name: str | None = None,
        kill_on: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"cmd": cmd}
        if cwd:
            payload["cwd"] = cwd
        if name:
            payload["name"] = name
        if kill_on:
            payload["kill_on"] = kill_on
        body = _json(
            self._client.request(
                "POST", self._instance_path("/jobs"), json=payload, timeout=180.0
            )
        )
        return body["job_id"]

    def job_status(self, job_id: str) -> JobStatus:
        return _job_status(
            _json(self._client.request("GET", f"/v1/jobs/{job_id}", timeout=180.0))
        )

    def job_tail(self, job_id: str, offset: int = 0) -> tuple[str, int, bool]:
        body = _json(
            self._client.request(
                "GET", f"/v1/jobs/{job_id}/log", params={"offset": offset}, timeout=180.0
            )
        )
        return body.get("data", ""), int(body.get("next_offset", offset)), bool(body.get("eof"))

    def job_kill(self, job_id: str, signal: str = "TERM") -> JobStatus:
        """Signal a job's process group.

        The service marks the job killed whether or not the signal reached anything,
        so a returned status of `killed` is a record of the request, not proof that
        the work stopped. Confirming that is `stop`'s job.
        """
        return _job_status(
            _json(
                self._client.request(
                    "POST",
                    f"/v1/jobs/{job_id}/kill",
                    json={"signal": signal},
                    timeout=180.0,
                )
            )
        )

    def current_workspace_size(self) -> tuple[float, int]:
        """Get current workspace CPU and memory from foamd."""
        try:
            body = _json(
                self._client.request(
                    "GET", self._instance_path("/resources"), timeout=10.0
                )
            )
            cpu = float(body.get("cpu", 4.0))
            mem_gb = int(body.get("mem_gb", 8))
            return (cpu, mem_gb)
        except Exception as e:
            raise BackendError(f"could not get workspace size: {e}")

    def estimate_resize_cost(
        self, from_cpu: float, from_mem_gb: int, to_cpu: float, to_mem_gb: int
    ) -> int:
        """Estimate resize cost from foamd."""
        try:
            body = _json(
                self._client.request(
                    "POST",
                    self._instance_path("/pricing"),
                    json={
                        "from": {"cpu": from_cpu, "mem_gb": from_mem_gb},
                        "to": {"cpu": to_cpu, "mem_gb": to_mem_gb}
                    },
                    timeout=10.0
                )
            )
            return int(body.get("cost_delta_cents", 0))
        except Exception as e:
            raise BackendError(f"could not estimate resize cost: {e}")

    def can_afford(self, cost_delta_cents: int) -> bool:
        """Check if cost delta fits within monthly budget."""
        try:
            body = _json(
                self._client.request(
                    "GET", "/v1/account/budget", timeout=10.0
                )
            )
            used_cents = int(body.get("used_cents", 0))
            limit_cents = int(body.get("limit_cents", 0))
            remaining = limit_cents - used_cents
            return cost_delta_cents <= remaining
        except Exception:
            # If we can't check budget, assume we can't afford it (safe default)
            return False

    def resize_workspace(self, cpu: float, mem_gb: int, reason: str | None) -> ResizeResult:
        """Request a workspace resize from foamd."""
        try:
            payload = {
                "cpu": cpu,
                "mem_gb": mem_gb,
            }
            if reason:
                payload["reason"] = reason

            body = _json(
                self._client.request(
                    "PATCH",
                    self._instance_path(""),
                    json=payload,
                    timeout=30.0
                )
            )
            return ResizeResult(
                success=True,
                new_cost_per_hour=int(body.get("cost_per_hour_cents", 0))
            )
        except BackendError as e:
            return ResizeResult(success=False, error=str(e))
        except Exception as e:
            return ResizeResult(success=False, error=f"resize failed: {e}")

def _job_status(body: dict[str, Any]) -> JobStatus:
    return JobStatus(
        job_id=body.get("job_id", ""),
        status=body.get("status", "running"),
        name=body.get("name"),
        exit_code=body.get("exit_code"),
        end_reason=body.get("end_reason"),
        started_at=body.get("started_at"),
        ended_at=body.get("ended_at"),
        log_size=body.get("log_size"),
        killed_by=body.get("killed_by"),
    )


def _tar_gz_of(local_dir: Path) -> bytes:
    """Build a deterministic gzipped tar of a directory's contents."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in sorted(local_dir.rglob("*")):
            if "__pycache__" in item.parts or item.name.endswith(".pyc"):
                continue
            info = tar.gettarinfo(str(item), arcname=str(item.relative_to(local_dir).as_posix()))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if item.is_file():
                with item.open("rb") as fh:
                    tar.addfile(info, fh)
            else:
                tar.addfile(info)
    return buf.getvalue()


def _extract_tar_gz(data: bytes, local_dir: Path) -> list[Path]:
    """Extract an archive under `local_dir`, refusing members that escape it."""
    local_dir.mkdir(parents=True, exist_ok=True)
    root = local_dir.resolve()
    written: list[Path] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise BackendError(
                    f"archive member escapes the local directory: {member.name}",
                    code="unsafe_archive",
                )
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            # Temp-then-rename: os.replace is atomic, so whatever anyone reads at
            # `target` is a complete version of the file, never the middle of one.
            # The temp name carries the thread id because two syncs of the same
            # study can legitimately overlap (a background cycle that outlived a
            # bounded join, `openreynolds pull` from another terminal).
            partial = target.with_name(f"{target.name}.part-{threading.get_ident()}")
            try:
                with source, partial.open("wb") as out:
                    out.write(source.read())
                partial.replace(target)
            except OSError:
                partial.unlink(missing_ok=True)
                raise
            written.append(target)
    return written


def acquire(
    base_url: str,
    api_key: str,
    instance_id: str | None = None,
) -> tuple[HostedBackend, FoamdClient, str]:
    """Get a workspace: the named instance, else an existing one, else a new one.

    The service caps concurrent instances (default 1) and deleting one destroys its
    persistent volume, so reuse is the default and nothing here ever deletes.
    """
    client = FoamdClient(base_url, api_key)
    try:
        was_running = False
        if instance_id is None:
            existing = [
                inst for inst in client.list_instances() if inst.get("status") != "deleted"
            ]
            if existing:
                instance_id = existing[0]["id"]
                was_running = existing[0].get("status") == "running"
            else:
                instance_id = client.create_instance()
        client.start_instance(instance_id)
    except BaseException:
        client.close()
        raise
    backend = HostedBackend(client, instance_id)
    # Whether it was already up decides whether whoever asked for it should put it
    # back down again. A command that borrows a container ought to leave the machine
    # as it found it; a session is what containers are for.
    backend.was_already_running = was_running
    return backend, client, instance_id
