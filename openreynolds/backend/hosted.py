"""HTTP client for the hosted OpenFOAM service.

Maps `Backend` 1:1 onto the service's published `/v1` contract. This is the only module
in the package that knows the contract exists.
"""

from __future__ import annotations

import io
import tarfile
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
    Stat,
)

_RETRY_STATUSES = frozenset({429, 502, 503, 504})
_MAX_ATTEMPTS = 5
_DEFAULT_RETRY_AFTER_S = 10.0


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
    return min(2.0**attempt, 30.0)


class FoamdClient:
    """Low-level transport: auth, retries, error decoding, instance lifecycle."""

    def __init__(self, base_url: str, api_key: str, *, connect_timeout: float = 10.0):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0, connect=connect_timeout),
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

    def create_instance(self, cpu: float = 8.0, mem_gb: int = 16) -> str:
        body = _json(
            self.request("POST", "/v1/instances", json={"cpu": cpu, "mem_gb": mem_gb})
        )
        return body["instance_id"]

    def start_instance(self, instance_id: str) -> dict[str, Any]:
        return _json(self.request("POST", f"/v1/instances/{instance_id}/start", timeout=180.0))

    # -- capture plane ---------------------------------------------------------

    def create_study(self, title: str | None, instance_id: str | None) -> str:
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        if instance_id:
            payload["instance_id"] = instance_id
        return _json(self.request("POST", "/v1/studies", json=payload))["study_id"]

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

    def job_kill(self, job_id: str) -> JobStatus:
        return _job_status(
            _json(
                self._client.request(
                    "POST", f"/v1/jobs/{job_id}/kill", json={"signal": "TERM"}, timeout=180.0
                )
            )
        )


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
            with source, target.open("wb") as out:
                out.write(source.read())
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
        if instance_id is None:
            existing = [
                inst for inst in client.list_instances() if inst.get("status") != "deleted"
            ]
            instance_id = existing[0]["id"] if existing else client.create_instance()
        client.start_instance(instance_id)
    except BaseException:
        client.close()
        raise
    return HostedBackend(client, instance_id), client, instance_id
