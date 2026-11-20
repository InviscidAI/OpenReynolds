"""`openreynolds serve`: the same capabilities over HTTP, for this machine's studies.

This is a **local control surface**, not the hosted Reynolds API. The listener is on
localhost, it needs the token it prints at startup, and every study it starts runs in
this process with this machine's configuration, workspace and `studies/` directory --
exactly as `openreynolds mcp` and a terminal session do. Nothing here is a client of
anything: there is no remote service in this module, and a program on another machine
is not what it is for.

It is the third face over `mcp/core.py`'s `Host` (the MCP server is the first, the
Python SDK the second), so the shapes it answers with -- the state names, the condensed
transcript, the cursor, the refusals -- are `core`'s own and cannot drift from the
tools'. What this module adds is routing, a token and a long poll:

    POST   /uploads                          {"path": "wing.stl"}
    POST   /studies                          {"task": "...", "depth": "quick", ...}
    GET    /studies?limit=20
    GET    /studies/<study_id>/results
    GET    /studies/<study_id>/file?path=/work/<study>/results.json
    GET    /sessions/<session_id>?after_seq=0&wait=30
    POST   /sessions/<session_id>/messages   {"text": "..."}
    POST   /sessions/<session_id>/approvals  {"decision": "approve"}
    DELETE /sessions/<session_id>
    GET    /                                 what each route does

The server is stdlib `http.server`: this package has no web framework among its
dependencies and a local control surface is not a reason to add one. One thread per
request, because a long poll holds its thread for up to `core.MAX_WAIT_S` seconds.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import shutil
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .mcp import core
from .mcp.core import McpError

log = logging.getLogger("openreynolds.serve")

DEFAULT_PORT = 8787
DEFAULT_BIND = "127.0.0.1"
TOKEN_ENV = "OPENREYNOLDS_SERVE_TOKEN"
MAX_BODY_BYTES = 256 * 1024
"""A request body here is a task in words and a few options."""

ROUTES = {
    "POST /uploads": "upload_file",
    "POST /studies": "start_study",
    "GET /sessions/<session_id>": "study_status",
    "POST /sessions/<session_id>/messages": "send_message",
    "POST /sessions/<session_id>/approvals": "answer_approval",
    "DELETE /sessions/<session_id>": "end_study",
    "GET /studies": "list_studies",
    "GET /studies/<study_id>/results": "get_results",
    "GET /studies/<study_id>/file?path=...": "download_url",
}
"""Every route, and the capability it is. The description each one answers with is the
tool's own (`core.DESCRIPTIONS`), so the two faces describe themselves identically."""

NOT_THE_HOSTED_API = (
    "A local control surface for this machine's own studies: they run in this process, "
    "with this machine's configuration and studies/ directory. This is not the hosted "
    "Reynolds service's API."
)


def index(host: core.Host) -> dict[str, Any]:
    """What this server is and what each route does."""
    return {
        "server": "openreynolds serve",
        "about": NOT_THE_HOSTED_API,
        "auth": "Authorization: Bearer <token>, or X-Reynolds-Token: <token>.",
        "states": list(core.STATES),
        "routes": [{"route": route, "capability": tool, "what": core.description(host, tool)}
                   for route, tool in ROUTES.items()],
    }


class Handler(BaseHTTPRequestHandler):
    """One request. `self.server` carries the host, the token and the deliveries."""

    server_version = "openreynolds"
    sys_version = ""

    # -- the routes ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server's own naming
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle("DELETE")

    def _handle(self, method: str) -> None:
        if not self._authorized():
            # No hint about which part was wrong, and the body is read and dropped so a
            # caller with the wrong token is not left writing into a closed socket. A
            # body this cannot even read is still answered 401, not 500: an
            # unauthenticated request must not reach a traceback.
            try:
                self._read_body()
            except (McpError, OSError):
                pass
            self._json(HTTPStatus.UNAUTHORIZED,
                       {"error": "a token is required: Authorization: Bearer <token>"})
            return
        host = self.server.host
        parts = urlparse(self.path)
        query = parse_qs(parts.query)
        segments = [s for s in parts.path.split("/") if s]
        try:
            self._route(method, segments, query, host)
        except McpError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except BrokenPipeError:
            pass  # the caller hung up mid-answer, which a long poll invites
        except Exception as exc:  # noqa: BLE001 - a host's own refusal, in its words
            words = host.explain(exc)
            if words is not None:
                self._json(HTTPStatus.BAD_REQUEST, {"error": words})
                return
            log.exception("serve: %s %s failed", method, parts.path)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"{type(exc).__name__}: {exc}"})

    def _route(self, method: str, segments: list[str], query: dict[str, list[str]],
               host: core.Host) -> None:
        deliveries = self.server.deliveries
        where = (method, tuple(segments))

        if where == ("GET", ()):
            self._json(HTTPStatus.OK, index(host))

        elif where == ("POST", ("uploads",)):
            body = self._read_json()
            path = str(body.get("path") or body.get("filename") or "")
            if not path:
                raise McpError('say which file: {"path": "..."}')
            self._json(HTTPStatus.OK, host.stage_upload(path))

        elif where == ("POST", ("studies",)):
            body = self._read_json()
            known = ("task", "upload_ids", "study_id", "depth", "mode", "model", "effort",
                     "notes", "solver")
            unknown = sorted(set(body) - set(known))
            if unknown:
                raise McpError(f"{', '.join(unknown)}: not a field of this request "
                               f"({', '.join(known)})")
            task = str(body.pop("task", "") or "")
            if "upload_ids" in body and not isinstance(body["upload_ids"], list):
                raise McpError("upload_ids is a list of ids from POST /uploads")
            self._json(HTTPStatus.CREATED, core.begin_study(host, deliveries, task, **body))

        elif where == ("GET", ("studies",)):
            self._json(HTTPStatus.OK, core.studies_list(host, self._int(query, "limit", 20)))

        elif method == "GET" and len(segments) == 3 and segments[0] == "studies" \
                and segments[2] == "results":
            self._json(HTTPStatus.OK, core.study_results(host, segments[1]))

        elif method == "GET" and len(segments) == 3 and segments[0] == "studies" \
                and segments[2] == "file":
            # `host.download` is the refusal too: a path that is not inside the study's
            # workspace never reaches the disk (`mcp/local.py::_check_path`).
            path = (query.get("path") or [""])[0]
            if not path:
                raise McpError("say which file: ?path=/work/<study>/...")
            self._file(host.download(segments[1], path)["path"])

        elif method == "GET" and len(segments) == 2 and segments[0] == "sessions":
            session_id = segments[1]
            # The query before the session: a request this cannot even read is a
            # refusal, and nothing of the caller's reaches a session over a typo.
            after_seq = self._int(query, "after_seq", 0)
            wait = self._int(query, "wait", 0)
            host.check(session_id)
            host.heartbeat(session_id)
            row, picture, state = core.wait_for_change(host, deliveries, session_id, wait)
            self._json(HTTPStatus.OK,
                       core.status_answer(host, session_id, row, picture, state, after_seq))

        elif method == "POST" and len(segments) == 3 and segments[0] == "sessions" \
                and segments[2] == "messages":
            text = core.said_text(str(self._read_json().get("text") or ""))
            host.say(segments[1], text)
            self._json(HTTPStatus.OK,
                       {"sent": True, "next": "Read the session to see the answer."})

        elif method == "POST" and len(segments) == 3 and segments[0] == "sessions" \
                and segments[2] == "approvals":
            body = self._read_json()
            line = core.approval_line(str(body.get("decision") or ""), str(body.get("note") or ""))
            host.say(segments[1], line)
            self._json(HTTPStatus.OK, {"sent": line, "next": "Read the session."})

        elif method == "DELETE" and len(segments) == 2 and segments[0] == "sessions":
            self._json(HTTPStatus.OK, {"status": host.end(segments[1])})

        else:
            self._json(HTTPStatus.NOT_FOUND,
                       {"error": f"no {method} {'/' + '/'.join(segments)} here",
                        "routes": sorted(ROUTES)})

    # -- the mechanics -------------------------------------------------------------

    def _authorized(self) -> bool:
        given = (self.headers.get("Authorization") or "").strip()
        given = given[7:].strip() if given[:7].lower() == "bearer " else \
            (self.headers.get("X-Reynolds-Token") or "")
        return hmac.compare_digest(given.encode(), str(self.server.token).encode())

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise McpError("Content-Length is not a number") from None
        if length > MAX_BODY_BYTES:
            raise McpError(f"the body is {length} bytes; the limit is {MAX_BODY_BYTES}")
        return self.rfile.read(length) if length > 0 else b""

    def _read_json(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise McpError(f"the body is not JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise McpError("the body is one JSON object")
        return body

    @staticmethod
    def _int(query: dict[str, list[str]], name: str, default: int) -> int:
        raw = (query.get(name) or [""])[0]
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise McpError(f"{name}={raw!r} is not a number") from exc

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _file(self, path: str) -> None:
        local = os.fspath(path)
        if not os.path.isfile(local):
            # A directory exists, so the host hands it over happily; there is nothing
            # to send back for one.
            raise McpError(f"{local} is not a file")
        size = os.path.getsize(local)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{os.path.basename(local)}"')
        self.end_headers()
        with open(local, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Not onto stderr unconditionally: this process is also running studies, whose
        # own reporting is what somebody watching the terminal is there to read.
        log.info("%s %s", self.address_string(), fmt % args)


def build(host: core.Host, *, token: str, bind: str = DEFAULT_BIND,
          port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """The listener, not yet serving: `serve_forever()` on it, `shutdown()` to stop.

    A thread per request (`daemon_threads`, so a long poll in flight cannot hold up a
    shutdown), and the host, the token and the deliveries hung on the server for the
    handler to find."""
    server = ThreadingHTTPServer((bind, port), Handler)
    server.daemon_threads = True
    server.host = host  # type: ignore[attr-defined]
    server.token = token  # type: ignore[attr-defined]
    server.deliveries = core.deliveries_for(host)  # type: ignore[attr-defined]
    return server


def token_for(given: str | None = None) -> str:
    """The token callers must send: the one asked for, the one in the environment, or a
    fresh one. There is no unauthenticated mode -- this process can start compute that
    costs money, and an open listener that does that is a defect, not a convenience."""
    return given or os.environ.get(TOKEN_ENV) or secrets.token_urlsafe(24)


def serve(*, bind: str = DEFAULT_BIND, port: int = DEFAULT_PORT,
          token: str | None = None) -> None:
    """`openreynolds serve`: run until interrupted, then end every live study."""
    from . import cli
    from .mcp.local import LocalHost

    # The studies' own reporting goes to stderr; stdout is for the address and the
    # token, which is what a caller has to read out of this command.
    cli.quiet_stdout()
    token = token_for(token)
    host = LocalHost()
    server = build(host, token=token, bind=bind, port=port)
    shown = f"[{bind}]" if ":" in bind else bind
    print(f"openreynolds serve on http://{shown}:{server.server_port}", flush=True)
    print(f"token: {token}", flush=True)
    print(NOT_THE_HOSTED_API, file=sys.stderr, flush=True)
    if bind not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {bind}, not just this machine -- anyone who can "
              "reach it and has the token can start studies you pay for",
              file=sys.stderr, flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        # Every live study ends as end_study would: jobs stopped, workspace stopped,
        # files kept. A study left behind holds a workspace that is being paid for.
        host.close()
