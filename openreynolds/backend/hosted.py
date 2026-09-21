"""HTTP client for the hosted OpenFOAM service.

Maps `Backend` 1:1 onto the service's published `/v1` contract. This is the only module
in the package that knows the contract exists.
"""

from __future__ import annotations

import io
import tarfile
import threading
import time
import zlib
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
    StoredEntry,
    StoredListing,
)
from .pending import PendingBackend

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

_DECLINED_STATUSES = frozenset({429, 503})
"""The subset of `_RETRY_STATUSES` that says the service turned the call away.

A 429 is the rate limiter and a 503 is a workspace still booting: in both the handler
never ran, so asking again cannot make the same thing happen twice. 500, 502 and 504
say nothing of the kind -- the work may have been done and only the answer lost -- and
neither does a read timeout."""

_NO_RETRY_CODES = frozenset({"modal_auth_failed"})
"""Coded errors that are never worth trying again, whatever their status says.

`_RETRY_STATUSES` is keyed on the status code, which is the bluntest thing a response
carries: it has to answer for every route at once. A coded error is the finer signal,
and this is the case that needs it.

`modal_auth_failed` is the service's own Modal credentials being rejected. It is a 500
because it genuinely is an internal fault of the service -- nothing else would be honest,
since the caller's request was fine and a 4xx would blame them for it -- and 500 is in
the retry set for good reasons that have nothing to do with this. So without this set, a
permanent configuration fault costs four backoffs and then reports itself as a transient
failure, which is exactly the cost foamd's carve-out was written to avoid, reached by the
other door.

Anything added here has to be an error where *no* later attempt can succeed until a
person acts. A workspace still booting is not one; rejected credentials are.
"""

_REPEATABLE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
"""Methods a client may send again when it does not know whether the first one landed.

A POST is not one of them, and the transcript is where that showed. A study's messages
are posted one at a time to `POST /v1/studies/{id}/messages` carrying a `seq` this
client assigns; the service inserts what it is given and `messages(study_id, seq)` is an
ordinary index, not a unique key. So a POST that was carried out and whose answer was
lost -- a read timeout, a 502 from the edge -- wrote the row, and every repeat wrote it
again. One `job_check` reply from a 3D transient run appeared in the captured transcript
three times, and five other messages of the same study twice; read back afterwards it
looked exactly like an agent being served a stale answer for twenty-six minutes, and
that is not what happened. Nothing here is retried in the dark now: an ambiguous failure
on a write is handed to the caller instead of being tried again."""

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
"""Methods that only read, which is a stronger claim than `_REPEATABLE_METHODS`.

A repeatable call may be sent again without doubling its effect; a safe one has no
effect to double. The distinction only matters for `sandbox_gone` below, where the
question is not "did this land?" but "is there anything here a second attempt could
make happen twice?" -- and the only honest answer for a write is that nobody knows."""

_SANDBOX_GONE_ATTEMPTS = 2
"""How many times a read may meet `409 sandbox_gone` before the model hears about it.

The hosted Sandbox cycles under load: 9 of 68 tool calls in one live study and 56 of
244 in another came back `sandbox_gone`, in bursts, three of the four beginning within
90 seconds of an 8-rank solve being launched (F-58). Nothing is lost when it happens --
`/work` is a persistent Volume and a job restarts from the latest time -- but every one
of those reached the model as a failed tool call, and one study spent about 16 minutes
improvising recovery and relaunched the same solve stage five times.

The service now answers the 409 with `Retry-After: 5`, the same number its cold-start
503 carries, because the replacement Sandbox is up in about that long. Two attempts,
not five: this is a read waiting for a container to come back, and if it has not by the
second ask the model is better told plainly than left sitting.

**409 is deliberately not in `_RETRY_STATUSES`, and must not be put there.** That list
is keyed on the status alone and so answers for every route at once, including
`job_start`: a retried POST once produced five duplicate running jobs, and `job_start`
would need an idempotency key before it could be sent twice safely. The carve-out here
is narrower on both axes -- one error code, and only for a method that reads."""

_SANDBOX_GONE_MAX_WAIT_S = 15.0
"""The longest a `sandbox_gone` retry may add to a call.

`Retry-After` is honoured verbatim by `_retry_delay`, which is right for a header the
service chose; this bounds what a header nobody is watching can cost a tool call that
is already slow. At the 5 seconds foamd sends, the cap never binds."""

STORED_LISTING_LIMIT = 5000
"""The most entries `GET /v1/instances/{id}/files?list=1` (and its study-scoped twin
`GET /v1/studies/{id}/workspace`) will list in one answer.

The routes' own ceiling (`OpenFoam_Instance/app/workspace.py`, `MAX_ENTRIES`; a
cap above it is a 422, not a bigger answer), and asked for in full because the
answer is one whole tree rather than a walk to a depth: `browse.MAX_ENTRIES` is cut
on this side, after the depth is applied, and asking for less here would cut the tree
before the depth had been looked at."""

LISTING_TIMEOUT_S = 60.0
"""How long one `list_files` request may run. The `find` it replaced was sent with
`timeout_s=60`; a listing that takes longer than a minute is the service stuck, not a
big workspace, and the mirror's cycle behind it should find that out rather than sit."""

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
    if 300 <= response.status_code < 400:
        raise _redirect_error(response)
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


def _redirect_error(response: httpx.Response) -> BackendError:
    """A 3xx that reached a decoder is a redirect nothing could do anything with.

    The service issues no redirects of its own; the edge in front of it does, and
    answers a request that ran past its window with a bare `303` and an empty body.
    Followed, that redirect returns the real result. Unfollowed, the empty body used
    to surface as `bad_response (303): the body was empty` -- true, and useless: it
    named the symptom and hid the one thing a reader can act on. So it is named.

    `_send_following_safe_redirects` follows the ones it can, so most of what reaches
    here from `FoamdClient.request` is a 3xx with no `Location` at all -- nowhere to go,
    and a body that is not the answer either. Not all of it: `has_redirect_location`
    (`httpx2/_models.py:770`) counts only 301/302/303/307/308, so a 300, 304 or 305
    arrives here with its `Location` intact and unfollowed. Both shapes are covered
    below -- `where` names the destination when there is one, and `why` explains the
    absence when there is not. The `Location` limb is also for the
    other way in: `_json` is handed responses directly (by tests, and by anything
    that gets a response without going through the follower), and a named redirect
    beats a JSON traceback there too.
    """
    location = response.headers.get("location", "")
    where = f" to {location}" if location else ""
    why = "" if location else "; it carried no Location header, so there was nowhere to follow it to"
    return BackendError(
        f"the service answered {response.status_code} with a redirect{where} that this "
        f"client did not follow{why}",
        code="redirect_not_followed",
        status=response.status_code,
    )


# -- redirects, followed only where following them cannot lose the request -------------
#
# F-45 was a redirect this client did not follow. The fix was a blanket
# `follow_redirects=True`, and that is broader than the fact it was built on: httpx
# follows a 303 (and a 302, and a 301 on a POST) by re-issuing the request as a **GET
# with no body** -- `_redirect_method` and `_redirect_stream` in httpx2 2.12.0, whose own
# comment for the 302 case reads "Do what the browsers do, despite standards". So each
# hop is decided here instead, on what the status actually means:
#
#   * the method survives the hop (307/308, or any redirect on a request with no body
#     to lose) -- followed, because the request that arrives is the request that was
#     made;
#   * a **303** on a request that did carry a body -- followed. Not a concession: 303 is
#     defined (RFC 9110 15.4.4) as "the answer to your request is at this other URI,
#     fetch it with GET". The origin has already received and acted on the request; the
#     hop collects a result, and dropping the body on it is what the status asks for
#     rather than something being lost. It is also the one redirect this stack is known
#     to produce -- Modal's edge answers a request past its 150 s window this way -- and
#     F-45's live measurement agrees with the reading: following it returned `sleep
#     200`'s own rc=0 and its output at 203.9 s, which only the origin that ran the
#     command could have supplied. The two alternatives were considered and are worse.
#     Refusing it outright is F-45 restored: the command ran and the caller is told it
#     failed. Re-sending the original POST instead of fetching the named URI is the
#     double-execution `_REPEATABLE_METHODS` exists to prevent -- the measurement above
#     says the first POST *ran the command*, so a second one runs it again, and 203.9 s
#     of `sleep 200` becomes 400 s of it. That stays true for a caller that passed
#     `repeatable=True`: `put_tree`'s archive is safe to unpack twice, but the hop it
#     would be replacing is the leg carrying the answer, so replaying the POST does not
#     collect the answer, it asks the question again;
#   * a **301/302** on a request that carried a body -- refused, and named. Those say
#     "the resource moved", not "the answer is over there", so httpx's downgrade to a
#     bodyless GET turns a write into a read with nothing anywhere saying so. The
#     service issues none of these, so this only ever fires on infrastructure nobody
#     here has seen, which is exactly when a loud error beats a plausible one.
#
# What this narrowing is NOT: an explanation of F-47. F-47 is one `bad_request (400):
# not a valid tar.gz archive` for an archive that opened cleanly on the machine that
# sent it. The redirect this stack is known to produce cannot be where those bytes went:
# a 303 is what an origin answers *after* it has received and acted on the request, so
# the POST carrying the archive goes out whole and the hop that follows collects a
# result rather than re-sending anything. There is no body to lose on that path.
#
# The relocation variant (301/302) is a different matter, and nothing measured rules it
# out. An earlier draft of this comment said the route table did: `OpenFoam_Instance/
# app/files.py` mounts `/v1/instances/{id}/tar` as a POST and nothing else, so "the
# bodyless GET answers 405, not the 400 that was seen". That does not follow. httpx
# re-issues a redirect at its **`Location`**, not at the path that was asked, so the 405
# arrives only if the `Location` names the tar route back again; aimed anywhere else the
# hop gets whatever lives there, 400 included. No `Location` has ever been captured for
# a 3xx on this path -- the edge's 303 is modelled in `tests/test_hosted.py`
# (`_edge_with_a_150s_redirect`) as pointing somewhere else entirely -- so the route
# table settles nothing here on its own. The claim that survives is the narrow one: the
# redirect anyone has actually observed does not drop a body, and the one that could is
# refused below instead of followed.
#
# The half of this change that answers the observation is in `put_tree`, which checks
# the claim the service is making about bytes it is still holding; this half is here so
# that a 301 or 302 can never quietly become a read, which is a different (and so far
# hypothetical) way to be told a true thing about the wrong request.
#
# Read the scope honestly: for the 303 -- the only redirect anything here has ever seen
# -- this is byte-for-byte what `follow_redirects=True` did. What changed is 301 and 302,
# which nobody has observed. So this half narrows a hazard; it does not remove the one
# F-47 recorded, and no comment below should be read as saying it does.

_MAX_REDIRECTS = 5
"""Hops to follow before giving up. The one redirect anything here is known to produce
comes from the edge in front of the service rather than the service itself -- foamd
issues none -- and it resolves in a single hop; the cap is here so a loop cannot become
an infinite one.

Exactly this many hops are *allowed*, so a chain of five that ends in an answer returns
the answer. Getting that boundary wrong is not a rounding error: an exec whose result
arrived after the last permitted hop would be reported as a hard failure for a command
that ran and succeeded, which is F-45's whole cost reintroduced at the bound."""


def _carries_a_body(request: httpx.Request) -> bool:
    """Whether this request has a body a redirect could quietly drop.

    Read off the request that was actually sent rather than the arguments that built
    it: `content=`, `json=`, `data=` and `files=` all end up as one stream with a
    length on it, and it is the length that decides whether anything is at stake.

    An unreadable `Content-Length` answers `True`. httpx always writes a parseable one,
    so this is unreachable in practice, but the default has to point at "assume there
    is something to lose": the failure this function guards is a body dropped without
    anyone noticing, and "I could not tell" must not resolve to "nothing was at stake".
    """
    if request.headers.get("transfer-encoding"):
        return True
    try:
        return int(request.headers.get("content-length", "0")) > 0
    except ValueError:
        return True


def _redirect_would_lose_the_body(response: httpx.Response) -> BackendError:
    """A 301/302 on a body-carrying request: refused rather than followed.

    Following it means httpx re-sending the request as a GET with the stream dropped
    (`_redirect_method`, `_redirect_stream`), so whatever comes back is an answer about
    a request the caller never made -- true, specific, and about the wrong thing.

    Not known to be F-47's mechanism, and not excluded as it either: where the bodyless
    GET lands is decided by the redirect's `Location`, and no `Location` has ever been
    captured for a 3xx on the tar path. (The service mounts that path POST-only, so a
    GET aimed *back at it* answers 405 rather than the 400 that was seen -- but that is
    one of the places the hop could go, not all of them.) This is the loud version of a
    failure mode nobody has observed, kept loud because the quiet version is a write
    that silently became a read.
    """
    location = response.headers.get("location", "")
    where = f" to {location}" if location else ""
    return BackendError(
        f"the service answered {response.status_code} with a redirect{where}, which "
        f"would be followed by re-sending this request as a GET with its body dropped; "
        f"refused, because the answer to that is an answer about a different request",
        code="redirect_would_lose_the_body",
        status=response.status_code,
    )


def _send_following_safe_redirects(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    """Send one request, following only the redirects that cannot corrupt it.

    `follow_redirects=False` is passed on the first send rather than relied on from the
    client's constructor: it is the whole safety property of this function, and a
    client handed in with redirects enabled would make it a no-op that says nothing --
    httpx would follow the 302 itself and `next_request` would be `None` on every hop.
    Tests replace `FoamdClient._client` outright, so the constructor is not a place
    this can be guaranteed from. It is positional-by-name rather than merged into
    `kwargs`, so a caller that passes its own `follow_redirects` gets a `TypeError`
    instead of an override -- deliberate: overriding it silently disables the only
    thing this function does.

    With it off, httpx hands back the redirect request it *would* have sent as
    `response.next_request`, so the decision of which hops to take is made here while
    httpx keeps the URL joining, the cross-origin `Authorization` stripping and the
    header fixups that go with it.
    """
    response = client.request(method, url, follow_redirects=False, **kwargs)
    for _ in range(_MAX_REDIRECTS):
        hop = response.next_request
        if hop is None:
            return response
        sent = response.request
        if hop.method != sent.method and _carries_a_body(sent):
            if response.status_code != 303:
                raise _redirect_would_lose_the_body(response)
        response = client.send(hop, follow_redirects=False)
    # Checked here and not at the top of the loop: the response to the last permitted
    # hop is an answer, and an answer is never thrown away for being late in a chain.
    # Reaching this line means a sixth hop was being asked for, which is what the
    # message says.
    if response.next_request is not None:
        raise BackendError(
            f"the service redirected more than {_MAX_REDIRECTS} times",
            code="too_many_redirects",
            status=response.status_code,
        )
    return response


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
#
# These four helpers pre-date `FoamdClient` and each build their own client. They ask
# short questions, but the edge in front of the service does not know that, so they
# follow redirects for the same reason `FoamdClient` does -- and because a client that
# does not is the mistake this module has already made once. They follow them the same
# way, too: through `_send_following_safe_redirects`, so that a `301 Moved` on a sign-in
# POST is named rather than silently re-sent as a GET with the credentials dropped and
# answered as if it had been asked properly. A 303 is still followed here, as everywhere
# in this module, because 303 means the answer is elsewhere and not that the POST needs
# re-sending.


def _post_json(base_url: str, path: str, body: dict[str, Any], *, headers: dict[str, str] | None = None,
               transport: Any = None) -> httpx.Response:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport,
                      follow_redirects=False) as client:
        try:
            return _send_following_safe_redirects(
                client, "POST", path, json=body, headers=headers or {})
        except httpx.HTTPError as exc:
            raise BackendError(f"cannot reach {base_url}: {exc}", code="unreachable") from exc


def auth_config(base_url: str, *, transport: Any = None) -> dict[str, Any]:
    """Where the service's identity provider is. Public by design: it is what the
    service's own sign-in page fetches, so a terminal can sign in the same way."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport,
                      follow_redirects=False) as client:
        try:
            response = _send_following_safe_redirects(
                client, "GET", "/dashboard/config.json")
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
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport,
                      follow_redirects=False) as client:
        try:
            response = _send_following_safe_redirects(
                client, "POST", "/v1/device/code",
                json={"name": name} if name else {})
        except httpx.HTTPError as exc:
            raise BackendError(f"cannot reach the service: {exc}", code="unreachable") from exc
    if response.status_code >= 400:
        raise _decode_error(response)
    return _json(response)


def device_token(base_url: str, code: str, *, transport: Any = None) -> dict[str, Any] | None:
    """Collect the key once the code has been approved; `None` while it has not.

    The service hands the plaintext over exactly once, so a caller that gets a dict
    must save it then and there."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, transport=transport,
                      follow_redirects=False) as client:
        try:
            response = _send_following_safe_redirects(
                client, "POST", "/v1/device/token", json={"device_code": code})
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
            # Redirects are still followed -- by `_send_following_safe_redirects`,
            # not by httpx. The service issues none; the edge in front of it does,
            # answering a long `POST .../exec` with a bare `303` and an empty body,
            # which this client once turned into `bad_response (303): the body was
            # empty` -- the command having usually *run* by then, so a render that had
            # already been written looked like a render that had not, four times over
            # (F-45: "I changed the rendering script four times and pulled a
            # byte-identical PNG each time"). Off here is not "do not follow": it is
            # what makes httpx hand each
            # redirect back as `next_request` so the follower can tell a 303 (fetch the
            # answer elsewhere) from a 302 (httpx would re-send this POST as a bodyless
            # GET). The follower passes it explicitly too, so this line is a default
            # and not the guarantee.
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        repeatable: bool | None = None,
        max_attempts: int | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Issue one request, retrying cold starts and transient failures.

        `repeatable` says whether sending this call a second time is the same as
        sending it once. It defaults to the method (`_REPEATABLE_METHODS`), and a
        caller overrides it where the method is a poor guide -- a `POST .../tar` that
        unpacks the same bytes over the same directory is repeatable, a `POST` that
        appends a row to a transcript is not. When it is false, only a failure that
        proves the service did nothing (`_DECLINED_STATUSES`, or never connecting at
        all) is tried again; an ambiguous one is raised, because the alternative is
        doing the work twice and never finding out.

        A `409 sandbox_gone` is outside both of those rules and has its own, narrower
        one: it is tried again only for a method that reads, only once, and only for
        as long as the service's own `Retry-After` asks for. `_SANDBOX_GONE_ATTEMPTS`
        says why a write is left alone.

        `max_attempts` overrides `_MAX_ATTEMPTS` for this call. Only a caller with a
        reason to bound its own worst case passes it -- the mirror's background
        cycles, which share the exec channel with whatever tool call is running and
        so must not sit through five retries of a request that is timing out because
        the service is stuck on it, not because it blinked (see `mirror.py`'s
        `LIVE_PULL_TIMEOUT_S`). Left `None`, behaviour is exactly what it was.
        """
        repeat_ok = (
            method.upper() in _REPEATABLE_METHODS if repeatable is None else repeatable
        )
        attempts = int(max_attempts) if max_attempts else _MAX_ATTEMPTS
        last_error: BackendError | None = None
        # Counted separately from `attempt`, because a vanished Sandbox is its own
        # kind of failure with its own much shorter patience.
        gone_attempts = 0
        for attempt in range(attempts):
            response = None
            # Whether this failure leaves it unknown whether the service acted.
            ambiguous = False
            try:
                response = _send_following_safe_redirects(
                    self._client, method, path, timeout=timeout, **kwargs
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # Nothing was ever handed over, so repeating this repeats no effect --
                # true whatever the method is.
                last_error = BackendError(f"cannot reach the service: {exc}", code="unreachable")
            except httpx.TimeoutException as exc:
                last_error = BackendError(f"request timed out: {exc}", code="timeout")
                ambiguous = True
            except httpx.HTTPError as exc:
                last_error = BackendError(f"cannot reach the service: {exc}", code="unreachable")
                ambiguous = True
            else:
                if 300 <= response.status_code < 400:
                    # A redirect the follower did not take: no `Location` at all, or a
                    # 3xx httpx2 does not count as one (300/304/305), which can carry a
                    # `Location` -- `_redirect_error` names it either way. Kept because
                    # of what happens when
                    # this is missing -- a bare 303 is under 400, so it used to be
                    # handed back as a success and died three frames later in the JSON
                    # decoder. Not retried: a redirect with nowhere to go is not a
                    # thing a second attempt improves.
                    raise _redirect_error(response)
                if response.status_code < 400:
                    return response
                last_error = _decode_error(response)
                # Before the status is consulted: a code in this set outranks it, because
                # the status cannot distinguish "the service blinked" from "the service
                # cannot reach Modal at all until somebody renews a token".
                if last_error.code in _NO_RETRY_CODES:
                    raise last_error
                # The other case the status cannot answer for, from the other side:
                # `409 sandbox_gone` is the Sandbox having cycled underneath a call
                # that was fine. For a read it means only that the answer has to be
                # asked for again once the replacement is up, and the service says
                # when that will be. For anything else it is unanswerable -- the
                # container is gone, so nothing can be asked about what the request
                # did before it went -- which is why the gate is `_SAFE_METHODS` and
                # not `repeat_ok`, and why 409 stays out of `_RETRY_STATUSES`
                # entirely. See `_SANDBOX_GONE_ATTEMPTS` for the duplicate jobs that
                # settled it.
                if response.status_code == 409 and last_error.code == "sandbox_gone":
                    may_ask_again = (
                        repeat_ok
                        and method.upper() in _SAFE_METHODS
                        and gone_attempts < _SANDBOX_GONE_ATTEMPTS - 1
                        and attempt < attempts - 1
                    )
                    if not may_ask_again:
                        raise last_error
                    gone_attempts += 1
                    time.sleep(min(_retry_delay(response, attempt),
                                   _SANDBOX_GONE_MAX_WAIT_S))
                    continue
                # A 4xx collected through a 303 hop gets no special treatment here, and
                # an earlier draft of the F-47 fix that gave it some was wrong: it made
                # every 4xx behind a hop "ambiguous", which retried a 401/403/413 like a
                # transient and, worse, pre-empted the `_DECLINED_STATUSES` refinement
                # below so that a 429 arriving through the edge's redirect raised on the
                # first answer instead of backing off -- a regression aimed squarely at
                # the long-exec path F-45 exists for. It rested on reading the 303 as
                # "the service may not have got your body", which contradicts the
                # reading that licenses following it at all (see the redirect note
                # above): a 303 says the origin has the request and the answer is
                # elsewhere. So the answer collected through it is the service's real
                # answer about the real request, and it is treated like any other.
                if response.status_code not in _RETRY_STATUSES:
                    raise last_error
                ambiguous = response.status_code not in _DECLINED_STATUSES

            if ambiguous and not repeat_ok:
                raise last_error

            if attempt < attempts - 1:
                time.sleep(_retry_delay(response, attempt))

        raise last_error or BackendError("request failed", code="unreachable")

    # -- instances -------------------------------------------------------------

    def list_instances(self) -> list[dict[str, Any]]:
        return _json(self.request("GET", "/v1/instances"))

    def create_instance(self, cpu: float | None = None, mem_gb: int | None = None) -> str:
        """Create an instance, letting the service pick any shape not asked for.

        The size a new instance gets is the service's to choose: it is the thing that
        knows the fleet's defaults and per-user ceilings, and it validates what it is
        sent. Carrying a copy of those numbers here is how the copy and the original
        drift apart -- which they did, silently, until an instance sized by one of them
        was described to the model by the other.
        """
        shape = {k: v for k, v in (("cpu", cpu), ("mem_gb", mem_gb)) if v is not None}
        body = _json(self.request("POST", "/v1/instances", json=shape))
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

    def list_files(self, instance_id: str, path: str) -> dict[str, Any]:
        """The tree under `path`, scoped to the instance: `GET /v1/instances/{id}/
        files?list=1`. Answered from the live machine when the workspace is up and
        from the service's own copy of the workspace when it is stopped; no instance
        is started, whatever state the workspace is in, and the workspace's
        last-activity clock is left alone (the route resolves the instance row without
        touching it -- foamd `files.read_or_stat`, #53).

        The instance's twin of `list_workspace`, and the one `list_stored` asks,
        because the study route refuses the one path a study whose home is the
        workspace root itself asks for -- `/work` -- and three of the owner's studies
        have that home. Measured 2026-09-21 09:58 SGT: `workspace?path=/work -> 400`,
        and the fallback exec adopted a machine from the pool to answer one `find` on
        the way out of a session. An instance's own `/work` is the owner's, whole.

        `path` is workspace-absolute (`/work`, `/work/<study>/case`) or relative to
        `/work`; the answer is `{root, entries, truncated}` in the study route's shape,
        `root` the resolved absolute path, entries workspace-absolute, cut breadth-first
        at `STORED_LISTING_LIMIT`. A path outside `/work` is a 400, a path the copy does
        not have is a 404; both arrive as the `BackendError` `request` makes of them.

        Bounded, unlike the other listings on this client: this one is asked every
        twenty seconds for the life of a session by the mirror's background cycles,
        and `request` given no timeout passes `None` through to httpx, which is no
        timeout at all rather than the client's default. The `find` this replaces ran
        under `timeout_s=60`; the same minute is the bound here. Retries are the
        client's ordinary ones for a GET.
        """
        return _json(
            self.request(
                "GET",
                f"/v1/instances/{instance_id}/files",
                params={"path": path, "list": 1, "recursive": 1, "entries": STORED_LISTING_LIMIT},
                timeout=LISTING_TIMEOUT_S,
            )
        )

    def list_workspace(self, study_id: str, path: str = "") -> dict[str, Any]:
        """The study's files as the service's own copy of the workspace holds them.
        No instance is started, whatever state the workspace is in.

        `path` is "" for the study's root, a path relative to it (`case/system`), or
        a workspace-absolute one (`/work/<study>/case/system`) -- the spelling every
        other surface uses -- and the answer is the same either way: `root` is the
        study's home, `entries` are workspace-absolute (`path`, `is_dir`, `size`,
        `mtime`) and `truncated` says whether the copy held more than
        `STORED_LISTING_LIMIT`. Recursive and not depth-limited: the whole tree under
        `path`, or the first `limit` entries of it.

        Two refusals are facts about the ask rather than the service: a path outside
        the study's own directory is a 400 (the copy holds every study the instance
        ever ran, and this route shows one), and a study whose workspace was never
        written to the copy -- or whose row predates the id being recorded -- is a
        404. Both arrive as the `BackendError` `request` makes of them, unretried.
        """
        return _json(
            self.request(
                "GET",
                f"/v1/studies/{study_id}/workspace",
                params={"path": path, "recursive": 1, "limit": STORED_LISTING_LIMIT},
            )
        )

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
        self.instances_held = 0
        """How many non-deleted instances the account held when this one was chosen.

        The service used to cap an account at one instance, so `acquire` picking a
        workspace was never a choice and nothing needed to report it. The cap is no
        longer 1, so it is: `acquire` collapses a whole account to a single id, and
        the workspaces it did not pick have their own Volumes with their own files
        on them. A caller that can say "there are five and I took this one" turns an
        empty-looking workspace into a sentence the user can act on.

        0 means nobody counted rather than "the account holds none": `acquire` given
        an explicit instance id never lists, because there was no choice to make."""
        self.study_id: str | None = None
        """Which study this session is serving, as the platform names it. Set by the
        session once it has opened or resumed the study's row (`cli.session`). Carried
        for the study-scoped routes; `list_stored` asks the instance's own route and
        needs no study id (see `FoamdClient.list_files` for why)."""

    def shutdown(self) -> None:
        """Put the container down. The volume is untouched, so nothing is lost."""
        self._client.stop_instance(self.instance_id)

    def list_stored(self, path: str, depth: int) -> StoredListing | None:
        """The files under `path` as the service lists them, cut to `depth`. None
        when the ask did not work.

        `GET /v1/instances/{id}/files?list=1` answers from the live machine while the
        workspace is up (the daemon's own directory listing, no command run and no
        cap on the answer's bytes) and from the copy the service writes at every
        checkpoint and stop once it is down -- the same copy `get_file` and
        `get_tree` are served from then, so a sync that lists this way stays off the
        machine end to end. It never starts a machine and never counts as use of one.
        "Stored" is the name from when only the stopped case came this way; since the
        listing of a running workspace over the exec channel was found cut at 64 KB
        mid-line (`browse.LIST_PIPELINE` has the incident), `Browser.tree` asks this
        first in both states. Asked of the instance, not the study: the study route
        refuses the workspace root, and a study whose home is the root (three of the
        owner's are) lists exactly that path on its way out -- which is how the first
        cut of this (asking `/v1/studies/{id}/workspace`) still started a machine on
        2026-09-21. Nothing here needs a study id. The route answers the whole tree,
        so the caller's depth is applied here, counted from `path` as `find
        -maxdepth` counts it.

        Every failure is None and none of them is raised: the caller has the exec
        walk to fall back on, and the one thing this must not do is turn a listing
        into an exception. A 400 is a path outside `/work`; a 404 is a path the
        service does not have, a workspace never written to the copy, or a service
        without the route; the rest is the network. In each case the fallback is
        what always happened -- a poll, then a foreground `exec`, which may start
        the workspace -- so nothing is lost by being quiet here, only the saving.
        """
        try:
            body = self._client.list_files(self.instance_id, path)
            root = str(body.get("root") or "").rstrip("/")
            # The route names the resolved path as `root` and its entries in full,
            # so a relative ask is measured from where the route put it.
            anchor = path if path.startswith("/") else (root or f"{WORKSPACE_ROOT}/{path.strip('/')}")
            base = anchor.rstrip("/") + "/"
            entries = []
            for item in body.get("entries") or []:
                where = str(item.get("path") or "")
                if not where.startswith(base):
                    continue
                if where[len(base):].count("/") >= int(depth):
                    continue  # deeper than asked: `a/b` under `base` is depth 2
                entries.append(
                    StoredEntry(
                        path=where,
                        is_dir=bool(item.get("is_dir")),
                        size=int(item.get("size") or 0),
                        # Whole seconds on the wire (`workspace.listing_payload` casts
                        # to int), where `find`'s `%T@` carried a fraction. Kept as
                        # the float the walk's rows became, unrounded: the one reader
                        # that compares it (`mirror._already_here`) sets this machine's
                        # clock against the workspace's, and no fraction of a second
                        # survives that.
                        mtime=float(item.get("mtime") or 0),
                    )
                )
            return StoredListing(entries, truncated=bool(body.get("truncated")))
        except Exception:  # noqa: BLE001 - see docstring: None, never an exception
            return None

    def active_jobs(self) -> list[dict[str, Any]]:
        """What is still running on this instance, whoever started it (F-46).

        `?active=1` is one read of the jobs rows and starts no container -- the
        service answers it without reconciling, deliberately, so asking what is on
        an instance is never the thing that boots one.

        A failure answers "nothing", not an exception: this is asked on the way out
        of a session, and a listing that could not be fetched must not be what stops
        a workspace being put down. The cost of the wrong answer is asymmetric only
        in the other direction -- a shutdown that runs anyway kills the job -- so the
        one line of warning `_close_down` prints on a failure is the honest place
        for it, not a refusal to exit."""
        try:
            return _json(self._client.request(
                "GET", self._instance_path("/jobs"), params={"active": 1}))
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad body is not worth an exit failing
            raise BackendError(f"could not list this workspace's jobs: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def _instance_path(self, suffix: str) -> str:
        return f"/v1/instances/{self.instance_id}{suffix}"

    # -- commands --------------------------------------------------------------

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120,
             *, background: bool = False) -> ExecResult:
        timeout_s = max(1, min(int(timeout_s), EXEC_MAX_TIMEOUT_S))
        payload: dict[str, Any] = {"cmd": cmd, "timeout_s": timeout_s}
        if cwd:
            payload["cwd"] = cwd
        if background:
            # Only when true, so an older service -- which would ignore the field
            # anyway -- sees exactly the request it saw before.
            payload["background"] = True
        # Left non-repeatable, and the note further down says why in the concrete: an
        # ambiguous failure here used to be retried and re-ran a 13-minute command. The
        # command is arbitrary -- it may already have moved files or written a case --
        # so a repeat is a second run, not a second look. The cold-start 503 a first
        # call meets is still retried, because that one is the service saying it did
        # nothing at all.
        body = _json(
            self._client.request(
                "POST", self._instance_path("/exec"), json=payload, timeout=timeout_s + 60.0
            )
        )
        if body.get("idle"):
            # The service had no Sandbox up and, this being a poll, did not build one.
            # Nothing ran. Deliberately NOT exit_code 0 with empty output: that reads
            # as "the workspace is empty", which is the one wrong answer a file mirror
            # acts on rather than ignores.
            return ExecResult(exit_code=-1, output="", truncated=False, log_path=None,
                              stderr="", idle=True)
        if body.get("promoted") and body.get("job_id"):
            # The command was still running at the service's synchronous window and
            # the service started it again from scratch as a detached job rather than
            # hold a fragile long exec connection (which used to 500 and get retried,
            # re-running a 13-minute command). The answer is `{exit_code: null,
            # promoted: true, job_id, output, stderr, note}` (`OpenFoam_Instance/app/
            # execs.py`), and it is carried here as what it says: no exit code, the
            # flag, the job. Until 2026-09-21 this line read `ExecResult(exit_code=0,
            # output=note, job_id=...)`, and `tools._bash` printed `exit_code: 0` as
            # its first line for `sleep 240`, `sleep 200` and `sleep 180`, each of
            # them still running (study 20260921-033019-e1b4; jobs 808edf10, 5432459d,
            # e839ac57). The model read three commands that had finished with no
            # output and never polled a job; the rows stayed `running` for 46, 35 and
            # 20 minutes until the reaper closed them. The service's `note` is prose
            # about this shape for a human reading the raw answer; the tool says the
            # same from the fields, in the words its caller reads, so it is not kept.
            # `output` is what the synchronous run had produced when it was moved --
            # empty from today's service, carried in case that changes -- and `stderr`
            # is the wrapper's own, the one field that tells a command that was still
            # running from an exec that never got one started.
            return ExecResult(
                exit_code=None,
                output=body.get("output", "") or "",
                truncated=bool(body.get("truncated")),
                log_path=body.get("log_path") or None,
                stderr=body.get("stderr", "") or "",
                job_id=str(body["job_id"]),
                promoted=True,
            )
        return ExecResult(
            exit_code=body.get("exit_code", -1),
            output=body.get("output", ""),
            truncated=bool(body.get("truncated")),
            log_path=body.get("log_path"),
            # The wrapper's own stderr, which the service already computes and returns
            # as "why the command did not run" -- dropped here until now, so a failure
            # of the workspace itself reached the model as a bare exit code.
            stderr=body.get("stderr", "") or "",
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

    def get_file(
        self, path: str, offset: int = 0, limit: int | None = None,
        *, timeout: float = 300.0, max_attempts: int | None = None,
    ) -> bytes:
        """`timeout`/`max_attempts` bound one request the way `get_tree`'s do (see
        there) -- unused by anything in this package today (`get_file` backs the
        `fetch`/`read_file` tools, not the mirror), offered so a caller for whom this
        request is sharing a container with something else can bound it the same way."""
        params: dict[str, Any] = {"path": path, "offset": offset}
        if limit is not None:
            params["limit"] = limit
        return self._client.request(
            "GET", self._instance_path("/files"), params=params,
            timeout=timeout, max_attempts=max_attempts,
        ).content

    def stat(self, path: str, *, timeout: float = 300.0, max_attempts: int | None = None) -> Stat:
        """See `get_file` above for what `timeout`/`max_attempts` are for."""
        body = _json(
            self._client.request(
                "GET", self._instance_path("/files"), params={"path": path, "stat": 1},
                timeout=timeout, max_attempts=max_attempts,
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
        for attempt in range(2):
            try:
                self._client.request(
                    "POST",
                    self._instance_path("/tar"),
                    params={"mode": "extract", "dest": remote_dir},
                    content=archive,
                    headers={"Content-Type": "application/gzip"},
                    timeout=300.0,
                    # A POST by method, idempotent in fact: it unpacks the same archive
                    # over the same directory, so a second one leaves the workspace as
                    # the first did. Said here so an upload over a flaky link still gets
                    # its retries.
                    repeatable=True,
                )
                return
            except BackendError as exc:
                # F-47, from the only vantage point that can settle it. The service
                # answered `bad_request (400): not a valid tar.gz archive` for
                # 1,324,532 bytes this method had built seconds earlier, which opened
                # cleanly on this machine and uploaded cleanly on the next attempt --
                # one occurrence in five. A 400 is the service saying the request was
                # wrong, and `_RETRY_STATUSES` rightly leaves it alone; but this caller
                # is holding the archive the service is complaining about, so it can
                # check the claim instead of believing it. If the bytes in hand are a
                # readable gzipped tar, then whatever reached the far end was not the
                # thing that was sent, and that is a transport failure wearing a client
                # error's clothes: worth exactly one more attempt, and then worth
                # saying plainly rather than pointing at the packing.
                #
                # Only that one claim, and the gate is the sentence rather than the
                # status. `mode=extract` in `OpenFoam_Instance/app/files.py` answers 400
                # for a family of other reasons, each with its own wording: `extract
                # requires ?dest=`, `path escapes the /work jail` and `cannot resolve
                # path: ...` from `resolve_in_jail`, `cannot parse tar listing entry:
                # ...`, `cannot parse tar member size: ...`, `tar member escapes jail:
                # ...`, `tar member is a link (not allowed): ...`, `tar member is a
                # special file (not allowed): ...`, and `extraction failed: ...` when the
                # listing read fine and the unpack did not. Every one of those is a true
                # statement about an archive that arrived intact. (The unpacked-size cap
                # is deliberately not in that list: `validate_tar_listing` raises
                # `payload_too_large` for it -- **413**, `request body exceeds the N MB
                # limit` -- so it never reaches the sentence test at all, the
                # `exc.status != 400` limb turns it away first. An earlier version of
                # this comment and of the test beside it called it a 400.) Gating on the
                # status alone
                # uploads a 1.3 MB archive a second time to be told the same thing, and
                # then contradicts a correct service with "what arrived was not what was
                # sent", which is F-47's own defect -- a message accusing the wrong
                # party -- with the parties swapped. The link case is not theoretical:
                # `_tar_gz_of` uses `tar.gettarinfo`, so a symlink in a case directory
                # becomes a member the service refuses by design.
                #
                # The gate is therefore the service's own sentence, which couples this
                # to `files.py`'s wording. That coupling fails safe: if the wording
                # changes, the check stops firing and the service's error is passed
                # through untouched, which is where this started.
                if exc.status != 400 or not _is_about_the_archives_bytes(exc.message):
                    raise
                described = _archive_opens_here(archive)
                if described is None:
                    raise
                if attempt == 0:
                    time.sleep(_SERVER_ERROR_RETRY_S)
                    continue
                raise BackendError(
                    f"the service could not read the upload ({exc.message}), twice, "
                    f"but these exact bytes open here as a gzipped tar ({described}): "
                    f"what arrived at the service was not what was sent",
                    code="archive_did_not_arrive",
                    status=exc.status,
                ) from exc

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

        `timeout` and `max_attempts` exist for the mirror's background cycles: a
        request left at its ordinary 300 s / 5-attempt defaults can sit stuck on one
        awkward path for minutes, and this is a copy nobody is waiting for. A caller
        in a hurry passes both down; `openreynolds pull`, asked for explicitly, passes
        neither and gets the patient defaults it always had.

        `via="volume"` asks the service to build the archive straight off the
        persistent volume instead of running `tar` in the workspace's own container.
        It exists for the two cases that need it -- a path the container's jail
        refuses (which is a real service-side bug, `qa-runs/FINDINGS.md` F-55) and a
        copy that must not wake a stopped workspace -- and it is deliberately NOT what
        the mirror uses. Measured (`qa-runs/tail_probe.py`): copying two case
        directories takes 39 s that way against 7 s through the container, and the
        contention it was introduced to avoid turned out to be the service blocking
        its own event loop, which is fixed. It also cannot carry the executable bit,
        which a file `fetch` was asked for by name may need.
        """
        if not remote_paths:
            return []
        params: dict[str, Any] = {"mode": "pack", "paths": remote_paths}
        if via:
            params["via"] = via
        response = self._client.request(
            "POST",
            self._instance_path("/tar"),
            params=params,
            timeout=timeout,
            max_attempts=max_attempts,
            # A read wearing a POST, because the path list travels in the body.
            repeatable=True,
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

        When the service finds no process group to signal it asks jobd what the job
        is doing: a job that had already ended comes back with its real ending, and
        one that may still be running is refused with `409 kill_not_delivered`
        rather than marked killed (F-63, where a `killed` answer covered a shell that
        went on running cases). A returned `killed` still says the signal was sent,
        not that every child of the job obeyed it; confirming that is `stop`'s job.
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
    """Build a gzipped tar of a directory's contents, owned by root, with the source mtimes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in sorted(local_dir.rglob("*")):
            if "__pycache__" in item.parts or item.name.endswith(".pyc"):
                continue
            info = tar.gettarinfo(str(item), arcname=str(item.relative_to(local_dir).as_posix()))
            # The source's own mtime, not 0: a 1970 file on the instance is one the
            # mirror cannot tell from a never-written one (F-54's `Allrun`).
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if item.is_file():
                with item.open("rb") as fh:
                    if fh.read(2) == b"#!":
                        # A shebang is the file saying it is a script. Windows has no
                        # execute bit to carry, so an `Allmesh` written there arrived
                        # as `Permission denied` and cost the model a turn.
                        info.mode |= 0o755
                    fh.seek(0)
                    tar.addfile(info, fh)
            else:
                tar.addfile(info)
    return buf.getvalue()


_ARCHIVE_UNREADABLE = "not a valid tar.gz archive"
"""The service's words for the one 400 that is a claim about the bytes themselves.

`OpenFoam_Instance/app/files.py:486` raises it when `tar tvzf` on the staged upload
exits non-zero, i.e. the upload would not open. Every other 400 on `mode=extract` is
about the request or about what a perfectly readable archive *contains* -- see
`put_tree`, which is why the gate is this sentence and not the status."""


def _is_about_the_archives_bytes(message: str) -> bool:
    """Whether a 400 is the service saying it could not read the upload at all.

    The service appends `tar`'s own stderr, so this is a prefix test rather than an
    equality one.
    """
    return message.startswith(_ARCHIVE_UNREADABLE)


def _archive_opens_here(data: bytes) -> str | None:
    """One line about the archive in hand, or `None` if it is not one.

    The whole of the evidence `put_tree` needs to answer a `not a valid tar.gz
    archive` from the far end: it walks the member headers of bytes already in memory,
    which for the 1.3 MB archive F-47 was about is a decompress and no I/O at all.
    Spent only on that one 400, after the message has been read -- an earlier draft
    called it before the status was checked and paid the decompress on every failure,
    a timeout and a 404 included.

    `zlib.error` is in the catch because it is not an `OSError` and because it really
    does escape `tarfile.open`. Fuzzing an archive with three random byte flips and
    reopening it, it came out of this call on 147 of 12,000 trials -- 1.2%, six seeds x
    2,000 flips of one 30,436-byte tar.gz, CPython 3.12.10. Read the reachability and
    not the rate: it moves with the archive and the seed, and no seed was ever recorded
    for the 1.3 MB archive the finding was about. The frame is the stable part. The
    innermost was `gzip.py:554` (`self._decompressor.decompress(b"", size)`) on all 147;
    the `TarFile.next` statement above it was `if not self.fileobj.read(1)`
    (`tarfile.py:2636`) on 130 and the `self.fileobj.seek(self.offset - 1)` on the line
    before it (`:2635`) on 17 -- both outside the block tarfile turns into a
    `ReadError`.

    That is the reachability of *this function on corrupt bytes*, which is not the
    reachability of the only caller: `put_tree` hands it an archive `_tar_gz_of` built
    moments earlier in the same process, and that cannot be a corrupt deflate stream.
    The catch earns its place from the contract rather than the odds -- a description or
    `None`, never an exception. This runs inside an `except` block, where anything
    escaping replaces the service's own error with a traceback about the client's
    evidence-gathering.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = sum(1 for _ in tar)
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        return None
    return f"{len(data)} bytes, {members} members"


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


def _most_recently_active_first(row: dict[str, Any]) -> tuple[str, str, str]:
    """Sort key for "the workspace this client is most likely still holding".

    `coalesce(last_active_at, created_at) desc, created_at desc` is the ordering
    `OpenFoam_Instance/sql/schema_f16.sql`'s repair step uses to decide which of an
    account's live rows to keep, and the two have to agree. That migration keeps the
    row a client is most likely still holding; this picks the row to re-attach to.
    If they disagree, a resumed session lands on the workspace the repair marked
    deleted -- the same study, on the other Volume, reading as empty.

    The values are the service's ISO-8601 timestamps, compared as strings. That is
    exact for same-offset ISO-8601, which is what the listing carries, and a trailing
    `Z` is normalised to `+00:00` so a mixed representation cannot invert the order.
    Where it is still inexact -- two rows written with different fractional-digit
    counts -- the disagreement is sub-second and the later keys settle it.

    `id` is the last key so the choice is deterministic even for two rows created in
    the same microsecond. That is the whole point of this function: an unordered
    `existing[0]` was not, and picked differently on two runs of the same resume.
    """
    created = _comparable_time(row.get("created_at"))
    active = _comparable_time(row.get("last_active_at")) or created
    return (active, created, str(row.get("id") or ""))


def _comparable_time(value: Any) -> str:
    """One timestamp as a string that sorts the way the instant does, or "" for none.

    `""` sorts before every real timestamp, which is the right place for a row whose
    time the service did not send: never the one preferred over a row that has one.
    """
    if not isinstance(value, str) or not value:
        return ""
    return value[:-1] + "+00:00" if value.endswith("Z") else value


class Starter:
    """The slow half of getting a workspace: bringing its container up.

    `reserve` settles WHICH instance in a fraction of a second and hands back one of
    these; `start()` sends the start call on a thread of its own, and `result()` is
    the live `HostedBackend` once that call has answered. Between the two, the
    session runs: header on screen, model talking, person typing. The one fact only
    the start call can settle -- whether this session brought the workspace up or
    joined one already running -- is read off its reply, exactly as `acquire` always
    read it.

    A started start cannot be called back: the service has the request. `cancel()`
    therefore only ever succeeds before `start()`, and says so with its answer.
    """

    def __init__(
        self,
        client: FoamdClient,
        instance_id: str,
        *,
        instances_held: int = 0,
        listed_as_running: bool = False,
    ):
        self.client = client
        self.instance_id = instance_id
        self.instances_held = instances_held
        """How many workspaces the account held when this one was chosen; 0 when
        nobody counted (an instance named outright). See `HostedBackend`."""
        self.listed_as_running = listed_as_running
        """What the listing said before the start call -- the pre-start answer to
        `was_already_running`, and the whole answer for a service too old to say."""
        self.started_at: float | None = None
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._backend: HostedBackend | None = None
        self._error: BackendError | None = None
        self._cancelled = False

    def start(self) -> None:
        """Begin bringing the workspace up, on a background thread. Idempotent."""
        if self._thread is not None or self._cancelled:
            return
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="workspace-start", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            reply = self.client.start_instance(self.instance_id)
        except BackendError as exc:
            self._error = exc
        except BaseException as exc:  # noqa: BLE001 - a thread's exception goes nowhere on its own
            self._error = BackendError(
                f"the workspace could not be started: {type(exc).__name__}: {exc}",
                code="start_failed",
            )
        else:
            backend = HostedBackend(self.client, self.instance_id)
            backend.instances_held = self.instances_held
            # Whether it was already up decides whether whoever asked for it should
            # put it back down again. The listing cannot answer that on its own: it
            # is read BEFORE the start call, a fresh row reads `stopped`, and a row
            # only turns `running` once the start has happened -- so two sessions
            # listing within the same second both concluded they had started the
            # workspace, and the first to exit stopped it from under the other. The
            # start route says which of the two it did. Its absence is tolerated so
            # an older service still works: there the pre-start listing is all there
            # is, which is the behaviour this has always had.
            started_new = reply.get("started_new") if isinstance(reply, dict) else None
            backend.was_already_running = (
                self.listed_as_running if started_new is None else not bool(started_new)
            )
            self._backend = backend
        finally:
            self._done.set()

    def cancel(self) -> bool:
        """Call the start off. True only if it had not begun -- a start that is under
        way is the service's now, and `PendingBackend.shutdown` is the way to undo
        one of those (wait for it, then stop it)."""
        if self._thread is not None:
            return False
        self._cancelled = True
        self._error = BackendError("the workspace start was called off", code="start_cancelled")
        self._done.set()
        return True

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def elapsed_s(self) -> float:
        """Seconds since `start()`; 0 before it."""
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def result(self, timeout: float | None = None) -> HostedBackend:
        """The live backend, waiting for the start to answer. Starts it if nobody has.

        Raises the start call's own `BackendError`, or `start_timeout` if `timeout`
        ran out first -- `None` waits as long as the start call itself does, which
        is bounded by the client's request timeout and retries."""
        self.start()
        if not self._done.wait(timeout):
            raise BackendError(
                f"the workspace did not answer its start within {timeout:.0f} s",
                code="start_timeout",
            )
        if self._error is not None:
            raise self._error
        assert self._backend is not None
        return self._backend

    def pending(self, **kwargs: Any) -> PendingBackend:
        """A `PendingBackend` for this workspace, with what is already known filled
        in. Whoever runs the start is expected to `resolve` it with `result()` -- or
        `fail` it -- once the workspace is set up for use."""
        return PendingBackend(
            self.instance_id,
            instances_held=self.instances_held,
            was_already_running=self.listed_as_running,
            closer=self.client.close,
            **kwargs,
        )


def reserve(
    base_url: str,
    api_key: str,
    instance_id: str | None = None,
) -> tuple[FoamdClient, str, Starter]:
    """Settle which workspace, without starting it: the named instance, else an
    existing one, else a new row. A fraction of a second, all of it database.

    Deleting an instance destroys its persistent volume, so reuse is the default and
    nothing here ever deletes.

    Which existing one is not arbitrary any more. The service caps concurrent
    instances and that cap is no longer 1, so an account can hold several workspaces
    at once, each with its own Volume and its own files. Taking `existing[0]` from an
    unordered listing silently attached a resumed session to whichever row the
    service happened to return first: the study's files are on a different Volume, so
    the failure does not look like a wrong choice, it looks like an empty workspace --
    the case directory gone, the mesh gone, nothing anywhere saying a different
    workspace was joined. So the listing is ordered (`_most_recently_active_first`)
    and the count is handed on (`Starter.instances_held`, then the backend's),
    because collapsing five workspaces to one id is a thing the user has to be told.

    The client is closed on failure here, as `acquire` always closed it; a starter
    handed back is the caller's, and so is the client under it.
    """
    client = FoamdClient(base_url, api_key)
    try:
        listed_as_running = False
        held = 0
        if instance_id is None:
            existing = sorted(
                (
                    inst
                    for inst in client.list_instances()
                    if inst.get("status") != "deleted"
                ),
                key=_most_recently_active_first,
                reverse=True,
            )
            held = len(existing)
            if existing:
                instance_id = existing[0]["id"]
                listed_as_running = existing[0].get("status") == "running"
            else:
                instance_id = client.create_instance()
                held = 1
    except BaseException:
        client.close()
        raise
    starter = Starter(client, instance_id, instances_held=held, listed_as_running=listed_as_running)
    return client, instance_id, starter


def acquire(
    base_url: str,
    api_key: str,
    instance_id: str | None = None,
) -> tuple[HostedBackend, FoamdClient, str]:
    """Get a workspace, up and ready: `reserve`, then wait for the start.

    The blocking shape, for callers with nothing to do while the workspace comes up
    -- `openreynolds files`, `pull`, `push`, `stop`. A session uses `reserve` and
    talks to the person meanwhile.
    """
    client, instance_id, starter = reserve(base_url, api_key, instance_id)
    try:
        backend = starter.result()
    except BaseException:
        client.close()
        raise
    return backend, client, instance_id
