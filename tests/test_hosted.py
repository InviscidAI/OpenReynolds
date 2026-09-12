"""The transport details the demo scripts skipped: retries and both error shapes."""

from __future__ import annotations

import zlib

import httpx2 as httpx
import pytest

from openreynolds.backend.base import BackendError
from openreynolds.backend import hosted as hosted_mod
from openreynolds.backend.hosted import FoamdClient, _decode_error, _retry_delay


def response(status: int, json_body=None, text: str = "", headers=None) -> httpx.Response:
    if json_body is not None:
        return httpx.Response(status, json=json_body, headers=headers or {})
    return httpx.Response(status, text=text, headers=headers or {})


def test_application_error_envelope():
    error = _decode_error(
        response(404, {"error": "not_found", "message": "instance not found"})
    )
    assert error.code == "not_found"
    assert error.status == 404
    assert "instance not found" in error.message


def test_validation_error_shape_is_also_understood():
    """Request-validation failures come from the web framework, not the app."""
    error = _decode_error(
        response(
            422,
            {"detail": [{"loc": ["body", "timeout_s"], "msg": "must be <= 300"}]},
        )
    )
    assert error.code == "invalid_request"
    assert error.status == 422
    assert "timeout_s: must be <= 300" in error.message


def test_non_json_body_still_produces_a_readable_error():
    error = _decode_error(response(500, text="upstream exploded"))
    assert error.status == 500
    assert "upstream exploded" in error.message


def test_retry_after_header_is_honoured():
    assert _retry_delay(response(503, {}, headers={"Retry-After": "7"}), 0) == 7.0


def test_cold_start_has_a_default_delay():
    """The service returns 503 while a workspace boots; that is expected, not fatal."""
    assert _retry_delay(response(503, {}), 0) == 10.0


def test_other_failures_back_off_exponentially():
    assert _retry_delay(None, 0) == 1.0
    assert _retry_delay(None, 3) == 8.0
    assert _retry_delay(None, 10) == 30.0


def test_a_server_error_is_retried_quickly_rather_than_backed_off_from():
    """A 500 here is the sandbox being rebuilt underneath the call, not a queue: the
    same read-only GET answered 500 and then succeeded four seconds later, unchanged.
    Backing off exponentially would turn an absorbed hiccup into a visible stall."""
    assert _retry_delay(response(500, {}), 0) == 1.0
    assert _retry_delay(response(500, {}), 3) == 1.0


def test_a_server_error_is_absorbed_instead_of_costing_a_model_turn(monkeypatch):
    """18% of one live study's tool calls came back http_error (500). Every one reached
    the model as a failed tool call, and a failed tool call costs a whole turn: the
    entire conversation re-read to learn that the backend blinked."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = {"n": 0}

    def flaky(method, path, timeout=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return response(500, {"error": "internal", "message": "boom"})
        return response(200, {"ok": True})

    monkeypatch.setattr(client._client, "request", flaky)
    assert client.request("GET", "/v1/whatever").status_code == 200
    assert calls["n"] == 2, "one retry, and the model never heard about it"


def test_client_retries_a_cold_start_then_succeeds(monkeypatch):
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        if len(calls) < 3:
            return response(503, {"error": "unavailable", "message": "booting"},
                            headers={"Retry-After": "0"})
        return response(200, {"ok": True})

    monkeypatch.setattr(client._client, "request", fake_request)
    result = client.request("GET", "/v1/instances")

    assert result.json() == {"ok": True}
    assert len(calls) == 3


def test_client_does_not_retry_a_client_error(monkeypatch):
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append(path)
        return response(404, {"error": "not_found", "message": "gone"})

    monkeypatch.setattr(client._client, "request", fake_request)
    with pytest.raises(BackendError) as excinfo:
        client.request("GET", "/v1/instances/x")

    assert excinfo.value.code == "not_found"
    assert len(calls) == 1


def test_client_gives_up_after_repeated_failures(monkeypatch):
    client = FoamdClient("https://example.invalid", "of_live_test")

    def always_down(method, path, **kwargs):
        return response(503, {"error": "unavailable", "message": "no"},
                        headers={"Retry-After": "0"})

    monkeypatch.setattr(client._client, "request", always_down)
    with pytest.raises(BackendError) as excinfo:
        client.request("GET", "/v1/instances")
    assert excinfo.value.code == "unavailable"


def test_a_bodyless_success_is_a_backend_error_not_a_json_traceback():
    """A long synchronous exec can come back as a bodyless 200 from something between
    here and the service. Seen live: JSONDecodeError escaped BackendError handling."""
    from openreynolds.backend.hosted import _json

    with pytest.raises(BackendError) as excinfo:
        _json(response(200, text=""))

    assert excinfo.value.code == "bad_response"
    assert "not JSON" in excinfo.value.message
    assert "body was empty" in excinfo.value.message


def test_a_non_json_success_body_is_quoted_back():
    from openreynolds.backend.hosted import _json

    with pytest.raises(BackendError) as excinfo:
        _json(response(200, text="<html>gateway timeout</html>"))
    assert "gateway timeout" in excinfo.value.message


def test_a_good_body_still_decodes():
    from openreynolds.backend.hosted import _json

    assert _json(response(200, {"exit_code": 0})) == {"exit_code": 0}


def test_extraction_is_atomic_per_file(tmp_path):
    """Files land whole or not at all: temp-then-rename, no .part leftovers.

    Two syncs of the same study can overlap (a cycle outliving a bounded join,
    `openreynolds pull` in another terminal), and a reader must never see the
    middle of a write."""
    import io as _io
    import tarfile as _tarfile

    from openreynolds.backend.hosted import _extract_tar_gz

    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w:gz") as tar:
        payload = b"x" * 4096
        info = _tarfile.TarInfo("case/log.simpleFoam")
        info.size = len(payload)
        tar.addfile(info, _io.BytesIO(payload))

    written = _extract_tar_gz(buf.getvalue(), tmp_path)

    assert [p.name for p in written] == ["log.simpleFoam"]
    assert (tmp_path / "case" / "log.simpleFoam").read_bytes() == b"x" * 4096
    leftovers = [p for p in tmp_path.rglob("*") if ".part" in p.name]
    assert leftovers == []


def test_a_script_by_shebang_is_shipped_executable(tmp_path):
    """The geometry desk writes `Allmesh` on the user's machine and ships it with
    `put_tree`. On Windows there is no execute bit to carry, so it arrived as
    `Permission denied` and cost the model a turn. A shebang is the file saying it is
    a script; the archive says so too."""
    import io as _io
    import tarfile as _tarfile

    from openreynolds.backend.hosted import _tar_gz_of

    case = tmp_path / "case"
    case.mkdir()
    (case / "Allmesh").write_text("#!/bin/sh\nset -e\n", encoding="utf-8")
    (case / "geometry.json").write_text("{}", encoding="utf-8")
    (case / "Allmesh").chmod(0o644)

    with _tarfile.open(fileobj=_io.BytesIO(_tar_gz_of(case)), mode="r:gz") as tar:
        modes = {m.name: m.mode for m in tar.getmembers()}
    assert modes["Allmesh"] & 0o111 == 0o111
    assert modes["geometry.json"] & 0o111 == 0


# -- the edge's 150 s redirect -------------------------------------------------------
#
# Modal's edge answers any web request past 150 s with a bodyless 303. The command has
# usually run by then, so the redirect is where the real answer is. `EXEC_MAX_TIMEOUT_S`
# is 300 and stays there: the two numbers are decoupled, and a 250 s exec is a supported
# thing to ask for -- it just needs a client that follows the redirect. Stubbed, because
# what is being tested is the redirect and not the wait.


def _edge_with_a_150s_redirect(result: dict) -> httpx.MockTransport:
    """The edge as it behaves: a bare 303, empty body, to where the answer really is."""

    def handler(request):
        if request.url.path.endswith("/exec"):
            return httpx.Response(303, headers={"Location": "/v1/exec-result"}, content=b"")
        return httpx.Response(200, json=result)

    return httpx.MockTransport(handler)


def _client_against(transport: httpx.MockTransport) -> FoamdClient:
    """A client wired to a stub transport and otherwise built exactly as the real one.

    `follow_redirects=False` is not a test setting: it is what `FoamdClient.__init__`
    does, and it is what makes httpx hand each redirect back as `next_request` for
    `_send_following_safe_redirects` to decide about. Building the stub any other way
    would test httpx's redirect policy instead of this module's."""
    client = FoamdClient("https://svc.example", "of_live_test")
    client._client = httpx.Client(
        base_url="https://svc.example",
        transport=transport,
        follow_redirects=False,
    )
    return client


def test_a_redirect_with_nowhere_to_go_is_named_not_a_json_decode_failure():
    """The whole of F-45, as the model saw it: `bad_response (303): the body was empty`.
    True, and useless -- it named the symptom and hid the one thing to act on.

    A 303 carrying no `Location` is the one redirect nothing can follow, so it is the
    case that still has to be named rather than followed."""
    from openreynolds.backend.hosted import HostedBackend

    def handler(request):
        return httpx.Response(303, content=b"")

    backend = HostedBackend(_client_against(httpx.MockTransport(handler)), "inst-1")

    with pytest.raises(BackendError) as excinfo:
        backend.exec("sleep 200", timeout_s=250)

    assert excinfo.value.code == "redirect_not_followed", (
        "not bad_response -- the JSON decoder must not be where this lands"
    )
    assert excinfo.value.status == 303
    assert "no Location" in excinfo.value.message


def test_a_long_exec_completes_through_the_redirect():
    """A 250 s exec is under `EXEC_MAX_TIMEOUT_S` and over the edge's window. Followed,
    the redirect carries the real result: measured live, `sleep 200` returns rc=0.

    This is F-45, and narrowing the redirect policy for F-47 must not cost it. The exec
    POST carries a body, so it is precisely the case a blanket "never follow a redirect
    on a request with a body" rule would break."""
    from openreynolds.backend.hosted import HostedBackend

    transport = _edge_with_a_150s_redirect(
        {"exit_code": 0, "output": "done", "truncated": False}
    )
    backend = HostedBackend(_client_against(transport), "inst-1")

    result = backend.exec("sleep 200", timeout_s=250)

    assert result.exit_code == 0
    assert result.output == "done"


# -- and only the redirects that keep the request whole -------------------------------
#
# F-47's second hypothesis: httpx re-issues a 303 (and a 302, and a 301 on a POST) as a
# GET with the body dropped, so the blanket `follow_redirects=True` that fixed F-45 made
# every body-carrying request a body-losing one.
#
# Half of that hypothesis is dead and half is only unobserved. The 303 -- the one
# redirect anything here has ever seen -- is what an origin answers after it has taken
# the request, so the POST carrying the archive goes out whole and the hop only collects
# the result: no body is lost. The 301/302 relocation is not disposed of by the service's
# route table, the way an earlier draft of this comment said it was. A followed redirect
# is re-issued at its `Location`, not at the path that was asked -- this file's own edge
# stub sends `Location: /v1/exec-result` -- so `app/files.py` mounting the tar path
# POST-only means a 405 for a GET aimed back at that path and says nothing at all about
# a GET aimed anywhere else. No `Location` for a 3xx on that path has ever been captured.
#
# So what follows is not F-47's fix; it is a hazard closed on its own terms. For 303 the
# behaviour is what `follow_redirects=True` already did, and 301/302 stop being able to
# turn a write into a read without a word. Each hop is pinned.


def _upload(client: FoamdClient, size: int = 64) -> None:
    client.request(
        "POST",
        "/v1/instances/inst-1/tar",
        params={"mode": "extract", "dest": "/work/c"},
        content=b"x" * size,
        headers={"Content-Type": "application/gzip"},
        repeatable=True,
    )


def test_a_302_on_an_upload_is_refused_rather_than_re_sent_without_its_body():
    """The body-losing event, refused where it used to be silent.

    httpx turns a 302 on any method but HEAD into a GET and drops the stream, which for
    an upload means the service is asked something other than what the caller asked and
    answers about that instead. Whether that is what happened in F-47 is not settled
    here: the GET goes to the redirect's `Location`, and no `Location` was ever captured
    for the 3xx that preceded that 400. But a 302 is `the resource moved`, not `the
    answer is over there`, so there is nothing to weigh: it is named, not followed."""
    seen = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(302, headers={"Location": "/v1/elsewhere"}, content=b"")

    with pytest.raises(BackendError) as excinfo:
        _upload(_client_against(httpx.MockTransport(handler)))

    assert excinfo.value.code == "redirect_would_lose_the_body"
    assert excinfo.value.status == 302
    assert seen == ["POST"], "the bodyless GET must never be sent at all"


def test_a_307_on_an_upload_is_replayed_whole():
    """307 and 308 are the two redirects that preserve the method and the body, so
    following them cannot lose anything -- httpx replays the same stream, and the
    service sees the request that was made."""
    seen = []

    def handler(request):
        seen.append((request.method, request.read()))
        if len(seen) == 1:
            return httpx.Response(307, headers={"Location": "/v1/instances/inst-1/tar2"})
        return httpx.Response(200, json={"extracted_to": "/work/c"})

    _upload(_client_against(httpx.MockTransport(handler)))

    assert [method for method, _ in seen] == ["POST", "POST"]
    assert seen[0][1] == seen[1][1] == b"x" * 64, "the archive travelled on both legs"


def test_a_redirect_on_a_read_is_followed_because_a_read_has_nothing_to_lose():
    """The other half of the narrowing: a GET has no body, so httpx's downgrade to a
    GET changes nothing about it. Every read this client makes keeps its redirects."""

    def handler(request):
        if request.url.path == "/v1/instances":
            return httpx.Response(303, headers={"Location": "/v1/list-result"}, content=b"")
        return httpx.Response(200, json=[{"id": "inst-1"}])

    client = _client_against(httpx.MockTransport(handler))
    assert client.list_instances() == [{"id": "inst-1"}]


def test_an_answer_collected_through_a_303_is_returned_as_the_answer():
    """The 303 is followed and its answer is the call's answer -- said out loud, because
    it is the one place this module accepts a hop the body did not travel on.

    The licence is what 303 means (RFC 9110 15.4.4): the origin has already received and
    acted on the request, and the answer is at the other URI. So the POST's 179 bytes
    going out on the first leg and 0 on the second is the status working, not a body
    being lost -- and F-45 measured the reading, `sleep 200`'s own rc=0 coming back
    through the hop at 203.9 s. If that reading is ever wrong for some route, this is
    the test that will have to change, so the shape is pinned rather than assumed."""
    legs = []

    def handler(request):
        legs.append((request.method, len(request.read())))
        if request.method == "POST":
            return httpx.Response(303, headers={"Location": "/v1/tar-result"}, content=b"")
        return httpx.Response(200, json={"extracted_to": "/work/c"})

    _upload(_client_against(httpx.MockTransport(handler)), size=179)

    assert legs == [("POST", 179), ("GET", 0)], (
        "one POST carrying the archive, one bodyless GET collecting the answer"
    )


def test_a_4xx_collected_through_a_303_is_the_services_own_answer(monkeypatch):
    """A 400 that arrives over the hop is not retried and not relabelled.

    An earlier draft of this fix made every 4xx behind a hop `ambiguous`, on the reading
    that the service might never have got the body. That reading contradicts the one
    that licenses following a 303 at all, and it cost real behaviour: a 401/403/413 was
    retried like a transient. `not a valid tar.gz archive` here is the service's
    sentence and the caller's to interpret -- `put_tree`, which is holding the bytes,
    is the only place in this module that argues with it."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request.url.path)
            return httpx.Response(303, headers={"Location": "/v1/tar-result"}, content=b"")
        return httpx.Response(400, json={"error": "bad_request",
                                         "message": "not a valid tar.gz archive"})

    with pytest.raises(BackendError) as excinfo:
        _upload(_client_against(httpx.MockTransport(handler)), size=1_324_532)

    assert len(posts) == 1, "a 400 through a hop is still a 400; one upload, not five"
    assert excinfo.value.message == "not a valid tar.gz archive"
    assert excinfo.value.code == "bad_request"


def test_a_429_through_a_303_still_backs_off_instead_of_raising_at_once(monkeypatch):
    """The refinement that draft pre-empted, pinned so it cannot be pre-empted again.

    429 is in `_DECLINED_STATUSES`: the service turned the call away without acting, so
    even a non-repeatable exec may try again. Making a hop-collected 4xx `ambiguous`
    made that exec raise `rate_limited` on the first answer -- a regression aimed
    squarely at the long-exec path the 303 handling exists for."""
    from openreynolds.backend.hosted import HostedBackend

    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    execs = []

    def handler(request):
        if request.method == "POST":
            execs.append(request.url.path)
            return httpx.Response(303, headers={"Location": "/v1/exec-result"}, content=b"")
        if len(execs) < 3:
            return httpx.Response(429, json={"error": "rate_limited", "message": "slow down"})
        return httpx.Response(200, json={"exit_code": 0, "output": "done",
                                         "truncated": False})

    backend = HostedBackend(_client_against(httpx.MockTransport(handler)), "inst-1")
    result = backend.exec("sleep 200", timeout_s=250)

    assert result.exit_code == 0
    assert len(execs) == 3, "backed off twice and got there, rather than raising at once"


def test_a_400_collected_through_a_303_hop_is_raised_on_the_first_answer():
    """A hop earns a 4xx no special treatment, in either direction.

    An exec is not repeatable -- the command is arbitrary and may already have moved
    files, so a repeat is a second run, not a second look, the 13-minute command re-run
    that `_REPEATABLE_METHODS` exists to prevent. But that is not what decides this one:
    a 400 is not in `_RETRY_STATUSES` at all, so it is raised on the first answer whether
    or not a hop was involved. (This test was named for a deleted branch that made a 4xx
    behind a hop ambiguous; what it pins is the absence of that branch.)"""
    from openreynolds.backend.hosted import HostedBackend

    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request.url.path)
            return httpx.Response(303, headers={"Location": "/v1/exec-result"}, content=b"")
        return httpx.Response(400, json={"error": "bad_request", "message": "nope"})

    backend = HostedBackend(_client_against(httpx.MockTransport(handler)), "inst-1")
    with pytest.raises(BackendError) as excinfo:
        backend.exec("sleep 200", timeout_s=250)

    assert len(posts) == 1
    assert excinfo.value.message == "nope"


# -- the cap on how far a chain may run ----------------------------------------------
#
# Untested until F-47's review measured it wrong: the loop used to check `next_request`
# only at the top, so a chain of exactly `_MAX_REDIRECTS` hops ending in a 200 made all
# six requests, received the answer, and then raised `too_many_redirects` carrying
# `status: 200` -- F-45's "the work succeeded and the client says it failed", back at the
# bound. Both sides of the boundary are pinned here.


def _a_chain_of(hops: int) -> httpx.MockTransport:
    """`hops` redirects and then an answer. Each 307 so the body survives the hop and
    the count is about the cap and nothing else."""
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if len(seen) <= hops:
            return httpx.Response(307, headers={"Location": f"/v1/hop-{len(seen)}"})
        return httpx.Response(200, json={"extracted_to": "/work/c"})

    return httpx.MockTransport(handler)


def test_a_chain_that_ends_within_the_cap_returns_its_answer():
    """Exactly `_MAX_REDIRECTS` hops are allowed, so the fifth hop's answer is an
    answer. Off by one here means an exec that ran and succeeded is reported as a
    failure, which is the whole cost of F-45 paid again at the bound."""
    from openreynolds.backend.hosted import _MAX_REDIRECTS

    _upload(_client_against(_a_chain_of(_MAX_REDIRECTS)))


def test_a_chain_that_runs_past_the_cap_is_named_and_carries_a_redirect_status():
    """One hop further is a loop this client will not chase. The error's `status` is the
    redirect that was refused -- never the success code of an answer being thrown away,
    which is what callers that branch on `.status` (`mirror.is_a_fact_about_the_path`)
    would otherwise be handed on a failure."""
    from openreynolds.backend.hosted import _MAX_REDIRECTS

    with pytest.raises(BackendError) as excinfo:
        _upload(_client_against(_a_chain_of(_MAX_REDIRECTS + 1)))

    assert excinfo.value.code == "too_many_redirects"
    assert excinfo.value.status == 307, "the redirect refused, not an answer discarded"
    assert str(_MAX_REDIRECTS) in excinfo.value.message


def test_an_ordinary_400_with_no_redirect_in_sight_is_still_not_retried():
    """The guard on the change above: nothing here makes 400 retryable in general."""
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(
            400, json={"error": "bad_request", "message": "path escapes the /work jail"}
        )

    with pytest.raises(BackendError) as excinfo:
        _upload(_client_against(httpx.MockTransport(handler)))

    assert len(seen) == 1, "one attempt; a 400 that means it is a 400 that means it"
    # `resolve_in_jail`'s own sentence (`app/files.py:200`), not an invented one.
    assert excinfo.value.message == "path escapes the /work jail"


def test_a_bare_redirect_reaching_the_decoder_is_named_there_too():
    """The sign-in helpers do not go through `FoamdClient.request`, so the decoder
    carries the same guard -- one for every path a response takes to a JSON body."""
    from openreynolds.backend.hosted import _json

    with pytest.raises(BackendError) as excinfo:
        _json(response(303, text="", headers={"Location": "/elsewhere"}))
    assert excinfo.value.code == "redirect_not_followed"


def test_the_signing_in_helpers_follow_redirects_too():
    """The mistake, already in this module four times over: a bare client with no
    `follow_redirects`. The edge does not know these questions are short ones."""
    from openreynolds.backend.hosted import auth_config, device_code

    def handler(request):
        if request.url.path in ("/dashboard/config.json", "/v1/device/code"):
            return httpx.Response(303, headers={"Location": "/after"}, content=b"")
        return httpx.Response(200, json={"supabase_url": "https://sb.example",
                                         "publishable_key": "pk", "device_code": "dc"})

    transport = httpx.MockTransport(handler)
    assert auth_config("https://svc.example", transport=transport)["publishable_key"] == "pk"
    assert device_code("https://svc.example", transport=transport)["device_code"] == "dc"


# -- the upload can check the claim being made about it -------------------------------
#
# F-47 as it actually arrived: `bad_request (400): not a valid tar.gz archive` for a
# 1,324,532-byte archive `_tar_gz_of` had built seconds earlier, which opened cleanly on
# the machine that sent it and uploaded cleanly on the next attempt -- one occurrence in
# five. The service is describing bytes; `put_tree` is holding them; nobody has to guess.


def _a_case(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "geometry.json").write_text('{"kind": "wing"}', encoding="utf-8")
    return case


def test_an_upload_rejected_as_a_bad_archive_is_tried_once_more(monkeypatch, tmp_path):
    """A 400 is the service saying the request was wrong, and `_RETRY_STATUSES` rightly
    leaves it alone. This caller can do better than believe it: it built the archive, it
    still has it, and if those bytes open then what reached the far end was not what was
    sent. In the live occurrence the very next attempt worked."""
    from openreynolds.backend.hosted import HostedBackend

    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    sent = []

    def rejects_once(method, path, timeout=None, **kwargs):
        sent.append(len(kwargs.get("content", b"")))
        if len(sent) == 1:
            return response(400, {"error": "bad_request",
                                  "message": "not a valid tar.gz archive: gzip: stdin: "
                                             "unexpected end of file"})
        return response(200, {"extracted_to": "/work/case"})

    monkeypatch.setattr(client._client, "request", rejects_once)
    HostedBackend(client, "inst-1").put_tree(_a_case(tmp_path), "/work/case")

    assert len(sent) == 2
    assert sent[0] == sent[1] > 0, "the same archive, whole, both times"


def test_an_upload_rejected_twice_names_the_transport_not_the_packing(monkeypatch, tmp_path):
    """What the finding is really about: a message pointing at the wrong party. `not a
    valid tar.gz archive` reads as a bad file, a wrong path or a packing bug, and every
    one of those is expensive to rule out and was wrong. The service's own words are
    kept -- and then contradicted, by the one machine in a position to contradict them."""
    from openreynolds.backend.hosted import HostedBackend

    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    sent = []

    def always_rejects(method, path, timeout=None, **kwargs):
        sent.append(path)
        return response(400, {"error": "bad_request", "message": "not a valid tar.gz archive"})

    monkeypatch.setattr(client._client, "request", always_rejects)
    with pytest.raises(BackendError) as excinfo:
        HostedBackend(client, "inst-1").put_tree(_a_case(tmp_path), "/work/case")

    assert len(sent) == 2, "tried twice, then said so -- not tried forever"
    assert excinfo.value.code == "archive_did_not_arrive"
    assert "not a valid tar.gz archive" in excinfo.value.message
    assert "open here as a gzipped tar" in excinfo.value.message
    assert "members" in excinfo.value.message, "the evidence, not just the assertion"


def test_an_upload_refused_for_a_reason_that_is_not_the_archive_is_passed_through(
    monkeypatch, tmp_path
):
    """A failure that is not about the bytes reaches the caller unchanged and on the
    first answer -- a 404 has nothing to do with what the archive contains."""
    from openreynolds.backend.hosted import HostedBackend

    client = FoamdClient("https://example.invalid", "of_live_test")
    sent = []

    def gone(method, path, timeout=None, **kwargs):
        sent.append(path)
        return response(404, {"error": "not_found", "message": "instance not found"})

    monkeypatch.setattr(client._client, "request", gone)
    with pytest.raises(BackendError) as excinfo:
        HostedBackend(client, "inst-1").put_tree(_a_case(tmp_path), "/work/case")

    assert len(sent) == 1
    assert excinfo.value.code == "not_found"


@pytest.mark.parametrize("message", [
    "tar member is a link (not allowed): 0/U",
    "tar member is a special file (not allowed): dev/null",
    "cannot parse tar listing entry: -rw-r--r-- garbage",
    "cannot parse tar member size: -rw-r--r-- user/group x 2026-08-01 12:00 p",
    "tar member escapes jail: ../etc/passwd",
    "extract requires ?dest=",
    "extraction failed: tar: write error",
    "path escapes the /work jail",
    "cannot resolve path: realpath: /work/c: Permission denied",
])
def test_a_400_about_what_the_archive_contains_is_believed_not_argued_with(
    monkeypatch, tmp_path, message
):
    """The gate is the service's sentence, not the status.

    Every message above is one `OpenFoam_Instance/app/files.py` really writes on the
    `mode=extract` path -- the first five from `validate_tar_listing`, then the missing
    `dest` and the failed unpack from the route itself, then the two `resolve_in_jail`
    raises for the destination. Each is a true statement about an archive that arrived
    perfectly intact. Gating on the status alone uploads 1.3 MB a second time to be told
    the same thing, and then answers a correct service with "what arrived was not what
    was sent" -- which is F-47's own defect, a message accusing the wrong party, with the
    parties swapped. The link case is not hypothetical: `_tar_gz_of` uses
    `tar.gettarinfo`, so a symlink in a case directory becomes a member the service
    refuses by design.

    Two sentences that stood in this list until now (`unpacked size exceeds the limit`,
    `dest is outside /work`) are not strings `files.py` produces; the real ones are the
    413 pinned in the test below and `path escapes the /work jail` above."""
    from openreynolds.backend.hosted import HostedBackend

    client = FoamdClient("https://example.invalid", "of_live_test")
    sent = []

    def refuses(method, path, timeout=None, **kwargs):
        sent.append(path)
        return response(400, {"error": "bad_request", "message": message})

    monkeypatch.setattr(client._client, "request", refuses)
    with pytest.raises(BackendError) as excinfo:
        HostedBackend(client, "inst-1").put_tree(_a_case(tmp_path), "/work/case")

    assert len(sent) == 1, "one upload -- the service is right and repeating proves it"
    assert excinfo.value.code == "bad_request"
    assert excinfo.value.message == message, "the service's words, whole and unargued"


def test_an_archive_over_the_unpacked_cap_is_a_413_and_never_reaches_the_sentence_test(
    monkeypatch, tmp_path
):
    """The one refusal about the archive's *contents* that is not a 400.

    `validate_tar_listing` raises `payload_too_large` when the members sum past
    `MAX_TAR_UNPACKED_MB` (`OpenFoam_Instance/app/files.py:380-381`), which is
    `ApiError(413, "payload_too_large", "request body exceeds the N MB limit")` --
    `app/errors.py:62-64`. So the `exc.status != 400` limb turns it away before
    `_is_about_the_archives_bytes` is consulted: one upload, the service's own words,
    and no argument. Pinned because both the comment in `put_tree` and the list above
    used to call this a 400, and a re-upload of an archive that is genuinely too big is
    exactly the wasted 1.3 MB the sentence gate exists to avoid."""
    from openreynolds.backend.hosted import HostedBackend

    client = FoamdClient("https://example.invalid", "of_live_test")
    sent = []

    def too_big(method, path, timeout=None, **kwargs):
        sent.append(path)
        return response(413, {"error": "payload_too_large",
                              "message": "request body exceeds the 512 MB limit"})

    monkeypatch.setattr(client._client, "request", too_big)
    with pytest.raises(BackendError) as excinfo:
        HostedBackend(client, "inst-1").put_tree(_a_case(tmp_path), "/work/case")

    assert len(sent) == 1, "413 is not a 400 and is not retried by this gate"
    assert excinfo.value.status == 413
    assert excinfo.value.code == "payload_too_large"
    assert excinfo.value.message == "request body exceeds the 512 MB limit"


def test_the_archive_check_reads_the_services_own_wording():
    """`_is_about_the_archives_bytes` is coupled to one sentence in `files.py`, which
    appends `tar`'s stderr to it -- hence a prefix test. The coupling fails safe: if the
    wording ever changes, the check stops firing and the service's error passes through
    untouched, which is exactly where this started."""
    from openreynolds.backend.hosted import _is_about_the_archives_bytes

    assert _is_about_the_archives_bytes("not a valid tar.gz archive")
    assert _is_about_the_archives_bytes(
        "not a valid tar.gz archive: gzip: stdin: unexpected end of file")
    assert not _is_about_the_archives_bytes("tar member is a link (not allowed): 0/U")


def test_bytes_that_do_not_open_here_leave_the_services_verdict_standing(
    monkeypatch, tmp_path
):
    """The contradiction is only earned when the evidence is in hand. If the archive
    this client is holding does not open either, the service is agreeing with the
    client and there is nothing to argue about -- one attempt, its words kept."""
    from openreynolds.backend.hosted import HostedBackend

    client = FoamdClient("https://example.invalid", "of_live_test")
    monkeypatch.setattr(hosted_mod, "_tar_gz_of", lambda _d: b"not a gzip at all")
    sent = []

    def refuses(method, path, timeout=None, **kwargs):
        sent.append(path)
        return response(400, {"error": "bad_request",
                              "message": "not a valid tar.gz archive"})

    monkeypatch.setattr(client._client, "request", refuses)
    with pytest.raises(BackendError) as excinfo:
        HostedBackend(client, "inst-1").put_tree(_a_case(tmp_path), "/work/case")

    assert len(sent) == 1
    assert excinfo.value.code == "bad_request"


def test_the_archive_check_answers_none_and_never_raises(monkeypatch):
    """`_archive_opens_here` is called from inside an `except BackendError`, so anything
    escaping it replaces the service's error with a traceback about the client's own
    evidence-gathering -- the caller would lose the very message it was trying to
    improve on.

    `zlib.error` is in the catch and is not an `OSError`. It escapes `tarfile.open(...)`
    + a member walk for real: three random byte flips in a good `w:gz` archive, 147 of
    12,000 trials (six seeds x 2,000, one 30,436-byte archive, CPython 3.12.10). The
    innermost frame was `gzip.py:554` every time; the `TarFile.next` statement above it
    was `if not self.fileobj.read(1)` (`tarfile.py:2636`) on 130 and the
    `self.fileobj.seek(self.offset - 1)` before it on 17 -- both outside the block
    tarfile converts into `ReadError`.

    That rate is a property of the fuzz, not of production: `put_tree` hands this
    function an archive built moments earlier in the same process. And which byte flip
    does it is not stable enough to pin in a test, so the escape is injected here and
    the contract -- a description or `None`, never an exception -- is what is
    asserted."""
    from openreynolds.backend.hosted import _archive_opens_here

    assert _archive_opens_here(b"") is None
    assert _archive_opens_here(b"not a gzip at all") is None

    def raises_zlib(*_a, **_k):
        raise zlib.error("Error -3 while decompressing data: invalid code lengths set")

    monkeypatch.setattr(hosted_mod.tarfile, "open", raises_zlib)
    assert _archive_opens_here(b"anything") is None


# -- signing in ---------------------------------------------------------------------


def test_device_code_and_token_speak_the_published_shapes():
    from openreynolds.backend.hosted import device_code, device_token

    seen = []

    def handler(request):
        seen.append((request.url.path, request.read()))
        if request.url.path == "/v1/device/code":
            return httpx.Response(200, json={"device_code": "dc", "user_code": "AB12-CD34", "interval": 5, "expires_in": 600})
        if len(seen) == 2:
            return httpx.Response(428, json={"error": "authorization_pending", "message": "not yet"})
        return httpx.Response(200, json={"api_key": "of_live_x", "key_id": "k", "name": "laptop"})

    transport = httpx.MockTransport(handler)
    offer = device_code("https://svc.example/", "laptop", transport=transport)
    assert offer["user_code"] == "AB12-CD34"
    assert b"laptop" in seen[0][1]
    assert device_token("https://svc.example", "dc", transport=transport) is None
    assert device_token("https://svc.example", "dc", transport=transport)["api_key"] == "of_live_x"


def test_a_refused_device_code_is_an_error_with_the_service_message():
    from openreynolds.backend.hosted import device_token

    transport = httpx.MockTransport(lambda request: httpx.Response(410, json={"error": "gone", "message": "already claimed"}))
    with pytest.raises(BackendError) as caught:
        device_token("https://svc.example", "dc", transport=transport)
    assert caught.value.code == "gone"


def test_password_sign_in_and_sign_up_against_the_identity_provider():
    from openreynolds.backend.hosted import auth_config, password_session, sign_up

    def handler(request):
        if request.url.path == "/dashboard/config.json":
            return httpx.Response(200, json={"supabase_url": "https://sb.example", "publishable_key": "pk"})
        assert request.headers["apikey"] == "pk"
        body = request.read()
        if request.url.path == "/auth/v1/token":
            assert request.url.params["grant_type"] == "password"
            if b"wrong" in body:
                return httpx.Response(400, json={"code": 400, "error_code": "invalid_credentials", "msg": "Invalid login credentials"})
            return httpx.Response(200, json={"access_token": "jwt", "user": {"email": "a@b.c"}})
        if request.url.path == "/auth/v1/signup":
            if b"taken" in body:
                return httpx.Response(422, json={"code": 422, "error_code": "user_already_exists", "msg": "User already registered"})
            if b"confirm" in body:
                return httpx.Response(200, json={"id": "u", "email": "a@b.c"})
            return httpx.Response(200, json={"access_token": "jwt-new"})
        return httpx.Response(404, json={"error": "not_found", "message": "?"})

    transport = httpx.MockTransport(handler)
    auth = auth_config("https://svc.example", transport=transport)
    assert auth["publishable_key"] == "pk"
    assert password_session("https://sb.example", "pk", "a@b.c", "right", transport=transport)["access_token"] == "jwt"
    with pytest.raises(BackendError) as refused:
        password_session("https://sb.example", "pk", "a@b.c", "wrong", transport=transport)
    assert refused.value.code == "invalid_credentials"
    assert sign_up("https://sb.example", "pk", "a@b.c", "pw", transport=transport)["access_token"] == "jwt-new"
    assert sign_up("https://sb.example", "pk", "a@b.c", "confirm-me", transport=transport) is None
    with pytest.raises(BackendError) as taken:
        sign_up("https://sb.example", "pk", "taken@b.c", "pw", transport=transport)
    assert taken.value.code == "user_already_exists"


def test_the_older_error_shape_is_also_read_as_bad_credentials():
    from openreynolds.backend.hosted import password_session

    transport = httpx.MockTransport(lambda r: httpx.Response(400, json={"error": "invalid_grant", "error_description": "Invalid login credentials"}))
    with pytest.raises(BackendError) as caught:
        password_session("https://sb.example", "pk", "a@b.c", "x", transport=transport)
    assert caught.value.code == "invalid_credentials"


def test_terms_and_mint_carry_the_session():
    from openreynolds.backend.hosted import accept_terms, mint_key

    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers.get("authorization"), request.read()))
        if request.url.path == "/v1/account/accept-terms":
            return httpx.Response(200, json={"user_id": "u", "tos_accepted_at": "now"})
        return httpx.Response(201, json={"key": "of_live_x", "prefix": "of_live_x", "key_id": "k", "name": "laptop"})

    transport = httpx.MockTransport(handler)
    accept_terms("https://svc.example", "jwt", transport=transport)
    assert mint_key("https://svc.example", "jwt", "laptop", transport=transport)["key"] == "of_live_x"
    assert all(auth == "Bearer jwt" for _, auth, _ in seen)
    assert b"laptop" in seen[1][2]


# -- a write is not repeated in the dark ----------------------------------------------
#
# From a 3D transient run whose captured transcript held one `job_check` reply three
# times, at three timestamps minutes apart, all under the same message `seq`. The agent
# produced that reply once -- `Store.append_message` hands out a seq once -- so the
# repeats were the client posting the same row again after the service had already
# written it, and the timestamps were insert times, not the agent's.


def test_a_write_is_not_repeated_when_its_answer_was_lost(monkeypatch):
    """A POST that reached the service and whose answer did not come back must not be
    sent again. The service cannot tell the repeat from a new message: `seq` comes from
    this client and the transcript table indexes it without making it unique."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    landed = []

    def commits_then_times_out(method, path, timeout=None, **kwargs):
        landed.append(path)
        raise httpx.ReadTimeout("the answer never came back")

    monkeypatch.setattr(client._client, "request", commits_then_times_out)
    with pytest.raises(BackendError) as excinfo:
        client.post_messages("study-1", [{"seq": 197, "role": "tool", "content": "x"}])

    assert excinfo.value.code == "timeout"
    assert len(landed) == 1, "the row was written once; asking again writes it again"


def test_an_ambiguous_server_error_does_not_repeat_a_write(monkeypatch):
    """500, 502 and 504 are retried on a read and not on a write: none of them says
    whether the work was done, and only the write leaves a mark if it was."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    sent = []

    def bad_gateway(method, path, timeout=None, **kwargs):
        sent.append(method)
        return response(502, {"error": "bad_gateway", "message": "upstream"})

    monkeypatch.setattr(client._client, "request", bad_gateway)
    with pytest.raises(BackendError):
        client.request("POST", "/v1/studies/s/messages", json=[])
    assert sent == ["POST"]

    sent.clear()
    with pytest.raises(BackendError):
        client.request("GET", "/v1/instances")
    assert len(sent) == 5, "a read is unchanged -- repeating it costs nothing"


def test_a_write_is_still_retried_when_the_service_declined_it(monkeypatch):
    """A 429 and a cold-start 503 are the service saying it did nothing, so repeating
    the call cannot repeat an effect. Losing these retries would mean losing every
    first call to a workspace that is still booting."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = []

    def booting(method, path, timeout=None, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            return response(503, {"error": "unavailable", "message": "booting"},
                            headers={"Retry-After": "0"})
        return response(201, {"inserted": 1})

    monkeypatch.setattr(client._client, "request", booting)
    client.post_messages("study-1", [{"seq": 1, "role": "user", "content": "hello"}])
    assert len(calls) == 3


def test_a_write_is_still_retried_when_the_connection_was_never_made(monkeypatch):
    """Nothing was handed over, so nothing can have happened twice."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = []

    def refused(method, path, timeout=None, **kwargs):
        calls.append(path)
        if len(calls) < 2:
            raise httpx.ConnectError("connection refused")
        return response(201, {"inserted": 1})

    monkeypatch.setattr(client._client, "request", refused)
    client.post_messages("study-1", [{"seq": 1, "role": "user", "content": "hello"}])
    assert len(calls) == 2


def test_a_tar_upload_says_it_may_be_repeated(monkeypatch):
    """The method is a rule of thumb, not the fact. Unpacking the same archive over
    the same directory twice leaves the workspace exactly as once does, so a tree
    transfer keeps its retries where a transcript row loses them."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = []

    def flaky(method, path, timeout=None, **kwargs):
        calls.append(path)
        if len(calls) < 2:
            raise httpx.ReadTimeout("the answer never came back")
        return response(200, {"ok": True})

    monkeypatch.setattr(client._client, "request", flaky)
    client.request("POST", "/v1/instances/i/tar", repeatable=True)
    assert len(calls) == 2


def test_rejected_service_credentials_are_not_retried(monkeypatch):
    """`modal_auth_failed` is a 500, and 500 is in the retry set for good reasons that
    have nothing to do with this one: the service's own Modal credentials have been
    rejected, so no later attempt can succeed until a person renews them.

    Without the code carve-out this cost four backoffs and then reported a permanent
    configuration fault as a transient failure -- which is the cost foamd's own carve-out
    was written to avoid, arriving by the other door."""
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append(path)
        return response(500, {"error": "modal_auth_failed",
                              "message": "this service's own Modal credentials were "
                                         "rejected; retrying will not clear it"})

    monkeypatch.setattr(client._client, "request", fake_request)
    with pytest.raises(BackendError) as caught:
        client.request("GET", "/v1/instances")

    assert len(calls) == 1, f"tried {len(calls)} times; no attempt can succeed"
    assert caught.value.code == "modal_auth_failed"
    assert "retrying will not clear it" in str(caught.value)


def test_an_ordinary_500_is_still_retried(monkeypatch):
    """The carve-out is one code, not a policy change. A bare 500 -- the sandbox being
    restarted underneath a read -- is the case `_RETRY_STATUSES` exists for, and it must
    keep its retries."""
    client = FoamdClient("https://example.invalid", "of_live_test")
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            return response(500, {"error": "http_error", "message": "internal"})
        return response(200, {"ok": True})

    monkeypatch.setattr(client._client, "request", fake_request)
    assert client.request("GET", "/v1/instances").json() == {"ok": True}
    assert len(calls) == 3


# -- who owns the workspace at teardown ---------------------------------------


class _Joining:
    """A service that lists one instance and answers the start route.

    The one fact `acquire` has to get right is whether THIS call brought the Sandbox
    up, because that is what decides whether the session stops it on the way out.
    """

    def __init__(self, listed_status, start_reply):
        self.listed_status = listed_status
        self.start_reply = start_reply
        self.closed = False

    def list_instances(self):
        return [{"id": "iid-1", "status": self.listed_status}]

    def create_instance(self):
        raise AssertionError("there was one to join")

    def start_instance(self, instance_id):
        return self.start_reply

    def close(self):
        self.closed = True


def _acquired(monkeypatch, listed_status, start_reply):
    client = _Joining(listed_status, start_reply)
    monkeypatch.setattr(hosted_mod, "FoamdClient", lambda *a, **k: client)
    backend, _client, _iid = hosted_mod.acquire("https://svc.example", "of_live_test")
    return backend


def test_the_start_route_decides_who_owns_the_workspace(monkeypatch):
    """The listing is read BEFORE the start call and a fresh row reads `stopped`, so
    two sessions listing within the same second both concluded they had started the
    workspace -- and the first to exit stopped it from under the other. The start
    route knows which of the two it did, so it is asked."""
    joined = _acquired(monkeypatch, "stopped", {"id": "iid-1", "status": "running",
                                                "started_new": False})
    assert joined.was_already_running is True, "it joined a Sandbox somebody else had"

    started = _acquired(monkeypatch, "stopped", {"id": "iid-1", "status": "running",
                                                 "started_new": True})
    assert started.was_already_running is False, "this call is the one that built it"


def test_a_service_that_does_not_say_leaves_the_old_answer_standing(monkeypatch):
    """`started_new` is additive, so an older deployment answers without it. There the
    pre-start listing is all there is, which is the behaviour this has always had."""
    running = _acquired(monkeypatch, "running", {"id": "iid-1", "status": "running"})
    assert running.was_already_running is True

    stopped = _acquired(monkeypatch, "stopped", {"id": "iid-1", "status": "running"})
    assert stopped.was_already_running is False
