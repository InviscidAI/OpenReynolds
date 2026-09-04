"""The transport details the demo scripts skipped: retries and both error shapes."""

from __future__ import annotations

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


def _client_against(transport: httpx.MockTransport, *, follow_redirects: bool) -> FoamdClient:
    client = FoamdClient("https://svc.example", "of_live_test")
    client._client = httpx.Client(
        base_url="https://svc.example",
        transport=transport,
        follow_redirects=follow_redirects,
    )
    return client


def test_an_unfollowed_redirect_is_named_not_a_json_decode_failure():
    """The whole bug, as the model saw it: `bad_response (303): the body was empty`.
    True, and useless -- it named the symptom and hid the one thing to act on."""
    from openreynolds.backend.hosted import HostedBackend

    client = _client_against(_edge_with_a_150s_redirect({}), follow_redirects=False)
    backend = HostedBackend(client, "inst-1")

    with pytest.raises(BackendError) as excinfo:
        backend.exec("sleep 200", timeout_s=250)

    assert excinfo.value.code == "redirect_not_followed", (
        "not bad_response -- the JSON decoder must not be where this lands"
    )
    assert excinfo.value.status == 303
    assert "follow_redirects" in excinfo.value.message
    assert "/v1/exec-result" in excinfo.value.message


def test_a_long_exec_completes_through_the_redirect():
    """A 250 s exec is under `EXEC_MAX_TIMEOUT_S` and over the edge's window. Followed,
    the redirect carries the real result: measured live, `sleep 200` returns rc=0."""
    from openreynolds.backend.hosted import HostedBackend

    transport = _edge_with_a_150s_redirect(
        {"exit_code": 0, "output": "done", "truncated": False}
    )
    backend = HostedBackend(_client_against(transport, follow_redirects=True), "inst-1")

    result = backend.exec("sleep 200", timeout_s=250)

    assert result.exit_code == 0
    assert result.output == "done"


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
