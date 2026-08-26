"""The transport details the demo scripts skipped: retries and both error shapes."""

from __future__ import annotations

import httpx2 as httpx
import pytest

from openreynolds.backend.base import BackendError
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
    assert _retry_delay(response(500, {}), 0) == 1.0
    assert _retry_delay(response(500, {}), 3) == 8.0
    assert _retry_delay(None, 10) == 30.0


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
