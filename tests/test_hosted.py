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
