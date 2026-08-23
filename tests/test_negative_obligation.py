"""The one obligation v1 carries toward a future local backend, and it is negative:

nothing above the Backend protocol may assume HTTP, containers, or the hosted service.
Only `backend/hosted.py` and `capture.py` (which is platform plumbing by definition) are
allowed to know the contract exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "openreynolds"

ABOVE_THE_PROTOCOL = ["tools.py", "loop.py", "prompt.py", "store.py", "watch.py"]

FORBIDDEN = ["httpx", "http://", "https://", "/v1/", "modal", "foamd", "Bearer"]


@pytest.mark.parametrize("module", ABOVE_THE_PROTOCOL)
def test_module_knows_nothing_about_transport(module):
    source = (PACKAGE / module).read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token.lower() not in source.lower(), f"{module} references {token!r}"


@pytest.mark.parametrize("module", ABOVE_THE_PROTOCOL)
def test_module_imports_only_the_protocol(module):
    source = (PACKAGE / module).read_text(encoding="utf-8")
    assert "backend.hosted" not in source
    assert "from .backend import hosted" not in source


def test_protocol_layer_is_transport_free():
    source = (PACKAGE / "backend" / "base.py").read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token.lower() not in source.lower()
