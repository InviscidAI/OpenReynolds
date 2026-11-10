"""Ending a session takes too long (OpenReynolds#82).

Measured on prod over 2026-11-03..10, `close.capture_case` ran 40 s at the median and up
to ten minutes, and it ran before the workspace was stopped. It is no network at all:
`casebundle.build` gzipped every tier of the local mirror in full, at level 9, before
finding out the tier was over the 40 MB budget and throwing it away. On a 3.4 GB study
that was 133 of the bundle's 134 seconds.

Fix 1: a pack is abandoned the moment its compressed size passes the budget.
Fix 2: the workspace is put down before the bundle is built, which needs nothing from it.
"""

from __future__ import annotations

import io
import os
import tarfile

import pytest

from openreynolds import casebundle, cli

from test_refused_start import _session
from conftest import message, text_block


def _incompressible(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))


@pytest.fixture
def heavy(tmp_path):
    """A case whose logs alone are ten times the budget the tests use."""
    root = tmp_path / "files"
    (root / "case" / "system").mkdir(parents=True)
    (root / "case" / "system" / "controlDict").write_text("application simpleFoam;\n")
    for i in range(40):
        _incompressible(root / "case" / f"log.part{i:02d}", 256 * 1024)
    return root


def test_a_tier_past_the_budget_is_abandoned_not_compressed_in_full(heavy, monkeypatch):
    added: list[str] = []
    original = tarfile.TarFile.add

    def counting(self, name, arcname=None, **kw):
        added.append(arcname)
        return original(self, name, arcname=arcname, **kw)

    monkeypatch.setattr(tarfile.TarFile, "add", counting)
    blob, report = casebundle.build(heavy, budget_mb=1)

    logs = [a for a in added if a and "log." in a]
    # 40 logs of 256 KB against 1 MB: the pack gives up after about four of them.
    assert 0 < len(logs) < 10, logs
    assert report.included == ["definition"]
    assert "record" in report.skipped
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        assert "case/system/controlDict" in tar.getnames()


def test_an_abandoned_pack_raises_and_leaves_no_noise(heavy, capfd):
    members = [(p, p.name) for p in sorted((heavy / "case").glob("log.*"))]
    with pytest.raises(casebundle._TooBig):
        casebundle._pack(members, limit=512 * 1024)
    import gc
    gc.collect()
    assert "Exception ignored" not in capfd.readouterr().err


def test_a_pack_under_its_limit_is_a_whole_archive(heavy):
    members = [(p, p.name) for p in sorted((heavy / "case").glob("log.part0*"))]
    blob = casebundle._pack(members, limit=40 * 1024 * 1024)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        assert len(tar.getnames()) == 10


def test_the_first_tier_still_travels_past_the_budget(heavy):
    """Unchanged: the irreplaceable tier goes whatever its size, and says so."""
    for i in range(8):
        _incompressible(heavy / "case" / "system" / f"big{i}", 256 * 1024)
    blob, report = casebundle.build(heavy, budget_mb=1)
    assert report.included[0] == "definition"
    assert report.oversized
    assert blob is not None and len(blob) > 1024 * 1024


# -- the order of the close ----------------------------------------------------------

def _record_close(monkeypatch, order, close_down_raises=False):
    def close_down(backend, store, keep_alive=False):
        order.append("close_down")
        if close_down_raises:
            raise RuntimeError("the stop failed")

    def capture_case(capture, store, view):
        order.append("capture_case")

    monkeypatch.setattr(cli, "_close_down", close_down)
    monkeypatch.setattr(cli, "_capture_the_case", capture_case)


def test_the_workspace_is_put_down_before_the_case_is_bundled(monkeypatch, tmp_path):
    order: list[str] = []
    _record_close(monkeypatch, order)
    _session(monkeypatch, tmp_path, [message([text_block("done")])],
             one_shot="cavity", capture=True)
    assert order == ["close_down", "capture_case"]


def test_the_case_is_still_bundled_when_the_close_down_fails(monkeypatch, tmp_path):
    order: list[str] = []
    _record_close(monkeypatch, order, close_down_raises=True)
    with pytest.raises(RuntimeError, match="the stop failed"):
        _session(monkeypatch, tmp_path, [message([text_block("done")])],
                 one_shot="cavity", capture=True)
    assert order == ["close_down", "capture_case"]
