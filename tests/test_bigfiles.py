"""`openreynolds pull` fetches what a sync left for its size, from the workspace's copy.

A sync leaves any file over its one-file limit on the instance; a 9.55 GB result was
out of reach even when asked for by name (InviscidAI/ui#137)."""
from __future__ import annotations

import httpx2 as httpx

from openreynolds.bigfiles import fetch
from openreynolds.mirror import Skip

STUDY = "20261126-080344-5bc8"
BIG = f"/work/{STUDY}/run/big.dat"
DATA = b"B" * 10_000


def client(copy: dict[str, bytes], asked: list | None = None) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/workspace/links"):
            assert request.headers["authorization"] == "Bearer key"
            path = request.url.params["path"]
            if asked is not None:
                asked.append(path)
            if path not in copy:
                return httpx.Response(404, json={"detail": "no path"})
            return httpx.Response(200, json={"as_of": "2026-11-26T08:00:00+00:00", "files": [
                {"path": path, "size": len(copy[path]), "mtime": 1_700_000_000, "url": f"https://s3.example{path}"}]})
        assert request.url.host == "s3.example"
        return httpx.Response(200, content=copy[request.url.path])
    return httpx.Client(transport=httpx.MockTransport(handle))


def test_a_file_too_big_for_a_sync_is_fetched_from_the_copy(tmp_path):
    skips = [Skip(BIG, "9.6 GB, over the 500 MB limit for one file", len(DATA), too_big=True),
             Skip(f"/work/{STUDY}/processor0/U", "processor data", 5)]
    asked: list = []
    got = fetch("https://foamd", "key", STUDY, skips, tmp_path, http=client({BIG: DATA}, asked), say=lambda s: None)
    assert asked == [BIG], "only the files left for their size"
    landed = tmp_path / STUDY / "run" / "big.dat"
    assert landed.read_bytes() == DATA and got.pulled == [landed] and got.bytes_pulled == len(DATA)
    assert not list(tmp_path.rglob("*.part"))


def test_a_file_not_in_the_copy_yet_or_cut_short_is_reported_not_kept(tmp_path):
    skips = [Skip(BIG, "over the budget", len(DATA), too_big=True)]
    got = fetch("https://foamd", "key", STUDY, skips, tmp_path, http=client({}), say=lambda s: None)
    assert got.pulled == [] and "not in the workspace's copy yet" in got.failed[0]

    def short(request):
        if request.url.path.endswith("/workspace/links"):
            return httpx.Response(200, json={"files": [{"path": BIG, "size": 99, "url": "https://s3.example/x"}]})
        return httpx.Response(200, content=b"only some")
    got = fetch("https://foamd", "key", STUDY, skips, tmp_path,
                http=httpx.Client(transport=httpx.MockTransport(short)), say=lambda s: None)
    assert got.pulled == [] and "came of 99" in got.failed[0]
    assert not (tmp_path / STUDY / "run" / "big.dat").exists() and not list(tmp_path.rglob("*.part"))


def test_a_path_that_would_land_outside_the_study_is_refused(tmp_path):
    skips = [Skip("/work/../../etc/passwd", "over", 5, too_big=True)]
    got = fetch("https://foamd", "key", STUDY, skips, tmp_path, http=client({}), say=lambda s: None)
    assert got.pulled == [] and "outside the study" in got.failed[0]


def test_a_same_size_file_the_copy_rewrote_is_fetched_again(tmp_path):
    """On a fixed mesh every write of a field is the same size: size alone kept the old one."""
    skips = [Skip(BIG, "over", len(DATA), too_big=True)]
    fetch("https://foamd", "key", STUDY, skips, tmp_path, http=client({BIG: DATA}), say=lambda s: None)
    landed = tmp_path / STUDY / "run" / "big.dat"
    assert int(landed.stat().st_mtime) == 1_700_000_000, "stamped with the copy's write time"
    again = fetch("https://foamd", "key", STUDY, skips, tmp_path, http=client({BIG: DATA}), say=lambda s: None)
    assert again.current == [landed] and again.pulled == []
    import os
    os.utime(landed, (1_600_000_000, 1_600_000_000))           # an older write of the same size
    newer = b"C" * len(DATA)
    third = fetch("https://foamd", "key", STUDY, skips, tmp_path, http=client({BIG: newer}), say=lambda s: None)
    assert third.pulled == [landed] and landed.read_bytes() == newer


def test_the_service_being_down_is_not_called_a_file_not_copied_yet(tmp_path):
    skips = [Skip(BIG, "over", len(DATA), too_big=True)]
    down = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    got = fetch("https://foamd", "key", STUDY, skips, tmp_path, http=down, say=lambda s: None)
    assert got.failed == [f"{BIG}: the workspace service answered 503"]
