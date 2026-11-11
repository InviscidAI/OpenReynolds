"""The record travels in part rather than not at all (OpenReynolds#94).

`record` went in or out whole, so one 190 MB solver log left out every log of a 3.4 GB
study while its mesh, which `Allmesh` rebuilds, went in. Now a long log or table keeps
its head and tail, and the sample surfaces in `postProcessing/` are their own tier,
last, newest first.
"""

from __future__ import annotations

import io
import json
import os
import random
import tarfile

from openreynolds import casebundle


def _write(path, data: bytes, mtime: float | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _log(lines: int, seed: int = 0) -> bytes:
    """A solver log that compresses about as badly as a real one does well: each line
    carries random residuals, so it cannot be squeezed to nothing."""
    rnd = random.Random(seed)
    out = ["Create time\n", "Starting time loop\n"]
    for i in range(lines):
        out.append(f"Time = {i}  Ux residual {rnd.random():.12e} p {rnd.random():.12e}\n")
    out.append("End\n")
    return "".join(out).encode()


def _case(root):
    _write(root / "c" / "system" / "controlDict", b"application pimpleFoam;\n")
    return root


def _tar(blob):
    return tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")


def _read(blob, name) -> bytes:
    with _tar(blob) as tar:
        return tar.extractfile(name).read()


def _manifest(blob) -> dict:
    return json.loads(_read(blob, "MANIFEST.json"))


def test_a_long_log_travels_as_its_head_and_tail(tmp_path):
    root = _case(tmp_path)
    body = _log(200_000)  # about 13 MB, several times the budget once compressed
    _write(root / "c" / "log.pimpleFoam", body)
    blob, report = casebundle.build(root, budget_mb=1)

    assert "record" in report.included and "record" in report.partial
    kept = _read(blob, "c/log.pimpleFoam")
    assert kept.startswith(b"Create time\nStarting time loop\n")
    assert kept.endswith(b"End\n")
    assert b"bytes of this file were left out here" in kept
    assert len(kept) < casebundle.CUT_HEAD + casebundle.CUT_TAIL + 200
    # Whole lines either side of the cut.
    for line in kept.splitlines():
        assert line.startswith((b"Time = ", b"#", b"Create", b"Starting", b"End")), line

    body_m = _manifest(blob)
    assert body_m["cut"] == {"c/log.pimpleFoam": len(body)}
    assert body_m["complete"] is False
    assert "long file(s) kept as their first 64 KB and last 1 MB" in " ".join(report.brief())


def test_a_record_that_fits_whole_is_not_cut(tmp_path):
    root = _case(tmp_path)
    body = _log(200_000)
    _write(root / "c" / "log.pimpleFoam", body)
    blob, report = casebundle.build(root, budget_mb=40)
    assert _read(blob, "c/log.pimpleFoam") == body
    assert report.partial == {} and _manifest(blob)["complete"] is True


def test_the_record_goes_in_before_the_mesh(tmp_path):
    """The study in #94: one huge log, and a mesh. The mesh is what goes."""
    root = _case(tmp_path)
    _write(root / "c" / "log.pimpleFoam", _log(400_000))
    _write(root / "c" / "constant" / "polyMesh" / "points", os.urandom(3 * 1024 * 1024))
    blob, report = casebundle.build(root, budget_mb=2)
    assert "record" in report.included
    assert "mesh" in report.skipped


def test_a_table_keeps_whole_rows(tmp_path):
    root = _case(tmp_path)
    rows = [b"# Time Fx Fy Fz\n"] + [f"{i} {i*0.1:.9f} {i*0.2:.9f} {i*0.3:.9f}\n".encode()
                                     for i in range(300_000)]
    _write(root / "c" / "postProcessing" / "forces" / "0" / "force.dat", b"".join(rows))
    blob, report = casebundle.build(root, budget_mb=1)
    kept = _read(blob, "c/postProcessing/forces/0/force.dat")
    data = [l for l in kept.splitlines() if not l.startswith(b"#")]
    assert data and all(len(l.split()) == 4 for l in data)
    assert data[-1].startswith(b"299999 ")


def test_postprocessing_tables_are_record_and_surfaces_are_samples(tmp_path):
    root = _case(tmp_path)
    latest = casebundle._latest_time_dirs(root)
    pp = root / "c" / "postProcessing"
    cases = {
        pp / "probes" / "0" / "p": "record",
        pp / "forces" / "0" / "force.dat": "record",
        pp / "forces" / "forces_physical.csv": "record",
        pp / "planes" / "2.36" / "mid.vtp": "samples",
        pp / "clipPlane" / "0.35" / "clip.vtp.gz": "samples",
    }
    for path, tier in cases.items():
        assert casebundle.classify(path, root, latest) == tier, path


def test_samples_go_last_and_newest_first(tmp_path):
    root = _case(tmp_path)
    _write(root / "c" / "log.pimpleFoam", _log(100))
    for i in range(20):
        _write(root / "c" / "postProcessing" / "planes" / f"{i}" / "mid.vtp",
               os.urandom(256 * 1024), mtime=1_700_000_000 + i)
    blob, report = casebundle.build(root, budget_mb=1)

    assert report.included[-1] == "samples" and "samples" in report.partial
    with _tar(blob) as tar:
        times = sorted(int(n.split("/")[3]) for n in tar.getnames() if n.endswith(".vtp"))
    assert times and times == list(range(20 - len(times), 20)), times
    assert "(the oldest)" in report.partial["samples"]
    assert "c/log.pimpleFoam" in _tar(blob).getnames()


def test_newest_samples_are_by_simulation_time_not_by_when_they_were_pulled(tmp_path):
    """The mirror writes each file fresh, so mtime is the pull order. Here it is the
    reverse of the simulation's: the oldest time was pulled last."""
    root = _case(tmp_path)
    for i in range(20):
        _write(root / "c" / "postProcessing" / "planes" / f"{i * 0.5:g}" / "mid.vtp",
               os.urandom(256 * 1024), mtime=1_700_000_000 - i)
    blob, report = casebundle.build(root, budget_mb=1)
    with _tar(blob) as tar:
        times = sorted(float(n.split("/")[3]) for n in tar.getnames() if n.endswith(".vtp"))
    assert times and times[-1] == 9.5 and times == [t * 0.5 for t in range(20 - len(times), 20)]


def test_a_record_too_big_on_its_own_is_cut_not_sent_oversized(tmp_path):
    """Only `definition` may go past the budget; a tier that can be cut is cut."""
    root = tmp_path  # no definition at all: the record is the first tier
    _write(root / "c" / "log.pimpleFoam", _log(200_000))
    blob, report = casebundle.build(root, budget_mb=1)
    assert report.included == ["record"]
    assert not report.oversized
    assert len(blob) <= 1024 * 1024


def test_the_bundle_summary_reaches_the_runner_log(monkeypatch, tmp_path, capsys):
    """With OPENREYNOLDS_TIMING on (the hosted runner), one `[bundle]` JSON line goes to
    stderr beside the `[timing]` marks, so prod coverage can be counted."""
    from openreynolds import cli
    from test_refused_start import _session
    from conftest import message, text_block

    monkeypatch.setattr(cli, "_TIMING", True)
    _session(monkeypatch, tmp_path, [message([text_block("done")])],
             one_shot="cavity", capture=True)
    lines = [l for l in capsys.readouterr().err.splitlines() if l.startswith("[bundle] ")]
    assert len(lines) == 1, lines
    said = json.loads(lines[0][len("[bundle] "):])
    assert {"study_id", "tiers", "skipped", "partial", "cut", "files", "bytes"} <= set(said)


def test_no_bundle_line_without_the_timing_switch(monkeypatch, tmp_path, capsys):
    from openreynolds import cli
    from test_refused_start import _session
    from conftest import message, text_block

    monkeypatch.setattr(cli, "_TIMING", False)
    _session(monkeypatch, tmp_path, [message([text_block("done")])],
             one_shot="cavity", capture=True)
    assert "[bundle]" not in capsys.readouterr().err
