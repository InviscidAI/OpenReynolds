"""Delivery: the mirror surfaces and assembles pictures, the agent never fetches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openreynolds.delivery import Gallery


def report(paths):
    return SimpleNamespace(pulled=[Path(p) for p in paths])


def gallery(tmp_path, assembled=None):
    files = tmp_path / "files"
    renders = tmp_path / "renders"
    calls = []

    def fake_assemble(frames, out, **kw):
        out.write_bytes(b"GIF89a")
        calls.append((tuple(str(f) for f in frames), out))

    g = Gallery(files, renders, assemble=fake_assemble, encoder=lambda: "ffmpeg")
    g._calls = calls
    if assembled is not None:
        assembled.append(g)
    return g


def make(tmp_path, rel, data=b"\x89PNG"):
    p = tmp_path / "files" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# -- surfacing stills ----------------------------------------------------------


def test_a_new_render_is_copied_into_the_flat_folder(tmp_path):
    g = gallery(tmp_path)
    src = make(tmp_path, "study/case/renders/p.png")

    event = g.ingest(report([src]))

    assert event.images == [tmp_path / "renders" / "p.png"]
    assert (tmp_path / "renders" / "p.png").read_bytes() == b"\x89PNG"
    assert "1 new render" in " ".join(event.lines())


def test_non_images_are_ignored(tmp_path):
    g = gallery(tmp_path)
    log = make(tmp_path, "study/case/log.simpleFoam", b"Time = 1")

    assert not g.ingest(report([log]))


def test_the_same_render_is_surfaced_once(tmp_path):
    g = gallery(tmp_path)
    src = make(tmp_path, "study/case/renders/p.png")

    assert g.ingest(report([src]))
    assert not g.ingest(report([src])), "already delivered; not again"


def test_a_name_collision_keeps_both_named_by_case(tmp_path):
    g = gallery(tmp_path)
    a = make(tmp_path, "cyl_Re20/p.png", b"A")
    b = make(tmp_path, "cyl_Re40/p.png", b"BB")

    g.ingest(report([a]))
    g.ingest(report([b]))

    assert (tmp_path / "renders" / "p.png").read_bytes() == b"A"
    assert (tmp_path / "renders" / "cyl_Re40_p.png").read_bytes() == b"BB"


# -- assembling animations -----------------------------------------------------


def frames(tmp_path, folder, n, prefix="shed_"):
    made = []
    for i in range(n):
        made.append(make(tmp_path, f"{folder}/{prefix}{i:03d}.png", b"\x89PNG" + bytes([i])))
    return made


def test_a_numbered_sequence_is_assembled_into_a_gif(tmp_path):
    g = gallery(tmp_path)
    seq = frames(tmp_path, "case/frames", 6)

    event = g.ingest(report(seq))

    assert event.videos == [tmp_path / "renders" / "frames.gif"]
    assert (tmp_path / "renders" / "frames.gif").exists()
    assert "assembled frames.gif" in " ".join(event.lines())
    # The frames themselves are not also dumped as loose stills.
    assert event.images == []


def test_a_gallery_of_stills_is_not_assembled(tmp_path):
    """mesh.png + p.png + U.png in a renders/ dir is not an animation."""
    g = gallery(tmp_path)
    stills = [
        make(tmp_path, "case/renders/mesh.png"),
        make(tmp_path, "case/renders/p.png"),
        make(tmp_path, "case/renders/U.png"),
    ]

    event = g.ingest(report(stills))

    assert event.videos == []
    assert len(event.images) == 3


def test_a_dir_named_anim_assembles_even_with_few_frames(tmp_path):
    g = gallery(tmp_path)
    seq = [make(tmp_path, "case/anim/a.png"), make(tmp_path, "case/anim/b.png")]

    event = g.ingest(report(seq))

    assert event.videos == [tmp_path / "renders" / "anim.gif"]


def test_a_growing_sequence_reassembles_only_after_it_grows(tmp_path):
    g = gallery(tmp_path)
    first = frames(tmp_path, "case/frames", 6)
    assert g.ingest(report(first)).videos  # assembled at 6

    # One more frame: not enough growth yet.
    seventh = frames(tmp_path, "case/frames", 7)[-1:]
    assert not g.ingest(report(seventh)).videos

    # Several more: re-assembled, now fuller (the "partial gif so far" case).
    more = frames(tmp_path, "case/frames", 12)[7:]
    ev = g.ingest(report(more))
    assert ev.videos == [tmp_path / "renders" / "frames.gif"]
    assert len(g._calls[-1][0]) == 12


def test_no_encoder_means_no_assembly_but_stills_still_surface(tmp_path):
    files, renders = tmp_path / "files", tmp_path / "renders"
    g = Gallery(files, renders, assemble=lambda *a, **k: None, encoder=lambda: None)
    seq = frames(tmp_path, "case/frames", 6)

    event = g.ingest(report(seq))

    assert event.videos == []  # nothing to encode with; `openreynolds video` explains


def test_a_bad_frame_does_not_crash_the_cycle(tmp_path):
    def boom(frames, out, **kw):
        from openreynolds.video import VideoError

        raise VideoError("corrupt frame")

    g = Gallery(tmp_path / "files", tmp_path / "renders", assemble=boom, encoder=lambda: "ffmpeg")
    seq = frames(tmp_path, "case/frames", 6)

    event = g.ingest(report(seq))  # must not raise

    assert event.videos == []


def test_newest_lists_the_folder_newest_first(tmp_path):
    g = gallery(tmp_path)
    import os, time

    g.ingest(report([make(tmp_path, "case/renders/a.png")]))
    b = tmp_path / "renders" / "b.png"
    b.write_bytes(b"\x89PNG")
    os.utime(b, (time.time() + 10, time.time() + 10))

    newest = g.newest()
    assert newest[0].name == "b.png"
