"""Delivery: the mirror surfaces and assembles pictures, the agent never fetches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openreynolds import video
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


def test_the_frames_say_what_they_were_rendered_for(tmp_path):
    """`animate.py --format webp --fps 24` renders the frames on the instance and
    writes `frames.json` beside them; the frames mirror home and, before this, the
    intent did not. Every animation came out a 10 fps gif whatever was asked for,
    and `webp` was unreachable by any path."""
    frames_dir = tmp_path / "wake_frames"
    frames_dir.mkdir()
    (frames_dir / "frames.json").write_text(
        '{"version": 1, "container": "webp", "fps": 24.0, "output": "wake_frames.webp"}',
        encoding="utf-8",
    )

    assert video.intent(frames_dir) == ("wake_frames.webp", 24.0)


def test_a_directory_with_no_sidecar_keeps_the_old_answer(tmp_path):
    """Frames somebody assembled by hand have no intent to read, and that is not an
    error -- it is the fallback that has always been there."""
    plain = tmp_path / "frames"
    plain.mkdir()
    assert video.intent(plain) == ("", video.DEFAULT_FPS)


def test_a_sidecar_that_will_not_parse_is_not_a_reason_to_lose_the_animation(tmp_path):
    broken = tmp_path / "frames"
    broken.mkdir()
    (broken / "frames.json").write_text("{not json", encoding="utf-8")
    assert video.intent(broken) == ("", video.DEFAULT_FPS)


def test_a_container_nobody_can_open_is_ignored_rather_than_trusted(tmp_path):
    """`assemble` dispatches on the extension: inventing one from a typo would
    produce a file nothing plays."""
    odd = tmp_path / "frames"
    odd.mkdir()
    (odd / "frames.json").write_text(
        '{"container": "mkv", "fps": 30.0, "output": "frames.mkv"}', encoding="utf-8"
    )
    name, fps = video.intent(odd)
    assert name == "" and fps == 30.0


def test_delivery_assembles_what_the_sidecar_asked_for(tmp_path):
    """The whole point of the sidecar: the mirror carries the frames home and the
    harness makes the animation the study asked for, not the one it defaults to."""
    g = gallery(tmp_path)
    frames_dir = tmp_path / "files" / "wake_frames"
    frames_dir.mkdir(parents=True)
    seq = []
    for n in range(4):
        frame = frames_dir / f"frame_{n:04d}.png"
        frame.write_bytes(b"\x89PNG")
        seq.append(frame)
    (frames_dir / "frames.json").write_text(
        '{"container": "mp4", "fps": 24.0, "output": "wake_frames.mp4"}', encoding="utf-8"
    )

    event = g.ingest(report(seq))

    assert [out.name for _frames, out in g._calls] == ["wake_frames.mp4"]
    assert any(path.name == "wake_frames.mp4" for path in event.videos)


# --- renders reach the platform, not just the renders folder --------------------


class _Keeper:
    """Stands in for Capture: records what delivery handed it."""

    def __init__(self, boom=False):
        self.kept: list[tuple[str, str]] = []
        self.boom = boom

    def artifact(self, path, kind=None):
        if self.boom:
            raise RuntimeError("platform is down")
        self.kept.append((path.name, kind))


def test_a_surfaced_render_is_also_kept(tmp_path):
    """Before this, a render reached the platform only if the model happened to call
    `fetch` on it -- and delivery exists so that it need not, so nothing was kept and
    a study looked at from anywhere else had no pictures."""
    files, renders = tmp_path / "files", tmp_path / "renders"
    (files / "case").mkdir(parents=True)
    src = files / "case" / "mesh.png"
    src.write_bytes(b"\x89PNG\r\n")
    keeper = _Keeper()
    gallery = Gallery(files, renders, capture=keeper)

    class R:
        pulled = [src]

    event = gallery.ingest(R())
    assert [p.name for p in event.images] == ["mesh.png"]
    assert keeper.kept == [("mesh.png", "render")]


def test_keeping_a_render_cannot_break_delivery(tmp_path):
    """A convenience may not end a session: the picture still lands locally."""
    files, renders = tmp_path / "files", tmp_path / "renders"
    (files / "case").mkdir(parents=True)
    src = files / "case" / "mesh.png"
    src.write_bytes(b"\x89PNG\r\n")
    gallery = Gallery(files, renders, capture=_Keeper(boom=True))

    class R:
        pulled = [src]

    event = gallery.ingest(R())
    assert [p.name for p in event.images] == ["mesh.png"]
    assert (renders / "mesh.png").is_file()


def test_no_capture_is_simply_no_keeping(tmp_path):
    files, renders = tmp_path / "files", tmp_path / "renders"
    (files / "case").mkdir(parents=True)
    src = files / "case" / "mesh.png"
    src.write_bytes(b"\x89PNG\r\n")

    class R:
        pulled = [src]

    event = Gallery(files, renders).ingest(R())
    assert [p.name for p in event.images] == ["mesh.png"]
