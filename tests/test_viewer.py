"""`openreynolds view`: a pulled study's pictures as one local page.

The page is built from the directory, not from the manifest, because the manifest
is written on the instance and most pulled studies on a laptop do not carry one.
Everything here is about what a browser is handed: relative links, no script, no
network, and a cap so a `*_frames/` directory does not bury the renders.
"""

from __future__ import annotations

import os
from pathlib import Path

from click.testing import CliRunner

from openreynolds import cli, viewer
from openreynolds.config import Config
from openreynolds.store import Store

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


def study(root: Path) -> Path:
    """A pulled study as `pull` leaves one: a flat `renders/` and a deep `files/`."""
    (root / "renders").mkdir(parents=True, exist_ok=True)
    (root / "files" / "case" / "renders_zoom").mkdir(parents=True, exist_ok=True)
    (root / ".reynolds").mkdir(exist_ok=True)
    (root / "renders" / "mesh_z.png").write_bytes(PNG)
    (root / "files" / "case" / "renders_zoom" / "U_500.png").write_bytes(PNG)
    (root / "files" / "case" / "REPORT.md").write_text("# it ran", encoding="utf-8")
    (root / "files" / "case" / "log.simpleFoam").write_text("Time = 1", encoding="utf-8")
    (root / ".reynolds" / "manifest.jsonl").write_text("{}", encoding="utf-8")
    return root


# -- what goes on the page -------------------------------------------------------


def test_the_pictures_are_found_without_a_manifest(tmp_path):
    """The point of the command. `toolbox/gallery.py` reads `.reynolds/manifest.jsonl`
    and prints "nothing has been recorded into it yet" without one; of the studies a
    laptop actually holds, almost none carry it."""
    pictures, reports = viewer.walk(study(tmp_path))

    assert sorted(p.name for p in pictures) == ["U_500.png", "mesh_z.png"]
    assert [r.name for r in reports] == ["REPORT.md"]


def test_the_studys_own_bookkeeping_is_not_a_result(tmp_path):
    """`.reynolds/` is how the study tracks itself, and a log is not a picture."""
    pictures, reports = viewer.walk(study(tmp_path))

    assert all(".reynolds" not in p.parts for p in pictures + reports)
    assert not any(r.name == "log.simpleFoam" for r in reports)


def test_every_rank_of_a_decomposed_case_is_skipped_not_just_the_first(tmp_path):
    """A decomposed case has as many `processorN` directories as it had cores. They
    are one rank's copy of the reconstructed case, and a skip list naming only
    `processor0` would walk the other seven."""
    for rank in range(4):
        where = tmp_path / "files" / "case" / f"processor{rank}"
        where.mkdir(parents=True)
        (where / "slice.png").write_bytes(PNG)
    (tmp_path / "files" / "case" / "processors4").mkdir()
    (tmp_path / "files" / "case" / "processors4" / "keep.png").write_bytes(PNG)

    pictures, _ = viewer.walk(tmp_path)

    assert [p.parent.name for p in pictures] == ["processors4"], pictures


def test_pictures_group_by_the_folder_that_wrote_them(tmp_path):
    """Without the manifest the directory is the only grouping there is -- and it is
    not a poor one: `renders_zoom` was named by whatever drew into it."""
    pictures, _ = viewer.walk(study(tmp_path))

    labels = {label for label, _, _ in viewer.sections(pictures, tmp_path)}

    assert labels == {"renders", "files/case/renders_zoom"}


def test_a_frames_directory_is_capped_and_says_how_many_it_left(tmp_path):
    """`animate.py` writes one PNG per write time. A page that lists all of them
    buries the one render somebody wanted, and a page that silently drops them is
    worse than one that says the number."""
    frames = tmp_path / "files" / "case" / "viv_frames"
    frames.mkdir(parents=True)
    for n in range(viewer.SECTION_LIMIT + 7):
        (frames / f"frame_{n:04d}.png").write_bytes(PNG)

    pictures, _ = viewer.walk(tmp_path)
    (_, shown, left), = viewer.sections(pictures, tmp_path)

    assert len(shown) == viewer.SECTION_LIMIT
    assert left == 7
    assert f"{left} more picture(s)" in viewer.render(
        tmp_path, tmp_path / viewer.PAGE_NAME, pictures, []
    )


# -- what the browser is handed --------------------------------------------------


def test_the_page_has_no_script_and_no_network_in_it(tmp_path):
    """The whole reason it works offline. There is no JavaScript dependency to pin
    or vendor because there is no JavaScript."""
    page, _, _ = viewer.build(study(tmp_path))
    text = page.read_text(encoding="utf-8")

    assert "<script" not in text
    assert "http://" not in text and "https://" not in text
    assert "<link" not in text  # no stylesheet, no font


def test_the_pictures_are_linked_not_embedded(tmp_path):
    """A 666 MB study's page is kilobytes, and `loading="lazy"` -- the browser's
    own, not a script's -- is what keeps it that way on screen. The page that
    embeds instead, so it can travel, is `toolbox/gallery.py`."""
    page, pictures, _ = viewer.build(study(tmp_path))
    text = page.read_text(encoding="utf-8")

    assert "data:image" not in text
    assert 'loading="lazy"' in text
    assert "renders/mesh_z.png" in text
    assert "files/case/renders_zoom/U_500.png" in text
    assert pictures == 2
    assert page.stat().st_size < 20_000


def test_a_page_written_elsewhere_still_points_at_the_pictures(tmp_path):
    """`--out` puts the page somewhere else; the links follow it rather than
    breaking."""
    root = study(tmp_path / "study")
    page = tmp_path / "elsewhere" / "p.html"

    viewer.build(root, out=page)

    assert "../study/renders/mesh_z.png" in page.read_text(encoding="utf-8")


def test_a_study_with_nothing_rendered_says_so_instead_of_being_blank(tmp_path):
    """An empty page with no explanation reads as a broken command."""
    (tmp_path / "files").mkdir()
    page, pictures, reports = viewer.build(tmp_path)

    assert (pictures, reports) == (0, 0)
    assert "openreynolds pull" in page.read_text(encoding="utf-8")


def test_a_picture_that_vanished_mid_walk_does_not_take_the_page_down(tmp_path):
    """A mirror is writing into this directory while the page is built."""
    pictures = [tmp_path / "gone.png"]

    text = viewer.render(tmp_path, tmp_path / viewer.PAGE_NAME, pictures, [])

    assert "gone.png" in text and "0 B" in text


# -- the command -----------------------------------------------------------------


def test_view_builds_the_page_for_the_newest_study_and_does_not_open_it(
    tmp_path, monkeypatch
):
    root = tmp_path / "studies"
    store = Store(root, "20260823-120000-abcd")
    store.save()  # a study is one `list_studies` can see once session.json is on disk
    study(store.dir)
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

    result = CliRunner().invoke(cli.main, ["view", "--no-open"])

    assert result.exit_code == 0, result.output
    assert (store.dir / viewer.PAGE_NAME).is_file()
    assert "2 picture(s)" in result.output
    assert opened == [], "--no-open means no browser"


def test_view_opens_the_page_as_a_file_url(tmp_path, monkeypatch):
    """`file://`, not a path and not a port: there is no server behind this page."""
    root = tmp_path / "studies"
    store = Store(root, "20260823-120000-abcd")
    store.save()  # a study is one `list_studies` can see once session.json is on disk
    study(store.dir)
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

    result = CliRunner().invoke(cli.main, ["view"])

    assert result.exit_code == 0, result.output
    assert len(opened) == 1 and opened[0].startswith("file://")
    assert opened[0].endswith(viewer.PAGE_NAME)


def test_view_with_no_studies_says_where_it_looked(tmp_path, monkeypatch):
    root = tmp_path / "studies"
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))

    result = CliRunner().invoke(cli.main, ["view"])

    assert result.exit_code == 1
    assert str(root) in result.output


def test_view_starts_no_instance_and_needs_no_token(tmp_path, monkeypatch):
    """It reads files that are already here. A picture viewer that reserved a
    workspace would bill for a look at a finished study -- which is also why this is
    not a route on `openreynolds serve`, whose host can start compute."""
    root = tmp_path / "studies"
    store = Store(root, "20260823-120000-abcd")
    store.save()  # a study is one `list_studies` can see once session.json is on disk
    study(store.dir)
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: True)
    reached = []
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: reached.append(a))

    result = CliRunner().invoke(cli.main, ["view", "--no-open"])

    assert result.exit_code == 0, result.output
    assert reached == [], "view reached for a workspace"


def test_out_puts_the_page_where_it_was_asked_for(tmp_path, monkeypatch):
    root = tmp_path / "studies"
    store = Store(root, "20260823-120000-abcd")
    store.save()  # a study is one `list_studies` can see once session.json is on disk
    study(store.dir)
    where = tmp_path / "pages" / "look.html"
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))

    result = CliRunner().invoke(
        cli.main, ["view", "--out", os.fspath(where), "--no-open"]
    )

    assert result.exit_code == 0, result.output
    assert where.is_file()
    assert not (store.dir / viewer.PAGE_NAME).exists()
