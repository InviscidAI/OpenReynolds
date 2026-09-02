"""The artifact gallery.

Everything it does is ordinary Python -- it reads the manifest, groups it, embeds
files as base64 and lays out a page -- so all of it is testable here. The two
properties worth guarding are that the page carries its pictures rather than
referencing them, and that a manifest row whose file has been deleted is reported
rather than crashed on.
"""

from __future__ import annotations

import base64
import importlib.util
import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gallery():
    return load("gallery")


@pytest.fixture
def state():
    return load("study_state")


# -- a study on disk, shaped like the real one ---------------------------------


def make_png(path: Path, colour=(0.9, 0.2, 0.2)) -> Path:
    """A real, small, readable-back PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((8, 8, 3))
    data[:, :] = colour
    plt.imsave(str(path), data)
    return path


@pytest.fixture
def study(tmp_path, state):
    """A study directory with a manifest and some artifacts in it."""
    root = tmp_path / "wing"
    (root / ".reynolds").mkdir(parents=True)
    return root


def record(state, root: Path, kind: str, rel: str, **kwargs):
    return state.record(kind, root / rel, root=root, **kwargs)


# -- reading the manifest ------------------------------------------------------


def test_collect_resolves_each_row_to_a_file_on_disk(gallery, state, study):
    make_png(study / "renders" / "mesh.png")
    record(state, study, "mesh-full", "renders/mesh.png", case="wing", label="the mesh")

    entries = gallery.collect(study)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "mesh-full"
    assert entry.exists
    assert entry.case == "wing"
    assert entry.label == "the mesh"
    assert entry.path == study / "renders" / "mesh.png"
    assert entry.rel == "renders/mesh.png"
    assert entry.size > 0


def test_a_row_whose_file_was_deleted_is_marked_not_dropped(gallery, state, study):
    make_png(study / "renders" / "mesh.png")
    record(state, study, "mesh-full", "renders/mesh.png")
    record(state, study, "velocity", "renders/gone.png", label="deleted since")

    entries = gallery.collect(study)

    assert [entry.exists for entry in entries] == [True, False]
    assert entries[1].size == 0


def test_rows_the_gallery_wrote_itself_are_skipped(gallery, state, study):
    """Otherwise every run adds two rows and the next gallery is mostly galleries."""
    make_png(study / "renders" / "mesh.png")
    record(state, study, "mesh-full", "renders/mesh.png")
    record(state, study, "gallery", "gallery.html", source=gallery.SOURCE_TAG)
    record(state, study, "contact-sheet", "contact_sheet.png", source=gallery.SOURCE_TAG)

    assert [entry.kind for entry in gallery.collect(study)] == ["mesh-full"]


def test_a_contact_sheet_from_another_script_is_kept(gallery, state, study):
    """Only this script's own output is filtered; first_look's sheet is an artifact."""
    make_png(study / "first_look.png")
    record(state, study, "contact-sheet", "first_look.png", label="first look")

    assert [entry.kind for entry in gallery.collect(study)] == ["contact-sheet"]


def test_collect_can_be_narrowed_to_one_case(gallery, state, study):
    make_png(study / "a.png")
    make_png(study / "b.png")
    record(state, study, "mesh-full", "a.png", case="coarse")
    record(state, study, "mesh-full", "b.png", case="fine")

    assert [entry.rel for entry in gallery.collect(study, case="fine")] == ["b.png"]


# -- grouping and picking ------------------------------------------------------


def test_groups_come_out_in_pipeline_order(gallery, state, study):
    for kind, name in (("report", "r.md"), ("velocity", "v.png"), ("mesh-full", "m.png")):
        (study / name).write_text("x")
        record(state, study, kind, name)

    kinds = [kind for kind, _group in gallery.group_by_kind(gallery.collect(study))]

    assert kinds == ["mesh-full", "velocity", "report"]


def test_a_kind_the_manifest_does_not_know_sorts_last(gallery, state, study):
    for kind, name in (("wild-idea", "w.png"), ("mesh-full", "m.png")):
        (study / name).write_text("x")
        record(state, study, kind, name)

    kinds = [kind for kind, _group in gallery.group_by_kind(gallery.collect(study))]

    assert kinds == ["mesh-full", "wild-idea"]
    assert gallery.kind_order("wild-idea") >= len(state.KINDS)


def test_a_group_keeps_the_manifest_order(gallery, state, study):
    for name in ("one.png", "two.png", "three.png"):
        make_png(study / name)
        record(state, study, "mesh-full", name)

    (_kind, group), = gallery.group_by_kind(gallery.collect(study))

    assert [entry.rel for entry in group] == ["one.png", "two.png", "three.png"]


def test_final_is_the_newest_of_each_kind(gallery, state, study):
    for name in ("old.png", "new.png"):
        make_png(study / name)
        record(state, study, "mesh-full", name)
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")

    finals = gallery.final_entries(gallery.collect(study))

    assert [(kind, entry.rel) for kind, entry in finals] == [
        ("mesh-full", "new.png"),
        ("velocity", "u.png"),
    ]


def test_final_falls_back_to_the_newest_that_still_exists(gallery, state, study):
    make_png(study / "kept.png")
    record(state, study, "mesh-full", "kept.png")
    record(state, study, "mesh-full", "deleted.png")

    finals = gallery.final_entries(gallery.collect(study))

    assert [entry.rel for _kind, entry in finals] == ["kept.png"]


def test_a_kind_with_nothing_left_on_disk_drops_out_of_final(gallery, state, study):
    record(state, study, "mesh-full", "deleted.png")

    assert gallery.final_entries(gallery.collect(study)) == []
    assert gallery.newest(gallery.collect(study)) is None


# -- what stands for an entry --------------------------------------------------


def test_a_frames_directory_is_shown_by_its_first_frame(gallery, state, study):
    frames = study / "wake_frames"
    make_png(frames / "frame_0002.png")
    make_png(frames / "frame_0001.png")
    record(state, study, "animation", "wake_frames", label="vorticity")

    entry, = gallery.collect(study)

    assert entry.is_dir and entry.frames == 2
    assert gallery.image_for(entry) == frames / "frame_0001.png"


def test_a_text_artifact_has_no_picture(gallery, state, study):
    (study / "report.md").write_text("# what happened\n")
    record(state, study, "report", "report.md")

    entry, = gallery.collect(study)

    assert entry.exists
    assert gallery.image_for(entry) is None


def test_a_missing_file_has_no_picture(gallery, state, study):
    record(state, study, "velocity", "gone.png")
    entry, = gallery.collect(study)
    assert gallery.image_for(entry) is None


def test_svg_belongs_on_the_page_but_not_on_the_sheet(gallery, state, study):
    (study / "plot.svg").write_text("<svg></svg>")
    record(state, study, "residuals", "plot.svg")

    entry, = gallery.collect(study)

    assert gallery.image_for(entry) is not None
    assert gallery.image_for(entry, gallery.RASTER_SUFFIXES) is None


# -- embedding -----------------------------------------------------------------


def test_data_uri_round_trips_the_bytes(gallery, study):
    path = make_png(study / "renders" / "mesh.png")

    uri = gallery.data_uri(path)

    assert uri.startswith("data:image/png;base64,")
    payload = base64.b64decode(uri.split(",", 1)[1])
    assert payload == path.read_bytes()


def test_data_uri_declines_a_file_over_the_limit(gallery, study):
    path = make_png(study / "big.png")
    assert gallery.data_uri(path, limit=10) == ""


def test_data_uri_declines_a_file_that_is_not_there(gallery, study):
    assert gallery.data_uri(study / "never.png") == ""


# -- the page ------------------------------------------------------------------


def built_page(gallery, root: Path, **kwargs) -> str:
    entries = gallery.collect(root)
    return gallery.html_document(
        gallery.group_by_kind(entries), finals=gallery.final_entries(entries), **kwargs
    )


def test_the_page_fetches_nothing(gallery, state, study):
    make_png(study / "renders" / "mesh.png")
    record(state, study, "mesh-full", "renders/mesh.png", label="the mesh")
    (study / "report.md").write_text("x")
    record(state, study, "report", "report.md")

    page = built_page(gallery, study)

    assert "http://" not in page
    assert "https://" not in page
    assert "//fonts" not in page
    assert not re.search(r"<script", page, re.IGNORECASE)
    assert not re.search(r"<link\b", page, re.IGNORECASE)
    # The only URIs in it are the ones it carries.
    for match in re.findall(r'src="([^"]*)"', page):
        assert match.startswith("data:")


def test_the_page_carries_the_image_bytes(gallery, state, study):
    path = make_png(study / "renders" / "mesh.png")
    record(state, study, "mesh-full", "renders/mesh.png")

    page = built_page(gallery, study)

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    assert encoded in page


def test_the_page_says_which_file_is_missing(gallery, state, study):
    record(state, study, "velocity", "renders/gone.png", label="the wake")

    page = built_page(gallery, study)

    assert "file missing" in page
    assert "gone.png" in page


def test_the_page_heads_each_kind_and_counts_it(gallery, state, study):
    for name in ("a.png", "b.png"):
        make_png(study / name)
        record(state, study, "mesh-full", name)
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")

    page = built_page(gallery, study)

    assert ">mesh-full<" in page
    assert ">velocity<" in page
    assert page.index(">mesh-full<") < page.index(">velocity<")
    # The newest of a kind with more than one entry is tagged as such. Asserted on
    # the tag itself rather than on the word, which the finals table also carries.
    assert '<span class="tag">latest</span>' in page
    assert page.count('<span class="tag">latest</span>') == 1


def test_the_page_lists_the_final_result_paths(gallery, state, study):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")

    page = built_page(gallery, study)

    assert "latest of each kind" in page
    assert str(study / "u.png").replace("&", "&amp;") in page


def test_a_label_cannot_inject_markup(gallery, state, study):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png", label='<script>alert("x")</script>')

    page = built_page(gallery, study)

    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_an_image_over_the_limit_is_named_rather_than_embedded(gallery, state, study):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")

    page = built_page(gallery, study, limit=10)

    assert "too large to embed" in page
    assert "data:image/png;base64," not in page
    assert str(study / "u.png") in page


def test_the_page_is_written_where_it_was_asked_for(gallery, state, study, tmp_path):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")
    out = tmp_path / "elsewhere" / "page.html"

    written = gallery.write_html(gallery.group_by_kind(gallery.collect(study)), out)

    assert written == out
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


# -- the text index ------------------------------------------------------------


def test_the_index_groups_by_kind_and_flags_the_missing(gallery, state, study):
    make_png(study / "m.png")
    record(state, study, "mesh-full", "m.png", label="coarse")
    record(state, study, "velocity", "gone.png")

    text = gallery.text_index(gallery.group_by_kind(gallery.collect(study)))

    assert "mesh-full  (1)" in text
    assert "velocity  (1)  1 missing" in text
    assert "MISSING" in text
    assert "coarse" in text


def test_final_paths_are_absolute_and_one_per_line(gallery, state, study):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")

    paths = gallery.final_paths(gallery.final_entries(gallery.collect(study)))

    assert paths == [str(study / "u.png")]
    assert all(Path(path).is_absolute() for path in paths)


def test_human_size_reads_as_sizes(gallery):
    assert gallery.human_size(0) == "-"
    assert gallery.human_size(512) == "512 B"
    assert gallery.human_size(2048) == "2.0 kB"


# -- the contact sheet ---------------------------------------------------------


def test_grid_shape_stays_within_three_columns(gallery):
    assert gallery.grid_shape(0) == (0, 0)
    assert gallery.grid_shape(1) == (1, 1)
    assert gallery.grid_shape(4) == (2, 2)
    assert gallery.grid_shape(7) == (3, 3)
    assert gallery.grid_shape(10)[1] == 3


def test_the_sheet_draws_one_panel_per_drawable_kind(gallery, state, study):
    make_png(study / "m.png")
    record(state, study, "mesh-full", "m.png")
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")
    (study / "report.md").write_text("x")
    record(state, study, "report", "report.md")

    finals = gallery.final_entries(gallery.collect(study))
    assert len(gallery.sheet_panels(finals)) == 2

    out = gallery.render_sheet(finals, study / "sheet.png", title="wing")

    assert out is not None and out.exists()
    assert out.stat().st_size > 1000


def test_the_sheet_is_not_drawn_when_there_is_no_picture(gallery, state, study):
    (study / "report.md").write_text("x")
    record(state, study, "report", "report.md")

    finals = gallery.final_entries(gallery.collect(study))

    assert gallery.render_sheet(finals, study / "sheet.png") is None
    assert not (study / "sheet.png").exists()


def test_the_sheet_caption_names_the_kind(gallery, state, study):
    make_png(study / "m.png")
    record(state, study, "mesh-full", "m.png", label="coarse", case="wing")
    (kind, entry), = gallery.final_entries(gallery.collect(study))

    caption = gallery.sheet_caption(kind, entry)

    assert caption.startswith("mesh-full\n")
    assert "coarse" in caption and "(wing)" in caption


# -- the command line ----------------------------------------------------------


def test_an_empty_manifest_says_so_and_leaves_no_files(gallery, study, capsys):
    assert gallery.main([str(study)]) == 0

    out = capsys.readouterr().out
    assert "no artifacts registered" in out
    assert "manifest" in out
    assert not (study / gallery.HTML_NAME).exists()
    assert not (study / gallery.SHEET_NAME).exists()


def test_a_run_writes_both_and_registers_them(gallery, state, study, capsys):
    make_png(study / "renders" / "mesh.png")
    record(state, study, "mesh-full", "renders/mesh.png")

    assert gallery.main([str(study)]) == 0

    page = study / gallery.HTML_NAME
    sheet = study / gallery.SHEET_NAME
    assert page.exists() and sheet.exists()
    out = capsys.readouterr().out
    assert str(page) in out and str(sheet) in out

    kinds = {row["kind"] for row in state.artifacts(root=study)}
    assert {"gallery", "contact-sheet"} <= kinds
    registered = state.latest("gallery", root=study)
    assert Path(registered["abspath"]) == page


def test_a_second_run_does_not_put_the_first_gallery_in_the_gallery(
    gallery, state, study, capsys
):
    make_png(study / "renders" / "mesh.png")
    record(state, study, "mesh-full", "renders/mesh.png")

    gallery.main([str(study)])
    gallery.main([str(study)])
    capsys.readouterr()

    entries = gallery.collect(study)
    assert [entry.kind for entry in entries] == ["mesh-full"]
    page = (study / gallery.HTML_NAME).read_text(encoding="utf-8")
    assert gallery.HTML_NAME not in page


def test_the_run_can_be_pointed_at_other_paths(gallery, state, study, tmp_path, capsys):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")
    page = tmp_path / "out" / "page.html"
    sheet = tmp_path / "out" / "sheet.png"

    assert gallery.main([str(study), "--html", str(page), "--sheet", str(sheet)]) == 0

    capsys.readouterr()
    assert page.exists() and sheet.exists()
    assert not (study / gallery.HTML_NAME).exists()


def test_list_prints_the_index_and_writes_nothing(gallery, state, study, capsys):
    make_png(study / "m.png")
    record(state, study, "mesh-full", "m.png", label="coarse")

    assert gallery.main([str(study), "--list"]) == 0

    out = capsys.readouterr().out
    assert "mesh-full" in out and "coarse" in out
    assert not (study / gallery.HTML_NAME).exists()
    assert not (study / gallery.SHEET_NAME).exists()


def test_final_prints_paths_and_nothing_else(gallery, state, study, capsys):
    for name in ("old.png", "new.png"):
        make_png(study / name)
        record(state, study, "mesh-full", name)
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")

    assert gallery.main([str(study), "--final"]) == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines == [str(study / "new.png"), str(study / "u.png")]


def test_final_says_so_when_every_file_has_gone(gallery, state, study, capsys):
    record(state, study, "mesh-full", "gone.png")

    assert gallery.main([str(study), "--final"]) == 0

    assert "has been deleted" in capsys.readouterr().out


def test_asking_for_a_page_alongside_the_list_still_writes_it(
    gallery, state, study, tmp_path, capsys
):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")
    page = tmp_path / "page.html"

    assert gallery.main([str(study), "--list", "--html", str(page)]) == 0

    capsys.readouterr()
    assert page.exists()


def test_naming_a_page_alongside_the_list_writes_only_that_page(
    gallery, state, study, tmp_path, capsys
):
    """`--list --html X` asks for X. It does not also ask for a contact sheet in the
    study and a manifest row pointing at it."""
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")
    page = tmp_path / "page.html"

    assert gallery.main([str(study), "--list", "--html", str(page)]) == 0

    capsys.readouterr()
    assert not (study / gallery.SHEET_NAME).exists()
    assert [row["kind"] for row in state.artifacts(root=study)] == ["velocity", "gallery"]


def test_naming_a_sheet_alongside_final_writes_only_that_sheet(
    gallery, state, study, tmp_path, capsys
):
    make_png(study / "u.png")
    record(state, study, "velocity", "u.png")
    sheet = tmp_path / "sheet.png"

    assert gallery.main([str(study), "--final", "--sheet", str(sheet)]) == 0

    capsys.readouterr()
    assert sheet.exists()
    assert not (study / gallery.HTML_NAME).exists()
    assert [row["kind"] for row in state.artifacts(root=study)] == ["velocity", "contact-sheet"]


def test_a_case_filter_reaches_the_page(gallery, state, study, capsys):
    make_png(study / "coarse.png")
    make_png(study / "fine.png")
    record(state, study, "mesh-full", "coarse.png", case="coarse")
    record(state, study, "mesh-full", "fine.png", case="fine")

    assert gallery.main([str(study), "--case", "fine"]) == 0

    capsys.readouterr()
    page = (study / gallery.HTML_NAME).read_text(encoding="utf-8")
    assert "fine.png" in page
    assert "coarse.png" not in page


def test_a_case_with_nothing_registered_is_not_an_error(gallery, state, study, capsys):
    make_png(study / "coarse.png")
    record(state, study, "mesh-full", "coarse.png", case="coarse")

    assert gallery.main([str(study), "--case", "nobody"]) == 0

    assert "no artifacts registered for case nobody" in capsys.readouterr().out
