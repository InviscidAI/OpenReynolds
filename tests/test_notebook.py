"""`openreynolds.show` and `openreynolds.follow`: the two things a notebook wants.

The SDK already drives a study from Python (`tests/test_sdk.py` holds that surface), so
what is tested here is only the notebook layer: that the figures reach the cell, that a
followed study does not forget what it has already said, and that both work with no
IPython installed -- which is the state of this repository's own test run, IPython being
deliberately not a dependency.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import openreynolds
from openreynolds import notebook


class FakeDisplay:
    """Enough of `IPython.display` for these tests: it records instead of rendering."""

    def __init__(self) -> None:
        self.shown: list = []
        self.clears = 0

    def display(self, item) -> None:
        self.shown.append(item)

    def clear_output(self, wait: bool = False) -> None:
        self.clears += 1

    def Markdown(self, text):  # noqa: N802 - IPython's own name
        return SimpleNamespace(kind="markdown", text=text)

    def Image(self, filename):  # noqa: N802 - IPython's own name
        return SimpleNamespace(kind="image", filename=filename)


@pytest.fixture
def fake_ipython(monkeypatch):
    display = FakeDisplay()
    monkeypatch.setattr(notebook, "_display", lambda: display)
    return display


def results_for(tmp_path: Path, pictures: int = 2, missing: int = 0) -> dict:
    """A `core.study_results` answer (`mcp/core.py:725-759`), with real files behind
    the pictures the way `LocalHost.render_link` hands back this machine's paths
    (`mcp/local.py:698-699`)."""
    rows = []
    for index in range(pictures):
        path = tmp_path / f"slice_{index}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(16))
        rows.append({"name": path.name, "url": str(path)})
    for index in range(missing):
        rows.append({"name": f"gone_{index}.png", "url": str(tmp_path / f"gone_{index}.png")})
    return {
        "study_id": "st-7", "title": "drag on a sphere", "workspace": "/work/st-7",
        "final_message": "Cd = 0.47 at Re = 1000.",
        "jobs": [{"name": "simpleFoam", "status": "finished", "exit_code": 0}],
        "pictures": rows,
        "files": [{"path": "/work/st-7/results.json", "size": 12}],
    }


# -- the surface -------------------------------------------------------------------


def test_the_package_exports_both_from_the_notebook_module():
    assert openreynolds.show is notebook.show
    assert openreynolds.follow is notebook.follow


def test_importing_the_package_does_not_drag_in_ipython():
    """IPython is not a dependency and must not become one: a CLI that pulls in a
    notebook stack to print a study id is the wrong trade."""
    import subprocess
    import sys

    probe = ("import sys, openreynolds; "
             "assert 'IPython' not in sys.modules; print(openreynolds.__version__)")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


# -- show --------------------------------------------------------------------------


def test_show_puts_the_figures_in_the_cell_and_returns_the_results(tmp_path, fake_ipython):
    """The figures are the result of a CFD study. A dict of paths makes the reader open
    them by hand."""
    results = results_for(tmp_path)
    study = SimpleNamespace(results=lambda: results)

    got = notebook.show(study)

    assert got is results
    images = [item.filename for item in fake_ipython.shown if item.kind == "image"]
    assert images == [str(tmp_path / "slice_0.png"), str(tmp_path / "slice_1.png")]


def test_show_renders_the_answer_and_the_jobs_as_markdown(tmp_path, fake_ipython):
    notebook.show(SimpleNamespace(results=lambda: results_for(tmp_path)))

    text = next(item.text for item in fake_ipython.shown if item.kind == "markdown")
    assert "drag on a sphere" in text
    assert "Cd = 0.47 at Re = 1000." in text
    assert "simpleFoam" in text


def test_show_leaves_out_a_picture_whose_file_is_not_here(tmp_path, fake_ipython):
    """The mirror reports every file it did not bring and why (`mirror.py:30-37`). A
    notebook cell is not where that argument belongs, and a broken image is not an
    explanation."""
    notebook.show(SimpleNamespace(results=lambda: results_for(tmp_path, missing=3)))

    assert len([i for i in fake_ipython.shown if i.kind == "image"]) == 2


def test_show_caps_how_many_figures_go_in_one_cell(tmp_path, fake_ipython):
    notebook.show(SimpleNamespace(results=lambda: results_for(tmp_path, pictures=20)))

    images = [i for i in fake_ipython.shown if i.kind == "image"]
    assert len(images) == notebook.MAX_PICTURES
    assert images[-1].filename.endswith("slice_19.png"), "the newest are the ones kept"


def test_show_leaves_videos_alone(tmp_path, fake_ipython):
    results = results_for(tmp_path, pictures=0)
    movie = tmp_path / "flythrough.mp4"
    movie.write_bytes(b"\x00" * 8)
    results["pictures"] = [{"name": movie.name, "url": str(movie)}]

    notebook.show(SimpleNamespace(results=lambda: results))

    assert not [i for i in fake_ipython.shown if i.kind == "image"]


def test_show_takes_a_study_id_as_well_as_a_study(tmp_path, monkeypatch, fake_ipython):
    monkeypatch.setattr("openreynolds.sdk.results",
                        lambda study_id, **kw: results_for(tmp_path) | {"study_id": study_id})

    got = notebook.show("st-99")

    assert got["study_id"] == "st-99"


def test_show_prints_without_ipython(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(notebook, "_display", lambda: None)

    notebook.show(SimpleNamespace(results=lambda: results_for(tmp_path)))

    out = capsys.readouterr().out
    assert "Cd = 0.47 at Re = 1000." in out
    assert "slice_0.png" in out


# -- follow ------------------------------------------------------------------------


def answers():
    """Three status answers in `core.status_answer`'s shape, each carrying only the
    rows after the cursor (`mcp/core.py:673-693`, `sdk.py:78-80`)."""
    return [
        {"state": "working", "study_id": "st-7", "stage": "meshing",
         "transcript": [{"seq": 1, "from": "caller", "text": "mesh the duct"}]},
        {"state": "working", "study_id": "st-7", "stage": "solving",
         "transcript": [{"seq": 2, "tool": "bash", "input": "simpleFoam"}],
         "jobs": [{"name": "simpleFoam", "status": "running"}]},
        {"state": "ended", "study_id": "st-7", "stage": "",
         "transcript": [{"seq": 3, "from": "reynolds", "text": "Cd = 0.47"}]},
    ]


def test_follow_rewrites_one_cell_rather_than_appending(fake_ipython):
    """Printing each answer leaves a forty-minute study as hundreds of scrolled lines.
    A notebook's moving line is `clear_output(wait=True)` and one re-render."""
    notebook.follow(SimpleNamespace(events=lambda: iter(answers())))

    assert fake_ipython.clears == 3
    assert len(fake_ipython.shown) == 3


def test_follow_does_not_forget_what_was_already_said(fake_ipython):
    """Each answer carries only the new rows, so re-rendering the latest batch alone
    would make a cell that forgets the study as it runs."""
    notebook.follow(SimpleNamespace(events=lambda: iter(answers())))

    last = fake_ipython.shown[-1].text
    assert "mesh the duct" in last
    assert "`bash`" in last
    assert "Cd = 0.47" in last


def test_follow_returns_the_last_state(fake_ipython):
    answer = notebook.follow(SimpleNamespace(events=lambda: iter(answers())))

    assert answer["state"] == "ended"


def test_follow_shows_an_open_approval_as_waiting_for_you(fake_ipython):
    rows = [{"state": "waiting_for_approval", "study_id": "st-7", "transcript": [],
             "approvals": [{"title": "run simpleFoam on 16 cores",
                            "choices": ["approve", "decline"]}]}]

    notebook.follow(SimpleNamespace(events=lambda: iter(rows)))

    text = fake_ipython.shown[-1].text
    assert "waiting for you" in text and "16 cores" in text


def test_follow_shows_an_error_rather_than_only_a_state(fake_ipython):
    rows = [{"state": "error", "study_id": "st-7", "transcript": [],
             "error": "the workspace service is unreachable"}]

    notebook.follow(SimpleNamespace(events=lambda: iter(rows)))

    assert "unreachable" in fake_ipython.shown[-1].text


def test_follow_prints_without_ipython(monkeypatch, capsys):
    monkeypatch.setattr(notebook, "_display", lambda: None)

    notebook.follow(SimpleNamespace(events=lambda: iter(answers())))

    assert "Cd = 0.47" in capsys.readouterr().out


def test_follow_passes_its_wait_on_to_events():
    seen = {}

    def events(**kw):
        seen.update(kw)
        return iter([])

    notebook.follow(SimpleNamespace(events=events), wait_s=3.0)

    assert seen == {"wait_s": 3.0}


def test_a_row_of_every_shape_condense_makes_reads_as_a_line():
    """`core.condense` names the speaker four different ways depending on what the row
    is (`mcp/core.py:421-445`), and a notebook has to render all four."""
    assert notebook._row({"from": "reynolds", "text": "done"}) == "- **reynolds**: done"
    assert notebook._row({"tool": "bash", "input": "ls"}) == "- `bash` ls"
    assert "(error)" in notebook._row({"tool": "bash", "input": "ls", "error": True})
    assert notebook._row({"desk": "cad", "text": "built"}) == "- desk cad: built"
    assert notebook._row({"event": "job finished"}) == "- job finished"
