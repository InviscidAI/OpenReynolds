"""The study ends up on the user's machine, and what does not is said out loud.

The complaint this answers was not "the filter is wrong". It was that a study which
ran for twenty-seven minutes left two local files and no way to tell whether that meant
the filter had eaten everything or nothing had ever been made. So most of what is
checked here is the reporting: a skip nobody hears about is the failure mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import ScriptedReader, install_model, message, text_block
from openreynolds import cli, mirror
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.tools import ToolContext
from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.browse import MAX_ENTRIES, Browser

HOME = "/work/study-test"


def listing(*rows: tuple[str, int]) -> str:
    """A `find -printf` listing. A trailing slash on a path means a directory."""
    lines = []
    for path, size in rows:
        kind = "d" if path.endswith("/") else "-"
        lines.append(f"{kind}\t{size}\t1700000000.0\t{path.rstrip('/')}")
    return "\n".join(lines) + "\n"


def workspace(backend, *rows: tuple[str, int]) -> None:
    backend.exec_result = ExecResult(0, listing(*rows), False, None)
    for path, size in rows:
        if not path.endswith("/"):
            backend.files[path] = b"x" * min(size, 64)


def browser_for(backend, store) -> Browser:
    return Browser(backend, store, home=HOME)


# -- what comes down, and what does not ----------------------------------------


def test_the_things_a_person_would_look_at_are_mirrored(backend, store):
    workspace(
        backend,
        (f"{HOME}/case/renders/velocity.png", 90_000),
        (f"{HOME}/notes.md", 400),
        (f"{HOME}/case/log.simpleFoam", 12_000),
        (f"{HOME}/case/postProcessing/forces/0/coefficient.dat", 900),
        (f"{HOME}/run_case.sh", 300),
    )

    report = mirror.sync(browser_for(backend, store))

    assert not report.skipped
    assert {Path(p).name for p in report.pulled} == {
        "velocity.png",
        "notes.md",
        "log.simpleFoam",
        "coefficient.dat",
        "run_case.sh",
    }
    assert (store.files_dir / "study-test" / "notes.md").is_file()


def test_the_case_dictionaries_come_down_although_they_have_no_extension(backend, store):
    """`fvSchemes` and `U` are the setup itself, and together weigh less than one
    render. Nothing about their names says so."""
    workspace(
        backend,
        (f"{HOME}/case/system/fvSchemes", 1_200),
        (f"{HOME}/case/system/controlDict", 900),
        (f"{HOME}/case/constant/transportProperties", 300),
        (f"{HOME}/case/0/U", 800),
        (f"{HOME}/case/0.orig/p", 800),
    )

    report = mirror.sync(browser_for(backend, store))

    assert len(report.pulled) == 5
    assert not report.skipped


@pytest.mark.parametrize(
    "path,expected",
    [
        (f"{HOME}/case/processor3/0.5/U", "processor decomposition data"),
        (f"{HOME}/case/250/U", "a written time directory (250/)"),
        (f"{HOME}/case/0.5/p", "a written time directory (0.5/)"),
        (f"{HOME}/case/constant/polyMesh/points", "mesh data"),
        (f"{HOME}/case/VTK/case_400.vtu", "written for a viewer that is not on this machine"),
        (f"{HOME}/case/case.foam", "written for a viewer that is not on this machine"),
        (f"{HOME}/tools/__pycache__/render.cpython-312.pyc", "compiled python"),
        (f"{HOME}/case/mesh.bin", "not an image, a report, a log or a case dictionary"),
    ],
)
def test_the_enormous_and_the_unreadable_stay_on_the_instance(backend, store, path, expected):
    workspace(backend, (path, 5_000))

    report = mirror.sync(browser_for(backend, store))

    assert report.pulled == []
    assert [(skip.path, skip.reason) for skip in report.skipped] == [(path, expected)]


def test_all_takes_what_the_filter_refused(backend, store):
    workspace(
        backend,
        (f"{HOME}/case/processor0/0.5/U", 5_000),
        (f"{HOME}/case/VTK/case_400.vtu", 5_000),
    )

    report = mirror.sync(browser_for(backend, store), everything=True)

    assert len(report.pulled) == 2
    assert report.skipped == []


def test_a_path_can_be_asked_for_on_its_own(backend, store):
    workspace(backend, ("/work/other-study/notes.md", 100))

    mirror.sync(browser_for(backend, store), path="/work/other-study")

    assert backend.last_exec[0].startswith("find -H /work/other-study")
    assert (store.files_dir / "other-study" / "notes.md").is_file()


# -- the caps ------------------------------------------------------------------


def test_one_huge_file_is_left_and_named(backend, store):
    workspace(
        backend,
        (f"{HOME}/case/log.simpleFoam", 400 * 1024 * 1024),
        (f"{HOME}/notes.md", 100),
    )

    report = mirror.sync(browser_for(backend, store))

    assert [Path(p).name for p in report.pulled] == ["notes.md"]
    assert "over the 25M limit for one file" in report.skipped[0].reason


def test_the_per_file_cap_holds_even_under_all(backend, store):
    """`--all` is a statement about which files are interesting, not about how much
    disk to fill."""
    workspace(backend, (f"{HOME}/case/processor0/mesh.bin", 400 * 1024 * 1024))

    report = mirror.sync(browser_for(backend, store), everything=True)

    assert report.pulled == []
    assert "over the" in report.skipped[0].reason


def test_a_sync_stops_at_the_total_budget(backend, store):
    workspace(
        backend,
        (f"{HOME}/a.png", 600),
        (f"{HOME}/b.png", 600),
        (f"{HOME}/c.png", 600),
    )

    report = mirror.sync(browser_for(backend, store), max_total_bytes=1_500)

    assert len(report.pulled) == 2
    assert [skip.reason for skip in report.skipped] == ["past the 1.5K budget for one sync"]


def test_the_budget_is_spent_on_the_small_files_first(backend, store):
    """Running out should cost one enormous log, not the three hundred dictionaries
    queued behind it."""
    workspace(
        backend,
        (f"{HOME}/big.png", 5_000),
        (f"{HOME}/case/system/fvSchemes", 100),
        (f"{HOME}/case/system/controlDict", 100),
    )

    report = mirror.sync(browser_for(backend, store), max_total_bytes=5_000)

    assert sorted(Path(p).name for p in report.pulled) == ["controlDict", "fvSchemes"]
    assert [Path(skip.path).name for skip in report.skipped] == ["big.png"]


# -- nothing is skipped quietly ------------------------------------------------


def test_every_skip_carries_a_reason_and_the_reasons_are_reported(backend, store):
    """A silent filter and an empty workspace look identical from this end."""
    workspace(
        backend,
        (f"{HOME}/case/processor0/0.5/U", 100),
        (f"{HOME}/case/processor1/0.5/U", 100),
        (f"{HOME}/case/500/U", 100),
        (f"{HOME}/notes.md", 100),
    )

    report = mirror.sync(browser_for(backend, store))
    brief = "\n".join(report.brief())

    assert all(skip.reason for skip in report.skipped)
    assert "3 file(s)" in brief
    assert "2 processor decomposition data" in brief
    assert "openreynolds pull --study study-test --all" in brief


def test_a_full_report_names_the_files_it_left(backend, store):
    workspace(backend, (f"{HOME}/case/VTK/case_400.vtu", 100))

    lines = "\n".join(mirror.sync(browser_for(backend, store)).lines())

    assert f"{HOME}/case/VTK/case_400.vtu" in lines


def test_a_truncated_listing_says_so_rather_than_looking_empty(backend, store):
    """`find` output is capped. A workspace too big to list is not an empty one."""
    rows = tuple((f"{HOME}/case/f{n}.bin", 10) for n in range(MAX_ENTRIES))
    workspace(backend, *rows)

    report = mirror.sync(browser_for(backend, store))

    assert any("was not looked at" in warning for warning in report.warnings)


def test_nothing_at_all_says_nothing_at_all(backend, store):
    """Silence is only allowed when there is genuinely nothing to report."""
    backend.exec_result = ExecResult(0, "", False, None)

    assert mirror.sync(browser_for(backend, store)).brief() == []


# -- only what changed ---------------------------------------------------------


def test_a_file_already_here_is_not_asked_for_twice(backend, store):
    workspace(backend, (f"{HOME}/notes.md", 1))

    first = mirror.sync(browser_for(backend, store))
    backend.fetched.clear()
    second = mirror.sync(browser_for(backend, store))

    assert len(first.pulled) == 1
    assert second.pulled == []
    assert second.unchanged == 1
    assert backend.fetched == []


def test_a_file_rewritten_on_the_instance_comes_down_again(backend, store):
    workspace(backend, (f"{HOME}/notes.md", 1))
    mirror.sync(browser_for(backend, store))
    backend.fetched.clear()

    # Same size, later mtime: the local copy is stale and nothing about its length
    # would ever say so.
    backend.exec_result = ExecResult(
        0, f"-\t1\t{2 ** 31}\t{HOME}/notes.md\n", False, None
    )
    report = mirror.sync(browser_for(backend, store))

    assert len(report.pulled) == 1
    assert backend.fetched == [f"{HOME}/notes.md"]


def test_a_file_that_grew_comes_down_again(backend, store):
    workspace(backend, (f"{HOME}/case/log.simpleFoam", 1))
    mirror.sync(browser_for(backend, store))
    backend.fetched.clear()

    workspace(backend, (f"{HOME}/case/log.simpleFoam", 40))
    report = mirror.sync(browser_for(backend, store))

    assert len(report.pulled) == 1


# -- it never breaks the session -----------------------------------------------


def test_a_workspace_that_cannot_be_listed_is_a_warning_not_a_crash(backend, store):
    def refuse(*args, **kwargs):
        raise BackendError("instance is gone", code="not_found", status=404)

    backend.exec = refuse

    report = mirror.sync(browser_for(backend, store))

    assert report.pulled == []
    assert any("instance is gone" in warning for warning in report.warnings)


def test_a_copy_that_fails_is_reported_and_the_rest_of_the_sync_survives(backend, store):
    workspace(backend, (f"{HOME}/notes.md", 100))

    def refuse(paths, local_dir):
        raise BackendError("the archive endpoint fell over", code="backend_error")

    backend.get_tree = refuse

    report = mirror.sync(browser_for(backend, store))

    assert report.pulled == []
    assert [skip.reason for skip in report.skipped] == ["the copy failed"]
    assert any("fell over" in warning for warning in report.warnings)


def test_a_study_with_nowhere_to_put_things_says_so(backend):
    report = mirror.sync(Browser(backend, None, home=HOME))

    assert report.warnings == ["no study directory to mirror into"]


# -- where a file lands --------------------------------------------------------


def test_a_pulled_file_keeps_the_shape_it_had_in_the_workspace(store):
    """The prediction has to match what `get_tree` does, or every sync re-pulls
    everything forever."""
    assert mirror.local_for(Path("/local"), "/work/s/case/0/U") == Path(
        "/local/s/case/0/U"
    )


# -- wired into the session, without anyone asking -----------------------------


def test_a_turn_ending_leaves_the_study_on_this_machine(
    backend, store, view, console, monkeypatch
):
    """The `fetch` tool existing is not the same as anything having called it, and
    for a whole release nothing did."""
    monkeypatch.setattr(cli, "console", console)
    workspace(backend, (f"{HOME}/case/renders/velocity.png", 900))
    loop = Loop(
        Config(anthropic_api_key="k", model="claude-opus-5"),
        ToolContext(backend=backend, store=store, max_output=1000),
        store,
        view,
    )
    install_model(loop, [message([text_block("rendered it")])])

    cli._run_interactive(
        loop,
        backend,
        store,
        view,
        browser_for(backend, store),
        ScriptedReader(["render the velocity field", "/exit"]),
    )

    assert (store.files_dir / "study-test" / "case" / "renders" / "velocity.png").is_file()
    assert any("mirrored" in line for line in view.infos)


def test_a_mirror_that_fails_outright_does_not_end_the_session(view, monkeypatch):
    """Every other failure here is caught and reported. This is the one that is not
    foreseen, and a session with jobs running may not be lost to it."""

    def explode(*args, **kwargs):
        raise RuntimeError("something nobody thought of")

    monkeypatch.setattr(cli, "mirror_sync", explode)

    cli._mirror(browser=None, view=view)

    assert any("something nobody thought of" in warning for warning in view.warnings)


def test_leaving_says_where_the_local_copy_is(backend, store, monkeypatch):
    """Somewhere on the instance is not an answer to "where are my files"."""
    import io as _io

    from rich.console import Console

    written = _io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=written, width=200))
    store.session.home = HOME

    cli._report_on_exit(backend, store)

    assert str(store.files_dir) in written.getvalue()


# -- asking for it by hand -----------------------------------------------------


def wide_console(monkeypatch):
    """`rich` folds long paths at 80 columns, which would split the very strings
    these tests are about."""
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(width=200))


def as_the_only_study(monkeypatch, backend, store):
    monkeypatch.setattr(
        Config, "load", classmethod(lambda cls: Config(studies_dir=store.dir.parent))
    )
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (backend, None, "instance-1"))
    store.session.home = HOME
    store.save()


def test_pull_brings_the_study_down_and_names_what_it_left(backend, store, monkeypatch):
    wide_console(monkeypatch)
    as_the_only_study(monkeypatch, backend, store)
    workspace(backend, (f"{HOME}/notes.md", 40), (f"{HOME}/case/VTK/a.vtu", 40))

    result = CliRunner().invoke(cli.main, ["pull", "--study", "study-test"])

    assert result.exit_code == 0, result.output
    assert (store.files_dir / "study-test" / "notes.md").is_file()
    assert "a.vtu" in result.output
    assert str(store.files_dir) in result.output


def test_pull_all_takes_the_files_the_filter_refused(backend, store, monkeypatch):
    wide_console(monkeypatch)
    as_the_only_study(monkeypatch, backend, store)
    workspace(backend, (f"{HOME}/case/VTK/a.vtu", 40))

    result = CliRunner().invoke(cli.main, ["pull", "--study", "study-test", "--all"])

    assert result.exit_code == 0, result.output
    assert (store.files_dir / "study-test" / "case" / "VTK" / "a.vtu").is_file()


def test_pull_can_be_pointed_at_one_directory(backend, store, monkeypatch):
    wide_console(monkeypatch)
    as_the_only_study(monkeypatch, backend, store)
    workspace(backend, ("/work/study-test/case/notes.md", 40))

    result = CliRunner().invoke(
        cli.main, ["pull", "/work/study-test/case", "--study", "study-test"]
    )

    assert result.exit_code == 0, result.output
    assert backend.last_exec[0].startswith("find -H /work/study-test/case")


def test_pull_fails_loudly_when_it_could_not_reach_the_workspace(backend, store, monkeypatch):
    """Someone put this in a script. Printing a complaint and exiting 0 is how a
    scripted mirror silently stops mirroring."""
    wide_console(monkeypatch)
    as_the_only_study(monkeypatch, backend, store)

    def refuse(*args, **kwargs):
        raise BackendError("instance is gone", code="not_found", status=404)

    backend.exec = refuse

    result = CliRunner().invoke(cli.main, ["pull", "--study", "study-test"])

    assert result.exit_code == 1
    assert "instance is gone" in result.output


# -- what a live run found -----------------------------------------------------


def test_a_case_that_uses_0_initial_is_still_a_case():
    """The convention is `0.orig`, but it is only a convention. A live run wrote
    `0.initial` and every field in it was skipped as "not a case dictionary",
    because the rule listed names instead of describing them."""
    from openreynolds.mirror import reason_to_skip

    assert reason_to_skip("case/0.initial/U") is None
    assert reason_to_skip("case/0.orig/p") is None
    assert reason_to_skip("case/0/k") is None
    assert reason_to_skip("case/0.5/U") == "a written time directory (0.5/)"
    assert reason_to_skip("case/250/p") == "a written time directory (250/)"


def test_a_dictionary_that_is_nine_megabytes_is_field_data():
    """`0/U` on a half-million-cell mesh is the same field data as `500/U`, sitting
    in the directory the solver started from. Keeping it by location alone turned a
    43-file mirror into 42 MB of it."""
    from openreynolds.mirror import reason_by_size

    assert reason_by_size("case/0/U", 9_600_000).startswith("field data")
    assert reason_by_size("case/0/U", 4_000) is None
    assert reason_by_size("case/system/controlDict", 3_000) is None


def test_a_big_render_or_log_is_still_wanted():
    """The size rule is about things kept for where they sit, not for what they are."""
    from openreynolds.mirror import reason_by_size

    assert reason_by_size("renders/big.png", 9_000_000) is None
    assert reason_by_size("case/log.simpleFoam", 5_000_000) is None


def test_a_round_trip_is_bounded_by_bytes_as_well_as_by_count():
    """A live run asked for 38 files at once and the connection closed at 6 MB of an
    expected 38 MB. The batch failed, and every keepable file in it failed with it."""
    from openreynolds.browse import Entry
    from openreynolds.mirror import BATCH_BYTES, _batches

    big = [Entry(path=f"/work/f{n}", is_dir=False, size=3_000_000) for n in range(5)]
    batches = _batches(big)

    assert len(batches) > 1
    for batch in batches:
        assert len(batch) == 1 or sum(e.size for e in batch) <= BATCH_BYTES


def test_one_file_over_the_batch_budget_still_goes():
    from openreynolds.browse import Entry
    from openreynolds.mirror import _batches

    huge = [Entry(path="/work/one", is_dir=False, size=20_000_000)]
    assert _batches(huge) == [huge]


def test_a_failed_batch_is_retried_one_at_a_time(backend, store, tmp_path):
    """Losing the good company of one awkward file is how a mirror comes back empty
    from a study that had plenty worth keeping."""
    from openreynolds.backend.base import BackendError
    from openreynolds.browse import Browser
    from openreynolds.mirror import sync

    for name in ("notes.md", "report.md", "plot.png"):
        backend.files[f"/work/s/{name}"] = b"x" * 100
    backend.exec_result = ExecResult(
        0,
        "".join(f"-\t100\t1700000000.0\t/work/s/{n}\n" for n in ("notes.md", "report.md", "plot.png")),
        False,
        None,
    )

    attempts = []
    original = backend.get_tree

    def awkward(paths, local_dir):
        attempts.append(list(paths))
        if len(paths) > 1:
            raise BackendError("peer closed connection", code="unreachable")
        if paths[0].endswith("plot.png"):
            raise BackendError("still no", code="unreachable")
        return original(paths, local_dir)

    backend.get_tree = awkward
    report = sync(Browser(backend, store, home="/work/s"), path="/work/s")

    assert len(report.pulled) == 2, "the two good files came down anyway"
    assert any("plot.png" in s.path for s in report.skipped)
    assert max(len(a) for a in attempts) > 1 and min(len(a) for a in attempts) == 1
