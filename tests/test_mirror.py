"""The study ends up on the user's machine, and what does not is said out loud.

The complaint this answers was not "the filter is wrong". It was that a study which
ran for twenty-seven minutes left two local files and no way to tell whether that meant
the filter had eaten everything or nothing had ever been made. So most of what is
checked here is the reporting: a skip nobody hears about is the failure mode.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import shutil
import pytest
from click.testing import CliRunner

from conftest import ScriptedReader, install_model, message, text_block
from openreynolds import cli, mirror
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.tools import ToolContext
from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.browse import MAX_ENTRIES, Browser
from openreynolds.store import Store

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
        (f"{HOME}/case/log.simpleFoam", mirror.MAX_FILE_BYTES + 1),
        (f"{HOME}/notes.md", 100),
    )

    report = mirror.sync(browser_for(backend, store))

    assert [Path(p).name for p in report.pulled] == ["notes.md"]
    assert "limit for one file" in report.skipped[0].reason


def test_the_per_file_cap_holds_even_when_taking_everything(backend, store):
    """"Everything" is a statement about which files are wanted, not about how much
    disk to fill. The cap is the one thing that stays a stop."""
    workspace(backend, (f"{HOME}/case/processor0/mesh.bin", mirror.MAX_FILE_BYTES + 1))

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
    assert "openreynolds pull --study study-test tries them again" in brief


def test_a_full_report_names_the_files_it_left(backend, store):
    workspace(backend, (f"{HOME}/case/VTK/case_400.vtu", 100))

    lines = "\n".join(mirror.sync(browser_for(backend, store)).lines())

    assert f"{HOME}/case/VTK/case_400.vtu" in lines


def test_a_truncated_listing_says_so_rather_than_looking_empty(backend, store):
    """`find` output is capped. A workspace too big to list is not an empty one."""
    rows = tuple((f"{HOME}/case/f{n}.bin", 10) for n in range(MAX_ENTRIES + 5))
    workspace(backend, *rows)

    report = mirror.sync(browser_for(backend, store))

    assert any("was not looked at" in warning for warning in report.warnings)


def test_a_workspace_of_exactly_the_cap_is_complete_and_says_nothing(backend, store):
    """The boundary this used to get wrong. `len(entries) >= MAX_ENTRIES` counted from
    the outside and called a workspace holding exactly the cap truncated -- it is not,
    it is a complete listing that happens to be that long. `browse` asks `find` for one
    line more than it keeps, so "there was more" is measured rather than inferred, and
    a warning here would be a false alarm about a workspace that is entirely visible."""
    rows = tuple((f"{HOME}/case/f{n}.bin", 10) for n in range(MAX_ENTRIES))
    workspace(backend, *rows)

    report = mirror.sync(browser_for(backend, store))

    assert not any("was not looked at" in warning for warning in report.warnings)


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


def test_a_path_that_would_land_outside_the_mirror_is_refused(tmp_path):
    """The join is in one place, so the promise that the answer is inside the study's
    own directory is made in one place too (#180). Every caller reads what comes
    back, so a path that walks out has to fail here rather than there."""
    inside = tmp_path / "files"
    assert mirror.local_for(inside, "/work/s/case/0/U") == inside / "s/case/0/U"
    for walk in ("/work/../../etc/passwd", "/work/s/../../../etc/passwd"):
        with pytest.raises(mirror.OutsideMirror):
            mirror.local_for(inside, walk)
    # On Windows `\` is a separator too, which is the spelling that got past the
    # string test in `mcp/local.py`. On Linux it is an ordinary character and the
    # path stays inside, so only the containment is asserted on both platforms.
    landed = None
    try:
        landed = mirror.local_for(inside, "/work/s\\..\\..\\..\\etc\\passwd")
    except mirror.OutsideMirror:
        pass
    if landed is not None:
        assert landed.resolve(strict=False).is_relative_to(inside.resolve(strict=False))


# -- wired into the session, without anyone asking -----------------------------


def test_a_turn_ending_leaves_the_study_on_this_machine(
    backend, store, view, console, monkeypatch
):
    """The `fetch` tool existing is not the same as anything having called it, and
    for a whole release nothing did."""
    monkeypatch.setattr(cli, "console", console)
    workspace(backend, (f"{HOME}/case/renders/velocity.png", 900))
    loop = Loop(
        Config(llm_api_key="k", model="claude-opus-5"),
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
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=written, width=200, force_terminal=False, soft_wrap=True),
    )
    store.session.home = HOME

    cli._report_on_exit(backend, store)

    assert str(store.files_dir) in written.getvalue()


# -- the live mirror -----------------------------------------------------------


def test_a_cycle_brings_files_home_without_a_turn_ending(backend, store, view):
    """The point of the live mirror: a job writing while the model's turn is over
    still reaches the user's machine."""
    workspace(backend, (f"{HOME}/case/postProcessing/forces/0/force.dat", 600))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.view = view

    report = live.sync_now()

    assert report.pulled, "the job's output came home"
    assert view.mirrors and view.mirrors[-1] is report
    assert (
        store.files_dir / "study-test" / "case" / "postProcessing" / "0" / "force.dat"
    ).parent.parent.is_dir()


def test_the_background_thread_cycles_on_its_own(backend, store, view):
    import time as _time

    workspace(backend, (f"{HOME}/notes.md", 40))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0.01)
    live.view = view
    live.start()
    try:
        deadline = _time.time() + 5.0
        while not view.mirrors and _time.time() < deadline:
            _time.sleep(0.01)
    finally:
        live.stop()

    assert view.mirrors, "a cycle ran without anyone asking"
    assert (store.files_dir / "study-test" / "notes.md").is_file()


def test_a_poke_syncs_now_rather_than_at_the_next_interval(backend, store, view):
    """The model just looked at a render; the user should not be an interval behind
    a picture that already exists."""
    import time as _time

    workspace(backend, (f"{HOME}/renders/mesh.png", 900))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=3600)
    live.view = view
    live.start()
    try:
        live.poke()
        deadline = _time.time() + 5.0
        while not view.mirrors and _time.time() < deadline:
            _time.sleep(0.01)
    finally:
        live.stop()

    assert view.mirrors, "the poke ran a cycle; the hour-long interval did not gate it"
    assert (store.files_dir / "study-test" / "renders" / "mesh.png").is_file()


def test_a_poke_with_no_thread_is_quietly_nothing(backend, store):
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.start()
    live.poke()  # must not raise, must not block
    live.stop()


def test_an_interval_of_zero_means_no_thread(backend, store):
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.start()
    assert live._thread is None
    live.stop()  # idempotent, and must not raise


def test_a_cycle_that_blows_up_becomes_a_report_not_an_end(backend, store, view, monkeypatch):
    """Nothing about a convenience may end a session with jobs in flight."""

    def explode(*args, **kwargs):
        raise RuntimeError("nobody foresaw this")

    monkeypatch.setattr(mirror, "sync", explode)
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.view = view

    report = live.sync_now()

    assert any("nobody foresaw this" in warning for warning in report.warnings)
    assert view.mirrors, "the failure was still reported to the view"


def test_a_view_that_dies_mid_telling_does_not_take_the_mirror_down(backend, store):
    workspace(backend, (f"{HOME}/notes.md", 40))

    class DyingView:
        def mirrored(self, report):
            raise RuntimeError("the interface is tearing down")

    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.view = DyingView()

    report = live.sync_now()  # must not raise

    assert report.pulled


def test_the_final_sync_waits_for_a_cycle_that_outlived_stop(backend, store, monkeypatch):
    """stop()'s join is bounded, so a slow cycle can survive it. The close-down
    sync must queue behind that cycle on the same lock, or the two interleave
    over the same files."""
    import threading
    import time as _time

    order = []
    cycle_started = threading.Event()
    release_cycle = threading.Event()

    def slow_sync(browser, everything=True, **kwargs):
        order.append("cycle-start")
        cycle_started.set()
        release_cycle.wait(5)
        order.append("cycle-end")
        return mirror.MirrorReport(local_dir=store.fetch_dir())

    monkeypatch.setattr(mirror, "sync", slow_sync)
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0.01)
    live.start()
    assert cycle_started.wait(5)
    live.stop(timeout=0.01)  # returns with the cycle still inside sync()

    def final():
        order.append("final-ask")
        live.sync_now()
        order.append("final-done")

    asker = threading.Thread(target=final)
    asker.start()
    _time.sleep(0.05)
    release_cycle.set()
    asker.join(5)

    assert not asker.is_alive()
    assert order.index("cycle-end") < order.index("final-done"), (
        "the final sync ran concurrently with the surviving cycle"
    )


def test_a_sync_remembers_its_listing_for_whoever_draws_the_workspace(backend, store):
    """The files pane reads the cache the mirror keeps; a sync that forgot its own
    listing would put the pane back on the network."""
    workspace(backend, (f"{HOME}/notes.md", 40))
    browser = browser_for(backend, store)

    mirror.sync(browser)

    cached = browser.cached(HOME)
    assert cached is not None
    assert any(entry.path == f"{HOME}/notes.md" for entry in cached)
    assert browser.cache_age() is not None


# -- asking for it by hand -----------------------------------------------------


def wide_console(monkeypatch):
    """`rich` folds long paths at 80 columns, which would split the very strings
    these tests are about."""
    from rich.console import Console

    monkeypatch.setattr(
        cli,
        "console",
        Console(width=200, force_terminal=False, soft_wrap=True),
    )


def as_the_only_study(monkeypatch, backend, store):
    monkeypatch.setattr(
        Config, "load", classmethod(lambda cls: Config(studies_dir=store.dir.parent))
    )
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (backend, None, "instance-1"))
    store.session.home = HOME
    store.session.instance_id = "instance-1"
    store.save()


def test_pull_brings_everything_down(backend, store, monkeypatch):
    """Everything, by default. The instruction was "all the files ... including every
    visualization and everything has to be brought over. Always"."""
    wide_console(monkeypatch)
    as_the_only_study(monkeypatch, backend, store)
    workspace(backend, (f"{HOME}/notes.md", 40), (f"{HOME}/case/VTK/a.vtu", 40))

    result = CliRunner().invoke(cli.main, ["pull", "--study", "study-test"])

    assert result.exit_code == 0, result.output
    assert (store.files_dir / "study-test" / "notes.md").is_file()
    assert (store.files_dir / "study-test" / "case" / "VTK" / "a.vtu").is_file()
    assert str(store.files_dir) in result.output


def test_readable_only_is_the_thing_you_opt_into(backend, store, monkeypatch):
    """The filter still exists for anyone who wants a small copy, but nobody gets it
    by accident."""
    wide_console(monkeypatch)
    as_the_only_study(monkeypatch, backend, store)
    workspace(backend, (f"{HOME}/notes.md", 40), (f"{HOME}/case/VTK/a.vtu", 40))

    result = CliRunner().invoke(
        cli.main, ["pull", "--study", "study-test", "--readable-only"]
    )

    assert result.exit_code == 0, result.output
    assert (store.files_dir / "study-test" / "notes.md").is_file()
    assert not (store.files_dir / "study-test" / "case" / "VTK" / "a.vtu").exists()
    assert "a.vtu" in result.output, "and it says what it left"


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


# -- a study this machine has no record of (#215) ------------------------------


class StudyRows:
    """The platform's `GET /v1/studies/{id}`, as `FoamdClient` asks it."""

    def __init__(self, rows: dict[str, dict] | None = None, error: BackendError | None = None):
        self.rows, self.error, self.asked = rows or {}, error, []

    def __call__(self, url, key):
        return self

    def get_study(self, study_id):
        self.asked.append(study_id)
        if self.error:
            raise self.error
        if study_id not in self.rows:
            raise BackendError("no such study", code="not_found", status=404)
        return self.rows[study_id]

    def close(self):
        pass


def from_the_browser(monkeypatch, tmp_path, backend, rows: StudyRows) -> list:
    """No study on this machine; the platform's rows are `rows`; `acquire` records
    which workspace it was asked for."""
    monkeypatch.setattr(
        Config, "load", classmethod(lambda cls: Config(studies_dir=tmp_path / "studies"))
    )
    monkeypatch.setattr(cli.hosted, "FoamdClient", rows)
    asked: list = []

    def acquire(url, key, instance_id=None):
        asked.append(instance_id)
        return backend, None, instance_id or "the-newest-workspace"

    monkeypatch.setattr(cli.hosted, "acquire", acquire)
    return asked


WEB_ROW = {"id": "web-study", "home": "/work/web-study", "instance_id": "its-own"}


def test_a_study_started_in_the_browser_is_pulled_from_its_own_workspace(backend, tmp_path, monkeypatch):
    """#215: with no record here, `acquire` took the account's newest workspace,
    usually another study's, and started it. The platform's row names the study's own."""
    wide_console(monkeypatch)
    asked = from_the_browser(monkeypatch, tmp_path, backend, StudyRows({"web-study": WEB_ROW}))
    run_here = Store(tmp_path / "studies", "run-here")   # another study, on its own workspace
    run_here.session.instance_id = "another-studys"
    run_here.save()
    workspace(backend, ("/work/web-study/notes.md", 40), ("/work/web-study/case/VTK/a.vtu", 40))

    result = CliRunner().invoke(cli.main, ["pull", "--study", "web-study"])

    assert result.exit_code == 0, result.output
    assert asked == ["its-own"]
    files = tmp_path / "studies" / "web-study" / "files" / "web-study"
    assert (files / "notes.md").is_file() and (files / "case" / "VTK" / "a.vtu").is_file()
    assert backend.last_exec[0].startswith("find -H /work/web-study"), "its home, not all of /work"
    assert not (tmp_path / "studies" / "web-study" / "session.json").exists(), "nothing recorded"


def test_files_and_push_go_to_a_browser_studys_own_workspace(backend, tmp_path, monkeypatch):
    wide_console(monkeypatch)
    asked = from_the_browser(monkeypatch, tmp_path, backend, StudyRows({"web-study": WEB_ROW}))
    workspace(backend, ("/work/web-study/notes.md", 40))
    geometry = tmp_path / "body.stl"
    geometry.write_text("solid body\nendsolid body\n")

    listed = CliRunner().invoke(cli.main, ["files", "--study", "web-study"])
    pushed = CliRunner().invoke(cli.main, ["push", str(geometry), "--study", "web-study"])

    assert listed.exit_code == 0, listed.output
    assert pushed.exit_code == 0, pushed.output
    assert asked == ["its-own", "its-own"]
    assert "/work/web-study/body.stl" in backend.files


def test_a_study_neither_here_nor_on_the_account_is_refused(backend, tmp_path, monkeypatch):
    wide_console(monkeypatch)
    asked = from_the_browser(monkeypatch, tmp_path, backend, StudyRows())

    result = CliRunner().invoke(cli.main, ["pull", "--study", "no-such-study"])

    assert result.exit_code == 1
    assert "No study no-such-study" in result.output
    assert asked == [], "no workspace was started for it"
    assert not (tmp_path / "studies" / "no-such-study").exists(), "and nothing left behind"


def test_a_browser_study_whose_workspace_was_released_is_refused(backend, tmp_path, monkeypatch):
    wide_console(monkeypatch)
    released = {"id": "web-study", "home": "/work/web-study", "instance_id": None}
    asked = from_the_browser(monkeypatch, tmp_path, backend, StudyRows({"web-study": released}))

    result = CliRunner().invoke(cli.main, ["pull", "--study", "web-study"])

    assert result.exit_code == 1
    assert "no workspace any more" in result.output
    assert asked == []


def test_a_service_that_cannot_be_asked_is_not_a_guess(backend, tmp_path, monkeypatch):
    wide_console(monkeypatch)
    down = StudyRows(error=BackendError("connection refused", code="unreachable"))
    asked = from_the_browser(monkeypatch, tmp_path, backend, down)

    result = CliRunner().invoke(cli.main, ["pull", "--study", "web-study"])

    assert result.exit_code == 1
    assert "Could not ask the workspace service" in result.output
    assert asked == []


def test_a_study_recorded_here_with_no_workspace_asks_the_platform(backend, tmp_path, monkeypatch):
    """A study quit before its first message settled where it runs has a record here
    and no workspace in it; the newest workspace is still not its."""
    wide_console(monkeypatch)
    asked = from_the_browser(monkeypatch, tmp_path, backend, StudyRows({"web-study": WEB_ROW}))
    Store(tmp_path / "studies", "web-study").save()
    workspace(backend, ("/work/web-study/notes.md", 40))

    result = CliRunner().invoke(cli.main, ["pull", "--study", "web-study"])

    assert result.exit_code == 0, result.output
    assert asked == ["its-own"]


def test_a_study_run_here_keeps_the_workspace_it_has_on_record(backend, store, monkeypatch):
    """Only a study unknown here asks the platform."""
    wide_console(monkeypatch)
    as_the_only_study(monkeypatch, backend, store)
    rows = StudyRows()
    monkeypatch.setattr(cli.hosted, "FoamdClient", rows)
    workspace(backend, (f"{HOME}/notes.md", 40))

    result = CliRunner().invoke(cli.main, ["pull", "--study", "study-test"])

    assert result.exit_code == 0, result.output
    assert rows.asked == []


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


def test_catching_up_does_not_wait_when_the_thread_is_running(backend, store, view):
    """The turn-end sync blocked the session thread for twenty-five minutes while a
    transient solve's per-processor fields came home, and two typed messages sat
    unread behind it. With the thread running, catching up is a poke and a return."""
    import time as _time

    workspace(backend, (f"{HOME}/renders/mesh.png", 900))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=3600)
    live.view = view
    live.start()
    try:
        began = _time.monotonic()
        result = live.catch_up()
        took = _time.monotonic() - began
        deadline = _time.time() + 5.0
        while not view.mirrors and _time.time() < deadline:
            _time.sleep(0.01)
    finally:
        live.stop()

    assert result is None, "nothing to hand back: the cycle ran on the other thread"
    assert took < 1.0
    assert view.mirrors, "and it did run"


def test_catching_up_syncs_here_when_there_is_no_thread(backend, store, view):
    """Interval 0 means the turn-end syncs are the only ones; they still happen."""
    workspace(backend, (f"{HOME}/notes.md", 40))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.view = view

    report = live.catch_up()

    assert report is not None and report.pulled


def test_the_tracker_hears_every_cycle(backend, store, view):
    class Ears:
        def __init__(self):
            self.events = []

        def sync_begin(self):
            self.events.append("begin")

        def sync_end(self, report):
            self.events.append(("end", bool(report.pulled)))

    workspace(backend, (f"{HOME}/notes.md", 40))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.progress = Ears()

    live.sync_now()

    assert live.progress.events == ["begin", ("end", True)]


def test_the_mirror_delivers_renders_it_pulls(backend, store, view):
    """Delivery rides on the sync: a render pulled home is surfaced without the agent."""
    from openreynolds.delivery import Gallery

    workspace(backend, (f"{HOME}/case/renders/p.png", 400))
    backend.files[f"{HOME}/case/renders/p.png"] = b"\x89PNG" + b"x" * 396
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=0)
    live.view = view
    live.gallery = Gallery(store.files_dir, store.renders_dir,
                           assemble=lambda *a, **k: None, encoder=lambda: None)

    live.sync_now()

    assert view.deliveries, "a delivered() event reached the view"
    assert (store.renders_dir / "p.png").exists()


# -- the live policy -----------------------------------------------------------
#
# A background cycle is not an explicit `pull` and must not use either of the two
# policies that existed. `everything=True` asked a decomposed solve for every
# per-processor field file it was writing, so a cycle never converged: one study spent
# 31 minutes after its answer was already written moving 233 files, 156 of them
# processorN/ data nothing in either repo opens. `--readable-only` would blind the
# hosted 3D viewer, which reads the mesh out of the mirror.


def test_a_live_cycle_leaves_the_per_processor_copies_behind(backend, store):
    workspace(
        backend,
        (f"{HOME}/case/processor0/2/U", 4_000),
        (f"{HOME}/case/processor1/2/U", 4_000),
        (f"{HOME}/case/2/U", 4_000),
        (f"{HOME}/case/log.simpleFoam", 900),
    )
    report = mirror.sync(browser_for(backend, store), live=True)

    assert sorted(Path(p).name for p in report.pulled) == ["U", "log.simpleFoam"]
    assert [Path(s.path).parent.name for s in report.skipped] == ["2", "2"]
    assert all("processor" in s.reason for s in report.skipped)


def test_a_live_cycle_keeps_the_mesh_the_viewer_reads(backend, store):
    """`--readable-only` calls polyMesh "mesh data" and drops it; the web viewer reads
    its mesh out of the mirror and the page 404s without it."""
    workspace(
        backend,
        (f"{HOME}/case/constant/polyMesh/points", 40_000),
        (f"{HOME}/case/constant/polyMesh/owner", 40_000),
        (f"{HOME}/case/constant/triSurface/body.stl", 9_000),
    )
    report = mirror.sync(browser_for(backend, store), live=True)
    assert sorted(Path(p).name for p in report.pulled) == ["body.stl", "owner", "points"]
    assert report.skipped == []


def whole(backend, *rows: tuple[str, int]) -> None:
    """`workspace`, with each file its full size, so a second cycle finds the copy
    the first made already here."""
    workspace(backend, *rows)
    for path, size in rows:
        backend.files[path] = b"x" * size


def cycle(browser, **kwargs):
    """One background cycle, as `LiveMirror` runs it: live, with a gate."""
    return mirror.sync(browser, live=True, background=True, gate=mirror.Gate(), **kwargs)


def test_a_live_cycle_brings_every_written_time_newest_first(backend, store):
    """A time written between two cycles used to stay on the instance for good: a
    staging study wrote a time every 2 s from 20 s to 100 s and the viewer got four.
    The newest comes first, then the earlier ones, newest of those first."""
    workspace(
        backend,
        (f"{HOME}/case/1/U", 3_000),
        (f"{HOME}/case/2/U", 3_000),
        (f"{HOME}/case/10/U", 3_000),
    )
    report = cycle(browser_for(backend, store))

    assert [Path(p).parent.name for p in report.pulled] == ["10", "2", "1"], (
        "newest by value, not by name, and the earlier ones after it"
    )
    assert report.skipped == []


def test_an_earlier_time_comes_whole_or_not_at_all(backend, store, monkeypatch):
    """A time with half its fields is one the viewer lists and cannot draw."""
    monkeypatch.setattr(mirror, "EARLIER_BYTES_PER_CYCLE", 5_000)
    whole(
        backend,
        (f"{HOME}/case/3/U", 3_000),
        (f"{HOME}/case/2/U", 2_000),
        (f"{HOME}/case/2/p", 2_000),
        (f"{HOME}/case/1/U", 1_500),
    )
    browser = browser_for(backend, store)
    first = cycle(browser)

    pulled = {Path(p).parent.name for p in first.pulled}
    assert pulled == {"3", "2"}, "the newest, then one earlier time, whole"
    waiting = {Path(s.path).parent.name for s in first.skipped}
    assert waiting == {"1"}, "an older, smaller time does not jump the queue"
    assert all("for a later cycle" in s.reason for s in first.skipped)

    second = cycle(browser)
    assert {Path(p).parent.name for p in second.pulled} == {"1"}
    assert second.skipped == []


def test_a_collated_run_brings_its_newest_time_and_never_backfills_the_earlier(backend, store):
    """ui#143: the viewer draws a parallel solve off `processorsN/` (ui#106), so its
    newest time still comes every cycle; an earlier one never does. A long solve wrote
    24.9 GB of them onto a runner disk every session there shares, and the runner now
    deletes a superseded one once it is in the bucket: a backfill would bring it back."""
    rows = [(f"{HOME}/case/processors4/3/U", 3_000), (f"{HOME}/case/processors4/2/U", 2_000),
            (f"{HOME}/case/processors4_0-1/3/U", 3_000), (f"{HOME}/case/processors4_0-1/1/U", 1_000),
            (f"{HOME}/plain/3/U", 3_000), (f"{HOME}/plain/2/U", 2_000)]
    whole(backend, *rows)
    browser = browser_for(backend, store)

    def tail(path) -> str:  # pulled names the copy here, skipped the workspace's path
        text = str(path)
        return text[max(text.rfind("/case/"), text.rfind("/plain/")) + 1:]

    first = cycle(browser)
    pulled = sorted(tail(p) for p in first.pulled)
    assert pulled == ["case/processors4/3/U", "case/processors4_0-1/3/U", "plain/2/U", "plain/3/U"], pulled
    decomposed = sorted(tail(s.path) for s in first.skipped)
    assert decomposed == ["case/processors4/2/U", "case/processors4_0-1/1/U"], decomposed
    assert {s.reason for s in first.skipped} == {f"{mirror.EARLIER_TIME} {mirror.DECOMPOSED_EARLIER}"}

    # The solve writes 4; the hosted runner deletes 3, superseded and uploaded. The next
    # cycle brings 4 and leaves 3 where it is.
    here = store.files_dir / "study-test" / "case"
    for gone in ("processors4/3", "processors4_0-1/3"):
        shutil.rmtree(here / gone)
    whole(backend, *rows, (f"{HOME}/case/processors4/4/U", 3_000), (f"{HOME}/case/processors4_0-1/4/U", 3_000))
    again = cycle(browser)
    assert sorted(tail(p) for p in again.pulled) == ["case/processors4/4/U", "case/processors4_0-1/4/U"]
    assert not (here / "processors4" / "3").exists() and not (here / "processors4_0-1" / "3").exists()


def test_the_sync_budget_goes_to_the_newest_time_before_the_backfill(backend, store):
    """#100: the budget was spent smallest first across everything, so two 2 KB
    fields of an earlier time took a 5 KB budget and the newest time's one 2 KB field
    was left behind -- the viewer one time behind the solver, with half of an earlier
    time it cannot draw."""
    workspace(
        backend,
        (f"{HOME}/case/3/U", 2_000),
        (f"{HOME}/case/2/U", 2_000),
        (f"{HOME}/case/2/p", 2_000),
    )
    report = cycle(browser_for(backend, store), max_total_bytes=5_000)

    assert [Path(p).parent.name for p in report.pulled] == ["3"]
    assert sorted(Path(s.path).name for s in report.skipped) == ["U", "p"], (
        "the earlier time is left whole, not half-copied"
    )
    assert all(
        s.reason.startswith(f"{mirror.EARLIER_TIME} (2/)") and "budget" in s.reason
        for s in report.skipped
    )


def test_earlier_times_within_the_sync_budget_stop_at_the_first_that_does_not_fit(
    backend, store
):
    """Newest first and whole, as `_earlier_within` orders them: an older, smaller
    time does not take the budget the next cycle would give the one before it."""
    workspace(
        backend,
        (f"{HOME}/case/4/U", 1_000),
        (f"{HOME}/case/3/U", 1_000),
        (f"{HOME}/case/2/U", 3_000),
        (f"{HOME}/case/1/U", 500),
    )
    report = cycle(browser_for(backend, store), max_total_bytes=4_000)

    assert [Path(p).parent.name for p in report.pulled] == ["4", "3"]
    assert sorted(Path(s.path).parent.name for s in report.skipped) == ["1", "2"]


def test_one_earlier_time_bigger_than_a_cycle_still_comes(backend, store, monkeypatch):
    """Slow rather than never: a large mesh's time is bigger than any one cycle."""
    monkeypatch.setattr(mirror, "EARLIER_BYTES_PER_CYCLE", 1_000)
    whole(
        backend,
        (f"{HOME}/case/3/U", 3_000),
        (f"{HOME}/case/2/U", 3_000),
        (f"{HOME}/case/1/U", 3_000),
    )
    browser = browser_for(backend, store)
    assert [Path(p).parent.name for p in cycle(browser).pulled] == ["3", "2"]
    assert [Path(p).parent.name for p in cycle(browser).pulled] == ["1"]


def test_earlier_times_stop_at_what_a_session_keeps(backend, store, monkeypatch):
    """The runner stages sixteen sessions on one disk; a long transient solve writes
    more than that. The ones already here count, and the newest still comes."""
    monkeypatch.setattr(mirror, "EARLIER_BYTES", 5_000)
    whole(
        backend,
        (f"{HOME}/case/2/U", 3_000),
        (f"{HOME}/case/1/U", 3_000),
    )
    browser = browser_for(backend, store)
    first = cycle(browser)
    assert [Path(p).parent.name for p in first.pulled] == ["2", "1"]

    whole(
        backend,
        (f"{HOME}/case/4/U", 3_000),
        (f"{HOME}/case/3/U", 3_000),
        (f"{HOME}/case/2/U", 3_000),
        (f"{HOME}/case/1/U", 3_000),
    )
    second = cycle(browser)
    assert [Path(p).parent.name for p in second.pulled] == ["4"], (
        "2/ is now earlier and already here: 3 KB of the 5 KB kept, so 3/ does not fit"
    )
    assert [Path(s.path).parent.name for s in second.skipped] == ["3"]
    assert "kept of them" in second.skipped[0].reason


def test_the_close_down_sync_brings_the_newest_time_and_leaves_the_backlog(backend, store):
    """The end of a session waits on this sync, and its round trips are not bounded
    the way a background cycle's are; a backlog of earlier times is not its job."""
    workspace(backend, (f"{HOME}/case/2/U", 30), (f"{HOME}/case/1/U", 30))
    report = mirror.sync(browser_for(backend, store), live=True)
    assert [Path(p).parent.name for p in report.pulled] == ["2"]
    assert [Path(s.path).parent.name for s in report.skipped] == ["1"]


def test_an_earlier_time_that_will_not_copy_does_not_hold_the_older_ones(backend, store, monkeypatch):
    """Taken newest first, a time whose file times out every cycle would be first in
    line every cycle, and every older time would wait behind it for good."""
    monkeypatch.setattr(mirror, "EARLIER_BYTES_PER_CYCLE", 60)
    whole(
        backend,
        (f"{HOME}/case/3/U", 30), (f"{HOME}/case/2/U", 30), (f"{HOME}/case/2/p", 30),
        (f"{HOME}/case/1/U", 30),
    )
    real = backend.get_tree

    def flaky(paths, local_dir, **kwargs):
        if any(path.endswith("/2/p") for path in paths):
            raise BackendError("timed out", code="timeout")
        return real(paths, local_dir, **kwargs)

    backend.get_tree = flaky
    browser = browser_for(backend, store)
    refused: dict[str, str] = {}
    first = cycle(browser, refused=refused)
    assert f"{HOME}/case/2/p" in refused
    assert "1" not in {Path(p).parent.name for p in first.pulled}, "1/ waits behind 2/"

    second = cycle(browser, refused=refused)
    assert [Path(p).parent.name for p in second.pulled] == ["1"], "the older time came"
    held = {s.path for s in second.skipped}
    assert f"{HOME}/case/2/p" in held


def test_an_earlier_time_whose_file_fails_is_not_left_half_copied(backend, store, monkeypatch):
    """#103: `2/p` times out after the round trip for it has brought `2/U` down.

    `2/U` lands in the staging directory, never in the mirror, so there is no `2/`
    for the viewer to list and fail to draw -- and nothing to delete afterwards."""
    whole(
        backend,
        (f"{HOME}/case/3/U", 30), (f"{HOME}/case/2/U", 30), (f"{HOME}/case/2/p", 30),
        (f"{HOME}/case/1/U", 30),
    )
    real = backend.get_tree

    def flaky(paths, local_dir, **kwargs):
        if any(path.endswith("/2/p") for path in paths):
            raise BackendError("timed out", code="timeout")
        return real(paths, local_dir, **kwargs)

    backend.get_tree = flaky
    browser = browser_for(backend, store)
    refused: dict[str, str] = {}
    first = cycle(browser, refused=refused)

    case = mirror.local_for(first.local_dir, f"{HOME}/case")
    assert sorted(p.name for p in case.iterdir()) == ["1", "3"], "no 2/, not even an empty one"
    assert sorted(Path(p).parent.name for p in first.pulled) == ["1", "3"]
    assert first.bytes_pulled == 60, "what never reached the mirror is not counted as copied"
    left = {Path(s.path).name: s.reason for s in first.skipped}
    assert left["U"] == left["p"], "the whole time is held back, with one reason"
    assert left["U"].startswith(f"{mirror.EARLIER_TIME} (2/),"), "and the reason names it"

    second = cycle(browser, refused=refused)
    assert second.pulled == [], "and the time is not asked for again this session"
    assert not (case / "2").exists()


def test_a_time_split_across_round_trips_is_not_left_half_copied(backend, store, monkeypatch):
    """#103: `2/T` lands, `2/U` times out in the next round trip, `2/p` is not asked for."""
    monkeypatch.setattr(mirror, "BATCH_BYTES", 30)
    whole(
        backend,
        (f"{HOME}/case/3/U", 30),
        (f"{HOME}/case/2/T", 30), (f"{HOME}/case/2/U", 30), (f"{HOME}/case/2/p", 30),
        (f"{HOME}/case/1/U", 30),
    )
    real = backend.get_tree

    def flaky(paths, local_dir, **kwargs):
        if any(path.endswith("/2/U") for path in paths):
            raise BackendError("timed out", code="timeout")
        return real(paths, local_dir, **kwargs)

    backend.get_tree = flaky
    report = cycle(browser_for(backend, store), refused={})

    case = mirror.local_for(report.local_dir, f"{HOME}/case")
    assert sorted(p.name for p in case.iterdir()) == ["1", "3"]
    assert f"{HOME}/case/2/p" not in backend.fetched, "no round trip for the rest of it"
    assert sorted(Path(p).parent.name for p in report.pulled) == ["1", "3"]
    assert {Path(s.path).name for s in report.skipped} == {"T", "U", "p"}
    assert all(
        s.reason.startswith(f"{mirror.EARLIER_TIME} (2/),") for s in report.skipped
    )


def test_a_time_already_whole_keeps_the_copies_it_has_when_a_fresh_one_fails(backend, store):
    """A restarted solve rewrites times that are already here. `2/` was mirrored
    whole; this cycle's `2/U` lands and its `2/p` times out. Deleting what landed
    would leave `2/` holding only the unchanged `T`, which is worse than the
    half-copied state #103 is about. Nothing is deleted, so it holds all three."""
    whole(
        backend,
        (f"{HOME}/case/3/U", 30),
        (f"{HOME}/case/2/T", 30), (f"{HOME}/case/2/U", 30), (f"{HOME}/case/2/p", 30),
    )
    here = mirror.local_for(store.fetch_dir(), f"{HOME}/case/2")
    here.mkdir(parents=True, exist_ok=True)
    (here / "T").write_bytes(b"x" * 30)  # the size the listing gives: already here
    (here / "U").write_bytes(b"older U")  # a different size: wanted again
    (here / "p").write_bytes(b"older p")
    real = backend.get_tree

    def flaky(paths, local_dir, **kwargs):
        if any(path.endswith("/2/p") for path in paths):
            raise BackendError("timed out", code="timeout")
        return real(paths, local_dir, **kwargs)

    backend.get_tree = flaky
    report = cycle(browser_for(backend, store), refused={})

    assert (here / "U").read_bytes() == b"older U", "the copy that was here is still here"
    assert (here / "p").read_bytes() == b"older p"
    assert (here / "T").read_bytes() == b"x" * 30
    assert [Path(p).parent.name for p in report.pulled] == ["3"]


def test_a_time_a_round_trip_leaves_a_file_out_of_is_not_half_copied(backend, store):
    """A file left out without a failure is the #103 state all the same: `get_tree`
    returns, having skipped the member (`backend/hosted.py:1505` passes over a
    non-regular one), and nothing raised for the cycle to react to."""
    whole(backend, (f"{HOME}/case/3/U", 30), (f"{HOME}/case/2/U", 30), (f"{HOME}/case/2/p", 30))
    real = backend.get_tree

    def leaves_one_out(paths, local_dir, **kwargs):
        return real([path for path in paths if not path.endswith("/2/p")], local_dir, **kwargs)

    backend.get_tree = leaves_one_out
    report = cycle(browser_for(backend, store), refused={})

    case = mirror.local_for(report.local_dir, f"{HOME}/case")
    assert [p.name for p in case.iterdir()] == ["3"], "no 2/U on its own"
    assert report.bytes_pulled == 30
    assert {Path(s.path).name for s in report.skipped} == {"U", "p"}
    assert "came back without 1 of its 2 files" in report.skipped[0].reason


def test_holding_a_time_back_touches_nothing_another_sync_mirrored(backend, store):
    """An `openreynolds pull` and a background cycle can both be fetching `2/U`. The
    cycle's `2/p` failing used to unlink the copy the pull had just reported
    mirrored; a cycle that holds a time back now deletes nothing at all."""
    whole(backend, (f"{HOME}/case/3/U", 30), (f"{HOME}/case/2/U", 30), (f"{HOME}/case/2/p", 30))
    pulled_elsewhere = mirror.local_for(store.fetch_dir(), f"{HOME}/case/2/U")
    real = backend.get_tree

    def flaky(paths, local_dir, **kwargs):
        if any(path.endswith("/2/p") for path in paths):
            # The other sync lands its copy while this round trip is in flight.
            pulled_elsewhere.parent.mkdir(parents=True, exist_ok=True)
            pulled_elsewhere.write_bytes(b"x" * 30)
            raise BackendError("timed out", code="timeout")
        return real(paths, local_dir, **kwargs)

    backend.get_tree = flaky
    cycle(browser_for(backend, store), refused={})

    assert pulled_elsewhere.read_bytes() == b"x" * 30


def test_a_time_with_a_file_left_behind_is_left_whole(backend, store, monkeypatch):
    """A time with half its fields is one the viewer lists and cannot draw."""
    whole(backend, (f"{HOME}/case/2/U", 30), (f"{HOME}/case/1/U", 20), (f"{HOME}/case/1/p", 60))
    report = cycle(browser_for(backend, store), max_file_bytes=50)
    assert [Path(p).parent.name for p in report.pulled] == ["2"]
    assert sorted(Path(s.path).name for s in report.skipped) == ["U", "p"]


def test_each_case_gets_its_turn_at_the_earlier_times(backend, store, monkeypatch):
    """A steady case's times are iteration counts in the thousands; ordering by value
    across cases would hand it every cycle and the transient case nothing."""
    monkeypatch.setattr(mirror, "EARLIER_BYTES_PER_CYCLE", 60)
    whole(
        backend,
        (f"{HOME}/steady/3000/U", 30), (f"{HOME}/steady/2000/U", 30),
        (f"{HOME}/steady/1000/U", 30),
        (f"{HOME}/transient/0.3/U", 30), (f"{HOME}/transient/0.2/U", 30),
        (f"{HOME}/transient/0.1/U", 30),
    )
    report = cycle(browser_for(backend, store))
    earlier = sorted(
        Path(p).parent.relative_to(Path(p).parents[2]).as_posix()
        for p in report.pulled if Path(p).parent.name in ("2000", "0.2", "1000", "0.1")
    )
    assert earlier == ["steady/2000", "transient/0.2"]


def test_numbered_directories_above_a_case_are_not_its_times(backend, store):
    """`sweep/5/case/100/U`: the time judged is 100/, under the case `sweep/5/case`."""
    reason, held = mirror._live_verdict("sweep/5/case/100/U", {"sweep": "5", "sweep/5/case": "200"})
    assert held == ("sweep/5/case", "100")
    assert "100/" in reason


def test_newest_times_reads_the_number_not_the_string(backend, store):
    entries = [
        SimpleNamespace(path=f"{HOME}/case/9/U", size=1, is_dir=False),
        SimpleNamespace(path=f"{HOME}/case/10/U", size=1, is_dir=False),
    ]
    assert mirror.newest_times(entries, HOME) == {"case": "10"}


def test_a_render_is_asked_for_before_the_bulk(backend, store):
    """A 200 KB render the model had just looked at sat behind thousands of smaller
    per-processor files: nine minutes for one picture, and a second that never came."""
    entries = [
        SimpleNamespace(path=f"{HOME}/case/2/U", size=10, is_dir=False),
        SimpleNamespace(path=f"{HOME}/renders/mesh.png", size=200_000, is_dir=False),
        SimpleNamespace(path=f"{HOME}/case/log.simpleFoam", size=50, is_dir=False),
    ]
    order = [Path(e.path).name for batch in mirror._batches(entries) for e in batch]
    assert order[0] == "mesh.png", "the picture goes in the first round trip"
    assert order[1] == "log.simpleFoam", "then what was written down"


def test_the_budget_is_still_spent_smallest_first(backend, store):
    """Ordering the round trips is not the same as choosing the files: running out
    should still cost one enormous picture rather than three hundred dictionaries."""
    workspace(
        backend,
        (f"{HOME}/big.png", 5_000),
        (f"{HOME}/case/system/fvSchemes", 100),
        (f"{HOME}/case/system/controlDict", 100),
    )
    report = mirror.sync(browser_for(backend, store), live=True, max_total_bytes=5_000)
    assert sorted(Path(p).name for p in report.pulled) == ["controlDict", "fvSchemes"]


# -- F-54: a cycle that cost minutes, every cycle, and blocked the tool call ----


def _bad_request(message="cannot resolve path"):
    return BackendError(message, code="bad_request", status=400)


def _workspace_of(backend, names):
    workspace(backend, *((f"{HOME}/{name}", 50) for name in names))


def test_a_refused_path_is_named_and_not_asked_for_again_this_session(backend, store):
    """The service answered a two-minute jail-probe timeout with the same 400 it gives
    a path outside the jail; the mirror read it as one awkward file, asked for it again
    every cycle, and paid the two minutes every cycle. The path was never in the log."""
    from openreynolds.mirror import sync

    names = ("geo/Allrun", "geo/Allmesh", "results.json")
    _workspace_of(backend, names)
    attempts = []
    original = backend.get_tree

    def service(paths, local_dir):
        attempts.append(list(paths))
        if any(p.endswith("geo/Allrun") for p in paths):
            raise _bad_request("cannot resolve paths" if len(paths) > 1 else "cannot resolve path")
        return original(paths, local_dir)

    backend.get_tree = service
    refused: dict[str, str] = {}

    first = sync(browser_for(backend, store), live=True, refused=refused)
    assert len(first.pulled) == 2
    assert f"{HOME}/geo/Allrun" in refused, "a 400 is a fact about the path"
    assert any(f"could not copy {HOME}/geo/Allrun" in w for w in first.warnings), first.warnings

    before = len(attempts)
    second = sync(browser_for(backend, store), live=True, refused=refused)
    assert len(attempts) == before, "the refused path cost no round trip the second time"
    assert not second.pulled and second.unchanged == 2
    left = [skip for skip in second.skipped if skip.path.endswith("geo/Allrun")]
    assert left and "refused" in left[0].reason and "cannot resolve path" in left[0].reason
    assert "openreynolds pull" in second.brief()[0], "the retry is still a command away"


def test_a_transient_failure_is_tried_again_next_cycle(backend, store):
    """Only a 400 is remembered. A dropped connection says nothing about the path."""
    from openreynolds.mirror import sync

    _workspace_of(backend, ("a.md", "b.md"))
    failures = {"n": 0}
    original = backend.get_tree

    def flaky(paths, local_dir):
        if any(p.endswith("a.md") for p in paths) and failures["n"] < 2:
            # The batch, then the single: a blink twice over, and then fine.
            failures["n"] += 1
            raise BackendError("peer closed connection", code="unreachable")
        return original(paths, local_dir)

    backend.get_tree = flaky
    refused: dict[str, str] = {}
    sync(browser_for(backend, store), live=True, refused=refused)
    assert not refused
    report = sync(browser_for(backend, store), live=True, refused=refused)
    assert any(p.name == "a.md" for p in report.pulled)


def test_a_failed_batch_finds_its_culprit_without_a_round_trip_per_file(backend, store):
    """Thirty files one at a time was thirty round trips for one bad file. One at a
    time only until the bad one, then the rest as the batch they were."""
    from openreynolds.mirror import sync

    names = [f"log.{i}" for i in range(8)]
    _workspace_of(backend, names)
    attempts = []
    original = backend.get_tree

    def service(paths, local_dir):
        attempts.append(len(paths))
        if any(p.endswith("log.2") for p in paths):
            raise _bad_request()
        return original(paths, local_dir)

    backend.get_tree = service
    report = sync(browser_for(backend, store), live=True, refused={})

    assert len(report.pulled) == 7
    # The batch, the singles up to and including the culprit, and the remainder once.
    assert attempts == [8, 1, 1, 1, 5], attempts
    assert report.round_trips == len(attempts)


def test_two_culprits_in_one_batch_are_both_found(backend, store):
    from openreynolds.mirror import sync

    names = [f"f{i}" for i in range(6)]
    _workspace_of(backend, names)
    bad = {"f1", "f4"}
    original = backend.get_tree

    def service(paths, local_dir):
        if any(p.rsplit("/", 1)[-1] in bad for p in paths):
            raise _bad_request()
        return original(paths, local_dir)

    backend.get_tree = service
    refused: dict[str, str] = {}
    report = sync(browser_for(backend, store), live=True, refused=refused)
    assert len(report.pulled) == 4
    assert {p.rsplit("/", 1)[-1] for p in refused} == bad


def test_a_report_says_what_the_cycle_cost(backend, store):
    """One line per cycle with the count and the duration, so the next run can show
    whether the cycles are a convenience or where the wall clock went."""
    from openreynolds import mirror
    from openreynolds.mirror import MirrorReport, sync

    _workspace_of(backend, ("a.md", "b.md"))
    report = sync(browser_for(backend, store), live=True)
    assert report.round_trips == 1 and report.seconds >= 0.0
    assert "1 round trip, 0 s" in report.brief()[0]

    slow = MirrorReport(local_dir=Path("."), round_trips=3, seconds=8 * 60 + 5)
    assert slow.brief() == ["mirror: nothing new after 3 round trips, 8 min 5 s"]
    assert MirrorReport(local_dir=Path("."), seconds=2).brief() == []
    assert mirror.took(125) == "2 min 5 s" and mirror.took(180) == "3 min"


def test_the_terminal_names_the_cost_and_the_failed_path_as_they_happen():
    """The batch failure used to reach the log at close-down only, without the path."""
    import io

    from rich.console import Console

    from openreynolds.mirror import MirrorReport
    from openreynolds.view import ConsoleView

    out = io.StringIO()
    view = ConsoleView(console=Console(file=out, width=200, force_terminal=False))
    report = MirrorReport(
        local_dir=Path("here"), round_trips=4, seconds=130,
        warnings=[f"could not copy {HOME}/geo/Allrun: bad_request (400): cannot resolve path"],
    )
    view.mirrored(report)
    text = out.getvalue()
    assert "nothing new after 4 round trips, 2 min 10 s" in text
    assert f"could not copy {HOME}/geo/Allrun" in text

    out.truncate(0)
    out.seek(0)
    view.mirrored(MirrorReport(local_dir=Path("here"), seconds=1.0))
    assert out.getvalue() == "", "an ordinary empty cycle still says nothing"


def test_a_background_cycle_stands_aside_for_a_tool_call(backend, store):
    """The cycle's transfers and the model's command share the container, and a 27 s
    finish step took five minutes behind a cycle. So a cycle waits for the call."""
    import time as _time

    from openreynolds import mirror

    _workspace_of(backend, ("a.md",))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=3600)
    live.start()
    try:
        with live.gate.held():
            live.poke()
            _time.sleep(0.4)
            assert live.last_report is None, "no cycle while the tool call is in flight"
        for _ in range(50):
            if live.last_report is not None:
                break
            _time.sleep(0.05)
        assert live.last_report is not None and live.last_report.pulled
    finally:
        live.stop()


def test_the_wait_for_a_tool_call_is_bounded(backend, store, monkeypatch):
    """A turn of back-to-back tool calls must still get mirrored eventually."""
    import time as _time

    from openreynolds import mirror

    monkeypatch.setattr(mirror, "GATE_WAIT_S", 0.2)
    _workspace_of(backend, ("a.md",))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=3600)
    live.start()
    try:
        with live.gate.held():
            live.poke()
            for _ in range(40):
                if live.last_report is not None:
                    break
                _time.sleep(0.05)
            assert live.last_report is not None, "the cycle went anyway once the bound ran out"
    finally:
        live.stop()


def test_a_waiting_cycle_does_not_outlive_stop(backend, store):
    """Close-down must not sit behind a held gate for two minutes."""
    import time as _time

    from openreynolds import mirror

    _workspace_of(backend, ("a.md",))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=3600)
    live.start()
    live.gate.enter()
    try:
        live.poke()
        _time.sleep(0.2)
        started = _time.monotonic()
        live.stop(timeout=5.0)
        assert _time.monotonic() - started < 2.0
        assert live._thread is None
    finally:
        live.gate.exit()


def test_the_gate_counts_and_clears():
    from openreynolds.mirror import Gate

    gate = Gate()
    assert gate.clear(0.01) and not gate.busy
    with gate.held():
        assert gate.busy
        with gate.held():
            assert gate.busy
        assert gate.busy
        assert not gate.clear(0.05)
        assert not gate.clear(10.0, unless=lambda: True)
    assert not gate.busy and gate.clear(0.01)


def test_the_loop_holds_the_gate_around_a_tool_call(backend, store, view, monkeypatch):
    from openreynolds import loop as loop_module
    from openreynolds.mirror import Gate

    loop = Loop(
        Config(llm_api_key="k", model="claude-opus-5"),
        ToolContext(backend=backend, store=store, max_output=1000),
        store,
        view,
    )
    loop.gate = Gate()
    seen = []

    def dispatch(ctx, name, tool_input, call_id=None):
        seen.append(loop.gate.busy)
        return "ok", False

    monkeypatch.setattr(loop_module, "dispatch", dispatch)
    block = SimpleNamespace(name="bash", input={"cmd": "ls"}, id="t1")
    loop._run_tool(block)
    assert seen == [True] and not loop.gate.busy


def test_a_cycle_listed_by_the_service_does_not_wait_for_a_tool_call(backend, store):
    """The service's files route is not the channel a tool call is on, so a hosted
    cycle has nothing to stand aside for. During a five-minute solve call on staging
    the first listing came 244 s in."""
    import time as _time

    from openreynolds.backend.base import StoredEntry, StoredListing

    backend.files[f"{HOME}/case/4/U"] = b"x" * 50
    backend.list_stored = lambda path, depth: StoredListing(
        [StoredEntry(path=f"{HOME}/case/4/U", is_dir=False, size=50, mtime=1700000000.0)]
    )
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=3600)
    live.start()
    try:
        with live.gate.held():
            live.poke()
            for _ in range(40):
                if live.last_report is not None:
                    break
                _time.sleep(0.05)
            assert live.last_report is not None, "the cycle ran during the tool call"
            assert [Path(p).name for p in live.last_report.pulled] == ["U"]
            assert live.last_report.gate_wait_seconds < 1.0
            assert backend.execs == [], "no command on the workspace while the call ran"
    finally:
        live.stop()


def test_a_walking_cycle_waits_for_a_tool_call_once(backend, store, monkeypatch):
    """Two bounded waits in a row add up: a cycle stood aside 120 s before it began
    and 120 s more before it listed. One bound per cycle, not one per step."""
    import time as _time

    monkeypatch.setattr(mirror, "GATE_WAIT_S", 1.0)
    _workspace_of(backend, ("a.md",))
    live = mirror.LiveMirror(browser_for(backend, store), interval_s=3600)
    live.start()
    try:
        with live.gate.held():
            started = _time.monotonic()
            live.poke()
            for _ in range(200):
                if live.last_report is not None:
                    break
                _time.sleep(0.02)
            took = _time.monotonic() - started
            assert live.last_report is not None
            # Two waits in a row would be 2 s; one is 1 s, with room for a slow host.
            assert took < 1.7, f"waited {took:.2f} s, more than one 1 s bound"
            assert live.last_report.gate_wait_seconds >= 0.9
    finally:
        live.stop()


def test_a_cycle_whose_wait_was_cut_by_stop_does_not_walk(backend, store):
    """The wait ending because the mirror stopped is not the call ending: walking
    then is the very command the wait keeps off the call's channel."""
    _workspace_of(backend, ("a.md",))
    gate = mirror.Gate()
    gate.enter()
    try:
        report = mirror.sync(
            browser_for(backend, store), live=True, background=True, gate=gate,
            stopping=lambda: True,
        )
    finally:
        gate.exit()
    assert backend.execs == [] and report.pulled == []


# -- a workspace on this machine ----------------------------------------------
#
# Measured before the fix below: a `LocalBackend` workspace holding a
# `system/controlDict` and a `plot.png` mirrored NOTHING. `mirror.sync` returned
# `pulled: []` and `studies/<id>/files/` stayed empty. That one fact was the root of
# three separate holes in a local study, all of them downstream of this file:
#
#   * no case bundle, because `casebundle` reads the mirror (`store.fetch_dir()`);
#   * no `studies/<id>/renders/`, because `delivery.Gallery.ingest` surfaces what a
#     sync `pulled`;
#   * no artifacts on the service, because `Gallery` is what sends them -- and before
#     it existed a render reached the platform only if the model happened to call
#     `fetch`, which is the bug it was built for.
#
# The cause was the listing. `Browser.tree` falls back to a walk over `exec`, one
# `find` over the ADVERTISED path, so on this backend it either searched a `/work`
# that does not exist -- under Git Bash on Windows, Git's own root -- or searched the
# right directory and printed every result as a real absolute path that nothing
# downstream can relate back to `/work/<id>`.


def _local_study(tmp_path, files: dict[str, bytes]):
    """A local workspace with these files in the study's own directory, and a store."""
    import os

    from openreynolds.backend.local import LocalBackend
    from openreynolds.store import Store

    work = tmp_path / "work"
    work.mkdir()
    study_id = "20260101-000000-aaaa"
    for name, blob in files.items():
        path = work / study_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    os.environ.pop("OPENREYNOLDS_FOAM_BASHRC", None)
    backend = LocalBackend(root=work, bashrc="")
    store = Store(tmp_path / "studies", study_id)
    store.session.home = f"/work/{study_id}"
    store.save()
    return backend, store


def test_a_local_workspace_lists_itself_in_the_paths_the_protocol_uses(tmp_path):
    """Advertised paths, because that is what `browse._from_store` passes straight
    into `Entry` and what every caller above it believes a workspace path is."""
    backend, _ = _local_study(tmp_path, {"tank/system/controlDict": b"endTime 1;"})

    listed = backend.list_stored("/work/20260101-000000-aaaa", 4)

    assert listed is not None, "the walk's fallback is what was broken"
    assert sorted(e.path for e in listed.entries) == [
        "/work/20260101-000000-aaaa/tank",
        "/work/20260101-000000-aaaa/tank/system",
        "/work/20260101-000000-aaaa/tank/system/controlDict",
    ]
    assert not listed.truncated


def test_a_local_study_is_mirrored_at_all(tmp_path):
    """The whole of it. `pulled` was empty and `files/` stayed empty."""
    backend, store = _local_study(tmp_path, {
        "tank/system/controlDict": b"endTime 1;",
        "tank/log.icoFoam": b"Time = 1\nEnd\n",
    })

    report = mirror.sync(Browser(backend, store, home=store.session.home), live=True)

    assert sorted(p.name for p in report.pulled) == ["controlDict", "log.icoFoam"]
    landed = sorted(p.name for p in store.files_dir.rglob("*") if p.is_file())
    assert landed == ["controlDict", "log.icoFoam"]


def test_a_render_from_a_local_study_reaches_the_folder_and_the_service(tmp_path):
    """The user-visible half. `Gallery` surfaces what a sync pulled and sends the same
    file for keeping, so a study run here can be looked at from the browser."""
    from openreynolds.delivery import Gallery

    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")
    backend, store = _local_study(tmp_path, {"hydro.png": png})
    sent: list[str] = []

    class _Capture:
        def artifact(self, path, kind=None):
            sent.append(path.name)

    gallery = Gallery(store.files_dir, store.renders_dir, capture=_Capture())
    report = mirror.sync(Browser(backend, store, home=store.session.home), live=True)
    gallery.ingest(report)

    assert [p.name for p in report.pulled] == ["hydro.png"]
    assert sorted(p.name for p in store.renders_dir.glob("*")) == ["hydro.png"]
    assert sent == ["hydro.png"], "and the service was sent it"


def test_a_path_outside_the_workspace_is_not_listed(tmp_path):
    """`_resolve` refuses it, and a refusal here has to read as "no listing" rather
    than as an exception out of a convenience the mirror runs on a timer."""
    backend, _ = _local_study(tmp_path, {"tank/system/controlDict": b"x"})

    assert backend.list_stored("/work/../..", 4) is None
    assert backend.list_stored("/work/20260101-000000-aaaa/nope", 4) is None


def test_a_symlink_is_listed_no_further_than_the_walk_would(tmp_path):
    """Deeper symlinks are left alone, so no loop can be walked into."""
    import os

    backend, _ = _local_study(tmp_path, {"tank/system/controlDict": b"x"})
    loop = tmp_path / "work" / "20260101-000000-aaaa" / "tank" / "self"
    try:
        os.symlink(tmp_path / "work" / "20260101-000000-aaaa", loop,
                   target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("this machine will not make a symlink without more privilege")

    listed = backend.list_stored("/work/20260101-000000-aaaa", 4)

    assert listed is not None
    assert not any(e.path.endswith("/self") for e in listed.entries)
