"""The listing on the way out of a session does not start a machine to be taken.

When a hosted session ends -- an idle timeout, or the person pressing End -- the
close-down syncs the study's directory one last time, and a sync begins with a listing:
`Browser.tree`, a `find` over the exec channel. On a workspace the service had already
stopped (its reaper stops an idle workspace after fifteen minutes) a foreground exec
lazy-starts a new machine to run it. In production on 2026-09-21 that was a c7i.2xlarge
adopted from the pool at 02:24:32 for an idle-timed-out session's final sync; it listed
the files and then sat until the reaper took it down again at 02:42 -- eighteen minutes
of instance for one listing whose answer the service holds without one. The service
keeps a copy of the workspace, written at every checkpoint and stop, and once the
workspace is stopped it serves files and archives from that copy already; the listing
was the one call in the close-down that still needed the machine.

What is checked here: that a foreground listing polls first and reads the copy when
nothing is running; that it still starts the machine when there is no copy, because
somebody asking is somebody working; that a running workspace's own answer is used and
the copy is never asked; and that the close-down sync brings a stopped workspace's files
home with no foreground exec at all. Then the hosted half -- the route, the depth cut,
every failure answering None -- and the wiring that tells the workspace which study it
is serving, without which the copy cannot be asked for.
"""

from __future__ import annotations

import json

import httpx2 as httpx
import pytest

from conftest import FakeBackend, ReadyStarter, install_model, message, text_block
from openreynolds import cli, mirror
from openreynolds.backend import hosted as hosted_mod
from openreynolds.backend.base import (
    BackendError,
    ExecResult,
    StoredEntry,
    StoredListing,
)
from openreynolds.backend.hosted import FoamdClient, HostedBackend
from openreynolds.backend.pending import PendingBackend
from openreynolds.browse import MAX_ENTRIES, Browser, Entry, Listing
from openreynolds.config import Config
from openreynolds.loop import Loop

HOME = "/work/study-test"

IDLE = ExecResult(exit_code=-1, output="", truncated=False, log_path=None,
                  stderr="", idle=True)


def as_find_output(*rows: tuple[str, int]) -> str:
    """What the walk prints. A trailing slash on a path means a directory."""
    return "".join(
        f"{'d' if path.endswith('/') else '-'}\t{size}\t1700000000.0\t{path.rstrip('/')}\n"
        for path, size in rows
    )


def stored(*rows: tuple[str, int], truncated: bool = False) -> StoredListing:
    """The same rows as the service's copy answers them."""
    return StoredListing(
        [
            StoredEntry(path=path.rstrip("/"), is_dir=path.endswith("/"), size=size,
                        mtime=1700000000.0)
            for path, size in rows
        ],
        truncated=truncated,
    )


class StoredWorkspace(FakeBackend):
    """A hosted workspace as the close-down finds it: stopped, with the service's copy
    of its files to hand.

    A poll finds nothing running. Ordinary work starts a machine -- counted in
    `machine_starts`, because that count is the whole of what the fix is measured
    by -- and from then on the workspace is up and answers `running_output`. A test
    that means to prove no machine was started asserts on the count; one that means
    to prove the old fallback still starts one asserts on it too."""

    def __init__(self, copy: StoredListing | None = None, *, running_output: str = ""):
        super().__init__()
        self.copy = copy
        self.copy_asked: list[tuple[str, int]] = []
        self.running = False
        self.running_output = running_output
        self.machine_starts = 0

    def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
        self.execs.append(cmd)
        self.exec_background.append(background)
        if not self.running:
            if background:
                return IDLE
            self.machine_starts += 1
            self.running = True
        return ExecResult(0, self.running_output, False, None)

    def list_stored(self, path, depth):
        self.copy_asked.append((path, depth))
        return self.copy


# -- the browser ---------------------------------------------------------------


def test_a_stopped_workspace_is_listed_from_the_copy_and_no_machine_is_started(store):
    backend = StoredWorkspace(stored(
        (f"{HOME}/case/", 4096),
        (f"{HOME}/case/log.simpleFoam", 120),
        (f"{HOME}/renders/", 4096),
        (f"{HOME}/renders/mesh.png", 90_000),
        (f"{HOME}/README.md", 9),
        truncated=True,
    ))

    listing = Browser(backend, store, home=HOME).tree(HOME, depth=4)

    assert backend.exec_background == [True], "one poll, and nothing sent as work"
    assert backend.machine_starts == 0
    assert backend.copy_asked == [(HOME, 4)], "the copy was asked for the same path and depth"
    assert isinstance(listing, Listing)
    assert [e.path for e in listing] == [
        f"{HOME}/case", f"{HOME}/renders", f"{HOME}/README.md",
        f"{HOME}/case/log.simpleFoam", f"{HOME}/renders/mesh.png",
    ], "in the walk's order: directories first, then files, parents before children"
    assert all(isinstance(e, Entry) for e in listing)
    picture = next(e for e in listing if e.name == "mesh.png")
    assert (picture.size, picture.mtime, picture.is_dir) == (90_000, 1700000000.0, False)
    assert listing.truncated is True, "the copy's own cap is carried"
    assert (listing.limit, listing.root, listing.depth) == (MAX_ENTRIES, HOME, 4)
    assert "capped" in listing.notice


def test_with_no_copy_to_read_the_listing_starts_the_machine_as_it_always_did(store):
    """Somebody asking for a listing is somebody working, and a workspace with no copy
    to read has exactly one place the answer can come from."""
    backend = StoredWorkspace(None, running_output=as_find_output((f"{HOME}/notes.md", 9)))

    listing = Browser(backend, store, home=HOME).tree(HOME)

    assert backend.exec_background == [True, False], "the poll found nothing; the work followed"
    assert backend.machine_starts == 1
    assert [e.path for e in listing] == [f"{HOME}/notes.md"]
    assert not listing.truncated


def test_a_running_workspace_answers_itself_and_the_copy_is_never_asked(store):
    """The copy is written at checkpoints and stops; a workspace that is up may be
    ahead of it, and its own answer is the one to use."""
    backend = StoredWorkspace(stored((f"{HOME}/stale.md", 1)),
                              running_output=as_find_output((f"{HOME}/fresh.md", 2)))
    backend.running = True

    listing = Browser(backend, store, home=HOME).tree(HOME)

    assert backend.exec_background == [True], "the poll ran, and its output is the listing"
    assert backend.copy_asked == []
    assert [e.name for e in listing] == ["fresh.md"]


def test_a_background_poll_of_a_stopped_workspace_still_refuses_to_guess(store):
    """The unattended cycle is unchanged: it runs every twenty seconds for as long as
    the process lives, and reading the copy on that clock would be the same listing
    of the same stopped workspace ~150 times an hour. It waits, as before."""
    backend = StoredWorkspace(stored((f"{HOME}/notes.md", 9)))

    with pytest.raises(BackendError) as caught:
        Browser(backend, store, home=HOME).tree(HOME, background=True)

    assert caught.value.code == "workspace_idle"
    assert backend.copy_asked == [] and backend.machine_starts == 0


def test_the_copy_is_held_to_the_same_cap_and_cut_breadth_first(store):
    """The copy answers up to 5,000 entries in its own order; this listing keeps 4,000.
    The cut is applied here as the walk applies it -- shallow entries first -- so a
    copy that lists the solver's bulk before the pictures still loses the bulk."""
    deep = [(f"{HOME}/run/processors4/{t}/U", 8) for t in range(MAX_ENTRIES + 5)]
    rows = deep + [(f"{HOME}/README.md", 9), (f"{HOME}/renders/shedding.gif", 90)]
    backend = StoredWorkspace(stored(*rows))

    listing = Browser(backend, store, home=HOME).tree(HOME, depth=6)

    assert len(listing) == MAX_ENTRIES
    assert listing.truncated
    listed = {e.path for e in listing}
    assert f"{HOME}/README.md" in listed and f"{HOME}/renders/shedding.gif" in listed
    assert f"{HOME}/run/processors4/{MAX_ENTRIES + 4}/U" not in listed, "the cut fell in the bulk"


def test_a_copy_that_fits_is_not_called_truncated(store):
    backend = StoredWorkspace(stored((f"{HOME}/notes.md", 9)))
    listing = Browser(backend, store, home=HOME).tree(HOME)
    assert not listing.truncated and listing.notice == ""


def test_a_stand_in_without_the_method_lists_exactly_as_before(store):
    """A backend that predates `list_stored` -- an embedder's, a test's -- is not asked
    to poll: its foreground listing is one foreground exec, as it always was."""
    calls = []

    class Bare:
        workspace_root = "/work"

        def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
            calls.append(background)
            return ExecResult(0, as_find_output((f"{HOME}/notes.md", 9)), False, None)

    listing = Browser(Bare(), store, home=HOME).tree(HOME)

    assert calls == [False]
    assert [e.name for e in listing] == ["notes.md"]


# -- the close-down sync -------------------------------------------------------


def test_the_close_down_sync_brings_a_stopped_workspace_home_without_a_machine(store):
    """The consequence the incident asked for. `sync_now` is the close-down's sync;
    on a stopped workspace its listing comes from the copy and its files through
    `get_tree`, which the service serves from the same copy -- so the whole sync stays
    off the machine, and the machine stays down."""
    backend = StoredWorkspace(stored(
        (f"{HOME}/case/", 4096),
        (f"{HOME}/case/log.simpleFoam", 120),
        (f"{HOME}/renders/", 4096),
        (f"{HOME}/renders/mesh.png", 90_000),
        (f"{HOME}/README.md", 9),
    ))
    for path in (f"{HOME}/case/log.simpleFoam", f"{HOME}/renders/mesh.png", f"{HOME}/README.md"):
        backend.files[path] = b"x"
    live = mirror.LiveMirror(Browser(backend, store, home=HOME), interval_s=0)

    report = live.sync_now()

    assert backend.machine_starts == 0, "nothing was started to take the listing"
    assert backend.exec_background == [True], "one poll was the only command sent"
    assert not backend.exec_background.count(False)
    assert not report.warnings, report.warnings
    assert {p.name for p in report.pulled} == {"log.simpleFoam", "mesh.png", "README.md"}
    assert sorted(backend.fetched) == sorted(
        [f"{HOME}/case/log.simpleFoam", f"{HOME}/renders/mesh.png", f"{HOME}/README.md"]
    ), "the files came through get_tree"
    assert (store.files_dir / "study-test" / "renders" / "mesh.png").is_file()
    assert live.browser.cached(HOME) is not None, "the listing was remembered like any other"


def test_the_close_down_sync_says_when_the_copy_was_cut_short(store):
    backend = StoredWorkspace(stored((f"{HOME}/README.md", 9), truncated=True))
    backend.files[f"{HOME}/README.md"] = b"x"

    report = mirror.LiveMirror(Browser(backend, store, home=HOME), interval_s=0).sync_now()

    assert any("not looked at" in line for line in report.warnings)
    assert backend.machine_starts == 0


# -- the hosted backend ----------------------------------------------------------


def _backend_against(handler) -> tuple[HostedBackend, list[httpx.Request]]:
    """A `HostedBackend` on a stub service, built as `test_hosted.py` builds one, with
    every request it made kept for the test to read."""
    seen: list[httpx.Request] = []

    def recording(request):
        seen.append(request)
        return handler(request)

    client = FoamdClient("https://svc.example", "of_live_test")
    client._client = httpx.Client(base_url="https://svc.example",
                                  transport=httpx.MockTransport(recording),
                                  follow_redirects=False)
    backend = HostedBackend(client, "inst-1")
    backend.study_id = "st-1"
    return backend, seen


def _workspace_route(entries, *, truncated=False, root="/work/st-1"):
    return lambda request: httpx.Response(200, json={
        "root": root, "entries": entries, "truncated": truncated,
    })


A_STUDY = [
    {"path": "/work/st-1/case", "is_dir": True, "size": 0, "mtime": 1700000000},
    {"path": "/work/st-1/case/system", "is_dir": True, "size": 0, "mtime": 1700000001},
    {"path": "/work/st-1/case/system/controlDict", "is_dir": False, "size": 900, "mtime": 1700000002},
    {"path": "/work/st-1/case/log.simpleFoam", "is_dir": False, "size": 120, "mtime": 1700000003},
    {"path": "/work/st-1/README.md", "is_dir": False, "size": 9, "mtime": 1700000004},
]
"""The route's answer: the whole tree, workspace-absolute, not depth-limited."""


def test_list_stored_asks_the_studys_workspace_route_for_the_whole_tree():
    backend, seen = _backend_against(_workspace_route(A_STUDY))

    backend.list_stored("/work/st-1", 4)

    (request,) = seen
    assert request.method == "GET"
    assert request.url.path == "/v1/studies/st-1/workspace"
    assert dict(request.url.params) == {
        "path": "/work/st-1", "recursive": "1", "limit": str(hosted_mod.STORED_LISTING_LIMIT),
    }, "the path as given, the whole tree, at the route's own ceiling"


def test_list_stored_cuts_the_tree_to_the_depth_asked():
    """The route is not depth-limited and `Browser.tree` is, so the cut is made here,
    counted as `find -maxdepth` counts: the path's own children are depth 1."""
    backend, _ = _backend_against(_workspace_route(A_STUDY))

    to_one = backend.list_stored("/work/st-1", 1)
    to_two = backend.list_stored("/work/st-1", 2)
    to_four = backend.list_stored("/work/st-1", 4)

    assert [e.path for e in to_one.entries] == ["/work/st-1/case", "/work/st-1/README.md"]
    assert [e.path for e in to_two.entries] == [
        "/work/st-1/case", "/work/st-1/case/system", "/work/st-1/case/log.simpleFoam",
        "/work/st-1/README.md",
    ]
    assert len(to_four.entries) == len(A_STUDY), "everything is within four"
    assert not to_four.truncated


def test_list_stored_carries_the_four_facts_the_walk_prints():
    backend, _ = _backend_against(_workspace_route(A_STUDY))

    listing = backend.list_stored("/work/st-1", 4)

    control_dict = next(e for e in listing.entries if e.path.endswith("controlDict"))
    assert control_dict == StoredEntry(path="/work/st-1/case/system/controlDict",
                                       is_dir=False, size=900, mtime=1700000002.0)
    assert next(e for e in listing.entries if e.path.endswith("/case")).is_dir is True


def test_list_stored_counts_depth_from_the_path_asked_not_the_study_root():
    """A listing of `case/` is what `find -H /work/st-1/case -maxdepth 1` would say:
    `system/` and the log, not the dictionary two levels down."""
    backend, seen = _backend_against(_workspace_route(A_STUDY))

    listing = backend.list_stored("/work/st-1/case", 1)

    assert dict(seen[0].url.params)["path"] == "/work/st-1/case"
    assert [e.path for e in listing.entries] == [
        "/work/st-1/case/system", "/work/st-1/case/log.simpleFoam",
    ]


def test_list_stored_anchors_a_relative_path_at_the_root_the_route_names():
    """Both spellings are in circulation on the route; the entries it answers are
    absolute either way, so a relative ask is measured from the study's own root."""
    backend, _ = _backend_against(_workspace_route(A_STUDY))

    listing = backend.list_stored("case", 1)

    assert [e.path for e in listing.entries] == [
        "/work/st-1/case/system", "/work/st-1/case/log.simpleFoam",
    ]


def test_list_stored_carries_the_routes_own_truncation():
    backend, _ = _backend_against(_workspace_route(A_STUDY, truncated=True))
    assert backend.list_stored("/work/st-1", 1).truncated is True


def test_list_stored_without_a_study_id_asks_nothing_and_answers_none():
    """No id, no row to read the copy under. The backend has not been told which study
    it serves -- capture off, or a session from before the wiring -- and the honest
    answer is "ask the machine", not a guess at the id."""
    backend, seen = _backend_against(_workspace_route(A_STUDY))
    backend.study_id = None

    assert backend.list_stored("/work/st-1", 4) is None
    assert seen == []


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            lambda r: httpx.Response(400, json={"error": "bad_request",
                                                 "message": "path is outside the study: /work"}),
            id="400-outside-the-study",
        ),
        pytest.param(
            lambda r: httpx.Response(404, json={"error": "not_found", "message": "workspace"}),
            id="404-never-written-or-no-route",
        ),
        pytest.param(
            lambda r: (_ for _ in ()).throw(httpx.ConnectError("no route to host")),
            id="network",
        ),
        pytest.param(
            lambda r: httpx.Response(200, json={"root": "/work/st-1", "entries": "not a list"}),
            id="a-body-that-is-not-the-shape",
        ),
    ],
)
def test_list_stored_answers_none_rather_than_raising(answer, monkeypatch):
    """Every failure is the same answer, because every failure has the same fallback:
    the foreground exec that was always there. What must not happen is a listing
    turning into an exception on the way out of a session."""
    monkeypatch.setattr(hosted_mod.time, "sleep", lambda _s: None)  # the network case retries
    backend, _ = _backend_against(answer)

    assert backend.list_stored("/work/st-1", 4) is None


def test_a_400_or_404_is_not_retried_on_the_way_out():
    """The two refusals are facts about the ask. Asking again costs the person leaving
    a round trip per attempt for the same answer."""
    for status, code in ((400, "bad_request"), (404, "not_found")):
        backend, seen = _backend_against(
            lambda r, s=status, c=code: httpx.Response(s, json={"error": c, "message": "no"}))
        backend.list_stored("/work/st-1", 4)
        assert len(seen) == 1, f"a {status} was asked again"


def test_the_client_route_is_the_one_the_service_publishes():
    """`FoamdClient.list_workspace` on its own: the answer is handed back whole, so a
    caller other than `list_stored` can read `root` and the raw entries."""
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"root": "/work/st-1", "entries": [], "truncated": False})

    client = FoamdClient("https://svc.example", "of_live_test")
    client._client = httpx.Client(base_url="https://svc.example",
                                  transport=httpx.MockTransport(handler), follow_redirects=False)

    body = client.list_workspace("st-1", "case/system")

    assert body == {"root": "/work/st-1", "entries": [], "truncated": False}
    assert seen[0].url.path == "/v1/studies/st-1/workspace"
    assert dict(seen[0].url.params)["path"] == "case/system"


# -- the stand-in for a workspace still coming up ---------------------------------


def test_a_pending_workspace_holds_the_study_id_for_the_live_one():
    """The study's row is opened while the machine is still starting, so the id
    reaches the stand-in first. It is handed on at `resolve`, and set straight
    through once the live backend is there."""
    pending = PendingBackend("iid-1")
    pending.study_id = "st-1"
    live = StoredWorkspace(stored((f"{HOME}/notes.md", 9)))

    assert pending.study_id == "st-1"
    assert pending.list_stored(HOME, 4) is None, "no live backend yet: nothing to read, no wait"
    pending.resolve(live)
    assert live.study_id == "st-1", "handed on with the workspace"
    assert pending.list_stored(HOME, 4) is live.copy
    assert live.copy_asked == [(HOME, 4)]

    pending.study_id = "st-2"
    assert live.study_id == "st-2", "and set straight through from here"


def test_a_pending_workspace_does_not_wait_for_a_copy():
    """A poll answers `idle` at once on a stand-in, and the copy must not be the call
    that then waits ten minutes for a machine: None sends the caller to the
    foreground exec, which waits exactly as it always did."""
    pending = PendingBackend("iid-1")
    pending.study_id = "st-1"

    assert pending.list_stored(HOME, 4) is None
    assert not pending.ready_event.is_set()


# -- the session tells the workspace which study it serves ----------------------


class _StudyService:
    """The capture plane, answering from memory: opens the study under the id it is
    given and remembers what it was asked."""

    def __init__(self):
        self.opened: list[dict] = []
        self.closed = 0

    def create_study(self, title, instance_id, study_id=None, home=None):
        self.opened.append({"title": title, "instance_id": instance_id,
                            "study_id": study_id, "home": home})
        return study_id or "remote-1"

    def get_study(self, study_id):
        return {"id": study_id, "home": f"/work/{study_id}", "instance_id": "iid-1",
                "title": "t"}

    def post_messages(self, study_id, messages):
        pass

    def post_result(self, study_id, payload):
        pass

    def post_artifact(self, study_id, filename, data, kind=None):
        pass

    def close(self):
        self.closed += 1


def _reserved_with(service, backend):
    """`reserve` as the session calls it, handing back a service to capture through
    and a workspace that is up the moment it is asked."""

    def reserve(url, key, iid=None):
        return service, "iid-1", ReadyStarter(backend, "iid-1")

    return reserve


def _capturing_config(tmp_path) -> Config:
    return Config(
        foamd_url="https://svc.example", foamd_api_key="of_live_key", llm_api_key="sk-test",
        model="claude-opus-5", studies_dir=tmp_path / "studies", capture=True, desk=False,
        mesh_tool=False, mirror_interval_s=0.0,
    )


def _one_turn_loop(made):
    class OneTurn(Loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_model(self, [message([text_block("done")])])
            made.append(self)

    return OneTurn


@pytest.fixture
def quiet_console(monkeypatch):
    import os

    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(file=open(os.devnull, "w"), force_terminal=False))


def test_the_session_binds_the_studys_id_onto_the_workspace_it_opens(
    tmp_path, monkeypatch, quiet_console
):
    """The row is opened while the workspace is a stand-in; the id must still be on
    the live backend by the time the close-down lists, or the copy is never asked."""
    backend = FakeBackend()
    service = _StudyService()
    monkeypatch.setattr(cli.hosted, "reserve", _reserved_with(service, backend))
    monkeypatch.setattr(cli, "Loop", _one_turn_loop([]))

    cli.session(_capturing_config(tmp_path), study_id=None, instance_id=None,
                one_shot="go", plain=True)

    (opened,) = service.opened
    assert opened["study_id"], "the row was opened under the study's own id"
    assert backend.study_id == opened["study_id"], (
        "the live workspace was told which study it serves, under the platform's name")


def test_a_resumed_session_binds_the_id_the_study_was_opened_under(
    tmp_path, monkeypatch, quiet_console
):
    """A resume opens no row: the id comes from the local record (or from the platform,
    `_recover_session`), and reaches the workspace the same way."""
    service = _StudyService()
    first = FakeBackend()
    monkeypatch.setattr(cli.hosted, "reserve", _reserved_with(service, first))
    monkeypatch.setattr(cli, "Loop", _one_turn_loop([]))
    cfg = _capturing_config(tmp_path)
    cli.session(cfg, study_id=None, instance_id=None, one_shot="go", plain=True)
    (study_dir,) = [p for p in (tmp_path / "studies").iterdir() if p.is_dir()]
    recorded = json.loads((study_dir / "session.json").read_text(encoding="utf-8"))
    assert recorded["remote_study_id"] == first.study_id

    second = FakeBackend()
    monkeypatch.setattr(cli.hosted, "reserve", _reserved_with(service, second))
    cli.session(cfg, study_id=study_dir.name, instance_id=None, one_shot="go", plain=True)

    assert len(service.opened) == 1, "a resume opens no second row"
    assert second.study_id == recorded["remote_study_id"]


def test_a_study_with_no_row_tells_the_workspace_nothing(tmp_path, monkeypatch, quiet_console):
    """Capture off, never on: there is no copy kept under any name, and the workspace
    lists as it always did rather than asking under a guessed one."""
    backend = FakeBackend()
    monkeypatch.setattr(cli.hosted, "reserve", _reserved_with(_StudyService(), backend))
    monkeypatch.setattr(cli, "Loop", _one_turn_loop([]))
    cfg = _capturing_config(tmp_path)
    cfg.capture = False

    cli.session(cfg, study_id=None, instance_id=None, one_shot="go", plain=True)

    assert backend.study_id is None
