"""A capped listing says it was capped, and says at what.

The shape that caused this is the one built here: a study tree with a dotted
bookkeeping directory beside it holding thousands of command logs. `find` walks in
directory order, the logs eat the budget, and what came back looked like a complete
picture of a workspace whose study tree simply ended.

Two things are asserted together, because either alone is the bug. The listing must
carry the fact, and the fact must reach something a person reads -- a flag nothing
renders is the original defect wearing a data structure. And the dotted directory must
still be *in* the listing: excluding it was proposed and rejected, on the ground that a
browser which silently omits a directory that exists is worse than one that admits it
ran out of room, and the person most likely to be looking there is somebody debugging
the thing that fills it.

The second half of the file is the other cap, the one the walk cannot see: the exec
channel's own limit on a command's output, applied by bytes and mid-line. On
2026-09-21 it cut a 127 KB listing at 64 KB and left a partial row that parsed as a
1.5 MB file at the study root. Where the walk still runs -- the local backend, and the
fallback when the service has no listing to give -- a cut output must lose rows, never
gain one, and must say it was cut.
"""

from __future__ import annotations

from openreynolds import mirror
from openreynolds.backend.base import ExecResult
from openreynolds.browse import DEFAULT_DEPTH, MAX_ENTRIES, Browser, Entry, Listing, _parse

HOME = "/work/20260905-101500-ab12"

EXEC_LOGS = 2_993
"""What was actually under one instance's `.foamd/exec` when this was found."""


def a_real_workspace(exec_logs: int = EXEC_LOGS, cases: int = 40, files: int = 60):
    """Paths in the order `find -H` produces them: parents first, siblings in turn.

    The dotted directory comes first, as it does on the volume, which is exactly why
    it is the thing that pushes the study tree past the cap.
    """
    paths: list[tuple[str, bool]] = [("/work/.foamd", True), ("/work/.foamd/exec", True)]
    paths += [(f"/work/.foamd/exec/{n:06d}.log", False) for n in range(exec_logs)]
    paths.append((HOME, True))
    for case in range(cases):
        paths.append((f"{HOME}/case{case:03d}", True))
        paths += [(f"{HOME}/case{case:03d}/f{n:03d}", False) for n in range(files)]
    return paths


def as_find_output(paths) -> str:
    return "".join(
        f"{'d' if is_dir else '-'}\t4096\t1700000000.0\t{path}\n" for path, is_dir in paths
    )


class WalkingBackend:
    """Answers `find` from a fixed tree, honouring `-maxdepth`, the breadth-first
    `sort -n -s` and the `head` cap.

    A stub that ignored the pipeline would let the truncation be asserted against a
    fiction. This one applies the same knobs the real command does, so narrowing
    `path` or `depth` genuinely changes what comes back, and so does the order the
    listing is cut in: a stable sort on depth, like `sort -n -s` over `find`'s `%d`.
    """

    def __init__(self, paths):
        self.paths = paths
        self.calls: list[str] = []

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 60,
             *, background: bool = False):
        # `background` is the poll flag the mirror's timed cycle sets; this stub has no
        # lifecycle to protect, so it runs the listing either way. Accepted rather than
        # ignored so the signature stays the one `Browser.tree` actually calls.
        self.calls.append(cmd)
        root = cmd.split()[2].strip("'")
        depth = int(cmd.split("-maxdepth ")[1].split()[0])
        cap = int(cmd.rsplit("head -n ", 1)[1].strip())
        base = root.rstrip("/").count("/")
        kept = [
            (path, is_dir)
            for path, is_dir in self.paths
            if (path == root or path.startswith(root.rstrip("/") + "/"))
            and path != root
            and path.count("/") - base <= depth
        ]
        if "| sort -n -s |" in cmd:
            kept.sort(key=lambda item: item[0].count("/") - base)  # stable, like sort -s
        return ExecResult(0, as_find_output(kept[:cap]), False, None)


def browser_for(paths, home: str = "/work") -> Browser:
    return Browser(WalkingBackend(paths), home=home)


# -- the listing says it was cut short -----------------------------------------


def test_the_exec_logs_bury_the_study_tree():
    """The premise. Without this the rest of the file proves nothing.

    The logs still eat the budget -- the listing is capped -- but the cap now falls
    among the deepest entries, so what it costs is the tail of the case *files*, not
    the study's directories: every case is in the listing, and the walk plainly says
    it was cut."""
    listing = browser_for(a_real_workspace()).tree("/work", depth=DEFAULT_DEPTH)

    assert len(listing) == MAX_ENTRIES
    paths = {e.path for e in listing}
    assert f"{HOME}/case000" in paths, "the walk should reach the study tree at all"
    assert f"{HOME}/case039" in paths, "a shallow directory is never what the cap costs"
    assert f"{HOME}/case039/f059" not in paths, "the tail was supposed to be cut off"
    assert listing.truncated


def a_transient_study(times: int = 600, fields: int = 8, frames: int = 201):
    """The study that found this: a 4-rank transient solve, its per-time field files
    under `run/processors4/` (9,301 entries under `run/` on the day), with the pictures
    the model made beside them. In `find`'s directory order `run/` came first and its
    walk alone passed the cap."""
    paths: list[tuple[str, bool]] = [(HOME, True), (f"{HOME}/run", True),
                                     (f"{HOME}/run/processors4", True)]
    for t in range(times):
        paths.append((f"{HOME}/run/processors4/{t}", True))
        paths += [(f"{HOME}/run/processors4/{t}/field{n}", False) for n in range(fields)]
    paths.append((f"{HOME}/run/log.pimpleFoam", False))
    paths.append((f"{HOME}/frames", True))
    paths += [(f"{HOME}/frames/frame_{n:03d}.png", False) for n in range(frames)]
    paths.append((f"{HOME}/renders", True))
    paths += [(f"{HOME}/renders/{name}", False)
              for name in ("forces.png", "shedding.gif", "shedding_small.gif", "vorticity_z_0.8.png")]
    paths.append((f"{HOME}/README.md", False))
    paths.append((f"{HOME}/make_gif.py", False))
    return paths


def test_the_solver_bulk_no_longer_hides_the_pictures():
    """`renders/shedding.gif`, `README.md` and the 201 frames were written, looked at
    and described to the person -- and were past the cap in every listing, because the
    walk had spent its 4,000 entries inside `run/processors4/` before reaching them.
    Never listed, never mirrored: the page for the finished study showed `run/` alone.
    Breadth-first, the shallow files are listed first and the cut lands in the bulk."""
    paths = a_transient_study()
    assert len(paths) > MAX_ENTRIES, "the premise: this study does not fit"
    listing = browser_for(paths, home=HOME).tree(HOME, depth=DEFAULT_DEPTH)

    listed = {e.path for e in listing}
    assert listing.truncated
    for wanted in (f"{HOME}/README.md", f"{HOME}/make_gif.py", f"{HOME}/renders/shedding.gif",
                   f"{HOME}/renders/forces.png", f"{HOME}/frames/frame_200.png",
                   f"{HOME}/run/log.pimpleFoam"):
        assert wanted in listed, wanted
    deep = [p for p in listed if "/run/processors4/" in p and p.count("/") == HOME.count("/") + 4]
    assert len(deep) < 600 * 8, "the cut fell where it should: among the per-time field files"


def test_a_capped_listing_says_so_and_says_at_what():
    listing = browser_for(a_real_workspace()).tree("/work")

    assert listing.truncated
    assert f"{MAX_ENTRIES:,}" in listing.notice
    assert "/work" in listing.notice
    assert str(DEFAULT_DEPTH) in listing.notice


def test_the_notice_reaches_what_is_drawn():
    """A flag nothing renders is the same bug again."""
    drawn = browser_for(a_real_workspace()).tree("/work").lines()

    assert any("capped" in line for line in drawn)
    assert drawn[-1] == browser_for(a_real_workspace()).tree("/work").notice


def test_the_notice_reaches_the_mirror_report_a_user_reads(backend, store):
    """`or pull` prints `report.lines()`, so this is a surface a person reads today.

    It is asserted here rather than only in `test_mirror.py` because the truncation is
    now decided in `browse.py`: if that stops being detectable, this is the report that
    goes quiet."""
    capped = a_real_workspace()[: MAX_ENTRIES + 1]
    backend.exec_result = ExecResult(0, as_find_output(capped), False, None)

    report = mirror.sync(Browser(backend, store, home="/work"), path="/work")

    assert any("not looked at" in line for line in report.lines())


def test_a_listing_that_fits_says_nothing():
    """Silence has to mean something, so it is only allowed when the walk finished."""
    listing = browser_for(a_real_workspace(exec_logs=10, cases=2, files=2)).tree("/work")

    assert not listing.truncated
    assert listing.notice == ""
    assert not any("capped" in line for line in listing.lines())


def test_exactly_the_cap_is_not_a_truncation():
    """`head -n N` returning N lines used to be the only signal, and it is ambiguous:
    a workspace of exactly N entries is complete. One line more is asked for so the
    answer is measured."""
    paths = [(f"/work/f{n:06d}", False) for n in range(MAX_ENTRIES)]

    listing = browser_for(paths).tree("/work")

    assert len(listing) == MAX_ENTRIES
    assert not listing.truncated


def test_one_entry_past_the_cap_is():
    paths = [(f"/work/f{n:06d}", False) for n in range(MAX_ENTRIES + 1)]

    listing = browser_for(paths).tree("/work")

    assert len(listing) == MAX_ENTRIES, "the extra line is a probe, not an entry"
    assert listing.truncated


# -- the knobs the user already has recover the tree ---------------------------


def test_narrowing_the_path_recovers_the_study_tree():
    browser = browser_for(a_real_workspace())
    assert browser.tree("/work").truncated

    listing = browser.tree(HOME)

    assert not listing.truncated
    cases = {e.path for e in listing if e.is_dir}
    assert len(cases) == 40, "every case directory should be back"


def test_lowering_the_depth_recovers_the_study_tree():
    browser = browser_for(a_real_workspace())

    listing = browser.tree("/work", depth=2)

    assert not listing.truncated
    assert f"{HOME}/case039" in {e.path for e in listing}


def test_the_notice_names_the_depth_that_was_walked():
    """So a person told to lower it knows what they are lowering from."""
    notice = browser_for(a_real_workspace()).tree("/work", depth=6).notice

    assert "depth 6" in notice


# -- what was rejected, and must not creep back in -----------------------------


def test_the_dotted_directory_is_never_filtered_out():
    """Excluding it was proposed and rejected. This is the record of that."""
    listing = browser_for(a_real_workspace()).tree("/work")

    assert "/work/.foamd" in {e.path for e in listing}
    assert "/work/.foamd/exec" in {e.path for e in listing}
    assert any(e.path.startswith("/work/.foamd/exec/") for e in listing)


def test_it_is_still_there_in_a_listing_with_room_for_everything():
    """The truncated case could hide the omission behind the cap; this one cannot."""
    listing = browser_for(a_real_workspace(exec_logs=5, cases=2, files=2)).tree("/work")

    assert not listing.truncated
    logs = [e.path for e in listing if e.path.startswith("/work/.foamd/exec/")]
    assert len(logs) == 5


def test_nothing_was_dropped_from_the_walk_itself():
    """The cap is the only thing that removes anything, and it removes the tail --
    not a directory chosen in advance."""
    browser = browser_for(a_real_workspace())
    browser.tree("/work")

    command = browser.backend.calls[0]
    for exclusion in ("-prune", "-not", "-name", "! -path", "grep -v"):
        assert exclusion not in command, f"the walk grew a filter: {exclusion}"


# -- the remembered copy tells the same story ----------------------------------


def test_a_remembered_capped_listing_is_still_capped():
    """The files pane is drawn from the mirror's remembered listing most of the time.
    Drawn from memory it must not look more complete than it was."""
    browser = browser_for(a_real_workspace())
    listing = browser.tree("/work")
    browser.remember("/work", listing)

    assert browser.cached("/work").truncated
    assert "capped" in browser.cached("/work").notice


def test_a_subtree_of_a_capped_listing_is_capped_too():
    """`find` stopped part-way through the whole walk, so what is missing from any
    branch of it is unknown."""
    browser = browser_for(a_real_workspace())
    browser.remember("/work", browser.tree("/work"))

    assert browser.cached(HOME).truncated


def test_a_remembered_complete_listing_says_nothing():
    browser = browser_for(a_real_workspace(exec_logs=3, cases=1, files=1))
    browser.remember("/work", browser.tree("/work"))

    assert not browser.cached("/work").truncated
    assert browser.cached("/work").notice == ""


def test_a_plain_list_can_still_be_remembered():
    """`remember` is called with whatever a caller has; it must not require a
    `Listing` and must not invent a truncation for a plain list."""
    browser = browser_for(a_real_workspace())
    browser.remember("/work", list(browser.tree(HOME)))

    assert isinstance(browser.cached("/work"), Listing)
    assert not browser.cached("/work").truncated


# -- the exec channel's own cap: by bytes, mid-line ----------------------------

STUDY = "/work/20260921-033356-076b"
"""The study whose listing found this: 1,532 rows at ~83 bytes each, ~127 KB, against
a 64 KB cap on what a command may print back."""


class CappedChannel:
    """A backend whose exec channel caps a command's output by bytes and cuts it
    mid-line, exactly as the hosted daemon does (`out[:cap]`, and `truncated: true`
    on the answer) and as the local backend does (`blob[:MAX_OUTPUT_BYTES]`). Answers
    one fixed walk, whatever it is asked, and keeps no copy of the workspace -- so the
    walk is what `Browser.tree` has to work from."""

    workspace_root = "/work"

    def __init__(self, output: str, cap: int):
        self.output = output
        self.cap = cap
        self.calls: list[str] = []

    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 60,
             *, background: bool = False):
        self.calls.append(cmd)
        return ExecResult(0, self.output[: self.cap], len(self.output) > self.cap, None)


def rows(*items: tuple[str, int, str]) -> str:
    """What the walk prints: `%y`, size, mtime, path -- a real `find` says `f` for a
    file, and this half of the file does too."""
    return "".join(f"{kind}\t{size}\t1700000000.0\t{path}\n" for kind, size, path in items)


THE_STUDYS_WALK = rows(
    ("d", 4096, f"{STUDY}/mesh"),
    ("d", 4096, f"{STUDY}/run"),
    ("f", 1631, f"{STUDY}/README.md"),
    ("f", 1_528_155, f"{STUDY}/mesh/zoom.png"),
    ("f", 88_120, f"{STUDY}/mesh/overview.png"),
    ("d", 4096, f"{STUDY}/run/processor0"),
    ("f", 2_400_000, f"{STUDY}/run/processor0/0.5/U"),
)


def cut_inside(output: str, row_path: str, keep: str) -> int:
    """The byte at which a cap would leave `row_path`'s row ending in `keep` -- the
    row's path cut part-way, as the daemon's cap left `.../<study>` of
    `.../<study>/mesh/zoom.png`."""
    row = output.index(f"\t{row_path}\n")
    return row + 1 + len(keep)


def test_a_row_cut_inside_its_path_does_not_become_a_file_at_the_root():
    """The incident. The cut fell right after the study id in the row for
    `mesh/zoom.png`, and `f\\t1528155\\t<mtime>\\t/work/<study>` is four well-formed
    fields naming a 1.5 MB file at the study root. The page drew the tree as that one
    unexpandable leaf; the mirror asked the service to archive it every cycle."""
    cap = cut_inside(THE_STUDYS_WALK, f"{STUDY}/mesh/zoom.png", STUDY)
    backend = CappedChannel(THE_STUDYS_WALK, cap)
    assert backend.exec("").output.endswith(f"\t{STUDY}"), "the premise: cut after the study id"

    listing = Browser(backend, home=STUDY).tree(STUDY)

    assert not any(e.path == STUDY for e in listing), "no file at the study root"
    assert not any(e.size == 1_528_155 for e in listing), "the cut row is gone, not misfiled"
    assert {e.path for e in listing} == {f"{STUDY}/mesh", f"{STUDY}/run", f"{STUDY}/README.md"}
    assert listing.truncated, "and the listing says it was cut"


def test_a_row_cut_inside_a_deeper_path_does_not_become_a_phantom_file():
    """The same cut, landing where the breadth-first order puts it now -- inside
    `run/processors4/<time>/` -- produced `/r`, `/run/processor` and `/run/p` in
    production, each a path the mirror then asked for and was told 404."""
    cap = cut_inside(THE_STUDYS_WALK, f"{STUDY}/run/processor0/0.5/U", f"{STUDY}/run/processor")

    listing = Browser(CappedChannel(THE_STUDYS_WALK, cap), home=STUDY).tree(STUDY)

    assert f"{STUDY}/run/processor" not in {e.path for e in listing}
    assert f"{STUDY}/run/processor0" in {e.path for e in listing}, "the whole rows before it stay"
    assert listing.truncated


def test_a_cut_on_a_line_end_loses_nothing_and_still_says_so():
    """When the cap happens to land right after a newline every row that arrived is
    whole, and none is dropped for having been last. The listing is still cut -- the
    backend said so -- and still says it."""
    whole_rows = THE_STUDYS_WALK.index(f"\t{STUDY}/mesh/overview.png\n")
    cap = THE_STUDYS_WALK.index("\n", whole_rows) + 1  # through overview.png's newline
    backend = CappedChannel(THE_STUDYS_WALK, cap)
    assert backend.exec("").output.endswith("overview.png\n")

    listing = Browser(backend, home=STUDY).tree(STUDY)

    assert f"{STUDY}/mesh/overview.png" in {e.path for e in listing}
    assert len(listing) == 5
    assert listing.truncated


def test_a_cut_output_says_the_output_was_capped_not_the_entries():
    """The notice for the entry cap counts to 4,000; over a listing of five rows that
    is a notice contradicting what is in front of the reader. The cut is named for
    what it was."""
    cap = cut_inside(THE_STUDYS_WALK, f"{STUDY}/mesh/zoom.png", STUDY)

    listing = Browser(CappedChannel(THE_STUDYS_WALK, cap), home=STUDY).tree(STUDY)

    assert listing.output_capped
    assert "cut short" in listing.notice and "output" in listing.notice
    assert "not looked at" in listing.notice, "the words the mirror's report is read for"
    assert STUDY in listing.notice and f"depth {DEFAULT_DEPTH}" in listing.notice
    assert f"{MAX_ENTRIES:,}" not in listing.notice
    assert listing.lines()[-1] == listing.notice, "and it reaches what is drawn"


def test_an_output_the_backend_did_not_cut_is_not_called_cut():
    backend = CappedChannel(THE_STUDYS_WALK, len(THE_STUDYS_WALK))

    listing = Browser(backend, home=STUDY).tree(STUDY)

    assert len(listing) == 7
    assert not listing.truncated and not listing.output_capped and listing.notice == ""


def test_when_both_caps_are_reached_the_entry_cap_is_the_one_named():
    """4,001 whole rows and then the byte cap: the entry cap is the fuller account --
    that many rows came through, and the notice that counts them is the right one."""
    walk = rows(*(("f", 1, f"/work/f{n:06d}") for n in range(MAX_ENTRIES + 1)))
    backend = CappedChannel(walk + rows(("f", 1, "/work/one-more")), len(walk))

    listing = Browser(backend).tree("/work")

    assert len(listing) == MAX_ENTRIES
    assert listing.truncated and not listing.output_capped
    assert f"{MAX_ENTRIES:,}" in listing.notice


def test_a_remembered_cut_listing_still_names_the_cause():
    """The files pane is drawn from the remembered listing; drawn from memory it must
    say the same thing the call that took it said."""
    cap = cut_inside(THE_STUDYS_WALK, f"{STUDY}/mesh/zoom.png", STUDY)
    browser = Browser(CappedChannel(THE_STUDYS_WALK, cap), home=STUDY)
    browser.remember(STUDY, browser.tree(STUDY))

    remembered = browser.cached(STUDY)
    assert remembered.truncated and remembered.output_capped
    assert "cut short" in remembered.notice
    assert browser.cached(f"{STUDY}/mesh").output_capped, "a subtree of a cut walk is cut too"


def test_the_mirror_does_not_fetch_the_phantom_root_file(backend, store):
    """What the incident cost, measured on the mirror: `tar?mode=pack&paths=/work/
    <study>` every twenty seconds, 502/504 each time, 04:07 to 04:16. A backend with
    no listing of its own to give (`FakeBackend.list_stored` is the protocol's None)
    walks over the channel, and the phantom must not reach the fetch."""
    cap = cut_inside(THE_STUDYS_WALK, f"{STUDY}/mesh/zoom.png", STUDY)
    backend.exec_result = ExecResult(0, THE_STUDYS_WALK[:cap], True, None)
    backend.files[f"{STUDY}/README.md"] = b"x"

    report = mirror.sync(Browser(backend, store, home=STUDY), path=STUDY, live=True)

    assert STUDY not in backend.fetched, "the study root was never asked for as a file"
    assert backend.fetched == [f"{STUDY}/README.md"], "what arrived whole came home"
    assert any("not looked at" in line for line in report.warnings), "and the cut was reported"


# -- rows that are not entries ---------------------------------------------------


def test_a_row_naming_the_listed_root_is_not_an_entry():
    """The walk is `-mindepth 1`, so a row whose path is the root itself was never
    printed whole: it is a row cut inside its path, or noise on the channel."""
    assert _parse(f"d\t4096\t1700000000.0\t{STUDY}", STUDY) is None
    assert _parse(f"f\t1528155\t1700000000.0\t{STUDY}", STUDY) is None


def test_a_row_outside_the_listed_path_is_not_an_entry():
    assert _parse("f\t9\t1700000000.0\t/work/another-study/notes.md", STUDY) is None
    assert _parse(f"f\t9\t1700000000.0\t{STUDY}-bis/notes.md", STUDY) is None, (
        "a sibling whose name merely begins with the root's is outside it")
    assert _parse("f\t9\t1700000000.0\tnotes.md", STUDY) is None


def test_a_row_under_the_listed_path_is_an_entry_however_the_root_is_spelled():
    row = f"f\t9\t1700000000.0\t{STUDY}/notes.md"
    for root in (STUDY, STUDY + "/"):
        entry = _parse(row, root)
        assert entry == Entry(path=f"{STUDY}/notes.md", is_dir=False, size=9, mtime=1700000000.0)


def test_a_root_row_in_a_whole_output_is_dropped_and_the_rest_kept():
    """Through `tree`, and with the backend saying nothing was cut: the row is refused
    on its own account, not because of the cap."""
    walk = rows(("d", 4096, STUDY)) + THE_STUDYS_WALK
    backend = CappedChannel(walk, len(walk))

    listing = Browser(backend, home=STUDY).tree(STUDY)

    assert STUDY not in {e.path for e in listing}
    assert len(listing) == 7
    assert not listing.truncated
