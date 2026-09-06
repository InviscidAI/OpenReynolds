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
"""

from __future__ import annotations

from openreynolds import mirror
from openreynolds.backend.base import ExecResult
from openreynolds.browse import DEFAULT_DEPTH, MAX_ENTRIES, Browser, Listing

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
    """Answers `find` from a fixed tree, honouring `-maxdepth` and the `head` cap.

    A stub that ignored the pipeline would let the truncation be asserted against a
    fiction. This one applies the same two knobs the real command does, so narrowing
    `path` or `depth` genuinely changes what comes back.
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
        return ExecResult(0, as_find_output(kept[:cap]), False, None)


def browser_for(paths, home: str = "/work") -> Browser:
    return Browser(WalkingBackend(paths), home=home)


# -- the listing says it was cut short -----------------------------------------


def test_the_exec_logs_bury_the_study_tree():
    """The premise. Without this the rest of the file proves nothing."""
    listing = browser_for(a_real_workspace()).tree("/work", depth=DEFAULT_DEPTH)

    assert len(listing) == MAX_ENTRIES
    paths = {e.path for e in listing}
    assert f"{HOME}/case000" in paths, "the walk should reach the study tree at all"
    assert f"{HOME}/case039" not in paths, "the tail was supposed to be cut off"


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
