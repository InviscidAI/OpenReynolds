"""Looking at the workspace is read-only, and never involves the model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.browse import Browser, human

LISTING = (
    "d\t4096\t1700000000.0\t/work/case\n"
    "-\t120\t1700000001.0\t/work/case/log.simpleFoam\n"
    "d\t4096\t1700000002.0\t/work/case/constant\n"
    "-\t2048\t1700000003.0\t/work/case/constant/transportProperties\n"
    "-\t9\t1700000004.0\t/work/notes.md\n"
)


def listing_backend(backend, output=LISTING, exit_code=0):
    backend.exec_result = ExecResult(exit_code, output, False, None)
    return backend


def test_the_tree_comes_back_in_one_call(backend):
    listing_backend(backend)
    entries = Browser(backend).tree()

    assert backend.last_exec[0].startswith("find /work")
    assert [e.path for e in entries][0] == "/work/case"
    assert len(entries) == 5


def test_a_parent_always_precedes_its_children():
    """The tree pane builds in one pass, which only works if this holds."""
    backend = listing_backend(_FakeExec())
    entries = Browser(backend).tree()
    seen: set[str] = set()
    for entry in entries:
        parent = entry.path.rpartition("/")[0]
        assert parent == "/work" or parent in seen, f"{entry.path} arrived before {parent}"
        seen.add(entry.path)


def test_directories_sort_before_files_within_a_directory():
    backend = listing_backend(_FakeExec())
    top = [e for e in Browser(backend).tree() if e.path.count("/") == 2]
    assert [e.name for e in top] == ["case", "notes.md"]


def test_a_garbled_line_is_skipped_not_fatal(backend):
    listing_backend(backend, "not a listing at all\n" + LISTING)
    assert len(Browser(backend).tree()) == 5


def test_an_unusable_find_falls_back_to_a_plain_listing(backend):
    listing_backend(backend, "", exit_code=127)
    backend.dirs["/work"] = ["case", "notes.md"]

    entries = Browser(backend).tree()

    assert [e.path for e in entries] == ["/work/case", "/work/notes.md"]


def test_reading_a_file_gives_its_text(backend):
    backend.files["/work/notes.md"] = b"# results\n"
    text, is_text = Browser(backend).read("/work/notes.md")
    assert is_text and "# results" in text


def test_a_truncated_read_says_so(backend):
    backend.files["/work/big.log"] = b"x" * 5_000
    text, _ = Browser(backend).read("/work/big.log", limit=100)
    assert "showing the first" in text


def test_binary_is_described_rather_than_dumped(backend):
    backend.files["/work/mesh.vtk"] = b"\x00\x01\x02binary"
    text, is_text = Browser(backend).read("/work/mesh.vtk")
    assert not is_text
    assert "binary data" in text
    assert "\x00" not in text


def test_a_directory_reads_as_its_listing(backend):
    backend.dirs["/work/case"] = ["0", "constant", "system"]
    text, is_text = Browser(backend).read("/work/case")
    assert is_text and "constant" in text


def test_pulling_lands_in_the_study_directory(backend, store):
    backend.files["/work/case/renders/mesh.png"] = b"\x89PNG"
    written = Browser(backend, store).pull("/work/case/renders/mesh.png")
    assert written and written[0].exists()
    assert store.files_dir in written[0].parents


def test_pulling_without_a_study_is_an_error(backend):
    with pytest.raises(BackendError):
        Browser(backend).pull("/work/x")


def test_local_lists_what_has_been_copied_out(backend, store):
    assert Browser(backend, store).local() == []
    backend.files["/work/a.png"] = b"\x89PNG"
    Browser(backend, store).pull("/work/a.png")
    assert [p.name for p in Browser(backend, store).local()] == ["a.png"]


@pytest.mark.parametrize(
    "size,shown", [(0, "0B"), (512, "512B"), (1536, "1.5K"), (1_500_000, "1.4M")]
)
def test_sizes_are_readable(size, shown):
    assert human(size) == shown


class _FakeExec:
    """Just enough backend to answer one listing."""

    workspace_root = "/work"
    exec_result = ExecResult(0, LISTING, False, None)

    def exec(self, cmd, cwd=None, timeout_s=120):
        self.last_exec = (cmd, cwd, timeout_s)
        return self.exec_result

    def stat(self, path):
        return SimpleNamespace(path=path, is_dir=True, entries=[], size=0, mtime=0)


# -- looking starts in the study's own directory --------------------------------


def test_looking_starts_in_the_study_s_own_directory(backend):
    """"Show me my files" means this study's, not every study that ever ran here."""
    listing_backend(backend)
    Browser(backend, home="/work/20260824-120000-abcd").tree()

    assert "find /work/20260824-120000-abcd" in backend.last_exec[0]


def test_an_explicit_path_still_wins(backend):
    listing_backend(backend)
    Browser(backend, home="/work/mine").tree("/work")
    assert "find /work " in backend.last_exec[0]


def test_a_browser_without_a_home_looks_at_the_whole_workspace(backend):
    listing_backend(backend)
    Browser(backend).tree()
    assert "find /work " in backend.last_exec[0]


def test_a_depth_is_honoured(backend):
    """`files --depth` was declared, documented, and did nothing for a while."""
    listing_backend(backend)
    Browser(backend, home="/work/mine").tree(depth=1)
    assert "-maxdepth 1" in backend.last_exec[0]
