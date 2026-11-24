"""The study's message log as a transcript (`openreynolds/transcript.py`), and where a
handed-in file lands (`openreynolds/uploads.py`): both shared by every interface that
shows a study or takes a file -- the MCP servers here and the hosted web app."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from openreynolds.backend.base import BackendError
from openreynolds.transcript import TURN_FAILED, TranscriptTail, row_to_item
from openreynolds.uploads import place_upload, safe_filename


def test_rows_become_items():
    assert row_to_item({"seq": 1, "role": "user", "content": "hi"})["kind"] == "user"
    assert row_to_item({"seq": 2, "role": "assistant", "content": ""}) is None, "a turn of tool calls only"
    failed = row_to_item({"seq": 3, "role": "event", "content": f"{TURN_FAILED} 402 plan_limit"},
                         refusal=lambda t: "plan_limit" if "plan_limit" in t else None)
    assert failed["kind"] == "turn_failed" and failed["refusal"] == "plan_limit"
    tool = row_to_item({"seq": 4, "role": "tool", "content": {"tool": "run", "output": "x" * 20_000}})
    assert tool["output_clipped"] and len(tool["output"]) == 12_000
    assert row_to_item({"seq": 5, "role": "event", "content": "hide me"}, say=lambda t: "") is None, \
        "an embedder's wording may hide a row"


def test_the_tail_reads_only_what_is_new_and_waits_for_a_half_written_line(tmp_path: Path):
    log = tmp_path / "messages.jsonl"
    log.write_text(json.dumps({"seq": 0, "role": "user", "content": "a"}) + "\n" + '{"seq": 1, "ro')
    tail = TranscriptTail(log)
    assert [i["text"] for i in tail.read_new()] == ["a"]
    with log.open("a") as fh:
        fh.write('le": "assistant", "content": "b"}\n')
    assert [i["text"] for i in tail.read_new()] == ["b"]
    assert tail.read_new() == []


class Backend:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def stat(self, path):
        if path not in self.files:
            raise BackendError("absent")
        return object()

    def put_file(self, path, data):
        self.files[path] = data



def test_a_second_upload_of_a_name_sits_beside_the_first():
    backend = Backend()
    first, _, _ = place_upload(backend, "/work/s", "wing.stl", b"1")
    second, _, _ = place_upload(backend, "/work/s", "wing.stl", b"2")
    assert (first, second) == ("/work/s/uploads/wing.stl", "/work/s/uploads/wing-2.stl")
    assert backend.files[first] == b"1"


def test_an_upload_past_the_numbered_names_never_overwrites_one():
    backend = Backend()
    placed = [place_upload(backend, "/work/s", "wing.stl", str(i).encode())[0] for i in range(25)]
    assert len(set(placed)) == 25, "every upload of a name lands at its own path"
    assert [backend.files[p] for p in placed] == [str(i).encode() for i in range(25)]
    assert placed[19] == "/work/s/uploads/wing-20.stl"
    assert all(p.startswith("/work/s/uploads/wing-") and p.endswith(".stl") for p in placed[20:])


def test_the_last_numbered_name_is_checked_before_it_is_written():
    backend = Backend()
    backend.files["/work/s/uploads/wing.stl"] = b"first"
    for n in range(2, 21):
        backend.files[f"/work/s/uploads/wing-{n}.stl"] = f"kept {n}".encode()
    target, _, _ = place_upload(backend, "/work/s", "wing.stl", b"new")
    assert target not in {"/work/s/uploads/wing.stl", *(f"/work/s/uploads/wing-{n}.stl" for n in range(2, 21))}
    assert backend.files["/work/s/uploads/wing-20.stl"] == b"kept 20"


def test_a_derived_file_goes_under_the_originals_stem():
    backend = Backend()
    target, extras, made = place_upload(backend, "/work/s", "plan.pdf", b"%PDF",
                                        derive=lambda path, data: ([("page-1.png", b"png")], "1 page"))
    assert extras == ["/work/s/uploads/plan/page-1.png"] and made == "unpacked to /work/s/uploads/plan/: 1 page"


def test_a_derived_line_claims_only_the_files_that_landed():
    class Full(Backend):
        def put_file(self, path, data):
            if path.endswith("text.md"):
                raise BackendError("disk full")
            super().put_file(path, data)
    made_of = lambda path, data: ([("page-1.png", b"p"), ("text.md", b"t")], "page-1.png and text.md")  # noqa: E731
    _, extras, made = place_upload(Full(), "/work/s", "plan.pdf", b"%PDF", derive=made_of)
    assert extras == ["/work/s/uploads/plan/page-1.png"]
    assert made == "1 of 2 files made in /work/s/uploads/plan/; the rest could not be written"
    assert placed_line("/work/s/uploads/plan.pdf", extras, made) == f"/work/s/uploads/plan.pdf ({made})"


@pytest.mark.parametrize("name, safe", [("../../etc/passwd", "passwd"), ("wing section.stl", "wing_section.stl"),
                                        ("", "upload"), ("C:\\\\x\\\\a b.step", "a_b.step")])
def test_names_are_made_safe(name, safe):
    assert safe_filename(name) == safe


# -- archives are unpacked beside themselves, on the workspace (#216) ---------------------

import io  # noqa: E402
import stat  # noqa: E402
import tarfile  # noqa: E402
import zipfile  # noqa: E402

from openreynolds import unpack_script  # noqa: E402
from openreynolds.backend.base import ExecResult  # noqa: E402
from openreynolds.backend.local import LocalBackend  # noqa: E402
from openreynolds.uploads import placed_line  # noqa: E402


def zip_of(members: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
        for name, target in (links or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, target)
    return out.getvalue()


def tar_of(members: dict[str, bytes], mode: str = "w:gz") -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode=mode) as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("geom/passwd")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        t.addfile(link)
    return out.getvalue()


def _a_shell_runs_here() -> bool:
    import subprocess

    from openreynolds.backend import local as local_backend
    try:
        return subprocess.run([local_backend.SHELL, "-lc", "echo 1"],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.fixture
def workspace(tmp_path: Path) -> LocalBackend:
    """A real local workspace: unpacking runs as a command on it, so these tests need a
    POSIX shell, which a Windows runner usually lacks (its `bash` is the WSL launcher)."""
    if not _a_shell_runs_here():
        pytest.skip("no POSIX shell here, so the unpacking command cannot run")
    return LocalBackend(tmp_path / "ws")


def on_disk(workspace: LocalBackend, path: str) -> Path:
    return workspace._resolve(path)


def test_a_zip_is_unpacked_on_the_workspace_without_its_own_top_folder(workspace):
    """The report that started #216: a zip of CAD files the agent could not open, so the
    person pasted geometry into the chat instead."""
    data = zip_of({"wing/wing.step": b"ISO-10303", "wing/parts/flap.stl": b"solid flap",
                   "__MACOSX/wing/._wing.step": b"junk", "wing/.DS_Store": b"junk"})
    target, extras, made = place_upload(workspace, "/work/s", "wing.zip", data)
    assert target == "/work/s/uploads/wing.zip" and on_disk(workspace, target).read_bytes() == data
    assert sorted(extras) == ["/work/s/uploads/wing/parts/flap.stl", "/work/s/uploads/wing/wing.step"]
    assert on_disk(workspace, "/work/s/uploads/wing/wing.step").read_bytes() == b"ISO-10303"
    assert made == "unpacked 2 files into /work/s/uploads/wing/"
    assert placed_line(target, extras, made) == f"{target} ({made})"
    assert not list(on_disk(workspace, "/work/s/uploads").glob("*.partial-unpack")), "nothing left behind"


@pytest.mark.parametrize("name, mode", [("case.tar.gz", "w:gz"), ("case.tgz", "w:gz"),
                                        ("case.tar", "w"), ("case.tar.xz", "w:xz")])
def test_a_tar_keeps_its_folders_and_leaves_links_out(workspace, name, mode):
    data = tar_of({"system/controlDict": b"FoamFile", "constant/polyMesh/points": b"(0 0 0)"}, mode)
    _, extras, made = place_upload(workspace, "/work/s", name, data)
    assert sorted(extras) == ["/work/s/uploads/case/constant/polyMesh/points",
                              "/work/s/uploads/case/system/controlDict"]
    assert not on_disk(workspace, "/work/s/uploads/case/geom").exists(), "no symlink"
    assert made.startswith("unpacked 2 files")


def test_a_member_cannot_climb_out_or_be_a_link(workspace, tmp_path):
    data = zip_of({"../../etc/cron.d/x": b"evil", "/abs/path": b"evil", "C:/win.ini": b"evil",
                   "ok/a b.txt": b"fine", "ok/../../up": b"evil"},
                  links={"ok/link": "/etc/passwd"})
    _, extras, _ = place_upload(workspace, "/work/s", "bad.zip", data)
    assert extras == ["/work/s/uploads/bad/ok/a_b.txt"]
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")
                     if p.is_file() and ".logs" not in p.parts)
    assert written == ["ws/s/uploads/bad.zip", "ws/s/uploads/bad/ok/a_b.txt"]


def test_names_that_sanitise_alike_are_all_kept(workspace):
    _, extras, made = place_upload(workspace, "/work/s", "parts.zip",
                                   zip_of({"翼.step": b"1", "機体.step": b"2", "a b.step": b"3", "a_b.step": b"4"}))
    assert sorted(extras) == ["/work/s/uploads/parts/_-2.step", "/work/s/uploads/parts/_.step",
                              "/work/s/uploads/parts/a_b-2.step", "/work/s/uploads/parts/a_b.step"]
    assert made.startswith("unpacked 4 files")


def test_an_archive_unpacks_beside_an_earlier_one_of_the_same_name(workspace):
    place_upload(workspace, "/work/s", "wing.zip", zip_of({"a.stl": b"1"}))
    target, extras, made = place_upload(workspace, "/work/s", "wing.zip", zip_of({"a.stl": b"2"}))
    assert target == "/work/s/uploads/wing-2.zip"
    assert extras == ["/work/s/uploads/wing-2/a.stl"] and made.endswith("into /work/s/uploads/wing-2/")
    assert on_disk(workspace, "/work/s/uploads/wing/a.stl").read_bytes() == b"1", "the first is untouched"


def test_an_archive_that_cannot_be_unpacked_is_still_placed_and_says_why(workspace):
    target, extras, made = place_upload(workspace, "/work/s", "broken.zip", b"PK\x03\x04 not really")
    assert on_disk(workspace, target).exists() and extras == []
    assert made == "not unpacked: it is not a readable zip"
    _, _, made = place_upload(workspace, "/work/s", "empty.zip", zip_of({}))
    assert made == "not unpacked: it holds no files"


def test_the_script_refuses_past_its_limits(tmp_path, monkeypatch):
    """The limits are the script's own, so they are checked by running it in-process
    with smaller ones: the bytes as they come out, and the member count."""
    archive = tmp_path / "z.zip"
    archive.write_bytes(zip_of({"zeros.bin": b"\0" * 5000}))
    monkeypatch.setattr(unpack_script, "MAX_BYTES", 1000)
    with pytest.raises(unpack_script.Refused, match="unpacks to more than"):
        unpack_script.main(str(archive), str(tmp_path / "z"), "z")
    assert not (tmp_path / "z").exists() and not (tmp_path / "z.partial-unpack").exists()
    monkeypatch.setattr(unpack_script, "MAX_FILES", 2)
    archive.write_bytes(zip_of({f"{i}.txt": b"x" for i in range(3)}))
    with pytest.raises(unpack_script.Refused, match="more than 2 files"):
        unpack_script.main(str(archive), str(tmp_path / "z"), "z")


class Answers(Backend):
    """A workspace whose exec answers what a test says."""

    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def exec(self, cmd, cwd=None, timeout_s=120, *, background=False):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_whatever_the_workspace_says_comes_back_as_a_line_not_an_error():
    def result(output, **kw):
        return ExecResult(exit_code=kw.pop("exit_code", 0), output=output, truncated=False, log_path=None, **kw)
    cases = [
        (OSError("connection reset"), "not unpacked: the workspace could not run the unpacking (connection reset)"),
        (result("", exit_code=None, job_id="j1", promoted=True), "not unpacked: it took too long"),
        (result("python3: command not found", exit_code=127),
         "not unpacked: the workspace could not unpack it (python3: command not found)"),
        (result('noise\nUNPACK_REFUSED {"why": "it is password-protected"}'), "not unpacked: it is password-protected"),
        (result('UNPACKED {"suffix": "", "count": 30000, "files": ["a.stl"]}'),
         "unpacked 30000 files into /work/s/uploads/big/"),
    ]
    for answer, said in cases:
        backend = Answers(answer)
        target, _, made = place_upload(backend, "/work/s", "big.zip", b"PK")
        assert backend.files[target] == b"PK" and made == said


def test_a_file_named_like_an_archive_only_by_its_stem_is_not_unpacked():
    backend = Backend()                      # no exec: nothing may be run for it
    _, extras, made = place_upload(backend, "/work/s", "zip.stl", b"solid")
    assert (extras, made) == ([], "")
    assert uploads.archive_stem(".zip") is None


from openreynolds import uploads  # noqa: E402


@pytest.mark.parametrize("name, safe", [("机翼.zip", "upload.zip"), ("模型 v2.tar.gz", "v2.tar.gz"),
                                        ("a" * 130 + ".zip", "a" * 116 + ".zip"), (".bashrc", "bashrc"),
                                        ("wing.STEP", "wing.STEP"), ("机翼.stl", "upload.stl")])
def test_a_name_keeps_its_extension_however_it_is_cleaned(name, safe):
    assert safe_filename(name) == safe


def test_an_archive_named_in_cjk_is_still_unpacked(workspace):
    target, extras, _ = place_upload(workspace, "/work/s", "机翼.zip", zip_of({"a.stl": b"1"}))
    assert target == "/work/s/uploads/upload.zip" and extras == ["/work/s/uploads/upload/a.stl"]


def test_an_answer_in_the_wrong_shape_is_a_line_not_an_error():
    for output in ('UNPACKED []', 'UNPACKED {"count": "x"}', 'UNPACK_REFUSED "no"'):
        backend = Answers(ExecResult(exit_code=0, output=output, truncated=False, log_path=None))
        _, extras, made = place_upload(backend, "/work/s", "a.zip", b"PK")
        assert extras == [] and made.startswith("not unpacked:")
