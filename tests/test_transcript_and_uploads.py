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


def test_a_derived_file_goes_under_the_originals_stem():
    backend = Backend()
    target, extras, made = place_upload(backend, "/work/s", "plan.pdf", b"%PDF",
                                        derive=lambda path, data: ([("page-1.png", b"png")], "1 page"))
    assert extras == ["/work/s/uploads/plan/page-1.png"] and made == "1 page"


@pytest.mark.parametrize("name, safe", [("../../etc/passwd", "passwd"), ("wing section.stl", "wing_section.stl"),
                                        ("", "upload"), ("C:\\\\x\\\\a b.step", "a_b.step")])
def test_names_are_made_safe(name, safe):
    assert safe_filename(name) == safe
