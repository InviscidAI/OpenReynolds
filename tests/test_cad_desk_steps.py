"""The CAD desk's steps reach the transcript, the chat and the operator's log.

Before this, a step of the desk's work went to the status line and nowhere else: the
study read back afterwards (`study_log.py`) showed a 700 s gap and then a mesh, and the
question "what did the reviewer say about each iteration" had no answer anywhere. Now
`cli._cad_desk_step` writes each step as an `event` row whose content is
`Step.as_event()`, uploads it with the capture plane, and tells the view -- and the
desk's own reading of the person's words is filtered so those rows cannot crowd it out.
"""

from __future__ import annotations

from types import SimpleNamespace

from openreynolds import cli
from openreynolds.cad.agent import Step
from openreynolds.cad.review import Problem, Review, Reviewer
from openreynolds.view import ConsoleView, desk_step_line


class Watching:
    """A view that remembers what it was told."""

    def __init__(self):
        self.narrated: list[str] = []
        self.steps: list[dict] = []

    def narration(self, text):
        self.narrated.append(text)

    def desk_step(self, event):
        self.steps.append(event)


class Uploading:
    def __init__(self):
        self.rows: list[tuple[int, str, object]] = []

    def message(self, seq, role, content):
        self.rows.append((seq, role, content))


# -- the record a step makes of itself -------------------------------------------


def test_a_cell_step_becomes_a_row_a_person_can_read():
    step = Step(cmd="import build123d as bd\nloop = bd.Circle(6)", exit_code=0, seconds=12.4,
                output="ok", image="1 picture", number=3,
                reasoning="Building the first loop branch as a 6 mm arc,\n  then measuring it.")
    event = step.as_event()
    assert event["desk"] == "cad" and event["kind"] == "cell" and event["step"] == 3
    assert event["text"] == "Building the first loop branch as a 6 mm arc, then measuring it."
    assert event["cmd"].startswith("import build123d")
    assert event["exit_code"] == 0 and event["seconds"] == 12.4 and event["image"] == "1 picture"


def test_a_step_with_no_reasoning_falls_back_to_its_first_line_of_code():
    event = Step(cmd="blockMesh = subprocess.run(['blockMesh'])", exit_code=1, seconds=3).as_event()
    assert event["text"] == "blockMesh = subprocess.run(['blockMesh'])"
    assert event["step"] == 0


def test_a_review_step_carries_the_verdict_as_its_text():
    step = Step(cmd="[review] fail: the loops do not rejoin", exit_code=1, seconds=40,
                output="independent review: FAIL -- looked at 3 views:\n- the loops do not rejoin",
                image="3 pictures", kind="review", number=1)
    event = step.as_event()
    assert event["kind"] == "review" and event["step"] == 1
    assert event["text"].startswith("independent review: FAIL")


def test_the_transcript_row_is_bounded():
    event = Step(cmd="x" * 5000, exit_code=0, seconds=1, output="y" * 5000,
                 reasoning="z" * 5000).as_event()
    assert len(event["cmd"]) == 600 and len(event["output"]) == 600 and len(event["text"]) == 400


def test_the_line_every_text_view_prints():
    cell = Step(cmd="a = 1", exit_code=0, seconds=12.4, number=3, image="1 picture",
                reasoning="Measuring the throat width.").as_event()
    assert desk_step_line(cell) == "cad desk step 3 · 12s · ok · picture · Measuring the throat width."
    failed = Step(cmd="a = 1", exit_code=1, seconds=2, number=4).as_event()
    assert "exit 1" in desk_step_line(failed)
    review = Step(cmd="[review] pass: no problems", exit_code=0, seconds=41, kind="review",
                  number=2, output="independent review: PASS -- looked at 3 views").as_event()
    assert desk_step_line(review) == "cad review 2 · 41s · independent review: PASS -- looked at 3 views"


# -- the hook: store, capture, view -------------------------------------------------


def test_a_desk_step_lands_in_the_store_the_capture_plane_and_the_view(store):
    view, capture = Watching(), Uploading()
    step = Step(cmd="import build123d as bd", exit_code=0, seconds=5, number=1,
                reasoning="Starting from a 5 mm channel.")

    cli._cad_desk_step(view, None, step, store=store, capture=capture)

    rows = store.recent_messages(10)
    assert len(rows) == 1 and rows[0]["role"] == "event"
    assert rows[0]["content"]["desk"] == "cad" and rows[0]["content"]["step"] == 1
    assert capture.rows == [(rows[0]["seq"], "event", rows[0]["content"])]
    assert view.steps == [rows[0]["content"]]
    assert view.narrated and view.narrated[0].startswith("cad desk [0] 5s")


def test_without_a_store_the_step_still_reaches_the_view():
    view = Watching()
    cli._cad_desk_step(view, None, Step(cmd="a = 1", exit_code=0, seconds=1))
    assert len(view.steps) == 1 and view.narrated


def test_a_store_that_will_not_write_does_not_end_the_step(store):
    view = Watching()
    store.append_message = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    cli._cad_desk_step(view, None, Step(cmd="a = 1", exit_code=0, seconds=1), store=store)
    assert len(view.steps) == 1


def test_the_reviewer_reports_a_review_step(store):
    view = Watching()
    reviewer = Reviewer.__new__(Reviewer)
    reviewer.on_step = lambda step: cli._cad_desk_step(view, None, step, store=store)
    review = Review(verdict="fail", confidence="sure", round=2, seconds=33.0, views=["a", "b", "c"],
                    problems=[Problem("the loops do not rejoin", "cells.png",
                                      "no diode without a return path", "blocking")],
                    summary="The return branches end short of the main channel.")

    reviewer._report(review)

    row = store.recent_messages(1)[0]["content"]
    assert row["kind"] == "review" and row["step"] == 2 and row["exit_code"] == 1
    assert "the loops do not rejoin" in row["text"]
    assert view.steps[0] is not None and "cad review 2" in desk_step_line(view.steps[0])


def test_the_console_view_prints_a_desk_step(capsys):
    view = ConsoleView.__new__(ConsoleView)
    from rich.console import Console
    view.console = Console(file=None, force_terminal=False, width=200)
    view.desk_step(Step(cmd="a = 1", exit_code=0, seconds=3, number=1,
                        reasoning="Drawing the outline.").as_event())
    assert "cad desk step 1" in capsys.readouterr().out


# -- the desk's own reading of the person is not crowded out ------------------------


def test_recent_messages_can_be_asked_for_one_role(store):
    store.append_message("user", "make it 2D")
    for n in range(50):
        store.append_message("event", Step(cmd=f"c{n}", exit_code=0, seconds=1, number=n).as_event())
    store.append_message("user", "and 5 mm wide")

    everything = store.recent_messages(40)
    assert [r["role"] for r in everything].count("user") == 1, "the window is full of steps"

    people = store.recent_messages(40, roles=("user",))
    assert [r["content"] for r in people] == ["make it 2D", "and 5 mm wide"]


def test_the_desk_reads_the_person_past_its_own_steps(store, backend):
    from openreynolds.cad.agent import CadDesk
    from openreynolds.config import Config

    store.append_message("user", "a Tesla valve, please")
    for n in range(45):
        store.append_message("event", Step(cmd=f"c{n}", exit_code=0, seconds=1).as_event())
    desk = CadDesk(Config(llm_api_key="k", model="m"), backend, store, "/work/study")
    assert desk._said() == ["a Tesla valve, please"]


def test_a_store_without_the_filter_is_still_read(backend):
    from openreynolds.cad.agent import CadDesk
    from openreynolds.config import Config

    class Older:
        def recent_messages(self, limit=30):
            return [{"role": "user", "content": "old words"}]

    desk = CadDesk(Config(llm_api_key="k", model="m"), backend, Older(), "/work/study")
    assert desk._said() == ["old words"]
