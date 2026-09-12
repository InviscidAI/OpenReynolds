"""The JSON interface: the stream out, the stream in, and what must not be on stdout.

An agent driving this tool reads one stream. Everything here exists because the one
thing that breaks a reader irrecoverably is a line that is not an object -- a rich
status line, a half-written line from another thread, an inline image's base64 escape
payload -- and none of those announce themselves. They just move the parser one object
out of step and leave it there.
"""

from __future__ import annotations

import io
import json
import sys
import threading

import pytest

from conftest import FakeBackend
from openreynolds import cli, jsonview, trace
from openreynolds.config import Config
from openreynolds.jsonview import JsonReader, JsonView, message_text
from openreynolds.store import JobRecord, Store
from openreynolds.watch import NOTHING


@pytest.fixture(autouse=True)
def leave_the_module_as_it_was_found(monkeypatch):
    """stream-json moves the module console onto stderr and turns tracing on, and both
    are process-wide on purpose -- a process runs in one mode for its whole life.

    A test session is not a process running in one mode. Without this, the first test
    here to ask for stream-json left every later CliRunner invocation in the whole
    suite printing into a closed stream, and the two tests that noticed were in a file
    about video assembly.
    """
    monkeypatch.setattr(cli, "console", cli.console)
    monkeypatch.setattr(trace, "on", trace.on)
    monkeypatch.setattr(trace, "_sink", trace._sink)
    monkeypatch.setattr(trace, "_study", trace._study)


def read(sink: io.StringIO) -> list[dict]:
    """Every line of the stream, parsed. Fails loudly on anything that is not one."""
    rows = []
    for number, line in enumerate(sink.getvalue().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise AssertionError(f"line {number} is not JSON: {line!r} ({exc})") from None
    return rows


def types(rows: list[dict]) -> list[str]:
    return [row["type"] for row in rows]


# -- the stream out ------------------------------------------------------------


def test_the_first_event_carries_the_id_a_resume_needs():
    """`--study <id>` has always been able to reopen a run. Until this event the id
    reached the screen only inside rich markup in an English sentence, so an agent
    wanting to resume had to scrape a styled line or read the studies directory
    behind the tool's back."""
    sink = io.StringIO()
    view = JsonView(sink)

    view.header("20260912-101500-ab12", "974f4406-11da", "claude-opus-5", "/local/studies/x")

    start = read(sink)[0]
    assert start["type"] == "session_start"
    assert start["study_id"] == "20260912-101500-ab12"
    assert start["instance_id"] == "974f4406-11da"
    assert start["model"] == "claude-opus-5"
    assert start["dir"] == "/local/studies/x"


def test_every_event_says_which_study_and_which_schema_it_belongs_to():
    """A stream with no study on it cannot be told from another terminal's stream,
    and a reader with no schema version has to guess whether a missing field is a
    missing field or an older build."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("study-7", "iid", "model", "/dir")

    view.info("carrying on")
    view.tool("bash", "blockMesh")

    for row in read(sink):
        assert row["study"] == "study-7"
        assert row["v"] == jsonview.SCHEMA
        assert isinstance(row["at"], float)


def test_the_terminal_event_says_how_the_run_ended():
    """The exit code says the same thing and a reader on the far side of a pipe may
    never see it."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("study-7", "iid", "model", "/dir")

    view.session_end("timeout")

    last = read(sink)[-1]
    assert last["type"] == "session_end"
    assert last["outcome"] == "timeout"


def test_model_text_is_not_one_object_per_token():
    """`text_delta` fires per token. One object per delta costs about twenty bytes of
    envelope for three bytes of text and tells a reader nothing a line does not."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")

    for word in "the residuals are falling steadily now".split():
        view.text_delta(word + " ")
    view.turn_end()

    text_events = [r for r in read(sink) if r["type"] == "text"]
    assert len(text_events) == 1, "six deltas became one event, not six"


def test_a_long_run_of_text_still_streams_before_the_turn_ends():
    """Buffering only until `turn_end` was the other option and it is worse: a turn
    that spends four minutes in tool calls would say nothing at all until it ended."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")

    view.text_delta("x" * (jsonview.DELTA_FLUSH_CHARS + 10))

    assert types(read(sink)).count("text") == 1, "it went out without waiting for the turn"


def test_a_finished_turn_repeats_the_whole_message_once():
    """An agent that buffered the deltas itself has to trust it saw all of them; an
    agent that ignored them wants one object to read."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")

    view.thinking_begin()
    view.thinking_delta("checking the mesh first")
    view.text_delta("the mesh ")
    view.text_delta("is fine")
    view.turn_end()

    whole = [r for r in read(sink) if r["type"] == "message"]
    assert len(whole) == 1
    assert whole[0]["text"] == "the mesh is fine"
    assert whole[0]["thinking"] == "checking the mesh first"
    assert whole[0]["role"] == "assistant"


def test_a_second_turn_does_not_repeat_the_first_one():
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")

    view.text_delta("first")
    view.turn_end()
    view.text_delta("second")
    view.turn_end()

    messages = [r["text"] for r in read(sink) if r["type"] == "message"]
    assert messages == ["first", "second"]


def test_job_state_reaches_the_stream_as_data():
    """The one structured fact an agent watching a four-hour solve wants."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")

    view.jobs([
        JobRecord(job_id="job-1", name="solve", status="exited", exit_code=0, cmd="simpleFoam"),
    ])

    event = [r for r in read(sink) if r["type"] == "jobs"][0]
    assert event["jobs"] == [{
        "id": "job-1", "name": "solve", "status": "exited", "exit_code": 0,
        "end_reason": None, "cmd": "simpleFoam", "cwd": "",
    }]


def test_an_unchanged_progress_picture_is_not_said_again():
    """The tracker pushes about once a second for the whole session, whether or not
    anything moved. Once a second forever is a lot of stream for `still solving`."""
    from openreynolds.progress import Progress

    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")

    for tick in range(5):
        view.progress(Progress(phase="solving", headline="solving simpleFoam",
                               fraction=0.4, tick=tick))
    view.progress(Progress(phase="solving", headline="solving simpleFoam", fraction=0.9))

    assert types(read(sink)).count("progress") == 2


def test_a_sink_that_went_away_ends_the_stream_and_not_the_study():
    """A reader that closed the pipe is a reader that stopped reading. It is not a
    reason to lose a study that has a solve running on an instance."""

    class Closed:
        def write(self, text):
            raise OSError("broken pipe")

        def flush(self):
            raise OSError("broken pipe")

    view = JsonView(Closed())
    view.header("s", "i", "m", "/d")  # must not raise
    view.info("still going")


def test_two_threads_writing_at_once_never_split_a_line():
    """`progress` comes from the tracker's thread, `delivered` and `mirrored` from the
    mirror's, `desk` and `narration` from the concierge's. TuiView solves this with
    call_from_thread; a writer to a stream has to hold a lock, and a half-written line
    is not a parse error a reader recovers from -- it silently rejoins mid-object."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")
    start = threading.Barrier(4)

    def shout(word):
        start.wait()
        for _ in range(100):
            view.narration(word * 50)

    threads = [threading.Thread(target=shout, args=(w,)) for w in "abc"]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    rows = read(sink)
    assert len(rows) == 301, "one object per call, and every one of them whole"


def test_a_shared_trace_sink_writes_whole_lines_too(monkeypatch):
    """`--output-format stream-json` puts the cost events on the same stream. Two
    writers racing for one stdout is the same broken line by another route."""
    sink = io.StringIO()
    view = JsonView(sink)
    view.header("s", "i", "m", "/d")
    monkeypatch.setattr(trace, "_study", "s")
    trace.to(view.trace_sink())
    try:
        trace.event("tool", tool="bash", seconds=1.5, cmd="blockMesh")
    finally:
        trace.off()

    cost = [r for r in read(sink) if r["type"] == "cost"]
    assert cost[0]["kind"] == "tool"
    assert cost[0]["tool"] == "bash"


# -- the stream in -------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ('{"type": "user", "text": "mesh the elbow"}', "mesh the elbow"),
        ('{"text": "no type is a user message"}', "no type is a user message"),
        ('{"type": "user", "content": "content works too"}', "content works too"),
        ('{"type": "user", "content": [{"type": "text", "text": "blocks too"}]}', "blocks too"),
        ('{"type": "message", "message": {"content": "nested"}}', "nested"),
        ('{"type": "result", "text": "not for us"}', None),
        ("not json at all", None),
        ("[1, 2, 3]", None),
        ("   ", None),
        ('{"type": "user"}', None),
    ],
)
def test_only_an_understood_object_is_treated_as_something_the_user_said(line, expected):
    """Guessing that a line of prose was meant as a message is how a malformed stream
    turns into a study nobody asked for."""
    assert message_text(line) == expected


class Silent:
    """A stream that hands over the lines it was given and then waits, the way a
    terminal or a pipe held open by an agent does -- rather than ending at once, which
    is the one thing a real conversational stream never does."""

    def __init__(self, lines=()):
        self.lines = list(lines)
        self.released = threading.Event()

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        self.released.wait(5)
        return ""


def test_nothing_typed_yet_is_not_the_same_as_end_of_input():
    """The drain between tool calls puts an EOF back for whoever waits at the prompt.
    A reader answering None for `nothing yet` would end every session at its first
    idle moment."""
    silent = Silent()
    waiting = JsonReader(silent)
    assert waiting.poll() is NOTHING, "nothing typed yet is not the end of anything"
    silent.released.set()
    waiting._thread.join(5)
    assert waiting.poll() is None, "the stream ended, and that is an EOF"


def test_messages_arrive_in_order_and_then_the_stream_ends():
    stream = io.StringIO(
        '{"type": "user", "text": "first"}\n'
        "garbage that is not an object\n"
        '{"type": "user", "text": "second"}\n'
    )
    reader = JsonReader(stream)
    reader._thread.join(2)

    assert reader.accepts_input is True
    assert reader.get(timeout=2) == "first"
    assert reader.get(timeout=2) == "second"
    assert reader.get(timeout=2) is None, "then EOF"
    assert reader.ignored == 1


def test_something_taken_but_not_used_goes_back():
    """The drain between tool calls takes what was typed and hands back only what was
    meant for the model; the rest has to be there for whoever asks next."""
    reader = JsonReader(Silent(['{"type": "user", "text": "hello"}\n']))

    taken = reader.get(timeout=2)
    reader.putback(taken)

    assert reader.pending() is True
    assert reader.get(timeout=2) == "hello"


# -- nothing but JSON on stdout ------------------------------------------------


class FakeLoop:
    """A session's loop with the model taken out, driving the view the way it does.

    It touches every view method the session's own machinery does not, because the
    question this test asks is about the STREAM, not about the model: anything that
    prints is a failure whoever printed it.
    """

    def __init__(self, cfg, ctx, store, view, capture=None, progress=None):
        self.cfg, self.ctx, self.store, self.view = cfg, ctx, store, view
        self.messages: list[dict] = []
        self.api_failures = 0
        self.blocked_reason = None
        self.needs_refresh = False
        self.gate = None
        self.interject = None

    def add_tokens(self, count):
        pass

    def brief(self, text):
        self.messages.append({"role": "system", "content": text})

    def say(self, text):
        self.messages.append({"role": "user", "content": text})

    def inform(self, text):
        self.messages.append({"role": "user", "content": text})

    def refresh(self, situation):
        pass

    def settle(self):
        pass

    def run(self):
        view = self.view
        view.thinking_begin()
        view.thinking_delta("the elbow is a standard tutorial")
        view.text_delta("running blockMesh")
        view.tool("bash", "blockMesh")
        view.tool_error("checkMesh found 3 skewed faces")
        view.stage("meshing")
        view.step(1, 12.5, 2)
        view.usage(812_000, 0.81)
        view.watching(["solve"])
        view.jobs([JobRecord(job_id="job-1", name="solve", status="running")])
        view.notice("the turn was truncated")
        view.warn("the mirror skipped a file")
        view.info("the thread is intact")
        view.narration("writing the mesh dictionary")
        view.desk("it is meshing; about two minutes left")
        view.interjection("use a finer mesh")
        view.status(["solving simpleFoam", "step 40 of 500"])
        view.show_files("/work/study-test", 1)
        view.prompt()
        view.turn_end()


def test_a_whole_stream_json_session_puts_nothing_but_json_on_stdout(
    tmp_path, monkeypatch, capsys
):
    """The deliverable. Roughly fifteen call sites in cli.py print through one rich
    console on stdout -- hard errors, the joined-workspace warning, close-down, the
    plain fallback -- and one of those lines in the middle of an NDJSON stream is not
    something a reader recovers from: it resynchronises on the next newline, in the
    middle of an object. So the check is not "the events are right", it is "there is
    nothing else here at all"."""
    monkeypatch.setattr(cli, "console", cli.console)  # restored on teardown
    backend = FakeBackend()
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (backend, None, "iid-1"))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    cfg = Config(
        foamd_url="https://svc.example",
        foamd_api_key="of_live_key",
        llm_api_key="sk-test",
        model="claude-opus-5",
        studies_dir=tmp_path / "studies",
        capture=False,
        desk=False,
        mesh_tool=False,
        mirror_interval_s=0.0,
    )

    outcome = cli.session(
        cfg, study_id=None, instance_id=None, one_shot="mesh the elbow",
        output_format="stream-json",
    )

    captured = capsys.readouterr()
    assert outcome == "ok"
    rows = []
    for number, line in enumerate(captured.out.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            raise AssertionError(
                f"stdout line {number} is not JSON: {line!r}\n"
                "something printed onto the stream"
            ) from None
    assert rows, "the session said something"
    assert rows[0]["type"] == "session_start"
    assert rows[0]["study_id"] == rows[-1]["study"]
    assert rows[-1]["type"] == "session_end"
    assert rows[-1]["outcome"] == "ok"
    seen = {row["type"] for row in rows}
    for needed in ("message", "tool", "jobs", "step", "status", "files", "prompt"):
        assert needed in seen, f"no {needed} event reached the stream"


def test_the_prose_a_stream_json_session_would_have_printed_goes_to_stderr(
    tmp_path, monkeypatch, capsys
):
    """Moved rather than dropped: close-down says which workspace was left up and what
    it will keep costing, and an agent that swallowed that would be hiding a bill."""
    monkeypatch.setattr(cli, "console", cli.console)
    backend = FakeBackend()
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (backend, None, "iid-1"))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    cfg = Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="sk", model="m",
        studies_dir=tmp_path / "studies", capture=False, desk=False, mesh_tool=False,
        mirror_interval_s=0.0,
    )

    cli.session(cfg, study_id=None, instance_id=None, one_shot="go",
                output_format="stream-json")

    captured = capsys.readouterr()
    assert "resume:" in captured.err, "close-down still says how to come back"


def test_without_a_prompt_the_same_flag_is_a_conversation(tmp_path, monkeypatch, capsys):
    """Two different features, and only one code path had them before: `-p` forces the
    no-input reader, so a JSON mode that only did `-p` would be one message and one
    reply forever. Here the other side of the stream is stdin."""
    backend = FakeBackend()
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (backend, None, "iid-1"))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    monkeypatch.setattr(sys, "stdin", Silent([
        '{"type": "user", "text": "how many cells?"}\n',
        '{"type": "user", "text": "/exit"}\n',
    ]))
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="sk", model="m",
                 studies_dir=tmp_path / "studies", capture=False, desk=False,
                 mesh_tool=False, mirror_interval_s=0.0)

    outcome = cli.session(cfg, study_id=None, instance_id=None, one_shot=None,
                          output_format="stream-json")

    assert outcome is None, "an interactive session says what happened on screen"
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    seen = [row["type"] for row in rows]
    assert "prompt" in seen, "it said when it was the reader's turn"
    assert "message" in seen, "and answered what was typed"
    assert seen[-1] == "session_end"


def test_the_option_reaches_the_session(monkeypatch):
    """A click option nothing reads is an option that does nothing, and `--help` will
    still advertise it."""
    from click.testing import CliRunner

    seen = {}
    monkeypatch.setattr(
        Config, "load",
        classmethod(lambda cls: Config(foamd_url="u", foamd_api_key="k",
                                       llm_api_key="sk", model="m")),
    )
    monkeypatch.setattr(cli, "session", lambda cfg, **kw: seen.update(kw) or "ok")

    result = CliRunner().invoke(
        cli.main, ["-p", "go", "--output-format", "stream-json"]
    )

    assert result.exit_code == 0, result.output
    assert seen["output_format"] == "stream-json"


def test_an_interface_of_its_own_is_not_given_a_second_one(tmp_path, monkeypatch):
    """The hosted web app passes `interface=`; that IS its presentation. Putting a
    JSON view on the same session would run two views over one loop."""
    monkeypatch.setattr(cli, "console", cli.console)
    backend = FakeBackend()
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (backend, None, "iid-1"))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="sk", model="m",
                 studies_dir=tmp_path / "studies", capture=False, desk=False,
                 mesh_tool=False, mirror_interval_s=0.0)
    used = []

    def interface(drive):
        from openreynolds.view import ConsoleView
        used.append("mine")
        drive(ConsoleView(cli.console), cli.NullReader())
        return False

    cli.session(cfg, study_id=None, instance_id=None, one_shot="go",
                output_format="stream-json", interface=interface)

    assert used == ["mine"]


# -- the plain terminal keeps up -----------------------------------------------


def test_the_plain_terminal_says_when_a_job_changes_state(console_sink):
    """ConsoleView inherited the protocol's empty body for its whole life, so plain
    and piped runs dropped every job state change -- and `-p`, the mode with nobody
    watching, was the mode that dropped it."""
    from openreynolds.view import ConsoleView

    view = ConsoleView(console_sink.console)
    view.jobs([JobRecord(job_id="job-1", name="solve", status="running")])
    view.jobs([JobRecord(job_id="job-1", name="solve", status="running")])
    view.jobs([
        JobRecord(job_id="job-1", name="solve", status="exited", exit_code=0),
    ])

    said = console_sink.text()
    assert said.count("job solve") == 2, "the unchanged repeat said nothing"
    assert "exit 0" in said


def test_a_console_nobody_is_looking_at_is_not_folded_at_eighty_columns():
    """rich falls back to eighty columns off a terminal, which folds a workspace path
    in the middle of a token and makes a piped run harder to read than the terminal it
    was copied from."""
    from openreynolds.view import PIPED_WIDTH, plain_console

    assert plain_console().width == PIPED_WIDTH


@pytest.fixture
def console_sink():
    from rich.console import Console

    class Sink:
        def __init__(self):
            self.buffer = io.StringIO()
            self.console = Console(file=self.buffer, force_terminal=False, width=200)

        def text(self):
            return self.buffer.getvalue()

    return Sink()


# -- what the trace knows about itself -----------------------------------------


def test_a_trace_can_be_turned_on_after_the_process_started(tmp_path, monkeypatch):
    """The path was read once at import, so the only way to record anything was to
    have set an environment variable before the process began. A flag could not reach
    it."""
    monkeypatch.setattr(trace, "on", False)
    monkeypatch.setattr(trace, "_path", "")
    sink = io.StringIO()
    trace.begin("study-9")
    trace.to(sink)
    try:
        trace.event("turn", seconds=2.0, model="m", messages=4, stop="end_turn")
    finally:
        trace.off()
        trace.begin("")

    row = json.loads(sink.getvalue().strip())
    assert row["kind"] == "turn", "the shapes that already existed are unchanged"
    assert row["study"] == "study-9", "and now they say which study"
    assert row["ts"].endswith("Z"), "and when, by a clock anything else can be lined up to"
    assert row["v"] == trace.SCHEMA


def test_turning_a_streamed_trace_off_leaves_the_environment_s_own_file_alone(monkeypatch):
    """A session that streamed its cost events must not switch off a trace file the
    person set OPENREYNOLDS_TRACE for."""
    monkeypatch.setattr(trace, "_path", str("/tmp/asked-for-this.jsonl"))
    trace.to(io.StringIO())
    trace.off()
    assert trace.on is True


def test_a_trace_that_cannot_be_written_does_not_break_the_study(monkeypatch):
    class Hostile:
        def write(self, text):
            raise OSError("no")

    monkeypatch.setattr(trace, "_path", "")
    trace.to(Hostile())
    try:
        trace.event("mirror", pulled=3, bytes=10)  # must not raise
    finally:
        trace.off()


# -- pictures do not go down a pipe --------------------------------------------


def test_an_inline_image_is_not_drawn_onto_a_pipe(tmp_path, monkeypatch):
    """protocol() answers from environment variables alone, which is the right answer
    to 'what can this terminal do' and the wrong answer to 'where are these bytes
    going'. A kitty session piped into an agent sent it megabytes of base64."""
    import base64
    import sys

    from openreynolds import images

    monkeypatch.setenv("TERM", "xterm-kitty")
    png = tmp_path / "mesh.png"
    png.write_bytes(base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        b"YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    ))

    piped = io.StringIO()
    monkeypatch.setattr(sys, "stdout", piped)

    assert images.show(png) is False
    assert piped.getvalue() == ""
    assert images.protocol() == "kitty", "the terminal can still do it; the pipe cannot"


def test_a_stream_given_on_purpose_is_still_drawn_on(tmp_path, monkeypatch):
    """A caller handing over a stream already knows where the bytes go."""
    import base64

    from openreynolds import images

    monkeypatch.setenv("TERM", "xterm-kitty")
    png = tmp_path / "mesh.png"
    png.write_bytes(base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        b"YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    ))

    out = io.StringIO()
    assert images.show(png, stream=out) is True


# -- the study a stream-json session leaves behind -----------------------------


def test_the_study_id_in_the_first_event_is_the_directory_on_disk(tmp_path, monkeypatch):
    """The id is only worth streaming if `--study <id>` actually finds it again."""
    monkeypatch.setattr(cli, "console", cli.console)
    backend = FakeBackend()
    monkeypatch.setattr(cli.hosted, "acquire", lambda *a, **k: (backend, None, "iid-1"))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="sk", model="m",
                 studies_dir=tmp_path / "studies", capture=False, desk=False,
                 mesh_tool=False, mirror_interval_s=0.0)

    cli.session(cfg, study_id=None, instance_id=None, one_shot="go",
                output_format="stream-json")

    study_id = read(sink)[0]["study_id"]
    assert Store(tmp_path / "studies", study_id).dir.is_dir()
