"""Being heard without derailing the work.

Two separate things go wrong when there is only one channel. Something typed while the
model is working sits unread until the whole turn ends, so "just run the coarse one"
arrives minutes late and reads as a contradiction. And asking "what is going on" costs
a turn, so people stop asking and then have no idea what is happening.
"""

from __future__ import annotations

from conftest import ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli, commands
from openreynolds.browse import Browser
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.watch import NOTHING


def make_loop(ctx, store, view):
    cfg = Config(anthropic_api_key="test-key", model="claude-opus-5")
    return Loop(cfg, ctx, store, view)


# -- typed while the model is working ------------------------------------------


def test_an_aside_typed_mid_turn_goes_to_the_model(ctx, store, view):
    loop = make_loop(ctx, store, view)
    browser = Browser(ctx.backend, store)
    reader = ScriptedReader(["/btw the inlet looks like mm/s"])

    said = cli._typed_while_working(loop, view, browser, store, reader)

    assert said and "mm/s" in said
    assert said.startswith("By the way")


def test_a_plain_remark_typed_mid_turn_goes_through_unchanged(ctx, store, view):
    loop = make_loop(ctx, store, view)
    reader = ScriptedReader(["just run the coarse one"])

    said = cli._typed_while_working(loop, view, Browser(ctx.backend, store), store, reader)

    assert said == "just run the coarse one"


def test_asking_what_is_going_on_mid_turn_never_reaches_the_model(ctx, store, view):
    """This is the whole point of it: the answer comes from here, so the question is
    free and the work is not interrupted."""
    loop = make_loop(ctx, store, view)
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    reader = ScriptedReader(["/status"])

    said = cli._typed_while_working(loop, view, Browser(ctx.backend, store), store, reader)

    assert said is None
    assert view.statuses, "it was answered"
    assert "solve" in "\n".join(view.statuses[0])


def test_several_lines_typed_mid_turn_arrive_together(ctx, store, view):
    loop = make_loop(ctx, store, view)
    reader = ScriptedReader(["stop the fine one", "/status", "coarse is enough"])

    said = cli._typed_while_working(loop, view, Browser(ctx.backend, store), store, reader)

    assert said == "stop the fine one\ncoarse is enough"
    assert view.statuses


def test_nothing_typed_means_nothing_sent(ctx, store, view):
    loop = make_loop(ctx, store, view)
    assert cli._typed_while_working(loop, view, Browser(ctx.backend, store), store,
                                    ScriptedReader([])) is None


def test_an_end_of_input_mid_turn_is_left_for_whoever_waits_on_it(ctx, store, view):
    """Swallowing it here would leave the session unendable: the prompt would wait
    forever on an EOF that had already been read by something else."""
    loop = make_loop(ctx, store, view)
    reader = ScriptedReader([None])

    cli._typed_while_working(loop, view, Browser(ctx.backend, store), store, reader)

    assert reader.poll() is None


def test_exit_typed_mid_turn_is_left_for_the_session_loop(ctx, store, view):
    loop = make_loop(ctx, store, view)
    reader = ScriptedReader(["/exit"])

    said = cli._typed_while_working(loop, view, Browser(ctx.backend, store), store, reader)

    assert said is None
    assert reader.poll() == "/exit"


def test_the_loop_carries_an_aside_into_the_next_message(ctx, store, view):
    """The mechanism, end to end: tool results have to come first, but a text block
    may follow them, and that is how a mid-turn remark lands at the next step."""
    loop = make_loop(ctx, store, view)
    browser = Browser(ctx.backend, store)
    reader = ScriptedReader(["/btw coarse is enough"])
    loop.interject = lambda: cli._typed_while_working(loop, view, browser, store, reader)
    ctx.backend.files["/work/log"] = b"Time = 1\n"

    fake = install_model(
        loop,
        [
            message([tool_block("read_file", {"path": "/work/log"})], stop_reason="tool_use"),
            message([text_block("running the coarse case only")]),
        ],
    )
    loop.say("run all three")
    loop.run()

    followed = [
        block
        for turn in fake.calls[-1]["messages"]
        if isinstance(turn["content"], list)
        for block in turn["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    assert any("coarse is enough" in block["text"] for block in followed)
    assert view.interjections, "and the user was shown that it was carried"


# -- typed at the prompt -------------------------------------------------------


def test_status_at_the_prompt_costs_no_turn(ctx, store, view):
    loop = make_loop(ctx, store, view)
    spoken = cli._apply(commands.parse("/status"), loop, view, Browser(ctx.backend, store), store)

    assert spoken is None
    assert loop.messages == [], "nothing was added to the thread"
    assert view.statuses


def test_files_at_the_prompt_goes_to_the_view(ctx, store, view):
    loop = make_loop(ctx, store, view)
    cli._apply(commands.parse("/files /work/case"), loop, view, Browser(ctx.backend, store), store)

    assert view.listings == ["/work/case"]
    assert loop.messages == []


def test_help_at_the_prompt_is_shown_not_sent(ctx, store, view):
    loop = make_loop(ctx, store, view)
    cli._apply(commands.parse("/help"), loop, view, Browser(ctx.backend, store), store)

    assert view.statuses and any("/btw" in line for line in view.statuses[0])
    assert loop.messages == []


def test_an_ordinary_message_still_becomes_a_turn(ctx, store, view):
    loop = make_loop(ctx, store, view)
    spoken = cli._apply(
        commands.parse("run the coarse case"), loop, view, Browser(ctx.backend, store), store
    )

    assert spoken == "run the coarse case"
    assert loop.messages[-1]["content"] == "run the coarse case"


def test_exit_at_the_prompt_asks_to_leave(ctx, store, view):
    loop = make_loop(ctx, store, view)
    assert cli._apply(commands.parse("/exit"), loop, view, Browser(ctx.backend, store), store) is cli.QUIT


def test_a_status_question_while_a_job_runs_does_not_end_the_watch(
    ctx, store, view, fast_polling
):
    """Typing `/status` during a solve must not be mistaken for a new instruction."""
    loop = make_loop(ctx, store, view)
    backend = ctx.backend
    job_id = backend.job_start("simpleFoam", name="solve")
    store.record_job(job_id, cmd="simpleFoam", name="solve")
    install_model(loop, [message([text_block("still going")])])

    cli._run_interactive(
        loop, backend, store, view, Browser(backend, store),
        ScriptedReader(["/status", "/exit"]),
    )

    assert view.statuses, "it was answered"
    assert loop.messages == [], "and the model was never woken for it"


def test_nothing_typed_is_still_nothing(ctx, store, view):
    reader = ScriptedReader([])
    assert reader.poll() is NOTHING
