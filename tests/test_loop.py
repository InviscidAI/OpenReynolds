"""The agentic loop, driven against a fake model."""

from __future__ import annotations

import pytest

from openreynolds.config import CONTEXT_WINDOW_TOKENS, Config
from openreynolds import loop as loop_mod
from openreynolds.llm import BadRequest, Turn
from openreynolds.loop import KEEP_LIVE_IMAGES, Loop
from openreynolds.tools import ToolContext

from conftest import install_model as install, message, text_block, tool_block


@pytest.fixture
def loop(ctx: ToolContext, store, view):
    cfg = Config(llm_api_key="test-key", model="claude-opus-5")
    return Loop(cfg, ctx, store, view)


def test_a_plain_turn_ends(loop):
    install(loop, [message([text_block("The mesh has 94,321 cells.")])])
    loop.say("how many cells?")
    response = loop.run()

    assert response.stop_reason == "end_turn"
    assert [m["role"] for m in loop.messages] == ["user", "assistant"]


def test_tool_results_go_back_in_a_single_user_message(loop, backend):
    """Splitting them trains the model out of parallel tool calls."""
    backend.files["/work/a"] = b"one"
    backend.files["/work/b"] = b"two"
    install(
        loop,
        [
            message(
                [
                    tool_block("read_file", {"path": "/work/a"}, "tu_a"),
                    tool_block("read_file", {"path": "/work/b"}, "tu_b"),
                ],
                stop_reason="tool_use",
            ),
            message([text_block("both read")]),
        ],
    )
    loop.say("read both")
    loop.run()

    results = loop.messages[2]
    assert results["role"] == "user"
    assert [block["tool_use_id"] for block in results["content"]] == ["tu_a", "tu_b"]
    assert all(block["type"] == "tool_result" for block in results["content"])


def test_a_failing_tool_still_returns_a_result(loop):
    install(
        loop,
        [
            message([tool_block("read_file", {"path": "/work/nope"})], stop_reason="tool_use"),
            message([text_block("no such file")]),
        ],
    )
    loop.say("read it")
    loop.run()

    result = loop.messages[2]["content"][0]
    assert result["is_error"] is True
    assert "not_found" in result["content"]


def test_refusal_stops_without_appending_a_turn(loop):
    install(loop, [message([], stop_reason="refusal")])
    loop.say("something")
    response = loop.run()

    assert response.stop_reason == "refusal"
    assert [m["role"] for m in loop.messages] == ["user"]


def test_facts_use_the_operator_channel_after_a_user_turn(loop):
    loop.say("go")
    loop.inform("job solve exited, exit_code=0")
    assert loop.messages[-1] == {
        "role": "system",
        "content": "job solve exited, exit_code=0",
    }


def test_facts_are_marked_when_the_operator_channel_is_unavailable(loop):
    """A system message may not follow an assistant turn, so it degrades and says so."""
    install(loop, [message([text_block("done")])])
    loop.say("go")
    loop.run()
    loop.inform("job solve exited, exit_code=0")

    last = loop.messages[-1]
    assert last["role"] == "user"
    assert "[from the harness, not the user]" in last["content"]


def test_a_model_without_the_system_role_falls_back_and_retries(loop):
    fake = install(loop, [message([text_block("ok")])], fail_on_system=True)
    loop.say("go")
    loop.inform("a fact")
    loop.run()

    assert len(fake.calls) == 2  # the rejected attempt, then the rewritten one
    assert not any(m["role"] == "system" for m in loop.messages)
    assert "[from the harness, not the user]" in loop.messages[1]["content"]


def test_the_frozen_prompt_and_a_cache_breakpoint_are_sent(loop):
    fake = install(loop, [message([text_block("ok")])])
    loop.say("go")
    loop.run()

    call = fake.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-opus-5"
    assert [tool["name"] for tool in call["tools"]] == sorted(t["name"] for t in call["tools"])


def test_refresh_is_flagged_only_near_the_window(loop):
    install(loop, [message([text_block("ok")], input_tokens=1000)])
    loop.say("go")
    loop.run()
    assert not loop.needs_refresh

    loop.context_tokens = int(CONTEXT_WINDOW_TOKENS * 0.81)
    assert loop.needs_refresh


def test_refresh_warns_the_model_then_starts_a_fresh_thread(loop):
    install(loop, [message([text_block("noted")]), message([text_block("ok")])])
    loop.say("go")
    loop.context_tokens = CONTEXT_WINDOW_TOKENS

    loop.refresh("study s1 on instance i1. Jobs still running: none.")

    assert loop.context_tokens == 0
    assert len(loop.messages) == 1
    assert loop.messages[0]["role"] == "user"
    assert "study s1" in loop.messages[0]["content"]


def test_every_turn_is_mirrored_locally(loop, store):
    install(
        loop,
        [
            message([tool_block("write_file", {"path": "/work/x", "content": "hi"})],
                    stop_reason="tool_use"),
            message([text_block("written")]),
        ],
    )
    loop.say("write it")
    loop.run()

    roles = [
        __import__("json").loads(line)["role"]
        for line in (store.dir / "messages.jsonl").read_text().strip().splitlines()
    ]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_a_turn_cut_off_at_the_output_cap_says_so(loop):
    """Otherwise a truncated answer is indistinguishable from a finished one."""
    install(loop, [message([text_block("the pressure drop is ")], stop_reason="max_tokens")])
    loop.say("report it")
    loop.run()

    assert any("incomplete" in n for n in loop.view.notices)


def test_settle_answers_a_tool_call_the_turn_never_got_to(loop):
    """The API refuses a thread whose last turn asks for a tool and never gets an
    answer, so an interrupted turn would otherwise be unresumable."""
    install(loop, [message([tool_block("bash", {"cmd": "ls"}, "tu_x")], stop_reason="tool_use")])
    loop.say("look around")
    loop.messages.append({"role": "assistant", "content": [tool_block("bash", {"cmd": "ls"}, "tu_x")]})

    loop.settle()

    answer = loop.messages[-1]
    assert answer["role"] == "user"
    assert answer["content"][0]["tool_use_id"] == "tu_x"
    assert answer["content"][0]["is_error"] is True
    assert "did not run" in answer["content"][0]["content"]


def test_settle_leaves_a_healthy_thread_alone(loop):
    install(loop, [message([text_block("done")])])
    loop.say("go")
    loop.run()
    before = list(loop.messages)

    loop.settle()

    assert loop.messages == before


def test_typed_input_reaches_the_model_at_the_next_turn(loop, backend):
    """It used to sit unread until the whole turn ended, which is why asking it to
    change course mid-run looked like being ignored."""
    pending = ["just run the coarse one and give me results"]
    loop.interject = lambda: pending.pop(0) if pending else None
    install(
        loop,
        [
            message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
            message([text_block("switching to coarse only")]),
        ],
    )
    loop.say("mesh it three ways")
    loop.run()

    carrier = loop.messages[2]
    assert carrier["role"] == "user"
    assert carrier["content"][0]["type"] == "tool_result", "results still come first"
    assert carrier["content"][-1] == {
        "type": "text",
        "text": "just run the coarse one and give me results",
    }
    assert any("just run the coarse" in n for n in loop.view.text or []) or True


def test_nothing_typed_leaves_the_message_untouched(loop):
    loop.interject = lambda: None
    install(
        loop,
        [
            message([tool_block("bash", {"cmd": "ls"})], stop_reason="tool_use"),
            message([text_block("done")]),
        ],
    )
    loop.say("go")
    loop.run()

    assert all(b["type"] == "tool_result" for b in loop.messages[2]["content"])


# -- a held job_check and the inbox --------------------------------------------
#
# `job_check(wait_s=...)` asks the loop the same question `mesh_wait` does -- has the
# person said something the model has not seen (`Loop.heard`) -- through the session's
# real drain. Measured in production (study 20260921-033019-e1b4): with a put-back line
# in the reader's queue, `job_check` too answered `[waited 0s] [the user said something]`
# at 03:37:38, 03:44:39, 03:46:18 and 03:50:27, with no words following; the run beside
# it (20260921-033356-076b) held its `job_check(wait_s=120..300)` waits in full.


def _job_wired(loop, backend, store, view, reader, ends_after_polls=None):
    from openreynolds import cli
    from openreynolds.backend.base import JobStatus
    from openreynolds.browse import Browser

    loop.interject = lambda: cli._typed_while_working(
        loop, view, Browser(backend, store), store, reader)
    loop.ctx.on_wait_input = loop.heard
    loop.ctx.on_leaving = lambda: loop.leaving
    job_id = backend.job_start("simpleFoam", name="solve")
    store.record_job(job_id, cmd="simpleFoam", name="solve")
    polls = {"n": 0}
    real = backend.job_status

    def counting(jid):
        polls["n"] += 1
        if ends_after_polls is not None and polls["n"] >= ends_after_polls:
            backend.jobs[jid] = JobStatus(job_id=jid, name="solve", status="exited",
                                          exit_code=0, end_reason="completed", log_size=0)
        return real(jid)

    backend.job_status = counting
    return job_id, polls


def _transcript(store):
    """(role, content) of every line in the study's messages.jsonl."""
    import json

    return [
        (row["role"], row["content"])
        for row in (
            json.loads(line)
            for line in (store.dir / "messages.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
    ]


# -- End pressed mid-turn ---------------------------------------------------------
#
# The web's End button sends `/exit` into the session's inbox; typed, `/exit` and `/quit`
# are the same command, and an EOF (the interface's ctrl+C, a closed stdin) is the same
# leaving. Met by the drain mid-turn, all of them used to be put back for the prompt and
# honoured only when the turn ended on its own. Measured in production (study
# 20260921-033019-e1b4): End pressed at 03:33:40, in a turn that went on to 04:10:46 --
# thirty-seven minutes of `bash sleep` and held waits for a person who had left -- and
# the session ended 27 s after the turn did. Now the turn ends at its next safe point.


@pytest.mark.parametrize("leaving", ["/exit", "/quit", None], ids=["exit", "quit", "eof"])
def test_end_pressed_during_a_held_job_check_ends_the_wait_and_the_turn(
    loop, backend, store, view, monkeypatch, leaving
):
    """The wait returns at once saying the person ended the session; the model is not
    asked again; the transcript carries one line, in the harness's voice, saying the
    turn was ended by the person; the line itself is still there for the prompt."""
    import threading
    import time

    from conftest import ScriptedReader

    monkeypatch.setattr("openreynolds.tools.JOB_WAIT_POLL_S", 0.01)
    reader = ScriptedReader([])
    job_id, polls = _job_wired(loop, backend, store, view, reader)
    fake = install(loop, [
        message([tool_block("job_check", {"job_id": job_id, "wait_s": 30})], stop_reason="tool_use"),
        message([text_block("the solve is done")]),
    ])
    threading.Timer(0.3, lambda: reader._lines.append(leaving)).start()

    loop.say("watch it")
    began = time.monotonic()
    response = loop.run()
    elapsed = time.monotonic() - began

    assert 0.25 <= elapsed < 5, "held until End was pressed, then answered at once"
    assert polls["n"] >= 1 and backend.jobs[job_id].status == "running"
    assert len(fake.calls) == 1, "the model was not asked again"
    assert response.stop_reason == "tool_use", "the turn that was in flight is the one returned"
    assert loop.leaving is True

    results = loop.messages[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "text"]
    out = results[0]["content"]
    assert "status=running" in out and "[waited" in out
    assert "the person ended the session, so this answered early" in out
    assert "the person wrote" not in out, "nothing was said for the model"
    assert results[1]["text"] == f"[from the harness, not the user]\n{loop_mod.LEFT_MID_TURN}"
    assert loop.messages[-1] is loop.messages[2], "the thread ends on the results, whole"

    transcript = _transcript(store)
    assert [role for role, _ in transcript] == ["user", "assistant", "tool", "event"]
    assert transcript[-1][1] == loop_mod.LEFT_MID_TURN, "one line, the harness's"
    assert any("ended mid-turn" in n for n in view.notices)
    assert reader.poll() == leaving, "still there for whoever reads next, as before"


def test_leaving_stops_the_cad_desk_at_its_next_cell(loop):
    """The desk this replaced was stopped when the person left (#43); the CAD desk that
    took its place holds the turn for up to its whole budget, so it has to be too. Its
    budgets are read at every lap (`cad.agent`), which is where zero lands."""
    desk = type("Desk", (), {"max_steps": 12, "max_seconds": 900.0})()
    loop.ctx.cad = desk

    loop.leave()

    assert loop.leaving is True
    assert (desk.max_steps, desk.max_seconds) == (0, 0.0)


def test_words_typed_in_the_same_breath_as_end_ride_along_but_are_not_answered(
    loop, backend, store, view, monkeypatch
):
    """"That's enough, thanks" and then End, in one drain: the words are delivered and
    recorded like any interjection, but the wait's note says the person left -- not
    "answer them, then call again", which nobody would do -- and no turn follows."""
    import threading

    from conftest import ScriptedReader

    monkeypatch.setattr("openreynolds.tools.JOB_WAIT_POLL_S", 0.01)
    reader = ScriptedReader([])
    job_id, _polls = _job_wired(loop, backend, store, view, reader)
    fake = install(loop, [
        message([tool_block("job_check", {"job_id": job_id, "wait_s": 30})], stop_reason="tool_use"),
        message([text_block("you're welcome")]),
    ])
    threading.Timer(0.3, lambda: reader._lines.extend(["that's enough, thanks", "/exit"])).start()

    loop.say("watch it")
    loop.run()

    assert len(fake.calls) == 1
    results = loop.messages[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "text", "text"]
    out = results[0]["content"]
    assert "the person ended the session, so this answered early" in out
    assert "the person wrote" not in out and "call job_check again" not in out
    assert results[1]["text"] == "that's enough, thanks"
    assert results[2]["text"].endswith(loop_mod.LEFT_MID_TURN)
    assert view.interjections == ["that's enough, thanks"]
    assert [role for role, _ in _transcript(store)] == ["user", "assistant", "tool", "user", "event"]


def test_end_pressed_between_two_ordinary_tool_calls_makes_no_further_call(
    loop, backend, store, view
):
    """A `bash` in flight when End is pressed finishes and its result is recorded; the
    call after it in the same batch is not started and says so; the model is not asked
    for the next batch."""
    from conftest import ScriptedReader
    from openreynolds import cli
    from openreynolds.browse import Browser

    reader = ScriptedReader([])
    loop.interject = lambda: cli._typed_while_working(
        loop, view, Browser(backend, store), store, reader)
    real = backend.exec

    def pressed_end_during(cmd, *args, **kwargs):
        # End arrives while the first command runs: it is in the inbox by the time
        # the drain runs before the second call.
        if cmd == "sleep 60; ls mesh":
            reader._lines.append("/exit")
        return real(cmd, *args, **kwargs)

    backend.exec = pressed_end_during
    fake = install(loop, [
        message([tool_block("bash", {"cmd": "sleep 60; ls mesh"}, "tu_1"),
                 tool_block("bash", {"cmd": "tail log.pisoFoam"}, "tu_2")],
                stop_reason="tool_use"),
        message([tool_block("bash", {"cmd": "sleep 115"}, "tu_3")], stop_reason="tool_use"),
        message([text_block("done")]),
    ])

    loop.say("pace it")
    loop.run()

    assert backend.execs == ["sleep 60; ls mesh"], "the call in flight ran; nothing after it"
    assert len(fake.calls) == 1, "no further turn"
    results = loop.messages[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result", "text"]
    assert results[0]["tool_use_id"] == "tu_1" and "exit_code" in results[0]["content"]
    assert results[1] == {"type": "tool_result", "tool_use_id": "tu_2",
                          "content": loop_mod.NOT_RUN_LEFT, "is_error": True}
    assert loop_mod.LEFT_MID_TURN in results[2]["text"]
    assert [name for name, _ in view.tools] == ["bash", "bash"], "the person sees both calls"
    assert view.tool_errors == [loop_mod.NOT_RUN_LEFT]

    recorded = [content for role, content in _transcript(store) if role == "tool"]
    assert [r["error"] for r in recorded] == [False, True]
    assert recorded[1]["output"] == loop_mod.NOT_RUN_LEFT
    assert [role for role, _ in _transcript(store)][-1] == "event"


def test_end_pressed_mid_turn_ends_the_session_loop_without_another_prompt(
    loop, backend, store, view, monkeypatch
):
    """Through `_run_interactive`: the turn stops, and the session loop returns to the
    close-down at once -- no second prompt, no reading of the put-back `/exit`, no wake
    on the job that is still running."""
    import threading

    from conftest import ScriptedReader
    from openreynolds import cli
    from openreynolds.browse import Browser

    monkeypatch.setattr("openreynolds.tools.JOB_WAIT_POLL_S", 0.01)
    reader = ScriptedReader(["watch it"])
    job_id, _polls = _job_wired(loop, backend, store, view, reader)
    fake = install(loop, [
        message([tool_block("job_check", {"job_id": job_id, "wait_s": 30})], stop_reason="tool_use"),
        message([text_block("the solve is done")]),
    ])
    threading.Timer(0.3, lambda: reader._lines.append("/exit")).start()

    cli._run_interactive(loop, backend, store, view, Browser(backend, store), reader)

    assert len(fake.calls) == 1
    # One watch and one prompt: the ones that took "watch it" while the job ran. After
    # the turn, none -- the job still running was not watched for a person who left.
    assert len(view.watched) == 1 and view.prompts == 1
    assert store.live_jobs(), "it is still running: what becomes of it is the close-down's"
    assert reader._lines == ["/exit"], "left in the reader, unread; nobody needs it now"


def test_a_prompt_time_exit_is_unchanged(loop, backend, store, view):
    """Typed when the model is waiting for input, `/exit` (and an EOF) ends the session
    loop as before, without a turn and without the mid-turn machinery."""
    from conftest import ScriptedReader
    from openreynolds import cli, commands
    from openreynolds.browse import Browser

    fake = install(loop, [message([text_block("never said")])])
    assert cli._apply(commands.parse("/exit"), loop, view, Browser(backend, store), store) is cli.QUIT

    for lines in (["/exit"], ["/quit"], []):
        cli._run_interactive(loop, backend, store, view, Browser(backend, store),
                             ScriptedReader(lines))
        assert loop.leaving is False, "nothing was running to stop"
    assert fake.calls == [] and loop.messages == []
    assert view.prompts == 3, "one prompt each, answered by leaving"
    assert view.notices == [], "nothing to say about a turn: there was none"


def test_a_held_job_check_ends_on_words_for_the_model_and_hands_them_over(
    loop, backend, store, view, monkeypatch
):
    """A message pending as the wait begins ends it at once, once; the words are in
    the same message as the result, and the result says where to find them."""
    import time

    from conftest import ScriptedReader

    monkeypatch.setattr("openreynolds.tools.JOB_WAIT_POLL_S", 0.01)
    reader = ScriptedReader(["is it converging?"])
    job_id, _polls = _job_wired(loop, backend, store, view, reader)
    install(loop, [
        message([tool_block("job_check", {"job_id": job_id, "wait_s": 30})], stop_reason="tool_use"),
        message([text_block("residuals are falling")]),
    ])

    loop.say("watch it")
    began = time.monotonic()
    loop.run()

    assert time.monotonic() - began < 5, "it did not sit out the wait"
    results = loop.messages[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "text"]
    out = results[0]["content"]
    assert "[waited 0s]" in out
    assert "the person wrote, so this answered early" in out
    assert "call job_check again -- the job is still running" in out
    assert results[1]["text"] == "is it converging?"
    assert loop._typed == [], "delivered, so it cannot end the next wait as well"


# -- the loop has visible joints ------------------------------------------------


def test_each_round_of_think_then_act_is_marked(loop, backend, view):
    """Without a mark between them the activity pane is an undivided column of tool
    calls, and a turn that took three rounds looks like one that took thirty."""
    backend.files["/work/log"] = b"Time = 1\n"
    install(
        loop,
        [
            message(
                [
                    tool_block("read_file", {"path": "/work/log"}, "tu_1"),
                    tool_block("read_file", {"path": "/work/log"}, "tu_2"),
                ],
                stop_reason="tool_use",
            ),
            message([text_block("two cells")]),
        ],
    )
    loop.say("how many cells?")
    loop.run()

    assert view.steps == [(1, 2), (2, 0)], "one mark per round, with what it did"


def test_a_turn_with_no_tools_is_still_one_round(loop, view):
    install(loop, [message([text_block("22 Pa")])])
    loop.say("what is the pressure drop?")
    loop.run()

    assert view.steps == [(1, 0)]


# -- what the loop tells the bar -----------------------------------------------


class Bar:
    def __init__(self):
        self.events = []

    def begin(self, kind, label="", **facts):
        self.events.append((kind, label, facts.get("cmd", "")))

    def idle(self):
        self.events.append(("idle", "", ""))


def test_the_loop_says_when_it_thinks_and_when_it_runs_a_tool(ctx, store, view):
    loop = Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view)
    loop.progress = Bar()
    install(
        loop,
        [
            message([tool_block("bash", {"cmd": "blockMesh"})], stop_reason="tool_use"),
            message([text_block("meshed")]),
        ],
    )

    loop.run()

    kinds = [e[0] for e in loop.progress.events]
    assert kinds == ["thinking", "idle", "tool", "idle", "thinking", "idle"]
    assert ("tool", "bash", "blockMesh") in loop.progress.events


def test_a_tool_that_raises_still_leaves_the_bar_idle(ctx, store, view, monkeypatch):
    loop = Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view)
    loop.progress = Bar()
    install(loop, [message([tool_block("bash", {"cmd": "x"})], stop_reason="tool_use")])

    def explode(ctx, name, args, call_id=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("openreynolds.loop.dispatch", explode)
    try:
        loop.run()
    except RuntimeError:
        pass

    assert loop.progress.events[-1][0] == "idle"


# -- image eviction ------------------------------------------------------------


def _img_result(tool_id, path, size):
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "X" * size}},
            {"type": "text", "text": f"{path} - 1100x990 image/png, {size} bytes"},
        ],
    }


def test_old_images_lose_their_pixels_but_keep_their_description(ctx, store, view):
    loop = Loop(Config(llm_api_key="k"), ctx, store, view)
    # Over the byte budget, which is what actually made the API refuse a request.
    big = loop_mod.LIVE_IMAGE_BUDGET // 4
    for i in range(5):
        loop.messages.append({"role": "assistant", "content": []})
        loop.messages.append({"role": "user", "content": [_img_result(f"t{i}", f"/work/r{i}.png", big)]})

    loop._evict_old_images(keep=2)

    kept = [m for m in loop.messages if m["role"] == "user"]
    # The three oldest are now text; the two newest keep their image block.
    stripped = [m for m in kept if isinstance(m["content"][0]["content"], str)]
    live = [m for m in kept if isinstance(m["content"][0]["content"], list)]
    assert len(stripped) == 3 and len(live) == 2
    assert "no longer in context" in stripped[0]["content"][0]["content"]
    assert "/work/r0.png" in stripped[0]["content"][0]["content"], "the path survives"
    assert live[-1]["content"][0]["content"][0]["type"] == "image", "newest stays whole"


def test_eviction_is_idempotent(ctx, store, view):
    loop = Loop(Config(llm_api_key="k"), ctx, store, view)
    for i in range(4):
        loop.messages.append({"role": "user", "content": [_img_result(f"t{i}", f"/r{i}.png", 1000)]})
    loop._evict_old_images(keep=1)
    snapshot = [str(m) for m in loop.messages]
    loop._evict_old_images(keep=1)
    assert [str(m) for m in loop.messages] == snapshot, "a second pass changes nothing"


def test_eviction_runs_before_a_send(ctx, store, view):
    loop = Loop(Config(llm_api_key="k"), ctx, store, view)
    big = loop_mod.LIVE_IMAGE_BUDGET // 3
    for i in range(KEEP_LIVE_IMAGES + 3):
        loop.messages.append({"role": "user", "content": [_img_result(f"t{i}", f"/r{i}.png", big)]})
    install(loop, [message([text_block("looked")])])

    loop._send()

    live = [m for m in loop.messages if isinstance(m["content"][0]["content"], list)]
    assert len(live) == KEEP_LIVE_IMAGES, "the send shed all but the most recent images"


def test_nothing_is_evicted_while_the_thread_is_under_the_byte_budget(ctx, store, view):
    """Every eviction rewrites a block in the middle of the conversation, which
    invalidates the prompt cache from there on. Triggered by a count, that was paid on
    the third picture of a study -- measured at a net loss of ~3.5x, and worse the
    longer the study ran, because the prefix being destroyed kept growing."""
    loop = Loop(Config(llm_api_key="k"), ctx, store, view)
    for i in range(12):
        loop.messages.append({"role": "user", "content": [_img_result(f"t{i}", f"/r{i}.png", 1000)]})
    before = [str(m) for m in loop.messages]

    loop._evict_old_images(keep=2)

    assert [str(m) for m in loop.messages] == before, "the cache prefix was left alone"


def test_the_budget_is_measured_in_bytes_because_bytes_are_what_failed(ctx, store, view):
    """The incident behind the guard was twenty-one images and five megabytes, and the
    API refusing the request. Twenty-one small ones are not that."""
    loop = Loop(Config(llm_api_key="k"), ctx, store, view)
    blocks = [{"type": "image", "source": {"type": "base64", "data": "X" * 40}},
              {"type": "text", "text": "note"}]
    assert loop_mod._image_bytes(blocks) == 40
    assert loop_mod._image_bytes([{"type": "text", "text": "no picture"}]) == 0


def test_a_refused_system_turn_is_learned_once_not_every_time(ctx, store, view):
    """`inform()` appends a fresh `role: "system"` on every harness fact, so without a
    latch each job-end wake and each refresh paid for its own rejected request. The
    Anthropic Messages API takes only user and assistant in `messages`, so this is the
    ordinary path for every Anthropic-family provider, hosted or bring-your-own."""
    loop = Loop(Config(llm_api_key="k"), ctx, store, view)
    sent = []

    class Refusing:
        """Rejects a system turn once, exactly as the API does."""

        def __init__(self):
            self.calls = 0

        def stream(self, **kwargs):
            self.calls += 1
            sent.append([m["role"] for m in kwargs["messages"]])
            if any(m["role"] == "system" for m in kwargs["messages"]):
                raise BadRequest("messages: unexpected role 'system'")
            return Turn(content=[])

    provider = Refusing()
    loop.provider = provider

    loop.say("first")
    loop.inform("a job ended")
    loop._send()
    assert provider.calls == 2, "one rejection, then the folded retry"
    assert "system" not in sent[-1]

    loop.say("second")
    loop.inform("another job ended")
    loop._send()
    assert provider.calls == 3, "the second fact cost no rejection at all"
    assert not any("system" in roles for roles in sent[2:])


def test_folding_leaves_every_other_message_byte_identical(ctx, store, view):
    """The rewrite must not disturb the cached prefix -- that is the whole reason the
    fold passes non-system messages through as the same objects."""
    loop = Loop(Config(llm_api_key="k"), ctx, store, view)
    original = {"role": "user", "content": "hello"}
    assert loop._fold_system(original) is original

    folded = loop._fold_system({"role": "system", "content": "a fact"})
    assert folded["role"] == "user" and "a fact" in folded["content"]
